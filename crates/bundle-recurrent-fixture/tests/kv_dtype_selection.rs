//! The KV layout `--kv-cache-dtype` resolves to is the one the instance is
//! created with, on both physical-KV paths.
//!
//! The emitted package is relaid three ways: BF16 and FP8 declared with BF16
//! preferred, the same with FP8 preferred, and FP8 alone without the legacy
//! FP8 capability. Each is loaded in a child process of this test, because a
//! process loads one bundle, and created for `auto`, `bf16` and `fp8`: with
//! the serving pool, which the module acquires by the declared layout ID and
//! the host sizes by that layout's pages, and without it, where the module
//! reports the layout it was given. A prefix snapshot fills whole pages of
//! the selected layout.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{env, fs};

use bundle_recurrent_fixture::{
    RECURRENT_ARCHITECTURE_V1, RECURRENT_FIXTURE_NAME_V1, RECURRENT_KV_FP8_LAYOUT_ID_V1,
    RECURRENT_KV_FP8_PAGE_BYTES_V1, RECURRENT_KV_LAYOUT_ID_V1, RECURRENT_KV_PAGE_BYTES_V1,
    RECURRENT_STATE_BYTES_V1, RECURRENT_TOPOLOGY_ID_V1, recurrent_kv_fp8_layout_v1,
    recurrent_kv_layout_v1,
};
use runtime_bundle::{
    BundleHostCompatibilityV1, BundleKvDtypeV1, BundleKvFormatV1, BundleKvLayoutV1,
    BundleManifestV1, BundleOpaqueId, MODEL_BUNDLE_V1_ABI_VERSION,
};
use runtime_host::fake::FakeHostDeviceV1;
use runtime_host::{
    BundleSelectionV1, KvCacheDtypeV1, ServingKvPolicyV1, load_selected_process_bundle_v1,
    resolve_kv_cache_dtype_v1,
};

const CHILD_PACKAGE: &str = "RECURRENT_KV_DTYPE_SELECTION_PACKAGE";
const CHILD_CHECKPOINT: &str = "RECURRENT_KV_DTYPE_SELECTION_CHECKPOINT";
const TEST_NAME: &str = "the_selected_kv_layout_reaches_creation_on_both_kv_paths";

fn host() -> BundleHostCompatibilityV1 {
    BundleHostCompatibilityV1::new(RECURRENT_ARCHITECTURE_V1, MODEL_BUNDLE_V1_ABI_VERSION)
        .expect("the test host descriptor is valid")
        .with_capabilities(
            [
                "chop.prompt-rows.v2",
                "chop.prefix-cache.state-slot.v1",
                "chop.prefix-cache.v1",
                "chop.serving-kv.v1",
                "chop.recurrent-state.v1",
                "chop.request-retirement.v1",
                "chop.speculative.v1",
                runtime_bundle::WEIGHT_PLAN_CAPABILITY_V1,
                runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1,
            ]
            .map(|id| BundleOpaqueId::new(id).unwrap()),
        )
}

/// One `auto`, `bf16` and `fp8` creation of the package the parent named,
/// reported one line each.
fn run_child(package: &Path, checkpoint: &Path) {
    runtime_host::direct::select_checkpoint_v1(checkpoint).expect("the synthetic checkpoint opens");
    let loaded = load_selected_process_bundle_v1(
        &BundleSelectionV1::Directory(package.to_path_buf()),
        &host(),
    )
    .expect("the relaid package loads");
    for (label, requested) in [
        ("auto", KvCacheDtypeV1::Auto),
        ("bf16", KvCacheDtypeV1::Bf16),
        ("fp8", KvCacheDtypeV1::Fp8),
    ] {
        let choice = match resolve_kv_cache_dtype_v1(loaded.manifest(), requested) {
            Ok(choice) => choice,
            Err(refusal) => {
                println!("\nkv-dtype:{label} refused {refusal}");
                continue;
            }
        };
        for pool in [true, false] {
            let policy = pool.then_some(ServingKvPolicyV1 {
                sampling_vocabulary: None,
                maximum_context_tokens: None,
                state_reserved_bytes: 0,
                utilization_millionths: 1_000_000,
                maximum_pages: None,
                ..Default::default()
            });
            let (backend, mut instance) = loaded
                .create_instance_with_serving_options(
                    FakeHostDeviceV1::new(),
                    RECURRENT_TOPOLOGY_ID_V1,
                    vec![0],
                    policy,
                    runtime_bundle_abi::BundleRawSpeculationV1::disabled(),
                    None,
                    Some(choice.dtype),
                )
                .expect("the module accepts either declared layout");
            let layout = backend.with_device(|device| {
                device
                    .logs()
                    .iter()
                    .map(|line| String::from_utf8_lossy(line).into_owned())
                    .find_map(|line| {
                        let (_, rest) = line.split_once("declared page layout '")?;
                        Some(rest.split_once('\'')?.0.to_owned())
                    })
                    .expect("the module names the layout it was given")
            });
            let pages = backend
                .serving_kv_pages()
                .map_or("none".to_owned(), |pages| pages.to_string());
            // A snapshot fills whole pages of the layout the pool holds.
            let snapshot = if pool {
                instance.describe_prefix_state().unwrap().to_string()
            } else {
                "none".to_owned()
            };
            println!(
                "\nkv-dtype:{label} {} pool={pool} layout={layout} snapshot={snapshot} \
                 pages={pages}",
                choice.dtype.as_str()
            );
            drop(instance);
        }
    }
}

#[test]
fn the_selected_kv_layout_reaches_creation_on_both_kv_paths() {
    if let (Some(package), Some(checkpoint)) =
        (env::var_os(CHILD_PACKAGE), env::var_os(CHILD_CHECKPOINT))
    {
        run_child(Path::new(&package), Path::new(&checkpoint));
        return;
    }
    let root = TestDirectory::new("recurrent-kv-dtype");
    let emitted = root.path().join("emitted");
    let checkpoint = root.path().join("checkpoint");
    let built = Command::new(env!("CARGO_BIN_EXE_build-recurrent-fixture"))
        .args(["--fixture", RECURRENT_FIXTURE_NAME_V1, "--output"])
        .arg(&emitted)
        .arg("--checkpoint")
        .arg(&checkpoint)
        .output()
        .expect("build-recurrent-fixture starts");
    assert!(built.status.success(), "{built:?}");
    let bf16 = recurrent_kv_layout_v1().unwrap();
    let fp8 = recurrent_kv_fp8_layout_v1().unwrap();
    let both = vec![
        BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, bf16.clone()),
        BundleKvFormatV1::new(BundleKvDtypeV1::Fp8E4m3, fp8.clone()),
    ];
    let only_fp8 = vec![BundleKvFormatV1::new(BundleKvDtypeV1::Fp8E4m3, fp8.clone())];
    let run = |name: &str, preferred: &BundleKvLayoutV1, formats: Vec<BundleKvFormatV1>| {
        let package = root.path().join(name);
        relay(&emitted, &package, &bf16, preferred, formats);
        let output = Command::new(env::current_exe().unwrap())
            .args(["--exact", TEST_NAME, "--nocapture", "--test-threads", "1"])
            .env(CHILD_PACKAGE, &package)
            .env(CHILD_CHECKPOINT, &checkpoint)
            .output()
            .expect("the child test starts");
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        assert!(output.status.success(), "{name}: {stdout}\n{output:?}");
        stdout
    };

    let bf16_id = format!("layout={RECURRENT_KV_LAYOUT_ID_V1} ");
    let fp8_id = format!("layout={RECURRENT_KV_FP8_LAYOUT_ID_V1} ");

    // BF16 preferred: auto and bf16 run BF16, fp8 runs the FP8 layout with
    // twice the pages, with and without the pool.
    let out = run("prefers-bf16", &bf16, both.clone());
    let line = |prefix: &str| line_of(&out, prefix);
    let pages = |prefix: &str| pages_of(&out, prefix);
    for prefix in [
        "auto bf16 pool=true",
        "auto bf16 pool=false",
        "bf16 bf16 pool=true",
    ] {
        assert!(line(prefix).contains(&bf16_id), "{}", line(prefix));
    }
    for prefix in ["fp8 fp8_e4m3 pool=true", "fp8 fp8_e4m3 pool=false"] {
        assert!(line(prefix).contains(&fp8_id), "{}", line(prefix));
    }
    assert!(line("fp8 fp8_e4m3 pool=false").ends_with("pages=none"));
    let bf16_pages = pages("bf16 bf16 pool=true");
    assert_eq!(pages("auto bf16 pool=true"), bf16_pages);
    assert!(bf16_pages > 0);
    let fp8_pages = pages("fp8 fp8_e4m3 pool=true");
    let snapshot = |bytes: u64| (RECURRENT_STATE_BYTES_V1 + 8).div_ceil(bytes);
    assert!(line("bf16 bf16 pool=true").contains(&format!(
        "snapshot={} ",
        snapshot(RECURRENT_KV_PAGE_BYTES_V1)
    )));
    assert!(line("fp8 fp8_e4m3 pool=true").contains(&format!(
        "snapshot={} ",
        snapshot(RECURRENT_KV_FP8_PAGE_BYTES_V1)
    )));
    assert!(
        fp8_pages / 2 == bf16_pages,
        "FP8 pages {fp8_pages} are half the bytes of BF16's {bf16_pages}"
    );

    // FP8 preferred: auto runs FP8; an explicit bf16 still runs BF16.
    let out = run("prefers-fp8", &fp8, both);
    let line = |prefix: &str| line_of(&out, prefix);
    let pages = |prefix: &str| pages_of(&out, prefix);
    assert!(line("auto fp8_e4m3 pool=true").contains(&fp8_id));
    assert!(line("bf16 bf16 pool=false").contains(&bf16_id));
    assert_eq!(pages("bf16 bf16 pool=true"), bf16_pages);
    assert_eq!(pages("auto fp8_e4m3 pool=true"), fp8_pages);

    // FP8 alone, without the legacy capability: auto and fp8 run it, bf16 is
    // refused before anything is created.
    let out = run("fp8-only", &fp8, only_fp8);
    let line = |prefix: &str| line_of(&out, prefix);
    assert!(line("auto fp8_e4m3 pool=true").contains(&fp8_id));
    assert!(line("fp8 fp8_e4m3 pool=false").contains(&fp8_id));
    assert!(
        line("bf16 refused").contains("declares no bf16 KV layout; it supports fp8"),
        "{}",
        line("bf16 refused")
    );
}

fn line_of(stdout: &str, prefix: &str) -> String {
    let marked = format!("kv-dtype:{prefix}");
    stdout
        .lines()
        .find_map(|line| {
            line.strip_prefix(&marked)
                .map(|rest| format!("{prefix}{rest}"))
        })
        .unwrap_or_else(|| panic!("no '{prefix}' line in {stdout}"))
}

fn pages_of(stdout: &str, prefix: &str) -> u64 {
    line_of(stdout, prefix)
        .rsplit_once("pages=")
        .unwrap()
        .1
        .parse()
        .unwrap()
}

/// Copies the emitted package with `preferred` as its `kv_layout` and
/// `formats` as its `kv_formats`; components and their digests are unchanged.
fn relay(
    emitted: &Path,
    package: &Path,
    declared: &BundleKvLayoutV1,
    preferred: &BundleKvLayoutV1,
    formats: Vec<BundleKvFormatV1>,
) {
    copy_tree(emitted, package);
    let path = package.join("manifest.json");
    let json = String::from_utf8(fs::read(&path).unwrap()).unwrap();
    let from = serde_json::to_string(declared).unwrap();
    assert_eq!(json.matches(&from).count(), 1, "{json}");
    let json = json.replace(&from, &serde_json::to_string(preferred).unwrap());
    let manifest = BundleManifestV1::from_canonical_json(json.as_bytes())
        .expect("the relaid layout is valid")
        .with_kv_formats(formats)
        .expect("the declared formats are valid");
    fs::write(&path, manifest.to_canonical_json().unwrap()).unwrap();
}

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let target = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target).unwrap();
        }
    }
}

/// A scratch directory this test owns and removes.
struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = env::temp_dir().join(format!(
            "{label}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
