//! Shared dependency probes authenticated against their kernel-lab lock.
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernels_sm120::probes::{COPY, CUBIN, LIVE};
use serde_json::Value;
use std::{fs, path::Path, path::PathBuf, sync::OnceLock};

const ARTIFACTS: &str = "CHOPINFER_DEEPSEEK41_DEPENDENCY_PROBES";
const SYMBOLS: &[&str] = &[LIVE, COPY];

fn lock() -> Value {
    let records: Vec<Value> = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../crates/kernel-lab/kernels.lock.d/dependency_probes.lock"
    )))
    .unwrap();
    assert_eq!(records.len(), 1);
    records.into_iter().next().unwrap()
}

fn authenticate(bytes: &[u8], record: &Value) -> Result<(), String> {
    if record["name"] != CUBIN || record["cubin_sha256"] != kernel_build::digest(bytes) {
        return Err("dependency probe bytes differ from the shared lock".into());
    }
    let entries = record["entries"]
        .as_array()
        .ok_or("missing dependency probe entries")?;
    if entries.len() != SYMBOLS.len()
        || SYMBOLS
            .iter()
            .any(|symbol| !entries.iter().any(|e| e["symbol"] == *symbol))
    {
        return Err("dependency probe exports differ from the required inventory".into());
    }
    Ok(())
}

fn read(root: &Path) -> Result<Vec<u8>, String> {
    let path = root.join(format!("{CUBIN}.cubin"));
    if !fs::symlink_metadata(&path)
        .map_err(|e| e.to_string())?
        .file_type()
        .is_file()
    {
        return Err("dependency probe must be a regular file".into());
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    authenticate(&bytes, &lock())?;
    Ok(bytes)
}

fn build(root: &Path) {
    let toolkit = Toolchain::detect().expect("CUDA toolkit required for dependency probes");
    assert!(!toolkit.skipped);
    let locks =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../crates/kernel-lab/kernels.lock.d");
    let lock = LockDir::new(locks).family("dependency_probes", [CUBIN]);
    let built = kernel_build::build_in_families(
        &toolkit,
        &[kernels_sm120::probes::cubin_spec()],
        root,
        &lock,
        BuildOptions {
            lock_update: false,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )
    .expect("locked dependency probe build");
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].name, CUBIN);
    fs::write(
        root.join(format!("{CUBIN}.cubin")),
        fs::read(&built[0].path).unwrap(),
    )
    .unwrap();
}

pub fn get() -> (&'static [u8], &'static [&'static str]) {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    let bytes = BYTES.get_or_init(|| {
        let root = if let Some(root) = std::env::var_os(ARTIFACTS) {
            PathBuf::from(root)
        } else {
            assert!(
                kernel_harness::selected_ordinals().unwrap().is_none()
                    && !kernel_harness::require_all().unwrap(),
                "GPU runs require prebuilt dependency probes in {ARTIFACTS}"
            );
            let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
                "deepseek41-dependency-probes-{}",
                std::process::id()
            ));
            build(&root);
            root
        };
        read(&root).expect("unchanged dependency probe bytes and exports")
    });
    (bytes, SYMBOLS)
}

#[test]
fn dependency_probe_identity_rejects_changed_bytes_and_missing_exports() {
    let (bytes, symbols) = get();
    assert_eq!(symbols, SYMBOLS);
    let mut changed = bytes.to_vec();
    changed[0] ^= 1;
    assert!(
        authenticate(&changed, &lock())
            .unwrap_err()
            .contains("bytes differ")
    );
    for missing in SYMBOLS {
        let mut record = lock();
        record["entries"]
            .as_array_mut()
            .unwrap()
            .retain(|e| e["symbol"] != *missing);
        assert!(
            authenticate(bytes, &record)
                .unwrap_err()
                .contains("exports differ")
        );
    }
}
