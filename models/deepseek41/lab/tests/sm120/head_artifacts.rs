//! Prepare head probes during CPU tests, then read frozen cubins in GPU jobs.
//! Synthetic records and probe locks stay under the external target.
use super::head_probes;
use bundle_deepseek41::sm120::f32_head as registration;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

#[allow(dead_code)]
#[path = "../../../../../crates/kernel-lab/families/f32_projection.rs"]
mod f32_projection;
#[allow(dead_code)]
#[path = "../../../../../crates/kernel-lab/families/f32_projection_mutants.rs"]
mod f32_projection_mutants;

fn build(out: &Path) {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("../../..").canonicalize().unwrap();
    let toolkit = Toolchain::detect().expect("CUDA toolkit required for head probes");
    assert!(!toolkit.skipped);
    let options = BuildOptions {
        lock_update: true,
        emit_cargo_directives: false,
        profile_line_info: false,
    };
    let base = fs::read_to_string(manifest.join("../bundle/kernels.lock.d/f32_head.lock")).unwrap();
    let head_out = out.join("head");
    fs::create_dir_all(&head_out).unwrap();
    let lock = head_out.join("f32_head.lock");
    fs::write(&lock, &base).unwrap();
    let provenance =
        fs::read(manifest.join("../bundle/kernels.lock.d/f32_head.provenance.json")).unwrap();
    fs::write(head_out.join("f32_head.provenance.json"), &provenance).unwrap();
    let specs = registration::cubins(&[
        model_deepseek41_contract::geometry::FLASH,
        head_probes::synthetic(),
    ])
    .unwrap()
    .into_iter()
    .map(|(_, spec)| spec)
    .collect::<Vec<_>>();
    let names = specs.iter().map(|s| s.name.clone()).collect::<Vec<_>>();
    let family = LockDir::new(&head_out).family(registration::FAMILY, &names);
    let mut built =
        kernel_build::build_in_families(&toolkit, &specs, &head_out, &family, options, None)
            .unwrap();
    for (suffix, base) in [
        ("lock", base.as_bytes()),
        ("provenance.json", provenance.as_slice()),
    ] {
        let extended = fs::read(head_out.join(format!("f32_head.{suffix}"))).unwrap();
        assert_extension(base, &extended);
        fs::write(out.join(format!("head-base.{suffix}")), base).unwrap();
    }

    let probe_out = out.join("controls");
    fs::create_dir_all(&probe_out).unwrap();
    let mut entries = head_probes::tails();
    entries.push(head_probes::control());
    let specs =
        [kernels_sm120::f32_projection::cubin_spec("head_tail_controls", &entries).unwrap()];
    built.extend(
        kernel_build::build(
            &toolkit,
            &specs,
            &probe_out,
            &probe_out.join("controls.lock"),
            options,
        )
        .unwrap(),
    );

    let mutant_out = out.join("mutants");
    fs::create_dir_all(&mutant_out).unwrap();
    let mut specs = f32_projection_mutants::cubins();
    for spec in &mut specs {
        spec.sources_root = root.clone();
    }
    let names = specs.iter().map(|s| s.name.clone()).collect::<Vec<_>>();
    let family = LockDir::new(root.join("crates/kernel-lab/kernels.lock.d"))
        .family(f32_projection_mutants::FAMILY, &names);
    built.extend(
        kernel_build::build_in_families(
            &toolkit,
            &specs,
            &mutant_out,
            &family,
            BuildOptions {
                lock_update: false,
                ..options
            },
            None,
        )
        .unwrap(),
    );
    let index: Index = built
        .into_iter()
        .map(|cubin| {
            let bytes = fs::read(&cubin.path).unwrap();
            let symbols = cubin
                .entries
                .into_iter()
                .map(|entry| entry.symbol)
                .collect();
            (
                cubin.name,
                cubin
                    .path
                    .strip_prefix(out)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_owned(),
                kernel_build::digest(&bytes),
                symbols,
            )
        })
        .collect();
    fs::write(
        out.join("index.json"),
        serde_json::to_vec_pretty(&index).unwrap(),
    )
    .unwrap();
}

fn assert_extension(base: &[u8], extended: &[u8]) {
    let prefix = std::str::from_utf8(base)
        .unwrap()
        .trim_end()
        .strip_suffix(']')
        .unwrap()
        .trim_end();
    assert!(
        extended.starts_with(prefix.as_bytes()),
        "base head record bytes changed"
    );
    let records: Vec<serde_json::Value> = serde_json::from_slice(extended).unwrap();
    let original: Vec<serde_json::Value> = serde_json::from_slice(base).unwrap();
    assert_eq!(&records[..original.len()], original.as_slice());
    assert_eq!(records.len(), original.len() + 1);
    assert_eq!(
        records.last().unwrap()["name"],
        registration::cubin_name(&head_probes::synthetic())
    );
}

// Name, relative file, digest, entry symbols. GPU jobs authenticate the whole
// snapshot; the reader also checks each cubin and its declared inventory.
type Index = Vec<(String, String, String, Vec<String>)>;
type Artifacts = Vec<(String, Vec<u8>, Vec<String>)>;
const ARTIFACTS: &str = "CHOPINFER_DEEPSEEK41_HEAD_ARTIFACTS";

fn read(root: &Path) -> Result<Artifacts, String> {
    let index: Index =
        serde_json::from_slice(&fs::read(root.join("index.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let mut expected = registration::cubins(&[
        model_deepseek41_contract::geometry::FLASH,
        head_probes::synthetic(),
    ])?
    .into_iter()
    .map(|(_, spec)| spec)
    .collect::<Vec<_>>();
    let mut entries = head_probes::tails();
    entries.push(head_probes::control());
    expected.push(
        kernels_sm120::f32_projection::cubin_spec("head_tail_controls", &entries)
            .map_err(|e| e.to_string())?,
    );
    expected.extend(f32_projection_mutants::cubins());
    let mut cubins = Vec::new();
    for (name, file, digest, symbols) in index {
        if Path::new(&file)
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
        {
            return Err("head artifact path must stay in the snapshot".into());
        }
        let position = expected
            .iter()
            .position(|spec| spec.name == name)
            .ok_or_else(|| format!("unexpected or duplicated head cubin {name}"))?;
        let spec = expected.remove(position);
        if symbols
            != spec
                .entries
                .iter()
                .map(|entry| entry.symbol.clone())
                .collect::<Vec<_>>()
        {
            return Err(format!("head entry inventory differs for {name}"));
        }
        let bytes = fs::read(root.join(file)).map_err(|e| e.to_string())?;
        if kernel_build::digest(&bytes) != digest {
            return Err(format!("changed head cubin {name}"));
        }
        cubins.push((name, bytes, symbols));
    }
    if !expected.is_empty() {
        return Err("missing head cubins".into());
    }
    Ok(cubins)
}

pub fn cubins() -> &'static Artifacts {
    static CUBINS: OnceLock<Artifacts> = OnceLock::new();
    CUBINS.get_or_init(|| {
        let root = if let Some(root) = env::var_os(ARTIFACTS) {
            PathBuf::from(root)
        } else {
            assert!(
                kernel_harness::selected_ordinals().unwrap().is_none()
                    && !kernel_harness::require_all().unwrap(),
                "GPU runs require prebuilt head artifacts in {ARTIFACTS}"
            );
            let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("deepseek41-head");
            build(&root);
            root
        };
        read(&root).expect("complete, unchanged head artifacts")
    })
}

#[test]
fn frozen_artifacts_reject_missing_changed_or_unexpected_cubins() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("head-artifact-reader");
    fs::create_dir_all(&root).unwrap();
    let index: Index = cubins()
        .iter()
        .map(|(name, bytes, symbols)| {
            let file = format!("{name}.cubin");
            fs::write(root.join(&file), bytes).unwrap();
            (
                name.clone(),
                file,
                kernel_build::digest(bytes),
                symbols.clone(),
            )
        })
        .collect();
    let check = |index: &Index| {
        fs::write(root.join("index.json"), serde_json::to_vec(index).unwrap()).unwrap();
        read(&root)
    };
    assert_eq!(check(&index).unwrap(), *cubins());
    let mut bad = index.clone();
    bad.pop();
    assert_eq!(check(&bad).unwrap_err(), "missing head cubins");
    let mut bad = index.clone();
    bad.push(index[0].clone());
    assert!(check(&bad).unwrap_err().contains("duplicated head cubin"));
    let mut bad = index.clone();
    bad[0].3.pop();
    assert!(check(&bad).unwrap_err().contains("entry inventory differs"));
    let mut bad = index.clone();
    bad[0].1 = "../outside.cubin".into();
    assert!(check(&bad).unwrap_err().contains("stay in the snapshot"));
    let mut bad = index.clone();
    bad[0].1 = "missing.cubin".into();
    assert!(check(&bad).is_err());
    fs::write(root.join(&index[0].1), b"changed").unwrap();
    assert!(check(&index).unwrap_err().contains("changed head cubin"));
}
