//! Installed-helper coverage for reuse before model-specific emission.
use runtime_bundle::preparation::{DirectPackage, Response, read_record};
use std::path::Path;
use std::process::Command;

#[allow(dead_code)]
mod common;
#[path = "../../../../crates/model-compiler-sdk/test-support/native.rs"]
mod native;
mod support;

fn prepare(helper: &Path, checkpoint: &Path, cache: &Path) -> DirectPackage {
    let output = Command::new(helper)
        .args(["prepare", "--direct", "--model"])
        .arg(checkpoint)
        .args(["--tensor-parallel-size", "4", "--cache-dir"])
        .arg(cache)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    match read_record(&mut output.stdout.as_slice()).unwrap() {
        Response::PreparedDirect(package) => *package,
        response => panic!("unexpected helper response: {response:?}"),
    }
}

#[test]
fn the_installed_helper_reuses_before_emission_and_invalidates_module_changes() {
    let Some(tokenizer) = common::checkpoint() else {
        return;
    };
    let root = support::sparse(&model_deepseek41_contract::geometry::FLASH);
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::copy(tokenizer.join(name), root.join(name)).unwrap();
    }
    let module = native::compile_native_fixture(
        "deepseek41-reuse-module",
        "deepseek41_reuse_fixture",
        "libbundle_deepseek41.so",
        r#"#[unsafe(no_mangle)]
pub extern "C" fn chop_bundle_get_api_v1() -> *const core::ffi::c_void {
    core::ptr::null()
}"#,
    );
    let install = module.parent().unwrap();
    let helper = install.join("chopinfer-model-deepseek41");
    std::fs::copy(env!("CARGO_BIN_EXE_chopinfer-model-deepseek41"), &helper).unwrap();
    // The cache is outside the checkpoint, so cache records are not metadata inputs.
    let cache = install.join("cache");
    let first = prepare(&helper, &root, &cache);
    assert!(first.measurements.stages.contains_key("package_assembly"));
    assert_eq!(first.profile.name, "tp4-dspark");
    std::thread::scope(|threads| {
        for _ in 0..2 {
            threads.spawn(|| {
                let again = prepare(&helper, &root, &cache);
                assert_eq!(again.package_identity, first.package_identity);
                assert_eq!(again.profile, first.profile);
                assert_eq!(
                    again
                        .measurements
                        .stages
                        .keys()
                        .map(String::as_str)
                        .collect::<Vec<_>>(),
                    ["package_reuse"]
                );
            });
        }
    });
    let refused = Command::new(&helper)
        .args(["prepare", "--direct", "--model"])
        .arg(&root)
        .args(["--tensor-parallel-size", "4", "--cache-dir"])
        .arg(&cache)
        .args(["--activation-policy", "bf16"])
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("activation-policy"));
    let inputs =
        model_compiler_sdk::preparation::direct::native_module_inputs(&module, &[]).unwrap();
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&module)
        .unwrap()
        .write_all(b"changed fixture identity")
        .unwrap();
    let request = runtime_bundle::preparation::PrepareDirectRequest {
        path: root.clone(),
        cache_root: cache.clone(),
        tensor_parallel_size: 4,
        activation_policy: None,
        modalities: Default::default(),
    };
    let output = install.join("refused-module");
    let error = model_deepseek41::preparation::emit_with_inputs(
        &root,
        &request,
        &module,
        &output,
        Some(&inputs),
    )
    .err()
    .expect("replacement after input capture must fail before publication");
    assert!(error.to_string().contains("bytes"), "{error}");
    assert!(!output.exists());
    let changed = prepare(&helper, &root, &cache);
    assert_ne!(changed.package_identity, first.package_identity);
    assert!(changed.measurements.stages.contains_key("package_assembly"));
    let again = prepare(&helper, &root, &cache);
    assert_eq!(again.package_identity, changed.package_identity);
    assert!(again.measurements.stages.contains_key("package_reuse"));
    std::fs::remove_dir_all(&root).unwrap();
    std::fs::remove_dir_all(install).unwrap();
}
