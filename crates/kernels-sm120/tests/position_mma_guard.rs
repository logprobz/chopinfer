//! Exercise the same portable predicate compiled into the CUDA scorer.
use std::{path::PathBuf, process::Command};
#[test]
fn exact_accumulation_domain_and_reordering_negative_control() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("index-guard-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let binary = dir.join("bound");
    let compiled = Command::new("c++")
        .args(["-O2", "-std=c++17", "-ffp-contract=off", "-I"])
        .arg(root.join("cuda"))
        .arg(root.join("tests/position_mma_guard.cpp"))
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let ran = Command::new(&binary).output().unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    assert!(
        ran.status.success(),
        "{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&ran.stdout),
        "GUARD_BOUND_COMPLETE accepted=20000 negative_control_rejected=1\n"
    );
}
