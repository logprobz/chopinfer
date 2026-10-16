//! Rejections from the installed family helper before checkpoint loading.
use std::process::Command;

#[test]
fn helper_refuses_tensor_parallel_size_eight() {
    let output = Command::new(env!("CARGO_BIN_EXE_chopinfer-model-mimo26"))
        .args(["profile", "--model"])
        .arg(std::env::temp_dir())
        .args(["--tensor-parallel-size", "8"])
        .output()
        .expect("run family helper");
    assert!(!output.status.success());
    let reply: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(reply["result"], "error");
    assert!(
        reply["message"]
            .as_str()
            .unwrap()
            .contains("MiMo supports 2 or 4 ranks, not 8")
    );
}
