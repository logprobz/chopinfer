//! A package invocation needs no installation configuration.
#[test]
fn a_package_invocation_reaches_validation_without_an_install_root() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_qwen38-package"))
        .env_remove("CHOPINFER_DATA")
        .args([
            "--checkpoint",
            "/nonexistent-ms-b7-checkpoint",
            "--module",
            "/nonexistent-ms-b7-module",
            "--output",
            "/nonexistent-ms-b7-output",
            "--drafter",
            "invalid-test-drafter",
        ])
        .output()
        .unwrap();
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(!error.contains("install-root"), "{error}");
    assert!(error.contains("drafter"), "{error}");
}

#[test]
fn activation_policy_cli_precedence_and_duplicates_fail_before_source_io() {
    let run = |args: &[&str]| {
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_qwen38-package"))
            .env("CHOPINFER_ACTIVATION_POLICY", "invalid-default")
            .args([
                "--checkpoint",
                "/nonexistent-precision-source",
                "--module",
                "/nonexistent-precision-module",
                "--output",
                "/nonexistent-precision-output",
            ])
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success());
        String::from_utf8(result.stderr).unwrap()
    };
    assert!(run(&[]).contains("invalid-default"));
    assert!(!run(&["--activation-policy", "bf16"]).contains("invalid-default"));
    let duplicate = run(&["--activation-policy", "auto", "--activation-policy", "bf16"]);
    assert!(duplicate.contains("activation-policy"), "{duplicate}");
    assert!(
        !duplicate.contains("nonexistent-precision-source"),
        "{duplicate}"
    );
}

/// The package is computed from the checkpoint alone: the rank-image
/// options are refused by name before any source is read.
#[test]
fn rank_image_options_are_refused() {
    for (flag, value) in [
        ("--rank-dir", "/nonexistent-ranks"),
        ("--images", "none"),
        ("--ngram-placement", "host"),
    ] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_qwen38-package"))
            .args([
                "--checkpoint",
                "/nonexistent-checkpoint",
                flag,
                value,
                "--module",
                "/nonexistent-module",
                "--output",
                "/nonexistent-output",
            ])
            .output()
            .unwrap();
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(!output.status.success());
        assert!(error.contains(&format!("{flag} is gone")), "{error}");
    }
}

#[test]
fn helper_refuses_tensor_parallel_size_eight_before_source_io() {
    let model = std::env::temp_dir().join(format!("qwen-helper-tp8-{}", std::process::id()));
    std::fs::create_dir(&model).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_chopinfer-model-qwen38"))
        .args([
            "profile",
            "--model",
            model.to_str().unwrap(),
            "--tensor-parallel-size",
            "8",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("serves 2 or 4 ranks, not 8"), "{error}");
    assert!(!error.contains("config.json"), "{error}");
    std::fs::remove_dir(model).unwrap();
}
