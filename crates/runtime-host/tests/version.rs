use std::process::Command;

#[test]
fn both_programs_report_the_package_version_without_loading_a_bundle() {
    for (program, binary) in [
        ("chopinfer", env!("CARGO_BIN_EXE_chopinfer")),
        ("chopinfer-host", env!("CARGO_BIN_EXE_chopinfer-host")),
    ] {
        for flag in ["--version", "-V"] {
            let output = Command::new(binary).arg(flag).output().unwrap();
            assert!(output.status.success());
            assert!(output.stderr.is_empty());
            let revision = env!("CHOPINFER_BUILD_REVISION");
            let suffix = if revision.is_empty() {
                String::new()
            } else {
                format!(" ({revision})")
            };
            assert_eq!(
                String::from_utf8(output.stdout).unwrap(),
                format!("{program} {}{suffix}\n", env!("CARGO_PKG_VERSION"))
            );
        }
    }
}
