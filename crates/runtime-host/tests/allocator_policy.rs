#![cfg(all(target_os = "linux", target_env = "gnu"))]

use std::process::{Command, Output};

const VARIABLE: &str = "CHOPINFER_MALLOC_ARENA_MAX";
const PROGRAMS: [&str; 2] = [
    env!("CARGO_BIN_EXE_chopinfer"),
    env!("CARGO_BIN_EXE_chopinfer-host"),
];

fn invoke(binary: &str, arguments: &[&str], setting: Option<&str>) -> Output {
    let mut command = Command::new(binary);
    command.args(arguments).env_remove(VARIABLE);
    if let Some(setting) = setting {
        command.env(VARIABLE, setting);
    }
    command.output().unwrap()
}

#[test]
fn both_programs_set_the_default_before_parsing_or_loading_serve() {
    for binary in PROGRAMS {
        let output = invoke(binary, &["serve", "--unknown-allocator-test-option"], None);
        assert!(!output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.starts_with("host allocator: arena_max=8\n"),
            "{stderr}"
        );
        assert!(stderr.contains("--unknown-allocator-test-option"));
    }
}

#[test]
fn serving_accepts_an_explicit_limit_and_the_glibc_startup_default() {
    for binary in PROGRAMS {
        for (setting, reported) in [("16", "16"), ("0", "default")] {
            let output = invoke(binary, &["serve", "--help"], Some(setting));
            assert!(output.status.success());
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert_eq!(stderr, format!("host allocator: arena_max={reported}\n"));
        }
    }
}

#[test]
fn malformed_limits_refuse_before_serving_starts() {
    for binary in PROGRAMS {
        for setting in ["", "-1", "many", "2147483648"] {
            let output = invoke(binary, &["serve", "--help"], Some(setting));
            assert!(!output.status.success(), "accepted {setting:?}");
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert!(stderr.contains(VARIABLE), "{stderr}");
            assert!(
                stderr.contains("0 or a positive 32-bit integer"),
                "{stderr}"
            );
        }
    }
}

#[test]
fn non_serving_commands_keep_their_allocator_policy() {
    for binary in PROGRAMS {
        let output = invoke(binary, &["--version"], Some("invalid"));
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
    }
}
