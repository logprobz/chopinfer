//! Optional export of the existing model-owned journal. No new host service.
use super::*;
use serde_json::json;
use std::io::Write;
use std::path::{Component, Path};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(super) fn export(
    root: &Path,
    lines: &[String],
    scenario: &str,
    expected: (usize, u64),
    limit: usize,
) -> Result<()> {
    export_with_mode(root, lines, scenario, expected, limit, false)
}

pub(super) fn export_geometry(
    root: &Path,
    lines: &[String],
    scenario: &str,
    expected: (usize, u64),
    limit: usize,
) -> Result<()> {
    export_with_mode(root, lines, scenario, expected, limit, true)
}

fn export_with_mode(
    root: &Path,
    lines: &[String],
    scenario: &str,
    expected: (usize, u64),
    limit: usize,
    geometry: bool,
) -> Result<()> {
    if !root.is_absolute()
        || root.components().any(|c| matches!(c, Component::ParentDir))
        || root.ancestors().any(|p| {
            p.join(".git").exists()
                || p.symlink_metadata()
                    .is_ok_and(|m| m.file_type().is_symlink())
        })
    {
        return Err("journal output must be external and must not traverse symlinks".into());
    }
    if limit > 64 << 20 {
        return Err("journal limit exceeds 64 MiB".into());
    }
    std::fs::create_dir_all(root.parent().ok_or("journal root needs a parent")?)?;
    std::fs::create_dir(root)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join("journal.txt"))?;
    let mut bytes = 0;
    let mut events = 0;
    let mut boundaries = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if bytes + line.len() + 1 > limit {
            break;
        }
        writeln!(file, "{line}")?;
        if let Some(step) = line.strip_prefix("step ") {
            boundaries.push(json!({"event":index,"step":step}));
        }
        bytes += line.len() + 1;
        events += 1;
    }
    file.sync_all()?;
    let hash = runtime_weights::sha256_reader(&mut std::fs::File::open(root.join("journal.txt"))?)?;
    let binary = std::env::current_exe()?;
    let binary_hash = runtime_weights::sha256_reader(&mut std::fs::File::open(&binary)?)?;
    let selector = if geometry {
        "fake_host_tests::every_geometry_topology_runs_the_fake_host_journal_and_lifecycle"
    } else if scenario.starts_with("fp8") {
        "fake_host_tests::qwen38_fp8_kv_journal_differs_only_in_kv_launches"
    } else if scenario.starts_with("bf16") {
        "fake_host_tests::qwen38_launch_journal_is_unchanged"
    } else {
        "fake_host_tests::tp2_launch_journal_is_unchanged"
    };
    let variable = if geometry {
        "CHOPINFER_QWEN38_GEOMETRY_JOURNAL_DIR"
    } else {
        "CHOPINFER_QWEN38_JOURNAL_DIR"
    };
    let reproduce = vec![
        "env".to_owned(),
        format!(
            "{variable}={}",
            root.with_file_name(format!("{scenario}-reproduced"))
                .display()
        ),
        binary.to_string_lossy().into_owned(),
        "--exact".into(),
        selector.into(),
        "--nocapture".into(),
    ];
    let value = json!({"format":"chopinfer.diagnostic.v1","mode":"journal",
        "identity":{"scenario":scenario,"served_capacity":SERVED,"kv_pages":KV_PAGES},
        "producer":{"binary_sha256":binary_hash,"kernel_lock_sha256":runtime_weights::sha256_bytes(include_bytes!("../../kernels.lock"))},
        "completion":if events == lines.len() { "full" } else { "truncated" },
        "events":events,"expected_events":lines.len(),"boundaries":boundaries,
        "attachment_metadata":{"schema":runtime_bundle_sdk::diagnostic::SCHEMA,
            "ordered_events":boundaries,"evidence":"structural-only","tensor_observations":[]},
        "golden":{"events":expected.0,"fnv1a":format!("{:016x}",expected.1)},
        "reproduction":reproduce,
        "artifacts":{"journal.txt":{"bytes":bytes,"sha256":hash}}});
    let mut terminal = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join("diagnostic.json"))?;
    serde_json::to_writer(&mut terminal, &value)?;
    terminal.sync_all()?;
    Ok(())
}

#[test]
fn journal_export_reports_the_first_changed_argument() {
    with_default_journal_settings(|| {
        let expected = EXPECTED;
        let lines = journal_steps(&Rc::new(FakeHost::default()));
        // Validate the original frozen journal without triggering the optional exporter.
        assert_eq!(journal_digest(&lines), expected);
        let root =
            std::env::temp_dir().join(format!("qwen-journal-diagnostic-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        for name in ["baseline", "matching"] {
            export(&root.join(name), &lines, "bf16", expected, 64 << 20).unwrap();
        }
        let mut changed = lines.clone();
        let begin = lines.iter().position(|s| s == "step prefill").unwrap();
        let index = (begin..lines.len())
            .find(|&i| lines[i].starts_with("launch ") && lines[i].contains("U32("))
            .unwrap();
        let start = changed[index].find("U32(").unwrap() + 4;
        let end = start + changed[index][start..].find(')').unwrap();
        let argument: u32 = changed[index][start..end].parse().unwrap();
        changed[index].replace_range(start..end, &(argument + 1).to_string());
        export(&root.join("mismatch"), &changed, "bf16", expected, 64 << 20).unwrap();
        export(&root.join("incomplete"), &lines, "bf16", expected, 128).unwrap();
        export(
            &root.join("incompatible"),
            &lines,
            "different-scenario",
            expected,
            64 << 20,
        )
        .unwrap();
        let tool = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tools/dev/diagnostics.py");
        for (name, code) in [
            ("matching", 0),
            ("mismatch", 1),
            ("incompatible", 2),
            ("incomplete", 3),
        ] {
            let output = std::process::Command::new("python3")
                .arg("-B")
                .arg(&tool)
                .arg("compare")
                .arg("--reference")
                .arg(root.join("baseline"))
                .arg("--candidate")
                .arg(root.join(name))
                .args(["--mode", "journal"])
                .output()
                .unwrap();
            assert_eq!(
                output.status.code(),
                Some(code),
                "{} {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            if name == "mismatch" {
                assert_eq!(report["first"]["event"], index);
                assert_eq!(report["first"]["step"], "prefill");
                assert_eq!(report["first"]["operation"], "launch");
                assert_eq!(report["first"]["expected"], lines[index]);
                assert_eq!(report["first"]["observed"], changed[index]);
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    });
}
