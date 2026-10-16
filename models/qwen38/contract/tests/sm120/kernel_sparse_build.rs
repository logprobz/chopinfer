//! CPU-only checks and ptxas audits of Qwen3.8's QSA catalog. No cubin is loaded on a GPU.

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use kernel_build::{BuildOptions, Toolchain, build};
use kernels_sm120::attention::sparse::{Kind, catalog, cubin_specs};
use model_qwen38_contract::sm120::catalogs;

#[test]
fn qsa_entries_compile_without_spills() {
    // TP4's cubin, then TP2's attention entries (12 grouped
    // heads over one KV head).
    let specs = cubin_specs(catalogs(), None);
    assert_eq!(specs.len(), model_qwen38_contract::sm120::PROFILES.len());
    assert_eq!(specs[0].name, "sparse_attention_qwen38_sm120");
    for profile in model_qwen38_contract::sm120::PROFILES {
        let name = format!("sparse_attention_{}_sm120", profile.tag());
        let spec = specs.iter().find(|spec| spec.name == name).unwrap();
        let entries = catalog(catalogs())
            .into_iter()
            .filter(|entry| entry.profile == profile)
            .collect::<Vec<_>>();
        let attention = entries
            .iter()
            .filter(|entry| entry.kind == Kind::Attention)
            .count();
        assert_eq!(attention, 6);
        assert_eq!(spec.entries.len(), entries.len() + attention);
    }
    let toolchain = Toolchain::detect().expect("CUDA 13.2 compiler");
    assert!(!toolchain.skipped, "ptxas coverage is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let scratch = std::env::temp_dir().join(format!("qsa-build-{}-{stamp}", std::process::id()));
    fs::create_dir(&scratch).unwrap();
    let built = build(
        &toolchain,
        &specs,
        &scratch.join("out"),
        &scratch.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    for entry in built.iter().flat_map(|cubin| &cubin.entries) {
        assert_eq!(entry.local_bytes, 0, "{} spilled", entry.symbol);
        assert_eq!(entry.stack_bytes, 0, "{} has stack", entry.symbol);
    }
    // Scoring and attention run BF16 MMA over ldmatrix fragments.
    for cubin in &built {
        let dump = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
            .arg("-sass")
            .arg(&cubin.path)
            .output()
            .unwrap();
        let sass = String::from_utf8(dump.stdout).unwrap();
        for section in sass.split("Function : ").skip(1) {
            let (symbol, body) = section.split_once('\n').unwrap();
            let symbol = symbol.trim();
            if catalog(catalogs()).iter().any(|entry| {
                entry.symbol() == symbol
                    && matches!(entry.kind, Kind::Scores { .. } | Kind::Attention)
            }) {
                for instruction in ["HMMA.16816.F32.BF16", "LDSM.16.M88.4", "LDGSTS"] {
                    assert!(body.contains(instruction), "{symbol} lacks {instruction}");
                }
            }
        }
    }
    fs::remove_dir_all(scratch).unwrap();
}
