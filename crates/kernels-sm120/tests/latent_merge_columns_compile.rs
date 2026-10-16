//! Builds the column merges of every splitting latent attention entry (16
//! heads, latent 512, 64-token pages, 2051-entry key lists) and each lab
//! mutant, and checks the audits and the SASS: no spills or local memory,
//! and the PDL release and wait.
use kernel_build::{BuildOptions, Toolchain, build};
use kernels_sm120::latent::merge_columns::{self, lab_cubin_specs};
use kernels_sm120::latent::{Geometry, entries};
use kernels_sm120::{Profile, ProfileNames};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "merge-columns-compile",
    tag: "merge_columns_compile",
});

#[test]
fn merge_columns_entries_compile_without_spills() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is available");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into())).join(
        format!("merge-columns-cubin-{}-{stamp}", std::process::id()),
    );
    fs::create_dir_all(&root).unwrap();
    let geometry = Geometry {
        heads: 16,
        latent: 512,
        page_size: 64,
        max_keys: 2051,
        softmax_scale: 1.0 / 16.0,
        nope: 256,
        v_dim: 256,
    };
    let merges = merge_columns::entries(&entries(PROFILE, geometry));
    let specs = lab_cubin_specs(PROFILE, &merges).unwrap();
    let cubins = build(
        &toolchain,
        &specs,
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("column merge cubins build and audit");
    assert_eq!(cubins.len(), 1 + merge_columns::MUTANTS.len());
    for cubin in &cubins {
        // Decode and verify, BF16 and FP8 pages, normalized and partial.
        assert_eq!(cubin.entries.len(), 8);
        for entry in &cubin.entries {
            eprintln!(
                "{}: {} registers, {} static shared bytes",
                entry.symbol, entry.registers, entry.shared_static_bytes
            );
            assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
        }
    }
    let sass = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&cubins[0].path)
        .output()
        .expect("inspect column merge SASS");
    assert!(sass.status.success());
    let text = String::from_utf8_lossy(&sass.stdout);
    for opcode in ["ACQBULK", "PREEXIT", "BAR.SYNC", "FFMA"] {
        assert!(text.contains(opcode), "the SASS has no {opcode}");
    }
    for opcode in ["LDL", "STL"] {
        assert!(
            !text.contains(opcode),
            "the SASS uses local memory ({opcode})"
        );
    }
    fs::remove_dir_all(root).unwrap();
}
