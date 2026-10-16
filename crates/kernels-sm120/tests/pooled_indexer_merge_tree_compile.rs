//! Builds the tree merges of every selection entry (32 heads of 128, pools
//! of 4, top 2048 with the tail, 64-token pages) and each lab mutant, and
//! checks the audits and the SASS: no spills or local memory, warp
//! shuffles, and the PDL release and wait.
use kernel_build::{BuildOptions, Toolchain, build};
use kernel_core::contract::DType;
use kernels_sm120::pooled_indexer::merge_tree::{self, lab_cubin_specs};
use kernels_sm120::pooled_indexer::{Geometry, entries};
use kernels_sm120::{Profile, ProfileNames};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "merge-tree-compile",
    tag: "merge_tree_compile",
});

#[test]
fn merge_tree_entries_compile_without_spills() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is available");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into()))
        .join(format!("merge-tree-cubin-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let geometry = Geometry {
        heads: 32,
        head_dim: 128,
        group: 4,
        page_size: 64,
        max_top_k: 2048,
        always_tail: true,
        position_dtype: DType::Bf16,
        key_norm_eps: 1e-6,
        verify_rows: 6,
    };
    let merges = merge_tree::entries(&entries(PROFILE, geometry));
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
    .expect("tree merge cubins build and audit");
    assert_eq!(cubins.len(), 1 + merge_tree::MUTANTS.len());
    for cubin in &cubins {
        // The global selection's merge and the per-rank forms at 2 and 4.
        assert_eq!(cubin.entries.len(), 3);
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
        .expect("inspect tree merge SASS");
    assert!(sass.status.success());
    let text = String::from_utf8_lossy(&sass.stdout);
    for opcode in ["SHFL.BFLY", "ACQBULK", "PREEXIT", "BAR.SYNC"] {
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
