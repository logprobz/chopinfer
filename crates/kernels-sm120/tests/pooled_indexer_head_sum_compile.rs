//! Builds every head-sum selection entry (four heads of 128, pools of 4,
//! top 2048 with the tail, 64-token pages) and checks the audits and the
//! SASS: per selection two chains of four BF16 MMAs and the scale folded to
//! the F32 nearest to 128^-0.5.
use kernel_build::{BuildOptions, Toolchain, build};
use kernels_sm120::pooled_indexer::head_sum::{Geometry, cubin_spec, entries};
use kernels_sm120::{Profile, ProfileNames};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "head-sum-compile",
    tag: "head_sum_compile",
});

#[test]
fn head_sum_entries_compile_without_spills() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is available");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into()))
        .join(format!("head-sum-cubin-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let geometry = Geometry {
        heads: 4,
        head_dim: 128,
        group: 4,
        page_size: 64,
        max_top_k: 2048,
        always_tail: true,
    };
    let spec = cubin_spec(PROFILE, &entries(PROFILE, geometry)).unwrap();
    let cubins = build(
        &toolchain,
        &[spec],
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("head-sum cubin builds and audits");
    assert_eq!(cubins.len(), 1);
    // Five entries and a merge for each of the three selections.
    assert_eq!(cubins[0].entries.len(), 8);
    assert!(cubins[0].entries.iter().all(|entry| entry.local_bytes == 0));
    for entry in &cubins[0].entries {
        eprintln!(
            "{}: {} registers, {} static shared bytes",
            entry.symbol, entry.registers, entry.shared_static_bytes
        );
    }
    let sass = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&cubins[0].path)
        .output()
        .expect("inspect head-sum SASS");
    assert!(sass.status.success());
    let text = String::from_utf8_lossy(&sass.stdout);
    for opcode in ["LDSM.16.M88.4", "LDGSTS", "ACQBULK", "PREEXIT"] {
        assert!(text.contains(opcode), "the SASS has no {opcode}");
    }
    // Three selections of eight k16 MMAs each; the merges have none.
    assert_eq!(text.matches("HMMA.16816.F32.BF16").count(), 24);
    // `rsqrtf(128.0f)` folds to 0x3db504f3, the value the emulation
    // (`kernel_core::pooled_indexer_head_sum_oracle`) multiplies by.
    assert_eq!((1.0 / 128_f64.sqrt()) as f32, f32::from_bits(0x3db5_04f3));
    assert!(text.contains("FMUL") && text.contains("0x3db504f3"));
    assert!(!text.contains("MUFU.RSQ"));
    fs::remove_dir_all(root).unwrap();
}
