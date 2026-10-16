//! Builds every pooled indexer entry at the served replicated-indexer shape
//! (32 heads of 128, pools of 4, top 2048 with the tail, 64-token pages,
//! verify items of 6 rows) and checks the audits and the SASS.
use kernel_build::{BuildOptions, Toolchain, build};
use kernel_core::contract::DType;
use kernels_sm120::pooled_indexer::{Geometry, cubin_spec, entries};
use kernels_sm120::{Profile, ProfileNames};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "pooled-indexer-compile",
    tag: "pooled_indexer_compile",
});

fn served(position_dtype: DType) -> Geometry {
    Geometry {
        heads: 32,
        head_dim: 128,
        group: 4,
        page_size: 64,
        max_top_k: 2048,
        always_tail: true,
        position_dtype,
        key_norm_eps: 1e-6,
        verify_rows: 6,
    }
}

#[test]
fn pooled_indexer_entries_compile_without_spills() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is available");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into())).join(
        format!("pooled-indexer-cubin-{}-{stamp}", std::process::id()),
    );
    fs::create_dir_all(&root).unwrap();
    let mut spec = cubin_spec(PROFILE, &entries(PROFILE, served(DType::Bf16))).unwrap();
    // The F32 position embedding form of the key write.
    let f32_position = entries(PROFILE, served(DType::F32))
        .into_iter()
        .find(|entry| entry.name.ends_with("key_write_prefill"))
        .map(|mut entry| {
            entry.name.push_str("_f32_position");
            entry
        })
        .unwrap();
    spec.entries.extend(f32_position.entry_specs().unwrap());
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
    .expect("pooled indexer cubin builds and audits");
    assert_eq!(cubins.len(), 1);
    // Nine entries, a merge for each of the three selections and the F32
    // position key write.
    assert_eq!(cubins[0].entries.len(), 13);
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
        .expect("inspect pooled indexer SASS");
    assert!(sass.status.success());
    let text = String::from_utf8_lossy(&sass.stdout);
    // BF16 MMA, ldmatrix and cp.async in the scores; the PDL pair everywhere.
    for opcode in [
        "HMMA.16816.F32.BF16",
        "LDSM.16.M88.4",
        "LDGSTS",
        "ACQBULK",
        "PREEXIT",
    ] {
        assert!(text.contains(opcode), "the SASS has no {opcode}");
    }
    fs::remove_dir_all(root).unwrap();
}
