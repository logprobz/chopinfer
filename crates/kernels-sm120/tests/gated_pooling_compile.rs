//! Compile and audit the F32 pooling pair without opening a GPU.

use kernel_build::{BuildOptions, Toolchain, build};
use kernel_core::contract::RecurrentPhase;
use kernel_core::gated_pooling::GatedPoolingQuery;
use kernels_sm120::gated_pooling::{Entry, cubin_spec};
use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn pool_and_norm_compile_with_pdl_and_without_spills_or_tf32() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is required");
    assert!(!toolchain.skipped);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("gated-pooling-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let mut entries = Vec::new();
    for width in [128, 512] {
        for (name, phase) in [
            ("prefill", RecurrentPhase::Prefill),
            ("decode", RecurrentPhase::Decode),
            ("verify", RecurrentPhase::Verify { max_rows: 6 }),
        ] {
            entries.push(
                Entry::new(
                    GatedPoolingQuery {
                        group: 2,
                        width,
                        norm_eps_bits: 1e-20f32.to_bits(),
                        phase,
                        capacity: 12,
                    },
                    &format!("chop_pool_{name}_w{width}"),
                )
                .unwrap(),
            );
        }
    }
    let mut spec = cubin_spec("gated_pooling_compile_v1", &entries).unwrap();
    for width in [128, 512] {
        let norm = kernels_sm120::gated_pooling::ratio_one_norm(width, 1e-20, 12).unwrap();
        spec.entries
            .push(norm.entry(format!("chop_ratio_one_w{width}_v1")));
    }
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
    .expect("pool and norm entries compile and audit");
    assert_eq!(cubins.len(), 1);
    assert_eq!(cubins[0].entries.len(), 14);
    let sass = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&cubins[0].path)
        .output()
        .unwrap();
    assert!(sass.status.success());
    let text = String::from_utf8(sass.stdout).unwrap();
    assert!(!text.contains("TF32"));
    for entry in &cubins[0].entries {
        assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
        assert_eq!(entry.stack_bytes, 0, "{} uses a stack", entry.symbol);
        let header = format!("Function : {}\n", entry.symbol);
        let body = text.split_once(&header).expect("entry has SASS").1;
        let body = body.split("Function : ").next().unwrap();
        assert!(body.contains("ACQBULK"), "{} has no PDL wait", entry.symbol);
        assert!(
            body.contains("PREEXIT"),
            "{} has no PDL release",
            entry.symbol
        );
    }
    fs::remove_dir_all(root).unwrap();
}
