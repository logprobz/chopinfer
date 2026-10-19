use kernel_core::{
    contract::{DType, RmsNormKind},
    rmsnorm::{RmsNormNumerics, RmsNormRequirement},
    rmsnorm_quantize::RmsNormQuantizeQuery,
};
use kernels_sm120::row::{rmsnorm_quantize, rmsnorm_quantize_tiles::Entry};
fn entry() -> Entry {
    Entry {
        reference: rmsnorm_quantize::Entry::new(
            RmsNormQuantizeQuery {
                norm: RmsNormRequirement::bf16(
                    5120,
                    1e-20,
                    RmsNormKind::Plain,
                    false,
                    RmsNormNumerics::TREE_256,
                ),
                amax_floor_bits: 1e-4_f32.to_bits(),
            },
            32,
            "reference_norm",
        )
        .unwrap(),
        prefix: "chop_norm_tiles".into(),
    }
}
#[test]
fn tiles_keep_original_rms_lanes_and_unique_output_owners() {
    let e = entry();
    for rows in [1, 2, 4, 8, 16, 32] {
        for tiled in [false, true] {
            let p = e.plan(rows, rows, tiled).unwrap();
            assert_eq!(p.grid, [if tiled { 20 } else { 1 }, rows, 1]);
            assert_eq!(p.block, [256, 1, 1]);
            assert_eq!(p.workspace_bytes, 0);
        }
        assert!(e.plan(rows, rows + 1, true).is_err());
    }
    assert!(e.plan(0, 0, true).is_err());
    assert!(e.plan(33, 1, true).is_err());
    let mut bad = e.clone();
    bad.reference.query.norm.input = DType::F32;
    assert!(bad.validate().is_err());
    for tiles in [1, 20] {
        let mut owners = vec![0; 5120];
        let mut groups = vec![0; 160];
        for tile in 0..tiles {
            for t in 0..256 {
                for c in (t * 8..5120).step_by(2048) {
                    if (c / 256) % tiles == tile {
                        for n in &mut owners[c..c + 8] {
                            *n += 1;
                        }
                    }
                }
            }
            for w in 0..8 {
                for g in (tile * 8 + w..160).step_by(tiles * 8) {
                    groups[g] += 1;
                }
            }
        }
        assert!(owners.into_iter().chain(groups).all(|n| n == 1));
    }
    let source = include_str!("../cuda/row/rmsnorm_quantize_tiles.cuh");
    for name in ["blockIdx", "griddep_", "__shared__"] {
        assert!(!source.contains(name));
    }
    assert!(source.contains("sum_cta(square, shared.sums)"));
    assert!(source.contains("group = warp; group < Width / 32; group += 8"));
    assert!(source.contains("row_status<8>"));
}
#[test]
fn norm_tiles_compile_without_spills() {
    let tc = kernel_build::Toolchain::detect().unwrap();
    assert!(!tc.skipped);
    let root = std::env::temp_dir().join(format!("norm-tiles-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let built = kernel_build::build(
        &tc,
        &[entry().cubin_spec("norm_tiles").unwrap()],
        &root.join("out"),
        &root.join("lock"),
        kernel_build::BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    for image in built {
        for usage in image.entries {
            assert_eq!((usage.local_bytes, usage.stack_bytes), (0, 0));
            eprintln!(
                "NORM_TILES_RESOURCE {} regs={} shared={} local={} stack={}",
                usage.symbol,
                usage.registers,
                usage.shared_static_bytes,
                usage.local_bytes,
                usage.stack_bytes
            );
        }
    }
}
