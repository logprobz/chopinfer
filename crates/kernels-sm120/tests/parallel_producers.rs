use kernel_core::{
    contract::RmsNormKind,
    rmsnorm::{RmsNormNumerics, RmsNormRequirement},
    rmsnorm_quantize::RmsNormQuantizeQuery,
    sinkhorn_expand_mix::SinkhornExpandMixQuery,
};
use kernels_sm120::{
    row::{rmsnorm_quantize, rmsnorm_quantize_cached},
    sinkhorn_hc::{expand_mix, lagged, round_chain_partials},
};
fn norm() -> rmsnorm_quantize_cached::Entry {
    rmsnorm_quantize_cached::Entry {
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
            "norm_reference",
        )
        .unwrap(),
        prefix: "norm_cached".into(),
    }
}
fn hc() -> round_chain_partials::Entry {
    round_chain_partials::Entry {
        reference: expand_mix::Entry {
            query: SinkhornExpandMixQuery {
                mix: lagged::LAB.mix,
            },
            capacity: 32,
            symbol: "round_reference".into(),
            probe: false,
        },
        prefix: "round_chain".into(),
    }
}
#[test]
fn producers_preserve_reduction_owners_and_private_workspace() {
    for rows in [1, 8] {
        for warps in [8, 16, 32] {
            let p = norm().plan(rows, rows, warps).unwrap();
            assert_eq!(p.grid, [1, rows, 1]);
            assert_eq!(p.block, [32 * warps, 1, 1]);
            assert_eq!(p.workspace_bytes, 0);
            let mut norms = vec![0; 5120];
            let mut groups = vec![0; 160];
            for t in 0..warps as usize * 32 {
                for c in (t * 8..5120).step_by(warps as usize * 256) {
                    for n in &mut norms[c..c + 8] {
                        *n += 1;
                    }
                }
            }
            for w in 0..warps as usize {
                for g in (w..160).step_by(warps as usize) {
                    groups[g] += 1;
                }
            }
            assert!(norms.into_iter().chain(groups).all(|n| n == 1));
        }
        for warps in [4, 8] {
            let p = hc().plan(rows, rows, warps).unwrap();
            assert_eq!(p.grid, [160, rows, 1]);
            assert_eq!(
                p.workspace_bytes,
                hc().reference.plans(rows).unwrap()[0].workspace_bytes
            );
        }
    }
    for (rows, live) in [(0, 0), (1, 8), (33, 1)] {
        assert!(norm().plan(rows, live, 32).is_err());
        assert!(hc().plan(rows, live, 4).is_err());
    }
    for source in [
        include_str!("../cuda/row/rmsnorm_quantize_cached.cuh"),
        include_str!("../cuda/sinkhorn_hc/round_chain_partials.cuh"),
    ] {
        for forbidden in ["blockIdx", "griddep_", "__shared__"] {
            assert!(!source.contains(forbidden));
        }
    }
    let source = include_str!("../cuda/row/rmsnorm_quantize_cached.cuh");
    assert!(source.contains("c += 256 * 8"));
    assert!(source.contains("lane < 8 ? shared.sums[lane] : 0.f"));
    let mut bad = hc();
    bad.reference.probe = true;
    assert!(bad.validate().is_err());
}
#[test]
fn parallel_producers_compile_without_spills() {
    let tc = kernel_build::Toolchain::detect().unwrap();
    assert!(!tc.skipped);
    let root = std::env::temp_dir().join(format!("parallel-producers-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let built = kernel_build::build(
        &tc,
        &[
            norm().cubin_spec("norm_cached").unwrap(),
            hc().cubin_spec("round_chain").unwrap(),
        ],
        &root.join("out"),
        &root.join("lock"),
        kernel_build::BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    assert_eq!(built.iter().map(|m| m.entries.len()).sum::<usize>(), 5);
    for m in built {
        for e in m.entries {
            assert_eq!((e.local_bytes, e.stack_bytes), (0, 0));
            eprintln!(
                "PRODUCER_RESOURCE {} regs={} shared={} local={} stack={}",
                e.symbol, e.registers, e.shared_static_bytes, e.local_bytes, e.stack_bytes
            );
        }
    }
}
