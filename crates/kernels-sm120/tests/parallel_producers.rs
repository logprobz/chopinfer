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

fn finish_backend() -> kernels_sm120::backend::Sm120 {
    kernels_sm120::backend::Sm120::default()
        .with_sinkhorn_round_expand_mix(hc().reference.clone())
        .with_sinkhorn_round_chain(hc(), vec![(1, 8), (8, 4)])
        .with_rmsnorm_quantize(norm().reference.clone())
        .with_rmsnorm_quantize_cached(norm(), vec![(1, 32), (8, 32)])
}
fn finish_entry() -> kernels_sm120::sinkhorn_hc::finish_norm::Entry {
    kernels_sm120::sinkhorn_hc::finish_norm::Entry {
        mixer: hc().reference,
        norm: norm(),
        prefix: "finish_norm".into(),
    }
}
#[test]
fn row_tiles_preserve_arguments_and_refuse_unqualified_or_ambiguous_buckets() {
    use kernel_core::{
        backend::{Backend, Counts, Need, View},
        sinkhorn_expand_mix::SinkhornRoundNormQuery,
    };
    let e = finish_entry();
    let query = SinkhornRoundNormQuery {
        mixer: e.mixer.query,
        norm: e.norm.reference.query,
    };
    let rows = kernels_sm120::sinkhorn_hc::round_rows::Entry {
        reference: hc(),
        prefix: "rounded_rows".into(),
    };
    let base = || finish_backend().with_sinkhorn_finish_norm(e.clone(), vec![8, 32]);
    let old = base()
        .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
        .unwrap();
    let new = base()
        .with_sinkhorn_finish_norm_rows(rows.clone(), vec![(32, 2)])
        .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
        .unwrap();
    let workspace = View {
        handle: 20,
        bytes: old.info().workspace,
    };
    assert_eq!(new.info().workspace, old.info().workspace);
    for count in [1, 6, 8, 16, 32] {
        let mut expected = old
            .launches(&finish_buffers(), workspace, Counts::rows(count))
            .unwrap();
        if count == 32 {
            expected[0].symbol = rows.symbols()[0].clone();
            expected[0].plan = rows.plan(count, count, 2).unwrap();
        }
        assert_eq!(
            new.launches(&finish_buffers(), workspace, Counts::rows(count))
                .unwrap(),
            expected
        );
    }
    for buckets in [
        vec![],
        vec![(6, 2)],
        vec![(32, 3)],
        vec![(32, 2), (8, 2)],
        vec![(32, 2), (32, 4)],
    ] {
        assert!(
            base()
                .with_sinkhorn_finish_norm_rows(rows.clone(), buckets)
                .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
                .is_err()
        );
    }
    assert!(
        base()
            .with_sinkhorn_finish_norm_rows(rows.clone(), vec![(32, 2)])
            .with_sinkhorn_finish_norm_rows(rows.clone(), vec![(32, 2)])
            .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
            .is_err()
    );
    let mut wrong = rows;
    wrong.reference.reference.symbol.push_str("_different");
    assert!(
        base()
            .with_sinkhorn_finish_norm_rows(wrong, vec![(32, 2)])
            .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
            .is_err()
    );
}
fn finish_buffers()
-> kernel_core::sinkhorn_expand_mix::SinkhornRoundNormBuffers<kernel_core::backend::View> {
    use kernel_core::{
        backend::View,
        rmsnorm_quantize::RmsNormQuantizeBuffers as Norm,
        sinkhorn_expand_mix::{
            SinkhornRoundExpandMixBuffers as Hc, SinkhornRoundNormBuffers as Buffers,
        },
    };
    let v = |handle| View {
        handle,
        bytes: 1 << 30,
    };
    Buffers {
        hc: Hc {
            x: v(1),
            sum: v(2),
            rounded: v(3),
            post_in: v(4),
            comb_in: v(5),
            expanded: v(6),
            fn_: v(7),
            base: v(8),
            scale: v(9),
            pre_in: v(10),
            collapsed: v(11),
            pre: v(12),
            post: v(4),
            comb: v(5),
            live_rows: v(13),
        },
        norm: Norm {
            input: v(11),
            weight: v(14),
            normed: v(15),
            widened: Some(v(16)),
            codes: v(17),
            scales: v(18),
            live_rows: v(13),
            status: v(19),
            status_row_offset: 3,
        },
        tickets: None,
    }
}
#[test]
fn fused_tail_preserves_arguments_and_refuses_cross_op_aliases() {
    use kernel_core::{
        backend::{Backend, Counts, Need, View},
        sinkhorn_expand_mix::SinkhornRoundNormQuery,
    };
    let e = finish_entry();
    let query = SinkhornRoundNormQuery {
        mixer: e.mixer.query,
        norm: e.norm.reference.query,
    };
    let backend = finish_backend();
    let hc = backend
        .sinkhorn_round_expand_mix(&query.mixer, Need::rows(32))
        .unwrap();
    let norm = backend
        .rmsnorm_quantize(&query.norm, Need::rows(32))
        .unwrap();
    let op = backend
        .with_sinkhorn_finish_norm(e, vec![1, 8])
        .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
        .unwrap();
    let workspace = View {
        handle: 20,
        bytes: op.info().workspace,
    };
    assert_eq!(op.info().workspace, hc.info().workspace);
    assert!(
        op.info()
            .symbols
            .iter()
            .any(|(s, _)| s == "finish_norm_w32_v1")
    );
    assert!(
        !op.info()
            .symbols
            .iter()
            .any(|(s, _)| s == "finish_norm_w16_v1")
    );
    for rows in [1, 2, 4, 8, 16, 32] {
        for widen in [false, true] {
            let mut b = finish_buffers();
            if !widen {
                b.norm.widened = None;
            }
            let mut old = hc.launches(&b.hc, workspace, Counts::rows(rows)).unwrap();
            old.extend(
                norm.launches(&b.norm, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
            );
            let actual = op.launches(&b, workspace, Counts::rows(rows)).unwrap();
            assert_eq!(format!("{:?}", actual[0]), format!("{:?}", old[0]));
            if [1, 8].contains(&rows) {
                assert_eq!(actual.len(), 2);
                assert_eq!(actual[1].symbol, "finish_norm_w32_v1");
                assert_eq!(actual[1].plan.grid, [1, rows, 1]);
                assert_eq!(actual[1].plan.block, [1024, 1, 1]);
                let mut args = old[1].arguments.0.clone();
                args.extend_from_slice(&old[2].arguments.0[1..6]);
                args.extend_from_slice(&old[2].arguments.0[7..]);
                assert_eq!(actual[1].arguments.0, args);
            } else {
                assert_eq!(format!("{actual:?}"), format!("{old:?}"));
            }
        }
    }
    for which in 0..8 {
        let mut b = finish_buffers();
        match which {
            0 => b.norm.input = b.hc.expanded,
            1 => b.norm.live_rows = b.hc.base,
            2 => b.norm.codes = b.hc.expanded,
            3 => b.norm.weight = workspace,
            4 => b.norm.status = b.hc.pre,
            5 => b.norm.status.bytes = 12,
            6 => b.norm.widened = Some(b.norm.codes),
            _ => b.hc.pre_in = b.hc.pre,
        }
        assert!(
            op.launches(&b, workspace, Counts::rows(1)).is_err(),
            "control {which}"
        );
    }
    assert!(
        op.launches(
            &finish_buffers(),
            View {
                bytes: 0,
                ..workspace
            },
            Counts::rows(1)
        )
        .is_err()
    );
    assert!(
        op.launches(&finish_buffers(), workspace, Counts::rows(33))
            .is_err()
    );
}
#[test]
fn fused_tail_admission_requires_matching_qualified_constituents() {
    use kernel_core::{
        backend::{Backend, Need},
        sinkhorn_expand_mix::SinkhornRoundNormQuery,
    };
    let e = finish_entry();
    let query = SinkhornRoundNormQuery {
        mixer: e.mixer.query,
        norm: e.norm.reference.query,
    };
    for buckets in [vec![], vec![8, 1], vec![1, 1], vec![0], vec![33]] {
        assert!(
            finish_backend()
                .with_sinkhorn_finish_norm(e.clone(), buckets)
                .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
                .is_err()
        );
    }
    assert!(
        finish_backend()
            .with_sinkhorn_finish_norm(e.clone(), vec![1, 8])
            .with_sinkhorn_finish_norm(e.clone(), vec![1, 8])
            .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
            .is_err()
    );
    let mut bad = query;
    bad.norm.norm.width = 256;
    assert!(
        finish_backend()
            .with_sinkhorn_finish_norm(e.clone(), vec![1, 8])
            .sinkhorn_round_expand_mix_norm(&bad, Need::rows(32))
            .is_err()
    );
    let mut bad = e;
    bad.norm.prefix = "different_cached_symbol".into();
    assert!(
        finish_backend()
            .with_sinkhorn_finish_norm(bad, vec![1, 8])
            .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
            .is_err()
    );
}
