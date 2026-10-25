//! The served selectors must reach qualified rows without widening topology
//! or projection admission.
use super::*;
use bundle_runtime::geometry::Ranks;
use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::sinkhorn_expand_mix::SinkhornRoundExpandMixBuffers;
use model_deepseek41_contract::geometry::FLASH;

#[test]
fn standalone_hc_chain_covers_small_flash_tp4_rows() {
    let v = |handle| View {
        handle,
        bytes: 1 << 30,
    };
    let buffers = SinkhornRoundExpandMixBuffers {
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
    };
    let entry = hc_round_expand_mix::entry(&FLASH);
    let reference = hc_round_expand_mix::register(Sm120::new(), &FLASH)
        .sinkhorn_round_expand_mix(&entry.query, Need::rows(32))
        .unwrap();
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = crate::shape::Shape::new(&FLASH, ranks).unwrap();
        let kernels = crate::Kernels::new(&shape).unwrap();
        let selected = kernels
            .backend
            .sinkhorn_round_expand_mix(&entry.query, Need::rows(32))
            .unwrap();
        assert_eq!(selected.info().workspace, reference.info().workspace);
        for rows in 1..=32 {
            let old = reference
                .launches(&buffers, v(14), Counts::rows(rows))
                .unwrap();
            let new = selected
                .launches(&buffers, v(14), Counts::rows(rows))
                .unwrap();
            let warps = match rows {
                1 => Some(8),
                8 => Some(4),
                2..=4 if ranks == Ranks::TP4 => Some(8),
                5..=7 if ranks == Ranks::TP4 => Some(4),
                _ => None,
            };
            let mut expected = old;
            if let Some(warps) = warps {
                let chain = hc_round_chain::entry(&FLASH);
                expected[0].symbol = chain.symbols()[usize::from(warps == 8)].clone();
                expected[0].plan = chain.plan(rows, rows, warps).unwrap();
            }
            assert_eq!(new, expected, "ranks={ranks:?} rows={rows}");
        }
    }
}

#[test]
fn query_fold_tails_keep_other_projections_and_topologies_scoped() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = crate::shape::Shape::new(&FLASH, ranks).unwrap();
        for dense in crate::ops::dense::projections(&shape)
            .into_iter()
            .filter(|d| matches!(d.id, "attention.q-a" | "attention.q-b" | "attention.kv"))
        {
            for rows in 1..=64 {
                let query = crate::Kernels::dense_decode_query(dense.query, rows);
                let selected =
                    crate::Kernels::prepare_local_fold(&shape, dense.id, query, Need::rows(rows))
                        .unwrap();
                let expected = ranks == Ranks::TP4
                    && if dense.id == "attention.kv" {
                        matches!(rows, 1 | 2 | 4 | 8 | 16 | 32)
                    } else {
                        rows <= 32 || matches!(rows, 35 | 40 | 45 | 50 | 55 | 60)
                    };
                assert_eq!(
                    selected.is_some(),
                    expected,
                    "ranks={ranks:?} id={} rows={rows}",
                    dense.id
                );
            }
        }
    }
}
