use super::*;
use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::contract::Argument;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;

#[test]
fn qualified_buckets_bind_the_measured_plans_and_private_scratch() {
    let mut count = 0;
    for ranks in [1, 2, 4] {
        let backend = register(
            super::super::mxfp8_fold::register(Sm120::new(), &FLASH, ranks).unwrap(),
            &FLASH,
            ranks,
        )
        .unwrap();
        for entry in entries().unwrap().into_iter().filter(|e| e.tile_n == 32) {
            for rows in 1..=64 {
                let q = super::super::mxfp8_fold::routed_query(entry.ticket.query, rows);
                let Ok(op) = backend.mxfp8_fold_gemm(&q, Need::rows(rows)) else {
                    assert_ne!(ranks, 4);
                    continue;
                };
                let selected = if ranks == 4 { choice(q, rows) } else { None };
                assert_eq!(
                    op.info()
                        .symbols
                        .iter()
                        .any(|(s, _)| s.contains("_reuse32_")),
                    selected.is_some()
                );
                let Some((tile, folds)) = selected else {
                    continue;
                };
                let mut entry = entry.clone();
                entry.tile_n = tile;
                entry.ticket.query = q;
                let sizes = entry.ticket.legacy().unwrap().bytes(rows);
                let v = |handle, bytes| View { handle, bytes };
                let b = Mxfp8FoldGemmBuffers {
                    activations: v(1, sizes.activations),
                    activation_scales: v(2, sizes.activation_scales),
                    weights: v(3, sizes.weights),
                    weight_scales: v(4, sizes.weight_scales),
                    output: v(5, sizes.output),
                    live_rows: v(6, 4),
                    status: v(7, sizes.status + 12),
                    status_row_offset: 3,
                };
                let work = v(8, entry.workspace(rows).unwrap());
                let calls = op.launches(&b, work, Counts::rows(rows)).unwrap();
                assert_eq!(calls.len(), 2);
                let plans = entry.plans(rows, folds).unwrap();
                for (call, (spec, plan)) in calls.iter().zip(entry.specs().into_iter().zip(plans)) {
                    assert_eq!(call.symbol, spec.symbol);
                    assert_eq!(call.plan, plan);
                }
                assert_eq!(calls[0].arguments.0[2], Argument::DeviceBuffer(8));
                assert_eq!(calls[0].arguments.0[5], Argument::U64(3));
                assert_eq!(calls[1].arguments.0[8], Argument::DeviceBuffer(8));
                assert_eq!(calls[1].arguments.0[11], Argument::U32(folds));
                assert!(
                    op.launches(&b, v(8, work.bytes - 1), Counts::rows(rows))
                        .is_err()
                );
                assert!(
                    op.launches(&b, v(1, work.bytes), Counts::rows(rows))
                        .is_err()
                );
                assert!(op.launches(&b, work, Counts::rows(rows - 1)).is_err());
                count += 1;
            }
        }
    }
    assert_eq!(count, 42);
}

#[test]
fn head_row_policy_is_limited_to_flash_tp4() {
    for ranks in [1, 2, 4] {
        let old = super::super::f32_head::entry(&FLASH, ranks)
            .unwrap()
            .with_split_decode();
        let routed = head(&FLASH, ranks, old).unwrap();
        for rows in 1..=64 {
            assert_eq!(
                routed.entry_for_rows(rows).uses_decode_reuse(rows),
                ranks == 4 && (9..=32).contains(&rows)
            );
        }
    }
}
