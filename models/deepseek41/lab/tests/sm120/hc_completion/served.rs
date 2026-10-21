//! The bundle's prepared HC finish/norm op: the served reference launches.
use bundle_deepseek41::sm120 as b;
use kernel_core::{
    backend::{Backend, Counts, Launch, Need, View},
    properties::Invariance,
    rmsnorm_quantize::RmsNormQuantizeBuffers as Norm,
    sinkhorn_expand_mix::{
        SinkhornRoundExpandMixBuffers as Hc, SinkhornRoundNormBuffers as Buffers,
        SinkhornRoundNormQuery,
    },
};
use kernel_harness::LabResult;
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::geometry::FLASH;

/// Views in role order: x, sum, rounded, post_in, comb_in, expanded, fn,
/// base, scale, pre_in, collapsed, pre, post, comb, live, norm weight,
/// normed, widened, codes, scales, status, workspace.
pub fn plans(rows: u32, views: &[View], widen: bool) -> LabResult<Vec<Launch>> {
    if views.len() != 22 {
        return Err("served HC plan needs 22 views".into());
    }
    let backend = b::norm_quantize::register(Sm120::new(), &FLASH)?;
    let backend = b::norm_quantize_cached::register(backend, &FLASH);
    let backend = b::hc_round_expand_mix::register(backend, &FLASH);
    let backend = b::hc_round_chain::register(backend, &FLASH);
    let backend = b::hc_finish_norm::register_pair(backend, &FLASH);
    let entry = b::hc_finish_norm::entry(&FLASH);
    let op = backend
        .sinkhorn_round_expand_mix_norm(
            &SinkhornRoundNormQuery {
                mixer: entry.mixer.query,
                norm: entry.norm.reference.query,
            },
            Need::rows(32).with_exact(Invariance::any_rows(32)),
        )
        .map_err(|e| format!("{e:?}"))?;
    let v = |index: usize| views[index];
    op.launches(
        &Buffers {
            hc: Hc {
                x: v(0),
                sum: v(1),
                rounded: v(2),
                post_in: v(3),
                comb_in: v(4),
                expanded: v(5),
                fn_: v(6),
                base: v(7),
                scale: v(8),
                pre_in: v(9),
                collapsed: v(10),
                pre: v(11),
                post: v(12),
                comb: v(13),
                live_rows: v(14),
            },
            norm: Norm {
                input: v(10),
                weight: v(15),
                normed: v(16),
                widened: widen.then(|| v(17)),
                codes: v(18),
                scales: v(19),
                live_rows: v(14),
                status: v(20),
                status_row_offset: 0,
            },
            tickets: None,
        },
        v(21),
        Counts::rows(rows),
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Candidate, VARIANTS, variant};
    use kernel_core::contract::Argument;
    use kernels_sm120::sinkhorn_hc::completion::ARGUMENTS;

    fn views(alias: bool) -> Vec<View> {
        (1..=23)
            .map(|handle| View {
                handle: match (alias, handle) {
                    (true, 13) => 4,
                    (true, 14) => 5,
                    _ => handle,
                },
                bytes: 1 << 30,
            })
            .collect()
    }

    #[test]
    fn served_pair_and_compositions_keep_every_argument() {
        let d = |index: u64| Argument::DeviceBuffer(index + 1);
        for rows in [1, 6, 8] {
            for widen in [false, true] {
                for alias in [false, true] {
                    let views = views(alias);
                    let pair = plans(rows, &views[..22], widen).unwrap();
                    assert_eq!(pair.len(), 2);
                    let chain = b::hc_round_chain::entry(&FLASH);
                    let warps = if rows == 1 { 8 } else { 4 };
                    assert_eq!(pair[0].symbol, format!("{}_w{warps}_v1", chain.prefix));
                    assert_eq!(pair[0].plan, chain.plan(rows, rows, warps).unwrap());
                    let tail = b::hc_finish_norm::entry(&FLASH);
                    assert_eq!(pair[1].symbol, tail.symbols()[1]);
                    let post = |i: u64| {
                        if alias && i == 12 {
                            d(3)
                        } else if alias && i == 13 {
                            d(4)
                        } else {
                            d(i)
                        }
                    };
                    let wide = if widen {
                        d(17)
                    } else {
                        Argument::NullDevicePointer
                    };
                    let mut expected = [0, 1, 2, 3, 4, 5, 6].map(d).to_vec();
                    expected.extend([7, 8, 9, 10, 11, 12, 13].map(post));
                    expected.extend([d(21), d(14), d(15), d(16), wide, d(18), d(19), d(20)]);
                    expected.extend([Argument::U64(0), Argument::U32(1e-20_f32.to_bits()), d(22)]);
                    for name in VARIANTS {
                        let (entry, w) = match variant(name).unwrap() {
                            Candidate::Completion(entry, w) => (entry, w),
                            Candidate::Twin(entry, w) => {
                                // The tail's own arguments on two CTAs per row;
                                // only the served tail is replaced.
                                let twin = entry.compose(&pair[1], w).unwrap();
                                assert_eq!(twin.arguments, pair[1].arguments, "{name}");
                                assert_eq!(twin.plan, entry.plan(rows, rows, w).unwrap());
                                assert_eq!(twin.symbol, entry.symbol(w));
                                assert!(entry.compose(&pair[0], w).is_err());
                                continue;
                            }
                        };
                        let fused = entry.compose(&pair[0], &pair[1], views[22], w).unwrap();
                        assert_eq!(fused.arguments.0.len(), ARGUMENTS);
                        assert_eq!(fused.arguments.0, expected, "{name}");
                        assert_eq!(fused.plan, entry.plan(rows, rows, w).unwrap());
                        assert_eq!(fused.symbol, entry.symbol(w));
                        // A ticket that aliases any role, or is too small, is refused.
                        for role in [0, 5, 12, 21] {
                            assert!(entry.compose(&pair[0], &pair[1], views[role], w).is_err());
                        }
                        let short = View {
                            handle: 23,
                            bytes: u64::from(rows) * 4 - 1,
                        };
                        assert!(entry.compose(&pair[0], &pair[1], short, w).is_err());
                        // Swapped or altered constituents are refused.
                        assert!(entry.compose(&pair[1], &pair[0], views[22], w).is_err());
                        let mut other = pair[1].clone();
                        other.symbol = tail.symbols()[0].clone();
                        assert!(entry.compose(&pair[0], &other, views[22], w).is_err());
                        let mut other = pair[0].clone();
                        other.arguments.0[7] = d(9);
                        assert!(entry.compose(&other, &pair[1], views[22], w).is_err());
                    }
                }
            }
        }
        for name in ["served", "tail4", "split32", "rows8", "twin4", "twin64"] {
            assert!(variant(name).is_err());
        }
    }
}
