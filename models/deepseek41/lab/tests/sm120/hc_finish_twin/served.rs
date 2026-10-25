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
    let backend = b::hc_round_chain::register(backend, &FLASH, 4);
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
    use crate::device::{Candidate, variant};

    #[test]
    fn verify_six_twin_keeps_the_composed_abi() {
        for widen in [false, true] {
            for alias in [false, true] {
                let views = (1..=22)
                    .map(|handle| View {
                        handle: match (alias, handle) {
                            (true, 13) => 4,
                            (true, 14) => 5,
                            _ => handle,
                        },
                        bytes: 1 << 30,
                    })
                    .collect::<Vec<_>>();
                let pair = plans(6, &views, widen).unwrap();
                assert_eq!(pair.len(), 2);
                let chain = b::hc_round_chain::entry(&FLASH);
                assert_eq!(pair[0].symbol, format!("{}_w4_v1", chain.prefix));
                assert_eq!(pair[0].plan, chain.plan(6, 6, 4).unwrap());
                assert_eq!(
                    pair[1].symbol,
                    b::hc_finish_norm::entry(&FLASH).symbols()[1]
                );
                let Candidate(entry, warps) = variant("twin32").unwrap();
                let twin = entry.compose(&pair[1], warps).unwrap();
                assert_eq!(twin.arguments, pair[1].arguments);
                assert_eq!(twin.plan, entry.plan(6, 6, 32).unwrap());
                assert_eq!(twin.symbol, entry.symbol(32));
                assert!(entry.compose(&pair[0], warps).is_err());
            }
        }
        for name in ["served", "tail8", "split8", "twin8", "twin16", "twin64"] {
            assert!(variant(name).is_err());
        }
    }
}
