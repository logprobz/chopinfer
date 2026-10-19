//! The hyper-connection ops (`model_deepseek41_contract::hc`): each
//! sublayer's lagged mix-collapse, the F32 expand of its output into the
//! streams, and the head's unrounded F32 collapse by the last mixer's
//! `pre`, each prepared for a prefill's rows.

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::sinkhorn_expand_mix::{
    SinkhornExpandMixBuffers, SinkhornExpandMixQuery, SinkhornRoundExpandMixBuffers,
    SinkhornRoundNormBuffers, SinkhornRoundNormQuery,
};
use kernel_core::sinkhorn_hyper_connection::{
    LaggedMixCollapseBuffers, MixScratch, MixerWeights, StreamExpandBuffers,
    WeightedStreamHeadBuffers,
};
use model_deepseek41_contract::hc;
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::{program::Phase, shape::Shape};

/// The role paths of the mixer scratch a backend may list
/// (`OpInfo::bytes`).
const PARTIALS: &str = "scratch.partials";
const TICKETS: &str = "scratch.tickets";

pub(crate) struct HcOps {
    mix: Prepared<LaggedMixCollapseBuffers<View>>,
    decode_mix: Prepared<LaggedMixCollapseBuffers<View>>,
    expand: Prepared<StreamExpandBuffers<View>>,
    head: Prepared<WeightedStreamHeadBuffers<View>>,
    fused: Option<Prepared<SinkhornExpandMixBuffers<View>>>,
    round_fused: Option<Prepared<SinkhornRoundExpandMixBuffers<View>>>,
    finish_norm: Option<Prepared<SinkhornRoundNormBuffers<View>>>,
}

/// The mixer scratch the mix-collapse lists, in bytes: its partials,
/// rewritten by every launch, and its tickets, zeroed once at create.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MixScratchBytes {
    pub partials: Option<u64>,
    pub tickets: Option<u64>,
}

/// A step's spans of a lagged mix-collapse: the streams and the sublayer's
/// mixer weights (`fn`, base, scale) it reads with the previous `pre`, and
/// the collapsed row, `pre`, `post` and `comb` it writes.
pub(crate) struct MixViews {
    pub x: DeviceBufferSpanV1,
    pub mixer: [DeviceBufferSpanV1; 3],
    pub pre_in: DeviceBufferSpanV1,
    pub collapsed: DeviceBufferSpanV1,
    pub pre: DeviceBufferSpanV1,
    pub post: DeviceBufferSpanV1,
    pub comb: DeviceBufferSpanV1,
    pub partials: Option<DeviceBufferSpanV1>,
    pub tickets: Option<DeviceBufferSpanV1>,
    pub live_rows: DeviceBufferSpanV1,
}

/// A step's spans of an expand: the streams the sublayer read, its reduced
/// output, the mixer's `post` and `comb`, and the streams it writes.
pub(crate) struct ExpandViews {
    pub x: DeviceBufferSpanV1,
    pub y: DeviceBufferSpanV1,
    pub post: DeviceBufferSpanV1,
    pub comb: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

/// A step's spans of the head's collapse.
pub(crate) struct HeadViews {
    pub x: DeviceBufferSpanV1,
    pub pre: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

impl HcOps {
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
    ) -> Result<HcOps, Refused> {
        // Every phase mixes and expands its sublayers and collapses the
        // head's rows.
        let need = exact::need(PREFILL_TOKENS, exact::every_phase(super::verify_rows()));
        let g = shape.geometry;
        Ok(HcOps {
            finish_norm: {
                let capacity = model_deepseek41_contract::serving::DECODE_BATCH;
                let need = exact::need(capacity, exact::every_phase(capacity));
                let query = SinkhornRoundNormQuery {
                    mixer: SinkhornExpandMixQuery { mix: hc::mix(g) },
                    norm: model_deepseek41_contract::rows::norm_quantizers(g)[0],
                };
                match asks.ask("hc.round_expand_mix_norm", need, |need| {
                    backend.sinkhorn_round_expand_mix_norm(&query, need)
                }) {
                    Ok(op) => Some(op),
                    Err(Refused::Unsupported) => None,
                    Err(e) => return Err(e),
                }
            },
            mix: asks.ask("hc.mix_collapse", need, |need| {
                backend.sinkhorn_lagged_mix_collapse(&hc::mix_collapse(g, need.capacity), need)
            })?,
            decode_mix: {
                let need = exact::need(8, exact::every_phase(8));
                asks.ask("hc.decode_mix", need, |need| {
                    backend.sinkhorn_lagged_mix_collapse(&hc::mix_collapse(g, need.capacity), need)
                })?
            },
            expand: asks.ask("hc.expand", need, |need| {
                backend.sinkhorn_expand(&hc::expand(g, need.capacity), need)
            })?,
            head: asks.ask("hc.head", need, |need| {
                backend.sinkhorn_weighted_head(&hc::head(g, need.capacity), need)
            })?,
            fused: {
                let capacity = model_deepseek41_contract::serving::DECODE_BATCH;
                let need = exact::need(capacity, exact::every_phase(capacity));
                match asks.ask("hc.expand_mix", need, |need| {
                    backend.sinkhorn_expand_mix(&SinkhornExpandMixQuery { mix: hc::mix(g) }, need)
                }) {
                    Ok(op) => Some(op),
                    Err(Refused::Unsupported) => None,
                    Err(e) => return Err(e),
                }
            },
            round_fused: {
                let capacity = model_deepseek41_contract::serving::DECODE_BATCH;
                let need = exact::need(capacity, exact::every_phase(capacity));
                match asks.ask("hc.round_expand_mix", need, |need| {
                    backend.sinkhorn_round_expand_mix(
                        &SinkhornExpandMixQuery { mix: hc::mix(g) },
                        need,
                    )
                }) {
                    Ok(op) => Some(op),
                    Err(Refused::Unsupported) => None,
                    Err(e) => return Err(e),
                }
            },
        })
    }

    pub(crate) fn has_expand_mix(&self) -> bool {
        self.fused.is_some()
    }
    pub(crate) fn expand_mix(
        &self,
        rows: u32,
        views: SinkhornExpandMixBuffers<DeviceBufferSpanV1>,
        workspace: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        let op = self.fused.as_ref().ok_or(kernel_core::contract::Refusal(
            "fused expansion mixer unavailable",
        ))?;
        let mut bind = Bind::default();
        let b = SinkhornExpandMixBuffers {
            x: bind.view(views.x),
            y: bind.view(views.y),
            post_in: bind.view(views.post_in),
            comb_in: bind.view(views.comb_in),
            expanded: bind.view(views.expanded),
            fn_: bind.view(views.fn_),
            base: bind.view(views.base),
            scale: bind.view(views.scale),
            pre_in: bind.view(views.pre_in),
            collapsed: bind.view(views.collapsed),
            pre: bind.view(views.pre),
            post: bind.view(views.post),
            comb: bind.view(views.comb),
            live_rows: bind.view(views.live_rows),
        };
        bind.launch_within(op, &b, workspace, Counts::rows(rows))
    }

    pub(crate) fn has_round_expand_mix(&self) -> bool {
        self.round_fused.is_some()
    }
    pub(crate) fn round_expand_mix(
        &self,
        rows: u32,
        views: SinkhornRoundExpandMixBuffers<DeviceBufferSpanV1>,
        workspace: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        let op = self
            .round_fused
            .as_ref()
            .ok_or(kernel_core::contract::Refusal(
                "fused expansion mixer unavailable",
            ))?;
        let mut bind = Bind::default();
        let b = SinkhornRoundExpandMixBuffers {
            x: bind.view(views.x),
            sum: bind.view(views.sum),
            rounded: bind.view(views.rounded),
            post_in: bind.view(views.post_in),
            comb_in: bind.view(views.comb_in),
            expanded: bind.view(views.expanded),
            fn_: bind.view(views.fn_),
            base: bind.view(views.base),
            scale: bind.view(views.scale),
            pre_in: bind.view(views.pre_in),
            collapsed: bind.view(views.collapsed),
            pre: bind.view(views.pre),
            post: bind.view(views.post),
            comb: bind.view(views.comb),
            live_rows: bind.view(views.live_rows),
        };
        bind.launch_within(op, &b, workspace, Counts::rows(rows))
    }

    pub(crate) fn has_finish_norm(&self, phase: Phase) -> bool {
        matches!(phase, Phase::Decode(1 | 8)) && self.finish_norm.is_some()
    }

    pub(crate) fn round_expand_mix_norm(
        &self,
        phase: Phase,
        views: SinkhornRoundNormBuffers<DeviceBufferSpanV1>,
        workspace: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        use kernel_core::allocation::Traversal;
        if !self.has_finish_norm(phase) {
            return Err(kernel_core::contract::Refusal(
                "unqualified HC finish norm phase",
            ));
        }
        let mut bind = Bind::default();
        // Shared collapsed/live roles and the allowed post/comb aliases
        // must retain one logical view when composing the two validators.
        let mut seen = Vec::new();
        let b = views.map(|span| {
            if let Some((_, view)) = seen.iter().find(|(old, _)| *old == span) {
                return *view;
            }
            let view = bind.view(span);
            seen.push((span, view));
            view
        });
        bind.launch_within(
            self.finish_norm.as_ref().unwrap(),
            &b,
            workspace,
            Counts::rows(phase.rows()),
        )
    }

    /// The mixer scratch every rank allocates for the mix-collapse.
    pub(crate) fn scratch(&self) -> MixScratchBytes {
        let bytes = |role: &str| {
            self.mix
                .info()
                .bytes
                .iter()
                .find(|(path, _)| path == role)
                .map(|&(_, bytes)| bytes)
        };
        MixScratchBytes {
            partials: bytes(PARTIALS),
            tickets: bytes(TICKETS),
        }
    }

    fn mix_for_phase(&self, phase: Phase) -> &Prepared<LaggedMixCollapseBuffers<View>> {
        if matches!(phase, Phase::Decode(1 | 8)) {
            &self.decode_mix
        } else {
            &self.mix
        }
    }

    pub(crate) fn mix(
        &self,
        phase: Phase,
        views: &MixViews,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        let mut bind = Bind::default();
        let mut view = |span| bind.view(span);
        let buffers = LaggedMixCollapseBuffers {
            x: view(views.x),
            mixer: MixerWeights {
                fn_: view(views.mixer[0]),
                base: view(views.mixer[1]),
                scale: view(views.mixer[2]),
            },
            pre_in: view(views.pre_in),
            collapsed: view(views.collapsed),
            pre: view(views.pre),
            post: view(views.post),
            comb: view(views.comb),
            scratch: MixScratch {
                partials: views.partials.map(&mut view),
                tickets: views.tickets.map(&mut view),
            },
            live_rows: view(views.live_rows),
        };
        bind.launch(
            self.mix_for_phase(phase),
            &buffers,
            Counts::rows(phase.rows()),
        )
    }

    pub(crate) fn expand(
        &self,
        rows: u32,
        views: &ExpandViews,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        let mut bind = Bind::default();
        let mut view = |span| bind.view(span);
        let buffers = StreamExpandBuffers {
            x: view(views.x),
            y: view(views.y),
            post: view(views.post),
            comb: view(views.comb),
            output: view(views.output),
            live_rows: view(views.live_rows),
        };
        bind.launch(&self.expand, &buffers, Counts::rows(rows))
    }

    pub(crate) fn head(
        &self,
        rows: u32,
        views: &HeadViews,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        let mut bind = Bind::default();
        let mut view = |span| bind.view(span);
        let buffers = WeightedStreamHeadBuffers {
            x: view(views.x),
            pre: view(views.pre),
            output: view(views.output),
            live_rows: view(views.live_rows),
        };
        bind.launch(&self.head, &buffers, Counts::rows(rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::FLASH;
    #[test]
    fn warp_mixer_is_limited_to_qualified_decode_buckets() {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            let ops = HcOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default()).unwrap();
            for rows in [1, 2, 4, 8, 16, 32] {
                for phase in [
                    Phase::Decode(rows),
                    Phase::Prefill { rows },
                    Phase::Verify { rows, items: 1 },
                ] {
                    let selected = ops
                        .mix_for_phase(phase)
                        .info()
                        .symbols
                        .iter()
                        .any(|(s, _)| s.ends_with("hc_mixer_probe_warp_v1"));
                    assert_eq!(selected, matches!(phase, Phase::Decode(1 | 8)), "{phase:?}");
                    assert_eq!(
                        ops.has_finish_norm(phase),
                        matches!(phase, Phase::Decode(1 | 8))
                    );
                }
            }
            assert_eq!(ops.scratch().partials, Some(32 * 15360));
            assert_eq!(ops.scratch().tickets, Some(32 * 4));
        }
    }
}
