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
    prefill_mix: Prepared<LaggedMixCollapseBuffers<View>>,
    decode_mix: Prepared<LaggedMixCollapseBuffers<View>>,
    expand: Prepared<StreamExpandBuffers<View>>,
    head: Prepared<WeightedStreamHeadBuffers<View>>,
    fused: Option<Prepared<SinkhornExpandMixBuffers<View>>>,
    /// The measured prefill rows' expansion and next mixer in one launch.
    prefill_fused: Option<Prepared<SinkhornExpandMixBuffers<View>>>,
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
            prefill_mix: asks.ask("hc.prefill_mix", need, |need| {
                backend.sinkhorn_lagged_prefill(&hc::mix_collapse(g, need.capacity), need)
            })?,
            decode_mix: {
                let need = exact::need(32, exact::every_phase(32));
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
            prefill_fused: match asks.ask("hc.prefill_expand_mix", need, |need| {
                backend.sinkhorn_expand_mix(&SinkhornExpandMixQuery { mix: hc::mix(g) }, need)
            }) {
                Ok(op) => Some(op),
                Err(Refused::Unsupported) => None,
                Err(e) => return Err(e),
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

    /// The prefill rows the backend's expansion mixer launches: it refuses
    /// every row count it has not measured, so ask it with distinct views.
    pub(crate) fn has_prefill_expand_mix(&self, phase: Phase) -> bool {
        let Some(op) = self.prefill_fused.as_ref().filter(|_| phase.prefill()) else {
            return false;
        };
        let view = |handle| View {
            handle,
            bytes: u64::MAX / 2,
        };
        let probe = SinkhornExpandMixBuffers {
            x: view(1),
            y: view(2),
            post_in: view(3),
            comb_in: view(4),
            expanded: view(5),
            fn_: view(6),
            base: view(7),
            scale: view(8),
            pre_in: view(9),
            collapsed: view(10),
            pre: view(11),
            post: view(12),
            comb: view(13),
            live_rows: view(14),
        };
        op.launches(&probe, View::EMPTY, Counts::rows(phase.rows()))
            .is_ok()
    }
    pub(crate) fn prefill_expand_mix(
        &self,
        rows: u32,
        views: SinkhornExpandMixBuffers<DeviceBufferSpanV1>,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        let op = self
            .prefill_fused
            .as_ref()
            .ok_or(kernel_core::contract::Refusal(
                "prefill expansion mixer unavailable",
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
        bind.launch(op, &b, Counts::rows(rows))
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
        matches!(
            phase,
            Phase::Decode(1 | 8 | 32) | Phase::Verify { rows: 6 | 32, .. }
        ) && self.finish_norm.is_some()
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
        // A one-launch finish/norm boundary elects its tail with the same
        // zeroed tickets; every use runs on the main stream and leaves them
        // zeroed, so one allocation covers both.
        let tail = self.finish_norm.as_ref().and_then(|op| {
            op.info()
                .bytes
                .iter()
                .find(|(path, _)| path == TICKETS)
                .map(|&(_, bytes)| bytes)
        });
        MixScratchBytes {
            partials: bytes(PARTIALS),
            tickets: bytes(TICKETS).max(tail),
        }
    }

    fn mix_for_phase(&self, phase: Phase) -> &Prepared<LaggedMixCollapseBuffers<View>> {
        if phase.prefill() {
            &self.prefill_mix
        } else if matches!(
            phase,
            Phase::Decode(1 | 8) | Phase::Verify { rows: 6 | 32, .. }
        ) {
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
    fn prefill_mixer_selects_only_measured_buckets_and_keeps_operands() {
        let view = |handle| View {
            handle,
            bytes: u64::MAX / 2,
        };
        let buffers = LaggedMixCollapseBuffers {
            x: view(1),
            mixer: MixerWeights {
                fn_: view(2),
                base: view(3),
                scale: view(4),
            },
            pre_in: view(5),
            collapsed: view(6),
            pre: view(7),
            post: view(8),
            comb: view(9),
            scratch: MixScratch {
                partials: Some(view(11)),
                tickets: Some(view(12)),
            },
            live_rows: View {
                bytes: 4,
                ..view(10)
            },
        };
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            let ops = HcOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default()).unwrap();
            for rows in [
                33, 63, 64, 65, 191, 192, 193, 320, 2239, 2240, 2241, 2495, 2496, 2497, 2623, 2624,
                2625, 8150, 8191, 8192,
            ] {
                let old = ops
                    .mix
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                for phase in [Phase::Prefill { rows }, Phase::Verify { rows, items: 1 }] {
                    let new = ops
                        .mix_for_phase(phase)
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap();
                    let chains = if phase.prefill() {
                        match rows {
                            64 | 192 => Some(8),
                            2240 | 2496 | 2624 | 8192 => Some(4),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if let Some(chains) = chains {
                        assert_eq!(new.len(), 1);
                        assert_eq!(new[0].arguments, old[0].arguments);
                        assert_eq!(new[0].rows, old[0].rows);
                        assert_eq!(
                            new[0].symbol,
                            format!(
                                "chop_sinkhorn_hc_deepseek41_prefill_mix_pipeline_r8_c{chains}_v1"
                            )
                        );
                        assert_eq!(new[0].plan.grid, [rows.div_ceil(8), 1, 1]);
                        assert_eq!(new[0].plan.block, [256, 1, 1]);
                        assert_eq!(new[0].plan.dynamic_shared_bytes, 0);
                    } else {
                        assert_eq!(new, old, "{phase:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn prefill_expand_mix_serves_only_measured_flash_rows() {
        let view = |handle| View {
            handle,
            bytes: u64::MAX / 2,
        };
        let buffers = SinkhornExpandMixBuffers {
            x: view(1),
            y: view(2),
            post_in: view(3),
            comb_in: view(4),
            expanded: view(5),
            fn_: view(6),
            base: view(7),
            scale: view(8),
            pre_in: view(9),
            collapsed: view(10),
            pre: view(11),
            post: view(12),
            comb: view(13),
            live_rows: View {
                bytes: 4,
                ..view(14)
            },
        };
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            let ops = HcOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default()).unwrap();
            let op = ops.prefill_fused.as_ref().unwrap();
            assert_eq!(op.info().workspace, 0);
            for rows in [1, 63, 64, 65, 192, 320, 2496, 4096, 8191, 8192] {
                let tile = match rows {
                    64 | 320 => Some("r4_c4"),
                    8192 => Some("r8_c8"),
                    _ => None,
                };
                for phase in [
                    Phase::Prefill { rows },
                    Phase::Verify { rows, items: 1 },
                    Phase::Decode(rows),
                ] {
                    assert_eq!(
                        ops.has_prefill_expand_mix(phase),
                        phase.prefill() && tile.is_some()
                    );
                }
                let launches = op.launches(&buffers, View::EMPTY, Counts::rows(rows));
                let Some(tile) = tile else {
                    assert!(launches.is_err(), "{rows}");
                    continue;
                };
                let launches = launches.unwrap();
                assert_eq!(launches.len(), 1);
                assert_eq!(
                    launches[0].symbol,
                    format!("chop_deepseek41_hc_prefill_expand_mix_{tile}_v1")
                );
                assert_eq!(launches[0].arguments.0.len(), 14);
                assert_eq!(launches[0].plan.block, [256, 1, 1]);
            }
        }
    }

    #[test]
    fn verify_tail_selection_does_not_change_standalone_producer_buckets() {
        use kernel_core::rmsnorm_quantize::RmsNormQuantizeBuffers;
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let kernels = crate::Kernels::new(&shape).unwrap();
        let ops = HcOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default()).unwrap();
        let tail = ops.finish_norm.as_ref().unwrap();
        let spans: std::collections::BTreeMap<_, _> = tail
            .info()
            .bytes
            .iter()
            .enumerate()
            .map(|(i, (name, bytes))| {
                (
                    name.as_str(),
                    View {
                        handle: i as u64 + 1,
                        bytes: *bytes,
                    },
                )
            })
            .collect();
        let v = |name: &str| spans[name];
        let hc = SinkhornRoundExpandMixBuffers {
            x: v("hc.x"),
            sum: v("hc.sum"),
            rounded: v("hc.rounded"),
            post_in: v("hc.post_in"),
            comb_in: v("hc.comb_in"),
            expanded: v("hc.expanded"),
            fn_: v("hc.fn_"),
            base: v("hc.base"),
            scale: v("hc.scale"),
            pre_in: v("hc.pre_in"),
            collapsed: v("hc.collapsed"),
            pre: v("hc.pre"),
            post: v("hc.post"),
            comb: v("hc.comb"),
            live_rows: v("hc.live_rows"),
        };
        let workspace = View {
            handle: 100,
            bytes: tail.info().workspace,
        };
        for rows in [1, 2, 4, 6, 8, 16, 32] {
            let standalone = ops
                .round_fused
                .as_ref()
                .unwrap()
                .launches(&hc, workspace, Counts::rows(rows))
                .unwrap();
            let expected = match rows {
                1 => "chop_deepseek41_hc_round_chain_w8_v1",
                8 => "chop_deepseek41_hc_round_chain_w4_v1",
                _ => "chop_deepseek41_hc_round_expand_mix_partials_v1",
            };
            assert_eq!(standalone[0].symbol, expected);
            for (widened, tickets) in [
                None,
                Some(View {
                    handle: 101,
                    bytes: 32 * 5120 * 4,
                }),
            ]
            .into_iter()
            .flat_map(|w| [(w, None), (w, Some(v(TICKETS)))])
            {
                let buffers = SinkhornRoundNormBuffers {
                    hc,
                    norm: RmsNormQuantizeBuffers {
                        input: hc.collapsed,
                        weight: v("norm.weight"),
                        normed: v("norm.normed"),
                        widened,
                        codes: v("norm.codes"),
                        scales: v("norm.scales"),
                        live_rows: hc.live_rows,
                        status: v("norm.status"),
                        status_row_offset: 0,
                    },
                    tickets,
                };
                let chain = tail.launches(&buffers, workspace, Counts::rows(rows));
                // Measured C1 and C8 buckets are one launch and need zeroed tickets.
                if let Some(&(_, warps)) = crate::sm120::hc_finish_norm::COMPLETION_BUCKETS
                    .iter()
                    .find(|(bucket, _)| *bucket == rows)
                {
                    let Some(tickets) = tickets else {
                        assert!(chain.is_err());
                        continue;
                    };
                    let chain = chain.unwrap();
                    let completion = crate::sm120::hc_completion::served();
                    assert_eq!(chain.len(), 1);
                    assert_eq!(chain[0].symbol, completion.symbol(warps));
                    assert_eq!(chain[0].plan, completion.plan(rows, rows, warps).unwrap());
                    assert_eq!(
                        chain[0].arguments.0.last(),
                        Some(&kernel_core::contract::Argument::DeviceBuffer(
                            tickets.handle
                        ))
                    );
                    continue;
                }
                let chain = chain.unwrap();
                if matches!(rows, 6 | 32) {
                    assert_eq!(chain.len(), 2);
                    assert!(chain[1].symbol.ends_with("hc_finish_norm_w32_v1"));
                    assert!(chain[0].symbol.ends_with(if rows == 32 {
                        "hc_round_rows_r2_v1"
                    } else {
                        "hc_round_chain_w4_v1"
                    }));
                } else {
                    assert_eq!(chain.len(), 3);
                    assert_eq!(chain[0].symbol, standalone[0].symbol);
                }
            }
        }
    }
    #[test]
    fn warp_mixer_is_limited_to_qualified_decode_and_verify_buckets() {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            let ops = HcOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default()).unwrap();
            for rows in [1, 2, 4, 6, 8, 16, 32] {
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
                    let qualified = matches!(
                        phase,
                        Phase::Decode(1 | 8) | Phase::Verify { rows: 6 | 32, .. }
                    );
                    assert_eq!(selected, qualified, "{phase:?}");
                    assert_eq!(
                        ops.has_finish_norm(phase),
                        qualified || matches!(phase, Phase::Decode(32))
                    );
                }
            }
            assert_eq!(ops.scratch().partials, Some(32 * 15360));
            assert_eq!(ops.scratch().tickets, Some(32 * 4));
        }
    }
}
