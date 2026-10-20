//! The routed MoE ops (`model_deepseek41_contract::moe`): each layer's
//! input quantized to E4M3 per 32 with a ceil UE8M0 scale, each rank's
//! row-bias routes (sqrt-softplus scores, the text or image bias selecting,
//! the top-k weights normalized and scaled), the routed experts' chained
//! MXFP4 gate/up (clamped SwiGLU, the route weight before the down
//! projection, quantized again) and down into F32 routed rows, and each
//! rank's routed sum, which adds the shared expert's F32 partial where a
//! step binds it. Every rank holds every expert and a slice of each
//! expert's intermediate columns, so each rank routes every token over all
//! of them as rank 0 of one expert shard.
//!
//! The projections and routes are prepared for each kind of phase: a
//! prefill's rows, a decode bucket's and a verify step's.

pub(crate) mod direct;
pub(crate) mod floor;

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::contract::{MoeQuery, Refusal};
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::moe::{RouteBuffers, RouteMetadata, RoutedSumBuffers};
use kernel_core::moe_experts::{GatedMlpBuffers, GatedMlpScratch, PreparedExperts};
use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
use kernel_core::precision::quantizer::BlockQuantizeBuffers;
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::router::RowBiasRouteBuffers;
use model_deepseek41_contract::moe::{self, Experts};
use model_deepseek41_contract::serving::{DECODE_BATCH, MAX_VERIFY_ROWS, PREFILL_TOKENS};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::row_buckets::RowBuckets;
use super::{Asks, Bind, Lowered, exact};
use crate::program::Phase;
use crate::shape::Shape;

/// The rank every route and routed sum is prepared for: each rank holds
/// every expert.
const ROUTE_RANK: u32 = 0;

/// The use of each kind's ops: a prefill's rows eagerly, decode and verify
/// buckets in graphs.
const USES: [OperationUse; 3] = [
    OperationUse {
        phase: ExecutionPhase::Prefill,
        capacity: PREFILL_TOKENS,
        graph: false,
    },
    OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: DECODE_BATCH,
        graph: true,
    },
    OperationUse {
        phase: ExecutionPhase::Verify,
        capacity: DECODE_BATCH * MAX_VERIFY_ROWS,
        graph: true,
    },
];

fn kind_of(phase: Phase) -> usize {
    match phase {
        Phase::Prefill { .. } => 0,
        Phase::Decode(_) => 1,
        Phase::Verify { .. } => 2,
    }
}

/// One kind's ops: the experts' gate/up and down and each rank's routes,
/// padded to the gate/up's route tile.
struct Kind {
    gate_up: Box<dyn PreparedExperts>,
    down: Box<dyn PreparedExperts>,
    routes: Vec<Prepared<RowBiasRouteBuffers<View>>>,
}

/// The routed MoE ops of an instance's backbone experts.
pub(crate) struct MoeOps {
    /// The layer input's quantizer, the gate/up's activation.
    quantize: RowBuckets<BlockQuantizeBuffers<View>>,
    direct: Vec<(ExecutionPhase, u32, direct::Prepared)>,
    kinds: [Kind; 3],
    /// Prepare measured decode/verify ranges separately so phase-wide
    /// capacities cannot hide the 8/64-row streamed entries.
    buckets: Vec<(OperationUse, Kind)>,
    /// Each rank's routed sum of F32 rows.
    sums: Vec<Prepared<RoutedSumBuffers<View>>>,
    floor: Option<floor::Prepared>,
}

/// The route metadata's spans.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RouteSpans {
    pub counts: DeviceBufferSpanV1,
    pub offsets: DeviceBufferSpanV1,
    pub sorted_rows: DeviceBufferSpanV1,
    pub padded_rows: DeviceBufferSpanV1,
    pub active_experts: DeviceBufferSpanV1,
}

impl RouteSpans {
    fn bind(&self, bind: &mut Bind) -> RouteMetadata<View> {
        RouteMetadata {
            counts: Some(bind.view(self.counts)),
            offsets: bind.view(self.offsets),
            sorted_rows: bind.view(self.sorted_rows),
            padded_rows: bind.view(self.padded_rows),
            active_experts: Some(bind.view(self.active_experts)),
        }
    }
}

/// A step's spans of a rank's routes: the router's F32 logits, the
/// layer's text and image biases and each token's bias choice in; the
/// unbiased scores, each token's expert ids and weights and the route
/// metadata out.
pub(crate) struct RouteViews {
    pub logits: DeviceBufferSpanV1,
    pub bias: DeviceBufferSpanV1,
    pub image_bias: DeviceBufferSpanV1,
    pub bias_select: DeviceBufferSpanV1,
    pub scores: DeviceBufferSpanV1,
    pub ids: DeviceBufferSpanV1,
    pub weights: DeviceBufferSpanV1,
    pub routes: RouteSpans,
    pub live_rows: DeviceBufferSpanV1,
}

/// A step's spans of a quantization: BF16 rows in, E4M3 codes and UE8M0
/// scales out.
pub(crate) struct QuantizeViews {
    pub input: DeviceBufferSpanV1,
    pub codes: DeviceBufferSpanV1,
    pub scales: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    /// The phase's row status (`Scratch::row_status`).
    pub status: DeviceBufferSpanV1,
}

/// A step's spans of an expert projection: the quantized activation, the
/// expert bank's MXFP4 codes and scales, the output rows (the gate/up's
/// BF16 SwiGLU rows with their quantized codes and scales, or the down's
/// F32 routed rows) and the routes.
pub(crate) struct ExpertViews {
    pub activation: DeviceBufferSpanV1,
    pub activation_scales: DeviceBufferSpanV1,
    pub codes: DeviceBufferSpanV1,
    pub scales: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub output_quantized: Option<(DeviceBufferSpanV1, DeviceBufferSpanV1)>,
    pub route_weights: DeviceBufferSpanV1,
    pub routes: RouteSpans,
    pub live_tokens: DeviceBufferSpanV1,
    pub workspace: DeviceBufferSpanV1,
    /// The phase's row status (`Scratch::row_status`).
    pub status: DeviceBufferSpanV1,
}

/// A step's spans of a routed sum.
pub(crate) struct SumViews {
    pub routed: DeviceBufferSpanV1,
    pub ids: DeviceBufferSpanV1,
    pub addend: Option<DeviceBufferSpanV1>,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

impl MoeOps {
    /// The routed MoE ops of the declared expert bank on `backend`.
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
        set: Experts,
    ) -> Result<MoeOps, Refused> {
        let g = shape.geometry;
        let ranks = shape.ranks();
        let refused = |error: String| Refused::Rule(error);
        let every = exact::need(PREFILL_TOKENS, exact::every_phase(super::verify_rows()));
        let quantize = RowBuckets::prepare(asks, "moe.quantize", every, |need| {
            backend.block_quantize(&moe::quantizer(g.shape.hidden), need)
        })?;
        let kind = |usage: OperationUse, exact_bucket: bool| -> Result<Kind, Refused> {
            let need = exact::need(usage.capacity, exact::of_phase(usage.phase, usage.capacity));
            let projection =
                |kind: GatedMlpProjection| -> Result<Box<dyn PreparedExperts>, Refused> {
                    let query =
                        moe::projection(g, ranks, set, kind, usage.capacity).map_err(refused)?;
                    let op = backend.chained_mlp_projection(&query, usage, need)?;
                    let op = if exact_bucket {
                        crate::Kernels::prefill_moe_projection(shape, set, kind, usage, op)?
                    } else {
                        op
                    };
                    // Opt-in and within-bound: every prefill kind, so chunking
                    // never changes a row's arithmetic.
                    let op = crate::Kernels::native_moe_prefill(shape, set, kind, usage, op)?;
                    asks.record(
                        format!("moe.experts {kind:?} {:?}", usage.phase),
                        need,
                        op.info(),
                    );
                    Ok(op)
                };
            let gate_up = projection(GatedMlpProjection::GateUp)?;
            let down = projection(GatedMlpProjection::Down)?;
            let tile = gate_up.route_tile();
            if down.route_tile() != tile {
                return Err(Refused::Rule(format!(
                    "{}'s expert projections read routes in different tiles",
                    super::NAME
                )));
            }
            let routes = MoeQuery::LocalRoutes {
                experts: set.count,
                local_experts: set.count,
                top_k: set.top_k,
                capacity: usage.capacity,
            };
            // The routes' claims are about the routed operation.
            let route_need = exact::need(
                usage.capacity,
                exact::of_phase(usage.phase, usage.capacity).of_routed_operation(),
            );
            let routes = (0..ranks)
                .map(|rank| {
                    asks.ask(
                        format!("moe.route {:?} rank {rank}", usage.phase),
                        route_need,
                        |need| {
                            backend.moe_row_bias_route(
                                &moe::router(g, set),
                                &routes,
                                ROUTE_RANK,
                                tile,
                                usage,
                                need,
                            )
                        },
                    )
                })
                .collect::<Result<_, _>>()?;
            Ok(Kind {
                gate_up,
                down,
                routes,
            })
        };
        let sum = MoeQuery::LocalRoutedSum {
            experts: set.count,
            local_experts: set.count,
            top_k: set.top_k,
            width: g.shape.hidden,
            output: kernel_core::contract::DType::F32,
            capacity: PREFILL_TOKENS,
        };
        let sums = (0..ranks)
            .map(|rank| {
                asks.ask(format!("moe.routed_sum rank {rank}"), every, |need| {
                    backend.moe_routed_sum(&sum, ROUTE_RANK, need)
                })
            })
            .collect::<Result<_, _>>()?;
        let floor = crate::Kernels::measured_moe_floor(shape, set)?;
        if let Some(op) = &floor {
            op.record(asks);
        }
        let direct = crate::Kernels::measured_moe_direct(shape, set)?;
        for (phase, rows, op) in &direct {
            op.record(asks, *phase, *rows);
        }
        let [prefill, decode, verify] = USES;
        Ok(MoeOps {
            quantize,
            direct,
            floor,
            kinds: [
                kind(prefill, false)?,
                kind(decode, false)?,
                kind(verify, false)?,
            ],
            buckets: crate::Kernels::moe_buckets(shape)
                .iter()
                .map(|&(phase, capacity)| {
                    let usage = OperationUse {
                        phase,
                        capacity,
                        graph: phase != ExecutionPhase::Prefill,
                    };
                    Ok((usage, kind(usage, true)?))
                })
                .collect::<Result<_, Refused>>()?,
            sums,
        })
    }

    pub(crate) fn floor(&self, phase: Phase) -> Option<&floor::Prepared> {
        if matches!(phase, Phase::Decode(1)) {
            self.floor.as_ref()
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(crate) fn retain_reference_in_test(&mut self) {
        self.direct.clear();
    }

    pub(crate) fn direct(&self, phase: Phase) -> Option<&direct::Prepared> {
        let kind = match phase {
            Phase::Decode(_) => ExecutionPhase::Decode,
            Phase::Verify { .. } => ExecutionPhase::Verify,
            Phase::Prefill { .. } => return None,
        };
        self.direct
            .iter()
            .find(|(p, n, _)| *p == kind && *n == phase.rows())
            .map(|(_, _, op)| op)
    }

    fn kind(&self, phase: Phase) -> &Kind {
        let index = kind_of(phase);
        self.buckets
            .iter()
            .find(|(usage, _)| {
                usage.phase == USES[index].phase
                    && if phase.prefill() {
                        phase.rows() == usage.capacity
                    } else {
                        phase.rows() <= usage.capacity
                    }
            })
            .map_or(&self.kinds[index], |(_, kind)| kind)
    }

    /// The sorted routed rows' bytes: the most any kind's routes write or
    /// its expert projections read (`OpInfo::bytes`), as a projection pads
    /// each expert's rows to its tile.
    pub(crate) fn sorted_rows_bytes(&self) -> u64 {
        let experts = self
            .kinds
            .iter()
            .chain(self.buckets.iter().map(|(_, kind)| kind))
            .flat_map(|kind| [kind.gate_up.info(), kind.down.info()])
            .flat_map(|info| info.bytes.iter())
            .filter(|(path, _)| path == "routing.routes.sorted_rows")
            .map(|&(_, bytes)| bytes)
            .max()
            .unwrap_or(0);
        self.route_bytes("route.routes.sorted_rows").max(experts)
    }

    /// The route metadata bytes the routes of every kind write, by role
    /// (`OpInfo::bytes`): the most any kind lists.
    fn route_bytes(&self, role: &str) -> u64 {
        self.kinds
            .iter()
            .chain(self.buckets.iter().map(|(_, kind)| kind))
            .flat_map(|kind| &kind.routes)
            .flat_map(|op| op.info().bytes.iter())
            .filter(|(path, _)| path == role)
            .map(|&(_, bytes)| bytes)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn quantize(
        &self,
        rows: u32,
        views: &QuantizeViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = BlockQuantizeBuffers {
            input: bind.view(views.input),
            output: bind.view(views.codes),
            scales: Some(bind.view(views.scales)),
            live_rows: bind.view(views.live_rows),
            status: bind.view(views.status),
            status_row_offset: 0,
        };
        bind.launch(self.quantize.at(rows), &buffers, Counts::rows(rows))
    }

    pub(crate) fn routes(
        &self,
        phase: Phase,
        rank: u32,
        views: &RouteViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self
            .kind(phase)
            .routes
            .get(rank as usize)
            .ok_or(Refusal("DeepSeek V4.1 has no routes for the rank"))?;
        let mut bind = Bind::default();
        let buffers = RowBiasRouteBuffers {
            route: RouteBuffers {
                logits: bind.view(views.logits),
                bias: Some(bind.view(views.bias)),
                ids: bind.view(views.ids),
                weights: bind.view(views.weights),
                routes: views.routes.bind(&mut bind),
                tile_counts: None,
                tile_offsets: None,
                ticket: None,
                live_rows: bind.view(views.live_rows),
            },
            alternate_bias: bind.view(views.image_bias),
            bias_select: bind.view(views.bias_select),
            scores: bind.view(views.scores),
        };
        bind.launch(op, &buffers, Counts::rows(phase.rows()))
    }

    /// The experts' gate/up (`gate_up`) or down projection of a `phase`
    /// step.
    pub(crate) fn experts(
        &self,
        phase: Phase,
        gate_up: bool,
        views: &ExpertViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let kind = self.kind(phase);
        let op = if gate_up { &kind.gate_up } else { &kind.down };
        let mut bind = Bind::default();
        // This schedule uses the partial-work role for exact BF16 decoded
        // operands. Its queue occupies the preceding, disjoint workspace bytes.
        let decoded = op
            .info()
            .bytes_of("scratch.partial")
            .map(|bytes| {
                let offset = op
                    .info()
                    .workspace
                    .checked_sub(bytes)
                    .ok_or(Refusal("invalid prefill workspace layout"))?;
                if views.workspace.length < op.info().workspace {
                    return Err(Refusal("prefill workspace is too short"));
                }
                let offset = views
                    .workspace
                    .offset
                    .checked_add(offset)
                    .ok_or(Refusal("prefill scratch offset overflow"))?;
                Ok(bind.view(DeviceBufferSpanV1::new(
                    views.workspace.buffer,
                    offset,
                    bytes,
                )))
            })
            .transpose()?;
        let buffers = GatedMlpBuffers {
            activation: bind.view(views.activation),
            activation_scales: Some(bind.view(views.activation_scales)),
            activation_global: None,
            live_tokens: bind.view(views.live_tokens),
            output: bind.view(views.output),
            weights: QuantizedWeights {
                codes: bind.view(views.codes),
                local_scales: Some(bind.view(views.scales)),
                global_scales: None,
            },
            routing: Some(ProjectionRouting {
                weights: bind.view(views.route_weights),
                routes: views.routes.bind(&mut bind),
            }),
            scratch: GatedMlpScratch {
                // Only the gate/up's output quantizer reports a status.
                quantizer_status: views.output_quantized.map(|_| bind.view(views.status)),
                quantizer_status_row_offset: 0,
                partial: decoded,
                output_codes: views.output_quantized.map(|(codes, _)| bind.view(codes)),
                output_scales: views.output_quantized.map(|(_, scales)| bind.view(scales)),
                unused: bind.view(views.output),
            },
        };
        bind.launch_experts(
            op.as_ref(),
            &buffers,
            views.workspace,
            Counts::rows(phase.rows()),
        )
    }

    pub(crate) fn routed_sum(
        &self,
        phase: Phase,
        rank: u32,
        views: &SumViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self
            .sums
            .get(rank as usize)
            .ok_or(Refusal("DeepSeek V4.1 has no routed sum for the rank"))?;
        let mut bind = Bind::default();
        let buffers = RoutedSumBuffers {
            routed: bind.view(views.routed),
            ids: bind.view(views.ids),
            addend: views.addend.map(|span| bind.view(span)),
            output: bind.view(views.output),
            live_rows: bind.view(views.live_rows),
        };
        bind.launch(op, &buffers, Counts::rows(phase.rows()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::FLASH;

    #[test]
    fn measured_crossover_survives_phase_wide_preparation() {
        crate::sm120::moe_native::with_native(false, check_exact_crossover);
    }

    // Opting out retains the measured exact expert buckets at every phase.
    fn check_exact_crossover() {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            for set in moe::expert_sets(&FLASH) {
                let asks = Asks::default();
                let ops = MoeOps::prepare(kernels.backend.as_ref(), &shape, &asks, set).unwrap();
                let most = asks.asked().iter().map(|a| a.workspace).max().unwrap();
                for phase in [
                    Phase::Decode(1),
                    Phase::Decode(2),
                    Phase::Decode(6),
                    Phase::Decode(8),
                    Phase::Decode(9),
                    Phase::Decode(16),
                    Phase::Decode(32),
                    Phase::Verify { rows: 1, items: 1 },
                    Phase::Verify { rows: 5, items: 1 },
                    Phase::Verify { rows: 6, items: 1 },
                    Phase::Verify { rows: 8, items: 2 },
                    Phase::Verify { rows: 9, items: 2 },
                    Phase::Verify { rows: 32, items: 8 },
                    Phase::Verify { rows: 48, items: 8 },
                    Phase::Verify {
                        rows: 64,
                        items: 16,
                    },
                    Phase::Verify {
                        rows: 65,
                        items: 16,
                    },
                    Phase::Verify {
                        rows: 192,
                        items: 32,
                    },
                    Phase::Prefill { rows: 1 },
                    Phase::Prefill { rows: 128 },
                    Phase::Prefill { rows: 511 },
                    Phase::Prefill { rows: 512 },
                    Phase::Prefill { rows: 513 },
                    Phase::Prefill { rows: 2048 },
                    Phase::Prefill { rows: 4096 },
                    Phase::Prefill { rows: 8191 },
                    Phase::Prefill { rows: 8192 },
                ] {
                    assert_eq!(
                        ops.floor(phase).is_some(),
                        shape.ranks() == 4 && set.count == 128 && matches!(phase, Phase::Decode(1))
                    );
                    let kind = ops.kind(phase);
                    let compact = shape.ranks() == 4
                        && phase == Phase::Decode(1)
                        && set.count == 128
                        && set.top_k == 3;
                    let streamed = !compact
                        && shape.ranks() == 4
                        && phase.rows() <= 64
                        && !matches!(phase, Phase::Prefill { .. });
                    for (op, projection) in [
                        (&kind.gate_up, GatedMlpProjection::GateUp),
                        (&kind.down, GatedMlpProjection::Down),
                    ] {
                        let vector_gate = shape.ranks() == 4
                            && projection == GatedMlpProjection::GateUp
                            && (phase == Phase::Prefill { rows: 8192 }
                                || (set.count == 128 && phase == Phase::Prefill { rows: 4096 }));
                        assert_eq!(
                            op.info().symbols[0].0.ends_with("_prefill_pack_v1"),
                            phase.prefill()
                                && shape.ranks() == 4
                                && [512, 2048, 4096].contains(&phase.rows())
                                && !vector_gate
                                && !(set.count == 128
                                    && projection == GatedMlpProjection::Down
                                    && phase.rows() == 2048),
                            "{set:?} {phase:?} {ranks:?}"
                        );
                        assert_eq!(
                            op.info().symbols[0].0.ends_with("_stream_pack_v1"),
                            streamed,
                            "{set:?} {phase:?} {ranks:?}"
                        );
                        assert_eq!(
                            op.info().symbols[0].0.starts_with("chop_exact_prefill_"),
                            vector_gate
                        );
                        assert!(op.info().workspace <= most);
                        if streamed {
                            let capacity = if phase.rows() <= 8 { 8 } else { 64 };
                            assert!(
                                op.info().symbols[0]
                                    .0
                                    .contains(&format!("_r64_t{capacity}_"))
                            );
                            assert!(op.info().workspace > 0);
                            assert!(
                                op.info()
                                    .symbols
                                    .iter()
                                    .any(|(s, _)| s.ends_with("_stream_merge_v1"))
                            );
                        }
                        if compact {
                            assert!(op.info().symbols[0].0.contains("_r16_t32_"));
                            assert_eq!(op.info().workspace, 0);
                        }
                        let tile = if streamed || !matches!(phase, Phase::Decode(_)) {
                            64
                        } else {
                            16
                        };
                        assert_eq!(op.route_tile(), tile);
                    }
                    if streamed {
                        assert_eq!(
                            kind.routes[0].info().symbols.len(),
                            if phase.rows() <= 8 { 1 } else { 3 }
                        );
                    }
                }
            }
        }
    }
}
