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
    quantize: Prepared<BlockQuantizeBuffers<View>>,
    kinds: [Kind; 3],
    /// Each rank's routed sum of F32 rows.
    sums: Vec<Prepared<RoutedSumBuffers<View>>>,
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
        let quantize = asks.ask("moe.quantize", every, |need| {
            backend.block_quantize(&moe::quantizer(g.shape.hidden), need)
        })?;
        let kind = |usage: OperationUse| -> Result<Kind, Refused> {
            let need = exact::need(usage.capacity, exact::of_phase(usage.phase, usage.capacity));
            let projection =
                |kind: GatedMlpProjection| -> Result<Box<dyn PreparedExperts>, Refused> {
                    let query =
                        moe::projection(g, ranks, set, kind, usage.capacity).map_err(refused)?;
                    let op = backend.chained_mlp_projection(&query, usage, need)?;
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
        let [prefill, decode, verify] = USES;
        Ok(MoeOps {
            quantize,
            kinds: [kind(prefill)?, kind(decode)?, kind(verify)?],
            sums,
        })
    }

    /// The sorted routed rows' bytes: the most any kind's routes write or
    /// its expert projections read (`OpInfo::bytes`), as a projection pads
    /// each expert's rows to its tile.
    pub(crate) fn sorted_rows_bytes(&self) -> u64 {
        let experts = self
            .kinds
            .iter()
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
        bind.launch(&self.quantize, &buffers, Counts::rows(rows))
    }

    pub(crate) fn routes(
        &self,
        phase: Phase,
        rank: u32,
        views: &RouteViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self.kinds[kind_of(phase)]
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
        let kind = &self.kinds[kind_of(phase)];
        let op = if gate_up { &kind.gate_up } else { &kind.down };
        let mut bind = Bind::default();
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
                partial: None,
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
