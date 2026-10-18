//! The MoE ops: each rank's routes, the routed experts', shared expert's
//! and dense MLP's gate/up and down projections, and each rank's routed
//! sum, stated in model math and prepared once for each kind of phase.
//!
//! A kind's ops serve every step of it: prefill steps of up to
//! `PREFILL_TOKENS` rows, decode buckets of up to `DECODE_BATCH` rows and
//! verify steps of up to `DECODE_BATCH * MAX_VERIFY_ROWS` rows. The backend
//! chooses each projection's schedule and row tile per row count and pads
//! the routes to the tile the projection reads them in; the lowering only
//! binds a step's spans.

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::contract::{
    ActivationFormat, DType, Epilogues, GemmQuery, GroupedGemm, MoeQuery, Refusal, RoutedCapacity,
    RouterKind, RowQuery, SwigluColumns, WeightFormat,
};
use kernel_core::gated_mlp::{
    GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint, SwigluClamp,
};
use kernel_core::moe::{RouteBuffers, RouteMetadata, RoutedSumBuffers};
use kernel_core::moe_experts::{GatedMlpBuffers, GatedMlpScratch, PreparedExperts};
use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use model_glm5next_contract::contract::{DECODE_BATCH, MAX_VERIFY_ROWS, PREFILL_TOKENS, Shape};
use model_glm5next_contract::quantization::Layout;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::lower::formats::{Class, weight_format};
use crate::program::Phase;

/// One of the MoE's projections: which MLP, its weights' format and which
/// of its two projections. The routed experts of the MTP layer are their
/// own key where their format differs from the main layers'.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Key {
    pub class: Class,
    pub weight: WeightFormat,
    pub gate_up: bool,
}

/// The ops of one kind of phase: its projections, and each rank's routes
/// padded to the routed gate/up's row tile.
struct Kind {
    projections: Vec<(Key, Box<dyn PreparedExperts>)>,
    routes: Vec<Prepared<RouteBuffers<View>>>,
}

/// The MoE ops of an instance: each kind's (prefill, decode, verify) and
/// each rank's routed sum.
pub(crate) struct MoeOps {
    kinds: [Kind; 3],
    sums: Vec<Prepared<RoutedSumBuffers<View>>>,
}

/// The index of `phase`'s kind in [`MoeOps::kinds`].
fn kind_of(phase: Phase) -> usize {
    match phase {
        Phase::Prefill { .. } => 0,
        Phase::Decode(_) => 1,
        Phase::Verify { .. } => 2,
    }
}

/// The use of each kind's ops: the most rows a step of it carries, decode
/// and verify buckets as graphs.
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

/// The projection of `key` holding up to `capacity` token rows: the
/// clamped SwiGLU gate/up into BF16 rows, or the routed experts' down
/// weighted by their routes into F32 rows; a routed projection's rows are
/// its routes, unpadded.
pub(crate) fn projection_query(shape: Shape, key: Key, capacity: u32) -> GatedMlpProjectionQuery {
    let p = shape.profile;
    let grouped = key.class == Class::Experts;
    let intermediate = match key.class {
        Class::Dense => p.dense_intermediate,
        Class::Shared => p.shared_intermediate,
        Class::Experts => p.expert_intermediate,
    };
    let (projection, n, k) = match key.gate_up {
        true => (GatedMlpProjection::GateUp, 2 * intermediate, p.hidden),
        false => (GatedMlpProjection::Down, p.hidden, intermediate),
    };
    let down = projection == GatedMlpProjection::Down;
    // A routed projection's rows are its routes, `top_k` per token.
    let rows = match grouped {
        true => capacity * p.top_k,
        false => capacity,
    };
    GatedMlpProjectionQuery {
        gemm: GemmQuery {
            m_capacity: rows,
            n,
            k,
            weight: key.weight,
            activation: ActivationFormat::Bf16,
            output: if grouped && down {
                DType::F32
            } else {
                DType::Bf16
            },
            epilogues: Epilogues {
                swiglu: key.gate_up.then_some(SwigluColumns::ConcatenatedHalves),
                weighted_moe_combine: grouped && down,
                ..Epilogues::default()
            },
            grouped: grouped.then_some(GroupedGemm {
                experts: p.local_experts,
                top_k: p.top_k,
                capacity: RoutedCapacity::TotalRows(rows),
            }),
        },
        projection,
        swiglu_clamp: key
            .gate_up
            .then(|| SwigluClamp::symmetric(shape.geometry.shape.swiglu_limit as f32)),
        route_weight: grouped.then_some(RouteWeightPoint::AfterDown),
    }
}

/// The rank argument of every rank's route and routed sum: every rank holds
/// a slice of every expert (`local_experts` is every expert) and routes
/// every token over all of them, as rank 0 of one expert shard.
const ROUTE_RANK: u32 = 0;

/// The router: the sigmoid of each expert's logit, the top-k after the
/// score correction bias, weights normalized and scaled.
fn router(shape: Shape) -> RowQuery {
    let moe = shape.geometry.shape.moe;
    RowQuery::Router {
        experts: moe.experts,
        top_k: shape.profile.top_k,
        kind: RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: moe.routed_scale as f32,
        },
    }
}

/// Every projection an instance of `shape` serving `layout` runs: the
/// dense MLP's and the shared expert's gate/up, the routed experts' gate/up
/// and down, and the MTP layer's routed experts' in their format.
pub(crate) fn keys(shape: Shape, layout: &Layout) -> Vec<Key> {
    let format = |layer, class| weight_format(layout, shape, layer, class);
    let mut keys = Vec::new();
    let mut add = |class, weight, gate_up| {
        let key = Key {
            class,
            weight,
            gate_up,
        };
        if !keys.contains(&key) {
            keys.push(key);
        }
    };
    add(Class::Dense, format(0, Class::Dense), true);
    for layer in [0, shape.layers()] {
        add(Class::Shared, format(layer, Class::Shared), true);
        for gate_up in [true, false] {
            add(Class::Experts, format(layer, Class::Experts), gate_up);
        }
    }
    keys
}

impl MoeOps {
    /// The MoE ops of an instance of `shape` serving `layout` on `backend`.
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: Shape,
        layout: &Layout,
        asks: &Asks,
    ) -> Result<Self, Refused> {
        let ranks = shape.ranks();
        let p = shape.profile;
        let experts = shape.geometry.shape.moe.experts;
        let keys = keys(shape, layout);
        let kind = |usage: OperationUse| -> Result<Kind, Refused> {
            let exact = exact::of_phase(usage.phase, usage.capacity);
            let need = exact::need(usage.capacity, exact);
            // The routes' claims are about the routed operation.
            let route_need = exact::need(usage.capacity, exact.of_routed_operation());
            let projections = keys
                .iter()
                .map(|&key| {
                    let op = projection_query(shape, key, usage.capacity);
                    let what = format!("moe.projection {:?} {key:?}", usage.phase);
                    let prepared = asks.ask(what, need, |need| {
                        backend.gated_mlp_projection(&op, usage, need)
                    })?;
                    Ok((key, prepared))
                })
                .collect::<Result<Vec<_>, Refused>>()?;
            // Every routed projection reads the routes in one tile.
            let mut tiles = projections
                .iter()
                .filter(|(key, _)| key.class == Class::Experts)
                .map(|(_, op)| op.route_tile());
            let tile = tiles.next().ok_or(Refused::Rule(
                "GLM 5.3 Flash routes no expert projection".into(),
            ))?;
            if tiles.any(|other| other != tile) {
                return Err(Refused::Rule(
                    "GLM 5.3 Flash's routed projections read routes in different tiles".into(),
                ));
            }
            let routes = MoeQuery::LocalRoutes {
                experts,
                local_experts: p.local_experts,
                top_k: p.top_k,
                capacity: usage.capacity,
            };
            let routes = (0..ranks)
                .map(|rank| {
                    let what = format!("moe.route {:?} rank {rank}", usage.phase);
                    asks.ask(what, route_need, |need| {
                        backend.moe_route(&router(shape), &routes, ROUTE_RANK, tile, usage, need)
                    })
                })
                .collect::<Result<_, _>>()?;
            Ok(Kind {
                projections,
                routes,
            })
        };
        let [prefill, decode, verify] = USES;
        let sum = MoeQuery::LocalRoutedSum {
            experts,
            local_experts: p.local_experts,
            top_k: p.top_k,
            width: p.hidden,
            output: DType::F32,
            capacity: PREFILL_TOKENS,
        };
        let sums = (0..ranks)
            .map(|rank| {
                asks.ask(
                    format!("moe.routed_sum rank {rank}"),
                    exact::need(PREFILL_TOKENS, exact::every_phase(exact::verify_rows())),
                    |need| backend.moe_routed_sum(&sum, ROUTE_RANK, need),
                )
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            kinds: [kind(prefill)?, kind(decode)?, kind(verify)?],
            sums,
        })
    }

    /// The launches of rank `rank`'s routes of a `phase` step: the router's
    /// selection, then the routes of the rank's experts, as the backend
    /// launches them.
    pub(crate) fn routes(
        &self,
        phase: Phase,
        rank: u32,
        views: &RouteViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self.kinds[kind_of(phase)]
            .routes
            .get(rank as usize)
            .ok_or(Refusal("GLM 5.3 Flash has no routes for the rank"))?;
        let mut bind = Bind::default();
        let buffers = RouteBuffers {
            logits: bind.view(views.logits),
            bias: Some(bind.view(views.bias)),
            ids: bind.view(views.ids),
            weights: bind.view(views.weights),
            routes: views.routes.bind(&mut bind),
            tile_counts: None,
            tile_offsets: None,
            ticket: None,
            live_rows: bind.view(views.live_rows),
        };
        bind.launch(op, &buffers, Counts::rows(phase.rows()))
    }

    /// The launch of the projection of `key` at a `phase` step.
    pub(crate) fn projection(
        &self,
        phase: Phase,
        key: Key,
        views: &ProjectionViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let (_, op) = self.kinds[kind_of(phase)]
            .projections
            .iter()
            .find(|(prepared, _)| *prepared == key)
            .ok_or(Refusal("GLM 5.3 Flash prepared no such MoE projection"))?;
        let mut bind = Bind::default();
        let weights = &views.weights;
        let buffers = GatedMlpBuffers {
            activation: bind.view(views.activation),
            activation_scales: None,
            activation_global: None,
            live_tokens: bind.view(views.live_tokens),
            output: bind.view(views.output),
            weights: QuantizedWeights {
                codes: bind.view(weights.codes),
                local_scales: weights.local_scales.map(|span| bind.view(span)),
                global_scales: weights.global_scales.map(|span| bind.view(span)),
            },
            routing: views.routing.as_ref().map(|routing| ProjectionRouting {
                weights: bind.view(routing.weights),
                routes: routing.routes.bind(&mut bind),
            }),
            // The engines read no argument they are not given.
            scratch: GatedMlpScratch {
                quantizer_status: None,
                quantizer_status_row_offset: 0,
                partial: None,
                output_codes: None,
                output_scales: None,
                unused: bind.view(views.output),
            },
        };
        bind.lowered_experts(op.as_ref(), &buffers, Counts::rows(phase.rows()))
    }

    /// The launch of rank `rank`'s routed sum of a `phase` step: plus
    /// `views.addend` where it binds one.
    pub(crate) fn routed_sum(
        &self,
        phase: Phase,
        rank: u32,
        views: &SumViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self
            .sums
            .get(rank as usize)
            .ok_or(Refusal("GLM 5.3 Flash has no routed sum for the rank"))?;
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

/// A step's spans of a rank's routes: the router logits and the layer's
/// score correction bias in, each token's expert ids and weights and the
/// route metadata out.
pub(crate) struct RouteViews {
    pub logits: DeviceBufferSpanV1,
    pub bias: DeviceBufferSpanV1,
    pub ids: DeviceBufferSpanV1,
    pub weights: DeviceBufferSpanV1,
    pub routes: RouteSpans,
    pub live_rows: DeviceBufferSpanV1,
}

/// A projection's packed weights: codes, the block or local scales and the
/// global scales where its format has them.
pub(crate) struct WeightSpans {
    pub codes: DeviceBufferSpanV1,
    pub local_scales: Option<DeviceBufferSpanV1>,
    pub global_scales: Option<DeviceBufferSpanV1>,
}

/// A routed projection's routes: each route's weight and the metadata.
pub(crate) struct RoutingSpans {
    pub weights: DeviceBufferSpanV1,
    pub routes: RouteSpans,
}

/// A step's spans of one projection.
pub(crate) struct ProjectionViews {
    pub activation: DeviceBufferSpanV1,
    pub weights: WeightSpans,
    pub output: DeviceBufferSpanV1,
    pub live_tokens: DeviceBufferSpanV1,
    /// A routed projection's routes; none for a dense one.
    pub routing: Option<RoutingSpans>,
}

/// A step's spans of a routed sum.
pub(crate) struct SumViews {
    pub routed: DeviceBufferSpanV1,
    pub ids: DeviceBufferSpanV1,
    pub addend: Option<DeviceBufferSpanV1>,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}
