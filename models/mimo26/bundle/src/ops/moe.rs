//! The MoE routes (each token's top-8 experts by the sigmoid-bias router
//! and the route metadata the experts read) and the combine of the routed
//! rows.

use kernel_core::backend::{Backend, Counts, Need, Prepared, Refused, View};
use kernel_core::contract::{
    ActivationFormat, DType, Epilogues, GemmQuery, MoeQuery, Refusal, RouterKind, RowQuery,
    WeightFormat,
};
use kernel_core::gated_mlp::RouteWeightPoint;
use kernel_core::moe::{
    CombineBuffers, ExpertCombineOp, ProjectedRouteBuffers, RouteBuffers, RouteMetadata,
};
use kernel_core::properties::{Invariance, Rows, Ties};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Bind, ByPhase, need, no_workspace, one, usage};
use crate::program_lower::Lowered;
use crate::selection::{MimoPhase as _, Phase};
use crate::switches::Switches;
use crate::target::Target;
use crate::topology::Tp;

type Route = Prepared<RouteBuffers<View>>;

/// The routed rows' tile of the prefill expert entries, which a step that
/// routes as prefill does pads each expert's routes to; the decode entries'
/// is 16.
const PREFILL_ROUTE_TILE: u32 = 64;
const DECODE_ROUTE_TILE: u32 = 16;

/// The launches of a route op a stage of the route takes, in order: the
/// router (or the whole decode route), then the tiled permutation's count,
/// prefix and scan, and scatter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutePart {
    Router,
    Count,
    Scan,
    Scatter,
}

/// The routes of `tp`'s router: each token's top 8 of the 256 experts by
/// sigmoid score plus correction bias, normalized; every expert is local
/// to every rank.
fn router(tp: Tp) -> RowQuery {
    RowQuery::Router {
        experts: tp.geometry.experts(),
        top_k: tp.geometry.top_k(),
        kind: RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: 1.0,
        },
    }
}

fn routes(tp: Tp, capacity: u32) -> MoeQuery {
    MoeQuery::LocalRoutes {
        experts: tp.geometry.experts(),
        local_experts: tp.geometry.experts(),
        top_k: tp.geometry.top_k(),
        capacity,
    }
}

/// The router's logits of `capacity` token rows: the BF16 hidden rows times
/// the BF16 router weight, F32 out, nothing fused.
fn router_projection(tp: Tp, capacity: u32) -> GemmQuery {
    GemmQuery {
        m_capacity: capacity,
        n: tp.geometry.experts(),
        k: tp.geometry.hidden(),
        weight: WeightFormat::Bf16,
        activation: ActivationFormat::Bf16,
        output: DType::F32,
        epilogues: Epilogues::default(),
        grouped: None,
    }
}

/// What a `phase` step needs of its route: [`need`], whose claims for an
/// op of route metadata are about the routed operation, and for exact
/// verify rows the lower expert of equal scores.
fn route_need(phase: Phase, switches: Switches) -> Result<Need, Refusal> {
    let mut need = need(phase, switches)?;
    if need.exact != Invariance::NONE {
        if need.exact.rows != Rows::None && matches!(phase, Phase::Verify(_)) {
            need.exact = need.exact.with_ties(Ties::LowerIndex);
        }
        need.exact = need.exact.of_routed_operation();
    }
    Ok(need)
}

/// Whether a decode or verify `phase` projects its router logits in its
/// route op: the buckets `target`'s fused router serves.
fn projects(target: &dyn Target, phase: Phase) -> Result<bool, Refusal> {
    Ok(target.fused_router(phase.capacity()?))
}

/// The MoE route and combine ops of an instance; every rank launches the
/// same.
#[derive(Default)]
pub(super) struct Moe {
    /// The route of steps that route as prefill does (router and tiled
    /// permutation) and of the 16-row bucket (the decode route).
    routes: ByPhase<Route>,
    /// The route of the buckets that project their router logits in the
    /// route op (`kernels::moe::prefer_fused_router`).
    projected: ByPhase<Prepared<ProjectedRouteBuffers<View>>>,
    /// The combine of steps whose MLP all-reduce does not fold it in.
    combine: ByPhase<Prepared<CombineBuffers<View>>>,
}

impl Moe {
    /// `tp`'s routes and combines on `target`; `folds(phase)` tells the
    /// steps whose MLP all-reduce sums their routed rows.
    pub(super) fn prepare(
        backend: &dyn Backend,
        target: &dyn Target,
        tp: Tp,
        switches: Switches,
        folds: &dyn Fn(Phase) -> Result<bool, Refused>,
    ) -> Result<Self, Refused> {
        let g = tp.geometry;
        let combine = ExpertCombineOp {
            experts: g.experts(),
            top_k: g.top_k(),
            width: g.hidden(),
            route_weight: Some(RouteWeightPoint::AfterDown),
        };
        Ok(Self {
            routes: ByPhase::prepare(|phase| {
                let capacity = phase.capacity()?;
                let (tile, launches) = if phase.prefill_route() {
                    (PREFILL_ROUTE_TILE, 5)
                } else if !projects(target, phase)? {
                    (DECODE_ROUTE_TILE, 1)
                } else {
                    return Ok(None);
                };
                let op = no_workspace(backend.moe_route(
                    &router(tp),
                    &routes(tp, capacity),
                    0,
                    tile,
                    usage(phase)?,
                    route_need(phase, switches)?,
                )?)?;
                // The route's stages take its launches by index.
                if op.info().symbols.len() != launches {
                    return Err(Refused::Rule(
                        "a MiMo route launches other kernels than its stages take".into(),
                    ));
                }
                Ok(Some(op))
            })?,
            projected: ByPhase::prepare(|phase| {
                if phase.prefill_route() || !projects(target, phase)? {
                    return Ok(None);
                }
                let capacity = phase.capacity()?;
                no_workspace(backend.moe_projected_route(
                    &router_projection(tp, capacity),
                    &router(tp),
                    &routes(tp, capacity),
                    0,
                    DECODE_ROUTE_TILE,
                    usage(phase)?,
                    route_need(phase, switches)?,
                )?)
                .map(Some)
            })?,
            combine: ByPhase::prepare(|phase| {
                if folds(phase)? {
                    return Ok(None);
                }
                no_workspace(backend.moe_expert_combine(&combine, need(phase, switches)?)?)
                    .map(Some)
            })?,
        })
    }
}

/// Routes per tile of the tiled permutation, whose per-tile expert counts
/// and offsets MiMo keeps at the start of the shared workspace.
const ROUTES_PER_TILE: u64 = 256;

/// The tiled permutation's tables in `workspace`: tile counts, then tile
/// offsets, each sized for the prefill capacity's routes.
pub fn route_tile_tables(
    tp: Tp,
    workspace: DeviceBufferSpanV1,
) -> Result<[DeviceBufferSpanV1; 2], Refusal> {
    let table_bytes = crate::contract::PREFILL_TOKENS as u64 * tp.geometry.top_k() as u64
        / ROUTES_PER_TILE
        * tp.geometry.experts() as u64
        * 4;
    if workspace.length < 2 * table_bytes {
        return Err(Refusal("MiMo route tile tables do not fit the workspace"));
    }
    Ok([
        DeviceBufferSpanV1::new(workspace.buffer, workspace.offset, table_bytes),
        DeviceBufferSpanV1::new(
            workspace.buffer,
            workspace.offset + table_bytes,
            table_bytes,
        ),
    ])
}

/// The views of a step's routes.
pub struct RouteViews {
    pub router_logits: DeviceBufferSpanV1,
    pub bias: DeviceBufferSpanV1,
    pub ids: DeviceBufferSpanV1,
    pub weights: DeviceBufferSpanV1,
    pub counts: DeviceBufferSpanV1,
    pub offsets: DeviceBufferSpanV1,
    pub sorted_rows: DeviceBufferSpanV1,
    pub padded_rows: DeviceBufferSpanV1,
    pub active_experts: DeviceBufferSpanV1,
    /// The tiled permutation's per-tile expert counts and offsets.
    pub tile_tables: [DeviceBufferSpanV1; 2],
    pub live_tokens: DeviceBufferSpanV1,
}

/// The route roles of `views`, with the tiled permutation's tables when
/// `tiles`.
fn route_buffers(bind: &mut Bind, views: &RouteViews, tiles: bool) -> RouteBuffers<View> {
    RouteBuffers {
        logits: bind.view(views.router_logits),
        bias: Some(bind.view(views.bias)),
        ids: bind.view(views.ids),
        weights: bind.view(views.weights),
        routes: RouteMetadata {
            counts: Some(bind.view(views.counts)),
            offsets: bind.view(views.offsets),
            sorted_rows: bind.view(views.sorted_rows),
            padded_rows: bind.view(views.padded_rows),
            active_experts: Some(bind.view(views.active_experts)),
        },
        tile_counts: tiles.then(|| bind.view(views.tile_tables[0])),
        tile_offsets: tiles.then(|| bind.view(views.tile_tables[1])),
        ticket: None,
        live_rows: bind.view(views.live_tokens),
    }
}

impl super::Ops {
    /// Whether a `phase` step's route (bucket `bucket` of `COMM_BUCKETS`)
    /// projects its router logits itself: the decode and verify buckets that
    /// route on the target's fused router.
    pub fn projects_router(&self, phase: Phase, bucket: usize) -> bool {
        self.moe.projected.get(phase, bucket).is_ok()
    }

    /// `part` of a `phase` step's routes (bucket `bucket` of
    /// `COMM_BUCKETS`) of `live_tokens` rows. The router grids the phase's
    /// row bucket (`MimoPhase::row_grid`), the permutation the phase's
    /// capacity; a decode route launches its router part alone.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_route(
        &self,
        part: RoutePart,
        phase: Phase,
        bucket: usize,
        switches: Switches,
        live_tokens: u32,
        views: &RouteViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self.moe.routes.get(phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = route_buffers(&mut bind, views, true);
        let counts = Counts::rows(phase.row_grid(live_tokens, switches, self.row_buckets)?);
        let mut launches = bind.lower(op.as_ref(), &buffers, counts)?;
        let range = match (launches.len(), part) {
            (1, RoutePart::Router) => 0..1,
            (5, RoutePart::Router) => 0..1,
            (5, RoutePart::Count) => 1..2,
            (5, RoutePart::Scan) => 2..4,
            (5, RoutePart::Scatter) => 4..5,
            _ => return Err(Refusal("a MiMo route stage takes no launch of its op")),
        };
        Ok(launches.drain(range).collect())
    }

    /// The route of a `phase` step (bucket `bucket` of `COMM_BUCKETS`) of
    /// `live_tokens` rows whose route op projects the router logits from
    /// the hidden rows `x` by the router `weight`, with the split partials
    /// and arrival semaphores of its projection. It grids and records the
    /// phase's bucket.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_projected_route(
        &self,
        phase: Phase,
        bucket: usize,
        switches: Switches,
        live_tokens: u32,
        x: DeviceBufferSpanV1,
        weight: DeviceBufferSpanV1,
        [partials, semaphores]: [DeviceBufferSpanV1; 2],
        views: &RouteViews,
    ) -> Result<Lowered, Refusal> {
        let op = self.moe.projected.get(phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = ProjectedRouteBuffers {
            x: bind.view(x),
            weight: bind.view(weight),
            route: route_buffers(&mut bind, views, false),
            partials: bind.view(partials),
            semaphores: bind.view(semaphores),
        };
        let counts = Counts::rows(phase.row_grid(live_tokens, switches, self.row_buckets)?);
        one(bind.lower(op.as_ref(), &buffers, counts)?)
    }

    /// The combine of a `phase` step's routed rows (bucket `bucket` of
    /// `COMM_BUCKETS`), already weighted, into one row per token. It grids
    /// the phase's row bucket, which it records as its rows.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine(
        &self,
        phase: Phase,
        bucket: usize,
        switches: Switches,
        live_tokens: u32,
        routed: DeviceBufferSpanV1,
        weights: DeviceBufferSpanV1,
        output: DeviceBufferSpanV1,
        live_rows: DeviceBufferSpanV1,
    ) -> Result<Lowered, Refusal> {
        let op = self.moe.combine.get(phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = CombineBuffers {
            routed: bind.view(routed),
            weights: Some(bind.view(weights)),
            output: bind.view(output),
            live_rows: bind.view(live_rows),
        };
        let counts = Counts::rows(phase.row_grid(live_tokens, switches, self.row_buckets)?);
        one(bind.lower(op.as_ref(), &buffers, counts)?)
    }
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::super::tests::{any_gathers, phases, resolved};
    use super::*;
    use crate::ops::{Ops, Plan};
    use crate::pipeline::PreparedLaunch;
    use crate::program::Stage;
    use crate::program_lower::launches;
    use crate::row_plan::MoeRowBuffers;

    /// A span of its own buffer, large enough for any route.
    fn span(buffer: u32) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(buffer, 1),
            256,
            1 << 34,
        )
    }

    fn prepared(tp: Tp, switches: Switches) -> Ops {
        Ops::prepare(
            &*crate::sm120::backend(tp, switches, true).unwrap(),
            &Plan {
                tp,
                switches,
                kv: kernel_core::contract::DType::Bf16,
                items: crate::contract::DECODE_ENTRY_ROWS,
                target: &crate::sm120::own_ops::Sm120Own::new(tp),
                state: &|_| [u64::MAX; 3],
                gathers: &any_gathers,
                dflash: None,
                own: crate::ops::OwnPlan::new(
                    &crate::sm120::own_ops::Sm120Own::new(tp),
                    crate::contract::DECODE_ENTRY_ROWS,
                ),
            },
        )
        .unwrap()
    }

    /// Every switch setting that moves a route or combine launch: the
    /// prefill row grids, and exact verify rows (whose buckets take the
    /// fused router).
    fn settings() -> Vec<Switches> {
        let mut settings = Vec::new();
        for capacity_row_grids in [false, true] {
            for exact_verify in [false, true] {
                settings.push(Switches {
                    capacity_row_grids,
                    exact_verify,
                    ..Switches::configured()
                });
            }
        }
        settings
    }

    /// The live row counts compared at a phase of `capacity` rows: every
    /// row grid's least and most rows.
    fn counts(capacity: u32) -> Vec<u32> {
        let mut counts = vec![1, capacity];
        let mut bucket = 1;
        while bucket < capacity {
            counts.extend([bucket, bucket + 1]);
            bucket *= 2;
        }
        counts.sort_unstable();
        counts.dedup();
        counts
    }

    fn buffers() -> MoeRowBuffers {
        MoeRowBuffers {
            router_logits: span(1),
            router_bias: span(2),
            topk_ids: span(3),
            topk_weights: span(4),
            counts: span(5),
            expert_offsets: span(6),
            sorted_rows: span(7),
            padded_rows: span(8),
            activation: span(9),
            gathered: span(10),
            routed_output: span(11),
            combined: span(12),
            live_tokens: span(13),
            workspace: span(14),
        }
    }

    fn views(tp: Tp) -> RouteViews {
        let b = buffers();
        RouteViews {
            router_logits: b.router_logits,
            bias: b.router_bias,
            ids: b.topk_ids,
            weights: b.topk_weights,
            counts: b.counts,
            offsets: b.expert_offsets,
            sorted_rows: b.sorted_rows,
            padded_rows: b.padded_rows,
            active_experts: span(15),
            tile_tables: route_tile_tables(tp, b.workspace).unwrap(),
            live_tokens: b.live_tokens,
        }
    }

    fn compare(lowered: Vec<Lowered>, planned: Vec<PreparedLaunch>, what: &str) {
        assert_eq!(resolved(&launches(lowered)), resolved(&planned), "{what}");
    }

    /// The prepared routes and combines launch what MiMo's row builders
    /// (`row_plan::moe_row`) and decode route planned: the router and
    /// tiled permutation stages of every step that routes as prefill does,
    /// the decode route of the 16-row bucket (at its rows and fewer, as an
    /// eager step runs), the projected route of the fused router's buckets
    /// (likewise), and the combine of every step that does not fold it, in
    /// every setting that moves them, at TP2 and TP4.
    #[test]
    fn routes_and_combines_launch_what_the_builders_planned() {
        let mut compared = 0;
        for tp in Tp::SUPPORTED {
            let views = views(tp);
            for switches in settings() {
                let ops = prepared(tp, switches);
                for (phase, bucket) in phases() {
                    let capacity = phase.capacity().unwrap();
                    let route =
                        |part, live| ops.moe_route(part, phase, bucket, switches, live, &views);
                    if phase.prefill_route() {
                        for live in counts(capacity) {
                            for (part, stage) in [
                                (RoutePart::Router, Stage::RouterTop8(1)),
                                (RoutePart::Count, Stage::PermuteCount(1)),
                                (RoutePart::Scan, Stage::PermuteScan(1)),
                                (RoutePart::Scatter, Stage::PermuteScatter(1)),
                            ] {
                                let planned = crate::row_plan::moe_row(
                                    stage,
                                    phase,
                                    switches,
                                    live,
                                    tp,
                                    buffers(),
                                )
                                .unwrap();
                                compare(
                                    route(part, live).unwrap(),
                                    planned,
                                    &format!("{tp:?} {switches:?} {phase:?} {live} {stage:?}"),
                                );
                                compared += 1;
                            }
                        }
                    } else if capacity == DECODE_ROUTE_TILE {
                        for live in [1, 9, 15, 16] {
                            let phase = match phase {
                                Phase::Decode(_) => Phase::Decode(live),
                                _ => Phase::Verify(live),
                            };
                            let b = buffers();
                            let planned = crate::sm120::reference::decode_route(
                                tp,
                                capacity,
                                live,
                                [
                                    b.router_logits,
                                    b.router_bias,
                                    b.topk_ids,
                                    b.topk_weights,
                                    b.counts,
                                    b.expert_offsets,
                                    b.sorted_rows,
                                    b.padded_rows,
                                    b.live_tokens,
                                    views.active_experts,
                                ],
                            )
                            .unwrap();
                            let lowered = ops
                                .moe_route(RoutePart::Router, phase, bucket, switches, live, &views)
                                .unwrap();
                            compare(lowered, vec![planned], &format!("{tp:?} {phase:?}"));
                            for part in [RoutePart::Count, RoutePart::Scan, RoutePart::Scatter] {
                                assert!(route(part, live).is_err());
                            }
                            compared += 1;
                        }
                    } else {
                        assert!(route(RoutePart::Router, capacity).is_err());
                        // The fused router's buckets project their logits.
                        for live in (1..=capacity).filter(|&live| live > capacity / 2) {
                            let phase = match phase {
                                Phase::Decode(_) => Phase::Decode(live),
                                _ => Phase::Verify(live),
                            };
                            let b = buffers();
                            let projection = [span(16), b.activation, b.workspace, span(17)];
                            let planned = crate::sm120::reference::fused_route(
                                tp,
                                capacity,
                                live,
                                projection,
                                [
                                    b.router_logits,
                                    b.router_bias,
                                    b.topk_ids,
                                    b.topk_weights,
                                    b.counts,
                                    b.expert_offsets,
                                    b.sorted_rows,
                                    b.padded_rows,
                                    b.live_tokens,
                                    views.active_experts,
                                ],
                            )
                            .unwrap();
                            let lowered = ops
                                .moe_projected_route(
                                    phase,
                                    bucket,
                                    switches,
                                    live,
                                    b.activation,
                                    span(16),
                                    [b.workspace, span(17)],
                                    &views,
                                )
                                .unwrap();
                            compare(vec![lowered], vec![planned], &format!("{tp:?} {phase:?}"));
                            compared += 1;
                        }
                    }
                    if !matches!(phase, Phase::Prefill) && phase.prefill_route() {
                        assert!(
                            ops.moe_projected_route(
                                phase,
                                bucket,
                                switches,
                                1,
                                span(9),
                                span(16),
                                [span(14), span(17)],
                                &views
                            )
                            .is_err()
                        );
                    }
                    let combine = |live| {
                        let b = buffers();
                        ops.moe_combine(
                            phase,
                            bucket,
                            switches,
                            live,
                            b.routed_output,
                            b.topk_weights,
                            b.combined,
                            b.live_tokens,
                        )
                    };
                    if ops.folds_combine(phase, bucket) {
                        assert!(combine(capacity).is_err());
                        continue;
                    }
                    for live in counts(capacity) {
                        let planned = crate::row_plan::moe_row(
                            Stage::CombineExperts(1),
                            phase,
                            switches,
                            live,
                            tp,
                            buffers(),
                        )
                        .unwrap();
                        compare(
                            vec![combine(live).unwrap()],
                            planned,
                            &format!("{tp:?} {switches:?} {phase:?} {live} combine"),
                        );
                        compared += 1;
                    }
                }
            }
        }
        assert!(compared > 1000, "{compared}");
    }
}
