//! Quantized chained experts and row-bias routing through their own
//! additive registrations. Legacy grouped/chained selection is unchanged.

use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::{
    Argument, ArgumentList, DType, MoeQuery, NumericalContract, Refusal, Validate,
};
use kernel_core::moe_chained::ChainedMlpProjectionQuery;
use kernel_core::moe_experts::{GatedMlpBuffers, PreparedExperts};
use kernel_core::precision::OperationUse;
use kernel_core::precision::WeightLayout;
use kernel_core::precision::quantizer::{
    BlockQuantization, BlockQuantizeBuffers, BlockQuantizeQuery, QuantizeOutput,
};
use kernel_core::router::{RowBiasRouteBuffers, RowBiasRouterBuffers, RowBiasRouterQuery};

use super::select::{Candidate, choose};
use super::{MAX_CAPACITY, check_plan};
use crate::moe::chained::{RouteKernel, quantized, row_bias};

#[derive(Default)]
pub(super) struct Selection {
    pub routers: Vec<row_bias::Router>,
    pub routes: Vec<row_bias::Route>,
    pub projections: Vec<quantized::Projection>,
}

pub(super) fn projection(
    backend: &dyn Backend,
    entries: &[quantized::Projection],
    op: &ChainedMlpProjectionQuery,
    usage: OperationUse,
    need: Need,
) -> Result<Box<dyn PreparedExperts>, Refused> {
    op.validate()?;
    capacity(need)?;
    let grouped = op.projection.gemm.grouped.expect("routed contract");
    if usage.capacity < need.capacity
        || need
            .capacity
            .checked_mul(grouped.top_k)
            .is_none_or(|r| r > op.projection.gemm.m_capacity)
        || op.projection.logical() != op.projection
    {
        return Err(
            Refusal("quantized chained request needs logical routed capacity for its use").into(),
        );
    }
    let math = |mut op: ChainedMlpProjectionQuery| {
        op.projection.gemm.m_capacity = 1;
        if let Some(g) = &mut op.projection.gemm.grouped {
            g.capacity = kernel_core::contract::RoutedCapacity::TotalRows(1);
        }
        op
    };
    let preferred_tile = if usage.phase == kernel_core::precision::ExecutionPhase::Prefill {
        64
    } else {
        16
    };
    let chosen = choose(
        "quantized chained projection",
        entries
            .iter()
            .filter(|e| math(e.query) == math(*op))
            .map(|e| {
                Candidate::new(e.clone(), &e.symbol, format!("{e:?}"), e.invariance())
                    .capacity(e.tokens_capacity)
                    .rank(u64::from(e.row_tile != preferred_tile))
            }),
        need,
    )?;
    let entry = chosen.entry;
    quantized::Projection::new(entry.query, entry.row_tile, entry.tokens_capacity)?;
    check_plan(&entry.plan(need.capacity)?)?;
    let g = op.projection.gemm;
    let k = u64::from(g.k);
    let e = u64::from(grouped.experts);
    let n = u64::from(g.n);
    let r = u64::from(need.capacity) * u64::from(grouped.top_k);
    let activation_rows = if entry.gate_up() {
        u64::from(need.capacity)
    } else {
        r
    };
    let padded = entry.padded_rows(need.capacity)?;
    let product = |factors: &[u64]| {
        factors
            .iter()
            .try_fold(1_u64, |a, &b| a.checked_mul(b))
            .ok_or(Refusal("quantized chained byte count overflows"))
    };
    let mut info = OpInfo::new(
        format!("sm120.{}[{}]", op.numerical_contract_id()?, entry.symbol),
        chosen.guarantees,
        vec![(entry.symbol.clone(), 0)],
    );
    // Same order as the compute entry's arguments. Down never reads a
    // route-weight view: it has already been applied before its input.
    info.bytes = vec![
        ("activation".into(), product(&[activation_rows, k])?),
        ("weights.codes".into(), product(&[e, n, k / 2])?),
        ("weights.local_scales".into(), product(&[e, n, k / 32])?),
        (
            "activation_scales".into(),
            product(&[activation_rows, k / 32])?,
        ),
        ("routing.routes.offsets".into(), (e + 1) * 4),
        ("routing.routes.sorted_rows".into(), u64::from(padded) * 4),
        ("routing.routes.padded_rows".into(), 4),
        ("live_tokens".into(), 4),
        (
            "output".into(),
            product(&[
                r,
                u64::from(entry.columns()),
                if entry.gate_up() { 2 } else { 4 },
            ])?,
        ),
    ];
    let quantizer = if entry.gate_up() {
        // Each token's routes are contiguous, and each route has whole
        // 32-value groups. Treat them as one wide quantizer row so the
        // shared operation reads the original device token count.
        let width = grouped
            .top_k
            .checked_mul(entry.columns())
            .ok_or(Refusal("quantized chained output row width overflows"))?;
        let quantizer = backend
            .block_quantize(
                &BlockQuantizeQuery {
                    width,
                    input: DType::Bf16,
                    format: BlockQuantization::E4M3_CEIL32,
                    output: QuantizeOutput::Codes,
                },
                need,
            )
            .map_err(|error| match error {
                Refused::Unsupported => Refused::from(Refusal(
                    "quantized chained gate/up requires the shared ceil quantizer registration",
                )),
                Refused::Rule(rule) => Refused::Rule(format!("shared ceil quantizer: {rule}")),
                other => other,
            })?;
        if !quantizer.info().state.is_empty() || !quantizer.info().weight_layouts.is_empty() {
            return Err(
                Refusal("chained output quantization must be stateless and weightless").into(),
            );
        }
        info.bytes
            .push(("routing.weights".into(), product(&[r, 4])?));
        info.bytes.push((
            "scratch.output_codes".into(),
            product(&[r, u64::from(entry.columns())])?,
        ));
        info.bytes.push((
            "scratch.output_scales".into(),
            product(&[r, u64::from(entry.columns()) / 32])?,
        ));
        info.bytes.push((
            "scratch.quantizer_status".into(),
            u64::from(need.capacity) * 4,
        ));
        for (path, bytes) in &quantizer.info().bytes {
            let target = match path.as_str() {
                "input" => "output",
                "output" => "scratch.output_codes",
                "scales" => "scratch.output_scales",
                "live_rows" => "live_tokens",
                "status" => "scratch.quantizer_status",
                _ => {
                    return Err(Refusal("chained output quantizer declared an unknown role").into());
                }
            };
            let (_, size) = info
                .bytes
                .iter_mut()
                .find(|(path, _)| path == target)
                .expect("quantizer role");
            *size = (*size).max(*bytes);
        }
        info.guarantees = info.guarantees.meet(quantizer.info().guarantees);
        info.workspace = quantizer.info().workspace;
        info.symbols
            .extend(quantizer.info().symbols.iter().cloned());
        info.implementation
            .push_str(&format!("+{}", quantizer.info().implementation));
        Some(quantizer)
    } else {
        None
    };
    info.weight_layouts = vec![("weights".into(), WeightLayout::row_major())];
    need.check(&info)?;
    Ok(Box::new(ProjectionOp {
        entry,
        capacity: need.capacity,
        padded,
        info,
        quantizer,
    }))
}

struct ProjectionOp {
    entry: quantized::Projection,
    capacity: u32,
    padded: u32,
    info: OpInfo,
    quantizer: Option<Prepared<BlockQuantizeBuffers<View>>>,
}

impl PreparedExperts for ProjectionOp {
    fn route_tile(&self) -> u32 {
        self.entry.row_tile
    }
    fn padded_rows(&self) -> u32 {
        self.padded
    }
}

impl PreparedOp<GatedMlpBuffers<View>> for ProjectionOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &GatedMlpBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.capacity)?;
        let routing = b
            .routing
            .ok_or(Refusal("quantized chained projection needs routes"))?;
        let views = [
            b.activation,
            b.weights.codes,
            required(b.weights.local_scales)?,
            required(b.activation_scales)?,
            routing.routes.offsets,
            routing.routes.sorted_rows,
            routing.routes.padded_rows,
            b.live_tokens,
            b.output,
        ];
        for (view, (_, bytes)) in views.iter().zip(&self.info.bytes) {
            view.covers(*bytes)?;
        }
        let quantized_output = self
            .quantizer
            .as_ref()
            .map(|_| -> Result<_, Refusal> {
                let codes = required(b.scratch.output_codes)?;
                let scales = required(b.scratch.output_scales)?;
                routing.weights.covers(
                    self.info
                        .bytes_of("routing.weights")
                        .expect("gate/up weights"),
                )?;
                codes.covers(
                    self.info
                        .bytes_of("scratch.output_codes")
                        .expect("gate/up codes"),
                )?;
                scales.covers(
                    self.info
                        .bytes_of("scratch.output_scales")
                        .expect("gate/up scales"),
                )?;
                workspace.covers(self.info.workspace)?;
                let status = required(b.scratch.quantizer_status)?;
                status.covers(kernel_core::precision::quantizer::status_extent(
                    b.scratch.quantizer_status_row_offset,
                    rows,
                )?)?;
                if views.iter().any(|v| v.handle == status.handle)
                    || [
                        codes.handle,
                        scales.handle,
                        routing.weights.handle,
                        workspace.handle,
                    ]
                    .contains(&status.handle)
                {
                    return Err(Refusal("chained quantizer status overlaps another operand"));
                }
                Ok(BlockQuantizeBuffers {
                    status,
                    status_row_offset: b.scratch.quantizer_status_row_offset,
                    input: b.output,
                    output: codes,
                    scales: Some(scales),
                    live_rows: b.live_tokens,
                })
            })
            .transpose()?;
        let arguments = self.entry.arguments([
            views[0].handle,
            views[1].handle,
            views[2].handle,
            views[3].handle,
            views[4].handle,
            views[5].handle,
            views[6].handle,
            views[7].handle,
            if self.entry.gate_up() {
                routing.weights.handle
            } else {
                0
            },
            views[8].handle,
        ])?;
        let mut launches = vec![Launch {
            symbol: self.entry.symbol.clone(),
            rows,
            plan: self.entry.plan(rows)?,
            arguments,
            collective: false,
            persistent: false,
        }];
        if let (Some(quantizer), Some(buffers)) = (&self.quantizer, quantized_output) {
            launches.extend(quantizer.launches(&buffers, workspace, counts)?);
        }
        Ok(launches)
    }
}

fn capacity(need: Need) -> Result<(), Refused> {
    if need.capacity == 0 || need.capacity > MAX_CAPACITY {
        return Err(Refusal("chained op capacity exceeds SM120 bounds").into());
    }
    Ok(())
}

pub(super) fn router(
    entries: &[row_bias::Router],
    op: &RowBiasRouterQuery,
    need: Need,
) -> Result<Prepared<RowBiasRouterBuffers<View>>, Refused> {
    op.validate()?;
    capacity(need)?;
    let chosen = choose(
        "row-bias router",
        entries
            .iter()
            .filter(|e| e.experts == op.experts && e.top_k == op.top_k)
            .map(|e| {
                Candidate::new(e.clone(), e.symbol(), format!("{e:?}"), e.invariance())
                    .capacity(e.capacity)
            }),
        need,
    )?;
    let entry = chosen.entry;
    check_plan(&entry.plan(need.capacity)?)?;
    let mut info = OpInfo::new(
        format!("sm120.{}", op.numerical_contract_id()?),
        chosen.guarantees,
        vec![(entry.symbol(), 0)],
    );
    let (tokens, experts, top_k) = (
        u64::from(need.capacity),
        u64::from(op.experts),
        u64::from(op.top_k),
    );
    info.bytes = vec![
        ("router.logits".into(), tokens * experts * 4),
        ("router.bias".into(), experts * 4),
        ("alternate_bias".into(), experts * 4),
        ("bias_select".into(), tokens),
        ("scores".into(), tokens * experts * 4),
        ("router.ids".into(), tokens * top_k * 4),
        ("router.weights".into(), tokens * top_k * 4),
    ];
    need.check(&info)?;
    Ok(Box::new(RouterOp {
        entry,
        op: *op,
        capacity: need.capacity,
        info,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn route(
    entries: &[row_bias::Route],
    op: &RowBiasRouterQuery,
    routes: &MoeQuery,
    rank: u32,
    route_tile: u32,
    usage: OperationUse,
    need: Need,
) -> Result<Prepared<RowBiasRouteBuffers<View>>, Refused> {
    op.validate()?;
    routes.validate()?;
    capacity(need)?;
    let MoeQuery::LocalRoutes {
        experts,
        local_experts,
        top_k,
        capacity: route_capacity,
    } = *routes
    else {
        return Err(Refusal("row-bias route requires local route metadata").into());
    };
    // Gate/up's shared quantizer covers every top-k slot. Partial expert
    // metadata would leave absent routes unwritten at its BF16 input.
    if local_experts != experts {
        return Err(Refusal(
            "row-bias chained routes require full-expert metadata for gate/up quantization",
        )
        .into());
    }
    if experts != op.experts
        || top_k != op.top_k
        || rank >= experts / local_experts
        || usage.capacity < need.capacity
        || route_capacity < need.capacity
    {
        return Err(Refusal("row-bias route geometry, rank or use differs from its router").into());
    }
    let chosen = choose(
        "row-bias local route",
        entries
            .iter()
            .filter(|e| {
                e.local.experts == experts
                    && e.local.local_experts == local_experts
                    && e.local.top_k == top_k
                    && e.local.pad == route_tile
            })
            .map(|e| {
                Candidate::new(
                    *e,
                    e.local.symbol(e.prefix(), e.local.kernels()[0]),
                    format!("{e:?}"),
                    e.local.invariance(RouteKernel::Decode),
                )
                .capacity(e.local.capacity)
            }),
        need,
    )?;
    let entry = chosen.entry;
    entry.validate()?;
    let symbols = entry
        .entries()?
        .into_iter()
        .map(|e| (e.symbol, 0))
        .collect();
    for &kernel in entry.local.kernels() {
        check_plan(&entry.local.plan(kernel, need.capacity)?)?;
    }
    let mut info = OpInfo::new(
        format!("sm120.{}.local-routes", op.numerical_contract_id()?),
        chosen.guarantees,
        symbols,
    );
    let (tokens, experts, top_k, local) = (
        u64::from(need.capacity),
        u64::from(experts),
        u64::from(top_k),
        u64::from(local_experts),
    );
    info.bytes = vec![
        ("route.logits".into(), tokens * experts * 4),
        ("route.bias".into(), experts * 4),
        ("alternate_bias".into(), experts * 4),
        ("bias_select".into(), tokens),
        ("scores".into(), tokens * experts * 4),
        ("route.ids".into(), tokens * top_k * 4),
        ("route.weights".into(), tokens * top_k * 4),
        ("route.routes.counts".into(), local * 4),
        ("route.routes.offsets".into(), (local + 1) * 4),
        (
            "route.routes.sorted_rows".into(),
            u64::from(entry.local.sorted_rows_capacity()) * 4,
        ),
        ("route.routes.padded_rows".into(), 4),
        ("route.routes.active_experts".into(), (local + 1) * 4),
    ];
    need.check(&info)?;
    Ok(Box::new(RouteOp {
        entry,
        op: *op,
        rank,
        capacity: need.capacity,
        info,
    }))
}

fn rows(counts: Counts, capacity: u32) -> Result<u32, Refusal> {
    if counts.rows == 0 || counts.rows > capacity {
        return Err(Refusal("row-bias launch exceeds prepared rows"));
    }
    Ok(counts.rows)
}

fn pointer(view: View, bytes: u64) -> Result<Argument, Refusal> {
    view.covers(bytes)?;
    Ok(Argument::DeviceBuffer(view.handle))
}

fn required(view: Option<View>) -> Result<View, Refusal> {
    view.ok_or(Refusal(
        "row-bias route needs its bias and metadata buffers",
    ))
}

fn launch(
    symbol: String,
    rows: u32,
    plan: kernel_core::contract::LaunchPlan,
    arguments: Vec<Argument>,
) -> Result<Launch, Refusal> {
    let arguments = ArgumentList(arguments);
    arguments.validate()?;
    Ok(Launch {
        symbol,
        rows,
        plan,
        arguments,
        collective: false,
        persistent: false,
    })
}

struct RouterOp {
    entry: row_bias::Router,
    op: RowBiasRouterQuery,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<RowBiasRouterBuffers<View>> for RouterOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        b: &RowBiasRouterBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.capacity)?;
        let e = u64::from(self.op.experts);
        let r = u64::from(rows);
        let chosen = r * u64::from(self.op.top_k) * 4;
        Ok(vec![launch(
            self.entry.symbol(),
            rows,
            self.entry.plan(rows)?,
            vec![
                pointer(b.router.logits, r * e * 4)?,
                pointer(required(b.router.bias)?, e * 4)?,
                pointer(b.alternate_bias, e * 4)?,
                pointer(b.bias_select, r)?,
                pointer(b.scores, r * e * 4)?,
                pointer(b.router.ids, chosen)?,
                pointer(b.router.weights, chosen)?,
                pointer(b.router.live_rows, 4)?,
                Argument::U32(self.op.scaling_factor.to_bits()),
            ],
        )?])
    }
}

struct RouteOp {
    entry: row_bias::Route,
    op: RowBiasRouterQuery,
    rank: u32,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<RowBiasRouteBuffers<View>> for RouteOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        b: &RowBiasRouteBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.capacity)?;
        let local = self.entry.local;
        let e = u64::from(local.experts);
        let l = u64::from(local.local_experts);
        let r = u64::from(rows);
        let count = r * u64::from(local.top_k) * 4;
        let logits = pointer(b.route.logits, r * e * 4)?;
        let bias = pointer(required(b.route.bias)?, e * 4)?;
        let alternate = pointer(b.alternate_bias, e * 4)?;
        let select = pointer(b.bias_select, r)?;
        let scores = pointer(b.scores, r * e * 4)?;
        let ids = pointer(b.route.ids, count)?;
        let weights = pointer(b.route.weights, count)?;
        let counts = pointer(required(b.route.routes.counts)?, l * 4)?;
        let offsets = pointer(b.route.routes.offsets, (l + 1) * 4)?;
        let sorted = pointer(
            b.route.routes.sorted_rows,
            u64::from(local.sorted_rows_capacity()) * 4,
        )?;
        let padded = pointer(b.route.routes.padded_rows, 4)?;
        let live = pointer(b.route.live_rows, 4)?;
        let active = pointer(required(b.route.routes.active_experts)?, (l + 1) * 4)?;
        let rank = Argument::U32(self.rank);
        let scale = Argument::U32(self.op.scaling_factor.to_bits());
        local
            .kernels()
            .iter()
            .map(|&kernel| {
                let args = match kernel {
                    RouteKernel::Decode => vec![
                        logits.clone(),
                        bias.clone(),
                        alternate.clone(),
                        select.clone(),
                        scores.clone(),
                        ids.clone(),
                        weights.clone(),
                        counts.clone(),
                        offsets.clone(),
                        sorted.clone(),
                        padded.clone(),
                        live.clone(),
                        active.clone(),
                        rank.clone(),
                        scale.clone(),
                    ],
                    RouteKernel::Select => vec![
                        logits.clone(),
                        bias.clone(),
                        alternate.clone(),
                        select.clone(),
                        scores.clone(),
                        ids.clone(),
                        weights.clone(),
                        live.clone(),
                        scale.clone(),
                    ],
                    RouteKernel::Count => {
                        vec![ids.clone(), counts.clone(), live.clone(), rank.clone()]
                    }
                    RouteKernel::Scatter => vec![
                        ids.clone(),
                        counts.clone(),
                        offsets.clone(),
                        sorted.clone(),
                        padded.clone(),
                        live.clone(),
                        active.clone(),
                        rank.clone(),
                    ],
                };
                launch(
                    local.symbol(self.entry.prefix(), kernel),
                    rows,
                    local.plan(kernel, rows)?,
                    args,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Sm120;
    use crate::moe::chained::LocalRoute;
    use kernel_core::backend::Backend;
    use kernel_core::contract::{
        ActivationFormat, DType, Epilogues, GemmQuery, GroupedGemm, RoutedCapacity,
    };
    use kernel_core::gated_mlp::{GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint};
    use kernel_core::moe::{RouteBuffers, RouteMetadata, RouterBuffers};
    use kernel_core::moe_chained::ChainedK;
    use kernel_core::moe_experts::GatedMlpScratch;
    use kernel_core::precision::ExecutionPhase;
    use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
    use kernel_core::properties::Invariance;
    use std::sync::Mutex;

    #[derive(Default)]
    struct QuantizerProbe {
        requests: Mutex<Vec<(BlockQuantizeQuery, Need)>>,
    }

    impl Backend for QuantizerProbe {
        fn name(&self) -> &str {
            "quantizer-probe"
        }

        fn block_quantize(
            &self,
            query: &BlockQuantizeQuery,
            need: Need,
        ) -> Result<Prepared<BlockQuantizeBuffers<View>>, Refused> {
            query.validate()?;
            self.requests.lock().unwrap().push((*query, need));
            let mut info = OpInfo::new(
                "probe.ceil32",
                Invariance::any_rows(need.capacity),
                vec![("probe.ceil32".into(), 0)],
            );
            let elements = u64::from(need.capacity) * u64::from(query.width);
            info.bytes = vec![
                ("input".into(), elements * 2),
                ("output".into(), elements),
                ("scales".into(), elements / 32),
                ("live_rows".into(), 4),
                ("status".into(), u64::from(need.capacity) * 4),
            ];
            info.workspace = 128;
            Ok(Box::new(QuantizerOpProbe {
                info,
                capacity: need.capacity,
            }))
        }
    }

    struct QuantizerOpProbe {
        info: OpInfo,
        capacity: u32,
    }

    impl PreparedOp<BlockQuantizeBuffers<View>> for QuantizerOpProbe {
        fn info(&self) -> &OpInfo {
            &self.info
        }
        fn launches(
            &self,
            b: &BlockQuantizeBuffers<View>,
            workspace: View,
            counts: Counts,
        ) -> Result<Vec<Launch>, Refusal> {
            Ok(vec![launch(
                "probe.ceil32".into(),
                counts.rows,
                kernel_core::contract::LaunchPlan {
                    grid: [self.capacity, 1, 1],
                    block: [128, 1, 1],
                    dynamic_shared_bytes: 0,
                    workspace_bytes: 128,
                    cluster: [1, 1, 1],
                    pdl: true,
                },
                [
                    b.input,
                    b.output,
                    b.scales.unwrap(),
                    b.live_rows,
                    b.status,
                    workspace,
                ]
                .into_iter()
                .map(|v| Argument::DeviceBuffer(v.handle))
                .chain(std::iter::once(Argument::U64(b.status_row_offset)))
                .collect(),
            )?])
        }
    }

    fn gate_up(experts: u32, top_k: u32) -> ChainedMlpProjectionQuery {
        ChainedMlpProjectionQuery {
            projection: GatedMlpProjectionQuery {
                gemm: GemmQuery {
                    m_capacity: 32 * top_k,
                    n: 1152,
                    k: 128,
                    weight: kernel_core::contract::WeightFormat::Mxfp4,
                    activation: ActivationFormat::Mxfp8,
                    output: DType::Bf16,
                    epilogues: Epilogues {
                        swiglu: Some(kernel_core::contract::SwigluColumns::ConcatenatedHalves),
                        output_quantization: Some(kernel_core::contract::OutputQuantization::Mxfp8),
                        ..Epilogues::default()
                    },
                    grouped: Some(GroupedGemm {
                        experts,
                        top_k,
                        capacity: RoutedCapacity::TotalRows(32 * top_k),
                    }),
                },
                projection: GatedMlpProjection::GateUp,
                swiglu_clamp: Some(kernel_core::gated_mlp::SwigluClamp::symmetric(10.0)),
                route_weight: Some(RouteWeightPoint::BeforeDown),
            },
            k: ChainedK::new(128).unwrap(),
        }
    }

    #[test]
    fn gate_up_composes_bf16_ceil_quantization_with_the_device_token_count() {
        for (experts, top_k) in [(384, 6), (128, 3)] {
            let query = gate_up(experts, top_k);
            let entry = quantized::Projection::new(query, 16, 32).unwrap();
            let backend = QuantizerProbe::default();
            let need = Need::rows(32).with_exact(Invariance::any_rows(32));
            let op = projection(
                &backend,
                std::slice::from_ref(&entry),
                &query,
                usage(32),
                need,
            )
            .unwrap();
            assert_eq!(
                *backend.requests.lock().unwrap(),
                vec![(
                    BlockQuantizeQuery {
                        width: top_k * 576,
                        input: DType::Bf16,
                        format: BlockQuantization::E4M3_CEIL32,
                        output: QuantizeOutput::Codes,
                    },
                    need
                )]
            );
            let elements = u64::from(32 * top_k * 576);
            assert_eq!(op.info().bytes_of("activation"), Some(32 * 128));
            assert_eq!(op.info().bytes_of("activation_scales"), Some(32 * 4));
            assert_eq!(op.info().bytes_of("output"), Some(elements * 2));
            assert_eq!(op.info().bytes_of("scratch.output_codes"), Some(elements));
            assert_eq!(
                op.info().bytes_of("scratch.output_scales"),
                Some(elements / 32)
            );
            assert_eq!(op.info().workspace, 128);
            assert_eq!(
                op.info().symbols,
                [(entry.symbol.clone(), 0), ("probe.ceil32".into(), 0)]
            );
            let sized = |path: &str, handle| View {
                handle,
                bytes: op.info().bytes_of(path).unwrap(),
            };
            let buffers = GatedMlpBuffers {
                activation: sized("activation", 1),
                activation_scales: Some(sized("activation_scales", 2)),
                activation_global: None,
                live_tokens: sized("live_tokens", 3),
                output: sized("output", 4),
                weights: QuantizedWeights {
                    codes: sized("weights.codes", 5),
                    local_scales: Some(sized("weights.local_scales", 6)),
                    global_scales: None,
                },
                routing: Some(ProjectionRouting {
                    weights: sized("routing.weights", 7),
                    routes: metadata(),
                }),
                scratch: GatedMlpScratch {
                    quantizer_status: Some(sized("scratch.quantizer_status", 11)),
                    quantizer_status_row_offset: 3,
                    partial: None,
                    output_codes: Some(sized("scratch.output_codes", 8)),
                    output_scales: Some(sized("scratch.output_scales", 9)),
                    unused: View::EMPTY,
                },
            };
            let workspace = View {
                handle: 10,
                bytes: 128,
            };
            let launches = op.launches(&buffers, workspace, Counts::rows(1)).unwrap();
            assert_eq!(launches.len(), 2);
            assert_eq!(launches[0].arguments.0[8], Argument::DeviceBuffer(7));
            assert_eq!(launches[0].arguments.0[9], Argument::DeviceBuffer(4));
            assert_eq!(
                launches[1].arguments.0,
                [4, 8, 9, 3, 11, 10]
                    .map(Argument::DeviceBuffer)
                    .into_iter()
                    .chain(std::iter::once(Argument::U64(3)))
                    .collect::<Vec<_>>()
            );
            assert_eq!(launches[1].rows, 1);
            assert_eq!(op.info().bytes_of("scratch.quantizer_status"), Some(32 * 4));
            for status in [
                None,
                Some(View {
                    handle: 11,
                    bytes: 15,
                }),
                Some(buffers.activation),
            ] {
                let mut bad = buffers;
                bad.scratch.quantizer_status = status;
                assert!(op.launches(&bad, workspace, Counts::rows(1)).is_err());
            }
            let mut bad = buffers;
            bad.scratch.quantizer_status_row_offset = u64::MAX;
            assert!(op.launches(&bad, workspace, Counts::rows(1)).is_err());
            let mut short = buffers;
            short.scratch.output_scales.as_mut().unwrap().bytes -= 1;
            assert!(op.launches(&short, workspace, Counts::rows(1)).is_err());
            short = buffers;
            short.scratch.output_codes = None;
            assert!(op.launches(&short, workspace, Counts::rows(1)).is_err());
            assert!(op.launches(&buffers, View::EMPTY, Counts::rows(1)).is_err());
        }
    }

    #[test]
    fn gate_up_refuses_without_the_shared_quantizer() {
        let query = gate_up(128, 3);
        let entry = quantized::Projection::new(query, 16, 32).unwrap();
        let backend = Sm120::new().with_quantized_chained_projection(entry);
        let Err(error) = backend.chained_mlp_projection(&query, usage(32), Need::rows(32)) else {
            panic!("a compute entry alone cannot quantize its output");
        };
        assert!(error.to_string().contains("shared ceil quantizer"));
    }

    #[test]
    fn token_rows_preserve_each_routes_quantization_groups() {
        for top_k in [3, 6] {
            let values = (0..2 * top_k * 576)
                .map(|i| {
                    let magnitudes = [0.0, 1e-4, 448.03125, 449.0, 896.0625, 0.125];
                    magnitudes[(i / 32) % magnitudes.len()] * if i % 2 == 0 { 1.0 } else { -1.0 }
                })
                .collect::<Vec<_>>();
            let format = BlockQuantization::E4M3_CEIL32;
            let per_route = format.apply(&values, 2 * top_k, 576).unwrap();
            let per_token = format.apply(&values, 2, top_k * 576).unwrap();
            assert_eq!(per_route.codes, per_token.codes);
            assert_eq!(per_route.scales, per_token.scales);
        }
    }

    fn usage(capacity: u32) -> OperationUse {
        OperationUse {
            phase: ExecutionPhase::Decode,
            capacity,
            graph: false,
        }
    }
    fn view() -> View {
        View {
            handle: 1,
            bytes: 1 << 30,
        }
    }
    fn metadata() -> RouteMetadata<View> {
        RouteMetadata {
            counts: Some(view()),
            offsets: view(),
            sorted_rows: view(),
            padded_rows: view(),
            active_experts: Some(view()),
        }
    }

    #[test]
    fn row_bias_router_prepares_both_geometries_and_checks_every_new_role() {
        for (experts, top_k) in [(128, 3), (384, 6)] {
            let backend = [64, 16, 32].into_iter().fold(Sm120::new(), |b, capacity| {
                b.with_row_bias_router(row_bias::Router {
                    experts,
                    top_k,
                    capacity,
                })
            });
            let query = RowBiasRouterQuery {
                experts,
                top_k,
                scaling_factor: 1.5,
            };
            let op = backend.moe_row_bias_router(&query, Need::rows(17)).unwrap();
            assert!(op.info().symbols[0].0.contains("_t32_"));
            let buffers = RowBiasRouterBuffers {
                router: RouterBuffers {
                    logits: view(),
                    bias: Some(view()),
                    ids: view(),
                    weights: view(),
                    live_rows: view(),
                },
                alternate_bias: view(),
                bias_select: view(),
                scores: view(),
            };
            for rows in [1, 7, 17] {
                let launches = op
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                assert_eq!(launches.len(), 1);
                assert_eq!(launches[0].arguments.0.len(), 9);
            }
            assert!(
                op.launches(&buffers, View::EMPTY, Counts::rows(18))
                    .is_err()
            );
            for bad in [
                RowBiasRouterBuffers {
                    alternate_bias: View::EMPTY,
                    ..buffers
                },
                RowBiasRouterBuffers {
                    bias_select: View::EMPTY,
                    ..buffers
                },
                RowBiasRouterBuffers {
                    scores: View::EMPTY,
                    ..buffers
                },
                RowBiasRouterBuffers {
                    router: RouterBuffers {
                        bias: None,
                        ..buffers.router
                    },
                    ..buffers
                },
            ] {
                assert!(op.launches(&bad, View::EMPTY, Counts::rows(1)).is_err());
            }
        }
    }

    #[test]
    fn row_bias_routes_refuse_missing_local_experts_before_gate_up_quantization() {
        // All three selected IDs could be 64, 65 and 66. Rank zero's
        // compute would then write no output for the shared row quantizer.
        let entry = row_bias::Route {
            local: LocalRoute {
                experts: 128,
                local_experts: 64,
                top_k: 3,
                capacity: 32,
                pad: 16,
            },
        };
        let backend = Sm120::new().with_row_bias_route(entry);
        let Err(error) = backend.moe_row_bias_route(
            &RowBiasRouterQuery {
                experts: 128,
                top_k: 3,
                scaling_factor: 1.5,
            },
            &MoeQuery::LocalRoutes {
                experts: 128,
                local_experts: 64,
                top_k: 3,
                capacity: 32,
            },
            0,
            16,
            usage(1),
            Need::rows(1),
        ) else {
            panic!("partial expert metadata leaves gate/up quantizer inputs unwritten");
        };
        assert!(
            error.to_string().contains("full-expert metadata"),
            "{error}"
        );
    }

    #[test]
    fn row_bias_metadata_uses_shared_decode_and_prefill_entries() {
        for (experts, top_k) in [(128, 3), (384, 6)] {
            for (capacity, pad, launches) in [(32, 16, 1), (64, 64, 3)] {
                let entry = row_bias::Route {
                    local: LocalRoute {
                        experts,
                        local_experts: experts,
                        top_k,
                        capacity,
                        pad,
                    },
                };
                let backend = Sm120::new().with_row_bias_route(entry);
                let query = RowBiasRouterQuery {
                    experts,
                    top_k,
                    scaling_factor: 1.5,
                };
                let routes = MoeQuery::LocalRoutes {
                    experts,
                    local_experts: experts,
                    top_k,
                    capacity,
                };
                let op = backend
                    .moe_row_bias_route(
                        &query,
                        &routes,
                        0,
                        pad,
                        usage(capacity),
                        Need::rows(capacity),
                    )
                    .unwrap();
                let b = RowBiasRouteBuffers {
                    route: RouteBuffers {
                        logits: view(),
                        bias: Some(view()),
                        ids: view(),
                        weights: view(),
                        routes: metadata(),
                        tile_counts: None,
                        tile_offsets: None,
                        ticket: None,
                        live_rows: view(),
                    },
                    alternate_bias: view(),
                    bias_select: view(),
                    scores: view(),
                };
                let actual = op.launches(&b, View::EMPTY, Counts::rows(1)).unwrap();
                assert_eq!(actual.len(), launches);
                assert_eq!(
                    actual[0].arguments.0.len(),
                    if capacity == 32 { 15 } else { 9 }
                );
                assert!(
                    backend
                        .moe_row_bias_route(
                            &query,
                            &routes,
                            1,
                            pad,
                            usage(capacity),
                            Need::rows(capacity)
                        )
                        .is_err()
                );
                assert!(
                    backend
                        .moe_row_bias_route(
                            &query,
                            &routes,
                            0,
                            32,
                            usage(capacity),
                            Need::rows(capacity)
                        )
                        .is_err()
                );
                let mut bad = b;
                bad.route.routes.active_experts = None;
                assert!(op.launches(&bad, View::EMPTY, Counts::rows(1)).is_err());
            }
        }
    }

    #[test]
    fn quantized_down_keeps_stored_strides_and_never_reads_route_weights() {
        let query = ChainedMlpProjectionQuery {
            projection: GatedMlpProjectionQuery {
                gemm: GemmQuery {
                    m_capacity: 96,
                    n: 64,
                    k: 576,
                    weight: kernel_core::contract::WeightFormat::Mxfp4,
                    activation: ActivationFormat::Mxfp8,
                    output: DType::F32,
                    epilogues: Epilogues::default(),
                    grouped: Some(GroupedGemm {
                        experts: 128,
                        top_k: 3,
                        capacity: RoutedCapacity::TotalRows(96),
                    }),
                },
                projection: GatedMlpProjection::Down,
                swiglu_clamp: None,
                route_weight: Some(RouteWeightPoint::BeforeDown),
            },
            k: ChainedK::new(576).unwrap(),
        };
        let entry = quantized::Projection::new(query, 16, 32).unwrap();
        let backend = Sm120::new().with_quantized_chained_projection(entry);
        let op = backend
            .chained_mlp_projection(&query, usage(32), Need::rows(32))
            .unwrap();
        assert_eq!(op.info().bytes_of("activation"), Some(96 * 576));
        assert_eq!(op.info().bytes_of("activation_scales"), Some(96 * 18));
        assert_eq!(op.info().bytes_of("weights.codes"), Some(128 * 64 * 288));
        let buffers = GatedMlpBuffers {
            activation: view(),
            activation_scales: Some(view()),
            activation_global: None,
            live_tokens: view(),
            output: view(),
            weights: QuantizedWeights {
                codes: view(),
                local_scales: Some(view()),
                global_scales: None,
            },
            routing: Some(ProjectionRouting {
                weights: View::EMPTY,
                routes: metadata(),
            }),
            scratch: GatedMlpScratch {
                quantizer_status: None,
                quantizer_status_row_offset: 0,
                partial: None,
                output_codes: None,
                output_scales: None,
                unused: View::EMPTY,
            },
        };
        let launches = op.launches(&buffers, View::EMPTY, Counts::rows(1)).unwrap();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].arguments.0[8], Argument::NullDevicePointer);
        let short = GatedMlpBuffers {
            activation_scales: Some(View {
                bytes: 96 * 18 - 1,
                ..view()
            }),
            ..buffers
        };
        assert!(op.launches(&short, View::EMPTY, Counts::rows(1)).is_err());
    }
}
