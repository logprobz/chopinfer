//! Select the measured gate/up kernel without changing routing or down.
use super::moe_pipeline_entries as family;
use super::moe_pipeline_vector as vector;
use kernel_core::backend::{Counts, Launch, OpInfo, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::moe_experts::{GatedMlpBuffers, PreparedExperts};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use model_deepseek41_contract::{
    geometry::{FLASH, Geometry},
    moe::Experts,
};

/// SM120 TP4, 250 W, 2026-10-19: vector decoded128 gate/up is bitwise.
/// Both-device minima over balanced and concentrated routes, against serving:
/// E384/8192 1.133x, E128/8192 1.110x, E128/4096 1.164x.
/// The measured baseline uses scalar decoded128 at E384/8192 and grouped E128/4096.
/// No down candidate wins by 3% in both distributions on both devices.
pub fn selected(
    g: &Geometry,
    ranks: u32,
    set: Experts,
    kind: GatedMlpProjection,
    usage: OperationUse,
) -> bool {
    g.id == FLASH.id
        && ranks == 4
        && matches!(
            (set.count, set.top_k, usage.capacity),
            (384, 6, 8192) | (128, 3, 8192 | 4096)
        )
        && kind == GatedMlpProjection::GateUp
        && usage.phase == ExecutionPhase::Prefill
        && !usage.graph
}
pub fn prepare(
    g: &Geometry,
    ranks: u32,
    set: Experts,
    kind: GatedMlpProjection,
    usage: OperationUse,
    base: Box<dyn PreparedExperts>,
) -> Result<Box<dyn PreparedExperts>, Refused> {
    if !selected(g, ranks, set, kind, usage) {
        return Ok(base);
    }
    let entry = family::projection(set.count, true);
    let baseline_symbols = if usage.capacity == 4096 {
        let grouped = entry.clone().with_prefill_tiles().map_err(Refused::from)?;
        kernels_sm120::moe::chained::prefill::STAGES
            .into_iter()
            .map(|stage| kernels_sm120::moe::chained::prefill::symbol(&grouped, stage))
            .collect::<Vec<_>>()
    } else {
        vec![entry.symbol.clone()]
    };
    if base.route_tile() != 64
        || base.info().symbols.len() <= baseline_symbols.len()
        || !base
            .info()
            .symbols
            .iter()
            .take(baseline_symbols.len())
            .zip(&baseline_symbols)
            .all(|((actual, _), suffix)| actual.ends_with(suffix))
    {
        return Err(Refusal("qualified prefill gate requires its measured baseline").into());
    }
    let baseline_symbols: Vec<_> = base
        .info()
        .symbols
        .iter()
        .take(baseline_symbols.len())
        .map(|(s, _)| s.clone())
        .collect();
    let variant = family::VARIANTS
        .into_iter()
        .find(|v| v.name == "decoded128")
        .unwrap();
    let decoded = u64::from(usage.capacity) * u64::from(entry.query.projection.gemm.k) * 2;
    let queue = family::workspace_bytes(&entry, variant);
    let mut info = base.info().clone();
    info.implementation
        .push_str("+decoded128_vector_prefill_v1");
    info.workspace = queue + decoded;
    for (name, bytes) in [
        ("scratch.partial".into(), decoded),
        ("routing.routes.counts".into(), u64::from(set.count) * 4),
        (
            "routing.routes.active_experts".into(),
            u64::from(set.count + 1) * 4,
        ),
    ] {
        if let Some(existing) = info.bytes.iter_mut().find(|(key, _)| key == &name) {
            existing.1 = bytes;
        } else {
            info.bytes.push((name, bytes));
        }
    }
    let launches = family::launches(
        &entry,
        variant,
        usage.capacity,
        family::Buffers {
            h: [1; 10],
            counts: 1,
            active: 1,
            scratch: 1,
            decoded: 1,
            stamps: 0,
        },
        false,
        false,
    )
    .map_err(Refused::Rule)?;
    info.symbols.splice(
        ..baseline_symbols.len(),
        launches
            .into_iter()
            .map(|l| (vector::symbol(&l.symbol), l.plan.dynamic_shared_bytes)),
    );
    Ok(Box::new(Prepared {
        base,
        entry,
        variant,
        info,
        capacity: usage.capacity,
        baseline_symbols,
    }))
}
struct Prepared {
    base: Box<dyn PreparedExperts>,
    entry: kernels_sm120::moe::chained::quantized::Projection,
    variant: family::Variant,
    info: OpInfo,
    capacity: u32,
    baseline_symbols: Vec<String>,
}
impl PreparedExperts for Prepared {
    fn route_tile(&self) -> u32 {
        self.base.route_tile()
    }
    fn padded_rows(&self) -> u32 {
        self.base.padded_rows()
    }
}
impl PreparedOp<GatedMlpBuffers<View>> for Prepared {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &GatedMlpBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows != self.capacity {
            return Err(Refusal("qualified prefill gate requires its exact bucket"));
        }
        workspace.covers(self.info.workspace)?;
        let decoded = b
            .scratch
            .partial
            .ok_or(Refusal("prefill gate requires decoded operand scratch"))?;
        decoded.covers(self.info.bytes_of("scratch.partial").unwrap())?;
        let routing = b.routing.ok_or(Refusal("prefill gate requires routes"))?;
        let route_counts = routing
            .routes
            .counts
            .ok_or(Refusal("prefill gate requires route counts"))?;
        let active = routing
            .routes
            .active_experts
            .ok_or(Refusal("prefill gate requires active experts"))?;
        route_counts.covers(self.info.bytes_of("routing.routes.counts").unwrap())?;
        active.covers(self.info.bytes_of("routing.routes.active_experts").unwrap())?;
        // Validate through the serving backend and retain its output quantizer.
        // The grouped baseline needs a workspace even though its gate launches
        // are replaced; the new queue allocation covers that validation.
        let mut old = self.base.launches(b, workspace, counts)?;
        if !old
            .iter()
            .take(self.baseline_symbols.len())
            .map(|l| &l.symbol)
            .eq(self.baseline_symbols.iter())
        {
            return Err(Refusal("prefill gate baseline launch sequence changed"));
        }
        old.drain(..self.baseline_symbols.len());
        let required = |v: Option<View>| v.ok_or(Refusal("prefill gate requires scaled operands"));
        let h = [
            b.activation.handle,
            b.weights.codes.handle,
            required(b.weights.local_scales)?.handle,
            required(b.activation_scales)?.handle,
            routing.routes.offsets.handle,
            routing.routes.sorted_rows.handle,
            routing.routes.padded_rows.handle,
            b.live_tokens.handle,
            routing.weights.handle,
            b.output.handle,
        ];
        let mut operands = h.to_vec();
        operands.extend([route_counts.handle, active.handle]);
        operands.extend(
            [
                b.scratch.output_codes,
                b.scratch.output_scales,
                b.scratch.quantizer_status,
            ]
            .into_iter()
            .flatten()
            .map(|v| v.handle),
        );
        if workspace.handle == 0
            || decoded.handle == 0
            || workspace.handle == decoded.handle
            || operands
                .iter()
                .any(|h| *h == workspace.handle || *h == decoded.handle)
        {
            return Err(Refusal("prefill scratch aliases an operand"));
        }
        let mut launches = family::launches(
            &self.entry,
            self.variant,
            counts.rows,
            family::Buffers {
                h,
                counts: route_counts.handle,
                active: active.handle,
                scratch: workspace.handle,
                decoded: decoded.handle,
                stamps: 0,
            },
            false,
            false,
        )
        .map_err(|_| Refusal("invalid prefill pipeline launch"))?;
        for op in &mut launches {
            op.symbol = vector::symbol(&op.symbol);
            op.plan.workspace_bytes = self.info.workspace;
        }
        launches.extend(old);
        Ok(launches)
    }
}
impl crate::Kernels {
    pub(crate) fn prefill_moe_projection(
        shape: &crate::shape::Shape,
        set: Experts,
        kind: GatedMlpProjection,
        usage: OperationUse,
        base: Box<dyn PreparedExperts>,
    ) -> Result<Box<dyn PreparedExperts>, Refused> {
        prepare(shape.geometry, shape.ranks(), set, kind, usage, base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use kernel_core::backend::{Backend, Need};
    use kernels_sm120::backend::Sm120;
    use model_deepseek41_contract::moe;
    #[test]
    fn measured_gate_only_selection_and_cubin_scope() {
        for ranks in [1, 2, 4] {
            for set in moe::expert_sets(&FLASH) {
                for phase in [
                    ExecutionPhase::Prefill,
                    ExecutionPhase::Decode,
                    ExecutionPhase::Verify,
                ] {
                    for capacity in [1, 512, 2048, 4096, 8191, 8192] {
                        for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                            let usage = OperationUse {
                                phase,
                                capacity,
                                graph: phase != ExecutionPhase::Prefill,
                            };
                            let yes = ranks == 4
                                && phase == ExecutionPhase::Prefill
                                && (capacity == 8192 || (set.count == 128 && capacity == 4096))
                                && kind == GatedMlpProjection::GateUp;
                            assert_eq!(selected(&FLASH, ranks, set, kind, usage), yes);
                            let mut other = FLASH;
                            other.id = "unqualified-same-shape";
                            assert!(!selected(&other, ranks, set, kind, usage));
                        }
                    }
                }
            }
        }
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = crate::shape::Shape::new(&FLASH, ranks).unwrap();
            assert_eq!(
                crate::Kernels::loads(vector::CUBIN, &shape),
                ranks == Ranks::TP4
            );
        }
    }
    #[test]
    fn prepared_gate_retains_quantizer_and_bounds_scratch() {
        use kernel_core::moe::RouteMetadata;
        use kernel_core::moe_experts::GatedMlpScratch;
        use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
        let backend = super::super::moe::register(Sm120::new(), &FLASH, 4).unwrap();
        for (experts, capacity, prefix) in [(384, 8192, 1), (128, 8192, 1), (128, 4096, 4)] {
            let set = moe::expert_sets(&FLASH)
                .into_iter()
                .find(|e| e.count == experts)
                .unwrap();
            let usage = OperationUse {
                phase: ExecutionPhase::Prefill,
                capacity,
                graph: false,
            };
            let query = moe::projection(&FLASH, 4, set, GatedMlpProjection::GateUp, 8192).unwrap();
            let old = backend
                .chained_mlp_projection(&query, usage, Need::rows(capacity))
                .unwrap();
            let quantizer = old.info().symbols[prefix..].to_vec();
            let op = prepare(&FLASH, 4, set, GatedMlpProjection::GateUp, usage, old).unwrap();
            assert_eq!(&op.info().symbols[6..], quantizer);
            let decoded = u64::from(capacity) * 5120 * 2;
            assert_eq!(
                op.info().workspace,
                decoded + if experts == 384 { 37_120 } else { 15_616 }
            );
            assert_eq!(op.info().bytes_of("scratch.partial"), Some(decoded));
            assert_eq!(op.info().symbols.len(), 6 + quantizer.len());
            assert_eq!(op.info().symbols[5].1, 80_400);
            for (symbol, _) in &op.info().symbols {
                assert!(
                    crate::sm120::fold_cubin_for_symbol(symbol).is_some(),
                    "{symbol}"
                );
            }
            let mut handle = 0;
            let mut sized = |name: &str| {
                handle += 1;
                View {
                    handle,
                    bytes: op.info().bytes_of(name).unwrap_or(0),
                }
            };
            let b = GatedMlpBuffers {
                activation: sized("activation"),
                activation_scales: Some(sized("activation_scales")),
                activation_global: None,
                live_tokens: sized("live_tokens"),
                output: sized("output"),
                weights: QuantizedWeights {
                    codes: sized("weights.codes"),
                    local_scales: Some(sized("weights.local_scales")),
                    global_scales: None,
                },
                routing: Some(ProjectionRouting {
                    weights: sized("routing.weights"),
                    routes: RouteMetadata {
                        offsets: sized("routing.routes.offsets"),
                        sorted_rows: sized("routing.routes.sorted_rows"),
                        padded_rows: sized("routing.routes.padded_rows"),
                        counts: Some(sized("routing.routes.counts")),
                        active_experts: Some(sized("routing.routes.active_experts")),
                    },
                }),
                scratch: GatedMlpScratch {
                    partial: Some(sized("scratch.partial")),
                    output_codes: Some(sized("scratch.output_codes")),
                    output_scales: Some(sized("scratch.output_scales")),
                    quantizer_status: Some(sized("scratch.quantizer_status")),
                    quantizer_status_row_offset: 0,
                    unused: View::EMPTY,
                },
            };
            let workspace = View {
                handle: 100,
                bytes: op.info().workspace,
            };
            let launches = op.launches(&b, workspace, Counts::rows(capacity)).unwrap();
            let old = backend
                .chained_mlp_projection(&query, usage, Need::rows(capacity))
                .unwrap();
            let old_calls = old.launches(&b, workspace, Counts::rows(capacity)).unwrap();
            assert_eq!(launches.len(), 6 + old_calls.len() - prefix);
            use kernel_core::contract::Argument::{DeviceBuffer, NullDevicePointer};
            let h = if prefix == 1 {
                old_calls[0].arguments.0.clone()
            } else {
                let pack = &old_calls[0].arguments.0;
                let compute = &old_calls[1].arguments.0;
                vec![
                    compute[0].clone(),
                    compute[1].clone(),
                    compute[2].clone(),
                    compute[3].clone(),
                    pack[1].clone(),
                    compute[4].clone(),
                    compute[5].clone(),
                    compute[6].clone(),
                    compute[8].clone(),
                    compute[9].clone(),
                ]
            };
            assert_eq!(h.len(), 10);
            let routes = b.routing.unwrap().routes;
            assert_eq!(
                launches[0].arguments.0,
                vec![
                    DeviceBuffer(routes.counts.unwrap().handle),
                    h[4].clone(),
                    DeviceBuffer(routes.active_experts.unwrap().handle),
                    h[7].clone(),
                    DeviceBuffer(workspace.handle)
                ]
            );
            let decoded = DeviceBuffer(b.scratch.partial.unwrap().handle);
            assert_eq!(
                launches[1].arguments.0,
                vec![h[0].clone(), h[3].clone(), h[7].clone(), decoded.clone()]
            );
            for call in &launches[2..6] {
                assert_eq!(
                    call.arguments.0,
                    vec![
                        decoded.clone(),
                        h[1].clone(),
                        h[2].clone(),
                        h[3].clone(),
                        h[5].clone(),
                        h[6].clone(),
                        h[7].clone(),
                        DeviceBuffer(workspace.handle),
                        h[8].clone(),
                        h[9].clone(),
                        NullDevicePointer
                    ]
                );
            }
            assert_eq!(
                op.info()
                    .bytes
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                op.info().bytes.len()
            );
            for (a, b) in launches[6..].iter().zip(&old_calls[prefix..]) {
                assert_eq!(a.symbol, b.symbol);
                assert_eq!(a.plan, b.plan);
                assert_eq!(a.arguments, b.arguments);
            }
            assert!(
                launches[..6]
                    .iter()
                    .all(|l| l.symbol.ends_with("_vector_v1"))
            );
            assert!(
                op.launches(
                    &b,
                    View {
                        bytes: workspace.bytes - 1,
                        ..workspace
                    },
                    Counts::rows(capacity)
                )
                .is_err()
            );
            assert!(
                op.launches(&b, workspace, Counts::rows(capacity - 1))
                    .is_err()
            );
            let mut bad = b;
            bad.scratch.partial.as_mut().unwrap().bytes -= 1;
            assert!(
                op.launches(&bad, workspace, Counts::rows(capacity))
                    .is_err()
            );
            for alias in [b.activation.handle, b.output.handle, workspace.handle] {
                let mut bad = b;
                bad.scratch.partial.as_mut().unwrap().handle = alias;
                assert!(
                    op.launches(&bad, workspace, Counts::rows(capacity))
                        .is_err()
                );
            }
        }
    }
}
