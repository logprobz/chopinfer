//! Default native block-scaled MoE prefill, a within-bound path that
//! changes bits (models/deepseek41/lab/native_moe_prefill.md). With
//! `CHOPINFER_DEEPSEEK41_NATIVE_MOE_PREFILL` unset or `1`, Flash TP4's target expert
//! bank runs its gate/up and down tiles through native MXFP4 x MXFP8 MMA
//! in every prefill step, whatever its row count, so a prompt row's
//! arithmetic still does not depend on the chunking. Routing, the gate/up
//! epilogue, the intermediate ceil quantizer and the routed sum are the
//! serving ones. Decode, verify and the drafter's bank are unchanged.
use super::moe_native_entries as entries;
use kernel_core::backend::{Counts, Launch, OpInfo, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::moe_experts::{GatedMlpBuffers, PreparedExperts};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::moe::native_grouped::{self as native, Shape, Tile, Variant};
use model_deepseek41_contract::{
    geometry::{FLASH, Geometry},
    moe::Experts,
};

pub const SWITCH: &str = "CHOPINFER_DEEPSEEK41_NATIVE_MOE_PREFILL";
pub const CUBIN: &str = entries::CUBIN;
/// The intermediate quantizer the gate/up op keeps after its tiles.
const QUANTIZER: &str = "_moe_ceil_";

fn selected(setting: Option<&str>) -> bool {
    matches!(setting, None | Some("1"))
}

/// Whether the instance selects native MoE prefill, on by default.
pub fn requested() -> bool {
    #[cfg(test)]
    return IN_TESTS.get();
    #[cfg(not(test))]
    match std::env::var(SWITCH) {
        Ok(value) => selected(Some(&value)),
        Err(std::env::VarError::NotPresent) => selected(None),
        Err(_) => false,
    }
}

#[cfg(test)]
thread_local! { static IN_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(true) }; }

#[cfg(test)]
pub(crate) fn with_native<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            IN_TESTS.set(self.0);
        }
    }
    let _reset = Reset(IN_TESTS.replace(enabled));
    run()
}

/// The topology and bank the entries compile for.
pub fn eligible(g: &Geometry, ranks: u32) -> bool {
    g.id == FLASH.id && ranks == 4
}

pub fn applies(g: &Geometry, ranks: u32, set: Experts, usage: OperationUse) -> bool {
    eligible(g, ranks)
        && set == entries::target_set()
        && usage.phase == ExecutionPhase::Prefill
        && !usage.graph
        && usage.capacity <= entries::CAPACITY
}

/// Replace `base`'s expert tiles with the native ones when requested and
/// `usage` is a prefill of the target bank; otherwise return `base`.
pub fn prepare(
    g: &Geometry,
    ranks: u32,
    set: Experts,
    kind: GatedMlpProjection,
    usage: OperationUse,
    base: Box<dyn PreparedExperts>,
) -> Result<Box<dyn PreparedExperts>, Refused> {
    if !requested() || !applies(g, ranks, set, usage) {
        return Ok(base);
    }
    wrap(kind, usage, base).map_err(Refused::from)
}

/// The wrapper itself, for the lab; it does not read the switch.
pub fn wrap(
    kind: GatedMlpProjection,
    usage: OperationUse,
    base: Box<dyn PreparedExperts>,
) -> Result<Box<dyn PreparedExperts>, Refusal> {
    let gate_up = kind == GatedMlpProjection::GateUp;
    let shape = entries::shape(kind);
    let tile = entries::tile(kind);
    let symbols = &base.info().symbols;
    let split = symbols
        .iter()
        .position(|(s, _)| s.contains(QUANTIZER))
        .unwrap_or(symbols.len());
    if split == 0
        || gate_up != (split < symbols.len())
        || symbols[split..].iter().any(|(s, _)| !s.contains(QUANTIZER))
        || base.route_tile() != 64
    {
        return Err(Refusal(
            "native prefill needs grouped tiles and the ceil quantizer",
        ));
    }
    let queue = native::workspace_bytes(&shape, &tile);
    let mut info = base.info().clone();
    info.implementation.push_str("+native_mxf8f6f4_prefill_v1");
    // The queue takes the workspace's first bytes. A base that carves
    // decoded operands from the workspace's end keeps its layout.
    let decoded = info.bytes_of("scratch.partial").unwrap_or(0);
    info.workspace = info.workspace.max(queue + decoded);
    for (name, bytes) in [
        (
            "routing.routes.counts".to_string(),
            u64::from(shape.experts) * 4,
        ),
        (
            "routing.routes.active_experts".to_string(),
            u64::from(shape.experts + 1) * 4,
        ),
    ] {
        if let Some(existing) = info.bytes.iter_mut().find(|(key, _)| key == &name) {
            existing.1 = existing.1.max(bytes);
        } else {
            info.bytes.push((name, bytes));
        }
    }
    let shared = native::shared_bytes(&shape, &tile);
    info.symbols.splice(
        ..split,
        [
            (
                native::symbol(&shape, &tile, native::Stage::Pack, Variant::Serving),
                0,
            ),
            (
                native::symbol(&shape, &tile, native::Stage::Compute, Variant::Serving),
                shared,
            ),
        ],
    );
    Ok(Box::new(Prepared {
        base,
        shape,
        tile,
        info,
        capacity: usage.capacity,
    }))
}

struct Prepared {
    base: Box<dyn PreparedExperts>,
    shape: Shape,
    tile: Tile,
    info: OpInfo,
    capacity: u32,
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
        if counts.rows == 0 || counts.rows > self.capacity {
            return Err(Refusal("native prefill rows exceed the prepared capacity"));
        }
        workspace.covers(self.info.workspace)?;
        // The base validates every operand and supplies the quantizer.
        let mut old = self.base.launches(b, workspace, counts)?;
        let split = old
            .iter()
            .position(|l| l.symbol.contains(QUANTIZER))
            .unwrap_or(old.len());
        if split == 0
            || self.shape.gate_up != (split < old.len())
            || old[split..].iter().any(|l| !l.symbol.contains(QUANTIZER))
        {
            return Err(Refusal("native prefill base launch sequence changed"));
        }
        let quantizer = old.split_off(split);
        let routing = b.routing.ok_or(Refusal("native prefill requires routes"))?;
        let route_counts = routing
            .routes
            .counts
            .ok_or(Refusal("native prefill requires route counts"))?;
        let active = routing
            .routes
            .active_experts
            .ok_or(Refusal("native prefill requires active experts"))?;
        route_counts.covers(self.info.bytes_of("routing.routes.counts").unwrap())?;
        active.covers(self.info.bytes_of("routing.routes.active_experts").unwrap())?;
        let required =
            |v: Option<View>| v.ok_or(Refusal("native prefill requires scaled operands"));
        let buffers = native::Buffers {
            input: b.activation.handle,
            weights: b.weights.codes.handle,
            weight_scales: required(b.weights.local_scales)?.handle,
            activation_scales: required(b.activation_scales)?.handle,
            offsets: routing.routes.offsets.handle,
            sorted_rows: routing.routes.sorted_rows.handle,
            padded_rows: routing.routes.padded_rows.handle,
            live_tokens: b.live_tokens.handle,
            route_weights: if self.shape.gate_up {
                routing.weights.handle
            } else {
                0
            },
            output: b.output.handle,
            counts: route_counts.handle,
            active: active.handle,
            scratch: workspace.handle,
            raw: 0,
        };
        let operands = [
            buffers.input,
            buffers.weights,
            buffers.weight_scales,
            buffers.activation_scales,
            buffers.offsets,
            buffers.sorted_rows,
            buffers.padded_rows,
            buffers.live_tokens,
            routing.weights.handle,
            buffers.output,
            buffers.counts,
            buffers.active,
        ];
        if workspace.handle == 0 || operands.contains(&workspace.handle) {
            return Err(Refusal("native prefill queue aliases an operand"));
        }
        let mut launches = native::launches(
            &self.shape,
            &self.tile,
            counts.rows,
            &buffers,
            Variant::Serving,
        )?;
        for op in &mut launches {
            op.plan.workspace_bytes = self.info.workspace;
        }
        launches.extend(quantizer);
        Ok(launches)
    }
}

impl crate::Kernels {
    pub(crate) fn native_moe_prefill(
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
    use kernel_core::backend::{Backend, Need};
    use kernel_core::moe::RouteMetadata;
    use kernel_core::moe_experts::GatedMlpScratch;
    use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
    use kernels_sm120::backend::Sm120;
    use model_deepseek41_contract::moe;

    #[test]
    fn switch_defaults_on_and_keeps_the_explicit_opt_out() {
        assert!(selected(None));
        assert!(selected(Some("1")));
        for value in [Some("0"), Some(""), Some("true")] {
            assert!(!selected(value));
        }
        assert!(requested());
        assert!(!with_native(false, requested));
        assert!(with_native(true, requested));
        assert!(requested());
    }

    #[test]
    fn applies_to_every_target_prefill_and_nothing_else() {
        let target = entries::target_set();
        for set in moe::expert_sets(&FLASH) {
            for phase in [
                ExecutionPhase::Prefill,
                ExecutionPhase::Decode,
                ExecutionPhase::Verify,
            ] {
                for capacity in [1, 65, 512, 2048, 4096, 8192, 8193] {
                    for ranks in [1, 2, 4] {
                        let usage = OperationUse {
                            phase,
                            capacity,
                            graph: phase != ExecutionPhase::Prefill,
                        };
                        assert_eq!(
                            applies(&FLASH, ranks, set, usage),
                            ranks == 4
                                && set == target
                                && phase == ExecutionPhase::Prefill
                                && capacity <= 8192
                        );
                    }
                }
            }
        }
        let mut other = FLASH;
        other.id = "unqualified-same-shape";
        let usage = OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        };
        assert!(!applies(&other, 4, target, usage));
    }

    fn views(op: &dyn PreparedExperts) -> (GatedMlpBuffers<View>, View) {
        let mut handle = 0;
        let mut sized = |name: &str| {
            handle += 1;
            View {
                handle,
                bytes: op.info().bytes_of(name).unwrap_or(1 << 30),
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
                partial: op
                    .info()
                    .bytes_of("scratch.partial")
                    .map(|_| sized("scratch.partial")),
                output_codes: Some(sized("scratch.output_codes")),
                output_scales: Some(sized("scratch.output_scales")),
                quantizer_status: Some(sized("scratch.quantizer_status")),
                quantizer_status_row_offset: 0,
                unused: View::EMPTY,
            },
        };
        (
            b,
            View {
                handle: 1000,
                bytes: op.info().workspace,
            },
        )
    }

    #[test]
    fn wraps_every_prefill_kind_and_keeps_the_quantizer() {
        let backend = super::super::moe::register(Sm120::new(), &FLASH, 4).unwrap();
        let set = entries::target_set();
        // The exact buckets and the general prefill kind (any row count).
        for (capacity, rows, bucket) in [
            (8192, 8192, true),
            (8192, 65, false),
            (8192, 8192, false),
            (4096, 4096, true),
            (2048, 2048, true),
            (512, 512, true),
        ] {
            let usage = OperationUse {
                phase: ExecutionPhase::Prefill,
                capacity,
                graph: false,
            };
            for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                let query = moe::projection(&FLASH, 4, set, kind, capacity).unwrap();
                // The serving baseline, including the measured vector gate/up.
                let serving = || {
                    let base = backend
                        .chained_mlp_projection(&query, usage, Need::rows(capacity))
                        .unwrap();
                    if bucket {
                        super::super::moe_pipeline::prepare(&FLASH, 4, set, kind, usage, base)
                            .unwrap()
                    } else {
                        base
                    }
                };
                let base = serving();
                let base_symbols = base.info().symbols.clone();
                let (b, workspace) = views(base.as_ref());
                let old = base.launches(&b, workspace, Counts::rows(rows)).unwrap();
                let unchanged =
                    with_native(false, || prepare(&FLASH, 4, set, kind, usage, serving())).unwrap();
                assert_eq!(unchanged.info().symbols, base_symbols);
                let op = with_native(true, || prepare(&FLASH, 4, set, kind, usage, base)).unwrap();
                let quantizer = base_symbols
                    .iter()
                    .filter(|(s, _)| s.contains(QUANTIZER))
                    .cloned()
                    .collect::<Vec<_>>();
                assert_eq!(quantizer.is_empty(), kind == GatedMlpProjection::Down);
                assert_eq!(op.info().symbols.len(), 2 + quantizer.len());
                assert_eq!(&op.info().symbols[2..], &quantizer[..]);
                assert!(
                    op.info()
                        .implementation
                        .ends_with("+native_mxf8f6f4_prefill_v1")
                );
                for (symbol, _) in &op.info().symbols {
                    assert!(
                        crate::sm120::fold_cubin_for_symbol(symbol).is_some(),
                        "{symbol}"
                    );
                }
                let (b, workspace) = views(op.as_ref());
                let launches = op.launches(&b, workspace, Counts::rows(rows)).unwrap();
                assert_eq!(launches.len(), 2 + quantizer.len());
                let tail = &old[old.len() - quantizer.len()..];
                for (new, old) in launches[2..].iter().zip(tail) {
                    assert_eq!(new.symbol, old.symbol);
                    assert_eq!(new.arguments, old.arguments);
                }
                assert_eq!(
                    launches[1].plan.dynamic_shared_bytes,
                    op.info().symbols[1].1
                );
                assert!(
                    launches[..2]
                        .iter()
                        .all(|l| l.plan.workspace_bytes == op.info().workspace)
                );
                assert!(
                    op.launches(&b, workspace, Counts::rows(capacity + 1))
                        .is_err()
                );
                assert!(
                    op.launches(
                        &b,
                        View {
                            bytes: workspace.bytes - 1,
                            ..workspace
                        },
                        Counts::rows(rows)
                    )
                    .is_err()
                );
                let mut alias = b;
                alias.output.handle = workspace.handle;
                assert!(op.launches(&alias, workspace, Counts::rows(rows)).is_err());
            }
        }
        // Decode is never wrapped.
        let usage = OperationUse {
            phase: ExecutionPhase::Decode,
            capacity: 8,
            graph: true,
        };
        let query = moe::projection(&FLASH, 4, set, GatedMlpProjection::Down, 8).unwrap();
        let base = backend
            .chained_mlp_projection(&query, usage, Need::rows(8))
            .unwrap();
        let symbols = base.info().symbols.clone();
        let op = with_native(true, || {
            prepare(&FLASH, 4, set, GatedMlpProjection::Down, usage, base)
        })
        .unwrap();
        assert_eq!(op.info().symbols, symbols);
    }
}
