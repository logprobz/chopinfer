//! MiMo's launch builders from before its ops were prepared through the
//! backend, which tests compare the prepared ops with.

use kernel_core::contract::Refusal;
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::dispatch::BufferBindings;
use crate::pipeline::PreparedLaunch;
use crate::selection::Phase;
use crate::topology::Tp;

/// The sigmoid-bias decode route of a 16-row bucket of `live_tokens` rows
/// over `spans`: router logits, score correction bias, top-k ids and
/// weights, expert counts and offsets, sorted rows, padded rows, live
/// tokens and active experts.
pub fn decode_route(
    tp: Tp,
    capacity: u32,
    live_tokens: u32,
    spans: [DeviceBufferSpanV1; 10],
) -> Result<PreparedLaunch, Refusal> {
    if spans[5].length < 257 * 4 || spans[9].length < 257 * 4 {
        return Err(Refusal("fused MiMo route metadata buffers are too small"));
    }
    let mut bindings = BufferBindings::default();
    let handles = spans.map(|span| bindings.bind(span));
    Ok(PreparedLaunch {
        symbol: crate::topology::tagged(model_mimo26_contract::sm120::moe::DECODE_ROUTE_SYMBOL, tp),
        rows: capacity,
        plan: kernels_sm120::moe::decode_route_plan(capacity, live_tokens)?,
        arguments: kernels_sm120::moe::decode_route_arguments(handles, 1.0)?,
        bindings,
    })
}

/// The fused router GEMV and route of a bucket of `capacity` rows (1, 2, 4
/// or 8), of `live_tokens` rows: the router `weight` times the `normalized`
/// hidden rows into the router logits, then the route, over `spans` as
/// [`decode_route`] takes them, with the GEMM `workspace` and
/// `semaphores`.
pub fn fused_route(
    tp: Tp,
    capacity: u32,
    live_tokens: u32,
    [weight, normalized, workspace, semaphores]: [DeviceBufferSpanV1; 4],
    spans: [DeviceBufferSpanV1; 10],
) -> Result<PreparedLaunch, Refusal> {
    let plan = model_mimo26_contract::sm120::moe::fused_router_plan(capacity, live_tokens)?;
    let mut bindings = BufferBindings::default();
    let handles = [
        weight, normalized, spans[0], workspace, semaphores, spans[8], spans[1], spans[2],
        spans[3], spans[4], spans[5], spans[6], spans[7], spans[9],
    ]
    .map(|span| bindings.bind(span));
    Ok(PreparedLaunch {
        symbol: format!(
            "{}_t{capacity}_v1",
            crate::topology::tagged(model_mimo26_contract::sm120::moe::FUSED_ROUTER_PREFIX, tp)
        ),
        rows: capacity,
        plan,
        arguments: kernels_sm120::moe::fused_router_arguments(handles, 1.0)?,
        bindings,
    })
}

#[derive(Clone, Copy)]
pub struct ExpertBuffers {
    pub activation: DeviceBufferSpanV1,
    pub weight: DeviceBufferSpanV1,
    pub weight_scales: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub topk_weights: DeviceBufferSpanV1,
    pub expert_offsets: DeviceBufferSpanV1,
    pub sorted_rows: DeviceBufferSpanV1,
    pub padded_rows: DeviceBufferSpanV1,
    pub live_tokens: DeviceBufferSpanV1,
    pub active_experts: DeviceBufferSpanV1,
    pub partials: DeviceBufferSpanV1,
    pub unused_scale: DeviceBufferSpanV1,
}

/// MiMo's grouped MXFP4 expert launches of `variant` over contiguous
/// rank-image views, as its registered launcher built them before it
/// launched prepared ops.
pub fn project_entry(
    variant: &crate::sm120::launch::moe::Variant,
    phase: Phase,
    live_tokens: u32,
    tp: Tp,
    views: ExpertBuffers,
) -> Result<Vec<PreparedLaunch>, Refusal> {
    if variant.path != crate::sm120::launch::moe::InputPath::A16Gather
        || variant.query.weight != kernel_core::contract::WeightFormat::Mxfp4
    {
        return Err(Refusal(
            "MiMo expert path must use gathered BF16 and MXFP4 weights",
        ));
    }
    let weight_bytes = u64::from(tp.geometry.experts() * variant.query.n * variant.query.k / 2);
    if views.weight.length < weight_bytes
        || views.weight_scales.length < weight_bytes / 16
        || views.live_tokens.length < 4
        || views.active_experts.length
            < if phase.token_bucketed() {
                (tp.geometry.experts() as u64 + 1) * 4
            } else {
                4
            }
    {
        return Err(Refusal("MiMo grouped expert views are incomplete"));
    }
    let counts = variant.counts(
        live_tokens,
        live_tokens
            .checked_mul(tp.geometry.top_k())
            .ok_or(Refusal("MiMo routed rows overflow"))?,
    );
    let mut bindings = BufferBindings::default();
    let weight_buffer = bindings.bind(views.weight);
    let spans = [
        views.activation,
        views.unused_scale,
        views.unused_scale,
        views.weight_scales,
        views.unused_scale,
        views.output,
        views.topk_weights,
        views.expert_offsets,
        views.sorted_rows,
        views.padded_rows,
        views.live_tokens,
        views.active_experts,
        views.unused_scale,
        views.unused_scale,
        views.partials,
    ];
    let handles = spans.map(|span| bindings.bind(span));
    let launches = crate::sm120::launch::moe::plan(variant, counts, handles, weight_buffer)?;
    if launches
        .iter()
        .any(|(_, plan, _)| plan.workspace_bytes > views.partials.length)
    {
        return Err(Refusal(
            "MiMo grouped expert partial workspace is too small",
        ));
    }
    Ok(launches
        .into_iter()
        .map(|(symbol, plan, arguments)| PreparedLaunch {
            symbol,
            rows: live_tokens,
            plan,
            arguments,
            bindings: bindings.clone(),
        })
        .collect())
}

/// The SM120 row catalog's entries of MiMo's catalogs.
pub fn row_catalog() -> Vec<kernels_sm120::row::RowInstantiation> {
    kernels_sm120::row::catalog(model_mimo26_contract::sm120::catalogs())
}

/// The medium-M entry that stands in for prefill entry `reference` at
/// `rows` rows, if any (`gemm::mid::select`).
pub fn medium_symbol(reference: &str, rows: u32) -> Option<String> {
    kernels_sm120::gemm::mid::select(model_mimo26_contract::sm120::catalogs(), reference, rows)
        .map(|entry| entry.kernel.symbol())
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::*;
    use crate::ops::{ExpertViews, Experts};
    use crate::selection::MimoPhase as _;
    use crate::switches::Switches;

    /// A span of its own buffer.
    fn span(buffer: u32, bytes: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(buffer, 1),
            256,
            bytes,
        )
    }

    /// Prefill and every decode and verify bucket.
    fn phases() -> Vec<Phase> {
        crate::scratch::COMM_BUCKETS
            .iter()
            .flat_map(|&rows| {
                [Phase::Decode(rows), Phase::Verify(rows), Phase::Prefill]
                    .into_iter()
                    .filter(move |phase| phase.capacity() == Ok(rows))
            })
            .collect()
    }

    /// Each launch's symbol, rows, plan and resolved arguments.
    fn resolved(
        launches: &[PreparedLaunch],
    ) -> Vec<(
        String,
        u32,
        kernel_core::contract::LaunchPlan,
        Vec<runtime_bundle_abi::KernelArgumentV1>,
    )> {
        launches
            .iter()
            .map(|launch| {
                (
                    launch.symbol.clone(),
                    launch.rows,
                    launch.plan,
                    crate::dispatch::arguments(&launch.arguments, &launch.bindings).unwrap(),
                )
            })
            .collect()
    }

    /// The row counts compared up to `capacity`: one, the capacity, and
    /// each power of two and the count past it.
    fn counts(capacity: u32) -> Vec<u32> {
        let mut counts = vec![1, capacity];
        let mut rows = 1;
        while rows < capacity {
            counts.extend([rows, rows + 1]);
            rows *= 2;
        }
        counts.sort_unstable();
        counts.dedup();
        counts
    }

    /// MiMo's expert uses launch through the prepared gated MLP ops of its
    /// registered launcher what its expert builder planned: every MXFP4
    /// entry the component binds (gate/up and down, the exact and inexact
    /// alternatives, template-scheduled), prepared for the use with what it
    /// needs in both exact verify settings, at every phase and bucket and
    /// row counts up to it, at TP2 and TP4.
    #[test]
    fn expert_ops_launch_what_the_expert_builder_planned() {
        let mut compared = 0;
        let mut persistent = 0;
        for tp in Tp::SUPPORTED {
            for exact_verify in [false, true] {
                let switches = Switches {
                    exact_verify,
                    ..Switches::configured()
                };
                for (gate_up, exact) in [(true, true), (true, false), (false, true), (false, false)]
                {
                    for phase in phases() {
                        let usage = model_mimo26_contract::prepared::usage(phase).unwrap();
                        let (model_mimo26_contract::sm120::prepared::Entry::Expert(variant), _) =
                            model_mimo26_contract::sm120::prepared::expert(
                                tp, gate_up, usage, exact,
                            )
                            .unwrap()
                        else {
                            unreachable!()
                        };
                        let id = format!(
                            "moe.experts.{}.3.bf16{}",
                            if gate_up { "gate-up" } else { "down" },
                            if exact { "" } else { ".inexact" }
                        );
                        let need = crate::projection::need(&id, usage, switches).unwrap();
                        let (backend, op) =
                            super::super::projection::expert_backend(variant.clone());
                        let experts = Experts::prepare(&*backend, &op, &[(usage, need)])
                            .unwrap_or_else(|e| panic!("{} {phase:?}: {e:?}", variant.symbol));
                        let capacity = phase.capacity().unwrap();
                        // Views that hold any expert launch.
                        let any = |buffer| span(buffer, 1 << 36);
                        let buffers = ExpertBuffers {
                            activation: any(1),
                            weight: any(2),
                            weight_scales: any(3),
                            output: any(4),
                            topk_weights: any(5),
                            expert_offsets: any(6),
                            sorted_rows: any(7),
                            padded_rows: any(8),
                            live_tokens: any(9),
                            active_experts: any(10),
                            partials: any(11),
                            unused_scale: any(12),
                        };
                        let views = ExpertViews {
                            activation: buffers.activation,
                            weight: buffers.weight,
                            weight_scales: buffers.weight_scales,
                            output: buffers.output,
                            route_weights: buffers.topk_weights,
                            offsets: buffers.expert_offsets,
                            sorted_rows: buffers.sorted_rows,
                            padded_rows: buffers.padded_rows,
                            active_experts: buffers.active_experts,
                            live_tokens: buffers.live_tokens,
                            partials: buffers.partials,
                            unused: buffers.unused_scale,
                        };
                        for rows in counts(capacity) {
                            let planned =
                                project_entry(&variant, phase, rows, tp, buffers).unwrap();
                            let lowered = experts.launch(usage, rows, &views).unwrap();
                            persistent += lowered
                                .iter()
                                .filter(|lowered| lowered.facts.persistent)
                                .count();
                            assert_eq!(
                                resolved(&crate::program_lower::launches(lowered)),
                                resolved(&planned),
                                "{tp:?} {} {phase:?} {rows}",
                                variant.symbol
                            );
                            compared += 1;
                        }
                    }
                }
            }
        }
        assert!(compared > 500, "{compared}");
        assert!(persistent > 0);
    }
}
