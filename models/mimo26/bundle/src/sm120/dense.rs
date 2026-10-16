//! MiMo's dense GEMMs on SM120 outside the backend's projection op: the
//! medium-M switch and the cubin it gates, and MiMo's dense launch builder
//! over named rank-image views, which tests compare the backend's prepared
//! dense projections with.

#[cfg(test)]
use kernel_core::contract::{
    ActivationFormat, DType, Refusal, RuntimeCounts, StaticQuery, TensorMap,
};
#[cfg(test)]
use runtime_bundle_abi::DeviceBufferSpanV1;

#[cfg(test)]
use crate::dispatch::BufferBindings;
#[cfg(test)]
use crate::pipeline::PreparedLaunch;

#[cfg(test)]
/// Bytes of an FP8 weight's F32 scales: one per 128 K columns and per 128
/// output rows, or per 64 rows for the regrouped QKV (`Fp8Rows64x128`).
fn scale_bytes(query: &kernel_core::contract::GemmQuery) -> u64 {
    let rows = match query.weight {
        kernel_core::contract::WeightFormat::Fp8Rows64x128(_) => query.n.div_ceil(64),
        _ => query.n.div_ceil(128),
    };
    u64::from(rows) * u64::from(query.k.div_ceil(128)) * 4
}

/// Whether `cubin` is the medium-M entries' cubin, which an instance loads
/// only when the medium-M switch is on.
pub fn medium_cubin(cubin: &str) -> bool {
    cubin
        == kernels_sm120::gemm::mid::cubin_name(Some(model_mimo26_contract::sm120::config::PROFILE))
}

/// Plans `entry` over `live_tokens` rows, or the medium-M entry that stands
/// in for it; `activation_map` is the prefill entry's.
#[cfg(test)]
fn plan_dense(
    entry: &kernel_core::contract::InstantiationSpec,
    medium: Option<kernels_sm120::gemm::mid::Entry>,
    counts: RuntimeCounts,
    maps: [TensorMap; 2],
    buffers: kernels_sm120::gemm::GemmBuffers,
) -> Result<
    (
        String,
        kernel_core::contract::LaunchPlan,
        kernel_core::contract::ArgumentList,
    ),
    Refusal,
> {
    let Some(medium) = medium else {
        if !crate::sm120::launch::dense::mimo_entries().any(|own| own.symbol == entry.symbol) {
            return Err(Refusal("dense entry is not a MiMo instantiation"));
        }
        let plan = kernels_sm120::gemm::plan(&entry.key, counts)?;
        let arguments = kernels_sm120::gemm::arguments(&entry.key, counts, maps, buffers)?;
        return Ok((entry.symbol.clone(), plan, arguments));
    };
    let [activation_map, weight_map] = maps;
    let mut medium_map = activation_map;
    medium_map.box_dims[1] = medium.activation_box_rows();
    let plan = medium.plan(counts.m)?;
    let arguments = medium.arguments_from(counts.m, medium_map, weight_map.base_buffer, buffers)?;
    Ok((medium.kernel.symbol(), plan, arguments))
}

#[cfg(test)]
pub struct ProjectViews {
    pub activation: DeviceBufferSpanV1,
    pub weight: DeviceBufferSpanV1,
    pub weight_scales: Option<DeviceBufferSpanV1>,
    pub output: DeviceBufferSpanV1,
    pub partial: DeviceBufferSpanV1,
    pub semaphores: DeviceBufferSpanV1,
}

/// A dense projection's launch as MiMo planned it before the SM120 backend
/// prepared every dense use; tests compare the backend's prepared ops with
/// it.
///
/// Every output must hold the entry's full row capacity, except a vocabulary
/// head's (`live_output`): logits buffers are sized for the rows a phase can
/// carry, and the fragment kernels store only rows below the live count.
#[cfg(test)]
pub(crate) fn plan_entry(
    entry: kernel_core::contract::InstantiationSpec,
    medium: Option<kernels_sm120::gemm::mid::Entry>,
    live_tokens: u32,
    views: ProjectViews,
    live_output: bool,
) -> Result<PreparedLaunch, Refusal> {
    let StaticQuery::Gemm(query) = entry.key.query else {
        return Err(Refusal("MiMo dense entry has no GEMM query"));
    };
    if query.activation != ActivationFormat::Bf16
        || live_tokens == 0
        || live_tokens > query.m_capacity
    {
        return Err(Refusal("MiMo dense activation geometry"));
    }
    let weight_bytes = if query.weight == kernel_core::contract::WeightFormat::Bf16 {
        2
    } else {
        1
    };
    let output_rows = if query.epilogues.swiglu.is_some() {
        query.n / 2
    } else {
        query.n
    };
    let output_bytes = if query.output == DType::F32 { 4 } else { 2 };
    let output_capacity = if live_output {
        live_tokens
    } else {
        query.m_capacity
    };
    if views.weight.length < u64::from(query.n) * u64::from(query.k) * weight_bytes
        || views.activation.length < u64::from(query.m_capacity) * u64::from(query.k) * 2
        || views.output.length < u64::from(output_capacity) * u64::from(output_rows) * output_bytes
        || views.semaphores.length < 4
    {
        return Err(Refusal("MiMo dense stage views are too small"));
    }
    let fp8 = weight_bytes == 1;
    if fp8 != views.weight_scales.is_some()
        || views
            .weight_scales
            .is_some_and(|scale| scale.length < scale_bytes(&query))
    {
        return Err(Refusal("MiMo dense weight scales are incomplete"));
    }
    let mut bindings = BufferBindings::default();
    let activation = bindings.bind(views.activation);
    let weight = bindings.bind(views.weight);
    let scale = views.weight_scales.map(|value| bindings.bind(value));
    let output = bindings.bind(views.output);
    let partial = bindings.bind(views.partial);
    let semaphores = bindings.bind(views.semaphores);
    let decode = query.m_capacity <= 32;
    let maps = kernels_sm120::gemm::fragment_maps(&entry.key, &query, activation, weight, decode);
    let counts = kernels_sm120::gemm::row_counts(&entry.key, live_tokens);
    let buffers = kernels_sm120::gemm::GemmBuffers {
        output,
        partial,
        semaphores,
        weight_scales: scale,
        activation_scales: None,
        residual: None,
        weight_global_scale: None,
        activation_global_scale: None,
    };
    let (symbol, plan, arguments) = plan_dense(&entry, medium, counts, maps, buffers)?;
    if views.partial.length < plan.workspace_bytes {
        return Err(Refusal("MiMo dense workspace is too small"));
    }
    Ok(PreparedLaunch {
        symbol,
        rows: live_tokens,
        plan,
        arguments,
        bindings,
    })
}

#[cfg(test)]
mod tests {
    use kernel_core::contract::InstantiationSpec;
    use model_mimo26_contract::capabilities::{Projection, sm120::symbol};
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, KernelArgumentV1};

    use super::*;
    use crate::selection::{MimoPhase as _, Phase};
    use crate::topology::Tp;

    /// The medium-M entry that replaces prefill entry `reference` in a verify
    /// or decode bucket above the decode batch at `live_tokens` rows, when
    /// `enabled`.
    fn medium_entry(
        reference: &str,
        phase: Phase,
        live_tokens: u32,
        enabled: bool,
    ) -> Option<kernels_sm120::gemm::mid::Entry> {
        if !enabled
            || !matches!(phase, Phase::Verify(_) | Phase::Decode(_))
            || !phase.prefill_kernels()
        {
            return None;
        }
        kernels_sm120::gemm::mid::select(
            model_mimo26_contract::sm120::catalogs(),
            reference,
            live_tokens,
        )
    }

    /// The catalog entry the precision component binds for `projection`.
    fn entry(projection: Projection, phase: Phase, tp: Tp) -> InstantiationSpec {
        let usage = model_mimo26_contract::prepared::usage(phase).unwrap();
        model_mimo26_contract::sm120::workspace::dense::entries_for(tp)
            .find(|e| e.symbol == symbol(projection, tp, usage))
            .unwrap()
            .clone()
    }

    fn project(
        projection: Projection,
        phase: Phase,
        live_tokens: u32,
        tp: Tp,
        views: ProjectViews,
        medium: bool,
    ) -> Result<PreparedLaunch, Refusal> {
        let entry = entry(projection, phase, tp);
        let medium = medium_entry(&entry.symbol, phase, live_tokens, medium);
        let head = matches!(projection, Projection::Head { .. });
        plan_entry(entry, medium, live_tokens, views, head)
    }

    fn qkv(
        layer: u32,
        phase: Phase,
        live: u32,
        tp: Tp,
        views: ProjectViews,
    ) -> Result<PreparedLaunch, Refusal> {
        let global = tp.geometry.is_global(layer);
        project(Projection::Qkv { global }, phase, live, tp, views, true)
    }

    #[test]
    fn qkv_maps_and_scales_follow_ga_and_swa_catalogs() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |offset, length| DeviceBufferSpanV1::new(buffer, offset, length);
        let cases: [(u32, Phase, u32, u32); 3] = [
            (0, Phase::Decode(3), 4, 3392),
            (1, Phase::Decode(1), 1, 3712),
            (5, Phase::Prefill, 8192, 3392),
        ];
        for (layer, phase, rows, n) in cases {
            let launch = qkv(
                layer,
                phase,
                if rows == 8192 { 512 } else { 1 },
                Tp::TP4,
                ProjectViews {
                    activation: span(0, u64::from(rows) * 4096 * 2),
                    weight: span(67_108_864, u64::from(n) * 4096),
                    weight_scales: Some(span(85_000_000, u64::from(n.div_ceil(128)) * 32 * 4)),
                    output: span(86_000_000, u64::from(rows) * u64::from(n) * 2),
                    partial: span(160_000_000, 2 << 30),
                    semaphores: span(250_000_000, 4096),
                },
            )
            .expect("QKV launch");
            assert!(launch.plan.pdl);
            let args = crate::dispatch::arguments(&launch.arguments, &launch.bindings)
                .expect("host GEMM arguments");
            assert_eq!(args.len(), 15);
            assert_eq!(
                args[10],
                KernelArgumentV1::DeviceBuffer(span(
                    85_000_000,
                    u64::from(n.div_ceil(128)) * 32 * 4
                ))
            );
            if rows <= 32 {
                assert_eq!(
                    args[0],
                    KernelArgumentV1::DeviceBuffer(span(67_108_864, u64::from(n) * 4096))
                );
                assert!(matches!(args[1], KernelArgumentV1::DeviceBuffer(_)));
            } else {
                let KernelArgumentV1::TensorMap(first) = args[0] else {
                    panic!("prefill activation TMA map")
                };
                assert_eq!(first.global_dims[0], 4096);
                assert_eq!(first.global_strides_bytes[0], 8192);
                assert_eq!(
                    args[1],
                    KernelArgumentV1::DeviceBuffer(span(67_108_864, u64::from(n) * 4096))
                );
            }
        }
    }

    #[test]
    fn scored_head_requires_exact_live_f32_output_extent() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |length| DeviceBufferSpanV1::new(buffer, 0, length);
        for rows in [1, 512] {
            let needed = u64::from(rows) * u64::from(crate::contract::VOCAB_PER_RANK) * 4;
            for bytes in [needed - 1, needed] {
                let result = project(
                    Projection::Head { f32: true },
                    Phase::Prefill,
                    rows,
                    Tp::TP4,
                    ProjectViews {
                        activation: span(8192 * 4096 * 2),
                        weight: span(u64::from(crate::contract::VOCAB_PER_RANK) * 4096 * 2),
                        weight_scales: None,
                        output: span(bytes),
                        partial: span(2 << 30),
                        semaphores: span(4096),
                    },
                    true,
                );
                assert_eq!(result.is_ok(), bytes == needed, "rows={rows} bytes={bytes}");
            }
        }
    }

    #[test]
    fn projections_other_than_the_head_need_their_full_row_capacity() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |length| DeviceBufferSpanV1::new(buffer, 0, length);
        let entry = entry(Projection::Output, Phase::Prefill, Tp::TP4);
        let StaticQuery::Gemm(query) = entry.key.query else {
            panic!("GEMM query")
        };
        let row_bytes = u64::from(query.n) * 2;
        let fp8 = query.weight != kernel_core::contract::WeightFormat::Bf16;
        for (bytes, accepted) in [
            (u64::from(query.m_capacity) * row_bytes, true),
            (u64::from(query.m_capacity) * row_bytes - 1, false),
            (512 * row_bytes, false),
        ] {
            let result = project(
                Projection::Output,
                Phase::Prefill,
                512,
                Tp::TP4,
                ProjectViews {
                    activation: span(u64::from(query.m_capacity) * u64::from(query.k) * 2),
                    weight: span(u64::from(query.n) * u64::from(query.k) * 2),
                    weight_scales: fp8.then(|| span(1 << 30)),
                    output: span(bytes),
                    partial: span(2 << 30),
                    semaphores: span(4096),
                },
                true,
            );
            assert_eq!(result.is_ok(), accepted, "bytes={bytes}");
        }
    }

    #[test]
    fn wide_verify_runs_the_medium_entries_behind_the_switch() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |offset, length| DeviceBufferSpanV1::new(buffer, offset, length);
        let views = || ProjectViews {
            activation: span(0, 8192 * 4096 * 2),
            weight: span(67_108_864, 3392 * 4096),
            weight_scales: Some(span(85_000_000, 27 * 32 * 4)),
            output: span(86_000_000, 8192 * 3392 * 2),
            partial: span(160_000_000, 2 << 30),
            semaphores: span(250_000_000, 4096),
        };
        let global = Projection::Qkv { global: true };
        // (phase, live rows, switch, symbol)
        for (phase, live, medium, symbol) in [
            (Phase::Verify(64), 64, true, "chop_gemm_mid_fp8b_t64_v1"),
            (
                Phase::Verify(64),
                64,
                false,
                "chop_gemm_mimo_qkv_ga_prefill_v1",
            ),
            (
                Phase::Verify(128),
                128,
                true,
                "chop_gemm_mimo_qkv_ga_prefill_v1",
            ),
            (
                Phase::Verify(256),
                256,
                true,
                "chop_gemm_mimo_qkv_ga_prefill_v1",
            ),
            (Phase::Verify(16), 16, true, "chop_gemm_mimo_qkv_ga_m16_v1"),
            // Decode above 32 rows plans its bucket's rows on the same
            // entries (CONC-MIMO).
            (Phase::Decode(64), 64, true, "chop_gemm_mid_fp8b_t64_v1"),
            (
                Phase::Decode(128),
                128,
                false,
                "chop_gemm_mimo_qkv_ga_prefill_v1",
            ),
            (
                Phase::Decode(256),
                256,
                true,
                "chop_gemm_mimo_qkv_ga_prefill_v1",
            ),
            (Phase::Decode(32), 32, true, "chop_gemm_mimo_qkv_ga_m32_v1"),
            (Phase::Prefill, 64, true, "chop_gemm_mimo_qkv_ga_prefill_v1"),
        ] {
            let launch = project(global, phase, live, Tp::TP4, views(), medium).unwrap();
            assert_eq!(launch.symbol, symbol, "{phase:?} {medium}");
            if symbol.starts_with("chop_gemm_mid_") {
                assert_eq!(launch.plan.grid, [188, 1, 1]);
                let args = crate::dispatch::arguments(&launch.arguments, &launch.bindings).unwrap();
                assert_eq!(args.len(), 15);
                let KernelArgumentV1::TensorMap(map) = args[0] else {
                    panic!("activation TMA map")
                };
                assert_eq!(map.box_dims[1], 64);
            }
        }
        let launch = project(
            Projection::Output,
            Phase::Verify(64),
            64,
            Tp::TP4,
            ProjectViews {
                activation: span(0, 8192 * 2048 * 2),
                weight: span(100, 4096 * 2048 * 2),
                weight_scales: None,
                output: span(300, 8192 * 4096 * 2),
                partial: span(400, 2 << 30),
                semaphores: span(500, 4096),
            },
            true,
        )
        .unwrap();
        assert_eq!(launch.symbol, "chop_gemm_mid_bf16_t64_v1");
        // Every medium kernel MiMo can select is embedded.
        for spec in kernels_sm120::gemm::mid::cubin_specs(
            model_mimo26_contract::sm120::catalogs(),
            Some(model_mimo26_contract::sm120::config::PROFILE),
        ) {
            for entry in spec.entries {
                assert!(
                    crate::CUBINS
                        .iter()
                        .any(|cubin| cubin.symbols.contains(&entry.symbol.as_str())),
                    "{}",
                    entry.symbol
                );
            }
        }
    }

    #[test]
    fn every_other_mimo_projection_packs_its_engine_plan() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |offset, length| DeviceBufferSpanV1::new(buffer, offset, length);
        for phase in [Phase::Decode(3), Phase::Prefill] {
            for projection in [
                Projection::Output,
                Projection::MlpGateUp,
                Projection::MlpDown,
                Projection::Router,
                Projection::Head {
                    f32: phase == Phase::Prefill,
                },
            ] {
                let StaticQuery::Gemm(query) = entry(projection, phase, Tp::TP4).key.query else {
                    panic!("GEMM query")
                };
                let fp8 = query.weight != kernel_core::contract::WeightFormat::Bf16;
                let launch = project(
                    projection,
                    phase,
                    1,
                    Tp::TP4,
                    ProjectViews {
                        activation: span(0, u64::from(query.m_capacity) * u64::from(query.k) * 2),
                        weight: span(
                            100,
                            u64::from(query.n) * u64::from(query.k) * if fp8 { 1 } else { 2 },
                        ),
                        weight_scales: fp8.then(|| {
                            span(
                                200,
                                u64::from(query.n.div_ceil(128))
                                    * u64::from(query.k.div_ceil(128))
                                    * 4,
                            )
                        }),
                        output: span(300, u64::from(query.m_capacity) * u64::from(query.n) * 4),
                        partial: span(400, 2 << 30),
                        semaphores: span(500, 4096),
                    },
                    true,
                )
                .expect("MiMo projection plan");
                assert!(launch.plan.pdl);
                // Fifteen common arguments.
                assert_eq!(
                    crate::dispatch::arguments(&launch.arguments, &launch.bindings)
                        .unwrap()
                        .len(),
                    launch.arguments.0.len()
                );
                assert_eq!(launch.arguments.0.len(), 15);
            }
        }
    }

    /// The regrouped QKV takes one scale row per 64 output rows at TP2,
    /// and refuses TP4's shorter block grid.
    #[test]
    fn regrouped_qkv_needs_64_row_scales() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |offset, length| DeviceBufferSpanV1::new(buffer, offset, length);
        let tp = Tp::TP2;
        for (layer, global) in [(0, true), (1, false)] {
            let n = tp.qkv_rows(global);
            let views = |scale_rows: u64| ProjectViews {
                activation: span(0, 8192 * 4096 * 2),
                weight: span(1 << 30, n * 4096),
                weight_scales: Some(span(3 << 30, scale_rows * 32 * 4)),
                output: span(4 << 30, 8192 * n * 2),
                partial: span(8 << 30, 2 << 30),
                semaphores: span(12 << 30, 4096),
            };
            for phase in [Phase::Decode(2), Phase::Prefill] {
                let launch = qkv(layer, phase, 2, tp, views(n / 64)).unwrap();
                assert!(launch.symbol.contains(crate::topology::kernel_tag(tp)));
                assert!(qkv(layer, phase, 2, tp, views(n.div_ceil(128))).is_err());
            }
        }
    }
}
