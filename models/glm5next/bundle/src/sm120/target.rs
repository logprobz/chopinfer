//! [`crate::target::Target`] on SM120, on the own kernels' type
//! (`Sm120Own`), which an instance builds once at create.

use std::collections::BTreeMap;

use bundle_runtime::acquire::{Log, RankHandles};
use bundle_runtime::encoder::Encoder;
use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::residency::SpinningEntry;
use bundle_runtime::resources::Held;
use kernel_core::backend::{Backend, Need, Refused};
use kernel_core::contract::WeightFormat;
use kernel_core::precision::OperationUse;
use model_glm5next_contract::own_ops::OwnOps;
use model_glm5next_contract::own_ops::stacked::{StackedPart, StackedProjection};
use model_glm5next_contract::quantization::Layout;
use model_glm5next_contract::sm120::engines::{
    ARGMAX_CUBIN, BF16_LOW_LATENCY_ROWS, COMM_EPILOGUE_CUBIN, LL_ROUND_SYMBOL, LL_SYMBOL,
    PHASED_CALLS_PER_RUN, PHASED_RESET_RUNS, ROUNDED_LOW_LATENCY_ROWS, bf16_phased_admission_plans,
    rounded_phased_admission_plans,
};
use model_glm5next_contract::sm120::{precision, projection_formats, stacked};
use runtime_bundle_abi::{
    BundleAbiError, BundleCudaSymbolHandleV1, DeviceBufferSpanV1 as Span, HostServicesV1,
};

use super::engines::Engines;
use super::own_ops::Sm120Own;
use crate::ops::Served;
use crate::ops::own::{Stacked, StackedOps};
use crate::target::{PoolGateSpans, Target, Uses};

/// A spinning entry at `plan`.
fn spinning(symbol: &'static str, plan: kernel_core::contract::LaunchPlan) -> SpinningEntry {
    SpinningEntry {
        symbol,
        plan,
        wide: false,
    }
}

/// Whether an instance serving an export stored in `layout` loads the
/// embedded `cubin`: the other formats' cubins (the chained engine's other
/// formats and its NVFP4 and BF16 block warps, of the dense MLP and shared
/// expert, `engines::MOE_FORMATS_FAMILY`, and of the split routed experts,
/// and the projections they add, `projection_formats`) load only for a
/// layout with a class outside block FP8, so an instance storing every
/// class in block FP8 loads none of them.
fn loads_format(layout: &Layout, cubin: &str) -> bool {
    use kernels_sm120::moe::chained::{self, Weights};
    let other = [Weights::Nvfp4, Weights::Mxfp8, Weights::Bf16];
    let warps = [Weights::Nvfp4, Weights::Bf16];
    let format = cubin == projection_formats::CUBIN
        || cubin.starts_with(chained::formats::PREFIX)
        || [chained::PREFIX, chained::split::PREFIX]
            .iter()
            .any(|prefix| {
                other
                    .iter()
                    .any(|weights| cubin == format!("{prefix}_{}", weights.tag()))
                    || warps
                        .iter()
                        .any(|weights| cubin == format!("{prefix}_{}_bw4", weights.tag()))
            });
    !layout.block_fp8() || !format
}

impl Target for Sm120Own {
    fn backend(&self, layout: &'static Layout) -> Result<(Box<dyn Backend>, Served), String> {
        let shape = self.shape();
        Ok(super::backend(Engines::shared(shape, layout)?, shape))
    }

    fn shared_bytes(&self, layout: &'static Layout) -> Result<BTreeMap<String, u32>, String> {
        Ok(Engines::shared(self.shape(), layout)?.shared_bytes())
    }

    /// The encoder's cubins load with the encoder (`encoders::load`), the
    /// context-parallel ones only above size 1 (the group all-gather's also
    /// with the drafter, whose argmax gathers its pairs through it), the
    /// shared argmax's only with the drafter, the NVFP4 exports' only for
    /// their layouts, and the comm epilogue's only with its switch, so an
    /// instance without it loads, resolves and journals what it did before
    /// the epilogue existed.
    fn loads(&self, cubin: &str, uses: Uses) -> bool {
        let [_, group_gather, _] = model_glm5next_contract::sm120::context_parallel::cubin_names();
        !model_glm5next_contract::sm120::encoder::is_cubin(cubin)
            && loads_format(uses.layout, cubin)
            && (uses.context_parallel
                || !model_glm5next_contract::sm120::context_parallel::is_cubin(cubin)
                || (uses.drafter && cubin == group_gather))
            && (uses.drafter || cubin != ARGMAX_CUBIN)
            && (uses.comm_epilogue || cubin != COMM_EPILOGUE_CUBIN)
    }

    fn low_latency(&self) -> Result<Vec<SpinningEntry>, String> {
        let shape = self.shape();
        let bf16 = super::low_latency_plan(shape)
            .map(|plan| spinning(LL_SYMBOL, plan))
            .map_err(|refusal| format!("{LL_SYMBOL} at {BF16_LOW_LATENCY_ROWS} rows: {refusal}"))?;
        let rounded = super::rounded_low_latency_plan(shape)
            .map(|plan| spinning(LL_ROUND_SYMBOL, plan))
            .map_err(|refusal| {
                format!("{LL_ROUND_SYMBOL} at {ROUNDED_LOW_LATENCY_ROWS} rows: {refusal}")
            })?;
        Ok(vec![bf16, rounded])
    }

    fn phased(&self) -> Result<Vec<SpinningEntry>, String> {
        let plans = bf16_phased_admission_plans()
            .and_then(|bf16| Ok([bf16, rounded_phased_admission_plans()?].concat()))
            .map_err(|refusal| format!("phased all-reduce plan: {refusal}"))?;
        Ok(plans
            .into_iter()
            .map(|(symbol, plan)| spinning(symbol, plan))
            .collect())
    }

    fn phased_reset_runs(&self) -> u64 {
        PHASED_RESET_RUNS
    }

    fn phased_calls_per_run(&self) -> u64 {
        PHASED_CALLS_PER_RUN
    }

    fn pool_gate(
        &self,
        key: &str,
        usage: OperationUse,
        rows: u32,
        spans: PoolGateSpans,
    ) -> Result<PreparedLaunch, String> {
        pool_gate(key, usage, rows, spans)
    }

    fn stacked(
        &self,
        own: &dyn OwnOps,
        need: &dyn Fn(OperationUse) -> Need,
    ) -> Result<StackedOps, Refused> {
        stacked(own, need)
    }

    fn row_bucket(&self, rows: u32) -> Option<u32> {
        kernels_sm120::row::smallest_bucket(rows)
    }

    fn load_encoder(
        &self,
        log: Log,
        ranks: &[RankHandles],
        held: &mut Vec<Held>,
    ) -> Result<Vec<BTreeMap<String, BundleCudaSymbolHandleV1>>, BundleAbiError> {
        super::encoders::load(log, ranks, held)
    }

    fn encoder(
        &self,
        services: &HostServicesV1,
        weights: Vec<BTreeMap<String, Span>>,
        allocate: &mut dyn FnMut(usize, u64) -> Result<Span, BundleAbiError>,
    ) -> Result<Encoder, BundleAbiError> {
        super::encoders::create(services, weights, allocate)
    }
}

/// [`Target::stacked`]: the stacked entry's ops at each decode and verify
/// use one runs (`stacked::part`), each part placed where its layout copy
/// would put its rows (`stacked::placed`), prepared for `need` of its use.
fn stacked(own: &dyn OwnOps, need: &dyn Fn(OperationUse) -> Need) -> Result<StackedOps, Refused> {
    let mut ops = Vec::new();
    let uses = model_glm5next_contract::precision::uses(stacked::PROJECTIONS[0]);
    for (usage, _) in uses.map_err(Refused::Rule)? {
        let Some(part) = stacked::part(stacked::PROJECTIONS[0], WeightFormat::Bf16, usage)
            .map_err(Refused::Rule)?
        else {
            continue;
        };
        let placed = stacked::placed(usage).map_err(Refused::Rule)?;
        let op = StackedProjection {
            k: part.entry.k,
            parts: placed
                .iter()
                .map(|placed| StackedPart {
                    rows: placed.part.rows,
                    width: placed.part.width,
                    row_stride: placed.part.row_stride,
                })
                .collect(),
        };
        ops.push(Stacked {
            usage,
            parts: placed
                .iter()
                .map(|placed| (placed.id, placed.column_offset))
                .collect(),
            op: own.stacked_projection(&op, usage, need(usage))?,
        });
    }
    Ok(StackedOps {
        projections: &stacked::PROJECTIONS,
        ops,
    })
}

/// [`Target::pool_gate`]: the tiled GEMM of the shared offer of `key`'s
/// BF16 use, whose spans must hold its query's capacity.
fn pool_gate(
    key: &str,
    usage: OperationUse,
    rows: u32,
    spans: PoolGateSpans,
) -> Result<PreparedLaunch, String> {
    use bundle_runtime::dispatch::BufferBindings;
    use kernel_core::contract::StaticQuery;
    use kernels_sm120::gemm;
    use kernels_sm120::precision::Compute;
    let generated = precision::offer(key, WeightFormat::Bf16, usage)?;
    let Compute::Dense(spec) = &generated.offer.handle().compute else {
        return Err(format!("{key} is not a dense GEMM"));
    };
    let StaticQuery::Gemm(query) = spec.key.query else {
        return Err(format!("{key} has no GEMM query"));
    };
    let capacity = u64::from(query.m_capacity);
    for (span, bytes, what) in [
        (
            spans.weight,
            u64::from(query.n) * u64::from(query.k) * 2,
            "weight",
        ),
        (spans.activation, capacity * u64::from(query.k) * 2, "input"),
        (spans.output, capacity * u64::from(query.n) * 2, "output"),
    ] {
        if span.length < bytes {
            return Err(format!(
                "the pooling gate's {what} holds {} of {bytes} bytes",
                span.length
            ));
        }
    }
    let counts = gemm::row_counts(&spec.key, rows);
    let mut bindings = BufferBindings::default();
    let maps = gemm::tensor_maps(
        &spec.key,
        bindings.bind(spans.weight),
        bindings.bind(spans.activation),
    )
    .map_err(|r| r.to_string())?;
    let arguments = gemm::arguments(
        &spec.key,
        counts,
        maps,
        gemm::GemmBuffers {
            output: bindings.bind(spans.output),
            partial: bindings.bind(spans.partial),
            semaphores: bindings.bind(spans.semaphores),
            weight_scales: None,
            activation_scales: None,
            residual: None,
            weight_global_scale: None,
            activation_global_scale: None,
        },
    )
    .map_err(|r| r.to_string())?;
    Ok(PreparedLaunch {
        symbol: spec.symbol.clone(),
        rows,
        plan: gemm::plan(&spec.key, counts).map_err(|r| r.to_string())?,
        arguments,
        bindings,
    })
}

#[cfg(test)]
mod tests {
    use model_glm5next_contract::catalog::DType;
    use model_glm5next_contract::contract::Shape;
    use model_glm5next_contract::quantization::{
        Classes, FP8_BLOCK128, FP8_NVFP4_ROUTED, NVFP4_DENSE_ROUTED, NVFP4_ROUTED,
    };

    use super::*;

    /// An instance storing every class in block FP8 skips exactly the format
    /// families' cubins and the split routed experts' cubins of the other
    /// formats, whatever its side tensor dtypes; every other layout loads
    /// them.
    #[test]
    fn only_layouts_outside_block_fp8_load_the_format_cubins() {
        use kernels_sm120::moe::chained::split::PREFIX;
        use model_glm5next_contract::sm120::engines::{
            MOE_FORMATS_FAMILY, MOE_SPLIT_FAMILY, cubin_specs,
        };
        let mut format = vec![projection_formats::CUBIN.to_string()];
        let mut other = Vec::new();
        for (family, spec) in cubin_specs(Shape::TP4).unwrap() {
            let split_format = family == MOE_SPLIT_FAMILY
                && ["nvfp4", "nvfp4_bw4", "mxfp8", "bf16", "bf16_bw4"]
                    .iter()
                    .any(|suffix| spec.name == format!("{PREFIX}_{suffix}"));
            if family == MOE_FORMATS_FAMILY || split_format {
                format.push(spec.name);
            } else {
                other.push(spec.name);
            }
        }
        assert_eq!(format.len(), 1 + 3 + 5);
        let side_mix = model_glm5next_contract::quantization::find(
            Classes::BLOCK_FP8,
            DType::F32,
            DType::Bf16,
        )
        .unwrap();
        for layout in [&FP8_BLOCK128, side_mix] {
            for name in &format {
                assert!(!loads_format(layout, name), "{name}");
            }
        }
        for layout in [
            &FP8_BLOCK128,
            &NVFP4_ROUTED,
            &NVFP4_DENSE_ROUTED,
            &FP8_NVFP4_ROUTED,
        ] {
            assert!(other.iter().all(|name| loads_format(layout, name)));
        }
        for layout in [&NVFP4_ROUTED, &NVFP4_DENSE_ROUTED, &FP8_NVFP4_ROUTED] {
            assert!(format.iter().all(|name| loads_format(layout, name)));
        }
    }
}
