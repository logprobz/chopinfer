//! Publish layer-20 candidates with the qualified fused producer.
use crate::ops::Lowered;
use bundle_runtime::pipeline::PreparedLaunch;
use kernel_core::{backend::Launch, contract::Argument, contract::Refusal};
use kernels_sm120::pooled_indexer::position_candidates_cached as cached;
use model_deepseek41_contract::geometry::FLASH;

impl crate::Kernels {
    /// Replace the three-stage valid producer of one bounded prefill batch
    /// with the fused publisher. Its public outputs keep their spans; only
    /// the private maxima and shard workspace are no longer written.
    pub(crate) fn cached_prefill_candidates(calls: Vec<Lowered>) -> Result<Vec<Lowered>, Refusal> {
        let [scores, local, merge] = calls.as_slice() else {
            return Err(Refusal(
                "fused candidates require the three-stage valid producer",
            ));
        };
        if calls.iter().any(|call| call.collective) {
            return Err(Refusal("fused candidates require local launches"));
        }
        // One bind serves the producer, so a buffer argument names the same
        // span in every launch. Check the scores the fused launch moves.
        let Some(Argument::DeviceBuffer(input)) = scores.launch.arguments.0.first() else {
            return Err(Refusal("fused candidate scores are not a direct buffer"));
        };
        let span = scores.launch.bindings.get(*input).ok();
        if span.is_none() || span != merge.launch.bindings.get(*input).ok() {
            return Err(Refusal("fused candidate scores change binding"));
        }
        let launch = |call: &Lowered| Launch {
            symbol: call.launch.symbol.clone(),
            rows: call.launch.rows,
            plan: call.launch.plan,
            arguments: call.launch.arguments.clone(),
            collective: false,
            persistent: false,
        };
        let fused = cached::remap(
            &super::indexer::candidates(&FLASH),
            &[launch(scores), launch(local), launch(merge)],
        )?;
        Ok(vec![Lowered {
            launch: PreparedLaunch {
                symbol: fused.symbol,
                rows: fused.rows,
                plan: fused.plan,
                arguments: fused.arguments,
                bindings: merge.launch.bindings.clone(),
            },
            collective: false,
        }])
    }
}
