//! Admission and quiescent epoch reset for optional collective transports.
use bundle_runtime::residency::{self, Verdict};
use runtime_bundle_abi::{BundleAbiError, BundleStatusV1};

// At most two parts of 255 layers: 1022 reductions per prefill run.
// Reset well before the phased protocol's 31-bit tags repeat.
const RESET_RUNS: u64 = (1 << 30) / 1024;

impl super::Instance {
    pub(super) fn admit_bf16_partials(&self) -> Result<(), String> {
        if !self
            .ops
            .reduce
            .as_ref()
            .is_some_and(|ops| ops.bf16_partials())
        {
            return Ok(());
        }
        for (entry, plan) in crate::Kernels::bf16_partial_plans()? {
            for (rank, symbols) in self.symbols.iter().enumerate() {
                let symbol = symbols
                    .get(entry)
                    .ok_or_else(|| format!("rank {rank} lacks BF16-partial entry {entry}"))?;
                let report = residency::query(&self.services, *symbol, plan)
                    .map_err(|error| error.to_string())?;
                let sharing = residency::ranks_on_device(&self.config.create.ordinals, rank);
                if residency::check(&plan, sharing, &report) != Verdict::Fits {
                    return Err(format!("rank {rank} cannot admit {entry}'s resident grid"));
                }
            }
        }
        self.log("BF16 partials admitted: rank-ordered F32 sum of rounded partials");
        Ok(())
    }

    pub(super) fn admit_prefill_reduce(&mut self) {
        if !self
            .ops
            .reduce
            .as_ref()
            .is_some_and(|ops| ops.prefill_prepared())
        {
            return;
        }
        let result = (|| -> Result<(), String> {
            let plans = crate::Kernels::prefill_reduction_plans()?;
            for (entry, plan) in plans {
                for (rank, symbols) in self.symbols.iter().enumerate() {
                    let symbol = symbols
                        .get(entry)
                        .ok_or_else(|| format!("rank {rank} lacks {entry}"))?;
                    let report = residency::query(&self.services, *symbol, plan)
                        .map_err(|error| error.to_string())?;
                    let sharing = residency::ranks_on_device(&self.config.create.ordinals, rank);
                    if residency::check(&plan, sharing, &report) != Verdict::Fits {
                        return Err(format!("rank {rank} cannot admit {entry}'s resident grid"));
                    }
                }
            }
            Ok(())
        })();
        self.log(&format!("prefill BF16 gather admission: {result:?}"));
        self.ops
            .reduce
            .as_mut()
            .expect("prepared reduction")
            .admit_prefill(result.is_ok());
    }

    /// The previous run joined both streams before status and commit. Still
    /// drain explicitly before reusing protocol tags; a failed drain poisons.
    pub(super) fn quiesce_prefill_reduce(&mut self) -> Result<(), BundleAbiError> {
        if self.collectives.is_poisoned() {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
        }
        let Some(state) = self.ops.reduce.as_ref().and_then(|ops| ops.prefill_state()) else {
            return Ok(());
        };
        if self.prefill_reduce_runs < RESET_RUNS {
            self.prefill_reduce_runs += 1;
            return Ok(());
        }
        let result: Result<(), BundleAbiError> = (|| {
            for &(_, _, stream) in &self.ranks {
                self.services.synchronize_stream(stream)?;
            }
            for (rank, buffers) in self.buffers.iter().enumerate() {
                for &span in &buffers.comm[state.clone()] {
                    self.services.zero_device_buffer(self.ranks[rank].2, span)?;
                }
            }
            for &(_, _, stream) in &self.ranks {
                self.services.synchronize_stream(stream)?;
            }
            Ok(())
        })();
        if result.is_err() {
            self.collectives.poison();
        }
        result?;
        self.prefill_reduce_runs = 1;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn reset_prefill_reduce_next_run(
        &mut self,
    ) -> Vec<runtime_bundle_abi::DeviceBufferSpanV1> {
        self.prefill_reduce_runs = RESET_RUNS;
        let range = self.ops.reduce.as_ref().unwrap().prefill_state().unwrap();
        self.buffers
            .iter()
            .flat_map(|buffers| buffers.comm[range.clone()].iter().copied())
            .collect()
    }
}
