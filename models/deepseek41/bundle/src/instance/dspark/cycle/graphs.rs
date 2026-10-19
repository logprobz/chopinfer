//! Stable target graph pieces around the first mapped-row consumer.
use super::Instance;
use crate::program::Stage;
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::graphs::GraphKey;
use bundle_runtime::pipeline::{self, PoisonOnUnwind, PreparedLaunch, Unit};
use runtime_bundle_abi::{
    BundleAbiError, BundleStatusV1, BundleStreamHandleV1, DeviceBufferSpanV1,
};

impl Instance {
    pub(crate) fn launch_dspark_target(
        &mut self,
        streams: &[BundleStreamHandleV1],
        graph: Option<(u32, u32)>,
        stages: &[Vec<Stage>],
        live_tokens: &[DeviceBufferSpanV1],
        mut programs: Vec<Vec<PreparedLaunch>>,
        mut collectives: Vec<Vec<bool>>,
    ) -> Result<(), DispatchError> {
        if self.collectives.is_poisoned() {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState).into());
        }
        let ready = graph.is_some_and(|(bucket, items)| {
            let tail = self.dspark.as_ref().and_then(|d| d.cycle.tail.as_ref());
            (0..streams.len()).all(|rank| {
                let key = GraphKey {
                    rank: rank as u32,
                    bucket,
                    items,
                };
                self.graphs.contains(key) && tail.is_some_and(|tail| tail.contains(key))
            })
        });
        if !ready {
            if programs.len() != streams.len()
                || collectives.len() != streams.len()
                || stages.len() != streams.len()
                || live_tokens.len() != streams.len()
            {
                return Err(DispatchError::InvalidPlan);
            }
            let mut boundaries = Vec::with_capacity(streams.len());
            let mut suffix = Vec::with_capacity(streams.len());
            let mut suffix_waits = Vec::with_capacity(streams.len());
            for rank in 0..streams.len() {
                if stages[rank].len() != programs[rank].len()
                    || collectives[rank].len() != programs[rank].len()
                {
                    return Err(DispatchError::InvalidPlan);
                }
                let boundary = match stages[rank]
                    .iter()
                    .position(|stage| matches!(stage, Stage::EngramRows(_)))
                {
                    Some(at) => at,
                    None if self.partial => 0,
                    None => {
                        self.log("DSpark target has no mapped-row consumer");
                        return Err(DispatchError::InvalidPlan);
                    }
                };
                boundaries.push(boundary);
                suffix.push(programs[rank].split_off(boundary));
                suffix_waits.push(collectives[rank].split_off(boundary));
            }
            for (ops, waits) in [(&programs, &collectives), (&suffix, &suffix_waits)] {
                if let Err(error) = pipeline::check_collectives(ops, waits) {
                    self.log(&error);
                    return Err(DispatchError::InvalidPlan);
                }
            }
            let Some((bucket, items)) = graph else {
                self.launch_eager(streams, &programs, &collectives)?;
                self.wait_dspark_rows()?;
                return self.launch_eager(streams, &suffix, &suffix_waits);
            };
            for (rank, &stream) in streams.iter().enumerate() {
                let key = GraphKey {
                    rank: rank as u32,
                    bucket,
                    items,
                };
                let lanes = self
                    .shared_overlap
                    .serves(bucket)
                    .then(|| self.side_lanes.get(rank).copied())
                    .flatten();
                if self.shared_overlap.serves(bucket) && lanes.is_none() {
                    return Err(DispatchError::InvalidPlan);
                }
                let scratch = &self.buffers[rank].scratch;
                let immutable = [
                    live_tokens[rank],
                    scratch.normed_codes,
                    scratch.normed_scales,
                ];
                let prepare = |program: &[PreparedLaunch], stages: &[Stage]| {
                    if lanes.is_some() {
                        crate::shared_overlap::checked_units(stages, program, &immutable).map_err(
                            |error| {
                                self.log(&error);
                                DispatchError::InvalidPlan
                            },
                        )
                    } else {
                        Ok(Vec::new())
                    }
                };
                let prefix_units = prepare(&programs[rank], &stages[rank][..boundaries[rank]])?;
                let suffix_units = prepare(&suffix[rank], &stages[rank][boundaries[rank]..])?;
                let issue = |program: &[PreparedLaunch],
                             units: &[Vec<crate::shared_overlap::Op>]| {
                    if let Some(lanes) = lanes {
                        units.iter().try_for_each(|unit| {
                            lanes.issue(&self.services, &self.symbols[rank], program, unit)
                        })
                    } else {
                        pipeline::launch_all(&self.services, stream, &self.symbols[rank], program)
                    }
                };
                // The Engram boundary is outside every shared-expert fork.
                // Each whole prefix/suffix capture retains those fork/join units.
                self.graphs
                    .capture::<DispatchError>(&self.services, stream, key, 1, |_| {
                        issue(&programs[rank], &prefix_units)
                    })?;
                self.dspark
                    .as_mut()
                    .and_then(|d| d.cycle.tail.as_mut())
                    .ok_or(DispatchError::InvalidPlan)?
                    .capture::<DispatchError>(&self.services, stream, key, 1, |_| {
                        issue(&suffix[rank], &suffix_units)
                    })?;
            }
        }
        let key = graph.ok_or(DispatchError::InvalidPlan)?;
        let _unwinding = PoisonOnUnwind(&self.collectives);
        let mut unit = Unit::default();
        let prefix = unit.record(
            &self.collectives,
            self.graphs
                .replay_piece_major(&self.services, streams, key, |_| {})
                .map_err(DispatchError::from),
        )?;
        if !prefix {
            return Err(DispatchError::InvalidPlan);
        }
        self.wait_dspark_rows()?;
        let tail = self
            .dspark
            .as_ref()
            .and_then(|d| d.cycle.tail.as_ref())
            .ok_or(DispatchError::InvalidPlan)?;
        let replayed = unit.record(
            &self.collectives,
            tail.replay_piece_major(&self.services, streams, key, |_| {})
                .map_err(DispatchError::from),
        )?;
        if !replayed {
            return Err(DispatchError::InvalidPlan);
        }
        Ok(())
    }
}
