//! Stable target graph pieces around the first mapped-row consumer.
use super::Instance;
use crate::program::Stage;
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::graphs::GraphKey;
use bundle_runtime::pipeline::{self, PoisonOnUnwind, PreparedLaunch, Unit};
use runtime_bundle_abi::{BundleAbiError, BundleStatusV1, BundleStreamHandleV1};

impl Instance {
    pub(crate) fn launch_dspark_target(
        &mut self,
        streams: &[BundleStreamHandleV1],
        graph: Option<(u32, u32)>,
        stages: &[Vec<Stage>],
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
            {
                return Err(DispatchError::InvalidPlan);
            }
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
            let tail = self
                .dspark
                .as_mut()
                .and_then(|d| d.cycle.tail.as_mut())
                .ok_or(DispatchError::InvalidPlan)?;
            for (rank, &stream) in streams.iter().enumerate() {
                let key = GraphKey {
                    rank: rank as u32,
                    bucket,
                    items,
                };
                // One capture unit per side. Both paths use the same split,
                // including mixed prefills whose rows were gathered on CPU.
                self.graphs
                    .capture::<DispatchError>(&self.services, stream, key, 1, |_| {
                        pipeline::launch_all(
                            &self.services,
                            stream,
                            &self.symbols[rank],
                            &programs[rank],
                        )
                    })?;
                tail.capture::<DispatchError>(&self.services, stream, key, 1, |_| {
                    pipeline::launch_all(&self.services, stream, &self.symbols[rank], &suffix[rank])
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
