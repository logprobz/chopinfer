//! Target graph pieces join each mapped layer immediately before its consumer.
mod audit;
use super::Instance;
use crate::program::Stage;
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::graphs::{GraphKey, GraphReservation};
use bundle_runtime::pipeline::{self, PoisonOnUnwind, PreparedLaunch, Unit};
use runtime_bundle_abi::{
    BundleAbiError, BundleStatusV1, BundleStreamHandleV1, DeviceBufferSpanV1,
};

impl Instance {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_dspark_target(
        &mut self,
        streams: &[BundleStreamHandleV1],
        graph: Option<(u32, u32)>,
        mut reservation: Option<&mut GraphReservation>,
        stages: &[Vec<Stage>],
        live_tokens: &[DeviceBufferSpanV1],
        programs: Vec<Vec<PreparedLaunch>>,
        collectives: Vec<Vec<bool>>,
    ) -> Result<(), DispatchError> {
        if self.collectives.is_poisoned() {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState).into());
        }
        let layers = &self
            .dspark
            .as_ref()
            .ok_or(DispatchError::InvalidPlan)?
            .cycle
            .layers;
        if layers.is_empty() {
            return Err(DispatchError::InvalidPlan);
        }
        let pieces = layers.len() + 1;
        let ready = graph.is_some_and(|(bucket, items)| {
            (0..streams.len()).all(|rank| {
                let key = GraphKey {
                    rank: rank as u32,
                    bucket,
                    items,
                };
                self.graphs.contains(key) && layers.iter().all(|layer| layer.graphs.contains(key))
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
            let mut ranges = Vec::with_capacity(streams.len());
            for rank in 0..streams.len() {
                if stages[rank].len() != programs[rank].len()
                    || collectives[rank].len() != programs[rank].len()
                {
                    return Err(DispatchError::InvalidPlan);
                }
                let sections =
                    audit::sections(&stages[rank], layers.iter().map(|l| l.layer), self.partial)?;
                // Pending DMA owns its layer slices. No earlier target piece,
                // including its shared-expert side stream, may bind those bytes.
                for (index, section) in sections.iter().take(layers.len()).enumerate() {
                    audit::before_join(&programs[rank][..section.end], &layers[index..])?;
                }
                ranges.push(sections);
            }
            for piece in 0..pieces {
                let ops = (0..streams.len())
                    .map(|rank| &programs[rank][ranges[rank][piece].clone()])
                    .collect::<Vec<_>>();
                let waits = (0..streams.len())
                    .map(|rank| &collectives[rank][ranges[rank][piece].clone()])
                    .collect::<Vec<_>>();
                if let Err(error) = pipeline::check_collectives(&ops, &waits) {
                    self.log(&error);
                    return Err(DispatchError::InvalidPlan);
                }
            }
            let Some((bucket, items)) = graph else {
                for piece in 0..pieces {
                    if piece > 0 {
                        self.wait_dspark_rows(piece - 1)?;
                    }
                    let ops = (0..streams.len())
                        .map(|rank| &programs[rank][ranges[rank][piece].clone()])
                        .collect::<Vec<_>>();
                    let waits = (0..streams.len())
                        .map(|rank| &collectives[rank][ranges[rank][piece].clone()])
                        .collect::<Vec<_>>();
                    self.launch_eager(streams, &ops, &waits)?;
                }
                return Ok(());
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
                for (piece, range) in ranges[rank].iter().enumerate() {
                    let program = &programs[rank][range.clone()];
                    let units = if lanes.is_some() {
                        crate::shared_overlap::checked_units(
                            &stages[rank][range.clone()],
                            program,
                            &immutable,
                        )
                        .map_err(|error| {
                            self.log(&error);
                            DispatchError::InvalidPlan
                        })?
                    } else {
                        Vec::new()
                    };
                    let issue = |_| {
                        if let Some(lanes) = lanes {
                            units.iter().try_for_each(|unit| {
                                lanes.issue(&self.services, &self.symbols[rank], program, unit)
                            })
                        } else {
                            pipeline::launch_all(
                                &self.services,
                                stream,
                                &self.symbols[rank],
                                program,
                            )
                        }
                    };
                    // Every piece contains whole shared-expert fork/join regions.
                    let cache = if piece == 0 {
                        &mut self.graphs
                    } else {
                        &mut self
                            .dspark
                            .as_mut()
                            .ok_or(DispatchError::InvalidPlan)?
                            .cycle
                            .layers[piece - 1]
                            .graphs
                    };
                    cache.capture_reserved::<DispatchError>(
                        &self.services,
                        stream,
                        key,
                        1,
                        reservation
                            .as_deref_mut()
                            .ok_or(DispatchError::InvalidPlan)?,
                        issue,
                    )?;
                }
            }
        }
        let key = graph.ok_or(DispatchError::InvalidPlan)?;
        let _unwinding = PoisonOnUnwind(&self.collectives);
        let mut unit = Unit::default();
        for piece in 0..pieces {
            if piece > 0 {
                self.wait_dspark_rows(piece - 1)?;
            }
            let cache = if piece == 0 {
                &self.graphs
            } else {
                &self
                    .dspark
                    .as_ref()
                    .ok_or(DispatchError::InvalidPlan)?
                    .cycle
                    .layers[piece - 1]
                    .graphs
            };
            let replayed = unit.record(
                &self.collectives,
                cache
                    .replay_piece_major(&self.services, streams, key, |_| {})
                    .map_err(DispatchError::from),
            )?;
            if !replayed {
                return Err(DispatchError::InvalidPlan);
            }
        }
        Ok(())
    }
}
