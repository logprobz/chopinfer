//! Gather staged features across prompts, then project each stage in batches.
use super::*;
use crate::dspark::device::COMMIT_ROWS;
use crate::dspark::features::Copy;

struct Batch {
    stage: u32,
    rows: u32,
    positions: Vec<u8>,
    slots: Vec<u8>,
    last: Vec<u8>,
    copies: Vec<Vec<Copy>>,
}

impl Instance {
    /// Bind every source before reusing the taps. The target has already
    /// packed them into separate staging, which remains live for retention.
    fn bind_commit_batches(&self, plan: &CommitPlan) -> Result<Vec<Batch>, String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let mut chunks = Vec::new();
        for stage in 0..device.config.stages {
            let mut offsets = [0, u64::from(DECODE_BASE)];
            for item in plan.items() {
                let phase = usize::from(!matches!(item.rows.kind, Kind::Prefill { .. }));
                if item.project_from_staging {
                    let project = if self.skip_prefill && phase == 0 {
                        transfers::staged_retained_projection
                    } else {
                        transfers::staged_projection
                    };
                    chunks.extend(project(
                        device.config,
                        *item,
                        stage,
                        offsets[phase],
                        device.config.window,
                    )?);
                }
                // Short and partially accepted items still occupy their full
                // target row domain in the packed source.
                offsets[phase] += u64::from(item.rows.kind.target_rows()?);
            }
        }
        transfers::staged_batches(device.config, &chunks, COMMIT_ROWS)?
            .into_iter()
            .map(|batch| {
                let mut positions = Vec::new();
                let mut slots = Vec::new();
                let mut last = vec![u32::MAX; device.config.slots as usize];
                for segment in &batch.segments {
                    let source = segment.source;
                    for row in 0..source.rows {
                        positions.push(
                            u32::try_from(source.position + u64::from(row))
                                .map_err(|_| "DSpark commit position exceeds its device word")?,
                        );
                        slots.push(source.slot);
                    }
                    last[source.slot as usize] = *positions.last().expect("nonempty segment");
                }
                let copies = device
                    .ranks
                    .iter()
                    .zip(&self.buffers)
                    .map(|(stored, buffers)| {
                        let features = device.features.bind(
                            buffers
                                .state
                                .named("dspark.features")
                                .ok_or("DSpark features are absent")?,
                            stored.packed,
                        )?;
                        batch
                            .segments
                            .iter()
                            .map(|&segment| features.gather_staged(segment, stored.taps))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Batch {
                    stage: batch.stage,
                    rows: batch.rows,
                    positions: words(positions),
                    slots: words(slots),
                    last: words(last),
                    copies,
                })
            })
            .collect()
    }

    pub(super) fn project_commit_batches(
        &self,
        plan: &CommitPlan,
        mut times: Option<&mut CommitTimes>,
    ) -> Result<(), String> {
        let batches = measure(
            times.as_deref_mut().map(|t| &mut t.projection.lower_us),
            || self.bind_commit_batches(plan),
        )?;
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        for batch in batches {
            if let Some(times) = times.as_deref_mut() {
                times.passes += 1;
                times.rows += u64::from(batch.rows);
            }
            let bound = &device.stages[batch.stage as usize];
            let mut programs = Vec::new();
            for (rank, buffers) in bound.iter().enumerate() {
                let stream = self.ranks[rank].2;
                let stored = &device.ranks[rank];
                let commit = stored
                    .commit
                    .as_ref()
                    .ok_or("DSpark commit storage is absent")?;
                measure(
                    times.as_deref_mut().map(|t| &mut t.projection.upload_us),
                    || {
                        for copy in &batch.copies[rank] {
                            self.services
                                .copy_device_to_device(stream, copy.destination, copy.source)
                                .map_err(|e| format!("DSpark commit gather: {e:?}"))?;
                        }
                        for (span, bytes) in [
                            (
                                commit.inputs.live_tokens,
                                batch.rows.to_le_bytes().as_slice(),
                            ),
                            (commit.inputs.positions, batch.positions.as_slice()),
                            (commit.inputs.token_batch, batch.slots.as_slice()),
                            (stored.last, batch.last.as_slice()),
                        ] {
                            upload(&self.services, stream, span, bytes)?;
                        }
                        for plane in [
                            buffers.scratch.row_status,
                            buffers.scratch.shared_row_status,
                        ] {
                            self.services
                                .zero_device_buffer(stream, rows(plane, batch.rows))
                                .map_err(|e| format!("DSpark commit status clear: {e:?}"))?;
                        }
                        Ok::<_, String>(())
                    },
                )?;
                programs.push(measure(
                    times.as_deref_mut().map(|t| &mut t.projection.lower_us),
                    || {
                        let base = Lowering {
                            prefill_row_origin: None,
                            shape: &device.shape,
                            ops: &device.ops,
                            buffers,
                            peers: bound,
                            inputs: &commit.inputs,
                            rank: rank as u32,
                            kv: self.kv_extent(),
                            head_rows: batch.rows,
                            prefill_hc_fusion: true,
                        };
                        dspark::catch_up(
                            &device.prepared,
                            &base,
                            batch.stage,
                            batch.rows,
                            &CatchUpViews {
                                raw_features: slice(
                                    stored.taps,
                                    0,
                                    u64::from(batch.rows) * device.features.row_bytes(),
                                )?,
                                feature_codes: commit.codes,
                                feature_scales: commit.scales,
                                ring: dspark::rings::committed(
                                    &device.shape,
                                    self.extent,
                                    &buffers.state,
                                    batch.stage,
                                )?,
                                row_slots: commit.inputs.token_batch,
                                last: stored.last,
                            },
                        )
                    },
                )?);
            }
            self.drafter_launch(
                programs,
                batch.rows,
                times.as_deref_mut().map(|t| &mut t.projection),
            )?;
        }
        Ok(())
    }
}
