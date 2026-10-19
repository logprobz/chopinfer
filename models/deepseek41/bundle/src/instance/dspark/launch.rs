//! Catch-up launches and captured full drafts over instance-owned storage.

use super::*;
use crate::buffers::rows;
use crate::dspark::device::{DECODE_BASE, slice};
use crate::dspark::state::{CommitPlan, Projection};
use crate::dspark::transfers::{self, ProjectionChunk};
use crate::lower::Lowering;
use crate::lower::dspark::heads::{HeadViews, Peer, PositionViews};
use crate::lower::dspark::{self, BlockViews, CatchUpViews, DraftMode};
use crate::ops::{Bind, Lowered};
use crate::program::Phase;
use kernel_core::backend::Counts;
use model_deepseek41_contract::own_ops::ring_gather::RingGatherBuffers;
use runtime_bundle_abi::{BundleStreamHandleV1, DeviceBufferSpanV1 as Span};

fn upload(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    span: Span,
    bytes: &[u8],
) -> Result<(), String> {
    services
        .copy_host_to_device(stream, slice(span, 0, bytes.len() as u64)?, bytes)
        .map_err(|e| format!("DSpark upload: {e:?}"))
}

fn words(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

impl Instance {
    fn drafter_launch(&self, lowered: Vec<Vec<Lowered>>, live: u32) -> Result<(), String> {
        let (programs, waits): (Vec<Vec<_>>, Vec<Vec<_>>) = lowered
            .into_iter()
            .map(|rank| {
                rank.into_iter()
                    .map(|op| (op.launch, op.collective))
                    .unzip()
            })
            .unzip();
        let streams = self.ranks.iter().map(|rank| rank.2).collect::<Vec<_>>();
        self.launch_eager(&streams, &programs, &waits)
            .map_err(|e| format!("DSpark dispatch: {e:?}"))?;
        self.drafter_status(live)
    }

    fn drafter_status(&self, live: u32) -> Result<(), String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        for (rank, buffers) in self.buffers.iter().enumerate() {
            for plane in [
                buffers.scratch.row_status,
                buffers.scratch.shared_row_status,
            ] {
                device.ranks[rank].status.save(
                    &self.services,
                    self.ranks[rank].2,
                    rows(plane, live),
                    crate::dspark::status::Kind::Quantized,
                )?;
            }
        }
        Ok(())
    }

    pub(in crate::instance) fn check_dspark_status(&self) -> Result<(), String> {
        if let Some(device) = &self.dspark {
            for (rank, stored) in device.ranks.iter().enumerate() {
                stored
                    .status
                    .check(&self.services, self.ranks[rank].2, rank)?;
            }
        }
        Ok(())
    }

    fn project_chunk(&self, pass: ProjectionChunk) -> Result<(), String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let bound = &device.stages[pass.stage as usize];
        let mut programs = Vec::new();
        let positions = words((0..pass.rows).map(|i| (pass.position + u64::from(i)) as u32));
        let slots = words(std::iter::repeat_n(pass.slot, pass.rows as usize));
        let mut last = vec![u32::MAX; device.config.slots as usize];
        last[pass.slot as usize] = (pass.position + u64::from(pass.rows) - 1) as u32;
        for (rank, buffers) in bound.iter().enumerate() {
            let stream = self.ranks[rank].2;
            let stored = &device.ranks[rank];
            upload(
                &self.services,
                stream,
                stored.inputs.live_tokens,
                &pass.rows.to_le_bytes(),
            )?;
            upload(&self.services, stream, stored.inputs.positions, &positions)?;
            upload(&self.services, stream, stored.inputs.token_batch, &slots)?;
            upload(
                &self.services,
                stream,
                stored.last,
                &words(last.iter().copied()),
            )?;
            self.services
                .zero_device_buffer(stream, rows(buffers.scratch.row_status, pass.rows))
                .map_err(|e| format!("DSpark status clear: {e:?}"))?;
            self.services
                .zero_device_buffer(stream, rows(buffers.scratch.shared_row_status, pass.rows))
                .map_err(|e| format!("DSpark shared status clear: {e:?}"))?;
            let features = device.features.bind(
                buffers
                    .state
                    .named("dspark.features")
                    .ok_or("DSpark features are absent")?,
                stored.packed,
            )?;
            let base = Lowering {
                prefill_row_origin: None,
                shape: &device.shape,
                ops: &device.ops,
                buffers,
                peers: bound,
                inputs: &stored.inputs,
                rank: rank as u32,
                kv: self.kv_extent(),
                head_rows: pass.rows,
            };
            programs.push(dspark::catch_up(
                &device.prepared,
                &base,
                pass.stage,
                pass.rows,
                &CatchUpViews {
                    raw_features: features.projection_source(pass)?,
                    feature_codes: stored.feature_codes,
                    feature_scales: stored.feature_scales,
                    ring: dspark::rings::committed(
                        &device.shape,
                        self.extent,
                        &buffers.state,
                        pass.stage,
                    )?,
                    row_slots: stored.inputs.token_batch,
                    last: stored.last,
                },
            )?);
        }
        self.drafter_launch(programs, pass.rows)
    }

    pub(super) fn dspark_project(&self, pass: Projection) -> Result<(), String> {
        let config = self.dspark.as_ref().ok_or("DSpark is absent")?.config;
        for chunk in transfers::retained_projection(config, pass, config.window)? {
            self.project_chunk(chunk)?;
        }
        Ok(())
    }

    pub(super) fn dspark_commit_features(&self, plan: &CommitPlan) -> Result<(), String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let mut offsets = [0, u64::from(DECODE_BASE)];
        let mut copies = Vec::new();
        // Bind every retention copy before issuing any physical mutation.
        for item in plan.items() {
            let phase = usize::from(!matches!(item.rows.kind, Kind::Prefill { .. }));
            let retention = transfers::retain_copies(device.config, *item, offsets[phase])?;
            for (rank, buffers) in self.buffers.iter().enumerate() {
                let features = device.features.bind(
                    buffers
                        .state
                        .named("dspark.features")
                        .ok_or("DSpark features are absent")?,
                    device.ranks[rank].packed,
                )?;
                copies.extend(
                    features
                        .retain(&retention)?
                        .into_iter()
                        .map(|copy| (rank, copy)),
                );
            }
            offsets[phase] += u64::from(item.rows.kind.target_rows()?);
        }
        let mut offsets = [0, u64::from(DECODE_BASE)];
        for item in plan.items() {
            let phase = usize::from(!matches!(item.rows.kind, Kind::Prefill { .. }));
            if item.project_from_staging {
                for stage in 0..device.config.stages {
                    let project = if self.skip_prefill && phase == 0 {
                        transfers::staged_retained_projection
                    } else {
                        transfers::staged_projection
                    };
                    for pass in project(
                        device.config,
                        *item,
                        stage,
                        offsets[phase],
                        device.config.window,
                    )? {
                        self.project_chunk(pass)?;
                    }
                }
            }
            offsets[phase] += u64::from(item.rows.kind.target_rows()?);
        }
        for (rank, copy) in copies {
            self.services
                .copy_device_to_device(self.ranks[rank].2, copy.destination, copy.source)
                .map_err(|e| format!("DSpark feature retention: {e:?}"))?;
        }
        // Copies remain ordered on each rank's stream. Any following draft,
        // prefix copy or slot reuse uses that same stream. Staged projections
        // still validate their saved status before the commit can succeed.
        self.check_dspark_status()
    }

    pub(super) fn dspark_draft(
        &mut self,
        inputs: &[crate::dspark::execution::DraftInput],
    ) -> Result<(), String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let g = device.shape.geometry;
        let meta = dspark::metadata(g, inputs, DraftMode::Greedy)?;
        let items = inputs.len() as u32;
        let live = items * BLOCK_ROWS;
        for (rank, stored) in device.ranks.iter().enumerate() {
            let stream = self.ranks[rank].2;
            let arrays = [
                (stored.inputs.token_ids, words(meta.tokens.iter().copied())),
                (
                    stored.inputs.positions,
                    words(meta.positions.iter().map(|&v| v as u32)),
                ),
                (
                    stored.inputs.token_batch,
                    words(meta.row_items.iter().map(|&v| v as u32)),
                ),
                (
                    stored.source_slots,
                    words(meta.slots.iter().map(|&v| v as u32)),
                ),
                (stored.first, words(meta.first.iter().map(|&v| v as u32))),
                (stored.last, words(meta.last.iter().map(|&v| v as u32))),
                (
                    stored.item_slots,
                    words(meta.staged_slots.iter().map(|&v| v as u32)),
                ),
                (stored.keys, words(meta.keys.iter().map(|&v| v as u32))),
                (
                    stored.predecessor_ids,
                    words(inputs.iter().map(|input| input.bonus)),
                ),
            ];
            for (span, bytes) in arrays {
                upload(&self.services, stream, span, &bytes)?;
            }
            upload(
                &self.services,
                stream,
                stored.inputs.live_tokens,
                &live.to_le_bytes(),
            )?;
            upload(
                &self.services,
                stream,
                stored.live_items,
                &items.to_le_bytes(),
            )?;
            upload(
                &self.services,
                stream,
                self.buffers[rank].scratch.bias_select,
                &meta.bias_select,
            )?;
            self.services
                .zero_device_buffer(stream, rows(self.buffers[rank].scratch.row_status, live))
                .map_err(|e| format!("DSpark status clear: {e:?}"))?;
            self.services
                .zero_device_buffer(
                    stream,
                    rows(self.buffers[rank].scratch.shared_row_status, live),
                )
                .map_err(|e| format!("DSpark shared status clear: {e:?}"))?;
        }
        self.replay_drafter(items)?;
        self.drafter_status(live)
    }

    pub(super) fn drafter_programs(&self, items: u32) -> Result<Vec<Vec<Lowered>>, String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let live = items * BLOCK_ROWS;
        let phase = Phase::Verify { rows: live, items };
        let mut all: Vec<Vec<Lowered>> = (0..self.ranks.len()).map(|_| Vec::new()).collect();
        for stage in 0..device.config.stages {
            let bound = &device.stages[stage as usize];
            for (rank, buffers) in bound.iter().enumerate() {
                let stored = &device.ranks[rank];
                let main =
                    dspark::rings::committed(&device.shape, self.extent, &buffers.state, stage)?;
                let mut program = Vec::new();
                for (op, source, destination) in [
                    (&device.ring_gathers[0], main.codes, stored.window.codes),
                    (&device.ring_gathers[1], main.scales, stored.window.scales),
                ] {
                    let mut bind = Bind::default();
                    let buffers = RingGatherBuffers {
                        source: bind.view(source),
                        source_slots: bind.view(stored.source_slots),
                        live_items: bind.view(stored.live_items),
                        destination: bind.view(destination),
                    };
                    program.extend(
                        bind.launch(op, &buffers, Counts::rows(items))
                            .map_err(|e| e.to_string())?,
                    );
                }
                let base = Lowering {
                    prefill_row_origin: None,
                    shape: &device.shape,
                    ops: &device.ops,
                    buffers,
                    peers: bound,
                    inputs: &stored.inputs,
                    rank: rank as u32,
                    kv: self.kv_extent(),
                    head_rows: live,
                };
                if stage == 0 {
                    program.extend(dspark::initialize(&base, phase)?);
                }
                program.extend(dspark::block(
                    &device.prepared,
                    &base,
                    stage,
                    phase,
                    BlockViews {
                        window: dspark::Window {
                            sequences: items,
                            ..stored.window
                        },
                        item_slots: stored.item_slots,
                        first: stored.first,
                        last: stored.last,
                        keys: stored.keys,
                    },
                )?);
                all[rank].extend(program);
            }
        }
        let stem = format!("mtp.{}", device.config.stages - 1);
        let programs = self
            .buffers
            .iter()
            .zip(&device.ranks)
            .map(|(buffers, stored)| {
                let scratch = &buffers.scratch;
                device
                    .heads
                    .vocabulary(
                        live,
                        &HeadViews {
                            streams: rows(scratch.streams, live),
                            pre: rows(crate::lower::pre(scratch, 1), live),
                            hidden_f32: rows(scratch.head_collapsed, live),
                            norm_weight: buffers.view(&format!("{stem}.norm.weight"))?,
                            normalized_f32: rows(scratch.normed_wide, live),
                            vocabulary_weight: buffers.view("head.weight")?,
                            projection_state: crate::ops::exact_f32::state(buffers, "head.weight"),
                            logits: slice(
                                stored.block_logits,
                                0,
                                u64::from(live) * u64::from(device.shape.vocab_rows()) * 4,
                            )?,
                            live_rows: stored.inputs.live_tokens,
                            status: rows(scratch.row_status, live),
                        },
                    )
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, String>>()?;
        for (rank, program) in all.iter_mut().zip(programs) {
            rank.extend(program);
        }
        let peers = device
            .ranks
            .iter()
            .map(|stored| Peer {
                embedding: stored.embedding,
                reduce_state: stored.reduce_state.clone(),
                gathered: stored.gathered,
                gather_state: stored.gather_state.clone(),
            })
            .collect::<Vec<_>>();
        for position in 0..BLOCK_ROWS {
            let programs = self
                .buffers
                .iter()
                .zip(&device.ranks)
                .enumerate()
                .map(|(rank, (buffers, stored))| {
                    device.heads.position(
                        rank as u32,
                        position,
                        items,
                        &PositionViews {
                            predecessor_ids: stored.predecessor_ids,
                            embedding_weight: buffers
                                .view(&format!("{stem}.markov_head.embed.weight"))?,
                            embedding_partial: stored.embedding_partial,
                            embedding_wide: stored.embedding_wide,
                            embedding: stored.embedding,
                            block_logits: stored.block_logits,
                            position_rows: stored.position_rows[position as usize],
                            position_logits: stored.position_logits,
                            markov_weight: buffers
                                .view(&format!("{stem}.markov_head.head.weight"))?,
                            hidden_f32: rows(buffers.scratch.head_collapsed, live),
                            confidence_weight: buffers
                                .view(&format!("{stem}.confidence_head.proj.weight"))?,
                            confidence: stored.confidence,
                            pairs: stored.pairs,
                            gathered: stored.gathered,
                            live_width: stored.live_width,
                            live_items: stored.live_items,
                            draft_ids: stored.draft_ids,
                        },
                        &peers,
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            for (rank, program) in all.iter_mut().zip(programs) {
                rank.extend(program);
            }
        }
        Ok(all)
    }
}
