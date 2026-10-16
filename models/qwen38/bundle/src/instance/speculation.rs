//! Speculation in the instance: the MTP drafter's runs, the target X rows
//! it retains, the drafts it emits, the checkpoint ownership exchanges at
//! verify commit, and the retained X copies.

use super::*;
use crate::drafter::{Drafter, Plan, Retain, Run};
use crate::lower::DrafterInputs;
use crate::scratch::{DRAFTER_ROWS, RETAINED_ROWS};

const ALIGN: u64 = 256;

/// The items an instance with the drafter serves at most, which its runs'
/// arenas, block tables and slot table hold.
const DRAFTER: Items = match Items::new(DRAFT_ITEMS) {
    Some(items) => items,
    None => panic!("the drafter serves at least one item"),
};

/// One drafter run's inputs beside its phase arena: per row its X and
/// token sources, and per draft row its drafting item and verify row.
#[derive(Clone, Copy)]
struct Extras;

impl Extras {
    const SOURCES: u64 = 0;
    const TOKEN_SOURCES: u64 = (4 * DRAFTER_ROWS).next_multiple_of(ALIGN);
    const SLOTS: u64 = Self::TOKEN_SOURCES + (4 * DRAFTER_ROWS).next_multiple_of(ALIGN);
    const TARGETS: u64 = Self::SLOTS + ALIGN;
    const BYTES: u64 = Self::TARGETS + ALIGN;

    fn pack(run: &Run) -> Vec<u8> {
        let mut bytes = vec![0_u8; Self::BYTES as usize];
        let mut put = |offset: u64, index: usize, value: i32| {
            let at = offset as usize + 4 * index;
            bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        };
        let (sources, tokens) = run.codes();
        for (row, (&source, &token)) in sources.iter().zip(&tokens).enumerate() {
            put(Self::SOURCES, row, source);
            put(Self::TOKEN_SOURCES, row, token);
        }
        for (index, draft) in run.drafts.iter().enumerate() {
            put(Self::SLOTS, index, draft.item as i32);
            put(Self::TARGETS, index, draft.target);
        }
        bytes
    }

    fn spans(arena: DeviceBufferSpanV1) -> [DeviceBufferSpanV1; 4] {
        let span = |offset: u64, bytes: u64| {
            DeviceBufferSpanV1::new(arena.buffer, arena.offset + offset, bytes)
        };
        [
            span(Self::SOURCES, 4 * DRAFTER_ROWS),
            span(Self::TOKEN_SOURCES, 4 * DRAFTER_ROWS),
            span(Self::SLOTS, 4 * u64::from(DRAFT_ITEMS)),
            span(Self::TARGETS, 4 * u64::from(DRAFT_ITEMS)),
        ]
    }
}

/// Index lists a copy uploads: `[source ids][destination ids][count]`.
pub(super) const INDEX_ENTRIES: u64 = 256;
const INDEX_REGION: u64 = (4 * (2 * INDEX_ENTRIES + 1)).next_multiple_of(ALIGN);
/// Regions of the index arena: retained and staged X rows and prefill
/// anchors. Former checkpoint-copy regions stay reserved.
const RETAIN_REGIONS: [usize; 2] = [1, 2];
const STAGE_REGION: usize = 3;
const ANCHOR_REGION: usize = 5;
pub(super) const INDEX_ARENA_BYTES: u64 = 6 * INDEX_REGION;

struct DrafterRank {
    arena: PhaseArena,
    side: DeviceBufferSpanV1,
    extras: DeviceBufferSpanV1,
}

pub(super) struct DrafterRuntime {
    pub(super) state: Drafter,
    ranks: Vec<DrafterRank>,
    block_tables: BlockTableMirror,
    /// Graphs of catch-up runs, and of runs that draft.
    graphs: [GraphCache; 2],
    /// `CHOPINFER_QWEN38_MTP_EAGER=1` runs every drafter run eagerly.
    eager: bool,
}

impl DrafterRuntime {
    pub(super) fn allocate(
        shape: Shape,
        services: &HostServicesV1,
        pools: &[(
            runtime_bundle_abi::BundleMemoryPoolHandleV1,
            BundleStreamHandleV1,
        )],
        target_graphs: &GraphCache,
        items: Items,
        mtp_eager: bool,
        held: &mut Vec<Held>,
    ) -> Result<Self, BundleAbiError> {
        let mut ranks = Vec::new();
        for &(pool, stream) in pools {
            let mut allocate = |bytes| allocate_span(services, pool, stream, bytes, held);
            ranks.push(DrafterRank {
                arena: PhaseArena {
                    arena: allocate(InputLayout::prefill_capacity_within::<Qwen>(items))?,
                    block_tables: allocate(BlockTableMirror::device_bytes(
                        shape.model_limits(),
                        items,
                    ))?,
                },
                side: allocate(SideLayout::for_shape(shape, PREFILL_TOKENS, items.capacity()).end)?,
                extras: allocate(Extras::BYTES)?,
            });
        }
        Ok(Self {
            state: Drafter::new(items.get()),
            ranks,
            block_tables: BlockTableMirror::new(shape.model_limits(), items),
            graphs: std::array::from_fn(|_| GraphCache::catch_up_sharing(target_graphs)),
            eager: super::eager() || mtp_eager,
        })
    }

    pub(super) fn release_graphs(
        &mut self,
        services: &HostServicesV1,
    ) -> Result<(), BundleAbiError> {
        for graphs in &mut self.graphs {
            graphs.release_all(services)?;
        }
        Ok(())
    }
}

/// How one drafter run is prepared: a graph bucket, or eagerly at its live
/// counts.
struct Prepared {
    phase: Phase,
    rows: u32,
    items: Vec<ItemRows>,
    head_rows: u32,
    context: u32,
    layout: InputLayout,
    side: SideLayout,
}

impl Prepared {
    fn new(
        shape: Shape,
        run: &Run,
        inputs: &PhaseInputs,
        eager: bool,
        served: Items,
    ) -> Result<Self, BundleStatusV1> {
        let rows = run.rows();
        let drafts = run.drafts.len() as u32;
        if !eager && rows <= DRAFT_ITEMS {
            let bucket = |count| DRAFTER.bucket(count).ok_or(BundleStatusV1::InternalError);
            let (rows, items) = (bucket(rows)?, bucket(inputs.live_batch())?);
            return Ok(Self {
                phase: Phase::Verify { rows, items },
                rows,
                items: (0..items).map(|_| ItemRows { start: 0, rows: 0 }).collect(),
                // A drafting item has one draft row per run.
                head_rows: if drafts > 0 { items } else { 0 },
                context: shape.context_tokens() as u32,
                layout: InputLayout::within(DRAFT_ITEMS, DRAFT_ITEMS, served),
                side: SideLayout::for_shape(shape, DRAFT_ITEMS, served.capacity()),
            });
        }
        if u64::from(rows) > DRAFTER_ROWS {
            return Err(BundleStatusV1::InternalError);
        }
        let layout = InputLayout::prefill_within::<Qwen>(inputs, drafts as usize, served);
        Ok(Self {
            phase: Phase::Prefill,
            rows,
            items: inputs
                .q_start
                .iter()
                .zip(&inputs.q_len)
                .map(|(&start, &rows)| ItemRows {
                    start: start as u32,
                    rows: rows as u32,
                })
                .collect(),
            head_rows: drafts,
            context: inputs.seq_lens.iter().copied().max().unwrap_or(1) as u32,
            side: SideLayout::for_shape(shape, layout.tokens as u32, served.capacity()),
            layout,
        })
    }

    /// The graph cache (catch-up runs, or runs that draft) and key of a
    /// graph run.
    fn graph(&self) -> Option<(usize, (u32, u32))> {
        match self.phase {
            Phase::Verify { rows, items } => Some((usize::from(self.head_rows > 0), (rows, items))),
            _ => None,
        }
    }
}

impl Instance {
    /// Uploads `src` and `dst` ids to index region `region` of every rank.
    fn upload_indices(
        &self,
        region: usize,
        src: &[i32],
        dst: &[i32],
    ) -> Result<[Vec<DeviceBufferSpanV1>; 3], BundleStatusV1> {
        if src.len() != dst.len() || src.len() as u64 > INDEX_ENTRIES {
            return Err(BundleStatusV1::InternalError);
        }
        let mut bytes = vec![0_u8; INDEX_REGION as usize];
        for (index, (&from, &to)) in src.iter().zip(dst).enumerate() {
            bytes[4 * index..4 * index + 4].copy_from_slice(&from.to_le_bytes());
            let at = 4 * (INDEX_ENTRIES as usize + index);
            bytes[at..at + 4].copy_from_slice(&to.to_le_bytes());
        }
        let at = 8 * INDEX_ENTRIES as usize;
        bytes[at..at + 4].copy_from_slice(&(src.len() as u32).to_le_bytes());
        let mut spans: [Vec<DeviceBufferSpanV1>; 3] = Default::default();
        for rank in 0..self.shape.ranks.count() {
            let arena = self.index_arenas[rank];
            let base = arena.offset + region as u64 * INDEX_REGION;
            let region = DeviceBufferSpanV1::new(arena.buffer, base, INDEX_REGION);
            upload_bytes(&self.services, self.streams[rank], region, &bytes).map_err(status)?;
            let span = |offset: u64, length: u64| {
                DeviceBufferSpanV1::new(arena.buffer, base + offset, length)
            };
            spans[0].push(span(0, 4 * INDEX_ENTRIES));
            spans[1].push(span(4 * INDEX_ENTRIES, 4 * INDEX_ENTRIES));
            spans[2].push(span(8 * INDEX_ENTRIES, 4));
        }
        Ok(spans)
    }

    /// Copies X row entry `src[i]` of `from` to entry `dst[i]` of `to` on
    /// every rank, through index region `region`.
    fn copy_indexed(
        &self,
        region: usize,
        buffers: impl Fn(&RankScratch) -> Option<(DeviceBufferSpanV1, DeviceBufferSpanV1)>,
        src: &[i32],
        dst: &[i32],
    ) -> Result<(), BundleStatusV1> {
        if src.is_empty() {
            return Ok(());
        }
        let copy = &self
            .copies
            .as_ref()
            .ok_or(BundleStatusV1::InternalError)?
            .rows;
        let [src_ids, dst_ids, count] = self.upload_indices(region, src, dst)?;
        for rank in 0..self.shape.ranks.count() {
            let (from, to) = buffers(&self.scratch[rank]).ok_or(BundleStatusV1::InternalError)?;
            let spans = [from, to, src_ids[rank], dst_ids[rank], count[rank]];
            let mut bind = crate::ops::Bind::default();
            let buffers = CopyEntriesBuffers {
                src: bind.view(spans[0]),
                dst: bind.view(spans[1]),
                src_ids: bind.view(spans[2]),
                dst_ids: bind.view(spans[3]),
                live_items: bind.view(spans[4]),
            };
            let items = src.len() as u32;
            let launches = bind.launch(
                copy,
                &buffers,
                kernel_core::backend::Counts::rows(1).with_items(items, 0),
            );
            let launches = launches.map_err(|_| BundleStatusV1::InternalError)?;
            crate::pipeline::launch_all(
                &self.services,
                self.streams[rank],
                &self.symbols[rank],
                &launches,
            )
            .map_err(crate::dispatch::status)?;
        }
        Ok(())
    }

    /// Saves (or, on cancel, restores) the drafter layer's ring when the
    /// step runs the drafter without a prefill, which saves the whole ring.
    pub(super) fn copy_drafter_ring(&self, restore: bool) -> Result<(), BundleStatusV1> {
        let ring = self.ring();
        let (offset, bytes) = (
            self.shape.full_layers() * ring.layer_bytes(),
            ring.layer_bytes(),
        );
        for rank in 0..self.shape.ranks.count() {
            let ring = self.rings[rank];
            let live = DeviceBufferSpanV1::new(ring.buffer, ring.offset + offset, bytes);
            let saved = DeviceBufferSpanV1::new(
                ring.buffer,
                ring.offset + self.ring().bytes() + offset,
                bytes,
            );
            let (to, from) = if restore {
                (live, saved)
            } else {
                (saved, live)
            };
            self.services
                .copy_device_to_device(self.streams[rank], to, from)
                .map_err(status)?;
        }
        Ok(())
    }

    /// Keeps a target phase's final X rows until commit: every decode or
    /// verify row, or each prefill item's last row.
    pub(super) fn stage_target_rows(
        &self,
        index: usize,
        phase: &PhaseInputs,
    ) -> Result<(), BundleStatusV1> {
        if index == DECODE {
            let bytes = u64::from(phase.live_tokens()) * crate::scratch::wide_row_bytes(self.shape);
            for rank in 0..self.shape.ranks.count() {
                let scratch = &self.scratch[rank];
                let drafter = scratch.drafter.ok_or(BundleStatusV1::InternalError)?;
                let last = scratch.x[crate::lower::target_x(self.shape)];
                self.services
                    .copy_device_to_device(
                        self.streams[rank],
                        DeviceBufferSpanV1::new(
                            drafter.staged[DECODE].buffer,
                            drafter.staged[DECODE].offset,
                            bytes,
                        ),
                        DeviceBufferSpanV1::new(last.buffer, last.offset, bytes),
                    )
                    .map_err(status)?;
            }
            return Ok(());
        }
        let src = phase
            .q_start
            .iter()
            .zip(&phase.q_len)
            .map(|(&start, &rows)| start + rows - 1)
            .collect::<Vec<_>>();
        let dst = (0..src.len() as i32).collect::<Vec<_>>();
        self.copy_indexed(
            STAGE_REGION,
            |scratch| {
                scratch.drafter.map(|drafter| {
                    (
                        scratch.x[crate::lower::target_x(self.shape)],
                        drafter.staged[PREFILL],
                    )
                })
            },
            &src,
            &dst,
        )?;
        // An anchor at A keeps the X of row A - 1, the drafter's frontier
        // there, at its position among its anchor item's retained rows.
        let anchors = self
            .ledger
            .step()
            .map(|step| step.anchors.clone())
            .unwrap_or_default();
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        for (slot, &(request, anchor)) in anchors.iter().enumerate() {
            let item = phase
                .request_ids
                .iter()
                .position(|&id| id == request)
                .ok_or(BundleStatusV1::InternalError)?;
            let row = anchor as i32 - 1 - phase.prefix_len[item];
            src.push(phase.q_start[item] + row);
            dst.push(
                (slot as u64 * u64::from(RETAINED_ROWS) + (anchor - 1) % u64::from(RETAINED_ROWS))
                    as i32,
            );
        }
        self.copy_indexed(
            ANCHOR_REGION,
            |scratch| {
                scratch.drafter.map(|drafter| {
                    (
                        scratch.x[crate::lower::target_x(self.shape)],
                        drafter.anchor_retained,
                    )
                })
            },
            &src,
            &dst,
        )
    }

    /// Runs one drafter run on every rank: eagerly, stage by stage across
    /// the ranks, or as graphs of its bucket.
    pub(super) fn run_drafter(
        &mut self,
        run: &Run,
        verify_tokens: Option<&[PhaseDevice]>,
    ) -> Result<(), BundleStatusV1> {
        let mut runtime = self.drafter.take().ok_or(BundleStatusV1::InternalError)?;
        let result = self.run_drafter_with(&mut runtime, run, verify_tokens);
        self.drafter = Some(runtime);
        result
    }

    fn run_drafter_with(
        &self,
        runtime: &mut DrafterRuntime,
        run: &Run,
        verify_tokens: Option<&[PhaseDevice]>,
    ) -> Result<(), BundleStatusV1> {
        let inputs = run.inputs();
        let prepared = Prepared::new(self.shape, run, &inputs, runtime.eager, self.items)?;
        let heads = run.drafts.iter().map(|draft| draft.row).collect::<Vec<_>>();
        let arena_bytes = prepared
            .layout
            .pack(&inputs, &heads)
            .ok_or(BundleStatusV1::InternalError)?;
        let none = ItemEntries {
            ple_working: -1,
            ple_checkpoints: [-1; MAX_VERIFY_ROWS as usize],
            committed: -1,
            working: -1,
            checkpoints: [-1; MAX_VERIFY_ROWS as usize],
            anchor: None,
        };
        // The prefill run's rows of an anchored item keep the drafter ring
        // at the anchor's frontier, A - 1: its positions A - 17 through A - 2.
        let anchors = self
            .ledger
            .step()
            .map(|step| step.anchors.clone())
            .unwrap_or_default();
        let entries = run
            .items
            .iter()
            .map(|item| {
                let anchor = anchors
                    .iter()
                    .position(|&(request, _)| request == item.request)
                    .and_then(|slot| {
                        let frontier = anchors[slot].1 - 1;
                        let ring = u64::from(QSA_RING);
                        (item.first + ring <= frontier
                            && item.first + item.rows.len() as u64 >= frontier)
                            .then_some(crate::side::Anchor {
                                entry: -1,
                                slot: slot as i32,
                                end: frontier as i32,
                            })
                    });
                let mut entries = none;
                entries.anchor = anchor;
                entries
            })
            .collect::<Vec<_>>();
        let side_bytes = prepared
            .side
            .pack(&inputs, &entries)
            .ok_or(BundleStatusV1::InternalError)?;
        let extras = Extras::pack(run);
        let writes = runtime.block_tables.update(&inputs.block_tables);
        let uploaded = (|| {
            for (rank, data) in runtime.ranks.iter().enumerate() {
                let stream = self.streams[rank];
                upload_bytes(&self.services, stream, data.arena.arena, &arena_bytes)?;
                upload_bytes(&self.services, stream, data.side, &side_bytes)?;
                upload_bytes(&self.services, stream, data.extras, &extras)?;
                runtime.block_tables.upload(
                    &self.services,
                    stream,
                    data.arena.block_tables,
                    &writes,
                )?;
            }
            Ok(())
        })();
        if let Err(error) = uploaded {
            runtime.block_tables.forget();
            return Err(status(error));
        }
        // A component that records the MTP layer runs it through its bound
        // operations; otherwise the drafter keeps its fixed program.
        let precision = self.precision.as_ref().filter(|p| {
            model_qwen38_contract::prepared::records_drafter(p.document(), self.shape)
                .unwrap_or(false)
        });
        let bound = |rank: usize| {
            let data = &runtime.ranks[rank];
            let [sources, token_sources, slots, targets] = Extras::spans(data.extras);
            let inputs = DeviceInputs::within(data.arena, &prepared.layout);
            RankStage {
                precision,
                norm_dtypes: &self.norm_dtypes,
                shape: self.shape,
                rank: rank as u32,
                phase: prepared.phase,
                rows: prepared.rows,
                items: &prepared.items,
                head_rows: prepared.head_rows,
                scored: false,
                anchored: false,
                context: prepared.context,
                views: &self.views[rank],
                inputs,
                side: SideInputs::within(data.side, &prepared.side),
                peers: &self.scratch,
                kv_pool: self.kv_views[rank],
                kv: self.layout.kv,
                ring: self.live_ring(rank),
                ring_layout: self.ring(),
                pool_entries: self.speculation.pool_entries(self.served.slots),
                wide_comm: self.wide_comm,
                hc_precompute: self.hc_precompute,
                hc_padded_mix: self.hc_padded_mix,
                drafter: Some(DrafterInputs {
                    sources,
                    token_sources,
                    slots,
                    targets,
                    verify_tokens: verify_tokens
                        .map_or(inputs.token_ids, |devices| devices[rank].inputs.token_ids),
                }),
                media: None,
                exchange: None,
                context_parallel: self.context_parallel.as_ref(),
                ops: self.ops.as_ref(),
                target: self.target.as_ref(),
            }
        };
        let stages = program::drafter_stages_for(self.shape, !run.drafts.is_empty());
        let lower = |rank: usize, stage: program::Stage| {
            stage_launches(&self.services, &bound(rank), rank, stage)
        };
        // Lowering refusals are logged by `stage_launches`; one unit of work
        // reaches every rank or poisons the collectives (`pipeline::Unit`).
        let program = |rank: usize, stages: &[program::Stage]| {
            let mut lowered = Vec::new();
            for &stage in stages {
                lowered.extend(
                    lower(rank, stage).map_err(|_| crate::dispatch::DispatchError::InvalidPlan)?,
                );
            }
            Ok(crate::ops::program(lowered))
        };
        let Some((cache, key)) = prepared.graph() else {
            // Eagerly the drafter queues one stage at a time, each whole on
            // every rank (`Program::stage`).
            for stage in &stages {
                crate::pipeline::run_ranks(
                    &self.services,
                    &self.streams,
                    &self.symbols,
                    None,
                    &mut runtime.graphs[0],
                    &self.collectives,
                    |rank| {
                        program(rank, std::slice::from_ref(stage))
                            .map(|program| crate::pipeline::Program::stage(program.launches))
                    },
                    |_| {},
                )
                .map_err(crate::dispatch::status)?;
            }
            return Ok(());
        };
        crate::pipeline::run_ranks(
            &self.services,
            &self.streams,
            &self.symbols,
            Some(key),
            &mut runtime.graphs[cache],
            &self.collectives,
            |rank| program(rank, &stages),
            |_| {},
        )
        .map_err(crate::dispatch::status)
    }

    /// Reads the drafts back from rank 0's verify tokens and emits each
    /// drafting item's drafts before any logits.
    pub(super) fn emit_drafts(
        &mut self,
        step: &PackedStep,
        plan: &Plan,
        devices: &[PhaseDevice],
        sink: &mut ExecuteResultSinkV1,
    ) -> Result<(), BundleStatusV1> {
        if plan.drafting.is_empty() {
            return Ok(());
        }
        let decode = &step.decode;
        let length = decode.token_ids.len() * 4;
        let mut bytes = vec![0_u8; length];
        let tokens = devices[0].inputs.token_ids;
        self.services
            .copy_device_to_host(
                self.streams[0],
                &mut bytes,
                DeviceBufferSpanV1::new(tokens.buffer, tokens.offset, length as u64),
            )
            .map_err(status)?;
        let tokens = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect::<Vec<_>>();
        let pending = self
            .open
            .as_mut()
            .and_then(|open| open.drafter.as_mut())
            .ok_or(BundleStatusV1::InternalError)?;
        for &(request, item, _) in &plan.drafting {
            let start = decode.q_start[item] as usize;
            let end = start + decode.q_len[item] as usize;
            let mut rows = tokens[start..end].to_vec();
            // Row 0 is the bonus token the host sent.
            rows[0] = decode.token_ids[start];
            if rows.iter().any(|&token| token >= self.shape.vocab()) {
                return Err(BundleStatusV1::BundleError);
            }
            pending
                .set_tokens(request, &rows)
                .map_err(|_| BundleStatusV1::InternalError)?;
            sink.emit_draft_tokens(request, &rows[1..])
                .map_err(status)?;
        }
        Ok(())
    }

    /// Validate checkpoint ownership, copy retained X, then exchange entries
    /// immediately before the caller commits the ledger.
    pub(super) fn commit_speculation(
        &mut self,
        step: Option<&StepInputs>,
        open: &Open,
        plan: &bundle_runtime::ledger::CommitPlan,
    ) -> Result<(), BundleStatusV1> {
        let exchanges = if let Some(step) = step.filter(|step| step.verify) {
            let invalid = BundleStatusV1::InternalError;
            if step.decode_items.len() != step.decode_entries.len() {
                return Err(invalid);
            }
            let mut linear = Vec::new();
            let mut ple = Vec::new();
            for (&(slot, rows, request), entries) in
                step.decode_items.iter().zip(&step.decode_entries)
            {
                let count = plan
                    .items()
                    .iter()
                    .find(|item| item.request == request)
                    .ok_or(invalid)?
                    .kept;
                let row = count.checked_sub(1).ok_or(invalid)? as usize;
                if rows > 1 {
                    linear.push((slot, *entries.checkpoints.get(row).ok_or(invalid)?));
                }
                ple.push((slot, *entries.ple_checkpoints.get(row).ok_or(invalid)?));
            }
            // Neither map changes if any item, pool or ownership choice is invalid.
            let linear = self.linear_entries.prepare(&linear).ok_or(invalid)?;
            let ple = self.ple_entries.prepare(&ple).ok_or(invalid)?;
            Some((linear, ple))
        } else {
            None
        };
        // The drafter's CPU state can change before a device copy fails;
        // the ABI client still poisons the instance on that terminal failure.
        let copies = if let (Some(runtime), Some(pending)) = (self.drafter.as_mut(), &open.drafter)
        {
            runtime
                .state
                .commit(pending, |request| {
                    plan.items()
                        .iter()
                        .find(|item| item.request == request)
                        .map(|item| item.kept)
                })
                .map_err(|_| BundleStatusV1::InvalidArgument)?
        } else {
            Vec::new()
        };
        for (phase, region) in RETAIN_REGIONS.into_iter().enumerate() {
            let (src, dst): (Vec<i32>, Vec<i32>) = copies
                .iter()
                .filter(|copy| copy.phase == phase)
                .map(
                    |&Retain {
                         staged, retained, ..
                     }| (staged as i32, retained as i32),
                )
                .unzip();
            self.copy_indexed(
                region,
                |scratch| {
                    scratch
                        .drafter
                        .map(|drafter| (drafter.staged[phase], drafter.retained))
                },
                &src,
                &dst,
            )?;
        }
        if let Some((linear, ple)) = exchanges {
            #[cfg(test)]
            self.test_before_entry_exchanges()?;
            let invalid = BundleStatusV1::InternalError;
            linear
                .commit_pair(&mut self.linear_entries, ple, &mut self.ple_entries)
                .ok_or(invalid)?;
            #[cfg(test)]
            self.test_after_entry_exchanges()?;
        }
        Ok(())
    }
}

/// Retained X entries of every slot.
const _: () = assert!(RETAINED_ROWS * DRAFT_ITEMS <= INDEX_ENTRIES as u32);

/// `stage`'s launches on `bound`, with their facts.
fn stage_launches(
    services: &HostServicesV1,
    bound: &RankStage<'_>,
    rank: usize,
    stage: program::Stage,
) -> Result<Vec<crate::ops::Lowered>, BundleStatusV1> {
    bound.lower_with_facts(stage, 0).map_err(|Refusal(reason)| {
        let _ = services.log(
            format!(
                "Qwen3.8 drafter rank {rank} {:?} stage {stage:?}: {reason}",
                bound.phase
            )
            .as_bytes(),
        );
        BundleStatusV1::Unsupported
    })
}

pub(super) fn drafted(plan: Option<&Plan>) -> bool {
    plan.is_some_and(|plan| !plan.is_empty())
}
