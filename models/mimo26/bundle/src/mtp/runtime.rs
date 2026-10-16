//! Instance-owned resources and execution for MTP layer 0.

use super::*;
use crate::mtp::{batch as draft_batch, program, staging, views};
use crate::program::Stage;
use crate::selection::{MimoPhase as _, Phase};
use crate::topology::Tp;

fn hidden_bytes(tp: Tp) -> u64 {
    tp.geometry.hidden() as u64 * 2
}
fn retained_rows(tp: Tp) -> u64 {
    crate::retain::capacity(tp.geometry.window() as u64)
}
/// Items an MTP instance serves at most (`DRAFT_ITEMS`).
const BATCH: u64 = crate::contract::DRAFT_ITEMS as u64;
/// Bytes of one rank's region of argmax partials: a score (or ID) per item.
const ARGMAX_RANK_BYTES: u64 = BATCH * 4;

/// Bytes of one half of the argmax pair buffers: every decode row's score
/// (or ID) from every rank. The row-major rank pairs come first, then per
/// rank one region of `BATCH` partials.
fn pair_half_bytes(tp: Tp) -> u64 {
    BATCH * u64::from(tp.ranks()) * 4
}

pub(super) struct Mtp {
    pub(crate) ranks: Vec<Rank>,
    state: staging::Staging,
    pending_rows: Vec<staging::TargetRows>,
    pending: bool,
    ring_changed: bool,
    caught: Vec<(u64, u64)>,
    staged: bool,
    /// Staged prefill rows of each pending row; decode rows are retained by
    /// the target graph.
    source_rows: Vec<Option<u64>>,
    pub(super) graphs: RefCell<[crate::graphs::GraphCache; 7]>,
    /// Enqueue every group eagerly: `CHOPINFER_MIMO_MTP_EAGER=1`, or an eager
    /// instance (see `Instance::eager`).
    pub(super) eager: bool,
}

#[derive(Clone, Copy)]
enum HiddenRows {
    Committed,
    Chain,
}

pub(crate) struct Rank {
    pub(crate) views: BTreeMap<String, DeviceBufferSpanV1>,
    arena: PhaseArena,
    ring: DeviceBufferSpanV1,
    before: DeviceBufferSpanV1,
    caught_up: DeviceBufferSpanV1,
    committed: DeviceBufferSpanV1,
    target_rows: DeviceBufferSpanV1,
    chain_hidden: DeviceBufferSpanV1,
    hidden_input: DeviceBufferSpanV1,
    next_tokens: DeviceBufferSpanV1,
    compact_logits: DeviceBufferSpanV1,
    packed: DeviceBufferSpanV1,
    scores: DeviceBufferSpanV1,
    ids: DeviceBufferSpanV1,
    token: DeviceBufferSpanV1,
    live_rows: DeviceBufferSpanV1,
    width: DeviceBufferSpanV1,
    event: BundleEventHandleV1,
}

fn slice(
    span: DeviceBufferSpanV1,
    offset: u64,
    length: u64,
) -> Result<DeviceBufferSpanV1, BundleAbiError> {
    if offset
        .checked_add(length)
        .is_none_or(|end| end > span.length)
    {
        return Err(BundleAbiError::InvalidRange("MTP device slice"));
    }
    let offset = span
        .offset
        .checked_add(offset)
        .ok_or(BundleAbiError::InvalidRange("MTP device offset"))?;
    Ok(DeviceBufferSpanV1::new(span.buffer, offset, length))
}

pub(super) fn acquire(
    services: &HostServicesV1,
    ranks: &[(
        BundleContextHandleV1,
        BundleMemoryPoolHandleV1,
        BundleStreamHandleV1,
    )],
    target: &[BTreeMap<String, DeviceBufferSpanV1>],
    held: &mut Vec<Held>,
    tp: Tp,
    items: Items,
) -> Result<Option<Mtp>, BundleAbiError> {
    let mut indexes = Vec::new();
    for rank in 0..tp.ranks() {
        let role = model_mimo26_contract::weight_plan::index_role(
            model_mimo26_contract::weight_plan::DRAFTER_IMAGE_ROLE,
            rank,
        );
        let bytes = acquire::read_component(super::log(services), &role, 1 << 20, held)?;
        let index = crate::index::parse(&bytes, rank)
            .map_err(|_| BundleAbiError::InvalidRange("MTP rank index"))?;
        indexes.push(index);
    }
    // A packaged DFlash drafter remains available for later bundle support.
    // Host drafts still work without allocating MTP state for that image.
    if indexes.iter().all(|index| {
        index.views.contains_key("mask_embedding.weight")
            && !index
                .views
                .keys()
                .any(|name| name.starts_with("model.mtp."))
    }) {
        return Ok(None);
    }
    // MTP drafts and catches up at the decode kernels' rows.
    if items.get() > model_mimo26_contract::serving::DRAFT_ITEMS {
        return Err(BundleAbiError::InvalidRange(
            "MTP drafts for at most 32 sequences",
        ));
    }
    for (rank, index) in indexes.iter().enumerate() {
        views::validate(index, tp).map_err(BundleAbiError::InvalidRange)?;
        bundle_runtime::index::authenticate_rank_identity(
            services,
            "mimo.drafter-rank-images.v1",
            rank as u32,
            index.image_bytes,
            index.sha256.as_deref(),
            index.direct_identity.as_deref(),
        )?;
    }
    let image = acquire::open_image(super::log(services), views::IMAGE_ROLE, held)?;
    let arenas = acquire::upload_arenas(super::log(services), image, ranks, held)?;
    if arenas.len() != tp.count() {
        return Err(BundleAbiError::InvalidRange("MTP rank arenas"));
    }
    let rows: BTreeMap<_, _> = model_mimo26_contract::resources::working_rows(
        tp,
        items.get(),
        Some(model_mimo26_contract::weight_plan::Drafter::Mtp),
    )
    .into_iter()
    .map(|row| (row.name, row.layout))
    .collect();
    let bytes = |name: &str| rows[name].allocation_bytes();
    let mut result = Vec::new();
    for (rank, arena) in arenas.into_iter().enumerate() {
        let image_span = services
            .resolve_compiled_image_view(image, arena, rank as u32, b"")?
            .buffer;
        let index = &indexes[rank];
        let views = views::aliases(index, image_span, &target[rank], tp)
            .map_err(BundleAbiError::InvalidRange)?;
        let (context, pool, stream) = ranks[rank];
        let mut alloc = |bytes| allocate_span(services, pool, stream, bytes, held);
        let arena = PhaseArena {
            arena: alloc(bytes("mtp.inputs"))?,
            block_tables: alloc(bytes("mtp.block_tables"))?,
        };
        let ring = alloc(bytes("mtp.ring"))?;
        let before = alloc(bytes("mtp.before"))?;
        let caught_up = alloc(bytes("mtp.caught_up"))?;
        let committed = alloc(bytes("mtp.committed"))?;
        let target_rows = alloc(bytes("mtp.target_rows"))?;
        let chain_hidden = alloc(bytes("mtp.chain_hidden"))?;
        let hidden_input = alloc(bytes("mtp.hidden_input"))?;
        let next_tokens = alloc(bytes("mtp.next_tokens"))?;
        let compact_logits = alloc(bytes("mtp.compact_logits"))?;
        let packed = alloc(bytes("mtp.packed"))?;
        let scores = alloc(bytes("mtp.scores"))?;
        let ids = alloc(bytes("mtp.ids"))?;
        let token = alloc(bytes("mtp.token"))?;
        let live_rows = alloc(bytes("mtp.live_rows"))?;
        let width = alloc(bytes("mtp.width"))?;
        services.zero_device_buffer(stream, ring)?;
        upload_bytes(services, stream, live_rows, &1_u32.to_le_bytes())?;
        upload_bytes(
            services,
            stream,
            width,
            &crate::topology::vocab_per_rank(tp).to_le_bytes(),
        )?;
        let event = services.create_event(context)?;
        held.push(Held::Event(event));
        result.push(Rank {
            views,
            arena,
            ring,
            before,
            caught_up,
            committed,
            target_rows,
            chain_hidden,
            hidden_input,
            next_tokens,
            compact_logits,
            packed,
            scores,
            ids,
            token,
            live_rows,
            width,
            event,
        });
    }
    Ok(Some(Mtp {
        ranks: result,
        state: staging::Staging::for_topology(tp),
        pending_rows: Vec::new(),
        pending: false,
        ring_changed: false,
        caught: Vec::new(),
        staged: false,
        source_rows: Vec::new(),
        graphs: RefCell::new(std::array::from_fn(|_| {
            crate::graphs::GraphCache::new(
                bundle_runtime::geometry::Ranks::new(tp.ranks()).expect("supported rank count"),
            )
        })),
        eager: std::env::var_os("CHOPINFER_MIMO_MTP_EAGER").is_some_and(|value| value == "1"),
    }))
}

impl Mtp {
    /// Appends the retained-row write to a decode or verify target program:
    /// every row's final normalized hidden state enters its slot's history.
    pub(super) fn extend(
        &self,
        rank: usize,
        bound: &crate::program_lower::RankStage<'_>,
        stage: Stage,
        launches: &mut Vec<crate::program_lower::Lowered>,
    ) -> Result<(), kernel_core::contract::Refusal> {
        if stage != Stage::GatherLogits || !bound.phase.token_bucketed() {
            return Ok(());
        }
        let committed = self
            .ranks
            .get(rank)
            .ok_or(kernel_core::contract::Refusal("MTP rank is missing"))?
            .committed;
        launches.push(bound.ops.retain_rows(
            &program::retain(bound.tp),
            bound.phase,
            bound.bucket,
            bound.normalized,
            committed,
            &bound.inputs,
        )?);
        Ok(())
    }
}

impl Instance {
    pub(super) fn mtp_begin(
        &mut self,
        step: &crate::inputs::PackedStep,
        batch: &runtime_bundle_sdk::BatchRequestV1,
    ) -> Result<(), BundleAbiError> {
        let Some(mut mtp) = self.mtp.take() else {
            return Ok(());
        };
        let result = (|| {
            if mtp.pending {
                return Err(BundleAbiError::InvalidRange(
                    "MTP transaction already pending",
                ));
            }
            mtp.pending = true;
            mtp.pending_rows.clear();
            mtp.caught.clear();
            self.upload_phase(&step.decode)?;
            if !batch.items().iter().any(|item| item.bundle_drafts()) {
                return Ok(());
            }
            for rank in 0..self.tp.count() {
                self.services.copy_device_to_device(
                    self.streams[rank],
                    mtp.ranks[rank].before,
                    mtp.ranks[rank].ring,
                )?;
            }
            mtp.ring_changed = true;
            let mut sequences = Vec::new();
            for phase in [&step.prefill, &step.decode] {
                for item in 0..phase.request_ids.len() {
                    let row = phase.q_start[item] as usize;
                    let request = phase.request_ids[item];
                    if !batch.items()[phase.item_order[item]].bundle_drafts() {
                        continue;
                    }
                    let catch_up = mtp
                        .state
                        .catch_up(request, phase.prefix_len[item] as u64, phase.token_ids[row])
                        .map_err(BundleAbiError::InvalidRange)?;
                    let bundle = batch.items()[phase.item_order[item]].bundle_drafts();
                    let Some(mut catch_up) = catch_up else {
                        if bundle {
                            return Err(BundleAbiError::InvalidRange(
                                "MTP cannot draft without target hidden rows",
                            ));
                        }
                        continue;
                    };
                    mtp.caught.push((request, phase.prefix_len[item] as u64));
                    if catch_up.tokens.len() > crate::contract::MAX_VERIFY_ROWS as usize {
                        let last = catch_up
                            .tokens
                            .pop()
                            .ok_or(BundleAbiError::InvalidRange("MTP empty history"))?;
                        self.mtp_layer(&mtp, &catch_up, HiddenRows::Committed, false)?;
                        catch_up.start += catch_up.tokens.len() as u64;
                        catch_up.tokens = vec![last];
                    }
                    if phase.decode {
                        sequences.push(draft_batch::Sequence {
                            catch_up,
                            target_start: row as u32,
                            drafts: if bundle {
                                phase.q_len[item] as u32 - 1
                            } else {
                                0
                            },
                        });
                    } else {
                        self.mtp_layer(&mtp, &catch_up, HiddenRows::Committed, false)?;
                    }
                }
            }
            if !sequences.is_empty() {
                let batches = draft_batch::plan_for(self.tp, &sequences)
                    .map_err(BundleAbiError::InvalidRange)?;
                for (position, batch) in batches.iter().enumerate() {
                    self.mtp_run_layer(
                        &mtp,
                        &batch.inputs,
                        batch.phase,
                        if position == 0 {
                            HiddenRows::Committed
                        } else {
                            HiddenRows::Chain
                        },
                        &batch.hidden_rows,
                        !batch.target_rows.is_empty(),
                    )?;
                    if position == 0 && batches.len() > 1 {
                        for rank in 0..self.tp.count() {
                            self.services.copy_device_to_device(
                                self.streams[rank],
                                mtp.ranks[rank].caught_up,
                                mtp.ranks[rank].ring,
                            )?;
                        }
                    }
                    if !batch.target_rows.is_empty() {
                        self.mtp_argmax(&mtp, batch)?;
                    }
                }
                // Keep catch-up, but discard rows based on generated hidden states.
                if batches.len() > 1 {
                    for rank in 0..self.tp.count() {
                        self.services.copy_device_to_device(
                            self.streams[rank],
                            mtp.ranks[rank].ring,
                            mtp.ranks[rank].caught_up,
                        )?;
                    }
                }
            }
            Ok(())
        })();
        self.mtp = Some(mtp);
        result
    }

    /// Stages a prefill phase's target hidden rows until commit. Decode and
    /// verify graphs write theirs into the committed history directly.
    pub(super) fn mtp_tap(&self, phase: &crate::inputs::PhaseInputs) -> Result<(), BundleAbiError> {
        let Some(mtp) = &self.mtp else {
            return Ok(());
        };
        if phase.live_tokens() == 0 || phase.decode {
            return Ok(());
        }
        let bytes = u64::from(phase.live_tokens()) * hidden_bytes(self.tp);
        for rank in 0..self.tp.count() {
            self.services.copy_device_to_device(
                self.streams[rank],
                slice(mtp.ranks[rank].target_rows, 0, bytes)?,
                slice(self.normalized[rank], 0, bytes)?,
            )?;
        }
        Ok(())
    }

    pub(super) fn mtp_emit(
        &mut self,
        step: &mut crate::inputs::PackedStep,
        batch: &runtime_bundle_sdk::BatchRequestV1,
        sink: &mut ExecuteResultSinkV1,
    ) -> Result<(), BundleAbiError> {
        let Some(mtp) = &mut self.mtp else {
            return Ok(());
        };
        if batch.items().iter().any(|item| item.bundle_drafts()) {
            let mut bytes = vec![0; step.decode.token_ids.len() * 4];
            self.services.copy_device_to_host(
                self.streams[0],
                &mut bytes,
                slice(
                    self.verify_inputs[0].token_ids,
                    0,
                    (step.decode.token_ids.len() * 4) as u64,
                )?,
            )?;
            for (dst, bytes) in step.decode.token_ids.iter_mut().zip(bytes.chunks_exact(4)) {
                *dst = u32::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| BundleAbiError::InvalidRange("MTP token bytes"))?,
                );
                if *dst >= self.tp.geometry.vocab() {
                    return Err(BundleAbiError::InvalidRange(
                        "MTP draft token outside vocabulary",
                    ));
                }
            }
            for (item, &wire_index) in step.decode.item_order.iter().enumerate() {
                if batch.items()[wire_index].bundle_drafts() {
                    let start = step.decode.q_start[item] as usize;
                    let end = start + step.decode.q_len[item] as usize;
                    sink.emit_draft_tokens(
                        step.decode.request_ids[item],
                        &step.decode.token_ids[start + 1..end],
                    )?;
                }
            }
        }
        let rows = [&step.prefill, &step.decode]
            .into_iter()
            .flat_map(|phase| {
                phase
                    .request_ids
                    .iter()
                    .enumerate()
                    .map(|(item, &request)| {
                        let start = phase.q_start[item] as usize;
                        staging::TargetRows {
                            request,
                            slot: phase.slot_ids[item] as u32,
                            start: phase.prefix_len[item] as u64,
                            tokens: phase.token_ids[start..start + phase.q_len[item] as usize]
                                .to_vec(),
                            verify: phase.verify_rows[item],
                        }
                    })
            })
            .collect::<Vec<_>>();
        mtp.state
            .prepare(rows.clone())
            .map_err(BundleAbiError::InvalidRange)?;
        mtp.source_rows = [&step.prefill, &step.decode]
            .into_iter()
            .flat_map(|phase| {
                phase
                    .q_start
                    .iter()
                    .map(move |&row| (!phase.decode).then_some(row as u64))
            })
            .collect();
        mtp.pending_rows = rows;
        mtp.staged = true;
        Ok(())
    }

    pub(super) fn mtp_finish(
        &mut self,
        kept: &[(u64, u32)],
        commit: Option<&CommitPlan>,
    ) -> Result<(), BundleAbiError> {
        let Some(mtp) = &mut self.mtp else {
            return Ok(());
        };
        if !mtp.pending {
            return Ok(());
        }
        if let Some(plan) = commit {
            if !mtp.staged {
                return Err(BundleAbiError::InvalidRange(
                    "MTP target rows were not staged",
                ));
            }
            // Decode and verify rows already sit in the history; committing
            // advances the range catch-up may read. Only staged prefill rows
            // are copied. Row counts cannot tell a single-token prefill from
            // a decode, so emit records each staged source row.
            for (row, &source_row) in mtp.pending_rows.iter().zip(&mtp.source_rows) {
                let Some(source_row) = source_row else {
                    continue;
                };
                // Phase-grouped rows join the validated wire-order plan by request.
                let count = u64::from(
                    plan.items()
                        .iter()
                        .find(|item| item.request == row.request)
                        .expect("MTP pending row belongs to the open ledger step")
                        .kept,
                );
                let skip = count.saturating_sub(self.tp.geometry.window() as u64);
                let count = count - skip;
                let position = (row.start + skip) % retained_rows(self.tp);
                let first = count.min(retained_rows(self.tp) - position);
                for rank in 0..self.tp.count() {
                    for (offset, destination, length) in
                        [(0, position, first), (first, 0, count - first)]
                    {
                        if length == 0 {
                            continue;
                        }
                        self.services.copy_device_to_device(
                            self.streams[rank],
                            slice(
                                mtp.ranks[rank].committed,
                                (u64::from(row.slot) * retained_rows(self.tp) + destination)
                                    * hidden_bytes(self.tp),
                                length * hidden_bytes(self.tp),
                            )?,
                            slice(
                                mtp.ranks[rank].target_rows,
                                (source_row + skip + offset) * hidden_bytes(self.tp),
                                length * hidden_bytes(self.tp),
                            )?,
                        )?;
                    }
                }
            }
            mtp.state
                .commit_materialized(kept, &mtp.caught)
                .map_err(BundleAbiError::InvalidRange)?;
        } else {
            if mtp.ring_changed {
                for rank in 0..self.tp.count() {
                    self.services.copy_device_to_device(
                        self.streams[rank],
                        mtp.ranks[rank].ring,
                        mtp.ranks[rank].before,
                    )?;
                }
            }
            if mtp.staged {
                mtp.state.cancel().map_err(BundleAbiError::InvalidRange)?;
            }
        }
        mtp.pending = false;
        mtp.ring_changed = false;
        mtp.caught.clear();
        mtp.staged = false;
        mtp.pending_rows.clear();
        Ok(())
    }

    pub(super) fn mtp_release_graphs(&self) -> Result<(), BundleAbiError> {
        if let Some(mtp) = &self.mtp {
            for cache in mtp.graphs.borrow_mut().iter_mut() {
                cache.release_all(&self.services)?;
            }
        }
        Ok(())
    }

    /// Retained committed hidden rows, copied without the materialized K/V.
    pub(super) fn mtp_prefix_span(&self, rank: usize) -> Option<DeviceBufferSpanV1> {
        self.mtp.as_ref().map(|mtp| mtp.ranks[rank].committed)
    }

    pub(super) fn mtp_prefix_ready(&self, request: u64, slot: u32, tokens: u64) -> bool {
        self.mtp.as_ref().is_none_or(|mtp| {
            !mtp.pending && mtp.state.committed_boundary(request) == Some((slot, tokens))
        })
    }

    /// Rebuilds token bookkeeping after the retained hidden rows are restored.
    pub(super) fn mtp_prefix_restore(
        &mut self,
        request: u64,
        slot: u32,
        tokens: &[u32],
    ) -> Result<(), BundleAbiError> {
        if let Some(mtp) = &mut self.mtp {
            mtp.state
                .restore(request, slot, tokens)
                .map_err(BundleAbiError::InvalidRange)?;
        }
        Ok(())
    }

    pub(super) fn mtp_retire(&mut self, request: u64) -> Result<(), BundleAbiError> {
        if let Some(mtp) = &mut self.mtp {
            mtp.state
                .retire(request)
                .map_err(BundleAbiError::InvalidRange)?;
        }
        Ok(())
    }
}

impl Instance {
    fn mtp_layer(
        &self,
        mtp: &Mtp,
        catch_up: &staging::CatchUp,
        source: HiddenRows,
        head: bool,
    ) -> Result<(), BundleAbiError> {
        let count = catch_up.tokens.len() as u32;
        if count == 0
            || count > PREFILL_TOKENS
            || catch_up.start + u64::from(count) > self.tp.geometry.context_tokens() as u64
        {
            return Err(BundleAbiError::InvalidRange("MTP row count or position"));
        }
        let decode = count == 1;
        let phase = if decode {
            Phase::Decode(1)
        } else {
            Phase::Prefill
        };
        let inputs = crate::inputs::PhaseInputs {
            decode,
            item_order: vec![0],
            request_ids: vec![0],
            emits_row: vec![head],
            prompt_rows: vec![false],
            verify_rows: vec![false],
            token_ids: catch_up.tokens.clone(),
            positions: (0..count)
                .map(|i| (catch_up.start + u64::from(i)) as i32)
                .collect(),
            position_axes: Vec::new(),
            token_batch: vec![0; count as usize],
            block_tables: vec![vec![]],
            slot_ids: vec![catch_up.slot as i32],
            q_start: vec![0],
            q_len: vec![count as i32],
            prefix_len: vec![catch_up.start as i32],
            seq_lens: vec![(catch_up.start + u64::from(count)) as i32],
            last_positions: vec![(catch_up.start + u64::from(count) - 1) as i64],
            ..Default::default()
        };
        self.mtp_run_layer(mtp, &inputs, phase, source, &[count - 1], head)
    }

    /// Each group ends before the next collective, so all ranks can enqueue
    /// the producer before any rank waits on the next reduction.
    fn mtp_group(
        &self,
        mtp: &Mtp,
        phase: Phase,
        items: u32,
        group: usize,
        prepare: impl Fn(usize) -> Result<Vec<crate::program_lower::Lowered>, BundleAbiError>,
    ) -> Result<(), BundleAbiError> {
        let graph = if phase.token_bucketed() && !mtp.eager {
            Some((
                phase
                    .capacity()
                    .map_err(|error| BundleAbiError::InvalidRange(error.0))?,
                if matches!(phase, Phase::Verify(_)) {
                    items.next_power_of_two()
                } else {
                    0
                },
            ))
        } else {
            None
        };
        let mut caches = mtp.graphs.borrow_mut();
        crate::pipeline::run_ranks(
            &self.services,
            &self.streams,
            &self.symbols,
            graph,
            &mut caches[group],
            &self.collectives,
            |rank| {
                prepare(rank)
                    .map(crate::program_lower::program)
                    .map_err(crate::dispatch::DispatchError::from)
            },
            |_| {},
        )
        .map_err(|error| match error {
            crate::dispatch::DispatchError::Host(error) => error,
            _ => BundleAbiError::InvalidRange("MTP graph or eager launch failed"),
        })
    }

    fn mtp_run_layer(
        &self,
        mtp: &Mtp,
        inputs: &crate::inputs::PhaseInputs,
        phase: Phase,
        source: HiddenRows,
        rows: &[u32],
        head: bool,
    ) -> Result<(), BundleAbiError> {
        let count = inputs.live_tokens();
        let context = inputs.seq_lens.iter().copied().max().unwrap_or(0) as u32;
        let layout = if phase.token_bucketed() {
            crate::upload::InputLayout::within(
                crate::contract::DRAFT_ITEMS,
                crate::contract::DRAFT_ITEMS,
                self.items,
            )
        } else {
            crate::upload::InputLayout::prefill::<Mimo>(inputs, rows.len())
        };
        let bytes = layout
            .pack(inputs, rows)
            .ok_or(BundleAbiError::InvalidRange("MTP input arena"))?;
        let refuse = |error: kernel_core::contract::Refusal| BundleAbiError::InvalidRange(error.0);
        for rank in 0..self.tp.count() {
            let data = &mtp.ranks[rank];
            upload_bytes(&self.services, self.streams[rank], data.arena.arena, &bytes)?;
            let device_inputs = DeviceInputs::within(data.arena, &layout);
            for (item, &slot) in inputs.slot_ids.iter().enumerate() {
                let row = inputs.q_start[item] as u64;
                let bytes = inputs.q_len[item] as u64 * hidden_bytes(self.tp);
                if matches!(source, HiddenRows::Chain) {
                    self.services.copy_device_to_device(
                        self.streams[rank],
                        slice(device_inputs.token_ids, row * 4, 4)?,
                        slice(data.token, slot as u64 * 4, 4)?,
                    )?;
                }
                match source {
                    HiddenRows::Chain => {
                        self.services.copy_device_to_device(
                            self.streams[rank],
                            slice(data.hidden_input, row * hidden_bytes(self.tp), bytes)?,
                            slice(
                                data.chain_hidden,
                                slot as u64 * hidden_bytes(self.tp),
                                bytes,
                            )?,
                        )?;
                    }
                    HiddenRows::Committed => {
                        let position = inputs.prefix_len[item] as u64 % retained_rows(self.tp);
                        let count = inputs.q_len[item] as u64;
                        let first = count.min(retained_rows(self.tp) - position);
                        for (offset, source, length) in
                            [(0, position, first), (first, 0, count - first)]
                        {
                            if length == 0 {
                                continue;
                            }
                            self.services.copy_device_to_device(
                                self.streams[rank],
                                slice(
                                    data.hidden_input,
                                    (row + offset) * hidden_bytes(self.tp),
                                    length * hidden_bytes(self.tp),
                                )?,
                                slice(
                                    data.committed,
                                    (slot as u64 * retained_rows(self.tp) + source)
                                        * hidden_bytes(self.tp),
                                    length * hidden_bytes(self.tp),
                                )?,
                            )?;
                        }
                    }
                }
            }
        }
        let prepared = if phase.token_bucketed() && !mtp.eager {
            (
                phase,
                phase.capacity().map_err(refuse)?,
                inputs.live_batch().next_power_of_two(),
                self.tp.geometry.context_tokens(),
            )
        } else {
            (phase, count, inputs.live_batch(), context)
        };
        self.mtp_group(mtp, phase, inputs.live_batch(), 0, |rank| {
            let data = &mtp.ranks[rank];
            let bound = self
                .mtp_rank(
                    mtp,
                    rank,
                    prepared,
                    DeviceInputs::within(data.arena, &layout),
                    rows,
                )
                .map_err(refuse)?;
            let norm = data.views[if program::eh_shard(self.tp, rank as u32).0 {
                "model.mtp.layers.0.hnorm.weight"
            } else {
                "model.mtp.layers.0.enorm.weight"
            }];
            let mut launches =
                program::pair(&bound, data.hidden_input, norm, data.packed).map_err(refuse)?;
            launches.push(program::eh(&bound, data.packed).map_err(refuse)?);
            Ok(launches)
        })?;
        // EH's reduction starts with a zero residual, after embedding has read it.
        for rank in 0..self.tp.count() {
            self.services.zero_device_buffer(
                self.streams[rank],
                slice(
                    self.activations[rank],
                    0,
                    u64::from(phase.capacity().map_err(refuse)?) * hidden_bytes(self.tp),
                )?,
            )?;
        }
        for (group, stages) in [
            &program::BLOCK[..7],
            &program::BLOCK[7..10],
            &program::BLOCK[10..],
        ]
        .into_iter()
        .enumerate()
        {
            self.mtp_group(mtp, phase, inputs.live_batch(), group + 1, |rank| {
                let bound = self
                    .mtp_rank(
                        mtp,
                        rank,
                        prepared,
                        DeviceInputs::within(mtp.ranks[rank].arena, &layout),
                        rows,
                    )
                    .map_err(refuse)?;
                let mut launches = Vec::new();
                for &stage in stages {
                    launches.extend(bound.lower_with_facts(stage).map_err(refuse)?);
                }
                Ok(launches)
            })?;
        }
        for rank in 0..self.tp.count() {
            for (item, &row) in rows.iter().enumerate() {
                self.services.copy_device_to_device(
                    self.streams[rank],
                    slice(
                        mtp.ranks[rank].chain_hidden,
                        inputs.slot_ids[item] as u64 * hidden_bytes(self.tp),
                        hidden_bytes(self.tp),
                    )?,
                    slice(
                        self.normalized[rank],
                        u64::from(row) * hidden_bytes(self.tp),
                        hidden_bytes(self.tp),
                    )?,
                )?;
            }
        }
        if head {
            self.mtp_group(mtp, phase, inputs.live_batch(), 4, |rank| {
                let bound = self
                    .mtp_rank(
                        mtp,
                        rank,
                        prepared,
                        DeviceInputs::within(mtp.ranks[rank].arena, &layout),
                        rows,
                    )
                    .map_err(refuse)?;
                let mut launches = bound
                    .lower_with_facts(Stage::GatherHeadRows)
                    .map_err(refuse)?;
                launches.extend(bound.lower_with_facts(Stage::LmHead).map_err(refuse)?);
                Ok(launches)
            })?;
        }
        Ok(())
    }

    fn mtp_argmax(&self, mtp: &Mtp, batch: &draft_batch::Batch) -> Result<(), BundleAbiError> {
        let refuse = |error: kernel_core::contract::Refusal| BundleAbiError::InvalidRange(error.0);
        let rows = batch.target_rows.len() as u32;
        let logits_bytes = self.tp.vocab() * 2;
        let half = pair_half_bytes(self.tp);
        let ranks = u64::from(self.tp.ranks());
        let root = &mtp.ranks[0];
        for rank in 0..self.tp.count() {
            let data = &mtp.ranks[rank];
            upload_bytes(
                &self.services,
                self.streams[rank],
                data.live_rows,
                &rows.to_le_bytes(),
            )?;
            for (index, &row) in batch.logits_rows.iter().enumerate() {
                self.services.copy_device_to_device(
                    self.streams[rank],
                    slice(
                        data.compact_logits,
                        index as u64 * logits_bytes,
                        logits_bytes,
                    )?,
                    slice(
                        self.logits[rank].local,
                        u64::from(row) * logits_bytes,
                        logits_bytes,
                    )?,
                )?;
            }
        }
        self.mtp_group(mtp, Phase::Decode(rows), rows, 5, |rank| {
            let data = &mtp.ranks[rank];
            Ok(vec![
                self.ops
                    .mtp_argmax_partial(
                        rank as u32,
                        rows,
                        &crate::ops::PartialViews {
                            logits: data.compact_logits,
                            live_width: data.width,
                            scores: slice(
                                root.scores,
                                half + rank as u64 * ARGMAX_RANK_BYTES,
                                ARGMAX_RANK_BYTES,
                            )?,
                            ids: slice(
                                root.ids,
                                half + rank as u64 * ARGMAX_RANK_BYTES,
                                ARGMAX_RANK_BYTES,
                            )?,
                            live_rows: data.live_rows,
                        },
                    )
                    .map_err(refuse)?,
            ])
        })?;
        for rank in 0..self.tp.count() {
            self.services
                .record_event(mtp.ranks[rank].event, self.streams[rank])?;
        }
        for rank in 1..self.tp.count() {
            self.services
                .wait_event(self.streams[0], mtp.ranks[rank].event)?;
        }
        // Partials write contiguous rank-major rows through peer leases. The
        // final kernel reads row-major rank pairs, transposed on the root stream.
        for row in 0..u64::from(rows) {
            for rank in 0..ranks {
                for buffer in [root.scores, root.ids] {
                    self.services.copy_device_to_device(
                        self.streams[0],
                        slice(buffer, row * ranks * 4 + rank * 4, 4)?,
                        slice(buffer, half + rank * ARGMAX_RANK_BYTES + row * 4, 4)?,
                    )?;
                }
            }
        }
        self.services.record_event(root.event, self.streams[0])?;
        for rank in 1..self.tp.count() {
            self.services.wait_event(self.streams[rank], root.event)?;
        }
        self.mtp_group(mtp, Phase::Decode(rows), rows, 6, |rank| {
            let data = &mtp.ranks[rank];
            Ok(vec![
                self.ops
                    .mtp_argmax_final(
                        rows,
                        root.scores,
                        root.ids,
                        data.next_tokens,
                        data.live_rows,
                    )
                    .map_err(refuse)?,
            ])
        })?;
        for rank in 0..self.tp.count() {
            let data = &mtp.ranks[rank];
            for (index, (&slot, &target)) in batch.slots.iter().zip(&batch.target_rows).enumerate()
            {
                let token = slice(data.next_tokens, index as u64 * 4, 4)?;
                self.services.copy_device_to_device(
                    self.streams[rank],
                    slice(data.token, u64::from(slot) * 4, 4)?,
                    token,
                )?;
                self.services.copy_device_to_device(
                    self.streams[rank],
                    slice(self.verify_inputs[rank].token_ids, u64::from(target) * 4, 4)?,
                    token,
                )?;
            }
        }
        Ok(())
    }
}

impl Instance {
    fn mtp_rank<'a>(
        &'a self,
        mtp: &'a Mtp,
        rank: usize,
        geometry: (Phase, u32, u32, u32),
        inputs: DeviceInputs,
        logit_rows: &'a [u32],
    ) -> Result<crate::program_lower::RankStage<'a>, kernel_core::contract::Refusal> {
        let (phase, tokens, batch, context) = geometry;
        let capacity = phase.capacity()?;
        let comm_index = crate::scratch::COMM_BUCKETS
            .iter()
            .position(|&bucket| bucket == capacity)
            .ok_or(kernel_core::contract::Refusal("MTP communication bucket"))?;
        let scratch = &self.scratch[rank];
        let peers = 0..self.tp.count();
        Ok(crate::program_lower::RankStage {
            services: Some(&self.services),
            tp: self.tp,
            rank: rank as u32,
            phase,
            live_tokens: tokens,
            live_batch: batch,
            context,
            ga_kv: self.kv_dtype,
            ga_kv_scales: self.kv_scales,
            switches: self.switches,
            ops: &self.ops,
            bucket: comm_index,
            views: &mtp.ranks[rank].views,
            projections: self.precision.as_ref().and_then(|p| p.rank(rank)),
            inputs,
            scratch,
            activation: self.activations[rank],
            normalized: self.normalized[rank],
            workspace: self.workspaces[rank],
            local_logits: self.logits[rank].local,
            scored_prompt_logits: false,
            logit_rows,
            logits_gather: crate::ops::LogitsBuffers {
                local: self.logits[rank].local,
                shard: crate::topology::vocab_per_rank(self.tp),
                outputs: peers
                    .clone()
                    .map(|r| self.logits[r].gathered_decode)
                    .collect(),
                flags: peers
                    .clone()
                    .map(|r| self.scratch[r].comm_state[comm_index].gather_rows_flags)
                    .collect(),
                epochs: peers
                    .clone()
                    .map(|r| self.scratch[r].comm_state[comm_index].gather_rows_epoch)
                    .collect(),
            },
            exchange_logits: false,
            hidden_inboxes: peers
                .clone()
                .map(|r| crate::scratch::comm_inbox(self.scratch[r].comm_inbox, comm_index))
                .collect(),
            hidden_flags: peers
                .clone()
                .map(|r| self.scratch[r].comm_state[comm_index].hidden_flags)
                .collect(),
            hidden_epochs: peers
                .clone()
                .map(|r| self.scratch[r].comm_state[comm_index].hidden_epoch)
                .collect(),
            hidden_gathered: peers.map(|r| self.normalized[r]).collect(),
            ring_state: mtp.ranks[rank].ring,
            // The drafter layer's own ring holds `DRAFT_ITEMS` slots.
            ring: crate::ring::RingGeometry::new(self.tp, crate::contract::DRAFT_ITEMS),
            items: self.items,
            kv_pool: self.kv_views[rank],
        })
    }
}

#[cfg(test)]
mod item_layout_tests {
    use super::{ARGMAX_RANK_BYTES, BATCH, pair_half_bytes};
    use crate::topology::Tp;

    /// The argmax pairs and per-rank partials follow the batch and the rank
    /// count, and keep TP4's offsets they had as literals at 32.
    #[test]
    fn argmax_offsets_follow_the_batch() {
        assert_eq!(ARGMAX_RANK_BYTES, 4 * BATCH);
        for tp in Tp::SUPPORTED {
            assert_eq!(
                pair_half_bytes(tp),
                u64::from(tp.ranks()) * ARGMAX_RANK_BYTES
            );
        }
        if BATCH == 32 {
            assert_eq!((pair_half_bytes(Tp::TP4), ARGMAX_RANK_BYTES), (512, 128));
        }
    }
}
