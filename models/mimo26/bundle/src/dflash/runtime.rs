//! Instance-owned DFlash resources and step bookkeeping.
//!
//! Target graphs retain raw taps. Decode and verify taps land directly in
//! each slot's circular history; prefill taps are staged and copied there on
//! commit. Before drafting, committed taps are projected into the context
//! cache. Only committed positions are ever projected, so cancelled and
//! rejected target rows cannot enter that cache.

use super::*;
use crate::dflash::{program, state, views};
use crate::selection::MimoPhase as _;

pub(super) struct Dflash {
    trace: bool,
    ranks: Vec<program::RankBuffers>,
    history: Vec<DeviceBufferSpanV1>,
    arenas: Vec<PhaseArena>,
    materialized: BTreeMap<u64, u64>,
    pending: Vec<(state::Rows, u64, bool)>,
    pub(super) graphs: RefCell<crate::graphs::GraphCache>,
    peers: program::Peers,
    state: state::State,
}

/// Opens the packaged drafter when it is DFlash; `None` leaves the drafter
/// image to MTP.
pub(super) fn acquire(
    services: &HostServicesV1,
    ranks: &[(
        BundleContextHandleV1,
        BundleMemoryPoolHandleV1,
        BundleStreamHandleV1,
    )],
    target: &[BTreeMap<String, DeviceBufferSpanV1>],
    held: &mut Vec<Held>,
    tp: crate::topology::Tp,
    items: Items,
) -> Result<Option<Dflash>, BundleAbiError> {
    let mut indexes = Vec::new();
    for rank in 0..tp.ranks() {
        let role = model_mimo26_contract::weight_plan::index_role(
            model_mimo26_contract::weight_plan::DRAFTER_IMAGE_ROLE,
            rank,
        );
        let bytes = acquire::read_component(super::log(services), &role, 1 << 20, held)?;
        indexes.push(
            crate::index::parse(&bytes, rank)
                .map_err(|_| BundleAbiError::InvalidRange("DFlash rank index"))?,
        );
    }
    if !indexes.iter().all(views::is_dflash) {
        return Ok(None);
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
    let image = acquire::open_image(super::log(services), crate::mtp::views::IMAGE_ROLE, held)?;
    let arenas = acquire::upload_arenas(super::log(services), image, ranks, held)?;
    if arenas.len() != tp.count() {
        return Err(BundleAbiError::InvalidRange("DFlash rank arenas"));
    }
    let rows: BTreeMap<_, _> = model_mimo26_contract::resources::working_rows(
        tp,
        items.get(),
        Some(model_mimo26_contract::weight_plan::Drafter::Dflash),
    )
    .into_iter()
    .map(|row| (row.name, row.layout))
    .collect();
    let bytes = |name: &str| rows[name].allocation_bytes();
    let mut buffers = Vec::new();
    let mut history = Vec::new();
    let mut input_arenas = Vec::new();
    for (rank, arena) in arenas.into_iter().enumerate() {
        let image_span = services
            .resolve_compiled_image_view(image, arena, rank as u32, b"")?
            .buffer;
        let views = views::spans(&indexes[rank], image_span, &target[rank], tp)
            .map_err(BundleAbiError::InvalidRange)?;
        let (_, pool, stream) = ranks[rank];
        let mut alloc = |bytes| allocate_span(services, pool, stream, bytes, held);
        let rank_buffers = program::RankBuffers {
            views,
            cache: alloc(bytes("dflash.cache"))?,
            taps: alloc(bytes("dflash.taps"))?,
            pairs: alloc(bytes("dflash.pairs"))?,
            gathered: alloc(bytes("dflash.gathered"))?,
            gather_flags: alloc(bytes("dflash.gather_flags"))?,
            gather_epoch: alloc(bytes("dflash.gather_epoch"))?,
            vocab_width: alloc(bytes("dflash.vocab_width"))?,
        };
        // Blocks read cached positions they have not written only through
        // masked keys, whose values still enter the products: keep them finite.
        for span in [
            rank_buffers.cache,
            rank_buffers.taps,
            rank_buffers.gather_flags,
            rank_buffers.gather_epoch,
        ] {
            services.zero_device_buffer(stream, span)?;
        }
        upload_bytes(
            services,
            stream,
            rank_buffers.vocab_width,
            &crate::topology::vocab_per_rank(tp).to_le_bytes(),
        )?;
        history.push(alloc(bytes("dflash.history"))?);
        input_arenas.push(PhaseArena {
            arena: alloc(bytes("dflash.inputs"))?,
            block_tables: alloc(bytes("dflash.block_tables"))?,
        });
        buffers.push(rank_buffers);
    }
    let peers = program::Peers {
        gathered: buffers.iter().map(|rank| rank.gathered).collect(),
        gather_flags: buffers.iter().map(|rank| rank.gather_flags).collect(),
        gather_epochs: buffers.iter().map(|rank| rank.gather_epoch).collect(),
    };
    Ok(Some(Dflash {
        trace: std::env::var("CHOPINFER_STEP_TRACE").as_deref() == Ok("1"),
        ranks: buffers,
        history,
        arenas: input_arenas,
        materialized: BTreeMap::new(),
        pending: Vec::new(),
        graphs: RefCell::new(crate::graphs::GraphCache::with_items(
            bundle_runtime::geometry::Ranks::new(tp.ranks()).expect("supported rank count"),
            items,
        )),
        peers,
        state: state::State::for_topology(tp, items.get()),
    }))
}

impl Dflash {
    /// The bytes of a rank's candidate all-gather flags and epoch.
    pub(super) fn candidate_state(&self) -> [u64; 2] {
        let rank = &self.ranks[0];
        [rank.gather_flags.length, rank.gather_epoch.length]
    }

    /// Each rank's drafter views, with the target views the drafter reads.
    pub(super) fn rank_views(&self) -> impl Iterator<Item = &BTreeMap<String, DeviceBufferSpanV1>> {
        self.ranks.iter().map(|rank| &rank.views)
    }

    pub(super) fn extend(
        &self,
        rank: usize,
        bound: &crate::program_lower::RankStage<'_>,
        stage: crate::program::Stage,
        row0: u64,
        launches: &mut Vec<crate::program_lower::Lowered>,
    ) -> Result<(), kernel_core::contract::Refusal> {
        let buffers = self
            .ranks
            .get(rank)
            .ok_or(kernel_core::contract::Refusal("DFlash rank is missing"))?;
        // Context projection reads committed features only, before drafting.
        if stage != crate::program::Stage::GatherLogits {
            program::extend(
                bound,
                buffers,
                &self.peers,
                stage,
                row0,
                self.history.get(rank).copied(),
                launches,
            )?;
        }
        Ok(())
    }
}

impl Instance {
    /// Opens the drafter step and projects missing committed context only
    /// for requests that will draft.
    pub(super) fn dflash_begin(
        &mut self,
        step: &crate::inputs::PackedStep,
        batch: &runtime_bundle_sdk::BatchRequestV1,
    ) -> Result<(), BundleAbiError> {
        let Some(mut dflash) = self.dflash.take() else {
            return Ok(());
        };
        let result = self.dflash_prepare(&mut dflash, step, batch);
        self.dflash = Some(dflash);
        result
    }

    fn dflash_prepare(
        &mut self,
        dflash: &mut Dflash,
        step: &crate::inputs::PackedStep,
        batch: &runtime_bundle_sdk::BatchRequestV1,
    ) -> Result<(), BundleAbiError> {
        let mut rows = Vec::new();
        let mut pending = Vec::new();
        for phase in [&step.prefill, &step.decode] {
            for item in 0..phase.request_ids.len() {
                let verify = phase.verify_rows[item];
                let drafted = batch.items()[phase.item_order[item]].bundle_drafts();
                // The draft block overwrites every later row of a verify item.
                if verify && !drafted {
                    return Err(BundleAbiError::InvalidRange(
                        "DFlash verifies only its own drafts",
                    ));
                }
                let row = state::Rows {
                    request: phase.request_ids[item],
                    slot: phase.slot_ids[item] as u32,
                    start: phase.prefix_len[item] as u64,
                    count: phase.q_len[item] as u32,
                    verify,
                    drafted,
                };
                pending.push((row, phase.q_start[item] as u64, phase.decode));
                rows.push(row);
            }
        }
        dflash
            .state
            .prepare(rows)
            .map_err(BundleAbiError::InvalidRange)?;
        dflash.pending = pending;
        let drafting = [&step.prefill, &step.decode]
            .into_iter()
            .flat_map(|phase| {
                (0..phase.request_ids.len())
                    .filter(|&item| batch.items()[phase.item_order[item]].bundle_drafts())
                    .map(|item| phase.request_ids[item])
            })
            .collect::<Vec<_>>();
        self.dflash_catch_up(dflash, &drafting)
    }

    /// Reads the drafted input tokens back and reports each verify item's
    /// drafts, before any logits view that covers its rows.
    pub(super) fn dflash_emit(
        &mut self,
        step: &mut crate::inputs::PackedStep,
        batch: &runtime_bundle_sdk::BatchRequestV1,
        sink: &mut ExecuteResultSinkV1,
    ) -> Result<(), BundleAbiError> {
        if self.dflash.is_none() || !batch.items().iter().any(|item| item.bundle_drafts()) {
            return Ok(());
        }
        let decode = &mut step.decode;
        let length = decode.token_ids.len() as u64 * 4;
        let mut bytes = vec![0; length as usize];
        let tokens = self.verify_inputs[0].token_ids;
        self.services.copy_device_to_host(
            self.streams[0],
            &mut bytes,
            DeviceBufferSpanV1::new(tokens.buffer, tokens.offset, length),
        )?;
        for (token, bytes) in decode.token_ids.iter_mut().zip(bytes.chunks_exact(4)) {
            *token = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if *token >= self.tp.geometry.vocab() {
                return Err(BundleAbiError::InvalidRange(
                    "DFlash draft token outside vocabulary",
                ));
            }
        }
        for (item, &wire_index) in decode.item_order.iter().enumerate() {
            if batch.items()[wire_index].bundle_drafts() {
                let start = decode.q_start[item] as usize;
                let end = start + decode.q_len[item] as usize;
                sink.emit_draft_tokens(
                    decode.request_ids[item],
                    &decode.token_ids[start + 1..end],
                )?;
            }
        }
        Ok(())
    }

    /// Closes the drafter's step; `kept` lists the verify items in order.
    pub(super) fn dflash_finish(
        &mut self,
        kept: &[(u64, u32)],
        commit: Option<&CommitPlan>,
    ) -> Result<(), BundleAbiError> {
        let Some(dflash) = &mut self.dflash else {
            return Ok(());
        };
        // A step that failed before opening its transaction has none.
        if !dflash.state.is_pending() {
            return Ok(());
        }
        if let Some(plan) = commit {
            // Host commit validation precedes this callback. Decode and verify
            // rows already sit in the history; committing advances the range
            // that projection may read. Prefill rows are copied there now.
            for (row, source, decode) in &dflash.pending {
                if *decode {
                    continue;
                }
                // Phase-grouped rows join the validated wire-order plan by request.
                let count = u64::from(
                    plan.items()
                        .iter()
                        .find(|item| item.request == row.request)
                        .expect("DFlash pending row belongs to the open ledger step")
                        .kept,
                );
                let skip = count.saturating_sub(state::history_rows(self.tp));
                let count = count - skip;
                let position = (row.start + skip) % state::history_capacity(self.tp);
                let first = count.min(state::history_capacity(self.tp) - position);
                for rank in 0..self.tp.count() {
                    let taps = dflash.ranks[rank].taps;
                    for (offset, destination, length) in
                        [(0, position, first), (first, 0, count - first)]
                    {
                        if length == 0 {
                            continue;
                        }
                        self.services.copy_device_to_device(
                            self.streams[rank],
                            feature_slice(
                                dflash.history[rank],
                                u64::from(row.slot) * state::history_capacity(self.tp)
                                    + destination,
                                length,
                                self.tp,
                            )?,
                            feature_slice(taps, source + skip + offset, length, self.tp)?,
                        )?;
                    }
                }
            }
            dflash
                .state
                .commit(kept)
                .map_err(BundleAbiError::InvalidRange)?;
        } else {
            dflash
                .state
                .cancel()
                .map_err(BundleAbiError::InvalidRange)?;
        }
        dflash.pending.clear();
        Ok(())
    }

    pub(super) fn dflash_release_graphs(&self) -> Result<(), BundleAbiError> {
        if let Some(dflash) = &self.dflash {
            dflash.graphs.borrow_mut().release_all(&self.services)?;
        }
        Ok(())
    }

    pub(super) fn dflash_prefix_span(&self, rank: usize) -> Option<DeviceBufferSpanV1> {
        self.dflash.as_ref().map(|dflash| dflash.history[rank])
    }

    pub(super) fn dflash_projected_span(&self, rank: usize) -> Option<DeviceBufferSpanV1> {
        self.dflash.as_ref().map(|dflash| dflash.ranks[rank].cache)
    }

    pub(super) fn dflash_snapshot_header(
        &self,
        request: u64,
        boundary: u64,
    ) -> Option<bundle_runtime::prefix::Header> {
        self.dflash
            .as_ref()
            .map(|dflash| bundle_runtime::prefix::Header {
                tokens: boundary,
                frontier: None,
                owner: None,
                cursor: dflash.materialized.get(&request).copied(),
            })
    }

    pub(super) fn dflash_prefix_ready(&self, request: u64, slot: u32, tokens: u64) -> bool {
        self.dflash.as_ref().is_none_or(|dflash| {
            !dflash.state.is_pending()
                && dflash.state.context(request).is_some_and(|context| {
                    context.slot == slot
                        && context.end == tokens
                        && context.start <= tokens.saturating_sub(state::history_rows(self.tp))
                })
        })
    }

    pub(super) fn dflash_prefix_restore(
        &mut self,
        request: u64,
        slot: u32,
        tokens: u64,
        materialized: Option<u64>,
    ) -> Result<(), BundleAbiError> {
        if let Some(dflash) = &mut self.dflash {
            dflash
                .state
                .restore(request, slot, tokens)
                .map_err(BundleAbiError::InvalidRange)?;
            if let Some(cursor) = materialized {
                dflash.materialized.insert(request, cursor);
            } else {
                dflash.materialized.remove(&request);
            }
        }
        Ok(())
    }

    pub(super) fn dflash_retire(&mut self, request: u64) -> Result<(), BundleAbiError> {
        if let Some(dflash) = &mut self.dflash {
            dflash
                .state
                .retire(request)
                .map_err(BundleAbiError::InvalidRange)?;
            dflash.materialized.remove(&request);
        }
        Ok(())
    }
}

/// One row's raw context feature on a rank: its `fc` K shard.
fn feature_bytes(tp: crate::topology::Tp) -> u64 {
    program::fc_shard(tp) * 2
}

fn feature_slice(
    span: DeviceBufferSpanV1,
    row: u64,
    rows: u64,
    tp: crate::topology::Tp,
) -> Result<DeviceBufferSpanV1, BundleAbiError> {
    let offset = row * feature_bytes(tp);
    let length = rows * feature_bytes(tp);
    if offset
        .checked_add(length)
        .is_none_or(|end| end > span.length)
    {
        return Err(BundleAbiError::InvalidRange("DFlash history slice"));
    }
    Ok(DeviceBufferSpanV1::new(
        span.buffer,
        span.offset + offset,
        length,
    ))
}

impl Instance {
    /// Projects the committed rows `requests` have not projected yet, in as
    /// few passes as fit: gaps that together fit the widest verify bucket
    /// share one captured graph, longer gaps share eager prefill passes of
    /// at most `PREFILL_TOKENS` rows. Each pass gathers its rows from the
    /// histories on the device.
    fn dflash_catch_up(
        &mut self,
        dflash: &mut Dflash,
        requests: &[u64],
    ) -> Result<(), BundleAbiError> {
        let started = dflash.trace.then(std::time::Instant::now);
        let mut gaps = Vec::new();
        for &request in requests {
            let context = dflash
                .state
                .context(request)
                .ok_or(BundleAbiError::InvalidRange("DFlash context missing"))?;
            let start = context
                .start
                .max(context.end.saturating_sub(state::history_rows(self.tp)))
                .max(dflash.materialized.get(&request).copied().unwrap_or(0));
            if context.end > start {
                gaps.push(Gap {
                    request,
                    slot: context.slot,
                    start,
                    end: context.end,
                });
            }
        }
        for pass in catch_up_passes(&gaps, catch_up_graph_rows(self.items)) {
            self.dflash_catch_up_pass(dflash, pass)?;
        }
        if let Some(started) = started {
            let record = serde_json::json!({
                "schema": "chopinfer.mimo.drafter-catch-up.v1",
                "elapsed_us": started.elapsed().as_micros() as u64,
                "request_ids": gaps.iter().map(|gap| gap.request).collect::<Vec<_>>(),
                "rows": gaps.iter().map(Gap::rows).collect::<Vec<_>>(),
                "graph_rows": catch_up_graph_rows(self.items),
            });
            let _ = self.services.log(record.to_string().as_bytes());
        }
        // These are all committed rows. Their projections remain valid even
        // when the following target transaction is cancelled.
        for gap in gaps {
            dflash.materialized.insert(gap.request, gap.end);
        }
        Ok(())
    }

    fn dflash_catch_up_pass(
        &mut self,
        dflash: &mut Dflash,
        gaps: &[Gap],
    ) -> Result<(), BundleAbiError> {
        if gaps.iter().any(|gap| gap.end > self.context_tokens) {
            return Err(BundleAbiError::InvalidRange(
                "DFlash RoPE position exceeds served context",
            ));
        }
        let count = gaps.iter().map(Gap::rows).sum::<u32>();
        let graph = count <= catch_up_graph_rows(self.items);
        let phase = if count == 1 {
            crate::selection::Phase::Decode(1)
        } else if graph {
            crate::selection::Phase::Verify(count)
        } else {
            crate::selection::Phase::Prefill
        };
        let mut inputs = crate::inputs::PhaseInputs {
            decode: graph,
            verify: graph && count > 1,
            ..crate::inputs::PhaseInputs::default()
        };
        for (item, gap) in gaps.iter().enumerate() {
            inputs.q_start.push(inputs.positions.len() as i32);
            inputs
                .positions
                .extend((gap.start..gap.end).map(|position| position as i32));
            inputs
                .token_batch
                .extend(std::iter::repeat_n(item as i32, gap.rows() as usize));
            inputs.item_order.push(item);
            inputs.request_ids.push(gap.request);
            inputs.emits_row.push(false);
            inputs.prompt_rows.push(false);
            inputs.verify_rows.push(false);
            inputs.block_tables.push(Vec::new());
            inputs.slot_ids.push(gap.slot as i32);
            inputs.q_len.push(gap.rows() as i32);
            inputs.prefix_len.push(gap.start as i32);
            inputs.seq_lens.push(gap.end as i32);
            inputs.last_positions.push(gap.end as i64 - 1);
        }
        inputs.token_ids = vec![0; count as usize];
        let layout = if graph {
            crate::upload::InputLayout::verify_within::<Mimo>(self.items)
        } else {
            crate::upload::InputLayout::prefill_within::<Mimo>(&inputs, 0, self.items)
        };
        let bytes = layout
            .pack(&inputs, &[])
            .ok_or(BundleAbiError::InvalidRange("DFlash catch-up inputs"))?;
        for rank in 0..self.tp.count() {
            upload_bytes(
                &self.services,
                self.streams[rank],
                dflash.arenas[rank].arena,
                &bytes,
            )?;
        }
        let planned = if graph {
            phase
                .capacity()
                .map_err(|error| BundleAbiError::InvalidRange(error.0))?
        } else {
            count
        };
        // Catch-up graphs are keyed by their token bucket; the item bucket
        // only has to be a valid one for it. Eager instances enqueue them.
        let graph = (graph && !self.eager).then_some((planned, planned.min(self.items.capacity())));
        let context = gaps.iter().map(|gap| gap.end).max().unwrap_or(0) as u32;
        let mut graphs = dflash.graphs.borrow_mut();
        crate::pipeline::run_ranks(
            &self.services,
            &self.streams,
            &self.symbols,
            graph,
            &mut graphs,
            &self.collectives,
            |rank| {
                self.plan_stage_in_mode(
                    rank,
                    phase,
                    planned,
                    gaps.len() as u32,
                    context,
                    &[],
                    false,
                    crate::program::Stage::GatherLogits,
                    DeviceInputs::within(dflash.arenas[rank], &layout),
                    0,
                    Some((&dflash.ranks[rank], dflash.history[rank])),
                )
                .map(crate::program_lower::program)
                .map_err(|error| BundleAbiError::InvalidRange(error.0).into())
            },
            |_| {},
        )
        .map_err(|error| match error {
            crate::dispatch::DispatchError::Host(error) => error,
            _ => BundleAbiError::InvalidRange("DFlash catch-up launch"),
        })?;
        Ok(())
    }
}

/// Committed rows `start..end` of `request`'s slot that its drafter cache
/// does not hold yet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Gap {
    request: u64,
    slot: u32,
    start: u64,
    end: u64,
}

impl Gap {
    fn rows(&self) -> u32 {
        (self.end - self.start) as u32
    }
}

/// The most rows a catch-up graph holds for an instance serving `items`:
/// its item bucket is at most the served capacity and each item carries at
/// most `MAX_VERIFY_ROWS` rows, so below 32 served this is under
/// `VERIFY_TOKENS`. Longer catch-ups run eagerly on the prefill kernels.
fn catch_up_graph_rows(items: Items) -> u32 {
    crate::contract::VERIFY_TOKENS.min(items.capacity() * crate::contract::MAX_VERIFY_ROWS)
}

/// Groups gaps into catch-up passes in order: consecutive gaps of at most
/// `graph_rows` rows while they fit one catch-up graph together, and longer
/// gaps while they fit one prefill chunk.
fn catch_up_passes(gaps: &[Gap], graph_rows: u32) -> Vec<&[Gap]> {
    let limit = |gap: &Gap| {
        if gap.rows() <= graph_rows {
            graph_rows
        } else {
            crate::contract::PREFILL_TOKENS
        }
    };
    let mut passes = Vec::new();
    let mut first = 0;
    let mut rows = 0;
    for (index, gap) in gaps.iter().enumerate() {
        let fits = index > first
            && limit(gap) == limit(&gaps[first])
            && rows + gap.rows() <= limit(&gaps[first]);
        if index > first && !fits {
            passes.push(&gaps[first..index]);
            first = index;
            rows = 0;
        }
        rows += gap.rows();
    }
    if first < gaps.len() {
        passes.push(&gaps[first..]);
    }
    passes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaps(rows: &[u64]) -> Vec<Gap> {
        rows.iter()
            .enumerate()
            .map(|(index, &rows)| Gap {
                request: index as u64,
                slot: index as u32,
                start: 100,
                end: 100 + rows,
            })
            .collect()
    }

    fn catch_up_passes_at(gaps: &[Gap]) -> Vec<&[Gap]> {
        catch_up_passes(gaps, catch_up_graph_rows(Items::new(32).unwrap()))
    }

    fn shape(passes: &[&[Gap]]) -> Vec<Vec<u32>> {
        passes
            .iter()
            .map(|pass| pass.iter().map(Gap::rows).collect())
            .collect()
    }

    #[test]
    fn catch_up_passes_fill_the_verify_bucket_then_the_prefill_chunk() {
        // Steady drafting: 32 items of up to eight kept rows share one pass.
        assert_eq!(shape(&catch_up_passes_at(&gaps(&[8; 32]))), [vec![8; 32]]);
        // Rows past 256 open a second pass.
        assert_eq!(
            shape(&catch_up_passes_at(&gaps(&[200, 50, 10, 1]))),
            [vec![200, 50], vec![10, 1]]
        );
        // Long gaps share prefill chunks of at most 8192 rows, apart from
        // the short ones around them.
        assert_eq!(
            shape(&catch_up_passes_at(&gaps(&[3, 1023, 1023, 4, 300]))),
            [vec![3], vec![1023, 1023], vec![4], vec![300]]
        );
        let long = gaps(&[1023; 9]);
        assert_eq!(
            shape(&catch_up_passes(&long, 256)),
            [vec![1023; 8], vec![1023]]
        );
        assert!(catch_up_passes(&[], 256).is_empty());
    }

    #[test]
    fn catch_up_graphs_shrink_with_the_served_count() {
        let rows = |served| catch_up_graph_rows(Items::new(served).unwrap());
        assert_eq!(
            [rows(1), rows(8), rows(16), rows(32), rows(256)],
            [8, 64, 128, 256, 256]
        );
        // At 16 served, 32 items of eight kept rows split into two graphs,
        // and a 150-row prompt projection runs eagerly on its own.
        assert_eq!(
            shape(&catch_up_passes(&gaps(&[8; 32]), rows(16))),
            [vec![8; 16], vec![8; 16]]
        );
        assert_eq!(
            shape(&catch_up_passes(&gaps(&[4, 150, 4]), rows(16))),
            [vec![4], vec![150], vec![4]]
        );
    }
}
