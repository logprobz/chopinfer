//! One step's per-request state: the ledger's items of a packed phase, the
//! KDA side inputs and checkpoint tables the lowered launches read, and the
//! state copies a commit makes.
//!
//! Decode advances each slot's KDA and indexer tail entries in place. A
//! verify phase writes every row's state to checkpoints instead: the KDA
//! checkpoints packed in the step's verify-row order, the tail's at the item's slot (every slot
//! of the bucket owns `MAX_VERIFY_ROWS` of them). The verify kernels write
//! a checkpoint for every row of every item, so a single-row item's one
//! "checkpoint" is its slot's own entry: it advances in place, as in
//! decode. Commit copies the last kept row's checkpoint into the slot;
//! cancel invalidates the slots the step touched (`Cancel::Invalidate`),
//! which the ledger retires.

use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::ledger::{Committed, Item, Step};
use bundle_runtime::state::{CheckpointRows, Commit, Extent, RequestState, Rollback, StateBuffer};

use crate::buffers::checkpoint_table;

/// The ledger items of `phase`, in its item order.
pub fn items(phase: &PhaseInputs) -> Vec<Item> {
    (0..phase.request_ids.len())
        .map(|item| Item {
            request: phase.request_ids[item],
            slot: u32::try_from(phase.slot_ids[item]).ok(),
            start: phase.prefix_len[item] as u64,
            rows: phase.q_len.get(item).copied().unwrap_or(1) as u32,
            verify: phase.verify && phase.verify_rows.get(item).copied().unwrap_or(false),
        })
        .collect()
}

/// Rows of one indexer piece of an eager prefill: the pooled indexer's
/// prefill capacity.
pub const PIECE_ROWS: u32 = 2048;

/// Most indexer pieces of one prefill: every piece but the last holds at
/// least `PIECE_ROWS - PREFILL_GRANULE + 1` rows.
pub const MAX_PIECES: u32 = model_glm5next_contract::contract::PREFILL_TOKENS
    .div_ceil(PIECE_ROWS - model_glm5next_contract::contract::PREFILL_GRANULE + 1);

/// The absolute position of each item's prefix anchor, for the items that
/// keep one (`PhaseInputs::anchors`): the last multiple of `granule`
/// strictly inside the item's rows, where the KDA engines take its state.
/// As the engines (`channel_decay_common.cuh`, `anchor_row`), an item that
/// ends on a multiple keeps none, so its flag is refused: its spares would
/// stay unwritten.
pub fn anchor_ends(phase: &PhaseInputs, granule: u32) -> Result<Vec<Option<i32>>, String> {
    (0..phase.item_order.len())
        .map(|item| {
            if !phase.anchors.get(item).copied().unwrap_or(false) {
                return Ok(None);
            }
            let (prefix, rows) = (phase.prefix_len[item], phase.q_len[item]);
            let granule = granule as i32;
            let tail = (prefix + rows) % granule;
            if tail == 0 || rows - tail <= 0 {
                return Err(format!(
                    "item {item} keeps an anchor but no multiple of {granule} lies strictly \
                     inside its rows from {prefix} to an end off the granule"
                ));
            }
            Ok(Some(prefix + rows - tail))
        })
        .collect()
}

/// The pieces the indexer stages of an eager prefill run in, each at most
/// `PIECE_ROWS` rows. A cut falls at an item boundary or inside an item at
/// an item-relative multiple of `granule` (64, a multiple of the pool of
/// 4), as far into the rows as that allows, so each piece is a successive
/// chunk of every item it touches: its own row offsets and committed
/// lengths (`PhaseInputs::split_at_with_rows`), the pooled key tail carried
/// through the item's tail state as between steps. A cut never falls on an
/// item's anchor: the KDA engines take the anchor's state only strictly
/// inside a call's rows.
/// A phase of at most `PIECE_ROWS` rows is one piece. Returns each piece's
/// first row with it.
pub fn pieces(phase: &PhaseInputs, granule: u32) -> Result<Vec<(u32, PhaseInputs)>, String> {
    let mut pieces = Vec::new();
    let mut first = 0;
    let mut rest = clone_phase(phase);
    while rest.live_tokens() > PIECE_ROWS {
        let anchors = anchor_ends(&rest, granule)?;
        // The furthest allowed cut within the piece's rows.
        let cut = (0..rest.q_start.len())
            .filter_map(|item| {
                let (start, rows) = (rest.q_start[item] as u32, rest.q_len[item] as u32);
                if start >= PIECE_ROWS {
                    return None;
                }
                let end = start + rows;
                if end <= PIECE_ROWS {
                    return Some(end);
                }
                // Item-relative multiples of the granule: the item resumes
                // there as a new chunk would.
                let mut inside = (PIECE_ROWS - start) / granule * granule;
                if anchors[item] == Some(rest.prefix_len[item] + inside as i32) {
                    inside = inside.saturating_sub(granule);
                }
                (inside > 0).then_some(start + inside)
            })
            .max()
            .ok_or("no granule-aligned cut within the indexer piece")?;
        let (head, tail) = rest
            .split_at_with_rows(cut, &anchors)
            .ok_or("the prefill does not split at the piece cut")?;
        pieces.push((first, head));
        first += cut;
        rest = tail;
    }
    pieces.push((first, rest));
    Ok(pieces)
}

fn clone_phase(phase: &PhaseInputs) -> PhaseInputs {
    PhaseInputs {
        decode: phase.decode,
        verify: phase.verify,
        verify_rows: phase.verify_rows.clone(),
        item_order: phase.item_order.clone(),
        request_ids: phase.request_ids.clone(),
        emits_row: phase.emits_row.clone(),
        prompt_rows: phase.prompt_rows.clone(),
        anchors: phase.anchors.clone(),
        token_ids: phase.token_ids.clone(),
        positions: phase.positions.clone(),
        position_axes: phase.position_axes.clone(),
        token_batch: phase.token_batch.clone(),
        block_tables: phase.block_tables.clone(),
        slot_ids: phase.slot_ids.clone(),
        q_start: phase.q_start.clone(),
        q_len: phase.q_len.clone(),
        prefix_len: phase.prefix_len.clone(),
        seq_lens: phase.seq_lens.clone(),
        last_positions: phase.last_positions.clone(),
    }
}

/// The KDA engine's per-sequence inputs of an eager prefill: each item's
/// row offsets with their end, whether it carries its slot's state (a
/// later chunk) or starts from zero, and the KDA entry of its anchor spare
/// (-1 for an item that keeps no anchor in these rows).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PrefillInputs {
    pub cu_seqlens: Vec<i32>,
    pub initial: Vec<i32>,
    pub anchor_ids: Vec<i32>,
}

/// The step's anchor items: each prefill item that keeps its state at its
/// anchor, as (request, anchor tokens), in item order. Anchor item `k` is
/// the `k`th, whose spares are the `k`th of each buffer.
pub fn anchors(phase: &PhaseInputs, granule: u32) -> Result<Vec<(u64, u64)>, String> {
    Ok(anchor_ends(phase, granule)?
        .into_iter()
        .enumerate()
        .filter_map(|(item, end)| end.map(|end| (phase.request_ids[item], end as u64)))
        .collect())
}

/// `phase`'s KDA inputs, with `spares` the KDA entry of each anchor
/// item's spare by request (both KDA buffers keep it at the same entry).
pub fn prefill_inputs(phase: &PhaseInputs, spares: &[(u64, i32)]) -> Result<PrefillInputs, String> {
    let mut cu_seqlens = phase.q_start.clone();
    cu_seqlens.push(phase.live_tokens() as i32);
    let anchor_ids = (0..phase.q_start.len())
        .map(|item| {
            if !phase.anchors.get(item).copied().unwrap_or(false) {
                return Ok(-1);
            }
            let request = phase.request_ids[item];
            spares
                .iter()
                .find(|&&(owner, _)| owner == request)
                .map(|&(_, entry)| entry)
                .ok_or_else(|| format!("request {request} keeps an anchor without a spare"))
        })
        .collect::<Result<_, String>>()?;
    Ok(PrefillInputs {
        cu_seqlens,
        initial: phase
            .prefix_len
            .iter()
            .map(|&prefix| i32::from(prefix != 0))
            .collect(),
        anchor_ids,
    })
}

/// The KDA engine's per-sequence inputs of a decode or verify phase.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SideInputs {
    /// Row offsets of the items, and their end (`[items + 1]`).
    pub cu_rows: Vec<i32>,
    /// The KDA checkpoint table at the verify entries' stride.
    pub checkpoint_ids: Vec<i32>,
    /// The indexer tail's checkpoint table at its own stride.
    pub tail_checkpoint_ids: Vec<i32>,
}

/// Buffer indexes of the KDA and tail state in the declaration.
const KDA: [&str; 2] = ["kda.conv", "kda.state"];
const TAIL: &str = "indexer.tail";

fn buffer<'a>(
    state: &'a RequestState,
    name: &str,
) -> Result<&'a bundle_runtime::state::StateBuffer, String> {
    state
        .buffers
        .iter()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| format!("no state buffer {name}"))
}

/// Keep every item's original row reservation through commit, even when
/// an earlier item keeps a shorter prefix. Decode siblings consume no checkpoint.
fn verify_row_bases(step: &Step) -> Result<Vec<Option<u32>>, String> {
    let mut next = 0_u32;
    step.items
        .iter()
        .map(|item| {
            if !item.verify {
                return Ok(None);
            }
            let first = next;
            next = next
                .checked_add(item.rows)
                .ok_or("verify-row offsets overflow")?;
            Ok(Some(first))
        })
        .collect()
}

fn checkpoint_owner(buffer: &StateBuffer, verify: u32, row_base: u32) -> u32 {
    if matches!(buffer.rollback, Rollback::Checkpoint(c) if matches!(c.rows, CheckpointRows::Packed(_)))
    {
        row_base
    } else {
        verify
    }
}

/// The side inputs of the open `step`'s items `first..first + rows.len()`,
/// each of `rows[item]` rows. Verify items take KDA checkpoints at their
/// packed row offset and tail checkpoints at their slot; any other item's rows
/// write its slot's own entry. The two KDA buffers share one table.
pub fn side_inputs(
    state: &RequestState,
    extent: Extent,
    step: &Step,
    first: usize,
    rows: &[u32],
    kda_stride: u32,
    tail_stride: u32,
) -> Result<SideInputs, String> {
    let mut cu_rows = vec![0];
    let mut kda = Vec::new();
    let mut tail = Vec::new();
    let conv = buffer(state, KDA[0])?;
    let delta = buffer(state, KDA[1])?;
    let tails = buffer(state, TAIL)?;
    let bases = verify_row_bases(step)?;
    for (offset, &count) in rows.iter().enumerate() {
        let index = first + offset;
        let placed = step
            .items
            .get(index)
            .ok_or_else(|| format!("no item {index} in the step"))?;
        cu_rows.push(cu_rows.last().copied().unwrap_or(0) + count as i32);
        let (mut kda_ids, mut tail_ids) = (Vec::new(), Vec::new());
        if step.verify_index(index).is_none() {
            if count != 1 {
                return Err(format!("item {index} has {count} rows but no checkpoints"));
            }
            kda_ids.push(placed.slot as i32);
            tail_ids.push(placed.slot as i32);
        }
        if let Some(verify) = step.verify_index(index) {
            if count != placed.rows {
                return Err(format!(
                    "verify item {index} has {count} rows, expected {}",
                    placed.rows
                ));
            }
            let row_base = bases[index].ok_or("verify item has no row offset")?;
            let owner = checkpoint_owner(conv, verify, row_base);
            for row in 0..count {
                let missing = || format!("item {index} row {row} has no checkpoint");
                let entry = conv.checkpoint(extent, owner, row).ok_or_else(missing)?;
                if delta.checkpoint(extent, checkpoint_owner(delta, verify, row_base), row)
                    != Some(entry)
                {
                    return Err("the KDA buffers' checkpoints differ".into());
                }
                kda_ids.push(entry as i32);
                let entry = tails
                    .checkpoint(extent, placed.slot, row)
                    .ok_or_else(missing)?;
                tail_ids.push(entry as i32);
            }
        }
        kda.push(kda_ids);
        tail.push(tail_ids);
    }
    Ok(SideInputs {
        cu_rows,
        checkpoint_ids: checkpoint_table(kda_stride, &kda),
        tail_checkpoint_ids: checkpoint_table(tail_stride, &tail),
    })
}

/// One entry-to-entry copy of a state buffer's layers `0..layers`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EntryCopy {
    /// The buffer's index in the declaration.
    pub buffer: usize,
    pub from: u64,
    pub to: u64,
    pub layers: u64,
}

/// The layers of `buffer` the target's verify rows checkpoint: all but the
/// MTP drafter's indexer tail, which the indexer tail holds last. The
/// drafter writes that layer in place, following its own frontier, and the
/// target never writes its checkpoints.
fn target_layers(state: &RequestState, buffer: &bundle_runtime::state::StateBuffer) -> u64 {
    if buffer.name != TAIL {
        return buffer.layers;
    }
    let drafter = state
        .pages
        .iter()
        .filter(|pages| pages.name == "mtp")
        .map(|pages| pages.layers)
        .sum::<u64>();
    buffer.layers - drafter
}

/// The checkpointing buffers of `state` a commit copies, each with the
/// layers it copies (`target_layers`), by their index in the declaration.
pub fn checkpoint_layers(state: &RequestState) -> Vec<(usize, u64)> {
    state
        .buffers
        .iter()
        .enumerate()
        .filter(|(_, buffer)| matches!(buffer.rollback, Rollback::Checkpoint(_)))
        .map(|(index, buffer)| (index, target_layers(state, buffer)))
        .collect()
}

/// The copies a commit makes: each verify item that keeps rows takes the
/// last kept row's checkpoint into its slot in every checkpointing buffer,
/// in the layers the target writes (`target_layers`). Assignments use the
/// original step's rows, so acceptance does not move another item's state.
pub fn commit_copies(
    state: &RequestState,
    extent: Extent,
    committed: &[Committed],
    step: &Step,
) -> Result<Vec<EntryCopy>, String> {
    let mut copies = Vec::new();
    let bases = verify_row_bases(step)?;
    if committed.len() != step.items.len() {
        return Err("commit and checkpoint step have different item counts".into());
    }
    for (item, done) in committed.iter().enumerate() {
        if done.request != step.items[item].request
            || done.rows != step.items[item].rows
            || done.slot != step.items[item].slot
        {
            return Err("commit does not name its checkpoint owner".into());
        }
        let Some(verify) = step.verify_index(item) else {
            continue;
        };
        if done.kept == 0 {
            continue;
        }
        for (index, buffer) in state.buffers.iter().enumerate() {
            let Commit::Checkpoint { row } = buffer.rollback.commit(done.rows, done.kept) else {
                continue;
            };
            let owner = match buffer.rollback {
                Rollback::Checkpoint(checkpoints) => match checkpoints.items {
                    bundle_runtime::state::CheckpointItems::Bucket => done.slot,
                    bundle_runtime::state::CheckpointItems::Drafting(_) => checkpoint_owner(
                        buffer,
                        verify,
                        bases[item].ok_or("verify item has no row offset")?,
                    ),
                },
                _ => continue,
            };
            let from = buffer
                .checkpoint(extent, owner, row)
                .ok_or_else(|| format!("{}: no checkpoint for item {item}", buffer.name))?;
            copies.push(EntryCopy {
                buffer: index,
                from,
                to: u64::from(done.slot),
                layers: target_layers(state, buffer),
            });
        }
    }
    Ok(copies)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::ledger::{Ledger, Placement, Rules};
    use model_glm5next_contract::contract::{DECODE_BATCH, MAX_VERIFY_ROWS, Shape};
    use model_glm5next_contract::state::request_state;

    const EXTENT: Extent = Extent {
        served: DECODE_BATCH,
        bucket: DECODE_BATCH,
        verify_rows: 4,
    };

    fn state() -> RequestState {
        let shape = Shape::TP4;
        request_state(shape, &shape.kv_formats()[0], false)
    }

    fn ledger() -> Ledger {
        Ledger::new(DECODE_BATCH, Placement::Host, Rules::of(&state()), 65_536)
    }

    fn item(request: u64, slot: u32, start: u64, rows: u32, verify: bool) -> Item {
        Item {
            request,
            slot: Some(slot),
            start,
            rows,
            verify,
        }
    }

    #[test]
    fn compact_checkpoints_keep_mixed_width_assignments_after_rejection() {
        let state = state();
        let mut ledger = ledger();
        ledger
            .begin(
                &[
                    item(7, 3, 0, 2, true),
                    item(8, 5, 0, 1, false),
                    item(9, 1, 0, 3, true),
                    item(10, 8, 0, 4, true),
                ],
                Vec::new(),
            )
            .unwrap();
        let step = ledger.step().unwrap().clone();
        let side =
            side_inputs(&state, EXTENT, &step, 0, &[2, 1, 3, 4], 8, MAX_VERIFY_ROWS).unwrap();
        assert_eq!(&side.checkpoint_ids[..2], &[32, 33]);
        assert_eq!(side.checkpoint_ids[8], 5);
        assert_eq!(&side.checkpoint_ids[16..19], &[34, 35, 36]);
        assert_eq!(&side.checkpoint_ids[24..28], &[37, 38, 39, 40]);
        assert!(ledger.plan_commit(&[(7, 0), (9, 1), (10, 4)]).is_err());
        let committed = ledger.plan_commit(&[(7, 1), (9, 1), (10, 4)]).unwrap();
        let copies = commit_copies(&state, EXTENT, committed.items(), &step).unwrap();
        let kda = copies
            .iter()
            .filter(|copy| copy.buffer == 0)
            .map(|copy| (copy.from, copy.to))
            .collect::<Vec<_>>();
        assert_eq!(kda, [(32, 3), (34, 1), (40, 8)]);
        let sliced = side_inputs(&state, EXTENT, &step, 2, &[3, 4], 8, MAX_VERIFY_ROWS).unwrap();
        assert_eq!(&sliced.checkpoint_ids[..3], &[34, 35, 36]);
        assert_eq!(&sliced.checkpoint_ids[8..12], &[37, 38, 39, 40]);
    }

    #[test]
    fn sixteen_two_row_items_fill_the_compact_budget() {
        let state = state();
        let mut ledger = ledger();
        let items = (0..16)
            .map(|i| item(100 + u64::from(i), 31 - i, 0, 2, true))
            .collect::<Vec<_>>();
        ledger.begin(&items, Vec::new()).unwrap();
        let step = ledger.step().unwrap();
        let side = side_inputs(&state, EXTENT, step, 0, &[2; 16], 8, MAX_VERIFY_ROWS).unwrap();
        let writes = side
            .checkpoint_ids
            .into_iter()
            .filter(|&v| v >= 0)
            .collect::<Vec<_>>();
        assert_eq!(writes, (32..64).collect::<Vec<_>>());
        let mut smaller = state;
        for buffer in &mut smaller.buffers[..2] {
            if let Rollback::Checkpoint(ref mut c) = buffer.rollback {
                c.rows = CheckpointRows::Packed(31);
            }
        }
        assert!(side_inputs(&smaller, EXTENT, step, 0, &[2; 16], 8, MAX_VERIFY_ROWS).is_err());
    }

    #[test]
    fn six_row_items_keep_their_reservations_through_commit() {
        let state = state();
        let extent = Extent {
            verify_rows: MAX_VERIFY_ROWS,
            ..EXTENT
        };
        let mut ledger = ledger();
        let widths = [6, 6, 6, 6, 6, 2];
        let items = widths
            .iter()
            .enumerate()
            .map(|(i, &rows)| item(100 + i as u64, 31 - i as u32, 0, rows, true))
            .collect::<Vec<_>>();
        ledger.begin(&items, Vec::new()).unwrap();
        let step = ledger.step().unwrap().clone();
        let side = side_inputs(&state, extent, &step, 0, &widths, 8, MAX_VERIFY_ROWS).unwrap();
        assert_eq!(
            side.checkpoint_ids
                .iter()
                .copied()
                .filter(|&v| v >= 0)
                .collect::<Vec<_>>(),
            (32..64).collect::<Vec<_>>()
        );
        let committed = ledger
            .plan_commit(&[(100, 1), (101, 2), (102, 3), (103, 4), (104, 5), (105, 1)])
            .unwrap();
        let copies = commit_copies(&state, extent, committed.items(), &step).unwrap();
        let sources = copies
            .iter()
            .filter(|c| c.buffer == 0)
            .map(|c| c.from)
            .collect::<Vec<_>>();
        assert_eq!(sources, [32, 39, 46, 53, 60, 62]);
    }

    /// A decode item's one row writes its slot's own entry.
    #[test]
    fn decode_items_write_their_slots() {
        let mut ledger = ledger();
        let items = [item(7, 0, 0, 1, false), item(8, 5, 0, 1, false)];
        ledger.begin(&items, Vec::new()).unwrap();
        let side = side_inputs(&state(), EXTENT, ledger.step().unwrap(), 0, &[1, 1], 8, 6).unwrap();
        assert_eq!(side.cu_rows, [0, 1, 2]);
        assert_eq!(side.checkpoint_ids[..9], [0, -1, -1, -1, -1, -1, -1, -1, 5]);
        assert_eq!(side.checkpoint_ids.len(), 16);
        assert_eq!(side.tail_checkpoint_ids[..7], [0, -1, -1, -1, -1, -1, 5]);
        assert_eq!(side.tail_checkpoint_ids.len(), 12);
    }

    /// Verify items take KDA checkpoints in verify order past the served
    /// slots, and tail checkpoints at their slot past the bucket's entries;
    /// commit copies the last kept row's into the slot, and retirement
    /// frees it.
    #[test]
    fn verify_checkpoints_commit_the_last_kept_row() {
        let state = state();
        let mut ledger = ledger();
        // A decode item between two verify items of 4 rows.
        let items = [
            item(7, 3, 0, 4, true),
            item(8, 5, 0, 1, false),
            item(9, 1, 0, 4, true),
        ];
        ledger.begin(&items, Vec::new()).unwrap();
        let step = ledger.step().unwrap().clone();
        let side = side_inputs(&state, EXTENT, &step, 0, &[4, 1, 4], 8, MAX_VERIFY_ROWS).unwrap();
        assert_eq!(side.cu_rows, [0, 4, 5, 9]);
        let served = i32::try_from(DECODE_BATCH).unwrap();
        // Verify index 0 and 1: checkpoints after the 32 slots, 4 per item.
        assert_eq!(
            side.checkpoint_ids[..5],
            [served, served + 1, served + 2, served + 3, -1]
        );
        // The decode item between them writes its slot, 5.
        assert_eq!(side.checkpoint_ids[8..10], [5, -1]);
        assert_eq!(side.tail_checkpoint_ids[6..8], [5, -1]);
        assert_eq!(
            side.checkpoint_ids[16..20],
            [served + 4, served + 5, served + 6, served + 7]
        );
        // Tail: slot 3's 6 checkpoints after the bucket's 32 entries.
        let tail = served + 3 * 6;
        assert_eq!(
            side.tail_checkpoint_ids[..4],
            [tail, tail + 1, tail + 2, tail + 3]
        );
        assert_eq!(side.tail_checkpoint_ids[12..14], [served + 6, served + 7]);
        // Item 7 keeps 2 rows, item 9 keeps 4.
        let committed = ledger.commit(&[(7, 2), (9, 4)]).unwrap();
        let copies = commit_copies(&state, EXTENT, &committed, &step).unwrap();
        let (linear, sparse) = (Shape::TP4.linear_layers(), Shape::TP4.sparse_layers());
        let expected = [
            EntryCopy {
                buffer: 0,
                from: 33,
                to: 3,
                layers: linear,
            },
            EntryCopy {
                buffer: 1,
                from: 33,
                to: 3,
                layers: linear,
            },
            EntryCopy {
                buffer: 2,
                from: 32 + 3 * 6 + 1,
                to: 3,
                layers: sparse,
            },
            EntryCopy {
                buffer: 0,
                from: 32 + 4 + 3,
                to: 1,
                layers: linear,
            },
            EntryCopy {
                buffer: 1,
                from: 32 + 4 + 3,
                to: 1,
                layers: linear,
            },
            EntryCopy {
                buffer: 2,
                from: 32 + 6 + 3,
                to: 1,
                layers: sparse,
            },
        ];
        assert_eq!(copies, expected);
        assert_eq!(ledger.slot_of(7), Some(3));
        assert_eq!(ledger.retire(7), Ok(Some(3)));
        assert_eq!(ledger.slot_of(7), None);
    }

    /// With the MTP drafter's layer, the indexer tail holds it last, and
    /// commit copies the target's checkpoints into every layer but that
    /// one: the drafter writes its tail in place, along its frontier.
    #[test]
    fn commit_copies_leave_the_drafters_tail_alone() {
        use model_glm5next_contract::contract::has_mtp_layer;
        let shape = Shape::TP4;
        let format = shape.kv_formats()[1];
        assert!(has_mtp_layer(&format));
        let state = request_state(shape, &format, true);
        let mut ledger = Ledger::new(DECODE_BATCH, Placement::Host, Rules::of(&state), 65_536);
        ledger.begin(&[item(7, 3, 0, 4, true)], Vec::new()).unwrap();
        let step = ledger.step().unwrap().clone();
        for kept in 1..=4 {
            let committed = ledger.plan_commit(&[(7, kept)]).unwrap();
            let copies = commit_copies(&state, EXTENT, committed.items(), &step).unwrap();
            let tail = state
                .buffers
                .iter()
                .position(|buffer| buffer.name == TAIL)
                .unwrap();
            assert_eq!(state.buffers[tail].layers, shape.sparse_layers() + 1);
            let copy = copies.iter().find(|copy| copy.buffer == tail).unwrap();
            assert_eq!(copy.layers, shape.sparse_layers());
            assert_eq!(copy.from, 32 + 3 * 6 + u64::from(kept) - 1);
            for copy in copies.iter().filter(|copy| copy.buffer != tail) {
                assert_eq!(copy.layers, state.buffers[copy.buffer].layers);
            }
        }
    }

    /// A prefill of one prompt of `rows` rows from `prefix` in slot 4, as
    /// the host would pack it.
    fn prompt(request: u64, prefix: u32, rows: u32) -> PhaseInputs {
        prompts(&[(request, prefix, rows)])
    }

    fn prompts(items: &[(u64, u32, u32)]) -> PhaseInputs {
        let mut phase = PhaseInputs::default();
        for (index, &(request, prefix, rows)) in items.iter().enumerate() {
            let batch = phase.item_order.len() as i32;
            phase.item_order.push(index);
            phase.request_ids.push(request);
            phase.emits_row.push(true);
            phase.prompt_rows.push(false);
            phase.anchors.push(false);
            phase.verify_rows.push(false);
            phase.slot_ids.push(4 + index as i32);
            phase.q_start.push(phase.token_ids.len() as i32);
            phase.q_len.push(rows as i32);
            phase.prefix_len.push(prefix as i32);
            phase.seq_lens.push((prefix + rows) as i32);
            phase.last_positions.push(i64::from(prefix + rows) - 1);
            phase.block_tables.push(vec![index as i32]);
            for offset in 0..rows {
                phase.token_ids.push(1000 + prefix + offset);
                phase.positions.push((prefix + offset) as i32);
                phase.token_batch.push(batch);
            }
        }
        phase
    }

    /// What a piece holds that a chunk across steps also holds.
    fn work(phase: &PhaseInputs) -> Vec<Vec<i64>> {
        let wide = |values: &[i32]| values.iter().map(|&v| i64::from(v)).collect::<Vec<_>>();
        vec![
            phase.request_ids.iter().map(|&v| v as i64).collect(),
            wide(&phase.slot_ids),
            wide(&phase.q_start),
            wide(&phase.q_len),
            wide(&phase.prefix_len),
            wide(&phase.seq_lens),
            phase.token_ids.iter().map(|&v| i64::from(v)).collect(),
            wide(&phase.positions),
            wide(&phase.token_batch),
            phase.last_positions.clone(),
        ]
    }

    /// A 5000-row prompt splits at 2048 and 4096 into pieces that equal the
    /// same prompt delivered as chunks of 2048, 2048 and 904 rows across
    /// steps.
    #[test]
    fn a_long_prompt_splits_into_the_chunks_steps_would_deliver() {
        let pieces = pieces(&prompt(7, 0, 5000), 64).unwrap();
        let firsts = pieces.iter().map(|(first, _)| *first).collect::<Vec<_>>();
        assert_eq!(firsts, [0, 2048, 4096]);
        for ((_, piece), (prefix, rows)) in
            pieces.iter().zip([(0, 2048), (2048, 2048), (4096, 904)])
        {
            assert_eq!(work(piece), work(&prompt(7, prefix, rows)), "{prefix}");
        }
    }

    /// Cuts inside an item fall at its granule multiples, else at item
    /// boundaries; no piece passes 2048 rows.
    #[test]
    fn cuts_fall_on_item_granules_or_boundaries() {
        // 1000 rows of request 7 (ending its prompt), 3000 of request 8
        // from token 128: 8 is cut 1024 rows in, at item-relative 1024.
        let phase = prompts(&[(7, 0, 1000), (8, 128, 3000)]);
        let cut = pieces(&phase, 64).unwrap();
        let cuts = cut
            .iter()
            .map(|(first, piece)| (*first, piece.live_tokens()))
            .collect::<Vec<_>>();
        assert_eq!(cuts, [(0, 2024), (2024, 1976)]);
        assert_eq!(cut[0].1.q_len, [1000, 1024]);
        assert_eq!(cut[1].1.prefix_len, [128 + 1024]);
        assert!(!cut[0].1.emits_row[1]);
        // Four 600-row prompts: the cut falls at an item boundary or at an
        // item-relative multiple of 64.
        let phase = prompts(&[(1, 0, 600), (2, 0, 600), (3, 0, 600), (4, 0, 600)]);
        let cut = pieces(&phase, 64).unwrap();
        let cuts = cut
            .iter()
            .map(|(first, piece)| (*first, piece.live_tokens()))
            .collect::<Vec<_>>();
        assert_eq!(cuts, [(0, 1992), (1992, 408)]);
        assert_eq!(cut[1].1.prefix_len, [192]);
        // One piece for a short prefill, and the KDA inputs carry from a
        // later chunk only.
        assert_eq!(pieces(&prompt(7, 64, 100), 64).unwrap().len(), 1);
        let inputs = prefill_inputs(&prompts(&[(7, 0, 100), (8, 64, 30)]), &[]).unwrap();
        assert_eq!(inputs.cu_seqlens, [0, 100, 130]);
        assert_eq!(inputs.initial, [0, 1]);
        assert_eq!(inputs.anchor_ids, [-1, -1]);
    }

    /// An anchored item is never cut at its anchor: the KDA engines take
    /// the anchor's state only strictly inside a call's rows, so the cut
    /// moves a granule earlier, and only the piece holding the anchor keeps
    /// it and names its spare. An anchor that no granule multiple inside
    /// the item's rows can hold is refused.
    #[test]
    fn piece_cuts_avoid_anchors_and_only_the_holding_piece_names_its_spare() {
        let mut phase = prompt(7, 0, 2049);
        phase.anchors[0] = true;
        assert_eq!(anchor_ends(&phase, 64).unwrap(), [Some(2048)]);
        assert_eq!(anchors(&phase, 64).unwrap(), [(7, 2048)]);
        let cut = pieces(&phase, 64).unwrap();
        let shapes = cut
            .iter()
            .map(|(first, piece)| (*first, piece.live_tokens(), piece.anchors.clone()))
            .collect::<Vec<_>>();
        assert_eq!(shapes, [(0, 1984, vec![false]), (1984, 65, vec![true])]);
        let spares = [(7, 40)];
        assert_eq!(prefill_inputs(&cut[0].1, &spares).unwrap().anchor_ids, [-1]);
        assert_eq!(prefill_inputs(&cut[1].1, &spares).unwrap().anchor_ids, [40]);
        assert_eq!(prefill_inputs(&phase, &spares).unwrap().anchor_ids, [40]);
        assert!(prefill_inputs(&phase, &[]).is_err());
        // An anchor further in keeps the furthest cut.
        let mut phase = prompt(7, 0, 3009);
        phase.anchors[0] = true;
        let firsts = pieces(&phase, 64)
            .unwrap()
            .iter()
            .map(|(first, piece)| (*first, piece.anchors.clone()))
            .collect::<Vec<_>>();
        assert_eq!(firsts, [(0, vec![false]), (2048, vec![true])]);
        // No multiple of the granule strictly inside 64..100's rows, and
        // none the engines take in an item that ends on one (0..128).
        for (prefix, rows) in [(64, 36), (0, 128), (64, 64)] {
            let mut phase = prompt(7, prefix, rows);
            phase.anchors[0] = true;
            assert!(anchor_ends(&phase, 64).is_err(), "{prefix}+{rows}");
        }
        // The engines' row (`anchor_row`): the last multiple before an end
        // off one, after the item's first row.
        for (prefix, rows, end) in [
            (0, 65, 64),
            (64, 70, 128),
            (0, 3009, 3008),
            (3072, 71, 3136),
        ] {
            let mut phase = prompt(7, prefix, rows);
            phase.anchors[0] = true;
            assert_eq!(anchor_ends(&phase, 64).unwrap(), [Some(end)]);
        }
    }

    /// Cancel keeps nothing: the step's slots were advanced in place, so
    /// their requests retire and start again.
    #[test]
    fn cancel_frees_the_slots_the_step_touched() {
        let mut ledger = ledger();
        ledger
            .begin(&[item(7, 0, 0, 1, false)], Vec::new())
            .unwrap();
        ledger.commit(&[]).unwrap();
        assert_eq!(ledger.slot_of(7), Some(0));
        ledger
            .begin(&[item(7, 0, 1, 1, false)], Vec::new())
            .unwrap();
        ledger.cancel().unwrap();
        assert_eq!(ledger.slot_of(7), None);
        assert!(ledger.step().is_none());
    }
}
