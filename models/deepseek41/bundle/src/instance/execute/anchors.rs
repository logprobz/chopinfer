//! Prompt-end anchors an upper-skip prefill keeps inside its items
//! (`execution_options::PREFILL_ANCHOR_ITEMS`), so the scheduler need not
//! end a chunk at an unaligned prompt's rounded end and run its last 1-63
//! tokens as a separate step.
//!
//! Each anchor item takes one spare entry of every captured state buffer,
//! held in separate buffers of the same layout
//! (`state::request_state_with_anchors`). Before the step, its slot's
//! compressor tails are copied into the spare: a chunk ending on a granule
//! leaves them untouched. When the item starts less than a ring before its
//! anchor, its slot's rings are copied too. Every layer's window write is
//! then repeated into the spare ring for the rows within a ring of the
//! anchor, from the same key rows. The spare thus holds the bytes the slot
//! would hold after a chunk ending at the anchor, and a capture there copies
//! it (`Holder::Anchor`).
//!
//! With DSpark the spare also holds a raw feature history. The step packs
//! the raw rows a chunk ending at the anchor retains
//! (`prefill_cone::anchor_feature_start`). Commit first retains the head,
//! copies its full history entry into the spare, catches up the drafter,
//! then retains the tail at its original staging offset. Both histories
//! match the item cut at the anchor. Snapshots hold no drafter rings; a
//! restore rebuilds them from the raw rows.

use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::resources::{Held, allocate_span};
use bundle_runtime::state::{Extent, Holder, RequestState};
use model_deepseek41_contract::serving::PREFILL_GRANULE;
use model_deepseek41_contract::state::Widths;
use runtime_bundle_abi::{
    BundleAbiError, BundleMemoryPoolHandleV1, BundleStreamHandleV1, DeviceBufferSpanV1 as Span,
    HostServicesV1,
};

/// The request state buffers whose spares the step writes.
const RING: &str = "window.ring";
const TAIL: &str = "compressor.tail";

/// One rank's anchor spares.
#[derive(Clone, Debug)]
pub(crate) struct Spares {
    /// Per buffer of the declaration, its separate spare buffer.
    pub buffers: Vec<Option<Span>>,
    /// I32 zeros, one a ring cell: the sequence of each row written into a
    /// one-sequence spare ring.
    pub zeros: Span,
}

impl Spares {
    /// Allocates and zeroes one rank's spares of `state`, whose captured
    /// buffers each keep `anchors` spare entries.
    pub(crate) fn allocate(
        services: &HostServicesV1,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        state: &RequestState,
        cells: u64,
        held: &mut Vec<Held>,
    ) -> Result<Self, BundleAbiError> {
        let mut buffers = Vec::with_capacity(state.buffers.len());
        for buffer in &state.buffers {
            let bytes = buffer.spare_bytes();
            buffers.push(if bytes == 0 {
                None
            } else {
                let span = allocate_span(services, pool, stream, bytes, held)?;
                services.zero_device_buffer(stream, span)?;
                Some(span)
            });
        }
        let zeros = allocate_span(services, pool, stream, cells * 4, held)?;
        services.zero_device_buffer(stream, zeros)?;
        Ok(Self { buffers, zeros })
    }
}

/// One anchor item of a step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Anchor {
    /// The item's index in its phase.
    pub item: usize,
    /// Its spare entry: the anchor items' order in the step.
    pub spare: u32,
    pub request: u64,
    /// The absolute anchor: the last multiple of the prefill granule
    /// strictly inside the item.
    pub tokens: u32,
    /// The anchor's packed row: the item's row of position `tokens`.
    pub row: u32,
    /// The item's first packed row.
    pub first: u32,
}

impl Anchor {
    /// The packed rows of the anchor's ring that this item computes.
    pub(crate) fn ring_rows(&self, cells: u32) -> std::ops::Range<u32> {
        self.row.saturating_sub(cells).max(self.first)..self.row
    }

    /// Whether some cells of the anchor's ring hold positions before the
    /// item, which only the slot's committed ring holds.
    pub(crate) fn reads_committed_ring(&self, cells: u32) -> bool {
        self.row - self.first < cells
    }
}

/// The cells of each window ring.
pub(crate) fn ring_cells(g: &model_deepseek41_contract::geometry::Geometry) -> u32 {
    model_deepseek41_contract::state::ring_cells(Widths::of(g)) as u32
}

/// The anchor items of `inputs`, in item order, within the instance's
/// allowance. Each must end its prompt past a granule strictly inside it.
pub(crate) fn anchors(inputs: &PhaseInputs, spares: u32) -> Result<Vec<Anchor>, String> {
    let granule = PREFILL_GRANULE;
    let mut found = Vec::new();
    for item in 0..inputs.request_ids.len() {
        if !inputs.anchors.get(item).copied().unwrap_or(false) {
            continue;
        }
        let (prefix, rows, first) = (
            u32::try_from(inputs.prefix_len[item]).map_err(|_| "negative prefix")?,
            u32::try_from(inputs.q_len[item]).map_err(|_| "negative rows")?,
            u32::try_from(inputs.q_start[item]).map_err(|_| "negative row start")?,
        );
        if !inputs.emits_row[item] || inputs.prompt_rows[item] {
            return Err(format!("anchor item {item} must end an unscored prompt"));
        }
        let end = prefix.checked_add(rows).ok_or("anchor endpoint overflow")?;
        let tokens = end / granule * granule;
        if tokens == end || tokens <= prefix {
            return Err(format!(
                "item {item} keeps an anchor but no multiple of {granule} lies strictly \
                 inside its rows from {prefix} to {end}"
            ));
        }
        let spare = found.len() as u32;
        if spare >= spares {
            return Err(format!("a step keeps at most {spares} anchors"));
        }
        found.push(Anchor {
            item,
            spare,
            request: inputs.request_ids[item],
            tokens,
            row: first + (tokens - prefix),
            first,
        });
    }
    Ok(found)
}

/// One rank's copies, as (destination, source), that seed `anchor`'s spare
/// from slot `slot` before the step: every compressor tail, and the rings
/// when the anchor's ring reaches before the item.
pub(crate) fn seed_copies(
    state: &RequestState,
    extent: Extent,
    held: &[Span],
    spares: &Spares,
    slot: u32,
    anchor: &Anchor,
    cells: u32,
) -> Result<Vec<(Span, Span)>, String> {
    let error = |e: bundle_runtime::state::Invalid| format!("{}: {}", e.name, e.rule);
    let from = state.pieces(extent, Holder::Slot(slot)).map_err(error)?;
    let to = state
        .pieces(extent, Holder::Anchor(anchor.spare))
        .map_err(error)?;
    let rings = anchor.reads_committed_ring(cells);
    let mut copies = Vec::new();
    for (from, to) in from.iter().zip(&to) {
        let name = state.buffers[from.buffer].name;
        if from.buffer != to.buffer || from.length != to.length || from.spare || !to.spare {
            return Err("anchor spare pieces differ from the slot's".into());
        }
        if !(name == TAIL || name == RING && rings) {
            continue;
        }
        let source = held.get(from.buffer).ok_or("a captured buffer is absent")?;
        let spare = spares.buffers[to.buffer].ok_or("a captured buffer has no spare")?;
        let span = |base: Span, offset: u64, length: u64| -> Result<Span, String> {
            if offset + length > base.length {
                return Err(format!("anchor spare piece of {name} exceeds its buffer"));
            }
            Ok(Span::new(base.buffer, base.offset + offset, length))
        };
        copies.push((
            span(spare, to.offset, to.length)?,
            span(*source, from.offset, from.length)?,
        ));
    }
    Ok(copies)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(prefix: i32, rows: i32) -> PhaseInputs {
        PhaseInputs {
            request_ids: vec![7],
            anchors: vec![true],
            emits_row: vec![true],
            prompt_rows: vec![false],
            prefix_len: vec![prefix],
            q_len: vec![rows],
            q_start: vec![0],
            ..Default::default()
        }
    }

    #[test]
    fn prompt_end_anchors_admit_any_start_within_the_allowance() {
        for (prefix, rows, tokens) in [(0, 65, 64), (829, 71, 896), (896, 65, 960)] {
            let inputs = input(prefix, rows);
            assert!(anchors(&inputs, 0).is_err());
            let found = anchors(&inputs, 1).unwrap();
            assert_eq!(found.len(), 1);
            assert_eq!(
                (found[0].tokens, found[0].row),
                (tokens, tokens - prefix as u32)
            );
            assert!(found[0].reads_committed_ring(136));
        }
        for (prefix, rows) in [(0, 64), (829, 3), (896, 1)] {
            assert!(anchors(&input(prefix, rows), 1).is_err());
        }
        let mut intermediate = input(829, 71);
        intermediate.emits_row[0] = false;
        assert!(anchors(&intermediate, 1).is_err());
        intermediate.anchors[0] = false;
        assert!(anchors(&intermediate, 0).unwrap().is_empty());
    }
}
