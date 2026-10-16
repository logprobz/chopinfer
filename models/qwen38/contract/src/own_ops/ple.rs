//! The per-layer embedding's own kernels (`cuda/qwen38_bundle.cuh`,
//! `cuda/qwen38_speculation.cuh` and the PLE convolution rows of
//! `cuda/row/qwen38_ple.cuh` in the bundle): the residual, each item's
//! state entry staged for a step, the state after each verify row and at a
//! prefill anchor, and the convolution over the staged history.
//!
//! A PLE state entry is the convolution history, `conv_rows` rows of `wide`
//! BF16 values (oldest first), then a 128-byte tail whose first two `u32`
//! words are the n-gram history (the older id first) and whose other bytes
//! the kernels carry unchanged.

use kernel_core::contract::{ConvActivation, RecurrentPhase, RecurrentQuery, Refusal, Validate};
use kernel_core::oracle::numeric_bound;

use super::{Bounded, ItemForm, bytes_at, bytes_at_mut, copy, entry, sum_bf16, verify_rows};
use crate::contract::Shape;

/// Bytes of a state entry's tail after the convolution history.
const TAIL_BYTES: u64 = 128;

/// One PLE state entry's layout. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleLayout {
    /// Channels of one convolution row: every stream's hidden values.
    pub wide: u32,
    /// Rows of the convolution history: `(taps - 1) * dilation`.
    pub conv_rows: u32,
    /// The end-of-document id a fresh n-gram history holds.
    pub eos: u32,
}

impl PleLayout {
    pub fn of(shape: Shape) -> PleLayout {
        PleLayout {
            wide: shape.wide(),
            conv_rows: shape.ple_conv_rows(),
            eos: shape.ngram_eos(),
        }
    }

    /// Bytes of one convolution row.
    pub fn row_bytes(self) -> u64 {
        u64::from(self.wide) * 2
    }

    /// Bytes of the convolution history.
    pub fn conv_bytes(self) -> u64 {
        u64::from(self.conv_rows) * self.row_bytes()
    }

    /// Bytes of one state entry.
    pub fn entry_bytes(self) -> u64 {
        self.conv_bytes() + TAIL_BYTES
    }

    /// The entry a fresh sequence stages: a zero history and an n-gram
    /// history of two end-of-document ids.
    pub fn fresh(self) -> Vec<u8> {
        let mut entry = vec![0; self.entry_bytes() as usize];
        let at = self.conv_bytes() as usize;
        entry[at..at + 4].copy_from_slice(&self.eos.to_le_bytes());
        entry[at + 4..at + 8].copy_from_slice(&self.eos.to_le_bytes());
        entry
    }
}

impl Validate for PleLayout {
    fn validate(&self) -> Result<(), Refusal> {
        if self.wide == 0 || self.conv_rows == 0 {
            return Err(Refusal("a PLE state entry needs channels and history rows"));
        }
        Ok(())
    }
}

/// `out = x + gated + conv` over every live row of `streams * hidden`
/// BF16 values: the three terms summed in F32 in that order and rounded
/// once to BF16 (ties to even). All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleResidual {
    pub hidden: u32,
    pub streams: u32,
}

impl PleResidual {
    pub fn of(shape: Shape) -> PleResidual {
        PleResidual {
            hidden: shape.hidden(),
            streams: shape.streams(),
        }
    }

    fn wide(self) -> u64 {
        u64::from(self.hidden) * u64::from(self.streams)
    }
}

impl Validate for PleResidual {
    fn validate(&self) -> Result<(), Refusal> {
        if self.hidden == 0 || self.streams == 0 {
            return Err(Refusal("a PLE residual needs streams of hidden values"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A PLE residual's buffers: X, the gated PLE rows, the convolution
    /// rows, the output rows and the device `u32` count of live rows. The
    /// rows start at 2-byte alignment, the count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PleResidualBuffers {
        pub x: value,
        pub gated: value,
        pub conv: value,
        pub out: value,
        pub live_rows: value,
    }
}

/// What the residual kernel computes over its first `rows` rows.
pub fn ple_residual(
    op: &PleResidual,
    x: &[u16],
    gated: &[u16],
    conv: &[u16],
    out: &mut [u16],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let count = (u64::from(rows.min(live_rows)) * op.wide()) as usize;
    if [x.len(), gated.len(), conv.len(), out.len()]
        .iter()
        .any(|&len| len < count)
    {
        return Err(Refusal("a PLE residual buffer is too small"));
    }
    for index in 0..count {
        out[index] = sum_bf16(&[x[index], gated[index], conv[index]]);
    }
    Ok(())
}

/// Stages each live item's state entry: the pool entry of its slot when the
/// item continues a sequence (a positive prefix length), else the fresh
/// entry. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleStage {
    pub layout: PleLayout,
}

impl Validate for PleStage {
    fn validate(&self) -> Result<(), Refusal> {
        self.layout.validate()
    }
}

kernel_core::scratch_record! {
    /// A PLE stage's buffers: the committed pool entries, the staging
    /// entries (one per item), the per-item slots and prefix lengths and
    /// the device `u32` count of live items. The pool and staging entries
    /// are copied in 16-byte vectors and start at 16-byte alignment; the
    /// tables and the count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PleStageBuffers {
        pub pool: value,
        pub staging: value,
        pub slot_ids: value,
        pub prefix_len: value,
        pub live_items: value,
    }
}

/// What the stage kernel writes over its first `items` items.
#[allow(clippy::too_many_arguments)]
pub fn ple_stage(
    op: &PleStage,
    pool: &[u8],
    staging: &mut [u8],
    slot_ids: &[i32],
    prefix_len: &[i32],
    live_items: u32,
    items: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let bytes = op.layout.entry_bytes();
    let fresh = op.layout.fresh();
    for item in 0..items.min(live_items) {
        let to = u64::from(item) * bytes;
        if entry(prefix_len, item as usize, "prefix_len")? > 0 {
            let slot = entry(slot_ids, item as usize, "slot_ids")?;
            let slot = u64::try_from(slot).map_err(|_| Refusal("a staged slot is negative"))?;
            copy(staging, to, pool, slot * bytes, bytes)?;
        } else {
            bytes_at_mut(staging, to, bytes)?.copy_from_slice(&fresh);
        }
    }
    Ok(())
}

/// The state entry after row `row` of an item whose rows start at phase row
/// `start`, from `before`, the item's entry before the step: what the
/// convolution and n-gram kernels leave as their output state for a chunk
/// ending at that row. History row `r` is input row
/// `row + 1 - conv_rows + r`, or row `row + 1 + r` of `before`'s history
/// when that is negative; the n-gram history becomes the ids of rows
/// `row - 1` (or `before`'s newer id at row 0) and `row`; the rest of the
/// tail is `before`'s.
pub fn ple_state_after(
    layout: PleLayout,
    input: &[u16],
    tokens: &[u32],
    before: &[u8],
    start: u64,
    row: u32,
) -> Result<Vec<u8>, Refusal> {
    layout.validate()?;
    let row_bytes = layout.row_bytes();
    let mut after = bytes_at(before, 0, layout.entry_bytes())?.to_vec();
    let wide = u64::from(layout.wide);
    for r in 0..layout.conv_rows {
        let source = i64::from(row) + 1 - i64::from(layout.conv_rows) + i64::from(r);
        let at = (u64::from(r) * row_bytes) as usize;
        let target = &mut after[at..at + row_bytes as usize];
        if source < 0 {
            let from = (source + i64::from(layout.conv_rows)) as u64 * row_bytes;
            target.copy_from_slice(bytes_at(before, from, row_bytes)?);
        } else {
            let first = ((start + source as u64) * wide) as usize;
            let values = input
                .get(first..first + layout.wide as usize)
                .ok_or(Refusal("a PLE input row is outside the input"))?;
            for (pair, value) in target.chunks_exact_mut(2).zip(values) {
                pair.copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    let token = |at: u64| -> Result<u32, Refusal> {
        entry(tokens, at as usize, "a PLE token is outside the tokens")
    };
    let tail = layout.conv_bytes() as usize;
    let older = if row >= 1 {
        token(start + u64::from(row) - 1)?
    } else {
        u32::from_le_bytes(after[tail + 4..tail + 8].try_into().unwrap())
    };
    let newer = token(start + u64::from(row))?;
    after[tail..tail + 4].copy_from_slice(&older.to_le_bytes());
    after[tail + 4..tail + 8].copy_from_slice(&newer.to_le_bytes());
    Ok(after)
}

/// Writes, for each live verify item and each of its rows, the state after
/// that row ([`ple_state_after`]) into the entry its checkpoint id names:
/// id `checkpoint_ids[item * verify_rows + row]`, skipped when negative or
/// past the entries. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleCheckpoints {
    pub layout: PleLayout,
    /// Rows one verify item carries at most.
    pub verify_rows: u32,
}

impl PleCheckpoints {
    pub fn of(shape: Shape) -> PleCheckpoints {
        PleCheckpoints {
            layout: PleLayout::of(shape),
            verify_rows: crate::contract::MAX_VERIFY_ROWS,
        }
    }
}

impl Validate for PleCheckpoints {
    fn validate(&self) -> Result<(), Refusal> {
        self.layout.validate()?;
        if self.verify_rows == 0 {
            return Err(Refusal("PLE checkpoints need verify rows"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// PLE checkpoints' buffers: the convolution input rows the verify rows
    /// read, their token ids, each item's staged entry, the pool of entries
    /// the ids name, each item's first row and row count, the device `u32`
    /// count of live items, and `verify_rows` checkpoint ids per item. The
    /// staged and pool entries are copied in 16-byte vectors and start at
    /// 16-byte alignment, the input rows too; the `u32` token ids, the
    /// tables and the count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PleCheckpointsBuffers {
        pub input: value,
        pub tokens: value,
        pub staged: value,
        pub pool: value,
        pub cu_rows: value,
        pub q_len: value,
        pub live_items: value,
        pub checkpoint_ids: value,
    }
}

/// What the checkpoints kernel writes into `pool` (of `pool.len() /
/// entry_bytes` entries) over its first `items` items. Two rows naming one
/// entry are refused: the kernel's result would depend on block order.
#[allow(clippy::too_many_arguments)]
pub fn ple_checkpoints(
    op: &PleCheckpoints,
    input: &[u16],
    tokens: &[u32],
    staged: &[u8],
    pool: &mut [u8],
    cu_rows: &[i32],
    q_len: &[i32],
    checkpoint_ids: &[i32],
    live_items: u32,
    items: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let bytes = op.layout.entry_bytes();
    let entries = pool.len() as u64 / bytes;
    let mut written = std::collections::BTreeSet::new();
    for item in 0..items.min(live_items) {
        let start = u64::try_from(entry(cu_rows, item as usize, "cu_rows")?)
            .map_err(|_| Refusal("a verify item starts before row 0"))?;
        let before = bytes_at(staged, u64::from(item) * bytes, bytes)?;
        for row in 0..verify_rows(q_len, item, op.verify_rows)? {
            let id = entry(
                checkpoint_ids,
                (item * op.verify_rows + row) as usize,
                "checkpoint_ids",
            )?;
            let Some(id) = u64::try_from(id).ok().filter(|&id| id < entries) else {
                continue;
            };
            if !written.insert(id) {
                return Err(Refusal("two checkpoint rows name one entry"));
            }
            let after = ple_state_after(op.layout, input, tokens, before, start, row)?;
            bytes_at_mut(pool, id * bytes, bytes)?.copy_from_slice(&after);
        }
    }
    Ok(())
}

/// Writes, for each live prefill item with an anchor, the state after its
/// row before position `anchor_ends[item]` ([`ple_state_after`] of row
/// `anchor_ends[item] - prefix_len[item] - 1`) into spare entry
/// `anchor_slots[item]`. Items with a negative anchor slot, or an anchor at
/// or before their first row, write nothing. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleAnchor {
    pub layout: PleLayout,
}

impl Validate for PleAnchor {
    fn validate(&self) -> Result<(), Refusal> {
        self.layout.validate()
    }
}

kernel_core::scratch_record! {
    /// A PLE anchor's buffers: the convolution input rows, their token ids,
    /// each item's staged entry, the spare entries, each item's first row,
    /// prefix length, anchor slot and anchor end, and the device `u32`
    /// count of live items. The entries are copied in 16-byte vectors and
    /// start at 16-byte alignment, the input rows too; the `u32` token ids,
    /// the tables and the count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PleAnchorBuffers {
        pub input: value,
        pub tokens: value,
        pub staged: value,
        pub anchors: value,
        pub cu_rows: value,
        pub prefix_len: value,
        pub anchor_slots: value,
        pub anchor_ends: value,
        pub live_items: value,
    }
}

/// What the anchor kernel writes into `anchors` over its first `items`
/// items. Two items naming one spare entry are refused.
#[allow(clippy::too_many_arguments)]
pub fn ple_anchor(
    op: &PleAnchor,
    input: &[u16],
    tokens: &[u32],
    staged: &[u8],
    anchors: &mut [u8],
    cu_rows: &[i32],
    prefix_len: &[i32],
    anchor_slots: &[i32],
    anchor_ends: &[i32],
    live_items: u32,
    items: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let bytes = op.layout.entry_bytes();
    let mut written = std::collections::BTreeSet::new();
    for item in 0..items.min(live_items) {
        let index = item as usize;
        let slot = entry(anchor_slots, index, "anchor_slots")?;
        let rows = i64::from(entry(anchor_ends, index, "anchor_ends")?)
            - i64::from(entry(prefix_len, index, "prefix_len")?);
        if slot < 0 || rows < 1 {
            continue;
        }
        if !written.insert(slot) {
            return Err(Refusal("two anchored items name one spare entry"));
        }
        let start = u64::try_from(entry(cu_rows, index, "cu_rows")?)
            .map_err(|_| Refusal("a prefill item starts before row 0"))?;
        let row = u32::try_from(rows - 1).map_err(|_| Refusal("an anchor is past the rows"))?;
        let before = bytes_at(staged, u64::from(item) * bytes, bytes)?;
        let after = ple_state_after(op.layout, input, tokens, before, start, row)?;
        bytes_at_mut(anchors, slot as u64 * bytes, bytes)?.copy_from_slice(&after);
    }
    Ok(())
}

/// The PLE convolution of each item over its staged history:
/// `RecurrentQuery::CausalConv1d` over `layout.wide` channels with SiLU,
/// weight `[wide][taps]`, the history the first `conv_bytes` of the item's
/// entry, and the output state (the last `conv_rows` window rows, exact
/// copies) written into the item's output entry. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleConv {
    pub layout: PleLayout,
    pub taps: u32,
    pub dilation: u32,
    pub items: ItemForm,
}

impl PleConv {
    pub fn of(shape: Shape, items: ItemForm) -> PleConv {
        PleConv {
            layout: PleLayout::of(shape),
            taps: shape.ple_conv_taps(),
            dilation: shape.ple_conv_dilation(),
            items,
        }
    }

    /// The convolution of one item over `rows` rows.
    pub fn query(&self, rows: u32) -> RecurrentQuery {
        RecurrentQuery::CausalConv1d {
            channels: self.layout.wide,
            kernel: self.taps,
            dilation: self.dilation,
            activation: ConvActivation::Silu,
            phase: match self.items {
                ItemForm::One => RecurrentPhase::Prefill,
                ItemForm::Decode => RecurrentPhase::Decode,
                ItemForm::Verify { rows } => RecurrentPhase::Verify { max_rows: rows },
            },
            capacity: rows,
        }
    }
}

impl Validate for PleConv {
    fn validate(&self) -> Result<(), Refusal> {
        self.layout.validate()?;
        if self.taps < 2
            || self.dilation == 0
            || (self.taps - 1).checked_mul(self.dilation) != Some(self.layout.conv_rows)
            || matches!(self.items, ItemForm::Verify { rows: 0 })
        {
            return Err(Refusal(
                "a PLE convolution's history must be its taps' reach over the dilation",
            ));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A PLE convolution's buffers: the input rows, the `[wide][taps]`
    /// weight, the staged state entries, the output rows, the output state
    /// entries, and each item's row count. Verify ops add each item's first
    /// row and the device `u32` count of live items, which the other forms
    /// do not take. The staged and output states of one item are one entry
    /// each (the prefill form's views start at its item's). Alignment by
    /// role, which the prefill form's offsets keep: the BF16 input, weight,
    /// states and output at 2 bytes; the row counts, first rows and count
    /// at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PleConvBuffers {
        pub input: value,
        pub weight: value,
        pub state: value,
        pub output: value,
        pub state_out: value,
        pub cu_rows: option,
        pub q_len: value,
        pub live_items: option,
    }
}

/// One item's convolution: the output rows' reference and the output
/// state's exact BF16 bits.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvItem {
    /// `[rows][wide]`, each bound half a BF16 ulp plus the accumulation
    /// gate (`kernel_core::oracle::numeric_bound`).
    pub output: Bounded,
    /// `[conv_rows][wide]`.
    pub state: Vec<u16>,
}

/// One item's convolution of `rows` input rows (`[rows][wide]`) after
/// `state` (`[conv_rows][wide]`).
pub fn ple_conv_item(
    op: &PleConv,
    input: &[u16],
    weight: &[u16],
    state: &[u16],
    rows: u32,
) -> Result<ConvItem, Refusal> {
    op.validate()?;
    let wide = op.layout.wide as usize;
    let history = op.layout.conv_rows as usize * wide;
    let input = input
        .get(..rows as usize * wide)
        .ok_or(Refusal("a PLE convolution input is too small"))?;
    let state = state
        .get(..history)
        .ok_or(Refusal("a PLE convolution state is too small"))?;
    let widen = |bits: &[u16]| {
        bits.iter()
            .map(|&b| f64::from(kernel_core::codec::bf16_to_f32(b)))
            .collect::<Vec<_>>()
    };
    let output = kernel_core::recurrent_oracle::causal_conv1d(
        op.query(rows),
        &widen(input),
        &widen(weight),
        &widen(state),
    )?;
    let bounds = (0..output.output.values.len())
        .map(|index| numeric_bound(&output.output, index, kernel_core::contract::DType::Bf16))
        .collect();
    let window = [state, input].concat();
    Ok(ConvItem {
        output: Bounded {
            values: output.output.values,
            bounds,
        },
        state: window[window.len() - history..].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layout of 8 channels, a 3-row history and EOS 9.
    const LAYOUT: PleLayout = PleLayout {
        wide: 8,
        conv_rows: 3,
        eos: 9,
    };

    fn noise(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed))
            .collect()
    }

    #[test]
    fn the_serving_layout_is_the_bundles_entry() {
        for shape in [Shape::TP2, Shape::TP4] {
            let layout = PleLayout::of(shape);
            assert_eq!(layout.entry_bytes(), shape.ple_entry_bytes());
            assert_eq!(layout.conv_bytes(), shape.ple_conv_bytes());
            assert!(PleConv::of(shape, ItemForm::One).validate().is_ok());
        }
        assert_eq!(PleLayout::of(Shape::TP4).entry_bytes(), 184_448);
    }

    #[test]
    fn residual_rows_round_the_f32_sum_once() {
        let op = PleResidual {
            hidden: 2,
            streams: 2,
        };
        let bf = kernel_core::codec::f32_to_bf16;
        let x = vec![bf(1.0); 8];
        let gated = vec![bf(2_f32.powi(-9)); 8];
        let conv = vec![bf(2_f32.powi(-9)); 8];
        let mut out = vec![0xaaaa; 8];
        ple_residual(&op, &x, &gated, &conv, &mut out, 1, 2).unwrap();
        assert_eq!(out[..4], [bf(1.0 + 2_f32.powi(-8)); 4]);
        assert_eq!(out[4..], [0xaaaa; 4]);
        assert!(ple_residual(&op, &x, &gated, &conv[..4], &mut out, 2, 2).is_err());
    }

    #[test]
    fn staging_copies_continuing_items_and_freshens_new_ones() {
        let op = PleStage { layout: LAYOUT };
        let bytes = LAYOUT.entry_bytes() as usize;
        let pool = noise(3 * bytes, 1);
        let mut staging = vec![0xaa; 3 * bytes];
        ple_stage(&op, &pool, &mut staging, &[2, 0, 1], &[4, 0, 1], 2, 3).unwrap();
        assert_eq!(staging[..bytes], pool[2 * bytes..]);
        assert_eq!(staging[bytes..2 * bytes], LAYOUT.fresh());
        assert_eq!(staging[2 * bytes..], vec![0xaa; bytes]);
        let fresh = LAYOUT.fresh();
        assert_eq!(fresh[48..56], [9, 0, 0, 0, 9, 0, 0, 0]);
        assert!(fresh[56..].iter().all(|&b| b == 0));
        assert!(ple_stage(&op, &pool, &mut staging, &[3], &[1], 1, 1).is_err());
    }

    /// The state after row `row`: older history rows, then input rows; the
    /// n-gram ids of the last two rows.
    #[test]
    fn the_state_after_a_row_slides_the_window_and_the_ids() {
        let before = noise(LAYOUT.entry_bytes() as usize, 3);
        let input = (0..4 * 8).map(|v| v as u16 + 100).collect::<Vec<_>>();
        let tokens = [70, 71, 72, 73];
        let history_row = |r: usize| before[r * 16..(r + 1) * 16].to_vec();
        let input_row = |r: usize| {
            input[r * 8..(r + 1) * 8]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        };
        // Row 0 of an item starting at input row 1.
        let after = ple_state_after(LAYOUT, &input, &tokens, &before, 1, 0).unwrap();
        assert_eq!(after[..16], history_row(1)[..]);
        assert_eq!(after[16..32], history_row(2)[..]);
        assert_eq!(after[32..48], input_row(1)[..]);
        assert_eq!(after[48..52], before[52..56]);
        assert_eq!(after[52..56], 71_u32.to_le_bytes());
        assert_eq!(after[56..], before[56..]);
        // Row 2: the window is the item's three rows.
        let after = ple_state_after(LAYOUT, &input, &tokens, &before, 1, 2).unwrap();
        assert_eq!(
            after[..48],
            [input_row(1), input_row(2), input_row(3)].concat()[..]
        );
        assert_eq!(
            after[48..56],
            [72_u32.to_le_bytes(), 73_u32.to_le_bytes()].concat()[..]
        );
        assert!(ple_state_after(LAYOUT, &input, &tokens, &before, 1, 3).is_err());
    }

    #[test]
    fn checkpoints_land_in_the_named_entries() {
        let op = PleCheckpoints {
            layout: LAYOUT,
            verify_rows: 2,
        };
        let bytes = LAYOUT.entry_bytes() as usize;
        let staged = noise(2 * bytes, 5);
        let input = (0..3 * 8).map(|v| v as u16).collect::<Vec<_>>();
        let tokens = [1, 2, 3];
        let mut pool = vec![0xaa; 4 * bytes];
        // Item 0: rows 0..2 into entries 3 and (skipped) -1; item 1: row 2
        // into entry 0, its second id past the pool.
        ple_checkpoints(
            &op,
            &input,
            &tokens,
            &staged,
            &mut pool,
            &[0, 2],
            &[2, 1],
            &[3, -1, 0, 9],
            2,
            2,
        )
        .unwrap();
        let state = |item: usize, start, row| {
            ple_state_after(LAYOUT, &input, &tokens, &staged[item * bytes..], start, row).unwrap()
        };
        assert_eq!(pool[3 * bytes..], state(0, 0, 0)[..]);
        assert_eq!(pool[..bytes], state(1, 2, 0)[..]);
        assert_eq!(pool[bytes..3 * bytes], vec![0xaa; 2 * bytes][..]);
        // Two rows naming one entry, and more rows than the op serves.
        let mut again = pool.clone();
        let args = |ids: &'static [i32], q_len: &'static [i32]| (ids, q_len);
        for (ids, q_len) in [args(&[1, 1, 2, 2], &[2, 0]), args(&[1, 2, 0, 0], &[3, 0])] {
            assert!(
                ple_checkpoints(
                    &op,
                    &input,
                    &tokens,
                    &staged,
                    &mut again,
                    &[0, 2],
                    q_len,
                    ids,
                    2,
                    2
                )
                .is_err()
            );
        }
    }

    #[test]
    fn anchors_keep_the_state_before_the_anchor() {
        let op = PleAnchor { layout: LAYOUT };
        let bytes = LAYOUT.entry_bytes() as usize;
        let staged = noise(3 * bytes, 7);
        let input = (0..6 * 8).map(|v| v as u16).collect::<Vec<_>>();
        let tokens = [1, 2, 3, 4, 5, 6];
        let mut anchors = vec![0xaa; 2 * bytes];
        // Item 0 (rows 0..4, prefix 10) anchors at 13 into spare 1; item 1
        // has no anchor; item 2 anchors at its first row (nothing).
        ple_anchor(
            &op,
            &input,
            &tokens,
            &staged,
            &mut anchors,
            &[0, 4, 5],
            &[10, 0, 7],
            &[1, -1, 0],
            &[13, 3, 7],
            3,
            3,
        )
        .unwrap();
        assert_eq!(
            anchors[bytes..],
            ple_state_after(LAYOUT, &input, &tokens, &staged, 0, 2).unwrap()[..]
        );
        assert_eq!(anchors[..bytes], vec![0xaa; bytes][..]);
    }

    #[test]
    fn convolution_states_are_the_window_tail_and_outputs_the_f64_conv() {
        let op = PleConv {
            layout: LAYOUT,
            taps: 4,
            dilation: 1,
            items: ItemForm::One,
        };
        let bf = kernel_core::codec::f32_to_bf16;
        let input = (0..2 * 8).map(|v| bf(v as f32 / 8.0)).collect::<Vec<_>>();
        let state = (0..3 * 8)
            .map(|v| bf(-(v as f32) / 16.0))
            .collect::<Vec<_>>();
        let weight = (0..8 * 4)
            .map(|v| bf((v % 5) as f32 / 4.0 - 0.5))
            .collect::<Vec<_>>();
        let item = ple_conv_item(&op, &input, &weight, &state, 2).unwrap();
        assert_eq!(item.state, [&state[16..], &input[..]].concat());
        // Channel 3 of row 1: taps over state row 2 and input rows 0, 1 (and
        // state row 1), SiLU.
        let x = |v: u16| f64::from(kernel_core::codec::bf16_to_f32(v));
        let w = |tap: usize| x(weight[3 * 4 + tap]);
        let sum = w(0) * x(state[8 + 3])
            + w(1) * x(state[16 + 3])
            + w(2) * x(input[3])
            + w(3) * x(input[8 + 3]);
        let silu = sum / (1.0 + (-sum).exp());
        assert!((item.output.values[8 + 3] - silu).abs() < 1e-12);
        // The kernel's F32 arithmetic is within the bounds.
        let emulated = (0..2 * 8)
            .map(|index| {
                let (row, channel) = (index / 8, index % 8);
                let window = [&state[..], &input[..]].concat();
                let mut sum = 0_f32;
                for tap in 0..4 {
                    let value =
                        kernel_core::codec::bf16_to_f32(window[(3 + row - 3 + tap) * 8 + channel]);
                    sum = kernel_core::codec::bf16_to_f32(weight[channel * 4 + tap])
                        .mul_add(value, sum);
                }
                bf(sum * (1.0 / (1.0 + (-sum).exp())))
            })
            .collect::<Vec<_>>();
        item.output.check_bf16(&emulated).unwrap();
    }
}
