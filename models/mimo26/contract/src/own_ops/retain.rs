//! Retained target rows for lazily caught-up drafters
//! (`cuda/mimo_retain.cuh` in the bundle).
//!
//! Each slot owns a circular history of `capacity` rows: sequence position
//! `p` of slot `s` is history row `s * capacity + p % capacity`. Decode and
//! verify phases write every row into its cell; a catch-up gathers committed
//! rows back. Both copy BF16 columns without reading them.

use kernel_core::contract::{Refusal, Validate};

use super::{copy, entry};

/// A history: `slots` slots of `capacity` rows, `stride` BF16 values apart.
/// Its view starts at the column the op writes or reads in slot 0's first
/// row. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct History {
    pub stride: u32,
    /// Rows per slot: the drafter's window plus the retained slack rows.
    pub capacity: u32,
    pub slots: u32,
}

impl History {
    /// The history row of `slot`'s position `position`, `None` for a row the
    /// kernels skip: a slot outside the history or a negative position.
    fn cell(self, slot: i32, position: i64) -> Option<u64> {
        (slot >= 0 && (slot as u32) < self.slots && position >= 0).then(|| {
            u64::from(slot as u32) * u64::from(self.capacity)
                + position as u64 % u64::from(self.capacity)
        })
    }

    /// Bytes from the view's start through the last cell's `width` columns.
    pub fn bytes(self, width: u32) -> u64 {
        ((u64::from(self.slots) * u64::from(self.capacity) - 1) * u64::from(self.stride)
            + u64::from(width))
            * 2
    }

    fn validate(self, width: u32) -> Result<(), Refusal> {
        if self.slots == 0
            || self.capacity <= crate::resources::RETAINED_SLACK_ROWS as u32
            || width > self.stride
        {
            return Err(Refusal(
                "a retained history needs slots, more rows than its slack and room for the width",
            ));
        }
        Ok(())
    }
}

/// Bytes from a row view's start through row `rows - 1`'s `width` columns.
pub fn row_bytes(rows: u32, stride: u32, width: u32) -> u64 {
    (u64::from(rows.max(1) - 1) * u64::from(stride) + u64::from(width)) * 2
}

/// Copies `width` BF16 columns of each live row (row stride
/// `input_stride`) into its history cell: row `r` belongs to item
/// `token_batch[r]` at position `prefix_len[item] + r - q_start[item]`.
/// Rows of an item whose slot is outside the history, and rows at negative
/// positions, are skipped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetainRows {
    pub width: u32,
    pub input_stride: u32,
    pub history: History,
}

impl Validate for RetainRows {
    fn validate(&self) -> Result<(), Refusal> {
        self.history.validate(self.width)?;
        if self.width == 0 || self.width > self.input_stride {
            return Err(Refusal("retained rows need a width within the input rows"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// Retained rows' buffers: the input rows, the history (from its first
    /// written column), the per-item slots, first rows and prefix lengths,
    /// each row's item, and the device count of live rows. The input and
    /// history are copied in 16-byte vectors and start at 16-byte
    /// alignment; the `i32` tables and the `u32` count at 4 bytes.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RetainRowsBuffers {
        pub input: value,
        pub history: value,
        pub slot_ids: value,
        pub token_batch: value,
        pub q_start: value,
        pub prefix_len: value,
        pub live_rows: value,
    }
}

/// What the retain kernel computes over its first `rows` rows. Two live
/// rows writing one cell are refused: the kernel's result would depend on
/// block order.
#[allow(clippy::too_many_arguments)]
pub fn retain_rows(
    op: &RetainRows,
    input: &[u8],
    history: &mut [u8],
    slot_ids: &[i32],
    token_batch: &[i32],
    q_start: &[i32],
    prefix_len: &[i32],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let mut written = std::collections::BTreeSet::new();
    for row in 0..rows.min(live_rows) {
        let item = entry(token_batch, row as usize, "token_batch")? as usize;
        let slot = entry(slot_ids, item, "slot_ids")?;
        let position = i64::from(entry(prefix_len, item, "prefix_len")?) + i64::from(row)
            - i64::from(entry(q_start, item, "q_start")?);
        let Some(cell) = op.history.cell(slot, position) else {
            continue;
        };
        if !written.insert(cell) {
            return Err(Refusal("two retained rows write one history cell"));
        }
        copy(
            history,
            cell * u64::from(op.history.stride) * 2,
            input,
            u64::from(row) * u64::from(op.input_stride) * 2,
            u64::from(op.width) * 2,
        )?;
    }
    Ok(())
}

/// Copies `width` BF16 columns of each live row's history cell into output
/// row `r` (row stride `output_stride`): row `r` is item `token_batch[r]`'s
/// position `positions[r]`. Rows of an item whose slot is outside the
/// history, and negative positions, are skipped and keep their bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatherRows {
    pub width: u32,
    pub output_stride: u32,
    pub history: History,
}

impl Validate for GatherRows {
    fn validate(&self) -> Result<(), Refusal> {
        self.history.validate(self.width)?;
        if self.width == 0 || self.width > self.output_stride {
            return Err(Refusal("gathered rows need a width within the output rows"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// Gathered rows' buffers: the history (from its first read column), the
    /// output rows, the per-item slots, each row's item and position, and
    /// the device count of live rows. The history and output start at
    /// 16-byte alignment (16-byte vectors); the tables and count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct GatherRowsBuffers {
        pub history: value,
        pub output: value,
        pub slot_ids: value,
        pub token_batch: value,
        pub positions: value,
        pub live_rows: value,
    }
}

/// What the gather kernel computes over its first `rows` rows.
#[allow(clippy::too_many_arguments)]
pub fn gather_rows(
    op: &GatherRows,
    history: &[u8],
    output: &mut [u8],
    slot_ids: &[i32],
    token_batch: &[i32],
    positions: &[i32],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    for row in 0..rows.min(live_rows) {
        let item = entry(token_batch, row as usize, "token_batch")? as usize;
        let slot = entry(slot_ids, item, "slot_ids")?;
        let position = entry(positions, row as usize, "positions")?;
        let Some(cell) = op.history.cell(slot, i64::from(position)) else {
            continue;
        };
        copy(
            output,
            u64::from(row) * u64::from(op.output_stride) * 2,
            history,
            cell * u64::from(op.history.stride) * 2,
            u64::from(op.width) * 2,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HISTORY: History = History {
        stride: 3,
        capacity: 10,
        slots: 2,
    };

    #[test]
    fn retained_rows_land_in_their_cells_and_skip_invalid_ones() {
        let op = RetainRows {
            width: 2,
            input_stride: 4,
            history: HISTORY,
        };
        let input = (0..4 * 4 * 2).map(|byte| byte as u8).collect::<Vec<_>>();
        let mut history = vec![0xaa_u8; HISTORY.bytes(2) as usize];
        // Item 0 (slot 1) rows 0 and 1 at positions 19, 20; item 1 has slot
        // -1; row 3 is not live.
        retain_rows(
            &op,
            &input,
            &mut history,
            &[1, -1],
            &[0, 0, 1, 1],
            &[0, 2],
            &[19, 0],
            3,
            4,
        )
        .unwrap();
        let mut expected = vec![0xaa_u8; history.len()];
        // Position 19 is cell 19, position 20 cell 10.
        expected[19 * 6..][..4].copy_from_slice(&input[0..4]);
        expected[10 * 6..][..4].copy_from_slice(&input[8..12]);
        assert_eq!(history, expected);
        // Positions 0 and 10 of one slot share a cell.
        assert!(
            retain_rows(
                &op,
                &input,
                &mut history,
                &[0, 0],
                &[0, 1],
                &[0, 1],
                &[0, 10],
                2,
                2
            )
            .is_err()
        );
    }

    #[test]
    fn gathered_rows_read_their_cells() {
        let op = GatherRows {
            width: 2,
            output_stride: 2,
            history: HISTORY,
        };
        let history = (0..HISTORY.bytes(2))
            .map(|byte| byte as u8)
            .collect::<Vec<_>>();
        let mut output = vec![0xaa_u8; 3 * 4];
        gather_rows(
            &op,
            &history,
            &mut output,
            &[1, 0],
            &[0, 1, 1],
            &[13, -1, 4],
            3,
            3,
        )
        .unwrap();
        assert_eq!(output[0..4], history[13 * 6..][..4]);
        assert_eq!(output[4..8], [0xaa; 4]);
        assert_eq!(output[8..12], history[4 * 6..][..4]);
    }
}
