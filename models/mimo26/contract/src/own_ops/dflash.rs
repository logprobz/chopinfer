//! The DFlash drafter's own row kernels (`cuda/mimo_dflash.cuh` in the
//! bundle). Row `r` of a phase belongs to item `token_batch[r]` at position
//! `positions[r]`; an item's first row sits at `prefix_len[item]`, and a row
//! past it is a draft row.

use kernel_core::contract::{Refusal, Validate};

use super::{bytes_at, copy, entry, scale_bf16};
use crate::Tp;

/// BF16 values of `bytes`.
fn bf16s(bytes: &[u8]) -> impl Iterator<Item = u16> + '_ {
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
}

/// Writes `values` as BF16 at `target[at..]`.
fn write_bf16s(
    target: &mut [u8],
    at: u64,
    values: impl Iterator<Item = u16>,
) -> Result<(), Refusal> {
    let mut at = at as usize;
    for value in values {
        target
            .get_mut(at..at + 2)
            .ok_or(Refusal("an oracle buffer is too small"))?
            .copy_from_slice(&value.to_le_bytes());
        at += 2;
    }
    Ok(())
}

/// Copies `width` BF16 columns of each live row into the output (row stride
/// `output_stride`), scaled by `scale_bits` unless it is exactly one: each
/// value times the scale in F32, rounded to the nearest BF16 (ties to even,
/// NaN canonical). With no input the rows are zeroed. All math: the drafter
/// scales V by its value scale here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CopyRows {
    pub width: u32,
    /// The input row stride, `None` for no input (zeros).
    pub input_stride: Option<u32>,
    pub output_stride: u32,
    pub scale_bits: u32,
}

impl Validate for CopyRows {
    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0
            || self.width > self.output_stride
            || self.input_stride.is_some_and(|stride| self.width > stride)
        {
            return Err(Refusal("a row copy needs a width within its rows"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A row copy's buffers: the input rows (none when the op has no input),
    /// the output rows and the device count of live rows. The input and
    /// output start at 16-byte alignment (16-byte vectors); the count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct CopyRowsBuffers {
        pub input: option,
        pub output: value,
        pub live_rows: value,
    }
}

/// What the row copy kernel computes over its first `rows` rows.
pub fn copy_rows(
    op: &CopyRows,
    input: Option<&[u8]>,
    output: &mut [u8],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    if input.is_some() != op.input_stride.is_some() {
        return Err(Refusal("a row copy's input does not match its op"));
    }
    let scale = f32::from_bits(op.scale_bits);
    let bytes = u64::from(op.width) * 2;
    for row in 0..u64::from(rows.min(live_rows)) {
        let to = row * u64::from(op.output_stride) * 2;
        match (input, op.input_stride) {
            (Some(input), Some(stride)) => {
                let source = bytes_at(input, row * u64::from(stride) * 2, bytes)?;
                if scale == 1.0 {
                    copy(output, to, source, 0, bytes)?;
                } else {
                    let scaled = bf16s(source)
                        .map(|value| scale_bf16(value, scale))
                        .collect::<Vec<_>>();
                    write_bf16s(output, to, scaled.into_iter())?;
                }
            }
            _ => write_bf16s(output, to, std::iter::repeat_n(0, op.width as usize))?,
        }
    }
    Ok(())
}

/// Whether row `row` is a draft row: not at its item's prefix.
fn draft_row(
    row: usize,
    positions: &[i32],
    prefix_len: &[i32],
    token_batch: &[i32],
) -> Result<bool, Refusal> {
    let item = entry(token_batch, row, "token_batch")? as usize;
    Ok(entry(positions, row, "positions")? != entry(prefix_len, item, "prefix_len")?)
}

/// Writes the `hidden`-wide mask embedding row into every live draft row of
/// the activation; a block's first row (its bonus token's embedding) keeps
/// its bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaskRows {
    pub hidden: u32,
}

impl Validate for MaskRows {
    fn validate(&self) -> Result<(), Refusal> {
        if self.hidden == 0 {
            return Err(Refusal("mask rows need a width"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// Mask rows' buffers: the activation rows, the mask embedding row,
    /// each row's position and item, the per-item prefix lengths and the
    /// device count of live rows. The activation and mask start at 16-byte
    /// alignment (16-byte vectors); the tables and count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct MaskRowsBuffers {
        pub activation: value,
        pub mask: value,
        pub positions: value,
        pub prefix_len: value,
        pub token_batch: value,
        pub live_rows: value,
    }
}

/// What the mask rows kernel computes over its first `rows` rows.
#[allow(clippy::too_many_arguments)]
pub fn mask_rows(
    op: &MaskRows,
    activation: &mut [u8],
    mask: &[u8],
    positions: &[i32],
    prefix_len: &[i32],
    token_batch: &[i32],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let bytes = u64::from(op.hidden) * 2;
    for row in 0..rows.min(live_rows) as usize {
        if draft_row(row, positions, prefix_len, token_batch)? {
            copy(activation, row as u64 * bytes, mask, 0, bytes)?;
        }
    }
    Ok(())
}

/// Writes one layer's context K and V rows into the per-slot circular cache
/// `[slot][position % tokens][head][head_dim]` (BF16), V times `v_scale`
/// (each value in F32, rounded to the nearest BF16, ties to even, NaN
/// canonical; even a scale of one rounds). A row at a negative position, or
/// `tokens` or more behind its item's last position, is skipped: a later row
/// of the phase overwrites its cell. A slot outside the cache is an input
/// error the kernel traps on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheWrite {
    /// The rank's drafter KV heads.
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Positions per slot.
    pub tokens: u32,
    /// Slots: one per item of the instance's largest bucket.
    pub slots: u32,
    pub k_stride: u32,
    pub v_stride: u32,
    /// The drafter's value scale.
    pub v_scale_bits: u32,
}

impl CacheWrite {
    /// The served drafter's cache of `slots` slots at `tp`, K and V read
    /// with row strides `k_stride` and `v_stride`.
    pub fn of(tp: Tp, slots: u32, k_stride: u32, v_stride: u32) -> CacheWrite {
        let dflash = tp.geometry.dflash();
        CacheWrite {
            kv_heads: tp.dflash_kv_heads() as u32,
            head_dim: dflash.head_dim,
            tokens: crate::state::DFLASH_CACHE_TOKENS as u32,
            slots,
            k_stride,
            v_stride,
            v_scale_bits: (dflash.dflash_config.attention_value_scale as f32).to_bits(),
        }
    }

    /// BF16 values of one cache entry: every head of one position.
    pub fn entry(&self) -> u32 {
        self.kv_heads * self.head_dim
    }

    /// Bytes of one cache plane (K or V).
    pub fn plane_bytes(&self) -> u64 {
        u64::from(self.slots) * u64::from(self.tokens) * u64::from(self.entry()) * 2
    }
}

impl Validate for CacheWrite {
    fn validate(&self) -> Result<(), Refusal> {
        let entry = self.entry();
        if entry == 0
            || self.tokens == 0
            || self.slots == 0
            || entry > self.k_stride
            || entry > self.v_stride
        {
            return Err(Refusal("a cache write needs entries within its rows"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A cache write's buffers: the K and V rows, the K and V cache planes,
    /// the per-item slots, each row's item and position, the per-item last
    /// positions (`i64`) and the device count of live rows. K, V and both
    /// planes start at 16-byte alignment (16-byte vectors), the last
    /// positions at 8 bytes, the other tables and the count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct CacheWriteBuffers {
        pub k: value,
        pub v: value,
        pub k_cache: value,
        pub v_cache: value,
        pub slot_ids: value,
        pub token_batch: value,
        pub positions: value,
        pub last_positions: value,
        pub live_rows: value,
    }
}

/// What the cache write kernel computes over its first `rows` rows. Two
/// kept rows writing one cell are refused, as is a kept row's slot outside
/// the cache.
#[allow(clippy::too_many_arguments)]
pub fn cache_write(
    op: &CacheWrite,
    k: &[u8],
    v: &[u8],
    k_cache: &mut [u8],
    v_cache: &mut [u8],
    slot_ids: &[i32],
    token_batch: &[i32],
    positions: &[i32],
    last_positions: &[i64],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let scale = f32::from_bits(op.v_scale_bits);
    let bytes = u64::from(op.entry()) * 2;
    let mut written = std::collections::BTreeSet::new();
    for row in 0..rows.min(live_rows) as usize {
        let item = entry(token_batch, row, "token_batch")? as usize;
        let position = i64::from(entry(positions, row, "positions")?);
        if position < 0
            || position + i64::from(op.tokens) <= entry(last_positions, item, "last_positions")?
        {
            continue;
        }
        let slot = entry(slot_ids, item, "slot_ids")? as u32;
        if slot >= op.slots {
            return Err(Refusal("a cache write slot is outside the cache"));
        }
        let cell = u64::from(slot) * u64::from(op.tokens) + position as u64 % u64::from(op.tokens);
        if !written.insert(cell) {
            return Err(Refusal("two cache write rows write one cell"));
        }
        let to = cell * bytes;
        copy(
            k_cache,
            to,
            k,
            row as u64 * u64::from(op.k_stride) * 2,
            bytes,
        )?;
        let source = bytes_at(v, row as u64 * u64::from(op.v_stride) * 2, bytes)?;
        let scaled = bf16s(source)
            .map(|value| scale_bf16(value, scale))
            .collect::<Vec<_>>();
        write_bf16s(v_cache, to, scaled.into_iter())?;
    }
    Ok(())
}

/// Takes each live draft row's best (score, id) pair over the ranks and
/// stores the id as the row's input token. Every rank's gathered pairs are
/// `[rank][2 * rows]` F32 words: `rows` scores, then `rows` ids as their
/// bits. The best is the largest score, the lowest id (as `i32`) on ties; a
/// NaN score never wins, and a row with no winner stores `i32::MAX`. Rows at
/// their item's prefix keep their bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Drafts {
    pub ranks: u32,
}

impl Validate for Drafts {
    fn validate(&self) -> Result<(), Refusal> {
        if self.ranks == 0 {
            return Err(Refusal("drafts need ranks"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// Drafts' buffers: every rank's gathered pairs, the input tokens it
    /// writes, each row's position and item, the per-item prefix lengths and
    /// the device count of live rows. Every view starts at 4-byte alignment
    /// (the kernel reads single words).
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct DraftsBuffers {
        pub gathered: value,
        pub token_ids: value,
        pub positions: value,
        pub prefix_len: value,
        pub token_batch: value,
        pub live_rows: value,
    }
}

/// What the drafts kernel computes over `rows` rows of pairs.
#[allow(clippy::too_many_arguments)]
pub fn drafts(
    op: &Drafts,
    gathered: &[f32],
    token_ids: &mut [i32],
    positions: &[i32],
    prefix_len: &[i32],
    token_batch: &[i32],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let rows = rows as usize;
    for row in 0..rows.min(live_rows as usize) {
        if !draft_row(row, positions, prefix_len, token_batch)? {
            continue;
        }
        let (mut best, mut best_id) = (f32::NEG_INFINITY, i32::MAX);
        for rank in 0..op.ranks as usize {
            let pairs = rank * 2 * rows;
            let score = entry(gathered, pairs + row, "gathered")?;
            let id = entry(gathered, pairs + rows + row, "gathered")?.to_bits() as i32;
            if score > best || (score == best && id < best_id) {
                (best, best_id) = (score, id);
            }
        }
        *token_ids
            .get_mut(row)
            .ok_or(Refusal("an oracle buffer is too small"))? = best_id;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn row_copies_scale_unless_the_scale_is_one() {
        let op = CopyRows {
            width: 2,
            input_stride: Some(3),
            output_stride: 2,
            scale_bits: 1.0_f32.to_bits(),
        };
        let input = bytes(&[0x7fc1, 0x3f80, 1, 0x4000, 0x3f81, 2]);
        let mut output = vec![0xaa; 3 * 4];
        copy_rows(&op, Some(&input), &mut output, 2, 3).unwrap();
        // NaN payloads survive an unscaled copy; row 2 is not live.
        assert_eq!(output[..8], bytes(&[0x7fc1, 0x3f80, 0x4000, 0x3f81]));
        assert_eq!(output[8..], [0xaa; 4]);
        let scaled = CopyRows {
            scale_bits: 0.5_f32.to_bits(),
            ..op
        };
        copy_rows(&scaled, Some(&input), &mut output, 2, 2).unwrap();
        assert_eq!(output[..8], bytes(&[0x7fff, 0x3f00, 0x3f80, 0x3f01]));
        let zeros = CopyRows {
            input_stride: None,
            ..op
        };
        copy_rows(&zeros, None, &mut output, 3, 3).unwrap();
        assert_eq!(output, [0; 12]);
        assert!(copy_rows(&zeros, Some(&input), &mut output, 3, 3).is_err());
    }

    #[test]
    fn mask_rows_replace_draft_rows_only() {
        let mut activation = vec![0_u8; 4 * 4];
        // One item of three rows from position 5, then an item at its prefix.
        mask_rows(
            &MaskRows { hidden: 2 },
            &mut activation,
            &[1, 2, 3, 4],
            &[5, 6, 7, 9],
            &[5, 9],
            &[0, 0, 0, 1],
            4,
            4,
        )
        .unwrap();
        assert_eq!(activation, [0, 0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0, 0]);
    }

    #[test]
    fn cache_writes_skip_superseded_rows_and_scale_v() {
        let op = CacheWrite {
            kv_heads: 1,
            head_dim: 2,
            tokens: 4,
            slots: 2,
            k_stride: 2,
            v_stride: 2,
            v_scale_bits: 0.5_f32.to_bits(),
        };
        let k = bytes(&[1, 2, 3, 4, 5, 6]);
        let v = bytes(&[0x3f80, 0x4000, 0x3f80, 0x4000, 0x7f80, 0xff80]);
        let mut k_cache = vec![0_u8; op.plane_bytes() as usize];
        let mut v_cache = k_cache.clone();
        // Row 0 (position 1) is 4 behind item 0's last position 5 and is
        // superseded by row 1; row 2 is item 1 (slot 0) at position 2.
        cache_write(
            &op,
            &k,
            &v,
            &mut k_cache,
            &mut v_cache,
            &[1, 0],
            &[0, 0, 1],
            &[1, 5, 2],
            &[5, 2],
            3,
            3,
        )
        .unwrap();
        let mut expected_k = vec![0_u8; k_cache.len()];
        let mut expected_v = expected_k.clone();
        // Item 0's position 5 is slot 1's cell 5; item 1's position 2 cell 2.
        expected_k[5 * 4..][..4].copy_from_slice(&bytes(&[3, 4]));
        expected_v[5 * 4..][..4].copy_from_slice(&bytes(&[0x3f00, 0x3f80]));
        expected_k[2 * 4..][..4].copy_from_slice(&bytes(&[5, 6]));
        expected_v[2 * 4..][..4].copy_from_slice(&bytes(&[0x7f80, 0xff80]));
        assert_eq!(k_cache, expected_k);
        assert_eq!(v_cache, expected_v);
        assert!(
            cache_write(
                &op,
                &k,
                &v,
                &mut k_cache,
                &mut v_cache,
                &[2],
                &[0],
                &[0],
                &[0],
                1,
                1
            )
            .is_err()
        );
    }

    #[test]
    fn drafts_take_the_best_score_and_the_lowest_id_on_ties() {
        let id = |value: i32| f32::from_bits(value as u32);
        // Two ranks, two rows: row 0 is its item's prefix, row 1 ties.
        let gathered = [9.0, 1.0, id(5), id(7), 8.0, 1.0, id(4), id(3)];
        let mut tokens = [-1, -1];
        drafts(
            &Drafts { ranks: 2 },
            &gathered,
            &mut tokens,
            &[0, 1],
            &[0],
            &[0, 0],
            2,
            2,
        )
        .unwrap();
        assert_eq!(tokens, [-1, 3]);
        // NaN never wins; no winner stores i32::MAX.
        let nan = [f32::NAN, f32::NAN, id(1), id(2)];
        let mut tokens = [0, 0];
        drafts(
            &Drafts { ranks: 1 },
            &nan,
            &mut tokens,
            &[1, 2],
            &[0],
            &[0, 0],
            2,
            2,
        )
        .unwrap();
        assert_eq!(tokens, [i32::MAX, i32::MAX]);
    }
}
