//! Bitwise gathering of the retained window prefix before a prefill write.
use crate::contract::{NumericalContract, NumericalContractId, Refusal, Validate};
use crate::latent_attention::LatentWindowBuffers;

/// Gather positions `[max(0, first - window + 1), first)` for each item.
/// The item maps to a ring state slot; `first` is its first new token and
/// `last` is that state slot's last committed ring position. Each position
/// is copied from `position % ring_slots` to `position - start` in the
/// item's staged block, preserving every code and scale bit. The chunk is
/// subsequently written at `first - start`, with `start` supplied to the
/// staged writer and key-list builder as its first absolute position.
///
/// Codes and UE8M0 scales have independent sequence strides. No decoding,
/// rounding or quantization occurs. Invalid item mappings, negative first
/// or last positions, and source positions outside the retained interval
/// `[max(0, last + 1 - ring_slots), last]` are holes: neither destination
/// plane is written for them and neither source plane is read. The caller
/// must mask holes in its attention lists. The staged chunk and all padding
/// remain untouched. No two items write the same staged block.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LatentWindowStage {
    pub latent: u32,
    pub window: u32,
    pub ring_slots: u32,
    pub staged_slots: u32,
}
impl Validate for LatentWindowStage {
    fn validate(&self) -> Result<(), Refusal> {
        if self.latent == 0
            || !self.latent.is_multiple_of(32)
            || self.window < 2
            || self.ring_slots < self.window
            || self.staged_slots < self.window - 1
            || self.ring_slots > i32::MAX as u32
            || self.staged_slots > i32::MAX as u32
        {
            return Err(Refusal(
                "window staging requires block-32 rows and space for the retained prefix",
            ));
        }
        Ok(())
    }
}
impl NumericalContract for LatentWindowStage {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId::new(format!(
            "latent.window-stage.prefix-bits.v1.d{}.w{}.ring{}.stage{}",
            self.latent, self.window, self.ring_slots, self.staged_slots
        )))
    }
}
crate::scratch_record! {
    /// `ring.item_slots` is I32 `[items]`; `first_positions` is I32
    /// `[items]`; `last_positions` is I32 `[ring.sequences]`. The ring's
    /// descriptor names `ring_slots` cells. The staged descriptor is
    /// `Staged { slots: staged_slots }`, with one sequence block per item.
    /// The staged item map and both views' key-list fields are unread.
    /// `live_items` is a device U32 count. All byte offsets use U64.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LatentWindowStageBuffers {
        pub ring: record(LatentWindowBuffers),
        pub staged: record(LatentWindowBuffers),
        pub first_positions: value,
        pub last_positions: value,
        pub live_items: value,
    }
}

/// Independent host images for a bitwise gather oracle. Strides are bytes.
pub struct PlaneImages<'a> {
    pub codes: &'a [u8],
    pub scales: &'a [u8],
    pub code_stride: usize,
    pub scale_stride: usize,
    pub sequences: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn gather_bits(
    query: LatentWindowStage,
    ring: PlaneImages<'_>,
    staged: PlaneImages<'_>,
    item_slots: &[i32],
    first_positions: &[i32],
    last_positions: &[i32],
    live: usize,
) -> Result<(Vec<u8>, Vec<u8>), Refusal> {
    query.validate()?;
    if live > item_slots.len()
        || live > first_positions.len()
        || live > staged.sequences
        || last_positions.len() < ring.sequences
    {
        return Err(Refusal("window staging metadata is too short"));
    }
    let d = query.latent as usize;
    for (view, slots) in [
        (&ring, query.ring_slots as usize),
        (&staged, query.staged_slots as usize),
    ] {
        for (bytes, stride, width) in [
            (view.codes, view.code_stride, d),
            (view.scales, view.scale_stride, d / 32),
        ] {
            let row = slots
                .checked_mul(width)
                .ok_or(Refusal("window plane extent overflows"))?;
            let extent = view
                .sequences
                .saturating_sub(1)
                .checked_mul(stride)
                .and_then(|offset| offset.checked_add(row))
                .ok_or(Refusal("window plane extent overflows"))?;
            if view.sequences == 0 || stride < row || bytes.len() < extent {
                return Err(Refusal("window staging plane is too short"));
            }
        }
    }
    let (mut codes, mut scales) = (staged.codes.to_vec(), staged.scales.to_vec());
    for item in 0..live {
        let Ok(slot) = usize::try_from(item_slots[item]) else {
            continue;
        };
        if slot >= ring.sequences || first_positions[item] < 0 || last_positions[slot] < 0 {
            continue;
        }
        let first = i64::from(first_positions[item]);
        let last = i64::from(last_positions[slot]);
        let start = (first - i64::from(query.window) + 1).max(0);
        for position in start..first {
            if position > last || position < (last - i64::from(query.ring_slots) + 1).max(0) {
                continue;
            }
            let source_row = position as usize % query.ring_slots as usize;
            let output_row = (position - start) as usize;
            for (source, destination, read_stride, write_stride, width) in [
                (
                    ring.codes,
                    &mut codes,
                    ring.code_stride,
                    staged.code_stride,
                    d,
                ),
                (
                    ring.scales,
                    &mut scales,
                    ring.scale_stride,
                    staged.scale_stride,
                    d / 32,
                ),
            ] {
                let from = slot * read_stride + source_row * width;
                let to = item * write_stride + output_row * width;
                destination[to..to + width].copy_from_slice(&source[from..from + width]);
            }
        }
    }
    Ok((codes, scales))
}

#[cfg(test)]
mod tests {
    use super::*;
    const D: usize = 32;
    fn query() -> LatentWindowStage {
        LatentWindowStage {
            latent: D as u32,
            window: 128,
            ring_slots: 136,
            staged_slots: 256,
        }
    }
    fn payload(position: i32, column: usize, scale: bool) -> u8 {
        ((position as usize * 13 + column * 19 + if scale { 7 } else { 3 }) % 251) as u8
    }
    fn ring(last: &[i32]) -> (Vec<u8>, Vec<u8>, usize, usize) {
        let (cs, ss) = (136 * D + 66, 136 + 23);
        let (mut c, mut s) = (vec![0xbd; last.len() * cs], vec![0xed; last.len() * ss]);
        for (slot, &end) in last.iter().enumerate() {
            for p in (end - 135).max(0)..=end {
                for col in 0..D {
                    c[slot * cs + p as usize % 136 * D + col] = payload(p, col, false);
                }
                s[slot * ss + p as usize % 136] = payload(p, 0, true);
            }
        }
        (c, s, cs, ss)
    }
    fn run(first: &[i32], ids: &[i32], last: &[i32]) -> (Vec<u8>, Vec<u8>, usize, usize) {
        let (c, s, cs, ss) = ring(last);
        let (dc, ds) = (256 * D + 18, 256 + 7);
        let (outc, outs) = gather_bits(
            query(),
            PlaneImages {
                codes: &c,
                scales: &s,
                code_stride: cs,
                scale_stride: ss,
                sequences: last.len(),
            },
            PlaneImages {
                codes: &vec![0xa5; first.len() * dc],
                scales: &vec![0xff; first.len() * ds],
                code_stride: dc,
                scale_stride: ds,
                sequences: first.len(),
            },
            ids,
            first,
            last,
            first.len(),
        )
        .unwrap();
        (outc, outs, dc, ds)
    }
    #[test]
    fn prefix_bits_cover_start_boundary_wrap_and_permuted_sequences() {
        for first in [3, 127, 128, 140, 260, i32::MAX] {
            let (c, s, cs, ss) = run(&[first, 0], &[2, 0], &[-1, 999, first - 1]);
            let start = (i64::from(first) - 127).max(0) as i32;
            for (row, p) in (start..first).enumerate() {
                for col in 0..D {
                    assert_eq!(c[row * D + col], payload(p, col, false));
                }
                assert_eq!(s[row], payload(p, 0, true));
            }
            assert!(
                c[(first - start) as usize * D..cs]
                    .iter()
                    .all(|&b| b == 0xa5)
            );
            assert!(s[(first - start) as usize..ss].iter().all(|&b| b == 0xff));
            assert!(c[cs..].iter().all(|&b| b == 0xa5));
        }
        let (c, s, cs, ss) = run(&[140, 7], &[2, 0], &[6, 999, 139]);
        assert_eq!(c[0], payload(13, 0, false));
        assert_eq!(c[cs], payload(0, 0, false));
        assert_eq!(s[ss + 6], payload(6, 0, true));
    }
    #[test]
    fn invalid_metadata_and_stale_ring_positions_never_alias_live_cells() {
        for (first, ids, last) in [
            (vec![8, 8], vec![-1, 3], vec![7, 7, 7]),
            (vec![-1, 0], vec![0, 1], vec![7, -1]),
            (vec![8, 8], vec![0, 1], vec![200, 200]),
        ] {
            let (c, s, _, _) = run(&first, &ids, &last);
            assert!(c.iter().all(|&b| b == 0xa5));
            assert!(s.iter().all(|&b| b == 0xff));
        }
    }
    #[test]
    fn split_prefill_prefix_matches_a_single_staged_history() {
        let (mut whole_c, mut whole_s, _, _) = run(&[140, 7], &[2, 0], &[6, 999, 139]);
        for p in 140..172 {
            let row = (p - 13) as usize;
            for col in 0..D {
                whole_c[row * D + col] = payload(p, col, false);
            }
            whole_s[row] = payload(p, 0, true);
        }
        // The circular writer has committed the first 32-row chunk.
        let (next_c, next_s, _, _) = run(&[172, 7], &[2, 0], &[6, 999, 171]);
        assert_eq!(&next_c[..127 * D], &whole_c[32 * D..159 * D]);
        assert_eq!(&next_s[..127], &whole_s[32..159]);
    }
}
