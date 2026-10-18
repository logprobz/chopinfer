//! Causal staged lists that exclude unavailable gathered ring cells.
use crate::contract::{NumericalContract, NumericalContractId, Refusal, Validate};

/// A staged prefill has an old prefix and a new chunk. Its absolute base
/// is `max(0, chunk_first - window + 1)`. Prefix positions are valid only
/// in the mapped ring slot's retained interval, ending at `ring_last`.
/// Chunk positions are valid in `[chunk_first, chunk_last]`. A query must
/// itself lie in that chunk. Emit the visible positions oldest first,
/// replacing each missing position by -1 without compacting the list.
/// Every non-hole result is within `[0, staged_slots)`.
/// Invalid row items, chunk bounds or query positions emit an all-hole
/// row. Invalid ring mappings or last positions mask only the prefix.
/// These are the lists to use after `LatentWindowStage`; storage bounds
/// alone do not establish that an old prefix cell was gathered.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LatentStagedWindowKeys {
    pub window: u32,
    pub ring_slots: u32,
    pub staged_slots: u32,
}
impl Validate for LatentStagedWindowKeys {
    fn validate(&self) -> Result<(), Refusal> {
        if self.window < 2
            || self.ring_slots < self.window
            || self.staged_slots < self.window - 1
            || self.ring_slots > i32::MAX as u32
            || self.staged_slots > i32::MAX as u32
        {
            return Err(Refusal(
                "staged window lists require bounded prefix and chunk storage",
            ));
        }
        Ok(())
    }
}
impl NumericalContract for LatentStagedWindowKeys {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId::new(format!(
            "latent.staged-window-keys.retained-prefix.v1.w{}.ring{}.stage{}",
            self.window, self.ring_slots, self.staged_slots
        )))
    }
}
crate::scratch_record! {
    /// I32 row positions and phase-item ids, I32 `[items]` chunk bounds
    /// and item-to-ring-slot map, I32 `[sequences]` ring last positions.
    /// Output is I32 `[rows, window]`; `live_rows` is device U32. The
    /// same chunk bounds and ring snapshot must drive gather and lists;
    /// the circular write follows attention and may then advance the ring.
    #[derive(Clone,Copy,Debug,Eq,PartialEq)]
    pub struct LatentStagedWindowKeyBuffers {
        pub positions: value,
        pub row_items: value,
        pub chunk_first: value,
        pub chunk_last: value,
        pub item_slots: value,
        pub ring_last: value,
        pub items: scalar(u32),
        pub sequences: scalar(u32),
        pub key_lists: value,
        pub live_rows: value,
    }
}
#[allow(clippy::too_many_arguments)]
pub fn keys(
    query: LatentStagedWindowKeys,
    positions: &[i32],
    row_items: &[i32],
    first: &[i32],
    last: &[i32],
    slots: &[i32],
    ring_last: &[i32],
) -> Result<Vec<i32>, Refusal> {
    query.validate()?;
    if positions.len() != row_items.len() || first.len() != last.len() || first.len() != slots.len()
    {
        return Err(Refusal("staged window key metadata shape mismatch"));
    }
    let size = positions
        .len()
        .checked_mul(query.window as usize)
        .ok_or(Refusal("staged key output size overflows"))?;
    let mut out = vec![-1; size];
    for (row, &p) in positions.iter().enumerate() {
        let Ok(item) = usize::try_from(row_items[row]) else {
            continue;
        };
        if item >= first.len()
            || first[item] < 0
            || last[item] < first[item]
            || p < first[item]
            || p > last[item]
        {
            continue;
        }
        let base = (i64::from(first[item]) - i64::from(query.window) + 1).max(0);
        let begin = (i64::from(p) - i64::from(query.window) + 1).max(0);
        let ring = usize::try_from(slots[item])
            .ok()
            .and_then(|s| ring_last.get(s))
            .copied();
        for i in 0..query.window as usize {
            let position = begin + i as i64;
            let offset = position - base;
            if position > i64::from(p) || offset < 0 || offset >= i64::from(query.staged_slots) {
                continue;
            }
            let available = position >= i64::from(first[item])
                || ring.is_some_and(|end| {
                    end >= 0
                        && position <= i64::from(end)
                        && position > i64::from(end) - i64::from(query.ring_slots)
                });
            if available {
                out[row * query.window as usize + i] = offset as i32;
            }
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn q() -> LatentStagedWindowKeys {
        LatentStagedWindowKeys {
            window: 128,
            ring_slots: 136,
            staged_slots: 256,
        }
    }
    #[test]
    fn expired_gathered_prefix_is_masked_before_the_current_chunk() {
        let got = keys(q(), &[8], &[0], &[8], &[8], &[0], &[200]).unwrap();
        assert!(got[..8].iter().all(|&i| i == -1));
        assert_eq!(got[8], 8);
        assert!(got[9..].iter().all(|&i| i == -1));
        let got = keys(q(), &[100], &[0], &[100], &[103], &[0], &[200]).unwrap();
        assert!(got[..65].iter().all(|&i| i == -1));
        assert_eq!(&got[65..101], &(65..101).collect::<Vec<_>>());
    }
    #[test]
    fn invalid_metadata_is_holes_and_missing_ring_does_not_hide_new_rows() {
        let got = keys(
            q(),
            &[8, 8, 8, 8],
            &[0, 1, -1, 3],
            &[8, 9],
            &[9, 8],
            &[-1, 0],
            &[7],
        )
        .unwrap();
        assert_eq!(got[8], 8);
        assert_eq!(got.iter().filter(|&&x| x >= 0).count(), 1);
        assert!(
            keys(q(), &[-1], &[0], &[0], &[0], &[0], &[-1])
                .unwrap()
                .iter()
                .all(|&x| x == -1)
        );
    }
    #[test]
    fn split_prefill_lists_retain_each_queries_causal_window() {
        for (first, end) in [(140, 171), (172, 187)] {
            let positions = (first..=end).collect::<Vec<_>>();
            let got = keys(
                q(),
                &positions,
                &vec![0; positions.len()],
                &[first],
                &[end],
                &[0],
                &[first - 1],
            )
            .unwrap();
            for (row, values) in got.chunks_exact(128).enumerate() {
                assert_eq!(values, &(row as i32..row as i32 + 128).collect::<Vec<_>>());
            }
        }
    }
}
