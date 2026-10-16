//! Gated group pooling: a learned per-channel softmax over each group of
//! consecutive keys, then the weighted sum of the group's keys.
//!
//! Tokens form groups of `group` from the start of the sequence (position 0
//! of the sequence, never a padded batch slot). Group g covers positions
//! `[g * group, (g + 1) * group)` and exists only once complete: the token at
//! position `(g + 1) * group - 1` emits it. For channel d of a complete
//! group, with `t` the offset 0..group of a token in the group:
//!
//! - `logit_t = f32(gate_t[d]) + f32(position[t][d])`, the BF16 gate logits
//!   of the token and the learned position embedding `[group][width]`;
//! - `m = max_t logit_t`, `e_t = exp(logit_t - m)`, `w_t = e_t / sum_t e_t`
//!   with the sums in offset order;
//! - `pooled[d] = bf16(sum_t w_t * f32(key_t[d]))`, the products and sum in
//!   F32 FMA in offset order, rounded once.
//!
//! The single rounding follows the indexer rounding policy in
//! crates/kernels-sm120/cuda/attention/SPARSE.md ("Indexer rounding"): a
//! reference that rounds the probabilities and each product to BF16 is
//! measured against, not copied.

use crate::contract::{DType, Refusal, Validate};

/// The pooling math of one key width.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GatedGroupPooling {
    /// Tokens per group (4); 2 to 16.
    pub group: u32,
    /// Channels of a key, gate logit row and position embedding row.
    pub width: u32,
    /// Storage type of the position embedding: BF16 or F32.
    pub position_dtype: DType,
}

impl GatedGroupPooling {
    /// Complete groups among the first `tokens` tokens of a sequence.
    pub fn complete_groups(&self, tokens: u64) -> u64 {
        tokens / u64::from(self.group)
    }

    /// Tokens of the incomplete tail after `tokens` tokens.
    pub fn tail_tokens(&self, tokens: u64) -> u64 {
        tokens % u64::from(self.group)
    }

    /// BF16 elements of the per-sequence tail state: two banks of
    /// `[group - 1][width]`, the normalized keys first, then the gate
    /// logits. Only tokens of an incomplete group enter it
    /// (`tail_slot`).
    ///
    /// After the first `tokens` tokens of a sequence the live slots are
    /// `0..tail_tokens(tokens)`, slot s holding position
    /// `complete_groups(tokens) * group + s`. The token that completes a
    /// group is never stored: it is pooled directly with the `group - 1`
    /// live slots (or with earlier rows of the same launch), after which the
    /// tail is empty. Slots at or above the live count are dead: never read,
    /// not cleared, and excluded from any state comparison.
    pub fn tail_state_elements(&self) -> u64 {
        2 * u64::from(self.group - 1) * u64::from(self.width)
    }

    /// The tail slot of the token at `position`: `position % group`, or
    /// `None` for the token that completes its group.
    pub fn tail_slot(&self, position: u64) -> Option<u32> {
        let slot = position % u64::from(self.group);
        (slot + 1 < u64::from(self.group)).then_some(slot as u32)
    }

    /// Element offsets of slot `slot`'s key row and gate logit row in the
    /// tail state, each `width` long; `None` for a slot outside the banks.
    pub fn tail_slot_offsets(&self, slot: u32) -> Option<(u64, u64)> {
        (u64::from(slot) + 1 < u64::from(self.group)).then(|| {
            let width = u64::from(self.width);
            let key = u64::from(slot) * width;
            (key, u64::from(self.group - 1) * width + key)
        })
    }
}

impl Validate for GatedGroupPooling {
    fn validate(&self) -> Result<(), Refusal> {
        if !(2..=16).contains(&self.group)
            || self.width == 0
            || !matches!(self.position_dtype, DType::Bf16 | DType::F32)
        {
            return Err(Refusal(
                "invalid gated pooling group, width or position embedding dtype",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn served_pooling_validates_and_counts_complete_groups() {
        let pooling = GatedGroupPooling {
            group: 4,
            width: 128,
            position_dtype: DType::F32,
        };
        pooling.validate().unwrap();
        assert_eq!(pooling.complete_groups(10), 2);
        assert_eq!(pooling.tail_tokens(10), 2);
        assert_eq!(pooling.complete_groups(3), 0);
        assert_eq!(pooling.tail_state_elements(), 2 * 3 * 128);
        // Any width is math; a backend refuses widths it does not build.
        GatedGroupPooling {
            width: 100,
            ..pooling
        }
        .validate()
        .unwrap();
        assert_eq!(
            GatedGroupPooling {
                group: 1,
                ..pooling
            }
            .validate(),
            Err(Refusal(
                "invalid gated pooling group, width or position embedding dtype"
            ))
        );
    }

    #[test]
    fn tail_slots_stay_inside_the_banks_across_pool_boundaries() {
        let pooling = GatedGroupPooling {
            group: 4,
            width: 128,
            position_dtype: DType::Bf16,
        };
        let size = pooling.tail_state_elements();
        assert_eq!(size, 768);
        let slots = (0..12).map(|p| pooling.tail_slot(p)).collect::<Vec<_>>();
        assert_eq!(
            slots,
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3].map(|s| (s < 3).then_some(s))
        );
        for position in 0..64 {
            let Some(slot) = pooling.tail_slot(position) else {
                // The completing token pools with the live slots; the tail
                // is empty afterwards.
                assert_eq!(pooling.tail_tokens(position + 1), 0);
                assert_eq!(pooling.tail_tokens(position), 3);
                continue;
            };
            assert_eq!(u64::from(slot), pooling.tail_tokens(position));
            let (key, gate) = pooling.tail_slot_offsets(slot).unwrap();
            assert!(key + 128 <= 384 && gate >= 384 && gate + 128 <= size);
        }
        assert_eq!(pooling.tail_slot_offsets(2), Some((256, 640)));
        assert_eq!(pooling.tail_slot_offsets(3), None);
    }
}
