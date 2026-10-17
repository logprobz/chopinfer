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

use crate::contract::{
    DType, NumericalContract, NumericalContractId, RecurrentPhase, Refusal, Validate, valid_eps,
    valid_phase,
};

/// F32 gated pooling without a position embedding, followed by RMSNorm.
///
/// Keys and gate logits are F32 `[rows][width]`. For a completed group,
/// apply a per-channel softmax to its gate logits, then sum weighted keys
/// in group order in F32. Round the pooled vector to BF16 before RMSNorm.
/// RMSNorm accumulates in F32, uses BF16 weights and `norm_eps_bits`,
/// multiplies the weight before rounding once to BF16. There is no bias,
/// mean subtraction, APE or RoPE. Width 512 and group 2 use this contract.
/// A ratio-1 projection uses ordinary RMSNorm without a pooling op.
///
/// Group j covers `[j * group, (j + 1) * group)` and is emitted only by
/// its completing row. Incomplete rows retain their original F32 key and
/// gate, not normalized or BF16-rounded values. State has two banks of
/// `[group - 1][width]` F32 elements, keys first. Live slots are
/// `0..(position + 1) % group`; other slots are dead and never read.
/// Prefill/decode update state in place. Verify writes the whole state
/// after each row to a distinct checkpoint; commit keeps the last accepted
/// checkpoint. A rejected compressed output is outside visibility and is
/// overwritten before reuse. Empty groups emit nothing, including during
/// decode; consumers still republish the existing compressed cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GatedPoolingQuery {
    pub group: u32,
    pub width: u32,
    pub norm_eps_bits: u32,
    pub phase: RecurrentPhase,
    pub capacity: u32,
}

impl GatedPoolingQuery {
    /// Shared group/tail geometry; this op has no position embedding.
    pub fn geometry(&self) -> GatedGroupPooling {
        GatedGroupPooling {
            group: self.group,
            width: self.width,
            position_dtype: DType::F32,
        }
    }

    /// Bytes of one F32 tail entry. Call only after validation.
    pub fn tail_state_bytes(&self) -> u64 {
        self.geometry().tail_state_elements() * 4
    }
}

impl Validate for GatedPoolingQuery {
    fn validate(&self) -> Result<(), Refusal> {
        self.geometry().validate()?;
        if !valid_eps(self.norm_eps_bits) || !valid_phase(self.phase) || self.capacity == 0 {
            return Err(Refusal("invalid gated pooling RMSNorm, phase or capacity"));
        }
        Ok(())
    }
}

impl NumericalContract for GatedPoolingQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let phase = match self.phase {
            RecurrentPhase::Prefill => "prefill",
            RecurrentPhase::Decode => "decode",
            RecurrentPhase::Verify { .. } => "verify",
        };
        Ok(NumericalContractId::new(format!(
            "gated-pooling-f32-no-position-bf16-rmsnorm.{phase}.v1"
        )))
    }
}

crate::scratch_record! {
    /// F32 tail entries, each `GatedPoolingQuery::tail_state_bytes` bytes.
    /// I32 `slot_ids[items]` name source entries; verify uses distinct I32
    /// `checkpoint_ids[items][max_rows]`, ignored in prefill/decode.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct GatedPoolingState {
        pub state: value,
        pub entries: scalar(u32),
        pub slot_ids: value,
        pub checkpoint_ids: option,
    }
}

crate::scratch_record! {
    /// F32 key and gate rows, BF16 RMSNorm weight `[width]`, I32 item and
    /// token positions, and a device live-row count. BF16 output is
    /// `[rows][width]`: only a completing row writes its row, with output
    /// position `position / group`; other rows write output position -1 and
    /// leave their output vector untouched. Negative input positions are
    /// holes and update no state. Rows of an item are consecutive.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct GatedPoolingBuffers {
        pub key: value,
        pub gate: value,
        pub norm_weight: value,
        pub positions: value,
        pub row_batch: value,
        pub live_rows: value,
        pub tail: record(GatedPoolingState),
        pub output: value,
        pub output_positions: value,
    }
}

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
