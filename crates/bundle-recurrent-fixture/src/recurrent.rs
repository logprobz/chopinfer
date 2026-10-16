//! The recurrent step this fixture computes, as exact integer arithmetic.
//!
//! There is no attention and no KV page here. The whole of the model is a fixed
//! number of 32-bit lanes that carry forward from one committed transaction to
//! the next, folded over the tokens of the batch in order, and a logits row read
//! back out of those lanes. Every operation is wrapping integer arithmetic, so
//! the same lanes and the same tokens give the same row on every host.

/// How many 32-bit lanes the recurrent state carries.
pub const RECURRENT_LANES_V1: usize = 16;

/// How many bytes those lanes occupy on the wire and in device memory.
pub const RECURRENT_STATE_BYTES_V1: u64 = (RECURRENT_LANES_V1 * 4) as u64;

/// The odd multiplier the fold mixes each token with.
const RECURRENT_MIX_V1: u32 = 0x9E37_79B9;

/// The odd multiplier the logits row spreads a lane across the vocabulary with.
const RECURRENT_SPREAD_V1: u32 = 0x0100_0193;

/// How far the fold rotates a lane before it mixes the carry in.
const RECURRENT_ROTATE_V1: u32 = 5;

/// The number of distinct values the 24 bits a logit is read from can take.
const RECURRENT_LOGIT_SCALE_V1: f32 = 16_777_216.0;

/// The lanes of one recurrent state.
pub type RecurrentLanesV1 = [u32; RECURRENT_LANES_V1];

/// Reads lanes from the exact little-endian bytes the host holds.
///
/// A slice of the wrong length is refused rather than padded, because a state
/// allocation that is not the declared one is not this fixture's state.
#[must_use]
pub fn lanes_from_bytes_v1(bytes: &[u8]) -> Option<RecurrentLanesV1> {
    if bytes.len() != RECURRENT_STATE_BYTES_V1 as usize {
        return None;
    }
    let mut lanes = [0_u32; RECURRENT_LANES_V1];
    for (lane, word) in lanes.iter_mut().zip(bytes.chunks_exact(4)) {
        *lane = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
    }
    Some(lanes)
}

/// Writes lanes back as the exact little-endian bytes the host stores.
#[must_use]
pub fn lanes_to_bytes_v1(lanes: &RecurrentLanesV1) -> Vec<u8> {
    lanes.iter().flat_map(|lane| lane.to_le_bytes()).collect()
}

/// Folds one batch of tokens into the lanes, in the order they were scheduled.
///
/// Each lane depends on the lane before it within the same token, and on every
/// token before it, so the result is a genuine sequential recurrence: reordering
/// the tokens or replaying one changes the state.
#[must_use]
pub fn fold_tokens_v1(lanes: &RecurrentLanesV1, tokens: &[u32]) -> RecurrentLanesV1 {
    let mut folded = *lanes;
    let mut carry = folded[RECURRENT_LANES_V1 - 1];
    for token in tokens {
        for (index, lane) in folded.iter_mut().enumerate() {
            let mixed = lane.rotate_left(RECURRENT_ROTATE_V1)
                ^ carry
                    .wrapping_add(token.wrapping_mul(RECURRENT_MIX_V1))
                    .wrapping_add(index as u32);
            carry = mixed;
            *lane = mixed;
        }
    }
    folded
}

/// Reads one bounded logits row out of the lanes, as little-endian `f32` bits.
///
/// Every value lands in `[-1, 1)` by construction: the 24 high bits of a lane
/// word are read as a fraction of one, then centred. The row is exactly
/// `vocabulary` logits wide, which is the width the program image declared.
#[must_use]
pub fn logits_row_v1(lanes: &RecurrentLanesV1, vocabulary: u32) -> Vec<u8> {
    let mut row = Vec::with_capacity(vocabulary as usize * 4);
    for token in 0..vocabulary {
        let lane = lanes[token as usize % RECURRENT_LANES_V1];
        let bits = lane ^ token.wrapping_mul(RECURRENT_SPREAD_V1);
        let fraction = (bits >> 8) as f32 / RECURRENT_LOGIT_SCALE_V1;
        row.extend_from_slice(&(fraction * 2.0 - 1.0).to_le_bytes());
    }
    row
}

#[cfg(test)]
mod tests {
    use super::{
        RECURRENT_LANES_V1, RECURRENT_STATE_BYTES_V1, fold_tokens_v1, lanes_from_bytes_v1,
        lanes_to_bytes_v1, logits_row_v1,
    };

    fn seed() -> [u32; RECURRENT_LANES_V1] {
        let mut lanes = [0_u32; RECURRENT_LANES_V1];
        for (index, lane) in lanes.iter_mut().enumerate() {
            *lane = 0x9E37_79B9_u32.wrapping_mul(index as u32 + 1);
        }
        lanes
    }

    #[test]
    fn lanes_round_trip_through_the_bytes_the_host_stores() {
        let lanes = seed();

        let bytes = lanes_to_bytes_v1(&lanes);

        assert_eq!(bytes.len(), RECURRENT_STATE_BYTES_V1 as usize);
        assert_eq!(lanes_from_bytes_v1(&bytes), Some(lanes));
        assert_eq!(lanes_from_bytes_v1(&bytes[..bytes.len() - 1]), None);
    }

    /// The fold is a recurrence, so token order and replay both change it.
    #[test]
    fn the_fold_depends_on_every_token_and_on_their_order() {
        let lanes = seed();

        let forward = fold_tokens_v1(&lanes, &[1, 2, 3]);
        let reversed = fold_tokens_v1(&lanes, &[3, 2, 1]);
        let repeated = fold_tokens_v1(&forward, &[1, 2, 3]);

        assert_ne!(forward, lanes);
        assert_ne!(forward, reversed);
        assert_ne!(repeated, forward);
        // The same lanes and the same tokens are the same state, every time.
        assert_eq!(forward, fold_tokens_v1(&lanes, &[1, 2, 3]));
        assert_eq!(fold_tokens_v1(&lanes, &[]), lanes);
    }

    #[test]
    fn every_logit_is_a_finite_value_inside_the_declared_bound() {
        let row = logits_row_v1(&fold_tokens_v1(&seed(), &[7, 11]), 64);

        assert_eq!(row.len(), 64 * 4);
        for logit in row.chunks_exact(4) {
            let value = f32::from_le_bytes([logit[0], logit[1], logit[2], logit[3]]);
            assert!(value.is_finite(), "logit {value} is not finite");
            assert!((-1.0..1.0).contains(&value), "logit {value} is unbounded");
        }
    }
}
