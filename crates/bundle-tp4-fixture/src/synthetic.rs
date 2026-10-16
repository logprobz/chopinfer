//! The synthetic bytes and the synthetic row, and nothing a model computes.
//!
//! Every value here is a bounded deterministic function of three things: the
//! digests of the four rank images this instance authenticated, the folded
//! history of every token a committed execution appended, and the tokens of the
//! batch being run. There is no attention, no MoE routing, no NVFP4 arithmetic,
//! and no TP4 weight. The only TP4 facts the row borrows are its width and the
//! rank count of TP4's deployment topology.
//!
//! The arithmetic is integer mixing plus one division per logit, so the same
//! authenticated images, the same committed history, and the same batch give the
//! same row on every host and every build.

use runtime_weights::sha256_bytes;

use crate::contract::{
    ProgramGeometryV1, TP4_FIXTURE_GEOMETRY_OFFSET_V1, TP4_FIXTURE_IMAGE_FILLER_OFFSET_V1,
    TP4_FIXTURE_IMAGE_MAGIC_OFFSET_V1, TP4_FIXTURE_IMAGE_MAGIC_V1,
    TP4_FIXTURE_IMAGE_RANK_COUNT_OFFSET_V1, TP4_FIXTURE_IMAGE_RANK_OFFSET_V1,
    TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1, TP4_FIXTURE_RANK_COUNT_V1, TP4_FIXTURE_STATE_BYTES_V1,
    TP4_FIXTURE_STATE_SEED_OFFSET_V1,
};

/// The bound every logit this fixture emits stays inside.
pub const TP4_FIXTURE_LOGIT_BOUND_V1: f32 = 8.0;

/// The logit the one peak token carries, above every drawn value.
pub const TP4_FIXTURE_PEAK_LOGIT_V1: f32 = TP4_FIXTURE_LOGIT_BOUND_V1;

/// What one logit's raw 22-bit draw is divided by before it is centred.
///
/// A 22-bit draw over this lands in `[0, 8)`, and centring it gives `[-4, 4)`,
/// so every drawn value stays strictly below the peak.
const TP4_FIXTURE_LOGIT_SCALE_V1: f32 = 524_288.0;

/// The token identifier every row this fixture emits peaks at.
///
/// This is a fixture filler identifier, not a TP4 token, and it is deliberately
/// not one of the published markers: it sits in the added block between the
/// published stop token and the first published marker, it is not special, and
/// the generated tokenizer decodes it to ordinary visible text. A greedy sample
/// over one of these rows therefore has one exact expected answer that a
/// completion can carry.
pub const TP4_FIXTURE_PEAK_TOKEN_ID_V1: u32 = 154_821;

/// The lanes the folded token history occupies on the wire and in memory.
pub type TokenHistoryV1 = u64;

/// Reads the folded history from the exact little-endian bytes the host stores.
///
/// A slice of the wrong length is refused rather than padded, because a state
/// allocation that is not the declared one is not this fixture's state.
#[must_use]
pub fn history_from_bytes_v1(bytes: &[u8]) -> Option<TokenHistoryV1> {
    let bytes: [u8; TP4_FIXTURE_STATE_BYTES_V1 as usize] = bytes.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}

/// Writes the folded history back as the exact bytes the host stores.
#[must_use]
pub fn history_to_bytes_v1(history: TokenHistoryV1) -> Vec<u8> {
    history.to_le_bytes().to_vec()
}

/// Folds one batch of tokens into the history, in the order they were scheduled.
///
/// Reordering the batch or replaying one token changes the result, so a
/// committed execution is observable in every row after it and a cancelled one
/// is observable in none.
#[must_use]
pub fn fold_tokens_v1(history: TokenHistoryV1, tokens: &[u32]) -> TokenHistoryV1 {
    tokens.iter().fold(history, |folded, token| {
        mix_v1(folded.rotate_left(7) ^ u64::from(*token))
    })
}

/// The history a fresh instance starts from, before anything is committed.
#[must_use]
pub fn state_seed_history_v1() -> TokenHistoryV1 {
    mix_v1(u64::from(ProgramGeometryV1::declared().vocabulary))
}

/// One rank's whole program image.
///
/// The layout is fixed: the geometry words, the prefix that names the image, the
/// rank ordinal and the rank count that image was emitted for, the seed history,
/// then deterministic filler seeded by the rank. No two ranks share an image, so
/// a package whose ranks were transposed or duplicated fails authentication.
#[must_use]
pub fn program_image_bytes_v1(rank: u32) -> Vec<u8> {
    let mut image = vec![0_u8; TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1 as usize];
    write_at_v1(
        &mut image,
        TP4_FIXTURE_GEOMETRY_OFFSET_V1,
        &ProgramGeometryV1::declared().to_bytes(),
    );
    write_at_v1(
        &mut image,
        TP4_FIXTURE_IMAGE_MAGIC_OFFSET_V1,
        TP4_FIXTURE_IMAGE_MAGIC_V1,
    );
    write_at_v1(
        &mut image,
        TP4_FIXTURE_IMAGE_RANK_OFFSET_V1,
        &rank.to_le_bytes(),
    );
    write_at_v1(
        &mut image,
        TP4_FIXTURE_IMAGE_RANK_COUNT_OFFSET_V1,
        &TP4_FIXTURE_RANK_COUNT_V1.to_le_bytes(),
    );
    write_at_v1(
        &mut image,
        TP4_FIXTURE_STATE_SEED_OFFSET_V1,
        &history_to_bytes_v1(state_seed_history_v1()),
    );
    let seed = mix_v1(mix_v1(TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1) ^ u64::from(rank));
    let filler = TP4_FIXTURE_IMAGE_FILLER_OFFSET_V1 as usize;
    for (offset, byte) in image[filler..].iter_mut().enumerate() {
        *byte = (mix_v1(seed ^ (filler + offset) as u64) >> 24) as u8;
    }
    image
}

/// The digest one synthetic rank image hashes to.
#[must_use]
pub fn program_image_digest_v1(rank: u32) -> String {
    sha256_bytes(&program_image_bytes_v1(rank))
}

/// Every rank's digest, in rank order.
#[must_use]
pub fn program_image_digests_v1() -> Vec<String> {
    (0..TP4_FIXTURE_RANK_COUNT_V1)
        .map(program_image_digest_v1)
        .collect()
}

/// The row this fixture emits for one batch, as little-endian `f32` bits.
///
/// Every value is finite and inside [`TP4_FIXTURE_LOGIT_BOUND_V1`] by
/// construction, the row is exactly `vocabulary` logits wide, and it peaks at
/// [`TP4_FIXTURE_PEAK_TOKEN_ID_V1`] so a greedy sampler has one exact answer
/// to reach. It depends on every rank's authenticated image, so a package
/// missing one rank produces a different row. None of it is TP4 arithmetic and
/// none of it is claimed to be.
#[must_use]
pub fn synthetic_row_v1(
    image_digests: &[String],
    history: TokenHistoryV1,
    tokens: &[u32],
    vocabulary: u32,
) -> Vec<u8> {
    let mut seed = mix_v1(u64::from(vocabulary));
    for digest in image_digests {
        seed = fold_bytes_v1(seed, digest.as_bytes());
    }
    seed = mix_v1(seed ^ history);
    seed = fold_tokens_v1(seed, tokens);
    let mut row = Vec::with_capacity(vocabulary as usize * 4);
    for index in 0..vocabulary {
        let logit = if index == TP4_FIXTURE_PEAK_TOKEN_ID_V1 {
            TP4_FIXTURE_PEAK_LOGIT_V1
        } else {
            // A 22-bit draw scaled into [0, 8) and centred, so no drawn value
            // can reach the peak and every one of them is finite.
            let drawn = (mix_v1(seed ^ u64::from(index)) >> 42) as f32;
            drawn / TP4_FIXTURE_LOGIT_SCALE_V1 - TP4_FIXTURE_LOGIT_BOUND_V1 / 2.0
        };
        row.extend_from_slice(&logit.to_le_bytes());
    }
    row
}

/// Reads one emitted row back as the values a sampler would consume.
#[must_use]
pub fn row_logits_v1(row: &[u8]) -> Vec<f32> {
    row.chunks_exact(4)
        .map(|bits| f32::from_le_bytes([bits[0], bits[1], bits[2], bits[3]]))
        .collect()
}

fn write_at_v1(image: &mut [u8], offset: u64, bytes: &[u8]) {
    let start = offset as usize;
    image[start..start + bytes.len()].copy_from_slice(bytes);
}

/// Folds one byte string into a seed, one byte at a time.
fn fold_bytes_v1(seed: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(seed, |folded, byte| mix_v1(folded ^ u64::from(*byte)))
}

/// The fixed bit mixer every deterministic byte of this fixture comes from.
const fn mix_v1(value: u64) -> u64 {
    let mut mixed = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::contract::{
        ProgramGeometryV1, TP4_FIXTURE_GEOMETRY_BYTES_V1, TP4_FIXTURE_GEOMETRY_OFFSET_V1,
        TP4_FIXTURE_IMAGE_MAGIC_OFFSET_V1, TP4_FIXTURE_IMAGE_MAGIC_V1,
        TP4_FIXTURE_IMAGE_RANK_COUNT_OFFSET_V1, TP4_FIXTURE_IMAGE_RANK_OFFSET_V1,
        TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1, TP4_FIXTURE_RANK_COUNT_V1, TP4_FIXTURE_STATE_BYTES_V1,
        TP4_FIXTURE_STATE_SEED_OFFSET_V1, tp4_fixture_vocabulary_v1,
    };

    use super::{
        TP4_FIXTURE_LOGIT_BOUND_V1, TP4_FIXTURE_PEAK_TOKEN_ID_V1, fold_tokens_v1,
        history_from_bytes_v1, history_to_bytes_v1, program_image_bytes_v1,
        program_image_digest_v1, program_image_digests_v1, row_logits_v1, state_seed_history_v1,
        synthetic_row_v1,
    };

    fn row(history: u64, tokens: &[u32]) -> Vec<f32> {
        row_logits_v1(&synthetic_row_v1(
            &program_image_digests_v1(),
            history,
            tokens,
            tp4_fixture_vocabulary_v1(),
        ))
    }

    /// Every rank ships a distinct image that names its own rank, and both
    /// declared views span bytes the image actually holds.
    #[test]
    fn each_rank_image_names_its_rank_and_carries_the_views_it_declares() {
        let images = (0..TP4_FIXTURE_RANK_COUNT_V1)
            .map(program_image_bytes_v1)
            .collect::<Vec<_>>();

        assert_eq!(
            images.len(),
            4,
            "TP4 deploys at TP4 and so does this fixture"
        );
        for (rank, image) in images.iter().enumerate() {
            let rank = rank as u32;
            assert_eq!(image.len(), TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1 as usize);
            let geometry_start = TP4_FIXTURE_GEOMETRY_OFFSET_V1 as usize;
            let geometry_end = geometry_start + TP4_FIXTURE_GEOMETRY_BYTES_V1 as usize;
            assert_eq!(
                ProgramGeometryV1::from_bytes(&image[geometry_start..geometry_end]),
                Some(ProgramGeometryV1::declared())
            );
            let magic_start = TP4_FIXTURE_IMAGE_MAGIC_OFFSET_V1 as usize;
            assert_eq!(
                &image[magic_start..magic_start + TP4_FIXTURE_IMAGE_MAGIC_V1.len()],
                TP4_FIXTURE_IMAGE_MAGIC_V1
            );
            let rank_start = TP4_FIXTURE_IMAGE_RANK_OFFSET_V1 as usize;
            assert_eq!(&image[rank_start..rank_start + 4], rank.to_le_bytes());
            let count_start = TP4_FIXTURE_IMAGE_RANK_COUNT_OFFSET_V1 as usize;
            assert_eq!(
                &image[count_start..count_start + 4],
                TP4_FIXTURE_RANK_COUNT_V1.to_le_bytes()
            );
            let seed_start = TP4_FIXTURE_STATE_SEED_OFFSET_V1 as usize;
            let seed_end = seed_start + TP4_FIXTURE_STATE_BYTES_V1 as usize;
            assert_eq!(
                history_from_bytes_v1(&image[seed_start..seed_end]),
                Some(state_seed_history_v1())
            );
            assert_eq!(
                *image,
                program_image_bytes_v1(rank),
                "the generator is deterministic"
            );
        }
        let distinct = images.iter().collect::<BTreeSet<_>>();
        assert_eq!(distinct.len(), images.len(), "no two ranks share an image");
        let digests = program_image_digests_v1();
        assert_eq!(digests.len(), TP4_FIXTURE_RANK_COUNT_V1 as usize);
        assert_eq!(digests[2], program_image_digest_v1(2));
    }

    /// The history round-trips through exactly the bytes the state layout holds.
    #[test]
    fn the_history_round_trips_through_the_declared_state_bytes() {
        let history = fold_tokens_v1(state_seed_history_v1(), &[7, 11, 13]);

        let bytes = history_to_bytes_v1(history);

        assert_eq!(bytes.len(), TP4_FIXTURE_STATE_BYTES_V1 as usize);
        assert_eq!(history_from_bytes_v1(&bytes), Some(history));
        assert_eq!(history_from_bytes_v1(&bytes[..bytes.len() - 1]), None);
    }

    /// The fold is a recurrence, so token order and replay both change it.
    #[test]
    fn the_fold_depends_on_every_token_and_on_their_order() {
        let seed = state_seed_history_v1();

        let forward = fold_tokens_v1(seed, &[1, 2, 3]);

        assert_ne!(forward, seed);
        assert_ne!(forward, fold_tokens_v1(seed, &[3, 2, 1]));
        assert_ne!(fold_tokens_v1(forward, &[1, 2, 3]), forward);
        assert_eq!(forward, fold_tokens_v1(seed, &[1, 2, 3]));
        assert_eq!(fold_tokens_v1(seed, &[]), seed);
    }

    /// Every logit is finite, bounded, and the row is the declared width.
    #[test]
    fn the_row_is_the_declared_width_finite_bounded_and_peaked() {
        let logits = row(state_seed_history_v1(), &[5; 32]);

        assert_eq!(logits.len(), tp4_fixture_vocabulary_v1() as usize);
        assert_eq!(logits.len(), 154_880);
        for (index, logit) in logits.iter().enumerate() {
            assert!(logit.is_finite(), "logit {index} is not finite");
            assert!(
                logit.abs() <= TP4_FIXTURE_LOGIT_BOUND_V1,
                "logit {index} left the declared bound"
            );
        }
        let peak = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .expect("a row of this width has a maximum");
        assert_eq!(peak, TP4_FIXTURE_PEAK_TOKEN_ID_V1 as usize);
    }

    /// The row is a function of every rank's image, the history, and the batch.
    #[test]
    fn the_row_is_deterministic_in_the_images_the_history_and_the_batch() {
        let seed = state_seed_history_v1();
        let committed = fold_tokens_v1(seed, &[9]);
        let vocabulary = tp4_fixture_vocabulary_v1();

        assert_eq!(row(seed, &[1, 2]), row(seed, &[1, 2]));
        assert_ne!(row(seed, &[1, 2]), row(seed, &[2, 1]));
        assert_ne!(
            row(seed, &[1, 2]),
            row(committed, &[1, 2]),
            "a committed execution has to move every row after it"
        );

        // Dropping a rank, or transposing two of them, is a different row.
        let digests = program_image_digests_v1();
        let full = synthetic_row_v1(&digests, seed, &[1, 2], vocabulary);
        assert_ne!(
            full,
            synthetic_row_v1(&digests[..3], seed, &[1, 2], vocabulary),
            "the row must depend on every rank this instance authenticated"
        );
        let mut swapped = digests.clone();
        swapped.swap(0, 3);
        assert_ne!(
            full,
            synthetic_row_v1(&swapped, seed, &[1, 2], vocabulary),
            "the row must depend on the order the ranks were authenticated in"
        );
    }
}
