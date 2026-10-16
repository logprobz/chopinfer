//! The sharded argmax oracle's order (`RowQuery::ShardedArgmaxPartial`,
//! `RowQuery::ShardedArgmaxFinal`): ties to the lower global id, -0 equal
//! to +0, and every NaN below every other value, so a row of only NaN
//! yields its lowest id. Its F32 form picks the id a packed-key draft
//! argmax picks, on finite rows and on rows with NaN.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{ArgmaxLayout, DType, RowQuery, Validate};
use kernel_core::oracle::{
    argmax_rank, sharded_argmax, sharded_argmax_final, sharded_argmax_partials,
};

const NEG_NAN: f32 = f32::from_bits(0xffc0_0001);
const PAYLOAD_NAN: f32 = f32::from_bits(0x7f80_0123);

fn ids(partials: &[(f32, usize)]) -> Vec<usize> {
    partials.iter().map(|&(_, id)| id).collect()
}

fn bits(partials: &[(f32, usize)]) -> Vec<u32> {
    partials.iter().map(|&(score, _)| score.to_bits()).collect()
}

#[test]
fn ranks_follow_the_ieee_order_with_nan_lowest() {
    let ordered = [
        f32::NEG_INFINITY,
        f32::MIN,
        -1.0,
        -f32::from_bits(1),
        0.0,
        f32::from_bits(1),
        1.0,
        f32::MAX,
        f32::INFINITY,
    ];
    for pair in ordered.windows(2) {
        assert!(argmax_rank(pair[0]) < argmax_rank(pair[1]), "{pair:?}");
    }
    assert_eq!(argmax_rank(-0.0), argmax_rank(0.0));
    for nan in [f32::NAN, -f32::NAN, NEG_NAN, PAYLOAD_NAN] {
        assert_eq!(argmax_rank(nan), 0);
        assert!(argmax_rank(nan) < argmax_rank(f32::NEG_INFINITY));
    }
}

#[test]
fn partials_keep_the_winning_bits_and_the_lowest_id() {
    let shards: [&[f32]; 4] = [
        // Ties: the lower id.
        &[2.0, 5.0, 5.0, -1.0],
        // -0 equals +0: the lower id, with its own bits.
        &[-0.0, 0.0, -3.0, f32::NEG_INFINITY],
        // NaN ranks below -inf.
        &[f32::NAN, f32::NEG_INFINITY, NEG_NAN, f32::NEG_INFINITY],
        // Only NaN: the lowest id, with its payload.
        &[PAYLOAD_NAN, NEG_NAN, f32::NAN, PAYLOAD_NAN],
    ];
    let partials = sharded_argmax_partials(&shards).unwrap();
    assert_eq!(ids(&partials), [1, 4, 9, 12]);
    assert_eq!(
        bits(&partials),
        [
            5.0_f32.to_bits(),
            (-0.0_f32).to_bits(),
            f32::NEG_INFINITY.to_bits(),
            PAYLOAD_NAN.to_bits()
        ]
    );
    assert_eq!(sharded_argmax_final(&partials).unwrap(), 1);
    // A rank of only NaN loses to any other rank, and only NaN everywhere
    // yields global id 0.
    let nans: [&[f32]; 2] = [&[f32::NAN, NEG_NAN], &[PAYLOAD_NAN, f32::NAN]];
    assert_eq!(sharded_argmax(&nans).unwrap(), 0);
    let late: [&[f32]; 2] = [&[f32::NAN, NEG_NAN], &[PAYLOAD_NAN, f32::NEG_INFINITY]];
    assert_eq!(sharded_argmax(&late).unwrap(), 3);
    // Equal scores across ranks: the lower id, whatever the rank order.
    let tied = [(1.0, 7), (-0.0, 2), (1.0, 5), (0.0, 1)];
    assert_eq!(sharded_argmax_final(&tied).unwrap(), 5);
    let zeros = [(0.0, 9), (-0.0, 3)];
    assert_eq!(sharded_argmax_final(&zeros).unwrap(), 3);
}

#[test]
fn empty_shards_and_partials_are_refused() {
    assert!(sharded_argmax_final(&[]).is_err());
    assert!(sharded_argmax_partials(&[&[]]).is_err());
    assert!(sharded_argmax_partials(&[]).is_err());
}

#[test]
fn queries_take_bf16_or_f32_logits_and_a_stride() {
    let partial = |logits| RowQuery::ShardedArgmaxPartial {
        vocab_per_rank: 64,
        ranks: 2,
        logits,
    };
    assert!(partial(DType::Bf16).validate().is_ok());
    assert!(partial(DType::F32).validate().is_ok());
    assert!(partial(DType::F16).validate().is_err());
    let last = |pairs| RowQuery::ShardedArgmaxFinal { ranks: 2, pairs };
    assert!(last(ArgmaxLayout::RowMajor).validate().is_ok());
    assert!(
        last(ArgmaxLayout::RankMajor { stride: 32 })
            .validate()
            .is_ok()
    );
    assert!(
        last(ArgmaxLayout::RankMajor { stride: 0 })
            .validate()
            .is_err()
    );
}

/// A packed-key draft argmax, independent of the oracle: each rank packs
/// its best candidate's order (`order`) and token as the key
/// `(order << 32) | (0xffffffff - token)`, the larger key winning, into
/// F32 words `{order >> 16, order & 0xffff, token}` in its slot of zeros,
/// an F32 all-reduce of the slots sums them exactly, and every rank takes
/// the best key of the summed slots.
fn packed_key_argmax(shards: &[&[f32]]) -> u32 {
    /// A logit's order: larger is better, -0 as +0, every NaN below -inf.
    fn order(value: f32) -> u32 {
        if value.is_nan() {
            return 0;
        }
        let mut bits = value.to_bits();
        if bits == 0x8000_0000 {
            bits = 0;
        }
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits | 0x8000_0000
        }
    }
    fn key(order: u32, token: u32) -> u64 {
        (u64::from(order) << 32) | u64::from(0xffff_ffff - token)
    }
    let ranks = shards.len();
    let mut summed = vec![0.0_f32; ranks * 4];
    let mut offset = 0_u32;
    for (slot, shard) in shards.iter().enumerate() {
        let best = shard
            .iter()
            .enumerate()
            .map(|(local, &v)| key(order(v), local as u32))
            .fold(0, u64::max);
        let order = (best >> 32) as u32;
        let token = 0xffff_ffff - best as u32 + offset;
        let mut words = vec![0.0_f32; ranks * 4];
        words[slot * 4] = (order >> 16) as f32;
        words[slot * 4 + 1] = (order & 0xffff) as f32;
        words[slot * 4 + 2] = token as f32;
        for (sum, word) in summed.iter_mut().zip(words) {
            *sum += word;
        }
        offset += shard.len() as u32;
    }
    let best = summed
        .chunks_exact(4)
        .map(|slot| key(((slot[0] as u32) << 16) | slot[1] as u32, slot[2] as u32))
        .fold(0, u64::max);
    0xffff_ffff - best as u32
}

/// Logits that tie often and sit near each other: steps of 2^-10 above a
/// few bases, in rows of four kinds by `seed`: plain; at most zero with
/// zeros of both signs (so the zeros win); with infinities of both signs;
/// and only -inf with a few -0.
fn near_tie_logits(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let pick = (state >> 33) as u32;
            let near =
                [20.0_f32, -3.5, 0.25][(pick % 3) as usize] + ((pick >> 8) % 64) as f32 / 1024.0;
            match (seed % 4, pick % 61) {
                (1, 0) => -0.0,
                (1, 1) => 0.0,
                (1, _) => -near.abs(),
                (2, 0) => f32::INFINITY,
                (2, 1) => f32::NEG_INFINITY,
                (3, 0) => -0.0,
                (3, _) => f32::NEG_INFINITY,
                _ => near,
            }
        })
        .collect()
}

#[test]
fn f32_partials_pick_the_packed_key_argmax_ids() {
    let vocab = 2048;
    for ranks in [1, 2, 4] {
        let shard = vocab / ranks;
        for seed in 0..64_u64 {
            let mut row = near_tie_logits(seed, vocab);
            // Some rows lose their winners to NaN or are NaN throughout.
            match seed % 16 {
                13 => row.iter_mut().step_by(3).for_each(|v| *v = f32::NAN),
                14 => row.iter_mut().for_each(|v| *v = NEG_NAN),
                15 => row.iter_mut().skip(shard).for_each(|v| *v = PAYLOAD_NAN),
                _ => {}
            }
            let shards = row.chunks_exact(shard).collect::<Vec<_>>();
            let oracle = sharded_argmax(&shards).unwrap();
            assert_eq!(
                oracle as u32,
                packed_key_argmax(&shards),
                "ranks {ranks} seed {seed}"
            );
        }
    }
}

#[test]
fn bf16_logits_widen_before_the_order() {
    let bf16 = [1.0, 1.001, -0.0, 0.0].map(f32_to_bf16);
    // 1.001 rounds to 1.0 in BF16: a tie, so the lower id.
    let widened = bf16.map(bf16_to_f32);
    assert_eq!(sharded_argmax(&[&widened]).unwrap(), 0);
    let nan = [f32_to_bf16(f32::NAN), f32_to_bf16(-f32::NAN)].map(bf16_to_f32);
    assert_eq!(sharded_argmax(&[&nan]).unwrap(), 0);
}
