//! Collective inputs that expose reordered sums and previous-call payloads.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Each group of twice the rank count's columns contains one marker per
/// source rank, present in that source only. At two and four ranks, every
/// eight-element wire vector has every source's marker. The marker changes
/// by two BF16 codes per logical call.
/// Other columns cancel a large pair around smaller terms, so F32 rank
/// order remains visible after BF16 rounding. Two ranks have no ordering
/// ambiguity, but retain the per-source stale-payload witnesses.
pub fn inputs(ranks: usize, rows: usize, width: usize, call: u64) -> Vec<Vec<u16>> {
    assert!(matches!(ranks, 2 | 4 | 8) && width.is_multiple_of(2 * ranks) && call < 1024);
    let mut inputs = vec![vec![0; rows * width]; ranks];
    for index in 0..rows * width {
        let lane = index % (2 * ranks);
        if lane < ranks {
            inputs[lane][index] = 0x3f00 + 2 * call as u16;
            continue;
        }
        let key = mix(index as u64 ^ call.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        for (rank, input) in inputs.iter_mut().enumerate() {
            let bits = mix(key ^ (rank as u64 + 1).wrapping_mul(0x1_0001));
            input[index] = f32_to_bf16(((bits % 257) as f32 - 128.0) / 64.0);
        }
        if ranks >= 4 {
            let first = key as usize % ranks;
            let second = (first + 1 + (key >> 8) as usize % (ranks - 1)) % ranks;
            let big = if key & (1 << 24) == 0 {
                16_777_216.0
            } else {
                -16_777_216.0
            };
            inputs[first][index] = f32_to_bf16(big);
            inputs[second][index] = f32_to_bf16(-big);
        }
    }
    inputs
}

/// Keep the payload markers observable after residual addition as well.
/// Other columns retain the fixture's varied residual operands.
pub fn clear_marker_residuals(residual: &mut [u16], ranks: usize) {
    for (index, value) in residual.iter_mut().enumerate() {
        if index % (2 * ranks) < ranks {
            *value = 0;
        }
    }
}

pub fn bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn rounded_sums(inputs: &[Vec<u16>], order: &[usize]) -> Vec<u16> {
    (0..inputs[0].len())
        .map(|i| {
            f32_to_bf16(
                order
                    .iter()
                    .fold(0.0_f32, |sum, &rank| sum + bf16_to_f32(inputs[rank][i])),
            )
        })
        .collect()
}

/// Reject fixtures whose BF16 sums can hide a reversed or rotated rank
/// order in any row. Test this after the output rounding, not just in F32.
pub fn require_order_sensitive(inputs: &[Vec<u16>], width: usize) -> Result<(), String> {
    let ranks = inputs.len();
    if ranks < 4 {
        return Ok(());
    }
    let forward: Vec<_> = (0..ranks).collect();
    let expected = rounded_sums(inputs, &forward);
    let orders = std::iter::once((0..ranks).rev().collect::<Vec<_>>())
        .chain((1..ranks).map(|offset| (0..ranks).map(|r| (r + offset) % ranks).collect()));
    for order in orders {
        let actual = rounded_sums(inputs, &order);
        for (row, (a, b)) in expected
            .chunks_exact(width)
            .zip(actual.chunks_exact(width))
            .enumerate()
        {
            let changed = a.iter().zip(b).filter(|(x, y)| x != y).count();
            if changed * 64 < width {
                return Err(format!(
                    "row {row}, order {order:?}: only {changed}/{width} BF16 sums differ"
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn cancellation_rejects_deliberately_wrong_rank_order() {
    for ranks in [4, 8] {
        for call in [0, 1, 17, 191, 511] {
            let values = inputs(ranks, 2, 2560, call);
            require_order_sensitive(&values, 2560).unwrap();
            let correct = rounded_sums(&values, &(0..ranks).collect::<Vec<_>>());
            let wrong = rounded_sums(&values, &(0..ranks).rev().collect::<Vec<_>>());
            assert_ne!(
                correct, wrong,
                "a reversed reduction must fail bitwise equality"
            );
        }
    }
    let witness: Vec<_> = [16_777_216.0, 1.0, -16_777_216.0, 0.0]
        .into_iter()
        .map(|v| vec![f32_to_bf16(v)])
        .collect();
    assert_eq!(rounded_sums(&witness, &[0, 1, 2, 3]), [f32_to_bf16(0.0)]);
    assert_eq!(rounded_sums(&witness, &[0, 2, 1, 3]), [f32_to_bf16(1.0)]);
}

#[test]
fn previous_peer_payload_fails_in_every_row_and_quarter() {
    for ranks in [2, 4, 8] {
        let old = inputs(ranks, 2, 2560, 0);
        let current = inputs(ranks, 2, 2560, 1);
        let order: Vec<_> = (0..ranks).collect();
        let expected = rounded_sums(&current, &order);
        for rank in 0..ranks {
            let mut stale = current.clone();
            stale[rank] = old[rank].clone();
            let wrong = rounded_sums(&stale, &order);
            for row in 0..2 {
                for quarter in 0..4 {
                    let marker = row * 2560 + quarter * 640 + rank;
                    assert_ne!(
                        expected[marker], wrong[marker],
                        "rank {rank}, row {row}, quarter {quarter}"
                    );
                }
            }
        }
    }
}
