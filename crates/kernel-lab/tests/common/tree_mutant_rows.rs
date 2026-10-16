//! Hidden-width rows on which the 256-lane tree norm's reversed-distances
//! mutant (`CHOP_RMSNORM_TREE_TEST_DISTANCES_REVERSED`: each warp adds its
//! lanes at distances 1, 2, 4, 8, 16 instead of 16, 8, 4, 2, 1) must give
//! other normed bits, proven on the CPU emulation before any launch. Random
//! rows do not separate the two orders: their sums differ by an ulp or
//! less, which the BF16 outputs absorb.
//!
//! - The overflow row: lane 0's squares sum exactly to `f32::MAX`, and each
//!   of lanes 16 to 31 holds one square `s` of 1.125 x 2^99, with u =
//!   2^104 the ulp of `f32::MAX`, so 8 s < u / 2 < 16 s. The served order
//!   adds them to lane 0 in groups of 1, 1, 2, 4 and 8 lanes (lanes 16;
//!   8 and 24; ...; the odd lanes), each below half an ulp, and the sum
//!   stays `f32::MAX`. The reversed order adds lanes 16 to 31 together
//!   first and then 16 s to lane 0, which rounds to +Inf: the reciprocal
//!   is zero and every nonzero output becomes zero.
//! - The probe row: lane 0 holds one large square and lanes 16 to 31 small
//!   ones, so the two orders' sums differ by an ulp, and 128 columns hold
//!   one value of each BF16 mantissa; at these pinned magnitudes one of
//!   them rounds to other BF16 bits under the two reciprocals. A finite
//!   row, so the catch does not rest on overflow alone.
//!
//! Every value is a finite, positive or +0 BF16 value; the norm weight is
//! one. The served order's emulation is `rmsnorm_tree::tree_sum`, whose
//! lane partials this module repeats for the reversed order.
#![allow(dead_code)] // Shared by test binaries that use different parts.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernels_sm120::row::rmsnorm_tree::{LANES, SPAN, hf_norm, tree_sum};

/// The rows' width, the hidden width.
pub const WIDTH: usize = 4096;
const WARP: usize = 32;

/// The BF16 value `m` x 2^`e`, which must be exact.
fn value(m: u32, e: i32) -> u16 {
    let v = m as f32 * 2f32.powi(e);
    let bits = f32_to_bf16(v);
    assert_eq!(bf16_to_f32(bits), v, "{m} x 2^{e} is not a BF16 value");
    bits
}

/// The tree's sum with each warp's lane distances `distances` (the eight
/// warp sums still at 4, 2, 1): lane `l` adds columns `8 l + 2048 i + j`
/// from +0, then lane m adds lane m + d at each distance, lane 0 ending
/// with the warp's sum.
pub fn tree_sum_with(squares: &[f32], distances: [usize; 5]) -> f32 {
    let mut lanes = [0.0_f32; LANES];
    for (lane, partial) in lanes.iter_mut().enumerate() {
        let mut base = 0;
        while base < squares.len() {
            for j in 0..8 {
                if let Some(&square) = squares.get(base + 8 * lane + j) {
                    *partial += square;
                }
            }
            base += SPAN;
        }
    }
    let mut warps = [0.0_f32; LANES / WARP];
    for (warp, total) in warps.iter_mut().enumerate() {
        let values = &mut lanes[warp * WARP..(warp + 1) * WARP];
        // Lane m reads lane m + d's value from before the level; lane 0's
        // sum depends only on the lanes the shrinking stride visits.
        for d in distances {
            let stride = if distances[0] == 1 { 2 * d } else { 1 };
            let end = if distances[0] == 1 { WARP } else { d };
            for m in (0..end).step_by(stride) {
                values[m] += values[m + d];
            }
        }
        *total = values[0];
    }
    for d in [4, 2, 1] {
        for m in 0..d {
            warps[m] += warps[m + d];
        }
    }
    warps[0]
}

/// The reversed-distances mutant's sum.
pub fn reversed_tree_sum(squares: &[f32]) -> f32 {
    tree_sum_with(squares, [1, 2, 4, 8, 16])
}

/// The overflow row (see the module documentation).
pub fn overflow_row() -> Vec<u16> {
    let mut x = vec![0_u16; WIDTH];
    // 2^24 - 1 = 255^2 4^4 + 180^2 4 + 34^2 + 7^2 + 3^2 + 1^2, in units of
    // 2^104: every partial sum is an integer below 2^24 units, so exact.
    for (column, (m, k)) in [(255, 4), (180, 1), (34, 0), (7, 0), (3, 0), (1, 0)]
        .into_iter()
        .enumerate()
    {
        x[column] = value(m, k + 52);
    }
    for lane in 16..WARP {
        x[8 * lane] = value(192, 42);
    }
    x
}

/// The probe row (see the module documentation).
pub fn probe_row() -> Vec<u16> {
    let mut x = vec![0_u16; WIDTH];
    x[0] = value(152, 0);
    for lane in 16..WARP {
        x[8 * lane] = value(181, -14);
    }
    for (i, column) in (256..256 + 128).enumerate() {
        x[column] = value(128 + i as u32, -6);
    }
    x
}

/// `rows` rows alternating the overflow row (even rows) and the probe row.
pub fn rows(rows: usize) -> Vec<u16> {
    let (overflow, probe) = (overflow_row(), probe_row());
    (0..rows)
        .flat_map(|row| {
            if row % 2 == 0 {
                overflow.clone()
            } else {
                probe.clone()
            }
        })
        .collect()
}

/// The norm weight: one in every column.
pub fn weight() -> Vec<u16> {
    vec![f32_to_bf16(1.0); WIDTH]
}

/// The served tree norm (`tree`) and the mutant's (`reversed`) of each row
/// of `x` at `eps`, with [`weight`].
pub fn norms(x: &[u16], eps: f32) -> (Vec<u16>, Vec<u16>) {
    let weight = weight();
    let mut tree = Vec::with_capacity(x.len());
    let mut reversed = Vec::with_capacity(x.len());
    for row in x.chunks_exact(WIDTH) {
        tree.extend(hf_norm(row, &weight, eps, tree_sum));
        reversed.extend(hf_norm(row, &weight, eps, reversed_tree_sum));
    }
    (tree, reversed)
}

fn squares(row: &[u16]) -> Vec<f32> {
    row.iter()
        .map(|&v| {
            let v = bf16_to_f32(v);
            v * v
        })
        .collect()
}

/// The CPU proof that the mutant changes every row's normed bits at `eps`:
/// this module's served order is `tree_sum`, the overflow row sums to
/// `f32::MAX` served and +Inf reversed, and both rows' norms differ.
/// Returns the outputs that differ per row kind (overflow, probe).
pub fn proof(eps: f32) -> Result<[usize; 2], String> {
    let mut changed = [0; 2];
    for (kind, row) in [overflow_row(), probe_row()].iter().enumerate() {
        if row.iter().any(|&v| v & 0x8000 != 0 || v & 0x7f80 == 0x7f80) {
            return Err(format!("row kind {kind}: a negative or non-finite value"));
        }
        let squares = squares(row);
        let served = tree_sum_with(&squares, [16, 8, 4, 2, 1]);
        if served.to_bits() != tree_sum(&squares).to_bits() {
            return Err(format!(
                "row kind {kind}: the repeated served order {served} is not tree_sum's"
            ));
        }
        if kind == 0 && (served != f32::MAX || reversed_tree_sum(&squares) != f32::INFINITY) {
            return Err(format!(
                "overflow row: sums {served} and {}, not f32::MAX and +Inf",
                reversed_tree_sum(&squares)
            ));
        }
        let (tree, reversed) = norms(row, eps);
        changed[kind] = tree.iter().zip(&reversed).filter(|(a, b)| a != b).count();
        if changed[kind] == 0 {
            return Err(format!(
                "row kind {kind}: the mutant's norm equals the tree's"
            ));
        }
    }
    Ok(changed)
}
