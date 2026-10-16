//! CPU model of the one row-count-dependent reduction in the fused
//! all-reduce, residual and RMSNorm kernel (`cuda/comm/push.cuh`). Rows up
//! to the low-latency limit (2) sum each row's squares over a two-CTA
//! cluster with one 8-element vector per thread; the one-shot protocol
//! (3 to 16 rows) sums them in one CTA with two vectors per thread. The
//! per-element all-reduce sum and residual add are identical in both, so
//! only the RMS scale can differ. This replays both orders in f32 and
//! counts how often the normalized BF16 output changes, so a decode row
//! (one token, low latency) can be compared with a verify row (bucket 4 or
//! 8, one-shot). The GPU op checks live in `tests/ops/*_verify_drift.rs`.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};

const WIDTH: usize = 4096;
const THREADS: usize = 256;
const EPS: f32 = 1e-6;

fn round(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(value))
}

/// `square_sum += low * low + high * high` per BF16 pair, with the pair
/// sum contracted to one FMA as nvcc does by default.
fn vector_squares(row: &[f32], vector: usize, mut sum: f32) -> f32 {
    for pair in row[vector * 8..vector * 8 + 8].chunks_exact(2) {
        sum += pair[0].mul_add(pair[0], pair[1] * pair[1]);
    }
    sum
}

/// Lane 0 of a warp after `value += shfl_xor(value, offset)` for offsets
/// 16 down to 1.
fn butterfly(mut lanes: [f32; 32]) -> f32 {
    let mut offset = 16;
    while offset > 0 {
        let previous = lanes;
        for (lane, value) in lanes.iter_mut().enumerate() {
            *value = previous[lane] + previous[lane ^ offset];
        }
        offset >>= 1;
    }
    lanes[0]
}

/// `row_sum`: warp butterflies, then the warp sums in one butterfly.
fn cta_sum(per_thread: &[f32]) -> f32 {
    let mut warps = [0.0_f32; 32];
    for (warp, sum) in warps.iter_mut().take(THREADS / 32).enumerate() {
        let mut lanes = [0.0_f32; 32];
        lanes.copy_from_slice(&per_thread[warp * 32..warp * 32 + 32]);
        *sum = butterfly(lanes);
    }
    butterfly(warps)
}

/// Low latency: CTA `part` of the two-CTA cluster owns vectors
/// `part * 256 + thread`; the cluster adds the CTA sums in order.
fn low_latency_squares(row: &[f32]) -> f32 {
    let mut total = 0.0_f32;
    for part in 0..2 {
        let per_thread = (0..THREADS)
            .map(|thread| vector_squares(row, part * THREADS + thread, 0.0))
            .collect::<Vec<_>>();
        total += cta_sum(&per_thread);
    }
    total
}

/// One-shot: one CTA; thread `t` owns vectors `t` and `256 + t`.
fn one_shot_squares(row: &[f32]) -> f32 {
    let per_thread = (0..THREADS)
        .map(|thread| {
            let first = vector_squares(row, thread, 0.0);
            vector_squares(row, THREADS + thread, first)
        })
        .collect::<Vec<_>>();
    cta_sum(&per_thread)
}

/// The phased gather's one-warp norm (`norm_row_warp`): lane `l` holds
/// vectors `l + 32 i`; each vector sums from zero, each group of 32 vectors
/// by the warp butterfly, the eight group sums of a low-latency CTA by the
/// explicit tree `((w0 + w4) + (w2 + w6)) + ((w1 + w5) + (w3 + w7))`, and
/// the two CTA sums from zero in order.
fn phased_warp_squares(row: &[f32]) -> f32 {
    let mut groups = [0.0_f32; 16];
    for (i, group) in groups.iter_mut().enumerate() {
        let mut lanes = [0.0_f32; 32];
        for (lane, value) in lanes.iter_mut().enumerate() {
            *value = vector_squares(row, lane + 32 * i, 0.0);
        }
        *group = butterfly(lanes);
    }
    let mut total = 0.0_f32;
    for w in groups.chunks_exact(8) {
        total += ((w[0] + w[4]) + (w[2] + w[6])) + ((w[1] + w[5]) + (w[3] + w[7]));
    }
    total
}

fn inverse_rms(squares: f32) -> f32 {
    1.0 / (squares / WIDTH as f32 + EPS).sqrt()
}

fn normalize(row: &[f32], weight: &[f32], squares: f32) -> Vec<f32> {
    let inv = inverse_rms(squares);
    row.iter()
        .zip(weight)
        .map(|(&value, &scale)| round(round(value * inv) * scale))
        .collect()
}

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1_u64 << 24) as f32
    }

    /// Approximately normal: centered sum of four uniforms, unit variance.
    fn normal(&mut self) -> f32 {
        ((0..4).map(|_| self.uniform()).sum::<f32>() - 2.0) * 3.0_f32.sqrt()
    }
}

#[test]
fn hidden_reduce_rms_order_low_latency_versus_one_shot() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let weight = (0..WIDTH)
        .map(|_| round(1.0 + 0.3 * rng.normal()))
        .collect::<Vec<_>>();
    let rows = 2000;
    let (mut sum_differs, mut scale_differs, mut elements_differ) = (0, 0, 0);
    let (mut exact_low, mut exact_one) = (0, 0);
    for index in 0..rows {
        // Hidden states with a spread of row scales and a few outliers.
        let scale = (-((index % 9) as f32)).exp2() * 4.0;
        let row = (0..WIDTH)
            .map(|column| {
                let outlier = if column % 509 == 0 { 30.0 } else { 1.0 };
                round(rng.normal() * scale * outlier)
            })
            .collect::<Vec<_>>();
        let low = low_latency_squares(&row);
        let one = one_shot_squares(&row);
        let exact = row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        exact_low += usize::from(f64::from(low) == f64::from(exact as f32));
        exact_one += usize::from(f64::from(one) == f64::from(exact as f32));
        sum_differs += usize::from(low.to_bits() != one.to_bits());
        if inverse_rms(low).to_bits() != inverse_rms(one).to_bits() {
            scale_differs += 1;
            let (a, b) = (normalize(&row, &weight, low), normalize(&row, &weight, one));
            elements_differ += a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
        }
    }
    eprintln!(
        "{}",
        serde_json::json!({
            "rows": rows,
            "rows_with_different_square_sum": sum_differs,
            "rows_with_different_rms_scale": scale_differs,
            "differing_bf16_outputs": elements_differ,
            "differing_output_fraction": elements_differ as f64 / (rows * WIDTH) as f64,
            "low_latency_square_sum_correctly_rounded_rows": exact_low,
            "one_shot_square_sum_correctly_rounded_rows": exact_one,
        })
    );
}

/// Prefill rows normalize in the low-latency order whatever protocol their
/// step's row count selects: the phased gather's one-warp tree must give
/// the low-latency square sum bit for bit.
#[test]
fn phased_warp_norm_sums_squares_in_low_latency_order() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for index in 0..2000 {
        let scale = (-((index % 9) as f32)).exp2() * 4.0;
        let row = (0..WIDTH)
            .map(|column| {
                let outlier = if column % 509 == 0 { 30.0 } else { 1.0 };
                round(rng.normal() * scale * outlier)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            phased_warp_squares(&row).to_bits(),
            low_latency_squares(&row).to_bits(),
            "row {index}"
        );
    }
}
