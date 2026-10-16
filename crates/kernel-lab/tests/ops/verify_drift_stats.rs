//! Row comparison helpers for the verify-versus-decode drift checks.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16, f64_to_bf16};
use serde_json::{Value, json};

/// Distance in representable steps between two values of one output type.
pub(crate) fn ulps(left: f32, right: f32, bf16: bool) -> u64 {
    let ordered = |value: f32| -> i64 {
        let (bits, sign) = if bf16 {
            (u64::from(f32_to_bf16(value)), 0x8000)
        } else {
            (u64::from(value.to_bits()), 0x8000_0000)
        };
        if bits & sign == 0 {
            bits as i64
        } else {
            -((bits & (sign - 1)) as i64)
        }
    };
    ordered(left).abs_diff(ordered(right))
}

/// Bitwise comparison of the capacity-1 and capacity-N rows.
pub(crate) fn paths(decode: &[f32], verify: &[f32], bf16: bool) -> Value {
    let mut differing = 0_usize;
    let mut max_abs = 0.0_f32;
    let mut max_ulps = 0_u64;
    for (&left, &right) in decode.iter().zip(verify) {
        if left.to_bits() != right.to_bits() {
            differing += 1;
            max_abs = max_abs.max((left - right).abs());
            max_ulps = max_ulps.max(ulps(left, right, bf16));
        }
    }
    json!({"elements": decode.len(), "differing": differing,
        "bitwise_equal": differing == 0, "max_abs": max_abs, "max_ulps": max_ulps})
}

/// The spacing of BF16 values in the binade of `x` (> 0).
pub(crate) fn bf16_step(x: f64) -> f64 {
    let x = x.max(f64::from(f32::MIN_POSITIVE));
    2.0_f64.powi(x.log2().floor() as i32 - 7)
}

/// Sums of one BF16 path's error against an f64 reference, in BF16 steps of
/// each element's scale: the reference's magnitude, but at least a 256th of
/// its summed absolute products, so outputs near cancellation do not
/// dominate. Row sums let a reader take standard errors over rows, whose
/// elements share their softmax. A step count is the distance in BF16
/// values from the reference rounded to BF16.
#[derive(Clone, Default)]
pub(crate) struct Precision {
    elements: u64,
    rows: u64,
    abs: f64,
    signed: f64,
    magnitude: f64,
    row_abs: f64,
    row_abs_squares: f64,
    row_magnitude: f64,
    row_magnitude_squares: f64,
    inexact: u64,
    above_one: u64,
    max_steps: u64,
}

impl Precision {
    /// Adds one row of outputs; returns each element's step count.
    pub(crate) fn row(&mut self, actual: &[f32], reference: &[f64], products: &[f64]) -> Vec<u64> {
        let mut abs = 0.0;
        let mut magnitude = 0.0;
        let steps = actual
            .iter()
            .zip(reference)
            .zip(products)
            .map(|((&value, &want), &sum)| {
                let unit = bf16_step(want.abs().max(sum / 256.0));
                let value = f64::from(value);
                let error = (value - want) / unit;
                let grown = (value.abs() - want.abs()) / unit;
                abs += error.abs();
                magnitude += grown;
                self.signed += error;
                let steps = ulps(value as f32, bf16_to_f32(f64_to_bf16(want)), true);
                self.inexact += u64::from(steps > 0);
                self.above_one += u64::from(steps > 1);
                self.max_steps = self.max_steps.max(steps);
                steps
            })
            .collect::<Vec<_>>();
        let n = actual.len().max(1) as f64;
        self.elements += actual.len() as u64;
        self.rows += 1;
        self.abs += abs;
        self.magnitude += magnitude;
        self.row_abs += abs / n;
        self.row_abs_squares += (abs / n).powi(2);
        self.row_magnitude += magnitude / n;
        self.row_magnitude_squares += (magnitude / n).powi(2);
        steps
    }

    /// The raw sums, which strata add, and the means they give.
    pub(crate) fn json(&self) -> Value {
        let n = self.elements.max(1) as f64;
        json!({"elements": self.elements, "rows": self.rows,
            "sum_abs": self.abs, "sum_signed": self.signed, "sum_magnitude": self.magnitude,
            "row_sum_abs": self.row_abs, "row_sum_abs_squares": self.row_abs_squares,
            "row_sum_magnitude": self.row_magnitude,
            "row_sum_magnitude_squares": self.row_magnitude_squares,
            "inexact": self.inexact, "above_one": self.above_one, "max_steps": self.max_steps,
            "mean_abs": self.abs / n, "mean_signed": self.signed / n,
            "magnitude_bias": self.magnitude / n,
            "inexact_fraction": self.inexact as f64 / n,
            "above_one_fraction": self.above_one as f64 / n})
    }
}

/// Error of one path against the f64 reference, with the reference rounded
/// to the output type for the step counts.
pub(crate) fn against(actual: &[f32], reference: &[f64], bf16: bool) -> Value {
    let mut sum = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut exact = 0_usize;
    let mut one = 0_usize;
    let mut more = 0_usize;
    for (&value, &want) in actual.iter().zip(reference) {
        let error = (f64::from(value) - want).abs();
        sum += error;
        max_abs = max_abs.max(error);
        let rounded = if bf16 {
            bf16_to_f32(f32_to_bf16(want as f32))
        } else {
            want as f32
        };
        match ulps(value, rounded, bf16) {
            0 => exact += 1,
            1 => one += 1,
            _ => more += 1,
        }
    }
    let n = actual.len().max(1) as f64;
    json!({"mean_abs": sum / n, "max_abs": max_abs,
        "exact_fraction": exact as f64 / n, "one_ulp_fraction": one as f64 / n,
        "over_one_ulp_fraction": more as f64 / n})
}
