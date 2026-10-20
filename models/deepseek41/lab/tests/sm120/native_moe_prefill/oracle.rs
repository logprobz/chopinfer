//! The f64 oracle over exact decoded operands, the per-element bound of
//! docs/numerics.md and the report-only error statistics.
use kernel_core::codec::{SmallFloat, decode_ue8m0_f64};

pub fn e4m3_table() -> [f64; 256] {
    std::array::from_fn(|c| SmallFloat::E4M3.decode_f64(c as u8))
}
pub fn e2m1_table() -> [f64; 16] {
    std::array::from_fn(|c| SmallFloat::E2M1.decode_f64(c as u8))
}

/// A decoded activation row: each E4M3 code times its group's UE8M0 scale.
pub fn activation(codes: &[u8], scales: &[u8], e4m3: &[f64; 256]) -> Vec<f64> {
    codes
        .iter()
        .enumerate()
        .map(|(k, &c)| e4m3[c as usize] * decode_ue8m0_f64(scales[k / 32]))
        .collect()
}

/// The exact dot product of `x` with one packed MXFP4 weight row (element
/// 2j in the low nibble of byte j), and its sum of absolute products.
/// Products are exact in F64; the F64 sum's rounding is far below the bound.
pub fn dot(x: &[f64], packed: &[u8], scales: &[u8], e2m1: &[f64; 16]) -> (f64, f64) {
    let mut value = 0.0;
    let mut absolute = 0.0;
    for (g, group) in x.chunks(32).enumerate() {
        let scale = decode_ue8m0_f64(scales[g]);
        let mut v = 0.0;
        let mut s = 0.0;
        for (i, &a) in group.iter().enumerate() {
            let k = g * 32 + i;
            let w = e2m1[((packed[k / 2] >> (4 * (k & 1))) & 15) as usize];
            v += a * w;
            s += (a * w).abs();
        }
        value += v * scale;
        absolute += s * scale;
    }
    (value, absolute)
}

/// `2^-22 * K * S`, rounded up: the F32 output bound with no rounding allowance.
pub fn bound(k: usize, s: f64) -> f64 {
    let b = (k as f64) * s * f64::powi(2.0, -22);
    b * (1.0 + f64::powi(2.0, -50)) + f64::from_bits(1)
}

/// One output row's comparison against its oracle row.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowCheck {
    pub elements: u64,
    pub failures: u64,
    /// Largest |y - y_ref| / bound.
    pub worst_ratio: f64,
    pub squared_error: f64,
    pub max_abs: f64,
    pub cosine: f64,
}

pub fn check_row(y: &[f32], reference: &[(f64, f64)], k: usize) -> RowCheck {
    assert_eq!(y.len(), reference.len());
    let mut c = RowCheck {
        elements: y.len() as u64,
        ..RowCheck::default()
    };
    let (mut dot, mut yy, mut rr) = (0.0, 0.0, 0.0);
    for (&v, &(r, s)) in y.iter().zip(reference) {
        let v = f64::from(v);
        let error = if v.is_finite() {
            (v - r).abs()
        } else {
            f64::INFINITY
        };
        let b = bound(k, s);
        if !(error <= b) {
            c.failures += 1;
        }
        c.worst_ratio = c.worst_ratio.max(error / b);
        c.squared_error += error * error;
        c.max_abs = c.max_abs.max(error);
        dot += v * r;
        yy += v * v;
        rr += r * r;
    }
    c.cosine = if yy == 0.0 && rr == 0.0 {
        1.0
    } else {
        dot / (yy.sqrt() * rr.sqrt())
    };
    c
}

/// Accumulated checks of one stage over many rows.
#[derive(Clone, Copy, Debug)]
pub struct Stage {
    pub rows: u64,
    pub elements: u64,
    pub failures: u64,
    pub worst_ratio: f64,
    pub rmse: f64,
    pub max_abs: f64,
    pub min_cosine: f64,
    pub cosine_failures: u64,
    squared: f64,
}
impl Default for Stage {
    fn default() -> Self {
        Self {
            rows: 0,
            elements: 0,
            failures: 0,
            worst_ratio: 0.0,
            rmse: 0.0,
            max_abs: 0.0,
            min_cosine: 1.0,
            cosine_failures: 0,
            squared: 0.0,
        }
    }
}
impl Stage {
    pub fn add(&mut self, c: RowCheck) {
        self.rows += 1;
        self.elements += c.elements;
        self.failures += c.failures;
        self.worst_ratio = self.worst_ratio.max(c.worst_ratio);
        self.squared += c.squared_error;
        self.rmse = (self.squared / self.elements.max(1) as f64).sqrt();
        self.max_abs = self.max_abs.max(c.max_abs);
        self.min_cosine = self.min_cosine.min(c.cosine);
        self.cosine_failures += u64::from(!(c.cosine >= 0.9999));
    }
    pub fn passes(&self) -> bool {
        self.rows > 0 && self.failures == 0 && self.cosine_failures == 0
    }
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "rows": self.rows,
            "elements": self.elements,
            "failures": self.failures,
            "worst_ratio": self.worst_ratio,
            "rmse": self.rmse,
            "max_abs": self.max_abs,
            "min_cosine": self.min_cosine,
            "cosine_failures": self.cosine_failures,
        })
    }
}

/// Report-only comparison of native and exact errors on the same oracle.
pub fn versus(native: &Stage, exact: &Stage) -> serde_json::Value {
    serde_json::json!({
        "native": native.json(),
        "exact": exact.json(),
        "rmse_ratio": native.rmse / exact.rmse,
        "max_abs_ratio": native.max_abs / exact.max_abs,
        "no_larger_error": native.rmse <= exact.rmse && native.max_abs <= exact.max_abs,
        "gate": false,
    })
}

/// The serving routed sum's arithmetic for one token: F32 adds of the
/// routes in order from +0, then the shared addend.
pub fn ordered_sum(routes: &[&[f32]], shared: &[f32]) -> Vec<f32> {
    (0..shared.len())
        .map(|c| {
            let mut v = 0.0_f32;
            for r in routes {
                v += r[c];
            }
            v + shared[c]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_a_naive_decode() {
        let e4m3 = e4m3_table();
        let e2m1 = e2m1_table();
        let mut seed = 0x9e37_79b9_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let k = 576;
        let codes: Vec<u8> = (0..k).map(|_| (next() as u8) & 0x7e).collect();
        let a_scales: Vec<u8> = (0..k / 32).map(|_| 110 + (next() % 20) as u8).collect();
        let packed: Vec<u8> = (0..k / 2).map(|_| next() as u8).collect();
        let w_scales: Vec<u8> = (0..k / 32).map(|_| 115 + (next() % 10) as u8).collect();
        let x = activation(&codes, &a_scales, &e4m3);
        let (v, s) = dot(&x, &packed, &w_scales, &e2m1);
        let mut naive = 0.0;
        let mut naive_abs = 0.0;
        for i in 0..k {
            let a = SmallFloat::E4M3.decode_f64(codes[i]) * decode_ue8m0_f64(a_scales[i / 32]);
            let nibble = if i % 2 == 0 {
                packed[i / 2] & 15
            } else {
                packed[i / 2] >> 4
            };
            let w = SmallFloat::E2M1.decode_f64(nibble) * decode_ue8m0_f64(w_scales[i / 32]);
            naive += a * w;
            naive_abs += (a * w).abs();
        }
        assert!((v - naive).abs() <= 1e-12 * naive_abs);
        assert!((s - naive_abs).abs() <= 1e-12 * naive_abs);
    }

    #[test]
    fn bound_and_cosine_reject_a_dropped_chain() {
        let reference: Vec<(f64, f64)> = (0..64).map(|i| (i as f64 + 0.5, 100.0)).collect();
        let exact: Vec<f32> = reference.iter().map(|&(r, _)| r as f32).collect();
        let c = check_row(&exact, &reference, 5120);
        assert_eq!((c.failures, c.elements), (0, 64));
        assert!(c.cosine > 0.999_999);
        // Dropping a 128-K chain of S/40 per element breaks the bound.
        let dropped: Vec<f32> = exact.iter().map(|v| v - 2.5).collect();
        assert_eq!(check_row(&dropped, &reference, 5120).failures, 64);
        let mut stage = Stage::default();
        stage.add(c);
        assert!(stage.passes());
        stage.add(check_row(&dropped, &reference, 5120));
        assert!(!stage.passes());
        // A NaN never passes.
        let mut nan = exact.clone();
        nan[3] = f32::NAN;
        assert_eq!(check_row(&nan, &reference, 5120).failures, 1);
    }

    #[test]
    fn ordered_sum_adds_routes_in_order_then_the_addend() {
        let a = [1.0e8_f32, 1.0];
        let b = [-1.0e8_f32, 2.0];
        let c = [1.0_f32, 3.0];
        let shared = [0.5_f32, 0.25];
        let s = ordered_sum(&[&a, &b, &c], &shared);
        assert_eq!(s, vec![((1.0e8_f32 + -1.0e8) + 1.0) + 0.5, 6.25]);
    }
}
