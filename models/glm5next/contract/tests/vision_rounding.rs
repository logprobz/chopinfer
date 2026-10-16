//! The vision tower's q and k RMSNorm and rotary at the reference's F32
//! rounding points. `fixtures/vision-rounding.json` holds what the
//! unmodified transformers modules compute in float32
//! (`fixtures/vision_rounding.py`). The F32-boundary reference reproduces
//! it bit for bit; the exact f64 oracle, which the decomposition checks use, differs
//! from it by F32 rounding only.
use kernel_core::encoder_oracle::{qk_rms_norm, rms_rotary_f32_boundaries};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/vision-rounding.json")).unwrap()
}

fn floats(fixture: &Value, key: &str) -> Vec<f32> {
    fixture[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|bits| f32::from_bits(bits.as_i64().unwrap() as u32))
        .collect()
}

/// The largest distance in F32 units in the last place of each head's
/// largest magnitude (`head_dim` values per head): a rotation can cancel
/// to near zero, so single values are no scale.
fn head_ulps(actual: &[f32], expected: &[f32], head_dim: usize) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .chunks_exact(head_dim)
        .zip(expected.chunks_exact(head_dim))
        .map(|(actual, expected)| {
            let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let ulp = f64::from(scale) * f64::from(f32::EPSILON);
            actual
                .iter()
                .zip(expected)
                .map(|(&a, &e)| (f64::from(a) - f64::from(e)).abs() / ulp)
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

#[test]
fn the_f32_boundaries_reproduce_the_reference_run() {
    let fixture = fixture();
    let head_dim = fixture["head_dim"].as_u64().unwrap() as usize;
    let eps = fixture["eps"].as_f64().unwrap() as f32;
    let (cos, sin) = (floats(&fixture, "cos"), floats(&fixture, "sin"));
    for (input, weight, output) in [("q", "q_weight", "q_out"), ("k", "k_weight", "k_out")] {
        let reference = floats(&fixture, output);
        let ours = rms_rotary_f32_boundaries(
            &floats(&fixture, input),
            head_dim,
            &floats(&fixture, weight),
            eps,
            &cos,
            &sin,
        )
        .unwrap();
        let same = ours.iter().zip(&reference).filter(|(a, b)| a == b).count();
        assert_eq!(same, reference.len(), "{input}: values differ");
    }
}

/// The exact oracle (normalization and rotation in f64, then rounded once)
/// stays within two F32 units (of each head's largest value) of the
/// reference run but differs in about half the values: the decomposition
/// checks against it say nothing about F32 rounding parity.
#[test]
fn the_exact_oracle_differs_from_the_reference_run_by_f32_rounding() {
    let fixture = fixture();
    let heads = fixture["heads"].as_u64().unwrap() as usize;
    let head_dim = fixture["head_dim"].as_u64().unwrap() as usize;
    let eps = fixture["eps"].as_f64().unwrap();
    let wide =
        |key: &str| -> Vec<f64> { floats(&fixture, key).into_iter().map(f64::from).collect() };
    let (q, k) = (wide("q"), wide("k"));
    let rows = q.len() / (heads * head_dim);
    // Fused rows `[q | k | v]` with zero v.
    let mut qkv = Vec::with_capacity(rows * 3 * heads * head_dim);
    for row in 0..rows {
        let at = row * heads * head_dim;
        qkv.extend_from_slice(&q[at..at + heads * head_dim]);
        qkv.extend_from_slice(&k[at..at + heads * head_dim]);
        qkv.extend(std::iter::repeat_n(0.0, heads * head_dim));
    }
    let normed = qk_rms_norm(
        &qkv,
        heads,
        heads,
        head_dim,
        &wide("q_weight"),
        &wide("k_weight"),
        eps,
    )
    .unwrap()
    .values;
    let (cos, sin) = (wide("cos"), wide("sin"));
    let half = head_dim / 2;
    let mut worst = 0.0f64;
    let mut differ = 0;
    for (plane, output) in [(0, "q_out"), (1, "k_out")] {
        let reference = floats(&fixture, output);
        let mut exact = Vec::with_capacity(reference.len());
        for row in 0..rows {
            for head in 0..heads {
                let at = row * 3 * heads * head_dim + (plane * heads + head) * head_dim;
                let x = &normed[at..at + head_dim];
                for d in 0..head_dim {
                    let rotated = if d < half { -x[d + half] } else { x[d - half] };
                    let angle = row * head_dim + d;
                    exact.push((x[d] * cos[angle] + rotated * sin[angle]) as f32);
                }
            }
        }
        worst = worst.max(head_ulps(&exact, &reference, head_dim));
        differ += exact.iter().zip(&reference).filter(|(a, b)| a != b).count();
    }
    assert!(worst <= 2.0, "{worst} ulps");
    assert!(differ > 0, "the exact oracle matched the F32 run bitwise");
    eprintln!("exact oracle: {differ} values differ, at most {worst} ulps");
}
