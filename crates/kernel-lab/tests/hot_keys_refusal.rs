//! The one-hot oracle (`common/hot_keys.rs`) gives exact bits only for
//! finite operands whose dots F32 sums exactly and finitely in any order
//! and whose F32 scores lie apart by more than any exp survives, but not so
//! far that an F32 exponent argument overflows. These inputs break one of
//! those and must be refused, not given bits a kernel cannot produce.

#[path = "common/hot_keys.rs"]
mod hot_keys;

use hot_keys::{Partials, bf16_bytes, hot, key_row, partials, query_row, value_row};
use kernel_lab::LabResult;

const WIDTH: usize = 256;
const SCALE: f32 = 0.0625;
/// BF16 +inf and NaN.
const INF: u16 = 0x7f80;
const NAN: u16 = 0x7fc0;

/// A query head of `[2^30, 2^12, 0...]` as decoded values.
fn wide_query() -> Vec<f64> {
    let mut query = vec![0.0; WIDTH];
    query[0] = 2_f64.powi(30);
    query[1] = 2_f64.powi(12);
    query
}

/// The partials of one head over keys `rows` (key, value) of one page.
fn over(query: &[f64], rows: &[(Vec<u16>, Vec<u16>)]) -> LabResult<Partials> {
    let k = bf16_bytes(&rows.iter().flat_map(|(k, _)| k.clone()).collect::<Vec<_>>());
    let v = bf16_bytes(&rows.iter().flat_map(|(_, v)| v.clone()).collect::<Vec<_>>());
    let keys = (0..rows.len()).map(|slot| (0, slot)).collect::<Vec<_>>();
    partials(query, 1, WIDTH, (&k, &v), k.len(), &keys, SCALE)
}

/// The error of a refused input.
fn refusal(result: LabResult<Partials>, what: &str) -> String {
    match result {
        Ok(_) => panic!("{what}: accepted"),
        Err(e) => e,
    }
}

fn code(code: usize) -> (Vec<u16>, Vec<u16>) {
    (key_row(code, WIDTH), value_row(code, WIDTH))
}

fn decoded(row: &[u16]) -> Vec<f64> {
    hot_keys::bf16_values(&bf16_bytes(row))
}

/// The generated operands keep their exact one-hot partials.
#[test]
fn the_generated_operands_are_one_hot() {
    for negative in [false, true] {
        let query = decoded(&query_row(WIDTH, negative));
        let partial = over(&query, &[code(0), code(4095), code(2049)]).expect("one-hot");
        let (best, dot) = if negative {
            (0, -(2_f64.powi(18) + 2_f64.powi(12)))
        } else {
            (4095, 64.0 * 2_f64.powi(18) + 64.0 * 2_f64.powi(12))
        };
        assert_eq!(partial.max, [dot as f32 * SCALE]);
        assert_eq!(partial.den, [1.0]);
        assert_eq!(
            partial.num,
            decoded(&value_row(best, WIDTH))
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<_>>()
        );
    }
}

/// Codes 4094 and 4095 under `[2^30, 2^12, 0...]`: the exact dots 2^36 +
/// 63 * 2^12 and 2^36 + 2^18 lie 4096 apart, so the exact scores 256, but
/// F32 holds neither and rounds both to 2^36 + 2^18, so the kernel sees two
/// equal scores, two exp(0) weights and a denominator of 2.
#[test]
fn a_gap_that_f32_rounding_closes_is_refused() {
    let dots = [
        2_f64.powi(36) + 63.0 * 2_f64.powi(12),
        2_f64.powi(36) + 2_f64.powi(18),
    ];
    assert_eq!((dots[1] - dots[0]) * f64::from(SCALE), 256.0);
    assert_eq!(dots[0] as f32, dots[1] as f32);
    let refused = hot(&dots, f64::from(SCALE)).expect_err("rounded scores tie");
    assert!(refused.contains("not one-hot"), "{refused}");
    let refused = refusal(
        over(&wide_query(), &[code(4094), code(4095)]),
        "rounded tie",
    );
    assert!(refused.contains("exactly"), "{refused}");
}

/// A gap below the margin is refused and one at it accepted.
#[test]
fn a_gap_below_the_margin_is_refused() {
    let refused = hot(&[0.0, 127.0], 1.0).expect_err("gap of 127");
    assert!(refused.contains("not one-hot"), "{refused}");
    assert_eq!(hot(&[0.0, 128.0], 1.0), Ok(Some(1)));
    assert_eq!(hot(&[], 1.0), Ok(None));
}

/// Infinite or NaN scores, or a finite dot whose F32 score overflows, have
/// no exact one-hot softmax.
#[test]
fn nonfinite_scores_are_refused() {
    for dots in [
        [f64::INFINITY, 0.0],
        [f64::NAN, 0.0],
        [0.0, f64::NAN],
        [f64::NEG_INFINITY, 0.0],
        [1e39, 0.0],
    ] {
        let refused = hot(&dots, 1.0).expect_err("nonfinite score");
        assert!(refused.contains("finite"), "{dots:?}: {refused}");
    }
    let refused = hot(&[3e38, 0.0], 4.0).expect_err("score overflows F32");
    assert!(refused.contains("finite"), "{refused}");
}

/// A kernel multiplies every dim, so an infinite key element where the
/// query is 0 makes a NaN dot, and an infinite value of a key whose weight
/// is +0 a NaN numerator; a NaN query element makes every dot NaN.
#[test]
fn nonfinite_operands_are_refused() {
    let query = decoded(&query_row(WIDTH, false));
    let (mut key, value) = code(7);
    key[5] = INF;
    let refused = refusal(over(&query, &[code(4095), (key, value)]), "infinite key");
    assert!(refused.contains("finite"), "{refused}");
    let (key, mut value) = code(7);
    value[9] = INF;
    let refused = refusal(over(&query, &[code(4095), (key, value)]), "infinite value");
    assert!(refused.contains("finite"), "{refused}");
    let mut query = query;
    query[3] = decoded(&[NAN])[0];
    let refused = refusal(over(&query, &[code(4095), code(7)]), "NaN query");
    assert!(refused.contains("finite"), "{refused}");
}

/// `[2^30, 1, -2^30]` against ones sums to 1 exactly, but F32 gives 1 or 0
/// by the order of the sum, so the score has no exact bits.
#[test]
fn a_dot_f32_sums_by_order_is_refused() {
    let mut query = vec![0.0; WIDTH];
    query[..3].copy_from_slice(&[2_f64.powi(30), 1.0, -2_f64.powi(30)]);
    let one = 0x3f80;
    let refused = refusal(
        over(&query, &[(vec![one; WIDTH], vec![one; WIDTH])]),
        "order-dependent dot",
    );
    assert!(refused.contains("exactly"), "{refused}");
}

/// `x` as BF16, which it must hold exactly.
fn bf16(x: f64) -> u16 {
    let bits = ((x as f32).to_bits() >> 16) as u16;
    assert_eq!(decoded(&[bits]), [x], "{x} is not a BF16");
    bits
}

/// The F32 dot of the paged and selected partial kernels: each of 32 lanes
/// chains FMAs over its 8 dims, then an XOR butterfly sums the lanes.
fn kernel_dot(query: &[f64], key: &[f64]) -> f32 {
    let mut lanes = [0_f32; 32];
    for (d, (&q, &k)) in query.iter().zip(key).enumerate() {
        lanes[d / 8] = (q as f32).mul_add(k as f32, lanes[d / 8]);
    }
    for offset in [16, 8, 4, 2, 1] {
        let sums = lanes;
        for (lane, sum) in lanes.iter_mut().enumerate() {
            *sum = sums[lane] + sums[lane ^ offset];
        }
    }
    lanes[0]
}

/// The partials of `query` (dims shown, then zeros) over one key of ones
/// past the dims of `key` and a value row of ones, which must be refused.
fn overflow_refusal(query: &[f64], key: &[f64]) -> String {
    let mut q = vec![0.0; WIDTH];
    q[..query.len()].copy_from_slice(query);
    let mut k = vec![1.0; WIDTH];
    k[..key.len()].copy_from_slice(key);
    assert!(!kernel_dot(&q, &k).is_finite(), "the kernel dot is finite");
    let key = k.iter().map(|&x| bf16(x)).collect();
    refusal(over(&q, &[(key, vec![bf16(1.0); WIDTH])]), "F32 overflow")
}

/// `[2^100, -2^100]` against `[2^100, 2^100]` cancels to 0 exactly, but
/// each F32 product overflows, so the kernel's score is infinite.
#[test]
fn a_dot_whose_f32_products_overflow_is_refused() {
    let big = 2_f64.powi(100);
    let refused = overflow_refusal(&[big, -big], &[big, big]);
    assert!(refused.contains("overflows F32"), "{refused}");
}

/// `[2^127, 2^127, -2^127, -2^127]` against ones cancels to 0 exactly and
/// each product is a finite F32, but the second F32 sum overflows; in the
/// order `[2^127, -2^127, 2^127, -2^127]` F32 gives 0.
#[test]
fn a_dot_whose_f32_partial_sum_overflows_is_refused() {
    let big = 2_f64.powi(127);
    let alternating = [big, -big, big, -big]
        .iter()
        .fold(0_f32, |sum, &x| (x as f32).mul_add(1.0, sum));
    assert_eq!(alternating, 0.0);
    let refused = overflow_refusal(&[big, big, -big, -big], &[1.0; 4]);
    assert!(refused.contains("overflows F32"), "{refused}");
}

/// Products below the smallest F32 subnormal round in F32: `[2^-75, 2^-75]`
/// against itself is exactly 2^-149, but each F32 product of 2^-150 ties to
/// 0, so the kernel's dot is 0.
#[test]
fn a_dot_whose_f32_products_underflow_is_refused() {
    let tiny = 2_f64.powi(-75);
    let mut query = vec![0.0; WIDTH];
    query[..2].copy_from_slice(&[tiny, tiny]);
    let mut key = vec![1.0; WIDTH];
    key[..2].copy_from_slice(&[tiny, tiny]);
    assert_eq!(kernel_dot(&query, &key), 0.0);
    let key = key.iter().map(|&x| bf16(x)).collect();
    let refused = refusal(
        over(&query, &[(key, vec![bf16(1.0); WIDTH])]),
        "F32 underflow",
    );
    assert!(refused.contains("subnormal"), "{refused}");
}

/// Finite F32 scores 3e38 and -3e38 lie 6e38 apart, which F32 cannot hold:
/// a kernel's exponent argument `(s - m) * log2(e)` is -inf.
#[test]
fn an_exponent_argument_that_overflows_f32_is_refused() {
    assert_eq!(-3e38_f32 - 3e38_f32, f32::NEG_INFINITY);
    let refused = hot(&[3e38, -3e38], 1.0).expect_err("exponent argument overflows");
    assert!(refused.contains("overflows F32"), "{refused}");
}
