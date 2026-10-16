//! The FP8 one-hot oracle (`common/hot_fp8.rs`) gives exact bits only for
//! E4M3 codes and scales whose dots F32 sums exactly in any order, whose
//! F32 scores lie at least the one-hot margin apart, and whose numerators
//! F32 holds exactly. The generated operands meet this; these inputs break
//! one part and must be refused, not given bits a kernel cannot produce.

#[path = "common/hot_keys.rs"]
mod hot_keys;

#[path = "common/hot_fp8.rs"]
mod hot_fp8;

use hot_fp8::{
    BLOCK, K_SCALE, V_SCALE, e4m3, fp8_values, key_row, latent_partials, latent_pool,
    latent_scales, latent_token, partials, query_row, value_row,
};
use hot_keys::{CODES, DECOY, MARGIN, Partials, bf16_bytes, bf16_values, dot_values};
use kernel_lab::LabResult;

const WIDTH: usize = 256;
const LATENT: usize = 512;
const PAGE: usize = 64;
const SCALE: f32 = 0.0625;

fn decoded(row: &[u16]) -> Vec<f64> {
    bf16_values(&bf16_bytes(row))
}

/// The partials of one head over keys `rows` (key, value) of one page at K
/// and V scales `scales`.
fn over(query: &[f64], rows: &[(Vec<u8>, Vec<u8>)], scales: (f32, f32)) -> LabResult<Partials> {
    let k = rows.iter().flat_map(|(k, _)| k.clone()).collect::<Vec<_>>();
    let v = rows.iter().flat_map(|(_, v)| v.clone()).collect::<Vec<_>>();
    let keys = (0..rows.len()).map(|slot| (0, slot)).collect::<Vec<_>>();
    partials(query, 1, WIDTH, (&k, &v), k.len(), &keys, SCALE, scales)
}

fn code(code: usize) -> (Vec<u8>, Vec<u8>) {
    (key_row(code, WIDTH), value_row(code, WIDTH))
}

/// The error of a refused input.
fn refusal(result: LabResult<Partials>, what: &str) -> String {
    match result {
        Ok(_) => panic!("{what}: accepted"),
        Err(e) => e,
    }
}

/// The dot of code `c` under a positive query head.
fn code_dot(c: usize) -> f64 {
    let digits = [c / 256, c / 16 % 16, c % 16];
    [20, 16, 12]
        .iter()
        .zip(digits)
        .map(|(&e, digit)| 2_f64.powi(e) * (digit + 1) as f64)
        .sum()
}

/// Every key and value element is an E4M3 value; the BF16 scheme's digits
/// of 64 are not, past 16.
#[test]
fn the_rows_are_e4m3_and_the_bf16_digits_are_not() {
    for c in [0, 1, 255, 256, 4095] {
        let key = fp8_values(&key_row(c, WIDTH));
        assert!(
            key.iter()
                .all(|&k| k.fract() == 0.0 && (1.0..=16.0).contains(&k))
        );
        assert!(
            fp8_values(&value_row(c, WIDTH))
                .iter()
                .all(|v| v.is_finite())
        );
    }
    assert!(e4m3(16.0).is_ok());
    for value in [17.0, 33.0, 65.0] {
        let refused = e4m3(value).expect_err("not E4M3");
        assert!(refused.contains("not an E4M3"), "{refused}");
    }
    assert!(fp8_values(&[DECOY])[0].is_nan(), "the decoy is E4M3 NaN");
}

/// Two codes' dots differ by at least 4096, so at K scale 2 and softmax
/// scale 1/16 their scores differ by at least 512, past the margin.
#[test]
fn codes_lie_at_least_4096_apart() {
    let query = decoded(&query_row(WIDTH, false));
    let mut dots = (0..CODES)
        .map(|c| {
            let dot = dot_values(&query, &fp8_values(&key_row(c, WIDTH))).expect("exact dot");
            assert_eq!(dot, code_dot(c), "code {c}");
            dot
        })
        .collect::<Vec<_>>();
    dots.sort_by(f64::total_cmp);
    let gap = dots
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(f64::INFINITY, f64::min);
    assert_eq!(gap, 4096.0);
    assert_eq!(*dots.last().unwrap(), code_dot(4095));
    assert!(code_dot(4095) < 2_f64.powi(25));
    assert!(gap * f64::from(K_SCALE * SCALE) >= 4.0 * MARGIN);
}

/// The generated operands keep their exact one-hot partials: the best
/// score times `k_scale * scale` and the best value row times `v_scale`.
#[test]
fn the_generated_operands_are_one_hot() {
    for negative in [false, true] {
        let query = decoded(&query_row(WIDTH, negative));
        let partial = over(
            &query,
            &[code(0), code(4095), code(2049)],
            (K_SCALE, V_SCALE),
        )
        .expect("one-hot");
        let (best, dot) = if negative {
            (0, -code_dot(0))
        } else {
            (4095, code_dot(4095))
        };
        assert_eq!(partial.max, [dot as f32 * (K_SCALE * SCALE)]);
        assert_eq!(partial.den, [1.0]);
        assert_eq!(
            partial.num,
            fp8_values(&value_row(best, WIDTH))
                .iter()
                .map(|&v| v as f32 * V_SCALE)
                .collect::<Vec<_>>()
        );
    }
}

/// A K scale of 1/16 leaves codes 4094 and 4095 a score of 16 apart, so
/// the softmax is not one-hot.
#[test]
fn a_k_scale_that_closes_the_gap_is_refused() {
    let query = decoded(&query_row(WIDTH, false));
    let refused = refusal(
        over(&query, &[code(4094), code(4095)], (0.0625, V_SCALE)),
        "K scale 1/16",
    );
    assert!(refused.contains("not one-hot"), "{refused}");
}

/// E4M3 NaN in a key where the query is 0, or in the value row of a key
/// whose weight is +0, has no exact partial.
#[test]
fn nan_codes_are_refused() {
    let query = decoded(&query_row(WIDTH, false));
    let (mut key, value) = code(7);
    key[5] = 0x7f;
    let refused = refusal(
        over(&query, &[code(4095), (key, value)], (K_SCALE, V_SCALE)),
        "NaN key",
    );
    assert!(refused.contains("finite"), "{refused}");
    let (key, mut value) = code(7);
    value[9] = 0xff;
    let refused = refusal(
        over(&query, &[code(4095), (key, value)], (K_SCALE, V_SCALE)),
        "NaN value",
    );
    assert!(refused.contains("finite"), "{refused}");
}

/// A pool of one latent page holding the codes and scales of `tokens`, at
/// a stride of the page alone.
fn latent_page(tokens: &[(Vec<u8>, Vec<f32>)]) -> Vec<u8> {
    let mut page = vec![0; PAGE * LATENT + PAGE * LATENT / BLOCK * 4];
    for (slot, (codes, scales)) in tokens.iter().enumerate() {
        page[slot * LATENT..][..LATENT].copy_from_slice(codes);
        let scales = hot_keys::f32_bytes(scales);
        page[PAGE * LATENT + slot * scales.len()..][..scales.len()].copy_from_slice(&scales);
    }
    page
}

fn latent_over(query: &[f64], tokens: &[(Vec<u8>, Vec<f32>)]) -> LabResult<Partials> {
    let page = latent_page(tokens);
    let keys = (0..tokens.len()).map(|slot| (0, slot)).collect::<Vec<_>>();
    latent_partials(query, 1, LATENT, &page, page.len(), PAGE, &keys, SCALE)
}

fn latent_code(code: usize) -> (Vec<u8>, Vec<f32>) {
    (key_row(code, LATENT), latent_scales(code, LATENT))
}

/// The latent pool lays out each page's codes, then its scales, and the
/// generated tokens keep their exact one-hot partials: block 0's chain
/// times K scale 2 (scores 512 apart), and each block of the best token's
/// codes times its own scale.
#[test]
fn the_latent_operands_are_one_hot() {
    let stride = PAGE * LATENT * 2;
    let pool = latent_pool(3, PAGE, LATENT, stride);
    for (page, slot) in [(0, 5), (2, 63)] {
        let code = page * PAGE + slot;
        let (codes, scales) = latent_token(&pool, stride, PAGE, LATENT, page as i32, slot).unwrap();
        assert_eq!(codes, fp8_values(&key_row(code, LATENT)));
        assert_eq!(scales, latent_scales(code, LATENT));
    }
    assert!(
        pool[PAGE * LATENT + PAGE * LATENT / BLOCK * 4..stride]
            .iter()
            .all(|&byte| byte == DECOY)
    );
    // Later blocks' scales vary by token.
    assert_ne!(latent_scales(5, LATENT)[1], latent_scales(6, LATENT)[1]);
    for negative in [false, true] {
        let query = decoded(&query_row(LATENT, negative));
        let partial = latent_over(
            &query,
            &[latent_code(0), latent_code(4095), latent_code(70)],
        )
        .expect("one-hot");
        let (best, dot) = if negative {
            (0, -code_dot(0))
        } else {
            (4095, code_dot(4095))
        };
        assert_eq!(partial.max, [(dot * f64::from(K_SCALE)) as f32 * SCALE]);
        assert_eq!(partial.den, [1.0]);
        let scales = latent_scales(best, LATENT);
        let expected = fp8_values(&key_row(best, LATENT))
            .iter()
            .enumerate()
            .map(|(d, &v)| v as f32 * scales[d / BLOCK])
            .collect::<Vec<_>>();
        assert_eq!(partial.num, expected);
    }
}

/// A block scale that is not a power of two would need more than one BF16
/// probability term and an inexact product with the codes.
#[test]
fn a_latent_scale_off_a_power_of_two_is_refused() {
    let query = decoded(&query_row(LATENT, false));
    let (codes, mut scales) = latent_code(9);
    scales[2] = 3.0;
    let refused = refusal(
        latent_over(&query, &[latent_code(4095), (codes, scales)]),
        "block scale 3",
    );
    assert!(refused.contains("power of two"), "{refused}");
}

/// Block chains 2^34 and 2^-10 after their scales are each exact, but F32
/// cannot add them exactly, so the score has no exact bits.
#[test]
fn block_scales_that_make_the_chain_sum_inexact_are_refused() {
    let mut query = vec![0.0; LATENT];
    query[0] = 2_f64.powi(20);
    query[BLOCK] = 1.0;
    let mut codes = vec![e4m3(1.0).unwrap(); LATENT];
    codes[0] = e4m3(16.0).unwrap();
    let mut scales = vec![1.0; LATENT / BLOCK];
    scales[0] = 2_f32.powi(10);
    scales[1] = 2_f32.powi(-10);
    let refused = refusal(latent_over(&query, &[(codes, scales)]), "inexact chain sum");
    assert!(refused.contains("exactly an F32 sum"), "{refused}");
}

/// Two losing keys whose block 1 values are each a finite `9 * 2^124`
/// (scale 2^124) sum past the largest F32 in a running numerator before
/// the winner arrives, and a later zero rescale of that infinity is NaN,
/// so their partials have no exact bits even though every scaled value is
/// finite.
#[test]
fn values_whose_running_sum_overflows_f32_are_refused() {
    let query = decoded(&query_row(LATENT, false));
    let (codes, mut scales) = latent_code(0);
    scales[1] = 2_f32.powi(124);
    let loser = (codes, scales);
    assert!((9.0 * 2_f32.powi(124)).is_finite());
    let refused = refusal(
        latent_over(&query, &[loser.clone(), loser, latent_code(4095)]),
        "overflowing running numerator",
    );
    assert!(refused.contains("overflow"), "{refused}");
}
