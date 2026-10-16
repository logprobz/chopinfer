//! CPU check of the FP8 KV tolerance the GPU attention tests
//! apply: a kernel that matches the oracle over the stored E4M3 codes passes
//! against the attention of the unrounded BF16 values, and one that drops,
//! misreads, swaps or takes another layer's K or V scale fails. The
//! tolerance's storage term assumes independent rounding errors; correlated
//! rounding exceeds it with correct arithmetic.

use std::collections::BTreeMap;

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{AttentionLayout, AttentionPhase, AttentionQuery, DType};
use kernel_core::kv_fp8::{Fp8Attention, TOLERANCE_SIGMAS, attention_sigma, half_step, round_e4m3};
use kernel_core::ngram_oracle::splitmix64;
use kernel_core::oracle::{self, PagedAttention, check_numeric};

const HEADS: usize = 4;
const DIM: usize = 128;
const KEYS: usize = 1500;
/// Per test layer, the K and V scales the codes were written with; they
/// differ within each layer and between layers.
const LAYERS: [[f32; 2]; 3] = [[0.0625, 0.015625], [0.25, 0.0078125], [0.125, 0.03125]];

fn uniform(seed: u64, count: usize, scale: f64) -> Vec<f64> {
    (0..count as u64)
        .map(|i| {
            let unit = (splitmix64(seed * 1_000_003 + i) >> 11) as f64 / (1_u64 << 53) as f64;
            f64::from(bf16_to_f32(f32_to_bf16(
                ((2.0 * unit - 1.0) * scale) as f32,
            )))
        })
        .collect()
}

/// `E4M3(value / scale)` of each value read back with `decode_scale` (the
/// scale a kernel multiplies in).
fn stored(values: &[f64], scale: f32, decode_scale: f32) -> Vec<f64> {
    values
        .iter()
        .map(|&value| round_e4m3(value / f64::from(scale)) * f64::from(decode_scale))
        .collect()
}

/// A kernel output: the oracle over keys and values read back with the given
/// scales, rounded to BF16.
fn kernel(
    query: &AttentionQuery,
    q: &[f64],
    keys: &[f64],
    values: &[f64],
) -> (Vec<f32>, oracle::OracleOutput) {
    kernel_over(query, q, keys, values, KEYS)
}

fn kernel_over(
    query: &AttentionQuery,
    q: &[f64],
    keys: &[f64],
    values: &[f64],
    count: usize,
) -> (Vec<f32>, oracle::OracleOutput) {
    let output = oracle::attention(
        query,
        &PagedAttention {
            q,
            keys,
            values,
            page_tables: &[(0..count.div_ceil(64)).collect()],
            query_lengths: &[1],
            query_starts: &[count - 1],
            key_lengths: &[count],
            sink_logits: None,
        },
    )
    .expect("attention oracle");
    let rounded = output
        .values
        .iter()
        .map(|&v| bf16_to_f32(f32_to_bf16(v as f32)))
        .collect();
    (rounded, output)
}

/// The decode query of the tests: `HEADS` query heads on one KV head.
fn decode_query(keys: usize) -> AttentionQuery {
    AttentionQuery {
        q_heads: HEADS as u32,
        kv_heads: 1,
        qk_dim: DIM as u32,
        v_dim: DIM as u32,
        page_size: 64,
        kv_dtype: DType::Fp8E4M3,
        layout: AttentionLayout::Paged,
        causal: true,
        window_left: None,
        sink: false,
        softcap: None,
        softmax_scale: 1.0 / (DIM as f32).sqrt(),
        phase: AttentionPhase::Decode,
        max_context_capacity: keys.next_multiple_of(64) as u32,
        max_batch_capacity: 1,
    }
}

#[test]
fn fp8_kv_tolerance_accepts_a_correct_kernel_and_refuses_a_lost_scale() {
    let query = decode_query(KEYS);
    let q = uniform(1, HEADS * DIM, 2.0);
    let padded = KEYS.next_multiple_of(64) * DIM;
    let mut keys = uniform(2, padded, 0.75);
    let mut values = uniform(3, padded, 0.75);
    keys[KEYS * DIM..].fill(0.0);
    values[KEYS * DIM..].fill(0.0);
    // The codes of layer 0.
    let [k_written, v_written] = LAYERS[0];
    let (mut exact, mut sigma) = (Vec::new(), Vec::new());
    for head in 0..HEADS {
        let (output, deviation) = attention_sigma(Fp8Attention {
            query: &q[head * DIM..(head + 1) * DIM],
            keys: &keys[..KEYS * DIM],
            values: &values[..KEYS * DIM],
            qk_dim: DIM,
            v_dim: DIM,
            softmax_scale: f64::from(query.softmax_scale),
            k_scale: f64::from(k_written),
            v_scale: f64::from(v_written),
            sink: None,
        });
        exact.extend(output);
        sigma.extend(deviation);
    }
    let check = |k_scale: f32, v_scale: f32| {
        let (actual, reference) = kernel(
            &query,
            &q,
            &stored(&keys, k_written, k_scale),
            &stored(&values, v_written, v_scale),
        );
        let mut receipt = BTreeMap::new();
        let result =
            kernel_lab::check_fp8_kv_output(&actual, &exact, &sigma, &reference, &mut receipt);
        (
            result,
            receipt["fp8_kv_vs_bf16"]["max_error_sigmas"].clone(),
        )
    };
    let [k, v] = [k_written, v_written];
    let (correct, sigmas) = check(k, v);
    assert!(correct.is_ok(), "{correct:?} at {sigmas} sigmas");
    // A lost scale, a 25% wrong one, the pair swapped, another layer's pair.
    assert!(check(k, 1.0).0.is_err());
    assert!(check(1.0, v).0.is_err());
    assert!(check(k, v * 1.25).0.is_err());
    assert!(check(k * 1.25, v).0.is_err());
    assert!(check(v, k).0.is_err());
    for [other_k, other_v] in LAYERS.into_iter().filter(|&pair| pair != [k, v]) {
        assert!(check(other_k, other_v).0.is_err());
    }
}

/// The sigma allowance assumes independent rounding errors that partly
/// cancel in the softmax-weighted sum. Values repeated across keys round the
/// same way, so their errors add: a kernel equal to the oracle over the
/// stored codes is refused, and the receipt attributes the excess to storage,
/// not to arithmetic. The cases are the decode shape above with 1024 keys:
/// a zero query over equal keys with every value 1.0625, an E4M3 tie that
/// rounds to 1, and one random BF16 value row repeated under random queries
/// and keys, as a token repeated through the context would store it.
#[test]
fn correlated_rounding_exceeds_the_independent_storage_allowance() {
    const COUNT: usize = 1024;
    let query = decode_query(COUNT);
    let padded = COUNT.next_multiple_of(64) * DIM;
    let repeated = uniform(5, DIM, 0.75);
    let cases = [
        (
            "tie",
            vec![0.0; HEADS * DIM],
            vec![0.25; padded],
            vec![1.0625; DIM],
        ),
        (
            "row",
            uniform(6, HEADS * DIM, 2.0),
            uniform(7, padded, 0.75),
            repeated,
        ),
    ];
    for (name, q, keys, row) in cases {
        let values: Vec<f64> = row.iter().copied().cycle().take(padded).collect();
        let (mut exact, mut sigma) = (Vec::new(), Vec::new());
        for head in 0..HEADS {
            let (output, deviation) = attention_sigma(Fp8Attention {
                query: &q[head * DIM..(head + 1) * DIM],
                keys: &keys,
                values: &values,
                qk_dim: DIM,
                v_dim: DIM,
                softmax_scale: f64::from(query.softmax_scale),
                k_scale: 1.0,
                v_scale: 1.0,
                sink: None,
            });
            exact.extend(output);
            sigma.extend(deviation);
        }
        let (actual, reference) = kernel_over(
            &query,
            &q,
            &stored(&keys, 1.0, 1.0),
            &stored(&values, 1.0, 1.0),
            COUNT,
        );
        // The arithmetic over the stored codes is correct.
        let as_f64: Vec<f64> = actual.iter().map(|&v| f64::from(v)).collect();
        assert!(
            check_numeric(&as_f64, &reference, DType::Bf16).is_ok(),
            "{name}"
        );
        // The output weights sum to one, so the storage error of each output
        // element is the rounding error of the repeated value: the
        // worst-case first-order bound, at most half an E4M3 step.
        let mut largest = 0.0_f64;
        for (index, (stored, exact)) in reference.values.iter().zip(&exact).enumerate() {
            let value = row[index % DIM];
            let rounding = (round_e4m3(value) - value).abs();
            assert!((stored - exact).abs() <= rounding + 1e-12, "{name} {index}");
            assert!(rounding <= half_step(value.abs()));
            largest = largest.max(rounding);
        }
        assert!(largest > 0.0, "{name}: the repeated row stores exactly");
        let mut receipt = BTreeMap::new();
        let result =
            kernel_lab::check_fp8_kv_output(&actual, &exact, &sigma, &reference, &mut receipt);
        let report = &receipt["fp8_kv_vs_bf16"];
        let storage = report["max_storage_error_sigmas"].as_f64().unwrap();
        assert!(result.is_err(), "{name}: accepted at {storage} sigmas");
        assert_eq!(report["device_within_bound"], true, "{name}");
        assert!(storage > TOLERANCE_SIGMAS, "{name}: {storage} sigmas");
        assert!(
            result
                .unwrap_err()
                .contains("independent-rounding allowance")
        );
    }
}

/// A non-finite device output is an arithmetic failure, never correct
/// arithmetic with a storage excess.
#[test]
fn a_nan_output_is_refused_as_arithmetic_not_storage() {
    let stored = oracle::OracleOutput {
        values: vec![1.0, 0.5],
        sum_abs_products: vec![1.0, 0.5],
        reduction_lengths: vec![1, 1],
    };
    let exact = [1.0, 0.5];
    let sigma = [0.01, 0.01];
    for actual in [[f32::NAN, 0.5], [f32::INFINITY, 0.5]] {
        let mut receipt = BTreeMap::new();
        let result =
            kernel_lab::check_fp8_kv_output(&actual, &exact, &sigma, &stored, &mut receipt);
        let report = &receipt["fp8_kv_vs_bf16"];
        assert_eq!(report["device_within_bound"], false, "{actual:?}");
        assert!(report["max_device_error_bounds"].is_null(), "{report}");
        assert_eq!(report["max_storage_error_sigmas"], 0.0, "{report}");
        let error = result.unwrap_err();
        assert!(error.contains("exceeds its bound"), "{error}");
    }
    // An infinite oracle magnitude is outside the arithmetic gate too.
    let infinite = oracle::OracleOutput {
        sum_abs_products: vec![f64::INFINITY, 0.5],
        ..stored.clone()
    };
    let mut receipt = BTreeMap::new();
    let result =
        kernel_lab::check_fp8_kv_output(&[1.0, 0.5], &exact, &sigma, &infinite, &mut receipt);
    assert_eq!(receipt["fp8_kv_vs_bf16"]["device_within_bound"], false);
    assert!(result.unwrap_err().contains("exceeds its bound"));
}
