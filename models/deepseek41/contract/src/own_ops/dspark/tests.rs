use super::*;
use kernel_core::codec::f32_to_bf16;
use kernel_core::oracle::accumulation_bound;

#[test]
fn confidence_preserves_the_unrounded_hidden_at_production_width() {
    let op = ConfidenceHead {
        hidden: 5120,
        markov_rank: 256,
        hidden_stride: 5120,
        score_stride: 1,
    };
    let mut hidden = vec![0.0_f32; op.hidden as usize];
    hidden[0] = f32::from_bits(0x3f80_8000); // Exactly 1.00390625.
    let mut weights = vec![f32_to_bf16(0.0); (op.hidden + op.markov_rank) as usize];
    weights[0] = f32_to_bf16(1.0);
    let reference =
        confidence_reference(&op, &hidden, &vec![0; op.markov_rank as usize], &weights).unwrap();
    let expected = f64::from(hidden[0]);
    let bound = accumulation_bound((op.hidden + op.markov_rank) as usize, expected);
    assert!((reference.values[0] - expected).abs() <= bound);
    assert!((f64::from(bf16_to_f32(f32_to_bf16(hidden[0]))) - expected).abs() > bound);
}

fn bf16(values: &[f32]) -> Vec<u16> {
    values.iter().copied().map(f32_to_bf16).collect()
}

fn chain128(x: &[u16], w: &[u16]) -> f32 {
    chain128_f32(&x.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(), w)
}

fn chain128_f32(x: &[f32], w: &[u16]) -> f32 {
    let mut sum = 0.0_f32;
    for (x, w) in x.chunks(128).zip(w.chunks(128)) {
        let mut partial = 0.0_f32;
        for (&x, &w) in x.iter().zip(w) {
            partial = x.mul_add(bf16_to_f32(w), partial);
        }
        sum += partial;
    }
    sum
}

fn within(actual: &[f32], reference: &OracleOutput) -> bool {
    actual.len() == reference.values.len()
        && actual.iter().enumerate().all(|(i, &value)| {
            value.is_finite()
                && (f64::from(value) - reference.values[i]).abs()
                    <= accumulation_bound(
                        reference.reduction_lengths[i],
                        reference.sum_abs_products[i],
                    )
        })
}

#[test]
fn markov_adds_a_predecessor_projection_to_each_local_logit() {
    let op = MarkovStep {
        rank: 3,
        vocab_rows: 2,
        logits_stride: 10,
    };
    let embedding = bf16(&[1.0, 2.0, -0.5]);
    let weights = bf16(&[0.25, -1.0, 2.0, -0.5, 0.5, 4.0]);
    let reference = markov_reference(&op, &embedding, &weights, &[0.75, -2.0]).unwrap();
    assert_eq!(reference.values, [-2.0, -3.5]);
    assert_eq!(reference.sum_abs_products, [4.0, 5.5]);
    assert_eq!(reference.reduction_lengths, [4, 4]);
    assert!(within(&[-2.0, -3.5], &reference));
    assert!(!within(&[0.75, -2.0], &reference));
}

#[test]
fn confidence_uses_collapsed_hidden_then_predecessor_embedding() {
    let op = ConfidenceHead {
        hidden: 2,
        markov_rank: 2,
        hidden_stride: 10,
        score_stride: 5,
    };
    let hidden = vec![1.0, -2.0];
    let predecessor = bf16(&[0.5, 4.0]);
    let weights = bf16(&[0.25, 0.5, 2.0, -1.0]);
    let reference = confidence_reference(&op, &hidden, &predecessor, &weights).unwrap();
    assert_eq!(reference.values, [-3.75]);
    assert_eq!(reference.sum_abs_products, [6.25]);
    assert_eq!(reference.reduction_lengths, [4]);
    let wrong_token = confidence_reference(&op, &hidden, &bf16(&[4.0, 0.5]), &weights).unwrap();
    assert!(!within(&[wrong_token.values[0] as f32], &reference));
}

#[test]
fn confidence_keeps_the_f32_projection_without_bf16_output_rounding() {
    let op = ConfidenceHead {
        hidden: 1,
        markov_rank: 1,
        hidden_stride: 5,
        score_stride: 5,
    };
    let hidden = vec![1.0078125];
    let markov = bf16(&[2.015625]);
    let weight = bf16(&[1.015625, 0.5]);
    let reference = confidence_reference(&op, &hidden, &markov, &weight).unwrap();
    let value = chain128_f32(
        &[hidden, markov.into_iter().map(bf16_to_f32).collect()].concat(),
        &weight,
    );
    assert!(within(&[value], &reference));
    assert!(!within(&[bf16_to_f32(f32_to_bf16(value))], &reference));
}

#[test]
fn chains_restart_after_128_products_before_the_base_logit_is_added() {
    let op = MarkovStep {
        rank: 256,
        vocab_rows: 1,
        logits_stride: 5,
    };
    let embedding = bf16(&[1.0; 256]);
    let mut weights = bf16(&[1.0; 256]);
    weights[0] = f32_to_bf16(2.0_f32.powi(25));
    weights[128] = f32_to_bf16(-2.0_f32.powi(25));
    let reference = markov_reference(&op, &embedding, &weights, &[1.25]).unwrap();
    assert_eq!(reference.values, [255.25]);
    let chained = chain128(&embedding, &weights) + 1.25;
    assert_eq!(chained.to_bits(), 1.25_f32.to_bits());
    let long = embedding
        .iter()
        .zip(&weights)
        .fold(0.0_f32, |sum, (&x, &w)| {
            bf16_to_f32(x).mul_add(bf16_to_f32(w), sum)
        })
        + 1.25;
    assert_eq!(long.to_bits(), 128.25_f32.to_bits());
    assert_ne!(chained.to_bits(), long.to_bits());
    assert!(within(&[chained], &reference));
}

#[test]
fn f32_chains_meet_the_f64_bound_including_partial_chains() {
    for rank in [1, 127, 128, 129, 256, 257, 5376] {
        let x = (0..rank)
            .map(|i| f32_to_bf16(((i * 17 % 29) as f32 - 14.0) / 13.0))
            .collect::<Vec<_>>();
        let weights = (0..3 * rank)
            .map(|i| f32_to_bf16(((i * 31 % 53) as f32 - 26.0) / 7.0))
            .collect::<Vec<_>>();
        let op = MarkovStep {
            rank: rank as u32,
            vocab_rows: 3,
            logits_stride: 15,
        };
        let base = [0.0_f32, -13.125, 3.75];
        let reference = markov_reference(&op, &x, &weights, &base).unwrap();
        let actual = weights
            .chunks_exact(rank)
            .zip(base)
            .map(|(w, base)| chain128(&x, w) + base)
            .collect::<Vec<_>>();
        assert!(within(&actual, &reference), "rank={rank}");
    }
}

#[test]
fn invalid_shapes_strides_and_nonfinite_values_are_refused() {
    let markov = MarkovStep {
        rank: 2,
        vocab_rows: 3,
        logits_stride: 15,
    };
    for invalid in [
        MarkovStep { rank: 0, ..markov },
        MarkovStep {
            vocab_rows: 0,
            ..markov
        },
        MarkovStep {
            logits_stride: 2,
            ..markov
        },
    ] {
        assert!(invalid.validate(Need::rows(1)).is_err());
    }
    assert!(markov.validate(Need::rows(0)).is_err());
    assert!(markov_reference(&markov, &[0; 1], &[0; 6], &[0.0; 3]).is_err());
    assert!(markov_reference(&markov, &[0; 2], &[0; 5], &[0.0; 3]).is_err());
    assert!(markov_reference(&markov, &[0; 2], &[0; 6], &[f32::NAN; 3]).is_err());
    assert!(markov_reference(&markov, &[0x7f80; 2], &[0; 6], &[0.0; 3]).is_err());
    let confidence = ConfidenceHead {
        hidden: 2,
        markov_rank: 2,
        hidden_stride: 10,
        score_stride: 5,
    };
    for invalid in [
        ConfidenceHead {
            hidden: 0,
            ..confidence
        },
        ConfidenceHead {
            markov_rank: 0,
            ..confidence
        },
        ConfidenceHead {
            hidden_stride: 1,
            ..confidence
        },
        ConfidenceHead {
            score_stride: 0,
            ..confidence
        },
        ConfidenceHead {
            hidden: u32::MAX,
            hidden_stride: u32::MAX,
            ..confidence
        },
    ] {
        assert!(invalid.validate(Need::rows(1)).is_err());
    }
    assert!(confidence.validate(Need::rows(0)).is_err());
    assert!(confidence_reference(&confidence, &[0.0; 2], &[0; 2], &[0; 3]).is_err());
    assert!(confidence_reference(&confidence, &[0.0; 2], &[0; 2], &[0x7fc1; 4]).is_err());
    assert!(confidence_reference(&confidence, &[f32::NAN, 0.0], &[0; 2], &[0; 4]).is_err());
}
