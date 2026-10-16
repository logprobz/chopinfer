use runtime_sampling::{
    LogitBias, SampleContext, SamplingError, SamplingParameters, SamplingState, sample,
};

fn mask(allowed: &[u32], size: usize) -> Vec<u32> {
    let mut words = vec![0; size.div_ceil(32)];
    for &token in allowed {
        words[token as usize / 32] |= 1 << (token % 32);
    }
    words
}

fn log_softmax(logits: &[f32], allowed: &[u32]) -> Vec<f64> {
    let maximum = allowed
        .iter()
        .map(|&token| f64::from(logits[token as usize]))
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = allowed
        .iter()
        .map(|&token| (f64::from(logits[token as usize]) - maximum).exp())
        .sum();
    logits
        .iter()
        .map(|&logit| f64::from(logit) - maximum - sum.ln())
        .collect()
}

#[test]
fn greedy_selects_within_the_mask_and_renormalizes_logprobs() {
    let logits = [3.0_f32, 2.0, 1.0, 0.5, -1.0, 4.0];
    let allowed = [1, 3, 4];
    let words = mask(&allowed, logits.len());
    let parameters = SamplingParameters::new(0.0)
        .unwrap()
        .with_top_logprobs(5)
        .unwrap();
    let mut state = SamplingState::new(1);
    let drawn = sample(
        &parameters,
        &logits,
        SampleContext::new(&[], 0, &[]).with_allowed_tokens(&words),
        &mut state,
    )
    .unwrap();
    let expected = log_softmax(&logits, &allowed);
    assert_eq!(drawn.token_id(), 1);
    assert!((drawn.logprob() - expected[1]).abs() < 1e-12);
    let top: Vec<u32> = drawn
        .top_logprobs()
        .iter()
        .map(|entry| entry.token_id())
        .collect();
    assert_eq!(top, vec![1, 3, 4, 0, 2]);
    for entry in drawn.top_logprobs() {
        if allowed.contains(&entry.token_id()) {
            assert!((entry.logprob() - expected[entry.token_id() as usize]).abs() < 1e-12);
        } else {
            assert_eq!(entry.logprob(), f64::NEG_INFINITY);
        }
    }
}

#[test]
fn sampled_rows_never_leave_the_mask() {
    let logits: Vec<f32> = (0..64).map(|token| (token % 7) as f32).collect();
    let allowed = [2, 9, 40, 63];
    let words = mask(&allowed, logits.len());
    let parameters = SamplingParameters::new(1.3)
        .unwrap()
        .with_top_k(3)
        .unwrap()
        .with_top_p(0.9)
        .unwrap()
        .with_min_p(0.01)
        .unwrap()
        .with_penalties(0.5, 0.5, 1.2)
        .unwrap()
        .with_logit_bias(vec![LogitBias::new(0, 100.0).unwrap()])
        .unwrap();
    let mut state = SamplingState::new(99);
    for _ in 0..200 {
        let drawn = sample(
            &parameters,
            &logits,
            SampleContext::new(&[0, 2, 9], 1, &[63]).with_allowed_tokens(&words),
            &mut state,
        )
        .unwrap();
        assert!(allowed.contains(&drawn.token_id()));
    }
}

#[test]
fn minimum_length_yields_to_a_mask_that_only_admits_a_stop_token() {
    let logits = [0.0_f32, 1.0, 2.0];
    let parameters = SamplingParameters::new(1.0)
        .unwrap()
        .with_minimum_new_tokens(5)
        .unwrap();
    let only_stop = mask(&[2], logits.len());
    let mut state = SamplingState::new(3);
    let first = sample(
        &parameters,
        &logits,
        SampleContext::new(&[], 0, &[2]).with_allowed_tokens(&only_stop),
        &mut state,
    )
    .unwrap();
    assert_eq!(first.token_id(), 2);
    // With another admitted token the stop token stays suppressed.
    let both = mask(&[1, 2], logits.len());
    for _ in 0..50 {
        let drawn = sample(
            &parameters,
            &logits,
            SampleContext::new(&[], 0, &[2]).with_allowed_tokens(&both),
            &mut state,
        )
        .unwrap();
        assert_eq!(drawn.token_id(), 1);
    }
}

#[test]
fn an_empty_mask_is_refused() {
    let logits = [0.0_f32; 4];
    let empty = mask(&[], logits.len());
    let parameters = SamplingParameters::new(0.0).unwrap();
    let mut state = SamplingState::new(0);
    assert_eq!(
        sample(
            &parameters,
            &logits,
            SampleContext::new(&[], 0, &[]).with_allowed_tokens(&empty),
            &mut state,
        ),
        Err(SamplingError::AllTokensMasked)
    );
}

#[test]
fn singleton_constraints_preserve_requested_top_logprob_count() {
    for temperature in [0.0, 1.0] {
        let p = SamplingParameters::new(temperature)
            .unwrap()
            .with_top_logprobs(3)
            .unwrap();
        let drawn = sample(
            &p,
            &[1.0, 2.0, 3.0, 4.0],
            SampleContext::new(&[], 0, &[]).with_allowed_tokens(&[2]),
            &mut SamplingState::new(7),
        )
        .unwrap();
        assert_eq!(drawn.token_id(), 1);
        assert_eq!(drawn.logprob(), 0.0);
        assert_eq!(drawn.top_logprobs().len(), 3);
        assert_eq!(drawn.top_logprobs()[0].token_id(), 1);
        for entry in &drawn.top_logprobs()[1..] {
            assert_ne!(entry.token_id(), 1);
            assert_eq!(entry.logprob(), f64::NEG_INFINITY);
        }
    }
}
