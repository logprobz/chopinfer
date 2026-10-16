use runtime_model_api::ModelVocabulary;
use runtime_sampling::{
    LogitBias, SampleContext, SamplingError, SamplingParameters, SamplingState, philox4x32_10,
    sample, sample_for_vocabulary, sample_padded_for_vocabulary,
};

fn params(temperature: f64) -> SamplingParameters {
    SamplingParameters::new(temperature).unwrap()
}

fn draw(
    parameters: &SamplingParameters,
    logits: &[f32],
    history: &[u32],
    emitted_tokens: usize,
    eos: &[u32],
    state: &mut SamplingState,
) -> runtime_sampling::Sample {
    sample(
        parameters,
        logits,
        SampleContext::new(history, emitted_tokens, eos),
        state,
    )
    .unwrap()
}

#[test]
fn validates_parameters_and_input_distributions() {
    assert_eq!(
        SamplingParameters::new(f64::NAN),
        Err(SamplingError::InvalidTemperature)
    );
    assert_eq!(
        SamplingParameters::new(-0.1),
        Err(SamplingError::InvalidTemperature)
    );
    assert_eq!(params(1.0).with_top_p(0.0), Err(SamplingError::InvalidTopP));
    assert_eq!(params(1.0).with_min_p(1.1), Err(SamplingError::InvalidMinP));
    assert_eq!(
        params(1.0).with_top_logprobs(21),
        Err(SamplingError::TooManyTopLogprobs)
    );
    assert_eq!(
        params(1.0).with_logit_bias(vec![
            LogitBias::new(1, 0.5).unwrap(),
            LogitBias::new(1, -0.5).unwrap(),
        ]),
        Err(SamplingError::DuplicateLogitBias(1))
    );

    let mut state = SamplingState::new(7);
    assert_eq!(
        sample(
            &params(1.0),
            &[],
            SampleContext::new(&[], 0, &[]),
            &mut state
        ),
        Err(SamplingError::EmptyVocabulary)
    );
    assert_eq!(
        sample(
            &params(1.0),
            &[0.0, f32::NAN],
            SampleContext::new(&[], 0, &[]),
            &mut state,
        ),
        Err(SamplingError::InvalidLogit { token_id: 1 })
    );
    assert_eq!(
        sample(
            &params(1.0).with_minimum_new_tokens(1).unwrap(),
            &[0.0],
            SampleContext::new(&[], 0, &[0]),
            &mut state,
        ),
        Err(SamplingError::AllTokensMasked)
    );
}

#[test]
fn greedy_temperature_zero_uses_token_id_for_ties() {
    let sample = draw(
        &params(0.0).with_top_logprobs(4).unwrap(),
        &[2.0, 2.0, 1.0],
        &[],
        0,
        &[],
        &mut SamplingState::new(99),
    );
    assert_eq!(sample.token_id(), 0);
    assert!((sample.logprob() - (-0.861_994_804_058_251_1)).abs() < 1.0e-12);
    assert_eq!(
        sample
            .top_logprobs()
            .iter()
            .map(|entry| entry.token_id())
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(sample.top_logprobs()[0].logprob(), sample.logprob());
    assert_eq!(
        sample.top_logprobs()[0].logprob(),
        sample.top_logprobs()[1].logprob()
    );
}

/// Temperatures below the device contract's floor sample greedily: the
/// parameters hold 0 and the draw is the argmax with ties to the lower ID,
/// for every seed. The floor itself stays a sampling temperature.
#[test]
fn temperatures_below_the_floor_sample_greedily() {
    let floor = kernel_core::sampling::MIN_TEMPERATURE;
    assert_eq!(params(floor).temperature(), floor);
    let logits = [1.0, 3.0, f32::MAX, f32::MAX, -f32::MAX];
    for temperature in [-0.0, 5e-324, f64::MIN_POSITIVE, 1e-300, floor.next_down()] {
        let parameters = params(temperature)
            .with_top_p(0.5)
            .unwrap()
            .with_top_logprobs(2)
            .unwrap();
        assert_eq!(parameters.temperature().to_bits(), 0.0_f64.to_bits());
        for seed in 0..32 {
            let greedy = draw(
                &params(0.0).with_top_logprobs(2).unwrap(),
                &logits,
                &[],
                0,
                &[],
                &mut SamplingState::new(seed),
            );
            let mut state = SamplingState::new(seed);
            let tiny = draw(&parameters, &logits, &[], 0, &[], &mut state);
            assert_eq!(tiny.token_id(), 2, "temperature {temperature:e}");
            assert_eq!(tiny, greedy);
            assert_eq!(state, SamplingState::new(seed).after(1));
        }
    }
}

#[test]
fn greedy_logprobs_match_a_known_log_softmax_vector() {
    let sample = draw(
        &params(0.0).with_top_logprobs(3).unwrap(),
        &[0.0, 1.0, 2.0],
        &[],
        0,
        &[],
        &mut SamplingState::new(7),
    );
    let normalizer = (1.0_f64 + 1.0_f64.exp() + 2.0_f64.exp()).ln();
    assert_eq!(sample.token_id(), 2);
    assert!((sample.logprob() - (2.0 - normalizer)).abs() < 1.0e-12);
    for (entry, expected_token) in sample.top_logprobs().iter().zip([2, 1, 0]) {
        assert_eq!(entry.token_id(), expected_token);
        assert!((entry.logprob() - (f64::from(expected_token) - normalizer)).abs() < 1.0e-12);
    }
}

#[test]
fn greedy_logprobs_are_invariant_to_a_large_common_offset() {
    let parameters = params(0.0).with_top_logprobs(3).unwrap();
    let draw_at =
        |logits: &[f32]| draw(&parameters, logits, &[], 0, &[], &mut SamplingState::new(7));
    let baseline = draw_at(&[0.0, 0.0, 0.0]);
    let offset = draw_at(&[1.0e30, 1.0e30, 1.0e30]);
    assert_eq!(baseline.token_id(), offset.token_id());
    assert!((offset.logprob() + 3.0_f64.ln()).abs() < 1.0e-12);
    assert_eq!(baseline.top_logprobs(), offset.top_logprobs());
}

#[test]
fn ranked_signed_zero_ties_use_ascending_token_ids() {
    let sample = draw(
        &params(0.0).with_top_logprobs(3).unwrap(),
        &[-0.0, 0.0, -1.0],
        &[],
        0,
        &[],
        &mut SamplingState::new(7),
    );
    assert_eq!(sample.token_id(), 0);
    assert_eq!(
        sample
            .top_logprobs()
            .iter()
            .map(|entry| entry.token_id())
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
}

#[test]
fn declared_vocabulary_rejects_empty_mismatched_and_non_finite_rows() {
    let vocabulary = ModelVocabulary::new(3).unwrap();
    let context = SampleContext::new(&[], 0, &[]);
    let mut state = SamplingState::new(1);
    assert_eq!(
        sample_for_vocabulary(&params(0.0), &[], vocabulary, context, &mut state),
        Err(SamplingError::VocabularyMismatch {
            declared: 3,
            logits: 0,
        })
    );
    assert_eq!(
        sample_for_vocabulary(&params(0.0), &[0.0, 1.0], vocabulary, context, &mut state),
        Err(SamplingError::VocabularyMismatch {
            declared: 3,
            logits: 2,
        })
    );
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            sample_for_vocabulary(
                &params(0.0),
                &[0.0, invalid, 1.0],
                vocabulary,
                context,
                &mut state,
            ),
            Err(SamplingError::InvalidLogit { token_id: 1 })
        );
    }
}

#[test]
fn greedy_large_vocabulary_row_has_bounded_exact_results() {
    let vocabulary = ModelVocabulary::new(154_880).unwrap();
    let mut logits = vec![-20.0; vocabulary.get()];
    logits[154_879] = 4.0;
    logits[42] = 3.0;
    let sample = sample_for_vocabulary(
        &params(0.0).with_top_logprobs(2).unwrap(),
        &logits,
        vocabulary,
        SampleContext::new(&[], 0, &[]),
        &mut SamplingState::new(9),
    )
    .unwrap();
    assert_eq!(sample.token_id(), 154_879);
    assert!(sample.logprob().is_finite() && sample.logprob() < 0.0);
    assert_eq!(
        sample
            .top_logprobs()
            .iter()
            .map(|entry| entry.token_id())
            .collect::<Vec<_>>(),
        [154_879, 42]
    );
}

#[test]
fn seeded_state_replays_a_request_sequence() {
    let parameters = params(0.8).with_top_logprobs(3).unwrap();
    let logits = [0.1, 0.2, 0.3, 0.4];
    let trace = |seed| {
        let mut state = SamplingState::new(seed);
        (0..32)
            .map(|_| draw(&parameters, &logits, &[], 0, &[], &mut state).token_id())
            .collect::<Vec<_>>()
    };
    assert_eq!(trace(1234), trace(1234));
    assert_ne!(trace(1234), trace(4321));
}

#[test]
fn top_k_top_p_and_min_p_truncate_in_documented_order() {
    let logits = [8.0, 7.0, 0.0, -1.0];
    for parameters in [
        params(1.0).with_top_k(1).unwrap(),
        params(1.0).with_top_p(0.6).unwrap(),
        params(1.0).with_min_p(0.5).unwrap(),
    ] {
        let sample = draw(
            &parameters.with_top_logprobs(4).unwrap(),
            &logits,
            &[],
            0,
            &[],
            &mut SamplingState::new(3),
        );
        assert_eq!(sample.token_id(), 0);
        assert_eq!(
            sample
                .top_logprobs()
                .iter()
                .map(|entry| entry.token_id())
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
    }
}

#[test]
fn bias_and_history_penalties_change_selection() {
    let biased = params(0.0)
        .with_logit_bias(vec![LogitBias::new(2, 3.0).unwrap()])
        .unwrap();
    assert_eq!(
        draw(
            &biased,
            &[2.0, 1.0, 0.0],
            &[],
            0,
            &[],
            &mut SamplingState::new(0),
        )
        .token_id(),
        2
    );

    let penalized = params(0.0).with_penalties(1.0, 0.5, 2.0).unwrap();
    assert_eq!(
        draw(
            &penalized,
            &[4.0, 3.0],
            &[0, 0],
            0,
            &[],
            &mut SamplingState::new(0),
        )
        .token_id(),
        1
    );
}

#[test]
fn a_history_split_into_a_tail_samples_like_the_joined_history() {
    let parameters = params(0.7)
        .with_penalties(1.3, 0.4, 0.6)
        .unwrap()
        .with_top_logprobs(3)
        .unwrap();
    let logits = [1.0, 2.5, 0.5, 2.0, 1.5, 0.0];
    let history = [1, 3, 3, 4, 1, 1, 0];
    for emitted in 0..=history.len() {
        for split in 0..=history.len() {
            let mut joined_state = SamplingState::new(9);
            let joined = sample(
                &parameters,
                &logits,
                SampleContext::new(&history, emitted, &[]),
                &mut joined_state,
            )
            .unwrap();
            let mut split_state = SamplingState::new(9);
            let (head, tail) = history.split_at(split);
            let parts = sample(
                &parameters,
                &logits,
                SampleContext::new(head, emitted, &[]).with_history_tail(tail),
                &mut split_state,
            )
            .unwrap();
            assert_eq!(parts, joined, "emitted {emitted}, split {split}");
            assert_eq!(split_state, joined_state);
        }
    }
}

#[test]
fn eos_is_masked_until_the_minimum_and_then_selectable() {
    let parameters = params(0.0).with_minimum_new_tokens(2).unwrap();
    let mut state = SamplingState::new(1);
    assert_eq!(
        draw(&parameters, &[1.0, 9.0], &[], 1, &[1], &mut state).token_id(),
        0
    );
    assert_eq!(
        draw(&parameters, &[1.0, 9.0], &[], 2, &[1], &mut state).token_id(),
        1
    );
}

#[test]
fn selected_and_top_logprobs_are_stable_and_bounded() {
    let sample = draw(
        &params(1.0).with_top_logprobs(3).unwrap(),
        &[1.0, 2.0, 2.0, 0.0],
        &[],
        0,
        &[],
        &mut SamplingState::new(5),
    );
    assert_eq!(sample.top_logprobs().len(), 3);
    assert_eq!(sample.top_logprobs()[0].token_id(), 1);
    assert_eq!(sample.top_logprobs()[1].token_id(), 2);
    let selected = sample
        .top_logprobs()
        .iter()
        .find(|entry| entry.token_id() == sample.token_id())
        .unwrap();
    assert_eq!(sample.logprob(), selected.logprob());
}

#[test]
fn numerical_extremes_use_stable_normalization() {
    let sample = draw(
        &params(1.0).with_top_logprobs(2).unwrap(),
        &[10_000.0, 9_999.0, -10_000.0],
        &[],
        0,
        &[],
        &mut SamplingState::new(2),
    );
    assert!(sample.logprob().is_finite());
    assert!(
        sample
            .top_logprobs()
            .iter()
            .all(|entry| entry.logprob().is_finite())
    );
    assert!(sample.token_id() < 2);
}

#[test]
fn philox_matches_the_zero_key_zero_counter_vector() {
    assert_eq!(
        philox4x32_10(0, 0, 0),
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
    );
}

#[test]
fn prompt_repetition_and_output_frequency_are_distinct() {
    let parameters = params(0.0).with_penalties(0.0, 2.0, 2.0).unwrap();
    let prompt_only = draw(
        &parameters,
        &[4.0, 3.0],
        &[0],
        0,
        &[],
        &mut SamplingState::new(1),
    );
    let output_only = draw(
        &parameters,
        &[4.0, 3.0],
        &[0],
        1,
        &[],
        &mut SamplingState::new(1),
    );
    assert_eq!(prompt_only.token_id(), 1);
    assert_eq!(output_only.token_id(), 1);
    let no_frequency = params(0.0).with_penalties(0.0, 0.0, 2.0).unwrap();
    let repeated_prompt = draw(
        &no_frequency,
        &[4.0, 1.0],
        &[0],
        0,
        &[],
        &mut SamplingState::new(1),
    );
    assert_eq!(repeated_prompt.token_id(), 0);
}

#[test]
fn padding_is_excluded_from_sampling_and_raw_logprobs() {
    let vocabulary = ModelVocabulary::new(2).unwrap();
    let sample = sample_padded_for_vocabulary(
        &params(0.0).with_top_logprobs(2).unwrap(),
        &[1.0, 2.0, f32::NAN, 1000.0],
        vocabulary,
        SampleContext::new(&[], 0, &[]),
        &mut SamplingState::new(9),
    )
    .unwrap();
    assert_eq!(sample.token_id(), 1);
    assert_eq!(sample.top_logprobs()[0].token_id(), 1);
    assert!((sample.logprob() + (1.0 + (-1.0_f64).exp()).ln()).abs() < 1e-12);
}

#[test]
fn processing_keeps_exact_bias_and_penalty_decisions_at_f32_boundaries() {
    let bias = params(0.0)
        .with_logit_bias(vec![LogitBias::new(1, 1e-8).unwrap()])
        .unwrap();
    assert_eq!(1.0_f32 + 1e-8_f32, 1.0);
    assert_eq!(
        draw(&bias, &[1.0, 1.0], &[], 0, &[], &mut SamplingState::new(0)).token_id(),
        1
    );

    let repetition = params(0.0).with_penalties(0.0, 0.0, 1.0 + 1e-8).unwrap();
    assert_eq!((1.0 + 1e-8) as f32, 1.0);
    assert_eq!(
        draw(
            &repetition,
            &[1.0, 1.0],
            &[0],
            0,
            &[],
            &mut SamplingState::new(0)
        )
        .token_id(),
        1
    );

    // F32 rounds this bias to -1 and misses the negative sign before repetition.
    let sign = params(0.0)
        .with_logit_bias(vec![LogitBias::new(0, -1.0 - 1e-8).unwrap()])
        .unwrap()
        .with_penalties(0.0, 0.0, 2.0)
        .unwrap();
    assert_eq!(1.0_f32 + (-1.0 - 1e-8) as f32, 0.0);
    assert_eq!(
        draw(
            &sign,
            &[1.0, -1.5e-8],
            &[0],
            0,
            &[],
            &mut SamplingState::new(0)
        )
        .token_id(),
        1
    );
}

#[test]
fn repetition_penalty_has_a_finite_supported_range() {
    for penalty in [1e-308, 0.009, 10.01, 1e308] {
        assert!(params(1.0).with_penalties(0.0, 0.0, penalty).is_err());
    }
    for penalty in [0.01, 0.6, 1.0, 10.0] {
        assert!(params(1.0).with_penalties(0.0, 0.0, penalty).is_ok());
    }
}
