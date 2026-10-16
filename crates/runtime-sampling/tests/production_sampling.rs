//! Large-vocabulary sampling acceptance.
//! The row width exercises the full declared vocabulary domain.

use runtime_model_api::ModelVocabulary;
use runtime_sampling::{
    Sample, SampleContext, SamplingError, SamplingParameters, SamplingState, sample_for_vocabulary,
};

const PRODUCTION_VOCABULARY: usize = 129_280;

/// A truncation width strictly inside the declared production vocabulary. The
/// top-k parameter gate must accept it and still truncate the row, so it pins
/// the gate to the vocabulary domain rather than to a smaller fixed cap.
const PRODUCTION_TOP_K: usize = 100_000;

fn vocabulary() -> ModelVocabulary {
    ModelVocabulary::new(PRODUCTION_VOCABULARY).expect("the production vocabulary is declarable")
}

fn greedy(top_logprobs: usize) -> SamplingParameters {
    SamplingParameters::new(0.0)
        .and_then(|parameters| parameters.with_top_logprobs(top_logprobs))
        .expect("greedy parameters are valid")
}

fn sampled(temperature: f64, top_logprobs: usize) -> SamplingParameters {
    SamplingParameters::new(temperature)
        .and_then(|parameters| parameters.with_top_logprobs(top_logprobs))
        .expect("sampled parameters are valid")
}

fn draw(
    parameters: &SamplingParameters,
    logits: &[f32],
    context: SampleContext<'_>,
    state: &mut SamplingState,
) -> Sample {
    sample_for_vocabulary(parameters, logits, vocabulary(), context, state)
        .expect("a finite production row samples")
}

fn flat_row(value: f32) -> Vec<f32> {
    vec![value; PRODUCTION_VOCABULARY]
}

/// Deterministic row source. Values are multiples of 1/1024 in `[-12, 12]`, so
/// a common offset of 8192 is exact in `f32` and the row carries the dense ties
/// a real 129,280-wide logits row carries.
struct RowRng {
    word: u64,
}

impl RowRng {
    const fn new(seed: u64) -> Self {
        Self { word: seed }
    }

    fn next_unit(&mut self) -> f64 {
        self.word = self.word.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.word;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        (value >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

fn production_row(seed: u64) -> Vec<f32> {
    let mut rng = RowRng::new(seed);
    (0..PRODUCTION_VOCABULARY)
        .map(|_| ((rng.next_unit() * 24.0 - 12.0) * 1024.0).round() as f32 / 1024.0)
        .collect()
}

/// Independent log-softmax over the whole row. This is the oracle the sampler's
/// normalizer is checked against.
fn reference_log_softmax(logits: &[f32]) -> Vec<f64> {
    let values = logits
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_sum = values
        .iter()
        .map(|value| (value - maximum).exp())
        .sum::<f64>()
        .ln();
    values
        .iter()
        .map(|value| (value - maximum) - log_sum)
        .collect()
}

/// Independent ranking: descending logit, ascending token ID on a tie.
fn reference_ranking(logits: &[f32]) -> Vec<u32> {
    let mut ranked = (0..PRODUCTION_VOCABULARY as u32).collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        logits[*right as usize]
            .partial_cmp(&logits[*left as usize])
            .expect("the row is finite")
            .then_with(|| left.cmp(right))
    });
    ranked
}

fn top_ids(sample: &Sample) -> Vec<u32> {
    sample
        .top_logprobs()
        .iter()
        .map(|entry| entry.token_id())
        .collect()
}

#[test]
fn production_vocabulary_row_selects_and_ranks_over_the_full_domain() {
    let row = production_row(0x5eed_0001);
    let reference = reference_log_softmax(&row);
    let ranking = reference_ranking(&row);
    let sample = draw(
        &greedy(20),
        &row,
        SampleContext::new(&[], 0, &[]),
        &mut SamplingState::new(1),
    );

    assert_eq!(sample.token_id(), ranking[0]);
    assert!((sample.logprob() - reference[ranking[0] as usize]).abs() < 1.0e-12);
    assert_eq!(sample.top_logprobs().len(), 20);
    assert_eq!(top_ids(&sample), ranking[..20]);
    for entry in sample.top_logprobs() {
        let expected = reference[entry.token_id() as usize];
        assert!((entry.logprob() - expected).abs() < 1.0e-12);
        assert!(entry.logprob().is_finite() && entry.logprob() < 0.0);
    }
}

#[test]
fn declared_production_width_rejects_narrow_and_wide_rows() {
    let row = production_row(0x5eed_0002);
    let context = SampleContext::new(&[], 0, &[]);
    let mut state = SamplingState::new(2);

    assert_eq!(
        sample_for_vocabulary(
            &greedy(0),
            &row[..PRODUCTION_VOCABULARY - 1],
            vocabulary(),
            context,
            &mut state,
        ),
        Err(SamplingError::VocabularyMismatch {
            declared: PRODUCTION_VOCABULARY,
            logits: PRODUCTION_VOCABULARY - 1,
        })
    );

    let mut wide = row;
    wide.push(0.0);
    assert_eq!(
        sample_for_vocabulary(&greedy(0), &wide, vocabulary(), context, &mut state),
        Err(SamplingError::VocabularyMismatch {
            declared: PRODUCTION_VOCABULARY,
            logits: PRODUCTION_VOCABULARY + 1,
        })
    );
}

#[test]
fn non_finite_production_logits_are_rejected_at_the_exact_token() {
    let context = SampleContext::new(&[], 0, &[]);
    for token_id in [0, 64_640, PRODUCTION_VOCABULARY - 1] {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut row = production_row(0x5eed_0003);
            row[token_id] = invalid;
            for parameters in [greedy(20), sampled(0.8, 20)] {
                assert_eq!(
                    sample_for_vocabulary(
                        &parameters,
                        &row,
                        vocabulary(),
                        context,
                        &mut SamplingState::new(3),
                    ),
                    Err(SamplingError::InvalidLogit { token_id })
                );
            }
        }
    }
}

#[test]
fn greedy_ties_select_the_lowest_token_id() {
    let context = SampleContext::new(&[], 0, &[]);

    let mut scattered = flat_row(-30.0);
    for token_id in [7, 64_640, PRODUCTION_VOCABULARY - 1] {
        scattered[token_id] = 6.0;
    }
    let sample = draw(
        &greedy(3),
        &scattered,
        context,
        &mut SamplingState::new(0x1111),
    );
    assert_eq!(sample.token_id(), 7);
    assert_eq!(
        top_ids(&sample),
        [7, 64_640, PRODUCTION_VOCABULARY as u32 - 1]
    );

    let mut tail = flat_row(-30.0);
    tail[PRODUCTION_VOCABULARY - 2] = 2.0;
    tail[PRODUCTION_VOCABULARY - 1] = 2.0;
    let sample = draw(&greedy(2), &tail, context, &mut SamplingState::new(0x2222));
    assert_eq!(sample.token_id(), PRODUCTION_VOCABULARY as u32 - 2);
    assert_eq!(
        top_ids(&sample),
        [
            PRODUCTION_VOCABULARY as u32 - 2,
            PRODUCTION_VOCABULARY as u32 - 1
        ]
    );

    let uniform = flat_row(0.25);
    let sample = draw(
        &greedy(3),
        &uniform,
        context,
        &mut SamplingState::new(0x3333),
    );
    assert_eq!(sample.token_id(), 0);
    assert_eq!(top_ids(&sample), [0, 1, 2]);
    assert!((sample.logprob() + (PRODUCTION_VOCABULARY as f64).ln()).abs() < 1.0e-12);
}

#[test]
fn production_logprobs_are_stable_and_normalized() {
    let row = production_row(0x5eed_0004);
    let context = SampleContext::new(&[], 0, &[]);

    let first = draw(&greedy(20), &row, context, &mut SamplingState::new(11));
    let second = draw(&greedy(20), &row, context, &mut SamplingState::new(97));
    assert_eq!(first.token_id(), second.token_id());
    assert_eq!(first.logprob().to_bits(), second.logprob().to_bits());
    assert_eq!(first.top_logprobs(), second.top_logprobs());

    // 8192 is exact for every row value, so the normalized result must not move
    // by a single bit.
    let offset = row.iter().map(|value| value + 8192.0).collect::<Vec<_>>();
    let shifted = draw(&greedy(20), &offset, context, &mut SamplingState::new(11));
    assert_eq!(shifted.token_id(), first.token_id());
    assert_eq!(shifted.logprob().to_bits(), first.logprob().to_bits());
    assert_eq!(shifted.top_logprobs(), first.top_logprobs());

    let reference = reference_log_softmax(&row);
    let mass = reference.iter().map(|logprob| logprob.exp()).sum::<f64>();
    assert!((mass - 1.0).abs() < 1.0e-9);
    assert!((first.logprob() - reference[first.token_id() as usize]).abs() < 1.0e-12);
}

#[test]
fn top_k_covers_the_production_vocabulary() {
    let uniform = flat_row(0.5);
    let context = SampleContext::new(&[], 0, &[]);

    let full = sampled(1.0, 20)
        .with_top_k(PRODUCTION_VOCABULARY)
        .expect("top-k must span the declared production vocabulary");
    let sample = draw(&full, &uniform, context, &mut SamplingState::new(0x4444));
    assert!((sample.logprob() + (PRODUCTION_VOCABULARY as f64).ln()).abs() < 1.0e-12);

    let truncated = sampled(1.0, 20)
        .with_top_k(PRODUCTION_TOP_K)
        .expect("top-k must truncate inside the declared production vocabulary");
    let sample = draw(
        &truncated,
        &uniform,
        context,
        &mut SamplingState::new(0x5555),
    );
    assert!(sample.token_id() < PRODUCTION_TOP_K as u32);
    assert!((sample.logprob() + (PRODUCTION_VOCABULARY as f64).ln()).abs() < 1.0e-12);
    for entry in sample.top_logprobs() {
        assert!(entry.token_id() < PRODUCTION_TOP_K as u32);
        assert!((entry.logprob() + (PRODUCTION_VOCABULARY as f64).ln()).abs() < 1.0e-12);
    }
}

#[test]
fn top_k_parameter_gate_is_derived_from_the_vocabulary_ceiling() {
    let declared = sampled(1.0, 0)
        .with_top_k(runtime_sampling::MAX_VOCABULARY)
        .expect("a top-k at the vocabulary ceiling is declarable");
    assert_eq!(declared.top_k(), runtime_sampling::MAX_VOCABULARY);
    assert_eq!(
        sampled(1.0, 0).with_top_k(runtime_sampling::MAX_VOCABULARY + 1),
        Err(SamplingError::TopKTooLarge)
    );
}

fn decode_stream(seed: u64, steps: usize) -> Vec<(u32, u64)> {
    let parameters = SamplingParameters::new(0.8)
        .and_then(|parameters| parameters.with_top_p(0.95))
        .and_then(|parameters| parameters.with_min_p(0.01))
        .and_then(|parameters| parameters.with_top_logprobs(5))
        .expect("decode parameters are valid");
    let mut state = SamplingState::new(seed);
    let mut history = Vec::new();
    let mut stream = Vec::new();
    for step in 0..steps {
        let row = production_row(0x5eed_1000 + step as u64);
        let sample = draw(
            &parameters,
            &row,
            SampleContext::new(&history, step, &[]),
            &mut state,
        );
        history.push(sample.token_id());
        stream.push((sample.token_id(), sample.logprob().to_bits()));
    }
    stream
}

#[test]
fn seeded_replay_reproduces_a_production_decode_stream() {
    let replayed = decode_stream(20_260_828, 6);
    assert_eq!(replayed.len(), 6);
    assert_eq!(replayed, decode_stream(20_260_828, 6));
    assert_ne!(replayed, decode_stream(7, 6));
}

struct ProductionRequest {
    parameters: SamplingParameters,
    state: SamplingState,
    history: Vec<u32>,
    eos: Vec<u32>,
    emitted: usize,
}

impl ProductionRequest {
    fn step(&mut self, row: &[f32]) -> (u32, u64) {
        let sample = draw(
            &self.parameters,
            row,
            SampleContext::new(&self.history, self.emitted, &self.eos),
            &mut self.state,
        );
        self.history.push(sample.token_id());
        self.emitted += 1;
        (sample.token_id(), sample.logprob().to_bits())
    }
}

fn concurrent_requests() -> Vec<ProductionRequest> {
    let penalized = SamplingParameters::new(0.7)
        .and_then(|parameters| parameters.with_penalties(1.0, 0.5, 1.2))
        .and_then(|parameters| parameters.with_top_logprobs(4))
        .expect("penalized parameters are valid");
    let masked = SamplingParameters::new(0.9)
        .and_then(|parameters| parameters.with_minimum_new_tokens(8))
        .and_then(|parameters| parameters.with_top_logprobs(4))
        .expect("masked parameters are valid");
    vec![
        ProductionRequest {
            parameters: greedy(4),
            state: SamplingState::new(101),
            history: Vec::new(),
            eos: Vec::new(),
            emitted: 0,
        },
        ProductionRequest {
            parameters: sampled(0.6, 4),
            state: SamplingState::new(202),
            history: vec![5, 5, 9],
            eos: Vec::new(),
            emitted: 3,
        },
        ProductionRequest {
            parameters: penalized,
            state: SamplingState::new(303),
            history: vec![64_640, 64_640, PRODUCTION_VOCABULARY as u32 - 1],
            eos: Vec::new(),
            emitted: 3,
        },
        ProductionRequest {
            parameters: masked,
            state: SamplingState::new(404),
            history: Vec::new(),
            eos: vec![0, 64_640, PRODUCTION_VOCABULARY as u32 - 1],
            emitted: 0,
        },
    ]
}

#[test]
fn concurrent_request_state_stays_independent() {
    const STEPS: usize = 3;
    let rows = (0..STEPS)
        .map(|step| production_row(0x5eed_2000 + step as u64))
        .collect::<Vec<_>>();

    let isolated = concurrent_requests()
        .into_iter()
        .map(|mut request| rows.iter().map(|row| request.step(row)).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let mut interleaved_requests = concurrent_requests();
    let mut interleaved = vec![Vec::new(); interleaved_requests.len()];
    for row in &rows {
        for (request, stream) in interleaved_requests.iter_mut().zip(&mut interleaved) {
            stream.push(request.step(row));
        }
    }

    assert_eq!(isolated, interleaved);
    assert!(isolated.iter().all(|stream| stream.len() == STEPS));

    // The eos-masked request never emits a masked token before its minimum.
    for (token_id, _) in &isolated[3] {
        assert!(
            *token_id != 0 && *token_id != 64_640 && *token_id != PRODUCTION_VOCABULARY as u32 - 1
        );
    }

    // Two requests that differ only in history diverge, so per-request context
    // is not shared.
    let row = &rows[0];
    let parameters = SamplingParameters::new(0.0)
        .and_then(|parameters| parameters.with_penalties(0.0, 0.0, 4.0))
        .and_then(|parameters| parameters.with_top_logprobs(1))
        .expect("repetition parameters are valid");
    let ranking = reference_ranking(row);
    let clean = draw(
        &parameters,
        row,
        SampleContext::new(&[], 0, &[]),
        &mut SamplingState::new(505),
    );
    let repeated = draw(
        &parameters,
        row,
        SampleContext::new(&[ranking[0]], 1, &[]),
        &mut SamplingState::new(505),
    );
    assert_eq!(clean.token_id(), ranking[0]);
    assert_ne!(repeated.token_id(), ranking[0]);
}

#[test]
fn common_logit_offsets_preserve_random_selection() {
    for width in [4_usize, 1025, 152_576, 248_320] {
        for filtered in [false, true] {
            let p = if filtered {
                sampled(0.7, 3)
                    .with_top_k(20)
                    .unwrap()
                    .with_top_p(0.9)
                    .unwrap()
            } else {
                sampled(1.0, 3)
            };
            let mut mask = vec![0_u32; width.div_ceil(32)];
            for token in [0, width / 2, width - 1] {
                mask[token / 32] |= 1 << (token % 32);
            }
            for masked in [false, true] {
                let context = SampleContext::new(&[], 0, &[]);
                let context = if masked {
                    context.with_allowed_tokens(&mask)
                } else {
                    context
                };
                for seed in 0..if width == 4 { 64 } else { 4 } {
                    let draw = |offset| {
                        runtime_sampling::sample(
                            &p,
                            &vec![offset; width],
                            context,
                            &mut SamplingState::new(seed),
                        )
                        .unwrap()
                    };
                    let plain = draw(0.0);
                    for offset in [1e20, -1e20] {
                        let shifted = draw(offset);
                        assert_eq!(
                            shifted.token_id(),
                            plain.token_id(),
                            "width={width} seed={seed} offset={offset} filtered={filtered} masked={masked}"
                        );
                        assert_eq!(shifted.logprob(), plain.logprob());
                        assert_eq!(shifted.top_logprobs(), plain.top_logprobs());
                    }
                }
            }
        }
    }
}
