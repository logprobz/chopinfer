//! The CPU sampler as the oracle of the device sampler's contract
//! (`kernel_core::sampling`).

use kernel_core::contract::{Refusal, Validate};
use kernel_core::sampling::{RowFailure, RowOutcome, Sampled, SamplingRow};

use crate::{LogitBias, SampleContext, SamplingError, SamplingParameters, SamplingState, sample};

/// Samples one contract row with [`sample`]: the counts become a history,
/// an EOS mask a minimum length not yet reached, and `(seed, generated)` the
/// request's state. Refuses a row outside the contract.
pub fn sample_row(row: &SamplingRow<'_>) -> Result<RowOutcome, Refusal> {
    row.validate()?;
    let settings = row.settings;
    let refused = |_| Refusal("the CPU sampler refuses the row");
    let biases = row
        .biases
        .iter()
        .map(|bias| LogitBias::new(bias.token_id, bias.bias))
        .collect::<Result<Vec<_>, _>>()
        .map_err(refused)?;
    let mut history = row
        .counts
        .iter()
        .filter(|count| count.prompt_seen)
        .map(|count| count.token_id)
        .collect::<Vec<_>>();
    let prompt = history.len();
    for count in row.counts {
        history.extend(std::iter::repeat_n(
            count.token_id,
            count.output_count as usize,
        ));
    }
    let emitted = history.len() - prompt;
    let (eos, minimum): (&[u32], usize) = if settings.mask_eos {
        (row.eos, emitted + 1)
    } else {
        (&[], 0)
    };
    let parameters = SamplingParameters::new(settings.temperature)
        .and_then(|parameters| parameters.with_top_k(settings.top_k as usize))
        .and_then(|parameters| parameters.with_top_p(settings.top_p))
        .and_then(|parameters| parameters.with_min_p(settings.min_p))
        .and_then(|parameters| {
            parameters.with_penalties(
                settings.presence_penalty,
                settings.frequency_penalty,
                settings.repetition_penalty,
            )
        })
        .and_then(|parameters| parameters.with_logit_bias(biases))
        .and_then(|parameters| parameters.with_top_logprobs(settings.top_logprobs as usize))
        .and_then(|parameters| parameters.with_minimum_new_tokens(minimum))
        .map_err(refused)?;
    let mut context = SampleContext::new(&history, emitted, eos);
    if let Some(mask) = row.mask {
        context = context.with_allowed_tokens(mask);
    }
    let mut state = SamplingState::new(settings.seed).after(settings.generated);
    match sample(&parameters, row.logits, context, &mut state) {
        // The device reports admitted top entries; the host adds the masked tail.
        Ok(sampled) => Ok(Ok(Sampled {
            token_id: sampled.token_id(),
            logprob: sampled.logprob(),
            top: sampled
                .top_logprobs()
                .iter()
                .take(row.top_count())
                .map(|entry| (entry.token_id(), entry.logprob()))
                .collect(),
        })),
        Err(SamplingError::InvalidLogit { .. }) => Ok(Err(RowFailure::InvalidLogit)),
        Err(SamplingError::AllTokensMasked) => Ok(Err(RowFailure::NoCandidate)),
        Err(error) => Err(refused(error)),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use kernel_core::sampling::{self as contract, RowRecord, RowSettings, TokenBias, TokenCount};

    use super::*;
    use crate::{
        MAX_HISTORY, MAX_LOGIT_BIASES, MAX_REPETITION_PENALTY, MAX_STOP_IDS, MAX_TOP_LOGPROBS,
        MAX_VOCABULARY, MIN_REPETITION_PENALTY, philox4x32_10,
    };

    fn row<'a>(logits: &'a [f32], settings: RowSettings) -> SamplingRow<'a> {
        SamplingRow {
            logits,
            settings,
            counts: &[],
            biases: &[],
            eos: &[],
            mask: None,
        }
    }

    fn sampled(row: &SamplingRow<'_>) -> Sampled {
        sample_row(row).unwrap().unwrap()
    }

    /// The logprob of each token, normalized over `admitted` in token order.
    fn logprobs(logits: &[f32], admitted: impl Fn(usize) -> bool) -> Vec<f64> {
        let raw = logits.iter().map(|&value| f64::from(value));
        let maximum = raw
            .clone()
            .enumerate()
            .filter(|&(token, _)| admitted(token))
            .map(|(_, value)| value)
            .fold(f64::NEG_INFINITY, f64::max);
        let sum = raw
            .clone()
            .enumerate()
            .filter(|&(token, _)| admitted(token))
            .map(|(_, value)| (value - maximum).exp())
            .sum::<f64>();
        raw.map(|value| (value - maximum) - sum.ln()).collect()
    }

    /// The tokens a row samples over many seeds.
    fn sampled_set(row: SamplingRow<'_>) -> BTreeSet<u32> {
        (0..256)
            .map(|seed| {
                let settings = RowSettings {
                    seed,
                    ..row.settings
                };
                sampled(&SamplingRow { settings, ..row }).token_id
            })
            .collect()
    }

    #[test]
    fn the_contract_limits_are_the_samplers() {
        assert_eq!(contract::MAX_WIDTH as usize, MAX_VOCABULARY);
        assert_eq!(contract::MAX_TOP_LOGPROBS as usize, MAX_TOP_LOGPROBS);
        assert_eq!(contract::MAX_EOS, MAX_STOP_IDS);
        assert_eq!(contract::MAX_HISTORY as usize, MAX_HISTORY);
        assert_eq!(contract::MAX_LOGIT_BIASES, MAX_LOGIT_BIASES);
        assert_eq!(
            contract::REPETITION_PENALTY,
            (MIN_REPETITION_PENALTY, MAX_REPETITION_PENALTY)
        );
        let bound = contract::MAX_LOGIT_BIAS;
        assert!(LogitBias::new(0, bound).is_ok() && LogitBias::new(0, -bound).is_ok());
        assert!(LogitBias::new(0, bound.next_up()).is_err());
        let penalty = contract::MAX_PENALTY;
        let base = SamplingParameters::new(contract::MAX_TEMPERATURE).unwrap();
        assert!(base.clone().with_penalties(penalty, -penalty, 1.0).is_ok());
        assert!(base.with_penalties(penalty.next_up(), 0.0, 1.0).is_err());
        assert!(SamplingParameters::new(contract::MAX_TEMPERATURE.next_up()).is_err());
    }

    #[test]
    fn greedy_rows_take_the_largest_processed_value_and_report_raw_logprobs() {
        let logits = [1.0, 3.0, 2.0, 2.5];
        let settings = RowSettings {
            top_logprobs: 2,
            frequency_penalty: 1.0,
            presence_penalty: 0.5,
            ..RowSettings::new(0.0, 7, 0)
        };
        let counts = [TokenCount {
            token_id: 1,
            prompt_seen: false,
            output_count: 1,
        }];
        let biases = [TokenBias {
            token_id: 2,
            bias: 0.5,
        }];
        // Token 1 falls to 1.5; tokens 2 and 3 tie at 2.5, and the lower wins.
        let row = SamplingRow {
            counts: &counts,
            biases: &biases,
            ..row(&logits, settings)
        };
        let expected = logprobs(&logits, |_| true);
        assert_eq!(
            sampled(&row),
            Sampled {
                token_id: 2,
                logprob: expected[2],
                top: vec![(1, expected[1]), (3, expected[3])],
            }
        );
    }

    #[test]
    fn repetition_divides_positive_values_and_multiplies_negative_ones() {
        let greedy = RowSettings {
            repetition_penalty: 2.0,
            presence_penalty: 2.0,
            ..RowSettings::new(0.0, 0, 0)
        };
        let prompt = [TokenCount {
            token_id: 0,
            prompt_seen: true,
            output_count: 0,
        }];
        let token = |logits: &[f32]| {
            sampled(&SamplingRow {
                counts: &prompt,
                ..row(logits, greedy)
            })
            .token_id
        };
        // 2 / 2 < 1.5, and -1 * 2 < -1.5; presence applies only to output.
        assert_eq!(token(&[2.0, 1.5]), 1);
        assert_eq!(token(&[-1.0, -1.5]), 1);
        assert_eq!(token(&[2.0, 0.9]), 0);
    }

    #[test]
    fn eos_masking_removes_candidates_but_not_logprobs() {
        let logits = [5.0, 1.0, 0.0];
        let settings = RowSettings {
            top_logprobs: 1,
            ..RowSettings::new(0.0, 0, 0)
        };
        let masked = RowSettings {
            mask_eos: true,
            ..settings
        };
        let expected = logprobs(&logits, |_| true);
        assert_eq!(
            sampled(&SamplingRow {
                eos: &[0],
                ..row(&logits, masked)
            }),
            Sampled {
                token_id: 1,
                logprob: expected[1],
                top: vec![(0, expected[0])],
            }
        );
        // The list is read only with the EOS mask.
        let unmasked = SamplingRow {
            eos: &[0],
            ..row(&logits, settings)
        };
        assert_eq!(sampled(&unmasked).token_id, 0);
        let all = SamplingRow {
            eos: &[2, 1, 0],
            ..row(&logits, masked)
        };
        assert_eq!(sample_row(&all), Ok(Err(RowFailure::NoCandidate)));
    }

    #[test]
    fn masked_tokens_take_no_part_but_still_fail_when_invalid() {
        let logits = [9.0, 1.0, 2.0, 0.5];
        let mask = [0b1010];
        let settings = RowSettings {
            top_logprobs: 4,
            ..RowSettings::new(0.0, 0, 0)
        };
        let masked = SamplingRow {
            mask: Some(&mask),
            ..row(&logits, settings)
        };
        let expected = logprobs(&logits, |token| token % 2 == 1);
        assert_eq!(
            sampled(&masked),
            Sampled {
                token_id: 1,
                logprob: expected[1],
                top: vec![(1, expected[1]), (3, expected[3])],
            }
        );
        let invalid = [f32::NAN, 1.0, 2.0, 0.5];
        assert_eq!(
            sample_row(&SamplingRow {
                logits: &invalid,
                ..masked
            }),
            Ok(Err(RowFailure::InvalidLogit))
        );
        let infinite = [9.0, f32::INFINITY];
        assert_eq!(
            sample_row(&row(&infinite, settings)),
            Ok(Err(RowFailure::InvalidLogit))
        );
        assert_eq!(
            sample_row(&SamplingRow {
                mask: Some(&[0]),
                ..masked
            }),
            Ok(Err(RowFailure::NoCandidate))
        );
    }

    /// The contract's Gumbel noise `g(t)`.
    fn gumbel(seed: u64, generated: u64, token: u32) -> f64 {
        let x = philox4x32_10(seed, generated, token);
        let bits = (u64::from(x[0]) << 21) | (u64::from(x[1]) >> 11);
        let unit = (bits as f64 + 0.5) * 2_f64.powi(-53);
        -(-unit.ln()).ln()
    }

    #[test]
    fn the_gumbel_choice_follows_the_contract() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let logits = (0..300)
            .map(|_| (next(&mut state) % 2001) as f32 / 256.0 - 4.0)
            .collect::<Vec<_>>();
        for (seed, generated, temperature) in [(1, 0, 0.7), (2, 5, 1.0), (u64::MAX, 1 << 40, 1.3)] {
            let settings = RowSettings::new(temperature, seed, generated);
            let scaled = logits
                .iter()
                .map(|&value| f64::from(value) / temperature)
                .collect::<Vec<_>>();
            let maximum = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut best = (f64::NEG_INFINITY, 0);
            for (token, value) in scaled.iter().enumerate() {
                let score = (value - maximum) + gumbel(seed, generated, token as u32);
                if score > best.0 {
                    best = (score, token as u32);
                }
            }
            assert_eq!(sampled(&row(&logits, settings)).token_id, best.1);
        }
    }

    #[test]
    fn the_smallest_temperature_samples_as_the_served_simple_path_does() {
        let temperature = contract::MIN_TEMPERATURE;
        // One tile of the served simple path: (raw - m) / T + g, lowest ID
        // among equal scores.
        let served = |logits: &[f32], seed: u64| {
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut best = (f64::NEG_INFINITY, 0);
            for (token, &raw) in logits.iter().enumerate() {
                let difference = f64::from(raw) - f64::from(maximum);
                let score = difference / temperature + gumbel(seed, 0, token as u32);
                assert!(score.is_finite());
                if score > best.0 {
                    best = (score, token as u32);
                }
            }
            best.1
        };
        let tied = [2.0, 1.0, 2.0, -f32::MAX];
        let extreme = [-f32::MAX, f32::MAX];
        let mut winners = BTreeSet::new();
        for seed in 0..32 {
            let settings = RowSettings::new(temperature, seed, 0);
            for logits in [&tied[..], &extreme[..]] {
                let token = sampled(&row(logits, settings)).token_id;
                assert_eq!(token, served(logits, seed));
                winners.insert(token);
            }
        }
        assert_eq!(winners, BTreeSet::from([0, 1, 2]));
        // The largest processed values a row can reach stay finite.
        let counts = [TokenCount {
            token_id: 1,
            prompt_seen: false,
            output_count: MAX_HISTORY as u32,
        }];
        let biases = [
            TokenBias {
                token_id: 0,
                bias: contract::MAX_LOGIT_BIAS,
            },
            TokenBias {
                token_id: 1,
                bias: -contract::MAX_LOGIT_BIAS,
            },
        ];
        let settings = RowSettings {
            repetition_penalty: contract::REPETITION_PENALTY.1,
            frequency_penalty: contract::MAX_PENALTY,
            presence_penalty: contract::MAX_PENALTY,
            ..RowSettings::new(temperature, 3, 0)
        };
        let penalized = SamplingRow {
            counts: &counts,
            biases: &biases,
            ..row(&[f32::MAX, -f32::MAX], settings)
        };
        assert_eq!(sampled(&penalized).token_id, 0);
        // Positive logits divided by the smallest repetition penalty: the
        // review's row, then every input at its extreme.
        let (smallest, largest) = contract::REPETITION_PENALTY;
        let seen = |token_id, output_count| TokenCount {
            token_id,
            prompt_seen: true,
            output_count,
        };
        let filters = |settings: RowSettings| {
            [
                settings,
                RowSettings {
                    top_p: 0.9,
                    ..settings
                },
                RowSettings {
                    top_p: 0.9,
                    min_p: 0.5,
                    top_k: 1,
                    ..settings
                },
            ]
        };
        let divided = RowSettings {
            repetition_penalty: smallest,
            ..RowSettings::new(temperature, 0, 0)
        };
        for settings in filters(divided) {
            let row = SamplingRow {
                counts: &[seen(0, 0)],
                ..row(&[f32::MAX, 0.0], settings)
            };
            assert_eq!(sampled(&row).token_id, 0, "{settings:?}");
        }
        let lifted = RowSettings {
            frequency_penalty: -contract::MAX_PENALTY,
            presence_penalty: -contract::MAX_PENALTY,
            ..divided
        };
        for settings in filters(lifted) {
            let row = SamplingRow {
                counts: &[seen(0, MAX_HISTORY as u32 - 1)],
                biases: &biases,
                ..row(&[f32::MAX, -f32::MAX], settings)
            };
            assert_eq!(sampled(&row).token_id, 0, "{settings:?}");
        }
        // Negative logits times the largest penalty, where every candidate,
        // the maximum included, takes the largest magnitude.
        let half = MAX_HISTORY as u32 / 2 - 1;
        let lowered = [
            TokenBias {
                token_id: 0,
                bias: -contract::MAX_LOGIT_BIAS,
            },
            TokenBias {
                token_id: 1,
                bias: -contract::MAX_LOGIT_BIAS,
            },
        ];
        let multiplied = RowSettings {
            repetition_penalty: largest,
            frequency_penalty: contract::MAX_PENALTY,
            presence_penalty: contract::MAX_PENALTY,
            ..RowSettings::new(temperature, 0, 0)
        };
        for settings in filters(multiplied) {
            let row = SamplingRow {
                counts: &[seen(0, half), seen(1, half)],
                biases: &lowered,
                ..row(&[-f32::MAX, -f32::MAX], settings)
            };
            assert!(matches!(sample_row(&row), Ok(Ok(_))), "{settings:?}");
        }
        // Below it the row is refused rather than reported without a candidate.
        let tiny = RowSettings::new(1e-320, 0, 0);
        assert!(sample_row(&row(&[1.0, 2.0], tiny)).is_err());
        let below = RowSettings::new(temperature.next_down(), 0, 0);
        assert!(sample_row(&row(&[1.0, 2.0], below)).is_err());
    }

    #[test]
    fn filters_keep_the_tokens_the_contract_states() {
        let flat = [0.0; 4];
        let plain = RowSettings::new(1.0, 0, 0);
        // Ties rank by token ID.
        let top_k = RowSettings { top_k: 2, ..plain };
        assert_eq!(sampled_set(row(&flat, top_k)), BTreeSet::from([0, 1]));
        // Terms 1, 1, 1, 1: the prefix of two reaches 0.5 * 4 exactly.
        let half = RowSettings {
            top_p: 0.5,
            ..plain
        };
        assert_eq!(sampled_set(row(&flat, half)), BTreeSet::from([0, 1]));
        let more = RowSettings {
            top_p: 0.5_f64.next_up(),
            ..plain
        };
        assert_eq!(sampled_set(row(&flat, more)), BTreeSet::from([0, 1, 2]));
        // ln(min_p) = -2 keeps tokens within two of the maximum, after the
        // temperature scales them.
        let logits = [0.0, -1.0, -3.0];
        let min_p = RowSettings {
            min_p: (-2.0_f64).exp(),
            ..plain
        };
        assert_eq!(sampled_set(row(&logits, min_p)), BTreeSet::from([0, 1]));
        let hot = RowSettings {
            temperature: 2.0,
            ..min_p
        };
        assert_eq!(sampled_set(row(&logits, hot)), BTreeSet::from([0, 1, 2]));
        // Greedy rows ignore every filter.
        let greedy = RowSettings {
            temperature: 0.0,
            top_k: 1,
            top_p: 0.1,
            min_p: 1.0,
            ..plain
        };
        assert_eq!(sampled(&row(&[0.0, 1.0], greedy)).token_id, 1);
    }

    #[test]
    fn rows_with_the_most_biases_are_sampled() {
        let logits = vec![0.0_f32; contract::MAX_LOGIT_BIASES + 1];
        let biases = (0..=contract::MAX_LOGIT_BIASES as u32)
            .map(|token_id| TokenBias {
                token_id,
                bias: if token_id == 7 { 1.0 } else { -1.0 },
            })
            .collect::<Vec<_>>();
        let greedy = RowSettings::new(0.0, 0, 0);
        let biased = |count: usize| {
            sample_row(&SamplingRow {
                biases: &biases[..count],
                ..row(&logits, greedy)
            })
        };
        assert_eq!(
            biased(contract::MAX_LOGIT_BIASES)
                .unwrap()
                .unwrap()
                .token_id,
            7
        );
        assert!(biased(contract::MAX_LOGIT_BIASES + 1).is_err());
    }

    #[test]
    fn an_eos_mask_needs_an_output_shorter_than_the_longest_history() {
        let settings = RowSettings {
            mask_eos: true,
            ..RowSettings::new(0.0, 0, 0)
        };
        let masked = |output_count: u32| {
            let counts = [TokenCount {
                token_id: 0,
                prompt_seen: false,
                output_count,
            }];
            sample_row(&SamplingRow {
                counts: &counts,
                eos: &[1],
                ..row(&[1.0, 2.0], settings)
            })
        };
        let longest = MAX_HISTORY as u32;
        assert_eq!(masked(longest - 1).unwrap().unwrap().token_id, 0);
        assert!(masked(longest).is_err());
    }

    #[test]
    fn rows_outside_the_contract_are_refused() {
        let logits = [0.0; 4];
        let counts = [
            TokenCount {
                token_id: 2,
                prompt_seen: true,
                output_count: 0,
            },
            TokenCount {
                token_id: 1,
                prompt_seen: true,
                output_count: 0,
            },
        ];
        let plain = RowSettings::new(1.0, 0, 0);
        assert!(
            sample_row(&SamplingRow {
                counts: &counts,
                ..row(&logits, plain)
            })
            .is_err()
        );
        assert!(
            sample_row(&row(
                &logits,
                RowSettings {
                    top_logprobs: 21,
                    ..plain
                }
            ))
            .is_err()
        );
    }

    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// A request's inputs and the contract row the host would build from
    /// them.
    struct Request {
        logits: Vec<f32>,
        parameters: SamplingParameters,
        state: SamplingState,
        history: Vec<u32>,
        emitted: usize,
        stops: Vec<u32>,
        mask: Option<Vec<u32>>,
    }

    impl Request {
        fn random(state: &mut u64) -> Request {
            let width = 1 + (next(state) % 200) as usize;
            let logits = (0..width)
                .map(|_| (next(state) % 4097) as f32 / 512.0 - 4.0)
                .collect::<Vec<_>>();
            let pick = |state: &mut u64, choices: &[f64]| {
                choices[(next(state) % choices.len() as u64) as usize]
            };
            let temperature = pick(state, &[0.0, 0.5, 1.0, 1.7]);
            let biases = (0..next(state) % 4)
                .map(|_| next(state) as u32 % width as u32)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(|token| LogitBias::new(token, pick(state, &[-3.0, 0.25, 2.0])).unwrap())
                .collect();
            let parameters = SamplingParameters::new(temperature)
                .unwrap()
                .with_top_k((next(state) % 5) as usize)
                .unwrap()
                .with_top_p(pick(state, &[1.0, 0.9, 0.5]))
                .unwrap()
                .with_min_p(pick(state, &[0.0, 0.05, 0.3]))
                .unwrap()
                .with_penalties(
                    pick(state, &[0.0, 0.5, -0.25]),
                    pick(state, &[0.0, 0.3, -1.0]),
                    pick(state, &[1.0, 1.3, 0.7]),
                )
                .unwrap()
                .with_logit_bias(biases)
                .unwrap()
                .with_top_logprobs((next(state) % 21) as usize)
                .unwrap()
                .with_minimum_new_tokens((next(state) % 4) as usize)
                .unwrap();
            let history = (0..next(state) % 12)
                .map(|_| next(state) as u32 % width as u32)
                .collect::<Vec<_>>();
            let emitted = (next(state) as usize) % (history.len() + 1);
            let stops = (0..next(state) % 3)
                .map(|_| next(state) as u32 % width as u32)
                .collect();
            let mask = (next(state) % 3 == 0).then(|| {
                let mut words = (0..width.div_ceil(32))
                    .map(|_| next(state) as u32)
                    .collect::<Vec<_>>();
                if width % 32 != 0 {
                    *words.last_mut().unwrap() &= (1 << (width % 32)) - 1;
                }
                words
            });
            Request {
                logits,
                parameters,
                state: SamplingState::new(next(state)).after(next(state) % 1000),
                history,
                emitted,
                stops,
                mask,
            }
        }

        fn sample(&self) -> Result<crate::Sample, SamplingError> {
            let mut context = SampleContext::new(&self.history, self.emitted, &self.stops);
            if let Some(mask) = &self.mask {
                context = context.with_allowed_tokens(mask);
            }
            let mut state = self.state;
            sample(&self.parameters, &self.logits, context, &mut state)
        }

        /// The host's encoding: counts when a penalty applies, and the EOS
        /// mask before the minimum length unless the mask admits only stops.
        fn counts(&self) -> Vec<TokenCount> {
            let parameters = &self.parameters;
            if parameters.repetition_penalty() == 1.0
                && parameters.frequency_penalty() == 0.0
                && parameters.presence_penalty() == 0.0
            {
                return Vec::new();
            }
            let prompt = self.history.len() - self.emitted;
            let mut seen = std::collections::BTreeMap::<u32, (bool, u32)>::new();
            for (index, &token) in self.history.iter().enumerate() {
                let entry = seen.entry(token).or_default();
                if index < prompt {
                    entry.0 = true;
                } else {
                    entry.1 += 1;
                }
            }
            seen.into_iter()
                .map(|(token_id, (prompt_seen, output_count))| TokenCount {
                    token_id,
                    prompt_seen,
                    output_count,
                })
                .collect()
        }

        fn mask_eos(&self) -> bool {
            self.emitted < self.parameters.minimum_new_tokens()
                && !self.mask.as_ref().is_some_and(|mask| {
                    (0..self.logits.len() as u32).all(|token| {
                        mask[token as usize / 32] & (1 << (token % 32)) == 0
                            || self.stops.contains(&token)
                    })
                })
        }
    }

    #[test]
    fn the_oracle_samples_as_the_request_sampler_does() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut outcomes = [0; 3];
        for _ in 0..3000 {
            let request = Request::random(&mut state);
            let parameters = &request.parameters;
            let counts = request.counts();
            let biases = parameters
                .logit_bias()
                .iter()
                .map(|bias| TokenBias {
                    token_id: bias.token_id(),
                    bias: bias.bias(),
                })
                .collect::<Vec<_>>();
            let mask_eos = request.mask_eos();
            let row = SamplingRow {
                logits: &request.logits,
                settings: RowSettings {
                    seed: request.state.seed(),
                    generated: request.state.generated(),
                    top_k: parameters.top_k() as u32,
                    top_logprobs: parameters.top_logprobs() as u32,
                    mask_eos,
                    temperature: parameters.temperature(),
                    top_p: parameters.top_p(),
                    min_p: parameters.min_p(),
                    repetition_penalty: parameters.repetition_penalty(),
                    frequency_penalty: parameters.frequency_penalty(),
                    presence_penalty: parameters.presence_penalty(),
                },
                counts: &counts,
                biases: &biases,
                eos: if mask_eos { &request.stops } else { &[] },
                mask: request.mask.as_deref(),
            };
            let outcome = sample_row(&row).unwrap();
            match (request.sample(), &outcome) {
                (Ok(expected), Ok(actual)) => {
                    outcomes[0] += 1;
                    assert_eq!(actual.token_id, expected.token_id());
                    assert_eq!(actual.logprob.to_bits(), expected.logprob().to_bits());
                    assert_eq!(actual.top.len(), row.top_count());
                    for (&(token, logprob), entry) in actual.top.iter().zip(expected.top_logprobs())
                    {
                        assert_eq!((token, logprob), (entry.token_id(), entry.logprob()));
                    }
                    // The host's masked tail follows the admitted entries.
                    let tail = &expected.top_logprobs()[actual.top.len()..];
                    assert!(tail.iter().all(|entry| !row.admits(entry.token_id())
                        && entry.logprob() == f64::NEG_INFINITY));
                    let record = RowRecord::sampled(actual);
                    assert_eq!(contract::check(&row, &outcome, &record), Ok(()));
                }
                (Err(SamplingError::AllTokensMasked), Err(RowFailure::NoCandidate)) => {
                    outcomes[1] += 1;
                }
                (expected, actual) => panic!("{expected:?} != {actual:?}"),
            }
            outcomes[2] += usize::from(mask_eos);
        }
        // The cases cover sampled rows, rows without a candidate and EOS masks.
        assert!(outcomes.iter().all(|&count| count > 0), "{outcomes:?}");
    }
}
