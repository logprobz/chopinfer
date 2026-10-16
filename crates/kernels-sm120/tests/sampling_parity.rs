//! Rust mirror of the device row stages, checked against runtime-sampling.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use runtime_sampling::{SampleContext, SamplingParameters, SamplingState, philox4x32_10, sample};

fn reference(logits: &[f32], parameters: &SamplingParameters, seed: u64, step: u64) -> u32 {
    let mut adjusted = logits
        .iter()
        .map(|&value| f64::from(value))
        .collect::<Vec<_>>();
    if parameters.temperature() == 0.0 {
        return adjusted
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0 as u32;
    }
    for value in &mut adjusted {
        *value /= parameters.temperature();
    }
    let maximum = adjusted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut order = (0..logits.len())
        .filter(|&token| {
            parameters.min_p() == 0.0 || adjusted[token] - maximum >= parameters.min_p().ln()
        })
        .collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        adjusted[right]
            .total_cmp(&adjusted[left])
            .then_with(|| left.cmp(&right))
    });
    if parameters.top_k() != 0 {
        order.truncate(parameters.top_k());
    }
    if parameters.top_p() != 1.0 {
        let peak = adjusted[order[0]];
        let total = order
            .iter()
            .copied()
            .map(|token| (adjusted[token] - peak).exp())
            .sum::<f64>();
        let mut cumulative = 0.0;
        let mut kept = 0;
        for &token in &order {
            cumulative += (adjusted[token] - peak).exp();
            kept += 1;
            if cumulative >= parameters.top_p() * total {
                break;
            }
        }
        order.truncate(kept);
    }
    let mut best = f64::NEG_INFINITY;
    let mut chosen = u32::MAX;
    for token in order {
        let words = philox4x32_10(seed, step, token as u32);
        let bits = (u64::from(words[0]) << 21) | (u64::from(words[1]) >> 11);
        let unit = (bits as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0);
        let score = adjusted[token] - (-unit.ln()).ln();
        if score > best || (score == best && (token as u32) < chosen) {
            best = score;
            chosen = token as u32;
        }
    }
    chosen
}

fn next(word: &mut u64) -> u64 {
    *word ^= *word << 13;
    *word ^= *word >> 7;
    *word ^= *word << 17;
    *word
}

#[test]
fn device_reference_matches_cpu_oracle_on_ten_thousand_rows() {
    let modes = [
        SamplingParameters::new(0.0).unwrap(),
        SamplingParameters::new(0.7).unwrap(),
        SamplingParameters::new(0.8)
            .unwrap()
            .with_min_p(0.1)
            .unwrap(),
        SamplingParameters::new(1.0).unwrap().with_top_k(7).unwrap(),
        SamplingParameters::new(0.9)
            .unwrap()
            .with_top_p(0.8)
            .unwrap(),
        SamplingParameters::new(1.2)
            .unwrap()
            .with_min_p(0.03)
            .unwrap()
            .with_top_k(17)
            .unwrap()
            .with_top_p(0.95)
            .unwrap(),
    ];
    let seed = 0x1234_5678_9abc_def0;
    let mut state = SamplingState::new(seed);
    let mut random = 0xa5a5_3333_1234_9876;
    for step in 0..10_000 {
        let width = 8 + (next(&mut random) as usize % 56);
        let logits = (0..width)
            .map(|_| {
                let value = (next(&mut random) % 257) as f32 * 0.125 - 16.0;
                if step % 2 == 0 {
                    value
                } else {
                    bf16_to_f32(f32_to_bf16(value))
                }
            })
            .collect::<Vec<_>>();
        let parameters = &modes[step % modes.len()];
        let expected = reference(&logits, parameters, seed, step as u64);
        let actual = sample(
            parameters,
            &logits,
            SampleContext::new(&[], step, &[]),
            &mut state,
        )
        .unwrap()
        .token_id();
        assert_eq!(actual, expected, "row {step}");
    }
}
