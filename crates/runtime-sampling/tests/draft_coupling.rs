//! A coupled draft (`kernel_core::draft_noise`) is the host sampler's own
//! sample when the drafter's logits are the target's: the same Philox
//! noise at the same generated index, the same temperature, the same ties.

use kernel_core::draft_noise::{NoiseRecord, proposal};
use runtime_sampling::{SampleContext, SamplingParameters, SamplingState, sample};

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn target(logits: &[f32], temperature: f64, seed: u64, generated: u64) -> u32 {
    sample(
        &SamplingParameters::new(temperature).unwrap(),
        logits,
        SampleContext::new(&[], 0, &[]),
        &mut SamplingState::new(seed).after(generated),
    )
    .unwrap()
    .token_id()
}

#[test]
fn equal_logits_draft_the_targets_sample_at_every_position() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut sampled = std::collections::BTreeSet::new();
    for case in 0..48_u64 {
        // A spread row, so the noise and not the logits picks most tokens.
        let logits = (0..257)
            .map(|_| (next(&mut state) % 4001) as f32 / 1024.0 - 2.0)
            .collect::<Vec<_>>();
        let temperature = [0.6, 1.0, 1.7][case as usize % 3];
        let seed = next(&mut state);
        let generated = next(&mut state) % 10_000;
        let record = NoiseRecord {
            seed,
            generated,
            temperature,
        };
        for position in 0..5 {
            let draft = proposal(&logits, &record, position).unwrap();
            let expected = target(&logits, temperature, seed, generated + u64::from(position));
            assert_eq!(draft, expected, "case {case} position {position}");
            sampled.insert(draft);
        }
    }
    // The coupling follows the noise: the 240 drafts are far from one token.
    assert!(sampled.len() > 100);
}

#[test]
fn a_greedy_record_drafts_the_greedy_sample_with_lower_id_ties() {
    let mut logits = vec![0.25_f32; 64];
    logits[17] = 3.0;
    logits[41] = 3.0;
    let draft = proposal(&logits, &NoiseRecord::GREEDY, 4).unwrap();
    assert_eq!(draft, 17);
    assert_eq!(draft, target(&logits, 0.0, 9, 4));
}

#[test]
fn the_noise_index_moves_with_the_position_and_restarts_after_a_rejection() {
    let logits = vec![0.0_f32; 512];
    let record = |generated| NoiseRecord {
        seed: 11,
        generated,
        temperature: 1.0,
    };
    // A cycle drafting at generated index 100 that keeps two drafts and the
    // target's correction emits three tokens, so the next cycle drafts at
    // 103: its first draft uses the noise the first cycle's position 3 used.
    let first = (0..5)
        .map(|position| proposal(&logits, &record(100), position).unwrap())
        .collect::<Vec<_>>();
    let next = (0..5)
        .map(|position| proposal(&logits, &record(103), position).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(first[3..], next[..2]);
    for (position, &draft) in next.iter().enumerate() {
        assert_eq!(draft, target(&logits, 1.0, 11, 103 + position as u64));
    }
    assert_ne!(first, next);
}
