//! Coupled sampled drafts (`model_glm5next_contract::draft_argmax::Proposals::Coupled`):
//! each drafting request's noise from a batch's draft sampling records, and
//! each drafter run's noise records by draft row.
//!
//! The host's record of an item (`runtime_bundle_sdk::DraftSamplingV1`)
//! holds the request's seed, its temperature and the generated index `n`
//! of the item's first verify row; the target samples verify row `j` at
//! `n + j`. A drafting verify item's first draft, the extend pass's, fills
//! its second row and is kept only when it equals the target's sample at
//! its first row, so draft position `j` (`Plan::before[j]`) draws at
//! `n + j`. The records carry that index, so a captured draft step serves
//! every position (`model_glm5next_contract::draft_argmax`).

use std::collections::BTreeMap;

use kernel_core::draft_noise::NoiseRecord;
use model_glm5next_contract::contract::DRAFT_ITEMS;
use runtime_bundle_sdk::BatchRequestV1;

use super::{Plan, Run};

/// Each request the batch drafts in the bundle at a temperature the host
/// samples at, by request: its seed, the generated index of its first
/// verify row and its temperature. A batch without records (an older
/// host's, or a V3 to V5 batch) drafts every item greedily, as does a
/// record outside the sampler's temperatures.
pub fn requests(batch: &BatchRequestV1) -> BTreeMap<u64, NoiseRecord> {
    let Some(records) = batch.draft_sampling() else {
        return BTreeMap::new();
    };
    batch
        .items()
        .iter()
        .zip(records)
        .filter(|(item, record)| item.bundle_drafts() && record.sampled())
        .map(|(item, record)| {
            let noise = NoiseRecord {
                seed: record.seed(),
                generated: record.generated(),
                temperature: record.temperature(),
            };
            (item.request_id(), noise)
        })
        .filter(|(_, noise)| noise.validate().is_ok())
        .collect()
}

/// The noise records of `run`, `plan`'s run at draft `position`, one per
/// draft row in run order, then greedy records to [`DRAFT_ITEMS`]: each
/// draft's request's record at the position's generated index, greedy for
/// a request `requests` does not sample.
pub fn records(
    plan: &Plan,
    run: &Run,
    position: u32,
    requests: &BTreeMap<u64, NoiseRecord>,
) -> Vec<NoiseRecord> {
    let mut records = run
        .drafts
        .iter()
        .map(|draft| {
            plan.drafting
                .get(draft.item as usize)
                .and_then(|&(request, _, _)| requests.get(&request))
                .map_or(NoiseRecord::GREEDY, |noise| NoiseRecord {
                    generated: noise.index(position),
                    ..*noise
                })
        })
        .collect::<Vec<_>>();
    records.resize(records.len().max(DRAFT_ITEMS as usize), NoiseRecord::GREEDY);
    records
}

/// `records` as the little-endian words the drafter run uploads.
pub fn words(records: &[NoiseRecord]) -> Vec<i32> {
    records
        .iter()
        .flat_map(NoiseRecord::encode)
        .collect::<Vec<_>>()
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use bundle_runtime::inputs::PhaseInputs;
    use runtime_bundle_sdk::{
        BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1, DraftSamplingV1,
    };

    use super::*;
    use crate::drafter::Drafter;

    fn sampled(seed: u64, generated: u64, temperature: f64) -> DraftSamplingV1 {
        DraftSamplingV1::new(seed, generated, temperature, 1.0, 0.0, 0, 0).unwrap()
    }

    /// Requests 7 and 9 drafted in the bundle and request 8 verifying host
    /// drafts, with `records` for the three items, or none.
    fn batch(records: Option<[DraftSamplingV1; 3]>) -> BatchRequestV1 {
        let items = vec![
            BatchItemV1::verify(7, 4, true, vec![21]),
            BatchItemV1::verify(8, 2, false, vec![31, 32]),
            BatchItemV1::verify(9, 3, true, vec![41]),
        ];
        let appends = [(10, 4), (20, 2), (30, 3)]
            .map(|(previous, rows)| {
                let end = previous + rows;
                BatchAppendV1::new(previous, rows, end, vec![0], None)
            })
            .to_vec();
        let batch = BatchRequestV1::new(BatchWireVersionV1::V3, items, appends);
        match records {
            Some(records) => batch.with_draft_sampling(records.to_vec()),
            None => batch,
        }
    }

    /// Only bundle-drafted items the host samples take noise, with the
    /// seed, first verify row's index and temperature of their record.
    #[test]
    fn requests_take_the_records_of_sampled_bundle_drafted_items() {
        let records = [
            sampled(5, 40, 0.7),
            sampled(6, 50, 1.0),
            sampled(7, 60, 1.0),
        ];
        let noise = requests(&batch(Some(records)));
        assert_eq!(
            noise.into_iter().collect::<Vec<_>>(),
            [
                (
                    7,
                    NoiseRecord {
                        seed: 5,
                        generated: 40,
                        temperature: 0.7
                    }
                ),
                (
                    9,
                    NoiseRecord {
                        seed: 7,
                        generated: 60,
                        temperature: 1.0
                    }
                ),
            ]
        );
        let greedy = [
            DraftSamplingV1::GREEDY,
            sampled(6, 50, 1.0),
            DraftSamplingV1::GREEDY,
        ];
        assert!(requests(&batch(Some(greedy))).is_empty());
        assert!(requests(&batch(None)).is_empty());
        // A temperature the host sampler would not sample at drafts greedily.
        let hot = [
            sampled(5, 40, 1000.0),
            sampled(6, 50, 1.0),
            sampled(7, 60, 1.0),
        ];
        assert_eq!(requests(&batch(Some(hot))).keys().collect::<Vec<_>>(), [&9]);
    }

    /// Each draft row of each drafter run takes its request's record at
    /// its draft position: the extend pass's first drafts at the first
    /// verify row's index, each draft step's one later; greedy requests'
    /// rows and the rows past the drafts stay greedy.
    #[test]
    fn each_draft_row_draws_at_its_positions_index() {
        // Two drafting verify items of 4 and 3 rows (three and two drafts),
        // each one row past its frontier.
        let mut decode = PhaseInputs {
            decode: true,
            ..PhaseInputs::default()
        };
        for (item, (request, length, rows)) in [(7, 10, 4), (9, 30, 3)].into_iter().enumerate() {
            decode.item_order.push(item);
            decode.request_ids.push(request);
            decode.slot_ids.push(item as i32);
            decode.q_start.push(decode.token_ids.len() as i32);
            decode.q_len.push(rows);
            decode.prefix_len.push(length);
            decode.seq_lens.push(length + rows);
            decode.verify_rows.push(true);
            decode.emits_row.push(true);
            decode.prompt_rows.push(false);
            decode.block_tables.push(vec![0]);
            decode.last_positions.push(i64::from(length + rows - 1));
            decode
                .token_ids
                .extend(std::iter::repeat_n(1, rows as usize));
            decode.positions.extend(length..length + rows);
            decode
                .token_batch
                .extend(std::iter::repeat_n(item as i32, rows as usize));
        }
        let mut drafter = Drafter::new(2);
        drafter.restore(0, &[1; 10], 9).unwrap();
        drafter.restore(1, &[1; 30], 29).unwrap();
        let plan = drafter
            .plan(&PhaseInputs::default(), &decode, &[3, 2])
            .unwrap();
        assert_eq!(plan.before.len(), 3);
        let noise = BTreeMap::from([(
            7,
            NoiseRecord {
                seed: 11,
                generated: 100,
                temperature: 0.5,
            },
        )]);
        let at = |position: u32| records(&plan, &plan.before[position as usize], position, &noise);
        for position in 0..3 {
            let rows = at(position);
            assert_eq!(rows.len(), DRAFT_ITEMS as usize);
            let drafts = plan.before[position as usize].drafts.len();
            assert_eq!(drafts, if position < 2 { 2 } else { 1 });
            // Request 7, drafting item 0, has the first draft row.
            assert_eq!(plan.drafting[0].0, 7);
            assert_eq!(
                rows[0],
                NoiseRecord {
                    seed: 11,
                    generated: 100 + u64::from(position),
                    temperature: 0.5
                }
            );
            assert!(rows[1..].iter().all(|row| *row == NoiseRecord::GREEDY));
        }
        let words = words(&at(1));
        assert_eq!(words.len(), DRAFT_ITEMS as usize * 8);
        assert_eq!(words[..6], [11, 0, 101, 0, 0, 0x3fe0_0000]);
        assert!(words[8..].iter().all(|&word| word == 0));
    }

    /// Four ranks' shards of a synthetic vocabulary, as the head shards it.
    const SHARD: usize = 64;
    const VOCAB: usize = 4 * SHARD;

    /// A pseudo-random row of `VOCAB` logits keyed by `key`, about half of
    /// its rows peaked on a few tokens.
    fn row(mut key: u64, amplitude: f32) -> Vec<f32> {
        let mut next = || {
            key ^= key << 13;
            key ^= key >> 7;
            key ^= key << 17;
            key
        };
        let peaked = next() % 2 == 0;
        (0..VOCAB)
            .map(|_| {
                let value = (next() % 4001) as f32 / 1000.0 - 2.0;
                let peak = if peaked && next() % 32 == 0 { 4.0 } else { 0.0 };
                (value + peak) * amplitude
            })
            .collect()
    }

    fn key(history: &[u32]) -> u64 {
        history.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &token| {
            (hash ^ u64::from(token)).wrapping_mul(0x0100_0000_01b3) | 1
        })
    }

    /// The target's logits after `history`.
    fn target(history: &[u32]) -> Vec<f32> {
        row(key(history), 1.0)
    }

    /// The drafter's: the target's, perturbed, so it ranks the leading
    /// tokens much as the target does.
    fn drafter(history: &[u32]) -> Vec<f32> {
        let noise = row(key(history) ^ 0x5555_5555_5555_5555, 0.15);
        target(history)
            .iter()
            .zip(noise)
            .map(|(x, y)| x + y)
            .collect()
    }

    /// The host sampler's token after `history` at generated index
    /// `generated`.
    fn target_sample(history: &[u32], seed: u64, generated: u64, temperature: f64) -> u32 {
        runtime_sampling::sample(
            &runtime_sampling::SamplingParameters::new(temperature).unwrap(),
            &target(history),
            runtime_sampling::SampleContext::new(&[], 0, &[]),
            &mut runtime_sampling::SamplingState::new(seed).after(generated),
        )
        .unwrap()
        .token_id()
    }

    /// `tokens` tokens AR sampling emits after `prompt`.
    fn autoregressive(prompt: &[u32], seed: u64, temperature: f64, tokens: usize) -> Vec<u32> {
        let mut history = prompt.to_vec();
        for generated in 0..tokens as u64 {
            history.push(target_sample(&history, seed, generated, temperature));
        }
        history.split_off(prompt.len())
    }

    /// `tokens` tokens a speculative loop of `drafts` drafts per cycle
    /// emits after `prompt`, and the drafts it accepted and the cycles it
    /// ran. The drafter proposes from its own logits over the ranks'
    /// shards, under its request's noise at the generated index of each
    /// draft's position when `coupled` (`coupled_token`), else greedily;
    /// the host samples each verify row at its own index and keeps the
    /// drafts while each equals its row's sample, then emits its sample at
    /// the first other row.
    fn speculative(
        prompt: &[u32],
        seed: u64,
        temperature: f64,
        tokens: usize,
        drafts: u32,
        coupled: bool,
    ) -> (Vec<u32>, u32, u32) {
        let mut history = prompt.to_vec();
        let (mut accepted, mut cycles) = (0, 0);
        while history.len() < prompt.len() + tokens {
            let generated = (history.len() - prompt.len()) as u64;
            let noise = NoiseRecord {
                seed,
                generated,
                temperature,
            };
            let noise = match coupled && noise.sampled() {
                true => noise,
                false => NoiseRecord::GREEDY,
            };
            let mut proposed = history.clone();
            for position in 0..drafts {
                let logits = drafter(&proposed);
                let shards = logits.chunks(SHARD).collect::<Vec<_>>();
                let at = NoiseRecord {
                    generated: noise.index(position),
                    ..noise
                };
                let at = if noise.sampled() { at } else { noise };
                let draft =
                    model_glm5next_contract::draft_argmax::coupled_token(&shards, &at).unwrap();
                proposed.push(draft);
            }
            cycles += 1;
            for row in 0..=drafts as usize {
                let verified = &proposed[..history.len()];
                let sample = target_sample(verified, seed, generated + row as u64, temperature);
                let draft = proposed.get(history.len()).copied();
                history.push(sample);
                if row == drafts as usize || draft != Some(sample) {
                    break;
                }
                accepted += 1;
            }
        }
        history.truncate(prompt.len() + tokens);
        (history.split_off(prompt.len()), accepted, cycles)
    }

    /// Coupled drafts emit what AR sampling with the same seed emits, as
    /// greedy drafts do: the host keeps a draft only when it equals its
    /// row's sample. At temperature 1 they are accepted far more often,
    /// since the drafter's noisy argmax is the target's sample whenever it
    /// ranks the leading Gumbel scores as the target does; at temperature
    /// 0 they are the greedy drafts.
    #[test]
    fn coupled_drafts_emit_what_ar_sampling_emits() {
        let prompt = [3, 1, 4, 1, 5];
        let mut rates = Vec::new();
        for (seed, temperature) in [(7, 1.0), (0x9e37_79b9_7f4a_7c15, 1.0), (11, 0.7), (5, 0.0)] {
            let ar = autoregressive(&prompt, seed, temperature, 200);
            let (coupled, coupled_accepted, coupled_cycles) =
                speculative(&prompt, seed, temperature, 200, 3, true);
            let (greedy, greedy_accepted, greedy_cycles) =
                speculative(&prompt, seed, temperature, 200, 3, false);
            assert_eq!(coupled, ar, "seed {seed} temperature {temperature}");
            assert_eq!(greedy, ar, "seed {seed} temperature {temperature}");
            if temperature == 0.0 {
                assert_eq!(
                    (coupled_accepted, coupled_cycles),
                    (greedy_accepted, greedy_cycles)
                );
            }
            let rate = |accepted: u32, cycles: u32| f64::from(accepted) / f64::from(cycles);
            rates.push((
                temperature,
                rate(coupled_accepted, coupled_cycles),
                rate(greedy_accepted, greedy_cycles),
            ));
        }
        eprintln!("(temperature, coupled, greedy) accepted drafts per cycle: {rates:?}");
        for &(temperature, coupled, greedy) in &rates {
            if temperature == 1.0 {
                assert!(coupled > greedy + 0.5, "{rates:?}");
            }
        }
    }
}
