//! CPU planning for position-major MTP drafting.

use std::collections::BTreeSet;

use crate::contract::{DRAFT_ITEMS, MAX_VERIFY_ROWS, PREFILL_TOKENS};
use crate::inputs::PhaseInputs;
use crate::selection::Phase;

use super::staging::CatchUp;

/// A speculating sequence and its rows in the target verify input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sequence {
    pub catch_up: CatchUp,
    pub target_start: u32,
    pub drafts: u32,
}

/// One launch group, with packed input rows and one output per sequence.
#[derive(Debug)]
pub struct Batch {
    pub inputs: PhaseInputs,
    pub phase: Phase,
    /// Last hidden row for each sequence, before any head compaction.
    pub hidden_rows: Vec<u32>,
    /// Logits rows after the phase's head compaction.
    pub logits_rows: Vec<u32>,
    /// Target verify row receiving each argmax result.
    pub target_rows: Vec<u32>,
    /// Source slot for each packed sequence, also used to store chain state.
    pub slots: Vec<u32>,
}

/// Keeps original sequence order at every position. Inactive sequences never
/// consume a chain row or overwrite a target row at later draft positions.
#[cfg(test)]
pub fn plan(sequences: &[Sequence]) -> Result<Vec<Batch>, &'static str> {
    plan_for(crate::topology::Tp::TP4, sequences)
}
pub fn plan_for(
    tp: crate::topology::Tp,
    sequences: &[Sequence],
) -> Result<Vec<Batch>, &'static str> {
    if sequences.is_empty() || sequences.len() > DRAFT_ITEMS as usize {
        return Err("invalid MTP draft batch size");
    }
    let mut slots = BTreeSet::new();
    let mut target_rows = BTreeSet::new();
    for sequence in sequences {
        let catch_up = &sequence.catch_up;
        let end = catch_up.start.checked_add(catch_up.tokens.len() as u64);
        if catch_up.tokens.is_empty()
            || catch_up.tokens.len() > MAX_VERIFY_ROWS as usize
            || catch_up.slot >= DRAFT_ITEMS
            || !slots.insert(catch_up.slot)
            || sequence.drafts >= MAX_VERIFY_ROWS
            || end.is_none_or(|end| {
                end.checked_add(u64::from(sequence.drafts))
                    .is_none_or(|end| end > tp.geometry.context_tokens() as u64)
            })
            || catch_up
                .tokens
                .iter()
                .any(|&token| token >= tp.geometry.vocab())
        {
            return Err("invalid MTP draft sequence");
        }
        let last = sequence
            .target_start
            .checked_add(sequence.drafts)
            .filter(|&last| last < DRAFT_ITEMS)
            .ok_or("invalid MTP target rows")?;
        for row in sequence.target_start..=last {
            if !target_rows.insert(row) {
                return Err("overlapping MTP target rows");
            }
        }
    }
    let depth = sequences
        .iter()
        .map(|sequence| sequence.drafts)
        .max()
        .unwrap_or(0)
        .max(1);
    (0..depth)
        .map(|position| batch(sequences, position))
        .collect()
}

fn batch(sequences: &[Sequence], position: u32) -> Result<Batch, &'static str> {
    let mut inputs = PhaseInputs::default();
    let mut hidden_rows = Vec::new();
    let mut target_rows = Vec::new();
    let mut slots = Vec::new();
    for (index, sequence) in sequences.iter().enumerate() {
        if position > 0 && position >= sequence.drafts {
            continue;
        }
        let catch_up = &sequence.catch_up;
        let count = if position == 0 {
            catch_up.tokens.len()
        } else {
            1
        };
        let start = if position == 0 {
            catch_up.start
        } else {
            catch_up.start + catch_up.tokens.len() as u64 + u64::from(position) - 1
        };
        let row = inputs.live_tokens();
        let item = inputs.live_batch();
        inputs.item_order.push(index);
        inputs.request_ids.push(index as u64);
        inputs.emits_row.push(sequence.drafts > 0);
        inputs.prompt_rows.push(false);
        inputs.verify_rows.push(false);
        inputs.q_start.push(row as i32);
        inputs.q_len.push(count as i32);
        inputs.prefix_len.push(start as i32);
        inputs.seq_lens.push((start + count as u64) as i32);
        inputs
            .last_positions
            .push((start + count as u64 - 1) as i64);
        inputs.slot_ids.push(catch_up.slot as i32);
        inputs.block_tables.push(Vec::new());
        for token in 0..count {
            // Chain token IDs are filled from device-resident per-slot state.
            inputs.token_ids.push(if position == 0 {
                catch_up.tokens[token]
            } else {
                0
            });
            inputs.positions.push((start + token as u64) as i32);
            inputs.token_batch.push(item as i32);
        }
        hidden_rows.push(row + count as u32 - 1);
        if sequence.drafts > 0 {
            target_rows.push(sequence.target_start + position + 1);
            slots.push(catch_up.slot);
        }
    }
    let tokens = inputs.live_tokens();
    if tokens > PREFILL_TOKENS {
        return Err("MTP draft batch exceeds input capacity");
    }
    let phase = if inputs.q_len.iter().all(|&count| count == 1) {
        inputs.decode = true;
        Phase::Decode(tokens)
    } else if tokens <= crate::contract::DECODE_ENTRY_ROWS {
        inputs.decode = true;
        inputs.verify = true;
        Phase::Verify(tokens)
    } else {
        Phase::Prefill
    };
    let logits_rows = hidden_rows
        .iter()
        .enumerate()
        .filter(|(index, _)| inputs.emits_row[*index])
        .map(|(index, &row)| {
            if phase == Phase::Prefill {
                index as u32
            } else {
                row
            }
        })
        .collect();
    Ok(Batch {
        inputs,
        phase,
        hidden_rows,
        logits_rows,
        target_rows,
        slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection::MimoPhase as _;

    fn sequence(slot: u32, target_start: u32, drafts: u32, tokens: &[u32]) -> Sequence {
        Sequence {
            catch_up: CatchUp {
                slot,
                start: 10,
                tokens: tokens.to_vec(),
            },
            target_start,
            drafts,
        }
    }

    #[test]
    fn ragged_depth_preserves_slots_positions_and_target_rows() {
        let groups = plan(&[
            sequence(7, 0, 3, &[1, 2]),
            sequence(2, 4, 1, &[3]),
            sequence(9, 6, 2, &[4, 5, 6]),
        ])
        .unwrap();
        assert_eq!(groups.len(), 3);
        let first = &groups[0];
        assert_eq!(first.phase, Phase::Verify(6));
        assert_eq!(first.inputs.token_ids, [1, 2, 3, 4, 5, 6]);
        assert_eq!(first.inputs.token_batch, [0, 0, 1, 2, 2, 2]);
        assert_eq!(first.hidden_rows, [1, 2, 5]);
        assert_eq!(first.logits_rows, first.hidden_rows);
        assert_eq!(first.target_rows, [1, 5, 7]);
        let second = &groups[1];
        assert_eq!(second.phase, Phase::Decode(2));
        assert_eq!(second.inputs.item_order, [0, 2]);
        assert_eq!(second.slots, [7, 9]);
        assert_eq!(second.inputs.positions, [12, 13]);
        assert_eq!(second.target_rows, [2, 8]);
        assert_eq!(groups[2].slots, [7]);
        assert_eq!(groups[2].inputs.positions, [13]);
        assert_eq!(groups[2].target_rows, [3]);
        for group in groups {
            let layout = crate::upload::InputLayout::new::<crate::contract::Mimo>(
                group.phase.capacity().unwrap(),
                group.inputs.live_batch(),
            );
            assert!(layout.pack(&group.inputs, &group.hidden_rows).is_some());
        }
    }

    #[test]
    fn catch_up_above_token_bucket_uses_prefill_and_compacts_head_rows() {
        let sequences = (0..16)
            .map(|item| sequence(item, item * 2, 1, &[1; 8]))
            .collect::<Vec<_>>();
        let groups = plan(&sequences).unwrap();
        assert_eq!(groups[0].phase, Phase::Prefill);
        assert_eq!(groups[0].inputs.live_tokens(), 128);
        assert_eq!(groups[0].logits_rows, (0..16).collect::<Vec<_>>());
        assert_eq!(groups[0].hidden_rows[15], 127);
    }

    #[test]
    fn invalid_slots_depths_and_target_ranges_fail_before_launch() {
        assert!(plan(&[]).is_err());
        let first = sequence(0, 0, 3, &[1]);
        assert!(plan(&[first.clone(), sequence(0, 4, 1, &[1])]).is_err());
        assert!(plan(&[first, sequence(1, 3, 1, &[1])]).is_err());
        assert!(plan(&[sequence(0, 31, 1, &[1])]).is_err());
        assert!(plan(&[sequence(32, 0, 1, &[1])]).is_err());
        let plain = plan(&[sequence(0, 0, 0, &[1])]).unwrap();
        assert_eq!(plain.len(), 1);
        assert!(plain[0].target_rows.is_empty());
        assert!(plain[0].logits_rows.is_empty());
        assert!(plan(&[sequence(0, 0, MAX_VERIFY_ROWS, &[1])]).is_err());
        assert!(plan(&[sequence(0, 0, 1, &[])]).is_err());
        let mut overflow = sequence(0, 0, 1, &[1]);
        overflow.catch_up.start = u64::MAX;
        assert!(plan(&[overflow]).is_err());
    }
}
