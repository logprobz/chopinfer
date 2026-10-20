use super::*;
use crate::dspark::state::{Config, Rows};
use runtime_bundle_abi::device_results::*;

fn pending(items: u32, k: u32, start: u64) -> (State, Feedback) {
    let mut state = State::new(Config {
        slots: 8,
        stages: 3,
        window: 8,
        max_prefill_rows: 64,
        context_tokens: 4096,
        position_capacity: 4101,
    })
    .unwrap();
    let inputs: Vec<_> = (0..items)
        .map(|slot| Input {
            rows: Rows {
                request: u64::from(slot + 1),
                slot,
                start,
                kind: Kind::Decode { drafts: k },
            },
            tokens: vec![7],
        })
        .collect();
    state
        .prepare(
            inputs
                .iter()
                .map(|input| Rows {
                    start: 0,
                    kind: Kind::Prefill { rows: start as u32 },
                    ..input.rows
                })
                .collect(),
        )
        .unwrap();
    state.begin_execution().unwrap();
    state.executed().unwrap();
    let plan = state.begin_commit(&[]).unwrap();
    state.complete_commit(plan).unwrap();
    for input in &inputs {
        for stage in 0..3 {
            if let Some(pass) = state.projection(input.rows.request, stage, 64).unwrap() {
                state.projected(pass).unwrap();
            }
        }
    }
    state
        .prepare(inputs.iter().map(|i| i.rows).collect())
        .unwrap();
    state.begin_execution().unwrap();
    state.executed().unwrap();
    let metadata = DeviceCycleMetadataV1 {
        cycle_epoch: 9,
        termination_stop_ids: vec![],
        items: inputs
            .iter()
            .enumerate()
            .map(|(i, input)| DeviceCycleItemV1 {
                request_id: input.rows.request,
                slot_generation: 22,
                acceptance: BundleAcceptanceInputV1 {
                    cycle_epoch: 9,
                    remaining_tokens: 100,
                    first_sample_row: i as u32 * (k + 1),
                    draft_item_index: i as u32,
                    draft_tokens: k,
                    stop_offset: 0,
                    stop_count: 0,
                    reserved: 0,
                },
            })
            .collect(),
    };
    let packet = vec![
        0;
        crate::dspark::results::SAMPLE_OFFSET as usize
            + items as usize * (k as usize + 1) * 256
    ];
    (
        state,
        Feedback {
            metadata,
            inputs,
            packet,
        },
    )
}

fn outcome(feedback: &mut Feedback, depth: u32) {
    for (i, item) in feedback.metadata.items.iter().enumerate() {
        let record = BundleAcceptanceResultV1 {
            cycle_epoch: 9,
            accepted_drafts: depth,
            emitted_rows: depth + 1,
            flags: if depth < item.acceptance.draft_tokens {
                BUNDLE_ACCEPTANCE_REJECTED_V1
            } else {
                0
            },
            status: 0,
            failure_row: BUNDLE_ACCEPTANCE_NO_FAILURE_V1,
            reserved: 0,
        };
        feedback.packet[i * 32..i * 32 + 32].copy_from_slice(&record.to_le_bytes());
        let at = crate::dspark::results::SAMPLE_OFFSET as usize
            + (item.acceptance.first_sample_row + depth) as usize * 256;
        feedback.packet[at..at + 4].copy_from_slice(&(17 + i as u32).to_le_bytes());
    }
}

#[test]
fn predicts_every_acceptance_depth_without_committing_original_state() {
    for start in [3, 7, 16] {
        for depth in 0..=5 {
            let (state, mut feedback) = pending(1, 5, start);
            let original = state.clone();
            outcome(&mut feedback, depth);
            let plan = feedback.plan(&state, 100).unwrap().unwrap();
            assert_eq!(state, original);
            assert!(state.is_pending());
            assert_eq!(plan.drafts[0].block.main_rows, start + u64::from(depth) + 1);
            assert_eq!(plan.drafts[0].bonus, 17);
            assert_eq!(plan.projections[0].count, depth + 1);
            let mut actual = state;
            let committed = actual.begin_commit(&[(1, depth + 1)]).unwrap();
            actual.complete_commit(committed).unwrap();
            let next = [Input {
                rows: Rows {
                    start: start + u64::from(depth) + 1,
                    ..feedback.inputs[0].rows
                },
                tokens: vec![17],
            }];
            assert!(plan.matches(&actual, &next));
            let mut changed = next;
            changed[0].tokens[0] += 1;
            assert!(!plan.matches(&actual, &changed));
            changed[0].tokens = vec![17];
            changed[0].rows.kind = Kind::Decode { drafts: 0 };
            assert!(!plan.matches(&actual, &changed));
        }
    }
}

#[test]
fn preserves_c8_batch_bound_and_declines_oversized_projection() {
    let (state, mut feedback) = pending(8, 3, 3);
    outcome(&mut feedback, 3);
    let plan = feedback.plan(&state, 100).unwrap().unwrap();
    assert_eq!(plan.projections.iter().map(|p| p.count).sum::<u32>(), 32);
    let (state, mut feedback) = pending(8, 5, 3);
    outcome(&mut feedback, 5);
    assert!(feedback.plan(&state, 100).unwrap().is_none());
}

#[test]
fn terminal_limits_bad_samples_and_stale_acceptance_never_predict() {
    let (state, mut feedback) = pending(1, 5, 3);
    outcome(&mut feedback, 0);
    feedback.metadata.items[0].acceptance.remaining_tokens = 2;
    assert!(feedback.plan(&state, 100).unwrap().is_none());
    feedback.metadata.items[0].acceptance.remaining_tokens = 100;
    let mut record = BundleAcceptanceResultV1::from_le_bytes(&feedback.packet[..32]).unwrap();
    record.accepted_drafts = 1;
    record.flags = BUNDLE_ACCEPTANCE_TERMINAL_V1;
    feedback.packet[..32].copy_from_slice(&record.to_le_bytes());
    assert!(feedback.plan(&state, 100).unwrap().is_none());
    outcome(&mut feedback, 0);
    let sample = crate::dspark::results::SAMPLE_OFFSET as usize;
    feedback.packet[sample + 4..sample + 8].copy_from_slice(&2_u32.to_le_bytes());
    assert!(feedback.plan(&state, 100).unwrap().is_none());
    feedback.packet[sample + 4..sample + 8].fill(0);
    feedback.packet[..8].copy_from_slice(&8_u64.to_le_bytes());
    assert!(feedback.plan(&state, 100).unwrap().is_none());
    feedback.packet[..32].fill(0);
    assert!(feedback.plan(&state, 100).unwrap().is_none());
    feedback.packet[..32].fill(0xff);
    assert!(feedback.plan(&state, 100).unwrap().is_none());
}

#[test]
fn a_different_commit_or_reused_slot_cannot_consume_the_preview() {
    let (mut state, mut feedback) = pending(1, 5, 3);
    outcome(&mut feedback, 3);
    let plan = feedback.plan(&state, 100).unwrap().unwrap();
    let commit = state.begin_commit(&[(1, 3)]).unwrap();
    state.complete_commit(commit).unwrap();
    let next = [Input {
        rows: Rows {
            start: 7,
            ..feedback.inputs[0].rows
        },
        tokens: vec![17],
    }];
    assert!(!plan.matches(&state, &next));
    let mut state = plan.committed.clone();
    state.retire(1).unwrap();
    state
        .restore(
            1,
            0,
            crate::dspark::state::Snapshot {
                start: 0,
                end: 7,
                window: 8,
                stages: 3,
            },
        )
        .unwrap();
    assert!(!plan.matches(&state, &next));
}
