use super::*;
use crate::dspark::execution::Input;
use crate::dspark::prefetch::{Feedback, with_enabled};
use crate::dspark::state::{Kind, Rows};
use crate::instance::dspark_prefetch_for_test;
use runtime_bundle_abi::device_results::*;

fn feedback(request: u64, start: u64, accepted: u32) -> Feedback {
    let mut packet = vec![0; crate::dspark::results::SAMPLE_OFFSET as usize + 6 * 256];
    let result = BundleAcceptanceResultV1 {
        cycle_epoch: 1,
        accepted_drafts: accepted,
        emitted_rows: accepted + 1,
        flags: if accepted < 5 {
            BUNDLE_ACCEPTANCE_REJECTED_V1
        } else {
            0
        },
        status: 0,
        failure_row: BUNDLE_ACCEPTANCE_NO_FAILURE_V1,
        reserved: 0,
    };
    packet[..32].copy_from_slice(&result.to_le_bytes());
    let sample = crate::dspark::results::SAMPLE_OFFSET as usize + accepted as usize * 256;
    packet[sample..sample + 4].copy_from_slice(&7_u32.to_le_bytes());
    Feedback {
        inputs: vec![Input {
            rows: Rows {
                request,
                slot: 0,
                start,
                kind: Kind::Decode { drafts: 5 },
            },
            tokens: vec![7],
        }],
        metadata: DeviceCycleMetadataV1 {
            cycle_epoch: 1,
            termination_stop_ids: vec![],
            items: vec![DeviceCycleItemV1 {
                request_id: request,
                slot_generation: 1,
                acceptance: BundleAcceptanceInputV1 {
                    cycle_epoch: 1,
                    remaining_tokens: 100,
                    first_sample_row: 0,
                    draft_item_index: 0,
                    draft_tokens: 5,
                    stop_offset: 0,
                    stop_count: 0,
                    reserved: 0,
                },
            }],
        },
        packet,
    }
}

#[test]
fn private_successor_restores_current_state_and_skips_the_next_draft() {
    for ranks in [Ranks::TP1, Ranks::TP4] {
        for accepted in 0..=5 {
            let (host, mut instance) = with_enabled(true, || {
                create_with_limits(Shape::new(&FLASH, ranks).unwrap(), 1, 32)
            });
            execute(
                &mut instance,
                &host,
                &super::super::prompt(1, 0, 0, 3, true),
            );
            instance.commit().unwrap();
            execute(&mut instance, &host, &verify(1, 0, 3, 5));
            let predicted = feedback(1, 3, accepted);
            let (before_state, spans) =
                dspark_prefetch_for_test(instance.handle(), predicted.clone(), false);
            let before: Vec<_> = spans
                .iter()
                .map(|&span| host.memory.borrow().read(span))
                .collect();
            let begin = host.journal.borrow().len();
            let (after_state, _) = dspark_prefetch_for_test(instance.handle(), predicted, true);
            assert_eq!(
                before_state, after_state,
                "prediction must not commit a row or cursor"
            );
            for (&span, before) in spans.iter().zip(before) {
                assert_eq!(
                    host.memory.borrow().read(span),
                    before,
                    "current outputs and state must be restored"
                );
            }
            let journal = host.journal.borrow();
            assert!(
                !journal[begin..]
                    .iter()
                    .any(|line| line.starts_with("download ") || line.starts_with("synchronize "))
            );
            assert_eq!(
                journal[begin..]
                    .iter()
                    .filter(|line| line.starts_with("replay "))
                    .count(),
                ranks.count() * 2
            );
            assert_eq!(
                journal[begin..]
                    .iter()
                    .filter(|line| line.starts_with("upload "))
                    .count(),
                ranks.count() * 3,
                "one metadata upload for projection, one for drafting, and one bias selector"
            );
            drop(journal);
            instance.commit_accepted(&[kept(1, accepted + 1)]).unwrap();
            let start = 4 + u64::from(accepted);
            let begin = host.journal.borrow().len();
            // Every budget still computes the same five positions.
            execute(&mut instance, &host, &verify(1, 0, start, accepted.max(1)));
            let journal = host.journal.borrow();
            let step = &journal[begin..];
            let hash = step
                .iter()
                .position(|s| s.contains(crate::sm120::engram_hash_rows::SYMBOL))
                .unwrap();
            assert!(
                !step[..hash].iter().any(|s| s.starts_with("replay ")),
                "a matching successor reuses projection and draft"
            );
            drop(journal);
            instance.commit_accepted(&[kept(1, 1)]).unwrap();
            instance.retire_request(1).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

#[test]
fn a_different_host_commit_discards_the_prediction_and_runs_normal_work() {
    let (host, mut instance) = with_enabled(true, || {
        create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 1, 32)
    });
    execute(
        &mut instance,
        &host,
        &super::super::prompt(1, 0, 0, 3, true),
    );
    instance.commit().unwrap();
    execute(&mut instance, &host, &verify(1, 0, 3, 5));
    dspark_prefetch_for_test(instance.handle(), feedback(1, 3, 5), true);
    instance.commit_accepted(&[kept(1, 1)]).unwrap();
    let begin = host.journal.borrow().len();
    execute(&mut instance, &host, &verify(1, 0, 4, 5));
    let journal = host.journal.borrow();
    let step = &journal[begin..];
    let hash = step
        .iter()
        .position(|s| s.contains(crate::sm120::engram_hash_rows::SYMBOL))
        .unwrap();
    assert_eq!(
        step[..hash]
            .iter()
            .filter(|s| s.starts_with("replay "))
            .count(),
        8
    );
    drop(journal);
    instance.commit_accepted(&[kept(1, 1)]).unwrap();
    instance.retire_request(1).unwrap();
    instance.destroy().unwrap();
}

#[test]
fn cancellation_and_slot_reuse_cannot_consume_a_private_successor() {
    let (host, mut instance) = with_enabled(true, || {
        create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 1, 32)
    });
    execute(
        &mut instance,
        &host,
        &super::super::prompt(1, 0, 0, 3, true),
    );
    instance.commit().unwrap();
    execute(&mut instance, &host, &verify(1, 0, 3, 5));
    dspark_prefetch_for_test(instance.handle(), feedback(1, 3, 5), true);
    instance.cancel().unwrap();
    execute(
        &mut instance,
        &host,
        &super::super::prompt(2, 0, 0, 3, true),
    );
    instance.commit().unwrap();
    execute(&mut instance, &host, &verify(2, 0, 3, 5));
    instance.commit_accepted(&[kept(2, 1)]).unwrap();
    assert_eq!(state(&instance, 2).end, 4);
    instance.retire_request(2).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
