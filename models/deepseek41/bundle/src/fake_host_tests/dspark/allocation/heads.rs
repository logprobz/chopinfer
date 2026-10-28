use super::*;
use crate::dspark::{
    execution::{Input, Noise},
    prefetch::Feedback,
    state::{Kind, Rows},
};
use runtime_bundle_abi::device_results::*;

fn request(allocated_step: bool, start: u64) -> Vec<u8> {
    if allocated_step {
        return allocated(4, start, 3, 8).encode().unwrap();
    }
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (1..=4)
            .map(|id| BatchItemV1::verify(id, 4, true, vec![7]))
            .collect(),
        (0..4)
            .map(|slot| BatchAppendV1::new(start, 4, start + 4, vec![slot], None))
            .collect(),
    )
    .with_state_slots((0..4).collect())
    .encode()
    .unwrap()
}

fn predicted(allocated_step: bool) -> Feedback {
    let mut packet = vec![0; crate::dspark::results::SAMPLE_OFFSET as usize + 16 * 256];
    for index in 0..4 {
        let result = BundleAcceptanceResultV1 {
            cycle_epoch: 1,
            accepted_drafts: 0,
            emitted_rows: 1,
            flags: BUNDLE_ACCEPTANCE_REJECTED_V1
                | if allocated_step {
                    BUNDLE_ACCEPTANCE_ALLOCATED_V1
                } else {
                    0
                },
            status: 0,
            failure_row: BUNDLE_ACCEPTANCE_NO_FAILURE_V1,
            verified_drafts: if allocated_step { 1 } else { 0 },
        };
        packet[index * 32..index * 32 + 32].copy_from_slice(&result.to_le_bytes());
        let at = crate::dspark::results::SAMPLE_OFFSET as usize + index * 4 * 256;
        packet[at..at + 4].copy_from_slice(&7_u32.to_le_bytes());
    }
    Feedback {
        allocated: allocated_step,
        metadata: metadata(4, 3, 1),
        packet,
        inputs: (0..4)
            .map(|slot| Input {
                rows: Rows {
                    request: u64::from(slot + 1),
                    slot,
                    start: 3,
                    kind: Kind::Decode { drafts: 3 },
                },
                tokens: vec![7],
                noise: Noise::GREEDY,
            })
            .collect(),
    }
}

fn assert_heads(host: &Host, lines: &[String], positions: usize) {
    for stream in host.rank_streams.borrow().values() {
        let prefix = format!(
            "launch {stream} {} ",
            crate::sm120::confidence_parallel::SYMBOL
        );
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.starts_with(&prefix))
                .count(),
            positions,
            "every computed position needs its hidden state and Markov confidence launch"
        );
        let vocabulary = format!("launch {stream} chop_deepseek41_head_f32");
        let heads: Vec<_> = lines
            .iter()
            .filter(|line| line.starts_with(&vocabulary))
            .collect();
        assert!(!heads.is_empty());
        assert!(
            heads
                .iter()
                .all(|line| line.contains(&format!("ScalarU32({})", 4 * positions))),
            "{heads:#?}"
        );
    }
}

#[test]
fn allocated_depth_three_executes_five_heads_and_checks_successor_depth() {
    use model_deepseek41_contract::execution_options::DIAGNOSTIC_STEP_LOGS;
    for (first_allocated, next_allocated) in [(false, false), (true, true), (false, true)] {
        let (host, mut instance) = create_with_options(
            Shape::new(&FLASH, Ranks::TP4).unwrap(),
            32,
            64,
            &[(DIAGNOSTIC_STEP_LOGS, true.into())],
        );
        for slot in 0..4 {
            execute(
                &mut instance,
                &host,
                &super::super::super::prompt(slot + 1, slot, 0, 3, true),
            );
            instance.commit().unwrap();
        }
        for (cycle, allocated_step) in [(1, first_allocated), (2, next_allocated)] {
            let mut sink = MappedSink {
                inner: DeviceSink {
                    host: host.clone(),
                    metadata: Some(metadata(4, 3, cycle)),
                    legacy: Sink::default(),
                    drafts: vec![],
                    samples: vec![],
                    completion: None,
                },
                mapped: vec![],
            };
            let start = host.journal.borrow().len();
            instance
                .execute(&request(allocated_step, cycle + 2), &mut sink)
                .unwrap_or_else(|e| panic!("{e:?} {:?}", host.logs.borrow()));
            let lines = host.journal.borrow()[start..].to_vec();
            let hash = launch_of(
                &lines,
                host.rank_streams.borrow()[&0],
                crate::sm120::engram_hash_rows::SYMBOL,
            )
            .unwrap();
            if cycle == 1 || first_allocated != next_allocated {
                assert_heads(&host, &lines[..hash], if allocated_step { 5 } else { 3 });
            } else {
                assert!(
                    !lines[..hash].iter().any(|line| line.starts_with("replay ")),
                    "a successor with sufficient heads must be reused"
                );
            }
            if cycle == 1 {
                let start = host.journal.borrow().len();
                crate::instance::dspark_prefetch_for_test(
                    instance.handle(),
                    predicted(allocated_step),
                    true,
                );
                assert_heads(
                    &host,
                    &host.journal.borrow()[start..],
                    if allocated_step { 5 } else { 3 },
                );
            }
            instance
                .commit_accepted(
                    &(1..=4)
                        .map(|id| super::super::kept(id, 1))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        }
        let expected = if first_allocated == next_allocated {
            "reused"
        } else {
            "discarded"
        };
        assert!(
            host.logs
                .borrow()
                .iter()
                .any(|line| line.contains(&format!("\"event\":\"{expected}\"")))
        );
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
