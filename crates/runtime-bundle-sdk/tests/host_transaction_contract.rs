use std::cell::RefCell;

use runtime_bundle_sdk::abi::{BundleAbiError, BundleStatusV1, BundleStreamHandleV1};
use runtime_bundle_sdk::{
    BatchReadErrorV1, BatchReaderV1, StreamTransactionErrorV1, StreamTransactionPhaseV1,
    StreamTransactionV1, host_service_v1,
};

const RANKS: &[u32] = &[0, 1, 2, 3];

fn stream(rank: u32) -> BundleStreamHandleV1 {
    BundleStreamHandleV1::from_parts(rank + 1, 1)
}

#[test]
fn stream_acquisition_failure_rolls_back_every_prior_rank_in_reverse_order() {
    for failed_rank in RANKS {
        let events = RefCell::new(Vec::new());
        let failure = StreamTransactionV1::acquire(
            7,
            RANKS,
            RANKS.iter().map(|rank| (*rank, rank * 10)).collect(),
            |rank, kind| {
                events.borrow_mut().push(("acquire", rank));
                assert_eq!(kind, 7);
                if rank == *failed_rank {
                    Err("acquire")
                } else {
                    Ok(stream(rank))
                }
            },
            |rank, _| {
                events.borrow_mut().push(("release", rank));
                Ok(())
            },
        )
        .expect_err("the selected acquisition fails");
        assert_eq!(
            failure.error(),
            &StreamTransactionErrorV1::Host {
                service: "create_stream",
                rank: *failed_rank,
                error: "acquire",
            }
        );
        let releases: Vec<u32> = events
            .borrow()
            .iter()
            .filter_map(|(call, rank)| (*call == "release").then_some(*rank))
            .collect();
        assert_eq!(releases, (0..*failed_rank).rev().collect::<Vec<_>>());
        assert_eq!(failure.evidence().phase, StreamTransactionPhaseV1::Released);
    }
}

#[test]
fn launch_failures_are_marked_before_call_and_all_streams_close_in_reverse_order() {
    for failed_rank in RANKS {
        let events = RefCell::new(Vec::new());
        let mut transaction = StreamTransactionV1::acquire(
            11,
            RANKS,
            RANKS.iter().map(|rank| (*rank, rank * 10)).collect(),
            |rank, _| Ok::<_, &'static str>(stream(rank)),
            |_, _| Ok(()),
        )
        .unwrap();
        let result = transaction.launch_all(|rank, prepared, _| {
            events.borrow_mut().push(("launch", rank));
            assert_eq!(*prepared, rank * 10);
            if rank == *failed_rank {
                Err("launch")
            } else {
                Ok(())
            }
        });
        assert_eq!(
            result,
            Err(StreamTransactionErrorV1::Host {
                service: "launch",
                rank: *failed_rank,
                error: "launch",
            })
        );
        assert_eq!(
            transaction.evidence().potentially_enqueued_ranks,
            *failed_rank as usize + 1
        );
        let evidence = transaction
            .close(
                |rank, _| {
                    events.borrow_mut().push(("sync", rank));
                    Ok::<_, &'static str>(())
                },
                |rank, _| {
                    events.borrow_mut().push(("release", rank));
                    Ok::<_, &'static str>(())
                },
            )
            .unwrap();
        assert_eq!(evidence.phase, StreamTransactionPhaseV1::Released);
        let releases: Vec<u32> = events
            .borrow()
            .iter()
            .filter_map(|(call, rank)| (*call == "release").then_some(*rank))
            .collect();
        assert_eq!(releases, RANKS.iter().rev().copied().collect::<Vec<_>>());
    }
}

#[test]
fn every_sync_and_release_failure_is_retained_then_retryable() {
    for failed_rank in RANKS {
        let events = RefCell::new(Vec::new());
        let mut transaction = StreamTransactionV1::acquire(
            3,
            RANKS,
            RANKS.iter().map(|rank| (*rank, ())).collect(),
            |rank, _| Ok::<_, &'static str>(stream(rank)),
            |_, _| Ok(()),
        )
        .unwrap();
        transaction
            .launch_all(|_, _, _| Ok::<_, &'static str>(()))
            .unwrap();
        let failure = transaction
            .close(
                |rank, _| {
                    events.borrow_mut().push(("sync", rank));
                    if rank == *failed_rank {
                        Err("sync")
                    } else {
                        Ok(())
                    }
                },
                |rank, _| {
                    events.borrow_mut().push(("release", rank));
                    Ok(())
                },
            )
            .expect_err("the selected synchronization fails");
        assert_eq!(
            failure.evidence().phase,
            StreamTransactionPhaseV1::Quarantined
        );
        assert_eq!(failure.evidence().cleanup_failures, vec![*failed_rank]);
        transaction
            .retry_close(
                |rank, _| {
                    events.borrow_mut().push(("retry-sync", rank));
                    Ok::<_, &'static str>(())
                },
                |rank, _| {
                    events.borrow_mut().push(("retry-release", rank));
                    Ok::<_, &'static str>(())
                },
            )
            .unwrap();
        assert_eq!(
            transaction.evidence().phase,
            StreamTransactionPhaseV1::Released
        );
    }

    for failed_rank in RANKS {
        let mut transaction = StreamTransactionV1::acquire(
            3,
            RANKS,
            RANKS.iter().map(|rank| (*rank, ())).collect(),
            |rank, _| Ok::<_, &'static str>(stream(rank)),
            |_, _| Ok(()),
        )
        .unwrap();
        transaction
            .launch_all(|_, _, _| Ok::<_, &'static str>(()))
            .unwrap();
        let failure = transaction
            .close(
                |_, _| Ok::<_, &'static str>(()),
                |rank, _| {
                    if rank == *failed_rank {
                        Err("release")
                    } else {
                        Ok(())
                    }
                },
            )
            .expect_err("the selected release fails");
        assert_eq!(failure.evidence().retained_streams, 1);
        assert_eq!(failure.evidence().cleanup_failures, vec![*failed_rank]);
        transaction
            .retry_close(
                |_, _| Ok::<_, &'static str>(()),
                |_, _| Ok::<_, &'static str>(()),
            )
            .unwrap();
    }
}

#[test]
fn batch_reader_preserves_v1_little_endian_shape_and_refuses_bad_bounds() {
    let mut bytes = Vec::from(&b"chopbatch-v1"[..]);
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.push(1);
    bytes.extend_from_slice(&99_u64.to_le_bytes());
    let mut reader = BatchReaderV1::new("execution request", &bytes);
    reader.expect_magic(b"chopbatch-v1").unwrap();
    assert_eq!(reader.exact_count("items", 2), Ok(2));
    assert_eq!(reader.flag("logits"), Ok(true));
    assert_eq!(reader.u64(), Ok(99));
    assert_eq!(reader.finish(), Ok(()));

    let mut truncated = BatchReaderV1::new("execution request", &[1, 2, 3]);
    assert_eq!(
        truncated.u64(),
        Err(BatchReadErrorV1::Truncated {
            surface: "execution request",
            needed: 8,
            remaining: 3,
        })
    );
    let mut trailing = BatchReaderV1::new("execution request", &[0, 1]);
    assert_eq!(trailing.u8(), Ok(0));
    assert_eq!(
        trailing.finish(),
        Err(BatchReadErrorV1::TrailingBytes {
            surface: "execution request",
            remaining: 1,
        })
    );
}

#[test]
fn host_service_errors_preserve_source_and_map_to_host_error() {
    let mapped = host_service_v1::<()>(
        "create_stream",
        Some(3),
        Err(BundleAbiError::Status(BundleStatusV1::ResourceExhausted)),
    )
    .unwrap_err();
    assert_eq!(mapped.service(), "create_stream");
    assert_eq!(mapped.rank(), Some(3));
    assert_eq!(
        mapped.source_error(),
        &BundleAbiError::Status(BundleStatusV1::ResourceExhausted)
    );
    assert_eq!(mapped.status(), BundleStatusV1::HostError);
    assert!(mapped.to_string().contains("rank 3"));
}
#[test]
fn acquisition_cleanup_refusals_retain_ownership_for_retry() {
    let events = RefCell::new(Vec::new());
    let mut failure = StreamTransactionV1::acquire(
        5,
        RANKS,
        RANKS.iter().map(|rank| (*rank, ())).collect(),
        |rank, _| {
            if rank == 3 {
                Err("acquire")
            } else {
                Ok(stream(rank))
            }
        },
        |rank, _| {
            events.borrow_mut().push(("rollback", rank));
            if rank == 1 { Err("rollback") } else { Ok(()) }
        },
    )
    .expect_err("acquisition and one rollback fail");
    assert_eq!(
        events.borrow().as_slice(),
        &[("rollback", 2), ("rollback", 1), ("rollback", 0)]
    );
    assert_eq!(
        failure.evidence(),
        runtime_bundle_sdk::StreamTransactionEvidenceV1 {
            phase: StreamTransactionPhaseV1::Quarantined,
            retained_streams: 1,
            potentially_enqueued_ranks: 0,
            accepted_ranks: 0,
            synchronized_ranks: 0,
            cleanup_failures: vec![1],
        }
    );
    failure
        .retry_close(
            |_, _| panic!("an unlaunched stream must not synchronize"),
            |rank, _| {
                assert_eq!(rank, 1);
                Ok::<_, &'static str>(())
            },
        )
        .unwrap();
}
