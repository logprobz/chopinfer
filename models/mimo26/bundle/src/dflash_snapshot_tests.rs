//! Snapshot bytes and cursor failures use the real bundle copy path. Numerical
//! DFlash arithmetic remains a GPU check; fake readback tokens are not an oracle.
use super::*;

fn fixture(fp8: bool) -> (Rc<FakeHost>, runtime_bundle_abi::BundleInstanceV1) {
    fixture_serving(fp8, 32)
}

/// A DFlash instance whose context cache holds one slot per item of the
/// largest bucket for `served` sequences.
fn fixture_serving(fp8: bool, served: u32) -> (Rc<FakeHost>, runtime_bundle_abi::BundleInstanceV1) {
    let host = Rc::new(FakeHost::default());
    host.pool_pages.set(Some(256));
    host.fp8_kv.set(fp8);
    host.served.set(served);
    host.dflash.set(true);
    host.speculation.set(Some((7, 32)));
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_serving(Tp::TP4, host.kv(), true, served),
        )
        .unwrap();
    (host, instance)
}

#[test]
fn snapshot_metadata_is_checked_on_every_rank_and_a_refused_restore_can_retry() {
    for (fp8, served) in [(false, 32), (true, 32), (false, 100), (true, 256)] {
        for inconsistent in [false, true] {
            let (host, mut instance) = fixture_serving(fp8, served);
            instance
                .execute(
                    &dflash_step(0, vec![40; 68], None),
                    &mut DraftSink::default(),
                )
                .unwrap();
            instance.commit().unwrap();
            let pages = (20..20 + instance.describe_prefix_state().unwrap())
                .rev()
                .collect::<Vec<_>>();
            assert_eq!(pages.len(), if fp8 { 149 } else { 75 });
            let before = host.device_copies.borrow().len();
            instance.capture_prefix_state(7, 68, &pages).unwrap();
            let headers = host.device_copies.borrow()[before..]
                .iter()
                .filter(|(from, _)| from.length == crate::prefix_projected::HEADER_BYTES)
                .map(|(_, to)| *to)
                .collect::<Vec<_>>();
            assert_eq!(headers.len(), 4);
            // Either one rank disagrees, or all ranks carry the same invalid
            // cursor beyond the committed boundary. Both must be refused.
            let damaged = if inconsistent {
                &headers[..1]
            } else {
                &headers[..]
            };
            let saved = host.snapshot_memory.borrow().clone();
            for header in damaged {
                for (offset, byte) in 69_u64.to_le_bytes().into_iter().enumerate() {
                    host.snapshot_memory.borrow_mut().insert(
                        (header.buffer.raw, header.offset + 16 + offset as u64),
                        byte,
                    );
                }
            }
            assert!(instance.restore_prefix_state(8, &[40; 68], &pages).is_err());
            assert!(!instance.is_poisoned());
            assert!(host.prefix_in_flight.borrow().is_empty());
            *host.snapshot_memory.borrow_mut() = saved;
            instance.restore_prefix_state(8, &[40; 68], &pages).unwrap();
            instance.destroy().unwrap();
        }
    }
}

#[test]
fn projected_copy_and_header_readback_failures_drain_and_allow_retry() {
    for restore in [false, true] {
        for rank in 0..RANKS {
            let (host, mut instance) = fixture(true);
            instance
                .execute(
                    &dflash_step(0, vec![40; 68], None),
                    &mut DraftSink::default(),
                )
                .unwrap();
            instance.commit().unwrap();
            let pages = (20..20 + instance.describe_prefix_state().unwrap()).collect::<Vec<_>>();
            let before = host.device_copies.borrow().len();
            instance.capture_prefix_state(7, 68, &pages).unwrap();
            let per_rank = (host.device_copies.borrow().len() - before) / RANKS as usize;
            // Fail after some projected bytes reached the destination rank.
            host.fail_copy.set(Some(rank * per_rank as u32 + 1));
            let result = if restore {
                instance.restore_prefix_state(8, &[40; 68], &pages)
            } else {
                instance.capture_prefix_state(7, 68, &pages)
            };
            assert!(result.is_err());
            assert_eq!(host.fail_copy.get(), None);
            assert!(!instance.is_poisoned());
            assert!(host.prefix_in_flight.borrow().is_empty());
            if restore {
                host.prefix_fault.set(Some((PrefixFault::Readback, rank)));
                assert!(instance.restore_prefix_state(8, &[40; 68], &pages).is_err());
                assert_eq!(host.prefix_fault.get(), None);
                assert!(host.prefix_in_flight.borrow().is_empty());
                instance.restore_prefix_state(8, &[40; 68], &pages).unwrap();
            } else {
                instance.capture_prefix_state(7, 68, &pages).unwrap();
                instance.restore_prefix_state(8, &[40; 68], &pages).unwrap();
            }
            instance.destroy().unwrap();
        }
    }
}

#[test]
fn a_zero_draft_gap_and_a_fully_materialized_boundary_keep_their_cursor() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    for gap in [0_u64, 2048] {
        let (host, mut instance) = fixture(false);
        instance
            .execute(
                &dflash_step(0, vec![40; 64], None),
                &mut DraftSink::default(),
            )
            .unwrap();
        instance.commit().unwrap();
        *host.readback_tokens.borrow_mut() = vec![40; 4];
        instance
            .execute(
                &dflash_step(64, vec![40], Some(4)),
                &mut DraftSink::default(),
            )
            .unwrap();
        // Cancel leaves the committed projections materialized at position 64.
        instance.cancel().unwrap();
        if gap > 0 {
            instance
                .execute(
                    &dflash_step(64, vec![40; gap as usize], None),
                    &mut DraftSink::default(),
                )
                .unwrap();
            instance.commit().unwrap();
        }
        let boundary = 64 + gap;
        let pages = (20..20 + instance.describe_prefix_state().unwrap()).collect::<Vec<_>>();
        instance.capture_prefix_state(7, boundary, &pages).unwrap();
        instance
            .restore_prefix_state(8, &vec![40; boundary as usize], &pages)
            .unwrap();
        let step = |request| {
            BatchRequestV1::new(
                BatchWireVersionV1::V3,
                vec![BatchItemV1::verify(request, 4, true, vec![40])],
                vec![BatchAppendV1::new(
                    boundary,
                    4,
                    boundary + 4,
                    (0..(boundary + 4).div_ceil(PAGE_TOKENS)).collect(),
                    None,
                )],
            )
            .encode()
            .unwrap()
        };
        for (request, slot) in [(7, 0), (8, 1)] {
            let before = host.gather_launches.borrow().len();
            instance
                .execute(&step(request), &mut DraftSink::default())
                .unwrap();
            if gap == 0 {
                assert_eq!(
                    host.gather_launches.borrow().len(),
                    before,
                    "already projected rows must not be recomputed"
                );
            } else {
                let cells = (boundary - 1023..boundary)
                    .map(|p| crate::retain::cell(slot, p, crate::dflash::state::HISTORY_CAPACITY))
                    .collect::<Vec<_>>();
                assert_eq!(host.histories_gathering(&cells).len(), RANKS as usize);
            }
            instance.cancel().unwrap();
        }
        instance.destroy().unwrap();
    }
}
