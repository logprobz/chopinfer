//! Internal execute rollback must either restore state or retain all resources.
use super::*;

struct RejectDraft {
    host: Rc<FakeHost>,
    fail_copy: Option<u32>,
    copies_before_restore: usize,
}

impl LogitsRowSinkV1 for RejectDraft {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(32, 152576, 32 * 152576)
    }

    fn emit_row(&mut self, _: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        unreachable!("drafts precede logits")
    }

    fn emit_draft_tokens(&mut self, _: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        self.copies_before_restore = self.host.copies.borrow().len();
        self.host.fail_copy.set(self.fail_copy);
        Err(BundleStatusV1::HostError)
    }
}

fn ready(dflash: bool) -> (Rc<FakeHost>, BundleInstanceV1, Vec<u8>) {
    ready_at(dflash, Tp::TP4)
}

fn ready_at(dflash: bool, tp: Tp) -> (Rc<FakeHost>, BundleInstanceV1, Vec<u8>) {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::at(tp));
    host.speculation.set(Some((7, 32)));
    host.dflash.set(dflash);
    let mut instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_serving(tp, &KV_FORMATS[0], true, 32),
        )
        .unwrap();
    instance
        .execute(
            &dflash_step(0, vec![40, 41, 42], None),
            &mut DraftSink::default(),
        )
        .unwrap();
    instance.commit().unwrap();
    *host.readback_tokens.borrow_mut() = vec![43, 44, 45, 46];
    (host, instance, dflash_step(3, vec![43], Some(4)))
}

fn failed_restore(dflash: bool, copy: u32) {
    let (host, mut instance, request) = ready(dflash);
    let mut sink = RejectDraft {
        host: Rc::clone(&host),
        fail_copy: Some(copy),
        copies_before_restore: 0,
    };
    assert_eq!(
        instance.execute(&request, &mut sink),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    assert_eq!(host.fail_copy.get(), None, "rollback reached the fault");
    assert_eq!(
        host.copies.borrow().len() - sink.copies_before_restore,
        copy as usize,
        "no copies follow the failed restore"
    );
    let journal = host.journal.borrow().len();
    let releases = host.released.get();
    let graph_releases = host.graph_releases.get();
    assert_eq!(
        instance.execute(&request, &mut DraftSink::default()),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    assert_eq!(
        bundle_runtime::entry::retire_request::<crate::instance::Instance>(instance.handle(), 7),
        BundleStatusV1::InvalidState
    );
    assert_eq!(
        instance.destroy(),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    assert!(instance.is_poisoned());
    assert_eq!(host.released.get(), releases);
    assert_eq!(host.graph_releases.get(), graph_releases);
    assert_eq!(host.journal.borrow().len(), journal);
}

#[test]
fn mtp_failed_internal_rollback_retains_resources() {
    for rank in 0..RANKS {
        // First restore the drafter ring, one copy on each rank.
        failed_restore(false, rank);
    }
    target_restore_faults(false);
}

#[test]
fn dflash_failed_internal_rollback_retains_resources() {
    target_restore_faults(true);
}

fn target_restore_faults(dflash: bool) {
    let drafter_copies = if dflash { 0 } else { RANKS };
    let copies_per_rank = 4 * SWA_LAYERS as u32 * 2;
    for rank in 0..RANKS {
        // Cover the first copy and a failure after almost a full rank restores.
        for offset in [0, copies_per_rank - 1] {
            failed_restore(dflash, drafter_copies + rank * copies_per_rank + offset);
        }
    }
}

#[test]
fn successful_internal_rollback_allows_the_identical_draft_step() {
    for dflash in [false, true] {
        let (host, mut instance, request) = ready(dflash);
        let mut sink = RejectDraft {
            host: Rc::clone(&host),
            fail_copy: None,
            copies_before_restore: 0,
        };
        assert_eq!(
            instance.execute(&request, &mut sink),
            Err(BundleAbiError::Status(BundleStatusV1::HostError))
        );
        assert!(!instance.is_poisoned());
        let mut sink = DraftSink::default();
        instance.execute(&request, &mut sink).unwrap();
        assert_eq!(sink.drafts.len(), 1);
        instance.commit_accepted(&[kept(7, 4)]).unwrap();
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
    }
}

#[test]
fn dflash_replay_requires_bundle_owned_proposals_at_full_and_partial_widths() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    for concurrency in [16_u32, 32] {
        let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
        let host = Rc::new(FakeHost::default());
        host.speculation.set(Some((7, 256)));
        host.dflash.set(true);
        host.served.set(32);
        let mut instance = api
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_serving(Tp::TP4, &KV_FORMATS[0], true, 32),
            )
            .unwrap();
        // Match the recorder's prompt length and production served capacity.
        // C16 still verifies 128/112 live rows after its 256-row catch-up passes.
        let prefill = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            (0..concurrency)
                .map(|slot| BatchItemV1::new(u64::from(slot + 1), false, 128, true, vec![40; 128]))
                .collect(),
            (0..concurrency)
                .map(|slot| {
                    BatchAppendV1::new(
                        0,
                        128,
                        128,
                        (0..128_u64.div_ceil(PAGE_TOKENS))
                            .map(|page| u64::from(slot) * 144_u64.div_ceil(PAGE_TOKENS) + page)
                            .collect(),
                        None,
                    )
                })
                .collect(),
        );
        instance
            .execute(&prefill.encode().unwrap(), &mut DraftSink::default())
            .unwrap();
        instance.commit().unwrap();
        for (index, width) in [8_u32, 7].into_iter().enumerate() {
            let start = 128 + index as u64;
            let request = |bundle_drafts| {
                BatchRequestV1::new(
                    BatchWireVersionV1::V3,
                    (0..concurrency)
                        .map(|slot| {
                            BatchItemV1::verify(
                                u64::from(slot + 1),
                                width,
                                bundle_drafts,
                                if bundle_drafts {
                                    vec![43]
                                } else {
                                    vec![43; width as usize]
                                },
                            )
                        })
                        .collect(),
                    (0..concurrency)
                        .map(|slot| {
                            BatchAppendV1::new(
                                start,
                                u64::from(width),
                                start + u64::from(width),
                                (0..(start + u64::from(width)).div_ceil(PAGE_TOKENS))
                                    .map(|page| {
                                        u64::from(slot) * 144_u64.div_ceil(PAGE_TOKENS) + page
                                    })
                                    .collect(),
                                None,
                            )
                        })
                        .collect(),
                )
                .encode()
                .unwrap()
            };
            // Reproduce the rejected host-proposal request before trying the
            // recorder's corrected bundle-owned request on the same instance.
            assert_eq!(
                instance.execute(&request(false), &mut DraftSink::default()),
                Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
            );
            *host.readback_tokens.borrow_mut() =
                (0..concurrency * width).map(|row| 1000 + row).collect();
            let mut sink = DraftSink::default();
            instance.execute(&request(true), &mut sink).unwrap();
            assert_eq!(sink.drafts.len(), concurrency as usize);
            for (slot, draft) in sink.drafts.iter().enumerate() {
                assert_eq!(draft.request_id, slot as u64 + 1);
                assert_eq!(
                    draft.tokens,
                    (1..width)
                        .map(|row| 1000 + slot as u32 * width + row)
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(
                sink.views
                    .iter()
                    .map(|view| view.raw().row_count)
                    .sum::<u32>(),
                concurrency * width
            );
            instance
                .commit_accepted(
                    &(0..concurrency)
                        .map(|slot| kept(u64::from(slot + 1), 1))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        }
        instance.destroy().unwrap();
    }
}

#[test]
fn retiring_during_a_draft_step_preserves_the_transaction() {
    for dflash in [false, true] {
        let (host, mut instance, request) = ready(dflash);
        instance
            .execute(&request, &mut DraftSink::default())
            .unwrap();
        let journal = host.journal.borrow().clone();
        assert_eq!(
            bundle_runtime::entry::retire_request::<crate::instance::Instance>(
                instance.handle(),
                7
            ),
            BundleStatusV1::InvalidState
        );
        assert_eq!(*host.journal.borrow(), journal);
        instance.commit_accepted(&[kept(7, 4)]).unwrap();
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
    }
}

// Raw callbacks isolate read-only ledger validation. The supported client
// poisons on any terminal callback error, including a validation refusal.
#[test]
fn raw_ledger_refusal_preserves_state() {
    for tp in [Tp::TP2, Tp::TP4] {
        for dflash in [false, true] {
            let (host, mut instance, request) = ready_at(dflash, tp);
            instance
                .execute(&request, &mut DraftSink::default())
                .unwrap();
            let journal = host.journal.borrow().clone();
            for records in [
                vec![],
                vec![kept(7, 0)],
                vec![kept(7, 5)],
                vec![kept(8, 2)],
                vec![kept(7, 2), kept(7, 2)],
            ] {
                assert_eq!(
                    bundle_runtime::entry::commit_accepted::<crate::instance::Instance>(
                        instance.handle(),
                        &records
                    ),
                    BundleStatusV1::InvalidArgument
                );
                assert_eq!(*host.journal.borrow(), journal);
            }
            crate::instance::set_test_tail_window(instance.handle(), Some(1));
            let journal = host.journal.borrow().clone();
            assert_eq!(
                bundle_runtime::entry::commit_accepted::<crate::instance::Instance>(
                    instance.handle(),
                    &[kept(7, 2)]
                ),
                BundleStatusV1::InvalidArgument
            );
            assert_eq!(*host.journal.borrow(), journal);
            instance.commit_accepted(&[kept(7, 4)]).unwrap();
            instance
                .execute(
                    &dflash_step(7, vec![47], Some(4)),
                    &mut DraftSink::default(),
                )
                .unwrap();
            instance.commit_accepted(&[kept(7, 4)]).unwrap();
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
        }
    }
}

/// A tail restore can fail after both drafter bookkeeping and the ledger
/// commit. The supported client contains that partial state by poisoning.
#[test]
fn late_commit_failure_poisons_and_blocks_further_work() {
    for dflash in [false, true] {
        let (control, mut committed, request) = ready(dflash);
        committed
            .execute(&request, &mut DraftSink::default())
            .unwrap();
        let before = control.launch_records.borrow().len();
        committed.commit_accepted(&[kept(7, 2)]).unwrap();
        let launches = control.launch_records.borrow()[before..].to_vec();
        assert!(launches.last().unwrap().1.contains("ring_restore"));
        let last = u32::try_from(launches.len() - 1).unwrap();
        committed.destroy().unwrap();

        let (host, mut instance, request) = ready(dflash);
        instance
            .execute(&request, &mut DraftSink::default())
            .unwrap();
        host.fail_launch.set(Some(last));
        assert_eq!(
            instance.commit_accepted(&[kept(7, 2)]),
            Err(BundleAbiError::Status(BundleStatusV1::HostError))
        );
        assert_eq!(
            host.fail_launch.get(),
            None,
            "late tail restore reached the fault"
        );
        assert!(instance.is_poisoned());
        let journal = host.journal.borrow().clone();
        let releases = (host.released.get(), host.graph_releases.get());
        assert_eq!(
            instance.execute(&request, &mut DraftSink::default()),
            Err(BundleAbiError::Poisoned)
        );
        assert_eq!(instance.commit(), Err(BundleAbiError::Poisoned));
        assert_eq!(
            instance.commit_accepted(&[kept(7, 2)]),
            Err(BundleAbiError::Poisoned)
        );
        assert_eq!(instance.cancel(), Err(BundleAbiError::Poisoned));
        assert_eq!(instance.retire_request(7), Err(BundleAbiError::Poisoned));
        assert_eq!(
            instance.capture_prefix_state(7, 3, &[]),
            Err(BundleAbiError::Poisoned)
        );
        assert_eq!(
            instance.restore_prefix_state(7, &[40, 41, 42], &[]),
            Err(BundleAbiError::Poisoned)
        );
        assert_eq!(instance.destroy(), Err(BundleAbiError::Poisoned));
        assert_eq!(*host.journal.borrow(), journal);
        assert_eq!((host.released.get(), host.graph_releases.get()), releases);
    }
}
