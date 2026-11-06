//! An upper-skip prefill's failure contract: its body, row copies and head
//! are one unit, so a failure once any of it was submitted poisons the
//! instance, and a refusal before the first submission refuses the step,
//! settling the gathers it began.
use super::*;
use crate::instance::execute::skip;
use crate::program::Stage;

/// A TP4 host whose Engram tables are host-mapped, so a prefill's Engram
/// rows arrive through its prepared gathers, AR and DSpark alike.
fn host(dspark: bool) -> Rc<Host> {
    let host = super::host(dspark);
    host.drafter.set(true);
    host
}

/// Restores `request` in `slot` at `prefix` committed tokens.
fn restore(instance: &BundleInstanceV1, request: u64, slot: u32, prefix: u64) {
    let ids: Vec<_> = (0..prefix).map(|p| token(0, p)).collect();
    plan_timing::restore_without_snapshot(instance.handle(), request, slot, &ids)
        .unwrap_or_else(|e| panic!("restore: {e}"));
}

/// The step of the failure tests: a 512-row chunk after a 200,000-token
/// context, scoring its last row.
fn step() -> Vec<u8> {
    envelope(&[chunk(1, 0, 200_000, 512)], &[], false)
}

/// Rank 0's body launches and head symbols of `step` on an instance like
/// the tests'.
fn symbols(dspark: bool) -> (Vec<(String, Stage)>, Vec<String>) {
    let host = host(dspark);
    let mut instance = create(&host, false);
    restore(&instance, 1, 0, 200_000);
    let request = batch(&[chunk(1, 0, 200_000, 512)], &[], false).0;
    let planned = plan_timing::plan(instance.handle(), &request).unwrap_or_else(|e| panic!("{e}"));
    instance.destroy().unwrap();
    planned.symbols
}

/// The journal lines `step` adds on a fresh instance.
fn reference_journal(dspark: bool) -> Vec<String> {
    let host = host(dspark);
    let mut instance = create(&host, false);
    restore(&instance, 1, 0, 200_000);
    let begin = host.device.journal.borrow().len();
    instance
        .execute_prefill(&step(), &mut Sink::default())
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow().last()));
    let journal = host.device.journal.borrow()[begin..].to_vec();
    instance.commit().unwrap();
    instance.retire_request(1).unwrap();
    instance.destroy().unwrap();
    journal
}

/// Whether `instance` refuses every later call as a poisoned one does.
fn assert_poisoned(instance: &mut BundleInstanceV1, what: &str) {
    assert_eq!(
        instance.execute_prefill(
            &envelope(&[chunk(3, 2, 0, 512)], &[], false),
            &mut Sink::default()
        ),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState)),
        "{what}"
    );
    assert_eq!(
        instance.destroy(),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState)),
        "{what}"
    );
}

#[derive(Clone, Copy, Debug)]
enum After {
    /// A local refusal of the head's first segment on rank 0.
    Head,
    /// The last row copy failing.
    Copy,
}

/// After the body ran, the row copies and the head are part of the prefill's
/// unit: a local refusal of the head's first segment, which a unit of its
/// own would refuse unpoisoned, or a failed row copy poisons the instance.
/// With and without upper-skip DMA, AR and DSpark.
#[test]
fn failures_after_the_body_poison_the_instance() {
    for dspark in [false, true] {
        let (body, head) = symbols(dspark);
        let first = head.first().expect("a head launch").clone();
        assert!(
            body.iter().all(|(symbol, _)| *symbol != first),
            "{first} also runs in the body"
        );
        for dma in [true, false] {
            let reference = skip::with_dma(dma, || reference_journal(dspark));
            let head_at = reference
                .iter()
                .position(|line| line.starts_with("launch ") && line.contains(&first))
                .expect("the head launched");
            // The copy countdown counts device and peer copies alike.
            let copies = reference[..head_at]
                .iter()
                .filter(|line| line.starts_with("copy ") || line.starts_with("peer-copy "))
                .count() as u32;
            for after in [After::Head, After::Copy] {
                let what = format!("dspark={dspark} dma={dma} {after:?}");
                skip::with_dma(dma, || {
                    let host = host(dspark);
                    let mut instance = create(&host, false);
                    restore(&instance, 1, 0, 200_000);
                    let begin = host.device.journal.borrow().len();
                    let failed = match after {
                        After::Head => crate::instance::with_refused(&first, 0, || {
                            instance.execute_prefill(&step(), &mut Sink::default())
                        }),
                        After::Copy => {
                            host.device.fail_copy.set(Some(copies - 1));
                            let failed = instance.execute_prefill(&step(), &mut Sink::default());
                            host.device.fail_copy.set(None);
                            failed
                        }
                    };
                    assert!(failed.is_err(), "{what}");
                    let journal = host.device.journal.borrow()[begin..].to_vec();
                    assert!(
                        journal.iter().filter(|l| l.starts_with("launch ")).count() > 1000,
                        "{what}: the body ran"
                    );
                    assert!(
                        !journal
                            .iter()
                            .any(|l| l.starts_with("launch ") && l.contains(&first)),
                        "{what}: no head launched"
                    );
                    assert_poisoned(&mut instance, &what);
                });
            }
        }
    }
}

/// Without upper-skip DMA, a prefill with prepared gathers issues its body
/// in pieces between the gathers' joins. The pieces are one unit: a local
/// refusal of the first segment after the last join, which a piece's own
/// unit would refuse unpoisoned, poisons the instance.
#[test]
fn a_refusal_at_a_later_engram_piece_poisons_the_instance() {
    for dspark in [false, true] {
        let (body, _) = symbols(dspark);
        let layers = model_deepseek41_contract::geometry::FLASH
            .shape
            .engram
            .layers;
        let last = *layers.last().unwrap();
        let at = body
            .iter()
            .position(|(_, stage)| *stage == Stage::EngramRows(last))
            .expect("the last Engram consumer");
        let consumer = body[at].0.clone();
        // Each earlier occurrence is a segment on every rank.
        let earlier = body[..at].iter().filter(|(s, _)| *s == consumer).count() as u32;
        let what = format!("dspark={dspark}");
        skip::with_dma(false, || {
            let host = host(dspark);
            let mut instance = create(&host, false);
            restore(&instance, 1, 0, 200_000);
            let begin = host.device.journal.borrow().len();
            let failed = crate::instance::with_refused(&consumer, 4 * earlier, || {
                instance.execute_prefill(&step(), &mut Sink::default())
            });
            assert!(failed.is_err(), "{what}");
            let journal = host.device.journal.borrow()[begin..].to_vec();
            assert_eq!(
                journal
                    .iter()
                    .filter(|l| l.starts_with("async mapped wait "))
                    .count(),
                4 * layers.len(),
                "{what}: every layer joined before the refusal"
            );
            assert_poisoned(&mut instance, &what);
        });
    }
}

/// A refusal before the prefill's first submission refuses the step without
/// poisoning: the execute settles the gathers it began and cancels the
/// step, which frees the request's slot, and the instance serves the step
/// again once the request is restored. With DSpark a failed step is
/// terminal for the drafter's runtime, but the gathers are settled before
/// the execute returns and nothing is poisoned. The upper-skip DMA body
/// poisons on any failure, its first segment's refusal included.
#[test]
fn a_refusal_before_the_first_submission_settles_and_refuses() {
    for dspark in [false, true] {
        let (body, _) = symbols(dspark);
        let first = body.first().expect("a body launch").0.clone();
        for dma in [true, false] {
            let what = format!("dspark={dspark} dma={dma}");
            skip::with_dma(dma, || {
                let host = host(dspark);
                let mut instance = create(&host, false);
                restore(&instance, 1, 0, 200_000);
                let begin = host.device.journal.borrow().len();
                let failed = crate::instance::with_refused(&first, 0, || {
                    instance.execute_prefill(&step(), &mut Sink::default())
                });
                assert!(failed.is_err(), "{what}");
                let journal = host.device.journal.borrow()[begin..].to_vec();
                assert!(
                    !journal
                        .iter()
                        .any(|l| l.starts_with("launch ") && l.contains(&first)),
                    "{what}: no body launch"
                );
                if dma {
                    assert_poisoned(&mut instance, &what);
                    return;
                }
                // Every gather the step began completed before it returned.
                let count = |prefix: &str| journal.iter().filter(|l| l.starts_with(prefix)).count();
                let begun = count("async mapped ids ");
                assert!(begun > 0, "{what}: the step began its gathers");
                assert_eq!(count("async mapped complete"), begun, "{what}");
                assert!(!instance.is_poisoned(), "{what}");
                if dspark {
                    let begin = host.device.journal.borrow().len();
                    assert!(
                        instance
                            .execute_prefill(
                                &envelope(&[chunk(3, 2, 0, 512)], &[], false),
                                &mut Sink::default()
                            )
                            .is_err(),
                        "{what}: the drafter's runtime refuses after a failed step"
                    );
                    assert!(
                        host.device.journal.borrow()[begin..].iter().all(|line| {
                            !line.starts_with("launch ") && !line.starts_with("async mapped ids ")
                        }),
                        "{what}: the refused step issued nothing"
                    );
                    match instance.destroy() {
                        Ok(()) => assert_eq!(host.acquired.get(), host.released.get(), "{what}"),
                        Err(_) => assert!(instance.is_poisoned(), "{what}"),
                    }
                } else {
                    restore(&instance, 1, 0, 200_000);
                    instance
                        .execute_prefill(&step(), &mut Sink::default())
                        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    instance.commit().unwrap();
                    instance.retire_request(1).unwrap();
                    instance.destroy().unwrap();
                    assert_eq!(host.acquired.get(), host.released.get(), "{what}");
                }
            });
        }
    }
}

/// A miss keeps its template, and may drop the least recently used one,
/// while it plans, before any upload: a
/// step that then fails at its first upload has kept its template and made
/// its drop, and a later step of its key reuses the template.
#[test]
fn a_miss_that_fails_after_planning_has_kept_its_template() {
    plan_templates::with_limit(1, || {
        let host = host(false);
        let mut instance = create(&host, false);
        restore(&instance, 1, 0, 200_000);
        instance
            .execute_prefill(&step(), &mut Sink::default())
            .unwrap();
        instance.commit().unwrap();
        instance.retire_request(1).unwrap();
        let first = counts(&instance);
        assert_eq!((first.misses, first.kept, first.evicted), (1, 1, 0));
        // Another key (a 1,024-row chunk), failing at its first upload.
        restore(&instance, 2, 1, 100_032);
        let other = envelope(&[chunk(2, 1, 100_032, 1024)], &[], false);
        host.fail_upload.set(Some(0));
        assert!(
            instance
                .execute_prefill(&other, &mut Sink::default())
                .is_err()
        );
        host.fail_upload.set(None);
        assert!(!instance.is_poisoned());
        let failed = counts(&instance);
        assert_eq!(
            (failed.misses, failed.kept, failed.evicted),
            (2, 2, 1),
            "the failed step kept its template and dropped the first"
        );
        // The cancelled step freed the request's slot.
        restore(&instance, 2, 1, 100_032);
        instance
            .execute_prefill(&other, &mut Sink::default())
            .unwrap();
        instance.commit().unwrap();
        instance.retire_request(2).unwrap();
        assert_eq!(counts(&instance).hits, failed.hits + 1);
        instance.destroy().unwrap();
    });
}
