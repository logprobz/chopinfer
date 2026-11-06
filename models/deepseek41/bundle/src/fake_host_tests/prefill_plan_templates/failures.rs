//! An upper-skip prefill's failure contract: its body, row copies and head
//! are one unit, so a failure once any of it was submitted poisons the
//! instance, and a refusal before the first submission refuses the step,
//! settling the gathers it began. A plan no template serves is lowered
//! while it issues, and a step that fails keeps nothing of it. Each test
//! runs AR, and DSpark with the early Engram begin and plan.
use super::*;
use crate::instance::execute::skip;
use crate::program::Stage;
use model_deepseek41_contract::execution_options::PREFILL_PLAN_CACHE_VERIFY;

/// An instance like the tests'.
fn create(host: &Rc<Host>) -> BundleInstanceV1 {
    super::super::prefill_skip::create_profile_with_options(
        host,
        8192,
        true,
        &[(PREFILL_PLAN_CACHE_VERIFY, false.into())],
    )
}

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
    let mut instance = create(&host);
    restore(&instance, 1, 0, 200_000);
    let request = batch(&[chunk(1, 0, 200_000, 512)], &[], false).0;
    let planned = plan_timing::plan(instance.handle(), &request).unwrap_or_else(|e| panic!("{e}"));
    instance.destroy().unwrap();
    planned.symbols
}

/// The journal lines `step` adds on a fresh instance.
fn reference_journal(dspark: bool) -> Vec<String> {
    let host = host(dspark);
    let mut instance = create(&host);
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

/// Whether `instance`'s collectives are poisoned, and it refuses every
/// later call as a poisoned one does. A failed execute leaves the host's
/// view of the instance (`BundleInstanceV1::is_poisoned`) as it was.
fn assert_poisoned(instance: &mut BundleInstanceV1, what: &str) {
    assert!(
        plan_timing::poisoned(instance.handle()),
        "{what}: the collectives are poisoned"
    );
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
                    let mut instance = create(&host);
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
            let mut instance = create(&host);
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
                let mut instance = create(&host);
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
                assert!(!plan_timing::poisoned(instance.handle()), "{what}");
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

/// Where a step of `failed` fails.
#[derive(Clone, Debug)]
enum Fault {
    /// The device's `n`th launch of the step, counting from zero.
    Launch(u32),
    /// The `nth` eager segment launching the symbol refused (`with_refused`).
    Refused(String, u32),
}

/// What a step failing at a fault left: its journal, its status, whether the
/// instance is poisoned, and the templates kept.
type Failed = (Vec<String>, String, bool, u64);

/// `step` on a fresh instance, failing at `fault`.
fn failed(dspark: bool, fault: &Fault) -> Failed {
    let host = host(dspark);
    let mut instance = create(&host);
    restore(&instance, 1, 0, 200_000);
    let begin = host.device.journal.borrow().len();
    let mut execute = || instance.execute_prefill(&step(), &mut Sink::default());
    let status = match fault {
        Fault::Launch(n) => {
            host.device.fail_launch.set(Some(*n));
            let status = execute();
            host.device.fail_launch.set(None);
            status
        }
        Fault::Refused(symbol, nth) => crate::instance::with_refused(symbol, *nth, execute),
    };
    let journal = host.device.journal.borrow()[begin..].to_vec();
    let poisoned = plan_timing::poisoned(instance.handle());
    let failed = (
        journal,
        format!("{status:?}"),
        poisoned,
        counts(&instance).kept,
    );
    if poisoned {
        assert_poisoned(&mut instance, &format!("{fault:?}"));
    } else {
        let _ = instance.destroy();
    }
    failed
}

/// A streamed miss fails at a launch (the first unit's, the next rank's,
/// later units', the first two after the first DMA call's first event, the
/// head's and the last) and at a refused segment (the first, before any
/// submission, and the first Engram consumer's on rank 0, within a later
/// issue). Poisoning is read from the collectives, and a poisoned instance
/// refuses its next calls. Every fault but the first refusal without DMA
/// poisons, and the failed miss keeps no template.
#[test]
fn a_failed_streamed_miss_poisons_unless_refused_before_its_first_submission() {
    for dspark in [false, true] {
        let (body, head) = symbols(dspark);
        let first = body[0].0.clone();
        let layer = model_deepseek41_contract::geometry::FLASH
            .shape
            .engram
            .layers[0];
        let at = body
            .iter()
            .position(|(_, stage)| *stage == Stage::EngramRows(layer))
            .expect("the first Engram consumer");
        let consumer = body[at].0.clone();
        let earlier = body[..at].iter().filter(|(s, _)| *s == consumer).count() as u32;
        for dma in [true, false] {
            let reference = skip::with_dma(dma, || reference_journal(dspark));
            let launches: Vec<_> = reference
                .iter()
                .filter(|line| line.starts_with("launch "))
                .collect();
            let count = launches.len() as u32;
            let head_at = launches
                .iter()
                .position(|line| line.contains(&head[0]))
                .expect("the head launched") as u32;
            // A DSpark step launches its drafter's work first.
            let base = launches
                .iter()
                .position(|line| line.contains(&first))
                .expect("the body launched") as u32;
            let mut faults = vec![
                Fault::Launch(base),
                Fault::Launch(base + 1),
                Fault::Launch(base + (count - base) / 4),
                Fault::Launch(base + (count - base) / 2),
                Fault::Launch(head_at),
                Fault::Launch(count - 1),
                Fault::Refused(first.clone(), 0),
                Fault::Refused(consumer.clone(), 4 * earlier),
            ];
            if dma {
                // The first DMA call's first event, then its launches.
                let call = reference
                    .iter()
                    .position(|line| line.starts_with("record "))
                    .expect("a DMA call");
                let before = reference[..call]
                    .iter()
                    .filter(|line| line.starts_with("launch "))
                    .count() as u32;
                faults.push(Fault::Launch(before));
                faults.push(Fault::Launch(before + 1));
            }
            assert_eq!(faults.len(), if dma { 10 } else { 8 }, "dma={dma}");
            for fault in &faults {
                let what = format!("dspark={dspark} dma={dma} {fault:?}");
                let (_, status, poisoned, kept) = skip::with_dma(dma, || failed(dspark, fault));
                assert!(status.starts_with("Err"), "{what}: {status}");
                assert_eq!(kept, 0, "{what}: kept");
                // Only a refusal before the first submission leaves a body
                // without DMA unpoisoned.
                let refused_first = matches!(fault, Fault::Refused(symbol, 0) if *symbol == first);
                assert_eq!(poisoned, dma || !refused_first, "{what}: poisoned");
            }
        }
    }
}

/// A streamed miss's lowering or audit failure before its first submission
/// refuses the step, settling the gathers it began, and the step runs again once its request is restored; after it,
/// at a later group or in the head, it poisons the instance. A failed
/// streamed miss keeps no template.
#[test]
fn a_streamed_lowering_failure_refuses_before_the_first_submission_and_poisons_after() {
    use skip::StreamedFailure::{Group, Head};
    for dspark in [false, true] {
        let (body, _) = symbols(dspark);
        let first = body[0].0.clone();
        for dma in [true, false] {
            for failure in [Group(0), Group(10), Head] {
                let what = format!("dspark={dspark} dma={dma} {failure:?}");
                skip::with_dma(dma, || {
                    let host = host(dspark);
                    let mut instance = create(&host);
                    restore(&instance, 1, 0, 200_000);
                    let begin = host.device.journal.borrow().len();
                    let failed = skip::with_streamed_failure(failure, || {
                        instance.execute_prefill(&step(), &mut Sink::default())
                    });
                    assert!(failed.is_err(), "{what}");
                    assert_eq!(counts(&instance).kept, 0, "{what}");
                    let journal = host.device.journal.borrow()[begin..].to_vec();
                    let launched = journal
                        .iter()
                        .any(|l| l.starts_with("launch ") && l.contains(&first));
                    if failure != Group(0) {
                        assert!(launched, "{what}: the body launched");
                        assert_poisoned(&mut instance, &what);
                        return;
                    }
                    assert!(!launched, "{what}: no body launch");
                    let count =
                        |prefix: &str| journal.iter().filter(|l| l.starts_with(prefix)).count();
                    let begun = count("async mapped ids ");
                    assert!(begun > 0, "{what}: the step began its gathers");
                    assert_eq!(count("async mapped complete"), begun, "{what}");
                    assert!(!plan_timing::poisoned(instance.handle()), "{what}");
                    if dspark {
                        // A failed step is terminal for the drafter's runtime.
                        match instance.destroy() {
                            Ok(()) => assert_eq!(host.acquired.get(), host.released.get()),
                            Err(_) => assert!(instance.is_poisoned(), "{what}"),
                        }
                        return;
                    }
                    restore(&instance, 1, 0, 200_000);
                    instance
                        .execute_prefill(&step(), &mut Sink::default())
                        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    assert_eq!(counts(&instance).kept, 1, "{what}");
                    instance.commit().unwrap();
                    instance.retire_request(1).unwrap();
                    instance.destroy().unwrap();
                    assert_eq!(host.acquired.get(), host.released.get(), "{what}");
                });
            }
        }
    }
}

/// A streamed miss keeps its template only once its launch completed: a
/// step that fails at its first upload, before its launch, keeps nothing
/// and drops nothing from a full cache. Run again, it keeps its template
/// and drops the oldest, and a later step of its key reuses it.
#[test]
fn a_streamed_miss_that_fails_keeps_nothing() {
    plan_templates::with_limit(1, || {
        let host = host(false);
        let mut instance = create(&host);
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
        assert!(!plan_timing::poisoned(instance.handle()));
        let failed = counts(&instance);
        assert_eq!(
            (failed.misses, failed.kept, failed.evicted),
            (2, 1, 0),
            "the failed step kept nothing and dropped nothing"
        );
        restore(&instance, 2, 1, 100_032);
        instance
            .execute_prefill(&other, &mut Sink::default())
            .unwrap();
        instance.commit().unwrap();
        instance.retire_request(2).unwrap();
        let again = counts(&instance);
        assert_eq!((again.misses, again.kept, again.evicted), (3, 2, 1));
        restore(&instance, 3, 1, 100_160);
        instance
            .execute_prefill(
                &envelope(&[chunk(3, 1, 100_160, 1024)], &[], false),
                &mut Sink::default(),
            )
            .unwrap();
        instance.commit().unwrap();
        instance.retire_request(3).unwrap();
        assert_eq!(counts(&instance).hits, again.hits + 1);
        instance.destroy().unwrap();
    });
}

/// `step` lowered while it issues on a fresh instance, failing at
/// `failure`: its journal, whether its collectives are poisoned, and its
/// log lines.
fn streamed_failure(
    dspark: bool,
    dma: bool,
    failure: skip::StreamedFailure,
) -> (Vec<String>, bool, Vec<String>) {
    skip::with_dma(dma, || {
        let host = host(dspark);
        let mut instance = create(&host);
        restore(&instance, 1, 0, 200_000);
        let begin = host.device.journal.borrow().len();
        let logs = host.logs.borrow().len();
        let failed = skip::with_streamed_failure(failure, || {
            instance.execute_prefill(&step(), &mut Sink::default())
        });
        assert!(failed.is_err(), "{failure:?}");
        let journal = host.device.journal.borrow()[begin..].to_vec();
        let logs = host.logs.borrow()[logs..].to_vec();
        let poisoned = plan_timing::poisoned(instance.handle());
        assert_eq!(counts(&instance).kept, 0, "{failure:?}");
        let _ = instance.destroy();
        (journal, poisoned, logs)
    })
}

/// A streamed launch failing at `fail(group)` with an error that logs
/// `logged` stops where a lowering failure at that group does: the same
/// launches and poisoning, before the first submission (group 0) and
/// after the body's first join (group 10), with and without DMA.
#[track_caller]
fn assert_stops_at_its_group(fail: fn(usize) -> skip::StreamedFailure, logged: &str) {
    for dspark in [false, true] {
        let (body, _) = symbols(dspark);
        let first = body[0].0.clone();
        for dma in [true, false] {
            for group in [0, 10] {
                let what = format!("dspark={dspark} dma={dma} {:?}", fail(group));
                let (lowering, poisoned, _) =
                    streamed_failure(dspark, dma, skip::StreamedFailure::Group(group));
                let (failed, failed_poisoned, logs) = streamed_failure(dspark, dma, fail(group));
                assert!(
                    logs.iter().any(|line| line.contains(logged)),
                    "{what}: {logs:#?}"
                );
                let launches = |journal: &[String]| -> Vec<String> {
                    journal
                        .iter()
                        .filter(|line| line.starts_with("launch "))
                        .cloned()
                        .collect()
                };
                // A DSpark step launches its drafter's work first. A refused
                // step settles its gathers, joining them after the failure;
                // group 10 fails after the body's first join.
                let body_launched = lowering
                    .iter()
                    .any(|line| line.starts_with("launch ") && line.contains(&first));
                let joined = lowering
                    .iter()
                    .any(|line| line.starts_with("async mapped wait "));
                assert_eq!(body_launched, group != 0, "{what}");
                assert!(joined, "{what}");
                assert_eq!(launches(&failed), launches(&lowering), "{what}");
                assert_eq!(
                    (failed_poisoned, poisoned),
                    (group != 0, group != 0),
                    "{what}"
                );
            }
        }
    }
}

/// A streamed launch checks before each group's Engram plan that its
/// prepared gathers are still the ones it began and has joined. Gathers
/// that fail before a group, as a failed host callback leaves them, stop
/// the launch there (`assert_stops_at_its_group`).
#[test]
fn failed_gathers_stop_a_streamed_launch_where_a_lowering_failure_does() {
    assert_stops_at_its_group(
        skip::StreamedFailure::Gathers,
        "prefill Engram gathers changed while their plan was made",
    );
}

/// A launch that reads ids the prepared gathers still hold fails the
/// pending-span audit the Engram plan makes of each launch as it is
/// lowered, which stops the launch at that launch's group
/// (`assert_stops_at_its_group`).
#[test]
fn a_launch_reading_pending_gathers_stops_a_streamed_launch_at_its_group() {
    assert_stops_at_its_group(skip::StreamedFailure::Alias, "alias");
}

/// A panic while a streamed launch lowers poisons the instance, before its
/// first submission and after it: the host may have queued work on some
/// ranks that waits at a collective its peers never reach, and a panic is
/// a defect, not a refusal. The host sees the panic's status.
#[test]
fn a_panic_in_a_streamed_lowering_poisons_the_instance() {
    use skip::StreamedFailure::Panic;
    for dspark in [false, true] {
        for dma in [true, false] {
            for group in [0, 10] {
                let what = format!("dspark={dspark} dma={dma} group={group}");
                skip::with_dma(dma, || {
                    let host = host(dspark);
                    let mut instance = create(&host);
                    restore(&instance, 1, 0, 200_000);
                    let failed = skip::with_streamed_failure(Panic(group), || {
                        instance.execute_prefill(&step(), &mut Sink::default())
                    });
                    assert_eq!(
                        failed,
                        Err(BundleAbiError::Status(BundleStatusV1::Panic)),
                        "{what}"
                    );
                    assert!(plan_timing::poisoned(instance.handle()), "{what}");
                    assert_poisoned(&mut instance, &what);
                });
            }
        }
    }
}
