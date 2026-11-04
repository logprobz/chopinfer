//! A mixed step's verify rows beside an upper-skip DMA prefill, with the
//! `dspark_mixed_prepared_engram` candidate: the cycle's prepared gathers
//! land the bytes the synchronous gather lands, begin once the prefill
//! joined its last Engram layer, and join each layer before its verify
//! consumer, in captured and replayed verify graphs and in eager pieces.
//! Steps outside the admitted range keep the synchronous gather, and a
//! failed verify join settles the gathers before the step is refused.
use super::*;
use crate::fake_host_tests::ar_engram::order;
use crate::fake_host_tests::prefill_engram::create_backend_with_options;
use model_deepseek41_contract::execution_options::{
    DSPARK_GRAPH_HANDLES_PER_RANK, DSPARK_MIXED_PREPARED_ENGRAM,
};
use runtime_bundle_sdk::prefill::{Boundary, encode};

/// The request of the verify item in slot `slot`.
fn verifier(slot: u64) -> u64 {
    100 + slot
}

/// A mixed step as the prefill API negotiates it: request `request`
/// prefills the first `rows` rows of its prompt in slot `slot` beside
/// verify items of `(slot, committed tokens, rows)`.
fn mixed(request: u64, slot: u64, rows: u64, items: &[(u64, u64, u32)]) -> Vec<u8> {
    let pages = |slot: u64, end: u64| (0..end.div_ceil(64)).map(|page| slot * 8 + page).collect();
    let mut requests = vec![BatchItemV1::new(
        request,
        false,
        rows as u32,
        true,
        (0..rows)
            .map(|position| ((position * 7 + request) % 1000) as u32)
            .collect(),
    )];
    let mut appends = vec![BatchAppendV1::new(0, rows, rows, pages(slot, rows), None)];
    let mut slots = vec![slot];
    for &(slot, start, rows) in items {
        let end = start + u64::from(rows);
        requests.push(BatchItemV1::verify(verifier(slot), rows, true, vec![7]));
        appends.push(BatchAppendV1::new(
            start,
            u64::from(rows),
            end,
            pages(slot, end),
            None,
        ));
        slots.push(slot);
    }
    let original = BatchRequestV1::new(BatchWireVersionV1::V3, requests, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap();
    encode(
        &original,
        &[Boundary {
            request_id: request,
            prompt_tokens: rows,
            chunk_end: rows,
            chunk_tokens: 8192,
        }],
    )
    .unwrap()
}

/// What one schedule left: its host and each mixed step's journal.
struct Run {
    host: Rc<Host>,
    steps: Vec<Vec<String>>,
}

/// An upper-skip instance with `options` serving `verify_rows` verify rows
/// on `backend`: three-token prompts in slots `0..widths.len()`, then a
/// mixed step per entry of `prefills`, each a fresh prompt of that many
/// rows in the next slot beside every item verifying its width, all
/// accepted.
fn schedule(
    verify_rows: u32,
    widths: &[u32],
    prefills: &[u64],
    options: &[(&str, serde_json::Value)],
    backend: impl FnOnce(Rc<Host>) -> MappedHost,
) -> Run {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    host.speculation.set(Some((5, verify_rows)));
    let mut instance = create_backend_with_options(&host, true, backend(Rc::clone(&host)), options)
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let mut ends = Vec::new();
    for slot in 0..widths.len() as u64 {
        execute(
            &mut instance,
            &host,
            &crate::fake_host_tests::prompt(verifier(slot), slot, 0, 3, true),
        );
        instance.commit().unwrap();
        ends.push(3);
    }
    let mut steps = Vec::new();
    for (index, &rows) in prefills.iter().enumerate() {
        let items: Vec<_> = widths
            .iter()
            .zip(&ends)
            .enumerate()
            .map(|(slot, (&width, &end))| (slot as u64, end, width))
            .collect();
        let request = 7 + index as u64;
        let begin = host.journal.borrow().len();
        let mut sink = Sink::default();
        instance
            .execute_prefill(
                &mixed(request, (widths.len() + index) as u64, rows, &items),
                &mut sink,
            )
            .unwrap_or_else(|e| panic!("mixed step: {e:?}: {:?}", host.logs.borrow()));
        steps.push(host.journal.borrow()[begin..].to_vec());
        assert_eq!(sink.draft_tokens().len(), widths.len());
        let kept: Vec<_> = items
            .iter()
            .map(|&(slot, _, width)| kept(verifier(slot), width))
            .collect();
        instance.commit_accepted(&kept).unwrap();
        for (end, &width) in ends.iter_mut().zip(widths) {
            *end += u64::from(width);
        }
        instance.retire_request(request).unwrap();
    }
    for slot in 0..widths.len() as u64 {
        instance.retire_request(verifier(slot)).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    Run { host, steps }
}

/// Each mapped-row upload of `journal`, as its gather note and upload line
/// (destination and byte digest), in order.
fn row_uploads(journal: &[String]) -> Vec<(String, String)> {
    let mut uploads: Vec<_> = journal
        .windows(2)
        .filter(|pair| pair[0].starts_with("mapped gather "))
        .map(|pair| (pair[0].clone(), pair[1].clone()))
        .collect();
    uploads.sort();
    uploads
}

/// The launches `run`'s route records count for `entry`.
fn launches(run: &Run, entry: &str) -> u64 {
    run.host
        .logs
        .borrow()
        .iter()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|value| value["schema"] == "chopinfer.route.v1")
        .filter_map(|value| value["entries"][entry]["launches"].as_u64())
        .sum()
}

/// `kinds` from the first enqueue, as runs of one kind and their lengths.
fn groups<'a>(kinds: &[&'a str]) -> Vec<(&'a str, usize)> {
    let first = kinds.iter().position(|&kind| kind == "ids").unwrap();
    kinds[first..].iter().fold(Vec::new(), |mut runs, &kind| {
        match runs.last_mut() {
            Some((last, count)) if *last == kind => *count += 1,
            _ => runs.push((kind, 1)),
        }
        runs
    })
}

/// The 32- and 64-row verify phases of C8 and C16 (four- and six-row
/// items), each beside two 512-row prefills, so the second step reuses the
/// cycle's gathers and replays the verify graphs the first captured; and
/// the same steps with eager DSpark pieces.
#[test]
fn mixed_verify_rows_land_the_synchronous_bytes_after_the_last_prefill_join() {
    let c16 = [vec![6; 10], vec![4]].concat();
    for (verify_rows, widths) in [(32, vec![4; 8]), (64, c16)] {
        let rows: u64 = widths.iter().map(|&w| u64::from(w)).sum();
        for eager in [false, true] {
            let mut options = Vec::new();
            if eager {
                options.push((DSPARK_GRAPH_HANDLES_PER_RANK, 0.into()));
            }
            let run = |options: &[(&str, serde_json::Value)]| {
                bundle_runtime::routes::with_logging(|| {
                    schedule(verify_rows, &widths, &[512, 512], options, |host| {
                        MappedHost::with_row_ids(host)
                    })
                })
            };
            let off = run(&options);
            options.push((DSPARK_MIXED_PREPARED_ENGRAM, true.into()));
            let on = run(&options);
            // Each step's verify rows take one route per Engram layer: the
            // prepared gather with the candidate, the synchronous one per
            // rank and plane without it.
            let layers = u64::from(FLASH.engram_layers());
            assert_eq!(
                launches(&off, "engram/sync/mixed_target"),
                2 * 2 * 4 * layers
            );
            assert_eq!(launches(&off, "engram/prepared/dspark_mixed_target"), 0);
            assert_eq!(launches(&on, "engram/sync/mixed_target"), 0);
            assert_eq!(
                launches(&on, "engram/prepared/dspark_mixed_target"),
                2 * layers
            );
            for (off, on) in off.steps.iter().zip(&on.steps) {
                assert_eq!(
                    row_uploads(off),
                    row_uploads(on),
                    "{rows} rows, eager {eager}"
                );
                let off = order(off);
                assert!(off.contains(&"sync"), "{off:?}");
                assert_eq!(off.iter().filter(|&&kind| kind == "ids").count(), 2);
                let kinds = order(on);
                assert!(!kinds.contains(&"sync"), "{kinds:?}");
                // The prefill's two layers, every rank's joins of both, the
                // verify layers, then the verify pieces between each rank's
                // join of each layer, and all four gathers complete.
                let expected: &[(&str, usize)] = if eager {
                    &[
                        ("ids", 2),
                        ("wait", 8),
                        ("ids", 2),
                        ("wait", 8),
                        ("complete", 4),
                    ]
                } else {
                    &[
                        ("ids", 2),
                        ("wait", 8),
                        ("ids", 2),
                        ("graph", 0),
                        ("wait", 4),
                        ("graph", 0),
                        ("wait", 4),
                        ("graph", 0),
                        ("complete", 4),
                    ]
                };
                let found = groups(&kinds);
                assert_eq!(found.len(), expected.len(), "{found:?}");
                for (&(kind, count), &(want, wanted)) in found.iter().zip(expected) {
                    assert_eq!(kind, want, "{found:?}");
                    if wanted > 0 {
                        assert_eq!(count, wanted, "{found:?}");
                    }
                }
            }
            for step in &on.steps {
                // Nothing launches between the prefill's last join and the
                // verify enqueue, and every verify join follows a launch of
                // its step's earlier pieces.
                let waits: Vec<_> = (0..step.len())
                    .filter(|&at| step[at].starts_with("async mapped wait "))
                    .collect();
                let enqueues: Vec<_> = (0..step.len())
                    .filter(|&at| step[at].starts_with("async mapped ids "))
                    .collect();
                assert_eq!((waits.len(), enqueues.len()), (16, 4));
                assert!(
                    step[waits[7]..enqueues[2]]
                        .iter()
                        .all(|line| !line.starts_with("launch ") && !line.starts_with("replay "))
                );
                for range in [enqueues[3]..waits[8], waits[11]..waits[12]] {
                    assert!(
                        step[range]
                            .iter()
                            .any(|line| line.starts_with("launch ") || line.starts_with("replay ")),
                        "a verify join without an earlier piece"
                    );
                }
            }
            let traced = |run: &Run| -> Vec<u64> {
                run.host
                    .logs
                    .borrow()
                    .iter()
                    .filter(|line| line.contains("chopinfer.deepseek41.ar-engram.v1"))
                    .map(|line| {
                        serde_json::from_str::<serde_json::Value>(line).unwrap()["rows"]
                            .as_u64()
                            .unwrap()
                    })
                    .collect()
            };
            assert!(traced(&off).is_empty());
            assert_eq!(traced(&on), [rows, rows]);
        }
    }
}

/// A 511-row prefill, a 66-row verify phase (above the graphed 64) and a
/// prefill without upper-skip DMA keep the synchronous gather.
#[test]
fn mixed_steps_outside_the_admitted_range_keep_the_synchronous_gather() {
    let on = [(DSPARK_MIXED_PREPARED_ENGRAM, true.into())];
    for (verify_rows, widths, prefill, dma) in [
        (32, vec![4; 8], 511, true),
        (192, vec![6; 11], 512, true),
        (32, vec![4; 8], 512, false),
    ] {
        let run = || {
            schedule(
                verify_rows,
                &widths,
                &[prefill],
                &on,
                MappedHost::with_row_ids,
            )
        };
        let run = crate::instance::execute::skip::with_dma(dma, run);
        let kinds = order(&run.steps[0]);
        assert!(kinds.contains(&"sync"), "{prefill} {widths:?} {kinds:?}");
        assert_eq!(
            kinds.iter().filter(|&&kind| kind == "ids").count(),
            2,
            "{prefill} {widths:?} {kinds:?}"
        );
    }
}

/// A verify join that fails after its prefill began the gathers refuses
/// the step; the cleanup makes every remaining join, completes all four
/// gathers and leaves no handle behind.
#[test]
fn a_failed_verify_join_settles_the_mixed_gathers() {
    let options = [(DSPARK_MIXED_PREPARED_ENGRAM, true.into())];
    // Eight three-token prompts on `backend`, and the waits they made.
    let prompted = |backend: fn(Rc<Host>) -> MappedHost| {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        host.speculation.set(Some((5, 32)));
        let mut instance =
            create_backend_with_options(&host, true, backend(Rc::clone(&host)), &options).unwrap();
        for slot in 0..8 {
            execute(
                &mut instance,
                &host,
                &crate::fake_host_tests::prompt(verifier(slot), slot, 0, 3, true),
            );
            instance.commit().unwrap();
        }
        let waits = order(&host.journal.borrow())
            .iter()
            .filter(|&&kind| kind == "wait")
            .count();
        (host, instance, waits)
    };
    let (_, _, waits) = prompted(MappedHost::with_row_ids);
    assert_eq!(waits, PROMPT_WAITS);
    // Then the prefill's eight joins and two of the first verify layer's.
    let (host, mut instance, _) =
        prompted(|host| MappedHost::with_row_ids(host).failing("wait", PROMPT_WAITS + 10));
    let items: Vec<_> = (0..8).map(|slot| (slot, 3, 4)).collect();
    let begin = host.journal.borrow().len();
    let mut sink = Sink::default();
    assert!(
        instance
            .execute_prefill(&mixed(7, 8, 512, &items), &mut sink)
            .is_err()
    );
    {
        let journal = host.journal.borrow();
        let step = &journal[begin..];
        let failure = step
            .iter()
            .position(|line| line == "async mapped failure wait")
            .unwrap();
        assert_eq!(
            order(&step[..failure])
                .iter()
                .filter(|&&kind| kind == "ids")
                .count(),
            4
        );
        let after = order(&step[failure..]);
        assert_eq!(
            after.iter().filter(|&&kind| kind == "wait").count(),
            6,
            "{after:?}"
        );
        assert_eq!(
            after.iter().filter(|&&kind| kind == "complete").count(),
            4,
            "{after:?}"
        );
    }
    let _ = instance.cancel();
    match instance.destroy() {
        Ok(()) => assert_eq!(host.acquired.get(), host.released.get()),
        Err(_) => assert!(instance.is_poisoned()),
    }
}

/// The prepared-gather joins of eight three-token prompts: two layers on
/// four ranks each.
const PROMPT_WAITS: usize = 64;
