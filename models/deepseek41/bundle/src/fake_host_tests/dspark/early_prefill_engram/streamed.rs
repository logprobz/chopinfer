//! Streamed lowering beside the DSpark early steps: a prefill no kept
//! plan or template serves lowers one layer at a time while it issues,
//! beside the early Engram begin and plan and the mixed steps' prepared
//! verify-row gathers, with the upper-skip DMA and without, and keeps its
//! plan once its launch completed. No plan is lowered whole before its
//! launch but where the verify mode lowers a reused one again. A mixed step
//! that fails keeps nothing.
use super::*;
use crate::instance::execute::{plan_templates, plan_timing, skip};

/// `schedule` with `options`, the upper-skip DMA as `dma`: the run, the
/// plans lowered while they issued and those lowered before their launch.
fn counted(
    schedule: &Schedule<'_>,
    options: &[(&str, serde_json::Value)],
    dma: bool,
) -> (Run, u64, u64) {
    let before = (skip::streamed_launches(), skip::lowerings());
    let run = skip::with_dma(dma, || schedule.run(options));
    (
        run,
        skip::streamed_launches() - before.0,
        skip::lowerings() - before.1,
    )
}

/// `schedule` with each of `options` and the DMA on and off streams
/// `launches` plans, each kept once its launch completed, and lowers none
/// whole before its launch outside the verify mode.
#[track_caller]
fn assert_streamed(schedule: &Schedule<'_>, options: &[(String, Options)], launches: u64) {
    for (name, options) in options {
        let verify = options.iter().any(|(option, value)| {
            *option == PREFILL_PLAN_CACHE_VERIFY && value.as_bool() == Some(true)
        });
        for dma in [true, false] {
            let context = format!("{name}, DMA {dma}");
            let (run, launched, lowered) = counted(schedule, options, dma);
            let keeps = run
                .logs
                .iter()
                .filter(|line| {
                    line.contains("prefill plan templates: kept")
                        || line.contains("prefill plan cache: kept")
                })
                .count() as u64;
            assert_eq!((launched, keeps), (launches, launches), "{context}");
            if !verify {
                assert_eq!(lowered, 0, "{context}");
            }
        }
    }
}

/// Instance options.
type Options = Vec<(&'static str, serde_json::Value)>;

/// The default options, named.
fn default() -> Vec<(String, Options)> {
    vec![("default".into(), vec![])]
}

/// The early Engram begin's schedule (template reuse and misses, 511 to
/// 8000 rows, two prefill items, a prefill step without verify rows, a
/// cancel and its retry): with every reused plan lowered again when
/// `verify`, else by default and eager (no graph handles).
fn assert_early_begin_schedule(verify: bool) {
    use Then::*;
    let steps = [
        step(&[prefill(7, 8, 0, 512)], true, Commit),
        step(&[prefill(8, 9, 0, 512)], true, CancelRetry),
        step(&[prefill(9, 10, 0, 511)], true, Commit),
        step(&[prefill(10, 11, 0, 2048)], true, Commit),
        step(&[prefill(11, 12, 0, 4097)], true, Commit),
        step(&[prefill(12, 13, 0, 8000)], true, Commit),
        step(
            &[prefill(13, 14, 0, 300), prefill(14, 15, 0, 700)],
            true,
            Commit,
        ),
        step(&[prefill(15, 8, 0, 512)], false, Commit),
    ];
    let schedule = Schedule {
        verify_rows: 32,
        widths: &[4; 8],
        restored: &[],
        steps: &steps,
    };
    let options = if verify {
        verified()
    } else {
        let eager = (
            "eager".into(),
            vec![(DSPARK_GRAPH_HANDLES_PER_RANK, 0.into())],
        );
        [default(), vec![eager]].concat()
    };
    // The first prompt and the six prefill structures.
    assert_streamed(&schedule, &options, 7);
}

/// The early begin's schedule by default, and eager.
#[test]
fn the_early_begin_schedule_streams_each_new_prefill_structure() {
    assert_early_begin_schedule(false);
}

/// The early begin's schedule with every reused plan lowered again.
#[test]
fn the_verified_early_begin_schedule_streams_each_new_prefill_structure() {
    assert_early_begin_schedule(true);
}

/// Every reused plan lowered again.
fn verified() -> Vec<(String, Options)> {
    vec![(
        "verified".into(),
        vec![(PREFILL_PLAN_CACHE_VERIFY, true.into())],
    )]
}

/// The live shape: 1536-row prefills after a 93,000-token context beside
/// eight verify items, a template reused after another 128 tokens and
/// after a cancel, and a 1537-row miss, with every reused plan lowered
/// again.
#[test]
fn long_context_misses_are_streamed() {
    use Then::*;
    let long = [
        step(&[prefill(20, 8, 93_000, 1536)], true, Commit),
        step(&[prefill(21, 9, 93_128, 1536)], true, CancelRetry),
        step(&[prefill(22, 10, 93_000, 1537)], true, Commit),
    ];
    let schedule = Schedule {
        verify_rows: 32,
        widths: &[4; 8],
        restored: &[(20, 8, 93_000), (21, 9, 93_128), (22, 10, 93_000)],
        steps: &long,
    };
    // The first prompt and the two prefill structures.
    assert_streamed(&schedule, &verified(), 3);
}

/// A 66-row eager verify phase beside a 512-row template and its reuse,
/// with every reused plan lowered again.
#[test]
fn eager_verify_misses_are_streamed() {
    use Then::*;
    let wide = [
        step(&[prefill(7, 11, 0, 512)], true, Commit),
        step(&[prefill(8, 12, 0, 512)], true, Commit),
    ];
    let schedule = Schedule {
        verify_rows: 192,
        widths: &[6; 11],
        restored: &[],
        steps: &wide,
    };
    // The first prompt and the prefill structure.
    assert_streamed(&schedule, &verified(), 2);
}

/// With room for two templates, prefills of 512, 600, 700 and 512 rows
/// drop the oldest template at each of the last two. A streamed miss keeps
/// its template once its launch completed, so the template it drops is
/// freed after the launch.
#[test]
fn a_streamed_miss_frees_the_template_it_drops_after_its_launch() {
    use Then::*;
    let steps = [
        step(&[prefill(7, 8, 0, 512)], true, Commit),
        step(&[prefill(8, 9, 0, 600)], true, Commit),
        step(&[prefill(9, 10, 0, 700)], true, Commit),
        step(&[prefill(10, 11, 0, 512)], true, Commit),
    ];
    let schedule = Schedule {
        verify_rows: 32,
        widths: &[4; 8],
        restored: &[],
        steps: &steps,
    };
    plan_templates::with_limit(2, || {
        // The first prompt and the four prefills: the last 512 rows miss,
        // as the 700 rows dropped their template.
        assert_streamed(&schedule, &default(), 5);
        let (host, mut instance, _) = prompted(&[], MappedHost::with_row_ids);
        let mut events = Vec::new();
        for (index, rows) in [512, 600, 700].into_iter().enumerate() {
            let request = 7 + index as u64;
            let committed = 3 + 4 * index as u64;
            let verifies: Vec<_> = (0..8).map(|slot| (slot, committed, 4)).collect();
            let step = batch(&[prefill(request, 8, 0, rows)], &verifies);
            let (_, seen) =
                crate::instance::execute::observe_events(|| run(&mut instance, &host, &step));
            events = seen;
            let kept: Vec<_> = (0..8).map(|slot| kept(verifier(slot), 4)).collect();
            instance.commit_accepted(&kept).unwrap();
            instance.retire_request(request).unwrap();
        }
        // The 700-row miss dropped the 512-row template.
        assert_eq!(plan_templates::counts(instance.handle()).evicted, 1);
        assert_eq!(events, ["prefill launch", "prefill launched", "plan freed"]);
        assert_eq!(plan_templates::retired(instance.handle()), 0);
        let _ = instance.destroy();
    });
}

/// Where a mixed step fails (`failed_mixed`).
#[derive(Clone, Debug)]
enum Fault {
    /// The step's `n`th launch, counting from zero.
    Launch(u32),
    /// The `nth` eager segment launching the symbol refused (`with_refused`).
    Refused(String, u32),
    /// An injected failure of the streamed launch (`with_streamed_failure`).
    Streamed(skip::StreamedFailure),
}

/// What a failed mixed step left: its journal, its status, whether its
/// collectives are poisoned, and the templates it kept.
type Failed = (Vec<String>, String, bool, u64);

/// `prompted`'s mixed step, failing at `fault`. A poisoned instance
/// refuses its next execute.
fn failed_mixed(fault: &Fault) -> Failed {
    let (host, mut instance, mixed) = prompted(&[], MappedHost::with_row_ids);
    let kept = plan_templates::counts(instance.handle()).kept;
    let begin = host.journal.borrow().len();
    let mut execute = || instance.execute_prefill(&mixed, &mut Sink::default());
    let status = match fault {
        Fault::Launch(n) => {
            host.fail_launch.set(Some(*n));
            let status = execute();
            host.fail_launch.set(None);
            status
        }
        Fault::Refused(symbol, nth) => crate::instance::with_refused(symbol, *nth, execute),
        Fault::Streamed(failure) => skip::with_streamed_failure(*failure, execute),
    };
    let journal = host.journal.borrow()[begin..].to_vec();
    let poisoned = plan_timing::poisoned(instance.handle());
    let kept = plan_templates::counts(instance.handle()).kept - kept;
    if poisoned {
        assert_eq!(
            instance.execute_prefill(&mixed, &mut Sink::default()),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidState)),
            "{fault:?}: a poisoned instance refuses its next execute"
        );
    }
    let _ = instance.cancel();
    let _ = instance.destroy();
    assert!(
        host.races.borrow().is_empty(),
        "{fault:?}: {:#?}",
        host.races.borrow()
    );
    (journal, format!("{status:?}"), poisoned, kept)
}

/// The mixed step's 512-row prefill: rank 0's body launches as (symbol,
/// stage).
fn prefill_body() -> Vec<(String, crate::program::Stage)> {
    let (_host, mut instance, _) = prompted(&[], MappedHost::with_row_ids);
    let (request, _) = original(&[prefill(7, 8, 0, 512)], &[]);
    let planned = plan_timing::plan(instance.handle(), &request).unwrap_or_else(|e| panic!("{e}"));
    let _ = instance.destroy();
    planned.symbols.0
}

/// A mixed C8 step beside a 512-row prefill that fails while it streams,
/// with the upper-skip DMA and without, keeps nothing. Launch failures at
/// the prefill's last launch before its first gather join, at its first
/// after it and at its first after the second layer's joins poison, read
/// from the collectives, as does a refusal of its first Engram consumer's
/// segment on rank 0, after that join; a refusal of the step's first
/// segment of the prefill's first symbol, which an earlier phase also
/// launches, refuses the step. Injected failures of the streamed launch: a
/// lowering error at its first group refuses unpoisoned, and one at a later
/// group, failed gathers, a launch reading pending ids, a panic and a head
/// audit error poison.
#[test]
fn a_mixed_step_that_fails_while_it_streams_keeps_nothing() {
    use crate::program::Stage;
    use skip::StreamedFailure::{Alias, Gathers, Group, Head, Panic};
    let body = prefill_body();
    let first = body[0].0.clone();
    let layer = FLASH.shape.engram.layers[0];
    let at = body
        .iter()
        .position(|(_, stage)| *stage == Stage::EngramRows(layer))
        .expect("the first Engram consumer");
    let consumer = body[at].0.clone();
    // Each earlier occurrence is a segment on every rank.
    let earlier = body[..at].iter().filter(|(s, _)| *s == consumer).count() as u32;
    for dma in [true, false] {
        let context = format!("DMA {dma}");
        let reference = skip::with_dma(dma, || {
            let (host, mut instance, mixed) = prompted(&[], MappedHost::with_row_ids);
            let begin = host.journal.borrow().len();
            run(&mut instance, &host, &mixed);
            let journal = host.journal.borrow()[begin..].to_vec();
            let _ = instance.cancel();
            let _ = instance.destroy();
            journal
        });
        let launches_before = |at: usize| {
            reference[..at]
                .iter()
                .filter(|line| line.starts_with("launch "))
                .count() as u32
        };
        // Four ranks join each of the two layers.
        let waits: Vec<_> = (0..reference.len())
            .filter(|&at| reference[at].starts_with("async mapped wait "))
            .collect();
        assert!(waits.len() >= 8, "{context}: {waits:?}");
        let faults = [
            Fault::Launch(launches_before(waits[0]) - 1),
            Fault::Launch(launches_before(waits[0])),
            Fault::Launch(launches_before(waits[4])),
            Fault::Refused(first.clone(), 0),
            Fault::Refused(consumer.clone(), 4 * earlier),
        ];
        for fault in &faults {
            let what = format!("{context}, {fault:?}");
            let (journal, status, poisoned, kept) = skip::with_dma(dma, || failed_mixed(fault));
            assert!(status.starts_with("Err"), "{what}: {status}");
            // A failed launch poisons; the consumer's refusal follows the
            // prefill's first join.
            let joined = journal.iter().any(|l| l.starts_with("async mapped wait "));
            match fault {
                Fault::Launch(_) => assert!(poisoned, "{what}: poisoned"),
                Fault::Refused(_, 0) => {}
                _ => assert!(joined && poisoned, "{what}: poisoned inside the prefill"),
            }
            assert_eq!(kept, 0, "{what}: kept");
        }
        for failure in [Group(0), Group(10), Gathers(10), Alias(10), Panic(10), Head] {
            let what = format!("{context}, {failure:?}");
            let (_, status, poisoned, kept) =
                skip::with_dma(dma, || failed_mixed(&Fault::Streamed(failure)));
            assert!(status.starts_with("Err"), "{what}: {status}");
            assert_eq!((poisoned, kept), (failure != Group(0), 0), "{what}");
        }
    }
}

/// With diagnostic step logs, each step-shape record says whether its
/// step's prefill was lowered while it issued (`streamed_lowering`): true
/// for a miss (the first prompt's, of the short-prompt cache, and each new
/// prefill structure's) and false for a kept plan's or template's reuse.
/// The record's template counts show the reuse is a hit.
#[test]
fn step_shape_records_say_whether_the_prefill_was_streamed() {
    use Then::*;
    use model_deepseek41_contract::execution_options::DIAGNOSTIC_STEP_LOGS;
    let steps = [
        step(&[prefill(7, 8, 0, 512)], true, Commit),
        step(&[prefill(8, 9, 0, 512)], true, Commit),
        step(&[prefill(9, 10, 0, 600)], true, Commit),
    ];
    let schedule = Schedule {
        verify_rows: 32,
        widths: &[4; 8],
        restored: &[],
        steps: &steps,
    };
    let run = schedule.run(&[(DIAGNOSTIC_STEP_LOGS, true.into())]);
    let field = |marker: &str| -> Vec<&str> {
        run.logs
            .iter()
            .filter_map(|line| line.find(marker).map(|at| &line[at + marker.len()..]))
            .collect()
    };
    let records: Vec<bool> = field("\"streamed_lowering\":")
        .iter()
        .map(|rest| rest.starts_with("true"))
        .collect();
    // Eight prompts, the first a miss, then a 512-row miss, its template's
    // reuse by another request and a 600-row miss.
    let expected = [vec![true], vec![false; 7], vec![true, false, true]].concat();
    assert_eq!(records, expected);
    let hits: Vec<u64> = field("\"prefill_templates\":")
        .iter()
        .map(|rest| {
            let templates: serde_json::Value =
                serde_json::from_str(&rest[..=rest.find('}').unwrap()]).unwrap();
            templates["hits"].as_u64().unwrap()
        })
        .collect();
    assert_eq!(hits[8..], [0, 1, 1], "the reuse is a template hit");
}
