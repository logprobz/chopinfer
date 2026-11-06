//! The `prefill_streamed_lowering` candidate beside the DSpark candidates:
//! a prefill no kept plan or template serves lowers one layer at a time
//! while it issues. Each step's journal (every launch with its arguments,
//! every upload with its bytes, the prepared gathers' begins and joins),
//! every log line and every result is the batch lowering's, with and
//! without each of `dspark_early_prefill_engram`, `dspark_early_prefill_plan`
//! and `dspark_mixed_prepared_engram`, with the upper-skip DMA and without.
//! Each combination of the three, pairs included, runs. A mixed step
//! fails streamed where it fails batch, and keeps nothing.
//! Only the line that keeps a streamed plan moves: it follows the launch's
//! own lines, as the plan is kept once its launch completed. The streamed
//! plans are the ones the batch run lowered before their launch, but for
//! the verify mode's, and in these schedules each is one the batch run
//! kept as a plan or template.
use super::*;
use crate::instance::execute::{plan_templates, plan_timing, skip};
use model_deepseek41_contract::execution_options::PREFILL_STREAMED_LOWERING;

/// Each step of `on` issues `off`'s journal and results exactly, and every
/// log line matches; the lines that keep a plan or template are compared
/// apart, in order. Returns how many there are.
#[track_caller]
fn assert_same_steps(off: &Run, on: &Run, context: &str) -> usize {
    assert_eq!(off.steps.len(), on.steps.len(), "{context}");
    for (index, (a, b)) in off.steps.iter().zip(&on.steps).enumerate() {
        let context = format!("{context}, step {index}");
        assert_same_lines(&a.journal, &b.journal, &context);
        assert_eq!(a.drafts, b.drafts, "{context}");
        assert_eq!(a.views, b.views, "{context}");
    }
    let keeps = |run: &Run| -> (Vec<String>, Vec<String>) {
        run.logs.iter().cloned().partition(|line| {
            line.contains("prefill plan templates: kept")
                || line.contains("prefill plan cache: kept")
        })
    };
    let (keeps_off, logs_off) = keeps(off);
    let (keeps_on, logs_on) = keeps(on);
    assert_same_lines(&keeps_off, &keeps_on, &format!("{context}: keeps"));
    assert_same_lines(&logs_off, &logs_on, &format!("{context}: logs"));
    keeps_on.len()
}

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

/// `schedule` with each of `options` and the DMA on and off issues the
/// same steps streamed as batch, and streams each plan the batch run
/// lowered but for the verify mode's: `launches`, each one the batch run
/// kept.
#[track_caller]
fn assert_streamed(schedule: &Schedule<'_>, options: &[(String, Options)], launches: u64) {
    let streamed = [(PREFILL_STREAMED_LOWERING, true.into())];
    for (name, options) in options {
        for dma in [true, false] {
            let context = format!("{name}, DMA {dma}");
            let (off, none, batch) = counted(schedule, options, dma);
            let (on, launched, lowered) = counted(schedule, &with(options, &streamed), dma);
            assert_eq!(none, 0, "{context}");
            assert_eq!(launched + lowered, batch, "{context}");
            let keeps = assert_same_steps(&off, &on, &context);
            assert_eq!((launched, keeps as u64), (launches, launches), "{context}");
        }
    }
}

/// Instance options.
type Options = Vec<(&'static str, serde_json::Value)>;

/// Every combination of the three DSpark candidates, from none to all,
/// named by the candidates it turns on, those with the early plan when
/// `plan`.
fn candidates(plan: bool) -> Vec<(String, Options)> {
    let named = [
        (DSPARK_EARLY_PREFILL_ENGRAM, "early begin"),
        (DSPARK_EARLY_PREFILL_PLAN, "early plan"),
        (DSPARK_MIXED_PREPARED_ENGRAM, "mixed"),
    ];
    (0..8)
        .filter(|mask| (mask & 2 != 0) == plan)
        .map(|mask| {
            let on: Vec<_> = (0..3).filter(|bit| mask & (1 << bit) != 0).collect();
            let name = on.iter().map(|&bit| named[bit].1).collect::<Vec<_>>();
            let options = on.iter().map(|&bit| (named[bit].0, true.into())).collect();
            (
                if name.is_empty() {
                    "default".into()
                } else {
                    name.join(", ")
                },
                options,
            )
        })
        .collect()
}

/// Every combination of the three DSpark candidates.
fn every_candidate() -> Vec<(String, Options)> {
    [candidates(false), candidates(true)].concat()
}

/// The early Engram begin's schedule (template reuse and misses, 511 to
/// 8000 rows, two prefill items, a prefill step without verify rows, a
/// cancel and its retry), with the candidate combinations `plan` selects
/// (`candidates`), and all three eager (no graph handles) or with every
/// reused plan lowered again.
fn assert_early_begin_schedule(plan: bool) {
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
    let mut options = candidates(plan);
    let all = options.last().unwrap().1.clone();
    options.push(if plan {
        (
            "all three, verified".into(),
            with(&all, &[(PREFILL_PLAN_CACHE_VERIFY, true.into())]),
        )
    } else {
        (
            "early begin, mixed, eager".into(),
            with(&all, &[(DSPARK_GRAPH_HANDLES_PER_RANK, 0.into())]),
        )
    });
    // The first prompt and the six prefill structures.
    assert_streamed(&schedule, &options, 7);
}

/// The early begin's schedule with the combinations without the early
/// plan, and two of them eager.
#[test]
fn streamed_plans_issue_the_batch_journal_beside_the_dspark_candidates() {
    assert_early_begin_schedule(false);
}

/// The early begin's schedule with the combinations with the early plan,
/// and all three verified.
#[test]
fn streamed_plans_issue_the_batch_journal_beside_the_early_dspark_plan() {
    assert_early_begin_schedule(true);
}

/// Every candidate combination with every reused plan lowered again.
fn verified() -> Vec<(String, Options)> {
    let verify = (PREFILL_PLAN_CACHE_VERIFY, true.into());
    every_candidate()
        .into_iter()
        .map(|(name, options)| (name, with(&options, std::slice::from_ref(&verify))))
        .collect()
}

/// The live shape: 1536-row prefills after a 93,000-token context beside
/// eight verify items, a template reused after another 128 tokens and
/// after a cancel, and a 1537-row miss, with every candidate combination
/// and every reused plan lowered again.
#[test]
fn long_context_streamed_plans_issue_the_batch_journal() {
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
/// with every candidate combination and every reused plan lowered again.
#[test]
fn eager_verify_streamed_plans_issue_the_batch_journal() {
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
/// drop the oldest template at each of the last two, streamed as batch.
/// A streamed miss keeps its template once its launch completed, so the
/// template it drops is freed after the launch with or without
/// `dspark_early_prefill_plan`, where the batch lowering without it frees
/// it before.
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
    let plan = (DSPARK_EARLY_PREFILL_PLAN, true.into());
    let streamed = (PREFILL_STREAMED_LOWERING, true.into());
    plan_templates::with_limit(2, || {
        let options = [
            ("default".into(), vec![]),
            ("early plan".into(), vec![plan.clone()]),
        ];
        // The first prompt and the four prefills: the last 512 rows miss,
        // as the 700 rows dropped their template.
        assert_streamed(&schedule, &options, 5);
        let launch = ["prefill launch", "prefill launched"];
        for (options, frees_first) in [
            (vec![], true),
            (vec![plan.clone()], false),
            (vec![streamed.clone()], false),
            (vec![plan.clone(), streamed.clone()], false),
        ] {
            let context = format!("{options:?}");
            let (host, mut instance, _) = prompted(&options, MappedHost::with_row_ids);
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
            assert_eq!(
                plan_templates::counts(instance.handle()).evicted,
                1,
                "{context}"
            );
            let expected = if frees_first {
                [&["plan freed"][..], &launch].concat()
            } else {
                [&launch[..], &["plan freed"]].concat()
            };
            assert_eq!(events, expected, "{context}");
            assert_eq!(plan_templates::retired(instance.handle()), 0, "{context}");
            let _ = instance.destroy();
        }
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

/// `prompted`'s mixed step with `options`, failing at `fault`. A poisoned
/// instance refuses its next execute.
fn failed_mixed(options: &[(&str, serde_json::Value)], fault: &Fault) -> Failed {
    let (host, mut instance, mixed) = prompted(options, MappedHost::with_row_ids);
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

/// A mixed C8 step beside a 512-row prefill fails streamed where it fails
/// batch, with the candidate combinations `plan` selects (`candidates`),
/// with the upper-skip DMA and without. Launch failures at the prefill's
/// last launch before its first gather join, at its first after it and at
/// its first after the second layer's joins, a refusal of its first Engram
/// consumer's segment on rank 0, after that join, and a refusal of the
/// step's first segment of the prefill's first symbol, which an earlier
/// phase also launches, give the same journal, status and poisoning, read
/// from the collectives; all but the last poison. The batch prefill kept
/// its template while it planned; the failed streamed miss keeps none. Injected failures of the streamed launch keep nothing
/// either: a lowering error at its first group refuses unpoisoned, and one
/// at a later group, failed gathers, a launch reading pending ids, a panic
/// and a head audit error poison.
fn assert_mixed_failures(plan: bool) {
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
    let streamed = [(PREFILL_STREAMED_LOWERING, true.into())];
    for (name, options) in candidates(plan) {
        for dma in [true, false] {
            let context = format!("{name}, DMA {dma}");
            let reference = skip::with_dma(dma, || {
                let (host, mut instance, mixed) = prompted(&options, MappedHost::with_row_ids);
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
                let (batch, on) = skip::with_dma(dma, || {
                    (
                        failed_mixed(&options, fault),
                        failed_mixed(&with(&options, &streamed), fault),
                    )
                });
                assert!(batch.1.starts_with("Err"), "{what}: {}", batch.1);
                assert_same_lines(&batch.0, &on.0, &what);
                assert_eq!((&batch.1, batch.2), (&on.1, on.2), "{what}");
                // A failed launch poisons; the consumer's refusal follows the
                // prefill's first join.
                let joined = batch.0.iter().any(|l| l.starts_with("async mapped wait "));
                match fault {
                    Fault::Launch(_) => assert!(batch.2, "{what}: poisoned"),
                    Fault::Refused(_, 0) => {}
                    _ => assert!(joined && batch.2, "{what}: poisoned inside the prefill"),
                }
                assert_eq!((batch.3, on.3), (1, 0), "{what}: kept");
            }
            for failure in [Group(0), Group(10), Gathers(10), Alias(10), Panic(10), Head] {
                let what = format!("{context}, {failure:?}");
                let (_, status, poisoned, kept) = skip::with_dma(dma, || {
                    failed_mixed(&with(&options, &streamed), &Fault::Streamed(failure))
                });
                assert!(status.starts_with("Err"), "{what}: {status}");
                assert_eq!((poisoned, kept), (failure != Group(0), 0), "{what}");
            }
        }
    }
}

/// The mixed step's failures with the combinations without the early plan.
#[test]
fn a_mixed_step_fails_streamed_as_batch_beside_the_dspark_candidates() {
    assert_mixed_failures(false);
}

/// The mixed step's failures with the combinations with the early plan.
#[test]
fn a_mixed_step_fails_streamed_as_batch_beside_the_early_dspark_plan() {
    assert_mixed_failures(true);
}

/// With diagnostic step logs, each step-shape record says whether its
/// step's prefill was lowered while it issued (`streamed_lowering`): true
/// for a miss (the first prompt's, of the short-prompt cache, and each new
/// prefill structure's), false for a kept plan's or template's reuse, and
/// false for every step with the option off, with and without the early
/// begin and plan. The record's template counts show the reuse is a hit.
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
    let logs = (DIAGNOSTIC_STEP_LOGS, serde_json::Value::from(true));
    let streamed = (PREFILL_STREAMED_LOWERING, true.into());
    let early = [
        (DSPARK_EARLY_PREFILL_ENGRAM, true.into()),
        (DSPARK_EARLY_PREFILL_PLAN, true.into()),
    ];
    // Eight prompts, the first a miss, then a 512-row miss, its template's
    // reuse by another request and a 600-row miss.
    let on = [vec![true], vec![false; 7], vec![true, false, true]].concat();
    for (options, expected) in [
        (vec![logs.clone()], vec![false; 11]),
        (vec![logs.clone(), streamed.clone()], on.clone()),
        (with(&[logs.clone(), streamed.clone()], &early), on),
    ] {
        let arm = format!("{options:?}");
        let run = schedule.run(&options);
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
        assert_eq!(records, expected, "{arm}");
        let hits: Vec<u64> = field("\"prefill_templates\":")
            .iter()
            .map(|rest| {
                let templates: serde_json::Value =
                    serde_json::from_str(&rest[..=rest.find('}').unwrap()]).unwrap();
                templates["hits"].as_u64().unwrap()
            })
            .collect();
        assert_eq!(hits[8..], [0, 1, 1], "{arm}: the reuse is a template hit");
    }
}
