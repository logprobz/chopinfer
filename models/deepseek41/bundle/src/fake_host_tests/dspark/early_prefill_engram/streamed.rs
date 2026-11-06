//! The `prefill_streamed_lowering` candidate beside the DSpark candidates:
//! a prefill no kept plan or template serves lowers one layer at a time
//! while it issues. Each step's journal (every launch with its arguments,
//! every upload with its bytes, the prepared gathers' begins and joins),
//! every log line and every result is the batch lowering's, with and
//! without each of `dspark_early_prefill_engram`, `dspark_early_prefill_plan`
//! and `dspark_mixed_prepared_engram`, with the upper-skip DMA and without.
//! Only the line that keeps a streamed plan moves: it follows the launch's
//! own lines, as the plan is kept once its launch completed. The streamed
//! plans are the ones the batch run lowered before their launch, but for
//! the verify mode's, and in these schedules each is one the batch run
//! kept as a plan or template.
use super::*;
use crate::instance::execute::{plan_templates, skip};
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
fn assert_streamed(
    schedule: &Schedule<'_>,
    options: &[(&str, Vec<(&'static str, serde_json::Value)>)],
    launches: u64,
) {
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

/// Each DSpark candidate alone, and all three together.
fn candidates() -> Vec<(&'static str, Vec<(&'static str, serde_json::Value)>)> {
    let begin = || (DSPARK_EARLY_PREFILL_ENGRAM, true.into());
    let plan = || (DSPARK_EARLY_PREFILL_PLAN, true.into());
    let mixed = || (DSPARK_MIXED_PREPARED_ENGRAM, true.into());
    vec![
        ("default", vec![]),
        ("early begin", vec![begin()]),
        ("early plan", vec![plan()]),
        ("mixed", vec![mixed()]),
        ("all three", vec![begin(), plan(), mixed()]),
    ]
}

/// The early Engram begin's schedule (template reuse and misses, 511 to
/// 8000 rows, two prefill items, a prefill step without verify rows, a
/// cancel and its retry) with each candidate, and eager (no graph handles)
/// and with every reused plan lowered again beside all three.
#[test]
fn streamed_plans_issue_the_batch_journal_beside_the_dspark_candidates() {
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
    let mut options = candidates();
    let all = options.last().unwrap().1.clone();
    options.push((
        "all three, eager",
        with(&all, &[(DSPARK_GRAPH_HANDLES_PER_RANK, 0.into())]),
    ));
    options.push((
        "all three, verified",
        with(&all, &[(PREFILL_PLAN_CACHE_VERIFY, true.into())]),
    ));
    // The first prompt and the six prefill structures.
    assert_streamed(&schedule, &options, 7);
}

/// The live shape: 1536-row prefills after a 93,000-token context beside
/// eight verify items, a template reused after another 128 tokens and
/// after a cancel, and a 1537-row miss; and a 66-row eager verify phase
/// beside a 512-row template and its reuse. Every reused plan is lowered
/// again.
#[test]
fn long_context_and_eager_verify_streamed_plans_issue_the_batch_journal() {
    use Then::*;
    let long = [
        step(&[prefill(20, 8, 93_000, 1536)], true, Commit),
        step(&[prefill(21, 9, 93_128, 1536)], true, CancelRetry),
        step(&[prefill(22, 10, 93_000, 1537)], true, Commit),
    ];
    let wide = [
        step(&[prefill(7, 11, 0, 512)], true, Commit),
        step(&[prefill(8, 12, 0, 512)], true, Commit),
    ];
    let verify = (PREFILL_PLAN_CACHE_VERIFY, true.into());
    let options: Vec<_> = candidates()
        .into_iter()
        .map(|(name, options)| (name, with(&options, std::slice::from_ref(&verify))))
        .collect();
    for (launches, schedule) in [
        (
            // The first prompt and the two prefill structures.
            3,
            Schedule {
                verify_rows: 32,
                widths: &[4; 8],
                restored: &[(20, 8, 93_000), (21, 9, 93_128), (22, 10, 93_000)],
                steps: &long,
            },
        ),
        (
            // The first prompt and the prefill structure.
            2,
            Schedule {
                verify_rows: 192,
                widths: &[6; 11],
                restored: &[],
                steps: &wide,
            },
        ),
    ] {
        assert_streamed(&schedule, &options, launches);
    }
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
        let options = [("default", vec![]), ("early plan", vec![plan.clone()])];
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
