//! The `dspark_early_prefill_plan` candidate: a DSpark step plans its
//! prefill while its drafter runs, before the draft readback, and frees
//! the plan templates that planning drops once the prefill launched.
//! Planning issues no device work, so each step's journal (every launch
//! with its arguments, every upload with its bytes), every log line and
//! every result is the default's, with or without the early Engram begin.
use super::*;
use crate::instance::execute::plan_templates;
use model_deepseek41_contract::execution_options::DSPARK_EARLY_PREFILL_PLAN;

/// Each step of `on` issues `off`'s journal and results exactly, and every
/// log line matches.
#[track_caller]
fn assert_same_runs(off: &Run, on: &Run, context: &str) {
    assert_eq!(off.steps.len(), on.steps.len(), "{context}");
    for (index, (a, b)) in off.steps.iter().zip(&on.steps).enumerate() {
        let context = format!("{context}, step {index}");
        assert_same_lines(&a.journal, &b.journal, &context);
        assert_eq!(a.drafts, b.drafts, "{context}");
        assert_eq!(a.views, b.views, "{context}");
    }
    assert_same_lines(&off.logs, &on.logs, &format!("{context}: logs"));
}

/// `schedule` with `options`, and the prefills it planned while their
/// drafter ran.
fn planned(schedule: &Schedule<'_>, options: &[(&str, serde_json::Value)]) -> (Run, u64) {
    let before = crate::instance::execute::early_prefill_plans();
    let run = schedule.run(options);
    (
        run,
        crate::instance::execute::early_prefill_plans() - before,
    )
}

fn mixed_steps(run: &Run) -> u64 {
    run.steps
        .iter()
        .filter(|step| step.kind == Kind::Mixed)
        .count() as u64
}

/// The early Engram begin's schedules (template reuse and misses, 511 to
/// 8000 rows, two prefill items, a prefill step without verify rows, a
/// cancel and its retry; graphed, eager, every reused plan lowered again
/// and beside the mixed verify-row gathers), with and without the early
/// begin in both arms: every mixed step plans early, and nothing else.
#[test]
fn early_plans_issue_the_default_journal() {
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
    let plan = [(DSPARK_EARLY_PREFILL_PLAN, true.into())];
    let early: [(&str, serde_json::Value); 1] = [(DSPARK_EARLY_PREFILL_ENGRAM, true.into())];
    for (name, options) in [
        ("graphed", vec![]),
        ("eager", vec![(DSPARK_GRAPH_HANDLES_PER_RANK, 0.into())]),
        ("verified", vec![(PREFILL_PLAN_CACHE_VERIFY, true.into())]),
        ("mixed", vec![(DSPARK_MIXED_PREPARED_ENGRAM, true.into())]),
        ("early begin", early.to_vec()),
        (
            "early begin, verified",
            with(&early, &[(PREFILL_PLAN_CACHE_VERIFY, true.into())]),
        ),
    ] {
        let (off, none) = planned(&schedule, &options);
        let (on, plans) = planned(&schedule, &with(&options, &plan));
        assert_eq!(none, 0, "{name}");
        assert_eq!(plans, mixed_steps(&on), "{name}");
        assert_eq!(plans, 8, "{name}");
        assert_same_runs(&off, &on, name);
    }
}

/// The live shape: 1536-row prefills after a 93,000-token context beside
/// eight verify items, a template reused after another 128 tokens and
/// after a cancel, and a 1537-row miss; and a 66-row eager verify phase
/// beside reused 512-row prefills. Every reused plan is lowered again.
#[test]
fn long_context_and_eager_verify_early_plans_issue_the_default_journal() {
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
    let verify = [(PREFILL_PLAN_CACHE_VERIFY, true.into())];
    for (name, schedule) in [
        (
            "long context",
            Schedule {
                verify_rows: 32,
                widths: &[4; 8],
                restored: &[(20, 8, 93_000), (21, 9, 93_128), (22, 10, 93_000)],
                steps: &long,
            },
        ),
        (
            "eager verify",
            Schedule {
                verify_rows: 192,
                widths: &[6; 11],
                restored: &[],
                steps: &wide,
            },
        ),
    ] {
        for options in [
            verify.to_vec(),
            with(&verify, &[(DSPARK_EARLY_PREFILL_ENGRAM, true.into())]),
        ] {
            let (off, _) = planned(&schedule, &options);
            let (on, plans) = planned(
                &schedule,
                &with(&options, &[(DSPARK_EARLY_PREFILL_PLAN, true.into())]),
            );
            assert_eq!(plans, mixed_steps(&on), "{name}");
            assert_same_runs(&off, &on, name);
        }
    }
}

/// A reused template's stale Engram plan, which the verify mode refuses,
/// planned while the drafter ran, refuses the step at its target as the
/// default's lowering there does: the same journal, logs and state.
#[test]
fn a_refused_early_plan_refuses_the_target_as_the_default_does() {
    let refuse = |options: &[(&str, serde_json::Value)]| {
        let (host, mut instance, mixed) = prompted(options, MappedHost::with_row_ids);
        run(&mut instance, &host, &mixed);
        let kept: Vec<_> = (0..8).map(|slot| kept(verifier(slot), 4)).collect();
        instance.commit_accepted(&kept).unwrap();
        instance.retire_request(7).unwrap();
        let verifies: Vec<_> = (0..8).map(|slot| (slot, 7, 4)).collect();
        let again = batch(&[prefill(8, 9, 0, 512)], &verifies);
        let begin = host.journal.borrow().len();
        let logs = host.logs.borrow().len();
        let refused = crate::instance::execute::skip::with_stale_engram(|| {
            instance.execute_prefill(&again, &mut Sink::default())
        });
        assert!(refused.is_err());
        let journal = host.journal.borrow()[begin..].to_vec();
        let logs: Vec<_> = host.logs.borrow()[logs..]
            .iter()
            .map(|line| untimed(line))
            .collect();
        assert!(
            logs.iter()
                .any(|line| line.contains("differs from a fresh one"))
        );
        let state = crate::instance::prefill_engram::refusal_state(instance.handle());
        let _ = instance.cancel();
        match instance.destroy() {
            Ok(()) => assert_eq!(host.acquired.get(), host.released.get()),
            Err(_) => assert!(instance.is_poisoned()),
        }
        (journal, logs, state)
    };
    let verify = [(PREFILL_PLAN_CACHE_VERIFY, true.into())];
    let off = refuse(&verify);
    let on = refuse(&with(&verify, &[(DSPARK_EARLY_PREFILL_PLAN, true.into())]));
    assert_same_lines(&off.0, &on.0, "the refused step");
    assert_same_lines(&off.1, &on.1, "its logs");
    assert_eq!(off.2, on.2);
}

/// With room for two templates, prefills of 512, 600, 700 and 512 rows drop
/// the oldest template at each of the last two, and the journals match.
/// With the candidate a dropped template is freed once the prefill that
/// dropped it launched, and one a plan outside a step dropped at the next
/// step's entry, with or without a prefill; without it, at once.
#[test]
fn dropped_templates_are_freed_after_the_prefill_launch() {
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
    let plan = [(DSPARK_EARLY_PREFILL_PLAN, true.into())];
    plan_templates::with_limit(2, || {
        let off = schedule.run(&[]);
        let on = schedule.run(&plan);
        assert_same_runs(&off, &on, "two templates");
        let dropped = |run: &Run| {
            run.logs
                .iter()
                .filter(|line| line.contains("dropped the oldest"))
                .count()
        };
        assert_eq!((dropped(&off), dropped(&on)), (2, 2));
        for (options, held) in [(&[][..], 0), (&plan[..], 1)] {
            let (host, mut instance, _) = prompted(options, MappedHost::with_row_ids);
            let plan = |instance: &BundleInstanceV1, rows: &[u64]| {
                for &rows in rows {
                    let (request, _) = original(&[prefill(20, 8, 0, rows)], &[]);
                    crate::instance::execute::plan_timing::plan(instance.handle(), &request)
                        .unwrap();
                }
            };
            let evicted =
                |instance: &BundleInstanceV1| plan_templates::counts(instance.handle()).evicted;
            plan(&instance, &[512, 600, 700]);
            // The third kept template dropped the first, which the
            // candidate holds for the next step.
            assert_eq!(evicted(&instance), 1);
            assert_eq!(plan_templates::retired(instance.handle()), held);
            // A step without a prefill frees it at its entry.
            let verifies: Vec<_> = (0..8).map(|slot| (slot, 3, 4)).collect();
            run(&mut instance, &host, &batch(&[], &verifies));
            assert_eq!(plan_templates::retired(instance.handle()), 0);
            let kept: Vec<_> = (0..8).map(|slot| kept(verifier(slot), 4)).collect();
            instance.commit_accepted(&kept).unwrap();
            plan(&instance, &[800, 900]);
            assert_eq!(evicted(&instance), 3);
            assert_eq!(plan_templates::retired(instance.handle()), 2 * held);
            // The mixed step frees those at its entry and the template its
            // own lowering drops once its prefill launched.
            let verifies: Vec<_> = (0..8).map(|slot| (slot, 7, 4)).collect();
            run(
                &mut instance,
                &host,
                &batch(&[prefill(7, 8, 0, 512)], &verifies),
            );
            assert_eq!(evicted(&instance), 4);
            assert_eq!(plan_templates::retired(instance.handle()), 0);
            instance.cancel().unwrap();
            let _ = instance.destroy();
        }
    });
}
