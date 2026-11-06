//! The `dspark_early_prefill_plan` candidate: a DSpark step plans its
//! prefill while its drafter runs, before the draft readback, and frees
//! the plan templates that planning drops once the prefill launched.
//! Planning issues no device work, so each step's journal (every launch
//! with its arguments, every upload with its bytes), every log line and
//! every result is the default's, with or without the early Engram begin.
use super::*;
use crate::instance::execute::plan_templates;

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

/// On an instance that also serves images, as every served instance does,
/// text-only mixed steps take the prepared verify-row gathers
/// (`dspark_mixed_prepared_engram` admits them per step). Beside those
/// gathers the early begin moves only the prefill gathers' begin, the
/// early plan plans every mixed step while its drafter runs and issues the
/// default's journal, and both together issue the early begin's journal.
#[test]
fn early_candidates_beside_mixed_gathers_on_an_instance_serving_images() {
    use Then::*;
    let steps = [
        step(&[prefill(7, 8, 0, 512)], true, Commit),
        step(&[prefill(8, 9, 0, 512)], true, Commit),
        step(&[prefill(9, 10, 0, 2048)], true, Commit),
    ];
    let schedule = Schedule {
        verify_rows: 32,
        widths: &[4; 8],
        restored: &[],
        steps: &steps,
    };
    let mixed = [(DSPARK_MIXED_PREPARED_ENGRAM, true.into())];
    let begin = with(&mixed, &[(DSPARK_EARLY_PREFILL_ENGRAM, true.into())]);
    let plan = [(DSPARK_EARLY_PREFILL_PLAN, true.into())];
    // The prepared verify-row gathers each step launched, as the route
    // records count them.
    let prepared = |run: &Run| -> Vec<u64> {
        run.logs
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|value| value["schema"] == "chopinfer.route.v1")
            .filter_map(|value| {
                value["entries"]["engram/prepared/dspark_mixed_target"]["launches"].as_u64()
            })
            .collect()
    };
    bundle_runtime::routes::with_logging(|| {
        let (off, none) = planned_serving_images(&schedule, &mixed);
        assert_eq!(none, 0);
        let layers = u64::from(FLASH.engram_layers());
        assert_eq!(prepared(&off).iter().sum::<u64>(), 3 * layers);
        let (early, none) = planned_serving_images(&schedule, &begin);
        assert_eq!(none, 0);
        // Eight prompts and three mixed steps begin early.
        assert_eq!(assert_moved_begins(&off, &early, "early begin"), 8 + 3);
        for (name, base, options) in [
            ("early plan", &off, with(&mixed, &plan)),
            ("both", &early, with(&begin, &plan)),
        ] {
            let (on, plans) = planned_serving_images(&schedule, &options);
            assert_eq!(plans, 3, "{name}");
            assert_eq!(prepared(&on), prepared(base), "{name}");
            assert_same_runs(base, &on, name);
        }
    });
}

/// [`planned`] on an instance that also serves images.
fn planned_serving_images(
    schedule: &Schedule<'_>,
    options: &[(&str, serde_json::Value)],
) -> (Run, u64) {
    let before = crate::instance::execute::early_prefill_plans();
    let run = schedule.run_serving_images(options);
    (
        run,
        crate::instance::execute::early_prefill_plans() - before,
    )
}

/// A reused template's stale Engram plan, which the verify mode refuses,
/// planned while the drafter ran, refuses the step at its target as the
/// default's lowering there does: the same journal, logs and state, with
/// or without the early begin.
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
        assert!(host.races.borrow().is_empty(), "{:#?}", host.races.borrow());
        (journal, logs, state)
    };
    let verify = [(PREFILL_PLAN_CACHE_VERIFY, true.into())];
    for options in [
        verify.to_vec(),
        with(&verify, &[(DSPARK_EARLY_PREFILL_ENGRAM, true.into())]),
    ] {
        let off = refuse(&options);
        let on = refuse(&with(&options, &[(DSPARK_EARLY_PREFILL_PLAN, true.into())]));
        assert_same_lines(&off.0, &on.0, "the refused step");
        assert_same_lines(&off.1, &on.1, "its logs");
        assert_eq!(off.2, on.2);
    }
}

/// A mixed step that fails after its prefill was planned while the drafter
/// ran, at the draft readback (an id outside the vocabulary, before the
/// target) or at the target's last upload, refuses as the default does:
/// the same journal, status and poisoning, with or without the early
/// begin, and no device access touches a pending gather's ids or rows.
/// Only the plan made before a readback failure differs: the default never
/// planned that step, so the candidate's logs hold one more template line.
#[test]
fn failures_after_an_early_plan_refuse_as_the_default_does() {
    let rows = |logs: &[String]| {
        logs.iter()
            .filter(|line| line.contains("prefill plan templates:"))
            .count()
    };
    for early in [false, true] {
        let begin = [(DSPARK_EARLY_PREFILL_ENGRAM, early.into())];
        for failure in ["readback", "upload"] {
            let fail = |options: &[(&str, serde_json::Value)]| {
                // The step's uploads, for failing its last one.
                let uploads = {
                    let (host, mut instance, mixed) = prompted(options, MappedHost::with_row_ids);
                    let begin = host.journal.borrow().len();
                    run(&mut instance, &host, &mixed);
                    let uploads = host.journal.borrow()[begin..]
                        .iter()
                        .filter(|line| line.starts_with("upload "))
                        .count();
                    instance.cancel().unwrap();
                    instance.destroy().unwrap();
                    uploads as u32
                };
                let (host, mut instance, mixed) = prompted(options, MappedHost::with_row_ids);
                let begin = host.journal.borrow().len();
                let logs = host.logs.borrow().len();
                if failure == "readback" {
                    host.injected.replace(Some(vec![0xff; 8 * 5 * 4]));
                } else {
                    host.fail_upload.set(Some(uploads - 1));
                }
                let status = instance.execute_prefill(&mixed, &mut Sink::default());
                assert!(status.is_err(), "{failure}");
                let journal = host.journal.borrow()[begin..].to_vec();
                let logs: Vec<_> = host.logs.borrow()[logs..]
                    .iter()
                    .map(|line| untimed(line))
                    .collect();
                let expected = if failure == "readback" {
                    "outside the vocabulary"
                } else {
                    "execute failed"
                };
                assert!(
                    logs.iter().any(|line| line.contains(expected)),
                    "{failure}: {logs:#?}"
                );
                let state = crate::instance::prefill_engram::refusal_state(instance.handle());
                let _ = instance.cancel();
                match instance.destroy() {
                    Ok(()) => assert_eq!(host.acquired.get(), host.released.get()),
                    Err(_) => assert!(instance.is_poisoned()),
                }
                assert!(host.races.borrow().is_empty(), "{:#?}", host.races.borrow());
                (journal, logs, state, format!("{status:?}"))
            };
            let context = format!("early begin {early}, failed {failure}");
            let (off, plans) = planned_with(|| fail(&begin));
            let (on, early_plans) =
                planned_with(|| fail(&with(&begin, &[(DSPARK_EARLY_PREFILL_PLAN, true.into())])));
            assert_eq!(plans, 0, "{context}");
            // Both runs of the candidate plan early: the dry run and the step.
            assert_eq!(early_plans, 2, "{context}");
            assert_same_lines(&off.0, &on.0, &context);
            assert_eq!((&off.2, &off.3), (&on.2, &on.3), "{context}");
            if failure == "readback" {
                assert_eq!(rows(&on.1), rows(&off.1) + 1, "{context}");
            }
            let other = |logs: &[String]| -> Vec<String> {
                logs.iter()
                    .filter(|line| !line.contains("prefill plan templates:"))
                    .cloned()
                    .collect()
            };
            assert_same_lines(&other(&off.1), &other(&on.1), &format!("{context}: logs"));
            if failure == "upload" {
                assert_same_lines(&off.1, &on.1, &format!("{context}: logs"));
            }
        }
    }
}

/// What `run` returned, and the prefills planned early meanwhile.
fn planned_with<T>(run: impl FnOnce() -> T) -> (T, u64) {
    let before = crate::instance::execute::early_prefill_plans();
    let result = run();
    (
        result,
        crate::instance::execute::early_prefill_plans() - before,
    )
}

/// With room for two templates, prefills of 512, 600, 700 and 512 rows drop
/// the oldest template at each of the last two, and the journals match.
/// With the candidate a dropped template is freed once the prefill that
/// dropped it launched, and one a plan outside a step dropped at the next
/// step's entry, with or without a prefill; without it, at once. The
/// frees are observed as they happen, in order with the prefill's launch.
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
            let (_, events) = crate::instance::execute::observe_events(|| {
                run(&mut instance, &host, &batch(&[], &verifies))
            });
            assert_eq!(events, vec!["plan freed"; held]);
            assert_eq!(plan_templates::retired(instance.handle()), 0);
            let kept: Vec<_> = (0..8).map(|slot| kept(verifier(slot), 4)).collect();
            instance.commit_accepted(&kept).unwrap();
            plan(&instance, &[800, 900]);
            assert_eq!(evicted(&instance), 3);
            assert_eq!(plan_templates::retired(instance.handle()), 2 * held);
            // The mixed step frees those at its entry and the template its
            // own lowering drops once its prefill launched.
            let verifies: Vec<_> = (0..8).map(|slot| (slot, 7, 4)).collect();
            let (_, events) = crate::instance::execute::observe_events(|| {
                run(
                    &mut instance,
                    &host,
                    &batch(&[prefill(7, 8, 0, 512)], &verifies),
                )
            });
            let launch = ["prefill launch", "prefill launched"];
            if held == 0 {
                assert_eq!(events, [&["plan freed"][..], &launch].concat());
            } else {
                let entry = ["plan freed"; 2];
                assert_eq!(events, [&entry[..], &launch, &["plan freed"]].concat());
            }
            assert_eq!(evicted(&instance), 4);
            assert_eq!(plan_templates::retired(instance.handle()), 0);
            instance.cancel().unwrap();
            let _ = instance.destroy();
        }
    });
}
