//! A DSpark step whose prefill takes the prepared gathers plans that
//! prefill while its drafter runs, before the draft readback, and frees
//! the plan templates that planning drops once the prefill launched.
//! Planning issues no device work; the journal pins hold each step's
//! journal, and these tests count the early plans, their step-shape
//! records and template counts, the order of the frees, and the refusals
//! after an early plan.
use super::*;
use crate::instance::execute::plan_templates;

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
/// cancel and its retry; graphed, eager and every reused plan lowered
/// again, each beside the mixed verify-row gathers and the early begin):
/// every mixed step plans early, and nothing else.
#[test]
fn every_mixed_step_plans_early() {
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
    for (name, options) in [
        ("graphed", vec![]),
        ("eager", vec![(DSPARK_GRAPH_HANDLES_PER_RANK, 0.into())]),
        ("verified", vec![(PREFILL_PLAN_CACHE_VERIFY, true.into())]),
    ] {
        let staged = crate::dspark::staging::early_stagings();
        let (run, plans) = planned(&schedule, &options);
        let staged = crate::dspark::staging::early_stagings() - staged;
        assert_eq!(plans, mixed_steps(&run), "{name}");
        assert_eq!(plans, 8, "{name}");
        // The early begin stages every prefill.
        assert_eq!(staged, run.steps.len() as u64, "{name}");
    }
}

/// The live shape: 1536-row prefills after a 93,000-token context beside
/// eight verify items, a template reused after another 128 tokens and
/// after a cancel, and a 1537-row miss; and a 66-row eager verify phase
/// beside reused 512-row prefills. Every reused plan is lowered again, and
/// every mixed step plans early.
#[test]
fn long_context_and_eager_verify_steps_plan_early() {
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
        let (run, plans) = planned(&schedule, &verify);
        assert_eq!(plans, mixed_steps(&run), "{name}");
    }
}

/// With diagnostic step logs, each step-shape record says whether its
/// step's prefill began its gathers early and whether it planned early:
/// every prefill begins early, and only the mixed step, which drafts,
/// plans early.
#[test]
fn step_shape_records_report_early_begins_and_plans() {
    use Then::*;
    use model_deepseek41_contract::execution_options::DIAGNOSTIC_STEP_LOGS;
    let steps = [step(&[prefill(7, 8, 0, 512)], true, Commit)];
    let schedule = Schedule {
        verify_rows: 32,
        widths: &[4; 8],
        restored: &[],
        steps: &steps,
    };
    let run = schedule.run(&[(DIAGNOSTIC_STEP_LOGS, true.into())]);
    let records: Vec<_> = run
        .logs
        .iter()
        .filter(|line| line.contains("\"early_prefill_begin\":"))
        .collect();
    let count = |text: &str| records.iter().filter(|line| line.contains(text)).count();
    assert!(!records.is_empty());
    assert_eq!(count("\"early_prefill_begin\":true"), records.len());
    assert_eq!(count("\"early_prefill_plan\":true"), 1);
}

/// With diagnostic step logs, each step-shape record holds the instance's
/// prefill plan templates after its step, so consecutive records give each
/// mixed step's template miss, reuse and drop. With room for two templates: a 512-row miss, its
/// reuse by another request, a 600-row miss and a 700-row miss that drops
/// the oldest.
#[test]
fn step_shape_records_count_each_steps_template_reuse() {
    use Then::*;
    use model_deepseek41_contract::execution_options::DIAGNOSTIC_STEP_LOGS;
    let steps = [
        step(&[prefill(7, 8, 0, 512)], true, Commit),
        step(&[prefill(8, 9, 0, 512)], true, Commit),
        step(&[prefill(9, 10, 0, 600)], true, Commit),
        step(&[prefill(10, 11, 0, 700)], true, Commit),
    ];
    let schedule = Schedule {
        verify_rows: 32,
        widths: &[4; 8],
        restored: &[],
        steps: &steps,
    };
    plan_templates::with_limit(2, || {
        let run = schedule.run(&[(DIAGNOSTIC_STEP_LOGS, true.into())]);
        // The run's lines have their host times masked, so only the
        // templates object is parsed.
        let marker = "\"prefill_templates\":";
        let records: Vec<serde_json::Value> = run
            .logs
            .iter()
            .filter_map(|line| line.find(marker).map(|at| &line[at + marker.len()..]))
            .map(|rest| serde_json::from_str(&rest[..=rest.find('}').unwrap()]).unwrap())
            .collect();
        // One record a step, the eight prompts' first.
        assert_eq!(records.len(), 8 + steps.len());
        let count = |record: &serde_json::Value, name: &str| record[name].as_u64().unwrap();
        let deltas: Vec<_> = records
            .windows(2)
            .skip(records.len() - 1 - steps.len())
            .map(|pair| {
                ["hits", "misses", "kept", "dropped"]
                    .map(|name| count(&pair[1], name) - count(&pair[0], name))
            })
            .collect();
        assert_eq!(
            deltas,
            [[0, 1, 1, 0], [1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1]]
        );
        let last = records.last().unwrap();
        assert_eq!((count(last, "templates"), count(last, "limit")), (2, 2));
        assert!(records.iter().all(|r| count(r, "templates") <= 2));
    });
}

/// On an instance that also serves images, as every served instance does,
/// text-only mixed steps take the prepared verify-row gathers
/// (`admits_mixed_engram` admits them per step). Beside those gathers
/// every prompt and mixed step begins its prefill gathers early, and every
/// mixed step plans while its drafter runs.
#[test]
fn early_begins_and_plans_beside_mixed_gathers_on_an_instance_serving_images() {
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
        let (run, plans) = planned_serving_images(&schedule);
        assert_eq!(plans, 3);
        let layers = u64::from(FLASH.engram_layers());
        assert_eq!(prepared(&run).iter().sum::<u64>(), 3 * layers);
        // Eight prompts and three mixed steps begin early.
        assert_eq!(assert_early_begins(&run, "early begin"), 8 + 3);
    });
}

/// [`planned`] on an instance that also serves images.
fn planned_serving_images(schedule: &Schedule<'_>) -> (Run, u64) {
    let before = crate::instance::execute::early_prefill_plans();
    let run = schedule.run_serving_images(&[]);
    (
        run,
        crate::instance::execute::early_prefill_plans() - before,
    )
}

/// A reused template's stale Engram plan, which the verify mode refuses,
/// planned while the drafter ran, refuses the step at its target: the
/// runtime is poisoned and the step's gathers are settled.
#[test]
fn a_refused_early_plan_refuses_the_step_at_its_target() {
    let options = [(PREFILL_PLAN_CACHE_VERIFY, true.into())];
    let (host, mut instance, mixed) = prompted(&options, MappedHost::with_row_ids);
    run(&mut instance, &host, &mixed);
    let kept: Vec<_> = (0..8).map(|slot| kept(verifier(slot), 4)).collect();
    instance.commit_accepted(&kept).unwrap();
    instance.retire_request(7).unwrap();
    let verifies: Vec<_> = (0..8).map(|slot| (slot, 7, 4)).collect();
    let again = batch(&[prefill(8, 9, 0, 512)], &verifies);
    let logs = host.logs.borrow().len();
    let (refused, plans) = planned_with(|| {
        crate::instance::execute::skip::with_stale_engram(|| {
            instance.execute_prefill(&again, &mut Sink::default())
        })
    });
    assert!(refused.is_err());
    assert_eq!(plans, 1);
    assert!(
        host.logs.borrow()[logs..]
            .iter()
            .any(|line| line.contains("differs from a fresh one"))
    );
    // The step's gathers are settled and its runtime is poisoned.
    assert_eq!(
        crate::instance::prefill_engram::refusal_state(instance.handle()),
        (true, true)
    );
    let _ = instance.cancel();
    match instance.destroy() {
        Ok(()) => assert_eq!(host.acquired.get(), host.released.get()),
        Err(_) => assert!(instance.is_poisoned()),
    }
    assert!(host.races.borrow().is_empty(), "{:#?}", host.races.borrow());
}

/// A mixed step that fails after its prefill was planned while the drafter
/// ran, at the draft readback (an id outside the vocabulary, before the
/// target) or at the target's last upload, refuses the step, settles its
/// gathers and poisons the DSpark runtime, and no device access touches a
/// pending gather's ids or rows. The early plan, a miss that keeps its
/// template once its launch completed, keeps it before the upload failure
/// only.
#[test]
fn failures_after_an_early_plan_refuse_the_step() {
    let rows = |logs: &[String]| {
        logs.iter()
            .filter(|line| line.contains("prefill plan templates:"))
            .count()
    };
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
            let logs = host.logs.borrow().len();
            if failure == "readback" {
                host.injected.replace(Some(vec![0xff; 8 * 5 * 4]));
            } else {
                host.fail_upload.set(Some(uploads - 1));
            }
            let status = instance.execute_prefill(&mixed, &mut Sink::default());
            assert!(status.is_err(), "{failure}");
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
            (logs, state)
        };
        let context = format!("failed {failure}");
        // Both runs plan early: the dry run and the step.
        let ((logs, state), plans) = planned_with(|| fail(&[]));
        assert_eq!(plans, 2, "{context}");
        assert_eq!(rows(&logs), usize::from(failure == "upload"), "{context}");
        assert_eq!(state, (true, true), "{context}");
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
/// the oldest template at each of the last two. A dropped template is freed
/// once the prefill that dropped it launched, and one a plan outside a step
/// dropped at the next step's entry, with or without a prefill. The frees
/// are observed as they happen, in order with the prefill's launch.
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
    plan_templates::with_limit(2, || {
        let dropped = schedule
            .run(&[])
            .logs
            .iter()
            .filter(|line| line.contains("dropped the oldest"))
            .count();
        assert_eq!(dropped, 2);
        let (host, mut instance, _) = prompted(&[], MappedHost::with_row_ids);
        let plan = |instance: &BundleInstanceV1, rows: &[u64]| {
            for &rows in rows {
                let (request, _) = original(&[prefill(20, 8, 0, rows)], &[]);
                crate::instance::execute::plan_timing::plan(instance.handle(), &request).unwrap();
            }
        };
        let evicted =
            |instance: &BundleInstanceV1| plan_templates::counts(instance.handle()).evicted;
        plan(&instance, &[512, 600, 700]);
        // The third kept template dropped the first, which waits for the
        // next step.
        assert_eq!(evicted(&instance), 1);
        assert_eq!(plan_templates::retired(instance.handle()), 1);
        // A step without a prefill frees it at its entry.
        let verifies: Vec<_> = (0..8).map(|slot| (slot, 3, 4)).collect();
        let (_, events) = crate::instance::execute::observe_events(|| {
            run(&mut instance, &host, &batch(&[], &verifies))
        });
        assert_eq!(events, ["plan freed"]);
        assert_eq!(plan_templates::retired(instance.handle()), 0);
        let kept: Vec<_> = (0..8).map(|slot| kept(verifier(slot), 4)).collect();
        instance.commit_accepted(&kept).unwrap();
        plan(&instance, &[800, 900]);
        assert_eq!(evicted(&instance), 3);
        assert_eq!(plan_templates::retired(instance.handle()), 2);
        // The mixed step frees those at its entry and the template its own
        // lowering drops once its prefill launched.
        let verifies: Vec<_> = (0..8).map(|slot| (slot, 7, 4)).collect();
        let (_, events) = crate::instance::execute::observe_events(|| {
            run(
                &mut instance,
                &host,
                &batch(&[prefill(7, 8, 0, 512)], &verifies),
            )
        });
        assert_eq!(
            events,
            [
                "plan freed",
                "plan freed",
                "prefill launch",
                "prefill launched",
                "plan freed"
            ]
        );
        assert_eq!(evicted(&instance), 4);
        assert_eq!(plan_templates::retired(instance.handle()), 0);
        instance.cancel().unwrap();
        let _ = instance.destroy();
    });
}
