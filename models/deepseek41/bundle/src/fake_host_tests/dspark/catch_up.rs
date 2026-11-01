use super::*;

pub(super) fn captured_catch_up(step: &[String]) -> Vec<String> {
    let mut projects = false;
    let mut handles = Vec::new();
    for line in step {
        if line.starts_with("capture ") {
            projects = false;
        } else if line.starts_with("launch ")
            && line.contains("chop_deepseek41_dspark_quantize_w15360_bf16_parallel_v1")
        {
            projects = true;
        } else if line.starts_with("captured ") && projects {
            handles.push(line.split_whitespace().nth(2).unwrap().to_owned());
            projects = false;
        }
    }
    handles
}

#[test]
fn catch_up_replay_failure_poison_prevents_another_step() {
    let (host, mut instance) = create(Shape::new(&FLASH, Ranks::TP4).unwrap());
    execute(
        &mut instance,
        &host,
        &super::super::prompt(7, 0, 0, 3, true),
    );
    instance.commit().unwrap();
    host.device.fail_graph_launch.set(Some(1));
    let mut sink = Sink::default();
    let begin = host.journal.borrow().len();
    assert!(instance.execute(&verify(7, 0, 3, 5), &mut sink).is_err());
    let journal = host.journal.borrow();
    assert_eq!(captured_catch_up(&journal[begin..]).len(), 4);
    assert!(
        !journal[begin..]
            .iter()
            .any(|line| line.starts_with("mapped gather"))
    );
    drop(journal);
    assert!(instance.cancel().is_err());
    let begin = host.journal.borrow().len();
    assert!(
        instance
            .execute(&super::super::prompt(8, 1, 0, 3, true), &mut sink)
            .is_err()
    );
    assert!(
        host.journal.borrow()[begin..]
            .iter()
            .all(|line| !line.starts_with("launch ") && !line.starts_with("replay "))
    );
    let released = host.released.get();
    assert!(instance.destroy().is_err());
    assert!(instance.is_poisoned());
    assert_eq!(host.released.get(), released);
}

#[test]
fn catch_up_graphs_reuse_exact_rows_across_slots_and_commits() {
    let (host, mut instance) = create(Shape::new(&FLASH, Ranks::TP4).unwrap());
    for (request, slot) in [(7, 0), (8, 1)] {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(request, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    let mut cached: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    // Catch-up rows depend on the last commit, not the current draft budget.
    for (request, slot, rows, keep) in [
        (7, 0, 3, 1),
        (8, 1, 3, 2),
        (7, 0, 1, 2),
        (8, 1, 2, 3),
        (7, 0, 2, 1),
        (8, 1, 3, 4),
        (8, 1, 4, 5),
        (8, 1, 5, 6),
        (8, 1, 6, 4),
        (8, 1, 4, 5),
        (8, 1, 5, 6),
        (8, 1, 6, 1),
    ] {
        let start = state(&instance, request).end;
        let begin = host.journal.borrow().len();
        execute(&mut instance, &host, &verify(request, slot, start, 5));
        let journal = host.journal.borrow();
        let step = &journal[begin..];
        let captured = captured_catch_up(step);
        if let Some(handles) = cached.get(&rows) {
            assert!(captured.is_empty(), "recaptured unchanged catch-up rows");
            for handle in handles {
                assert_eq!(
                    step.iter()
                        .filter(|line| line.starts_with("replay ")
                            && line.split_whitespace().nth(2) == Some(handle.as_str()))
                        .count(),
                    1
                );
            }
            assert_eq!(catch_up_launches(&host, begin), 0, "catch-up stayed eager");
        } else {
            assert_eq!(captured.len(), 4, "each rank must capture catch-up");
            let nodes: usize = step
                .iter()
                .filter(|line| {
                    line.starts_with("captured ")
                        && captured
                            .iter()
                            .any(|handle| line.split_whitespace().nth(2) == Some(handle.as_str()))
                })
                .map(|line| {
                    line.split_whitespace()
                        .nth(3)
                        .unwrap()
                        .parse::<usize>()
                        .unwrap()
                })
                .sum();
            println!("catch-up rows={rows}: eager launches={nodes}, graph replays=4");
            cached.insert(rows, captured);
        }
        drop(journal);
        instance.commit_accepted(&[kept(request, keep)]).unwrap();
    }
    assert_eq!(cached.len(), 6);
    for request in [7, 8] {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A prefix capture catches the drafter up outside any target step
/// (`instance/prefix.rs`), and with the `verify_row_fusions` option its
/// window writes of at most the draft rows take the parallel ceil32 entry,
/// as a step's do. The eager lifetime (a short prefill, its capture, a
/// restore into another slot and a drafted verify step) with the option is
/// the one without it up to the verify step but for those launches, some
/// of them inside the capture, and its journal is pinned; the verify
/// step's fusions are `verify_rows`' to check.
#[test]
fn prefix_capture_catch_up_quantizes_small_writes_with_the_row_fusions() {
    use model_deepseek41_contract::execution_options::{
        DSPARK_GRAPH_HANDLES_PER_RANK, VERIFY_ROW_FUSIONS,
    };
    const SERIAL: &str = "chop_deepseek41_latent_ceil32_v1 ";
    const PARALLEL: &str = "chop_deepseek41_latent_ceil32_parallel_v1 ";
    let run = |fused: bool| {
        let (host, mut instance) = create_with_options(
            Shape::new(&FLASH, Ranks::TP4).unwrap(),
            4,
            32,
            &[
                (DSPARK_GRAPH_HANDLES_PER_RANK, 0.into()),
                (VERIFY_ROW_FUSIONS, fused.into()),
            ],
        );
        execute(
            &mut instance,
            &host,
            &super::super::prompt(1, 0, 0, 3, true),
        );
        instance.commit().unwrap();
        let pages = (0..instance.describe_prefix_state().unwrap())
            .map(|page| 40 + page)
            .collect::<Vec<_>>();
        let start = host.journal.borrow().len();
        instance.capture_prefix_state(1, 3, &pages).unwrap();
        let capture = start..host.journal.borrow().len();
        instance.retire_request(1).unwrap();
        instance
            .restore_prefix_state_in_slot(2, &[0, 1, 2], &pages, 1)
            .unwrap();
        let step = host.journal.borrow().len();
        execute(&mut instance, &host, &verify(2, 1, 3, 5));
        instance.commit_accepted(&[kept(2, 3)]).unwrap();
        instance.retire_request(2).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        let lines = host.journal.borrow().clone();
        (lines, capture, step)
    };
    let (served, capture, step) = run(false);
    let (fused, fused_capture, fused_step) = run(true);
    assert_eq!((&capture, step), (&fused_capture, fused_step));
    let mut changed = Vec::new();
    for (at, (a, b)) in served[..step].iter().zip(&fused[..step]).enumerate() {
        if a != b {
            assert!(a.starts_with("launch ") && a.contains(SERIAL), "{at}: {a}");
            assert_eq!(
                *b,
                a.replacen(SERIAL, PARALLEL, 1)
                    .replacen("block_x: 32,", "block_x: 512,", 1),
                "{at}"
            );
            changed.push(at);
        }
    }
    // The capture's catch-up writes the three prompt rows in 12 small
    // window writes over the four ranks; nothing else before the step
    // changes.
    assert_eq!(changed.len(), 12);
    assert!(changed.iter().all(|at| capture.contains(at)), "{changed:?}");
    let name = "flash-tp4-dspark-prefix-capture-rowfuse.journal";
    if let Err(error) =
        bundle_runtime::fake_host::check_journal(&fused, (40104, 0x00ca4ee0728ce1b9), name)
    {
        panic!("{name}: {error}");
    }
}
