use std::sync::Arc;

use runtime_kv::{KvGeometry, PrefixChainV1};
use runtime_scheduler::{
    BatchOutcome, BatchPhase, PrefixRequestV1, RequestId, RequestSpec, RequestState, Scheduler,
    SchedulerConfig,
};

fn scheduler(chunk: usize, guaranteed: bool) -> Scheduler {
    scheduler_with_cuts(chunk, guaranteed, false)
}

fn scheduler_with_cuts(chunk: usize, guaranteed: bool, cuts: bool) -> Scheduler {
    let mut config = SchedulerConfig::new(4, 64, 8192, chunk, 1)
        .unwrap()
        .with_prefill_granule(64)
        .unwrap()
        .with_maximum_scored_rows(64)
        .unwrap()
        .with_prefix_cache(1)
        .with_prefill_cuts(cuts);
    if guaranteed {
        config = config.with_guaranteed_prefill_chunks(chunk).unwrap();
    }
    Scheduler::new(
        config.with_lone_prefill_expansion(true),
        KvGeometry::new(64, 8192).unwrap(),
    )
    .unwrap()
}

fn submit(s: &mut Scheduler, id: u64, tokens: &[u32], scored: bool) {
    let mut spec = RequestSpec::new(RequestId::new(id), tokens.len(), 1).unwrap();
    if scored {
        spec = spec.with_prompt_scoring();
    }
    let chain = PrefixChainV1::new(64, Arc::from(tokens), Vec::new()).unwrap();
    s.submit_with_prefix(spec, Some(PrefixRequestV1::new(Arc::new(chain), !scored)))
        .unwrap();
}

fn step(s: &mut Scheduler) -> (Vec<(usize, usize)>, Vec<usize>) {
    let batch = s.next_batch().unwrap().unwrap();
    let rows = batch
        .items()
        .iter()
        .zip(batch.kv_plan().appends())
        .map(|(item, append)| {
            let BatchPhase::Prefill { tokens } = item.phase() else {
                panic!("prefill expected")
            };
            (append.previous_tokens(), tokens)
        })
        .collect();
    let outcomes = batch
        .items()
        .iter()
        .map(|item| {
            BatchOutcome::success(
                item.request_id(),
                item.phase(),
                usize::from(item.expects_output()),
                item.expects_output(),
            )
        })
        .collect::<Vec<_>>();
    s.commit_batch(batch.id(), &outcomes).unwrap();
    let captures = s.take_prefix_captures();
    let ends = captures.iter().map(|capture| capture.tokens()).collect();
    for capture in captures {
        let exists = s.has_capture_anchor(&capture);
        assert!(s.begin_prefix_capture(&capture).unwrap().is_some() || exists);
        s.end_prefix_capture(&capture).unwrap();
    }
    s.audit().unwrap();
    (rows, ends)
}

fn finish(s: &mut Scheduler, id: u64) -> (Vec<(usize, usize)>, Vec<usize>) {
    let mut rows = Vec::new();
    let mut captures = Vec::new();
    while s.request_state(RequestId::new(id)).unwrap() != RequestState::Finished {
        let (r, c) = step(s);
        rows.extend(r);
        captures.extend(c);
    }
    (rows, captures)
}

#[test]
fn fixed_grid_survives_lone_expansion_and_captures_the_unaligned_prompt_end() {
    for chunk in [2048, 8192] {
        let mut s = scheduler(chunk, true);
        let n = 2 * chunk + 17;
        let tokens = (0..n as u32).collect::<Vec<_>>();
        submit(&mut s, 1, &tokens, false);
        assert_eq!(
            s.prefill_boundary_contract(RequestId::new(1)).unwrap(),
            Some((chunk, n))
        );
        let (rows, captures) = finish(&mut s, 1);
        assert_eq!(rows, [(0, chunk), (chunk, chunk), (2 * chunk, 17)]);
        assert_eq!(captures, [chunk, 2 * chunk, n]);
    }
}

#[test]
fn shared_prefix_cannot_add_an_incomplete_boundary() {
    for guaranteed in [false, true] {
        let mut s = scheduler(256, guaranteed);
        let first = (0..529).collect::<Vec<u32>>();
        submit(&mut s, 1, &first, false);
        finish(&mut s, 1);
        let mut second = first[..192].to_vec();
        second.extend(1000..1337);
        submit(&mut s, 2, &second, false);
        let (rows, captures) = step(&mut s);
        let expected = if guaranteed { 256 } else { 192 };
        assert_eq!(rows, [(0, expected)]);
        assert_eq!(captures, [expected]);
        finish(&mut s, 2);
    }
}

#[test]
fn declared_cuts_capture_observed_prefixes_and_restore_them() {
    for cuts in [false, true] {
        let mut s = scheduler_with_cuts(256, true, cuts);
        let first = (0..529).collect::<Vec<u32>>();
        submit(&mut s, 1, &first, false);
        finish(&mut s, 1);
        let mut second = first[..192].to_vec();
        second.extend(1000..1337);
        submit(&mut s, 2, &second, false);
        let (rows, captures) = step(&mut s);
        let expected = if cuts { 192 } else { 256 };
        assert_eq!(rows, [(0, expected)]);
        assert_eq!(captures, [expected]);
        let (rows, _) = finish(&mut s, 2);
        assert_eq!(
            rows[0],
            (expected, 512.min((expected / 256 + 1) * 256) - expected)
        );
        let mut third = first[..192].to_vec();
        third.extend(2000..2337);
        submit(&mut s, 3, &third, false);
        s.admit().unwrap();
        assert_eq!(
            s.cached_tokens(RequestId::new(3)).unwrap(),
            if cuts { 192 } else { 0 }
        );
        if cuts {
            s.complete_prefix_restore(RequestId::new(3)).unwrap();
            let (rows, _) = finish(&mut s, 3);
            assert_eq!(rows[0], (192, 64));
        } else {
            finish(&mut s, 3);
        }
        s.audit().unwrap();
    }
}

#[test]
fn declared_cuts_add_rounded_prompt_anchors_for_repeats() {
    for cuts in [false, true] {
        let mut s = scheduler_with_cuts(256, true, cuts);
        let tokens = (0..465).collect::<Vec<u32>>();
        submit(&mut s, 1, &tokens, false);
        let (rows, captures) = finish(&mut s, 1);
        if cuts {
            assert_eq!(rows, [(0, 256), (256, 192), (448, 17)]);
            assert_eq!(captures, [256, 448, 465]);
        } else {
            assert_eq!(captures, [256, 465]);
        }
        submit(&mut s, 2, &tokens, false);
        s.admit().unwrap();
        assert_eq!(
            s.cached_tokens(RequestId::new(2)).unwrap(),
            if cuts { 448 } else { 256 }
        );
        s.complete_prefix_restore(RequestId::new(2)).unwrap();
        finish(&mut s, 2);
    }
}

#[test]
fn declared_cuts_share_queued_requests_at_the_last_whole_granule() {
    for cuts in [false, true] {
        let mut s = scheduler_with_cuts(256, true, cuts);
        let first = (0..529).collect::<Vec<u32>>();
        let mut second = first[..209].to_vec();
        second.extend(1000..1320);
        submit(&mut s, 1, &first, false);
        submit(&mut s, 2, &second, false);
        s.share_prompt_prefixes(&[RequestId::new(1), RequestId::new(2)])
            .unwrap();
        let (rows, captures) = step(&mut s);
        if cuts {
            assert_eq!(rows, [(0, 192)]);
            assert_eq!(captures, [192]);
            s.admit().unwrap();
            assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 192);
            s.complete_prefix_restore(RequestId::new(2)).unwrap();
        } else {
            assert!(rows.iter().all(|row| *row == (0, 256)));
        }
        finish(&mut s, 1);
        finish(&mut s, 2);
    }
}

#[test]
fn a_prompt_end_hit_resumes_to_the_next_absolute_grid_boundary() {
    let mut s = scheduler(256, true);
    let first = (0..529).collect::<Vec<u32>>();
    submit(&mut s, 1, &first, false);
    finish(&mut s, 1);
    let mut second = first.clone();
    second.extend(1000..1500);
    submit(&mut s, 2, &second, false);
    s.admit().unwrap();
    assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 529);
    s.complete_prefix_restore(RequestId::new(2)).unwrap();
    let (rows, captures) = finish(&mut s, 2);
    assert_eq!(rows, [(529, 239), (768, 256), (1024, 5)]);
    assert_eq!(captures, [768, 1024, 1029]);
}

#[test]
fn prompt_scores_keep_the_full_schedule_without_a_skip_contract() {
    let mut s = scheduler(256, true);
    submit(&mut s, 1, &(0..321).collect::<Vec<_>>(), true);
    assert_eq!(
        s.prefill_boundary_contract(RequestId::new(1)).unwrap(),
        None
    );
    let (rows, _) = finish(&mut s, 1);
    assert!(rows.iter().all(|&(_, count)| count <= 64));
    assert_eq!(rows.iter().map(|&(_, count)| count).sum::<usize>(), 321);
}

#[test]
fn an_abandoned_restore_reinstates_all_cold_capture_points() {
    let mut s = scheduler(256, true);
    let tokens = (0..529).collect::<Vec<u32>>();
    submit(&mut s, 1, &tokens, false);
    finish(&mut s, 1);
    submit(&mut s, 2, &tokens, false);
    s.admit().unwrap();
    assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 512);
    s.abandon_prefix_restore(RequestId::new(2)).unwrap();
    let (rows, captures) = finish(&mut s, 2);
    assert_eq!(rows, [(0, 256), (256, 256), (512, 17)]);
    assert_eq!(captures, [256, 512, 529]);
}

#[test]
fn terminal_decode_does_not_publish_response_state() {
    let mut s = scheduler(2048, true);
    let first = vec![3; 65];
    let id = RequestId::new(1);
    let chain = PrefixChainV1::new(64, Arc::from(first.clone()), Vec::new()).unwrap();
    s.submit_with_prefix(
        RequestSpec::new(id, first.len(), 2).unwrap(),
        Some(PrefixRequestV1::new(Arc::new(chain), true)),
    )
    .unwrap();
    let batch = s.next_batch().unwrap().unwrap();
    s.commit_batch(
        batch.id(),
        &[BatchOutcome::success(
            id,
            batch.items()[0].phase(),
            1,
            false,
        )],
    )
    .unwrap();
    for capture in s.take_prefix_captures() {
        s.begin_prefix_capture(&capture).unwrap();
        s.end_prefix_capture(&capture).unwrap();
    }
    let batch = s.next_batch().unwrap().unwrap();
    assert_eq!(batch.items()[0].phase(), BatchPhase::Decode);
    let mut retained = first;
    retained.push(7);
    s.commit_batch(
        batch.id(),
        &[BatchOutcome::success(id, BatchPhase::Decode, 1, true)],
    )
    .unwrap();
    let captures = s.take_prefix_captures();
    assert!(captures.is_empty());
    // The following turn recomputes the response from the prompt anchor.
    retained.push(9);
    submit(&mut s, 2, &retained, false);
    s.admit().unwrap();
    assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 65);
    s.audit().unwrap();
}

#[test]
fn declared_cuts_refresh_shared_prefixes_after_a_shorter_grid_hit() {
    for (guaranteed, cuts, final_hit) in
        [(false, false, 384), (true, false, 256), (true, true, 384)]
    {
        let mut s = scheduler_with_cuts(256, guaranteed, cuts);
        let first = (0..449).collect::<Vec<u32>>();
        submit(&mut s, 1, &first, false);
        finish(&mut s, 1);
        let mut second = first[..384].to_vec();
        second.extend(1000..1065);
        submit(&mut s, 2, &second, false);
        s.admit().unwrap();
        let hit = s.cached_tokens(RequestId::new(2)).unwrap();
        assert_eq!(hit, if guaranteed { 256 } else { 0 });
        if hit > 0 {
            s.complete_prefix_restore(RequestId::new(2)).unwrap();
        }
        let (_, captures) = finish(&mut s, 2);
        assert_eq!(
            captures.contains(&384),
            final_hit == 384,
            "guaranteed={guaranteed}, cuts={cuts}"
        );
        let mut third = first[..384].to_vec();
        third.extend(2000..2065);
        submit(&mut s, 3, &third, false);
        s.admit().unwrap();
        assert_eq!(s.cached_tokens(RequestId::new(3)).unwrap(), final_hit);
        s.complete_prefix_restore(RequestId::new(3)).unwrap();
        finish(&mut s, 3);
        s.audit().unwrap();
    }
}

#[test]
fn declared_repeat_cuts_must_improve_on_the_last_grid_anchor() {
    for prompt in [1088, 1024] {
        for (guaranteed, cuts) in [(true, false), (false, false), (false, true), (true, true)] {
            let mut s = scheduler_with_cuts(256, guaranteed, cuts);
            let first = (0..prompt).collect::<Vec<u32>>();
            submit(&mut s, 1, &first, false);
            finish(&mut s, 1);
            let mut second = first[..192].to_vec();
            second.extend(2000..2000 + prompt - 192);
            submit(&mut s, 2, &second, false);
            let (rows, captures) = finish(&mut s, 2);
            let repeat_hit = if guaranteed {
                let mut expected = (256..prompt as usize).step_by(256).collect::<Vec<_>>();
                expected.push(prompt as usize);
                if cuts {
                    expected.push(192);
                    if prompt == 1024 {
                        expected.push(960);
                    }
                }
                expected.sort_unstable();
                assert_eq!(captures, expected, "prompt={prompt}, cuts={cuts}");
                if prompt == 1088 && cuts {
                    assert_eq!(
                        rows,
                        [
                            (0, 192),
                            (192, 64),
                            (256, 256),
                            (512, 256),
                            (768, 256),
                            (1024, 64)
                        ]
                    );
                }
                if cuts && prompt == 1024 {
                    960
                } else {
                    (prompt as usize - 1) / 256 * 256
                }
            } else {
                // Ordinary full-prefill profiles keep their shifted repeat cut.
                assert_eq!(captures, [192, 960, prompt as usize]);
                960
            };
            submit(&mut s, 3, &second, false);
            s.admit().unwrap();
            assert_eq!(s.cached_tokens(RequestId::new(3)).unwrap(), repeat_hit);
            s.complete_prefix_restore(RequestId::new(3)).unwrap();
            finish(&mut s, 3);
            s.audit().unwrap();
        }
    }
}

#[test]
fn abandoned_declared_restore_does_not_observe_its_own_prompt() {
    for (guaranteed, cuts, abandon) in [
        (true, false, true),
        (false, false, true),
        (false, true, true),
        (true, true, false),
        (true, true, true),
    ] {
        let mut s = scheduler_with_cuts(256, guaranteed, cuts);
        let first = (0..512).collect::<Vec<u32>>();
        submit(&mut s, 1, &first, false);
        finish(&mut s, 1);
        let mut second = first.clone();
        second.extend(3000..3512);
        submit(&mut s, 2, &second, false);
        s.admit().unwrap();
        assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 512);
        if abandon {
            s.abandon_prefix_restore(RequestId::new(2)).unwrap();
        } else {
            s.complete_prefix_restore(RequestId::new(2)).unwrap();
        }
        let (rows, captures) = finish(&mut s, 2);
        if guaranteed && abandon {
            assert_eq!(rows, [(0, 256), (256, 256), (512, 256), (768, 256)]);
            assert_eq!(captures, [256, 512, 768, 1024]);
        } else {
            assert_eq!(captures, [768, 1024]);
        }
        assert!(!captures.contains(&960));
        s.audit().unwrap();
    }
}

#[test]
fn aligned_short_prompt_keeps_an_anchor_for_its_first_repeat() {
    let config = SchedulerConfig::new(4, 64, 256, 256, 1)
        .unwrap()
        .with_prefill_granule(64)
        .unwrap()
        .with_prefix_cache(1)
        .with_aligned_prompt_capture(true);
    let mut s = Scheduler::new(config, KvGeometry::new(64, 128).unwrap()).unwrap();
    let prompt: Vec<_> = (0..128).collect();
    submit(&mut s, 1, &prompt, false);
    let (rows, captures) = finish(&mut s, 1);
    assert_eq!(captures, [64, 128]);
    assert_eq!(rows, [(0, 64), (64, 64)]);
    submit(&mut s, 2, &prompt, false);
    s.admit().unwrap();
    assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 64);
}

#[test]
fn aligned_capture_respects_grid_admission_scored_rows_and_model_granules() {
    for granule in [1usize, 8, 64] {
        let chunk = 4 * granule;
        for guaranteed in [false, true] {
            for cuts in [false, true] {
                for scored in [false, true] {
                    for length in [
                        granule.saturating_sub(1).max(1),
                        granule,
                        granule + 1,
                        2 * granule,
                        chunk - 1,
                        chunk,
                        chunk + 1,
                    ] {
                        let mut config = SchedulerConfig::new(4, 64, 2 * chunk, chunk, 1)
                            .unwrap()
                            .with_prefill_granule(granule)
                            .unwrap()
                            .with_maximum_scored_rows(chunk)
                            .unwrap()
                            .with_prefix_cache(1)
                            .with_aligned_prompt_capture(true)
                            .with_prefill_cuts(cuts);
                        if guaranteed {
                            config = config.with_guaranteed_prefill_chunks(chunk).unwrap();
                        }
                        let mut s =
                            Scheduler::new(config, KvGeometry::new(64, 128).unwrap()).unwrap();
                        let tokens: Vec<_> = (0..length as u32).collect();
                        submit(&mut s, 1, &tokens, scored);
                        let (rows, captures) = finish(&mut s, 1);
                        assert_eq!(rows.iter().map(|(_, n)| n).sum::<usize>(), length);
                        assert!(rows.iter().all(|(_, n)| *n > 0 && *n <= chunk));
                        if guaranteed && !cuts && !scored {
                            assert!(captures.iter().all(|&n| n == length || n % chunk == 0));
                        } else if length == 2 * granule && granule >= 64 {
                            assert_eq!(captures, [granule, length]);
                        }
                        submit(&mut s, 2, &tokens, scored);
                        s.admit().unwrap();
                        if scored {
                            assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 0);
                        }
                        s.audit().unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn aligned_candidate_does_not_cut_granule_one_prompts() {
    for enabled in [false, true] {
        let config = SchedulerConfig::new(4, 64, 256, 256, 1)
            .unwrap()
            .with_prefix_cache(1)
            .with_aligned_prompt_capture(enabled);
        let mut s = Scheduler::new(config, KvGeometry::new(64, 128).unwrap()).unwrap();
        submit(&mut s, 1, &(0..100).collect::<Vec<_>>(), false);
        let (rows, captures) = finish(&mut s, 1);
        assert_eq!(rows, [(0, 100)]);
        assert_eq!(captures, [100]);
    }
}

#[test]
fn aligned_short_capture_does_not_split_media() {
    let config = SchedulerConfig::new(4, 64, 256, 256, 1)
        .unwrap()
        .with_prefill_granule(64)
        .unwrap()
        .with_prefix_cache(1)
        .with_aligned_prompt_capture(true);
    let mut s = Scheduler::new(config, KvGeometry::new(64, 128).unwrap()).unwrap();
    let chain = PrefixChainV1::new(
        64,
        Arc::from(vec![1; 128]),
        vec![runtime_kv::PrefixMediaSpanV1 {
            identity: [3; 32],
            start: 32,
            rows: 64,
            media_row: 0,
        }],
    )
    .unwrap();
    s.submit_with_prefix(
        RequestSpec::new(RequestId::new(1), 128, 1).unwrap(),
        Some(PrefixRequestV1::new(Arc::new(chain), true)),
    )
    .unwrap();
    let (rows, captures) = finish(&mut s, 1);
    assert_eq!(rows, [(0, 128)]);
    assert_eq!(captures, [128]);
}

fn scheduler_with_grid_anchors(
    chunk: usize,
    cuts: bool,
    grid: usize,
    manifest: usize,
) -> Scheduler {
    let config = SchedulerConfig::new(4, 64, 8192, chunk, 1)
        .unwrap()
        .with_prefill_granule(64)
        .unwrap()
        .with_maximum_scored_rows(64)
        .unwrap()
        .with_prefix_cache(1)
        .with_prefill_cuts(cuts)
        .with_anchor_items(manifest)
        .with_guaranteed_prefill_chunks(chunk)
        .unwrap()
        .with_grid_anchor_items(grid);
    Scheduler::new(
        config.with_lone_prefill_expansion(true),
        KvGeometry::new(64, 8192).unwrap(),
    )
    .unwrap()
}

/// Each prefill item of the next batch: (request, start, tokens, anchor).
fn anchored_step(s: &mut Scheduler) -> Vec<(u64, usize, usize, Option<usize>)> {
    let batch = s.next_batch().unwrap().unwrap();
    let items = batch
        .items()
        .iter()
        .zip(batch.kv_plan().appends())
        .filter_map(|(item, append)| match item.phase() {
            BatchPhase::Prefill { tokens } => Some((
                item.request_id().get(),
                append.previous_tokens(),
                tokens,
                item.anchor_tokens(),
            )),
            _ => None,
        })
        .collect();
    let outcomes = batch
        .items()
        .iter()
        .map(|item| {
            BatchOutcome::success(
                item.request_id(),
                item.phase(),
                usize::from(item.expects_output()),
                item.expects_output(),
            )
        })
        .collect::<Vec<_>>();
    s.commit_batch(batch.id(), &outcomes).unwrap();
    for capture in s.take_prefix_captures() {
        let exists = s.has_capture_anchor(&capture);
        assert!(s.begin_prefix_capture(&capture).unwrap().is_some() || exists);
        s.end_prefix_capture(&capture).unwrap();
    }
    s.audit().unwrap();
    items
}

/// An instance that keeps anchors on its grid prefills an unaligned prompt
/// past its rounded end in one chunk and captures both ends. An anchor on
/// a grid end, or past a later grid end, keeps its chunk ends.
#[test]
fn grid_anchor_items_keep_the_rounded_prompt_end_inside_one_chunk() {
    for (n, rows, captures) in [
        (789, vec![(0, 789)], vec![768, 789]),
        (
            8192 + 17,
            vec![(0, 8192), (8192, 17)],
            vec![8192, 8192 + 17],
        ),
        (
            10000,
            vec![(0, 8192), (8192, 1792), (9984, 16)],
            vec![8192, 9984, 10000],
        ),
        (50, vec![(0, 50)], vec![50]),
        (768, vec![(0, 768)], vec![768]),
    ] {
        let mut s = scheduler_with_grid_anchors(8192, true, 1, 0);
        submit(&mut s, 1, &(0..n as u32).collect::<Vec<_>>(), false);
        assert_eq!(finish(&mut s, 1), (rows, captures), "{n}");
    }
    let mut s = scheduler_with_grid_anchors(8192, true, 1, 0);
    submit(&mut s, 1, &(0..789).collect::<Vec<_>>(), false);
    assert_eq!(anchored_step(&mut s), [(1, 0, 789, Some(768))]);
}

/// Without an instance allowance, or without cuts, a guaranteed grid keeps
/// today's ends: the manifest's anchor items alone never span them.
#[test]
fn grid_anchor_items_need_an_instance_allowance_and_cuts() {
    for (cuts, grid, manifest, rows, captures) in [
        (true, 0, 0, vec![(0, 768), (768, 21)], vec![768, 789]),
        (true, 0, 2, vec![(0, 768), (768, 21)], vec![768, 789]),
        (false, 2, 0, vec![(0, 789)], vec![789]),
    ] {
        let mut s = scheduler_with_grid_anchors(8192, cuts, grid, manifest);
        submit(&mut s, 1, &(0..789).collect::<Vec<_>>(), false);
        assert_eq!(
            finish(&mut s, 1),
            (rows, captures),
            "cuts={cuts} grid={grid} manifest={manifest}"
        );
    }
}

/// The instance allowance replaces the manifest's anchor items and alone
/// marks the grid's anchors.
#[test]
fn grid_anchor_items_replace_the_manifest_value() {
    let config = SchedulerConfig::new(4, 64, 8192, 8192, 1)
        .unwrap()
        .with_anchor_items(2);
    assert!(!config.grid_anchors());
    for items in [0, 3] {
        let grid = config.with_grid_anchor_items(items);
        assert_eq!(grid.anchor_items(), items);
        assert_eq!(grid.grid_anchors(), items > 0);
    }
}

/// One anchor item a step: a second unaligned prompt in the same step ends
/// its chunk at its anchor and finishes in a later step.
#[test]
fn grid_anchor_items_bound_the_anchored_items_of_a_step() {
    let mut s = scheduler_with_grid_anchors(8192, true, 1, 0);
    submit(&mut s, 1, &(0..789).collect::<Vec<_>>(), false);
    submit(&mut s, 2, &(5000..5789).collect::<Vec<_>>(), false);
    assert_eq!(
        anchored_step(&mut s),
        [(1, 0, 789, Some(768)), (2, 0, 768, None)]
    );
    assert!(anchored_step(&mut s).contains(&(2, 768, 21, None)));
}

fn submit_outputs(s: &mut Scheduler, id: u64, tokens: &[u32], outputs: usize) {
    let spec = RequestSpec::new(RequestId::new(id), tokens.len(), outputs).unwrap();
    let chain = PrefixChainV1::new(64, Arc::from(tokens), Vec::new()).unwrap();
    s.submit_with_prefix(spec, Some(PrefixRequestV1::new(Arc::new(chain), true)))
        .unwrap();
}

/// Commits the next batch and returns its queued captures, not yet run.
fn commit_next(s: &mut Scheduler) -> Vec<runtime_scheduler::PrefixCaptureV1> {
    let batch = s.next_batch().unwrap().unwrap();
    let outcomes = batch
        .items()
        .iter()
        .map(|item| {
            BatchOutcome::success(
                item.request_id(),
                item.phase(),
                usize::from(item.expects_output()),
                item.expects_output(),
            )
        })
        .collect::<Vec<_>>();
    s.commit_batch(batch.id(), &outcomes).unwrap();
    s.take_prefix_captures()
}

/// A request that finishes in its anchored chunk captures the anchor, then
/// its prompt end, and keeps its sequence until the last capture ends.
#[test]
fn grid_anchor_items_release_a_finished_sequence_after_its_last_capture() {
    let mut s = scheduler_with_grid_anchors(8192, true, 1, 0);
    submit(&mut s, 1, &(0..789).collect::<Vec<_>>(), false);
    let captures = commit_next(&mut s);
    assert_eq!(
        captures.iter().map(|c| c.tokens()).collect::<Vec<_>>(),
        [768, 789]
    );
    assert_eq!(
        s.request_state(RequestId::new(1)).unwrap(),
        RequestState::Finished
    );
    for capture in &captures {
        assert_eq!(s.kv_stats().active_sequences(), 1);
        assert!(s.begin_prefix_capture(capture).unwrap().is_some());
        s.end_prefix_capture(capture).unwrap();
    }
    assert_eq!(s.kv_stats().active_sequences(), 0);
    s.audit().unwrap();
}

/// Cancelling a request after its anchored chunk commits, before its
/// captures run, drops both with its sequence: the same prompt then finds
/// no anchor and keeps its rounded end inside one chunk again.
#[test]
fn grid_anchor_items_drop_both_captures_of_a_request_cancelled_before_them() {
    let mut s = scheduler_with_grid_anchors(8192, true, 1, 0);
    let tokens: Vec<u32> = (0..789).collect();
    submit_outputs(&mut s, 1, &tokens, 4);
    let batch = s.next_batch().unwrap().unwrap();
    let outcomes = batch
        .items()
        .iter()
        .map(|item| BatchOutcome::success(item.request_id(), item.phase(), 1, false))
        .collect::<Vec<_>>();
    s.commit_batch(batch.id(), &outcomes).unwrap();
    s.cancel(RequestId::new(1)).unwrap();
    assert!(s.take_prefix_captures().is_empty());
    assert_eq!(s.kv_stats().active_sequences(), 0);
    submit(&mut s, 2, &tokens, false);
    assert_eq!(anchored_step(&mut s), [(2, 0, 789, Some(768))]);
    s.audit().unwrap();
}

/// A request restored at an odd prompt end keeps the next rounded end as
/// an ordinary cut, even when the instance has anchor spares.
#[test]
fn grid_anchor_items_keep_a_restored_odd_start_split() {
    let mut s = scheduler_with_grid_anchors(8192, true, 1, 0);
    let first: Vec<u32> = (0..789).collect();
    submit(&mut s, 1, &first, false);
    assert_eq!(finish(&mut s, 1), (vec![(0, 789)], vec![768, 789]));
    let mut second = first.clone();
    second.extend(2000..2300);
    submit(&mut s, 2, &second, false);
    s.admit().unwrap();
    assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 789);
    s.complete_prefix_restore(RequestId::new(2)).unwrap();
    assert_eq!(anchored_step(&mut s), [(2, 789, 299, None)]);
    assert_eq!(anchored_step(&mut s), [(2, 1088, 1, None)]);
    assert!(s.next_batch().unwrap().is_none());
}

/// Serves a prompt of `document` to its end as the engine does: its prefill,
/// then decodes, which publish no anchor (decode-written state is private to
/// the request). Returns the cached tokens and the prefill items as (start,
/// tokens).
fn serve(
    s: &mut Scheduler,
    id: u64,
    document: &[u32],
    prompt: usize,
    outputs: usize,
) -> (usize, Vec<(usize, usize)>) {
    let request = RequestId::new(id);
    submit_outputs(s, id, &document[..prompt], outputs);
    s.admit().unwrap();
    let cached = s.cached_tokens(request).unwrap();
    if cached > 0 {
        s.complete_prefix_restore(request).unwrap();
    }
    let (mut emitted, mut rows) = (0, Vec::new());
    while s.request_state(request).unwrap() != RequestState::Finished {
        let batch = s.next_batch().unwrap().unwrap();
        let [item] = batch.items() else {
            panic!("one request")
        };
        let emits = usize::from(item.expects_output());
        if let BatchPhase::Prefill { tokens } = item.phase() {
            rows.push((batch.kv_plan().appends()[0].previous_tokens(), tokens));
        }
        let outcome =
            BatchOutcome::success(request, item.phase(), emits, emitted + emits == outputs);
        s.commit_batch(batch.id(), &[outcome]).unwrap();
        emitted += emits;
        for capture in s.take_prefix_captures() {
            let exists = s.has_capture_anchor(&capture);
            assert!(s.begin_prefix_capture(&capture).unwrap().is_some() || exists);
            s.end_prefix_capture(&capture).unwrap();
        }
        s.audit().unwrap();
    }
    (cached, rows)
}

/// Every cache hit of the served exactness plan is the split grid's. Only
/// prefill-written anchors are public: a 16-token probe at 769 leaves its
/// prompt end, the longest hit of a later repeat of 789. Repeats before
/// probes and single-token seeds keep the rounded ends and the seeds' odd
/// ends the longest hits.
#[test]
fn grid_anchor_items_keep_every_cache_hit_of_the_split_grid() {
    for grid in [0, 1] {
        let split = |cached: usize, end: usize| {
            let anchor = end / 64 * 64;
            if (grid == 0 || cached > 0) && anchor > cached {
                vec![(cached, anchor - cached), (anchor, end - anchor)]
            } else {
                vec![(cached, end - cached)]
            }
        };
        let mut s = scheduler_with_grid_anchors(8192, true, grid, 0);
        let first: Vec<u32> = (0..1000).collect();
        assert_eq!(serve(&mut s, 1, &first, 789, 16), (0, split(0, 789)));
        assert_eq!(serve(&mut s, 2, &first, 769, 16), (768, split(768, 769)));
        assert_eq!(serve(&mut s, 3, &first, 789, 16), (769, split(769, 789)));
        let second: Vec<u32> = (1000..2000).collect();
        assert_eq!(serve(&mut s, 4, &second, 789, 16), (0, split(0, 789)));
        assert_eq!(serve(&mut s, 5, &second, 789, 16), (768, split(768, 789)));
        assert_eq!(serve(&mut s, 6, &second, 769, 16), (768, split(768, 769)));
        for (id, seed) in [(7, 760), (10, 761), (13, 762)] {
            let document: Vec<u32> = (id as u32 * 1000..id as u32 * 1000 + 1000).collect();
            assert_eq!(serve(&mut s, id, &document, seed, 1), (0, split(0, seed)));
            assert_eq!(
                serve(&mut s, id + 1, &document, 900, 16),
                (seed, split(seed, 900))
            );
            assert_eq!(
                serve(&mut s, id + 2, &document, 900, 16),
                (896, split(896, 900))
            );
        }
    }
}

#[test]
fn grid_anchor_items_and_aligned_prompt_cuts_keep_separate_boundaries() {
    for aligned in [false, true] {
        for anchors in [0, 1] {
            let config = SchedulerConfig::new(4, 64, 8192, 8192, 1)
                .unwrap()
                .with_prefill_granule(64)
                .unwrap()
                .with_prefix_cache(1)
                .with_prefill_cuts(true)
                .with_guaranteed_prefill_chunks(8192)
                .unwrap()
                .with_grid_anchor_items(anchors)
                .with_aligned_prompt_capture(aligned);
            let mut s = Scheduler::new(config, KvGeometry::new(64, 8192).unwrap()).unwrap();
            submit(&mut s, 1, &(0..128).collect::<Vec<_>>(), false);
            submit(&mut s, 2, &(1000..1789).collect::<Vec<_>>(), false);
            let first = anchored_step(&mut s);
            assert_eq!(
                first,
                [
                    (1, 0, if aligned { 64 } else { 128 }, None),
                    if anchors == 0 {
                        (2, 0, 768, None)
                    } else {
                        (2, 0, 789, Some(768))
                    },
                ],
                "aligned={aligned}, anchors={anchors}"
            );
            let mut tail = Vec::new();
            if aligned {
                tail.push((1, 64, 64, None));
            }
            if anchors == 0 {
                tail.push((2, 768, 21, None));
            }
            if !tail.is_empty() {
                let mut actual = anchored_step(&mut s);
                actual.sort_unstable();
                assert_eq!(actual, tail);
            }
            assert!(s.next_batch().unwrap().is_none());
            s.audit().unwrap();
        }
    }
}

#[test]
fn grid_anchor_items_keep_a_continuation_after_768_split() {
    let mut s = scheduler_with_grid_anchors(8192, true, 8, 0);
    let prime: Vec<u32> = (0..839).collect();
    submit(&mut s, 1, &prime, false);
    assert_eq!(finish(&mut s, 1), (vec![(0, 839)], vec![832, 839]));
    let mut prompt = prime[..768].to_vec();
    prompt.extend(2000..2092);
    submit(&mut s, 2, &prompt, false);
    assert_eq!(anchored_step(&mut s), [(2, 0, 768, None)]);
    assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 0);
    assert_eq!(anchored_step(&mut s), [(2, 768, 64, None)]);
    assert_eq!(anchored_step(&mut s), [(2, 832, 28, None)]);
    assert!(s.next_batch().unwrap().is_none());
    s.audit().unwrap();
}

#[test]
fn grid_anchor_items_preserve_cold_single_and_packed_shapes() {
    // Cold prompt lengths and packed shapes. Disjoint prefixes prevent an
    // earlier shared-prefix boundary from changing these whole-prompt items.
    let shapes: &[&[usize]] = &[
        &[3],
        &[6],
        &[12],
        &[24],
        &[49],
        &[49, 51, 357, 714, 670, 730, 680],
        &[49, 51, 467, 357, 714, 730, 680],
        &[49, 357, 714],
        &[49, 490, 463, 805, 922, 435, 1903],
        &[49, 805, 490],
        &[49, 902, 805],
        &[49, 922, 463, 435, 1903, 805, 490],
        &[51],
        &[51, 53, 789, 602, 871, 863, 835],
        &[51, 53, 871, 835, 602, 863, 789],
        &[51, 490],
        &[51, 789, 602],
        &[51, 871, 789],
        &[53],
        &[53, 478],
        &[53, 478, 835],
        &[53, 835, 478],
        &[53, 835, 863],
        &[53, 863, 835],
        &[54],
        &[54, 643, 938],
        &[54, 865, 938],
        &[54, 938, 865],
        &[357],
        &[357, 714],
        &[357, 714, 670],
        &[434],
        &[434, 54, 788, 835, 458, 938, 865],
        &[434, 788, 458, 865, 835, 54, 938],
        &[434, 835, 458],
        &[435],
        &[435, 1903, 463],
        &[458],
        &[458, 434, 54, 835, 938, 865, 643],
        &[458, 835, 434, 938, 54, 643, 865],
        &[463],
        &[463, 435, 49, 902, 490, 1903, 805],
        &[463, 435, 922],
        &[467],
        &[478],
        &[478, 863],
        &[490],
        &[490, 435, 463, 805, 902, 49, 1903],
        &[602],
        &[602, 53, 863, 789, 835, 871, 478],
        &[602, 789, 871],
        &[643],
        &[670],
        &[680],
        &[680, 51, 730],
        &[680, 467, 730],
        &[714],
        &[714, 51, 357, 49, 730, 680, 670],
        &[730],
        &[730, 51, 357, 49, 714, 680, 467],
        &[788],
        &[788, 458, 434],
        &[789],
        &[789, 51, 53, 835, 602, 863, 871],
        &[789, 53, 478, 871, 602, 835, 863],
        &[789, 602, 51, 871],
        &[789, 602, 51, 871, 835, 478, 863, 53],
        &[805],
        &[835],
        &[835, 53, 478, 863, 871, 602, 789],
        &[835, 434, 54, 938, 865, 458, 643],
        &[835, 434, 458],
        &[863],
        &[865],
        &[871],
        &[871, 51, 789],
        &[871, 789, 602],
        &[902],
        &[922],
        &[938],
        &[1903],
    ];
    for &shape in shapes {
        let config = SchedulerConfig::new(8, 64, 8192, 8192, 1)
            .unwrap()
            .with_prefill_granule(64)
            .unwrap()
            .with_prefix_cache(1)
            .with_prefill_cuts(true)
            .with_guaranteed_prefill_chunks(8192)
            .unwrap()
            .with_grid_anchor_items(8);
        let mut s = Scheduler::new(config, KvGeometry::new(64, 8192).unwrap()).unwrap();
        let mut expected = Vec::new();
        for (i, &tokens) in shape.iter().enumerate() {
            let id = i as u64 + 1;
            let base = i as u32 * 10000;
            submit(
                &mut s,
                id,
                &(base..base + tokens as u32).collect::<Vec<_>>(),
                false,
            );
            let anchor = tokens / 64 * 64;
            expected.push((
                id,
                0,
                tokens,
                (anchor > 0 && anchor < tokens).then_some(anchor),
            ));
        }
        let mut actual = anchored_step(&mut s);
        actual.sort_unstable();
        assert_eq!(actual, expected, "{shape:?}");
        assert!(s.next_batch().unwrap().is_none());
        s.audit().unwrap();
    }
}
