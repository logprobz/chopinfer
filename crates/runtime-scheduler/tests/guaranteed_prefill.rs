use std::sync::Arc;

use runtime_kv::{KvGeometry, PrefixChainV1};
use runtime_scheduler::{
    BatchOutcome, BatchPhase, PrefixRequestV1, RequestId, RequestSpec, RequestState, Scheduler,
    SchedulerConfig,
};

fn scheduler(chunk: usize, guaranteed: bool) -> Scheduler {
    let mut config = SchedulerConfig::new(4, 64, 8192, chunk, 1)
        .unwrap()
        .with_prefill_granule(64)
        .unwrap()
        .with_maximum_scored_rows(64)
        .unwrap()
        .with_prefix_cache(1);
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
fn terminal_decode_publishes_complete_response_state_for_the_next_turn() {
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
    let chain = PrefixChainV1::new(64, Arc::from(retained.clone()), Vec::new()).unwrap();
    s.prepare_decode_capture(id, PrefixRequestV1::new(Arc::new(chain), true))
        .unwrap();
    s.commit_batch(
        batch.id(),
        &[BatchOutcome::success(id, BatchPhase::Decode, 1, true)],
    )
    .unwrap();
    let captures = s.take_prefix_captures();
    assert_eq!(captures.len(), 1);
    assert_eq!(captures[0].tokens(), 66);
    for capture in captures {
        assert!(s.begin_prefix_capture(&capture).unwrap().is_some());
        s.end_prefix_capture(&capture).unwrap();
    }
    // The last sampled token has no KV. The following turn recomputes it.
    retained.push(9);
    submit(&mut s, 2, &retained, false);
    s.admit().unwrap();
    assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 66);
    s.audit().unwrap();
}
