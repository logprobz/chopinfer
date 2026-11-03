use super::*;
use std::sync::Arc;

fn chain(tokens: &[u32], salt: Option<&[u8]>) -> PrefixChainV1 {
    PrefixChainV1::salted(4, Arc::from(tokens), vec![], salt).unwrap()
}
fn count(metrics: &KvMetrics, counter: Count) -> u64 {
    metrics.snapshot().reply_history.counters[counter as usize]
}
fn rows(metrics: &KvMetrics, kind: Rows) -> u64 {
    metrics.snapshot().reply_history.rows[kind as usize]
        .iter()
        .sum()
}
fn complete(registry: &mut Registry, id: u64, prompt: &[u32], reply: &[u32], salt: Option<&[u8]>) {
    let id = RequestId::new(id);
    registry.submit(id, &chain(prompt, salt), reply.len());
    registry.append(id, reply);
    registry.engine_finish(id, FinishReason::Stop);
}
fn consumer(registry: &mut Registry, id: u64, tokens: &[u32], salt: Option<&[u8]>, hit: usize) {
    let id = RequestId::new(id);
    registry.submit(id, &chain(tokens, salt), 2);
    registry.prefill(id, hit, tokens.len() - 1, hit);
}

#[test]
fn exact_reply_and_final_token_difference_have_the_same_written_boundary() {
    for final_token in [15, 99] {
        let metrics = KvMetrics::default();
        let mut registry = Registry::new(metrics.clone()).unwrap();
        complete(
            &mut registry,
            1,
            &[1, 2, 3],
            &(4..16).collect::<Vec<_>>(),
            Some(b"scope"),
        );
        let mut tokens = (1..15).collect::<Vec<_>>();
        tokens.extend([final_token, 22]);
        consumer(&mut registry, 2, &tokens, Some(b"scope"), 0);
        assert_eq!(rows(&metrics, Rows::Repeated), 8);
        assert_eq!(rows(&metrics, Rows::Candidate), 11);
        assert_eq!(rows(&metrics, Rows::Eligible), 11);
        assert_eq!(rows(&metrics, Rows::Rounding), 3);
        assert_eq!(count(&metrics, Count::EligibleTurn), 1);
        assert_eq!(
            count(
                &metrics,
                if final_token == 15 {
                    Count::CompleteReply
                } else {
                    Count::FinalToken
                }
            ),
            1
        );
    }
}

#[test]
fn near_end_mismatch_repeats_pages_but_has_no_reply_end_opportunity() {
    let metrics = KvMetrics::default();
    let mut registry = Registry::new(metrics.clone()).unwrap();
    let prompt = [1, 2, 3, 4];
    complete(
        &mut registry,
        1,
        &prompt,
        &(5..30).collect::<Vec<_>>(),
        None,
    );
    let mut tokens = (1..31).collect::<Vec<_>>();
    tokens[26] = 99;
    consumer(&mut registry, 2, &tokens, None, 4);
    assert_eq!(rows(&metrics, Rows::Repeated), 20);
    assert_eq!(rows(&metrics, Rows::Eligible), 0);
    assert_eq!(count(&metrics, Count::InsideReply), 1);
}

#[test]
fn exact_short_reply_has_rows_without_a_generated_page_and_boundary_leaves_a_token() {
    for length in [6, 7, 8] {
        let metrics = KvMetrics::default();
        let mut registry = Registry::new(metrics.clone()).unwrap();
        complete(&mut registry, 1, &[1, 2, 3, 4, 5], &[6, 7], None);
        consumer(&mut registry, 2, &(1..=length).collect::<Vec<_>>(), None, 0);
        assert_eq!(rows(&metrics, Rows::Repeated), 0);
        assert_eq!(rows(&metrics, Rows::Eligible), u64::from(length >= 7));
        assert_eq!(count(&metrics, Count::Shorter), u64::from(length == 6));
    }
}

#[test]
fn salts_media_and_before_reply_changes_do_not_claim_continuation() {
    for salt in [None, Some(b"different".as_slice())] {
        let metrics = KvMetrics::default();
        let mut registry = Registry::new(metrics.clone()).unwrap();
        complete(
            &mut registry,
            1,
            &[1, 2, 3, 4, 5],
            &[6, 7, 8, 9, 10],
            Some(b"scope"),
        );
        consumer(&mut registry, 2, &(1..12).collect::<Vec<_>>(), salt, 0);
        assert_eq!(rows(&metrics, Rows::Candidate), 0);
        assert_eq!(count(&metrics, Count::NoHistory), 2);
    }
    let metrics = KvMetrics::default();
    let mut registry = Registry::new(metrics.clone()).unwrap();
    complete(&mut registry, 1, &[1, 2, 3, 4, 5], &[6, 7, 8, 9, 10], None);
    consumer(
        &mut registry,
        2,
        &[1, 2, 3, 4, 99, 6, 7, 8, 9, 10, 11],
        None,
        0,
    );
    assert_eq!(count(&metrics, Count::BeforeReply), 1);
    assert_eq!(rows(&metrics, Rows::Eligible), 0);
    let media = runtime_kv::PrefixMediaSpanV1 {
        identity: [3; 32],
        start: 1,
        rows: 1,
        media_row: 0,
    };
    let media_chain =
        PrefixChainV1::new(4, Arc::from((1..12).collect::<Vec<_>>()), vec![media]).unwrap();
    registry.submit(RequestId::new(3), &media_chain, 2);
    registry.prefill(RequestId::new(3), 0, 10, 0);
    assert_eq!(rows(&metrics, Rows::Eligible), 0);
}

#[test]
fn duplicates_union_pages_and_only_deepest_anchor_counts() {
    let metrics = KvMetrics::default();
    let mut registry = Registry::new(metrics.clone()).unwrap();
    for id in 1..4 {
        complete(
            &mut registry,
            id,
            &[1, 2, 3, 4],
            &(5..17).collect::<Vec<_>>(),
            None,
        );
    }
    complete(
        &mut registry,
        4,
        &(1..13).collect::<Vec<_>>(),
        &(13..25).collect::<Vec<_>>(),
        None,
    );
    consumer(&mut registry, 5, &(1..30).collect::<Vec<_>>(), None, 0);
    assert_eq!(rows(&metrics, Rows::Repeated), 16);
    assert_eq!(rows(&metrics, Rows::Eligible), 11);
    assert_eq!(rows(&metrics, Rows::Candidate), 19);
    assert_eq!(count(&metrics, Count::EligibleTurn), 1);
    assert_eq!(count(&metrics, Count::Ambiguous), 0);
}

#[test]
fn forks_and_mixed_annotations_are_reported_without_rendering_claims() {
    let metrics = KvMetrics::default();
    let mut registry = Registry::new(metrics.clone()).unwrap();
    let id = RequestId::new(1);
    registry.submit(id, &chain(&[1, 2, 3, 4], None), 8);
    registry.defer(id);
    registry.append(id, &[5, 6, 7, 8, 9, 10, 11, 12]);
    registry.annotate(id, 6, ReplyRegion::ToolCalls);
    registry.annotate(id, 6, ReplyRegion::Content);
    registry.engine_finish(id, FinishReason::Stop);
    registry.finish(id, FinishReason::Stop);
    complete(
        &mut registry,
        2,
        &[1, 2, 3, 4],
        &[5, 6, 7, 8, 9, 99, 11, 12],
        None,
    );
    consumer(
        &mut registry,
        3,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 12, 13],
        None,
        0,
    );
    assert_eq!(count(&metrics, Count::Ambiguous), 1);
    assert_eq!(count(&metrics, Count::RegionMixed), 1);
    assert_eq!(rows(&metrics, Rows::Eligible), 0);
}

#[test]
fn frontend_completion_and_submission_order_control_visibility() {
    let metrics = KvMetrics::default();
    let mut registry = Registry::new(metrics.clone()).unwrap();
    let id = RequestId::new(1);
    registry.submit(id, &chain(&[1, 2, 3, 4], None), 4);
    registry.defer(id);
    registry.append(id, &[5, 6, 7, 8]);
    registry.engine_finish(id, FinishReason::Length);
    registry.retire(id);
    registry.submit(
        RequestId::new(2),
        &chain(&(1..10).collect::<Vec<_>>(), None),
        2,
    );
    registry.finish(id, FinishReason::Stop);
    registry.prefill(RequestId::new(2), 0, 8, 0);
    assert_eq!(rows(&metrics, Rows::Eligible), 0);
    consumer(&mut registry, 3, &(1..10).collect::<Vec<_>>(), None, 0);
    assert_eq!(rows(&metrics, Rows::Eligible), 3);
    registry.finish(id, FinishReason::Cancelled);
    assert_eq!(
        registry.completed.len(),
        1,
        "late finalization cannot change the first finish"
    );
}

#[test]
fn cancellations_errors_and_late_annotations_do_not_recreate_history() {
    for reason in [Count::Cancelled, Count::DeliveryFailed] {
        let metrics = KvMetrics::default();
        let mut registry = Registry::new(metrics.clone()).unwrap();
        let id = RequestId::new(1);
        registry.submit(id, &chain(&[1, 2, 3, 4], None), 4);
        registry.append(id, &[5, 6]);
        registry.discard(id, reason);
        registry.annotate(id, 0, ReplyRegion::Content);
        registry.finish(id, FinishReason::Stop);
        assert!(registry.active.is_empty() && registry.completed.is_empty());
        assert_eq!(count(&metrics, Count::LateAnnotation), 1);
    }
}

#[test]
fn caps_include_reserved_reply_capacity_and_many_scopes() {
    let metrics = KvMetrics::default();
    let mut registry = Registry::new(metrics.clone()).unwrap();
    for id in 1..300 {
        complete(
            &mut registry,
            id,
            &[1, 2, 3, 4],
            &[5, 6],
            Some(&id.to_le_bytes()),
        );
        assert!(registry.bytes() <= MAX_BYTES);
    }
    assert_eq!(registry.completed.len(), MAX_COMPLETED);
    registry.submit(RequestId::new(301), &chain(&[1, 2, 3, 4], None), MAX_BYTES);
    assert_eq!(count(&metrics, Count::CapacityDrop), 1);
    for id in 302..330 {
        registry.submit(RequestId::new(id), &chain(&[1, 2, 3, 4], None), 100_000);
        assert!(registry.bytes() <= MAX_BYTES);
    }
    assert!(count(&metrics, Count::CapacityDrop) > 1);
    assert!(count(&metrics, Count::Evicted) > 0);
}

#[test]
fn page_rounding_respects_partial_hits_and_splits_context_buckets() {
    let metrics = KvMetrics::default();
    let mut registry = Registry::new(metrics.clone()).unwrap();
    complete(
        &mut registry,
        1,
        &[1, 2, 3],
        &(4..20).collect::<Vec<_>>(),
        None,
    );
    consumer(&mut registry, 2, &(1..22).collect::<Vec<_>>(), None, 5);
    assert_eq!(rows(&metrics, Rows::Repeated), 8);
    assert_eq!(rows(&metrics, Rows::Eligible), 13);
    assert_eq!(rows(&metrics, Rows::Rounding), 5);
    metrics.reply_history_rows(Rows::Candidate, 32_766, 32_770);
    let buckets = metrics.snapshot().reply_history.rows[Rows::Candidate as usize];
    assert_eq!(buckets[1], 2);
}

#[test]
fn frontend_stop_truncates_an_accepted_run_before_relay_handoff() {
    let metrics = KvMetrics::default();
    let mut registry = Registry::new(metrics.clone()).unwrap();
    let id = RequestId::new(1);
    registry.submit(id, &chain(&[1, 2, 3], None), 10);
    registry.defer(id);
    registry.require_handoff(id);
    registry.append(id, &[4, 5, 6, 7, 8, 9]);
    registry.engine_finish(id, FinishReason::Length);
    registry.delivered(id, FinishReason::Stop, 3);
    assert!(registry.completed.is_empty());
    registry.handoff(id);
    assert_eq!(registry.completed[0].end(), 6);
    assert_eq!(
        registry.completed[0].written_key,
        chain(&[1, 2, 3, 4, 5], None).key_at(5).unwrap()
    );
    consumer(&mut registry, 2, &[1, 2, 3, 4, 5, 99, 11], None, 0);
    assert_eq!(rows(&metrics, Rows::Eligible), 2);
    assert_eq!(count(&metrics, Count::FrontendStop), 1);
}

#[test]
fn a_relay_dropped_after_frontend_finish_does_not_retain_history() {
    let metrics = KvMetrics::default();
    let observer = ReplyHistoryObserver(Arc::new(Mutex::new(
        Registry::new(metrics.clone()).unwrap(),
    )));
    let id = RequestId::new(1);
    {
        let mut registry = observer.0.lock().unwrap();
        registry.submit(id, &chain(&[1, 2, 3, 4], None), 2);
        registry.defer(id);
        registry.require_handoff(id);
        registry.append(id, &[5, 6]);
        registry.engine_finish(id, FinishReason::Length);
        registry.delivered(id, FinishReason::Length, 2);
    }
    drop(observer.delivery(id));
    let registry = observer.0.lock().unwrap();
    assert!(registry.active.is_empty() && registry.completed.is_empty());
    assert_eq!(count(&metrics, Count::DeliveryFailed), 1);
}

#[test]
fn lookup_probe_work_depends_on_pages_logarithmically_and_reply_hashes_once() {
    let metrics = KvMetrics::default();
    let mut registry = Registry::new(metrics.clone()).unwrap();
    let prompt = vec![1; 160_000];
    let reply = vec![2; 550];
    for id in 1..=256 {
        complete(&mut registry, id, &prompt, &reply, None);
    }
    let mut tokens = prompt;
    tokens.extend(reply);
    tokens.push(3);
    let before_hash = count(&metrics, Count::Fingerprints);
    let before_probe = count(&metrics, Count::Probes);
    consumer(&mut registry, 1000, &tokens, None, 0);
    assert!(count(&metrics, Count::Fingerprints) - before_hash <= 256 * 4 + 550);
    assert!(count(&metrics, Count::Probes) - before_probe <= 256 * 20);
}

#[test]
fn delivery_receipts_have_a_shared_cap_even_before_admission() {
    let metrics = KvMetrics::default();
    let observer = ReplyHistoryObserver(Arc::new(Mutex::new(Registry::new(metrics).unwrap())));
    let receipts = (0..MAX_DELIVERIES)
        .map(|id| observer.delivery(RequestId::new(id as u64)).unwrap())
        .collect::<Vec<_>>();
    assert!(observer.delivery(RequestId::new(999)).is_none());
    assert_eq!(observer.0.lock().unwrap().delivery_handles, MAX_DELIVERIES);
    drop(receipts);
    assert_eq!(observer.0.lock().unwrap().delivery_handles, 0);
    assert!(observer.delivery(RequestId::new(999)).is_some());
}
