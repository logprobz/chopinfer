use std::sync::Arc;

use runtime_kv::{KvGeometry, PrefixChainV1};
use runtime_scheduler::{
    AdmissionPolicy, Batch, BatchOutcome, BatchPhase, MixedPolicy, OptimisticAdmission,
    PrefillBudget, PrefillTarget, PrefixRequestV1, RequestId, RequestSpec, Scheduler,
    SchedulerConfig,
};

fn scheduler(policy: PrefillBudget, budget: usize) -> Scheduler {
    Scheduler::new(
        SchedulerConfig::new(8, 16, budget, 2048, 1)
            .unwrap()
            .with_lone_prefill_expansion(true)
            .with_prefill_granule(64)
            .unwrap()
            .with_prefill_budget(policy),
        KvGeometry::new(16, 16384).unwrap(),
    )
    .unwrap()
}

fn submit(s: &mut Scheduler, id: u64, prompt: usize, output: usize) {
    s.submit(RequestSpec::new(RequestId::new(id), prompt, output).unwrap())
        .unwrap();
}

fn commit(s: &mut Scheduler, batch: Batch) {
    let outcomes: Vec<_> = batch
        .items()
        .iter()
        .map(|item| {
            BatchOutcome::success(
                item.request_id(),
                item.phase(),
                usize::from(item.expects_output()),
                false,
            )
        })
        .collect();
    s.commit_batch(batch.id(), &outcomes).unwrap();
    s.audit().unwrap();
}

#[test]
fn lone_prompt_uses_step_budget_and_aligns_nonfinal_chunks() {
    for policy in [
        PrefillBudget::Fixed,
        PrefillBudget::DecodeAware(PrefillTarget::default()),
        PrefillBudget::Tbt(MixedPolicy::default()),
    ] {
        for budget in [8192, 8223] {
            let mut s = scheduler(policy, budget);
            submit(&mut s, 1, 20001, 1);
            for tokens in [8192, 8192, 3617] {
                let batch = s.next_batch().unwrap().unwrap();
                assert_eq!(batch.items().len(), 1);
                assert_eq!(batch.items()[0].phase(), BatchPhase::Prefill { tokens });
                commit(&mut s, batch);
            }
            assert!(s.next_batch().unwrap().is_none());
        }
    }
}

#[test]
fn competing_prefills_keep_configured_chunks() {
    let mut s = scheduler(PrefillBudget::Fixed, 8192);
    submit(&mut s, 1, 20000, 1);
    submit(&mut s, 2, 20000, 1);
    let batch = s.next_batch().unwrap().unwrap();
    assert_eq!(batch.items().len(), 2);
    for item in batch.items() {
        assert_eq!(item.phase(), BatchPhase::Prefill { tokens: 2048 });
    }
    commit(&mut s, batch);
}

#[test]
fn a_decode_neighbor_keeps_its_mixed_step_budget() {
    for (policy, tokens) in [
        (PrefillBudget::Fixed, 2048),
        (PrefillBudget::DecodeAware(PrefillTarget::default()), 512),
        (PrefillBudget::Tbt(MixedPolicy::default()), 512),
    ] {
        let mut s = scheduler(policy, 8192);
        submit(&mut s, 1, 1, 64);
        let batch = s.next_batch().unwrap().unwrap();
        commit(&mut s, batch);
        submit(&mut s, 2, 20000, 1);
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.shape().decode_items(), 1);
        assert_eq!(batch.shape().prefill_tokens(), tokens);
        commit(&mut s, batch);
    }
}

#[test]
fn scored_rows_still_bound_a_lone_prompt() {
    for rows in [1024, 8192] {
        let config = SchedulerConfig::new(8, 16, 8192, 2048, 1)
            .unwrap()
            .with_lone_prefill_expansion(true)
            .with_maximum_scored_rows(rows)
            .unwrap();
        let mut s = Scheduler::new(config, KvGeometry::new(16, 16384).unwrap()).unwrap();
        s.submit(
            RequestSpec::new(RequestId::new(1), 20000, 1)
                .unwrap()
                .with_prompt_scoring(),
        )
        .unwrap();
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.shape().prefill_tokens(), rows.min(2048));
        commit(&mut s, batch);
    }
}

#[test]
fn a_lone_recompute_uses_the_budget_without_repeating_output() {
    let config = SchedulerConfig::new(8, 16, 8192, 2048, 1)
        .unwrap()
        .with_lone_prefill_expansion(true)
        .with_admission(AdmissionPolicy::Optimistic(
            OptimisticAdmission::new(0, 1)
                .unwrap()
                .with_forced_preemption(2)
                .unwrap(),
        ));
    let mut s = Scheduler::new(config, KvGeometry::new(16, 16384).unwrap()).unwrap();
    submit(&mut s, 1, 10000, 5);
    let mut emitted = 0;
    let mut replayed = false;
    for _ in 0..20 {
        s.preempt_for_next_step(0, |_, _| None).unwrap();
        let Some(batch) = s.next_batch().unwrap() else {
            break;
        };
        if s.preemptions(RequestId::new(1)).unwrap() > 0
            && batch.items()[0].phase() == (BatchPhase::Prefill { tokens: 8192 })
        {
            replayed = true;
            assert!(!batch.items()[0].expects_output());
        }
        emitted += usize::from(batch.items()[0].expects_output());
        commit(&mut s, batch);
    }
    assert!(replayed);
    assert_eq!(emitted, 5);
    assert_eq!(s.preemptions(RequestId::new(1)).unwrap(), 1);
    assert!(s.next_batch().unwrap().is_none());
}

#[test]
fn a_pending_restore_neighbor_keeps_the_configured_chunk() {
    let config = SchedulerConfig::new(8, 16, 8192, 2048, 1)
        .unwrap()
        .with_lone_prefill_expansion(true)
        .with_prefill_granule(64)
        .unwrap()
        .with_prefix_cache(1);
    let mut s = Scheduler::new(config, KvGeometry::new(64, 16384).unwrap()).unwrap();
    let mut tokens: Vec<u32> = (0..1024).collect();
    let chain = PrefixChainV1::new(64, Arc::from(tokens.clone()), Vec::new()).unwrap();
    s.submit_with_prefix(
        RequestSpec::new(RequestId::new(1), tokens.len(), 1).unwrap(),
        Some(PrefixRequestV1::new(Arc::new(chain), true)),
    )
    .unwrap();
    let batch = s.next_batch().unwrap().unwrap();
    commit(&mut s, batch);
    for capture in s.take_prefix_captures() {
        assert!(s.begin_prefix_capture(&capture).unwrap().is_some());
        s.end_prefix_capture(&capture).unwrap();
    }
    tokens.extend([9999; 64]);
    let chain = PrefixChainV1::new(64, Arc::from(tokens.clone()), Vec::new()).unwrap();
    s.submit_with_prefix(
        RequestSpec::new(RequestId::new(2), tokens.len(), 1).unwrap(),
        Some(PrefixRequestV1::new(Arc::new(chain), true)),
    )
    .unwrap();
    submit(&mut s, 3, 20000, 1);
    s.admit().unwrap();
    let restores = s.pending_prefix_restores();
    assert_eq!(restores.len(), 1);
    assert_eq!(restores[0].request_id(), RequestId::new(2));
    assert_eq!(restores[0].tokens(), 1024);
    let batch = s.next_batch().unwrap().unwrap();
    assert_eq!(batch.items().len(), 1);
    assert_eq!(batch.items()[0].request_id(), RequestId::new(3));
    assert_eq!(
        batch.items()[0].phase(),
        BatchPhase::Prefill { tokens: 2048 }
    );
    commit(&mut s, batch);
    s.complete_prefix_restore(RequestId::new(2)).unwrap();
    let batch = s.next_batch().unwrap().unwrap();
    assert_eq!(batch.items().len(), 2);
    commit(&mut s, batch);
    let batch = s.next_batch().unwrap().unwrap();
    assert_eq!(
        batch.items()[0].phase(),
        BatchPhase::Prefill { tokens: 8192 }
    );
    commit(&mut s, batch);
}

#[test]
fn undeclared_prefill_capacity_keeps_the_configured_chunk() {
    let config = SchedulerConfig::new(1, 2, 128, 32, 1).unwrap();
    let mut s = Scheduler::new(config, KvGeometry::new(16, 16).unwrap()).unwrap();
    submit(&mut s, 1, 64, 1);
    for _ in 0..2 {
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.items()[0].phase(), BatchPhase::Prefill { tokens: 32 });
        commit(&mut s, batch);
    }
    assert!(s.next_batch().unwrap().is_none());
}
