use runtime_kv::KvGeometry;
use runtime_scheduler::{
    AdmissionPolicy, Batch, BatchOutcome, BatchPhase, OptimisticAdmission, PrefillBudget,
    PrefillTarget, RequestId, RequestSpec, Scheduler, SchedulerConfig,
};

/// The first decode-aware policy: prefill costs at most one decode step.
const ORIGINAL: PrefillBudget = PrefillBudget::DecodeAware(PrefillTarget::DecodeSteps(1));

fn scheduler(policy: PrefillBudget, admission: AdmissionPolicy) -> Scheduler {
    Scheduler::new(
        SchedulerConfig::new(64, 128, 8192, 2048, 1)
            .unwrap()
            .with_prefill_budget(policy)
            .with_admission(admission),
        KvGeometry::new(16, 65536).unwrap(),
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
fn counts(batch: &Batch) -> (usize, usize) {
    batch
        .items()
        .iter()
        .fold((0, 0), |(p, d), item| match item.phase() {
            BatchPhase::Prefill { tokens } => (p + tokens, d),
            BatchPhase::Decode => (p, d + 1),
            _ => panic!("unexpected verify"),
        })
}

#[test]
fn fixed_is_unchanged_and_aware_adapts_without_starving_prefill() {
    for policy in [PrefillBudget::Fixed, ORIGINAL] {
        let mut s = scheduler(policy, AdmissionPolicy::Reserved);
        submit(&mut s, 1, 1, 64);
        let batch = s.next_batch().unwrap().unwrap();
        commit(&mut s, batch);
        for id in 2..=9 {
            submit(&mut s, id, 16384, 8);
        }
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(
            counts(&batch),
            (
                if policy == PrefillBudget::Fixed {
                    6144
                } else {
                    512
                },
                1
            )
        );
        commit(&mut s, batch);
        s.observe_step_time(2048, 0, 2048);
        s.observe_step_time(0, 1, 4096);
        s.observe_step_time(0, 1, 4096);
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(
            counts(&batch),
            (
                if policy == PrefillBudget::Fixed {
                    6144
                } else {
                    4096
                },
                1
            )
        );
        commit(&mut s, batch);
        // A much slower prefill lowers the budget, but every waiting prefill
        // request gets a full 512-token chunk in least-recently-run order.
        s.observe_step_time(2048, 0, 1_000_000);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..8 {
            let batch = s.next_batch().unwrap().unwrap();
            let (prefill, decode) = counts(&batch);
            assert!(prefill + decode <= 8192);
            assert_eq!(decode, 1);
            if policy == ORIGINAL {
                assert_eq!(prefill, 512);
            }
            for item in batch.items() {
                if matches!(item.phase(), BatchPhase::Prefill { .. }) {
                    seen.insert(item.request_id());
                }
            }
            commit(&mut s, batch);
        }
        assert_eq!(seen.len(), 8);
    }
}

#[test]
fn optimistic_recompute_uses_the_same_budget_and_releases_pages() {
    let admission = AdmissionPolicy::Optimistic(
        OptimisticAdmission::new(0, 1)
            .unwrap()
            .with_forced_preemption(2)
            .unwrap(),
    );
    let mut s = scheduler(ORIGINAL, admission);
    submit(&mut s, 1, 2048, 8);
    let b = s.next_batch().unwrap().unwrap();
    commit(&mut s, b);
    submit(&mut s, 2, 1, 32);
    s.preempt_for_next_step(0, |_, _| None).unwrap();
    let b = s.next_batch().unwrap().unwrap();
    commit(&mut s, b);
    let victims = s.preempt_for_next_step(0, |_, _| None).unwrap();
    assert_eq!(victims, vec![RequestId::new(1)]);
    let b = s.next_batch().unwrap().unwrap();
    assert_eq!(counts(&b), (512, 1));
    assert!(
        b.items()
            .iter()
            .any(|item| item.request_id() == RequestId::new(1)
                && item.phase() == BatchPhase::Prefill { tokens: 512 })
    );
    commit(&mut s, b);
    for _ in 0..100 {
        s.preempt_for_next_step(0, |_, _| None).unwrap();
        let Some(b) = s.next_batch().unwrap() else {
            break;
        };
        commit(&mut s, b);
    }
    assert_eq!(s.kv_stats().active_sequences(), 0);
    assert_eq!(s.reserved_pages(), 0);
}

#[test]
fn large_timing_budget_still_reserves_all_decode_rows() {
    let mut s = scheduler(ORIGINAL, AdmissionPolicy::Reserved);
    for id in 1..=32 {
        submit(&mut s, id, 1, 64);
    }
    let b = s.next_batch().unwrap().unwrap();
    commit(&mut s, b);
    for id in 33..=64 {
        submit(&mut s, id, 8192, 8);
    }
    s.observe_step_time(2048, 0, 1);
    s.observe_step_time(0, 32, 1_000_000);
    s.observe_step_time(0, 32, 1_000_000);
    let b = s.next_batch().unwrap().unwrap();
    let (prefill, decode) = counts(&b);
    assert_eq!(decode, 32);
    assert!(prefill >= 512 && prefill <= 8192 - decode);
    commit(&mut s, b);
}

#[test]
fn batches_report_the_selected_cap_and_the_itl_target_raises_it() {
    for (policy, caps) in [
        (PrefillBudget::Fixed, [None, None]),
        (ORIGINAL, [Some(512), Some(512)]),
        (
            PrefillBudget::DecodeAware(PrefillTarget::default()),
            [Some(512), Some(2048)],
        ),
    ] {
        let mut s = scheduler(policy, AdmissionPolicy::Reserved);
        submit(&mut s, 1, 1, 64);
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.prefill_cap(), None, "{policy:?}: nothing decodes yet");
        commit(&mut s, batch);
        for id in 2..=9 {
            submit(&mut s, id, 16384, 8);
        }
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.prefill_cap(), caps[0], "{policy:?}: cold");
        commit(&mut s, batch);
        // 16.4 ms decode; 2048-token steps at 77.6 us per token.
        s.observe_step_time(2048, 0, 158_900);
        s.observe_step_time(0, 1, 16_400);
        s.observe_step_time(0, 1, 16_400);
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.prefill_cap(), caps[1], "{policy:?}: measured");
        if let Some(cap) = caps[1] {
            assert_eq!(counts(&batch).0, cap);
        }
        commit(&mut s, batch);
        assert_eq!(
            s.step_time_estimates().map(|e| e.decode_micros(1)),
            (policy != PrefillBudget::Fixed).then_some(Some(16_400))
        );
    }
}
