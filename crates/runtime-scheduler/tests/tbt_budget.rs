use runtime_kv::KvGeometry;
use runtime_scheduler::{
    AdmissionPolicy, Batch, BatchOutcome, BatchPhase, ConfigError, DraftLimits, MixedPolicy,
    OptimisticAdmission, PrefillBudget, RequestId, RequestSpec, RequestState, Scheduler,
    SchedulerConfig, StepShape,
};

fn scheduler(policy: PrefillBudget, admission: AdmissionPolicy) -> Scheduler {
    Scheduler::new(
        SchedulerConfig::new(64, 128, 8192, 2048, 1)
            .unwrap()
            .with_lone_prefill_expansion(true)
            .with_prefill_budget(policy)
            .with_admission(admission),
        KvGeometry::new(16, 65536).unwrap(),
    )
    .unwrap()
}

fn tbt(target: u32, decode_steps: u32) -> PrefillBudget {
    PrefillBudget::Tbt(MixedPolicy::new(target, 512, decode_steps).unwrap())
}

fn submit(s: &mut Scheduler, id: u64, prompt: usize, output: usize) {
    s.submit(RequestSpec::new(RequestId::new(id), prompt, output).unwrap())
        .unwrap();
}

/// A synthetic TP4 step: 14 ms of decode, and for prefill 36 ms fixed, 20 ms
/// more beside decodes, 72 us per token and 0.4 ns per attention pair.
fn cost(shape: StepShape) -> u64 {
    let decode = if shape.decode_items() > 0 { 14_000 } else { 0 };
    let prefill = if shape.prefill_tokens() == 0 {
        0
    } else {
        36_000
            + if shape.decode_items() > 0 { 20_000 } else { 0 }
            + shape.prefill_tokens() as u64 * 72
            + shape.prefill_pairs() * 4 / 10_000
    };
    decode + prefill
}

fn prefill_tokens(batch: &Batch) -> usize {
    batch
        .items()
        .iter()
        .map(|item| match item.phase() {
            BatchPhase::Prefill { tokens } => tokens,
            _ => 0,
        })
        .sum()
}

/// Commits one batch and reports its synthetic execution time to the model.
fn run(s: &mut Scheduler, batch: Batch) -> u64 {
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
    let micros = cost(batch.shape());
    s.observe_batch_time(batch.shape(), micros);
    micros
}

/// Whether any of `ids` still waits for or runs its prefill.
fn prefilling(s: &Scheduler, ids: std::ops::Range<u64>) -> bool {
    ids.into_iter().any(|id| {
        matches!(
            s.request_state(RequestId::new(id)),
            Ok(RequestState::Queued | RequestState::Admitted | RequestState::Prefilling)
        )
    })
}

/// Trains the TBT model from synthetic observations alone: decode steps
/// of `decode` items and varied prefill chunks beside them.
fn train(s: &mut Scheduler, decode: usize) {
    for _ in 0..2 {
        s.observe_batch_time(
            StepShape::new(0, 0, decode),
            cost(StepShape::new(0, 0, decode)),
        );
    }
    for step in 0..16_usize {
        let tokens = [2048, 1536, 700, 333, 1100, 512][step % 6];
        let context = step * 997;
        let pairs = (tokens * context + tokens * (tokens + 1) / 2) as u64;
        let shape = StepShape::new(tokens, pairs, decode);
        s.observe_batch_time(shape, cost(shape));
    }
    assert!(s.mixed_estimates().unwrap().trained());
}

/// Commits a batch without reporting its time to the model.
fn commit_only(s: &mut Scheduler, batch: &Batch) {
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

/// Each batch's items as (request, phase), in batch order.
fn items(batch: &Batch) -> Vec<(RequestId, BatchPhase)> {
    batch
        .items()
        .iter()
        .map(|item| (item.request_id(), item.phase()))
        .collect()
}

/// Four 8192-token prompts with nothing decoding, served to completion,
/// optionally after training the model. Returns each batch's items and
/// asserts that no step selected a cap or a prediction.
fn prefill_only_wave(policy: PrefillBudget, trained: bool) -> Vec<Vec<(RequestId, BatchPhase)>> {
    let mut s = scheduler(policy, AdmissionPolicy::Reserved);
    if trained {
        train(&mut s, 0);
    }
    for id in 1..=4 {
        submit(&mut s, id, 8192, 1);
    }
    let mut batches = Vec::new();
    while prefilling(&s, 1..5) {
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.shape().decode_items(), 0);
        assert_eq!(batch.prefill_cap(), None);
        assert_eq!(batch.predicted_micros(), None);
        batches.push(items(&batch));
        run(&mut s, batch);
        assert!(batches.len() <= 16, "prefill starved");
    }
    batches
}

/// Serves one `prompt`-token request alone to completion and returns the
/// prefill tokens of each step that prefilled.
fn serve_alone(s: &mut Scheduler, id: u64, prompt: usize) -> Vec<usize> {
    submit(s, id, prompt, 4);
    let mut chunks = Vec::new();
    while let Some(batch) = s.next_batch().unwrap() {
        assert_eq!(batch.items().len(), 1);
        let tokens = prefill_tokens(&batch);
        if tokens > 0 {
            chunks.push(tokens);
        }
        run(s, batch);
    }
    assert_eq!(
        s.request_state(RequestId::new(id)),
        Ok(RequestState::Finished)
    );
    s.reap_terminal(64).unwrap();
    chunks
}

/// 32 long decodes, then `prompts` long prompts that arrive while they run.
fn decoding_with_prompts(policy: PrefillBudget, prompts: usize, prompt: usize) -> Scheduler {
    let mut s = scheduler(policy, AdmissionPolicy::Reserved);
    for id in 1..=32 {
        submit(&mut s, id, 16, 4000);
    }
    let batch = s.next_batch().unwrap().unwrap();
    run(&mut s, batch);
    for id in 0..prompts as u64 {
        submit(&mut s, 100 + id, prompt, 8);
    }
    s
}

#[test]
fn policy_validation() {
    assert_eq!(
        MixedPolicy::new(0, 512, 0),
        Err(ConfigError::InvalidMixedPolicy)
    );
    assert_eq!(
        MixedPolicy::new(100, 0, 0),
        Err(ConfigError::InvalidMixedPolicy)
    );
    assert_eq!(
        MixedPolicy::new(100, 512, 9),
        Err(ConfigError::InvalidMixedPolicy)
    );
    let policy = MixedPolicy::default();
    assert_eq!(
        (
            policy.target_millis(),
            policy.min_prefill_tokens(),
            policy.decode_steps()
        ),
        (100, 512, 0)
    );
}

#[test]
fn cold_tbt_bounds_aggregate_prefill_before_the_fit_trains() {
    for decode_steps in [0, 2, 8] {
        let mut s = scheduler(tbt(150, decode_steps), AdmissionPolicy::Reserved);
        submit(&mut s, 1, 16, 1000);
        let batch = s.next_batch().unwrap().unwrap();
        commit_only(&mut s, &batch);
        for id in 2..10 {
            submit(&mut s, id, 8192, 8);
        }
        for _ in 0..8 {
            let batch = s.next_batch().unwrap().unwrap();
            assert!(cost(batch.shape()) < 150_000);
            assert_eq!(batch.prefill_cap(), Some(512));
            assert_eq!(batch.predicted_micros(), None);
            assert_eq!(prefill_tokens(&batch), 512);
            assert_eq!(batch.shape().decode_items(), 1);
            assert!(cost(batch.shape()) < 150_000);
            commit_only(&mut s, &batch);
        }
    }
}

#[test]
fn trained_tbt_prefill_only_steps_take_the_configured_chunk_and_budget() {
    let batches = prefill_only_wave(tbt(150, 2), true);
    // Four 2048-token chunks fill the 8192-token budget. The fit would cap
    // a step below 2048 tokens at 150 ms (36 ms plus 72 us per token), but
    // not without decodes.
    assert_eq!(batches.len(), 4);
    for batch in &batches {
        assert_eq!(batch.len(), 4);
        assert!(
            batch
                .iter()
                .all(|(_, phase)| *phase == BatchPhase::Prefill { tokens: 2048 })
        );
    }
    assert_eq!(batches, prefill_only_wave(PrefillBudget::Fixed, false));
}

#[test]
fn cold_tbt_prefill_only_steps_take_the_configured_chunk_and_budget() {
    // Before the fit trains, a prefill-only step is not held to the
    // 512-token progress floor.
    let batches = prefill_only_wave(tbt(150, 2), false);
    assert_eq!(batches.len(), 4);
    assert!(batches.iter().all(|batch| batch.len() == 4));
    assert_eq!(batches, prefill_only_wave(PrefillBudget::Fixed, false));
}

#[test]
fn a_prompt_served_alone_is_chunked_the_same_before_and_after_training() {
    let mut fixed = scheduler(PrefillBudget::Fixed, AdmissionPolicy::Reserved);
    let expected = serve_alone(&mut fixed, 1, 20000);
    assert_eq!(expected, [8192, 8192, 3616]);
    let mut s = scheduler(tbt(150, 2), AdmissionPolicy::Reserved);
    // Each serve contributes three prefill samples, crossing the model's
    // eight-sample warmup even with expanded chunks. The first serves
    // run cold and the later ones with a fit.
    let mut trained_serves = 0;
    for id in 1..=6 {
        let trained = s.mixed_estimates().unwrap().trained();
        assert_eq!(serve_alone(&mut s, id, 20000), expected, "serve {id}");
        trained_serves += usize::from(trained);
    }
    assert!((1..6).contains(&trained_serves), "{trained_serves}");
}

#[test]
fn decoding_requests_keep_the_tbt_cap_beside_them() {
    for trained in [false, true] {
        let mut s = scheduler(tbt(150, 0), AdmissionPolicy::Reserved);
        if trained {
            train(&mut s, 1);
        }
        submit(&mut s, 1, 16, 1000);
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.prefill_cap(), None, "nothing decodes yet");
        commit_only(&mut s, &batch);
        submit(&mut s, 2, 8192, 8);
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.shape().decode_items(), 1);
        let cap = batch.prefill_cap().unwrap();
        assert_eq!(prefill_tokens(&batch), cap);
        if trained {
            // 70 ms before tokens and 72 us per token: about 1024 tokens
            // fit 150 ms, fewer than the 2048-token chunk.
            assert!((640..2048).contains(&cap) && cap % 128 == 0, "{cap}");
            assert!(batch.predicted_micros().unwrap() <= 150_000);
        } else {
            assert_eq!(cap, 512, "the progress floor");
            assert_eq!(batch.predicted_micros(), None);
        }
    }
}

#[test]
fn fixed_budget_keeps_full_chunks() {
    let mut s = decoding_with_prompts(PrefillBudget::Fixed, 3, 8192);
    let batch = s.next_batch().unwrap().unwrap();
    assert_eq!(prefill_tokens(&batch), 3 * 2048);
    assert_eq!(batch.prefill_cap(), None);
    assert_eq!(batch.predicted_micros(), None);
    run(&mut s, batch);
    assert_eq!(s.mixed_estimates(), None);
}

#[test]
fn trained_budget_bounds_mixed_steps_and_every_prompt_completes() {
    let target = 150;
    let mut s = decoding_with_prompts(tbt(target, 0), 12, 20_000);
    let (mut worst_trained, mut worst_untrained) = (0, 0);
    let (mut steps, mut trained) = (0, 0);
    while prefilling(&s, 100..112) {
        steps += 1;
        assert!(steps < 1000, "prefill starved");
        let batch = s.next_batch().unwrap().unwrap();
        let tokens = prefill_tokens(&batch);
        let predicted = batch.predicted_micros();
        let cap = batch.prefill_cap();
        let micros = run(&mut s, batch);
        if tokens == 0 {
            continue;
        }
        match (cap, predicted) {
            (Some(cap), Some(predicted)) => {
                assert!(tokens <= cap);
                trained += 1;
                // Early fits are still settling. Then the synthetic cost,
                // which has the model's form, fits within the small pull
                // towards the previous fit.
                if trained > 8 {
                    assert!(
                        predicted.abs_diff(micros) <= micros / 500,
                        "{predicted} {micros}"
                    );
                    worst_trained = worst_trained.max(micros);
                }
            }
            (Some(cap), None) => {
                assert!(tokens <= cap && cap <= 512);
                worst_untrained = worst_untrained.max(micros);
            }
            other => panic!("{other:?}"),
        }
    }
    assert!(
        worst_untrained > 0 && worst_untrained < 150_000,
        "{worst_untrained}"
    );
    assert!(worst_trained > 0 && worst_trained <= u64::from(target) * 1000);
    let estimates = s.mixed_estimates().unwrap();
    assert!(estimates.trained());
    // Every step here prefills, so no decode estimate exists and the fixed
    // terms absorb the 14 ms decode; the predictions above check the fit.
    assert!(s.mixed_estimates().unwrap().decode().next().is_none());
    let [.., (_, pair)] = estimates.terms_seconds();
    assert!((pair - 0.4e-9).abs() < 1e-11, "{pair}");
}

#[test]
fn the_budget_shrinks_as_the_cached_context_grows() {
    let mut s = decoding_with_prompts(tbt(150, 0), 1, 60_000);
    let mut chunks = Vec::new();
    for _ in 0..200 {
        let Some(batch) = s.next_batch().unwrap() else {
            break;
        };
        let tokens = prefill_tokens(&batch);
        let trained = batch.predicted_micros().is_some();
        run(&mut s, batch);
        if tokens > 0 && trained {
            chunks.push(tokens);
        }
        if !prefilling(&s, 100..101) {
            break;
        }
    }
    // Cold chunks train the fit, then fitted chunks fall as context grows.
    assert!(chunks.len() > 10, "{chunks:?}");
    // The first chunks come from early fits that are still settling; the
    // last is the tail.
    let body = &chunks[4..chunks.len() - 1];
    assert!(body.windows(2).all(|pair| pair[1] <= pair[0]), "{chunks:?}");
    assert!(body.first() > body.last(), "{chunks:?}");
    assert!(
        body.iter()
            .all(|&tokens| tokens >= 512 && tokens % 128 == 0)
    );
}

#[test]
fn verify_items_count_as_decode_items_and_train_the_decode_estimate() {
    let mut s = scheduler(tbt(100, 0), AdmissionPolicy::Reserved);
    for id in 1..=4 {
        submit(&mut s, id, 16, 64);
    }
    let batch = s.next_batch().unwrap().unwrap();
    run(&mut s, batch);
    for _ in 0..3 {
        let batch = s
            .next_batch_with_drafts(DraftLimits::new(3, 32), |candidates, _| {
                vec![3; candidates.len()]
            })
            .unwrap()
            .unwrap();
        assert!(
            batch
                .items()
                .iter()
                .all(|item| item.phase() == BatchPhase::Verify { draft_tokens: 3 })
        );
        assert_eq!(batch.shape(), StepShape::new(0, 0, 4));
        run(&mut s, batch);
    }
    let estimates = s.mixed_estimates().unwrap();
    assert_eq!(
        estimates.decode().collect::<Vec<_>>(),
        [(4, 14_000, 3)],
        "verify steps train the decode estimate"
    );
}

#[test]
fn preemption_recompute_uses_the_trained_budget_and_releases_pages() {
    let mut s = scheduler(
        tbt(100, 0),
        AdmissionPolicy::Optimistic(
            OptimisticAdmission::new(0, 1)
                .unwrap()
                .with_forced_preemption(2)
                .unwrap(),
        ),
    );
    submit(&mut s, 1, 512, 64);
    let b = s.next_batch().unwrap().unwrap();
    run(&mut s, b);
    // Train the model on steps of one decode beside varied chunks.
    for _ in 0..3 {
        s.observe_batch_time(StepShape::new(0, 0, 1), 14_000);
    }
    for tokens in [2048, 1536, 700, 2048, 333, 1024, 1900, 256, 640] {
        let shape = StepShape::new(tokens, (tokens * (tokens + 1) / 2) as u64, 1);
        s.observe_batch_time(shape, cost(shape));
    }
    submit(&mut s, 2, 1, 32);
    s.preempt_for_next_step(0, |_, _| None).unwrap();
    let b = s.next_batch().unwrap().unwrap();
    run(&mut s, b);
    let victims = s.preempt_for_next_step(0, |_, _| None).unwrap();
    assert_eq!(victims, vec![RequestId::new(1)]);
    let b = s.next_batch().unwrap().unwrap();
    let cap = b.prefill_cap().expect("a decode is running");
    let recompute = b
        .items()
        .iter()
        .find(|item| item.request_id() == RequestId::new(1))
        .unwrap();
    // 14 + 56 ms before tokens leaves 30 ms: 384 tokens, below the floor.
    assert_eq!(cap, 512);
    assert_eq!(recompute.phase(), BatchPhase::Prefill { tokens: 512 });
    run(&mut s, b);
    for _ in 0..200 {
        s.preempt_for_next_step(0, |_, _| None).unwrap();
        let Some(b) = s.next_batch().unwrap() else {
            break;
        };
        run(&mut s, b);
    }
    assert_eq!(s.kv_stats().active_sequences(), 0);
    assert_eq!(s.reserved_pages(), 0);
}

#[test]
fn interleaved_decode_steps_follow_each_trained_mixed_step_and_prefill_still_finishes() {
    let mut s = decoding_with_prompts(tbt(150, 2), 0, 0);
    train(&mut s, 32);
    for id in 100..102 {
        submit(&mut s, id, 6000, 8);
    }
    let mut pattern = String::new();
    for _ in 0..200 {
        let batch = s.next_batch().unwrap().unwrap();
        pattern.push(if prefill_tokens(&batch) > 0 { 'P' } else { 'd' });
        commit_only(&mut s, &batch);
        if !prefilling(&s, 100..102) {
            break;
        }
    }
    let prefill_steps = pattern.matches('P').count();
    // 12,000 prompt tokens at 512 or more per mixed step, two decode steps
    // after each: the starvation bound.
    assert!(prefill_steps <= 12_000 / 512 + 2, "{pattern}");
    assert!(pattern.len() <= 3 * prefill_steps, "{pattern}");
    assert!(
        pattern.starts_with("Pdd") && pattern.ends_with('P'),
        "{pattern}"
    );
    let gaps: Vec<_> = pattern.split('P').skip(1).collect();
    assert!(
        gaps[..gaps.len() - 1].iter().all(|gap| *gap == "dd"),
        "{pattern}"
    );
}

#[test]
fn cold_cap_trains_without_owing_decode_steps() {
    for steps in [0, 1, 8] {
        let mut s = decoding_with_prompts(tbt(150, steps), 12, 6000);
        let mut warm = 0;
        loop {
            let trained = s.mixed_estimates().unwrap().trained();
            let batch = s.next_batch().unwrap().unwrap();
            if trained {
                assert!(batch.predicted_micros().is_some(), "N={steps}");
                break;
            }
            assert_eq!(batch.prefill_cap(), Some(512));
            assert_eq!(prefill_tokens(&batch), 512, "N={steps}, step {warm}");
            run(&mut s, batch);
            warm += 1;
            assert!(warm < 100);
        }
        assert!(warm >= 7, "N={steps}: {warm}");
    }
}

#[test]
fn owed_decode_steps_change_only_when_a_batch_commits() {
    let mut s = decoding_with_prompts(tbt(150, 2), 0, 0);
    train(&mut s, 32);
    submit(&mut s, 100, 20_000, 8);
    // A mixed batch that is rolled back owes nothing.
    let batch = s.next_batch().unwrap().unwrap();
    assert!(prefill_tokens(&batch) > 0);
    s.rollback_batch(batch.id()).unwrap();
    let batch = s.next_batch().unwrap().unwrap();
    assert!(prefill_tokens(&batch) > 0, "the rollback left a debt");
    commit_only(&mut s, &batch);
    // Rolled-back decode-only batches do not pay the debt.
    for _ in 0..3 {
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(prefill_tokens(&batch), 0);
        s.rollback_batch(batch.id()).unwrap();
    }
    let mut pattern = String::new();
    for _ in 0..4 {
        let batch = s.next_batch().unwrap().unwrap();
        pattern.push(if prefill_tokens(&batch) > 0 { 'P' } else { 'd' });
        commit_only(&mut s, &batch);
    }
    assert_eq!(pattern, "ddPd");
}

#[test]
fn owed_decode_steps_lapse_when_no_request_decodes() {
    let mut s = scheduler(tbt(150, 4), AdmissionPolicy::Reserved);
    train(&mut s, 1);
    submit(&mut s, 1, 16, 2);
    let batch = s.next_batch().unwrap().unwrap();
    commit_only(&mut s, &batch);
    submit(&mut s, 2, 12_000, 8);
    submit(&mut s, 3, 40_000, 8);
    // Request 1's last token comes from a mixed step, which owes four
    // decode-only steps that no request is left to use.
    let batch = s.next_batch().unwrap().unwrap();
    assert!(prefill_tokens(&batch) > 0 && batch.shape().decode_items() == 1);
    commit_only(&mut s, &batch);
    assert_eq!(
        s.request_state(RequestId::new(1)),
        Ok(RequestState::Finished)
    );
    let mut prefill_only = 0;
    while prefilling(&s, 2..3) {
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.shape().decode_items(), 0);
        commit_only(&mut s, &batch);
        prefill_only += 1;
    }
    assert!(prefill_only > 0);
    assert!(prefilling(&s, 3..4));
    // Request 2 now decodes beside request 3's prefill, which continues.
    let batch = s.next_batch().unwrap().unwrap();
    assert_eq!(batch.shape().decode_items(), 1);
    assert!(prefill_tokens(&batch) > 0);
}

#[test]
fn a_stream_of_new_prompts_cannot_starve_an_older_prefill() {
    let mut s = scheduler(tbt(100, 0), AdmissionPolicy::Reserved);
    submit(&mut s, 1, 16, 100_000);
    let batch = s.next_batch().unwrap().unwrap();
    commit_only(&mut s, &batch);
    train(&mut s, 1);
    submit(&mut s, 2, 20_000, 8);
    let old = RequestId::new(2);
    let mut starved = 0;
    let mut longest = 0;
    for id in 1000..1128 {
        // A short new turn arrives before every step and takes the whole
        // 512-token cap by itself.
        submit(&mut s, id, 512, 1);
        let before = s.kv_tokens(old).unwrap_or(0);
        let batch = s.next_batch().unwrap().unwrap();
        assert!(batch.prefill_cap().is_some_and(|cap| cap <= 1024));
        commit_only(&mut s, &batch);
        s.reap_terminal(64).unwrap();
        if s.kv_tokens(old).unwrap_or(0) == before {
            starved += 1;
            longest = longest.max(starved);
        } else {
            starved = 0;
        }
        if !prefilling(&s, 2..3) {
            break;
        }
    }
    assert!(
        longest <= 2,
        "the old prefill waited {longest} steps in a row"
    );
    assert!(!prefilling(&s, 2..3), "the old prefill never finished");
}

#[test]
fn ineligible_timings_and_new_arrivals_cannot_starve_a_cold_prefill() {
    for with_decode in [false, true] {
        let mut s = scheduler(tbt(150, 2), AdmissionPolicy::Reserved);
        if with_decode {
            submit(&mut s, 1, 16, 100_000);
            let batch = s.next_batch().unwrap().unwrap();
            commit_only(&mut s, &batch);
        }
        // Beside a decode the cold cap is the floor; without one, the step
        // takes the whole step budget.
        let (cap, first) = if with_decode {
            (Some(512), 512)
        } else {
            (None, 8192)
        };
        submit(&mut s, 2, 20_000, 8);
        let batch = s.next_batch().unwrap().unwrap();
        commit_only(&mut s, &batch);
        assert_eq!(s.kv_tokens(RequestId::new(2)).unwrap(), first);
        for id in 1000..1128 {
            submit(&mut s, id, 512, 1);
            let batch = s.next_batch().unwrap().unwrap();
            assert_eq!(batch.prefill_cap(), cap);
            // The engine excludes media encode timings from observations.
            commit_only(&mut s, &batch);
            assert!(!s.mixed_estimates().unwrap().trained());
            s.reap_terminal(64).unwrap();
            if !prefilling(&s, 2..3) {
                break;
            }
        }
        assert!(!prefilling(&s, 2..3), "the cold prefill never finished");
    }
}

#[test]
fn tiny_tbt_token_budgets_preserve_prefill_and_decode_progress() {
    for budget in [1, 2, 127] {
        for trained in [false, true] {
            let mut s = Scheduler::new(
                SchedulerConfig::new(4, 8, budget, budget, 1)
                    .unwrap()
                    .with_prefill_budget(tbt(150, 0)),
                KvGeometry::new(16, 65536).unwrap(),
            )
            .unwrap();
            if trained {
                train(&mut s, 1);
            }
            submit(&mut s, 1, 1, 8);
            let batch = s.next_batch().unwrap().unwrap();
            commit_only(&mut s, &batch);
            submit(&mut s, 2, budget * 3, 2);
            for _ in 0..32 {
                let Some(batch) = s.next_batch().unwrap() else {
                    break;
                };
                assert!(prefill_tokens(&batch) + batch.shape().decode_items() <= budget);
                commit_only(&mut s, &batch);
            }
            for id in [1, 2] {
                assert_eq!(
                    s.request_state(RequestId::new(id)),
                    Ok(RequestState::Finished)
                );
            }
        }
    }
}

#[test]
fn a_pessimistic_tbt_fit_keeps_the_progress_floor() {
    for with_decode in [false, true] {
        let mut s = scheduler(tbt(1, 0), AdmissionPolicy::Reserved);
        train(&mut s, 1);
        if with_decode {
            submit(&mut s, 1, 16, 1000);
            let batch = s.next_batch().unwrap().unwrap();
            commit_only(&mut s, &batch);
        }
        submit(&mut s, 2, 2048, 8);
        if !with_decode {
            // Without a decode the target does not apply: one full chunk.
            let batch = s.next_batch().unwrap().unwrap();
            assert_eq!(batch.prefill_cap(), None);
            assert_eq!(prefill_tokens(&batch), 2048);
            commit_only(&mut s, &batch);
            assert!(!prefilling(&s, 2..3));
            continue;
        }
        for _ in 0..4 {
            let batch = s.next_batch().unwrap().unwrap();
            assert_eq!(batch.prefill_cap(), Some(512));
            assert_eq!(prefill_tokens(&batch), 512);
            assert!(cost(batch.shape()) > 1000);
            assert!(batch.predicted_micros().unwrap() > 1000);
            commit_only(&mut s, &batch);
        }
        assert!(!prefilling(&s, 2..3));
    }
}

/// A 64-token granule beside `decoders` decoding requests, with a 64-token
/// step budget, so the mixed prefill cap is 64 - `decoders`: a long prompt
/// still prefills, in chunks ending on the granule or at its end, and the
/// decoders keep decoding. A granule of one takes the cap.
#[test]
fn a_prefill_cap_below_the_granule_still_prefills_beside_decodes() {
    let policies = [
        tbt(100, 0),
        PrefillBudget::DecodeAware(runtime_scheduler::PrefillTarget::default()),
    ];
    for policy in policies {
        for granule in [64, 1] {
            // Caps of 33, of 1, and of zero, which falls back to a chunk.
            for decoders in [31_u64, 63, 64] {
                let mut s = Scheduler::new(
                    SchedulerConfig::new(80, 160, 64, 64, 1)
                        .unwrap()
                        .with_prefill_granule(granule)
                        .unwrap()
                        .with_prefill_budget(policy),
                    KvGeometry::new(16, 65536).unwrap(),
                )
                .unwrap();
                for id in 0..decoders {
                    submit(&mut s, id, 1, 1000);
                }
                while prefilling(&s, 0..decoders) {
                    let batch = s.next_batch().unwrap().unwrap();
                    commit_only(&mut s, &batch);
                }
                let long = decoders;
                submit(&mut s, long, 300, 4);
                let mut chunks = Vec::new();
                let mut without_decode = 0;
                for _ in 0..700 {
                    if !prefilling(&s, long..long + 1) {
                        break;
                    }
                    let batch = s.next_batch().unwrap().unwrap();
                    assert!(batch.token_cost() <= 64);
                    let mut decoded = false;
                    for item in batch.items() {
                        match item.phase() {
                            BatchPhase::Prefill { tokens } => chunks.push(tokens),
                            _ => decoded = true,
                        }
                    }
                    // The decoders run at least every other step.
                    without_decode = if decoded { 0 } else { without_decode + 1 };
                    assert!(without_decode < 2, "{policy:?} granule {granule}");
                    commit_only(&mut s, &batch);
                }
                let context = format!("{policy:?} granule {granule}, {decoders} decoding");
                assert!(!prefilling(&s, long..long + 1), "{context}: {chunks:?}");
                assert_eq!(chunks.iter().sum::<usize>(), 300, "{context}");
                if granule == 64 {
                    assert_eq!(chunks, [64, 64, 64, 64, 44], "{context}");
                } else {
                    let cap = match 64 - decoders as usize {
                        0 => 64,
                        cap => cap,
                    };
                    assert!(chunks.iter().all(|&tokens| tokens <= cap), "{context}");
                }
            }
        }
    }
}
