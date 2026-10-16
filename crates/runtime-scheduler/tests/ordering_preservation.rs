//! The uncapped prefill order keeps its previous schedule where no started
//! prompt is being starved: one-chunk prompts, prompt-end anchors and
//! preempted requests under speculation. These traces were recorded before
//! the starvation guard; each line is one batch item.

use std::sync::Arc;

use runtime_kv::{KvGeometry, PrefixChainV1};
use runtime_scheduler::{
    AdmissionPolicy, BatchOutcome, BatchPhase, DraftLimits, OptimisticAdmission, PrefixRequestV1,
    RequestId, RequestSpec, Scheduler, SchedulerConfig,
};

fn submit(s: &mut Scheduler, id: u64, prompt: usize, output: usize, prefix: bool) {
    let chain = prefix.then(|| {
        let tokens = (0..prompt)
            .map(|position| id as u32 * 1000 + position as u32)
            .collect::<Vec<_>>();
        PrefixRequestV1::new(
            Arc::new(PrefixChainV1::new(4, Arc::from(tokens), Vec::new()).unwrap()),
            true,
        )
    });
    s.submit_with_prefix(
        RequestSpec::new(RequestId::new(id), prompt, output).unwrap(),
        chain,
    )
    .unwrap();
}

/// Commits one batch, appending its items to `trace`. Verify rows answer a
/// different token than decode, as phase-specific arithmetic may at a near
/// tie, so a changed phase shows in the trace.
fn commit(
    s: &mut Scheduler,
    trace: &mut Vec<String>,
    name: &str,
    tick: usize,
    drafts: bool,
) -> bool {
    if drafts {
        s.preempt_for_next_step(8, |_, _| None).unwrap();
    }
    let limits = if drafts {
        DraftLimits::new(8, 32)
    } else {
        DraftLimits::NONE
    };
    let Some(batch) = s
        .next_batch_with_drafts(limits, |candidates, budget| {
            let mut left = budget.spare_draft_tokens().min(budget.spare_rows());
            candidates
                .iter()
                .map(|candidate| {
                    let tokens = left.min(candidate.maximum_draft_tokens());
                    left -= tokens;
                    tokens
                })
                .collect()
        })
        .unwrap()
    else {
        return false;
    };
    let mut outcomes = Vec::new();
    for (item, append) in batch.items().iter().zip(batch.kv_plan().appends()) {
        let emitted = s.emitted_tokens(item.request_id()).unwrap();
        let token = if matches!(item.phase(), BatchPhase::Verify { .. }) {
            20
        } else {
            10
        };
        trace.push(format!(
            "{name} tick={tick} id={} phase={:?} start={} anchor={:?} output={} position={} token={token}",
            item.request_id().get(),
            item.phase(),
            append.previous_tokens(),
            item.anchor_tokens(),
            item.expects_output(),
            emitted
        ));
        outcomes.push(BatchOutcome::success(
            item.request_id(),
            item.phase(),
            usize::from(item.expects_output()),
            false,
        ));
    }
    s.commit_batch(batch.id(), &outcomes).unwrap();
    for capture in s.take_prefix_captures() {
        s.begin_prefix_capture(&capture).unwrap();
        s.end_prefix_capture(&capture).unwrap();
    }
    s.reap_terminal(64).unwrap();
    s.audit().unwrap();
    true
}

#[test]
fn unstarved_workloads_keep_their_previous_schedule() {
    let mut trace = Vec::new();
    let mut ordinary = Scheduler::new(
        SchedulerConfig::new(32, 256, 64, 16, 1).unwrap(),
        KvGeometry::new(4, 1024).unwrap(),
    )
    .unwrap();
    submit(&mut ordinary, 1, 13, 4, false);
    for tick in 0..16 {
        if tick < 6 {
            submit(&mut ordinary, tick as u64 * 2 + 10, 7, 4, false);
            submit(&mut ordinary, tick as u64 * 2 + 11, 13, 4, false);
        }
        if !commit(&mut ordinary, &mut trace, "ordinary-one-chunk", tick, false) {
            break;
        }
    }
    let anchored_config = |budget| {
        SchedulerConfig::new(8, 64, budget, 16, 1)
            .unwrap()
            .with_prefill_granule(8)
            .unwrap()
            .with_anchor_items(1)
            .with_prefix_cache(1)
    };
    let mut anchored =
        Scheduler::new(anchored_config(64), KvGeometry::new(4, 512).unwrap()).unwrap();
    submit(&mut anchored, 1, 45, 1, true);
    commit(&mut anchored, &mut trace, "anchor-chunks", 0, false);
    commit(&mut anchored, &mut trace, "anchor-chunks", 1, false);
    submit(&mut anchored, 2, 13, 1, true);
    for tick in 2..6 {
        if !commit(&mut anchored, &mut trace, "anchor-chunks", tick, false) {
            break;
        }
    }
    let mut one_chunk =
        Scheduler::new(anchored_config(32), KvGeometry::new(4, 512).unwrap()).unwrap();
    for id in 1..=3 {
        submit(&mut one_chunk, id, 13, 1, true);
    }
    commit(&mut one_chunk, &mut trace, "one-chunk-anchors", 0, false);
    for id in 4..=6 {
        submit(&mut one_chunk, id, 13, 1, true);
    }
    for tick in 1..8 {
        if !commit(&mut one_chunk, &mut trace, "one-chunk-anchors", tick, false) {
            break;
        }
    }
    let admission = AdmissionPolicy::Optimistic(
        OptimisticAdmission::new(0, 1)
            .unwrap()
            .with_forced_preemption(1)
            .unwrap(),
    );
    let mut preempted = Scheduler::new(
        SchedulerConfig::new(4, 64, 9, 9, 1)
            .unwrap()
            .with_admission(admission),
        KvGeometry::new(4, 512).unwrap(),
    )
    .unwrap();
    submit(&mut preempted, 1, 6, 12, false);
    commit(&mut preempted, &mut trace, "preemption-verify", 0, true);
    submit(&mut preempted, 2, 8, 1, false);
    for tick in 1..18 {
        if !commit(&mut preempted, &mut trace, "preemption-verify", tick, true) {
            break;
        }
    }
    let expected = EXPECTED.lines().collect::<Vec<_>>();
    for (index, (line, expected)) in trace.iter().zip(&expected).enumerate() {
        assert_eq!(line, expected, "trace line {index}");
    }
    assert_eq!(trace.len(), expected.len());
}

const EXPECTED: &str = "\
ordinary-one-chunk tick=0 id=1 phase=Prefill { tokens: 13 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=0 id=10 phase=Prefill { tokens: 7 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=0 id=11 phase=Prefill { tokens: 13 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=1 id=1 phase=Decode start=13 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=1 id=12 phase=Prefill { tokens: 7 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=1 id=10 phase=Decode start=7 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=1 id=11 phase=Decode start=13 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=1 id=13 phase=Prefill { tokens: 13 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=2 id=1 phase=Decode start=14 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=2 id=14 phase=Prefill { tokens: 7 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=2 id=10 phase=Decode start=8 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=2 id=11 phase=Decode start=14 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=2 id=12 phase=Decode start=7 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=2 id=13 phase=Decode start=13 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=2 id=15 phase=Prefill { tokens: 13 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=3 id=1 phase=Decode start=15 anchor=None output=true position=3 token=10
ordinary-one-chunk tick=3 id=16 phase=Prefill { tokens: 7 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=3 id=10 phase=Decode start=9 anchor=None output=true position=3 token=10
ordinary-one-chunk tick=3 id=11 phase=Decode start=15 anchor=None output=true position=3 token=10
ordinary-one-chunk tick=3 id=12 phase=Decode start=8 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=3 id=13 phase=Decode start=14 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=3 id=14 phase=Decode start=7 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=3 id=15 phase=Decode start=13 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=3 id=17 phase=Prefill { tokens: 13 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=4 id=12 phase=Decode start=9 anchor=None output=true position=3 token=10
ordinary-one-chunk tick=4 id=18 phase=Prefill { tokens: 7 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=4 id=13 phase=Decode start=15 anchor=None output=true position=3 token=10
ordinary-one-chunk tick=4 id=14 phase=Decode start=8 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=4 id=15 phase=Decode start=14 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=4 id=16 phase=Decode start=7 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=4 id=17 phase=Decode start=13 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=4 id=19 phase=Prefill { tokens: 13 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=5 id=14 phase=Decode start=9 anchor=None output=true position=3 token=10
ordinary-one-chunk tick=5 id=20 phase=Prefill { tokens: 7 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=5 id=15 phase=Decode start=15 anchor=None output=true position=3 token=10
ordinary-one-chunk tick=5 id=16 phase=Decode start=8 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=5 id=17 phase=Decode start=14 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=5 id=18 phase=Decode start=7 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=5 id=19 phase=Decode start=13 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=5 id=21 phase=Prefill { tokens: 13 } start=0 anchor=None output=true position=0 token=10
ordinary-one-chunk tick=6 id=16 phase=Decode start=9 anchor=None output=true position=3 token=10
ordinary-one-chunk tick=6 id=17 phase=Decode start=15 anchor=None output=true position=3 token=10
ordinary-one-chunk tick=6 id=18 phase=Decode start=8 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=6 id=19 phase=Decode start=14 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=6 id=20 phase=Decode start=7 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=6 id=21 phase=Decode start=13 anchor=None output=true position=1 token=10
ordinary-one-chunk tick=7 id=18 phase=Decode start=9 anchor=None output=true position=3 token=10
ordinary-one-chunk tick=7 id=19 phase=Decode start=15 anchor=None output=true position=3 token=10
ordinary-one-chunk tick=7 id=20 phase=Decode start=8 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=7 id=21 phase=Decode start=14 anchor=None output=true position=2 token=10
ordinary-one-chunk tick=8 id=20 phase=Decode start=9 anchor=None output=true position=3 token=10
ordinary-one-chunk tick=8 id=21 phase=Decode start=15 anchor=None output=true position=3 token=10
anchor-chunks tick=0 id=1 phase=Prefill { tokens: 16 } start=0 anchor=None output=false position=0 token=10
anchor-chunks tick=1 id=1 phase=Prefill { tokens: 16 } start=16 anchor=None output=false position=0 token=10
anchor-chunks tick=2 id=2 phase=Prefill { tokens: 13 } start=0 anchor=Some(8) output=true position=0 token=10
anchor-chunks tick=2 id=1 phase=Prefill { tokens: 8 } start=32 anchor=None output=false position=0 token=10
anchor-chunks tick=3 id=1 phase=Prefill { tokens: 5 } start=40 anchor=None output=true position=0 token=10
one-chunk-anchors tick=0 id=1 phase=Prefill { tokens: 13 } start=0 anchor=Some(8) output=true position=0 token=10
one-chunk-anchors tick=0 id=2 phase=Prefill { tokens: 8 } start=0 anchor=None output=false position=0 token=10
one-chunk-anchors tick=0 id=3 phase=Prefill { tokens: 8 } start=0 anchor=None output=false position=0 token=10
one-chunk-anchors tick=1 id=4 phase=Prefill { tokens: 13 } start=0 anchor=Some(8) output=true position=0 token=10
one-chunk-anchors tick=1 id=5 phase=Prefill { tokens: 8 } start=0 anchor=None output=false position=0 token=10
one-chunk-anchors tick=1 id=6 phase=Prefill { tokens: 8 } start=0 anchor=None output=false position=0 token=10
one-chunk-anchors tick=2 id=2 phase=Prefill { tokens: 5 } start=8 anchor=None output=true position=0 token=10
one-chunk-anchors tick=2 id=3 phase=Prefill { tokens: 5 } start=8 anchor=None output=true position=0 token=10
one-chunk-anchors tick=2 id=5 phase=Prefill { tokens: 5 } start=8 anchor=None output=true position=0 token=10
one-chunk-anchors tick=2 id=6 phase=Prefill { tokens: 5 } start=8 anchor=None output=true position=0 token=10
preemption-verify tick=0 id=1 phase=Prefill { tokens: 6 } start=0 anchor=None output=true position=0 token=10
preemption-verify tick=1 id=2 phase=Prefill { tokens: 8 } start=0 anchor=None output=true position=0 token=10
preemption-verify tick=2 id=1 phase=Prefill { tokens: 7 } start=0 anchor=None output=true position=1 token=10
preemption-verify tick=3 id=1 phase=Verify { draft_tokens: 8 } start=7 anchor=None output=true position=2 token=20
preemption-verify tick=4 id=1 phase=Verify { draft_tokens: 8 } start=8 anchor=None output=true position=3 token=20
preemption-verify tick=5 id=1 phase=Verify { draft_tokens: 7 } start=9 anchor=None output=true position=4 token=20
preemption-verify tick=6 id=1 phase=Verify { draft_tokens: 6 } start=10 anchor=None output=true position=5 token=20
preemption-verify tick=7 id=1 phase=Verify { draft_tokens: 5 } start=11 anchor=None output=true position=6 token=20
preemption-verify tick=8 id=1 phase=Verify { draft_tokens: 4 } start=12 anchor=None output=true position=7 token=20
preemption-verify tick=9 id=1 phase=Verify { draft_tokens: 3 } start=13 anchor=None output=true position=8 token=20
preemption-verify tick=10 id=1 phase=Verify { draft_tokens: 2 } start=14 anchor=None output=true position=9 token=20
preemption-verify tick=11 id=1 phase=Verify { draft_tokens: 1 } start=15 anchor=None output=true position=10 token=20
preemption-verify tick=12 id=1 phase=Decode start=16 anchor=None output=true position=11 token=10";
