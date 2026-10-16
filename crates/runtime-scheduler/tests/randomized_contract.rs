//! Seeded scheduling interleavings preserve ownership and accounting.
use std::collections::BTreeMap;
use std::sync::Arc;

use runtime_kv::{KvGeometry, PrefixChainV1};
use runtime_scheduler::*;
use std::cell::RefCell;
thread_local! { static STATS: RefCell<[usize;4]> = const { RefCell::new([0;4]) }; }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }
    fn chance(&mut self, p: u64) -> bool {
        self.below(100) < p
    }
}

const BLOCK: usize = 4;

fn toks(seed: u64, len: usize) -> Vec<u32> {
    // Shared prefix families so prefix hits happen.
    (0..len)
        .map(|p| (seed % 3) as u32 * 100_000 + p as u32)
        .collect()
}

fn chain(seed: u64, len: usize) -> PrefixRequestV1 {
    PrefixRequestV1::new(
        Arc::new(PrefixChainV1::new(BLOCK, Arc::from(toks(seed, len)), Vec::new()).unwrap()),
        true,
    )
}

fn run_case(case: u64) {
    let mut rng = Rng(case.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    let prefix = rng.chance(50);
    let rewrites = rng.chance(30);
    let granule = if rng.chance(30) { 4 } else { 1 };
    let drafts = if rng.chance(50) {
        1 + rng.below(4) as usize
    } else {
        0
    };
    let pages = 8 + rng.below(40) as usize;
    let seqs = 1 + rng.below(8) as usize;
    let policy = match rng.below(3) {
        0 => PrefillBudget::Fixed,
        1 => PrefillBudget::Tbt(
            MixedPolicy::new(
                1 + rng.below(50) as u32,
                1 + rng.below(12) as usize,
                rng.below(3) as u32,
            )
            .unwrap(),
        ),
        _ => PrefillBudget::DecodeAware(PrefillTarget::DecodeSteps(1)),
    };
    let admission = if rng.chance(80) {
        AdmissionPolicy::Optimistic(
            OptimisticAdmission::new(rng.below(100_000) as u32, rng.below(4) as usize).unwrap(),
        )
    } else {
        AdmissionPolicy::Reserved
    };
    let mut config = SchedulerConfig::new(seqs, 64, 32, 8, 1)
        .unwrap()
        .with_maximum_scored_rows(if rng.chance(50) { 8 } else { 16 })
        .unwrap()
        .with_admission(admission)
        .with_prefill_budget(policy)
        .with_last_page_rewrites(rewrites)
        .with_anchor_items(rng.below(2) as usize)
        .with_media_pages(4);
    if granule > 1 {
        config = config.with_prefill_granule(granule).unwrap();
    }
    if prefix {
        config = config.with_prefix_cache(1);
    }
    let mut s = Scheduler::new(config, KvGeometry::new(BLOCK, pages).unwrap()).unwrap();
    let mut next_id = 1u64;
    let mut stops: BTreeMap<u64, (u64, usize, usize)> = BTreeMap::new(); // id -> (seed, prompt, stop_after)
    let mut emitted_total: BTreeMap<u64, usize> = BTreeMap::new();
    let mut media: BTreeMap<u64, usize> = BTreeMap::new(); // id -> media pages
    let cap = BLOCK * pages;
    let mut idle_steps = 0;
    for step in 0..3000 {
        // submit
        let mut group = Vec::new();
        let group_size = if rng.chance(20) { 2 + rng.below(3) } else { 1 };
        for _ in 0..group_size {
            if step < 1500 && rng.chance(if group_size > 1 { 100 } else { 30 }) {
                let prompt = 1 + rng.below((cap as u64 / 2).max(2)) as usize;
                let max_new = 1 + rng.below(30) as usize;
                if prompt + max_new - 1 <= cap {
                    let id = next_id;
                    next_id += 1;
                    let seed = rng.below(1000);
                    let spec = RequestSpec::new(RequestId::new(id), prompt, max_new).unwrap();
                    let spec = if rng.chance(10) {
                        spec.with_reserved_admission()
                    } else {
                        spec
                    };
                    let media_pages = if rng.chance(10) {
                        1 + rng.below(4) as usize
                    } else {
                        0
                    };
                    let spec = if media_pages > 0 {
                        spec.with_media_pages(media_pages)
                    } else {
                        spec
                    };
                    let spec = if rng.chance(5) {
                        spec.with_prompt_scoring()
                    } else {
                        spec
                    };
                    let stop_after = 1 + rng.below(max_new as u64) as usize;
                    match s.submit_with_prefix(spec, prefix.then(|| chain(seed, prompt))) {
                        Ok(()) => {
                            stops.insert(id, (seed, prompt, stop_after));
                            media.insert(id, media_pages);
                            group.push(RequestId::new(id));
                        }
                        Err(SubmitError::Overloaded(_)) => {}
                        Err(e) => panic!("case {case}: submit {e:?}"),
                    }
                }
            }
        }
        if group.len() > 1 {
            s.share_prompt_prefixes(&group)
                .unwrap_or_else(|e| panic!("case {case}: share {e:?}"));
        }
        // cancel
        if rng.chance(5) {
            let live: Vec<u64> = stops
                .keys()
                .copied()
                .filter(|id| {
                    s.request_state(RequestId::new(*id))
                        .is_ok_and(|st| !st.is_terminal())
                })
                .collect();
            if !live.is_empty() {
                let id = live[rng.below(live.len() as u64) as usize];
                STATS.with(|x| x.borrow_mut()[2] += 1);
                s.cancel(RequestId::new(id))
                    .unwrap_or_else(|e| panic!("case {case}: cancel {e:?}"));
                s.audit().unwrap_or_else(|e| {
                    panic!("case {case} step {step}: audit after cancel {e:?}")
                });
            }
        }
        let st2 = stops.clone();
        let preempted = s
            .preempt_for_next_step(drafts, |id, len| {
                let (seed, prompt, _) = st2[&id.get()];
                // history = prompt tokens then generated tokens (deterministic per id)
                let mut t = toks(seed, prompt);
                t.extend((0..len - prompt).map(|i| 900_000 + id.get() as u32 * 100 + i as u32));
                Some(PrefixRequestV1::new(
                    Arc::new(PrefixChainV1::new(BLOCK, Arc::from(t), Vec::new()).unwrap()),
                    true,
                ))
            })
            .unwrap_or_else(|e| panic!("case {case} step {step}: preempt {e:?}"));
        STATS.with(|x| x.borrow_mut()[0] += preempted.len());
        for c in s.take_prefix_captures() {
            match s.begin_prefix_capture(&c) {
                Ok(Some(_)) if rng.chance(10) => {
                    s.abandon_prefix_capture(&c).unwrap();
                }
                Ok(_) => {}
                Err(e) => panic!("case {case}: begin capture {e:?}"),
            }
            s.end_prefix_capture(&c)
                .unwrap_or_else(|e| panic!("case {case}: end capture {e:?}"));
        }
        s.audit()
            .unwrap_or_else(|e| panic!("case {case} step {step}: audit after preempt {e:?}"));
        loop {
            let rep = s
                .admit_with_lookahead(drafts)
                .unwrap_or_else(|e| panic!("case {case}: admit {e:?}"));
            let restores = s.pending_prefix_restores();
            for r in &restores {
                if rng.chance(15) {
                    s.abandon_prefix_restore(r.request_id()).unwrap();
                } else {
                    STATS.with(|x| x.borrow_mut()[1] += 1);
                    s.complete_prefix_restore(r.request_id())
                        .unwrap_or_else(|e| panic!("case {case}: complete restore {e:?}"));
                }
            }
            s.audit()
                .unwrap_or_else(|e| panic!("case {case} step {step}: audit after admit {e:?}"));
            if rep.admitted().is_empty() {
                break;
            }
        }
        let limits = if drafts == 0 {
            DraftLimits::NONE
        } else {
            DraftLimits::new(drafts, 16)
        };
        let mut r2 = Rng(rng.next() | 1);
        // A media request's items span its whole prompt, so every chunk and
        // the start gate see its step demand.
        let chunk_media = |id: RequestId, _: usize, _: usize| ChunkMedia {
            encode_cost: 0,
            pages: media.get(&id.get()).copied().unwrap_or(0),
        };
        let batch = s.next_batch_planned(limits, chunk_media, |cands, budget| {
            let mut spare = budget.spare_draft_tokens().min(budget.spare_rows());
            cands
                .iter()
                .map(|c| {
                    let d = (r2.below(c.maximum_draft_tokens() as u64 + 1) as usize).min(spare);
                    spare -= d;
                    d
                })
                .collect()
        });
        let batch = match batch {
            Ok(Some(b)) => {
                idle_steps = 0;
                b
            }
            Ok(None) => {
                idle_steps += 1;
                if step > 1500 && idle_steps > 3 {
                    break;
                }
                continue;
            }
            Err(e) => panic!(
                "case {case} step {step}: next_batch {e:?} prefix={prefix} seqs={seqs} pages={pages} live={}",
                stops.len()
            ),
        };
        if rng.chance(5) {
            s.rollback_batch(batch.id()).unwrap();
            s.audit()
                .unwrap_or_else(|e| panic!("case {case} step {step}: audit after rollback {e:?}"));
            continue;
        }
        let mut outcomes = Vec::new();
        for item in batch.items() {
            let id = item.request_id();
            let em = s.emitted_tokens(id).unwrap();
            let (_, _, stop_after) = stops[&id.get()];
            let count = match item.phase() {
                _ if !item.expects_output() => 0,
                BatchPhase::Verify { draft_tokens } => {
                    ({
                        STATS.with(|x| x.borrow_mut()[3] += 1);
                        1
                    }) * (1 + r2.below(draft_tokens as u64 + 1) as usize)
                        .min(stop_after.saturating_sub(em).max(1))
                }
                _ => 1,
            };
            let terminal = count > 0 && em + count >= stop_after;
            outcomes.push(BatchOutcome::success(id, item.phase(), count, terminal));
            *emitted_total.entry(id.get()).or_default() += count;
        }
        let shape = batch.shape();
        s.observe_batch_time(
            shape,
            1000 + (shape.prefill_tokens() as u64) * (50 + r2.below(100))
                + shape.decode_items() as u64 * 300
                + r2.below(5000),
        );
        s.observe_step_time(
            shape.prefill_tokens(),
            shape.decode_items(),
            1000 + (shape.prefill_tokens() as u64) * 70 + r2.below(3000),
        );
        s.commit_batch(batch.id(), &outcomes)
            .unwrap_or_else(|e| panic!("case {case} step {step}: commit {e:?}"));
        for c in s.take_prefix_captures() {
            match s.begin_prefix_capture(&c) {
                Ok(Some(_)) if rng.chance(10) => {
                    s.abandon_prefix_capture(&c).unwrap();
                }
                Ok(_) => {}
                Err(e) => panic!("case {case}: begin capture2 {e:?}"),
            }
            s.end_prefix_capture(&c).unwrap();
        }
        s.audit()
            .unwrap_or_else(|e| panic!("case {case} step {step}: audit after commit {e:?}"));
        assert!(
            s.kv_stats().used_blocks() <= s.reserved_pages(),
            "case {case} step {step}: used {:?} > reserved {}",
            s.kv_stats(),
            s.reserved_pages()
        );
        if rng.chance(20) {
            s.reap_terminal(4).unwrap();
        }
    }
    // all must finish
    for id in stops.keys() {
        if let Ok(st) = s.request_state(RequestId::new(*id)) {
            assert!(
                st.is_terminal(),
                "case {case}: request {id} stuck in {st:?}"
            );
        }
    }
    let stats = s.kv_stats();
    assert_eq!(stats.active_sequences(), 0, "case {case}");
    assert_eq!(s.reserved_pages(), 0, "case {case}");
}

#[test]
fn randomized_scheduler_interleavings() {
    let n: u64 = std::env::var("CHOPINFER_SCHEDULER_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let start: u64 = std::env::var("CHOPINFER_SCHEDULER_FUZZ_START")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    for case in start..start + n {
        run_case(case);
    }
    STATS.with(|x| eprintln!("STATS preempt/restore/cancel/verify {:?}", x.borrow()));
}
