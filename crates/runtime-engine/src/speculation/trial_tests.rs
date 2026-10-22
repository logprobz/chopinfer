//! Bounded drafting trials. Where drafting is off and the estimates price it
//! near AR, its own steady steps decide. Replays of DSpark's served
//! C8 and C16 steps, where one server process kept AR for whole cohorts
//! while another, whose estimate happened to clear the resume margin, drafted
//! the same cohorts faster.
use super::*;

/// C8 AR steps at 300 W, in microseconds: the intervals between the step
/// completions of the C8 priming cohort that drafted (OVG power repeat run).
const C8_AR: [u64; 40] = [
    15624, 16009, 16309, 16206, 16420, 16457, 16450, 16677, 16694, 16616, 16586, 16583, 16431,
    16647, 16960, 16858, 16992, 16894, 16842, 16759, 16810, 16614, 16673, 16604, 16699, 16462,
    16554, 16490, 16603, 16512, 16575, 16358, 16677, 16632, 16697, 16830, 16674, 16611, 16679,
    16559,
];
/// The same process's timed C8 cohort: six to eight drafters at 32 rows,
/// (accepted of 24 drafts, micros). They realized 1.2 times AR.
const C8_STEADY: [(usize, u64); 49] = [
    (12, 36255),
    (9, 34856),
    (18, 35164),
    (14, 35968),
    (9, 37155),
    (12, 36068),
    (17, 36769),
    (15, 36327),
    (12, 36935),
    (11, 36997),
    (14, 37167),
    (11, 37195),
    (7, 37579),
    (7, 37175),
    (12, 37347),
    (12, 37757),
    (11, 37805),
    (10, 37742),
    (17, 45989),
    (8, 37355),
    (17, 37030),
    (12, 35671),
    (10, 35795),
    (14, 35788),
    (14, 35518),
    (17, 35033),
    (9, 35764),
    (18, 35913),
    (15, 36496),
    (14, 36132),
    (15, 35773),
    (10, 35875),
    (16, 36010),
    (21, 36385),
    (17, 36048),
    (15, 36609),
    (12, 36188),
    (16, 36171),
    (15, 36134),
    (10, 35865),
    (10, 35664),
    (19, 35433),
    (21, 35940),
    (14, 36123),
    (19, 36394),
    (14, 36879),
    (15, 36154),
    (9, 35771),
    (14, 36077),
];
/// The probes of the process that kept AR on that cohort: three drafters,
/// (accepted of 12 or 13 drafts, micros).
const C8_PROBES: [(usize, u64); 3] = [(4, 38021), (4, 35156), (3, 36663)];

/// C16 at 250 W (round 10 batch 5): one 28 ms AR step; probes of eight
/// fresh drafters that accepted 12, 8 and 7 of 16 drafts; and four chosen
/// drafters' steady 32-row steps, which realized 593 tok/s against AR's 571.
const C16_AR: u64 = 28_000;
const C16_PROBES: [(usize, u64); 3] = [(12, 44_700), (8, 52_800), (7, 48_000)];
const C16_STEADY: [(usize, u64); 4] = [(12, 45_044), (10, 43_395), (12, 43_534), (8, 47_224)];

/// A DSpark-like policy at `items` items: K5 blocks, a 32-row verify cap,
/// exact rows, an AR baseline from `ar`, the 32-row block as earlier probes
/// left it (`block` microseconds with `sequences` drafting sequences), and a
/// shared prior with these per-position acceptances, learned long enough to
/// move slowly.
fn dspark(items: usize, ar: &[u64], block: u64, sequences: usize, prior: [f64; 5]) -> Replay {
    let mut policy = SpeculationPolicy::new(
        SpeculationConfig::new(5, 32, SpeculationMode::Adaptive, DraftSource::Bundle)
            .unwrap()
            .with_draft_cost_model(DraftCostModel::PerBlock),
    )
    .with_step_rows(|rows| rows)
    .with_diagnostics(true);
    for &micros in ar {
        policy.observe_step(&vec![0; items], 0, micros);
    }
    // The block's 32 - items drafts, spread over its sequences.
    let mut drafts = vec![0; items];
    for draft in 0..32 - items {
        drafts[draft % sequences] += 1;
    }
    for _ in 0..2 {
        policy.observe_step(&drafts, 0, block);
    }
    policy.prior.positions[..5].copy_from_slice(&prior);
    policy.prior_observations = [64; MAX_DRAFT_TOKENS];
    Replay {
        policy,
        items,
        ar: ar.to_vec(),
        steps: [0; 3],
    }
}

struct Replay {
    policy: SpeculationPolicy,
    items: usize,
    ar: Vec<u64>,
    /// AR, probe and steady steps taken so far.
    steps: [usize; 3],
}

/// What a cohort served: plans, drafting plans, tokens, time and reasons.
#[derive(Debug, Default)]
struct Served {
    plans: usize,
    drafting: usize,
    first_draft: Option<usize>,
    tokens: usize,
    micros: u64,
    reasons: BTreeMap<String, usize>,
}

impl Served {
    fn rate(&self) -> f64 {
        self.tokens as f64 * 1e6 / self.micros as f64
    }
}

impl Replay {
    /// The AR rate of the baseline steps, in tokens per second.
    fn ar_rate(&self) -> f64 {
        self.items as f64 * 1e6 * self.ar.len() as f64 / self.ar.iter().sum::<u64>() as f64
    }

    /// The trace of the next plan for new requests at the prior, unplanned.
    fn priced(&self) -> (f64, f64) {
        let mut policy = self.policy.clone();
        let prior = policy.prior();
        let candidates = vec![
            PlanCandidate {
                maximum_draft_tokens: 5,
                acceptance: &prior,
                fresh: true,
                remaining_tokens: 192,
            };
            self.items
        ];
        policy.plan(
            &candidates,
            DraftBudget::new(32 - self.items, 32 - self.items),
        );
        let trace = policy.decision().unwrap().record();
        let expected = trace["candidate_expected_tokens"].as_f64().unwrap();
        let cost = trace["candidate_estimate_us"].as_f64().unwrap();
        (expected * 1e6 / cost, self.ar_rate())
    }

    /// Serves `plans` plans of a cohort of new requests, which start at the
    /// shared prior. A probe takes the next of `probes`, a steady drafting
    /// step the next of `steady` (accepted drafts, micros), an AR step the
    /// next AR baseline step, all cycled. The accepted drafts go to the
    /// drafting requests in order, each up to its drafts, so their own
    /// estimates leave the prior. `estimate` may replace every request's
    /// estimate before a plan.
    fn cohort(
        &mut self,
        plans: usize,
        probes: &[(usize, u64)],
        steady: &[(usize, u64)],
        mut estimate: impl FnMut(usize) -> Option<AcceptanceEstimate>,
    ) -> Served {
        let mut estimates = vec![self.policy.prior(); self.items];
        let mut verified = vec![false; self.items];
        let mut served = Served::default();
        for plan in 0..plans {
            if let Some(replaced) = estimate(plan) {
                estimates.fill(replaced);
            }
            let candidates = estimates
                .iter()
                .zip(&verified)
                .map(|(acceptance, &verified)| PlanCandidate {
                    maximum_draft_tokens: 5,
                    acceptance,
                    fresh: !verified,
                    remaining_tokens: 192,
                })
                .collect::<Vec<_>>();
            let spare = 32 - self.items;
            let drafts = self
                .policy
                .plan(&candidates, DraftBudget::new(spare, spare));
            let reason = self.policy.decision().unwrap().record()["reason"]
                .as_str()
                .unwrap()
                .to_owned();
            *served.reasons.entry(reason.clone()).or_default() += 1;
            let fresh = drafts
                .iter()
                .zip(&verified)
                .filter(|&(&count, &verified)| count > 0 && !verified)
                .count();
            let (mut accepted, micros) = if drafts.iter().all(|&count| count == 0) {
                self.steps[0] += 1;
                (0, self.ar[(self.steps[0] - 1) % self.ar.len()])
            } else if reason == "probe" {
                self.steps[1] += 1;
                probes[(self.steps[1] - 1) % probes.len()]
            } else {
                self.steps[2] += 1;
                steady[(self.steps[2] - 1) % steady.len()]
            };
            let mut emitted = self.items;
            for ((estimate, verified), &count) in
                estimates.iter_mut().zip(&mut verified).zip(&drafts)
            {
                if count > 0 {
                    let mine = accepted.min(count);
                    accepted -= mine;
                    emitted += mine;
                    self.policy.observe(estimate, count, mine);
                }
                *verified = count > 0;
            }
            self.policy.observe_step(&drafts, fresh, micros);
            served.plans += 1;
            if drafts.iter().any(|&count| count > 0) {
                served.drafting += 1;
                served.first_draft.get_or_insert(plan);
            }
            served.tokens += emitted;
            served.micros += micros;
        }
        served
    }
}

/// At 300 W one process's estimate priced a C8 drafting plan at 0.91 to
/// 0.94 of AR, from the shared prior and a 32-row block that fresh probes
/// timed at 38.7 ms, and it kept AR for the whole timed cohort. Another
/// process, whose estimate happened to clear the resume margin, drafted the
/// cohort in 35-37 ms steps at 1.2 times AR. Priced within the trial band,
/// the cohort is drafted from its first plan.
#[test]
fn a_c8_cohort_priced_under_the_resume_margin_drafts_as_a_trial() {
    let mut replay = dspark(8, &C8_AR, 38_735, 5, [0.55, 0.75, 0.8, 0.8, 0.8]);
    let (priced, ar) = replay.priced();
    let steady = C8_STEADY
        .iter()
        .map(|&(accepted, micros)| (micros, 8 + accepted));
    let realized = rate(steady);
    println!("C8_TRIAL priced={priced:.1} ar={ar:.1} realized={realized:.1}");
    assert!(
        TRIAL_BAND * ar <= priced && priced < RESUME_GAIN * ar && realized > 1.15 * ar,
        "{priced} {ar} {realized}"
    );
    let served = replay.cohort(150, &C8_PROBES, &C8_STEADY, |_| None);
    println!("C8_TRIAL {served:?} rate={:.1}", served.rate());
    assert_eq!(served.first_draft, Some(0), "{served:?}");
    assert!(served.drafting * 100 >= 95 * served.plans, "{served:?}");
    assert!(served.rate() > 1.15 * ar, "{served:?}");
    assert!(replay.policy.realized[&8].holding());
}

/// At C16 the estimates priced the chosen drafters' 32-row plan near 0.92
/// of AR after probes of fresh drafters set the block cost, so whole cohorts
/// of new requests kept AR, while their steady drafting would have beaten
/// it. Each cohort drafts from its first plan, and the verdict that drafting
/// holds carries to the next cohort.
#[test]
fn c16_cohorts_draft_after_probes_priced_the_block_high() {
    let mut replay = dspark(16, &[C16_AR; 3], 46_600, 8, [0.65, 0.85, 0.9, 0.9, 0.9]);
    let (priced, ar) = replay.priced();
    let realized = rate(
        C16_STEADY
            .iter()
            .map(|&(accepted, micros)| (micros, 16 + accepted)),
    );
    println!("C16_TRIAL priced={priced:.1} ar={ar:.1} realized={realized:.1}");
    assert!(
        TRIAL_BAND * ar <= priced && priced < RESUME_GAIN * ar && realized > ar,
        "{priced} {ar} {realized}"
    );
    for cohort in 0..3 {
        let served = replay.cohort(96, &C16_PROBES, &C16_STEADY, |_| None);
        println!(
            "C16_TRIAL cohort {cohort} {served:?} rate={:.1}",
            served.rate()
        );
        assert_eq!(served.first_draft, Some(0), "cohort {cohort}: {served:?}");
        assert_eq!(served.drafting, served.plans, "cohort {cohort}: {served:?}");
        assert!(served.rate() > ar, "cohort {cohort}: {served:?}");
    }
    assert!(replay.policy.realized[&16].holding());
}

/// A C16 process started drafting once its estimate cleared the resume
/// margin and realized 644 tok/s against AR's 569 over seven steady steps;
/// then its estimate dipped just under AR and drafting stopped, one step
/// short of a verdict. A drafting run that starts while drafting is off is
/// a trial: priced within the band, it drafts on to its verdict, which
/// keeps it drafting.
#[test]
fn a_dip_in_the_estimates_does_not_end_a_trial_short_of_its_verdict() {
    let mut replay = dspark(16, &[C16_AR; 3], 44_000, 4, [0.9; 5]);
    let estimate = |positions: f64| AcceptanceEstimate {
        unverified: false,
        positions: [positions; MAX_DRAFT_TOKENS],
    };
    // Four chosen drafters' steady steps at 644 tok/s.
    let steady = [(12, 43_500), (13, 45_000), (11, 41_900)];
    // The estimates clear the resume margin, then from the fourth plan on
    // lag the drafting they price, just under AR.
    let served = replay.cohort(64, &C16_PROBES, &steady, |plan| {
        Some(estimate(if plan < 3 { 0.9 } else { 0.74 }))
    });
    let reasons = &served.reasons;
    println!("DIP_TRIAL {served:?} rate={:.1}", served.rate());
    // The dip priced drafting under AR, and only the trial kept it on.
    assert!(reasons.contains_key("trial"), "{served:?}");
    assert_eq!(served.drafting, served.plans, "{served:?}");
    assert!(replay.policy.realized[&16].holding());
}

/// Where steady drafting loses (460 tok/s against AR's 571) but every new
/// cohort's prior prices it within the trial band, trials stop at their
/// verdict and back off: drafting takes a small share of plans, and the
/// wait between trials doubles.
#[test]
fn trials_where_drafting_loses_back_off() {
    let mut replay = dspark(16, &[C16_AR; 3], 46_600, 8, [0.65, 0.85, 0.9, 0.9, 0.9]);
    let losing = [(4, 43_500)];
    let (mut plans, mut drafting, mut tokens, mut micros) = (0, 0, 0, 0);
    for _ in 0..40 {
        let served = replay.cohort(100, &losing, &losing, |_| None);
        plans += served.plans;
        drafting += served.drafting;
        tokens += served.tokens;
        micros += served.micros;
    }
    let ratio = tokens as f64 * 1e6 / micros as f64 / replay.ar_rate();
    let state = replay.policy.zero_draft[16];
    println!("LOSING_TRIALS drafting={drafting} plans={plans} ratio={ratio:.4} {state:?}");
    assert!(drafting * 100 <= 3 * plans, "{drafting} of {plans}");
    assert!(ratio >= 0.985, "{ratio}");
    assert!(state.trial_interval >= 4 * TRIAL_INTERVAL, "{state:?}");
}

/// Tokens per second over recorded steps of (micros, tokens).
fn rate(steps: impl Iterator<Item = (u64, usize)>) -> f64 {
    let (micros, tokens) = steps.fold((0, 0), |(m, t), (micros, tokens)| (m + micros, t + tokens));
    tokens as f64 * 1e6 / micros as f64
}
