//! The C8 lock of round 10's batch 10 DSpark speed run, replayed from its
//! recorded steps. Two server processes of one install served the same
//! greedy C8 cohorts at 300 W. Process B1's first drafting steps ran 22 rows,
//! seven drafters of two drafts each, at 44.7 to 45.2 ms: no specialized
//! 22-row verify graph, and a second drafter chunk. Priced from that bucket,
//! every drafting plan cost more than AR, so B1 kept AR through its timed C8
//! window (418 tok/s). Process A1's first drafting measurement was a probe of
//! 32 rows with five drafters, 35.6 ms caught up; it drafted 32-row steps at
//! 1.4 to 1.5 times AR (574 tok/s). Same prompts, same tokens: the first
//! measured shape decided the mode. Balanced plans read no step time, so
//! both histories plan alike.
use super::*;

/// B1's C8 AR baseline, in microseconds.
const B1_AR: [u64; 2] = [17_621, 17_902];
/// B1's first drafting steps: seven drafters of two drafts each, 22 rows,
/// (accepted of 14 drafts, micros).
const B1_R22: [(usize, u64); 4] = [(8, 44_728), (9, 43_921), (13, 45_213), (7, 45_235)];
/// The AR steps B1 ran after them.
const B1_AFTER: [u64; 64] = [
    18_801, 18_487, 18_567, 18_411, 18_537, 18_722, 18_764, 18_663, 18_597, 18_702, 18_683, 18_709,
    18_587, 18_617, 18_580, 18_693, 18_878, 18_650, 18_691, 18_667, 18_685, 18_701, 18_648, 18_804,
    18_751, 18_709, 18_860, 18_699, 18_519, 18_836, 18_957, 18_794, 18_898, 18_774, 18_733, 18_878,
    18_656, 18_738, 18_665, 18_891, 18_889, 18_836, 18_743, 18_867, 18_830, 18_856, 18_698, 18_908,
    18_679, 18_889, 18_859, 18_626, 18_722, 18_751, 18_760, 18_597, 18_753, 18_708, 18_717, 18_787,
    18_902, 18_794, 18_491, 18_841,
];
/// A1's C8 AR steps up to its first probe.
const A1_AR: [u64; 17] = [
    17_524, 17_860, 18_173, 18_176, 18_393, 18_221, 18_232, 18_591, 18_826, 18_680, 18_666, 18_726,
    18_568, 18_890, 18_884, 18_791, 18_930,
];
/// A1's first probe: five drafters, 24 drafts, 32 rows, (accepted, micros).
const A1_PROBE: [(usize, u64); 3] = [(12, 41_296), (13, 35_580), (7, 35_605)];
/// A1's 32-row drafting steps after it, (accepted of 24 drafts, micros).
const A1_STEADY: [(usize, u64); 33] = [
    (7, 43_775),
    (14, 37_610),
    (10, 41_404),
    (6, 36_206),
    (10, 35_556),
    (18, 39_117),
    (13, 38_214),
    (11, 38_319),
    (15, 38_260),
    (8, 41_857),
    (11, 38_755),
    (15, 38_752),
    (17, 39_061),
    (16, 37_977),
    (14, 38_412),
    (13, 38_309),
    (14, 41_459),
    (23, 36_075),
    (13, 37_150),
    (15, 39_977),
    (17, 38_153),
    (14, 36_138),
    (15, 37_773),
    (11, 38_300),
    (17, 36_392),
    (19, 38_281),
    (22, 36_191),
    (17, 39_583),
    (12, 36_951),
    (18, 36_601),
    (11, 36_553),
    (12, 36_301),
    (20, 36_249),
];
/// Per-position acceptance of the C1 priming both processes ran first
/// (accepted at each position over accepted at the one before).
const PRIOR: [f64; 5] = [0.62, 0.62, 0.58, 0.39, 0.43];
/// A1's steady C8 shape: six drafters of four drafts, 32 rows.
const R32: [usize; 8] = [4, 4, 4, 4, 4, 4, 0, 0];

/// One C8 server process's policy and its cohort's requests.
struct Process {
    policy: SpeculationPolicy,
    estimates: Vec<AcceptanceEstimate>,
    verified: Vec<bool>,
}

impl Process {
    fn new(mode: SpeculationMode) -> Self {
        let mut policy = SpeculationPolicy::new(
            SpeculationConfig::new(5, 32, mode, DraftSource::Bundle)
                .unwrap()
                .with_draft_cost_model(DraftCostModel::PerBlock),
        )
        .with_step_rows(|rows| rows)
        .with_diagnostics(true);
        policy.prior.positions[..5].copy_from_slice(&PRIOR);
        policy.prior_observations = [64; MAX_DRAFT_TOKENS];
        let estimates = vec![policy.prior(); 8];
        Self {
            policy,
            estimates,
            verified: vec![false; 8],
        }
    }

    /// Reports a step as the engine does: each drafting request's
    /// verification, `accepted` drafts going to the drafting requests in
    /// order, then the step time.
    fn step(&mut self, drafts: &[usize], mut accepted: usize, micros: u64) {
        let fresh = drafts
            .iter()
            .zip(&self.verified)
            .filter(|&(&count, &verified)| count > 0 && !verified)
            .count();
        for ((estimate, verified), &count) in self
            .estimates
            .iter_mut()
            .zip(&mut self.verified)
            .zip(drafts)
        {
            if count > 0 {
                let mine = accepted.min(count);
                accepted -= mine;
                self.policy.observe(estimate, count, mine);
            }
            *verified = count > 0;
        }
        self.policy.observe_step(drafts, fresh, micros);
    }

    /// The policy's price of A1's steady 32-row step, caught up.
    fn r32_cost(&self) -> f64 {
        self.policy.step_cost(8, DraftShape::from_counts(&R32), 0.0)
    }

    fn plan(&mut self) -> (Vec<usize>, PlanTrace) {
        let candidates = self
            .estimates
            .iter()
            .zip(&self.verified)
            .map(|(acceptance, &verified)| PlanCandidate {
                maximum_draft_tokens: 5,
                acceptance,
                fresh: !verified,
                remaining_tokens: 192,
            })
            .collect::<Vec<_>>();
        let drafts = self.policy.plan(&candidates, DraftBudget::new(24, 24));
        (drafts, self.policy.decision().unwrap())
    }

    /// Plans `plans` steps, serving AR plans from `ar` and drafting plans
    /// from `drafting` (accepted, micros), each cycled in order. Returns the
    /// plans that drafted and every plan's trace.
    fn serve(
        &mut self,
        plans: usize,
        ar: &[u64],
        drafting: &[(usize, u64)],
    ) -> (Vec<Vec<usize>>, Vec<PlanTrace>) {
        let (mut served, mut traces) = (Vec::new(), Vec::new());
        let (mut next_ar, mut next_drafting) = (0, 0);
        for _ in 0..plans {
            let (drafts, trace) = self.plan();
            if drafts.iter().all(|&count| count == 0) {
                self.step(&drafts, 0, ar[next_ar % ar.len()]);
                next_ar += 1;
            } else {
                let (accepted, micros) = drafting[next_drafting % drafting.len()];
                self.step(&drafts, accepted, micros);
                next_drafting += 1;
                served.push(drafts);
            }
            traces.push(trace);
        }
        (served, traces)
    }

    /// B1's history: its AR baseline, then its recorded 22-row steps.
    fn b1(mode: SpeculationMode) -> Self {
        let mut process = Self::new(mode);
        for micros in B1_AR {
            process.step(&[0; 8], 0, micros);
        }
        for (accepted, micros) in B1_R22 {
            process.step(&[2, 2, 2, 2, 2, 2, 2, 0], accepted, micros);
        }
        process
    }

    /// A1's history: its AR steps, then its recorded 32-row probe.
    fn a1(mode: SpeculationMode) -> Self {
        let mut process = Self::new(mode);
        for micros in A1_AR {
            process.step(&[0; 8], 0, micros);
        }
        for (accepted, micros) in A1_PROBE {
            process.step(&[5, 5, 5, 5, 4, 0, 0, 0], accepted, micros);
        }
        process
    }
}

/// After B1's recorded 22-row steps, the policy prices every drafting plan
/// above AR and keeps AR for the 60 plans before its next probe, served at
/// B1's recorded AR times. Scaled up from the 22-row bucket, A1's steady
/// 32-row shape, which A1 ran at 35.6 to 43.8 ms, is priced above 45 ms, so
/// B1 never runs it.
#[test]
fn the_recorded_22_row_steps_lock_c8_on_ar() {
    let mut b1 = Process::b1(SpeculationMode::Adaptive);
    let (drafted, traces) = b1.serve(60, &B1_AFTER, &[]);
    assert!(drafted.is_empty(), "{drafted:?}");
    for trace in &traces {
        let record = trace.record();
        assert_eq!(record["reason"], "estimated_slower", "{record}");
    }
    assert!(b1.r32_cost() > 45_000.0, "{}", b1.r32_cost());
    assert!(!b1.policy.realized[&8].holding());
}

/// After A1's recorded 32-row probe, the same policy prices that shape
/// under 40 ms, drafts at once and holds over A1's recorded 32-row steps,
/// whose realized throughput beats AR's.
#[test]
fn the_recorded_32_row_probe_keeps_c8_drafting() {
    let mut a1 = Process::a1(SpeculationMode::Adaptive);
    assert!(a1.r32_cost() < 40_000.0, "{}", a1.r32_cost());
    let (drafted, traces) = a1.serve(33, &A1_AR, &A1_STEADY);
    assert!(drafted.len() >= 31, "{} of 33: {traces:?}", drafted.len());
    let realized = a1.policy.realized[&8];
    assert!(realized.holding(), "{realized:?}");
    assert!(
        realized.drafting.unwrap() > 1.25 * realized.ar.unwrap(),
        "{realized:?}"
    );
}

/// Balanced plans read no step time: fed either recorded history and served
/// at its own recorded times, both processes plan three drafts for each of
/// the eight requests at every step.
#[test]
fn balanced_plans_the_same_after_either_recorded_history() {
    let balanced = SpeculationMode::Balanced(DraftDepths::default());
    let mut b1 = Process::b1(balanced);
    let mut a1 = Process::a1(balanced);
    let (b1_plans, _) = b1.serve(40, &B1_AFTER, &[(14, 45_000)]);
    let (a1_plans, _) = a1.serve(40, &A1_AR, &A1_STEADY);
    assert_eq!(b1_plans, a1_plans);
    assert_eq!(b1_plans, vec![vec![3; 8]; 40]);
    for process in [&b1, &a1] {
        assert!(process.policy.costs.micros.is_empty());
        assert!(process.policy.block_steps.steps.is_empty());
        assert!(process.policy.realized.is_empty());
    }
}
