//! A hard prompt's realized drafting throughput must not keep drafting off
//! for the next request. Timings are the DSpark C1 steps of the round 9
//! exactness head, where a hard prompt kept drafting off for the next case.
use super::*;

/// One-row AR steps, in microseconds.
const AR: [u64; 24] = [
    10719, 10587, 10639, 10499, 10546, 10562, 10633, 10737, 10408, 10481, 10375, 10461, 10372,
    10457, 10382, 10453, 10349, 10450, 11857, 10447, 10363, 10451, 10366, 10440,
];

/// Five-draft verify steps of a well predicted stretch: (micros, accepted).
const EASY: [(u64, usize); 53] = [
    (16443, 1),
    (16122, 1),
    (16298, 4),
    (15820, 1),
    (16380, 3),
    (15746, 5),
    (15750, 5),
    (16232, 2),
    (15646, 5),
    (15754, 1),
    (15983, 4),
    (15714, 0),
    (15630, 5),
    (16422, 2),
    (16064, 5),
    (15976, 4),
    (16268, 1),
    (15761, 5),
    (16135, 5),
    (16052, 5),
    (15924, 5),
    (16117, 2),
    (15809, 5),
    (15727, 5),
    (16118, 5),
    (15806, 1),
    (15790, 5),
    (16200, 5),
    (15933, 5),
    (15944, 5),
    (16215, 5),
    (15605, 5),
    (16487, 2),
    (16336, 2),
    (15796, 0),
    (15850, 1),
    (15896, 5),
    (16340, 5),
    (15993, 5),
    (16364, 5),
    (16342, 5),
    (16322, 5),
    (16381, 5),
    (16317, 5),
    (16226, 5),
    (16091, 5),
    (16325, 5),
    (17000, 5),
    (16507, 1),
    (16040, 5),
    (16196, 3),
    (16207, 5),
    (16447, 5),
];

/// The stretch of the same request that followed, mostly rejected.
const HARD: [(u64, usize); 36] = [
    (16288, 1),
    (15602, 4),
    (15983, 5),
    (16000, 5),
    (16230, 4),
    (15730, 0),
    (15652, 1),
    (15726, 0),
    (15981, 0),
    (15711, 0),
    (16074, 1),
    (16714, 0),
    (15727, 0),
    (16175, 0),
    (15926, 0),
    (15990, 0),
    (16120, 2),
    (16377, 1),
    (16516, 0),
    (16379, 1),
    (16318, 1),
    (16870, 0),
    (16255, 1),
    (16085, 5),
    (15852, 0),
    (15822, 0),
    (16039, 0),
    (16197, 0),
    (15943, 1),
    (16269, 0),
    (16141, 1),
    (16028, 0),
    (15779, 0),
    (16084, 0),
    (16106, 1),
    (15692, 0),
];

// Round 9's hard prompts in realized-r1 (d2-0460z0r). The base drafted
// every step of them, at 114-119 tok/s against 94 tok/s AR.
/// diverse-02: the base's 132 verify steps (accepted of up to five drafts).
const D02_MICROS: [u64; 132] = [
    20972, 15875, 15801, 16432, 15779, 16676, 16499, 15907, 16793, 16746, 16816, 16532, 16542,
    17059, 16777, 16816, 16993, 17098, 17463, 17409, 17473, 16847, 17314, 16169, 16360, 16521,
    16932, 17077, 16802, 17389, 17255, 17031, 16831, 16631, 16914, 16739, 17008, 16772, 16842,
    16788, 16822, 15541, 15802, 15971, 16008, 17003, 17184, 16950, 16937, 17397, 17004, 16934,
    16430, 16626, 17335, 17121, 16840, 16748, 17244, 16977, 16632, 16902, 17182, 17257, 16938,
    16037, 16080, 16598, 16512, 16961, 17012, 17134, 17288, 16551, 16912, 16019, 16678, 16773,
    16622, 17045, 16844, 16830, 17037, 16846, 16165, 16969, 16681, 16743, 17280, 16770, 16234,
    16517, 16898, 16398, 16317, 16757, 16726, 16679, 17015, 16564, 16872, 16452, 16934, 16846,
    17297, 17277, 17401, 17223, 17060, 16452, 16796, 16873, 16332, 16166, 16965, 17261, 17249,
    16810, 17120, 17529, 17142, 16992, 17124, 17247, 16740, 16985, 17093, 16713, 16778, 16587,
    18435, 17779,
];
const D02_ACCEPTED: [usize; 132] = [
    1, 4, 5, 5, 4, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1, 0, 1, 1, 0, 1, 5, 0, 0, 0, 0, 1, 0, 1, 0,
    0, 0, 1, 0, 1, 1, 0, 1, 3, 5, 5, 5, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 3, 0, 1, 2, 2, 4, 5, 1, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 1, 0,
    0, 2, 0, 1, 0, 3, 0, 0, 0, 1, 0, 1, 2, 1, 0, 2, 3, 1, 0, 0, 3, 1, 1, 0, 0, 1, 0, 2, 1, 0, 3, 1,
    1, 1, 0, 1,
];
/// diverse-04: the base's 133 verify steps (accepted of up to five drafts).
const D04_MICROS: [u64; 133] = [
    20850, 16652, 16818, 16672, 16802, 16864, 17109, 16793, 17485, 16531, 16276, 16744, 16443,
    16611, 16783, 16783, 17227, 16421, 16802, 16683, 16410, 16675, 16436, 16685, 16839, 17026,
    16691, 16889, 16553, 16821, 16682, 16183, 17279, 17470, 16263, 16757, 17026, 17225, 17110,
    17248, 16429, 16663, 16342, 16677, 16170, 16711, 16830, 17059, 16491, 16665, 16833, 17027,
    16883, 16429, 16249, 16053, 18393, 16324, 16265, 17149, 16967, 16491, 17485, 16115, 17030,
    16507, 16762, 16372, 16718, 17288, 16509, 16795, 16881, 16753, 16660, 17126, 16499, 16493,
    17130, 15890, 16398, 17036, 16523, 16665, 16293, 16615, 16853, 16676, 17074, 16613, 15659,
    16191, 16297, 16676, 16486, 16805, 16471, 16953, 17081, 16713, 16948, 17154, 16794, 16545,
    17109, 17130, 16766, 17170, 16552, 16882, 17024, 16367, 16298, 16467, 16309, 16354, 16925,
    16808, 16912, 16659, 17117, 16726, 16833, 16617, 16934, 16854, 16904, 17012, 17081, 18941,
    18587, 17781, 15759,
];
const D04_ACCEPTED: [usize; 133] = [
    2, 0, 4, 0, 2, 0, 1, 1, 2, 0, 2, 1, 1, 0, 1, 1, 3, 0, 1, 5, 0, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 1,
    0, 0, 1, 0, 0, 0, 2, 1, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 2, 1, 1, 3, 0, 1, 0, 1, 1,
    1, 0, 0, 0, 1, 0, 0, 1, 0, 3, 1, 0, 0, 0, 4, 1, 1, 0, 0, 0, 2, 1, 0, 5, 1, 0, 1, 0, 2, 4, 0, 1,
    1, 0, 0, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 0, 4, 2, 1, 0, 1, 1, 1, 1, 0, 1, 4, 2, 0, 0, 1, 0, 2,
    3, 0, 0, 0, 1,
];
/// diverse-10: the base's 130 verify steps (accepted of up to five drafts).
const D10_MICROS: [u64; 130] = [
    20537, 16347, 15771, 15999, 15798, 16102, 16504, 17072, 16129, 16862, 16518, 16609, 16611,
    16737, 16752, 16735, 16556, 16108, 15870, 16450, 16838, 16677, 16558, 16729, 16652, 16710,
    17124, 16841, 17227, 18614, 16892, 16979, 17075, 16984, 16800, 16937, 17204, 16881, 16843,
    16533, 16140, 15600, 16245, 15904, 16152, 16484, 16907, 16553, 16581, 16334, 16396, 16508,
    16608, 16088, 16694, 16922, 16644, 16502, 16525, 16861, 16853, 16687, 16917, 16593, 15905,
    15744, 15580, 15795, 15883, 16541, 16153, 16237, 16490, 15983, 16436, 16291, 16437, 16117,
    16333, 16181, 15764, 16429, 16853, 16689, 16357, 16316, 16496, 16368, 16161, 16046, 16718,
    16344, 16051, 16579, 15861, 16563, 16647, 15884, 16143, 16348, 16844, 16459, 16354, 16428,
    16439, 16786, 16421, 16503, 16534, 16659, 16101, 16476, 16430, 16137, 16256, 16616, 15994,
    16942, 15903, 16323, 16431, 16349, 16800, 16116, 16409, 16207, 18905, 18575, 17910, 15798,
];
const D10_ACCEPTED: [usize; 130] = [
    0, 5, 5, 5, 5, 4, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 1,
    0, 1, 0, 0, 1, 0, 3, 1, 4, 5, 1, 5, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1,
    5, 5, 5, 5, 3, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 1, 0, 0, 2, 0, 1, 3, 0, 3, 2, 3, 1, 0, 3, 0, 0, 0, 0,
    0, 1,
];
/// Steady one-row AR steps of the candidate in those cases.
const HARD_AR: [u64; 48] = [
    11915, 10682, 10585, 10706, 10594, 10661, 10567, 10669, 10581, 10663, 10615, 10672, 10520,
    10564, 10583, 10659, 10566, 10656, 10527, 10651, 10545, 10657, 10634, 10735, 10615, 10723,
    10600, 10642, 10676, 10777, 10685, 10706, 10673, 10711, 10560, 10786, 10643, 10692, 10524,
    10533, 10623, 10696, 10554, 10670, 10581, 10669, 10591, 10656,
];

struct Replay {
    policy: SpeculationPolicy,
    ar_steps: &'static [u64],
    ar: usize,
    verified_last: bool,
    reasons: Vec<String>,
}

/// What one request did: its plans, drafting steps, tokens and time.
#[derive(Debug, Default)]
struct Outcome {
    plans: usize,
    first_draft: Option<usize>,
    drafting_steps: usize,
    tokens: usize,
    micros: u64,
}

impl Replay {
    fn new() -> Self {
        Self::with_ar(&AR)
    }

    /// A policy whose one-row AR baseline is warmed with `ar_steps`, which
    /// also time the AR steps it plans.
    fn with_ar(ar_steps: &'static [u64]) -> Self {
        let mut policy = SpeculationPolicy::new(
            SpeculationConfig::new(5, 32, SpeculationMode::Adaptive, DraftSource::Bundle)
                .unwrap()
                .with_draft_cost_model(DraftCostModel::PerBlock),
        )
        .with_step_rows(|rows| rows)
        .with_diagnostics(true);
        for &micros in ar_steps {
            policy.observe_step(&[0], 0, micros);
        }
        Self {
            policy,
            ar_steps,
            ar: 0,
            verified_last: false,
            reasons: Vec::new(),
        }
    }

    /// Measures every one-row drafting width at `micros`, as serving's
    /// warmup captures every verify width before requests arrive: at C1 a
    /// one- and a five-draft step cost about the same.
    fn warm_widths(mut self, micros: u64) -> Self {
        for drafts in 1..=5 {
            for _ in 0..3 {
                self.policy.observe_step(&[drafts], 0, micros);
            }
        }
        self
    }

    /// Runs one C1 request for `tokens` output tokens under the policy's
    /// own plans. A drafting step takes the next recorded verify step and
    /// its acceptance, capped at the planned drafts; an AR step the next
    /// recorded AR time.
    fn request(
        &mut self,
        mut samples: impl Iterator<Item = (u64, usize)>,
        tokens: usize,
    ) -> Outcome {
        let mut estimate = self.policy.prior();
        let mut outcome = Outcome::default();
        while outcome.tokens < tokens {
            let candidate = PlanCandidate {
                maximum_draft_tokens: 5,
                acceptance: &estimate,
                fresh: !self.verified_last,
                remaining_tokens: tokens - outcome.tokens,
            };
            let drafts = self.policy.plan(&[candidate], DraftBudget::new(31, 31))[0];
            let reason = self.policy.decision().unwrap().record()["reason"]
                .as_str()
                .unwrap()
                .to_owned();
            self.reasons.push(reason);
            let sample = (drafts > 0).then(|| samples.next()).flatten();
            let (micros, emitted) = if let Some((micros, accepted)) = sample {
                let accepted = accepted.min(drafts);
                outcome.first_draft.get_or_insert(outcome.plans);
                outcome.drafting_steps += 1;
                self.policy.observe(&mut estimate, drafts, accepted);
                self.policy
                    .observe_step(&[drafts], usize::from(!self.verified_last), micros);
                self.verified_last = true;
                (micros, 1 + accepted)
            } else {
                // A plan past the recorded steps runs as AR.
                let micros = self.ar_steps[self.ar % self.ar_steps.len()];
                self.ar += 1;
                self.policy.observe_step(&[0], 0, micros);
                self.verified_last = false;
                (micros, 1)
            };
            outcome.plans += 1;
            outcome.tokens += emitted;
            outcome.micros += micros;
        }
        outcome
    }

    fn count(&self, reason: &str) -> usize {
        self.reasons.iter().filter(|r| *r == reason).count()
    }
}

/// Tokens per second over recorded steps.
fn rate(steps: impl Iterator<Item = (u64, usize)>) -> f64 {
    let (micros, tokens) = steps.fold((0, 0), |(m, t), (micros, tokens)| (m + micros, t + tokens));
    tokens as f64 * 1e6 / micros as f64
}

#[test]
fn a_hard_prompt_does_not_keep_drafting_off_for_the_next_request() {
    let ar_rate = rate(AR.iter().map(|&micros| (micros, 1)));
    let steps = |samples: &[(u64, usize)]| {
        rate(
            samples
                .iter()
                .map(|&(micros, accepted)| (micros, 1 + accepted)),
        )
    };
    let worst = HARD.windows(8).map(steps).fold(f64::MAX, f64::min);
    let request: Vec<_> = EASY.iter().chain(&HARD).copied().collect();
    // About 95 tok/s AR. Drafting realized about 227 tok/s over the
    // recorded request, but its worst eight steps in a row realize 70.
    assert!(worst < ar_rate && 1.5 * ar_rate < steps(&request));

    let mut replay = Replay::new();
    // The request continues as hard as its last recorded steps, so the
    // probes it gets keep losing, as the round 9 head's did.
    let continuation = HARD[HARD.len() - 12..].iter().copied().cycle();
    let first = replay.request(request.iter().copied().chain(continuation), 900);
    let slower = replay.count("realized_slower");
    let probes = replay.count("probe");
    let interval = replay.policy.zero_draft[1].probe_interval;
    println!("REALIZED_REPLAY first={first:?} slower={slower} probes={probes} interval={interval}");
    // The rejected run pulls the realized drafting average below AR and
    // drafting stops; losing probes back off.
    assert!(slower > 0, "{first:?} {:?}", replay.reasons);
    assert!(first.drafting_steps >= EASY.len(), "{first:?}");
    assert!(
        probes >= 2 && interval > 2 * PROBE_INTERVAL,
        "{probes} {interval}"
    );

    // The next request is as easy as the first one's start. Its own
    // estimate decides: it drafts from its first plan, every plan, well
    // above AR, as the round 9 base did.
    let next = replay.request(EASY.iter().copied(), 90);
    let next_rate = next.tokens as f64 * 1e6 / next.micros as f64;
    println!(
        "REALIZED_REPLAY_OK ar={ar_rate:.1} worst={worst:.1} request={:.1} next={next:?} \
         next_rate={next_rate:.1}",
        steps(&request)
    );
    assert_eq!(next.first_draft, Some(0), "{next:?}");
    assert_eq!(next.drafting_steps, next.plans, "{next:?}");
    assert!(next_rate > 1.2 * ar_rate, "{next_rate} against {ar_rate}");
}

/// Drafting at round 9's hard prompts still beats AR over each case, though
/// runs of rejected steps fall well below it. Drafting stays on throughout.
#[test]
fn hard_prompts_that_still_beat_ar_keep_drafting() {
    let cases: [(&str, &[u64], &[usize]); 3] = [
        ("diverse-02", &D02_MICROS, &D02_ACCEPTED),
        ("diverse-04", &D04_MICROS, &D04_ACCEPTED),
        ("diverse-10", &D10_MICROS, &D10_ACCEPTED),
    ];
    let ar_rate = rate(HARD_AR.iter().map(|&micros| (micros, 1)));
    for (name, micros, accepted) in cases {
        let steps: Vec<_> = micros
            .iter()
            .copied()
            .zip(accepted.iter().copied())
            .collect();
        let drafting = rate(
            steps
                .iter()
                .map(|&(micros, accepted)| (micros, 1 + accepted)),
        );
        assert!(
            drafting > 1.2 * ar_rate,
            "{name}: {drafting} against {ar_rate}"
        );
        // As served, easier requests come first and train the prior and
        // the block costs.
        let mut replay = Replay::with_ar(&HARD_AR).warm_widths(16_000);
        replay.request(EASY.iter().copied(), 150);
        let before = replay.reasons.len();
        let outcome = replay.request(steps.iter().copied().cycle(), 255);
        replay.reasons.drain(..before);
        let served = outcome.tokens as f64 * 1e6 / outcome.micros as f64;
        println!(
            "HARD_REPLAY {name} ar={ar_rate:.1} drafting={drafting:.1} served={served:.1} \
             {outcome:?} slower={}",
            replay.count("realized_slower")
        );
        assert_eq!(
            replay.count("realized_slower"),
            0,
            "{name}: {:?}",
            replay.reasons
        );
        assert_eq!(
            outcome.drafting_steps, outcome.plans,
            "{name}: {:?}",
            replay.reasons
        );
        assert!(
            served > 1.15 * ar_rate,
            "{name}: {served} against {ar_rate}"
        );
    }
}

/// A request whose drafts are all rejected realizes 62 tok/s against 94
/// tok/s AR: drafting stops within a few plans and stays mostly off.
#[test]
fn a_request_whose_drafting_loses_switches_it_off() {
    let mut replay = Replay::with_ar(&HARD_AR).warm_widths(16_000);
    let outcome = replay.request(std::iter::repeat((16_000, 0)), 400);
    println!("LOSING_REPLAY {outcome:?} {:?}", &replay.reasons[..24]);
    let first_off = replay
        .reasons
        .iter()
        .position(|reason| reason != "drafting");
    assert!(
        first_off.is_some_and(|plan| plan <= 16),
        "{:?}",
        replay.reasons
    );
    assert!(
        outcome.drafting_steps * 5 <= outcome.plans,
        "{outcome:?} {:?}",
        replay.reasons
    );
}

/// The drafting average weighs its first steps evenly, so one step does not
/// set it, and drafting loses once that average falls below AR.
#[test]
fn realized_drafting_averages_its_first_steps_evenly() {
    let mut realized = Realized {
        ar: Some(1.0),
        ..Realized::default()
    };
    realized.record_drafting(2.0, Realized::draft_weight(1));
    realized.record_drafting(0.0, Realized::draft_weight(1));
    assert!((realized.drafting.unwrap() - 2.0 * 31.0 / 63.0).abs() < 1e-12);
    let steady = |rate: f64| {
        let mut realized = Realized {
            ar: Some(1.0),
            ..Realized::default()
        };
        for _ in 0..REALIZED_STEPS {
            realized.record_drafting(rate, Realized::draft_weight(1));
        }
        realized.losing()
    };
    assert!(steady(0.97));
    assert!(!steady(1.01));
    // A step averages one request per item: 32 request-steps, at least 8.
    let weights = [1, 2, 4, 32].map(Realized::draft_weight);
    assert_eq!(weights, [1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 8.0]);
}

// C16 as one DeepSeek DSpark process, B1, served it in a C16 priming phase:
// 16 items, a 28.0 ms AR step, and drafting by the four requests the
// estimates chose, four drafts each in a 32-row step.
const C16_AR: u64 = 28_000;
/// B1's probes before it drafted: (drafting items, drafts each, accepted,
/// micros). Probes that ran 14 drafts are rounded up to three of five.
const C16_PROBES: [(usize, usize, usize, u64); 7] = [
    (4, 4, 7, 41_961),
    (4, 4, 13, 40_115),
    (4, 4, 3, 40_849),
    (3, 5, 9, 51_405),
    (3, 5, 10, 50_541),
    (3, 5, 12, 50_666),
    (3, 5, 9, 49_352),
];
/// B1's steady drafting steps that followed: (accepted of 16, micros). They
/// realized 593 tok/s against AR's 571.
const C16_STEADY: [(usize, u64); 4] = [(12, 45_044), (10, 43_395), (12, 43_534), (8, 47_224)];

fn c16_drafts(drafting: usize, each: usize) -> [usize; 16] {
    let mut drafts = [0; 16];
    drafts[..drafting].fill(each);
    drafts
}

/// A C16 policy with B1's AR baseline and both probe shapes warmed, as
/// serving's warmup runs every verify graph before requests arrive.
fn c16_policy() -> SpeculationPolicy {
    let mut policy = SpeculationPolicy::new(
        SpeculationConfig::new(5, 32, SpeculationMode::Adaptive, DraftSource::Bundle)
            .unwrap()
            .with_draft_cost_model(DraftCostModel::PerBlock),
    )
    .with_step_rows(|rows| rows)
    .with_diagnostics(true);
    for _ in 0..3 {
        policy.observe_step(&[0; 16], 0, C16_AR);
    }
    policy.observe_step(&c16_drafts(4, 4), 0, 41_000);
    policy.observe_step(&c16_drafts(3, 5), 0, 50_000);
    policy
}

/// Plans `plans` C16 steps. Four requests may draft, and their estimates
/// stay at 0.85 per position, which prices four drafts each above AR. A
/// probe step is charged `probe`, a steady drafting step the next of
/// `steady`, an AR step B1's AR time. Returns the reasons and drafting
/// steps.
fn serve_c16(
    policy: &mut SpeculationPolicy,
    plans: usize,
    probe: (usize, u64),
    mut steady: impl Iterator<Item = (usize, u64)>,
) -> (Vec<String>, usize) {
    let good = AcceptanceEstimate {
        unverified: false,
        positions: [0.85; MAX_DRAFT_TOKENS],
    };
    let candidates: Vec<_> = (0..16)
        .map(|item| PlanCandidate {
            maximum_draft_tokens: if item < 4 { 5 } else { 0 },
            acceptance: &good,
            fresh: false,
            remaining_tokens: usize::MAX,
        })
        .collect();
    let mut reasons = Vec::new();
    let mut drafting = 0;
    for _ in 0..plans {
        let drafts = policy.plan(&candidates, DraftBudget::new(16, 16));
        let reason = policy.decision().unwrap().record()["reason"]
            .as_str()
            .unwrap()
            .to_owned();
        let count = drafts.iter().sum::<usize>();
        let micros = if count == 0 {
            C16_AR
        } else {
            drafting += 1;
            let (accepted, micros) = if reason == "probe" {
                probe
            } else {
                steady.next().unwrap()
            };
            policy.step_accepted = accepted.min(count);
            micros
        };
        policy.observe_step(&drafts, 0, micros);
        reasons.push(reason);
    }
    (reasons, drafting)
}

/// B1 started drafting at C16 once its estimate cleared the resume margin,
/// and its steady steps beat AR, but its seven earlier probes, which lost,
/// outvoted them: `realized_slower` stopped drafting after four steady
/// steps, and C16 stayed at AR. Probes no longer count as steps, and the
/// drafting run clears an average of probes alone, so the steady steps
/// decide and drafting holds.
#[test]
fn probe_losses_followed_by_profitable_steady_steps_keep_drafting() {
    let ar = 16.0 * 1e6 / C16_AR as f64;
    let steady = rate(
        C16_STEADY
            .iter()
            .map(|&(accepted, micros)| (micros, 16 + accepted)),
    );
    let probes = rate(
        C16_PROBES
            .iter()
            .map(|&(_, _, accepted, micros)| (micros, 16 + accepted)),
    );
    assert!(probes < ar && ar < steady, "{probes} {ar} {steady}");

    let mut policy = c16_policy();
    for (drafting, each, accepted, micros) in C16_PROBES {
        policy.probing = Some(16);
        policy.step_accepted = accepted;
        policy.observe_step(&c16_drafts(drafting, each), 0, micros);
    }
    let (reasons, drafting) =
        serve_c16(&mut policy, 64, (0, 0), C16_STEADY.iter().copied().cycle());
    println!("C16_PROBE_LOSSES drafting={drafting} {reasons:?}");
    assert!(
        !reasons.iter().any(|r| r == "realized_slower"),
        "{reasons:?}"
    );
    assert_eq!(drafting, reasons.len(), "{reasons:?}");
    assert!(policy.realized[&16].holding(), "{:?}", policy.realized[&16]);
}
