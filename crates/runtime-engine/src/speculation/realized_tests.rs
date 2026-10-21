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

struct Replay {
    policy: SpeculationPolicy,
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
        let mut policy = SpeculationPolicy::new(
            SpeculationConfig::new(5, 32, SpeculationMode::Adaptive, DraftSource::Bundle)
                .unwrap()
                .with_draft_cost_model(DraftCostModel::PerBlock),
        )
        .with_step_rows(|rows| rows)
        .with_diagnostics(true);
        for micros in AR {
            policy.observe_step(&[0], 0, micros);
        }
        Self {
            policy,
            ar: 0,
            verified_last: false,
            reasons: Vec::new(),
        }
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
            let drafts = self.policy.plan(&[candidate], DraftBudget::new(1, 31))[0];
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
                let micros = AR[self.ar % AR.len()];
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
