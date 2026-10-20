//! Draft-count policy for speculative decoding.
//!
//! The engine asks the policy how many drafts each decode item of a forming
//! batch should verify. Acceptance itself does not live here: it compares
//! drafts with the target's own samples, so no policy choice can change which
//! tokens a request emits, only how many it emits per step.

use std::collections::{BTreeMap, BTreeSet};

use runtime_scheduler::{DraftBudget, DraftLimits};

mod balanced;
mod diagnostics;
pub(crate) use diagnostics::PlanTrace;
pub mod confidence;
pub mod table;

pub use table::{StepCost, StepCosts, StepCostsError};

/// The most drafts one item may verify in a step.
pub const MAX_DRAFT_TOKENS: usize = 16;

/// Exponential weight of one observation in a request's acceptance estimate.
const REQUEST_WEIGHT: f64 = 1.0 / 8.0;
/// Exponential weight of one observation in the shared prior.
const PRIOR_WEIGHT: f64 = 1.0 / 64.0;
/// Acceptance assumed at every position before anything is observed.
const INITIAL_ACCEPTANCE: f64 = 0.6;
/// Exponential weight of one measured step in the cost table.
const COST_WEIGHT: f64 = 1.0 / 4.0;
/// Relative cost of one more decode-phase row, used between measurements.
const PRIOR_ROW_COST: f64 = 0.1;
/// Rows the cost table records; larger steps use the nearest measurement.
const MAX_COST_ROWS: usize = 1024;
/// After its realized throughput lost to AR or a probe, reconsider a
/// concurrency once every eight eligible plans.
const ZERO_DRAFT_RECHECK: u8 = 7;
/// Resume only when the estimated throughput gain clears timing noise.
const RESUME_GAIN: f64 = 1.05;
/// One drafting plan in this many verifies an item to its full depth, so
/// positions the estimates would never choose are still observed.
const EXPLORE_INTERVAL: u8 = 16;
/// An unprofitable concurrency runs its best drafting plan once after this
/// many plans without drafting, so drafting and acceptance measurements that
/// no longer hold are remeasured. The interval doubles after every probe
/// that does not resume drafting, up to `MAX_PROBE_INTERVAL`.
const PROBE_INTERVAL: u16 = 64;
const MAX_PROBE_INTERVAL: u16 = 4096;
/// A probe that took the time of n AR steps is followed by at least this
/// many times n plans without one, so probes that keep losing take at most
/// about 1/64 of the time: a C32 probe that catches up 32 drafters can cost
/// ten AR steps.
const PROBE_SPACING: f64 = 64.0;
/// An unmeasured block bucket or a sparsely measured per-token shape
/// gets its first probe after this many idle plans.
const FIRST_PROBE: u16 = 16;
/// A fresh drafter pays its catch-up once and then drafts from a caught-up
/// history, so a plan charges each step this share of it: a drafting run of
/// this many steps.
const FRESH_RUN_STEPS: f64 = 16.0;
/// Drafting steps a concurrency must realize before its measured throughput
/// can switch drafting off.
const REALIZED_STEPS: u32 = 8;
/// Exponential weight of one step in a concurrency's realized throughput.
const REALIZED_WEIGHT: f64 = 1.0 / 8.0;
/// Cost of one more drafting sequence in a block step of the same row
/// bucket, relative to a one-row AR step. One batched pass drafts every
/// sequence: kernel-lab measured 1.29 ms for one K7 block and 2.3 ms for 16,
/// and a caught-up sequence adds about 0.12 ms of catch-up, together about
/// 1/20 of a 4 ms AR step per sequence.
const PRIOR_SEQUENCE_COST: f64 = 1.0 / 16.0;
/// A row bucket not measured at a concurrency is priced from its measured
/// neighbours there: a wider one at this factor per doubling of rows over
/// the nearest narrower one, a narrower one at the nearest wider one divided
/// by `NARROWER_BUCKET_RATIO` per halving. TP4 verify steps grew 1.2 to 1.7
/// times per doubling from 32 to 256 rows, so both lean high.
const WIDER_BUCKET_RATIO: f64 = 1.7;
const NARROWER_BUCKET_RATIO: f64 = 1.25;
/// The same ratios leaning low, for deciding whether an unmeasured bucket
/// might pay and so deserves a probe.
const OPTIMISTIC_WIDER_RATIO: f64 = 1.2;
const OPTIMISTIC_NARROWER_RATIO: f64 = 1.7;

#[derive(Clone, Copy, Debug, PartialEq)]
struct ZeroDraftState {
    disabled: bool,
    remaining: u8,
    /// Plans since this concurrency last drafted.
    idle: u16,
    /// Plans without drafting before the next probe.
    probe_interval: u16,
    /// AR time when the last probe chose its backoff.
    probe_ar_micros: Option<f64>,
    /// The next probe repeats the last one, so the interval stays.
    again: bool,
}

impl Default for ZeroDraftState {
    /// A concurrency starts as if drafting had lost: its first drafting plan
    /// must clear the resume margin.
    fn default() -> Self {
        Self {
            disabled: true,
            remaining: 0,
            idle: 0,
            probe_interval: PROBE_INTERVAL,
            probe_ar_micros: None,
            again: false,
        }
    }
}

/// Tokens per microsecond one concurrency realized, as exponential
/// averages over its AR steps and its drafting steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Realized {
    ar: Option<f64>,
    drafting: Option<f64>,
    drafting_steps: u32,
}

impl Realized {
    fn record(average: &mut Option<f64>, rate: f64, weight: f64) {
        *average = Some(average.map_or(rate, |previous| previous + weight * (rate - previous)));
    }

    /// Drafting at this concurrency emitted fewer tokens per unit of time
    /// than AR did, over enough steps to trust it.
    fn losing(&self) -> bool {
        self.drafting_steps >= REALIZED_STEPS
            && matches!((self.drafting, self.ar), (Some(drafting), Some(ar)) if drafting < ar)
    }
}

/// Where verify items get their drafts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DraftSource {
    /// The executor drafts inside `execute` and reports the drafts.
    Bundle,
    /// The engine asks `ModelExecutor::draft_tokens_v1` before execution and
    /// passes the drafts as input tokens.
    Host,
}

/// How drafter work scales with the number of tokens requested per sequence.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DraftCostModel {
    /// Each additional draft position requires another forward pass.
    #[default]
    PerToken,
    /// One forward pass produces a block, regardless of the requested depth.
    PerBlock,
}

impl DraftCostModel {
    /// What the drafting cost of a step depends on. Per-token costs follow
    /// the items and shape, which fix the rows. A block step is keyed by its
    /// items, drafting sequences and total drafts, which also fix its rows.
    fn key(self, items: usize, shape: DraftShape) -> DraftKey {
        match self {
            Self::PerToken => DraftKey { items, shape },
            Self::PerBlock => {
                let mut block = DraftShape::default();
                block.0[0] = shape.0[0];
                block.0[1] = shape.work();
                DraftKey {
                    items,
                    shape: block,
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
struct DraftKey {
    items: usize,
    shape: DraftShape,
}

/// How the engine picks each item's draft count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeculationMode {
    /// Every eligible item verifies as many drafts as its limits allow, in
    /// batch order, until the batch budget runs out.
    Fixed,
    /// Give eligible items a common executable depth, then spend spare
    /// rows on the next prefixes with the highest expected acceptance.
    /// Batch and request limits determine depth; elapsed time does not.
    Balanced,
    /// Drafts are added where they are expected to raise tokens per unit of
    /// step time, from per-request acceptance and measured step costs.
    Adaptive,
    /// Each step's drafts maximize expected tokens per unit of step time,
    /// from per-request acceptance and this fixed step-cost table. No
    /// measured time enters a plan.
    Table(StepCosts),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeculationConfigError {
    ZeroDraftTokens,
    TooManyDraftTokens { maximum: usize, requested: usize },
    ZeroVerifyTokens,
}

/// Speculative decoding settings for one engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpeculationConfig {
    maximum_draft_tokens: usize,
    maximum_verify_tokens: usize,
    mode: SpeculationMode,
    source: DraftSource,
    draft_cost_model: DraftCostModel,
    /// The most items that draft in one step, as the bundle declares.
    maximum_draft_items: Option<usize>,
}

impl SpeculationConfig {
    /// `maximum_verify_tokens` bounds one batch's decode-phase rows: one per
    /// plain decode item and `drafts + 1` per verify item.
    pub const fn new(
        maximum_draft_tokens: usize,
        maximum_verify_tokens: usize,
        mode: SpeculationMode,
        source: DraftSource,
    ) -> Result<Self, SpeculationConfigError> {
        if maximum_draft_tokens == 0 {
            return Err(SpeculationConfigError::ZeroDraftTokens);
        }
        if maximum_draft_tokens > MAX_DRAFT_TOKENS {
            return Err(SpeculationConfigError::TooManyDraftTokens {
                maximum: MAX_DRAFT_TOKENS,
                requested: maximum_draft_tokens,
            });
        }
        if maximum_verify_tokens == 0 {
            return Err(SpeculationConfigError::ZeroVerifyTokens);
        }
        Ok(Self {
            maximum_draft_tokens,
            maximum_verify_tokens,
            mode,
            source,
            draft_cost_model: DraftCostModel::PerToken,
            maximum_draft_items: None,
        })
    }

    #[must_use]
    pub const fn maximum_draft_tokens(self) -> usize {
        self.maximum_draft_tokens
    }

    #[must_use]
    pub const fn maximum_verify_tokens(self) -> usize {
        self.maximum_verify_tokens
    }

    #[must_use]
    pub const fn mode(self) -> SpeculationMode {
        self.mode
    }

    #[must_use]
    pub const fn source(self) -> DraftSource {
        self.source
    }

    /// Select the cost shape without changing draft or verify limits.
    #[must_use]
    pub const fn with_draft_cost_model(mut self, model: DraftCostModel) -> Self {
        self.draft_cost_model = model;
        self
    }

    #[must_use]
    pub const fn draft_cost_model(self) -> DraftCostModel {
        self.draft_cost_model
    }

    /// Bounds the items that draft in one step (`None`: every item may).
    /// Above the bound the policy rotates which items draft; all others
    /// still decode one row.
    #[must_use]
    pub const fn with_maximum_draft_items(mut self, items: Option<usize>) -> Self {
        self.maximum_draft_items = items;
        self
    }

    #[must_use]
    pub const fn maximum_draft_items(self) -> Option<usize> {
        self.maximum_draft_items
    }

    pub(crate) const fn limits(self) -> DraftLimits {
        DraftLimits::new(self.maximum_draft_tokens, self.maximum_verify_tokens)
    }
}

/// Per-position acceptance: the chance draft j is accepted when every
/// earlier draft of the same step was.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AcceptanceEstimate {
    positions: [f64; MAX_DRAFT_TOKENS],
}

impl AcceptanceEstimate {
    const fn initial() -> Self {
        Self {
            positions: [INITIAL_ACCEPTANCE; MAX_DRAFT_TOKENS],
        }
    }

    /// Chance that drafts `0..=position` are all accepted.
    fn survival(&self, position: usize) -> f64 {
        self.positions[..=position].iter().product()
    }

    fn observe(&mut self, verified: usize, accepted: usize, weight: f64) {
        self.observe_weighted(verified, accepted, |_| weight);
    }

    /// Updates each position the verification reached, with its own weight.
    fn observe_weighted(
        &mut self,
        verified: usize,
        accepted: usize,
        mut weight: impl FnMut(usize) -> f64,
    ) {
        for (position, estimate) in self.positions.iter_mut().enumerate().take(verified) {
            let observed = if position < accepted {
                1.0
            } else if position == accepted {
                0.0
            } else {
                break;
            };
            *estimate += weight(position) * (observed - *estimate);
        }
    }
}

/// One decode item the policy may extend.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlanCandidate<'a> {
    pub(crate) maximum_draft_tokens: usize,
    pub(crate) acceptance: &'a AcceptanceEstimate,
    /// The request did not verify in its previous step, so a drafter that
    /// catches up lazily must first process the rows it decoded since.
    pub(crate) fresh: bool,
    /// Output tokens the request may still emit; a fresh drafter's catch-up
    /// is spread over at most the steps they take.
    pub(crate) remaining_tokens: usize,
}

/// Measured step time by decode-phase rows, as an exponential average.
#[derive(Clone, Debug, Default, PartialEq)]
struct StepCostTable {
    micros: BTreeMap<usize, f64>,
}

impl StepCostTable {
    fn record(&mut self, rows: usize, micros: u64) {
        if rows > MAX_COST_ROWS {
            return;
        }
        let micros = micros.max(1) as f64;
        self.micros
            .entry(rows)
            .and_modify(|previous| *previous += COST_WEIGHT * (micros - *previous))
            .or_insert(micros);
    }

    fn measured(&self, rows: usize) -> bool {
        self.micros.contains_key(&rows)
    }

    /// Estimated cost of a step with `rows` decode-phase rows.
    ///
    /// A measured size uses its average. Otherwise the nearest measured size
    /// is scaled by the per-row prior; with no measurement at all, the prior
    /// alone gives relative costs. One plan compares estimates from the same
    /// table, so either unit is consistent within it.
    fn estimate(&self, rows: usize) -> f64 {
        let prior = |rows: usize| 1.0 + PRIOR_ROW_COST * rows as f64;
        let rows = rows.min(MAX_COST_ROWS);
        if let Some(&micros) = self.micros.get(&rows) {
            return micros;
        }
        let below = self.micros.range(..rows).next_back();
        let above = self.micros.range(rows + 1..).next();
        let nearest = match (below, above) {
            (Some(low), Some(high)) => {
                if rows - low.0 <= high.0 - rows {
                    Some(low)
                } else {
                    Some(high)
                }
            }
            (low, high) => low.or(high),
        };
        nearest.map_or_else(
            || prior(rows),
            |(&size, &micros)| micros * prior(rows) / prior(size),
        )
    }
}

/// Counts active sequences at each draft position, preserving mixed K shapes.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
struct DraftShape([usize; MAX_DRAFT_TOKENS]);

impl DraftShape {
    fn from_counts(drafts: &[usize]) -> Self {
        let mut shape = Self::default();
        for &count in drafts {
            for active in shape.0.iter_mut().take(count) {
                *active += 1;
            }
        }
        shape
    }

    fn work(self) -> usize {
        self.0.iter().sum()
    }
}

/// Wall time of a per-token drafter beyond the AR estimate of its row bucket,
/// including drafting and catch-up. This needs no bundle-specific timing or
/// ABI extension.
#[derive(Clone, Debug, Default, PartialEq)]
struct DraftCostTable {
    micros: BTreeMap<DraftKey, f64>,
}

impl DraftCostTable {
    fn record(&mut self, key: DraftKey, micros: f64, weight: f64) {
        if self.micros.len() >= MAX_COST_ROWS && !self.micros.contains_key(&key) {
            self.micros.pop_first();
        }
        self.micros
            .entry(key)
            .and_modify(|previous| *previous += weight * (micros - *previous))
            .or_insert(micros);
    }

    /// A same-bucket measurement with far fewer drafting sequences cannot
    /// separate fixed cost from cost per sequence. Its excess is only an
    /// optimistic probe price, never evidence to enable the larger shape.
    fn sparse_excess(
        &self,
        items: usize,
        shape: DraftShape,
        bucket: impl Fn(usize) -> usize,
    ) -> Option<f64> {
        if self
            .micros
            .contains_key(&DraftCostModel::PerToken.key(items, shape))
        {
            return None;
        }
        let rows = bucket(items + shape.work());
        let mut maximum = 0;
        let mut excess: Option<f64> = None;
        for (known, &micros) in self.micros.range(
            DraftKey {
                items,
                shape: DraftShape::default(),
            }..,
        ) {
            if known.items != items {
                break;
            }
            if bucket(items + known.shape.work()) == rows {
                maximum = maximum.max(known.shape.0[0]);
                excess = Some(excess.map_or(micros, |previous| previous.min(micros)));
            }
        }
        // More than doubling the measured sequence count needs a probe:
        // neither the sign nor work-scaling of one excess supplies a slope.
        excess.filter(|_| shape.0[0] > 2 * maximum)
    }

    /// The time of a step of `items` items drafting `shape` beyond the AR
    /// estimate of its row bucket, which `bucket` gives for a row count.
    fn estimate(&self, items: usize, shape: DraftShape, bucket: impl Fn(usize) -> usize) -> f64 {
        if shape.work() == 0 {
            return 0.0;
        }
        let key = DraftCostModel::PerToken.key(items, shape);
        if let Some(&micros) = self.micros.get(&key) {
            return micros;
        }
        let distance = |known: &DraftKey| {
            known
                .shape
                .0
                .iter()
                .zip(shape.0)
                .map(|(a, b)| a.abs_diff(b))
                .sum::<usize>()
        };
        // The same bucket runs the same target graph, but a larger shape
        // can draft more work. Keep the more conservative of the borrowed
        // excess and its work-scaled value, including negative excesses.
        let rows = bucket(items + shape.work());
        if let Some((known, &micros)) = self
            .micros
            .range(
                DraftKey {
                    items,
                    shape: DraftShape::default(),
                }..,
            )
            .take_while(|(known, _)| known.items == items)
            .filter(|(known, _)| bucket(items + known.shape.work()) == rows)
            .min_by_key(|(known, _)| distance(known))
        {
            return micros.max(micros * shape.work() as f64 / known.shape.work() as f64);
        }
        // Otherwise scale the nearest shape by total draft work. Only its
        // drafting carries over to other items or buckets, not a cheaper
        // target graph. Serial and batched implementations learn separate
        // shapes.
        self.micros
            .iter()
            .min_by_key(|(known, _)| (distance(known), known.items.abs_diff(items)))
            .map_or(0.0, |(known, micros)| {
                micros.max(0.0) * shape.work() as f64 / known.shape.work() as f64
            })
    }
}

/// Measured wall time of a block drafter's steps at one concurrency and row
/// bucket, as exponential averages of the time and of the drafting
/// sequences. The bucket fixes the step's target graph: every step in it
/// runs the same rows. The time includes the draft pass and the catch-up of
/// caught-up drafters.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BlockStep {
    micros: f64,
    sequences: f64,
}

/// Block steps keyed by items and row bucket.
#[derive(Clone, Debug, Default, PartialEq)]
struct BlockStepTable {
    steps: BTreeMap<(usize, usize), BlockStep>,
}

impl BlockStepTable {
    fn record(&mut self, key: (usize, usize), micros: f64, sequences: usize, weight: f64) {
        if self.steps.len() >= MAX_COST_ROWS && !self.steps.contains_key(&key) {
            self.steps.pop_first();
        }
        let sequences = sequences as f64;
        self.steps
            .entry(key)
            .and_modify(|step| {
                step.micros += weight * (micros - step.micros);
                step.sequences += weight * (sequences - step.sequences);
            })
            .or_insert(BlockStep { micros, sequences });
    }

    fn measured(&self, items: usize, bucket: usize) -> bool {
        self.steps.contains_key(&(items, bucket))
    }

    /// The nearest measured buckets at `items` below and above `bucket`,
    /// each `value` scaled to `bucket` by `wider` per doubling up and by
    /// `narrower` per halving down; the lower of the two, if any.
    fn scaled(
        &self,
        items: usize,
        bucket: usize,
        (wider, narrower): (f64, f64),
        value: impl Fn(&BlockStep) -> f64,
    ) -> Option<f64> {
        let from_narrower = self
            .steps
            .range((items, 0)..(items, bucket))
            .next_back()
            .map(|(&(_, known), step)| value(step) * wider.powf(doublings(known, bucket)));
        let from_wider = self
            .steps
            .range((items, bucket + 1)..(items + 1, 0))
            .next()
            .map(|(&(_, known), step)| value(step) / narrower.powf(doublings(bucket, known)));
        match (from_narrower, from_wider) {
            (Some(narrower), Some(wider)) => Some(narrower.min(wider)),
            (narrower, wider) => narrower.or(wider),
        }
    }

    /// The least an unmeasured bucket may plausibly cost at these items:
    /// its measured neighbours scaled by the optimistic bucket ratios, or
    /// without any, the AR step `ar` in `ar_bucket` scaled up.
    fn optimistic(&self, items: usize, bucket: usize, ar: f64, ar_bucket: usize) -> f64 {
        let ratios = (OPTIMISTIC_WIDER_RATIO, OPTIMISTIC_NARROWER_RATIO);
        self.scaled(items, bucket, ratios, |step| step.micros)
            .unwrap_or_else(|| {
                ar * OPTIMISTIC_WIDER_RATIO.powf(doublings(ar_bucket, bucket).max(0.0))
            })
            .max(ar)
    }

    /// Estimated time of a step of `items` items whose `sequences` drafting
    /// sequences bring it to row bucket `bucket`.
    ///
    /// A measured bucket uses its average, plus `sequence_cost` for each
    /// drafting sequence above the ones it averaged; fewer sequences cost
    /// no less, as the bucket's rows still run. Otherwise the same bucket
    /// measured at the nearest other item count lends its time: the bucket
    /// fixes the target graph, and at TP4 a 128-row verify step took 37 ms
    /// at 16 and at 32 items. Without one, the nearest measured buckets at
    /// these items lend their time, scaled by the bucket ratios; the lower
    /// of the two estimates stands. At 11 to 15 items that scaling priced
    /// the 128-row step at 70 ms, so drafting stayed shallow as a cohort
    /// finished. Only a concurrency without any drafting measurement
    /// borrows another's overhead over its AR rows, never a negative one:
    /// verify rows that cost less than AR rows at C1 say nothing about the
    /// same rows at C16.
    fn estimate(
        &self,
        items: usize,
        bucket: usize,
        sequences: usize,
        ar: &StepCostTable,
        sequence_cost: f64,
    ) -> f64 {
        let more = |step: &BlockStep| {
            step.micros + (sequences as f64 - step.sequences).max(0.0) * sequence_cost
        };
        if let Some(step) = self.steps.get(&(items, bucket)) {
            return more(step);
        }
        if let Some(step) = self
            .steps
            .iter()
            .filter(|&(&(_, known), _)| known == bucket)
            .min_by_key(|&(&(known, _), _)| known.abs_diff(items))
            .map(|(_, step)| step)
        {
            return more(step);
        }
        let ratios = (WIDER_BUCKET_RATIO, NARROWER_BUCKET_RATIO);
        if let Some(estimate) = self.scaled(items, bucket, ratios, more) {
            return estimate;
        }
        let borrowed = self
            .steps
            .iter()
            .min_by_key(|&(&(known_items, known_bucket), _)| {
                (known_items.abs_diff(items), known_bucket.abs_diff(bucket))
            })
            .map_or(0.0, |(&(_, known_bucket), step)| {
                (more(step) - ar.estimate(known_bucket)).max(0.0)
            });
        ar.estimate(bucket) + borrowed
    }
}

/// Doublings of rows from `narrow` to `wide`.
fn doublings(narrow: usize, wide: usize) -> f64 {
    (wide as f64 / narrow.max(1) as f64).log2()
}

/// A draft count per candidate with its expected emitted tokens and
/// estimated step cost.
#[derive(Clone, Debug, PartialEq)]
struct Plan {
    drafts: Vec<usize>,
    rows: usize,
    /// The drafting fresh candidates' shares of their catch-up this step.
    catch_up: f64,
    expected: f64,
    cost: f64,
}

impl Plan {
    fn rate(&self) -> f64 {
        self.expected / self.cost
    }
}

/// A probe step whose drafters were fresh: its items, row bucket, shape,
/// fresh drafters and time. The next matching probe remeasures it with them
/// caught up, which separates the catch-up from the steady step.
#[derive(Clone, Copy, Debug, PartialEq)]
struct FreshProbe {
    items: usize,
    bucket: usize,
    shape: DraftShape,
    fresh: usize,
    micros: f64,
}

/// Speculation state one engine keeps across steps.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpeculationPolicy {
    config: SpeculationConfig,
    prior: AcceptanceEstimate,
    /// Observations of each prior position, saturating.
    prior_observations: [u32; MAX_DRAFT_TOKENS],
    costs: StepCostTable,
    draft_costs: DraftCostTable,
    block_steps: BlockStepTable,
    /// Per concurrency, the catch-up time of one fresh drafting sequence
    /// beyond its step's steady estimate, as an exponential average.
    fresh_costs: BTreeMap<usize, f64>,
    /// Steps already executed once, by items, row bucket (block drafters)
    /// and draft shape (per-token drafters).
    warm_shapes: BTreeSet<(usize, usize, DraftShape)>,
    zero_draft: Vec<ZeroDraftState>,
    /// The items of a probe plan whose step has not been observed yet.
    probing: Option<usize>,
    fresh_probe: Option<FreshProbe>,
    /// Per concurrency, the throughput its steps realized.
    realized: BTreeMap<usize, Realized>,
    /// Drafts accepted in the step being observed, from `observe`.
    step_accepted: usize,
    /// Drafting plans left before the next full-depth exploration.
    explore: u8,
    /// Draft counts the executor runs without rounding them down. A plan
    /// holds only these, so the step it measures is the step it priced.
    executable: [bool; MAX_DRAFT_TOKENS + 1],
    /// The rows the executor runs for a step of each row count up to
    /// `MAX_COST_ROWS`: its row bucket.
    step_rows: Vec<usize>,
    draft_cursor: usize,
    trace_enabled: bool,
    decision: Option<PlanTrace>,
}

impl SpeculationPolicy {
    pub(crate) fn new(config: SpeculationConfig) -> Self {
        Self {
            config,
            prior: AcceptanceEstimate::initial(),
            prior_observations: [0; MAX_DRAFT_TOKENS],
            costs: StepCostTable::default(),
            draft_costs: DraftCostTable::default(),
            block_steps: BlockStepTable::default(),
            fresh_costs: BTreeMap::new(),
            warm_shapes: BTreeSet::new(),
            zero_draft: vec![ZeroDraftState::default(); MAX_COST_ROWS + 1],
            probing: None,
            fresh_probe: None,
            realized: BTreeMap::new(),
            step_accepted: 0,
            explore: EXPLORE_INTERVAL - 1,
            executable: [true; MAX_DRAFT_TOKENS + 1],
            step_rows: (0..=MAX_COST_ROWS).collect(),
            draft_cursor: 0,
            trace_enabled: false,
            decision: None,
        }
    }

    /// Plans only the draft counts `round` keeps, such as the counts whose
    /// verify rows fill a captured row bucket.
    pub(crate) fn with_draft_rounding(mut self, round: impl Fn(usize) -> usize) -> Self {
        for (count, executable) in self.executable.iter_mut().enumerate() {
            *executable = round(count) == count;
        }
        self
    }

    /// Prices a step of `rows` rows as the step of `bucket(rows)` rows the
    /// executor runs for it.
    pub(crate) fn with_step_rows(mut self, bucket: impl Fn(usize) -> usize) -> Self {
        for (rows, step) in self.step_rows.iter_mut().enumerate() {
            *step = bucket(rows).max(rows);
        }
        self
    }

    /// The row bucket of a step with `rows` rows.
    fn bucket(&self, rows: usize) -> usize {
        self.step_rows.get(rows).copied().unwrap_or(rows)
    }

    /// The largest executable draft count up to `count`.
    fn executable_at_most(&self, count: usize) -> usize {
        (0..=count.min(MAX_DRAFT_TOKENS))
            .rev()
            .find(|&count| self.executable[count])
            .unwrap_or(0)
    }

    pub(crate) const fn config(&self) -> SpeculationConfig {
        self.config
    }

    /// The estimate a request starts from.
    pub(crate) fn prior(&self) -> AcceptanceEstimate {
        self.prior.clone()
    }

    /// Records one verified item: `verified` drafts were checked and the first
    /// `accepted` matched. A run that ended on a stop or length cut before a
    /// mismatch passes `accepted == verified`.
    pub(crate) fn observe(
        &mut self,
        estimate: &mut AcceptanceEstimate,
        verified: usize,
        accepted: usize,
    ) {
        estimate.observe(verified, accepted, REQUEST_WEIGHT);
        self.step_accepted += accepted;
        // A position's first observations average with the initial value as
        // one more sample, so explored positions learn quickly; later ones
        // move the prior by PRIOR_WEIGHT.
        let observations = &mut self.prior_observations;
        self.prior.observe_weighted(verified, accepted, |position| {
            let count = &mut observations[position];
            *count = count.saturating_add(1);
            (1.0 / (f64::from(*count) + 1.0)).max(PRIOR_WEIGHT)
        });
    }

    /// Records the time of a step whose items were all decode or verify
    /// items, `fresh` of whose drafting items were fresh.
    ///
    /// A step with fresh drafters also paid their catch-up, which a steady
    /// step does not: its time beyond the steady estimate is spread over
    /// them as catch-up cost, and the steady tables keep only steps without
    /// fresh drafters. A probe follows long K=0 spells, so all its drafters
    /// are fresh; it caps the catch-up estimate at its own time beyond AR.
    /// Where neither the bucket's steady time nor the concurrency's
    /// catch-up is known, it probes again with the same shape: the two steps
    /// differ by the catch-ups of the drafters that were fresh only in the
    /// first.
    pub(crate) fn observe_step(&mut self, drafts: &[usize], fresh: usize, micros: u64) {
        if matches!(
            self.config.mode,
            SpeculationMode::Table(_) | SpeculationMode::Balanced
        ) {
            // These plans never read step times.
            self.step_accepted = 0;
            return;
        }
        let items = drafts.len();
        let shape = DraftShape::from_counts(drafts);
        // What the step emitted per unit of time: one token per item plus
        // every accepted draft. It decides the realized comparison.
        let emitted = (items + std::mem::take(&mut self.step_accepted)) as f64;
        let model = self.config.draft_cost_model;
        let key = model.key(items, shape);
        let rows = items + shape.work();
        let bucket = self.bucket(rows);
        // A probe replaces what it remeasures: the stored value is stale.
        let probe = shape.work() > 0 && self.probing.take_if(|probed| *probed == items).is_some();
        // Every probe step, a first execution's included, spaces the next.
        let spaced = (PROBE_SPACING * micros as f64 / self.costs.estimate(items))
            .clamp(f64::from(PROBE_INTERVAL), f64::from(MAX_PROBE_INTERVAL))
            as u16;
        if probe && let Some(state) = self.zero_draft.get_mut(items) {
            state.probe_interval = state.probe_interval.max(spaced);
            state.probe_ar_micros = Some(self.costs.estimate(items));
        }
        // A fresh probe is remeasured by the very next drafting step, if at
        // all.
        let first = if shape.work() > 0 {
            self.fresh_probe.take()
        } else {
            None
        };
        if probe && fresh > 0 {
            // The whole probe step bounds its catch-up, even when it also
            // paid a first execution: an estimate above what the step took
            // beyond AR is stale.
            let bound = ((micros as f64 - self.costs.estimate(items)) / fresh as f64).max(0.0);
            if self.fresh_cost(items) > bound {
                self.fresh_costs.insert(items, bound);
            }
        }
        // The first execution can include lazy preparation or graph capture.
        // Do not let a one-time cost suppress all subsequent measurements.
        let warm = match (shape.work(), model) {
            (0, _) => (items, 0, DraftShape::default()),
            (_, DraftCostModel::PerBlock) => (items, bucket, DraftShape::default()),
            (_, DraftCostModel::PerToken) => (items, 0, key.shape),
        };
        if !self.warm_shapes.contains(&warm) {
            if self.warm_shapes.len() >= MAX_COST_ROWS {
                self.warm_shapes.pop_first();
            }
            self.warm_shapes.insert(warm);
            if probe {
                self.probe_again(items);
            }
            return;
        }
        let weight = if probe { 1.0 } else { COST_WEIGHT };
        // A probe decides whether drafting resumes, and all its drafters
        // are fresh: it counts their catch-up as a drafting run would, spread
        // over the run, or at high concurrency no probe could ever lift the
        // drafting average above AR.
        let run_catch_up = if probe {
            fresh as f64 * self.fresh_cost(items) * (1.0 - 1.0 / FRESH_RUN_STEPS)
        } else {
            0.0
        };
        let rate = emitted / (micros as f64 - run_catch_up).max(1.0);
        let realized = self.realized.entry(items).or_default();
        if shape.work() == 0 {
            Realized::record(&mut realized.ar, rate, REALIZED_WEIGHT);
        } else {
            // A probe carries more weight: the drafting average it corrects
            // is stale.
            let weight = if probe { 0.5 } else { REALIZED_WEIGHT };
            Realized::record(&mut realized.drafting, rate, weight);
            realized.drafting_steps = realized.drafting_steps.saturating_add(1);
            // Only a probe that clears the resume margin resets spacing.
            // Marginal gains keep backing off while drafting stays disabled.
            if probe
                && realized.ar.is_some_and(|ar| rate > ar * RESUME_GAIN)
                && let Some(state) = self.zero_draft.get_mut(items)
            {
                state.probe_interval = spaced;
            }
        }
        let mut micros = micros as f64;
        // Only matching shapes isolate catch-up from changes in draft work.
        if let Some(first) = first.filter(|first| {
            probe && first.items == items && first.bucket == bucket && first.shape == shape
        }) {
            if first.fresh > fresh {
                let catch_up = ((first.micros - micros) / (first.fresh - fresh) as f64).max(0.0);
                self.fresh_costs.insert(items, catch_up);
                let steady = (micros - fresh as f64 * catch_up).max(self.costs.estimate(items));
                match model {
                    DraftCostModel::PerToken => {
                        let excess = steady - self.costs.estimate(bucket);
                        self.draft_costs.record(key, excess, weight);
                    }
                    DraftCostModel::PerBlock => {
                        self.block_steps
                            .record((items, bucket), steady, shape.0[0], weight);
                    }
                }
            }
            return;
        }
        // Without a steady measurement of the bucket or a catch-up
        // measurement at these items, one step cannot tell the catch-up from
        // the step: probe again once the drafters caught up.
        let steady_known = match model {
            DraftCostModel::PerToken => self.draft_costs.micros.contains_key(&key),
            DraftCostModel::PerBlock => self.block_steps.measured(items, bucket),
        };
        if probe && fresh > 0 && (!steady_known || !self.fresh_costs.contains_key(&items)) {
            self.fresh_probe = Some(FreshProbe {
                items,
                bucket,
                shape,
                fresh,
                micros,
            });
            self.probe_again(items);
            return;
        }
        if fresh > 0 && shape.work() > 0 {
            let catch_up = fresh as f64 * self.fresh_cost(items);
            if !probe {
                let steady = self.step_cost(items, shape, 0.0);
                let sample = ((micros - steady) / fresh as f64).max(0.0);
                self.fresh_costs
                    .entry(items)
                    .and_modify(|previous| *previous += COST_WEIGHT * (sample - *previous))
                    .or_insert(sample);
                return;
            }
            micros = (micros - catch_up).max(self.costs.estimate(items));
        }
        let micros_u64 = micros as u64;
        match (shape.work(), model) {
            (0, _) => self.costs.record(rows, micros_u64),
            (_, DraftCostModel::PerToken) => {
                let excess = micros - self.costs.estimate(bucket);
                self.draft_costs.record(key, excess, weight);
            }
            (_, DraftCostModel::PerBlock) => {
                self.block_steps
                    .record((items, bucket), micros, shape.0[0], weight);
            }
        }
    }

    /// Runs the next plan at `items` items as a probe too.
    fn probe_again(&mut self, items: usize) {
        if let Some(state) = self.zero_draft.get_mut(items) {
            state.idle = state.probe_interval;
            state.again = true;
        }
    }

    /// Chooses a draft count for each candidate, in candidate order.
    pub(crate) fn plan(
        &mut self,
        candidates: &[PlanCandidate<'_>],
        budget: DraftBudget,
    ) -> Vec<usize> {
        self.begin_diagnostics(
            candidates.len(),
            budget.spare_draft_tokens().min(budget.spare_rows()),
        );
        let drafts = self.plan_inner(candidates, budget);
        self.finish_diagnostics(&drafts);
        drafts
    }

    fn plan_inner(&mut self, candidates: &[PlanCandidate<'_>], budget: DraftBudget) -> Vec<usize> {
        let eligible = candidates
            .iter()
            .filter(|c| c.maximum_draft_tokens > 0)
            .count();
        if let Some(limit) = self.config.maximum_draft_items
            && eligible > limit
        {
            // Plan in rotated order from the cursor, so the budget goes
            // first to the items that waited longest, and offer drafting to
            // at most `limit` eligible items.
            let count = candidates.len();
            let start = self.draft_cursor % count;
            let order: Vec<usize> = (0..count).map(|offset| (start + offset) % count).collect();
            let mut rotated: Vec<_> = order.iter().map(|&index| candidates[index]).collect();
            let mut offered = 0;
            for candidate in &mut rotated {
                if candidate.maximum_draft_tokens > 0 {
                    offered += 1;
                    if offered > limit {
                        candidate.maximum_draft_tokens = 0;
                    }
                }
            }
            let planned = self.plan_unbounded(&rotated, budget);
            // The next step starts after the last item that drafted, so
            // every eligible item drafts in turn however few rows are spare.
            if let Some(last) = planned.iter().rposition(|&drafts| drafts > 0) {
                self.draft_cursor = (start + last + 1) % count;
            }
            let mut drafts = vec![0; count];
            for (&index, planned) in order.iter().zip(planned) {
                drafts[index] = planned;
            }
            return drafts;
        }
        self.plan_unbounded(candidates, budget)
    }

    fn plan_unbounded(
        &mut self,
        candidates: &[PlanCandidate<'_>],
        budget: DraftBudget,
    ) -> Vec<usize> {
        let mut drafts = vec![0; candidates.len()];
        let mut remaining = budget.spare_draft_tokens().min(budget.spare_rows());
        if remaining == 0 {
            self.trace_reason("no_spare_rows");
            return drafts;
        }
        if candidates
            .iter()
            .all(|candidate| candidate.maximum_draft_tokens == 0)
        {
            self.trace_reason("no_eligible_items");
            return drafts;
        }
        match self.config.mode {
            SpeculationMode::Fixed => {
                self.trace_reason("fixed");
                for (draft, candidate) in drafts.iter_mut().zip(candidates) {
                    *draft = candidate.maximum_draft_tokens.min(remaining);
                    remaining -= *draft;
                }
            }
            SpeculationMode::Balanced => {
                self.trace_reason("balanced");
                drafts = self.plan_balanced(candidates, remaining);
            }
            SpeculationMode::Adaptive if budget.prefill() => {
                self.trace_reason("prefill_fresh");
                // A batch that also prefills is no decode step: its time is
                // never observed, and a fresh drafter would add its whole
                // catch-up to the prefill that every waiting request's first
                // token waits for. Only caught-up requests draft there, and
                // no probe runs.
                let steady = candidates
                    .iter()
                    .map(|candidate| PlanCandidate {
                        maximum_draft_tokens: if candidate.fresh {
                            0
                        } else {
                            candidate.maximum_draft_tokens
                        },
                        ..*candidate
                    })
                    .collect::<Vec<_>>();
                if steady
                    .iter()
                    .any(|candidate| candidate.maximum_draft_tokens > 0)
                {
                    drafts = self.plan_adaptive(&steady, remaining, false);
                }
            }
            SpeculationMode::Adaptive => {
                drafts = self.plan_adaptive(candidates, remaining, true);
            }
            SpeculationMode::Table(costs) => {
                self.trace_reason("table");
                drafts = self.plan_table(candidates, &costs, remaining);
            }
        }
        drafts
    }

    /// Table mode: the table plan, then the periodic full-depth exploration
    /// that keeps every position's acceptance observed. Both depend only on
    /// the plan's inputs and the count of earlier drafting plans.
    fn plan_table(
        &mut self,
        candidates: &[PlanCandidate<'_>],
        costs: &StepCosts,
        remaining: usize,
    ) -> Vec<usize> {
        let survival = candidates
            .iter()
            .map(|candidate| {
                let depth = self.executable_at_most(candidate.maximum_draft_tokens);
                (0..depth)
                    .map(|position| candidate.acceptance.survival(position))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let table_candidates = candidates
            .iter()
            .zip(&survival)
            .map(|(candidate, survival)| table::TableCandidate {
                maximum_draft_tokens: candidate.maximum_draft_tokens,
                survival,
            })
            .collect::<Vec<_>>();
        let mut drafts = table::plan(
            &table_candidates,
            costs,
            remaining,
            self.config.maximum_draft_items.unwrap_or(usize::MAX),
            &self.executable,
        )
        .drafts;
        if drafts.iter().any(|&draft| draft > 0) {
            self.explore_depth(candidates, &mut drafts, remaining);
        }
        drafts
    }

    /// Catch-up time of one fresh drafting sequence at `items` items, from
    /// the nearest concurrency that measured one; zero before any did.
    fn fresh_cost(&self, items: usize) -> f64 {
        let below = self.fresh_costs.range(..=items).next_back();
        let above = self.fresh_costs.range(items..).next();
        match (below, above) {
            (Some(low), Some(high)) if high.0 - items < items - low.0 => *high.1,
            (Some(low), _) => *low.1,
            (None, high) => high.map_or(0.0, |(_, &micros)| micros),
        }
    }

    /// The share of its catch-up a fresh candidate pays in one step: a
    /// drafting run of `FRESH_RUN_STEPS` steps, or of the steps its
    /// remaining output takes at its expected tokens per step when fewer.
    fn catch_up_share(&self, candidate: &PlanCandidate<'_>) -> f64 {
        if !candidate.fresh {
            return 0.0;
        }
        let depth = self.executable_at_most(candidate.maximum_draft_tokens);
        let per_step = 1.0
            + (0..depth)
                .map(|position| candidate.acceptance.survival(position))
                .sum::<f64>();
        1.0 / (candidate.remaining_tokens as f64 / per_step).clamp(1.0, FRESH_RUN_STEPS)
    }

    /// Estimated time of a step of `items` items drafting `shape`, whose
    /// fresh drafters pay `catch_up` catch-ups in all. A step costs what its
    /// row bucket costs, and never less than its AR step.
    fn step_cost(&self, items: usize, shape: DraftShape, catch_up: f64) -> f64 {
        let ar = self.costs.estimate(items);
        if shape.work() == 0 {
            return ar;
        }
        let bucket = self.bucket(items + shape.work());
        match self.config.draft_cost_model {
            // Per-token drafters price the target at its executed bucket,
            // from the AR table, plus the drafting beyond it. Priced at the
            // exact row count, a step that crosses into a wider graph (16 to
            // 32 rows at C16) charged the jump to its drafts and scaled it
            // with their count.
            DraftCostModel::PerToken => {
                self.costs.estimate(bucket)
                    + self
                        .draft_costs
                        .estimate(items, shape, |rows| self.bucket(rows))
            }
            DraftCostModel::PerBlock => self.block_steps.estimate(
                items,
                bucket,
                shape.0[0],
                &self.costs,
                self.costs.estimate(1) * PRIOR_SEQUENCE_COST,
            ),
        }
        .max(ar)
            + catch_up * self.fresh_cost(items)
    }

    /// Picks the plan with the most expected tokens per unit of step time:
    /// greedy single-item additions, or the best joint plan extended
    /// greedily. Leaving drafting off must first clear the resume margin,
    /// except for a probe, which runs the best drafting plan to its full
    /// depth once every probe interval when `may_probe`. A block drafter's
    /// probe first measures the row buckets it never ran at this
    /// concurrency, widest first.
    fn plan_adaptive(
        &mut self,
        candidates: &[PlanCandidate<'_>],
        remaining: usize,
        may_probe: bool,
    ) -> Vec<usize> {
        let items = candidates.len();
        // Obtain a warmed AR baseline at this concurrency before learning
        // drafting overhead. Fixed mode is unchanged.
        if !self.costs.measured(items) {
            self.trace_reason("await_ar_baseline");
            return vec![0; items];
        }
        // Backoff learned against faster AR can outlive that comparison.
        // A material slowdown permits one early remeasurement, even when
        // the candidate shape was already measured. Stable timings retain
        // the interval, and observing the probe resets this reference.
        let slower_ar = self.config.draft_cost_model == DraftCostModel::PerToken
            && self.zero_draft.get(items).is_some_and(|state| {
                state.idle.saturating_add(1) >= FIRST_PROBE
                    && state
                        .probe_ar_micros
                        .is_some_and(|previous| self.costs.estimate(items) > previous * RESUME_GAIN)
            });
        let token_probe = (may_probe
            && self.config.draft_cost_model == DraftCostModel::PerToken
            && self.zero_draft.get(items).is_some_and(|state| {
                let idle = state.idle.saturating_add(1);
                state.disabled && (idle == FIRST_PROBE || idle >= state.probe_interval || slower_ar)
            }))
        .then(|| {
            let shares = candidates
                .iter()
                .map(|candidate| self.catch_up_share(candidate))
                .collect::<Vec<_>>();
            self.joint(candidates, &shares, remaining, true)
        })
        .flatten()
        .filter(|plan| plan.rate() > RESUME_GAIN * items as f64 / self.costs.estimate(items));
        let mut probe = false;
        let resume_gain = if let Some(state) = self.zero_draft.get_mut(items) {
            if state.disabled {
                state.idle = state.idle.saturating_add(1);
                let first = state.idle == FIRST_PROBE
                    && (token_probe.is_some()
                        || (self.config.draft_cost_model == DraftCostModel::PerBlock
                            && self
                                .block_steps
                                .steps
                                .range((items, 0)..(items + 1, 0))
                                .next()
                                .is_none()));
                probe = may_probe && (state.idle >= state.probe_interval || first || slower_ar);
            }
            if state.remaining > 0 && !probe {
                state.remaining -= 1;
                self.trace_reason("cooldown");
                return vec![0; items];
            }
            if state.disabled { RESUME_GAIN } else { 1.0 }
        } else {
            1.0
        };
        // Whatever the estimates say, drafting stays off where its realized
        // throughput is below AR's; only probes measure it again.
        if !probe && self.realized.get(&items).is_some_and(Realized::losing) {
            if let Some(state) = self.zero_draft.get_mut(items) {
                state.disabled = true;
                state.remaining = ZERO_DRAFT_RECHECK;
            }
            self.trace_reason("realized_slower");
            return vec![0; items];
        }
        // A probe pays its fresh drafters' catch-up whole, as drafting
        // stays off after it unless it pays: it picks few fresh drafters.
        let shares = candidates
            .iter()
            .map(|candidate| match (probe, candidate.fresh) {
                (true, true) => 1.0,
                _ => self.catch_up_share(candidate),
            })
            .collect::<Vec<_>>();
        let empty = Plan {
            drafts: vec![0; items],
            rows: items,
            catch_up: 0.0,
            expected: items as f64,
            cost: self.costs.estimate(items),
        };
        let baseline = empty.rate();
        let margin = if probe { 0.0 } else { resume_gain };
        let mut best = self.extend(candidates, &shares, empty, remaining, margin);
        if let Some(joint) = self.joint(candidates, &shares, remaining, false) {
            self.trace_candidate(&joint);
            if joint.rate() > margin * baseline {
                let joint = self.extend(candidates, &shares, joint, remaining, 1.0);
                if joint.rate() > best.rate() {
                    best = joint;
                }
            }
        }
        self.trace_candidate(&best);
        self.trace_reason("estimated_slower");
        let mut drafts = best.drafts;
        let block = self.config.draft_cost_model == DraftCostModel::PerBlock;
        // Measurement probes run exactly as filled: deeper exploration
        // would measure a different shape or bucket.
        let unmeasured = if probe && block {
            self.unmeasured_bucket(candidates, remaining, 0)
        } else if probe {
            token_probe.map(|plan| plan.drafts)
        } else {
            None
        };
        let measurement_probe = unmeasured.is_some();
        if let Some(unmeasured) = unmeasured {
            drafts = unmeasured;
        }
        // Do not sustain a distant extrapolation in either direction. A
        // scheduled probe warms and measures it before normal selection.
        if !probe
            && !block
            && self
                .draft_costs
                .sparse_excess(items, DraftShape::from_counts(&drafts), |rows| {
                    self.bucket(rows)
                })
                .is_some()
        {
            drafts.fill(0);
            self.trace_reason("sparse_cost");
        }
        let drafting = drafts.iter().any(|&draft| draft > 0);
        if drafting {
            self.trace_reason(if probe { "probe" } else { "drafting" });
        }
        if drafting && !measurement_probe {
            if probe {
                self.explore = 0;
            }
            let rows = items + drafts.iter().sum::<usize>();
            // Exploration also measures a wider bucket than the chosen one
            // that this concurrency has never run, in a step that is timed.
            let wider = (block && may_probe && !probe && self.explore == 0)
                .then(|| self.unmeasured_bucket(candidates, remaining, self.bucket(rows)))
                .flatten();
            if let Some(wider) = wider {
                drafts = wider;
                self.explore = EXPLORE_INTERVAL - 1;
            } else {
                self.explore_depth(candidates, &mut drafts, remaining);
            }
        }
        if let Some(state) = self.zero_draft.get_mut(items) {
            if probe {
                // Drafting stays off until a recheck clears the resume margin
                // on what the probe measured.
                state.idle = 0;
                state.remaining = ZERO_DRAFT_RECHECK;
                if !std::mem::take(&mut state.again) {
                    state.probe_interval = state
                        .probe_interval
                        .saturating_mul(2)
                        .min(MAX_PROBE_INTERVAL);
                }
                self.probing = drafting.then_some(items);
            } else {
                // An estimate that chose AR is rechecked on the next plan at
                // these items, behind the resume margin. A count recurs
                // mostly as requests finish, often cohorts later with other
                // requests, so skipping its next plans there would repeat
                // a stale choice.
                state.disabled = !drafting;
                state.remaining = 0;
                if drafting {
                    state.idle = 0;
                    state.probe_interval = PROBE_INTERVAL;
                }
            }
        }
        drafts
    }

    /// The widest row bucket above `above` rows that a block step at these
    /// items never ran and that might beat AR, filled by the candidates
    /// expected to accept the most at their deepest executable count,
    /// caught-up ones first among equals; `None` when there is none.
    ///
    /// Estimates extrapolated from narrow steps misprice wide ones: at C32 a
    /// probe of one drafting sequence moved the step from the 32-row graph
    /// to the 64-row one, and the full K7 block priced from it came out at
    /// twice the 53 ms it measured.
    fn unmeasured_bucket(
        &self,
        candidates: &[PlanCandidate<'_>],
        remaining: usize,
        above: usize,
    ) -> Option<Vec<usize>> {
        let items = candidates.len();
        let mut ranked = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                let drafts = self.executable_at_most(candidate.maximum_draft_tokens);
                let gain = (0..drafts)
                    .map(|position| candidate.acceptance.survival(position))
                    .sum::<f64>();
                (index, drafts, gain)
            })
            .filter(|&(_, drafts, _)| drafts > 0)
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| {
            b.2.total_cmp(&a.2)
                .then(candidates[a.0].fresh.cmp(&candidates[b.0].fresh))
        });
        let ar = self.costs.estimate(items);
        let widest = items + remaining.min(ranked.iter().map(|&(_, drafts, _)| drafts).sum());
        let mut rows = widest;
        while rows > items {
            let bucket = self.bucket(rows);
            if bucket <= above {
                return None;
            }
            // The narrowest row count of this bucket.
            let narrowest = (items + 1..=rows)
                .find(|&count| self.bucket(count) == bucket)
                .unwrap_or(rows);
            if !self.block_steps.measured(items, bucket) {
                let mut drafts = vec![0; items];
                let mut left = rows - items;
                let mut expected = items as f64;
                for &(index, deepest, _) in &ranked {
                    let count = self.executable_at_most(deepest.min(left));
                    drafts[index] = count;
                    left -= count;
                    expected += (0..count)
                        .map(|position| candidates[index].acceptance.survival(position))
                        .sum::<f64>();
                }
                let filled = items + drafts.iter().sum::<usize>();
                // Probe only a bucket that could beat AR at its optimistic cost.
                let cost = self
                    .block_steps
                    .optimistic(items, bucket, ar, self.bucket(items));
                if filled >= narrowest && expected / cost > items as f64 / ar {
                    return Some(drafts);
                }
            }
            rows = narrowest - 1;
        }
        None
    }

    /// Adds the draft that most raises tokens per unit of step time until
    /// none does. The first draft of an empty plan must beat `margin`.
    fn extend(
        &self,
        candidates: &[PlanCandidate<'_>],
        shares: &[f64],
        mut plan: Plan,
        remaining: usize,
        margin: f64,
    ) -> Plan {
        let model = self.config.draft_cost_model;
        let items = candidates.len();
        let mut left = remaining.saturating_sub(plan.drafts.iter().sum());
        // A block step's cost before catch-up depends only on its rows and
        // drafting sequences: price each once per round.
        let mut block_costs = BTreeMap::new();
        while left > 0 {
            let shape = DraftShape::from_counts(&plan.drafts);
            let mut best = None;
            block_costs.clear();
            for (index, (candidate, &draft)) in candidates.iter().zip(&plan.drafts).enumerate() {
                let available = candidate
                    .maximum_draft_tokens
                    .saturating_sub(draft)
                    .min(left);
                let mut next_shape = shape;
                let mut survival = 0.0;
                let catch_up = plan.catch_up + if draft == 0 { shares[index] } else { 0.0 };
                for added in 1..=available {
                    next_shape.0[draft + added - 1] += 1;
                    survival += candidate.acceptance.survival(draft + added - 1);
                    if !self.executable[draft + added] {
                        continue;
                    }
                    let next_cost = match model {
                        DraftCostModel::PerToken => self.step_cost(items, next_shape, catch_up),
                        DraftCostModel::PerBlock => {
                            let key = (plan.rows + added, next_shape.0[0]);
                            let steady = *block_costs
                                .entry(key)
                                .or_insert_with(|| self.step_cost(items, next_shape, 0.0));
                            steady + catch_up * self.fresh_cost(items)
                        }
                    };
                    let rate = (plan.expected + survival) / next_cost;
                    if best.is_none_or(|(_, _, _, _, previous, _)| rate >= previous) {
                        best = Some((index, added, survival, next_cost, rate, catch_up));
                    }
                    // A per-token drafter adds one executable step at a time.
                    if model == DraftCostModel::PerToken {
                        break;
                    }
                }
            }
            let Some((index, added, survival, next_cost, rate, catch_up)) = best else {
                break;
            };
            // Keep the draft only if total tokens per wall time improve.
            let margin = if plan.rows == items { margin } else { 1.0 };
            if rate <= margin * plan.rate() {
                break;
            }
            plan.drafts[index] += added;
            plan.rows += added;
            plan.catch_up = catch_up;
            plan.expected += survival;
            plan.cost = next_cost;
            left -= added;
        }
        plan
    }

    /// The best joint plan: for each depth k, the m items expected to accept
    /// the most drafts at k each draft up to k, for every m the budget
    /// admits. A block drafter pays one pass for all of them, which single
    /// additions cannot see when the first item alone does not repay it.
    fn joint(
        &self,
        candidates: &[PlanCandidate<'_>],
        shares: &[f64],
        remaining: usize,
        sparse_probe: bool,
    ) -> Option<Plan> {
        let items = candidates.len();
        // gains[i][d]: expected accepted drafts when item i drafts d.
        let gains = candidates
            .iter()
            .map(|candidate| {
                let mut gains = vec![0.0; candidate.maximum_draft_tokens + 1];
                for position in 0..candidate.maximum_draft_tokens {
                    gains[position + 1] = gains[position] + candidate.acceptance.survival(position);
                }
                gains
            })
            .collect::<Vec<_>>();
        let deepest = candidates
            .iter()
            .map(|candidate| candidate.maximum_draft_tokens)
            .max()
            .unwrap_or(0)
            .min(remaining);
        // Most expected drafts first; among equals, candidates whose
        // drafter needs no catch-up, then candidate order.
        let rank = |ranked: &mut Vec<(usize, usize, f64)>| {
            ranked.sort_by(|a, b| {
                b.2.total_cmp(&a.2)
                    .then(candidates[a.0].fresh.cmp(&candidates[b.0].fresh))
            });
        };
        let mut best: Option<(f64, usize, usize)> = None;
        let mut ranked = Vec::with_capacity(items);
        for depth in (1..=deepest).filter(|&depth| self.executable[depth]) {
            ranked.clear();
            ranked.extend(
                candidates
                    .iter()
                    .enumerate()
                    .filter(|(_, candidate)| candidate.maximum_draft_tokens > 0)
                    .map(|(index, candidate)| {
                        let drafts =
                            self.executable_at_most(depth.min(candidate.maximum_draft_tokens));
                        (index, drafts, gains[index][drafts])
                    })
                    .filter(|&(_, drafts, _)| drafts > 0),
            );
            rank(&mut ranked);
            let mut shape = DraftShape::default();
            let mut used = 0;
            let mut catch_up = 0.0;
            let mut expected = items as f64;
            for (members, &(index, drafts, gain)) in ranked.iter().enumerate() {
                if used + drafts > remaining {
                    break;
                }
                for active in shape.0.iter_mut().take(drafts) {
                    *active += 1;
                }
                used += drafts;
                catch_up += shares[index];
                expected += gain;
                let cost = if sparse_probe {
                    let Some(excess) = self
                        .draft_costs
                        .sparse_excess(items, shape, |rows| self.bucket(rows))
                    else {
                        continue;
                    };
                    (self.costs.estimate(self.bucket(items + shape.work())) + excess)
                        .max(self.costs.estimate(items))
                        + catch_up * self.fresh_cost(items)
                } else {
                    self.step_cost(items, shape, catch_up)
                };
                let rate = expected / cost;
                if best.is_none_or(|(previous, _, _)| rate > previous) {
                    best = Some((rate, depth, members + 1));
                }
            }
        }
        let (rate, depth, members) = best?;
        let mut ranked = candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.maximum_draft_tokens > 0)
            .map(|(index, candidate)| {
                let drafts = self.executable_at_most(depth.min(candidate.maximum_draft_tokens));
                (index, drafts, gains[index][drafts])
            })
            .filter(|&(_, drafts, _)| drafts > 0)
            .collect::<Vec<_>>();
        rank(&mut ranked);
        let mut drafts = vec![0; items];
        let mut expected = items as f64;
        let mut catch_up = 0.0;
        for &(index, count, gain) in ranked.iter().take(members) {
            drafts[index] = count;
            expected += gain;
            catch_up += shares[index];
        }
        let rows = items + drafts.iter().sum::<usize>();
        Some(Plan {
            cost: if sparse_probe {
                expected / rate
            } else {
                self.step_cost(items, DraftShape::from_counts(&drafts), catch_up)
            },
            drafts,
            rows,
            catch_up,
            expected,
        })
    }

    /// Once every EXPLORE_INTERVAL drafting plans, verifies the drafting item
    /// most likely to reach its deeper positions to its full depth, within
    /// the budget. Without this, positions the estimates never choose keep
    /// their initial acceptance forever.
    fn explore_depth(
        &mut self,
        candidates: &[PlanCandidate<'_>],
        drafts: &mut [usize],
        remaining: usize,
    ) {
        if self.explore > 0 {
            self.explore -= 1;
            return;
        }
        let left = remaining.saturating_sub(drafts.iter().sum());
        let deepest = candidates
            .iter()
            .zip(drafts.iter())
            .enumerate()
            .filter(|(_, (candidate, draft))| {
                **draft > 0 && **draft < self.executable_at_most(candidate.maximum_draft_tokens)
            })
            .max_by(|(_, (a, a_drafts)), (_, (b, b_drafts))| {
                a.acceptance
                    .survival(**a_drafts - 1)
                    .total_cmp(&b.acceptance.survival(**b_drafts - 1))
            })
            .map(|(index, _)| index);
        match deepest {
            // Every drafting item already verifies its full depth.
            None => self.explore = EXPLORE_INTERVAL - 1,
            Some(_) if left == 0 => {}
            Some(index) => {
                let deeper = self.executable_at_most(
                    (drafts[index] + left).min(candidates[index].maximum_draft_tokens),
                );
                if deeper > drafts[index] {
                    drafts[index] = deeper;
                    self.explore = EXPLORE_INTERVAL - 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(mode: SpeculationMode) -> SpeculationPolicy {
        policy_with_items(mode, None)
    }

    fn policy_with_items(mode: SpeculationMode, items: Option<usize>) -> SpeculationPolicy {
        let mut policy = SpeculationPolicy::new(
            SpeculationConfig::new(8, 64, mode, DraftSource::Bundle)
                .unwrap()
                .with_maximum_draft_items(items),
        );
        for counts in [&[0][..], &[0, 0][..]] {
            policy.observe_step(counts, 0, 1100);
            policy.observe_step(counts, 0, 1100);
        }
        policy
    }

    fn budget(tokens: usize, rows: usize) -> DraftBudget {
        DraftBudget::new(tokens, rows)
    }

    /// Sets the drafting cost of a step with these draft counts.
    /// Sets the drafting cost of a step with these draft counts: its time
    /// beyond the AR estimate of its rows.
    fn set_draft_cost(policy: &mut SpeculationPolicy, drafts: &[usize], micros: f64) {
        let shape = DraftShape::from_counts(drafts);
        let key = policy.config.draft_cost_model.key(drafts.len(), shape);
        match policy.config.draft_cost_model {
            DraftCostModel::PerToken => {
                policy.draft_costs.micros.insert(key, micros);
            }
            DraftCostModel::PerBlock => {
                let rows = policy.bucket(drafts.len() + shape.work());
                let total = policy.costs.estimate(rows) + micros;
                policy.block_steps.steps.insert(
                    (drafts.len(), rows),
                    BlockStep {
                        micros: total,
                        sequences: shape.0[0] as f64,
                    },
                );
            }
        }
    }

    /// A declared bound of 32 drafting items rotates across every item;
    /// without a declaration every item may draft in one step.
    #[test]
    fn wide_steps_bound_drafting_and_rotate_across_every_item() {
        const LIMIT: usize = 32;
        for items in [33, 64, 128, 256] {
            let mut bounded = policy_with_items(SpeculationMode::Fixed, Some(LIMIT));
            let estimate = AcceptanceEstimate::initial();
            let candidates = vec![
                PlanCandidate {
                    maximum_draft_tokens: 3,
                    acceptance: &estimate,
                    fresh: false,
                    remaining_tokens: usize::MAX,
                };
                items
            ];
            let mut seen = vec![false; items];
            for _ in 0..items.div_ceil(LIMIT) {
                let drafts = bounded.plan(&candidates, budget(1024, 1024));
                assert_eq!(drafts.iter().filter(|&&n| n > 0).count(), LIMIT);
                for (seen, count) in seen.iter_mut().zip(drafts) {
                    *seen |= count > 0;
                }
            }
            assert!(seen.into_iter().all(|seen| seen));
            let drafts = policy(SpeculationMode::Fixed).plan(&candidates, budget(1024, 1024));
            assert_eq!(drafts.iter().filter(|&&n| n > 0).count(), items);
        }
    }

    /// With the bound exceeded and too few spare rows for all the offered
    /// items, the budget still reaches every eligible item in turn: one
    /// spare row serves 64 eligible items within 64 steps, with ineligible
    /// items between them and the candidate set shrinking.
    #[test]
    fn a_tight_row_budget_still_rotates_drafting_across_every_item() {
        let estimate = AcceptanceEstimate::initial();
        let candidate = |eligible: bool| PlanCandidate {
            maximum_draft_tokens: if eligible { 3 } else { 0 },
            acceptance: &estimate,
            fresh: false,
            remaining_tokens: usize::MAX,
        };
        for (items, gap) in [(64, None), (96, Some(3))] {
            let candidates: Vec<_> = (0..items)
                .map(|index| candidate(gap.is_none_or(|gap| index % gap != 0)))
                .collect();
            let eligible = candidates
                .iter()
                .filter(|candidate| candidate.maximum_draft_tokens > 0)
                .count();
            let mut policy = policy_with_items(SpeculationMode::Fixed, Some(32));
            let mut drafted = vec![false; items];
            for _ in 0..eligible {
                let drafts = policy.plan(&candidates, budget(1, 1));
                assert_eq!(drafts.iter().filter(|&&n| n > 0).count(), 1);
                for (index, count) in drafts.into_iter().enumerate() {
                    if count > 0 {
                        assert!(candidates[index].maximum_draft_tokens > 0);
                        drafted[index] = true;
                    }
                }
            }
            // Within as many steps as eligible items, each drafted once.
            for (index, candidate) in candidates.iter().enumerate() {
                assert_eq!(
                    drafted[index],
                    candidate.maximum_draft_tokens > 0,
                    "{items} {index}"
                );
            }
        }
        // A candidate set that changes size still reaches every item it
        // always holds.
        let mut policy = policy_with_items(SpeculationMode::Fixed, Some(32));
        let mut served = std::collections::BTreeSet::new();
        for step in 0..80 {
            let candidates = vec![candidate(true); 40 - step % 5];
            let drafts = policy.plan(&candidates, budget(1, 1));
            served.extend(drafts.iter().position(|&count| count > 0));
        }
        assert!((0..36).all(|index| served.contains(&index)), "{served:?}");
    }

    #[test]
    fn fixed_mode_fills_items_in_order_within_the_budget() {
        let mut policy = policy(SpeculationMode::Fixed);
        let estimate = AcceptanceEstimate::initial();
        let candidates = [
            PlanCandidate {
                maximum_draft_tokens: 3,
                acceptance: &estimate,
                fresh: false,
                remaining_tokens: usize::MAX,
            },
            PlanCandidate {
                maximum_draft_tokens: 8,
                acceptance: &estimate,
                fresh: false,
                remaining_tokens: usize::MAX,
            },
            PlanCandidate {
                maximum_draft_tokens: 8,
                acceptance: &estimate,
                fresh: false,
                remaining_tokens: usize::MAX,
            },
        ];
        assert_eq!(policy.plan(&candidates, budget(10, 64)), [3, 7, 0]);
        assert_eq!(policy.plan(&candidates, budget(64, 5)), [3, 2, 0]);
    }

    #[test]
    fn adaptive_mode_drafts_deeper_for_requests_that_accept_more() {
        let mut policy = policy(SpeculationMode::Adaptive);
        let mut good = policy.prior();
        let mut poor = policy.prior();
        for _ in 0..64 {
            policy.observe(&mut good, 8, 8);
            policy.observe(&mut poor, 8, 0);
        }
        let candidates = [
            PlanCandidate {
                maximum_draft_tokens: 8,
                acceptance: &good,
                fresh: false,
                remaining_tokens: usize::MAX,
            },
            PlanCandidate {
                maximum_draft_tokens: 8,
                acceptance: &poor,
                fresh: false,
                remaining_tokens: usize::MAX,
            },
        ];
        let drafts = policy.plan(&candidates, budget(64, 64));
        assert_eq!(drafts[1], 0);
        assert!(drafts[0] >= 6, "{drafts:?}");
    }

    #[test]
    fn adaptive_mode_stops_when_rows_cost_more_than_they_return() {
        let mut policy = policy(SpeculationMode::Adaptive);
        policy.costs = StepCostTable::default();
        // One row costs 1 ms and every added row another 1 ms, so a draft
        // pays only if it is accepted more often than it costs.
        for rows in 1..=16 {
            policy.costs.record(rows, 1000 * rows as u64);
        }
        let estimate = policy.prior();
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 8,
            acceptance: &estimate,
            fresh: false,
            remaining_tokens: usize::MAX,
        }];
        assert_eq!(policy.plan(&candidates, budget(64, 64)), [0]);

        // With nearly free rows the same estimate drafts to the limit.
        let mut cheap = self::policy(SpeculationMode::Adaptive);
        cheap.costs = StepCostTable::default();
        for rows in 1..=16 {
            cheap.costs.record(rows, 1000 + rows as u64);
        }
        assert_eq!(cheap.plan(&candidates, budget(64, 64)), [8]);
    }

    #[test]
    fn draft_wall_time_can_disable_speculation_without_polluting_ar_cost() {
        let mut policy = policy(SpeculationMode::Adaptive);
        let estimate = policy.prior();
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 3,
            acceptance: &estimate,
            fresh: false,
            remaining_tokens: usize::MAX,
        }];
        assert!(policy.plan(&candidates, budget(32, 32))[0] > 0);
        let baseline = policy.costs.clone();
        policy.observe_step(&[3], 0, 10_000);
        policy.observe_step(&[3], 0, 10_000);
        assert_eq!(policy.costs, baseline);
        assert_eq!(policy.plan(&candidates, budget(32, 32)), [0]);
    }

    #[test]
    fn measured_drafting_pays_at_c1_but_not_c32_with_low_acceptance() {
        let mut policy = policy(SpeculationMode::Adaptive);
        policy.costs = StepCostTable::default();
        for rows in 1..=128 {
            policy.costs.record(rows, 1000 + 10 * rows as u64);
        }
        policy.observe_step(&[3], 0, 1190);
        policy.observe_step(&[3], 0, 1190);
        let good = policy.prior();
        let one = [PlanCandidate {
            maximum_draft_tokens: 3,
            acceptance: &good,
            fresh: false,
            remaining_tokens: usize::MAX,
        }];
        assert!(policy.plan(&one, budget(96, 96))[0] > 0);
        let poor = AcceptanceEstimate {
            positions: [0.1; MAX_DRAFT_TOKENS],
        };
        let many = [PlanCandidate {
            maximum_draft_tokens: 3,
            acceptance: &poor,
            fresh: false,
            remaining_tokens: usize::MAX,
        }; 32];
        assert_eq!(policy.plan(&many, budget(96, 96)), vec![0; 32]);
    }

    #[test]
    fn draft_shapes_distinguish_depth_from_width_and_average_observations() {
        let mut costs = DraftCostTable::default();
        let model = DraftCostModel::PerToken;
        let deep = DraftShape::from_counts(&[3]);
        let wide = DraftShape::from_counts(&[1, 1, 1]);
        costs.record(model.key(1, deep), 300.0, COST_WEIGHT);
        costs.record(model.key(3, wide), 100.0, COST_WEIGHT);
        costs.record(model.key(1, deep), 500.0, COST_WEIGHT);
        let bucket = |rows: usize| rows.next_power_of_two();
        assert_eq!(costs.estimate(1, deep, bucket), 350.0);
        assert_eq!(costs.estimate(3, wide, bucket), 100.0);
        assert_eq!(costs.estimate(1, DraftShape::default(), bucket), 0.0);
        // Other batch sizes scale the nearest shape's drafting by work.
        assert_eq!(
            costs.estimate(2, DraftShape::from_counts(&[1, 1]), bucket),
            200.0 / 3.0
        );
        // At three items one K3 and three K1 drafts both run the 8-row
        // graph: the measured shape prices the other until it runs.
        assert_eq!(costs.estimate(3, deep, bucket), 100.0);
    }

    #[test]
    fn a_shape_in_a_measured_bucket_keeps_its_cheaper_target() {
        let mut costs = DraftCostTable::default();
        let model = DraftCostModel::PerToken;
        let bucket = |rows: usize| rows.next_power_of_two();
        // Sixteen K1 items ran the 32-row graph 800 us under its AR
        // estimate; four K3 items ran the 16-row graph 1.9 ms under it.
        let k1 = DraftShape::from_counts(&[1; 16]);
        costs.record(model.key(16, k1), -800.0, 1.0);
        costs.record(
            model.key(4, DraftShape::from_counts(&[3; 4])),
            -1_900.0,
            1.0,
        );
        // A K3 mix at 16 items runs the same 32-row graph as K1.
        let mix = DraftShape::from_counts(&[3, 3, 3, 3, 3, 1]);
        assert_eq!(costs.estimate(16, mix, bucket), -800.0);
        // In another bucket or at other items, only positive drafting time
        // carries over.
        assert_eq!(
            costs.estimate(8, DraftShape::from_counts(&[3; 4]), bucket),
            0.0
        );
    }

    #[test]
    fn adaptive_obtains_an_ar_baseline_before_drafting() {
        let mut policy = SpeculationPolicy::new(
            SpeculationConfig::new(3, 32, SpeculationMode::Adaptive, DraftSource::Host).unwrap(),
        );
        let estimate = policy.prior();
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 3,
            acceptance: &estimate,
            fresh: false,
            remaining_tokens: usize::MAX,
        }];
        assert_eq!(policy.plan(&candidates, budget(32, 32)), [0]);
        policy.observe_step(&[0], 0, 1000);
        assert_eq!(policy.plan(&candidates, budget(32, 32)), [0]);
        policy.observe_step(&[0], 0, 1000);
        assert!(policy.plan(&candidates, budget(32, 32))[0] > 0);
    }

    #[test]
    fn cold_graph_capture_does_not_permanently_disable_profitable_drafts() {
        let mut policy = SpeculationPolicy::new(
            SpeculationConfig::new(3, 32, SpeculationMode::Adaptive, DraftSource::Bundle).unwrap(),
        );
        let estimate = policy.prior();
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 3,
            acceptance: &estimate,
            fresh: false,
            remaining_tokens: usize::MAX,
        }];
        // Both AR and draft shapes can pay one-time capture costs.
        policy.observe_step(&[0], 0, 100_000);
        assert_eq!(policy.plan(&candidates, budget(32, 32)), [0]);
        policy.observe_step(&[0], 0, 1000);
        assert_eq!(policy.costs.estimate(1), 1000.0);
        assert!(policy.plan(&candidates, budget(32, 32))[0] > 0);
        policy.observe_step(&[3], 0, 100_000);
        assert!(policy.plan(&candidates, budget(32, 32))[0] > 0);
        policy.observe_step(&[3], 0, 1500);
        assert!(policy.plan(&candidates, budget(32, 32))[0] > 0);
        // A genuinely slow warmed drafter still switches off.
        policy.observe_step(&[3], 0, 100_000);
        assert_eq!(policy.plan(&candidates, budget(32, 32)), [0]);
    }

    #[test]
    fn a_flat_block_cost_can_pay_for_seven_drafts_when_one_does_not() {
        let mut policy = policy(SpeculationMode::Adaptive);
        for rows in 1..=8 {
            policy
                .costs
                .micros
                .insert(rows, 1000.0 + 100.0 * (rows - 1) as f64);
        }
        // One measured pass; per-token estimates scale it by depth.
        set_draft_cost(&mut policy, &[1], 1000.0);
        let estimate = AcceptanceEstimate {
            positions: [0.9; MAX_DRAFT_TOKENS],
        };
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 7,
            acceptance: &estimate,
            fresh: false,
            remaining_tokens: usize::MAX,
        }];
        assert_eq!(policy.plan(&candidates, budget(7, 7)), [0]);
        policy.config = policy
            .config
            .with_draft_cost_model(DraftCostModel::PerBlock);
        // Production policy configuration is immutable. This test switches
        // cost models in place, so clear the previous model's cooldown. A
        // measured K7 block costs the same pass.
        policy.zero_draft[1] = ZeroDraftState::default();
        policy.block_steps = BlockStepTable::default();
        set_draft_cost(&mut policy, &[7], 1000.0);
        assert_eq!(policy.plan(&candidates, budget(7, 7)), [7]);
        assert_eq!(policy.plan(&candidates, budget(4, 4)), [4]);
    }

    #[test]
    fn block_steps_are_priced_by_their_row_bucket() {
        let mut ar = StepCostTable::default();
        for rows in [1, 16, 32] {
            ar.record(rows, 4_000 + 250 * (rows as u64 - 1));
        }
        let sequence = 250.0;
        let mut steps = BlockStepTable::default();
        // Nothing drafted anywhere: the AR estimate of the bucket's rows.
        assert_eq!(steps.estimate(1, 8, 1, &ar, sequence), ar.estimate(8));
        // C32 measured its full K7 block in the 256-row bucket.
        steps.record((32, 256), 53_000.0, 32, COST_WEIGHT);
        assert_eq!(steps.estimate(32, 256, 32, &ar, sequence), 53_000.0);
        // A partial block in the same bucket runs the same rows: it costs no
        // less, whatever rows it leaves empty.
        assert_eq!(steps.estimate(32, 256, 20, &ar, sequence), 53_000.0);
        // Narrower buckets lend from it at 1.25 per halving, not from AR
        // rows the wide step never ran.
        assert!((steps.estimate(32, 64, 4, &ar, sequence) - 53_000.0 / 1.5625).abs() < 1e-6);
        // Once measured, a narrower bucket prices a wider one too, and the
        // lower of the two estimates stands.
        steps.record((32, 64), 26_000.0, 1, COST_WEIGHT);
        let wide = steps.estimate(32, 128, 8, &ar, sequence);
        let from_64 = (26_000.0 + 7.0 * sequence) * WIDER_BUCKET_RATIO;
        assert!(
            (wide - from_64.min(53_000.0 / NARROWER_BUCKET_RATIO)).abs() < 1e-6,
            "{wide}"
        );
        // More sequences than a bucket averaged add their draft passes.
        assert_eq!(
            steps.estimate(32, 64, 5, &ar, sequence),
            26_000.0 + 4.0 * sequence
        );
        // Without a drafting measurement at C16, C32 lends its overhead over
        // the AR estimate of a bucket, never a negative one.
        // Its nearest bucket, 64 rows, lends it; C16 drafts 15 more
        // sequences than that step averaged.
        let overhead = (26_000.0 + 15.0 * sequence - ar.estimate(64)).max(0.0);
        assert_eq!(
            steps.estimate(16, 128, 16, &ar, sequence),
            ar.estimate(128) + overhead
        );
        // Observations average; a probe's weight replaces.
        steps.record((32, 256), 57_000.0, 32, COST_WEIGHT);
        assert_eq!(steps.estimate(32, 256, 32, &ar, sequence), 54_000.0);
        steps.record((32, 256), 50_000.0, 32, 1.0);
        assert_eq!(steps.estimate(32, 256, 32, &ar, sequence), 50_000.0);
    }

    #[test]
    fn observations_update_only_verified_positions() {
        let mut estimate = AcceptanceEstimate::initial();
        estimate.observe(4, 2, 0.5);
        assert_eq!(estimate.positions[0], 0.8);
        assert_eq!(estimate.positions[1], 0.8);
        assert_eq!(estimate.positions[2], 0.3);
        assert_eq!(estimate.positions[3], INITIAL_ACCEPTANCE);
        estimate.observe(2, 2, 0.5);
        assert_eq!(estimate.positions[2], 0.3);
    }

    #[test]
    fn cost_estimates_scale_the_nearest_measurement() {
        let mut table = StepCostTable::default();
        assert_eq!(table.estimate(0), 1.0);
        table.record(10, 2000);
        assert_eq!(table.estimate(10), 2000.0);
        assert!((table.estimate(20) - 2000.0 * 3.0 / 2.0).abs() < 1e-9);
        table.record(10, 4000);
        assert_eq!(table.estimate(10), 2500.0);
    }

    #[test]
    fn zero_draft_hysteresis_requires_a_sustained_gain_and_is_per_concurrency() {
        for model in [DraftCostModel::PerToken, DraftCostModel::PerBlock] {
            let mut policy = policy(SpeculationMode::Adaptive);
            policy.config = policy.config.with_draft_cost_model(model);
            for rows in 1..=4 {
                policy.costs.micros.insert(rows, 1000.0);
            }
            let estimate = policy.prior();
            let candidates = [PlanCandidate {
                maximum_draft_tokens: 1,
                acceptance: &estimate,
                fresh: false,
                remaining_tokens: usize::MAX,
            }];
            set_draft_cost(&mut policy, &[1], 700.0);
            assert_eq!(policy.plan(&candidates, budget(1, 1)), [0]);
            // A marginal positive estimate must not flap back into drafting.
            set_draft_cost(&mut policy, &[1], 550.0);
            for _ in 0..8 {
                assert_eq!(policy.plan(&candidates, budget(1, 1)), [0]);
            }
            // A gain that clears the margin resumes on the next plan.
            set_draft_cost(&mut policy, &[1], 400.0);
            assert_eq!(policy.plan(&candidates, budget(1, 1)), [1]);
            // Once active, a smaller positive gain can keep drafting enabled.
            set_draft_cost(&mut policy, &[1], 550.0);
            assert_eq!(policy.plan(&candidates, budget(1, 1)), [1]);
            set_draft_cost(&mut policy, &[1], 700.0);
            assert_eq!(policy.plan(&candidates, budget(1, 1)), [0]);
            set_draft_cost(&mut policy, &[1], 200.0);
            let other = [
                candidates[0],
                PlanCandidate {
                    maximum_draft_tokens: 0,
                    acceptance: &estimate,
                    fresh: false,
                    remaining_tokens: usize::MAX,
                },
            ];
            assert_eq!(policy.plan(&other, budget(1, 1)), [1, 0]);
        }
    }

    /// A block drafter over the shared-device SPEC-VAL step times: AR steps
    /// of 1, 4, 16 and 32 rows, and a measured one-sequence block.
    fn block_policy() -> SpeculationPolicy {
        let mut policy = SpeculationPolicy::new(
            SpeculationConfig::new(7, 32, SpeculationMode::Adaptive, DraftSource::Bundle)
                .unwrap()
                .with_draft_cost_model(DraftCostModel::PerBlock),
        );
        for (rows, micros) in [(1, 4500), (4, 8000), (16, 14_800), (32, 19_700)] {
            policy.costs.record(rows, micros);
        }
        set_draft_cost(&mut policy, &[1], 1300.0);
        policy
    }

    /// Plans `steps` decode steps of `items` requests that accept each draft
    /// with probability `truth` once every earlier one was accepted, and
    /// charges each step `cost(drafts)`. Returns the drafts of every step.
    fn simulate(
        policy: &mut SpeculationPolicy,
        items: usize,
        steps: usize,
        truth: f64,
        cost: impl Fn(&[usize]) -> u64,
    ) -> Vec<Vec<usize>> {
        let mut estimates = vec![policy.prior(); items];
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut unit = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut history = Vec::new();
        for _ in 0..steps {
            let candidates = estimates
                .iter()
                .map(|acceptance| PlanCandidate {
                    maximum_draft_tokens: 7,
                    acceptance,
                    fresh: false,
                    remaining_tokens: usize::MAX,
                })
                .collect::<Vec<_>>();
            let drafts = policy.plan(&candidates, budget(1024, 32 - items));
            for (estimate, &verified) in estimates.iter_mut().zip(&drafts) {
                if verified > 0 {
                    let accepted = (0..verified).take_while(|_| unit() < truth).count();
                    policy.observe(estimate, verified, accepted);
                }
            }
            policy.observe_step(&drafts, 0, cost(&drafts));
            history.push(drafts);
        }
        history
    }

    #[test]
    fn c1_learns_deep_positions_and_their_verify_row_cost() {
        let mut policy = block_policy();
        // A C1 verify step costs its block and a little per row; AR steps of
        // four items (the nearest measured rows) overestimate those rows.
        // Deep positions start untried at the 0.6 prior.
        let history = simulate(&mut policy, 1, 800, 0.85, |drafts| {
            4500 + if drafts[0] > 0 {
                1300 + 60 * drafts[0] as u64
            } else {
                0
            }
        });
        assert!(
            policy.prior.positions[..7].iter().all(|&p| p > 0.75),
            "{:?}",
            policy.prior.positions
        );
        assert!(
            history[700..].iter().all(|drafts| drafts == &[7]),
            "{:?}",
            &history[700..]
        );
    }

    #[test]
    fn a_joint_block_plan_pays_at_c4_where_no_single_item_does() {
        let mut policy = block_policy();
        let acceptance = AcceptanceEstimate {
            positions: [0.75; MAX_DRAFT_TOKENS],
        };
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 7,
            acceptance: &acceptance,
            fresh: false,
            remaining_tokens: usize::MAX,
        }; 4];
        // One item's block and verify rows barely repay its expected drafts,
        // so a single addition cannot clear the resume margin.
        let empty = Plan {
            drafts: vec![0; 4],
            rows: 4,
            catch_up: 0.0,
            expected: 4.0,
            cost: policy.costs.estimate(4),
        };
        assert_eq!(
            policy
                .extend(&candidates, &[0.0; 4], empty, 28, RESUME_GAIN)
                .drafts,
            [0; 4]
        );
        // Four items share one block.
        let drafts = policy.plan(&candidates, budget(1024, 28));
        assert!(
            drafts.iter().all(|&count| count >= 5 && count == drafts[0]),
            "{drafts:?}"
        );

        // Measured C4 steps: rows cost as AR steps between the measured
        // sizes, plus one pass for every drafting sequence. With 0.75
        // acceptance fixed K7 would gain 33% here; the learned plans must
        // gain at least 30% in expectation.
        let cost = |drafts: &[usize]| {
            let rows = 4 + drafts.iter().sum::<usize>() as u64;
            let sequences = drafts.iter().filter(|&&count| count > 0).count() as u64;
            let block = if sequences > 0 {
                1300 + 217 * (sequences - 1)
            } else {
                0
            };
            8000 + 418 * (rows - 4) + block
        };
        let mut policy = block_policy();
        let history = simulate(&mut policy, 4, 400, 0.75, cost);
        let rate = |drafts: &[usize]| {
            let tokens = drafts
                .iter()
                .map(|&count| 1.0 + (1..=count).map(|j| 0.75_f64.powi(j as i32)).sum::<f64>())
                .sum::<f64>();
            tokens / cost(drafts) as f64
        };
        let ar = rate(&[0; 4]);
        assert!((rate(&[7; 4]) / ar - 1.33).abs() < 0.01);
        let tail = &history[300..];
        let gain = tail.iter().map(|drafts| rate(drafts)).sum::<f64>() / tail.len() as f64 / ar;
        assert!(gain >= 1.3, "{gain} {tail:?}");
    }

    #[test]
    fn high_concurrency_learns_that_drafting_loses_and_stays_at_zero() {
        let mut policy = block_policy();
        // At C16 verify steps cost a quarter more than AR steps of the same
        // rows, as fixed K7 measured (-11% to -25%).
        let history = simulate(&mut policy, 16, 400, 0.75, |drafts| {
            let rows = 16 + drafts.iter().sum::<usize>();
            let sequences = drafts.iter().filter(|&&count| count > 0).count();
            let ar = 14_800 + (rows - 16) as u64 * 306;
            if sequences == 0 {
                ar
            } else {
                ar * 5 / 4 + 1300 + 150 * sequences as u64
            }
        });
        let drafting = history
            .iter()
            .filter(|drafts| drafts.iter().any(|&count| count > 0))
            .count();
        assert!(drafting <= 12, "{drafting} drafting steps");
        // Past step 200 only the probe after PROBE_INTERVAL plans drafts,
        // twice when its first run is discarded as a warm-up.
        let late = history[200..]
            .iter()
            .filter(|drafts| drafts.iter().any(|&count| count > 0))
            .count();
        assert!(late <= 2, "{late} late drafting steps");
        // C32 leaves no spare row within the 32-row cap.
        let acceptance = policy.prior();
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 7,
            acceptance: &acceptance,
            fresh: false,
            remaining_tokens: usize::MAX,
        }; 32];
        assert_eq!(policy.plan(&candidates, budget(1024, 0)), [0; 32]);
    }

    #[test]
    fn exploration_deepens_one_drafting_item_every_interval() {
        let mut policy = block_policy();
        for rows in 2..=8 {
            policy.costs.record(rows, 4500 + 90 * (rows as u64 - 1));
        }
        let shallow = AcceptanceEstimate {
            positions: [
                0.9, 0.9, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                0.05, 0.05,
            ],
        };
        let never = AcceptanceEstimate {
            positions: [0.0; MAX_DRAFT_TOKENS],
        };
        let candidates = [
            PlanCandidate {
                maximum_draft_tokens: 7,
                acceptance: &never,
                fresh: false,
                remaining_tokens: usize::MAX,
            },
            PlanCandidate {
                maximum_draft_tokens: 7,
                acceptance: &shallow,
                fresh: false,
                remaining_tokens: usize::MAX,
            },
        ];
        let plans = (0..2 * usize::from(EXPLORE_INTERVAL))
            .map(|_| policy.plan(&candidates, budget(1024, 5)))
            .collect::<Vec<_>>();
        let normal = &plans[0];
        assert_eq!(normal[0], 0);
        assert!(normal[1] > 0 && normal[1] < 5, "{normal:?}");
        let explored = plans
            .iter()
            .enumerate()
            .filter(|(_, drafts)| *drafts != normal)
            .collect::<Vec<_>>();
        // Only the drafting item deepens, within the five spare rows.
        assert_eq!(explored.len(), 2, "{plans:?}");
        for (index, drafts) in explored {
            assert_eq!(index % usize::from(EXPLORE_INTERVAL), 15);
            assert_eq!(drafts, &[0, 5]);
        }
    }

    #[test]
    fn a_prior_position_averages_its_first_observations() {
        let mut policy = policy(SpeculationMode::Adaptive);
        let mut estimate = policy.prior();
        policy.observe(&mut estimate, 3, 3);
        assert_eq!(policy.prior.positions[0], 0.8);
        policy.observe(&mut estimate, 3, 3);
        assert!((policy.prior.positions[0] - 0.6 / 3.0 - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(policy.prior.positions[3], INITIAL_ACCEPTANCE);
        for _ in 0..200 {
            policy.observe(&mut estimate, 1, 0);
        }
        // Past 63 observations each one moves the prior by PRIOR_WEIGHT.
        let before = policy.prior.positions[0];
        policy.observe(&mut estimate, 1, 1);
        assert!((policy.prior.positions[0] - before - PRIOR_WEIGHT * (1.0 - before)).abs() < 1e-12);
    }

    /// Verify row buckets of 2, 4 and 8 rows, as the executor rounds a
    /// proposed draft count.
    fn round_to_row_buckets(count: usize) -> usize {
        [2, 4, 8]
            .into_iter()
            .filter(|rows| *rows <= count + 1)
            .max()
            .map_or(count.min(1), |rows| rows - 1)
    }

    #[test]
    fn plans_hold_only_draft_counts_the_executor_runs() {
        let mut policy = block_policy().with_draft_rounding(round_to_row_buckets);
        assert_eq!(policy.executable_at_most(6), 3);
        assert_eq!(policy.executable_at_most(2), 1);
        assert_eq!(policy.executable_at_most(9), 7);
        let acceptance = AcceptanceEstimate {
            positions: [0.9; MAX_DRAFT_TOKENS],
        };
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 7,
            acceptance: &acceptance,
            fresh: false,
            remaining_tokens: usize::MAX,
        }; 4];
        for spare in 1..=28 {
            let drafts = policy.plan(&candidates, budget(1024, spare));
            assert!(drafts.iter().sum::<usize>() <= spare, "{spare} {drafts:?}");
            assert!(
                drafts.iter().all(|count| [0, 1, 3, 7].contains(count)),
                "{spare} {drafts:?}"
            );
        }
    }

    #[test]
    fn c16_does_not_draft_at_a_loss_when_the_executor_rounds_counts() {
        // Measured TP4 serving: an AR C16 step took 8.5 ms, and any drafting
        // at C16 ran the 32-row graph (11.55 ms) plus the draft block. Counts
        // the policy chose were rounded down to K1, K3 or K7 after planning,
        // so it priced K2, K5 and K6 plans it never ran and never measured,
        // and settled on K3 plans that emit less per second than AR.
        let cost = |drafts: &[usize]| {
            if drafts.iter().all(|&count| count == 0) {
                8_520
            } else {
                11_550 + 2_300
            }
        };
        // Earlier cohorts measured AR at C1 and C32, and K7 at C1 with every
        // draft accepted.
        let run = |mut policy: SpeculationPolicy| {
            let mut c1 = policy.prior();
            for _ in 0..4 {
                policy.observe_step(&[0], 0, 4_020);
                policy.observe_step(&[0; 32], 0, 11_550);
            }
            for _ in 0..70 {
                policy.observe_step(&[7], 0, 10_300);
                policy.observe(&mut c1, 7, 7);
            }
            let items = 16;
            let mut estimates = vec![policy.prior(); items];
            let mut state = 0x2545_f491_4f6c_dd1d_u64;
            let mut tokens = 0.0;
            let mut micros = 0.0;
            for step in 0..2_000 {
                let candidates = estimates
                    .iter()
                    .map(|acceptance| PlanCandidate {
                        maximum_draft_tokens: 7,
                        acceptance,
                        fresh: false,
                        remaining_tokens: usize::MAX,
                    })
                    .collect::<Vec<_>>();
                let mut drafts = policy.plan(&candidates, budget(1024, 32 - items));
                for count in &mut drafts {
                    *count = round_to_row_buckets(*count).min(*count);
                }
                let mut emitted = items as f64;
                for (estimate, &verified) in estimates.iter_mut().zip(&drafts) {
                    let accepted = (0..verified)
                        .take_while(|_| {
                            state = state
                                .wrapping_mul(6_364_136_223_846_793_005)
                                .wrapping_add(1_442_695_040_888_963_407);
                            ((state >> 11) as f64 / (1_u64 << 53) as f64) < 0.95
                        })
                        .count();
                    if verified > 0 {
                        policy.observe(estimate, verified, accepted);
                    }
                    emitted += accepted as f64;
                }
                policy.observe_step(&drafts, 0, cost(&drafts));
                if step >= 1_000 {
                    tokens += emitted;
                    micros += cost(&drafts) as f64;
                }
            }
            tokens / micros / (16.0 / 8_520.0)
        };
        let policy = || {
            SpeculationPolicy::new(
                SpeculationConfig::new(7, 32, SpeculationMode::Adaptive, DraftSource::Bundle)
                    .unwrap()
                    .with_draft_cost_model(DraftCostModel::PerBlock),
            )
        };
        let unaware = run(policy());
        let aware = run(policy()
            .with_draft_rounding(round_to_row_buckets)
            .with_step_rows(usize::next_power_of_two));
        // The realized throughput guard keeps even the policy that ignores
        // rounding at AR once its drafting steps lose.
        assert!(unaware >= 0.99, "{unaware}");
        assert!(aware >= 0.999, "{aware}");
    }

    /// One phase of a synthetic serving trace: `items` requests decode for
    /// `steps` steps with `spare_rows` decode-phase rows beyond one per item,
    /// accept each draft with probability `truth` once every earlier one was
    /// accepted, and each step costs `cost(drafts, previous step's drafts)`.
    struct Phase<'a> {
        items: usize,
        steps: usize,
        spare_rows: usize,
        truth: f64,
        cost: &'a dyn Fn(&[usize], &[usize]) -> u64,
    }

    /// Runs `phases` in order through one policy whose executor rounds
    /// counts to 2, 4 and 8 row buckets. Returns each phase's realized
    /// tokens per microsecond over its second half, relative to AR.
    fn serve(policy: &mut SpeculationPolicy, phases: &[Phase<'_>]) -> Vec<f64> {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut unit = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        phases
            .iter()
            .map(|phase| {
                let mut estimates = vec![policy.prior(); phase.items];
                let mut previous = vec![0; phase.items];
                let (mut tokens, mut micros) = (0.0, 0.0);
                for step in 0..phase.steps {
                    let candidates = estimates
                        .iter()
                        .zip(&previous)
                        .map(|(acceptance, &before)| PlanCandidate {
                            maximum_draft_tokens: 7,
                            acceptance,
                            fresh: before == 0,
                            remaining_tokens: usize::MAX,
                        })
                        .collect::<Vec<_>>();
                    let mut drafts = policy.plan(&candidates, budget(1024, phase.spare_rows));
                    for count in &mut drafts {
                        *count = round_to_row_buckets(*count).min(*count);
                    }
                    let mut emitted = phase.items as f64;
                    for (estimate, &verified) in estimates.iter_mut().zip(&drafts) {
                        let accepted = (0..verified).take_while(|_| unit() < phase.truth).count();
                        if verified > 0 {
                            policy.observe(estimate, verified, accepted);
                        }
                        emitted += accepted as f64;
                    }
                    let cost = (phase.cost)(&drafts, &previous);
                    let fresh = drafts
                        .iter()
                        .zip(&previous)
                        .filter(|&(&count, &before)| count > 0 && before == 0)
                        .count();
                    policy.observe_step(&drafts, fresh, cost);
                    if step >= phase.steps / 2 {
                        tokens += emitted;
                        micros += cost as f64;
                    }
                    previous = drafts;
                }
                let zero = vec![0; phase.items];
                let ar = phase.items as f64 / (phase.cost)(&zero, &zero) as f64;
                tokens / micros / ar
            })
            .collect()
    }

    /// Exclusive TP4 AR step times on 2026-09-25 (4.02, 8.52 and
    /// 11.55 ms at 1, 16 and 32 rows), with the buckets between them
    /// interpolated; a step runs its power-of-two row bucket.
    fn g9_ar(rows: usize) -> f64 {
        match rows.next_power_of_two() {
            1 => 4_020.0,
            2 => 4_400.0,
            4 => 5_000.0,
            8 => 6_300.0,
            16 => 8_520.0,
            _ => 11_550.0,
        }
    }

    /// A drafting step runs the target at its row bucket (`target`), the
    /// draft block (1.29 ms for one sequence, 2.3 ms at 16) and a catch-up
    /// per drafting sequence: 0.12 ms after a drafting step, 2 ms after a
    /// gap (the long-gap catch-up runs eagerly).
    fn g9_cost(target: impl Fn(usize) -> f64) -> impl Fn(&[usize], &[usize]) -> u64 {
        move |drafts: &[usize], previous: &[usize]| {
            let items = drafts.len();
            let rows = items + drafts.iter().sum::<usize>();
            let sequences = drafts.iter().filter(|&&count| count > 0).count();
            if sequences == 0 {
                return g9_ar(items) as u64;
            }
            let block = 1_290.0 + 1_010.0 * (sequences - 1) as f64 / 15.0;
            let catch_up = drafts
                .iter()
                .zip(previous)
                .filter(|(count, _)| **count > 0)
                .map(|(_, &before)| if before > 0 { 120.0 } else { 2_000.0 })
                .sum::<f64>();
            (target(rows.next_power_of_two()) + block + catch_up) as u64
        }
    }

    /// Verify targets by row bucket. On `bench.py`'s repetitive text more
    /// rows read few more experts (G9: 10.3 ms for a C1 K7 step, 11.55 ms at
    /// 32 rows); on diverse text every row reads its own experts.
    fn bench_target(bucket: usize) -> f64 {
        match bucket {
            1..=8 => 8_560.0,
            16 => 9_500.0,
            32 => 11_550.0,
            64 => 14_000.0,
            128 => 19_000.0,
            _ => 30_000.0,
        }
    }

    fn diverse_target(bucket: usize) -> f64 {
        match bucket {
            1..=8 => 8_560.0,
            16 => 10_000.0,
            32 => 13_000.0,
            64 => 20_000.0,
            128 => 33_000.0,
            _ => 55_000.0,
        }
    }

    fn block_policy_with_cap(cap: usize) -> SpeculationPolicy {
        SpeculationPolicy::new(
            SpeculationConfig::new(7, cap, SpeculationMode::Adaptive, DraftSource::Bundle)
                .unwrap()
                .with_draft_cost_model(DraftCostModel::PerBlock),
        )
        .with_draft_rounding(round_to_row_buckets)
        .with_step_rows(usize::next_power_of_two)
    }

    /// C1, C4, C16 and C32 cohorts, three times over.
    fn cohorts<'a>(
        cap: usize,
        truth: f64,
        cost: &'a dyn Fn(&[usize], &[usize]) -> u64,
    ) -> Vec<Phase<'a>> {
        (0..3)
            .flat_map(|_| [1, 4, 16, 32])
            .map(|items| Phase {
                items,
                steps: 600,
                spare_rows: cap.saturating_sub(items),
                truth,
                cost,
            })
            .collect()
    }

    #[test]
    fn serving_traces_never_draft_below_ar_and_gain_where_rows_allow() {
        let bench = g9_cost(bench_target);
        let diverse = g9_cost(diverse_target);
        for (name, cap, truth, cost, gains) in [
            // C1 and C4 gain; C16 at most breaks even within 32 rows.
            (
                "bench, 32 rows",
                32,
                1.0,
                &bench as &dyn Fn(&[usize], &[usize]) -> u64,
                [2.5, 2.5, 1.0, 1.0],
            ),
            // More rows let C16 and C32 draft deep, in every cohort: the
            // catch-up at a cohort's start does not count as a steady cost.
            ("bench, 256 rows", 256, 1.0, &bench, [2.5, 2.5, 2.8, 2.4]),
            // Nothing pays at 0.6 acceptance on diverse text.
            ("diverse, 256 rows", 256, 0.6, &diverse, [1.0; 4]),
            // At 0.8 C1 and C4 pay, which the 0.6 prior hides at first.
            (
                "diverse 0.8, 256 rows",
                256,
                0.8,
                &diverse,
                [1.5, 1.3, 1.0, 1.0],
            ),
        ] {
            let mut policy = block_policy_with_cap(cap);
            let rates = serve(&mut policy, &cohorts(cap, truth, cost));
            for (phase, rate) in rates.iter().enumerate() {
                let floor = gains[phase % 4];
                // Never below AR beyond 2%; at least the scenario's gain.
                assert!(*rate >= 0.98 && *rate >= floor * 0.98, "{name}: {rates:?}");
            }
        }
    }

    /// The G9 C16 regression: an AR step of 8.52 ms, and a drafting step of
    /// about 16.8 ms (the 32-row graph, the draft block and catch-up) that
    /// emits at most 32 tokens, after C1 measured cheap K7 steps.
    #[test]
    fn c16_verify_steps_are_priced_from_their_own_measurements() {
        let cost = |drafts: &[usize], _: &[usize]| {
            let items = drafts.len();
            let drafting = drafts.iter().any(|&count| count > 0);
            match (items, drafting) {
                (_, false) => g9_ar(items) as u64,
                (1, true) => 10_300,
                (_, true) => 16_800,
            }
        };
        let mut policy = block_policy_with_cap(32);
        let rates = serve(&mut policy, &cohorts(32, 1.0, &cost));
        for (phase, rate) in rates.iter().enumerate() {
            match phase % 4 {
                0 => assert!(*rate > 2.9, "{rates:?}"),
                _ => assert!(*rate >= 0.99, "{rates:?}"),
            }
        }
    }

    /// Plans `steps` steps of `items` requests whose freshness follows the
    /// previous plan, charging `cost(drafting, fresh)`; returns the drafting
    /// step indices.
    fn serve_fresh(
        policy: &mut SpeculationPolicy,
        items: usize,
        steps: usize,
        cost: impl Fn(bool, usize) -> u64,
    ) -> Vec<usize> {
        let acceptance = AcceptanceEstimate {
            positions: [0.99; MAX_DRAFT_TOKENS],
        };
        let mut previous = vec![0; items];
        let mut drafted = Vec::new();
        for step in 0..steps {
            let candidates = previous
                .iter()
                .map(|&before| PlanCandidate {
                    maximum_draft_tokens: 7,
                    acceptance: &acceptance,
                    fresh: before == 0,
                    remaining_tokens: usize::MAX,
                })
                .collect::<Vec<_>>();
            let drafts = policy.plan(&candidates, budget(1024, 256 - items));
            let fresh = drafts
                .iter()
                .zip(&previous)
                .filter(|&(&count, &before)| count > 0 && before == 0)
                .count();
            let drafting = drafts.iter().any(|&count| count > 0);
            // Every draft is accepted.
            for &count in drafts.iter().filter(|&&count| count > 0) {
                let mut estimate = acceptance.clone();
                policy.observe(&mut estimate, count, count);
            }
            policy.observe_step(&drafts, fresh, cost(drafting, fresh));
            if drafting {
                drafted.push(step);
            }
            previous = drafts;
        }
        drafted
    }

    #[test]
    fn a_probe_refreshes_a_stale_catch_up_estimate() {
        let mut policy = block_policy_with_cap(256);
        for items in [1, 4] {
            // Warm AR at 1 ms and K7 at 1.5 ms, then one fresh step whose
            // catch-up took 200 ms, as in a transient slowdown.
            let steady = vec![7; items];
            for _ in 0..2 {
                policy.observe_step(&vec![0; items], 0, 1000);
                policy.observe_step(&steady, 0, 1500);
            }
            policy.observe_step(&steady, items, 201_500);
            // The slowdown is over: drafting and catching up cost 1.5 ms.
            let drafted = serve_fresh(&mut policy, items, 2_000, |drafting, _| {
                if drafting { 1500 } else { 1000 }
            });
            let first = *drafted.first().expect("a probe drafts");
            assert!(first < 2 * usize::from(PROBE_INTERVAL), "C{items}: {first}");
            // Ordinary drafting resumes and keeps going.
            assert!(
                (1_000..2_000).all(|step| drafted.contains(&step)),
                "C{items}: {} drafting steps",
                drafted.len()
            );
        }
        // Where drafting keeps losing, fresh or not, probes stay rare.
        let mut policy = block_policy_with_cap(256);
        policy.observe_step(&[0], 0, 1000);
        policy.observe_step(&[0], 0, 1000);
        let drafted = serve_fresh(
            &mut policy,
            1,
            10_000,
            |drafting, _| {
                if drafting { 20_000 } else { 1000 }
            },
        );
        assert!(drafted.len() <= 12, "{drafted:?}");
    }

    /// The 2026-09-25 release re-smoke: the startup warmup verified two
    /// single drafts and both were rejected, which moved the shared prior's
    /// first position to 0.2; the policy never drafted again in a run where
    /// fixed K7 accepted 43% of its drafts.
    #[test]
    fn warmup_rejections_do_not_lock_drafting_off() {
        let mut policy = block_policy_with_cap(32);
        policy.observe_step(&[0], 0, 4_020);
        policy.observe_step(&[0], 0, 4_020);
        let mut warmup = policy.prior();
        for _ in 0..2 {
            policy.observe(&mut warmup, 1, 0);
            policy.observe_step(&[1], 1, 9_000);
        }
        assert!(policy.prior.positions[0] < 0.25);
        let cost = g9_cost(bench_target);
        let phases = [1, 4].map(|items| Phase {
            items,
            steps: 2_000,
            spare_rows: 32 - items,
            truth: 0.8,
            cost: &cost,
        });
        let rates = serve(&mut policy, &phases);
        assert!(rates[0] > 1.4 && rates[1] > 1.3, "{rates:?}");
    }

    #[test]
    fn a_probe_recovers_from_a_transient_expensive_draft_measurement() {
        let mut policy = block_policy_with_cap(32);
        policy.observe_step(&[0], 0, 1000);
        policy.observe_step(&[0], 0, 1000);
        policy.observe_step(&[7], 0, 20_000);
        policy.observe_step(&[7], 0, 20_000);
        let good = AcceptanceEstimate {
            positions: [0.99; MAX_DRAFT_TOKENS],
        };
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 7,
            acceptance: &good,
            fresh: false,
            remaining_tokens: usize::MAX,
        }];
        // The slowdown is over: a K7 step now costs 1.5 AR steps.
        let mut first = None;
        for step in 0..usize::from(PROBE_INTERVAL) + 16 {
            let drafts = policy.plan(&candidates, budget(31, 31));
            if drafts[0] > 0 {
                let mut estimate = good.clone();
                policy.observe(&mut estimate, drafts[0], drafts[0]);
            }
            policy.observe_step(&drafts, 0, if drafts[0] > 0 { 1500 } else { 1000 });
            if drafts[0] > 0 && first.is_none() {
                first = Some(step);
            }
        }
        let first = first.expect("a probe runs");
        assert!(first >= usize::from(PROBE_INTERVAL) - 1, "{first}");
        // Drafting resumed after the probe's recheck.
        assert_eq!(policy.plan(&candidates, budget(31, 31)), [7]);

        // Where drafting keeps losing, probes back off: a handful of steps
        // in ten thousand.
        let mut policy = block_policy_with_cap(32);
        policy.observe_step(&[0], 0, 1000);
        policy.observe_step(&[0], 0, 1000);
        policy.observe_step(&[7], 0, 20_000);
        policy.observe_step(&[7], 0, 20_000);
        let probes = (0..10_000)
            .filter(|_| {
                let drafts = policy.plan(&candidates, budget(31, 31));
                if drafts[0] > 0 {
                    let mut estimate = good.clone();
                    policy.observe(&mut estimate, drafts[0], drafts[0]);
                }
                policy.observe_step(&drafts, 0, if drafts[0] > 0 { 20_000 } else { 1000 });
                drafts[0] > 0
            })
            .count();
        assert!((1..=8).contains(&probes), "{probes} probes");
    }

    #[test]
    fn fresh_drafters_pay_their_catch_up_apart_from_steady_steps() {
        let mut policy = block_policy().with_draft_rounding(round_to_row_buckets);
        for _ in 0..2 {
            policy.observe_step(&[0; 4], 0, 8_000);
            policy.observe_step(&[3; 4], 0, 12_000);
        }
        let steady = policy.block_steps.clone();
        assert_eq!(
            policy.step_cost(4, DraftShape::from_counts(&[3; 4]), 0.0),
            12_000.0
        );
        // Two fresh drafters caught up for 3 ms each.
        policy.observe_step(&[3; 4], 2, 18_000);
        assert_eq!(policy.block_steps, steady);
        assert_eq!(policy.fresh_cost(4), 3_000.0);
        // A plan charges a fresh drafter its catch-up over a drafting run;
        // other concurrencies borrow the nearest measured catch-up.
        assert_eq!(
            policy.step_cost(4, DraftShape::from_counts(&[3; 4]), 2.0 / FRESH_RUN_STEPS),
            12_000.0 + 2.0 * 3_000.0 / FRESH_RUN_STEPS
        );
        assert_eq!(policy.fresh_cost(16), 3_000.0);
        // Among equally promising candidates the joint plan drafts those
        // that need no catch-up.
        let acceptance = AcceptanceEstimate {
            positions: [0.9; MAX_DRAFT_TOKENS],
        };
        let candidates = [true, false, true, false].map(|fresh| PlanCandidate {
            maximum_draft_tokens: 3,
            acceptance: &acceptance,
            fresh,
            remaining_tokens: usize::MAX,
        });
        let shares = candidates
            .iter()
            .map(|candidate| policy.catch_up_share(candidate))
            .collect::<Vec<_>>();
        assert_eq!(
            shares,
            [1.0 / FRESH_RUN_STEPS, 0.0, 1.0 / FRESH_RUN_STEPS, 0.0]
        );
        let joint = policy
            .joint(&candidates, &shares, 6, false)
            .expect("joint plan");
        assert_eq!(joint.drafts, [0, 3, 0, 3]);
        assert_eq!(joint.catch_up, 0.0);
    }

    /// A `spec_bench.py`-like cohort: `items` requests of `tokens` output
    /// tokens each, replaced as they finish, with acceptance `truth`. Each
    /// step costs `cost(drafts, catch_up)`, where `catch_up` counts drafting
    /// requests that did not verify in their previous step (a new request's
    /// first draft catches up its whole prompt). Returns realized throughput
    /// over the second half against AR.
    fn serve_cohorts(
        policy: &mut SpeculationPolicy,
        items: usize,
        steps: usize,
        tokens: usize,
        truth: f64,
        cost: &dyn Fn(&[usize], usize) -> u64,
    ) -> f64 {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut unit = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut estimates = vec![policy.prior(); items];
        let mut left = (0..items)
            .map(|item| tokens - item * tokens / items)
            .collect::<Vec<_>>();
        // A negative `truth` alternates requests between accepting 0.95 and
        // -truth of their drafts.
        let mut started = 0_usize;
        let mut request_truth = (0..items)
            .map(|item| {
                if truth >= 0.0 {
                    truth
                } else if item % 2 == 0 {
                    0.95
                } else {
                    -truth
                }
            })
            .collect::<Vec<_>>();
        let mut previous = vec![0; items];
        let (mut emitted_total, mut micros_total) = (0.0, 0.0);
        for step in 0..steps {
            let candidates = estimates
                .iter()
                .zip(&previous)
                .zip(&left)
                .map(|((acceptance, &before), &left)| PlanCandidate {
                    maximum_draft_tokens: 7.min(left.saturating_sub(1)),
                    acceptance,
                    fresh: before == 0,
                    remaining_tokens: left,
                })
                .collect::<Vec<_>>();
            let mut drafts = policy.plan(&candidates, budget(1024, 32 - items));
            for count in &mut drafts {
                *count = round_to_row_buckets(*count).min(*count);
            }
            let fresh = drafts
                .iter()
                .zip(&previous)
                .filter(|&(&count, &before)| count > 0 && before == 0)
                .count();
            let mut emitted = 0.0;
            for item in 0..items {
                let verified = drafts[item];
                let accepted = (0..verified)
                    .take_while(|_| unit() < request_truth[item])
                    .count();
                if verified > 0 {
                    policy.observe(&mut estimates[item], verified, accepted);
                }
                let out = (accepted + 1).min(left[item]);
                emitted += out as f64;
                left[item] -= out;
                if left[item] == 0 {
                    // The next request of the cohort starts fresh.
                    left[item] = tokens;
                    estimates[item] = policy.prior();
                    drafts[item] = 0;
                    started += 1;
                    if truth < 0.0 {
                        request_truth[item] = if started % 2 == 0 { 0.95 } else { -truth };
                    }
                }
            }
            let micros = cost(&drafts, fresh);
            policy.observe_step(&drafts, fresh, micros);
            if step >= steps / 2 {
                emitted_total += emitted;
                micros_total += micros as f64;
            }
            previous = drafts;
        }
        let zero = vec![0; items];
        emitted_total / micros_total / (items as f64 / cost(&zero, 0) as f64)
    }

    /// Verify steps at the G9 costs, where each drafting request that did
    /// not draft in its previous step pays a 4 ms eager catch-up.
    fn cohort_cost(drafts: &[usize], catch_up: usize) -> u64 {
        let items = drafts.len();
        let rows = items + drafts.iter().sum::<usize>();
        let sequences = drafts.iter().filter(|&&count| count > 0).count();
        if sequences == 0 {
            return g9_ar(items) as u64;
        }
        let block = 1_290.0 + 1_010.0 * (sequences - 1) as f64 / 15.0;
        (bench_target(rows.next_power_of_two())
            + block
            + 120.0 * sequences as f64
            + 4_000.0 * catch_up as f64) as u64
    }

    #[test]
    fn short_requests_keep_ar_where_drafting_loses() {
        for (items, truth) in [(16, 0.7), (32, 0.7), (4, 0.4)] {
            let mut policy = block_policy_with_cap(32);
            policy.observe_step(&vec![0; items], 0, g9_ar(items) as u64);
            policy.observe_step(&vec![0; items], 0, g9_ar(items) as u64);
            let rate = serve_cohorts(&mut policy, items, 4_000, 64, truth, &cohort_cost);
            eprintln!("COHORT C{items} {truth}: {rate}");
            assert!(rate >= 0.98, "C{items} at {truth}: {rate}");
        }
    }

    /// Per-position acceptance of DFlash K7 drafts on `spec_bench.py`,
    /// given every earlier draft was accepted: fixed K7 at C16 and C32 in
    /// a TP4 measurement on 2026-09-25.
    const CHAT_ACCEPTANCE: [f64; 7] = [0.80, 0.74, 0.72, 0.65, 0.73, 0.61, 0.71];
    const GSM8K_ACCEPTANCE: [f64; 7] = [0.86, 0.83, 0.86, 0.78, 0.83, 0.84, 0.81];

    /// One concurrency of a recorded `spec_bench.py` cohort: `items`
    /// requests of 64 output tokens, each replaced by a fresh request when
    /// it finishes, verifying at most `cap` rows per step. A step without
    /// drafts takes `ar` microseconds; a drafting step takes `bucket(rows)`
    /// for its power-of-two row bucket plus `catch_up` for each drafter that
    /// did not verify in its previous step.
    struct Recorded<'a> {
        items: usize,
        cap: usize,
        acceptance: [f64; 7],
        ar: f64,
        bucket: &'a dyn Fn(usize) -> f64,
        catch_up: f64,
    }

    impl Recorded<'_> {
        fn cost(&self, drafts: &[usize], fresh: usize) -> f64 {
            let rows = drafts.len() + drafts.iter().sum::<usize>();
            if rows == drafts.len() {
                self.ar
            } else {
                (self.bucket)(rows.next_power_of_two()) + self.catch_up * fresh as f64
            }
        }
    }

    /// Runs each phase for `steps` steps through `policy` (or fixed K7 when
    /// `None`) and returns, per phase, the tokens per unit of time of its
    /// second half against AR's and the share of those steps that drafted.
    fn serve_recorded(
        mut policy: Option<&mut SpeculationPolicy>,
        phases: &[Recorded<'_>],
        steps: usize,
    ) -> Vec<(f64, f64)> {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut unit = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let prior = |policy: &Option<&mut SpeculationPolicy>| {
            policy
                .as_ref()
                .map_or_else(AcceptanceEstimate::initial, |policy| policy.prior())
        };
        phases
            .iter()
            .map(|phase| {
                let items = phase.items;
                let mut estimates = vec![prior(&policy); items];
                // Requests start staggered, as a cohort's do after its first
                // replacements.
                let mut left = (0..items)
                    .map(|item| 64 - item * 64 / items)
                    .collect::<Vec<_>>();
                let mut previous = vec![0; items];
                let (mut tokens, mut micros, mut drafting) = (0.0, 0.0, 0);
                for step in 0..steps {
                    let candidates = estimates
                        .iter()
                        .zip(&previous)
                        .zip(&left)
                        .map(|((acceptance, &before), &left)| PlanCandidate {
                            maximum_draft_tokens: 7.min(left - 1),
                            acceptance,
                            fresh: before == 0,
                            remaining_tokens: left,
                        })
                        .collect::<Vec<_>>();
                    let budget = budget(1024, phase.cap - items);
                    let mut drafts = match policy.as_mut() {
                        Some(policy) => policy.plan(&candidates, budget),
                        None => {
                            let mut spare = phase.cap - items;
                            candidates
                                .iter()
                                .map(|candidate| {
                                    let count = candidate.maximum_draft_tokens.min(spare);
                                    spare -= count;
                                    count
                                })
                                .collect()
                        }
                    };
                    for count in &mut drafts {
                        *count = round_to_row_buckets(*count).min(*count);
                    }
                    let fresh = drafts
                        .iter()
                        .zip(&previous)
                        .filter(|&(&count, &before)| count > 0 && before == 0)
                        .count();
                    let mut emitted = 0.0;
                    let mut next = drafts.clone();
                    for item in 0..items {
                        let verified = drafts[item];
                        let accepted = (0..verified)
                            .take_while(|&position| unit() < phase.acceptance[position])
                            .count();
                        if verified > 0
                            && let Some(policy) = policy.as_mut()
                        {
                            policy.observe(&mut estimates[item], verified, accepted);
                        }
                        let out = (accepted + 1).min(left[item]);
                        emitted += out as f64;
                        left[item] -= out;
                        if left[item] == 0 {
                            // The next request starts fresh.
                            left[item] = 64;
                            estimates[item] = prior(&policy);
                            next[item] = 0;
                        }
                    }
                    let cost = phase.cost(&drafts, fresh);
                    if let Some(policy) = policy.as_mut() {
                        policy.observe_step(&drafts, fresh, cost as u64);
                    }
                    if step >= steps / 2 {
                        tokens += emitted;
                        micros += cost;
                        drafting += usize::from(drafts.iter().any(|&count| count > 0));
                    }
                    previous = next;
                }
                (
                    tokens / micros / (items as f64 / phase.ar),
                    drafting as f64 / (steps - steps / 2) as f64,
                )
            })
            .collect()
    }

    /// The recorded `spec_bench.py` steps per concurrency (TP4,
    /// 2026-09-25, medians, milliseconds): AR decode steps took 3.97, 6.86,
    /// 13.14 and 18.3 ms at C1, C4, C16 and C32, and fixed K7 verify steps
    /// 10.0 (8-row bucket), 20.2 (32), 38.8 (128) and 51.3 ms (256). The
    /// one-sequence C32 probe ran the 64-row bucket at twice the AR step,
    /// and a TP4 measurement gave a C16 drafting step in the 32-row bucket at 1.97
    /// times its AR step (16.8 against 8.52 ms). Other buckets lie on the
    /// geometric line between the recorded ones. A fresh drafter's
    /// catch-up costs 2 ms.
    fn recorded(items: usize, cap: usize, acceptance: [f64; 7]) -> Recorded<'static> {
        fn line(from: (usize, f64), to: (usize, f64), bucket: usize) -> f64 {
            let span = (to.0 as f64 / from.0 as f64).log2();
            let at = (bucket as f64 / from.0 as f64).log2();
            from.1 * (to.1 / from.1).powf(at / span)
        }
        let (ar, bucket): (f64, &'static dyn Fn(usize) -> f64) = match items {
            1 => (3_970.0, &|bucket| line((1, 3_970.0), (8, 10_000.0), bucket)),
            4 => (6_860.0, &|bucket| {
                line((4, 6_860.0), (32, 20_200.0), bucket)
            }),
            16 => (13_140.0, &|bucket| match bucket {
                ..=32 => 13_140.0 * 1.97,
                _ => line((32, 13_140.0 * 1.97), (128, 38_800.0), bucket),
            }),
            _ => (18_300.0, &|bucket| match bucket {
                ..=64 => 36_600.0,
                _ => line((64, 36_600.0), (256, 51_300.0), bucket),
            }),
        };
        Recorded {
            items,
            cap,
            acceptance,
            ar,
            bucket,
            catch_up: 2_000.0,
        }
    }

    /// SPEC-HICONC's TP4 arms at 256 rows: fixed K7 gained 21-29% on GSM8K
    /// at C16 and C32, while adaptive drafted nothing at C32. One drafting
    /// sequence there moves the step into the 64-row bucket at twice the AR
    /// time for about 2.5 more tokens, and the policy priced the full block
    /// from it at twice its measured time. Priced by bucket, with probes
    /// that measure the widest bucket first, the policy drafts at C16 and
    /// C32 wherever the recorded steps make the block pay, and keeps C1 and
    /// C4 at their gains.
    #[test]
    fn recorded_steps_draft_at_c16_and_c32_where_the_block_pays() {
        for (name, acceptance) in [("chat", CHAT_ACCEPTANCE), ("gsm8k", GSM8K_ACCEPTANCE)] {
            let phases = [1, 4, 16, 32].map(|items| recorded(items, 256, acceptance));
            let fixed = serve_recorded(None, &phases, 1_200);
            let mut policy = block_policy_with_cap(256);
            let adaptive = serve_recorded(Some(&mut policy), &phases, 1_200);
            eprintln!("{name}: fixed {fixed:?}");
            eprintln!("{name}: adaptive {adaptive:?}");
            for (items, ((rate, drafting), (fixed, _))) in
                [1, 4, 16, 32].into_iter().zip(adaptive.iter().zip(&fixed))
            {
                // Never below AR; where fixed K7 gains 15% or more, adaptive
                // drafts nearly every step and keeps most of the gain. Chat
                // at C4-C32 gains 8-12% with fixed K7, within the realized
                // throughput's noise: adaptive may keep AR there.
                assert!(*rate >= 0.98, "{name} C{items}: {rate}");
                if *fixed >= 1.15 {
                    assert!(*drafting > 0.9, "{name} C{items}: drafting {drafting}");
                    assert!(
                        *rate >= 0.9 * fixed,
                        "{name} C{items}: {rate} against {fixed}"
                    );
                }
            }
            assert!(
                adaptive[0].0 >= 1.3 && adaptive[1].0 >= 1.15,
                "{name}: {adaptive:?}"
            );
        }
    }

    /// Where the recorded steps say drafting loses, the policy stays at AR:
    /// within 32 rows a C16 drafting step runs the 32-row bucket at 1.97
    /// times the AR step for at most 16 drafts, and C32 has no spare row;
    /// and at 256 rows, text that accepts half its drafts repays no bucket.
    #[test]
    fn recorded_steps_keep_ar_at_c16_and_c32_where_drafting_loses() {
        for (name, cap, acceptance) in [
            ("chat, 32 rows", 32, CHAT_ACCEPTANCE),
            ("gsm8k, 32 rows", 32, GSM8K_ACCEPTANCE),
            ("half accepted, 256 rows", 256, [0.5; 7]),
        ] {
            let phases = [1, 4, 16, 32].map(|items| recorded(items, cap, acceptance));
            let mut policy = block_policy_with_cap(cap);
            let adaptive = serve_recorded(Some(&mut policy), &phases, 1_200);
            eprintln!("{name}: adaptive {adaptive:?}");
            for (rate, drafting) in &adaptive[2..] {
                assert!(*rate >= 0.99, "{name}: {adaptive:?}");
                assert!(*drafting < 0.02, "{name}: {adaptive:?}");
            }
        }
    }

    /// A probe of an unmeasured bucket runs in that bucket: the full-depth
    /// exploration a probe also triggers must not push it into the next.
    #[test]
    fn a_bucket_probe_stays_in_its_bucket() {
        let mut policy = block_policy_with_cap(256);
        policy.observe_step(&[0; 16], 0, 13_140);
        policy.observe_step(&[0; 16], 0, 13_140);
        // The 128-row bucket is measured and the 64-row one is not.
        policy.block_steps.record((16, 128), 38_800.0, 16, 1.0);
        policy.zero_draft[16].idle = PROBE_INTERVAL;
        let acceptance = AcceptanceEstimate {
            positions: [0.9; MAX_DRAFT_TOKENS],
        };
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 7,
            acceptance: &acceptance,
            fresh: false,
            remaining_tokens: 64,
        }; 16];
        let drafts = policy.plan(&candidates, budget(1024, 240));
        let rows = 16 + drafts.iter().sum::<usize>();
        assert!((33..=64).contains(&rows), "{rows} rows: {drafts:?}");
    }

    /// A probe whose drafters are fresh cannot tell the catch-up from the
    /// step. The next probe of the same bucket can, when fewer of its
    /// drafters are fresh; with as many, it records nothing.
    #[test]
    fn a_second_probe_separates_the_catch_up_from_the_step() {
        let mut policy = block_policy_with_cap(256);
        policy.observe_step(&[0; 16], 0, 13_140);
        policy.observe_step(&[0; 16], 0, 13_140);
        let drafts = [7; 16];
        // The bucket is warm, then a probe with every drafter fresh: 38.8 ms
        // and a 2 ms catch-up each.
        policy.observe_step(&drafts, 0, 38_800);
        policy.block_steps = BlockStepTable::default();
        let probe = |policy: &mut SpeculationPolicy, fresh: usize| {
            policy.probing = Some(16);
            policy.observe_step(&drafts, fresh, 38_800 + 2_000 * fresh as u64);
        };
        probe(&mut policy, 16);
        assert!(policy.fresh_probe.is_some());
        assert!(!policy.block_steps.measured(16, 128));
        // The next probe finds every drafter fresh again, for example after
        // a batch that also prefilled.
        probe(&mut policy, 16);
        assert!(
            !policy.block_steps.measured(16, 128),
            "{:?}",
            policy.block_steps
        );
        assert!(policy.fresh_probe.is_none());
        // Two requests were replaced before the next probe's second step.
        probe(&mut policy, 16);
        probe(&mut policy, 2);
        let shape = DraftShape::from_counts(&drafts);
        assert!((policy.step_cost(16, shape, 0.0) - 38_800.0).abs() < 1e-6);
        assert!((policy.fresh_cost(16) - 2_000.0).abs() < 1e-6);
        // A stale fresh probe is not applied to a later step.
        probe(&mut policy, 16);
        policy.observe_step(&[7; 8], 0, 20_000);
        assert!(policy.fresh_probe.is_none());
    }

    /// A batch that also prefills takes the prefill's time, which the
    /// policy never observes, and a fresh drafter there adds its whole
    /// catch-up to it: every request that waits for its first token pays.
    /// Only caught-up requests draft in such a batch, and no probe runs.
    #[test]
    fn batches_that_prefill_draft_only_caught_up_requests() {
        let mut policy = block_policy_with_cap(256);
        let acceptance = AcceptanceEstimate {
            positions: [0.95; MAX_DRAFT_TOKENS],
        };
        for _ in 0..2 {
            policy.observe_step(&[0; 4], 0, 6_860);
            policy.observe_step(&[7; 4], 0, 12_000);
        }
        let candidates = |fresh: [bool; 4]| {
            fresh.map(|fresh| PlanCandidate {
                maximum_draft_tokens: 7,
                acceptance: &acceptance,
                fresh,
                remaining_tokens: 64,
            })
        };
        let decode = budget(1024, 252);
        let prefill = decode.with_prefill(true);
        assert_eq!(policy.plan(&candidates([true; 4]), decode), [7; 4]);
        assert_eq!(policy.plan(&candidates([true; 4]), prefill), [0; 4]);
        let drafts = policy.plan(&candidates([true, false, true, false]), prefill);
        assert!(
            drafts[0] == 0 && drafts[1] > 0 && drafts[2] == 0 && drafts[3] > 0,
            "{drafts:?}"
        );
        // A due probe waits for a batch without prefill.
        let mut policy = block_policy_with_cap(256);
        policy.observe_step(&[0; 4], 0, 6_860);
        policy.observe_step(&[0; 4], 0, 6_860);
        policy.zero_draft[4].idle = PROBE_INTERVAL;
        let poor = AcceptanceEstimate {
            positions: [0.05; MAX_DRAFT_TOKENS],
        };
        let steady = [false; 4].map(|fresh| PlanCandidate {
            maximum_draft_tokens: 7,
            acceptance: &poor,
            fresh,
            remaining_tokens: 64,
        });
        assert_eq!(policy.plan(&steady, prefill), [0; 4]);
        assert_eq!(policy.probing, None);
        assert!(policy.plan(&steady, decode).iter().any(|&count| count > 0));
        assert_eq!(policy.probing, Some(4));
    }

    /// A fresh drafter's catch-up is spread over the steps its request has
    /// left, at most `FRESH_RUN_STEPS`: near the end of a request it is
    /// not worth paying.
    #[test]
    fn a_catch_up_is_spread_over_the_steps_a_request_has_left() {
        let policy = block_policy_with_cap(256);
        let acceptance = AcceptanceEstimate {
            positions: [0.5; MAX_DRAFT_TOKENS],
        };
        let candidate = |fresh, remaining_tokens| PlanCandidate {
            maximum_draft_tokens: 7,
            acceptance: &acceptance,
            fresh,
            remaining_tokens,
        };
        assert_eq!(policy.catch_up_share(&candidate(false, 4)), 0.0);
        assert_eq!(
            policy.catch_up_share(&candidate(true, usize::MAX)),
            1.0 / FRESH_RUN_STEPS
        );
        // About two tokens a step: 64 tokens take more than 16 steps, 8
        // take about four.
        assert_eq!(
            policy.catch_up_share(&candidate(true, 64)),
            1.0 / FRESH_RUN_STEPS
        );
        let per_step = 1.0 + (1..=7).map(|j| 0.5_f64.powi(j)).sum::<f64>();
        assert!((policy.catch_up_share(&candidate(true, 8)) - per_step / 8.0).abs() < 1e-12);
        assert_eq!(policy.catch_up_share(&candidate(true, 1)), 1.0);
    }

    /// An AR choice at some items does not hold later plans there: counts
    /// recur as a later cohort's requests finish, and the next plan at the
    /// same items decides on current estimates behind the resume margin.
    #[test]
    fn an_ar_choice_is_rechecked_on_the_next_plan_at_the_same_items() {
        for model in [DraftCostModel::PerToken, DraftCostModel::PerBlock] {
            let mut policy = policy(SpeculationMode::Adaptive);
            policy.config = policy.config.with_draft_cost_model(model);
            for rows in 1..=4 {
                policy.costs.micros.insert(rows, 1000.0);
            }
            let estimate = policy.prior();
            let one = PlanCandidate {
                maximum_draft_tokens: 1,
                acceptance: &estimate,
                fresh: false,
                remaining_tokens: usize::MAX,
            };
            set_draft_cost(&mut policy, &[1], 700.0);
            assert_eq!(policy.plan(&[one], budget(1, 1)), [0]);
            // Plans at other items in between do not matter.
            policy.plan(&[one, one], budget(2, 2));
            set_draft_cost(&mut policy, &[1], 400.0);
            assert_eq!(policy.plan(&[one], budget(1, 1)), [1]);
            // Drafting that loses in realized throughput still holds the
            // next plans off.
            policy.realized.insert(
                1,
                Realized {
                    ar: Some(1.0),
                    drafting: Some(0.5),
                    drafting_steps: REALIZED_STEPS,
                },
            );
            assert_eq!(policy.plan(&[one], budget(1, 1)), [0]);
            policy.realized.remove(&1);
            for _ in 0..ZERO_DRAFT_RECHECK {
                assert_eq!(policy.plan(&[one], budget(1, 1)), [0]);
            }
            assert_eq!(policy.plan(&[one], budget(1, 1)), [1]);
        }
    }

    /// A row bucket never run at these items is priced from the same bucket
    /// at the nearest other item count: the bucket fixes the target graph.
    /// b4 at C16: once one request finished, 15 items priced the 128-row K7
    /// step from their 32-row one at 1.7 per doubling, 65 ms against the
    /// 37 ms it took at 16 items, and drafted one token each.
    #[test]
    fn an_unmeasured_bucket_uses_the_same_bucket_at_other_items() {
        let mut policy = block_policy_with_cap(256);
        policy.costs.micros.insert(15, 14_000.0);
        policy.costs.micros.insert(16, 13_300.0);
        for (key, micros, sequences) in [
            ((16, 128), 37_000.0, 16.0),
            ((16, 64), 27_500.0, 16.0),
            ((15, 32), 22_500.0, 15.0),
        ] {
            let step = BlockStep { micros, sequences };
            policy.block_steps.steps.insert(key, step);
        }
        let k7 = DraftShape::from_counts(&[7; 15]);
        assert_eq!(policy.step_cost(15, k7, 0.0), 37_000.0);
        // The nearest measured item count lends it, on either side.
        let step = BlockStep {
            micros: 30_000.0,
            sequences: 10.0,
        };
        policy.block_steps.steps.insert((10, 128), step);
        policy.costs.micros.insert(11, 12_000.0);
        let k7 = DraftShape::from_counts(&[7; 11]);
        assert_eq!(
            policy.step_cost(15, DraftShape::from_counts(&[7; 15]), 0.0),
            37_000.0
        );
        let sequence = policy.costs.estimate(1) * PRIOR_SEQUENCE_COST;
        assert_eq!(policy.step_cost(11, k7, 0.0), 30_000.0 + sequence);
        policy.block_steps.steps.remove(&(10, 128));
        // More drafting sequences than the borrowed step averaged add
        // their cost.
        assert_eq!(
            policy
                .block_steps
                .estimate(17, 128, 17, &policy.costs, sequence),
            37_000.0 + sequence
        );
        // A bucket no item count ran keeps the ratio from these items.
        assert_eq!(
            policy
                .block_steps
                .estimate(15, 256, 15, &policy.costs, sequence),
            22_500.0 * WIDER_BUCKET_RATIO.powi(3)
        );
        let acceptance = AcceptanceEstimate {
            positions: [0.8; MAX_DRAFT_TOKENS],
        };
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 7,
            acceptance: &acceptance,
            fresh: false,
            remaining_tokens: usize::MAX,
        }; 15];
        assert_eq!(policy.plan(&candidates, budget(4096, 241)), [7; 15]);
    }

    /// Plan time with every table populated, as after a long serving run:
    /// AR costs at every item count, block steps or draft costs at depths
    /// 1, 3 and 7 for each, and catch-up costs. With drafting off every
    /// plan is evaluated in full, as an AR choice no longer skips the next
    /// plans. Run with `--ignored --nocapture` in a
    /// release build.
    #[test]
    #[ignore = "timing"]
    fn plan_time_with_populated_tables() {
        for (model, depth, rows) in [
            (DraftCostModel::PerToken, 3, 1024),
            (DraftCostModel::PerBlock, 7, 1024),
        ] {
            for items in [16, 32, 256] {
                for drafting_pays in [true, false] {
                    let mut policy = SpeculationPolicy::new(
                        SpeculationConfig::new(
                            depth,
                            rows,
                            SpeculationMode::Adaptive,
                            DraftSource::Bundle,
                        )
                        .unwrap()
                        .with_draft_cost_model(model),
                    )
                    .with_draft_rounding(round_to_row_buckets)
                    .with_step_rows(usize::next_power_of_two);
                    let bucket_micros = |rows: usize| 4_000 + 12 * rows.next_power_of_two() as u64;
                    let verify = if drafting_pays { 1 } else { 40 };
                    // AR steps price per-token verify rows, so measure every
                    // row count a plan can reach.
                    for count in 1..=(items * (depth + 1)).min(MAX_COST_ROWS) {
                        for _ in 0..2 {
                            policy.observe_step(&vec![0; count], 0, bucket_micros(count));
                        }
                    }
                    for count in 1..=items {
                        for draft in [1, 3, 7].into_iter().filter(|&d| d <= depth) {
                            let drafts = vec![draft; count];
                            let micros = bucket_micros(count * (draft + 1)) * verify;
                            for fresh in [count, 0, 0] {
                                for _ in 0..count {
                                    let mut estimate = AcceptanceEstimate::initial();
                                    policy.observe(&mut estimate, draft, draft / 2);
                                }
                                policy.observe_step(&drafts, fresh, micros + 20 * fresh as u64);
                            }
                        }
                    }
                    // Only the estimates decide, so a losing concurrency
                    // evaluates every plan in full.
                    policy.realized.clear();
                    let acceptance = AcceptanceEstimate {
                        positions: [0.7; MAX_DRAFT_TOKENS],
                    };
                    let candidates = (0..items)
                        .map(|index| PlanCandidate {
                            maximum_draft_tokens: depth,
                            acceptance: &acceptance,
                            fresh: index % 4 == 0,
                            remaining_tokens: 64,
                        })
                        .collect::<Vec<_>>();
                    let budget = budget(1 << 20, rows - items);
                    let iterations = 200;
                    let mut drafted = 0;
                    let started = std::time::Instant::now();
                    for _ in 0..iterations {
                        let plan = policy.plan(&candidates, budget);
                        drafted += usize::from(plan.iter().any(|&count| count > 0));
                    }
                    let micros = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;
                    eprintln!(
                        "{model:?} K{depth} items={items} drafting_pays={drafting_pays}: \
                         {micros:.1} us per plan, {drafted}/{iterations} drafted"
                    );
                    assert_eq!(drafted > iterations / 2, drafting_pays);
                }
            }
        }
    }

    /// Estimates can be wrong in ways the cost tables cannot see: here every
    /// request's acceptance estimate stays at 0.99 while no draft is
    /// accepted, as for short requests whose estimates lag. The realized
    /// throughput of drafting steps falls below AR's, and drafting stops.
    #[test]
    fn realized_throughput_below_ar_switches_drafting_off() {
        let mut policy = block_policy_with_cap(32);
        policy.observe_step(&[0; 4], 0, 5_000);
        policy.observe_step(&[0; 4], 0, 5_000);
        let optimistic = AcceptanceEstimate {
            positions: [0.99; MAX_DRAFT_TOKENS],
        };
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 7,
            acceptance: &optimistic,
            fresh: false,
            remaining_tokens: usize::MAX,
        }; 4];
        let mut drafting = Vec::new();
        for step in 0..2_000 {
            let drafts = policy.plan(&candidates, budget(1024, 28));
            for &count in drafts.iter().filter(|&&count| count > 0) {
                let mut estimate = optimistic.clone();
                policy.observe(&mut estimate, count, 0);
            }
            let rows = 4 + drafts.iter().sum::<usize>();
            policy.observe_step(&drafts, 0, 5_000 + 100 * (rows as u64 - 4));
            if drafts.iter().any(|&count| count > 0) {
                drafting.push(step);
            }
        }
        assert!(!drafting.is_empty());
        assert!(
            drafting.len() <= 24,
            "{} drafting steps: {drafting:?}",
            drafting.len()
        );
    }

    /// Measured TP4 steps of a per-token MTP drafter at C16 (serving step
    /// traces of a 512-token C16 bench, medians): AR took 14.68 ms with 16 items and
    /// 7.7 to 10.5 ms with 1 to 8; verify steps took 8.4, 9.2, 10.3 and
    /// 12.9 ms in the 2- to 16-row graphs and 23.2 ms with 16 K1 items in
    /// the 32-row graph (18.7 ms with 8 items). The host adds 0.2 ms.
    fn mtp_step(drafts: &[usize], fresh: usize) -> f64 {
        let items = drafts.len();
        let ar = [(1, 7.7), (2, 7.7), (4, 8.5), (8, 10.5), (16, 14.68)]
            .windows(2)
            .find(|pair| items <= pair[1].0)
            .map_or(14.68, |pair| {
                let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
                y0 + (y1 - y0) * (items - x0) as f64 / (x1 - x0) as f64
            });
        let rows = items + drafts.iter().sum::<usize>();
        let ms = match rows.next_power_of_two() {
            _ if rows == items => ar,
            ..=2 => 8.4,
            4 => 9.2,
            8 => 10.3,
            16 => 12.9,
            _ => 15.4 + 0.49 * items as f64,
        };
        (ms.max(ar) + 0.2) * 1000.0 + 150.0 * fresh as f64
    }

    /// Serves `cohorts` cohorts of `items` requests of 512 tokens each with
    /// per-position acceptance `acceptance`, through `policy` or, without
    /// one, fixed K1. Returns the tokens per microsecond of all but the
    /// first cohort and their share of drafting steps.
    fn serve_mtp(
        mut policy: Option<&mut SpeculationPolicy>,
        items: usize,
        acceptance: [f64; 3],
        cohorts: usize,
        seed: u64,
    ) -> (f64, f64) {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut unit = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let (mut tokens, mut micros, mut drafting, mut steps) = (0.0, 0.0, 0, 0);
        for cohort in 0..cohorts {
            let prior = policy
                .as_ref()
                .map_or_else(AcceptanceEstimate::initial, |policy| policy.prior());
            let mut left = vec![511_usize; items];
            let mut estimates = vec![prior; items];
            let mut verified = vec![false; items];
            while !left.is_empty() {
                let candidates = left
                    .iter()
                    .zip(&estimates)
                    .zip(&verified)
                    .map(|((&left, acceptance), &verified)| PlanCandidate {
                        maximum_draft_tokens: 3.min(left - 1),
                        acceptance,
                        fresh: !verified,
                        remaining_tokens: left,
                    })
                    .collect::<Vec<_>>();
                let mut drafts = match policy.as_mut() {
                    Some(policy) => policy.plan(&candidates, budget(1024, 32 - left.len())),
                    None => candidates
                        .iter()
                        .map(|candidate| candidate.maximum_draft_tokens.min(1))
                        .collect(),
                };
                for count in &mut drafts {
                    *count = round_to_row_buckets(*count).min(*count);
                }
                let fresh = drafts
                    .iter()
                    .zip(&verified)
                    .filter(|&(&count, &verified)| count > 0 && !verified)
                    .count();
                let cost = mtp_step(&drafts, fresh);
                let mut emitted = 0;
                for (index, &count) in drafts.iter().enumerate() {
                    let accepted = (0..count)
                        .take_while(|&position| unit() < acceptance[position])
                        .count();
                    if count > 0
                        && let Some(policy) = policy.as_mut()
                    {
                        policy.observe(&mut estimates[index], count, accepted);
                    }
                    let out = (accepted + 1).min(left[index]);
                    emitted += out;
                    left[index] -= out;
                    verified[index] = count > 0;
                }
                if let Some(policy) = policy.as_mut() {
                    policy.observe_step(&drafts, fresh, cost as u64);
                }
                if cohort > 0 {
                    tokens += emitted as f64;
                    micros += cost;
                    steps += 1;
                    drafting += usize::from(drafts.iter().any(|&count| count > 0));
                }
                // Finished requests leave the batch.
                let mut index = 0..;
                let live = left.iter().map(|&left| left > 0).collect::<Vec<_>>();
                estimates.retain(|_| live[index.next().unwrap()]);
                let mut index = 0..;
                verified.retain(|_| live[index.next().unwrap()]);
                left.retain(|&left| left > 0);
            }
        }
        (tokens / micros, drafting as f64 / steps as f64)
    }

    fn mtp_policy() -> SpeculationPolicy {
        SpeculationPolicy::new(
            SpeculationConfig::new(3, 32, SpeculationMode::Adaptive, DraftSource::Bundle).unwrap(),
        )
        .with_draft_rounding(round_to_row_buckets)
        .with_step_rows(|rows| rows.next_power_of_two().max(1))
    }

    /// At C16 any drafting runs the 32-row graph at 1.6 times the AR step.
    /// Fixed K1 gains about 10% on chat and GSM8K there, yet adaptive K3
    /// stayed at AR: priced at their exact row counts, a probe's few drafts
    /// carried the 16- to 32-row jump as drafting cost, scaled up for full
    /// K1, and probes that beat AR backed off before the estimates learned
    /// the acceptance. Now adaptive drafts K1 and keeps its gain.
    #[test]
    fn per_token_drafts_k1_at_c16_where_the_32_row_step_pays() {
        for (name, acceptance) in [("chat", [0.75, 0.68, 0.71]), ("gsm8k", [0.735, 0.80, 0.84])] {
            for seed in 0..4 {
                let (fixed, _) = serve_mtp(None, 16, acceptance, 3, seed);
                let mut policy = mtp_policy();
                let (rate, drafting) = serve_mtp(Some(&mut policy), 16, acceptance, 3, seed);
                let ar = 16.0 / mtp_step(&[0; 16], 0);
                assert!(
                    fixed > 1.08 * ar,
                    "{name} {seed}: fixed K1 {fixed} against AR {ar}"
                );
                assert!(
                    rate > 0.98 * fixed,
                    "{name} {seed}: {rate} against fixed K1 {fixed}"
                );
                assert!(drafting > 0.9, "{name} {seed}: drafting {drafting}");
            }
        }
    }

    /// Where K1 at C16 emits less per unit of time than AR, adaptive stays
    /// at AR apart from spaced probes.
    #[test]
    fn per_token_keeps_ar_at_c16_where_k1_loses() {
        for seed in 0..4 {
            let acceptance = [0.4, 0.4, 0.4];
            let (fixed, _) = serve_mtp(None, 16, acceptance, 3, seed);
            let mut policy = mtp_policy();
            let (rate, drafting) = serve_mtp(Some(&mut policy), 16, acceptance, 3, seed);
            let ar = 16.0 / mtp_step(&[0; 16], 0);
            assert!(fixed < ar, "{seed}: fixed K1 {fixed} against AR {ar}");
            assert!(rate > 0.98 * ar, "{seed}: {rate} against AR {ar}");
            assert!(drafting < 0.05, "{seed}: drafting {drafting}");
        }
    }
    #[test]
    fn borrowed_small_shape_does_not_enable_losing_full_k1() {
        let mut policy = mtp_policy();
        policy.costs.record(16, 14_000);
        policy.costs.record(32, 15_000);
        let mut small = [0; 16];
        small[0] = 1;
        for _ in 0..2 {
            policy.observe_step(&small, 0, 16_000);
        }
        let acceptance = policy.prior();
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 1,
            acceptance: &acceptance,
            fresh: false,
            remaining_tokens: usize::MAX,
        }; 16];
        assert_eq!(
            policy.step_cost(16, DraftShape::from_counts(&small), 0.0),
            16_000.0
        );
        assert_eq!(policy.plan(&candidates, budget(16, 16)), [0; 16]);
        // Once this exact shape is measured, use its own time without scaling.
        for _ in 0..2 {
            policy.observe_step(&[1; 16], 0, 24_000);
        }
        assert_eq!(
            policy.step_cost(16, DraftShape::from_counts(&[1; 16]), 0.0),
            24_000.0
        );
    }

    #[test]
    fn per_token_different_probe_shapes_do_not_invent_catch_up() {
        let mut policy = mtp_policy();
        policy.costs.record(16, 14_000);
        policy.costs.record(32, 15_000);
        let full = [1; 16];
        let mut half = [0; 16];
        half[..8].fill(1);
        // Warm both shapes without recording steady measurements.
        policy.observe_step(&full, 0, 24_000);
        policy.observe_step(&half, 0, 20_000);
        policy.probing = Some(16);
        policy.observe_step(&full, 16, 40_000);
        policy.probing = Some(16);
        policy.observe_step(&half, 8, 28_000);
        assert!(!policy.fresh_costs.contains_key(&16));
        assert!(
            !policy
                .draft_costs
                .micros
                .contains_key(&DraftCostModel::PerToken.key(16, DraftShape::from_counts(&half)))
        );
        // A matching follow-up separates the true 1 ms per-item catch-up.
        policy.probing = Some(16);
        policy.observe_step(&half, 0, 20_000);
        assert_eq!(policy.fresh_cost(16), 1_000.0);
        assert_eq!(
            policy.step_cost(16, DraftShape::from_counts(&half), 0.0),
            20_000.0
        );
    }

    #[test]
    fn marginal_probe_keeps_backoff_for_both_draft_models() {
        for model in [DraftCostModel::PerToken, DraftCostModel::PerBlock] {
            for (micros, clears_margin) in [(1_950, false), (1_800, true)] {
                let mut policy = SpeculationPolicy::new(
                    SpeculationConfig::new(1, 32, SpeculationMode::Adaptive, DraftSource::Bundle)
                        .unwrap()
                        .with_draft_cost_model(model),
                );
                for _ in 0..2 {
                    policy.observe_step(&[0], 0, 1_000);
                }
                policy.observe_step(&[1], 0, micros);
                policy.zero_draft[1].probe_interval = 512;
                policy.probing = Some(1);
                policy.step_accepted = 1;
                policy.observe_step(&[1], 0, micros);
                if clears_margin {
                    assert!(policy.zero_draft[1].probe_interval < 512);
                } else {
                    assert_eq!(policy.zero_draft[1].probe_interval, 512, "{model:?}");
                }
            }
        }
    }
    fn sparse_shape_recovers(small_micros: u64, full_micros: u64, acceptance: f64) {
        let mut policy = mtp_policy();
        policy.costs.record(16, 14_000);
        policy.costs.record(32, 15_000);
        let mut small = [0; 16];
        small[0] = 1;
        for _ in 0..2 {
            policy.observe_step(&small, 0, small_micros);
        }
        let estimate = AcceptanceEstimate {
            positions: [acceptance; MAX_DRAFT_TOKENS],
        };
        let candidates = [PlanCandidate {
            maximum_draft_tokens: 1,
            acceptance: &estimate,
            fresh: false,
            remaining_tokens: usize::MAX,
        }; 16];
        let mut full_steps = Vec::new();
        for step in 0..40 {
            let drafts = policy.plan(&candidates, budget(16, 16));
            let count = drafts.iter().sum::<usize>();
            let key = DraftCostModel::PerToken.key(16, DraftShape::from_counts(&drafts));
            if count == 16 {
                if !policy.draft_costs.micros.contains_key(&key) {
                    assert_eq!(
                        policy.probing,
                        Some(16),
                        "unmeasured full shape committed at {step}"
                    );
                }
                full_steps.push(step);
            }
            let micros = if count == 0 {
                14_000
            } else {
                small_micros + (full_micros - small_micros) * (count as u64 - 1) / 15
            };
            policy.step_accepted = (count as f64 * acceptance) as usize;
            policy.observe_step(&drafts, 0, micros);
        }
        assert!(
            full_steps
                .iter()
                .filter(|&&step| step <= usize::from(FIRST_PROBE) + 2)
                .count()
                >= 2,
            "full shape must be warmed and measured promptly: {full_steps:?}"
        );
        let settles_on_full =
            (16.0 * (1.0 + acceptance)) / full_micros as f64 > 16.0 / 14_000.0 * RESUME_GAIN;
        assert_eq!(
            full_steps.iter().filter(|&&step| step >= 30).count(),
            if settles_on_full { 10 } else { 0 },
            "{full_steps:?}"
        );
        assert_eq!(
            policy.step_cost(16, DraftShape::from_counts(&[1; 16]), 0.0),
            full_micros as f64
        );
    }

    #[test]
    fn negative_small_shape_excess_probes_before_committing_full_k1() {
        sparse_shape_recovers(13_000, 24_000, 0.6);
    }

    #[test]
    fn positive_small_shape_excess_promptly_discovers_profitable_full_k1() {
        sparse_shape_recovers(16_000, 23_200, 0.75);
    }
    #[test]
    fn slower_ar_rechecks_a_measured_shape_before_stale_backoff_expires() {
        for ar_micros in [12_400, 14_000] {
            let mut policy = mtp_policy();
            for _ in 0..2 {
                policy.observe_step(&[0; 16], 0, 12_000);
            }
            // Full K1 is known, but a low-acceptance run lost against fast AR.
            for _ in 0..10 {
                policy.step_accepted = 10;
                policy.observe_step(&[1; 16], 0, 21_200);
            }
            let acceptance = AcceptanceEstimate {
                positions: [0.75; MAX_DRAFT_TOKENS],
            };
            let candidates = [PlanCandidate {
                maximum_draft_tokens: 1,
                acceptance: &acceptance,
                fresh: false,
                remaining_tokens: usize::MAX,
            }; 16];
            policy.zero_draft[16].idle = PROBE_INTERVAL;
            assert_eq!(policy.plan(&candidates, budget(16, 16)), [1; 16]);
            policy.step_accepted = 10;
            policy.observe_step(&[1; 16], 0, 21_200);
            assert!(policy.zero_draft[16].probe_interval >= 128);
            let mut drafting = Vec::new();
            for step in 0..40 {
                let drafts = policy.plan(&candidates, budget(16, 16));
                let count = drafts.iter().sum::<usize>();
                if count > 0 {
                    drafting.push(step);
                }
                policy.step_accepted = count * 3 / 4;
                policy.observe_step(&drafts, 0, if count > 0 { 21_200 } else { ar_micros });
            }
            if ar_micros == 14_000 {
                assert!(
                    drafting.first().is_some_and(|&step| step <= 17),
                    "{drafting:?}"
                );
                assert_eq!(
                    drafting.iter().filter(|&&step| step >= 30).count(),
                    10,
                    "{drafting:?}"
                );
            } else {
                assert!(
                    drafting.is_empty(),
                    "small timing changes must retain backoff: {drafting:?}"
                );
            }
        }
    }
}
