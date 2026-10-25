//! The all-reduce of BF16 partials (`CommQuery::AllReduce` of dtype BF16,
//! unfused): every rank's F32 partial rounded once to BF16, the ranks' BF16
//! values summed in F32 in rank order from zero, the sum rounded once to
//! BF16 into the BF16 output. Every form below gives those bits, so the
//! choice of form moves no bit.
//!
//! Each row count's form is chosen when the op is prepared, from SM120's
//! table ([`FORMS`]) and the model's registrations:
//!
//! - The low-latency entry (`comm::ll_bf16`) at the table's low-latency
//!   counts, through a registration whose admitted grid holds the rows
//!   (`Sm120::with_low_latency_bf16_all_reduce`), at power-of-two rank
//!   counts and rows of whole 2048-element slices; else what serves the
//!   phased one-shot's counts.
//! - The phased one-shot and two-shot entries (`comm::phased_bf16`) and the
//!   turns entry (`comm::phased_turns_bf16`) at the table's counts, senders
//!   and workers, once the model registers them
//!   (`Sm120::with_phased_bf16_all_reduce`, after its create admitted
//!   their largest grids, [`phased_admission_plans`]) and the rows are
//!   whole 1024-element granules.
//! - Otherwise the composed form: the model's round row of the input into
//!   the output, its widen row back into the input, the comm engine's F32
//!   reduce-scatter then all-gather into every rank's F32 sum, and the
//!   round row of the sum into the output (`Sm120::with_partial_rounding`).
//!   It leaves the input rows rounded to BF16. Without a rounding
//!   registration of the op's width, an op with a row count left to it is
//!   refused.
//!
//! An entry whose declarations do not keep `Need.exact` is passed over at
//! the low-latency counts, and the op claims what every launched form
//! declares together (their meet). The composed form claims what its
//! stages declare together: the F32 pair's declarations
//! (`comm::declared_entries`) and the model's rounding rows' (registered
//! with them, `PartialRounding::invariance`). Each stage keeps a row's
//! bits apart from its row count and the other rows (the pair's owner sums
//! every rank in rank order, whichever rank owns an element), so their
//! meet holds for the form.
//!
//! State regions, in order (`OpInfo::state`): each low-latency state that
//! serves (its inbox and epochs, least rows first), at one capacity
//! whatever row count launches it; the phased state every phased entry and
//! row count shares (inbox, flags with the turns words, epoch); the F32
//! pair's (inbox, flags, epoch) where the composed form serves; and the F32
//! sum `[capacity, width]` that the pair's all-gather writes on every rank
//! (also present when the comm epilogue asks for it). The phased entries
//! write every rank's output (`RankViews::output`); the other forms write
//! this rank's.
//!
//! The phased entries and the composed form's rows read the device live
//! count (`AllReduceBuffers::live_rows`): the phased entries reduce the
//! rows below it when given one, the composed form refuses launches
//! without one. The low-latency entry reduces every row of its launch.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::collective::AllReduceBuffers;
use kernel_core::contract::{
    Argument, ArgumentList, CommQuery, DType, LaunchPlan, MAX_COMM_RANKS, Refusal, Validate,
};
use kernel_core::properties::Invariance;

use super::select::{Candidate, choose, sole};
use super::{MAX_CAPACITY, check_plan};
use crate::comm::{self, OneShotCall, Variant, ll_bf16, phased_bf16, phased_turns_bf16};
use crate::row::rounding::{self, PartialRounding};

/// A row count's form in SM120's table ([`tuned`]). An op takes it where
/// the model registered the form's entry and the rows suit it, else the
/// form after it (the module's list).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tuned {
    /// The low-latency entry (`comm::ll_bf16`).
    LowLatency,
    /// The phased one-shot entry (`comm::phased_bf16`).
    OneShot,
    /// The phased two-shot entry (`comm::phased_bf16`).
    TwoShot { senders: u32, workers: u32 },
    /// The phased turns entry (`comm::phased_turns_bf16`).
    Turns { senders: u32, workers: u32 },
}

impl Tuned {
    /// The entry's symbol.
    pub fn symbol(self) -> &'static str {
        match self {
            Self::LowLatency => ll_bf16::SYMBOL,
            Self::OneShot => phased_bf16::ONE_SHOT_SYMBOL,
            Self::TwoShot { .. } => phased_bf16::SYMBOL,
            Self::Turns { .. } => phased_turns_bf16::SYMBOL,
        }
    }
}

/// The fewest rows the phased turns entry serves. The all-reduces of long
/// steps follow compute whose time differs between ranks, so the ranks
/// start them at different times; the turns keep one writer per receiving
/// link where the two-shot lets two or three share one. Below this the
/// turns' handoffs cost more than they save.
pub const TURNS_ROWS: u32 = 1024;

/// Sending CTAs of every phased call, and the workers of a two-shot or
/// turns call above [`PHASED_DECODE_ROWS`].
const SENDERS: u32 = 16;
const WIDE_WORKERS: u32 = 32;
/// The most rows of a phased two-shot call without workers.
const PHASED_DECODE_ROWS: u32 = 32;

/// SM120's form of each row count: the first bound that holds the rows,
/// past every bound the phased turns. Timed on four ranks of one PCIe host
/// for the phased F32 entries, which the BF16 entries keep until their own
/// timing moves them (us per call: the low-latency entry 5.6 at one row
/// and 10.0 at two; the phased one-shot 8.0 at one, 8.9 at two and 10.8 at
/// four; the two-shot 16.3 to 26.3 at 8 to 32 rows, against the F32 pair's
/// 15.7 to 26.6 plus a round row's ~2). Up to 32 rows the two-shot runs its
/// 16 senders alone (16.3, 18.9 and 26.3 us at 8, 16 and 32 rows, against
/// 16.4, 19.6 and 27.4 at 8 senders and 8 workers); above, 32 workers join.
const FORMS: [(u32, Tuned); 4] = [
    (1, Tuned::LowLatency),
    (4, Tuned::OneShot),
    (
        PHASED_DECODE_ROWS,
        Tuned::TwoShot {
            senders: SENDERS,
            workers: 0,
        },
    ),
    (
        TURNS_ROWS - 1,
        Tuned::TwoShot {
            senders: SENDERS,
            workers: WIDE_WORKERS,
        },
    ),
];

/// The most rows the table gives the low-latency entry: the rows a model's
/// create admits its grid for (`ll_bf16::plan` of these rows).
pub const LOW_LATENCY_ROWS: u32 = {
    let mut most = 0;
    let mut at = 0;
    while at < FORMS.len() {
        if matches!(FORMS[at].1, Tuned::LowLatency) && FORMS[at].0 > most {
            most = FORMS[at].0;
        }
        at += 1;
    }
    most
};

pub(super) const ONE_SHOT_ROWS: u32 = {
    let mut most = 0;
    let mut at = 0;
    while at < FORMS.len() {
        if matches!(FORMS[at].1, Tuned::OneShot) && FORMS[at].0 > most {
            most = FORMS[at].0;
        }
        at += 1;
    }
    most
};

/// The form of `rows` rows in SM120's table ([`FORMS`]): the first bound
/// that holds them, past every bound the phased turns.
pub fn tuned(rows: u32) -> Tuned {
    FORMS.iter().find(|&&(most, _)| rows <= most).map_or(
        Tuned::Turns {
            senders: SENDERS,
            workers: WIDE_WORKERS,
        },
        |&(_, tuned)| tuned,
    )
}

/// The largest plan of each phased entry, which a model's create admits
/// against every rank's residency before it registers them: the one-shot's,
/// the two-shot's and the turns entry's.
pub fn phased_admission_plans() -> Result<Vec<(&'static str, LaunchPlan)>, Refusal> {
    Ok(vec![
        (
            phased_bf16::ONE_SHOT_SYMBOL,
            phased_bf16::plan_with(SENDERS, 0)?,
        ),
        (
            phased_bf16::SYMBOL,
            phased_bf16::plan_with(SENDERS, WIDE_WORKERS)?,
        ),
        (
            phased_turns_bf16::SYMBOL,
            phased_bf16::plan_with(SENDERS, WIDE_WORKERS)?,
        ),
    ])
}

/// A model's registration of the low-latency BF16 entry: the most CTAs of
/// one launch, which its create admitted against every rank's residency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LowLatency {
    pub(super) ctas: u32,
}

/// The model's registrations of the BF16-partials all-reduce.
#[derive(Clone, Debug, Default)]
pub(super) struct Selection {
    pub(super) low_latency: Vec<LowLatency>,
    pub(super) phased: bool,
    pub(super) rounding: Vec<PartialRounding>,
}

/// One row count's launches and the first of their state regions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Form {
    /// The low-latency entry over the state of `held` rows.
    LowLatency { held: u32, state: usize },
    /// A phased entry with its senders and workers.
    Phased {
        symbol: &'static str,
        senders: u32,
        workers: u32,
    },
    /// The rounding rows around the F32 pair.
    Composed,
}

/// A low-latency row count's choice: a registration holding `held` rows,
/// or the form the phased one-shot's counts take.
#[derive(Clone, Copy, Debug)]
enum Pick {
    LowLatency { held: u32 },
    Fallback,
}

pub(super) const SCATTER_SYMBOL: &str = "chop_comm_reduce_scatter_f32_v1";
pub(super) const GATHER_SYMBOL: &str = "chop_comm_allgather_scattered_f32_v1";

/// What `symbol`'s catalog declaration claims every launch keeps.
pub(super) fn declared(symbol: &str) -> Invariance {
    comm::declared_entries()
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .map_or(Invariance::NONE, |entry| entry.invariance)
}

/// The rows of a `width`-element op that registration `entry` holds: its
/// state's capacity at every call. Zero when one row exceeds its grid or
/// the row is no whole number of slices. Refuses a grid past the entry's
/// CTAs.
fn low_latency_rows(entry: &LowLatency, width: u64) -> Result<u32, Refused> {
    let LowLatency { ctas } = *entry;
    if ctas == 0 || ctas > ll_bf16::MAX_CTAS {
        return Err(Refused::Rule(format!(
            "the SM120 low-latency BF16 all-reduce launches 1 to {} CTAs",
            ll_bf16::MAX_CTAS
        )));
    }
    if !width.is_multiple_of(ll_bf16::SLICE_ELEMENTS) {
        return Ok(0);
    }
    Ok((u64::from(ctas) * ll_bf16::SLICE_ELEMENTS / width) as u32)
}

/// What a registration holding `held` rows launches.
fn low_latency_key(held: u32) -> String {
    format!("{}[rows={held}]", ll_bf16::SYMBOL)
}

/// The BF16-partials all-reduce over its row counts' forms.
pub(super) struct Bf16AllReduceOp {
    ranks: u32,
    /// Elements of one row.
    width: u64,
    rank: u32,
    /// The form of each row count from one to the capacity.
    forms: Vec<Form>,
    rounding: Option<PartialRounding>,
    /// The first state region of the phased state, the pair's and the sum.
    phased: Option<usize>,
    pair: Option<usize>,
    sum: Option<usize>,
    info: OpInfo,
}

/// The all-reduce `op` that rank `rank` launches.
pub(super) fn prepare(
    selection: &Selection,
    op: &CommQuery,
    rank: u32,
    need: Need,
) -> Result<Prepared<AllReduceBuffers<View>>, Refused> {
    let op = Bf16AllReduceOp::new(selection, *op, rank, need, false)?;
    need.check(&op.info)?;
    Ok(Box::new(op))
}

impl Bf16AllReduceOp {
    /// The op, with an F32 sum region whatever form serves when `sum`
    /// holds (the comm epilogue's, which [`Self::widen_sum`] fills).
    pub(super) fn new(
        selection: &Selection,
        op: CommQuery,
        rank: u32,
        need: Need,
        sum: bool,
    ) -> Result<Self, Refused> {
        let capacity = need.capacity;
        let CommQuery::AllReduce {
            ranks,
            elements: width,
            dtype: DType::Bf16,
            fused_residual: false,
            fused_rmsnorm: None,
        } = op
        else {
            return Err(Refused::Rule(
                "the SM120 BF16-partials all-reduce sums unfused rows".into(),
            ));
        };
        op.validate()?;
        if rank >= ranks || capacity == 0 {
            return Err(Refused::Rule(
                "an all-reduce needs a rank below its ranks and rows".into(),
            ));
        }
        if ranks as usize > MAX_COMM_RANKS {
            return Err(Refused::Rule(format!(
                "the SM120 BF16-partials all-reduce takes at most {MAX_COMM_RANKS} ranks"
            )));
        }
        if capacity > MAX_CAPACITY {
            return Err(Refused::Rule(format!(
                "an SM120 all-reduce serves at most {MAX_CAPACITY} rows"
            )));
        }
        u32::try_from(width).map_err(|_| {
            Refused::Rule("an SM120 BF16-partials all-reduce row exceeds u32 elements".into())
        })?;
        // Every rank's pair inbox holds the whole reduction in F32.
        let elements = u64::from(capacity)
            .checked_mul(width)
            .filter(|elements| elements.checked_mul(4 * u64::from(ranks)).is_some())
            .ok_or_else(|| Refused::Rule("all-reduce sizes overflow u64".into()))?;
        // The phased entries take whole granules, and the one-shot its
        // most elements at every count the table gives it.
        let one_shot_rows = (1..=capacity.min(TURNS_ROWS))
            .filter(|&rows| tuned(rows) == Tuned::OneShot)
            .max()
            .unwrap_or(0);
        let phased = selection.phased
            && width.is_multiple_of(phased_bf16::GRANULE)
            && u64::from(one_shot_rows) * width <= phased_bf16::ONE_SHOT_MAX_ELEMENTS;
        let rounding = sole(
            selection
                .rounding
                .iter()
                .filter(|pair| u64::from(pair.width) == width)
                .map(Rounding),
            |pair| pair.0.round.symbol.clone(),
            "partial rounding",
        )?
        .map(|pair| pair.0.clone());
        if let Some(pair) = &rounding {
            pair.check()?;
        }
        let mut held = Vec::with_capacity(selection.low_latency.len());
        for entry in &selection.low_latency {
            let rows = low_latency_rows(entry, width)?;
            if ranks.is_power_of_two() && rows > 0 {
                held.push(rows);
            }
        }
        let ll_claims = declared(ll_bf16::SYMBOL);
        let fallback = match phased {
            true => Form::Phased {
                symbol: phased_bf16::ONE_SHOT_SYMBOL,
                senders: SENDERS,
                workers: 0,
            },
            false => Form::Composed,
        };
        let composed_claims = match &rounding {
            Some(pair) => [SCATTER_SYMBOL, GATHER_SYMBOL]
                .map(declared)
                .into_iter()
                .fold(pair.invariance, Invariance::meet),
            None => Invariance::NONE,
        };
        let claims = |form: Form| match form {
            Form::LowLatency { .. } => ll_claims,
            Form::Phased { symbol, .. } => declared(symbol),
            Form::Composed => composed_claims,
        };
        let phased_form = |tuned| match (tuned, phased) {
            (_, false) => Form::Composed,
            (Tuned::LowLatency | Tuned::OneShot, true) => fallback,
            (Tuned::TwoShot { senders, workers }, true) => Form::Phased {
                symbol: phased_bf16::SYMBOL,
                senders,
                workers,
            },
            (Tuned::Turns { senders, workers }, true) => Form::Phased {
                symbol: phased_turns_bf16::SYMBOL,
                senders,
                workers,
            },
        };
        let mut picks = Vec::with_capacity(capacity as usize);
        let mut guarantees: Option<Invariance> = None;
        for rows in 1..=capacity {
            let (form, claimed) = match tuned(rows) {
                Tuned::LowLatency if held.iter().any(|&held| rows <= held) => {
                    let candidates = held.iter().map(|&held| {
                        Candidate::new(
                            Pick::LowLatency { held },
                            ll_bf16::SYMBOL,
                            low_latency_key(held),
                            ll_claims,
                        )
                        .capacity(held)
                    });
                    let other = Candidate::new(
                        Pick::Fallback,
                        "the phased one-shot's form",
                        "fallback".to_owned(),
                        claims(fallback),
                    );
                    let chosen = choose(
                        "BF16-partials all-reduce",
                        candidates.chain([other]),
                        Need {
                            capacity: rows,
                            ..need
                        },
                    )?;
                    let form = match chosen.entry {
                        Pick::LowLatency { held } => Form::LowLatency { held, state: 0 },
                        Pick::Fallback => fallback,
                    };
                    (form, chosen.guarantees)
                }
                tuned => {
                    let form = phased_form(tuned);
                    (form, claims(form))
                }
            };
            if form == Form::Composed {
                if rounding.is_none() {
                    return Err(Refused::Rule(format!(
                        "{rows} rows of the SM120 BF16-partials all-reduce need the composed \
                         form, which reads the model's partial rounding rows of {width} \
                         elements (Sm120::with_partial_rounding)"
                    )));
                }
                rounding::plan(rows)?;
            }
            guarantees = Some(guarantees.map_or(claimed, |held| held.meet(claimed)));
            picks.push(form);
        }
        if sum && rounding.is_none() {
            return Err(Refused::Rule(format!(
                "the SM120 BF16-partials comm epilogue widens its sums through the model's \
                 partial rounding rows of {width} elements (Sm120::with_partial_rounding)"
            )));
        }
        // The state regions in role order.
        let mut state = Vec::new();
        let mut served: Vec<u32> = picks
            .iter()
            .filter_map(|form| match form {
                Form::LowLatency { held, .. } => Some(*held),
                _ => None,
            })
            .collect();
        served.sort_unstable();
        served.dedup();
        let mut ll_states = Vec::with_capacity(served.len());
        for &rows in &served {
            let bytes = ll_bf16::state_bytes(ranks, u64::from(rows) * width)?;
            ll_states.push((rows, state.len()));
            state.extend([bytes.inbox, bytes.epochs]);
        }
        let any = |matches: fn(&Form) -> bool| picks.iter().any(matches);
        let phased_state = any(|form| matches!(form, Form::Phased { .. })).then_some(state.len());
        if phased_state.is_some() {
            let bytes = phased_turns_bf16::buffers(ranks, elements)?;
            state.extend([bytes.inbox_bytes, bytes.flag_bytes, bytes.epoch_bytes]);
        }
        let composed = any(|form| *form == Form::Composed);
        let pair = composed.then_some(state.len());
        if composed {
            let bytes = comm::phased_buffers(ranks, elements, DType::F32)?;
            state.extend([bytes.inbox_bytes, bytes.flag_bytes, bytes.epoch_bytes]);
        }
        let sum_state = (composed || sum).then_some(state.len());
        if sum_state.is_some() {
            state.push(elements * 4);
        }
        let forms = picks
            .into_iter()
            .map(|form| match form {
                Form::LowLatency { held, .. } => {
                    let &(_, at) = ll_states
                        .iter()
                        .find(|(own, _)| *own == held)
                        .expect("every served state is laid out");
                    Form::LowLatency { held, state: at }
                }
                form => form,
            })
            .collect::<Vec<_>>();
        // Every form's plan at every row count, so launches cannot refuse
        // a plan SM120 cannot take.
        for (rows, &form) in (1..=capacity).zip(&forms) {
            for plan in plans(ranks, width, rows, form)? {
                check_plan(&plan)?;
            }
        }
        let mut symbols = Vec::new();
        if !served.is_empty() {
            symbols.push((ll_bf16::SYMBOL.to_owned(), 0));
        }
        for symbol in [
            phased_bf16::ONE_SHOT_SYMBOL,
            phased_bf16::SYMBOL,
            phased_turns_bf16::SYMBOL,
        ] {
            if forms
                .iter()
                .any(|form| matches!(form, Form::Phased { symbol: own, .. } if *own == symbol))
            {
                symbols.push((symbol.to_owned(), 0));
            }
        }
        if composed || sum {
            let pair = rounding.as_ref().expect("checked above");
            if composed {
                symbols.push((pair.round.symbol.clone(), 0));
            }
            symbols.push((pair.widen.symbol.clone(), 0));
        }
        if composed {
            let shared = pair_plan(ranks, width, capacity)?.dynamic_shared_bytes;
            symbols.push((SCATTER_SYMBOL.to_owned(), shared));
            symbols.push((GATHER_SYMBOL.to_owned(), shared));
        }
        let low_latency = match served.is_empty() {
            true => String::new(),
            false => format!(
                "low_latency_rows={},",
                served
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join("/")
            ),
        };
        let transport = if phased { "phased" } else { "composed" };
        let sum_label = if sum { ",sum" } else { "" };
        let mut info = OpInfo::new(
            format!("sm120.comm.allreduce.bf16_partials.v1[{low_latency}{transport}{sum_label}]"),
            guarantees.unwrap_or(Invariance::NONE),
            symbols,
        );
        info.state = state;
        Ok(Self {
            ranks,
            width,
            rank,
            forms,
            rounding,
            phased: phased_state,
            pair,
            sum: sum_state,
            info,
        })
    }

    /// Every rank's `count` state regions from `first`, each checked to
    /// hold its declared bytes.
    fn regions<'a>(
        &self,
        buffers: &'a AllReduceBuffers<View>,
        first: usize,
        count: usize,
    ) -> Result<Vec<&'a [View]>, Refusal> {
        let declared = &self.info.state[first..first + count];
        buffers
            .ranks
            .iter()
            .map(|rank| {
                let state = rank
                    .state
                    .get(first..first + count)
                    .ok_or(Refusal("an all-reduce rank lacks its state regions"))?;
                for (view, &bytes) in state.iter().zip(declared) {
                    view.covers(bytes)?;
                }
                Ok(state)
            })
            .collect()
    }

    /// The live count the rounding rows read.
    fn live_rows(buffers: &AllReduceBuffers<View>) -> Result<View, Refusal> {
        let live = buffers.live_rows.ok_or(Refusal(
            "the composed BF16-partials all-reduce reads a device live count",
        ))?;
        live.covers(4)?;
        Ok(live)
    }

    /// One launch of the rounding row `rows` over `input` and `output`.
    fn rounding_launch(
        &self,
        symbol: &str,
        rows: u32,
        input: View,
        output: View,
        live: View,
    ) -> Result<Launch, Refusal> {
        let plan = rounding::plan(rows)?;
        Ok(Launch {
            symbol: symbol.into(),
            // A row stage records the bucket it processes.
            rows: plan.grid[0],
            plan,
            arguments: rounding::arguments(input.handle, output.handle, live.handle)?,
            collective: false,
            persistent: false,
        })
    }

    /// What the widen row [`Self::widen_sum`] launches declares: the
    /// model's rounding rows' declaration, none without them.
    pub(super) fn widen_claims(&self) -> Invariance {
        self.rounding
            .as_ref()
            .map_or(Invariance::NONE, |pair| pair.invariance)
    }

    /// This rank's F32 sum region of `buffers`, which the comm epilogue
    /// reads.
    pub(super) fn sum_view(&self, buffers: &AllReduceBuffers<View>) -> Result<View, Refusal> {
        let at = self
            .sum
            .ok_or(Refusal("the all-reduce was prepared without a sum region"))?;
        let own = self.regions(buffers, at, 1)?;
        Ok(own[self.rank as usize][0])
    }

    /// The widen row of `rows` rounded sums (the output) into this rank's
    /// F32 sum region, as the comm epilogue reads them.
    pub(super) fn widen_sum(
        &self,
        buffers: &AllReduceBuffers<View>,
        rows: u32,
    ) -> Result<Launch, Refusal> {
        let pair = self
            .rounding
            .as_ref()
            .ok_or(Refusal("the all-reduce has no partial rounding rows"))?;
        let sum = self.sum_view(buffers)?;
        let elements = u64::from(rows) * self.width;
        buffers.output.covers(elements * 2)?;
        sum.covers(elements * 4)?;
        let live = Self::live_rows(buffers)?;
        self.rounding_launch(&pair.widen.symbol, rows, buffers.output, sum, live)
    }
}

/// The plan of the comm engine's F32 pair over `rows` rows.
pub(super) fn pair_plan(ranks: u32, width: u64, rows: u32) -> Result<LaunchPlan, Refusal> {
    let query = CommQuery::AllReduce {
        ranks,
        elements: u64::from(rows) * width,
        dtype: DType::F32,
        fused_residual: false,
        fused_rmsnorm: None,
    };
    comm::plan_variant(Variant::TwoShot, &query, 1)
}

/// Every plan `form` launches at `rows` rows.
fn plans(ranks: u32, width: u64, rows: u32, form: Form) -> Result<Vec<LaunchPlan>, Refusal> {
    Ok(match form {
        Form::LowLatency { .. } => vec![ll_bf16::plan(ranks, u64::from(rows) * width)?],
        Form::Phased {
            senders, workers, ..
        } => vec![phased_bf16::plan_with(senders, workers)?],
        Form::Composed => vec![rounding::plan(rows)?, pair_plan(ranks, width, rows)?],
    })
}

/// A rounding registration compared by everything it launches.
struct Rounding<'a>(&'a PartialRounding);

impl PartialEq for Rounding<'_> {
    fn eq(&self, other: &Self) -> bool {
        format!("{:?}", self.0) == format!("{:?}", other.0)
    }
}

impl PreparedOp<AllReduceBuffers<View>> for Bf16AllReduceOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &AllReduceBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        let form = *rows
            .checked_sub(1)
            .and_then(|index| self.forms.get(index as usize))
            .ok_or(Refusal("all-reduce rows exceed the prepared capacity"))?;
        if buffers.ranks.len() != self.ranks as usize {
            return Err(Refusal("an all-reduce takes the views of every rank"));
        }
        let elements = u64::from(rows) * self.width;
        buffers.input.covers(elements * 4)?;
        buffers.output.covers(elements * 2)?;
        if let Some(live) = buffers.live_rows {
            live.covers(4)?;
        }
        let collective = |symbol: &str, plan, arguments| Launch {
            symbol: symbol.into(),
            rows,
            plan,
            arguments,
            collective: true,
            persistent: false,
        };
        match form {
            Form::LowLatency { held, state } => {
                // Every rank's inbox, this rank's epochs.
                let regions = self.regions(buffers, state, 2)?;
                let arguments = ll_bf16::arguments(&ll_bf16::Call {
                    input: buffers.input.handle,
                    output: buffers.output.handle,
                    peer_inboxes: regions.iter().map(|state| state[0].handle).collect(),
                    epochs: regions[self.rank as usize][1].handle,
                    elements,
                    capacity_elements: u64::from(held) * self.width,
                    rank: self.rank,
                    ranks: self.ranks,
                })?;
                let plan = ll_bf16::plan(self.ranks, elements)?;
                Ok(vec![collective(ll_bf16::SYMBOL, plan, arguments)])
            }
            Form::Phased {
                symbol,
                senders,
                workers,
            } => {
                let state = self.phased.expect("a phased form has its state");
                let regions = self.regions(buffers, state, 3)?;
                for rank in &buffers.ranks {
                    rank.output.covers(elements * 2)?;
                }
                let call = phased_bf16::Call {
                    input: buffers.input.handle,
                    output: buffers.output.handle,
                    inboxes: regions.iter().map(|state| state[0].handle).collect(),
                    outputs: buffers
                        .ranks
                        .iter()
                        .map(|rank| rank.output.handle)
                        .collect(),
                    flags: regions.iter().map(|state| state[1].handle).collect(),
                    epoch: regions[self.rank as usize][2].handle,
                    elements,
                    row_elements: self.width as u32,
                    live_rows: buffers.live_rows.map(|live| live.handle),
                    rank: self.rank,
                    ranks: self.ranks,
                    senders,
                };
                let plan = phased_bf16::plan_with(senders, workers)?;
                Ok(vec![collective(
                    symbol,
                    plan,
                    phased_bf16::arguments(symbol, call)?,
                )])
            }
            Form::Composed => {
                let pair = self
                    .rounding
                    .as_ref()
                    .expect("a composed form has its rounding rows");
                let live = Self::live_rows(buffers)?;
                let state = self.pair.expect("a composed form has its pair state");
                let regions = self.regions(buffers, state, 3)?;
                let sums = self.regions(buffers, self.sum.expect("and its sum"), 1)?;
                for sum in &sums {
                    sum[0].covers(elements * 4)?;
                }
                let own = sums[self.rank as usize][0];
                let (input, output) = (buffers.input, buffers.output);
                let plan = pair_plan(self.ranks, self.width, rows)?;
                let scatter = comm::arguments(OneShotCall {
                    input: input.handle,
                    output: own.handle,
                    peer_inboxes: regions.iter().map(|state| state[0].handle).collect(),
                    peer_flags: regions.iter().map(|state| state[1].handle).collect(),
                    epoch: regions[self.rank as usize][2].handle,
                    elements,
                    rank: self.rank,
                    ranks: self.ranks,
                })?;
                // Each rank's reduced tiles into every rank's sum.
                let gather = ArgumentList(vec![
                    Argument::DeviceBuffer(own.handle),
                    Argument::DevicePointerArray(sums.iter().map(|sum| sum[0].handle).collect()),
                    Argument::DevicePointerArray(
                        regions.iter().map(|state| state[1].handle).collect(),
                    ),
                    Argument::DeviceBuffer(regions[self.rank as usize][2].handle),
                    Argument::U64(elements),
                    Argument::U32(self.rank),
                    Argument::U32(self.ranks),
                ]);
                gather.validate()?;
                Ok(vec![
                    self.rounding_launch(&pair.round.symbol, rows, input, output, live)?,
                    self.rounding_launch(&pair.widen.symbol, rows, output, input, live)?,
                    collective(SCATTER_SYMBOL, plan, scatter),
                    collective(GATHER_SYMBOL, plan, gather),
                    self.rounding_launch(&pair.round.symbol, rows, own, output, live)?,
                ])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};
    use kernel_core::backend::Backend;
    use kernel_core::collective::RankViews;

    const WIDTH: u64 = 4096;
    /// A capacity past the turns entry's first rows.
    const CAPACITY: u32 = TURNS_ROWS + 8;

    fn reduce(ranks: u32, width: u64) -> CommQuery {
        CommQuery::AllReduce {
            ranks,
            elements: width,
            dtype: DType::Bf16,
            fused_residual: false,
            fused_rmsnorm: None,
        }
    }

    /// The low-latency grid of the table's rows at `width`, at least one
    /// CTA.
    fn ll_ctas(width: u64) -> u32 {
        (u64::from(LOW_LATENCY_ROWS) * width / ll_bf16::SLICE_ELEMENTS).max(1) as u32
    }

    /// What a model registers: the low-latency entry, the phased entries,
    /// the rounding rows at `width`.
    #[derive(Clone, Copy, Debug)]
    struct Registered {
        low_latency: bool,
        phased: bool,
        rounding: bool,
    }

    const ALL: Registered = Registered {
        low_latency: true,
        phased: true,
        rounding: true,
    };

    /// What the composed form claims over the shared rounding rows: the
    /// meet of the F32 pair's declarations and theirs.
    fn composed_claims() -> Invariance {
        [SCATTER_SYMBOL, GATHER_SYMBOL]
            .map(declared)
            .into_iter()
            .fold(rounding::shared_invariance(), Invariance::meet)
    }

    /// The composed form claims what its stages declare together: one-row
    /// bits up to the pair's 8192 rows, any chunk split and row
    /// independence over the shared rounding rows.
    #[test]
    fn the_composed_form_claims_its_stages_together() {
        assert_eq!(composed_claims(), Invariance::any_rows(8192));
        let composed = backend(
            Registered {
                phased: false,
                low_latency: false,
                ..ALL
            },
            WIDTH,
        );
        let exact = Need::rows(64).with_exact(Invariance::any_rows(64));
        let op = composed.all_reduce(&reduce(4, WIDTH), 0, exact).unwrap();
        assert_eq!(op.info().guarantees, Invariance::any_rows(8192));
    }

    fn backend(registered: Registered, width: u64) -> Sm120 {
        let mut backend = Sm120::new();
        if registered.low_latency {
            backend = backend.with_low_latency_bf16_all_reduce(ll_ctas(width));
        }
        if registered.phased {
            backend = backend.with_phased_bf16_all_reduce();
        }
        if registered.rounding {
            backend = backend.with_partial_rounding(rounding::shared(width as u32));
        }
        backend
    }

    /// Rank `r`'s output is handle 10 + r and its state regions
    /// 1000 * (r + 1) + i; the input is 1, the output 2 and the live
    /// count 3.
    fn buffers(info: &OpInfo, ranks: u32) -> AllReduceBuffers<View> {
        AllReduceBuffers {
            input: view(1, u64::MAX),
            output: view(2, u64::MAX),
            live_rows: Some(view(3, 4)),
            ranks: (0..ranks)
                .map(|r| RankViews {
                    output: view(10 + u64::from(r), u64::MAX),
                    state: info
                        .state
                        .iter()
                        .enumerate()
                        .map(|(i, &bytes)| view(handle(r, i), bytes))
                        .collect(),
                })
                .collect(),
        }
    }

    fn handle(rank: u32, region: usize) -> u64 {
        1000 * (u64::from(rank) + 1) + region as u64
    }

    /// The state layout a model allocates for `registered` at `capacity`
    /// rows: the low-latency state of the table's rows, the phased state of
    /// the capacity (turns flags), the pair's and the F32 sum where the
    /// composed form serves; and where each starts.
    fn layout(
        ranks: u32,
        width: u64,
        capacity: u32,
        registered: Registered,
    ) -> (Vec<u64>, [Option<usize>; 4]) {
        let elements = u64::from(capacity) * width;
        let mut state = Vec::new();
        let mut at = [None; 4];
        if registered.low_latency && ranks.is_power_of_two() {
            at[0] = Some(state.len());
            let ll = ll_bf16::state_bytes(ranks, u64::from(LOW_LATENCY_ROWS) * width).unwrap();
            state.extend([ll.inbox, ll.epochs]);
        }
        let low_latency_only =
            registered.low_latency && ranks.is_power_of_two() && capacity <= LOW_LATENCY_ROWS;
        let phased = registered.phased && !low_latency_only;
        if phased {
            at[1] = Some(state.len());
            let bytes = phased_turns_bf16::buffers(ranks, elements).unwrap();
            state.extend([bytes.inbox_bytes, bytes.flag_bytes, bytes.epoch_bytes]);
        }
        if !registered.phased {
            at[2] = Some(state.len());
            let bytes = comm::phased_buffers(ranks, elements, DType::F32).unwrap();
            state.extend([bytes.inbox_bytes, bytes.flag_bytes, bytes.epoch_bytes]);
            at[3] = Some(state.len());
            state.push(elements * 4);
        }
        (state, at)
    }

    /// What a model's lowering launches for `rows` rows on rank `rank` of
    /// `ranks` with `registered`: SM120's table (the low-latency entry at
    /// one row over its state of the table's rows; the phased one-shot to
    /// four rows; the two-shot of 16 senders to 32 rows, with 32 workers
    /// past them; the turns of 16 and 32 from 1024 rows), the phased
    /// one-shot in place of a low-latency entry not registered or not
    /// taken, the composed form in place of every phased entry not
    /// registered.
    fn lowering(
        ranks: u32,
        rank: u32,
        rows: u32,
        registered: Registered,
        at: [Option<usize>; 4],
    ) -> Vec<Launch> {
        let elements = u64::from(rows) * WIDTH;
        let collective = |symbol: &str, plan, arguments| Launch {
            symbol: symbol.into(),
            rows,
            plan,
            arguments,
            collective: true,
            persistent: false,
        };
        let low_latency = registered.low_latency && ranks.is_power_of_two() && rows <= 1;
        if low_latency {
            let ll = at[0].unwrap();
            let call = ll_bf16::Call {
                input: 1,
                output: 2,
                peer_inboxes: (0..ranks).map(|r| handle(r, ll)).collect(),
                epochs: handle(rank, ll + 1),
                elements,
                capacity_elements: WIDTH,
                rank,
                ranks,
            };
            return vec![collective(
                ll_bf16::SYMBOL,
                ll_bf16::plan(ranks, elements).unwrap(),
                ll_bf16::arguments(&call).unwrap(),
            )];
        }
        if registered.phased {
            let p = at[1].unwrap();
            let (symbol, workers) = match rows {
                ..=4 => (phased_bf16::ONE_SHOT_SYMBOL, 0),
                5..=32 => (phased_bf16::SYMBOL, 0),
                33..=1023 => (phased_bf16::SYMBOL, 32),
                _ => (phased_turns_bf16::SYMBOL, 32),
            };
            let call = phased_bf16::Call {
                input: 1,
                output: 2,
                inboxes: (0..ranks).map(|r| handle(r, p)).collect(),
                outputs: (0..ranks).map(|r| 10 + u64::from(r)).collect(),
                flags: (0..ranks).map(|r| handle(r, p + 1)).collect(),
                epoch: handle(rank, p + 2),
                elements,
                row_elements: WIDTH as u32,
                live_rows: Some(3),
                rank,
                ranks,
                senders: 16,
            };
            return vec![collective(
                symbol,
                phased_bf16::plan_with(16, workers).unwrap(),
                phased_bf16::arguments(symbol, call).unwrap(),
            )];
        }
        let (pair, sum) = (at[2].unwrap(), at[3].unwrap());
        let bucket = crate::row::smallest_bucket(rows).unwrap();
        let row = |symbol: String, input, output| Launch {
            symbol,
            rows: bucket,
            plan: rounding::plan(rows).unwrap(),
            arguments: rounding::arguments(input, output, 3).unwrap(),
            collective: false,
            persistent: false,
        };
        let round = rounding::shared_symbol(rounding::Rounding::Round, WIDTH as u32);
        let widen = rounding::shared_symbol(rounding::Rounding::Widen, WIDTH as u32);
        let query = CommQuery::AllReduce {
            ranks,
            elements,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        let plan = comm::plan_variant(Variant::TwoShot, &query, 1).unwrap();
        let scatter = comm::arguments(OneShotCall {
            input: 1,
            output: handle(rank, sum),
            peer_inboxes: (0..ranks).map(|r| handle(r, pair)).collect(),
            peer_flags: (0..ranks).map(|r| handle(r, pair + 1)).collect(),
            epoch: handle(rank, pair + 2),
            elements,
            rank,
            ranks,
        })
        .unwrap();
        let gather = ArgumentList(vec![
            Argument::DeviceBuffer(handle(rank, sum)),
            Argument::DevicePointerArray((0..ranks).map(|r| handle(r, sum)).collect()),
            Argument::DevicePointerArray((0..ranks).map(|r| handle(r, pair + 1)).collect()),
            Argument::DeviceBuffer(handle(rank, pair + 2)),
            Argument::U64(elements),
            Argument::U32(rank),
            Argument::U32(ranks),
        ]);
        vec![
            row(round.clone(), 1, 2),
            row(widen, 2, 1),
            collective(SCATTER_SYMBOL, plan, scatter),
            collective(GATHER_SYMBOL, plan, gather),
            row(round, handle(rank, sum), 2),
        ]
    }

    /// At 2, 4, 6 and 8 ranks, with each registration set, every row
    /// count up to past the turns entry's first launches what the lowering
    /// does over the layout it allocates.
    #[test]
    fn launches_equal_the_lowering_at_every_row_count() {
        let sets = [
            ALL,
            Registered {
                low_latency: false,
                ..ALL
            },
            Registered {
                phased: false,
                ..ALL
            },
            Registered {
                low_latency: false,
                phased: false,
                rounding: true,
            },
            Registered {
                rounding: false,
                ..ALL
            },
        ];
        for ranks in [2, 4, 6, 8] {
            for registered in sets {
                let rank = ranks - 1;
                let prepared = backend(registered, WIDTH)
                    .all_reduce(&reduce(ranks, WIDTH), rank, Need::rows(CAPACITY))
                    .unwrap();
                let info = prepared.info();
                let (state, at) = layout(ranks, WIDTH, CAPACITY, registered);
                assert_eq!(info.state, state, "{ranks} ranks, {registered:?}");
                let buffers = buffers(info, ranks);
                let mut symbols = Vec::new();
                for rows in 1..=CAPACITY {
                    let launches = prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap();
                    assert_eq!(
                        launches,
                        lowering(ranks, rank, rows, registered, at),
                        "{ranks} ranks, {registered:?}, {rows} rows"
                    );
                    for launch in launches {
                        if !symbols.contains(&launch.symbol) {
                            symbols.push(launch.symbol);
                        }
                    }
                }
                // The info lists every launched symbol.
                let mut listed: Vec<_> = info.symbols.iter().map(|(s, _)| s.clone()).collect();
                listed.sort();
                symbols.sort();
                assert_eq!(listed, symbols, "{ranks} ranks, {registered:?}");
                // The entries' declarations hold together, and the
                // composed form's stages' with them.
                let guarantees = match registered.phased {
                    true => declared(phased_bf16::SYMBOL),
                    false => composed_claims(),
                };
                assert_eq!(info.guarantees, guarantees, "{registered:?}");
            }
        }
    }

    /// The table's bounds: the forms change exactly past 1, 4, 32 and 1023
    /// rows, and [`tuned`] names the entry and grid that launch.
    #[test]
    fn forms_change_at_the_table_bounds() {
        let prepared = backend(ALL, WIDTH)
            .all_reduce(&reduce(4, WIDTH), 0, Need::rows(CAPACITY))
            .unwrap();
        let buffers = buffers(prepared.info(), 4);
        let form = |rows| {
            let launches = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            (launches[0].symbol.clone(), launches[0].plan.grid[0])
        };
        let ll = (ll_bf16::SYMBOL.to_owned(), 2);
        let one = (phased_bf16::ONE_SHOT_SYMBOL.to_owned(), 16);
        let two = |grid| (phased_bf16::SYMBOL.to_owned(), grid);
        let turns = (phased_turns_bf16::SYMBOL.to_owned(), 48);
        let expected = [
            (1, ll),
            (2, one.clone()),
            (4, one),
            (5, two(16)),
            (32, two(16)),
            (33, two(48)),
            (1023, two(48)),
            (1024, turns.clone()),
            (CAPACITY, turns),
        ];
        for (rows, want) in expected {
            assert_eq!(form(rows), want, "{rows} rows");
            let tuned = tuned(rows);
            assert_eq!(tuned.symbol(), want.0, "{rows} rows");
            if let Tuned::TwoShot { senders, workers } | Tuned::Turns { senders, workers } = tuned {
                assert_eq!(senders + workers, want.1, "{rows} rows");
            }
        }
        assert_eq!(
            prepared.info().implementation,
            "sm120.comm.allreduce.bf16_partials.v1[low_latency_rows=1,phased]"
        );
    }

    /// Each admission plan is the widest grid the table launches its
    /// entry at.
    #[test]
    fn admission_plans_are_the_widest_grids() {
        let prepared = backend(ALL, WIDTH)
            .all_reduce(&reduce(4, WIDTH), 0, Need::rows(CAPACITY))
            .unwrap();
        let buffers = buffers(prepared.info(), 4);
        let mut widest = std::collections::BTreeMap::<String, LaunchPlan>::new();
        for rows in 1..=CAPACITY {
            for launch in prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap()
            {
                let plan = widest.entry(launch.symbol).or_insert(launch.plan);
                if launch.plan.grid[0] > plan.grid[0] {
                    *plan = launch.plan;
                }
            }
        }
        for (symbol, plan) in phased_admission_plans().unwrap() {
            assert_eq!(widest[symbol], plan, "{symbol}");
        }
        assert_eq!(LOW_LATENCY_ROWS, 1);
        assert_eq!(
            widest[ll_bf16::SYMBOL],
            ll_bf16::plan(4, u64::from(LOW_LATENCY_ROWS) * WIDTH).unwrap()
        );
    }

    /// Registrations an op cannot take leave its rows to the forms after
    /// them: a row of partial slices or an odd rank count passes the
    /// low-latency entry over, a row of partial granules the phased
    /// entries.
    #[test]
    fn ops_the_entries_do_not_take_fall_through() {
        let first = |width: u64, ranks: u32| {
            let prepared = backend(ALL, width)
                .all_reduce(&reduce(ranks, width), 0, Need::rows(8))
                .unwrap();
            let buffers = buffers(prepared.info(), ranks);
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(1))
                .unwrap()[0]
                .symbol
                .clone()
        };
        assert_eq!(first(4096, 4), ll_bf16::SYMBOL);
        assert_eq!(first(3072, 4), phased_bf16::ONE_SHOT_SYMBOL);
        assert_eq!(first(4096, 6), phased_bf16::ONE_SHOT_SYMBOL);
        assert_eq!(
            first(1000, 4),
            rounding::shared_symbol(rounding::Rounding::Round, 1000)
        );
        // A grid that holds more rows serves at most the table's.
        let wide = Sm120::new()
            .with_low_latency_bf16_all_reduce(8)
            .with_phased_bf16_all_reduce()
            .all_reduce(&reduce(4, WIDTH), 0, Need::rows(8))
            .unwrap();
        let buffers = buffers(wide.info(), 4);
        let at = |rows| {
            wide.launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap()
        };
        assert_eq!(at(1)[0].arguments.0[5], Argument::U32(8));
        assert_eq!(at(2)[0].symbol, phased_bf16::ONE_SHOT_SYMBOL);
        assert!(
            wide.info()
                .implementation
                .contains("low_latency_rows=4,phased")
        );
    }

    /// The choice follows the op's fields and the registrations in any
    /// order; a registration repeated is one.
    #[test]
    fn registrations_serve_in_every_order() {
        let prepare = |order: &[u32]| {
            let backend = order.iter().fold(
                Sm120::new().with_phased_bf16_all_reduce(),
                |backend, &ctas| backend.with_low_latency_bf16_all_reduce(ctas),
            );
            let prepared = backend
                .all_reduce(&reduce(4, WIDTH), 1, Need::rows(8))
                .unwrap();
            let info = prepared.info().clone();
            let buffers = buffers(&info, 4);
            let launches = (1..=8)
                .map(|rows| {
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                })
                .collect::<Vec<_>>();
            (info, launches)
        };
        let registered = [6, 2, 4];
        let all = crate::backend::select::orders(&registered);
        let first = prepare(&all[0]);
        for order in &all[1..] {
            assert_eq!(prepare(order), first, "{order:?}");
        }
        // The least grid that holds a row serves it.
        assert_eq!(first.1[0][0].arguments.0[5], Argument::U32(2));
        assert_eq!(prepare(&[2, 2]), prepare(&[2]));
        // Two rounding registrations of one width that launch differently
        // tie.
        let mut other = rounding::shared(WIDTH as u32);
        other.round.symbol = "chop_row_other_round_v1".into();
        let tied = Sm120::new()
            .with_partial_rounding(rounding::shared(WIDTH as u32))
            .with_partial_rounding(other)
            .all_reduce(&reduce(4, WIDTH), 0, Need::rows(8));
        assert!(matches!(tied, Err(Refused::Rule(rule)) if rule.contains("two SM120")));
        let repeated = Sm120::new()
            .with_partial_rounding(rounding::shared(WIDTH as u32))
            .with_partial_rounding(rounding::shared(WIDTH as u32))
            .all_reduce(&reduce(4, WIDTH), 0, Need::rows(8));
        assert!(repeated.is_ok());
    }

    #[test]
    fn prepare_refuses_what_no_form_serves() {
        let refused = |backend: Sm120, query: CommQuery, need: Need| match backend
            .all_reduce(&query, 0, need)
        {
            Err(Refused::Rule(rule)) => rule,
            Err(other) => format!("{other:?}"),
            Ok(_) => panic!("{query:?} prepared"),
        };
        // Rows left to the composed form need the rounding rows.
        let rule = refused(
            backend(
                Registered {
                    rounding: false,
                    phased: false,
                    ..ALL
                },
                WIDTH,
            ),
            reduce(4, WIDTH),
            Need::rows(2),
        );
        assert!(rule.contains("with_partial_rounding"), "{rule}");
        // Fused BF16 queries are the fused all-reduce's.
        let fused = CommQuery::AllReduce {
            ranks: 4,
            elements: WIDTH,
            dtype: DType::Bf16,
            fused_residual: true,
            fused_rmsnorm: None,
        };
        assert!(refused(backend(ALL, WIDTH), fused, Need::rows(2)).contains("plain F32"));
        assert!(refused(backend(ALL, WIDTH), reduce(16, WIDTH), Need::rows(2)).contains("8 ranks"));
        assert!(
            refused(
                backend(ALL, WIDTH),
                reduce(4, WIDTH),
                Need::rows(MAX_CAPACITY + 1)
            )
            .contains("at most")
        );
        // The composed form's rows past the largest row bucket.
        let composed = backend(
            Registered {
                phased: false,
                ..ALL
            },
            WIDTH,
        );
        assert!(
            composed
                .all_reduce(&reduce(4, WIDTH), 0, Need::rows(1 << 14))
                .is_err()
        );
        // A grid past the entry's CTAs.
        for ctas in [0, ll_bf16::MAX_CTAS + 1] {
            let rule = refused(
                Sm120::new().with_low_latency_bf16_all_reduce(ctas),
                reduce(4, WIDTH),
                Need::rows(2),
            );
            assert!(rule.contains("1 to 128 CTAs"), "{rule}");
        }
        // The composed form keeps what its stages declare together, and
        // refuses exactness its rounding rows do not declare, which the
        // entries keep.
        let exact = Need::rows(2).with_exact(Invariance::one_row(2));
        assert!(composed.all_reduce(&reduce(4, WIDTH), 0, exact).is_ok());
        let undeclared = Sm120::new().with_partial_rounding(PartialRounding {
            invariance: Invariance::NONE,
            ..rounding::shared(WIDTH as u32)
        });
        assert!(matches!(
            undeclared.all_reduce(&reduce(4, WIDTH), 0, exact),
            Err(Refused::Inexact { .. })
        ));
        assert!(
            backend(ALL, WIDTH)
                .all_reduce(&reduce(4, WIDTH), 0, exact)
                .is_ok()
        );
        // F32 all-reduces keep their op.
        let f32 = CommQuery::AllReduce {
            ranks: 4,
            elements: WIDTH,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        let plain = Sm120::new().all_reduce(&f32, 0, Need::rows(8)).unwrap();
        let with = backend(ALL, WIDTH)
            .all_reduce(&f32, 0, Need::rows(8))
            .unwrap();
        assert_eq!(plain.info(), with.info());
    }

    #[test]
    fn launches_check_every_view_and_count() {
        let prepared = backend(
            Registered {
                phased: false,
                ..ALL
            },
            WIDTH,
        )
        .all_reduce(&reduce(4, WIDTH), 2, Need::rows(16))
        .unwrap();
        let info = prepared.info().clone();
        let buffers = buffers(&info, 4);
        let ok = |buffers: &AllReduceBuffers<View>, rows| {
            prepared
                .launches(buffers, View::EMPTY, Counts::rows(rows))
                .is_ok()
        };
        assert!(ok(&buffers, 1) && ok(&buffers, 16));
        assert!(!ok(&buffers, 17) && !ok(&buffers, 0));
        // The composed rows read the live count; the low-latency row not.
        let unlived = AllReduceBuffers {
            live_rows: None,
            ..buffers.clone()
        };
        assert!(ok(&unlived, 1) && !ok(&unlived, 2));
        let mut short = buffers.clone();
        short.input.bytes = 16 * WIDTH * 4 - 1;
        assert!(!ok(&short, 16));
        let mut short = buffers.clone();
        short.output.bytes = 16 * WIDTH * 2 - 1;
        assert!(!ok(&short, 16));
        // A peer's sum and pair flags, this rank's pair epoch.
        let last = info.state.len() - 1;
        for (rank, region) in [(0, last), (1, last - 2), (2, last - 1)] {
            let mut short = buffers.clone();
            short.ranks[rank].state[region].bytes -= 1;
            assert!(!ok(&short, 16), "rank {rank} region {region}");
        }
        let missing = AllReduceBuffers {
            ranks: buffers.ranks[..3].to_vec(),
            ..buffers.clone()
        };
        assert!(!ok(&missing, 1));
    }

    /// The comm epilogue over BF16 partials: the all-reduce's launches as
    /// `Backend::all_reduce` gives them, its rounded sums widened into the
    /// F32 sum region the op adds, then the split epilogue over those sums;
    /// at the composed form's rows the sum region is the pair's. Without
    /// rounding rows the op is refused.
    #[test]
    fn the_comm_epilogue_widens_the_rounded_sums() {
        use crate::comm::sinkhorn_epilogue::EpilogueBuffers;
        use crate::comm::sinkhorn_split::{self, SplitBuffers};
        use crate::sinkhorn_hc::SERVED;
        use crate::{Profile, ProfileNames};
        use kernel_core::sinkhorn_hyper_connection::{
            MixScratch, MixerWeights, ReduceExpandMixNormBuffers, SinkhornHyperConnectionQuery,
        };
        const PROFILE: Profile = Profile::synthetic(ProfileNames {
            name: "bf16-epilogue-test",
            tag: "test",
        });
        let eps = 1e-5_f32.to_bits();
        let (ranks, rank, capacity) = (4, 1, 64);
        let entry = sinkhorn_split::entry(PROFILE, SERVED.mix, eps, SERVED.capacity).unwrap();
        let query = |dtype| SinkhornHyperConnectionQuery::AllReduceExpandMixNorm {
            ranks,
            dtype,
            mix: SERVED.mix,
            rmsnorm_eps_bits: eps,
            norm: kernel_core::rmsnorm::RmsNormReduction::Ordered,
            capacity,
        };
        for registered in [
            ALL,
            Registered {
                phased: false,
                ..ALL
            },
        ] {
            let backend = backend(registered, WIDTH).with_sinkhorn_epilogue(entry.clone());
            let prepared = backend
                .sinkhorn_reduce_expand_mix_norm(&query(DType::Bf16), rank, Need::rows(capacity))
                .unwrap();
            let reduce = backend
                .all_reduce(&reduce(ranks, WIDTH), rank, Need::rows(capacity))
                .unwrap();
            let info = prepared.info();
            // The all-reduce's state, and the sum region after it unless
            // the composed form already holds one.
            let mut state = reduce.info().state.clone();
            if registered.phased {
                state.push(u64::from(capacity) * WIDTH * 4);
            }
            assert_eq!(info.state, state, "{registered:?}");
            let sum = state.len() - 1;
            let mut symbols: Vec<_> = reduce.info().symbols.clone();
            let widen = rounding::shared_symbol(rounding::Rounding::Widen, WIDTH as u32);
            if !symbols.iter().any(|(symbol, _)| *symbol == widen) {
                symbols.push((widen.clone(), 0));
            }
            symbols.push((entry.symbol.clone(), 0));
            assert_eq!(info.symbols, symbols);
            let view = |handle| view(handle, u64::MAX);
            let buffers = ReduceExpandMixNormBuffers {
                reduce: buffers(info, ranks),
                streams: view(20),
                reduced: view(21),
                expanded: view(22),
                mixer: MixerWeights {
                    fn_: view(23),
                    base: view(24),
                    scale: view(25),
                },
                post: view(26),
                comb: view(27),
                collapsed: view(28),
                norm_weight: view(29),
                normed: view(30),
                scratch: MixScratch {
                    partials: Some(view(31)),
                    tickets: Some(view(32)),
                },
                live_rows: view(3),
            };
            let reduce_buffers = AllReduceBuffers {
                ranks: buffers
                    .reduce
                    .ranks
                    .iter()
                    .map(|rank| RankViews {
                        state: rank.state[..reduce.info().state.len()].to_vec(),
                        ..rank.clone()
                    })
                    .collect(),
                ..buffers.reduce.clone()
            };
            let split = SplitBuffers {
                epilogue: EpilogueBuffers {
                    streams: 20,
                    post: 26,
                    comb: 27,
                    reduced: 21,
                    expanded: 22,
                    fn_: 23,
                    base: 24,
                    scale: 25,
                    collapsed: 28,
                    weight: 29,
                    normed: 30,
                    live_rows: 3,
                    eps_bits: eps,
                },
                partials: 31,
                tickets: 32,
            };
            for rows in [1, 2, 5, 33, capacity] {
                let launches = prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                let mut expected = reduce
                    .launches(&reduce_buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                let plan = rounding::plan(rows).unwrap();
                expected.push(Launch {
                    symbol: widen.clone(),
                    rows: plan.grid[0],
                    plan,
                    arguments: rounding::arguments(2, handle(rank, sum), 3).unwrap(),
                    collective: false,
                    persistent: false,
                });
                expected.push(Launch {
                    symbol: entry.symbol.clone(),
                    rows,
                    plan: entry.plan(rows, ranks).unwrap(),
                    arguments: sinkhorn_split::rows_arguments(handle(rank, sum), split).unwrap(),
                    collective: false,
                    persistent: false,
                });
                assert_eq!(launches, expected, "{registered:?}, {rows} rows");
            }
        }
        let unrounded = backend(
            Registered {
                rounding: false,
                ..ALL
            },
            WIDTH,
        )
        .with_sinkhorn_epilogue(entry.clone());
        assert!(matches!(
            unrounded.sinkhorn_reduce_expand_mix_norm(&query(DType::Bf16), 0, Need::rows(8)),
            Err(Refused::Rule(rule)) if rule.contains("with_partial_rounding")
        ));
        // F32 partials keep the F32 all-reduce.
        let f32 = unrounded
            .sinkhorn_reduce_expand_mix_norm(&query(DType::F32), 0, Need::rows(8))
            .unwrap();
        assert!(
            f32.info()
                .implementation
                .starts_with("sm120.comm.allreduce.f32.v1")
        );
    }
}
