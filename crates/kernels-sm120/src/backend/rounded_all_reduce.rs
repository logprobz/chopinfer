//! The rounded all-reduce (`CommQuery::AllReduceRounded`): every rank's F32
//! rows summed in F32 in rank order from zero, each sum rounded once to
//! BF16 into the BF16 output. Every form below gives those bits, so the
//! choice of form moves no bit.
//!
//! Each row count's form is chosen when the op is prepared, from SM120's
//! table ([`FORMS`]) and the model's registrations:
//!
//! - The low-latency round entry (`comm::ll_f32::ROUND_SYMBOL`) at the
//!   table's low-latency counts, through a registration whose admitted grid
//!   holds the rows (`Sm120::with_low_latency_rounded_all_reduce`), at
//!   power-of-two rank counts and rows of whole 1024-element slices; else
//!   what serves the one-shot round's counts.
//! - The phased F32 entries' rounding forms (`comm::phased_f32`): the
//!   one-shot round entry at the table's one-shot counts, the round
//!   two-shot above them at the table's senders and workers, once the model
//!   registers them (`Sm120::with_phased_f32_all_reduce`, after its create
//!   admitted their largest grids, [`phased_admission_plans`]) and the rows
//!   are whole 1024-element granules.
//! - Otherwise the composed form: the comm engine's F32 reduce-scatter then
//!   all-gather into every rank's F32 sum, and the model's round row of the
//!   sum into the output (`Sm120::with_partial_rounding`). Without a
//!   rounding registration of the op's width, an op with a row count left
//!   to it is refused.
//!
//! An entry whose declarations do not keep `Need.exact` is passed over at
//! the low-latency counts, and the op claims what every launched form
//! declares together (their meet): the entries' declarations
//! (`comm::declared_entries`), or the F32 pair's with the model's rounding
//! row's. Each keeps a row's bits apart
//! from its row count and the other rows (an element's owner sums every
//! rank in rank order, whichever rank owns it), so their meet holds for
//! the op.
//!
//! State regions, in order (`OpInfo::state`): each low-latency state that
//! serves (its inbox and epochs, least rows first), at one capacity
//! whatever row count launches it; the phased F32 state every phased row
//! count shares (inbox, flags, epoch); the F32 pair's (inbox, flags, epoch)
//! and the F32 sum `[capacity, width]` that the pair's all-gather writes on
//! every rank, where the composed form serves. The phased entries write
//! every rank's output (`RankViews::output`); the other forms write this
//! rank's. The phased entries and the composed form read the device live
//! count (`AllReduceBuffers::live_rows`): the phased entries reduce the
//! rows below it when given one, the composed form refuses launches without
//! one. The low-latency entry reduces every row of its launch.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::collective::AllReduceBuffers;
use kernel_core::contract::{
    Argument, ArgumentList, CommQuery, DType, LaunchPlan, MAX_COMM_RANKS, Refusal, Validate,
};
use kernel_core::properties::Invariance;

use super::bf16_all_reduce::{GATHER_SYMBOL, SCATTER_SYMBOL, declared, pair_plan};
use super::select::{Candidate, choose, sole};
use super::{MAX_CAPACITY, check_plan};
use crate::comm::{self, OneShotCall, ll_f32, phased_f32};
use crate::row::rounding::{self, PartialRounding};

/// Sending CTAs of every phased call, and the workers of a two-shot call
/// above [`DECODE_ROWS`].
const SENDERS: u32 = 16;
const WIDE_WORKERS: u32 = 32;
/// The most rows of the one-shot round entry.
pub(super) const ONE_SHOT_ROWS: u32 = 4;
/// The most rows of a two-shot call without workers.
const DECODE_ROWS: u32 = 32;

/// A row count's form in SM120's table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Tuned {
    LowLatency,
    OneShot,
    TwoShot { senders: u32, workers: u32 },
}

/// SM120's form of each row count: the first bound that holds the rows,
/// past every bound the two-shot with workers. Timed on four ranks of one
/// PCIe host (kernel-lab `phased_bf16_timing`, us per call at 16 senders):
/// the one-shot round 8.4, 9.0 and 10.9 at 1, 2 and 4 rows against the
/// round two-shot's 14.6, 15.3 and 15.7; the round two-shot 16.5, 18.9 and
/// 26.5 at 8, 16 and 32 rows alone; at 1,024 to 8,192 rows with 32 workers
/// 539, 910, 1,833 and 3,646. The low-latency transport (the F32 entry's,
/// which the round entry shares) takes 5.6 us at one row and 9.9 at two
/// (kernel-lab `allreduce_ll_f32`), so it serves one row.
const FORMS: [(u32, Tuned); 3] = [
    (LOW_LATENCY_ROWS, Tuned::LowLatency),
    (ONE_SHOT_ROWS, Tuned::OneShot),
    (
        DECODE_ROWS,
        Tuned::TwoShot {
            senders: SENDERS,
            workers: 0,
        },
    ),
];

/// The most rows the table gives the low-latency round entry: the rows a
/// model's create admits its grid for (`ll_f32::plan` of these rows).
pub const LOW_LATENCY_ROWS: u32 = 1;

fn tuned(rows: u32) -> Tuned {
    FORMS.iter().find(|&&(most, _)| rows <= most).map_or(
        Tuned::TwoShot {
            senders: SENDERS,
            workers: WIDE_WORKERS,
        },
        |&(_, tuned)| tuned,
    )
}

/// The largest plan of each phased entry, which a model's create admits
/// against every rank's residency before it registers them: the one-shot
/// round's and the round two-shot's.
pub fn phased_admission_plans() -> Result<Vec<(&'static str, LaunchPlan)>, Refusal> {
    Ok(vec![
        (
            phased_f32::ONE_SHOT_ROUND_SYMBOL,
            phased_f32::plan_with(SENDERS, 0)?,
        ),
        (
            phased_f32::ROUND_SYMBOL,
            phased_f32::plan_with(SENDERS, WIDE_WORKERS)?,
        ),
    ])
}

/// A model's registration of the low-latency round entry: the most CTAs
/// of one launch, which its create admitted against every rank's
/// residency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LowLatency {
    pub(super) ctas: u32,
}

/// The model's registrations of the low-latency round entry and the
/// phased F32 entries.
#[derive(Clone, Debug, Default)]
pub(super) struct Selection {
    pub(super) low_latency: Vec<LowLatency>,
    pub(super) phased: bool,
}

/// The rows of a `width`-element op that registration `entry` holds: its
/// state's capacity at every call. Zero when one row exceeds its grid or
/// the row is no whole number of slices. Refuses a grid past the entry's
/// CTAs.
fn low_latency_rows(entry: &LowLatency, width: u64) -> Result<u32, Refused> {
    let LowLatency { ctas } = *entry;
    if ctas == 0 || ctas > ll_f32::MAX_CTAS {
        return Err(Refused::Rule(format!(
            "the SM120 low-latency rounded all-reduce launches 1 to {} CTAs",
            ll_f32::MAX_CTAS
        )));
    }
    if !width.is_multiple_of(ll_f32::SLICE_ELEMENTS) {
        return Ok(0);
    }
    Ok((u64::from(ctas) * ll_f32::SLICE_ELEMENTS / width) as u32)
}

/// What a registration holding `held` rows launches.
fn low_latency_key(held: u32) -> String {
    format!("{}[rows={held}]", ll_f32::ROUND_SYMBOL)
}

/// A low-latency row count's choice: a registration holding `held` rows,
/// or the form the one-shot round's counts take.
#[derive(Clone, Copy, Debug)]
enum Pick {
    LowLatency { held: u32 },
    Fallback,
}

/// One row count's launches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Form {
    /// The low-latency round entry over the state of `held` rows.
    LowLatency { held: u32, state: usize },
    /// A phased rounding entry with its senders and workers.
    Phased {
        symbol: &'static str,
        senders: u32,
        workers: u32,
    },
    /// The F32 pair, then the round row.
    Composed,
}

/// The rounded all-reduce over its row counts' forms.
struct RoundedAllReduceOp {
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

/// The rounded all-reduce `op` that rank `rank` launches, through the
/// phased entries where `selection` registers them and the model's
/// `rounding` rows otherwise.
pub(super) fn prepare(
    selection: &Selection,
    rounding: &[PartialRounding],
    op: &CommQuery,
    rank: u32,
    need: Need,
) -> Result<Prepared<AllReduceBuffers<View>>, Refused> {
    let op = RoundedAllReduceOp::new(selection, rounding, *op, rank, need)?;
    need.check(&op.info)?;
    Ok(Box::new(op))
}

impl RoundedAllReduceOp {
    fn new(
        selection: &Selection,
        rounding: &[PartialRounding],
        op: CommQuery,
        rank: u32,
        need: Need,
    ) -> Result<Self, Refused> {
        let capacity = need.capacity;
        let CommQuery::AllReduceRounded {
            ranks,
            elements: width,
        } = op
        else {
            return Err(Refused::Rule(
                "the SM120 rounded all-reduce serves rounded all-reduces".into(),
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
                "the SM120 rounded all-reduce takes at most {MAX_COMM_RANKS} ranks"
            )));
        }
        if capacity > MAX_CAPACITY {
            return Err(Refused::Rule(format!(
                "an SM120 all-reduce serves at most {MAX_CAPACITY} rows"
            )));
        }
        u32::try_from(width).map_err(|_| {
            Refused::Rule("an SM120 rounded all-reduce row exceeds u32 elements".into())
        })?;
        // Every rank's pair inbox holds the whole reduction in F32.
        let elements = u64::from(capacity)
            .checked_mul(width)
            .filter(|elements| elements.checked_mul(4 * u64::from(ranks)).is_some())
            .ok_or_else(|| Refused::Rule("all-reduce sizes overflow u64".into()))?;
        // The phased entries take whole granules, and the one-shot its most
        // elements at every count the table gives it.
        let one_shot_rows = capacity.min(ONE_SHOT_ROWS);
        let phased = selection.phased
            && width.is_multiple_of(phased_f32::GRANULE)
            && u64::from(one_shot_rows) * width <= phased_f32::ONE_SHOT_MAX_ELEMENTS;
        let rounding = sole(
            rounding
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
        let composed_claims = match &rounding {
            Some(pair) => [SCATTER_SYMBOL, GATHER_SYMBOL]
                .map(declared)
                .into_iter()
                .fold(pair.invariance, Invariance::meet),
            None => Invariance::NONE,
        };
        let mut held = Vec::with_capacity(selection.low_latency.len());
        for entry in &selection.low_latency {
            let rows = low_latency_rows(entry, width)?;
            if ranks.is_power_of_two() && rows > 0 {
                held.push(rows);
            }
        }
        let ll_claims = declared(ll_f32::ROUND_SYMBOL);
        let phased_form = |tuned| match (tuned, phased) {
            (_, false) => Form::Composed,
            (Tuned::LowLatency | Tuned::OneShot, true) => Form::Phased {
                symbol: phased_f32::ONE_SHOT_ROUND_SYMBOL,
                senders: SENDERS,
                workers: 0,
            },
            (Tuned::TwoShot { senders, workers }, true) => Form::Phased {
                symbol: phased_f32::ROUND_SYMBOL,
                senders,
                workers,
            },
        };
        let claims = |form: Form| match form {
            Form::LowLatency { .. } => ll_claims,
            Form::Phased { symbol, .. } => declared(symbol),
            Form::Composed => composed_claims,
        };
        let fallback = phased_form(Tuned::OneShot);
        let mut picks = Vec::with_capacity(capacity as usize);
        let mut guarantees: Option<Invariance> = None;
        for rows in 1..=capacity {
            let (form, claimed) = match tuned(rows) {
                Tuned::LowLatency if held.iter().any(|&held| rows <= held) => {
                    let candidates = held.iter().map(|&held| {
                        Candidate::new(
                            Pick::LowLatency { held },
                            ll_f32::ROUND_SYMBOL,
                            low_latency_key(held),
                            ll_claims,
                        )
                        .capacity(held)
                    });
                    let other = Candidate::new(
                        Pick::Fallback,
                        "the one-shot round's form",
                        "fallback".to_owned(),
                        claims(fallback),
                    );
                    let chosen = choose(
                        "rounded all-reduce",
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
                        "{rows} rows of the SM120 rounded all-reduce need the composed \
                         form, which reads the model's partial rounding rows of {width} \
                         elements (Sm120::with_partial_rounding)"
                    )));
                }
                rounding::plan(rows)?;
            }
            guarantees = Some(guarantees.map_or(claimed, |held| held.meet(claimed)));
            picks.push(form);
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
            let bytes = ll_f32::state_bytes(ranks, u64::from(rows) * width)?;
            ll_states.push((rows, state.len()));
            state.extend([bytes.inbox, bytes.epochs]);
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
        let any = |matches: fn(&Form) -> bool| forms.iter().any(matches);
        let phased_state = any(|form| matches!(form, Form::Phased { .. })).then_some(state.len());
        if phased_state.is_some() {
            let bytes = phased_f32::buffers(ranks, elements)?;
            state.extend([bytes.inbox_bytes, bytes.flag_bytes, bytes.epoch_bytes]);
        }
        let composed = any(|form| *form == Form::Composed);
        let pair = composed.then_some(state.len());
        let sum = composed.then_some(state.len() + 3);
        if composed {
            let bytes = comm::phased_buffers(ranks, elements, DType::F32)?;
            state.extend([bytes.inbox_bytes, bytes.flag_bytes, bytes.epoch_bytes]);
            state.push(elements * 4);
        }
        // Every form's plan at every row count, so launches cannot refuse a
        // plan SM120 cannot take.
        for (rows, &form) in (1..=capacity).zip(&forms) {
            for plan in plans(ranks, width, rows, form)? {
                check_plan(&plan)?;
            }
        }
        let mut symbols = Vec::new();
        if !served.is_empty() {
            symbols.push((ll_f32::ROUND_SYMBOL.to_owned(), 0));
        }
        for symbol in [phased_f32::ONE_SHOT_ROUND_SYMBOL, phased_f32::ROUND_SYMBOL] {
            if forms
                .iter()
                .any(|form| matches!(form, Form::Phased { symbol: own, .. } if *own == symbol))
            {
                symbols.push((symbol.to_owned(), 0));
            }
        }
        if composed {
            let pair = rounding.as_ref().expect("checked above");
            symbols.push((pair.round.symbol.clone(), 0));
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
        let mut info = OpInfo::new(
            format!("sm120.comm.allreduce.rounded_f32.v1[{low_latency}{transport}]"),
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
            sum,
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
}

/// Every plan `form` launches at `rows` rows.
fn plans(ranks: u32, width: u64, rows: u32, form: Form) -> Result<Vec<LaunchPlan>, Refusal> {
    Ok(match form {
        Form::LowLatency { .. } => vec![ll_f32::plan(ranks, u64::from(rows) * width)?],
        Form::Phased {
            senders, workers, ..
        } => vec![phased_f32::plan_with(senders, workers)?],
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

impl PreparedOp<AllReduceBuffers<View>> for RoundedAllReduceOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn defer_bf16_round(
        &self,
        buffers: &AllReduceBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Option<kernel_core::backend::DeferredBf16Round>, Refusal> {
        // Validate with the ordinary path before exposing any state view.
        let mut launches = self.launches(buffers, workspace, counts)?;
        if self.forms[counts.rows as usize - 1] != Form::Composed {
            return Ok(None);
        }
        let sum = self.regions(buffers, self.sum.expect("composed sum"), 1)?;
        let round = launches.pop().expect("composed round");
        debug_assert_eq!(round.symbol, self.rounding.as_ref().unwrap().round.symbol);
        Ok(Some(kernel_core::backend::DeferredBf16Round {
            launches,
            input: sum[self.rank as usize][0],
            output: buffers.output,
            live_rows: buffers.live_rows.expect("validated live rows"),
            width: self.width as u32,
        }))
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
                let arguments = ll_f32::arguments(&ll_f32::Call {
                    input: buffers.input.handle,
                    output: buffers.output.handle,
                    peer_inboxes: regions.iter().map(|state| state[0].handle).collect(),
                    epochs: regions[self.rank as usize][1].handle,
                    elements,
                    capacity_elements: u64::from(held) * self.width,
                    rank: self.rank,
                    ranks: self.ranks,
                })?;
                let plan = ll_f32::plan(self.ranks, elements)?;
                Ok(vec![collective(ll_f32::ROUND_SYMBOL, plan, arguments)])
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
                let call = phased_f32::Call {
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
                let plan = phased_f32::plan_with(senders, workers)?;
                Ok(vec![collective(
                    symbol,
                    plan,
                    phased_f32::arguments(symbol, call)?,
                )])
            }
            Form::Composed => {
                let pair = self
                    .rounding
                    .as_ref()
                    .expect("a composed form has its rounding rows");
                let live = buffers.live_rows.ok_or(Refusal(
                    "the composed rounded all-reduce reads a device live count",
                ))?;
                let state = self.pair.expect("a composed form has its pair state");
                let regions = self.regions(buffers, state, 3)?;
                let sums = self.regions(buffers, self.sum.expect("and its sum"), 1)?;
                for sum in &sums {
                    sum[0].covers(elements * 4)?;
                }
                let own = sums[self.rank as usize][0];
                let plan = pair_plan(self.ranks, self.width, rows)?;
                let scatter = comm::arguments(OneShotCall {
                    input: buffers.input.handle,
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
                let round_plan = rounding::plan(rows)?;
                Ok(vec![
                    collective(SCATTER_SYMBOL, plan, scatter),
                    collective(GATHER_SYMBOL, plan, gather),
                    Launch {
                        symbol: pair.round.symbol.clone(),
                        // A row stage records the bucket it processes.
                        rows: round_plan.grid[0],
                        plan: round_plan,
                        arguments: rounding::arguments(
                            own.handle,
                            buffers.output.handle,
                            live.handle,
                        )?,
                        collective: false,
                        persistent: false,
                    },
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
    const CAPACITY: u32 = 64;

    fn reduce(ranks: u32, width: u64) -> CommQuery {
        CommQuery::AllReduceRounded {
            ranks,
            elements: width,
        }
    }

    fn backend(phased: bool, rounding: bool) -> Sm120 {
        let mut backend = Sm120::new();
        if phased {
            backend = backend.with_phased_f32_all_reduce();
        }
        if rounding {
            backend = backend.with_partial_rounding(rounding::shared(WIDTH as u32));
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

    #[test]
    fn deferred_round_keeps_collective_launches_state_and_refusals() {
        for ranks in [1, 2, 4] {
            for rank in 0..ranks {
                let op = backend(false, true)
                    .all_reduce(&reduce(ranks, WIDTH), rank, Need::rows(CAPACITY))
                    .unwrap();
                let b = buffers(op.info(), ranks);
                for rows in [1, 8, 32] {
                    let counts = Counts::rows(rows);
                    let old = op.launches(&b, View::EMPTY, counts).unwrap();
                    let split = op
                        .defer_bf16_round(&b, View::EMPTY, counts)
                        .unwrap()
                        .unwrap();
                    assert_eq!(split.launches.len() + 1, old.len());
                    for (new, old) in split.launches.iter().zip(&old) {
                        assert_eq!(new.symbol, old.symbol);
                        assert_eq!(new.plan, old.plan);
                        assert_eq!(new.arguments, old.arguments);
                        assert_eq!(new.collective, old.collective);
                    }
                    assert_eq!(split.input.handle, handle(rank, 3));
                    assert_eq!(split.output, b.output);
                    assert_eq!(split.live_rows, b.live_rows.unwrap());
                    assert_eq!(split.width, WIDTH as u32);
                    assert_eq!(
                        old.last().unwrap().arguments,
                        rounding::arguments(
                            split.input.handle,
                            split.output.handle,
                            split.live_rows.handle
                        )
                        .unwrap()
                    );
                }
                for bad in [0, CAPACITY + 1] {
                    assert!(
                        op.defer_bf16_round(&b, View::EMPTY, Counts::rows(bad))
                            .is_err()
                    );
                }
                let mut bad = b;
                bad.ranks[rank as usize].state[3].bytes = 1;
                assert!(
                    op.defer_bf16_round(&bad, View::EMPTY, Counts::rows(1))
                        .is_err()
                );
            }
        }
        let op = backend(true, false)
            .all_reduce(&reduce(4, WIDTH), 0, Need::rows(CAPACITY))
            .unwrap();
        assert!(
            op.defer_bf16_round(&buffers(op.info(), 4), View::EMPTY, Counts::rows(1))
                .unwrap()
                .is_none()
        );
    }

    /// The phased forms take the one-shot round entry to four rows, the
    /// round two-shot of 16 senders to 32 and with 32 workers past them,
    /// over one phased F32 state of the capacity, writing every rank's
    /// output.
    #[test]
    fn the_phased_forms_follow_the_table_over_one_state() {
        let (ranks, rank) = (4, 2);
        let op = backend(true, false)
            .all_reduce(&reduce(ranks, WIDTH), rank, Need::rows(CAPACITY))
            .unwrap();
        let state = phased_f32::buffers(ranks, u64::from(CAPACITY) * WIDTH).unwrap();
        assert_eq!(
            op.info().state,
            [state.inbox_bytes, state.flag_bytes, state.epoch_bytes]
        );
        assert_eq!(
            op.info()
                .symbols
                .iter()
                .map(|(symbol, _)| symbol.as_str())
                .collect::<Vec<_>>(),
            [phased_f32::ONE_SHOT_ROUND_SYMBOL, phased_f32::ROUND_SYMBOL]
        );
        let buffers = buffers(op.info(), ranks);
        for rows in 1..=CAPACITY {
            let launches = op
                .launches(&buffers, view(0, 0), Counts::rows(rows))
                .unwrap();
            let (symbol, workers) = match rows {
                ..=4 => (phased_f32::ONE_SHOT_ROUND_SYMBOL, 0),
                5..=32 => (phased_f32::ROUND_SYMBOL, 0),
                _ => (phased_f32::ROUND_SYMBOL, 32),
            };
            let call = phased_f32::Call {
                input: 1,
                output: 2,
                inboxes: (0..ranks).map(|r| handle(r, 0)).collect(),
                outputs: (0..ranks).map(|r| 10 + u64::from(r)).collect(),
                flags: (0..ranks).map(|r| handle(r, 1)).collect(),
                epoch: handle(rank, 2),
                elements: u64::from(rows) * WIDTH,
                row_elements: WIDTH as u32,
                live_rows: Some(3),
                rank,
                ranks,
                senders: 16,
            };
            assert_eq!(
                launches,
                [Launch {
                    symbol: symbol.into(),
                    rows,
                    plan: phased_f32::plan_with(16, workers).unwrap(),
                    arguments: phased_f32::arguments(symbol, call).unwrap(),
                    collective: true,
                    persistent: false,
                }],
                "{rows}"
            );
        }
        assert!(
            op.launches(&buffers, view(0, 0), Counts::rows(CAPACITY + 1))
                .is_err()
        );
        assert_eq!(op.info().guarantees, Invariance::any_rows(8192));
    }

    /// One row takes the low-latency round entry through a registration
    /// that holds it, over a state of its own laid out before the phased
    /// state; two rows and up keep the phased forms. Rank counts that are
    /// no power of two and grids past the entry's CTAs take no
    /// low-latency entry.
    #[test]
    fn one_row_takes_the_low_latency_entry_where_registered() {
        let (ranks, rank) = (4, 3);
        let ctas = (u64::from(LOW_LATENCY_ROWS) * WIDTH / ll_f32::SLICE_ELEMENTS) as u32;
        let registered =
            |phased, rounding| backend(phased, rounding).with_low_latency_rounded_all_reduce(ctas);
        let op = registered(true, false)
            .all_reduce(&reduce(ranks, WIDTH), rank, Need::rows(CAPACITY))
            .unwrap();
        let ll = ll_f32::state_bytes(ranks, WIDTH).unwrap();
        let phased = phased_f32::buffers(ranks, u64::from(CAPACITY) * WIDTH).unwrap();
        assert_eq!(
            op.info().state,
            [
                ll.inbox,
                ll.epochs,
                phased.inbox_bytes,
                phased.flag_bytes,
                phased.epoch_bytes
            ]
        );
        assert_eq!(
            op.info()
                .symbols
                .iter()
                .map(|(symbol, _)| symbol.as_str())
                .collect::<Vec<_>>(),
            [
                ll_f32::ROUND_SYMBOL,
                phased_f32::ONE_SHOT_ROUND_SYMBOL,
                phased_f32::ROUND_SYMBOL
            ]
        );
        assert!(
            op.info()
                .implementation
                .ends_with("[low_latency_rows=1,phased]")
        );
        let buffers = buffers(op.info(), ranks);
        let launches = op.launches(&buffers, view(0, 0), Counts::rows(1)).unwrap();
        let call = ll_f32::Call {
            input: 1,
            output: 2,
            peer_inboxes: (0..ranks).map(|r| handle(r, 0)).collect(),
            epochs: handle(rank, 1),
            elements: WIDTH,
            capacity_elements: WIDTH,
            rank,
            ranks,
        };
        assert_eq!(
            launches,
            [Launch {
                symbol: ll_f32::ROUND_SYMBOL.into(),
                rows: 1,
                plan: ll_f32::plan(ranks, WIDTH).unwrap(),
                arguments: ll_f32::arguments(&call).unwrap(),
                collective: true,
                persistent: false,
            }]
        );
        let two = op.launches(&buffers, view(0, 0), Counts::rows(2)).unwrap();
        assert_eq!(two[0].symbol, phased_f32::ONE_SHOT_ROUND_SYMBOL);
        // The phased state follows the low-latency one.
        assert_eq!(
            two[0].arguments,
            phased_f32::arguments(
                phased_f32::ONE_SHOT_ROUND_SYMBOL,
                phased_f32::Call {
                    input: 1,
                    output: 2,
                    inboxes: (0..ranks).map(|r| handle(r, 2)).collect(),
                    outputs: (0..ranks).map(|r| 10 + u64::from(r)).collect(),
                    flags: (0..ranks).map(|r| handle(r, 3)).collect(),
                    epoch: handle(rank, 4),
                    elements: 2 * WIDTH,
                    row_elements: WIDTH as u32,
                    live_rows: Some(3),
                    rank,
                    ranks,
                    senders: 16,
                }
            )
            .unwrap()
        );
        assert_eq!(op.info().guarantees, Invariance::any_rows(8192));
        // Without the phased entries one row still takes it, the rest the
        // composed form.
        let composed = registered(false, true)
            .all_reduce(&reduce(ranks, WIDTH), rank, Need::rows(CAPACITY))
            .unwrap();
        assert!(
            composed
                .info()
                .implementation
                .ends_with("[low_latency_rows=1,composed]")
        );
        let three = registered(false, true)
            .all_reduce(&reduce(3, WIDTH), 0, Need::rows(4))
            .unwrap();
        assert!(!three.info().implementation.contains("low_latency"));
        assert!(
            backend(true, false)
                .with_low_latency_rounded_all_reduce(ll_f32::MAX_CTAS + 1)
                .all_reduce(&reduce(ranks, WIDTH), rank, Need::rows(4))
                .is_err()
        );
    }

    /// Without the phased entries the op sums through the F32 pair into
    /// every rank's F32 sum and rounds this rank's sum with the model's
    /// round row; without either it refuses.
    #[test]
    fn the_composed_form_rounds_the_pair_sum_or_the_op_refuses() {
        let (ranks, rank) = (4, 1);
        let op = backend(false, true)
            .all_reduce(&reduce(ranks, WIDTH), rank, Need::rows(CAPACITY))
            .unwrap();
        let elements = u64::from(CAPACITY) * WIDTH;
        let pair = comm::phased_buffers(ranks, elements, DType::F32).unwrap();
        assert_eq!(
            op.info().state,
            [
                pair.inbox_bytes,
                pair.flag_bytes,
                pair.epoch_bytes,
                elements * 4
            ]
        );
        let buffers = buffers(op.info(), ranks);
        let launches = op.launches(&buffers, view(0, 0), Counts::rows(5)).unwrap();
        let symbols = launches
            .iter()
            .map(|launch| launch.symbol.as_str())
            .collect::<Vec<_>>();
        let round = rounding::shared(WIDTH as u32).round.symbol;
        assert_eq!(symbols, [SCATTER_SYMBOL, GATHER_SYMBOL, round.as_str()]);
        // The round row reads this rank's sum and writes the output.
        assert_eq!(
            launches[2].arguments,
            rounding::arguments(handle(rank, 3), 2, 3).unwrap()
        );
        let no_live = AllReduceBuffers {
            live_rows: None,
            ..buffers.clone()
        };
        assert!(op.launches(&no_live, view(0, 0), Counts::rows(5)).is_err());
        let refused = backend(false, false)
            .all_reduce(&reduce(ranks, WIDTH), rank, Need::rows(CAPACITY))
            .err()
            .unwrap();
        assert!(format!("{refused:?}").contains("with_partial_rounding"));
    }

    /// Rows of 512 elements are no whole granule, so they take the
    /// composed form: refused without rounding rows of their width, served
    /// with them.
    #[test]
    fn rows_of_partial_granules_take_the_composed_form() {
        // The rounding rows are registered at WIDTH only.
        assert!(
            backend(true, true)
                .all_reduce(&reduce(2, 512), 0, Need::rows(8))
                .is_err()
        );
        let mut backend = backend(true, false);
        backend = backend.with_partial_rounding(rounding::shared(512));
        let op = backend
            .all_reduce(&reduce(2, 512), 0, Need::rows(8))
            .unwrap();
        assert!(op.info().implementation.contains("composed"));
    }

    /// Other all-reduces are not this op's.
    #[test]
    fn other_queries_are_refused() {
        let op = CommQuery::AllReduce {
            ranks: 4,
            elements: WIDTH,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        assert!(
            RoundedAllReduceOp::new(
                &Selection {
                    phased: true,
                    ..Selection::default()
                },
                &[],
                op,
                0,
                Need::rows(4)
            )
            .is_err()
        );
        assert!(
            backend(true, false)
                .all_reduce(&reduce(4, WIDTH), 4, Need::rows(4))
                .is_err()
        );
    }
}
