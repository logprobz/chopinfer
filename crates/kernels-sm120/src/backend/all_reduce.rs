//! The F32 all-reduce through the kernel base's comm entries and, where a
//! model registers it, the low-latency entry (`comm::ll_f32`).
//!
//! Every entry sums each element from F32 zero in rank order, the same on
//! every rank. Each row count's entry is chosen when the op is prepared,
//! from the op's own fields (`select::choose`): a low-latency registration
//! that holds the row count, the least of them, else the transport (one
//! shot up to the tuned bytes, then the phased reduce-scatter and
//! all-gather). An entry whose declarations do not keep `Need.exact` is
//! passed over, and the op claims what every launched entry declares
//! together (their meet), so never more than the catalog says. The entries
//! read no device live count: every row of a launch holds its sum, which a
//! live count in the buffers allows.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::collective::AllReduceBuffers;
use kernel_core::contract::{
    Argument, ArgumentList, CommQuery, DType, LaunchPlan, Refusal, Validate,
};
use kernel_core::properties::Invariance;

use super::select::{Candidate, choose};
use super::{MAX_CAPACITY, check_plan};
use crate::comm::{self, OneShotCall, Variant, ll_f32};

/// A model's registration of the low-latency F32 all-reduce
/// (`comm::ll_f32`), whose CTAs spin on their counterparts on the peer
/// ranks: the most CTAs of one launch, which the model's create admitted
/// against every rank's residency (the grid of its largest call). It serves
/// an op's row counts whose reduction fits that grid in whole 1024-element
/// slices, at power-of-two rank counts; a model whose devices cannot keep
/// the grid resident does not register it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LowLatency {
    pub(super) ctas: u32,
}

/// The all-reduce `op` that rank `rank` launches, through the model's
/// `low_latency` registrations where they serve.
pub(super) fn prepare(
    low_latency: &[LowLatency],
    op: &CommQuery,
    rank: u32,
    need: Need,
) -> Result<Prepared<AllReduceBuffers<View>>, Refused> {
    let op = AllReduceOp::new(low_latency, *op, rank, need)?;
    need.check(&op.info)?;
    Ok(Box::new(op))
}

/// An F32 all-reduce over its row counts' entries. Each one-shot row count
/// keeps its own state, since the kernel's flag offsets and epoch cells
/// follow the tile count; the phased pair's state holds any count up to the
/// capacity. Every row count a low-latency registration serves shares one
/// state, whose capacity (the slot layout and the CTAs' epoch cells) is the
/// same at every call, so each CTA's epoch advances over all of them alike.
struct AllReduceOp {
    ranks: u32,
    /// Elements of one row.
    width: u64,
    rank: u32,
    /// The form and plan of each row count from one to the capacity.
    forms: Vec<(Form, LaunchPlan)>,
    info: OpInfo,
}

/// One row count's launches and the first of its state regions.
#[derive(Clone, Copy, Debug)]
enum Form {
    /// The low-latency entry over its state's inbox and epochs, which hold
    /// `capacity` elements.
    LowLatency { state: usize, capacity: u64 },
    /// The one-shot over its row count's inbox, flags and epoch.
    OneShot { state: usize },
    /// The reduce-scatter then all-gather over the pair's inbox, flags and
    /// epoch.
    Phased { state: usize },
}

/// The entry a row count takes.
#[derive(Clone, Copy, Debug)]
enum Pick {
    /// A low-latency registration holding `rows` rows of the op.
    LowLatency {
        rows: u32,
    },
    Transport,
}

const ONE_SHOT_SYMBOL: &str = "chop_comm_allreduce_oneshot_f32_v1";
const SCATTER_SYMBOL: &str = "chop_comm_reduce_scatter_f32_v1";
const GATHER_SYMBOL: &str = "chop_comm_allgather_scattered_f32_v1";
/// State regions of each one-shot or phased state, and of a low-latency
/// state.
const TRANSPORT_REGIONS: usize = 3;
const LOW_LATENCY_REGIONS: usize = 2;

/// The rows of an op of `width`-element rows that registration `entry`
/// holds (its launch key): its state's capacity at every call, whatever
/// the op's capacity, and the most rows it serves. Zero when one row
/// exceeds its grid or the row is no whole number of slices. Refuses a
/// grid past the entry's CTAs.
fn low_latency_rows(entry: &LowLatency, width: u64) -> Result<u32, Refused> {
    let LowLatency {
        ctas, // the admitted grid: the rows it holds
    } = *entry;
    if ctas == 0 || ctas > ll_f32::MAX_CTAS {
        return Err(Refused::Rule(format!(
            "the SM120 low-latency all-reduce launches 1 to {} CTAs",
            ll_f32::MAX_CTAS
        )));
    }
    if !width.is_multiple_of(ll_f32::SLICE_ELEMENTS) {
        return Ok(0);
    }
    Ok((u64::from(ctas) * ll_f32::SLICE_ELEMENTS / width) as u32)
}

/// What a registration holding `rows` rows of an op launches: the entry
/// over a state of that many rows, which its capacity argument names.
fn low_latency_key(rows: u32) -> String {
    format!("{}[rows={rows}]", ll_f32::SYMBOL)
}

/// What `symbol`'s catalog declaration claims every launch keeps.
fn declared(symbol: &str) -> Invariance {
    comm::declared_entries()
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .map_or(Invariance::NONE, |entry| entry.invariance)
}

impl AllReduceOp {
    fn new(
        low_latency: &[LowLatency],
        op: CommQuery,
        rank: u32,
        need: Need,
    ) -> Result<Self, Refused> {
        let capacity = need.capacity;
        let CommQuery::AllReduce {
            ranks,
            elements: width,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        } = op
        else {
            return Err(Refused::Rule(
                "the SM120 all-reduce sums plain F32 rows".into(),
            ));
        };
        if rank >= ranks || capacity == 0 || width == 0 {
            return Err(Refused::Rule(
                "an all-reduce needs a rank below its ranks, rows and a width".into(),
            ));
        }
        if capacity > MAX_CAPACITY {
            return Err(Refused::Rule(format!(
                "an SM120 all-reduce serves at most {MAX_CAPACITY} rows"
            )));
        }
        // Every rank's inbox holds the whole reduction in F32.
        u64::from(capacity)
            .checked_mul(width)
            .and_then(|elements| elements.checked_mul(4 * u64::from(ranks)))
            .ok_or_else(|| Refused::Rule("all-reduce sizes overflow u64".into()))?;
        let query = |rows: u32| CommQuery::AllReduce {
            ranks,
            elements: u64::from(rows) * width,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        // The largest reduction first, so no table is built for one SM120
        // cannot plan.
        check_plan(&comm::plan(&query(capacity), 1)?)?;
        // The rows each registration holds; none at a rank count the entry
        // does not take.
        let mut held = Vec::with_capacity(low_latency.len());
        for entry in low_latency {
            let rows = low_latency_rows(entry, width)?;
            if ranks.is_power_of_two() && rows > 0 {
                held.push(rows);
            }
        }
        let most_held = held.iter().copied().max().unwrap_or(0);
        let ll_claims = declared(ll_f32::SYMBOL);
        let mut one_shot_rows = 0;
        while one_shot_rows < capacity
            && comm::choose(&query(one_shot_rows + 1))? == Variant::OneShot
        {
            one_shot_rows += 1;
        }
        let one_shot_claims = declared(ONE_SHOT_SYMBOL);
        let phased_claims = declared(SCATTER_SYMBOL).meet(declared(GATHER_SYMBOL));
        let transport_claims = |rows: u32| {
            if rows <= one_shot_rows {
                one_shot_claims
            } else {
                phased_claims
            }
        };
        let mut picks = Vec::with_capacity(capacity as usize);
        let mut guarantees: Option<Invariance> = None;
        for rows in 1..=capacity {
            let (pick, claims) = if rows <= most_held {
                let transport = Candidate::new(
                    Pick::Transport,
                    "the transport",
                    "transport".to_owned(),
                    transport_claims(rows),
                );
                let candidates = held.iter().map(|&held| {
                    Candidate::new(
                        Pick::LowLatency { rows: held },
                        ll_f32::SYMBOL,
                        low_latency_key(held),
                        ll_claims,
                    )
                    .capacity(held)
                });
                let chosen = choose(
                    "all-reduce",
                    candidates.chain([transport]),
                    Need {
                        capacity: rows,
                        ..need
                    },
                )?;
                (chosen.entry, chosen.guarantees)
            } else {
                (Pick::Transport, transport_claims(rows))
            };
            guarantees = Some(guarantees.map_or(claims, |held| held.meet(claims)));
            picks.push(pick);
        }
        // The state regions in role order: each one-shot row count's and
        // the phased pair's as the transport lays them out without the
        // entry (the rows the entry serves keep theirs, so a host's layout
        // does not follow the registration), then each low-latency state's,
        // least rows first.
        let mut state = Vec::new();
        let ranks64 = u64::from(ranks);
        for rows in 1..=one_shot_rows {
            let elements = u64::from(rows) * width;
            let tiles = elements.div_ceil(4096);
            // Per rank an inbox of the whole reduction, a data flag and an
            // acknowledgment per rank and tile, and an epoch cell per tile.
            state.extend([ranks64 * elements * 4, 2 * ranks64 * tiles * 4, tiles * 4]);
        }
        let phased_state = state.len();
        if one_shot_rows < capacity {
            let phased = comm::phased_buffers(ranks, u64::from(capacity) * width, DType::F32)?;
            state.extend([phased.inbox_bytes, phased.flag_bytes, phased.epoch_bytes]);
        }
        let mut served: Vec<u32> = picks
            .iter()
            .filter_map(|pick| match pick {
                Pick::LowLatency { rows } => Some(*rows),
                Pick::Transport => None,
            })
            .collect();
        served.sort_unstable();
        served.dedup();
        let mut ll_states = Vec::with_capacity(served.len());
        for &rows in &served {
            let elements = u64::from(rows) * width;
            let bytes = ll_f32::state_bytes(ranks, elements)?;
            ll_states.push((
                rows,
                Form::LowLatency {
                    state: state.len(),
                    capacity: elements,
                },
            ));
            state.extend([bytes.inbox, bytes.epochs]);
        }
        let forms = (1..=capacity)
            .zip(&picks)
            .map(|(rows, pick)| {
                let form = match pick {
                    Pick::LowLatency { rows: held } => {
                        let &(_, form) = ll_states
                            .iter()
                            .find(|(own, _)| own == held)
                            .expect("every served state is laid out");
                        form
                    }
                    Pick::Transport if rows <= one_shot_rows => Form::OneShot {
                        state: (rows as usize - 1) * TRANSPORT_REGIONS,
                    },
                    Pick::Transport => Form::Phased {
                        state: phased_state,
                    },
                };
                let plan = match form {
                    Form::LowLatency { .. } => ll_f32::plan(ranks, u64::from(rows) * width)?,
                    _ => comm::plan(&query(rows), 1)?,
                };
                check_plan(&plan)?;
                Ok((form, plan))
            })
            .collect::<Result<Vec<_>, Refused>>()?;
        let mut symbols = Vec::new();
        if !served.is_empty() {
            symbols.push((ll_f32::SYMBOL.to_owned(), 0));
        }
        if one_shot_rows > 0
            && picks[..one_shot_rows as usize]
                .iter()
                .any(|pick| matches!(pick, Pick::Transport))
        {
            symbols.push((ONE_SHOT_SYMBOL.to_owned(), 0));
        }
        if forms
            .iter()
            .any(|(form, _)| matches!(form, Form::Phased { .. }))
        {
            let shared = comm::plan(&query(capacity), 1)?.dynamic_shared_bytes;
            symbols.push((SCATTER_SYMBOL.to_owned(), shared));
            symbols.push((GATHER_SYMBOL.to_owned(), shared));
        }
        // The low-latency states' rows, when any serves, then the most rows
        // that would reduce in one shot.
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
        let mut info = OpInfo::new(
            format!("sm120.comm.allreduce.f32.v1[{low_latency}one_shot_rows={one_shot_rows}]"),
            guarantees.unwrap_or(Invariance::NONE),
            symbols,
        );
        info.state = state;
        Ok(Self {
            ranks,
            width,
            rank,
            forms,
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

impl PreparedOp<AllReduceBuffers<View>> for AllReduceOp {
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
        let &(form, plan) = rows
            .checked_sub(1)
            .and_then(|index| self.forms.get(index as usize))
            .ok_or(Refusal("all-reduce rows exceed the prepared capacity"))?;
        if buffers.ranks.len() != self.ranks as usize {
            return Err(Refusal("an all-reduce takes the views of every rank"));
        }
        let elements = u64::from(rows) * self.width;
        buffers.input.covers(elements * 4)?;
        buffers.output.covers(elements * 4)?;
        let launch = |symbol: &str, arguments| Launch {
            symbol: symbol.into(),
            rows,
            plan,
            arguments,
            collective: true,
            persistent: false,
        };
        let state = match form {
            Form::LowLatency { state, capacity } => {
                // Every rank's inbox, this rank's epochs.
                let regions = self.regions(buffers, state, LOW_LATENCY_REGIONS)?;
                let arguments = ll_f32::arguments(&ll_f32::Call {
                    input: buffers.input.handle,
                    output: buffers.output.handle,
                    peer_inboxes: regions.iter().map(|state| state[0].handle).collect(),
                    epochs: regions[self.rank as usize][1].handle,
                    elements,
                    capacity_elements: capacity,
                    rank: self.rank,
                    ranks: self.ranks,
                })?;
                return Ok(vec![launch(ll_f32::SYMBOL, arguments)]);
            }
            Form::OneShot { state } | Form::Phased { state } => state,
        };
        let regions = self.regions(buffers, state, TRANSPORT_REGIONS)?;
        for rank in &buffers.ranks {
            rank.output.covers(elements * 4)?;
        }
        let epoch = regions[self.rank as usize][2];
        let arguments = comm::arguments(OneShotCall {
            input: buffers.input.handle,
            output: buffers.output.handle,
            peer_inboxes: regions.iter().map(|state| state[0].handle).collect(),
            peer_flags: regions.iter().map(|state| state[1].handle).collect(),
            epoch: epoch.handle,
            elements,
            rank: self.rank,
            ranks: self.ranks,
        })?;
        if let Form::OneShot { .. } = form {
            return Ok(vec![launch(ONE_SHOT_SYMBOL, arguments)]);
        }
        // Each rank's reduced tiles into every rank's output.
        let gather = ArgumentList(vec![
            Argument::DeviceBuffer(buffers.output.handle),
            Argument::DevicePointerArray(buffers.ranks.iter().map(|r| r.output.handle).collect()),
            Argument::DevicePointerArray(regions.iter().map(|state| state[1].handle).collect()),
            Argument::DeviceBuffer(epoch.handle),
            Argument::U64(elements),
            Argument::U32(self.rank),
            Argument::U32(self.ranks),
        ]);
        gather.validate()?;
        Ok(vec![
            launch(SCATTER_SYMBOL, arguments),
            launch(GATHER_SYMBOL, gather),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};
    use kernel_core::backend::Backend;
    use kernel_core::collective::RankViews;

    #[test]
    fn all_reduce_takes_each_count_its_own_state_and_marks_collectives() {
        let (ranks, width, capacity) = (2_u32, 4096_u64, 512_u32);
        let query = CommQuery::AllReduce {
            ranks,
            elements: width,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        let prepared = Sm120::new()
            .all_reduce(&query, 1, Need::rows(capacity))
            .unwrap();
        let info = prepared.info().clone();
        let one_shot_rows = (info.state.len() / 3 - 1) as u32;
        assert!(one_shot_rows > 0 && one_shot_rows < capacity);
        // Handles: rank r's output 10 + r, its state regions 100 * (r + 1) + i.
        let buffers = AllReduceBuffers {
            input: view(1, u64::MAX),
            output: view(11, u64::MAX),
            live_rows: None,
            ranks: (0..ranks)
                .map(|r| RankViews {
                    output: view(10 + u64::from(r), u64::MAX),
                    state: info
                        .state
                        .iter()
                        .enumerate()
                        .map(|(i, &bytes)| view(100 * (u64::from(r) + 1) + i as u64, bytes))
                        .collect(),
                })
                .collect(),
        };
        for rows in 1..=capacity {
            let launches = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            let query = CommQuery::AllReduce {
                ranks,
                elements: u64::from(rows) * width,
                dtype: DType::F32,
                fused_residual: false,
                fused_rmsnorm: None,
            };
            assert!(
                launches
                    .iter()
                    .all(|l| l.plan == comm::plan(&query, 1).unwrap())
            );
        }
        for rows in [1, one_shot_rows] {
            let launches = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            let first = u64::from(rows - 1) * 3;
            let elements = u64::from(rows) * width;
            let expected = comm::arguments(OneShotCall {
                input: 1,
                output: 11,
                peer_inboxes: vec![100 + first, 200 + first],
                peer_flags: vec![101 + first, 201 + first],
                epoch: 202 + first,
                elements,
                rank: 1,
                ranks,
            })
            .unwrap();
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].symbol, ONE_SHOT_SYMBOL);
            assert_eq!(launches[0].arguments, expected);
            assert!(launches[0].collective);
        }
        let launches = prepared
            .launches(&buffers, View::EMPTY, Counts::rows(capacity))
            .unwrap();
        let symbols: Vec<_> = launches.iter().map(|l| l.symbol.as_str()).collect();
        assert_eq!(symbols, [SCATTER_SYMBOL, GATHER_SYMBOL]);
        assert!(launches.iter().all(|l| l.collective && l.rows == capacity));
        let phased = u64::from(one_shot_rows) * 3;
        assert_eq!(
            launches[1].arguments.0[1],
            Argument::DevicePointerArray(vec![10, 11])
        );
        assert_eq!(
            launches[1].arguments.0[2],
            Argument::DevicePointerArray(vec![101 + phased, 201 + phased])
        );
        // Every rank's views, each region of its declared size.
        let missing = AllReduceBuffers {
            ranks: buffers.ranks[..1].to_vec(),
            ..buffers.clone()
        };
        assert!(
            prepared
                .launches(&missing, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        let mut short = buffers.clone();
        short.ranks[0].state[0].bytes -= 1;
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(capacity + 1))
                .is_err()
        );
    }

    const WIDTH: u64 = 4096;

    fn reduce(ranks: u32, width: u64) -> CommQuery {
        CommQuery::AllReduce {
            ranks,
            elements: width,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        }
    }

    /// Rank `r`'s output is handle 10 + r and its state regions
    /// 1000 * (r + 1) + i; the input is 1 and the output 2.
    fn buffers(info: &OpInfo, ranks: u32) -> AllReduceBuffers<View> {
        AllReduceBuffers {
            input: view(1, u64::MAX),
            output: view(2, u64::MAX),
            live_rows: None,
            ranks: (0..ranks)
                .map(|r| RankViews {
                    output: view(10 + u64::from(r), u64::MAX),
                    state: info
                        .state
                        .iter()
                        .enumerate()
                        .map(|(i, &bytes)| view(1000 * (u64::from(r) + 1) + i as u64, bytes))
                        .collect(),
                })
                .collect(),
        }
    }

    fn handle(rank: u32, region: usize) -> u64 {
        1000 * (u64::from(rank) + 1) + region as u64
    }

    /// What a model's lowering launches for `rows` rows of `WIDTH` on rank
    /// `rank` of `ranks` when the low-latency entry serves through
    /// `ll_rows` rows over one state (its inbox and epochs at regions
    /// `ll`), with a one-shot state per one-shot row count (from region
    /// zero) and the phased pair's after them: the low-latency entry at its
    /// state's capacity, else the transport.
    fn lowering(ranks: u32, rank: u32, rows: u32, ll_rows: u32, ll: usize) -> Vec<Launch> {
        let elements = u64::from(rows) * WIDTH;
        let one_shot_rows = (1..)
            .take_while(|&rows| {
                comm::choose(&reduce(ranks, u64::from(rows) * WIDTH)).unwrap() == Variant::OneShot
            })
            .last()
            .unwrap_or(0);
        let launch = |symbol: &str, plan, arguments| Launch {
            symbol: symbol.into(),
            rows,
            plan,
            arguments,
            collective: true,
            persistent: false,
        };
        if rows <= ll_rows {
            let call = ll_f32::Call {
                input: 1,
                output: 2,
                peer_inboxes: (0..ranks).map(|r| handle(r, ll)).collect(),
                epochs: handle(rank, ll + 1),
                elements,
                capacity_elements: u64::from(ll_rows) * WIDTH,
                rank,
                ranks,
            };
            let plan = ll_f32::plan(ranks, elements).unwrap();
            return vec![launch(
                ll_f32::SYMBOL,
                plan,
                ll_f32::arguments(&call).unwrap(),
            )];
        }
        let one_shot = rows <= one_shot_rows;
        let first = if one_shot { rows - 1 } else { one_shot_rows } as usize * 3;
        let plan = comm::plan(&reduce(ranks, elements), 1).unwrap();
        let arguments = comm::arguments(OneShotCall {
            input: 1,
            output: 2,
            peer_inboxes: (0..ranks).map(|r| handle(r, first)).collect(),
            peer_flags: (0..ranks).map(|r| handle(r, first + 1)).collect(),
            epoch: handle(rank, first + 2),
            elements,
            rank,
            ranks,
        })
        .unwrap();
        if one_shot {
            return vec![launch(ONE_SHOT_SYMBOL, plan, arguments)];
        }
        let gather = ArgumentList(vec![
            Argument::DeviceBuffer(2),
            Argument::DevicePointerArray((0..ranks).map(|r| 10 + u64::from(r)).collect()),
            Argument::DevicePointerArray((0..ranks).map(|r| handle(r, first + 1)).collect()),
            Argument::DeviceBuffer(handle(rank, first + 2)),
            Argument::U64(elements),
            Argument::U32(rank),
            Argument::U32(ranks),
        ]);
        vec![
            launch(SCATTER_SYMBOL, plan, arguments),
            launch(GATHER_SYMBOL, plan, gather),
        ]
    }

    /// At 2, 4 and 8 ranks, every low-latency bound from 1 to 16 rows
    /// (registered as its grid, four CTAs a row) and every row count up to
    /// the capacity launches what the lowering does: the low-latency entry over
    /// one state of the bound's rows, with the bound's capacity, else the
    /// transport as it launches without the entry.
    #[test]
    fn low_latency_launches_equal_the_lowering_at_every_row_count() {
        let capacity = 64;
        for ranks in [2, 4, 8] {
            let rank = ranks - 1;
            let plain = Sm120::new()
                .all_reduce(&reduce(ranks, WIDTH), rank, Need::rows(capacity))
                .unwrap();
            for ll_rows in 1..=16 {
                let prepared = Sm120::new()
                    .with_low_latency_all_reduce(ll_rows * 4)
                    .all_reduce(&reduce(ranks, WIDTH), rank, Need::rows(capacity))
                    .unwrap();
                let info = prepared.info();
                // The transport's states as without the entry, then its own.
                let ll = plain.info().state.len();
                let bytes = ll_f32::state_bytes(ranks, u64::from(ll_rows) * WIDTH).unwrap();
                assert_eq!(info.state[ll..], [bytes.inbox, bytes.epochs]);
                assert_eq!(info.state[..ll], plain.info().state);
                assert_eq!(info.symbols[0], (ll_f32::SYMBOL.to_owned(), 0));
                assert_eq!(info.guarantees, Invariance::NONE);
                let buffers = buffers(info, ranks);
                for rows in 1..=capacity {
                    let launches = prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap();
                    assert_eq!(
                        launches,
                        lowering(ranks, rank, rows, ll_rows, ll),
                        "{ranks} ranks, {ll_rows} low-latency rows, {rows} rows"
                    );
                }
            }
        }
    }

    /// Every row count the entry serves takes the one state at one
    /// capacity, so each CTA's epoch cell advances over all of them alike
    /// and its parity agrees on every rank; the transport never touches
    /// it, and its slots (two parities, a slot per source and CTA) fill
    /// the inbox exactly.
    #[test]
    fn low_latency_rows_share_one_state_at_one_capacity() {
        let (ranks, ll_rows) = (4, 8);
        let prepared = Sm120::new()
            .with_low_latency_all_reduce(ll_rows * 4)
            .all_reduce(&reduce(ranks, WIDTH), 0, Need::rows(64))
            .unwrap();
        let info = prepared.info().clone();
        let buffers = buffers(&info, ranks);
        let ll = info.state.len() - 2;
        let capacity_ctas = ll_rows * 4;
        assert_eq!(
            info.state[ll],
            2 * u64::from(ranks) * u64::from(capacity_ctas) * 4 * 8 * 256
        );
        assert_eq!(info.state[ll + 1], u64::from(capacity_ctas) * 4);
        for rows in [1, 2, 1, 8, 5, 64, 3, 9, 1] {
            let launches = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            let handles: Vec<_> = launches
                .iter()
                .flat_map(|launch| &launch.arguments.0)
                .flat_map(|argument| match argument {
                    Argument::DeviceBuffer(handle) => vec![*handle],
                    Argument::DevicePointerArray(handles) => handles.clone(),
                    _ => Vec::new(),
                })
                .collect();
            let ll_handles = [handle(0, ll), handle(0, ll + 1)];
            if rows <= ll_rows {
                let args = &launches[0].arguments.0;
                assert_eq!(args[3], Argument::DeviceBuffer(handle(0, ll + 1)));
                assert_eq!(args[5], Argument::U32(capacity_ctas));
            } else {
                assert!(!handles.iter().any(|h| ll_handles.contains(h)));
            }
        }
    }

    /// The entry's choice follows the op's fields and the registrations,
    /// in whatever order a model registers them: the least grid that holds
    /// a row count serves it, registrations that hold the same rows of the
    /// op launch the same and are one, and the op's info and launches do
    /// not change with the order.
    #[test]
    fn registrations_serve_in_every_order() {
        let (ranks, capacity) = (4, 32);
        let prepare = |order: &[u32]| {
            let backend = order.iter().fold(Sm120::new(), |backend, &ctas| {
                backend.with_low_latency_all_reduce(ctas)
            });
            let prepared = backend
                .all_reduce(&reduce(ranks, WIDTH), 1, Need::rows(capacity))
                .unwrap();
            let info = prepared.info().clone();
            let buffers = buffers(&info, ranks);
            let launches = (1..=capacity)
                .map(|rows| {
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                })
                .collect::<Vec<_>>();
            (info, launches)
        };
        // Grids of 2, 4 and 4 rows (the last two of 16 and 19 CTAs hold
        // the same rows).
        let registered = [19, 8, 16];
        let all = crate::backend::select::orders(&registered);
        let (info, launches) = prepare(&all[0]);
        for order in &all[1..] {
            assert_eq!(
                prepare(order),
                (info.clone(), launches.clone()),
                "{order:?}"
            );
        }
        assert!(
            info.implementation
                .starts_with("sm120.comm.allreduce.f32.v1[low_latency_rows=2/4,")
        );
        let ll = info.state.len() - 4;
        for rows in 1..=4 {
            let args = &launches[rows - 1][0].arguments.0;
            let (state, ctas) = if rows <= 2 { (ll, 8) } else { (ll + 2, 16) };
            assert_eq!(args[3], Argument::DeviceBuffer(handle(1, state + 1)));
            assert_eq!(args[5], Argument::U32(ctas));
        }
        assert_eq!(launches[4][0].symbol, ONE_SHOT_SYMBOL);
        // A registration repeated is one.
        assert_eq!(prepare(&[8, 8]), prepare(&[8]));
    }

    /// The registration's key is the rows of the op its grid holds, which
    /// its launches follow.
    #[test]
    fn low_latency_keys_hold_the_rows_the_grid_holds() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let names = field_names!(LowLatency { ctas });
        let fields: Vec<Field<LowLatency>> =
            vec![("ctas", Class::Launch, Box::new(|e| e.ctas = 16))];
        let capacity = 32;
        keys_follow_launches(
            &LowLatency { ctas: 8 },
            names,
            fields,
            |e| low_latency_key(low_latency_rows(e, WIDTH).unwrap()),
            |e| {
                let prepared = Sm120::new()
                    .with_low_latency_all_reduce(e.ctas)
                    .all_reduce(&reduce(4, WIDTH), 0, Need::rows(capacity))?;
                let buffers = buffers(prepared.info(), 4);
                let launches = (1..=4)
                    .map(|rows| {
                        prepared
                            .launches(&buffers, View::EMPTY, Counts::rows(rows))
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                Ok((String::new(), (prepared.info().state.clone(), launches)))
            },
        );
    }

    /// The entry serves only ops it takes: power-of-two ranks, rows of
    /// whole slices and a row its grid holds; other ops take the transport
    /// as without it.
    #[test]
    fn ops_the_entry_does_not_take_use_the_transport() {
        let ll = Sm120::new().with_low_latency_all_reduce(8);
        for (ranks, width) in [(6, WIDTH), (4, 1000), (4, 16384)] {
            let with = ll
                .all_reduce(&reduce(ranks, width), 0, Need::rows(16))
                .unwrap();
            let without = Sm120::new()
                .all_reduce(&reduce(ranks, width), 0, Need::rows(16))
                .unwrap();
            assert_eq!(with.info(), without.info(), "{ranks} ranks of {width}");
        }
        // A 2048-element row fits the grid four times.
        let narrow = ll.all_reduce(&reduce(2, 2048), 0, Need::rows(16)).unwrap();
        assert!(narrow.info().implementation.contains("low_latency_rows=4,"));
    }

    /// The reported state is the layout a model allocates today at its
    /// hidden width of 4096 and its prefill capacity of 8192 rows, at 2, 4
    /// and 8 ranks: a one-shot state per one-shot row count (the rows the
    /// entry serves included), the phased pair's of the capacity, then the
    /// entry's one state of its grid's rows (2 rows, 8 CTAs). The state and
    /// the capacity argument follow the registration alone, so every
    /// capacity the op is prepared at launches the same entry calls.
    #[test]
    fn state_follows_the_served_layout_at_every_capacity() {
        let (capacity, ll_rows) = (8192_u32, 2_u32);
        for ranks in [2, 4, 8] {
            let ranks64 = u64::from(ranks);
            let one_shot_rows = (1..=capacity)
                .take_while(|&rows| {
                    comm::choose(&reduce(ranks, u64::from(rows) * WIDTH)).unwrap()
                        == Variant::OneShot
                })
                .count() as u32;
            let mut layout = Vec::new();
            for rows in 1..=u64::from(one_shot_rows) {
                let (elements, tiles) = (rows * WIDTH, rows);
                layout.extend([ranks64 * elements * 4, 2 * ranks64 * tiles * 4, tiles * 4]);
            }
            let phased =
                comm::phased_buffers(ranks, u64::from(capacity) * WIDTH, DType::F32).unwrap();
            layout.extend([phased.inbox_bytes, phased.flag_bytes, phased.epoch_bytes]);
            let ll = ll_f32::state_bytes(ranks, u64::from(ll_rows) * WIDTH).unwrap();
            layout.extend([ll.inbox, ll.epochs]);
            let backend = Sm120::new().with_low_latency_all_reduce(8);
            let prepared = backend
                .all_reduce(&reduce(ranks, WIDTH), 0, Need::rows(capacity))
                .unwrap();
            assert_eq!(prepared.info().state, layout, "{ranks} ranks");
            if ranks == 4 {
                assert_eq!(one_shot_rows, 16);
                assert_eq!(layout.len(), 16 * 3 + 3 + 2);
                assert_eq!(layout[48..], [phased.inbox_bytes, 82 * 4, 4, 524288, 32]);
            }
            // Fewer rows keep the entry's state and its capacity argument.
            for need in [1, 2, 3] {
                let small = backend
                    .all_reduce(&reduce(ranks, WIDTH), 0, Need::rows(need))
                    .unwrap();
                let state = &small.info().state;
                assert_eq!(state[state.len() - 2..], [ll.inbox, ll.epochs]);
                let buffers = buffers(small.info(), ranks);
                let launches = small
                    .launches(&buffers, View::EMPTY, Counts::rows(1))
                    .unwrap();
                assert_eq!(launches[0].symbol, ll_f32::SYMBOL);
                assert_eq!(launches[0].arguments.0[5], Argument::U32(8));
            }
        }
    }

    #[test]
    fn low_latency_refusals() {
        let query = reduce(4, WIDTH);
        // Grids past the entry's CTAs, or none.
        for ctas in [0, ll_f32::MAX_CTAS + 1] {
            assert!(matches!(
                Sm120::new()
                    .with_low_latency_all_reduce(ctas)
                    .all_reduce(&query, 0, Need::rows(16)),
                Err(Refused::Rule(rule)) if rule.contains("1 to 128 CTAs")
            ));
        }
        let ll = Sm120::new().with_low_latency_all_reduce(ll_f32::MAX_CTAS);
        // Ranks past the transport's.
        assert!(
            ll.all_reduce(&reduce(16, WIDTH), 0, Need::rows(16))
                .is_err()
        );
        // Exactness neither entry declares.
        assert!(matches!(
            ll.all_reduce(&query, 0, Need::rows(16).with_exact(Invariance::one_row(2))),
            Err(Refused::Inexact { guaranteed, .. }) if guaranteed == Invariance::NONE
        ));
        let prepared = ll.all_reduce(&query, 2, Need::rows(16)).unwrap();
        let info = prepared.info().clone();
        let buffers = buffers(&info, 4);
        let ll_state = info.state.len() - 2;
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(17))
                .is_err()
        );
        // Undersized state views: a peer's inbox, this rank's epochs.
        for (rank, region) in [(0, ll_state), (2, ll_state + 1)] {
            let mut short = buffers.clone();
            short.ranks[rank].state[region].bytes -= 1;
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(1))
                    .is_err()
            );
        }
        let mut missing = buffers.clone();
        missing.ranks[3].state.pop();
        assert!(
            prepared
                .launches(&missing, View::EMPTY, Counts::rows(16))
                .is_err()
        );
        let mut short = buffers.clone();
        short.input.bytes = 16 * WIDTH * 4 - 1;
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(16))
                .is_err()
        );
    }
}
