//! The fused BF16 all-reduce (residual add and HF RMSNorm) through the
//! kernel base's push, wide and phased entries (`comm`).
//!
//! The use chooses the entries when the op is prepared:
//! - Prefill takes the exact push entries up to 32 rows and the round-once
//!   phased pair above, so a prompt row normalizes as it would in any chunk.
//! - A decode or verify step whose rows need one-row exactness takes the
//!   exact push entries, up to their 32 rows.
//! - Other decode and verify steps take the push entries up to 32 rows
//!   (the decode entries up to `DECODE_MAX_ROWS`), then in a graph the wide
//!   entry where the tuning lists it, else the phased pair.
//!
//! The exact entries are the model's verify cubin entries
//! (`comm::verify_cubin_spec`); at low-latency row counts, whose rows
//! already sum in the one-row order, the decode entries serve instead. A
//! routed input (the MoE combine folded in) takes the combine form of each
//! push or wide entry; the phased pair has none.
//!
//! The op guarantees what the declarations of the entries its use launches
//! at each row count claim together (`Invariance::routed`): the comm
//! declarations for the push, decode, wide and phased entries, the model's
//! for its exact entries. The decode entries declare one-row bits up to the
//! low-latency rows only; the exact entries' one-row bits are the decode
//! entries' where the comm catalog declares so
//! (`comm::verify_one_row_reference`), so there an exact use of up to 32
//! rows keeps one-row bits at all its rows. Elsewhere it claims the
//! low-latency rows, and a `Need.exact` past that is refused at prepare.
//!
//! A model's overlap tuning (fewer CTAs for the phased pair beside other
//! work) is its own SM120 choice: it adjusts the plans of the prepared
//! launches, as it adjusts the comm planner's today.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::collective::{FusedAllReduceBuffers, ReduceInput};
use kernel_core::contract::{ArgumentList, CommQuery, DType, LaunchPlan, MAX_COMM_RANKS, Refusal};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::properties::{Invariance, RowRoute, Rows};

use super::select::{Candidate, choose};
use super::{MAX_CAPACITY, check_plan};
use crate::comm::{
    self, FusedOneShotCall, FusedWithResidualOutputCall, OneShotCall, PushProtocol, Variant,
};

/// A model's exact push entries: its verify cubin's plain and combine
/// entries, and the guarantees it declares for them.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ExactEntries {
    pub(super) plain: String,
    pub(super) combine: String,
    pub(super) invariance: Invariance,
}

const ONE_SHOT_V2: &str = "chop_comm_allreduce_oneshot_norm_bf16_v2";
const COMBINE_V1: &str = "chop_comm_allreduce_combine_norm_bf16_v1";
/// The routed rows each combine entry sums (`input::ROUTED_COMBINE_TOP8`).
const COMBINE_ROUTES: u32 = 8;

/// Where the all-reduce runs, from its use and need.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Route {
    Prefill,
    Exact,
    Graph { wide: bool },
}

/// One row count's launches: a push or wide entry, or the phased pair.
#[derive(Clone, Debug)]
enum Form {
    Push(String),
    Phased,
}

/// The fused all-reduce `op` of `input` at `usage` that rank `rank`
/// launches; `wide` is false when the model keeps graphs off the wide
/// entries.
pub(super) fn prepare(
    exact: &[ExactEntries],
    wide: bool,
    op: &CommQuery,
    input: ReduceInput,
    usage: OperationUse,
    rank: u32,
    need: Need,
) -> Result<Prepared<FusedAllReduceBuffers<View>>, Refused> {
    let op = FusedOp::new(exact, wide, *op, input, usage, rank, need)?;
    need.check(&op.info)?;
    Ok(Box::new(op))
}

struct FusedOp {
    ranks: u32,
    rank: u32,
    width: u64,
    epsilon: f32,
    /// Routed rows per row of a routed input, one for plain rows.
    routes: u64,
    /// Each row count's form and plan, from one to the capacity.
    forms: Vec<(Form, LaunchPlan)>,
    info: OpInfo,
}

impl FusedOp {
    fn new(
        exact: &[ExactEntries],
        wide: bool,
        op: CommQuery,
        input: ReduceInput,
        usage: OperationUse,
        rank: u32,
        need: Need,
    ) -> Result<Self, Refused> {
        let CommQuery::AllReduce {
            ranks,
            elements: width,
            dtype: DType::Bf16,
            fused_residual: true,
            fused_rmsnorm: Some(norm),
        } = op
        else {
            return Err(Refused::Rule(
                "the SM120 fused all-reduce adds a residual to BF16 rows and normalizes them"
                    .into(),
            ));
        };
        if width != u64::from(norm.width) {
            return Err(Refused::Rule(
                "a fused all-reduce query describes one normalized row".into(),
            ));
        }
        if ranks as usize > MAX_COMM_RANKS || rank >= ranks {
            return Err(Refused::Rule(format!(
                "an SM120 collective takes a rank below at most {MAX_COMM_RANKS} ranks"
            )));
        }
        let capacity = need.capacity;
        if capacity == 0 || capacity > usage.capacity || capacity > MAX_CAPACITY {
            return Err(Refused::Rule(format!(
                "an SM120 fused all-reduce serves 1 to {MAX_CAPACITY} rows of its use"
            )));
        }
        let routed = match input {
            ReduceInput::Rows => false,
            ReduceInput::Routed {
                routes: COMBINE_ROUTES,
            } => true,
            ReduceInput::Routed { .. } => {
                return Err(Refused::Rule(format!(
                    "the SM120 combine entries sum {COMBINE_ROUTES} routed rows"
                )));
            }
        };
        let route = if usage.phase == ExecutionPhase::Prefill {
            Route::Prefill
        } else if need.exact.rows != Rows::None {
            Route::Exact
        } else {
            Route::Graph {
                wide: wide && usage.graph,
            }
        };
        if routed && route == Route::Prefill {
            return Err(Refused::Rule(
                "SM120 folds the combine into decode and verify all-reduces only".into(),
            ));
        }
        if route == Route::Exact && capacity > comm::verify_entry_rows(ranks)? {
            return Err(Refused::Rule(
                "the SM120 exact all-reduce serves at most the push entries' rows".into(),
            ));
        }
        let query = |rows: u32| CommQuery::AllReduce {
            ranks,
            elements: u64::from(rows) * width,
            dtype: DType::Bf16,
            fused_residual: true,
            fused_rmsnorm: Some(norm),
        };
        // The largest reduction first, so no table is built for one SM120
        // cannot plan.
        comm::choose(&query(capacity))?;
        let declared = comm::declared_entries();
        let declared = |symbol: &str| {
            declared
                .iter()
                .find(|entry| entry.symbol == symbol)
                .map_or(Invariance::NONE, |entry| entry.invariance)
        };
        let mut forms = Vec::with_capacity(capacity as usize);
        let mut symbols: Vec<(String, u32)> = Vec::new();
        let mut routes = Vec::with_capacity(capacity as usize);
        for rows in 1..=capacity {
            let query = query(rows);
            let (variant, plan) = if route == (Route::Graph { wide: true }) {
                (comm::graph_choose(&query)?, comm::graph_plan(&query, rows)?)
            } else {
                (comm::choose(&query)?, comm::plan(&query, rows)?)
            };
            check_plan(&plan)?;
            // Each row count's form, the declarations it launches under
            // and the one-row entry their rows claim refers to: the entry
            // itself unless the comm catalog names another.
            let (form, claim, reference) = match variant {
                Variant::FusedWide => {
                    let symbol = if routed {
                        comm::WIDE_COMBINE_SYMBOL
                    } else {
                        comm::WIDE_SYMBOL
                    };
                    (Form::Push(symbol.into()), declared(symbol), symbol.into())
                }
                Variant::FusedOneShot if !matches!(route, Route::Exact | Route::Prefill) => {
                    let symbol = push_symbol(ranks, rows, routed);
                    (Form::Push(symbol.into()), declared(symbol), symbol.into())
                }
                Variant::FusedOneShot => {
                    // Low-latency rows already sum in the one-row order and
                    // take the decode entries, under their own declarations;
                    // only the other rows launch the exact entries.
                    if comm::push_protocol(ranks, rows)? == PushProtocol::LowLatency {
                        let symbol = push_symbol(ranks, rows, routed);
                        (Form::Push(symbol.into()), declared(symbol), symbol.into())
                    } else {
                        let (symbol, claims) = exact_entries(exact, routed, need)?;
                        let reference = comm::verify_one_row_reference(ranks, routed)
                            .map_or_else(|| symbol.clone(), String::from);
                        (Form::Push(symbol), claims, reference)
                    }
                }
                Variant::FusedTwoShot if !routed => {
                    let pair = [
                        comm::PREFILL_ROUND_ONCE_SYMBOL,
                        comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL,
                    ];
                    let claim = declared(pair[0]).meet(declared(pair[1]));
                    (Form::Phased, claim, pair[1].into())
                }
                _ => {
                    return Err(Refused::Rule(
                        "SM120 folds the combine into the push and wide entries only".into(),
                    ));
                }
            };
            let launched: &[&str] = match &form {
                Form::Push(symbol) => &[symbol.as_str()],
                Form::Phased => &[
                    comm::PREFILL_ROUND_ONCE_SYMBOL,
                    comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL,
                ],
            };
            for &symbol in launched {
                if !symbols.iter().any(|(own, _)| own == symbol) {
                    symbols.push((symbol.to_owned(), plan.dynamic_shared_bytes));
                }
            }
            routes.push(RowRoute {
                rows,
                invariance: claim,
                reference,
            });
            forms.push((form, plan));
        }
        let state = comm::fused_state_buffers(ranks, capacity)?;
        let mut info = OpInfo::new(
            format!(
                "sm120.comm.allreduce_norm.v1[{}]",
                symbols
                    .iter()
                    .map(|(symbol, _)| symbol.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Invariance::routed(&routes),
            symbols,
        );
        info.state = vec![state.inbox_bytes, state.flag_bytes, state.epoch_bytes];
        Ok(Self {
            ranks,
            rank,
            width,
            epsilon: norm.eps,
            routes: if routed { u64::from(COMBINE_ROUTES) } else { 1 },
            forms,
            info,
        })
    }
}

/// The exact entry a row count of plain or `routed` rows launches,
/// whatever the registration order (`select::choose`), and what the
/// registrations of that launch declare: exact entries serve every exact
/// and prefill use alike. Pairs are told apart by the entry the rows
/// launch alone (the plain entry for plain rows, the combine for routed),
/// so pairs differing in the other are one. A use none of whose rows
/// launch them never resolves them, so an unused registration changes
/// nothing.
fn exact_entries(
    exact: &[ExactEntries],
    routed: bool,
    need: Need,
) -> Result<(String, Invariance), Refused> {
    if exact.is_empty() {
        return Err(Refused::Rule(
            "an exact or prefill SM120 all-reduce needs the model's exact entries".into(),
        ));
    }
    let chosen = choose(
        "exact all-reduce",
        exact.iter().map(|entry| {
            let launched = exact_launch(entry, routed);
            Candidate::new(
                launched.clone(),
                launched.clone(),
                launched,
                entry.invariance,
            )
        }),
        Need {
            capacity: 1,
            ..need
        },
    )?;
    Ok((chosen.entry, chosen.guarantees))
}

/// The exact entry plain or `routed` rows launch.
fn exact_launch(entry: &ExactEntries, routed: bool) -> String {
    let ExactEntries {
        plain,         // plain rows' entry
        combine,       // routed rows' entry
        invariance: _, // claims
    } = entry;
    if routed { combine } else { plain }.clone()
}

/// The push entry of `rows` rows: the decode entries up to
/// `DECODE_MAX_ROWS` (the same protocols and bits), v2 and combine v1
/// above.
fn push_symbol(ranks: u32, rows: u32, routed: bool) -> &'static str {
    let decode = rows <= comm::DECODE_MAX_ROWS && comm::push_protocol(ranks, rows).is_ok();
    match (decode, routed) {
        (true, false) => comm::DECODE_SYMBOL_V2,
        (true, true) => comm::DECODE_COMBINE_SYMBOL,
        (false, false) => ONE_SHOT_V2,
        (false, true) => COMBINE_V1,
    }
}

impl PreparedOp<FusedAllReduceBuffers<View>> for FusedOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &FusedAllReduceBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        let (form, plan) = rows
            .checked_sub(1)
            .and_then(|index| self.forms.get(index as usize))
            .ok_or(Refusal(
                "fused all-reduce rows exceed the prepared capacity",
            ))?;
        if buffers.ranks.len() != self.ranks as usize {
            return Err(Refusal("a fused all-reduce takes the views of every rank"));
        }
        let elements = u64::from(rows) * self.width;
        let row_bytes = elements * 2;
        buffers.input.covers(row_bytes * self.routes)?;
        for view in [
            buffers.residual.input,
            buffers.output,
            buffers.residual.output,
        ] {
            view.covers(row_bytes)?;
        }
        buffers.weight.covers(self.width * 2)?;
        let live_rows = match (self.routes > 1, buffers.live_rows) {
            (true, Some(live_rows)) => {
                live_rows.covers(4)?;
                Some(live_rows.handle)
            }
            (false, None) => None,
            _ => {
                return Err(Refusal(
                    "a routed fused all-reduce takes a live count, plain rows none",
                ));
            }
        };
        let mut peers = Vec::with_capacity(buffers.ranks.len());
        for rank in &buffers.ranks {
            let [inbox, flags, epoch] = rank.state[..] else {
                return Err(Refusal("a fused all-reduce rank lacks its state regions"));
            };
            for (view, &bytes) in [inbox, flags, epoch].iter().zip(&self.info.state) {
                view.covers(bytes)?;
            }
            if matches!(form, Form::Phased) {
                rank.output.covers(row_bytes)?;
            }
            peers.push((rank.output, inbox, flags, epoch));
        }
        let call = || OneShotCall {
            input: buffers.input.handle,
            output: buffers.output.handle,
            peer_inboxes: peers.iter().map(|p| p.1.handle).collect(),
            peer_flags: peers.iter().map(|p| p.2.handle).collect(),
            epoch: peers[self.rank as usize].3.handle,
            elements,
            rank: self.rank,
            ranks: self.ranks,
        };
        let launch = |symbol: &str, arguments: ArgumentList| Launch {
            symbol: symbol.into(),
            rows,
            plan: *plan,
            arguments,
            collective: true,
            persistent: false,
        };
        Ok(match form {
            Form::Push(symbol) => {
                let fused = FusedWithResidualOutputCall {
                    fused: FusedOneShotCall {
                        base: call(),
                        residual: Some(buffers.residual.input.handle),
                        weights: buffers.weight.handle,
                        epsilon: self.epsilon,
                    },
                    residual_output: buffers.residual.output.handle,
                };
                let arguments = match live_rows {
                    Some(live_rows) => comm::fused_combine_arguments(fused, Some(live_rows))?,
                    None => comm::fused_arguments_v2(fused)?,
                };
                vec![launch(symbol, arguments)]
            }
            Form::Phased => {
                // The scatter adds the residual; the gather sends the sums
                // and normalizes them into every rank's output.
                let gather = comm::round_once_gather_arguments(
                    &call(),
                    peers.iter().map(|p| p.0.handle).collect(),
                    buffers.weight.handle,
                    self.epsilon,
                    buffers.residual.output.handle,
                )?;
                let scatter =
                    comm::round_once_scatter_arguments(call(), buffers.residual.input.handle)?;
                vec![
                    launch(comm::PREFILL_ROUND_ONCE_SYMBOL, scatter),
                    launch(comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL, gather),
                ]
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};
    use kernel_core::backend::Backend;
    use kernel_core::collective::RankViews;
    use kernel_core::contract::{Argument, RmsNormEpilogue, RmsNormKind};
    use kernel_core::properties::Batch;
    use kernel_core::rmsnorm::FusedResidual;

    const PLAIN: &str = "chop_comm_test_verify_norm_bf16_v1";
    const COMBINE: &str = "chop_comm_test_verify_combine_norm_bf16_v1";
    const EPS: f32 = 1e-5;

    fn row(ranks: u32, elements: u64) -> CommQuery {
        CommQuery::AllReduce {
            ranks,
            elements,
            dtype: DType::Bf16,
            fused_residual: true,
            fused_rmsnorm: Some(RmsNormEpilogue {
                kind: RmsNormKind::HfBf16Intermediate,
                width: 4096,
                eps: EPS,
            }),
        }
    }

    fn usage(phase: ExecutionPhase, capacity: u32) -> OperationUse {
        OperationUse {
            phase,
            capacity,
            graph: phase != ExecutionPhase::Prefill,
        }
    }

    fn backend() -> Sm120 {
        Sm120::new().with_exact_all_reduce(PLAIN, COMBINE, Invariance::any_rows(32))
    }

    /// Today's selection of a 4096-wide hidden all-reduce's launches by
    /// route, as the bundle builds them: (symbol, plan, arguments).
    fn legacy(
        route: Route,
        rows: u32,
        rank: u32,
        ranks: u32,
        live_rows: Option<u64>,
    ) -> Vec<(String, LaunchPlan, ArgumentList)> {
        let query = row(ranks, u64::from(rows) * 4096);
        let (variant, plan) = if route == (Route::Graph { wide: true }) {
            (
                comm::graph_choose(&query).unwrap(),
                comm::graph_plan(&query, rows).unwrap(),
            )
        } else {
            (
                comm::choose(&query).unwrap(),
                comm::plan(&query, rows).unwrap(),
            )
        };
        let call = || OneShotCall {
            input: 1,
            output: 5,
            peer_inboxes: (0..u64::from(ranks)).map(|r| 100 * (r + 1)).collect(),
            peer_flags: (0..u64::from(ranks)).map(|r| 100 * (r + 1) + 1).collect(),
            epoch: 100 * u64::from(rank + 1) + 2,
            elements: u64::from(rows) * 4096,
            rank,
            ranks,
        };
        let push = |rows: u32| {
            if rows <= comm::DECODE_MAX_ROWS && comm::push_protocol(ranks, rows).is_ok() {
                (comm::DECODE_SYMBOL_V2, comm::DECODE_COMBINE_SYMBOL)
            } else {
                (ONE_SHOT_V2, COMBINE_V1)
            }
        };
        let exact = |rows: u32| {
            if comm::push_protocol(ranks, rows).unwrap() == PushProtocol::LowLatency {
                push(rows)
            } else {
                (PLAIN, COMBINE)
            }
        };
        let fused = || FusedWithResidualOutputCall {
            fused: FusedOneShotCall {
                base: call(),
                residual: Some(2),
                weights: 3,
                epsilon: EPS,
            },
            residual_output: 4,
        };
        match variant {
            Variant::FusedOneShot | Variant::FusedWide => {
                let symbols = if variant == Variant::FusedWide {
                    (comm::WIDE_SYMBOL, comm::WIDE_COMBINE_SYMBOL)
                } else if matches!(route, Route::Exact | Route::Prefill) {
                    exact(rows)
                } else {
                    push(rows)
                };
                let (symbol, arguments) = match live_rows {
                    Some(live) => (
                        symbols.1,
                        comm::fused_combine_arguments(fused(), Some(live)).unwrap(),
                    ),
                    None => (symbols.0, comm::fused_arguments_v2(fused()).unwrap()),
                };
                vec![(symbol.into(), plan, arguments)]
            }
            Variant::FusedTwoShot => {
                let gather = comm::round_once_gather_arguments(
                    &call(),
                    (0..u64::from(ranks)).map(|r| 10 + r).collect(),
                    3,
                    EPS,
                    4,
                )
                .unwrap();
                vec![
                    (
                        comm::PREFILL_ROUND_ONCE_SYMBOL.into(),
                        plan,
                        comm::round_once_scatter_arguments(call(), 2).unwrap(),
                    ),
                    (comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL.into(), plan, gather),
                ]
            }
            _ => unreachable!(),
        }
    }

    /// Operands 1 to 5 (input, residual, weight, residual_output, output),
    /// rank r's output 10 + r and state regions 100 * (r + 1) + i.
    fn buffers(info: &OpInfo, ranks: u32, live_rows: Option<View>) -> FusedAllReduceBuffers<View> {
        FusedAllReduceBuffers {
            input: view(1, u64::MAX),
            residual: FusedResidual {
                input: view(2, u64::MAX),
                output: view(4, u64::MAX),
            },
            weight: view(3, 4096 * 2),
            output: view(5, u64::MAX),
            live_rows,
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
        }
    }

    fn check(
        backend: &Sm120,
        route: Route,
        usage: OperationUse,
        need: Need,
        input: ReduceInput,
        ranks: u32,
        counts: &[u32],
    ) {
        let rank = ranks - 1;
        let prepared = backend
            .fused_all_reduce(&row(ranks, 4096), input, usage, rank, need)
            .unwrap();
        let info = prepared.info().clone();
        let state = comm::fused_state_buffers(ranks, need.capacity).unwrap();
        assert_eq!(
            info.state,
            [state.inbox_bytes, state.flag_bytes, state.epoch_bytes]
        );
        let live = (input != ReduceInput::Rows).then_some(view(6, 4));
        let buffers = buffers(&info, ranks, live);
        for &rows in counts {
            let launches = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            let expected = legacy(route, rows, rank, ranks, live.map(|v| v.handle));
            assert_eq!(launches.len(), expected.len(), "{route:?} {rows} rows");
            for (launch, (symbol, plan, arguments)) in launches.iter().zip(&expected) {
                assert_eq!(&launch.symbol, symbol, "{route:?} {rows} rows");
                assert_eq!(&launch.plan, plan);
                assert_eq!(&launch.arguments, arguments);
                assert!(launch.collective && !launch.persistent && launch.rows == rows);
                assert!(info.symbols.iter().any(|(own, _)| own == symbol));
            }
        }
    }

    #[test]
    fn fused_all_reduce_launches_the_bundle_launches_of_each_use() {
        let counts = [1, 2, 3, 12, 13, 16, 17, 32];
        for ranks in [2, 4, 8] {
            // Decode graphs, the wide entry above 32 rows, the phased pair
            // without it.
            let graph = usage(ExecutionPhase::Decode, 256);
            let wide = [1, 2, 13, 32, 33, 64, 256];
            check(
                &backend(),
                Route::Graph { wide: true },
                graph,
                Need::rows(256),
                ReduceInput::Rows,
                ranks,
                &wide,
            );
            check(
                &backend().without_wide_collectives(),
                Route::Graph { wide: false },
                graph,
                Need::rows(256),
                ReduceInput::Rows,
                ranks,
                &wide,
            );
            // The combine folded in, up to 32 rows.
            let decode = usage(ExecutionPhase::Decode, 32);
            check(
                &backend(),
                Route::Graph { wide: true },
                decode,
                Need::rows(32),
                ReduceInput::Routed { routes: 8 },
                ranks,
                &counts,
            );
            // Exact verify rows, plain and combined.
            let verify = usage(ExecutionPhase::Verify, 32);
            // The decode entries' declared one-row bits are all an exact
            // use guarantees.
            let exact = Need::rows(32).with_exact(Invariance::one_row(2));
            for input in [ReduceInput::Rows, ReduceInput::Routed { routes: 8 }] {
                check(
                    &backend(),
                    Route::Exact,
                    verify,
                    exact,
                    input,
                    ranks,
                    &counts,
                );
            }
            // Prefill: the exact entries, then the round-once pair.
            check(
                &backend(),
                Route::Prefill,
                usage(ExecutionPhase::Prefill, 8192),
                Need::rows(8192),
                ReduceInput::Rows,
                ranks,
                &[1, 2, 3, 17, 32, 33, 100, 8192],
            );
        }
    }

    #[test]
    fn fused_all_reduce_reports_the_declared_guarantees_of_its_entries() {
        let prepare = |ranks, phase, capacity, exact| {
            backend().fused_all_reduce(
                &row(ranks, 4096),
                ReduceInput::Rows,
                usage(phase, capacity),
                0,
                Need::rows(capacity).with_exact(exact),
            )
        };
        let none = Invariance::NONE;
        // Prefill launches the decode entries at low-latency rows, which
        // declare no chunk claim, beside the exact entries and the pair:
        // only batch independence holds for all of them.
        let prefill = prepare(2, ExecutionPhase::Prefill, 512, none).unwrap();
        let independent = Invariance {
            batch: Batch::Independent,
            ..Invariance::NONE
        };
        assert_eq!(prefill.info().guarantees, independent);
        assert!(matches!(
            prepare(2, ExecutionPhase::Prefill, 512, Invariance::PREFILL),
            Err(Refused::Inexact { .. })
        ));
        // Where the exact entries' one-row bits are the decode entries',
        // an exact use keeps them at all its rows (`exact_uses_...`); at
        // eight ranks only at the low-latency rows.
        let exact = Invariance::one_row(2);
        let prepared = prepare(8, ExecutionPhase::Verify, 16, exact).unwrap();
        assert_eq!(prepared.info().guarantees, Invariance::one_row(2));
        assert!(matches!(
            prepare(8, ExecutionPhase::Verify, 16, Invariance::one_row(8)),
            Err(Refused::Inexact { .. })
        ));
        // An exact use that launches only the decode entries.
        let low = prepare(2, ExecutionPhase::Verify, 2, Invariance::one_row(2)).unwrap();
        assert_eq!(low.info().guarantees, Invariance::one_row(2));
        // Decode graphs up to 32 rows keep the decode entries' claim; the
        // wide entry declares nothing, so neither does a graph reaching it.
        let decode = prepare(2, ExecutionPhase::Decode, 32, none).unwrap();
        assert_eq!(decode.info().guarantees, Invariance::one_row(2));
        let wide = prepare(2, ExecutionPhase::Decode, 64, none).unwrap();
        assert_eq!(wide.info().guarantees, Invariance::NONE);
        assert!(matches!(
            prepare(2, ExecutionPhase::Decode, 64, Invariance::PREFILL),
            Err(Refused::Inexact { .. })
        ));
    }

    /// An exact verify use launches the decode entries at the low-latency
    /// rows and the exact entries above. Where the comm catalog declares the
    /// exact entries' one-row bits the decode entries', its rows keep them
    /// at every row count it serves and no more: plain and routed, for the
    /// shared pair and a geometry-renamed one.
    #[test]
    fn exact_uses_keep_one_row_bits_where_the_verify_entries_refer_to_the_decode_entries() {
        let pairs = [
            (comm::VERIFY_SYMBOL, comm::VERIFY_COMBINE_SYMBOL),
            (
                "chop_comm_test_allreduce_verify_norm_bf16_v1",
                "chop_comm_test_allreduce_verify_combine_norm_bf16_v1",
            ),
        ];
        let prepare = |backend: &Sm120, ranks, input, phase, capacity, exact| {
            backend.fused_all_reduce(
                &row(ranks, 4096),
                input,
                usage(phase, capacity),
                0,
                Need::rows(capacity).with_exact(exact),
            )
        };
        let inexact =
            |prepared: Result<_, Refused>| matches!(prepared, Err(Refused::Inexact { .. }));
        for (plain, combine) in pairs {
            let backend =
                Sm120::new().with_exact_all_reduce(plain, combine, Invariance::any_rows(32));
            for ranks in comm::VERIFY_REFERENCE_RANKS {
                for input in [ReduceInput::Rows, ReduceInput::Routed { routes: 8 }] {
                    for capacity in [1, 2, 3, 4, 5, 8, 16, 32] {
                        let verify = ExecutionPhase::Verify;
                        let exact = Invariance::one_row(capacity);
                        let prepared = prepare(&backend, ranks, input, verify, capacity, exact);
                        // At one row, the decode entries' own claim.
                        assert_eq!(
                            prepared.unwrap().info().guarantees,
                            Invariance::one_row(capacity.max(2)),
                            "{plain} {ranks} ranks {input:?} {capacity} rows"
                        );
                        if capacity > 1 {
                            let more = Invariance::one_row(capacity + 1);
                            assert!(inexact(prepare(
                                &backend, ranks, input, verify, capacity, more
                            )));
                        }
                    }
                }
            }
        }
        let verify = |backend: &Sm120, ranks, capacity, exact| {
            prepare(
                backend,
                ranks,
                ReduceInput::Rows,
                ExecutionPhase::Verify,
                capacity,
                exact,
            )
        };
        // Exact entries that claim fewer rows than a use launches them at
        // add nothing to the decode entries' claim.
        let short = Sm120::new().with_exact_all_reduce(PLAIN, COMBINE, Invariance::one_row(4));
        let four = verify(&short, 4, 4, Invariance::one_row(4)).unwrap();
        assert_eq!(four.info().guarantees, Invariance::one_row(4));
        let eight = verify(&short, 4, 8, Invariance::one_row(2)).unwrap();
        assert_eq!(eight.info().guarantees, Invariance::one_row(2));
        assert!(inexact(verify(&short, 4, 8, Invariance::one_row(4))));
        // Without the catalog's reference (eight ranks), only the decode
        // entries' rows.
        let eight = verify(&backend(), 8, 8, Invariance::one_row(2)).unwrap();
        assert_eq!(eight.info().guarantees, Invariance::one_row(2));
        assert!(inexact(verify(&backend(), 8, 8, Invariance::one_row(8))));
        // A prefill use of the push rows launches the same entries, and
        // keeps the decode entries' lack of a chunk claim.
        let prefill = prepare(
            &backend(),
            2,
            ReduceInput::Rows,
            ExecutionPhase::Prefill,
            32,
            Invariance::NONE,
        )
        .unwrap();
        assert_eq!(prefill.info().guarantees, Invariance::one_row(32));
    }

    #[test]
    fn fused_all_reduce_refuses_what_sm120_does_not_serve() {
        let decode = usage(ExecutionPhase::Decode, 32);
        let refused = |backend: &Sm120, op: &CommQuery, input, usage, rank, need| {
            matches!(
                backend.fused_all_reduce(op, input, usage, rank, need),
                Err(Refused::Rule(_))
            )
        };
        let rows = ReduceInput::Rows;
        // Another width, dtype or norm; a query of more than one row.
        let CommQuery::AllReduce { fused_rmsnorm, .. } = row(2, 4096) else {
            unreachable!()
        };
        let narrow = CommQuery::AllReduce {
            ranks: 2,
            elements: 2048,
            dtype: DType::Bf16,
            fused_residual: true,
            fused_rmsnorm: fused_rmsnorm.map(|norm| RmsNormEpilogue {
                width: 2048,
                ..norm
            }),
        };
        assert!(refused(&backend(), &narrow, rows, decode, 0, Need::rows(4)));
        assert!(refused(
            &backend(),
            &row(2, 8192),
            rows,
            decode,
            0,
            Need::rows(4)
        ));
        let plain = CommQuery::AllReduce {
            ranks: 2,
            elements: 4096,
            dtype: DType::Bf16,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        assert!(refused(&backend(), &plain, rows, decode, 0, Need::rows(4)));
        // Sixteen ranks are a valid contract past SM120's pointer slots.
        assert!(refused(
            &backend(),
            &row(16, 4096),
            rows,
            decode,
            0,
            Need::rows(4)
        ));
        assert!(refused(
            &backend(),
            &row(2, 4096),
            rows,
            decode,
            2,
            Need::rows(4)
        ));
        // Capacities past the use's; exact rows past the push entries.
        assert!(refused(
            &backend(),
            &row(2, 4096),
            rows,
            decode,
            0,
            Need::rows(33)
        ));
        let verify = usage(ExecutionPhase::Verify, 64);
        let exact = Need::rows(64).with_exact(Invariance::one_row(8));
        assert!(refused(&backend(), &row(2, 4096), rows, verify, 0, exact));
        // Prefill and exact rows need the model's exact entries.
        let prefill = usage(ExecutionPhase::Prefill, 64);
        assert!(refused(
            &Sm120::new(),
            &row(2, 4096),
            rows,
            prefill,
            0,
            Need::rows(64)
        ));
        assert!(refused(
            &Sm120::new(),
            &row(2, 4096),
            rows,
            usage(ExecutionPhase::Verify, 8),
            0,
            Need::rows(8).with_exact(Invariance::one_row(8))
        ));
        // Routed rows: eight routes, graphs only, no phased pair.
        let routed = ReduceInput::Routed { routes: 8 };
        assert!(refused(
            &backend(),
            &row(2, 4096),
            ReduceInput::Routed { routes: 4 },
            decode,
            0,
            Need::rows(4)
        ));
        assert!(refused(
            &backend(),
            &row(2, 4096),
            routed,
            prefill,
            0,
            Need::rows(4)
        ));
        let narrow_graph = backend().without_wide_collectives();
        let graph = usage(ExecutionPhase::Decode, 64);
        assert!(refused(
            &narrow_graph,
            &row(2, 4096),
            routed,
            graph,
            0,
            Need::rows(64)
        ));
        // Launch checks: sizes, the live count, every rank's views.
        let prepared = backend()
            .fused_all_reduce(
                &row(2, 4096),
                rows,
                usage(ExecutionPhase::Prefill, 64),
                1,
                Need::rows(64),
            )
            .unwrap();
        let info = prepared.info().clone();
        let good = buffers(&info, 2, None);
        assert!(
            prepared
                .launches(&good, View::EMPTY, Counts::rows(64))
                .is_ok()
        );
        assert!(
            prepared
                .launches(&good, View::EMPTY, Counts::rows(65))
                .is_err()
        );
        let mut short = good.clone();
        short.residual.input.bytes = 64 * 4096 * 2 - 1;
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(64))
                .is_err()
        );
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(63))
                .is_ok()
        );
        let mut short = good.clone();
        short.ranks[0].output.bytes = 64 * 4096 * 2 - 1;
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(64))
                .is_err()
        );
        let mut short = good.clone();
        short.ranks[1].state[1].bytes -= 1;
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        let mut live = good.clone();
        live.live_rows = Some(view(6, 4));
        assert!(
            prepared
                .launches(&live, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        let mut missing = good.clone();
        missing.ranks.pop();
        assert!(
            prepared
                .launches(&missing, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        // The normalized output may not alias the residual output.
        let mut aliased = good.clone();
        aliased.output = aliased.residual.output;
        assert!(
            prepared
                .launches(&aliased, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        let combine = backend()
            .fused_all_reduce(&row(2, 4096), routed, decode, 0, Need::rows(32))
            .unwrap();
        let info = combine.info().clone();
        assert!(
            combine
                .launches(&buffers(&info, 2, None), View::EMPTY, Counts::rows(1))
                .is_err()
        );
        let mut routed_short = buffers(&info, 2, Some(view(6, 4)));
        routed_short.input.bytes = 32 * 8 * 4096 * 2 - 1;
        assert!(
            combine
                .launches(&routed_short, View::EMPTY, Counts::rows(32))
                .is_err()
        );
        let launches = combine
            .launches(
                &buffers(&info, 2, Some(view(6, 4))),
                View::EMPTY,
                Counts::rows(32),
            )
            .unwrap();
        assert_eq!(
            launches[0].arguments.0.last(),
            Some(&Argument::DeviceBuffer(6))
        );
    }

    #[test]
    fn exact_entries_resolve_in_any_registration_order() {
        let other = "chop_comm_other_verify_norm_bf16_v1";
        let prefill = usage(ExecutionPhase::Prefill, 64);
        let prepare = |backend: &Sm120, usage| {
            backend.fused_all_reduce(&row(2, 4096), ReduceInput::Rows, usage, 0, Need::rows(64))
        };
        // Identical registrations count as one.
        let once = prepare(&backend(), prefill).unwrap();
        let twice = backend().with_exact_all_reduce(PLAIN, COMBINE, Invariance::any_rows(32));
        let twice = prepare(&twice, prefill).unwrap();
        assert_eq!(once.info(), twice.info());
        let info = once.info().clone();
        let buffers = buffers(&info, 2, None);
        for rows in [3, 64] {
            assert_eq!(
                once.launches(&buffers, View::EMPTY, Counts::rows(rows)),
                twice.launches(&buffers, View::EMPTY, Counts::rows(rows))
            );
        }
        // Two different sets tie in either order, named sorted; a use that
        // launches no exact entry is not affected.
        for backend in [
            backend().with_exact_all_reduce(other, COMBINE, Invariance::any_rows(32)),
            Sm120::new()
                .with_exact_all_reduce(other, COMBINE, Invariance::any_rows(32))
                .with_exact_all_reduce(PLAIN, COMBINE, Invariance::any_rows(32)),
        ] {
            assert_eq!(
                prepare(&backend, prefill).err(),
                Some(Refused::Rule(format!(
                    "two SM120 exact all-reduce entries serve the op, {other} and {PLAIN}"
                )))
            );
            assert!(prepare(&backend, usage(ExecutionPhase::Decode, 64)).is_ok());
        }
        // The same set with other declarations launches the same: one set,
        // claiming what both declare.
        let declared = backend().with_exact_all_reduce(PLAIN, COMBINE, Invariance::NONE);
        let declared = prepare(&declared, prefill).unwrap();
        assert_eq!(declared.info().symbols, once.info().symbols);
        assert_eq!(
            declared.info().guarantees,
            once.info().guarantees.meet(Invariance::NONE)
        );
    }

    #[test]
    fn unused_exact_registrations_change_nothing() {
        let conflicting = || {
            backend().with_exact_all_reduce(
                "chop_comm_other_verify_norm_bf16_v1",
                COMBINE,
                Invariance::any_rows(32),
            )
        };
        for ranks in [2, 4, 8] {
            // Uses whose rows all take the decode entries at low-latency
            // rows launch no exact entry: none, one or two conflicting
            // registrations prepare the same op.
            for (phase, need) in [
                (ExecutionPhase::Prefill, Need::rows(2)),
                (
                    ExecutionPhase::Verify,
                    Need::rows(2).with_exact(Invariance::one_row(2)),
                ),
            ] {
                let prepare = |backend: &Sm120| {
                    backend
                        .fused_all_reduce(
                            &row(ranks, 4096),
                            ReduceInput::Rows,
                            usage(phase, 2),
                            0,
                            need,
                        )
                        .unwrap()
                };
                let alone = prepare(&Sm120::new());
                let info = alone.info().clone();
                let buffers = buffers(&info, ranks, None);
                for backend in [backend(), conflicting()] {
                    let prepared = prepare(&backend);
                    assert_eq!(prepared.info(), &info);
                    for rows in [1, 2] {
                        assert_eq!(
                            prepared.launches(&buffers, View::EMPTY, Counts::rows(rows)),
                            alone.launches(&buffers, View::EMPTY, Counts::rows(rows))
                        );
                    }
                }
            }
            // A use with a row that launches them resolves them.
            let three = |backend: &Sm120| {
                backend.fused_all_reduce(
                    &row(ranks, 4096),
                    ReduceInput::Rows,
                    usage(ExecutionPhase::Prefill, 3),
                    0,
                    Need::rows(3),
                )
            };
            assert!(three(&backend()).is_ok());
            assert!(matches!(three(&Sm120::new()), Err(Refused::Rule(_))));
            assert!(matches!(three(&conflicting()), Err(Refused::Rule(_))));
        }
    }

    /// The auditor's case: exact pairs with one plain entry and different
    /// combine entries are one pair for plain rows, which never launch the
    /// combine, in either order; routed rows launch it, so there they tie.
    #[test]
    fn exact_pairs_differing_in_an_unlaunched_entry_are_one() {
        let other = "chop_comm_other_verify_combine_norm_bf16_v1";
        let registered = [COMBINE, other];
        for order in crate::backend::select::orders(&registered) {
            let backend = order.iter().fold(Sm120::new(), |backend, &combine| {
                backend.with_exact_all_reduce(PLAIN, combine, Invariance::one_row(32))
            });
            let prepare = |input| {
                backend.fused_all_reduce(
                    &row(4, 4096),
                    input,
                    usage(ExecutionPhase::Verify, 8),
                    0,
                    Need::rows(8).with_exact(Invariance::one_row(2)),
                )
            };
            let plain = prepare(ReduceInput::Rows).unwrap();
            assert!(
                plain
                    .info()
                    .symbols
                    .iter()
                    .any(|(symbol, _)| symbol == PLAIN)
            );
            assert_eq!(
                prepare(ReduceInput::Routed { routes: 8 }).err(),
                Some(Refused::Rule(format!(
                    "two SM120 exact all-reduce entries serve the op, {other} and {COMBINE}"
                )))
            );
        }
    }

    /// The exact pairs' key is the entry the rows launch, every field
    /// changed alone: the plain entry for plain rows and the combine for
    /// routed rows, never the claims.
    #[test]
    fn exact_keys_hold_the_entry_the_rows_launch() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let base = ExactEntries {
            plain: PLAIN.into(),
            combine: COMBINE.into(),
            invariance: Invariance::one_row(32),
        };
        let names = field_names!(ExactEntries {
            plain,
            combine,
            invariance
        });
        for (input, live_rows) in [
            (ReduceInput::Rows, None),
            (ReduceInput::Routed { routes: 8 }, Some(view(9, 4))),
        ] {
            let routed = live_rows.is_some();
            let class = |launched: bool| {
                if launched {
                    Class::Launch
                } else {
                    Class::Inert
                }
            };
            let fields: Vec<Field<ExactEntries>> = vec![
                (
                    "plain",
                    class(!routed),
                    Box::new(|e| e.plain = "chop_comm_x_norm_bf16_v1".into()),
                ),
                (
                    "combine",
                    class(routed),
                    Box::new(|e| e.combine = "chop_comm_x_combine_bf16_v1".into()),
                ),
                (
                    "invariance",
                    Class::Inert,
                    Box::new(|e| e.invariance = Invariance::one_row(16)),
                ),
            ];
            keys_follow_launches(
                &base,
                names.clone(),
                fields,
                |e| exact_launch(e, routed),
                |e| {
                    let prepared = Sm120::new()
                        .with_exact_all_reduce(e.plain.clone(), e.combine.clone(), e.invariance)
                        .fused_all_reduce(
                            &row(4, 4096),
                            input,
                            usage(ExecutionPhase::Verify, 8),
                            0,
                            Need::rows(8).with_exact(Invariance::one_row(2)),
                        )?;
                    let buffers = buffers(prepared.info(), 4, live_rows);
                    let launches = [1, 4, 8].map(|rows| {
                        prepared
                            .launches(&buffers, View::EMPTY, Counts::rows(rows))
                            .unwrap()
                    });
                    Ok((String::new(), launches))
                },
            );
        }
    }
}
