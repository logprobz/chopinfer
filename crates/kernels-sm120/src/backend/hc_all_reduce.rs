//! The hyper-connection all-reduce through a model's transports.
//!
//! The use chooses the form when the op is prepared, as the bundle's
//! routes do today:
//! - A decode or verify graph of up to `MAX_ROWS` rows takes the decode
//!   entries (`comm::hyper_connection_decode`): v3 up to `LL_MAX_ROWS`
//!   rows, v2 above, on the decode state.
//! - A graph above that, up to the rank count's wide cap, takes the wide
//!   entry on the decode state extended for it, unless the model keeps
//!   graphs off the wide entries.
//! - Everything else takes the general transport (`comm::hyper_connection`):
//!   the one-shot entry up to `ONE_SHOT_ROWS` rows, the phased scatter and
//!   gather above, on one state of the query's capacity.
//!
//! A graph launches at a bucket of prepared rows with the step's live count
//! below it on the device. A decode row count whose (row, stream) groups
//! fill no whole sending cluster has no decode plan, so it launches the
//! smallest plannable bucket above it within the capacity, with the real
//! count as its live count: the launch a graph captured at that bucket
//! makes today. A capacity that reaches no plannable bucket is refused.
//! Every form gives the same bits; the op guarantees what the model
//! declares for the entries its form launches.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::collective::HcAllReduceBuffers;
use kernel_core::contract::{DType, LaunchPlan, Refusal, SigmoidMeanHyperConnectionQuery as H};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::properties::Invariance;

use super::select::{Candidate, choose};
use super::{MAX_CAPACITY, check_plan};
use crate::comm::hyper_connection::{self as hc, Call, HcShape, HcTransport, contract, phased};
use crate::comm::hyper_connection_decode::{self as hcd, HcDecodeTransport};
use crate::comm::input;

/// A model's general transport and the guarantees it declares for its
/// entries.
pub(super) type Transport = (HcTransport, Invariance);

/// A model's decode transport and the guarantees it declares for its
/// decode (v2 and v3) and wide entries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DecodeTransport {
    pub(super) transport: HcDecodeTransport,
    pub(super) decode: Invariance,
    pub(super) wide: Invariance,
}

/// A graph form of the decode transport.
#[derive(Clone, Copy, Debug)]
struct Graph {
    transport: HcDecodeTransport,
    wide: bool,
}

/// The hyper-connection all-reduce `op` of `input` partials and `weight`
/// norm weights at `usage` that rank `rank` launches, through the model's
/// transports of that geometry; `wide` is false when the model keeps graphs
/// off the wide entries.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
    transports: &[Transport],
    decodes: &[DecodeTransport],
    wide: bool,
    op: &H,
    input: DType,
    weight: DType,
    usage: OperationUse,
    rank: u32,
    need: Need,
) -> Result<Prepared<HcAllReduceBuffers<View>>, Refused> {
    let H::AllReduceInjectNorm {
        ranks,
        streams,
        width,
        eps_bits,
        capacity,
    } = *op
    else {
        return Err(Refused::Rule(
            "a hyper-connection all-reduce takes an all-reduce query".into(),
        ));
    };
    if rank >= ranks
        || need.capacity == 0
        || need.capacity > usage.capacity
        || need.capacity > capacity.min(MAX_CAPACITY)
    {
        return Err(Refused::Rule(
            "a hyper-connection all-reduce needs a rank below its ranks and rows within its use"
                .into(),
        ));
    }
    let serves = |shape: &HcShape| {
        (shape.epilogue.width, shape.epilogue.streams) == (width, streams)
            && input_dtype(shape) == Some(input)
            && weight_dtype(shape) == weight
    };
    // A graph use takes the decode transport up to `MAX_ROWS` rows and, where
    // the model allows the wide entries, up to the rank count's wide cap;
    // only such a use resolves the decode registrations, so an unused one
    // changes nothing. Every transport serves any rows of its form, so the
    // use takes the one registered transport whatever the registration order.
    let in_graph = usage.graph && usage.phase != ExecutionPhase::Prefill;
    let decode_rows = in_graph && need.capacity <= hcd::MAX_ROWS;
    let wide_rows =
        in_graph && !decode_rows && wide && hcd::graph_takes_wide(ranks, need.capacity)?;
    // A transport launches what its own record states; the declaration of
    // the form the use takes is its claim.
    let decode = if decode_rows || wide_rows {
        let candidates = decodes
            .iter()
            .filter(|d| {
                let t = &d.transport;
                serves(&t.shape) && t.eps_bits == eps_bits && hcd::contract(t, ranks).is_ok()
            })
            .map(|d| {
                let name = hcd::symbol_v2(&d.transport, ranks).unwrap_or_default();
                let claims = if decode_rows { d.decode } else { d.wide };
                let graph = Graph {
                    transport: d.transport,
                    wide: !decode_rows,
                };
                let launches = graph_launch(graph, ranks, need.capacity);
                Candidate::new(d, name, launches, claims)
            })
            .collect::<Vec<_>>();
        (!candidates.is_empty())
            .then(|| choose("hyper-connection decode", candidates, need))
            .transpose()?
    } else {
        None
    };
    let graph = decode.map(|chosen| {
        (
            Graph {
                transport: chosen.entry.transport,
                wide: !decode_rows,
            },
            chosen.guarantees,
        )
    });
    // Each row count's graph bucket and plan, prepared up front: the
    // count itself, or the smallest plannable bucket above it.
    let (form, invariance, shape, state) = match graph {
        Some((g, invariance)) => {
            let buckets = buckets(g, need.capacity)?;
            let buffers = if g.wide {
                hcd::wide_buffers_with(g.transport.shape, ranks)?
            } else {
                hcd::buffers_with(g.transport.shape, ranks)?
            };
            (
                Form::Graph(g, buckets),
                invariance,
                g.transport.shape,
                vec![
                    buffers.inbox_bytes,
                    buffers.flags_bytes,
                    buffers.epoch_bytes,
                ],
            )
        }
        None => {
            let served = transports
                .iter()
                .filter(|(t, _)| serves(&t.shape) && contract(t, *op).is_ok())
                .map(|pair| {
                    let (t, invariance) = pair;
                    Candidate::new(
                        pair,
                        t.symbol,
                        general_launch(t, need.capacity),
                        *invariance,
                    )
                })
                .collect::<Vec<_>>();
            if served.is_empty() {
                return Err(Refused::Rule(
                    "no hyper-connection transport of the model serves the all-reduce".into(),
                ));
            }
            let chosen = choose("hyper-connection all-reduce", served, need)?;
            let (&(t, _), invariance) = (chosen.entry, chosen.guarantees);
            let buffers = hc::buffers_with(t.shape, ranks, capacity);
            (
                Form::General(t),
                invariance,
                t.shape,
                vec![
                    buffers.inbox_bytes,
                    buffers.flags_bytes,
                    buffers.epoch_bytes,
                ],
            )
        }
    };
    let mut symbols = Vec::new();
    match &form {
        Form::General(t) => {
            symbols.push(t.symbol);
            if phased(need.capacity) {
                symbols.extend([t.scatter_symbol, t.gather_symbol]);
            }
        }
        Form::Graph(g, buckets) => symbols.extend(graph_symbols(*g, buckets, ranks)?),
    }
    let mut info = OpInfo::new(
        format!("sm120.comm.hc_allreduce.v1[{}]", symbols.join(",")),
        invariance,
        symbols
            .iter()
            .map(|&symbol| (symbol.to_owned(), 0))
            .collect(),
    );
    info.state = state;
    let op = HcAllReduceOp {
        form,
        query: *op,
        shape,
        ranks,
        rank,
        capacity: need.capacity,
        weight_bytes: if weight == DType::F32 { 4 } else { 2 },
        info,
    };
    need.check(&op.info)?;
    Ok(Box::new(op))
}

/// What the general transport launches for up to `capacity` rows: its
/// cubin's one-shot entry, and the phased scatter and gather only when the
/// rows reach them.
fn general_launch(t: &HcTransport, capacity: u32) -> String {
    let HcTransport {
        shape,          // the epilogue and input the code compiles; selects the op
        eps_bits,       // the norm's epsilon; selects the op
        cubin,          // where the symbols' code lives
        symbol,         // the one-shot entry
        scatter_symbol, // the phased pair's scatter, at a phased capacity only
        gather_symbol,  // the phased pair's gather, at a phased capacity only
    } = t;
    let phased = if phased(capacity) {
        format!(" {scatter_symbol} {gather_symbol}")
    } else {
        String::new()
    };
    format!("{shape:?} {eps_bits} {cubin} {symbol}{phased}")
}

/// What a graph form launches for up to `capacity` rows at `ranks` ranks:
/// its cubin and the entries of that rank count its buckets reach.
fn graph_launch(g: Graph, ranks: u32, capacity: u32) -> String {
    let Graph {
        transport, // below
        wide,      // the form the use takes, which every candidate shares
    } = g;
    let HcDecodeTransport {
        shape,           // the epilogue and input the code compiles; selects the op
        eps_bits,        // the norm's epsilon; selects the op
        cubin,           // the decode entries' code, on the decode form
        wide_cubin,      // the wide entries' code, on the wide form
        symbols_v2: _,   // the rank count's entry, when its buckets reach it (below)
        symbols_v3: _,   // the rank count's entry, when its buckets reach it (below)
        wide_symbols: _, // the rank count's entry, on the wide form (below)
    } = transport;
    let cubin = if wide { wide_cubin } else { cubin };
    let symbols = buckets(g, capacity)
        .map_err(|refused| format!("{refused:?}"))
        .and_then(|buckets| {
            graph_symbols(g, &buckets, ranks).map_err(|refusal| format!("{refusal:?}"))
        });
    format!("{shape:?} {eps_bits} {cubin} {symbols:?}")
}

/// Each row count's graph bucket and plan, up to `capacity`: the count
/// itself, or the smallest plannable bucket above it.
fn buckets(g: Graph, capacity: u32) -> Result<Vec<(u32, LaunchPlan)>, Refused> {
    let plans = (1..=capacity)
        .map(|rows| {
            if g.wide {
                hcd::wide_plan_with(g.transport.shape, rows)
            } else {
                hcd::plan_with(g.transport.shape, rows)
            }
            .ok()
        })
        .collect::<Vec<_>>();
    (1..=capacity)
        .map(|rows| {
            let bucket = (rows..=capacity)
                .find(|&bucket| plans[bucket as usize - 1].is_some())
                .ok_or_else(|| {
                    Refused::Rule(
                        "the decode all-reduce reaches no plannable bucket within the capacity"
                            .into(),
                    )
                })?;
            let plan = plans[bucket as usize - 1].expect("a plannable bucket");
            check_plan(&plan)?;
            Ok((bucket, plan))
        })
        .collect()
}

/// The entries a graph form's buckets launch at `ranks` ranks: the wide
/// entry, or the LL entry for buckets up to `LL_MAX_ROWS` and the v2 entry
/// above.
fn graph_symbols(
    g: Graph,
    buckets: &[(u32, LaunchPlan)],
    ranks: u32,
) -> Result<Vec<&'static str>, Refusal> {
    let t = &g.transport;
    if g.wide {
        return Ok(vec![hcd::wide_symbol(t, ranks)?]);
    }
    let mut symbols = Vec::new();
    if buckets
        .iter()
        .any(|&(bucket, _)| bucket <= hcd::LL_MAX_ROWS)
    {
        symbols.push(hcd::symbol_v3(t, ranks)?);
    }
    if buckets.iter().any(|&(bucket, _)| bucket > hcd::LL_MAX_ROWS) {
        symbols.push(hcd::symbol_v2(t, ranks)?);
    }
    symbols.dedup();
    Ok(symbols)
}

fn input_dtype(shape: &HcShape) -> Option<DType> {
    if shape.input == input::BF16 {
        Some(DType::Bf16)
    } else if shape.input == input::F32 {
        Some(DType::F32)
    } else {
        None
    }
}

fn weight_dtype(shape: &HcShape) -> DType {
    if shape.epilogue.weight_pointer_type() == "const float" {
        DType::F32
    } else {
        DType::Bf16
    }
}

/// The transport and per-row-count choices of a prepared op.
enum Form {
    /// The general transport, at every row count.
    General(HcTransport),
    /// The decode or wide entry, with each row count's bucket of prepared
    /// rows and its plan.
    Graph(Graph, Vec<(u32, LaunchPlan)>),
}

struct HcAllReduceOp {
    form: Form,
    /// The general state's query: its capacity sizes that state and every
    /// general call.
    query: H,
    shape: HcShape,
    ranks: u32,
    rank: u32,
    capacity: u32,
    weight_bytes: u64,
    info: OpInfo,
}

impl PreparedOp<HcAllReduceBuffers<View>> for HcAllReduceOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &HcAllReduceBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows == 0 || counts.rows > self.capacity {
            return Err(Refusal(
                "hyper-connection all-reduce rows exceed the prepared capacity",
            ));
        }
        // A graph count launches its bucket's prepared rows, whose views
        // the launch reads; the device live count stays the step's.
        let rows = match &self.form {
            Form::General(_) => counts.rows,
            Form::Graph(_, buckets) => buckets[counts.rows as usize - 1].0,
        };
        if buffers.ranks.len() != self.ranks as usize {
            return Err(Refusal(
                "a hyper-connection all-reduce takes the state of every rank",
            ));
        }
        let mut peers = Vec::with_capacity(buffers.ranks.len());
        for rank in &buffers.ranks {
            let [inbox, flags, epoch] = rank.state[..] else {
                return Err(Refusal(
                    "a hyper-connection all-reduce rank lacks its state regions",
                ));
            };
            for (view, &bytes) in [inbox, flags, epoch].iter().zip(&self.info.state) {
                view.covers(bytes)?;
            }
            peers.push([inbox, flags, epoch]);
        }
        let epilogue = self.shape.epilogue;
        let (width, streams) = (u64::from(epilogue.width), u64::from(epilogue.streams));
        let rows64 = u64::from(rows);
        let input_bytes = u64::from(self.shape.input.element_bytes);
        buffers.input.covers(rows64 * width * input_bytes)?;
        for streams_view in [
            buffers.residual.input,
            buffers.residual.output,
            buffers.normed,
        ] {
            streams_view.covers(rows64 * streams * width * 2)?;
        }
        buffers.logits.covers(rows64 * streams * 2)?;
        buffers.weight.covers(streams * width * self.weight_bytes)?;
        buffers.live_rows.covers(4)?;
        let inboxes = peers.iter().map(|p| p[0].handle).collect();
        let flags = peers.iter().map(|p| p[1].handle).collect();
        let epoch = peers[self.rank as usize][2].handle;
        let launch = |symbol: &str, plan, arguments| Launch {
            symbol: symbol.into(),
            rows,
            plan,
            arguments,
            collective: true,
            persistent: false,
        };
        let (graph, plan) = match &self.form {
            Form::General(t) => {
                let H::AllReduceInjectNorm { capacity, .. } = self.query else {
                    unreachable!("prepared from an all-reduce query")
                };
                let call = Call {
                    input: buffers.input.handle,
                    residual: buffers.residual.input.handle,
                    logits: buffers.logits.handle,
                    weight: buffers.weight.handle,
                    residual_out: buffers.residual.output.handle,
                    normed: buffers.normed.handle,
                    inboxes,
                    flags,
                    epoch,
                    live_rows: buffers.live_rows.handle,
                    capacity,
                    rank: self.rank,
                };
                return Ok(hc::launches(t, self.query, rows, call)?
                    .into_iter()
                    .map(|l| launch(l.symbol, l.plan, l.arguments))
                    .collect());
            }
            Form::Graph(graph, buckets) => (graph, buckets[counts.rows as usize - 1].1),
        };
        let t = &graph.transport;
        let call = hcd::Call {
            input: buffers.input.handle,
            residual: buffers.residual.input.handle,
            logits: buffers.logits.handle,
            weight: buffers.weight.handle,
            residual_out: buffers.residual.output.handle,
            normed: buffers.normed.handle,
            inboxes,
            flags,
            epochs: epoch,
            live_rows: buffers.live_rows.handle,
            rows,
            rank: self.rank,
        };
        Ok(vec![if graph.wide {
            launch(
                hcd::wide_symbol(t, self.ranks)?,
                plan,
                hcd::wide_arguments(call)?,
            )
        } else {
            let symbol = if rows <= hcd::LL_MAX_ROWS {
                hcd::symbol_v3(t, self.ranks)?
            } else {
                hcd::symbol_v2(t, self.ranks)?
            };
            launch(symbol, plan, hcd::arguments(call)?)
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};
    use crate::comm::epilogue::{Epilogue, HC_INJECT_NORM_EPS_BITS};
    use crate::comm::hyper_connection::ONE_SHOT_ROWS;
    use kernel_core::backend::Backend;
    use kernel_core::collective::RankState;
    use kernel_core::contract::Validate;
    use kernel_core::rmsnorm::FusedResidual;

    const EPILOGUE: Epilogue = Epilogue {
        header: "comm/epilogue/hc_inject_norm.cuh",
        template: "chop::comm::epilogue::HcInjectNorm",
        width: 2560,
        streams: 4,
    };

    fn transport(weight: DType) -> HcTransport {
        let f32 = weight == DType::F32;
        HcTransport {
            shape: HcShape {
                epilogue: EPILOGUE.with_norm_weight_dtype(weight).unwrap(),
                input: input::BF16,
            },
            eps_bits: HC_INJECT_NORM_EPS_BITS,
            cubin: "comm_hc_test_sm120",
            symbol: if f32 {
                "chop_comm_hc_f32_v1"
            } else {
                "chop_comm_hc_v1"
            },
            scatter_symbol: if f32 {
                "chop_comm_hc_scatter_f32_v1"
            } else {
                "chop_comm_hc_scatter_v1"
            },
            gather_symbol: if f32 {
                "chop_comm_hc_gather_f32_v1"
            } else {
                "chop_comm_hc_gather_v1"
            },
        }
    }

    fn prefill() -> OperationUse {
        OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        }
    }

    fn graph(capacity: u32) -> OperationUse {
        OperationUse {
            phase: ExecutionPhase::Decode,
            capacity,
            graph: true,
        }
    }

    fn decode_transport(weight: DType) -> HcDecodeTransport {
        let names = |stem: &'static str| -> [&'static str; 3] {
            match (stem, weight) {
                ("v2", DType::Bf16) => [
                    "chop_comm_hcd_r2_v2",
                    "chop_comm_hcd_r4_v2",
                    "chop_comm_hcd_r8_v2",
                ],
                ("v3", DType::Bf16) => [
                    "chop_comm_hcd_r2_v3",
                    "chop_comm_hcd_r4_v3",
                    "chop_comm_hcd_r8_v3",
                ],
                ("wide", DType::Bf16) => [
                    "chop_comm_hcw_r2_v1",
                    "chop_comm_hcw_r4_v1",
                    "chop_comm_hcw_r8_v1",
                ],
                ("v2", _) => [
                    "chop_comm_hcd_f32_r2_v2",
                    "chop_comm_hcd_f32_r4_v2",
                    "chop_comm_hcd_f32_r8_v2",
                ],
                ("v3", _) => [
                    "chop_comm_hcd_f32_r2_v3",
                    "chop_comm_hcd_f32_r4_v3",
                    "chop_comm_hcd_f32_r8_v3",
                ],
                _ => [
                    "chop_comm_hcw_f32_r2_v1",
                    "chop_comm_hcw_f32_r4_v1",
                    "chop_comm_hcw_f32_r8_v1",
                ],
            }
        };
        HcDecodeTransport {
            shape: transport(weight).shape,
            eps_bits: HC_INJECT_NORM_EPS_BITS,
            cubin: "comm_hcd_test_sm120",
            wide_cubin: "comm_hcw_test_sm120",
            symbols_v2: names("v2"),
            symbols_v3: names("v3"),
            wide_symbols: names("wide"),
        }
    }

    /// The general transports, and the decode ones with one-row bits up to
    /// four rows for the decode entries and none for the wide ones.
    fn graph_backend() -> Sm120 {
        let decode = Invariance::one_row(4);
        backend()
            .with_hc_decode_all_reduce(decode_transport(DType::Bf16), decode, Invariance::NONE)
            .with_hc_decode_all_reduce(decode_transport(DType::F32), decode, Invariance::NONE)
    }

    fn backend() -> Sm120 {
        Sm120::new()
            .with_hc_all_reduce(transport(DType::Bf16), Invariance::PREFILL)
            .with_hc_all_reduce(transport(DType::F32), Invariance::PREFILL)
    }

    /// Rank r's state regions are 100 * (r + 1) + i, each of its declared
    /// size; the operands are 1 to 7.
    fn buffers(info: &OpInfo, ranks: u32) -> HcAllReduceBuffers<View> {
        HcAllReduceBuffers {
            input: view(1, u64::MAX),
            residual: FusedResidual {
                input: view(2, u64::MAX),
                output: view(5, u64::MAX),
            },
            logits: view(3, u64::MAX),
            weight: view(4, u64::MAX),
            normed: view(6, u64::MAX),
            live_rows: view(7, 4),
            ranks: (0..ranks)
                .map(|r| RankState {
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

    #[test]
    fn hc_all_reduce_launches_the_transport_launches() {
        for (ranks, weight) in [(2, DType::Bf16), (4, DType::F32), (8, DType::Bf16)] {
            let t = transport(weight);
            let query = t.query(ranks);
            let rank = ranks - 1;
            let prepared = backend()
                .hc_all_reduce(
                    &query,
                    DType::Bf16,
                    weight,
                    prefill(),
                    rank,
                    Need::rows(512),
                )
                .unwrap();
            let info = prepared.info().clone();
            let state = hc::buffers_with(t.shape, ranks, 8192);
            assert_eq!(
                info.state,
                [state.inbox_bytes, state.flags_bytes, state.epoch_bytes]
            );
            assert_eq!(info.guarantees, Invariance::PREFILL);
            assert_eq!(info.symbols.len(), 3);
            let buffers = buffers(&info, ranks);
            for rows in [1, 2, 31, 32, 33, 100, 512] {
                let launches = prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                let call = Call {
                    input: 1,
                    residual: 2,
                    logits: 3,
                    weight: 4,
                    residual_out: 5,
                    normed: 6,
                    inboxes: (0..u64::from(ranks)).map(|r| 100 * (r + 1)).collect(),
                    flags: (0..u64::from(ranks)).map(|r| 100 * (r + 1) + 1).collect(),
                    epoch: 100 * u64::from(ranks) + 2,
                    live_rows: 7,
                    capacity: 8192,
                    rank,
                };
                let expected = hc::launches(&t, query, rows, call).unwrap();
                assert_eq!(launches.len(), expected.len());
                for (launch, expected) in launches.iter().zip(&expected) {
                    assert_eq!(launch.symbol, expected.symbol);
                    assert_eq!(launch.plan, expected.plan);
                    assert_eq!(launch.arguments, expected.arguments);
                    assert!(launch.collective && !launch.persistent && launch.rows == rows);
                }
                assert_eq!(launches.len(), if rows > ONE_SHOT_ROWS { 2 } else { 1 });
            }
            // Short operands and state, every rank's state, the capacity.
            let mut short = buffers.clone();
            short.normed.bytes = 32 * 4 * 2560 * 2 - 1;
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(32))
                    .is_err()
            );
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(31))
                    .is_ok()
            );
            let mut short = buffers.clone();
            short.weight.bytes = 4 * 2560 * if weight == DType::F32 { 4 } else { 2 } - 1;
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(1))
                    .is_err()
            );
            let mut short = buffers.clone();
            short.ranks[0].state[0].bytes -= 1;
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(1))
                    .is_err()
            );
            let mut missing = buffers.clone();
            missing.ranks.pop();
            assert!(
                prepared
                    .launches(&missing, View::EMPTY, Counts::rows(1))
                    .is_err()
            );
            assert!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(513))
                    .is_err()
            );
        }
    }

    #[test]
    fn hc_all_reduce_refuses_what_no_transport_serves() {
        let t = transport(DType::Bf16);
        let query = t.query(2);
        // A decode-only capacity launches only the one-shot entry.
        let decode = backend()
            .hc_all_reduce(
                &query,
                DType::Bf16,
                DType::Bf16,
                prefill(),
                0,
                Need::rows(32),
            )
            .unwrap();
        assert_eq!(decode.info().symbols.len(), 1);
        // Six ranks are a valid contract SM120's entries do not take.
        let six = t.query(6);
        assert_eq!(six.validate(), Ok(()));
        assert!(matches!(
            backend().hc_all_reduce(&six, DType::Bf16, DType::Bf16, prefill(), 0, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
        // Another width, input dtype or weight dtype than the model's
        // transports; no transport at all.
        let H::AllReduceInjectNorm {
            ranks,
            streams,
            eps_bits,
            capacity,
            ..
        } = query
        else {
            unreachable!()
        };
        let wide = H::AllReduceInjectNorm {
            ranks,
            streams,
            width: 4096,
            eps_bits,
            capacity,
        };
        assert!(
            backend()
                .hc_all_reduce(&wide, DType::Bf16, DType::Bf16, prefill(), 0, Need::rows(4))
                .is_err()
        );
        assert!(
            backend()
                .hc_all_reduce(&query, DType::F32, DType::Bf16, prefill(), 0, Need::rows(4))
                .is_err()
        );
        assert!(
            Sm120::new()
                .with_hc_all_reduce(t, Invariance::PREFILL)
                .hc_all_reduce(&query, DType::Bf16, DType::F32, prefill(), 0, Need::rows(4))
                .is_err()
        );
        assert_eq!(
            Sm120::new()
                .hc_all_reduce(
                    &query,
                    DType::Bf16,
                    DType::Bf16,
                    prefill(),
                    0,
                    Need::rows(4)
                )
                .err()
                .map(|refused| matches!(refused, Refused::Rule(_))),
            Some(true)
        );
        // Ranks, rows past the state's capacity, unmet exactness.
        assert!(
            backend()
                .hc_all_reduce(
                    &query,
                    DType::Bf16,
                    DType::Bf16,
                    prefill(),
                    2,
                    Need::rows(4)
                )
                .is_err()
        );
        assert!(
            backend()
                .hc_all_reduce(
                    &query,
                    DType::Bf16,
                    DType::Bf16,
                    prefill(),
                    0,
                    Need::rows(8193)
                )
                .is_err()
        );
        let undeclared = Sm120::new().with_hc_all_reduce(t, Invariance::NONE);
        assert!(matches!(
            undeclared.hc_all_reduce(
                &query,
                DType::Bf16,
                DType::Bf16,
                prefill(),
                0,
                Need::rows(64).with_exact(Invariance::PREFILL)
            ),
            Err(Refused::Inexact { .. })
        ));
    }

    /// Today's decode or wide launch of a graph of `rows` prepared rows on
    /// the decode state: v3 at low-latency rows, v2 above, or the wide entry.
    fn legacy(t: &HcDecodeTransport, wide: bool, ranks: u32, rank: u32, rows: u32) -> Launch {
        let call = hcd::Call {
            input: 1,
            residual: 2,
            logits: 3,
            weight: 4,
            residual_out: 5,
            normed: 6,
            inboxes: (0..u64::from(ranks)).map(|r| 100 * (r + 1)).collect(),
            flags: (0..u64::from(ranks)).map(|r| 100 * (r + 1) + 1).collect(),
            epochs: 100 * u64::from(rank + 1) + 2,
            live_rows: 7,
            rows,
            rank,
        };
        let (symbol, plan, arguments) = if wide {
            (
                hcd::wide_symbol(t, ranks).unwrap(),
                hcd::wide_plan_with(t.shape, rows).unwrap(),
                hcd::wide_arguments(call).unwrap(),
            )
        } else {
            (
                if rows <= hcd::LL_MAX_ROWS {
                    hcd::symbol_v3(t, ranks).unwrap()
                } else {
                    hcd::symbol_v2(t, ranks).unwrap()
                },
                hcd::plan_with(t.shape, rows).unwrap(),
                hcd::arguments(call).unwrap(),
            )
        };
        Launch {
            symbol: symbol.into(),
            rows,
            plan,
            arguments,
            collective: true,
            persistent: false,
        }
    }

    /// Today's general launches of `rows` rows.
    fn legacy_general(t: &HcTransport, ranks: u32, rank: u32, rows: u32) -> Vec<Launch> {
        let call = Call {
            input: 1,
            residual: 2,
            logits: 3,
            weight: 4,
            residual_out: 5,
            normed: 6,
            inboxes: (0..u64::from(ranks)).map(|r| 100 * (r + 1)).collect(),
            flags: (0..u64::from(ranks)).map(|r| 100 * (r + 1) + 1).collect(),
            epoch: 100 * u64::from(rank + 1) + 2,
            live_rows: 7,
            capacity: 8192,
            rank,
        };
        hc::launches(t, t.query(ranks), rows, call)
            .unwrap()
            .into_iter()
            .map(|l| Launch {
                symbol: l.symbol.into(),
                rows,
                plan: l.plan,
                arguments: l.arguments,
                collective: true,
                persistent: false,
            })
            .collect()
    }

    #[test]
    fn graph_uses_take_the_decode_and_wide_entries_at_every_row_count() {
        for (ranks, weight, wide_cap) in [
            (2, DType::Bf16, 256),
            (4, DType::F32, 128),
            (8, DType::Bf16, 128),
        ] {
            let t = decode_transport(weight);
            let general = transport(weight);
            let query = general.query(ranks);
            let rank = ranks - 1;
            let prepare = |backend: &Sm120, usage, bucket| {
                backend
                    .hc_all_reduce(&query, DType::Bf16, weight, usage, rank, Need::rows(bucket))
                    .unwrap()
            };
            // Decode buckets: v3 at low-latency rows, v2 above. A count
            // whose groups fill no whole cluster launches the smallest
            // plannable bucket above it, with its live count on the device.
            for bucket in [1, 2, 4, 8, 16, 32] {
                let prepared = prepare(&graph_backend(), graph(bucket), bucket);
                let info = prepared.info().clone();
                let decode = hcd::buffers_with(t.shape, ranks).unwrap();
                assert_eq!(
                    info.state,
                    [decode.inbox_bytes, decode.flags_bytes, decode.epoch_bytes]
                );
                assert_eq!(info.guarantees, Invariance::one_row(4));
                let buffers = buffers(&info, ranks);
                for rows in 1..=bucket {
                    let padded = (rows..=bucket)
                        .find(|&padded| hcd::plan_with(t.shape, padded).is_ok())
                        .unwrap();
                    let launches = prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap();
                    assert_eq!(launches, [legacy(&t, false, ranks, rank, padded)]);
                    assert!(
                        info.symbols
                            .iter()
                            .any(|(own, _)| *own == launches[0].symbol)
                    );
                }
                if bucket >= 4 {
                    // Three live rows run today's bucket-4 launch.
                    let launches = prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(3))
                        .unwrap();
                    assert_eq!(launches, [legacy(&t, false, ranks, rank, 4)]);
                    assert_eq!(launches[0].rows, 4);
                }
            }
            // Wide buckets up to the rank count's cap, on the extended state:
            // every count plans.
            for bucket in [33, 64, wide_cap] {
                let prepared = prepare(&graph_backend(), graph(bucket), bucket);
                let info = prepared.info().clone();
                let state = hcd::wide_buffers_with(t.shape, ranks).unwrap();
                assert_eq!(
                    info.state,
                    [state.inbox_bytes, state.flags_bytes, state.epoch_bytes]
                );
                assert_eq!(info.guarantees, Invariance::NONE);
                let buffers = buffers(&info, ranks);
                for rows in 1..=bucket {
                    let launches = prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap();
                    assert_eq!(launches, [legacy(&t, true, ranks, rank, rows)]);
                }
            }
            // Above the cap, without the wide entries and in prefill: the
            // general transport at every count.
            let mut generals = vec![
                (graph_backend().without_wide_collectives(), graph(64), 64),
                (graph_backend(), prefill(), 64),
            ];
            if wide_cap < 256 {
                generals.push((graph_backend(), graph(256), 256));
            }
            for (backend, usage, bucket) in generals {
                let prepared = prepare(&backend, usage, bucket);
                let info = prepared.info().clone();
                assert_eq!(info.guarantees, Invariance::PREFILL);
                let buffers = buffers(&info, ranks);
                for rows in 1..=bucket {
                    let launches = prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap();
                    assert_eq!(launches, legacy_general(&general, ranks, rank, rows));
                }
            }
        }
    }

    #[test]
    fn graph_uses_refuse_what_their_transports_do_not_serve() {
        let query = transport(DType::Bf16).query(2);
        let prepare = |backend: Sm120, usage, need| {
            backend.hc_all_reduce(&query, DType::Bf16, DType::Bf16, usage, 0, need)
        };
        let prepared = prepare(graph_backend(), graph(32), Need::rows(32)).unwrap();
        let info = prepared.info().clone();
        let buffers = buffers(&info, 2);
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(33))
                .is_err()
        );
        let mut short = buffers.clone();
        short.ranks[1].state[2].bytes -= 1;
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        let mut missing = buffers.clone();
        missing.ranks[0].state.pop();
        assert!(
            prepared
                .launches(&missing, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        // A padded count reads its bucket's rows.
        let mut short = buffers.clone();
        short.input.bytes = 4 * 2560 * 2 - 1;
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(3))
                .is_err()
        );
        short.input.bytes += 1;
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(3))
                .is_ok()
        );
        // A capacity whose own count reaches no plannable bucket.
        assert!(matches!(
            prepare(graph_backend(), graph(3), Need::rows(3)),
            Err(Refused::Rule(_))
        ));
        // The decode entries alone serve a graph; no general transport.
        let decode_only = || {
            Sm120::new().with_hc_decode_all_reduce(
                decode_transport(DType::Bf16),
                Invariance::one_row(4),
                Invariance::NONE,
            )
        };
        assert!(prepare(decode_only(), graph(4), Need::rows(4)).is_ok());
        assert!(prepare(decode_only(), prefill(), Need::rows(4)).is_err());
        // Capacities past the use's; exactness past the entries' claims.
        assert!(prepare(graph_backend(), graph(16), Need::rows(32)).is_err());
        assert!(
            prepare(
                graph_backend(),
                graph(4),
                Need::rows(4).with_exact(Invariance::one_row(4))
            )
            .is_ok()
        );
        assert!(matches!(
            prepare(
                graph_backend(),
                graph(8),
                Need::rows(8).with_exact(Invariance::one_row(8))
            ),
            Err(Refused::Inexact { .. })
        ));
        // Without a decode transport, graphs take the general one.
        let general = prepare(backend(), graph(8), Need::rows(8)).unwrap();
        assert_eq!(general.info().symbols, [("chop_comm_hc_v1".to_owned(), 0)]);
    }

    #[test]
    fn transports_resolve_in_any_registration_order() {
        type Register = fn(Sm120) -> Sm120;
        let registrations: [Register; 4] = [
            |b| b.with_hc_all_reduce(transport(DType::Bf16), Invariance::PREFILL),
            |b| b.with_hc_all_reduce(transport(DType::F32), Invariance::PREFILL),
            |b| {
                b.with_hc_decode_all_reduce(
                    decode_transport(DType::Bf16),
                    Invariance::one_row(4),
                    Invariance::NONE,
                )
            },
            |b| {
                b.with_hc_decode_all_reduce(
                    decode_transport(DType::F32),
                    Invariance::one_row(4),
                    Invariance::NONE,
                )
            },
        ];
        // Every order of the four registrations (lexicographic successors).
        let mut orders = vec![vec![0, 1, 2, 3]];
        for _ in 0..23 {
            let mut next = orders.last().unwrap().clone();
            let i = (0..3).rev().find(|&i| next[i] < next[i + 1]).unwrap();
            let j = (i + 1..4).rev().find(|&j| next[i] < next[j]).unwrap();
            next.swap(i, j);
            next[i + 1..].reverse();
            orders.push(next);
        }
        let launched = |backend: &Sm120| {
            let mut all = Vec::new();
            for weight in [DType::Bf16, DType::F32] {
                let query = transport(weight).query(4);
                for (usage, capacity) in [(graph(8), 8), (graph(64), 64), (prefill(), 64)] {
                    let prepared = backend
                        .hc_all_reduce(&query, DType::Bf16, weight, usage, 1, Need::rows(capacity))
                        .unwrap();
                    let info = prepared.info().clone();
                    let buffers = buffers(&info, 4);
                    for rows in [1, 3, capacity] {
                        all.push((
                            info.clone(),
                            prepared
                                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                                .unwrap(),
                        ));
                    }
                }
            }
            all
        };
        let expected = launched(&graph_backend());
        for order in &orders {
            // The first registration doubled: identical entries count as one.
            let backend = order
                .iter()
                .chain(&order[..1])
                .fold(Sm120::new(), |backend, &i| registrations[i](backend));
            assert_eq!(launched(&backend), expected, "order {order:?}");
        }
        // Two different transports of one geometry tie in either order.
        let other = HcTransport {
            symbol: "chop_comm_hc_other_v1",
            ..transport(DType::Bf16)
        };
        let query = transport(DType::Bf16).query(2);
        for backend in [
            graph_backend().with_hc_all_reduce(other, Invariance::PREFILL),
            Sm120::new()
                .with_hc_all_reduce(other, Invariance::PREFILL)
                .with_hc_all_reduce(transport(DType::Bf16), Invariance::PREFILL),
        ] {
            assert_eq!(
                backend
                    .hc_all_reduce(
                        &query,
                        DType::Bf16,
                        DType::Bf16,
                        prefill(),
                        0,
                        Need::rows(8)
                    )
                    .err(),
                Some(Refused::Rule(
                    "two SM120 hyper-connection all-reduce entries serve the op, \
                     chop_comm_hc_other_v1 and chop_comm_hc_v1"
                        .into()
                ))
            );
        }
        // A decode transport registered again with other declarations
        // launches the same: one transport, claiming what both declare.
        let prepare = |backend: Sm120, need| {
            backend
                .hc_all_reduce(&query, DType::Bf16, DType::Bf16, graph(8), 0, need)
                .map(|op| op.info().clone())
        };
        let alone = prepare(graph_backend(), Need::rows(8)).unwrap();
        let unclaimed = || {
            graph_backend().with_hc_decode_all_reduce(
                decode_transport(DType::Bf16),
                Invariance::NONE,
                Invariance::NONE,
            )
        };
        let both = prepare(unclaimed(), Need::rows(8)).unwrap();
        assert_eq!(both.symbols, alone.symbols);
        assert_eq!(both.guarantees, alone.guarantees.meet(Invariance::NONE));
        // A need only the claiming registration keeps takes it.
        let exact = Need::rows(8).with_exact(Invariance::one_row(4));
        assert_eq!(prepare(unclaimed(), exact), prepare(graph_backend(), exact));
    }

    #[test]
    fn unused_decode_registrations_change_nothing() {
        let conflicting = || {
            let other = HcDecodeTransport {
                symbols_v2: [
                    "chop_comm_x_r2_v2",
                    "chop_comm_x_r4_v2",
                    "chop_comm_x_r8_v2",
                ],
                ..decode_transport(DType::Bf16)
            };
            graph_backend().with_hc_decode_all_reduce(
                other,
                Invariance::one_row(4),
                Invariance::NONE,
            )
        };
        let query = transport(DType::Bf16).query(4);
        // Prefill and graphs past the wide cap take the general transport
        // alone: conflicting decode registrations change nothing.
        for (usage, capacity) in [(prefill(), 64), (graph(256), 256)] {
            let prepare = |backend: &Sm120| {
                backend
                    .hc_all_reduce(
                        &query,
                        DType::Bf16,
                        DType::Bf16,
                        usage,
                        3,
                        Need::rows(capacity),
                    )
                    .unwrap()
            };
            let alone = prepare(&backend());
            let prepared = prepare(&conflicting());
            assert_eq!(prepared.info(), alone.info());
            let buffers = buffers(alone.info(), 4);
            for rows in [1, 33, capacity] {
                assert_eq!(
                    prepared.launches(&buffers, View::EMPTY, Counts::rows(rows)),
                    alone.launches(&buffers, View::EMPTY, Counts::rows(rows))
                );
            }
        }
        // A graph use that launches the decode entries resolves them; the
        // wide entries launch no v2 entry, so the two transports are one.
        assert!(matches!(
            conflicting().hc_all_reduce(
                &query,
                DType::Bf16,
                DType::Bf16,
                graph(8),
                3,
                Need::rows(8)
            ),
            Err(Refused::Rule(_))
        ));
        let wide = |backend: Sm120| {
            backend
                .hc_all_reduce(
                    &query,
                    DType::Bf16,
                    DType::Bf16,
                    graph(64),
                    3,
                    Need::rows(64),
                )
                .map(|op| op.info().clone())
        };
        assert_eq!(wide(conflicting()), wide(graph_backend()));
        // Without the wide entries, a wide-sized graph takes the general
        // transport and leaves the decode registrations unused.
        assert!(
            conflicting()
                .without_wide_collectives()
                .hc_all_reduce(
                    &query,
                    DType::Bf16,
                    DType::Bf16,
                    graph(64),
                    3,
                    Need::rows(64)
                )
                .is_ok()
        );
    }

    /// The auditor's case: general transports differing only in the
    /// scatter entry are one transport for a one-row prefill, which launches
    /// the one-shot entry alone, in either order; a phased capacity
    /// launches the scatter, so there they tie.
    #[test]
    fn transports_differing_in_unlaunched_entries_are_one() {
        let a = transport(DType::Bf16);
        let b = HcTransport {
            scatter_symbol: "chop_comm_hc_scatter_other_v1",
            ..a
        };
        let query = a.query(4);
        let usage = |capacity| OperationUse {
            capacity,
            ..prefill()
        };
        for order in crate::backend::select::orders(&[a, b]) {
            let backend = order.iter().fold(Sm120::new(), |backend, &t| {
                backend.with_hc_all_reduce(t, Invariance::NONE)
            });
            let prepare = |capacity| {
                backend.hc_all_reduce(
                    &query,
                    DType::Bf16,
                    DType::Bf16,
                    usage(capacity),
                    0,
                    Need::rows(capacity),
                )
            };
            assert_eq!(
                prepare(1).unwrap().info().symbols,
                [(a.symbol.to_owned(), 0)]
            );
            assert!(matches!(
                prepare(ONE_SHOT_ROWS + 1).err(),
                Some(Refused::Rule(_))
            ));
        }
    }

    /// The general and decode keys hold what each form launches, every
    /// field changed alone, at a one-shot and a phased capacity, and at
    /// decode, LL-only and wide graph capacities: the cubin and the entries
    /// of the rank count the rows reach. The shape and epsilon select the
    /// op.
    #[test]
    fn launch_keys_hold_what_each_form_launches() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let query = transport(DType::Bf16).query(4);
        let launches = |prepared: Prepared<HcAllReduceBuffers<View>>, rows: &[u32]| {
            let buffers = buffers(prepared.info(), 4);
            rows.iter()
                .map(|&rows| {
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let names = field_names!(HcTransport {
            shape,
            eps_bits,
            cubin,
            symbol,
            scatter_symbol,
            gather_symbol
        });
        let wider = transport(DType::F32).shape;
        for capacity in [1, ONE_SHOT_ROWS + 1] {
            let phased = if capacity > ONE_SHOT_ROWS {
                Class::Launch
            } else {
                Class::Inert
            };
            let fields: Vec<Field<HcTransport>> = vec![
                ("shape", Class::Selects, Box::new(move |t| t.shape = wider)),
                ("eps_bits", Class::Selects, Box::new(|t| t.eps_bits += 1)),
                (
                    "cubin",
                    Class::Launch,
                    Box::new(|t| t.cubin = "comm_hc_other_sm120"),
                ),
                (
                    "symbol",
                    Class::Launch,
                    Box::new(|t| t.symbol = "chop_comm_hc_other_v1"),
                ),
                (
                    "scatter_symbol",
                    phased,
                    Box::new(|t| t.scatter_symbol = "chop_comm_hc_scatter_other_v1"),
                ),
                (
                    "gather_symbol",
                    phased,
                    Box::new(|t| t.gather_symbol = "chop_comm_hc_gather_other_v1"),
                ),
            ];
            let usage = OperationUse {
                capacity,
                ..prefill()
            };
            keys_follow_launches(
                &transport(DType::Bf16),
                names.clone(),
                fields,
                |t| general_launch(t, capacity),
                |t| {
                    let prepared = Sm120::new()
                        .with_hc_all_reduce(*t, Invariance::NONE)
                        .hc_all_reduce(
                            &query,
                            DType::Bf16,
                            DType::Bf16,
                            usage,
                            0,
                            Need::rows(capacity),
                        )?;
                    Ok((t.cubin.to_owned(), launches(prepared, &[1, capacity])))
                },
            );
        }
        // A graph form's fields, its transport's field by field.
        let names: Vec<&'static str> = field_names!(Graph { transport, wide })
            .into_iter()
            .flat_map(|name| {
                if name == "transport" {
                    field_names!(HcDecodeTransport {
                        shape,
                        eps_bits,
                        cubin,
                        wide_cubin,
                        symbols_v2,
                        symbols_v3,
                        wide_symbols
                    })
                } else {
                    vec![name]
                }
            })
            .collect();
        let ranks = 4;
        // Index 1 is the four-rank entry.
        let renamed = |names: [&'static str; 3], name: &'static str| {
            let mut names = names;
            names[1] = name;
            names
        };
        for capacity in [2, 8, 64] {
            let wide = capacity > hcd::MAX_ROWS;
            let v2 = capacity > hcd::LL_MAX_ROWS && !wide;
            let class = |launched: bool| {
                if launched {
                    Class::Launch
                } else {
                    Class::Inert
                }
            };
            // The use takes the form: a graph of the other form is another
            // use's, so it serves no op of this one.
            let fields: Vec<Field<Graph>> = vec![
                (
                    "shape",
                    Class::Selects,
                    Box::new(move |g| g.transport.shape = wider),
                ),
                (
                    "eps_bits",
                    Class::Selects,
                    Box::new(|g| g.transport.eps_bits += 1),
                ),
                (
                    "cubin",
                    class(!wide),
                    Box::new(|g| g.transport.cubin = "comm_hcd_other_sm120"),
                ),
                (
                    "wide_cubin",
                    class(wide),
                    Box::new(|g| g.transport.wide_cubin = "comm_hcw_other_sm120"),
                ),
                (
                    "symbols_v2",
                    class(v2),
                    Box::new(move |g| {
                        g.transport.symbols_v2 = renamed(g.transport.symbols_v2, "x_v2")
                    }),
                ),
                (
                    "symbols_v3",
                    class(!wide),
                    Box::new(move |g| {
                        g.transport.symbols_v3 = renamed(g.transport.symbols_v3, "x_v3")
                    }),
                ),
                (
                    "wide_symbols",
                    class(wide),
                    Box::new(move |g| {
                        g.transport.wide_symbols = renamed(g.transport.wide_symbols, "x_w")
                    }),
                ),
                ("wide", Class::Selects, Box::new(|g| g.wide = !g.wide)),
            ];
            let base = Graph {
                transport: decode_transport(DType::Bf16),
                wide,
            };
            keys_follow_launches(
                &base,
                names.clone(),
                fields,
                |g| graph_launch(*g, ranks, capacity),
                |g| {
                    if g.wide != wide {
                        return Err(Refused::Rule("the use takes the other form".into()));
                    }
                    let t = g.transport;
                    let prepared = Sm120::new()
                        .with_hc_decode_all_reduce(t, Invariance::NONE, Invariance::NONE)
                        .hc_all_reduce(
                            &query,
                            DType::Bf16,
                            DType::Bf16,
                            graph(capacity),
                            0,
                            Need::rows(capacity),
                        )?;
                    let cubin = if wide { t.wide_cubin } else { t.cubin };
                    Ok((cubin.to_owned(), launches(prepared, &[1, 2, capacity])))
                },
            );
        }
    }
}
