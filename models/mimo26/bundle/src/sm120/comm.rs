//! MiMo's collectives on the kernel base's comm entries: the hidden
//! all-reduce routes and their entries, the spinning-entry residency list
//! and the logits row gather (moved from `launch::comm`).

use crate::topology::Tp;
#[cfg(test)]
use kernel_core::contract::ArgumentList;
use kernel_core::contract::{CommQuery, DType, LaunchPlan, Refusal, RmsNormEpilogue};
#[cfg(test)]
pub use kernels_sm120::comm::OneShotCall;
/// The comm entries' symbols and state sizes MiMo's tests name.
#[cfg(test)]
pub use kernels_sm120::comm::{
    DECODE_COMBINE_SYMBOL, DECODE_SYMBOL_V2, EXCHANGE_ROWS_EPOCH_BYTES, EXCHANGE_ROWS_FLAG_BYTES,
    GATHER_ROWS_EPOCH_BYTES, GATHER_ROWS_FLAG_BYTES, VERIFY_COMBINE_SYMBOL, VERIFY_SYMBOL,
    WIDE_COMBINE_SYMBOL, WIDE_SYMBOL, fused_state_buffers, phased_buffers, push_buffers,
};

use crate::contract::PREFILL_TOKENS;

/// A rank count the comm kernels take, with `rank` one of its ranks.
#[cfg(test)]
fn valid_ranks(rank: u32, ranks: u32) -> bool {
    (2..=bundle_runtime::geometry::MAX_RANKS).contains(&ranks) && rank < ranks
}

#[cfg(test)]
pub type Stage = (String, LaunchPlan, ArgumentList);

#[cfg(test)]
pub struct ResidualBuffers {
    pub input: u64,
    pub output: u64,
}

/// The plain and combine entries of a hidden all-reduce of `rows` rows
/// (1 to 32) over `ranks`, which take the same arguments and plans: the
/// decode entries up to `DECODE_MAX_ROWS` rows, v2 and combine v1 above
/// (the same protocols and bits).
pub fn push_symbols(ranks: u32, rows: u32) -> (&'static str, &'static str) {
    if rows <= kernels_sm120::comm::DECODE_MAX_ROWS
        && kernels_sm120::comm::push_protocol(ranks, rows).is_ok()
    {
        return (
            kernels_sm120::comm::DECODE_SYMBOL_V2,
            kernels_sm120::comm::DECODE_COMBINE_SYMBOL,
        );
    }
    (
        "chop_comm_allreduce_oneshot_norm_bf16_v2",
        "chop_comm_allreduce_combine_norm_bf16_v1",
    )
}

/// The fused hidden all-reduce query of `elements` values over `ranks`.
fn hidden_query(tp: Tp, ranks: u32, elements: u64, epsilon: f32) -> CommQuery {
    CommQuery::AllReduce {
        ranks,
        elements,
        dtype: DType::Bf16,
        fused_residual: true,
        fused_rmsnorm: Some(RmsNormEpilogue {
            kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
            width: tp.geometry.hidden(),
            eps: epsilon,
        }),
    }
}

/// The phased reduce-scatter and fused gather symbols: the scatter adds
/// the residual before its one rounding and the gather normalizes rows
/// in the low-latency order.
pub(crate) const PHASED_SYMBOLS: [&str; 2] = [
    kernels_sm120::comm::PREFILL_ROUND_ONCE_SYMBOL,
    kernels_sm120::comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL,
];

/// Where a hidden all-reduce runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HiddenRoute {
    /// Prefill: the reduce-scatter and fused gather stages above 32 rows,
    /// the exact entries below, so a prompt row normalizes as it would
    /// in any chunk.
    Prefill,
    /// A decode or verify graph: the wide entry above 32 rows when
    /// `wide`, the phased stages otherwise. Instance create turns `wide`
    /// off for `CHOPINFER_MIMO_WIDE_COMM=0` or when the device cannot
    /// keep the wide grids of the ranks on it resident at once.
    Graph { wide: bool },
    /// A verify graph whose rows must be bitwise those of the same rows
    /// reduced alone: the verify push entries, which normalize one-shot
    /// rows in the low-latency order, up to their row limit.
    ExactRows,
}

/// The plain and combine entries of an [`HiddenRoute::ExactRows`] or
/// prefill all-reduce of `rows` rows over `ranks`: the verify entries,
/// or at low-latency row counts, whose rows already sum in the one-row
/// order, the entries a decode step of that many rows takes.
fn exact_symbols_at(
    tp: Tp,
    ranks: u32,
    rows: u32,
) -> Result<(&'static str, &'static str), Refusal> {
    if rows == 0 || rows > kernels_sm120::comm::verify_entry_rows(ranks)? {
        return Err(Refusal(
            "MiMo exact-row all-reduce above the verify entries' rows",
        ));
    }
    if kernels_sm120::comm::push_protocol(ranks, rows)?
        == kernels_sm120::comm::PushProtocol::LowLatency
    {
        return Ok(push_symbols(ranks, rows));
    }
    Ok((
        verify_symbol(tp, kernels_sm120::comm::VERIFY_SYMBOL)?,
        verify_symbol(tp, kernels_sm120::comm::VERIFY_COMBINE_SYMBOL)?,
    ))
}

/// The verify cubin's entry `symbol` in a `tp` instance: the shared name on
/// a pinned geometry, the geometry's renamed entry otherwise
/// (`kernels::comm::verify_for`), which a cubin it loads holds.
fn verify_symbol(tp: Tp, symbol: &'static str) -> Result<&'static str, Refusal> {
    if model_mimo26_contract::sm120::geometry::pinned_names(*tp.geometry) {
        return Ok(symbol);
    }
    let stem = symbol
        .strip_prefix("chop_comm_")
        .unwrap()
        .strip_suffix("_v1")
        .unwrap();
    let symbol = super::selection::entry_symbol(tp, "comm", stem);
    crate::CUBINS
        .iter()
        .filter(|cubin| crate::topology::cubin_serves(cubin.name, tp))
        .flat_map(|cubin| cubin.symbols.iter().copied())
        .find(|own| *own == symbol)
        .ok_or(Refusal("MiMo geometry exact-row collective is missing"))
}

/// The exact all-reduce entries of a `tp` instance (its verify cubin's plain
/// and combine entries) and what they keep: what the comm declarations claim
/// for the shared verify entries, whose code a geometry's renamed entries
/// compile unchanged (`kernels::comm::verify_for` renames symbols only).
pub fn exact_entries(
    tp: Tp,
) -> Result<
    (
        &'static str,
        &'static str,
        kernel_core::properties::Invariance,
    ),
    Refusal,
> {
    let declared = kernels_sm120::comm::declared_entries();
    let claim = |symbol: &str| {
        declared
            .iter()
            .find(|entry| entry.symbol == symbol)
            .map(|entry| entry.invariance)
            .ok_or(Refusal("the comm entries declare no exact all-reduce"))
    };
    let invariance = claim(kernels_sm120::comm::VERIFY_SYMBOL)?;
    if claim(kernels_sm120::comm::VERIFY_COMBINE_SYMBOL)? != invariance {
        return Err(Refusal(
            "the exact all-reduce entries declare different claims",
        ));
    }
    Ok((
        verify_symbol(tp, kernels_sm120::comm::VERIFY_SYMBOL)?,
        verify_symbol(tp, kernels_sm120::comm::VERIFY_COMBINE_SYMBOL)?,
        invariance,
    ))
}

pub use bundle_runtime::residency::SpinningEntry;

/// The spinning comm entries of an instance over `ranks` whose graph
/// buckets reach `capacity` rows: the push entries at 32 rows, the
/// phased stages at the prefill maximum and, with the pipelined prefill's
/// `lanes`, at the comm lane's grid, and, above 32 rows, the wide entry at
/// its largest bucket. Instance create checks each against the device
/// (`bundle_runtime::residency`). Refuses a comm lane grid past what an
/// SM120 GPU can hold at once (`overlap::MAX_COMM_CTAS`), which no device
/// check could pass and whose workers' warp count wraps from 2^29.
pub fn spinning_entries_for(
    tp: Tp,
    capacity: u32,
    lanes: Option<&crate::overlap::Config>,
) -> Result<Vec<SpinningEntry>, Refusal> {
    let ranks = tp.ranks();
    let mut entries = Vec::new();
    // Every row count up to 32 picks its push protocol and its plain and
    // combine symbols (the decode entries where they are faster); each
    // symbol and cluster shape is checked at its largest grid.
    for rows in 1..=32 {
        let (variant, push) = hidden_variant(
            tp,
            ranks,
            rows,
            HiddenRoute::Graph { wide: false },
            tp.geometry.eps() as f32,
        )?;
        if variant != kernels_sm120::comm::Variant::FusedOneShot {
            continue;
        }
        let (plain, combine) = push_symbols(ranks, rows);
        let exact = exact_symbols_at(tp, ranks, rows).ok();
        for symbol in [plain, combine].into_iter().chain(
            exact
                .into_iter()
                .flat_map(|(plain, combine)| [plain, combine]),
        ) {
            entries.push(SpinningEntry {
                symbol,
                plan: push,
                wide: false,
            });
        }
    }
    let (_, phased) = hidden_variant(
        tp,
        ranks,
        PREFILL_TOKENS,
        HiddenRoute::Prefill,
        tp.geometry.eps() as f32,
    )?;
    for symbol in PHASED_SYMBOLS {
        entries.push(SpinningEntry {
            symbol,
            plan: phased,
            wide: false,
        });
    }
    if let Some(lanes) = lanes {
        if lanes
            .comm_ctas
            .is_some_and(|ctas| ctas > super::overlap::MAX_COMM_CTAS)
        {
            return Err(Refusal(
                "the overlap's comm grid exceeds the CTAs an SM120 GPU keeps resident",
            ));
        }
        // The comm lane's grid for the phased pair (`overlap::adjust`).
        let mut plan = phased;
        let facts = crate::overlap::Facts {
            persistent: false,
            phased_comm: true,
            collective: true,
        };
        bundle_runtime::lanes::adjust_grid(
            &mut plan.grid,
            crate::overlap::Lane::Comm,
            lanes,
            facts,
        );
        for symbol in PHASED_SYMBOLS {
            entries.push(SpinningEntry {
                symbol,
                plan,
                wide: false,
            });
        }
    }
    let rows = capacity.min(kernels_sm120::comm::wide_max_rows(ranks)?);
    if rows > 32 {
        let (variant, wide) = hidden_variant(
            tp,
            ranks,
            rows,
            HiddenRoute::Graph { wide: true },
            tp.geometry.eps() as f32,
        )?;
        if variant == kernels_sm120::comm::Variant::FusedWide {
            entries.push(SpinningEntry {
                symbol: kernels_sm120::comm::WIDE_SYMBOL,
                plan: wide,
                wide: true,
            });
        }
    }
    Ok(bundle_runtime::residency::largest(entries))
}

/// The variant and plan of a hidden all-reduce of `rows` rows: graphs
/// that may take the wide entry take it above 32 rows, the rest the
/// reduce-scatter and fused gather stages.
fn hidden_variant(
    tp: Tp,
    ranks: u32,
    rows: u32,
    route: HiddenRoute,
    epsilon: f32,
) -> Result<(kernels_sm120::comm::Variant, LaunchPlan), Refusal> {
    let query = hidden_query(
        tp,
        ranks,
        u64::from(rows) * tp.geometry.hidden() as u64,
        epsilon,
    );
    if route == (HiddenRoute::Graph { wide: true }) {
        Ok((
            kernels_sm120::comm::graph_choose(&query)?,
            kernels_sm120::comm::graph_plan(&query, rows)?,
        ))
    } else {
        Ok((
            kernels_sm120::comm::choose(&query)?,
            kernels_sm120::comm::plan(&query, rows)?,
        ))
    }
}

/// Whether a graph of `rows` rows folds the MoE combine into its MLP
/// all-reduce: only the push entries of 32 rows and fewer do. The wide
/// entry has a combine variant, but at TP4 it measured 39.9, 66.2 and
/// 117.0 us per call at 64, 128 and 256 rows against 31.0, 52.1 and
/// 92.6 for the plain wide entry, while the combine row costs about 4
/// to 8 us: its eight sending CTAs read eight routed rows per vector
/// sent. So wider graphs keep the combine row.
pub fn graph_folds_combine_for(tp: Tp, rows: u32) -> bool {
    let ranks = tp.ranks();
    hidden_variant(
        tp,
        ranks,
        rows,
        HiddenRoute::Graph { wide: true },
        tp.geometry.eps() as f32,
    )
    .is_ok_and(|(variant, _)| variant == kernels_sm120::comm::Variant::FusedOneShot)
}

/// Rank-local residual add and RMSNorm after a hidden reduction over
/// `call.ranks` ranks.
/// Prefill uses a reduce-scatter stage followed by a fused gather stage
/// above 32 rows; decode and verify graphs the wide entry when their
/// route allows it.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn fused_hidden_for(
    tp: Tp,
    rows: u32,
    rank: u32,
    route: HiddenRoute,
    call: OneShotCall,
    gathered_outputs: Vec<u64>,
    residual: ResidualBuffers,
    weights: u64,
    epsilon: f32,
) -> Result<Vec<Stage>, Refusal> {
    if call.ranks != tp.ranks() {
        return Err(Refusal(
            "MiMo hidden collective rank count differs from topology",
        ));
    }
    fused_hidden_at(
        tp,
        rows,
        rank,
        route,
        call,
        gathered_outputs,
        residual,
        weights,
        epsilon,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn fused_hidden_at(
    tp: Tp,
    rows: u32,
    rank: u32,
    route: HiddenRoute,
    call: OneShotCall,
    gathered_outputs: Vec<u64>,
    residual: ResidualBuffers,
    weights: u64,
    epsilon: f32,
) -> Result<Vec<Stage>, Refusal> {
    let ranks = call.ranks;
    if rows == 0
        || rows > 8192
        || !valid_ranks(rank, ranks)
        || call.rank != rank
        || call.elements != u64::from(rows) * tp.geometry.hidden() as u64
        || call.peer_inboxes.len() != ranks as usize
        || call.peer_flags.len() != ranks as usize
        || gathered_outputs.len() != ranks as usize
    {
        return Err(Refusal("MiMo fused hidden communication geometry"));
    }
    if route == HiddenRoute::ExactRows {
        exact_symbols_at(tp, ranks, rows)?;
    }
    let (variant, plan) = hidden_variant(tp, ranks, rows, route, epsilon)?;
    match variant {
        kernels_sm120::comm::Variant::FusedOneShot | kernels_sm120::comm::Variant::FusedWide => {
            let symbol = if variant == kernels_sm120::comm::Variant::FusedWide {
                kernels_sm120::comm::WIDE_SYMBOL
            } else if matches!(route, HiddenRoute::ExactRows | HiddenRoute::Prefill) {
                exact_symbols_at(tp, ranks, rows)?.0
            } else {
                push_symbols(ranks, rows).0
            };
            Ok(vec![(
                symbol.into(),
                plan,
                kernels_sm120::comm::fused_arguments_v2(
                    kernels_sm120::comm::FusedWithResidualOutputCall {
                        fused: kernels_sm120::comm::FusedOneShotCall {
                            base: call,
                            residual: Some(residual.input),
                            weights,
                            epsilon,
                        },
                        residual_output: residual.output,
                    },
                )?,
            )])
        }
        kernels_sm120::comm::Variant::FusedTwoShot => {
            let scatter = OneShotCall {
                input: call.input,
                output: call.output,
                peer_inboxes: call.peer_inboxes,
                peer_flags: call.peer_flags.clone(),
                epoch: call.epoch,
                elements: call.elements,
                rank,
                ranks,
            };
            // The scatter adds the residual; the gather sends the sums.
            let gather = kernels_sm120::comm::round_once_gather_arguments(
                &scatter,
                gathered_outputs,
                weights,
                epsilon,
                residual.output,
            )?;
            Ok(vec![
                (
                    PHASED_SYMBOLS[0].into(),
                    plan,
                    kernels_sm120::comm::round_once_scatter_arguments(scatter, residual.input)?,
                ),
                (PHASED_SYMBOLS[1].into(), plan, gather),
            ])
        }
        _ => Err(Refusal("MiMo fused hidden communication variant")),
    }
}

/// Decode and verify MoE all-reduce that also sums each row's eight
/// preweighted routed rows `[rows][8][4096]`, as
/// `chop_row_combine_mimo4096_v1` would; rows at or past `*live_rows`
/// contribute zero. The push entries (up to 32 rows) and the wide ones
/// have this variant; the bundle folds only the push entries'
/// (`graph_folds_combine`).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn fused_combine_hidden_for(
    tp: Tp,
    rows: u32,
    rank: u32,
    route: HiddenRoute,
    call: OneShotCall,
    residual: ResidualBuffers,
    weights: u64,
    epsilon: f32,
    live_rows: u64,
) -> Result<Stage, Refusal> {
    if call.ranks != tp.ranks() {
        return Err(Refusal(
            "MiMo hidden collective rank count differs from topology",
        ));
    }
    fused_combine_hidden_at(
        tp, rows, rank, route, call, residual, weights, epsilon, live_rows,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn fused_combine_hidden_at(
    tp: Tp,
    rows: u32,
    rank: u32,
    route: HiddenRoute,
    call: OneShotCall,
    residual: ResidualBuffers,
    weights: u64,
    epsilon: f32,
    live_rows: u64,
) -> Result<Stage, Refusal> {
    let ranks = call.ranks;
    if rows == 0
        || !valid_ranks(rank, ranks)
        || call.rank != rank
        || call.elements != u64::from(rows) * tp.geometry.hidden() as u64
        || call.peer_inboxes.len() != ranks as usize
        || call.peer_flags.len() != ranks as usize
    {
        return Err(Refusal("MiMo fused combine communication geometry"));
    }
    if route == HiddenRoute::Prefill {
        return Err(Refusal("MiMo folds the MoE combine in graphs only"));
    }
    if route == HiddenRoute::ExactRows {
        exact_symbols_at(tp, ranks, rows)?;
    }
    let (variant, plan) = hidden_variant(tp, ranks, rows, route, epsilon)?;
    let symbol = match variant {
        kernels_sm120::comm::Variant::FusedOneShot if route == HiddenRoute::ExactRows => {
            exact_symbols_at(tp, ranks, rows)?.1
        }
        kernels_sm120::comm::Variant::FusedOneShot => push_symbols(ranks, rows).1,
        kernels_sm120::comm::Variant::FusedWide => kernels_sm120::comm::WIDE_COMBINE_SYMBOL,
        _ => {
            return Err(Refusal(
                "MiMo fused combine needs the push or wide all-reduce",
            ));
        }
    };
    Ok((
        symbol.into(),
        plan,
        kernels_sm120::comm::fused_combine_arguments(
            kernels_sm120::comm::FusedWithResidualOutputCall {
                fused: kernels_sm120::comm::FusedOneShotCall {
                    base: call,
                    residual: Some(residual.input),
                    weights,
                    epsilon,
                },
                residual_output: residual.output,
            },
            Some(live_rows),
        )?,
    ))
}

#[cfg(test)]
pub fn gather_logits_rows(
    dtype: DType,
    call: kernels_sm120::comm::GatherRowsCall,
) -> Result<Stage, Refusal> {
    if call.rows == 0 || call.rows > PREFILL_TOKENS || !matches!(dtype, DType::Bf16 | DType::F32) {
        return Err(Refusal("MiMo vocabulary row gather geometry"));
    }
    let arguments = kernels_sm120::comm::gather_rows_arguments(call)?;
    let symbol = match dtype {
        DType::Bf16 => "chop_comm_gather_rows_bf16_v1",
        DType::F32 => "chop_comm_gather_rows_f32_v1",
        _ => unreachable!(),
    };
    Ok((
        symbol.into(),
        kernels_sm120::comm::gather_rows_plan(),
        arguments,
    ))
}
#[cfg(test)]
pub fn exact_symbols_for(tp: Tp, rows: u32) -> Result<(&'static str, &'static str), Refusal> {
    exact_symbols_at(tp, tp.ranks(), rows)
}
#[cfg(test)]
pub fn exact_symbols(ranks: u32, rows: u32) -> Result<(&'static str, &'static str), Refusal> {
    exact_symbols_at(Tp::TP4, ranks, rows)
}
#[cfg(test)]
pub fn graph_folds_combine(ranks: u32, rows: u32) -> bool {
    hidden_variant(
        Tp::TP4,
        ranks,
        rows,
        HiddenRoute::Graph { wide: true },
        Tp::TP4.geometry.eps() as f32,
    )
    .is_ok_and(|(variant, _)| variant == kernels_sm120::comm::Variant::FusedOneShot)
}
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn fused_hidden(
    rows: u32,
    rank: u32,
    route: HiddenRoute,
    call: OneShotCall,
    gathered_outputs: Vec<u64>,
    residual: ResidualBuffers,
    weights: u64,
    epsilon: f32,
) -> Result<Vec<Stage>, Refusal> {
    fused_hidden_at(
        Tp::TP4,
        rows,
        rank,
        route,
        call,
        gathered_outputs,
        residual,
        weights,
        epsilon,
    )
}
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn fused_combine_hidden(
    rows: u32,
    rank: u32,
    route: HiddenRoute,
    call: OneShotCall,
    residual: ResidualBuffers,
    weights: u64,
    epsilon: f32,
    live_rows: u64,
) -> Result<Stage, Refusal> {
    fused_combine_hidden_at(
        Tp::TP4,
        rows,
        rank,
        route,
        call,
        residual,
        weights,
        epsilon,
        live_rows,
    )
}

/// DFlash's candidate all-gather as MiMo planned it before `crate::ops`
/// prepared it, which tests compare the prepared op with: rank `rank`'s
/// `2 * capacity` F32 words of `pairs` onto every rank's `gathered`.
#[cfg(test)]
pub fn candidates(
    rank: u32,
    capacity: u32,
    pairs: runtime_bundle_abi::DeviceBufferSpanV1,
    gathered: &[runtime_bundle_abi::DeviceBufferSpanV1],
    flags: &[runtime_bundle_abi::DeviceBufferSpanV1],
    epoch: runtime_bundle_abi::DeviceBufferSpanV1,
) -> Result<crate::pipeline::PreparedLaunch, Refusal> {
    use kernel_core::contract::{Argument, Validate};
    let ranks = gathered.len() as u32;
    let words = 2 * u64::from(capacity);
    let query = CommQuery::AllGather {
        ranks,
        elements_per_rank: words,
        dtype: DType::F32,
    };
    let mut bindings = crate::dispatch::BufferBindings::default();
    let arguments = ArgumentList(vec![
        Argument::DeviceBuffer(bindings.bind(pairs)),
        Argument::DevicePointerArray(gathered.iter().map(|&span| bindings.bind(span)).collect()),
        Argument::DevicePointerArray(flags.iter().map(|&span| bindings.bind(span)).collect()),
        Argument::DeviceBuffer(bindings.bind(epoch)),
        Argument::U64(words),
        Argument::U32(rank),
        Argument::U32(ranks),
        // A target equal to the rank count gathers onto every rank.
        Argument::U32(ranks),
    ]);
    arguments.validate()?;
    Ok(crate::pipeline::PreparedLaunch {
        symbol: "chop_comm_allgather_f32_v1".into(),
        rows: capacity,
        plan: kernels_sm120::comm::plan(&query, 1)?,
        arguments,
        bindings,
    })
}
