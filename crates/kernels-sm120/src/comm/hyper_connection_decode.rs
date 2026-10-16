//! Decode hyper-connection all-reduce, inject and grouped Gemma norm for
//! graph buckets of up to 32 rows at 2, 4 or 8 ranks, bitwise as the
//! one-shot entry of `hyper_connection`. Low latency (recursive doubling of
//! flagged entries) up to `LL_MAX_ROWS` prepared rows, one-shot pushes with
//! one CTA per (row, stream) above. Owns a separate comm state; all ranks use
//! the same prepared and live rows. Each rank count has its own entry
//! (`symbol_v2`), whose rank count the call's pointer arrays give; the entry
//! also takes it as its last argument and traps on any other. A model's
//! transport ([`HcDecodeTransport`]) names its shape, epsilon, cubins and
//! entries.

use super::hyper_connection::{self as prefill, EPILOGUE_ARGUMENTS, HcShape, HcTransport, RANKS};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, InstantiationSpec, LaunchPlan,
    MAX_COMM_RANKS, OpFamily, Refusal, SigmoidMeanHyperConnectionQuery, StaticQuery, Target,
    Validate,
};
use std::path::PathBuf;

/// A model's decode hyper-connection transport: its shape, its grouped
/// norm's epsilon, its cubins, and its entries for each rank count of
/// `RANKS`, in its order. Each entry is compiled for its rank count and
/// declares that many pointer slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HcDecodeTransport {
    pub shape: HcShape,
    /// The grouped norm's epsilon, as F32 bits.
    pub eps_bits: u32,
    /// The decode entries' cubin.
    pub cubin: &'static str,
    /// The wide entries' cubin.
    pub wide_cubin: &'static str,
    /// The decode entries (`allreduce_hc` in `hyper_connection_decode.cuh`).
    pub symbols_v2: [&'static str; 3],
    /// Whole-row LL inbox readers, with the v2 arguments, state and bits.
    /// Aliasing v2 keeps transports with only their v2 entries.
    pub symbols_v3: [&'static str; 3],
    /// Wide entries for graph buckets above 32 rows
    /// (`comm/hyper_connection_decode_wide.cuh`): the decode entry's arguments and
    /// bits for 1 to `WIDE_MAX_ROWS` prepared rows, on a decode state
    /// extended by `wide_buffers_with`, where calls of both entries may
    /// follow each other.
    pub wide_symbols: [&'static str; 3],
}

/// The index of `ranks` in `RANKS`.
fn rank_index(ranks: u32) -> Result<usize, Refusal> {
    RANKS.iter().position(|&r| r == ranks).ok_or(Refusal(
        "hyper-connection decode comm takes 2, 4 or 8 ranks",
    ))
}

/// The decode entry for `ranks` ranks.
pub fn symbol_v2(t: &HcDecodeTransport, ranks: u32) -> Result<&'static str, Refusal> {
    rank_index(ranks).map(|index| t.symbols_v2[index])
}

/// The whole-row LL entry for `ranks` ranks.
pub fn symbol_v3(t: &HcDecodeTransport, ranks: u32) -> Result<&'static str, Refusal> {
    rank_index(ranks).map(|index| t.symbols_v3[index])
}

pub const MAX_ROWS: u32 = 32;
/// Prepared rows up to this take the low-latency protocol.
pub const LL_MAX_ROWS: u32 = 2;
const SENDERS: u32 = 8;

pub fn query(t: &HcDecodeTransport, ranks: u32) -> SigmoidMeanHyperConnectionQuery {
    SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        ranks,
        streams: t.shape.epilogue.streams,
        width: t.shape.epilogue.width,
        eps_bits: t.eps_bits,
        capacity: MAX_ROWS,
    }
}

pub fn contract(t: &HcDecodeTransport, ranks: u32) -> Result<InstantiationSpec, Refusal> {
    rank_index(ranks)?;
    t.shape.validate()?;
    t.shape.epilogue.validate_hc_epsilon(t.eps_bits)?;
    let spec = InstantiationSpec {
        family: OpFamily::Comm,
        key: CompileKey {
            query: StaticQuery::HyperConnection(query(t, ranks)),
            dense: None,
            target: Target::Sm120f,
            knobs: CompileKnobs {
                tile_m: 1,
                tile_n: 256,
                tile_k: 1,
                stages: 1,
                swizzle: 0,
                warps: 8,
                split_k_capacity: 1,
            },
        },
        symbol: symbol_v2(t, ranks)?.into(),
    };
    spec.validate()?;
    Ok(spec)
}

/// The rows at or below which `shape`'s entries take the LL protocol: none
/// when its LL layout does not fit (`HcShape::low_latency_fits`).
pub fn low_latency_rows(shape: HcShape) -> u32 {
    if shape.low_latency_fits() {
        LL_MAX_ROWS
    } else {
        0
    }
}

/// Geometry for `rows` prepared rows (the graph bucket) of `shape`; the live
/// count is a device scalar at most `rows`. It does not depend on the rank
/// count. One CTA per (row, stream) group. A shape
/// without LL pads its groups to whole sending clusters, so its one-shot
/// takes every row count (`qwen38_decode::Rows::group_ctas`); a shape with
/// LL takes the one-shot only for rows whose groups fill them.
pub fn plan_with(shape: HcShape, rows: u32) -> Result<LaunchPlan, Refusal> {
    shape.validate()?;
    if rows == 0 || rows > MAX_ROWS {
        return Err(Refusal(
            "hyper-connection decode comm rows exceed the decode buckets",
        ));
    }
    let streams = shape.epilogue.streams;
    let (grid, cluster) = if rows <= low_latency_rows(shape) {
        (rows * streams, streams)
    } else if !shape.low_latency_fits() {
        (
            SENDERS + (rows * streams).next_multiple_of(SENDERS),
            SENDERS,
        )
    } else if (rows * streams) % SENDERS == 0 {
        (SENDERS + rows * streams, SENDERS)
    } else {
        return Err(Refusal(
            "hyper-connection decode one-shot rows must fill whole clusters",
        ));
    };
    let plan = LaunchPlan {
        grid: [grid, 1, 1],
        block: [256, 1, 1],
        cluster: [cluster, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

#[derive(Clone, Copy, Debug)]
pub struct Buffers {
    pub inbox_bytes: u64,
    /// The inbox's first bytes, the LL region, which must start zeroed:
    /// other bytes can look like LL entries.
    pub zeroed_inbox_bytes: u64,
    pub flags_bytes: u64,
    pub epoch_bytes: u64,
}

/// The state of `shape`'s calls at `ranks` ranks. Flags, epochs and the
/// first `zeroed_inbox_bytes` of the inbox start at zero; the rest of the
/// inbox may hold any bytes. Calls with any prepared rows and any live count
/// up to them, zero included, may follow each other on one state without a
/// host barrier. Each region has a slot per rank. The shape's input
/// elements fill the LL entries and one-shot slots. The LL region is kept
/// when its layout does not fit, since the one-shot region's offset follows
/// it.
pub fn buffers_with(shape: HcShape, ranks: u32) -> Result<Buffers, Refusal> {
    shape.validate()?;
    if !RANKS.contains(&ranks) {
        return Err(Refusal(
            "hyper-connection decode comm takes 2, 4 or 8 ranks",
        ));
    }
    let ranks = u64::from(ranks);
    let rows = u64::from(MAX_ROWS);
    let row_bytes = u64::from(shape.epilogue.width) * u64::from(shape.input.element_bytes);
    // LL entries carry 4 bytes of data in 8; both regions are double-buffered.
    let low_latency = 2 * ranks * rows * row_bytes * 2;
    let one_shot = 2 * ranks * rows * row_bytes;
    Ok(Buffers {
        inbox_bytes: low_latency + one_shot,
        zeroed_inbox_bytes: low_latency,
        flags_bytes: 2 * ranks * 4,
        epoch_bytes: (rows + 2) * 4,
    })
}

pub struct Call {
    pub input: u64,
    pub residual: u64,
    pub logits: u64,
    pub weight: u64,
    pub residual_out: u64,
    pub normed: u64,
    pub inboxes: Vec<u64>,
    pub flags: Vec<u64>,
    pub epochs: u64,
    pub live_rows: u64,
    /// Prepared rows, as given to `plan`.
    pub rows: u32,
    pub rank: u32,
}

/// One inbox and one flags slot per rank; the rank count is their length.
pub fn arguments(call: Call) -> Result<ArgumentList, Refusal> {
    arguments_up_to(call, MAX_ROWS)
}

fn arguments_up_to(call: Call, max_rows: u32) -> Result<ArgumentList, Refusal> {
    let ranks = u32::try_from(call.inboxes.len()).unwrap_or(0);
    if !RANKS.contains(&ranks)
        || call.flags.len() != call.inboxes.len()
        || call.rank >= ranks
        || call.rows == 0
        || call.rows > max_rows
    {
        return Err(Refusal("invalid hyper-connection decode comm call"));
    }
    let mut args: Vec<_> = [
        call.input,
        call.residual,
        call.logits,
        call.weight,
        call.residual_out,
        call.normed,
    ]
    .into_iter()
    .map(Argument::DeviceBuffer)
    .collect();
    args.extend([
        Argument::DevicePointerArray(call.inboxes),
        Argument::DevicePointerArray(call.flags),
        Argument::DeviceBuffer(call.epochs),
        Argument::DeviceBuffer(call.live_rows),
        Argument::U32(call.rows),
        Argument::U32(call.rank),
        Argument::U32(ranks),
    ]);
    let args = ArgumentList(args);
    args.validate()?;
    Ok(args)
}

/// The entry `symbol` of `shape` for `ranks` ranks, calling
/// `chop::comm::qwen38_decode::{function}<E, Input, {ranks}{flags}>`.
fn entry_spec(
    shape: HcShape,
    ranks: u32,
    symbol: &str,
    function: &str,
    flags: &str,
    min_blocks: u32,
) -> EntrySpec {
    let element = shape.input.cxx_type;
    let input_type = format!("const {element}");
    let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let mut params = vec![ptr("input", &input_type)];
    params.extend(["residual", "logits", "weight"].into_iter().map(|n| {
        ptr(
            n,
            if n == "weight" {
                shape.epilogue.weight_pointer_type()
            } else {
                "const uint16_t"
            },
        )
    }));
    params.extend([
        ptr("residual_out", "uint16_t"),
        ptr("normed", "uint16_t"),
        EntryParam::PointerArray {
            name: "inboxes".into(),
            slots: ranks as u8,
            cxx_type: element.into(),
        },
        EntryParam::PointerArray {
            name: "flags".into(),
            slots: ranks as u8,
            cxx_type: "uint32_t".into(),
        },
        ptr("epochs", "uint32_t"),
        ptr("live_rows", "const uint32_t"),
        EntryParam::U32 {
            name: "rows".into(),
        },
        EntryParam::U32 {
            name: "rank".into(),
        },
        EntryParam::U32 {
            name: "ranks".into(),
        },
    ]);
    let e = format!("{}, {}", shape.epilogue.functor(), shape.input.policy);
    EntrySpec::new(
        symbol.into(),
        params,
        format!(
            "chop::comm::qwen38_decode::{function}<{e}, {ranks}{flags}>(input, {EPILOGUE_ARGUMENTS}, inboxes, flags, epochs, live_rows, rows, rank, ranks);"
        ),
        256,
        min_blocks,
    )
}

/// `t`'s decode cubin with the entries of `ranks` (every rank count in
/// `RANKS` for a lab, one for a bundle built for one topology),
/// instantiated with its shape: v2, then v3 unless it aliases v2.
pub fn cubin_specs(t: &HcDecodeTransport, ranks: &[u32]) -> Vec<CubinSpec> {
    let shape = t.shape;
    shape
        .validate()
        .expect("hyper-connection comm shape the transports take");
    let ranks: Vec<_> = RANKS.into_iter().filter(|r| ranks.contains(r)).collect();
    let entry = |r: u32, full_row: bool| {
        let (symbol, flags) = if full_row {
            (symbol_v3(t, r), ", true")
        } else {
            (symbol_v2(t, r), "")
        };
        let symbol = symbol.expect("catalog rank count");
        entry_spec(shape, r, symbol, "allreduce_hc", flags, 1)
    };
    let full_row = if t.symbols_v3 != t.symbols_v2 {
        &[false, true][..]
    } else {
        &[false]
    };
    vec![CubinSpec {
        name: t.cubin.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![
            "#include \"comm/hyper_connection_decode.cuh\"".into(),
            shape.epilogue.include(),
            shape.input.include(),
        ],
        defines: defines_with(shape),
        entries: full_row
            .iter()
            .flat_map(|&full| ranks.iter().map(move |&r| entry(r, full)))
            .collect(),
    }]
}

/// The LL row limit plus the push tuning that `comm.cuh` needs, whose LL
/// and flag helpers this entry uses: no LL rows when `shape`'s LL layout
/// does not fit.
pub fn defines_with(shape: HcShape) -> Vec<(String, String)> {
    let mut defines = super::push_defines();
    defines.push((
        "CHOP_QWEN38_LL_MAX_ROWS".into(),
        low_latency_rows(shape).to_string(),
    ));
    defines
}

/// Most prepared rows of a wide call.
pub const WIDE_MAX_ROWS: u32 = 256;
/// Rows per flagged block of the wide protocols' last phase.
const WIDE_BLOCK_ROWS: u32 = 16;
/// Most worker CTAs of a wide call, each taking (row, stream) groups; the
/// kernel takes any count.
const WIDE_WORKERS: u32 = 128;
/// Prepared rows from which the wide entry of each rank count in `RANKS`
/// takes the two-shot rather than the one-shot. At two ranks the one-shot
/// sends 2 bytes per element against the two-shot's 3 (F32 sums), in one
/// phase; at four and eight the two-shot sends 6 (ranks - 1) / ranks bytes
/// against the one-shot's 2 (ranks - 1).
const WIDE_TWO_SHOT_ROWS: [u32; 3] = [WIDE_MAX_ROWS + 1, 33, 33];
/// Flag words of `wide_buffers` (`kWideFlagWords`): the decode words, then
/// one-shot [parity][source] and [parity][block], two-shot [parity][source],
/// [source] and two block sets, a ready word, and shape words and flags
/// [parity][source].
const WIDE_FLAG_WORDS: u64 = {
    let ranks = MAX_COMM_RANKS as u64;
    let blocks = (SENDERS * (WIDE_MAX_ROWS.div_ceil(SENDERS * WIDE_BLOCK_ROWS))) as u64;
    2 * ranks + 2 * ranks + 2 * blocks + 2 * ranks + ranks + 2 * blocks + 1 + 4 * ranks
};

/// Most rows of a graph that takes the wide entry at each rank count in
/// `RANKS`; graphs above it take the phased pair. At four ranks (TP4,
/// exclusive, 2026-09-26) the wide entry measured 53.4 and 87.2 us per call
/// at 64 and 128 rows against the phased pair's 101.0 and 106.2, but 143.6
/// to 156.2 us at 255 and 256 rows against 124 to 128; eight ranks take the
/// same cap until measured. At two ranks it measured 81.3 us at 256 rows
/// against 112.1.
const WIDE_GRAPH_ROWS: [u32; 3] = [WIDE_MAX_ROWS, 128, 128];

/// Whether a decode graph of `rows` prepared rows takes the wide entry
/// (above `MAX_ROWS`, up to the rank count's cap) rather than the decode
/// entry (up to `MAX_ROWS`) or the phased pair (above the cap).
pub fn graph_takes_wide(ranks: u32, rows: u32) -> Result<bool, Refusal> {
    let index = rank_index(ranks)?;
    Ok(rows > MAX_ROWS && rows <= WIDE_GRAPH_ROWS[index])
}

/// The wide entry for `ranks` ranks.
pub fn wide_symbol(t: &HcDecodeTransport, ranks: u32) -> Result<&'static str, Refusal> {
    rank_index(ranks).map(|index| t.wide_symbols[index])
}

/// Protocol of the wide entry, by the prepared rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WideProtocol {
    /// Each rank's live rows to every peer in turn.
    OneShot,
    /// Column owners: BF16 partials in, F32 sums out.
    TwoShot,
}

pub fn wide_protocol(ranks: u32, rows: u32) -> Result<WideProtocol, Refusal> {
    let index = rank_index(ranks)?;
    if rows == 0 || rows > WIDE_MAX_ROWS {
        return Err(Refusal(
            "hyper-connection wide decode comm rows exceed its buckets",
        ));
    }
    Ok(if rows >= WIDE_TWO_SHOT_ROWS[index] {
        WideProtocol::TwoShot
    } else {
        WideProtocol::OneShot
    })
}

pub fn wide_query(t: &HcDecodeTransport, ranks: u32) -> SigmoidMeanHyperConnectionQuery {
    SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        ranks,
        streams: t.shape.epilogue.streams,
        width: t.shape.epilogue.width,
        eps_bits: t.eps_bits,
        capacity: WIDE_MAX_ROWS,
    }
}

pub fn wide_contract(t: &HcDecodeTransport, ranks: u32) -> Result<InstantiationSpec, Refusal> {
    let mut spec = contract(t, ranks)?;
    spec.key.query = StaticQuery::HyperConnection(wide_query(t, ranks));
    spec.symbol = wide_symbol(t, ranks)?.into();
    spec.validate()?;
    Ok(spec)
}

/// Geometry for `rows` prepared rows of `shape`: the sending cluster and a
/// worker per (row, stream) group up to `WIDE_WORKERS`. It does not depend
/// on the rank count or the live rows.
pub fn wide_plan_with(shape: HcShape, rows: u32) -> Result<LaunchPlan, Refusal> {
    shape.validate()?;
    if rows == 0 || rows > WIDE_MAX_ROWS {
        return Err(Refusal(
            "hyper-connection wide decode comm rows exceed its buckets",
        ));
    }
    let workers = (rows * shape.epilogue.streams)
        .min(WIDE_WORKERS)
        .next_multiple_of(SENDERS);
    let plan = LaunchPlan {
        grid: [SENDERS + workers, 1, 1],
        block: [256, 1, 1],
        cluster: [SENDERS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// A decode state of `shape` that also takes wide calls:
/// `buffers_with(shape, ranks)`'s regions, then the wide one-shot's banks
/// [parity][source][256][width] and the two-shot's reduce-scatter
/// [source][256][width / ranks] of input elements and all-gather F32 slots;
/// flags and epochs grow likewise and start zeroed. Its prefix is the
/// decode state, so the decode entry runs on it unchanged.
pub fn wide_buffers_with(shape: HcShape, ranks: u32) -> Result<Buffers, Refusal> {
    let decode = buffers_with(shape, ranks)?;
    let ranks = u64::from(ranks);
    let rows = u64::from(WIDE_MAX_ROWS);
    let width = u64::from(shape.epilogue.width);
    let element = u64::from(shape.input.element_bytes);
    let one_shot = 2 * ranks * rows * width * element;
    let scatter = rows * width * element;
    let gather = rows * width * 4;
    Ok(Buffers {
        inbox_bytes: decode.inbox_bytes + one_shot + scatter + gather,
        zeroed_inbox_bytes: decode.zeroed_inbox_bytes,
        flags_bytes: decode.flags_bytes.max(WIDE_FLAG_WORDS * 4),
        epoch_bytes: decode.epoch_bytes,
    })
}

/// The decode state of `shape`'s calls at `ranks` ranks for graph buckets
/// of up to `rows` prepared rows: the decode state, or above `MAX_ROWS`
/// the one that also takes wide calls (whose zeroed LL region is the
/// decode state's).
pub fn buffers_for_rows(shape: HcShape, ranks: u32, rows: u32) -> Result<Buffers, Refusal> {
    let decode = buffers_with(shape, ranks)?;
    if rows > MAX_ROWS {
        wide_buffers_with(shape, ranks)
    } else {
        Ok(decode)
    }
}

/// An entry of the hyper-connection all-reduces that spins on its peers,
/// at its largest plan, and whether it is the wide entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spinning {
    pub symbol: &'static str,
    pub plan: LaunchPlan,
    pub wide: bool,
}

/// The spinning entries of a model's hyper-connection all-reduces at
/// `ranks` ranks, for graph buckets `buckets` (up to `MAX_ROWS`) and graph
/// buckets of up to `rows` rows: each bucket's decode and whole-row LL
/// entries at its plan, the prefill transport's one-shot at one row and its
/// phased pair at `capacity` rows (`query`'s), and the wide entry at the
/// largest bucket above `buckets` that takes it, whose grid grows with the
/// rows.
pub fn spinning_entries(
    transport: &HcTransport,
    decode: &HcDecodeTransport,
    query: SigmoidMeanHyperConnectionQuery,
    capacity: u32,
    ranks: u32,
    buckets: &[u32],
    rows: u32,
) -> Result<Vec<Spinning>, Refusal> {
    let entry = |symbol, plan| Spinning {
        symbol,
        plan,
        wide: false,
    };
    let phased = prefill::prefill_plan(transport, query, capacity)?;
    let mut entries = Vec::new();
    for &bucket in buckets {
        let plan = plan_with(decode.shape, bucket)?;
        entries.push(entry(symbol_v2(decode, ranks)?, plan));
        entries.push(entry(symbol_v3(decode, ranks)?, plan));
    }
    entries.extend([
        entry(transport.symbol, prefill::plan(transport, query, 1)?),
        entry(transport.scatter_symbol, phased),
        entry(transport.gather_symbol, phased),
    ]);
    let largest = buckets.iter().copied().max().unwrap_or(0);
    if let Some(wide_rows) = (largest + 1..=rows)
        .rev()
        .find(|&rows| graph_takes_wide(ranks, rows).unwrap_or(false))
    {
        entries.push(Spinning {
            symbol: wide_symbol(decode, ranks)?,
            plan: wide_plan_with(decode.shape, wide_rows)?,
            wide: true,
        });
    }
    Ok(entries)
}

/// Arguments of the wide entry; `call.rows` is the prepared rows, up to
/// `WIDE_MAX_ROWS`.
pub fn wide_arguments(call: Call) -> Result<ArgumentList, Refusal> {
    arguments_up_to(call, WIDE_MAX_ROWS)
}

/// The wide defines on top of `defines_with(shape)`.
pub fn wide_defines_with(shape: HcShape) -> Vec<(String, String)> {
    let mut defines = defines_with(shape);
    defines.push((
        "CHOP_QWEN38_WIDE_BLOCK_ROWS".into(),
        WIDE_BLOCK_ROWS.to_string(),
    ));
    for (ranks, rows) in RANKS.into_iter().zip(WIDE_TWO_SHOT_ROWS) {
        defines.push((
            format!("CHOP_QWEN38_WIDE_TWO_SHOT_ROWS_R{ranks}"),
            rows.to_string(),
        ));
    }
    defines
}

/// `t`'s wide entries of `ranks` in a cubin of their own (`wide_cubin`),
/// instantiated with its shape. They keep two CTAs per SM for the spinning
/// workers' residency (`bundle_runtime::residency`).
pub fn wide_cubin_specs(t: &HcDecodeTransport, ranks: &[u32]) -> Vec<CubinSpec> {
    let shape = t.shape;
    shape
        .validate()
        .expect("hyper-connection comm shape the transports take");
    vec![CubinSpec {
        name: t.wide_cubin.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![
            "#include \"comm/hyper_connection_decode_wide.cuh\"".into(),
            shape.epilogue.include(),
            shape.input.include(),
        ],
        defines: wide_defines_with(shape),
        entries: RANKS
            .into_iter()
            .filter(|r| ranks.contains(r))
            .map(|r| {
                let symbol = wide_symbol(t, r).expect("catalog rank count");
                entry_spec(
                    shape,
                    r,
                    symbol,
                    "allreduce_hc_wide",
                    "",
                    super::WIDE_CTAS_PER_SM,
                )
            })
            .collect(),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::epilogue::{Epilogue, HC_INJECT_NORM_EPS_BITS};
    use crate::comm::input;

    const SHAPE: HcShape = HcShape {
        epilogue: Epilogue {
            header: "comm/epilogue/hc_inject_norm.cuh",
            template: "chop::comm::epilogue::HcInjectNorm",
            width: 2560,
            streams: 4,
        },
        input: input::BF16,
    };

    const TRANSPORT: HcTransport = HcTransport {
        shape: SHAPE,
        eps_bits: HC_INJECT_NORM_EPS_BITS,
        cubin: "comm_hc_test_sm120",
        symbol: "chop_comm_hc_v1",
        scatter_symbol: "chop_comm_hc_scatter_v1",
        gather_symbol: "chop_comm_hc_gather_v1",
    };

    const DECODE: HcDecodeTransport = HcDecodeTransport {
        shape: SHAPE,
        eps_bits: HC_INJECT_NORM_EPS_BITS,
        cubin: "comm_hcd_test_sm120",
        wide_cubin: "comm_hcw_test_sm120",
        symbols_v2: ["hcd_r2_v2", "hcd_r4_v2", "hcd_r8_v2"],
        symbols_v3: ["hcd_r2_v3", "hcd_r4_v3", "hcd_r8_v3"],
        wide_symbols: ["hcw_r2", "hcw_r4", "hcw_r8"],
    };

    /// Graphs above the decode rows take the wide state, whose zeroed LL
    /// region is the decode state's.
    #[test]
    fn state_for_rows_widens_above_the_decode_rows() {
        let decode = buffers_with(SHAPE, 4).unwrap();
        let wide = wide_buffers_with(SHAPE, 4).unwrap();
        let of = |rows| buffers_for_rows(SHAPE, 4, rows).unwrap().inbox_bytes;
        assert_eq!(of(MAX_ROWS), decode.inbox_bytes);
        assert_eq!(of(MAX_ROWS + 1), wide.inbox_bytes);
        assert_eq!(
            buffers_for_rows(SHAPE, 4, 256).unwrap().zeroed_inbox_bytes,
            decode.zeroed_inbox_bytes
        );
        assert!(buffers_for_rows(SHAPE, 3, 1).is_err());
    }

    /// Each decode bucket's entries at its plan, the prefill one-shot and
    /// phased pair, and the wide entry only where a graph above the
    /// buckets takes it, at its largest such bucket.
    #[test]
    fn spinning_entries_cover_every_spinning_symbol_at_its_largest_plan() {
        let query = TRANSPORT.query(4);
        let SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm { capacity, .. } = query else {
            unreachable!()
        };
        let buckets = [1, 2, 4, 8, 16, 32];
        let narrow =
            spinning_entries(&TRANSPORT, &DECODE, query, capacity, 4, &buckets, 32).unwrap();
        assert_eq!(narrow.len(), 2 * buckets.len() + 3);
        assert!(narrow.iter().all(|entry| !entry.wide));
        assert_eq!(narrow[0].plan, plan_with(SHAPE, 1).unwrap());
        assert_eq!(
            [narrow[0].symbol, narrow[1].symbol],
            ["hcd_r4_v2", "hcd_r4_v3"]
        );
        let phased = prefill::prefill_plan(&TRANSPORT, query, capacity).unwrap();
        assert_eq!(
            narrow[12..],
            [
                Spinning {
                    symbol: "chop_comm_hc_v1",
                    plan: prefill::plan(&TRANSPORT, query, 1).unwrap(),
                    wide: false,
                },
                Spinning {
                    symbol: "chop_comm_hc_scatter_v1",
                    plan: phased,
                    wide: false,
                },
                Spinning {
                    symbol: "chop_comm_hc_gather_v1",
                    plan: phased,
                    wide: false,
                },
            ]
        );
        let wide =
            spinning_entries(&TRANSPORT, &DECODE, query, capacity, 4, &buckets, 4096).unwrap();
        let last = wide.last().unwrap();
        let rows = (33..=4096)
            .rev()
            .find(|&r| graph_takes_wide(4, r).unwrap())
            .unwrap();
        assert_eq!(
            *last,
            Spinning {
                symbol: "hcw_r4",
                plan: wide_plan_with(SHAPE, rows).unwrap(),
                wide: true,
            }
        );
    }
}
