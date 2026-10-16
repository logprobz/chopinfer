//! Hyper-connection all-reduce, inject and grouped Gemma norm at 2, 4 or 8
//! ranks (`RANKS`). Owns a separate comm state. All ranks use the same
//! capacity and live count. The entries take the rank count at run time, so
//! one cubin serves every rank count; the call's pointer arrays hold one slot
//! per rank and the query names the rank count.
//!
//! A model's transport ([`HcTransport`]) names its shape, its norm epsilon,
//! its cubin and its three entries. Up to `ONE_SHOT_ROWS` rows (short
//! prefills) one launch of the one-shot entry pushes whole partials with one
//! cluster. Above it, prefill runs the phased two-shot pair, scatter then
//! gather, with many CTAs (`prefill_plan`). Both paths produce the same bits
//! and share one state: the one-shot entry then runs with
//! `one_shot_capacity`. Decode graphs use `hyper_connection_decode` on a
//! state of their own, with the same bits.

use super::{Epilogue, Input};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, InstantiationSpec, LaunchPlan,
    MAX_COMM_RANKS, OpFamily, Refusal, SigmoidMeanHyperConnectionQuery, StaticQuery, Target,
    Validate,
};
use std::path::PathBuf;

/// Largest row count of the one-shot entry on a shared state; the device
/// inbox layout (`OneShotRows` in `comm/hyper_connection.cuh`) assumes it.
pub const ONE_SHOT_ROWS: u32 = 32;
/// Rank counts the entries and their contract accept.
pub const RANKS: [u32; 3] = [2, 4, 8];
/// The entries build the epilogue from these arguments (`HcInjectNorm`'s
/// members in order).
pub(super) const EPILOGUE_ARGUMENTS: &str = "{residual, logits, weight, residual_out, normed}";
/// Most rows of one call: a prefill chunk.
pub const MAX_CAPACITY: u32 = 8192;

/// The epilogue and input type of a hyper-connection transport instance
/// (`qwen38::Geometry` in `comm/hyper_connection.cuh`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HcShape {
    pub epilogue: Epilogue,
    pub input: Input,
}

impl HcShape {
    /// Input elements per 16-byte vector.
    pub const fn per_vector(&self) -> u32 {
        16 / self.input.element_bytes
    }

    /// Input vectors per row.
    pub const fn vectors(&self) -> u32 {
        self.epilogue.width / self.per_vector()
    }

    /// The checks `qwen38::Geometry` makes at compile time: nonempty rows
    /// and streams, whole columns per 256-thread epilogue CTA, BF16 or F32
    /// input elements and whole 16-byte input vectors per row.
    pub fn validate(&self) -> Result<(), Refusal> {
        let (width, streams, bytes) = (
            self.epilogue.width,
            self.epilogue.streams,
            self.input.element_bytes,
        );
        if width == 0
            || streams == 0
            || (self.epilogue.is_hc_inject_norm()
                && streams != super::epilogue::HC_INJECT_NORM_STREAMS)
            || !width.is_multiple_of(256)
            || !matches!(bytes, 2 | 4)
            || !width.is_multiple_of(16 / bytes)
        {
            return Err(Refusal("unsupported hyper-connection comm geometry"));
        }
        Ok(())
    }

    /// Whether the decode entries' LL layout holds the shape
    /// (`qwen38_decode::Rows::kLlFits`): one CTA per stream, at most eight,
    /// each with whole vectors of at most one per thread. A shape that does
    /// not fit takes the one-shot at every row count.
    pub const fn low_latency_fits(&self) -> bool {
        let streams = self.epilogue.streams;
        streams <= 8 && self.vectors().is_multiple_of(streams) && self.vectors() / streams <= 256
    }
}

/// A model's hyper-connection transport: its shape, its grouped norm's
/// epsilon, and the names of its cubin and its one-shot, scatter and
/// gather entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HcTransport {
    pub shape: HcShape,
    /// The grouped norm's epsilon, as F32 bits.
    pub eps_bits: u32,
    pub cubin: &'static str,
    pub symbol: &'static str,
    pub scatter_symbol: &'static str,
    pub gather_symbol: &'static str,
}

impl HcTransport {
    /// The all-reduce at `ranks` ranks with the prefill capacity.
    pub fn query(&self, ranks: u32) -> SigmoidMeanHyperConnectionQuery {
        SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
            ranks,
            streams: self.shape.epilogue.streams,
            width: self.shape.epilogue.width,
            eps_bits: self.eps_bits,
            capacity: MAX_CAPACITY,
        }
    }
}

const CLUSTER: u32 = 8;
const MAX_WORKERS: u32 = 128;
/// 64-bit block flags of the phased pair: the scatter's last phase, and the
/// gather's per source.
const BLOCKS: u64 = 32;

fn ranks_of(query: SigmoidMeanHyperConnectionQuery) -> u32 {
    match query {
        SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm { ranks, .. } => ranks,
        _ => 0,
    }
}

/// `query`'s instantiation of `t`'s one-shot entry, refusing another
/// geometry, rank count or epsilon.
pub fn contract(
    t: &HcTransport,
    query: SigmoidMeanHyperConnectionQuery,
) -> Result<InstantiationSpec, Refusal> {
    t.shape.validate()?;
    t.shape.epilogue.validate_hc_epsilon(t.eps_bits)?;
    let SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        ranks,
        streams,
        width,
        eps_bits,
        capacity: 1..=MAX_CAPACITY,
    } = query
    else {
        return Err(Refusal("unsupported hyper-connection comm geometry"));
    };
    if (streams, width) != (t.shape.epilogue.streams, t.shape.epilogue.width) {
        return Err(Refusal("unsupported hyper-connection comm geometry"));
    }
    if !RANKS.contains(&ranks) {
        return Err(Refusal("hyper-connection comm takes 2, 4 or 8 ranks"));
    }
    if eps_bits != t.eps_bits {
        return Err(Refusal("unsupported grouped norm epsilon"));
    }
    let spec = InstantiationSpec {
        family: OpFamily::Comm,
        key: CompileKey {
            query: StaticQuery::HyperConnection(query),
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
        symbol: t.symbol.into(),
    };
    spec.validate()?;
    Ok(spec)
}

pub fn plan(
    t: &HcTransport,
    query: SigmoidMeanHyperConnectionQuery,
    live: u32,
) -> Result<LaunchPlan, Refusal> {
    contract(t, query)?;
    let SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm { capacity, .. } = query else {
        unreachable!()
    };
    if live > capacity {
        return Err(Refusal("hyper-connection comm live rows exceed capacity"));
    }
    let plan = LaunchPlan {
        grid: [8, 1, 1],
        block: [256, 1, 1],
        cluster: [8, 1, 1],
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
    pub flags_bytes: u64,
    pub epoch_bytes: u64,
}

/// Flags and epoch start at zero. Inboxes need no initialization. Capacity
/// stays fixed for the lifetime of this state, including mixed-shape graphs.
/// The inbox holds the one-shot banks for `ONE_SHOT_ROWS` rows, the prefill
/// scatter slots and the F32 prefill sums, each with a slot per rank. The
/// flags hold the one-shot entry's 2 * ranks words, the scatter's 64-bit
/// flags (2 * ranks per source and 32 per block), its arrival counter and
/// 28 spare bytes, then the gather's 64-bit flags (32 blocks per source):
/// 1408 bytes at four ranks. The epoch holds the one-shot word and the
/// pair's 64-bit generation at byte 8.
pub fn buffers(
    t: &HcTransport,
    query: SigmoidMeanHyperConnectionQuery,
) -> Result<Buffers, Refusal> {
    contract(t, query)?;
    let SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        ranks, capacity, ..
    } = query
    else {
        unreachable!()
    };
    Ok(buffers_with(t.shape, ranks, capacity))
}

/// `buffers` for another shape: the one-shot banks and scatter slots hold
/// its input elements, the sums F32.
pub fn buffers_with(shape: HcShape, ranks: u32, capacity: u32) -> Buffers {
    let (ranks, width) = (u64::from(ranks), u64::from(shape.epilogue.width));
    let element = u64::from(shape.input.element_bytes);
    let one_shot = 2 * ranks * u64::from(ONE_SHOT_ROWS) * width * element;
    let scatter = ranks * u64::from(capacity).div_ceil(ranks) * width * element;
    let sums = u64::from(capacity) * width * 4;
    // `Layout` in hyper_connection.cuh: the u32 arrival counter follows ranks + 2 ranks
    // + 32 64-bit slots, then 28 spare bytes and the gather's ranks * 32 slots.
    let flags_bytes = 8 * (3 * ranks + BLOCKS) + 32 + ranks * BLOCKS * 8;
    Buffers {
        inbox_bytes: one_shot + scatter + sums,
        flags_bytes,
        epoch_bytes: 16,
    }
}

/// The capacity argument of the one-shot entry on a state from `buffers`.
pub fn one_shot_capacity(
    t: &HcTransport,
    query: SigmoidMeanHyperConnectionQuery,
) -> Result<u32, Refusal> {
    contract(t, query)?;
    let SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm { capacity, .. } = query else {
        unreachable!()
    };
    Ok(capacity.min(ONE_SHOT_ROWS))
}

/// Whether `rows` prepared rows take the phased prefill pair.
pub fn phased(rows: u32) -> bool {
    rows > ONE_SHOT_ROWS
}

/// Launch plan of both prefill entries for up to `rows` live rows: one
/// sending cluster and a worker CTA per row, 8 to 128 workers. Every CTA
/// fits at once beside the sender, which the workers' waits need.
pub fn prefill_plan(
    t: &HcTransport,
    query: SigmoidMeanHyperConnectionQuery,
    rows: u32,
) -> Result<LaunchPlan, Refusal> {
    contract(t, query)?;
    let SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm { capacity, .. } = query else {
        unreachable!()
    };
    if rows > capacity {
        return Err(Refusal("hyper-connection comm live rows exceed capacity"));
    }
    let workers = rows.next_multiple_of(CLUSTER).clamp(CLUSTER, MAX_WORKERS);
    let plan = LaunchPlan {
        grid: [CLUSTER + workers, 1, 1],
        block: [256, 1, 1],
        cluster: [CLUSTER, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
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
    pub epoch: u64,
    pub live_rows: u64,
    pub capacity: u32,
    pub rank: u32,
}

/// The call's rank count: one inbox and one flags slot per rank.
fn call_ranks(call: &Call) -> Result<u32, Refusal> {
    let ranks = u32::try_from(call.inboxes.len()).unwrap_or(0);
    if !RANKS.contains(&ranks)
        || call.flags.len() != call.inboxes.len()
        || call.rank >= ranks
        || call.capacity == 0
        || call.capacity > MAX_CAPACITY
    {
        return Err(Refusal("invalid hyper-connection comm call"));
    }
    Ok(ranks)
}

fn check(call: &Call) -> Result<(), Refusal> {
    call_ranks(call).map(|_| ())
}

fn finish(mut args: Vec<Argument>, call: &Call) -> Result<ArgumentList, Refusal> {
    let ranks = call_ranks(call)?;
    args.extend([
        Argument::DevicePointerArray(call.inboxes.clone()),
        Argument::DevicePointerArray(call.flags.clone()),
        Argument::DeviceBuffer(call.epoch),
        Argument::DeviceBuffer(call.live_rows),
        Argument::U32(call.capacity),
        Argument::U32(call.rank),
        Argument::U32(ranks),
    ]);
    let args = ArgumentList(args);
    args.validate()?;
    Ok(args)
}

/// Arguments of the one-shot entry on a state from `buffers`: its capacity
/// must be at most `ONE_SHOT_ROWS`, or its banks would overlap the prefill
/// regions. `launches` picks it and its capacity by rows.
pub fn arguments(call: Call) -> Result<ArgumentList, Refusal> {
    if call.capacity > ONE_SHOT_ROWS {
        return Err(Refusal(
            "hyper-connection one-shot capacity above ONE_SHOT_ROWS on a shared state",
        ));
    }
    standalone_arguments(call)
}

/// Arguments of the one-shot entry on a state of its own at any capacity:
/// an inbox of `standalone_inbox_bytes(ranks, capacity)`, flags and epoch as in
/// `buffers`. Test oracles use it; the bundle never does.
pub fn standalone_arguments(call: Call) -> Result<ArgumentList, Refusal> {
    check(&call)?;
    let args = [
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
    finish(args, &call)
}

/// Arguments of the prefill scatter; `capacity` is the state's.
fn scatter_arguments(call: &Call) -> Result<ArgumentList, Refusal> {
    check(call)?;
    finish(vec![Argument::DeviceBuffer(call.input)], call)
}

/// Arguments of the prefill gather; `capacity` is the state's.
fn gather_arguments(call: &Call) -> Result<ArgumentList, Refusal> {
    check(call)?;
    let args = [
        call.residual,
        call.logits,
        call.weight,
        call.residual_out,
        call.normed,
    ]
    .into_iter()
    .map(Argument::DeviceBuffer)
    .collect();
    finish(args, call)
}

/// Inbox bytes of a one-shot state of its own of `shape` at `capacity`
/// rows: two banks of `ranks` capacity-row slots.
pub fn standalone_inbox_bytes(shape: HcShape, ranks: u32, capacity: u32) -> u64 {
    2 * u64::from(ranks)
        * u64::from(capacity)
        * u64::from(shape.epilogue.width)
        * u64::from(shape.input.element_bytes)
}

/// One kernel launch of an all-reduce call.
#[derive(Clone, Debug)]
pub struct Launch {
    pub symbol: &'static str,
    pub plan: LaunchPlan,
    pub arguments: ArgumentList,
}

/// The launches of one all-reduce call of up to `rows` live rows on a state
/// from `buffers(query)`; `call.capacity` is the state's. Up to
/// `ONE_SHOT_ROWS` rows: the one-shot entry at `one_shot_capacity`. More:
/// the phased scatter and gather with `prefill_plan`. Both give the same
/// bits.
pub fn launches(
    t: &HcTransport,
    query: SigmoidMeanHyperConnectionQuery,
    rows: u32,
    call: Call,
) -> Result<Vec<Launch>, Refusal> {
    contract(t, query)?;
    let SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm { capacity, .. } = query else {
        unreachable!()
    };
    if call.capacity != capacity {
        return Err(Refusal(
            "hyper-connection comm call capacity differs from its state's",
        ));
    }
    if call_ranks(&call)? != ranks_of(query) {
        return Err(Refusal(
            "hyper-connection comm call rank count differs from its query's",
        ));
    }
    if !phased(rows) {
        let call = Call {
            capacity: one_shot_capacity(t, query)?,
            ..call
        };
        return Ok(vec![Launch {
            symbol: t.symbol,
            plan: plan(t, query, rows)?,
            arguments: arguments(call)?,
        }]);
    }
    let plan = prefill_plan(t, query, rows)?;
    Ok(vec![
        Launch {
            symbol: t.scatter_symbol,
            plan,
            arguments: scatter_arguments(&call)?,
        },
        Launch {
            symbol: t.gather_symbol,
            plan,
            arguments: gather_arguments(&call)?,
        },
    ])
}

/// `t`'s cubin: the one-shot entry and the phased pair instantiated with
/// its shape, input and inbox pointers of its input elements.
pub fn cubin_specs(t: &HcTransport) -> Vec<CubinSpec> {
    let shape = t.shape;
    shape
        .validate()
        .expect("hyper-connection comm shape the transports take");
    let element = shape.input.cxx_type;
    let input_type = format!("const {element}");
    let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let tail = || {
        vec![
            EntryParam::PointerArray {
                name: "inboxes".into(),
                slots: MAX_COMM_RANKS as u8,
                cxx_type: element.into(),
            },
            EntryParam::PointerArray {
                name: "flags".into(),
                slots: MAX_COMM_RANKS as u8,
                cxx_type: "uint32_t".into(),
            },
            ptr("epoch", "uint32_t"),
            ptr("live_rows", "const uint32_t"),
            EntryParam::U32 {
                name: "capacity".into(),
            },
            EntryParam::U32 {
                name: "rank".into(),
            },
            EntryParam::U32 {
                name: "ranks".into(),
            },
        ]
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
    params.extend([ptr("residual_out", "uint16_t"), ptr("normed", "uint16_t")]);
    params.extend(tail());
    let mut scatter = vec![ptr("input", &input_type)];
    scatter.extend(tail());
    let mut gather: Vec<_> = ["residual", "logits", "weight"]
        .into_iter()
        .map(|n| {
            ptr(
                n,
                if n == "weight" {
                    shape.epilogue.weight_pointer_type()
                } else {
                    "const uint16_t"
                },
            )
        })
        .collect();
    gather.extend([ptr("residual_out", "uint16_t"), ptr("normed", "uint16_t")]);
    gather.extend(tail());
    let e = format!("{}, {}", shape.epilogue.functor(), shape.input.policy);
    vec![CubinSpec {
        name: t.cubin.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![
            "#include \"comm/hyper_connection.cuh\"".into(),
            shape.epilogue.include(),
            shape.input.include(),
        ],
        // push.cuh's helpers need the comm tuning defines.
        defines: super::push_defines(),
        entries: vec![
            EntrySpec::new(
                t.symbol.into(),
                params,
                format!(
                    "chop::comm::qwen38::allreduce_hc<{e}>(input, {EPILOGUE_ARGUMENTS}, inboxes, flags, epoch, live_rows, capacity, rank, ranks);"
                ),
                256,
                1,
            ),
            EntrySpec::new(
                t.scatter_symbol.into(),
                scatter,
                format!(
                    "chop::comm::qwen38::reduce_scatter<{e}>(input, inboxes, flags, epoch, live_rows, capacity, rank, ranks);"
                ),
                256,
                1,
            ),
            EntrySpec::new(
                t.gather_symbol.into(),
                gather,
                format!(
                    "chop::comm::qwen38::allgather_hc_v2<{e}>({EPILOGUE_ARGUMENTS}, inboxes, flags, epoch, live_rows, capacity, rank, ranks);"
                ),
                256,
                1,
            ),
        ],
    }]
}
