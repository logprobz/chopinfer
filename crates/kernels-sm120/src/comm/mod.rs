//! PCIe peer communication plans and cubin instantiations.

mod catalog;
pub mod epilogue;
pub mod hyper_connection;
pub mod hyper_connection_decode;
pub mod input;
pub mod ll_bf16;
pub mod ll_f32;
pub mod norm_weights;
pub mod owned_sum_bf16;
pub mod owned_sum_f32;
pub mod phased_bf16;
pub mod phased_f32;
pub mod phased_turns_bf16;
pub mod sinkhorn_epilogue;
pub mod sinkhorn_split;
pub mod sinkhorn_split_tree;

pub use catalog::{CATALOG_RANKS, Entry, Shape, catalog, own_catalog_for};
pub use epilogue::{Epilogue, GroupedRmsNorm, NormReduction};
pub use input::Input;

use crate::invariance::Invariance;

use kernel_core::contract::{
    Argument, ArgumentList, CommQuery, DType, LaunchPlan, MAX_COMM_RANKS, Refusal, RmsNormEpilogue,
    RmsNormKind, Validate,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The epilogue `comm_sm120`'s fused entries instantiate the transport
/// with: the residual add and HF RMSNorm of 4096-wide rows. Its width sets their row
/// geometry (`push::Rows` in `push.cuh`) and the planners' rows here. A
/// fused query with any other width is refused, never normalized as rows of
/// this width.
pub const FUSED_EPILOGUE: Epilogue = epilogue::RESIDUAL_RMSNORM_4096;
const FUSED_WIDTH: u32 = FUSED_EPILOGUE.width;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Variant {
    OneShot,
    FusedOneShot,
    FusedTwoShot,
    /// The wide push two-shot of decode and verify graphs (`graph_choose`).
    FusedWide,
    TwoShot,
    AllGather,
}

#[derive(Deserialize)]
struct TuningTable {
    targets: BTreeMap<String, ShapeBuckets>,
}
#[derive(Deserialize)]
struct ShapeBuckets {
    decode: Bucket,
    prefill: Bucket,
    push: PushTuning,
}
#[derive(Deserialize)]
struct Bucket {
    one_shot_max_bytes: u64,
    threads: u32,
}
/// Protocol thresholds of the fused BF16 push path. The cubin receives the
/// same values as defines, so kernel dispatch and launch geometry agree.
#[derive(Clone, Deserialize)]
struct PushTuning {
    low_latency_cluster: u32,
    cluster: u32,
    /// Rows per flagged block of the wide protocol's last phases (blocks
    /// come in multiples of `cluster`); the cubin receives it as a define.
    wide_block_rows: u32,
    /// Most worker CTAs of a wide call; the kernel takes any count.
    wide_workers: u32,
    /// Row thresholds for every rank count from 1 to `MAX_COMM_RANKS`.
    ranks: BTreeMap<u32, RankThresholds>,
}

/// Rows up to `low_latency_max_rows` take the low-latency protocol, up to
/// `one_shot_max_rows` the one-shot, and more the two-shot. Low latency needs
/// a power-of-two rank count and two-shot one that divides a row's 512
/// vectors; rank counts without them set 0 and `PUSH_MAX_ROWS`. Graphs send
/// rows above `PUSH_MAX_ROWS` up to `wide_max_rows` through the wide
/// two-shot (power-of-two ranks only; 0 for none) and more through the
/// phased pair.
#[derive(Clone, Copy, Deserialize)]
struct RankThresholds {
    low_latency_max_rows: u32,
    one_shot_max_rows: u32,
    wide_max_rows: u32,
}

impl PushTuning {
    fn thresholds(&self, ranks: u32) -> Result<RankThresholds, Refusal> {
        self.ranks
            .get(&ranks)
            .copied()
            .ok_or(Refusal("invalid fused push geometry"))
    }

    /// Row stride of the low-latency region: the most rows any rank count
    /// sends with it.
    fn low_latency_rows(&self) -> u32 {
        self.ranks
            .values()
            .map(|t| t.low_latency_max_rows)
            .max()
            .unwrap_or(0)
    }

    /// The fewest ranks that take the two-shot, which bounds the columns one
    /// owner reduces (`kPer` in `push.cuh`); `MAX_COMM_RANKS` if none does.
    fn two_shot_min_ranks(&self) -> u32 {
        self.ranks
            .iter()
            .filter(|(_, t)| u64::from(t.one_shot_max_rows) < PUSH_MAX_ROWS)
            .map(|(&ranks, _)| ranks)
            .min()
            .unwrap_or(MAX_RANKS as u32)
    }

    /// Flagged blocks of a wide call of `rows` rows: a multiple of the
    /// cluster, so every sending CTA owns one (`wide_blocks` in push.cuh).
    fn wide_blocks(&self, rows: u32) -> u32 {
        self.cluster * rows.div_ceil(self.cluster * self.wide_block_rows)
    }

    /// The first local word of the wide protocol (its ready word, then the
    /// arrival counter): after the reduce-scatter and all-gather words per
    /// source and the last phases' block words (push.cuh `kWideReady`).
    fn wide_local_words(&self) -> u64 {
        WIDE_FLAGS + 2 * MAX_RANKS + 2 * u64::from(self.wide_blocks(WIDE_MAX_ROWS as u32))
    }

    /// Flag words of the wide protocol: `wide_local_words`, its two local
    /// words, then a shape word and a shape flag per source (push.cuh
    /// `kWideShapeFlags` + `kMaxRanks`).
    fn wide_flag_words(&self) -> u64 {
        self.wide_local_words() + 2 + 2 * MAX_RANKS
    }

    fn validate(&self) -> Result<(), Refusal> {
        let keys: Vec<_> = self.ranks.keys().copied().collect();
        let expected: Vec<_> = (1..=MAX_RANKS as u32).collect();
        let valid = keys == expected
            && self.ranks.iter().all(|(&ranks, t)| {
                let low_latency = t.low_latency_max_rows == 0 || ranks.is_power_of_two();
                // Two-shot owners split a row's 512 vectors evenly.
                let two_shot = u64::from(t.one_shot_max_rows) >= PUSH_MAX_ROWS
                    || (ranks >= 2 && ranks.is_power_of_two());
                let wide = t.wide_max_rows == 0
                    || (ranks.is_power_of_two()
                        && u64::from(t.wide_max_rows) > PUSH_MAX_ROWS
                        && u64::from(t.wide_max_rows) <= WIDE_MAX_ROWS);
                low_latency
                    && two_shot
                    && wide
                    && t.low_latency_max_rows <= t.one_shot_max_rows
                    && u64::from(t.one_shot_max_rows) <= PUSH_MAX_ROWS
            })
            && self.cluster >= 1
            && self.cluster <= 8
            && self.wide_block_rows >= 1
            && self.wide_workers >= 1
            && self.wide_workers.is_multiple_of(self.cluster);
        if valid {
            Ok(())
        } else {
            Err(Refusal("invalid comm push tuning"))
        }
    }
}

/// The tuning of a device, from the comm tuning table, which is parsed
/// once: ops prepare thousands of row counts' choices and plans.
fn tuning(compute_capability: &str, sm_count: u32) -> Result<&'static ShapeBuckets, Refusal> {
    static TABLE: std::sync::OnceLock<Option<TuningTable>> = std::sync::OnceLock::new();
    TABLE
        .get_or_init(|| serde_json::from_str(include_str!("../../tuning/comm.json")).ok())
        .as_ref()
        .ok_or(Refusal("invalid comm tuning table"))?
        .targets
        .get(&format!("{compute_capability}:{sm_count}"))
        .ok_or(Refusal("unsupported comm tuning target"))
}

/// The comm engine's entries sum F32 or BF16 rows into rows of the same
/// type; the rounded all-reduce is the phased F32 entries' rounding forms'
/// (`phased_f32`).
const ROUNDED_REFUSAL: Refusal =
    Refusal("the comm engine does not round its sums; the phased F32 entries do");

pub fn choose_for(
    query: &CommQuery,
    compute_capability: &str,
    sm_count: u32,
) -> Result<Variant, Refusal> {
    query.validate()?;
    let target = tuning(compute_capability, sm_count)?;
    let elements = match *query {
        CommQuery::AllReduce { elements, .. } => elements,
        CommQuery::AllReduceRounded { .. } => return Err(ROUNDED_REFUSAL),
        CommQuery::AllGather {
            elements_per_rank, ..
        } => elements_per_rank,
    };
    let bucket = if elements <= 32 * 4096 {
        &target.decode
    } else {
        &target.prefill
    };
    if bucket.threads != 256 {
        return Err(Refusal("unsupported comm thread count"));
    }
    match *query {
        CommQuery::AllReduce {
            ranks,
            elements,
            dtype,
            fused_residual: false,
            fused_rmsnorm: None,
        } if u64::from(ranks) <= MAX_RANKS => {
            let bytes = elements.saturating_mul(if dtype == DType::Bf16 { 2 } else { 4 });
            Ok(if bytes <= bucket.one_shot_max_bytes {
                Variant::OneShot
            } else {
                Variant::TwoShot
            })
        }
        CommQuery::AllReduce {
            fused_rmsnorm: Some(norm),
            ..
        } if norm.kind != RmsNormKind::HfBf16Intermediate => {
            Err(Refusal("fused comm requires HF BF16-intermediate RMSNorm"))
        }
        CommQuery::AllReduce {
            fused_rmsnorm: Some(_),
            fused_residual: false,
            ..
        } => Err(Refusal("fused comm requires a residual")),
        CommQuery::AllReduce {
            ranks,
            elements,
            dtype: DType::Bf16,
            fused_residual: true,
            fused_rmsnorm:
                Some(RmsNormEpilogue {
                    kind: RmsNormKind::HfBf16Intermediate,
                    width: FUSED_WIDTH,
                    ..
                }),
        } if u64::from(ranks) <= MAX_RANKS => {
            Ok(if elements.saturating_mul(2) <= bucket.one_shot_max_bytes {
                Variant::FusedOneShot
            } else {
                Variant::FusedTwoShot
            })
        }
        CommQuery::AllGather { ranks, .. } if u64::from(ranks) <= MAX_RANKS => {
            Ok(Variant::AllGather)
        }
        CommQuery::AllReduce {
            fused_residual: true,
            ..
        } => Err(Refusal(
            "unsupported fused comm dtype, row width or epilogue",
        )),
        _ => Err(Refusal("comm supports at most MAX_COMM_RANKS ranks")),
    }
}

pub fn choose(query: &CommQuery) -> Result<Variant, Refusal> {
    choose_for(query, "12.0", 188)
}

/// Protocol behind `chop_comm_allreduce_oneshot_norm_bf16_v2` (up to 32
/// rows, `push_protocol`) and the wide entries (`graph_protocol`). Every
/// path sends to one peer at a time; see `cuda/comm/push.cuh`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PushProtocol {
    /// Recursive doubling with flags inside 8-byte data entries.
    LowLatency,
    /// Full input to each peer in turn, with a fence between peers.
    OneShot,
    /// Reduce-scatter and all-gather, one peer at a time.
    TwoShot,
    /// Reduce-scatter and all-gather for up to `WIDE_MAX_ROWS` rows, whose
    /// last phases flag blocks of rows (`chop_comm_allreduce_wide_*`).
    Wide,
}

fn push_tuning() -> Result<PushTuning, Refusal> {
    let push = tuning("12.0", 188)?.push.clone();
    push.validate()?;
    Ok(push)
}

/// The protocol of the fused push entries for `rows` rows, at most 32;
/// above, prefill takes the phased pair (`choose`) and graphs the wide
/// protocol (`graph_protocol`).
pub fn push_protocol(ranks: u32, rows: u32) -> Result<PushProtocol, Refusal> {
    if ranks == 0 || u64::from(ranks) > MAX_RANKS || rows == 0 {
        return Err(Refusal("invalid fused push geometry"));
    }
    // The device traps above this row count; refuse it here as well.
    if u64::from(rows) > PUSH_MAX_ROWS {
        return Err(Refusal("fused push supports at most 32 rows"));
    }
    let thresholds = push_tuning()?.thresholds(ranks)?;
    Ok(if rows <= thresholds.low_latency_max_rows {
        PushProtocol::LowLatency
    } else if rows <= thresholds.one_shot_max_rows {
        PushProtocol::OneShot
    } else {
        PushProtocol::TwoShot
    })
}

/// Most rows of any wide call; the inbox and flag layout hold this many.
pub fn wide_max_rows(ranks: u32) -> Result<u32, Refusal> {
    Ok(push_tuning()?.thresholds(ranks)?.wide_max_rows)
}

/// The protocol of a decode or verify graph's hidden all-reduce of `rows`
/// rows (`graph_choose`): the push protocols up to 32 rows, the wide
/// protocol above, up to the rank count's `wide_max_rows`.
pub fn graph_protocol(ranks: u32, rows: u32) -> Result<PushProtocol, Refusal> {
    if u64::from(rows) <= PUSH_MAX_ROWS {
        return push_protocol(ranks, rows);
    }
    if ranks == 0 || u64::from(ranks) > MAX_RANKS || rows > wide_max_rows(ranks)? {
        return Err(Refusal("graph rows exceed the wide protocol"));
    }
    Ok(PushProtocol::Wide)
}

/// Per-rank buffers of `graph_protocol`'s protocol for `rows` rows:
/// `push_buffers` up to 32 rows, the wide protocol's above.
pub fn graph_buffers(ranks: u32, rows: u32) -> Result<PushBuffers, Refusal> {
    if graph_protocol(ranks, rows)? != PushProtocol::Wide {
        return push_buffers(ranks, rows);
    }
    // Slots of all ranks hold WIDE_MAX_ROWS full rows at any rank count;
    // the wide protocol uses only the first epoch cell.
    let inbox_bytes = WIDE_SCATTER_BASE + WIDE_MAX_ROWS * ROW_BYTES;
    Ok(PushBuffers {
        inbox_bytes,
        // Any state may take low-latency row counts.
        zeroed_inbox_bytes: inbox_bytes.min(ONE_SHOT_BASE),
        flag_bytes: push_tuning()?.wide_flag_words() * 4,
        epoch_bytes: 4,
    })
}

/// Per-rank buffers of the fused push path. Flags, epochs and the first
/// `zeroed_inbox_bytes` of every inbox start zeroed; the rest of the inbox
/// may hold any bytes. That prefix is the low-latency region: its entries
/// carry their flag in the data, so a stale entry whose bytes happen to hold
/// the flag of an upcoming call would be taken for the peer's data. The
/// protocol clears every entry it reads, so the region stays valid once
/// zeroed. Buffers can be shared with other comm protocols as long as calls
/// on them are ordered on every rank.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PushBuffers {
    pub inbox_bytes: u64,
    pub zeroed_inbox_bytes: u64,
    pub flag_bytes: u64,
    pub epoch_bytes: u64,
}

const ROW_BYTES: u64 = FUSED_EPILOGUE.width as u64 * 2;
const MAX_RANKS: u64 = MAX_COMM_RANKS as u64;
// Entries declare one pointer slot per rank.
const _: () = assert!(kernel_build::MAX_POINTER_ARRAY_SLOTS as usize == MAX_COMM_RANKS);
/// Largest row count of the fused push path (FusedOneShot, 256 KiB).
const PUSH_MAX_ROWS: u64 = 32;
/// Largest row count of the wide two-shot, which sizes its slots and flags.
pub const WIDE_MAX_ROWS: u64 = 256;
/// Every protocol owns a fixed byte region of the inbox, so a peer already in
/// a call with another protocol never overwrites data a rank still reads.
/// The low-latency region starts at zero.
const ONE_SHOT_BASE: u64 = 1 << 20;
const TWO_SHOT_BASE: u64 = ONE_SHOT_BASE + 2 * MAX_RANKS * PUSH_MAX_ROWS * ROW_BYTES;
const PHASED_BASE: u64 = 8 << 20;
/// Flag words of the phased two-shot sit above the push protocol's words.
const PHASED_FLAG_WORDS: u64 = 82;
/// Two-shot flag words follow the phased ones, so no push protocol shares a
/// peer-written flag word with another (see `push.cuh`).
const TWO_SHOT_FLAGS: u64 = PHASED_FLAG_WORDS;
/// The wide protocol's all-gather slots sit between the two-shot and phased
/// regions (it reads them last, so no other protocol may write there); its
/// reduce-scatter slots overlap the phased region only, which no protocol
/// reads after it sends. Its flag words follow the two-shot's.
const WIDE_GATHER_BASE: u64 = TWO_SHOT_BASE + 2 * PUSH_MAX_ROWS * ROW_BYTES;
const WIDE_SCATTER_BASE: u64 = PHASED_BASE;
const WIDE_FLAGS: u64 = TWO_SHOT_FLAGS + 2 * MAX_RANKS;
const _: () = assert!(WIDE_GATHER_BASE + WIDE_MAX_ROWS * ROW_BYTES <= PHASED_BASE);

pub fn push_buffers(ranks: u32, rows: u32) -> Result<PushBuffers, Refusal> {
    let (ranks64, rows64) = (u64::from(ranks), u64::from(rows));
    let low_latency_rows = u64::from(push_tuning()?.low_latency_rows());
    // Low latency doubles every row into flagged entries; both it and the
    // one-shot double-buffer by epoch parity with the largest row count.
    let (inbox_bytes, flag_words) = match push_protocol(ranks, rows)? {
        PushProtocol::LowLatency => (2 * ranks64 * low_latency_rows * 2 * ROW_BYTES, 0),
        PushProtocol::OneShot => (
            ONE_SHOT_BASE + 2 * ranks64 * PUSH_MAX_ROWS * ROW_BYTES,
            2 * ranks64 + 2,
        ),
        PushProtocol::TwoShot => (
            TWO_SHOT_BASE + 2 * PUSH_MAX_ROWS * ROW_BYTES,
            TWO_SHOT_FLAGS + 2 * ranks64,
        ),
        PushProtocol::Wide => unreachable!("push_protocol never answers wide"),
    };
    Ok(PushBuffers {
        inbox_bytes,
        // Any state may take low-latency row counts.
        zeroed_inbox_bytes: inbox_bytes.min(ONE_SHOT_BASE),
        flag_bytes: flag_words * 4,
        // Low latency keeps an epoch cell per row; the bulk protocols use
        // the first.
        epoch_bytes: rows64 * 4,
    })
}

/// Per-rank buffers of the phased reduce-scatter and gather stages
/// (`TwoShot` and `FusedTwoShot`). Flags and the epoch start zeroed.
pub fn phased_buffers(ranks: u32, elements: u64, dtype: DType) -> Result<PushBuffers, Refusal> {
    if ranks == 0 || u64::from(ranks) > MAX_RANKS || elements == 0 {
        return Err(Refusal("invalid phased comm geometry"));
    }
    let element_bytes = match dtype {
        DType::Bf16 => 2,
        DType::F32 => 4,
        _ => return Err(Refusal("phased comm supports BF16 and F32")),
    };
    let inbox_bytes = elements
        .div_ceil(4096)
        .div_ceil(u64::from(ranks))
        .checked_mul(4096)
        .and_then(|slot| slot.checked_mul(u64::from(ranks)))
        .and_then(|slots| slots.checked_mul(element_bytes))
        .and_then(|bytes| bytes.checked_add(PHASED_BASE))
        .ok_or(Refusal("phased comm inbox overflows u64"))?;
    Ok(PushBuffers {
        inbox_bytes,
        zeroed_inbox_bytes: 0,
        flag_bytes: PHASED_FLAG_WORDS * 4,
        epoch_bytes: 4,
    })
}

/// Buffers of one comm state that takes fused hidden all-reduces of any row
/// count from 1 to `capacity` at `ranks` ranks: the push protocols the
/// tuning picks up to 32 rows and the phased pair above, each field the
/// largest any of them needs. A bundle sizes a state's inbox span, flags and
/// epoch cells from this rather than a per-row budget, which does not hold
/// at every rank count (one-shot needs 18 flag words at 8 ranks, two-shot
/// 98, whatever the row count).
pub fn fused_state_buffers(ranks: u32, capacity: u32) -> Result<PushBuffers, Refusal> {
    if capacity == 0 {
        return Err(Refusal("comm state capacity must be nonzero"));
    }
    let mut needed = PushBuffers {
        inbox_bytes: 0,
        zeroed_inbox_bytes: 0,
        flag_bytes: 0,
        epoch_bytes: 0,
    };
    let mut take = |buffers: PushBuffers| {
        needed.inbox_bytes = needed.inbox_bytes.max(buffers.inbox_bytes);
        needed.zeroed_inbox_bytes = needed.zeroed_inbox_bytes.max(buffers.zeroed_inbox_bytes);
        needed.flag_bytes = needed.flag_bytes.max(buffers.flag_bytes);
        needed.epoch_bytes = needed.epoch_bytes.max(buffers.epoch_bytes);
    };
    for rows in 1..=capacity.min(PUSH_MAX_ROWS as u32) {
        take(push_buffers(ranks, rows)?);
    }
    // Graphs send rows above 32 through the wide protocol where the tuning
    // lists it, and prefill through the phased pair.
    if u64::from(capacity) > PUSH_MAX_ROWS && wide_max_rows(ranks)? > PUSH_MAX_ROWS as u32 {
        take(graph_buffers(ranks, PUSH_MAX_ROWS as u32 + 1)?);
    }
    if u64::from(capacity) > PUSH_MAX_ROWS {
        take(phased_buffers(
            ranks,
            u64::from(capacity) * u64::from(FUSED_EPILOGUE.width),
            DType::Bf16,
        )?);
    }
    Ok(needed)
}

/// The launch plan of `graph_protocol`'s protocol; up to 32 rows the push
/// entries' plan.
fn push_plan(ranks: u32, rows: u32) -> Result<LaunchPlan, Refusal> {
    let push = push_tuning()?;
    let (grid, cluster) = match graph_protocol(ranks, rows)? {
        PushProtocol::LowLatency => (rows * push.low_latency_cluster, push.low_latency_cluster),
        PushProtocol::OneShot | PushProtocol::TwoShot => (
            push.cluster * (1 + rows.div_ceil(push.cluster)),
            push.cluster,
        ),
        // The sending cluster and up to `wide_workers` normalizing CTAs.
        PushProtocol::Wide => (
            push.cluster + rows.min(push.wide_workers).next_multiple_of(push.cluster),
            push.cluster,
        ),
    };
    let buffers = graph_buffers(ranks, rows)?;
    let result = LaunchPlan {
        grid: [grid, 1, 1],
        block: [256, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: buffers.inbox_bytes + buffers.flag_bytes + buffers.epoch_bytes,
        cluster: [cluster, 1, 1],
        pdl: true,
    };
    result.validate()?;
    Ok(result)
}

/// The variant of a fused hidden all-reduce in a decode or verify graph:
/// `choose`'s, except that rows above the push path's 32 take the wide
/// two-shot up to the tuning's `wide_max_rows` for the rank count. Prefill
/// keeps `choose` (the phased pair above 32 rows).
pub fn graph_choose(query: &CommQuery) -> Result<Variant, Refusal> {
    let variant = choose(query)?;
    if let (
        Variant::FusedTwoShot,
        CommQuery::AllReduce {
            ranks, elements, ..
        },
    ) = (variant, *query)
        && elements <= u64::from(wide_max_rows(ranks)?) * u64::from(FUSED_EPILOGUE.width)
    {
        return Ok(Variant::FusedWide);
    }
    Ok(variant)
}

pub fn plan(query: &CommQuery, rows: u32) -> Result<LaunchPlan, Refusal> {
    plan_variant(choose(query)?, query, rows)
}

/// The plan of `graph_choose`'s variant.
pub fn graph_plan(query: &CommQuery, rows: u32) -> Result<LaunchPlan, Refusal> {
    plan_variant(graph_choose(query)?, query, rows)
}

/// The plan of `variant` for `query`, for a caller that picks the variant
/// itself (a bundle with its own threshold between the F32 one-shot and the
/// phased pair).
pub fn plan_variant(variant: Variant, query: &CommQuery, rows: u32) -> Result<LaunchPlan, Refusal> {
    if rows == 0 {
        return Err(Refusal("comm row count must be nonzero"));
    }
    if let (
        Variant::TwoShot | Variant::FusedTwoShot,
        CommQuery::AllReduce {
            ranks,
            elements,
            dtype,
            ..
        },
    ) = (variant, *query)
    {
        // One sending cluster plus up to seven clusters of reducing and
        // normalizing workers; both stages use this plan.
        let push = push_tuning()?;
        let tiles = elements.div_ceil(4096);
        let worker_clusters =
            u32::try_from(tiles.div_ceil(u64::from(push.cluster)).clamp(1, 7)).unwrap_or(7);
        let buffers = phased_buffers(ranks, elements, dtype)?;
        let workspace_bytes = buffers
            .inbox_bytes
            .checked_add(buffers.flag_bytes + buffers.epoch_bytes)
            .ok_or(Refusal("comm workspace overflows u64"))?;
        let result = LaunchPlan {
            grid: [push.cluster * (1 + worker_clusters), 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes,
            cluster: [push.cluster, 1, 1],
            pdl: true,
        };
        result.validate()?;
        return Ok(result);
    }
    if let (
        Variant::FusedOneShot | Variant::FusedWide,
        CommQuery::AllReduce {
            ranks, elements, ..
        },
    ) = (variant, *query)
    {
        let tokens = u32::try_from(elements / u64::from(FUSED_EPILOGUE.width))
            .map_err(|_| Refusal("comm grid exceeds u32"))?;
        return push_plan(ranks, tokens);
    }
    let grid = match *query {
        CommQuery::AllReduce { elements, .. } => {
            let tiles = elements.div_ceil(4096);
            u32::try_from(tiles).map_err(|_| Refusal("comm grid exceeds u32"))?
        }
        CommQuery::AllReduceRounded { .. } => return Err(ROUNDED_REFUSAL),
        CommQuery::AllGather {
            elements_per_rank, ..
        } => {
            let tiles = elements_per_rank.div_ceil(4096);
            u32::try_from(tiles).map_err(|_| Refusal("comm grid exceeds u32"))?
        }
    };
    let result = LaunchPlan {
        grid: [grid.max(rows), 1, 1],
        block: [256, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: workspace_bytes(query, u64::from(grid))?,
        cluster: [1, 1, 1],
        pdl: true,
    };
    result.validate()?;
    Ok(result)
}

fn workspace_bytes(query: &CommQuery, tiles: u64) -> Result<u64, Refusal> {
    let (ranks, inbox_bytes) = match *query {
        CommQuery::AllReduce {
            ranks,
            elements,
            dtype,
            ..
        } => {
            let element_bytes = if dtype == DType::Bf16 { 2 } else { 4 };
            let inbox = u64::from(ranks)
                .checked_mul(elements)
                .and_then(|v| v.checked_mul(element_bytes))
                .ok_or(Refusal("comm workspace overflows u64"))?;
            (ranks, inbox)
        }
        CommQuery::AllReduceRounded { .. } => return Err(ROUNDED_REFUSAL),
        CommQuery::AllGather { ranks, .. } => (ranks, 0),
    };
    let flags = u64::from(ranks)
        .checked_mul(tiles)
        .and_then(|v| v.checked_mul(2 * 4))
        .ok_or(Refusal("comm flag workspace overflows u64"))?;
    inbox_bytes
        .checked_add(flags)
        .and_then(|v| v.checked_add(tiles * 4))
        .ok_or(Refusal("comm workspace overflows u64"))
}

/// Rows gathered to one rank by `chop_comm_gather_rows_{bf16,f32}_v1`.
/// Rank q's shard of row r lands at `outputs[target] + (r * ranks + q) *
/// shard`, the row-major layout of full vocabulary rows. `row_indices` names a
/// device array of `rows` u32 row indices; `None` gathers rows 0..rows.
/// Handles are resolved by the caller; slot order is rank order.
pub struct GatherRowsCall {
    pub input: u64,
    pub outputs: Vec<u64>,
    pub flags: Vec<u64>,
    pub epoch: u64,
    pub row_indices: Option<u64>,
    pub rows: u32,
    pub shard: u64,
    pub rank: u32,
    pub ranks: u32,
    pub target: u32,
}

/// Eight independent CTAs of 256 threads. Each needs its own epoch cell and
/// the flags hold one word per (source, CTA) plus one per CTA; both start
/// zeroed and must not be shared with the `allgather` entries.
pub const GATHER_ROWS_EPOCH_BYTES: u64 = GATHER_ROWS_CTAS * 4;
pub const GATHER_ROWS_FLAG_BYTES: u64 = (MAX_RANKS * GATHER_ROWS_CTAS + GATHER_ROWS_CTAS) * 4;
const GATHER_ROWS_CTAS: u64 = 8;

pub fn gather_rows_plan() -> LaunchPlan {
    LaunchPlan {
        grid: [8, 1, 1],
        block: [256, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: GATHER_ROWS_EPOCH_BYTES + GATHER_ROWS_FLAG_BYTES,
        cluster: [1, 1, 1],
        pdl: true,
    }
}

pub fn gather_rows_arguments(call: GatherRowsCall) -> Result<ArgumentList, Refusal> {
    let ranks = call.ranks as usize;
    if call.ranks == 0
        || ranks > MAX_COMM_RANKS
        || call.rank >= call.ranks
        || call.target >= call.ranks
        || call.outputs.len() != ranks
        || call.flags.len() != ranks
        || call.rows == 0
        || call.shard == 0
    {
        return Err(Refusal("invalid row gather geometry"));
    }
    let args = ArgumentList(vec![
        Argument::DeviceBuffer(call.input),
        Argument::DevicePointerArray(call.outputs),
        Argument::DevicePointerArray(call.flags),
        Argument::DeviceBuffer(call.epoch),
        match call.row_indices {
            Some(handle) => Argument::DeviceBuffer(handle),
            None => Argument::NullDevicePointer,
        },
        Argument::U32(call.rows),
        Argument::U64(call.shard),
        Argument::U32(call.rank),
        Argument::U32(call.ranks),
        Argument::U32(call.target),
    ]);
    args.validate()?;
    Ok(args)
}

// The model-neutral row-exchange contract is also used by bundle mechanics.
pub use kernel_core::row_exchange::{
    EXCHANGE_ROWS_EPOCH_BYTES, EXCHANGE_ROWS_FLAG_BYTES, ExchangeRowsCall, exchange_rows_arguments,
    exchange_rows_owned, exchange_rows_plan,
};

/// Handles are resolved by kernel-lab before launch. Slot order is rank order.
pub struct OneShotCall {
    pub input: u64,
    pub output: u64,
    pub peer_inboxes: Vec<u64>,
    pub peer_flags: Vec<u64>,
    pub epoch: u64,
    pub elements: u64,
    pub rank: u32,
    pub ranks: u32,
}

pub fn arguments(call: OneShotCall) -> Result<ArgumentList, Refusal> {
    let args = ArgumentList(vec![
        Argument::DeviceBuffer(call.input),
        Argument::DeviceBuffer(call.output),
        Argument::DevicePointerArray(call.peer_inboxes),
        Argument::DevicePointerArray(call.peer_flags),
        Argument::DeviceBuffer(call.epoch),
        Argument::U64(call.elements),
        Argument::U32(call.rank),
        Argument::U32(call.ranks),
    ]);
    args.validate()?;
    Ok(args)
}

/// The phased reduce-scatter of a fused prefill all-reduce above the push
/// path's rows: it adds the residual to the rank-ordered F32 sum before its
/// one rounding, as the push and wide entries do.
pub const PREFILL_ROUND_ONCE_SYMBOL: &str = "chop_comm_reduce_scatter_residual_bf16_v1";
/// The fused gather after [`PREFILL_ROUND_ONCE_SYMBOL`]: it sends the
/// owner's pre-norm sums as they are and normalizes every row in the
/// low-latency order, so its rows are bitwise the verify entries'.
pub const PREFILL_ROUND_ONCE_GATHER_SYMBOL: &str = "chop_comm_allgather_presummed_norm_bf16_v1";

/// Arguments of [`PREFILL_ROUND_ONCE_SYMBOL`]: `arguments(call)` with the
/// residual after the output.
pub fn round_once_scatter_arguments(
    call: OneShotCall,
    residual: u64,
) -> Result<ArgumentList, Refusal> {
    let mut args = arguments(call)?;
    args.0.insert(2, Argument::DeviceBuffer(residual));
    args.validate()?;
    Ok(args)
}

/// Arguments of [`PREFILL_ROUND_ONCE_GATHER_SYMBOL`] after a scatter of
/// `call` into `call.output`: the peers' outputs, the norm weights and where
/// the pre-norm rows go.
pub fn round_once_gather_arguments(
    call: &OneShotCall,
    outputs: Vec<u64>,
    weights: u64,
    epsilon: f32,
    residual_output: u64,
) -> Result<ArgumentList, Refusal> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Refusal("fused comm epsilon must be positive and finite"));
    }
    let args = ArgumentList(vec![
        Argument::DeviceBuffer(call.output),
        Argument::DevicePointerArray(outputs),
        Argument::DevicePointerArray(call.peer_flags.clone()),
        Argument::DeviceBuffer(call.epoch),
        Argument::U64(call.elements),
        Argument::U32(call.rank),
        Argument::U32(call.ranks),
        Argument::DeviceBuffer(weights),
        Argument::U32(epsilon.to_bits()),
        Argument::DeviceBuffer(residual_output),
    ]);
    args.validate()?;
    Ok(args)
}

pub struct FusedOneShotCall {
    pub base: OneShotCall,
    pub residual: Option<u64>,
    pub weights: u64,
    pub epsilon: f32,
}

/// The original v1 entry accepts an absent residual.
pub fn fused_arguments(call: FusedOneShotCall) -> Result<ArgumentList, Refusal> {
    if !call.epsilon.is_finite() || call.epsilon <= 0.0 {
        return Err(Refusal("fused comm epsilon must be positive and finite"));
    }
    let mut args = arguments(call.base)?;
    args.0.push(match call.residual {
        Some(handle) => Argument::DeviceBuffer(handle),
        None => Argument::NullDevicePointer,
    });
    args.0.push(Argument::DeviceBuffer(call.weights));
    args.0.push(Argument::U32(call.epsilon.to_bits()));
    args.validate()?;
    Ok(args)
}

/// The v2 fused ABI requires a residual and keeps the rounded pre-norm sum
/// for the next layer. The residual input may alias the residual output.
pub struct FusedWithResidualOutputCall {
    pub fused: FusedOneShotCall,
    pub residual_output: u64,
}

pub fn fused_arguments_v2(call: FusedWithResidualOutputCall) -> Result<ArgumentList, Refusal> {
    if call.fused.residual.is_none() {
        return Err(Refusal("fused comm v2 requires a residual"));
    }
    if call.fused.base.output == call.residual_output {
        return Err(Refusal("normed and residual outputs must be distinct"));
    }
    let mut args = fused_arguments(call.fused)?;
    args.0.push(Argument::DeviceBuffer(call.residual_output));
    args.validate()?;
    Ok(args)
}

/// Decode all-reduce with residual and RMSNorm for 1 to 32 rows, in
/// [`DECODE_CUBIN_V2`]: the arguments (`fused_arguments_v2`), plans
/// ([`decode_plan`]) and state of `chop_comm_allreduce_oneshot_norm_bf16_v2`,
/// with earlier loads and the same bits. Its low-latency rows sum the norm
/// statistic over `st.async`, as the combine entry does (`AsyncNorm` in
/// push.cuh).
pub const DECODE_SYMBOL_V2: &str = "chop_comm_allreduce_decode_norm_bf16_v2";
/// The decode entry with the MoE combine folded in, as
/// `chop_comm_allreduce_combine_norm_bf16_v1` (`fused_combine_arguments`).
pub const DECODE_COMBINE_SYMBOL: &str = "chop_comm_allreduce_decode_combine_norm_bf16_v1";
/// Row counts up to this take the decode entries; above, where they run the
/// v2 protocols unchanged, callers keep the v2 entries (the combine entry
/// measured 0.1 us slower there).
pub const DECODE_MAX_ROWS: u32 = 12;

/// Launch plan of the decode entries for `rows` rows (1 to 32) at `ranks`
/// ranks. Their buffers are `push_buffers`'.
pub fn decode_plan(ranks: u32, rows: u32) -> Result<LaunchPlan, Refusal> {
    push_plan(ranks, rows)
}

/// The decode entry whose one-shot and two-shot rows sum their RMSNorm
/// squares in the low-latency order, so every row's outputs are bitwise
/// those of the same row reduced alone (at one row, the low-latency
/// protocol). Verify steps take it, since a verify row must equal the AR
/// row at its position, and prefill steps of up to 32 rows, since a prompt
/// row must equal the same row in a longer chunk. The arguments, plans and
/// state are the decode entry's; at low-latency row counts it is the decode
/// entry.
pub const VERIFY_SYMBOL: &str = "chop_comm_allreduce_verify_norm_bf16_v1";
/// [`VERIFY_SYMBOL`] with the MoE combine folded in, as
/// [`DECODE_COMBINE_SYMBOL`].
pub const VERIFY_COMBINE_SYMBOL: &str = "chop_comm_allreduce_verify_combine_norm_bf16_v1";

/// Rows the verify entries hold a row's low-latency order for: every row
/// count of the push path (low-latency, one-shot and two-shot).
pub fn verify_entry_rows(ranks: u32) -> Result<u32, Refusal> {
    let rows = PUSH_MAX_ROWS as u32;
    push_protocol(ranks, rows)?;
    Ok(rows)
}

/// Rank counts at which the one-row claims of a verify cubin's entries
/// (`verify_cubin_spec`) refer to the decode entries
/// ([`verify_one_row_reference`]): kernel-lab compared every row of every
/// row count up to 32, plain and combined, normalized and pre-norm, on
/// every rank of distinct devices, bitwise with the row reduced alone by
/// the decode entry (the verify rows case of kernel-lab's
/// `comm_latency.rs`).
pub const VERIFY_REFERENCE_RANKS: [u32; 2] = [2, 4];

/// The entry whose one-row launch a verify entry's rows equal at `ranks`
/// ranks: the decode entry of the same input form, plain
/// ([`DECODE_SYMBOL_V2`]) or `routed` ([`DECODE_COMBINE_SYMBOL`]). At one
/// row both run the low-latency protocol: the combine entries the same
/// code, the plain entries summing the norm statistic in the same order
/// through cluster loads or `st.async` (`row_sum`, `row_sum_async`). `None`
/// at rank counts without that evidence or whose one row takes another
/// protocol, where a verify entry's claims refer to its own one-row launch.
pub fn verify_one_row_reference(ranks: u32, routed: bool) -> Option<&'static str> {
    let low_latency = push_protocol(ranks, 1).is_ok_and(|p| p == PushProtocol::LowLatency);
    (low_latency && VERIFY_REFERENCE_RANKS.contains(&ranks)).then_some(if routed {
        DECODE_COMBINE_SYMBOL
    } else {
        DECODE_SYMBOL_V2
    })
}

pub(crate) fn push_defines() -> Vec<(String, String)> {
    let push = push_tuning().expect("valid comm push tuning");
    // Thresholds of ranks 1 to MAX_COMM_RANKS, one byte each from the low
    // byte up, for `by_ranks` (nvcc splits a define's value at commas).
    let by_ranks = |value: fn(&RankThresholds) -> u32| {
        let packed = push
            .ranks
            .values()
            .rev()
            .fold(0_u64, |packed, t| packed << 8 | u64::from(value(t)));
        format!("{packed:#018x}ull")
    };
    let mut defines: Vec<(String, String)> = [
        ("CHOP_COMM_LL_MAX_ROWS", push.low_latency_rows()),
        ("CHOP_COMM_LL_CLUSTER", push.low_latency_cluster),
        ("CHOP_COMM_PUSH_CLUSTER", push.cluster),
        ("CHOP_COMM_TWO_SHOT_MIN_RANKS", push.two_shot_min_ranks()),
        ("CHOP_COMM_ONE_SHOT_BASE", ONE_SHOT_BASE as u32),
        ("CHOP_COMM_TWO_SHOT_BASE", TWO_SHOT_BASE as u32),
        ("CHOP_COMM_PHASED_BASE", PHASED_BASE as u32),
        ("CHOP_COMM_TWO_SHOT_FLAGS", TWO_SHOT_FLAGS as u32),
        ("CHOP_COMM_MAX_RANKS", MAX_RANKS as u32),
        ("CHOP_COMM_WIDE_MAX_ROWS", WIDE_MAX_ROWS as u32),
        ("CHOP_COMM_WIDE_BLOCK_ROWS", push.wide_block_rows),
        ("CHOP_COMM_WIDE_GATHER_BASE", WIDE_GATHER_BASE as u32),
        ("CHOP_COMM_WIDE_SCATTER_BASE", WIDE_SCATTER_BASE as u32),
        ("CHOP_COMM_WIDE_FLAGS", WIDE_FLAGS as u32),
    ]
    .into_iter()
    .map(|(name, value)| (name.into(), value.to_string()))
    .collect();
    defines.push((
        "CHOP_COMM_LL_ROWS_BY_RANKS".into(),
        by_ranks(|t| t.low_latency_max_rows),
    ));
    defines.push((
        "CHOP_COMM_ONESHOT_ROWS_BY_RANKS".into(),
        by_ranks(|t| t.one_shot_max_rows),
    ));
    defines
}

/// CTAs per SM the wide entries (these and the hyper-connection ones) compile for,
/// their `min_blocks`. Their workers spin on flags that the sending CTAs of
/// every rank release, so spinning CTAs must never keep a sending CTA off
/// the GPU. The compile tests check that the cubins' reported registers
/// and shared memory allow this many
/// (`sm120_ctas_per_sm`); the bundles check the device at create
/// (`bundle_runtime::residency`).
pub const WIDE_CTAS_PER_SM: u32 = 2;
/// The SM120 target of the tuning table (188 SMs): GB202's 12 GPCs of 16
/// SMs with four disabled, so every GPC keeps at least 12.
const TARGET_GPCS: u32 = 12;
const TARGET_MIN_GPC_SMS: u32 = 12;

/// A tuning check, not a device check: whether one wide grid per GPU fits
/// the target layout at `WIDE_CTAS_PER_SM` CTAs per SM. A cluster's CTAs
/// share a GPC, so the model counts whole clusters in the
/// smallest GPC (three 8-CTA clusters, 36 in all). Unit tests hold every
/// planned grid to it and kernel-lab its worker overrides; serving checks
/// the driver's own count at create instead.
pub fn wide_fits_target_layout(plan: &LaunchPlan) -> Result<(), Refusal> {
    let cluster = plan.cluster.iter().product::<u32>().max(1);
    let clusters = plan.grid.iter().product::<u32>().div_ceil(cluster);
    let capacity = TARGET_GPCS * (TARGET_MIN_GPC_SMS * WIDE_CTAS_PER_SM / cluster);
    if clusters > capacity {
        return Err(Refusal(
            "wide all-reduce grid does not fit the target layout",
        ));
    }
    Ok(())
}

/// Resident CTAs per SM120 SM of an entry of `threads` threads with the
/// cubin's reported registers per thread and static shared memory: 64K
/// registers in four partitions allocated per warp in 256-register units,
/// 1536 threads and 100 KiB of shared memory (1 KiB reserved per CTA).
pub fn sm120_ctas_per_sm(threads: u32, registers: u32, shared_static_bytes: u32) -> u32 {
    let warps = threads.div_ceil(32).max(1);
    let warp_registers = (registers.max(1) * 32).next_multiple_of(256);
    let by_registers = 4 * (16384 / warp_registers) / warps;
    let by_threads = 1536 / (warps * 32);
    let by_shared = 102_400 / (shared_static_bytes + 1024);
    by_registers.min(by_threads).min(by_shared).min(32)
}

/// The wide two-shot (`Variant::FusedWide`), with `fused_arguments_v2`.
pub const WIDE_SYMBOL: &str = "chop_comm_allreduce_wide_norm_bf16_v1";
/// The wide two-shot with the MoE combine folded in, with
/// `fused_combine_arguments`.
pub const WIDE_COMBINE_SYMBOL: &str = "chop_comm_allreduce_wide_combine_norm_bf16_v1";

/// `chop_comm_allreduce_combine_norm_bf16_v1`: the v2 fused all-reduce whose
/// input holds the preweighted top-8 routed rows `[rows][8][4096]`, summed as
/// `chop_row_combine_mimo4096_v1` sums them, so the combine kernel and its
/// launch boundary drop out. Rows at or past `*live_rows` contribute zero;
/// `None` treats every row as live. Launch it with `plan` like v2.
pub fn fused_combine_arguments(
    call: FusedWithResidualOutputCall,
    live_rows: Option<u64>,
) -> Result<ArgumentList, Refusal> {
    let mut args = fused_arguments_v2(call)?;
    args.0.push(match live_rows {
        Some(handle) => Argument::DeviceBuffer(handle),
        None => Argument::NullDevicePointer,
    });
    args.validate()?;
    Ok(args)
}

pub fn cubin_spec() -> kernel_build::CubinSpec {
    use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
    let pointer = |name: &str, cxx_type: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let array = |name: &str, cxx_type: &str| P::PointerArray {
        name: name.into(),
        slots: MAX_COMM_RANKS as u8,
        cxx_type: cxx_type.into(),
    };
    let params = vec![
        pointer("input", "const float"),
        pointer("output", "float"),
        array("inboxes", "float"),
        array("flags", "unsigned int"),
        pointer("epoch", "unsigned int"),
        P::U64 {
            name: "elements".into(),
        },
        P::U32 {
            name: "rank".into(),
        },
        P::U32 {
            name: "ranks".into(),
        },
    ];
    let one = EntrySpec::new("chop_comm_allreduce_oneshot_f32_v1".into(), params,
        "chop::comm::allreduce_oneshot(input, output, inboxes, flags, epoch, elements, rank, ranks);".into(), 256, 1);
    let fused_params = vec![
        pointer("input", "const uint16_t"),
        pointer("output", "uint16_t"),
        array("inboxes", "uint16_t"),
        array("flags", "unsigned int"),
        pointer("epoch", "unsigned int"),
        P::U64 {
            name: "elements".into(),
        },
        P::U32 {
            name: "rank".into(),
        },
        P::U32 {
            name: "ranks".into(),
        },
        pointer("residual", "const uint16_t"),
        pointer("weights", "const uint16_t"),
        P::U32 {
            name: "eps_bits".into(),
        },
    ];
    let mut fused_params_v2 = fused_params.clone();
    fused_params_v2.push(pointer("residual_output", "uint16_t"));
    // The v2 arguments plus live_rows; input holds the preweighted top-8
    // routed rows [rows][8][4096], combined as the MoE row combine does.
    let mut combine_params = fused_params_v2.clone();
    combine_params.push(pointer("live_rows", "const uint32_t"));
    // The fused entries instantiate the transport with the epilogue functor
    // and build it from their epilogue arguments.
    let e = FUSED_EPILOGUE.functor();
    // The plain entries read BF16 rows as they are (the default input); the
    // combine entries sum MiMo's preweighted routed rows first.
    let combine = input::ROUTED_COMBINE_TOP8.policy;
    let fused_body = |transport: &str, live_rows: &str| {
        format!(
            "chop::comm::{transport}(input, inboxes, flags, epoch, elements, rank, ranks, {{output, residual, weights, eps_bits, residual_output}}{live_rows});"
        )
    };
    let fused_combine = EntrySpec::new(
        "chop_comm_allreduce_combine_norm_bf16_v1".into(),
        combine_params.clone(),
        fused_body(&format!("allreduce_push<{e}, {combine}>"), ", live_rows"),
        256,
        1,
    );
    // Keep two CTAs per SM for the spinning workers' residency
    // (`bundle_runtime::residency`).
    let wide = EntrySpec::new(
        WIDE_SYMBOL.into(),
        fused_params_v2.clone(),
        fused_body(&format!("allreduce_wide<{e}>"), ""),
        256,
        WIDE_CTAS_PER_SM,
    );
    let wide_combine = EntrySpec::new(
        WIDE_COMBINE_SYMBOL.into(),
        combine_params,
        fused_body(&format!("allreduce_wide<{e}, {combine}>"), ", live_rows"),
        256,
        WIDE_CTAS_PER_SM,
    );
    let fused_v2 = EntrySpec::new(
        "chop_comm_allreduce_oneshot_norm_bf16_v2".into(),
        fused_params_v2,
        fused_body(&format!("allreduce_push<{e}>"), ""),
        256,
        1,
    );
    // The decode combine entry: the combine arguments, state and protocols,
    // with the latency changes of `Fast` in push.cuh; bitwise the same.
    // Plain decode is in `decode_cubin_spec_v2`.
    let decode_combine_v1 = EntrySpec::new(
        DECODE_COMBINE_SYMBOL.into(),
        fused_combine.params.clone(),
        fused_body(
            &format!("allreduce_push<{e}, {combine}, true>"),
            ", live_rows",
        ),
        256,
        1,
    );
    let fused = EntrySpec::new(
        "chop_comm_allreduce_oneshot_norm_bf16_v1".into(),
        fused_params,
        format!(
            "chop::comm::allreduce_oneshot_norm<{e}>(input, inboxes, flags, epoch, elements, rank, ranks, {{output, residual, weights, eps_bits}});"
        ),
        256,
        1,
    );
    let scatter_params = vec![
        pointer("input", "const float"),
        pointer("partial", "float"),
        array("inboxes", "float"),
        array("flags", "unsigned int"),
        pointer("epoch", "const unsigned int"),
        P::U64 {
            name: "elements".into(),
        },
        P::U32 {
            name: "rank".into(),
        },
        P::U32 {
            name: "ranks".into(),
        },
    ];
    let scatter = EntrySpec::new(
        "chop_comm_reduce_scatter_f32_v1".into(),
        scatter_params,
        "chop::comm::reduce_scatter(input, partial, inboxes, flags, epoch, elements, rank, ranks);"
            .into(),
        256,
        1,
    );
    let gather_params = vec![
        pointer("output", "float"),
        array("outputs", "float"),
        array("flags", "unsigned int"),
        pointer("epoch", "unsigned int"),
        P::U64 {
            name: "elements".into(),
        },
        P::U32 {
            name: "rank".into(),
        },
        P::U32 {
            name: "ranks".into(),
        },
    ];
    let gather = EntrySpec::new(
        "chop_comm_allgather_scattered_f32_v1".into(),
        gather_params,
        "chop::comm::allgather_scattered(output, outputs, flags, epoch, elements, rank, ranks);"
            .into(),
        256,
        1,
    );
    let allgather_params = vec![
        pointer("input", "const float"),
        array("outputs", "float"),
        array("flags", "unsigned int"),
        pointer("epoch", "unsigned int"),
        P::U64 {
            name: "elements_per_rank".into(),
        },
        P::U32 {
            name: "rank".into(),
        },
        P::U32 {
            name: "ranks".into(),
        },
        P::U32 {
            name: "target".into(),
        },
    ];
    let allgather = EntrySpec::new(
        "chop_comm_allgather_f32_v1".into(),
        allgather_params,
        "chop::comm::allgather(input, outputs, flags, epoch, elements_per_rank, rank, ranks, target);"
            .into(),
        256,
        1,
    );
    // The fused phased pair (`PREFILL_ROUND_ONCE_SYMBOL`): the reduce-scatter
    // adds the residual before its one rounding, the gather sends the sums
    // and normalizes them.
    let scatter_residual = EntrySpec::new(
        PREFILL_ROUND_ONCE_SYMBOL.into(),
        vec![
            pointer("input", "const uint16_t"),
            pointer("partial", "uint16_t"),
            pointer("residual", "const uint16_t"),
            array("inboxes", "uint16_t"),
            array("flags", "unsigned int"),
            pointer("epoch", "const unsigned int"),
            P::U64 {
                name: "elements".into(),
            },
            P::U32 {
                name: "rank".into(),
            },
            P::U32 {
                name: "ranks".into(),
            },
        ],
        format!(
            "chop::comm::reduce_scatter_presum<{e}>(input, partial, residual, inboxes, flags, epoch, elements, rank, ranks);"
        ),
        256,
        1,
    );
    let gather_presummed = EntrySpec::new(
        PREFILL_ROUND_ONCE_GATHER_SYMBOL.into(),
        vec![
            pointer("output", "uint16_t"),
            array("outputs", "uint16_t"),
            array("flags", "unsigned int"),
            pointer("epoch", "unsigned int"),
            P::U64 {
                name: "elements".into(),
            },
            P::U32 {
                name: "rank".into(),
            },
            P::U32 {
                name: "ranks".into(),
            },
            pointer("weights", "const uint16_t"),
            P::U32 {
                name: "eps_bits".into(),
            },
            pointer("residual_output", "uint16_t"),
        ],
        format!(
            "chop::comm::allgather_presummed<{e}>(output, outputs, flags, epoch, elements, rank, ranks, weights, eps_bits, residual_output);"
        ),
        256,
        1,
    );
    let gather_rows = EntrySpec::new(
        "chop_comm_gather_rows_f32_v1".into(),
        vec![
            pointer("input", "const float"),
            array("outputs", "float"),
            array("flags", "unsigned int"),
            pointer("epoch", "unsigned int"),
            pointer("row_indices", "const uint32_t"),
            P::U32 {
                name: "rows".into(),
            },
            P::U64 {
                name: "shard".into(),
            },
            P::U32 {
                name: "rank".into(),
            },
            P::U32 {
                name: "ranks".into(),
            },
            P::U32 {
                name: "target".into(),
            },
        ],
        "chop::comm::gather_rows(input, outputs, flags, epoch, row_indices, rows, shard, rank, ranks, target);"
            .into(),
        256,
        1,
    );
    let exchange_rows = EntrySpec::new(
        "chop_comm_exchange_rows_f32_v1".into(),
        vec![
            pointer("input", "const float"),
            array("outputs", "float"),
            array("flags", "unsigned int"),
            pointer("epoch", "unsigned int"),
            P::U32 {
                name: "rows".into(),
            },
            P::U64 {
                name: "shard".into(),
            },
            P::U32 {
                name: "rank".into(),
            },
            P::U32 {
                name: "ranks".into(),
            },
        ],
        "chop::comm::exchange_rows(input, outputs, flags, epoch, rows, shard, rank, ranks);".into(),
        256,
        1,
    );
    let bf16 = [
        one.clone(),
        scatter.clone(),
        gather.clone(),
        allgather.clone(),
        gather_rows.clone(),
        exchange_rows.clone(),
    ]
    .into_iter()
    .map(|mut entry| {
        entry.symbol = entry.symbol.replace("_f32_", "_bf16_");
        for param in &mut entry.params {
            match param {
                P::DevicePointer { cxx_type, .. } | P::PointerArray { cxx_type, .. } => {
                    *cxx_type = cxx_type.replace("float", "uint16_t");
                }
                _ => {}
            }
        }
        entry
    })
    .collect::<Vec<_>>();
    let mut entries = vec![
        one,
        scatter,
        gather,
        allgather,
        gather_rows,
        exchange_rows,
        fused,
        fused_v2,
        fused_combine,
        wide,
        wide_combine,
        decode_combine_v1,
        scatter_residual,
        gather_presummed,
        EntrySpec::new(
            "chop_comm_local_norm_bf16_v1".into(),
            vec![
                pointer("input", "const uint16_t"),
                pointer("output", "uint16_t"),
                pointer("residual", "const uint16_t"),
                pointer("weights", "const uint16_t"),
                P::U32 {
                    name: "eps_bits".into(),
                },
            ],
            format!("chop::comm::local_norm<{e}>(input, {{output, residual, weights, eps_bits}});"),
            1024,
            1,
        ),
        EntrySpec::new(
            "chop_comm_local_flags_v1".into(),
            vec![
                pointer("flags", "unsigned int"),
                pointer("epoch", "unsigned int"),
            ],
            "chop::comm::local_flags(flags, epoch);".into(),
            256,
            1,
        ),
    ];
    entries.extend(bf16);
    CubinSpec {
        name: "comm_sm120".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![PathBuf::from("comm"), PathBuf::from("common")],
        headers: vec![
            "#include \"comm/comm.cuh\"".into(),
            FUSED_EPILOGUE.include(),
            input::ROUTED_COMBINE_TOP8.include(),
        ],
        defines: push_defines(),
        entries,
    }
}

/// The cubin of [`DECODE_SYMBOL_V2`]. In `comm_sm120` its asynchronous
/// norm state would share one shared-memory layout with the decode combine
/// entry's and change the code of both.
pub const DECODE_CUBIN_V2: &str = "comm_decode_v2_sm120";

pub fn decode_cubin_spec_v2() -> kernel_build::CubinSpec {
    let mut spec = cubin_spec();
    let params = spec
        .entries
        .iter()
        .find(|entry| entry.symbol == "chop_comm_allreduce_oneshot_norm_bf16_v2")
        .expect("comm v2 entry")
        .params
        .clone();
    let (e, plain) = (FUSED_EPILOGUE.functor(), input::BF16.policy);
    spec.name = DECODE_CUBIN_V2.into();
    spec.entries = vec![kernel_build::EntrySpec::new(
        DECODE_SYMBOL_V2.into(),
        params,
        format!(
            "chop::comm::allreduce_push<{e}, {plain}, true, true>(input, inboxes, flags, epoch, elements, rank, ranks, {{output, residual, weights, eps_bits, residual_output}});"
        ),
        256,
        1,
    )];
    spec
}

/// A comm cubin entry's declaration. Comm entries are fixed cubin symbols
/// rather than catalog structs, so they are declared here by symbol.
#[derive(Clone, Debug)]
pub struct Declared {
    pub symbol: String,
    pub invariance: Invariance,
}

/// The declarations of the shared transport's entries that serve prefill or
/// exact verify rows; every other entry of `comm_sm120` and the verify
/// cubin claims nothing. A model declares its own transports' entries.
///
/// - The round-once prefill pair and the verify entries sum a row's ranks
///   and its RMSNorm squares in one order whatever rows share the call, so
///   prefill rows keep their bits over chunks (the bundle takes the verify
///   entries up to 32 rows, the pair above) and verify rows are one-row
///   exact. Their one-row bits are the decode entries' at
///   [`VERIFY_REFERENCE_RANKS`] ([`verify_one_row_reference`]).
/// - The decode one-shot and combine entries do so at low-latency row
///   counts, where exact verify rows take them.
/// - The row gathers and exchanges move rows without arithmetic, and the
///   F32 all-gather (the context-parallel group gather) copies every
///   source's words with 32-bit moves, whatever its word count and target;
///   so does the phased group gather (`context_parallel::phased::SYMBOL`),
///   one destination at a time.
/// - The F32 reduce-scatter sums each element's ranks in rank order from
///   +0 (`sum_range`), whichever owner span and worker take it, and the
///   scattered all-gather copies each owner's span to every rank: the pair
///   keeps every row's bits at any row count and chunk split.
/// - The BF16-partials all-reduce's entries (the low-latency entry, the
///   phased one-shot and two-shot and the turns entry) sum each element's
///   rounded partials in rank order from F32 zero, whatever rows share the
///   call, the grid's senders and workers or the chunk split, so each row
///   keeps its bits over rows, chunks and batches up to the prefill
///   capacity. The rounded all-reduce's entries (the low-latency round
///   entry and the phased one-shot round and round two-shot) do the same
///   over F32 partials, each sum rounded once to BF16.
pub fn declared_entries() -> Vec<Declared> {
    let low_latency = [2, 4, 8]
        .into_iter()
        .map(|ranks| {
            push_tuning()
                .and_then(|push| push.thresholds(ranks))
                .map(|t| t.low_latency_max_rows)
                .expect("comm push tuning of every catalog rank count")
        })
        .min()
        .expect("catalog rank counts");
    // Rows up to the prefill capacity.
    let moved = Invariance::any_rows(8192);
    let verify = Invariance::any_rows(PUSH_MAX_ROWS as u32);
    let mut declared = vec![
        (PREFILL_ROUND_ONCE_SYMBOL, Invariance::PREFILL),
        (PREFILL_ROUND_ONCE_GATHER_SYMBOL, Invariance::PREFILL),
        (VERIFY_SYMBOL, verify),
        (VERIFY_COMBINE_SYMBOL, verify),
        (
            "chop_comm_allreduce_oneshot_norm_bf16_v2",
            Invariance::one_row(low_latency),
        ),
        (
            "chop_comm_allreduce_combine_norm_bf16_v1",
            Invariance::one_row(low_latency),
        ),
        (DECODE_COMBINE_SYMBOL, Invariance::one_row(low_latency)),
        (DECODE_SYMBOL_V2, Invariance::one_row(low_latency)),
    ];
    for symbol in [
        ll_bf16::SYMBOL,
        phased_bf16::ONE_SHOT_SYMBOL,
        phased_bf16::SYMBOL,
        phased_turns_bf16::SYMBOL,
        ll_f32::ROUND_SYMBOL,
        phased_f32::ONE_SHOT_ROUND_SYMBOL,
        phased_f32::ROUND_SYMBOL,
    ] {
        declared.push((symbol, moved));
    }
    for symbol in [
        "chop_comm_gather_rows_f32_v1",
        "chop_comm_gather_rows_bf16_v1",
        "chop_comm_exchange_rows_f32_v1",
        "chop_comm_exchange_rows_bf16_v1",
        "chop_comm_allgather_f32_v1",
        crate::context_parallel::phased::SYMBOL,
        "chop_comm_reduce_scatter_f32_v1",
        "chop_comm_allgather_scattered_f32_v1",
    ] {
        declared.push((symbol, moved));
    }
    declared
        .into_iter()
        .map(|(symbol, invariance)| Declared {
            symbol: symbol.to_owned(),
            invariance,
        })
        .collect::<Vec<_>>()
}

/// The verify entries ([`VERIFY_SYMBOL`], [`VERIFY_COMBINE_SYMBOL`]) in a
/// cubin `name` of their own: the decode entries' arguments with one-shot
/// rows normalized in the low-latency order. Only a model that runs exact
/// verify rows loads it, so `comm_sm120` and the models that load it keep
/// their entries.
pub fn verify_cubin_spec(name: &str) -> kernel_build::CubinSpec {
    use kernel_build::EntrySpec;
    let mut spec = cubin_spec();
    let params = |symbol: &str| {
        spec.entries
            .iter()
            .find(|entry| entry.symbol == symbol)
            .expect("comm decode entry")
            .params
            .clone()
    };
    let e = FUSED_EPILOGUE.functor();
    let (plain, combine) = (input::BF16.policy, input::ROUTED_COMBINE_TOP8.policy);
    let entries = vec![
        EntrySpec::new(
            VERIFY_SYMBOL.into(),
            params("chop_comm_allreduce_oneshot_norm_bf16_v2"),
            format!(
                "chop::comm::allreduce_push<{e}, {plain}, true, false, true>(input, inboxes, flags, epoch, elements, rank, ranks, {{output, residual, weights, eps_bits, residual_output}});"
            ),
            256,
            1,
        ),
        EntrySpec::new(
            VERIFY_COMBINE_SYMBOL.into(),
            params(DECODE_COMBINE_SYMBOL),
            format!(
                "chop::comm::allreduce_push<{e}, {combine}, true, true, true>(input, inboxes, flags, epoch, elements, rank, ranks, {{output, residual, weights, eps_bits, residual_output}}, live_rows);"
            ),
            256,
            1,
        ),
    ];
    spec.name = name.into();
    spec.entries = entries;
    spec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_grids_are_resident_at_once_on_one_gpu() {
        // Every wide plan's grid on one GPU fits the 36 8-CTA clusters the
        // target holds at two CTAs per SM.
        for ranks in [1, 2, 4, 8] {
            for rows in 33..=wide_max_rows(ranks).unwrap() {
                let plan = graph_plan(&fused_query(ranks, rows), rows).unwrap();
                wide_fits_target_layout(&plan).unwrap();
            }
        }
        let grid = |ctas| LaunchPlan {
            grid: [ctas, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [8, 1, 1],
            pdl: true,
        };
        assert!(wide_fits_target_layout(&grid(36 * 8)).is_ok());
        assert!(wide_fits_target_layout(&grid(36 * 8 + 1)).is_err());
        // The entries' 256 threads at the 128 registers two CTAs per SM
        // allow; one more register, or 50 KiB of shared memory, leaves one.
        assert_eq!(sm120_ctas_per_sm(256, 128, 0), 2);
        assert_eq!(sm120_ctas_per_sm(256, 104, 4096), 2);
        assert_eq!(sm120_ctas_per_sm(256, 129, 0), 1);
        assert_eq!(sm120_ctas_per_sm(256, 64, 50 * 1024), 1);
        assert_eq!(sm120_ctas_per_sm(256, 32, 0), 6);
    }
    fn fused_query(ranks: u32, rows: u32) -> CommQuery {
        CommQuery::AllReduce {
            ranks,
            elements: u64::from(rows) * u64::from(FUSED_EPILOGUE.width),
            dtype: DType::Bf16,
            fused_residual: true,
            fused_rmsnorm: Some(RmsNormEpilogue {
                kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
                width: FUSED_EPILOGUE.width,
                eps: 1e-6,
            }),
        }
    }

    #[test]
    fn fused_norm_requires_residual() {
        let mut accepted = Vec::new();
        for rows in [1, 64, 8192] {
            let mut query = fused_query(4, rows);
            assert!(plan(&query, rows).is_ok());
            assert!(graph_plan(&query, rows).is_ok());
            let CommQuery::AllReduce { fused_residual, .. } = &mut query else {
                unreachable!()
            };
            *fused_residual = false;
            for (name, result) in [
                ("choose", choose(&query).map(|_| ())),
                ("plan", plan(&query, rows).map(|_| ())),
                ("graph_choose", graph_choose(&query).map(|_| ())),
                ("graph_plan", graph_plan(&query, rows).map(|_| ())),
            ] {
                match result {
                    Ok(()) => accepted.push(format!("{name}, rows={rows}")),
                    Err(error) => assert!(error.0.contains("residual"), "{error}"),
                }
            }
        }
        fn call(residual: Option<u64>) -> FusedOneShotCall {
            FusedOneShotCall {
                base: OneShotCall {
                    input: 1,
                    output: 2,
                    peer_inboxes: vec![3, 4, 5, 6],
                    peer_flags: vec![7, 8, 9, 10],
                    epoch: 11,
                    elements: 4096,
                    rank: 0,
                    ranks: 4,
                },
                residual,
                weights: 12,
                epsilon: 1e-6,
            }
        }
        let v2 = |residual| FusedWithResidualOutputCall {
            fused: call(residual),
            residual_output: 14,
        };
        assert_eq!(
            fused_arguments(call(None)).unwrap().0[8],
            Argument::NullDevicePointer
        );
        for residual in [13, 14] {
            assert!(fused_arguments_v2(v2(Some(residual))).is_ok());
            assert!(fused_combine_arguments(v2(Some(residual)), None).is_ok());
        }
        for (name, result) in [
            ("v2", fused_arguments_v2(v2(None))),
            ("combine", fused_combine_arguments(v2(None), Some(15))),
        ] {
            match result {
                Ok(_) => accepted.push(name.into()),
                Err(error) => assert!(error.0.contains("residual"), "{error}"),
            }
        }
        assert!(
            accepted.is_empty(),
            "missing residual accepted: {accepted:?}"
        );
    }
    #[test]
    fn fused_norm_refuses_incompatible_kinds() {
        use kernel_core::contract::RmsNormKind;
        let mut accepted = Vec::new();
        for kind in [
            RmsNormKind::Plain,
            RmsNormKind::GemmaOnePlusWeight,
            RmsNormKind::QkPerHead { head_dim: 128 },
        ] {
            for rows in [1, 64, 8192] {
                let mut query = fused_query(4, rows);
                let CommQuery::AllReduce {
                    fused_rmsnorm: Some(norm),
                    ..
                } = &mut query
                else {
                    unreachable!()
                };
                norm.kind = kind;
                for (name, result) in [
                    ("choose", choose(&query).map(|_| ())),
                    ("plan", plan(&query, rows).map(|_| ())),
                    ("graph_choose", graph_choose(&query).map(|_| ())),
                    ("graph_plan", graph_plan(&query, rows).map(|_| ())),
                ] {
                    match result {
                        Ok(()) => accepted.push(format!("{name}: {kind:?}, rows={rows}")),
                        Err(error) => assert!(error.0.contains("HF BF16-intermediate"), "{error}"),
                    }
                }
            }
        }
        assert!(
            accepted.is_empty(),
            "incompatible norms accepted: {accepted:?}"
        );
    }

    #[test]
    fn threshold_and_rank_limit() {
        let q = |elements, ranks| CommQuery::AllReduce {
            ranks,
            elements,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        assert_eq!(choose(&q(2048, 2)).unwrap(), Variant::OneShot);
        assert_eq!(choose(&q(65537, 2)).unwrap(), Variant::TwoShot);
        assert!(choose(&q(16, 9)).is_err());
        assert!(choose_for(&q(16, 2), "12.1", 188).is_err());
        let fused_large = CommQuery::AllReduce {
            ranks: 4,
            elements: 4096 * 8192,
            dtype: DType::Bf16,
            fused_residual: true,
            fused_rmsnorm: Some(RmsNormEpilogue {
                kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
                width: 4096,
                eps: 1e-6,
            }),
        };
        assert_eq!(choose(&fused_large).unwrap(), Variant::FusedTwoShot);
        for ranks in CATALOG_RANKS {
            for spec in own_catalog_for(ranks).unwrap() {
                spec.instantiation
                    .validate()
                    .expect("valid comm instantiation");
            }
        }
    }

    #[test]
    fn fused_norm_refuses_widths_without_an_instantiation() {
        let fused = |width, elements| CommQuery::AllReduce {
            ranks: 4,
            elements,
            dtype: DType::Bf16,
            fused_residual: true,
            fused_rmsnorm: Some(RmsNormEpilogue {
                kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
                width,
                eps: 1e-6,
            }),
        };
        assert_eq!(
            choose(&fused(4096, 4096 * 2)).unwrap(),
            Variant::FusedOneShot
        );
        // Two 2048-wide rows used to pass as one 4096-wide row.
        for (width, elements) in [(2048, 4096), (2048, 2048 * 64), (2560, 2560 * 4)] {
            assert!(choose(&fused(width, elements)).is_err(), "{width}");
            assert!(plan(&fused(width, elements), 1).is_err(), "{width}");
        }
        // The contract refuses a length that is not whole rows, and a zero width.
        assert!(fused(4096, 4096 + 2048).validate().is_err());
        assert!(fused(0, 4096).validate().is_err());
    }

    #[test]
    fn fused_push_plans_follow_the_tuning_table() {
        let fused = |ranks, rows: u32| CommQuery::AllReduce {
            ranks,
            elements: u64::from(rows) * 4096,
            dtype: DType::Bf16,
            fused_residual: true,
            fused_rmsnorm: Some(RmsNormEpilogue {
                kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
                width: 4096,
                eps: 1e-6,
            }),
        };
        for (ranks, rows, protocol, grid, cluster) in [
            (4, 1, PushProtocol::LowLatency, 2, 2),
            (4, 2, PushProtocol::LowLatency, 4, 2),
            (4, 4, PushProtocol::OneShot, 16, 8),
            (4, 16, PushProtocol::OneShot, 24, 8),
            (4, 17, PushProtocol::TwoShot, 32, 8),
            (4, 32, PushProtocol::TwoShot, 40, 8),
            // Two ranks: one-shot moves the two-shot's bytes in one phase.
            (2, 1, PushProtocol::LowLatency, 2, 2),
            (2, 2, PushProtocol::LowLatency, 4, 2),
            (2, 3, PushProtocol::OneShot, 16, 8),
            (2, 17, PushProtocol::OneShot, 32, 8),
            (2, 32, PushProtocol::OneShot, 40, 8),
            (8, 2, PushProtocol::LowLatency, 4, 2),
            (8, 3, PushProtocol::OneShot, 16, 8),
            (8, 16, PushProtocol::OneShot, 24, 8),
            (8, 17, PushProtocol::TwoShot, 32, 8),
            (8, 32, PushProtocol::TwoShot, 40, 8),
            (3, 1, PushProtocol::OneShot, 16, 8),
            (6, 32, PushProtocol::OneShot, 40, 8),
            (1, 1, PushProtocol::LowLatency, 2, 2),
            // Wide: the sending cluster and a worker per row up to 64.
            (4, 33, PushProtocol::Wide, 48, 8),
            (4, 64, PushProtocol::Wide, 72, 8),
            (4, 256, PushProtocol::Wide, 72, 8),
            (2, 100, PushProtocol::Wide, 72, 8),
            (8, 128, PushProtocol::Wide, 72, 8),
            (1, 40, PushProtocol::Wide, 48, 8),
        ] {
            assert_eq!(graph_protocol(ranks, rows).unwrap(), protocol);
            let plan = graph_plan(&fused(ranks, rows), rows).unwrap();
            assert_eq!(plan.grid, [grid, 1, 1], "ranks {ranks} rows {rows}");
            assert_eq!(plan.cluster, [cluster, 1, 1]);
            assert_eq!(plan.grid[0] % plan.cluster[0], 0);
            assert!(plan.pdl);
            let buffers = graph_buffers(ranks, rows).unwrap();
            assert_eq!(
                plan.workspace_bytes,
                buffers.inbox_bytes + buffers.flag_bytes + buffers.epoch_bytes
            );
        }
        assert!(push_protocol(9, 1).is_err());
        assert!(graph_protocol(4, 257).is_err() && graph_protocol(6, 33).is_err());
        // The push entries' API keeps its 32-row limit; callers sizing a
        // prefill state above 32 rows get a refusal, not the wide sizes.
        assert!(push_protocol(4, 33).is_err());
        let defines = push_defines();
        for (name, value) in [
            ("CHOP_COMM_LL_MAX_ROWS", "2"),
            ("CHOP_COMM_LL_ROWS_BY_RANKS", "0x0200000002000202ull"),
            ("CHOP_COMM_ONESHOT_ROWS_BY_RANKS", "0x1020202010202020ull"),
            ("CHOP_COMM_TWO_SHOT_MIN_RANKS", "4"),
            ("CHOP_COMM_MAX_RANKS", "8"),
            ("CHOP_COMM_WIDE_MAX_ROWS", "256"),
            ("CHOP_COMM_WIDE_BLOCK_ROWS", "16"),
            ("CHOP_COMM_WIDE_GATHER_BASE", "5767168"),
            ("CHOP_COMM_WIDE_SCATTER_BASE", "8388608"),
            ("CHOP_COMM_WIDE_FLAGS", "98"),
        ] {
            assert!(
                defines.contains(&(name.into(), value.into())),
                "{name} {defines:?}"
            );
        }
    }

    #[test]
    fn comm_state_buffers_follow_the_protocols_of_each_rank_count() {
        // (ranks, capacity, flag words, epoch bytes): one-shot needs 2 ranks
        // + 2 words, two-shot 82 + 2 ranks, the phased pair 82; low latency
        // one epoch cell per row.
        // Above 32 rows the wide protocol needs 164 words at 2, 4 and 8
        // ranks (98 + 2 * 8 per source + 2 * 16 blocks + 2 local + 2 * 8
        // shape words and flags per source).
        for (ranks, capacity, words, epoch_bytes) in [
            (2, 1, 0, 4),
            (2, 2, 0, 8),
            (2, 4, 6, 16),
            (2, 32, 6, 128),
            (2, 64, 164, 128),
            (2, 8192, 164, 128),
            (4, 1, 0, 4),
            (4, 16, 10, 64),
            (4, 32, 90, 128),
            (4, 256, 164, 128),
            (4, 8192, 164, 128),
            (8, 4, 18, 16),
            (8, 16, 18, 64),
            (8, 32, 98, 128),
            (8, 8192, 164, 128),
            (3, 8192, 82, 128),
        ] {
            let state = fused_state_buffers(ranks, capacity).unwrap();
            assert_eq!(
                state.flag_bytes,
                words * 4,
                "{ranks} ranks, {capacity} rows"
            );
            assert_eq!(
                state.epoch_bytes, epoch_bytes,
                "{ranks} ranks, {capacity} rows"
            );
            for rows in 1..=capacity.min(32) {
                let push = push_buffers(ranks, rows).unwrap();
                assert!(push.inbox_bytes <= state.inbox_bytes);
                assert!(push.zeroed_inbox_bytes <= state.zeroed_inbox_bytes);
            }
            if capacity > 32 {
                let phased =
                    phased_buffers(ranks, u64::from(capacity) * 4096, DType::Bf16).unwrap();
                assert!(phased.inbox_bytes <= state.inbox_bytes);
                for rows in 33..=capacity.min(wide_max_rows(ranks).unwrap()) {
                    let wide = graph_buffers(ranks, rows).unwrap();
                    assert!(wide.inbox_bytes <= state.inbox_bytes);
                    assert!(wide.flag_bytes <= state.flag_bytes);
                    assert!(wide.epoch_bytes <= state.epoch_bytes);
                }
            }
        }
        // A per-row flag budget of 16 bytes, which the TP4 tests assumed,
        // does not hold at eight ranks.
        assert!(fused_state_buffers(8, 4).unwrap().flag_bytes > 16 * 4);
        assert!(fused_state_buffers(4, 0).is_err());
        assert!(fused_state_buffers(9, 1).is_err());
    }

    #[test]
    fn decode_entry_takes_the_v2_arguments_and_plans() {
        let (spec, decode_spec) = (cubin_spec(), decode_cubin_spec_v2());
        assert_eq!(decode_spec.entries.len(), 1);
        let decode = &decode_spec.entries[0];
        assert_eq!(decode.symbol, DECODE_SYMBOL_V2);
        assert!(
            spec.entries
                .iter()
                .all(|entry| entry.symbol != DECODE_SYMBOL_V2)
        );
        let v2 = spec
            .entries
            .iter()
            .find(|entry| entry.symbol == "chop_comm_allreduce_oneshot_norm_bf16_v2")
            .unwrap();
        assert_eq!(decode.params, v2.params);
        assert_eq!(decode.max_threads, v2.max_threads);
        assert_eq!(decode.min_blocks, v2.min_blocks);
        assert!(decode.body.contains("::Bf16, true, true>("));
        for ranks in [2, 4, 8] {
            for rows in 1..=32 {
                let plan = decode_plan(ranks, rows).unwrap();
                assert_eq!(plan.block, [256, 1, 1]);
                assert!(plan.cluster[0] > 1);
                assert_eq!(plan.dynamic_shared_bytes, 0);
            }
        }
    }

    #[test]
    fn verify_entries_refer_to_the_decode_entries_where_qualified() {
        // The decode entry of each form, whose arguments the verify entry
        // takes.
        let verify = verify_cubin_spec("verify");
        let decode = [cubin_spec().entries, decode_cubin_spec_v2().entries].concat();
        for routed in [false, true] {
            let symbol = if routed {
                VERIFY_COMBINE_SYMBOL
            } else {
                VERIFY_SYMBOL
            };
            let entry = verify.entries.iter().find(|e| e.symbol == symbol).unwrap();
            for ranks in 1..=MAX_RANKS as u32 {
                let reference = verify_one_row_reference(ranks, routed);
                assert_eq!(
                    reference.is_some(),
                    VERIFY_REFERENCE_RANKS.contains(&ranks),
                    "{ranks} ranks"
                );
                let Some(reference) = reference else {
                    continue;
                };
                // One row runs the low-latency protocol in both.
                assert_eq!(push_protocol(ranks, 1).unwrap(), PushProtocol::LowLatency);
                let reference = decode.iter().find(|e| e.symbol == reference).unwrap();
                assert_eq!(entry.params, reference.params);
            }
        }
        assert_eq!(verify_one_row_reference(8, false), None);
    }

    #[test]
    fn two_shot_owners_hold_at_most_the_device_share() {
        // `kPer` in push.cuh bounds an owner's vectors per thread by the
        // fewest two-shot ranks; every rank count that takes the two-shot has
        // at least that many and splits the 512 row vectors evenly.
        let push = push_tuning().unwrap();
        let min_ranks = push.two_shot_min_ranks();
        let per = (PUSH_MAX_ROWS * 512 / u64::from(min_ranks)).div_ceil(8 * 256);
        assert_eq!((min_ranks, per), (4, 2));
        for ranks in 1..=MAX_RANKS as u32 {
            if push_protocol(ranks, 32).unwrap() == PushProtocol::TwoShot {
                assert!(ranks >= min_ranks && 512 % ranks == 0, "{ranks}");
                let owned = PUSH_MAX_ROWS * 512 / u64::from(ranks);
                assert!(owned.div_ceil(8 * 256) <= per);
            }
        }
    }

    #[test]
    fn exchanged_rows_cover_every_row_once_in_rank_order() {
        for rows in 1..=64_u32 {
            let mut next = 0;
            for rank in 0..4 {
                let owned = exchange_rows_owned(rows, 4, rank);
                assert_eq!(owned.start, next, "rows {rows} rank {rank}");
                assert!(owned.len() <= rows.div_ceil(4) as usize);
                next = owned.end;
            }
            assert_eq!(next, rows);
        }
        assert_eq!(exchange_rows_owned(1, 4, 0), 0..1);
        assert_eq!(exchange_rows_owned(1, 4, 1), 1..1);
        assert_eq!(exchange_rows_owned(17, 4, 3), 15..17);
        assert_eq!(exchange_rows_owned(32, 4, 2), 16..24);
        let plan = exchange_rows_plan();
        assert_eq!((plan.grid, plan.cluster), ([8, 1, 1], [8, 1, 1]));
        assert!(plan.pdl);
        let call = |rank, outputs: usize| ExchangeRowsCall {
            input: 1,
            outputs: vec![2; outputs],
            flags: vec![3; 4],
            epoch: 4,
            rows: 32,
            shard: 38_144,
            rank,
            ranks: 4,
        };
        let arguments = exchange_rows_arguments(call(1, 4)).unwrap();
        assert_eq!(arguments.0.len(), 8);
        assert_eq!(arguments.0[4], Argument::U32(32));
        assert_eq!(arguments.0[6], Argument::U32(1));
        assert!(exchange_rows_arguments(call(4, 4)).is_err());
        assert!(exchange_rows_arguments(call(0, 3)).is_err());
    }

    #[test]
    fn phased_plans_do_not_depend_on_flags_beyond_a_fixed_prefix() {
        for (rows, grid) in [(33_u64, 48), (100, 64), (2048, 64), (8192, 64)] {
            let query = CommQuery::AllReduce {
                ranks: 4,
                elements: rows * 4096,
                dtype: DType::Bf16,
                fused_residual: true,
                fused_rmsnorm: Some(RmsNormEpilogue {
                    kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
                    width: 4096,
                    eps: 1e-6,
                }),
            };
            assert_eq!(choose(&query).unwrap(), Variant::FusedTwoShot);
            let plan = plan(&query, rows as u32).unwrap();
            assert_eq!(plan.grid, [grid, 1, 1], "rows {rows}");
            assert_eq!(plan.cluster, [8, 1, 1]);
            let buffers = phased_buffers(4, rows * 4096, DType::Bf16).unwrap();
            // The MiMo prefill state holds 64 bytes of flags per row and a
            // 256 MiB inbox; the flag prefix is the same for every row count.
            assert_eq!(buffers.flag_bytes, 328);
            assert!(buffers.flag_bytes <= 64 * 33);
            assert!(buffers.inbox_bytes <= 4 * 8192 * 4096 * 2);
        }
        let unfused = CommQuery::AllReduce {
            ranks: 4,
            elements: 1_000_003,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        assert_eq!(choose(&unfused).unwrap(), Variant::TwoShot);
        assert_eq!(plan(&unfused, 245).unwrap().grid, [64, 1, 1]);
    }

    /// Inbox bytes that `source` writes into a receiver in a two-shot call of
    /// `rows` rows, reduce-scatter then all-gather, placed as `two_shot` in
    /// `push.cuh` places them: slots use the largest row count.
    fn two_shot_writes(ranks: u64, rows: u64, source: u64) -> [std::ops::Range<u64>; 2] {
        let slot = PUSH_MAX_ROWS * ROW_BYTES / ranks;
        let used = rows * ROW_BYTES / ranks;
        let scatter = TWO_SHOT_BASE + source * slot;
        let gather = TWO_SHOT_BASE + (ranks + source) * slot;
        [scatter..scatter + used, gather..gather + used]
    }

    #[test]
    fn two_shot_scatter_never_overlaps_an_earlier_gather() {
        // A peer's next scatter may arrive while this rank still reads the
        // previous call's gather region, whatever the two row counts.
        for ranks in [4_u64, 8] {
            for previous in 17..=32 {
                for next in 17..=32 {
                    for (old, new) in (0..ranks).flat_map(|a| (0..ranks).map(move |b| (a, b))) {
                        let [_, gather] = two_shot_writes(ranks, previous, old);
                        let [scatter, _] = two_shot_writes(ranks, next, new);
                        assert!(
                            scatter.end <= gather.start || gather.end <= scatter.start,
                            "ranks {ranks}: {previous} then {next} rows"
                        );
                    }
                }
                let buffers = push_buffers(ranks as u32, previous as u32).unwrap();
                for source in 0..ranks {
                    for range in two_shot_writes(ranks, previous, source) {
                        assert!(range.start >= TWO_SHOT_BASE);
                        assert!(range.end <= buffers.inbox_bytes.min(PHASED_BASE));
                    }
                }
            }
        }
    }

    /// Flag words of one call that peers write and that only the rank itself
    /// writes, as `push.cuh` lays them out; `None` is the phased pair.
    fn flag_words(protocol: Option<PushProtocol>, ranks: u64) -> [Vec<std::ops::Range<u64>>; 2] {
        let words = |start: u64, end: u64| std::iter::once(start..end).collect::<Vec<_>>();
        match protocol {
            Some(PushProtocol::LowLatency) => [Vec::new(), Vec::new()],
            Some(PushProtocol::OneShot) => [words(0, 2 * ranks), words(2 * ranks, 2 * ranks + 2)],
            Some(PushProtocol::TwoShot) => [
                words(TWO_SHOT_FLAGS, TWO_SHOT_FLAGS + 2 * ranks),
                words(2 * ranks, 2 * ranks + 2),
            ],
            Some(PushProtocol::Wide) => {
                let push = push_tuning().unwrap();
                let local = push.wide_local_words();
                [
                    vec![WIDE_FLAGS..local, local + 2..push.wide_flag_words()],
                    words(local, local + 2),
                ]
            }
            None => [vec![32..32 + ranks, 40..40 + ranks, 50..82], words(48, 50)],
        }
    }

    #[test]
    fn protocols_share_no_peer_written_flag_word() {
        // A peer can start the next call, with any protocol, while this rank
        // still waits on a flag of the previous one.
        let overlaps =
            |a: &std::ops::Range<u64>, b: &std::ops::Range<u64>| a.start < b.end && b.start < a.end;
        for ranks in 1..=8_u32 {
            let ranks64 = u64::from(ranks);
            let mut protocols = vec![(None, PHASED_FLAG_WORDS)];
            for rows in [1_u32, 2, 3, 16, 17, 32, 33, 256] {
                let Ok(protocol) = graph_protocol(ranks, rows) else {
                    // No wide protocol at this rank count.
                    assert!(rows > 32 && wide_max_rows(ranks).unwrap() == 0);
                    continue;
                };
                let words = graph_buffers(ranks, rows).unwrap().flag_bytes / 4;
                protocols.push((Some(protocol), words));
            }
            for &(protocol, words) in &protocols {
                let [peer, local] = flag_words(protocol, ranks64);
                for range in peer.iter().chain(&local) {
                    assert!(range.end <= words, "{protocol:?} at {ranks} ranks");
                }
                for &(other, _) in &protocols {
                    if other == protocol {
                        continue;
                    }
                    let [other_peer, _] = flag_words(other, ranks64);
                    for (a, b) in peer
                        .iter()
                        .chain(&local)
                        .flat_map(|a| other_peer.iter().map(move |b| (a, b)))
                    {
                        assert!(
                            !overlaps(a, b),
                            "{protocol:?} and {other:?} at {ranks} ranks"
                        );
                    }
                }
            }
        }
        assert_eq!(push_buffers(4, 32).unwrap().flag_bytes, (82 + 8) * 4);
        assert_eq!(push_buffers(4, 16).unwrap().flag_bytes, 10 * 4);
    }

    #[test]
    fn every_push_state_zeroes_its_low_latency_entries() {
        let low_latency_rows = u64::from(push_tuning().unwrap().low_latency_rows());
        for ranks in 1..=8_u32 {
            // [parity][source][row][2048] entries of 8 bytes.
            let entries = 2 * u64::from(ranks) * low_latency_rows * 2048 * 8;
            assert!(entries <= ONE_SHOT_BASE);
            for rows in 1..=32 {
                let buffers = push_buffers(ranks, rows).unwrap();
                assert!(buffers.zeroed_inbox_bytes <= buffers.inbox_bytes);
                if ranks.is_power_of_two() {
                    assert!(
                        buffers.zeroed_inbox_bytes >= entries,
                        "{ranks} ranks, {rows} rows"
                    );
                }
            }
        }
        assert_eq!(
            push_buffers(4, 32).unwrap().zeroed_inbox_bytes,
            ONE_SHOT_BASE
        );
        let phased = phased_buffers(4, 8192 * 4096, DType::Bf16).unwrap();
        assert_eq!(phased.zeroed_inbox_bytes, 0);
    }

    #[test]
    fn impossible_comm_sizes_are_refused_instead_of_wrapping() {
        // 2^63 BF16 elements at TP4 used to wrap the inbox to its 8 MiB prefix.
        for elements in [1_u64 << 63, u64::MAX, u64::MAX - 4095] {
            for dtype in [DType::Bf16, DType::F32] {
                assert!(phased_buffers(4, elements, dtype).is_err(), "{elements}");
            }
            let query = CommQuery::AllReduce {
                ranks: 4,
                elements,
                dtype: DType::Bf16,
                fused_residual: false,
                fused_rmsnorm: None,
            };
            assert!(plan(&query, 1).is_err(), "{elements}");
        }
        // The largest representable inbox is still sized exactly.
        let largest = (u64::MAX - PHASED_BASE) / 4 / 4096 * 4096;
        assert_eq!(
            phased_buffers(1, largest, DType::F32).unwrap().inbox_bytes,
            PHASED_BASE + largest * 4
        );
        assert!(phased_buffers(4, 4096, DType::F16).is_err());
        assert!(phased_buffers(4, 4096, DType::U8).is_err());
        // Host and device agree on the fused push and wide row limits.
        assert!(push_buffers(4, 32).is_ok());
        assert!(push_buffers(4, 33).is_err());
        assert!(graph_buffers(4, 33).is_ok() && graph_buffers(4, 256).is_ok());
        assert!(graph_buffers(4, 257).is_err());
        assert!(graph_buffers(3, 33).is_err());
        assert!(push_buffers(4, u32::MAX).is_err());
    }
    #[test]
    fn graphs_send_rows_above_32_through_the_wide_protocol() {
        let fused = |ranks, rows: u64| CommQuery::AllReduce {
            ranks,
            elements: rows * 4096,
            dtype: DType::Bf16,
            fused_residual: true,
            fused_rmsnorm: Some(RmsNormEpilogue {
                kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
                width: 4096,
                eps: 1e-6,
            }),
        };
        for ranks in [1, 2, 4, 8] {
            for rows in [1, 16, 32] {
                assert_eq!(
                    graph_choose(&fused(ranks, rows)).unwrap(),
                    Variant::FusedOneShot
                );
                assert_eq!(
                    graph_plan(&fused(ranks, rows), rows as u32).unwrap(),
                    plan(&fused(ranks, rows), rows as u32).unwrap()
                );
            }
            for rows in [33, 64, 100, 128, 255, 256] {
                assert_eq!(
                    graph_choose(&fused(ranks, rows)).unwrap(),
                    Variant::FusedWide
                );
                // Prefill keeps the phased pair.
                assert_eq!(choose(&fused(ranks, rows)).unwrap(), Variant::FusedTwoShot);
            }
            assert_eq!(
                graph_choose(&fused(ranks, 257)).unwrap(),
                Variant::FusedTwoShot
            );
        }
        // No wide protocol where a row's 512 vectors do not split evenly.
        assert_eq!(graph_choose(&fused(3, 64)).unwrap(), Variant::FusedTwoShot);
        let unfused = CommQuery::AllReduce {
            ranks: 4,
            elements: 64 * 4096,
            dtype: DType::Bf16,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        assert_eq!(graph_choose(&unfused).unwrap(), Variant::TwoShot);
    }

    #[test]
    fn wide_blocks_cover_every_row_once_in_order() {
        // `wide_block` and the workers' `row / per` in push.cuh.
        let push = push_tuning().unwrap();
        let max_blocks = push.wide_blocks(WIDE_MAX_ROWS as u32);
        assert_eq!(max_blocks, 16);
        for rows in 1..=WIDE_MAX_ROWS as u32 {
            let blocks = push.wide_blocks(rows);
            assert!(blocks.is_multiple_of(push.cluster) && blocks <= max_blocks);
            let per = rows.div_ceil(blocks);
            let mut next = 0;
            for block in 0..blocks {
                let first = (block * per).min(rows);
                let end = (first + per).min(rows);
                assert_eq!(first, next, "rows {rows} block {block}");
                for row in first..end {
                    assert_eq!(row / per, block);
                }
                next = end;
            }
            assert_eq!(next, rows);
        }
    }

    /// Inbox bytes a protocol's call writes into a peer, and the bytes a rank
    /// may still read after its last send of the call (a peer can finish and
    /// start its next call from then on), as `push.cuh` lays them out.
    fn inbox_use(
        protocol: Option<PushProtocol>,
        ranks: u64,
        rows: u64,
    ) -> [std::ops::Range<u64>; 3] {
        let empty = 0..0;
        match protocol {
            Some(PushProtocol::LowLatency) => {
                let low_latency = 2 * ranks * 2 * 2048 * 8;
                [0..low_latency, empty, 0..low_latency]
            }
            Some(PushProtocol::OneShot) => {
                let end = ONE_SHOT_BASE + 2 * ranks * PUSH_MAX_ROWS * ROW_BYTES;
                [ONE_SHOT_BASE..end, empty, ONE_SHOT_BASE..end]
            }
            Some(PushProtocol::TwoShot) => {
                let half = PUSH_MAX_ROWS * ROW_BYTES;
                [
                    TWO_SHOT_BASE..TWO_SHOT_BASE + 2 * half,
                    empty,
                    TWO_SHOT_BASE + half..TWO_SHOT_BASE + 2 * half,
                ]
            }
            Some(PushProtocol::Wide) => {
                let slots = WIDE_MAX_ROWS * ROW_BYTES;
                [
                    WIDE_SCATTER_BASE..WIDE_SCATTER_BASE + slots,
                    WIDE_GATHER_BASE..WIDE_GATHER_BASE + slots,
                    WIDE_GATHER_BASE..WIDE_GATHER_BASE + slots,
                ]
            }
            // The phased stages read their inbox before the gather sends.
            None => {
                let bytes = phased_buffers(ranks as u32, rows * 4096, DType::Bf16).unwrap();
                [PHASED_BASE..bytes.inbox_bytes, empty.clone(), empty]
            }
        }
    }

    #[test]
    fn no_protocol_writes_where_another_reads_late() {
        // [first write range, second write range, late reads]. A peer's next
        // call, with any protocol and row count, may write while this rank
        // still reads the late region of its previous call.
        let overlaps =
            |a: &std::ops::Range<u64>, b: &std::ops::Range<u64>| a.start < b.end && b.start < a.end;
        for ranks in [1_u32, 2, 4, 8] {
            let ranks64 = u64::from(ranks);
            let mut uses = vec![
                (None, inbox_use(None, ranks64, 33)),
                (None, inbox_use(None, ranks64, 8192)),
            ];
            for rows in [1_u32, 3, 17, 32, 33, 256] {
                let protocol = graph_protocol(ranks, rows).unwrap();
                uses.push((
                    Some(protocol),
                    inbox_use(Some(protocol), ranks64, u64::from(rows)),
                ));
                // Every range lies inside the protocol's inbox.
                let bytes = graph_buffers(ranks, rows).unwrap().inbox_bytes;
                for range in inbox_use(Some(protocol), ranks64, u64::from(rows)) {
                    assert!(range.end <= bytes, "{protocol:?} at {ranks} ranks");
                }
            }
            for (previous, [_, _, late]) in &uses {
                for (next, [first, second, _]) in &uses {
                    if previous == next {
                        continue;
                    }
                    for write in [first, second] {
                        assert!(
                            !overlaps(write, late),
                            "{next:?} writes where {previous:?} reads late at {ranks} ranks"
                        );
                    }
                }
            }
        }
        // The wide protocol's own scatter slots lie outside its gather slots.
        let [scatter, gather, _] = inbox_use(Some(PushProtocol::Wide), 4, 256);
        assert!(!overlaps(&scatter, &gather));
    }
}
