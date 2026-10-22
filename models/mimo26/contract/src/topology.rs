//! Topology identity and per-rank resource sizes at each supported rank
//! count. TP4's values are the `contract` constants every TP4 package and
//! launch journal already has.

pub use crate::Tp;
pub use crate::encoders::Encoders;

use crate::serving::{BF16_BYTES, DECODE_ENTRY_ROWS, PAGE_TOKENS};

/// Topology ID of the package at `tp`.
pub fn topology_id(tp: Tp) -> &'static str {
    static IDS: std::sync::OnceLock<Vec<(Tp, String)>> = std::sync::OnceLock::new();
    IDS.get_or_init(|| {
        crate::geometry::GEOMETRIES
            .iter()
            .flat_map(|g| {
                Tp::SUPPORTED.map(|tp| {
                    (
                        Tp::of_geometry(g, tp.ranks()).unwrap(),
                        format!("mimo-{}-tp{}", g.id, tp.ranks()),
                    )
                })
            })
            .collect()
    })
    .iter()
    .find(|(shape, _)| *shape == tp)
    .expect("listed topology")
    .1
    .as_str()
}

/// The geometry and rank count named by a canonical topology ID.
pub fn from_topology_id(id: &str) -> Option<Tp> {
    let (geometry, ranks) = id.strip_prefix("mimo-")?.rsplit_once("-tp")?;
    let geometry = crate::geometry::GEOMETRIES
        .iter()
        .find(|g| g.id == geometry)?;
    let shape = Tp::of_geometry(geometry, ranks.parse().ok()?).ok()?;
    (topology_id(shape) == id).then_some(shape)
}

/// One GA KV page of every GA layer on one rank, BF16 or FP8.
pub fn ga_page_bytes(tp: Tp, fp8: bool) -> u64 {
    let element = if fp8 { 1 } else { BF16_BYTES };
    tp.geometry.global_layers() as u64
        * tp.kv_heads(true)
        * PAGE_TOKENS
        * (tp.geometry.head_dim() + tp.geometry.value_head_dim()) as u64
        * element
}

/// One slot's SWA ring over every SWA layer on one rank.
pub fn ring_slot_bytes(tp: Tp) -> u64 {
    tp.geometry.sliding_layers() as u64
        * tp.kv_heads(false)
        * tp.geometry.window() as u64
        * (tp.geometry.swa_head_dim() + tp.geometry.swa_value_head_dim()) as u64
        * BF16_BYTES
}

/// One state slot of the SWA ring state on one rank: the slot's working and
/// shadow rings. A package declares its state per slot
/// (`chop.state-per-item.v1`).
pub fn ring_state_slot_bytes(tp: Tp) -> u64 {
    2 * ring_slot_bytes(tp)
}

/// The SWA ring state of one rank with `slots` ring slots: the working
/// rings of every slot, then their shadow. An instance holds one slot per
/// item of its largest graph bucket (`Items::capacity`).
pub fn ring_state_bytes(tp: Tp, slots: u32) -> u64 {
    u64::from(slots) * ring_state_slot_bytes(tp)
}

/// The SWA ring state of an instance serving 32 sequences, the deployment
/// default.
pub fn ring_state_bytes_per_rank(tp: Tp) -> u64 {
    ring_state_bytes(tp, DECODE_ENTRY_ROWS)
}

pub use model_compiler_sdk::device::DEVICE_BYTES;
/// The average context, prompt and output so far, the package maximum of
/// sequences keeps KV for in the package's memory estimate.
const AVERAGE_CONTEXT_TOKENS: u64 = 8192;
/// What the host keeps before it sizes the KV pool: its reserve and the
/// 512 MiB sampling workspace.
const HOST_RESERVED_BYTES: u64 = model_compiler_sdk::device::HOST_RESERVE_BYTES + (1 << 29);
/// The share of the rest the package's memory estimate counts for the KV
/// pool, in tenths: a conservative planning margin, below the host's default
/// of all of it.
const KV_TENTHS: u64 = 9;

/// The device memory one rank may take: a device's [`DEVICE_BYTES`].
pub fn rank_budget(_tp: Tp) -> u64 {
    DEVICE_BYTES
}

/// Global-attention KV tokens one rank holds when it serves `sequences`
/// sequences with its images (`image_bytes`, the largest rank's text,
/// drafter and encoder images), `drafter` and `encoders`: the host's KV
/// fraction of the rank budget left after them, the rank's working set and
/// the host's reserve, in BF16 pages.
pub fn kv_tokens(
    tp: Tp,
    image_bytes: u64,
    sequences: u32,
    drafter: Option<crate::weight_plan::Drafter>,
    encoders: Encoders,
) -> u64 {
    let used = image_bytes
        + working_bytes(tp, sequences, drafter)
        + encoders.media_bytes_for(tp.geometry)
        + HOST_RESERVED_BYTES;
    let pool = rank_budget(tp).saturating_sub(used) / 10 * KV_TENTHS;
    pool / (ga_page_bytes(tp, false) / PAGE_TOKENS)
}

/// The most live sequences an instance serves: the largest graph bucket, up
/// to 256, whose KV holds each sequence an 8K average context
/// ([`kv_tokens`]), at least one. The MTP drafter drafts and catches up at
/// the decode kernels' 32 rows, so an MTP package serves at most 32
/// ([`crate::serving::DRAFT_ITEMS`]). With the checkpoint's images TP4
/// serves 256; TP2, whose ranks hold twice the weights, 64 (32
/// with DFlash), and half that with both media encoders. Packages declare
/// this, and the host serves at most that many.
pub fn maximum_sequences(
    tp: Tp,
    drafter: Option<crate::weight_plan::Drafter>,
    encoders: Encoders,
    image_bytes: u64,
) -> u32 {
    let limit = if drafter == Some(crate::weight_plan::Drafter::Mtp) {
        crate::serving::DRAFT_ITEMS
    } else {
        crate::serving::DECODE_BATCH
    };
    (0..=limit.ilog2())
        .rev()
        .map(|power| 1 << power)
        .find(|&sequences| {
            kv_tokens(tp, image_bytes, sequences, drafter, encoders)
                >= u64::from(sequences) * AVERAGE_CONTEXT_TOKENS
        })
        .unwrap_or(1)
}

/// Vocabulary rows of one rank's `lm_head` shard.
pub fn vocab_per_rank(tp: Tp) -> u32 {
    tp.vocab() as u32
}

/// The name tag of the topology's own kernels: the geometry's tag (`mimo`)
/// at TP4, whose names predate the other topologies, then `_tp` and the
/// rank count otherwise (`mimo_tp2`). The kernel profiles carry the same tags.
pub fn kernel_tag(tp: Tp) -> &'static str {
    match tp.ranks() {
        4 => tp.geometry.kernel_tag,
        ranks => crate::intern(&format!("{}_tp{ranks}", tp.geometry.kernel_tag)),
    }
}

/// The `_tp2` infix of per-topology row kernel names; empty at TP4.
pub fn row_infix(tp: Tp) -> &'static str {
    kernel_tag(tp)
        .strip_prefix(tp.geometry.kernel_tag)
        .expect("every MiMo kernel tag extends its geometry's")
}

/// Room a rank keeps beyond what its bundle allocates: the loaded modules,
/// the host's sampling workspace and 1 GiB reserve, and a small KV pool.
const RANK_SLACK_BYTES: u64 = 4 << 30;

/// Sequences the working set model starts from.
const BASE_SEQUENCES: u32 = crate::serving::DECODE_ENTRY_ROWS;
/// What `create` allocates on every rank whole at 32 sequences, verifying
/// the most rows a step holds: the logits, input arenas, hidden rows, comm
/// state, planner workspace and RoPE tables at the served context.
const WHOLE_WORKING_BYTES: u64 = 2816 << 20;
/// What it splits over the ranks at 32 sequences: the sharded attention,
/// MoE and logits rows.
const SPLIT_WORKING_BYTES: u64 = 2520 << 20;
/// What each served sequence (rounded up to the next graph bucket) adds
/// past 32 besides its ring slot: its arenas, block tables and restore rows.
const SEQUENCE_BYTES: u64 = 1 << 19;
/// The DFlash drafter's buffers, split over the ranks: its taps and pairs,
/// and per state slot its KV cache and history.
const DFLASH_FIXED_BYTES: u64 = 512 << 20;
const DFLASH_SLOT_BYTES: u64 = 81 << 20;
/// The MTP drafter's buffers on every rank.
const MTP_BYTES: u64 = 64 << 20;
/// Device bytes `create` allocates on one rank, and its SWA ring state,
/// when it serves `sequences` sequences with `drafter`, verifying the most
/// rows a step holds with bundle or host drafts. A model: whole and split
/// parts that cover up to 32 sequences, a per-sequence part above, and the
/// ring state and DFlash cache of the served count's graph bucket of slots.
/// The bundle's tests check that it bounds what `create` allocates on the
/// fake host at every graph bucket each topology admits.
fn flash_working_bound(
    tp: Tp,
    sequences: u32,
    drafter: Option<crate::weight_plan::Drafter>,
) -> u64 {
    let slots = sequences.next_power_of_two();
    let ranks = u64::from(tp.ranks());
    let drafter = match drafter {
        Some(crate::weight_plan::Drafter::Dflash) => {
            (DFLASH_FIXED_BYTES + u64::from(slots) * DFLASH_SLOT_BYTES) / ranks
        }
        Some(crate::weight_plan::Drafter::Mtp) => MTP_BYTES,
        None => 0,
    };
    WHOLE_WORKING_BYTES
        + SPLIT_WORKING_BYTES / ranks
        + u64::from(slots.saturating_sub(BASE_SEQUENCES)) * SEQUENCE_BYTES
        + ring_state_bytes(tp, slots)
        + drafter
}

/// Exact named allocations plus the pinned Flash reservation for prefix pages
/// and allocation padding. The reservation does not absorb or
/// rescale geometry-dependent rows; every such row is in `working_rows`.
pub fn working_bytes(tp: Tp, sequences: u32, drafter: Option<crate::weight_plan::Drafter>) -> u64 {
    let flash = Tp::new(tp.ranks()).expect("supported ranks");
    let flash_rows = crate::resources::working_row_bytes(flash, sequences, drafter);
    let residual = flash_working_bound(flash, sequences, drafter)
        .checked_sub(flash_rows)
        .expect("Flash row table fits the pinned working bound");
    crate::resources::working_row_bytes(tp, sequences, drafter) + residual
}

/// Device bytes one rank of a package needs before its KV pool can grow,
/// for the manifest's resource minimum: `image_bytes`, the largest rank's
/// text, drafter and encoder images, the [`working_bytes`] of `sequences`,
/// the package maximum, with `drafter`, the media bytes of `encoders`
/// ([`Encoders::media_bytes`]), and `RANK_SLACK_BYTES`, rounded up to GiB.
pub fn minimum_rank_bytes(
    tp: Tp,
    image_bytes: u64,
    sequences: u32,
    drafter: Option<crate::weight_plan::Drafter>,
    encoders: Encoders,
) -> u64 {
    let total = image_bytes
        + working_bytes(tp, sequences, drafter)
        + encoders.media_bytes_for(tp.geometry)
        + RANK_SLACK_BYTES;
    total.div_ceil(1 << 30) << 30
}
