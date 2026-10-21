//! Host-owned resource acquisition and instance lifecycle.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::thread::LocalKey;

use bundle_runtime::acquire::{self, Log};
use bundle_runtime::entry::{Bundle, Failed};
use bundle_runtime::prefix::PrefixInstance;
use bundle_runtime::resources::{Held, allocate_span, status, upload_bytes};
use bundle_runtime::table::{InstanceTable, Live};
use bundle_runtime::transaction::{self, End, Ended, Failure, Retire, Stage, Steps};
#[cfg(test)]
use runtime_bundle_abi::BundleHandleV1;
use runtime_bundle_abi::{
    BundleAbiError, BundleContextHandleV1, BundleCudaSymbolHandleV1, BundleEventHandleV1,
    BundleMemoryPoolHandleV1, BundleStatusV1, BundleStreamHandleV1, DeviceBufferSpanV1,
    ExecuteResultSinkV1, HostServicesV1,
};
use runtime_bundle_sdk::BatchCodecV1;

use crate::CUBINS;

#[path = "dflash/runtime.rs"]
mod dflash;
#[path = "mtp/runtime.rs"]
mod mtp;
#[path = "prefix_state.rs"]
mod prefix_state;
use bundle_runtime::geometry::Items;

use crate::contract::{DECODE_ENTRY_ROWS, MAX_VERIFY_ROWS, Mimo, PAGE_TOKENS, PREFILL_TOKENS};
use crate::selection::MimoPhase as _;
use crate::target::Target;
use crate::topology::Tp;
use crate::upload::{DeviceInputs, PhaseArena};
use bundle_runtime::ledger::{CommitPlan, Item, Ledger, LedgerError, Placement, Rules, Step};
use bundle_runtime::ring::{Cell, Tail, WindowRing};
use model_mimo26_contract::backend::{Kernels, Selected};
use model_mimo26_contract::encoders::Encoders;

const RANK_IMAGE_ROLE: &[u8] = b"mimo.rank-images.v1";
const RING_STATE_ID: &[u8] = b"mimo.swa-ring.v1";

struct Configuration {
    context_tokens: u64,
    tp: Tp,
    /// One distinct device ordinal per rank.
    ordinals: Vec<i32>,
    /// The declared KV format, with this rank count's page bytes.
    kv: bundle_runtime::kv::KvFormat,
    state_id: Vec<u8>,
    media: bool,
    /// Whether the package names the media its encoders serve
    /// (`weight_plan::ENCODER_SET_ROLE`); without it they serve every one.
    encoder_set: bool,
    drafter: bool,
    /// The sequences the instance serves (the host's `--max-sequences`).
    items: Items,
    /// The serving process's startup nonce, which tap steps record.
    #[cfg_attr(not(feature = "diagnostic-taps"), allow(dead_code))]
    tap_producer: Option<String>,
}

fn decode_configuration(bytes: &[u8], log: impl Fn(&str)) -> Result<Configuration, BundleStatusV1> {
    let id = bundle_runtime::config::topology_id(bytes)?;
    let tp = crate::topology::from_topology_id(id).ok_or(BundleStatusV1::Unsupported)?;
    let topologies = [bundle_runtime::config::Topology {
        id: crate::topology::topology_id(tp),
        ranks: bundle_runtime::geometry::Ranks::new(tp.ranks())
            .ok_or(BundleStatusV1::Unsupported)?,
    }];
    let config = bundle_runtime::config::decode_with_log::<Mimo>(bytes, &topologies, &log)?;
    config.validate_context(tp.model_limits())?;
    let context_tokens = config
        .context_tokens
        .unwrap_or(tp.geometry.context_tokens() as u64);
    if !crate::rope_tables::valid_context_for(tp.geometry, context_tokens) {
        return Err(BundleStatusV1::InvalidArgument);
    }
    let mut has_image = false;
    let mut has_index = vec![false; tp.count()];
    let mut has_drafter_image = false;
    let mut has_drafter_index = vec![false; tp.count()];
    let mut has_encoder_image = false;
    let mut has_encoder_index = vec![false; tp.count()];
    let mut has_encoder_set = false;
    for role in &config.roles {
        has_encoder_set |= role == model_mimo26_contract::weight_plan::ENCODER_SET_ROLE.as_bytes();
        has_image |= role == RANK_IMAGE_ROLE;
        has_drafter_image |= role == crate::mtp::views::IMAGE_ROLE;
        has_encoder_image |= role == crate::encoders::views::IMAGE_ROLE;
        for rank in 0..tp.ranks() {
            has_drafter_index[rank as usize] |= role
                == format!("chop.bundle-metadata.mimo.drafter-rank-index.r{rank}.v1").as_bytes();
            has_index[rank as usize] |=
                role == format!("chop.bundle-metadata.mimo.rank-index.r{rank}.v1").as_bytes();
            has_encoder_index[rank as usize] |= role
                == format!("chop.bundle-metadata.mimo.encoder-rank-index.r{rank}.v1").as_bytes();
        }
    }
    let (kv, state, items) = (config.kv, config.state, config.items);
    // Select the geometry-owned calibration by its compiled layout ID.
    let formats = model_mimo26_contract::serving::kv_formats(tp);
    let Some(format) = formats
        .iter()
        .find(|format| format.layout_id == kv.layout_id.as_slice())
    else {
        log(&format!(
            "unknown KV layout {} for geometry {}",
            String::from_utf8_lossy(&kv.layout_id),
            tp.geometry.id
        ));
        return Err(BundleStatusV1::Unsupported);
    };
    if format.page_bytes != kv.page_bytes {
        log(&format!(
            "KV layout {} page bytes: expected {}, received {}",
            String::from_utf8_lossy(format.layout_id),
            format.page_bytes,
            kv.page_bytes
        ));
        return Err(BundleStatusV1::Unsupported);
    }
    let format = *format;
    if kv.page_tokens != PAGE_TOKENS
        || kv.ranks != u64::from(tp.ranks())
        || kv.lifetime != 0
        || kv.alignment < 256
        || !kv.alignment.is_power_of_two()
        || state.state_id != RING_STATE_ID
        // One state slot per item of the largest bucket
        // (`chop.state-per-item.v1`).
        || state.bytes != crate::topology::ring_state_bytes(tp, items.capacity())
        || state.alignment < 256
        || !state.alignment.is_power_of_two()
        || (has_drafter_image && has_drafter_index.contains(&false))
        || (!has_drafter_image && has_drafter_index.contains(&true))
        || !has_image
        || has_index.contains(&false)
        || (has_encoder_image && has_encoder_index.contains(&false))
        || (!has_encoder_image && has_encoder_index.contains(&true))
        || (!has_encoder_image && has_encoder_set)
        || (has_encoder_image && !crate::topology::encoders_available(tp))
    {
        return Err(BundleStatusV1::Unsupported);
    }
    if config.ordinals.len() != tp.count() {
        return Err(BundleStatusV1::InvalidArgument);
    }
    let precision_role = model_mimo26_contract::prepared::COMPONENT_ROLE.as_bytes();
    if !config.roles.iter().any(|role| role == precision_role) {
        // Packages are prepared from the checkpoint; an older package is
        // rebuilt by `serve` or `package` rather than served another way.
        log(&format!(
            "package lacks the precision component {}; rebuild it from the checkpoint",
            model_mimo26_contract::prepared::COMPONENT_ROLE
        ));
        return Err(BundleStatusV1::Unsupported);
    }
    Ok(Configuration {
        context_tokens,
        tp,
        ordinals: config.ordinals,
        kv: format,
        state_id: state.state_id,
        media: has_encoder_image,
        encoder_set: has_encoder_set,
        drafter: has_drafter_image,
        items,
        tap_producer: config.tap_producer,
    })
}

pub(crate) struct Instance {
    context_tokens: u64,
    services: HostServicesV1,
    /// The package topology: the rank count and every per-rank shape.
    tp: Tp,
    /// The GPU target its ops and own kernels run on, built at create.
    target: Box<dyn Target>,
    /// The sequences the instance serves (the host's `--max-sequences`):
    /// slot tables hold one entry per sequence, and per-row buffers, graph
    /// buckets and the ring's slots cover the largest bucket.
    items: Items,
    /// Decode, verify, MTP and DFlash steps enqueue their planned launches
    /// eagerly instead of replaying graphs when tests request it.
    eager: bool,
    held: Vec<Held>,
    /// The declared KV format, whose element sizes the declared state.
    kv: bundle_runtime::kv::KvFormat,
    /// The SWA ring's byte layout, which its kernels address.
    ring: crate::ring::RingGeometry,
    /// Each request's ring slot and committed tokens, and the open step.
    ledger: Ledger,
    /// The declared SWA ring's cells, which a step saves and a cancel or a
    /// rejected tail restores.
    window: WindowRing,
    /// Per rank, the ring slots whose backup this transaction has enqueued.
    /// Cancel restores only these; the others were never overwritten.
    ring_saved: Vec<Vec<u32>>,
    views: Vec<BTreeMap<String, DeviceBufferSpanV1>>,
    precision: Option<crate::projection::Prepared>,
    /// The vision and audio encoders, when the package ships them.
    encoder: Option<bundle_runtime::encoder::Encoder>,
    contexts: Vec<BundleContextHandleV1>,
    streams: Vec<BundleStreamHandleV1>,
    state_views: Vec<DeviceBufferSpanV1>,
    kv_views: Vec<DeviceBufferSpanV1>,
    kv_page_capacity: u64,
    #[allow(
        dead_code,
        reason = "used when the rank program is connected to execute"
    )]
    kv_dtype: kernel_core::contract::DType,
    /// Per global layer, the K and V scales of an FP8 pool; empty is 1.0.
    kv_scales: &'static [[f32; 2]],
    /// Decode and verify graphs above 32 rows take the wide all-reduce: off
    /// for `CHOPINFER_MIMO_WIDE_COMM=0` or when the device cannot keep the
    /// wide grids of the ranks on it resident (`check_comm_residency`).
    wide_comm: bool,
    /// The environment switches every step follows, read at create.
    switches: crate::switches::Switches,
    /// The tracing, eager verify and decode folding controls, read at
    /// create.
    controls: crate::switches::Controls,
    /// The ops prepared through the backend at create.
    ops: crate::ops::Ops,
    /// Every logits row comes from the F32 head unless `F32_LOGITS` is `0`.
    f32_logits: bool,
    /// Plain decode logits rows go to their owning ranks (`ROW_OWNED_LOGITS`)
    /// when no drafter is configured.
    row_owned_logits: bool,
    /// With `row_owned_logits`, also verify rows and a DFlash instance's
    /// decode rows (`ROW_OWNED_SPECULATION`), both on by default.
    row_owned_speculation: bool,
    /// The smallest verify token bucket (`VERIFY_BUCKET`), or zero.
    verify_bucket: u32,
    /// Per rank, the fixed decode inputs every decode graph binds.
    decode_inputs: Vec<DeviceInputs>,
    /// Per rank, the fixed verify inputs every verify graph binds: the
    /// decode arrays, with token arrays for the widest verify step.
    verify_inputs: Vec<DeviceInputs>,
    /// Per rank, this step's prefill inputs, packed around its live counts.
    prefill_inputs: Vec<DeviceInputs>,
    prefill_layout: crate::upload::InputLayout,
    /// Per input set (prefill, decode, two prefill halves, verify) and
    /// rank, the input arena and block table.
    arenas: [Vec<PhaseArena>; 5],
    /// Per input set, what the ranks' block-table rows hold.
    block_tables: [crate::upload::BlockTableMirror; 5],
    /// Per rank, the stream a pipelined prefill runs its all-reduces on, and
    /// the events that order it against the rank's stream.
    comm_streams: Vec<BundleStreamHandleV1>,
    overlap_events: Vec<Vec<BundleEventHandleV1>>,
    /// Set once work reached some ranks but not their peers; see
    /// `pipeline::Collectives`.
    collectives: crate::pipeline::Collectives,
    /// How prefill chunks pipeline over two halves (`overlap`), read from
    /// the environment at create; `None` runs them unpipelined.
    overlap: Option<crate::overlap::Config>,
    /// Tests fold decode rows without the process environment.
    #[cfg(test)]
    fold_override: Option<bool>,
    activations: Vec<DeviceBufferSpanV1>,
    normalized: Vec<DeviceBufferSpanV1>,
    workspaces: Vec<DeviceBufferSpanV1>,
    logits: Vec<LogitsBuffers>,
    symbols: Vec<BTreeMap<String, BundleCudaSymbolHandleV1>>,
    scratch: Vec<crate::scratch::RankScratch>,
    graphs: crate::graphs::GraphCache,
    /// The verify rows the host enabled at create; none when speculation is
    /// off.
    verify_limits: crate::inputs::VerifyLimits,
    /// Per rank, the rejected ranges the ring restore reads, then a prefix
    /// snapshot's page table.
    restore_inputs: Vec<DeviceBufferSpanV1>,
    mtp: Option<mtp::Mtp>,
    dflash: Option<dflash::Dflash>,
    /// Per-layer taps (`taps`), when the environment asks for them.
    #[cfg(feature = "diagnostic-taps")]
    taps: Option<runtime_bundle_sdk::taps::Taps>,
    /// This step's encoder rows, read for the taps of its prefill.
    #[cfg(feature = "diagnostic-taps")]
    tap_encoders: Vec<runtime_bundle_sdk::taps::EncoderRows>,
    /// Each prompt's media occurrences across its prefill steps, for taps.
    #[cfg(feature = "diagnostic-taps")]
    tap_occurrences: bundle_runtime::media::Occurrences,
    /// Captures awaiting the transaction outcome.
    #[cfg(feature = "diagnostic-taps")]
    pending_taps: Vec<runtime_bundle_sdk::taps::TapStep>,
}

/// Byte layout of one rank's restore input buffer: the live item count, then
/// per item the slot, first rejected position and rejected count, each in a
/// region of one u32 per item of the largest bucket rounded up to 128 bytes,
/// then a prefix snapshot's page table and a DFlash snapshot header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RestoreLayout {
    region: u64,
}

const PREFIX_PAGES_BYTES: u64 = 256 * 8;

impl RestoreLayout {
    const fn new(items: Items) -> Self {
        Self {
            region: (items.capacity() as u64 * 4).next_multiple_of(128),
        }
    }
    const LIVE: u64 = 0;
    const fn slots(self) -> u64 {
        self.region
    }
    const fn starts(self) -> u64 {
        2 * self.region
    }
    const fn counts(self) -> u64 {
        3 * self.region
    }
    const fn prefix_pages(self) -> u64 {
        4 * self.region
    }
    const fn prefix_metadata(self) -> u64 {
        self.prefix_pages() + PREFIX_PAGES_BYTES
    }
    #[cfg(test)]
    const fn bytes(self) -> u64 {
        self.prefix_metadata() + crate::prefix_projected::HEADER_BYTES
    }
}

struct Acquired {
    mtp: Option<mtp::Mtp>,
    dflash: Option<dflash::Dflash>,
    views: Vec<BTreeMap<String, DeviceBufferSpanV1>>,
    precision: crate::projection::Prepared,
    /// The vision and audio encoders, when the package ships them.
    encoder: Option<bundle_runtime::encoder::Encoder>,
    contexts: Vec<BundleContextHandleV1>,
    streams: Vec<BundleStreamHandleV1>,
    state_views: Vec<DeviceBufferSpanV1>,
    kv_views: Vec<DeviceBufferSpanV1>,
    kv_page_capacity: u64,
    arenas: [Vec<PhaseArena>; 5],
    comm_streams: Vec<BundleStreamHandleV1>,
    overlap_events: Vec<Vec<BundleEventHandleV1>>,
    activations: Vec<DeviceBufferSpanV1>,
    normalized: Vec<DeviceBufferSpanV1>,
    workspaces: Vec<DeviceBufferSpanV1>,
    logits: Vec<LogitsBuffers>,
    symbols: Vec<BTreeMap<String, BundleCudaSymbolHandleV1>>,
    scratch: Vec<crate::scratch::RankScratch>,
    restore_inputs: Vec<DeviceBufferSpanV1>,
}

const PREFILL: usize = 0;
const DECODE: usize = 1;
/// Input sets of a pipelined prefill's two halves.
const HALVES: [usize; 2] = [2, 3];
const VERIFY: usize = 4;

#[derive(Clone, Copy)]
struct LogitsBuffers {
    local: DeviceBufferSpanV1,
    gathered_prefill: DeviceBufferSpanV1,
    gathered_decode: DeviceBufferSpanV1,
    local_scored: DeviceBufferSpanV1,
    gathered_scored: DeviceBufferSpanV1,
}

/// What one phase's rank programs are lowered from.
struct PhaseProgram {
    phase: crate::selection::Phase,
    stages: Vec<crate::program::Stage>,
    planned_tokens: u32,
    live_batch: u32,
    context: u32,
    logit_rows: Vec<u32>,
    scored_prompt_logits: bool,
}

impl PhaseProgram {
    /// With `f32_logits`, every logits row uses the F32 head that scored
    /// prompt rows use, not the BF16 head. A verify phase runs in at least
    /// the `verify_bucket` token bucket (see `VERIFY_BUCKET`).
    fn new(
        tp: Tp,
        inputs: &crate::inputs::PhaseInputs,
        f32_logits: bool,
        verify_bucket: u32,
    ) -> Result<Option<Self>, BundleStatusV1> {
        if inputs.live_batch() == 0 {
            return Ok(None);
        }
        let (verify_tokens, verify_items) = verify_shape(inputs, verify_bucket);
        let phase = if inputs.verify {
            crate::selection::Phase::Verify(verify_tokens)
        } else if inputs.decode {
            crate::selection::Phase::Decode(inputs.live_batch())
        } else {
            crate::selection::Phase::Prefill
        };
        let bucketed = phase.token_bucketed();
        // Verify can split sliding-window items. The attention planner omits
        // its merge only for buckets whose partial writes the final output.
        let stages = crate::program::stages_for(tp)
            .into_iter()
            .filter(|stage| {
                bucketed
                    || !matches!(
                        stage,
                        crate::program::Stage::AttentionMerge(layer)
                            if !tp.geometry.is_global(*layer)
                    )
            })
            .collect();
        let token_bucket = phase
            .capacity()
            .map_err(|_| BundleStatusV1::InvalidArgument);
        Ok(Some(Self {
            phase,
            stages,
            planned_tokens: if inputs.decode {
                token_bucket?
            } else {
                inputs.live_tokens()
            },
            live_batch: if inputs.verify {
                verify_items
            } else {
                inputs.live_batch()
            },
            // Verify plans attention for the largest context, so its plan
            // does not depend on the step and can be replayed as a graph.
            context: match phase {
                crate::selection::Phase::Decode(_) => 1,
                crate::selection::Phase::Verify(_) => tp.geometry.context_tokens(),
                crate::selection::Phase::Prefill => {
                    inputs.seq_lens.iter().copied().max().unwrap_or(1) as u32
                }
            },
            // Decode-capacity heads project every row of the token bucket.
            logit_rows: if inputs.decode {
                (0..token_bucket?).collect()
            } else {
                inputs.logit_rows()
            },
            scored_prompt_logits: f32_logits
                || !inputs.decode && inputs.prompt_rows.iter().any(|&row| row),
        }))
    }
}

/// Decode, verify and unscored prefill rows take F32 logits from the F32
/// head, as scored prompt rows do. Set to `0` for the BF16 head.
pub const F32_LOGITS: &str = "CHOPINFER_MIMO_F32_LOGITS";

fn f32_logits_from_environment() -> bool {
    std::env::var_os(F32_LOGITS).is_none_or(|value| value != "0")
}

/// `1` runs a step's plain decode rows inside its prefill forward, as
/// one-token prefill items, so the step reads the weights once instead of
/// running the decode graph after the prefill (docs/serving-profiles.md#prefill-policy).
/// Off by default. Instances with a drafter, steps with verify rows or
/// media, and the BF16 logits head (`CHOPINFER_MIMO_F32_LOGITS=0`) keep the
/// decode graph.
pub const FOLD_DECODE: &str = "CHOPINFER_MIMO_FOLD_DECODE";

/// Plain decode logits rows are exchanged so each rank holds and samples a
/// block of full rows, instead of gathering every row to rank 0. Set to `0`
/// for the rank-0 gather. Prefill and MTP steps keep the gather.
pub const ROW_OWNED_LOGITS: &str = "CHOPINFER_MIMO_ROW_OWNED_LOGITS";

#[cfg(test)]
thread_local! {
    /// Tests force the exchange on or off for instances created on their
    /// thread.
    pub(crate) static ROW_OWNED_LOGITS_FOR_TEST: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn row_owned_logits_from_environment() -> bool {
    #[cfg(test)]
    if let Some(forced) = ROW_OWNED_LOGITS_FOR_TEST.with(std::cell::Cell::get) {
        return forced;
    }
    std::env::var_os(ROW_OWNED_LOGITS).is_none_or(|value| value != "0")
}

/// With `ROW_OWNED_LOGITS`, verify rows (every verify bucket up to 256
/// rows) and the decode rows of a DFlash instance are exchanged as well.
/// The DFlash draft block reads its own head's logits, which it writes
/// after the exchange has sent the target rows. Set to `0` to keep those
/// rows on the rank-0 gather. MTP instances and prefill always gather.
pub const ROW_OWNED_SPECULATION: &str = "CHOPINFER_MIMO_ROW_OWNED_SPECULATION";

#[cfg(test)]
thread_local! {
    pub(crate) static ROW_OWNED_SPECULATION_FOR_TEST: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn row_owned_speculation_from_environment() -> bool {
    #[cfg(test)]
    if let Some(forced) = ROW_OWNED_SPECULATION_FOR_TEST.with(std::cell::Cell::get) {
        return forced && row_owned_logits_from_environment();
    }
    row_owned_logits_from_environment()
        && std::env::var_os(ROW_OWNED_SPECULATION).is_none_or(|value| value != "0")
}

/// The widest comm bucket whose logits rows the exchange serves, or zero:
/// every verify bucket, or every decode bucket of an instance serving
/// `items`.
fn exchanged_rows_through(items: Items) -> u32 {
    if row_owned_speculation_from_environment() {
        crate::contract::VERIFY_TOKENS.max(items.capacity())
    } else if row_owned_logits_from_environment() {
        items.capacity().max(DECODE_ENTRY_ROWS)
    } else {
        0
    }
}

/// Qualification only: every verify step runs in at least this token
/// bucket (64, 128 or 256), with its rows padded, so each wide bucket's
/// numerics can be scored on the same teacher-forced rows as narrower steps.
pub const VERIFY_BUCKET: &str = "CHOPINFER_MIMO_VERIFY_BUCKET";

/// The forced verify bucket, when `VERIFY_BUCKET` names one within the
/// enabled rows; zero otherwise.
fn verify_bucket_from_environment(rows_per_step: u32) -> u32 {
    parse_verify_bucket(std::env::var(VERIFY_BUCKET).ok().as_deref(), rows_per_step)
}

fn parse_verify_bucket(value: Option<&str>, rows_per_step: u32) -> u32 {
    value
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|&bucket| {
            crate::graphs::verify_bucket(bucket) == Some(bucket)
                && crate::graphs::verify_bucket(rows_per_step).is_some_and(|rows| bucket <= rows)
        })
        .unwrap_or(0)
}

/// The rows and items a verify phase is planned and keyed for: its live
/// counts, raised to `verify_bucket` rows over enough item slots to hold
/// them. Padded rows and items are not live; the kernels read live counts.
fn verify_shape(inputs: &crate::inputs::PhaseInputs, verify_bucket: u32) -> (u32, u32) {
    (
        inputs.live_tokens().max(verify_bucket),
        inputs
            .live_batch()
            .max(verify_bucket.div_ceil(crate::contract::MAX_VERIFY_ROWS)),
    )
}

/// Shape metadata only: keep graph replay and device synchronization unchanged.
fn step_shape_trace(
    step: &crate::inputs::PackedStep,
    items: Items,
    forced_verify_rows: u32,
) -> serde_json::Value {
    let decode = &step.decode;
    let (rows, item_count) = verify_shape(decode, forced_verify_rows);
    let capacity = if decode.live_batch() == 0 {
        None
    } else if decode.verify {
        bundle_runtime::graphs::verify_bucket_within::<Mimo>(rows, items)
    } else {
        items.bucket(decode.live_batch())
    };
    serde_json::json!({
        "schema": "chopinfer.mimo.step-shape.v1",
        "prefill_rows": step.prefill.live_tokens(),
        "live_target_rows": decode.live_tokens(),
        "graph_capacity": capacity,
        "graph_item_capacity": items.bucket(if decode.verify { item_count } else { decode.live_batch() }),
        "verify": decode.verify,
        "request_ids": decode.request_ids,
        "query_rows": decode.q_len,
        // Committed tokens before each item's first row: with the prompt
        // length, the output positions whose logits these rows computed.
        "prefix_len": decode.prefix_len,
        "prefill_request_ids": step.prefill.request_ids,
        "prefill_query_rows": step.prefill.q_len,
        "prefill_prefix_len": step.prefill.prefix_len,
        "draft_tokens": decode.q_len.iter().zip(&decode.verify_rows)
            .map(|(&rows, &verify)| if verify { rows - 1 } else { 0 }).collect::<Vec<_>>(),
    })
}

fn trace_unix_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
thread_local! {
    /// Tests force an instance to run eagerly (`Some(true)`)
    /// or with graphs (`Some(false)`), without the process environment.
    pub(crate) static FORCE_EAGER: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// Tests trace every step's launches, as `CHOPINFER_MIMO_TRACE_LAUNCHES`
    /// does, without the process environment.
    pub(crate) static FORCE_TRACE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Every launch the stages lower on this thread: its rank, symbol and
    /// the rows its plan processes (`PreparedLaunch::rows`), for the
    /// invariance journal.
    pub(crate) static PLANNED: std::cell::RefCell<Vec<(usize, String, u32)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether tests force traced launches.
fn traced() -> bool {
    #[cfg(test)]
    if FORCE_TRACE.with(std::cell::Cell::get) {
        return true;
    }
    false
}

/// Whether tests force eager execution.
fn eager() -> bool {
    #[cfg(test)]
    if let Some(eager) = FORCE_EAGER.with(std::cell::Cell::get) {
        return eager;
    }
    false
}

impl Instance {
    /// Whether `phase`'s logits rows go to their owning ranks.
    fn exchanges_logits(&self, phase: crate::selection::Phase) -> bool {
        self.row_owned_logits
            && self.mtp.is_none()
            && match phase {
                crate::selection::Phase::Decode(_) => {
                    self.dflash.is_none() || self.row_owned_speculation
                }
                crate::selection::Phase::Verify(_) => self.row_owned_speculation,
                crate::selection::Phase::Prefill => false,
            }
    }

    fn plan_phase(
        &self,
        inputs: &crate::inputs::PhaseInputs,
    ) -> Result<Vec<crate::pipeline::Program>, BundleStatusV1> {
        (0..self.tp.count())
            .map(|rank| self.plan_rank(inputs, rank))
            .collect()
    }

    /// Lowers one rank's program for a phase. A decode plan depends only on
    /// its bucket, so it is needed only to capture that bucket's graph.
    fn plan_rank(
        &self,
        inputs: &crate::inputs::PhaseInputs,
        rank: usize,
    ) -> Result<crate::pipeline::Program, BundleStatusV1> {
        let Some(program) =
            PhaseProgram::new(self.tp, inputs, self.f32_logits, self.verify_bucket)?
        else {
            return Ok(Default::default());
        };
        let mut lowered = Vec::new();
        for &stage in &program.stages {
            lowered.extend(self.plan_program_stage_lowered(&program, rank, stage)?);
        }
        Ok(crate::program_lower::program(lowered))
    }

    fn plan_program_stage(
        &self,
        program: &PhaseProgram,
        rank: usize,
        stage: crate::program::Stage,
    ) -> Result<Vec<crate::pipeline::PreparedLaunch>, BundleStatusV1> {
        self.plan_program_stage_lowered(program, rank, stage)
            .map(crate::program_lower::launches)
    }

    /// [`Self::plan_program_stage`], with each launch's facts for the
    /// pipelined prefill.
    fn plan_program_stage_lowered(
        &self,
        program: &PhaseProgram,
        rank: usize,
        stage: crate::program::Stage,
    ) -> Result<Vec<crate::program_lower::Lowered>, BundleStatusV1> {
        self.plan_stage_lowered(
            rank,
            program.phase,
            program.planned_tokens,
            program.live_batch,
            program.context,
            &program.logit_rows,
            program.scored_prompt_logits,
            stage,
        )
        .map_err(|error| {
            let _ = self.services.log(
                format!(
                    "MiMo rank {rank} {:?} stage {stage:?}: {error}",
                    program.phase
                )
                .as_bytes(),
            );
            BundleStatusV1::Unsupported
        })
    }

    /// Plans and issues eager prefill one stage at a time: a stage is lowered
    /// for every rank, then queued on every rank. The ranks advance together,
    /// and the device starts on the first stage while the host lowers the rest.
    fn run_prefill_staged(
        &mut self,
        inputs: &crate::inputs::PhaseInputs,
    ) -> Result<(), BundleStatusV1> {
        let Some(program) =
            PhaseProgram::new(self.tp, inputs, self.f32_logits, self.verify_bucket)?
        else {
            return Ok(());
        };
        #[cfg(feature = "diagnostic-taps")]
        let mut tapped = self.begin_taps(program.phase, inputs)?;
        for &stage in &program.stages {
            let lowered = (0..self.tp.count())
                .map(|rank| self.plan_program_stage(&program, rank, stage))
                .collect::<Result<Vec<_>, _>>()?;
            let mut unit = crate::pipeline::Unit::default();
            let mut submitted = Vec::new();
            for (rank, launches) in lowered.iter().enumerate() {
                if launches.is_empty() {
                    continue;
                }
                unit.record(
                    &self.collectives,
                    crate::pipeline::launch_all(
                        &self.services,
                        self.streams[rank],
                        &self.symbols[rank],
                        launches,
                    ),
                )
                .map_err(crate::dispatch::status)?;
                submitted.push(rank);
            }
            if stage == crate::program::Stage::RingBackup {
                for rank in submitted {
                    self.ring_saved[rank].extend(inputs.slot_ids.iter().map(|&slot| slot as u32));
                }
            }
            #[cfg(feature = "diagnostic-taps")]
            if let Some(step) = &mut tapped {
                let whole = (0, inputs.live_tokens());
                self.tap_stage(step, program.phase, stage, whole, &self.streams)?;
            }
        }
        #[cfg(feature = "diagnostic-taps")]
        if let Some(step) = tapped {
            self.pending_taps.push(step);
        }
        Ok(())
    }

    /// Whether every phase runs stage by stage for the taps.
    const fn tapped(&self) -> bool {
        #[cfg(feature = "diagnostic-taps")]
        {
            self.taps.is_some()
        }
        #[cfg(not(feature = "diagnostic-taps"))]
        {
            false
        }
    }

    /// Begins `phase`'s tap step over `inputs`, with this step's encoder rows,
    /// when the instance taps.
    #[cfg(feature = "diagnostic-taps")]
    fn begin_taps(
        &mut self,
        phase: crate::selection::Phase,
        inputs: &crate::inputs::PhaseInputs,
    ) -> Result<Option<runtime_bundle_sdk::taps::TapStep>, BundleStatusV1> {
        let encoders = std::mem::take(&mut self.tap_encoders);
        let folds = self.ops.folds_combine_in(phase);
        let Some(taps) = self.taps.as_mut() else {
            return Ok(None);
        };
        let step = taps.next_step().and_then(|dir| {
            crate::taps::begin(taps, &dir, phase, self.tp, folds, inputs, encoders)
        });
        step.map(Some).map_err(|error| self.tap_failed(error))
    }

    /// Copies `stage`'s selected taps of `part`, phase rows `start..start +
    /// rows` (a pipelined prefill's half, or the whole phase), on every rank
    /// that holds them, after the work queued on that rank's stream of
    /// `streams`. Hidden buffers hold a half's rows from its first row, as
    /// `plan_stage_in` shifts them; the other scratch from row 0.
    #[cfg(feature = "diagnostic-taps")]
    fn tap_stage(
        &self,
        step: &mut runtime_bundle_sdk::taps::TapStep,
        phase: crate::selection::Phase,
        stage: crate::program::Stage,
        (start, rows): (u32, u32),
        streams: &[BundleStreamHandleV1],
    ) -> Result<(), BundleStatusV1> {
        let Some(taps) = &self.taps else {
            return Ok(());
        };
        let shift = |span: DeviceBufferSpanV1| {
            let offset = u64::from(start) * self.tp.geometry.hidden() as u64 * 2;
            let length = span.length.saturating_sub(offset);
            DeviceBufferSpanV1::new(span.buffer, span.offset + offset, length)
        };
        let folds = self.ops.folds_combine_in(phase);
        for tap in crate::taps::taps(stage, self.tp, folds) {
            if !taps.selected(&tap.spec) {
                continue;
            }
            let ranks = tap.spec.sharding.ranks(self.tp.ranks()) as usize;
            for (rank, &stream) in streams.iter().enumerate().take(ranks) {
                let mut scratch = self.scratch[rank].clone();
                scratch.hidden_partial = shift(scratch.hidden_partial);
                scratch.combined = shift(scratch.combined);
                let buffers = crate::taps::RankBuffers {
                    activation: shift(self.activations[rank]),
                    normalized: shift(self.normalized[rank]),
                    scratch: &scratch,
                };
                step.capture_part(
                    &self.services,
                    stream,
                    &tap.spec,
                    rank as u32,
                    (tap.buffer)(&buffers),
                    start,
                    rows,
                )
                .map_err(|error| match error {
                    runtime_bundle_sdk::taps::TapError::Host(error) => status(error),
                    runtime_bundle_sdk::taps::TapError::Io(error) => self.tap_failed(error),
                })?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "diagnostic-taps")]
    fn tap_failed(&self, error: std::io::Error) -> BundleStatusV1 {
        let _ = self.services.log(format!("MiMo taps: {error}").as_bytes());
        BundleStatusV1::InternalError
    }

    /// Moves a step's plain decode rows into its prefill phase when
    /// [`FOLD_DECODE`] allows it; returns the rows moved.
    fn fold_decode(&self, step: &mut crate::inputs::PackedStep, request: &[u8]) -> u32 {
        #[cfg(test)]
        let enabled = self.fold_override.unwrap_or(self.controls.fold_decode);
        #[cfg(not(test))]
        let enabled = self.controls.fold_decode;
        // Drafters tap decode rows in their graphs; media injects index the
        // prefill rows; the BF16 head would change the decode rows' logits.
        if !enabled
            || self.mtp.is_some()
            || self.dflash.is_some()
            || !self.f32_logits
            || BatchCodecV1::decode(request).map_or(true, |batch| !batch.media().is_empty())
        {
            return 0;
        }
        step.fold_decode(
            crate::contract::PREFILL_TOKENS,
            crate::contract::SCORED_ROWS as usize,
        )
    }

    /// Runs a prefill chunk as two pipelined halves: compute on each rank's
    /// stream, all-reduces on its comm stream, ordered by events (see
    /// `overlap`). The ring backup and the vocabulary head run for the whole
    /// chunk, whose inputs the caller uploaded.
    fn run_prefill_overlapped(
        &mut self,
        whole: &crate::inputs::PhaseInputs,
        halves: (crate::inputs::PhaseInputs, crate::inputs::PhaseInputs),
        config: &crate::overlap::Config,
    ) -> Result<(), BundleStatusV1> {
        use crate::overlap::Lane;
        let Some(program) = PhaseProgram::new(self.tp, whole, self.f32_logits, self.verify_bucket)?
        else {
            return Ok(());
        };
        let halves = [halves.0, halves.1];
        let mut half_inputs: [Vec<DeviceInputs>; 2] = Default::default();
        let mut half_programs = Vec::with_capacity(2);
        for (half, inputs) in halves.iter().enumerate() {
            let layout = crate::upload::InputLayout::prefill_within::<Mimo>(
                inputs,
                inputs.logit_rows().len(),
                self.items,
            );
            self.upload_inputs(HALVES[half], layout, inputs)
                .map_err(status)?;
            half_inputs[half] = self.arenas[HALVES[half]]
                .iter()
                .map(|arena| DeviceInputs::within(*arena, &layout))
                .collect();
            half_programs.push(
                PhaseProgram::new(self.tp, inputs, self.f32_logits, self.verify_bucket)?
                    .ok_or(BundleStatusV1::InvalidArgument)?,
            );
        }
        let first_rows = u64::from(halves[0].live_tokens());
        #[cfg(feature = "diagnostic-taps")]
        let tapped = self.begin_taps(program.phase, whole)?;
        struct PrefillOverlap<'a> {
            instance: &'a mut Instance,
            whole: &'a crate::inputs::PhaseInputs,
            #[cfg(feature = "diagnostic-taps")]
            halves: &'a [crate::inputs::PhaseInputs; 2],
            program: &'a PhaseProgram,
            half_programs: &'a [PhaseProgram],
            half_inputs: &'a [Vec<DeviceInputs>; 2],
            first_rows: u64,
            config: &'a crate::overlap::Config,
            #[cfg(feature = "diagnostic-taps")]
            tapped: Option<runtime_bundle_sdk::taps::TapStep>,
        }
        impl bundle_runtime::lanes::Executor<crate::program::Stage> for PrefillOverlap<'_> {
            type Error = BundleStatusV1;

            fn launch(
                &mut self,
                stage: crate::program::Stage,
                half: Option<usize>,
                lane: Lane,
            ) -> Result<(), Self::Error> {
                let instance = &mut self.instance;
                let mut ranks = Vec::with_capacity(instance.tp.count());
                for rank in 0..instance.tp.count() {
                    let lowered = match half {
                        None => instance.plan_program_stage_lowered(self.program, rank, stage)?,
                        Some(half) => {
                            let part = &self.half_programs[half];
                            instance
                                .plan_stage_in_lowered(
                                    rank,
                                    part.phase,
                                    part.planned_tokens,
                                    part.live_batch,
                                    part.context,
                                    &part.logit_rows,
                                    part.scored_prompt_logits,
                                    stage,
                                    self.half_inputs[half][rank],
                                    if half == 0 { 0 } else { self.first_rows },
                                )
                                .map_err(|error| {
                                    let _ = instance.services.log(
                                        format!(
                                            "MiMo rank {rank} half {half} stage {stage:?}: {error}"
                                        )
                                        .as_bytes(),
                                    );
                                    BundleStatusV1::Unsupported
                                })?
                        }
                    };
                    let target = &*instance.target;
                    let launches = lowered
                        .into_iter()
                        .map(|crate::program_lower::Lowered { mut launch, facts }| {
                            #[cfg(test)]
                            crate::sm120::overlap::count(&launch, facts, lane, self.config);
                            target.adjust(&mut launch, facts, lane, self.config);
                            launch
                        })
                        .collect::<Vec<_>>();
                    ranks.push(launches);
                }
                let mut unit = crate::pipeline::Unit::default();
                for (rank, launches) in ranks.iter().enumerate() {
                    if launches.is_empty() {
                        continue;
                    }
                    let stream = match lane {
                        Lane::Compute => instance.streams[rank],
                        Lane::Comm => instance.comm_streams[rank],
                    };
                    unit.record(
                        &instance.collectives,
                        crate::pipeline::launch_all(
                            &instance.services,
                            stream,
                            &instance.symbols[rank],
                            launches,
                        ),
                    )
                    .map_err(crate::dispatch::status)?;
                    if stage == crate::program::Stage::RingBackup {
                        instance.ring_saved[rank]
                            .extend(self.whole.slot_ids.iter().map(|&slot| slot as u32));
                    }
                }
                #[cfg(feature = "diagnostic-taps")]
                if let Some(step) = &mut self.tapped {
                    let part = match half {
                        None => (0, self.whole.live_tokens()),
                        Some(half) => (
                            if half == 0 { 0 } else { self.first_rows as u32 },
                            self.halves[half].live_tokens(),
                        ),
                    };
                    let streams = match lane {
                        Lane::Compute => &instance.streams,
                        Lane::Comm => &instance.comm_streams,
                    };
                    instance.tap_stage(step, self.program.phase, stage, part, streams)?;
                }
                Ok(())
            }

            fn record(
                &mut self,
                lane: Lane,
                mark: crate::overlap::Mark,
            ) -> Result<(), Self::Error> {
                self.event(lane, mark, true)
            }

            fn wait(&mut self, lane: Lane, mark: crate::overlap::Mark) -> Result<(), Self::Error> {
                self.event(lane, mark, false)
            }
        }
        impl PrefillOverlap<'_> {
            fn event(
                &self,
                lane: Lane,
                mark: crate::overlap::Mark,
                record: bool,
            ) -> Result<(), BundleStatusV1> {
                let instance = &self.instance;
                for rank in 0..instance.tp.count() {
                    let stream = match lane {
                        Lane::Compute => instance.streams[rank],
                        Lane::Comm => instance.comm_streams[rank],
                    };
                    let event = instance.overlap_events[rank][mark.index(crate::overlap::PARTS)];
                    if record {
                        instance.services.record_event(event, stream)
                    } else {
                        instance.services.wait_event(stream, event)
                    }
                    .map_err(status)?;
                }
                Ok(())
            }
        }
        let schedule = crate::overlap::schedule_for(self.tp, &program.stages);
        let mut executor = PrefillOverlap {
            instance: self,
            whole,
            #[cfg(feature = "diagnostic-taps")]
            halves: &halves,
            program: &program,
            half_programs: &half_programs,
            half_inputs: &half_inputs,
            first_rows,
            config,
            #[cfg(feature = "diagnostic-taps")]
            tapped,
        };
        bundle_runtime::lanes::execute(schedule, &mut executor)?;
        #[cfg(feature = "diagnostic-taps")]
        let tapped = executor.tapped;
        #[cfg(feature = "diagnostic-taps")]
        if let Some(step) = tapped {
            self.pending_taps.push(step);
        }
        Ok(())
    }

    /// A failed pipelined prefill can leave an all-reduce on a comm stream
    /// that its compute stream never waited for, because the event record
    /// or wait that orders them failed or was never reached. Cancel's restore
    /// copies and a retry reuse the buffers it reads and writes, so every
    /// rank's comm and compute streams are drained on the host first,
    /// relying on no event of the failed schedule. An instance whose streams
    /// cannot be drained is poisoned; a poisoned one may have streams stuck
    /// at a collective and is left alone.
    fn drain_overlap_streams(&self) {
        bundle_runtime::lanes::drain(
            self.collectives.is_poisoned(),
            self.comm_streams.iter().copied(),
            self.streams.iter().copied(),
            |stream| self.services.synchronize_stream(stream),
            |error| {
                let _ = self.services.log(
                    format!(
                        "MiMo pipelined prefill drain failed: {error}; the instance is poisoned"
                    )
                    .as_bytes(),
                );
                self.collectives.poison();
            },
        );
    }

    fn run_phase(
        &mut self,
        inputs: &crate::inputs::PhaseInputs,
        plans: &[crate::pipeline::Program],
        trace: &mut Option<Vec<crate::pipeline::TracedLaunch>>,
    ) -> Result<(), BundleStatusV1> {
        if inputs.live_batch() == 0 {
            return Ok(());
        }
        if !inputs.decode || self.mtp.is_none() {
            self.upload_phase(inputs).map_err(status)?;
        }
        // Prefill runs eagerly, and only prefill splits into overlapped
        // halves. Verify is planned independently of its context, so it is
        // captured as a graph unless eager verify is requested. A tapped
        // instance runs decode and verify stage by stage too.
        if (!inputs.decode || inputs.verify && self.controls.verify_eager || self.tapped())
            && trace.is_none()
        {
            if !inputs.decode
                && let Some(config) = self.overlap
                && inputs.live_tokens() >= config.min_tokens.max(2)
                && let Some(halves) =
                    inputs.split_at(crate::overlap::split_point(inputs.live_tokens()))
            {
                return self
                    .run_prefill_overlapped(inputs, halves, &config)
                    .inspect_err(|_| self.drain_overlap_streams());
            }
            return self.run_prefill_staged(inputs);
        }
        let items = self.items;
        let bucket = |live| {
            items.bucket(live).ok_or_else(|| {
                let _ = self.services.log(
                    format!(
                        "MiMo execute refused a decode phase of {} items and {} rows: \
                         no graph bucket holds {live}",
                        inputs.live_batch(),
                        inputs.live_tokens()
                    )
                    .as_bytes(),
                );
                BundleStatusV1::InvalidArgument
            })
        };
        // A decode graph is keyed by its item bucket; a verify graph by its
        // token and item buckets.
        let bucket = match (inputs.decode, inputs.verify) {
            (true, false) => Some((bucket(inputs.live_batch())?, 0)),
            (true, true) => {
                let (tokens, items) = verify_shape(inputs, self.verify_bucket);
                Some((
                    bundle_runtime::graphs::verify_bucket_within::<Mimo>(tokens, self.items)
                        .ok_or(BundleStatusV1::InvalidArgument)?,
                    bucket(items)?,
                ))
            }
            (false, _) => None,
        };
        if let (Some((bucket, items)), None) = (bucket, trace.as_ref()) {
            // Plan only the ranks whose bucket graph is missing; every rank
            // is captured before any launches (see `pipeline::run_ranks`).
            // Eager instances plan and enqueue every rank each step.
            let graph = (!self.eager).then_some((bucket, items));
            let mut lowered = (0..self.tp.count())
                .map(|rank| {
                    let key = crate::graphs::GraphKey {
                        rank: rank as u32,
                        bucket,
                        items,
                    };
                    if graph.is_some() && self.graphs.contains(key) {
                        Ok(None)
                    } else {
                        self.plan_rank(inputs, rank).map(Some)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let ring_saved = &mut self.ring_saved;
            return crate::pipeline::run_ranks(
                &self.services,
                &self.streams,
                &self.symbols,
                graph,
                &mut self.graphs,
                &self.collectives,
                |rank| {
                    lowered[rank]
                        .take()
                        .ok_or(crate::dispatch::DispatchError::InvalidPlan)
                },
                // The first piece holds the ring backup; once it is enqueued,
                // cancel must restore the phase's slots on this rank.
                |rank| ring_saved[rank].extend(inputs.slot_ids.iter().map(|&slot| slot as u32)),
            )
            .map_err(crate::dispatch::status);
        }
        let Some(trace) = trace.as_mut() else {
            return Err(BundleStatusV1::InternalError);
        };
        // Traced steps launch every planned rank eagerly, one launch at a
        // time with an event after each, collective by collective across
        // the ranks as untraced eager programs go (`pipeline::run_ranks`).
        if plans.len() != self.tp.count() {
            return Err(BundleStatusV1::InternalError);
        }
        let launches = plans
            .iter()
            .map(|plan| plan.launches.as_slice())
            .collect::<Vec<_>>();
        let waits = plans
            .iter()
            .map(|plan| plan.collective.as_slice())
            .collect::<Vec<_>>();
        if let Err(error) = crate::pipeline::check_collectives(&launches, &waits) {
            let _ = self.services.log(error.as_bytes());
            return Err(BundleStatusV1::InvalidArgument);
        }
        // The ring backup, a program's first launch, goes alone: once it is
        // enqueued, cancel must restore the phase's slots on that rank.
        let segments = plans
            .iter()
            .map(|plan| {
                let mut segments =
                    crate::pipeline::collective_segments(&plan.launches, &plan.collective);
                if let Some(first) = segments.first_mut()
                    && first.len() > 1
                {
                    let (backup, rest) = first.split_at(1);
                    *first = rest;
                    segments.insert(0, backup);
                }
                segments
            })
            .collect::<Vec<_>>();
        let mut queued = vec![0; plans.len()];
        let ring_saved = &mut self.ring_saved;
        let _unwinding = crate::pipeline::PoisonOnUnwind(&self.collectives);
        crate::pipeline::launch_segments(
            &self.collectives,
            &mut crate::pipeline::Unit::default(),
            &mut vec![false; plans.len()],
            &segments,
            |rank, launches| {
                // Traced as before: the backup as launch 0, and the rest of
                // the program from launch 0 again.
                let at = queued[rank];
                queued[rank] += launches.len();
                crate::pipeline::launch_all_traced(
                    &self.services,
                    self.contexts[rank],
                    self.streams[rank],
                    rank as u32,
                    &self.symbols[rank],
                    launches,
                    at.saturating_sub(1),
                    trace,
                )
            },
            |rank| ring_saved[rank].extend(inputs.slot_ids.iter().map(|&slot| slot as u32)),
            |_| Ok(()),
        )
        .map_err(crate::dispatch::status)
    }

    fn execute_step(&mut self, request: &[u8], sink: &mut ExecuteResultSinkV1) -> BundleStatusV1 {
        let mut step = match self.preflight(request) {
            Ok(step) => step,
            Err(error) => return error,
        };
        let folded_rows = self.fold_decode(&mut step, request);
        let shape_trace = self.controls.step_trace.then(|| {
            let mut shape = step_shape_trace(&step, self.items, self.verify_bucket);
            // Folded rows ran in the prefill forward: no decode graph.
            shape["folded_decode_rows"] = folded_rows.into();
            (shape, trace_unix_ns())
        });
        let mut trace = (self.controls.trace_launches || traced()).then(Vec::new);
        if step.prefill.live_batch() > 0 {
            self.prefill_layout = crate::upload::InputLayout::prefill_within::<Mimo>(
                &step.prefill,
                step.prefill.logit_rows().len(),
                self.items,
            );
            self.prefill_inputs = self.arenas[PREFILL]
                .iter()
                .map(|arena| DeviceInputs::within(*arena, &self.prefill_layout))
                .collect();
        }
        // Traced steps run eagerly from whole plans. Otherwise prefill is
        // lowered stage by stage as it is issued, and decode graph replay
        // needs no plan.
        let (prefill, decode) = if trace.is_some() {
            match (
                self.plan_phase(&step.prefill),
                self.plan_phase(&step.decode),
            ) {
                (Ok(prefill), Ok(decode)) => (prefill, decode),
                (Err(error), _) | (_, Err(error)) => return error,
            }
        } else {
            (Vec::new(), Vec::new())
        };
        let batch = match BatchCodecV1::decode(request) {
            Ok(batch) => batch,
            Err(_) => {
                return self.refuse(
                    BundleStatusV1::InvalidArgument,
                    "undecodable batch request",
                    None,
                );
            }
        };
        let media_plan = match crate::encoders::injection::plan_for(
            self.tp.geometry,
            batch.media(),
            &step.prefill,
            batch.items().len(),
        ) {
            Ok(plan) => plan,
            Err(reason) => {
                return self.refuse(BundleStatusV1::InvalidArgument, reason, Some(&batch));
            }
        };
        let items = crate::ring::items(&batch);
        // Encoder failures precede the ring transaction: no backup exists yet.
        if let Some(encoder) = &self.encoder {
            let started = std::time::Instant::now();
            let injects = match encoder.run(
                &self.services,
                &self.streams,
                &self.symbols,
                request,
                &step.prefill,
                &self.collectives,
            ) {
                Ok(injects) => injects,
                Err(status) => return status,
            };
            // The prefill's media override reads every row's source; without
            // injects they all keep their token rows.
            if !injects {
                let sources: Vec<u8> = media_plan
                    .source_rows
                    .iter()
                    .flat_map(|row| row.to_le_bytes())
                    .collect();
                for (rank, &stream) in encoder.ranks.iter().zip(&self.streams) {
                    let uploaded = rank.region("source-rows").and_then(|span| {
                        upload_bytes(&self.services, stream, span, &sources)
                            .map_err(|e| e.to_string())
                    });
                    if let Err(error) = uploaded {
                        // An upload may have queued work: drain, then poison.
                        for &stream in &self.streams {
                            let _ = self.services.synchronize_stream(stream);
                        }
                        self.collectives.poison();
                        let _ = self
                            .services
                            .log(format!("MiMo encoder execution: {error}").as_bytes());
                        return BundleStatusV1::InternalError;
                    }
                }
            }
            let encodes = &batch.media().encodes;
            if !encodes.is_empty() {
                let _ = self.services.log(
                    format!(
                        "MiMo encoders mode={} records={} encoder={} elapsed_ms={:.3} capture={}",
                        match encoder.mode {
                            bundle_runtime::encoder::Mode::Serial => "serial",
                            bundle_runtime::encoder::Mode::Relocated => "parallel",
                        },
                        encodes.len(),
                        if encodes.iter().all(|r| r.encoder == 0) {
                            "vision"
                        } else if encodes.iter().all(|r| r.encoder == 1) {
                            "audio"
                        } else {
                            "mixed"
                        },
                        started.elapsed().as_secs_f64() * 1000.0,
                        encoder.capture.is_some() || encoder.arena_capture.is_some(),
                    )
                    .as_bytes(),
                );
            }
        }
        // Every step of a tapped instance carries each prompt's media
        // occurrences, so a later chunk's rows keep their request and items.
        #[cfg(feature = "diagnostic-taps")]
        if let (Some(taps), Some(encoder)) = (&self.taps, &self.encoder) {
            use bundle_runtime::media::OccurrenceItem;
            let arena = match crate::encoders::arena::for_geometry(self.tp.geometry) {
                Ok(arena) => arena,
                Err(reason) => {
                    return self.refuse(BundleStatusV1::InvalidArgument, reason, Some(&batch));
                }
            };
            let read = self
                .tap_occurrences
                .step(
                    &arena,
                    batch.media(),
                    &OccurrenceItem::of(&batch),
                    &crate::taps::kinds_for(self.tp),
                )
                .map_err(|error| error.to_string())
                .and_then(|step| {
                    let blocks = encoder.take_block_rows();
                    let blocks = crate::taps::block_rows_for(
                        self.tp,
                        taps,
                        &mut self.tap_occurrences,
                        &step,
                        blocks,
                    )?;
                    if step.reads.is_empty() {
                        return Ok(blocks);
                    }
                    let arena = encoder.ranks[0].region("arena")?;
                    let mut rows = crate::taps::encoder_rows_for(
                        self.tp,
                        &self.services,
                        self.streams[0],
                        arena,
                        &step,
                    )?;
                    rows.extend(blocks);
                    Ok(rows)
                });
            match read {
                Ok(rows) => self.tap_encoders = rows,
                Err(error) => {
                    let _ = self
                        .services
                        .log(format!("MiMo encoder taps: {error}").as_bytes());
                    return BundleStatusV1::InternalError;
                }
            }
        }
        let result = self.prepare_ring(&items);
        if result != BundleStatusV1::Success {
            let reason = match result {
                BundleStatusV1::InvalidArgument => {
                    "sliding-window ring append does not continue its request"
                }
                BundleStatusV1::NotFound => "sliding-window ring has no slot for a request",
                BundleStatusV1::ResourceExhausted => "sliding-window ring has no free slot",
                _ => "sliding-window ring transaction is still open",
            };
            return self.refuse(result, reason, Some(&batch));
        }
        let executed = (|| {
            self.mtp_begin(&step, &batch).map_err(|error| {
                let _ = self
                    .services
                    .log(format!("MTP execute: {error}").as_bytes());
                drafter_status(error)
            })?;
            self.dflash_begin(&step, &batch).map_err(|error| {
                let _ = self
                    .services
                    .log(format!("DFlash execute: {error}").as_bytes());
                drafter_status(error)
            })?;
            self.run_phase(&step.prefill, &prefill, &mut trace)?;
            self.mtp_tap(&step.prefill).map_err(status)?;
            self.run_phase(&step.decode, &decode, &mut trace)?;
            self.mtp_tap(&step.decode).map_err(status)?;
            if let Some(trace) = trace.as_mut() {
                crate::pipeline::synchronize_trace(&self.services, trace).map_err(status)?;
            }
            // Rank 0 gathers the logits. The other ranks finish in their own
            // stream order, and their collectives keep them within a step.
            // With owned decode rows every rank holds logits the host reads.
            let synchronized = if self.decode_rows_owned(&step.decode) {
                self.tp.count()
            } else {
                1
            };
            for rank in 0..synchronized {
                log(&self.services)
                    .named(
                        "execute synchronize_stream",
                        self.services.synchronize_stream(self.streams[rank]),
                    )
                    .map_err(status)?;
            }
            self.mtp_emit(&mut step, &batch, sink).map_err(status)?;
            self.dflash_emit(&mut step, &batch, sink).map_err(status)?;
            for view in self.ordered_logit_views(&step).map_err(status)? {
                sink.emit_device_view(view).map_err(status)?;
            }
            Ok(())
        })();
        let trace_cleanup = trace.as_mut().map_or(Ok(()), |trace| {
            crate::pipeline::release_trace(&self.services, trace)
        });
        if executed.is_err() || trace_cleanup.is_err() {
            // Restore copies would queue behind a collective that never
            // completes; a poisoned instance leaves its streams alone.
            if self.collectives.is_poisoned() {
                let _ = self.services.log(
                    b"MiMo execute left collectives unmatched across ranks; the instance is poisoned",
                );
                return executed.err().unwrap_or_else(|| {
                    status(trace_cleanup.expect_err("execution or trace cleanup failed"))
                });
            }
            let restore = transaction::cancel(self);
            if restore != BundleStatusV1::Success {
                // Execute's refusal leaves the ABI handle Ready, so the host
                // cannot cancel this transaction. Retain its resources instead.
                self.collectives.poison();
                let _ = self.services.log(
                    format!(
                        "MiMo execute: {executed:?}; trace cleanup: {trace_cleanup:?}; rollback failed: {restore:?}; the instance is poisoned"
                    )
                    .as_bytes(),
                );
                return restore;
            }
            return trace_cleanup
                .err()
                .map(status)
                .or_else(|| executed.err())
                .expect("execution or trace cleanup failed");
        }
        if let Some((mut shape, started)) = shape_trace {
            shape["execute_start_unix_ns"] = started.into();
            shape["execute_end_unix_ns"] = trace_unix_ns().into();
            let _ = self.services.log(shape.to_string().as_bytes());
        }
        BundleStatusV1::Success
    }

    #[allow(
        dead_code,
        reason = "used when the rank program is connected to execute"
    )]
    #[allow(clippy::too_many_arguments)]
    fn plan_stage(
        &self,
        rank: usize,
        phase: crate::selection::Phase,
        live_tokens: u32,
        live_batch: u32,
        context: u32,
        logit_rows: &[u32],
        scored_prompt_logits: bool,
        stage: crate::program::Stage,
    ) -> Result<Vec<crate::pipeline::PreparedLaunch>, kernel_core::contract::Refusal> {
        self.plan_stage_lowered(
            rank,
            phase,
            live_tokens,
            live_batch,
            context,
            logit_rows,
            scored_prompt_logits,
            stage,
        )
        .map(crate::program_lower::launches)
    }

    /// [`Self::plan_stage`], with each launch's facts for the pipelined
    /// prefill.
    #[allow(clippy::too_many_arguments)]
    fn plan_stage_lowered(
        &self,
        rank: usize,
        phase: crate::selection::Phase,
        live_tokens: u32,
        live_batch: u32,
        context: u32,
        logit_rows: &[u32],
        scored_prompt_logits: bool,
        stage: crate::program::Stage,
    ) -> Result<Vec<crate::program_lower::Lowered>, kernel_core::contract::Refusal> {
        let inputs = *match phase {
            crate::selection::Phase::Decode(_) => &self.decode_inputs,
            crate::selection::Phase::Verify(_) => &self.verify_inputs,
            crate::selection::Phase::Prefill => &self.prefill_inputs,
        }
        .get(rank)
        .ok_or(kernel_core::contract::Refusal(
            "MiMo rank input buffers are missing",
        ))?;
        self.plan_stage_in_lowered(
            rank,
            phase,
            live_tokens,
            live_batch,
            context,
            logit_rows,
            scored_prompt_logits,
            stage,
            inputs,
            0,
        )
    }

    /// Plans one stage on `inputs` for token rows starting at `row0` of the
    /// token-indexed buffers: the residual, the normalized hidden state (on
    /// every rank, which the all-reduce writes), the attention output and the
    /// MoE output. A pipelined prefill half uses its own rows of each, so the
    /// other half's work never touches rows its all-reduce reads or writes.
    /// Each launch comes with its facts for the pipelined prefill.
    #[allow(clippy::too_many_arguments)]
    fn plan_stage_in_lowered(
        &self,
        rank: usize,
        phase: crate::selection::Phase,
        live_tokens: u32,
        live_batch: u32,
        context: u32,
        logit_rows: &[u32],
        scored_prompt_logits: bool,
        stage: crate::program::Stage,
        inputs: DeviceInputs,
        row0: u64,
    ) -> Result<Vec<crate::program_lower::Lowered>, kernel_core::contract::Refusal> {
        let lowered = self.plan_stage_in_mode(
            rank,
            phase,
            live_tokens,
            live_batch,
            context,
            logit_rows,
            scored_prompt_logits,
            stage,
            inputs,
            row0,
            None,
        );
        #[cfg(test)]
        if let Ok(launches) = &lowered {
            PLANNED.with(|planned| {
                planned.borrow_mut().extend(
                    launches
                        .iter()
                        .map(|lowered| (rank, lowered.launch.symbol.clone(), lowered.launch.rows)),
                )
            });
        }
        lowered
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_stage_in_mode(
        &self,
        rank: usize,
        phase: crate::selection::Phase,
        live_tokens: u32,
        live_batch: u32,
        context: u32,
        logit_rows: &[u32],
        scored_prompt_logits: bool,
        stage: crate::program::Stage,
        inputs: DeviceInputs,
        row0: u64,
        context_buffers: Option<(&crate::dflash::program::RankBuffers, DeviceBufferSpanV1)>,
    ) -> Result<Vec<crate::program_lower::Lowered>, kernel_core::contract::Refusal> {
        if stage == crate::program::Stage::MediaInject {
            let Some(encoder) = self
                .encoder
                .as_ref()
                .filter(|_| matches!(phase, crate::selection::Phase::Prefill))
            else {
                return Ok(Vec::new());
            };
            // The part's rows of the activation, and its first row's offset
            // in the step's arena source rows.
            let activation = self.activations[rank];
            let activation_offset = row0 * self.tp.geometry.hidden() as u64 * 2;
            if activation_offset > activation.length {
                return Err(kernel_core::contract::Refusal(
                    "media activation row offset",
                ));
            }
            let first_row = u32::try_from(row0)
                .map_err(|_| kernel_core::contract::Refusal("media source row offset"))?;
            return Ok(vec![
                encoder.ranks[rank]
                    .inject_part(
                        DeviceBufferSpanV1::new(
                            activation.buffer,
                            activation.offset + activation_offset,
                            activation.length - activation_offset,
                        ),
                        inputs.live_tokens,
                        first_row,
                        live_tokens,
                    )?
                    .into(),
            ]);
        }
        // Hidden buffers are allocated with slack rows past their span, so the
        // shifted span keeps the capacity every planner checks.
        let rows = |span: DeviceBufferSpanV1| {
            if row0 > crate::scratch::HIDDEN_SLACK_ROWS {
                return Err(kernel_core::contract::Refusal(
                    "MiMo row offset exceeds the hidden buffer slack",
                ));
            }
            Ok(DeviceBufferSpanV1::new(
                span.buffer,
                span.offset + row0 * self.tp.geometry.hidden() as u64 * 2,
                span.length,
            ))
        };
        let mut scratch = self
            .scratch
            .get(rank)
            .ok_or(kernel_core::contract::Refusal(
                "MiMo rank scratch is missing",
            ))?
            .clone();
        scratch.hidden_partial = rows(scratch.hidden_partial)?;
        scratch.combined = rows(scratch.combined)?;
        let scratch = &scratch;
        let capacity = phase.capacity()?;
        let comm_index = crate::scratch::COMM_BUCKETS
            .iter()
            .position(|&bucket| bucket == capacity)
            .ok_or(kernel_core::contract::Refusal(
                "MiMo communication bucket is missing",
            ))?;
        let gather_outputs = self
            .logits
            .iter()
            .map(|logits| match (phase, scored_prompt_logits) {
                (crate::selection::Phase::Decode(_) | crate::selection::Phase::Verify(_), _) => {
                    logits.gathered_decode
                }
                (crate::selection::Phase::Prefill, true) => logits.gathered_scored,
                (crate::selection::Phase::Prefill, false) => logits.gathered_prefill,
            })
            .collect::<Vec<_>>();
        let exchange = context_buffers.is_none() && self.exchanges_logits(phase);
        let (gather_flags, gather_epochs): (Vec<_>, Vec<_>) = self
            .scratch
            .iter()
            .map(|peer| {
                peer.comm_state
                    .get(comm_index)
                    .and_then(|state| {
                        if exchange {
                            state
                                .exchange_rows
                                .map(|exchange| (exchange.flags, exchange.epoch))
                        } else {
                            Some((state.gather_rows_flags, state.gather_rows_epoch))
                        }
                    })
                    .ok_or(kernel_core::contract::Refusal(
                        "MiMo peer logits flags are missing",
                    ))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        let hidden_inboxes = self
            .scratch
            .iter()
            .map(|peer| crate::scratch::comm_inbox(peer.comm_inbox, comm_index))
            .collect::<Vec<_>>();
        let hidden_flags = self
            .scratch
            .iter()
            .map(|peer| peer.comm_state[comm_index].hidden_flags)
            .collect::<Vec<_>>();
        let hidden_gathered = self
            .normalized
            .iter()
            .map(|span| rows(*span))
            .collect::<Result<Vec<_>, _>>()?;
        if [
            gather_outputs.len(),
            gather_flags.len(),
            hidden_inboxes.len(),
            hidden_flags.len(),
            hidden_gathered.len(),
        ]
        .iter()
        .any(|&peers| peers != self.tp.count())
        {
            return Err(kernel_core::contract::Refusal(
                "MiMo peer buffers do not cover the ranks",
            ));
        }
        let bound = crate::program_lower::RankStage {
            services: Some(&self.services),
            tp: self.tp,
            rank: rank as u32,
            phase,
            live_tokens,
            live_batch,
            context,
            ga_kv: self.kv_dtype,
            ga_kv_scales: self.kv_scales,
            switches: self.switches,
            ops: &self.ops,
            bucket: comm_index,
            views: self.views.get(rank).ok_or(kernel_core::contract::Refusal(
                "MiMo rank views are missing",
            ))?,
            projections: self.precision.as_ref().and_then(|p| p.rank(rank)),
            inputs,
            scratch,
            activation: rows(self.activations[rank])?,
            normalized: rows(self.normalized[rank])?,
            workspace: self.workspaces[rank],
            local_logits: if scored_prompt_logits {
                self.logits[rank].local_scored
            } else {
                self.logits[rank].local
            },
            scored_prompt_logits,
            logit_rows,
            logits_gather: crate::ops::LogitsBuffers {
                shard: crate::topology::vocab_per_rank(self.tp),
                local: if scored_prompt_logits {
                    self.logits[rank].local_scored
                } else {
                    self.logits[rank].local
                },
                outputs: gather_outputs,
                flags: gather_flags,
                epochs: gather_epochs,
            },
            exchange_logits: exchange,
            hidden_inboxes,
            hidden_flags,
            hidden_epochs: self
                .scratch
                .iter()
                .map(|peer| peer.comm_state[comm_index].hidden_epoch)
                .collect(),
            hidden_gathered,
            ring_state: self.state_views[rank],
            ring: self.ring,
            items: self.items,
            kv_pool: self.kv_views[rank],
        };
        if let Some((buffers, history)) = context_buffers {
            return crate::dflash::program::context(&bound, buffers, Some(history));
        }
        let mut launches = bound.lower_with_facts(stage)?;
        if let Some(dflash) = &self.dflash {
            dflash.extend(rank, &bound, stage, row0, &mut launches)?;
        }
        if let Some(mtp) = &self.mtp {
            mtp.extend(rank, &bound, stage, &mut launches)?;
        }
        Ok(launches)
    }

    fn preflight(&self, bytes: &[u8]) -> Result<crate::inputs::PackedStep, BundleStatusV1> {
        if self.kv_views.len() != self.tp.count() || self.ledger.step().is_some() {
            return Err(BundleStatusV1::InvalidState);
        }
        let request = BatchCodecV1::decode(bytes).map_err(|_| {
            self.refuse(
                BundleStatusV1::InvalidArgument,
                "undecodable batch request",
                None,
            )
        })?;
        // MiMo takes implied 1D positions and declares no position capability.
        if request.has_position_axes() {
            return Err(self.refuse(
                BundleStatusV1::InvalidArgument,
                "explicit position axes",
                Some(&request),
            ));
        }
        if !request.media().is_empty()
            && (!crate::encoders::SERVING_READY || self.encoder.is_none())
        {
            return Err(self.refuse(
                BundleStatusV1::Unsupported,
                "media without loaded encoders",
                Some(&request),
            ));
        }
        let step = crate::inputs::pack_for_ring_within(
            self.tp.model_limits(),
            &request,
            &self.ledger,
            self.kv_page_capacity,
            self.verify_limits,
            self.items,
        )
        .map_err(|error| self.refuse(error.status(), &error.to_string(), Some(&request)))?;
        if [&step.prefill, &step.decode].iter().any(|phase| {
            !crate::rope_tables::positions_fit_for(
                self.tp.geometry,
                self.context_tokens,
                &phase.positions,
            )
        }) {
            return Err(self.refuse(
                BundleStatusV1::InvalidArgument,
                "RoPE position exceeds served context",
                Some(&request),
            ));
        }
        if step.prefill.logit_rows().len() > crate::contract::SCORED_ROWS as usize {
            return Err(self.refuse(
                BundleStatusV1::Unsupported,
                "prefill logits rows exceed the scored row cap",
                Some(&request),
            ));
        }
        Ok(step)
    }

    /// Logs why execute refused a batch and returns `status`. The ABI carries
    /// only the status, so the host log names the check and the batch shape.
    fn refuse(
        &self,
        status: BundleStatusV1,
        reason: &str,
        request: Option<&runtime_bundle_sdk::BatchRequestV1>,
    ) -> BundleStatusV1 {
        let shape = request.map_or_else(
            || "batch unreadable".to_owned(),
            |request| {
                bundle_runtime::inputs::BatchShape::of(request)
                    .describe_within::<Mimo>(self.verify_limits, self.items)
            },
        );
        let _ = self.services.log(
            format!(
                "MiMo execute refused a batch with status {}: {reason}; {shape}",
                status.0
            )
            .as_bytes(),
        );
        status
    }

    fn ordered_logit_views(
        &self,
        step: &crate::inputs::PackedStep,
    ) -> Result<Vec<runtime_bundle_abi::BundleLogitsViewV1>, BundleAbiError> {
        let context = *self
            .contexts
            .first()
            .ok_or(BundleAbiError::InvalidRange("missing rank-zero context"))?;
        let logits = self
            .logits
            .first()
            .ok_or(BundleAbiError::InvalidRange("missing rank-zero logits"))?;
        let owners = self
            .contexts
            .iter()
            .zip(&self.logits)
            .map(|(&context, logits)| (context, logits.gathered_decode))
            .collect::<Vec<_>>();
        // The exchange splits the phase bucket's rows, as the lowering does.
        let block = self
            .decode_phase(&step.decode)
            .capacity()
            .unwrap_or(0)
            .div_ceil(self.tp.ranks());
        crate::logits::ordered_views_owned(
            self.tp.model_limits(),
            context,
            crate::logits::PhaseOutput {
                inputs: &step.prefill,
                buffer: if self.f32_logits || step.prefill.prompt_rows.iter().any(|&row| row) {
                    logits.gathered_scored
                } else {
                    logits.gathered_prefill
                },
                element_type: if self.f32_logits || step.prefill.prompt_rows.iter().any(|&row| row)
                {
                    runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32
                } else {
                    runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_BF16
                },
                compact: true,
            },
            crate::logits::PhaseOutput {
                inputs: &step.decode,
                buffer: logits.gathered_decode,
                element_type: if self.f32_logits {
                    runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32
                } else {
                    runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_BF16
                },
                compact: false,
            },
            self.decode_rows_owned(&step.decode)
                .then_some(crate::logits::RowOwners {
                    block,
                    ranks: &owners,
                }),
        )
    }

    /// The decode or verify phase the program plans for these inputs.
    fn decode_phase(&self, decode: &crate::inputs::PhaseInputs) -> crate::selection::Phase {
        if decode.verify {
            crate::selection::Phase::Verify(verify_shape(decode, self.verify_bucket).0)
        } else {
            crate::selection::Phase::Decode(decode.live_batch())
        }
    }

    /// Whether this step's decode or verify rows were exchanged to their owners.
    fn decode_rows_owned(&self, decode: &crate::inputs::PhaseInputs) -> bool {
        decode.live_batch() > 0 && self.exchanges_logits(self.decode_phase(decode))
    }

    /// Uploads one phase's inputs: one packed copy per rank, plus the
    /// block-table entries that changed since that phase's last step.
    fn upload_phase(&mut self, phase: &crate::inputs::PhaseInputs) -> Result<(), BundleAbiError> {
        if phase.live_batch() == 0 {
            return Ok(());
        }
        let (index, layout) = match (phase.decode, phase.verify) {
            (true, false) => (
                DECODE,
                crate::upload::InputLayout::decode_within(self.items),
            ),
            (true, true) => (
                VERIFY,
                crate::upload::InputLayout::verify_within::<Mimo>(self.items),
            ),
            (false, _) => (PREFILL, self.prefill_layout),
        };
        self.upload_inputs(index, layout, phase)
    }

    /// Uploads `phase` into input set `index` laid out as `layout`.
    fn upload_inputs(
        &mut self,
        index: usize,
        layout: crate::upload::InputLayout,
        phase: &crate::inputs::PhaseInputs,
    ) -> Result<(), BundleAbiError> {
        let rows = if phase.decode {
            Vec::new()
        } else {
            phase.logit_rows()
        };
        let bytes = layout
            .pack(phase, &rows)
            .ok_or(BundleAbiError::InvalidRange(
                "MiMo inputs exceed their arena",
            ))?;
        let writes = self.block_tables[index].update(&phase.block_tables);
        let uploaded = (|| {
            for (stream, arena) in self.streams.iter().zip(&self.arenas[index]) {
                upload_bytes(&self.services, *stream, arena.arena, &bytes)?;
                self.block_tables[index].upload(
                    &self.services,
                    *stream,
                    arena.block_tables,
                    &writes,
                )?;
            }
            Ok(())
        })();
        if uploaded.is_err() {
            self.block_tables[index].forget();
        }
        uploaded
    }

    #[cfg(test)]
    fn run_prepared(
        &mut self,
        rank: usize,
        graph: Option<(u32, u32)>,
        launches: &[crate::pipeline::PreparedLaunch],
    ) -> Result<(), crate::dispatch::DispatchError> {
        let stream = *self
            .streams
            .get(rank)
            .ok_or(crate::dispatch::DispatchError::InvalidPlan)?;
        let symbols = self
            .symbols
            .get(rank)
            .ok_or(crate::dispatch::DispatchError::InvalidPlan)?;
        crate::pipeline::run_rank(
            &self.services,
            stream,
            rank as u32,
            graph,
            symbols,
            &mut self.graphs,
            launches,
        )
    }
}

#[cfg(test)]
pub(crate) fn prefill_ga_split_probe(
    handle: BundleHandleV1,
    context: u32,
) -> Result<u32, BundleStatusV1> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).ok_or(BundleStatusV1::StaleHandle)?;
        let launches = instance
            .plan_stage(
                0,
                crate::selection::Phase::Prefill,
                1,
                1,
                context,
                &[0],
                true,
                crate::program::Stage::AttentionPartial(5),
            )
            .map_err(|_| BundleStatusV1::Unsupported)?;
        let partial = launches.first().ok_or(BundleStatusV1::InvalidState)?;
        let Some(kernel_core::contract::Argument::U32(splits)) =
            partial.arguments.0.iter().rev().nth(8)
        else {
            return Err(BundleStatusV1::InvalidState);
        };
        Ok(*splits)
    })
}

#[test]
fn step_shape_trace_distinguishes_live_rows_capacity_and_per_item_k() {
    let decode = crate::inputs::PhaseInputs {
        decode: true,
        verify: true,
        item_order: vec![0, 1, 2],
        request_ids: vec![10, 11, 12],
        token_ids: vec![0; 12],
        q_len: vec![8, 3, 1],
        prefix_len: vec![40, 9, 17],
        verify_rows: vec![true, true, false],
        ..Default::default()
    };
    let prefill = crate::inputs::PhaseInputs {
        item_order: vec![0],
        request_ids: vec![13],
        token_ids: vec![0; 5],
        q_len: vec![5],
        prefix_len: vec![512],
        ..Default::default()
    };
    let mut step = crate::inputs::PackedStep { prefill, decode };
    let items = Items::new(32).unwrap();
    let shape = step_shape_trace(&step, items, 0);
    assert_eq!(shape["live_target_rows"], 12);
    assert_eq!(shape["graph_capacity"], 16);
    assert_eq!(shape["draft_tokens"], serde_json::json!([7, 2, 0]));
    assert_eq!(shape["prefix_len"], serde_json::json!([40, 9, 17]));
    assert_eq!(shape["prefill_request_ids"], serde_json::json!([13]));
    assert_eq!(shape["prefill_query_rows"], serde_json::json!([5]));
    assert_eq!(shape["prefill_prefix_len"], serde_json::json!([512]));
    assert_eq!(step_shape_trace(&step, items, 128)["graph_capacity"], 128);
    step.decode.verify = false;
    step.decode.verify_rows.fill(false);
    step.decode.token_ids.truncate(3);
    step.decode.q_len.fill(1);
    let shape = step_shape_trace(&step, items, 0);
    assert_eq!(shape["graph_capacity"], 4);
    assert_eq!(shape["draft_tokens"], serde_json::json!([0, 0, 0]));
}

#[test]
fn verify_buckets_are_forced_only_within_the_enabled_rows() {
    assert_eq!(parse_verify_bucket(Some("256"), 256), 256);
    assert_eq!(parse_verify_bucket(Some("64"), 40), 64);
    assert_eq!(parse_verify_bucket(Some("128"), 32), 0);
    assert_eq!(parse_verify_bucket(Some("96"), 256), 0);
    assert_eq!(parse_verify_bucket(Some("x"), 256), 0);
    assert_eq!(parse_verify_bucket(None, 256), 0);
}

/// Forces the smallest verify token bucket, as `VERIFY_BUCKET` does.
#[cfg(test)]
pub(crate) fn force_verify_bucket(handle: BundleHandleV1, bucket: u32) {
    INSTANCES.with(|table| {
        if let Some(instance) = table.borrow_mut().get_mut(handle) {
            instance.verify_bucket = bucket;
        }
    });
}

/// The decode-phase rows this instance lets one verify step carry.
#[cfg(test)]
pub(crate) fn verify_rows_per_step(handle: BundleHandleV1) -> Option<u32> {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .map(|instance| instance.verify_limits.rows_per_step)
    })
}

/// The tokens a probe plans a `phase` stage for: a decode step's token
/// bucket, as the decode program plans it, and otherwise one.
#[cfg(test)]
fn probe_tokens(phase: crate::selection::Phase) -> Result<u32, BundleStatusV1> {
    match phase {
        crate::selection::Phase::Decode(_) => phase
            .capacity()
            .map_err(|_| BundleStatusV1::InvalidArgument),
        crate::selection::Phase::Verify(_) | crate::selection::Phase::Prefill => Ok(1),
    }
}

#[cfg(test)]
pub(crate) fn plan_stage_probe(
    handle: BundleHandleV1,
    phase: crate::selection::Phase,
    stage: crate::program::Stage,
) -> Result<Vec<String>, BundleStatusV1> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).ok_or(BundleStatusV1::StaleHandle)?;
        let launches = instance
            .plan_stage(
                0,
                phase,
                probe_tokens(phase)?,
                1,
                1,
                &[0],
                phase == crate::selection::Phase::Prefill,
                stage,
            )
            .map_err(|_| BundleStatusV1::Unsupported)?;
        for launch in &launches {
            crate::dispatch::arguments(&launch.arguments, &launch.bindings)
                .map_err(|_| BundleStatusV1::InvalidState)?;
            if !launch.plan.pdl || !instance.symbols[0].contains_key(&launch.symbol) {
                return Err(BundleStatusV1::InvalidState);
            }
        }
        Ok(launches.into_iter().map(|launch| launch.symbol).collect())
    })
}

#[cfg(test)]
pub(crate) fn grouped_gate_input_is_normalized(
    handle: BundleHandleV1,
    phase: crate::selection::Phase,
) -> Result<bool, BundleStatusV1> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).ok_or(BundleStatusV1::StaleHandle)?;
        let launches = instance
            .plan_stage(
                0,
                phase,
                1,
                1,
                1,
                &[0],
                false,
                crate::program::Stage::GroupedGateUp(1),
            )
            .map_err(|_| BundleStatusV1::Unsupported)?;
        let compute = launches.first().ok_or(BundleStatusV1::InvalidState)?;
        let Some(kernel_core::contract::Argument::DeviceBuffer(handle)) =
            compute.arguments.0.first()
        else {
            return Err(BundleStatusV1::InvalidState);
        };
        let input = compute
            .bindings
            .get(*handle)
            .map_err(|_| BundleStatusV1::InvalidState)?;
        Ok(input == instance.normalized[0])
    })
}

/// Whether a stage's first launch reads rank 0's packed QKV rows first.
#[cfg(test)]
pub(crate) fn stage_reads_qkv(
    handle: BundleHandleV1,
    phase: crate::selection::Phase,
    stage: crate::program::Stage,
) -> Result<bool, BundleStatusV1> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).ok_or(BundleStatusV1::StaleHandle)?;
        let launches = instance
            .plan_stage(0, phase, probe_tokens(phase)?, 1, 1, &[0], false, stage)
            .map_err(|_| BundleStatusV1::Unsupported)?;
        let launch = launches.first().ok_or(BundleStatusV1::InvalidState)?;
        let Some(kernel_core::contract::Argument::DeviceBuffer(handle)) =
            launch.arguments.0.first()
        else {
            return Err(BundleStatusV1::InvalidState);
        };
        let input = launch
            .bindings
            .get(*handle)
            .map_err(|_| BundleStatusV1::InvalidState)?;
        Ok(input == instance.scratch[0].qkv)
    })
}

/// The ring's ledger in an instance serving `items` with the declared
/// `state`: a slot per sequence, the first free one for a new request.
fn ring_ledger(
    context_tokens: u64,
    items: Items,
    state: &bundle_runtime::state::RequestState,
) -> Ledger {
    Ledger::new(
        items.get(),
        Placement::Ledger,
        Rules::of(state),
        context_tokens,
    )
}

fn drafter_status(error: BundleAbiError) -> BundleStatusV1 {
    match error {
        BundleAbiError::InvalidRange(_) => BundleStatusV1::InvalidArgument,
        error => status(error),
    }
}

fn ledger_status(error: LedgerError) -> BundleStatusV1 {
    match error {
        LedgerError::StalePlan => BundleStatusV1::InternalError,
        LedgerError::NoSlot => BundleStatusV1::ResourceExhausted,
        LedgerError::UnknownRequest => BundleStatusV1::NotFound,
        LedgerError::Open | LedgerError::NotOpen => BundleStatusV1::InvalidState,
        LedgerError::ForeignSlot
        | LedgerError::Tokens
        | LedgerError::InvalidStep
        | LedgerError::Kept => BundleStatusV1::InvalidArgument,
    }
}

impl Instance {
    /// Copies the saved shadow cells back to working on each rank, for the
    /// slots whose backup that rank enqueued in this transaction.
    fn restore_ring(&self, entries: &[Cell]) -> Result<(), BundleAbiError> {
        for ((stream, state), saved) in self
            .streams
            .iter()
            .zip(&self.state_views)
            .zip(&self.ring_saved)
        {
            for entry in entries.iter().filter(|entry| saved.contains(&entry.slot)) {
                let (source_offset, target_offset) = (entry.shadow_offset, entry.working_offset);
                let source = state
                    .offset
                    .checked_add(source_offset)
                    .ok_or(BundleAbiError::InvalidRange("ring source offset"))?;
                let target = state
                    .offset
                    .checked_add(target_offset)
                    .ok_or(BundleAbiError::InvalidRange("ring target offset"))?;
                if source_offset
                    .checked_add(entry.length)
                    .is_none_or(|end| end > state.length)
                    || target_offset
                        .checked_add(entry.length)
                        .is_none_or(|end| end > state.length)
                {
                    return Err(BundleAbiError::InvalidRange("ring copy range"));
                }
                self.services.copy_device_to_device(
                    *stream,
                    DeviceBufferSpanV1::new(state.buffer, target, entry.length),
                    DeviceBufferSpanV1::new(state.buffer, source, entry.length),
                )?;
            }
        }
        Ok(())
    }

    #[allow(
        dead_code,
        reason = "called by execute once the communication engine lands"
    )]
    /// Opens the ring transaction. Each phase's first launch saves the
    /// cells it touches on the device; cancel copies them back.
    fn prepare_ring(&mut self, items: &[Item]) -> BundleStatusV1 {
        for saved in &mut self.ring_saved {
            saved.clear();
        }
        self.ledger
            .begin(items, Vec::new())
            .map_or_else(ledger_status, |_| BundleStatusV1::Success)
    }

    /// Publishes committed captures and removes cancelled captures.
    #[cfg(feature = "diagnostic-taps")]
    fn finish_taps(&mut self, commit: bool) -> BundleStatusV1 {
        self.tap_occurrences.finish(commit);
        self.tap_encoders.clear();
        for capture in std::mem::take(&mut self.pending_taps) {
            let result = if commit {
                capture.finish()
            } else {
                capture.cancel()
            };
            if result.is_err() {
                return BundleStatusV1::InternalError;
            }
        }
        BundleStatusV1::Success
    }
}

impl Instance {
    /// Uploads the rejected ranges and launches the restore on every rank's
    /// stream, behind the step that wrote them.
    fn restore_ring_tails(&self, restores: &[Tail]) -> Result<(), BundleAbiError> {
        let layout = RestoreLayout::new(self.items);
        let mut bytes = vec![0_u8; layout.prefix_pages() as usize];
        let put = |bytes: &mut Vec<u8>, offset: u64, index: usize, value: u32| {
            let at = offset as usize + 4 * index;
            bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        };
        put(&mut bytes, RestoreLayout::LIVE, 0, restores.len() as u32);
        for (index, range) in restores.iter().enumerate() {
            let start = u32::try_from(range.start)
                .map_err(|_| BundleAbiError::InvalidRange("MiMo ring restore position"))?;
            put(&mut bytes, layout.slots(), index, range.slot);
            put(&mut bytes, layout.starts(), index, start);
            put(&mut bytes, layout.counts(), index, range.count);
        }
        for rank in 0..self.tp.count() {
            let buffer = self.restore_inputs[rank];
            upload_bytes(&self.services, self.streams[rank], buffer, &bytes)?;
            let span = |offset: u64, length: u64| {
                DeviceBufferSpanV1::new(buffer.buffer, buffer.offset + offset, length)
            };
            let launch = self
                .ops
                .ring_restore(
                    restores.len() as u32,
                    &crate::ops::RingRestoreViews {
                        ring_state: self.state_views[rank],
                        slots: self.ring.slots,
                        slot_ids: span(layout.slots(), layout.region),
                        starts: span(layout.starts(), layout.region),
                        counts: span(layout.counts(), layout.region),
                        live_items: span(RestoreLayout::LIVE, 4),
                    },
                )
                .map_err(|refusal| BundleAbiError::InvalidRange(refusal.0))?
                .launch;
            crate::pipeline::launch_all(
                &self.services,
                self.streams[rank],
                &self.symbols[rank],
                &[launch],
            )
            .map_err(|_| BundleAbiError::InvalidRange("MiMo ring restore launch"))?;
        }
        Ok(())
    }
}

thread_local! {
    static INSTANCES: RefCell<InstanceTable<Instance>> = const { RefCell::new(InstanceTable::new()) };
}

/// MiMo's create services, which log each failure.
fn log(services: &HostServicesV1) -> Log<'_> {
    Log {
        services,
        bundle: "MiMo",
    }
}

fn acquire_encoder_views(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
    services: &HostServicesV1,
    ranks: &[(
        BundleContextHandleV1,
        BundleMemoryPoolHandleV1,
        BundleStreamHandleV1,
    )],
    encoder_set: bool,
    held: &mut Vec<Held>,
) -> Result<(Encoders, Vec<BTreeMap<String, DeviceBufferSpanV1>>), BundleAbiError> {
    let log = log(services);
    // The media the encoders serve, whose encoders' views the image holds.
    let media = if encoder_set {
        let role = model_mimo26_contract::weight_plan::ENCODER_SET_ROLE;
        let bytes = acquire::read_component(log, role, 1 << 10, held)?;
        model_mimo26_contract::weight_plan::parse_encoder_set(&bytes)
            .map_err(|_| BundleAbiError::InvalidRange("MiMo encoder set"))?
    } else {
        Encoders::ALL
    };
    // Every rank's index, bound to the image the host serves for that rank
    // before the image is opened.
    let log = Log {
        services,
        bundle: "MiMo encoder",
    };
    let images = acquire::RankImages {
        role: crate::encoders::views::IMAGE_ROLE,
        index_role: &|rank| format!("chop.bundle-metadata.mimo.encoder-rank-index.r{rank}.v1"),
        max_index_bytes: 1 << 20,
    };
    let (image, indexes) = acquire::rank_images(
        log,
        images,
        ranks.len() as u32,
        |rank, bytes| {
            crate::encoders::views::parse_for(geometry, bytes, rank)
                .ok()
                .filter(|index| (index.vision, index.audio) == (media.vision(), media.audio))
                .ok_or(BundleAbiError::InvalidRange("MiMo encoder rank index"))
        },
        |index| acquire::Identity {
            image_bytes: index.image_bytes,
            sha256: index.sha256.as_deref(),
            direct_identity: index.direct_identity.as_deref(),
        },
        held,
    )?;
    let arenas = acquire::upload_arenas(log, image, ranks, held)?;
    // The image is laid out as the encoder kernels read it: they read its
    // views in place.
    let views = arenas
        .into_iter()
        .enumerate()
        .map(|(rank, arena)| {
            let image_span =
                services.resolve_compiled_image_view(image, arena, rank as u32, b"")?;
            crate::encoders::views::spans(&indexes[rank], image_span.buffer)
                .map_err(|_| BundleAbiError::InvalidRange("MiMo encoder image view"))
        })
        .collect::<Result<_, _>>()?;
    Ok((media, views))
}

/// `verify_rows` is the most decode-phase rows the host enabled for one verify
/// step, zero without speculation: the verify arena and the decode and
/// verify logits buffer are sized for them, so an instance that never
/// verifies wide rows does not give KV capacity to them.
fn acquire(
    services: &HostServicesV1,
    config: &Configuration,
    target: &dyn Target,
    enable_mtp: bool,
    verify_rows: u32,
    switches: crate::switches::Switches,
    held: &mut Vec<Held>,
) -> Result<Acquired, BundleAbiError> {
    let mut stages = bundle_runtime::startup::Stages::start();
    let log = log(services);
    let authenticated = crate::projection::Prepared::authenticate(
        &acquire::read_component(
            log,
            model_mimo26_contract::prepared::COMPONENT_ROLE,
            kernel_core::precision::component::MAX_COMPONENT_BYTES as u64,
            held,
        )?,
        config.tp,
        switches,
        target,
    )
    .map_err(|message| {
        let _ = services.log(message.as_bytes());
        BundleAbiError::InvalidRange("MiMo precision component")
    })?;
    stages.end("precision component");
    let tp = config.tp;
    // Every rank's index, bound to the image the host serves for that rank
    // before the image is opened.
    let images = acquire::RankImages {
        role: RANK_IMAGE_ROLE,
        index_role: &|rank| format!("chop.bundle-metadata.mimo.rank-index.r{rank}.v1"),
        max_index_bytes: 64 << 20,
    };
    let (image, indexes) = acquire::rank_images(
        log,
        images,
        tp.ranks(),
        |rank, bytes| {
            crate::index::parse(bytes, rank)
                .and_then(|index| {
                    crate::index::validate_text(&index, tp)?;
                    Ok(index)
                })
                .map_err(|_| BundleAbiError::InvalidRange("MiMo rank index"))
        },
        acquire::identity,
        held,
    )?;
    stages.end("rank indexes");
    let ranks = acquire::open_ranks(log, &config.ordinals, held)?;
    stages.end("rank contexts");
    let arenas = acquire::upload_arenas(log, image, &ranks, held)?;
    stages.end("rank weight uploads");
    let mut all_views = Vec::new();
    for (arena, index) in arenas.into_iter().zip(&indexes) {
        // Every MiMo view is in the device arena.
        let views =
            acquire::place_views(services, image, arena, index, &[]).map_err(
                |error| match error {
                    acquire::PlaceError::Host(error) => error,
                    acquire::PlaceError::Layout(message) => {
                        let _ = services.log(message.as_bytes());
                        BundleAbiError::InvalidRange("MiMo rank arena length")
                    }
                },
            )?;
        let views = acquire::spans(views);
        crate::rank_views::validate_complete(&views, index)
            .map_err(BundleAbiError::InvalidRange)?;
        all_views.push(views);
    }
    stages.end("rank views");
    let mtp = if enable_mtp {
        mtp::acquire(services, &ranks, &all_views, held, tp, config.items)?
    } else {
        None
    };
    let dflash = if enable_mtp && mtp.is_none() {
        dflash::acquire(services, &ranks, &all_views, held, tp, config.items)?
    } else {
        None
    };
    let mut views = all_views.clone();
    if let Some(mtp) = &mtp {
        for (views, rank) in views.iter_mut().zip(&mtp.ranks) {
            views.extend(
                rank.views
                    .iter()
                    .filter(|(name, _)| name.starts_with("model.mtp."))
                    .map(|(name, span)| (name.clone(), *span)),
            );
        }
    }
    if let Some(dflash) = &dflash {
        for (views, drafter) in views.iter_mut().zip(dflash.rank_views()) {
            for (name, span) in drafter {
                views.entry(name.clone()).or_insert(*span);
            }
        }
    }
    let precision = crate::projection::Prepared::bind(&authenticated, tp, switches, target, &views)
        .map_err(|message| {
            let _ = services.log(message.as_bytes());
            BundleAbiError::InvalidRange("MiMo precision resident views")
        })?;
    stages.end("drafter");
    let items = config.items;
    // MTP verifies within the decode kernels' rows (see `create`).
    let verify_rows = if mtp.is_some() {
        verify_rows.min(DECODE_ENTRY_ROWS)
    } else {
        verify_rows
    };
    // A verify step gathers logits for every row of its token bucket, and a
    // decode step for every item of its bucket.
    let logit_rows = bundle_runtime::graphs::verify_bucket_within::<Mimo>(verify_rows, items)
        .unwrap_or(0)
        .max(items.capacity())
        .max(DECODE_ENTRY_ROWS);
    let mut working_rows = model_mimo26_contract::resources::working_rows(
        tp,
        items.get(),
        if mtp.is_some() {
            Some(model_mimo26_contract::weight_plan::Drafter::Mtp)
        } else if dflash.is_some() {
            Some(model_mimo26_contract::weight_plan::Drafter::Dflash)
        } else {
            None
        },
    )
    .into_iter()
    .map(|row| (row.name, row.layout))
    .collect::<BTreeMap<_, _>>();
    let logit_layout = model_mimo26_contract::resources::LogitsLayout::new(
        tp,
        logit_rows,
        f32_logits_from_environment(),
    );
    for (name, bytes) in [
        ("logits.local", logit_layout.local),
        ("logits.prefill", logit_layout.gathered_prefill),
        ("logits.decode", logit_layout.gathered_decode),
        ("logits.local_scored", logit_layout.local_scored),
        ("logits.scored", logit_layout.gathered_scored),
    ] {
        let row = working_rows.get_mut(name).expect("shared logits row");
        if bytes > row.allocation_bytes() {
            return Err(BundleAbiError::InvalidRange(
                "MiMo logits exceed the shared row table",
            ));
        }
        *row = model_mimo26_contract::resources::BufferLayout::bytes(bytes);
    }
    let row_bytes = |name: &str| working_rows[name].allocation_bytes();
    let encoder = if config.media {
        let (media, views) =
            acquire_encoder_views(tp.geometry, services, &ranks, config.encoder_set, held)?;
        let encoder = target.encoders(services, media, views, &mut |rank, bytes| {
            let (_, pool, stream) = ranks[rank];
            allocate_span(services, pool, stream, bytes, held)
        })?;
        stages.end("encoders");
        Some(encoder)
    } else {
        None
    };
    acquire::grant_peers(services, &ranks, held)?;
    stages.end("peer access");
    let mut arenas: [Vec<PhaseArena>; 5] = Default::default();
    let mut comm_streams = Vec::with_capacity(tp.count());
    let mut overlap_events = Vec::with_capacity(tp.count());
    let mut restore_inputs = Vec::with_capacity(tp.count());
    let mut activations = Vec::with_capacity(tp.count());
    let mut normalized = Vec::with_capacity(tp.count());
    let mut workspaces = Vec::with_capacity(tp.count());
    let mut logits = Vec::with_capacity(tp.count());
    let mut all_symbols = Vec::with_capacity(tp.count());
    let mut scratch = Vec::with_capacity(tp.count());
    // One host owner spans every rank upload and drops when acquisition ends.
    let tables = crate::rope_tables::tables_for(config.tp.geometry, config.context_tokens);
    stages.end("RoPE tables");
    let mut modules = std::time::Duration::ZERO;
    // Each symbol's dynamic shared memory, planned once for every rank.
    let mut shared_bytes = BTreeMap::new();
    for (context, pool, stream) in &ranks {
        // A pipelined prefill's all-reduce stream and its ordering events.
        let comm = bundle_runtime::lanes::SideLane::create(
            services,
            *context,
            crate::overlap::EVENTS,
            held,
        )?;
        comm_streams.push(comm.stream);
        overlap_events.push(comm.events);
        let workspace = row_bytes("workspace");
        let prefill_capacity = row_bytes("inputs.prefill");
        for (index, arena_bytes) in [
            (PREFILL, prefill_capacity),
            (DECODE, row_bytes("inputs.decode")),
            (HALVES[0], prefill_capacity),
            (HALVES[1], prefill_capacity),
            (VERIFY, row_bytes("inputs.verify")),
        ]
        .into_iter()
        .filter(|&(index, _)| index != VERIFY || verify_rows > 0)
        {
            let arena = allocate_span(services, *pool, *stream, arena_bytes, held)?;
            let block_tables = allocate_span(
                services,
                *pool,
                *stream,
                row_bytes("inputs.decode.block_tables"),
                held,
            )?;
            arenas[index].push(PhaseArena {
                arena,
                block_tables,
            });
        }
        restore_inputs.push(allocate_span(
            services,
            *pool,
            *stream,
            row_bytes("restore_inputs"),
            held,
        )?);
        activations.push(crate::scratch::bind_hidden(
            working_rows["activations"],
            &mut |bytes| allocate_span(services, *pool, *stream, bytes, held),
        )?);
        normalized.push(crate::scratch::bind_hidden(
            working_rows["normalized"],
            &mut |bytes| allocate_span(services, *pool, *stream, bytes, held),
        )?);
        let local = allocate_span(services, *pool, *stream, row_bytes("logits.local"), held)?;
        let gathered_prefill =
            allocate_span(services, *pool, *stream, row_bytes("logits.prefill"), held)?;
        let gathered_decode =
            allocate_span(services, *pool, *stream, row_bytes("logits.decode"), held)?;
        workspaces.push(allocate_span(services, *pool, *stream, workspace, held)?);
        let mut rank_scratch =
            crate::scratch::RankScratch::allocate(exchanged_rows_through(items), tp, |bytes| {
                allocate_span(services, *pool, *stream, bytes, held)
            })?;
        let table_bytes =
            config.context_tokens * crate::rope_tables::bytes_per_position(config.tp.geometry);
        debug_assert!(table_bytes <= row_bytes("rope.ga"));
        let ga = allocate_span(services, *pool, *stream, table_bytes, held)?;
        let swa = allocate_span(services, *pool, *stream, table_bytes, held)?;
        upload_bytes(services, *stream, ga, &tables[0])?;
        upload_bytes(services, *stream, swa, &tables[1])?;
        rank_scratch.rope_tables = Some([ga, swa]);
        services.zero_device_buffer(*stream, rank_scratch.gemm_semaphores)?;
        services.zero_device_buffer(*stream, rank_scratch.attention_fold_counters)?;
        services.zero_device_buffer(*stream, rank_scratch.route_ticket)?;
        // Collectives advance tagged epochs, so calls with any row count and
        // protocol can follow each other on the same flags. They only need
        // every flag, epoch and low-latency inbox slot to start at zero.
        services.zero_device_buffer(*stream, rank_scratch.comm_inbox)?;
        for state in &rank_scratch.comm_state {
            for span in [
                state.hidden_flags,
                state.hidden_epoch,
                state.gather_rows_flags,
                state.gather_rows_epoch,
            ]
            .into_iter()
            .chain(
                state
                    .exchange_rows
                    .into_iter()
                    .flat_map(|exchange| [exchange.flags, exchange.epoch]),
            ) {
                services.zero_device_buffer(*stream, span)?;
            }
        }
        upload_bytes(
            services,
            *stream,
            rank_scratch.active_experts,
            &tp.geometry.experts().to_le_bytes(),
        )?;
        upload_bytes(
            services,
            *stream,
            rank_scratch.unit_scale,
            &1.0_f32.to_le_bytes(),
        )?;
        let all_experts = std::iter::once(tp.geometry.experts())
            .chain(0..tp.geometry.experts())
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        upload_bytes(services, *stream, rank_scratch.all_experts, &all_experts)?;
        scratch.push(rank_scratch);
        logits.push(LogitsBuffers {
            local,
            gathered_prefill,
            gathered_decode,
            local_scored: allocate_span(
                services,
                *pool,
                *stream,
                row_bytes("logits.local_scored"),
                held,
            )?,
            gathered_scored: allocate_span(
                services,
                *pool,
                *stream,
                row_bytes("logits.scored"),
                held,
            )?,
        });
        let loading = std::time::Instant::now();
        let mut rank_symbols = BTreeMap::new();
        for cubin in CUBINS {
            if !target.loads(cubin.name, switches) {
                continue;
            }
            // Only the shared cubins and this topology's own are loaded.
            if !crate::topology::cubin_serves(cubin.name, tp) {
                continue;
            }
            // The ring entries of other slot counts, for an instance whose
            // largest bucket is not 32.
            if cubin.name == Selected::bundle_cubin(tp, "ring_slots")
                && items.capacity() == DECODE_ENTRY_ROWS
            {
                continue;
            }
            debug_assert!(!cubin.name.is_empty());
            let shared = |symbol: &str| match shared_bytes.get(symbol) {
                Some(&shared) => Ok(shared),
                None => {
                    let shared = target
                        .dynamic_shared_bytes(symbol, items)
                        .map_err(|_| BundleAbiError::InvalidRange("MiMo symbol shared memory"))?;
                    shared_bytes.insert(symbol.to_owned(), shared);
                    Ok(shared)
                }
            };
            acquire::load_module(
                log,
                *context,
                (cubin.name, cubin.bytes),
                cubin.symbols.iter().copied(),
                shared,
                &mut rank_symbols,
                held,
            )?;
        }
        all_symbols.push(rank_symbols);
        modules += loading.elapsed();
    }
    stages.end_excluding("rank buffers", modules);
    bundle_runtime::startup::report("rank modules", modules);
    let ring_bytes = crate::topology::ring_state_bytes(tp, items.capacity());
    let state_views = acquire::zeroed_state(services, &config.state_id, ring_bytes, &ranks, held)?;
    stages.end("ring state");
    // The host sizes and owns the serving pool after weights and state are live.
    let (kv_page_capacity, kv_views) =
        acquire::serving_kv(log, config.kv.layout_id, config.kv.page_bytes, tp.ranks())?;
    stages.end("KV pool acquisition");
    acquire::synchronize(log, &ranks)?;
    stages.end("create synchronize");
    Ok(Acquired {
        mtp,
        dflash,
        views: all_views,
        precision,
        encoder,
        contexts: ranks.iter().map(|(context, _, _)| *context).collect(),
        streams: ranks.into_iter().map(|(_, _, stream)| stream).collect(),
        state_views,
        kv_views,
        kv_page_capacity,
        arenas,
        comm_streams,
        overlap_events,
        activations,
        normalized,
        workspaces,
        logits,
        symbols: all_symbols,
        scratch,
        restore_inputs,
    })
}

impl Instance {
    /// Checks that each rank's own device keeps every spinning comm grid
    /// resident, from what the host's driver reports
    /// (`bundle_runtime::residency`). An MPS client may share SMs with
    /// other processes. The wide all-reduce is
    /// turned off unless it surely fits, so graphs above 32 rows take the
    /// phased stages; any other entry that does not fit fails create, since
    /// its calls could hang, and one the device cannot vouch for (an MPS
    /// client) is logged.
    fn check_comm_residency(&mut self, ordinals: &[i32]) -> Result<(), String> {
        // The largest graph: the decode items or the verify rows of a step.
        let graph_rows = self.items.capacity().max(self.verify_limits.rows_per_step);
        let entries = self
            .target
            .spinning_entries(graph_rows, self.overlap.as_ref())
            .map_err(|refusal| format!("MiMo comm entries: {}", refusal.0))?;
        self.wide_comm = bundle_runtime::residency::check_entries(
            &self.services,
            "MiMo",
            ordinals,
            &self.symbols,
            &entries,
            self.wide_comm,
            bundle_runtime::residency::unchecked_from_environment(),
        )?;
        Ok(())
    }
}

fn create(services: HostServicesV1, bytes: &[u8]) -> Result<Instance, Failed<Instance>> {
    let config = decode_configuration(bytes, |reason| {
        let _ = services.log(format!("MiMo create failed: {reason}").as_bytes());
    })?;
    if config.media && !crate::encoders::SERVING_READY {
        return Err(BundleStatusV1::Unsupported.into());
    }
    if let Err(error) = bundle_runtime::config::check_scored_rows(
        &services,
        "MiMo V2.6",
        crate::contract::SCORED_ROWS,
    ) {
        return Err(error.into());
    }
    // Only a diagnostic bundle runs eagerly to write taps.
    #[cfg(not(feature = "diagnostic-taps"))]
    if let Some(variable) = runtime_bundle_sdk::taps::Taps::requested(None) {
        let _ = services.log(
            format!(
                "{variable} is set, but this MiMo V2.6 bundle was built without the \
                 diagnostic-taps feature; unset it to serve"
            )
            .as_bytes(),
        );
        return Err(BundleStatusV1::InvalidArgument.into());
    }
    #[cfg(feature = "diagnostic-taps")]
    let taps = match crate::taps::from_env_for(config.tp) {
        Ok(taps) => taps.map(|taps| taps.with_producer(config.tap_producer.clone())),
        Err(error) => {
            let _ = services.log(error.as_bytes());
            return Err(BundleStatusV1::InvalidArgument.into());
        }
    };
    // The ring's declaration is the same with any drafter; the ledger takes
    // the loaded drafter's declaration below.
    let declared = model_mimo26_contract::state::request_state(
        config.tp,
        &config.kv,
        crate::prefix::Drafter::None,
    );
    let Ok(window) = WindowRing::new(&declared, 0, prefix_state::state_extent(config.items)) else {
        return Err(BundleStatusV1::InvalidArgument.into());
    };
    // The instance's target: SM120, with its own-op set.
    let target: Box<dyn Target> = Box::new(crate::sm120::own_ops::Sm120Own::new(config.tp));
    let overlap = target.lanes();
    let mut instance = Instance {
        services,
        context_tokens: config.context_tokens,
        tp: config.tp,
        target,
        items: config.items,
        #[cfg(not(feature = "diagnostic-taps"))]
        eager: eager(),
        // A tapped instance captures no graph.
        #[cfg(feature = "diagnostic-taps")]
        eager: eager() || taps.is_some(),
        held: Vec::new(),
        kv: config.kv,
        ring: crate::ring::RingGeometry::served(config.tp, config.items),
        ledger: ring_ledger(config.context_tokens, config.items, &declared),
        window,
        ring_saved: vec![Vec::new(); config.tp.count()],
        views: Vec::new(),
        precision: None,
        encoder: None,
        contexts: Vec::new(),
        streams: Vec::new(),
        state_views: Vec::new(),
        kv_views: Vec::new(),
        kv_page_capacity: 0,
        kv_dtype: config.kv.element,
        kv_scales: config.kv.scales,
        wide_comm: crate::switches::wide_comm_enabled(),
        switches: crate::switches::Switches::configured(),
        controls: crate::switches::Controls::configured(),
        ops: Default::default(),
        f32_logits: f32_logits_from_environment(),
        row_owned_logits: row_owned_logits_from_environment(),
        row_owned_speculation: row_owned_speculation_from_environment(),
        verify_bucket: 0,
        decode_inputs: Vec::new(),
        verify_inputs: Vec::new(),
        prefill_inputs: Vec::new(),
        prefill_layout: crate::upload::InputLayout::decode_within(config.items),
        arenas: Default::default(),
        block_tables: std::array::from_fn(|_| {
            crate::upload::BlockTableMirror::new(config.tp.model_limits(), config.items)
        }),
        comm_streams: Vec::new(),
        overlap_events: Vec::new(),
        collectives: Default::default(),
        overlap,
        #[cfg(test)]
        fold_override: None,
        activations: Vec::new(),
        normalized: Vec::new(),
        workspaces: Vec::new(),
        logits: Vec::new(),
        symbols: Vec::new(),
        scratch: Vec::new(),
        graphs: crate::graphs::GraphCache::with_items(
            bundle_runtime::geometry::Ranks::new(config.tp.ranks()).expect("supported rank count"),
            config.items,
        ),
        verify_limits: crate::inputs::VerifyLimits::NONE,
        restore_inputs: Vec::new(),
        mtp: None,
        dflash: None,
        #[cfg(feature = "diagnostic-taps")]
        taps,
        #[cfg(feature = "diagnostic-taps")]
        tap_encoders: Vec::new(),
        #[cfg(feature = "diagnostic-taps")]
        tap_occurrences: Default::default(),
        #[cfg(feature = "diagnostic-taps")]
        pending_taps: Vec::new(),
    };
    // A host without the speculation service reports speculation off.
    match instance.services.query_speculation() {
        Ok(enabled) => {
            instance.verify_limits = crate::inputs::VerifyLimits::enabled_within::<Mimo>(
                enabled.maximum_draft_tokens,
                enabled.maximum_verify_tokens,
                config.items,
            );
        }
        Err(error) => return Err(status(error).into()),
    }
    let enable_mtp = config.drafter && instance.verify_limits.rows_per_item > 0;
    match acquire(
        &instance.services,
        &config,
        &*instance.target,
        enable_mtp,
        instance.verify_limits.rows_per_step,
        instance.switches,
        &mut instance.held,
    ) {
        Ok(acquired) => {
            instance.mtp = acquired.mtp;
            instance.dflash = acquired.dflash;
            instance.ledger = ring_ledger(
                instance.context_tokens,
                instance.items,
                &instance.request_state(),
            );
            if let Some(mtp) = &mut instance.mtp {
                mtp.eager |= instance.eager;
            }
            // Drafter and target graphs interleave in one host release order.
            if let Some(mtp) = &instance.mtp {
                for cache in mtp.graphs.borrow_mut().iter_mut() {
                    *cache = crate::graphs::GraphCache::sharing(&instance.graphs);
                }
            }
            if let Some(dflash) = &instance.dflash {
                *dflash.graphs.borrow_mut() = crate::graphs::GraphCache::sharing(&instance.graphs);
            }
            instance.verify_limits.bundle_drafter =
                instance.mtp.is_some() || instance.dflash.is_some();
            // MTP drafts and catches up at decode capacities, so its verify
            // steps stay within the decode kernels' rows.
            if instance.mtp.is_some() {
                instance.verify_limits.rows_per_step =
                    instance.verify_limits.rows_per_step.min(DECODE_ENTRY_ROWS);
            }
            instance.verify_bucket =
                verify_bucket_from_environment(instance.verify_limits.rows_per_step);
            instance.views = acquired.views;
            instance.precision = Some(acquired.precision);
            instance.encoder = acquired.encoder;
            // The encoder runtime reads its blocks back only for block taps.
            #[cfg(feature = "diagnostic-taps")]
            if let (Some(encoder), Some(taps)) = (instance.encoder.as_mut(), &instance.taps) {
                encoder.block_taps = crate::taps::selects_blocks_for(instance.tp, taps);
                encoder.code_taps = true;
            }
            instance.contexts = acquired.contexts;
            instance.streams = acquired.streams;
            instance.state_views = acquired.state_views;
            instance.kv_views = acquired.kv_views;
            instance.kv_page_capacity = acquired.kv_page_capacity;
            instance.decode_inputs = acquired.arenas[DECODE]
                .iter()
                .map(|arena| {
                    DeviceInputs::within(
                        *arena,
                        &crate::upload::InputLayout::decode_within(config.items),
                    )
                })
                .collect();
            instance.verify_inputs = acquired.arenas[VERIFY]
                .iter()
                .map(|arena| {
                    DeviceInputs::within(
                        *arena,
                        &crate::upload::InputLayout::verify_within::<Mimo>(config.items),
                    )
                })
                .collect();
            instance.prefill_layout =
                crate::upload::InputLayout::prefill_largest_within::<Mimo>(config.items);
            instance.prefill_inputs = acquired.arenas[PREFILL]
                .iter()
                .map(|arena| DeviceInputs::within(*arena, &instance.prefill_layout))
                .collect();
            instance.arenas = acquired.arenas;
            instance.comm_streams = acquired.comm_streams;
            instance.overlap_events = acquired.overlap_events;
            instance.activations = acquired.activations;
            instance.normalized = acquired.normalized;
            instance.workspaces = acquired.workspaces;
            instance.logits = acquired.logits;
            instance.symbols = acquired.symbols;
            instance.scratch = acquired.scratch;
            instance.restore_inputs = acquired.restore_inputs;
        }
        Err(error) => {
            let _ = instance
                .services
                .log(format!("MiMo create failed: {error}").as_bytes());
            let result = status(error);
            return Err(Failed::releasing(instance, result));
        }
    }
    let residency = std::time::Instant::now();
    let checked = instance.check_comm_residency(&config.ordinals);
    bundle_runtime::startup::report("comm residency", residency.elapsed());
    if let Err(error) = checked {
        let _ = instance
            .services
            .log(format!("MiMo create failed: {error}").as_bytes());
        return Err(Failed::releasing(instance, BundleStatusV1::Unsupported));
    }
    // Every op is prepared once the wide all-reduce's residency is known.
    let preparing = std::time::Instant::now();
    let prepared = instance
        .target
        .backend(instance.switches, instance.wide_comm)
        .map_err(kernel_core::backend::Refused::from)
        .and_then(|backend| {
            let scratch = &instance.scratch[0];
            crate::ops::Ops::prepare(
                &*backend,
                &crate::ops::Plan {
                    tp: instance.tp,
                    switches: instance.switches,
                    kv: instance.kv_dtype,
                    items: instance.items.capacity(),
                    target: &*instance.target,
                    state: &|bucket| {
                        [
                            crate::scratch::comm_inbox(scratch.comm_inbox, bucket).length,
                            scratch.comm_state[bucket].hidden_flags.length,
                            scratch.comm_state[bucket].hidden_epoch.length,
                        ]
                    },
                    gathers: &|bucket| {
                        let state = &scratch.comm_state[bucket];
                        crate::ops::GatherRegions {
                            rows: [
                                state.gather_rows_flags.length,
                                state.gather_rows_epoch.length,
                            ],
                            exchange: state
                                .exchange_rows
                                .map(|exchange| [exchange.flags.length, exchange.epoch.length]),
                        }
                    },
                    dflash: instance
                        .dflash
                        .as_ref()
                        .map(|dflash| dflash.candidate_state()),
                    own: crate::ops::OwnPlan {
                        ops: instance.target.own(),
                        prefix: instance.prefix_copy_op(),
                        served: instance.items.get(),
                        mtp: instance.mtp.is_some(),
                    },
                },
            )
        });
    bundle_runtime::startup::report("ops", preparing.elapsed());
    match prepared {
        Ok(ops) => instance.ops = ops,
        Err(refused) => {
            let _ = instance
                .services
                .log(format!("MiMo create failed: an op was refused: {refused}").as_bytes());
            return Err(Failed::releasing(instance, BundleStatusV1::Unsupported));
        }
    }
    // Runners record which all-reduce and dense GEMMs served the buckets
    // above 32 rows.
    if config.items.capacity() > DECODE_ENTRY_ROWS {
        let medium = instance.switches.dense_mid;
        let _ = instance.services.log(
            format!(
                "MiMo decode graphs above {DECODE_ENTRY_ROWS} rows: wide all-reduce {} ({}={}), \
                 medium-M dense GEMMs {} ({}={})",
                if instance.wide_comm { "on" } else { "off" },
                crate::switches::WIDE_COMM,
                u8::from(crate::switches::wide_comm_enabled()),
                if medium { "on" } else { "off" },
                crate::switches::VERIFY_DENSE_MID,
                u8::from(medium),
            )
            .as_bytes(),
        );
    }
    Ok(instance)
}

impl Live for Instance {
    fn collectives(&self) -> &crate::pipeline::Collectives {
        &self.collectives
    }
}

impl Bundle for Instance {
    fn instances() -> &'static LocalKey<RefCell<InstanceTable<Self>>> {
        &INSTANCES
    }

    fn create(services: HostServicesV1, configuration: &[u8]) -> Result<Self, Failed<Self>> {
        create(services, configuration)
    }

    fn resources(&mut self) -> (&HostServicesV1, &mut Vec<Held>) {
        (&self.services, &mut self.held)
    }

    fn release_graphs(&mut self) -> Result<(), BundleAbiError> {
        self.mtp_release_graphs()?;
        self.dflash_release_graphs()?;
        self.graphs.release_all(&self.services)
    }

    fn execute(&mut self, request: &[u8], sink: &mut ExecuteResultSinkV1) -> BundleStatusV1 {
        if self.views.len() != self.tp.count() {
            return BundleStatusV1::InvalidState;
        }
        self.execute_step(request, sink)
    }

    fn commit(&mut self) -> BundleStatusV1 {
        transaction::commit(self)
    }

    fn commit_accepted(
        &mut self,
        records: &[runtime_bundle_abi::BundleKeptRowsV1],
    ) -> BundleStatusV1 {
        transaction::commit_accepted(self, records)
    }

    fn cancel(&mut self) -> BundleStatusV1 {
        transaction::cancel(self)
    }

    fn retire_request(&mut self, request_id: u64) -> BundleStatusV1 {
        // A request cancelled before it ran, or before its prefix restore,
        // never had a slot, and retiring it is not an error.
        transaction::retire(self, request_id)
    }
}

/// Ends the ring transaction. The shadow already holds every touched
/// entry's value from before the step, so a cancel copies the touched cells
/// back, and a commit of verify items restores their rejected tails from the
/// shadow on every rank. A host failure during drafter history copies or
/// tail restoration can leave partial state; the ABI client poisons the
/// instance and forbids further work.
impl Steps for Instance {
    // MiMo refuses a retire during any open step, not only one that holds
    // the request.
    const RETIRE: Retire = Retire::DuringNoStep;

    fn ledger(&mut self) -> &mut Ledger {
        &mut self.ledger
    }

    fn status(&self, failure: Failure) -> BundleStatusV1 {
        match failure {
            Failure::VerifyItems | Failure::NotOpen | Failure::StepOpen => {
                BundleStatusV1::InvalidState
            }
            Failure::Ledger(Stage::Retire, _) => BundleStatusV1::BundleError,
            Failure::Ledger(_, error) => ledger_status(error),
        }
    }

    /// A cancel ends the drafters' pending rows first, whether or not a
    /// step is open.
    fn begin(&mut self, end: End) -> Result<(), BundleStatusV1> {
        if end == End::Cancel {
            self.mtp_finish(&[], None).map_err(status)?;
            self.dflash_finish(&[], None).map_err(status)?;
        }
        Ok(())
    }

    fn prepare(&mut self, kept: &[(u64, u32)], plan: &CommitPlan) -> Result<(), BundleStatusV1> {
        self.mtp_finish(kept, Some(plan)).map_err(status)?;
        self.dflash_finish(kept, Some(plan)).map_err(status)
    }

    fn restore(&mut self, step: &Step) -> Result<(), BundleStatusV1> {
        let appends = step
            .items
            .iter()
            .map(|placed| (placed.slot, placed.start, placed.rows))
            .collect::<Vec<_>>();
        let cells = self
            .window
            .touched(&appends)
            .map_err(|_| BundleStatusV1::InvalidState)?;
        self.restore_ring(&cells).map_err(status)
    }

    fn ended(&mut self, ended: Ended<'_>) -> Result<(), BundleStatusV1> {
        let Ended::Committed(committed) = ended else {
            return Ok(());
        };
        let restores = bundle_runtime::ring::tails(committed);
        if !restores.is_empty() {
            self.restore_ring_tails(&restores).map_err(status)?;
        }
        Ok(())
    }

    #[cfg(feature = "diagnostic-taps")]
    fn finish_taps(&mut self, commit: bool) -> BundleStatusV1 {
        Instance::finish_taps(self, commit)
    }

    fn before_retire(&mut self, request: u64) -> Result<(), BundleStatusV1> {
        self.mtp_retire(request).map_err(status)?;
        self.dflash_retire(request).map_err(status)
    }

    #[cfg(feature = "diagnostic-taps")]
    fn retired(&mut self, request: u64) {
        self.tap_occurrences.retire(request);
    }
}

impl PrefixInstance for Instance {
    fn state_pages(&self) -> Result<u64, BundleStatusV1> {
        self.prefix_state_pages().map_err(status)
    }

    fn capture(
        &mut self,
        request: &runtime_bundle_abi::PrefixStateRequestV1,
    ) -> Result<(), BundleStatusV1> {
        self.capture_prefix(request)
    }

    /// The ring picks the slot, so a restore into a host slot is not
    /// supported.
    fn restore(
        &mut self,
        request: &runtime_bundle_abi::PrefixStateRequestV1,
        slot: Option<u64>,
    ) -> Result<(), BundleStatusV1> {
        if slot.is_some() {
            return Err(BundleStatusV1::Unsupported);
        }
        self.restore_prefix(request)
    }
}

#[cfg(test)]
pub(crate) fn preflight_request(handle: BundleHandleV1, request: &[u8]) -> BundleStatusV1 {
    INSTANCES.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        instance
            .preflight(request)
            .map_or_else(|error| error, |_| BundleStatusV1::Success)
    })
}

#[cfg(test)]
pub(crate) fn rank_view(
    handle: BundleHandleV1,
    rank: u32,
    name: &str,
) -> Result<DeviceBufferSpanV1, BundleStatusV1> {
    INSTANCES.with(|table| {
        let mut table = table
            .try_borrow_mut()
            .map_err(|_| BundleStatusV1::InvalidState)?;
        let instance = table.get_mut(handle).ok_or(BundleStatusV1::StaleHandle)?;
        instance
            .views
            .get(rank as usize)
            .and_then(|views| views.get(name))
            .copied()
            .ok_or(BundleStatusV1::NotFound)
    })
}

#[cfg(test)]
pub(crate) fn upload_packed_phase(
    handle: BundleHandleV1,
    phase: &crate::inputs::PhaseInputs,
) -> BundleStatusV1 {
    INSTANCES.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        instance
            .upload_phase(phase)
            .map_or_else(status, |_| BundleStatusV1::Success)
    })
}

#[cfg(test)]
fn entry_probe(instance: &Instance) -> Vec<crate::pipeline::PreparedLaunch> {
    let phase = crate::selection::Phase::Decode(1);
    vec![
        crate::row_plan::embedding(
            instance.tp,
            phase,
            instance.switches,
            1,
            instance.views[0]["model.embed_tokens.weight"],
            instance.decode_inputs[0].token_ids,
            instance.activations[0],
            instance.decode_inputs[0].live_tokens,
        )
        .expect("full embedding launch"),
        instance
            .ops
            .hidden_norm(
                phase,
                0,
                instance.switches,
                1,
                instance.activations[0],
                instance.views[0]["model.layers.0.input_layernorm.weight"],
                instance.normalized[0],
                instance.decode_inputs[0].live_tokens,
            )
            .expect("initial norm launch")
            .launch,
    ]
}

#[cfg(test)]
pub(crate) fn launch_embedding_probe(
    handle: BundleHandleV1,
    decode_bucket: Option<u32>,
) -> BundleStatusV1 {
    INSTANCES.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        let launches = entry_probe(instance);
        instance
            .run_prepared(0, decode_bucket.map(|bucket| (bucket, 0)), &launches)
            .map_or(BundleStatusV1::InvalidState, |_| BundleStatusV1::Success)
    })
}

#[cfg(test)]
pub(crate) fn prepare_ring_append(
    handle: BundleHandleV1,
    appends: &[(u64, u64, u32)],
) -> BundleStatusV1 {
    let items = appends
        .iter()
        .map(|&(request, start, rows)| Item {
            request,
            slot: None,
            start,
            rows,
            verify: false,
        })
        .collect::<Vec<_>>();
    INSTANCES.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        instance.prepare_ring(&items)
    })
}

/// Sets the live instance's encoder mode and arena captures, and drops
/// `missing_rank`'s image views.
#[cfg(test)]
pub(crate) fn encoder_test_options(
    handle: BundleHandleV1,
    serial: bool,
    arena: Option<std::path::PathBuf>,
    missing_rank: Option<usize>,
) {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).unwrap();
        let encoder = instance
            .encoder
            .as_mut()
            .expect("an instance with encoders");
        encoder.mode = match serial {
            true => bundle_runtime::encoder::Mode::Serial,
            false => bundle_runtime::encoder::Mode::Relocated,
        };
        if arena.is_some() {
            encoder.arena_capture = arena;
        }
        if let Some(rank) = missing_rank {
            encoder.ranks[rank].weights.clear();
        }
    });
}

/// Sets the live instance's encoder capture directory
/// (`CHOPINFER_ENCODER_CAPTURE_DIR`).
#[cfg(test)]
pub(crate) fn encoder_test_capture(handle: BundleHandleV1, dir: std::path::PathBuf) {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).unwrap();
        instance
            .encoder
            .as_mut()
            .expect("an instance with encoders")
            .capture = Some(dir);
    });
}

/// The MediaInject launches of `rank` for a prefill part of `rows` rows
/// from `row0`, and the spans the injection reads: the rank's activation,
/// its prefill live token count and its encoder's arena and source rows.
#[cfg(test)]
pub(crate) fn media_inject_probe(
    handle: BundleHandleV1,
    rank: usize,
    row0: u64,
    rows: u32,
) -> (
    Result<Vec<crate::pipeline::PreparedLaunch>, kernel_core::contract::Refusal>,
    [DeviceBufferSpanV1; 4],
) {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("live instance");
        let inputs = instance.prefill_inputs[rank];
        let encoder = &instance.encoder.as_ref().expect("an encoder").ranks[rank];
        let spans = [
            instance.activations[rank],
            inputs.live_tokens,
            encoder.region("arena").expect("arena"),
            encoder.region("source-rows").expect("source rows"),
        ];
        let launches = instance
            .plan_stage_in_lowered(
                rank,
                crate::selection::Phase::Prefill,
                rows,
                1,
                1,
                &[0],
                false,
                crate::program::Stage::MediaInject,
                inputs,
                row0,
            )
            .map(crate::program_lower::launches);
        (launches, spans)
    })
}

/// Gives the live instance `encoder`.
#[cfg(test)]
pub(crate) fn force_encoder(handle: BundleHandleV1, encoder: bundle_runtime::encoder::Encoder) {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("live instance");
        instance.encoder = Some(encoder);
    });
}

#[cfg(test)]
pub(crate) fn force_fold_decode(handle: BundleHandleV1, enabled: bool) {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("live instance")
            .fold_override = Some(enabled);
    });
}

#[cfg(test)]
pub(crate) fn force_overlap(handle: BundleHandleV1, config: Option<crate::overlap::Config>) {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("live instance")
            .overlap = config;
    });
}

/// Replaces the FP8 pool's per-layer scales, so tests can see them reach
/// the launches.
#[cfg(test)]
pub(crate) fn force_kv_scales(handle: BundleHandleV1, scales: &'static [[f32; 2]]) {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("live instance")
            .kv_scales = scales;
    });
}

#[cfg(test)]
pub(crate) fn force_f32_logits(handle: BundleHandleV1, enabled: bool) {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("live instance")
            .f32_logits = enabled;
    });
}

/// Taps `taps` from the next step on, as a create with them would.
#[cfg(all(test, feature = "diagnostic-taps"))]
pub(crate) fn force_taps(handle: BundleHandleV1, taps: runtime_bundle_sdk::taps::Taps) {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("live instance");
        if let Some(encoder) = instance.encoder.as_mut() {
            encoder.block_taps = crate::taps::selects_blocks_for(instance.tp, &taps);
            encoder.code_taps = true;
        }
        instance.taps = Some(taps);
        instance.eager = true;
    });
}

#[cfg(test)]
pub(crate) fn dflash_cache_views(handle: BundleHandleV1) -> Vec<DeviceBufferSpanV1> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("live instance");
        (0..instance.tp.count())
            .map(|rank| instance.dflash_projected_span(rank).expect("DFlash cache"))
            .collect()
    })
}

#[cfg(test)]
pub(crate) fn comm_streams(handle: BundleHandleV1) -> Vec<u32> {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("live instance")
            .comm_streams
            .iter()
            .map(|stream| stream.slot())
            .collect()
    })
}

#[cfg(test)]
pub(crate) fn resource_spans(handle: BundleHandleV1) -> Vec<[DeviceBufferSpanV1; 8]> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("live instance");
        instance
            .logits
            .iter()
            .enumerate()
            .map(|(rank, logits)| {
                [
                    instance.activations[rank],
                    instance.normalized[rank],
                    logits.local,
                    logits.gathered_prefill,
                    logits.gathered_decode,
                    instance.workspaces[rank],
                    logits.local_scored,
                    logits.gathered_scored,
                ]
            })
            .collect()
    })
}

/// Observes the ledger and drafter boundary after a terminal callback failure.
#[cfg(test)]
pub(crate) fn lifecycle_test_state(
    handle: BundleHandleV1,
    request: u64,
) -> (bool, Option<u64>, bool) {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).unwrap();
        let boundary = instance.ledger.slot_of(request).and_then(|slot| {
            instance
                .ledger
                .holder(slot)
                .map(|(_, tokens)| (slot, tokens))
        });
        let ready = boundary.is_some_and(|(slot, tokens)| {
            instance.mtp_prefix_ready(request, slot, tokens)
                && instance.dflash_prefix_ready(request, slot, tokens)
        });
        (
            instance.ledger.step().is_some(),
            boundary.map(|(_, tokens)| tokens),
            ready,
        )
    })
}

#[cfg(test)]
pub(crate) fn set_test_tail_window(handle: BundleHandleV1, window: Option<u64>) {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .unwrap()
            .ledger
            .set_test_tail_window(window);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{
        GA_PAGE_BYTES_BF16, GA_PAGE_BYTES_FP8, KV_LAYOUT_BF16, KV_LAYOUT_FP8, RANKS,
        RING_STATE_BYTES_PER_RANK, TOPOLOGY,
    };

    fn decode_configuration(bytes: &[u8]) -> Result<Configuration, BundleStatusV1> {
        super::decode_configuration(bytes, |_| {})
    }

    #[test]
    fn configuration_refuses_unknown_and_noncanonical_topology_ids() {
        for id in [
            "mimo-absent-tp4",
            "mimo-v26-flash-tp02",
            "mimo-v26-flash-tp8",
        ] {
            let mut bytes = bundle_runtime::config::MAGIC.to_vec();
            bytes.extend_from_slice(&(id.len() as u32).to_le_bytes());
            bytes.extend_from_slice(id.as_bytes());
            assert!(matches!(
                decode_configuration(&bytes),
                Err(BundleStatusV1::Unsupported)
            ));
        }
    }

    #[test]
    fn drafter_validation_and_host_failures_have_distinct_statuses() {
        let mut staging = crate::mtp::staging::Staging::default();
        for refusal in [
            staging.catch_up(7, 1, 40).unwrap_err(),
            staging.prepare(Vec::new()).unwrap_err(),
        ] {
            assert_eq!(
                drafter_status(BundleAbiError::InvalidRange(refusal)),
                BundleStatusV1::InvalidArgument
            );
        }
        assert_eq!(
            drafter_status(BundleAbiError::Status(BundleStatusV1::HostError)),
            BundleStatusV1::HostError
        );
    }

    #[test]
    fn restore_layout_regions_hold_one_u32_per_largest_bucket_item() {
        for served in [1, 32, 33, 100, 256] {
            let items = Items::new(served).unwrap();
            let layout = RestoreLayout::new(items);
            let regions = [
                RestoreLayout::LIVE,
                layout.slots(),
                layout.starts(),
                layout.counts(),
            ];
            for (index, &offset) in regions.iter().enumerate() {
                assert_eq!(offset, index as u64 * layout.region);
                assert_eq!(offset % 128, 0);
            }
            assert!(layout.region >= u64::from(items.capacity()) * 4);
            assert_eq!(layout.prefix_pages(), 4 * layout.region);
            assert_eq!(
                layout.prefix_metadata(),
                layout.prefix_pages() + PREFIX_PAGES_BYTES
            );
            assert_eq!(
                layout.bytes(),
                layout.prefix_metadata() + crate::prefix_projected::HEADER_BYTES
            );
            // The item arrays of the largest bucket never reach the next region.
            assert!(layout.counts() + 4 * u64::from(items.capacity()) <= layout.prefix_pages());
        }
        // 32 served keeps the layout its launch journals record.
        let layout = RestoreLayout::new(Items::new(32).unwrap());
        assert_eq!(
            [
                layout.slots(),
                layout.starts(),
                layout.counts(),
                layout.prefix_pages()
            ],
            [128, 256, 384, 512]
        );
    }

    fn configuration_with_media(media_indexes: u32) -> Vec<u8> {
        configuration_with(KV_LAYOUT_BF16, GA_PAGE_BYTES_BF16, media_indexes)
    }

    fn configuration_with(layout_id: &[u8], page_bytes: u64, media_indexes: u32) -> Vec<u8> {
        configuration_with_encoder_set(layout_id, page_bytes, media_indexes, false)
    }

    fn configuration_with_encoder_set(
        layout_id: &[u8],
        page_bytes: u64,
        media_indexes: u32,
        encoder_set: bool,
    ) -> Vec<u8> {
        let mut roles = vec![
            RANK_IMAGE_ROLE.to_vec(),
            model_mimo26_contract::prepared::COMPONENT_ROLE
                .as_bytes()
                .to_vec(),
        ];
        if encoder_set {
            roles.push(
                model_mimo26_contract::weight_plan::ENCODER_SET_ROLE
                    .as_bytes()
                    .to_vec(),
            );
        }
        roles.extend(
            (0..RANKS).map(|rank| {
                format!("chop.bundle-metadata.mimo.rank-index.r{rank}.v1").into_bytes()
            }),
        );
        if media_indexes != 0 {
            roles.push(crate::encoders::views::IMAGE_ROLE.to_vec());
            roles.extend((0..media_indexes).map(|rank| {
                format!("chop.bundle-metadata.mimo.encoder-rank-index.r{rank}.v1").into_bytes()
            }));
        }
        bundle_runtime::config::test_support::encode(
            TOPOLOGY.id.as_bytes(),
            &(0..RANKS as i32).collect::<Vec<_>>(),
            &bundle_runtime::config::KvLayout {
                layout_id: layout_id.to_vec(),
                page_bytes,
                alignment: 256,
                page_tokens: PAGE_TOKENS,
                ranks: u64::from(RANKS),
                lifetime: 0,
            },
            &bundle_runtime::config::StateLayout {
                state_id: RING_STATE_ID.to_vec(),
                bytes: RING_STATE_BYTES_PER_RANK,
                alignment: 256,
            },
            &roles,
            // 32 served sequences, whose ring state the record declares.
            Some(32),
        )
    }

    #[test]
    fn unknown_kv_layout_names_the_geometry_and_layout() {
        let messages = RefCell::new(Vec::new());
        let record = configuration_with(b"mimo.ga-paged-fp8.v2.unknown", GA_PAGE_BYTES_FP8, 0);
        assert!(matches!(
            super::decode_configuration(&record, |message| messages
                .borrow_mut()
                .push(message.to_owned())),
            Err(BundleStatusV1::Unsupported)
        ));
        assert_eq!(
            *messages.borrow(),
            ["unknown KV layout mimo.ga-paged-fp8.v2.unknown for geometry v26-flash"]
        );
    }

    fn configuration() -> Vec<u8> {
        configuration_with_media(0)
    }

    /// The served item count closes the record: counts up to the package
    /// maximum are accepted, larger ones refused, and an older host's record
    /// without it serves the package maximum. The ring state holds one slot
    /// per item of the largest bucket (`chop.state-per-item.v1`), so the
    /// record's state bytes must match the count.
    #[test]
    fn the_served_item_count_is_read_within_the_package_maximum() {
        let ring =
            |served: u32| crate::topology::ring_state_bytes(Tp::TP4, served.next_power_of_two());
        let served = |tail: Option<u32>, state: u64| {
            let mut bytes = configuration();
            bytes.truncate(bytes.len() - 4);
            let at = bytes
                .windows(RING_STATE_ID.len())
                .position(|window| window == RING_STATE_ID)
                .unwrap()
                + RING_STATE_ID.len();
            bytes[at..at + 8].copy_from_slice(&state.to_le_bytes());
            if let Some(count) = tail {
                bytes.extend_from_slice(&count.to_le_bytes());
            }
            decode_configuration(&bytes).map(|config| config.items.get())
        };
        let package = crate::contract::DECODE_BATCH;
        assert_eq!(served(None, ring(package)), Ok(package));
        for count in [1, 8, 32, 33, 100, package]
            .into_iter()
            .filter(|&count| count <= package)
        {
            assert_eq!(served(Some(count), ring(count)), Ok(count), "{count}");
            // A state of another served count's size is refused.
            assert_eq!(
                served(Some(count), ring(2 * count)),
                Err(BundleStatusV1::Unsupported)
            );
        }
        assert_eq!(
            served(Some(package + 1), ring(package)),
            Err(BundleStatusV1::Unsupported)
        );
        assert_eq!(
            served(Some(0), ring(1)),
            Err(BundleStatusV1::InvalidArgument)
        );
    }

    /// Creation reads any served count up to 256 at every topology, with or
    /// without the encoders and a drafter image. The package declares the
    /// maximum its images leave KV room for (`topology::maximum_sequences`),
    /// which the host enforces; only MTP's index limits the count further,
    /// once acquisition tells it from DFlash.
    #[test]
    fn creation_reads_every_count_up_to_the_item_limit() {
        let record = |tp: Tp, media: bool, drafter: bool, served: u32| {
            let mut roles = vec![
                RANK_IMAGE_ROLE.to_vec(),
                model_mimo26_contract::prepared::COMPONENT_ROLE
                    .as_bytes()
                    .to_vec(),
            ];
            roles.extend((0..tp.ranks()).map(|rank| {
                format!("chop.bundle-metadata.mimo.rank-index.r{rank}.v1").into_bytes()
            }));
            if media {
                roles.push(crate::encoders::views::IMAGE_ROLE.to_vec());
                roles.extend((0..tp.ranks()).map(|rank| {
                    format!("chop.bundle-metadata.mimo.encoder-rank-index.r{rank}.v1").into_bytes()
                }));
            }
            if drafter {
                roles.push(crate::mtp::views::IMAGE_ROLE.to_vec());
                roles.extend((0..tp.ranks()).map(|rank| {
                    format!("chop.bundle-metadata.mimo.drafter-rank-index.r{rank}.v1").into_bytes()
                }));
            }
            let bytes = bundle_runtime::config::test_support::encode(
                crate::topology::topology_id(tp).as_bytes(),
                &(0..tp.ranks() as i32).collect::<Vec<_>>(),
                &bundle_runtime::config::KvLayout {
                    layout_id: KV_LAYOUT_BF16.to_vec(),
                    page_bytes: crate::topology::ga_page_bytes(tp, false),
                    alignment: 256,
                    page_tokens: PAGE_TOKENS,
                    ranks: u64::from(tp.ranks()),
                    lifetime: 0,
                },
                &bundle_runtime::config::StateLayout {
                    state_id: RING_STATE_ID.to_vec(),
                    bytes: crate::topology::ring_state_bytes(tp, served.next_power_of_two()),
                    alignment: 256,
                },
                &roles,
                Some(served),
            );
            decode_configuration(&bytes).map(|config| config.items.get())
        };
        let limit = crate::contract::DECODE_BATCH;
        for tp in crate::topology::all() {
            for media in [false, true] {
                for drafter in [false, true] {
                    for served in [1, 32, 64, limit] {
                        assert_eq!(
                            record(tp, media, drafter, served),
                            if media && !crate::topology::encoders_available(tp) {
                                Err(BundleStatusV1::Unsupported)
                            } else {
                                Ok(served)
                            },
                            "{tp:?} media {media} drafter {drafter} {served}"
                        );
                    }
                }
            }
        }
    }

    /// Every snapshot fits the prefix page table in either layout at every
    /// rank count; FP8's half-size pages double the count.
    #[test]
    fn prefix_snapshots_fit_the_page_table_in_every_kv_layout() {
        use crate::prefix::Drafter;
        for format in crate::contract::KV_FORMATS {
            for tp in Tp::SUPPORTED {
                for drafter in [Drafter::None, Drafter::Mtp, Drafter::Dflash] {
                    let state = model_mimo26_contract::state::request_state(tp, &format, drafter);
                    assert!(
                        state.snapshot_pages() <= PREFIX_PAGES_BYTES / 8,
                        "{tp:?} {drafter:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn fp8_layout_selects_fp8_pages_and_per_layer_scales() {
        let config = decode_configuration(&configuration_with(KV_LAYOUT_FP8, GA_PAGE_BYTES_FP8, 0))
            .expect("FP8 geometry");
        assert!(config.kv.fp8());
        assert_eq!(config.kv.page_bytes, GA_PAGE_BYTES_FP8);
        assert_eq!(config.kv.scales, &crate::contract::GA_FP8_KV_SCALES);
        for (layout, page) in [
            (KV_LAYOUT_FP8, GA_PAGE_BYTES_BF16),
            (KV_LAYOUT_BF16, GA_PAGE_BYTES_FP8),
            (&b"mimo.ga-paged-fp16.v1"[..], GA_PAGE_BYTES_BF16),
            // The unscaled FP8 layout of earlier packages.
            (&b"mimo.ga-paged-fp8.v1"[..], GA_PAGE_BYTES_FP8),
        ] {
            assert!(matches!(
                decode_configuration(&configuration_with(layout, page, 0)),
                Err(BundleStatusV1::Unsupported)
            ));
        }
    }

    #[test]
    fn encoder_set_requires_an_encoder_image() {
        for (indexes, accepted) in [(0, false), (4, true)] {
            let bytes =
                configuration_with_encoder_set(KV_LAYOUT_BF16, GA_PAGE_BYTES_BF16, indexes, true);
            match decode_configuration(&bytes) {
                Ok(config) => {
                    assert!(accepted);
                    assert!(config.encoder_set);
                }
                Err(status) => {
                    assert!(!accepted);
                    assert_eq!(status, BundleStatusV1::Unsupported);
                }
            }
        }
    }

    #[test]
    fn accepts_declared_resource_geometry_and_rejects_bad_pages() {
        let bytes = configuration();
        let config = decode_configuration(&bytes).expect("declared geometry");
        assert_eq!(config.kv.page_bytes, GA_PAGE_BYTES_BF16);
        assert!(!config.kv.fp8());
        assert!(!config.media);
        assert!(
            decode_configuration(&configuration_with_media(4))
                .unwrap()
                .media
        );
        assert!(matches!(
            decode_configuration(&configuration_with_media(3)),
            Err(BundleStatusV1::Unsupported)
        ));
        let mut wrong = bytes.clone();
        let marker = GA_PAGE_BYTES_BF16.to_le_bytes();
        let offset = wrong
            .windows(8)
            .position(|window| window == marker)
            .expect("page field");
        wrong[offset] ^= 1;
        assert!(matches!(
            decode_configuration(&wrong),
            Err(BundleStatusV1::Unsupported)
        ));
        assert!(matches!(
            decode_configuration(&bytes[..bytes.len() - 1]),
            Err(BundleStatusV1::InvalidArgument)
        ));
    }
}
