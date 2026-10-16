//! The per-channel-decay delta rule engine (contract:
//! `kernel_core::per_channel_decay_delta_rule`): the q, k and v short
//! convolutions with F32 weights, the chunked prefill (prepare, then scan),
//! decode and verify of the recurrence, and the per-head gated RMSNorm of its
//! output (an instance of the gated delta rule's norm template). States live
//! in pools addressed by slot id; live row and sequence counts stay on the
//! device. See `cuda/linear/CHANNEL_DECAY.md` for the ABI and the
//! numerics.
//!
//! Declared exactness ([`Entry::invariance`], the same under every prefill
//! schedule, which store the same values in the same order):
//!
//! - The convolution adds each channel's taps in one FMA chain from zero,
//!   oldest first, its window from the launch's earlier rows or the slot;
//!   its state is the last raw inputs, taken before any store, and its
//!   anchor state (the window at the last multiple of 64 strictly inside
//!   the call) the same way. A prefill keeps every row and the state it
//!   leaves under any chunk split, sequences ignore each other, and it
//!   writes a spare entry at the anchor (`Anchors::Spare`).
//! - The delta-rule prefill chunks a sequence at absolute multiples of 64
//!   positions ([`CHUNK`]): a call that ends on one leaves exactly the
//!   state after that chunk, so rows and states keep their bits for chunk
//!   splits at multiples of 64 (`Chunk::Granule`), never others. The
//!   prepare writes workspace only; the scan also writes the state at the
//!   anchor into a spare entry.
//! - A verify call's rows and per-row checkpoints equal successive one-row
//!   decode calls of the same step (one template, `MaxRows` only choosing
//!   the checkpoint targets): one-row bits up to the capacity, the decode
//!   entry being its one-row entry, and sequences independent. Decode
//!   claims nothing. Both keep per-slot state without a spare.
//! - The gated norm is row-local (a CTA per row, a warp per head): one-row
//!   bits, any chunk split and row independence.

use std::path::PathBuf;

use super::{Bucket, CHUNK, MAX_CONV_TAPS, Stage, check_phase};
use crate::{Profile, ProfileNames};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, GateActivation, InstantiationSpec,
    LaunchPlan, NumericalContract, OpFamily, RecurrentPhase, RecurrentQuery, Refusal, StaticQuery,
    Target, Validate,
};
use kernel_core::per_channel_decay_delta_rule::{
    DELTA_RULE_CHUNK, DecayGate, PerChannelDecayDeltaRuleQuery, QkvConvs,
};
use kernel_core::properties::{Anchors, Chunk, Invariance};

const _: () = assert!(DELTA_RULE_CHUNK == CHUNK);

/// Sequences one prefill launch may carry.
pub const PREFILL_SEQUENCES: u32 = 256;
/// Value rows of one head per scan CTA.
pub const SCAN_ROWS: u32 = 16;
/// Value rows of one head per decode or verify CTA.
pub const RECURRENT_ROWS: u32 = 16;
/// The head dimension the CUDA entries compile; 64 is declared by the
/// contract and covered by the oracle only.
pub const HEAD_DIM: u32 = 128;
/// F32 workspace elements per (prefill chunk, head) at head dim 128: W, U,
/// QD and KD `[64, 128]`, P `[64, 64]` and EGL `[128]`.
pub const WORK_FLOATS: u64 = 4 * 64 * 128 + 64 * 64 + 128;
/// F32 row stride of the shared tiles, one bank apart per row.
const KEY_STRIDE: u32 = 129;
const TRIANGLE_STRIDE: u32 = 65;
/// Dynamic shared bytes of the tiled stages
/// (`channel_decay_prefill_tiled.cuh`'s `PrepareTiles` and `ScanTiles`):
/// K, G, beta and Q (then A and T transposed at a row stride of 68), and two
/// operand tiles, P, the U slice, EGL, the state slice and V' transposed.
const TILED_PREPARE_SHARED: u32 =
    (2 * CHUNK * HEAD_DIM + CHUNK + CHUNK * CHUNK + CHUNK * (CHUNK + 4)) * 4;
const TILED_SCAN_SHARED: u32 = (2 * CHUNK * HEAD_DIM
    + CHUNK * CHUNK
    + CHUNK * SCAN_ROWS
    + HEAD_DIM
    + SCAN_ROWS * HEAD_DIM
    + SCAN_ROWS * CHUNK)
    * 4;
/// Dynamic shared bytes of the streamed stages
/// (`channel_decay_prefill_streamed.cuh`'s `PrepareTiles` and `ScanTiles`):
/// K, G, beta and Q (then A and T transposed, both at a row stride of 68),
/// and four half tiles, P, the U slice, EGL, the state slice and V'
/// transposed.
const STREAMED_PREPARE_SHARED: u32 = (2 * CHUNK * HEAD_DIM + CHUNK + 2 * CHUNK * (CHUNK + 4)) * 4;
const STREAMED_SCAN_SHARED: u32 = (4 * CHUNK * CHUNK
    + CHUNK * CHUNK
    + CHUNK * SCAN_ROWS
    + HEAD_DIM
    + SCAN_ROWS * HEAD_DIM
    + SCAN_ROWS * CHUNK)
    * 4;
/// SM120's dynamic shared memory limit per block.
pub const SHARED_LIMIT: u32 = 101_376;

const CONV_THREADS: u32 = 128;
const RECURRENT_THREADS: u32 = 128;
const PREFILL_THREADS: u32 = 256;
const STREAMED_SCAN_THREADS: u32 = 128;
const NORM_THREADS: u32 = 128;

/// The ops of the engine: the contract's convolution and recurrence, and
/// the gated RMSNorm of the output.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Query {
    Rule(PerChannelDecayDeltaRuleQuery),
    Norm(RecurrentQuery),
}

/// The schedule of a delta-rule prefill stage. Both compute every stored
/// value with the same F32 operations in the same order (the same
/// workspace, outputs and states bit for bit); they differ in how the work
/// is spread over threads.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum PrefillSchedule {
    /// Scalar chains over operands staged in shared memory
    /// (`channel_decay_prefill.cuh`); every non-prefill entry has it.
    #[default]
    Staged,
    /// Register tiles, swizzled shared tiles and cp.async streaming
    /// (`channel_decay_prefill_tiled.cuh`).
    Tiled,
    /// The tiled prepare with its latency-bound phases spread over every
    /// warp, and a four-warp scan that streams half tiles and shares each
    /// state load between V' and the inter-chunk output
    /// (`channel_decay_prefill_streamed.cuh`).
    Streamed,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub symbol: String,
    pub query: Query,
    /// `Prepare` then `Scan` for the delta-rule prefill, `Single` otherwise.
    pub stage: Stage,
    /// `Tiled` or `Streamed` only for the prepare and scan stages of
    /// `tiled_entries` and `streamed_entries`.
    pub schedule: PrefillSchedule,
    pub params: Vec<EntryParam>,
}

/// Prefill chunks of a bucket for chunks that start at absolute multiples
/// of 64: a sequence of `r` rows at phase `p` has `ceil((p + r) / 64)`
/// chunks, at most `ceil(r / 64) + 1`, so the bucket's rows over 64 plus two
/// per sequence bound them.
pub fn aligned_chunks(bucket: Bucket) -> u32 {
    bucket.rows.div_ceil(CHUNK) + 2 * bucket.sequences
}

fn phase(query: Query) -> Option<RecurrentPhase> {
    match query {
        Query::Rule(
            PerChannelDecayDeltaRuleQuery::ShortConv { phase, .. }
            | PerChannelDecayDeltaRuleQuery::DeltaRule { phase, .. },
        ) => Some(phase),
        Query::Norm(_) => None,
    }
}

/// Validates `query` against its contract, then refuses a verify phase of
/// more than `MAX_VERIFY_ROWS` rows, the limit both engines share.
pub fn limits(query: PerChannelDecayDeltaRuleQuery) -> Result<(), Refusal> {
    query.validate()?;
    match query {
        PerChannelDecayDeltaRuleQuery::ShortConv { phase, .. }
        | PerChannelDecayDeltaRuleQuery::DeltaRule { phase, .. } => check_phase(phase),
    }
}

fn capacity(query: Query) -> u32 {
    match query {
        Query::Rule(
            PerChannelDecayDeltaRuleQuery::ShortConv { capacity, .. }
            | PerChannelDecayDeltaRuleQuery::DeltaRule { capacity, .. },
        ) => capacity,
        Query::Norm(
            RecurrentQuery::GatedRmsNorm { capacity, .. }
            | RecurrentQuery::CausalConv1d { capacity, .. }
            | RecurrentQuery::GatedDeltaRule { capacity, .. },
        ) => capacity,
    }
}

impl Entry {
    pub fn threads(&self) -> u32 {
        match (self.query, self.stage) {
            (Query::Rule(PerChannelDecayDeltaRuleQuery::ShortConv { .. }), _) => CONV_THREADS,
            (Query::Rule(PerChannelDecayDeltaRuleQuery::DeltaRule { .. }), Stage::Single) => {
                RECURRENT_THREADS
            }
            (Query::Rule(_), Stage::Scan) if self.schedule == PrefillSchedule::Streamed => {
                STREAMED_SCAN_THREADS
            }
            (Query::Rule(_), _) => PREFILL_THREADS,
            (Query::Norm(_), _) => NORM_THREADS,
        }
    }

    /// The CUDA refusals on top of the contract's: head dim 128 only (64 is
    /// declared; 128 also gives the convolution whole channel pairs per q,
    /// k and v part), convolutions of at most `MAX_CONV_TAPS` taps, verify
    /// launches of at most `MAX_VERIFY_ROWS` rows (`limits`), and the
    /// prepare and scan stages belong to the delta-rule prefill alone.
    fn check(&self) -> Result<(), Refusal> {
        match self.query {
            Query::Rule(query) => {
                limits(query)?;
                let (head_dim, prefill_rule) = match query {
                    PerChannelDecayDeltaRuleQuery::ShortConv {
                        head_dim, kernel, ..
                    } => {
                        if kernel > MAX_CONV_TAPS {
                            return Err(Refusal(
                                "per-channel-decay short convolution CUDA entries take at most 8 taps",
                            ));
                        }
                        (head_dim, false)
                    }
                    PerChannelDecayDeltaRuleQuery::DeltaRule {
                        head_dim, phase, ..
                    } => (head_dim, phase == RecurrentPhase::Prefill),
                };
                if head_dim != HEAD_DIM {
                    return Err(Refusal(
                        "per-channel-decay delta rule CUDA entries need head dim 128",
                    ));
                }
                if prefill_rule == (self.stage == Stage::Single) {
                    return Err(Refusal(
                        "the delta-rule prefill runs as prepare and scan stages",
                    ));
                }
                if !prefill_rule {
                    self.check_staged()?;
                }
            }
            Query::Norm(_) if self.schedule != PrefillSchedule::Staged => {
                self.check_staged()?;
            }
            Query::Norm(query) => {
                query.validate()?;
                let RecurrentQuery::GatedRmsNorm { head_dim, .. } = query else {
                    return Err(Refusal("the engine's norm is the gated RMSNorm"));
                };
                if head_dim != HEAD_DIM || self.stage != Stage::Single {
                    return Err(Refusal("the gated RMSNorm entry needs head dim 128"));
                }
            }
        }
        Ok(())
    }

    /// Refuses a schedule other than `Staged`, which only the delta-rule
    /// prefill stages have.
    fn check_staged(&self) -> Result<(), Refusal> {
        match self.schedule {
            PrefillSchedule::Staged => Ok(()),
            PrefillSchedule::Tiled => Err(Refusal(
                "only the delta-rule prefill stages have a tiled schedule",
            )),
            PrefillSchedule::Streamed => Err(Refusal(
                "only the delta-rule prefill stages have a streamed schedule",
            )),
        }
    }

    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.check()?;
        let (query, family) = match self.query {
            Query::Rule(query) => (
                StaticQuery::PerChannelDecayDeltaRule(query),
                OpFamily::Recurrent,
            ),
            Query::Norm(query) => (StaticQuery::Recurrent(query), OpFamily::Row),
        };
        let (tile_m, tile_n) = match self.stage {
            Stage::Single => (1, 1),
            Stage::Prepare => (CHUNK as u16, CHUNK as u16),
            Stage::Scan => (CHUNK as u16, SCAN_ROWS as u16),
        };
        let spec = InstantiationSpec {
            family,
            key: CompileKey {
                query,
                knobs: CompileKnobs {
                    tile_m,
                    tile_n,
                    tile_k: 1,
                    stages: match (self.schedule, self.stage) {
                        (PrefillSchedule::Tiled, Stage::Scan) => 2,
                        (PrefillSchedule::Streamed, Stage::Scan) => 4,
                        _ => 1,
                    },
                    swizzle: match self.schedule {
                        PrefillSchedule::Staged => 0,
                        PrefillSchedule::Tiled => 1,
                        PrefillSchedule::Streamed => 2,
                    },
                    warps: (self.threads() / 32) as u8,
                    split_k_capacity: 1,
                },
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol.clone(),
        };
        spec.validate()?;
        match self.query {
            Query::Rule(query) => query.numerical_contract_id()?,
            Query::Norm(query) => query.numerical_contract_id()?,
        };
        Ok(spec)
    }

    /// The entry's declared exactness (see the module docs). Nothing for an
    /// entry its kernel refuses.
    pub fn invariance(&self) -> Invariance {
        if self.contract().is_err() {
            return Invariance::NONE;
        }
        let rows = self.capacity().rows;
        let rule = |query| match query {
            PerChannelDecayDeltaRuleQuery::ShortConv { phase, .. }
            | PerChannelDecayDeltaRuleQuery::DeltaRule { phase, .. } => phase,
        };
        match self.query {
            Query::Norm(_) => Invariance::any_rows(rows),
            Query::Rule(query) => match (rule(query), self.stage) {
                (RecurrentPhase::Prefill, Stage::Single) => {
                    Invariance::PREFILL.with_anchors(Anchors::Spare)
                }
                (RecurrentPhase::Prefill, Stage::Prepare) => {
                    Invariance::PREFILL.with_chunk(Chunk::Granule(CHUNK))
                }
                (RecurrentPhase::Prefill, Stage::Scan) => Invariance::PREFILL
                    .with_chunk(Chunk::Granule(CHUNK))
                    .with_anchors(Anchors::Spare),
                (RecurrentPhase::Verify { .. }, _) => {
                    Invariance::one_row(rows).with_anchors(Anchors::None)
                }
                (RecurrentPhase::Decode, _) => Invariance::NONE.with_anchors(Anchors::None),
            },
        }
    }

    /// The catalog's prepared maximum: rows, and sequences (the row capacity
    /// for decode and the norm, rows over `max_rows` for verify,
    /// `PREFILL_SEQUENCES` for prefill).
    pub fn capacity(&self) -> Bucket {
        let rows = capacity(self.query);
        let sequences = match phase(self.query) {
            Some(RecurrentPhase::Prefill) => PREFILL_SEQUENCES,
            Some(RecurrentPhase::Decode) | None => rows,
            Some(RecurrentPhase::Verify { max_rows }) => rows / max_rows,
        };
        Bucket { rows, sequences }
    }

    /// Workspace bytes shared by the prefill stages of a bucket:
    /// `WORK_FLOATS` F32 per head of every aligned chunk.
    pub fn workspace_bytes(&self, bucket: Bucket) -> u64 {
        match (self.query, self.stage) {
            (
                Query::Rule(PerChannelDecayDeltaRuleQuery::DeltaRule { heads, .. }),
                Stage::Prepare | Stage::Scan,
            ) => u64::from(aligned_chunks(bucket)) * u64::from(heads) * WORK_FLOATS * 4,
            _ => 0,
        }
    }

    pub fn dynamic_shared_bytes(&self) -> u32 {
        match (self.schedule, self.stage) {
            (PrefillSchedule::Tiled, Stage::Prepare) => return TILED_PREPARE_SHARED,
            (PrefillSchedule::Tiled, Stage::Scan) => return TILED_SCAN_SHARED,
            (PrefillSchedule::Streamed, Stage::Prepare) => return STREAMED_PREPARE_SHARED,
            (PrefillSchedule::Streamed, Stage::Scan) => return STREAMED_SCAN_SHARED,
            _ => {}
        }
        match self.stage {
            Stage::Single => 0,
            // K, Q (then T) and G at the padded stride, and beta.
            Stage::Prepare => (3 * CHUNK * KEY_STRIDE + CHUNK) * 4,
            // The state slice, V', a staged operand tile, P and EGL.
            Stage::Scan => {
                (SCAN_ROWS * KEY_STRIDE
                    + CHUNK * SCAN_ROWS
                    + CHUNK * KEY_STRIDE
                    + CHUNK * TRIANGLE_STRIDE
                    + HEAD_DIM)
                    * 4
            }
        }
    }

    /// Launch plan for a prepared bucket at most the catalog capacity. The
    /// grid depends only on the bucket, never on live counts, which the
    /// kernels read on the device, so one plan serves every graph replay of
    /// the bucket; `live_rows` and `live_sequences` are only checked.
    pub fn plan(
        &self,
        bucket: Bucket,
        live_rows: u32,
        live_sequences: u32,
    ) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        let limit = self.capacity();
        if bucket.rows == 0
            || bucket.sequences == 0
            || bucket.rows > limit.rows
            || bucket.sequences > limit.sequences
        {
            return Err(Refusal("recurrent bucket exceeds the catalog capacity"));
        }
        if live_rows > bucket.rows || live_sequences > bucket.sequences {
            return Err(Refusal(
                "recurrent rows or sequences exceed the prepared bucket",
            ));
        }
        match phase(self.query) {
            Some(RecurrentPhase::Decode)
                if bucket.rows != bucket.sequences || live_rows != live_sequences =>
            {
                return Err(Refusal("decode takes one row per sequence"));
            }
            Some(RecurrentPhase::Verify { max_rows })
                if live_rows > live_sequences.saturating_mul(max_rows) =>
            {
                return Err(Refusal("verify rows exceed max_rows per sequence"));
            }
            _ => {}
        }
        let grid = match (self.query, self.stage) {
            (
                Query::Rule(PerChannelDecayDeltaRuleQuery::ShortConv {
                    heads, head_dim, ..
                }),
                _,
            ) => {
                let blocks = (3 * heads * head_dim).div_ceil(2 * CONV_THREADS);
                match phase(self.query) {
                    Some(RecurrentPhase::Prefill) => [blocks, bucket.chunks(), 1],
                    _ => [blocks, bucket.sequences, 1],
                }
            }
            (
                Query::Rule(PerChannelDecayDeltaRuleQuery::DeltaRule {
                    heads, head_dim, ..
                }),
                stage,
            ) => match stage {
                Stage::Single => [head_dim / RECURRENT_ROWS, heads, bucket.sequences],
                Stage::Prepare => [heads, aligned_chunks(bucket), 1],
                Stage::Scan => [head_dim / SCAN_ROWS, heads, bucket.sequences],
            },
            (Query::Norm(_), _) => [bucket.rows, 1, 1],
        };
        let plan = LaunchPlan {
            grid,
            block: [self.threads(), 1, 1],
            dynamic_shared_bytes: self.dynamic_shared_bytes(),
            workspace_bytes: self.workspace_bytes(bucket),
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        if values.len() != self.params.len() {
            return Err(Refusal(
                "per-channel-decay delta rule argument count mismatch",
            ));
        }
        for (param, value) in self.params.iter().zip(&values) {
            if !matches!(
                (param, value),
                (
                    EntryParam::DevicePointer { .. },
                    Argument::DeviceBuffer(_) | Argument::NullDevicePointer
                ) | (EntryParam::U32 { .. }, Argument::U32(_))
            ) {
                return Err(Refusal(
                    "per-channel-decay delta rule argument ABI kind mismatch",
                ));
            }
        }
        let args = ArgumentList(values);
        args.validate()?;
        Ok(args)
    }

    pub fn body(&self) -> String {
        match self.query {
            Query::Rule(PerChannelDecayDeltaRuleQuery::ShortConv {
                heads,
                head_dim,
                kernel,
                convs,
                phase,
                ..
            }) => {
                let t = format!(
                    "{},{kernel},{}",
                    3 * heads * head_dim,
                    convs == QkvConvs::Separate
                );
                let weights = "weight_q,weight_k,weight_v";
                let ns = "chop::linear::channel_decay";
                match phase {
                    RecurrentPhase::Prefill => format!(
                        "{ns}::short_conv_prefill<{t}>(input,input_stride,{weights},state,output,cu_seqlens,start_positions,slot_ids,initial,anchor_ids,live_sequences);"
                    ),
                    RecurrentPhase::Decode => format!(
                        "{ns}::short_conv_decode<{t}>(input,input_stride,{weights},state,output,slot_ids,live_sequences);"
                    ),
                    RecurrentPhase::Verify { max_rows } => format!(
                        "{ns}::short_conv_verify<{t},{max_rows}>(input,input_stride,{weights},state,output,cu_rows,slot_ids,checkpoint_ids,live_sequences);"
                    ),
                }
            }
            Query::Rule(PerChannelDecayDeltaRuleQuery::DeltaRule {
                heads,
                head_dim,
                gate,
                phase,
                ..
            }) => {
                let gate = match gate {
                    DecayGate::BoundedSigmoid { lower_bound_bits } => {
                        format!("chop::linear::channel_decay::kBoundedSigmoid,{lower_bound_bits}u")
                    }
                    DecayGate::Softplus => "chop::linear::channel_decay::kSoftplus,0u".into(),
                };
                let ns = "chop::linear::channel_decay";
                let operands = "qkv,qkv_stride,f,f_stride,b,b_stride,a_log,dt_bias";
                let (ns_stage, suffix) = match self.schedule {
                    PrefillSchedule::Staged => (ns.to_string(), ""),
                    PrefillSchedule::Tiled => (format!("{ns}::tiled"), "_tiled"),
                    PrefillSchedule::Streamed => (format!("{ns}::streamed"), "_streamed"),
                };
                match (phase, self.stage) {
                    (RecurrentPhase::Prefill, Stage::Prepare) => format!(
                        "{ns_stage}::channel_delta_prepare{suffix}<{heads},{head_dim},{gate}>({operands},workspace,cu_seqlens,start_positions,live_sequences);"
                    ),
                    (RecurrentPhase::Prefill, _) => format!(
                        "{ns_stage}::channel_delta_scan{suffix}<{heads},{head_dim}>(workspace,state,output,cu_seqlens,start_positions,slot_ids,initial,anchor_ids,live_sequences);"
                    ),
                    (RecurrentPhase::Decode, _) => format!(
                        "{ns}::channel_delta_recurrent<{heads},{head_dim},{gate},0>({operands},state,output,nullptr,slot_ids,nullptr,live_sequences);"
                    ),
                    (RecurrentPhase::Verify { max_rows }, _) => format!(
                        "{ns}::channel_delta_recurrent<{heads},{head_dim},{gate},{max_rows}>({operands},state,output,cu_rows,slot_ids,checkpoint_ids,live_sequences);"
                    ),
                }
            }
            Query::Norm(RecurrentQuery::GatedRmsNorm {
                width,
                head_dim,
                eps_bits,
                gate,
                ..
            }) => format!(
                "chop::linear::gated_rmsnorm<{width},{head_dim},{}>(input,gate,gate_stride,weight,output,live_rows,__uint_as_float({eps_bits}u));",
                gate == GateActivation::Sigmoid
            ),
            Query::Norm(_) => String::new(),
        }
    }
}

fn ptr(name: &str, ty: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}
fn u32_arg(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}

/// The short convolution ABI: BF16 input `[rows, input_stride]` (q, k, v
/// columns first), the F32 weights (`weight_q` holds the fused weight and
/// `weight_k` and `weight_v` are unused for fused convolutions), the BF16
/// state pool and output `[rows, 3 * heads * head_dim]`.
pub fn conv_params(phase: RecurrentPhase) -> Vec<EntryParam> {
    let mut params = vec![
        ptr("input", "const uint16_t"),
        u32_arg("input_stride"),
        ptr("weight_q", "const float"),
        ptr("weight_k", "const float"),
        ptr("weight_v", "const float"),
        ptr("state", "uint16_t"),
        ptr("output", "uint16_t"),
    ];
    match phase {
        RecurrentPhase::Prefill => params.extend([
            ptr("cu_seqlens", "const int32_t"),
            ptr("start_positions", "const int32_t"),
            ptr("slot_ids", "const int32_t"),
            ptr("initial", "const int32_t"),
            ptr("anchor_ids", "const int32_t"),
        ]),
        RecurrentPhase::Decode => params.push(ptr("slot_ids", "const int32_t")),
        RecurrentPhase::Verify { .. } => params.extend([
            ptr("cu_rows", "const int32_t"),
            ptr("slot_ids", "const int32_t"),
            ptr("checkpoint_ids", "const int32_t"),
        ]),
    }
    params.push(ptr("live_sequences", "const uint32_t"));
    params
}

/// The delta-rule ABI for a phase and stage: convolved BF16 `qkv`, the BF16
/// gate projection `f` `[rows, f_stride]`, the BF16 beta logits `b`
/// `[rows, b_stride]`, F32 `a_log` `[heads]` and `dt_bias` `[heads *
/// head_dim]`; the F32 state pool `[slots, heads, head_dim, head_dim]` and
/// the BF16 output `[rows, heads, head_dim]`.
pub fn delta_params(phase: RecurrentPhase, stage: Stage) -> Vec<EntryParam> {
    let operands = [
        ptr("qkv", "const uint16_t"),
        u32_arg("qkv_stride"),
        ptr("f", "const uint16_t"),
        u32_arg("f_stride"),
        ptr("b", "const uint16_t"),
        u32_arg("b_stride"),
        ptr("a_log", "const float"),
        ptr("dt_bias", "const float"),
    ];
    let mut params = Vec::new();
    match (phase, stage) {
        (RecurrentPhase::Prefill, Stage::Prepare) => {
            params.extend(operands);
            params.extend([
                ptr("workspace", "float"),
                ptr("cu_seqlens", "const int32_t"),
                ptr("start_positions", "const int32_t"),
            ]);
        }
        (RecurrentPhase::Prefill, _) => params.extend([
            ptr("workspace", "const float"),
            ptr("state", "float"),
            ptr("output", "uint16_t"),
            ptr("cu_seqlens", "const int32_t"),
            ptr("start_positions", "const int32_t"),
            ptr("slot_ids", "const int32_t"),
            ptr("initial", "const int32_t"),
            ptr("anchor_ids", "const int32_t"),
        ]),
        (RecurrentPhase::Decode, _) => {
            params.extend(operands);
            params.extend([
                ptr("state", "float"),
                ptr("output", "uint16_t"),
                ptr("slot_ids", "const int32_t"),
            ]);
        }
        (RecurrentPhase::Verify { .. }, _) => {
            params.extend(operands);
            params.extend([
                ptr("state", "float"),
                ptr("output", "uint16_t"),
                ptr("cu_rows", "const int32_t"),
                ptr("slot_ids", "const int32_t"),
                ptr("checkpoint_ids", "const int32_t"),
            ]);
        }
    }
    params.push(ptr("live_sequences", "const uint32_t"));
    params
}

/// The gated RMSNorm ABI: BF16 input `[rows, width]`, BF16 gate `[rows,
/// gate_stride]`, BF16 weight `[head_dim]`, BF16 output `[rows, width]`.
pub fn norm_params() -> Vec<EntryParam> {
    vec![
        ptr("input", "const uint16_t"),
        ptr("gate", "const uint16_t"),
        u32_arg("gate_stride"),
        ptr("weight", "const uint16_t"),
        ptr("output", "uint16_t"),
        ptr("live_rows", "const uint32_t"),
    ]
}

/// One layer shape of the engine and the capacities its entries are
/// prepared for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    /// Heads per rank.
    pub heads: u32,
    pub head_dim: u32,
    pub kernel: u32,
    pub convs: QkvConvs,
    pub gate: DecayGate,
    /// F32 bits of the gated RMSNorm's eps.
    pub norm_eps_bits: u32,
    pub prefill_rows: u32,
    pub decode_sequences: u32,
    pub verify_rows: u32,
    pub verify_sequences: u32,
}

/// Every entry of `geometry` under `profile`. Symbols are
/// `chop_channel_decay_{tag}_<entry>_v1`.
pub fn entries(profile: Profile, geometry: Geometry) -> Vec<Entry> {
    let Geometry {
        heads,
        head_dim,
        kernel,
        convs,
        gate,
        norm_eps_bits,
        prefill_rows,
        decode_sequences,
        verify_rows,
        verify_sequences,
    } = geometry;
    let symbol = |name: &str| format!("chop_channel_decay_{}_{name}_v1", profile.tag());
    let verify = RecurrentPhase::Verify {
        max_rows: verify_rows,
    };
    let phases = [
        ("prefill", RecurrentPhase::Prefill, prefill_rows),
        ("decode", RecurrentPhase::Decode, decode_sequences),
        ("verify", verify, verify_rows * verify_sequences),
    ];
    let mut list = Vec::new();
    for (name, phase, capacity) in phases {
        list.push(Entry {
            profile,
            symbol: symbol(&format!("conv_{name}")),
            query: Query::Rule(PerChannelDecayDeltaRuleQuery::ShortConv {
                heads,
                head_dim,
                kernel,
                convs,
                phase,
                capacity,
            }),
            stage: Stage::Single,
            schedule: PrefillSchedule::Staged,
            params: conv_params(phase),
        });
    }
    for (name, phase, capacity) in phases {
        let query = Query::Rule(PerChannelDecayDeltaRuleQuery::DeltaRule {
            heads,
            head_dim,
            gate,
            phase,
            capacity,
        });
        let stages: &[(&str, Stage)] = if phase == RecurrentPhase::Prefill {
            &[("_prepare", Stage::Prepare), ("_scan", Stage::Scan)]
        } else {
            &[("", Stage::Single)]
        };
        for &(suffix, stage) in stages {
            list.push(Entry {
                profile,
                symbol: symbol(&format!("delta_{name}{suffix}")),
                query,
                stage,
                schedule: PrefillSchedule::Staged,
                params: delta_params(phase, stage),
            });
        }
    }
    list.push(Entry {
        profile,
        symbol: symbol("gated_norm"),
        query: Query::Norm(RecurrentQuery::GatedRmsNorm {
            width: heads * head_dim,
            head_dim,
            eps_bits: norm_eps_bits,
            gate: GateActivation::Sigmoid,
            capacity: prefill_rows,
        }),
        stage: Stage::Single,
        schedule: PrefillSchedule::Staged,
        params: norm_params(),
    });
    list
}

/// The served shape at four ranks: 16 heads of 128, kernel 4, separate q, k
/// and v convolutions, the bounded gate at -5 and norm eps 1e-5. Capacities:
/// prefill 8192 rows, decode 256 sequences, verify 8 rows of 256 sequences.
pub const SERVED: Geometry = Geometry {
    heads: 16,
    head_dim: 128,
    kernel: 4,
    convs: QkvConvs::Separate,
    gate: DecayGate::BoundedSigmoid {
        lower_bound_bits: 0xc0a0_0000,
    },
    norm_eps_bits: 0x3727_c5ac,
    prefill_rows: 8192,
    decode_sequences: 256,
    verify_rows: 8,
    verify_sequences: 256,
};

/// The served shape's profile, named by its numbers.
pub const SERVED_PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "channel-decay-16x128",
    tag: "h16x128",
});

/// The engine's own catalog: the served shape's entries.
pub fn catalog() -> Vec<Entry> {
    entries(SERVED_PROFILE, SERVED)
}

/// The delta-rule prefill's prepare and scan under the tiled schedule for
/// `geometry` under `profile`, symbols
/// `chop_channel_decay_{tag}_delta_prefill_{prepare,scan}_tiled_v1`: the
/// same queries, ABI, grids and workspace as `entries`' staged pair, so
/// either stage pairs with either schedule of the other.
pub fn tiled_entries(profile: Profile, geometry: Geometry) -> Vec<Entry> {
    entries(profile, geometry)
        .into_iter()
        .filter(|entry| matches!(entry.stage, Stage::Prepare | Stage::Scan))
        .map(|entry| Entry {
            symbol: format!(
                "{}_tiled_v1",
                entry.symbol.strip_suffix("_v1").expect("versioned symbol")
            ),
            schedule: PrefillSchedule::Tiled,
            ..entry
        })
        .collect()
}

/// The engine's tiled catalog: the served shape's tiled prefill stages.
pub fn tiled_catalog() -> Vec<Entry> {
    tiled_entries(SERVED_PROFILE, SERVED)
}

/// The cubin of tiled `entries` (one profile's), named
/// `channel_decay_tiled_{tag}_sm120`. It includes only the tiled header, so
/// the staged cubin's sources and locks are unchanged.
pub fn tiled_cubin_spec(profile: Profile, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name: format!("channel_decay_tiled_{}_sm120", profile.tag()),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"linear/channel_decay_prefill_tiled.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .filter(|entry| entry.profile == profile && entry.schedule == PrefillSchedule::Tiled)
            .map(|entry| {
                EntrySpec::new(
                    entry.symbol.clone(),
                    entry.params.clone(),
                    entry.body(),
                    entry.threads(),
                    1,
                )
            })
            .collect(),
    }
}

/// The engine's tiled cubins: one for the served shape.
pub fn tiled_cubin_specs() -> Vec<CubinSpec> {
    vec![tiled_cubin_spec(SERVED_PROFILE, &tiled_catalog())]
}

/// The lab's mutants of the tiled stages and the define that compiles each
/// (`channel_decay_prefill_tiled.cuh`): a dependency wait after the first
/// input reads, streamed tiles read without waiting for their copies, V'
/// chains in descending x, forward-substitution chains one term short, and
/// the single pairs of rows 62 and 63 never formed. kernel-lab must catch
/// every one.
pub const TILED_MUTANTS: [(&str, &str); 5] = [
    ("late_wait", "CHOP_CHANNEL_DECAY_TILED_TEST_LATE_WAIT"),
    ("no_tile_wait", "CHOP_CHANNEL_DECAY_TILED_TEST_NO_TILE_WAIT"),
    (
        "reversed_chain",
        "CHOP_CHANNEL_DECAY_TILED_TEST_REVERSED_CHAIN",
    ),
    (
        "short_triangle",
        "CHOP_CHANNEL_DECAY_TILED_TEST_SHORT_TRIANGLE",
    ),
    ("no_singles", "CHOP_CHANNEL_DECAY_TILED_TEST_NO_SINGLES"),
];

/// A tiled entry's symbol in the cubin of `mutant`.
pub fn tiled_mutant_symbol(symbol: &str, mutant: &str) -> String {
    format!(
        "{}_{mutant}_v1",
        symbol.strip_suffix("_v1").expect("versioned symbol")
    )
}

/// The lab's tiled cubins: the served shape's, then one per
/// [`TILED_MUTANTS`] entry, `channel_decay_tiled_{mutant}_{tag}_sm120`, with
/// symbols from [`tiled_mutant_symbol`].
pub fn tiled_lab_cubin_specs() -> Vec<CubinSpec> {
    let mut specs = tiled_cubin_specs();
    for (mutant, define) in TILED_MUTANTS {
        let mut spec = tiled_cubin_spec(SERVED_PROFILE, &tiled_catalog());
        spec.name = format!(
            "channel_decay_tiled_{mutant}_{}_sm120",
            SERVED_PROFILE.tag()
        );
        for entry in &mut spec.entries {
            entry.symbol = tiled_mutant_symbol(&entry.symbol, mutant);
        }
        spec.defines.push((define.into(), "1".into()));
        specs.push(spec);
    }
    specs
}

/// The delta-rule prefill's prepare and scan under the streamed schedule
/// for `geometry` under `profile`, symbols
/// `chop_channel_decay_{tag}_delta_prefill_{prepare,scan}_streamed_v1`: the
/// staged pair's queries, ABI, grids and workspace (the scan runs 128
/// threads per CTA), so either stage pairs with any schedule of the other.
pub fn streamed_entries(profile: Profile, geometry: Geometry) -> Vec<Entry> {
    entries(profile, geometry)
        .into_iter()
        .filter(|entry| matches!(entry.stage, Stage::Prepare | Stage::Scan))
        .map(|entry| Entry {
            symbol: format!(
                "{}_streamed_v1",
                entry.symbol.strip_suffix("_v1").expect("versioned symbol")
            ),
            schedule: PrefillSchedule::Streamed,
            ..entry
        })
        .collect()
}

/// The engine's streamed catalog: the served shape's streamed prefill
/// stages.
pub fn streamed_catalog() -> Vec<Entry> {
    streamed_entries(SERVED_PROFILE, SERVED)
}

/// The cubin of streamed `entries` (one profile's), named
/// `channel_decay_streamed_{tag}_sm120`. It includes only the streamed
/// header (which includes the tiled one), so no existing cubin's sources or
/// locks change.
pub fn streamed_cubin_spec(profile: Profile, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name: format!("channel_decay_streamed_{}_sm120", profile.tag()),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"linear/channel_decay_prefill_streamed.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .filter(|entry| entry.profile == profile && entry.schedule == PrefillSchedule::Streamed)
            .map(|entry| {
                EntrySpec::new(
                    entry.symbol.clone(),
                    entry.params.clone(),
                    entry.body(),
                    entry.threads(),
                    1,
                )
            })
            .collect(),
    }
}

/// The engine's streamed cubins: one for the served shape.
pub fn streamed_cubin_specs() -> Vec<CubinSpec> {
    vec![streamed_cubin_spec(SERVED_PROFILE, &streamed_catalog())]
}

/// The lab's mutants of the streamed stages and the define that compiles
/// each (`channel_decay_prefill_streamed.cuh`): a dependency wait after the
/// first input reads, half tiles read without waiting for their copies, the
/// fused V' and output chains in descending key order, forward-substitution
/// chains one term short, a column's finished element taken from the wrong
/// row owner, and the single pairs of rows 62 and 63 never formed.
/// kernel-lab must catch every one.
pub const STREAMED_MUTANTS: [(&str, &str); 6] = [
    ("late_wait", "CHOP_CHANNEL_DECAY_STREAMED_TEST_LATE_WAIT"),
    (
        "no_tile_wait",
        "CHOP_CHANNEL_DECAY_STREAMED_TEST_NO_TILE_WAIT",
    ),
    (
        "reversed_chain",
        "CHOP_CHANNEL_DECAY_STREAMED_TEST_REVERSED_CHAIN",
    ),
    (
        "short_triangle",
        "CHOP_CHANNEL_DECAY_STREAMED_TEST_SHORT_TRIANGLE",
    ),
    (
        "wrong_owner",
        "CHOP_CHANNEL_DECAY_STREAMED_TEST_WRONG_OWNER",
    ),
    ("no_singles", "CHOP_CHANNEL_DECAY_STREAMED_TEST_NO_SINGLES"),
];

/// The lab's streamed cubins: the served shape's, then one per
/// [`STREAMED_MUTANTS`] entry, `channel_decay_streamed_{mutant}_{tag}_sm120`,
/// with symbols from [`tiled_mutant_symbol`].
pub fn streamed_lab_cubin_specs() -> Vec<CubinSpec> {
    let mut specs = streamed_cubin_specs();
    for (mutant, define) in STREAMED_MUTANTS {
        let mut spec = streamed_cubin_spec(SERVED_PROFILE, &streamed_catalog());
        spec.name = format!(
            "channel_decay_streamed_{mutant}_{}_sm120",
            SERVED_PROFILE.tag()
        );
        for entry in &mut spec.entries {
            entry.symbol = tiled_mutant_symbol(&entry.symbol, mutant);
        }
        spec.defines.push((define.into(), "1".into()));
        specs.push(spec);
    }
    specs
}

/// The cubin of `entries` (one profile's), named
/// `channel_decay_{tag}_sm120`.
pub fn cubin_spec(profile: Profile, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name: format!("channel_decay_{}_sm120", profile.tag()),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![
            "#include \"linear/channel_decay_conv.cuh\"".into(),
            "#include \"linear/channel_decay_delta.cuh\"".into(),
            "#include \"linear/channel_decay_prefill.cuh\"".into(),
            "#include \"linear/norm.cuh\"".into(),
        ],
        defines: vec![],
        entries: entries
            .iter()
            .filter(|entry| entry.profile == profile)
            .map(|entry| {
                EntrySpec::new(
                    entry.symbol.clone(),
                    entry.params.clone(),
                    entry.body(),
                    entry.threads(),
                    1,
                )
            })
            .collect(),
    }
}

/// The engine's cubins: one for the served shape.
pub fn cubin_specs() -> Vec<CubinSpec> {
    vec![cubin_spec(SERVED_PROFILE, &catalog())]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn served_entries_validate_and_plan() {
        assert_eq!(f32::from_bits(SERVED.norm_eps_bits), 1e-5);
        let entries = catalog();
        assert_eq!(entries.len(), 8);
        let mut grids = Vec::new();
        for entry in &entries {
            entry.contract().unwrap();
            let capacity = entry.capacity();
            let plan = entry
                .plan(capacity, capacity.rows, capacity.sequences)
                .unwrap();
            assert!(plan.pdl);
            assert!(plan.dynamic_shared_bytes <= SHARED_LIMIT);
            assert_eq!(plan.block[0], entry.threads());
            grids.push((entry.symbol.clone(), plan.grid, plan.workspace_bytes));
        }
        let prefill = Bucket {
            rows: 8192,
            sequences: 256,
        };
        let chunks = aligned_chunks(prefill);
        assert_eq!(chunks, 128 + 512);
        let workspace = u64::from(chunks) * 16 * 147_968;
        let tag = |name: &str| format!("chop_channel_decay_h16x128_{name}_v1");
        assert_eq!(
            grids,
            vec![
                (tag("conv_prefill"), [24, 128 + 256, 1], 0),
                (tag("conv_decode"), [24, 256, 1], 0),
                (tag("conv_verify"), [24, 256, 1], 0),
                (tag("delta_prefill_prepare"), [16, chunks, 1], workspace),
                (tag("delta_prefill_scan"), [8, 16, 256], workspace),
                (tag("delta_decode"), [8, 16, 256], 0),
                (tag("delta_verify"), [8, 16, 256], 0),
                (tag("gated_norm"), [8192, 1, 1], 0),
            ]
        );
        assert_eq!(entries[3].dynamic_shared_bytes(), 99_328);
        assert_eq!(entries[4].dynamic_shared_bytes(), 62_528);
    }

    #[test]
    fn tiled_entries_keep_the_staged_abi_and_plans() {
        let staged = catalog();
        let tiled = tiled_catalog();
        assert_eq!(
            tiled.iter().map(|e| e.symbol.as_str()).collect::<Vec<_>>(),
            [
                "chop_channel_decay_h16x128_delta_prefill_prepare_tiled_v1",
                "chop_channel_decay_h16x128_delta_prefill_scan_tiled_v1",
            ]
        );
        for (entry, base) in tiled.iter().zip(&staged[3..5]) {
            assert_eq!(entry.schedule, PrefillSchedule::Tiled);
            assert_eq!((entry.query, entry.stage), (base.query, base.stage));
            assert_eq!(entry.params, base.params);
            assert_ne!(entry.contract().unwrap().key, base.contract().unwrap().key);
            for bucket in [
                Bucket {
                    rows: 64,
                    sequences: 1,
                },
                Bucket {
                    rows: 2048,
                    sequences: 32,
                },
                entry.capacity(),
            ] {
                let plan = entry.plan(bucket, bucket.rows, bucket.sequences).unwrap();
                let base_plan = base.plan(bucket, bucket.rows, bucket.sequences).unwrap();
                assert_eq!(
                    (plan.grid, plan.block, plan.workspace_bytes, plan.pdl),
                    (
                        base_plan.grid,
                        base_plan.block,
                        base_plan.workspace_bytes,
                        base_plan.pdl
                    )
                );
                assert!(plan.dynamic_shared_bytes <= SHARED_LIMIT);
            }
        }
        assert_eq!(tiled[0].dynamic_shared_bytes(), 99_584);
        assert_eq!(tiled[1].dynamic_shared_bytes(), 98_816);
        let spec = tiled_cubin_spec(SERVED_PROFILE, &tiled);
        assert_eq!(spec.name, "channel_decay_tiled_h16x128_sm120");
        assert_eq!(spec.entries.len(), 2);
        let mut decode = staged[5].clone();
        decode.schedule = PrefillSchedule::Tiled;
        assert_eq!(
            decode.contract(),
            Err(Refusal(
                "only the delta-rule prefill stages have a tiled schedule"
            ))
        );
    }

    #[test]
    fn streamed_entries_keep_the_staged_abi_and_grids() {
        let staged = catalog();
        let streamed = streamed_catalog();
        assert_eq!(
            streamed
                .iter()
                .map(|e| e.symbol.as_str())
                .collect::<Vec<_>>(),
            [
                "chop_channel_decay_h16x128_delta_prefill_prepare_streamed_v1",
                "chop_channel_decay_h16x128_delta_prefill_scan_streamed_v1",
            ]
        );
        let tiled = tiled_catalog();
        for ((entry, base), other) in streamed.iter().zip(&staged[3..5]).zip(&tiled) {
            assert_eq!(entry.schedule, PrefillSchedule::Streamed);
            assert_eq!((entry.query, entry.stage), (base.query, base.stage));
            assert_eq!(entry.params, base.params);
            let key = entry.contract().unwrap().key;
            assert_ne!(key, base.contract().unwrap().key);
            assert_ne!(key, other.contract().unwrap().key);
            for bucket in [
                Bucket {
                    rows: 64,
                    sequences: 1,
                },
                Bucket {
                    rows: 2048,
                    sequences: 32,
                },
                entry.capacity(),
            ] {
                let plan = entry.plan(bucket, bucket.rows, bucket.sequences).unwrap();
                let base_plan = base.plan(bucket, bucket.rows, bucket.sequences).unwrap();
                assert_eq!(
                    (plan.grid, plan.workspace_bytes, plan.pdl),
                    (base_plan.grid, base_plan.workspace_bytes, base_plan.pdl)
                );
                assert_eq!(plan.block, [entry.threads(), 1, 1]);
                assert!(plan.dynamic_shared_bytes <= SHARED_LIMIT);
            }
        }
        assert_eq!((streamed[0].threads(), streamed[1].threads()), (256, 128));
        assert_eq!(streamed[0].dynamic_shared_bytes(), 100_608);
        assert_eq!(streamed[1].dynamic_shared_bytes(), 98_816);
        let spec = streamed_cubin_spec(SERVED_PROFILE, &streamed);
        assert_eq!(spec.name, "channel_decay_streamed_h16x128_sm120");
        assert_eq!(spec.entries.len(), 2);
        let lab = streamed_lab_cubin_specs();
        assert_eq!(lab.len(), 1 + STREAMED_MUTANTS.len());
        let mut decode = staged[5].clone();
        decode.schedule = PrefillSchedule::Streamed;
        assert_eq!(
            decode.contract(),
            Err(Refusal(
                "only the delta-rule prefill stages have a streamed schedule"
            ))
        );
    }

    #[test]
    fn unsupported_shapes_and_counts_refuse_by_name() {
        let narrow = entries(
            SERVED_PROFILE,
            Geometry {
                head_dim: 64,
                ..SERVED
            },
        );
        for entry in narrow.iter().filter(|e| matches!(e.query, Query::Rule(_))) {
            assert_eq!(
                entry.contract(),
                Err(Refusal(
                    "per-channel-decay delta rule CUDA entries need head dim 128"
                ))
            );
        }
        let mut entries = catalog();
        let decode = &entries[5];
        let bucket = Bucket {
            rows: 4,
            sequences: 4,
        };
        assert!(decode.plan(bucket, 4, 4).is_ok());
        assert_eq!(
            decode.plan(bucket, 3, 4),
            Err(Refusal("decode takes one row per sequence"))
        );
        let verify = &entries[6];
        assert_eq!(
            verify.plan(
                Bucket {
                    rows: 64,
                    sequences: 4
                },
                33,
                4
            ),
            Err(Refusal("verify rows exceed max_rows per sequence"))
        );
        assert_eq!(
            entries[0].plan(
                Bucket {
                    rows: 8193,
                    sequences: 1
                },
                1,
                1
            ),
            Err(Refusal("recurrent bucket exceeds the catalog capacity"))
        );
        entries[5].stage = Stage::Scan;
        assert_eq!(
            entries[5].contract(),
            Err(Refusal(
                "the delta-rule prefill runs as prepare and scan stages"
            ))
        );
    }
}
