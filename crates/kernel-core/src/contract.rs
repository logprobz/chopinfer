//! Exact, model-neutral operation and launch contracts.

use std::fmt;

use crate::context_parallel::ContextParallelQuery;
use crate::gated_mlp::GatedMlpProjectionQuery;
use crate::latent_attention::{HeadBatchedGemmQuery, LatentAttentionQuery};
use crate::per_channel_decay_delta_rule::PerChannelDecayDeltaRuleQuery;
use crate::pooled_indexer::PooledIndexerQuery;
use crate::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery;

/// Largest rank count of a collective, and the slot count of every pointer
/// array argument, which carries one address per rank. Crates that link no
/// kernel crate keep a copy that a dependent asserts equal:
/// `kernel_build::MAX_POINTER_ARRAY_SLOTS`,
/// `runtime_cuda::ffi::DEVICE_POINTER_ARRAY_SLOTS`,
/// `BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS` and the host's largest
/// device count (eight devices). The comm cubins receive it
/// as `CHOP_COMM_MAX_RANKS`. Placement arithmetic has its own, larger limit
/// (`topology::MAX_RANKS`).
pub const MAX_COMM_RANKS: usize = 8;

/// The only native target currently emitted by the kernel library.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Target {
    Sm120f,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum DType {
    Bf16,
    F16,
    F32,
    I32,
    I64,
    U8,
    Fp8E4M3,
    Fp8E5M2,
    Fp6E2M3,
    Fp6E3M2,
    Fp4E2M1,
    Ue8M0,
    Ue4M3,
}

impl DType {
    /// Safetensors element spelling, when this type has a byte-addressable source encoding.
    pub fn checkpoint_dtype(self) -> Option<&'static str> {
        Some(match self {
            Self::Bf16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::U8 => "U8",
            Self::Fp8E4M3 | Self::Ue4M3 => "F8_E4M3",
            Self::Fp8E5M2 => "F8_E5M2",
            Self::Ue8M0 => "F8_E8M0",
            _ => return None,
        })
    }
}

/// The reduction order is part of a normalization numerical contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(rename_all = "snake_case"))]
pub enum NormReduction {
    ShuffleDown,
    XorThenSerial,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum ScaleMode {
    /// F32 scale is the exact block maximum divided by 448.
    ExactF32,
    /// F32 scale is rounded upward to the next power of two.
    PowerOfTwoF32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum WeightFormat {
    Bf16,
    Fp8PerTensor,
    Fp8PerChannel,
    Fp8Block128x128(ScaleMode),
    /// FP8 E4M3 with one F32 scale per 64 output rows and 128 K values, for
    /// rows regrouped from 128-row blocks (for example a QKV projection whose
    /// heads are regrouped across ranks). Each row's scale is its source
    /// block's, so the product per element equals the block-scaled one.
    Fp8Rows64x128(ScaleMode),
    Mxfp8,
    /// E2M1 and UE8M0, one scale per 32 K values. Even K is the low nibble.
    Mxfp4,
    /// E2M1 and UE4M3 per 16 K values, plus one F32 global scale.
    Nvfp4,
    /// E4M3 with one multiplicative UE8M0 scale per 32 rows and 32 K.
    /// A lossless repack repeats each block scale for each of its rows.
    Fp8Block32x32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum ActivationFormat {
    Bf16,
    Fp8Group128(ScaleMode),
    Mxfp8,
    Nvfp4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum OutputQuantization {
    Fp8Group128(ScaleMode),
    Mxfp8,
    Mxfp4,
    Nvfp4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum SwigluColumns {
    ConcatenatedHalves,
    Interleaved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum GeluMode {
    Erf,
    Tanh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct Epilogues {
    pub residual_add: bool,
    /// Add one value per output column before activation and residual.
    pub bias: bool,
    pub gelu: Option<GeluMode>,
    pub swiglu: Option<SwigluColumns>,
    pub output_quantization: Option<OutputQuantization>,
    pub weighted_moe_combine: bool,
}

/// Static contracts for encoder kernels. These entries are contract-only until
/// the corresponding CUDA engines qualify them on hardware. Every field is
/// math the model states except `capacity` (a GEMM's `m_capacity`), the
/// rows a backend prepares for (`crate::multimodal`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MultimodalQuery {
    AudioRopeBf16 {
        capacity: u32,
    },
    EmbeddingSumBf16 {
        tables: u32,
        vocab: u32,
        width: u32,
        capacity: u32,
    },
    ResidualAdd {
        width: u32,
        capacity: u32,
    },
    Bf16ToF32 {
        width: u32,
        capacity: u32,
    },
    GeluRows {
        width: u32,
        capacity: u32,
    },
    SplitQkv {
        q_width: u32,
        kv_width: u32,
        capacity: u32,
    },
    ContiguousAttention {
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        causal: bool,
        window_left: Option<u32>,
        window_right: Option<u32>,
        key0_sink: bool,
        capacity: u32,
    },
    LayerNorm {
        width: u32,
        eps_bits: u32,
        capacity: u32,
    },
    Gemm(GemmQuery),
    GemmBf16(GemmQuery),
    ScratchRope {
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        style: RotaryStyle,
        capacity: u32,
    },
    Permute {
        width: u32,
        capacity: u32,
    },
    Im2col1d {
        channels: u32,
        kernel: u32,
        stride: u32,
        pad_left: u32,
        pad_right: u32,
        input: DType,
        capacity: u32,
    },
    EmbeddingSum {
        tables: u32,
        vocab: u32,
        width: u32,
        capacity: u32,
    },
    EmbeddingOverride {
        width: u32,
        capacity: u32,
    },
    /// Learned position table resampled per image: each F32 hidden row adds
    /// the F32 weighted sum of four table rows, with F32 output. Indices
    /// `[rows,4]` i32 and weights `[rows,4]` F32 come from the host; the
    /// table is `[table_rows,width]` BF16.
    InterpolatedPositionAddF32 {
        width: u32,
        table_rows: u32,
        capacity: u32,
    },
    /// `LayerNorm` over F32 rows, with F32 output.
    LayerNormF32 {
        width: u32,
        eps_bits: u32,
        capacity: u32,
    },
    /// In place over packed F32 rows `[T, (q_heads + 2 * kv_heads) *
    /// head_dim]` (q, k, v): rotates q and k as `ScratchRope` does, then
    /// rewrites the row as BF16 pairs `hi = bf16(x)`, `lo = bf16(x - hi)`
    /// in six planes of the same total width: q hi, q lo, k hi, k lo, v hi,
    /// v lo.
    RopeQkvPairs {
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        style: RotaryStyle,
        capacity: u32,
    },
    /// `ContiguousAttention` over the BF16 pair rows `RopeQkvPairs` writes,
    /// with F32 output `[T,q_heads,head_dim]`. Scores sum q hi k hi, q hi
    /// k lo and q lo k hi; every BF16 probability term (at least two, three
    /// below 128 keys) multiplies both halves of v.
    ContiguousAttentionPairs {
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        causal: bool,
        window_left: Option<u32>,
        window_right: Option<u32>,
        key0_sink: bool,
        capacity: u32,
    },
    /// Encoder GEMM over F32 activations: each value is split into a BF16
    /// pair `hi + lo` (as `RopeQkvPairs`), and both halves multiply the BF16
    /// weights with F32 accumulation. The output is F32 or BF16; a residual
    /// adds the F32 output row in place.
    GemmPairs(GemmQuery),
    /// RMSNorm over F32 rows with a BF16 weight, F32 output: `x * rstd *
    /// weight`, with `rstd` the correctly rounded F32 reciprocal square root
    /// of the F32 mean square plus eps.
    RmsNormF32 {
        width: u32,
        eps_bits: u32,
        capacity: u32,
    },
    /// `Permute` over F32 rows.
    PermuteF32 {
        width: u32,
        capacity: u32,
    },
    /// SwiGLU over F32 rows `[T, 2 * hidden]`, the gate half first:
    /// `silu(gate) * up` in F32, F32 output `[T, hidden]`.
    SwigluF32 {
        hidden: u32,
        capacity: u32,
    },
    /// F32 residual add: `output = input + residual`, one F32 rounding.
    ResidualAddF32 {
        width: u32,
        capacity: u32,
    },
    /// `Im2col1d` of F32 frames to F32 rows, without rounding.
    Im2col1dF32 {
        channels: u32,
        kernel: u32,
        stride: u32,
        pad_left: u32,
        pad_right: u32,
        capacity: u32,
    },
    /// In place over packed F32 rows `[T, (q_heads + 2 kv_heads) head_dim]`:
    /// each q and k head becomes `x * rstd * weight` (BF16 weights
    /// `[head_dim]`, one for q heads and one for k heads), with `rstd` the
    /// correctly rounded F32 reciprocal square root of the head's F32 mean
    /// square plus eps; v passes through.
    QkRmsNormF32 {
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        eps_bits: u32,
        capacity: u32,
    },
    /// Clamped SwiGLU over F32 rows `[T, 2 * hidden]`, the gate half first:
    /// `silu(min(gate, limit)) * clamp(up, -limit, limit)` in F32 (a NaN
    /// stays NaN), F32 output `[T, hidden]`.
    SwigluClampedF32 {
        hidden: u32,
        limit_bits: u32,
        capacity: u32,
    },
    /// Exact-erf GELU of F32 rows `[T, width]`, F32 output, in place or not,
    /// as the encoder GEMMs' erf epilogue computes it in F32 (`0.5 x (1 +
    /// erf(x / sqrt 2))`, through erfc below -1 so negative tails stay
    /// representable).
    GeluRowsF32 {
        width: u32,
        capacity: u32,
    },
    /// F32 rows `[T, input]` copied unchanged into rows `[T, output]`, zero
    /// past `input`, so a GEMM whose K is a multiple of its tile reads them.
    PadColumnsF32 {
        input: u32,
        output: u32,
        capacity: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ConvActivation {
    Identity,
    Silu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GateActivation {
    Sigmoid,
    Silu,
}

/// How a recurrent op advances the state of one sequence slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RecurrentPhase {
    /// Any number of rows; the slot holds the state after the last row.
    Prefill,
    /// One row per sequence.
    Decode,
    /// Up to `max_rows` rows after a committed prefix. The state after every
    /// row goes to its own checkpoint, and commit keeps row `kept - 1`.
    Verify { max_rows: u32 },
}

/// Per-sequence recurrent contracts. States live in slots the
/// runtime assigns to requests, never in batch order (`recurrent`, "State",
/// for the buffer roles and the anchor and checkpoint rules).
///
/// Every field is model math: shapes, activations, the phase (verify's
/// `max_rows` sizes its checkpoints) and the capacity, the most rows one
/// launch carries. Nothing here is a backend choice: chunk lengths, tiles
/// and thread counts belong to the backend, which reports the chunk granule
/// and anchors it keeps as guarantees. Head dims a backend cannot run are
/// its refusals, not the contract's (the convolution's 8-tap bound is still
/// here, see `validate`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RecurrentQuery {
    /// Depthwise causal convolution without bias. Tap 0 multiplies the oldest
    /// input, `(kernel - 1) * dilation` rows back. The slot state holds that
    /// many previous inputs as [state_rows, channels], oldest first.
    CausalConv1d {
        channels: u32,
        kernel: u32,
        dilation: u32,
        activation: ConvActivation,
        phase: RecurrentPhase,
        capacity: u32,
    },
    /// Gated delta rule with q/k L2 norm (eps 1e-6), the 1/sqrt(key_dim) q
    /// scale, beta = sigmoid(b) and decay g = -exp(A_log) * softplus(a +
    /// dt_bias) fused in. Value head h reads key head h / (value_heads /
    /// key_heads). The F32 slot state is [value_heads, value_dim, key_dim].
    GatedDeltaRule {
        key_heads: u32,
        value_heads: u32,
        key_dim: u32,
        value_dim: u32,
        phase: RecurrentPhase,
        capacity: u32,
    },
    /// Per-head RMSNorm with a plain weight shared by all heads, times
    /// act(gate). The normalized value is not rounded before the weight.
    GatedRmsNorm {
        width: u32,
        head_dim: u32,
        eps_bits: u32,
        gate: GateActivation,
        capacity: u32,
    },
}

impl RecurrentQuery {
    /// Elements of one sequence slot's state (F32 for the delta rule, the
    /// activation dtype for the convolution). Norms keep no state.
    pub fn state_elements(&self) -> u64 {
        match *self {
            Self::CausalConv1d {
                channels,
                kernel,
                dilation,
                ..
            } => u64::from(kernel.saturating_sub(1)) * u64::from(dilation) * u64::from(channels),
            Self::GatedDeltaRule {
                value_heads,
                key_dim,
                value_dim,
                ..
            } => u64::from(value_heads) * u64::from(value_dim) * u64::from(key_dim),
            Self::GatedRmsNorm { .. } => 0,
        }
    }

    /// The state's elements, or `None` where they overflow a u64.
    fn checked_state_elements(&self) -> Option<u64> {
        let product = |factors: [u64; 3]| {
            factors
                .into_iter()
                .try_fold(1_u64, |total, factor| total.checked_mul(factor))
        };
        match *self {
            Self::CausalConv1d {
                channels,
                kernel,
                dilation,
                ..
            } => product([
                u64::from(kernel.saturating_sub(1)),
                u64::from(dilation),
                u64::from(channels),
            ]),
            Self::GatedDeltaRule {
                value_heads,
                key_dim,
                value_dim,
                ..
            } => product([
                u64::from(value_heads),
                u64::from(value_dim),
                u64::from(key_dim),
            ]),
            Self::GatedRmsNorm { .. } => Some(0),
        }
    }
}

/// Gated attention row contracts: Gemma per-head Q/K norm,
/// interleaved M-RoPE and the sigmoid output gate. The attention itself is an
/// `AttentionQuery`; group size is any divisor, such as 24 / 4 ranks / 1.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GatedAttentionQuery {
    /// Each Q row is [q_heads, 2 * head_dim] with a head's query followed by
    /// its gate. Q and K get `x * rsqrt(mean(x^2) + eps) * (1 + w)` per head,
    /// then NeoX rotate-half RoPE over the first `rotary_dim` dimensions with
    /// interleaved M-RoPE: pair i takes the H position when i % 3 == 1 and
    /// i < 3 * `mrope[1]`, the W position when i % 3 == 2 and i < 3 * `mrope[2]`,
    /// and the T position otherwise. Rotated K and raw V go through the page
    /// table; Q and the gate stay in scratch.
    QkNormRopeKvWrite {
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        theta_bits: u32,
        mrope: [u32; 3],
        eps_bits: u32,
        page_size: u32,
        kv_dtype: DType,
        capacity: u32,
    },
    /// `attention * sigmoid(gate)`, element by element over [heads, head_dim].
    OutputGate {
        heads: u32,
        head_dim: u32,
        capacity: u32,
    },
}

/// Block-sparse attention with a learned indexer. Tokens form
/// complete blocks of `block` from position 0; a row sees the blocks and the
/// incomplete tail at or before its own position. The CUDA engines, their
/// schedules and selection precision are described in
/// crates/kernels-sm120/cuda/attention/SPARSE.md.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SparseAttentionQuery {
    /// Indexer queries: per-head Gemma RMSNorm (`x * rsqrt(mean(x^2) + eps)
    /// * (1 + w)`), then the attention's NeoX interleaved M-RoPE over the
    /// first `rotary_dim` dimensions. Other norms or rotations are other ops.
    IndexerQueryGemmaNormMrope {
        heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        theta_bits: u32,
        mrope: [u32; 3],
        eps_bits: u32,
        capacity: u32,
    },
    /// A raw key and its RoPE positions go to ring row `position % ring`. A
    /// row that completes a block also writes the block key to row
    /// `position / block` of the block-key cache: the mean of the block's raw
    /// keys rounded to BF16 (as both references do), Gemma RMSNorm, then RoPE
    /// at the block's first token.
    IndexerKeyWrite {
        head_dim: u32,
        rotary_dim: u32,
        theta_bits: u32,
        mrope: [u32; 3],
        eps_bits: u32,
        block: u32,
        ring: u32,
        capacity: u32,
    },
    /// `sum_h relu(q_h . key_b) / sqrt(head_dim)` for every complete visible
    /// block b of each row.
    BlockScores {
        heads: u32,
        head_dim: u32,
        max_blocks: u32,
        capacity: u32,
    },
    /// Keep the `top_blocks` highest scores, ties to the lower block index.
    /// The row's key set lists the kept blocks' tokens in ascending order,
    /// then the incomplete tail. With at most `top_blocks` complete blocks
    /// every visible token is kept, so rows up to `dense_tokens()` are dense.
    BlockTopK {
        block: u32,
        top_blocks: u32,
        max_blocks: u32,
        capacity: u32,
    },
    /// Softmax attention of every query head over its row's key set, which
    /// all heads share, from the paged KV cache. No sink, window or softcap.
    SparseAttention {
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block: u32,
        top_blocks: u32,
        softmax_scale_bits: u32,
        page_size: u32,
        kv_dtype: DType,
        phase: AttentionPhase,
        capacity: u32,
    },
}

impl SparseAttentionQuery {
    /// The longest visible length whose key set is every visible token.
    pub fn dense_tokens(&self) -> Option<u32> {
        match *self {
            Self::BlockTopK {
                block, top_blocks, ..
            }
            | Self::SparseAttention {
                block, top_blocks, ..
            } => Some(top_blocks * block + block - 1),
            _ => None,
        }
    }
}

/// Expert-sharded MoE rows. Routing itself is
/// `RowQuery::Router` and the experts are grouped `GemmQuery` entries.
///
/// `experts`, `local_experts`, `top_k`, `width` and `output` are math;
/// `capacity` is the most token rows one launch serves, a sizing fact
/// (`backend::Need`). The padding of each expert's routes is the backend's
/// choice. Buffer roles are in `moe`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MoeQuery {
    /// Keep the routes whose global expert lies in this rank's range
    /// `[rank * local_experts, (rank + 1) * local_experts)`, renumbered from
    /// zero, in token then top-k order. The rank is a runtime scalar. With
    /// replicated activations the ranks' routed outputs sum to the layer's.
    /// With every expert local (`local_experts == experts`, rank zero) this
    /// is the permutation of all routes by expert (`RowQuery::ExpertPermute`).
    LocalRoutes {
        experts: u32,
        local_experts: u32,
        top_k: u32,
        capacity: u32,
    },
    /// One rank's MoE output partial per token: the token's routed BF16 rows
    /// (one per route, already weighted) whose expert is local to the rank
    /// (by its expert ids, as `LocalRoutes` keeps them) added in top-k order
    /// in F32, plus `sigmoid(gate_logit) * shared` with the BF16 shared
    /// expert output, rounded to BF16 once. The gate logit is the dot
    /// product of the token's BF16 input row and the BF16 shared expert gate
    /// weight row, both `width` wide. Linear in the routed and shared
    /// outputs, so each rank's partial (of its routes and its K-sharded
    /// shared output) sums over the ranks to the layer's output.
    SharedExpertGateCombine { width: u32, capacity: u32 },
    /// One rank's routed output partial per token: the token's routed F32
    /// rows of `width` (one per route, already weighted) whose global expert
    /// lies in the rank's range (as `LocalRoutes` keeps them) added in top-k
    /// order in F32 from +0, then written as F32 or rounded to BF16 once
    /// (`output`). A token with no local route gets zeros. With replicated
    /// activations the ranks' partials sum to the layer's routed output.
    /// With an addend (`moe::RoutedSumBuffers::addend`, F32 output only),
    /// each token's F32 sum then adds the token's F32 addend row of `width`
    /// once, `sum + addend` rounded to nearest even, and is written as F32;
    /// a token with no local route gets `+0 + addend`.
    LocalRoutedSum {
        experts: u32,
        local_experts: u32,
        top_k: u32,
        width: u32,
        output: DType,
        capacity: u32,
    },
}

/// Sigmoid-mean hyper-connection residual rows with a low-rank mixer (a
/// Sinkhorn-mixed hyper-connection is a different op). The residual X holds
/// `streams` copies of width H per token. A sublayer reads
/// `mean_s(sigmoid(up_s) * Xn_s)` with `Xn` the grouped Gemma norm of X and
/// `up = W_up(silu(W_down(Xn) / streams))`; afterwards
/// `X_s += 2 * sigmoid(inject_s / streams) * y` with `inject = W_inject(Xn)`.
/// The projections are dense GEMMs.
///
/// `streams`, `width`, `rank`, `eps_bits` and `ranks` are math; `capacity`
/// is the most rows one launch serves, a sizing fact (`backend::Need`), not
/// math. Buffer roles are in `rows`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SigmoidMeanHyperConnectionQuery {
    /// `x * rsqrt(mean(x^2) + eps) * (1 + w)` over each stream's H values,
    /// with a separate weight per stream.
    GroupedNorm {
        streams: u32,
        width: u32,
        eps_bits: u32,
        capacity: u32,
    },
    /// `silu(down / streams)` over the rank-wide down projection.
    Bottleneck {
        streams: u32,
        rank: u32,
        capacity: u32,
    },
    /// `mean_s(sigmoid(up_s) * Xn_s)`, one H-wide sublayer input per token.
    Mix {
        streams: u32,
        width: u32,
        capacity: u32,
    },
    /// `X_s += 2 * sigmoid(inject_s / streams) * y` for every stream.
    Inject {
        streams: u32,
        width: u32,
        capacity: u32,
    },
    /// Sum the ranks' sublayer partials, inject into X, and emit X and the
    /// grouped norm of the updated X for the next sublayer.
    AllReduceInjectNorm {
        ranks: u32,
        streams: u32,
        width: u32,
        eps_bits: u32,
        capacity: u32,
    },
}

/// Hashed n-gram embedding and its per-layer gate. Hash heads
/// cover n-gram orders 2..=orders, `heads_per_order` each, in order-major
/// order; head h owns table rows `[offset_h, offset_h + size_h)`.
///
/// The geometry fields are math; `local_heads` is a rank's share of the
/// heads and `capacity` the most rows one launch serves, a sizing fact
/// (`backend::Need`). Buffer roles are in `rows`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NgramQuery {
    /// Hash each token with its predecessors and gather FP8 E4M3 rows of
    /// `head_width` for heads `[rank * local_heads, (rank + 1) * local_heads)`,
    /// dequantized by one per-tensor scale (multiplied). A predecessor at or
    /// behind the newest end-of-document token reads as that token. The
    /// multipliers, head sizes and offsets are runtime tables.
    HashGather {
        orders: u32,
        heads_per_order: u32,
        head_width: u32,
        local_heads: u32,
        capacity: u32,
    },
    /// Per stream, `d = <key_s, query_s> / sqrt(width)`, then
    /// `sigmoid(sign(d) * sqrt(max(|d|, 1e-6))) * value`.
    PleGate {
        streams: u32,
        width: u32,
        capacity: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum RoutedCapacity {
    PerExpert(u32),
    TotalRows(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct GroupedGemm {
    pub experts: u32,
    pub top_k: u32,
    pub capacity: RoutedCapacity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct GemmQuery {
    pub m_capacity: u32,
    pub n: u32,
    pub k: u32,
    pub weight: WeightFormat,
    pub activation: ActivationFormat,
    pub output: DType,
    pub epilogues: Epilogues,
    pub grouped: Option<GroupedGemm>,
}

/// The dense GEMM kernel of an entry and the K reduction it fixes. Each
/// decides output bits, so every dense entry declares one: no shape, row
/// count or symbol name implies it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum DenseKernel {
    /// Row-major weights through TMA tiles; every launch splits K into
    /// `splits` partitions. The compiled split capacity only sizes the grid
    /// and workspace.
    Tiled { splits: u8 },
    /// Fragment-layout weights for at most 32 rows; every launch splits K
    /// into `splits` partitions, as `Tiled`.
    FragmentDecode { splits: u8 },
    /// Fragment-layout weights for at most 32 rows, one thread-block
    /// cluster of K chunks per row tile; each warp sums `warp_blocks`
    /// 128-K blocks.
    FragmentCluster { warp_blocks: u32 },
    /// The F32 expert router over fragment-layout weights: K in `splits`
    /// partitions.
    FragmentRouter { splits: u8 },
    /// Fragment-layout prefill: each tile's K in `segments` fixed segments.
    FragmentPrefill { segments: u8 },
    /// `Tiled` arithmetic at one split for at most 32 rows: every launch
    /// spreads K chunks over `splits` partitions, keeps each MMA chain's
    /// temporary and adds them in K order as one split does, so the split
    /// count never changes output bits.
    TiledChains { splits: u8 },
    /// `Tiled` arithmetic at one split for at most 32 rows over BF16 or
    /// FP8 128x128 block weights, in CTAs of `strips` 16-row weight strips (one math warp
    /// each, reading MMA fragments with ldmatrix) and every row of the
    /// bucket: every launch spreads K chunks over `splits` partitions and,
    /// above one, keeps and adds the chains as `TiledChains` does, so
    /// neither count changes output bits.
    StripChains { strips: u8, splits: u8 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AttentionPhase {
    Prefill,
    Decode,
    /// Up to a few query tokens per item after a cached prefix: a verify
    /// step's current token and drafts, or a drafter block. Paged items read
    /// their own rows from the cache (written before attention); ring items
    /// read them from contiguous `k_new`/`v_new` (written after attention).
    /// Non-causal verify shows every token all of its item's keys.
    Verify,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AttentionLayout {
    Paged,
    Ring,
}

/// Dense attention over a paged or ring KV cache. Every field is math or a
/// layout the model shares with its other ops; how a backend splits keys,
/// tiles heads or merges partials is its own choice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionQuery {
    /// Math: query heads, each group of `q_heads / kv_heads` sharing a KV
    /// head.
    pub q_heads: u32,
    pub kv_heads: u32,
    /// Math: the widths of a query and key head and of a value head.
    pub qk_dim: u32,
    pub v_dim: u32,
    /// Layout: tokens of one cache page, or of one item's ring.
    pub page_size: u32,
    /// Math: the stored KV dtype, BF16 or scaled FP8 E4M3.
    pub kv_dtype: DType,
    /// Layout: pages through a block table, or one ring per item.
    pub layout: AttentionLayout,
    /// Math: the mask, the window, a per-head sink logit, a softcap and
    /// the softmax scale.
    pub causal: bool,
    /// Includes the current token and this many preceding tokens.
    pub window_left: Option<u32>,
    pub sink: bool,
    pub softcap: Option<f32>,
    pub softmax_scale: f32,
    /// Math: which keys each query row sees and where the step's own rows
    /// come from.
    pub phase: AttentionPhase,
    /// Layout: the most keys of an item, which also fixes a paged block
    /// table's row: `max_context_capacity` over `page_size`, rounded up.
    pub max_context_capacity: u32,
    /// Capacity: the most items of one step.
    pub max_batch_capacity: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RmsNormKind {
    Plain,
    GemmaOnePlusWeight,
    QkPerHead {
        head_dim: u32,
    },
    /// HF rounds x * rsqrt(mean(x*x) + eps) to BF16 before multiplying weight.
    HfBf16Intermediate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RotaryStyle {
    Neox,
    GptJ,
}

/// Where a rotation's cosines and sines come from. Pair `i` at position `p`
/// turns by the angle `p * f_i`, with `f_i = theta^(-2i / rotary_dim)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RotarySource {
    /// An F32 table `[position][rotary_dim / 2][cos, sin]` the model fills
    /// (`rope_kv::RopeBuffers::cos_sin`). A backend uses each entry as
    /// stored, neither recomputing nor rounding it, so a check gives the
    /// oracle the same table and no allowance for it. The entries are the
    /// model's choice; models fill them from
    /// `rotary::AngleRule::HfF32AngleBf16CosSin`, their references' table
    /// with BF16 cosines and sines.
    CosSinTable,
    /// Computed by the backend from `theta`. The models' math is their
    /// references' F32 table: an F32 `f_i`, the F32 angle `fl32(p * f_i)`,
    /// and its cosine and sine rounded to F32, which the references' BF16
    /// runs round to BF16 (`rotary::AngleRule::HfF32AngleBf16CosSin`). A
    /// backend may round the cosine and sine to BF16, and may form `f_i`,
    /// the angle and its cosine and sine in its own F32 or wider arithmetic,
    /// within these bounds. Its angle is within
    /// `(8 + 3 |log2 f_i|) * 2^-24 * p * f_i` of the exact `p * f_i`: the
    /// error scales with the pair's angle, through a few F32 roundings and
    /// the relative error an F32 exponent `log2 f_i` carries into `f_i`. Its
    /// cosine and sine are within `2^-21` of those of its angle (an F32
    /// evaluation), and a BF16 rounding adds at most half a BF16 step, `2^-8`
    /// of the value's magnitude. `rotary::check_computed` judges a cosine
    /// and sine against those of the exact angle, which `oracle::rope`
    /// computes in f64 without a table. A skipped or wrong rotation fails
    /// wherever it moves a cosine or sine by more than the allowance, as at
    /// the position where the pair turns by a radian.
    Computed,
}

/// Router presets, named by their math; `oracle::Routing::of_kind` gives
/// each one's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RouterKind {
    /// Softmax over every expert, top-k of the logits, optionally
    /// renormalized over the selected weights.
    SoftmaxTopK { renormalize: bool },
    /// Select the top-k of sigmoid(logit) + bias; weight by the unbiased
    /// sigmoid scores divided by their sum plus 1e-20, times
    /// `scaling_factor`.
    SigmoidBiasSelectNormalized { scaling_factor: f32 },
    /// Select the top-k of sqrt(softplus(logit)) + bias in F32, with
    /// lower expert ids first on ties. Weights use only the unbiased
    /// scores, divided by their selected sum plus 1e-20, then multiplied
    /// by `scaling_factor`. `router::RowBiasRouterQuery` selects one of
    /// two bias vectors independently for each row.
    SqrtSoftplusBiasSelectNormalized { scaling_factor: f32 },
    /// Keep the `topk_group` of `n_group` contiguous expert groups with the
    /// largest sums of their two largest logits (one in groups of one), take
    /// the top-k logits among them, and weight by the softmax over the
    /// selected logits.
    GroupLimitedSelectedSoftmax { n_group: u32, topk_group: u32 },
}

/// Where a sharded argmax final reads rank `r`'s pair of row `i`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArgmaxLayout {
    /// At `i * ranks + r`: each row's pairs in rank order.
    RowMajor,
    /// At `r * stride + i`: each rank's pairs of every row, as an
    /// all-gather of every rank's `stride` pairs lays them out. A prepared
    /// op's rows are at most `stride`.
    RankMajor { stride: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RowQuery {
    RmsNorm {
        width: u32,
        eps: f32,
        kind: RmsNormKind,
        fused_residual: bool,
        output_quantization: Option<OutputQuantization>,
    },
    /// Rotary embedding of the first `rotary_dim` values of each head at its
    /// row's position, paired as `style` says, with cosines and sines from
    /// `source` (whose precision [`RotarySource`] states). Each rotated value
    /// `x cos -/+ y sin` is computed in F32 or wider before its output
    /// rounding. The oracle is `oracle::rope`.
    Rope {
        head_dim: u32,
        rotary_dim: u32,
        theta: f32,
        style: RotaryStyle,
        source: RotarySource,
        fused_paged_kv_write: bool,
    },
    Quantize {
        width: u32,
        format: OutputQuantization,
    },
    Swiglu {
        width: u32,
        columns: SwigluColumns,
    },
    RingWrite {
        qk_dim: u32,
        v_dim: u32,
        kv_heads: u32,
        ring_size: u32,
        kv_dtype: DType,
    },
    /// Each token's `top_k` experts and route weights from F32 logits
    /// (`moe::RouterBuffers`), ties to the lower expert id. Every field is
    /// math.
    Router {
        experts: u32,
        top_k: u32,
        kind: RouterKind,
    },
    /// The routes of every expert sorted by expert, padded to
    /// `block_size`: `MoeQuery::LocalRoutes` with every expert local, whose
    /// padding is a backend choice. Kept for the entries that declare it.
    ExpertPermute {
        experts: u32,
        block_size: u32,
    },
    /// Each token's `top_k` routed rows added into one row. The op's full
    /// math (width, where the route weights apply) is `moe::ExpertCombineOp`.
    ExpertCombine {
        experts: u32,
        top_k: u32,
    },
    /// Copy table row `id` per token (`rows::EmbeddingBuffers`). Both
    /// fields are math: `vocab` names the table the op was built for and
    /// `width` is the row width. The table rows a launch reads are a buffer
    /// fact, so a sharded table bounds the ids by its own rows.
    EmbeddingGather {
        vocab: u32,
        width: u32,
    },
    /// The id of each row's largest F32 logit, ties to the lower id. `vocab`
    /// is math: the row stride and the most live logits.
    LogitsArgmax {
        vocab: u32,
    },
    /// One rank's argmax over its contiguous vocabulary shard, whose first
    /// global id is `rank * vocab_per_rank`: per row, the largest of the
    /// shard's live `logits` (BF16 or F32; BF16 widens exactly to F32) and
    /// its global id, as a pair of the winning logit's F32 bits and the id.
    /// Logits order by value with -0 equal to +0, and a NaN of any sign or
    /// payload ranks below every other value (-inf included) and equal to
    /// every other NaN; equal logits go to the lower id, so a row of only
    /// NaN yields its lowest global id, which is always in the vocabulary.
    /// Every field is math.
    ShardedArgmaxPartial {
        vocab_per_rank: u32,
        ranks: u32,
        logits: DType,
    },
    /// The argmax over every rank's pair of each row, read as `pairs`
    /// states: the id of the largest score in the partial's order, equal
    /// scores to the lower id. Over the partials of a row's shards it is
    /// that row's argmax over the whole vocabulary. `ranks` is math;
    /// `pairs` is where the pairs lie. Scattered ids go to distinct
    /// destinations (`rows::ArgmaxScatter`'s precondition).
    ShardedArgmaxFinal {
        ranks: u32,
        pairs: ArgmaxLayout,
    },
    /// The ids and log-probabilities of each row's `k` largest logits.
    /// Both fields are math; `k` is at most 20, the most top
    /// log-probabilities a request may ask for.
    LogitsTopK {
        vocab: u32,
        k: u32,
    },
}

/// RMSNorm applied to each row of a fused all-reduce's sum. The row width is
/// explicit: the row count is `elements / width`, and a kernel instantiated
/// for one width must refuse any other. Kind describes the weight and
/// intermediate rounding convention; a provider also specifies its reduction
/// order and residual requirements. Kind alone does not establish bit parity.
/// The epilogue pins no numerics: each collective protocol sums squares in
/// its own fixed tree. Its gate is not the row RMSNorm's: kernel-lab's
/// collective cases, all of the HF kind, take HF's ordered reference over the
/// expected residual sum (rounded to BF16) and require the residual output
/// bit exact, every rank's output equal, and each normalized output within
/// the protocol's stated number of BF16 steps of the reference (one or two,
/// two where a weight above one can widen a step).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormEpilogue {
    pub kind: RmsNormKind,
    pub width: u32,
    pub eps: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CommQuery {
    /// The sum over ranks of every rank's rows of `elements`, the same bits
    /// on every rank. Without fusion (`fused_residual` false, no
    /// `fused_rmsnorm`), `dtype` names the partials the ranks sum:
    ///
    /// - F32: F32 input and output rows, within the F64 sum's bound
    ///   (`oracle::all_reduce`).
    /// - BF16: BF16 partials of F32 input rows, written as BF16 output
    ///   rows. Each rank's F32 element is rounded once to BF16 (round to
    ///   nearest even), the ranks' BF16 values are summed in F32 in rank
    ///   order from +0.0, and the sum is rounded once to BF16
    ///   (`oracle::all_reduce_bf16_partials`), bitwise. Both roundings are
    ///   round to nearest even over the BF16 values with infinity past the
    ///   largest finite one: a magnitude rounds to an infinity of its sign
    ///   from the midpoint above the largest finite BF16 (F32 0x7f7f8000,
    ///   the midpoint included) and to the largest finite BF16 below it.
    ///   The sum follows IEEE F32 addition (a sum past the F32 range is an
    ///   infinity), and a NaN sum (a NaN partial, or
    ///   infinities of opposite signs) is the canonical BF16 NaN 0x7fff,
    ///   whatever the NaNs' signs and payloads. The input rows may be left
    ///   rounded to BF16 (an F32 value of the BF16 partial).
    ///
    /// The fused forms add a residual and normalize the sum
    /// ([`RmsNormEpilogue`]).
    AllReduce {
        ranks: u32,
        elements: u64,
        dtype: DType,
        fused_residual: bool,
        fused_rmsnorm: Option<RmsNormEpilogue>,
    },
    /// The sum over ranks of every rank's F32 rows of `elements`, written
    /// as BF16 output rows, the same bits on every rank: each element's F32
    /// values summed in F32 in rank order from +0.0 under IEEE addition and
    /// the sum rounded once to BF16 (`oracle::all_reduce_rounded`), bitwise.
    /// The rounding is `AllReduce`'s BF16 one (round to nearest even, an
    /// infinity from the midpoint above the largest finite BF16), and a NaN
    /// sum is the canonical BF16 NaN 0x7fff. The F32 partials of
    /// `AllReduce` with `dtype` F32, the output of its BF16 form.
    AllReduceRounded { ranks: u32, elements: u64 },
    AllGather {
        ranks: u32,
        elements_per_rank: u64,
        dtype: DType,
    },
}

/// A rejected query is never silently mapped to a different operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Refusal(pub &'static str);

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for Refusal {}

pub trait Validate {
    fn validate(&self) -> Result<(), Refusal>;
}

impl Validate for GemmQuery {
    fn validate(&self) -> Result<(), Refusal> {
        if self.m_capacity == 0 || self.n == 0 || self.k == 0 {
            return Err(Refusal("GEMM dimensions must be nonzero"));
        }
        if self.output != DType::Bf16 && self.output != DType::F32 {
            return Err(Refusal(
                "GEMM output must be BF16 or F32 before output quantization",
            ));
        }
        let w_block = match self.weight {
            WeightFormat::Fp8Block32x32 | WeightFormat::Mxfp8 | WeightFormat::Mxfp4 => 32,
            WeightFormat::Nvfp4 => 16,
            WeightFormat::Fp8Block128x128(_) | WeightFormat::Fp8Rows64x128(_) => 128,
            _ => 1,
        };
        let a_block = match self.activation {
            ActivationFormat::Mxfp8 => 32,
            ActivationFormat::Nvfp4 => 16,
            ActivationFormat::Fp8Group128(_) => 128,
            ActivationFormat::Bf16 => 1,
        };
        if !self.k.is_multiple_of(w_block) || !self.k.is_multiple_of(a_block) {
            return Err(Refusal(
                "K does not align to the requested quantization blocks",
            ));
        }
        // The final N scale row may describe a partial block (for example
        // a 3392-wide projection). K blocks must remain complete.
        if self.epilogues.swiglu.is_some() && !self.n.is_multiple_of(2) {
            return Err(Refusal("SwiGLU needs an even number of columns"));
        }
        if self.epilogues.gelu.is_some() && self.epilogues.swiglu.is_some() {
            return Err(Refusal("GELU and SwiGLU cannot share one epilogue"));
        }
        if let Some(g) = self.grouped {
            let capacity = match g.capacity {
                RoutedCapacity::PerExpert(rows) => rows.checked_mul(g.experts),
                RoutedCapacity::TotalRows(rows) => Some(rows),
            };
            if g.experts == 0
                || g.top_k == 0
                || g.top_k > g.experts
                || capacity.is_none_or(|rows| rows < self.m_capacity)
            {
                return Err(Refusal(
                    "invalid grouped GEMM experts, top-k or routed capacity",
                ));
            }
        } else if self.epilogues.weighted_moe_combine {
            return Err(Refusal("weighted MoE combine requires grouped GEMM"));
        }
        Ok(())
    }
}

impl Validate for MultimodalQuery {
    fn validate(&self) -> Result<(), Refusal> {
        match *self {
            Self::AudioRopeBf16 { capacity: 0 } => Err(Refusal("zero audio RoPE capacity")),
            Self::ContiguousAttention {
                q_heads,
                kv_heads,
                head_dim,
                capacity,
                ..
            }
            | Self::ContiguousAttentionPairs {
                q_heads,
                kv_heads,
                head_dim,
                capacity,
                ..
            } if q_heads == 0
                || kv_heads == 0
                || !q_heads.is_multiple_of(kv_heads)
                || head_dim == 0
                || capacity == 0 =>
            {
                Err(Refusal("invalid contiguous attention geometry"))
            }
            Self::LayerNorm {
                width,
                eps_bits,
                capacity,
            }
            | Self::LayerNormF32 {
                width,
                eps_bits,
                capacity,
            }
            | Self::RmsNormF32 {
                width,
                eps_bits,
                capacity,
            } if width == 0
                || capacity == 0
                || !f32::from_bits(eps_bits).is_finite()
                || f32::from_bits(eps_bits) <= 0.0 =>
            {
                Err(Refusal("invalid LayerNorm geometry or epsilon"))
            }
            Self::Gemm(query) | Self::GemmBf16(query) => {
                query.validate()?;
                if query.grouped.is_some()
                    || query.weight != WeightFormat::Bf16
                    || query.activation != ActivationFormat::Bf16
                {
                    return Err(Refusal("encoder GEMM requires dense BF16 operands"));
                }
                Ok(())
            }
            Self::GemmPairs(query) => {
                Self::Gemm(query).validate()?;
                let epilogues = query.epilogues;
                if epilogues.swiglu.is_some() || epilogues.output_quantization.is_some() {
                    return Err(Refusal(
                        "BF16 pair GEMM has bias, GELU and residual epilogues",
                    ));
                }
                if epilogues.residual_add && query.output != DType::F32 {
                    return Err(Refusal("BF16 pair GEMM residual needs F32 output"));
                }
                Ok(())
            }
            Self::ScratchRope {
                q_heads,
                kv_heads,
                head_dim,
                rotary_dim,
                capacity,
                ..
            }
            | Self::RopeQkvPairs {
                q_heads,
                kv_heads,
                head_dim,
                rotary_dim,
                capacity,
                ..
            } if q_heads == 0
                || kv_heads == 0
                || !q_heads.is_multiple_of(kv_heads)
                || head_dim == 0
                || rotary_dim == 0
                || rotary_dim > head_dim
                || capacity == 0
                || !rotary_dim.is_multiple_of(2) =>
            {
                Err(Refusal("invalid scratch RoPE geometry"))
            }
            Self::ResidualAdd { width, capacity }
            | Self::Bf16ToF32 { width, capacity }
            | Self::GeluRows { width, capacity }
            | Self::Permute { width, capacity }
            | Self::EmbeddingOverride { width, capacity }
            | Self::PermuteF32 { width, capacity }
            | Self::ResidualAddF32 { width, capacity }
                if width == 0 || capacity == 0 =>
            {
                Err(Refusal("invalid row width or capacity"))
            }
            Self::InterpolatedPositionAddF32 {
                width,
                table_rows,
                capacity,
            } if width == 0 || table_rows == 0 || capacity == 0 || table_rows > i32::MAX as u32 => {
                Err(Refusal("invalid interpolated position geometry"))
            }
            Self::SplitQkv {
                q_width,
                kv_width,
                capacity,
            } if q_width == 0
                || kv_width == 0
                || capacity == 0
                || kv_width
                    .checked_mul(2)
                    .and_then(|kv| kv.checked_add(q_width))
                    .is_none() =>
            {
                Err(Refusal("invalid QKV split geometry"))
            }
            Self::Im2col1d {
                channels,
                kernel,
                stride,
                input,
                capacity,
                ..
            } if channels == 0
                || kernel == 0
                || stride == 0
                || capacity == 0
                || !matches!(input, DType::F32 | DType::Bf16) =>
            {
                Err(Refusal("invalid im2col geometry or input dtype"))
            }
            Self::SwigluF32 { hidden, capacity }
                if hidden == 0 || capacity == 0 || hidden.checked_mul(2).is_none() =>
            {
                Err(Refusal("invalid SwiGLU row width or capacity"))
            }
            Self::SwigluClampedF32 {
                hidden,
                limit_bits,
                capacity,
            } if hidden == 0
                || capacity == 0
                || hidden.checked_mul(2).is_none()
                || !valid_eps(limit_bits) =>
            {
                Err(Refusal("invalid clamped SwiGLU width, limit or capacity"))
            }
            Self::QkRmsNormF32 {
                q_heads,
                kv_heads,
                head_dim,
                eps_bits,
                capacity,
            } if q_heads == 0
                || kv_heads == 0
                || !q_heads.is_multiple_of(kv_heads)
                || head_dim == 0
                || capacity == 0
                || !valid_eps(eps_bits)
                || (u64::from(q_heads) + 2 * u64::from(kv_heads))
                    .checked_mul(u64::from(head_dim))
                    .is_none_or(|width| width > u64::from(u32::MAX)) =>
            {
                Err(Refusal("invalid q/k RMSNorm geometry or epsilon"))
            }
            Self::GeluRowsF32 { width, capacity } if width == 0 || capacity == 0 => {
                Err(Refusal("invalid GELU row width or capacity"))
            }
            Self::PadColumnsF32 {
                input,
                output,
                capacity,
            } if input == 0 || output <= input || capacity == 0 => {
                Err(Refusal("invalid padded row widths or capacity"))
            }
            Self::Im2col1dF32 {
                channels,
                kernel,
                stride,
                capacity,
                ..
            } if channels == 0 || kernel == 0 || stride == 0 || capacity == 0 => {
                Err(Refusal("invalid im2col geometry"))
            }
            Self::EmbeddingSumBf16 {
                tables,
                vocab,
                width,
                capacity,
            }
            | Self::EmbeddingSum {
                tables,
                vocab,
                width,
                capacity,
            } if tables == 0 || vocab == 0 || width == 0 || capacity == 0 => {
                Err(Refusal("invalid embedding sum geometry"))
            }
            _ => Ok(()),
        }
    }
}

/// A verify phase verifies at least one row per sequence. The most rows a
/// launch verifies is a backend's limit, not math.
pub(crate) fn valid_phase(phase: RecurrentPhase) -> bool {
    !matches!(phase, RecurrentPhase::Verify { max_rows: 0 })
}

pub(crate) fn valid_eps(bits: u32) -> bool {
    let eps = f32::from_bits(bits);
    eps.is_finite() && eps > 0.0
}

/// Math only: a backend refuses the shapes it cannot run.
impl Validate for RecurrentQuery {
    fn validate(&self) -> Result<(), Refusal> {
        if self.checked_state_elements().is_none() {
            return Err(Refusal("recurrent slot state overflows u64 elements"));
        }
        match *self {
            Self::CausalConv1d {
                channels,
                kernel,
                dilation,
                phase,
                capacity,
                ..
            } if channels == 0
                // At most 8 taps is SM120's limit, not math; a model's
                // compiler test pins this text until that model migrates.
                || !(2..=8).contains(&kernel)
                || dilation == 0
                || capacity == 0
                || !valid_phase(phase) =>
            {
                Err(Refusal("invalid causal convolution geometry or phase"))
            }
            Self::GatedDeltaRule {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                phase,
                capacity,
            } if key_heads == 0
                || value_heads == 0
                || !value_heads.is_multiple_of(key_heads)
                || key_dim == 0
                || value_dim == 0
                || capacity == 0
                || !valid_phase(phase) =>
            {
                Err(Refusal("invalid gated delta rule geometry or phase"))
            }
            Self::GatedRmsNorm {
                width,
                head_dim,
                eps_bits,
                capacity,
                ..
            } if head_dim == 0
                || !width.is_multiple_of(head_dim)
                || width == 0
                || capacity == 0
                || !valid_eps(eps_bits) =>
            {
                Err(Refusal("invalid gated RMSNorm geometry or epsilon"))
            }
            _ => Ok(()),
        }
    }
}

impl Validate for GatedAttentionQuery {
    fn validate(&self) -> Result<(), Refusal> {
        match *self {
            Self::QkNormRopeKvWrite {
                q_heads,
                kv_heads,
                head_dim,
                rotary_dim,
                theta_bits,
                mrope,
                eps_bits,
                page_size,
                kv_dtype,
                capacity,
            } => {
                let theta = f32::from_bits(theta_bits);
                if q_heads == 0
                    || kv_heads == 0
                    || !q_heads.is_multiple_of(kv_heads)
                    || head_dim == 0
                    || rotary_dim == 0
                    || rotary_dim > head_dim
                    || !rotary_dim.is_multiple_of(2)
                    || !theta.is_finite()
                    || theta <= 0.0
                    || !valid_eps(eps_bits)
                    || page_size == 0
                    || !page_size.is_power_of_two()
                    || !matches!(kv_dtype, DType::Bf16 | DType::Fp8E4M3)
                    || capacity == 0
                {
                    return Err(Refusal("invalid gated attention norm and RoPE geometry"));
                }
                if mrope != [0; 3] && mrope.iter().sum::<u32>() != rotary_dim / 2 {
                    return Err(Refusal("M-RoPE sections must cover every rotary pair"));
                }
                Ok(())
            }
            Self::OutputGate {
                heads,
                head_dim,
                capacity,
            } if heads == 0 || head_dim == 0 || capacity == 0 => {
                Err(Refusal("invalid attention output gate geometry"))
            }
            Self::OutputGate { .. } => Ok(()),
        }
    }
}

fn valid_rope(head_dim: u32, rotary_dim: u32, theta_bits: u32, mrope: [u32; 3]) -> bool {
    let theta = f32::from_bits(theta_bits);
    rotary_dim > 0
        && rotary_dim <= head_dim
        && rotary_dim.is_multiple_of(2)
        && theta.is_finite()
        && theta > 0.0
        && (mrope == [0; 3] || mrope.iter().sum::<u32>() == rotary_dim / 2)
}

impl Validate for SparseAttentionQuery {
    fn validate(&self) -> Result<(), Refusal> {
        let valid = match *self {
            Self::IndexerQueryGemmaNormMrope {
                heads,
                head_dim,
                rotary_dim,
                theta_bits,
                mrope,
                eps_bits,
                capacity,
            } => {
                heads > 0
                    && valid_rope(head_dim, rotary_dim, theta_bits, mrope)
                    && valid_eps(eps_bits)
                    && capacity > 0
            }
            Self::IndexerKeyWrite {
                head_dim,
                rotary_dim,
                theta_bits,
                mrope,
                eps_bits,
                block,
                ring,
                capacity,
            } => {
                valid_rope(head_dim, rotary_dim, theta_bits, mrope)
                    && valid_eps(eps_bits)
                    && block > 1
                    && ring >= block
                    && ring.is_multiple_of(block)
                    && capacity > 0
            }
            Self::BlockScores {
                heads,
                head_dim,
                max_blocks,
                capacity,
            } => heads > 0 && head_dim > 0 && max_blocks > 0 && capacity > 0,
            Self::BlockTopK {
                block,
                top_blocks,
                max_blocks,
                capacity,
            } => block > 1 && top_blocks > 0 && max_blocks > 0 && capacity > 0,
            Self::SparseAttention {
                q_heads,
                kv_heads,
                head_dim,
                block,
                top_blocks,
                softmax_scale_bits,
                page_size,
                kv_dtype,
                capacity,
                ..
            } => {
                let scale = f32::from_bits(softmax_scale_bits);
                q_heads > 0
                    && kv_heads > 0
                    && q_heads.is_multiple_of(kv_heads)
                    && head_dim > 0
                    && block > 1
                    && top_blocks > 0
                    && scale.is_finite()
                    && scale > 0.0
                    && page_size > 0
                    && page_size.is_power_of_two()
                    && matches!(kv_dtype, DType::Bf16 | DType::Fp8E4M3)
                    && capacity > 0
            }
        };
        if valid {
            Ok(())
        } else {
            Err(Refusal("invalid block-sparse attention geometry"))
        }
    }
}

impl Validate for MoeQuery {
    fn validate(&self) -> Result<(), Refusal> {
        match *self {
            Self::LocalRoutes {
                experts,
                local_experts,
                top_k,
                capacity,
            } if local_experts == 0
                || !experts.is_multiple_of(local_experts)
                || top_k == 0
                || top_k > experts
                || capacity == 0 =>
            {
                Err(Refusal("invalid local expert route geometry"))
            }
            Self::SharedExpertGateCombine { width, capacity } if width == 0 || capacity == 0 => {
                Err(Refusal("invalid shared expert combine geometry"))
            }
            Self::LocalRoutedSum {
                experts,
                local_experts,
                top_k,
                width,
                output,
                capacity,
            } if local_experts == 0
                || !experts.is_multiple_of(local_experts)
                || top_k == 0
                || top_k > experts
                || width == 0
                || capacity == 0
                || !matches!(output, DType::Bf16 | DType::F32) =>
            {
                Err(Refusal("invalid local routed sum geometry or output"))
            }
            _ => Ok(()),
        }
    }
}

impl Validate for SigmoidMeanHyperConnectionQuery {
    fn validate(&self) -> Result<(), Refusal> {
        let (streams, width, eps_bits, capacity) = match *self {
            Self::GroupedNorm {
                streams,
                width,
                eps_bits,
                capacity,
            } => (streams, width, Some(eps_bits), capacity),
            Self::Bottleneck {
                streams,
                rank,
                capacity,
            } => (streams, rank, None, capacity),
            Self::Mix {
                streams,
                width,
                capacity,
            }
            | Self::Inject {
                streams,
                width,
                capacity,
            } => (streams, width, None, capacity),
            Self::AllReduceInjectNorm {
                ranks,
                streams,
                width,
                eps_bits,
                capacity,
            } => {
                if ranks == 0 {
                    return Err(Refusal("hyper-connection all-reduce needs ranks"));
                }
                (streams, width, Some(eps_bits), capacity)
            }
        };
        if streams < 2 || width == 0 || capacity == 0 || eps_bits.is_some_and(|e| !valid_eps(e)) {
            return Err(Refusal("invalid hyper-connection geometry or epsilon"));
        }
        Ok(())
    }
}

impl Validate for NgramQuery {
    fn validate(&self) -> Result<(), Refusal> {
        match *self {
            Self::HashGather {
                orders,
                heads_per_order,
                head_width,
                local_heads,
                capacity,
            } if orders < 2
                || heads_per_order == 0
                || head_width == 0
                || local_heads == 0
                || !(orders - 1)
                    .checked_mul(heads_per_order)
                    .is_some_and(|heads| heads.is_multiple_of(local_heads))
                || capacity == 0 =>
            {
                Err(Refusal("invalid n-gram hash geometry"))
            }
            Self::PleGate {
                streams,
                width,
                capacity,
            } if streams == 0 || width == 0 || capacity == 0 => {
                Err(Refusal("invalid n-gram gate geometry"))
            }
            _ => Ok(()),
        }
    }
}

impl Validate for AttentionQuery {
    fn validate(&self) -> Result<(), Refusal> {
        if self.q_heads == 0 || self.kv_heads == 0 || !self.q_heads.is_multiple_of(self.kv_heads) {
            return Err(Refusal("Q heads must be a positive multiple of KV heads"));
        }
        if self.qk_dim == 0 || self.v_dim == 0 {
            return Err(Refusal("attention heads must be at least one wide"));
        }
        if self.page_size == 0
            || !self.page_size.is_power_of_two()
            || self.max_context_capacity == 0
            || self.max_batch_capacity == 0
        {
            return Err(Refusal("invalid attention page or capacity"));
        }
        if !matches!(self.kv_dtype, DType::Bf16 | DType::Fp8E4M3) {
            return Err(Refusal("unsupported KV dtype"));
        }
        if self.layout == AttentionLayout::Ring
            && self.window_left.is_none_or(|left| left >= self.page_size)
        {
            return Err(Refusal("ring KV requires a window within its size"));
        }
        if !self.softmax_scale.is_finite()
            || self.softmax_scale <= 0.0
            || self.softcap.is_some_and(|v| !v.is_finite() || v <= 0.0)
        {
            return Err(Refusal("invalid attention scale or softcap"));
        }
        if self
            .window_left
            .is_some_and(|v| v >= self.max_context_capacity)
        {
            return Err(Refusal("attention window exceeds context capacity"));
        }
        Ok(())
    }
}

impl Validate for RowQuery {
    fn validate(&self) -> Result<(), Refusal> {
        match *self {
            RowQuery::RmsNorm {
                width, eps, kind, ..
            } => {
                if width == 0 || !eps.is_finite() || eps <= 0.0 {
                    return Err(Refusal("invalid RMSNorm width or epsilon"));
                }
                if let RmsNormKind::QkPerHead { head_dim } = kind
                    && (head_dim == 0 || !width.is_multiple_of(head_dim))
                {
                    return Err(Refusal("QK RMSNorm head dimension does not divide width"));
                }
            }
            RowQuery::Rope {
                head_dim,
                rotary_dim,
                theta,
                ..
            } => {
                if head_dim == 0
                    || rotary_dim == 0
                    || rotary_dim > head_dim
                    || !rotary_dim.is_multiple_of(2)
                    || !theta.is_finite()
                    || theta <= 0.0
                {
                    return Err(Refusal("invalid RoPE dimension or theta"));
                }
            }
            RowQuery::Quantize { width: 0, .. } => return Err(Refusal("zero quantization width")),
            RowQuery::Swiglu { width: 0, .. } => return Err(Refusal("zero SwiGLU width")),
            RowQuery::RingWrite {
                qk_dim,
                v_dim,
                kv_heads,
                ring_size,
                kv_dtype,
            } if qk_dim == 0
                || v_dim == 0
                || kv_heads == 0
                || ring_size == 0
                || !matches!(kv_dtype, DType::Bf16 | DType::Fp8E4M3) =>
            {
                return Err(Refusal("invalid ring KV write shape or dtype"));
            }
            RowQuery::Router {
                experts,
                top_k,
                kind,
            } => {
                if experts == 0 || top_k == 0 || top_k > experts {
                    return Err(Refusal("invalid router experts or top-k"));
                }
                match kind {
                    RouterKind::GroupLimitedSelectedSoftmax {
                        n_group,
                        topk_group,
                    } if n_group == 0
                        || !experts.is_multiple_of(n_group)
                        || topk_group == 0
                        || topk_group > n_group =>
                    {
                        return Err(Refusal("invalid grouped top-k groups"));
                    }
                    RouterKind::SigmoidBiasSelectNormalized { scaling_factor }
                    | RouterKind::SqrtSoftplusBiasSelectNormalized { scaling_factor }
                        if !scaling_factor.is_finite() || scaling_factor <= 0.0 =>
                    {
                        return Err(Refusal("invalid router scaling factor"));
                    }
                    _ => {}
                }
            }
            RowQuery::ExpertPermute {
                experts,
                block_size,
            } if experts == 0 || block_size == 0 => {
                return Err(Refusal("invalid permute experts or block size"));
            }
            RowQuery::ExpertCombine { experts, top_k }
                if experts == 0 || top_k == 0 || top_k > experts =>
            {
                return Err(Refusal("invalid combine experts or top-k"));
            }
            RowQuery::EmbeddingGather { vocab, width } if vocab == 0 || width == 0 => {
                return Err(Refusal("invalid embedding shape"));
            }
            RowQuery::LogitsArgmax { vocab: 0 } => return Err(Refusal("empty vocabulary")),
            RowQuery::ShardedArgmaxPartial {
                vocab_per_rank: 0, ..
            }
            | RowQuery::ShardedArgmaxPartial { ranks: 0, .. }
            | RowQuery::ShardedArgmaxFinal { ranks: 0, .. }
            | RowQuery::ShardedArgmaxFinal {
                pairs: ArgmaxLayout::RankMajor { stride: 0 },
                ..
            } => {
                return Err(Refusal("invalid sharded argmax shape"));
            }
            RowQuery::ShardedArgmaxPartial { logits, .. }
                if !matches!(logits, DType::Bf16 | DType::F32) =>
            {
                return Err(Refusal("sharded argmax logits are BF16 or F32"));
            }
            RowQuery::LogitsTopK { vocab, k } if vocab == 0 || k == 0 || k > 20 || k > vocab => {
                return Err(Refusal("logits top-k must be between 1 and min(20, vocab)"));
            }
            _ => {}
        }
        Ok(())
    }
}

impl Validate for CommQuery {
    fn validate(&self) -> Result<(), Refusal> {
        match *self {
            CommQuery::AllReduce {
                ranks,
                elements,
                dtype,
                fused_rmsnorm,
                ..
            } => {
                if ranks == 0
                    || elements == 0
                    || !matches!(dtype, DType::Bf16 | DType::F32)
                    || fused_rmsnorm.is_some_and(|norm| {
                        !norm.eps.is_finite()
                            || norm.eps <= 0.0
                            || norm.width == 0
                            || !elements.is_multiple_of(u64::from(norm.width))
                    })
                {
                    return Err(Refusal(
                        "invalid all-reduce rank, length, dtype, norm width or epsilon",
                    ));
                }
            }
            CommQuery::AllReduceRounded { ranks, elements } => {
                if ranks == 0 || elements == 0 {
                    return Err(Refusal("invalid rounded all-reduce rank or length"));
                }
            }
            CommQuery::AllGather {
                ranks,
                elements_per_rank,
                dtype,
            } => {
                if ranks == 0
                    || elements_per_rank == 0
                    || !matches!(dtype, DType::Bf16 | DType::F32)
                {
                    return Err(Refusal("invalid all-gather rank, length or dtype"));
                }
            }
        }
        Ok(())
    }
}

/// Stable semantic ID. A16 and A8 use distinct activation format segments.
fn gemm_numerical_id(query: &GemmQuery) -> String {
    let weight = match query.weight {
        WeightFormat::Bf16 => "bf16",
        WeightFormat::Fp8PerTensor => "fp8-tensor",
        WeightFormat::Fp8PerChannel => "fp8-channel",
        WeightFormat::Fp8Block128x128(ScaleMode::ExactF32) => "fp8-block128-f32",
        WeightFormat::Fp8Block128x128(ScaleMode::PowerOfTwoF32) => "fp8-block128-pow2",
        WeightFormat::Fp8Rows64x128(ScaleMode::ExactF32) => "fp8-rows64-f32",
        WeightFormat::Fp8Rows64x128(ScaleMode::PowerOfTwoF32) => "fp8-rows64-pow2",
        WeightFormat::Fp8Block32x32 => "fp8-block32-ue8m0",
        WeightFormat::Mxfp8 => "mxfp8",
        WeightFormat::Mxfp4 => "mxfp4",
        WeightFormat::Nvfp4 => "nvfp4",
    };
    let activation = match query.activation {
        ActivationFormat::Bf16 => "bf16",
        ActivationFormat::Fp8Group128(ScaleMode::ExactF32) => "fp8-group128-f32",
        ActivationFormat::Fp8Group128(ScaleMode::PowerOfTwoF32) => "fp8-group128-pow2",
        ActivationFormat::Mxfp8 => "mxfp8",
        ActivationFormat::Nvfp4 => "nvfp4",
    };
    let output = match query.epilogues.output_quantization {
        None => "",
        Some(OutputQuantization::Fp8Group128(ScaleMode::ExactF32)) => ".out-fp8-group128-f32",
        Some(OutputQuantization::Fp8Group128(ScaleMode::PowerOfTwoF32)) => ".out-fp8-group128-pow2",
        Some(OutputQuantization::Mxfp8) => ".out-mxfp8",
        Some(OutputQuantization::Mxfp4) => ".out-mxfp4",
        Some(OutputQuantization::Nvfp4) => ".out-nvfp4",
    };
    let bias = if query.epilogues.bias { ".bias" } else { "" };
    let gelu = match query.epilogues.gelu {
        None => "",
        Some(GeluMode::Erf) => ".gelu-erf",
        Some(GeluMode::Tanh) => ".gelu-tanh",
    };
    let version = if query.activation == ActivationFormat::Bf16 {
        match query.weight {
            WeightFormat::Nvfp4 => "exact-local-global-after-mma.v2",
            WeightFormat::Mxfp8 => "exact-scaled-operand.v2",
            _ => "v1",
        }
    } else {
        "v1"
    };
    format!("gemm.w-{weight}.a-{activation}.acc-f32{bias}{gelu}{output}.{version}")
}

/// Stable, versioned algorithm identity. Shape and scalar parameters remain
/// in the typed query; the ID distinguishes numerical algorithms and formats.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct NumericalContractId(String);

impl NumericalContractId {
    pub(crate) fn new(id: String) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NumericalContractId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

pub trait NumericalContract {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal>;
}

impl NumericalContract for GemmQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId(gemm_numerical_id(self)))
    }
}

impl NumericalContract for AttentionQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let dtype = match self.kv_dtype {
            DType::Bf16 => "bf16",
            DType::Fp8E4M3 => "fp8-e4m3",
            _ => return Err(Refusal("unsupported KV dtype")),
        };
        let phase = match self.phase {
            AttentionPhase::Prefill => "prefill",
            AttentionPhase::Decode => "decode",
            AttentionPhase::Verify if self.causal => "verify",
            AttentionPhase::Verify => "verify-bidirectional",
        };
        let layout = match self.layout {
            AttentionLayout::Paged => "paged",
            AttentionLayout::Ring => "ring",
        };
        Ok(NumericalContractId(format!(
            "attention.{layout}-kv-{dtype}.{phase}.v1"
        )))
    }
}

impl NumericalContract for MultimodalQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match self {
            Self::AudioRopeBf16 { .. } => "row.rope-audio-bf16-intermediates.v1".to_owned(),
            Self::EmbeddingSumBf16 { .. } => {
                "row.multi-embedding-sum-bf16-intermediates.v1".to_owned()
            }
            Self::ResidualAdd { .. } => "row.residual-add-bf16.v1".to_owned(),
            Self::Bf16ToF32 { .. } => "row.bf16-to-f32.v1".to_owned(),
            Self::GeluRows { .. } => "row.gelu-erf-bf16.v1".to_owned(),
            Self::SplitQkv { .. } => "row.split-qkv-bf16.v1".to_owned(),
            Self::ContiguousAttention { .. } => "attention.contiguous-varlen-bf16.v1".to_owned(),
            Self::LayerNorm { .. } => "row.layernorm-affine.v1".to_owned(),
            Self::Gemm(query) => return query.numerical_contract_id(),
            Self::GemmBf16(query) => {
                return Ok(NumericalContractId(format!(
                    "{}.linear-bf16",
                    query.numerical_contract_id()?
                )));
            }
            Self::ScratchRope {
                style: RotaryStyle::Neox,
                ..
            } => "row.rope-scratch-neox.v1".to_owned(),
            Self::ScratchRope {
                style: RotaryStyle::GptJ,
                ..
            } => "row.rope-scratch-gptj.v1".to_owned(),
            Self::Permute { .. } => "row.permute-index.v1".to_owned(),
            Self::Im2col1d {
                input: DType::F32, ..
            } => "row.im2col1d-f32-to-bf16.v1".to_owned(),
            Self::Im2col1d { .. } => "row.im2col1d-bf16.v1".to_owned(),
            Self::EmbeddingSum { .. } => "row.multi-embedding-sum.v1".to_owned(),
            Self::EmbeddingOverride { .. } => "row.embedding-override.v1".to_owned(),
            Self::InterpolatedPositionAddF32 { .. } => {
                "row.interpolated-position-add-f32.v1".to_owned()
            }
            Self::LayerNormF32 { .. } => "row.layernorm-affine-f32.v1".to_owned(),
            Self::RopeQkvPairs {
                style: RotaryStyle::Neox,
                ..
            } => "row.rope-qkv-neox-f32-to-bf16-pairs.v1".to_owned(),
            Self::RopeQkvPairs {
                style: RotaryStyle::GptJ,
                ..
            } => "row.rope-qkv-gptj-f32-to-bf16-pairs.v1".to_owned(),
            Self::ContiguousAttentionPairs { .. } => {
                "attention.contiguous-varlen-bf16-pairs-f32.v1".to_owned()
            }
            Self::GemmPairs(query) => {
                let epilogues = query.epilogues;
                let bias = if epilogues.bias { ".bias" } else { "" };
                let gelu = match epilogues.gelu {
                    None => "",
                    Some(GeluMode::Erf) => ".gelu-erf",
                    Some(GeluMode::Tanh) => ".gelu-tanh",
                };
                let residual = if epilogues.residual_add {
                    ".residual"
                } else {
                    ""
                };
                let output = if query.output == DType::F32 {
                    "f32"
                } else {
                    "bf16"
                };
                format!(
                    "gemm.w-bf16.a-f32-bf16-pairs.acc-f32{bias}{gelu}{residual}.out-{output}.v1"
                )
            }
            Self::RmsNormF32 { .. } => "row.rmsnorm-f32.v1".to_owned(),
            Self::PermuteF32 { .. } => "row.permute-index-f32.v1".to_owned(),
            Self::SwigluF32 { .. } => "row.swiglu-f32.v1".to_owned(),
            Self::ResidualAddF32 { .. } => "row.residual-add-f32.v1".to_owned(),
            Self::Im2col1dF32 { .. } => "row.im2col1d-f32.v1".to_owned(),
            Self::QkRmsNormF32 { .. } => "row.qk-rmsnorm-f32.v1".to_owned(),
            Self::SwigluClampedF32 { .. } => "row.swiglu-clamped-f32.v1".to_owned(),
            Self::GeluRowsF32 { .. } => "row.gelu-erf-f32.v1".to_owned(),
            Self::PadColumnsF32 { .. } => "row.pad-zero-columns-f32.v1".to_owned(),
        };
        Ok(NumericalContractId(name))
    }
}

fn recurrent_phase_name(phase: RecurrentPhase) -> &'static str {
    match phase {
        RecurrentPhase::Prefill => "prefill",
        RecurrentPhase::Decode => "decode",
        RecurrentPhase::Verify { .. } => "verify",
    }
}

impl NumericalContract for RecurrentQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match *self {
            Self::CausalConv1d {
                activation, phase, ..
            } => {
                let activation = match activation {
                    ConvActivation::Identity => "",
                    ConvActivation::Silu => "-silu",
                };
                format!(
                    "recurrent.causal-conv1d-depthwise{activation}.{}.v1",
                    recurrent_phase_name(phase)
                )
            }
            Self::GatedDeltaRule { phase, .. } => format!(
                "recurrent.gated-delta-rule.{}.v1",
                recurrent_phase_name(phase)
            ),
            Self::GatedRmsNorm {
                gate: GateActivation::Sigmoid,
                ..
            } => "row.rmsnorm-gated-sigmoid.v1".to_owned(),
            Self::GatedRmsNorm {
                gate: GateActivation::Silu,
                ..
            } => "row.rmsnorm-gated-silu.v1".to_owned(),
        };
        Ok(NumericalContractId(name))
    }
}

impl NumericalContract for GatedAttentionQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match self {
            Self::QkNormRopeKvWrite {
                kv_dtype: DType::Fp8E4M3,
                ..
            } => "row.qk-gemma-norm-mrope-kv-write-fp8.v1",
            Self::QkNormRopeKvWrite { .. } => "row.qk-gemma-norm-mrope-kv-write-bf16.v1",
            Self::OutputGate { .. } => "row.attention-output-sigmoid-gate.v1",
        };
        Ok(NumericalContractId(name.to_owned()))
    }
}

impl NumericalContract for SparseAttentionQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match *self {
            Self::IndexerQueryGemmaNormMrope { .. } => {
                "sparse.indexer-query-gemma-norm-mrope.v1".to_owned()
            }
            Self::IndexerKeyWrite { .. } => {
                "sparse.indexer-key-ring-block-mean-write.v1".to_owned()
            }
            Self::BlockScores { .. } => "sparse.indexer-block-scores-relu-sum.v1".to_owned(),
            Self::BlockTopK { .. } => "sparse.indexer-block-topk-lower-index.v1".to_owned(),
            Self::SparseAttention {
                kv_dtype, phase, ..
            } => {
                let dtype = if kv_dtype == DType::Fp8E4M3 {
                    "fp8-e4m3"
                } else {
                    "bf16"
                };
                let phase = match phase {
                    AttentionPhase::Prefill => "prefill",
                    AttentionPhase::Decode => "decode",
                    AttentionPhase::Verify => "verify",
                };
                format!("sparse.attention-paged-kv-{dtype}.{phase}.v1")
            }
        };
        Ok(NumericalContractId(name))
    }
}

impl NumericalContract for MoeQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match self {
            Self::LocalRoutes { .. } => "row.moe-local-expert-routes.v1",
            Self::SharedExpertGateCombine { .. } => "row.moe-shared-expert-sigmoid-combine.v1",
            Self::LocalRoutedSum { .. } => "row.moe-local-routed-sum.v1",
        };
        Ok(NumericalContractId(name.to_owned()))
    }
}

impl NumericalContract for SigmoidMeanHyperConnectionQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match self {
            Self::GroupedNorm { .. } => "row.hc-grouped-gemma-rmsnorm.v1",
            Self::Bottleneck { .. } => "row.hc-bottleneck-silu.v1",
            Self::Mix { .. } => "row.hc-mix-sigmoid-mean.v1",
            Self::Inject { .. } => "row.hc-inject-residual.v1",
            Self::AllReduceInjectNorm { .. } => "comm.all-reduce-hc-inject-grouped-rmsnorm.v1",
        };
        Ok(NumericalContractId(name.to_owned()))
    }
}

impl NumericalContract for NgramQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match self {
            Self::HashGather { .. } => "row.ngram-hash-gather-fp8-tensor.v1",
            Self::PleGate { .. } => "row.ngram-gate-signed-sqrt-sigmoid.v1",
        };
        Ok(NumericalContractId(name.to_owned()))
    }
}

impl NumericalContract for RowQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match self {
            RowQuery::RmsNorm {
                kind: RmsNormKind::Plain,
                ..
            } => "rmsnorm-plain",
            RowQuery::RmsNorm {
                kind: RmsNormKind::GemmaOnePlusWeight,
                ..
            } => "rmsnorm-gemma",
            RowQuery::RmsNorm {
                kind: RmsNormKind::QkPerHead { .. },
                ..
            } => "rmsnorm-qk",
            RowQuery::RmsNorm {
                kind: RmsNormKind::HfBf16Intermediate,
                ..
            } => "rmsnorm-hf-bf16",
            RowQuery::Rope {
                style: RotaryStyle::Neox,
                ..
            } => "rope-neox",
            RowQuery::Rope {
                style: RotaryStyle::GptJ,
                ..
            } => "rope-gptj",
            RowQuery::Quantize {
                format: OutputQuantization::Fp8Group128(ScaleMode::ExactF32),
                ..
            } => "quant-fp8-group128-f32",
            RowQuery::Quantize {
                format: OutputQuantization::Fp8Group128(ScaleMode::PowerOfTwoF32),
                ..
            } => "quant-fp8-group128-pow2",
            RowQuery::Quantize {
                format: OutputQuantization::Mxfp8,
                ..
            } => "quant-mxfp8",
            RowQuery::Quantize {
                format: OutputQuantization::Mxfp4,
                ..
            } => "quant-mxfp4",
            RowQuery::Quantize {
                format: OutputQuantization::Nvfp4,
                ..
            } => "quant-nvfp4",
            RowQuery::Swiglu { .. } => "swiglu",
            RowQuery::RingWrite { .. } => "ring-kv-write",
            RowQuery::Router {
                kind: RouterKind::SoftmaxTopK { .. },
                ..
            } => "router-softmax-topk",
            RowQuery::Router {
                kind: RouterKind::SigmoidBiasSelectNormalized { .. },
                ..
            } => "router-sigmoid-bias-noaux-tc",
            RowQuery::Router {
                kind: RouterKind::SqrtSoftplusBiasSelectNormalized { .. },
                ..
            } => "router-sqrt-softplus-bias-normalized",
            RowQuery::Router {
                kind: RouterKind::GroupLimitedSelectedSoftmax { .. },
                ..
            } => "router-grouped-topk",
            RowQuery::ExpertPermute { .. } => "expert-permute-align",
            RowQuery::ExpertCombine { .. } => "expert-combine",
            RowQuery::EmbeddingGather { .. } => "embedding-gather",
            RowQuery::LogitsArgmax { .. } => "logits-argmax",
            RowQuery::ShardedArgmaxPartial { .. } => "logits-sharded-argmax-partial",
            RowQuery::ShardedArgmaxFinal { .. } => "logits-sharded-argmax-final",
            RowQuery::LogitsTopK { .. } => "logits-topk",
        };
        Ok(NumericalContractId(format!("row.{name}.v1")))
    }
}

impl NumericalContract for CommQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match self {
            CommQuery::AllReduce { .. } => "all-reduce-sum",
            CommQuery::AllReduceRounded { .. } => "all-reduce-rounded",
            CommQuery::AllGather { .. } => "all-gather",
        };
        Ok(NumericalContractId(format!("comm.{name}.v1")))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OpFamily {
    Gemm,
    Attention,
    Row,
    Comm,
    Recurrent,
}

/// Values in this type are prepared capacities, never live batch sizes.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum StaticQuery {
    Gemm(GemmQuery),
    Multimodal(MultimodalQuery),
    Recurrent(RecurrentQuery),
    GatedAttention(GatedAttentionQuery),
    SparseAttention(SparseAttentionQuery),
    Moe(MoeQuery),
    HyperConnection(SigmoidMeanHyperConnectionQuery),
    Ngram(NgramQuery),
    GatedMlpProjection(GatedMlpProjectionQuery),
    LatentAttention(LatentAttentionQuery),
    HeadBatchedGemm(HeadBatchedGemmQuery),
    PooledIndexer(PooledIndexerQuery),
    SinkhornHyperConnection(SinkhornHyperConnectionQuery),
    PerChannelDecayDeltaRule(PerChannelDecayDeltaRuleQuery),
    ContextParallel(ContextParallelQuery),
    Attention {
        q_heads: u32,
        kv_heads: u32,
        qk_dim: u32,
        v_dim: u32,
        page_size: u32,
        kv_dtype: DType,
        phase: AttentionPhase,
        causal: bool,
        window_left: Option<u32>,
        sink: bool,
        softcap_bits: Option<u32>,
        layout: AttentionLayout,
        max_context_capacity: u32,
        max_batch_capacity: u32,
    },
    Row {
        family: &'static str,
        width_capacity: u32,
        dtype: DType,
        /// Stable digest of static row traits omitted from the generic width.
        traits_digest: u64,
    },
    Comm {
        family: &'static str,
        ranks: u32,
        elements_capacity: u64,
        dtype: DType,
    },
}

impl From<GemmQuery> for StaticQuery {
    fn from(query: GemmQuery) -> Self {
        Self::Gemm(query)
    }
}

impl From<AttentionQuery> for StaticQuery {
    fn from(q: AttentionQuery) -> Self {
        Self::Attention {
            q_heads: q.q_heads,
            kv_heads: q.kv_heads,
            qk_dim: q.qk_dim,
            v_dim: q.v_dim,
            page_size: q.page_size,
            kv_dtype: q.kv_dtype,
            phase: q.phase,
            causal: q.causal,
            window_left: q.window_left,
            sink: q.sink,
            softcap_bits: q.softcap.map(f32::to_bits),
            layout: q.layout,
            max_context_capacity: q.max_context_capacity,
            max_batch_capacity: q.max_batch_capacity,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct CompileKnobs {
    pub tile_m: u16,
    pub tile_n: u16,
    pub tile_k: u16,
    pub stages: u8,
    pub swizzle: u8,
    pub warps: u8,
    /// Maximum supported split-K factor, not the live factor.
    pub split_k_capacity: u8,
}

impl Validate for CompileKnobs {
    fn validate(&self) -> Result<(), Refusal> {
        if self.tile_m == 0
            || self.tile_n == 0
            || self.tile_k == 0
            || self.stages == 0
            || self.warps == 0
            || self.split_k_capacity == 0
        {
            return Err(Refusal("compile knobs must be nonzero"));
        }
        Ok(())
    }
}

/// This key has no field capable of holding a live count.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CompileKey {
    pub query: StaticQuery,
    pub knobs: CompileKnobs,
    /// The kernel of a dense (ungrouped) GEMM engine entry; `None` for every
    /// other operation.
    pub dense: Option<DenseKernel>,
    pub target: Target,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstantiationSpec {
    pub family: OpFamily,
    pub key: CompileKey,
    pub symbol: String,
}

impl Validate for InstantiationSpec {
    fn validate(&self) -> Result<(), Refusal> {
        self.key.knobs.validate()?;
        let family = match self.key.query {
            StaticQuery::Gemm(_) => OpFamily::Gemm,
            StaticQuery::Multimodal(
                MultimodalQuery::Gemm(_)
                | MultimodalQuery::GemmBf16(_)
                | MultimodalQuery::GemmPairs(_),
            ) => OpFamily::Gemm,
            StaticQuery::Multimodal(
                MultimodalQuery::ContiguousAttention { .. }
                | MultimodalQuery::ContiguousAttentionPairs { .. },
            ) => OpFamily::Attention,
            StaticQuery::Multimodal(_) => OpFamily::Row,
            StaticQuery::Attention { .. } => OpFamily::Attention,
            StaticQuery::Row { .. } => OpFamily::Row,
            StaticQuery::Comm { .. } => OpFamily::Comm,
            StaticQuery::Recurrent(RecurrentQuery::GatedRmsNorm { .. }) => OpFamily::Row,
            StaticQuery::Recurrent(_) => OpFamily::Recurrent,
            StaticQuery::GatedAttention(_) => OpFamily::Row,
            StaticQuery::SparseAttention(SparseAttentionQuery::SparseAttention { .. }) => {
                OpFamily::Attention
            }
            StaticQuery::SparseAttention(_) => OpFamily::Row,
            StaticQuery::Moe(_) => OpFamily::Row,
            StaticQuery::HyperConnection(
                SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm { .. },
            ) => OpFamily::Comm,
            StaticQuery::HyperConnection(_) => OpFamily::Row,
            StaticQuery::Ngram(_) => OpFamily::Row,
            StaticQuery::GatedMlpProjection(_) => OpFamily::Gemm,
            StaticQuery::LatentAttention(LatentAttentionQuery::Attention { .. }) => {
                OpFamily::Attention
            }
            StaticQuery::LatentAttention(_) => OpFamily::Row,
            StaticQuery::HeadBatchedGemm(_) => OpFamily::Gemm,
            StaticQuery::PooledIndexer(_) => OpFamily::Row,
            StaticQuery::SinkhornHyperConnection(
                SinkhornHyperConnectionQuery::AllReduceExpandMixNorm { .. },
            ) => OpFamily::Comm,
            StaticQuery::SinkhornHyperConnection(_) => OpFamily::Row,
            StaticQuery::PerChannelDecayDeltaRule(_) => OpFamily::Recurrent,
            StaticQuery::ContextParallel(
                crate::context_parallel::ContextParallelQuery::PagedAttentionPartials { .. }
                | crate::context_parallel::ContextParallelQuery::SelectedAttentionPartials { .. },
            ) => OpFamily::Attention,
            StaticQuery::ContextParallel(_) => OpFamily::Row,
        };
        if self.family != family {
            return Err(Refusal("op family does not match static query"));
        }
        if self.key.dense.is_some()
            && !matches!(self.key.query, StaticQuery::Gemm(q) if q.grouped.is_none())
            && !matches!(
                self.key.query,
                StaticQuery::GatedMlpProjection(q) if q.gemm.grouped.is_none()
            )
        {
            return Err(Refusal("only a dense GEMM entry declares a dense kernel"));
        }
        let Some((base, version)) = self.symbol.rsplit_once("_v") else {
            return Err(Refusal("symbol must end in _v<N>"));
        };
        if base.is_empty()
            || !base.bytes().all(|v| v.is_ascii_alphanumeric() || v == b'_')
            || version.parse::<u32>().ok().filter(|&v| v > 0).is_none()
        {
            return Err(Refusal("invalid versioned symbol"));
        }
        Ok(())
    }
}

/// Live values are passed at launch and cannot enter `CompileKey`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeCounts {
    pub m: u32,
    pub batch: u32,
    pub context: u32,
    pub routed_rows: u32,
    pub split_k: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct LaunchPlan {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub dynamic_shared_bytes: u32,
    pub workspace_bytes: u64,
    pub cluster: [u32; 3],
    pub pdl: bool,
}

impl LaunchPlan {
    /// The geometry every CUDA backend shares: nonzero grid, block and
    /// cluster extents within CUDA's launch limits (a grid of at most
    /// 2^31 - 1 by 65,535 by 65,535 blocks, a block of at most 1,024 threads
    /// and 1,024 by 1,024 by 64), and a cluster of at most the eight blocks
    /// every cluster-capable GPU takes. Each backend bounds shared memory
    /// itself.
    pub fn validate_geometry(&self) -> Result<(), Refusal> {
        let cluster_blocks = self
            .cluster
            .iter()
            .try_fold(1_u32, |n, &x| n.checked_mul(x));
        let threads = self.block.iter().try_fold(1_u32, |n, &x| n.checked_mul(x));
        if self.grid.contains(&0)
            || self.block.contains(&0)
            || self.cluster.contains(&0)
            || cluster_blocks.is_none_or(|v| v > 8)
        {
            return Err(Refusal("invalid grid, block or cluster geometry"));
        }
        if self.grid[0] > i32::MAX as u32
            || self.grid[1] > 65_535
            || self.grid[2] > 65_535
            || self.block[0] > 1024
            || self.block[1] > 1024
            || self.block[2] > 64
            || threads.is_none_or(|v| v > 1024)
        {
            return Err(Refusal("grid or block exceeds CUDA's launch limits"));
        }
        Ok(())
    }
}

/// The geometry check and SM120's dynamic shared memory bound, for callers
/// that do not plan through a backend yet; the SM120 backend applies the
/// same bound when it prepares an op.
impl Validate for LaunchPlan {
    fn validate(&self) -> Result<(), Refusal> {
        self.validate_geometry()?;
        if self.dynamic_shared_bytes > 101_376 {
            return Err(Refusal("dynamic shared memory exceeds SM120 limit"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TensorMapElement {
    U8,
    F16,
    Bf16,
    F32,
    Fp8E4M3,
    Fp8E5M2,
    U4Align8B,
    U4Align16B,
    U6Align16B,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TensorMapSwizzle {
    None,
    B32,
    B64,
    B128,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TensorMapInterleave {
    None,
    B16,
    B32,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TensorMapL2Promotion {
    None,
    B64,
    B128,
    B256,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TensorMapOobFill {
    Zero,
    Nan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorMap {
    /// Opaque handle for the pool whose base address is encoded in the map.
    pub base_buffer: u64,
    pub dims: Vec<u64>,
    pub strides_bytes: Vec<u64>,
    pub box_dims: Vec<u32>,
    pub element_strides: Vec<u32>,
    pub swizzle: TensorMapSwizzle,
    pub interleave: TensorMapInterleave,
    pub l2_promotion: TensorMapL2Promotion,
    pub oob_fill: TensorMapOobFill,
    pub element: TensorMapElement,
}

impl Validate for TensorMap {
    fn validate(&self) -> Result<(), Refusal> {
        let rank = self.dims.len();
        if !(1..=5).contains(&rank)
            || self.strides_bytes.len() != rank
            || self.box_dims.len() != rank
            || self.element_strides.len() != rank
            || self.dims.contains(&0)
            || self.box_dims.contains(&0)
            || self.element_strides.contains(&0)
        {
            return Err(Refusal(
                "tensor map must have matching nonzero rank 1 to 5 geometry",
            ));
        }
        Ok(())
    }
}

/// DeviceBuffer is an opaque host-issued handle, never a raw device address.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Argument {
    DeviceBuffer(u64),
    U64(u64),
    U32(u32),
    I32(i32),
    NullDevicePointer,
    TensorMap(TensorMap),
    DevicePointerArray(Vec<u64>),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ArgumentList(pub Vec<Argument>);

impl Validate for ArgumentList {
    fn validate(&self) -> Result<(), Refusal> {
        for arg in &self.0 {
            match arg {
                Argument::TensorMap(map) => map.validate()?,
                Argument::DevicePointerArray(items)
                    if !(1..=MAX_COMM_RANKS).contains(&items.len()) =>
                {
                    return Err(Refusal("pointer array needs 1 to MAX_COMM_RANKS handles"));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compile_key_has_only_static_geometry() {
        let q = GemmQuery {
            m_capacity: 16,
            n: 128,
            k: 128,
            weight: WeightFormat::Mxfp4,
            activation: ActivationFormat::Bf16,
            output: DType::Bf16,
            epilogues: Epilogues::default(),
            grouped: None,
        };
        q.validate().unwrap();
        let key = CompileKey {
            query: q.into(),
            knobs: CompileKnobs {
                tile_m: 16,
                tile_n: 64,
                tile_k: 32,
                stages: 2,
                swizzle: 1,
                warps: 4,
                split_k_capacity: 8,
            },
            dense: None,
            target: Target::Sm120f,
        };
        let live_a = RuntimeCounts {
            m: 1,
            batch: 1,
            context: 1,
            routed_rows: 0,
            split_k: 1,
        };
        let live_b = RuntimeCounts {
            m: 12,
            batch: 12,
            context: 512,
            routed_rows: 0,
            split_k: 8,
        };
        assert_ne!(live_a, live_b);
        // Exhaustive patterns make a new live-count field fail this test's build.
        let CompileKey {
            query,
            knobs: _,
            dense: _,
            target: _,
        } = key;
        let StaticQuery::Gemm(GemmQuery {
            m_capacity: _,
            n: _,
            k: _,
            weight: _,
            activation: _,
            output: _,
            epilogues: _,
            grouped: _,
        }) = query
        else {
            panic!("expected GEMM static query")
        };
        assert_eq!(
            q.numerical_contract_id().unwrap().as_str(),
            "gemm.w-mxfp4.a-bf16.acc-f32.v1"
        );
        let a8 = GemmQuery {
            activation: ActivationFormat::Fp8Group128(ScaleMode::ExactF32),
            ..q
        };
        assert_ne!(
            q.numerical_contract_id().unwrap(),
            a8.numerical_contract_id().unwrap()
        );
        let quantized = GemmQuery {
            epilogues: Epilogues {
                output_quantization: Some(OutputQuantization::Mxfp8),
                ..a8.epilogues
            },
            ..a8
        };
        assert!(
            quantized
                .numerical_contract_id()
                .unwrap()
                .as_str()
                .contains(".out-mxfp8.v1")
        );
        assert_ne!(
            a8.numerical_contract_id().unwrap(),
            quantized.numerical_contract_id().unwrap()
        );
    }

    #[test]
    fn fp8_block_accepts_partial_final_n_scale_row() {
        for n in [3392, 3712] {
            let query = GemmQuery {
                m_capacity: 1,
                n,
                k: 4096,
                weight: WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
                activation: ActivationFormat::Bf16,
                output: DType::Bf16,
                epilogues: Epilogues::default(),
                grouped: None,
            };
            assert!(query.validate().is_ok());
        }
    }

    #[test]
    fn launch_geometry_is_shared_and_validate_keeps_the_sm120_bound() {
        let plan = LaunchPlan {
            grid: [1, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 101_376,
            workspace_bytes: 0,
            cluster: [2, 2, 2],
            pdl: true,
        };
        assert!(plan.validate().is_ok());
        let over = LaunchPlan {
            dynamic_shared_bytes: 101_377,
            ..plan
        };
        assert!(over.validate_geometry().is_ok());
        assert!(over.validate().is_err());
        for bad in [
            LaunchPlan {
                grid: [0, 1, 1],
                ..plan
            },
            LaunchPlan {
                block: [256, 0, 1],
                ..plan
            },
            LaunchPlan {
                cluster: [4, 4, 1],
                ..plan
            },
            LaunchPlan {
                grid: [1, 65_536, 1],
                ..plan
            },
            LaunchPlan {
                grid: [1, 1, 65_536],
                ..plan
            },
            LaunchPlan {
                grid: [1 << 31, 1, 1],
                ..plan
            },
            LaunchPlan {
                block: [1025, 1, 1],
                ..plan
            },
            LaunchPlan {
                block: [1, 1, 65],
                ..plan
            },
            LaunchPlan {
                block: [64, 32, 1],
                ..plan
            },
        ] {
            assert!(bad.validate_geometry().is_err());
            assert!(bad.validate().is_err());
        }
        // CUDA's largest launch.
        let widest = LaunchPlan {
            grid: [i32::MAX as u32, 65_535, 65_535],
            block: [1024, 1, 1],
            cluster: [1, 1, 1],
            ..plan
        };
        assert!(widest.validate_geometry().is_ok());
        assert!(
            LaunchPlan {
                block: [16, 1, 64],
                ..widest
            }
            .validate_geometry()
            .is_ok()
        );
    }
}
