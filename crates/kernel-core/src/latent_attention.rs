//! Contracts for latent attention: every query head attends over one shared
//! latent row per token, which serves as both key and value (K == V), and
//! the head projections are absorbed into the query and the output.
//!
//! Per token the cache holds the latent `c` of width `latent` (512): the
//! normalized output of the latent down projection, BF16 before storage.
//! Each head h has an up projection whose key half `W_UK_h` [nope, latent]
//! and value half `W_UV_h` [v_dim, latent] are BF16. Instead of expanding
//! keys and values per head, the absorbed form computes
//!
//! - `q_abs_h = W_UK_h^T q_nope_h` (a `HeadBatchedGemmQuery`, BF16 output),
//! - `o_h = softmax_j(scale * q_abs_h . c_j) c_j` over the row's key list
//!   (`LatentAttentionQuery::Attention`),
//! - `out_h = W_UV_h o_h` (a `HeadBatchedGemmQuery`, BF16 output),
//!
//! which equals the expanded form in exact arithmetic; the rounding points
//! differ from a reference that expands K and V, and the accuracy gate
//! measures that. With no rotary part, `scale` is the expanded head's
//! `qk_dim^-0.5` (1/16 for a 256-wide head), a parameter here.

pub use crate::attention::BlockTables;
use crate::contract::{
    AttentionPhase, DType, NumericalContract, NumericalContractId, Refusal, Validate,
};

/// Latent elements covered by one FP8 scale.
pub const LATENT_FP8_SCALE_BLOCK: u32 = 128;
/// Slots in the fixed-size native window ring, independently owned by each layer.
pub const LATENT_WINDOW_SLOTS: u32 = 128;

/// The window's physical slot index space. `Ring128` fixes its size at 128.
/// `Staged` accepts a caller-sized space in every phase: a whole prefill
/// chunk, or a larger ring with spare slots for cancellable verify writes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LatentWindowStorage {
    Ring128,
    Staged { slots: u32 },
}

/// How the latent cache stores one token.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LatentKvFormat {
    /// `latent` BF16 values, 2 bytes each.
    Bf16,
    /// `latent` E4M3 codes, then one F32 scale per 128 latent elements: 528
    /// bytes per token at width 512. A block's scale is its BF16 values'
    /// maximum magnitude divided by 448 in F32 (scale 1 and zero codes for
    /// an all-zero block), and each code is `E4M3(x / scale)` with the F32
    /// division and `cvt.rn.satfinite`, as `quantize::fp8_activation_1x128`.
    /// A reader decodes `E4M3(code) * scale`.
    Fp8E4M3Block128,
    /// E4M3 codes and UE8M0 scales, one scale per 32 channels. The
    /// post-rotation BF16 input is quantized with amax floored at 1e-4,
    /// scale = 2^ceil(log2(amax * RN32(1/448))), and RNE saturated codes.
    /// Codes times scales must decode exactly to finite BF16 operands.
    Fp8E4M3Ue8M0Block32,
    /// Packed E2M1 codes (even channel in the low nibble) and positive
    /// E4M3 scales, one per 16 channels. Scale = E4M3(max(amax, 6*2^-9)/6)
    /// in F32, then code = E2M1(clamp(x/scale, -6, 6)), RNE. There is no
    /// global scale. Code times scale is exact in BF16.
    Fp4E2M1E4M3Block16,
}

impl LatentKvFormat {
    /// Bytes of one token's latent row.
    pub fn token_bytes(self, latent: u32) -> u64 {
        let latent = u64::from(latent);
        match self {
            Self::Bf16 => 2 * latent,
            Self::Fp8E4M3Block128 => latent + 4 * latent.div_ceil(LATENT_FP8_SCALE_BLOCK.into()),
            Self::Fp8E4M3Ue8M0Block32 => latent + latent.div_ceil(32),
            Self::Fp4E2M1E4M3Block16 => latent.div_ceil(2) + latent.div_ceil(16),
        }
    }

    /// Bytes of one page. A page holds `page_size` consecutive tokens: for
    /// BF16, rows `[page_size][latent]`; for FP8, every token's codes
    /// `[page_size][latent]` first, then their scales
    /// `[page_size][latent / 128]` F32, so code rows stay 16-byte aligned.
    /// The native formats also store every code row first, then all scale
    /// rows: one byte per scale, per 32 or 16 channels respectively.
    pub fn page_bytes(self, latent: u32, page_size: u32) -> u64 {
        self.token_bytes(latent) * u64::from(page_size)
    }

    fn dtype(self) -> DType {
        match self {
            Self::Bf16 => DType::Bf16,
            Self::Fp8E4M3Block128 => DType::Fp8E4M3,
            Self::Fp8E4M3Ue8M0Block32 => DType::Fp8E4M3,
            Self::Fp4E2M1E4M3Block16 => DType::Fp4E2M1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8E4M3Block128 => "fp8-e4m3-block128",
            Self::Fp8E4M3Ue8M0Block32 => "fp8-e4m3-ue8m0-block32",
            Self::Fp4E2M1E4M3Block16 => "fp4-e2m1-e4m3-block16",
        }
    }
}

/// What a latent attention entry writes per (row, head).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LatentAttentionOutput {
    /// `bf16(o / l)` with an F32 division, `[rows, heads, latent]`; zeros
    /// for a row whose key list is empty.
    Normalized,
    /// The unnormalized partial for a context-parallel merge
    /// (`context_parallel::ContextParallelQuery::PartialMerge`): running
    /// maximum `m` `[rows, heads]` F32, denominator `l` `[rows, heads]` F32
    /// and numerator `o` `[rows, heads, latent]` F32. An empty key list
    /// writes `m = -inf`, `l = 0` and `o = 0`. Partials exclude the sink
    /// and output inverse rotation. The final merge adds the sink once,
    /// normalizes, rounds to BF16, then applies the inverse tail rotation.
    Partial,
}

/// Latent attention contracts.
///
/// Math, which the model needs: `heads`, `latent`, `softmax_scale_bits`,
/// `kv` (the cache's numerics), `output`, `max_keys` (the key list row the
/// model's selection writes) and the declared extensions. Layout shared
/// with the model's other cache ops: `page_size`. `phase` changes only the
/// schedule; which partition a row's sums take is a backend choice, and the
/// exactness a model relies on (verify rows equal to decode rows, prefill
/// chunks) is its `Need`. `capacity` is the most rows of one launch. The
/// arithmetic below pins the latent block chains and key tiles today; the
/// splits of a row's keys, their merge and every tile size are the
/// backend's.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LatentAttentionQuery {
    /// Store each row's BF16 latent `[rows, latent]` at its token position
    /// through the sequence's page table, in `kv` format. The latent is
    /// already normalized; this entry only stores it, before the attention
    /// of the same step reads it. A row whose position is negative (-1)
    /// stores nothing and reads no page table entry, as a -1 key list entry
    /// is a hole: a context-parallel localization
    /// (`context_parallel::ContextParallelQuery::LocalizeKeys`, width 1)
    /// leaves -1 at the rows whose page another rank owns.
    KvWrite {
        latent: u32,
        page_size: u32,
        kv: LatentKvFormat,
        capacity: u32,
    },
    /// Softmax attention of `heads` absorbed query heads `[rows, heads,
    /// latent]` BF16 over each row's key list.
    ///
    /// Key list: `[rows, max_keys]` i32 token positions of the row's
    /// sequence, read through its page table; -1 is a hole and is skipped.
    /// The live budget (entries per row a launch reads) is a runtime value
    /// up to `max_keys` (2051 for 2048 selected tokens plus a 3-token
    /// incomplete pool tail). The list alone decides visibility, so the
    /// phase changes only the schedule and capacities; a list may repeat no
    /// position.
    ///
    /// Arithmetic, all F32 with IEEE FMA in a fixed order:
    /// - score `s_j = scale * (q . c_j)`. The dot product runs in 128-wide
    ///   latent blocks, each an MMA chain from zero, added into the
    ///   accumulator in block order. For FP8 each block's chain multiplies
    ///   the E4M3 codes (exact in BF16) and the block's sum is multiplied by
    ///   its scale before the add.
    /// - `m = max_j s_j`, `p_j = exp(s_j - m)`, `l = sum_j p_j`,
    ///   `o = sum_j p_j c_j` over the decoded latent. The engine may carry
    ///   `p_j` as BF16 terms (as the sparse attention engine does) within
    ///   its numerical bound; a running maximum rescales earlier terms.
    ///   PV sums one key tile of at most 128 keys per chain from zero, added
    ///   into `o` in key-list order.
    /// - output per `LatentAttentionOutput`.
    ///
    /// Native code/scale products decode exactly into BF16 before both QK
    /// and PV. They do not apply scales after the QK reduction. The native
    /// online softmax rounds `exp(score - running_max)` to BF16 after the
    /// max shift and before PV; its denominator uses the unrounded F32
    /// probabilities. The f64 oracle keeps ideal probabilities and bounds
    /// this storage rounding separately from accumulation, independently
    /// of the engine's key tiles and split count.
    ///
    /// With two sources, concatenate the window list then the compressed
    /// list before one softmax, with at most `max_keys` entries in total.
    /// Window lists name physical slots of the declared window storage.
    /// Producers guarantee an in-range slot or -1; a reader treats every
    /// index outside `[0, slots)` as a hole without reading codes or scales.
    /// Compressed lists name positions through their own page table and skip
    /// -1. Valid positions must not repeat within a source.
    /// The lists alone specify visibility, including block-visible verify.
    Attention {
        heads: u32,
        latent: u32,
        max_keys: u32,
        softmax_scale_bits: u32,
        page_size: u32,
        kv: LatentKvFormat,
        phase: AttentionPhase,
        output: LatentAttentionOutput,
        /// Learned F32 logits `[heads]`. Add exp(sink[h] - m) to the final
        /// denominator exactly once, after all sources and splits. The sink
        /// contributes no value. An all-hole row produces zeros.
        sink: bool,
        /// 0 or 64. Adjacent pairs in the last 64 of `latent` channels are
        /// rotated in place, never appended. Q and K arrive already rotated
        /// using F32 cos/sin and rounded to BF16; cache quantization follows
        /// that rounding. After normalization round to BF16, apply the
        /// conjugate query rotation in F32, and round to BF16 again. A
        /// 512-wide query still uses scale 512^-0.5 and writes 512 channels.
        tail_rope_dims: u32,
        /// Paged `Fp4E2M1E4M3Block16` keys alongside a native FP8 window.
        second_fp4_keys: bool,
        capacity: u32,
    },
}

/// Weight layout of one head's matrix in a `HeadBatchedGemmQuery`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum HeadWeightLayout {
    /// Row-major `[n, k]`: `y[i] = sum_k W[i][k] x[k]` (the value half,
    /// `W_UV_h` [v_dim, latent]).
    NByK,
    /// Row-major `[k, n]`: `y[i] = sum_k W[k][i] x[k]` (the key half used
    /// transposed, `W_UK_h` [nope, latent] for `q_abs = W_UK_h^T q`).
    KByN,
}

/// A head-batched BF16 GEMM: for every row and head h, `y[row, h, :] =
/// W_h x[row, h, :]` with x `[rows, heads, k]` BF16, y `[rows, heads, n]`,
/// and head h's weight at a launch-supplied element stride from the first
/// (so both halves of an interleaved up projection bind in place). F32
/// accumulation in 128-K chains from zero added in K order, one rounding to
/// `output`. `GemmQuery` has no head batch, so the absorbed projections use
/// this contract rather than one dense GEMM per head. For 16 heads, the key
/// absorption is `heads 16, n 512, k 256, KByN` and the value projection
/// `heads 16, n 256, k 512, NByK`. Every field is math but `m_capacity`,
/// the most rows of one launch; tiles and MMA shapes are the backend's.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct HeadBatchedGemmQuery {
    pub heads: u32,
    pub n: u32,
    pub k: u32,
    pub layout: HeadWeightLayout,
    pub output: DType,
    pub m_capacity: u32,
}

impl Validate for LatentAttentionQuery {
    fn validate(&self) -> Result<(), Refusal> {
        let valid_latent = |latent: u32| latent > 0 && latent.is_multiple_of(128);
        let valid_page = |page_size: u32| page_size > 0;
        match *self {
            Self::KvWrite {
                latent,
                page_size,
                capacity,
                ..
            } => {
                if !valid_latent(latent) || !valid_page(page_size) || capacity == 0 {
                    return Err(Refusal("invalid latent KV write geometry"));
                }
            }
            Self::Attention {
                heads,
                latent,
                max_keys,
                softmax_scale_bits,
                page_size,
                tail_rope_dims,
                second_fp4_keys,
                kv,
                capacity,
                ..
            } => {
                let scale = f32::from_bits(softmax_scale_bits);
                if heads == 0
                    || !valid_latent(latent)
                    || max_keys == 0
                    || !scale.is_finite()
                    || scale <= 0.0
                    || !valid_page(page_size)
                    || capacity == 0
                {
                    return Err(Refusal("invalid latent attention geometry or scale"));
                }
                if !matches!(tail_rope_dims, 0 | 64) {
                    return Err(Refusal(
                        "latent attention tail RoPE supports 0 or 64 in-place dimensions",
                    ));
                }
                if second_fp4_keys && kv != LatentKvFormat::Fp8E4M3Ue8M0Block32 {
                    return Err(Refusal(
                        "latent attention second FP4 source requires the native FP8 window",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Validate for HeadBatchedGemmQuery {
    fn validate(&self) -> Result<(), Refusal> {
        if self.heads == 0 || self.n == 0 || self.k == 0 || self.m_capacity == 0 {
            return Err(Refusal("head-batched GEMM dimensions must be nonzero"));
        }
        if !matches!(self.output, DType::Bf16 | DType::F32) {
            return Err(Refusal("head-batched GEMM output must be BF16 or F32"));
        }
        Ok(())
    }
}

fn phase_name(phase: AttentionPhase) -> &'static str {
    match phase {
        AttentionPhase::Prefill => "prefill",
        AttentionPhase::Decode => "decode",
        AttentionPhase::Verify => "verify",
    }
}

impl NumericalContract for LatentAttentionQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match *self {
            Self::KvWrite { kv, .. } => format!("row.latent-kv-write-{}.v1", kv.name()),
            Self::Attention {
                kv,
                phase,
                output,
                sink,
                tail_rope_dims,
                second_fp4_keys,
                ..
            } => {
                let output = match output {
                    LatentAttentionOutput::Normalized => "",
                    LatentAttentionOutput::Partial => ".partial-lse",
                };
                let extensions = format!(
                    "{}{}{}",
                    if sink { ".sink-once" } else { "" },
                    if tail_rope_dims == 64 {
                        ".tail-rope64-in-place"
                    } else {
                        ""
                    },
                    if second_fp4_keys {
                        ".window-fp4-paged"
                    } else {
                        ""
                    }
                );
                format!(
                    "attention.latent-paged-kv-{}-gathered.{}{output}{extensions}.v1",
                    kv.name(),
                    phase_name(phase)
                )
            }
        };
        Ok(NumericalContractId::new(name))
    }
}

impl NumericalContract for HeadBatchedGemmQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let layout = match self.layout {
            HeadWeightLayout::NByK => "nk",
            HeadWeightLayout::KByN => "kn",
        };
        let output = if self.output == DType::F32 {
            "f32"
        } else {
            "bf16"
        };
        Ok(NumericalContractId::new(format!(
            "gemm.head-batched-w-bf16-{layout}.a-bf16.acc-f32-chain128.out-{output}.v1"
        )))
    }
}

impl LatentAttentionQuery {
    /// The cache element type of the entry's latent rows.
    pub fn kv_dtype(&self) -> DType {
        match *self {
            Self::KvWrite { kv, .. } | Self::Attention { kv, .. } => kv.dtype(),
        }
    }
}

crate::scratch_record! {
    /// A latent cache: `pages` pages of the format's layout
    /// ([`LatentKvFormat::page_bytes`]), page p at byte `p * page_stride` of
    /// `pool`. The stride holds at least one page, so other per-token data
    /// may share a page's allocation.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LatentKvCache {
        pub pool: value,
        pub pages: scalar(u32),
        pub page_stride: scalar(u64),
    }
}

crate::scratch_record! {
    /// A latent KV write's buffers (`LatentAttentionQuery::KvWrite`): the
    /// BF16 `latent` rows, and per row its I32 sequence (`row_batch`, a
    /// row of `block_tables`) and token position (`positions`).
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LatentKvWriteBuffers {
        pub latent: value,
        pub cache: record(LatentKvCache),
        pub block_tables: record(BlockTables),
        pub row_batch: value,
        pub positions: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A latent attention's buffers (`LatentAttentionQuery::Attention`):
    /// the BF16 absorbed queries `q` `[rows, heads, latent]`, each row's I32
    /// sequence (`row_batch`) and key list `key_lists` `[rows, max_keys]`,
    /// of which the first `budget` entries are live (at most `max_keys`).
    /// A normalized op writes `output`, a partial op `partial`; the other
    /// is absent. `workspace` holds the partials the op sizes in its info.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LatentAttentionBuffers {
        pub q: value,
        pub cache: record(LatentKvCache),
        pub block_tables: record(BlockTables),
        pub row_batch: value,
        pub key_lists: value,
        pub budget: scalar(u32),
        pub output: option,
        pub partial: optional(LatentPartial),
        pub workspace: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// Per-layer native FP8 window, with separate code and UE8M0 scale
    /// views. Sequence s starts at `s * code_stride` bytes in `codes` and
    /// `s * scale_stride` bytes in `scales`. Each holds contiguous slot rows
    /// of `latent` codes and `latent / 32` scales. Lists name physical ring
    /// slots or staged rows, with `key_stride` entries and `budget` live.
    /// Producers supply in-range slots or -1. Readers mask every index
    /// outside `[0, slots)` as a hole, without reading either plane.
    /// Code bases are two-byte aligned; pair-loading backends also require
    /// an even code stride. Scale bases and strides need only byte alignment.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LatentWindowBuffers {
        pub codes: value,
        pub scales: value,
        pub sequences: scalar(u32),
        pub code_stride: scalar(u64),
        pub scale_stride: scalar(u64),
        pub storage: scalar(LatentWindowStorage),
        pub key_lists: value,
        pub key_stride: scalar(u32),
        pub budget: scalar(u32),
    }
}

crate::scratch_record! {
    /// Native compressed FP4 keys, with a separate index space and table.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LatentCompressedBuffers {
        pub cache: record(LatentKvCache),
        pub block_tables: record(BlockTables),
        pub key_lists: value,
        pub key_stride: scalar(u32),
        pub budget: scalar(u32),
    }
}

crate::scratch_record! {
    /// Native attention extension operands. `q` is already tail-rotated
    /// BF16. `sinks` is F32 `[heads]`, present exactly when declared.
    /// `cos_sin` is F32 `[rows, tail_rope_dims/2, 2]`, for the normalized
    /// output's inverse rotation; partials leave it to their final merge.
    /// `compressed` is present exactly when `second_fp4_keys` is set.
    /// Role bases have their scalar element alignment. The normalized
    /// BF16 output base is four-byte aligned for paired stores. Window
    /// plane alignment is stated on [`LatentWindowBuffers`]; compressed
    /// codes and scales need only byte alignment.
    /// The original paged attention buffer record and ABI stay unchanged.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LatentSourcesBuffers {
        pub q: value,
        pub row_batch: value,
        pub window: record(LatentWindowBuffers),
        pub compressed: optional(LatentCompressedBuffers),
        pub sinks: option,
        pub cos_sin: option,
        pub output: option,
        pub partial: optional(LatentPartial),
        pub workspace: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// An unnormalized attention partial (`LatentAttentionOutput::Partial`):
    /// F32 running maximum `max` and denominator `den` `[rows, heads]`, and
    /// F32 numerator `num` `[rows, heads, latent]`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LatentPartial {
        pub max: value,
        pub den: value,
        pub num: value,
    }
}

crate::scratch_record! {
    /// A head-batched GEMM's buffers (`HeadBatchedGemmQuery`): BF16 `x`
    /// `[rows, heads, k]`, the BF16 `weight` whose head h starts
    /// `h * head_stride` elements in (a stride of at least one head's
    /// `n * k`), and `y` `[rows, heads, n]` of the query's output dtype.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct HeadGemmBuffers {
        pub x: value,
        pub weight: value,
        pub head_stride: scalar(u64),
        pub y: value,
        pub live_rows: value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attention_with(
        phase: AttentionPhase,
        kv: LatentKvFormat,
        output: LatentAttentionOutput,
    ) -> LatentAttentionQuery {
        LatentAttentionQuery::Attention {
            heads: 16,
            latent: 512,
            max_keys: 2051,
            softmax_scale_bits: (1.0_f32 / 16.0).to_bits(),
            page_size: 64,
            kv,
            phase,
            output,
            sink: false,
            tail_rope_dims: 0,
            second_fp4_keys: false,
            capacity: 256,
        }
    }

    fn attention(phase: AttentionPhase, kv: LatentKvFormat) -> LatentAttentionQuery {
        attention_with(phase, kv, LatentAttentionOutput::Normalized)
    }

    #[test]
    fn served_shapes_validate_in_every_phase_format_and_output() {
        for kv in [LatentKvFormat::Fp8E4M3Block128, LatentKvFormat::Bf16] {
            for phase in [
                AttentionPhase::Prefill,
                AttentionPhase::Decode,
                AttentionPhase::Verify,
            ] {
                let query = attention(phase, kv);
                let partial = attention_with(phase, kv, LatentAttentionOutput::Partial);
                query.validate().unwrap();
                partial.validate().unwrap();
                assert_ne!(
                    query.numerical_contract_id().unwrap(),
                    partial.numerical_contract_id().unwrap()
                );
            }
            let write = LatentAttentionQuery::KvWrite {
                latent: 512,
                page_size: 64,
                kv,
                capacity: 8192,
            };
            write.validate().unwrap();
            assert_eq!(write.kv_dtype(), kv.dtype());
        }
        assert_eq!(
            attention(AttentionPhase::Decode, LatentKvFormat::Fp8E4M3Block128)
                .numerical_contract_id()
                .unwrap()
                .as_str(),
            "attention.latent-paged-kv-fp8-e4m3-block128-gathered.decode.v1"
        );
        assert_eq!(
            attention_with(
                AttentionPhase::Verify,
                LatentKvFormat::Bf16,
                LatentAttentionOutput::Partial
            )
            .numerical_contract_id()
            .unwrap()
            .as_str(),
            "attention.latent-paged-kv-bf16-gathered.verify.partial-lse.v1"
        );
    }

    #[test]
    fn latent_token_and_page_bytes_are_exact() {
        assert_eq!(LatentKvFormat::Fp8E4M3Block128.token_bytes(512), 528);
        assert_eq!(LatentKvFormat::Bf16.token_bytes(512), 1024);
        assert_eq!(
            LatentKvFormat::Fp8E4M3Block128.page_bytes(512, 64),
            64 * 528
        );
        assert_eq!(LatentKvFormat::Bf16.page_bytes(512, 64), 65_536);
    }

    #[test]
    fn extensions_have_distinct_contracts_and_refuse_unknown_tail_widths() {
        let base = attention(AttentionPhase::Decode, LatentKvFormat::Bf16);
        let mut sink = base;
        if let LatentAttentionQuery::Attention { sink, .. } = &mut sink {
            *sink = true;
        }
        sink.validate().unwrap();
        assert_ne!(
            base.numerical_contract_id().unwrap(),
            sink.numerical_contract_id().unwrap()
        );
        let mut tail = base;
        if let LatentAttentionQuery::Attention { tail_rope_dims, .. } = &mut tail {
            *tail_rope_dims = 64;
        }
        tail.validate().unwrap();
        assert_ne!(
            base.numerical_contract_id().unwrap(),
            tail.numerical_contract_id().unwrap()
        );
        if let LatentAttentionQuery::Attention { tail_rope_dims, .. } = &mut tail {
            *tail_rope_dims = 32;
        }
        assert_eq!(
            tail.validate(),
            Err(Refusal(
                "latent attention tail RoPE supports 0 or 64 in-place dimensions"
            ))
        );
    }

    #[test]
    fn invalid_geometry_refuses() {
        let bad_latent = LatentAttentionQuery::KvWrite {
            latent: 500,
            page_size: 64,
            kv: LatentKvFormat::Bf16,
            capacity: 1,
        };
        assert_eq!(
            bad_latent.validate(),
            Err(Refusal("invalid latent KV write geometry"))
        );
        let zero_scale = LatentAttentionQuery::Attention {
            heads: 16,
            latent: 512,
            max_keys: 2051,
            softmax_scale_bits: 0,
            page_size: 64,
            kv: LatentKvFormat::Bf16,
            phase: AttentionPhase::Prefill,
            output: LatentAttentionOutput::Normalized,
            sink: false,
            tail_rope_dims: 0,
            second_fp4_keys: false,
            capacity: 1,
        };
        assert_eq!(
            zero_scale.validate(),
            Err(Refusal("invalid latent attention geometry or scale"))
        );
    }

    #[test]
    fn attention_roles_are_named_by_path() {
        use crate::allocation::Traversal;

        let cache = LatentKvCache {
            pool: 2,
            pages: 4,
            page_stride: 65_536,
        };
        let tables = BlockTables {
            tables: 3,
            sequences: 1,
            max_pages: 4,
        };
        let buffers = LatentAttentionBuffers {
            q: 1,
            cache,
            block_tables: tables,
            row_batch: 4,
            key_lists: 5,
            budget: 64,
            output: None,
            partial: Some(LatentPartial {
                max: 6,
                den: 7,
                num: 8,
            }),
            workspace: 9,
            live_rows: 10,
        };
        let mut names = Vec::new();
        buffers.visit_named(&mut |name, &handle| names.push((name.to_owned(), handle)));
        let expected = [
            "q",
            "cache.pool",
            "block_tables.tables",
            "row_batch",
            "key_lists",
            "partial.max",
            "partial.den",
            "partial.num",
            "workspace",
            "live_rows",
        ];
        assert_eq!(
            names,
            expected
                .iter()
                .zip(1..)
                .map(|(name, handle)| (name.to_string(), handle))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn absorbed_projections_are_head_batched_bf16_gemms() {
        let key = HeadBatchedGemmQuery {
            heads: 16,
            n: 512,
            k: 256,
            layout: HeadWeightLayout::KByN,
            output: DType::Bf16,
            m_capacity: 8192,
        };
        let value = HeadBatchedGemmQuery {
            n: 256,
            k: 512,
            layout: HeadWeightLayout::NByK,
            ..key
        };
        key.validate().unwrap();
        value.validate().unwrap();
        assert_eq!(
            key.numerical_contract_id().unwrap().as_str(),
            "gemm.head-batched-w-bf16-kn.a-bf16.acc-f32-chain128.out-bf16.v1"
        );
        assert_ne!(
            key.numerical_contract_id().unwrap(),
            value.numerical_contract_id().unwrap()
        );
        // MMA shapes are a backend's limit, not the contract's.
        assert_eq!(HeadBatchedGemmQuery { k: 100, ..key }.validate(), Ok(()));
        assert_eq!(
            HeadBatchedGemmQuery { k: 0, ..key }.validate(),
            Err(Refusal("head-batched GEMM dimensions must be nonzero"))
        );
    }

    #[test]
    fn hardware_limits_are_not_contract_rules() {
        let LatentAttentionQuery::Attention {
            max_keys,
            softmax_scale_bits,
            kv,
            phase,
            output,
            capacity,
            ..
        } = attention(AttentionPhase::Decode, LatentKvFormat::Bf16)
        else {
            unreachable!()
        };
        // Any head count, any width in 128-wide blocks and any page size.
        let wide = LatentAttentionQuery::Attention {
            heads: 136,
            latent: 2048,
            max_keys,
            softmax_scale_bits,
            page_size: 48,
            kv,
            phase,
            output,
            sink: false,
            tail_rope_dims: 0,
            second_fp4_keys: false,
            capacity,
        };
        assert_eq!(wide.validate(), Ok(()));
        let write = LatentAttentionQuery::KvWrite {
            latent: 2048,
            page_size: 48,
            kv,
            capacity: 1,
        };
        assert_eq!(write.validate(), Ok(()));
    }
}
