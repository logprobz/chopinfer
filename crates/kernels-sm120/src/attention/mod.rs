//! Attention instantiations and capacity-safe launch planning for SM120.

pub mod qualification;
pub mod sparse;

use std::path::{Path, PathBuf};

use crate::invariance::{Invariance, Rows};
use crate::{Catalogs, Profile};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, AttentionPhase, AttentionQuery, CompileKey, CompileKnobs, DType,
    InstantiationSpec, LaunchPlan, OpFamily, Refusal, RuntimeCounts, Target, TensorMap,
    TensorMapElement, TensorMapInterleave, TensorMapL2Promotion, TensorMapOobFill,
    TensorMapSwizzle, Validate,
};
use serde::Deserialize;

pub mod gated;

const SMS: u32 = 188;
const DECODE_THREADS: u32 = 128;
const PREFILL_THREADS: u32 = 256;
const MERGE_THREADS: u32 = 256;
/// A paged prefill row's keys split into at most this many segments, at
/// multiples of a length that is a multiple of PREFILL_SEGMENT_KEYS
/// (`Prefill::max_pieces` and `segment_keys` in cuda/attention/prefill.cuh).
const PREFILL_SEGMENTS: u32 = 4;
const PREFILL_SEGMENT_KEYS: u32 = 256;
/// Widest decode merge CTA; `decode_merge` in decode.cuh.
const DECODE_MERGE_THREADS: u32 = 512;
/// Most items of one decode or verify step: the kernels' shared row and item
/// tables hold this many (`max_decode_rows` in decode.cuh) and trap above it,
/// so the planner refuses larger buckets before any launch. A variant's
/// `max_batch_capacity` (32 in the catalog) may be raised up to it with
/// [`Variant::with_batch_capacity`].
pub const MAX_BATCH_ROWS: u32 = 256;
/// Refusal for a batch above `MAX_BATCH_ROWS`.
const BATCH_ROWS_REFUSAL: &str =
    "attention batch exceeds the 256-item decode and verify tables (MAX_BATCH_ROWS)";
/// Rows of one item the exact verify rows entries give one-row bits: the
/// verify items of up to eight rows.
pub const EXACT_ROWS: u32 = 8;

/// Most query tokens of one verify step: the partial workspace and merge
/// grid hold this many token rows, packed in item order.
pub const MAX_VERIFY_TOKENS: u32 = 256;
/// Bytes of one verify item in the partial entry's shared table
/// (`VerifyEntry` in verify.cuh).
const VERIFY_ITEM_BYTES: u32 = 28;

/// Whether `batch` is a prepared batch bucket: a power of two up to
/// `MAX_BATCH_ROWS`.
pub fn is_batch_bucket(batch: u32) -> bool {
    batch.is_power_of_two() && batch <= MAX_BATCH_ROWS
}

pub use kernel_core::contract::AttentionLayout as CacheLayout;

/// Head widths the attention entries instantiate.
pub const HEAD_DIMS: [u32; 4] = [64, 128, 192, 256];

/// SM120's limits on any attention query: its entries instantiate the
/// widths of [`HEAD_DIMS`] and read rings of BF16 only.
pub fn check_query(query: &AttentionQuery) -> Result<(), Refusal> {
    if !HEAD_DIMS.contains(&query.qk_dim) || !HEAD_DIMS.contains(&query.v_dim) {
        return Err(Refusal("unsupported attention head dimension"));
    }
    if query.layout == CacheLayout::Ring && query.kv_dtype != DType::Bf16 {
        return Err(Refusal("ring KV requires BF16"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub name: &'static str,
    pub profile: Profile,
    pub query: AttentionQuery,
    pub layout: CacheLayout,
    pub query_capacity: u32,
    /// Rotary width of the NeoX RoPE that the decode variant's
    /// `chop_attn_{name}_rope_v1` entry folds in, with the KV write.
    pub rope_fold: Option<u32>,
    /// The gated attention norm and M-RoPE that the decode variant's
    /// `chop_attn_{name}_rope_v1` entry folds in instead, with the KV write.
    pub gated_fold: Option<GatedFold>,
    /// The phase entries' declaration (partial or direct, and merge). The
    /// exact verify rows entries declare their own (`declared_symbols`).
    pub invariance: Invariance,
    /// Whether a decode variant also has exact verify rows entries
    /// (`with_exact_rows`), which run each row of a verify item as a lone
    /// decode row.
    pub exact_rows: bool,
}

/// Static geometry of the gated attention fold: Q rows hold
/// each head's query followed by its gate, Q and K get the per-head Gemma
/// RMSNorm, then NeoX RoPE over the first `rotary` dimensions with
/// interleaved M-RoPE `sections` read from an F32 cos/sin table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedFold {
    pub rotary: u32,
    pub sections: [u32; 3],
    pub eps: f32,
}

#[derive(Clone, Debug)]
pub struct Buffers {
    pub q: u64,
    pub k_cache: u64,
    pub v_cache: u64,
    pub k_page_stride_bytes: u64,
    pub v_page_stride_bytes: u64,
    pub k_map: Option<TensorMap>,
    pub v_map: Option<TensorMap>,
    pub block_tables: Option<u64>,
    pub slot_ids: Option<u64>,
    pub seq_lens: Option<u64>,
    pub q_start: Option<u64>,
    pub q_len: Option<u64>,
    pub prefix_len: Option<u64>,
    pub live_batch: u64,
    pub k_new: Option<u64>,
    pub v_new: Option<u64>,
    pub sink: Option<u64>,
    pub output: u64,
    pub workspace: u64,
    pub k_scale: f32,
    pub v_scale: f32,
}

/// Runtime inputs of RoPE folded into decode. `qkv` holds the packed rows
/// `[row][q_heads * qk | kv_heads * qk | kv_heads * v]` and replaces
/// `Buffers::q`; the K and V scales quantize an FP8 cache's current row.
#[derive(Clone, Copy, Debug)]
pub struct RopeInputs {
    pub qkv: u64,
    pub theta: f32,
    pub v_multiplier: f32,
}

/// Runtime inputs of the gated fold. `qkv` holds the packed rows
/// `[row][q_heads * 2 qk | kv_heads * qk | kv_heads * v]` and replaces
/// `Buffers::q`; the norm weights are BF16 `[qk]`, `rope_positions` I32
/// `[row][3]` (T, H, W) and `cos_sin` F32 `[position][rotary / 2][2]`.
#[derive(Clone, Copy, Debug)]
pub struct GatedRopeInputs {
    pub qkv: u64,
    pub q_norm: u64,
    pub k_norm: u64,
    pub rope_positions: u64,
    pub cos_sin: u64,
}

/// A variant's choices for one prepared bucket ([`Variant::choose`]), which
/// [`Variant::plan_chosen`] binds to a step's buffers.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Choice {
    pub(crate) prepared_batch: u32,
    pub(crate) partial: LaunchPlan,
    pub(crate) merge: LaunchPlan,
    pub(crate) splits: u32,
    pub(crate) split_target: u32,
    pub(crate) split_units: Option<u32>,
    pub(crate) key_tile: u32,
    pub(crate) direct_output: bool,
}

#[derive(Clone, Debug)]
pub struct PlannedAttention {
    pub partial: LaunchPlan,
    pub merge: LaunchPlan,
    pub partial_args: ArgumentList,
    pub merge_args: ArgumentList,
    pub split_capacity: u32,
    pub partial_symbol: String,
    /// The merge entry, which a direct-output plan does not launch.
    pub merge_symbol: String,
    pub direct_output: bool,
    /// The partial entry rotates Q and writes each row's current K and V to
    /// the cache, so no RoPE and KV write runs before it.
    pub writes_cache: bool,
}

fn pointer(value: Option<u64>) -> Argument {
    value.map_or(Argument::NullDevicePointer, Argument::DeviceBuffer)
}

/// Keys per prefill or decode shared-memory stage; wide heads halve it to fit
/// two stages.
fn key_tile(query: &AttentionQuery) -> u32 {
    if query.qk_dim + query.v_dim <= 320 {
        tuning().prefill_key_tile
    } else {
        tuning().prefill_key_tile / 2
    }
}

/// Warps per windowed decode CTA: one per 16 positions of the window, so
/// one CTA owns a row and no merge runs.
fn decode_window_warps(query: &AttentionQuery) -> u32 {
    query.window_left.map_or(1, |left| (left + 1).div_ceil(16))
}

/// Dynamic shared bytes of a windowed decode CTA: the window's BF16 K and V.
fn decode_window_bytes(query: &AttentionQuery) -> u32 {
    decode_window_warps(query) * 16 * (query.qk_dim + query.v_dim) * 2
}

/// Decode<T> folds each key tile's max, denominator and V partials into
/// its K tile. Keep this aligned with decode.cuh's key tile scratch assert.
fn decode_fold_fits(query: &AttentionQuery) -> bool {
    let keys = if query.qk_dim + query.v_dim <= 320 {
        32
    } else {
        16
    };
    let tiles = keys / 16;
    let parts = 4 / tiles;
    let columns = query.v_dim / parts;
    let group_pad = (query.q_heads / query.kv_heads).max(8);
    let fold_floats = (2 + columns) * group_pad;
    tiles == 1 || parts * fold_floats * 4 <= keys * query.qk_dim * 2
}

/// Rows per CTA of the shared-key exact rows entry (`decode_rows_group` in
/// decode_rows.cuh), four warps each.
const ROWS_SHARED_GROUP: u32 = 4;

/// Dynamic shared bytes of the shared-key exact rows entry
/// (`DecodeRowsGroup<T, Group>`): decode's two K/V stages (BF16 tiles, or raw
/// FP8 rows and the BF16 widen tile) and each row's Q stage.
fn rows_shared_bytes(query: &AttentionQuery) -> u32 {
    let keys = if query.qk_dim + query.v_dim <= 320 {
        32
    } else {
        16
    };
    let group_pad = (query.q_heads / query.kv_heads).max(8);
    4 * keys * (query.qk_dim + query.v_dim) + ROWS_SHARED_GROUP * group_pad * query.qk_dim * 2
}

/// Rows of one verify item that span at most two key units, as the
/// shared-key rows entry needs, at any unit width.
const ROWS_SHARED_ITEM_ROWS: u32 = 16;

/// Warps per verify CTA: one per 16 of an item's grouped query rows.
fn verify_warps(query: &AttentionQuery, rows: u32) -> u32 {
    rows * (query.q_heads / query.kv_heads) / 16
}

/// Dynamic shared bytes of a verify CTA for steps of up to `items` items:
/// two K/V stages (plus the FP8 widen buffer), the item's grouped Q rows and
/// the item table, which holds at least 32 items (`Verify<T, Rows>` in
/// verify.cuh). The kernel sizes its table from the launch's bytes, so a
/// bucket of at most 32 items launches with the same bytes at any capacity.
fn verify_bytes(query: &AttentionQuery, rows: u32, items: u32) -> u32 {
    let keys = if query.qk_dim + query.v_dim <= 320 {
        32
    } else {
        16
    };
    let tile = keys * (query.qk_dim + query.v_dim);
    let stages = if query.kv_dtype == DType::Fp8E4M3 {
        2 * tile + 2 * tile
    } else {
        2 * 2 * tile
    };
    let tables = VERIFY_ITEM_BYTES * items.max(32);
    stages + rows * (query.q_heads / query.kv_heads) * query.qk_dim * 2 + tables
}

/// Piece slots per verify item: fewer for wider buckets, so the workspace
/// stays near 17 to 34 MiB for its token rows. In a bucket of 32 or more
/// items of at most 128 window keys, each item is one piece whose CTA writes
/// its output, so no merge runs; smaller buckets gain more from splitting.
fn verify_split_limit(query: &AttentionQuery, prepared_batch: u32) -> u32 {
    if prepared_batch >= 32 && query.window_left.is_some_and(|left| left < 128) {
        1
    } else {
        (256 / prepared_batch).max(16)
    }
}

/// Query token rows of a verify bucket of `prepared_batch` items: the
/// partial workspace and merge grid rows.
fn verify_tokens(query_capacity: u32, prepared_batch: u32) -> u32 {
    (prepared_batch * query_capacity).min(MAX_VERIFY_TOKENS)
}

/// Query tokens per prefill CTA: 128 grouped rows hold whole GQA groups, a
/// group that does not divide 16 padded to eight rows.
fn prefill_tokens(query: &AttentionQuery) -> u32 {
    let group = query.q_heads / query.kv_heads;
    tuning().prefill_rows / if 16 % group == 0 { group } else { 8 }
}

/// Persistent paged prefill CTAs per KV head; together they fill the SMs.
fn prefill_ctas(query: &AttentionQuery) -> u32 {
    tuning().prefill_ctas.div_ceil(query.kv_heads)
}

impl Variant {
    pub fn cache_maps(
        self,
        pool_slots: u32,
        k_buffer: u64,
        v_buffer: u64,
    ) -> Result<(TensorMap, TensorMap), Refusal> {
        let bytes = if self.query.kv_dtype == DType::Bf16 {
            2
        } else {
            1
        };
        let tokens = u64::from(self.query.page_size) * u64::from(self.query.kv_heads);
        self.cache_maps_with_strides(
            pool_slots,
            k_buffer,
            v_buffer,
            tokens * u64::from(self.query.qk_dim) * bytes,
            tokens * u64::from(self.query.v_dim) * bytes,
        )
    }

    pub fn cache_maps_with_strides(
        self,
        pool_slots: u32,
        k_buffer: u64,
        v_buffer: u64,
        k_page_stride_bytes: u64,
        v_page_stride_bytes: u64,
    ) -> Result<(TensorMap, TensorMap), Refusal> {
        self.validate()?;
        self.maps(
            key_tile(&self.query),
            pool_slots,
            k_buffer,
            v_buffer,
            k_page_stride_bytes,
            v_page_stride_bytes,
        )
    }

    /// The cache maps of a validated variant whose prefill key tile is
    /// `key_tile` (`cache_maps_with_strides`).
    pub(crate) fn maps(
        self,
        key_tile: u32,
        pool_slots: u32,
        k_buffer: u64,
        v_buffer: u64,
        k_page_stride_bytes: u64,
        v_page_stride_bytes: u64,
    ) -> Result<(TensorMap, TensorMap), Refusal> {
        if pool_slots == 0 {
            return Err(Refusal("KV pool needs at least one page or slot"));
        }
        let element = if self.query.kv_dtype == DType::Bf16 {
            TensorMapElement::Bf16
        } else {
            TensorMapElement::Fp8E4M3
        };
        let bytes = if self.query.kv_dtype == DType::Bf16 {
            2
        } else {
            1
        };
        let page_elements = u64::from(self.query.page_size) * u64::from(self.query.kv_heads);
        let k_block = page_elements * u64::from(self.query.qk_dim) * bytes;
        let v_block = page_elements * u64::from(self.query.v_dim) * bytes;
        if self.layout == CacheLayout::Paged
            && (k_page_stride_bytes < k_block
                || v_page_stride_bytes < v_block
                || k_page_stride_bytes % 16 != 0
                || v_page_stride_bytes % 16 != 0)
        {
            return Err(Refusal(
                "paged KV page strides must cover and align each page block",
            ));
        }
        let make = |base_buffer, dim, page_stride| TensorMap {
            base_buffer,
            dims: vec![
                u64::from(dim),
                u64::from(self.query.kv_heads),
                u64::from(self.query.page_size),
                u64::from(pool_slots),
            ],
            strides_bytes: vec![
                bytes,
                u64::from(dim) * bytes,
                u64::from(dim) * u64::from(self.query.kv_heads) * bytes,
                page_stride,
            ],
            box_dims: vec![
                if self.layout == CacheLayout::Paged && self.query.kv_dtype == DType::Bf16 {
                    64
                } else {
                    dim
                },
                if self.layout == CacheLayout::Ring {
                    self.query.kv_heads
                } else {
                    1
                },
                if self.layout == CacheLayout::Paged {
                    self.query.page_size.min(key_tile)
                } else {
                    16
                },
                1,
            ],
            element_strides: vec![1; 4],
            swizzle: if self.layout == CacheLayout::Paged && self.query.kv_dtype == DType::Bf16 {
                TensorMapSwizzle::B128
            } else {
                TensorMapSwizzle::None
            },
            interleave: TensorMapInterleave::None,
            l2_promotion: TensorMapL2Promotion::B256,
            oob_fill: TensorMapOobFill::Zero,
            element,
        };
        Ok((
            make(k_buffer, self.query.qk_dim, k_page_stride_bytes),
            make(v_buffer, self.query.v_dim, v_page_stride_bytes),
        ))
    }

    pub fn validate(self) -> Result<(), Refusal> {
        self.query.validate()?;
        check_query(&self.query)?;
        let table = tuning();
        if table.compute_capability != 120
            || table.sm_count != SMS
            || table.prefill_rows != 128
            || table.prefill_key_tile != 32
            || table.prefill_ctas == 0
            || table.decode_buckets.iter().any(|row| {
                !(1..=512).contains(&row.split_limit)
                    || !(1..=row.split_limit).contains(&row.split_units)
                    || ![32, 64, 128, 256, 512].contains(&row.merge_threads)
            })
            || (0..=MAX_BATCH_ROWS.trailing_zeros()).any(|shift| {
                !table
                    .decode_buckets
                    .iter()
                    .any(|row| row.batch == 1 << shift)
            })
            || table.decode_ctas == 0
        {
            return Err(Refusal("attention tuning does not match SM120 traits"));
        }
        if self.layout != self.query.layout {
            return Err(Refusal("attention layout differs from its query"));
        }
        if self.query.q_heads / self.query.kv_heads > 16 {
            return Err(Refusal("attention GQA group exceeds 16 Q heads"));
        }
        let group = self.query.q_heads / self.query.kv_heads;
        if group > 8 && group != 16 {
            return Err(Refusal(
                "attention GQA group must be at most 8 or 16 Q heads",
            ));
        }
        if self.query.phase == AttentionPhase::Decode
            && self.layout == CacheLayout::Paged
            && !decode_fold_fits(&self.query)
        {
            return Err(Refusal(
                "paged decode key tile partials exceed K tile fold scratch",
            ));
        }
        if self.query.max_batch_capacity > MAX_BATCH_ROWS {
            return Err(Refusal(BATCH_ROWS_REFUSAL));
        }
        let verify = self.query.phase == AttentionPhase::Verify;
        if !(self.query.causal || verify && self.layout == CacheLayout::Ring) {
            return Err(Refusal(
                "attention requires causal masking outside windowed verify",
            ));
        }
        if verify
            && (self.query_capacity > 8
                || (self.query_capacity * (self.query.q_heads / self.query.kv_heads)) % 16 != 0
                || verify_bytes(
                    &self.query,
                    self.query_capacity,
                    self.query.max_batch_capacity,
                ) > 96 * 1024)
        {
            return Err(Refusal(
                "verify items hold at most 8 tokens whose grouped rows fill m16 tiles within 96 KiB",
            ));
        }
        if self.query.max_context_capacity > 1_048_576
            || self.query_capacity == 0
            || self.query_capacity > 8192
        {
            return Err(Refusal(
                "attention prepared capacity exceeds catalog limits",
            ));
        }
        match self.layout {
            CacheLayout::Paged if ![16, 32, 64].contains(&self.query.page_size) => {
                return Err(Refusal("paged KV page size must be 16, 32 or 64"));
            }
            // Paged prefill segments count a row's keys from position zero.
            CacheLayout::Paged
                if self.query.phase == AttentionPhase::Prefill
                    && self.query.window_left.is_some() =>
            {
                return Err(Refusal("paged prefill attends to every earlier key"));
            }
            CacheLayout::Ring
                if !(self.query.page_size == 128 || verify && self.query.page_size == 2048)
                    || self
                        .query
                        .window_left
                        .is_none_or(|w| w >= self.query.page_size)
                    || self.query.kv_dtype != DType::Bf16 =>
            {
                return Err(Refusal(
                    "ring KV requires BF16, 128 slots (2048 for verify) and a window within them",
                ));
            }
            _ => {}
        }
        if self.query.phase == AttentionPhase::Decode
            && self.layout == CacheLayout::Ring
            && (self.query.q_heads != 8 * self.query.kv_heads
                || self.query.softcap.is_some()
                || decode_window_bytes(&self.query) > 96 * 1024)
        {
            return Err(Refusal(
                "windowed decode needs eight Q heads per KV head, no softcap and a window within 96 KiB",
            ));
        }
        if self.query.phase == AttentionPhase::Decode && self.query_capacity != 1 {
            return Err(Refusal("decode has one query token per sequence"));
        }
        if self.rope_fold.is_some_and(|rotary| {
            self.query.phase != AttentionPhase::Decode
                || rotary != 64
                || self.query.qk_dim != 192
                || self.query.v_dim != 128
        }) {
            return Err(Refusal(
                "folded RoPE needs decode with 64 rotary of 192 K and 128 V dimensions",
            ));
        }
        if self.gated_fold.is_some_and(|fold| {
            self.rope_fold.is_some()
                || self.query.phase != AttentionPhase::Decode
                || self.layout != CacheLayout::Paged
                || ![128, 256].contains(&self.query.qk_dim)
                || self.query.v_dim != self.query.qk_dim
                || ![32, 64].contains(&fold.rotary)
                || fold.sections.iter().sum::<u32>() != fold.rotary / 2
                || !fold.eps.is_finite()
                || fold.eps <= 0.0
        }) {
            return Err(Refusal(
                "gated fold needs paged decode of 128- or 256-wide heads with 32 or 64 rotary dimensions in M-RoPE sections",
            ));
        }
        Ok(())
    }

    /// This variant prepared for steps of up to `items` items (decode rows or
    /// verify items), at most `MAX_BATCH_ROWS`; its largest bucket is the
    /// power of two at or above `items`. It plans the same kernels.
    pub fn with_batch_capacity(self, items: u32) -> Result<Self, Refusal> {
        if items == 0 || items > MAX_BATCH_ROWS {
            return Err(Refusal(BATCH_ROWS_REFUSAL));
        }
        let mut variant = self;
        variant.query.max_batch_capacity = items.next_power_of_two();
        variant.validate()?;
        Ok(variant)
    }

    pub fn split_capacity(self, prepared_batch: u32) -> Result<u32, Refusal> {
        if prepared_batch > MAX_BATCH_ROWS {
            return Err(Refusal(BATCH_ROWS_REFUSAL));
        }
        if !is_batch_bucket(prepared_batch) || prepared_batch > self.query.max_batch_capacity {
            return Err(Refusal("unsupported prepared batch bucket"));
        }
        Ok(match (self.query.phase, self.layout) {
            (AttentionPhase::Prefill, CacheLayout::Ring) => 1,
            (AttentionPhase::Prefill, CacheLayout::Paged) => PREFILL_SEGMENTS,
            (AttentionPhase::Decode, _) => tuning().bucket(prepared_batch).split_limit,
            (AttentionPhase::Verify, _) => verify_split_limit(&self.query, prepared_batch),
        })
    }

    pub fn plan(self, live: RuntimeCounts, buffers: Buffers) -> Result<PlannedAttention, Refusal> {
        self.plan_for_bucket(self.query.max_batch_capacity, live, buffers)
    }

    pub fn plan_for_bucket(
        self,
        prepared_batch: u32,
        live: RuntimeCounts,
        buffers: Buffers,
    ) -> Result<PlannedAttention, Refusal> {
        self.plan_inner(prepared_batch, live, buffers, None)
    }

    /// Whether this decode variant has exact verify rows entries
    /// ([`Variant::exact_rows`]).
    pub fn has_rows(self) -> bool {
        self.query.phase == AttentionPhase::Decode && self.exact_rows
    }

    /// Plans exact verify rows over a verify step of up to `rows` token rows:
    /// each token row runs as the lone row of a decode step at its position,
    /// with that step's plan (bucket 1's split capacity, CTAs and merge
    /// threads), so its output bits are those of the unfused decode entry.
    /// `buffers` are a verify step's: `q` holds rotated query rows, the item
    /// inputs describe the rows and, for the ring, `k_new`/`v_new` hold the
    /// step's own keys, which the ring receives after attention. `live`
    /// counts are those of one decode row; the kernels read the step's.
    pub fn plan_rows(
        self,
        rows: u32,
        live: RuntimeCounts,
        buffers: Buffers,
    ) -> Result<PlannedAttention, Refusal> {
        self.check_rows(rows, &buffers)?;
        let one = RuntimeCounts { batch: 1, ..live };
        let planned = self.plan_inner(1, one, buffers, None)?;
        self.spread_rows(planned, rows)
    }

    /// Refuses exact verify rows this variant cannot plan over `buffers`.
    pub(crate) fn check_rows(self, rows: u32, buffers: &Buffers) -> Result<(), Refusal> {
        if !self.has_rows() {
            return Err(Refusal(
                "exact verify rows need a decode variant with exact rows",
            ));
        }
        if rows == 0 || rows > MAX_VERIFY_TOKENS {
            return Err(Refusal("exact verify rows exceed the verify token rows"));
        }
        if buffers.q_start.is_none()
            || buffers.q_len.is_none()
            || buffers.prefix_len.is_none()
            || self.layout == CacheLayout::Ring
                && (buffers.k_new.is_none() || buffers.v_new.is_none())
        {
            return Err(Refusal("exact verify rows need the verify item inputs"));
        }
        Ok(())
    }

    /// The one-row decode plan `planned` spread over `rows` exact rows.
    pub(crate) fn spread_rows(
        self,
        mut planned: PlannedAttention,
        rows: u32,
    ) -> Result<PlannedAttention, Refusal> {
        if planned.direct_output {
            // Windowed: one CTA per token row and KV head.
            planned.partial.grid[0] = rows;
        } else {
            planned.partial.grid[2] = rows;
            planned.merge.grid[2] = rows;
            planned.partial.workspace_bytes *= u64::from(rows);
            planned.merge.workspace_bytes = planned.partial.workspace_bytes;
        }
        planned.partial.validate()?;
        planned.merge.validate()?;
        planned.partial_symbol = self.rows_symbol();
        planned.merge_symbol = self.rows_merge_symbol();
        Ok(planned)
    }

    /// Plans exact verify rows of a paged decode variant with the shared-key
    /// entry: the same partials as `plan_rows`, with CTA (group, j) running
    /// up to `ROWS_SHARED_GROUP` rows of an item that share the one-row plan
    /// over its range j, whose keys it reads once. Items hold at most
    /// `ROWS_SHARED_ITEM_ROWS` rows.
    pub fn plan_rows_shared(
        self,
        rows: u32,
        live: RuntimeCounts,
        buffers: Buffers,
    ) -> Result<PlannedAttention, Refusal> {
        self.check_rows_shared(rows)?;
        let planned = self.plan_rows(rows, live, buffers)?;
        self.share_rows(planned, rows)
    }

    /// Refuses shared-key exact rows this variant cannot plan.
    pub(crate) fn check_rows_shared(self, rows: u32) -> Result<(), Refusal> {
        if self.layout != CacheLayout::Paged || self.query.window_left.is_some() {
            return Err(Refusal(
                "shared-key exact verify rows need full paged attention",
            ));
        }
        if rows > ROWS_SHARED_ITEM_ROWS {
            return Err(Refusal("shared-key exact verify rows exceed 16 rows"));
        }
        Ok(())
    }

    /// The exact rows plan `planned` of `rows` rows on the shared-key entry.
    pub(crate) fn share_rows(
        self,
        mut planned: PlannedAttention,
        rows: u32,
    ) -> Result<PlannedAttention, Refusal> {
        // Grid (group, CTA, KV head): the step never holds more row groups
        // than token rows, and a range's groups launch together.
        let [ctas, kv_heads, _] = planned.partial.grid;
        planned.partial.grid = [rows, ctas, kv_heads];
        planned.partial.block[0] = ROWS_SHARED_GROUP * DECODE_THREADS;
        planned.partial.dynamic_shared_bytes = rows_shared_bytes(&self.query);
        planned.partial.validate()?;
        planned.partial_symbol = self.rows_shared_symbol();
        Ok(planned)
    }

    /// The shared-key exact verify rows partial entry of a paged decode variant.
    pub fn rows_shared_symbol(self) -> String {
        format!("chop_attn_{}_rows_shared_v1", self.name)
    }

    /// The exact verify rows partial (or, windowed, direct) entry.
    pub fn rows_symbol(self) -> String {
        format!("chop_attn_{}_rows_v1", self.name)
    }

    /// The exact verify rows merge entry of a paged decode variant.
    pub fn rows_merge_symbol(self) -> String {
        format!("chop_attn_{}_rows_merge_v1", self.name)
    }

    /// Plans the decode entry that folds RoPE and the KV write in: it reads
    /// `rope.qkv` in place of `buffers.q` and writes the cache.
    pub fn plan_folded(
        self,
        prepared_batch: u32,
        live: RuntimeCounts,
        buffers: Buffers,
        rope: RopeInputs,
    ) -> Result<PlannedAttention, Refusal> {
        if self.rope_fold.is_none() {
            return Err(Refusal("attention variant does not fold RoPE"));
        }
        if !rope.theta.is_finite() || rope.theta <= 1.0 || !rope.v_multiplier.is_finite() {
            return Err(Refusal(
                "folded RoPE needs a finite theta above 1 and V multiplier",
            ));
        }
        self.plan_inner(prepared_batch, live, buffers, Some(Fold::Neox(rope)))
    }

    /// The table-backed NeoX fold preserves the computed entry's ABI.
    /// The caller bounds positions and binds F32 [position][pair][cos, sin].
    pub fn plan_folded_table(
        self,
        prepared_batch: u32,
        live: RuntimeCounts,
        buffers: Buffers,
        rope: RopeInputs,
        cos_sin: u64,
    ) -> Result<PlannedAttention, Refusal> {
        let planned = self.plan_folded(prepared_batch, live, buffers, rope)?;
        Ok(self.with_table(planned, cos_sin))
    }

    /// The computed NeoX fold `planned` on its table-backed entry.
    pub(crate) fn with_table(
        self,
        mut planned: PlannedAttention,
        cos_sin: u64,
    ) -> PlannedAttention {
        planned.partial_symbol = self.rope_table_symbol();
        planned.partial_args.0.push(Argument::DeviceBuffer(cos_sin));
        planned
    }

    /// Plans the decode entry that folds the gated Q/K norm, M-RoPE and the
    /// KV write in: it reads `rope.qkv` in place of `buffers.q` and writes
    /// the cache.
    pub fn plan_folded_gated(
        self,
        prepared_batch: u32,
        live: RuntimeCounts,
        buffers: Buffers,
        rope: GatedRopeInputs,
    ) -> Result<PlannedAttention, Refusal> {
        if self.gated_fold.is_none() {
            return Err(Refusal(
                "attention variant does not fold the gated norm and RoPE",
            ));
        }
        self.plan_inner(prepared_batch, live, buffers, Some(Fold::Gated(rope)))
    }

    fn plan_inner(
        self,
        prepared_batch: u32,
        live: RuntimeCounts,
        buffers: Buffers,
        rope: Option<Fold>,
    ) -> Result<PlannedAttention, Refusal> {
        self.validate()?;
        self.check_launch(prepared_batch, live, &buffers, key_tile(&self.query))?;
        let choice = self.choose(prepared_batch, self.short_prefill(live))?;
        self.assemble(&choice, buffers, rope)
    }

    /// Whether a prefill step's keys are too few to split, so it plans with
    /// one split.
    pub(crate) fn short_prefill(self, live: RuntimeCounts) -> bool {
        self.query.phase == AttentionPhase::Prefill && live.context <= PREFILL_SEGMENT_KEYS
    }

    /// Plans a launch of a validated variant from `choice`, which
    /// [`Variant::choose`] made for its bucket: checks the counts and
    /// buffers and binds them, reading no tuning.
    pub(crate) fn plan_chosen(
        self,
        choice: &Choice,
        live: RuntimeCounts,
        buffers: Buffers,
        rope: Option<Fold>,
    ) -> Result<PlannedAttention, Refusal> {
        self.check_launch(choice.prepared_batch, live, &buffers, choice.key_tile)?;
        self.assemble(choice, buffers, rope)
    }

    /// Refuses live counts past the bucket and buffers the phase cannot
    /// launch over.
    fn check_launch(
        self,
        prepared_batch: u32,
        live: RuntimeCounts,
        buffers: &Buffers,
        key_tile: u32,
    ) -> Result<(), Refusal> {
        if prepared_batch > MAX_BATCH_ROWS || live.batch > MAX_BATCH_ROWS {
            return Err(Refusal(BATCH_ROWS_REFUSAL));
        }
        if !is_batch_bucket(prepared_batch)
            || prepared_batch > self.query.max_batch_capacity
            || live.batch == 0
            || live.batch > prepared_batch
            || live.context == 0
            || live.context > self.query.max_context_capacity
            || (self.query.phase == AttentionPhase::Prefill
                && (live.m == 0 || live.m > self.query_capacity))
            || (self.query.phase == AttentionPhase::Verify
                && (live.m == 0 || live.m > verify_tokens(self.query_capacity, prepared_batch)))
        {
            return Err(Refusal("live attention counts exceed prepared capacity"));
        }
        if !buffers.k_scale.is_finite()
            || buffers.k_scale <= 0.0
            || !buffers.v_scale.is_finite()
            || buffers.v_scale <= 0.0
        {
            return Err(Refusal("KV scales must be positive and finite"));
        }
        if self.layout == CacheLayout::Paged {
            let element_bytes = if self.query.kv_dtype == DType::Bf16 {
                2_u64
            } else {
                1_u64
            };
            let tokens = u64::from(self.query.page_size) * u64::from(self.query.kv_heads);
            let max_pool_page = u64::from(self.query.max_batch_capacity)
                * u64::from(
                    self.query
                        .max_context_capacity
                        .div_ceil(self.query.page_size),
                )
                - 1;
            for (stride, dim) in [
                (buffers.k_page_stride_bytes, self.query.qk_dim),
                (buffers.v_page_stride_bytes, self.query.v_dim),
            ] {
                let block = tokens * u64::from(dim) * element_bytes;
                if stride < block
                    || stride % 16 != 0
                    || max_pool_page
                        .checked_mul(stride)
                        .and_then(|offset| offset.checked_add(block))
                        .is_none()
                {
                    return Err(Refusal("paged KV page stride is too small or overflows"));
                }
            }
        }
        if self.query.phase == AttentionPhase::Prefill {
            let k_map = buffers.k_map.as_ref().ok_or(Refusal("K map missing"))?;
            let v_map = buffers.v_map.as_ref().ok_or(Refusal("V map missing"))?;
            let pool_slots =
                u32::try_from(*k_map.dims.get(3).ok_or(Refusal("K map rank mismatch"))?)
                    .map_err(|_| Refusal("K map pool size exceeds u32"))?;
            let (expected_k, expected_v) = self.maps(
                key_tile,
                pool_slots,
                buffers.k_cache,
                buffers.v_cache,
                buffers.k_page_stride_bytes,
                buffers.v_page_stride_bytes,
            )?;
            if *k_map != expected_k || *v_map != expected_v {
                return Err(Refusal("KV tensor map does not match cache layout"));
            }
        }
        if self.layout == CacheLayout::Paged && buffers.block_tables.is_none()
            || self.layout == CacheLayout::Ring && buffers.slot_ids.is_none()
            || self.query.sink && buffers.sink.is_none()
            || self.query.phase == AttentionPhase::Prefill
                && (buffers.q_start.is_none()
                    || buffers.q_len.is_none()
                    || buffers.prefix_len.is_none()
                    || buffers.k_map.is_none()
                    || buffers.v_map.is_none())
            || self.query.phase == AttentionPhase::Prefill
                && self.layout == CacheLayout::Ring
                && (buffers.k_new.is_none() || buffers.v_new.is_none())
            || self.query.phase == AttentionPhase::Decode && buffers.seq_lens.is_none()
            || self.query.phase == AttentionPhase::Verify
                && (buffers.q_start.is_none()
                    || buffers.q_len.is_none()
                    || buffers.prefix_len.is_none()
                    || self.layout == CacheLayout::Ring
                        && (buffers.k_new.is_none() || buffers.v_new.is_none()))
        {
            return Err(Refusal("attention buffer layout is incomplete"));
        }
        Ok(())
    }

    /// The choices of a validated variant for steps of `prepared_batch`
    /// (and in prefill, `short` steps of one split) from the tuning table:
    /// splits, CTA targets, piece units and the partial and merge plans.
    pub(crate) fn choose(self, prepared_batch: u32, short: bool) -> Result<Choice, Refusal> {
        // A paged prefill row splits its keys only past PREFILL_SEGMENT_KEYS,
        // so shorter steps need no merge.
        let splits = if short {
            1
        } else {
            self.split_capacity(prepared_batch)?
        };
        let prefill = self.query.phase == AttentionPhase::Prefill;
        let verify = self.query.phase == AttentionPhase::Verify;
        // One CTA combines each windowed ring decode row.
        let windowed =
            self.query.phase == AttentionPhase::Decode && self.layout == CacheLayout::Ring;
        let direct_output = (prefill || verify) && splits == 1 || windowed;
        // Paged decode cuts each row, and verify each item, into pieces from
        // its own length.
        let split_units = (self.query.phase == AttentionPhase::Decode && !windowed || verify)
            .then(|| tuning().bucket(prepared_batch).split_units);
        // Prefill, decode and verify run persistent stream-K CTAs per KV
        // head; a verify CTA fills its SM.
        let split_target = if prefill {
            prefill_ctas(&self.query)
        } else if verify {
            SMS.div_ceil(self.query.kv_heads)
        } else {
            tuning().decode_ctas.div_ceil(self.query.kv_heads)
        };
        let query_slots = match self.query.phase {
            AttentionPhase::Decode => prepared_batch,
            AttentionPhase::Verify => verify_tokens(self.query_capacity, prepared_batch),
            AttentionPhase::Prefill => self.query_capacity,
        };
        let workspace = u64::from(query_slots)
            * u64::from(self.query.q_heads)
            * u64::from(splits)
            * u64::from(self.query.v_dim + 2)
            * 4;
        // Paged prefill runs persistent CTAs over query tile segments. Ring
        // prefill gives each
        // query tile a CTA; each batch adds at most one partial tile beyond the
        // packed token capacity.
        let grid = if prefill && self.layout == CacheLayout::Paged {
            [split_target, self.query.kv_heads, 1]
        } else if prefill {
            [
                self.query_capacity.div_ceil(prefill_tokens(&self.query)) + prepared_batch,
                self.query.kv_heads,
                1,
            ]
        } else if windowed {
            [prepared_batch, self.query.kv_heads, 1]
        } else {
            [split_target, self.query.kv_heads, 1]
        };
        let partial = LaunchPlan {
            grid,
            block: [
                if prefill {
                    PREFILL_THREADS
                } else if windowed {
                    32 * decode_window_warps(&self.query)
                } else if verify {
                    32 * verify_warps(&self.query, self.query_capacity)
                } else {
                    DECODE_THREADS
                },
                1,
                1,
            ],
            dynamic_shared_bytes: if windowed {
                decode_window_bytes(&self.query)
            } else if verify {
                verify_bytes(&self.query, self.query_capacity, prepared_batch)
            } else {
                0
            },
            workspace_bytes: workspace,
            cluster: [1, 1, 1],
            pdl: true,
        };
        let merge = LaunchPlan {
            grid: if prefill {
                [prefill_ctas(&self.query), self.query.kv_heads, 1]
            } else {
                [
                    self.query.v_dim.div_ceil(64),
                    self.query.q_heads,
                    if verify {
                        verify_tokens(self.query_capacity, prepared_batch)
                    } else {
                        prepared_batch
                    },
                ]
            },
            block: [
                if prefill {
                    MERGE_THREADS
                } else {
                    tuning().bucket(prepared_batch).merge_threads
                },
                1,
                1,
            ],
            dynamic_shared_bytes: 0,
            workspace_bytes: workspace,
            cluster: [1, 1, 1],
            pdl: true,
        };
        partial.validate()?;
        merge.validate()?;
        Ok(Choice {
            prepared_batch,
            partial,
            merge,
            splits,
            split_target,
            split_units,
            key_tile: key_tile(&self.query),
            direct_output,
        })
    }

    /// The launch of `choice` over `buffers`.
    fn assemble(
        self,
        choice: &Choice,
        buffers: Buffers,
        rope: Option<Fold>,
    ) -> Result<PlannedAttention, Refusal> {
        let Choice {
            partial,
            merge,
            splits,
            split_target,
            split_units,
            direct_output,
            ..
        } = choice.clone();
        let prefill = self.query.phase == AttentionPhase::Prefill;
        let verify = self.query.phase == AttentionPhase::Verify;
        let windowed =
            self.query.phase == AttentionPhase::Decode && self.layout == CacheLayout::Ring;
        let max_pages = self
            .query
            .max_context_capacity
            .div_ceil(self.query.page_size);
        let window = self.query.window_left.unwrap_or(u32::MAX);
        let cap = self.query.softcap.unwrap_or(1.0).to_bits();
        let common = [
            pointer(buffers.seq_lens),
            pointer(buffers.q_start),
            pointer(buffers.q_len),
            pointer(buffers.prefix_len),
            Argument::DeviceBuffer(buffers.live_batch),
        ];
        let mut partial_args = vec![
            Argument::DeviceBuffer(match rope {
                None => buffers.q,
                Some(Fold::Neox(rope)) => rope.qkv,
                Some(Fold::Gated(rope)) => rope.qkv,
            }),
            Argument::DeviceBuffer(buffers.k_cache),
            Argument::DeviceBuffer(buffers.v_cache),
        ];
        if self.query.phase == AttentionPhase::Prefill {
            partial_args.push(Argument::TensorMap(buffers.k_map.unwrap()));
            partial_args.push(Argument::TensorMap(buffers.v_map.unwrap()));
        }
        partial_args.extend([pointer(buffers.block_tables), pointer(buffers.slot_ids)]);
        partial_args.extend(common.clone());
        partial_args.extend([
            pointer(buffers.k_new),
            pointer(buffers.v_new),
            Argument::DeviceBuffer(buffers.workspace),
            Argument::U32(max_pages),
            Argument::U64(buffers.k_page_stride_bytes),
            Argument::U64(buffers.v_page_stride_bytes),
            Argument::U32(splits),
            Argument::U32(split_target),
        ]);
        partial_args.extend(split_units.map(Argument::U32));
        partial_args.extend([
            Argument::U32(window),
            Argument::U32(self.query.softmax_scale.to_bits()),
            Argument::U32(cap),
            Argument::U32(buffers.k_scale.to_bits()),
            Argument::U32(buffers.v_scale.to_bits()),
        ]);
        if prefill || windowed || verify {
            partial_args.push(pointer(buffers.sink));
            partial_args.push(Argument::DeviceBuffer(buffers.output));
        }
        match rope {
            None => {}
            Some(Fold::Neox(rope)) => {
                partial_args.push(Argument::U32(rope.theta.to_bits()));
                partial_args.push(Argument::U32(rope.v_multiplier.to_bits()));
            }
            Some(Fold::Gated(rope)) => {
                partial_args.extend(
                    [rope.q_norm, rope.k_norm, rope.rope_positions, rope.cos_sin]
                        .map(Argument::DeviceBuffer),
                );
                let eps = self.gated_fold.map_or(0.0, |fold| fold.eps);
                partial_args.push(Argument::U32(eps.to_bits()));
            }
        }
        let mut merge_args = vec![
            Argument::DeviceBuffer(buffers.workspace),
            pointer(buffers.sink),
        ];
        merge_args.extend(common);
        merge_args.extend([
            Argument::DeviceBuffer(buffers.output),
            Argument::U32(splits),
            Argument::U32(split_target),
        ]);
        merge_args.extend(split_units.map(Argument::U32));
        merge_args.push(Argument::U32(window));
        let partial_args = ArgumentList(partial_args);
        let merge_args = ArgumentList(merge_args);
        partial_args.validate()?;
        merge_args.validate()?;
        Ok(PlannedAttention {
            partial,
            merge,
            partial_args,
            merge_args,
            split_capacity: splits,
            partial_symbol: if rope.is_some() {
                self.rope_symbol()
            } else if prefill || windowed {
                format!("chop_attn_{}_direct_v1", self.name)
            } else {
                format!("chop_attn_{}_partial_v1", self.name)
            },
            merge_symbol: format!("chop_attn_{}_merge_v1", self.name),
            direct_output,
            writes_cache: rope.is_some(),
        })
    }

    /// Dynamic shared bytes that a launch of `symbol` needs, when `symbol`
    /// is one of this variant's entries: the planned launch's, or for verify
    /// the largest bucket's at the variant's batch capacity (smaller buckets
    /// launch with fewer). Launchers opt in above 48 KiB. Windowed decode and verify keep their
    /// stages in dynamic shared memory; every other entry uses static shared
    /// memory.
    pub fn dynamic_shared_bytes(self, symbol: &str) -> Option<u32> {
        let partial = self.instantiation().ok()?.symbol;
        let folded = self.rope_fold.is_some() || self.gated_fold.is_some();
        if self.has_rows() && (symbol == self.rows_symbol() || symbol == self.rows_merge_symbol()) {
            let windowed = self.layout == CacheLayout::Ring && symbol == self.rows_symbol();
            return Some(if windowed {
                decode_window_bytes(&self.query)
            } else {
                0
            });
        }
        if self.has_rows()
            && self.layout == CacheLayout::Paged
            && symbol == self.rows_shared_symbol()
        {
            return Some(rows_shared_bytes(&self.query));
        }
        if symbol == partial
            || folded
                && (symbol == self.rope_symbol()
                    || self.rope_fold.is_some() && symbol == self.rope_table_symbol())
        {
            let windowed =
                self.query.phase == AttentionPhase::Decode && self.layout == CacheLayout::Ring;
            Some(if windowed {
                decode_window_bytes(&self.query)
            } else if self.query.phase == AttentionPhase::Verify {
                verify_bytes(
                    &self.query,
                    self.query_capacity,
                    self.query.max_batch_capacity,
                )
            } else {
                0
            })
        } else {
            (symbol == format!("chop_attn_{}_merge_v1", self.name)).then_some(0)
        }
    }

    /// The decode entry that folds RoPE (or the gated norm and M-RoPE) and
    /// the KV write in.
    pub fn rope_symbol(self) -> String {
        format!("chop_attn_{}_rope_v1", self.name)
    }

    pub fn rope_table_symbol(self) -> String {
        format!("chop_attn_{}_rope_table_v1", self.name)
    }

    pub fn instantiation(self) -> Result<InstantiationSpec, Refusal> {
        self.validate()?;
        let key = CompileKey {
            query: self.query.into(),
            knobs: CompileKnobs {
                tile_m: match self.query.phase {
                    AttentionPhase::Prefill => tuning().prefill_rows as u16,
                    AttentionPhase::Verify => {
                        (self.query_capacity * self.query.q_heads / self.query.kv_heads) as u16
                    }
                    AttentionPhase::Decode => (self.query.q_heads / self.query.kv_heads) as u16,
                },
                tile_n: if self.query.phase == AttentionPhase::Prefill {
                    key_tile(&self.query) as u16
                } else {
                    16
                },
                tile_k: if self.query.phase == AttentionPhase::Prefill {
                    16
                } else {
                    self.query.qk_dim as u16
                },
                stages: 2,
                swizzle: 0,
                warps: match self.query.phase {
                    AttentionPhase::Prefill => (PREFILL_THREADS / 32) as u8,
                    AttentionPhase::Verify => verify_warps(&self.query, self.query_capacity) as u8,
                    AttentionPhase::Decode => (DECODE_THREADS / 32) as u8,
                },
                split_k_capacity: self.split_capacity(1)?.min(255) as u8,
            },
            dense: None,
            target: Target::Sm120f,
        };
        let spec = InstantiationSpec {
            family: OpFamily::Attention,
            key,
            symbol: if self.query.phase == AttentionPhase::Prefill
                || self.query.phase == AttentionPhase::Decode && self.layout == CacheLayout::Ring
            {
                format!("chop_attn_{}_direct_v1", self.name)
            } else {
                format!("chop_attn_{}_partial_v1", self.name)
            },
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Each entry of this variant's cubin with its declaration: the exact
    /// verify rows entries give every row of up to `EXACT_ROWS` rows its
    /// one-row decode bits (`attention_exact_rows_equal_one_row_decode`); the
    /// folded RoPE entries and a merge that never runs claim nothing; the
    /// phase entries declare `invariance`.
    pub fn declared_symbols(self) -> Vec<(String, Invariance)> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda");
        let rows = [
            self.rows_symbol(),
            self.rows_shared_symbol(),
            self.rows_merge_symbol(),
        ];
        let rope = [self.rope_symbol(), self.rope_table_symbol()];
        // A prefill variant of one split writes its output directly and
        // never launches its merge.
        let merge = format!("chop_attn_{}_merge_v1", self.name);
        let unlaunched_merge =
            self.query.phase == AttentionPhase::Prefill && self.split_capacity(1) == Ok(1);
        self.cubin(&root)
            .expect("catalog cubin")
            .entries
            .into_iter()
            .map(|entry| {
                // A verify item's rows attend to each other's keys, so the
                // rows entries claim one-row bits, not row independence.
                let invariance = if self.has_rows() && rows.contains(&entry.symbol) {
                    Invariance {
                        rows: Rows::OneRow { max: EXACT_ROWS },
                        ..Invariance::NONE
                    }
                } else if rope.contains(&entry.symbol) || unlaunched_merge && entry.symbol == merge
                {
                    Invariance::NONE
                } else {
                    self.invariance
                };
                (entry.symbol, invariance)
            })
            .collect()
    }

    pub fn cubin(self, sources_root: &Path) -> Result<CubinSpec, Refusal> {
        let inst = self.instantiation()?;
        let trait_name = format!(
            "chop::attention::Traits<{},{},{},{},{},{},{},{},{}>",
            self.query.qk_dim,
            self.query.v_dim,
            self.query.q_heads,
            self.query.kv_heads,
            self.query.page_size,
            self.layout == CacheLayout::Ring,
            self.query.kv_dtype == DType::Fp8E4M3,
            self.query.sink,
            self.query.softcap.is_some(),
        );
        let decode = self.query.phase == AttentionPhase::Decode;
        let verify = self.query.phase == AttentionPhase::Verify;
        let paged_decode = decode && self.layout == CacheLayout::Paged;
        let rows = self.query_capacity;
        // Only paged decode and verify merges take `split_units`; the
        // others pass 0.
        let merge_units = if paged_decode { "split_units" } else { "0u" };
        let merge_body = if verify {
            format!(
                "chop::attention::merge_verify<{trait_name}, {rows}>(workspace,sink,seq_lens,q_start,q_len,prefix_len,live_batch,out,split_capacity,split_target,split_units,window_left);"
            )
        } else {
            format!(
                "chop::attention::merge<{trait_name}, {decode}>(workspace,sink,seq_lens,q_start,q_len,prefix_len,live_batch,out,split_capacity,split_target,{merge_units},window_left);"
            )
        };
        let merge = EntrySpec::new(
            format!("chop_attn_{}_merge_v1", self.name),
            if paged_decode || verify {
                with_split_units(merge_params())
            } else {
                merge_params()
            },
            merge_body,
            if decode || verify {
                DECODE_MERGE_THREADS
            } else {
                MERGE_THREADS
            },
            1,
        );
        let mut entries = if verify {
            let causal = self.query.causal;
            vec![
                EntrySpec::new(
                    inst.symbol,
                    {
                        let mut params = with_split_units(partial_params(false));
                        params.push(ptr("sink", "const __nv_bfloat16"));
                        params.push(ptr("out", "__nv_bfloat16"));
                        params
                    },
                    format!(
                        "chop::attention::verify<{trait_name}, {rows}, {causal}>(q,k_cache,v_cache,block_tables,slot_ids,q_start,q_len,prefix_len,live_batch,k_new,v_new,workspace,max_pages,k_page_stride_bytes,v_page_stride_bytes,split_capacity,split_target,split_units,window_left,scale_bits,cap_bits,k_scale_bits,v_scale_bits,sink,out);"
                    ),
                    32 * verify_warps(&self.query, rows),
                    1,
                ),
                merge,
            ]
        } else if decode && self.layout == CacheLayout::Ring {
            let mut params = partial_params(false);
            params.push(ptr("sink", "const __nv_bfloat16"));
            params.push(ptr("out", "__nv_bfloat16"));
            let warps = decode_window_warps(&self.query);
            let mut entries = vec![
                EntrySpec::new(
                    inst.symbol,
                    params.clone(),
                    format!(
                        "chop::attention::decode_window<{trait_name}, {warps}>(q,k_cache,v_cache,slot_ids,seq_lens,live_batch,window_left,scale_bits,sink,out);"
                    ),
                    32 * warps,
                    1,
                ),
                merge,
            ];
            if self.has_rows() {
                entries.push(EntrySpec::new(
                    self.rows_symbol(),
                    params.clone(),
                    format!(
                        "chop::attention::decode_window<{trait_name}, {warps}, chop::attention::NoRope, true>(q,k_cache,v_cache,slot_ids,seq_lens,live_batch,window_left,scale_bits,sink,out,{{}},{{q_start,q_len,prefix_len,k_new,v_new}});"
                    ),
                    32 * warps,
                    1,
                ));
            }
            if let Some(rotary) = self.rope_fold {
                entries.push(EntrySpec::new(
                    self.rope_symbol(),
                    rope_params(params),
                    format!(
                        "chop::attention::decode_window<{trait_name}, {warps}, chop::attention::NeoxRope<{rotary}>>(qkv,k_cache,v_cache,slot_ids,seq_lens,live_batch,window_left,scale_bits,sink,out,{ROPE_ARGS});"
                    ),
                    32 * warps,
                    1,
                ));
            }
            if self.rope_fold.is_some() {
                // The shared double sincos helper uses 40 stack bytes.
                for entry in &mut entries {
                    if entry.symbol == self.rope_symbol() {
                        entry.allowed_local_bytes = 40;
                    }
                }
            }
            entries
        } else if decode {
            let decode_params = || with_split_units(partial_params(false));
            let mut entries = vec![
                EntrySpec::new(
                    inst.symbol,
                    decode_params(),
                    format!(
                        "chop::attention::decode<{trait_name}>(q,k_cache,v_cache,block_tables,slot_ids,seq_lens,live_batch,workspace,max_pages,k_page_stride_bytes,v_page_stride_bytes,split_capacity,split_target,split_units,window_left,scale_bits,cap_bits,k_scale_bits,v_scale_bits);"
                    ),
                    DECODE_THREADS,
                    1,
                ),
                merge,
            ];
            if self.has_rows() {
                entries.push(EntrySpec::new(
                    self.rows_symbol(),
                    decode_params(),
                    format!(
                        "chop::attention::decode_rows<{trait_name}>(q,k_cache,v_cache,block_tables,slot_ids,q_start,q_len,prefix_len,live_batch,workspace,max_pages,k_page_stride_bytes,v_page_stride_bytes,split_capacity,split_target,split_units,window_left,scale_bits,cap_bits,k_scale_bits,v_scale_bits);"
                    ),
                    DECODE_THREADS,
                    1,
                ));
                entries.push(EntrySpec::new(
                    self.rows_shared_symbol(),
                    decode_params(),
                    format!(
                        "chop::attention::decode_rows_group<{trait_name}, {ROWS_SHARED_GROUP}>(q,k_cache,v_cache,block_tables,q_start,q_len,prefix_len,live_batch,workspace,max_pages,k_page_stride_bytes,v_page_stride_bytes,split_capacity,split_target,split_units,scale_bits,cap_bits,k_scale_bits,v_scale_bits);"
                    ),
                    ROWS_SHARED_GROUP * DECODE_THREADS,
                    1,
                ));
                entries.push(EntrySpec::new(
                    self.rows_merge_symbol(),
                    with_split_units(merge_params()),
                    format!(
                        "chop::attention::merge_rows<{trait_name}>(workspace,sink,seq_lens,q_start,q_len,prefix_len,live_batch,out,split_capacity,split_target,split_units,window_left);"
                    ),
                    DECODE_MERGE_THREADS,
                    1,
                ));
            }
            if let Some(rotary) = self.rope_fold {
                entries.push(EntrySpec::new(
                    self.rope_symbol(),
                    rope_params(decode_params()),
                    format!(
                        "chop::attention::decode<{trait_name}, chop::attention::NeoxRope<{rotary}>>(qkv,k_cache,v_cache,block_tables,slot_ids,seq_lens,live_batch,workspace,max_pages,k_page_stride_bytes,v_page_stride_bytes,split_capacity,split_target,split_units,window_left,scale_bits,cap_bits,k_scale_bits,v_scale_bits,{ROPE_ARGS});"
                    ),
                    DECODE_THREADS,
                    1,
                ));
            }
            if let Some(fold) = self.gated_fold {
                entries.push(EntrySpec::new(
                    self.rope_symbol(),
                    gated_rope_params(decode_params()),
                    format!(
                        "chop::attention::decode<{trait_name}, chop::attention::GatedRope<{},{},{}>>(qkv,k_cache,v_cache,block_tables,slot_ids,seq_lens,live_batch,workspace,max_pages,k_page_stride_bytes,v_page_stride_bytes,split_capacity,split_target,split_units,window_left,scale_bits,cap_bits,k_scale_bits,v_scale_bits,{GATED_ROPE_ARGS});",
                        fold.rotary, fold.sections[1], fold.sections[2],
                    ),
                    DECODE_THREADS,
                    1,
                ));
            }
            if self.rope_fold.is_some() {
                // The shared double sincos helper uses 40 stack bytes.
                for entry in &mut entries {
                    if entry.symbol == self.rope_symbol() {
                        entry.allowed_local_bytes = 40;
                    }
                }
            }
            entries
        } else {
            let mut params = partial_params(true);
            params.push(ptr("sink", "const __nv_bfloat16"));
            params.push(ptr("out", "__nv_bfloat16"));
            vec![
                EntrySpec::new(
                    inst.symbol,
                    params,
                    format!(
                        "chop::attention::prefill<{trait_name}>(q,k_cache,v_cache,&k_map,&v_map,block_tables,slot_ids,seq_lens,q_start,q_len,prefix_len,live_batch,k_new,v_new,workspace,max_pages,k_page_stride_bytes,v_page_stride_bytes,split_capacity,split_target,window_left,scale_bits,cap_bits,k_scale_bits,v_scale_bits,sink,out);"
                    ),
                    PREFILL_THREADS,
                    1,
                ),
                merge,
            ]
        };
        if let Some(rotary) = self.rope_fold {
            let mut table = entries
                .iter()
                .find(|entry| entry.symbol == self.rope_symbol())
                .expect("computed fold entry")
                .clone();
            table.symbol = self.rope_table_symbol();
            table.params.push(ptr("cos_sin", "const float"));
            table.body = table.body
                .replace(&format!("NeoxRope<{rotary}>"), &format!("NeoxRope<{rotary}, true>"))
                .replace(ROPE_ARGS, "{k_cache,v_cache,__uint_as_float(theta_bits),__uint_as_float(v_multiplier_bits),cos_sin}");
            table.allowed_local_bytes = 0;
            entries.push(table);
        }
        Ok(CubinSpec {
            name: self.name.into(),
            sources_root: PathBuf::from(sources_root),
            include_dirs: vec![PathBuf::from(".")],
            headers: vec!["#include \"attention/attention.cuh\"".into()],
            defines: vec![],
            entries,
        })
    }
}

fn ptr(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}
fn u32_param(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}
fn partial_params(prefill: bool) -> Vec<EntryParam> {
    let mut p = vec![
        ptr("q", "const __nv_bfloat16"),
        ptr("k_cache", "const void"),
        ptr("v_cache", "const void"),
    ];
    if prefill {
        p.push(EntryParam::TensorMap {
            name: "k_map".into(),
        });
        p.push(EntryParam::TensorMap {
            name: "v_map".into(),
        });
    }
    p.extend([
        ptr("block_tables", "const int32_t"),
        ptr("slot_ids", "const int32_t"),
        ptr("seq_lens", "const int32_t"),
        ptr("q_start", "const int32_t"),
        ptr("q_len", "const int32_t"),
        ptr("prefix_len", "const int32_t"),
        ptr("live_batch", "const int32_t"),
        ptr("k_new", "const __nv_bfloat16"),
        ptr("v_new", "const __nv_bfloat16"),
        ptr("workspace", "float"),
    ]);
    p.extend(
        [
            "max_pages",
            "k_page_stride_bytes",
            "v_page_stride_bytes",
            "split_capacity",
            "split_target",
            "window_left",
            "scale_bits",
            "cap_bits",
            "k_scale_bits",
            "v_scale_bits",
        ]
        .map(|name| {
            if name.ends_with("page_stride_bytes") {
                EntryParam::U64 { name: name.into() }
            } else {
                u32_param(name)
            }
        }),
    );
    p
}
/// The folded entry's trailing `RopeArgs`, from `rope_params`.
const ROPE_ARGS: &str =
    "{k_cache,v_cache,__uint_as_float(theta_bits),__uint_as_float(v_multiplier_bits)}";

/// A decode entry's parameters with RoPE folded in: packed QKV rows in place
/// of `q`, writable caches, then theta and the V multiplier.
fn rope_params(mut params: Vec<EntryParam>) -> Vec<EntryParam> {
    params[0] = ptr("qkv", "const __nv_bfloat16");
    params[1] = ptr("k_cache", "void");
    params[2] = ptr("v_cache", "void");
    params.extend(["theta_bits", "v_multiplier_bits"].map(u32_param));
    params
}

/// The gated fold entry's trailing `GatedRopeArgs`, from `gated_rope_params`.
const GATED_ROPE_ARGS: &str =
    "{k_cache,v_cache,q_norm,k_norm,rope_positions,cos_sin,__uint_as_float(eps_bits)}";

/// A decode entry's parameters with the gated fold: packed rows in place of
/// `q`, writable caches, then the norm weights, RoPE positions and table,
/// and epsilon.
fn gated_rope_params(mut params: Vec<EntryParam>) -> Vec<EntryParam> {
    params[0] = ptr("qkv", "const __nv_bfloat16");
    params[1] = ptr("k_cache", "void");
    params[2] = ptr("v_cache", "void");
    params.extend([
        ptr("q_norm", "const uint16_t"),
        ptr("k_norm", "const uint16_t"),
        ptr("rope_positions", "const int32_t"),
        ptr("cos_sin", "const float"),
        u32_param("eps_bits"),
    ]);
    params
}

/// `params` with paged decode's and verify's `split_units` after
/// `split_target`.
fn with_split_units(mut params: Vec<EntryParam>) -> Vec<EntryParam> {
    let at = params
        .iter()
        .position(|p| matches!(p, EntryParam::U32 { name } if name == "split_target"))
        .expect("attention entries take split_target");
    params.insert(at + 1, u32_param("split_units"));
    params
}

fn merge_params() -> Vec<EntryParam> {
    let mut p = vec![
        ptr("workspace", "const float"),
        ptr("sink", "const __nv_bfloat16"),
        ptr("seq_lens", "const int32_t"),
        ptr("q_start", "const int32_t"),
        ptr("q_len", "const int32_t"),
        ptr("prefix_len", "const int32_t"),
        ptr("live_batch", "const int32_t"),
        ptr("out", "__nv_bfloat16"),
    ];
    p.extend(["split_capacity", "split_target", "window_left"].map(u32_param));
    p
}

#[derive(Deserialize)]
pub struct TuningTable {
    pub compute_capability: u32,
    pub sm_count: u32,
    pub prefill_rows: u32,
    pub prefill_key_tile: u32,
    pub decode_ctas: u32,
    pub prefill_ctas: u32,
    pub decode_buckets: Vec<DecodeBucket>,
}

#[derive(Deserialize)]
pub struct DecodeBucket {
    pub batch: u32,
    pub split_limit: u32,
    /// Paged decode's smallest piece of a row (verify: of an item, at the
    /// item bucket), in units: a row is cut into pieces from its own length
    /// alone (decode.cuh `piece_units`).
    pub split_units: u32,
    pub merge_threads: u32,
}

impl TuningTable {
    fn bucket(&self, prepared_batch: u32) -> &DecodeBucket {
        self.decode_buckets
            .iter()
            .find(|row| row.batch == prepared_batch)
            .expect("attention tuning covers each prepared batch bucket")
    }
}

/// A folded decode's runtime inputs.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Fold {
    Neox(RopeInputs),
    Gated(GatedRopeInputs),
}

pub fn tuning() -> TuningTable {
    serde_json::from_str(include_str!("../../tuning/attention.json"))
        .expect("attention tuning table is valid")
}

/// The per-rank shape of a variant's heads and cache.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub qk_dim: u32,
    pub v_dim: u32,
    pub page_size: u32,
    pub kv_dtype: DType,
    pub layout: CacheLayout,
    pub window_left: Option<u32>,
    pub sink: bool,
    pub softcap: Option<f32>,
    pub causal: bool,
}

/// A variant of `shape` at `phase` with the catalog's default capacities:
/// 32 items, a 1,048,576-token context, and 1, 8 or 8192 query rows per
/// item in decode, verify and prefill. It folds nothing and declares
/// nothing.
pub fn variant(
    name: &'static str,
    profile: Profile,
    phase: AttentionPhase,
    shape: Shape,
) -> Variant {
    Variant {
        name,
        profile,
        query: AttentionQuery {
            q_heads: shape.q_heads,
            kv_heads: shape.kv_heads,
            qk_dim: shape.qk_dim,
            v_dim: shape.v_dim,
            page_size: shape.page_size,
            kv_dtype: shape.kv_dtype,
            layout: shape.layout,
            causal: shape.causal,
            window_left: shape.window_left,
            sink: shape.sink,
            softcap: shape.softcap,
            softmax_scale: 1.0 / (shape.qk_dim as f32).sqrt(),
            phase,
            max_context_capacity: 1_048_576,
            max_batch_capacity: 32,
        },
        layout: shape.layout,
        query_capacity: match phase {
            AttentionPhase::Decode => 1,
            AttentionPhase::Verify => 8,
            AttentionPhase::Prefill => 8192,
        },
        rope_fold: None,
        gated_fold: None,
        invariance: Invariance::NONE,
        exact_rows: false,
    }
}

/// The kernel base's synthetic variants (`Profile::StdDenseMoe`).
pub fn own_catalog() -> Vec<Variant> {
    use AttentionPhase::{Decode, Prefill};
    use CacheLayout::Paged;
    use DType::{Bf16, Fp8E4M3};
    #[allow(non_snake_case)]
    let Std = Profile::StdDenseMoe;
    let std128 = Shape {
        q_heads: 16,
        kv_heads: 2,
        qk_dim: 128,
        v_dim: 128,
        page_size: 32,
        kv_dtype: Bf16,
        layout: Paged,
        window_left: None,
        sink: false,
        softcap: None,
        causal: true,
    };
    let std_fp8 = Shape {
        kv_dtype: Fp8E4M3,
        ..std128
    };
    let std_softcap = Shape {
        softcap: Some(50.0),
        ..std128
    };
    let std64 = Shape {
        qk_dim: 64,
        v_dim: 64,
        page_size: 16,
        ..std128
    };
    let std256 = Shape {
        qk_dim: 256,
        v_dim: 256,
        page_size: 64,
        ..std128
    };
    vec![
        variant("std128_bf16_prefill", Std, Prefill, std128),
        variant("std128_bf16_decode", Std, Decode, std128),
        variant("std128_fp8_prefill", Std, Prefill, std_fp8),
        variant("std128_fp8_decode", Std, Decode, std_fp8),
        variant("std128_softcap_prefill", Std, Prefill, std_softcap),
        variant("std128_softcap_decode", Std, Decode, std_softcap),
        variant("std64_prefill", Std, Prefill, std64),
        variant("std64_decode", Std, Decode, std64),
        variant("std256_prefill", Std, Prefill, std256),
        variant("std256_decode", Std, Decode, std256),
    ]
}

/// Every variant of `catalogs`: the kernel base's, then each model's.
pub fn catalog(catalogs: &Catalogs) -> Vec<Variant> {
    let mut all = own_catalog();
    for model in catalogs.models() {
        all.extend((model.attention)());
    }
    all
}

/// One cubin per profile of `catalogs` (or `profile`'s alone), so the lab
/// loads each family's entries once, with the profile's gated attention
/// rows (`gated::catalog`). A profile with gated rows and no variants has a
/// cubin of its rows alone.
pub fn cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<CubinSpec> {
    let all = catalog(catalogs);
    let gated = gated::catalog(catalogs);
    catalogs
        .profiles()
        .into_iter()
        .filter(|&tag| profile.is_none_or(|p| p == tag))
        .filter(|&tag| {
            all.iter().any(|v| v.profile == tag) || gated.iter().any(|e| e.profile == tag)
        })
        .map(|tag| {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda");
            let mut variants = all.iter().copied().filter(|v| v.profile == tag);
            let mut spec = if let Some(first) = variants.next() {
                let mut spec = first.cubin(&root).expect("catalog cubin");
                for variant in variants {
                    spec.entries
                        .extend(variant.cubin(&root).expect("catalog cubin").entries);
                }
                spec
            } else {
                CubinSpec {
                    name: String::new(),
                    sources_root: root,
                    include_dirs: vec![PathBuf::from(".")],
                    headers: vec!["#include \"attention/gated.cuh\"".into()],
                    defines: vec![],
                    entries: vec![],
                }
            };
            spec.name = format!("attention_{}_sm120", tag.tag());
            spec.entries.extend(
                gated
                    .iter()
                    .filter(|entry| entry.profile == tag)
                    .map(gated::Entry::entry_spec),
            );
            spec
        })
        .collect()
}
