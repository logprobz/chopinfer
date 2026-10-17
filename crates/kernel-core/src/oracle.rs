//! F64 CPU oracles over decoded operands, and numerical gates.
//! Inputs must be decoded from the exact bytes and scales presented to a kernel.

use crate::attend::{Attend, AttendValues, KvRows, paged_rows, window};
use crate::codec::{bf16_to_f32, f32_to_bf16};
use crate::contract::{
    AttentionLayout, AttentionQuery, DType, OutputQuantization, Refusal, RmsNormKind, RotaryStyle,
    RouterKind, SwigluColumns, Validate,
};
use crate::head::StreamingHead;
use crate::quantize::{self, Quantized};
use crate::rmsnorm::RmsNormNumerics;
use crate::rotary::{
    AngleRule, Axes, ComputedTable, FreqTable, RotaryPositions, RotaryRounding, RotarySlice,
    Rotation,
};
use crate::select::{NanKeys, Ties, topk};

/// Four interleaved f64 lanes, combined in a fixed order, including a scalar tail.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0; 4];
    let (left, right) = (a.chunks_exact(4), b.chunks_exact(4));
    let tail = left
        .remainder()
        .iter()
        .zip(right.remainder())
        .map(|(x, y)| x * y)
        .sum::<f64>();
    for (x, y) in left.zip(right) {
        for lane in 0..4 {
            acc[lane] += x[lane] * y[lane];
        }
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// Exact E4M3 values indexed by their byte encoding.
pub fn e4m3_table() -> &'static [f64; 256] {
    static TABLE: std::sync::OnceLock<[f64; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|code| crate::codec::SmallFloat::E4M3.decode_f64(code as u8))
    })
}

/// Round through F32 to BF16 with the codec's ties-to-even rule.
pub fn bf16_round(value: f64) -> f64 {
    f64::from(bf16_to_f32(f32_to_bf16(value as f32)))
}

#[derive(Clone, Debug, PartialEq)]
pub struct OracleOutput {
    pub values: Vec<f64>,
    /// Per-output sum of absolute products in its reduction.
    pub sum_abs_products: Vec<f64>,
    /// K for each output element. Attention rows can have different live K.
    pub reduction_lengths: Vec<usize>,
}

impl OracleOutput {
    pub fn new(
        values: Vec<f64>,
        sum_abs_products: Vec<f64>,
        reduction_len: usize,
    ) -> Result<Self, Refusal> {
        if values.len() != sum_abs_products.len() || reduction_len == 0 {
            return Err(Refusal(
                "oracle values and S must have matching lengths and nonzero K",
            ));
        }
        let reduction_lengths = vec![reduction_len; values.len()];
        Ok(Self {
            values,
            sum_abs_products,
            reduction_lengths,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertRowRange {
    pub expert: usize,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct GemmEpilogue<'a> {
    pub residual: Option<&'a [f64]>,
    pub swiglu: Option<SwigluColumns>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GemmShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub experts: usize,
}

/// A is `[M,K]`, B is `[experts,N,K]`, both row-major. Expert ranges cover each
/// routed A row exactly once; without ranges, the sole expert is 0.
pub fn gemm(
    a: &[f64],
    b: &[f64],
    shape: GemmShape,
    ranges: &[ExpertRowRange],
    epilogue: GemmEpilogue<'_>,
) -> Result<OracleOutput, Refusal> {
    let GemmShape { m, n, k, experts } = shape;
    if m == 0 || n == 0 || k == 0 || experts == 0 || a.len() != m * k || b.len() != experts * n * k
    {
        return Err(Refusal("invalid GEMM operand shape"));
    }
    if epilogue.swiglu.is_some() && !n.is_multiple_of(2) {
        return Err(Refusal("SwiGLU N must be even"));
    }
    if ranges.is_empty() && experts != 1 {
        return Err(Refusal("grouped GEMM needs expert row ranges"));
    }
    let mut row_expert = vec![usize::MAX; m];
    if ranges.is_empty() {
        row_expert.fill(0);
    }
    for range in ranges {
        if range.expert >= experts || range.start >= range.end || range.end > m {
            return Err(Refusal("invalid expert row range"));
        }
        for slot in &mut row_expert[range.start..range.end] {
            if *slot != usize::MAX {
                return Err(Refusal("overlapping expert row ranges"));
            }
            *slot = range.expert;
        }
    }
    if row_expert.contains(&usize::MAX) {
        return Err(Refusal("expert row ranges leave uncovered rows"));
    }
    let mut values = vec![0.0; m * n];
    let mut s = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            let mut abs_sum = 0.0;
            for kk in 0..k {
                let product = a[row * k + kk] * b[(row_expert[row] * n + col) * k + kk];
                sum += product;
                abs_sum += product.abs();
            }
            values[row * n + col] = sum;
            s[row * n + col] = abs_sum;
        }
    }
    if let Some(residual) = epilogue.residual {
        if residual.len() != values.len() {
            return Err(Refusal("GEMM residual shape mismatch"));
        }
        for (v, &r) in values.iter_mut().zip(residual) {
            *v += r;
        }
    }
    let output = OracleOutput::new(values, s, k)?;
    match epilogue.swiglu {
        Some(columns) => silu_glu(&output, n, columns, SiluGlu::UNCLAMPED),
        None => Ok(output),
    }
}

/// `silu(gate) * up` with `silu(g) = g / (1 + exp(-g))`, gate and up each
/// optionally clamped first to inclusive `(lo, hi)` bounds (an infinite
/// bound leaves that side open). `silu_output` rounds `silu(gate)` before the
/// multiply, as a framework that materializes the activation does; it
/// accepts only BF16 (through F32).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SiluGlu {
    pub gate_limit: Option<(f64, f64)>,
    pub up_limit: Option<(f64, f64)>,
    pub silu_output: Option<DType>,
}

impl SiluGlu {
    pub const UNCLAMPED: Self = Self {
        gate_limit: None,
        up_limit: None,
        silu_output: None,
    };

    fn check(self) -> Result<(), Refusal> {
        let clamps_valid = [self.gate_limit, self.up_limit]
            .iter()
            .flatten()
            .all(|&(lo, hi)| !lo.is_nan() && !hi.is_nan() && lo <= hi);
        if !clamps_valid || !matches!(self.silu_output, None | Some(DType::Bf16)) {
            return Err(Refusal("SiLU GLU clamp or rounding invalid"));
        }
        Ok(())
    }

    /// `(silu, up, silu * up)` of one checked pair, after clamps and rounding.
    fn pair(self, gate: f64, up: f64) -> (f64, f64, f64) {
        let clamp = |value: f64, limit: Option<(f64, f64)>| {
            limit.map_or(value, |(lo, hi)| value.clamp(lo, hi))
        };
        let (gate, up) = (clamp(gate, self.gate_limit), clamp(up, self.up_limit));
        let mut silu = gate / (1.0 + (-gate).exp());
        if self.silu_output.is_some() {
            silu = bf16_round(silu);
        }
        (silu, up, silu * up)
    }

    /// The gated unit over equal-length gate and up values.
    pub fn apply(self, gate: &[f64], up: &[f64]) -> Result<Vec<f64>, Refusal> {
        self.check()?;
        if gate.len() != up.len() {
            return Err(Refusal("SiLU GLU gate and up lengths differ"));
        }
        Ok(gate
            .iter()
            .zip(up)
            .map(|(&g, &u)| self.pair(g, u).2)
            .collect())
    }
}

/// The gated unit over GEMM outputs [m, n] whose gate and up columns
/// `columns` lays out, giving [m, n / 2]. S is `S_gate |up| + S_up |silu|`
/// and K is the GEMM's; clamps are 1-Lipschitz, so S still bounds the error.
pub fn silu_glu(
    gemm: &OracleOutput,
    n: usize,
    columns: SwigluColumns,
    activation: SiluGlu,
) -> Result<OracleOutput, Refusal> {
    activation.check()?;
    if n == 0 || !n.is_multiple_of(2) || !gemm.values.len().is_multiple_of(n) {
        return Err(Refusal("SiLU GLU shape or clamp invalid"));
    }
    let half = n / 2;
    let rows = gemm.values.len() / n;
    let mut values = Vec::with_capacity(rows * half);
    let mut s = Vec::with_capacity(rows * half);
    let mut k = Vec::with_capacity(rows * half);
    for row in 0..rows {
        for col in 0..half {
            let (gate_col, up_col) = match columns {
                SwigluColumns::ConcatenatedHalves => (col, col + half),
                SwigluColumns::Interleaved => (2 * col, 2 * col + 1),
            };
            let (gate_index, up_index) = (row * n + gate_col, row * n + up_col);
            let (silu, up, value) = activation.pair(gemm.values[gate_index], gemm.values[up_index]);
            values.push(value);
            s.push(
                gemm.sum_abs_products[gate_index] * up.abs()
                    + gemm.sum_abs_products[up_index] * silu.abs(),
            );
            k.push(gemm.reduction_lengths[gate_index]);
        }
    }
    Ok(OracleOutput {
        values,
        sum_abs_products: s,
        reduction_lengths: k,
    })
}

/// Reference output quantization returns the exact decoded value of the
/// quantized bytes, alongside the bytes/scales to compare bit-for-bit.
pub fn output_quantize(
    values: &[f64],
    rows: usize,
    k: usize,
    format: OutputQuantization,
) -> Result<Quantized, Refusal> {
    let input = values.iter().map(|&v| v as f32).collect::<Vec<_>>();
    match format {
        OutputQuantization::Fp8Group128(mode) => {
            quantize::fp8_activation_1x128(&input, rows, k, mode)
        }
        OutputQuantization::Mxfp8 => quantize::mxfp8(&input, rows, k),
        OutputQuantization::Mxfp4 => quantize::mxfp4(&input, rows, k),
        OutputQuantization::Nvfp4 => quantize::nvfp4(&input, rows, k),
    }
}

#[derive(Clone, Debug)]
pub struct PagedAttention<'a> {
    /// Q rows are packed in batch order, each row `[q_heads,qk_dim]`.
    pub q: &'a [f64],
    /// Physical cache layout is `[page,page_size,kv_heads,head_dim]`.
    pub keys: &'a [f64],
    pub values: &'a [f64],
    pub page_tables: &'a [Vec<usize>],
    pub query_lengths: &'a [usize],
    /// Absolute position of the first query in each batch item.
    pub query_starts: &'a [usize],
    /// Valid cached tokens, including the current prefill or decode chunk.
    pub key_lengths: &'a [usize],
    /// One extra per-head logit contributes to max and denominator only.
    pub sink_logits: Option<&'a [f64]>,
}

fn cache_offset(
    page: usize,
    offset: usize,
    head: usize,
    dim: usize,
    page_size: usize,
    heads: usize,
    head_dim: usize,
) -> usize {
    (((page * page_size + offset) * heads + head) * head_dim) + dim
}

pub fn attention(
    query: &AttentionQuery,
    input: &PagedAttention<'_>,
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    if query.layout != AttentionLayout::Paged {
        return Err(Refusal("paged oracle requires paged KV"));
    }
    let batch = input.query_lengths.len();
    if batch == 0
        || batch > query.max_batch_capacity as usize
        || input.page_tables.len() != batch
        || input.query_starts.len() != batch
        || input.key_lengths.len() != batch
    {
        return Err(Refusal("attention batch metadata mismatch"));
    }
    let q_heads = query.q_heads as usize;
    let kv_heads = query.kv_heads as usize;
    let qk_dim = query.qk_dim as usize;
    let v_dim = query.v_dim as usize;
    let page_size = query.page_size as usize;
    let q_rows: usize = input.query_lengths.iter().sum();
    if input.q.len() != q_rows * q_heads * qk_dim
        || input.sink_logits.is_some_and(|s| s.len() != q_heads)
        || input.keys.len() % (page_size * kv_heads * qk_dim) != 0
        || input.values.len() % (page_size * kv_heads * v_dim) != 0
    {
        return Err(Refusal("attention Q, cache or sink shape mismatch"));
    }
    let key_pages = input.keys.len() / (page_size * kv_heads * qk_dim);
    let value_pages = input.values.len() / (page_size * kv_heads * v_dim);
    let keys = [input.keys];
    let values = [input.values];
    let (keys, values) = (
        KvRows {
            segments: &keys,
            heads: kv_heads,
            dim: qk_dim,
        },
        KvRows {
            segments: &values,
            heads: kv_heads,
            dim: v_dim,
        },
    );
    let core = Attend {
        softcap: query.softcap.map(f64::from),
        sink: input.sink_logits.filter(|_| query.sink),
        ..Attend::f64(f64::from(query.softmax_scale))
    };
    let mut output = OracleOutput {
        values: Vec::with_capacity(q_rows * q_heads * v_dim),
        sum_abs_products: Vec::with_capacity(q_rows * q_heads * v_dim),
        reduction_lengths: Vec::with_capacity(q_rows * q_heads * v_dim),
    };
    let mut q_base = 0;
    for batch_index in 0..batch {
        let key_length = input.key_lengths[batch_index];
        if key_length == 0
            || key_length > query.max_context_capacity as usize
            || input.page_tables[batch_index].len() < key_length.div_ceil(page_size)
            || input.query_starts[batch_index] + input.query_lengths[batch_index] > key_length
        {
            return Err(Refusal("attention context or page table exceeds capacity"));
        }
        for qi in 0..input.query_lengths[batch_index] {
            let position = input.query_starts[batch_index] + qi;
            let visible = window(
                position,
                0..key_length,
                query.causal,
                query.window_left.map(|left| left as usize),
                None,
            );
            let list = paged_rows(
                visible,
                &input.page_tables[batch_index],
                page_size,
                key_pages.min(value_pages),
            )
            .ok_or(Refusal("attention page index outside KV cache"))?;
            if query.sink && input.sink_logits.is_none() {
                return Err(Refusal("sink logit missing"));
            }
            let row = (q_base + qi) * q_heads * qk_dim;
            core.row(
                &input.q[row..row + q_heads * qk_dim],
                keys,
                AttendValues::Rows(values),
                &list,
                &mut output,
            )?;
        }
        q_base += input.query_lengths[batch_index];
    }
    Ok(output)
}

#[derive(Clone, Debug)]
pub struct RingAttention<'a> {
    pub q: &'a [f64],
    pub keys: &'a [f64],
    pub values: &'a [f64],
    pub slot_ids: &'a [usize],
    pub query_lengths: &'a [usize],
    /// Tokens preceding the current chunk, or preceding the decode token.
    pub prefix_lengths: &'a [usize],
    pub k_new: Option<&'a [f64]>,
    pub v_new: Option<&'a [f64]>,
    pub sink_logits: Option<&'a [f64]>,
}

/// Ring cache is `[slot,ring_position,kv_head,dim]`. Prefill and verify read
/// new K/V from packed contiguous arrays; decode reads the current token from
/// the ring. Non-causal verify shows every query all of its item's new rows;
/// the window bounds each query at its own position.
pub fn ring_attention(
    query: &AttentionQuery,
    input: &RingAttention<'_>,
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let verify = query.phase == crate::contract::AttentionPhase::Verify;
    if query.layout != AttentionLayout::Ring || !query.causal && !verify {
        return Err(Refusal(
            "ring oracle requires causal ring KV outside verify",
        ));
    }
    let contiguous = query.phase != crate::contract::AttentionPhase::Decode;
    let batch = input.query_lengths.len();
    let q_heads = query.q_heads as usize;
    let kv_heads = query.kv_heads as usize;
    let qk_dim = query.qk_dim as usize;
    let v_dim = query.v_dim as usize;
    let ring = query.page_size as usize;
    let rows: usize = input.query_lengths.iter().sum();
    if batch == 0
        || batch > query.max_batch_capacity as usize
        || input.slot_ids.len() != batch
        || input.prefix_lengths.len() != batch
        || input.q.len() != rows * q_heads * qk_dim
        || input.keys.len() % (ring * kv_heads * qk_dim) != 0
        || input.values.len() % (ring * kv_heads * v_dim) != 0
        || input.sink_logits.is_some_and(|s| s.len() != q_heads)
    {
        return Err(Refusal("ring attention shape mismatch"));
    }
    if contiguous
        && (input
            .k_new
            .is_none_or(|k| k.len() != rows * kv_heads * qk_dim)
            || input
                .v_new
                .is_none_or(|v| v.len() != rows * kv_heads * v_dim))
    {
        return Err(Refusal("ring prefill needs packed new K and V"));
    }
    let slots = input.keys.len() / (ring * kv_heads * qk_dim);
    if input.values.len() / (ring * kv_heads * v_dim) != slots {
        return Err(Refusal("ring K and V slot counts differ"));
    }
    // Rows are the ring cache's slot rows, then (outside decode) the packed
    // new rows.
    let cache_rows = slots * ring;
    let key_segments = [input.keys, input.k_new.unwrap_or_default()];
    let value_segments = [input.values, input.v_new.unwrap_or_default()];
    let segments = if contiguous { 2 } else { 1 };
    let (keys, values) = (
        KvRows {
            segments: &key_segments[..segments],
            heads: kv_heads,
            dim: qk_dim,
        },
        KvRows {
            segments: &value_segments[..segments],
            heads: kv_heads,
            dim: v_dim,
        },
    );
    let core = Attend {
        softcap: query.softcap.map(f64::from),
        sink: input.sink_logits.filter(|_| query.sink),
        ..Attend::f64(f64::from(query.softmax_scale))
    };
    let mut output = OracleOutput {
        values: Vec::with_capacity(rows * q_heads * v_dim),
        sum_abs_products: Vec::with_capacity(rows * q_heads * v_dim),
        reduction_lengths: Vec::with_capacity(rows * q_heads * v_dim),
    };
    let mut q_base = 0;
    for batch_index in 0..batch {
        let prefix = input.prefix_lengths[batch_index];
        let q_len = input.query_lengths[batch_index];
        let slot = input.slot_ids[batch_index];
        if slot >= slots
            || q_len == 0
            || prefix + q_len > query.max_context_capacity as usize
            || query.phase == crate::contract::AttentionPhase::Decode && q_len != 1
        {
            return Err(Refusal("ring attention descriptor exceeds capacity"));
        }
        for qi in 0..q_len {
            let position = prefix + qi;
            let visible = window(
                position,
                0..prefix + q_len,
                query.causal,
                Some(query.window_left.unwrap_or(0) as usize),
                None,
            );
            let list = visible
                .map(|key_pos| {
                    if contiguous && key_pos >= prefix {
                        (cache_rows + q_base + key_pos - prefix) as i64
                    } else {
                        (slot * ring + key_pos % ring) as i64
                    }
                })
                .collect::<Vec<_>>();
            if query.sink && input.sink_logits.is_none() {
                return Err(Refusal("sink logit missing"));
            }
            let row = (q_base + qi) * q_heads * qk_dim;
            core.row(
                &input.q[row..row + q_heads * qk_dim],
                keys,
                AttendValues::Rows(values),
                &list,
                &mut output,
            )?;
        }
        q_base += q_len;
    }
    Ok(output)
}

/// Exact paged KV layout helper. Both K and V are `[tokens,kv_heads,dim]`.
pub fn paged_kv_write(
    cache: &mut [f64],
    block_table: &[usize],
    positions: &[usize],
    source: &[f64],
    page_size: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<(), Refusal> {
    let block_bytes = page_size
        .checked_mul(kv_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .ok_or(Refusal("paged KV write shape mismatch"))?;
    paged_kv_write_strided(
        cache,
        block_table,
        positions,
        source,
        page_size,
        kv_heads,
        head_dim,
        block_bytes as u64,
        std::mem::size_of::<f64>(),
    )
}

/// Write into a page-local K or V block. Each f64 cell models one cache element;
/// `element_bytes` is the device element size used to interpret the byte stride.
#[allow(clippy::too_many_arguments)]
pub fn paged_kv_write_strided(
    cache: &mut [f64],
    block_table: &[usize],
    positions: &[usize],
    source: &[f64],
    page_size: usize,
    kv_heads: usize,
    head_dim: usize,
    page_stride_bytes: u64,
    element_bytes: usize,
) -> Result<(), Refusal> {
    let block_len = page_size
        .checked_mul(kv_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or(Refusal("paged KV write shape mismatch"))?;
    if block_len == 0
        || element_bytes == 0
        || source.len()
            != positions
                .len()
                .checked_mul(kv_heads)
                .and_then(|n| n.checked_mul(head_dim))
                .ok_or(Refusal("paged KV write shape mismatch"))?
        || page_stride_bytes % element_bytes as u64 != 0
        || page_stride_bytes < (block_len as u64) * element_bytes as u64
    {
        return Err(Refusal("paged KV write shape mismatch"));
    }
    let stride = usize::try_from(page_stride_bytes / element_bytes as u64)
        .map_err(|_| Refusal("KV page outside cache"))?;
    for (index, &position) in positions.iter().enumerate() {
        let page = *block_table
            .get(position / page_size)
            .ok_or(Refusal("KV page missing"))?;
        let page_base = page
            .checked_mul(stride)
            .ok_or(Refusal("KV page outside cache"))?;
        if page_base
            .checked_add(block_len)
            .is_none_or(|end| end > cache.len())
        {
            return Err(Refusal("KV page outside cache"));
        }
        for head in 0..kv_heads {
            for d in 0..head_dim {
                cache[page_base
                    + cache_offset(
                        0,
                        position % page_size,
                        head,
                        d,
                        page_size,
                        kv_heads,
                        head_dim,
                    )] = source[(index * kv_heads + head) * head_dim + d];
            }
        }
    }
    Ok(())
}

/// Weight form of an RMSNorm: `w`, or Gemma's `1 + w` (added in the norm's
/// arithmetic).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormOffset {
    Plain,
    OnePlusWeight,
}

/// Arithmetic of an RMSNorm reference. F32 converts inputs and weights to F32
/// first and does every step below in F32; F64 does them in f64.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormArithmetic {
    F64,
    F32,
}

/// How an RMSNorm applies its root mean square `rms = sqrt(sum(x * x) / n +
/// eps)`: `x * (1 / rms)`, or `x / rms`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormScale {
    MultiplyReciprocal,
    Divide,
}

/// Every numerically relevant choice of an RMSNorm reference. Each group of
/// `n` values computes `rms = sqrt(sum(x * x) / n + eps)`, summing squares in
/// index order, then `y = (x * (1 / rms)) * w` or `y = (x / rms) * w` by
/// `scale`. `before_weight` rounds the normalized `x` and `output` rounds
/// `y`; both accept only BF16 (round to nearest even) and only under F32
/// arithmetic. F32 arithmetic rounds eps to F32.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNorm {
    pub eps: f64,
    pub offset: NormOffset,
    pub arithmetic: NormArithmetic,
    pub scale: NormScale,
    pub before_weight: Option<DType>,
    pub output: Option<DType>,
}

impl RmsNorm {
    /// F64 arithmetic, a reciprocal multiply and no rounding points.
    pub fn f64(eps: f64, offset: NormOffset) -> Self {
        Self {
            eps,
            offset,
            arithmetic: NormArithmetic::F64,
            scale: NormScale::MultiplyReciprocal,
            before_weight: None,
            output: None,
        }
    }

    /// Normalizes `input` [rows, weights.len()] in independent groups of
    /// `group` values; weights index the position within the row. S is
    /// `sum(x * x) + |x * w|` per output and K is `group`.
    pub fn apply(
        &self,
        input: &[f64],
        weights: &[f64],
        group: usize,
    ) -> Result<OracleOutput, Refusal> {
        self.apply_summed(input, weights, group, serial_sum)
    }

    /// [`Self::apply`] whose F32 arithmetic adds each group's squares with
    /// `sum` (F64 arithmetic sums in index order).
    fn apply_summed(
        &self,
        input: &[f64],
        weights: &[f64],
        group: usize,
        sum: fn(&[f32]) -> f32,
    ) -> Result<OracleOutput, Refusal> {
        let width = weights.len();
        if width == 0
            || group == 0
            || !width.is_multiple_of(group)
            || input.is_empty()
            || !input.len().is_multiple_of(width)
            || !self.eps.is_finite()
            || self.eps <= 0.0
            || (self.arithmetic == NormArithmetic::F32
                && (self.eps as f32 == 0.0 || !(self.eps as f32).is_finite()))
        {
            return Err(Refusal("RMSNorm shape or epsilon mismatch"));
        }
        let bf16 = |point: Option<DType>| match point {
            None => Ok(false),
            Some(DType::Bf16) if self.arithmetic == NormArithmetic::F32 => Ok(true),
            Some(_) => Err(Refusal(
                "RMSNorm rounding points are BF16 under F32 arithmetic only",
            )),
        };
        let (round_mid, round_out) = (bf16(self.before_weight)?, bf16(self.output)?);
        let mut values = Vec::with_capacity(input.len());
        let mut s = Vec::with_capacity(input.len());
        for (index, x) in input.chunks_exact(group).enumerate() {
            let w = &weights[index * group % width..][..group];
            match self.arithmetic {
                NormArithmetic::F64 => {
                    let square_sum: f64 = x.iter().map(|v| v * v).sum();
                    let rms = (square_sum / group as f64 + self.eps).sqrt();
                    let inv = rms.recip();
                    for (&x, &w) in x.iter().zip(w) {
                        let weight = match self.offset {
                            NormOffset::Plain => w,
                            NormOffset::OnePlusWeight => 1.0 + w,
                        };
                        let normalized = match self.scale {
                            NormScale::MultiplyReciprocal => x * inv,
                            NormScale::Divide => x / rms,
                        };
                        values.push(normalized * weight);
                        s.push(square_sum + (x * weight).abs());
                    }
                }
                NormArithmetic::F32 => {
                    let x = x.iter().map(|&v| v as f32).collect::<Vec<_>>();
                    let squares = sum(&x.iter().map(|&v| v * v).collect::<Vec<_>>());
                    let rms = (squares / group as f32 + self.eps as f32).sqrt();
                    let inv = rms.recip();
                    for (&x, &w) in x.iter().zip(w) {
                        let weight = match self.offset {
                            NormOffset::Plain => w as f32,
                            NormOffset::OnePlusWeight => 1.0 + w as f32,
                        };
                        let mut y = match self.scale {
                            NormScale::MultiplyReciprocal => x * inv,
                            NormScale::Divide => x / rms,
                        };
                        if round_mid {
                            y = bf16_to_f32(f32_to_bf16(y));
                        }
                        y *= weight;
                        if round_out {
                            y = bf16_to_f32(f32_to_bf16(y));
                        }
                        values.push(f64::from(y));
                        s.push(f64::from(squares) + (f64::from(x) * f64::from(weight)).abs());
                    }
                }
            }
        }
        OracleOutput::new(values, s, group)
    }

    /// `apply` over BF16 rows and weights, stored back to BF16 through F32.
    pub fn apply_bf16(
        &self,
        input: &[u16],
        weights: &[u16],
        group: usize,
    ) -> Result<Vec<u16>, Refusal> {
        let widen = |bits: &[u16]| {
            bits.iter()
                .map(|&b| f64::from(bf16_to_f32(b)))
                .collect::<Vec<_>>()
        };
        let out = self.apply(&widen(input), &widen(weights), group)?;
        Ok(out.values.iter().map(|&v| f32_to_bf16(v as f32)).collect())
    }
}

/// Contract adapter: the RMSNorm of a row query's kind, after an optional
/// residual add. HfBf16Intermediate rounds the summed input and the weights
/// to BF16 and reports zero S over the full width.
///
/// The oracle gate of a row RMSNorm whose requirement leaves its numerics to
/// the backend (`RmsNormRequirement::numerics` is `None`), the one the row
/// RMSNorm cases apply today:
/// - Plain, Gemma and per-head kinds: [`check_numeric`] against this
///   reference in the requirement's output format, half a BF16 ulp of the
///   reference plus 2^-22*K*S with K the group width and S the group's sum
///   of squares plus `|x * w|`. The reference is F64, so any F32 reduction
///   order within that bound passes.
/// - HF kind: [`check_exact`] of the output's BF16 words against this
///   reference's, which is HF's own F32 order rounded to BF16. A
///   requirement that pins [`RmsNormNumerics::HF_TREE_256`] is judged
///   against [`rmsnorm_numerics`]' tree-order reference instead.
///   [`check_numeric`] is not this gate: with zero S its bound is half an
///   ulp of the reference, which admits the BF16 value just below a power of
///   two (1.0 against 0.99609375).
///
/// A fused collective's RMSNorm epilogue has its own gate; see
/// `contract::RmsNormEpilogue`.
pub fn rmsnorm(
    input: &[f64],
    width: usize,
    weights: &[f64],
    eps: f64,
    kind: RmsNormKind,
    residual: Option<&[f64]>,
) -> Result<OracleOutput, Refusal> {
    rmsnorm_summed(input, width, weights, eps, kind, residual, serial_sum)
}

/// [`rmsnorm`] whose HF reference follows `numerics`, the requirement's
/// resolved numerics: [`RmsNormNumerics::HF_TREE_256`] sums the HF kind's
/// squares in its 256-lane tree ([`hf_tree_256_sum`]), and
/// [`RmsNormNumerics::HF_ORDERED`] or numerics left to the backend in HF's
/// serial order. Other numerics name no exact HF reference and are
/// refused. The other kinds' F64 reference and gate do not depend on the
/// order.
pub fn rmsnorm_numerics(
    input: &[f64],
    width: usize,
    weights: &[f64],
    eps: f64,
    kind: RmsNormKind,
    residual: Option<&[f64]>,
    numerics: Option<RmsNormNumerics>,
) -> Result<OracleOutput, Refusal> {
    let sum = match (kind, numerics) {
        (RmsNormKind::HfBf16Intermediate, Some(RmsNormNumerics::HF_TREE_256)) => hf_tree_256_sum,
        (RmsNormKind::HfBf16Intermediate, Some(numerics))
            if numerics != RmsNormNumerics::HF_ORDERED =>
        {
            return Err(Refusal(
                "the HF RMSNorm has an exact reference for HF_ORDERED and HF_TREE_256 only",
            ));
        }
        _ => serial_sum,
    };
    rmsnorm_summed(input, width, weights, eps, kind, residual, sum)
}

/// The F32 sum of `squares` from +0 in index order, HF's order.
fn serial_sum(squares: &[f32]) -> f32 {
    squares.iter().fold(0.0, |sum, &square| sum + square)
}

/// The F32 sum of `squares` in [`RmsNormNumerics::HF_TREE_256`]'s 256-lane
/// tree, with IEEE adds: lane `l` adds columns `8 l + 2048 i + j` (`i`
/// ascending, then `j` = 0 to 7) from +0; each warp of 32 lanes adds its
/// partials pairwise at distances 16, 8, 4, 2, 1 (lane `m` adds lane
/// `m + d`), and the eight warp sums the same way at distances 4, 2, 1.
pub fn hf_tree_256_sum(squares: &[f32]) -> f32 {
    const LANES: usize = 256;
    const SPAN: usize = LANES * 8;
    let mut lanes = [0.0_f32; LANES];
    for (lane, partial) in lanes.iter_mut().enumerate() {
        for base in (0..squares.len()).step_by(SPAN) {
            for j in 0..8 {
                if let Some(&square) = squares.get(base + 8 * lane + j) {
                    *partial += square;
                }
            }
        }
    }
    // Lane m adds lane m + d at each distance; lane 0 ends with the sum.
    let halve = |values: &mut [f32], distances: &[usize]| {
        for &d in distances {
            for m in 0..d {
                values[m] += values[m + d];
            }
        }
        values[0]
    };
    let mut warps = [0.0_f32; LANES / 32];
    for (warp, total) in warps.iter_mut().enumerate() {
        *total = halve(&mut lanes[warp * 32..(warp + 1) * 32], &[16, 8, 4, 2, 1]);
    }
    halve(&mut warps, &[4, 2, 1])
}

/// [`rmsnorm`] with the HF kind's F32 squares added by `hf_sum`.
fn rmsnorm_summed(
    input: &[f64],
    width: usize,
    weights: &[f64],
    eps: f64,
    kind: RmsNormKind,
    residual: Option<&[f64]>,
    hf_sum: fn(&[f32]) -> f32,
) -> Result<OracleOutput, Refusal> {
    if width == 0
        || input.is_empty()
        || !input.len().is_multiple_of(width)
        || weights.len() != width
        || !eps.is_finite()
        || eps <= 0.0
        || residual.is_some_and(|r| r.len() != input.len())
    {
        return Err(Refusal("RMSNorm shape or epsilon mismatch"));
    }
    let summed = |round: fn(f64) -> f64| {
        input
            .iter()
            .enumerate()
            .map(|(i, &v)| round(v + residual.map_or(0.0, |r| r[i])))
            .collect::<Vec<_>>()
    };
    let (norm, group) = match kind {
        RmsNormKind::Plain => (RmsNorm::f64(eps, NormOffset::Plain), width),
        RmsNormKind::GemmaOnePlusWeight => (RmsNorm::f64(eps, NormOffset::OnePlusWeight), width),
        RmsNormKind::QkPerHead { head_dim } => {
            let head_dim = head_dim as usize;
            if head_dim == 0 || !width.is_multiple_of(head_dim) {
                return Err(Refusal("RMSNorm head width mismatch"));
            }
            (RmsNorm::f64(eps, NormOffset::Plain), head_dim)
        }
        RmsNormKind::HfBf16Intermediate => {
            let eps32 = eps as f32;
            if !eps32.is_finite() || eps32 <= 0.0 {
                return Err(Refusal("HF RMSNorm shape or epsilon mismatch"));
            }
            let weights = weights.iter().map(|&v| bf16_round(v)).collect::<Vec<_>>();
            let values =
                hf_bf16_norm(eps).apply_summed(&summed(bf16_round), &weights, width, hf_sum)?;
            let len = values.values.len();
            return OracleOutput::new(values.values, vec![0.0; len], width);
        }
    };
    norm.apply(&summed(|v| v), weights, group)
}

/// HF's BF16 RMSNorm: F32 arithmetic, `x * inv` rounded to BF16 before the
/// weight and the output rounded to BF16.
fn hf_bf16_norm(eps: f64) -> RmsNorm {
    RmsNorm {
        eps,
        offset: NormOffset::Plain,
        arithmetic: NormArithmetic::F32,
        scale: NormScale::MultiplyReciprocal,
        before_weight: Some(DType::Bf16),
        output: Some(DType::Bf16),
    }
}

/// Table entries are per-position pairs of cosine and sine, with rotary_dim/2
/// pairs per position. When absent, angles use position/theta^(2i/rotary_dim).
pub fn rope(
    input: &[f64],
    positions: &[usize],
    head_dim: usize,
    rotary_dim: usize,
    theta: f64,
    style: RotaryStyle,
    table: Option<&[(f64, f64)]>,
) -> Result<Vec<f64>, Refusal> {
    rope_shape(input, positions, head_dim, rotary_dim, theta)?;
    let computed = ComputedTable {
        theta,
        rule: AngleRule::DivPowF64,
        axes: Axes::Linear,
    };
    rotate_head(
        input,
        positions,
        head_dim,
        rotary_dim,
        style,
        table.map_or(FreqTable::Computed(computed), FreqTable::Captured),
        RotaryRounding::F64,
    )
}

pub(crate) fn rope_shape(
    input: &[f64],
    positions: &[usize],
    head_dim: usize,
    rotary_dim: usize,
    theta: f64,
) -> Result<(), Refusal> {
    if head_dim == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || !theta.is_finite()
        || theta <= 0.0
        || input.len() != positions.len() * head_dim
    {
        return Err(Refusal("RoPE shape or theta mismatch"));
    }
    Ok(())
}

/// A forward rotation of the first `rotary_dim` values of each row.
fn rotate_head(
    input: &[f64],
    positions: &[usize],
    head_dim: usize,
    rotary_dim: usize,
    pairing: RotaryStyle,
    table: FreqTable<'_>,
    rounding: RotaryRounding,
) -> Result<Vec<f64>, Refusal> {
    let rotation = Rotation {
        slice: RotarySlice {
            offset: 0,
            len: rotary_dim,
        },
        pairing,
        inverse: false,
        rounding,
    };
    rotation.apply(input, head_dim, RotaryPositions::Linear(positions), &table)
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouterSelection {
    pub expert_ids: Vec<usize>,
    pub weights: Vec<f64>,
}

/// Callers refuse NaN keys first.
fn top_indices(scores: &[f64], k: usize, ties: Ties) -> Vec<usize> {
    topk(scores, k, ties, NanKeys::Refuse, None).expect("top-k keys are not NaN")
}

/// A router's score function over tempered logits `v = logit / temperature`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouterScore {
    /// `exp(v - max) / sum` over every expert, max and sum in expert order.
    Softmax,
    /// Softmax over the selected experts' `v` only, in selection order.
    SelectedSoftmax,
    /// `1 / (1 + exp(-v))`.
    Sigmoid,
    /// `sqrt(softplus(v))`, softplus as `max(v, 0) + ln_1p(exp(-|v|))`.
    SqrtSoftplus,
}

/// What top-k ranks: tempered logits or scores, plus the row's bias if one
/// is given.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteKey {
    Logits,
    Scores,
}

/// Top-k order of router keys, as `select::topk` ranks them.
pub use crate::select::Ties as RouteTies;

/// Group-limited selection: experts form `groups` contiguous groups, each
/// scored by the sum of its `per_group` largest keys (in top order, f64);
/// only the `keep` best groups stay selectable, equal sums keeping the lower
/// group. `per_group` above the group size is refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupLimit {
    pub groups: usize,
    pub keep: usize,
    pub per_group: usize,
    pub overflow: GroupSumOverflow,
}

/// What a group sum that overflows f64 (even transiently, before later keys
/// cancel it) does.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroupSumOverflow {
    /// Refuse the row: an infinite sum can rank a weaker group first.
    Refuse,
    /// Legacy `RouterKind::GroupLimitedSelectedSoftmax` compatibility only: rank the IEEE
    /// sums as they are, so overflowing groups tie at infinity and the lower
    /// group wins even when a later group is stronger.
    LegacyIeee,
}

/// Every numerically relevant choice of a router, all evaluated in f64.
/// Selected weights are the scores of the chosen experts; `normalize`
/// divides them by their sum (in selection order) plus eps, then each is
/// multiplied by `scale`. A row whose tempered logits, scores, keys,
/// normalization divisor or weights leave the finite range (or whose divisor
/// is zero) is refused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Routing {
    pub score: RouterScore,
    pub temperature: f64,
    pub key: RouteKey,
    pub group_limit: Option<GroupLimit>,
    pub normalize: Option<f64>,
    pub scale: f64,
    pub ties: RouteTies,
}

impl Routing {
    /// The routing a contract `RouterKind` names for `experts` logits.
    /// Softmax and grouped kinds rank raw logits and take no bias; grouped
    /// kinds score each group by its top two keys, or by its only one, and
    /// keep the legacy IEEE ranking of overflowing group sums.
    pub fn of_kind(kind: RouterKind, experts: usize) -> Self {
        let base = Self {
            score: RouterScore::Softmax,
            temperature: 1.0,
            key: RouteKey::Logits,
            group_limit: None,
            normalize: None,
            scale: 1.0,
            ties: RouteTies::LowerIndex,
        };
        match kind {
            RouterKind::SoftmaxTopK { renormalize } => Self {
                normalize: renormalize.then_some(0.0),
                ..base
            },
            RouterKind::SigmoidBiasSelectNormalized { scaling_factor } => Self {
                score: RouterScore::Sigmoid,
                key: RouteKey::Scores,
                normalize: Some(1e-20),
                scale: f64::from(scaling_factor),
                ..base
            },
            RouterKind::SqrtSoftplusBiasSelectNormalized { scaling_factor } => Self {
                score: RouterScore::SqrtSoftplus,
                key: RouteKey::Scores,
                normalize: Some(1e-20),
                scale: f64::from(scaling_factor),
                ..base
            },
            RouterKind::GroupLimitedSelectedSoftmax {
                n_group,
                topk_group,
            } => Self {
                score: RouterScore::SelectedSoftmax,
                group_limit: Some(GroupLimit {
                    groups: n_group as usize,
                    keep: topk_group as usize,
                    per_group: 2.min(experts.checked_div(n_group as usize).unwrap_or(2)),
                    overflow: GroupSumOverflow::LegacyIeee,
                }),
                ..base
            },
        }
    }
}

/// Route one row of logits. `bias`, when given, is this row's selection
/// bias (for example a separate bias for image tokens); it never enters the
/// weights.
pub fn route(
    logits: &[f64],
    bias: Option<&[f64]>,
    top_k: usize,
    routing: Routing,
) -> Result<RouterSelection, Refusal> {
    if logits.is_empty()
        || logits.iter().any(|v| !v.is_finite())
        || top_k == 0
        || top_k > logits.len()
        || bias.is_some_and(|b| b.len() != logits.len() || b.iter().any(|v| !v.is_finite()))
    {
        return Err(Refusal("router scores, bias or top-k invalid"));
    }
    let Routing {
        score,
        temperature,
        key,
        group_limit,
        normalize,
        scale,
        ties,
    } = routing;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Refusal("router scaling factor invalid"));
    }
    if !temperature.is_finite()
        || temperature <= 0.0
        || normalize.is_some_and(|eps| !eps.is_finite() || eps < 0.0)
    {
        return Err(Refusal("router temperature or normalization eps invalid"));
    }
    if score == RouterScore::SelectedSoftmax && key == RouteKey::Scores {
        return Err(Refusal("selected-softmax routing must rank logits"));
    }
    let nonfinite = || Refusal("router arithmetic left the finite range");
    let tempered = logits.iter().map(|&v| v / temperature).collect::<Vec<_>>();
    if tempered.iter().any(|v| !v.is_finite()) {
        return Err(nonfinite());
    }
    let scores = match score {
        RouterScore::Softmax => {
            let max = tempered.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps = tempered
                .iter()
                .map(|&v| (v - max).exp())
                .collect::<Vec<_>>();
            let total = exps.iter().sum::<f64>();
            exps.iter().map(|&e| e / total).collect()
        }
        // Weights come from the selected logits below.
        RouterScore::SelectedSoftmax => Vec::new(),
        RouterScore::Sigmoid => tempered.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
        RouterScore::SqrtSoftplus => tempered
            .iter()
            .map(|&v| (v.max(0.0) + (-v.abs()).exp().ln_1p()).sqrt())
            .collect(),
    };
    let mut keys = match key {
        RouteKey::Logits => tempered.clone(),
        RouteKey::Scores => scores.clone(),
    };
    if let Some(bias) = bias {
        for (k, b) in keys.iter_mut().zip(bias) {
            *k += b;
        }
    }
    if scores.iter().chain(&keys).any(|v| !v.is_finite()) {
        return Err(nonfinite());
    }
    if let Some(GroupLimit {
        groups,
        keep,
        per_group,
        overflow,
    }) = group_limit
    {
        if groups == 0
            || !keys.len().is_multiple_of(groups)
            || keep == 0
            || keep > groups
            || per_group == 0
            || per_group > keys.len() / groups
        {
            return Err(Refusal("grouped router geometry invalid"));
        }
        let size = keys.len() / groups;
        let group_scores = keys
            .chunks_exact(size)
            .map(|group| {
                let ids = top_indices(group, per_group, ties);
                ids.iter().map(|&i| group[i]).sum::<f64>()
            })
            .collect::<Vec<_>>();
        if overflow == GroupSumOverflow::Refuse && group_scores.iter().any(|v| !v.is_finite()) {
            return Err(nonfinite());
        }
        let allowed = top_indices(&group_scores, keep, ties);
        for (i, k) in keys.iter_mut().enumerate() {
            if !allowed.contains(&(i / size)) {
                *k = f64::NEG_INFINITY;
            }
        }
        if top_k > keep * size {
            return Err(Refusal("top-k exceeds selected group capacity"));
        }
    }
    let ids = top_indices(&keys, top_k, ties);
    let mut weights = if score == RouterScore::SelectedSoftmax {
        let max = ids
            .iter()
            .map(|&i| tempered[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let exps = ids
            .iter()
            .map(|&i| (tempered[i] - max).exp())
            .collect::<Vec<_>>();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|v| v / sum).collect()
    } else {
        ids.iter().map(|&i| scores[i]).collect::<Vec<_>>()
    };
    if let Some(eps) = normalize {
        let sum = weights.iter().sum::<f64>() + eps;
        if !sum.is_finite() || sum <= 0.0 {
            return Err(nonfinite());
        }
        for weight in &mut weights {
            *weight /= sum;
        }
    }
    for weight in &mut weights {
        *weight *= scale;
    }
    if weights.iter().any(|w| !w.is_finite()) {
        return Err(nonfinite());
    }
    Ok(RouterSelection {
        expert_ids: ids,
        weights,
    })
}

/// The router a contract `RouterKind` names; see `Routing::of_kind`. A bias
/// is validated even when the kind does not use it.
pub fn router(
    scores: &[f64],
    bias: Option<&[f64]>,
    top_k: usize,
    kind: RouterKind,
) -> Result<RouterSelection, Refusal> {
    if bias.is_some_and(|b| b.len() != scores.len() || b.iter().any(|v| !v.is_finite())) {
        return Err(Refusal("router scores, bias or top-k invalid"));
    }
    let bias = match kind {
        RouterKind::SigmoidBiasSelectNormalized { .. }
        | RouterKind::SqrtSoftplusBiasSelectNormalized { .. } => bias,
        RouterKind::SoftmaxTopK { .. } | RouterKind::GroupLimitedSelectedSoftmax { .. } => None,
    };
    route(scores, bias, top_k, Routing::of_kind(kind, scores.len()))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertPermutation {
    /// Source route index, or None for a padded expert slot.
    pub forward: Vec<Option<usize>>,
    /// Position in `forward` for each source route.
    pub inverse: Vec<usize>,
    pub expert_offsets: Vec<usize>,
}

pub fn expert_permute(
    expert_ids: &[usize],
    experts: usize,
    block_size: usize,
) -> Result<ExpertPermutation, Refusal> {
    if experts == 0 || block_size == 0 || expert_ids.iter().any(|&id| id >= experts) {
        return Err(Refusal("invalid expert permutation shape"));
    }
    let mut forward = Vec::new();
    let mut inverse = vec![0; expert_ids.len()];
    let mut offsets = Vec::with_capacity(experts + 1);
    for expert in 0..experts {
        offsets.push(forward.len());
        for (route, &id) in expert_ids.iter().enumerate() {
            if id == expert {
                inverse[route] = forward.len();
                forward.push(Some(route));
            }
        }
        let count = forward.len() - offsets[expert];
        forward.extend(std::iter::repeat_n(
            None,
            (block_size - count % block_size) % block_size,
        ));
    }
    offsets.push(forward.len());
    Ok(ExpertPermutation {
        forward,
        inverse,
        expert_offsets: offsets,
    })
}

/// Weighted routed rows are added in route order into their original token.
pub fn expert_combine(
    routed: &[f64],
    route_tokens: &[usize],
    route_weights: &[f64],
    tokens: usize,
    width: usize,
) -> Result<OracleOutput, Refusal> {
    if width == 0
        || tokens == 0
        || routed.len() != route_tokens.len() * width
        || route_tokens.len() != route_weights.len()
        || route_tokens.iter().any(|&token| token >= tokens)
    {
        return Err(Refusal("expert combine shape mismatch"));
    }
    let mut out = vec![0.0; tokens * width];
    let mut s = vec![0.0; tokens * width];
    for (route, &token) in route_tokens.iter().enumerate() {
        for col in 0..width {
            let product = routed[route * width + col] * route_weights[route];
            out[token * width + col] += product;
            s[token * width + col] += product.abs();
        }
    }
    OracleOutput::new(out, s, route_tokens.len().max(1))
}

/// A rank's routed sum (`MoeQuery::LocalRoutedSum`) in its contract's F32
/// arithmetic, as exact F32 values: per token, the F32 `routed` rows of
/// `width` (one per route, `top_k` per token) whose global expert in `ids`
/// lies in rank `rank`'s range of `local_experts` (negative ids never do)
/// added in top-k order from +0, each add rounded to nearest even, then the
/// token's `addend` row added once when given. A caller rounds an F32 sum
/// without addend to BF16 once for a BF16 output. Over finite values the
/// result is a kernel's in-order `__fadd_rn` sum bit for bit; a NaN sum is
/// NaN on both sides, but PTX leaves F32 NaN encodings unspecified, so only
/// NaN-ness compares.
pub fn local_routed_sum(
    routed: &[f32],
    ids: &[i32],
    top_k: usize,
    width: usize,
    local_experts: usize,
    rank: usize,
    addend: Option<&[f32]>,
) -> Result<Vec<f32>, Refusal> {
    if top_k == 0 || width == 0 || local_experts == 0 || !ids.len().is_multiple_of(top_k) {
        return Err(Refusal("routed sum geometry mismatch"));
    }
    let tokens = ids.len() / top_k;
    if routed.len() != ids.len() * width || addend.is_some_and(|a| a.len() != tokens * width) {
        return Err(Refusal("routed sum shape mismatch"));
    }
    let local = |id: i32| usize::try_from(id).is_ok_and(|id| id / local_experts == rank);
    let mut out = Vec::with_capacity(tokens * width);
    for token in 0..tokens {
        for col in 0..width {
            let mut sum = 0.0_f32;
            for slot in 0..top_k {
                let route = token * top_k + slot;
                if local(ids[route]) {
                    sum += routed[route * width + col];
                }
            }
            if let Some(addend) = addend {
                sum += addend[token * width + col];
            }
            out.push(sum);
        }
    }
    Ok(out)
}

/// Standalone SwiGLU over either two contiguous halves or interleaved pairs.
pub fn swiglu(
    input: &[f64],
    width: usize,
    columns: SwigluColumns,
) -> Result<OracleOutput, Refusal> {
    if width == 0 || input.is_empty() || !input.len().is_multiple_of(width * 2) {
        return Err(Refusal("SwiGLU input shape mismatch"));
    }
    let mut values = Vec::with_capacity(input.len() / 2);
    let mut sum_abs_products = Vec::with_capacity(input.len() / 2);
    for row in input.chunks_exact(width * 2) {
        for col in 0..width {
            let (gate, up) = match columns {
                SwigluColumns::ConcatenatedHalves => (row[col], row[width + col]),
                SwigluColumns::Interleaved => (row[2 * col], row[2 * col + 1]),
            };
            let value = SiluGlu::UNCLAMPED.pair(gate, up).2;
            values.push(value);
            sum_abs_products.push(value.abs());
        }
    }
    OracleOutput::new(values, sum_abs_products, 2)
}

pub fn sharded_embedding_gather(
    table: &[f64],
    shard_vocab: usize,
    width: usize,
    vocab_offset: usize,
    token_ids: &[usize],
) -> Result<Vec<f64>, Refusal> {
    if shard_vocab == 0 || width == 0 || table.len() != shard_vocab * width {
        return Err(Refusal("embedding shard shape mismatch"));
    }
    let mut out = vec![0.0; token_ids.len() * width];
    for (row, &id) in token_ids.iter().enumerate() {
        if let Some(local) = id.checked_sub(vocab_offset).filter(|&v| v < shard_vocab) {
            out[row * width..(row + 1) * width]
                .copy_from_slice(&table[local * width..(local + 1) * width]);
        }
    }
    Ok(out)
}

pub struct RingKvWrite<'a> {
    pub k_ring: &'a mut [f64],
    pub v_ring: &'a mut [f64],
    pub k_new: &'a [f64],
    pub v_new: &'a [f64],
    pub positions: &'a [usize],
    pub token_batch: &'a [usize],
    pub slot_ids: &'a [usize],
    pub last_positions: &'a [usize],
    pub ring: usize,
    pub kv_heads: usize,
    pub qk_dim: usize,
    pub v_dim: usize,
}

pub fn ring_kv_write(write: RingKvWrite<'_>) -> Result<(), Refusal> {
    let RingKvWrite {
        k_ring,
        v_ring,
        k_new,
        v_new,
        positions,
        token_batch,
        slot_ids,
        last_positions,
        ring,
        kv_heads,
        qk_dim,
        v_dim,
    } = write;
    let tokens = positions.len();
    let slot_capacity = if ring == 0 || kv_heads == 0 || qk_dim == 0 {
        0
    } else {
        k_ring.len() / (ring * kv_heads * qk_dim)
    };
    if ring == 0
        || kv_heads == 0
        || qk_dim == 0
        || v_dim == 0
        || token_batch.len() != tokens
        || k_new.len() != tokens * kv_heads * qk_dim
        || v_new.len() != tokens * kv_heads * v_dim
        || k_ring.len() != slot_capacity * ring * kv_heads * qk_dim
        || v_ring.len() != slot_capacity * ring * kv_heads * v_dim
        || token_batch
            .iter()
            .any(|&b| b >= slot_ids.len() || b >= last_positions.len())
        || slot_ids.iter().any(|&s| s >= slot_capacity)
    {
        return Err(Refusal("ring KV write shape mismatch"));
    }
    for token in 0..tokens {
        let batch = token_batch[token];
        let position = positions[token];
        if position.saturating_add(ring) <= last_positions[batch] {
            continue;
        }
        let slot = slot_ids[batch];
        let cache_token = slot * ring + position % ring;
        for head in 0..kv_heads {
            let k_dst = (cache_token * kv_heads + head) * qk_dim;
            let k_src = (token * kv_heads + head) * qk_dim;
            k_ring[k_dst..k_dst + qk_dim].copy_from_slice(&k_new[k_src..k_src + qk_dim]);
            let v_dst = (cache_token * kv_heads + head) * v_dim;
            let v_src = (token * kv_heads + head) * v_dim;
            v_ring[v_dst..v_dst + v_dim].copy_from_slice(&v_new[v_src..v_src + v_dim]);
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogitsTopK {
    pub ids: Vec<usize>,
    pub log_probs: Vec<f64>,
}

pub fn logits_topk_log_probs(
    logits: &[f64],
    live_width: usize,
    k: usize,
) -> Result<LogitsTopK, Refusal> {
    if live_width == 0
        || live_width > logits.len()
        || k > live_width
        || logits[..live_width].iter().any(|v| !v.is_finite())
    {
        return Err(Refusal("logits live width or top-k invalid"));
    }
    if k == 0 || k > 20 {
        return Err(Refusal("logits top-k shape or values invalid"));
    }
    let mut head = StreamingHead::new(k, Ties::LowerIndex, None);
    head.push(&logits[..live_width])?;
    let (ids, log_probs) = head.finish()?.top.into_iter().unzip();
    Ok(LogitsTopK { ids, log_probs })
}

/// A logit's rank in the sharded argmax's order
/// (`RowQuery::ShardedArgmaxPartial`): a larger rank wins. Non-NaN values
/// rank in their IEEE order with -0 as +0, and every NaN ranks zero, below
/// -inf.
pub fn argmax_rank(value: f32) -> u32 {
    if value.is_nan() {
        return 0;
    }
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// The pair of `candidates` (score, global ID) that wins under the sharded
/// argmax's order: the highest rank, equal ranks to the lower ID.
fn argmax_winner(candidates: impl IntoIterator<Item = (f32, usize)>) -> Option<(f32, usize)> {
    candidates.into_iter().reduce(|best, next| {
        let (best_rank, next_rank) = (argmax_rank(best.0), argmax_rank(next.0));
        if next_rank > best_rank || (next_rank == best_rank && next.1 < best.1) {
            next
        } else {
            best
        }
    })
}

/// Each contiguous shard's argmax pair of one row
/// (`RowQuery::ShardedArgmaxPartial`): its largest logit's bits in the
/// contract's order, equal logits to the lower global ID, and that ID.
/// Shards hold F32 values (BF16 logits widen exactly first); the first
/// starts at global ID 0 and each other after the one before it.
pub fn sharded_argmax_partials(shards: &[&[f32]]) -> Result<Vec<(f32, usize)>, Refusal> {
    if shards.is_empty() || shards.iter().any(|shard| shard.is_empty()) {
        return Err(Refusal("sharded argmax has an empty shard"));
    }
    let mut offset = 0_usize;
    let mut partials = Vec::with_capacity(shards.len());
    for shard in shards {
        let winner = argmax_winner(shard.iter().enumerate().map(|(id, &v)| (v, offset + id)))
            .expect("a shard is not empty");
        partials.push(winner);
        offset += shard.len();
    }
    Ok(partials)
}

/// The global ID of the argmax over a row's partial pairs
/// (`RowQuery::ShardedArgmaxFinal`): the largest score in the contract's
/// order, equal scores to the lower ID.
pub fn sharded_argmax_final(partials: &[(f32, usize)]) -> Result<usize, Refusal> {
    argmax_winner(partials.iter().copied())
        .map(|(_, id)| id)
        .ok_or(Refusal("invalid sharded argmax partials"))
}

pub fn sharded_argmax(shards: &[&[f32]]) -> Result<usize, Refusal> {
    sharded_argmax_final(&sharded_argmax_partials(shards)?)
}

pub fn all_reduce(
    rank_inputs: &[Vec<f64>],
    residual: Option<&[f64]>,
    rmsnorm_weights: Option<(&[f64], f64)>,
) -> Result<OracleOutput, Refusal> {
    if rank_inputs.is_empty()
        || rank_inputs[0].is_empty()
        || rank_inputs.iter().any(|r| r.len() != rank_inputs[0].len())
        || residual.is_some_and(|r| r.len() != rank_inputs[0].len())
    {
        return Err(Refusal("all-reduce rank or residual shape mismatch"));
    }
    let width = rank_inputs[0].len();
    let mut values = vec![0.0; width];
    let mut s = vec![0.0; width];
    for rank in rank_inputs {
        for (i, &value) in rank.iter().enumerate() {
            values[i] += value;
            s[i] += value.abs();
        }
    }
    if let Some(residual) = residual {
        for (i, &value) in residual.iter().enumerate() {
            values[i] += value;
            s[i] += value.abs();
        }
    }
    if let Some((weights, eps)) = rmsnorm_weights {
        return rmsnorm(&values, width, weights, eps, RmsNormKind::Plain, None);
    }
    OracleOutput::new(values, s, rank_inputs.len())
}

/// The canonical BF16 NaN the BF16-partials and rounded all-reduces write
/// for every NaN sum, whatever the signs and payloads of the NaNs that led
/// to it.
pub const BF16_PARTIALS_NAN: u16 = 0x7fff;

/// The all-reduce of BF16 partials (`CommQuery::AllReduce` with `dtype`
/// BF16, unfused): per element, each rank's F32 value rounded once to BF16
/// (round to nearest even: from F32 magnitude 0x7f7f8000, the midpoint
/// above the largest finite BF16, to an infinity), the ranks'
/// BF16 values summed in F32 in rank order from +0.0 under IEEE addition,
/// the sum rounded once to BF16; a NaN sum (a NaN partial, or infinities of
/// opposite signs) gives [`BF16_PARTIALS_NAN`]. Every rank's output holds
/// these bits.
pub fn all_reduce_bf16_partials(rank_inputs: &[Vec<f32>]) -> Result<Vec<u16>, Refusal> {
    if rank_inputs.is_empty()
        || rank_inputs[0].is_empty()
        || rank_inputs.iter().any(|r| r.len() != rank_inputs[0].len())
    {
        return Err(Refusal("all-reduce rank shape mismatch"));
    }
    Ok((0..rank_inputs[0].len())
        .map(|i| {
            let sum = rank_inputs
                .iter()
                .fold(0.0_f32, |sum, rank| sum + bf16_to_f32(f32_to_bf16(rank[i])));
            match sum.is_nan() {
                true => BF16_PARTIALS_NAN,
                false => f32_to_bf16(sum),
            }
        })
        .collect())
}

/// The rounded all-reduce (`CommQuery::AllReduceRounded`): per element,
/// the ranks' F32 values summed in F32 in rank order from +0.0 under IEEE
/// addition and the sum rounded once to BF16 (round to nearest even, an
/// infinity from F32 magnitude 0x7f7f8000); a NaN sum (a NaN value, or
/// infinities of opposite signs) gives [`BF16_PARTIALS_NAN`]. Every rank's
/// output holds these bits.
pub fn all_reduce_rounded(rank_inputs: &[Vec<f32>]) -> Result<Vec<u16>, Refusal> {
    if rank_inputs.is_empty()
        || rank_inputs[0].is_empty()
        || rank_inputs.iter().any(|r| r.len() != rank_inputs[0].len())
    {
        return Err(Refusal("all-reduce rank shape mismatch"));
    }
    Ok((0..rank_inputs[0].len())
        .map(|i| {
            let sum = rank_inputs.iter().fold(0.0_f32, |sum, rank| sum + rank[i]);
            match sum.is_nan() {
                true => BF16_PARTIALS_NAN,
                false => f32_to_bf16(sum),
            }
        })
        .collect())
}

pub fn all_gather(rank_inputs: &[Vec<f64>]) -> Result<Vec<f64>, Refusal> {
    if rank_inputs.is_empty()
        || rank_inputs[0].is_empty()
        || rank_inputs.iter().any(|r| r.len() != rank_inputs[0].len())
    {
        return Err(Refusal("all-gather rank shape mismatch"));
    }
    Ok(rank_inputs.concat())
}

#[derive(Clone, Debug, PartialEq)]
pub struct CheckFailure {
    pub index: usize,
    pub actual: f64,
    pub reference: f64,
    pub bound: f64,
    pub error: f64,
}

fn half_bf16_ulp(reference: f64) -> f64 {
    if reference == 0.0 || reference.abs() < 2_f64.powi(-126) {
        2_f64.powi(-134)
    } else {
        2_f64.powi((reference.abs().log2().floor() as i32).max(-126) - 8)
    }
}

/// Neumaier-compensated running sum: terms are added in call order and the
/// rounding error of each add is carried separately until [`Self::value`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NeumaierSum {
    sum: f64,
    compensation: f64,
}

impl NeumaierSum {
    pub fn add(&mut self, term: f64) {
        let next = self.sum + term;
        self.compensation += if self.sum.abs() >= term.abs() {
            (self.sum - next) + term
        } else {
            (term - next) + self.sum
        };
        self.sum = next;
    }

    pub fn value(&self) -> f64 {
        self.sum + self.compensation
    }
}

/// Neumaier-compensated sum of `terms` in iteration order.
pub fn compensated_sum(terms: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = NeumaierSum::default();
    terms.into_iter().for_each(|term| sum.add(term));
    sum.value()
}

/// Neumaier-compensated sum of `terms` in iteration order, with the plain
/// sum of their absolute values (the S of [`accumulation_bound`] when the
/// terms are a reduction's products).
pub fn compensated_sum_with_magnitude(terms: impl IntoIterator<Item = f64>) -> (f64, f64) {
    let (mut sum, mut magnitude) = (NeumaierSum::default(), 0.0_f64);
    for term in terms {
        sum.add(term);
        magnitude += term.abs();
    }
    (sum.value(), magnitude)
}

/// The accumulation gate's allowance for an F32 reduction of length `k`
/// whose products' absolute values sum to `s`: 2^-22*K*S.
pub fn accumulation_bound(k: usize, s: f64) -> f64 {
    2_f64.powi(-22) * k as f64 * s
}

/// Numerical contract's bound on element `index`: half a BF16 ulp of the
/// reference (for BF16 outputs) plus 2^-22*K*S.
pub fn numeric_bound(reference: &OracleOutput, index: usize, output: DType) -> f64 {
    contract_bound(
        reference.values[index],
        reference.reduction_lengths[index],
        reference.sum_abs_products[index],
        output,
    )
}

fn contract_bound(reference: f64, k: usize, s: f64, output: DType) -> f64 {
    let round = if output == DType::Bf16 {
        half_bf16_ulp(reference)
    } else {
        0.0
    };
    round + accumulation_bound(k, s)
}

/// Largest `|actual - reference|` over its [`numeric_bound`], for reports.
/// NaN ratios are skipped, so an all-NaN or empty output reports 0.
pub fn max_bound_use(actual: &[f32], reference: &OracleOutput, output: DType) -> f64 {
    actual
        .iter()
        .zip(&reference.values)
        .zip(
            reference
                .sum_abs_products
                .iter()
                .zip(&reference.reduction_lengths),
        )
        .map(|((&a, &r), (&s, &k))| (f64::from(a) - r).abs() / contract_bound(r, k, s, output))
        .fold(0.0, f64::max)
}

/// Numerical contract: |y-ref| <= half BF16 ulp (if BF16) + 2^-22*K*S.
/// The returned failure is the element with the largest excess over its bound.
pub fn check_numeric(
    actual: &[f64],
    reference: &OracleOutput,
    output: DType,
) -> Result<(), CheckFailure> {
    if actual.len() != reference.values.len()
        || reference.values.len() != reference.sum_abs_products.len()
        || reference.values.len() != reference.reduction_lengths.len()
    {
        return Err(CheckFailure {
            index: actual.len().min(reference.values.len()),
            actual: f64::NAN,
            reference: f64::NAN,
            bound: 0.0,
            error: f64::INFINITY,
        });
    }
    let mut worst = None;
    for (index, (&y, (&r, &s))) in actual
        .iter()
        .zip(reference.values.iter().zip(&reference.sum_abs_products))
        .enumerate()
    {
        let bound = numeric_bound(reference, index, output);
        let error = (y - r).abs();
        if !y.is_finite() || !r.is_finite() || !s.is_finite() || error > bound {
            let failure = CheckFailure {
                index,
                actual: y,
                reference: r,
                bound,
                error,
            };
            if worst
                .as_ref()
                .is_none_or(|w: &CheckFailure| error - bound > w.error - w.bound)
            {
                worst = Some(failure);
            }
        }
    }
    worst.map_or(Ok(()), Err)
}

/// Bit-exact gate for codecs, quantizers, repacks and cache layouts.
pub fn check_exact<T: PartialEq + std::fmt::Debug>(
    actual: &[T],
    reference: &[T],
) -> Result<(), usize> {
    if actual.len() != reference.len() {
        return Err(actual.len().min(reference.len()));
    }
    actual
        .iter()
        .zip(reference)
        .position(|(a, b)| a != b)
        .map_or(Ok(()), Err)
}

/// Per-token cosine similarity. Both zero vectors match; one zero vector fails.
pub fn check_layer_cosine(
    actual: &[f64],
    reference: &[f64],
    width: usize,
) -> Result<(), CheckFailure> {
    if width == 0 || actual.len() != reference.len() || !actual.len().is_multiple_of(width) {
        return Err(CheckFailure {
            index: 0,
            actual: f64::NAN,
            reference: f64::NAN,
            bound: 0.9999,
            error: f64::INFINITY,
        });
    }
    let mut worst = None;
    for (token, (a, r)) in actual
        .chunks_exact(width)
        .zip(reference.chunks_exact(width))
        .enumerate()
    {
        let dot = a.iter().zip(r).map(|(x, y)| x * y).sum::<f64>();
        let an = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let rn = r.iter().map(|x| x * x).sum::<f64>().sqrt();
        let cosine = if an == 0.0 && rn == 0.0 {
            1.0
        } else if an == 0.0 || rn == 0.0 {
            0.0
        } else {
            dot / (an * rn)
        };
        if !cosine.is_finite() || cosine < 0.9999 {
            let failure = CheckFailure {
                index: token,
                actual: cosine,
                reference: 1.0,
                bound: 0.9999,
                error: 1.0 - cosine,
            };
            if worst
                .as_ref()
                .is_none_or(|w: &CheckFailure| cosine < w.actual)
            {
                worst = Some(failure);
            }
        }
    }
    worst.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each partial rounds once before the sum, the sum runs in rank order
    /// from +0.0 in F32 and rounds once.
    #[test]
    fn bf16_partials_round_each_rank_then_sum_in_rank_order() {
        let tie = 1.0 + 2_f32.powi(-8);
        let big = 2_f32.powi(24);
        let ranks = vec![
            vec![tie, big, -0.0, 1.5],
            vec![2_f32.powi(-8), 1.0, -0.0, 1.0],
            vec![0.0, -big, -0.0, f32::NAN],
        ];
        let sums = all_reduce_bf16_partials(&ranks).unwrap();
        // 1 + 2^-8 rounds to 1 alone; unrounded, the F32 sum would be
        // 1 + 2^-7.
        assert_eq!(sums[0], f32_to_bf16(1.0));
        // 2^24 + 1 is 2^24 in F32, so the rank order gives zero; the
        // reverse order would give one.
        assert_eq!(sums[1], f32_to_bf16(0.0));
        // The sum starts from +0.0.
        assert_eq!(sums[2], 0x0000);
        assert_eq!(sums[3], BF16_PARTIALS_NAN);
        let reversed: Vec<_> = ranks.iter().rev().cloned().collect();
        assert_eq!(
            all_reduce_bf16_partials(&reversed).unwrap()[1],
            f32_to_bf16(1.0)
        );
        assert!(all_reduce_bf16_partials(&[]).is_err());
        // Special values: every NaN sum is the canonical NaN, whatever its
        // inputs' signs and payloads; infinities add as IEEE says; a
        // partial or a finite F32 sum rounds to an infinity from the
        // midpoint above the largest finite BF16 (the tie included, as
        // 0x7f7f is odd) and to 0x7f7f below it.
        let nan = |sign: u32, payload: u32| f32::from_bits(sign << 31 | 0x7fc0_0000 | payload);
        let cases = [
            ([nan(0, 0x1234), 1.0], BF16_PARTIALS_NAN),
            ([1.0, nan(1, 0x55)], BF16_PARTIALS_NAN),
            ([f32::INFINITY, 1.0], 0x7f80),
            ([-1.0, f32::NEG_INFINITY], 0xff80),
            ([f32::INFINITY, f32::NEG_INFINITY], BF16_PARTIALS_NAN),
            ([f32::MAX, -f32::MAX], BF16_PARTIALS_NAN),
            ([3.0e38, 3.0e38], 0x7f80),
            ([bf16_to_f32(0x7f7f), 1.0e36], 0x7f80),
            ([f32::from_bits(0x7f7f_0001), 0.0], 0x7f7f),
            ([f32::from_bits(0x7f7f_7fff), 0.0], 0x7f7f),
            ([f32::from_bits(0x7f7f_8000), 0.0], 0x7f80),
            ([f32::from_bits(0xff7f_7fff), 0.0], 0xff7f),
            ([f32::from_bits(0xff7f_8000), 0.0], 0xff80),
            // Sums of finite BF16 partials: F32 0x7f7f0001, below the
            // midpoint, and 0x7f7f8000, at it.
            ([bf16_to_f32(0x7f7f), bf16_to_f32(0x7380)], 0x7f7f),
            ([bf16_to_f32(0x7f7f), bf16_to_f32(0x7b00)], 0x7f80),
            ([1.0e-40, 0.0], f32_to_bf16(1.0e-40)),
        ];
        for (inputs, expected) in cases {
            let ranks = inputs.iter().map(|&value| vec![value]).collect::<Vec<_>>();
            assert_eq!(
                all_reduce_bf16_partials(&ranks).unwrap(),
                [expected],
                "{inputs:?}"
            );
        }
        assert!(all_reduce_bf16_partials(&[vec![1.0], vec![1.0, 2.0]]).is_err());
    }

    /// The F32 values sum in rank order from +0.0 unrounded, and only the
    /// sum rounds once.
    #[test]
    fn rounded_all_reduce_sums_f32_in_rank_order_then_rounds_once() {
        let tie = 1.0 + 2_f32.powi(-8);
        let big = 2_f32.powi(24);
        let ranks = vec![
            vec![tie, big, -0.0, 1.5],
            vec![2_f32.powi(-8), 1.0, -0.0, 1.0],
            vec![0.0, -big, -0.0, f32::NAN],
        ];
        let sums = all_reduce_rounded(&ranks).unwrap();
        // 1 + 2^-8 + 2^-8 is 1 + 2^-7 in F32, a BF16 value; rounding each
        // value first would give 1.
        assert_eq!(sums[0], f32_to_bf16(1.0 + 2_f32.powi(-7)));
        assert_ne!(sums[0], all_reduce_bf16_partials(&ranks).unwrap()[0]);
        // 2^24 + 1 is 2^24 in F32, so the rank order gives zero; the
        // reverse order would give one.
        assert_eq!(sums[1], f32_to_bf16(0.0));
        let reversed: Vec<_> = ranks.iter().rev().cloned().collect();
        assert_eq!(all_reduce_rounded(&reversed).unwrap()[1], f32_to_bf16(1.0));
        // The sum starts from +0.0.
        assert_eq!(sums[2], 0x0000);
        assert_eq!(sums[3], BF16_PARTIALS_NAN);
        // One rounding of the F32 sum: halfway cases round to even.
        let ranks = vec![vec![1.0], vec![2_f32.powi(-8)]];
        assert_eq!(all_reduce_rounded(&ranks).unwrap(), [f32_to_bf16(1.0)]);
        let ranks = vec![vec![1.0], vec![3.0 * 2_f32.powi(-8)]];
        assert_eq!(
            all_reduce_rounded(&ranks).unwrap(),
            [f32_to_bf16(1.0 + 2_f32.powi(-6))]
        );
        let nan = |sign: u32, payload: u32| f32::from_bits(sign << 31 | 0x7fc0_0000 | payload);
        let cases = [
            ([nan(1, 0x55), 1.0], BF16_PARTIALS_NAN),
            ([f32::INFINITY, 1.0], 0x7f80),
            ([f32::INFINITY, f32::NEG_INFINITY], BF16_PARTIALS_NAN),
            ([3.0e38, 3.0e38], 0x7f80),
            ([f32::from_bits(0x7f7f_7fff), 0.0], 0x7f7f),
            ([f32::from_bits(0x7f7f_8000), 0.0], 0x7f80),
            ([f32::from_bits(0xff7f_8000), 0.0], 0xff80),
        ];
        for (inputs, expected) in cases {
            let ranks = inputs.iter().map(|&value| vec![value]).collect::<Vec<_>>();
            assert_eq!(
                all_reduce_rounded(&ranks).unwrap(),
                [expected],
                "{inputs:?}"
            );
        }
        assert!(all_reduce_rounded(&[]).is_err());
        assert!(all_reduce_rounded(&[vec![1.0], vec![1.0, 2.0]]).is_err());
    }

    #[test]
    fn compensated_sums_recover_cancelled_terms() {
        assert_eq!(compensated_sum([1.0, 1e100, 1.0, -1e100]), 2.0);
        assert_eq!(compensated_sum([1e16, 1.0, -1e16]), 1.0);
        assert_eq!(compensated_sum([]), 0.0);
        assert_eq!(
            compensated_sum_with_magnitude([1e16, 1.0, -1e16, 1.0]),
            (2.0, 2e16 + 2.0)
        );
        // The magnitude is a plain sum: each 1.0 rounds away against 1e16,
        // while the compensated value keeps all four.
        assert_eq!(
            compensated_sum_with_magnitude([1e16, 1.0, 1.0, 1.0, 1.0]),
            (1e16 + 4.0, 1e16)
        );
        let mut sum = NeumaierSum::default();
        [1.0, 1e100, 1.0, -1e100]
            .into_iter()
            .for_each(|t| sum.add(t));
        assert_eq!(sum.value(), 2.0);
        assert_eq!(NeumaierSum::default().value(), 0.0);
    }

    #[test]
    fn accumulation_bound_is_two_to_the_minus_22_k_s() {
        assert_eq!(accumulation_bound(128, 2.0), 2_f64.powi(-14));
        assert_eq!(accumulation_bound(16_384, 128.0), 0.5);
        let reference = OracleOutput::new(vec![1.0, 0.0], vec![4.0, 1.0], 128).unwrap();
        assert_eq!(numeric_bound(&reference, 0, DType::F32), 2_f64.powi(-13));
        assert_eq!(
            numeric_bound(&reference, 0, DType::Bf16),
            2_f64.powi(-8) + 2_f64.powi(-13)
        );
        // Element 1 uses a quarter of its F32 bound, element 0 half of its.
        let actual = [1.0 + 2_f32.powi(-14), 2_f32.powi(-17)];
        assert_eq!(max_bound_use(&actual, &reference, DType::F32), 0.5);
        assert_eq!(max_bound_use(&[f32::NAN], &reference, DType::F32), 0.0);
        assert_eq!(max_bound_use(&[], &reference, DType::F32), 0.0);
    }

    /// The routed sum adds a token's local routes in top-k order from +0 in
    /// F32, then the addend once: an addend first would round differently,
    /// and a token with no local route keeps `+0 + addend`.
    #[test]
    fn local_routed_sums_add_routes_in_order_then_the_addend() {
        // Two tokens, top 2, width 1, rank 1 of two local experts each:
        // token 0 routes to experts 2 and 3 (both local), token 1 to 0
        // (another rank's) and -1 (none).
        let ids = [2, 3, 0, -1];
        let routed = [1.0, 1.0, 5.0, 7.0];
        let sum = local_routed_sum(&routed, &ids, 2, 1, 2, 1, None).unwrap();
        assert_eq!(sum, [2.0, 0.0]);
        assert!(sum[1].is_sign_positive());
        // (1 + 1) + 2^24 is exact; the addend first would round each 1 away.
        let big = 2_f32.powi(24);
        let added = local_routed_sum(&routed, &ids, 2, 1, 2, 1, Some(&[big, -0.0])).unwrap();
        assert_eq!(added[0], big + 2.0);
        assert_eq!((big + 1.0) + 1.0, big);
        // +0 + -0 is +0.
        assert_eq!(added[1].to_bits(), 0);
        // Rank 0 keeps token 1's first route instead.
        let other = local_routed_sum(&routed, &ids, 2, 1, 2, 0, None).unwrap();
        assert_eq!(other, [0.0, 5.0]);
        assert!(local_routed_sum(&routed, &ids[..3], 2, 1, 2, 1, None).is_err());
        assert!(local_routed_sum(&routed, &ids, 2, 1, 2, 1, Some(&[1.0])).is_err());
    }

    #[test]
    fn gemm_swiglu_and_numeric_gate() {
        let a = [2.0, 3.0];
        let b = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0];
        let out = gemm(
            &a,
            &b,
            GemmShape {
                m: 1,
                n: 4,
                k: 2,
                experts: 1,
            },
            &[],
            GemmEpilogue {
                residual: None,
                swiglu: Some(SwigluColumns::ConcatenatedHalves),
            },
        )
        .unwrap();
        assert_eq!(out.values.len(), 2);
        check_numeric(&out.values, &out, DType::F32).unwrap();
        let mut wrong = out.values.clone();
        wrong[1] += 1.0;
        assert_eq!(
            check_numeric(&wrong, &out, DType::F32).unwrap_err().index,
            1
        );
    }

    #[test]
    fn grouped_gemm_uses_each_experts_weight_rows() {
        let out = gemm(
            &[2.0, 3.0],
            &[10.0, 20.0, 30.0, 40.0],
            GemmShape {
                m: 2,
                n: 2,
                k: 1,
                experts: 2,
            },
            &[
                ExpertRowRange {
                    expert: 0,
                    start: 0,
                    end: 1,
                },
                ExpertRowRange {
                    expert: 1,
                    start: 1,
                    end: 2,
                },
            ],
            GemmEpilogue {
                residual: Some(&[1.0, 1.0, 1.0, 1.0]),
                swiglu: None,
            },
        )
        .unwrap();
        assert_eq!(out.values, [21.0, 41.0, 91.0, 121.0]);
        assert_eq!(out.sum_abs_products, [20.0, 40.0, 90.0, 120.0]);
    }

    #[test]
    fn router_selection_and_permute() {
        let routed = router(
            &[0.0, 0.0, 0.0],
            Some(&[0.0, 1.0, 0.0]),
            2,
            RouterKind::SigmoidBiasSelectNormalized {
                scaling_factor: 2.0,
            },
        )
        .unwrap();
        assert_eq!(routed.expert_ids, vec![1, 0]);
        assert!((routed.weights[0] - 1.0).abs() < 1e-10);
        let perm = expert_permute(&[2, 0, 2], 3, 4).unwrap();
        assert_eq!(perm.forward.len(), 8);
        assert_eq!(perm.forward[perm.inverse[0]], Some(0));
    }
    #[test]
    fn rope_and_logits_ties() {
        assert_eq!(
            topk(&[1.0, 2.0, 2.0], 2, Ties::LowerIndex, NanKeys::Refuse, None).unwrap(),
            vec![1, 2]
        );
        let out = rope(
            &[1.0, 0.0, 7.0],
            &[0],
            3,
            2,
            10000.0,
            RotaryStyle::GptJ,
            None,
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 0.0, 7.0]);
    }

    #[test]
    fn standalone_row_oracles_cover_layout_and_live_vocab() {
        assert_eq!(
            swiglu(&[0.0, 2.0, 3.0, 4.0], 2, SwigluColumns::Interleaved)
                .unwrap()
                .values,
            vec![0.0, 3.0 / (1.0 + (-3.0_f64).exp()) * 4.0]
        );
        assert_eq!(
            sharded_embedding_gather(&[1.0, 2.0, 3.0, 4.0], 2, 2, 10, &[9, 10, 11]).unwrap(),
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        );
        let top = logits_topk_log_probs(&[2.0, 2.0, 99.0], 2, 2).unwrap();
        assert_eq!(top.ids, vec![0, 1]);
        assert!(
            top.log_probs
                .iter()
                .all(|&value| (value + 2.0_f64.ln()).abs() < 1e-15)
        );
        let mut k_ring = vec![0.0; 4];
        let mut v_ring = vec![0.0; 4];
        ring_kv_write(RingKvWrite {
            k_ring: &mut k_ring,
            v_ring: &mut v_ring,
            k_new: &[1.0, 2.0, 3.0],
            v_new: &[4.0, 5.0, 6.0],
            positions: &[0, 1, 2],
            token_batch: &[0, 0, 0],
            slot_ids: &[0],
            last_positions: &[2],
            ring: 2,
            kv_heads: 1,
            qk_dim: 1,
            v_dim: 1,
        })
        .unwrap();
        assert_eq!(&k_ring[..2], &[3.0, 2.0]);
        assert_eq!(&v_ring[..2], &[6.0, 5.0]);
    }

    #[test]
    fn paged_attention_sink_changes_denominator() {
        let query = AttentionQuery {
            q_heads: 1,
            kv_heads: 1,
            qk_dim: 64,
            v_dim: 64,
            page_size: 1,
            kv_dtype: DType::Bf16,
            layout: AttentionLayout::Paged,
            causal: true,
            window_left: None,
            sink: true,
            softcap: None,
            softmax_scale: 1.0,
            phase: crate::contract::AttentionPhase::Decode,
            max_context_capacity: 1,
            max_batch_capacity: 1,
        };
        let mut q = vec![0.0; 64];
        q[0] = 1.0;
        let mut k = vec![0.0; 64];
        k[0] = 1.0;
        let v = vec![2.0; 64];
        let table = vec![vec![0]];
        let input = PagedAttention {
            q: &q,
            keys: &k,
            values: &v,
            page_tables: &table,
            query_lengths: &[1],
            query_starts: &[0],
            key_lengths: &[1],
            sink_logits: Some(&[1.0]),
        };
        let out = attention(&query, &input).unwrap();
        assert_eq!(out.values[0], 1.0);
    }

    #[test]
    fn ring_prefill_reads_prefix_and_new_chunk_with_sink() {
        let query = AttentionQuery {
            q_heads: 1,
            kv_heads: 1,
            qk_dim: 64,
            v_dim: 64,
            page_size: 4,
            kv_dtype: DType::Bf16,
            layout: AttentionLayout::Ring,
            causal: true,
            window_left: Some(1),
            sink: true,
            softcap: None,
            softmax_scale: 1.0,
            phase: crate::contract::AttentionPhase::Prefill,
            max_context_capacity: 5,
            max_batch_capacity: 1,
        };
        let mut cache_values = vec![0.0; 4 * 64];
        cache_values[2 * 64..3 * 64].fill(2.0);
        let mut new_values = vec![4.0; 2 * 64];
        new_values[64..].fill(8.0);
        let output = ring_attention(
            &query,
            &RingAttention {
                q: &[0.0; 2 * 64],
                keys: &[0.0; 4 * 64],
                values: &cache_values,
                slot_ids: &[0],
                query_lengths: &[2],
                prefix_lengths: &[3],
                k_new: Some(&[0.0; 2 * 64]),
                v_new: Some(&new_values),
                sink_logits: Some(&[0.0]),
            },
        )
        .unwrap();
        assert!(output.values[..64].iter().all(|v| *v == 2.0));
        assert!(output.values[64..].iter().all(|v| *v == 4.0));
        assert!(output.reduction_lengths.iter().all(|&k| k == 3));
        let mut decode = query;
        decode.phase = crate::contract::AttentionPhase::Decode;
        cache_values[3 * 64..].fill(4.0);
        cache_values[..64].fill(8.0);
        let decoded = ring_attention(
            &decode,
            &RingAttention {
                q: &[0.0; 64],
                keys: &[0.0; 4 * 64],
                values: &cache_values,
                slot_ids: &[0],
                query_lengths: &[1],
                prefix_lengths: &[4],
                k_new: None,
                v_new: None,
                sink_logits: Some(&[0.0]),
            },
        )
        .unwrap();
        assert!(decoded.values.iter().all(|v| *v == 4.0));
    }

    /// Zero keys give uniform weights, so each output is the mean of the
    /// visible keys' values, which here equal their positions.
    fn ring_verify_means(causal: bool) -> Vec<f64> {
        let query = AttentionQuery {
            q_heads: 2,
            kv_heads: 1,
            qk_dim: 64,
            v_dim: 64,
            page_size: 8,
            kv_dtype: DType::Bf16,
            layout: AttentionLayout::Ring,
            causal,
            window_left: Some(3),
            sink: false,
            softcap: None,
            softmax_scale: 1.0,
            phase: crate::contract::AttentionPhase::Verify,
            max_context_capacity: 64,
            max_batch_capacity: 2,
        };
        // Item 0: prefix 6 in slot 1, three queries; item 1: prefix 2 in
        // slot 0, one query. Ring entries hold their position.
        let mut values = vec![0.0; 2 * 8 * 64];
        for position in 0..6 {
            values[(8 + position % 8) * 64..(8 + position % 8 + 1) * 64].fill(position as f64);
        }
        for position in 0..2 {
            values[(position % 8) * 64..(position % 8 + 1) * 64].fill(position as f64);
        }
        let mut v_new = vec![0.0; 4 * 64];
        for (row, position) in [6.0, 7.0, 8.0, 2.0].into_iter().enumerate() {
            v_new[row * 64..(row + 1) * 64].fill(position);
        }
        let output = ring_attention(
            &query,
            &RingAttention {
                q: &[0.0; 4 * 2 * 64],
                keys: &[0.0; 2 * 8 * 64],
                values: &values,
                slot_ids: &[1, 0],
                query_lengths: &[3, 1],
                prefix_lengths: &[6, 2],
                k_new: Some(&[0.0; 4 * 64]),
                v_new: Some(&v_new),
                sink_logits: None,
            },
        )
        .unwrap();
        (0..4).map(|row| output.values[row * 2 * 64]).collect()
    }

    #[test]
    fn ring_verify_reads_prefix_from_ring_and_own_rows_contiguously() {
        // Query at position p sees [p - 3, p].
        assert_eq!(ring_verify_means(true), vec![4.5, 5.5, 6.5, 1.0]);
    }

    #[test]
    fn bidirectional_verify_shows_every_query_its_whole_item() {
        // Query at position p sees [p - 3, last position of its item].
        assert_eq!(ring_verify_means(false), vec![5.5, 6.0, 6.5, 1.0]);
    }

    #[test]
    fn attention_uses_each_querys_visible_key_count() {
        let query = AttentionQuery {
            q_heads: 1,
            kv_heads: 1,
            qk_dim: 64,
            v_dim: 64,
            page_size: 1,
            kv_dtype: DType::Bf16,
            layout: AttentionLayout::Paged,
            causal: true,
            window_left: None,
            sink: false,
            softcap: None,
            softmax_scale: 1.0,
            phase: crate::contract::AttentionPhase::Prefill,
            max_context_capacity: 2,
            max_batch_capacity: 1,
        };
        let table = vec![vec![0, 1]];
        let input = PagedAttention {
            q: &[0.0; 128],
            keys: &[0.0; 128],
            values: &[2.0; 128],
            page_tables: &table,
            query_lengths: &[2],
            query_starts: &[0],
            key_lengths: &[2],
            sink_logits: None,
        };
        let out = attention(&query, &input).unwrap();
        assert_eq!(out.reduction_lengths[0], 1);
        assert_eq!(out.reduction_lengths[64], 2);
        assert!(out.values.iter().all(|v| *v == 2.0));
    }

    #[test]
    fn gqa_sliding_window_allows_different_qk_and_v_dims() {
        let query = AttentionQuery {
            q_heads: 2,
            kv_heads: 1,
            qk_dim: 64,
            v_dim: 128,
            page_size: 2,
            kv_dtype: DType::Bf16,
            layout: AttentionLayout::Paged,
            causal: true,
            window_left: Some(0),
            sink: false,
            softcap: None,
            softmax_scale: 1.0,
            phase: crate::contract::AttentionPhase::Decode,
            max_context_capacity: 2,
            max_batch_capacity: 1,
        };
        let mut values = vec![1.0; 128];
        values.extend_from_slice(&[3.0; 128]);
        let tables = vec![vec![0]];
        let input = PagedAttention {
            q: &[0.0; 128],
            keys: &[0.0; 128],
            values: &values,
            page_tables: &tables,
            query_lengths: &[1],
            query_starts: &[1],
            key_lengths: &[2],
            sink_logits: None,
        };
        let out = attention(&query, &input).unwrap();
        assert_eq!(out.values.len(), 256);
        assert!(out.values.iter().all(|v| *v == 3.0));
        assert!(out.reduction_lengths.iter().all(|&k| k == 1));
    }

    #[test]
    fn paged_kv_layout_uses_block_table_order() {
        let mut cache = vec![0.0; 2 * 2 * 2 * 2];
        let source = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        paged_kv_write(&mut cache, &[1, 0], &[0, 2], &source, 2, 2, 2).unwrap();
        assert_eq!(&cache[8..12], &source[0..4]);
        assert_eq!(&cache[0..4], &source[4..8]);
        assert_eq!(&cache[4..8], &[0.0; 4]);
    }

    #[test]
    fn paged_kv_layout_preserves_other_layers_in_shared_pages() {
        // Two pages, three layers, each with a four-element K block and V block.
        for element_bytes in [1, 2] {
            let mut pool = vec![-9.0; 2 * 3 * 8];
            let page_stride_bytes = (3 * 8 * element_bytes) as u64;
            let k_base = 8;
            let v_base = 12;
            paged_kv_write_strided(
                &mut pool[k_base..],
                &[1, 0],
                &[0, 2],
                &[1.0, 2.0, 3.0, 4.0],
                2,
                1,
                2,
                page_stride_bytes,
                element_bytes,
            )
            .unwrap();
            paged_kv_write_strided(
                &mut pool[v_base..],
                &[1, 0],
                &[0, 2],
                &[5.0, 6.0, 7.0, 8.0],
                2,
                1,
                2,
                page_stride_bytes,
                element_bytes,
            )
            .unwrap();
            assert_eq!(&pool[8..10], &[3.0, 4.0]);
            assert_eq!(&pool[12..14], &[7.0, 8.0]);
            assert_eq!(&pool[32..34], &[1.0, 2.0]);
            assert_eq!(&pool[36..38], &[5.0, 6.0]);
            assert_eq!(pool.iter().filter(|&&v| v != -9.0).count(), 8);
            assert!(
                paged_kv_write_strided(
                    &mut pool[k_base..],
                    &[0],
                    &[0],
                    &[1.0, 2.0],
                    2,
                    1,
                    2,
                    3,
                    element_bytes,
                )
                .is_err()
            );
        }
    }
}

#[cfg(all(test, debug_assertions))]
mod dot_shape_tests {
    #[test]
    #[should_panic]
    fn padded_weight_row_requires_matching_activation() {
        super::dot(&[1.0; 132], &[1.0; 128]);
    }
}
