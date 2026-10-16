//! F64 reference arithmetic for the encoder contracts in the media contract.
//! Inputs are decoded operands, in row-major order unless stated otherwise.

use crate::attend::{Attend, AttendValues, KeyBias, KvRows, Normalize, window};
use crate::codec::{bf16_to_f32, f32_to_bf16};
use crate::contract::{
    DType, GeluMode, MultimodalQuery, Refusal, RotaryStyle, SwigluColumns, Validate,
};
use crate::oracle::{GemmEpilogue, GemmShape, OracleOutput, gemm};
use crate::rotary::{
    FreqTable, RotaryArithmetic, RotaryPositions, RotaryRounding, RotarySlice, Rotation,
};

fn output(values: Vec<f64>, s: Vec<f64>, k: usize) -> Result<OracleOutput, Refusal> {
    OracleOutput::new(values, s, k)
}

/// Contiguous attention for the packed query tokens in `rows`. K and V are
/// `[tokens,kv_heads,dim]`; `q_rows` is `[rows.len(),q_heads,dim]` and the
/// output is in the same order. The sequence boundaries apply to all three
/// tensors. A key-0 sink changes only the first key's logit within each
/// sequence, before the softmax. Long single-image sequences check sampled
/// rows this way.
///
/// `ContiguousAttentionPairs` has F32 output, so no BF16 rounding absorbs
/// the logits' own reduction error. Its `S` adds that term: with `K` keys,
/// `(dim / K) * sum_j p_j |v_j - o| * scale * sum_i |q_i k_ji|`, so that
/// `2^-22 K S` covers a `2^-22 dim` error in every scaled logit.
///
/// The operands are the pairs' values; the served kernel leaves out two
/// products, both inside that bound. The logits drop q lo k lo: each low
/// half is at most 2^-8 of its value (BF16 rounds to nearest with 8
/// significand bits), so the product is at most 2^-16 of |q k|, within the
/// logit term's `2^-22 dim` for `dim >= 64`. The values take [`pair_probability_terms`] BF16 terms of each
/// probability, every term times both halves of v; the probability left out
/// is at most `2^(-8 terms)` of it, so `sum_j |p_j v_j|` shrinks by at most
/// that, within `2^-22 K S` (`pair_terms_fit_the_bound`).
#[allow(clippy::too_many_arguments)]
pub fn contiguous_attention_rows(
    query: MultimodalQuery,
    rows: &[usize],
    q_rows: &[f64],
    k: &[f64],
    v: &[f64],
    cu_seqlens: &[i32],
    scale: f64,
    sinks: Option<&[f64]>,
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let (MultimodalQuery::ContiguousAttention {
        q_heads,
        kv_heads,
        head_dim,
        causal,
        window_left,
        window_right,
        key0_sink,
        capacity,
    }
    | MultimodalQuery::ContiguousAttentionPairs {
        q_heads,
        kv_heads,
        head_dim,
        causal,
        window_left,
        window_right,
        key0_sink,
        capacity,
    }) = query
    else {
        return Err(Refusal("expected contiguous attention query"));
    };
    let (qh, kh, d) = (q_heads as usize, kv_heads as usize, head_dim as usize);
    let tokens = cu_seqlens.last().copied().unwrap_or(0);
    if cu_seqlens.len() < 2
        || cu_seqlens[0] != 0
        || tokens <= 0
        || tokens as u32 > capacity
        || !cu_seqlens.windows(2).all(|x| x[0] < x[1])
        || rows.iter().any(|&row| row >= tokens as usize)
        || q_rows.len() != rows.len() * qh * d
        || k.len() != tokens as usize * kh * d
        || v.len() != k.len()
        || !scale.is_finite()
        || scale <= 0.0
        || sinks.is_some() != key0_sink
        || sinks.is_some_and(|x| x.len() != qh)
    {
        return Err(Refusal(
            "contiguous attention operands or sequence bounds mismatch",
        ));
    }
    let starts = cu_seqlens[..cu_seqlens.len() - 1]
        .iter()
        .map(|&start| start as usize)
        .collect::<Vec<_>>();
    let core = Attend {
        key_bias: sinks.map(|per_head| KeyBias {
            rows: &starts,
            per_head,
        }),
        normalize: Normalize::Weights,
        logit_reduction_bound: matches!(query, MultimodalQuery::ContiguousAttentionPairs { .. }),
        ..Attend::f64(scale)
    };
    let (keys, values) = ([k], [v]);
    let rows_of = |segments| KvRows {
        segments,
        heads: kh,
        dim: d,
    };
    let mut output = OracleOutput {
        values: Vec::with_capacity(q_rows.len()),
        sum_abs_products: Vec::with_capacity(q_rows.len()),
        reduction_lengths: Vec::with_capacity(q_rows.len()),
    };
    for (slot, &i) in rows.iter().enumerate() {
        let seq = cu_seqlens.partition_point(|&start| start as usize <= i);
        let (start, end) = (cu_seqlens[seq - 1] as usize, cu_seqlens[seq] as usize);
        let visible = window(
            i,
            start..end,
            causal,
            window_left.map(|left| left as usize),
            window_right.map(|right| right as usize),
        );
        if visible.is_empty() {
            return Err(Refusal("attention window has no keys"));
        }
        let list = visible.map(|j| j as i64).collect::<Vec<_>>();
        core.row(
            &q_rows[slot * qh * d..(slot + 1) * qh * d],
            rows_of(&keys),
            AttendValues::Rows(rows_of(&values)),
            &list,
            &mut output,
        )?;
    }
    Ok(output)
}

/// The BF16 terms each probability of a pair attention row with `keys`
/// visible keys keeps (`cuda/attention/varlen_pairs.cuh`): at least two, and
/// three below 128 keys.
pub fn pair_probability_terms(keys: usize) -> i32 {
    if keys >= 128 { 2 } else { 3 }
}

/// Population variance, affine scale and bias. Each row is independent.
pub fn layernorm_affine(
    input: &[f64],
    width: usize,
    weight: &[f64],
    bias: &[f64],
    eps: f64,
) -> Result<OracleOutput, Refusal> {
    if width == 0
        || input.is_empty()
        || !input.len().is_multiple_of(width)
        || weight.len() != width
        || bias.len() != width
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Refusal("LayerNorm shape or epsilon mismatch"));
    }
    let mut values = Vec::with_capacity(input.len());
    let mut bounds = Vec::with_capacity(input.len());
    for row in input.chunks_exact(width) {
        let mean = row.iter().sum::<f64>() / width as f64;
        let variance = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / width as f64;
        let inverse = (variance + eps).sqrt().recip();
        for ((&x, &w), &b) in row.iter().zip(weight).zip(bias) {
            let product = (x - mean) * inverse * w;
            values.push(product + b);
            bounds.push(product.abs() + b.abs());
        }
    }
    output(values, bounds, width)
}

/// RMSNorm with a weight: `x / sqrt(mean(x^2) + eps) * w`. Each row is
/// independent; S is the output's magnitude and K the width, as for
/// `layernorm_affine`.
pub fn rms_norm(
    input: &[f64],
    width: usize,
    weight: &[f64],
    eps: f64,
) -> Result<OracleOutput, Refusal> {
    if width == 0
        || input.is_empty()
        || !input.len().is_multiple_of(width)
        || weight.len() != width
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Refusal("RMSNorm shape or epsilon mismatch"));
    }
    let mut values = Vec::with_capacity(input.len());
    for row in input.chunks_exact(width) {
        let mean = row.iter().map(|x| x * x).sum::<f64>() / width as f64;
        let inverse = (mean + eps).sqrt().recip();
        values.extend(row.iter().zip(weight).map(|(&x, &w)| x * inverse * w));
    }
    output(
        values.clone(),
        values.iter().map(|x| x.abs()).collect(),
        width,
    )
}

fn erf(x: f64) -> f64 {
    if x.abs() > 6.0 {
        return x.signum();
    }
    let mut term = x.abs();
    let mut sum = term;
    for n in 1..400 {
        term *= 2.0 * x * x / (2 * n + 1) as f64;
        sum += term;
        if term <= sum * 1e-17 {
            break;
        }
    }
    x.signum() * (2.0 / std::f64::consts::PI.sqrt()) * (-x * x).exp() * sum
}

// Complementary error function for x >= 1, via the continued fraction for
// Q(1/2, x*x). It never subtracts the negative tail from one.
fn erfc_tail(x: f64) -> f64 {
    let z = x * x;
    let mut b = z + 0.5;
    let mut c = 1e300;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..400 {
        let a = -(i as f64) * (i as f64 - 0.5);
        b += 2.0;
        d = 1.0 / (a * d + b);
        c = b + a / c;
        let delta = c * d;
        h *= delta;
        if (delta - 1.0).abs() < 2e-16 {
            break;
        }
    }
    (-z).exp() * x / std::f64::consts::PI.sqrt() * h
}

pub fn gelu(x: f64, mode: GeluMode) -> f64 {
    match mode {
        GeluMode::Erf if x < -std::f64::consts::SQRT_2 => {
            0.5 * x * erfc_tail(-x / std::f64::consts::SQRT_2)
        }
        GeluMode::Erf => 0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2)),
        GeluMode::Tanh => {
            0.5 * x
                * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        }
    }
}

/// Bias precedes either activation. SwiGLU's output has N/2 columns, so its
/// residual also has N/2 columns.
pub fn gemm_bias_epilogue(
    a: &[f64],
    b: &[f64],
    shape: GemmShape,
    bias: Option<&[f64]>,
    gelu_mode: Option<GeluMode>,
    swiglu: Option<SwigluColumns>,
    residual: Option<&[f64]>,
) -> Result<OracleOutput, Refusal> {
    let mut base = gemm(
        a,
        b,
        shape,
        &[],
        GemmEpilogue {
            residual: None,
            swiglu: None,
        },
    )?;
    if bias.is_some_and(|x| x.len() != shape.n)
        || gelu_mode.is_some() && swiglu.is_some()
        || swiglu.is_some() && !shape.n.is_multiple_of(2)
        || residual.is_some_and(|x| {
            x.len()
                != shape.m
                    * if swiglu.is_some() {
                        shape.n / 2
                    } else {
                        shape.n
                    }
        })
    {
        return Err(Refusal("GEMM epilogue operand shape mismatch"));
    }
    for (index, value) in base.values.iter_mut().enumerate() {
        *value += bias.map_or(0.0, |x| x[index % shape.n]);
        if let Some(mode) = gelu_mode {
            *value = gelu(*value, mode);
        }
        base.sum_abs_products[index] += bias.map_or(0.0, |x| x[index % shape.n].abs());
    }
    if let Some(layout) = swiglu {
        let width = shape.n / 2;
        let mut values = Vec::with_capacity(shape.m * width);
        let mut bounds = Vec::with_capacity(shape.m * width);
        for row in 0..shape.m {
            for col in 0..width {
                let (gate_col, up_col) = match layout {
                    SwigluColumns::ConcatenatedHalves => (col, col + width),
                    SwigluColumns::Interleaved => (2 * col, 2 * col + 1),
                };
                let gate = base.values[row * shape.n + gate_col];
                let up = base.values[row * shape.n + up_col];
                let silu = gate / (1.0 + (-gate).exp());
                let index = row * width + col;
                values.push(silu * up + residual.map_or(0.0, |x| x[index]));
                bounds.push(
                    base.sum_abs_products[row * shape.n + gate_col] * up.abs()
                        + base.sum_abs_products[row * shape.n + up_col] * silu.abs()
                        + residual.map_or(0.0, |x| x[index].abs()),
                );
            }
        }
        return output(values, bounds, shape.k);
    }
    for (index, value) in base.values.iter_mut().enumerate() {
        *value += residual.map_or(0.0, |x| x[index]);
        base.sum_abs_products[index] += residual.map_or(0.0, |x| x[index].abs());
    }
    Ok(base)
}

/// Cos and sin are [tokens,rotary_dim/2], allowing arbitrary 1D or 2D
/// positions. Values beyond rotary_dim pass through unchanged.
pub fn scratch_rope(
    input: &[f64],
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    style: RotaryStyle,
    cos: &[f64],
    sin: &[f64],
) -> Result<OracleOutput, Refusal> {
    if heads == 0
        || head_dim == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || input.is_empty()
        || !input.len().is_multiple_of(heads * head_dim)
        || cos.len() != input.len() / (heads * head_dim) * rotary_dim / 2
        || sin.len() != cos.len()
    {
        return Err(Refusal("scratch RoPE shape mismatch"));
    }
    let tokens = input.len() / (heads * head_dim);
    let pairs = rotary_dim / 2;
    let values = rotate_token_rows(
        input,
        heads,
        head_dim,
        rotary_dim,
        style,
        cos,
        sin,
        RotaryRounding {
            arithmetic: RotaryArithmetic::F64CosFirst,
            ..RotaryRounding::F64
        },
    )?;
    let mut bounds = input.iter().map(|x| x.abs()).collect::<Vec<_>>();
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for pair in 0..pairs {
                let (a, b) = match style {
                    RotaryStyle::Neox => (pair, pair + pairs),
                    RotaryStyle::GptJ => (2 * pair, 2 * pair + 1),
                };
                let c = cos[token * pairs + pair];
                let s = sin[token * pairs + pair];
                let bound = input[base + a].abs() * c.abs() + input[base + b].abs() * s.abs();
                bounds[base + a] = bound;
                bounds[base + b] = bound;
            }
        }
    }
    output(values, bounds, 2)
}

/// One captured `(cos, sin)` table row per token, applied to every head of
/// [tokens, heads, head_dim].
#[allow(clippy::too_many_arguments)]
fn rotate_token_rows(
    input: &[f64],
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    pairing: RotaryStyle,
    cos: &[f64],
    sin: &[f64],
    rounding: RotaryRounding,
) -> Result<Vec<f64>, Refusal> {
    let table = cos
        .iter()
        .copied()
        .zip(sin.iter().copied())
        .collect::<Vec<_>>();
    let positions = (0..input.len() / head_dim)
        .map(|row| row / heads)
        .collect::<Vec<_>>();
    let rotation = Rotation {
        slice: RotarySlice {
            offset: 0,
            len: rotary_dim,
        },
        pairing,
        inverse: false,
        rounding,
    };
    rotation.apply(
        input,
        head_dim,
        RotaryPositions::Linear(&positions),
        &FreqTable::Captured(&table),
    )
}

/// `output[row] = input[indices[row]]`, requiring a full permutation.
pub fn permute_rows(input: &[f64], width: usize, indices: &[u32]) -> Result<OracleOutput, Refusal> {
    if width == 0 || input.len() != indices.len() * width || indices.is_empty() {
        return Err(Refusal("permutation shape mismatch"));
    }
    let mut seen = vec![false; indices.len()];
    let mut values = Vec::with_capacity(input.len());
    for &index in indices {
        let index = index as usize;
        if index >= indices.len() || seen[index] {
            return Err(Refusal("indices are not a permutation"));
        }
        seen[index] = true;
        values.extend_from_slice(&input[index * width..(index + 1) * width]);
    }
    output(values.clone(), values.iter().map(|x| x.abs()).collect(), 1)
}

/// Input is `[time,channels]`, with zero padding and output rows at stride
/// positions. The K order is `[kernel,channel]`. F32 inputs round once to BF16.
pub fn im2col1d(
    input: &[f64],
    channels: usize,
    kernel: usize,
    stride: usize,
    pad_left: usize,
    pad_right: usize,
    dtype: DType,
) -> Result<OracleOutput, Refusal> {
    if !matches!(dtype, DType::F32 | DType::Bf16) {
        return Err(Refusal("im2col shape or dtype mismatch"));
    }
    let values = unfold1d(input, channels, kernel, stride, pad_left, pad_right)?
        .into_iter()
        .map(|x| f64::from(bf16_to_f32(f32_to_bf16(x as f32))))
        .collect::<Vec<_>>();
    output(values.clone(), values.iter().map(|x| x.abs()).collect(), 1)
}

/// `im2col1d` without rounding: the rows of F32 frames as they are.
pub fn im2col1d_f32(
    input: &[f64],
    channels: usize,
    kernel: usize,
    stride: usize,
    pad_left: usize,
    pad_right: usize,
) -> Result<OracleOutput, Refusal> {
    let values = unfold1d(input, channels, kernel, stride, pad_left, pad_right)?;
    output(values.clone(), values.iter().map(|x| x.abs()).collect(), 1)
}

/// The taps of every output row of a 1D convolution, [kernel,channel] each.
fn unfold1d(
    input: &[f64],
    channels: usize,
    kernel: usize,
    stride: usize,
    pad_left: usize,
    pad_right: usize,
) -> Result<Vec<f64>, Refusal> {
    if channels == 0
        || kernel == 0
        || stride == 0
        || input.is_empty()
        || !input.len().is_multiple_of(channels)
    {
        return Err(Refusal("im2col shape or dtype mismatch"));
    }
    let time = input.len() / channels;
    let padded = time
        .checked_add(pad_left)
        .and_then(|x| x.checked_add(pad_right))
        .ok_or(Refusal("im2col padded time overflows"))?;
    if padded < kernel {
        return Err(Refusal("im2col kernel exceeds padded time"));
    }
    let rows = (padded - kernel) / stride + 1;
    let mut values = Vec::with_capacity(rows * kernel * channels);
    for row in 0..rows {
        for tap in 0..kernel {
            let position = row * stride + tap;
            for channel in 0..channels {
                let x = if position < pad_left || position >= pad_left + time {
                    0.0
                } else {
                    input[(position - pad_left) * channels + channel]
                };
                values.push(x);
            }
        }
    }
    Ok(values)
}

#[derive(Clone, Debug, PartialEq)]
pub struct RvqOutput {
    pub codes: Vec<u32>,
    pub residual: Vec<f64>,
}

/// Codebooks are stage-major, each `[codes,dim]`. Ties choose the lower code.
pub fn rvq_nearest(
    input: &[f64],
    rows: usize,
    dim: usize,
    codebooks: &[&[f64]],
    sizes: &[usize],
) -> Result<RvqOutput, Refusal> {
    if rows == 0
        || dim == 0
        || input.len() != rows * dim
        || codebooks.is_empty()
        || codebooks.len() != sizes.len()
        || codebooks
            .iter()
            .zip(sizes)
            .any(|(table, &size)| size == 0 || table.len() != size * dim)
    {
        return Err(Refusal("RVQ input or codebook shape mismatch"));
    }
    let mut residual = input.to_vec();
    let mut codes = vec![0; rows * sizes.len()];
    for (stage, (table, &size)) in codebooks.iter().zip(sizes).enumerate() {
        for row in 0..rows {
            let mut best = (f64::INFINITY, 0);
            for code in 0..size {
                let distance = (0..dim)
                    .map(|col| (residual[row * dim + col] - table[code * dim + col]).powi(2))
                    .sum::<f64>();
                if distance < best.0 {
                    best = (distance, code);
                }
            }
            codes[row * sizes.len() + stage] = best.1 as u32;
            for col in 0..dim {
                residual[row * dim + col] -= table[best.1 * dim + col];
            }
        }
    }
    Ok(RvqOutput { codes, residual })
}

/// Each table is `[vocab,width]`; indices are `[rows,tables]`. The sum is in
/// table order, then rounded once by the consumer to BF16.
pub fn multi_embedding_sum(
    tables: &[&[f64]],
    indices: &[u32],
    vocab: usize,
    width: usize,
) -> Result<OracleOutput, Refusal> {
    if tables.is_empty()
        || vocab == 0
        || width == 0
        || indices.is_empty()
        || !indices.len().is_multiple_of(tables.len())
        || tables.iter().any(|t| t.len() != vocab * width)
        || indices.iter().any(|&i| i as usize >= vocab)
    {
        return Err(Refusal("embedding sum shape or index mismatch"));
    }
    let rows = indices.len() / tables.len();
    let mut values = vec![0.0; rows * width];
    let mut bounds = vec![0.0; values.len()];
    for row in 0..rows {
        for (table_index, table) in tables.iter().enumerate() {
            let start = indices[row * tables.len() + table_index] as usize * width;
            for col in 0..width {
                let x = table[start + col];
                values[row * width + col] += x;
                bounds[row * width + col] += x.abs();
            }
        }
    }
    output(values, bounds, tables.len())
}

/// A source of -1 preserves the gathered token embedding; nonnegative values
/// address flat arena rows. Injection is in-place in the device contract.
pub fn embedding_override(
    activation: &[f64],
    arena: &[f64],
    width: usize,
    source: &[i64],
) -> Result<OracleOutput, Refusal> {
    if width == 0
        || activation.len() != source.len() * width
        || !arena.len().is_multiple_of(width)
        || source.is_empty()
    {
        return Err(Refusal("embedding override shape mismatch"));
    }
    let mut values = activation.to_vec();
    for (row, &index) in source.iter().enumerate() {
        if index == -1 {
            continue;
        }
        if index < 0 || index as usize >= arena.len() / width {
            return Err(Refusal("embedding source outside arena"));
        }
        values[row * width..(row + 1) * width]
            .copy_from_slice(&arena[index as usize * width..(index as usize + 1) * width]);
    }
    output(values.clone(), values.iter().map(|x| x.abs()).collect(), 1)
}

/// Rounds to the nearest BF16 value, ties to even, directly from f64 (no
/// intermediate F32 rounding). Finite inputs within BF16 range only.
fn round_bf16(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let unit = bf16_unit(x);
    (x / unit).round_ties_even() * unit
}

/// The BF16 unit in the last place for the binade holding `x`.
fn bf16_unit(x: f64) -> f64 {
    2_f64.powi((x.abs().log2().floor() as i32).max(-126) - 7)
}

/// F32 interpolated position add: for each row `r` and column `c`,
/// `input[r][c] + sum_j weights[r][j] * table[indices[r][j]][c]` over the
/// four bilinear taps, HF's F32 weighted sum and add.
pub fn interpolated_position_add_f32(
    input: &[f64],
    table: &[f64],
    width: usize,
    indices: &[i32],
    weights: &[f64],
) -> Result<OracleOutput, Refusal> {
    if width == 0
        || input.is_empty()
        || !input.len().is_multiple_of(width)
        || table.is_empty()
        || !table.len().is_multiple_of(width)
        || indices.len() != input.len() / width * 4
        || weights.len() != indices.len()
        || weights.iter().any(|w| !w.is_finite())
    {
        return Err(Refusal("interpolated position operands mismatch"));
    }
    let table_rows = table.len() / width;
    if indices
        .iter()
        .any(|&index| index < 0 || index as usize >= table_rows)
    {
        return Err(Refusal("interpolated position index outside the table"));
    }
    let mut values = Vec::with_capacity(input.len());
    let mut bounds = Vec::with_capacity(input.len());
    for (row, hidden) in input.chunks_exact(width).enumerate() {
        let taps = &indices[row * 4..row * 4 + 4];
        let tap_weights = &weights[row * 4..row * 4 + 4];
        for (col, &x) in hidden.iter().enumerate() {
            let (mut sum, mut abs_sum) = (0.0, 0.0);
            for (&index, &weight) in taps.iter().zip(tap_weights) {
                let product = weight * table[index as usize * width + col];
                sum += product;
                abs_sum += product.abs();
            }
            values.push(x + sum);
            bounds.push(x.abs() + abs_sum);
        }
    }
    output(values, bounds, 5)
}

/// The value of the BF16 pair of an F32 value `x`: `hi = bf16(x)` and
/// `lo = bf16(x - hi)`, both rounded to nearest even. `x - hi` is exact in
/// F32, so the pair's value is within 2^-16 |x| of `x`.
pub fn bf16_pair(x: f64) -> f64 {
    let hi = round_bf16(x);
    hi + round_bf16(x - hi)
}

/// `RopeQkvPairs` before the split: packed F32 rows `[T, (q_heads + 2 *
/// kv_heads) * head_dim]` give the rotated q, the rotated k and v. Each output
/// is exact; a kernel's BF16 pairs differ by the F32 rotation's rounding and
/// at most 2^-16 of the value for the split.
pub fn rope_qkv_pairs(
    query: MultimodalQuery,
    rows: &[f64],
    cos: &[f64],
    sin: &[f64],
) -> Result<[OracleOutput; 3], Refusal> {
    query.validate()?;
    let MultimodalQuery::RopeQkvPairs {
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        style,
        capacity,
    } = query
    else {
        return Err(Refusal("expected BF16 pair QKV RoPE query"));
    };
    let (q_width, kv_width) = (
        (q_heads * head_dim) as usize,
        (kv_heads * head_dim) as usize,
    );
    let width = q_width + 2 * kv_width;
    if rows.is_empty()
        || !rows.len().is_multiple_of(width)
        || rows.len() / width > capacity as usize
    {
        return Err(Refusal("BF16 pair QKV rows mismatch"));
    }
    let part = |start: usize, len: usize| {
        rows.chunks_exact(width)
            .flat_map(|row| &row[start..start + len])
            .copied()
            .collect::<Vec<_>>()
    };
    let rotate = |x: &[f64], heads: u32| {
        scratch_rope(
            x,
            heads as usize,
            head_dim as usize,
            rotary_dim as usize,
            style,
            cos,
            sin,
        )
    };
    let v = part(q_width + kv_width, kv_width);
    Ok([
        rotate(&part(0, q_width), q_heads)?,
        rotate(&part(q_width, kv_width), kv_heads)?,
        output(v.clone(), v.iter().map(|x| x.abs()).collect(), 1)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{NumericalContract, WeightFormat};

    #[test]
    fn attention_respects_sequences_causality_windows_and_sink() {
        let query = MultimodalQuery::ContiguousAttention {
            q_heads: 1,
            kv_heads: 1,
            head_dim: 64,
            causal: false,
            window_left: None,
            window_right: None,
            key0_sink: true,
            capacity: 4,
        };
        let q = vec![0.0; 3 * 64];
        let k = q.clone();
        let mut v = vec![0.0; 3 * 64];
        v[0] = 2.0;
        v[64] = 4.0;
        v[128] = 100.0;
        let rows = [0, 1, 2];
        let out = contiguous_attention_rows(
            query,
            &rows,
            &q,
            &k,
            &v,
            &[0, 2, 3],
            1.0,
            Some(&[2_f64.ln()]),
        )
        .unwrap();
        assert!((out.values[0] - 8.0 / 3.0).abs() < 1e-12);
        assert_eq!(out.values[128], 100.0);
        let causal = MultimodalQuery::ContiguousAttention {
            q_heads: 1,
            kv_heads: 1,
            head_dim: 64,
            causal: true,
            window_left: Some(1),
            window_right: None,
            key0_sink: false,
            capacity: 4,
        };
        let out =
            contiguous_attention_rows(causal, &rows, &q, &k, &v, &[0, 2, 3], 1.0, None).unwrap();
        assert_eq!(out.values[0], 2.0);
        assert_eq!(out.values[64], 3.0);
    }

    #[test]
    fn norm_bias_gelu_and_scratch_rope_have_fixed_order() {
        let norm = layernorm_affine(&[1.0, 3.0], 2, &[2.0, 3.0], &[0.5, -0.5], 1.0).unwrap();
        let inv = 2_f64.sqrt().recip();
        assert!((norm.values[0] - (0.5 - 2.0 * inv)).abs() < 1e-12);
        assert!((norm.values[1] - (3.0 * inv - 0.5)).abs() < 1e-12);
        assert!((gelu(1.0, GeluMode::Erf) - 0.8413447460685429).abs() < 1e-14);
        assert!((gelu(1.0, GeluMode::Tanh) - 0.8411919906082768).abs() < 1e-12);
        let shape = GemmShape {
            m: 1,
            n: 1,
            k: 2,
            experts: 1,
        };
        let out = gemm_bias_epilogue(
            &[1.0, 2.0],
            &[3.0, 4.0],
            shape,
            Some(&[1.0]),
            Some(GeluMode::Erf),
            None,
            Some(&[2.0]),
        )
        .unwrap();
        assert!((out.values[0] - (gelu(12.0, GeluMode::Erf) + 2.0)).abs() < 1e-12);
        let swiglu = gemm_bias_epilogue(
            &[1.0],
            &[2.0, 3.0],
            GemmShape {
                m: 1,
                n: 2,
                k: 1,
                experts: 1,
            },
            Some(&[1.0, 2.0]),
            None,
            Some(SwigluColumns::ConcatenatedHalves),
            None,
        )
        .unwrap();
        assert!((swiglu.values[0] - (3.0 / (1.0 + (-3.0_f64).exp()) * 5.0)).abs() < 1e-12);
        let rope = scratch_rope(
            &[1.0, 2.0, 3.0, 4.0],
            1,
            4,
            4,
            RotaryStyle::Neox,
            &[0.0, 0.0],
            &[1.0, 1.0],
        )
        .unwrap();
        assert_eq!(rope.values, [-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn permute_im2col_rvq_embedding_and_override() {
        assert_eq!(
            permute_rows(&[1.0, 2.0, 3.0], 1, &[2, 0, 1])
                .unwrap()
                .values,
            [3.0, 1.0, 2.0]
        );
        assert!(permute_rows(&[1.0, 2.0], 1, &[0, 0]).is_err());
        assert_eq!(
            im2col1d(&[1.0, 2.0, 3.0], 1, 2, 2, 0, 1, DType::F32)
                .unwrap()
                .values,
            [1.0, 2.0, 3.0, 0.0]
        );
        let rvq = rvq_nearest(&[3.0], 1, 1, &[&[0.0, 2.0], &[0.0, 1.0]], &[2, 2]).unwrap();
        assert_eq!(rvq.codes, [1, 1]);
        assert_eq!(rvq.residual, [0.0]);
        let tie = rvq_nearest(&[1.0], 1, 1, &[&[0.0, 2.0]], &[2]).unwrap();
        assert_eq!(tie.codes, [0]);
        assert_eq!(
            multi_embedding_sum(&[&[1.0, 2.0], &[10.0, 20.0]], &[0, 1], 2, 1)
                .unwrap()
                .values,
            [21.0]
        );
        assert_eq!(
            embedding_override(&[1.0, 2.0], &[5.0, 6.0], 1, &[-1, 1])
                .unwrap()
                .values,
            [1.0, 6.0]
        );
        assert!(embedding_override(&[1.0], &[5.0], 1, &[2]).is_err());
    }

    #[test]
    fn semantic_ids_distinguish_bias_and_gelu_and_refuse_bad_shapes() {
        let query = crate::contract::GemmQuery {
            m_capacity: 8192,
            n: 1024,
            k: 1024,
            weight: WeightFormat::Bf16,
            activation: crate::contract::ActivationFormat::Bf16,
            output: DType::Bf16,
            epilogues: crate::contract::Epilogues {
                bias: true,
                gelu: Some(GeluMode::Erf),
                ..Default::default()
            },
            grouped: None,
        };
        assert_eq!(
            query.numerical_contract_id().unwrap().as_str(),
            "gemm.w-bf16.a-bf16.acc-f32.bias.gelu-erf.v1"
        );
        assert_ne!(
            query.numerical_contract_id().unwrap(),
            crate::contract::GemmQuery {
                epilogues: Default::default(),
                ..query
            }
            .numerical_contract_id()
            .unwrap()
        );
        assert!(
            MultimodalQuery::LayerNorm {
                width: 0,
                eps_bits: 1e-5_f32.to_bits(),
                capacity: 8192,
            }
            .validate()
            .is_err()
        );
        // BF16 pair GEMMs name their F32 activations, residual and output.
        let pairs = |output, residual_add| {
            MultimodalQuery::GemmPairs(crate::contract::GemmQuery {
                output,
                epilogues: crate::contract::Epilogues {
                    residual_add,
                    ..query.epilogues
                },
                ..query
            })
        };
        assert_eq!(
            pairs(DType::F32, true)
                .numerical_contract_id()
                .unwrap()
                .as_str(),
            "gemm.w-bf16.a-f32-bf16-pairs.acc-f32.bias.gelu-erf.residual.out-f32.v1"
        );
        assert_eq!(
            pairs(DType::Bf16, false)
                .numerical_contract_id()
                .unwrap()
                .as_str(),
            "gemm.w-bf16.a-f32-bf16-pairs.acc-f32.bias.gelu-erf.out-bf16.v1"
        );
        assert!(pairs(DType::Bf16, true).validate().is_err());
        let swiglu = MultimodalQuery::GemmPairs(crate::contract::GemmQuery {
            epilogues: crate::contract::Epilogues {
                bias: true,
                swiglu: Some(SwigluColumns::ConcatenatedHalves),
                ..Default::default()
            },
            ..query
        });
        assert!(swiglu.validate().is_err());
    }
}

/// BF16 operands are decoded before this exact f64 sum; rounding is the output
/// format's half-ULP allowance in the common numerical gate.
pub fn residual_add(input: &[f64], residual: &[f64]) -> Result<OracleOutput, Refusal> {
    if input.is_empty() || input.len() != residual.len() {
        return Err(Refusal("residual operands differ"));
    }
    output(
        input.iter().zip(residual).map(|(a, b)| a + b).collect(),
        input
            .iter()
            .zip(residual)
            .map(|(a, b)| a.abs() + b.abs())
            .collect(),
        2,
    )
}

/// SwiGLU of rows `[gate | up]` of `2 * hidden` values: `silu(gate) * up`.
/// S is the output's magnitude and K is 4, the F32 operations of the row
/// pass (the exponential, the add, the division and the product).
pub fn swiglu_rows(input: &[f64], hidden: usize) -> Result<OracleOutput, Refusal> {
    if hidden == 0 || input.is_empty() || !input.len().is_multiple_of(2 * hidden) {
        return Err(Refusal("SwiGLU rows shape mismatch"));
    }
    let values = input
        .chunks_exact(2 * hidden)
        .flat_map(|row| {
            let (gate, up) = row.split_at(hidden);
            gate.iter()
                .zip(up)
                .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
        })
        .collect::<Vec<_>>();
    output(values.clone(), values.iter().map(|x| x.abs()).collect(), 4)
}

pub fn gelu_rows(input: &[f64]) -> Result<OracleOutput, Refusal> {
    if input.is_empty() {
        return Err(Refusal("empty GELU input"));
    }
    let values = input
        .iter()
        .map(|&x| gelu(x, GeluMode::Erf))
        .collect::<Vec<_>>();
    output(values.clone(), values.iter().map(|x| x.abs()).collect(), 1)
}

#[cfg(test)]
mod transform_tests {
    use super::*;

    #[test]
    fn gelu_negative_tail_reference_avoids_cancellation() {
        let expected = [
            -0.00012668496733247986,
            -1.4332578593959731e-06,
            -5.919525870226207e-09,
            -8.958687807200846e-12,
            -4.9767684594174555e-15,
        ];
        for (x, expected) in [-4.0, -5.0, -6.0, -7.0, -8.0].into_iter().zip(expected) {
            let actual = gelu(x, GeluMode::Erf);
            assert!(
                (actual - expected).abs() <= expected.abs() * 1e-13,
                "GELU({x})={actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn residual_rounding_is_deferred_to_the_output_contract() {
        let result = residual_add(&[1.0, -1.0], &[0.00390625, 1.0]).unwrap();
        assert_eq!(result.values, [1.00390625, 0.0]);
        assert!((gelu_rows(&[1.0]).unwrap().values[0] - 0.8413447460685429).abs() < 1e-14);
    }
}

#[cfg(test)]
mod position_tests {
    use super::*;
    use crate::contract::{MultimodalQuery, NumericalContract};

    #[test]
    fn f32_position_sum_is_added_unrounded() {
        // Two rows of width 2, table of three rows.
        let table = [1.0, 2.0, 0.5, -4.0, 0.0, 1.0 + 2_f64.powi(-7)];
        let result = interpolated_position_add_f32(
            &[1.0, -1.0, 0.25, 0.0],
            &table,
            2,
            &[0, 1, 2, 0, 2, 2, 2, 2],
            &[0.25, 0.25, 0.25, 0.25, 1.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        // Row 0: p = (1 + 0.5 + 0 + 1) / 4 and (2 - 4 + 1 + 2^-7 + 2) / 4,
        // added unrounded.
        assert_eq!(result.values[0], 1.625);
        assert_eq!(result.values[1], -1.0 + (1.0 + 2_f64.powi(-7)) / 4.0);
        assert_eq!(
            result.sum_abs_products[1],
            1.0 + (9.0 + 2_f64.powi(-7)) / 4.0
        );
        // Row 1: one full-weight tap copies table row 2 exactly.
        assert_eq!(result.values[2], 0.25);
        assert_eq!(result.values[3], 1.0 + 2_f64.powi(-7));
        assert_eq!(result.reduction_lengths, [5; 4]);
    }

    #[test]
    fn pair_terms_fit_the_bound() {
        // The probability a pair attention row leaves out, 2^(-8 terms) of
        // each, stays within the contract's 2^-22 K for every key count.
        for keys in (1..=70_000).chain([1 << 20]) {
            let left_out = 2_f64.powi(-8 * pair_probability_terms(keys));
            assert!(left_out <= 2_f64.powi(-22) * keys as f64, "{keys} keys");
        }
        // The dropped q lo k lo product is at most 2^-16 of |q k|: every
        // split of [1, 2) on a 2^-20 grid keeps a low half within 2^-8.
        let low = |x: f64| x - round_bf16(x);
        for i in 0..1 << 20 {
            let x = 1.0 + f64::from(i) * 2_f64.powi(-20);
            assert!(low(x).abs() <= 2_f64.powi(-8) * x, "{x}");
        }
        // At a rounding midpoint, 1 + 2^-8 splits into 1 and 2^-8, and the
        // product reaches 1/66049, above 2^-18.
        let x = 1.0 + 2_f64.powi(-8);
        assert_eq!((round_bf16(x), low(x)), (1.0, 2_f64.powi(-8)));
        let dropped = low(x) * low(x) / (x * x);
        assert_eq!(dropped, 1.0 / 66049.0);
        assert!(dropped > 2_f64.powi(-18) && dropped <= 2_f64.powi(-16));
        // 2^-16 fits the logit term's 2^-22 dim at the served head dims
        // (64 and 72), not below 64.
        for dim in [64_f64, 72.0] {
            assert!(2_f64.powi(-16) <= 2_f64.powi(-22) * dim);
        }
        assert!(2_f64.powi(-16) > 2_f64.powi(-22) * 32.0);
    }

    #[test]
    fn bf16_pairs_hold_sixteen_bits() {
        assert_eq!(bf16_pair(1.0), 1.0);
        // 1 + 2^-8 + 2^-16 splits into 1 and 2^-8 + 2^-16 (BF16 exact).
        let x = 1.0 + 2_f64.powi(-8) + 2_f64.powi(-16);
        assert_eq!(bf16_pair(x), x);
        // Past 16 bits the low half rounds.
        let x = f64::from(1.1_f32);
        assert_ne!(bf16_pair(x), x);
        assert!((bf16_pair(x) - x).abs() <= 2_f64.powi(-16) * x);
        assert_eq!(bf16_pair(-x), -bf16_pair(x));
    }

    #[test]
    fn rope_qkv_pairs_rotate_q_and_k_and_pass_v() {
        let query = MultimodalQuery::RopeQkvPairs {
            q_heads: 2,
            kv_heads: 1,
            head_dim: 4,
            rotary_dim: 4,
            style: RotaryStyle::Neox,
            capacity: 2,
        };
        let rows = (0..32).map(f64::from).collect::<Vec<_>>();
        let (cos, sin) = ([0.6, 1.0, 0.0, 0.5], [0.8, 0.0, 1.0, -0.5]);
        let [q, k, v] = rope_qkv_pairs(query, &rows, &cos, &sin).unwrap();
        assert_eq!(q.values.len(), 16);
        assert_eq!(k.values.len(), 8);
        // Token 0, q head 0: pairs (0, 2) and (1, 3).
        assert_eq!(q.values[0], 0.0 * 0.6 - 2.0 * 0.8);
        assert_eq!(q.values[2], 2.0 * 0.6 + 0.0 * 0.8);
        assert_eq!(q.values[1], 1.0);
        // Token 1, k: row 1 columns 8..12 under the second token's angles.
        assert_eq!(k.values[4], 24.0 * 0.0 - 26.0 * 1.0);
        assert_eq!(v.values, [12.0, 13.0, 14.0, 15.0, 28.0, 29.0, 30.0, 31.0]);
        assert!(rope_qkv_pairs(query, &rows[..31], &cos, &sin).is_err());
        assert_eq!(
            query.numerical_contract_id().unwrap().to_string(),
            "row.rope-qkv-neox-f32-to-bf16-pairs.v1"
        );
    }

    #[test]
    fn position_add_refuses_bad_operands_and_has_an_identity() {
        let table = [1.0, 2.0];
        assert!(
            interpolated_position_add_f32(&[0.0], &table, 1, &[0, 1, 2, 0], &[0.25; 4]).is_err()
        );
        assert!(
            interpolated_position_add_f32(&[0.0], &table, 1, &[0, 1, -1, 0], &[0.25; 4]).is_err()
        );
        assert!(interpolated_position_add_f32(&[0.0], &table, 1, &[0; 3], &[0.25; 3]).is_err());
        assert!(
            interpolated_position_add_f32(&[0.0], &table, 1, &[0; 4], &[f64::NAN, 0.0, 0.0, 0.0])
                .is_err()
        );
        let query = MultimodalQuery::InterpolatedPositionAddF32 {
            width: 1152,
            table_rows: 2304,
            capacity: 65536,
        };
        assert_eq!(
            query.numerical_contract_id().unwrap().to_string(),
            "row.interpolated-position-add-f32.v1"
        );
        for bad in [
            MultimodalQuery::InterpolatedPositionAddF32 {
                width: 0,
                table_rows: 2304,
                capacity: 1,
            },
            MultimodalQuery::InterpolatedPositionAddF32 {
                width: 1152,
                table_rows: 0,
                capacity: 1,
            },
            MultimodalQuery::InterpolatedPositionAddF32 {
                width: 1152,
                table_rows: 2304,
                capacity: 0,
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn sampled_attention_rows_match_the_full_oracle() {
        let query = MultimodalQuery::ContiguousAttention {
            q_heads: 2,
            kv_heads: 1,
            head_dim: 72,
            causal: false,
            window_left: None,
            window_right: None,
            key0_sink: false,
            capacity: 65536,
        };
        let tokens = 7;
        let value = |i: usize| ((i * 37 % 29) as f64 - 14.0) / 16.0;
        let q = (0..tokens * 2 * 72).map(value).collect::<Vec<_>>();
        let k = (0..tokens * 72).map(|i| value(i + 5)).collect::<Vec<_>>();
        let v = (0..tokens * 72).map(|i| value(i + 11)).collect::<Vec<_>>();
        let cu = [0, 3, 7];
        let all = (0..tokens).collect::<Vec<_>>();
        let full = contiguous_attention_rows(query, &all, &q, &k, &v, &cu, 72_f64.powf(-0.5), None)
            .unwrap();
        let rows = [6, 0, 3];
        let q_rows = rows
            .iter()
            .flat_map(|&row| q[row * 144..(row + 1) * 144].iter().copied())
            .collect::<Vec<_>>();
        let sampled =
            contiguous_attention_rows(query, &rows, &q_rows, &k, &v, &cu, 72_f64.powf(-0.5), None)
                .unwrap();
        for (slot, &row) in rows.iter().enumerate() {
            assert_eq!(
                sampled.values[slot * 144..(slot + 1) * 144],
                full.values[row * 144..(row + 1) * 144]
            );
            assert_eq!(
                sampled.reduction_lengths[slot * 144],
                if row < 3 { 3 } else { 4 }
            );
        }
        assert!(contiguous_attention_rows(query, &[7], &q[..144], &k, &v, &cu, 1.0, None).is_err());
        // The pair contract keeps the values and adds the logits' term.
        let MultimodalQuery::ContiguousAttention {
            q_heads,
            kv_heads,
            head_dim,
            capacity,
            ..
        } = query
        else {
            unreachable!()
        };
        let pairs = MultimodalQuery::ContiguousAttentionPairs {
            q_heads,
            kv_heads,
            head_dim,
            causal: false,
            window_left: None,
            window_right: None,
            key0_sink: false,
            capacity,
        };
        let widened =
            contiguous_attention_rows(pairs, &all, &q, &k, &v, &cu, 72_f64.powf(-0.5), None)
                .unwrap();
        assert_eq!(widened.values, full.values);
        assert_eq!(widened.reduction_lengths, full.reduction_lengths);
        assert!(
            widened
                .sum_abs_products
                .iter()
                .zip(&full.sum_abs_products)
                .all(|(wide, plain)| wide > plain)
        );
    }
}
