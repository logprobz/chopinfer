//! F64 reference arithmetic for the gated attention rows:
//! Gemma per-head Q/K norm, interleaved M-RoPE with the paged KV write, and
//! the sigmoid output gate.

use crate::contract::{GatedAttentionQuery, Refusal, RmsNormKind, RotaryStyle, Validate};
use crate::oracle::{OracleOutput, paged_kv_write, rmsnorm, rope_shape};
use crate::recurrent_oracle::sigmoid;
use crate::rotary::{
    AngleRule, Axes, ComputedTable, FreqTable, RotaryArithmetic, RotaryPositions, RotaryRounding,
    RotarySlice, Rotation,
};

pub use crate::rotary::mrope_axis;

/// Cosine and sine for the gated/QSA contract: HF F32 inverse frequencies
/// and angle products, accurate trig, then BF16 cos/sin stored as F32.
/// These kernels read a supplied table at every rotary width. They do not
/// use the CUDA common helper's generic computed-angle path.
pub fn mrope_cos_sin(
    position: [usize; 3],
    rotary_dim: usize,
    theta: f64,
    sections: [u32; 3],
) -> Vec<(f64, f64)> {
    let table = ComputedTable {
        theta,
        rule: AngleRule::HfF32AngleBf16CosSin,
        axes: Axes::Mrope(sections),
    };
    (0..rotary_dim / 2)
        .map(|pair| table.entry(position, pair, rotary_dim / 2))
        .collect()
}

/// Gated/QSA rotation keeps F32 normalized operands: round the mate times
/// sine to F32, then F32 FMA with cosine, for both 16-pair and 32-pair NeoX
/// rotations. Output BF16/FP8 rounding belongs to the caller. Without a
/// captured table, compute the contract's HF table at theta 1e7.
pub fn rope_normalized_f32(
    input: &[f64],
    positions: &[usize],
    head_dim: usize,
    rotary_dim: usize,
    table: Option<&[(f64, f64)]>,
) -> Result<Vec<f64>, Refusal> {
    rope_shape(input, positions, head_dim, rotary_dim, 1e7)?;
    let computed = ComputedTable {
        theta: 1e7,
        rule: AngleRule::HfF32AngleBf16CosSin,
        axes: Axes::Linear,
    };
    let rotation = Rotation {
        slice: RotarySlice {
            offset: 0,
            len: rotary_dim,
        },
        pairing: RotaryStyle::Neox,
        inverse: false,
        rounding: RotaryRounding {
            table: None,
            arithmetic: RotaryArithmetic::F32Fma,
            output: None,
        },
    };
    rotation.apply(
        input,
        head_dim,
        RotaryPositions::Linear(positions),
        &table.map_or(FreqTable::Computed(computed), FreqTable::Captured),
    )
}

/// Apply NeoX RoPE with per-token tables to [rows, heads, head_dim] input.
/// S propagates as `S_x |cos| + S_y |sin|` for each rotated element.
fn rotate_rows(
    input: &OracleOutput,
    tables: &[Vec<(f64, f64)>],
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<OracleOutput, Refusal> {
    let rows = tables.len();
    let positions = (0..rows * heads).collect::<Vec<_>>();
    let flat = tables
        .iter()
        .flat_map(|table| std::iter::repeat_n(table, heads).flatten().copied())
        .collect::<Vec<_>>();
    let values = rope_normalized_f32(&input.values, &positions, head_dim, rotary_dim, Some(&flat))?;
    let absolute = flat
        .iter()
        .map(|&(cos, sin)| (cos.abs(), sin.abs()))
        .collect::<Vec<_>>();
    let mut bounds = input.sum_abs_products.clone();
    let half = rotary_dim / 2;
    for (index, head) in bounds.chunks_exact_mut(head_dim).enumerate() {
        let source = &input.sum_abs_products[index * head_dim..(index + 1) * head_dim];
        for pair in 0..half {
            let (cos, sin) = absolute[index * half + pair];
            head[pair] = source[pair] * cos + source[pair + half] * sin;
            head[pair + half] = source[pair + half] * cos + source[pair] * sin;
        }
    }
    Ok(OracleOutput {
        values,
        sum_abs_products: bounds,
        reduction_lengths: input.reduction_lengths.clone(),
    })
}

/// Per-head Gemma RMSNorm of [rows, heads, head_dim], then NeoX RoPE with
/// one table per row. The weight is [head_dim].
pub(crate) fn gemma_norm_rotate(
    input: &[f64],
    weight: &[f64],
    eps: f64,
    tables: &[Vec<(f64, f64)>],
    heads: usize,
    rotary_dim: usize,
) -> Result<OracleOutput, Refusal> {
    let head_dim = weight.len();
    let normed = rmsnorm(
        input,
        head_dim,
        weight,
        eps,
        RmsNormKind::GemmaOnePlusWeight,
        None,
    )?;
    rotate_rows(&normed, tables, heads, head_dim, rotary_dim)
}

#[derive(Clone, Debug, PartialEq)]
pub struct GatedAttentionRows {
    /// Normalized, rotated queries, [rows, q_heads, head_dim].
    pub q: OracleOutput,
    /// Gate logits copied from the Q rows, [rows, q_heads, head_dim].
    pub gate: Vec<f64>,
    /// Normalized, rotated keys as written to the cache, [rows, kv_heads, head_dim].
    pub k: OracleOutput,
}

/// Paged K and V caches, each [page, page_size, kv_heads, head_dim].
pub struct PagedKvPair<'a> {
    pub keys: &'a mut [f64],
    pub values: &'a mut [f64],
    pub block_table: &'a [usize],
}

/// Operands of one sequence chunk. `q_gate` is [rows, q_heads, 2 * head_dim],
/// k and v are `[rows, kv_heads, head_dim]`, the norm weights are `[head_dim]`
/// (Gemma, applied as 1 + w), and `rope_positions` hold (T, H, W) per row.
/// `token_positions` are the rows' sequence indexes, which place K and V in
/// the cache; for text they equal every RoPE axis. `tables` overrides the
/// computed cosines and sines, one table per row.
pub struct GatedAttentionInputs<'a> {
    pub q_gate: &'a [f64],
    pub k: &'a [f64],
    pub v: &'a [f64],
    pub q_norm: &'a [f64],
    pub k_norm: &'a [f64],
    pub rope_positions: &'a [[usize; 3]],
    pub token_positions: &'a [usize],
    pub tables: Option<&'a [Vec<(f64, f64)>]>,
}

pub fn qk_norm_rope_kv_write(
    query: GatedAttentionQuery,
    inputs: GatedAttentionInputs<'_>,
    cache: PagedKvPair<'_>,
) -> Result<GatedAttentionRows, Refusal> {
    query.validate()?;
    let GatedAttentionQuery::QkNormRopeKvWrite {
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        theta_bits,
        mrope,
        eps_bits,
        page_size,
        capacity,
        ..
    } = query
    else {
        return Err(Refusal("expected gated attention norm and RoPE query"));
    };
    let (qh, kh, d, rd) = (
        q_heads as usize,
        kv_heads as usize,
        head_dim as usize,
        rotary_dim as usize,
    );
    let rows = inputs.token_positions.len();
    if rows == 0
        || inputs.rope_positions.len() != rows
        || rows > capacity as usize
        || inputs.q_gate.len() != rows * qh * 2 * d
        || inputs.k.len() != rows * kh * d
        || inputs.v.len() != inputs.k.len()
        || inputs.q_norm.len() != d
        || inputs.k_norm.len() != d
        || inputs.tables.is_some_and(|t| t.len() != rows)
    {
        return Err(Refusal("gated attention row operand shape mismatch"));
    }
    let tables = match inputs.tables {
        Some(tables) => tables.to_vec(),
        None => inputs
            .rope_positions
            .iter()
            .map(|&p| mrope_cos_sin(p, rd, f64::from(f32::from_bits(theta_bits)), mrope))
            .collect(),
    };
    let mut q = Vec::with_capacity(rows * qh * d);
    let mut gate = Vec::with_capacity(rows * qh * d);
    for head in inputs.q_gate.chunks_exact(2 * d) {
        q.extend_from_slice(&head[..d]);
        gate.extend_from_slice(&head[d..]);
    }
    let eps = f64::from(f32::from_bits(eps_bits));
    let q = gemma_norm_rotate(&q, inputs.q_norm, eps, &tables, qh, rd)?;
    let k = gemma_norm_rotate(inputs.k, inputs.k_norm, eps, &tables, kh, rd)?;
    let page = page_size as usize;
    paged_kv_write(
        cache.keys,
        cache.block_table,
        inputs.token_positions,
        &k.values,
        page,
        kh,
        d,
    )?;
    paged_kv_write(
        cache.values,
        cache.block_table,
        inputs.token_positions,
        inputs.v,
        page,
        kh,
        d,
    )?;
    Ok(GatedAttentionRows { q, gate, k })
}

/// `attention * sigmoid(gate)`; both are [rows, heads, head_dim].
pub fn attention_output_gate(
    query: GatedAttentionQuery,
    attention: &[f64],
    gate: &[f64],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let GatedAttentionQuery::OutputGate {
        heads,
        head_dim,
        capacity,
    } = query
    else {
        return Err(Refusal("expected attention output gate query"));
    };
    let width = heads as usize * head_dim as usize;
    if attention.is_empty()
        || !attention.len().is_multiple_of(width)
        || attention.len() / width > capacity as usize
        || gate.len() != attention.len()
    {
        return Err(Refusal("attention output gate shape mismatch"));
    }
    let values = attention
        .iter()
        .zip(gate)
        .map(|(&x, &z)| x * sigmoid(z))
        .collect::<Vec<_>>();
    let bounds = values.iter().map(|v| v.abs()).collect();
    OracleOutput::new(values, bounds, 1)
}
