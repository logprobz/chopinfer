//! F64 reference arithmetic for block-sparse attention with a learned indexer.
//! One call covers one sequence. A row at position p sees
//! tokens 0..=p: complete blocks of `block` tokens from position 0, then an
//! incomplete tail of fewer than `block` tokens.

use crate::attend::{Attend, AttendValues, KvRows, paged_rows, selected_blocks};
use crate::codec::{bf16_to_f32, f32_to_bf16};
use crate::contract::{Refusal, SparseAttentionQuery, Validate};
use crate::gated_attention_oracle::{gemma_norm_rotate, mrope_cos_sin};
use crate::oracle::OracleOutput;
use crate::select::{NanKeys, Ties, topk};

fn rope_tables(
    positions: &[[usize; 3]],
    rotary_dim: u32,
    theta_bits: u32,
    mrope: [u32; 3],
) -> Vec<Vec<(f64, f64)>> {
    let theta = f64::from(f32::from_bits(theta_bits));
    positions
        .iter()
        .map(|&p| mrope_cos_sin(p, rotary_dim as usize, theta, mrope))
        .collect()
}

/// q is [rows, heads, head_dim] from the indexer projection and weight is the
/// Gemma norm weight `[head_dim]`. `tables` overrides the computed cosines.
pub fn indexer_query(
    query: SparseAttentionQuery,
    q: &[f64],
    weight: &[f64],
    rope_positions: &[[usize; 3]],
    tables: Option<&[Vec<(f64, f64)>]>,
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let SparseAttentionQuery::IndexerQueryGemmaNormMrope {
        heads,
        head_dim,
        rotary_dim,
        theta_bits,
        mrope,
        eps_bits,
        capacity,
    } = query
    else {
        return Err(Refusal("expected indexer query"));
    };
    let rows = rope_positions.len();
    if rows == 0
        || rows > capacity as usize
        || q.len() != rows * heads as usize * head_dim as usize
        || weight.len() != head_dim as usize
        || tables.is_some_and(|t| t.len() != rows)
    {
        return Err(Refusal("indexer query operand shape mismatch"));
    }
    let tables = match tables {
        Some(tables) => tables.to_vec(),
        None => rope_tables(rope_positions, rotary_dim, theta_bits, mrope),
    };
    let eps = f64::from(f32::from_bits(eps_bits));
    gemma_norm_rotate(q, weight, eps, &tables, heads as usize, rotary_dim as usize)
}

/// One sequence's indexer key state.
pub struct IndexerKeyCache<'a> {
    /// Raw keys, [ring, head_dim].
    pub ring_keys: &'a mut [f64],
    /// RoPE positions (T, H, W) of each ring row.
    pub ring_positions: &'a mut [[usize; 3]],
    /// Block keys, [blocks, head_dim]; row b is valid once block b completes.
    pub block_keys: &'a mut [f64],
}

#[derive(Clone, Debug, PartialEq)]
pub struct IndexerBlockWrites {
    /// Blocks completed by this call, in row order.
    pub blocks: Vec<usize>,
    /// Their keys, [blocks, head_dim], as written to the block-key cache.
    pub keys: OracleOutput,
}

/// Raw keys are [rows, head_dim] at `token_positions`, written in row order.
pub fn indexer_key_write(
    query: SparseAttentionQuery,
    raw_keys: &[f64],
    token_positions: &[usize],
    rope_positions: &[[usize; 3]],
    weight: &[f64],
    cache: IndexerKeyCache<'_>,
) -> Result<IndexerBlockWrites, Refusal> {
    query.validate()?;
    let SparseAttentionQuery::IndexerKeyWrite {
        head_dim,
        rotary_dim,
        theta_bits,
        mrope,
        eps_bits,
        block,
        ring,
        capacity,
    } = query
    else {
        return Err(Refusal("expected indexer key write query"));
    };
    let (d, block, ring) = (head_dim as usize, block as usize, ring as usize);
    let rows = token_positions.len();
    if rows == 0
        || rows > capacity as usize
        || rope_positions.len() != rows
        || raw_keys.len() != rows * d
        || weight.len() != d
        || cache.ring_keys.len() != ring * d
        || cache.ring_positions.len() != ring
        || !cache.block_keys.len().is_multiple_of(d)
    {
        return Err(Refusal("indexer key write operand shape mismatch"));
    }
    let eps = f64::from(f32::from_bits(eps_bits));
    let mut blocks = Vec::new();
    let mut pooled = Vec::new();
    let mut first_positions = Vec::new();
    for (row, &position) in token_positions.iter().enumerate() {
        let slot = position % ring;
        cache.ring_keys[slot * d..(slot + 1) * d]
            .copy_from_slice(&raw_keys[row * d..(row + 1) * d]);
        cache.ring_positions[slot] = rope_positions[row];
        if position % block != block - 1 {
            continue;
        }
        let first = position + 1 - block;
        for col in 0..d {
            let sum = (first..=position)
                .map(|p| cache.ring_keys[(p % ring) * d + col])
                .sum::<f64>();
            pooled.push(f64::from(bf16_to_f32(f32_to_bf16(
                (sum / block as f64) as f32,
            ))));
        }
        blocks.push(position / block);
        first_positions.push(cache.ring_positions[first % ring]);
    }
    if blocks.is_empty() {
        return Ok(IndexerBlockWrites {
            blocks,
            keys: OracleOutput {
                values: Vec::new(),
                sum_abs_products: Vec::new(),
                reduction_lengths: Vec::new(),
            },
        });
    }
    let tables = rope_tables(&first_positions, rotary_dim, theta_bits, mrope);
    let keys = gemma_norm_rotate(&pooled, weight, eps, &tables, 1, rotary_dim as usize)?;
    for (index, &b) in blocks.iter().enumerate() {
        let destination = cache
            .block_keys
            .get_mut(b * d..(b + 1) * d)
            .ok_or(Refusal("block-key cache too short"))?;
        destination.copy_from_slice(&keys.values[index * d..(index + 1) * d]);
    }
    Ok(IndexerBlockWrites { blocks, keys })
}

/// Scores of one row: q is [heads, head_dim] and block_keys holds the row's
/// complete visible blocks, [blocks, head_dim].
pub fn block_scores(
    query: SparseAttentionQuery,
    q: &[f64],
    block_keys: &[f64],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let SparseAttentionQuery::BlockScores {
        heads,
        head_dim,
        max_blocks,
        ..
    } = query
    else {
        return Err(Refusal("expected block score query"));
    };
    let (h, d) = (heads as usize, head_dim as usize);
    if q.len() != h * d
        || !block_keys.len().is_multiple_of(d)
        || block_keys.len() / d > max_blocks as usize
    {
        return Err(Refusal("block score operand shape mismatch"));
    }
    let scale = (d as f64).sqrt().recip();
    let mut values = Vec::with_capacity(block_keys.len() / d);
    let mut bounds = Vec::with_capacity(values.capacity());
    for key in block_keys.chunks_exact(d) {
        let mut score = 0.0;
        let mut abs_sum = 0.0;
        for head in q.chunks_exact(d) {
            let dot = head.iter().zip(key).map(|(a, b)| a * b).sum::<f64>();
            score += dot.max(0.0);
            abs_sum += head
                .iter()
                .zip(key)
                .map(|(a, b)| (a * b).abs())
                .sum::<f64>();
        }
        values.push(score * scale);
        bounds.push(abs_sum * scale);
    }
    if values.is_empty() {
        return Ok(OracleOutput {
            values,
            sum_abs_products: bounds,
            reduction_lengths: Vec::new(),
        });
    }
    OracleOutput::new(values, bounds, h * d)
}

/// The key set of a row that sees `visible` tokens, given one score per
/// complete block. Exact: compare kernel output bit for bit.
pub fn block_topk(
    query: SparseAttentionQuery,
    scores: &[f64],
    visible: usize,
) -> Result<Vec<usize>, Refusal> {
    query.validate()?;
    let SparseAttentionQuery::BlockTopK {
        block,
        top_blocks,
        max_blocks,
        ..
    } = query
    else {
        return Err(Refusal("expected block top-k query"));
    };
    let (block, top) = (block as usize, top_blocks as usize);
    let complete = visible / block;
    if visible == 0
        || scores.len() != complete
        || complete > max_blocks as usize
        || scores.iter().any(|s| s.is_nan())
    {
        return Err(Refusal(
            "block top-k scores do not match the visible blocks",
        ));
    }
    let mut kept = topk(scores, top, Ties::LowerIndex, NanKeys::Refuse, None)?;
    kept.sort_unstable();
    Ok(selected_blocks(&kept, block, visible))
}

/// Paged K and V caches, each [page, page_size, kv_heads, head_dim].
#[derive(Clone, Copy, Debug)]
pub struct SparseKv<'a> {
    pub keys: &'a [f64],
    pub values: &'a [f64],
    pub block_table: &'a [usize],
}

/// q is [rows, q_heads, head_dim] and `key_sets` lists each row's tokens.
/// Query head h reads KV head h / (q_heads / kv_heads).
pub fn sparse_attention(
    query: SparseAttentionQuery,
    q: &[f64],
    key_sets: &[Vec<usize>],
    kv: SparseKv<'_>,
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let SparseAttentionQuery::SparseAttention {
        q_heads,
        kv_heads,
        head_dim,
        softmax_scale_bits,
        page_size,
        capacity,
        ..
    } = query
    else {
        return Err(Refusal("expected sparse attention query"));
    };
    let (qh, kh, d, page) = (
        q_heads as usize,
        kv_heads as usize,
        head_dim as usize,
        page_size as usize,
    );
    let limit = query.dense_tokens().unwrap_or(0) as usize;
    let page_len = page * kh * d;
    let rows = key_sets.len();
    if rows == 0
        || rows > capacity as usize
        || q.len() != rows * qh * d
        || key_sets
            .iter()
            .any(|set| set.is_empty() || set.len() > limit)
        || !kv.keys.len().is_multiple_of(page_len)
        || kv.values.len() != kv.keys.len()
    {
        return Err(Refusal("sparse attention operand shape mismatch"));
    }
    let pages = kv.keys.len() / page_len;
    let core = Attend::f64(f64::from(f32::from_bits(softmax_scale_bits)));
    let (keys, values) = ([kv.keys], [kv.values]);
    let rows_of = |segments| KvRows {
        segments,
        heads: kh,
        dim: d,
    };
    let mut output = OracleOutput {
        values: Vec::with_capacity(q.len()),
        sum_abs_products: Vec::with_capacity(q.len()),
        reduction_lengths: Vec::with_capacity(q.len()),
    };
    for (row, set) in key_sets.iter().enumerate() {
        let list = paged_rows(set.iter().copied(), kv.block_table, page, pages)
            .ok_or(Refusal("sparse attention key outside the page table"))?;
        core.row(
            &q[row * qh * d..(row + 1) * qh * d],
            rows_of(&keys),
            AttendValues::Rows(rows_of(&values)),
            &list,
            &mut output,
        )?;
    }
    Ok(output)
}
