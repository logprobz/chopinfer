//! Position indexer scores over decoded operands and exact selection over
//! the kernel's own F32 scores. Quantization errors and BF16 selection
//! agreement are separate measurements, not score or selection tolerances.

use crate::contract::{Refusal, Validate};
use crate::oracle::{OracleOutput, compensated_sum};
use crate::pooled_indexer::{
    CandidateBlocksQuery, IndexerCandidates, IndexerGranularity, PooledIndexerQuery,
    PositionIndexerQuery,
};
use crate::pooled_indexer_oracle::score_rank;

/// The trained MXFP4 quantizer. Its caller rounds to BF16 before amax;
/// this is also the boundary after tail rotation. The shared quantizer
/// owns code generation, not the indexer score oracle.
pub fn quantizer() -> crate::quantize::Quantizer {
    use crate::quantize::{AmaxRatio, QuantFormat, Quantizer, ScaleFormat, ScaleRule};
    Quantizer {
        format: QuantFormat {
            element: crate::codec::SmallFloat::E2M1,
            block: (1, 32),
            scale: ScaleFormat::Ue8M0,
            chunk_rows: None,
        },
        rule: ScaleRule::CeilPowerOfTwo(AmaxRatio::MultiplyReciprocal),
        amax_floor: Some(6.0 * 2.0_f32.powi(-126)),
    }
}

/// Key RMSNorm before its BF16 boundary, over decoded BF16 projections.
pub fn key_norm(
    query: PositionIndexerQuery,
    keys: &[f64],
    weight: &[f64],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    crate::oracle::rmsnorm(
        keys,
        query.head_dim as usize,
        weight,
        f64::from(f32::from_bits(query.key_norm_eps_bits)),
        crate::contract::RmsNormKind::Plain,
        None,
    )
}

/// Tail-only interleaved RoPE of one decoded BF16 vector using the F32
/// table row at the query token or compressed key's first token. The f64
/// output precedes BF16 storage and trained quantization.
pub fn rotate_tail(
    query: PositionIndexerQuery,
    vector: &[f64],
    cos_sin: &[[f32; 2]],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    if vector.len() != query.head_dim as usize || cos_sin.len() != query.tail_rope_dims as usize / 2
    {
        return Err(Refusal("position indexer RoPE operand shape mismatch"));
    }
    let mut values = vector.to_vec();
    let mut magnitudes = vector.iter().map(|v| v.abs()).collect::<Vec<_>>();
    let start = vector.len() - query.tail_rope_dims as usize;
    for (pair, &[cos, sin]) in cos_sin.iter().enumerate() {
        let i = start + 2 * pair;
        let (a, b, c, s) = (vector[i], vector[i + 1], f64::from(cos), f64::from(sin));
        values[i] = a * c - b * s;
        values[i + 1] = a * s + b * c;
        magnitudes[i] = (a * c).abs() + (b * s).abs();
        magnitudes[i + 1] = (a * s).abs() + (b * c).abs();
    }
    OracleOutput::new(values, magnitudes, 2)
}

fn shape(query: PooledIndexerQuery) -> Result<(usize, usize), Refusal> {
    query.validate()?;
    match query {
        PooledIndexerQuery::Select {
            heads,
            head_dim,
            granularity: IndexerGranularity::Position,
            ..
        } => Ok((heads as usize, head_dim as usize)),
        _ => Err(Refusal("expected a position indexer selection")),
    }
}

/// The single F32 scale applied to a projected head weight. It is computed
/// from the whole head count, never a rank-local count.
pub fn weight_scale(query: PooledIndexerQuery) -> Result<f32, Refusal> {
    let (heads, width) = shape(query)?;
    Ok(((width as f64).sqrt().recip() * (heads as f64).sqrt().recip()) as f32)
}

/// F64 scores on exact decoded q `[heads][head_dim]`, BF16 projected
/// weights `[heads]` and decoded keys `[positions][head_dim]`. The combined
/// scale is the declared F32 constant. K is width + heads + 1, and S
/// includes absolute dot products even when cancellation or ReLU hides
/// them. The score gate is `2^-22 * K * S`, with no output-rounding term.
pub fn scores(
    query: PooledIndexerQuery,
    q: &[f64],
    weights: &[f64],
    keys: &[f64],
) -> Result<OracleOutput, Refusal> {
    let (heads, width) = shape(query)?;
    if q.len() != heads * width
        || weights.len() != heads
        || !keys.len().is_multiple_of(width)
        || q.iter().chain(weights).chain(keys).any(|x| !x.is_finite())
    {
        return Err(Refusal(
            "position indexer score operand shape or value mismatch",
        ));
    }
    let scale = f64::from(weight_scale(query)?);
    let mut values = Vec::with_capacity(keys.len() / width);
    let mut magnitudes = Vec::with_capacity(values.capacity());
    for key in keys.chunks_exact(width) {
        let mut terms = Vec::with_capacity(heads);
        let mut absolute = Vec::with_capacity(heads);
        for (head, weight) in q.chunks_exact(width).zip(weights) {
            let dot = compensated_sum(head.iter().zip(key).map(|(a, b)| a * b));
            let magnitude = compensated_sum(head.iter().zip(key).map(|(a, b)| (a * b).abs()));
            terms.push(dot.max(0.0) * (weight * scale));
            absolute.push(magnitude * (weight * scale).abs());
        }
        values.push(compensated_sum(terms));
        magnitudes.push(compensated_sum(absolute));
    }
    OracleOutput::new(values, magnitudes, width + heads + 1)
}

/// One explicit F32 implementation of the score contract, useful for CPU
/// controls. Each 128-channel chain starts from zero and is folded in
/// order; the head sum uses FMA. GPU selection tests must supply the GPU's
/// own scores, not this emulation or narrowed f64 scores.
pub fn scores_f32(
    query: PooledIndexerQuery,
    q: &[f32],
    weights: &[f32],
    keys: &[f32],
) -> Result<Vec<f32>, Refusal> {
    let (heads, width) = shape(query)?;
    if q.len() != heads * width
        || weights.len() != heads
        || !keys.len().is_multiple_of(width)
        || q.iter().chain(weights).chain(keys).any(|x| !x.is_finite())
    {
        return Err(Refusal(
            "position indexer score operand shape or value mismatch",
        ));
    }
    let scale = weight_scale(query)?;
    let mut scores = Vec::with_capacity(keys.len() / width);
    for key in keys.chunks_exact(width) {
        let mut score = 0.0_f32;
        for (head, weight) in q.chunks_exact(width).zip(weights) {
            let mut dot = 0.0_f32;
            for (q, k) in head.chunks(128).zip(key.chunks(128)) {
                let chain = q
                    .iter()
                    .zip(k)
                    .fold(0.0_f32, |sum, (&q, &k)| q.mul_add(k, sum));
                dot += chain;
            }
            let rectified = if dot > 0.0 { dot } else { 0.0 };
            score = rectified.mul_add(weight * scale, score);
        }
        if !score.is_finite() {
            return Err(Refusal("non-finite position indexer score"));
        }
        scores.push(score);
    }
    Ok(scores)
}

/// Exact second-level selection. The input width may include future
/// positions; those never participate, even if their scores are larger.
/// A false mask bit removes a candidate instead of turning it into an
/// ordinary -inf-scored candidate. A visible admitted -inf is selectable;
/// NaNs rank below -inf as in pooled selection. Output is sorted by index.
pub fn select(
    query: PositionIndexerQuery,
    top_k: u32,
    position: u64,
    scores: &[f32],
    mask: Option<&[bool]>,
) -> Result<Vec<i32>, Refusal> {
    query.validate()?;
    let visible = query
        .visible_positions(position)
        .and_then(|count| usize::try_from(count).ok())
        .ok_or(Refusal("position indexer visible count overflow"))?;
    if top_k == 0
        || top_k > query.max_top_k
        || visible > scores.len()
        || scores.len() > i32::MAX as usize
    {
        return Err(Refusal("position indexer score width or top-k mismatch"));
    }
    match (query.candidates, mask) {
        (IndexerCandidates::All, None) => {}
        (IndexerCandidates::Mask, Some(mask)) if mask.len() == scores.len() => {}
        _ => return Err(Refusal("position indexer candidate mask mismatch")),
    }
    let mut kept = (0..visible)
        .filter(|&j| mask.is_none_or(|mask| mask[j]))
        .collect::<Vec<_>>();
    kept.sort_unstable_by(|&a, &b| {
        score_rank(scores[b])
            .cmp(&score_rank(scores[a]))
            .then(a.cmp(&b))
    });
    kept.truncate(top_k as usize);
    kept.sort_unstable();
    let mut output = kept.into_iter().map(|j| j as i32).collect::<Vec<_>>();
    output.resize(query.max_top_k as usize, -1);
    Ok(output)
}

/// Exact block max and top-k over the provided F32 scores. Visibility is
/// applied here even when the score producer already masked future rows.
/// Pinned membership expands to the whole block, clipped to score width.
pub fn select_candidate_blocks(
    query: CandidateBlocksQuery,
    scores: &[f32],
    visible: usize,
) -> Result<Vec<bool>, Refusal> {
    query.validate()?;
    if visible > scores.len()
        || scores[..visible]
            .iter()
            .any(|s| s.is_nan() || *s == f32::INFINITY)
    {
        return Err(Refusal("candidate block scores or visible count invalid"));
    }
    let mut mask = vec![false; scores.len()];
    if visible == 0 {
        return Ok(mask);
    }
    let block = query.block_size as usize;
    let mut maxima = scores[..visible]
        .chunks(block)
        .map(|chunk| chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .collect::<Vec<_>>();
    maxima[(visible - 1) / block] = f32::INFINITY;
    let mut kept = (0..maxima.len()).collect::<Vec<_>>();
    kept.sort_unstable_by(|&a, &b| {
        score_rank(maxima[b])
            .cmp(&score_rank(maxima[a]))
            .then(a.cmp(&b))
    });
    kept.truncate(query.top_k_blocks as usize);
    for index in kept {
        if maxima[index] > f32::NEG_INFINITY {
            let start = index * block;
            let end = start.saturating_add(block).min(scores.len());
            mask[start..end].fill(true);
        }
    }
    Ok(mask)
}
