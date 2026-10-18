//! F64 reference arithmetic for the pooled top-k indexer (`pooled_indexer`)
//! and the exact merge of per-rank selections
//! (`context_parallel::ContextParallelQuery::TopKMerge`). One call covers
//! one sequence or one row. Key norms, pooled keys and scores are f64 values
//! for the numerical gate; selections take F32 scores (a kernel's own) and
//! are exact.
//!
//! Tail state (`gated_pooling::GatedGroupPooling::tail_slot`). Only tokens
//! of a sequence's incomplete group enter it; the token that completes a
//! group is pooled with the live slots and never stored, which leaves the
//! tail empty. Dead slots keep whatever they held and are never compared.

use crate::codec::{bf16_to_f32, f32_to_bf16};
use crate::context_parallel::ContextParallelQuery;
use crate::contract::{RecurrentPhase, Refusal, Validate};
use crate::gated_pooling_oracle;
use crate::oracle::{OracleOutput, bf16_round, compensated_sum};
use crate::pooled_indexer::{IndexerSelectionMode, PooledIndexerQuery};

/// LayerNorm of raw key rows `[rows, width]` with weight and bias `[width]`,
/// in f64 before the single BF16 rounding. S of element d is `|w_d| * rstd *
/// (|x_d - mean| + mean(|x|)) + |b_d|` and K is the width.
pub fn key_norm(
    query: PooledIndexerQuery,
    raw: &[f64],
    weight: &[f64],
    bias: &[f64],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let PooledIndexerQuery::KeyWrite {
        pooling,
        key_norm_eps_bits,
        ..
    } = query
    else {
        return Err(Refusal("expected an indexer key write"));
    };
    let width = pooling.width as usize;
    if raw.is_empty()
        || !raw.len().is_multiple_of(width)
        || weight.len() != width
        || bias.len() != width
    {
        return Err(Refusal("indexer key norm operand shape mismatch"));
    }
    let eps = f64::from(f32::from_bits(key_norm_eps_bits));
    let mut values = Vec::with_capacity(raw.len());
    let mut s = Vec::with_capacity(raw.len());
    for x in raw.chunks_exact(width) {
        let mean = compensated_sum(x.iter().copied()) / width as f64;
        let variance = compensated_sum(x.iter().map(|v| (v - mean) * (v - mean))) / width as f64;
        let rstd = (variance + eps).sqrt().recip();
        let magnitude = compensated_sum(x.iter().map(|v| v.abs())) / width as f64;
        for ((&x, &w), &b) in x.iter().zip(weight).zip(bias) {
            values.push((x - mean) * rstd * w + b);
            s.push(w.abs() * rstd * ((x - mean).abs() + magnitude) + b.abs());
        }
    }
    OracleOutput::new(values, s, width)
}

/// A sequence's tail state: `[group - 1][width]` normalized keys and gate
/// logits, as decoded BF16 values.
#[derive(Clone, Debug, PartialEq)]
pub struct TailState {
    pub keys: Vec<f64>,
    pub gates: Vec<f64>,
}

impl TailState {
    /// An empty tail state of `query`'s pooling (zeros).
    pub fn empty(query: PooledIndexerQuery) -> Result<TailState, Refusal> {
        let PooledIndexerQuery::KeyWrite { pooling, .. } = query else {
            return Err(Refusal("expected an indexer key write"));
        };
        let len = (pooling.tail_state_elements() / 2) as usize;
        Ok(TailState {
            keys: vec![0.0; len],
            gates: vec![0.0; len],
        })
    }
}

/// The operands of one sequence's key write: `rows` raw keys and gate
/// logits `[rows, width]` (decoded BF16), the LayerNorm weight and bias
/// `[width]`, and the position embedding `[group][width]`.
#[derive(Clone, Copy, Debug)]
pub struct KeyWriteOperands<'a> {
    pub raw: &'a [f64],
    pub gates: &'a [f64],
    pub norm_weight: &'a [f64],
    pub norm_bias: &'a [f64],
    pub position: &'a [f64],
}

/// One pooled key written by a key write.
#[derive(Clone, Debug, PartialEq)]
pub struct PooledKey {
    pub pool: u64,
    /// Before the single BF16 rounding; K is the group.
    pub key: OracleOutput,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KeyWrite {
    /// The rows' normalized keys before their BF16 rounding.
    pub normalized: OracleOutput,
    /// The groups the rows complete, in row order.
    pub pools: Vec<PooledKey>,
    /// The tail state after each row: what a verify launch writes per row;
    /// prefill and decode keep the last.
    pub checkpoints: Vec<TailState>,
}

/// The key write of one sequence's rows at positions `start..start + rows`
/// after the committed `tail` (whose first `start % group` rows hold the
/// incomplete group). A completed group pools its tokens' BF16 normalized
/// keys, the oracle's f64 norm rounded once.
pub fn key_write(
    query: PooledIndexerQuery,
    start: u64,
    tail: &TailState,
    operands: KeyWriteOperands<'_>,
) -> Result<KeyWrite, Refusal> {
    let normalized = key_norm(
        query,
        operands.raw,
        operands.norm_weight,
        operands.norm_bias,
    )?;
    let PooledIndexerQuery::KeyWrite {
        pooling,
        phase,
        capacity,
        ..
    } = query
    else {
        unreachable!("key_norm checked the query kind");
    };
    let (group, width) = (pooling.group as usize, pooling.width as usize);
    let rows = operands.raw.len() / width;
    let max_rows = match phase {
        RecurrentPhase::Prefill => capacity as usize,
        RecurrentPhase::Decode => 1,
        RecurrentPhase::Verify { max_rows } => max_rows as usize,
    };
    if rows > max_rows
        || operands.gates.len() != rows * width
        || operands.position.len() != group * width
        || tail.keys.len() != (group - 1) * width
        || tail.gates.len() != (group - 1) * width
    {
        return Err(Refusal("indexer key write operand shape mismatch"));
    }
    let mut state = tail.clone();
    let mut pools = Vec::new();
    let mut checkpoints = Vec::with_capacity(rows);
    for row in 0..rows {
        let position = start + row as u64;
        let key = (0..width)
            .map(|d| bf16_round(normalized.values[row * width + d]))
            .collect::<Vec<_>>();
        let gate = &operands.gates[row * width..][..width];
        match pooling.tail_slot(position) {
            Some(slot) => {
                let (key_row, gate_row) = pooling
                    .tail_slot_offsets(slot)
                    .expect("a tail slot lies inside the banks");
                let bank = (group - 1) * width;
                state.keys[key_row as usize..][..width].copy_from_slice(&key);
                state.gates[gate_row as usize - bank..][..width].copy_from_slice(gate);
            }
            None => {
                let keys = [&state.keys[..], &key[..]].concat();
                let gates = [&state.gates[..], gate].concat();
                let pooled = gated_pooling_oracle::pool(pooling, &keys, &gates, operands.position)?;
                pools.push(PooledKey {
                    pool: position / group as u64,
                    key: pooled,
                });
            }
        }
        checkpoints.push(state.clone());
    }
    Ok(KeyWrite {
        normalized,
        pools,
        checkpoints,
    })
}

/// The geometry a selection reads from its query.
struct Selection {
    group: u32,
    max_top_k: u32,
    always_tail: bool,
}

fn selection(query: PooledIndexerQuery) -> Result<(Selection, IndexerSelectionMode), Refusal> {
    query.validate()?;
    match query {
        PooledIndexerQuery::Select {
            group,
            max_top_k,
            always_tail,
            mode,
            ..
        }
        | PooledIndexerQuery::HeadSumSelect {
            group,
            max_top_k,
            always_tail,
            mode,
            ..
        } => Ok((
            Selection {
                group,
                max_top_k,
                always_tail,
            },
            mode,
        )),
        _ => Err(Refusal("expected an indexer selection")),
    }
}

impl Selection {
    /// Pools kept per row for the runtime `top_k`.
    fn kept(&self, top_k: u32) -> Result<usize, Refusal> {
        if top_k == 0 || top_k > self.max_top_k || !top_k.is_multiple_of(self.group) {
            return Err(Refusal(
                "runtime indexer top-k is outside the compiled maximum",
            ));
        }
        Ok((top_k / self.group) as usize)
    }

    /// Complete pools visible to a row at `position`.
    fn visible(&self, position: u64) -> usize {
        ((position + 1) / u64::from(self.group)) as usize
    }

    /// The key list of a row at `position` keeping `kept` pools: the pools
    /// ascending, each as its `group` positions, then the always-selected
    /// tail, then -1.
    fn key_list(&self, position: u64, kept: &[usize]) -> Vec<i32> {
        let group = u64::from(self.group);
        let width = (self.max_top_k + if self.always_tail { self.group - 1 } else { 0 }) as usize;
        let mut pools = kept.to_vec();
        pools.sort_unstable();
        let mut list = pools
            .iter()
            .flat_map(|&pool| (0..group).map(move |t| (pool as u64 * group + t) as i32))
            .collect::<Vec<_>>();
        if self.always_tail {
            let base = (position + 1) / group * group;
            list.extend((base..=position).map(|p| p as i32));
        }
        list.resize(width, -1);
        list
    }
}

/// F64 scores of one row's pools: `q` is `[heads, head_dim]`, `weights`
/// `[heads]` (the head weights, decoded BF16) and `pooled_keys` `[pools,
/// head_dim]` (decoded BF16). S of pool g is `heads^-0.5 * sum_h |w_h| *
/// head_dim^-0.5 * sum_d |q_hd K_gd|`; K is `head_dim + heads`, the dot
/// chain then the head sum.
pub fn pool_scores(
    query: PooledIndexerQuery,
    q: &[f64],
    weights: &[f64],
    pooled_keys: &[f64],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let PooledIndexerQuery::Select {
        heads,
        head_dim,
        granularity,
        ..
    } = query
    else {
        return Err(Refusal("expected an indexer selection"));
    };
    if granularity == crate::pooled_indexer::IndexerGranularity::Position {
        return crate::position_indexer_oracle::scores(query, q, weights, pooled_keys);
    }
    let (h, d) = (heads as usize, head_dim as usize);
    if q.len() != h * d || weights.len() != h || !pooled_keys.len().is_multiple_of(d) {
        return Err(Refusal("indexer score operand shape mismatch"));
    }
    let (dot_scale, head_scale) = ((d as f64).sqrt().recip(), (h as f64).sqrt().recip());
    let mut values = Vec::with_capacity(pooled_keys.len() / d);
    let mut s = Vec::with_capacity(values.capacity());
    for key in pooled_keys.chunks_exact(d) {
        let heads = q.chunks_exact(d).zip(weights).map(|(head, &w)| {
            let dot = compensated_sum(head.iter().zip(key).map(|(a, b)| a * b));
            let magnitude = compensated_sum(head.iter().zip(key).map(|(a, b)| (a * b).abs()));
            let w = w * head_scale;
            (
                w * (dot * dot_scale).max(0.0),
                w.abs() * magnitude * dot_scale,
            )
        });
        let (score, bound): (Vec<f64>, Vec<f64>) = heads.unzip();
        values.push(compensated_sum(score));
        s.push(compensated_sum(bound));
    }
    if values.is_empty() {
        return Ok(OracleOutput {
            values,
            sum_abs_products: s,
            reduction_lengths: Vec::new(),
        });
    }
    OracleOutput::new(values, s, d + h)
}

/// The rank of an F32 score in the selection order: IEEE order with -0
/// equal to +0, and every NaN below -inf. A larger rank is better.
pub fn score_rank(score: f32) -> u32 {
    if score.is_nan() {
        return 0;
    }
    let bits = if score == 0.0 { 0 } else { score.to_bits() };
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// The score a per-rank pair carries: +0 for either zero and the canonical
/// NaN `0xffffffff` for any NaN, so one rank order maps back to one value.
pub fn canonical_score(score: f32) -> f32 {
    if score.is_nan() {
        f32::from_bits(u32::MAX)
    } else if score == 0.0 {
        0.0
    } else {
        score
    }
}

/// Orders (score, pool) candidates best first: score rank descending, then
/// the lower pool.
fn by_rank(a: (f32, usize), b: (f32, usize)) -> std::cmp::Ordering {
    score_rank(b.0).cmp(&score_rank(a.0)).then(a.1.cmp(&b.1))
}

/// The kept pools of one row, best first: the exact top `top_k / group` of
/// its candidates' F32 `scores` (one per visible pool) in the selection
/// order (`score_rank`), ties to the lower pool, or every candidate when
/// there are no more. A NaN score ranks last but stays a candidate. `owned`
/// restricts the candidates to a rank's pools.
fn kept_pools(k: usize, scores: &[f32], owned: Option<&[bool]>) -> Vec<usize> {
    let mut pools = (0..scores.len())
        .filter(|&pool| owned.is_none_or(|owned| owned[pool]))
        .collect::<Vec<_>>();
    pools.sort_by(|&a, &b| by_rank((scores[a], a), (scores[b], b)));
    pools.truncate(k);
    pools
}

/// The `Global` key list of a row at `position`, from the F32 scores of its
/// visible pools. Position queries are refused: use
/// [`crate::position_indexer_oracle::select`] for their visibility, masks
/// and non-finite rejection policy.
pub fn select(
    query: PooledIndexerQuery,
    top_k: u32,
    position: u64,
    scores: &[f32],
) -> Result<Vec<i32>, Refusal> {
    if matches!(
        query,
        PooledIndexerQuery::Select {
            granularity: crate::pooled_indexer::IndexerGranularity::Position,
            ..
        }
    ) {
        return Err(Refusal(
            "Position selection requires the position indexer oracle",
        ));
    }
    let (selection, mode) = selection(query)?;
    let k = selection.kept(top_k)?;
    if mode != IndexerSelectionMode::Global || scores.len() != selection.visible(position) {
        return Err(Refusal(
            "global indexer selection scores do not match the row",
        ));
    }
    Ok(selection.key_list(position, &kept_pools(k, scores, None)))
}

/// The `PerRank` output of a row at `position`: its top `top_k / group`
/// among the pools it `owns`, as (score, pool) pairs best first, padded to
/// `max_top_k / group` with `(-inf, -1)`.
pub fn select_per_rank(
    query: PooledIndexerQuery,
    top_k: u32,
    position: u64,
    scores: &[f32],
    owned: &[bool],
) -> Result<Vec<(f32, i32)>, Refusal> {
    let (selection, mode) = selection(query)?;
    let k = selection.kept(top_k)?;
    if !matches!(mode, IndexerSelectionMode::PerRank { .. })
        || scores.len() != selection.visible(position)
        || owned.len() != scores.len()
    {
        return Err(Refusal(
            "per-rank indexer selection scores do not match the row",
        ));
    }
    let mut pairs = kept_pools(k, scores, Some(owned))
        .into_iter()
        .map(|pool| (canonical_score(scores[pool]), pool as i32))
        .collect::<Vec<_>>();
    pairs.resize(
        (selection.max_top_k / selection.group) as usize,
        (f32::NEG_INFINITY, -1),
    );
    Ok(pairs)
}

/// The merged key list of a row at `position` from every rank's per-rank
/// pairs, in rank order.
pub fn merge(
    query: ContextParallelQuery,
    top_k: u32,
    position: u64,
    ranks: &[&[(f32, i32)]],
) -> Result<Vec<i32>, Refusal> {
    query.validate()?;
    let ContextParallelQuery::TopKMerge {
        ranks: rank_count,
        group,
        max_top_k,
        always_tail,
        ..
    } = query
    else {
        return Err(Refusal("expected a top-k merge"));
    };
    let selection = Selection {
        group,
        max_top_k,
        always_tail,
    };
    let k = selection.kept(top_k)?;
    let width = (max_top_k / group) as usize;
    if ranks.len() != rank_count as usize || ranks.iter().any(|list| list.len() != width) {
        return Err(Refusal("top-k merge operand shape mismatch"));
    }
    // The candidates' pools are distinct, so ordering by rank then pool is
    // the single-rank order.
    Ok(selection.key_list(position, &merged_pools(ranks, k)))
}

/// The kept pools of a `TopKMergeBlocks` row from every rank's per-rank
/// pairs, in rank order: ascending block ids `[max_top_k / group]`, then -1.
pub fn merge_blocks(
    query: ContextParallelQuery,
    top_k: u32,
    ranks: &[&[(f32, i32)]],
) -> Result<Vec<i32>, Refusal> {
    query.validate()?;
    let ContextParallelQuery::TopKMergeBlocks {
        ranks: rank_count,
        group,
        max_top_k,
        ..
    } = query
    else {
        return Err(Refusal("expected a top-k block merge"));
    };
    let selection = Selection {
        group,
        max_top_k,
        always_tail: false,
    };
    let k = selection.kept(top_k)?;
    let width = (max_top_k / group) as usize;
    if ranks.len() != rank_count as usize || ranks.iter().any(|list| list.len() != width) {
        return Err(Refusal("top-k merge operand shape mismatch"));
    }
    let mut kept = merged_pools(ranks, k);
    kept.sort_unstable();
    let mut blocks = kept.into_iter().map(|pool| pool as i32).collect::<Vec<_>>();
    blocks.resize(width, -1);
    Ok(blocks)
}

/// The first `k` of every rank's candidates in the selection order; padding
/// pairs (pool -1) are no candidates.
fn merged_pools(ranks: &[&[(f32, i32)]], k: usize) -> Vec<usize> {
    let mut candidates = ranks
        .iter()
        .flat_map(|list| list.iter())
        .filter(|&&(_, pool)| pool >= 0)
        .map(|&(score, pool)| (score, pool as usize))
        .collect::<Vec<_>>();
    candidates.sort_by(|&a, &b| by_rank(a, b));
    candidates
        .into_iter()
        .take(k)
        .map(|(_, pool)| pool)
        .collect()
}

/// `out[i] = lists[source[i]]` over key lists `[rows, width]`.
pub fn compact(
    query: PooledIndexerQuery,
    lists: &[i32],
    source: &[u32],
) -> Result<Vec<i32>, Refusal> {
    query.validate()?;
    let PooledIndexerQuery::CompactSelection { width, capacity } = query else {
        return Err(Refusal("expected a selection compaction"));
    };
    let width = width as usize;
    if !lists.len().is_multiple_of(width)
        || source.len() > capacity as usize
        || source
            .iter()
            .any(|&row| row as usize >= lists.len() / width)
    {
        return Err(Refusal("selection compaction operand shape mismatch"));
    }
    Ok(source
        .iter()
        .flat_map(|&row| lists[row as usize * width..][..width].iter().copied())
        .collect())
}

/// Decoded BF16 values of `bits`.
pub fn decode_bf16(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

/// BF16 storage of f64 `values`, rounded once through F32.
pub fn encode_bf16(values: &[f64]) -> Vec<u16> {
    values.iter().map(|&v| f32_to_bf16(v as f32)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::DType;
    use crate::gated_pooling::GatedGroupPooling;
    use crate::pooled_indexer::{IndexerGranularity, IndexerKeyFormat};

    fn select_query(mode: IndexerSelectionMode, max_top_k: u32) -> PooledIndexerQuery {
        PooledIndexerQuery::Select {
            heads: 2,
            head_dim: 64,
            group: 4,
            max_top_k,
            always_tail: true,
            granularity: IndexerGranularity::Pool,
            key_format: IndexerKeyFormat::Bf16,
            mode,
            page_size: 16,
            capacity: 16,
        }
    }

    fn write_query(phase: RecurrentPhase) -> PooledIndexerQuery {
        PooledIndexerQuery::KeyWrite {
            pooling: GatedGroupPooling {
                group: 4,
                width: 8,
                position_dtype: DType::Bf16,
            },
            key_norm_eps_bits: 1e-6_f32.to_bits(),
            page_size: 16,
            key_format: IndexerKeyFormat::Bf16,
            phase,
            capacity: 64,
        }
    }

    const GLOBAL: IndexerSelectionMode = IndexerSelectionMode::Global;

    #[test]
    fn key_norm_matches_hand_values() {
        // x = [1, 2, ..., 8]: mean 4.5, variance 5.25.
        let raw = (1..=8).map(f64::from).collect::<Vec<_>>();
        let weight = [2.0; 8];
        let bias = [0.5; 8];
        let out = key_norm(write_query(RecurrentPhase::Prefill), &raw, &weight, &bias).unwrap();
        let rstd = (5.25_f64 + f64::from(1e-6_f32)).sqrt().recip();
        for (d, &v) in out.values.iter().enumerate() {
            assert!((v - ((d as f64 + 1.0 - 4.5) * rstd * 2.0 + 0.5)).abs() < 1e-14);
        }
        assert_eq!(out.reduction_lengths, vec![8; 8]);
    }

    /// The token completing a group is pooled with the tail and never
    /// stored; the tail holds only the incomplete group's tokens.
    #[test]
    fn key_write_pools_completed_groups_and_keeps_the_incomplete_tail() {
        let query = write_query(RecurrentPhase::Prefill);
        let rows = 7;
        let raw = (0..rows * 8)
            .map(|i| f64::from(((i * 37) % 11) as u8) - 5.0)
            .collect::<Vec<_>>();
        let gates = (0..rows * 8)
            .map(|i| f64::from(((i * 13) % 7) as u8) * 0.25)
            .collect::<Vec<_>>();
        let operands = KeyWriteOperands {
            raw: &raw,
            gates: &gates,
            norm_weight: &[1.0; 8],
            norm_bias: &[0.0; 8],
            position: &[0.0; 32],
        };
        // Start at position 2 after two committed tail tokens.
        let mut tail = TailState::empty(query).unwrap();
        tail.keys[..16].copy_from_slice(&[0.5; 16]);
        tail.gates[..16].copy_from_slice(&[0.0; 16]);
        let out = key_write(query, 2, &tail, operands).unwrap();
        // Positions 2..=8 complete pool 0 (at 3) and pool 1 (at 7).
        assert_eq!(out.pools.iter().map(|p| p.pool).collect::<Vec<_>>(), [0, 1]);
        let key = |row: usize| {
            (0..8)
                .map(|d| bf16_round(out.normalized.values[row * 8 + d]))
                .collect::<Vec<_>>()
        };
        let expected = gated_pooling_oracle::pool(
            GatedGroupPooling {
                group: 4,
                width: 8,
                position_dtype: DType::Bf16,
            },
            &[vec![0.5; 16], key(0), key(1)].concat(),
            &[vec![0.0; 16], gates[..16].to_vec()].concat(),
            &[0.0; 32],
        )
        .unwrap();
        assert_eq!(out.pools[0].key, expected);
        // After position 6 the tail holds positions 4, 5 and 6; after
        // position 8 one token, position 8, in row 0.
        assert_eq!(out.checkpoints.len(), rows);
        assert_eq!(out.checkpoints[4].keys, [key(2), key(3), key(4)].concat());
        assert_eq!(out.checkpoints[4].gates, gates[16..40]);
        assert_eq!(out.checkpoints[6].keys[..8], key(6)[..]);
        assert_eq!(out.checkpoints[6].gates[..8], gates[48..56]);
        // A decode launch takes one row; verify takes up to its rows.
        assert!(key_write(write_query(RecurrentPhase::Decode), 2, &tail, operands).is_err());
        let verify = write_query(RecurrentPhase::Verify { max_rows: 7 });
        assert_eq!(key_write(verify, 2, &tail, operands).unwrap(), out);
    }

    #[test]
    fn scores_weight_relu_heads() {
        let query = select_query(GLOBAL, 16);
        let mut q = vec![0.0; 128];
        q[0] = 8.0; // head 0 . key = 8 * k[0]
        q[64] = -8.0; // head 1 . key = -8 * k[0]
        let key = [&[2.0][..], &[0.0; 63]].concat();
        let out = pool_scores(query, &q, &[1.0, 3.0], &key).unwrap();
        // relu(16 / 8) * 1 / sqrt(2) + relu(-2) * 3 / sqrt(2).
        assert!((out.values[0] - 2.0 / 2_f64.sqrt()).abs() < 1e-15);
        assert!((out.sum_abs_products[0] - (2.0 + 3.0 * 2.0) / 2_f64.sqrt()).abs() < 1e-15);
        assert_eq!(out.reduction_lengths, vec![66]);
    }

    #[test]
    fn ties_go_to_the_lower_pool() {
        let query = select_query(GLOBAL, 8);
        // Position 23 sees pools 0..6 and no tail; keep 2 of 6.
        let scores = [1.0, 3.0, 2.0, 3.0, 3.0, 0.0];
        let list = select(query, 8, 23, &scores).unwrap();
        assert_eq!(&list[..8], &[4, 5, 6, 7, 12, 13, 14, 15]);
        assert!(list[8..].iter().all(|&p| p == -1));
        assert_eq!(list.len(), 11);
        // -0 and +0 tie.
        let list = select(query, 4, 7, &[-0.0, 0.0]).unwrap();
        assert_eq!(&list[..4], &[0, 1, 2, 3]);
    }

    #[test]
    fn tail_is_always_selected_and_own_key_follows_the_rule() {
        let query = select_query(GLOBAL, 4);
        // Position 9 (offset 1 of pool 2) attends its own key in the tail.
        let list = select(query, 4, 9, &[5.0, 1.0]).unwrap();
        assert_eq!(&list[..7], &[0, 1, 2, 3, 8, 9, -1]);
        // Position 11 completes pool 2: no tail, and its own key appears
        // only if pool 2 wins.
        let lost = select(query, 4, 11, &[5.0, 1.0, 2.0]).unwrap();
        assert_eq!(&lost[..], &[0, 1, 2, 3, -1, -1, -1]);
        let won = select(query, 4, 11, &[5.0, 1.0, 9.0]).unwrap();
        assert_eq!(&won[..], &[8, 9, 10, 11, -1, -1, -1]);
        // Position 2: no complete pool, tail 0..=2.
        let early = select(query, 4, 2, &[]).unwrap();
        assert_eq!(&early[..], &[0, 1, 2, -1, -1, -1, -1]);
    }

    #[test]
    fn fewer_pools_than_k_selects_every_pool_ascending() {
        let query = select_query(GLOBAL, 16);
        let list = select(query, 16, 13, &[0.5, 7.0, -3.0]).unwrap();
        assert_eq!(
            &list[..],
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1, -1, -1, -1, -1
            ]
        );
        // The runtime top-k is a multiple of the group within the maximum.
        assert!(select(query, 18, 13, &[0.5, 7.0, -3.0]).is_err());
        assert!(select(query, 20, 13, &[0.5, 7.0, -3.0]).is_err());
        assert!(select(query, 0, 13, &[0.5, 7.0, -3.0]).is_err());
    }

    /// Deterministic scores with many ties, owned by page across ranks.
    fn tied_scores(pools: usize) -> Vec<f32> {
        (0..pools)
            .map(|g| f32::from(((g * 7919) % 13) as u8) * 0.5 - 2.0)
            .collect()
    }

    #[test]
    fn per_rank_lists_merge_into_the_global_selection() {
        for ranks in [2_u32, 4] {
            for (position, top_k) in [(1023_u64, 64_u32), (1021, 64), (40, 64), (777, 16)] {
                let visible = (position as usize + 1) / 4;
                let scores = tied_scores(visible);
                let global = select(select_query(GLOBAL, 64), top_k, position, &scores).unwrap();
                let per_rank = select_query(IndexerSelectionMode::PerRank { ranks }, 64);
                // A page of 16 tokens holds 4 pools; pages go round robin.
                let lists = (0..ranks as usize)
                    .map(|rank| {
                        let owned = (0..visible)
                            .map(|g| (g / 4) % ranks as usize == rank)
                            .collect::<Vec<_>>();
                        select_per_rank(per_rank, top_k, position, &scores, &owned).unwrap()
                    })
                    .collect::<Vec<_>>();
                for list in &lists {
                    assert_eq!(list.len(), 16);
                    assert!(list.windows(2).all(|w| w[0].1 < 0
                        || w[1].1 < 0
                        || w[0].0 > w[1].0
                        || (w[0].0 == w[1].0 && w[0].1 < w[1].1)));
                }
                let merge_query = ContextParallelQuery::TopKMerge {
                    ranks,
                    group: 4,
                    max_top_k: 64,
                    always_tail: true,
                    capacity: 16,
                };
                let refs = lists.iter().map(|l| &l[..]).collect::<Vec<_>>();
                let merged = merge(merge_query, top_k, position, &refs).unwrap();
                assert_eq!(merged, global, "ranks {ranks} position {position}");
            }
        }
    }

    /// The block-id merge keeps the key-list merge's pools, ascending, then
    /// -1.
    #[test]
    fn block_merges_keep_the_merged_pools() {
        for ranks in [2_u32, 4] {
            for (position, top_k) in [(1023_u64, 64_u32), (40, 64), (777, 16)] {
                let visible = (position as usize + 1) / 4;
                let scores = tied_scores(visible);
                let global = select(select_query(GLOBAL, 64), top_k, position, &scores).unwrap();
                let per_rank = select_query(IndexerSelectionMode::PerRank { ranks }, 64);
                let lists = (0..ranks as usize)
                    .map(|rank| {
                        let owned = (0..visible)
                            .map(|g| (g / 4) % ranks as usize == rank)
                            .collect::<Vec<_>>();
                        select_per_rank(per_rank, top_k, position, &scores, &owned).unwrap()
                    })
                    .collect::<Vec<_>>();
                let refs = lists.iter().map(|l| &l[..]).collect::<Vec<_>>();
                let blocks = merge_blocks(
                    ContextParallelQuery::TopKMergeBlocks {
                        ranks,
                        group: 4,
                        max_top_k: 64,
                        capacity: 16,
                    },
                    top_k,
                    &refs,
                )
                .unwrap();
                assert_eq!(blocks.len(), 16);
                let kept = blocks.iter().take_while(|&&b| b >= 0).count();
                assert!(blocks[kept..].iter().all(|&b| b == -1));
                let expected = global
                    .iter()
                    .filter(|&&t| t >= 0 && (t as usize) < visible * 4)
                    .step_by(4)
                    .map(|&t| t / 4)
                    .collect::<Vec<_>>();
                assert_eq!(
                    &blocks[..kept],
                    &expected[..],
                    "{ranks} ranks at {position}"
                );
            }
        }
    }

    /// One order for every score: +inf, finite, signed zeros as one value
    /// (lower pool first), -inf, then NaN; per-rank pairs carry +0 and the
    /// canonical NaN, and the merge orders them exactly as one rank does.
    #[test]
    fn special_scores_rank_alike_in_selection_and_merge() {
        let scores = [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -0.0, 0.0, 1.0];
        let order = [2, 5, 3, 4, 1, 0];
        let global = select_query(GLOBAL, 32);
        for kept in 1..=6 {
            let list = select(global, 4 * kept as u32, 23, &scores).unwrap();
            let mut expected = order[..kept].to_vec();
            expected.sort_unstable();
            let positions = expected
                .iter()
                .flat_map(|&pool| (0..4).map(move |t| pool * 4 + t))
                .collect::<Vec<_>>();
            assert_eq!(&list[..positions.len()], &positions[..]);
            assert!(list[positions.len()..].iter().all(|&p| p == -1));
        }
        let per_rank = select_query(IndexerSelectionMode::PerRank { ranks: 2 }, 32);
        let pairs = select_per_rank(per_rank, 24, 23, &scores, &[true; 6]).unwrap();
        let bits = pairs
            .iter()
            .map(|&(score, pool)| (score.to_bits(), pool))
            .collect::<Vec<_>>();
        assert_eq!(
            &bits[..7],
            &[
                (f32::INFINITY.to_bits(), 2),
                (1.0_f32.to_bits(), 5),
                (0, 3),
                (0, 4),
                (f32::NEG_INFINITY.to_bits(), 1),
                (u32::MAX, 0),
                (f32::NEG_INFINITY.to_bits(), -1),
            ]
        );
        // Pages of 4 pools: pools 0..4 on rank 0, 4..6 on rank 1.
        let merge_query = ContextParallelQuery::TopKMerge {
            ranks: 2,
            group: 4,
            max_top_k: 32,
            always_tail: true,
            capacity: 1,
        };
        for top_k in [4, 12, 20, 24] {
            let lists = (0..2)
                .map(|rank| {
                    let owned = (0..6).map(|g| g / 4 == rank).collect::<Vec<_>>();
                    select_per_rank(per_rank, top_k, 23, &scores, &owned).unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(
                merge(merge_query, top_k, 23, &[&lists[0], &lists[1]]).unwrap(),
                select(global, top_k, 23, &scores).unwrap()
            );
        }
        // A NaN pair is a candidate that ranks last, never a panic.
        let nan = [[(f32::NAN, 0)].as_slice(), &[(f32::NEG_INFINITY, -1); 7]].concat();
        let zero = [[(0.0, 1)].as_slice(), &[(f32::NEG_INFINITY, -1); 7]].concat();
        let list = merge(merge_query, 4, 7, &[&nan, &zero]).unwrap();
        assert_eq!(&list[..4], &[4, 5, 6, 7]);
        let list = merge(merge_query, 8, 7, &[&nan, &zero]).unwrap();
        assert_eq!(&list[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(score_rank(-0.0), score_rank(0.0));
        assert!(score_rank(f32::NEG_INFINITY) > score_rank(f32::NAN));
        assert!(score_rank(-f32::NAN) == 0 && score_rank(f32::MIN) > score_rank(f32::NEG_INFINITY));
    }

    #[test]
    fn compaction_gathers_source_rows() {
        let query = PooledIndexerQuery::CompactSelection {
            width: 3,
            capacity: 4,
        };
        let lists = [0, 1, 2, 10, 11, -1];
        assert_eq!(
            compact(query, &lists, &[1, 1, 0]).unwrap(),
            [10, 11, -1, 10, 11, -1, 0, 1, 2]
        );
        assert!(compact(query, &lists, &[2]).is_err());
    }

    #[test]
    fn compensated_sum_keeps_cancelled_terms() {
        assert_eq!(compensated_sum([1e16, 1.0, -1e16]), 1.0);
        assert_eq!(decode_bf16(&encode_bf16(&[1.5, -2.0])), [1.5, -2.0]);
    }
}
