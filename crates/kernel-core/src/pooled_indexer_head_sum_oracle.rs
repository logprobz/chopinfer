//! Reference arithmetic for the head-sum indexer selection
//! (`pooled_indexer::PooledIndexerQuery::HeadSumSelect`): the f64 scores
//! for the numerical gate, and an F32 emulation of the engine's score
//! arithmetic for exact selections. Selections, per-rank pairs and the
//! merge take the F32 scores through `pooled_indexer_oracle::{select,
//! select_per_rank, merge}`.

use crate::codec::bf16_to_f32;
use crate::contract::{Refusal, Validate};
use crate::oracle::{OracleOutput, compensated_sum};
use crate::pooled_indexer::PooledIndexerQuery;

/// The (heads, head_dim) of a head-sum selection.
fn shape(query: PooledIndexerQuery) -> Result<(usize, usize), Refusal> {
    query.validate()?;
    match query {
        PooledIndexerQuery::HeadSumSelect {
            heads, head_dim, ..
        } => Ok((heads as usize, head_dim as usize)),
        _ => Err(Refusal("expected a head-sum indexer selection")),
    }
}

fn check_operands(h: usize, d: usize, q: usize, keys: usize) -> Result<(), Refusal> {
    if q != h * d || !keys.is_multiple_of(d) {
        return Err(Refusal("head-sum indexer score operand shape mismatch"));
    }
    Ok(())
}

/// F64 scores of one row's pools: `q` is `[heads, head_dim]` and
/// `pooled_keys` `[pools, head_dim]` (decoded BF16). The value of pool g is
/// `head_dim^-0.5 * sum_h relu(q_h . K_g)` with compensated sums. S is
/// `head_dim^-0.5 * sum_h sum_d |q_hd K_gd|` and K is `head_dim + heads`,
/// the dot chains then the head sum.
pub fn head_sum_scores(
    query: PooledIndexerQuery,
    q: &[f64],
    pooled_keys: &[f64],
) -> Result<OracleOutput, Refusal> {
    let (h, d) = shape(query)?;
    check_operands(h, d, q.len(), pooled_keys.len())?;
    let scale = (d as f64).sqrt().recip();
    let mut values = Vec::with_capacity(pooled_keys.len() / d);
    let mut s = Vec::with_capacity(values.capacity());
    for key in pooled_keys.chunks_exact(d) {
        let relu = q
            .chunks_exact(d)
            .map(|head| compensated_sum(head.iter().zip(key).map(|(a, b)| a * b)).max(0.0));
        values.push(compensated_sum(relu) * scale);
        let magnitude = q
            .chunks_exact(d)
            .map(|head| compensated_sum(head.iter().zip(key).map(|(a, b)| (a * b).abs())));
        s.push(compensated_sum(magnitude) * scale);
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

/// The head sum's F32 pairwise tree over `relu` (4, 8 or 16 heads):
/// `(r0 + r1) + (r2 + r3)` per four heads, two of those added at eight, and
/// at sixteen the eight-head tree over `r_h + r_(h + 8)`.
fn head_tree(relu: &[f32]) -> f32 {
    let folded;
    let r = if relu.len() == 16 {
        folded = (0..8).map(|h| relu[h] + relu[h + 8]).collect::<Vec<_>>();
        &folded[..]
    } else {
        relu
    };
    let four = |b: usize| (r[b] + r[b + 1]) + (r[b + 2] + r[b + 3]);
    if r.len() == 4 {
        four(0)
    } else {
        four(0) + four(4)
    }
}

/// The engine's F32 scores of one row's pools: `q` `[heads, head_dim]` and
/// `pooled_keys` `[pools, head_dim]` as BF16 bits. Per head, the even and
/// the odd k16 steps each run a chain from +0: a step adds its 16 products
/// (exact in f64, summed in f64) to its chain with one rounding to F32.
/// Then `relu(even + odd)` in F32, the head tree (`head_tree`) and the F32
/// multiply by the F32 nearest to `head_dim^-0.5`.
///
/// The tensor core's alignment inside a k16 step is not modelled: the
/// emulation is the engine's bits wherever each step's sum and its chain
/// add are exact in F32 (small-integer and other short operands), and holds
/// the score gate otherwise. Everything after the chains is the engine's F32
/// arithmetic exactly.
pub fn head_sum_scores_f32(
    query: PooledIndexerQuery,
    q: &[u16],
    pooled_keys: &[u16],
) -> Result<Vec<f32>, Refusal> {
    let (h, d) = shape(query)?;
    check_operands(h, d, q.len(), pooled_keys.len())?;
    let decode = |bits: &[u16]| {
        bits.iter()
            .map(|&b| f64::from(bf16_to_f32(b)))
            .collect::<Vec<_>>()
    };
    let (q, keys) = (decode(q), decode(pooled_keys));
    let scale = (d as f64).sqrt().recip() as f32;
    Ok(keys
        .chunks_exact(d)
        .map(|key| {
            let relu = q
                .chunks_exact(d)
                .map(|head| {
                    let mut chains = [0_f32; 2];
                    let steps = head.chunks_exact(16).zip(key.chunks_exact(16));
                    for (step, (a, b)) in steps.enumerate() {
                        let products = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
                        let chain = &mut chains[step % 2];
                        *chain = (f64::from(*chain) + products) as f32;
                    }
                    let x = chains[0] + chains[1];
                    if x > 0.0 { x } else { 0.0 }
                })
                .collect::<Vec<_>>();
            head_tree(&relu) * scale
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::f32_to_bf16;
    use crate::context_parallel::ContextParallelQuery;
    use crate::contract::SparseAttentionQuery;
    use crate::pooled_indexer::IndexerSelectionMode;
    use crate::pooled_indexer_oracle::{merge, select, select_per_rank};
    use crate::sparse_attention_oracle as blocks;

    const HEADS: usize = 4;
    const DIM: usize = 128;
    const GROUP: usize = 4;
    const PAGE: usize = 64;
    const MAX_TOP_K: u32 = 2048;
    const MAX_BLOCKS: u32 = 1 << 16;

    fn query(heads: u32, mode: IndexerSelectionMode) -> PooledIndexerQuery {
        PooledIndexerQuery::HeadSumSelect {
            heads,
            head_dim: DIM as u32,
            group: GROUP as u32,
            max_top_k: MAX_TOP_K,
            always_tail: true,
            mode,
            page_size: PAGE as u32,
            capacity: 1,
        }
    }

    const GLOBAL: IndexerSelectionMode = IndexerSelectionMode::Global;

    /// A deterministic generator (xorshift64*).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        /// `k * 2^-shift` for an integer k in -range..=range, as BF16.
        fn int(&mut self, range: u64, shift: i32) -> u16 {
            let k = (self.next() % (2 * range + 1)) as i64 - range as i64;
            f32_to_bf16(k as f32 * 2_f32.powi(-shift))
        }
    }

    fn decode(bits: &[u16]) -> Vec<f64> {
        bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
    }

    /// The block selection's own reference: its f64 block scores and exact
    /// top-k, as the token set of a row at `position`.
    fn block_reference(q: &[u16], keys: &[u16], position: usize, top_k: u32) -> Vec<usize> {
        let visible = position + 1;
        let scores = blocks::block_scores(
            SparseAttentionQuery::BlockScores {
                heads: HEADS as u32,
                head_dim: DIM as u32,
                max_blocks: MAX_BLOCKS,
                capacity: 1,
            },
            &decode(q),
            &decode(&keys[..visible / GROUP * DIM]),
        )
        .unwrap();
        blocks::block_topk(
            SparseAttentionQuery::BlockTopK {
                block: GROUP as u32,
                top_blocks: top_k / GROUP as u32,
                max_blocks: MAX_BLOCKS,
                capacity: 1,
            },
            &scores.values,
            visible,
        )
        .unwrap()
    }

    fn tokens(list: &[i32]) -> Vec<usize> {
        list.iter()
            .take_while(|&&t| t >= 0)
            .map(|&t| t as usize)
            .collect()
    }

    /// Small-integer operands scaled by powers of two: every dot, chain,
    /// head sum and score is exact before the scale, few distinct values
    /// make many ties, and the emulation's selections equal the block
    /// selection reference's (f64 scores, exact top-k, ties to the lower
    /// block) at and around the dense limit, at both top-k values, and
    /// through the per-rank pairs merged at 2 and 4 ranks.
    #[test]
    fn emulated_selections_equal_the_block_selection_reference() {
        let mut rng = Rng(0x5eed_4ead_5a11_0001);
        let pools = 3000;
        let keys = (0..pools * DIM).map(|_| rng.int(1, 2)).collect::<Vec<_>>();
        for pattern in 0..3 {
            let q = (0..HEADS * DIM)
                .map(|i| match pattern {
                    // Only head 0 sees the keys: every other head is zero.
                    0 if i >= DIM => 0,
                    _ => rng.int(2, pattern),
                })
                .collect::<Vec<_>>();
            let scores = head_sum_scores_f32(query(4, GLOBAL), &q, &keys).unwrap();
            let exact = head_sum_scores(query(4, GLOBAL), &decode(&q), &decode(&keys)).unwrap();
            let scale = (DIM as f64).sqrt().recip();
            let (qf, kf) = (decode(&q), decode(&keys));
            for (pool, key) in kf.chunks_exact(DIM).enumerate() {
                // Exact in f64 and in F32: at most 256 in steps of 2^-4.
                let sum = qf
                    .chunks_exact(DIM)
                    .map(|head| {
                        head.iter()
                            .zip(key)
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            .max(0.0)
                    })
                    .sum::<f64>();
                assert_eq!(scores[pool], sum as f32 * scale as f32, "pool {pool}");
                assert_eq!(exact.values[pool], sum * scale, "pool {pool}");
            }
            for position in [0_usize, 3, 2050, 2051, 2052, 2055, 4095, 4097, 8191, 11_999] {
                for top_k in [MAX_TOP_K, 1024] {
                    let visible = (position + 1) / GROUP;
                    let row = &scores[..visible];
                    let list = select(query(4, GLOBAL), top_k, position as u64, row).unwrap();
                    assert_eq!(
                        tokens(&list),
                        block_reference(&q, &keys, position, top_k),
                        "pattern {pattern} position {position} top {top_k}"
                    );
                    for ranks in [2_u32, 4] {
                        let per_rank = query(4, IndexerSelectionMode::PerRank { ranks });
                        let lists = (0..ranks as usize)
                            .map(|rank| {
                                let owned = (0..visible)
                                    .map(|g| g * GROUP / PAGE % ranks as usize == rank)
                                    .collect::<Vec<_>>();
                                select_per_rank(per_rank, top_k, position as u64, row, &owned)
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        let refs = lists.iter().map(|l| &l[..]).collect::<Vec<_>>();
                        let merged = merge(
                            ContextParallelQuery::TopKMerge {
                                ranks,
                                group: GROUP as u32,
                                max_top_k: MAX_TOP_K,
                                always_tail: true,
                                capacity: 1,
                            },
                            top_k,
                            position as u64,
                            &refs,
                        )
                        .unwrap();
                        assert_eq!(merged, list, "{ranks} ranks, position {position}");
                    }
                }
            }
        }
    }

    /// The emulation adds the even and odd k16 steps in two chains and the
    /// heads in a tree, which a single chain or a head-order sum would not
    /// reproduce: with the products 1, 2^-24 and 2^-24 at k16 steps 0, 1 and
    /// 3, one chain rounds both small terms away and two chains keep
    /// 2^-23; with head dots 1, 2^-24, 2^-24 and 2^-24, a head-order sum
    /// stays at 1 and the tree gives 1 + 2^-23.
    #[test]
    fn chains_and_head_tree_keep_their_order() {
        let tiny = f32_to_bf16(2_f32.powi(-12));
        let one = f32_to_bf16(1.0);
        let scale = (DIM as f64).sqrt().recip() as f32;
        let expected = (1.0 + 2_f32.powi(-23)) * scale;
        let mut key = vec![0_u16; DIM];
        key[0] = one;
        key[16] = tiny;
        key[48] = tiny;
        let mut q = vec![0_u16; HEADS * DIM];
        q[0] = one;
        q[16] = tiny;
        q[48] = tiny;
        let chains = head_sum_scores_f32(query(4, GLOBAL), &q, &key).unwrap();
        assert_eq!(chains, vec![expected]);
        let mut q = vec![0_u16; HEADS * DIM];
        q[0] = one;
        for head in 1..HEADS {
            q[head * DIM] = f32_to_bf16(2_f32.powi(-24));
        }
        let mut key = vec![0_u16; DIM];
        key[0] = one;
        let tree = head_sum_scores_f32(query(4, GLOBAL), &q, &key).unwrap();
        assert_eq!(tree, vec![expected]);
        assert_eq!(
            ((1.0 + 2_f32.powi(-24)) + 2_f32.powi(-24)) + 2_f32.powi(-24),
            1.0
        );
    }

    /// Eight heads add two four-head trees; sixteen first fold head h with
    /// head h + 8. Head dots of 1 at head 0 and 2^-24 elsewhere tell the
    /// orders apart.
    #[test]
    fn wider_head_trees_follow_the_engine_order() {
        let small = 2_f32.powi(-24);
        let mut relu = vec![small; 16];
        relu[0] = 1.0;
        let ulp = 2_f32.powi(-23);
        // Eight: (1 + s) + (s + s) is 1 + 2^-23, plus the other four's 4s.
        assert_eq!(head_tree(&relu[..8]), 1.0 + 3.0 * ulp);
        // Sixteen: the fold 1 + s rounds to 1 and the other seven are 2s,
        // so the eight-head tree gives (1 + 2^-23 + 2^-22) + 8s.
        assert_eq!(head_tree(&relu), 1.0 + 7.0 * ulp);
        assert_eq!(head_tree(&relu[..4]), 1.0 + 2_f32.powi(-23));
    }

    #[test]
    fn scores_refuse_other_queries_and_shapes() {
        let q = vec![0_u16; HEADS * DIM];
        assert!(head_sum_scores_f32(query(4, GLOBAL), &q[1..], &q[..DIM]).is_err());
        assert!(head_sum_scores_f32(query(4, GLOBAL), &q, &q[..DIM - 1]).is_err());
        assert!(head_sum_scores_f32(query(12, GLOBAL), &q, &q[..DIM]).is_err());
        assert!(head_sum_scores(query(4, GLOBAL), &decode(&q[DIM..]), &[]).is_err());
        let empty = head_sum_scores(query(4, GLOBAL), &decode(&q), &[]).unwrap();
        assert!(empty.values.is_empty());
    }
}
