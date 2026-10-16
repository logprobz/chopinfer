//! Host references of the context-parallel entries (`context_parallel`):
//! the exact key list localization, an F32 emulation of the partial
//! merge's arithmetic and the f64 paged attention partials. The f64 partial merge is
//! `latent_attention_oracle::merge_partials`; the top-k merge oracle is
//! `pooled_indexer_oracle::merge`.

use crate::codec::f32_to_bf16;
use crate::context_parallel::ContextParallelQuery;
use crate::contract::{Refusal, Validate};
use crate::latent_attention_oracle::LatentPartials;
use crate::oracle::{NeumaierSum, OracleOutput};

/// `log2(e)` in F32, as the engine multiplies it in before `exp2`.
pub const LOG2_E: f32 = std::f32::consts::LOG2_E;

/// One rank's attention partials: F32 maximum and denominator `[pairs]` and
/// numerator `[pairs, width]`.
#[derive(Clone, Copy, Debug)]
pub struct PartialsF32<'a> {
    pub maximum: &'a [f32],
    pub denominator: &'a [f32],
    pub numerator: &'a [f32],
}

/// Group rank `rank`'s local key lists for `lists` under a `LocalizeKeys`
/// query: each owned position becomes its local position, every other entry
/// -1.
pub fn localize_keys(
    query: ContextParallelQuery,
    rank: u32,
    lists: &[i32],
) -> Result<Vec<i32>, Refusal> {
    query.validate()?;
    let ContextParallelQuery::LocalizeKeys {
        ranks, page, width, ..
    } = query
    else {
        return Err(Refusal("key localization needs a LocalizeKeys query"));
    };
    if rank >= ranks || !lists.len().is_multiple_of(width as usize) {
        return Err(Refusal("key localization rank or list shape is invalid"));
    }
    let (ranks, page) = (i64::from(ranks), i64::from(page));
    Ok(lists
        .iter()
        .map(|&position| {
            let p = i64::from(position);
            if p < 0 || p / page % ranks != i64::from(rank) {
                -1
            } else {
                (p / (ranks * page) * page + p % page) as i32
            }
        })
        .collect())
}

/// Where group rank `rank` writes each row's K and V under an
/// `OwnedQkNormRopeKvWrite` query: the local position of each owned cache
/// position, -1 for the others (the `LocalizeKeys` list of width 1 the
/// write reads).
pub fn owned_kv_positions(
    query: ContextParallelQuery,
    rank: u32,
    positions: &[i32],
) -> Result<Vec<i32>, Refusal> {
    query.validate()?;
    let ContextParallelQuery::OwnedQkNormRopeKvWrite {
        write: crate::contract::GatedAttentionQuery::QkNormRopeKvWrite { page_size, .. },
        ranks,
    } = query
    else {
        return Err(Refusal("an owner-routed KV write query is expected"));
    };
    localize_keys(
        ContextParallelQuery::LocalizeKeys {
            ranks,
            page: page_size,
            width: 1,
            capacity: positions.len().max(1) as u32,
        },
        rank,
        positions,
    )
}

/// Where group rank `rank` stores block `block`'s key under an
/// `OwnedIndexerKeyWrite` query: `Some((stripe, row))`, the rank's local page
/// `table[stripe]` and its row, or `None` when another rank owns it.
pub fn owned_block_row(
    query: ContextParallelQuery,
    rank: u32,
    block: u64,
) -> Result<Option<(u64, u64)>, Refusal> {
    query.validate()?;
    let ContextParallelQuery::OwnedIndexerKeyWrite {
        write: crate::contract::SparseAttentionQuery::IndexerKeyWrite { block: tokens, .. },
        page,
        ranks,
    } = query
    else {
        return Err(Refusal(
            "an owner-routed indexer key write query is expected",
        ));
    };
    if rank >= ranks {
        return Err(Refusal("group rank outside the group"));
    }
    let page_blocks = u64::from(page / tokens);
    let physical = block / page_blocks;
    Ok((physical % u64::from(ranks) == u64::from(rank))
        .then_some((physical / u64::from(ranks), block % page_blocks)))
}

/// The partial merge in F32 exactly as the contract orders it, with each
/// weight computed as `exp2((m_r - M) * LOG2_E)`: `M = max m_r`, `a_r` (0
/// for an empty rank), `L` and `O` FMA sums from zero in rank order, BF16
/// `O / L` (zeros when every rank is empty). Bitwise wherever every weight
/// is exact (one rank, or ranks of equal maxima); otherwise the device's
/// `exp2` may differ by its error bound.
pub fn partial_merge_f32(ranks: &[PartialsF32<'_>]) -> Result<Vec<u16>, Refusal> {
    let Some(first) = ranks.first() else {
        return Err(Refusal("a partial merge needs a rank"));
    };
    let pairs = first.maximum.len();
    let elements = first.numerator.len();
    if pairs == 0
        || !elements.is_multiple_of(pairs)
        || ranks.iter().any(|rank| {
            rank.maximum.len() != pairs
                || rank.denominator.len() != pairs
                || rank.numerator.len() != elements
        })
    {
        return Err(Refusal("partial merge ranks differ in shape"));
    }
    let width = elements / pairs;
    let mut out = Vec::with_capacity(elements);
    for pair in 0..pairs {
        let top = ranks
            .iter()
            .fold(f32::NEG_INFINITY, |top, rank| top.max(rank.maximum[pair]));
        let weights: Vec<f32> = ranks
            .iter()
            .map(|rank| {
                let m = rank.maximum[pair];
                if m == f32::NEG_INFINITY {
                    0.0
                } else {
                    ((m - top) * LOG2_E).exp2()
                }
            })
            .collect();
        let total = ranks.iter().zip(&weights).fold(0.0_f32, |sum, (rank, &a)| {
            a.mul_add(rank.denominator[pair], sum)
        });
        for column in 0..width {
            let numerator = ranks.iter().zip(&weights).fold(0.0_f32, |sum, (rank, &a)| {
                a.mul_add(rank.numerator[pair * width + column], sum)
            });
            out.push(f32_to_bf16(if total == 0.0 {
                0.0
            } else {
                numerator / total
            }));
        }
    }
    Ok(out)
}

/// One `PagedAttentionPartials` launch's operands, decoded to f64.
#[derive(Clone, Copy, Debug)]
pub struct PagedPartialsInput<'a> {
    /// `[rows][kv_heads * q_heads][head_dim]`.
    pub q: &'a [f64],
    /// The rank's decoded K and V pages `[page][page_size][kv_heads]
    /// [head_dim]`, FP8 values before their layer scale.
    pub keys: &'a [f64],
    pub values: &'a [f64],
    /// Per item, the rank's own block table of local pages.
    pub tables: &'a [Vec<usize>],
    pub row_batch: &'a [usize],
    pub positions: &'a [usize],
    pub scale: f64,
    pub k_scale: f64,
    pub v_scale: f64,
    pub cp_rank: usize,
    pub cp_size: usize,
}

/// The f64 reference of `ContextParallelQuery::PagedAttentionPartials`.
/// With n kept keys, width d and score magnitude `g_j = scale * k_scale *
/// sum_i |q_i k_ji|`, the maximum's S is the maximizing key's g and K is d;
/// the denominator's S is `sum_j w_j (1 + (d / n) g_j)` and a numerator's
/// `v_scale * sum_j w_j |v_j| (1 + (d / n) g_j)`, each with K = n (the
/// latent attention partials' bounds, `latent_attention_oracle`).
pub fn paged_attention_partials(
    query: ContextParallelQuery,
    input: PagedPartialsInput<'_>,
) -> Result<LatentPartials, Refusal> {
    query.validate()?;
    let ContextParallelQuery::PagedAttentionPartials {
        q_heads,
        kv_heads,
        head_dim,
        page,
        capacity,
        ..
    } = query
    else {
        return Err(Refusal("not a paged attention partials query"));
    };
    let shape = Shape::new(q_heads, kv_heads, head_dim, page, capacity, &input)?;
    shape.partials(&input, |row| Ok((0..=input.positions[row]).collect()))
}

/// The f64 reference of `ContextParallelQuery::SelectedAttentionPartials`:
/// `paged_attention_partials` over each row's key set, the tokens of its
/// selected blocks `selected[row][..min(c, top_blocks)]` (`[rows,
/// top_blocks]`, distinct and ascending below the row's c complete blocks)
/// then its incomplete tail, with the same bounds.
pub fn selected_attention_partials(
    query: ContextParallelQuery,
    input: PagedPartialsInput<'_>,
    selected: &[i32],
) -> Result<LatentPartials, Refusal> {
    query.validate()?;
    let ContextParallelQuery::SelectedAttentionPartials {
        q_heads,
        kv_heads,
        head_dim,
        page,
        block,
        top_blocks,
        capacity,
        ..
    } = query
    else {
        return Err(Refusal("not a selected attention partials query"));
    };
    let shape = Shape::new(q_heads, kv_heads, head_dim, page, capacity, &input)?;
    let (block, top) = (block as usize, top_blocks as usize);
    if selected.len() != input.positions.len() * top {
        return Err(Refusal("selected attention partial list shape mismatch"));
    }
    shape.partials(&input, |row| {
        let position = input.positions[row];
        let complete = (position + 1) / block;
        let list = &selected[row * top..][..complete.min(top)];
        if list.iter().any(|&b| b < 0 || b as usize >= complete)
            || list.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(Refusal(
                "selected blocks must be distinct, ascending and complete",
            ));
        }
        Ok(list
            .iter()
            .flat_map(|&b| b as usize * block..(b as usize + 1) * block)
            .chain(complete * block..=position)
            .collect())
    })
}

/// The partial operands' shape, checked against the launch's input.
struct Shape {
    q_heads: usize,
    kv_heads: usize,
    d: usize,
    page: usize,
    pages: usize,
}

impl Shape {
    fn new(
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        page: u32,
        capacity: u32,
        input: &PagedPartialsInput<'_>,
    ) -> Result<Shape, Refusal> {
        let (q_heads, kv_heads, d, page) = (
            q_heads as usize,
            kv_heads as usize,
            head_dim as usize,
            page as usize,
        );
        let rows = input.positions.len();
        let page_floats = page * kv_heads * d;
        if rows == 0
            || rows > capacity as usize
            || input.row_batch.len() != rows
            || input.q.len() != rows * q_heads * kv_heads * d
            || input.keys.len() != input.values.len()
            || !input.keys.len().is_multiple_of(page_floats)
            || input.cp_size == 0
            || input.cp_rank >= input.cp_size
        {
            return Err(Refusal("paged attention partial operand shape mismatch"));
        }
        Ok(Shape {
            q_heads,
            kv_heads,
            d,
            page,
            pages: input.keys.len() / page_floats,
        })
    }

    /// Every row's partials over the rank's share of `key_set(row)`, the
    /// row's positions in order.
    fn partials(
        &self,
        input: &PagedPartialsInput<'_>,
        key_set: impl Fn(usize) -> Result<Vec<usize>, Refusal>,
    ) -> Result<LatentPartials, Refusal> {
        let (q_heads, kv_heads, d, page) = (self.q_heads, self.kv_heads, self.d, self.page);
        let heads = q_heads * kv_heads;
        let empty = || OracleOutput {
            values: Vec::new(),
            sum_abs_products: Vec::new(),
            reduction_lengths: Vec::new(),
        };
        let (mut maximum, mut denominator, mut numerator) = (empty(), empty(), empty());
        let push = |out: &mut OracleOutput, value: f64, bound: f64, k: usize| {
            out.values.push(value);
            out.sum_abs_products.push(bound);
            out.reduction_lengths.push(k.max(1));
        };
        for row in 0..input.positions.len() {
            let table = input
                .tables
                .get(input.row_batch[row])
                .ok_or(Refusal("paged attention partial row names no item"))?;
            // The kept keys' element offsets, in position order.
            let mut kept = Vec::new();
            for t in key_set(row)? {
                let physical = t / page;
                if physical % input.cp_size != input.cp_rank {
                    continue;
                }
                let local = *table
                    .get(physical / input.cp_size)
                    .ok_or(Refusal("paged attention partial table is too short"))?;
                if local >= self.pages {
                    return Err(Refusal("paged attention partial page outside the cache"));
                }
                kept.push((local * page + t % page) * kv_heads * d);
            }
            let n = kept.len();
            let spread = if n == 0 { 0.0 } else { d as f64 / n as f64 };
            for head in 0..heads {
                let kv = head / q_heads;
                let q = &input.q[(row * heads + head) * d..][..d];
                let factor = input.scale * input.k_scale;
                let (mut scores, mut magnitudes) = (Vec::with_capacity(n), Vec::with_capacity(n));
                for &base in &kept {
                    let key = &input.keys[base + kv * d..][..d];
                    let (mut dot, mut magnitude) = (NeumaierSum::default(), NeumaierSum::default());
                    for (&a, &b) in q.iter().zip(key) {
                        dot.add(a * b);
                        magnitude.add((a * b).abs());
                    }
                    scores.push(factor * dot.value());
                    magnitudes.push(factor * magnitude.value());
                }
                let (m, m_bound) = scores.iter().zip(&magnitudes).fold(
                    (f64::NEG_INFINITY, 0.0),
                    |(m, b), (&s, &g)| {
                        if s > m { (s, g) } else { (m, b) }
                    },
                );
                let weights = scores.iter().map(|&s| (s - m).exp()).collect::<Vec<_>>();
                push(&mut maximum, m, m_bound, if n == 0 { 1 } else { d });
                let (mut l, mut l_bound) = (NeumaierSum::default(), NeumaierSum::default());
                for (&w, &g) in weights.iter().zip(&magnitudes) {
                    l.add(w);
                    l_bound.add(w * (1.0 + spread * g));
                }
                push(&mut denominator, l.value(), l_bound.value(), n);
                for column in 0..d {
                    let (mut o, mut bound) = (NeumaierSum::default(), NeumaierSum::default());
                    for ((&w, &base), &g) in weights.iter().zip(&kept).zip(&magnitudes) {
                        let v = input.values[base + kv * d + column];
                        o.add(w * v);
                        bound.add((w * v).abs() * (1.0 + spread * g));
                    }
                    push(
                        &mut numerator,
                        input.v_scale * o.value(),
                        input.v_scale * bound.value(),
                        n,
                    );
                }
            }
        }
        Ok(LatentPartials {
            maximum,
            denominator,
            numerator,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::bf16_to_f32;

    fn localize(ranks: u32) -> ContextParallelQuery {
        ContextParallelQuery::LocalizeKeys {
            ranks,
            page: 4,
            width: 6,
            capacity: 4,
        }
    }

    #[test]
    fn owned_positions_become_local_and_others_holes() {
        // Pages of 4 tokens: page j on rank j % 2, at local stripe j / 2.
        let list = [0, 3, 4, 9, 13, -1];
        assert_eq!(
            localize_keys(localize(2), 0, &list).unwrap(),
            [0, 3, -1, 5, -1, -1]
        );
        assert_eq!(
            localize_keys(localize(2), 1, &list).unwrap(),
            [-1, -1, 0, -1, 5, -1]
        );
        // Four ranks: position 13 is page 3, rank 3's first page.
        assert_eq!(
            localize_keys(localize(4), 3, &list).unwrap(),
            [-1, -1, -1, -1, 1, -1]
        );
        // Every valid position has exactly one owner.
        let all: Vec<i32> = (0..48).collect();
        let query = ContextParallelQuery::LocalizeKeys {
            ranks: 4,
            page: 4,
            width: 48,
            capacity: 1,
        };
        let owned = (0..4)
            .map(|rank| localize_keys(query, rank, &all).unwrap())
            .collect::<Vec<_>>();
        for position in 0..48 {
            assert_eq!(owned.iter().filter(|list| list[position] >= 0).count(), 1);
        }
        assert!(localize_keys(localize(2), 2, &list).is_err());
        assert!(localize_keys(localize(2), 0, &list[..5]).is_err());
    }

    #[test]
    fn one_rank_reproduces_the_normalized_output() {
        let (m, l, o) = (
            [0.5_f32, f32::NEG_INFINITY],
            [3.0_f32, 0.0],
            [1.0_f32, 2.0, 0.0, 0.0],
        );
        let one = PartialsF32 {
            maximum: &m,
            denominator: &l,
            numerator: &o,
        };
        let out = partial_merge_f32(&[one]).unwrap();
        assert_eq!(out, [f32_to_bf16(1.0 / 3.0), f32_to_bf16(2.0 / 3.0), 0, 0]);
    }

    #[test]
    fn ranks_fold_in_order_with_exact_weights_at_equal_maxima() {
        // Equal maxima give weights of exactly 1; an empty rank weighs 0.
        let ranks = [
            ([2.0_f32], [1.0_f32], [1.0_f32, -1.0]),
            ([f32::NEG_INFINITY], [0.0], [0.0, 0.0]),
            ([2.0], [3.0], [5.0, 1.0]),
        ];
        let views: Vec<_> = ranks
            .iter()
            .map(|(m, l, o)| PartialsF32 {
                maximum: m,
                denominator: l,
                numerator: o,
            })
            .collect();
        let out = partial_merge_f32(&views).unwrap();
        assert_eq!(out, [f32_to_bf16(6.0 / 4.0), f32_to_bf16(0.0 / 4.0)]);
        // A smaller maximum is scaled by exp(m - M).
        let low = PartialsF32 {
            maximum: &[0.0],
            denominator: &[1.0],
            numerator: &[1.0, 1.0],
        };
        let high = PartialsF32 {
            maximum: &[1.0],
            denominator: &[1.0],
            numerator: &[1.0, 1.0],
        };
        let out = partial_merge_f32(&[low, high]).unwrap();
        let a = (-LOG2_E).exp2();
        let expected = f64::from(a.mul_add(1.0, 1.0)) / f64::from(a.mul_add(1.0, 1.0));
        assert_eq!(f64::from(bf16_to_f32(out[0])), expected);
        // Every rank empty: zeros.
        let empty = PartialsF32 {
            maximum: &[f32::NEG_INFINITY],
            denominator: &[0.0],
            numerator: &[0.0, 0.0],
        };
        assert_eq!(partial_merge_f32(&[empty, empty]).unwrap(), [0, 0]);
        assert!(partial_merge_f32(&[]).is_err());
    }

    /// Partials of each group rank, merged by `merge_partials`, equal the
    /// size-1 partial's normalized output, and every rank's keys partition
    /// the visible keys.
    #[test]
    fn rank_partials_merge_into_the_full_attention() {
        use crate::contract::{AttentionPhase, DType};
        use crate::latent_attention_oracle::merge_partials;
        let query = ContextParallelQuery::PagedAttentionPartials {
            q_heads: 2,
            kv_heads: 1,
            head_dim: 8,
            page: 4,
            kv: DType::Bf16,
            phase: AttentionPhase::Verify,
            capacity: 8,
        };
        let mut state = 0x1234_5678_u64;
        let mut unit = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0
        };
        // 13 tokens of one item: 4 physical pages.
        let (tokens, d, page) = (13_usize, 8_usize, 4_usize);
        let physical: Vec<f64> = (0..tokens.div_ceil(page) * page * d)
            .map(|_| unit())
            .collect();
        let values: Vec<f64> = (0..physical.len()).map(|_| unit()).collect();
        let rows = [0_usize, 5, 12];
        let q: Vec<f64> = (0..rows.len() * 2 * d).map(|_| 3.0 * unit()).collect();
        let run = |rank: usize, size: usize| {
            // The rank's local pages hold its physical pages j, j + size, ...
            let mine = (0..tokens.div_ceil(page))
                .filter(|j| j % size == rank)
                .collect::<Vec<_>>();
            let pick = |cache: &[f64]| {
                mine.iter()
                    .flat_map(|&j| cache[j * page * d..(j + 1) * page * d].to_vec())
                    .collect::<Vec<_>>()
            };
            let (keys, vals) = (pick(&physical), pick(&values));
            let tables = vec![(0..mine.len()).collect::<Vec<_>>()];
            paged_attention_partials(
                query,
                PagedPartialsInput {
                    q: &q,
                    keys: &keys,
                    values: &vals,
                    tables: &tables,
                    row_batch: &[0, 0, 0],
                    positions: &rows,
                    scale: 0.35,
                    k_scale: 1.0,
                    v_scale: 1.0,
                    cp_rank: rank,
                    cp_size: size,
                },
            )
            .unwrap()
        };
        let full = merge_partials(&[run(0, 1)]).unwrap();
        for size in [2, 4] {
            let parts = (0..size).map(|rank| run(rank, size)).collect::<Vec<_>>();
            let merged = merge_partials(&parts).unwrap();
            for (a, b) in merged.values.iter().zip(&full.values) {
                assert!((a - b).abs() <= 1e-12, "size {size}: {a} vs {b}");
            }
            // Row 0 sees one key, on rank 0 only.
            assert_eq!(parts[1].maximum.values[0], f64::NEG_INFINITY);
            assert_eq!(parts[1].denominator.values[0], 0.0);
        }
    }

    /// Selected partials attend each row's selected blocks then its tail:
    /// one rank's merged output equals the block-sparse attention oracle on
    /// the same key sets, every group's ranks merge into it, and malformed
    /// lists are refused.
    #[test]
    fn selected_partials_merge_into_the_sparse_attention() {
        use crate::contract::{AttentionPhase, DType, SparseAttentionQuery};
        use crate::latent_attention_oracle::merge_partials;
        use crate::sparse_attention_oracle::{SparseKv, sparse_attention};
        let (tokens, d, page, block, top) = (40_usize, 64_usize, 8_usize, 2_usize, 3_usize);
        let query = ContextParallelQuery::SelectedAttentionPartials {
            q_heads: 2,
            kv_heads: 1,
            head_dim: d as u32,
            page: page as u32,
            block: block as u32,
            top_blocks: top as u32,
            kv: DType::Bf16,
            phase: AttentionPhase::Verify,
            capacity: 8,
        };
        let mut state = 0x5e1e_c7ed_u64;
        let mut unit = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0
        };
        let keys: Vec<f64> = (0..tokens * d).map(|_| unit()).collect();
        let values: Vec<f64> = (0..tokens * d).map(|_| unit()).collect();
        // Complete blocks 0, 3, 4, 12 and 19; the first rows are dense.
        let positions = [0_usize, 6, 9, 24, 39];
        let selected: Vec<i32> =
            [[-1, -1, -1], [0, 1, 2], [0, 2, 3], [1, 5, 11], [3, 4, 17]].concat();
        let q: Vec<f64> = (0..positions.len() * 2 * d).map(|_| 3.0 * unit()).collect();
        let run = |rank: usize, size: usize, selected: &[i32]| {
            let mine = (0..tokens / page)
                .filter(|j| j % size == rank)
                .collect::<Vec<_>>();
            let pick = |cache: &[f64]| {
                mine.iter()
                    .flat_map(|&j| cache[j * page * d..(j + 1) * page * d].to_vec())
                    .collect::<Vec<_>>()
            };
            let (k, v) = (pick(&keys), pick(&values));
            let tables = vec![(0..mine.len()).collect::<Vec<_>>()];
            selected_attention_partials(
                query,
                PagedPartialsInput {
                    q: &q,
                    keys: &k,
                    values: &v,
                    tables: &tables,
                    row_batch: &[0; 5],
                    positions: &positions,
                    scale: f64::from(0.35_f32),
                    k_scale: 1.0,
                    v_scale: 1.0,
                    cp_rank: rank,
                    cp_size: size,
                },
                selected,
            )
        };
        let full = merge_partials(&[run(0, 1, &selected).unwrap()]).unwrap();
        let key_sets = positions
            .iter()
            .enumerate()
            .map(|(row, &p)| {
                let complete = (p + 1) / block;
                selected[row * top..][..complete.min(top)]
                    .iter()
                    .flat_map(|&b| b as usize * block..(b as usize + 1) * block)
                    .chain(complete * block..=p)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(key_sets[1], vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(key_sets[4], vec![6, 7, 8, 9, 34, 35]);
        let reference = sparse_attention(
            SparseAttentionQuery::SparseAttention {
                q_heads: 2,
                kv_heads: 1,
                head_dim: d as u32,
                block: block as u32,
                top_blocks: top as u32,
                softmax_scale_bits: 0.35_f32.to_bits(),
                page_size: page as u32,
                kv_dtype: DType::Bf16,
                phase: AttentionPhase::Verify,
                capacity: 8,
            },
            &q,
            &key_sets,
            SparseKv {
                keys: &keys,
                values: &values,
                block_table: &(0..tokens / page).collect::<Vec<_>>(),
            },
        )
        .unwrap();
        for (a, b) in full.values.iter().zip(&reference.values) {
            assert!((a - b).abs() <= 1e-12, "{a} vs {b}");
        }
        for size in [2, 4] {
            let parts = (0..size)
                .map(|rank| run(rank, size, &selected).unwrap())
                .collect::<Vec<_>>();
            let merged = merge_partials(&parts).unwrap();
            for (a, b) in merged.values.iter().zip(&full.values) {
                assert!((a - b).abs() <= 1e-12, "size {size}: {a} vs {b}");
            }
            // Row 0 sees token 0 alone, on rank 0.
            assert_eq!(parts[1].maximum.values[0], f64::NEG_INFINITY);
        }
        for bad in [[1, 0, 2], [0, 0, 2], [0, 1, 3]] {
            let mut list = selected.clone();
            list[3..6].copy_from_slice(&bad);
            assert!(run(0, 1, &list).is_err(), "{bad:?}");
        }
        assert!(run(0, 1, &selected[1..]).is_err());
    }

    /// Owner routing: a position's K and V go to the rank of its page at
    /// its local position, and a block key to the rank of its page.
    #[test]
    fn owner_routing_places_rows_on_their_pages() {
        use crate::contract::{DType, GatedAttentionQuery, SparseAttentionQuery};
        let write = GatedAttentionQuery::QkNormRopeKvWrite {
            q_heads: 6,
            kv_heads: 1,
            head_dim: 256,
            rotary_dim: 64,
            theta_bits: 1e7_f32.to_bits(),
            mrope: [11, 11, 10],
            eps_bits: 1e-6_f32.to_bits(),
            page_size: 64,
            kv_dtype: DType::Bf16,
            capacity: 64,
        };
        let kv = ContextParallelQuery::OwnedQkNormRopeKvWrite { write, ranks: 2 };
        let positions = [0, 63, 64, 127, 128, 200, 300];
        assert_eq!(
            owned_kv_positions(kv, 0, &positions).unwrap(),
            vec![0, 63, -1, -1, 64, -1, 172]
        );
        assert_eq!(
            owned_kv_positions(kv, 1, &positions).unwrap(),
            vec![-1, -1, 0, 63, -1, 72, -1]
        );
        let keys = ContextParallelQuery::OwnedIndexerKeyWrite {
            write: SparseAttentionQuery::IndexerKeyWrite {
                head_dim: 128,
                rotary_dim: 64,
                theta_bits: 1e7_f32.to_bits(),
                mrope: [11, 11, 10],
                eps_bits: 1e-6_f32.to_bits(),
                block: 4,
                ring: 16,
                capacity: 64,
            },
            page: 64,
            ranks: 2,
        };
        // 16 blocks per page: block 37 is on page 2 (rank 0, stripe 1).
        assert_eq!(owned_block_row(keys, 0, 37).unwrap(), Some((1, 5)));
        assert_eq!(owned_block_row(keys, 1, 37).unwrap(), None);
        assert_eq!(owned_block_row(keys, 1, 16).unwrap(), Some((0, 0)));
        assert!(owned_block_row(keys, 2, 0).is_err());
        assert!(owned_kv_positions(keys, 0, &positions).is_err());
    }
}
