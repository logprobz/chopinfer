//! F64 reference arithmetic for latent attention (`latent_attention`): the
//! latent cache write and its decoding, the attention over gathered key lists
//! in both output modes, the log-sum-exp merge of partials, and the
//! head-batched projections. Sums whose length matters (the 128-to-1024-wide
//! dot products, the key sums and the GEMM reductions) are
//! Neumaier-compensated.

use crate::codec::bf16_to_f32;
use crate::contract::{Refusal, ScaleMode, Validate};
use crate::latent_attention::{
    HeadBatchedGemmQuery, HeadWeightLayout, LATENT_FP8_SCALE_BLOCK, LatentAttentionOutput,
    LatentAttentionQuery, LatentKvFormat,
};
use crate::oracle::{NeumaierSum, OracleOutput, e4m3_table};
use crate::quantize::{self, Scales};

/// Where a sequence's tokens live: `block_tables` is `[sequences, max_pages]`
/// i32 page ids, and page p starts at byte `p * page_stride` of the pool (at
/// least one page's bytes, so other data may share a page allocation).
#[derive(Clone, Copy, Debug)]
pub struct LatentPaging<'a> {
    pub block_tables: &'a [i32],
    pub max_pages: usize,
    pub page_stride: usize,
}

/// The cache geometry an entry reads or writes.
#[derive(Clone, Copy, Debug)]
struct Layout {
    latent: usize,
    page_size: usize,
    kv: LatentKvFormat,
}

impl Layout {
    fn page_bytes(&self) -> usize {
        self.kv
            .page_bytes(self.latent as u32, self.page_size as u32) as usize
    }

    fn scale_blocks(&self) -> usize {
        self.latent / LATENT_FP8_SCALE_BLOCK as usize
    }

    /// The byte offset of the page holding `position` of `sequence`, and the
    /// position's slot in that page.
    fn locate(
        &self,
        paging: LatentPaging<'_>,
        pool_bytes: usize,
        sequence: i32,
        position: i32,
    ) -> Result<(usize, usize), Refusal> {
        let (Ok(sequence), Ok(position)) = (usize::try_from(sequence), usize::try_from(position))
        else {
            return Err(Refusal("latent sequence or position is negative"));
        };
        let column = position / self.page_size;
        if column >= paging.max_pages {
            return Err(Refusal("latent position past the page table"));
        }
        let page = paging
            .block_tables
            .get(sequence * paging.max_pages + column)
            .copied()
            .and_then(|page| usize::try_from(page).ok())
            .ok_or(Refusal("latent page table entry missing or negative"))?;
        let base = page
            .checked_mul(paging.page_stride)
            .filter(|&base| base + self.page_bytes() <= pool_bytes)
            .ok_or(Refusal("latent page outside the pool"))?;
        Ok((base, position % self.page_size))
    }

    /// Token `slot`'s decoded latent row of the page at `page`.
    fn decode(&self, page: &[u8], slot: usize) -> Vec<f64> {
        match self.kv {
            LatentKvFormat::Bf16 => page[slot * self.latent * 2..(slot + 1) * self.latent * 2]
                .chunks_exact(2)
                .map(|b| f64::from(bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))))
                .collect(),
            LatentKvFormat::Fp8E4M3Block128 => {
                let codes = &page[slot * self.latent..(slot + 1) * self.latent];
                let scales = self.page_size * self.latent + slot * self.scale_blocks() * 4;
                let table = e4m3_table();
                codes
                    .iter()
                    .enumerate()
                    .map(|(i, &code)| {
                        let at = scales + (i / LATENT_FP8_SCALE_BLOCK as usize) * 4;
                        let scale = f32::from_le_bytes(page[at..at + 4].try_into().unwrap());
                        // A 4-bit significand times a 24-bit one: exact in f64.
                        table[code as usize] * f64::from(scale)
                    })
                    .collect()
            }
        }
    }
}

fn kv_write_layout(query: LatentAttentionQuery) -> Result<(Layout, u32), Refusal> {
    query.validate()?;
    match query {
        LatentAttentionQuery::KvWrite {
            latent,
            page_size,
            kv,
            capacity,
        } => Ok((
            Layout {
                latent: latent as usize,
                page_size: page_size as usize,
                kv,
            },
            capacity,
        )),
        LatentAttentionQuery::Attention { .. } => Err(Refusal("expected a latent KV write query")),
    }
}

/// One encoded latent row: its value bytes (BF16 little endian, or E4M3
/// codes) and, for FP8, one F32 scale per 128 elements.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedLatent {
    pub values: Vec<u8>,
    pub scales: Vec<f32>,
}

/// Encodes one BF16 latent row (`latent` bit patterns) in `kv` format with
/// the contract's scale rule: per 128 elements, scale = max |x| / 448 in F32
/// (1 for an all-zero block) and code = `E4M3(x / scale)`, rounded to
/// nearest and saturated (`quantize::fp8_activation_1x128`).
pub fn encode_latent_row(kv: LatentKvFormat, latent: &[u16]) -> Result<EncodedLatent, Refusal> {
    let block = LATENT_FP8_SCALE_BLOCK as usize;
    if latent.is_empty() || !latent.len().is_multiple_of(block) {
        return Err(Refusal("latent row width must be a multiple of 128"));
    }
    match kv {
        LatentKvFormat::Bf16 => Ok(EncodedLatent {
            values: latent.iter().flat_map(|x| x.to_le_bytes()).collect(),
            scales: Vec::new(),
        }),
        LatentKvFormat::Fp8E4M3Block128 => {
            let values = latent.iter().map(|&x| bf16_to_f32(x)).collect::<Vec<_>>();
            let quantized =
                quantize::fp8_activation_1x128(&values, 1, latent.len(), ScaleMode::ExactF32)?;
            let Scales::F32(scales) = quantized.scales else {
                return Err(Refusal("FP8 latent scales must be F32"));
            };
            Ok(EncodedLatent {
                values: quantized.codes,
                scales,
            })
        }
    }
}

/// The latent KV write (`LatentAttentionQuery::KvWrite`): stores each row of
/// `latent` (`[rows, latent]` BF16 bit patterns) at `positions[row]` of
/// sequence `row_batch[row]`, in row order. A row with a negative position
/// writes nothing. Bit exact: FP8 rows follow [`encode_latent_row`].
pub fn latent_kv_write(
    query: LatentAttentionQuery,
    latent: &[u16],
    row_batch: &[i32],
    positions: &[i32],
    paging: LatentPaging<'_>,
    pool: &mut [u8],
) -> Result<(), Refusal> {
    let (layout, capacity) = kv_write_layout(query)?;
    let rows = positions.len();
    if rows == 0
        || rows > capacity as usize
        || row_batch.len() != rows
        || latent.len() != rows * layout.latent
        || paging.page_stride < layout.page_bytes()
    {
        return Err(Refusal("latent KV write operand shape mismatch"));
    }
    for row in 0..rows {
        if positions[row] < 0 {
            continue;
        }
        let (base, slot) = layout.locate(paging, pool.len(), row_batch[row], positions[row])?;
        let encoded = encode_latent_row(
            layout.kv,
            &latent[row * layout.latent..(row + 1) * layout.latent],
        )?;
        let width = encoded.values.len();
        pool[base + slot * width..base + (slot + 1) * width].copy_from_slice(&encoded.values);
        if layout.kv == LatentKvFormat::Fp8E4M3Block128 {
            let scales = base + layout.page_size * layout.latent + slot * layout.scale_blocks() * 4;
            for (i, scale) in encoded.scales.iter().enumerate() {
                pool[scales + 4 * i..scales + 4 * i + 4].copy_from_slice(&scale.to_le_bytes());
            }
        }
    }
    Ok(())
}

/// The operands of one latent attention launch.
#[derive(Clone, Copy, Debug)]
pub struct LatentAttentionInput<'a> {
    /// Absorbed queries `[rows, heads, latent]`, decoded from BF16.
    pub q: &'a [f64],
    /// Each row's sequence (its block-table row).
    pub row_batch: &'a [i32],
    /// Key lists `[rows, max_keys]` of token positions; -1 is a hole.
    pub key_lists: &'a [i32],
    /// Entries of each list the launch reads, at most `max_keys`.
    pub budget: usize,
    pub paging: LatentPaging<'a>,
    pub pool: &'a [u8],
}

/// The unnormalized partials of `LatentAttentionOutput::Partial`, per (row,
/// head) in row-major order. An empty key list has maximum -inf,
/// denominator 0 and a zero numerator, each with S = 0 and K = 1.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentPartials {
    /// `m = max_j s_j`. S is the maximizing key's `scale * sum_i |q_i c_i|`
    /// and K the latent width, the bound of one score's dot product.
    pub maximum: OracleOutput,
    /// `l = sum_j exp(s_j - m)`.
    pub denominator: OracleOutput,
    /// `o = sum_j exp(s_j - m) c_j`, `[rows, heads, latent]`.
    pub numerator: OracleOutput,
}

/// A latent attention launch's reference result, by output mode.
#[derive(Clone, Debug, PartialEq)]
pub enum LatentAttentionResult {
    /// `o / l` unrounded (the kernel's BF16 output is gated with half a BF16
    /// ulp), zeros for an empty list.
    Normalized(OracleOutput),
    Partial(LatentPartials),
}

/// One (row, head)'s f64 softmax over its keys: scores, weights relative
/// to the maximum, and each score's dot-product magnitude.
struct HeadSoftmax {
    maximum: f64,
    maximum_bound: f64,
    weights: Vec<f64>,
    magnitudes: Vec<f64>,
}

fn head_softmax(q: &[f64], keys: &[Vec<f64>], scale: f64) -> HeadSoftmax {
    let mut scores = Vec::with_capacity(keys.len());
    let mut magnitudes = Vec::with_capacity(keys.len());
    for key in keys {
        let (mut dot, mut magnitude) = (NeumaierSum::default(), NeumaierSum::default());
        for (&a, &b) in q.iter().zip(key) {
            dot.add(a * b);
            magnitude.add((a * b).abs());
        }
        scores.push(scale * dot.value());
        magnitudes.push(scale * magnitude.value());
    }
    // Ties keep the first key's bound; any tied key has the same score.
    let (maximum, maximum_bound) =
        scores
            .iter()
            .zip(&magnitudes)
            .fold((f64::NEG_INFINITY, 0.0), |(m, b), (&s, &mag)| {
                if s > m { (s, mag) } else { (m, b) }
            });
    let weights = scores.iter().map(|&s| (s - maximum).exp()).collect();
    HeadSoftmax {
        maximum,
        maximum_bound,
        weights,
        magnitudes,
    }
}

/// The latent attention reference (`LatentAttentionQuery::Attention`). Each
/// row reads entries `0..budget` of its key list; -1 entries are skipped and
/// a listed position must not repeat. Only the list decides visibility: a
/// row's own position is attended exactly when it is listed. Per head, `s_j
/// = scale * (q . c_j)` over the decoded latent, `m = max s_j`, `w_j =
/// exp(s_j - m)`, `l = sum w_j`, `o = sum w_j c_j`.
///
/// The bounds follow the attention core (`attend::Attend` with
/// `logit_reduction_bound`): with n listed keys, width d and score
/// magnitude `g_j = scale * sum_i |q_i c_ji|`, a normalized output's S is
/// `(sum_j w_j |c_j| + (d / n) sum_j w_j |c_j - o / l| g_j) / l` and K is n;
/// a numerator's S is `sum_j w_j |c_j| (1 + (d / n) g_j)` and the
/// denominator's `sum_j w_j (1 + (d / n) g_j)`.
pub fn latent_attention(
    query: LatentAttentionQuery,
    input: LatentAttentionInput<'_>,
) -> Result<LatentAttentionResult, Refusal> {
    query.validate()?;
    let LatentAttentionQuery::Attention {
        heads,
        latent,
        max_keys,
        softmax_scale_bits,
        page_size,
        kv,
        output,
        capacity,
        ..
    } = query
    else {
        return Err(Refusal("expected a latent attention query"));
    };
    let layout = Layout {
        latent: latent as usize,
        page_size: page_size as usize,
        kv,
    };
    let (heads, d, max_keys) = (heads as usize, latent as usize, max_keys as usize);
    let rows = input.row_batch.len();
    if rows == 0
        || rows > capacity as usize
        || input.q.len() != rows * heads * d
        || input.key_lists.len() != rows * max_keys
        || input.budget == 0
        || input.budget > max_keys
        || input.paging.page_stride < layout.page_bytes()
    {
        return Err(Refusal("latent attention operand shape mismatch"));
    }
    let scale = f64::from(f32::from_bits(softmax_scale_bits));
    let mut normalized = OracleOutput {
        values: Vec::new(),
        sum_abs_products: Vec::new(),
        reduction_lengths: Vec::new(),
    };
    let (mut maximum, mut denominator, mut numerator) =
        (normalized.clone(), normalized.clone(), normalized.clone());
    let push = |out: &mut OracleOutput, value: f64, bound: f64, k: usize| {
        out.values.push(value);
        out.sum_abs_products.push(bound);
        out.reduction_lengths.push(k.max(1));
    };
    for row in 0..rows {
        let list = &input.key_lists[row * max_keys..row * max_keys + input.budget];
        let mut positions = Vec::with_capacity(list.len());
        for &position in list {
            match position {
                -1 => {}
                p if p < 0 => return Err(Refusal("latent key list entry below -1")),
                p if positions.contains(&p) => {
                    return Err(Refusal("latent key list repeats a position"));
                }
                p => positions.push(p),
            }
        }
        let keys = positions
            .iter()
            .map(|&p| {
                let (base, slot) =
                    layout.locate(input.paging, input.pool.len(), input.row_batch[row], p)?;
                Ok(layout.decode(&input.pool[base..base + layout.page_bytes()], slot))
            })
            .collect::<Result<Vec<_>, Refusal>>()?;
        let n = keys.len();
        let spread = if n == 0 { 0.0 } else { d as f64 / n as f64 };
        for head in 0..heads {
            let q = &input.q[(row * heads + head) * d..(row * heads + head + 1) * d];
            let softmax = head_softmax(q, &keys, scale);
            let mut l = NeumaierSum::default();
            let mut l_bound = NeumaierSum::default();
            for (&w, &g) in softmax.weights.iter().zip(&softmax.magnitudes) {
                l.add(w);
                l_bound.add(w * (1.0 + spread * g));
            }
            let l = l.value();
            match output {
                LatentAttentionOutput::Partial => {
                    push(
                        &mut maximum,
                        softmax.maximum,
                        softmax.maximum_bound,
                        if n == 0 { 1 } else { d },
                    );
                    push(&mut denominator, l, l_bound.value(), n);
                }
                LatentAttentionOutput::Normalized => {}
            }
            for column in 0..d {
                let (mut o, mut abs, mut abs_spread) = (
                    NeumaierSum::default(),
                    NeumaierSum::default(),
                    NeumaierSum::default(),
                );
                for ((&w, key), &g) in softmax.weights.iter().zip(&keys).zip(&softmax.magnitudes) {
                    o.add(w * key[column]);
                    abs.add((w * key[column]).abs());
                    abs_spread.add((w * key[column]).abs() * g);
                }
                let o = o.value();
                match output {
                    LatentAttentionOutput::Partial => push(
                        &mut numerator,
                        o,
                        abs.value() + spread * abs_spread.value(),
                        n,
                    ),
                    LatentAttentionOutput::Normalized if n == 0 => {
                        push(&mut normalized, 0.0, 0.0, 1);
                    }
                    LatentAttentionOutput::Normalized => {
                        let value = o / l;
                        let mut centered = NeumaierSum::default();
                        for ((&w, key), &g) in
                            softmax.weights.iter().zip(&keys).zip(&softmax.magnitudes)
                        {
                            centered.add(w * (key[column] - value).abs() * g);
                        }
                        let bound = (abs.value() + spread * centered.value()) / l;
                        push(&mut normalized, value, bound, n);
                    }
                }
            }
        }
    }
    Ok(match output {
        LatentAttentionOutput::Normalized => LatentAttentionResult::Normalized(normalized),
        LatentAttentionOutput::Partial => LatentAttentionResult::Partial(LatentPartials {
            maximum,
            denominator,
            numerator,
        }),
    })
}

/// The f64 log-sum-exp merge of per-rank partials over disjoint key sets
/// (`context_parallel::ContextParallelQuery::PartialMerge`), in rank order:
/// `M = max_r m_r`, `a_r = exp(m_r - M)`, `L = sum_r a_r l_r`, `O = sum_r a_r
/// o_r`, output `O / L` (zeros when every rank is empty). S is `sum_r a_r
/// S_r / L` and K the ranks' summed key counts.
pub fn merge_partials(ranks: &[LatentPartials]) -> Result<OracleOutput, Refusal> {
    let Some(first) = ranks.first() else {
        return Err(Refusal("latent partial merge needs a rank"));
    };
    let pairs = first.maximum.values.len();
    let elements = first.numerator.values.len();
    if pairs == 0
        || !elements.is_multiple_of(pairs)
        || ranks.iter().any(|r| {
            r.maximum.values.len() != pairs
                || r.denominator.values.len() != pairs
                || r.numerator.values.len() != elements
        })
    {
        return Err(Refusal("latent partial ranks differ in shape"));
    }
    let width = elements / pairs;
    let mut merged = OracleOutput {
        values: Vec::with_capacity(elements),
        sum_abs_products: Vec::with_capacity(elements),
        reduction_lengths: Vec::with_capacity(elements),
    };
    for pair in 0..pairs {
        let top = ranks
            .iter()
            .map(|r| r.maximum.values[pair])
            .fold(f64::NEG_INFINITY, f64::max);
        let weights = ranks
            .iter()
            .map(|r| {
                let m = r.maximum.values[pair];
                if m == f64::NEG_INFINITY {
                    0.0
                } else {
                    (m - top).exp()
                }
            })
            .collect::<Vec<_>>();
        let mut l = NeumaierSum::default();
        for (rank, &a) in ranks.iter().zip(&weights) {
            l.add(a * rank.denominator.values[pair]);
        }
        let l = l.value();
        for column in 0..width {
            let index = pair * width + column;
            let (mut o, mut bound, mut keys) = (NeumaierSum::default(), NeumaierSum::default(), 0);
            for (rank, &a) in ranks.iter().zip(&weights) {
                o.add(a * rank.numerator.values[index]);
                bound.add(a * rank.numerator.sum_abs_products[index]);
                if a != 0.0 {
                    keys += rank.numerator.reduction_lengths[index];
                }
            }
            let (value, bound) = if top == f64::NEG_INFINITY {
                (0.0, 0.0)
            } else {
                (o.value() / l, bound.value() / l)
            };
            merged.values.push(value);
            merged.sum_abs_products.push(bound);
            merged.reduction_lengths.push(keys.max(1));
        }
    }
    Ok(merged)
}

/// The head-batched GEMM reference (`HeadBatchedGemmQuery`): x `[rows,
/// heads, k]`, head h's weight at element `h * head_stride` of `weights`
/// in the query's layout, output `[rows, heads, n]` unrounded (BF16 output
/// is gated with half a BF16 ulp) with S = `sum |x w|` and K = k.
pub fn head_batched_gemm(
    query: HeadBatchedGemmQuery,
    x: &[f64],
    weights: &[f64],
    head_stride: usize,
    rows: usize,
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let (heads, n, k) = (query.heads as usize, query.n as usize, query.k as usize);
    if rows == 0
        || rows > query.m_capacity as usize
        || x.len() != rows * heads * k
        || head_stride < n * k
        || weights.len() < (heads - 1) * head_stride + n * k
    {
        return Err(Refusal("head-batched GEMM operand shape mismatch"));
    }
    let mut values = Vec::with_capacity(rows * heads * n);
    let mut bounds = Vec::with_capacity(rows * heads * n);
    for row in 0..rows {
        for head in 0..heads {
            let input = &x[(row * heads + head) * k..(row * heads + head + 1) * k];
            let weight = &weights[head * head_stride..head * head_stride + n * k];
            for i in 0..n {
                let (mut sum, mut abs) = (NeumaierSum::default(), NeumaierSum::default());
                for (kk, &a) in input.iter().enumerate() {
                    let w = match query.layout {
                        HeadWeightLayout::NByK => weight[i * k + kk],
                        HeadWeightLayout::KByN => weight[kk * n + i],
                    };
                    sum.add(a * w);
                    abs.add((a * w).abs());
                }
                values.push(sum.value());
                bounds.push(abs.value());
            }
        }
    }
    OracleOutput::new(values, bounds, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::f32_to_bf16;
    use crate::contract::{AttentionPhase, DType};
    use crate::oracle::check_numeric;

    const LATENT: usize = 256;
    const PAGE: usize = 4;

    fn attention(kv: LatentKvFormat, output: LatentAttentionOutput) -> LatentAttentionQuery {
        LatentAttentionQuery::Attention {
            heads: 2,
            latent: LATENT as u32,
            max_keys: 8,
            softmax_scale_bits: 0.25_f32.to_bits(),
            page_size: PAGE as u32,
            kv,
            phase: AttentionPhase::Decode,
            output,
            sink: false,
            tail_rope_dims: 0,
            second_fp4_keys: false,
            capacity: 4,
        }
    }

    fn write_query(kv: LatentKvFormat) -> LatentAttentionQuery {
        LatentAttentionQuery::KvWrite {
            latent: LATENT as u32,
            page_size: PAGE as u32,
            kv,
            capacity: 64,
        }
    }

    /// A deterministic BF16 value in (-2, 2).
    fn value(seed: usize) -> u16 {
        let x = ((seed * 2_654_435_761) % 4001) as f32 / 1000.0 - 2.0;
        f32_to_bf16(x)
    }

    /// One sequence of `tokens` tokens written through a reversed page
    /// table, so token pages differ from their table columns.
    struct Cache {
        kv: LatentKvFormat,
        table: Vec<i32>,
        pool: Vec<u8>,
        stride: usize,
        latent: Vec<u16>,
    }

    impl Cache {
        fn new(kv: LatentKvFormat, tokens: usize) -> Cache {
            let pages = tokens.div_ceil(PAGE);
            let stride = kv.page_bytes(LATENT as u32, PAGE as u32) as usize + 64;
            let table = (0..pages as i32).rev().collect::<Vec<_>>();
            let latent = (0..tokens * LATENT).map(value).collect::<Vec<_>>();
            let mut cache = Cache {
                kv,
                table,
                pool: vec![0xa5; pages * stride],
                stride,
                latent,
            };
            let positions = (0..tokens as i32).collect::<Vec<_>>();
            let table = cache.table.clone();
            let paging = LatentPaging {
                block_tables: &table,
                max_pages: pages,
                page_stride: stride,
            };
            latent_kv_write(
                write_query(kv),
                &cache.latent,
                &vec![0; tokens],
                &positions,
                paging,
                &mut cache.pool,
            )
            .unwrap();
            cache
        }

        fn paging(&self) -> LatentPaging<'_> {
            LatentPaging {
                block_tables: &self.table,
                max_pages: self.table.len(),
                page_stride: self.stride,
            }
        }

        /// The decoded latent row at `position` of sequence 0: BF16 values,
        /// or `E4M3(code) * scale` exactly.
        fn decoded(&self, position: i32) -> Vec<f64> {
            let layout = Layout {
                latent: LATENT,
                page_size: PAGE,
                kv: self.kv,
            };
            let (base, slot) = layout
                .locate(self.paging(), self.pool.len(), 0, position)
                .unwrap();
            layout.decode(&self.pool[base..base + layout.page_bytes()], slot)
        }

        fn run(
            &self,
            output: LatentAttentionOutput,
            q: &[f64],
            lists: &[i32],
        ) -> LatentAttentionResult {
            let rows = lists.len() / 8;
            latent_attention(
                attention(self.kv, output),
                LatentAttentionInput {
                    q,
                    row_batch: &vec![0; rows],
                    key_lists: lists,
                    budget: 8,
                    paging: self.paging(),
                    pool: &self.pool,
                },
            )
            .unwrap()
        }
    }

    fn queries(rows: usize) -> Vec<f64> {
        (0..rows * 2 * LATENT)
            .map(|i| f64::from(bf16_to_f32(value(i + 7_777))))
            .collect()
    }

    fn normalized(result: LatentAttentionResult) -> OracleOutput {
        match result {
            LatentAttentionResult::Normalized(out) => out,
            LatentAttentionResult::Partial(_) => panic!("expected normalized output"),
        }
    }

    fn partials(result: LatentAttentionResult) -> LatentPartials {
        match result {
            LatentAttentionResult::Partial(out) => out,
            LatentAttentionResult::Normalized(_) => panic!("expected partial output"),
        }
    }

    #[test]
    fn fp8_encoding_follows_the_contract_scale_rule() {
        // amax 3.5 gives scale 3.5 / 448 = 2^-7 exactly, so every E4M3
        // multiple of it round trips; the zero block keeps scale 1.
        let mut row = vec![0_u16; 256];
        row[0] = f32_to_bf16(3.5);
        row[1] = f32_to_bf16(-1.75);
        row[2] = f32_to_bf16(0.0078125);
        row[3] = f32_to_bf16(-0.0);
        let encoded = encode_latent_row(LatentKvFormat::Fp8E4M3Block128, &row).unwrap();
        assert_eq!(encoded.scales, vec![2_f32.powi(-7), 1.0]);
        // 448 = 0x7e, -224 = 0xf6, 1 = 0x38, -0 = 0x80.
        assert_eq!(&encoded.values[..4], &[0x7e, 0xf6, 0x38, 0x80]);
        assert!(encoded.values[128..].iter().all(|&code| code == 0));

        // amax 1 gives the rounded F32 scale 1 / 448. 1 / scale rounds to the
        // code 448 (saturating if above it), which decodes to 448 * scale:
        // not 1, the FP8 rounding point the contract states.
        let mut row = vec![0_u16; 128];
        row[0] = f32_to_bf16(1.0);
        let encoded = encode_latent_row(LatentKvFormat::Fp8E4M3Block128, &row).unwrap();
        let scale = 1.0_f32 / 448.0;
        assert_eq!(encoded.scales, vec![scale]);
        assert_eq!(encoded.values[0], 0x7e);
        let decoded = e4m3_table()[0x7e] * f64::from(scale);
        assert_ne!(decoded, 1.0);
        assert!((decoded - 1.0).abs() < 2_f64.powi(-23));
    }

    #[test]
    fn fp8_pages_decode_exactly_and_keep_codes_before_scales() {
        let cache = Cache::new(LatentKvFormat::Fp8E4M3Block128, 6);
        assert_eq!(LatentKvFormat::Fp8E4M3Block128.token_bytes(256), 264);
        for position in 0..6 {
            let row = &cache.latent[position * LATENT..(position + 1) * LATENT];
            let encoded = encode_latent_row(LatentKvFormat::Fp8E4M3Block128, row).unwrap();
            let page = cache.table[position / PAGE] as usize * cache.stride;
            let slot = position % PAGE;
            assert_eq!(
                &cache.pool[page + slot * LATENT..page + (slot + 1) * LATENT],
                &encoded.values[..]
            );
            let scales = page + PAGE * LATENT + slot * 8;
            for (i, scale) in encoded.scales.iter().enumerate() {
                assert_eq!(
                    &cache.pool[scales + 4 * i..scales + 4 * i + 4],
                    &scale.to_le_bytes()
                );
            }
            // The decoded row is each code's exact value times its scale,
            // within half an E4M3 step of the BF16 input.
            let decoded = cache.decoded(position as i32);
            for (i, (&x, &bits)) in decoded.iter().zip(row).enumerate() {
                let scale = f64::from(encoded.scales[i / 128]);
                assert_eq!(x, e4m3_table()[encoded.values[i] as usize] * scale);
                let input = f64::from(bf16_to_f32(bits));
                assert!((x - input).abs() <= input.abs() / 16.0 + scale / 512.0 * 0.5 + 1e-30);
            }
        }
        // Bytes past each page and the unwritten slots keep their fill.
        let last = cache.table[1] as usize * cache.stride;
        assert!(
            cache.pool[last + 2 * LATENT..last + PAGE * LATENT]
                .iter()
                .all(|&b| b == 0xa5)
        );
    }

    #[test]
    fn bf16_pages_store_the_row_bits() {
        let cache = Cache::new(LatentKvFormat::Bf16, 5);
        for position in 0..5 {
            let row = &cache.latent[position * LATENT..(position + 1) * LATENT];
            let decoded = cache.decoded(position as i32);
            for (&x, &bits) in decoded.iter().zip(row) {
                assert_eq!(x, f64::from(bf16_to_f32(bits)));
            }
        }
    }

    #[test]
    fn hand_computed_two_key_softmax() {
        // Keys c0 = e0 and c1 = 2 e1 + e0, query q = e1 per head, scale 1/4:
        // s0 = 0, s1 = 1/2, so o = (c0 + e^(1/2) c1) / (1 + e^(1/2)).
        let mut pool = vec![0_u8; 2 * PAGE * LATENT];
        let mut put = |slot: usize, i: usize, x: f32| {
            pool[(slot * LATENT + i) * 2..(slot * LATENT + i) * 2 + 2]
                .copy_from_slice(&f32_to_bf16(x).to_le_bytes());
        };
        put(0, 0, 1.0);
        put(1, 0, 1.0);
        put(1, 1, 2.0);
        let mut q = vec![0.0; 2 * LATENT];
        q[1] = 1.0;
        q[LATENT + 1] = 1.0;
        let table = [0];
        let lists = [0, -1, 1, -1, -1, -1, -1, -1];
        let input = LatentAttentionInput {
            q: &q,
            row_batch: &[0],
            key_lists: &lists,
            budget: 8,
            paging: LatentPaging {
                block_tables: &table,
                max_pages: 1,
                page_stride: 2 * PAGE * LATENT,
            },
            pool: &pool,
        };
        let out = normalized(
            latent_attention(
                attention(LatentKvFormat::Bf16, LatentAttentionOutput::Normalized),
                input,
            )
            .unwrap(),
        );
        let e = 0.5_f64.exp();
        for head in 0..2 {
            let o = &out.values[head * LATENT..(head + 1) * LATENT];
            assert!((o[0] - 1.0).abs() < 1e-15);
            assert!((o[1] - 2.0 * e / (1.0 + e)).abs() < 1e-15);
            assert!(o[2..].iter().all(|&x| x == 0.0));
            assert_eq!(out.reduction_lengths[head * LATENT], 2);
        }
        let parts = partials(
            latent_attention(
                attention(LatentKvFormat::Bf16, LatentAttentionOutput::Partial),
                input,
            )
            .unwrap(),
        );
        assert_eq!(parts.maximum.values, vec![0.5, 0.5]);
        // The maximizing key's |q . c| is 2, times the scale.
        assert_eq!(parts.maximum.sum_abs_products, vec![0.5, 0.5]);
        assert!((parts.denominator.values[0] - (1.0 + (-0.5_f64).exp())).abs() < 1e-15);
        assert!((parts.numerator.values[1] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn holes_are_skipped_and_empty_lists_give_the_contract_values() {
        let cache = Cache::new(LatentKvFormat::Fp8E4M3Block128, 8);
        let q = queries(3);
        let lists = [
            -1, 3, -1, 5, -1, -1, 0, -1, // holes between keys
            3, 5, 0, -1, -1, -1, -1, -1, // the same keys, packed
            -1, -1, -1, -1, -1, -1, -1, -1, // empty
        ];
        let out = normalized(cache.run(LatentAttentionOutput::Normalized, &q, &lists));
        let width = 2 * LATENT;
        let reordered = normalized(cache.run(
            LatentAttentionOutput::Normalized,
            &[&q[width..2 * width], &q[width..2 * width]].concat(),
            &lists[..16],
        ));
        // Row 0 with row 1's query matches row 1: holes change nothing.
        assert_eq!(reordered.values[..width], reordered.values[width..]);
        assert_eq!(out.values[width..2 * width], reordered.values[width..]);
        assert!(out.values[2 * width..].iter().all(|&x| x == 0.0));
        assert!(out.sum_abs_products[2 * width..].iter().all(|&s| s == 0.0));

        let parts = partials(cache.run(LatentAttentionOutput::Partial, &q, &lists));
        assert_eq!(parts.maximum.values[4..], [f64::NEG_INFINITY; 2]);
        assert_eq!(parts.denominator.values[4..], [0.0; 2]);
        assert!(
            parts.numerator.values[2 * width..]
                .iter()
                .all(|&x| x == 0.0)
        );
    }

    #[test]
    fn the_own_position_is_attended_only_when_listed() {
        // A row at position 6 whose list holds pooled keys 0..4 and the
        // incomplete tail 4..=6 sees its own key; the same row whose own
        // pool completed and was not kept does not.
        let cache = Cache::new(LatentKvFormat::Bf16, 8);
        let q = queries(2);
        let lists = [
            0, 1, 2, 3, 4, 5, 6, -1, // own position in the tail
            0, 1, 2, 3, -1, -1, -1, -1, // own pool not selected
        ];
        let rows = [&q[..2 * LATENT], &q[..2 * LATENT]].concat();
        let out = normalized(cache.run(LatentAttentionOutput::Normalized, &rows, &lists));
        assert_eq!(out.reduction_lengths[0], 7);
        assert_eq!(out.reduction_lengths[2 * LATENT], 4);
        assert_ne!(out.values[..2 * LATENT], out.values[2 * LATENT..]);
        // Recomputing row 1 from its four keys alone gives the same output.
        let keys = (0..4).map(|p| cache.decoded(p)).collect::<Vec<_>>();
        let softmax = head_softmax(&rows[..LATENT], &keys, 0.25);
        let l = softmax.weights.iter().sum::<f64>();
        let o0 = softmax
            .weights
            .iter()
            .zip(&keys)
            .map(|(w, c)| w * c[0])
            .sum::<f64>()
            / l;
        assert!((out.values[2 * LATENT] - o0).abs() <= 1e-14 * o0.abs().max(1.0));
        // A listed position may not repeat.
        let repeated = [0, 1, 0, -1, -1, -1, -1, -1];
        assert_eq!(
            latent_attention(
                attention(LatentKvFormat::Bf16, LatentAttentionOutput::Normalized),
                LatentAttentionInput {
                    q: &q[..2 * LATENT],
                    row_batch: &[0],
                    key_lists: &repeated,
                    budget: 8,
                    paging: cache.paging(),
                    pool: &cache.pool,
                },
            ),
            Err(Refusal("latent key list repeats a position"))
        );
    }

    #[test]
    fn the_budget_limits_the_entries_read() {
        let cache = Cache::new(LatentKvFormat::Bf16, 8);
        let q = queries(1);
        let lists = [0, 1, 2, 3, 4, 5, 6, 7];
        let run = |budget| {
            normalized(
                latent_attention(
                    attention(LatentKvFormat::Bf16, LatentAttentionOutput::Normalized),
                    LatentAttentionInput {
                        q: &q,
                        row_batch: &[0],
                        key_lists: &lists,
                        budget,
                        paging: cache.paging(),
                        pool: &cache.pool,
                    },
                )
                .unwrap(),
            )
        };
        let short = run(3);
        let holes = normalized(cache.run(
            LatentAttentionOutput::Normalized,
            &q,
            &[0, 1, 2, -1, -1, -1, -1, -1],
        ));
        assert_eq!(short, holes);
        assert_eq!(run(8).reduction_lengths[0], 8);
    }

    #[test]
    fn partials_merged_over_disjoint_ranks_equal_the_full_softmax() {
        for kv in [LatentKvFormat::Bf16, LatentKvFormat::Fp8E4M3Block128] {
            let cache = Cache::new(kv, 8);
            let q = queries(1);
            let full = normalized(cache.run(
                LatentAttentionOutput::Normalized,
                &q,
                &[7, 0, 5, 2, 6, 1, 4, 3],
            ));
            let single = partials(cache.run(
                LatentAttentionOutput::Partial,
                &q,
                &[7, 0, 5, 2, 6, 1, 4, 3],
            ));
            // One rank: O / L is the normalized output.
            let one = merge_partials(std::slice::from_ref(&single)).unwrap();
            for (a, b) in one.values.iter().zip(&full.values) {
                assert!((a - b).abs() <= 1e-14 * b.abs().max(1e-3));
            }
            for split in [
                vec![vec![7, 0, 5, 2], vec![6, 1, 4, 3]],
                vec![vec![7, 0], vec![5, 2], vec![6, 1], vec![4, 3]],
                // A rank with no keys and an uneven split.
                vec![vec![7], vec![], vec![0, 5, 2, 6, 1], vec![4, 3]],
            ] {
                let ranks = split
                    .iter()
                    .map(|keys| {
                        let mut list = keys.clone();
                        list.resize(8, -1);
                        partials(cache.run(LatentAttentionOutput::Partial, &q, &list))
                    })
                    .collect::<Vec<_>>();
                let merged = merge_partials(&ranks).unwrap();
                for (index, (a, b)) in merged.values.iter().zip(&full.values).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-14 * b.abs().max(1e-3),
                        "{kv:?} {} ranks, element {index}: {a} vs {b}",
                        split.len()
                    );
                }
                assert_eq!(merged.reduction_lengths, full.reduction_lengths);
            }
        }
        // Every rank empty: zeros.
        let cache = Cache::new(LatentKvFormat::Bf16, 4);
        let empty = partials(cache.run(LatentAttentionOutput::Partial, &queries(1), &[-1; 8]));
        let merged = merge_partials(&[empty.clone(), empty]).unwrap();
        assert!(merged.values.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn rounded_outputs_pass_the_gate_and_perturbed_ones_fail() {
        let cache = Cache::new(LatentKvFormat::Fp8E4M3Block128, 8);
        let q = queries(1);
        let out = normalized(cache.run(
            LatentAttentionOutput::Normalized,
            &q,
            &[0, 1, 2, 3, 4, 5, 6, 7],
        ));
        let rounded = out
            .values
            .iter()
            .map(|&x| f64::from(bf16_to_f32(f32_to_bf16(x as f32))))
            .collect::<Vec<_>>();
        check_numeric(&rounded, &out, DType::Bf16).unwrap();
        let mut wrong = rounded.clone();
        wrong[3] += 0.01;
        assert_eq!(
            check_numeric(&wrong, &out, DType::Bf16).unwrap_err().index,
            3
        );
    }

    #[test]
    fn head_batched_gemm_reads_both_layouts_in_place() {
        // Two heads, each a key half [nope 2, latent 3] (`KByN`) followed by
        // a value half [v 2, latent 3] (`NByK`), embedded in zeros at the
        // contract's MMA-shaped n 8 and k 16 with a head stride of 256.
        let up = (1..=24).map(f64::from).collect::<Vec<_>>();
        let key = HeadBatchedGemmQuery {
            heads: 2,
            n: 8,
            k: 16,
            layout: HeadWeightLayout::KByN,
            output: DType::Bf16,
            m_capacity: 4,
        };
        let mut padded = vec![0.0; 2 * 2 * 8 * 16];
        for head in 0..2 {
            for r in 0..2 {
                for c in 0..3 {
                    // KByN: W[k][n] at k * n + i.
                    padded[head * 256 + r * 8 + c] = up[head * 12 + r * 3 + c];
                    // NByK value half at offset 128: W[n][k] at i * k + kk.
                    padded[head * 256 + 128 + r * 16 + c] = up[head * 12 + 6 + r * 3 + c];
                }
            }
        }
        let mut x = vec![0.0; 2 * 16];
        x[0] = 1.0;
        x[1] = -2.0;
        x[16] = 0.5;
        x[17] = 1.0;
        let y = head_batched_gemm(key, &x, &padded, 256, 1).unwrap();
        // Head 0: q_abs = W_UK^T q = 1 * [1, 2, 3] - 2 * [4, 5, 6].
        assert_eq!(y.values[..3], [-7.0, -8.0, -9.0]);
        // Head 1: 0.5 * [13, 14, 15] + [16, 17, 18].
        assert_eq!(y.values[8..11], [22.5, 24.0, 25.5]);
        assert_eq!(y.sum_abs_products[0], 9.0);
        assert_eq!(y.reduction_lengths[0], 16);

        let value = HeadBatchedGemmQuery {
            n: 8,
            k: 16,
            layout: HeadWeightLayout::NByK,
            ..key
        };
        let mut o = vec![0.0; 2 * 16];
        o[..3].copy_from_slice(&[1.0, 0.0, -1.0]);
        o[16..19].copy_from_slice(&[0.0, 2.0, 0.0]);
        let v = head_batched_gemm(value, &o, &padded[128..], 256, 1).unwrap();
        // Head 0: W_UV o = [7 - 9, 10 - 12]; head 1: 2 * [20, 23].
        assert_eq!(v.values[..2], [-2.0, -2.0]);
        assert_eq!(v.values[8..10], [40.0, 46.0]);
        assert_eq!(
            head_batched_gemm(value, &o, &padded[128..], 100, 1),
            Err(Refusal("head-batched GEMM operand shape mismatch"))
        );
    }
}
