//! One attention core over explicit per-query key index lists. Causal,
//! windowed, paged, ring, indexer-selected and block-sparse attention are
//! index-list generators around it; the core assumes no layout or mask.
//!
//! Finite results follow the documented arithmetic bit for bit. A NaN result
//! has no guaranteed payload or sign: Rust leaves them unspecified, and
//! builds of the same source (an incremental and a fresh debug build) have
//! been seen to propagate different NaNs. The oracles ported onto this core
//! match their former copies in finite bits and NaN positions only.

use std::ops::Range;

use crate::contract::{DType, Refusal};
use crate::oracle::{OracleOutput, bf16_round};

/// Rows of `heads` x `dim` values, one key or value row per index. Index i
/// addresses the rows of all segments in order: the rows of segment s + 1
/// follow the last row of segment s.
#[derive(Clone, Copy, Debug)]
pub struct KvRows<'a> {
    pub segments: &'a [&'a [f64]],
    pub heads: usize,
    pub dim: usize,
}

impl KvRows<'_> {
    fn width(&self) -> usize {
        self.heads * self.dim
    }

    fn get(&self, mut row: usize, head: usize) -> Option<&[f64]> {
        let width = self.width();
        for segment in self.segments {
            let rows = segment.len() / width;
            if row < rows {
                let start = row * width + head * self.dim;
                return Some(&segment[start..start + self.dim]);
            }
            row -= rows;
        }
        None
    }
}

/// Where the values come from: their own rows, or the key rows themselves
/// (K = V, as in a shared latent).
#[derive(Clone, Copy, Debug)]
pub enum AttendValues<'a> {
    Rows(KvRows<'a>),
    SharedKeys,
}

/// Where the softmax denominator divides. `AfterSum` sums `weight * value`
/// and divides the sum; `Weights` divides each weight first and sums
/// `probability * value`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Normalize {
    AfterSum,
    Weights,
}

/// Rounding points: the q.k dot product, the logit (after scale, softcap
/// and bias), each probability (`Normalize::Weights` only) and each output.
/// All accept BF16 only (through F32, ties to even).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AttendRounding {
    pub dot: Option<DType>,
    pub logit: Option<DType>,
    pub probability: Option<DType>,
    pub output: Option<DType>,
}

/// A per-head logit added to the listed key rows (for example a learned
/// bias on each sequence's first key). `rows` is ascending.
#[derive(Clone, Copy, Debug)]
pub struct KeyBias<'a> {
    pub rows: &'a [usize],
    pub per_head: &'a [f64],
}

/// Every numerically relevant choice of the core, evaluated in f64 except
/// at the rounding points. Per query head h (reading KV head
/// `h / (q_heads / kv_heads)`) and listed key j:
/// `logit_j = dot(q, k_j) * scale`, then `cap * tanh(logit / cap)` with a
/// softcap, plus the key bias. `sink[h]`, when given, joins the max and the
/// denominator as one more logit with no value. Weights are
/// `exp(logit - max)`; dot products and value sums run in list order from
/// +0. S is the sum of `|weight * value|` scaled like the output, and K is
/// the listed key count plus one for a sink.
///
/// `logit_reduction_bound` adds the logits' own reduction error to S, for
/// outputs no storage rounding absorbs: with `n` listed keys, dot width `d`,
/// probabilities `p_j = weight_j / denominator` and the unrounded output `o`,
/// `(d / n) * sum_j p_j |v_j - o| * scale * sum_i |q_i k_ji|`.
#[derive(Clone, Copy, Debug)]
pub struct Attend<'a> {
    pub scale: f64,
    pub softcap: Option<f64>,
    pub sink: Option<&'a [f64]>,
    pub key_bias: Option<KeyBias<'a>>,
    pub normalize: Normalize,
    pub rounding: AttendRounding,
    pub logit_reduction_bound: bool,
}

impl Attend<'_> {
    /// Plain f64 attention: no softcap, sink or bias, divided after the sum.
    pub fn f64(scale: f64) -> Attend<'static> {
        Attend {
            scale,
            softcap: None,
            sink: None,
            key_bias: None,
            normalize: Normalize::AfterSum,
            rounding: AttendRounding::default(),
            logit_reduction_bound: false,
        }
    }

    /// One query row, q [q_heads, keys.dim], over the key rows in `list`
    /// (-1 entries are empty slots). Appends [q_heads, v_dim] outputs. A row
    /// with no listed key and no sink is refused; with a sink its output is
    /// zero.
    pub fn row(
        &self,
        q: &[f64],
        keys: KvRows<'_>,
        values: AttendValues<'_>,
        list: &[i64],
        output: &mut OracleOutput,
    ) -> Result<(), Refusal> {
        let values = match values {
            AttendValues::Rows(values) => values,
            AttendValues::SharedKeys => keys,
        };
        let bf16 = |point: Option<DType>| match point {
            None => Ok(false),
            Some(DType::Bf16) => Ok(true),
            Some(_) => Err(Refusal("attention rounding points are BF16 only")),
        };
        let rounding = self.rounding;
        let (round_dot, round_logit) = (bf16(rounding.dot)?, bf16(rounding.logit)?);
        let (round_probability, round_output) =
            (bf16(rounding.probability)?, bf16(rounding.output)?);
        let width = keys.width();
        if keys.heads == 0
            || keys.dim == 0
            || values.heads != keys.heads
            || values.dim == 0
            || q.is_empty()
            || !q.len().is_multiple_of(keys.dim)
            || !(q.len() / keys.dim).is_multiple_of(keys.heads)
            || keys.segments.iter().any(|s| !s.len().is_multiple_of(width))
            || values
                .segments
                .iter()
                .any(|s| !s.len().is_multiple_of(values.width()))
        {
            return Err(Refusal("attention core operand shape mismatch"));
        }
        let q_heads = q.len() / keys.dim;
        if !self.scale.is_finite()
            || self.scale <= 0.0
            || self
                .softcap
                .is_some_and(|cap| !cap.is_finite() || cap <= 0.0)
            || self.sink.is_some_and(|s| s.len() != q_heads)
            || self
                .key_bias
                .is_some_and(|b| b.per_head.len() != q_heads || !b.rows.is_sorted())
            || (round_probability && self.normalize != Normalize::Weights)
        {
            return Err(Refusal("attention core parameters invalid"));
        }
        let mut rows = Vec::with_capacity(list.len());
        for &index in list {
            match usize::try_from(index) {
                Ok(row) => rows.push(row),
                Err(_) if index == -1 => {}
                Err(_) => return Err(Refusal("attention key index below -1")),
            }
        }
        if rows.is_empty() && self.sink.is_none() {
            return Err(Refusal("attention row has no visible key"));
        }
        let round = |value: f64, on: bool| if on { bf16_round(value) } else { value };
        let group = q_heads / keys.heads;
        let mut value_rows = Vec::with_capacity(rows.len());
        for head in 0..q_heads {
            let kv_head = head / group;
            let q_row = &q[head * keys.dim..(head + 1) * keys.dim];
            let mut logits = Vec::with_capacity(rows.len());
            let mut magnitudes = Vec::new();
            value_rows.clear();
            for &row in &rows {
                let (Some(key), Some(value)) = (keys.get(row, kv_head), values.get(row, kv_head))
                else {
                    return Err(Refusal("attention key index outside the key rows"));
                };
                value_rows.push(value);
                let mut dot = 0.0;
                for (a, b) in q_row.iter().zip(key) {
                    dot += a * b;
                }
                if self.logit_reduction_bound {
                    let mut magnitude = 0.0;
                    for (a, b) in q_row.iter().zip(key) {
                        magnitude += (a * b).abs();
                    }
                    magnitudes.push(magnitude * self.scale);
                }
                let mut logit = round(dot, round_dot) * self.scale;
                if let Some(cap) = self.softcap {
                    logit = cap * (logit / cap).tanh();
                }
                if let Some(bias) = self.key_bias
                    && bias.rows.binary_search(&row).is_ok()
                {
                    logit += bias.per_head[head];
                }
                logits.push(round(logit, round_logit));
            }
            let sink = self.sink.map(|s| s[head]);
            let max = logits
                .iter()
                .copied()
                .chain(sink)
                .fold(f64::NEG_INFINITY, f64::max);
            let weights = logits.iter().map(|&v| (v - max).exp()).collect::<Vec<_>>();
            let mut denominator = weights.iter().sum::<f64>();
            if let Some(sink) = sink {
                denominator += (sink - max).exp();
            }
            let reduction = weights.len() + usize::from(sink.is_some());
            for d in 0..values.dim {
                let mut sum = 0.0;
                let mut abs_sum = 0.0;
                for (&weight, value) in weights.iter().zip(&value_rows) {
                    let product = match self.normalize {
                        Normalize::AfterSum => weight * value[d],
                        Normalize::Weights => {
                            round(weight / denominator, round_probability) * value[d]
                        }
                    };
                    sum += product;
                    abs_sum += product.abs();
                }
                let (value, mut bound) = match self.normalize {
                    Normalize::AfterSum => (sum / denominator, abs_sum / denominator),
                    Normalize::Weights => (sum, abs_sum),
                };
                if self.logit_reduction_bound && !rows.is_empty() {
                    let spread = weights
                        .iter()
                        .zip(&value_rows)
                        .zip(&magnitudes)
                        .map(|((&weight, v), &magnitude)| {
                            weight / denominator * (v[d] - value).abs() * magnitude
                        })
                        .sum::<f64>();
                    bound += spread * keys.dim as f64 / rows.len() as f64;
                }
                output.values.push(round(value, round_output));
                output.sum_abs_products.push(bound);
                output.reduction_lengths.push(reduction);
            }
        }
        Ok(())
    }
}

/// q is [lists.len(), q_heads, keys.dim]; list r holds row r's key indexes
/// (-1 = none). The output is [lists.len(), q_heads, v_dim].
pub fn attend(
    q: &[f64],
    keys: KvRows<'_>,
    values: AttendValues<'_>,
    lists: &[Vec<i64>],
    params: &Attend<'_>,
) -> Result<OracleOutput, Refusal> {
    if lists.is_empty() || !q.len().is_multiple_of(lists.len()) {
        return Err(Refusal("attention core operand shape mismatch"));
    }
    let width = q.len() / lists.len();
    let mut output = OracleOutput {
        values: Vec::new(),
        sum_abs_products: Vec::new(),
        reduction_lengths: Vec::new(),
    };
    for (row, list) in lists.iter().enumerate() {
        params.row(
            &q[row * width..(row + 1) * width],
            keys,
            values,
            list,
            &mut output,
        )?;
    }
    Ok(output)
}

/// Key positions visible to a query at `position` among a sequence's keys
/// `keys`: at or before it when `causal`, and at most `left` before and
/// `right` after it when given. A bidirectional block (DSpark style) is
/// `causal: false` over the context plus the block. The range may be empty.
pub fn window(
    position: usize,
    keys: Range<usize>,
    causal: bool,
    left: Option<usize>,
    right: Option<usize>,
) -> Range<usize> {
    let lo = left.map_or(keys.start, |l| keys.start.max(position.saturating_sub(l)));
    let mut hi = if causal {
        keys.end.min(position + 1)
    } else {
        keys.end
    };
    if let Some(r) = right {
        hi = hi.min(position.saturating_add(r).saturating_add(1));
    }
    lo..hi
}

/// Token positions of the kept blocks of `block` tokens, in the given block
/// order, then the incomplete tail `visible / block * block..visible`.
pub fn selected_blocks(kept: &[usize], block: usize, visible: usize) -> Vec<usize> {
    let mut tokens = kept
        .iter()
        .flat_map(|&b| b * block..(b + 1) * block)
        .collect::<Vec<_>>();
    tokens.extend(visible / block * block..visible);
    tokens
}

/// Rows of key positions through a page table: position p is row
/// `table[p / page_size] * page_size + p % page_size`. None when a position
/// has no table entry or its page is not below `pages`.
pub fn paged_rows(
    positions: impl IntoIterator<Item = usize>,
    table: &[usize],
    page_size: usize,
    pages: usize,
) -> Option<Vec<i64>> {
    positions
        .into_iter()
        .map(|p| {
            let page = *table.get(p / page_size).filter(|&&page| page < pages)?;
            Some((page * page_size + p % page_size) as i64)
        })
        .collect()
}
