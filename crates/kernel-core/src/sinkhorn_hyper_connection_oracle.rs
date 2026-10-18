//! F64 reference arithmetic for Sinkhorn-mixed hyper-connections
//! (`sinkhorn_hyper_connection`), and the exact F32 forms of the BF16 expand
//! and stream head.
//!
//! `MixCollapse` is checked within the decision-0081 gate. Its oracle splits
//! at the projection: [`projection`] and [`rstd`] are the long reductions
//! (K = N * H), and [`mix_collapse_from_logits`] evaluates everything after
//! them per token from given logits and logit error bounds, with bounds that
//! enclose every kernel result over the whole logit interval (monotone
//! sigmoid ranges, not a linearization). Twenty Sinkhorn iterations have no
//! useful finite enclosure, so the comb is checked by its parts:
//! [`mix_stage_from_kernel_logits`] bounds a kernel's softmax cells from its
//! own F32 logits, and [`sinkhorn_f32`] reproduces its normalizations from
//! those cells bit for bit.
//! `Expand` and `Head` are exact functions of their BF16 operands, so their
//! references are the contract's F32 arithmetic, compared bit for bit.
//!
//! The lagged placement's mix-collapse ([`lagged_mix_collapse`]) bounds the
//! same `pre`, `post` and `comb` stored in F32, and its collapsed row is an
//! exact function of the given `pre_in` ([`collapse`]). `ExpandF32` and the
//! sigmoid-weighted head are exact functions of their BF16 and F32 operands
//! ([`expand_f32`], [`weighted_head`], [`weighted_head_f32`]); [`expand_f32_reference`] and
//! [`weighted_head_reference`] are their f64 forms, which the exact forms
//! lie within.
//! `MixCollapseNorm`'s mix-collapse outputs are `MixCollapse`'s, and its
//! normed row is an exact function of the stored collapsed row
//! ([`input_norm`]), as is `AllReduceExpandMixNorm`'s in the order its
//! query names ([`epilogue_norm`]).

use crate::codec::{bf16_to_f32, f32_to_bf16};
use crate::contract::{Refusal, Validate};
use crate::oracle::{OracleOutput, bf16_round, compensated_sum, compensated_sum_with_magnitude};
use crate::rmsnorm::RmsNormReduction;
use crate::sinkhorn_hyper_connection::{
    HyperConnectionPlacement, SinkhornHyperConnectionQuery, SinkhornMix, SinkhornOrder,
    StreamHeadMode,
};
use std::ops::{Add, Div};

/// The F32 unit roundoff.
const U: f64 = 1.0 / 16_777_216.0;
/// The gate's per-term factor, 2^-22.
const GATE: f64 = 1.0 / 4_194_304.0;
/// Relative error of one F32 `1 / (1 + expf(-v))`: `expf` within 2 ulp
/// (4u), then one rounding each for the add and the division (6u in all),
/// with margin.
const SIGMOID_ROUNDING: f64 = 8.0 * U;
/// Absolute slack for subnormal exponentials, 2^-140.
const TINY: f64 = 7.174_648_137_343_064e-43;

/// The learned per-sublayer parameters, decoded: `projection` is `fn`
/// `[(2 + N) * N, N * H]`, `base` `[(2 + N) * N]` and `scale` `[3]`.
#[derive(Clone, Copy, Debug)]
pub struct MixWeights<'a> {
    pub projection: &'a [f64],
    pub base: &'a [f64],
    pub scale: &'a [f64],
}

/// The per-token outputs of `MixCollapse`, unrounded: the gate adds half a
/// BF16 ulp for the single rounding of each stored value.
#[derive(Clone, Debug, PartialEq)]
pub struct MixCollapseOutput {
    /// `[rows, H]`.
    pub collapsed: OracleOutput,
    /// `[rows, N]`.
    pub post: OracleOutput,
    /// `[rows, N, N]`, `comb[i][j]` at `i * N + j`.
    pub comb: OracleOutput,
}

/// The per-token outputs of a lagged `MixCollapse`: `collapsed` unrounded
/// (the gate adds half a BF16 ulp for its rounding), and this mixer's `pre`
/// `[rows, N]`, `post` `[rows, N]` and `comb` `[rows, N, N]`, which the
/// kernel stores in F32, with bounds that enclose every F32 kernel result
/// (check them as F32 outputs).
#[derive(Clone, Debug, PartialEq)]
pub struct LaggedMixCollapseOutput {
    pub collapsed: OracleOutput,
    pub pre: OracleOutput,
    pub post: OracleOutput,
    pub comb: OracleOutput,
}

/// [`lagged_mix_stage_from_kernel_logits`]: `pre` and `post` as in
/// [`LaggedMixCollapseOutput`], and the F32 Sinkhorn input `cells`
/// `[rows, N, N]`.
#[derive(Clone, Debug, PartialEq)]
pub struct LaggedMixStageOutput {
    pub pre: OracleOutput,
    pub post: OracleOutput,
    pub cells: OracleOutput,
}

/// [`mix_stage_from_kernel_logits`]: `collapsed` and `post` as in
/// [`MixCollapseOutput`], and the F32 Sinkhorn input `cells` `[rows, N, N]`.
#[derive(Clone, Debug, PartialEq)]
pub struct MixStageOutput {
    pub collapsed: OracleOutput,
    pub post: OracleOutput,
    pub cells: OracleOutput,
}

/// The mixer and capacity of a mix-collapse, alone or before its input
/// norm (`MixCollapseNorm`, whose mix-collapse outputs are `MixCollapse`'s).
fn mix_of(query: SinkhornHyperConnectionQuery) -> Result<(SinkhornMix, u32), Refusal> {
    query.validate()?;
    match query {
        SinkhornHyperConnectionQuery::MixCollapse { mix, capacity }
        | SinkhornHyperConnectionQuery::MixCollapseNorm { mix, capacity, .. } => {
            Ok((mix, capacity))
        }
        _ => Err(Refusal("expected a Sinkhorn mix-collapse query")),
    }
}

/// [`mix_of`] for a mixer of `placement`.
fn placed_mix_of(
    query: SinkhornHyperConnectionQuery,
    placement: HyperConnectionPlacement,
) -> Result<(SinkhornMix, u32), Refusal> {
    let (mix, capacity) = mix_of(query)?;
    if mix.placement != placement {
        return Err(Refusal(match placement {
            HyperConnectionPlacement::PerSublayer => {
                "expected a per-sublayer Sinkhorn mix-collapse query"
            }
            HyperConnectionPlacement::LaggedSinglePass => {
                "expected a lagged Sinkhorn mix-collapse query"
            }
        }));
    }
    Ok((mix, capacity))
}

fn rows_of(len: usize, width: usize, capacity: u32) -> Result<usize, Refusal> {
    if len == 0 || !len.is_multiple_of(width) || len / width > capacity as usize {
        return Err(Refusal("Sinkhorn hyper-connection operand shape mismatch"));
    }
    Ok(len / width)
}

fn check_weights(mix: SinkhornMix, weights: MixWeights<'_>) -> Result<(), Refusal> {
    let mixes = mix.mixes() as usize;
    if weights.projection.len() != mixes * mix.flat_width() as usize
        || weights.base.len() != mixes
        || weights.scale.len() != 3
    {
        return Err(Refusal("Sinkhorn mixer parameter shape mismatch"));
    }
    Ok(())
}

/// The raw projection `sum_k fn[m][k] * x_k` over each row's flat streams,
/// before the RMS scale: `[rows, (2 + N) * N]` with K = N * H and S the sum
/// of absolute products. `x` is `[rows, N, H]`.
pub fn projection(
    query: SinkhornHyperConnectionQuery,
    x: &[f64],
    weights: MixWeights<'_>,
) -> Result<OracleOutput, Refusal> {
    let (mix, capacity) = mix_of(query)?;
    check_weights(mix, weights)?;
    let k = mix.flat_width() as usize;
    let rows = rows_of(x.len(), k, capacity)?;
    let mixes = mix.mixes() as usize;
    let mut values = Vec::with_capacity(rows * mixes);
    let mut sums = Vec::with_capacity(rows * mixes);
    for row in x.chunks_exact(k).take(rows) {
        for weight in weights.projection.chunks_exact(k) {
            let (value, sum) =
                compensated_sum_with_magnitude(weight.iter().zip(row).map(|(w, x)| w * x));
            values.push(value);
            sums.push(sum);
        }
    }
    OracleOutput::new(values, sums, k)
}

/// `rsqrt(sum_k x_k^2 / (N * H) + norm_eps)` per row, `[rows]`, with K = 1
/// and S set so that 2^-22 S bounds an F32 evaluation: the sum of squares
/// within the gate (K = N * H) halves under the square root, plus one
/// rounding each for the mean and the correctly rounded `rsqrt`.
pub fn rstd(query: SinkhornHyperConnectionQuery, x: &[f64]) -> Result<OracleOutput, Refusal> {
    let (mix, capacity) = mix_of(query)?;
    let k = mix.flat_width() as usize;
    let rows = rows_of(x.len(), k, capacity)?;
    let eps = f64::from(f32::from_bits(mix.norm_eps_bits));
    let mut values = Vec::with_capacity(rows);
    let mut bounds = Vec::with_capacity(rows);
    for row in x.chunks_exact(k).take(rows) {
        let squares = compensated_sum(row.iter().map(|x| x * x));
        let mean = squares / k as f64;
        let value = 1.0 / (mean + eps).sqrt();
        values.push(value);
        let relative = 0.5 * GATE * k as f64 * mean / (mean + eps) + 2.0 * U;
        bounds.push(value * relative / GATE);
    }
    OracleOutput::new(values, bounds, 1)
}

/// Sinkhorn of the `n x n` logits `c` (row-major) under `order`, in f64:
/// the row softmax (max subtracted) plus `eps`, then `iterations`
/// normalizations as `SinkhornOrder` defines them. Returns the matrix.
pub fn sinkhorn(order: SinkhornOrder, c: &[f64], n: usize, iterations: u32, eps: f64) -> Vec<f64> {
    assert_eq!(c.len(), n * n, "Sinkhorn input must be n x n");
    let mut cells = Vec::with_capacity(n * n);
    for row in c.chunks_exact(n) {
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp = row.iter().map(|&v| (v - max).exp()).collect::<Vec<_>>();
        let total = exp.iter().sum::<f64>();
        cells.extend(exp.iter().map(|e| e / total + eps));
    }
    normalize(order, &mut cells, n, iterations, eps);
    cells
}

/// A kernel's Sinkhorn normalizations from its softmax cells (the row
/// softmax plus eps), exactly: every line sum in F32 from zero in index
/// order, then `cell / (sum + eps)`. All of it is IEEE addition and
/// division, so a kernel's comb must equal this bit for bit.
pub fn sinkhorn_f32(
    order: SinkhornOrder,
    cells: &[f32],
    n: usize,
    iterations: u32,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(cells.len(), n * n, "Sinkhorn input must be n x n");
    let mut cells = cells.to_vec();
    normalize(order, &mut cells, n, iterations, eps);
    cells
}

/// The normalizations of `order` over the softmax cells, in place.
fn normalize<T>(order: SinkhornOrder, cells: &mut [T], n: usize, iterations: u32, eps: T)
where
    T: Copy + Default + Add<Output = T> + Div<Output = T>,
{
    let SinkhornOrder::RowSoftmaxThenColumnFirst = order;
    let step = |cells: &mut [T], rows: bool| {
        let published = cells.to_vec();
        for i in 0..n {
            for j in 0..n {
                let mut sum = T::default();
                for k in 0..n {
                    sum = sum + published[if rows { i * n + k } else { k * n + j }];
                }
                cells[i * n + j] = published[i * n + j] / (sum + eps);
            }
        }
    };
    if iterations > 0 {
        step(cells, false);
    }
    for _ in 1..iterations {
        step(cells, true);
        step(cells, false);
    }
}

fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// Half a BF16 ulp of `value`, as the gate counts it.
fn half_bf16_ulp(value: f64) -> f64 {
    if value == 0.0 || value.abs() < 2_f64.powi(-126) {
        2_f64.powi(-134)
    } else {
        2_f64.powi((value.abs().log2().floor() as i32).max(-126) - 8)
    }
}

/// The bound of a BF16-stored output whose F32 value lies in `[lo, hi]`,
/// against the reference `value`: the farther endpoint, plus the larger
/// rounding of an endpoint in a higher binade than the reference (the gate
/// adds half a BF16 ulp of the reference only).
fn stored_bound(value: f64, lo: f64, hi: f64) -> f64 {
    let far = (value - lo).max(hi - value).max(0.0);
    far + (half_bf16_ulp(lo.abs().max(hi.abs())) - half_bf16_ulp(value)).max(0.0) + TINY
}

/// A kernel's `fma(l, s, b)` for every logit `l` within `error` of `l0`:
/// the reference input and the range of the F32 result. With zero error and
/// F32 operands it is the F32 FMA itself, exactly; otherwise the exact
/// affine range widened by one rounding.
fn affine(l0: f64, error: f64, s: f64, b: f64) -> (f64, f64, f64) {
    let f32_exact = |v: f64| f64::from(v as f32) == v;
    if error == 0.0 && f32_exact(l0) && f32_exact(s) && f32_exact(b) {
        let v = f64::from((l0 as f32).mul_add(s as f32, b as f32));
        return (v, v, v);
    }
    let (a, c) = ((l0 - error) * s + b, (l0 + error) * s + b);
    let widen = 1.01 * U * a.abs().max(c.abs());
    (l0 * s + b, a.min(c) - widen, a.max(c) + widen)
}

/// A kernel's F32 `sigmoid` over `[lo, hi]` (monotone), widened by its
/// rounding: the reference and the range.
fn sigmoid_range((v, lo, hi): (f64, f64, f64)) -> (f64, f64, f64) {
    (
        sigmoid(v),
        sigmoid(lo) * (1.0 - SIGMOID_ROUNDING),
        sigmoid(hi) * (1.0 + SIGMOID_ROUNDING) + TINY,
    )
}

/// The bound of an F32-stored output whose value lies in `[lo, hi]`,
/// against the reference `value`: the farther endpoint (the gate adds no
/// rounding for F32 outputs).
fn f32_bound(value: f64, lo: f64, hi: f64) -> f64 {
    (value - lo).max(hi - value).max(0.0) + TINY
}

/// One token's mixer outputs from its logits `l` within errors `e`, each
/// as (reference, low, high) enclosing the kernel's F32 value: `pre` (with
/// `+ eps` and its rounding), `post`, and the f64 Sinkhorn of the comb.
struct RowStage {
    pre: Vec<(f64, f64, f64)>,
    post: Vec<(f64, f64, f64)>,
    comb: Vec<f64>,
}

fn row_stage(mix: SinkhornMix, l: &[f64], e: &[f64], base: &[f64], scale: &[f64]) -> RowStage {
    let n = mix.streams as usize;
    let eps = f64::from(f32::from_bits(mix.eps_bits));
    let range = |m: usize, s: f64| affine(l[m], e[m], s, base[m]);
    let mut pre = Vec::with_capacity(n);
    let mut post = Vec::with_capacity(n);
    for i in 0..n {
        let (s, lo, hi) = sigmoid_range(range(i, scale[0]));
        pre.push((s + eps, (lo + eps) * (1.0 - U), (hi + eps) * (1.0 + U)));
        let (p, lo, hi) = sigmoid_range(range(n + i, scale[1]));
        post.push((2.0 * p, 2.0 * lo, 2.0 * hi));
    }
    let c = (0..n * n)
        .map(|k| range(2 * n + k, scale[2]).0)
        .collect::<Vec<_>>();
    let comb = sinkhorn(mix.order, &c, n, mix.iterations, eps);
    RowStage { pre, post, comb }
}

/// Column `col` of `sum_i w_i * X_i` over one token's streams `[N, H]`,
/// for weights given as (reference, low, high): the reference value and
/// the distance `far` that encloses an F32 FMA chain from zero in stream
/// order for any weights in their ranges (the weight spread plus `N`
/// roundings of the sum of absolute terms).
fn weighted_column(
    weights: &[(f64, f64, f64)],
    streams: &[f64],
    h: usize,
    col: usize,
) -> (f64, f64) {
    let (mut value, mut spread, mut absolute) = (0.0, 0.0, 0.0);
    for (i, &(p, lo, hi)) in weights.iter().enumerate() {
        let xi = streams[i * h + col];
        value += p * xi;
        spread += (p - lo).max(hi - p) * xi.abs();
        absolute += lo.abs().max(hi.abs()) * xi.abs();
    }
    (value, spread + 1.01 * weights.len() as f64 * U * absolute)
}

/// Checks the logits, their errors and the affine parameters of `rows`.
fn check_logits(
    mix: SinkhornMix,
    rows: usize,
    logits: &[f64],
    logit_errors: &[f64],
    base: &[f64],
    scale: &[f64],
) -> Result<(), Refusal> {
    let mixes = mix.mixes() as usize;
    if logits.len() != rows * mixes
        || logit_errors.len() != logits.len()
        || base.len() != mixes
        || scale.len() != 3
    {
        return Err(Refusal("Sinkhorn mixer logit shape mismatch"));
    }
    Ok(())
}

/// `MixCollapse` from given logits `[rows, (2 + N) * N]` (already scaled by
/// rstd) with per-logit error bounds `logit_errors` of the same shape.
/// `x` is `[rows, N, H]`. Each output's bound (K = 1, S = bound / 2^-22)
/// encloses every kernel result for logits within their errors:
///
/// - the affine `fma(l, scale, base)` over the whole logit interval plus one
///   rounding, or the F32 FMA itself exactly for zero error;
/// - `pre` and `post` through the monotone sigmoid at the interval's ends,
///   widened by `SIGMOID_ROUNDING`, plus one rounding for `+ eps`;
/// - the collapse through those `pre` ranges plus `N` roundings of its sum
///   of absolute terms;
/// - for BF16 outputs, the rounding of an endpoint in a higher binade.
///
/// `comb` gets only its range, `[0, 1]`: no useful finite enclosure of 20
/// Sinkhorn iterations is known (interval propagation loses the
/// contraction and saturates), so this bound does not discriminate. Check a
/// kernel's comb by its parts instead: its softmax cells against
/// [`mix_stage_from_kernel_logits`] and its normalizations bit for bit
/// against [`sinkhorn_f32`].
///
/// The mixer's placement is `PerSublayer`; a lagged mixer's outputs are
/// [`lagged_mix_collapse_from_logits`]'s.
pub fn mix_collapse_from_logits(
    query: SinkhornHyperConnectionQuery,
    x: &[f64],
    logits: &[f64],
    logit_errors: &[f64],
    base: &[f64],
    scale: &[f64],
) -> Result<MixCollapseOutput, Refusal> {
    let (mix, capacity) = placed_mix_of(query, HyperConnectionPlacement::PerSublayer)?;
    let (n, h) = (mix.streams as usize, mix.width as usize);
    let mixes = mix.mixes() as usize;
    let rows = rows_of(x.len(), n * h, capacity)?;
    check_logits(mix, rows, logits, logit_errors, base, scale)?;
    let nn = n * n;
    let mut collapsed = (Vec::with_capacity(rows * h), Vec::with_capacity(rows * h));
    let mut post = (Vec::with_capacity(rows * n), Vec::with_capacity(rows * n));
    let mut comb = (Vec::with_capacity(rows * nn), Vec::with_capacity(rows * nn));
    for row in 0..rows {
        let l = &logits[row * mixes..][..mixes];
        let e = &logit_errors[row * mixes..][..mixes];
        let stage = row_stage(mix, l, e, base, scale);
        for &(p, lo, hi) in &stage.post {
            post.0.push(p);
            post.1.push(stored_bound(p, lo, hi) / GATE);
        }
        for value in stage.comb {
            comb.0.push(value);
            comb.1.push(stored_bound(value, 0.0, 1.0) / GATE);
        }
        let streams = &x[row * n * h..][..n * h];
        for col in 0..h {
            let (value, far) = weighted_column(&stage.pre, streams, h, col);
            collapsed.0.push(value);
            collapsed
                .1
                .push(stored_bound(value, value - far, value + far) / GATE);
        }
    }
    Ok(MixCollapseOutput {
        collapsed: OracleOutput::new(collapsed.0, collapsed.1, 1)?,
        post: OracleOutput::new(post.0, post.1, 1)?,
        comb: OracleOutput::new(comb.0, comb.1, 1)?,
    })
}

/// A lagged `MixCollapse` from given logits and errors, as
/// [`mix_collapse_from_logits`] bounds them, and the F32 collapse weights
/// `pre_in` `[rows, N]` the launch reads (the previous sublayer's stored
/// `pre`, or [`one_hot_pre`] before the first sublayer). `pre`, `post` and
/// `comb` are stored in F32, so their bounds hold the ranges alone (`comb`
/// its range `[0, 1]`, checked by its parts as in the per-sublayer case,
/// through [`lagged_mix_stage_from_kernel_logits`] and [`sinkhorn_f32`]).
/// `collapsed` is `sum_i pre_in_i * X_i` within `N` roundings of its sum of
/// absolute terms plus its BF16 rounding; it is also an exact function of
/// `pre_in` ([`collapse`]).
pub fn lagged_mix_collapse_from_logits(
    query: SinkhornHyperConnectionQuery,
    x: &[f64],
    pre_in: &[f32],
    logits: &[f64],
    logit_errors: &[f64],
    base: &[f64],
    scale: &[f64],
) -> Result<LaggedMixCollapseOutput, Refusal> {
    let (mix, capacity) = placed_mix_of(query, HyperConnectionPlacement::LaggedSinglePass)?;
    let (n, h) = (mix.streams as usize, mix.width as usize);
    let mixes = mix.mixes() as usize;
    let rows = rows_of(x.len(), n * h, capacity)?;
    check_logits(mix, rows, logits, logit_errors, base, scale)?;
    if pre_in.len() != rows * n {
        return Err(Refusal("Sinkhorn collapse weight shape mismatch"));
    }
    let nn = n * n;
    let mut collapsed = (Vec::with_capacity(rows * h), Vec::with_capacity(rows * h));
    let mut pre = (Vec::with_capacity(rows * n), Vec::with_capacity(rows * n));
    let mut post = (Vec::with_capacity(rows * n), Vec::with_capacity(rows * n));
    let mut comb = (Vec::with_capacity(rows * nn), Vec::with_capacity(rows * nn));
    for row in 0..rows {
        let l = &logits[row * mixes..][..mixes];
        let e = &logit_errors[row * mixes..][..mixes];
        let stage = row_stage(mix, l, e, base, scale);
        for (values, out) in [(&stage.pre, &mut pre), (&stage.post, &mut post)] {
            for &(p, lo, hi) in values {
                out.0.push(p);
                out.1.push(f32_bound(p, lo, hi) / GATE);
            }
        }
        for value in stage.comb {
            comb.0.push(value);
            comb.1.push(f32_bound(value, 0.0, 1.0) / GATE);
        }
        let given = given_weights(&pre_in[row * n..][..n]);
        let streams = &x[row * n * h..][..n * h];
        for col in 0..h {
            let (value, far) = weighted_column(&given, streams, h, col);
            collapsed.0.push(value);
            collapsed
                .1
                .push(stored_bound(value, value - far, value + far) / GATE);
        }
    }
    Ok(LaggedMixCollapseOutput {
        collapsed: OracleOutput::new(collapsed.0, collapsed.1, 1)?,
        pre: OracleOutput::new(pre.0, pre.1, 1)?,
        post: OracleOutput::new(post.0, post.1, 1)?,
        comb: OracleOutput::new(comb.0, comb.1, 1)?,
    })
}

/// Exact F32 weights as (reference, low, high).
fn given_weights(weights: &[f32]) -> Vec<(f64, f64, f64)> {
    weights
        .iter()
        .map(|&w| {
            let w = f64::from(w);
            (w, w, w)
        })
        .collect()
}

/// The collapse weights before the first sublayer of a lagged stack: the
/// one-hot `(1, 0, ..., 0)` per row, `[rows, N]` F32.
pub fn one_hot_pre(rows: usize, streams: usize) -> Vec<f32> {
    let mut pre = vec![0.0; rows * streams];
    for row in pre.chunks_exact_mut(streams.max(1)) {
        row[0] = 1.0;
    }
    pre
}

/// The nonlinear stage of `MixCollapse` from a kernel's own F32 logits
/// `[rows, (2 + N) * N]` (its `projected * rstd`), with `base` and `scale`
/// as F32. Every affine FMA, the row maximum and the subtraction before
/// `expf` are emulated exactly, so the bounds hold only the roundings of the
/// exponentials, sums and divisions. `collapsed` and `post` are as
/// [`mix_collapse_from_logits`] with zero logit error. `cells` are the
/// Sinkhorn input `exp(d) / sum + eps` `[rows, N, N]`, F32 outputs: each
/// `expf` within 2 ulp, the index-order sum from zero within `N - 1`
/// roundings, then one rounding each for the division and `+ eps`.
pub fn mix_stage_from_kernel_logits(
    query: SinkhornHyperConnectionQuery,
    x: &[f64],
    logits: &[f32],
    base: &[f32],
    scale: &[f32],
) -> Result<MixStageOutput, Refusal> {
    let (mix, _) = placed_mix_of(query, HyperConnectionPlacement::PerSublayer)?;
    check_kernel_logits(mix, logits, base, scale)?;
    let zero = vec![0.0; logits.len()];
    let stage =
        mix_collapse_from_logits(query, x, &widen(logits), &zero, &widen(base), &widen(scale))?;
    Ok(MixStageOutput {
        collapsed: stage.collapsed,
        post: stage.post,
        cells: kernel_cells(mix, logits, base, scale)?,
    })
}

/// The nonlinear stage of a lagged `MixCollapse` from a kernel's own F32
/// logits, as [`mix_stage_from_kernel_logits`] emulates it: `pre` and
/// `post` as [`lagged_mix_collapse_from_logits`] with zero logit error, and
/// the Sinkhorn input `cells`. The collapsed row does not depend on the
/// logits ([`collapse`]).
pub fn lagged_mix_stage_from_kernel_logits(
    query: SinkhornHyperConnectionQuery,
    logits: &[f32],
    base: &[f32],
    scale: &[f32],
) -> Result<LaggedMixStageOutput, Refusal> {
    let (mix, capacity) = placed_mix_of(query, HyperConnectionPlacement::LaggedSinglePass)?;
    check_kernel_logits(mix, logits, base, scale)?;
    let mixes = mix.mixes() as usize;
    rows_of(logits.len(), mixes, capacity)?;
    let (base64, scale64) = (widen(base), widen(scale));
    let zero = vec![0.0; mixes];
    let (mut pre, mut post) = ((Vec::new(), Vec::new()), (Vec::new(), Vec::new()));
    for l in logits.chunks_exact(mixes) {
        let stage = row_stage(mix, &widen(l), &zero, &base64, &scale64);
        for (values, out) in [(&stage.pre, &mut pre), (&stage.post, &mut post)] {
            for &(p, lo, hi) in values {
                out.0.push(p);
                out.1.push(f32_bound(p, lo, hi) / GATE);
            }
        }
    }
    Ok(LaggedMixStageOutput {
        pre: OracleOutput::new(pre.0, pre.1, 1)?,
        post: OracleOutput::new(post.0, post.1, 1)?,
        cells: kernel_cells(mix, logits, base, scale)?,
    })
}

fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

fn check_kernel_logits(
    mix: SinkhornMix,
    logits: &[f32],
    base: &[f32],
    scale: &[f32],
) -> Result<(), Refusal> {
    let mixes = mix.mixes() as usize;
    if base.len() != mixes || scale.len() != 3 || !logits.len().is_multiple_of(mixes) {
        return Err(Refusal("Sinkhorn mixer logit shape mismatch"));
    }
    Ok(())
}

/// The Sinkhorn input cells `exp(d) / sum + eps` `[rows, N, N]` from a
/// kernel's own F32 logits, F32 outputs: the affine FMAs, the row maximum
/// and the subtraction emulated exactly, each `expf` within 2 ulp, the
/// index-order sum from zero within `N - 1` roundings, then one rounding
/// each for the division and `+ eps`.
fn kernel_cells(
    mix: SinkhornMix,
    logits: &[f32],
    base: &[f32],
    scale: &[f32],
) -> Result<OracleOutput, Refusal> {
    let (n, mixes) = (mix.streams as usize, mix.mixes() as usize);
    let eps = f32::from_bits(mix.eps_bits);
    let relative = 1.01 * (n as f64 + 9.0) * U;
    let (mut values, mut bounds) = (Vec::new(), Vec::new());
    for l in logits.chunks_exact(mixes) {
        for i in 0..n {
            let v = (0..n)
                .map(|j| {
                    let m = 2 * n + i * n + j;
                    l[m].mul_add(scale[2], base[m])
                })
                .collect::<Vec<_>>();
            let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp = v
                .iter()
                .map(|&v| f64::from(v - max).exp())
                .collect::<Vec<_>>();
            let total = exp.iter().sum::<f64>();
            for e in &exp {
                let q = e / total;
                values.push(q + f64::from(eps));
                bounds.push((q * relative + 1.01 * U * (q + f64::from(eps)) + TINY) / GATE);
            }
        }
    }
    OracleOutput::new(values, bounds, 1)
}

/// The whole `MixCollapse` from the streams: the projection and rstd, the
/// logits `rstd * projection` with error bound `rstd * 2^-22 K S` plus the
/// rstd bound times the projection, then [`mix_collapse_from_logits`].
pub fn mix_collapse(
    query: SinkhornHyperConnectionQuery,
    x: &[f64],
    weights: MixWeights<'_>,
) -> Result<MixCollapseOutput, Refusal> {
    let (logits, errors) = bounded_logits(query, x, weights)?;
    mix_collapse_from_logits(query, x, &logits, &errors, weights.base, weights.scale)
}

/// The whole lagged `MixCollapse` from the streams and the F32 collapse
/// weights `pre_in` `[rows, N]`: [`mix_collapse`]'s logits and errors, then
/// [`lagged_mix_collapse_from_logits`].
pub fn lagged_mix_collapse(
    query: SinkhornHyperConnectionQuery,
    x: &[f64],
    pre_in: &[f32],
    weights: MixWeights<'_>,
) -> Result<LaggedMixCollapseOutput, Refusal> {
    let (logits, errors) = bounded_logits(query, x, weights)?;
    lagged_mix_collapse_from_logits(
        query,
        x,
        pre_in,
        &logits,
        &errors,
        weights.base,
        weights.scale,
    )
}

/// The logits `rstd * projection` and their error bounds.
fn bounded_logits(
    query: SinkhornHyperConnectionQuery,
    x: &[f64],
    weights: MixWeights<'_>,
) -> Result<(Vec<f64>, Vec<f64>), Refusal> {
    let projected = projection(query, x, weights)?;
    let scales = rstd(query, x)?;
    let mixes = weights.base.len();
    let k = projected.reduction_lengths[0] as f64;
    let mut logits = Vec::with_capacity(projected.values.len());
    let mut errors = Vec::with_capacity(projected.values.len());
    for (index, (&p, &s)) in projected
        .values
        .iter()
        .zip(&projected.sum_abs_products)
        .enumerate()
    {
        let row = index / mixes;
        let r = scales.values[row];
        let er = GATE * scales.sum_abs_products[row];
        logits.push(r * p);
        errors.push(r * GATE * k * s + er * (p.abs() + GATE * k * s) + U * (r * p).abs());
    }
    Ok((logits, errors))
}

/// The BF16 `post` and `comb` a kernel stores, rounded once from F32.
pub fn rounded(values: &[f64]) -> Vec<u16> {
    values.iter().map(|&v| f32_to_bf16(v as f32)).collect()
}

/// `Expand` exactly: `X'_j = bf16(bf16(post_j * y) +
/// bf16(sum_i comb[i][j] * X_i))` per element in F32, the stream sum an FMA
/// chain from zero in stream order. `x` is `[rows, N, H]`, `y` `[rows, H]`, `post` `[rows, N]`
/// and `comb` `[rows, N, N]`, all BF16 bits; the result is `[rows, N, H]`.
pub fn expand(
    query: SinkhornHyperConnectionQuery,
    x: &[u16],
    y: &[u16],
    post: &[u16],
    comb: &[u16],
) -> Result<Vec<u16>, Refusal> {
    query.validate()?;
    let SinkhornHyperConnectionQuery::Expand {
        streams,
        width,
        capacity,
    } = query
    else {
        return Err(Refusal("expected a Sinkhorn expand query"));
    };
    let (n, h) = (streams as usize, width as usize);
    let rows = rows_of(x.len(), n * h, capacity)?;
    if y.len() != rows * h || post.len() != rows * n || comb.len() != rows * n * n {
        return Err(Refusal("Sinkhorn expand operand shape mismatch"));
    }
    let f = bf16_to_f32;
    let mut out = vec![0; rows * n * h];
    for row in 0..rows {
        for j in 0..n {
            let p = f(post[row * n + j]);
            for col in 0..h {
                let placed = f(f32_to_bf16(p * f(y[row * h + col])));
                let mut mixed = 0.0_f32;
                for i in 0..n {
                    let weight = f(comb[(row * n + i) * n + j]);
                    mixed = weight.mul_add(f(x[(row * n + i) * h + col]), mixed);
                }
                let sum = placed + f(f32_to_bf16(mixed));
                out[(row * n + j) * h + col] = f32_to_bf16(sum);
            }
        }
    }
    Ok(out)
}

/// `Head` exactly: `bf16((sum_i f32(X_i)) / N)` with the sum in F32 from
/// zero in stream order. `x` is `[rows, N, H]` BF16 bits; the result is
/// `[rows, H]`.
pub fn head(query: SinkhornHyperConnectionQuery, x: &[u16]) -> Result<Vec<u16>, Refusal> {
    query.validate()?;
    let SinkhornHyperConnectionQuery::Head {
        streams,
        width,
        mode: StreamHeadMode::MeanOfStreams,
        capacity,
    } = query
    else {
        return Err(Refusal("expected a mean-of-streams head query"));
    };
    let (n, h) = (streams as usize, width as usize);
    let rows = rows_of(x.len(), n * h, capacity)?;
    let mut out = vec![0; rows * h];
    for row in 0..rows {
        for col in 0..h {
            let mut sum = 0.0_f32;
            for i in 0..n {
                sum += bf16_to_f32(x[(row * n + i) * h + col]);
            }
            out[row * h + col] = f32_to_bf16(sum / n as f32);
        }
    }
    Ok(out)
}

/// `ExpandF32` exactly: `X'_j = bf16(fl(post_j * y) + s_j)` per element in
/// F32, with `s_j = sum_i comb[i][j] * X_i` an FMA chain from zero in stream
/// order. `x` is `[rows, N, H]` and `y` `[rows, H]` BF16 bits, `post`
/// `[rows, N]` and `comb` `[rows, N, N]` F32; the result is `[rows, N, H]`.
pub fn expand_f32(
    query: SinkhornHyperConnectionQuery,
    x: &[u16],
    y: &[u16],
    post: &[f32],
    comb: &[f32],
) -> Result<Vec<u16>, Refusal> {
    let (n, h, rows) = expand_f32_shape(query, x.len(), y.len(), post.len(), comb.len())?;
    let f = bf16_to_f32;
    let mut out = vec![0; rows * n * h];
    for row in 0..rows {
        for j in 0..n {
            let p = post[row * n + j];
            for col in 0..h {
                let placed = p * f(y[row * h + col]);
                let mut mixed = 0.0_f32;
                for i in 0..n {
                    let weight = comb[(row * n + i) * n + j];
                    mixed = weight.mul_add(f(x[(row * n + i) * h + col]), mixed);
                }
                out[(row * n + j) * h + col] = f32_to_bf16(placed + mixed);
            }
        }
    }
    Ok(out)
}

/// `ExpandF32` in f64 from the operands' values (`x`, `y` BF16, `post`,
/// `comb` F32, shaped as in [`expand_f32`]): `post_j * y + sum_i comb[i][j]
/// * X_i`, with a bound of `N + 2` roundings of the sum of absolute terms
/// plus the BF16 rounding (K = 1). [`expand_f32`] lies within it.
pub fn expand_f32_reference(
    query: SinkhornHyperConnectionQuery,
    x: &[f64],
    y: &[f64],
    post: &[f64],
    comb: &[f64],
) -> Result<OracleOutput, Refusal> {
    let (n, h, rows) = expand_f32_shape(query, x.len(), y.len(), post.len(), comb.len())?;
    let mut values = Vec::with_capacity(rows * n * h);
    let mut bounds = Vec::with_capacity(rows * n * h);
    for row in 0..rows {
        for j in 0..n {
            for col in 0..h {
                let placed = post[row * n + j] * y[row * h + col];
                let (mut value, mut absolute) = (placed, placed.abs());
                for i in 0..n {
                    let term = comb[(row * n + i) * n + j] * x[(row * n + i) * h + col];
                    value += term;
                    absolute += term.abs();
                }
                let far = 1.01 * (n as f64 + 2.0) * U * absolute;
                values.push(value);
                bounds.push(stored_bound(value, value - far, value + far) / GATE);
            }
        }
    }
    OracleOutput::new(values, bounds, 1)
}

fn expand_f32_shape(
    query: SinkhornHyperConnectionQuery,
    x: usize,
    y: usize,
    post: usize,
    comb: usize,
) -> Result<(usize, usize, usize), Refusal> {
    query.validate()?;
    let SinkhornHyperConnectionQuery::ExpandF32 {
        streams,
        width,
        capacity,
    } = query
    else {
        return Err(Refusal("expected a Sinkhorn F32 expand query"));
    };
    let (n, h) = (streams as usize, width as usize);
    let rows = rows_of(x, n * h, capacity)?;
    if y != rows * h || post != rows * n || comb != rows * n * n {
        return Err(Refusal("Sinkhorn expand operand shape mismatch"));
    }
    Ok((n, h, rows))
}

/// The sigmoid-weighted `Head` exactly: `bf16(sum_i pre_i * f32(X_i))`, an
/// FMA chain from zero in stream order, with `pre` `[rows, N]` the F32
/// weights the last lagged mix-collapse stored. `x` is `[rows, N, H]` BF16
/// bits; the result is `[rows, H]`.
pub fn weighted_head(
    query: SinkhornHyperConnectionQuery,
    x: &[u16],
    pre: &[f32],
) -> Result<Vec<u16>, Refusal> {
    require_head_mode(query, StreamHeadMode::SigmoidWeighted)?;
    let (n, h, rows) = weighted_head_shape(query, x.len(), pre.len())?;
    let mut out = vec![0; rows * h];
    for row in 0..rows {
        for col in 0..h {
            let mut sum = 0.0_f32;
            for i in 0..n {
                sum = pre[row * n + i].mul_add(bf16_to_f32(x[(row * n + i) * h + col]), sum);
            }
            out[row * h + col] = f32_to_bf16(sum);
        }
    }
    Ok(out)
}

/// The F32-output sigmoid-weighted `Head` exactly: an F32 FMA chain from
/// +0 in stream order with no BF16 rounding. `x` is `[rows, N, H]` BF16
/// bits and `pre` is `[rows, N]` F32; the result is `[rows, H]` F32.
/// Compare its bits, including signed zero, with the device output.
pub fn weighted_head_f32(
    query: SinkhornHyperConnectionQuery,
    x: &[u16],
    pre: &[f32],
) -> Result<Vec<f32>, Refusal> {
    require_head_mode(query, StreamHeadMode::SigmoidWeightedF32)?;
    let (n, h, rows) = weighted_head_shape(query, x.len(), pre.len())?;
    let mut out = vec![0.0; rows * h];
    for row in 0..rows {
        for col in 0..h {
            let mut sum = 0.0_f32;
            for i in 0..n {
                sum = pre[row * n + i].mul_add(bf16_to_f32(x[(row * n + i) * h + col]), sum);
            }
            out[row * h + col] = sum;
        }
    }
    Ok(out)
}

fn require_head_mode(
    query: SinkhornHyperConnectionQuery,
    expected: StreamHeadMode,
) -> Result<(), Refusal> {
    if !matches!(query, SinkhornHyperConnectionQuery::Head { mode, .. } if mode == expected) {
        return Err(Refusal("weighted head output mode mismatch"));
    }
    Ok(())
}

/// Either sigmoid-weighted `Head` in f64 from the streams' values and the F32
/// weights `pre`: `sum_i pre_i * X_i` within `N` roundings of its sum of
/// absolute terms, plus underflow slack (K = 1). The BF16 mode also
/// includes its output rounding; the F32 mode has no BF16 allowance.
/// [`weighted_head`] and [`weighted_head_f32`] lie within their bounds,
/// checked with their respective output dtype. Exact bits remain required.
pub fn weighted_head_reference(
    query: SinkhornHyperConnectionQuery,
    x: &[f64],
    pre: &[f32],
) -> Result<OracleOutput, Refusal> {
    let (n, h, rows) = weighted_head_shape(query, x.len(), pre.len())?;
    let mut values = Vec::with_capacity(rows * h);
    let mut bounds = Vec::with_capacity(rows * h);
    for row in 0..rows {
        let given = given_weights(&pre[row * n..][..n]);
        let streams = &x[row * n * h..][..n * h];
        for col in 0..h {
            let (value, far) = weighted_column(&given, streams, h, col);
            values.push(value);
            let bound = if matches!(
                query,
                SinkhornHyperConnectionQuery::Head {
                    mode: StreamHeadMode::SigmoidWeightedF32,
                    ..
                }
            ) {
                far + n as f64 * TINY
            } else {
                stored_bound(value, value - far, value + far)
            };
            bounds.push(bound / GATE);
        }
    }
    OracleOutput::new(values, bounds, 1)
}

fn weighted_head_shape(
    query: SinkhornHyperConnectionQuery,
    x: usize,
    pre: usize,
) -> Result<(usize, usize, usize), Refusal> {
    query.validate()?;
    let SinkhornHyperConnectionQuery::Head {
        streams,
        width,
        mode: StreamHeadMode::SigmoidWeighted | StreamHeadMode::SigmoidWeightedF32,
        capacity,
    } = query
    else {
        return Err(Refusal("expected a sigmoid-weighted head query"));
    };
    let (n, h) = (streams as usize, width as usize);
    let rows = rows_of(x, n * h, capacity)?;
    if pre != rows * n {
        return Err(Refusal("Sinkhorn head weight shape mismatch"));
    }
    Ok((n, h, rows))
}

/// Step 5 of `MixCollapse` exactly, from given F32 collapse weights `pre`
/// `[rows, N]`: `bf16(sum_i pre_i * f32(X_i))`, an FMA chain from zero in
/// stream order. `x` is `[rows, N, H]` BF16 bits; the result is `[rows, H]`.
/// A per-sublayer kernel's collapsed rows equal it bit for bit whenever its
/// `pre` equals the given one, such as where the sigmoid saturates (`pre` =
/// `1 + eps` or `eps`). A lagged kernel's collapsed rows always equal it
/// for the `pre_in` the kernel read.
pub fn collapse(
    query: SinkhornHyperConnectionQuery,
    x: &[u16],
    pre: &[f32],
) -> Result<Vec<u16>, Refusal> {
    let (mix, capacity) = mix_of(query)?;
    let (n, h) = (mix.streams as usize, mix.width as usize);
    let rows = rows_of(x.len(), n * h, capacity)?;
    if pre.len() != rows * n {
        return Err(Refusal("Sinkhorn collapse weight shape mismatch"));
    }
    let mut out = vec![0; rows * h];
    for row in 0..rows {
        for col in 0..h {
            let mut sum = 0.0_f32;
            for i in 0..n {
                sum = pre[row * n + i].mul_add(bf16_to_f32(x[(row * n + i) * h + col]), sum);
            }
            out[row * h + col] = f32_to_bf16(sum);
        }
    }
    Ok(out)
}

// `MixCollapseNorm`'s sum of squares: the contract's 256-lane tree
// (`RmsNormNumerics::HF_TREE_256`), with IEEE F32 adds.
use crate::oracle::hf_tree_256_sum as tree_sum;

/// `MixCollapseNorm`'s input norm exactly: the HF RMSNorm of each stored
/// BF16 `collapsed` row `[rows, H]` with the BF16 `weight` `[H]`, its sum of
/// squares in the contract's 256-lane tree. The result is `[rows, H]` BF16
/// bits, which a kernel's normed rows equal for finite collapsed rows.
pub fn input_norm(
    query: SinkhornHyperConnectionQuery,
    collapsed: &[u16],
    weight: &[u16],
) -> Result<Vec<u16>, Refusal> {
    query.validate()?;
    let SinkhornHyperConnectionQuery::MixCollapseNorm {
        mix,
        rmsnorm_eps_bits,
        capacity,
    } = query
    else {
        return Err(Refusal("expected a Sinkhorn mix-collapse and norm query"));
    };
    hf_norm(mix, rmsnorm_eps_bits, capacity, collapsed, weight, tree_sum)
}

/// `AllReduceExpandMixNorm`'s input norm exactly: the HF RMSNorm of each
/// stored BF16 `collapsed` row `[rows, H]` with the BF16 `weight` `[H]`,
/// its sum of squares in the order the query's `norm` names (from zero in
/// column order, or [`input_norm`]'s 256-lane tree). The result is
/// `[rows, H]` BF16 bits, which a kernel's normed rows equal for finite
/// collapsed rows.
pub fn epilogue_norm(
    query: SinkhornHyperConnectionQuery,
    collapsed: &[u16],
    weight: &[u16],
) -> Result<Vec<u16>, Refusal> {
    query.validate()?;
    let SinkhornHyperConnectionQuery::AllReduceExpandMixNorm {
        mix,
        rmsnorm_eps_bits,
        norm,
        capacity,
        ..
    } = query
    else {
        return Err(Refusal("expected a fused Sinkhorn comm epilogue query"));
    };
    let ordered = |squares: &[f32]| squares.iter().fold(0.0_f32, |sum, &square| sum + square);
    match norm {
        RmsNormReduction::Ordered => {
            hf_norm(mix, rmsnorm_eps_bits, capacity, collapsed, weight, ordered)
        }
        RmsNormReduction::WarpTree { .. } => {
            hf_norm(mix, rmsnorm_eps_bits, capacity, collapsed, weight, tree_sum)
        }
    }
}

/// The HF RMSNorm of BF16 `collapsed` rows with the BF16 `weight`, the
/// rounded squares added by `sum`: `inv = 1 / sqrtf(s / H + eps)` and
/// `bf16(bf16(x_k * inv) * w_k)`.
fn hf_norm(
    mix: SinkhornMix,
    rmsnorm_eps_bits: u32,
    capacity: u32,
    collapsed: &[u16],
    weight: &[u16],
    sum: impl Fn(&[f32]) -> f32,
) -> Result<Vec<u16>, Refusal> {
    let h = mix.width as usize;
    rows_of(collapsed.len(), h, capacity)?;
    if weight.len() != h {
        return Err(Refusal("Sinkhorn input norm weight shape mismatch"));
    }
    let eps = f32::from_bits(rmsnorm_eps_bits);
    let mut out = Vec::with_capacity(collapsed.len());
    for row in collapsed.chunks_exact(h) {
        let squares = row
            .iter()
            .map(|&v| {
                let v = bf16_to_f32(v);
                v * v
            })
            .collect::<Vec<_>>();
        let inv = 1.0 / (sum(&squares) / h as f32 + eps).sqrt();
        out.extend(row.iter().zip(weight).map(|(&v, &w)| {
            let normalized = bf16_to_f32(f32_to_bf16(bf16_to_f32(v) * inv));
            f32_to_bf16(normalized * bf16_to_f32(w))
        }));
    }
    Ok(out)
}

/// The f64 value of `bf16(post_j * y)` for a stored `post`, the rounding
/// the reference's BF16 mixing applies.
pub fn placed_bf16(post: f64, y: f64) -> f64 {
    bf16_round(bf16_round(post) * y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::DType;
    use crate::oracle::check_numeric;
    use crate::sinkhorn_hyper_connection::HyperConnectionPlacement;

    const ORDER: SinkhornOrder = SinkhornOrder::RowSoftmaxThenColumnFirst;

    fn mix(streams: u32, width: u32, iterations: u32, eps: f32) -> SinkhornMix {
        SinkhornMix {
            streams,
            width,
            iterations,
            eps_bits: eps.to_bits(),
            norm_eps_bits: 1e-5_f32.to_bits(),
            order: ORDER,
            placement: HyperConnectionPlacement::PerSublayer,
            fn_dtype: DType::Bf16,
        }
    }

    fn query(mix: SinkhornMix) -> SinkhornHyperConnectionQuery {
        SinkhornHyperConnectionQuery::MixCollapse { mix, capacity: 8 }
    }

    /// Deterministic values on a BF16 grid in [-1, 1).
    fn grid(len: usize, seed: u64) -> Vec<f64> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                bf16_round(((state >> 40) % 4096) as f64 / 2048.0 - 1.0)
            })
            .collect()
    }

    fn close(a: f64, b: f64, tolerance: f64) {
        assert!((a - b).abs() <= tolerance, "{a} vs {b}");
    }

    #[test]
    fn sinkhorn_starts_with_a_column_normalization_then_row_column_rounds() {
        // Row softmax of [[0, ln 3], [0, 0]] is [[1/4, 3/4], [1/2, 1/2]].
        let c = [0.0, 3_f64.ln(), 0.0, 0.0];
        let eps = 1e-30;
        let one = sinkhorn(ORDER, &c, 2, 1, eps);
        for (value, expected) in one.iter().zip([1.0 / 3.0, 3.0 / 5.0, 2.0 / 3.0, 2.0 / 5.0]) {
            close(*value, expected, 1e-15);
        }
        let two = sinkhorn(ORDER, &c, 2, 2, eps);
        for (value, expected) in two
            .iter()
            .zip([4.0 / 11.0, 12.0 / 19.0, 7.0 / 11.0, 7.0 / 19.0])
        {
            close(*value, expected, 1e-15);
        }
        // A column normalization comes last, so columns sum to one exactly
        // where rows do not.
        close(two[0] + two[2], 1.0, 1e-15);
        assert!((two[0] + two[1] - 1.0).abs() > 1e-3);
        // Zero iterations leave the softmax plus eps.
        let none = sinkhorn(ORDER, &c, 2, 0, 0.5);
        close(none[1], 1.25, 1e-15);
    }

    #[test]
    fn eps_joins_the_softmax_and_every_divisor() {
        // Uniform logits: every softmax value is 1/2, plus eps, and the one
        // column normalization divides by 2 (1/2 + eps) + eps.
        let eps = 0.125;
        let one = sinkhorn(ORDER, &[0.0; 4], 2, 1, eps);
        for value in one {
            close(value, 0.625 / (1.25 + 0.125), 1e-15);
        }
    }

    #[test]
    fn twenty_iterations_differ_from_nineteen_at_bf16() {
        // Upper triangular up to eps: the scaling tends to the identity
        // only as 1 / iterations, so one more round still moves BF16 bits.
        let mut c = vec![0.0; 16];
        for i in 0..4 {
            for j in 0..i {
                c[i * 4 + j] = -40.0;
            }
        }
        let nineteen = sinkhorn(ORDER, &c, 4, 19, 1e-6);
        let twenty = sinkhorn(ORDER, &c, 4, 20, 1e-6);
        let largest = nineteen
            .iter()
            .zip(&twenty)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(largest > 1e-3, "difference {largest}");
        assert_ne!(rounded(&nineteen), rounded(&twenty));
    }

    #[test]
    fn mix_collapse_matches_hand_values() {
        // Two streams of 128, one iteration. x_0[0] = 2 and every other
        // element 0, so rstd = 1 / sqrt(4 / 256 + 1e-5); fn[0][0] = 1 and
        // fn[3][0] = 1 give logits 2 rstd for pre_0 and post_1.
        let m = mix(2, 128, 1, 1e-6);
        let q = query(m);
        let mut x = vec![0.0; 256];
        x[0] = 2.0;
        let mut projection = vec![0.0; 8 * 256];
        projection[0] = 1.0;
        projection[3 * 256] = 1.0;
        let base = [0.0, 0.5, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0];
        let scale = [0.125, 0.25, 1.0];
        let weights = MixWeights {
            projection: &projection,
            base: &base,
            scale: &scale,
        };
        let r = 1.0 / (4.0_f64 / 256.0 + f64::from(1e-5_f32)).sqrt();
        close(rstd(q, &x).unwrap().values[0], r, 1e-12);
        let out = mix_collapse(q, &x, weights).unwrap();
        let eps = f64::from(1e-6_f32);
        let pre0 = sigmoid(2.0 * r * 0.125) + eps;
        close(out.collapsed.values[0], pre0 * 2.0, 1e-12);
        assert_eq!(out.collapsed.values[1..].iter().sum::<f64>(), 0.0);
        close(out.post.values[0], 2.0 * sigmoid(-0.25), 1e-15);
        close(out.post.values[1], 2.0 * sigmoid(2.0 * r * 0.25), 1e-12);
        // Zero comb logits: softmax 1/2 + eps, one column normalization.
        for value in &out.comb.values {
            close(*value, (0.5 + eps) / (1.0 + 3.0 * eps), 1e-15);
        }
        for bound in out
            .comb
            .sum_abs_products
            .iter()
            .chain(&out.post.sum_abs_products)
        {
            assert!(*bound > 0.0 && bound.is_finite());
        }
    }

    #[test]
    fn bounds_enclose_logits_anywhere_in_their_interval() {
        // Logits moved to either end of their whole error interval keep
        // post and collapsed inside the gate, after BF16 rounding.
        let m = mix(4, 128, 20, 1e-6);
        let q = query(m);
        let x = grid(2 * 512, 3);
        let projection = grid(24 * 512, 5)
            .iter()
            .map(|v| v / 16.0)
            .collect::<Vec<_>>();
        let base = grid(24, 9);
        let scale = [0.5, 0.75, 2.0];
        let weights = MixWeights {
            projection: &projection,
            base: &base,
            scale: &scale,
        };
        let out = mix_collapse(q, &x, weights).unwrap();
        let projected = projection_values(q, &x, weights);
        let scales = rstd(q, &x).unwrap().values;
        let k = 512.0;
        for sign in [-1.0, 1.0] {
            let moved = projected
                .values
                .iter()
                .zip(&projected.sum_abs_products)
                .enumerate()
                .map(|(i, (p, s))| scales[i / 24] * (p + sign * GATE * k * s))
                .collect::<Vec<_>>();
            let zero = vec![0.0; moved.len()];
            let moved = mix_collapse_from_logits(q, &x, &moved, &zero, &base, &scale).unwrap();
            let stored = |values: &[f64]| {
                rounded(values)
                    .iter()
                    .map(|&b| f64::from(bf16_to_f32(b)))
                    .collect::<Vec<_>>()
            };
            check_numeric(&stored(&moved.post.values), &out.post, DType::Bf16).unwrap();
            check_numeric(
                &stored(&moved.collapsed.values),
                &out.collapsed,
                DType::Bf16,
            )
            .unwrap();
            check_numeric(&stored(&moved.comb.values), &out.comb, DType::Bf16).unwrap();
        }
    }

    /// Four streams of 4096 ones; mix row `m` of fn starts with [-65, 63]
    /// and is otherwise zero, so its raw projection is -2 with S = 128 and
    /// the projection gate allows 2^-22 * 16384 * 128 = 0.5.
    fn served_width_case(m: usize) -> (SinkhornHyperConnectionQuery, Vec<f64>, Vec<f64>) {
        let q = query(mix(4, 4096, 20, 1e-6));
        let x = vec![1.0; 16_384];
        let mut projection = vec![0.0; 24 * 16_384];
        projection[m * 16_384] = -65.0;
        projection[m * 16_384 + 1] = 63.0;
        (q, x, projection)
    }

    #[test]
    fn a_projection_moved_within_its_gate_stays_inside_the_post_bound() {
        let (q, x, projection) = served_width_case(4);
        let base = [0.0; 24];
        let scale = [1.0; 3];
        let weights = MixWeights {
            projection: &projection,
            base: &base,
            scale: &scale,
        };
        let projected = projection_values(q, &x, weights);
        assert_eq!(
            (projected.values[4], projected.sum_abs_products[4]),
            (-2.0, 128.0)
        );
        let out = mix_collapse(q, &x, weights).unwrap();
        let r = rstd(q, &x).unwrap().values[0];
        let mut moved = projected.values.iter().map(|p| p * r).collect::<Vec<_>>();
        moved[4] = -1.5 * r;
        let zero = vec![0.0; 24];
        let moved = mix_collapse_from_logits(q, &x, &moved, &zero, &base, &scale).unwrap();
        let post = f64::from(bf16_to_f32(rounded(&moved.post.values)[0]));
        assert!((post - out.post.values[0]).abs() > 0.1268);
        check_numeric(
            &[post],
            &OracleOutput::new(
                vec![out.post.values[0]],
                vec![out.post.sum_abs_products[0]],
                1,
            )
            .unwrap(),
            DType::Bf16,
        )
        .unwrap();
        // The same logit supplied directly with error 0.5.
        let mut logits = vec![0.0; 24];
        let mut errors = vec![0.0; 24];
        logits[4] = -2.0;
        errors[4] = 0.5;
        let direct = mix_collapse_from_logits(q, &x, &logits, &errors, &base, &scale).unwrap();
        let shifted = 2.0 * sigmoid(-1.5);
        let stored = f64::from(bf16_to_f32(f32_to_bf16(shifted as f32)));
        check_numeric(
            &[stored],
            &OracleOutput::new(
                vec![direct.post.values[0]],
                vec![direct.post.sum_abs_products[0]],
                1,
            )
            .unwrap(),
            DType::Bf16,
        )
        .unwrap();
        // comb[0][0] from C[0][0] = -2 moved to -1.5 stays in its range bound.
        logits[4] = 0.0;
        errors[4] = 0.0;
        logits[8] = -2.0;
        errors[8] = 0.5;
        let reference = mix_collapse_from_logits(q, &x, &logits, &errors, &base, &scale).unwrap();
        logits[8] = -1.5;
        errors[8] = 0.0;
        let moved = mix_collapse_from_logits(q, &x, &logits, &errors, &base, &scale).unwrap();
        let comb = rounded(&moved.comb.values)
            .iter()
            .map(|&b| f64::from(bf16_to_f32(b)))
            .collect::<Vec<_>>();
        check_numeric(&comb, &reference.comb, DType::Bf16).unwrap();
    }

    #[test]
    fn a_zero_comb_fails_at_an_extreme_common_offset() {
        // Zero logits and scales, every comb base 1e8 (an F32 value): the
        // affine FMAs are exact, every softmax cell is 1/4 + eps, and the
        // comb is about 0.24999975.
        let m = mix(4, 128, 20, 1e-6);
        let q = query(m);
        let x = grid(512, 19);
        let logits = [0.0_f32; 24];
        let mut base = [0.0_f32; 24];
        base[8..].fill(1e8);
        let scale = [0.0_f32; 3];
        let stage = mix_stage_from_kernel_logits(q, &x, &logits, &base, &scale).unwrap();
        for bound in stage
            .post
            .sum_abs_products
            .iter()
            .chain(&stage.cells.sum_abs_products)
        {
            assert!(GATE * bound < 1e-5, "bound {}", GATE * bound);
        }
        // A kernel's cells: expf(0) = 1, 1 / 4, plus eps.
        let eps = 1e-6_f32;
        let cells = [1.0_f32 / 4.0 + eps; 16];
        let as_f64 = |v: &[f32]| v.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
        check_numeric(&as_f64(&cells), &stage.cells, DType::F32).unwrap();
        assert!(check_numeric(&[0.0; 16], &stage.cells, DType::F32).is_err());
        let comb = sinkhorn_f32(ORDER, &cells, 4, 20, eps);
        let bits = comb.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>();
        assert_ne!(bits, vec![0; 16]);
        for value in &comb {
            close(f64::from(*value), 0.25 * (1.0 - 4e-6), 1e-6);
        }
    }

    #[test]
    fn the_f32_normalizations_follow_the_f64_order() {
        // The 2 x 2 case of the order test from its F32 softmax cells.
        let cells = [0.25_f32, 0.75, 0.5, 0.5];
        let two = sinkhorn_f32(ORDER, &cells, 2, 2, 1e-30);
        for (value, expected) in two
            .iter()
            .zip([4.0 / 11.0, 12.0 / 19.0, 7.0 / 11.0, 7.0 / 19.0])
        {
            close(f64::from(*value), expected, 1e-7);
        }
        // Each line sum runs from zero in index order: one rounding per add.
        let wide = sinkhorn_f32(ORDER, &[1.0, 2.0, 3.0, 4.0], 2, 1, 0.0);
        assert_eq!(wide, vec![0.25, 2.0 / 6.0, 0.75, 4.0 / 6.0]);
    }

    fn projection_values(
        q: SinkhornHyperConnectionQuery,
        x: &[f64],
        weights: MixWeights<'_>,
    ) -> OracleOutput {
        projection(q, x, weights).unwrap()
    }

    #[test]
    fn the_projection_is_compensated_in_k_order() {
        // 2^60 + 1 - 2^60: a plain f64 sum loses the 1.
        let q = query(mix(2, 128, 20, 1e-6));
        let mut x = vec![0.0; 256];
        x[..3].copy_from_slice(&[2_f64.powi(60), 1.0, -(2_f64.powi(60))]);
        let projection = vec![1.0; 8 * 256];
        let base = [0.0; 8];
        let out = projection_values(
            q,
            &x,
            MixWeights {
                projection: &projection,
                base: &base,
                scale: &[1.0; 3],
            },
        );
        assert_eq!(out.values[0], 1.0);
        assert_eq!(out.sum_abs_products[0], 2_f64.powi(61));
        assert_eq!(out.reduction_lengths[0], 256);
    }

    #[test]
    fn a_wider_mixer_with_a_tiny_eps_is_doubly_stochastic() {
        // 4 x 5120 with eps 1e-20 and F32 fn, the oracle-only case.
        let m = SinkhornMix {
            fn_dtype: DType::F32,
            ..mix(4, 5120, 20, 1e-20)
        };
        let q = query(m);
        let x = grid(4 * 5120, 11);
        let projection = grid(24 * 4 * 5120, 13)
            .iter()
            .map(|v| v / 64.0 + 2_f64.powi(-20))
            .collect::<Vec<_>>();
        let base = grid(24, 17);
        let scale = [1.0, 1.0, 4.0];
        let weights = MixWeights {
            projection: &projection,
            base: &base,
            scale: &scale,
        };
        let out = mix_collapse(q, &x, weights).unwrap();
        let comb = &out.comb.values;
        for j in 0..4 {
            close((0..4).map(|i| comb[i * 4 + j]).sum::<f64>(), 1.0, 1e-15);
            close(comb[j * 4..][..4].iter().sum::<f64>(), 1.0, 1e-3);
        }
        // The eps of 1e-6 instead shifts pre by about 1e-6.
        let coarse = mix_collapse(
            query(SinkhornMix {
                eps_bits: 1e-6_f32.to_bits(),
                ..m
            }),
            &x,
            weights,
        )
        .unwrap();
        let shift = (coarse.collapsed.values[0] - out.collapsed.values[0])
            / (0..4).map(|i| x[i * 5120]).sum::<f64>();
        close(shift, f64::from(1e-6_f32), 1e-12);
        assert_eq!(out.post.values.len(), 4);
        assert_eq!(out.collapsed.values.len(), 5120);
    }

    #[test]
    fn expand_rounds_the_placement_the_mix_and_the_sum_separately() {
        // post * y = 1 + 2^-6 + 2^-14 rounds to 1 + 2^-6; adding the mixed
        // 2^-8 lands on a tie that rounds to even, 1 + 2^-6 (0x3f82). One
        // rounding of the exact sum would give 1 + 3 * 2^-7 (0x3f83).
        let q = SinkhornHyperConnectionQuery::Expand {
            streams: 2,
            width: 1,
            capacity: 1,
        };
        let post = [0x3f81, 0x0000]; // 1 + 2^-7, 0
        let y = [0x3f81];
        let comb = [0x3f80, 0x0000, 0x0000, 0x3f80]; // identity
        let x = [0x3b80, 0x4000]; // 2^-8, 2
        let out = expand(q, &x, &y, &post, &comb).unwrap();
        assert_eq!(out, vec![0x3f82, 0x4000]);
        let exact = 1.0078125_f64 * 1.0078125 + 2_f64.powi(-8);
        assert_eq!(f32_to_bf16(exact as f32), 0x3f83);
        // The rounded post feeds the product, as `placed_bf16` documents.
        assert_eq!(placed_bf16(1.0078125, 1.0078125), 1.015625);
    }

    #[test]
    fn expand_sums_streams_by_fma_from_zero_in_stream_order() {
        // comb column 0 = (1, 1, 1): 1 + 2^-8 + 2^-8 in F32 is exact, so
        // the BF16 rounding of the sum sees 1 + 2^-7 (0x3f81) rather than
        // two separately rounded additions (1, then 1).
        let q = SinkhornHyperConnectionQuery::Expand {
            streams: 3,
            width: 1,
            capacity: 1,
        };
        let comb = [0x3f80; 9];
        let x = [0x3f80, 0x3b80, 0x3b80];
        let out = expand(q, &x, &[0], &[0; 3], &comb).unwrap();
        assert_eq!(out, vec![0x3f81; 3]);
        // A negative zero product added to the zero start becomes +0.
        let signed = expand(q, &[0x8000; 3], &[0x8000], &[0x3f80; 3], &comb).unwrap();
        assert_eq!(signed, vec![0x0000; 3]);
    }

    #[test]
    fn the_head_is_the_mean_rounded_once() {
        let q = SinkhornHyperConnectionQuery::Head {
            streams: 4,
            width: 2,
            mode: StreamHeadMode::MeanOfStreams,
            capacity: 1,
        };
        // Column 0: (1 + 3 * 2^-8) / 4 = 0.25 + 3 * 2^-10, a tie that rounds
        // to even at 0.25 + 2^-8 (0x3e82). Column 1: -0 inputs sum from +0.
        let x = [
            0x3f80, 0x8000, 0x3b80, 0x8000, 0x3b80, 0x8000, 0x3b80, 0x8000,
        ];
        assert_eq!(head(q, &x).unwrap(), vec![0x3e82, 0x0000]);
        let refused = SinkhornHyperConnectionQuery::Head {
            streams: 4,
            width: 2,
            mode: StreamHeadMode::SigmoidWeighted,
            capacity: 1,
        };
        assert!(head(refused, &x).is_err());
    }

    #[test]
    fn shapes_are_checked() {
        let q = query(mix(4, 128, 20, 1e-6));
        let weights = MixWeights {
            projection: &[0.0; 24 * 512],
            base: &[0.0; 24],
            scale: &[0.0; 3],
        };
        assert!(projection(q, &[0.0; 511], weights).is_err());
        assert!(projection(q, &vec![0.0; 9 * 512], weights).is_err());
        assert!(
            projection(
                q,
                &[0.0; 512],
                MixWeights {
                    base: &[0.0; 23],
                    ..weights
                }
            )
            .is_err()
        );
        let expand_query = SinkhornHyperConnectionQuery::Expand {
            streams: 4,
            width: 128,
            capacity: 1,
        };
        assert!(expand(q, &[0; 512], &[0; 128], &[0; 4], &[0; 16]).is_err());
        assert!(expand(expand_query, &[0; 512], &[0; 128], &[0; 4], &[0; 15]).is_err());
        let fused = norm_query(mix(4, 128, 20, 1e-6));
        assert!(collapse(fused, &[0; 512], &[0.0; 3]).is_err());
        assert!(collapse(expand_query, &[0; 512], &[0.0; 4]).is_err());
        assert!(input_norm(fused, &[0; 127], &[0; 128]).is_err());
        assert!(input_norm(fused, &[0; 128], &[0; 127]).is_err());
        assert!(input_norm(q, &[0; 128], &[0; 128]).is_err());
    }

    fn norm_query(mix: SinkhornMix) -> SinkhornHyperConnectionQuery {
        SinkhornHyperConnectionQuery::MixCollapseNorm {
            mix,
            rmsnorm_eps_bits: 1e-5_f32.to_bits(),
            capacity: 8,
        }
    }

    fn bits(values: &[f64]) -> Vec<u16> {
        values.iter().map(|&v| f32_to_bf16(v as f32)).collect()
    }

    #[test]
    fn the_fused_query_collapses_as_the_mix_collapse() {
        let mix = mix(4, 128, 20, 1e-6);
        let (rows, n, h) = (3, 4, 128);
        let x = grid(rows * n * h, 3);
        let projection = grid(24 * n * h, 4)
            .into_iter()
            .map(|v| v / 64.0)
            .collect::<Vec<_>>();
        // Saturated collapse weights: base 64 gives pre = 1 + eps, base -200
        // gives pre = eps, whatever the logits.
        let mut base = grid(24, 5);
        for (i, b) in base.iter_mut().take(n).enumerate() {
            *b = if i % 2 == 0 { 64.0 } else { -200.0 };
        }
        let weights = MixWeights {
            projection: &projection,
            base: &base,
            scale: &[0.0, 0.5, 1.0],
        };
        let plain = mix_collapse(query(mix), &x, weights).unwrap();
        assert_eq!(mix_collapse(norm_query(mix), &x, weights).unwrap(), plain);
        let eps = f32::from_bits(mix.eps_bits);
        let pre = (0..rows * n)
            .map(|i| if i % 2 == 0 { 1.0 + eps } else { eps })
            .collect::<Vec<_>>();
        let exact = collapse(norm_query(mix), &bits(&x), &pre).unwrap();
        let actual = exact
            .iter()
            .map(|&b| f64::from(bf16_to_f32(b)))
            .collect::<Vec<_>>();
        check_numeric(&actual, &plain.collapsed, DType::Bf16).unwrap();
    }

    #[test]
    fn the_input_norm_sums_its_squares_in_the_256_lane_tree() {
        // 4096 then ones: the serial sum loses every one to 2^24's rounding,
        // the tree keeps the eight of lane 1 (columns 8 to 15).
        let mut squares = vec![0.0_f32; 4096];
        squares[0] = 16_777_216.0;
        for square in &mut squares[1..16] {
            *square = 1.0;
        }
        assert_eq!(squares.iter().fold(0.0_f32, |s, &t| s + t), 16_777_216.0);
        assert_eq!(tree_sum(&squares), 16_777_224.0);
        // Columns past the first span land on lane l's next pass.
        let mut wide = vec![0.0_f32; 5000];
        wide[2048 + 8 * 3 + 2] = 2.0;
        wide[4999] = 0.5;
        assert_eq!(tree_sum(&wide), 2.5);

        // Where every order sums exactly, the norm is HF's ordered reference.
        let mix = mix(4, 4096, 20, 1e-6);
        let query = norm_query(mix);
        let x = (0..2 * 4096)
            .map(|i| f32_to_bf16(if i % 3 == 0 { -0.5 } else { 1.0 }))
            .collect::<Vec<_>>();
        let weight = bits(&grid(4096, 6));
        let reference = crate::oracle::rmsnorm(
            &x.iter()
                .map(|&b| f64::from(bf16_to_f32(b)))
                .collect::<Vec<_>>(),
            4096,
            &weight
                .iter()
                .map(|&b| f64::from(bf16_to_f32(b)))
                .collect::<Vec<_>>(),
            1e-5,
            crate::contract::RmsNormKind::HfBf16Intermediate,
            None,
        )
        .unwrap();
        assert_eq!(
            input_norm(query, &x, &weight).unwrap(),
            bits(&reference.values)
        );
        // The normed row follows the tree: the row above differs from HF's
        // ordered sum in its reciprocal.
        let big = (0..4096)
            .map(|i| {
                f32_to_bf16(if i == 0 {
                    4096.0
                } else if i < 16 {
                    1.0
                } else {
                    0.0
                })
            })
            .collect::<Vec<_>>();
        let tree = input_norm(query, &big, &weight[..4096]).unwrap();
        let inv = 1.0 / (16_777_224.0_f32 / 4096.0 + 1e-5).sqrt();
        assert_eq!(
            tree[1],
            f32_to_bf16(bf16_to_f32(f32_to_bf16(inv)) * bf16_to_f32(weight[1]))
        );
    }

    #[test]
    fn the_epilogue_norm_sums_its_squares_in_the_order_its_query_names() {
        let fused = |norm| SinkhornHyperConnectionQuery::AllReduceExpandMixNorm {
            ranks: 4,
            dtype: crate::contract::DType::F32,
            mix: mix(4, 4096, 20, 1e-6),
            rmsnorm_eps_bits: 1e-5_f32.to_bits(),
            norm,
            capacity: 8,
        };
        let ordered = fused(RmsNormReduction::Ordered);
        let tree = fused(RmsNormReduction::WarpTree { threads: 256 });
        // 4096 then fifteen ones: the column order loses the ones to 2^24's
        // rounding (s = 2^24), the tree keeps eight of them (s = 2^24 + 8).
        // A small column value whose normalized BF16 lies between the two
        // reciprocals' roundings then differs between the orders; its square
        // is lost in either sum.
        let inv = |sum: f32| 1.0 / (sum / 4096.0 + 1e-5_f32).sqrt();
        let (in_order_inv, in_tree_inv) = (inv(16_777_216.0), inv(16_777_224.0));
        let flips = (0x0080_u16..0x3d80)
            .find(|&v| {
                let v = bf16_to_f32(v);
                f32_to_bf16(v * in_order_inv) != f32_to_bf16(v * in_tree_inv)
            })
            .expect("a value whose normalized row tells the orders apart");
        let big = (0..4096)
            .map(|i| match i {
                0 => f32_to_bf16(4096.0),
                1..16 => f32_to_bf16(1.0),
                16 => flips,
                _ => 0,
            })
            .collect::<Vec<_>>();
        let weight = vec![f32_to_bf16(1.0); 4096];
        let normed = |inv: f32| f32_to_bf16(bf16_to_f32(flips) * inv);
        let in_order = epilogue_norm(ordered, &big, &weight).unwrap();
        let in_tree = epilogue_norm(tree, &big, &weight).unwrap();
        assert_eq!(in_order[16], normed(in_order_inv));
        assert_eq!(in_tree[16], normed(in_tree_inv));
        assert_ne!(in_order[16], in_tree[16]);
        // The tree order is `MixCollapseNorm`'s input norm.
        assert_eq!(
            in_tree,
            input_norm(norm_query(mix(4, 4096, 20, 1e-6)), &big, &weight).unwrap()
        );
        // Only the fused query has an epilogue norm.
        assert!(epilogue_norm(norm_query(mix(4, 4096, 20, 1e-6)), &big, &weight).is_err());
    }

    fn lagged(m: SinkhornMix) -> SinkhornHyperConnectionQuery {
        query(SinkhornMix {
            placement: HyperConnectionPlacement::LaggedSinglePass,
            ..m
        })
    }

    fn values_of(bits: &[u16]) -> Vec<f64> {
        bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
    }

    #[test]
    fn a_lagged_mix_collapse_collapses_with_the_given_pre_and_keeps_its_own() {
        let m = SinkhornMix {
            fn_dtype: DType::F32,
            norm_eps_bits: 1e-20_f32.to_bits(),
            ..mix(4, 128, 20, 1e-6)
        };
        let (rows, n, h) = (2, 4, 128);
        let x = grid(rows * n * h, 21);
        let projection = grid(24 * n * h, 22)
            .into_iter()
            .map(|v| v / 64.0)
            .collect::<Vec<_>>();
        // Saturated own weights (1 + eps, eps, 1 + eps, eps), so a collapse
        // with them is far from the one-hot collapse.
        let mut base = grid(24, 23);
        for (i, b) in base.iter_mut().take(n).enumerate() {
            *b = if i % 2 == 0 { 64.0 } else { -200.0 };
        }
        let weights = MixWeights {
            projection: &projection,
            base: &base,
            scale: &[0.0, 0.5, 1.0],
        };
        let pre_in = one_hot_pre(rows, n);
        assert_eq!(&pre_in[..8], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let out = lagged_mix_collapse(lagged(m), &x, &pre_in, weights).unwrap();
        let own = mix_collapse(query(m), &x, weights).unwrap();
        // The mixer's math is the placement's twin; only storage and the
        // collapse weights differ.
        assert_eq!(out.post.values, own.post.values);
        assert_eq!(out.comb.values, own.comb.values);
        let eps = f64::from(1e-6_f32);
        for (i, &value) in out.pre.values.iter().enumerate() {
            let expected = if i % 2 == 0 { 1.0 + eps } else { eps };
            close(value, expected, 1e-15);
        }
        // The one-hot collapse is stream 0, exactly, and the kernel's exact
        // collapse lies within the f64 bound.
        let bits = bits(&x);
        let exact = collapse(lagged(m), &bits, &pre_in).unwrap();
        for row in 0..rows {
            assert_eq!(
                &exact[row * h..][..h],
                &bits[row * n * h..][..h],
                "row {row}"
            );
        }
        check_numeric(&values_of(&exact), &out.collapsed, DType::Bf16).unwrap();
        assert!(check_numeric(&values_of(&exact), &own.collapsed, DType::Bf16).is_err());
        // The stored F32 values of a kernel lie within the F32 bounds.
        let as_f32 = |v: &[f64]| v.iter().map(|&v| f64::from(v as f32)).collect::<Vec<_>>();
        check_numeric(&as_f32(&out.pre.values), &out.pre, DType::F32).unwrap();
        check_numeric(&as_f32(&out.post.values), &out.post, DType::F32).unwrap();
        check_numeric(&as_f32(&out.comb.values), &out.comb, DType::F32).unwrap();
    }

    #[test]
    fn lagged_f32_bounds_enclose_logits_anywhere_in_their_interval() {
        let m = mix(4, 128, 20, 1e-6);
        let q = lagged(m);
        let x = grid(2 * 512, 31);
        let projection = grid(24 * 512, 32)
            .iter()
            .map(|v| v / 16.0)
            .collect::<Vec<_>>();
        let base = grid(24, 33);
        let scale = [0.5, 0.75, 2.0];
        let weights = MixWeights {
            projection: &projection,
            base: &base,
            scale: &scale,
        };
        let pre_in = grid(8, 34).iter().map(|&v| v as f32).collect::<Vec<_>>();
        let out = lagged_mix_collapse(q, &x, &pre_in, weights).unwrap();
        let projected = projection_values(q, &x, weights);
        let scales = rstd(q, &x).unwrap().values;
        for sign in [-1.0, 1.0] {
            let moved = projected
                .values
                .iter()
                .zip(&projected.sum_abs_products)
                .enumerate()
                .map(|(i, (p, s))| scales[i / 24] * (p + sign * GATE * 512.0 * s))
                .collect::<Vec<_>>();
            let zero = vec![0.0; moved.len()];
            let moved =
                lagged_mix_collapse_from_logits(q, &x, &pre_in, &moved, &zero, &base, &scale)
                    .unwrap();
            let stored = |v: &[f64]| v.iter().map(|&v| f64::from(v as f32)).collect::<Vec<_>>();
            check_numeric(&stored(&moved.pre.values), &out.pre, DType::F32).unwrap();
            check_numeric(&stored(&moved.post.values), &out.post, DType::F32).unwrap();
            check_numeric(&stored(&moved.comb.values), &out.comb, DType::F32).unwrap();
            // The collapse does not depend on the logits.
            assert_eq!(moved.collapsed, out.collapsed);
        }
        // A BF16-rounded post (the per-sublayer storage) is outside the F32
        // bound somewhere.
        let rounded_post = rounded(&out.post.values)
            .iter()
            .map(|&b| f64::from(bf16_to_f32(b)))
            .collect::<Vec<_>>();
        assert!(check_numeric(&rounded_post, &out.post, DType::F32).is_err());
    }

    #[test]
    fn the_lagged_kernel_stage_matches_the_per_sublayer_cells() {
        let m = mix(4, 128, 20, 1e-6);
        let x = grid(512, 41);
        let logits = grid(24, 42)
            .iter()
            .map(|&v| 4.0 * v as f32)
            .collect::<Vec<_>>();
        let base = grid(24, 43).iter().map(|&v| v as f32).collect::<Vec<_>>();
        let scale = [0.5_f32, 0.75, 2.0];
        let own = mix_stage_from_kernel_logits(query(m), &x, &logits, &base, &scale).unwrap();
        let stage = lagged_mix_stage_from_kernel_logits(lagged(m), &logits, &base, &scale).unwrap();
        assert_eq!(stage.cells, own.cells);
        assert_eq!(stage.post.values, own.post.values);
        // pre is sigmoid(fma(l, s, b)) + eps of the kernel's own logits.
        for i in 0..4 {
            let v = f64::from(logits[i].mul_add(scale[0], base[i]));
            close(stage.pre.values[i], sigmoid(v) + f64::from(1e-6_f32), 1e-15);
        }
        // Each placement's functions refuse the other's queries.
        assert!(lagged_mix_stage_from_kernel_logits(query(m), &logits, &base, &scale).is_err());
        assert!(mix_stage_from_kernel_logits(lagged(m), &x, &logits, &base, &scale).is_err());
        let weights = MixWeights {
            projection: &[0.0; 24 * 512],
            base: &[0.0; 24],
            scale: &[0.0; 3],
        };
        assert!(mix_collapse(lagged(m), &x, weights).is_err());
        assert!(lagged_mix_collapse(query(m), &x, &[0.0; 4], weights).is_err());
        assert!(lagged_mix_collapse(lagged(m), &x, &[0.0; 3], weights).is_err());
        // Projection and rstd do not depend on the placement.
        assert_eq!(
            projection(lagged(m), &x, weights).unwrap(),
            projection(query(m), &x, weights).unwrap()
        );
        assert_eq!(rstd(lagged(m), &x).unwrap(), rstd(query(m), &x).unwrap());
    }

    #[test]
    fn the_f32_expand_rounds_once_from_unrounded_post_and_comb() {
        let q = SinkhornHyperConnectionQuery::ExpandF32 {
            streams: 2,
            width: 1,
            capacity: 1,
        };
        // The BF16 expand's case: post * y = 1 + 2^-6 + 2^-14 plus 2^-8 is
        // rounded once, to 1 + 3 * 2^-7 (0x3f83), not to the BF16 expand's
        // tie 0x3f82.
        let post = [1.0078125_f32, 0.0];
        let comb = [1.0_f32, 0.0, 0.0, 1.0];
        let (x, y) = ([0x3b80, 0x4000], [0x3f81]);
        assert_eq!(
            expand_f32(q, &x, &y, &post, &comb).unwrap(),
            vec![0x3f83, 0x4000]
        );
        let bf16_query = SinkhornHyperConnectionQuery::Expand {
            streams: 2,
            width: 1,
            capacity: 1,
        };
        assert_eq!(
            expand(bf16_query, &x, &y, &[0x3f81, 0], &[0x3f80, 0, 0, 0x3f80]).unwrap(),
            vec![0x3f82, 0x4000]
        );
        // post and comb are not rounded to BF16. post = 1 + 2^-10 times
        // y = 1 + 2^-7, plus x_0 = 2^-8 through comb 1 - 2^-12, lies above
        // the tie 1 + 3 * 2^-8 and rounds up to 0x3f82; with post rounded
        // to BF16 (1) the sum lies below the tie and rounds to 0x3f81.
        let post = [1.0 + 2_f32.powi(-10), 0.0];
        let comb = [1.0 - 2_f32.powi(-12), 0.0, 0.0, 1.0];
        let out = expand_f32(q, &x, &y, &post, &comb).unwrap();
        let exact = 1.0009765625_f64 * 1.0078125 + (1.0 - 2_f64.powi(-12)) * 2_f64.powi(-8);
        assert_eq!(out[0], f32_to_bf16(exact as f32));
        assert_eq!(out[0], 0x3f82);
        let rounded_post = [bf16_to_f32(f32_to_bf16(post[0])), 0.0];
        assert_eq!(rounded_post[0], 1.0);
        assert_eq!(
            expand_f32(q, &x, &y, &rounded_post, &comb).unwrap()[0],
            0x3f81
        );
        // The product is rounded in F32 before the add, and the stream sum
        // is an FMA chain from zero, as for the BF16 expand.
        let three = SinkhornHyperConnectionQuery::ExpandF32 {
            streams: 3,
            width: 1,
            capacity: 1,
        };
        let out = expand_f32(three, &[0x3f80, 0x3b80, 0x3b80], &[0], &[0.0; 3], &[1.0; 9]).unwrap();
        assert_eq!(out, vec![0x3f81; 3]);
        let signed = expand_f32(three, &[0x8000; 3], &[0x8000], &[1.0; 3], &[1.0; 9]).unwrap();
        assert_eq!(signed, vec![0x0000; 3]);
        // Shapes and the variant are checked.
        assert!(expand_f32(bf16_query, &x, &y, &post, &comb).is_err());
        assert!(expand_f32(q, &x, &y, &post, &comb[..3]).is_err());
        assert!(expand(q, &x, &y, &[0; 2], &[0; 4]).is_err());
    }

    #[test]
    fn the_exact_f32_expand_lies_within_its_f64_reference() {
        let (rows, n, h) = (3, 4, 96);
        let q = SinkhornHyperConnectionQuery::ExpandF32 {
            streams: 4,
            width: 96,
            capacity: 4,
        };
        let x = bits(&grid(rows * n * h, 51));
        let y = bits(
            &grid(rows * h, 52)
                .iter()
                .map(|v| 8.0 * v)
                .collect::<Vec<_>>(),
        );
        let post = grid(rows * n, 53)
            .iter()
            .map(|&v| (1.0 + v) as f32 + 2_f32.powi(-20))
            .collect::<Vec<_>>();
        let comb = grid(rows * n * n, 54)
            .iter()
            .map(|&v| (0.25 + v / 8.0) as f32 + 2_f32.powi(-21))
            .collect::<Vec<_>>();
        let exact = expand_f32(q, &x, &y, &post, &comb).unwrap();
        let reference = expand_f32_reference(
            q,
            &values_of(&x),
            &values_of(&y),
            &widen(&post),
            &widen(&comb),
        )
        .unwrap();
        check_numeric(&values_of(&exact), &reference, DType::Bf16).unwrap();
        // The BF16 expand of the rounded post and comb is a different op.
        let bf16_query = SinkhornHyperConnectionQuery::Expand {
            streams: 4,
            width: 96,
            capacity: 4,
        };
        let round = |v: &[f32]| v.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>();
        assert_ne!(
            expand(bf16_query, &x, &y, &round(&post), &round(&comb)).unwrap(),
            exact
        );
    }

    #[test]
    fn weighted_head_f32_preserves_low_bits_and_stream_order() {
        let q = SinkhornHyperConnectionQuery::Head {
            streams: 4,
            width: 3,
            mode: StreamHeadMode::SigmoidWeightedF32,
            capacity: 2,
        };
        // Column 0 loses 2^-24 before cancellation in stream order, but
        // retains it in reverse order. Column 1 lies halfway between BF16
        // values. Column 2 pins the +0 start for negative zero inputs.
        let x = [
            0x3f80, 0x3f80, 0x8000, 0x3380, 0x3b80, 0x8000, 0xbf80, 0x0000, 0x8000, 0x0000, 0x0000,
            0x8000,
        ];
        let pre = [1.0; 4];
        let out = weighted_head_f32(q, &x, &pre).unwrap();
        let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&out), [0, 1.00390625_f32.to_bits(), 0]);
        let rounded = out
            .iter()
            .map(|&v| bf16_to_f32(f32_to_bf16(v)))
            .collect::<Vec<_>>();
        let reversed = (0..3)
            .map(|col| {
                (0..4).rev().fold(0.0_f32, |sum, i| {
                    pre[i].mul_add(bf16_to_f32(x[i * 3 + col]), sum)
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            crate::oracle::check_exact(&bits(&rounded), &bits(&out)),
            Err(1)
        );
        assert_eq!(
            crate::oracle::check_exact(&bits(&reversed), &bits(&out)),
            Err(0)
        );
        let reference = weighted_head_reference(q, &values_of(&x), &pre).unwrap();
        check_numeric(&widen(&out), &reference, DType::F32).unwrap();
        assert!(check_numeric(&widen(&rounded), &reference, DType::F32).is_err());
        // The f64 bound alone cannot pin the stream order.
        check_numeric(&widen(&reversed), &reference, DType::F32).unwrap();
        assert!(weighted_head(q, &x, &pre).is_err());
        assert!(weighted_head_f32(q, &x[..11], &pre).is_err());
        assert!(weighted_head_f32(q, &x, &pre[..3]).is_err());
        assert!(weighted_head_f32(q, &x.repeat(3), &pre.repeat(3)).is_err());
        for mode in [
            StreamHeadMode::MeanOfStreams,
            StreamHeadMode::SigmoidWeighted,
        ] {
            let wrong = SinkhornHyperConnectionQuery::Head {
                streams: 4,
                width: 3,
                mode,
                capacity: 2,
            };
            assert!(weighted_head_f32(wrong, &x, &pre).is_err());
        }
    }

    #[test]
    fn weighted_head_f32_at_5120_lies_within_its_f64_bound() {
        let (rows, n, h) = (3, 4, 5120);
        let q = SinkhornHyperConnectionQuery::Head {
            streams: n as u32,
            width: h as u32,
            mode: StreamHeadMode::SigmoidWeightedF32,
            capacity: 8,
        };
        let x = bits(&grid(rows * n * h, 63));
        let pre = grid(rows * n, 64)
            .iter()
            .map(|&v| (0.5 + v / 2.0) as f32 + 1e-6)
            .collect::<Vec<_>>();
        let out = weighted_head_f32(q, &x, &pre).unwrap();
        let reference = weighted_head_reference(q, &values_of(&x), &pre).unwrap();
        check_numeric(&widen(&out), &reference, DType::F32).unwrap();
        let bf16_query = SinkhornHyperConnectionQuery::Head {
            streams: n as u32,
            width: h as u32,
            mode: StreamHeadMode::SigmoidWeighted,
            capacity: 8,
        };
        assert_eq!(
            out.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>(),
            weighted_head(bf16_query, &x, &pre).unwrap()
        );
        let rounded = out
            .iter()
            .map(|&v| bf16_to_f32(f32_to_bf16(v)))
            .collect::<Vec<_>>();
        assert!(check_numeric(&widen(&rounded), &reference, DType::F32).is_err());
    }

    #[test]
    fn the_weighted_head_collapses_with_the_last_pre() {
        let (rows, n, h) = (2, 4, 64);
        let q = SinkhornHyperConnectionQuery::Head {
            streams: 4,
            width: 64,
            mode: StreamHeadMode::SigmoidWeighted,
            capacity: 2,
        };
        let x = bits(&grid(rows * n * h, 61));
        let pre = grid(rows * n, 62)
            .iter()
            .map(|&v| (0.5 + v / 2.0) as f32 + 1e-6)
            .collect::<Vec<_>>();
        let out = weighted_head(q, &x, &pre).unwrap();
        // It is the mix-collapse's step 5 with the same weights.
        let collapse_query = lagged(mix(4, 64, 20, 1e-6));
        assert_eq!(out, collapse(collapse_query, &x, &pre).unwrap());
        let reference = weighted_head_reference(q, &values_of(&x), &pre).unwrap();
        check_numeric(&values_of(&out), &reference, DType::Bf16).unwrap();
        // The mean of streams is a different head.
        let mean = SinkhornHyperConnectionQuery::Head {
            streams: 4,
            width: 64,
            mode: StreamHeadMode::MeanOfStreams,
            capacity: 2,
        };
        assert_ne!(head(mean, &x).unwrap(), out);
        assert!(weighted_head(mean, &x, &pre).is_err());
        assert!(weighted_head(q, &x, &pre[..7]).is_err());
        // A one-hot pre selects stream 0.
        let one_hot = weighted_head(q, &x, &one_hot_pre(rows, n)).unwrap();
        for row in 0..rows {
            assert_eq!(&one_hot[row * h..][..h], &x[row * n * h..][..h]);
        }
    }
}
