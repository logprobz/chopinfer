//! F64 references for the encoder row passes `multimodal_oracle` lacks: the
//! per-head q and k RMSNorm of fused q/k/v rows, the clamped SwiGLU, the
//! F32 GELU row pass and the BF16 cast.
//! Inputs are decoded operands, in row-major order.

use crate::contract::{GeluMode, Refusal};
use crate::multimodal_oracle::gelu;
use crate::oracle::OracleOutput;

/// RMSNorm of every `head_dim` slice of the q and k columns of fused
/// `[q | k | v]` rows (`q_heads`, then `kv_heads` of k, then `kv_heads` of
/// v): `x / sqrt(mean(x^2) + eps) * w`, with `w` the `q_weight` or
/// `k_weight` of `head_dim` values. v passes through unchanged. S is the
/// output's magnitude and K the head dim, as for
/// [`crate::multimodal_oracle::rms_norm`].
pub fn qk_rms_norm(
    qkv: &[f64],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    q_weight: &[f64],
    k_weight: &[f64],
    eps: f64,
) -> Result<OracleOutput, Refusal> {
    let width = (q_heads + 2 * kv_heads) * head_dim;
    if q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || qkv.is_empty()
        || !qkv.len().is_multiple_of(width)
        || q_weight.len() != head_dim
        || k_weight.len() != head_dim
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Refusal("q/k RMSNorm shape or epsilon mismatch"));
    }
    let mut values = Vec::with_capacity(qkv.len());
    for row in qkv.chunks_exact(width) {
        for (head, x) in row.chunks_exact(head_dim).enumerate() {
            let weight = match head {
                head if head < q_heads => q_weight,
                head if head < q_heads + kv_heads => k_weight,
                _ => {
                    values.extend_from_slice(x);
                    continue;
                }
            };
            let mean = x.iter().map(|x| x * x).sum::<f64>() / head_dim as f64;
            let inverse = (mean + eps).sqrt().recip();
            values.extend(x.iter().zip(weight).map(|(&x, &w)| x * inverse * w));
        }
    }
    let bounds = values.iter().map(|x| x.abs()).collect();
    OracleOutput::new(values, bounds, head_dim)
}

/// The clamped SwiGLU of rows `[gate | up]` of `2 * hidden` values:
/// `silu(min(gate, limit)) * clamp(up, -limit, limit)`. A NaN stays NaN, as
/// in torch's clamp. S is the output's magnitude and K is 4, as for
/// [`crate::multimodal_oracle::swiglu_rows`]; the clamps are exact.
pub fn clamped_swiglu_rows(
    input: &[f64],
    hidden: usize,
    limit: f64,
) -> Result<OracleOutput, Refusal> {
    if hidden == 0
        || input.is_empty()
        || !input.len().is_multiple_of(2 * hidden)
        || !limit.is_finite()
        || limit <= 0.0
    {
        return Err(Refusal("clamped SwiGLU rows shape or limit mismatch"));
    }
    let values = input
        .chunks_exact(2 * hidden)
        .flat_map(|row| {
            let (gate, up) = row.split_at(hidden);
            gate.iter().zip(up).map(|(&g, &u)| {
                let g = if g > limit { limit } else { g };
                g / (1.0 + (-g).exp()) * u.clamp(-limit, limit)
            })
        })
        .collect::<Vec<_>>();
    let bounds = values.iter().map(|x| x.abs()).collect();
    OracleOutput::new(values, bounds, 4)
}

/// Rows of `width` values copied into rows of `columns`, zero past
/// `width`: exact, so S is the magnitude and K is 1.
pub fn pad_columns(input: &[f64], width: usize, columns: usize) -> Result<OracleOutput, Refusal> {
    if width == 0 || columns <= width || input.is_empty() || !input.len().is_multiple_of(width) {
        return Err(Refusal("padded row widths mismatch"));
    }
    let values: Vec<f64> = input
        .chunks_exact(width)
        .flat_map(|row| {
            row.iter()
                .copied()
                .chain(std::iter::repeat_n(0.0, columns - width))
        })
        .collect();
    let bounds = values.iter().map(|x| x.abs()).collect();
    OracleOutput::new(values, bounds, 1)
}

/// The BF16 bits of F32 rows rounded once to nearest even
/// (`CastBf16F32`): each value's exact nearest BF16, ties to the even
/// code, magnitudes from halfway past the largest finite BF16 to infinity,
/// and every NaN the canonical `0x7fff`. A cast is exact arithmetic on its
/// input, so the check compares bits.
pub fn cast_bf16_rows(input: &[f32]) -> Vec<u16> {
    input
        .iter()
        .map(|&x| {
            if x.is_nan() {
                0x7fff
            } else {
                crate::codec::f64_to_bf16(f64::from(x))
            }
        })
        .collect()
}

/// Exact-erf GELU of F32 rows, as an F32 row pass computes it
/// (`0.5 x (1 + erf(x / sqrt 2))`, through erfc below -1). K is 8: the F32
/// erf or erfc's few units in the last place, the cancellation in `1 + erf`
/// above -1 (at most a factor of about 3) and the roundings of the
/// argument, the add and the products. S is the output's magnitude, times
/// `1 + x^2` below zero: the argument `x / sqrt 2` is rounded to F32
/// before erfc, and erfc's relative condition there is `x^2`.
pub fn gelu_rows_f32(input: &[f64]) -> Result<OracleOutput, Refusal> {
    if input.is_empty() {
        return Err(Refusal("empty GELU input"));
    }
    let values: Vec<f64> = input.iter().map(|&x| gelu(x, GeluMode::Erf)).collect();
    let bounds = values
        .iter()
        .zip(input)
        .map(|(v, &x)| v.abs() * if x < 0.0 { 1.0 + x * x } else { 1.0 })
        .collect();
    OracleOutput::new(values, bounds, 8)
}

/// The per-head RMSNorm then neox rotary of q (or k) heads at the PyTorch
/// reference's F32 rounding points, for F32 operands `x` `[rows, heads,
/// head_dim]`, `weight` `[head_dim]` and `cos`, `sin` `[rows, head_dim]` (the
/// angles of each pair repeated in both halves), `head_dim` a multiple of
/// 32. Per head: the mean of squares summed in F32 in the order PyTorch's
/// CPU float reduction takes for such a row (four accumulators of eight
/// lanes), then divided by `head_dim`;
/// `1 / sqrt(mean + eps)` rounded at each step; `weight * (x * r)`, each
/// product rounded. Per pair: `x * cos` and `rotate_half(x) * sin` each
/// rounded, then their sum. Unlike the exact [`qk_rms_norm`], this
/// reproduces the reference's own F32 run bit for bit, so a model can
/// measure its served values' distance from that run rather than from
/// exact arithmetic.
pub fn rms_rotary_f32_boundaries(
    x: &[f32],
    head_dim: usize,
    weight: &[f32],
    eps: f32,
    cos: &[f32],
    sin: &[f32],
) -> Result<Vec<f32>, Refusal> {
    let rows = cos.len() / head_dim.max(1);
    if head_dim == 0
        || !head_dim.is_multiple_of(32)
        || rows == 0
        || cos.len() != rows * head_dim
        || sin.len() != cos.len()
        || x.is_empty()
        || !x.len().is_multiple_of(rows * head_dim)
        || weight.len() != head_dim
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Refusal("F32 RMSNorm and rotary shape or epsilon mismatch"));
    }
    let heads = x.len() / (rows * head_dim);
    let half = head_dim / 2;
    let mut out = Vec::with_capacity(x.len());
    for (index, head) in x.chunks_exact(head_dim).enumerate() {
        let row = index / heads;
        // Four accumulators of eight lanes, vector `i / 8` into accumulator
        // `(i / 8) % 4`; the accumulators added in order, then the lanes.
        let mut lanes = [[0.0f32; 8]; 4];
        for (i, &value) in head.iter().enumerate() {
            lanes[(i / 8) % 4][i % 8] += value * value;
        }
        let mut total = lanes[0];
        for accumulator in &lanes[1..] {
            for (lane, value) in total.iter_mut().zip(accumulator) {
                *lane += value;
            }
        }
        let sum = total.iter().fold(0.0f32, |sum, &lane| sum + lane);
        let r = 1.0 / (sum / head_dim as f32 + eps).sqrt();
        let normed: Vec<f32> = head
            .iter()
            .zip(weight)
            .map(|(&v, &w)| w * (v * r))
            .collect();
        let (cos, sin) = (
            &cos[row * head_dim..][..head_dim],
            &sin[row * head_dim..][..head_dim],
        );
        out.extend((0..head_dim).map(|d| {
            let rotated = if d < half {
                -normed[d + half]
            } else {
                normed[d - half]
            };
            normed[d] * cos[d] + rotated * sin[d]
        }));
    }
    Ok(out)
}

/// Each component of an encoder is checked against a hand-written
/// computation that follows the PyTorch reference's math step by step (its
/// reshapes, permutes, convolutions and clamps written as index loops),
/// while the oracle side runs the declaration's decomposition: fused q/k/v
/// rows, concatenated gate and up rows, consecutive-row merges with
/// permuted weight columns, and the F32 rotary table. Both sides compute in
/// f64 without the reference's F32 casts (its RMSNorm and rotary cast to
/// F32 even for f64 inputs), so agreement here proves the decomposition,
/// not rounding parity; [`rms_rotary_f32_boundaries`] keeps those casts.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal_oracle::{gelu_rows, gemm_bias_epilogue, layernorm_affine};
    use crate::oracle::GemmShape;
    use crate::rotary::{AngleRule, Axes, ComputedTable};

    /// A fixed pseudo-random sequence in `[-scale, scale)`.
    fn values(seed: u64, count: usize, scale: f64) -> Vec<f64> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    #[track_caller]
    fn close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= 1e-12 * (1.0 + e.abs()),
                "element {index}: {a} != {e}"
            );
        }
    }

    /// `torch.nn.functional.linear`: `x @ w.T + b` over `[rows, k]` and
    /// `[n, k]`.
    fn linear(x: &[f64], w: &[f64], b: Option<&[f64]>, k: usize, n: usize) -> Vec<f64> {
        x.chunks_exact(k)
            .flat_map(|row| {
                (0..n).map(move |o| {
                    let dot: f64 = (0..k).map(|i| row[i] * w[o * k + i]).sum();
                    dot + b.map_or(0.0, |b| b[o])
                })
            })
            .collect()
    }

    fn gemm(x: &[f64], w: &[f64], b: Option<&[f64]>, k: usize, n: usize) -> Vec<f64> {
        let shape = GemmShape {
            m: x.len() / k,
            n,
            k,
            experts: 1,
        };
        gemm_bias_epilogue(x, w, shape, b, None, None, None)
            .unwrap()
            .values
    }

    /// The reference RMSNorm module's math over the last axis of `[.., d]`,
    /// unrounded: the module itself casts to F32 first.
    fn torch_rms(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
        let d = w.len();
        x.chunks_exact(d)
            .flat_map(|x| {
                let variance = x.iter().map(|v| v.powi(2)).sum::<f64>() / d as f64;
                let r = 1.0 / (variance + eps).sqrt();
                x.iter().zip(w).map(move |(v, w)| w * (v * r))
            })
            .collect()
    }

    /// The reference vision MLP: separate gate and up projections, the gate
    /// clamped above, the up clamped both ways, then SiLU and the down
    /// projection.
    #[allow(clippy::too_many_arguments)]
    fn torch_clamped_mlp(
        x: &[f64],
        gate: (&[f64], Option<&[f64]>),
        up: (&[f64], Option<&[f64]>),
        down: (&[f64], Option<&[f64]>),
        width: usize,
        hidden: usize,
        limit: f64,
    ) -> Vec<f64> {
        let g = linear(x, gate.0, gate.1, width, hidden);
        let u = linear(x, up.0, up.1, width, hidden);
        let act: Vec<f64> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| {
                let g = g.min(limit);
                let u = u.max(-limit).min(limit);
                g * (1.0 / (1.0 + (-g).exp())) * u
            })
            .collect();
        linear(&act, down.0, down.1, hidden, width)
    }

    /// The declaration's MLP: one gate/up GEMM over the concatenated
    /// weights and biases, the clamped SwiGLU row pass, the down GEMM.
    #[allow(clippy::too_many_arguments)]
    fn declared_clamped_mlp(
        x: &[f64],
        gate: (&[f64], Option<&[f64]>),
        up: (&[f64], Option<&[f64]>),
        down: (&[f64], Option<&[f64]>),
        width: usize,
        hidden: usize,
        limit: f64,
    ) -> Vec<f64> {
        let weight = [gate.0, up.0].concat();
        let bias = gate.1.zip(up.1).map(|(g, u)| [g, u].concat());
        let gate_up = gemm(x, &weight, bias.as_deref(), width, 2 * hidden);
        let product = clamped_swiglu_rows(&gate_up, hidden, limit).unwrap().values;
        gemm(&product, down.0, down.1, hidden, width)
    }

    #[test]
    fn vision_row_queries_validate_and_name_their_contracts() {
        use crate::contract::{MultimodalQuery, NumericalContract, Validate};
        let eps = 1e-5_f32.to_bits();
        let norm = |q_heads, kv_heads, head_dim, eps_bits| MultimodalQuery::QkRmsNormF32 {
            q_heads,
            kv_heads,
            head_dim,
            eps_bits,
            capacity: 32768,
        };
        assert_eq!(
            norm(16, 16, 64, eps)
                .numerical_contract_id()
                .unwrap()
                .as_str(),
            "row.qk-rmsnorm-f32.v1"
        );
        for bad in [
            norm(0, 16, 64, eps),
            norm(16, 6, 64, eps),
            norm(16, 16, 64, 0),
            norm(16, 16, 64, f32::NAN.to_bits()),
            // A packed width past u64, refused rather than wrapped.
            norm(u32::MAX, u32::MAX, 1_431_655_766, eps),
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
        // Head dims are math; a backend's kernels may serve fewer.
        for head_dim in [72, 288] {
            assert!(norm(16, 16, head_dim, eps).validate().is_ok());
        }
        let swiglu = |hidden, limit: f32| MultimodalQuery::SwigluClampedF32 {
            hidden,
            limit_bits: limit.to_bits(),
            capacity: 32768,
        };
        assert_eq!(
            swiglu(4096, 10.0).numerical_contract_id().unwrap().as_str(),
            "row.swiglu-clamped-f32.v1"
        );
        for bad in [
            swiglu(0, 10.0),
            swiglu(4096, 0.0),
            swiglu(4096, f32::INFINITY),
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
        let gelu = MultimodalQuery::GeluRowsF32 {
            width: 4096,
            capacity: 32768,
        };
        assert_eq!(
            gelu.numerical_contract_id().unwrap().as_str(),
            "row.gelu-erf-f32.v1"
        );
        assert!(
            MultimodalQuery::GeluRowsF32 {
                width: 0,
                capacity: 1
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn f32_boundaries_round_each_step() {
        let head_dim = 32;
        let x: Vec<f32> = (0..head_dim).map(|i| i as f32 - 7.5).collect();
        let ones = vec![1.0f32; head_dim];
        let zeros = vec![0.0f32; head_dim];
        // cos 1, sin 0: the rotation keeps the normalized head.
        let out = rms_rotary_f32_boundaries(&x, head_dim, &ones, 1e-5, &ones, &zeros).unwrap();
        let sum: f32 = x.iter().map(|v| v * v).sum();
        let r = 1.0 / (sum / head_dim as f32 + 1e-5).sqrt();
        for (o, v) in out.iter().zip(&x) {
            assert_eq!(*o, v * r);
        }
        // cos 0, sin 1: rotate_half, `[-x2, x1]`.
        let out = rms_rotary_f32_boundaries(&x, head_dim, &ones, 1e-5, &zeros, &ones).unwrap();
        assert_eq!(out[0], -(x[16] * r));
        assert_eq!(out[16], x[0] * r);
        assert!(
            rms_rotary_f32_boundaries(&x[..16], 16, &ones[..16], 1e-5, &ones[..16], &zeros[..16])
                .is_err()
        );
        assert!(rms_rotary_f32_boundaries(&x, head_dim, &ones, 0.0, &ones, &zeros).is_err());
    }

    #[test]
    fn f32_gelu_rows_widen_the_negative_tail() {
        let out = gelu_rows_f32(&[2.0, 0.0, -1.0, -5.0]).unwrap();
        assert_eq!(out.values[1], 0.0);
        assert_eq!(out.values[0], gelu(2.0, GeluMode::Erf));
        assert_eq!(out.sum_abs_products[0], out.values[0].abs());
        assert_eq!(out.sum_abs_products[2], 2.0 * out.values[2].abs());
        assert_eq!(out.sum_abs_products[3], 26.0 * out.values[3].abs());
        assert!(out.reduction_lengths.iter().all(|&k| k == 8));
        assert!(gelu_rows_f32(&[]).is_err());
    }

    #[test]
    fn clamped_swiglu_clamps_each_half() {
        let limit = 10.0;
        // gate | up for three columns: inside, above and below the limit.
        let row = [3.0, 12.0, -40.0, 2.0, 11.0, -15.0];
        let out = clamped_swiglu_rows(&row, 3, limit).unwrap().values;
        let silu = |g: f64| g / (1.0 + (-g).exp());
        close(
            &out,
            &[silu(3.0) * 2.0, silu(10.0) * 10.0, silu(-40.0) * -10.0],
        );
        assert!(
            clamped_swiglu_rows(&[f64::NAN, 1.0], 1, limit)
                .unwrap()
                .values[0]
                .is_nan()
        );
        for limit in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(clamped_swiglu_rows(&row, 3, limit).is_err());
        }
        assert!(clamped_swiglu_rows(&row, 2, 10.0).is_err());
    }

    #[test]
    fn clamped_mlp_with_bias_matches_the_reference_module() {
        let (rows, width, hidden, limit) = (5, 8, 6, 10.0);
        // Large enough that gates pass the limit and ups pass both bounds.
        let x = values(1, rows * width, 6.0);
        let (wg, bg) = (values(2, hidden * width, 1.0), values(3, hidden, 2.0));
        let (wu, bu) = (values(4, hidden * width, 1.0), values(5, hidden, 2.0));
        let (wd, bd) = (values(6, width * hidden, 1.0), values(7, width, 2.0));
        let gate_up = linear(
            &x,
            &[wg.clone(), wu.clone()].concat(),
            None,
            width,
            2 * hidden,
        );
        assert!(gate_up.iter().any(|&v| v > limit));
        assert!(gate_up.iter().any(|&v| v < -limit));
        let parts = (
            (&wg[..], Some(&bg[..])),
            (&wu[..], Some(&bu[..])),
            (&wd[..], Some(&bd[..])),
        );
        let reference = torch_clamped_mlp(&x, parts.0, parts.1, parts.2, width, hidden, limit);
        let declared = declared_clamped_mlp(&x, parts.0, parts.1, parts.2, width, hidden, limit);
        close(&declared, &reference);
    }

    #[test]
    fn qk_rms_norm_matches_the_reference_heads() {
        let (seq, heads, head_dim, eps) = (3, 2, 4, 1e-5);
        let width = 3 * heads * head_dim;
        let qkv = values(11, seq * width, 3.0);
        let (q_w, k_w) = (values(12, head_dim, 1.5), values(13, head_dim, 1.5));
        // reshape(seq, 3, heads, head_dim): q, k and v are planes 0, 1, 2;
        // the norms run over the last axis, v is untouched.
        let mut reference = qkv.clone();
        for s in 0..seq {
            for (plane, weight) in [(0, &q_w), (1, &k_w)] {
                for h in 0..heads {
                    let at = s * width + (plane * heads + h) * head_dim;
                    let normed = torch_rms(&qkv[at..at + head_dim], weight, eps);
                    reference[at..at + head_dim].copy_from_slice(&normed);
                }
            }
        }
        let out = qk_rms_norm(&qkv, heads, heads, head_dim, &q_w, &k_w, eps).unwrap();
        close(&out.values, &reference);
        // Grouped heads: 4 q heads share 2 k and v heads. Per row, heads
        // 0-3 take the q weight, 4-5 the k weight, and 6-7 (v) pass.
        let qkv = values(14, 2 * 8 * head_dim, 3.0);
        let out = qk_rms_norm(&qkv, 4, 2, head_dim, &q_w, &k_w, eps).unwrap();
        let reference: Vec<f64> = qkv
            .chunks_exact(head_dim)
            .enumerate()
            .flat_map(|(slice, x)| match slice % 8 {
                0..4 => torch_rms(x, &q_w, eps),
                4..6 => torch_rms(x, &k_w, eps),
                _ => x.to_vec(),
            })
            .collect();
        close(&out.values, &reference);
        assert!(qk_rms_norm(&qkv, 4, 2, head_dim, &q_w, &k_w, 0.0).is_err());
        assert!(qk_rms_norm(&qkv, 4, 2, head_dim, &q_w[..3], &k_w, eps).is_err());
    }

    /// A patch embedding `Conv3d(c, out, kernel = stride = (t, p, p))` over
    /// patches viewed as `[n, c, t, p, p]` is a linear over the flattened
    /// patch with the weight flattened the same way.
    #[test]
    fn a_stride_equal_conv3d_is_a_linear_over_flattened_patches() {
        let (n, c, t, p, out) = (3, 3, 2, 2, 5);
        let patch = c * t * p * p;
        let x = values(21, n * patch, 1.0);
        let w = values(22, out * patch, 1.0);
        let b = values(23, out, 1.0);
        let mut reference = Vec::new();
        for i in 0..n {
            for o in 0..out {
                let mut sum = b[o];
                for ci in 0..c {
                    for ti in 0..t {
                        for y in 0..p {
                            for z in 0..p {
                                let at = ((ci * t + ti) * p + y) * p + z;
                                sum += w[o * patch + at] * x[i * patch + at];
                            }
                        }
                    }
                }
                reference.push(sum);
            }
        }
        close(&gemm(&x, &w, Some(&b), patch, out), &reference);
    }

    /// The patch payload padded to a multiple of 64 columns, with the patch
    /// weight padded the same way, gives the unpadded patch GEMM bit for
    /// bit: every added product is an exact zero, and the reduction's other
    /// terms and their order are unchanged.
    #[test]
    fn zero_padded_patches_give_the_unpadded_gemm_bitwise() {
        let (rows, patch, padded, out) = (5, 1176, 1216, 24);
        let x = values(51, rows * patch, 1.0);
        let w = values(52, out * patch, 0.05);
        let b = values(53, out, 0.5);
        let x_padded = pad_columns(&x, patch, padded).unwrap().values;
        assert!(
            x_padded
                .chunks_exact(padded)
                .all(|row| row[patch..].iter().all(|&v| v == 0.0))
        );
        let w_padded = pad_columns(&w, patch, padded).unwrap().values;
        let shape = |k| GemmShape {
            m: rows,
            n: out,
            k,
            experts: 1,
        };
        let plain = gemm_bias_epilogue(&x, &w, shape(patch), Some(&b), None, None, None).unwrap();
        let wide = gemm_bias_epilogue(
            &x_padded,
            &w_padded,
            shape(padded),
            Some(&b),
            None,
            None,
            None,
        )
        .unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&wide.values), bits(&plain.values));
        assert_eq!(bits(&wide.sum_abs_products), bits(&plain.sum_abs_products));
        assert!(pad_columns(&x, patch, patch).is_err());
    }

    /// The cast rounds ties to the even code, at normal and subnormal
    /// magnitudes alike, overflows to infinity from halfway past the
    /// largest finite BF16, keeps infinities and signed zeros, and gives
    /// every NaN the canonical code.
    #[test]
    fn the_bf16_cast_rounds_once_to_nearest_even() {
        let cases: [(u32, u16); 16] = [
            (0x3f80_0000, 0x3f80),
            (0x3f80_8000, 0x3f80),
            (0x3f81_8000, 0x3f82),
            (0x3f80_8001, 0x3f81),
            (0xbf81_8000, 0xbf82),
            (0x0000_8000, 0x0000),
            (0x0001_8000, 0x0002),
            (0x8000_0000, 0x8000),
            (0x7f7f_7fff, 0x7f7f),
            (0x7f7f_8000, 0x7f80),
            (0xff7f_ffff, 0xff80),
            (0x7f80_0000, 0x7f80),
            (0xff80_0000, 0xff80),
            (0x7fc0_0001, 0x7fff),
            (0xff80_0001, 0x7fff),
            (0x7f80_0001, 0x7fff),
        ];
        let input: Vec<f32> = cases.iter().map(|&(x, _)| f32::from_bits(x)).collect();
        let expected: Vec<u16> = cases.iter().map(|&(_, code)| code).collect();
        assert_eq!(cast_bf16_rows(&input), expected);
    }

    /// The reference downsample: rows of `m x m` blocks viewed as `[n, m,
    /// m, c]`, permuted to `[n, c, m, m]` and convolved with `[out, c, m,
    /// m]` at stride `m`. The declaration merges `m * m` consecutive rows
    /// into one position-major row and multiplies it by the weight whose
    /// flattened columns move from channel-major `c * m * m + ky * m + kx`
    /// to position-major `(ky * m + kx) * c + c` (`space_to_depth_cols`).
    #[test]
    fn a_block_merge_and_permuted_linear_is_the_strided_conv2d() {
        let (blocks, m, c, out) = (3, 2, 4, 6);
        let cell = m * m;
        let x = values(31, blocks * cell * c, 1.0);
        let w = values(32, out * c * cell, 1.0);
        let b = values(33, out, 1.0);
        let mut reference = Vec::new();
        for n in 0..blocks {
            for o in 0..out {
                let mut sum = b[o];
                for ci in 0..c {
                    for ky in 0..m {
                        for kx in 0..m {
                            // view(-1, m, m, c)[n, ky, kx, ci], then permute.
                            let input = x[((n * m + ky) * m + kx) * c + ci];
                            sum += w[((o * c + ci) * m + ky) * m + kx] * input;
                        }
                    }
                }
                reference.push(sum);
            }
        }
        let mut permuted = vec![0.0; w.len()];
        for o in 0..out {
            for ci in 0..c {
                for position in 0..cell {
                    permuted[o * c * cell + position * c + ci] =
                        w[o * c * cell + ci * cell + position];
                }
            }
        }
        // The merged rows are the same values read `cell * c` wide.
        close(&gemm(&x, &permuted, Some(&b), cell * c, out), &reference);
    }

    /// The reference projector: a bias-free linear, LayerNorm with weight
    /// and bias, exact GELU, then the clamped MLP without biases. The
    /// declaration runs the linear stage, then the norm, the GELU row pass
    /// and the MLP slot of the projector stage.
    #[test]
    fn the_projector_matches_the_reference_merger() {
        let (rows, dim, context, limit, eps) = (4, 8, 12, 10.0, 1e-5);
        let x = values(41, rows * dim, 4.0);
        let proj = values(42, dim * dim, 1.0);
        let (norm_w, norm_b) = (values(43, dim, 1.5), values(44, dim, 0.5));
        let wg = values(45, context * dim, 8.0);
        let wu = values(46, context * dim, 8.0);
        let wd = values(47, dim * context, 1.0);
        let mlp = ((&wg[..], None), (&wu[..], None), (&wd[..], None));

        let hidden = linear(&x, &proj, None, dim, dim);
        let normed: Vec<f64> = hidden
            .chunks_exact(dim)
            .flat_map(|row| {
                let mean = row.iter().sum::<f64>() / dim as f64;
                let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / dim as f64;
                let r = 1.0 / (var + eps).sqrt();
                row.iter()
                    .zip(&norm_w)
                    .zip(&norm_b)
                    .map(move |((v, w), b)| (v - mean) * r * w + b)
            })
            .collect();
        let activated: Vec<f64> = normed.iter().map(|&v| gelu(v, GeluMode::Erf)).collect();
        let reference = torch_clamped_mlp(&activated, mlp.0, mlp.1, mlp.2, dim, context, limit);

        let declared = gemm(&x, &proj, None, dim, dim);
        let declared = layernorm_affine(&declared, dim, &norm_w, &norm_b, eps)
            .unwrap()
            .values;
        let declared = gelu_rows(&declared).unwrap().values;
        let gate_up = linear(
            &declared,
            &[wg.clone(), wu.clone()].concat(),
            None,
            dim,
            2 * context,
        );
        assert!(gate_up.iter().any(|&v| v > limit) && gate_up.iter().any(|&v| v < -limit));
        let declared = declared_clamped_mlp(&declared, mlp.0, mlp.1, mlp.2, dim, context, limit);
        close(&declared, &reference);
    }

    /// The reference vision rotary: `inv_freq = 1 / theta^(arange(0, d, 2)
    /// / d)` with `d = head_dim / 2`, block-major (row, column) positions,
    /// angles `position * inv_freq` laid out as all row angles then all
    /// column angles, duplicated, and rotated by `rotate_half`. Pair `i`
    /// (dimensions `i` and `i + head_dim / 2`) then follows axis `i / (d /
    /// 2)` at frequency `i % (d / 2)`: the axial table's layout and angle
    /// rule.
    #[test]
    fn the_axial_table_follows_the_reference_rotary_layout() {
        let (head_dim, theta) = (64, 10000.0);
        let d = head_dim / 2;
        let table = ComputedTable {
            theta,
            rule: AngleRule::MulRecipF64PowF32,
            axes: Axes::Axial2d,
        };
        let inv_freq: Vec<f32> = (0..d)
            .step_by(2)
            .map(|step| 1.0 / f64::powf(theta, f64::from(step as f32 / d as f32)) as f32)
            .collect();
        assert_eq!(inv_freq.len(), d / 2);
        // A 4 x 6 grid in 2 x 2 blocks, as the reference orders it.
        let mut positions = Vec::new();
        for bh in 0..2 {
            for bw in 0..3 {
                for y in 0..2 {
                    for x in 0..2 {
                        positions.push([bh * 2 + y, bw * 2 + x]);
                    }
                }
            }
        }
        for &[row, column] in &positions {
            let angles: Vec<f32> = [row, column]
                .iter()
                .flat_map(|&p| inv_freq.iter().map(move |f| p as f32 * f))
                .collect();
            let emb: Vec<f32> = angles.iter().chain(&angles).copied().collect();
            let entries = table.row([row, column, 0], d).unwrap();
            for (pair, (cos, sin)) in entries.into_iter().enumerate() {
                let angle = f64::from(emb[pair]);
                assert_eq!(emb[pair], emb[pair + d]);
                assert_eq!((cos, sin), (angle.cos(), angle.sin()), "pair {pair}");
            }
        }
    }
}
