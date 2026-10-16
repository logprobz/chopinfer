//! F64 reference arithmetic for the recurrent contracts:
//! causal depthwise convolution, the gated delta rule and gated RMSNorm.
//! Each call covers one sequence slot. Operands are decoded and row-major.
//!
//! Recurrent outputs have no single reduction. The gated delta rule's S terms
//! bound the rounding error an F32 recurrence accumulates over the rows
//! processed so far in the call, and K is `key_dim` times that row count; see
//! `gated_delta_rule` for the derivation.

use crate::contract::{
    ConvActivation, GateActivation, RecurrentPhase, RecurrentQuery, Refusal, Validate,
};
use crate::oracle::{NormOffset, OracleOutput, RmsNorm};

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// PyTorch softplus with beta 1 and threshold 20, as both references use.
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// `x * rsqrt(sum(x^2) + 1e-6)`, the FLA and HF L2 norm.
pub fn l2_normalize(x: &[f64]) -> Vec<f64> {
    let inverse = (x.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt().recip();
    x.iter().map(|v| v * inverse).collect()
}

/// Per-step decay exponent g = -exp(A_log) * softplus(a + dt_bias).
pub fn gated_delta_decay(a: f64, a_log: f64, dt_bias: f64) -> f64 {
    -a_log.exp() * softplus(a + dt_bias)
}

fn check_rows(phase: RecurrentPhase, rows: usize, capacity: u32) -> Result<(), Refusal> {
    let fits_phase = match phase {
        RecurrentPhase::Prefill => true,
        RecurrentPhase::Decode => rows == 1,
        RecurrentPhase::Verify { max_rows } => rows <= max_rows as usize,
    };
    if rows == 0 || rows > capacity as usize || !fits_phase {
        return Err(Refusal("row count does not fit the recurrent phase"));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvOutput {
    /// [rows, channels].
    pub output: OracleOutput,
    /// Slot state after the last row, [state_rows, channels]. Exact copies.
    pub state: Vec<f64>,
    /// Verify only: the slot state after each row.
    pub row_states: Vec<Vec<f64>>,
}

/// Input is [rows, channels], weight is [channels, kernel] (the checkpoint's
/// [channels, 1, kernel]) and state is [(kernel - 1) * dilation, channels].
/// SiLU is at most 1.1-Lipschitz, so S is 1.1 times the pre-activation sum.
pub fn causal_conv1d(
    query: RecurrentQuery,
    input: &[f64],
    weight: &[f64],
    state: &[f64],
) -> Result<ConvOutput, Refusal> {
    query.validate()?;
    let RecurrentQuery::CausalConv1d {
        channels,
        kernel,
        dilation,
        activation,
        phase,
        capacity,
    } = query
    else {
        return Err(Refusal("expected causal convolution query"));
    };
    let (c, taps, step) = (channels as usize, kernel as usize, dilation as usize);
    let history = (taps - 1) * step;
    if !input.len().is_multiple_of(c) || weight.len() != c * taps || state.len() != history * c {
        return Err(Refusal("causal convolution operand shape mismatch"));
    }
    let rows = input.len() / c;
    check_rows(phase, rows, capacity)?;
    let window = [state, input].concat();
    let mut values = Vec::with_capacity(input.len());
    let mut bounds = Vec::with_capacity(input.len());
    for t in 0..rows {
        let newest = history + t;
        for ch in 0..c {
            let mut sum = 0.0;
            let mut abs_sum = 0.0;
            for tap in 0..taps {
                let row = newest - (taps - 1 - tap) * step;
                let product = weight[ch * taps + tap] * window[row * c + ch];
                sum += product;
                abs_sum += product.abs();
            }
            match activation {
                ConvActivation::Identity => {
                    values.push(sum);
                    bounds.push(abs_sum);
                }
                ConvActivation::Silu => {
                    values.push(silu(sum));
                    bounds.push(1.1 * abs_sum);
                }
            }
        }
    }
    let tail = |end: usize| window[(end - history) * c..end * c].to_vec();
    let row_states = if matches!(phase, RecurrentPhase::Verify { .. }) {
        (1..=rows).map(|r| tail(history + r)).collect()
    } else {
        Vec::new()
    };
    Ok(ConvOutput {
        output: OracleOutput::new(values, bounds, taps)?,
        state: tail(history + rows),
        row_states,
    })
}

/// Operands of one sequence after the convolution. q and k are
/// [rows, key_heads, key_dim], v is [rows, value_heads, value_dim], a and b
/// are `[rows, value_heads]`, A_log and dt_bias are `[value_heads]`, and the
/// state is [value_heads, value_dim, key_dim].
#[derive(Clone, Copy, Debug)]
pub struct GatedDeltaInputs<'a> {
    pub q: &'a [f64],
    pub k: &'a [f64],
    pub v: &'a [f64],
    pub a: &'a [f64],
    pub b: &'a [f64],
    pub a_log: &'a [f64],
    pub dt_bias: &'a [f64],
    pub state: &'a [f64],
}

#[derive(Clone, Debug, PartialEq)]
pub struct GatedDeltaOutput {
    /// [rows, value_heads, value_dim].
    pub output: OracleOutput,
    /// Slot state after the last row.
    pub state: OracleOutput,
    /// Verify only: the slot state after each row.
    pub row_states: Vec<OracleOutput>,
}

struct DeltaGeometry {
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    rows: usize,
    phase: RecurrentPhase,
}

fn delta_geometry(
    query: RecurrentQuery,
    inputs: &GatedDeltaInputs<'_>,
) -> Result<DeltaGeometry, Refusal> {
    query.validate()?;
    let RecurrentQuery::GatedDeltaRule {
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        phase,
        capacity,
    } = query
    else {
        return Err(Refusal("expected gated delta rule query"));
    };
    let (kh, vh, kd, vd) = (
        key_heads as usize,
        value_heads as usize,
        key_dim as usize,
        value_dim as usize,
    );
    let rows = inputs.a.len() / vh;
    if inputs.a.len() != rows * vh
        || inputs.b.len() != rows * vh
        || inputs.q.len() != rows * kh * kd
        || inputs.k.len() != rows * kh * kd
        || inputs.v.len() != rows * vh * vd
        || inputs.a_log.len() != vh
        || inputs.dt_bias.len() != vh
        || inputs.state.len() != vh * vd * kd
    {
        return Err(Refusal("gated delta rule operand shape mismatch"));
    }
    check_rows(phase, rows, capacity)?;
    Ok(DeltaGeometry {
        key_heads: kh,
        value_heads: vh,
        key_dim: kd,
        value_dim: vd,
        rows,
        phase,
    })
}

/// Normalized and scaled q, normalized k, beta and g for one row and value head.
fn delta_step_operands(
    geometry: &DeltaGeometry,
    inputs: &GatedDeltaInputs<'_>,
    row: usize,
    head: usize,
) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let kd = geometry.key_dim;
    let key_head = head / (geometry.value_heads / geometry.key_heads);
    let base = (row * geometry.key_heads + key_head) * kd;
    let scale = (kd as f64).sqrt().recip();
    let q = l2_normalize(&inputs.q[base..base + kd])
        .into_iter()
        .map(|x| x * scale)
        .collect();
    let k = l2_normalize(&inputs.k[base..base + kd]);
    let index = row * geometry.value_heads + head;
    let beta = sigmoid(inputs.b[index]);
    let g = gated_delta_decay(inputs.a[index], inputs.a_log[head], inputs.dt_bias[head]);
    (q, k, beta, g)
}

/// The token-by-token recurrence, which defines every phase:
/// `S *= exp(g); d = beta * (v - S k); S += d k^T; o = S q`.
///
/// Error scale. Each value row s of the state (a vector over keys) takes the
/// step `s -> s M + beta v k^T` with `M = gamma (I - beta k k^T)`, where
/// `gamma = exp(g)` lies in (0, 1], `beta = sigmoid(b)` in (0, 1), and
/// `||k||_2 < 1` because k is L2-normalized with eps. M is symmetric with
/// eigenvalues gamma (orthogonal to k) and `gamma (1 - beta ||k||^2)`, so its
/// spectral norm is at most gamma: an error e carried into step t leaves it
/// with norm at most `gamma_t ||e||`, and errors decay instead of compounding.
///
/// Step t itself rounds terms whose 2-norm over keys is at most
/// `gamma ||s|| + beta (|v| + gamma ||s||) <= 2 ||s_{t-1}|| + |v_t|` (the
/// product `s . k` has a sum of absolute products of at most `||s|| ||k||`
/// by Cauchy-Schwarz), with reductions of `key_dim` terms, where s_{t-1} is
/// the exact state this oracle holds. An F32 step therefore adds an error of
/// norm at most `c u key_dim (2 ||s_{t-1}|| + |v_t|)` to first order in
/// u = 2^-24, with c a small constant, and after t steps the row's error norm
/// is at most `c u key_dim W_t` with
/// `W_t = gamma_t W_{t-1} + 2 ||s_{t-1}|| + |v_t|`. Every state element is
/// bounded by its row's norm, so it gets `S = W_t / t` with `K = key_dim t`,
/// and the numerical bound `2^-22 K S = 4 u key_dim W_t` covers c up to 4. The
/// output `o_t = s_t . q_t` adds `||q_t||` times the state error and its own
/// reduction over a sum of at most `||s_t|| ||q_t||`, so it gets
/// `S = ||q_t|| (W_t + ||s_t||) / t`. W_t grows at most linearly in the rows
/// (per-element magnitude propagation grows geometrically), and it saturates
/// near `(2 ||s|| + |v|) / (1 - gamma)` for heads that decay.
pub fn gated_delta_rule(
    query: RecurrentQuery,
    inputs: GatedDeltaInputs<'_>,
) -> Result<GatedDeltaOutput, Refusal> {
    let geometry = delta_geometry(query, &inputs)?;
    let (vh, kd, vd, rows) = (
        geometry.value_heads,
        geometry.key_dim,
        geometry.value_dim,
        geometry.rows,
    );
    let norm2 = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut state = inputs.state.to_vec();
    // W_t per value row.
    let mut accumulated = vec![0.0; vh * vd];
    let state_scale = |accumulated: &[f64], steps: usize| -> Vec<f64> {
        accumulated
            .iter()
            .flat_map(|&w| std::iter::repeat_n(w / steps as f64, kd))
            .collect()
    };
    let mut values = vec![0.0; rows * vh * vd];
    let mut bounds = vec![0.0; rows * vh * vd];
    let mut lengths = vec![0; rows * vh * vd];
    let mut row_states = Vec::new();
    for row in 0..rows {
        let steps = (row + 1) as f64;
        for head in 0..vh {
            let (q, k, beta, g) = delta_step_operands(&geometry, &inputs, row, head);
            let decay = g.exp();
            let q_norm = norm2(&q);
            let head_base = head * vd * kd;
            for v in 0..vd {
                let s = &mut state[head_base + v * kd..head_base + (v + 1) * kd];
                let previous = norm2(s);
                let mut memory = 0.0;
                for i in 0..kd {
                    s[i] *= decay;
                    memory += s[i] * k[i];
                }
                let value = inputs.v[(row * vh + head) * vd + v];
                let delta = beta * (value - memory);
                let mut out = 0.0;
                for i in 0..kd {
                    s[i] += k[i] * delta;
                    out += s[i] * q[i];
                }
                let r = head * vd + v;
                accumulated[r] = decay * accumulated[r] + 2.0 * previous + value.abs();
                let index = (row * vh + head) * vd + v;
                values[index] = out;
                bounds[index] = q_norm * (accumulated[r] + norm2(s)) / steps;
                lengths[index] = kd * (row + 1);
            }
        }
        if matches!(geometry.phase, RecurrentPhase::Verify { .. }) {
            row_states.push(OracleOutput::new(
                state.clone(),
                state_scale(&accumulated, row + 1),
                kd * (row + 1),
            )?);
        }
    }
    Ok(GatedDeltaOutput {
        output: OracleOutput {
            values,
            sum_abs_products: bounds,
            reduction_lengths: lengths,
        },
        state: OracleOutput::new(state, state_scale(&accumulated, rows), kd * rows)?,
        row_states,
    })
}

/// Input and gate are `[rows, width]`, weight is `[head_dim]`. Each head computes
/// `x * rsqrt(mean(x^2) + eps) * w * act(gate)`.
///
/// Error scale, in output units because the normalization removes the input
/// scale. In F32 with u = 2^-24, the head's sum of squares has only
/// nonnegative terms, so its relative error is at most `head_dim u`; adding
/// eps (exact and positive) can only shrink that relative error, so
/// `mean + eps` is within `(head_dim + 1) u`. The reciprocal square root
/// halves it and adds a few ulps of its own, and the three products and the
/// gate activation (an exponential, an add and a divide) add a few more. The
/// output's relative error is therefore at most about `(head_dim / 2 + 11) u`
/// whatever the input scale, and `S = |y|` with `K = head_dim` gives the numerical
/// bound `2^-22 K S = 4 u head_dim |y|`, which covers it for any head_dim of
/// 4 or more.
pub fn gated_rmsnorm(
    query: RecurrentQuery,
    input: &[f64],
    gate: &[f64],
    weight: &[f64],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let RecurrentQuery::GatedRmsNorm {
        width,
        head_dim,
        eps_bits,
        gate: activation,
        capacity,
    } = query
    else {
        return Err(Refusal("expected gated RMSNorm query"));
    };
    let (width, d) = (width as usize, head_dim as usize);
    if input.is_empty()
        || !input.len().is_multiple_of(width)
        || input.len() / width > capacity as usize
        || gate.len() != input.len()
        || weight.len() != d
    {
        return Err(Refusal("gated RMSNorm operand shape mismatch"));
    }
    let eps = f64::from(f32::from_bits(eps_bits));
    let normed = RmsNorm::f64(eps, NormOffset::Plain).apply(input, weight, d)?;
    let values = normed
        .values
        .iter()
        .zip(gate)
        .map(|(&y, &z)| {
            y * match activation {
                GateActivation::Sigmoid => sigmoid(z),
                GateActivation::Silu => silu(z),
            }
        })
        .collect::<Vec<_>>();
    let bounds = values.iter().map(|y: &f64| y.abs()).collect();
    OracleOutput::new(values, bounds, d)
}
