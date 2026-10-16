//! F64 reference arithmetic for the per-channel-decay delta rule contracts
//! (`per_channel_decay_delta_rule`): the short q, k, v convolutions, the
//! forget gate in both forms and the token-by-token recurrence that defines
//! every phase. Each call covers one sequence slot. Operands are decoded and
//! row-major.
//!
//! The rounding points the contract states are applied here: beta rounds to
//! BF16 before it enters the recurrence. Everything else is exact f64 over
//! the decoded BF16 and F32 operands; the stored outputs' BF16 rounding is
//! the comparison's half-ulp term. Long reductions use Neumaier-compensated
//! sums.

use crate::contract::{ConvActivation, RecurrentPhase, RecurrentQuery, Refusal, Validate};
use crate::oracle::{OracleOutput, bf16_round, compensated_sum};
use crate::per_channel_decay_delta_rule::{DecayGate, PerChannelDecayDeltaRuleQuery, QkvConvs};
use crate::recurrent_oracle::{ConvOutput, causal_conv1d, l2_normalize, sigmoid};

fn dot(x: &[f64], y: &[f64]) -> f64 {
    compensated_sum(x.iter().zip(y).map(|(a, b)| a * b))
}

fn norm2(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

/// PyTorch's literal `where(x > 20, x, log(1 + exp(x)))`, as the reference's
/// forget gate writes it.
pub fn gate_softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// The per-step log decay g of one channel: `f` the gate projection output,
/// `dt_bias` the channel's bias and `a_log` the head's `A_log`.
pub fn log_decay(gate: DecayGate, f: f64, dt_bias: f64, a_log: f64) -> f64 {
    let (x, a) = (f + dt_bias, a_log.exp());
    match gate {
        DecayGate::BoundedSigmoid { lower_bound_bits } => {
            f64::from(f32::from_bits(lower_bound_bits)) * sigmoid(a * x)
        }
        DecayGate::Softplus => -a * gate_softplus(x),
    }
}

/// beta as the reference computes it: `sigmoid` of the BF16 logit, rounded
/// to BF16.
pub fn beta(logit: f64) -> f64 {
    bf16_round(sigmoid(logit))
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

/// The q, k and v convolutions with SiLU. Input is `[rows, 3 * heads *
/// head_dim]`, q then k then v; `weights` is one `[3 * heads * head_dim,
/// kernel]` weight for fused convolutions or the q, k and v weights `[heads *
/// head_dim, kernel]` for separate ones; state is `[kernel - 1, 3 * heads *
/// head_dim]`, oldest first. The math is one depthwise convolution over all
/// channels (`recurrent_oracle::causal_conv1d`), whose S terms apply.
pub fn short_conv(
    query: PerChannelDecayDeltaRuleQuery,
    input: &[f64],
    weights: &[&[f64]],
    state: &[f64],
) -> Result<ConvOutput, Refusal> {
    query.validate()?;
    let PerChannelDecayDeltaRuleQuery::ShortConv {
        heads,
        head_dim,
        kernel,
        convs,
        phase,
        capacity,
    } = query
    else {
        return Err(Refusal("expected short convolution query"));
    };
    let width = (heads * head_dim) as usize * kernel as usize;
    let parts = match convs {
        QkvConvs::Fused => 1,
        QkvConvs::Separate => 3,
    };
    if weights.len() != parts || weights.iter().any(|w| w.len() * parts != 3 * width) {
        return Err(Refusal("short convolution weight shape mismatch"));
    }
    causal_conv1d(
        RecurrentQuery::CausalConv1d {
            channels: 3 * heads * head_dim,
            kernel,
            dilation: 1,
            activation: ConvActivation::Silu,
            phase,
            capacity,
        },
        input,
        &weights.concat(),
        state,
    )
}

/// Operands of one sequence after the convolutions: q, k and v
/// `[rows, heads, head_dim]`, the beta logits `b` `[rows, heads]`, the gate
/// projection output `f` `[rows, heads * head_dim]`, `a_log` `[heads]`,
/// `dt_bias` `[heads * head_dim]` and the state `[heads, head_dim (value),
/// head_dim (key)]`.
#[derive(Clone, Copy, Debug)]
pub struct DeltaRuleInputs<'a> {
    pub q: &'a [f64],
    pub k: &'a [f64],
    pub v: &'a [f64],
    pub b: &'a [f64],
    pub f: &'a [f64],
    pub a_log: &'a [f64],
    pub dt_bias: &'a [f64],
    pub state: &'a [f64],
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeltaRuleOutput {
    /// `[rows, heads, head_dim]`.
    pub output: OracleOutput,
    /// Slot state after the last row.
    pub state: OracleOutput,
    /// Verify only: the slot state after each row; commit keeps row
    /// `kept - 1`.
    pub row_states: Vec<OracleOutput>,
}

struct Geometry {
    heads: usize,
    dim: usize,
    rows: usize,
    gate: DecayGate,
    phase: RecurrentPhase,
}

fn geometry(
    query: PerChannelDecayDeltaRuleQuery,
    inputs: &DeltaRuleInputs<'_>,
) -> Result<Geometry, Refusal> {
    query.validate()?;
    let PerChannelDecayDeltaRuleQuery::DeltaRule {
        heads,
        head_dim,
        gate,
        phase,
        capacity,
    } = query
    else {
        return Err(Refusal("expected per-channel-decay delta rule query"));
    };
    let (h, d) = (heads as usize, head_dim as usize);
    let rows = inputs.b.len() / h;
    if inputs.b.len() != rows * h
        || inputs.q.len() != rows * h * d
        || inputs.k.len() != rows * h * d
        || inputs.v.len() != rows * h * d
        || inputs.f.len() != rows * h * d
        || inputs.a_log.len() != h
        || inputs.dt_bias.len() != h * d
        || inputs.state.len() != h * d * d
    {
        return Err(Refusal(
            "per-channel-decay delta rule operand shape mismatch",
        ));
    }
    check_rows(phase, rows, capacity)?;
    Ok(Geometry {
        heads: h,
        dim: d,
        rows,
        gate,
        phase,
    })
}

/// One row and head: q normalized then scaled by `head_dim^-0.5`, k
/// normalized, v, beta and g per key channel.
struct Step {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    beta: f64,
    g: Vec<f64>,
}

fn step(geometry: &Geometry, inputs: &DeltaRuleInputs<'_>, row: usize, head: usize) -> Step {
    let d = geometry.dim;
    let base = (row * geometry.heads + head) * d;
    let scale = (d as f64).sqrt().recip();
    Step {
        q: l2_normalize(&inputs.q[base..base + d])
            .into_iter()
            .map(|x| x * scale)
            .collect(),
        k: l2_normalize(&inputs.k[base..base + d]),
        v: inputs.v[base..base + d].to_vec(),
        beta: beta(inputs.b[row * geometry.heads + head]),
        g: (0..d)
            .map(|x| {
                log_decay(
                    geometry.gate,
                    inputs.f[base + x],
                    inputs.dt_bias[head * d + x],
                    inputs.a_log[head],
                )
            })
            .collect(),
    }
}

/// The token-by-token recurrence, which defines every phase:
/// `S[v][k] *= exp(g[k]); d = beta (v - S k); S += d k^T; o = S q`.
///
/// Error scale, as for `recurrent_oracle::gated_delta_rule`. Each value row
/// s of the state takes `s -> s M + beta v k^T` with
/// `M = diag(exp(g)) (I - beta k k^T)`. The diagonal has entries in (0, 1]
/// and `I - beta k k^T` has eigenvalues 1 and `1 - beta ||k||^2` in (0, 1]
/// (beta in [0, 1], `||k|| < 1`), so `||M||` is at most `gamma = max exp(g)`
/// and carried errors decay. Step t rounds terms of 2-norm at most
/// `2 ||s_{t-1}|| + |v_t|` in reductions of `head_dim` terms, so the row's
/// error after t steps is at most `c u head_dim W_t` with
/// `W_t = gamma_t W_{t-1} + 2 ||s_{t-1}|| + |v_t|`: S is `W_t / t` with
/// `K = head_dim t` for the state, and `||q_t|| (W_t + ||s_t||) / t` for the
/// output.
pub fn delta_rule(
    query: PerChannelDecayDeltaRuleQuery,
    inputs: DeltaRuleInputs<'_>,
) -> Result<DeltaRuleOutput, Refusal> {
    let geometry = geometry(query, &inputs)?;
    let (h, d, rows) = (geometry.heads, geometry.dim, geometry.rows);
    let mut state = inputs.state.to_vec();
    // W_t per value row.
    let mut accumulated = vec![0.0; h * d];
    let state_scale = |accumulated: &[f64], steps: usize| -> Vec<f64> {
        accumulated
            .iter()
            .flat_map(|&w| std::iter::repeat_n(w / steps as f64, d))
            .collect()
    };
    let mut values = vec![0.0; rows * h * d];
    let mut bounds = vec![0.0; rows * h * d];
    let mut lengths = vec![0; rows * h * d];
    let mut row_states = Vec::new();
    for row in 0..rows {
        let steps = (row + 1) as f64;
        for head in 0..h {
            let Step { q, k, v, beta, g } = step(&geometry, &inputs, row, head);
            let decay = g.iter().map(|x| x.exp()).collect::<Vec<_>>();
            let gamma = decay.iter().copied().fold(0.0, f64::max);
            let q_norm = norm2(&q);
            for value in 0..d {
                let s = &mut state[(head * d + value) * d..(head * d + value + 1) * d];
                let previous = norm2(s);
                for (x, factor) in s.iter_mut().zip(&decay) {
                    *x *= factor;
                }
                let delta = (v[value] - dot(s, &k)) * beta;
                for (x, key) in s.iter_mut().zip(&k) {
                    *x += key * delta;
                }
                let r = head * d + value;
                accumulated[r] = gamma * accumulated[r] + 2.0 * previous + v[value].abs();
                let index = (row * h + head) * d + value;
                values[index] = dot(s, &q);
                bounds[index] = q_norm * (accumulated[r] + norm2(s)) / steps;
                lengths[index] = d * (row + 1);
            }
        }
        if matches!(geometry.phase, RecurrentPhase::Verify { .. }) {
            row_states.push(OracleOutput::new(
                state.clone(),
                state_scale(&accumulated, row + 1),
                d * (row + 1),
            )?);
        }
    }
    Ok(DeltaRuleOutput {
        output: OracleOutput {
            values,
            sum_abs_products: bounds,
            reduction_lengths: lengths,
        },
        state: OracleOutput::new(state, state_scale(&accumulated, rows), d * rows)?,
        row_states,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOUND: DecayGate = DecayGate::BoundedSigmoid {
        lower_bound_bits: 0xc0a0_0000, // -5.0
    };

    fn rule(heads: u32, gate: DecayGate, phase: RecurrentPhase) -> PerChannelDecayDeltaRuleQuery {
        PerChannelDecayDeltaRuleQuery::DeltaRule {
            heads,
            head_dim: 64,
            gate,
            phase,
            capacity: 1024,
        }
    }

    /// Deterministic values in [-1, 1).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
        fn vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
            (0..n).map(|_| self.next() * scale).collect()
        }
    }

    struct Operands {
        q: Vec<f64>,
        k: Vec<f64>,
        v: Vec<f64>,
        b: Vec<f64>,
        f: Vec<f64>,
        a_log: Vec<f64>,
        dt_bias: Vec<f64>,
        state: Vec<f64>,
    }

    impl Operands {
        fn random(seed: u64, rows: usize, heads: usize, f_scale: f64) -> Operands {
            let mut r = Lcg(seed);
            let n = rows * heads * 64;
            Operands {
                q: r.vec(n, 2.0),
                k: r.vec(n, 2.0),
                v: r.vec(n, 1.0),
                b: r.vec(rows * heads, 3.0),
                f: r.vec(n, f_scale),
                a_log: r.vec(heads, 1.0),
                dt_bias: r.vec(heads * 64, 1.0),
                state: r.vec(heads * 64 * 64, 0.5),
            }
        }

        /// The operands of `rows` over `state`.
        fn inputs<'a>(
            &'a self,
            rows: std::ops::Range<usize>,
            state: &'a [f64],
        ) -> DeltaRuleInputs<'a> {
            let total = self.b.len() / self.a_log.len();
            let slice = |all: &'a [f64]| {
                let width = all.len() / total;
                &all[rows.start * width..rows.end * width]
            };
            DeltaRuleInputs {
                q: slice(&self.q),
                k: slice(&self.k),
                v: slice(&self.v),
                b: slice(&self.b),
                f: slice(&self.f),
                a_log: &self.a_log,
                dt_bias: &self.dt_bias,
                state,
            }
        }
    }

    #[test]
    fn gate_forms_match_hand_values() {
        // a = exp(0) = 1 and x = f + dt_bias = 0: sigmoid(0) = 1/2.
        assert_eq!(log_decay(BOUND, 0.25, -0.25, 0.0), -2.5);
        // Saturated: the lower bound per step.
        assert_eq!(log_decay(BOUND, 10.0, 0.0, 10f64.ln()), -5.0);
        // Softplus(0) = ln 2, so the decay is exp(-ln 2) = 1/2.
        assert!((log_decay(DecayGate::Softplus, 0.0, 0.0, 0.0).exp() - 0.5).abs() < 1e-15);
        // Above 20 softplus passes x through: -exp(1) * 30.
        assert_eq!(
            log_decay(DecayGate::Softplus, 25.0, 5.0, 1.0),
            -(1f64.exp()) * 30.0
        );
        // Unbounded below.
        assert!(log_decay(DecayGate::Softplus, 1000.0, 0.0, 3.0) < -20_000.0);
        // beta rounds to BF16: sigmoid(1) = 0.731058... becomes 374 / 512.
        assert_eq!(beta(1.0), 374.0 / 512.0);
        assert_eq!(beta(0.0), 0.5);
    }

    #[test]
    fn one_hot_rows_match_hand_computed_recurrence() {
        let d = 64;
        let (mut q, mut k, mut v) = (vec![0.0; 2 * d], vec![0.0; 2 * d], vec![0.0; 2 * d]);
        // Row 0: q = k = 3 e0, v = e5 * 2. Row 1: q = e0, k = e1, v = e5.
        q[0] = 3.0;
        k[0] = 3.0;
        v[5] = 2.0;
        q[d] = 1.0;
        k[d + 1] = 1.0;
        v[d + 5] = 1.0;
        // Channel decay exp(-2.5) on every channel, beta = bf16(sigmoid(1)).
        let f = vec![0.0; 2 * d];
        let operands = DeltaRuleInputs {
            q: &q,
            k: &k,
            v: &v,
            b: &[1.0, 1.0],
            f: &f,
            a_log: &[0.0],
            dt_bias: &vec![0.0; d],
            state: &vec![0.0; d * d],
        };
        let out = delta_rule(
            rule(1, BOUND, RecurrentPhase::Verify { max_rows: 2 }),
            operands,
        )
        .unwrap();
        let b = 374.0 / 512.0;
        let n0 = 3.0 / (9.0 + 1e-6_f64).sqrt();
        let n1 = 1.0 / (1.0 + 1e-6_f64).sqrt();
        let scale = 1.0 / 8.0;
        // Row 0 from zero state: S[5][0] = b * 2 * n0, o[5] = S[5][0] n0 / 8.
        let s50 = b * 2.0 * n0;
        assert!((out.row_states[0].values[5 * d] - s50).abs() < 1e-15);
        assert!((out.output.values[5] - s50 * n0 * scale).abs() < 1e-15);
        assert_eq!(out.output.values[4], 0.0);
        // Row 1: S decays by exp(-2.5); k = e1 reads nothing, so
        // S[5][1] = b * n1 and o[5] = exp(-2.5) S[5][0] n1 / 8.
        let decayed = s50 * (-2.5_f64).exp();
        assert!((out.state.values[5 * d] - decayed).abs() < 1e-15);
        assert!((out.state.values[5 * d + 1] - b * n1).abs() < 1e-15);
        assert!((out.output.values[d + 5] - decayed * n1 * scale).abs() < 1e-15);
        // The verify state after the last row is the final state.
        assert_eq!(out.row_states[1].values, out.state.values);
    }

    #[test]
    fn short_conv_matches_hand_values_and_separate_equals_fused() {
        let (heads, d, kernel) = (1, 64, 4);
        let channels = 3 * heads * d;
        let mut r = Lcg(7);
        let input = r.vec(5 * channels, 2.0);
        let state = r.vec(3 * channels, 2.0);
        let fused = r.vec(channels * kernel, 1.0);
        let query = |convs| PerChannelDecayDeltaRuleQuery::ShortConv {
            heads: heads as u32,
            head_dim: d as u32,
            kernel: kernel as u32,
            convs,
            phase: RecurrentPhase::Prefill,
            capacity: 64,
        };
        let one = short_conv(query(QkvConvs::Fused), &input, &[&fused], &state).unwrap();
        let width = d * kernel;
        let parts = [
            &fused[..width],
            &fused[width..2 * width],
            &fused[2 * width..],
        ];
        let three = short_conv(query(QkvConvs::Separate), &input, &parts, &state).unwrap();
        assert_eq!(one, three);
        // Row 0, channel 70 (a k channel): taps oldest first over the three
        // state rows and the input.
        let c = 70;
        let w = &fused[c * kernel..(c + 1) * kernel];
        let x = [
            state[c],
            state[channels + c],
            state[2 * channels + c],
            input[c],
        ];
        let sum = w.iter().zip(x).map(|(a, b)| a * b).sum::<f64>();
        let silu = sum / (1.0 + (-sum).exp());
        assert!((one.output.values[c] - silu).abs() < 1e-14);
        // The final state is the last three inputs.
        assert_eq!(one.state, input[2 * channels..].to_vec());
        assert!(short_conv(query(QkvConvs::Separate), &input, &[&fused], &state).is_err());
    }

    #[test]
    fn verify_checkpoints_equal_decoding_each_row() {
        let (rows, heads) = (4, 2);
        for gate in [BOUND, DecayGate::Softplus] {
            let operands = Operands::random(31, rows, heads, 2.0);
            let verify = delta_rule(
                rule(2, gate, RecurrentPhase::Verify { max_rows: 4 }),
                operands.inputs(0..rows, &operands.state),
            )
            .unwrap();
            assert_eq!(verify.row_states.len(), rows);
            let mut state = operands.state.clone();
            for kept in 1..=rows {
                let decode = delta_rule(
                    rule(2, gate, RecurrentPhase::Decode),
                    operands.inputs(kept - 1..kept, &state),
                )
                .unwrap();
                let width = heads * 64;
                assert_eq!(
                    decode.output.values,
                    verify.output.values[(kept - 1) * width..kept * width]
                );
                // Commit with `kept` accepted rows keeps checkpoint kept - 1.
                assert_eq!(decode.state.values, verify.row_states[kept - 1].values);
                state = decode.state.values;
            }
        }
        assert!(
            delta_rule(
                rule(2, BOUND, RecurrentPhase::Decode),
                Operands::random(1, 2, 2, 1.0).inputs(0..2, &[0.0; 2 * 64 * 64]),
            )
            .is_err()
        );
    }
}
