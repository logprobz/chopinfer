use kernel_core::contract::{
    ConvActivation, DType, GateActivation, NumericalContract, RecurrentPhase, RecurrentQuery,
    Validate,
};
use kernel_core::oracle::check_numeric;
use kernel_core::recurrent_oracle::{
    GatedDeltaInputs, causal_conv1d, gated_delta_rule, gated_rmsnorm, sigmoid,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
    }

    fn vec(&mut self, len: usize, scale: f64) -> Vec<f64> {
        (0..len).map(|_| self.next() * scale).collect()
    }
}

fn conv(channels: u32, dilation: u32, phase: RecurrentPhase) -> RecurrentQuery {
    RecurrentQuery::CausalConv1d {
        channels,
        kernel: 4,
        dilation,
        activation: ConvActivation::Silu,
        phase,
        capacity: 4096,
    }
}

fn delta(key_heads: u32, value_heads: u32, dim: u32, phase: RecurrentPhase) -> RecurrentQuery {
    RecurrentQuery::GatedDeltaRule {
        key_heads,
        value_heads,
        key_dim: dim,
        value_dim: dim,
        phase,
        capacity: 4096,
    }
}

struct DeltaCase {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    a_log: Vec<f64>,
    dt_bias: Vec<f64>,
    state: Vec<f64>,
}

impl DeltaCase {
    fn random(rng: &mut Rng, rows: usize, kh: usize, vh: usize, d: usize) -> Self {
        Self {
            q: rng.vec(rows * kh * d, 1.0),
            k: rng.vec(rows * kh * d, 1.0),
            v: rng.vec(rows * vh * d, 1.0),
            a: rng.vec(rows * vh, 2.0),
            b: rng.vec(rows * vh, 2.0),
            a_log: rng.vec(vh, 1.0),
            dt_bias: rng.vec(vh, 1.0),
            state: rng.vec(vh * d * d, 0.5),
        }
    }

    fn inputs(&self) -> GatedDeltaInputs<'_> {
        GatedDeltaInputs {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            a: &self.a,
            b: &self.b,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            state: &self.state,
        }
    }

    /// Rows [start, end) continuing from `state`.
    fn rows(
        &self,
        start: usize,
        end: usize,
        state: &[f64],
        kh: usize,
        vh: usize,
        d: usize,
    ) -> Self {
        Self {
            q: self.q[start * kh * d..end * kh * d].to_vec(),
            k: self.k[start * kh * d..end * kh * d].to_vec(),
            v: self.v[start * vh * d..end * vh * d].to_vec(),
            a: self.a[start * vh..end * vh].to_vec(),
            b: self.b[start * vh..end * vh].to_vec(),
            a_log: self.a_log.clone(),
            dt_bias: self.dt_bias.clone(),
            state: state.to_vec(),
        }
    }
}

fn max_difference(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn qwen38_recurrent_contracts_validate_with_stable_ids() {
    let linear = conv(10240, 1, RecurrentPhase::Prefill);
    let ple = conv(10240, 3, RecurrentPhase::Decode);
    assert_eq!(linear.state_elements(), 3 * 10240);
    assert_eq!(ple.state_elements(), 9 * 10240);
    assert_eq!(
        linear.numerical_contract_id().unwrap().as_str(),
        "recurrent.causal-conv1d-depthwise-silu.prefill.v1"
    );
    assert_eq!(
        ple.numerical_contract_id().unwrap().as_str(),
        "recurrent.causal-conv1d-depthwise-silu.decode.v1"
    );
    // TP4 holds 4 key heads and 12 value heads per rank.
    let rank = delta(4, 12, 128, RecurrentPhase::Verify { max_rows: 4 });
    assert_eq!(rank.state_elements() * 4, 786_432);
    assert_eq!(
        rank.numerical_contract_id().unwrap().as_str(),
        "recurrent.gated-delta-rule.verify.v1"
    );
    assert_eq!(
        delta(16, 48, 128, RecurrentPhase::Decode).state_elements(),
        48 * 128 * 128
    );
    let norm = RecurrentQuery::GatedRmsNorm {
        width: 1536,
        head_dim: 128,
        eps_bits: 1e-6_f32.to_bits(),
        gate: GateActivation::Sigmoid,
        capacity: 4096,
    };
    assert_eq!(
        norm.numerical_contract_id().unwrap().as_str(),
        "row.rmsnorm-gated-sigmoid.v1"
    );
    assert!(
        delta(16, 40, 128, RecurrentPhase::Decode)
            .validate()
            .is_err()
    );
    // Head dims are math, any nonzero width; a backend refuses the ones it
    // cannot run.
    delta(16, 48, 96, RecurrentPhase::Decode)
        .validate()
        .unwrap();
    assert!(delta(16, 48, 0, RecurrentPhase::Decode).validate().is_err());
    assert!(
        delta(16, 48, 128, RecurrentPhase::Verify { max_rows: 0 })
            .validate()
            .is_err()
    );
    // Verify rows are math from one up; a backend refuses past its limit.
    delta(16, 48, 128, RecurrentPhase::Verify { max_rows: 64 })
        .validate()
        .unwrap();
    assert!(conv(10240, 0, RecurrentPhase::Decode).validate().is_err());
    // A state whose elements overflow a u64 is refused, not wrapped.
    assert!(
        delta(1, 1 << 31, 1 << 20, RecurrentPhase::Decode)
            .validate()
            .is_err()
    );
    assert!(
        conv(u32::MAX, u32::MAX, RecurrentPhase::Decode)
            .validate()
            .is_err()
    );
}

#[test]
fn causal_conv_matches_hand_computed_taps() {
    let query = |dilation, activation| RecurrentQuery::CausalConv1d {
        channels: 1,
        kernel: 4,
        dilation,
        activation,
        phase: RecurrentPhase::Prefill,
        capacity: 8,
    };
    let identity = ConvActivation::Identity;
    // Oldest tap first: 1*1 + 2*2 + 3*3 + 4*4, then 2*1 + 3*2 + 4*3 + 5*4.
    let out = causal_conv1d(
        query(1, identity),
        &[4.0, 5.0],
        &[1.0, 2.0, 3.0, 4.0],
        &[1.0, 2.0, 3.0],
    )
    .unwrap();
    assert_eq!(out.output.values, [30.0, 40.0]);
    assert_eq!(out.state, [3.0, 4.0, 5.0]);
    // Dilation 3 reads 9, 6 and 3 rows back; the state keeps 9 rows.
    let state = (1..=9).map(f64::from).collect::<Vec<_>>();
    let out = causal_conv1d(
        query(3, identity),
        &[10.0],
        &[1.0, 10.0, 100.0, 1000.0],
        &state,
    )
    .unwrap();
    assert_eq!(out.output.values, [1.0 + 40.0 + 700.0 + 10000.0]);
    assert_eq!(out.state, (2..=10).map(f64::from).collect::<Vec<_>>());
    let silu = query(1, ConvActivation::Silu);
    let out = causal_conv1d(silu, &[4.0], &[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(out.output.values, [30.0 * sigmoid(30.0)]);
}

#[test]
fn causal_conv_prefill_decode_and_verify_states_agree() {
    let mut rng = Rng(7);
    let channels = 6;
    for dilation in [1, 3] {
        let history = 3 * dilation as usize;
        let weight = rng.vec(channels * 4, 1.0);
        let state = rng.vec(history * channels, 1.0);
        let input = rng.vec(5 * channels, 1.0);
        let prefill = causal_conv1d(
            conv(channels as u32, dilation, RecurrentPhase::Prefill),
            &input,
            &weight,
            &state,
        )
        .unwrap();
        let verify = causal_conv1d(
            conv(
                channels as u32,
                dilation,
                RecurrentPhase::Verify { max_rows: 5 },
            ),
            &input,
            &weight,
            &state,
        )
        .unwrap();
        assert_eq!(verify.output, prefill.output);
        let mut rolling = state.clone();
        for (row, x) in input.chunks_exact(channels).enumerate() {
            let step = causal_conv1d(
                conv(channels as u32, dilation, RecurrentPhase::Decode),
                x,
                &weight,
                &rolling,
            )
            .unwrap();
            assert_eq!(
                step.output.values,
                prefill.output.values[row * channels..(row + 1) * channels]
            );
            rolling = step.state;
            // Rolling back to `kept` rows restores this checkpoint.
            assert_eq!(verify.row_states[row], rolling);
        }
        assert_eq!(rolling, prefill.state);
        assert!(
            causal_conv1d(
                conv(channels as u32, dilation, RecurrentPhase::Decode),
                &input,
                &weight,
                &state
            )
            .is_err()
        );
    }
}

#[test]
fn delta_rule_first_steps_match_hand_computation() {
    let d = 64;
    let query = delta(1, 1, d as u32, RecurrentPhase::Prefill);
    let mut q = vec![0.0; 2 * d];
    q[0] = 1.0;
    q[d] = 1.0;
    let mut k = vec![0.0; 2 * d];
    k[0] = 1.0;
    k[d + 1] = 1.0;
    let mut v = vec![0.0; 2 * d];
    v[0] = 2.0;
    v[d] = 5.0;
    let state = vec![0.0; d * d];
    // a + dt_bias = 0 and A_log = 0 give g = -ln 2; b = 0 gives beta = 1/2.
    let out = gated_delta_rule(
        query,
        GatedDeltaInputs {
            q: &q,
            k: &k,
            v: &v,
            a: &[0.0, 0.0],
            b: &[0.0, 0.0],
            a_log: &[0.0],
            dt_bias: &[0.0],
            state: &state,
        },
    )
    .unwrap();
    let unit = 1.0 / (1.0 + 1e-6_f64).sqrt();
    let scale = 1.0 / (d as f64).sqrt();
    let first = 0.5 * 2.0 * unit * unit * scale;
    assert!((out.output.values[0] - first).abs() < 1e-15);
    assert!(out.output.values[1..d].iter().all(|&x| x == 0.0));
    // Step 2 halves the state; its key is orthogonal, so q still reads only
    // the first column, and the new value lands in a column q does not read.
    assert!((out.output.values[d] - 0.5 * first).abs() < 1e-15);
    let s = &out.state.values;
    assert!((s[0] - 0.5 * 0.5 * 2.0 * unit).abs() < 1e-15);
    assert!((s[1] - 0.5 * 5.0 * unit).abs() < 1e-15);
}

#[test]
fn split_prefill_decode_and_verify_checkpoints_continue_one_recurrence() {
    let mut rng = Rng(13);
    let (kh, vh, d) = (1, 3, 64);
    let rows = 9;
    let case = DeltaCase::random(&mut rng, rows, kh, vh, d);
    let whole = gated_delta_rule(delta(1, 3, 64, RecurrentPhase::Prefill), case.inputs()).unwrap();
    let head = case.rows(0, 5, &case.state, kh, vh, d);
    let first = gated_delta_rule(delta(1, 3, 64, RecurrentPhase::Prefill), head.inputs()).unwrap();
    let tail = case.rows(5, rows, &first.state.values, kh, vh, d);
    let verify = gated_delta_rule(
        delta(1, 3, 64, RecurrentPhase::Verify { max_rows: 4 }),
        tail.inputs(),
    )
    .unwrap();
    let width = vh * d;
    assert!(max_difference(&verify.output.values, &whole.output.values[5 * width..]) < 1e-12);
    assert_eq!(verify.row_states.len(), 4);
    let mut state = first.state.values.clone();
    for row in 5..rows {
        let step = case.rows(row, row + 1, &state, kh, vh, d);
        let out = gated_delta_rule(delta(1, 3, 64, RecurrentPhase::Decode), step.inputs()).unwrap();
        assert!(
            max_difference(
                &out.output.values,
                &whole.output.values[row * width..(row + 1) * width]
            ) < 1e-12
        );
        state = out.state.values;
        // Committing `kept` verify rows keeps checkpoint `kept - 1`.
        assert_eq!(verify.row_states[row - 5].values, state);
    }
    assert!(max_difference(&state, &whole.state.values) < 1e-12);
    let too_many = case.rows(0, 5, &case.state, kh, vh, d);
    assert!(
        gated_delta_rule(
            delta(1, 3, 64, RecurrentPhase::Verify { max_rows: 4 }),
            too_many.inputs()
        )
        .is_err()
    );
}

#[test]
fn value_heads_read_key_heads_by_integer_division() {
    let mut rng = Rng(17);
    let (kh, vh, d) = (2, 6, 64);
    let case = DeltaCase::random(&mut rng, 3, kh, vh, d);
    let base = gated_delta_rule(delta(2, 6, 64, RecurrentPhase::Prefill), case.inputs()).unwrap();
    let mut changed = DeltaCase::random(&mut Rng(17), 3, kh, vh, d);
    for row in 0..3 {
        for x in 0..d {
            changed.k[(row * kh) * d + x] += 0.25;
        }
    }
    let other =
        gated_delta_rule(delta(2, 6, 64, RecurrentPhase::Prefill), changed.inputs()).unwrap();
    for row in 0..3 {
        for head in 0..vh {
            let range = (row * vh + head) * d..(row * vh + head + 1) * d;
            let same = base.output.values[range.clone()] == other.output.values[range];
            assert_eq!(same, head >= 3, "row {row} head {head}");
        }
    }
}

#[test]
fn f32_recurrence_passes_the_propagated_bound() {
    let mut rng = Rng(19);
    let (kh, vh, d, rows) = (1, 2, 64, 48);
    let case = DeltaCase::random(&mut rng, rows, kh, vh, d);
    let reference =
        gated_delta_rule(delta(1, 2, 64, RecurrentPhase::Prefill), case.inputs()).unwrap();
    let mut state = case.state.iter().map(|&x| x as f32).collect::<Vec<_>>();
    let mut output = vec![0.0_f64; rows * vh * d];
    for row in 0..rows {
        for head in 0..vh {
            let base = row * kh * d;
            let norm = |x: &[f64]| {
                let x = x.iter().map(|&v| v as f32).collect::<Vec<_>>();
                let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
                x.iter().map(|v| v * inv).collect::<Vec<_>>()
            };
            let q = norm(&case.q[base..base + d])
                .iter()
                .map(|v| v / (d as f32).sqrt())
                .collect::<Vec<_>>();
            let k = norm(&case.k[base..base + d]);
            let index = row * vh + head;
            let beta = 1.0 / (1.0 + (-(case.b[index] as f32)).exp());
            let x = case.a[index] as f32 + case.dt_bias[head] as f32;
            let g = -(case.a_log[head] as f32).exp() * (1.0 + x.exp()).ln();
            for v in 0..d {
                let s = &mut state[(head * d + v) * d..(head * d + v + 1) * d];
                let mut memory = 0.0_f32;
                for i in 0..d {
                    s[i] *= g.exp();
                    memory += s[i] * k[i];
                }
                let delta = beta * (case.v[index * d + v] as f32 - memory);
                let mut out = 0.0_f32;
                for i in 0..d {
                    s[i] += k[i] * delta;
                    out += s[i] * q[i];
                }
                output[index * d + v] = f64::from(out);
            }
        }
    }
    check_numeric(&output, &reference.output, DType::F32).unwrap();
    let state = state.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    check_numeric(&state, &reference.state, DType::F32).unwrap();
}

#[test]
fn gated_rmsnorm_matches_hand_computation() {
    let norm = |gate| RecurrentQuery::GatedRmsNorm {
        width: 4,
        head_dim: 2,
        eps_bits: 1e-6_f32.to_bits(),
        gate,
        capacity: 1,
    };
    let query = norm(GateActivation::Sigmoid);
    let eps = f64::from(1e-6_f32);
    let out = gated_rmsnorm(
        query,
        &[3.0, 4.0, 1.0, 1.0],
        &[0.0, 0.0, 2.0, -2.0],
        &[1.0, 2.0],
    )
    .unwrap();
    let first = (12.5 + eps).sqrt().recip();
    let second = (1.0 + eps).sqrt().recip();
    let expected = [
        3.0 * first * 0.5,
        4.0 * first * 2.0 * 0.5,
        second * sigmoid(2.0),
        second * 2.0 * sigmoid(-2.0),
    ];
    assert!(max_difference(&out.values, &expected) < 1e-15);
    let out = gated_rmsnorm(
        norm(GateActivation::Silu),
        &[3.0, 4.0, 1.0, 1.0],
        &[0.0, 1.0, 0.0, 0.0],
        &[1.0, 1.0],
    )
    .unwrap();
    assert!((out.values[1] - 4.0 * first * sigmoid(1.0)).abs() < 1e-15);
    assert!(gated_rmsnorm(query, &[1.0; 8], &[0.0; 8], &[1.0, 1.0]).is_err());
}

/// The norm's error scale follows the output, not the input: an F32
/// evaluation of tiny inputs rounded to BF16 passes, and two BF16 ulps on one
/// output fail.
#[test]
fn gated_rmsnorm_scale_is_in_output_units() {
    use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
    let (width, d) = (256, 128);
    let query = RecurrentQuery::GatedRmsNorm {
        width: width as u32,
        head_dim: d as u32,
        eps_bits: 1e-6_f32.to_bits(),
        gate: GateActivation::Sigmoid,
        capacity: 4,
    };
    let mut rng = Rng(29);
    let round = |x: f32| f64::from(bf16_to_f32(f32_to_bf16(x)));
    for scale in [1e-3, 0.02, 30.0] {
        let x = rng.vec(4 * width, scale);
        let z = rng.vec(4 * width, 3.0);
        let w = rng.vec(d, 0.1).iter().map(|v| 0.95 + v).collect::<Vec<_>>();
        let out = gated_rmsnorm(query, &x, &z, &w).unwrap();
        let mut f32_rounded = Vec::with_capacity(x.len());
        for (head, gates) in x.chunks_exact(d).zip(z.chunks_exact(d)) {
            let squares = head.iter().map(|&v| (v as f32) * (v as f32)).sum::<f32>();
            let inverse = 1.0 / (squares / d as f32 + 1e-6).sqrt();
            for j in 0..d {
                let act = 1.0 / (1.0 + (-(gates[j] as f32)).exp());
                f32_rounded.push(round(head[j] as f32 * inverse * w[j] as f32 * act));
            }
        }
        assert!(
            check_numeric(&f32_rounded, &out, DType::Bf16).is_ok(),
            "scale {scale}"
        );
        let index = 3 * width + 77;
        let ulp = 2_f64.powi(f32_rounded[index].abs().log2().floor() as i32 - 7);
        let mut perturbed = f32_rounded.clone();
        perturbed[index] += 2.0 * ulp;
        let failure = check_numeric(&perturbed, &out, DType::Bf16).unwrap_err();
        assert_eq!(failure.index, index, "scale {scale}");
    }
}

/// 8192 rows of one slowly decaying head (exp(A_log) 0.03, dt_bias -8, so
/// the decay stays within about 1e-5 of 1) and one head that roughly halves
/// its state per row.
fn long_prefill() -> (DeltaCase, kernel_core::recurrent_oracle::GatedDeltaOutput) {
    let rows = 8192;
    let mut case = DeltaCase::random(&mut Rng(23), rows, 1, 2, 64);
    case.a_log = vec![0.03_f64.ln(), 0.0];
    case.dt_bias = vec![-8.0, 0.0];
    let query = RecurrentQuery::GatedDeltaRule {
        key_heads: 1,
        value_heads: 2,
        key_dim: 64,
        value_dim: 64,
        phase: RecurrentPhase::Prefill,
        capacity: rows as u32,
    };
    let out = gated_delta_rule(query, case.inputs()).unwrap();
    (case, out)
}

#[test]
fn delta_error_scale_stays_finite_and_bounded_over_8192_rows() {
    let (_, out) = long_prefill();
    let width = 2 * 64;
    for bounds in [&out.output.sum_abs_products, &out.state.sum_abs_products] {
        assert!(bounds.iter().all(|s| s.is_finite() && *s > 0.0));
    }
    // S is a per-row error scale divided by the rows so far; it must not
    // grow with the row count, even for the head that does not decay.
    let s = &out.output.sum_abs_products;
    let max_over = |rows: std::ops::Range<usize>| {
        rows.flat_map(|r| s[r * width..(r + 1) * width].iter().copied())
            .fold(0.0, f64::max)
    };
    let (early, late) = (max_over(64..128), max_over(8128..8192));
    assert!(late <= 4.0 * early, "late S {late} early S {early}");
    let state_max = out
        .state
        .sum_abs_products
        .iter()
        .copied()
        .fold(0.0, f64::max);
    assert!(state_max < 100.0, "state S {state_max}");
}

#[test]
fn late_bf16_perturbations_fail_the_delta_gate() {
    let (_, out) = long_prefill();
    let round = |x: f64| {
        f64::from(kernel_core::codec::bf16_to_f32(
            kernel_core::codec::f32_to_bf16(x as f32),
        ))
    };
    let ulp = |x: f64| 2_f64.powi(x.abs().log2().floor() as i32 - 7);
    // Exact values rounded to BF16 pass; F32 state copies pass.
    let rounded = out
        .output
        .values
        .iter()
        .map(|&x| round(x))
        .collect::<Vec<_>>();
    assert!(check_numeric(&rounded, &out.output, DType::Bf16).is_ok());
    let state = out
        .state
        .values
        .iter()
        .map(|&x| f64::from(x as f32))
        .collect::<Vec<_>>();
    assert!(check_numeric(&state, &out.state, DType::F32).is_ok());
    // Four BF16 ulps on one late output or state element of the decaying head
    // fail, so the gate stays meaningful 8000 rows in.
    let output_index = (8000 * 2 + 1) * 64 + 5;
    let mut perturbed = rounded.clone();
    perturbed[output_index] += 4.0 * ulp(rounded[output_index]);
    let failure = check_numeric(&perturbed, &out.output, DType::Bf16).unwrap_err();
    assert_eq!(failure.index, output_index);
    let state_index = 64 * 64 + 17 * 64 + 9;
    let mut perturbed = state.clone();
    perturbed[state_index] += 4.0 * ulp(state[state_index]);
    let failure = check_numeric(&perturbed, &out.state, DType::F32).unwrap_err();
    assert_eq!(failure.index, state_index);
}
