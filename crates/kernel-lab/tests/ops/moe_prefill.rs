//! MiMo grouped MoE prefill with skewed routing and distinct weights for all
//! 256 experts. The full row-engine path routes the tokens; sampled routes
//! are checked against the f64 oracle, and the layer output by cosine.

use super::*;
use kernel_core::moe_oracle::{
    ExpertMlp, ExpertMlpRounding, ExpertWeights, LocalRoutes, RouteWeightAt, expert_mlp,
};
use kernel_core::oracle::SiluGlu;
use kernel_core::quantize::Quantizer;

const EXPERTS: usize = 256;
const GATE_SEED: u64 = 0x61a7_e000;
const DOWN_SEED: u64 = 0xd0e1_f000;

fn stream(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut bytes = Vec::with_capacity(len + 8);
    while bytes.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        bytes.extend_from_slice(&state.to_le_bytes());
    }
    bytes.truncate(len);
    bytes
}

/// Row-major E2M1 codes and UE8M0 scales of one expert, regenerated on demand.
fn expert_source(seed: u64, expert: usize, n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let codes = stream(seed ^ (expert as u64) << 20, n * k / 2);
    let scales = stream(seed ^ 0x5ca1e ^ (expert as u64) << 24, n * k / 32)
        .into_iter()
        .map(|byte| 117 + byte % 7)
        .collect();
    (codes, scales)
}

fn decoded_expert(seed: u64, expert: usize, n: usize, k: usize) -> LabResult<Vec<f64>> {
    let (codes, scales) = expert_source(seed, expert, n, k);
    mxfp4_decoded(n, k, &codes, &scales)
}

fn upload_bank(session: &Session, v: &Variant, seed: u64) -> LabResult<GpuWeights> {
    let n = v.query.n as usize;
    let k = v.query.k as usize;
    let one = Mxfp4MoeFragmentLayout {
        experts: 1,
        rows_per_expert: n,
        k,
    };
    let codes = session.allocate(EXPERTS * n * k / 2)?;
    let scales = session.allocate(EXPERTS * n * k / 32)?;
    for expert in 0..EXPERTS {
        let (code_source, scale_source) = expert_source(seed, expert, n, k);
        let packed_codes = one.pack_codes(&code_source).map_err(err)?.bytes;
        let packed_scales = one.pack_scales(&scale_source).map_err(err)?.bytes;
        codes
            .copy_from_bytes_at(expert * n * k / 2, &packed_codes, &session.stream)
            .map_err(err)?;
        scales
            .copy_from_bytes_at(expert * n * k / 32, &packed_scales, &session.stream)
            .map_err(err)?;
        session.stream.synchronize().map_err(err)?;
    }
    Ok(GpuWeights {
        codes,
        scales,
        f32_scales: session.upload_f32(&[1.0])?,
    })
}

/// `CHOPINFER_MOE_PREFILL_ROUTING=balanced` gives every expert exactly
/// tokens * 8 / 256 rows; `hot` sends every token to experts 0-3 and spreads
/// its other four routes, so four experts hold every token (several 32-row
/// tiles of the verify stream entries, and 128-row tiles of prefill); the
/// default is the skewed distribution below.
fn balanced_routing() -> bool {
    std::env::var("CHOPINFER_MOE_PREFILL_ROUTING").as_deref() == Ok("balanced")
}

fn hot_routing() -> bool {
    std::env::var("CHOPINFER_MOE_PREFILL_ROUTING").as_deref() == Ok("hot")
}

fn routing_label() -> &'static str {
    if std::env::var_os("CHOPINFER_MOE_PREFILL_LOGITS").is_some() {
        "replay"
    } else if balanced_routing() {
        "balanced"
    } else if hot_routing() {
        "hot"
    } else {
        "skewed"
    }
}

/// Expert popularity follows a signed power law; per-token noise spreads the
/// remaining routes. The resulting counts are recorded in each receipt.
fn skewed_logits(tokens: usize) -> Vec<f32> {
    if balanced_routing() {
        return (0..tokens * EXPERTS)
            .map(|index| {
                let (token, expert) = (index / EXPERTS, index % EXPERTS);
                if (expert + EXPERTS - token * TOP_K % EXPERTS) % EXPERTS < TOP_K {
                    4.0 + sample(0x8181 ^ index as u64) * 0.1
                } else {
                    -4.0
                }
            })
            .collect();
    }
    if hot_routing() {
        return (0..tokens * EXPERTS)
            .map(|index| {
                let expert = index % EXPERTS;
                let noise = sample(0x8181 ^ index as u64);
                if expert < 4 { 6.0 + noise * 0.1 } else { noise }
            })
            .collect();
    }
    let popularity = (0..EXPERTS)
        .map(|expert| {
            let u = sample(0x7070 ^ expert as u64);
            0.2 * u.signum() * u.abs().powf(1.5)
        })
        .collect::<Vec<_>>();
    (0..tokens * EXPERTS)
        .map(|index| popularity[index % EXPERTS] + sample(0x8181 ^ index as u64))
        .collect()
}

struct Routing {
    counts: Vec<usize>,
    offsets: Vec<i32>,
    sorted: Vec<i32>,
    ids: Vec<i32>,
    weights: Vec<f32>,
}

impl Routing {
    fn stats(&self) -> Value {
        let active = self.counts.iter().filter(|&&c| c > 0).count();
        let mean = self.counts.iter().sum::<usize>() as f64 / EXPERTS as f64;
        let max = *self.counts.iter().max().unwrap_or(&0);
        let min = *self.counts.iter().min().unwrap_or(&0);
        let tiles = self.counts.iter().map(|c| c.div_ceil(64)).sum::<usize>();
        json!({"active_experts": active, "mean_rows": mean, "max_rows": max,
            "min_rows": min, "max_over_mean": max as f64 / mean,
            "padded_rows": self.sorted.len(), "row_blocks_64": tiles})
    }
}

fn read_routing(session: &Session, case: &FullCase) -> LabResult<Routing> {
    let routes = case.tokens * TOP_K;
    let ids = download_i32(session, &case.ids, routes)?;
    let weights = download_f32(session, &case.topk, routes)?;
    let offsets = download_i32(session, &case.offsets, EXPERTS + 1)?;
    let padded = download_i32(session, &case.padded, 1)?[0] as usize;
    let sorted = download_i32(session, &case.sorted, padded)?;
    let permutation = oracle::expert_permute(
        &ids.iter().map(|&id| id as usize).collect::<Vec<_>>(),
        EXPERTS,
        case.gate.knobs.tile_m as usize,
    )
    .map_err(err)?;
    let expected = permutation
        .forward
        .iter()
        .map(|x| x.map_or(-1, |r| r as i32))
        .collect::<Vec<_>>();
    if sorted != expected
        || offsets
            != permutation
                .expert_offsets
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>()
    {
        return Err("prefill expert permutation differs from the oracle".into());
    }
    let mut counts = vec![0; EXPERTS];
    for &id in &ids {
        counts[id as usize] += 1;
    }
    Ok(Routing {
        counts,
        offsets,
        sorted,
        ids,
        weights,
    })
}

/// Routes of three tokens plus the tile-edge routes of the most and least
/// loaded experts, which exercise partial 128-row tiles and their row groups.
fn sampled_routes(case: &FullCase, routing: &Routing) -> (Vec<usize>, Vec<usize>) {
    let tokens = vec![0, case.tokens / 2, case.tokens - 1];
    let mut routes = tokens
        .iter()
        .flat_map(|&token| (0..TOP_K).map(move |slot| token * TOP_K + slot))
        .collect::<Vec<_>>();
    let busiest = (0..EXPERTS).max_by_key(|&e| routing.counts[e]).unwrap_or(0);
    let quietest = (0..EXPERTS)
        .filter(|&e| routing.counts[e] > 0)
        .min_by_key(|&e| routing.counts[e])
        .unwrap_or(0);
    for expert in [busiest, quietest] {
        let first = routing.offsets[expert] as usize;
        let count = routing.counts[expert];
        // Row-group, 32-row verify tile and 128-row prefill tile edges.
        for local in [
            0,
            15,
            16,
            31,
            32,
            33,
            63,
            64,
            95,
            96,
            127,
            128,
            129,
            255,
            count / 2,
            count - 1,
        ] {
            if local < count {
                routes.push(routing.sorted[first + local] as usize);
            }
        }
    }
    routes.sort_unstable();
    routes.dedup();
    (tokens, routes)
}

fn check_bf16_rows(
    session: &Session,
    all: &[u16],
    width: usize,
    rows: &[usize],
    reference: &OracleOutput,
    case: &str,
    stage: &str,
) -> LabResult<()> {
    let actual = rows
        .iter()
        .flat_map(|&row| bf16_f32(&all[row * width..(row + 1) * width]))
        .collect::<Vec<_>>();
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!(case));
    receipt.insert("stage".into(), json!(stage));
    receipt.insert("checked_rows".into(), json!(rows));
    let gate = check_output(&actual, reference, DType::Bf16, &mut receipt);
    let worst = receipt.get("worst_element").cloned();
    session.receipt("moe", receipt)?;
    gate.map_err(|error| format!("{case} {stage}: {error}; worst element {worst:?}"))
}

fn concat(outputs: Vec<OracleOutput>) -> LabResult<OracleOutput> {
    let mut values = Vec::new();
    let mut sums = Vec::new();
    let mut lengths = Vec::new();
    for output in outputs {
        values.extend(output.values);
        sums.extend(output.sum_abs_products);
        lengths.extend(output.reduction_lengths);
    }
    Ok(OracleOutput {
        values,
        sum_abs_products: sums,
        reduction_lengths: lengths,
    })
}

/// Cosine and relative L2 error of each width-sized row against a reference.
fn row_errors(actual: &[f64], reference: &[f64], width: usize) -> Vec<(f64, f64)> {
    actual
        .chunks(width)
        .zip(reference.chunks(width))
        .map(|(a, r)| {
            let dot = a.iter().zip(r).map(|(x, y)| x * y).sum::<f64>();
            let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_r = r.iter().map(|y| y * y).sum::<f64>().sqrt();
            let diff = a
                .iter()
                .zip(r)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt();
            (dot / (norm_a * norm_r), diff / norm_r)
        })
        .collect()
}

fn check_case(session: &Session, case: &FullCase, routing: &Routing, name: &str) -> LabResult<()> {
    let routes = case.tokens * TOP_K;
    let (tokens, sampled) = sampled_routes(case, routing);
    let gate_all = session.download_bf16(&case.gate_output, routes * 512)?;
    let down_all = session.download_bf16(&case.down_output, routes * HIDDEN)?;
    let final_all = session.download_bf16(&case.final_output, case.tokens * HIDDEN)?;
    let mut experts = sampled
        .iter()
        .map(|&r| routing.ids[r] as usize)
        .collect::<Vec<_>>();
    experts.sort_unstable();
    experts.dedup();
    // Oracle outputs are grouped by expert; `order` lists their routes.
    let mut order = Vec::new();
    let mut gate_refs = Vec::new();
    let mut down_refs = Vec::new();
    let mut full_down = BTreeMap::new();
    // The same chain with MXFP8 activations into both GEMMs, as in the A8
    // path with its fused MXFP8 gate/up output.
    let mut a8_down = BTreeMap::new();
    for &expert in &experts {
        let mine = sampled
            .iter()
            .copied()
            .filter(|&r| routing.ids[r] as usize == expert)
            .collect::<Vec<_>>();
        let m = mine.len();
        let ranges = [ExpertRowRange {
            expert: 0,
            start: 0,
            end: m,
        }];
        let gate_input = mine
            .iter()
            .flat_map(|&r| {
                case.input_host[(r / TOP_K) * HIDDEN..(r / TOP_K + 1) * HIDDEN]
                    .iter()
                    .map(|&v| f64::from(bf16_to_f32(v)))
            })
            .collect::<Vec<_>>();
        let gate_weight = decoded_expert(GATE_SEED, expert, 1024, HIDDEN)?;
        let down_weight = decoded_expert(DOWN_SEED, expert, HIDDEN, 512)?;
        let routes = LocalRoutes {
            tokens: (0..m).collect(),
            experts: vec![0; m],
            routes: mine.clone(),
        };
        let route_weights = mine
            .iter()
            .map(|&r| f64::from(routing.weights[r]))
            .collect::<Vec<_>>();
        let weights = ExpertWeights {
            gate_up: &gate_weight,
            down: &down_weight,
            experts: 1,
            hidden: HIDDEN,
            intermediate: 512,
        };
        // The served chain: a BF16 gate/up output into down, and each route's
        // weighted down output stored as BF16.
        let chain = |rounding| {
            let mlp = ExpertMlp {
                activation: SiluGlu::UNCLAMPED,
                route_weight_at: RouteWeightAt::AfterDown,
                rounding,
            };
            expert_mlp(&gate_input, m, &routes, &route_weights, weights, mlp).map_err(err)
        };
        let full = chain(ExpertMlpRounding {
            intermediate: Some(DType::Bf16),
            weighted_output: Some(DType::Bf16),
            ..ExpertMlpRounding::default()
        })?;
        let mxfp8 = Some(Quantizer::mx(SmallFloat::E4M3));
        let a8 = chain(ExpertMlpRounding {
            input: mxfp8,
            down_input: mxfp8,
            weighted_output: Some(DType::Bf16),
            ..ExpertMlpRounding::default()
        })?;
        let gate_ref = full.intermediate;
        let down_input = mine
            .iter()
            .flat_map(|&r| {
                gate_all[r * 512..(r + 1) * 512]
                    .iter()
                    .map(|&v| f64::from(bf16_to_f32(v)))
            })
            .collect::<Vec<_>>();
        let shape = GemmShape {
            m,
            n: HIDDEN,
            k: 512,
            experts: 1,
        };
        let plain = GemmEpilogue {
            residual: None,
            swiglu: None,
        };
        let mut down_ref =
            oracle::gemm(&down_input, &down_weight, shape, &ranges, plain).map_err(err)?;
        for (row, &r) in mine.iter().enumerate() {
            let weight = route_weights[row];
            for col in 0..HIDDEN {
                let at = row * HIDDEN + col;
                down_ref.values[at] *= weight;
                down_ref.sum_abs_products[at] *= weight.abs();
            }
            let at = row * HIDDEN..(row + 1) * HIDDEN;
            full_down.insert(r, full.weighted.values[at.clone()].to_vec());
            a8_down.insert(r, a8.weighted.values[at].to_vec());
        }
        order.extend(mine);
        gate_refs.push(gate_ref);
        down_refs.push(down_ref);
    }
    check_bf16_rows(
        session,
        &gate_all,
        512,
        &order,
        &concat(gate_refs)?,
        name,
        "gate_up",
    )?;
    check_bf16_rows(
        session,
        &down_all,
        HIDDEN,
        &order,
        &concat(down_refs)?,
        name,
        "weighted_down",
    )?;
    let token_routes = tokens
        .iter()
        .flat_map(|&t| (0..TOP_K).map(move |slot| t * TOP_K + slot))
        .collect::<Vec<_>>();
    let route_tokens = (0..tokens.len())
        .flat_map(|t| std::iter::repeat_n(t, TOP_K))
        .collect::<Vec<_>>();
    let combine_input = token_routes
        .iter()
        .flat_map(|&r| {
            down_all[r * HIDDEN..(r + 1) * HIDDEN]
                .iter()
                .map(|&v| f64::from(bf16_to_f32(v)))
        })
        .collect::<Vec<_>>();
    let combine_ref = oracle::expert_combine(
        &combine_input,
        &route_tokens,
        &vec![1.0; token_routes.len()],
        tokens.len(),
        HIDDEN,
    )
    .map_err(err)?;
    check_bf16_rows(
        session,
        &final_all,
        HIDDEN,
        &tokens,
        &combine_ref,
        name,
        "preweighted_combine",
    )?;
    let chained = token_routes
        .iter()
        .flat_map(|r| full_down[r].iter().copied())
        .collect::<Vec<_>>();
    let layer_ref = oracle::expert_combine(
        &chained,
        &route_tokens,
        &vec![1.0; token_routes.len()],
        tokens.len(),
        HIDDEN,
    )
    .map_err(err)?;
    let actual = tokens
        .iter()
        .flat_map(|&t| {
            final_all[t * HIDDEN..(t + 1) * HIDDEN]
                .iter()
                .map(|&v| f64::from(bf16_to_f32(v)))
        })
        .collect::<Vec<_>>();
    let a8_layer = oracle::expert_combine(
        &token_routes
            .iter()
            .flat_map(|r| a8_down[r].iter().copied())
            .collect::<Vec<_>>(),
        &route_tokens,
        &vec![1.0; token_routes.len()],
        tokens.len(),
        HIDDEN,
    )
    .map_err(err)?;
    let a8_errors = row_errors(&a8_layer.values, &layer_ref.values, HIDDEN);
    let a16_errors = row_errors(&actual, &layer_ref.values, HIDDEN);
    let mut a8_receipt = BTreeMap::new();
    a8_receipt.insert("case".into(), json!(name));
    a8_receipt.insert("stage".into(), json!("a8_numerics_layer_error"));
    a8_receipt.insert("checked_tokens".into(), json!(tokens));
    a8_receipt.insert(
        "note".into(),
        json!("f64 chain with MXFP8 gate/up and down inputs versus the A16 f64 chain; A16 is the GPU output versus the same reference"),
    );
    a8_receipt.insert(
        "a8_cosine".into(),
        json!(a8_errors.iter().map(|e| e.0).collect::<Vec<_>>()),
    );
    a8_receipt.insert(
        "a8_relative_l2".into(),
        json!(a8_errors.iter().map(|e| e.1).collect::<Vec<_>>()),
    );
    a8_receipt.insert(
        "a16_gpu_cosine".into(),
        json!(a16_errors.iter().map(|e| e.0).collect::<Vec<_>>()),
    );
    a8_receipt.insert(
        "a16_gpu_relative_l2".into(),
        json!(a16_errors.iter().map(|e| e.1).collect::<Vec<_>>()),
    );
    session.receipt("moe", a8_receipt)?;
    let cosine = oracle::check_layer_cosine(&actual, &layer_ref.values, HIDDEN);
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!(name));
    receipt.insert("stage".into(), json!("end_to_end_layer_cosine"));
    receipt.insert("checked_tokens".into(), json!(tokens));
    receipt.insert("checked_routes".into(), json!(order.len()));
    receipt.insert("routing".into(), routing.stats());
    receipt.insert("correct".into(), json!(cosine.is_ok()));
    if let Err(worst) = &cosine {
        receipt.insert(
            "worst_token".into(),
            json!({"index": worst.index, "cosine": worst.actual, "bound": worst.bound}),
        );
    }
    session.receipt("moe", receipt)?;
    cosine.map_err(|worst| {
        format!(
            "{name} token {} cosine {} < {}",
            worst.index, worst.actual, worst.bound
        )
    })
}

fn selected_token_counts() -> Vec<usize> {
    let all = [64, 128, 256, 512, 1024, 2048, 8192];
    match std::env::var("CHOPINFER_MOE_PREFILL_TOKENS") {
        Ok(value) => all
            .into_iter()
            .filter(|&t| value.split(',').any(|v| v == t.to_string()))
            .collect(),
        Err(_) => vec![512, 2048, 8192],
    }
}

fn skewed_case(
    session: &Session,
    tokens: usize,
    gate: &mut GpuWeights,
    down: &mut GpuWeights,
) -> LabResult<FullCase> {
    skewed_phase_case(session, Phase::Prefill, tokens, gate, down)
}

/// Decode cases keep the fused decode route, which also writes the active
/// expert list, but take the skewed logits instead of the router GEMM.
fn skewed_phase_case(
    session: &Session,
    phase: Phase,
    tokens: usize,
    gate: &mut GpuWeights,
    down: &mut GpuWeights,
) -> LabResult<FullCase> {
    let mut case = FullCase::new(
        session,
        model_mimo26_contract::sm120::config::PROFILE,
        phase,
        tokens,
    )?;
    std::mem::swap(&mut case.gate_weights, gate);
    std::mem::swap(&mut case.down_weights, down);
    case.logits_host = skewed_logits(tokens);
    case.logits = session.upload_f32(&case.logits_host)?;
    if phase == Phase::Decode {
        case.router_gemm = None;
        case.bias_host = (0..EXPERTS)
            .map(|expert| sample(0xc500 ^ expert as u64) * 0.01)
            .collect();
        case.bias = session.upload_f32(&case.bias_host)?;
        return Ok(case);
    }
    if let Some(logits) = prefill_diagnostics::logits(tokens, EXPERTS)? {
        case.logits_host = logits;
        case.logits = session.upload_f32(&case.logits_host)?;
        case.bias_host.fill(0.0);
        case.bias = session.upload_f32(&case.bias_host)?;
    }
    // Legacy prefill entries stop after this many nonempty experts.
    case.active = upload_bytes(session, &(EXPERTS as i32).to_ne_bytes())?;
    Ok(case)
}

#[test]
fn grouped_moe_mimo_prefill_skewed() {
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let bench = std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() == Ok("1");
        let (dram_peak, mma_peak, _, source) = peak_rates(session);
        let gate_variant = variant(
            model_mimo26_contract::sm120::config::PROFILE,
            Phase::Prefill,
            Projection::GateUp,
            InputPath::A16Gather,
            WeightFormat::Mxfp4,
            8192,
        )?;
        let down_variant = variant(
            model_mimo26_contract::sm120::config::PROFILE,
            Phase::Prefill,
            Projection::Down,
            InputPath::A16Gather,
            WeightFormat::Mxfp4,
            8192,
        )?;
        let mut gate = upload_bank(session, &gate_variant, GATE_SEED)?;
        let mut down = upload_bank(session, &down_variant, DOWN_SEED)?;
        let mut failures = replay::Failures::default();
        for tokens in selected_token_counts() {
            let mut case = skewed_case(session, tokens, &mut gate, &mut down)?;
            let name = format!("mimo_prefill_{}_t{tokens}", routing_label());
            let result = (|| {
            let mut checks = case.replay_checks(session, true, true, true, true)?;
            checks.phase(ReplayPhase::Eager, false, || case.launch(session, true))?;
            session.stream.synchronize().map_err(err)?;
            let routing = read_routing(session, &case)?;
            if let Some(root) = std::env::var_os("CHOPINFER_MOE_PREFILL_LOGITS") {
                let expected = fs::read(PathBuf::from(root).join(format!("t{tokens}.ids"))).map_err(err)?;
                let actual: Vec<_> = routing.ids.iter().flat_map(|id| id.to_le_bytes()).collect();
                if actual != expected {
                    return Err("prefill replay did not reproduce captured expert IDs and order".into());
                }
            }
            check_case(session, &case, &routing, &name)?;
            let eager = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
            if std::env::var_os("CHOPINFER_MOE_PREFILL_BASELINE").is_some() {
                prefill_diagnostics::snapshot(&name, &[
                session.download_bf16(&case.gate_output, tokens * TOP_K * 512)?,
                session.download_bf16(&case.down_output, tokens * TOP_K * HIDDEN)?,
                eager.clone(),
                ])?;
            }
            let graph = session.capture(|_| case.launch(session, true))?;
            checks.phase(ReplayPhase::Graph, false, || graph.launch(&session.stream).map_err(err))?;
            let replay = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
            exact_receipt(session, &name, "graph_replay", &replay, &eager)?;
            checks.phase(ReplayPhase::Relaunch, false, || case.launch(session, false))?;
            let non_pdl = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
            exact_receipt(session, &name, "pdl_matches_serialized", &non_pdl, &eager)?;
            checks.phase(ReplayPhase::Pdl, true, || case.launch(session, true))?;
            if bench {
                let full = time(session, 2, 5, |stream| graph.launch(stream).map_err(err))?;
                let gate_graph =
                    session.capture(|_| case.launch_projection(session, Projection::GateUp))?;
                let down_graph =
                    session.capture(|_| case.launch_projection(session, Projection::Down))?;
                let gate_time = time(session, 2, 5, |stream| {
                    gate_graph.launch(stream).map_err(err)
                })?;
                let down_time = time(session, 2, 5, |stream| {
                    down_graph.launch(stream).map_err(err)
                })?;
                let route_graph = session.capture(|_| case.launch_route_stages(session, &case.input, true))?;
                let route_time = time(session, 2, 5, |stream| route_graph.launch(stream).map_err(err))?;
                let combine_graph = session.capture(|_| case.launch_combine_stage(session, true))?;
                let combine_time = time(session, 2, 5, |stream| combine_graph.launch(stream).map_err(err))?;
                let routes = (tokens * TOP_K) as f64;
                let gate_flops = 2.0 * routes * 1024.0 * 4096.0;
                let down_flops = 2.0 * routes * 4096.0 * 512.0;
                let percent = |flops: f64, ms: f32| flops / f64::from(ms) / 1e9 / mma_peak * 100.0;
                // Active experts stream their FP4 codes and compact scales once.
                let active = routing.counts.iter().filter(|&&c| c > 0).count() as f64;
                let gate_bytes = active * (1024.0 * 4096.0 / 2.0) * (1.0 + 1.0 / 16.0);
                let down_bytes = active * (4096.0 * 512.0 / 2.0) * (1.0 + 1.0 / 16.0);
                let dram = |bytes: f64, ms: f32| bytes / f64::from(ms) / 1e6 / dram_peak * 100.0;
                let mut receipt = BTreeMap::new();
                receipt.insert("engine".into(), json!("grouped_moe"));
                receipt.insert("profile".into(), json!("mimo-v26-flash"));
                receipt.insert("phase".into(), json!("prefill"));
                receipt.insert("tokens".into(), json!(tokens));
                receipt.insert("gate_up_entry".into(), json!(case.gate.symbol));
                receipt.insert("down_entry".into(), json!(case.down.symbol));
                receipt.insert(
                    "path".into(),
                    json!(format!(
                        "{}_top8_permute_a16_gateup_down_combine",
                        routing_label()
                    )),
                );
                receipt.insert("routing".into(), routing.stats());
                receipt.insert("expert_rows".into(), json!(routing.counts));
                if std::env::var("CHOPINFER_MOE_PREFILL_CONTEXT").as_deref() == Ok("1") {
                    // MiMo overlap reserves 16 SMs for communication. Changing
                    // the persistent grid must preserve every output element.
                    if case.gate.fragment_prefill() {
                        let mut serving = BTreeMap::new();
                        for (projection, buffer, width, label) in [
                            (Projection::GateUp, &case.gate_output, 512, "gate_up"),
                            (Projection::Down, &case.down_output, HIDDEN, "down"),
                        ] {
                            let mut projection_checks = case.replay_checks(session, false, projection == Projection::GateUp, projection == Projection::Down, false)?;
                            projection_checks.phase(ReplayPhase::Eager, false, || case.launch_projection(session, projection))?;
                            let expected = session.download_bf16(buffer, tokens * TOP_K * width)?;
                            projection_checks.phase(ReplayPhase::Relaunch, true, || case.launch_projection_grid(session, projection, Some(172)))?;
                            let actual = session.download_bf16(buffer, tokens * TOP_K * width)?;
                            exact_receipt(session, &name, &format!("{label}_grid_172"), &actual, &expected)?;
                            let us = prefill_diagnostics::context_time(session, || Ok(()), || {
                                case.launch_projection_grid(session, projection, Some(172))
                            })?;
                            serving.insert(format!("{label}_cold"), us);
                            let us = prefill_diagnostics::context_time(session, || {
                                if projection == Projection::GateUp {
                                    case.launch_route_stages(session, &case.input, true)
                                } else {
                                    case.launch_projection_grid(session, Projection::GateUp, Some(172))
                                }
                            }, || case.launch_projection_grid(session, projection, Some(172)))?;
                            serving.insert(format!("{label}_after_predecessor"), us);
                        }
                        let serving_graph = session.capture(|_| {
                            case.launch_with_input_stages_grid(session, &case.input, true, false, 0, Some(172))
                        })?;
                        checks.phase(ReplayPhase::Graph, false, || serving_graph.launch(&session.stream).map_err(err))?;
                        let actual = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
                        exact_receipt(session, &name, "full_grid_172", &actual, &eager)?;
                        let timing = time(session, 2, 5, |stream| serving_graph.launch(stream).map_err(err))?;
                        serving.insert("full_pdl".into(), timing.median_ms * 1000.0);
                        receipt.insert("serving_grid_172_us".into(), json!(serving));
                        receipt.insert("gate_up_schedule_172".into(), prefill_diagnostics::schedule_grid(&routing.counts, 4, 172));
                        receipt.insert("down_schedule_172".into(), prefill_diagnostics::schedule_grid(&routing.counts, 16, 172));
                    }
                    let mut contexts = BTreeMap::new();
                    for context in ["cold", "after_route", "after_down", "after_gate"] {
                        let us = prefill_diagnostics::context_time(session, || match context {
                            "after_route" => case.launch_route_stages(session, &case.input, true),
                            "after_down" => case.launch_projection(session, Projection::Down),
                            "after_gate" => case.launch_projection(session, Projection::GateUp),
                            _ => Ok(()),
                        }, || case.launch_projection(session, Projection::GateUp))?;
                        contexts.insert(context, us);
                    }
                    let serialized = session.capture(|_| case.launch(session, false))?;
                    let timing = time(session, 2, 5, |stream| serialized.launch(stream).map_err(err))?;
                    receipt.insert("serialized_full_us".into(), json!(timing.median_ms * 1000.0));
                    receipt.insert("gate_up_context_us".into(), json!(contexts));
                }
                receipt.insert("gate_up_schedule".into(), prefill_diagnostics::schedule(&routing.counts, 4));
                receipt.insert("down_schedule".into(), prefill_diagnostics::schedule(&routing.counts, 16));
                receipt.insert("median_us".into(), json!(full.median_ms * 1000.0));
                receipt.insert("min_us".into(), json!(full.min_ms * 1000.0));
                receipt.insert("gate_up_us".into(), json!(gate_time.median_ms * 1000.0));
                receipt.insert("down_us".into(), json!(down_time.median_ms * 1000.0));
                receipt.insert("route_us".into(), json!(route_time.median_ms * 1000.0));
                receipt.insert("combine_us".into(), json!(combine_time.median_ms * 1000.0));
                receipt.insert(
                    "full_percent_of_bf16_peak".into(),
                    json!(percent(gate_flops + down_flops, full.median_ms)),
                );
                receipt.insert(
                    "gate_up_percent_of_bf16_peak".into(),
                    json!(percent(gate_flops, gate_time.median_ms)),
                );
                receipt.insert(
                    "down_percent_of_bf16_peak".into(),
                    json!(percent(down_flops, down_time.median_ms)),
                );
                receipt.insert("logical_gemm_flops".into(), json!(gate_flops + down_flops));
                receipt.insert("weight_bytes".into(), json!(gate_bytes + down_bytes));
                receipt.insert(
                    "gate_up_percent_of_dram_read".into(),
                    json!(dram(gate_bytes, gate_time.median_ms)),
                );
                receipt.insert(
                    "down_percent_of_dram_read".into(),
                    json!(dram(down_bytes, down_time.median_ms)),
                );
                receipt.insert("peak_dram_read_gb_per_second".into(), json!(dram_peak));
                receipt.insert("peak_bf16_tflops".into(), json!(mma_peak));
                receipt.insert("peak_source".into(), json!(source));
                receipt.insert("rounds".into(), json!(full.rounds));
                let path = session.receipt("bench", receipt)?;
                eprintln!(
                    "{name}: route {:.1} us, combine {:.1} us, full {:.1} us, gate/up {:.1} us ({:.1}% mma, {:.1}% dram), down {:.1} us ({:.1}% mma, {:.1}% dram), {} active, {}",
                    route_time.median_ms * 1000.0,
                    combine_time.median_ms * 1000.0,
                    full.median_ms * 1000.0,
                    gate_time.median_ms * 1000.0,
                    percent(gate_flops, gate_time.median_ms),
                    dram(gate_bytes, gate_time.median_ms),
                    down_time.median_ms * 1000.0,
                    percent(down_flops, down_time.median_ms),
                    dram(down_bytes, down_time.median_ms),
                    active,
                    path.display()
                );
            }
            let mut replay_receipt = BTreeMap::from([("case".into(), json!(name))]);
            checks.record(&mut replay_receipt);
            session.receipt("moe_replay", replay_receipt)?;
            Ok(())
            })();
            failures.check(&format!("skewed t{tokens}"), result);
            std::mem::swap(&mut case.gate_weights, &mut gate);
            std::mem::swap(&mut case.down_weights, &mut down);
        }
        failures.finish()
    })
    .expect("MiMo skewed prefill MoE");
}

#[test]
fn grouped_moe_mimo_decode_skewed() {
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let bank = |projection| {
            variant(
                model_mimo26_contract::sm120::config::PROFILE,
                Phase::Decode,
                projection,
                InputPath::A16Gather,
                WeightFormat::Mxfp4,
                1,
            )
        };
        let mut gate = upload_bank(session, &bank(Projection::GateUp)?, GATE_SEED)?;
        let mut down = upload_bank(session, &bank(Projection::Down)?, DOWN_SEED)?;
        let mut failures = replay::Failures::default();
        for tokens in [1, 2, 4, 8, 16] {
            if std::env::var("CHOPINFER_MOE_DECODE_TOKENS")
                .is_ok_and(|value| !value.split(',').any(|v| v == tokens.to_string()))
            {
                continue;
            }
            let name = format!("mimo_decode_skewed_t{tokens}");
            let mut case = skewed_phase_case(session, Phase::Decode, tokens, &mut gate, &mut down)?;
            let result = (|| {
                let mut checks = case.replay_checks(session, true, true, true, true)?;
                checks.phase(ReplayPhase::Eager, false, || case.launch(session, true))?;
                session.stream.synchronize().map_err(err)?;
                let routing = read_routing(session, &case)?;
                check_case(session, &case, &routing, &name)?;
                let eager = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
                let graph = session.capture(|_| case.launch(session, true))?;
                checks.phase(ReplayPhase::Graph, false, || {
                    graph.launch(&session.stream).map_err(err)
                })?;
                let replay = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
                exact_receipt(session, &name, "graph_replay", &replay, &eager)?;
                checks.phase(ReplayPhase::Relaunch, false, || case.launch(session, false))?;
                let non_pdl = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
                exact_receipt(session, &name, "pdl_matches_serialized", &non_pdl, &eager)?;
                checks.phase(ReplayPhase::Pdl, true, || case.launch(session, true))?;
                let mut replay_receipt = BTreeMap::from([("case".into(), json!(name))]);
                checks.record(&mut replay_receipt);
                session.receipt("moe_replay", replay_receipt)?;
                Ok(())
            })();
            failures.check(&format!("skewed t{tokens}"), result);
            std::mem::swap(&mut case.gate_weights, &mut gate);
            std::mem::swap(&mut case.down_weights, &mut down);
        }
        failures.finish()
    })
    .expect("MiMo skewed decode MoE");
}

#[path = "moe_prefill_boundaries.rs"]
mod boundaries;
