//! Token 0 of the MiMo grouped expert decode path at capacity 1 versus 8.
//! Both cases route token 0 to the same experts with identical synthetic
//! MXFP4 weights and input. The down projection and the combine are rerun
//! on the capacity-8 case with the capacity-1 gate output and route weights
//! copied in, so each stage is compared on identical inputs.

use super::{
    FullCase, GpuWeights, HIDDEN, MoeOperands, TOP_K, bf16_f32, download_i32, err, f32_bytes,
    i32_bytes, launch_moe, load_moe, mxfp4_gate_up, read_mimo_rank0_gateup_checkpoint, sample,
    upload_bytes, variant,
};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{RuntimeCounts, SwigluColumns, WeightFormat};
use kernel_core::oracle::{self, ExpertRowRange, GemmEpilogue, GemmShape};
use kernel_core::transform::Mxfp4MoeFragmentLayout;
use kernel_lab::{LabResult, Session, gpu_test};
use kernels_sm120::moe::{InputPath, Phase, Projection};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::verify_drift_stats as stats;

fn report(
    session: &Session,
    input: &str,
    stage: &str,
    fields: Vec<(&str, Value)>,
) -> LabResult<()> {
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!("verify_drift_moe"));
    receipt.insert("stage".into(), json!(stage));
    receipt.insert("input".into(), json!(input));
    for (key, value) in fields {
        receipt.insert(key.into(), value);
    }
    eprintln!("{}", serde_json::to_string(&receipt).map_err(err)?);
    session.receipt("verify_drift", receipt)?;
    Ok(())
}

fn replace_prefix_bf16(
    session: &Session,
    rows: &[u16],
    target: &runtime_cuda::DeviceBuffer,
) -> LabResult<()> {
    let mut all = session.download_bf16(target, target.byte_len() / 2)?;
    all[..rows.len()].copy_from_slice(rows);
    let bytes = all.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>();
    target.copy_from_bytes(&bytes, &session.stream).map_err(err)
}

/// Replace the fixture input with values spread over thirteen binades, so
/// that f32 accumulation rounds and a changed summation order can show. The
/// fixture's own input sums exactly, which hides any order difference.
fn widen_input(session: &Session, case: &mut FullCase) -> LabResult<()> {
    case.input_host = (0..case.tokens * HIDDEN)
        .map(|index| {
            let binade = ((index as u64).wrapping_mul(2_654_435_761) >> 7) % 13;
            f32_to_bf16(sample(0x5a17 ^ index as u64) * 0.25 * (-(binade as f32)).exp2())
        })
        .collect();
    let bytes = case
        .input_host
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect::<Vec<_>>();
    case.input
        .copy_from_bytes(&bytes, &session.stream)
        .map_err(err)
}

#[test]
fn grouped_moe_verify_drift_token0() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        for wide in [false, true] {
            compare(session, wide)?;
        }
        Ok(())
    })
    .expect("MoE verify drift");
}

fn compare(session: &Session, wide: bool) -> LabResult<()> {
    let mut one = FullCase::new(
        session,
        model_mimo26_contract::sm120::config::PROFILE,
        Phase::Decode,
        1,
    )?;
    let mut eight = FullCase::new(
        session,
        model_mimo26_contract::sm120::config::PROFILE,
        Phase::Decode,
        8,
    )?;
    if wide {
        widen_input(session, &mut one)?;
        widen_input(session, &mut eight)?;
    }
    for (label, case) in [("t1", &one), ("t8", &eight)] {
        report(
            session,
            if wide { "wide" } else { "fixture" },
            "variants",
            vec![
                ("tokens", json!(label)),
                ("gate_symbol", json!(format!("{:?}", case.gate.symbol))),
                ("gate_knobs", json!(format!("{:?}", case.gate.knobs))),
                ("down_symbol", json!(format!("{:?}", case.down.symbol))),
                ("down_knobs", json!(format!("{:?}", case.down.knobs))),
            ],
        )?;
        case.launch(session, true)?;
    }
    let intermediate = one.gate.query.n as usize / 2;
    let ids_one = download_i32(session, &one.ids, TOP_K)?;
    let ids_eight = download_i32(session, &eight.ids, TOP_K)?;
    if ids_one != ids_eight {
        return Err(format!(
            "token 0 routes differ: {ids_one:?} vs {ids_eight:?}"
        ));
    }
    let topk_one = session.download_f32(&one.topk, TOP_K)?;
    let topk_eight = session.download_f32(&eight.topk, TOP_K)?;
    report(
        session,
        if wide { "wide" } else { "fixture" },
        "route_weights",
        vec![("t8_versus_t1", stats::paths(&topk_one, &topk_eight, false))],
    )?;
    let ranges = (0..TOP_K)
        .map(|row| ExpertRowRange {
            expert: ids_one[row] as usize & 1,
            start: row,
            end: row + 1,
        })
        .collect::<Vec<_>>();

    let gate_rows = TOP_K * intermediate;
    let gate_one = session.download_bf16(&one.gate_output, gate_rows)?;
    let gate_eight = session.download_bf16(&eight.gate_output, gate_rows)?;
    let gate_input = (0..TOP_K)
        .flat_map(|_| one.input_host[..HIDDEN].iter())
        .map(|&v| f64::from(bf16_to_f32(v)))
        .collect::<Vec<_>>();
    let gate_reference = oracle::gemm(
        &gate_input,
        &one.gate_decoded.concat(),
        GemmShape {
            m: TOP_K,
            n: one.gate.query.n as usize,
            k: HIDDEN,
            experts: 2,
        },
        &ranges,
        GemmEpilogue {
            residual: None,
            swiglu: Some(SwigluColumns::ConcatenatedHalves),
        },
    )
    .map_err(err)?;
    let (gate_one_f, gate_eight_f) = (bf16_f32(&gate_one), bf16_f32(&gate_eight));
    report(
        session,
        if wide { "wide" } else { "fixture" },
        "gate_up",
        vec![
            (
                "t8_versus_t1",
                stats::paths(&gate_one_f, &gate_eight_f, true),
            ),
            (
                "t1_against_f64",
                stats::against(&gate_one_f, &gate_reference.values, true),
            ),
            (
                "t8_against_f64",
                stats::against(&gate_eight_f, &gate_reference.values, true),
            ),
        ],
    )?;

    replace_prefix_bf16(session, &gate_one, &eight.gate_output)?;
    let mut topk_all = session.download_f32(&eight.topk, 8 * TOP_K)?;
    topk_all[..TOP_K].copy_from_slice(&topk_one);
    eight
        .topk
        .copy_from_bytes(&f32_bytes(&topk_all), &session.stream)
        .map_err(err)?;
    eight.launch_projection(session, Projection::Down)?;
    let down_rows = TOP_K * HIDDEN;
    let down_one = bf16_f32(&session.download_bf16(&one.down_output, down_rows)?);
    let down_eight = bf16_f32(&session.download_bf16(&eight.down_output, down_rows)?);
    let down_input = gate_one_f.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let mut down_reference = oracle::gemm(
        &down_input,
        &one.down_decoded.concat(),
        GemmShape {
            m: TOP_K,
            n: HIDDEN,
            k: intermediate,
            experts: 2,
        },
        &ranges,
        GemmEpilogue {
            residual: None,
            swiglu: None,
        },
    )
    .map_err(err)?;
    for (at, value) in down_reference.values.iter_mut().enumerate() {
        *value *= f64::from(topk_one[at / HIDDEN]);
    }
    report(
        session,
        if wide { "wide" } else { "fixture" },
        "weighted_down",
        vec![
            ("t8_versus_t1", stats::paths(&down_one, &down_eight, true)),
            (
                "t1_against_f64",
                stats::against(&down_one, &down_reference.values, true),
            ),
            (
                "t8_against_f64",
                stats::against(&down_eight, &down_reference.values, true),
            ),
        ],
    )?;

    let down_one_bits = session.download_bf16(&one.down_output, down_rows)?;
    replace_prefix_bf16(session, &down_one_bits, &eight.down_output)?;
    eight.launch_combine_row(session, 8, true)?;
    let final_one = bf16_f32(&session.download_bf16(&one.final_output, HIDDEN)?);
    let final_eight = bf16_f32(&session.download_bf16(&eight.final_output, HIDDEN)?);
    let combine_reference = (0..HIDDEN)
        .map(|col| {
            (0..TOP_K)
                .map(|row| f64::from(down_one[row * HIDDEN + col]))
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    report(
        session,
        if wide { "wide" } else { "fixture" },
        "combine",
        vec![
            ("t8_versus_t1", stats::paths(&final_one, &final_eight, true)),
            (
                "t1_against_f64",
                stats::against(&final_one, &combine_reference, true),
            ),
            (
                "t8_against_f64",
                stats::against(&final_eight, &combine_reference, true),
            ),
        ],
    )?;
    Ok(())
}

/// Layer 1 expert 0 (rank 0 slice) from the checkpoint: token 0 routed to
/// the expert alone at capacity 1, and together with seven more tokens at
/// capacity 8, as a verify item whose rows share an expert.
#[test]
fn grouped_moe_verify_drift_real_gate_up() {
    gpu_test(1024 * 1024 * 1024, |session| {
        let Some(root) = std::env::var_os("CHOPINFER_MIMO_CHECKPOINT").map(PathBuf::from) else {
            kernel_lab::skip(
                "grouped_moe_verify_drift_real_gate_up: set CHOPINFER_MIMO_CHECKPOINT",
            )?;
            return Ok(());
        };
        let (codes, scales) = read_mimo_rank0_gateup_checkpoint(&root)?;
        let fragment = Mxfp4MoeFragmentLayout {
            experts: 1,
            rows_per_expert: 1024,
            k: HIDDEN,
        };
        let packed_codes = fragment.pack_codes(&codes).map_err(err)?.bytes;
        let packed_scales = fragment.pack_scales(&scales).map_err(err)?.bytes;
        let full_codes = session.allocate(256 * codes.len())?;
        full_codes
            .copy_from_bytes(&packed_codes, &session.stream)
            .map_err(err)?;
        let weights = GpuWeights {
            codes: full_codes,
            scales: upload_bytes(session, &packed_scales)?,
            f32_scales: session.upload_f32(&[1.0])?,
        };
        let input = (0..8 * HIDDEN)
            .map(|index| {
                let binade = ((index as u64).wrapping_mul(2_654_435_761) >> 7) % 8;
                f32_to_bf16(sample(0xd711 ^ index as u64) * 2.0 * (-(binade as f32)).exp2())
            })
            .collect::<Vec<_>>();
        let reference = mxfp4_gate_up(&input, 1, &codes, &scales)?;
        let mut rows = Vec::new();
        for tokens in [1_usize, 8] {
            let v = variant(
                model_mimo26_contract::sm120::config::PROFILE,
                Phase::Decode,
                Projection::GateUp,
                InputPath::A16Gather,
                WeightFormat::Mxfp4,
                tokens,
            )?;
            let routes = tokens * TOP_K;
            let mut offsets = vec![16_i32; 257];
            offsets[0] = 0;
            let mut sorted = vec![-1_i32; 16];
            for (token, slot) in sorted.iter_mut().take(tokens).enumerate() {
                *slot = (token * TOP_K) as i32;
            }
            let output = session.allocate(routes * 512 * 2)?;
            let counts = RuntimeCounts {
                m: tokens as u32,
                batch: 0,
                context: 0,
                routed_rows: routes as u32,
                split_k: v.knobs.split_k_capacity,
            };
            let operands = MoeOperands {
                activation: &session.upload_bf16(&input[..tokens * HIDDEN])?,
                activation_scales: &upload_bytes(session, &[127])?,
                activation_f32_scales: &session.upload_f32(&[1.0])?,
                weights: &weights,
                output: &output,
                topk_weights: &session.upload_f32(&vec![1.0; routes])?,
                offsets: &upload_bytes(session, &i32_bytes(&offsets))?,
                sorted: &upload_bytes(session, &i32_bytes(&sorted))?,
                padded: &upload_bytes(session, &i32_bytes(&[16]))?,
                live: &upload_bytes(session, &i32_bytes(&[tokens as i32]))?,
                active: &upload_bytes(session, &i32_bytes(&[1, 0]))?,
                quantized_output: &upload_bytes(session, &[0])?,
                quantized_scales: &upload_bytes(session, &[0])?,
                partials: &session
                    .allocate(v.plan(counts).map_err(err)?.workspace_bytes.max(1) as usize)?,
            };
            let functions = load_moe(session, &v, counts)?;
            launch_moe(session, &v, &functions, counts, &operands, true)?;
            let row = bf16_f32(&session.download_bf16(&output, 512)?);
            report(
                session,
                "real_layer1_expert0",
                "gate_up",
                vec![
                    ("tokens", json!(tokens)),
                    ("symbol", json!(v.symbol)),
                    ("knobs", json!(format!("{:?}", v.knobs))),
                    ("against_f64", stats::against(&row, &reference, true)),
                ],
            )?;
            rows.push(row);
        }
        report(
            session,
            "real_layer1_expert0",
            "gate_up",
            vec![("t8_versus_t1", stats::paths(&rows[0], &rows[1], true))],
        )
    })
    .expect("MoE real-weight verify drift");
}
