//! Row 0 of the MiMo dense decode GEMMs at capacity 1 versus 2, 4 and 8.

use super::{ACTIVATION, Case, query, seeded_values, watched_gpu_test};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use serde_json::json;
use std::collections::BTreeMap;

use crate::verify_drift_stats as stats;

#[test]
fn gemm_verify_drift_row0() {
    watched_gpu_test(2 * 1024 * 1024 * 1024, 300, |session| {
        for name in [
            "qkv_ga",
            "qkv_swa",
            "o",
            "router",
            "head",
            "mlp_gate_up",
            "mlp_down",
        ] {
            let mut decode: Option<(Vec<f32>, Vec<f64>)> = None;
            for m in [1_u32, 2, 4, 8] {
                let symbol = format!("chop_gemm_mimo_{name}_m{m}_v1");
                let case = Case::prepare(session, &symbol, m, true)?;
                let function = case.function(session)?;
                case.launch(session, &function, &session.stream)?;
                let width = case.output_len / m as usize;
                let row = case.output(session)?[..width].to_vec();
                let reference =
                    case.reference.as_ref().ok_or("missing GEMM oracle")?.values[..width].to_vec();
                let bf16 = query(&case.variant).output == DType::Bf16;
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!("verify_drift_row0"));
                receipt.insert("symbol".into(), json!(symbol));
                receipt.insert("split_k".into(), json!(case.counts.split_k));
                receipt.insert(
                    "knobs".into(),
                    json!(format!("{:?}", case.variant.key.knobs)),
                );
                receipt.insert("against_f64".into(), stats::against(&row, &reference, bf16));
                match &decode {
                    None => decode = Some((row, reference)),
                    Some((first, first_reference)) => {
                        if first_reference != &reference {
                            return Err(format!("{symbol}: row 0 reference differs from m1"));
                        }
                        receipt.insert("versus_m1".into(), stats::paths(first, &row, bf16));
                    }
                }
                eprintln!(
                    "{}",
                    serde_json::to_string(&receipt).map_err(|e| e.to_string())?
                );
                session.receipt("verify_drift", receipt)?;
            }
        }
        Ok(())
    })
    .expect("GEMM verify drift");
}

/// The BF16-weight entries sum the fixture's grid-valued inputs exactly in
/// f32, which hides the reduction order. Rerun them with activations spread
/// over thirteen binades so f32 accumulation rounds.
#[test]
fn gemm_verify_drift_row0_wide_bf16() {
    watched_gpu_test(2 * 1024 * 1024 * 1024, 300, |session| {
        for name in ["o", "router", "head"] {
            let mut decode: Option<Vec<f32>> = None;
            let mut reference = Vec::new();
            for m in [1_u32, 2, 4, 8] {
                let symbol = format!("chop_gemm_mimo_{name}_m{m}_v1");
                let case = Case::prepare(session, &symbol, m, false)?;
                let q = query(&case.variant);
                let (n, k) = (q.n as usize, q.k as usize);
                let activation = (0..q.m_capacity as usize * k)
                    .map(|index| {
                        let binade = ((index as u64).wrapping_mul(2_654_435_761) >> 7) % 13;
                        f32_to_bf16(
                            seeded_values(1, 0x5157 ^ index as u64)[0]
                                * 64.0
                                * (-(binade as f32)).exp2(),
                        )
                    })
                    .collect::<Vec<_>>();
                let bytes = activation
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>();
                case.buffers[&ACTIVATION]
                    .copy_from_bytes(&bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
                if reference.is_empty() {
                    let weight = seeded_values(n * k, 0x0047_5745_4947_4854);
                    reference = (0..n)
                        .map(|row| {
                            (0..k)
                                .map(|i| {
                                    f64::from(bf16_to_f32(activation[i]))
                                        * f64::from(bf16_to_f32(f32_to_bf16(weight[row * k + i])))
                                })
                                .sum::<f64>()
                        })
                        .collect();
                }
                let function = case.function(session)?;
                case.launch(session, &function, &session.stream)?;
                let row = case.output(session)?[..n].to_vec();
                let bf16 = q.output == DType::Bf16;
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!("verify_drift_row0_wide"));
                receipt.insert("symbol".into(), json!(symbol));
                receipt.insert("split_k".into(), json!(case.counts.split_k));
                receipt.insert("against_f64".into(), stats::against(&row, &reference, bf16));
                match &decode {
                    None => decode = Some(row),
                    Some(first) => {
                        receipt.insert("versus_m1".into(), stats::paths(first, &row, bf16));
                    }
                }
                eprintln!(
                    "{}",
                    serde_json::to_string(&receipt).map_err(|e| e.to_string())?
                );
                session.receipt("verify_drift", receipt)?;
            }
        }
        Ok(())
    })
    .expect("GEMM verify drift, wide BF16 inputs");
}
