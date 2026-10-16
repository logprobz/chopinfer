//! Real inputs of the Qwen3.8-owned GEMM entries (`gemm::fused::catalog(kernel_lab::catalogs())`):
//! checkpoint weights of each entry family with dumped activations
//! (`accumulation::QwenSources`). Shared by `qwen38_gemm.rs` and
//! `dense_template.rs`, which include it next to `accumulation.rs`.

use super::accumulation;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_lab::LabResult;
use kernels_sm120::gemm::fused::Entry;

/// Real inputs of a Qwen-owned entry: its segments' weights as dense rows
/// [n][k] in segment order, and a pool of activation rows.
pub struct OwnInputs {
    pub weight: Vec<u16>,
    pub rows: Vec<u16>,
    pub origin: accumulation::Origin,
    pub note: &'static str,
}

/// The HC bottleneck the up projection reads: the layer's down projection
/// rounded to BF16, v = down / 4, then v * sigmoid(v) rounded to BF16.
pub fn bottleneck(s: &accumulation::QwenSources) -> LabResult<Vec<u16>> {
    let normed = s.rows(0, "x.hc.attn.xn")?;
    let down = s.weight(&accumulation::QwenSources::name(
        0,
        "attn_hyper_connection.input_mix_weight_down.weight",
    ))?;
    Ok(accumulation::project(&normed.1, &down.1, down.0)
        .into_iter()
        .map(|value| {
            let v = f64::from(bf16_to_f32(f32_to_bf16(value as f32))) / 4.0;
            f32_to_bf16((v / (1.0 + (-v).exp())) as f32)
        })
        .collect())
}

pub fn own_inputs(
    s: &accumulation::QwenSources,
    entry: &Entry,
    family: &str,
    ranks: usize,
) -> LabResult<Option<OwnInputs>> {
    use accumulation::{Origin, QwenSources, block, rows_of};
    let (n, k) = (entry.query.n as usize, entry.query.k as usize);
    let name = QwenSources::name;
    let top = |tensor: &str, count: u32| -> LabResult<Vec<u16>> {
        let weight = s.weight(tensor)?;
        Ok(rows_of(&weight.1, weight.0, count as usize, k))
    };
    let rows_from = |layer: usize, tensor: &str| -> LabResult<Vec<u16>> {
        let dumped = s.rows(layer, tensor)?;
        Ok(block(
            &dumped.1,
            dumped.0,
            0..dumped.1.len() / dumped.0,
            k.min(dumped.0),
        ))
    };
    let segments = |tensors: &[String]| -> LabResult<Vec<u16>> {
        let mut weight = Vec::new();
        for (tensor, &rows) in tensors.iter().zip(&entry.segments) {
            weight.extend(top(tensor, rows)?);
        }
        Ok(weight)
    };
    let dump = |weight: Vec<u16>, rows: Vec<u16>, note: &'static str| OwnInputs {
        weight,
        rows,
        origin: Origin::Dump,
        note,
    };
    Ok(Some(match family {
        "lin_in" | "lin_ab" => {
            let parts: &[&str] = if family == "lin_in" {
                &["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"]
            } else {
                &["in_proj_a", "in_proj_b"]
            };
            let tensors = parts
                .iter()
                .map(|p| name(0, &format!("linear_attn.{p}.weight")))
                .collect::<Vec<_>>();
            dump(
                segments(&tensors)?,
                rows_from(0, "x.hc.attn")?,
                "layer 0 linear-attention input projections; x.hc.attn",
            )
        }
        "attn_in" => {
            let tensors = [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.indexer.index_qk_proj",
            ]
            .map(|p| name(3, &format!("{p}.weight")));
            dump(
                segments(&tensors)?,
                rows_from(3, "x.hc.attn")?,
                "layer 3 q, k, v and indexer; x.hc.attn",
            )
        }
        "out" => {
            let weight = s.weight(&name(0, "linear_attn.out_proj.weight"))?;
            dump(
                block(&weight.1, weight.0, 0..n, k),
                rows_from(0, "attn.core")?,
                "layer 0 out_proj K shard 0; attn.core",
            )
        }
        "router" => dump(
            top(&name(0, "mlp.gate.weight"), n as u32)?,
            rows_from(0, "x.hc.mlp")?,
            "layer 0 router; x.hc.mlp",
        ),
        "hc_in" | "hc_inject" => {
            let parts: &[&str] = if family == "hc_in" {
                &["input_mix_weight_down", "block_inject_weight"]
            } else {
                &["block_inject_weight"]
            };
            let tensors = parts
                .iter()
                .map(|p| name(0, &format!("attn_hyper_connection.{p}.weight")))
                .collect::<Vec<_>>();
            dump(
                segments(&tensors)?,
                rows_from(0, "x.hc.attn.xn")?,
                "layer 0 attention down and inject; x.hc.attn.xn",
            )
        }
        "hc_down" => OwnInputs {
            weight: top(
                "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                n as u32,
            )?,
            rows: rows_from(7, "x.hc.mlp.xn")?,
            origin: Origin::Proxy,
            note: "mixer down; layer 7 x.hc.mlp.xn stands in",
        },
        "ple_key" => OwnInputs {
            weight: top(&name(1, "ple.key_proj.weight"), n as u32)?,
            rows: rows_from(1, "x.hc.attn")?,
            origin: Origin::Proxy,
            note: "layer 1 PLE key; x.hc.attn stands in",
        },
        "lin_qkv" => dump(
            top(&name(0, "linear_attn.in_proj_qkv.weight"), n as u32)?,
            rows_from(0, "x.hc.attn")?,
            "layer 0 in_proj_qkv rows (PLE value and MTP fc shape); x.hc.attn",
        ),
        "shared_gate_up" | "shared_down" => {
            let real = 640 / ranks;
            let gate = s.weight(&name(0, "mlp.shared_expert.gate_proj.weight"))?;
            let up = s.weight(&name(0, "mlp.shared_expert.up_proj.weight"))?;
            let hidden_k = gate.0;
            let gate = rows_of(&gate.1, hidden_k, real, hidden_k);
            let up = rows_of(&up.1, hidden_k, real, hidden_k);
            let mixed = s.rows(0, "x.hc.mlp")?;
            if family == "shared_gate_up" {
                let half = n / 2;
                let mut weight = gate;
                weight.resize(half * k, 0);
                weight.extend(up);
                weight.resize(n * k, 0);
                dump(
                    weight,
                    mixed.1.clone(),
                    "layer 0 shared expert gate and up, zero padded; x.hc.mlp",
                )
            } else {
                let down = s.weight(&name(0, "mlp.shared_expert.down_proj.weight"))?;
                let hidden = accumulation::swiglu_rows(
                    &accumulation::project(&mixed.1, &gate, hidden_k),
                    &accumulation::project(&mixed.1, &up, hidden_k),
                );
                dump(
                    rows_of(&down.1, down.0, n, k),
                    rows_of(&hidden, real, hidden.len() / real, k),
                    "layer 0 shared expert down, zero padded; its SwiGLU of x.hc.mlp",
                )
            }
        }
        "hc_up_decode" | "hc_up_prefill" => dump(
            top(
                &name(0, "attn_hyper_connection.input_mix_weight_up.weight"),
                n as u32,
            )?,
            bottleneck(s)?,
            "layer 0 attention up; its bottleneck of x.hc.attn.xn",
        ),
        _ => return Ok(None),
    }))
}
