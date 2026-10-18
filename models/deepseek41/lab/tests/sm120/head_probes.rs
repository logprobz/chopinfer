//! Head-only synthetic geometry and tail probes. The registered head uses
//! only hidden width and vocabulary size; read both from the recipe fixture.
use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::f32_projection::Entry;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub fn synthetic() -> Geometry {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek41/geometries/synthetic-small/geometry.json"
    ))
    .unwrap();
    let mut g = FLASH;
    g.id = "synthetic_small";
    g.kernel_tag = "deepseek41_synthetic_small";
    g.base = false;
    g.shape.hidden = fixture["text_config"]["hidden_size"].as_u64().unwrap() as u32;
    g.shape.vocab = fixture["text_config"]["vocab_size"].as_u64().unwrap() as u32;
    g
}

pub fn control() -> Entry {
    entry(17, 257, 9, "chop_exact_f32_bf16_m9_n17_k257")
}

pub fn tails() -> Vec<Entry> {
    [FLASH.shape.hidden, synthetic().shape.hidden]
        .into_iter()
        .map(|k| entry(17, k, 64, &format!("chop_head_tail_n17_k{k}")))
        .collect()
}

fn entry(n: u32, k: u32, m: u32, stem: &str) -> Entry {
    Entry::new(
        GemmQuery {
            m_capacity: m,
            n,
            k,
            weight: WeightFormat::Bf16,
            activation: ActivationFormat::F32,
            output: DType::F32,
            epilogues: Epilogues::default(),
            grouped: None,
        },
        OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: m,
            graph: false,
        },
        stem,
    )
    .unwrap()
}
