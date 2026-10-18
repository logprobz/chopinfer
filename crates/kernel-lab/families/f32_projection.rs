//! Exact F32 dense projections at representative widths and fold boundaries.
use kernel_build::CubinSpec;
use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::f32_projection::{Entry, cubin_spec};

pub const FAMILY: &str = "f32_projection";
pub const CUBIN: &str = "f32_projection_v1";

pub fn entries() -> Vec<Entry> {
    let mut entries = Vec::new();
    for (n, k, weight) in [
        (512, 5120, WeightFormat::Bf16),
        (128, 512, WeightFormat::Bf16),
        (32, 5120, WeightFormat::Bf16),
        (16, 5120, WeightFormat::Bf16),
        (8, 5120, WeightFormat::Bf16),
        (384, 5120, WeightFormat::Bf16),
        (384, 5120, WeightFormat::F32),
    ] {
        for m in [1, 17] {
            entries.push(entry(n, k, m, weight));
        }
    }
    for k in [63, 64, 65, 127, 128, 129, 255, 256, 257] {
        entries.push(entry(17, k, 9, WeightFormat::Bf16));
    }
    entries.push(entry(17, 257, 9, WeightFormat::F32));
    entries
}

pub fn entry(n: u32, k: u32, m: u32, weight: WeightFormat) -> Entry {
    let tag = if weight == WeightFormat::F32 {
        "f32"
    } else {
        "bf16"
    };
    Entry::new(
        GemmQuery {
            m_capacity: m,
            n,
            k,
            weight,
            activation: ActivationFormat::F32,
            output: DType::F32,
            epilogues: Epilogues::default(),
            grouped: None,
        },
        OperationUse {
            phase: if m == 1 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            capacity: m,
            graph: false,
        },
        &format!("chop_exact_f32_{tag}_m{m}_n{n}_k{k}"),
    )
    .expect("lab F32 shape")
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[CUBIN],
        vec![cubin_spec(CUBIN, &entries()).expect("lab F32 entries")],
    )
}
