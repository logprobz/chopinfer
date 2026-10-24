//! Separate-span exact-F32 producers for bounded Flash TP4 verification.
use kernel_build::CubinSpec;
use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, Refusal, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::f32_projection::{
    Entry,
    cached_spans::{self, Candidate},
};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "f32_verify_spans";
pub const CUBIN: &str = "f32_verify_spans_deepseek41_sm120";

pub fn entry(n: u32, k: u32) -> Result<Candidate, Refusal> {
    Candidate::new(
        Entry::new(
            GemmQuery {
                m_capacity: cached_spans::CAPACITY,
                n,
                k,
                weight: WeightFormat::Bf16,
                activation: ActivationFormat::F32,
                output: DType::F32,
                epilogues: Epilogues::default(),
                grouped: None,
            },
            OperationUse {
                phase: ExecutionPhase::Verify,
                capacity: cached_spans::CAPACITY,
                graph: true,
            },
            &format!("chop_{}_verify_f32_n{n}_k{k}", FLASH.kernel_tag),
        )?
        .with_decode(),
    )
}

pub fn router() -> Result<Candidate, Refusal> {
    entry(FLASH.shape.moe.experts, FLASH.shape.hidden)
}

pub fn entries() -> Result<Vec<Candidate>, Refusal> {
    let s = &FLASH.shape;
    [
        (s.moe.experts, s.hidden),
        (s.vocab / 4, s.hidden),
        (s.attention.head_dim, s.hidden),
        (s.indexer.dim, s.attention.head_dim),
        (s.indexer.heads, s.hidden),
    ]
    .into_iter()
    .map(|(n, k)| entry(n, k))
    .collect()
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(vec![]);
    }
    Ok(vec![(
        FAMILY,
        cached_spans::cubin_spec(CUBIN, &entries().map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?,
    )])
}
