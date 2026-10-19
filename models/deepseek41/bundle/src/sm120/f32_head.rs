//! The LM head's exact F32 projection the bundle registers on SM120 and
//! its build compiles (`kernels_sm120::f32_projection`): the vendor's
//! `F.linear(x.float(), head.weight.float())` over the rank's vocabulary
//! shard of BF16 head rows, F32 activations and F32 logits. One entry per
//! supported rank count, at the served prefill capacity, which every phase
//! asks within (as the dense folds do through M1).

use kernel_build::CubinSpec;
use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::f32_projection::{Entry, cubin_spec};
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::serving::PREFILL_TOKENS;

pub const FAMILY: &str = "f32_head";

/// The name of `g`'s head cubin.
pub fn cubin_name(g: &Geometry) -> String {
    format!("f32_head_{}_sm120", g.kernel_tag)
}

/// The use every head entry serves: the served prefill capacity.
pub const fn usage() -> OperationUse {
    OperationUse {
        phase: ExecutionPhase::Prefill,
        capacity: PREFILL_TOKENS,
        graph: false,
    }
}

/// The head's projection of `g` at `ranks` ranks: the rank's vocabulary
/// rows of the hidden width.
pub fn query(g: &Geometry, ranks: u32) -> GemmQuery {
    GemmQuery {
        m_capacity: PREFILL_TOKENS,
        n: g.shape.vocab / ranks,
        k: g.shape.hidden,
        weight: WeightFormat::Bf16,
        activation: ActivationFormat::F32,
        output: DType::F32,
        epilogues: Epilogues::default(),
        grouped: None,
    }
}

/// The head entry of `g` at `ranks` ranks.
pub fn entry(g: &Geometry, ranks: u32) -> Result<Entry, String> {
    check_ranks(g, ranks)?;
    let q = query(g, ranks);
    Entry::new(
        q,
        usage(),
        &format!(
            "chop_{}_head_f32_m{}_n{}_k{}",
            g.kernel_tag, q.m_capacity, q.n, q.k
        ),
    )
    .map_err(|e| e.to_string())
}

/// The serving head caches its weight-domain table at create and selects
/// the ordered split entry for up to 32 live-row capacity. Its original
/// cubin remains the locked comparison baseline.
pub fn decode_entry(g: &Geometry, ranks: u32) -> Result<Entry, String> {
    entry(g, ranks).map(Entry::with_decode)
}

/// Each geometry's head cubin, with every supported rank count's entry.
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        let mut entries: Vec<Entry> = Vec::new();
        for &ranks in SUPPORTED_RANKS {
            let Ok(entry) = entry(g, ranks) else {
                continue;
            };
            if !entries.iter().any(|e| e.symbol() == entry.symbol()) {
                entries.push(entry);
            }
        }
        specs.push((
            FAMILY,
            cubin_spec(cubin_name(g), &entries).map_err(|e| e.to_string())?,
        ));
    }
    Ok(specs)
}
