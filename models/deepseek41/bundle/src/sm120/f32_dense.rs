//! The exact F32 dense projections the bundle registers on SM120 beside
//! the head (`f32_head`) and its build compiles
//! (`kernels_sm120::f32_projection`): BF16 checkpoint weights over F32
//! activations into F32 rows (F32-TQ). Their shapes, replicated on every
//! rank: the router (experts from the hidden width), a KV source's
//! compressor `wkv` and `wgate` (the head dimension from the hidden
//! width), an index source's `weights_proj` (its heads from the hidden
//! width) and a KV source's indexer `wk` (the index dimension from the
//! normalized F32 latent). One entry per shape at the served prefill
//! capacity, which every phase asks within (as the dense folds do through
//! M1).

use kernel_build::CubinSpec;
use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, WeightFormat};
use kernels_sm120::f32_projection::{Entry, cubin_spec};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::serving::PREFILL_TOKENS;

pub const FAMILY: &str = "f32_dense";

/// The name of `g`'s dense F32 cubin.
pub fn cubin_name(g: &Geometry) -> String {
    format!("f32_dense_{}_sm120", g.kernel_tag)
}

/// Each distinct `(rows, k)` of `g`'s dense F32 projections.
pub fn shapes(g: &Geometry) -> Vec<(u32, u32)> {
    let s = &g.shape;
    let mut shapes = vec![(s.moe.experts, s.hidden)];
    if !s.csa.kv_sources.is_empty() {
        shapes.push((s.attention.head_dim, s.hidden));
        shapes.push((s.indexer.dim, s.attention.head_dim));
    }
    if !s.csa.index_sources.is_empty() {
        shapes.push((s.indexer.heads, s.hidden));
    }
    let mut distinct = Vec::new();
    for shape in shapes {
        if !distinct.contains(&shape) {
            distinct.push(shape);
        }
    }
    distinct
}

/// The dense F32 projection of `rows` outputs from `k` inputs.
pub fn query(rows: u32, k: u32) -> GemmQuery {
    GemmQuery {
        m_capacity: PREFILL_TOKENS,
        n: rows,
        k,
        weight: WeightFormat::Bf16,
        activation: ActivationFormat::F32,
        output: DType::F32,
        epilogues: Epilogues::default(),
        grouped: None,
    }
}

/// `g`'s dense F32 entries.
pub fn entries(g: &Geometry) -> Result<Vec<Entry>, String> {
    shapes(g)
        .into_iter()
        .map(|(n, k)| {
            let q = query(n, k);
            Entry::new(
                q,
                super::f32_head::usage(),
                &format!(
                    "chop_{}_dense_f32_m{}_n{}_k{}",
                    g.kernel_tag, q.m_capacity, q.n, q.k
                ),
            )
            .map_err(|e| e.to_string())
        })
        .collect()
}

/// Each geometry's dense F32 cubin.
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            let spec = cubin_spec(cubin_name(g), &entries(g)?).map_err(|e| e.to_string())?;
            Ok((FAMILY, spec))
        })
        .collect()
}
