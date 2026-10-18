//! The lagged Sinkhorn hyper-connection entries the bundle registers on
//! SM120 and its build compiles (`kernels_sm120::sinkhorn_hc::lagged`):
//! each geometry's mix-collapse, F32 expand and sigmoid-weighted head at a
//! prefill's rows, one cubin per geometry, and its unrounded F32 weighted
//! head, which the final norm reads, in a cubin of its own so adding it
//! kept the first cubin's bytes. The contracts are the model's
//! (`model_deepseek41_contract::hc`).

use kernel_build::CubinSpec;
use kernel_core::sinkhorn_hyper_connection::{
    SinkhornHyperConnectionQuery as Query, StreamHeadMode,
};
use kernels_sm120::backend::Sm120;
use kernels_sm120::sinkhorn_hc::Geometry as Mixer;
use kernels_sm120::sinkhorn_hc::lagged::{self, Entry};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::hc;
use model_deepseek41_contract::serving::PREFILL_TOKENS;

/// The lock family of the lagged Sinkhorn cubins, one per geometry.
pub const FAMILY: &str = "sinkhorn_hc_lagged";
/// The lock family of the F32 weighted head cubins, one per geometry.
pub const HEAD_F32_FAMILY: &str = "sinkhorn_hc_head_f32";

/// The name of `g`'s lagged Sinkhorn cubin.
pub fn cubin_name(g: &Geometry) -> String {
    format!("sinkhorn_hc_{}_sm120", g.kernel_tag)
}

/// The name of `g`'s F32 weighted head cubin.
pub fn head_f32_cubin_name(g: &Geometry) -> String {
    format!("sinkhorn_hc_{}_head_f32_sm120", g.kernel_tag)
}

fn mixer(g: &Geometry) -> Mixer {
    Mixer {
        mix: hc::mix(g),
        capacity: PREFILL_TOKENS,
    }
}

fn f32_head(entry: &Entry) -> bool {
    matches!(
        entry.query,
        Query::Head {
            mode: StreamHeadMode::SigmoidWeightedF32,
            ..
        }
    )
}

/// `g`'s entries of the first cubin: no probe and no F32 head.
pub fn entries(g: &Geometry) -> Vec<Entry> {
    lagged::entries(super::rows::profile(g), mixer(g), false)
        .into_iter()
        .filter(|entry| !f32_head(entry))
        .collect()
}

/// `g`'s F32 weighted head (`hc::head`).
pub fn head_f32_entry(g: &Geometry) -> Entry {
    lagged::head_f32_entry(super::rows::profile(g), mixer(g))
}

/// Every geometry's lagged Sinkhorn cubin, then every geometry's F32
/// weighted head cubin, each with its lock family.
pub fn cubins(geometries: &[Geometry]) -> Vec<(&'static str, CubinSpec)> {
    let served = geometries
        .iter()
        .map(|g| (FAMILY, lagged::cubin_spec(cubin_name(g), &entries(g))));
    let heads = geometries.iter().map(|g| {
        (
            HEAD_F32_FAMILY,
            lagged::cubin_spec(head_f32_cubin_name(g), &[head_f32_entry(g)]),
        )
    });
    served.chain(heads).collect()
}

/// `sm120` with `g`'s lagged Sinkhorn entries registered.
pub fn register(sm120: Sm120, g: &Geometry) -> Sm120 {
    sm120.with_sinkhorn_lagged(registered_entries(g))
}

/// `g`'s entries `register` registers: the first cubin's, then the F32 head.
pub fn registered_entries(g: &Geometry) -> Vec<Entry> {
    entries(g).into_iter().chain([head_f32_entry(g)]).collect()
}
