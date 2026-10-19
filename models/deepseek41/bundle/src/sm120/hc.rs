//! The lagged Sinkhorn hyper-connection entries the bundle registers on
//! SM120 and its build compiles (`kernels_sm120::sinkhorn_hc::lagged`):
//! each geometry's mix-collapse, F32 expand and sigmoid-weighted head at a
//! prefill's rows, one cubin per geometry, and its unrounded F32 weighted
//! head, which the final norm reads, in a cubin of its own so adding it
//! kept the first cubin's bytes. The contracts are the model's
//! (`model_deepseek41_contract::hc`). A separate split family serves
//! buckets of at most 32 rows. Prefill buckets from 256 rows share fn
//! loads across eight rows; intermediate buckets keep the original mixer.

use kernel_build::CubinSpec;
use kernel_core::sinkhorn_hyper_connection::{
    SinkhornHyperConnectionQuery as Query, StreamHeadMode,
};
use kernels_sm120::backend::Sm120;
use kernels_sm120::sinkhorn_hc::Geometry as Mixer;
use kernels_sm120::sinkhorn_hc::lagged::{self, Entry};
use kernels_sm120::sinkhorn_hc::{lagged_split, lagged_tiled};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::hc;
use model_deepseek41_contract::serving::PREFILL_TOKENS;

/// The lock family of the lagged Sinkhorn cubins, one per geometry.
pub const FAMILY: &str = "sinkhorn_hc_lagged";
/// The lock family of the F32 weighted head cubins, one per geometry.
pub const HEAD_F32_FAMILY: &str = "sinkhorn_hc_head_f32";
/// Decode-only split entries, isolated from the existing cubins and locks.
pub const SPLIT_FAMILY: &str = "sinkhorn_hc_lagged_split";
pub const SPLIT_ROWS: u32 = 32;

/// Prefill-only row tiles, isolated from the existing cubins and locks.
pub const PREFILL_FAMILY: &str = "sinkhorn_hc_lagged_prefill";
pub const PREFILL_FROM_ROWS: u32 = 256;
pub const PREFILL_TILE: u32 = 8;

pub fn prefill_entry(g: &Geometry) -> lagged_tiled::Entry {
    lagged_tiled::entry(
        super::rows::profile(g),
        mixer(g),
        PREFILL_TILE,
        PREFILL_FROM_ROWS,
        false,
    )
}

pub fn prefill_cubins(geometries: &[Geometry]) -> Vec<(&'static str, CubinSpec)> {
    geometries
        .iter()
        .map(|g| {
            (
                PREFILL_FAMILY,
                lagged_tiled::cubin_spec(
                    format!("sinkhorn_hc_{}_lagged_prefill_sm120", g.kernel_tag),
                    &[prefill_entry(g)],
                )
                .expect("the lagged prefill geometry is valid"),
            )
        })
        .collect()
}

pub fn split_entry(g: &Geometry) -> lagged_split::Entry {
    lagged_split::entry(
        super::rows::profile(g),
        Mixer {
            capacity: SPLIT_ROWS,
            ..mixer(g)
        },
        false,
    )
}

pub fn split_cubins(geometries: &[Geometry]) -> Vec<(&'static str, CubinSpec)> {
    geometries
        .iter()
        .map(|g| {
            (
                SPLIT_FAMILY,
                lagged_split::cubin_spec(
                    format!("sinkhorn_hc_{}_lagged_split_sm120", g.kernel_tag),
                    &[split_entry(g)],
                )
                .expect("the lagged split geometry is valid"),
            )
        })
        .collect()
}

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
/// weighted head and decode split cubins, each with its lock family.
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
    served
        .chain(heads)
        .chain(split_cubins(geometries))
        .chain(prefill_cubins(geometries))
        .collect()
}

/// `sm120` with `g`'s lagged Sinkhorn entries registered.
pub fn register(sm120: Sm120, g: &Geometry) -> Sm120 {
    let sm120 = sm120
        .with_sinkhorn_lagged(registered_entries(g))
        .with_sinkhorn_lagged_split([split_entry(g)])
        .with_sinkhorn_lagged_tiled([prefill_entry(g)]);
    if g.id == model_deepseek41_contract::geometry::FLASH.id {
        sm120.with_sinkhorn_lagged_warp(kernels_sm120::sinkhorn_hc::lagged_warp::Entry {
            source: super::hc_mixer_probe::entry(g),
            buckets: vec![1, 6, 8, 32],
        })
    } else {
        sm120
    }
}

/// `g`'s original entries: the first cubin's, then the F32 head.
pub fn registered_entries(g: &Geometry) -> Vec<Entry> {
    entries(g).into_iter().chain([head_f32_entry(g)]).collect()
}
