//! The bundle's SM120 module: the entries the build compiles and their
//! registration on `Sm120`, and the `OwnOps` implementation over the model's
//! own cubins. Each engine adds one registration file and its module line
//! here. Generic code names nothing in this module beyond constructing the
//! backend and the own ops.

pub mod dspark_entries;
pub mod latent;
use kernels_sm120::backend::{F32ProjectionOffers, Sm120};

/// The build's identity of the exact F32 projection cubins (`f32_head`,
/// `f32_dense`), which authenticates their offer.
const F32_DEPENDENCY: &str = include_str!(concat!(env!("OUT_DIR"), "/f32_dependency.txt"));

// The model's own kernels: their entries and the `OwnOps` implementation.
#[allow(
    dead_code,
    reason = "instance creation prepares the own ops once the program lands"
)]
pub(crate) mod own_entries;
#[allow(
    dead_code,
    reason = "instance creation prepares the own ops once the program lands"
)]
pub(crate) mod own_ops;
// The build compiles each registration file's cubins; the bundle registers
// their entries.
#[allow(dead_code, reason = "the build compiles the entries and declarations")]
mod compressor;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
pub(crate) mod encoder;
#[allow(
    dead_code,
    reason = "the program acquires the encoder once media steps land"
)]
pub(crate) mod encoders;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod f32_dense;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
pub mod f32_head;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod hc;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod indexer;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod indexer_query;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod latent_glue;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod latent_window_stage;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod reduce;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod rows;
mod widen;

/// The serving latent cubins and symbols, for the model's kernel lab.
pub fn latent_cubins()
-> impl Iterator<Item = (&'static str, &'static [u8], &'static [&'static str])> {
    crate::CUBINS
        .iter()
        .filter(|cubin| cubin.name.starts_with("latent_sources_"))
        .map(|cubin| (cubin.name, cubin.bytes, cubin.symbols))
}

/// The model-owned module holding a selected fold entry, for module loading
/// and the model's kernel tests. Rank-independent cubins serve every topology.
pub fn fold_cubin_for_symbol(symbol: &str) -> Option<(&'static str, &'static [u8])> {
    crate::CUBINS
        .iter()
        .find(|c| c.symbols.contains(&symbol))
        .map(|c| (c.name, c.bytes))
}

/// Locked PDL producers for the model's kernel lab.
pub fn dependency_probes() -> (&'static [u8], &'static [&'static str]) {
    let cubin = crate::CUBINS
        .iter()
        .find(|cubin| cubin.name == kernels_sm120::probes::CUBIN)
        .expect("embedded dependency probes");
    (cubin.bytes, cubin.symbols)
}

/// Register the exact F32 projections of `geometry` at `ranks`, the head's
/// and the dense ones, as one offer over the symbols their loaded cubins
/// hold, authenticated by their build dependency. The model lab exercises
/// the serving registration through it.
pub fn register_f32_projections(
    backend: Sm120,
    geometry: &model_deepseek41_contract::geometry::Geometry,
    ranks: u32,
) -> Result<Sm120, String> {
    let mut entries = f32_dense::entries(geometry)?;
    entries.push(f32_head::entry(geometry, ranks)?);
    let cubins = [
        f32_head::cubin_name(geometry),
        f32_dense::cubin_name(geometry),
    ];
    let available = crate::CUBINS
        .iter()
        .filter(|cubin| cubins.iter().any(|name| cubin.name == name))
        .flat_map(|cubin| cubin.symbols.iter().map(|s| (*s).to_owned()))
        .collect();
    let offers = F32ProjectionOffers::new(entries, available, F32_DEPENDENCY.to_owned())
        .map_err(|e| e.to_string())?;
    Ok(backend.with_f32_projections(offers))
}

impl crate::Kernels {
    /// The SM120 backend over the entries the build compiles for `shape`,
    /// and the model's own kernels.
    pub(crate) fn new(shape: &crate::shape::Shape) -> Result<Self, String> {
        let sm120 = rows::register(Sm120::new(), shape.geometry, shape.ranks());
        let sm120 = reduce::register(sm120, shape.geometry);
        let sm120 = hc::register(sm120, shape.geometry);
        let sm120 = latent::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = indexer::register(sm120, shape.geometry);
        let sm120 = compressor::CompressorKernels::serving(shape.geometry)
            .map_err(|e| e.to_string())?
            .register(sm120);
        let sm120 = indexer_query::register(sm120, shape.geometry);
        let sm120 = latent_glue::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = latent_window_stage::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = register_f32_projections(sm120, shape.geometry, shape.ranks())?;
        // The vision encoder's shared entries (`crate::encoders`).
        let sm120 = encoder::register(sm120)?;
        Ok(Self {
            backend: Box::new(sm120),
            own: Box::new(own_ops::Sm120Own::new(shape.geometry)),
        })
    }
}
