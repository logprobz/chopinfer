//! The bundle's SM120 module: the entries the build compiles and their
//! registration on `Sm120`, and the `OwnOps` implementation over the model's
//! own cubins. Each engine adds one registration file and its module line
//! here. Generic code names nothing in this module beyond constructing the
//! backend and the own ops.

pub mod dspark;
#[allow(dead_code, reason = "the build compiles the drafter auxiliary entries")]
mod dspark_aux;
pub mod dspark_entries;
#[allow(dead_code, reason = "the build compiles the drafter router")]
mod dspark_router;
pub mod latent;
pub mod mxfp8_fold;
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
#[allow(dead_code, reason = "the build compiles the entries and declarations")]
mod compressor_bf16;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod dense;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
pub(crate) mod encoder;
pub(crate) mod encoders;
pub mod f32_decode;
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
#[allow(dead_code)]
mod latent_window_bf16;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod latent_window_stage;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod moe;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod reduce;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod routed_sum;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod rows;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod shared;
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
    let entries = f32_decode::entries(geometry, ranks)?;
    let cubins = [f32_decode::cubin_name(geometry)];
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
    /// Entries used only by the drafter's smaller expert bank. An AR-only
    /// instance has no consumer for these modules.
    pub fn drafter_only(cubin: &str, g: &model_deepseek41_contract::geometry::Geometry) -> bool {
        cubin == dspark_router::cubin_name(g)
            || dspark_router::sum_cubins(g)
                .iter()
                .any(|name| name == cubin)
    }

    /// Whether an instance loads `cubin` with the program's cubins: every
    /// cubin but the dependency probes, which only the kernel lab launches,
    /// and the vision encoder's, which only an encoder instance loads.
    /// Streamed MoE modules load only for the qualified topology.
    pub fn loads(cubin: &str, shape: &crate::shape::Shape) -> bool {
        cubin != kernels_sm120::probes::CUBIN
            && !encoder::is_cubin(cubin)
            && !cubin.starts_with("prefill_moe_")
            && (!cubin.starts_with("streamed_moe_")
                || moe::streamed_qualified(shape.geometry, shape.ranks()))
    }

    /// Model-qualified preparation ranges, in increasing capacity within
    /// each phase. Unchanged topologies retain their original C1 override.
    pub(crate) fn moe_buckets(
        shape: &crate::shape::Shape,
    ) -> &'static [(kernel_core::precision::ExecutionPhase, u32)] {
        use kernel_core::precision::ExecutionPhase::{Decode, Verify};
        if moe::streamed_qualified(shape.geometry, shape.ranks()) {
            &[(Decode, 1), (Decode, 8), (Verify, 8), (Verify, 64)]
        } else {
            &[(Decode, 1)]
        }
    }

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
        let sm120 = compressor_bf16::register(sm120, shape.geometry)?;
        let sm120 = moe::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = routed_sum::register(sm120, shape.geometry);
        let sm120 = dspark_router::register(sm120, shape.geometry);
        let sm120 = dense::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = shared::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = mxfp8_fold::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = indexer_query::register(sm120, shape.geometry);
        let sm120 = latent_glue::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = latent_window_stage::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = dspark_aux::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = latent_window_bf16::register(sm120);
        let sm120 = register_f32_projections(sm120, shape.geometry, shape.ranks())?;
        // The vision encoder's shared entries (`crate::encoders`).
        let sm120 = encoder::register(sm120)?;
        Ok(Self {
            backend: Box::new(sm120),
            own: Box::new(own_ops::Sm120Own::new(shape.geometry)),
            gather_state: [
                kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES,
                kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES,
            ],
        })
    }
}

// The build script includes the registration files, and so does the
// projection assembly test; a test that reads the bundle's own modules
// lives here.
#[cfg(test)]
mod tests {
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::GEOMETRIES;

    /// The dense registration quantizes exactly the inputs the dense ops
    /// ask for.
    #[test]
    fn registered_dense_inputs_are_the_dense_ops_inputs() {
        for g in GEOMETRIES {
            for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
                let Ok(shape) = crate::shape::Shape::new(g, ranks) else {
                    continue;
                };
                assert_eq!(
                    super::dense::inputs(g, ranks.get()),
                    crate::ops::dense::quantized_inputs(&shape)
                );
            }
        }
    }

    /// Each rank count's head entry projects its shard of the vocabulary
    /// from the hidden width, the projection the head ops ask for.
    #[test]
    fn the_head_entry_is_the_ranks_vocabulary_shard() {
        use model_deepseek41_contract::geometry::FLASH;
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let entry = super::f32_head::entry(&FLASH, ranks.get()).unwrap();
            let q = entry.query();
            assert_eq!(
                (q.n * ranks.get(), q.k),
                (FLASH.shape.vocab, FLASH.shape.hidden)
            );
            let shape = crate::shape::Shape::new(&FLASH, ranks).unwrap();
            assert_eq!(q, crate::ops::head::query(&shape));
            assert_eq!(entry.usage(), crate::ops::head::USE);
        }
    }

    /// Every dense F32 projection the ops ask for has an entry of its query
    /// at the use they ask it at, and every entry serves one.
    #[test]
    fn each_f32_projection_has_its_entry() {
        use crate::ops::exact_f32;
        for g in GEOMETRIES {
            let entries = super::f32_dense::entries(g).unwrap();
            let projections = exact_f32::projections(g);
            for dense in &projections {
                let q = exact_f32::query(dense.n, dense.k);
                let entry = entries.iter().find(|e| e.query() == q);
                let entry = entry.unwrap_or_else(|| panic!("{} has no entry", dense.id));
                assert_eq!(entry.usage(), exact_f32::USE);
            }
            for entry in &entries {
                let q = entry.query();
                assert!(projections.iter().any(|d| (d.n, d.k) == (q.n, q.k)));
            }
        }
    }

    /// Every block quantizer the instance registers serves a query no other
    /// entry serves, at every geometry and rank count: SM120 refuses a
    /// query two entries serve.
    #[test]
    fn each_quantizer_query_is_registered_once() {
        use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
        for g in GEOMETRIES {
            for &ranks in SUPPORTED_RANKS {
                if check_ranks(g, ranks).is_err() {
                    continue;
                }
                let entries: Vec<_> = super::moe::quantizers(g, ranks)
                    .unwrap()
                    .into_iter()
                    .chain(super::dense::quantizers(g, ranks).unwrap())
                    .chain(super::latent_glue::quantizers())
                    .chain([super::indexer_query::quantizer(g)])
                    .collect();
                for (at, entry) in entries.iter().enumerate() {
                    let twin = entries[at + 1..].iter().find(|e| e.query == entry.query);
                    assert!(
                        twin.is_none(),
                        "{} tp{ranks}: {} and {} serve one query",
                        g.id,
                        entry.symbol,
                        twin.unwrap().symbol
                    );
                }
            }
        }
    }

    /// Every RMSNorm requirement the instance registers has one entry: the
    /// drafter's F32 head norm is the main head's final norm, registered
    /// once by the rows (`rows::norms`), and the DSpark rows register none.
    #[test]
    fn each_rmsnorm_requirement_is_registered_once() {
        use kernels_sm120::backend::Sm120;
        use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
        for g in GEOMETRIES {
            let mut norms = super::rows::norms(g);
            if SUPPORTED_RANKS
                .iter()
                .any(|&ranks| super::latent_glue::register(Sm120::new(), g, ranks).is_ok())
            {
                norms.push(super::latent_glue::query_norm(g));
            }
            for (at, (requirement, symbol)) in norms.iter().enumerate() {
                let twin = norms[at + 1..].iter().find(|(r, _)| r == requirement);
                assert!(
                    twin.is_none(),
                    "{}: {symbol} and {} serve one requirement",
                    g.id,
                    twin.unwrap().1
                );
            }
            for &ranks in SUPPORTED_RANKS {
                if check_ranks(g, ranks).is_err() {
                    continue;
                }
                let drafter = super::dspark_aux::register(Sm120::new(), g, ranks).unwrap();
                let refused = kernel_core::backend::Backend::rmsnorm(
                    &drafter,
                    &model_deepseek41_contract::rows::final_norm(g),
                    kernel_core::backend::Need::rows(1),
                );
                assert!(
                    refused.is_err(),
                    "{} tp{ranks}: DSpark registers a norm",
                    g.id
                );
            }
        }
    }

    #[test]
    fn each_hc_query_is_registered_once_including_dspark() {
        use kernel_core::sinkhorn_hyper_connection::{
            SinkhornHyperConnectionQuery as Query, StreamHeadMode,
        };
        for g in GEOMETRIES {
            let entries: Vec<_> = super::hc::registered_entries(g)
                .into_iter()
                .map(|entry| (entry.query, entry.symbol))
                .chain(
                    super::dspark_aux::feature_entries(g)
                        .into_iter()
                        .map(|entry| (entry.query, entry.symbol)),
                )
                .collect();
            for (at, (query, symbol)) in entries.iter().enumerate() {
                assert!(
                    !entries[at + 1..].iter().any(|(q, _)| q == query),
                    "{} repeats {symbol}'s query",
                    g.id
                );
            }
            assert_eq!(
                entries
                    .iter()
                    .filter(|(q, _)| matches!(
                        q,
                        Query::Head {
                            mode: StreamHeadMode::SigmoidWeightedF32,
                            ..
                        }
                    ))
                    .count(),
                1
            );
        }
    }
}

#[cfg(test)]
mod window_bf16_tests {
    use super::latent_window_bf16::*;
    use kernel_core::backend::{Backend, Need};
    use kernels_sm120::backend::Sm120;
    use model_deepseek41_contract::{
        geometry::GEOMETRIES, latent_glue, serving::PREFILL_TOKENS, state::KvStorage,
    };
    #[test]
    fn bf16_windows_prepare_through_the_shared_backend() {
        for g in GEOMETRIES {
            let b = super::latent_glue::register(Sm120::new(), g, 4).unwrap();
            let b = register(b);
            let write =
                latent_glue::window_write_for_storage(g, latent_glue::ring(g), KvStorage::Bf16);
            let op = b.latent_window_write(&write, Need::rows(32)).unwrap();
            assert!(op.info().symbols.iter().any(|(s, _)| s == &store().symbol));
            assert_eq!(
                op.info()
                    .bytes
                    .iter()
                    .find(|(s, _)| s == "window.scales")
                    .unwrap()
                    .1,
                0
            );
            let stage =
                latent_glue::window_stage_for_storage(g, PREFILL_TOKENS + 127, KvStorage::Bf16)
                    .unwrap();
            assert!(b.latent_window_stage(&stage, Need::rows(32)).is_ok());
        }
    }
}
