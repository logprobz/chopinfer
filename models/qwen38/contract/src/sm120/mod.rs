//! Qwen3.8 Flash's kernel catalog entries on the shared engines of
//! `kernels-sm120`: the dense GEMMs ([`gemm`]), the grouped experts and
//! their routes ([`moe`]), the attention ([`attention`]) and the rows
//! ([`row`]), with its grouped-template schedule table
//! (`tuning/flash/schedules.json`). [`catalogs`] is the [`Catalogs`] value its
//! bundle reads.

mod admission;
pub mod attention;
mod backend;
pub mod comm;
pub mod config;
pub mod context_parallel;
pub mod dense;
pub mod gemm;
pub mod generated;
pub mod geometry;
pub mod linear;
pub mod moe;
pub mod owned;
#[cfg(test)]
mod precision_tests;
mod registered;
pub mod row;
pub mod tuning;

use kernels_sm120::{Catalogs, ModelCatalog, ProjectionInventory};

use std::sync::OnceLock;

pub use geometry::PROFILES;

/// Qwen's SM120 kernels as the contract's backend
/// (`crate::backend::Selected`).
pub struct Sm120;

/// Qwen3.8 Flash's catalog entries.
pub static CATALOG: ModelCatalog = ModelCatalog {
    profiles: &PROFILES,
    name: "qwen38-flash",
    gemm: gemm::catalog,
    gemm_fused: gemm::fused_catalog,
    gemm_fused_sources: gemm::sources_for,
    gemm_mid: gemm::mid_references,
    gemm_lab_splits_for: gemm::lab_splits,
    moe: moe::catalog,
    moe_routes: moe::routes,
    moe_route_cubins: moe::route_cubins,
    attention: attention::catalog,
    gated: attention::gated_entries,
    sparse: attention::sparse_catalog,
    rows: row::rows,
    linear: linear::catalog,
    linear_cubin_name: linear::cubin_name,
    hyper_connection_rows: row::hyper_connection_rows,
    hyper_connection_sources: row::sources_for,
    comm_cubins: comm::cubins,
    declared: || {
        let mut entries = geometry::MODELS
            .iter()
            .map(|owner| {
                let mut declaration = gemm::hc_up_mix_declaration();
                declaration.symbol = owner.hc.up_mix_symbol.into();
                declaration
            })
            .collect::<Vec<_>>();
        entries.extend(comm::declared());
        entries.extend(context_parallel::declarations());
        entries
    },
    schedule_table: Some(|| moe::schedules().clone()),
    geometry_schedules: Some(|| {
        crate::geometry::GEOMETRIES
            .iter()
            .filter_map(|g| {
                g.tuning.map(|data| {
                    (
                        g.id.to_owned(),
                        kernels_sm120::moe::template::parse_table(data.schedules)
                            .expect("geometry schedules"),
                    )
                })
            })
            .collect()
    }),
    schedule_keys: moe::schedule_keys,
    projections,
    row_shape_probes,
    norm_weights: comm::norm_weight_forms,
    ngram_hashes: || generated::ngram_hash_forms().expect("Qwen n-gram hash forms"),
    context_parallel: context_parallel::groups,
    ..ModelCatalog::EMPTY
};

/// The generated projection offers and layout copies of every geometry and
/// served topology, in geometry order.
fn projections() -> Result<ProjectionInventory, String> {
    let mut inventory = ProjectionInventory::default();
    for geometry in crate::geometry::GEOMETRIES {
        for topology in crate::contract::TOPOLOGIES {
            let shape = crate::contract::Shape::new(geometry, topology.ranks);
            let generated = generated::inventory(shape)?;
            inventory.0.extend(generated.offers);
            inventory.2.extend(generated.copies);
        }
    }
    Ok(inventory)
}

/// The layout copies and n-gram lookups no served inventory holds: every
/// offered port and use past the declaration's refusals. Both topologies
/// admit the mixed exports, so this is empty until a declaration refuses an
/// offered format.
fn row_shape_probes() -> Result<ProjectionInventory, String> {
    let served = projections()?;
    let key = |copy: &kernels_sm120::row::precision::LayoutCopy| {
        format!(
            "{} {} {:?}",
            copy.stage().entry().symbol,
            copy.stage().capacity(),
            copy.layout()
        )
    };
    let lookup = |offer: &kernels_sm120::precision::KernelOffer| {
        format!(
            "{:?} {:?}",
            offer.lookup_request(),
            offer.handle().row_stages()[0].entry().symbol
        )
    };
    let served_copies: std::collections::BTreeSet<_> = served.2.iter().map(key).collect();
    let served_lookups: std::collections::BTreeSet<_> = served
        .0
        .iter()
        .filter(|offer| offer.lookup_request().is_some())
        .map(lookup)
        .collect();
    let mut probes = ProjectionInventory::default();
    for geometry in crate::geometry::GEOMETRIES {
        for topology in crate::contract::TOPOLOGIES {
            let shape = crate::contract::Shape::new(geometry, topology.ranks);
            let (offers, copies) = generated::row_uses(shape)?;
            probes.0.extend(
                offers
                    .into_iter()
                    .filter(|o| !served_lookups.contains(&lookup(o))),
            );
            probes.2.extend(
                copies
                    .into_iter()
                    .filter(|c| !served_copies.contains(&key(c))),
            );
        }
    }
    Ok(probes)
}

/// The kernel base's catalogs and Qwen3.8 Flash's.
pub fn catalogs() -> &'static Catalogs {
    static CATALOGS: OnceLock<Catalogs> = OnceLock::new();
    CATALOGS.get_or_init(|| Catalogs::new(&[&CATALOG]).expect("Qwen3.8 catalogs are consistent"))
}
