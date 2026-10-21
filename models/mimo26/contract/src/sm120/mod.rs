//! MiMo V2.6 Flash's kernel catalog entries on the shared engines of
//! `kernels-sm120`: the dense GEMMs ([`gemm`]), the grouped experts and
//! their route cubins ([`moe`]) and the paged attention ([`attention`]),
//! with its grouped-template schedule table (`tuning/schedules.json`), and
//! lab-only variants ([`lab`]). [`catalogs`] is the [`Catalogs`] value its
//! bundle reads.

pub mod attention;
mod backend;
pub mod comm;
pub mod config;
pub mod encoders;
pub mod entries;
pub mod gemm;
pub mod geometry;
pub mod lab;
pub mod moe;
pub mod prepared;
pub mod row;
pub mod tuning;
pub mod workspace;

use kernels_sm120::invariance::{Declared, OpKind};
use kernels_sm120::{Catalogs, ModelCatalog, ProjectionInventory};

use std::sync::OnceLock;

pub use geometry::PROFILES;

/// MiMo's SM120 kernels as the contract's backend
/// (`crate::backend::Selected`).
pub struct Sm120;

/// MiMo V2.6 Flash's catalog entries.
pub static CATALOG: ModelCatalog = ModelCatalog {
    profiles: &PROFILES,
    name: kernels_sm120::const_name!("mimo-", crate::geometry::V26_FLASH.id),
    gemm: gemm::catalog,
    gemm_fused: Vec::new,
    gemm_mid: gemm::mid_references,
    gemm_lab_splits: &gemm::LAB_SPLITS,
    moe: moe::catalog,
    moe_routes: Vec::new,
    moe_route_cubins: moe::route_cubins,
    attention: attention::catalog,
    rows: row::catalog,
    comm_shapes: comm::shapes,
    comm_cubins: comm::cubins,
    declared,
    schedule_table: Some(|| moe::schedules().clone()),
    geometry_schedules: Some(|| {
        tuning::measured_schedules()
            .expect("measured schedules")
            .into_iter()
            .collect()
    }),
    schedule_keys: || kernels_sm120::moe::template::schedule_keys(&moe::catalog()),
    projections,
    ..ModelCatalog::EMPTY
};

/// The generated projection offers and retained capabilities of every
/// geometry, in geometry order.
fn projections() -> Result<ProjectionInventory, String> {
    let mut inventory = ProjectionInventory::default();
    for geometry in crate::geometry::GEOMETRIES {
        let generated = crate::capabilities::sm120::inventory(geometry)?;
        inventory.0.extend(generated.offers);
        inventory.1.extend(generated.catalog);
    }
    Ok(inventory)
}

/// The verify steps whose rows are exact (docs/numerics.md#state-and-verification-contract):
/// up to `serving::EXACT_VERIFY_ROWS` rows of one request served at a time,
/// since AR itself depends on its batch above C1.
pub const EXACT_VERIFY: kernels_sm120::invariance::ExactVerify =
    kernels_sm120::invariance::ExactVerify {
        rows: crate::serving::EXACT_VERIFY_ROWS,
        items: 1,
    };

/// The kernels profile of `tp`'s per-rank shapes.
pub fn profile(tp: crate::Tp) -> kernels_sm120::Profile {
    let name = format!(
        "mimo-{}{}",
        tp.geometry.id,
        if tp.ranks() == 4 { "" } else { "-tp2" }
    );
    PROFILES
        .iter()
        .copied()
        .find(|p| p.as_str() == name && p.ranks() == Some(tp.ranks()))
        .expect("listed MiMo geometry has a kernel profile")
}

/// The fused router entries' declarations.
fn declared() -> Vec<Declared> {
    moe::fused_router_declarations()
        .into_iter()
        .map(|(symbol, invariance)| Declared {
            symbol,
            profile: None,
            op: OpKind::GroupedMoe,
            invariance,
        })
        .collect()
}

/// The bundle-owned kernel `name` (`ring_backup`, `dflash_drafts`, ...) of
/// the topology's cubins: `chop_mimo_ring_backup_v1` at TP4.
pub fn bundle_symbol(tp: crate::Tp, name: &str) -> String {
    format!("chop_{}_{name}_v1", crate::topology::kernel_tag(tp))
}

/// The kernel base's catalogs and MiMo V2.6 Flash's.
pub fn catalogs() -> &'static Catalogs {
    static CATALOGS: OnceLock<Catalogs> = OnceLock::new();
    CATALOGS.get_or_init(|| Catalogs::new(&[&CATALOG]).expect("MiMo catalogs are consistent"))
}
