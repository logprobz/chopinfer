//! GPU-gated correctness, probe, and timing tests of the in-tree models'
//! and the lab's own SM120 cubins, on the model-free harness
//! (`kernel_harness`, re-exported here). The lab builds the cubins of every
//! catalog it selects ([`MODELS`]) and of its lock families; a new model runs
//! its kernel tests from its own test crate on the harness instead.

#[path = "../shapes.rs"]
pub mod lab_shapes;
pub use kernel_lab_families as families;
pub use kernel_lab_families::latent_attention::shapes as latent_shapes;

pub use kernel_harness::*;

pub mod declared;
pub mod device_check;
mod lab_modules;
#[path = "../models.rs"]
pub mod models;
pub mod projection_cases;
pub mod row_shapes;

/// The conformance kit (`kernel_harness::kit`) with the lab's modules.
pub mod kit {
    pub use crate::lab_modules::LabModules;
    pub use kernel_harness::kit::*;
}

/// The catalogs the lab builds and tests: the kernel base's, then every
/// model's ([`MODELS`]).
pub fn catalogs() -> &'static kernels_sm120::Catalogs {
    static CATALOGS: std::sync::OnceLock<kernels_sm120::Catalogs> = std::sync::OnceLock::new();
    CATALOGS.get_or_init(|| {
        kernels_sm120::Catalogs::new(MODELS).expect("the lab's model catalogs are consistent")
    })
}

/// Every model's catalog, in the lab's order (MiMo V2.6 Flash, then
/// Qwen3.8 Flash): `models.rs`, which the build script includes too.
pub static MODELS: &[&kernels_sm120::ModelCatalog] = models::CATALOGS;

include!(concat!(env!("OUT_DIR"), "/cubins.rs"));

/// The cubins this crate built, by spec name, with their entry symbols.
pub const LAB_CUBINS: Cubins = Cubins::new(CUBINS, CUBIN_ENTRIES);

/// The bytes of any cubin this crate built, by spec name.
pub fn cubin(name: &str) -> LabResult<&'static [u8]> {
    LAB_CUBINS.bytes(name)
}

/// The entry symbols of the cubin `name`.
pub fn cubin_entries(name: &str) -> LabResult<&'static [&'static str]> {
    LAB_CUBINS.entries(name)
}

/// Whether every entry of the cubin `name` is a comm entry, whose grids
/// spin on their peer ranks' grids.
pub fn comm_cubin(name: &str) -> LabResult<bool> {
    LAB_CUBINS.comm(name)
}

/// [`kernel_harness::gpu_test`] on a session that loads the lab's cubins.
pub fn gpu_test<T>(
    need_bytes: u64,
    run: impl FnOnce(&Session) -> LabResult<T>,
) -> LabResult<Option<T>> {
    gpu_test_with(LAB_CUBINS, need_bytes, run)
}

/// [`kernel_harness::gpu_test_multi`] on sessions that load the lab's
/// cubins.
pub fn gpu_test_multi<T>(
    need_bytes_per_rank: u64,
    run: impl FnOnce(&[Session]) -> LabResult<T>,
) -> LabResult<Option<T>> {
    gpu_test_multi_with(LAB_CUBINS, need_bytes_per_rank, run)
}

/// The function of the typed projection entry `symbol` in the prepared
/// projection report (`CHOPINFER_PROJECTION_REPORT`), with `shared` bytes
/// of dynamic shared memory; `None` without a report or when it does not
/// hold `symbol`. The session verifies the report once and keeps its
/// modules.
pub fn projection_function(
    session: &Session,
    symbol: &str,
    shared: u32,
) -> LabResult<Option<runtime_cuda::Function>> {
    if std::env::var_os("CHOPINFER_PROJECTION_REPORT").is_none() {
        return Ok(None);
    }
    let prepared = session.state(projection_cases::Prepared::read)?;
    if prepared.contains(symbol) {
        Ok(Some(prepared.function(session, symbol, shared)?))
    } else {
        Ok(None)
    }
}

/// The resources ptxas reported for the lab's chained decoded-tile cubins
/// and the split instances' cubins, which SM120 admits those forms on
/// (`kit::cases::moe::sm120`).
pub fn lab_resources() -> Vec<(String, kernel_core::occupancy::EntryResources)> {
    [
        include_str!("../kernels.lock.d/moe_chained_decoded.lock"),
        include_str!("../kernels.lock.d/moe_chained_split.lock"),
    ]
    .into_iter()
    .flat_map(|lock| {
        kernels_sm120::backend::locked_resources(lock)
            .expect("the lab's chained locks list resources")
    })
    .collect()
}

/// The Qwen3.8 topology the per-rank Qwen kernel tests run:
/// `CHOPINFER_QWEN38_RANKS` of 2 or 4, default 4 (TP4's entries and
/// receipts).
pub fn qwen38_ranks() -> LabResult<u32> {
    match std::env::var("CHOPINFER_QWEN38_RANKS") {
        Err(std::env::VarError::NotPresent) => parse_ranks(None),
        Ok(value) => parse_ranks(Some(&value)),
        Err(error) => Err(error.to_string()),
    }
}

fn parse_ranks(value: Option<&str>) -> LabResult<u32> {
    let Some(value) = value else { return Ok(4) };
    match value.trim().parse() {
        Ok(ranks @ (2 | 4)) => Ok(ranks),
        _ => Err(format!(
            "CHOPINFER_QWEN38_RANKS must be 2 or 4, not {value:?}"
        )),
    }
}

#[cfg(test)]
mod rank_tests {
    use super::parse_ranks;

    #[test]
    fn rank_selection_refuses_eight() {
        assert_eq!(parse_ranks(None).unwrap(), 4);
        assert_eq!(parse_ranks(Some("2")).unwrap(), 2);
        assert_eq!(parse_ranks(Some("4")).unwrap(), 4);
        assert!(parse_ranks(Some("8")).unwrap_err().contains("2 or 4"));
    }
}

/// Lean runs: the TP2 topology runs the smallest cases that still
/// launch every per-rank entry, with oracles on sampled outputs
/// (`sample_indices`), so each topology fits one short GPU window. TP4 (the
/// default) keeps its full cases.
pub fn qwen38_lean() -> LabResult<bool> {
    Ok(qwen38_ranks()? != 4)
}

/// The kernels-sm120 profile of `qwen38_ranks`.
pub fn qwen38_profile() -> LabResult<kernels_sm120::Profile> {
    let ranks = qwen38_ranks()?;
    model_qwen38_contract::sm120::config::profile(ranks)
        .ok_or_else(|| format!("no Qwen3.8 profile at {ranks}"))
}

/// The per-rank shapes of `qwen38_ranks`.
pub fn qwen38_shape() -> LabResult<model_qwen38_contract::sm120::config::RankShape> {
    Ok(model_qwen38_contract::sm120::config::RankShape {
        ranks: qwen38_ranks()?,
    })
}

#[cfg(test)]
mod family_tests {
    /// Every file of `kernels.lock.d` belongs to a family file of
    /// `families/`, and every family has its lock and provenance files.
    #[test]
    fn lock_directory_pairs_with_family_files() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels.lock.d");
        let problems =
            crate::families::unpaired(&dir, crate::families::NAMES).expect("read kernels.lock.d");
        assert!(
            problems.is_empty(),
            "kernels.lock.d: {}",
            problems.join("; ")
        );
    }
}

#[cfg(test)]
mod probe_tests {
    use crate::poison::SM120_COPY_PRODUCER;

    /// The latent lab cubin holds SM120's PDL copy producer under the name
    /// the harness launches.
    #[test]
    fn the_latent_lab_cubin_holds_the_copy_producer() {
        assert_eq!(crate::latent_shapes::pdl_copy_symbol(), SM120_COPY_PRODUCER);
        assert_eq!(
            crate::LAB_CUBINS.holding(SM120_COPY_PRODUCER),
            Some(crate::latent_shapes::cubin_name().as_str())
        );
    }
}
