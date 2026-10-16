//! The vision encoder's SM120 side (`crate::target::Target`): its cubins'
//! loads, configured for the dynamic shared memory each catalog entry
//! plans, and the encoder runtime with every slot prepared on the SM120
//! backend over the encoder's catalog (`model_glm5next_contract::sm120::encoder`).

use std::collections::BTreeMap;

use bundle_runtime::acquire::{self, Log, RankHandles};
use bundle_runtime::encoder::{Encoder, SlotOp};
use bundle_runtime::resources::Held;
use kernels_sm120::backend::Sm120;
use kernels_sm120::encoder::{Shared, Slot};
use model_glm5next_contract::encoder::declarations;
use model_glm5next_contract::sm120::encoder::{is_cubin, kernels};
use runtime_bundle_abi::{
    BundleAbiError, BundleCudaSymbolHandleV1, DeviceBufferSpanV1 as Span, HostServicesV1,
};

use crate::encoders::CONFIG;

/// Loads the encoder's cubins on every rank: per rank, each entry's
/// symbol, configured for the dynamic shared memory its catalog entry
/// plans.
pub(crate) fn load(
    log: Log,
    ranks: &[RankHandles],
    held: &mut Vec<Held>,
) -> Result<Vec<BTreeMap<String, BundleCudaSymbolHandleV1>>, BundleAbiError> {
    let invalid = |_| BundleAbiError::InvalidRange("the glm5next vision declaration is invalid");
    let catalog = kernels().map_err(invalid)?;
    let mut symbols = Vec::with_capacity(ranks.len());
    for &(context, _, _) in ranks {
        let mut resolved = BTreeMap::new();
        for cubin in crate::CUBINS.iter().filter(|cubin| is_cubin(cubin.name)) {
            acquire::load_module(
                log,
                context,
                (cubin.name, cubin.bytes),
                cubin.symbols.iter().copied(),
                |symbol| {
                    catalog
                        .iter()
                        .find(|entry| entry.symbol == symbol)
                        .map(|entry| entry.dynamic_shared_bytes().unwrap_or(0))
                        .ok_or(BundleAbiError::InvalidRange(
                            "unplanned glm5next encoder symbol",
                        ))
                },
                &mut resolved,
                held,
            )?;
        }
        symbols.push(resolved);
    }
    Ok(symbols)
}

/// The encoder runtime over each rank's encoder image `weights`, every slot
/// prepared on the SM120 backend over the encoder's catalog from the query
/// its entry was generated from, every rank's regions allocated with
/// `allocate`.
pub(crate) fn create(
    services: &HostServicesV1,
    weights: Vec<BTreeMap<String, Span>>,
    allocate: &mut dyn FnMut(usize, u64) -> Result<Span, BundleAbiError>,
) -> Result<Encoder, BundleAbiError> {
    let invalid = |_| BundleAbiError::InvalidRange("the glm5next vision declaration is invalid");
    let [vision] = declarations().map_err(invalid)?;
    let slots = kernels_sm120::encoder::slots(vision, &Shared).map_err(invalid)?;
    let catalog = kernels().map_err(invalid)?;
    let backend = Sm120::new().with_encoders(catalog.iter().cloned());
    // Each slot's query, which the backend serves from the catalog entry of
    // its code.
    let resolve = |_: usize, stage: usize, part: &'static str, block: Option<u32>| {
        let slot = Slot { stage, part, block };
        let (_, entry) = slots
            .iter()
            .find(|(known, _)| *known == slot)
            .ok_or_else(|| format!("the vision catalog has no slot {slot:?}"))?;
        Ok(SlotOp::Query(entry.query))
    };
    Encoder::with_backend(
        services,
        &[Some(vision)],
        CONFIG,
        &backend,
        &resolve,
        weights,
        allocate,
    )
}

/// The full symbol of the catalog entry whose symbol reads `stem`.
#[cfg(test)]
pub(crate) fn full(stem: &str) -> &'static str {
    kernels_sm120::encoder::by_stem(kernels().unwrap(), stem)
        .unwrap()
        .symbol
        .as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both encoder cubins embed, holding every catalog entry.
    #[test]
    fn the_encoder_cubins_embed_every_entry() {
        let embedded: Vec<&str> = crate::CUBINS
            .iter()
            .filter(|cubin| is_cubin(cubin.name))
            .flat_map(|cubin| cubin.symbols.iter().copied())
            .collect();
        let mut catalog: Vec<&str> = kernels()
            .unwrap()
            .iter()
            .map(|entry| entry.symbol.as_str())
            .collect();
        let mut sorted = embedded.clone();
        sorted.sort_unstable();
        catalog.sort_unstable();
        assert_eq!(sorted, catalog);
    }
}
