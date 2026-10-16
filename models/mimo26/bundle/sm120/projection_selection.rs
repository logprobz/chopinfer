//! Preserve the locked mixture and order of projection and catalog entries.
use std::collections::{BTreeMap, BTreeSet};

use kernel_build::{EntrySpec, KernelSelection};
use kernels_sm120::precision::generated::{self, Adoption, placement};
use model_mimo26_contract as contract;

pub fn replacements(
    selection: &KernelSelection,
    geometry_of: &BTreeMap<String, &str>,
    ledger: &[Adoption],
) -> Result<(Vec<EntrySpec>, Vec<Adoption>), String> {
    let mut entries = Vec::new();
    let mut identities = Vec::new();
    let active = generated::active(ledger);
    let adopted: BTreeSet<_> = active
        .values()
        .map(|record| {
            let place = record.placement.as_ref().ok_or("missing cubin placement")?;
            geometry_of
                .get(&place.cubin)
                .copied()
                .ok_or_else(|| format!("missing geometry owner for adopted cubin {}", place.cubin))
        })
        .collect::<Result<_, String>>()?;
    // The ledger adopts entries, not every newly registered geometry. Keep
    // unadopted geometries' catalog schedules until their entries are adopted.
    for geometry in contract::geometry::GEOMETRIES
        .iter()
        .filter(|geometry| adopted.contains(geometry.id))
    {
        let inventory = contract::capabilities::sm120::inventory(geometry)?;
        for offer in &inventory.offers {
            entries.extend(offer.handle().entries().map_err(|e| e.to_string())?);
        }
        let retained = inventory
            .catalog
            .iter()
            .flat_map(|c| c.stages.iter())
            .collect::<std::collections::BTreeSet<_>>();
        entries.extend(
            selection
                .specs()
                .iter()
                .flat_map(|s| &s.entries)
                .filter(|e| retained.contains(&e.symbol))
                .cloned(),
        );
        identities.extend(inventory.adoptions);
    }
    Ok((entries, identities))
}

pub fn assemble(
    selection: &KernelSelection,
    geometry_of: &BTreeMap<String, &str>,
) -> Result<KernelSelection, String> {
    let ledger = contract::capabilities::sm120::ledger()?;
    let (entries, identities) = replacements(selection, geometry_of, &ledger)?;
    placement::assemble(selection, &entries, &identities, &ledger)
}
