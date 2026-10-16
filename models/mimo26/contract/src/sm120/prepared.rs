//! The SM120 entries MiMo's precision provider (`crate::prepared`) binds a
//! declared projection to: a dense catalog entry, with the medium-M entry
//! that stands in for it at a `.mid` use, a grouped expert variant with its
//! capability, or an embedding lookup row entry.

use kernel_core::contract::InstantiationSpec;
use kernel_core::precision::OperationUse;
use kernel_core::precision::capability::OperationCapability;
use sha2::{Digest, Sha256};

use crate::Tp;
use crate::backend::Alternative;
use crate::capabilities::Projection;

/// The SM120 entry a projection use binds.
#[derive(Clone)]
pub enum Entry {
    Dense {
        entry: InstantiationSpec,
        medium: Option<kernels_sm120::gemm::mid::Entry>,
        head: bool,
    },
    Expert(kernels_sm120::moe::Variant),
    Lookup(kernels_sm120::row::RowInstantiation),
}

/// The symbols of every medium-M kernel, which a component's dependency
/// retains whether or not the process selects them.
pub fn medium_kernels() -> impl Iterator<Item = String> {
    kernels_sm120::gemm::mid::Kernel::ALL
        .into_iter()
        .map(|kernel| kernel.symbol())
}

/// The medium-M entry that stands in for fragment prefill entry `reference`
/// where the catalog has one.
pub fn medium(reference: &str) -> Option<kernels_sm120::gemm::mid::Entry> {
    kernels_sm120::gemm::mid::select(crate::sm120::catalogs(), reference, 64)
}

/// The expert entry of gate/up (`gate_up`) or down at `usage`, exact or
/// inexact, with its catalog capability.
pub fn expert(
    tp: Tp,
    gate_up: bool,
    usage: OperationUse,
    exact: bool,
) -> Result<(Entry, OperationCapability), String> {
    let variant = crate::sm120::workspace::moe::entry(
        tp,
        if gate_up {
            kernels_sm120::moe::Projection::GateUp
        } else {
            kernels_sm120::moe::Projection::Down
        },
        usage.phase,
        usage.capacity,
        crate::sm120::workspace::moe::Selection { exact },
    )
    .map_err(|e| e.to_string())?;
    let capability = kernels_sm120::precision::catalog_grouped_capability(variant, usage)
        .map_err(|e| e.to_string())?;
    Ok((Entry::Expert(variant.clone()), capability))
}

/// The entry serving `projection` at `usage` as `alternative`
/// ([`crate::backend::Kernels::bind`]): the grouped expert entry with its
/// capability, the lookup or dense entry the capability's stages name, and
/// for a medium-M use the dense entry's stand-in as the stage.
pub fn bind(
    tp: Tp,
    projection: Projection,
    usage: OperationUse,
    alternative: Alternative,
    capability: &mut OperationCapability,
) -> Result<Entry, String> {
    if projection.routed() {
        let (expert, grouped) = expert(
            tp,
            projection == Projection::ExpertGateUp,
            usage,
            alternative != Alternative::Inexact,
        )?;
        *capability = grouped;
        return Ok(expert);
    }
    if projection.lookup() {
        let entry = crate::sm120::workspace::row::entries_for(tp)
            .find(|e| capability.stages.contains(&e.symbol))
            .ok_or("lookup entry missing")?;
        return Ok(Entry::Lookup(entry.clone()));
    }
    let entry = crate::sm120::workspace::dense::entries_for(tp)
        .find(|e| capability.stages.contains(&e.symbol))
        .ok_or("dense entry missing")?
        .clone();
    let medium = if alternative == Alternative::Medium {
        let medium = self::medium(&entry.symbol).ok_or("medium entry missing")?;
        capability.stages = vec![medium.kernel.symbol()];
        Some(medium)
    } else {
        None
    };
    Ok(Entry::Dense {
        entry,
        medium,
        head: matches!(projection, Projection::Head { .. }),
    })
}

/// Whether a dense entry has a medium-M stand-in.
pub fn has_medium(entry: &Entry) -> Option<bool> {
    match entry {
        Entry::Dense { entry, .. } => Some(medium(&entry.symbol).is_some()),
        _ => None,
    }
}

/// The identity of every stage a `tp` component can bind: the dense, row
/// and expert entries `tp` selects from, every medium-M kernel and the
/// bundle's kernel lock.
pub fn dependency(tp: Tp) -> String {
    static IDENTITIES: std::sync::OnceLock<Vec<(Tp, String)>> = std::sync::OnceLock::new();
    IDENTITIES
        .get_or_init(|| {
            crate::geometry::GEOMETRIES
                .iter()
                .flat_map(|geometry| {
                    Tp::SUPPORTED.map(|ranks| {
                        let tp = Tp::of_geometry(geometry, ranks.ranks()).expect("listed geometry");
                        let mut stages = std::collections::BTreeSet::new();
                        stages.extend(
                            crate::sm120::workspace::dense::entries_for(tp)
                                .map(|e| e.symbol.clone()),
                        );
                        stages.extend(
                            crate::sm120::workspace::row::entries_for(tp).map(|e| e.symbol.clone()),
                        );
                        stages.extend(
                            crate::sm120::workspace::moe::mimo_variants(tp)
                                .iter()
                                .map(|e| e.symbol.clone()),
                        );
                        stages.extend(medium_kernels());
                        let mut digest = Sha256::new();
                        digest.update(include_bytes!("../../../bundle/kernels.lock"));
                        for stage in stages {
                            digest.update((stage.len() as u64).to_le_bytes());
                            digest.update(stage.as_bytes());
                        }
                        (tp, format!("{:x}", digest.finalize()))
                    })
                })
                .collect()
        })
        .iter()
        .find(|(shape, _)| *shape == tp)
        .expect("listed topology")
        .1
        .clone()
}
