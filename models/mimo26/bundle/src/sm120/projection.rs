//! MiMo's dense projections on the SM120 backend: each use binds the
//! catalog entry the precision component authenticates for it
//! (`prepared::Provider`), launched over the fragment maps MiMo's entries
//! are planned for, as the medium-M entry that stands in for it where the
//! component binds one, and with a vocabulary head's output holding only
//! the launched rows.
//!
//! A medium-M use wider than its entry's rows serves only the entry's rows
//! ([`served_rows`]). Expert and lookup projections keep MiMo's registered
//! launchers (`crate::projection`). The backend's grouped compute is built for the
//! packed weight layout and engine, where MiMo's served expert entries read
//! MXFP4 fragment weights on the template schedules. Its lookup launches a
//! different kernel ABI: eight arguments (table, ids, local and global
//! scales, output, live rows, row offset, shard rows) where MiMo's embedding
//! kernel takes six (table, ids, output, live count, row offset,
//! vocabulary), and it records the live rows as a launch's rows where MiMo
//! records the row grid it launches. MiMo's lookup launcher launches the
//! rows family's embedding gather (`Backend::embedding_gather`) instead,
//! prepared over the authenticated entry ([`super::lookup_backend`]).

use std::collections::BTreeMap;
use std::ops::RangeInclusive;

use kernel_core::backend::Backend;
use kernel_core::precision::execution::{Binding, BoundUse, ImplementationProvider};
use kernel_core::precision::{OperationSpec, OperationUse};
use kernels_sm120::backend::{DenseForm, NamedOffers, ProjectionOffer, Sm120};
use kernels_sm120::precision::KernelOffer;
use model_mimo26_contract::prepared;
use model_mimo26_contract::sm120::prepared::Entry;

use crate::selection::{MimoPhase as _, Phase};
use crate::target::Registered;
use crate::topology::Tp;

/// The offer of each dense use the component binds, under the identity it
/// authenticates.
struct Offers(prepared::Provider);

impl ImplementationProvider for Offers {
    type Handle = ProjectionOffer;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<ProjectionOffer>, String> {
        let bound = self.0.bind(operation, usage)?;
        let Entry::Dense {
            entry,
            medium,
            head,
        } = bound.handle
        else {
            return Err(format!(
                "operation {}: MiMo prepares only its dense projections on SM120",
                operation.id
            ));
        };
        let offer = match &medium {
            // A medium-M entry claims nothing of its own (`gemm::mid`).
            Some(_) => KernelOffer::catalog_dense(entry, usage),
            None => {
                let variant =
                    model_mimo26_contract::sm120::workspace::dense::variants_for(self.0.tp)
                        .find(|variant| variant.instantiation.symbol == entry.symbol)
                        .ok_or_else(|| {
                            format!("operation {}: dense entry missing", operation.id)
                        })?;
                KernelOffer::catalog_variant(variant, usage)
            }
        }
        .map_err(|e| e.to_string())?;
        Ok(Binding {
            identity: bound.identity,
            handle: ProjectionOffer {
                offer,
                copy: None,
                dense: medium.map_or(DenseForm::Fragment, |medium| {
                    DenseForm::Medium(Box::new(medium))
                }),
                live_output: head,
            },
        })
    }
}

/// The backend and op of MiMo's registered launcher of an expert use: the
/// grouped entry the component authenticated for it, and the contract that
/// entry computes.
pub fn expert_backend(
    entry: kernels_sm120::moe::Variant,
) -> (
    Box<dyn Backend>,
    kernel_core::gated_mlp::GatedMlpProjectionQuery,
) {
    let op = kernels_sm120::backend::grouped_contract(&entry);
    (Box::new(Sm120::new().with_experts(entry)), op)
}

/// The backend a `tp` instance prepares its dense projections through.
pub fn backend(tp: Tp) -> Box<dyn Backend> {
    Box::new(
        Sm120::new().with_projections(Box::new(NamedOffers(Offers(prepared::Provider { tp })))),
    )
}

/// The medium-M alternative (`{id}.mid`) that stands in for projection
/// `id`'s dense use `bound`, and the live rows it serves: at decode and
/// verify buckets that run the prefill entries, when the medium-M switch is
/// on, where the use's fragment prefill entry has a medium-M entry
/// (`gemm::mid::select`). Expert uses have none.
pub fn medium_alternative(
    id: &str,
    bound: &BoundUse,
    switches: crate::switches::Switches,
) -> Result<Option<(String, RangeInclusive<u32>)>, String> {
    let phase = crate::ops::phase_of(bound.usage).map_err(|e| e.0.to_owned())?;
    if !switches.dense_mid
        || id.starts_with("moe.experts.")
        || !phase.prefill_kernels()
        || !matches!(phase, Phase::Verify(_) | Phase::Decode(_))
    {
        return Ok(None);
    }
    static MEDIUM: std::sync::OnceLock<BTreeMap<String, RangeInclusive<u32>>> =
        std::sync::OnceLock::new();
    Ok(bound
        .implementation
        .stages
        .first()
        .and_then(|stage| MEDIUM.get_or_init(medium_rows).get(stage))
        .map(|rows| (format!("{id}.mid"), rows.clone())))
}

/// The live rows at which a medium-M entry stands in for each fragment
/// prefill entry that has one (`gemm::mid::select`), by the prefill
/// entry's symbol.
fn medium_rows() -> BTreeMap<String, RangeInclusive<u32>> {
    let mut rows = BTreeMap::new();
    for entry in kernels_sm120::gemm::mid::catalog(model_mimo26_contract::sm120::catalogs()) {
        let previous = rows.insert(entry.reference, entry.rows);
        assert!(previous.is_none(), "one medium-M entry per reference");
    }
    rows
}

/// The rows the backend serves the use of `operation` at `usage` that the
/// component binds, under the identity it authenticates: every dense use
/// ([`served_rows`]); none for expert and lookup uses, which keep MiMo's
/// registered launchers.
pub fn rows(
    tp: Tp,
    operation: &OperationSpec,
    usage: OperationUse,
) -> Result<Binding<Option<u32>>, String> {
    let bound = prepared::Provider { tp }.bind(operation, usage)?;
    Ok(Binding {
        identity: bound.identity,
        handle: matches!(bound.handle, Entry::Dense { .. })
            .then(|| served_rows(&bound.handle, usage)),
    })
}

/// What MiMo's registered launcher of the use of `operation` at `usage`
/// launches on: the grouped entry the component authenticated for an
/// expert use, at the workspace of its largest plan, or the embedding
/// entry of a lookup; none for a dense use, which the backend prepares.
pub fn registered(
    tp: Tp,
    operation: &OperationSpec,
    usage: OperationUse,
) -> Result<Option<Registered>, String> {
    let bound = prepared::Provider { tp }.bind(operation, usage)?;
    Ok(match bound.handle {
        Entry::Dense { .. } => None,
        Entry::Expert(entry) => {
            let workspace_bytes = entry
                .plan(entry.counts(entry.tokens_capacity, entry.query.m_capacity))
                .map_err(|refusal| format!("operation {}: {}", operation.id, refusal.0))?
                .workspace_bytes;
            let (backend, op) = expert_backend(entry);
            Some(Registered::Experts {
                backend,
                op,
                workspace_bytes,
            })
        }
        Entry::Lookup(entry) => Some(Registered::Lookup {
            backend: super::lookup_backend(entry),
        }),
    })
}

/// The most rows a use the component binds to `entry` at `usage` launches:
/// its capacity, or a medium-M entry's rows where they are fewer. MiMo's
/// components also bind the medium alternative at buckets above 64 rows,
/// which a step whose verify bucket is set wider than its live rows
/// launches at 33 to 64 rows (`projection::Choices`); the backend serves
/// such a use at the entry's rows and refuses wider launches.
fn served_rows(entry: &Entry, usage: OperationUse) -> u32 {
    match entry {
        Entry::Dense {
            medium: Some(medium),
            ..
        } => usage.capacity.min(medium.max_rows()),
        _ => usage.capacity,
    }
}
