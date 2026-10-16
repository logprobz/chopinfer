//! The bundle's own kernels and their declared invariance
//! (`kernel_core::properties`), which kernel-lab's bundle adapter tests.

use crate::backend::{Kernels, OwnedKernel, Selected};
use kernel_core::properties::{Anchors, Chunk, Invariance};

/// The declarations of the owned kernels prefill and exact verify steps
/// launch:
/// - the PLE residual is row-local;
/// - the PLE stage and checkpoints copy state entries without arithmetic;
/// - the PLE anchor leaves, in a spare entry, the state a prefill truncated
///   at the anchor leaves, for prompt chunks at multiples of the granule;
/// - the verify n-gram and PLE convolution give each verify row the bits of
///   the decode entries run one row at a time (`contract::EXACT_VERIFY`);
///   TP2 launches its own n-gram verify rows (at its own symbol tag), whose
///   hash heads differ, with the same declaration;
/// - the registered n-gram hash entries ([`hash_declared`]) of the
///   topologies that publish the packed n-gram lookup.
pub fn declared() -> Vec<<Selected as Kernels>::Declared> {
    crate::geometry::GEOMETRIES
        .iter()
        .flat_map(|geometry| declared_for(*geometry))
        .collect()
}

pub fn declared_for(geometry: crate::geometry::Geometry) -> Vec<<Selected as Kernels>::Declared> {
    // The TP4 tag, then the other served rank counts' (`_tp{N}`).
    let tag = |ranks: u32| {
        if ranks == crate::contract::TOPOLOGY.ranks.get() {
            geometry.kernel_tag.to_owned()
        } else {
            format!("{}_tp{ranks}", geometry.kernel_tag)
        }
    };
    let tp4 = tag(crate::contract::TOPOLOGY.ranks.get());
    let rows = Invariance::any_rows(crate::contract::PREFILL_TOKENS);
    let verify = Invariance::one_row(crate::contract::MAX_VERIFY_ROWS);
    let entry = Selected::owned_entry;
    let ngram_verify = std::iter::once(entry(OwnedKernel::NgramVerify, &tp4)).chain(
        crate::contract::SERVED_RANKS
            .into_iter()
            .filter(|&ranks| ranks != crate::contract::TOPOLOGY.ranks.get())
            .map(|ranks| entry(OwnedKernel::NgramVerify, &tag(ranks))),
    );
    [
        (OwnedKernel::PleResidual, rows),
        (OwnedKernel::PleStage, Invariance::COPY),
        (OwnedKernel::PleCheckpoints, Invariance::COPY),
        (OwnedKernel::PleCheckpointsIndexed, Invariance::COPY),
        (
            OwnedKernel::PleAnchor,
            Invariance {
                chunk: Chunk::Granule(crate::contract::PREFILL_GRANULE),
                anchors: Anchors::Spare,
                ..Invariance::NONE
            },
        ),
        (OwnedKernel::PleConvVerify, verify),
    ]
    .into_iter()
    .map(|(kernel, invariance)| (entry(kernel, &tp4), invariance))
    .chain(ngram_verify.map(|symbol| (symbol, verify)))
    .chain(hash_declared(geometry))
    .map(|(symbol, invariance)| Selected::bundle_declaration(symbol, invariance))
    .collect()
}

/// The registered n-gram hash entries of `geometry`'s topologies
/// (the backend's registered hash kernels) and what they keep. Each
/// row's ids are an integer hash of its own token and the two before it,
/// read from the item's earlier rows or, before its first row, from the
/// two-token history the entry was given; the history after the last row
/// is its last two tokens (the given history's newer one and the row's
/// with one row). Nothing is accumulated or reordered, so:
/// - a prefill item split into chunks anywhere, each chunk given the
///   history the previous one left, gives every row the same ids
///   (`Chunk::Any`); the history it writes is a slot's state at the end of
///   the item's rows, and the PLE anchor launched after it derives the
///   history at an anchor from the same tokens (`Anchors::Companion`);
/// - a verify item's rows get the ids one-row decode steps give them, the
///   history carried, up to the verify rows, whatever the other items
///   (`one_row(MAX_VERIFY_ROWS)`, the verify n-gram's declaration).
///
/// The decode entries serve no exact step, so they declare nothing.
pub fn hash_declared(geometry: crate::geometry::Geometry) -> Vec<(String, Invariance)> {
    use kernel_core::precision::ExecutionPhase;
    let forms = Selected::ngram_hash_symbols().expect("Qwen n-gram hash forms");
    let geometry = crate::geometry::GEOMETRIES
        .iter()
        .find(|served| **served == geometry);
    crate::contract::TOPOLOGIES
        .iter()
        .filter_map(|topology| Some(crate::contract::Shape::new(geometry?, topology.ranks)))
        .filter_map(|shape| {
            let prefill = Selected::ngram_hash_symbol(shape, ExecutionPhase::Prefill);
            forms.iter().find(|(form, _)| *form == prefill)
        })
        .flat_map(|(prefill, verify)| {
            [
                (
                    prefill.clone(),
                    Invariance {
                        chunk: Chunk::Any,
                        anchors: Anchors::Companion,
                        ..Invariance::NONE
                    },
                ),
                (
                    verify.clone(),
                    Invariance::one_row(crate::contract::MAX_VERIFY_ROWS),
                ),
            ]
        })
        .collect()
}
