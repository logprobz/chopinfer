//! Per-rank bytes of MiMo's images, from the checkpoint catalog: a package
//! without rank images declares the most any rank holds, which admission
//! checks against device memory. The bytes are those of the direct-load
//! plan's views, each on the view alignment.

use std::collections::BTreeSet;

use crate::MimoResult;
use crate::catalog::{Catalog, DrafterKind};
use model_mimo26_contract::Tp;
use model_mimo26_contract::weight_plan::{self, PlannedView, VIEW_ALIGNMENT};

fn aligned(views: &[PlannedView]) -> u64 {
    views
        .iter()
        .map(|view| view.bytes.div_ceil(VIEW_ALIGNMENT) * VIEW_ALIGNMENT)
        .sum()
}

pub fn estimated_drafter_rank_bytes(
    catalog: &Catalog,
    kind: DrafterKind,
    tp: Tp,
) -> MimoResult<Vec<u64>> {
    // Only the mask's extent counts; its bytes come from the checkpoint.
    let mask = [0; model_mimo26_contract::drafters::MASK_BYTES];
    (0..tp.count())
        .map(|rank| {
            Ok(aligned(&weight_plan::drafter_views(
                &catalog.tensors,
                &catalog.operands,
                kind.planned(),
                Some(&mask),
                rank,
                tp,
            )?))
        })
        .collect()
}

pub fn estimated_rank_bytes(
    catalog: &Catalog,
    layers: Option<&BTreeSet<usize>>,
    tp: Tp,
) -> MimoResult<Vec<u64>> {
    (0..tp.count())
        .map(|rank| {
            Ok(aligned(&weight_plan::text_views(
                &catalog.tensors,
                &catalog.operands,
                layers,
                rank,
                tp,
            )?))
        })
        .collect()
}
