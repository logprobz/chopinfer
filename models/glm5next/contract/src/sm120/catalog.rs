//! glm5next's catalog for the shared kernels' consumers (kernel-lab's
//! model-neutral projection inventory, `kernels_sm120::ModelCatalog`).
//! glm5next's own engines are compiled by its bundle; the catalog
//! contributes only the generated projection offers of the projections the
//! bundle lowers through the projection API (`crate::precision`) and their
//! layout copies, so the lab's prepared projection jobs check them at their
//! served shapes.

use std::collections::BTreeSet;

use kernel_core::contract::WeightFormat;
use kernels_sm120::{ModelCatalog, ProjectionInventory};

use super::precision;
use crate::contract::Shape;
use crate::quantization::LAYOUTS;

/// glm5next's catalog entries: its projection inventory only.
pub static CATALOG: ModelCatalog = ModelCatalog {
    name: "glm5next",
    projections,
    ..ModelCatalog::EMPTY
};

/// Each lowered projection with every weight format a checkpoint layout
/// stores it in on `shape`'s rank 0 (`crate::precision::layout_formats`), in
/// [`crate::precision::LOWERED`] order then layout order: block FP8 where the FP8
/// export quantizes it, BF16 where a layout keeps it unquantized, NVFP4 for
/// the NVIDIA export's dense down projection.
pub fn lowered_formats(shape: Shape) -> Result<Vec<(&'static str, WeightFormat)>, String> {
    let mut stored = Vec::new();
    for layout in LAYOUTS {
        stored.push(crate::precision::layout_formats(shape, layout)?);
    }
    let mut formats = Vec::new();
    for &id in crate::precision::LOWERED {
        for layout in &stored {
            for &(stored_id, format) in layout {
                if stored_id == id && !formats.contains(&(id, format)) {
                    formats.push((id, format));
                }
            }
        }
    }
    Ok(formats)
}

/// The generated offers of every lowered projection's format at every
/// declared use, on the served shape (an offer repeating another's stages
/// at the same use, projections of one shape, is listed once), and the
/// layout copies those uses run after their GEMMs (`precision::copies`).
fn projections() -> Result<ProjectionInventory, String> {
    let mut inventory = ProjectionInventory::default();
    let mut seen = BTreeSet::new();
    for (id, weight) in lowered_formats(Shape::TP4)? {
        for (usage, _) in crate::precision::uses(id)? {
            let generated = precision::offer(id, weight, usage)?;
            let stages = generated
                .offer
                .capability()
                .map_err(|e| e.to_string())?
                .stages;
            if seen.insert((stages, format!("{usage:?}"))) {
                inventory.0.push(generated.offer);
            }
        }
    }
    inventory.2 = precision::copies(Shape::TP4)?;
    Ok(inventory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::ScaleMode;
    use kernels_sm120::precision::Compute;

    const FP8: WeightFormat = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);

    #[test]
    fn every_lowered_projection_has_its_stored_formats() {
        let formats = lowered_formats(Shape::TP4).unwrap();
        for &id in crate::precision::LOWERED {
            assert!(formats.iter().any(|(f, _)| *f == id), "{id}");
        }
        // The NVFP4 exports keep the FP8 export's quantized latent
        // projections in BF16, and NVIDIA's dense down is NVFP4; the head
        // stays BF16 everywhere.
        for (id, fp8, nvfp4) in [
            ("latent.q-a", true, false),
            ("dense.down", true, true),
            ("head", false, false),
        ] {
            let of = |weight| formats.contains(&(id, weight));
            assert_eq!(
                (of(FP8), of(WeightFormat::Bf16), of(WeightFormat::Nvfp4)),
                (fp8, true, nvfp4),
                "{id}"
            );
        }
        assert_eq!(
            formats
                .iter()
                .filter(|(_, w)| *w == WeightFormat::Nvfp4)
                .count(),
            1
        );
    }

    #[test]
    fn the_inventory_offers_every_lowered_use_once() {
        let (offers, retained, copies) = (CATALOG.projections)().unwrap();
        assert!(retained.is_empty());
        let symbol =
            |copy: &kernels_sm120::row::precision::LayoutCopy| copy.stage().entry().symbol.clone();
        assert_eq!(
            copies.iter().map(symbol).collect::<Vec<_>>(),
            precision::copies(Shape::TP4)
                .unwrap()
                .iter()
                .map(symbol)
                .collect::<Vec<_>>()
        );
        assert!(!copies.is_empty());
        let mut keys = BTreeSet::new();
        for offer in &offers {
            assert!(matches!(offer.handle().compute, Compute::Dense(_)));
            let stages = offer.capability().unwrap().stages;
            assert!(keys.insert((stages, format!("{:?}", offer.usage()))));
        }
        // Every (projection, format, use) is served by a listed offer.
        for (id, weight) in lowered_formats(Shape::TP4).unwrap() {
            for (usage, _) in crate::precision::uses(id).unwrap() {
                let stages = precision::offer(id, weight, usage)
                    .unwrap()
                    .offer
                    .capability()
                    .unwrap()
                    .stages;
                assert!(keys.contains(&(stages, format!("{usage:?}"))), "{id}");
            }
        }
    }
}
