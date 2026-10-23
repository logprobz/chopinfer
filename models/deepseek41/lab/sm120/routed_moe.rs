//! The routed MoE family's cubins as the bundle builds them: the bundle's
//! registration (`sm120/moe.rs`, `registration` beside this module) with
//! each entry served on a row-tiled bank replaced in place by its row-tiled
//! build (`moe_row_tiles`). `build_moe` builds them against the family's
//! lock for the GPU snapshot.
use bundle_deepseek41::sm120::moe_row_tiles;
use kernel_build::CubinSpec;
use model_deepseek41_contract::geometry::Geometry;
use std::collections::BTreeSet;

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<CubinSpec>, String> {
    let mut specs = super::registration::cubins(geometries)?;
    let symbols: BTreeSet<String> = specs
        .iter()
        .flat_map(|(_, spec)| spec.entries.iter().map(|e| e.symbol.clone()))
        .collect();
    for g in geometries {
        // Builds of entries in other families are theirs to serve.
        let builds = moe_row_tiles::builds(g)?
            .into_iter()
            .filter(|(rows, _)| symbols.contains(rows))
            .collect();
        moe_row_tiles::serve(&mut specs, &builds)?;
    }
    Ok(specs.into_iter().map(|(_, spec)| spec).collect())
}
