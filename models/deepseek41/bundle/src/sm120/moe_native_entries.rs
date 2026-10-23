//! The opt-in native MoE prefill entries (`kernels_sm120::moe::native_grouped`)
//! for Flash TP4's 384-expert bank: one serving tile per projection, shared
//! by the build script, the bundle and the qualification lab, and the
//! gate/up tile with the intermediate ceil quantizer in its epilogue
//! (`native_grouped::Ceil`) with its fixup, in their own cubin. Unfused
//! gate/up steps run the wider r2w6s3 tile from a third cubin, with the
//! scanned inner loop from a fourth, which also holds the two-CTA r1w4s2
//! down tile of unfused steps and the admission scans before them
//! (models/deepseek41/lab/native_tune.md, native_admission.md).
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::precision::quantizer::BlockQuantization;
use kernels_sm120::moe::native_grouped::{self as native, Ceil, Shape, Tile, Tune, Variant};
use model_deepseek41_contract::{geometry::FLASH, moe};
pub const FAMILY: &str = "native_moe_prefill";
pub const CUBIN: &str = "native_moe_prefill_deepseek41_sm120";
pub const CEIL_FAMILY: &str = "native_ceil";
pub const CEIL_CUBIN: &str = "native_ceil_deepseek41_sm120";
pub const WIDE_FAMILY: &str = "native_moe_gateup_wide";
pub const WIDE_CUBIN: &str = "native_moe_gateup_wide_deepseek41_sm120";
pub const LEAN_FAMILY: &str = "native_moe_lean";
pub const LEAN_CUBIN: &str = "native_moe_lean_deepseek41_sm120";
/// Entries compile for the largest prefill bucket; smaller steps launch
/// the same arithmetic with fewer live rows.
pub const CAPACITY: u32 = 8192;
/// The gate/up tile of [`CUBIN`] and of the fused ceil.
pub const GATE_UP_TILE: Tile = native::GATE_UP_TILES[0];
/// The unfused gate/up's tile: 96 columns per strip instead of 64, the
/// same arithmetic per output (the native MoE lab found every tile variant
/// bitwise equal) and 3.9% faster per captured 8192-row layer.
pub const WIDE_GATE_UP_TILE: Tile = native::GATE_UP_TILES[2];
pub const DOWN_TILE: Tile = native::DOWN_TILES[0];
/// Unfused down steps' tile: 64-row tiles whose registers are bounded for
/// two CTAs per SM, run with the lean loop. Bitwise the served tile on
/// every qualified case and 4.3% faster per 8192-row layer (6.1% on the
/// captured layers) in the native tuning pair.
pub const PAIRED_DOWN_TILE: Tile = Tile {
    row_warps: 1,
    col_warps: 4,
    stages: 2,
};
/// The unfused steps' inner loop: the lean loop (scale bytes by the MMA's
/// byte selectors) with the whole activation admission, NaN codes and the
/// scale-range product check, in an admission scan before the pack.
/// Bitwise the served loop and refusing exactly its inputs on every
/// qualified case and control; 13% (gate/up) and 8% (two-CTA down) faster
/// than the lean loop per 8192-row layer.
pub const UNFUSED_LOOP: Tune = Tune::Scanned;

/// The target bank's expert set (384 experts, top 6).
pub fn target_set() -> moe::Experts {
    moe::expert_sets(&FLASH)
        .into_iter()
        .find(|set| set.count == FLASH.shape.moe.experts)
        .expect("Flash's target expert bank")
}

pub fn shape(kind: GatedMlpProjection) -> Shape {
    let query = moe::projection(&FLASH, 4, target_set(), kind, CAPACITY).expect("Flash TP4 MoE");
    Shape::from_query(&query, CAPACITY).expect("native tiles serve Flash's experts")
}

/// The served tile of an unfused step of `kind`.
pub fn tile(kind: GatedMlpProjection) -> Tile {
    match kind {
        GatedMlpProjection::GateUp => WIDE_GATE_UP_TILE,
        GatedMlpProjection::Down => DOWN_TILE,
    }
}

pub fn cubins() -> Vec<kernel_build::CubinSpec> {
    let instances = [
        (shape(GatedMlpProjection::GateUp), GATE_UP_TILE),
        (shape(GatedMlpProjection::Down), DOWN_TILE),
    ];
    vec![native::cubin_spec(CUBIN, &instances, false).expect("native MoE serving cubin")]
}

/// The wide gate/up's pack and tiles.
pub fn wide_cubins() -> Vec<kernel_build::CubinSpec> {
    let instances = [(shape(GatedMlpProjection::GateUp), WIDE_GATE_UP_TILE)];
    vec![native::cubin_spec(WIDE_CUBIN, &instances, false).expect("wide gate/up serving cubin")]
}

/// The unfused steps' entries: the wide gate/up tiles (their pack is
/// [`WIDE_CUBIN`]'s), the two-CTA down pack and tiles, and both
/// projections' admission scans.
pub fn lean_cubins() -> Vec<kernel_build::CubinSpec> {
    let gate_up = shape(GatedMlpProjection::GateUp);
    let down = shape(GatedMlpProjection::Down);
    let mut entries = native::tune_entries(
        &gate_up,
        &WIDE_GATE_UP_TILE,
        &[UNFUSED_LOOP],
        Variant::Serving,
    )
    .expect("lean gate/up tiles");
    let mut pack = native::entries(&down, &PAIRED_DOWN_TILE, false).expect("two-CTA down");
    pack.truncate(1);
    entries.extend(pack);
    let mut tiles =
        native::tune_entries(&down, &PAIRED_DOWN_TILE, &[UNFUSED_LOOP], Variant::Serving)
            .expect("lean two-CTA down tiles");
    // Two CTAs per SM; the build refuses the bound if it spills.
    tiles[0].min_blocks = 2;
    entries.extend(tiles);
    entries.push(native::admission_scan_entry(&gate_up));
    entries.push(native::admission_scan_entry(&down));
    let mut spec = native::cubin_spec(LEAN_CUBIN, &[], false).expect("lean serving cubin");
    spec.entries = entries;
    vec![spec]
}

/// The serving intermediate quantizer's amax floor (`moe::quantizer` of
/// the gate/up rows), which the fused tiles reproduce.
pub fn ceil() -> Ceil {
    let shape = shape(GatedMlpProjection::GateUp);
    match moe::quantizer(shape.top_k * shape.n).format {
        BlockQuantization::E4M3Ceil32 { amax_floor_bits } => Ceil {
            floor_bits: amax_floor_bits,
        },
        other => panic!("the intermediate quantizer is {other:?}"),
    }
}

/// The fused gate/up tiles and the fixup; the pack stays in [`CUBIN`].
pub fn ceil_cubins() -> Vec<kernel_build::CubinSpec> {
    let gate_up = GatedMlpProjection::GateUp;
    vec![
        native::ceil_cubin_spec(CEIL_CUBIN, &ceil(), &[(shape(gate_up), GATE_UP_TILE)])
            .expect("fused ceil serving cubin"),
    ]
}

/// `spec`, a served native cubin, with each of its gate/up compute
/// entries' row-tiled build after its entries
/// (`native_grouped::row_tiles_entry`): the entry list the served module
/// takes when it binds the tiled bank.
pub fn row_tiles(mut spec: kernel_build::CubinSpec) -> kernel_build::CubinSpec {
    let gate_up = shape(GatedMlpProjection::GateUp);
    let tiled = spec
        .entries
        .iter()
        .filter_map(|e| native::row_tiles_entry(&gate_up, e).ok())
        .collect::<Vec<_>>();
    spec.entries.extend(tiled);
    spec
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each served native cubin with its tiled gate/up entries keeps every
    /// served entry first; a tiled entry is its row-major one with the bank
    /// read in row tiles (`Config`'s `Tiles`), the same ABI, bounds and
    /// launch, and no down, pack or scan entry is tiled.
    #[test]
    fn row_tiled_native_entries_differ_only_in_their_weight_addresses() {
        let gate_up = shape(GatedMlpProjection::GateUp);
        let rows_config = gate_up.config();
        let tiled_config = format!("{},true>", &rows_config[..rows_config.len() - 1]);
        let mut added = Vec::new();
        for (spec, count) in [
            (cubins().remove(0), 1),
            (ceil_cubins().remove(0), 1),
            (wide_cubins().remove(0), 1),
            (lean_cubins().remove(0), 1),
        ] {
            let with = row_tiles(spec.clone());
            let ids = |entries: &[kernel_build::EntrySpec]| {
                entries
                    .iter()
                    .map(|e| (e.symbol.clone(), e.body.clone(), e.params.clone()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(ids(&with.entries[..spec.entries.len()]), ids(&spec.entries));
            assert_eq!(
                with.entries.len(),
                spec.entries.len() + count,
                "{}",
                spec.name
            );
            for tiled in &with.entries[spec.entries.len()..] {
                let rows = spec
                    .entries
                    .iter()
                    .find(|e| native::row_tiles_symbol(&e.symbol) == tiled.symbol)
                    .unwrap();
                assert!(rows.body.contains(&rows_config));
                assert_eq!(
                    tiled.body,
                    rows.body.replace(&rows_config, &tiled_config),
                    "{}",
                    tiled.symbol
                );
                assert_eq!(
                    (&tiled.params, tiled.max_threads, tiled.min_blocks),
                    (&rows.params, rows.max_threads, rows.min_blocks)
                );
                added.push(tiled.symbol.clone());
            }
        }
        added.sort();
        added.dedup();
        assert_eq!(added.len(), 4);
        let down = shape(GatedMlpProjection::Down);
        for e in native::entries(&down, &DOWN_TILE, false).unwrap() {
            assert!(native::row_tiles_entry(&down, &e).is_err());
            assert!(native::row_tiles_entry(&gate_up, &e).is_err());
        }
    }
}
