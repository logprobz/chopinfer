//! The Flash TP4 backbone gate/up bank as the bundle serves it
//! (`bundle_deepseek41::sm120::moe_row_tiles`): in 64-row strips of 64-byte
//! code and 4-byte scale tiles, the codes in quad-lane order where the
//! served layout is `RoutedLayout::Lanes`, with each row-major gate/up
//! compute entry a served path launches on it embedded only as its build
//! for that layout. A harness that plans launches through the kernels
//! backend (row-major symbols) runs them as the bundle does with
//! [`launches`] over a bank tiled by [`pack`].
use bundle_deepseek41::sm120::moe_row_tiles::{self, EXPERTS, RANKS};
use kernel_core::backend::{Launch, View};
use kernel_core::contract::Argument;
use kernel_core::precision::buffers::QuantizedWeights;
use kernel_core::transform::{QuadLaneLayout, RowTileLayout};
use model_deepseek41_contract::geometry::FLASH;
use model_deepseek41_contract::moe::Experts;
use model_deepseek41_contract::weight_plan::RoutedLayout;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Whether the bundle stores the Flash gate/up bank of `set` at `ranks` in
/// row tiles, in either code order.
pub fn tiled(ranks: u32, set: Experts) -> bool {
    ranks == RANKS && set.count == EXPERTS && RoutedLayout::served(&FLASH, ranks).tiled()
}

/// Stores a row-major gate/up bank of `rows` rows of `k` MXFP4 elements in
/// place as the bundle serves it at TP4: its codes and its UE8M0 scales in
/// row tiles, the codes in quad-lane order for `RoutedLayout::Lanes`.
pub fn pack(codes: &mut [u8], scales: &mut [u8], rows: usize, k: usize) -> Result<(), String> {
    let mut scratch = Vec::new();
    for (bytes, (row_bytes, tile_bytes)) in [(&mut *codes, (k / 2, 64)), (scales, (k / 32, 4))] {
        RowTileLayout {
            rows,
            row_bytes,
            tile_rows: 64,
            tile_bytes,
        }
        .pack_in_place(bytes, &mut scratch)
        .map_err(|e| e.to_string())?;
    }
    if RoutedLayout::served(&FLASH, RANKS) == RoutedLayout::Lanes {
        QuadLaneLayout { bytes: codes.len() }
            .pack_in_place(codes)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Each row-major entry the bundle serves as a build for its tiled layout,
/// with that build's symbol.
pub fn symbols() -> &'static BTreeMap<String, String> {
    static SYMBOLS: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    SYMBOLS.get_or_init(|| moe_row_tiles::symbols(&FLASH).expect("row-tiled MoE builds"))
}

/// `ops` as the bundle serves them: each launch of an entry it serves as a
/// tiled build runs that build over `tiles` in place of the row-major
/// `rows` bank, which it must read once. Every other launch is unchanged.
pub fn launches(
    ops: Vec<Launch>,
    rows: QuantizedWeights<View>,
    tiles: Option<QuantizedWeights<View>>,
) -> Result<Vec<Launch>, String> {
    let handles = |w: QuantizedWeights<View>| {
        [w.codes, w.local_scales.unwrap()].map(|v| Argument::DeviceBuffer(v.handle))
    };
    let bank = handles(rows);
    ops.into_iter()
        .map(|mut op| {
            let Some(tiled) = symbols().get(&op.symbol) else {
                return Ok(op);
            };
            let tiles = handles(tiles.ok_or_else(|| format!("{}: no row tiles", op.symbol))?);
            let reads: Vec<usize> = bank
                .iter()
                .map(|b| op.arguments.0.iter().filter(|a| *a == b).count())
                .collect();
            if reads != [1, 1] {
                return Err(format!(
                    "{}: does not read the gate/up bank once",
                    op.symbol
                ));
            }
            for argument in &mut op.arguments.0 {
                if let Some(i) = bank.iter().position(|b| b == argument) {
                    *argument = tiles[i].clone();
                }
            }
            op.symbol.clone_from(tiled);
            Ok(op)
        })
        .collect()
}
