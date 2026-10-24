//! Tile-scale consumer entries for Flash TP4. Each build retains its raw
//! twin's pointer ABI, launch bounds and arithmetic. Wide gate and column
//! entries stay outside this catalog because their register counts rose.
use super::{moe, moe_direct_kernels};
use kernel_build::CubinSpec;
use kernels_sm120::moe::{
    chained::{down_scale_tiles, lanes, streamed},
    scale_tiles_expand,
};
use model_deepseek41_contract::geometry::FLASH;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

pub const GATE_CUBIN: &str = "moe_gate_scale_tiles_deepseek41_sm120";
pub const DOWN_CUBIN: &str = "moe_down_scale_tiles_deepseek41_sm120";
pub const EXPAND_CUBIN: &str = "moe_scale_tiles_expand_deepseek41_sm120";

pub const GATE_FAMILY: &str = "routed_moe_gate_scale_tiles";
pub const DOWN_FAMILY: &str = "routed_moe_down_scale_tiles";
pub const EXPAND_FAMILY: &str = "routed_moe_scale_tiles_expand";

pub fn cubins() -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let entries = moe_direct_kernels::entries(&FLASH)?;
    let e = entries
        .iter()
        .find(|e| e.experts() == 384)
        .ok_or("missing Flash TP4 backbone")?;
    let projections: Vec<_> = moe::streamed_projections(&FLASH)?
        .into_iter()
        .filter(|p| p.query.projection.gemm.grouped.unwrap().experts == 384)
        .collect();
    let mut gate = lanes::scale_tiles_cubin_spec(GATE_CUBIN, e).map_err(|e| e.to_string())?;
    gate.entries.retain(|e| e.symbol.contains("_scale_tiles_w"));
    for width in [1, 2, 4] {
        gate.entries
            .push(lanes::resident_scale_tiles_entry(&e.gate, width).map_err(|e| e.to_string())?);
        gate.entries
            .push(lanes::absorb_scale_tiles_entry(e, width).map_err(|e| e.to_string())?);
    }
    for p in projections.iter().filter(|p| p.gate_up()) {
        for width in [1, 2, 4] {
            gate.entries.push(
                lanes::streamed_scale_tiles_entry(p, streamed::Stage::Small, width)
                    .map_err(|e| e.to_string())?,
            );
        }
    }
    let mut down =
        down_scale_tiles::cubin_spec(DOWN_CUBIN, e, &projections).map_err(|e| e.to_string())?;
    down.entries.retain(|e| e.symbol.contains("_scale_tiles_w"));
    if gate.entries.len() != 15 || down.entries.len() != 21 {
        return Err("scale consumer catalog changed size".into());
    }
    Ok(vec![
        (GATE_FAMILY, gate),
        (DOWN_FAMILY, down),
        (EXPAND_FAMILY, scale_tiles_expand::cubin_spec(EXPAND_CUBIN)),
    ])
}

/// Each encoded consumer declares its exact raw counterpart and width.
/// This closed map is also the scale-layout declaration used at binding.
fn variants() -> &'static BTreeMap<(String, u32), String> {
    static MAP: LazyLock<BTreeMap<(String, u32), String>> = LazyLock::new(|| {
        let mut map = BTreeMap::new();
        for (family, spec) in cubins().expect("Flash scale consumer catalog") {
            if family == EXPAND_FAMILY {
                continue;
            }
            for entry in spec.entries {
                let (base, width) = entry
                    .symbol
                    .rsplit_once("_scale_tiles_w")
                    .expect("encoded scale symbol");
                let width = width.strip_suffix("_v1").unwrap().parse::<u32>().unwrap();
                assert!(
                    map.insert((format!("{base}_v1"), width), entry.symbol)
                        .is_none()
                );
            }
        }
        map
    });
    &MAP
}

pub fn variant(raw: &str, width: u32) -> Option<&'static str> {
    variants().get(&(raw.into(), width)).map(String::as_str)
}

/// Streamed pack and merge stages do not read either weight scale bank.
/// Unknown auxiliary stages conservatively retain expansion.
pub fn auxiliary(symbol: &str) -> bool {
    static SYMBOLS: LazyLock<BTreeSet<String>> = LazyLock::new(|| {
        moe::streamed_projections(&FLASH)
            .expect("Flash streamed projections")
            .iter()
            .filter(|p| p.query.projection.gemm.grouped.unwrap().experts == 384)
            .flat_map(|p| {
                [
                    streamed::symbol(p, streamed::Stage::Pack),
                    streamed::symbol(p, streamed::Stage::Merge),
                ]
            })
            .collect()
    });
    SYMBOLS.contains(symbol)
}
