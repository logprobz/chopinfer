//! Lab tile and stage probes for dense guarded FP4 scores.
use super::{position, position_mma_guard, position_mma_guard_fast};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::{backend::Launch, contract::Refusal};

/// Only mode zero retains the reference score semantics. Other modes are ablations.
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub tiles: u32,
    pub mode: u32,
    pub trace: bool,
}
pub const VARIANTS: [Variant; 6] = [
    Variant {
        tiles: 1,
        mode: 0,
        trace: false,
    },
    Variant {
        tiles: 4,
        mode: 0,
        trace: false,
    },
    Variant {
        tiles: 1,
        mode: 1,
        trace: false,
    },
    Variant {
        tiles: 1,
        mode: 2,
        trace: false,
    },
    Variant {
        tiles: 1,
        mode: 0,
        trace: true,
    },
    Variant {
        tiles: 4,
        mode: 0,
        trace: true,
    },
];
impl Variant {
    pub fn symbol(self, source: &str) -> Result<String, Refusal> {
        if !matches!(self.tiles, 1 | 4) || self.mode > 2 {
            return Err(Refusal("unsupported dense score probe"));
        }
        let stem = source
            .strip_suffix("_mma_fp4_guard_fast_v1")
            .ok_or(Refusal("score probe requires the accepted coarse guard"))?;
        Ok(format!(
            "{stem}_mma_fp4_tile{}_mode{}_trace{}_v1",
            self.tiles,
            self.mode,
            u32::from(self.trace)
        ))
    }
    pub fn spec(self, entry: &position::Entry) -> Result<EntrySpec, Refusal> {
        let mut spec = position_mma_guard::dense_specs(entry)?.remove(1);
        spec.symbol = self.symbol(&position_mma_guard_fast::score_symbol(&spec.symbol))?;
        // Preserve the pack-failure dispatch verbatim, including its grid.
        spec.body = spec.body.replace(
            "chop::position_mma_guard::scores<",
            &format!(
                "chop::position_mma_tile::scores<{},{},{},",
                self.tiles, self.mode, self.trace
            ),
        );
        if self.trace {
            spec.params.push(EntryParam::DevicePointer {
                name: "trace".into(),
                cxx_type: "uint64_t".into(),
            });
            spec.body = spec.body.replace(
                "reinterpret_cast<const uint32_t*>(scratch+qstatus_offset));",
                "reinterpret_cast<const uint32_t*>(scratch+qstatus_offset),trace);",
            );
        }
        Ok(spec)
    }
    pub fn remap(self, call: &mut Launch) -> Result<(), Refusal> {
        if call.plan.block != [128, 1, 1] || call.plan.grid[2] != 1 {
            return Err(Refusal("score probe requires a four-warp row grid"));
        }
        call.symbol = self.symbol(&call.symbol)?;
        Ok(())
    }
}
pub fn cubin_spec(entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position_mma_guard::cubin_spec("position_mma_tile_sm120", entries);
    spec.headers = vec!["#include \"pooled_indexer/position_mma_tile.cuh\"".into()];
    spec
}
