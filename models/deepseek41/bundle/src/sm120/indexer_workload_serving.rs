//! Remap only the measured dense scorer. Packing and every argument stay fixed.
use crate::ops::Lowered;
use kernel_core::contract::Refusal;
use kernels_sm120::pooled_indexer::position_mma_tile as tile;

impl crate::Kernels {
    pub(crate) fn tiled_indexer_scores(mut calls: Vec<Lowered>) -> Result<Vec<Lowered>, Refusal> {
        let mut changed = 0;
        for call in &mut calls {
            if !call.launch.symbol.ends_with("_mma_fp4_guard_fast_v1") {
                continue;
            }
            if call.collective
                || !call.launch.symbol.starts_with("chop_position_indexer_")
                || call.launch.plan.grid != [192, call.launch.rows, 1]
                || call.launch.plan.block != [128, 1, 1]
            {
                return Err(Refusal("tiled scoring requires the qualified dense grid"));
            }
            call.launch.symbol = tile::VARIANTS[1].symbol(&call.launch.symbol)?;
            changed += 1;
        }
        if changed != 1 {
            return Err(Refusal("tiled scoring requires one dense score launch"));
        }
        Ok(calls)
    }
}
