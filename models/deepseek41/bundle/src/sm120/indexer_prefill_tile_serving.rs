//! Opt-in eager prefill scoring. Qualification and timing are still pending.
use crate::{ops::Lowered, shape::Shape};
use kernel_core::contract::Refusal;
use model_deepseek41_contract::{geometry::FLASH, state::KvStorage};

impl crate::Kernels {
    pub(crate) fn prefill_tile_enabled(
        shape: &Shape,
        storage: KvStorage,
        positions: Option<&[i32]>,
    ) -> bool {
        shape.geometry == &FLASH
            && shape.runtime.indexer_prefill_tile
            && shape.runtime.indexer_mma
            && storage == KvStorage::Native
            && shape.runtime.index_topk == 512
            && shape.runtime.candidate_topk_blocks == 2048
            && positions.is_some_and(|positions| {
                !positions.is_empty() && positions.iter().all(|&p| (0..32768).contains(&p))
            })
    }

    pub(crate) fn prefill_tile_scores(
        mut calls: Vec<Lowered>,
        compact: bool,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut changed = 0;
        for call in &mut calls {
            if !call.launch.symbol.ends_with("_mma_fp4_guard_fast_v1") {
                continue;
            }
            if !matches!(call.launch.rows, 512 | 4096) {
                return Ok(calls);
            }
            if call.collective
                || call.launch.plan.grid != [192, call.launch.rows, 1]
                || call.launch.plan.block != [128, 1, 1]
            {
                return Err(Refusal(
                    "prefill scoring must retain its local four-warp grid",
                ));
            }
            call.launch.symbol =
                kernels_sm120::pooled_indexer::position_mma_prefill_tile::score_symbol(
                    &call.launch.symbol,
                    compact,
                )?;
            call.launch.plan.grid[0] = 8;
            changed += 1;
        }
        if changed != 1 {
            return Err(Refusal("prefill tiling requires one guarded score launch"));
        }
        Ok(calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;

    #[test]
    fn prefill_tile_admission_is_explicit_and_bounded() {
        let mut shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let positions = [0, 8191, 32767];
        assert!(!crate::Kernels::prefill_tile_enabled(
            &shape,
            KvStorage::Native,
            Some(&positions)
        ));
        shape.runtime.indexer_prefill_tile = true;
        assert!(crate::Kernels::prefill_tile_enabled(
            &shape,
            KvStorage::Native,
            Some(&positions)
        ));
        assert!(!crate::Kernels::prefill_tile_enabled(
            &shape,
            KvStorage::Bf16,
            Some(&positions)
        ));
        for positions in [None, Some(&[][..]), Some(&[-1, 0][..]), Some(&[32768][..])] {
            assert!(!crate::Kernels::prefill_tile_enabled(
                &shape,
                KvStorage::Native,
                positions
            ));
        }
        shape.runtime.index_topk = 31;
        assert!(!crate::Kernels::prefill_tile_enabled(
            &shape,
            KvStorage::Native,
            Some(&positions)
        ));
        shape.runtime.index_topk = 512;
        shape.runtime.candidate_topk_blocks = 1024;
        assert!(!crate::Kernels::prefill_tile_enabled(
            &shape,
            KvStorage::Native,
            Some(&positions)
        ));
        shape.runtime.candidate_topk_blocks = 2048;
        shape.runtime.indexer_mma = false;
        assert!(!crate::Kernels::prefill_tile_enabled(
            &shape,
            KvStorage::Native,
            Some(&positions)
        ));
    }
}
