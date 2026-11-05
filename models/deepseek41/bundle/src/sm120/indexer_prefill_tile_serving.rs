//! Qualified eager prefill scoring at the measured full-batch sizes, and the
//! long band past the short band.
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

    /// The long band: eager full-index prefill whose largest position lies
    /// past the short band, with a page-rounded extent of at most 256K, under
    /// the same opt-outs as the short band.
    pub(crate) fn prefill_long_enabled(
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
                positions.iter().all(|&p| p >= 0)
                    && positions.iter().max().is_some_and(|&last| {
                        last >= 32768 && (u64::from(last as u32) + 1).next_multiple_of(64) <= 262144
                    })
            })
    }

    /// The tile4 CTA count of a `rows`-row full-source long-band batch, if
    /// the table admits its rows.
    pub(crate) fn prefill_long_selected(candidates: bool, rows: u32) -> Option<u32> {
        use model_deepseek41_contract::tuning::{Family, select};
        let (op, capacity) = if candidates {
            ("indexer.long_candidate_scores.prefill", 512)
        } else {
            ("indexer.long_compressed_scores.prefill", 4096)
        };
        select(
            &FLASH,
            Family::Indexer,
            op,
            "fp4",
            &[("top_k", 512)],
            capacity,
            rows,
        )?
        .knobs
        .get("ctas")?
        .count()
    }

    /// Retarget one batch's guarded scorer to the tile4 scorer at `ctas`
    /// CTAs, and its live selector and merge to the early-stop entries.
    /// Arguments, row grids, workspace and the selectors' grids are unchanged.
    pub(crate) fn prefill_long_route(
        mut calls: Vec<Lowered>,
        ctas: u32,
    ) -> Result<Vec<Lowered>, Refusal> {
        if !matches!(ctas, 4 | 8 | 16 | 32 | 192) {
            return Err(Refusal("long-band scoring requires a tile4 CTA count"));
        }
        let mut changed = 0;
        for call in &mut calls {
            if !call.launch.symbol.ends_with("_mma_fp4_guard_fast_v1") {
                continue;
            }
            if call.collective
                || call.launch.plan.grid != [192, call.launch.rows, 1]
                || call.launch.plan.block != [128, 1, 1]
            {
                return Err(Refusal(
                    "long-band scoring must retain its local four-warp grid",
                ));
            }
            call.launch.symbol =
                kernels_sm120::pooled_indexer::position_mma_prefill_tile::score_symbol(
                    &call.launch.symbol,
                    false,
                )?;
            call.launch.plan.grid[0] = ctas;
            changed += 1;
        }
        if changed != 1 {
            return Err(Refusal(
                "long-band scoring requires one guarded score launch",
            ));
        }
        Self::early_prefill_select(calls)
    }

    pub(crate) fn prefill_tile_selected(compact: bool, ratio2: bool, rows: u32) -> bool {
        let (op, capacity) = if compact || ratio2 {
            ("indexer.compressed_scores.prefill", 4096)
        } else {
            ("indexer.candidate_scores.prefill", 512)
        };
        use model_deepseek41_contract::tuning::{Family, select};
        select(
            &FLASH,
            Family::Indexer,
            op,
            "fp4",
            &[("top_k", 512)],
            capacity,
            rows,
        )
        .is_some()
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
            if !Self::prefill_tile_selected(
                compact,
                call.launch.symbol.contains("_r2_"),
                call.launch.rows,
            ) {
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
    fn prefill_tile_admission_is_bounded_and_can_be_disabled() {
        let mut shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let positions = [0, 8191, 32767];
        assert!(crate::Kernels::prefill_tile_enabled(
            &shape,
            KvStorage::Native,
            Some(&positions)
        ));
        shape.runtime.indexer_prefill_tile = false;
        assert!(!crate::Kernels::prefill_tile_enabled(
            &shape,
            KvStorage::Native,
            Some(&positions)
        ));
        shape.runtime.indexer_prefill_tile = true;
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

    /// The table's long-band rows, at each range edge and its neighbours:
    /// tile4 at 192 CTAs for small batches, fewer CTAs for wider ones, and
    /// today's route below 32 rows and above the largest measured batch.
    #[test]
    fn long_band_ctas_follow_the_measured_row_ranges() {
        let ctas = crate::Kernels::prefill_long_selected;
        for (rows, compressed, candidates) in [
            (1, None, None),
            (31, None, None),
            (32, Some(192), Some(192)),
            (191, Some(192), Some(192)),
            (192, Some(32), Some(192)),
            (447, Some(32), Some(192)),
            (448, Some(32), Some(32)),
            (512, Some(32), Some(32)),
            (513, Some(32), None),
            (1535, Some(32), None),
            (1536, Some(16), None),
            (4032, Some(16), None),
            (4033, None, None),
            (4096, None, None),
        ] {
            assert_eq!(ctas(false, rows), compressed, "{rows}");
            assert_eq!(ctas(true, rows), candidates, "{rows}");
        }
    }

    /// The long band starts where the short band ends and stops at a
    /// page-rounded 256K extent, for every row of the batch.
    #[test]
    fn long_band_admission_is_bounded_and_follows_the_tile_opt_out() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let long = |shape: &Shape, positions: &[i32]| {
            crate::Kernels::prefill_long_enabled(shape, KvStorage::Native, Some(positions))
        };
        let short = |positions: &[i32]| {
            crate::Kernels::prefill_tile_enabled(&shape, KvStorage::Native, Some(positions))
        };
        // Short band through 32767, long band from 32768: never both.
        assert!(short(&[0, 32767]) && !long(&shape, &[0, 32767]));
        assert!(!short(&[0, 32768]) && long(&shape, &[0, 32768]));
        // The largest position decides; a packed short row stays in its batch.
        assert!(long(&shape, &[5, 200000, 17, 131071]));
        // A page-rounded extent of 262144 is in, one position more is out.
        assert!(long(&shape, &[262143]));
        assert!(!long(&shape, &[262144]));
        assert!(!long(&shape, &[262080, 262145]));
        for positions in [&[][..], &[-1, 40000][..]] {
            assert!(!long(&shape, positions));
        }
        assert!(!crate::Kernels::prefill_long_enabled(
            &shape,
            KvStorage::Native,
            None
        ));
        assert!(!crate::Kernels::prefill_long_enabled(
            &shape,
            KvStorage::Bf16,
            Some(&[40000])
        ));
        // The tiled-route opt-out and the native scorer each disable it; so
        // do other top-k and candidate pools.
        for edit in [
            |s: &mut Shape| s.runtime.indexer_prefill_tile = false,
            |s: &mut Shape| s.runtime.indexer_mma = false,
            |s: &mut Shape| s.runtime.index_topk = 256,
            |s: &mut Shape| s.runtime.candidate_topk_blocks = 1024,
        ] {
            let mut changed = shape;
            edit(&mut changed);
            assert!(!long(&changed, &[40000]));
        }
    }
}
