//! Host-selected indexer schedules for the measured native decode layouts.
use bundle_runtime::inputs::PhaseInputs;
use model_deepseek41_contract::{geometry::FLASH, state::KvStorage};

use crate::{program::Phase, shape::Shape};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum Route {
    #[default]
    Accepted,
    VerifyHierarchy,
    Combined,
}

impl Route {
    pub(crate) fn select(
        shape: &Shape,
        storage: KvStorage,
        phase: Phase,
        inputs: &PhaseInputs,
    ) -> Self {
        if shape.geometry != &FLASH
            || storage != KvStorage::Native
            || !shape.runtime.indexer_mma
            || shape.runtime.index_topk != 512
            || shape.runtime.candidate_topk_blocks != 2048
        {
            return Self::Accepted;
        }
        let (items, per_item) = match phase {
            Phase::Decode(rows @ (8 | 32)) => (rows, 1),
            Phase::Verify { rows: 32, items: 8 } => (8, 4),
            _ => return Self::Accepted,
        };
        if inputs.live_tokens() != items * per_item
            || inputs.live_batch() != items
            || inputs.q_len.len() != items as usize
            || inputs.q_len.iter().any(|&n| n != per_item as i32)
            || inputs.seq_lens.len() != items as usize
        {
            return Self::Accepted;
        }
        // A key page contains 64 source tokens. Rounding request ends to pages
        // keeps the measured causal rows of one page in the same context band.
        // Every request must fit the band; mixed short/long batches stay on the
        // accepted schedule, as do contexts beyond the measured 256K maximum.
        let ends = inputs
            .seq_lens
            .iter()
            .map(|&n| u32::try_from(n).unwrap_or(0).div_ceil(64) * 64);
        let (lo, hi) = ends.fold((u32::MAX, 0), |(lo, hi), n| (lo.min(n), hi.max(n)));
        if lo >= 131072 && hi <= 262144 {
            Self::Combined
        } else if per_item == 4 && lo >= 32768 && hi < 131072 {
            Self::VerifyHierarchy
        } else {
            Self::Accepted
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;

    fn inputs(items: u32, per: u32, context: i32) -> PhaseInputs {
        PhaseInputs {
            item_order: (0..items as usize).collect(),
            token_ids: vec![7; (items * per) as usize],
            q_len: vec![per as i32; items as usize],
            seq_lens: (0..items).map(|i| context - (i % 3 * per) as i32).collect(),
            ..PhaseInputs::default()
        }
    }

    #[test]
    fn selects_only_measured_layouts_and_context_bands() {
        let mut shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        for phase in [
            Phase::Decode(8),
            Phase::Decode(32),
            Phase::Verify { rows: 32, items: 8 },
        ] {
            let verify = matches!(phase, Phase::Verify { .. });
            let (items, per) = if verify { (8, 4) } else { (phase.rows(), 1) };
            for context in [
                1024, 8192, 32768, 65536, 131072, 196608, 262144, 262145, 1048576,
            ] {
                let input = inputs(items, per, context);
                let expected = if (131072..=262144).contains(&context) {
                    Route::Combined
                } else if verify && (32768..131072).contains(&context) {
                    Route::VerifyHierarchy
                } else {
                    Route::Accepted
                };
                assert_eq!(
                    Route::select(&shape, KvStorage::Native, phase, &input),
                    expected
                );
                assert_eq!(
                    Route::select(&shape, KvStorage::Bf16, phase, &input),
                    Route::Accepted
                );
                shape.runtime.indexer_mma = false;
                assert_eq!(
                    Route::select(&shape, KvStorage::Native, phase, &input),
                    Route::Accepted
                );
                shape.runtime.indexer_mma = true;
                let mut partial = inputs(items, per, context);
                partial.token_ids.pop();
                assert_eq!(
                    Route::select(&shape, KvStorage::Native, phase, &partial),
                    Route::Accepted
                );
                let mut mixed = input;
                mixed.seq_lens[0] = 8192;
                assert_eq!(
                    Route::select(&shape, KvStorage::Native, phase, &mixed),
                    Route::Accepted
                );
            }
        }
        for phase in [
            Phase::Decode(1),
            Phase::Decode(4),
            Phase::Decode(16),
            Phase::Prefill { rows: 32 },
            Phase::Verify {
                rows: 32,
                items: 16,
            },
        ] {
            assert_eq!(
                Route::select(&shape, KvStorage::Native, phase, &inputs(8, 4, 131072)),
                Route::Accepted
            );
        }
        let mut uneven = inputs(8, 4, 131072);
        uneven.q_len[0] = 3;
        uneven.q_len[1] = 5;
        assert_eq!(
            Route::select(
                &shape,
                KvStorage::Native,
                Phase::Verify { rows: 32, items: 8 },
                &uneven
            ),
            Route::Accepted
        );
        for (kept, blocks) in [(31, 2048), (512, 1024), (1, 1)] {
            shape.runtime.index_topk = kept;
            shape.runtime.candidate_topk_blocks = blocks;
            assert_eq!(
                Route::select(
                    &shape,
                    KvStorage::Native,
                    Phase::Decode(32),
                    &inputs(32, 1, 131072),
                ),
                Route::Accepted
            );
        }
    }
}
