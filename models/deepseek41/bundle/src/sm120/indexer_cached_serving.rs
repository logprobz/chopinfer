//! The measured 128K eager-prefill band retains coarse scoring and its merge.
use crate::ops::Lowered;
use kernel_core::contract::Refusal;
use kernels_sm120::pooled_indexer::position_select_cached as cached;

impl crate::Kernels {
    pub(crate) fn cached_prefill_indexer_enabled(
        ratio: u32,
        width: u32,
        rows: u32,
        top_k: u32,
    ) -> bool {
        matches!(ratio, 1 | 2)
            && u64::from(width) * u64::from(ratio) == 131072
            && top_k == 512
            && Self::cached_prefill_indexer_rows(width, rows, rows)
    }

    pub(crate) fn cached_prefill_indexer_rows(width: u32, capacity: u32, rows: u32) -> bool {
        use model_deepseek41_contract::{
            geometry::FLASH,
            tuning::{Family, select},
        };
        select(
            &FLASH,
            Family::Indexer,
            "indexer.cached.prefill",
            "f32",
            &[("width", width), ("top_k", 512)],
            capacity,
            rows,
        )
        .is_some()
    }

    pub(crate) fn cached_prefill_indexer_select(
        mut calls: Vec<Lowered>,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut changed = 0;
        for call in &mut calls {
            if !call.launch.symbol.ends_with("_select_v3_live_v1") {
                continue;
            }
            if call.collective
                || call.launch.plan.grid != [32, call.launch.rows, 1]
                || call.launch.plan.block != [256, 1, 1]
            {
                return Err(Refusal(
                    "cached prefill selection requires the qualified local grid",
                ));
            }
            call.launch.symbol = cached::select_symbol(&call.launch.symbol)?;
            changed += 1;
        }
        if changed != 1 {
            return Err(Refusal(
                "cached prefill selection requires one local selector",
            ));
        }
        Ok(calls)
    }
}
