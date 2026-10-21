//! Exact early-stop prefill selection under the prefill tile admission.
use crate::ops::Lowered;
use kernel_core::contract::Refusal;
use kernels_sm120::pooled_indexer::position_select_early as early;

impl crate::Kernels {
    /// Retarget the one local selector and its merge; scoring, arguments
    /// and grids are unchanged.
    pub(crate) fn early_prefill_select(mut calls: Vec<Lowered>) -> Result<Vec<Lowered>, Refusal> {
        let mut changed = [0; 2];
        for call in &mut calls {
            let symbol = &call.launch.symbol;
            if !(symbol.ends_with("_select_v3_live_v1")
                || symbol.ends_with("_select_v3_live_merge_v1")
                || symbol.ends_with("_select_v3_valid_v1")
                || symbol.ends_with("_select_v3_valid_merge_v1"))
            {
                continue;
            }
            changed[usize::from(symbol.ends_with("_merge_v1"))] += 1;
            call.launch.symbol =
                early::target(symbol, call.launch.plan, call.launch.rows, call.collective)?;
        }
        if changed != [1, 1] {
            return Err(Refusal(
                "early selection requires one local selector and its merge",
            ));
        }
        Ok(calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::{dispatch::BufferBindings, pipeline::PreparedLaunch};
    use kernel_core::contract::{ArgumentList, LaunchPlan};

    fn call(symbol: &str, grid_x: u32) -> Lowered {
        Lowered {
            launch: PreparedLaunch {
                symbol: symbol.into(),
                rows: 512,
                plan: LaunchPlan {
                    grid: [grid_x, 512, 1],
                    block: [256, 1, 1],
                    dynamic_shared_bytes: 0,
                    workspace_bytes: 0,
                    cluster: [1, 1, 1],
                    pdl: true,
                },
                arguments: ArgumentList::default(),
                bindings: BufferBindings::default(),
            },
            collective: false,
        }
    }

    #[test]
    fn early_selection_retargets_exactly_one_local_selector_and_merge() {
        let stem = "chop_position_blocks_deepseek41_r1_select_v3_valid";
        let scores =
            "chop_position_blocks_deepseek41_r1_scores_v3_valid_v1_mma_fp4_prefill_tile_v1";
        let calls = vec![
            call(scores, 8),
            call(&format!("{stem}_v1"), 32),
            call(&format!("{stem}_merge_v1"), 1),
        ];
        let symbols: Vec<_> = crate::Kernels::early_prefill_select(calls)
            .unwrap()
            .into_iter()
            .map(|c| c.launch.symbol)
            .collect();
        assert_eq!(
            symbols,
            [
                scores.into(),
                format!("{stem}_early_v1"),
                format!("{stem}_merge_early_v1")
            ]
        );
        for calls in [
            vec![call(scores, 8), call(&format!("{stem}_v1"), 32)],
            vec![call(&format!("{stem}_merge_v1"), 1)],
            vec![
                call(&format!("{stem}_v1"), 32),
                call(&format!("{stem}_v1"), 32),
                call(&format!("{stem}_merge_v1"), 1),
            ],
            vec![
                call(&format!("{stem}_v1"), 8),
                call(&format!("{stem}_merge_v1"), 1),
            ],
        ] {
            assert!(crate::Kernels::early_prefill_select(calls).is_err());
        }
        let mut collective = call(&format!("{stem}_v1"), 32);
        collective.collective = true;
        assert!(
            crate::Kernels::early_prefill_select(vec![
                collective,
                call(&format!("{stem}_merge_v1"), 1)
            ])
            .is_err()
        );
    }
}
