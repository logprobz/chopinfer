use kernel_core::{
    backend::Launch,
    contract::ArgumentList,
    pooled_indexer::{IndexerCandidates, PositionIndexerQuery},
};
use kernels_sm120::pooled_indexer::{
    position, position_live, position_mma_guard, position_mma_guard_fast,
    position_mma_prefill as tiled, position_select_cached as cached,
};

fn entry(ratio: u32) -> position::Entry {
    position::Entry {
        name: format!("prefill_r{ratio}_all"),
        query: PositionIndexerQuery {
            heads: 32,
            head_dim: 128,
            compression_ratio: ratio,
            tail_rope_dims: 64,
            key_norm_eps_bits: 1e-6_f32.to_bits(),
            max_top_k: 512,
            page_size: 64 / ratio,
            candidates: IndexerCandidates::All,
            capacity: 8192,
        },
    }
}

#[test]
fn tiled_rows_preserve_arguments_and_local_selector_keeps_merge() {
    for ratio in [1, 2] {
        let e = entry(ratio);
        let coarse =
            position_mma_guard_fast::score(position_mma_guard::dense_specs(&e).unwrap().remove(1));
        let candidate = tiled::score(&e).unwrap();
        assert_eq!(candidate.params, coarse.params);
        // The tiled entry owns per-row packing fallback. Retaining the old
        // blockIdx.y wrapper would inspect only one of the four query rows.
        assert!(
            candidate
                .body
                .starts_with("chop::position_mma_prefill::scores")
        );
        let originals = position_live::dense_specs(&e, false).unwrap();
        let selector = cached::select(&e).unwrap();
        assert_eq!(selector.params, originals[1].params);
        assert_eq!(
            selector
                .body
                .replace("position_select_cached::", "position_live::"),
            originals[1].body
        );
        assert!(cached::select_symbol(&originals[2].symbol).is_err());
        for rows in [1, 2, 3, 4, 5, 63, 64, 65, 448, 1984, 2496, 8191, 8192] {
            let mut plan = e.score_plan(rows, 131072 / ratio).unwrap();
            plan.grid[0] = position_live::SCORE_CTAS;
            let mut call = Launch {
                symbol: coarse.symbol.clone(),
                rows,
                plan,
                arguments: ArgumentList(vec![]),
                collective: false,
                persistent: false,
            };
            tiled::remap(&e, &mut call).unwrap();
            assert_eq!(call.plan.grid, [192, rows.div_ceil(4), 1]);
            assert_eq!(call.symbol, candidate.symbol);
            assert!(tiled::remap(&e, &mut call).is_err());
        }
        let mut bad = e.clone();
        bad.query.candidates = IndexerCandidates::Mask;
        assert!(tiled::score(&bad).is_err());
        bad.query.candidates = IndexerCandidates::All;
        bad.query.head_dim = 256;
        assert!(tiled::score(&bad).is_err());
    }
}
