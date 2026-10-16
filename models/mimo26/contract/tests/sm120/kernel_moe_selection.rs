//! The grouped expert entries MiMo V2.6 Flash serves never launch the
//! packed engine's split-K reduction.

use kernel_core::precision::ExecutionPhase;
use kernels_sm120::moe::{InputPath, Projection};
use model_mimo26_contract::Tp;
use model_mimo26_contract::serving::{PREFILL_TOKENS, VERIFY_TOKENS};
use model_mimo26_contract::sm120::workspace::moe::{Selection, entry};

/// Every entry `workspace::moe::entry` selects, for every rank count,
/// projection, phase, token bucket and verify selection, reads gathered
/// BF16 rows and has no reduction launch.
#[test]
fn served_expert_entries_have_no_split_k_reduction() {
    let buckets = || {
        (0..)
            .map(|shift| 1_u32 << shift)
            .take_while(|&b| b <= VERIFY_TOKENS)
    };
    let mut selected = 0;
    for tp in model_mimo26_contract::geometry::GEOMETRIES
        .iter()
        .flat_map(|g| Tp::SUPPORTED.map(|tp| Tp::of_geometry(g, tp.ranks()).unwrap()))
    {
        for projection in [Projection::GateUp, Projection::Down] {
            for exact in [false, true] {
                let selection = Selection { exact };
                let steps = buckets()
                    .map(|b| (ExecutionPhase::Decode, b))
                    .chain(buckets().map(|b| (ExecutionPhase::Verify, b)))
                    .chain([(ExecutionPhase::Prefill, PREFILL_TOKENS)]);
                for (phase, capacity) in steps {
                    let v = entry(tp, projection, phase, capacity, selection)
                        .unwrap_or_else(|e| panic!("{tp:?} {phase:?} {capacity}: {}", e.0));
                    assert_eq!(v.path, InputPath::A16Gather, "{}", v.symbol);
                    assert!(
                        v.reduction_symbol().is_none(),
                        "{} launches a split-K reduction",
                        v.symbol
                    );
                    selected += 1;
                }
            }
        }
    }
    assert!(selected > 0);
}
