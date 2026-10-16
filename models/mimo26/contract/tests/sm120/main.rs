//! The SM120 backend chooses MiMo's grouped expert entries as MiMo's own
//! selector does (`workspace::moe::entry`) under both exact verify
//! settings, with MiMo registering its exact entries only when its policy
//! runs them.

use kernel_core::backend::{Backend, Need};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::backend::{Sm120, grouped_contract};
use kernels_sm120::moe::Projection;
use model_mimo26_contract::Tp;
use model_mimo26_contract::geometry::GEOMETRIES;
use model_mimo26_contract::sm120::workspace::moe::{Selection, entry, mimo_variants};

#[test]
fn the_backend_chooses_the_entries_mimos_selector_chooses() {
    let mut exact_runs = 0;
    for geometry in GEOMETRIES.iter() {
        for ranks in Tp::SUPPORTED {
            let tp = Tp::of_geometry(geometry, ranks.ranks()).unwrap();
            for exact in [false, true] {
                let selection = Selection { exact };
                let backend = mimo_variants(tp)
                    .iter()
                    .filter(|variant| exact || !variant.exact)
                    .cloned()
                    .fold(Sm120::new(), Sm120::with_experts);
                for projection in [Projection::GateUp, Projection::Down] {
                    for phase in [
                        ExecutionPhase::Prefill,
                        ExecutionPhase::Decode,
                        ExecutionPhase::Verify,
                    ] {
                        for capacity in [1, 2, 4, 8, 16, 32, 64, 512, 8192] {
                            let expected = entry(tp, projection, phase, capacity, selection)
                                .unwrap_or_else(|e| panic!("{e:?}"));
                            exact_runs += usize::from(expected.exact);
                            let usage = OperationUse {
                                phase,
                                capacity,
                                graph: false,
                            };
                            let prepared = backend
                                .gated_mlp_projection(
                                    &grouped_contract(expected),
                                    usage,
                                    Need::rows(capacity),
                                )
                                .unwrap_or_else(|refused| {
                                    panic!("{} {phase:?} {capacity}: {refused}", expected.symbol)
                                });
                            assert_eq!(
                                prepared.info().symbols[0].0,
                                expected.symbol,
                                "{tp:?} exact {exact} {projection:?} {phase:?} at {capacity}"
                            );
                        }
                    }
                }
            }
        }
    }
    // The exact setting runs exact entries somewhere, so both settings are
    // exercised.
    assert!(exact_runs > 0);
}
