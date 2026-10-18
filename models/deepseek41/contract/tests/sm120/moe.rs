//! Model contracts prepared through the registered backend at every topology.
#[path = "../../../bundle/src/sm120/moe.rs"]
#[allow(dead_code)]
mod registration;

use kernel_core::backend::{Backend, Need};
use kernel_core::contract::MoeQuery;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::properties::Invariance;
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::GEOMETRIES, moe, serving};

#[test]
fn kernel_moe_prepares_all_geometries_rank_counts_and_phases() {
    if std::env::var_os("CHOPINFER_DEEPSEEK41_GEOMETRY_DIRS").is_some() {
        assert!(
            GEOMETRIES.iter().any(|g| !g.base),
            "extra geometry was not compiled"
        );
    }
    let mut prepared = 0;
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            let backend = registration::register(Sm120::new(), g, ranks).unwrap();
            for (phase, tokens) in [
                (ExecutionPhase::Decode, serving::DECODE_BATCH),
                (
                    ExecutionPhase::Verify,
                    serving::DECODE_BATCH * serving::MAX_VERIFY_ROWS,
                ),
                (ExecutionPhase::Prefill, serving::PREFILL_TOKENS),
            ] {
                let need = Need::rows(tokens).with_exact(Invariance::any_rows(tokens));
                let usage = OperationUse {
                    phase,
                    capacity: tokens,
                    graph: true,
                };
                backend
                    .block_quantize(&moe::quantizer(g.shape.hidden), need)
                    .unwrap();
                for set in moe::expert_sets(g) {
                    let router = moe::router(g, set);
                    backend.moe_row_bias_router(&router, need).unwrap();
                    let mut tile = 0;
                    for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                        let query = moe::projection(g, ranks, set, kind, tokens).unwrap();
                        let op = backend.chained_mlp_projection(&query, usage, need).unwrap();
                        if tile == 0 {
                            tile = op.route_tile();
                        }
                        assert_eq!(op.route_tile(), tile);
                        assert_eq!(
                            op.info().symbols.len(),
                            if kind == GatedMlpProjection::GateUp {
                                2
                            } else {
                                1
                            }
                        );
                        assert_eq!(
                            op.info().bytes_of("weights.codes"),
                            Some(
                                u64::from(set.count)
                                    * u64::from(query.projection.gemm.n)
                                    * u64::from(query.k.stored)
                                    / 2
                            )
                        );
                        if kind == GatedMlpProjection::Down && query.k.stored == 576 {
                            assert_eq!(query.k.loop_k, 640);
                        }
                        prepared += 1;
                    }
                    let routes = MoeQuery::LocalRoutes {
                        experts: set.count,
                        local_experts: set.count,
                        top_k: set.top_k,
                        capacity: tokens,
                    };
                    // Experts are tensor-parallel: every rank routes all experts.
                    backend
                        .moe_row_bias_route(&router, &routes, 0, tile, usage, Need::rows(tokens))
                        .unwrap();
                }
            }
        }
    }
    let expected: usize = GEOMETRIES
        .iter()
        .map(|g| moe::expert_sets(g).len() * SUPPORTED_RANKS.len() * 3 * 2)
        .sum();
    assert_eq!(prepared, expected);
    assert!(prepared > 0);
}

#[test]
fn kernel_moe_only_adds_its_model_owned_lock_family() {
    let specs = registration::cubins(GEOMETRIES).unwrap();
    assert_eq!(specs.len(), 3 * GEOMETRIES.len());
    for (family, spec) in specs {
        assert_eq!(family, registration::FAMILY);
        assert!(!spec.entries.is_empty());
    }
}
