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
                        let streamed = phase != ExecutionPhase::Prefill
                            && tokens <= 64
                            && registration::streamed_qualified(g, ranks);
                        assert_eq!(
                            op.info().symbols.len(),
                            if kind == GatedMlpProjection::GateUp {
                                if streamed { 5 } else { 2 }
                            } else {
                                if streamed { 4 } else { 1 }
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

#[test]
fn measured_active_crossover_keeps_unmeasured_buckets_on_the_original_entries() {
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            let backend = registration::register(Sm120::new(), g, ranks).unwrap();
            for phase in [
                ExecutionPhase::Decode,
                ExecutionPhase::Verify,
                ExecutionPhase::Prefill,
            ] {
                for tokens in [
                    1,
                    2,
                    4,
                    5,
                    6,
                    8,
                    32,
                    48,
                    64,
                    65,
                    serving::DECODE_BATCH * serving::MAX_VERIFY_ROWS,
                ] {
                    for set in moe::expert_sets(g) {
                        for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                            let q = moe::projection(g, ranks, set, kind, tokens).unwrap();
                            let usage = OperationUse {
                                phase,
                                capacity: tokens,
                                graph: true,
                            };
                            let op = backend
                                .chained_mlp_projection(&q, usage, Need::rows(tokens))
                                .unwrap();
                            let compact = phase == ExecutionPhase::Decode
                                && tokens == 1
                                && set.count == 128
                                && set.top_k == 3
                                && registration::streamed_qualified(g, ranks);
                            let streamed = !compact
                                && phase != ExecutionPhase::Prefill
                                && tokens <= 64
                                && registration::streamed_qualified(g, ranks);
                            let active = !streamed
                                && phase != ExecutionPhase::Prefill
                                && tokens == 1
                                && ranks == 4
                                && g.id == model_deepseek41_contract::geometry::FLASH.id
                                && set.count == g.shape.moe.experts
                                && kind == GatedMlpProjection::GateUp;
                            assert_eq!(op.info().symbols[0].0.ends_with("_active_bw4_v1"), active);
                            assert_eq!(
                                op.info().bytes_of("routing.routes.active_experts"),
                                (active || streamed).then_some(u64::from(set.count + 1) * 4)
                            );
                            assert_eq!(
                                op.info().symbols[0].0.ends_with("_stream_pack_v1"),
                                streamed
                            );
                            if streamed {
                                assert_eq!(op.route_tile(), 64);
                            } else if active || compact {
                                assert_eq!(op.route_tile(), 16);
                            }
                            let routes = MoeQuery::LocalRoutes {
                                experts: set.count,
                                local_experts: set.count,
                                top_k: set.top_k,
                                capacity: tokens,
                            };
                            backend
                                .moe_row_bias_route(
                                    &moe::router(g, set),
                                    &routes,
                                    0,
                                    op.route_tile(),
                                    usage,
                                    Need::rows(tokens),
                                )
                                .unwrap();
                        }
                    }
                }
            }
            for entry in registration::active_projections(g, ranks).unwrap() {
                for rows in [1, 8, 32] {
                    let grid = entry.plan(rows).unwrap().grid;
                    assert!(grid[0] > 0 && grid[0] <= 376);
                    assert_eq!(&grid[1..], &[1, 1]);
                }
            }
        }
    }
}
