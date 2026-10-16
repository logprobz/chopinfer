//! The chained MoE engine's decoded tiles and tile pairs: which phases
//! and layouts run them, and that each is admitted on its MoE lock's
//! locked resources.

use kernel_core::gated_mlp::GatedMlpProjection;
use kernels_sm120::backend::TILE_PAIR_MIN_ROWS;
use kernels_sm120::moe::chained::{self, LocalRoute, Projection, Schedule};

use super::engines::Engines;
use crate::program::Phase;

/// The bundle's chained MoE lock of the dense MLP and shared expert, whose
/// records carry each entry's registers and static shared memory as ptxas
/// reported them.
const MOE_LOCK: &str = include_str!("../../kernels.lock.d/moe_chained.lock");
/// The split routed experts' lock.
const SPLIT_LOCK: &str = include_str!("../../kernels.lock.d/moe_chained_split.lock");

/// Admits the tile-pair projections in `moe`: each reads the prefill
/// route's padding (`pad`) as its row tile, runs eight warps per CTA with
/// static shared memory only, and its resources in `lock` keep at least one
/// CTA resident per SM of the target (`kernels_sm120::device::TARGET`;
/// the served gate/up's 252 registers over 256 threads take the whole
/// register file, so exactly one). The CTAs do not wait on each other, so
/// one resident CTA per SM is all a launch needs.
fn admit_tile_pairs(moe: &[Projection], pad: u32, lock: &str) -> Result<(), String> {
    use kernels_sm120::backend::locked_resources;
    use kernels_sm120::device::TARGET;
    let mut resources = None;
    for pairs in moe
        .iter()
        .filter(|p| matches!(p.schedule, Schedule::TilePairs { .. }))
    {
        let plan = pairs
            .plan(pairs.tokens_capacity)
            .map_err(|r| format!("{}: {r}", pairs.symbol))?;
        if pairs.row_tile != pad
            || plan.block != [chained::TILE_PAIR_THREADS, 1, 1]
            || plan.dynamic_shared_bytes != 0
        {
            return Err(format!(
                "{}: tile pairs need the prefill route's {pad}-row padding, {} threads and no dynamic shared memory",
                pairs.symbol,
                chained::TILE_PAIR_THREADS
            ));
        }
        if resources.is_none() {
            resources = Some(locked_resources(lock).map_err(|r| r.to_string())?);
        }
        let locked = resources
            .as_ref()
            .and_then(|r| r.get(&pairs.symbol))
            .ok_or_else(|| format!("{}: not in the MoE lock", pairs.symbol))?;
        if TARGET.ctas_per_sm(*locked, plan.block[0], plan.dynamic_shared_bytes) == 0 {
            return Err(format!(
                "{}: {} registers and {} static shared bytes over {} threads keep no CTA resident",
                pairs.symbol, locked.registers, locked.shared_static_bytes, plan.block[0]
            ));
        }
    }
    Ok(())
}

/// Admits the decoded-tile projections in `moe` on their resources in
/// `lock`: the static shared bytes ptxas reported are those
/// `chained::decoded_tile_static_bytes` bounds (so the kernel library's
/// refusal of longer K holds for the compiled entries), they and the stages'
/// dynamic bytes fit one SM120 block, and at least one CTA stays resident
/// per SM.
fn admit_decoded_tiles(moe: &[Projection], lock: &str) -> Result<(), String> {
    use kernels_sm120::backend::locked_resources;
    use kernels_sm120::device::TARGET;
    let mut resources = None;
    for decoded in moe.iter().filter(|p| p.schedule == Schedule::DecodedTiles) {
        let plan = decoded
            .plan(decoded.tokens_capacity)
            .map_err(|r| format!("{}: {r}", decoded.symbol))?;
        if resources.is_none() {
            resources = Some(locked_resources(lock).map_err(|r| r.to_string())?);
        }
        let locked = resources
            .as_ref()
            .and_then(|r| r.get(&decoded.symbol))
            .ok_or_else(|| format!("{}: not in the MoE lock", decoded.symbol))?;
        let bound = chained::decoded_tile_static_bytes(decoded.query.gemm.k);
        if locked.shared_static_bytes != bound
            || bound + plan.dynamic_shared_bytes > kernels_sm120::backend::SHARED_MEMORY_BYTES
            || TARGET.ctas_per_sm(*locked, plan.block[0], plan.dynamic_shared_bytes) == 0
        {
            return Err(format!(
                "{}: {} static shared bytes (bounded at {bound}) and {} dynamic over {} threads do not fit an SM120 block",
                decoded.symbol,
                locked.shared_static_bytes,
                plan.dynamic_shared_bytes,
                plan.block[0]
            ));
        }
    }
    Ok(())
}

impl Engines {
    /// The route serving `rows` tokens: the one-CTA decode route up to its
    /// capacity, else the prefill route.
    fn route(&self, rows: u32) -> LocalRoute {
        if rows <= self.routes[0].capacity {
            self.routes[0]
        } else {
            self.routes[1]
        }
    }

    /// The compiled chained projection a step of `phase` runs: the one at
    /// its route's padding, in the tile-pair form when it has one and the
    /// step has at least `TILE_PAIR_MIN_ROWS` rows, else at a prefill step
    /// in its decoded-tile form when it has one. Decode and verify steps
    /// never run decoded tiles.
    fn projection(
        &self,
        phase: Phase,
        grouped: bool,
        gate_up: bool,
        n: u32,
        weights: chained::Weights,
    ) -> Result<&Projection, String> {
        let rows = phase.rows();
        let row_tile = if self.route(rows) == self.routes[0] {
            self.routes[0].pad
        } else {
            self.routes[1].pad
        };
        let wanted = if gate_up {
            GatedMlpProjection::GateUp
        } else {
            GatedMlpProjection::Down
        };
        // A projection with a tile-pair form runs it from
        // `TILE_PAIR_MIN_ROWS` rows, its other form below.
        let tile_pairs = rows >= TILE_PAIR_MIN_ROWS;
        let matching = || {
            self.moe.iter().filter(move |p| {
                p.row_tile == row_tile
                    && p.weights() == weights
                    && p.query.projection == wanted
                    && p.query.gemm.grouped.is_some() == grouped
                    && p.query.gemm.n == n
            })
        };
        let is_pair = |p: &&Projection| matches!(p.schedule, Schedule::TilePairs { .. });
        let is_decoded = |p: &&Projection| p.schedule == Schedule::DecodedTiles;
        // A projection with a decoded-tile form runs it at every prefill
        // step of its row tile, and its other form at decode and verify.
        let prefill = matches!(phase, Phase::Prefill { .. });
        tile_pairs
            .then(|| matching().find(is_pair))
            .flatten()
            .or_else(|| prefill.then(|| matching().find(is_decoded)).flatten())
            .or_else(|| matching().find(|p| !is_pair(p) && !is_decoded(p)))
            .ok_or_else(|| {
                format!("no {weights:?} chained projection of {n} columns at row tile {row_tile}")
            })
    }
}

mod decoded_tile_tests {
    use super::*;
    use chained::Weights;
    use model_glm5next_contract::contract::Shape;
    use model_glm5next_contract::quantization::{FP8_BLOCK128, LAYOUTS};

    /// Every phase the package admits: each prefill row count, each decode
    /// bucket and each verify bucket of up to `MAX_VERIFY_ROWS` rows per
    /// item.
    fn admitted_phases() -> impl Iterator<Item = Phase> {
        use model_glm5next_contract::contract::{DECODE_BATCH, MAX_VERIFY_ROWS, PREFILL_TOKENS};
        (1..=PREFILL_TOKENS)
            .map(|rows| Phase::Prefill { rows })
            .chain((1..=DECODE_BATCH).map(Phase::Decode))
            .chain(
                (1..=DECODE_BATCH * MAX_VERIFY_ROWS).map(|rows| Phase::Verify {
                    rows,
                    items: rows.div_ceil(MAX_VERIFY_ROWS),
                }),
            )
    }

    /// The dense MLP's and the shared expert's FP8 gate/up run as decoded
    /// tiles at every prefill step of more than 32 rows; every decode and
    /// verify bucket (wide verify buckets share the prefill row tile) and
    /// shorter prefill steps keep the block warps or columns of today.
    /// Module loading gives the decoded tiles their stages' dynamic shared
    /// memory and every other chained projection none. Other formats stay
    /// off decoded tiles, and no layout's decode or verify step selects a
    /// decoded-tile projection.
    #[test]
    fn dense_fp8_prefill_gate_up_runs_decoded_tiles() {
        let engines = Engines::shared(Shape::TP4, &FP8_BLOCK128).unwrap();
        let shared = engines.shared_bytes();
        let mut verify_wide = 0;
        for n in [6144, 1024] {
            let decoded = format!(
                "chop_moe_chain_gateup_fp8block_dense_n{n}_k4096_r64_clamp4120000041200000_dt_v1"
            );
            let columns = decoded.replace("_dt_v1", "_v1");
            for phase in admitted_phases() {
                let projection = engines
                    .projection(phase, false, true, n, Weights::Fp8Block128)
                    .unwrap();
                let rows = phase.rows();
                if rows <= 32 {
                    assert_eq!(projection.schedule, Schedule::BlockWarps, "{phase:?}");
                } else if matches!(phase, Phase::Prefill { .. }) {
                    assert_eq!(projection.schedule, Schedule::DecodedTiles, "{phase:?}");
                    assert_eq!(projection.symbol, decoded, "{phase:?}");
                    assert_eq!(
                        shared[&projection.symbol],
                        chained::DECODED_TILE_SHARED_BYTES
                    );
                } else {
                    assert_eq!(projection.schedule, Schedule::Columns, "{phase:?}");
                    assert_eq!(projection.symbol, columns, "{phase:?}");
                    verify_wide += 1;
                }
            }
        }
        // Verify buckets of 33 to 192 rows, for both projections.
        assert_eq!(verify_wide, 2 * (192 - 32));
        for projection in &engines.moe {
            let decoded = projection.schedule == Schedule::DecodedTiles;
            assert_eq!(
                shared[&projection.symbol] != 0,
                decoded,
                "{}",
                projection.symbol
            );
        }
        for layout in LAYOUTS {
            let engines = Engines::shared(Shape::TP4, layout).unwrap();
            let decoded: Vec<_> = engines
                .moe
                .iter()
                .filter(|p| p.schedule == Schedule::DecodedTiles)
                .collect();
            assert_eq!(decoded.len(), 2, "{}", layout.id);
            assert!(
                decoded
                    .iter()
                    .all(|p| p.weights() == Weights::Fp8Block128 && p.gate_up()),
                "{}",
                layout.id
            );
            for p in &engines.moe {
                let (grouped, n) = (p.query.gemm.grouped.is_some(), p.query.gemm.n);
                for phase in admitted_phases().filter(|p| !matches!(p, Phase::Prefill { .. })) {
                    let selected = engines
                        .projection(phase, grouped, p.gate_up(), n, p.weights())
                        .unwrap();
                    assert_ne!(
                        selected.schedule,
                        Schedule::DecodedTiles,
                        "{} {phase:?}",
                        layout.id
                    );
                }
            }
        }
    }

    /// The served decoded tiles' locked 1,408 static shared bytes are the
    /// kernel library's bound at K = 4096 and fit an SM120 block beside
    /// their stages, admitted; locked bytes past the bound are refused, and
    /// so is a symbol missing from the lock.
    #[test]
    fn decoded_tiles_are_admitted_on_their_locked_resources() {
        let engines = Engines::new(Shape::TP4).unwrap();
        let resources = kernels_sm120::backend::locked_resources(MOE_LOCK).unwrap();
        let decoded: Vec<_> = engines
            .moe
            .iter()
            .filter(|p| p.schedule == Schedule::DecodedTiles)
            .collect();
        assert_eq!(decoded.len(), 2);
        for p in &decoded {
            assert_eq!(resources[&p.symbol].shared_static_bytes, 1408);
        }
        admit_decoded_tiles(&engines.moe, MOE_LOCK).unwrap();
        let mut lock: serde_json::Value = serde_json::from_str(MOE_LOCK).unwrap();
        for record in lock.as_array_mut().unwrap() {
            for entry in record["resources"].as_array_mut().unwrap() {
                if entry["symbol"] == decoded[0].symbol.as_str() {
                    entry["shared_static_bytes"] = 2176.into();
                }
            }
        }
        let error = admit_decoded_tiles(&engines.moe, &lock.to_string()).unwrap_err();
        assert!(error.contains("do not fit an SM120 block"), "{error}");
        let no_entry = MOE_LOCK.replace(&decoded[1].symbol, "chop_moe_chain_unrelated_v1");
        let error = admit_decoded_tiles(&engines.moe, &no_entry).unwrap_err();
        assert!(error.contains("not in the MoE lock"), "{error}");
    }
}

mod tile_pair_tests {
    use super::*;
    use chained::Weights;
    use model_glm5next_contract::contract::Shape;
    use model_glm5next_contract::quantization::LAYOUTS;

    const PAIRS: [&str; 2] = [
        "chop_moe_chain_gateup_fp8block_e288k8_n1024_k4096_r64_clamp4120000041200000_tp2_v1",
        "chop_moe_chain_down_fp8block_e288k8_n4096_k512_r64_tp0_v1",
    ];

    /// The split routed FP8 gate/up and down run as tile pairs from 2,048
    /// rows and in columns below; the dense and shared projections and the
    /// other formats never do (the dense and shared FP8 prefill gate/up run
    /// as decoded tiles).
    #[test]
    fn routed_fp8_projections_run_tile_pairs_from_their_threshold() {
        let engines = Engines::new(Shape::TP4).unwrap();
        let symbol = |rows, grouped, gate_up, n, weights| {
            engines
                .projection(Phase::Prefill { rows }, grouped, gate_up, n, weights)
                .unwrap()
                .symbol
                .clone()
        };
        assert_eq!(TILE_PAIR_MIN_ROWS, 2048);
        for (pairs, gate_up, n) in [(PAIRS[0], true, 1024), (PAIRS[1], false, 4096)] {
            let columns = pairs.replace("_tp2_v1", "_v1").replace("_tp0_v1", "_v1");
            for rows in [2048, 2049, 4096, 8192] {
                assert_eq!(symbol(rows, true, gate_up, n, Weights::Fp8Block128), pairs);
            }
            for rows in [33, 64, 192, 2047] {
                assert_eq!(
                    symbol(rows, true, gate_up, n, Weights::Fp8Block128),
                    columns
                );
            }
        }
        assert!(
            symbol(32, true, true, 1024, Weights::Fp8Block128)
                .ends_with("_r16_clamp4120000041200000_ws_v1")
        );
        for rows in [2047, 2048, 8192] {
            for (grouped, gate_up, n, weights) in [
                (false, false, 4096, Weights::Fp8Block128),
                (true, true, 1024, Weights::Nvfp4),
                (true, false, 4096, Weights::Nvfp4),
                (true, true, 1024, Weights::Mxfp8),
                (true, true, 1024, Weights::Bf16),
            ] {
                let projection = engines
                    .projection(Phase::Prefill { rows }, grouped, gate_up, n, weights)
                    .unwrap();
                assert_eq!(
                    projection.schedule,
                    Schedule::Columns,
                    "{}",
                    projection.symbol
                );
            }
        }
        // Every layout lowers (and admits) the same tile pairs.
        for layout in LAYOUTS {
            let engines = Engines::for_layout(Shape::TP4, layout).unwrap();
            let pairs: Vec<_> = engines
                .moe
                .iter()
                .filter(|p| matches!(p.schedule, Schedule::TilePairs { .. }))
                .map(|p| p.symbol.as_str())
                .collect();
            assert_eq!(pairs, PAIRS, "{}", layout.id);
        }
    }

    /// The served tile pairs' locked resources keep one CTA per SM
    /// resident, admitted. Resources that keep none, a symbol missing from
    /// the lock, or a tile pair off the prefill padding are refused.
    #[test]
    fn tile_pairs_are_admitted_on_their_locked_resources() {
        let engines = Engines::new(Shape::TP4).unwrap();
        let resources = kernels_sm120::backend::locked_resources(SPLIT_LOCK).unwrap();
        for pairs in PAIRS {
            let locked = resources[pairs];
            assert!(
                kernels_sm120::device::TARGET.ctas_per_sm(locked, 256, 0) >= 1,
                "{pairs}"
            );
        }
        admit_tile_pairs(&engines.moe, 64, SPLIT_LOCK).unwrap();

        let mut lock: serde_json::Value = serde_json::from_str(SPLIT_LOCK).unwrap();
        for record in lock.as_array_mut().unwrap() {
            for entry in record["resources"].as_array_mut().unwrap() {
                if entry["symbol"] == PAIRS[0] {
                    entry["shared_static_bytes"] = 101_400.into();
                }
            }
        }
        let error = admit_tile_pairs(&engines.moe, 64, &lock.to_string()).unwrap_err();
        assert!(error.contains("keep no CTA resident"), "{error}");

        let no_entry = SPLIT_LOCK.replace(PAIRS[1], "chop_moe_chain_unrelated_v1");
        let error = admit_tile_pairs(&engines.moe, 64, &no_entry).unwrap_err();
        assert!(error.contains("not in the MoE lock"), "{error}");
        let error = admit_tile_pairs(&engines.moe, 16, SPLIT_LOCK).unwrap_err();
        assert!(error.contains("row padding"), "{error}");
    }
}
