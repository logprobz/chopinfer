//! The served routed experts of [`TP4_288_TOP8`] split over the four ranks
//! on their intermediate (tensor parallel): every rank holds a quarter of
//! every expert, so all 288 experts are local, each expert's gate/up writes
//! 512 intermediate columns (N 1,024 over K 4,096) and its down reads K 512
//! (N 4,096) ([`SHAPES`]). Every rank runs every route, so the ranks' work
//! is the same at every row count.
//!
//! The split changes no kernel: the engine's projections, routes and route
//! sums take any expert count and intermediate, and their declarations
//! ([`super`]'s module docs) hold at these shapes. A gate/up output element
//! is the unsplit expert's (the split only chooses which rank computes its
//! column). A down output is the rank's partial over its 512-K quarter:
//! four 128-K chains folded in K order, then the route sum's top-k adds of
//! the rank's eight routes; the ranks' partials meet in the all-reduce.
//!
//! The dense MLP and the shared expert keep [`super::catalog`]'s
//! projections. [`catalog`] holds the routed projections in every routed
//! format the served checkpoints use (FP8 128x128, NVFP4 and MXFP8 from
//! [`TP4_288_TOP8`]'s expert formats, and BF16 for BF16 multi-token
//! prediction experts) at the decode and prefill capacities; [`forms`] the
//! schedules a model registers for them. [`accumulation_projections`] are
//! kernel-lab's grouped downs at K = 4096 over the same experts, which
//! nothing serves.

use super::{
    GatedMlpProjection, LocalRoute, Projection, Refusal, RoutedSum, RoutedSumOutput, ServedShapes,
    TP4_288_TOP8, Weights, projection_cubin_specs, route_cubin_spec, routed_sum_add_cubin_spec,
};
use kernel_build::CubinSpec;
use kernel_core::contract::WeightFormat;

/// One rank's shapes when each of [`TP4_288_TOP8`]'s routed experts is split
/// over its four ranks.
pub const SHAPES: ServedShapes = ServedShapes {
    local_experts: 288,
    expert_intermediate: 512,
    ..TP4_288_TOP8
};

/// The cubin name prefix of the split instances.
pub const PREFIX: &str = "chop_moe_chain_e288l288k8";

/// Row tiles an expert needs before the routed FP8 prefill gate/up pairs
/// them under one weight tile (`Schedule::TilePairs`), as the unsplit
/// gate/up.
pub const GATE_UP_MIN_PAIR_TILES: u32 = 2;

/// Row tiles an expert needs before the routed FP8 prefill down pairs them
/// under one weight tile. At K 512 an activation tile (64 rows of 512 BF16)
/// is twice the bytes of its 64-column FP8 weight tile, so the down pairs
/// column tiles under one activation tile at every expert (0 never pairs
/// rows). Timed on one GPU in two runs (kernel-lab `split_expert_latency`,
/// us per layer): on captured serving routes 1,174-1,216, 2,033-2,048 and
/// 3,736-3,805 at 2,048, 4,096 and 8,192 tokens, against the columns'
/// 1,240-1,254, 2,060-2,115 and 3,657-3,798 and at least 1,257, 2,058 and
/// 3,795 pairing at two or four tiles; on uniform routes the columns lead
/// by 8% (900 against 973 at 2,048 tokens). The choice is close: pairing
/// leads by 3 to 5% at 2,048 tokens, the columns by up to 2% at 8,192.
pub const DOWN_MIN_PAIR_TILES: u32 = 0;

/// The most tokens the routed FP8 decode down runs in block warps; above,
/// columns. Timed on one GPU (us per layer): block warps 18.3 against the
/// columns' 33.3 at one token, about even at four (35.4 and 46.7 against
/// 39.1 and 44.8), behind from eight (83.0, 129.3 and 192.3 against 76.7,
/// 114.7 and 160.3 at 8, 16 and 32).
pub const DOWN_BLOCK_WARP_TOKENS: u32 = 4;

/// The routed experts' formats: [`TP4_288_TOP8`]'s, then BF16.
fn formats() -> Vec<WeightFormat> {
    let mut formats = SHAPES.expert_weights.to_vec();
    formats.push(WeightFormat::Bf16);
    formats
}

/// Every routed projection of [`SHAPES`] in the column schedule: per format,
/// gate/up then down, each at the decode (row tile 16) and prefill (row tile
/// 64) capacity.
pub fn catalog() -> Vec<Projection> {
    let mut out = Vec::new();
    for weight in formats() {
        for projection in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
            for (row_tile, tokens) in [(16, SHAPES.decode_tokens), (64, SHAPES.prefill_tokens)] {
                let query =
                    SHAPES.query(weight, projection, true, SHAPES.expert_intermediate, tokens);
                out.push(
                    Projection::new(query, row_tile, tokens)
                        .expect("the split routed projections are supported"),
                );
            }
        }
    }
    out
}

/// The schedules a model registers for [`catalog`]'s projections, each
/// bitwise the column schedule's outputs:
///
/// - decode (16-row tiles): the FP8 gate/up in warp streams; the FP8 down in
///   block warps to [`DOWN_BLOCK_WARP_TOKENS`] tokens (warp streams take K
///   of two or more 512-K rounds) and in columns above; NVFP4 and BF16 in
///   block warps; MXFP8 in columns;
/// - prefill (64-row tiles): every projection in columns, which steps below
///   the backend's tile-pair rows run, and the FP8 gate/up and down also in
///   tile pairs ([`GATE_UP_MIN_PAIR_TILES`], [`DOWN_MIN_PAIR_TILES`]).
pub fn forms() -> Vec<Projection> {
    let mut out = Vec::new();
    for projection in catalog() {
        if projection.row_tile == 16 {
            if projection.weights() == Weights::Fp8Block128 && !projection.gate_up() {
                let mut warps = projection
                    .block_warps()
                    .expect("block warps take the split decode down");
                warps.tokens_capacity = DOWN_BLOCK_WARP_TOKENS;
                out.extend([warps, projection]);
                continue;
            }
            let decode = match projection.weights() {
                Weights::Fp8Block128 => projection.warp_streams(),
                Weights::Nvfp4 | Weights::Bf16 => projection.block_warps(),
                Weights::Mxfp8 => Ok(projection.clone()),
            };
            out.push(decode.expect("the split decode schedules take their projections"));
            continue;
        }
        if projection.weights() == Weights::Fp8Block128 {
            let pairs = match projection.gate_up() {
                true => GATE_UP_MIN_PAIR_TILES,
                false => DOWN_MIN_PAIR_TILES,
            };
            out.push(
                projection
                    .tile_pairs(pairs)
                    .expect("tile pairs take the split FP8 prefill projections"),
            );
        }
        out.push(projection);
    }
    out
}

/// The decode and prefill routes over every expert. A rank keeps the routes
/// of experts `[rank * 288, (rank + 1) * 288)`, so every rank launches them
/// as rank 0 (the permutation of all routes by expert).
pub fn routes() -> [LocalRoute; 2] {
    SHAPES.routes()
}

/// The route sum of every expert's routes, at the prefill capacity, which
/// also serves decode; its F32 form is [`RoutedSum::with_output`].
pub fn sums() -> [RoutedSum; 1] {
    SHAPES.sums()
}

/// Grouped down projections over every expert at K = 4096 (the shapes'
/// longest K) with F32 routed outputs, whose raw sums kernel-lab's 128-K
/// accumulation gate reads; nothing serves them: FP8 128x128 at the
/// prefill row tile (64 tokens) and in [`Schedule::BlockWarps`] at the
/// decode row tile and tokens, then NVFP4, MXFP8 and BF16 at the prefill
/// row tile.
///
/// [`Schedule::BlockWarps`]: super::Schedule::BlockWarps
pub fn accumulation_projections() -> Vec<Projection> {
    let down = |weight, row_tile, tokens| {
        let query = SHAPES.query(
            weight,
            GatedMlpProjection::Down,
            true,
            SHAPES.hidden,
            tokens,
        );
        Projection::new(query, row_tile, tokens).expect("K = 4096 grouped downs are supported")
    };
    let fp8 = SHAPES.expert_weights[0];
    let mut out = vec![
        down(fp8, 64, 64),
        down(fp8, 16, SHAPES.decode_tokens)
            .block_warps()
            .expect("block warps take the K = 4096 decode down"),
    ];
    for weight in [WeightFormat::Nvfp4, WeightFormat::Mxfp8, WeightFormat::Bf16] {
        out.push(down(weight, 64, 64));
    }
    out
}

/// The split cubins: [`forms`] per format and schedule, the routes with the
/// route sum in both output forms (`{PREFIX}_route`), and the F32 sum that
/// adds a row (`{PREFIX}_sum_add`).
pub fn cubin_specs() -> Result<Vec<CubinSpec>, Refusal> {
    let mut specs = projection_cubin_specs(PREFIX, &forms());
    let [sum] = sums();
    specs.push(route_cubin_spec(
        PREFIX,
        &routes(),
        &[sum, sum.with_output(RoutedSumOutput::F32)],
    )?);
    specs.push(routed_sum_add_cubin_spec(
        format!("{PREFIX}_sum_add"),
        PREFIX,
        &[sum.with_output(RoutedSumOutput::F32)],
    )?);
    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::super::Schedule;
    use super::*;
    use kernel_core::contract::RoutedCapacity;

    #[test]
    fn every_expert_is_local_and_a_quarter_of_its_intermediate() {
        assert_eq!(SHAPES.local_experts, SHAPES.experts);
        assert_eq!(
            SHAPES.expert_intermediate * 4,
            TP4_288_TOP8.expert_intermediate
        );
        let [decode, prefill] = routes();
        for route in [decode, prefill] {
            route.validate().unwrap();
            assert_eq!(route.local_experts, 288);
        }
        assert_eq!((decode.capacity, decode.pad), (32, 16));
        assert_eq!((prefill.capacity, prefill.pad), (8192, 64));
        // Every route plus each expert's padding.
        assert_eq!(prefill.sorted_rows_capacity(), 8192 * 8 + 288 * 63);
    }

    #[test]
    fn the_catalog_holds_every_routed_format_projection_and_phase() {
        let projections = catalog();
        assert_eq!(projections.len(), 4 * 2 * 2);
        let mut symbols = projections
            .iter()
            .map(|p| p.symbol.clone())
            .collect::<Vec<_>>();
        symbols.sort();
        symbols.dedup();
        assert_eq!(symbols.len(), projections.len());
        for p in &projections {
            let gemm = p.query.gemm;
            let grouped = gemm.grouped.expect("routed");
            assert_eq!((grouped.experts, grouped.top_k), (288, 8));
            let tokens = if p.row_tile == 16 { 32 } else { 8192 };
            assert_eq!(p.tokens_capacity, tokens);
            assert_eq!(grouped.capacity, RoutedCapacity::TotalRows(tokens * 8));
            if p.gate_up() {
                assert_eq!((gemm.n, gemm.k), (1024, 4096));
            } else {
                assert_eq!((gemm.n, gemm.k), (4096, 512));
            }
        }
        // No symbol of the unsplit catalogs.
        for other in super::super::catalog()
            .into_iter()
            .chain(super::super::formats::catalog())
        {
            assert!(!symbols.contains(&other.symbol), "{}", other.symbol);
        }
        let down = projections
            .iter()
            .find(|p| p.weights() == Weights::Fp8Block128 && !p.gate_up() && p.row_tile == 16)
            .unwrap();
        assert_eq!(
            down.symbol,
            "chop_moe_chain_down_fp8block_e288k8_n4096_k512_r16_v1"
        );
    }

    #[test]
    fn the_forms_take_every_decode_and_prefill_schedule() {
        let forms = forms();
        let find = |weights, gate_up, row_tile, schedule: fn(Schedule) -> bool| {
            forms
                .iter()
                .filter(|p| {
                    p.weights() == weights
                        && p.gate_up() == gate_up
                        && p.row_tile == row_tile
                        && schedule(p.schedule)
                })
                .count()
        };
        let fp8 = Weights::Fp8Block128;
        assert_eq!(find(fp8, true, 16, |s| s == Schedule::WarpStreams), 1);
        assert_eq!(find(fp8, false, 16, |s| s == Schedule::BlockWarps), 1);
        assert_eq!(find(fp8, false, 16, |s| s == Schedule::Columns), 1);
        // Block warps take the decode down's fewest tokens, columns the rest.
        let capacities = forms
            .iter()
            .filter(|p| p.weights() == fp8 && !p.gate_up() && p.row_tile == 16)
            .map(|p| (p.schedule, p.tokens_capacity))
            .collect::<Vec<_>>();
        assert_eq!(
            capacities,
            [
                (Schedule::BlockWarps, DOWN_BLOCK_WARP_TOKENS),
                (Schedule::Columns, SHAPES.decode_tokens)
            ]
        );
        assert_eq!(find(fp8, true, 64, |s| s == Schedule::Columns), 1);
        assert_eq!(find(fp8, false, 64, |s| s == Schedule::Columns), 1);
        assert_eq!(
            find(fp8, true, 64, |s| s
                == Schedule::TilePairs {
                    min_pair_tiles: GATE_UP_MIN_PAIR_TILES
                }),
            1
        );
        assert_eq!(
            find(fp8, false, 64, |s| s
                == Schedule::TilePairs {
                    min_pair_tiles: DOWN_MIN_PAIR_TILES
                }),
            1
        );
        for weights in [Weights::Nvfp4, Weights::Bf16] {
            for gate_up in [true, false] {
                assert_eq!(find(weights, gate_up, 16, |s| s == Schedule::BlockWarps), 1);
                assert_eq!(find(weights, gate_up, 64, |s| s == Schedule::Columns), 1);
            }
        }
        for gate_up in [true, false] {
            assert_eq!(
                find(Weights::Mxfp8, gate_up, 16, |s| s == Schedule::Columns),
                1
            );
        }
        assert_eq!(forms.len(), 4 * 2 * 2 + 3);
        assert_eq!(
            forms
                .iter()
                .filter(|p| matches!(p.schedule, Schedule::TilePairs { .. }))
                .count(),
            2
        );
        // Warp streams refuse the K 512 down: one 512-K round.
        let down = catalog()
            .into_iter()
            .find(|p| p.weights() == fp8 && !p.gate_up() && p.row_tile == 16)
            .unwrap();
        assert!(down.warp_streams().is_err());
    }

    #[test]
    fn the_accumulation_projections_take_k_4096_in_every_format() {
        let projections = accumulation_projections();
        let symbols = projections
            .iter()
            .map(|p| p.symbol.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            symbols,
            [
                "chop_moe_chain_down_fp8block_e288k8_n4096_k4096_r64_v1",
                "chop_moe_chain_down_fp8block_e288k8_n4096_k4096_r16_bw4_v1",
                "chop_moe_chain_down_nvfp4_e288k8_n4096_k4096_r64_v1",
                "chop_moe_chain_down_mxfp8_e288k8_n4096_k4096_r64_v1",
                "chop_moe_chain_down_bf16_e288k8_n4096_k4096_r64_v1",
            ]
        );
        for p in &projections {
            let gemm = p.query.gemm;
            assert_eq!((gemm.n, gemm.k), (4096, 4096));
            assert_eq!(gemm.grouped.expect("routed").experts, 288);
            assert!(!p.gate_up());
        }
        assert_eq!(projections[1].schedule, Schedule::BlockWarps);
        assert_eq!(projections[1].tokens_capacity, SHAPES.decode_tokens);
        // No symbol of the served forms.
        let served = forms();
        assert!(
            projections
                .iter()
                .all(|p| served.iter().all(|s| s.symbol != p.symbol))
        );
    }

    #[test]
    fn the_cubins_hold_the_forms_routes_and_sums() {
        let specs = cubin_specs().unwrap();
        let names = specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "chop_moe_chain_e288l288k8_fp8block_ws",
                "chop_moe_chain_e288l288k8_fp8block_tp",
                "chop_moe_chain_e288l288k8_fp8block",
                "chop_moe_chain_e288l288k8_fp8block_bw4",
                "chop_moe_chain_e288l288k8_nvfp4_bw4",
                "chop_moe_chain_e288l288k8_nvfp4",
                "chop_moe_chain_e288l288k8_mxfp8",
                "chop_moe_chain_e288l288k8_bf16_bw4",
                "chop_moe_chain_e288l288k8_bf16",
                "chop_moe_chain_e288l288k8_route",
                "chop_moe_chain_e288l288k8_sum_add",
            ]
        );
        let entries = specs.iter().map(|s| s.entries.len()).sum::<usize>();
        // 19 forms; the decode route, three prefill route kernels, two sums;
        // one sum that adds a row.
        assert_eq!(entries, 19 + 4 + 2 + 1);
        let route = specs.iter().find(|s| s.name.ends_with("_route")).unwrap();
        let symbols = route
            .entries
            .iter()
            .map(|e| e.symbol.as_str())
            .collect::<Vec<_>>();
        assert!(symbols.contains(&"chop_moe_chain_e288l288k8_decode_e288l288k8_t32_p16_v1"));
        assert!(symbols.contains(&"chop_moe_chain_e288l288k8_sum_f32_k8_w4096_l288_t8192_v1"));
    }
}
