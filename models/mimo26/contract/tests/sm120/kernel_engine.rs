//! The shared GEMM and grouped expert engines on MiMo V2.6 Flash's served
//! entries (`sm120::catalogs`): plans, tensor maps, arguments, medium-M
//! stand-ins, exact and verify buckets, capability offers and arithmetic
//! identities. CPU only.

use kernel_build::{BuildContext, KernelSelection};
use kernel_core::contract::*;
use kernel_core::precision::*;
use kernels_sm120::arithmetic::DeclaredCubin;
use kernels_sm120::attention::sparse;
use kernels_sm120::moe::{self, InputPath, Phase, Projection, WeightLayout, template};
use kernels_sm120::precision::*;
use kernels_sm120::{Catalogs, Profile, gemm};

use model_mimo26_contract::sm120;
use std::collections::BTreeMap;

fn cat() -> &'static Catalogs {
    sm120::catalogs()
}

fn counts(m: u32, split_k: u8) -> RuntimeCounts {
    RuntimeCounts {
        m,
        batch: m,
        context: 0,
        routed_rows: 0,
        split_k,
    }
}

/// (N, K) of `name` in MiMo's operation list at four ranks.
fn nk(name: &str) -> (u32, u32) {
    sm120::gemm::rank_shapes(4)
        .into_iter()
        .find(|shape| shape.0 == name)
        .map(|(_, n, k, ..)| (n, k))
        .unwrap()
}

fn dense_entry(symbol: &str) -> InstantiationSpec {
    gemm::catalog(cat())
        .into_iter()
        .find(|item| item.symbol == symbol)
        .unwrap()
        .instantiation
}

fn mid_entry(reference: &str) -> gemm::mid::Entry {
    gemm::mid::catalog(cat())
        .into_iter()
        .find(|e| e.reference == reference)
        .unwrap()
}

fn identity(cubin: DeclaredCubin) -> String {
    let context = BuildContext::new("arithmetic-identity@0", "test", "arithmetic-identity")
        .source_root("sdk", kernels_sm120::cuda_source_root());
    KernelSelection::new(
        context,
        vec![cubin.select("test", "test.v1", "unit test").unwrap()],
    )
    .unwrap()
    .contract_identity()
    .unwrap()
}

#[test]
fn mimo_decode_entries_run_their_declared_kernels() {
    let items = gemm::catalog(cat());
    for ranks in [2, 4] {
        let tag = model_mimo26_contract::sm120::config::profile(ranks)
            .unwrap()
            .tag();
        for (name, .., decode) in sm120::gemm::rank_shapes(ranks) {
            for m in [1, 2, 4, 8, 16, 32] {
                let symbol = format!("chop_gemm_{tag}_{name}_m{m}_v1");
                let item = items
                    .iter()
                    .find(|item| item.symbol == symbol)
                    .unwrap_or_else(|| panic!("{symbol}"));
                assert_eq!(item.key.dense, Some(decode), "{symbol}");
            }
        }
    }
    let declared = |ranks, name| {
        sm120::gemm::rank_shapes(ranks)
            .into_iter()
            .find(|shape| shape.0 == name)
            .unwrap()
            .6
    };
    assert_eq!(declared(2, "mlp_down"), gemm::WHOLE_K_DECODE);
    assert_eq!(declared(4, "mlp_down"), gemm::CLUSTER_FP8);
    assert_eq!(declared(2, "dflash_q"), gemm::CLUSTER_BF16);
    assert_eq!(declared(4, "dflash_q"), gemm::WHOLE_K_DECODE);
}

#[test]
fn mimo_rank_shapes_at_tp4_match_tp4_entries() {
    let items = gemm::catalog(cat());
    let mut matched = 0;
    for (name, n, k, _, swiglu, ..) in sm120::gemm::rank_shapes(4) {
        for symbol in [
            format!("chop_gemm_mimo_{name}_m1_v1"),
            format!("chop_gemm_mimo_{name}_prefill_v1"),
        ] {
            let Some(item) = items.iter().find(|item| item.symbol == symbol) else {
                continue;
            };
            let StaticQuery::Gemm(query) = item.key.query else {
                panic!("{symbol} is a GEMM");
            };
            assert_eq!((query.n, query.k), (n, k), "{symbol}");
            assert_eq!(query.epilogues.swiglu.is_some(), swiglu, "{symbol}");
            assert_eq!(item.profile, model_mimo26_contract::sm120::config::PROFILE);
            matched += 1;
        }
    }
    assert!(matched >= 10, "{matched} TP4 entries matched");
}

#[test]
fn split_and_workspace_use_prepared_capacity() {
    let router = gemm::catalog(cat())
        .into_iter()
        .find(|s| s.symbol == "chop_gemm_mimo_router_m1_v1")
        .unwrap();
    assert_eq!(gemm::choose_split_k(&router.key).unwrap(), 4);
    let launch = gemm::plan(&router.key, counts(1, 4)).unwrap();
    assert_eq!(launch.grid, [64, 1, 1]);
    assert_eq!(launch.block, [256, 1, 1]);
    assert_eq!(launch.workspace_bytes, (16 * 4 * 16 * 8 + 16) * 4);
    assert!(launch.pdl);
    assert!(gemm::plan(&router.key, counts(1, 1)).is_err());
}

#[test]
fn cluster_decode_tiles_split_k_within_eight_ctas() {
    let items = gemm::catalog(cat());
    // (name, m, rows per CTA, token tile, cluster, grid)
    for (name, m, rows, token_tile, cluster, grid) in [
        ("qkv_ga", 1, 32, 8, 2, 212),
        ("qkv_ga", 16, 64, 16, 4, 212),
        ("qkv_ga", 32, 128, 32, 8, 216),
        ("qkv_swa", 8, 64, 8, 4, 232),
        ("o", 1, 32, 8, 2, 256),
        ("o", 32, 128, 32, 8, 256),
        ("mlp_gate_up", 1, 32, 8, 2, 512),
        ("mlp_gate_up", 32, 128, 32, 8, 512),
        ("mlp_down", 16, 64, 16, 4, 256),
    ] {
        let symbol = format!("chop_gemm_mimo_{name}_m{m}_v1");
        let item = items.iter().find(|item| item.symbol == symbol).unwrap();
        assert_eq!(item.key.knobs.tile_m, rows, "{symbol}");
        assert_eq!(item.key.knobs.tile_n, token_tile, "{symbol}");
        let launch = gemm::plan(&item.key, counts(m, 1)).unwrap();
        assert_eq!(launch.grid, [grid, 1, 1], "{symbol}");
        assert_eq!(launch.cluster, [cluster, 1, 1], "{symbol}");
        assert_eq!(launch.block, [256, 1, 1], "{symbol}");
        assert_eq!(
            launch.workspace_bytes,
            u64::from(grid) * 128 * u64::from(token_tile) * 4,
            "{symbol}"
        );
    }
}

#[test]
fn router_prefill_plans_the_segment_partials_at_every_m() {
    let router = gemm::catalog(cat())
        .into_iter()
        .find(|s| s.symbol == "chop_gemm_mimo_router_prefill_v1")
        .unwrap();
    for m in [1, 48, 100, 129, 8192] {
        let launch = gemm::plan(&router.key, counts(m, 1)).unwrap();
        assert_eq!(launch.grid, [gemm::SM_COUNT, 1, 1], "M={m}");
        assert_eq!(launch.block, [384, 1, 1], "M={m}");
        // Eight partial tiles per CTA.
        assert_eq!(launch.workspace_bytes, 188 * 8 * 8 * 16 * 4 * 32 * 4);
    }
    let spec = gemm::cubin_spec(std::slice::from_ref(&router.instantiation)).unwrap();
    assert_eq!(spec.entries[0].params.len(), 15);
}

#[test]
fn router_kernel_refuses_swiglu() {
    let mut item = gemm::catalog(cat())
        .into_iter()
        .find(|item| item.symbol == "chop_gemm_mimo_router_m1_v1")
        .unwrap()
        .instantiation;
    let StaticQuery::Gemm(ref mut query) = item.key.query else {
        unreachable!()
    };
    query.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
    let expected = Refusal("router GEMM does not implement SwiGLU");
    assert_eq!(
        gemm::plan(&item.key, counts(1, item.key.knobs.split_k_capacity)),
        Err(expected)
    );
    assert_eq!(gemm::cubin_spec(&[item]).err(), Some(expected));
}

#[test]
fn qkv_partial_scale_row_and_prefill_tiles() {
    let items = gemm::catalog(cat());
    let qkv = items
        .iter()
        .find(|s| s.symbol == "chop_gemm_mimo_qkv_ga_m1_v1")
        .unwrap();
    assert_eq!(gemm::choose_split_k(&qkv.key).unwrap(), 1);
    let launch = gemm::plan(&qkv.key, counts(1, 1)).unwrap();
    assert_eq!(launch.grid[0], 212);
    assert_eq!(launch.block[0], 256);
    // Eight padded 4 KiB activation rows plus the partial tile.
    assert_eq!(
        launch.dynamic_shared_bytes,
        8 * (16 * 256 + 16) + 8 * 16 * 8 * 4
    );
    let head = items
        .iter()
        .find(|s| s.symbol == "chop_gemm_mimo_head_prefill_v1")
        .unwrap();
    let head_plan = gemm::plan(&head.key, counts(8192, 1)).unwrap();
    assert_eq!(
        head_plan.dynamic_shared_bytes,
        3 * (128 * 128 + 8 * 2048) + 1024
    );
    assert_eq!(head_plan.grid[0], 188);
    assert_eq!(head_plan.block[0], 384);
    let qkv_prefill = items
        .iter()
        .find(|s| s.symbol == "chop_gemm_mimo_qkv_ga_prefill_v1")
        .unwrap();
    let qkv_prefill_plan = gemm::plan(&qkv_prefill.key, counts(8192, 1)).unwrap();
    assert_eq!(
        qkv_prefill_plan.dynamic_shared_bytes,
        4 * (128 * 128 + 8 * 1024) + 1024
    );
}

#[test]
fn packed_prefill_tensor_maps_match_argument_contract() {
    for symbol in [
        "chop_gemm_mimo_qkv_ga_prefill_v1",
        "chop_gemm_mimo_o_prefill_v1",
    ] {
        let item = gemm::catalog(cat())
            .into_iter()
            .find(|item| item.symbol == symbol)
            .unwrap();
        let maps = gemm::tensor_maps(&item.key, 0x1000, 0x2000).unwrap();
        let weight_scales = (symbol != "chop_gemm_mimo_o_prefill_v1").then_some(0x5000);
        gemm::arguments(
            &item.key,
            counts(128, 1),
            maps.clone(),
            gemm::GemmBuffers {
                output: 0x3000,
                partial: 0x4000,
                semaphores: 0x4100,
                weight_scales,
                activation_scales: None,
                residual: None,
                weight_global_scale: None,
                activation_global_scale: None,
            },
        )
        .unwrap_or_else(|error| panic!("{symbol}: {error:?}; maps={maps:?}"));
    }
}

#[test]
fn swiglu_pairs_tiles_and_prefill_partials() {
    let items = gemm::catalog(cat());
    let decode = items
        .iter()
        .find(|s| s.symbol == "chop_gemm_mimo_mlp_gate_up_m1_v1")
        .unwrap();
    assert_eq!(gemm::choose_split_k(&decode.key).unwrap(), 1);
    let decode_plan = gemm::plan(&decode.key, counts(1, 1)).unwrap();
    assert_eq!(decode_plan.grid[0], 512);
    assert_eq!(decode_plan.workspace_bytes, 512 * 128 * 8 * 4);
    let prefill = items
        .iter()
        .find(|s| s.symbol == "chop_gemm_mimo_mlp_gate_up_prefill_v1")
        .unwrap();
    let prefill_plan = gemm::plan(&prefill.key, counts(8192, 1)).unwrap();
    assert_eq!(prefill_plan.grid[0], 188);
    assert_eq!(prefill_plan.workspace_bytes, 188 * 8 * 8 * 16 * 4 * 32 * 4);
}

#[test]
fn specialized_fragment_plans_refuse_unsupported_shapes() {
    let mut paths = [[false; 2]; 2];
    for v in sm120::moe::catalog()
        .into_iter()
        .filter(|v| v.fragment_decode() || v.fragment_prefill())
    {
        let path = usize::from(v.fragment_prefill());
        paths[path][usize::from(v.projection == Projection::Down)] = true;
        let counts = RuntimeCounts {
            m: 1,
            batch: 0,
            context: 0,
            routed_rows: 8,
            split_k: v.knobs.split_k_capacity,
        };
        assert!(v.plan(counts).is_ok(), "{}", v.symbol);
        for field in ["top_k", "experts", "hidden", "intermediate"] {
            let mut invalid = v.clone();
            let mut shape = sm120::moe::SHAPE;
            match field {
                "top_k" => shape.top_k = 3,
                "experts" => shape.experts = 128,
                "hidden" => shape.hidden = 2048,
                // 1024 and 512 are MiMo's at TP2 and TP4.
                "intermediate" => shape.intermediate = 384,
                _ => unreachable!(),
            }
            let (n, k) = shape.projection(v.projection);
            invalid.query.n = n;
            invalid.query.k = k;
            invalid.query.m_capacity = v.tokens_capacity * shape.top_k;
            invalid.query.grouped = Some(GroupedGemm {
                experts: shape.experts,
                top_k: shape.top_k,
                capacity: RoutedCapacity::TotalRows(
                    moe::padded_capacity(
                        invalid.query.m_capacity,
                        shape.experts,
                        u32::from(v.knobs.tile_m),
                    )
                    .unwrap(),
                ),
            });
            // The storage implementation, not the profile tag, limits shapes.
            invalid.profile = Profile::StdDenseMoe;
            assert_eq!(
                invalid.plan(RuntimeCounts {
                    routed_rows: shape.top_k,
                    ..counts
                }),
                Err(Refusal("unsupported specialized MXFP4 fragment shape")),
                "{}: {field}",
                v.symbol,
            );
        }
    }
    assert_eq!(paths, [[true; 2]; 2]);
}

#[test]
fn exact_entries_take_the_one_token_reduction_order() {
    let all = sm120::moe::catalog();
    for profile in model_mimo26_contract::sm120::PROFILES {
        for projection in [Projection::GateUp, Projection::Down] {
            let find = |tokens, exact| {
                all.iter().find(|v| {
                    v.profile == profile
                        && v.projection == projection
                        && v.phase == Phase::Decode
                        && v.path == InputPath::A16Gather
                        && v.tokens_capacity == tokens
                        && v.exact == exact
                })
            };
            let one = find(1, false).unwrap();
            assert!(one.fragment_decode() && find(1, true).is_none());
            for tokens in sm120::moe::EXACT_BUCKETS {
                let decode = find(tokens, false).unwrap();
                let exact = find(tokens, true);
                // Down runs one K split at every bucket, bucket 1's order.
                assert_eq!(
                    exact.is_some(),
                    projection == Projection::GateUp,
                    "{profile:?} {projection:?} t{tokens}"
                );
                let served = exact.unwrap_or(decode);
                assert!(served.fragment_decode(), "{}", served.symbol);
                assert_eq!(
                    (served.knobs.warps, served.knobs.split_k_capacity),
                    (one.knobs.warps, one.knobs.split_k_capacity),
                    "{}",
                    served.symbol
                );
                if let Some(exact) = exact {
                    assert!(exact.symbol.contains("_exact_"), "{}", exact.symbol);
                    assert_eq!(
                        (exact.query, exact.tokens_capacity),
                        (decode.query, decode.tokens_capacity)
                    );
                }
            }
        }
    }
    assert!(all.iter().all(|v| {
        !v.exact
            || sm120::geometry::MODELS
                .iter()
                .any(|owner| v.profile.ranks_of(owner.model).is_some())
    }));
}

#[test]
fn decode_cubins_hold_buckets_up_to_16_rows() {
    // Wider decode and verify buckets run the prefill entries.
    let specs = moe::cubin_specs(cat(), Some(model_mimo26_contract::sm120::config::PROFILE));
    for name in ["gateup", "down"] {
        let decode = format!("chop_moe_mimo_{name}_decode_mxfp4_a16");
        assert_eq!(
            specs
                .iter()
                .find(|spec| spec.name == decode)
                .unwrap()
                .entries
                .len(),
            sm120::moe::DECODE_BUCKETS.len()
        );
    }
    let table = sm120::moe::tuning();
    for (profile, weight, path) in [
        (
            Profile::StdDenseMoe,
            WeightFormat::Mxfp8,
            InputPath::A16Gather,
        ),
        (
            model_mimo26_contract::sm120::config::PROFILE,
            WeightFormat::Mxfp4,
            InputPath::A8Pregather,
        ),
    ] {
        assert!(
            moe::variant(
                profile,
                "moe",
                if weight == WeightFormat::Mxfp4 {
                    WeightLayout::Mxfp4Fragment
                } else {
                    WeightLayout::Packed
                },
                Projection::GateUp,
                Phase::Decode,
                64,
                weight,
                path,
                &table
            )
            .is_err()
        );
    }
}

#[test]
fn split_selection_and_reduction_workspace_match_decode_shapes() {
    let variants = sm120::moe::catalog();
    let get = |profile, projection, path, tokens, quantized| {
        variants
            .iter()
            .find(|v| {
                v.profile == profile
                    && v.projection == projection
                    && v.path == path
                    && v.phase == Phase::Decode
                    && v.tokens_capacity == tokens
                    && v.query.epilogues.output_quantization.is_some() == quantized
            })
            .unwrap()
    };
    let gate = get(
        model_mimo26_contract::sm120::config::PROFILE,
        Projection::GateUp,
        InputPath::A16Gather,
        1,
        false,
    );
    assert_eq!(gate.knobs.split_k_capacity, 16);
    assert_eq!(gate.reduction_symbol(), None);
    let down = get(
        model_mimo26_contract::sm120::config::PROFILE,
        Projection::Down,
        InputPath::A16Gather,
        1,
        false,
    );
    assert_eq!(down.knobs.split_k_capacity, 1);
    let qmx8 = get(
        model_mimo26_contract::sm120::config::PROFILE,
        Projection::GateUp,
        InputPath::A8Pregather,
        32,
        true,
    );
    assert_eq!(qmx8.knobs.split_k_capacity, 2);
    let counts = RuntimeCounts {
        m: 32,
        batch: 0,
        context: 0,
        routed_rows: 256,
        split_k: 2,
    };
    assert_eq!(
        qmx8.plan(counts).unwrap().workspace_bytes,
        256 * 512 * 2 * 2 * 4
    );
    assert_eq!(qmx8.plan(counts).unwrap().dynamic_shared_bytes, 65_536);
    assert_eq!(
        qmx8.reduction_plan(counts).unwrap().unwrap().grid,
        [1024, 1, 1]
    );
    assert!(qmx8.reduction_symbol().unwrap().ends_with("_reduce_v1"));
    let prefill = variants.iter().find(|v| v.phase == Phase::Prefill).unwrap();
    assert_eq!(prefill.knobs.split_k_capacity, 1);
}

#[test]
fn mimo_fragment_prefill_plans_a_persistent_grid() {
    let catalog = sm120::moe::catalog();
    let prefill = catalog
        .iter()
        .filter(|v| v.fragment_prefill())
        .collect::<Vec<_>>();
    // Gate/up and down for each geometry at TP4 and TP2.
    assert_eq!(prefill.len(), 4 * sm120::geometry::MODELS.len());
    for v in prefill {
        assert_eq!(v.knobs.tile_m, 64, "route alignment stays at 64 rows");
        assert_eq!(v.knobs.tile_n, moe::FRAGMENT_PREFILL_TOKEN_TILE);
        let counts = RuntimeCounts {
            m: 8192,
            batch: 0,
            context: 0,
            routed_rows: 65536,
            split_k: 1,
        };
        let plan = v.plan(counts).unwrap();
        assert_eq!(plan.grid, [moe::SM_COUNT, 1, 1]);
        assert_eq!(plan.block, [384, 1, 1]);
        // The served plan takes its rings from the entry's schedule row; the
        // engine's stages below bound the rows the table may choose.
        let Some(template::ScheduleKnobs::Prefill(row)) = v.schedule else {
            panic!("{}: no prefill schedule row", v.symbol);
        };
        assert_eq!(
            plan.dynamic_shared_bytes,
            moe::fragment_prefill_shared_bytes(
                row.weight_stages as u8,
                row.activation_stages as u8
            )
        );
        assert!(plan.dynamic_shared_bytes + moe::FRAGMENT_PREFILL_RESERVED_BYTES <= 101_376);
        assert!(v.reduction_plan(counts).unwrap().is_none());
        let refused = Err(Refusal("fragment prefill rings do not fit"));
        let mut knobs = v.knobs;
        knobs.stages += 1;
        assert_eq!(
            moe::supported_fragment_prefill(knobs, v.weight_stages),
            refused
        );
        assert_eq!(moe::supported_fragment_prefill(v.knobs, 1), refused);
    }
    assert!(
        catalog
            .iter()
            .filter(|v| !v.fragment_prefill() && !v.fp8_fragment_weights())
            .all(|v| (v.knobs.tile_n, v.knobs.tile_k, v.weight_stages) == (16, 32, 0))
    );
}

#[test]
fn weight_map_and_scale_metadata_match_mimo_shapes() {
    let v = &sm120::moe::catalog()[0];
    let map = v.weight_tensor_map(1).unwrap();
    assert_eq!(map.dims, [2048, 256 * 1024]);
    assert_eq!(map.box_dims, [128, 16]);
    let layout = v.weight_scale_layout().unwrap();
    assert_eq!(layout.k_blocks, 128);
    assert_eq!(layout.k_blocks_per_tile, 8);
    assert_eq!(
        v.weight_fragment_layout(),
        Some(kernel_core::transform::Mxfp4MoeFragmentLayout {
            experts: 256,
            rows_per_expert: 1024,
            k: 4096,
        })
    );
    // The kernel base's entries declare interleaved columns.
    let interleaved = moe::catalog(cat())
        .into_iter()
        .find(|v| v.query.epilogues.swiglu == Some(SwigluColumns::Interleaved))
        .unwrap();
    assert_eq!(interleaved.weight_tensor_map(1).unwrap().box_dims[1], 32);
}

#[test]
fn fused_router_tail_refuses_batches_above_32_tokens() {
    assert!(sm120::moe::fused_router_plan(32, 32).is_ok());
    for rows in [33, 64, 256] {
        for (capacity, live) in [(rows, rows), (rows, 1), (32, rows)] {
            let refusal = sm120::moe::fused_router_plan(capacity, live).unwrap_err();
            assert!(refusal.0.contains("at most 32 tokens"), "{}", refusal.0);
        }
    }
}

#[test]
fn selects_by_rows_above_the_decode_batch_only() {
    let reference = "chop_gemm_mimo_qkv_ga_prefill_v1";
    for m in [1, 32, 65, 257] {
        assert!(gemm::mid::select(cat(), reference, m).is_none(), "M={m}");
    }
    for m in [33, 64] {
        assert_eq!(
            gemm::mid::select(cat(), reference, m)
                .unwrap()
                .kernel
                .symbol(),
            "chop_gemm_mid_fp8b_t64_v1",
            "M={m}"
        );
    }
    assert!(gemm::mid::select(cat(), "chop_gemm_mimo_head_f32_prefill_v1", 64).is_some());
}

#[test]
fn plans_one_cta_per_sm_and_fits_shared_memory() {
    let fp8 = mid_entry("chop_gemm_mimo_qkv_ga_prefill_v1");
    let plan = fp8.plan(64).unwrap();
    assert_eq!(plan.grid, [188, 1, 1]);
    assert_eq!(plan.block, [384, 1, 1]);
    // Six 16 KiB stages; eight half-size partial tiles per CTA.
    assert_eq!(plan.dynamic_shared_bytes, 6 * 16 * 1024 + 1024);
    assert_eq!(plan.workspace_bytes, 188 * 8 * 8 * 8 * 32 * 4 * 4);
    assert!(fp8.plan(65).is_err());
    let bf16 = mid_entry("chop_gemm_mimo_o_prefill_v1");
    assert_eq!(
        bf16.plan(33).unwrap().dynamic_shared_bytes,
        4 * 24 * 1024 + 1024
    );
    // The reference plans the same grid with its 128-token ring.
    let reference = gemm::catalog(cat())
        .into_iter()
        .find(|item| item.symbol == "chop_gemm_mimo_o_prefill_v1")
        .unwrap();
    let counts = RuntimeCounts {
        m: 64,
        batch: 64,
        context: 0,
        routed_rows: 0,
        split_k: 1,
    };
    let prefill = gemm::plan(&reference.key, counts).unwrap();
    assert_eq!(bf16.plan(64).unwrap().grid, prefill.grid);
    assert_eq!(
        prefill.workspace_bytes,
        2 * bf16.plan(64).unwrap().workspace_bytes
    );
    for kernel in gemm::mid::Kernel::ALL {
        assert!(kernel.dynamic_shared_bytes() + 1152 <= 101_376);
    }
}

#[test]
fn arguments_follow_the_reference_abi() {
    let e = mid_entry("chop_gemm_mimo_mlp_down_prefill_v1");
    let map = e.activation_map(0x2000, TensorMapL2Promotion::B256);
    assert_eq!(map.box_dims, [64, 64]);
    let buffers = gemm::mid::Buffers {
        activation: 0x2000,
        weight: 0x1000,
        weight_scales: Some(0x5000),
        output: 0x3000,
        partial: 0x4000,
        semaphores: 0x4100,
    };
    let args = e.arguments(64, map.clone(), buffers).unwrap();
    assert_eq!(
        args.0.len(),
        gemm::cubin_spec(std::slice::from_ref(&dense_entry(
            "chop_gemm_mimo_mlp_down_prefill_v1"
        )))
        .unwrap()
        .entries[0]
            .params
            .len()
    );
    assert!(
        e.arguments(
            64,
            map.clone(),
            gemm::mid::Buffers {
                weight_scales: None,
                ..buffers
            }
        )
        .is_err()
    );
    let mut wrong = map;
    wrong.box_dims = vec![64, 128];
    assert!(e.arguments(64, wrong, buffers).is_err());
}

#[test]
fn cubins_hold_only_selected_kernels() {
    let specs = gemm::mid::cubin_specs(cat(), Some(model_mimo26_contract::sm120::config::PROFILE));
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "gemm_mid_mimo_sm120");
    let symbols: Vec<_> = specs[0].entries.iter().map(|e| e.symbol.as_str()).collect();
    assert_eq!(
        symbols,
        [
            "chop_gemm_mid_bf16_t64_v1",
            "chop_gemm_mid_bf16_swiglu_t64_v1",
            "chop_gemm_mid_fp8b_t64_v1",
            "chop_gemm_mid_fp8b_swiglu_t64_v1",
        ]
    );
    assert_eq!(
        gemm::mid::dynamic_shared_bytes("chop_gemm_mid_fp8b_t64_v1"),
        Some(6 * 16 * 1024 + 1024)
    );
    assert_eq!(
        gemm::mid::dynamic_shared_bytes("chop_gemm_mimo_o_prefill_v1"),
        None
    );
    assert!(gemm::mid::cubin_specs(cat(), Some(Profile::StdDenseMoe)).is_empty());
    // One MiMo cubin serves every topology: TP2 adds the QKV
    // kernel with 64-row scales.
    let mimo = gemm::mid::cubin_spec_for(
        cat(),
        "gemm_mid_mimo_sm120",
        &model_mimo26_contract::sm120::PROFILES,
    )
    .unwrap();
    assert_eq!(mimo.entries.len(), 5);
    assert!(
        mimo.entries
            .iter()
            .any(|e| e.symbol == "chop_gemm_mid_fp8r64_t64_v1")
    );
    let e = gemm::mid::select(cat(), "chop_gemm_mimo_tp2_qkv_ga_prefill_v1", 64).unwrap();
    assert_eq!(e.kernel.symbol(), "chop_gemm_mid_fp8r64_t64_v1");
}

#[test]
fn mimo_ring_depth_matches_the_planner() {
    for v in sm120::moe::catalog().into_iter().filter(|v| {
        template::schedule(v) == Some(template::Schedule::Ring) && !v.fp8_fragment_weights()
    }) {
        let depth = moe::mxfp4_fragment::ring_depth(&v);
        let parts = if v.projection == Projection::GateUp {
            2
        } else {
            1
        };
        let slot = parts * 640
            + if v.tokens_capacity > 1 {
                v.tokens_capacity * 144
            } else {
                0
            };
        let row = if v.tokens_capacity == 1 {
            v.query.k * 2
        } else {
            0
        };
        let plan = v
            .engine_plan(RuntimeCounts {
                m: 1,
                batch: 0,
                context: 0,
                routed_rows: 1,
                split_k: v.knobs.split_k_capacity,
            })
            .unwrap();
        assert_eq!(
            plan.dynamic_shared_bytes,
            u32::from(v.knobs.warps) * depth * slot + row,
            "{}",
            v.symbol
        );
    }
}

#[test]
fn planner_uses_capacity_and_refuses_excess_live_counts() {
    let v = &sm120::moe::catalog()[0];
    let counts = RuntimeCounts {
        m: 1,
        batch: 0,
        context: 0,
        routed_rows: 8,
        split_k: v.knobs.split_k_capacity,
    };
    let plan = v.engine_plan(counts).unwrap();
    // Eight experts by 32 gate/up strips; sixteen warps split K and each
    // streams two K64 tiles of gate and up ahead, then the 8 KB row.
    assert_eq!(plan.block, [512, 1, 1]);
    assert_eq!(plan.grid, [256, 1, 1]);
    assert_eq!(plan.workspace_bytes, 0);
    assert_eq!(plan.dynamic_shared_bytes, 16 * 2 * 2 * 640 + 4096 * 2);
    assert!(plan.pdl);
    assert_eq!(v.reduction_plan(counts).unwrap(), None);
    assert_eq!(
        v.engine_plan(RuntimeCounts { m: 2, ..counts }),
        Err(Refusal(
            "live MoE counts exceed prepared capacity or split-K differs"
        ))
    );
    assert_eq!(
        v.engine_plan(RuntimeCounts {
            split_k: 1,
            ..counts
        }),
        Err(Refusal(
            "live MoE counts exceed prepared capacity or split-K differs"
        ))
    );
}

#[test]
fn a16_and_a8_have_distinct_numerical_contracts() {
    let catalog = sm120::moe::catalog();
    let a16 = &catalog[0];
    let a8 = catalog
        .iter()
        .find(|v| {
            v.profile == a16.profile
                && v.projection == a16.projection
                && v.phase == a16.phase
                && v.tokens_capacity == a16.tokens_capacity
                && v.path == InputPath::A8Pregather
                && v.query.epilogues.output_quantization.is_none()
        })
        .unwrap();
    assert_ne!(
        a16.query.numerical_contract_id().unwrap(),
        a8.query.numerical_contract_id().unwrap()
    );
    let qmx8 = catalog
        .iter()
        .find(|v| v.query.epilogues.output_quantization == Some(OutputQuantization::Mxfp8))
        .unwrap();
    assert_ne!(
        a8.query.numerical_contract_id().unwrap(),
        qmx8.query.numerical_contract_id().unwrap()
    );
    let handles = [1_u64; 15];
    let a16_args = a16
        .arguments(handles, a16.weight_tensor_map(1).unwrap())
        .unwrap();
    assert_eq!(a16_args.0[12], Argument::NullDevicePointer);
    assert_eq!(a16_args.0[13], Argument::NullDevicePointer);
    let qmx8_args = qmx8
        .arguments(handles, qmx8.weight_tensor_map(1).unwrap())
        .unwrap();
    assert_eq!(qmx8_args.0[12], Argument::DeviceBuffer(1));
    assert_eq!(qmx8_args.0[13], Argument::DeviceBuffer(1));
    assert_eq!(qmx8_args.0[14], Argument::DeviceBuffer(1));
    assert_eq!(
        qmx8.reduction_arguments(handles).unwrap().unwrap().0.len(),
        12
    );
}

/// MiMo's GA QKV prefill entry at two K segments instead of eight compiles
/// to the same wrapper, so the same cubin, but its cubin's producer declares
/// the arithmetic of the instantiation it was given, so it has another
/// contract identity.
#[test]
fn prefill_segments_change_the_identity_but_not_the_cubin() {
    let entry = gemm::catalog(cat())
        .into_iter()
        .find(|v| v.symbol == "chop_gemm_mimo_qkv_ga_prefill_v1")
        .unwrap()
        .instantiation;
    assert_eq!(
        entry.key.dense,
        Some(DenseKernel::FragmentPrefill { segments: 8 })
    );
    let mut other = entry.clone();
    other.key.dense = Some(DenseKernel::FragmentPrefill { segments: 2 });
    other.key.knobs.split_k_capacity = 2;
    let cubin = gemm::declared_cubin_spec(std::slice::from_ref(&entry)).unwrap();
    let other_cubin = gemm::declared_cubin_spec(std::slice::from_ref(&other)).unwrap();
    assert_eq!(
        format!("{:?}", cubin.spec()),
        format!("{:?}", other_cubin.spec())
    );
    assert!(
        other_cubin
            .arithmetic(&entry.symbol)
            .unwrap()
            .contains("FragmentPrefill { segments: 2 }")
    );
    assert_ne!(identity(cubin), identity(other_cubin));
}

/// An entry that takes its K splits at launch cannot be declared without
/// its arithmetic, as a cubin whose wrappers compile theirs in or with an
/// entry left out.
#[test]
fn launch_split_entries_require_their_arithmetic() {
    let entry = gemm::catalog(cat())
        .into_iter()
        .find(|v| v.symbol == "chop_gemm_mimo_qkv_ga_prefill_v1")
        .unwrap()
        .instantiation;
    let spec = gemm::cubin_spec(std::slice::from_ref(&entry)).unwrap();
    assert!(DeclaredCubin::compiled(spec.clone()).is_err());
    assert!(DeclaredCubin::new(spec.clone(), BTreeMap::new()).is_err());
    let undeclared = BTreeMap::from([(entry.symbol.clone(), None)]);
    assert!(DeclaredCubin::new(spec.clone(), undeclared).is_err());
    let declared = BTreeMap::from([(entry.symbol.clone(), Some("declared".into()))]);
    assert!(DeclaredCubin::new(spec, declared).is_ok());
    // QSA attention splits and medium-M segments are launch arguments too
    // (a registered model's skinny prefill splits as well: kernel-lab's
    // `catalogs` test).
    for cubin in sparse::cubin_specs(model_mimo26_contract::sm120::catalogs(), None)
        .into_iter()
        .chain(gemm::mid::cubin_specs(cat(), None))
    {
        assert!(DeclaredCubin::compiled(cubin).is_err());
    }
}

/// A compiled grouped entry offers its own experts, routing, layout and
/// scale encoding, within its token capacity.
#[test]
fn grouped_catalog_entries_offer_their_own_capability() {
    let catalog = moe::catalog(cat());
    let variant = catalog
        .iter()
        .find(|v| {
            v.profile == model_mimo26_contract::sm120::config::PROFILE
                && v.projection == moe::Projection::GateUp
                && v.phase == moe::Phase::Decode
                && v.tokens_capacity == 4
                && v.path == moe::InputPath::A16Gather
                && !v.exact
        })
        .unwrap();
    let usage = |capacity| OperationUse {
        phase: ExecutionPhase::Decode,
        capacity,
        graph: true,
    };
    let capability = catalog_grouped_capability(variant, usage(4)).unwrap();
    assert_eq!(capability.stages, std::slice::from_ref(&variant.symbol));
    assert_eq!(
        capability.kind,
        OperationKind::Grouped {
            experts: 256,
            top_k: 8
        }
    );
    assert_eq!(capability.operands, 2);
    assert_eq!(capability.weight, WeightFormat::Mxfp4);
    assert_eq!(capability.local_scale_dtypes, [DType::Ue8M0]);
    assert_eq!(capability.layout.id, "moe-mxfp4-fragment");
    assert_eq!(
        capability.arithmetic.decode_order,
        DecodeOrder::ExactScaledOperand
    );
    assert!(catalog_grouped_capability(variant, usage(8)).is_err());
}

/// A compiled catalog entry offers what it computes, as itself, and only
/// for BF16 activations within its capacity.
#[test]
fn catalog_entries_offer_their_own_dense_capability() {
    let catalog = gemm::catalog(cat());
    let entry = |symbol: &str| {
        catalog
            .iter()
            .find(|entry| entry.symbol == symbol)
            .unwrap()
            .instantiation
            .clone()
    };
    let usage = |capacity| OperationUse {
        phase: ExecutionPhase::Decode,
        capacity,
        graph: true,
    };
    let block = entry("chop_gemm_mimo_mlp_down_m4_v1");
    let capability = KernelOffer::catalog_dense(block.clone(), usage(4))
        .unwrap()
        .capability()
        .unwrap();
    assert_eq!(capability.stages, ["chop_gemm_mimo_mlp_down_m4_v1"]);
    assert_eq!(
        capability.weight,
        WeightFormat::Fp8Block128x128(ScaleMode::ExactF32)
    );
    assert_eq!(
        capability.arithmetic.decode_order,
        DecodeOrder::BlockScaleAfterMma
    );
    assert!(KernelOffer::catalog_dense(block, usage(8)).is_err());
    assert!(KernelOffer::catalog_dense(entry("chop_gemm_std_dense_a8_m1_v1"), usage(1)).is_err());
}

/// TP4's derived GEMM shapes equal the literals the TP4 catalogs, their
/// locks and tuning keys were written with (REVIEW-A4 finding 1).
#[test]
fn derived_tp4_shapes_equal_the_former_literals() {
    for (name, shape) in [
        ("o", (4096, 2048)),
        ("head", (38144, 4096)),
        ("head_f32", (38144, 4096)),
        ("qkv_ga", (3392, 4096)),
        ("qkv_swa", (3712, 4096)),
        ("mlp_gate_up", (8192, 4096)),
        ("mlp_down", (4096, 4096)),
        ("dflash_q", (2048, 4096)),
        ("dflash_qkv", (2560, 4096)),
        ("dflash_context_kv", (2560, 4096)),
        ("dflash_o", (4096, 2048)),
        ("dflash_gate_up", (8192, 4096)),
        ("dflash_down", (4096, 4096)),
        ("dflash_fc", (4096, 5120)),
    ] {
        assert_eq!(nk(name), shape, "{name}");
    }
    assert_eq!(
        (
            model_mimo26_contract::sm120::config::MOE.experts,
            model_mimo26_contract::sm120::config::HIDDEN
        ),
        (256, 4096)
    );
}

/// MiMo's fragment prefill entries compile their declared K segments.
#[test]
fn prefill_entries_carry_their_declared_segments() {
    let segments = |symbol: &str| gemm::prefill_segments(&dense_entry(symbol).key).unwrap();
    assert_eq!(segments("chop_gemm_mimo_dflash_o_prefill_v1"), 4);
    assert_eq!(segments("chop_gemm_mimo_o_prefill_v1"), 2);
    assert_eq!(segments("chop_gemm_mimo_head_f32_prefill_v1"), 4);
    for item in sm120::gemm::catalog() {
        if let Ok(segments) = gemm::prefill_segments(&item.key) {
            assert_eq!(item.key.knobs.split_k_capacity, segments, "{}", item.symbol);
        }
    }
}

/// MiMo's expert and route catalog sizes, and its route plans.
#[test]
fn capacity_and_catalog_are_static() {
    let catalog = sm120::moe::catalog();
    // TP4: both projections of the A16 path's five decode buckets and
    // prefill, the A8 path's six decode buckets and prefill, the A8
    // gate/up's MXFP8-quantized copies and the exact entries; TP2: A16
    // decode and prefill and the exact entries.
    assert_eq!(
        catalog
            .iter()
            .filter(
                |v| [sm120::config::profile(2).unwrap(), sm120::config::PROFILE]
                    .contains(&v.profile)
            )
            .count(),
        2 * 6 + 2 * 7 + 7 + 3 + (2 * 6 + 3)
    );
    assert!(catalog.iter().all(|v| v.instantiation().validate().is_ok()));
    assert_eq!(
        moe::catalog(cat()).len(),
        moe::own_catalog().len() + catalog.len()
    );
    let profile = Some(model_mimo26_contract::sm120::config::PROFILE);
    let tp4 = moe::cubin_specs(cat(), profile);
    assert!(tp4.iter().all(|spec| spec.name.contains("mimo")));
    let all_cubins = moe::cubin_specs(cat(), None);
    for spec in &tp4 {
        assert_eq!(
            all_cubins
                .iter()
                .filter(|all| all.name == spec.name)
                .count(),
            1
        );
    }
    for owner in sm120::geometry::MODELS {
        for ranks in [2, 4] {
            let profile = owner.profile(ranks);
            let specs = moe::cubin_specs(cat(), Some(profile));
            let expected = catalog.iter().filter(|v| v.profile == profile).count()
                + catalog
                    .iter()
                    .filter(|v| v.profile == profile && v.reduction_symbol().is_some())
                    .count()
                + sm120::moe::route_cubins_for(*owner)
                    .iter()
                    .map(|spec| spec.entries.len())
                    .sum::<usize>()
                    * usize::from(ranks == 4);
            assert_eq!(
                specs.iter().map(|spec| spec.entries.len()).sum::<usize>(),
                expected
            );
        }
    }
    assert_eq!(
        sm120::moe::fused_router_plan(8, 8).unwrap().grid,
        [64, 1, 1]
    );
    assert_eq!(
        sm120::moe::fused_router_plan(32, 32).unwrap().grid,
        [64, 1, 1]
    );
    assert!(sm120::moe::fused_router_plan(32, 33).is_err());
    assert!(sm120::moe::prefer_fused_down_combine(1));
    assert!(!sm120::moe::prefer_fused_down_combine(8));
    assert!(!sm120::moe::prefer_fused_down_combine(32));
}

/// A Variant declaring a route weight point or SwiGLU clamps the grouped
/// engines do not implement, or query epilogues that disagree with its
/// declaration, is refused by planning and arguments on both the template
/// and the packed engine path; neither falls back to the after-down,
/// unclamped arithmetic.
#[test]
fn undeclared_expert_numerics_are_refused_on_every_path() {
    use kernels_sm120::moe::{RouteWeight, SwigluClamp};
    let catalog = moe::catalog(cat());
    let template = catalog
        .iter()
        .find(|v| moe::template::covers(v) && v.projection == moe::Projection::GateUp)
        .unwrap()
        .clone();
    let packed = catalog
        .iter()
        .find(|v| !moe::template::covers(v) && v.projection == moe::Projection::GateUp)
        .unwrap()
        .clone();
    let clamp = SwigluClamp {
        gate_max: 10.0,
        up_limit: 10.0,
    };
    for base in [template, packed] {
        let counts = kernel_core::contract::RuntimeCounts {
            m: base.tokens_capacity,
            batch: 1,
            context: 0,
            routed_rows: base.query.m_capacity,
            split_k: base.knobs.split_k_capacity,
        };
        assert!(base.plan(counts).is_ok(), "{}", base.symbol);
        let map = base.weight_tensor_map(1).unwrap();
        let mut unweighted_down = base.clone();
        unweighted_down.projection = moe::Projection::Down;
        for bad in [
            moe::Variant {
                route_weight: RouteWeight::BeforeDown,
                ..base.clone()
            },
            moe::Variant {
                swiglu_clamp: Some(clamp),
                ..base.clone()
            },
            unweighted_down,
        ] {
            assert!(bad.check_numerics().is_err(), "{}", bad.symbol);
            assert!(bad.plan(counts).is_err(), "{}", bad.symbol);
            assert!(bad.engine_plan(counts).is_err(), "{}", bad.symbol);
            assert!(bad.reduction_plan(counts).is_err(), "{}", bad.symbol);
            assert!(
                bad.arguments([1; 15], map.clone()).is_err(),
                "{}",
                bad.symbol
            );
            assert!(bad.weight_tensor_map(1).is_err(), "{}", bad.symbol);
        }
    }
}

/// MiMo's fragment template reads the up half at column + N/2, so a
/// declaration of interleaved gate/up columns is refused before planning,
/// tensor maps and code generation; the packed engine keeps the entries it
/// implements them for.
#[test]
fn template_entries_refuse_interleaved_columns() {
    use kernel_core::contract::SwigluColumns;

    let catalog = moe::catalog(cat());
    let mut bad = catalog
        .iter()
        .find(|v| v.symbol == "chop_moe_mimo_gateup_decode_mxfp4_a16_t1_v1")
        .unwrap()
        .clone();
    assert_eq!(
        bad.query.epilogues.swiglu,
        Some(SwigluColumns::ConcatenatedHalves)
    );
    let counts = kernel_core::contract::RuntimeCounts {
        m: bad.tokens_capacity,
        batch: 1,
        context: 0,
        routed_rows: bad.query.m_capacity,
        split_k: bad.knobs.split_k_capacity,
    };
    let map = bad.weight_tensor_map(1).unwrap();
    bad.query.epilogues.swiglu = Some(SwigluColumns::Interleaved);
    assert!(bad.check_numerics().is_err());
    assert!(bad.plan(counts).is_err());
    assert!(bad.engine_plan(counts).is_err());
    assert!(bad.reduction_plan(counts).is_err());
    assert!(bad.arguments([1; 15], map).is_err());
    assert!(bad.weight_tensor_map(1).is_err());
    assert!(std::panic::catch_unwind(|| moe::template::knobs(&bad)).is_err());
    let interleaved = catalog
        .iter()
        .filter(|v| v.query.epilogues.swiglu == Some(SwigluColumns::Interleaved))
        .collect::<Vec<_>>();
    assert!(!interleaved.is_empty());
    for v in interleaved {
        assert!(v.check_numerics().is_ok(), "{}", v.symbol);
    }
}
