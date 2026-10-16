//! Qwen3.8's per-rank kernel shapes at 2 ranks, over its catalogs
//! (`sm120::catalogs`): every entry the TP2
//! bundle looks up exists in its own profile with TP4's parameters,
//! buckets and capacities at the documented shape (docs/topology-qwen.md),
//! validates, plans and matches its cubin entry; the TP4 profile keeps
//! exactly its own entries; and every symbol is unique.

use std::collections::{BTreeMap, BTreeSet};

use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{
    AttentionPhase, DType, GatedAttentionQuery, GemmQuery, NgramQuery, RecurrentQuery,
    RuntimeCounts, SparseAttentionQuery, StaticQuery, SwigluColumns,
};
use kernels_sm120::moe::{Phase, Projection};
use kernels_sm120::{Profile, attention, gemm, linear, moe};
use model_qwen38_contract::sm120::config::RankShape as Qwen38Shape;

use model_qwen38_contract::sm120;

/// The rank counts with per-rank profiles besides TP4.
const OTHER_RANKS: [u32; 1] = [2];
const DECODE_BUCKETS: [u32; 6] = [1, 2, 4, 8, 16, 32];

/// Every cubin of `profile` on the kernels-sm120 families the Qwen3.8
/// bundle builds per rank count.
fn qwen_cubins(profile: Profile) -> Vec<CubinSpec> {
    let profile = Some(profile);
    let mut specs = kernels_sm120::row::hyper_connection::cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    );
    specs.extend(attention::sparse::cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    ));
    specs.extend(linear::cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    ));
    specs.extend(attention::cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    ));
    specs.extend(gemm::cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    ));
    specs.extend(gemm::fused::cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    ));
    specs.extend(moe::cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    ));
    specs
}

/// The entry of `symbol` in `specs`, which must hold it exactly once.
fn entry<'a>(specs: &'a [CubinSpec], symbol: &str) -> &'a EntrySpec {
    let found = specs
        .iter()
        .flat_map(|spec| &spec.entries)
        .filter(|entry| entry.symbol == symbol)
        .collect::<Vec<_>>();
    assert_eq!(found.len(), 1, "{symbol}");
    found[0]
}

fn gemm_query(key: &kernel_core::contract::CompileKey) -> GemmQuery {
    let StaticQuery::Gemm(query) = key.query else {
        panic!("not a GEMM key");
    };
    query
}

#[test]
fn profiles_name_their_rank_counts() {
    for (ranks, profile, name, tag) in [
        (
            2,
            model_qwen38_contract::sm120::config::MODEL.at(2),
            "qwen38-flash-tp2",
            "qwen38_tp2",
        ),
        (
            4,
            model_qwen38_contract::sm120::config::PROFILE,
            "qwen38-flash",
            "qwen38",
        ),
    ] {
        assert_eq!(
            model_qwen38_contract::sm120::config::profile(ranks),
            Some(profile)
        );
        assert_eq!(
            profile.ranks_of(model_qwen38_contract::sm120::config::MODEL),
            Some(ranks)
        );
        assert_eq!((profile.as_str(), profile.tag()), (name, tag));
        assert!(sm120::PROFILES.contains(&profile));
    }
    assert_eq!(model_qwen38_contract::sm120::config::profile(1), None);
    assert_eq!(
        kernels_sm120::Model::new(
            kernels_sm120::profile_names!("other-model", "other", 4, 2),
            model_qwen38_contract::sm120::config::CONFIG
        )
        .at(4)
        .ranks_of(model_qwen38_contract::sm120::config::MODEL),
        None
    );
    // docs/topology-qwen.md, "Per-rank shapes": TP2 and TP4.
    let table = [2, 4].map(|ranks| {
        let s = Qwen38Shape { ranks };
        [
            s.q_heads(),
            s.q_proj_rows(),
            s.attn_row(),
            s.attn_value(),
            s.linear_key_heads(),
            s.linear_value_heads(),
            s.linear_qkv(),
            s.linear_value(),
            s.lin_ab_stride(),
            s.local_experts(),
            s.shared_padded(),
            s.vocab_shard(),
            s.ngram_heads(),
            s.conv_entry_bytes() as u32,
            s.delta_entry_bytes() as u32,
        ]
    });
    assert_eq!(
        table,
        [
            [
                12, 6144, 6656, 3072, 8, 24, 5120, 3072, 48, 256, 384, 124_160, 8, 30_720,
                1_572_864
            ],
            [
                6, 3072, 3584, 1536, 4, 12, 2560, 1536, 24, 128, 256, 62_080, 4, 15_360, 786_432
            ],
        ]
    );
}

#[test]
fn each_geometry_keeps_its_rank_entry_parameters() {
    for owner in sm120::geometry::MODELS {
        let base_profile = owner.model.at(4);
        let base = qwen_cubins(base_profile);
        assert!(!base.is_empty());
        for profile in owner
            .model
            .profiles()
            .into_iter()
            .filter(|profile| profile.tag() != base_profile.tag())
        {
            assert_eq!(
                sm120::geometry::owner(profile).unwrap().geometry.id,
                owner.geometry.id
            );
            let specs = qwen_cubins(profile);
            let mut compared = 0;
            for variant in specs.iter().flat_map(|spec| &spec.entries) {
                let symbol = variant
                    .symbol
                    .replacen(profile.tag(), base_profile.tag(), 1);
                let reference = base
                    .iter()
                    .flat_map(|spec| &spec.entries)
                    .find(|reference| reference.symbol == symbol)
                    .unwrap_or_else(|| {
                        panic!("{}: no TP4 entry for {}", owner.geometry.id, variant.symbol)
                    });
                assert_eq!(
                    variant.params, reference.params,
                    "{}: {}",
                    owner.geometry.id, variant.symbol
                );
                entry(&specs, &variant.symbol);
                compared += 1;
            }
            assert!(compared > 0, "{}: {}", owner.geometry.id, profile.tag());
        }
    }
}

#[test]
fn tp4_cubins_keep_their_entries() {
    let fixture = include_str!("../fixtures/qwen38_tp4_symbols.txt");
    let expected = fixture
        .lines()
        .map(|line| line.split_once(' ').expect("cubin and symbol"))
        .map(|(cubin, symbol)| (cubin.to_owned(), symbol.to_owned()))
        .collect::<Vec<_>>();
    let mut actual = qwen_cubins(model_qwen38_contract::sm120::config::PROFILE)
        .into_iter()
        .filter(|spec| !spec.entries.is_empty())
        .flat_map(|spec| {
            spec.entries
                .into_iter()
                .map(move |entry| (spec.name.clone(), entry.symbol))
        })
        .collect::<Vec<_>>();
    // The lock lists cubins by name; entries keep their order within one.
    actual.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(actual, expected);
    for (_, symbol) in &actual {
        assert!(
            !symbol.contains("_tp2") && !symbol.contains("_tp8"),
            "{symbol}"
        );
    }
}

#[test]
fn every_symbol_is_unique_and_tp_cubins_hold_only_their_profile() {
    let mut seen = BTreeSet::new();
    let mut names = BTreeSet::new();
    for profile in model_qwen38_contract::sm120::PROFILES {
        for spec in qwen_cubins(profile) {
            if spec.entries.is_empty() {
                continue;
            }
            assert!(names.insert(spec.name.clone()), "{}", spec.name);
            if profile != model_qwen38_contract::sm120::config::PROFILE {
                assert!(spec.name.contains(profile.tag()), "{}", spec.name);
            }
            for entry in spec.entries {
                if profile != model_qwen38_contract::sm120::config::PROFILE {
                    assert!(
                        entry.symbol.contains(&format!("{}_", profile.tag())),
                        "{}",
                        entry.symbol
                    );
                }
                assert!(seen.insert(entry.symbol.clone()), "{}", entry.symbol);
            }
        }
    }
}

#[test]
fn gated_rows_and_sparse_attention_at_each_rank_count() {
    for ranks in OTHER_RANKS {
        let profile = model_qwen38_contract::sm120::config::profile(ranks).unwrap();
        let tag = profile.tag();
        let q_heads = Qwen38Shape { ranks }.q_heads();
        let specs = attention::cubin_specs(model_qwen38_contract::sm120::catalogs(), Some(profile));
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, format!("attention_{tag}_sm120"));
        // Dense paged attention is TP4's only; the TP cubin holds the rows.
        assert!(
            attention::catalog(model_qwen38_contract::sm120::catalogs())
                .iter()
                .all(|variant| variant.profile != profile)
        );
        let rows = model_qwen38_contract::sm120::attention::gated_catalog(profile);
        assert_eq!(rows.len(), 3);
        assert_eq!(specs[0].entries.len(), 3);
        let tp4 = model_qwen38_contract::sm120::attention::gated_catalog(
            model_qwen38_contract::sm120::config::PROFILE,
        );
        for (row, tp4) in rows.iter().zip(&tp4) {
            assert_eq!(row.profile, profile);
            assert_eq!(row.symbol, tp4.symbol.replacen("qwen38", tag, 1));
            match row.query {
                GatedAttentionQuery::QkNormRopeKvWrite {
                    q_heads: heads,
                    kv_heads,
                    capacity,
                    ..
                } => assert_eq!((heads, kv_heads, capacity), (q_heads, 1, 8192)),
                GatedAttentionQuery::OutputGate {
                    heads, capacity, ..
                } => assert_eq!((heads, capacity), (q_heads, 8192)),
            }
            row.contract().unwrap();
            assert_eq!(row.plan(8192).unwrap().grid, [8192, 1, 1]);
            let spec = entry(&specs, &row.symbol);
            assert_eq!(spec.params, row.params);
            assert_eq!(spec.params, tp4.params);
            assert!(spec.body.contains(&format!("<{q_heads},1,256,256")));
        }

        let sparse =
            attention::sparse::cubin_specs(model_qwen38_contract::sm120::catalogs(), Some(profile));
        assert_eq!(sparse.len(), 1);
        assert_eq!(sparse[0].name, format!("sparse_attention_{tag}_sm120"));
        let entries = attention::sparse::catalog(model_qwen38_contract::sm120::catalogs())
            .into_iter()
            .filter(|entry| entry.profile == profile)
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 6);
        for dtype in ["bf16", "fp8"] {
            for (phase_name, phase, capacity) in [
                ("prefill", AttentionPhase::Prefill, 8192),
                ("decode", AttentionPhase::Decode, 256),
                ("verify", AttentionPhase::Verify, 256),
            ] {
                let name = format!("{tag}_attn_{dtype}_{phase_name}");
                let e = entries.iter().find(|e| e.name == name).unwrap();
                let SparseAttentionQuery::SparseAttention {
                    q_heads: heads,
                    kv_heads,
                    phase: found,
                    capacity: rows,
                    kv_dtype,
                    ..
                } = e.query
                else {
                    panic!("{name} is not attention");
                };
                assert_eq!(
                    (heads, kv_heads, found, rows),
                    (q_heads, 1, phase, capacity)
                );
                assert_eq!(kv_dtype == DType::Fp8E4M3, dtype == "fp8");
                assert_eq!(e.symbol(), format!("chop_qsa_{name}_v1"));
                e.contract().unwrap();
                for prepared in [1, capacity] {
                    let plan = e.plan(prepared, 262_144).unwrap();
                    assert_eq!(plan.grid[1], 1);
                    assert!(plan.dynamic_shared_bytes <= 101_376);
                    if let Some(merge) = e.merge_plan(prepared).unwrap() {
                        assert_eq!(merge.grid, [prepared, q_heads, 1]);
                        assert_eq!(
                            merge.workspace_bytes,
                            u64::from(prepared)
                                * u64::from(q_heads)
                                * u64::from(e.splits(prepared))
                                * 258
                                * 4
                        );
                    }
                }
                assert!(e.plan(capacity + 1, 1).is_err());
                // Twelve grouped heads pad to two n8 tiles, three to one.
                let padded = if q_heads > 8 { 16 } else { 8 };
                assert_eq!(
                    e.dynamic_shared_bytes(),
                    4 * e.unit_keys() * 256 * 2 + padded * 256 * 2 + 512 * 4
                );
                for spec in e.entry_specs().unwrap() {
                    let built = entry(&sparse, &spec.symbol);
                    assert_eq!(built.params, spec.params);
                    assert!(built.body.contains(&format!("Attention<{q_heads},1,256,")));
                }
            }
        }
    }
}

#[test]
fn linear_attention_at_each_rank_count() {
    let tp4 = model_qwen38_contract::sm120::linear::qwen38_catalog(
        model_qwen38_contract::sm120::config::PROFILE,
    );
    for ranks in OTHER_RANKS {
        let profile = model_qwen38_contract::sm120::config::profile(ranks).unwrap();
        let shape = Qwen38Shape { ranks };
        let specs = linear::cubin_specs(model_qwen38_contract::sm120::catalogs(), Some(profile));
        assert_eq!(specs.len(), 1);
        let entries = model_qwen38_contract::sm120::linear::qwen38_catalog(profile);
        assert_eq!(entries.len(), tp4.len());
        for (e, tp4) in entries.iter().zip(&tp4) {
            assert_eq!(e.profile, profile);
            assert_eq!(e.symbol, tp4.symbol.replacen("qwen38", profile.tag(), 1));
            assert_eq!((e.stage, &e.params), (tp4.stage, &tp4.params));
            match e.query {
                RecurrentQuery::CausalConv1d { channels, .. } => {
                    assert_eq!(channels, shape.linear_qkv());
                }
                RecurrentQuery::GatedDeltaRule {
                    key_heads,
                    value_heads,
                    ..
                } => assert_eq!(
                    (key_heads, value_heads),
                    (shape.linear_key_heads(), shape.linear_value_heads())
                ),
                RecurrentQuery::GatedRmsNorm { width, .. } => {
                    assert_eq!(width, shape.linear_value());
                }
            }
            assert_eq!(e.capacity(), tp4.capacity());
            e.contract().unwrap();
            let full = e.capacity();
            let plan = e
                .plan(full, full.sequences.min(full.rows), full.sequences)
                .unwrap();
            assert!(plan.dynamic_shared_bytes <= 101_376, "{}", e.symbol);
            assert_eq!(
                linear::dynamic_shared_bytes(model_qwen38_contract::sm120::catalogs(), &e.symbol),
                Some(plan.dynamic_shared_bytes)
            );
            // Capacity buckets: 8192 prefill rows over 256 items, 256
            // decode sequences.
            if e.symbol.ends_with("_delta_prefill_prepare_v1") {
                assert_eq!(plan.grid, [shape.linear_key_heads(), 384, 1]);
            }
            if e.symbol.ends_with("_delta_decode_v1") {
                assert_eq!(plan.grid, [8, shape.linear_value_heads(), 256]);
            }
            assert_eq!(entry(&specs, &e.symbol).params, e.params);
        }
    }
}

#[test]
fn experts_and_routes_at_each_rank_count() {
    let catalog = moe::catalog(model_qwen38_contract::sm120::catalogs());
    let tp4 = sm120::moe::route_catalog()
        .into_iter()
        .filter(|entry| entry.profile == model_qwen38_contract::sm120::config::PROFILE)
        .collect::<Vec<_>>();
    for ranks in OTHER_RANKS {
        let profile = model_qwen38_contract::sm120::config::profile(ranks).unwrap();
        let tag = profile.tag();
        let local = Qwen38Shape { ranks }.local_experts();
        assert_eq!(sm120::moe::local_experts(profile).unwrap(), local);
        let specs = moe::cubin_specs(model_qwen38_contract::sm120::catalogs(), Some(profile));
        assert_eq!(specs.len(), 6);
        for spec in &specs {
            if spec.name == format!("chop_moe_{tag}_route_parallel") {
                assert!(spec.defines.is_empty());
                // Only the C32 bucket dispatches the parallel route.
                assert_eq!(spec.entries.len(), 1);
                assert!(spec.entries[0].symbol.ends_with("_parallel_t32_v1"));
            } else {
                let mut expected = vec![("CHOP_QWEN38_LOCAL_EXPERTS".into(), local.to_string())];
                if spec.name == format!("chop_moe_{tag}_route") {
                    expected.extend([
                        ("CHOP_QWEN38_GLOBAL_EXPERTS".into(), "512".into()),
                        ("CHOP_QWEN38_TOP_K".into(), "10".into()),
                    ]);
                }
                assert_eq!(spec.defines, expected);
            }
        }
        for projection in [Projection::GateUp, Projection::Down] {
            for (phase, tokens) in DECODE_BUCKETS
                .map(|tokens| (Phase::Decode, tokens))
                .into_iter()
                .chain([(Phase::Prefill, 8192)])
            {
                let pick = |profile| {
                    catalog
                        .iter()
                        .find(|v| {
                            v.profile == profile
                                && v.projection == projection
                                && v.phase == phase
                                && v.tokens_capacity == tokens
                        })
                        .unwrap()
                };
                let (v, base) = (
                    pick(profile),
                    pick(model_qwen38_contract::sm120::config::PROFILE),
                );
                assert!(v.fp8_fragment_weights());
                assert_eq!(v.symbol, base.symbol.replacen("qwen38", tag, 1));
                assert_eq!((v.query.n, v.query.k), (base.query.n, base.query.k));
                assert_eq!((v.knobs, v.weight_stages), (base.knobs, base.weight_stages));
                let grouped = v.query.grouped.unwrap();
                assert_eq!(grouped.experts, local);
                assert_eq!(
                    v.query.grouped.unwrap().capacity,
                    kernel_core::contract::RoutedCapacity::TotalRows(
                        moe::padded_capacity(tokens * 10, local, 16).unwrap()
                    )
                );
                assert_eq!(
                    v.fp8_fragment_layout().unwrap().rows,
                    (local * v.query.n) as usize
                );
                let plan = v
                    .plan(RuntimeCounts {
                        m: tokens,
                        batch: 0,
                        context: 0,
                        routed_rows: v.query.m_capacity,
                        split_k: v.knobs.split_k_capacity,
                    })
                    .unwrap();
                assert!(plan.dynamic_shared_bytes + 1152 <= 101_376, "{}", v.symbol);
                if phase == Phase::Decode {
                    // The engines' decode splits and strips carry across rank
                    // counts; tuned table rows (the served plans) need not.
                    let counts = RuntimeCounts {
                        m: tokens,
                        batch: 0,
                        context: 0,
                        routed_rows: v.query.m_capacity,
                        split_k: v.knobs.split_k_capacity,
                    };
                    let plan = v.engine_plan(counts).unwrap();
                    let base_plan = base
                        .engine_plan(RuntimeCounts {
                            m: tokens,
                            batch: 0,
                            context: 0,
                            routed_rows: base.query.m_capacity,
                            split_k: base.knobs.split_k_capacity,
                        })
                        .unwrap();
                    let active = |experts: u32| (10 * tokens).min(experts);
                    assert_eq!(
                        plan.grid[0] / active(local),
                        base_plan.grid[0] / active(128),
                        "{}",
                        v.symbol
                    );
                }
                let built = entry(&specs, &v.symbol);
                assert!(built.body.starts_with("chop::moe::grouped::"));
            }
        }
        let routes = sm120::moe::route_catalog()
            .into_iter()
            .filter(|entry| entry.profile == profile)
            .collect::<Vec<_>>();
        // Every TP4 route entry but the router, which is rank-invariant.
        assert_eq!(routes.len(), tp4.len() - 1);
        let route_spec = sm120::moe::route_cubin_spec_for(profile).unwrap();
        assert_eq!(route_spec.name, format!("chop_moe_{tag}_route"));
        for route in &routes {
            let base = tp4
                .iter()
                .find(|entry| entry.symbol == route.symbol.replacen(tag, "qwen38", 1))
                .unwrap();
            assert_eq!(
                (route.tokens_capacity, &route.params, &route.body),
                (base.tokens_capacity, &base.params, &base.body)
            );
            route.instantiation().unwrap();
            let plan = route
                .plan(route.tokens_capacity, route.tokens_capacity)
                .unwrap();
            // A route CTA covers the local experts.
            assert!(plan.block[0] >= local, "{}", route.symbol);
            let pointers = route
                .params
                .iter()
                .filter(|p| matches!(p, kernel_build::EntryParam::DevicePointer { .. }))
                .count();
            let takes_rank = pointers < route.params.len();
            let handles = vec![1_u64; pointers];
            assert!(
                route
                    .arguments(&handles, takes_rank.then_some(ranks - 1))
                    .is_ok()
            );
            if takes_rank {
                // Ranks 0 to ranks - 1 own the 512 experts.
                assert!(route.arguments(&handles, Some(ranks)).is_err());
            }
            assert_eq!(
                entry(std::slice::from_ref(&route_spec), &route.symbol).params,
                route.params
            );
        }
    }
}

#[test]
fn dense_fragment_gemms_at_each_rank_count() {
    let catalog = gemm::catalog(model_qwen38_contract::sm120::catalogs());
    for ranks in OTHER_RANKS {
        let profile = model_qwen38_contract::sm120::config::profile(ranks).unwrap();
        let shape = Qwen38Shape { ranks };
        let tag = profile.tag();
        let specs = gemm::cubin_specs(model_qwen38_contract::sm120::catalogs(), Some(profile));
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, format!("gemm_{tag}_sm120"));
        let shared = shape.shared_padded();
        let mut expected = vec![];
        for m in DECODE_BUCKETS {
            expected.push((format!("head_m{m}"), shape.vocab_shard(), 2560, false));
            expected.push((format!("head_f32_m{m}"), shape.vocab_shard(), 2560, false));
            expected.push((format!("shared_gate_up_m{m}"), 2 * shared, 2560, true));
            expected.push((format!("shared_down_m{m}"), 2560, shared, false));
        }
        for (name, n, k) in [
            ("lin_qkv_prefill", shape.linear_qkv(), 2560),
            ("lin_z_prefill", shape.linear_value(), 2560),
            ("out_prefill", 2560, shape.linear_value()),
            ("attn_qkv_prefill", shape.attn_row(), 2560),
            ("head_prefill", shape.vocab_shard(), 2560),
            ("head_f32_prefill", shape.vocab_shard(), 2560),
        ] {
            expected.push((name.into(), n, k, false));
        }
        expected.push(("shared_gate_up_prefill".into(), 2 * shared, 2560, true));
        expected.push(("shared_down_prefill".into(), 2560, shared, false));
        let entries = catalog
            .iter()
            .filter(|v| v.profile == profile)
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), expected.len());
        assert_eq!(specs[0].entries.len(), expected.len());
        for (name, n, k, swiglu) in expected {
            let symbol = format!("chop_gemm_{tag}_{name}_v1");
            let base = catalog
                .iter()
                .find(|v| v.symbol == symbol.replacen(tag, "qwen38", 1))
                .unwrap();
            let variant = entries.iter().find(|v| v.symbol == symbol).unwrap();
            let (query, tp4) = (gemm_query(&variant.key), gemm_query(&base.key));
            assert_eq!((query.n, query.k), (n, k), "{symbol}");
            assert_eq!(query.epilogues.swiglu.is_some(), swiglu, "{symbol}");
            assert_eq!(
                (query.m_capacity, query.output, query.weight),
                (tp4.m_capacity, tp4.output, tp4.weight)
            );
            assert!(gemm::fragment_weight_enabled(&variant.key), "{symbol}");
            let split_k = gemm::choose_split_k(&variant.key).unwrap();
            gemm::plan(
                &variant.key,
                RuntimeCounts {
                    m: query.m_capacity,
                    batch: query.m_capacity,
                    context: 0,
                    routed_rows: 0,
                    split_k,
                },
            )
            .unwrap_or_else(|error| panic!("{symbol}: {error}"));
            entry(&specs, &symbol);
        }
    }
}

#[test]
fn qwen_owned_gemms_at_each_rank_count() {
    use gemm::fused::{Buffers, Kind};
    let catalog = gemm::fused::catalog(model_qwen38_contract::sm120::catalogs());
    for ranks in OTHER_RANKS {
        let profile = model_qwen38_contract::sm120::config::profile(ranks).unwrap();
        let shape = Qwen38Shape { ranks };
        let tag = profile.tag();
        let specs =
            gemm::fused::cubin_specs(model_qwen38_contract::sm120::catalogs(), Some(profile));
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, format!("gemm_{tag}_own_sm120"));
        let heads = shape.linear_value_heads();
        let shared = shape.shared_padded();
        let mut expected: BTreeMap<String, (Vec<u32>, u32, bool)> = BTreeMap::new();
        for m in DECODE_BUCKETS {
            for (name, segments, k, swiglu) in [
                (
                    "lin_in",
                    vec![shape.linear_qkv(), shape.linear_value(), heads, heads],
                    2560,
                    false,
                ),
                (
                    "attn_in",
                    vec![shape.q_proj_rows(), 256, 256, 640],
                    2560,
                    false,
                ),
                ("out", vec![2560], shape.linear_value(), false),
                ("shared_gate_up", vec![2 * shared], 2560, true),
                ("shared_down", vec![2560], shared, false),
            ] {
                expected.insert(format!("{name}_c{m}"), (segments, k, swiglu));
            }
        }
        expected.insert("lin_ab".into(), (vec![heads, heads], 2560, false));
        let entries = catalog
            .iter()
            .filter(|e| e.profile == profile)
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), expected.len());
        assert_eq!(specs[0].entries.len(), expected.len());
        for (name, (segments, k, swiglu)) in expected {
            let symbol = format!("chop_gemm_{tag}_{name}_v1");
            let e = entries.iter().find(|e| e.symbol == symbol).unwrap();
            let base = catalog
                .iter()
                .find(|b| b.symbol == symbol.replacen(tag, "qwen38", 1))
                .unwrap();
            assert_eq!(e.segments, segments, "{symbol}");
            assert_eq!(e.query.k, k, "{symbol}");
            assert_eq!(e.query.n, segments.iter().sum::<u32>());
            assert_eq!(
                e.query.epilogues.swiglu,
                swiglu.then_some(SwigluColumns::ConcatenatedHalves)
            );
            assert_eq!(
                (e.query.m_capacity, e.query.output),
                (base.query.m_capacity, base.query.output)
            );
            e.contract().unwrap();
            for live in [1, e.query.m_capacity] {
                let plan = e.plan(live).unwrap();
                assert!(plan.dynamic_shared_bytes + 1152 <= 101_376, "{symbol}");
            }
            e.workspace_capacity().unwrap();
            entry(&specs, &symbol);
        }
        // lin_ab writes a then b at columns [0, heads) and [heads, 2 heads)
        // of a row padded to a multiple of 8.
        let ab = entries
            .iter()
            .find(|e| e.symbol == format!("chop_gemm_{tag}_lin_ab_v1"))
            .unwrap();
        assert_eq!(
            ab.kind,
            Kind::Skinny {
                rows: heads,
                weights: 2,
                prefill_splits: 5
            }
        );
        let body = &entry(&specs, &ab.symbol).body;
        assert!(body.contains(&format!("run_skinny<2560, {heads}, 2>")));
        let buffers = |stride| Buffers {
            weights: [1, 2, 0, 0],
            activation: 3,
            outputs: [4, 0, 0, 0],
            output_strides: [stride, 0, 0, 0],
            partial: 5,
            semaphores: 6,
        };
        assert!(ab.arguments(5, buffers(shape.lin_ab_stride())).is_ok());
        assert!(ab.arguments(5, buffers(2 * heads - 2)).is_err());
        // A split of one token tile: every strip of both weights, 32 tokens.
        let strips = 2 * heads.div_ceil(16);
        let one = ab.plan(1).unwrap();
        assert_eq!(
            one.workspace_bytes,
            u64::from(one.grid[0] * strips * 16 * 32 * 4)
        );
    }
}

#[test]
fn ngram_rows_at_each_rank_count() {
    let catalog = model_qwen38_contract::sm120::row::hyper_connection_rows();
    for ranks in OTHER_RANKS {
        let profile = model_qwen38_contract::sm120::config::profile(ranks).unwrap();
        let tag = profile.tag();
        let local = Qwen38Shape { ranks }.ngram_heads();
        let specs = kernels_sm120::row::hyper_connection::cubin_specs(
            model_qwen38_contract::sm120::catalogs(),
            Some(profile),
        );
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, format!("row_{tag}"));
        let entries = catalog
            .iter()
            .filter(|e| e.profile == profile)
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 2);
        for (name, template) in [("ngram", "ngram<"), ("ngram_decode", "ngram_decode<")] {
            let symbol = format!("chop_row_{tag}_{name}_v1");
            let e = entries.iter().find(|e| e.symbol == symbol).unwrap();
            let base = catalog
                .iter()
                .find(|b| b.symbol == format!("chop_row_qwen38_{name}_v1"))
                .unwrap();
            let StaticQuery::Ngram(NgramQuery::HashGather { local_heads, .. }) = e.query else {
                panic!("{symbol} is not an n-gram row");
            };
            assert_eq!(local_heads, local);
            assert_eq!(e.params, base.params);
            e.contract().unwrap();
            let plan = e.plan(32, 1).unwrap();
            assert_eq!(plan.grid, [32, local, 1]);
            let built = entry(&specs, &symbol);
            assert_eq!(built.params, e.params);
            assert!(built.body.contains(&format!("{template}{local}>(")));
        }
    }
}
