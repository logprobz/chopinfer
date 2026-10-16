//! MiMo's per-rank kernel shapes at 2 ranks, over its catalogs
//! (`sm120::catalogs`): every entry the TP2 bundles select exists
//! in its own profile, and the TP4 profile keeps exactly its own entries.

use kernel_core::contract::{
    ArgmaxLayout, DType, RowQuery, RuntimeCounts, StaticQuery, WeightFormat,
};
use kernels_sm120::{attention, gemm, moe, row};

use model_mimo26_contract::sm120;

fn counts(m: u32) -> RuntimeCounts {
    RuntimeCounts {
        m,
        batch: m,
        context: 0,
        routed_rows: 0,
        split_k: 1,
    }
}

#[test]
fn tp4_profile_holds_no_other_topology_entries() {
    let tp4 = model_mimo26_contract::sm120::config::PROFILE;
    for symbol in gemm::catalog(sm120::catalogs())
        .iter()
        .filter(|v| v.profile == tp4)
        .map(|v| v.symbol.clone())
        .chain(
            row::catalog(model_mimo26_contract::sm120::catalogs())
                .iter()
                .filter(|r| r.profile == tp4)
                .map(|r| r.symbol.clone()),
        )
        .chain(
            moe::catalog(sm120::catalogs())
                .iter()
                .filter(|v| v.profile == tp4)
                .map(|v| v.symbol.clone()),
        )
        .chain(
            attention::catalog(model_mimo26_contract::sm120::catalogs())
                .iter()
                .filter(|v| v.profile == tp4)
                .map(|v| v.name.to_owned()),
        )
    {
        assert!(!symbol.contains("_tp2"), "{symbol}");
    }
}

#[test]
fn every_rank_gemm_shape_plans_in_its_profile() {
    let catalog = gemm::catalog(sm120::catalogs());
    for ranks in sm120::gemm::OTHER_RANKS {
        let profile = model_mimo26_contract::sm120::config::profile(ranks).unwrap();
        assert_eq!(
            profile.ranks_of(model_mimo26_contract::sm120::config::MODEL),
            Some(ranks)
        );
        for (name, n, k, weight, ..) in sm120::gemm::rank_shapes(ranks) {
            for bucket in ["m1", "m2", "m4", "m8", "m16", "m32", "prefill"] {
                let symbol = format!("chop_gemm_mimo_tp{ranks}_{name}_{bucket}_v1");
                let variant = catalog
                    .iter()
                    .find(|v| v.symbol == symbol)
                    .unwrap_or_else(|| panic!("missing {symbol}"));
                assert_eq!(variant.profile, profile);
                let StaticQuery::Gemm(query) = variant.key.query else {
                    panic!("{symbol} is not GEMM");
                };
                assert_eq!((query.n, query.k, query.weight), (n, k, weight), "{symbol}");
                assert!(gemm::fragment_weight_enabled(&variant.key), "{symbol}");
                let m = query.m_capacity.min(32);
                let split_k = gemm::choose_split_k(&variant.key).unwrap();
                gemm::plan(
                    &variant.key,
                    RuntimeCounts {
                        split_k,
                        ..counts(m)
                    },
                )
                .unwrap_or_else(|error| panic!("{symbol}: {error}"));
            }
        }
        // Regrouped QKV rows carry 64-row scales; nothing else does.
        let qkv = sm120::gemm::rank_shapes(ranks)
            .into_iter()
            .filter(|shape| matches!(shape.3, WeightFormat::Fp8Rows64x128(_)))
            .map(|shape| (shape.0, shape.1))
            .collect::<Vec<_>>();
        let expected = [("qkv_ga", 6784), ("qkv_swa", 7424)];
        assert_eq!(qkv, expected);
    }
}

#[test]
fn rows64_probe_mirrors_the_tp4_qkv_entries() {
    let catalog = gemm::catalog(sm120::catalogs());
    if let Err(error) = sm120::geometry::require_base_geometries("kernel-lab") {
        assert!(
            error.contains("kernel-lab refuses fixture geometries"),
            "{error}"
        );
        assert!(std::panic::catch_unwind(sm120::lab::rows64_probe_catalog).is_err());
        assert!(std::panic::catch_unwind(sm120::lab::rows64_probe_cubin_spec).is_err());
        return;
    }
    let probes = sm120::lab::rows64_probe_catalog();
    assert_eq!(probes.len(), 2 * 7);
    for probe in probes {
        let original = catalog
            .iter()
            .find(|v| v.symbol == probe.symbol.replace("_rows64_v1", "_v1"))
            .unwrap();
        let (StaticQuery::Gemm(a), StaticQuery::Gemm(b)) =
            (probe.key.query.clone(), original.key.query.clone())
        else {
            panic!("probe is not GEMM");
        };
        assert!(matches!(a.weight, WeightFormat::Fp8Rows64x128(_)));
        assert!(matches!(b.weight, WeightFormat::Fp8Block128x128(_)));
        assert_eq!((a.m_capacity, a.n, a.k), (b.m_capacity, b.n, b.k));
        assert_eq!(probe.key.knobs, original.key.knobs);
    }
    assert!(sm120::lab::rows64_probe_cubin_spec().entries.len() == 14);
}

#[test]
fn rank_attention_splits_heads_into_legal_groups() {
    let catalog = attention::catalog(model_mimo26_contract::sm120::catalogs());
    for ranks in sm120::gemm::OTHER_RANKS {
        let profile = model_mimo26_contract::sm120::config::profile(ranks).unwrap();
        let variants = catalog
            .iter()
            .filter(|v| v.profile == profile)
            .collect::<Vec<_>>();
        assert_eq!(variants.len(), 10);
        for variant in variants {
            let kv_total = if variant.query.window_left.is_none() {
                4
            } else {
                8
            };
            assert_eq!(variant.query.q_heads, 64 / ranks, "{}", variant.name);
            assert_eq!(
                variant.query.kv_heads,
                (kv_total / ranks).max(1),
                "{}",
                variant.name
            );
            variant.instantiation().unwrap();
        }
    }
}

#[test]
fn rank_experts_split_the_intermediate() {
    for ranks in sm120::gemm::OTHER_RANKS {
        let profile = model_mimo26_contract::sm120::config::profile(ranks).unwrap();
        let variants = moe::catalog(sm120::catalogs())
            .into_iter()
            .filter(|v| v.profile == profile)
            .collect::<Vec<_>>();
        // Five decode buckets and prefill per projection, plus the gate/up
        // exact entries (one-token reduction order) of verify buckets 2, 4
        // and 8; down already runs one K split at every decode bucket.
        let exact = variants.iter().filter(|v| v.exact).collect::<Vec<_>>();
        assert_eq!(variants.len() - exact.len(), 2 * 6);
        assert_eq!(
            exact
                .iter()
                .map(|v| (v.projection, v.tokens_capacity))
                .collect::<Vec<_>>(),
            [
                (moe::Projection::GateUp, 2),
                (moe::Projection::GateUp, 4),
                (moe::Projection::GateUp, 8),
            ]
        );
        let intermediate = 2048 / ranks;
        for v in &variants {
            let (n, k) = match v.projection {
                moe::Projection::GateUp => (2 * intermediate, 4096),
                moe::Projection::Down => (4096, intermediate),
            };
            assert_eq!((v.query.n, v.query.k), (n, k), "{}", v.symbol);
            assert_eq!(v.path, moe::InputPath::A16Gather);
            assert!(v.fragment_weights());
            assert!(v.symbol.contains(&format!("mimo_tp{ranks}_")));
            kernel_core::contract::Validate::validate(&v.instantiation()).unwrap();
        }
        assert!(
            variants
                .iter()
                .any(|v| v.phase == moe::Phase::Prefill && v.fragment_prefill())
        );
    }
}

#[test]
fn rank_rows_follow_heads_and_vocabulary_shards() {
    let catalog = row::catalog(model_mimo26_contract::sm120::catalogs());
    for ranks in sm120::gemm::OTHER_RANKS {
        let profile = model_mimo26_contract::sm120::config::profile(ranks).unwrap();
        let find = |name: &str| {
            let symbol = format!("chop_row_{name}_v1");
            catalog
                .iter()
                .find(|r| r.symbol == symbol)
                .unwrap_or_else(|| panic!("missing {symbol}"))
        };
        let tag = format!("tp{ranks}");
        let partial = find(&format!("dflash_{tag}_argmax_partial"));
        assert_eq!(partial.profile, profile);
        assert_eq!(
            partial.query,
            RowQuery::ShardedArgmaxPartial {
                vocab_per_rank: 152_576 / ranks,
                ranks,
                logits: DType::Bf16,
            }
        );
        let fin = find(&format!("dflash_{tag}_argmax_final"));
        assert_eq!(
            fin.query,
            RowQuery::ShardedArgmaxFinal {
                ranks,
                pairs: ArgmaxLayout::RowMajor,
            }
        );
        for (name, heads) in [("q", 64 / ranks), ("k", 8 / ranks)] {
            let plan = find(&format!("dflash_{tag}_{name}_norm"))
                .plan(2, counts(2))
                .unwrap();
            assert_eq!(plan.grid, [2, heads, 1]);
        }
        let swa = (8 / ranks).max(1);
        let ring = find(&format!("ring_write_mimo_{tag}_bf16_pos64"));
        assert_eq!(ring.plan(2, counts(2)).unwrap().grid, [2, swa, 1]);
        for base in ["ga_bf16", "ga_fp8", "swa_decode", "swa_prefill"] {
            let symbol = if base.starts_with("ga") {
                format!("rope_mimo_{tag}_{base}_computed_p64_pos64")
            } else {
                format!("rope_mimo_{tag}_{base}_computed_p32_pos64")
            };
            let rope = find(&symbol);
            assert_eq!(rope.profile, profile);
            let kv = if base.starts_with("ga") {
                (4 / ranks).max(1)
            } else {
                swa
            };
            assert_eq!(rope.width, 64 / ranks * 192 + kv * 320, "{symbol}");
            // Small batches run a block per Q head, large ones a block per
            // four; either way every head of the shard gets RoPE.
            assert_eq!(
                rope.plan(2, counts(2)).unwrap().grid[1],
                64 / ranks,
                "{symbol}"
            );
            assert_eq!(
                rope.plan(512, counts(512)).unwrap().grid[1],
                16 / ranks,
                "{symbol}"
            );
        }
        assert_eq!(
            row::cubin_specs(model_mimo26_contract::sm120::catalogs(), Some(profile))[0]
                .entries
                .len(),
            catalog.iter().filter(|r| r.profile == profile).count()
        );
    }
}
