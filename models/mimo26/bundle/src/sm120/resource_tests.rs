//! Compare native allocation with direct engine plans, independent of the shared selectors.
use crate::topology::Tp;
use bundle_runtime::geometry::Items;
use kernel_core::contract::{
    AttentionPhase, CommQuery, DType, RmsNormEpilogue, RuntimeCounts, StaticQuery,
};

pub(crate) fn planner_maximum(tp: Tp, items: Items) -> u64 {
    let own = model_mimo26_contract::sm120::profile(tp);
    let mut sizes = Vec::new();
    let attention = kernels_sm120::attention::catalog(model_mimo26_contract::sm120::catalogs())
        .into_iter()
        .filter(|entry| entry.profile == own)
        .collect::<Vec<_>>();
    assert_eq!(
        attention
            .iter()
            .map(|entry| entry.instantiation().unwrap().symbol)
            .collect::<Vec<_>>(),
        crate::sm120::launch::attention::mimo_variants(tp)
            .iter()
            .map(|entry| entry.instantiation().unwrap().symbol)
            .collect::<Vec<_>>()
    );
    for entry in attention {
        use kernels_sm120::attention::{Buffers, CacheLayout};
        let slots = if entry.layout == CacheLayout::Paged {
            32_768
        } else {
            32
        };
        let (k_map, v_map) = entry.cache_maps(slots, 1, 2).unwrap();
        let element = if entry.query.kv_dtype == DType::Bf16 {
            2
        } else {
            1
        };
        let stride = |dim| {
            u64::from(entry.query.page_size)
                * u64::from(entry.query.kv_heads)
                * u64::from(dim)
                * element
        };
        let buffers = Buffers {
            q: 3,
            k_cache: 1,
            v_cache: 2,
            k_page_stride_bytes: stride(entry.query.qk_dim),
            v_page_stride_bytes: stride(entry.query.v_dim),
            k_map: Some(k_map),
            v_map: Some(v_map),
            block_tables: Some(4),
            slot_ids: Some(5),
            seq_lens: Some(6),
            q_start: Some(7),
            q_len: Some(8),
            prefix_len: Some(9),
            live_batch: 10,
            k_new: Some(11),
            v_new: Some(12),
            sink: Some(13),
            output: 14,
            workspace: 15,
            k_scale: 1.0,
            v_scale: 1.0,
        };
        let prepared = entry
            .with_batch_capacity(items.capacity().max(entry.query.max_batch_capacity))
            .unwrap();
        for bucket in std::iter::once(entry.query.max_batch_capacity).chain(
            items
                .buckets()
                .filter(|&bucket| bucket > entry.query.max_batch_capacity),
        ) {
            let wide = bucket > entry.query.max_batch_capacity;
            let counts = RuntimeCounts {
                m: match entry.query.phase {
                    AttentionPhase::Prefill => entry.query_capacity,
                    AttentionPhase::Verify if wide => (bucket * entry.query_capacity)
                        .min(kernels_sm120::attention::MAX_VERIFY_TOKENS),
                    _ => 1,
                },
                batch: if wide { bucket } else { 1 },
                context: if entry.query.phase == AttentionPhase::Prefill {
                    entry.query.max_context_capacity
                } else {
                    1
                },
                routed_rows: 8,
                split_k: 1,
            };
            let plan = crate::sm120::launch::attention::plan_for_bucket(
                prepared,
                bucket,
                counts,
                buffers.clone(),
            )
            .unwrap();
            sizes.push(plan.partial.workspace_bytes);
            sizes.push(plan.merge.workspace_bytes);
        }
    }
    let rank_invariant = |profile| {
        profile == own
            || profile
                == model_mimo26_contract::sm120::profile(Tp::of_geometry(tp.geometry, 4).unwrap())
    };
    let dense = kernels_sm120::gemm::catalog(model_mimo26_contract::sm120::catalogs())
        .into_iter()
        .filter(|entry| rank_invariant(entry.profile))
        .collect::<Vec<_>>();
    assert_eq!(
        dense
            .iter()
            .map(|entry| entry.instantiation.symbol.as_str())
            .collect::<Vec<_>>(),
        crate::sm120::launch::dense::entries_for(tp)
            .map(|entry| entry.symbol.as_str())
            .collect::<Vec<_>>()
    );
    for entry in dense {
        let key = entry.instantiation.key;
        let StaticQuery::Gemm(query) = key.query else {
            panic!("GEMM catalog entry")
        };
        let counts = RuntimeCounts {
            m: query.m_capacity,
            batch: 1,
            context: 1,
            routed_rows: 8,
            split_k: kernels_sm120::gemm::required_split_k(&key).unwrap_or(1),
        };
        sizes.push(
            kernels_sm120::gemm::plan(&key, counts)
                .unwrap()
                .workspace_bytes,
        );
    }
    let experts = kernels_sm120::moe::catalog(model_mimo26_contract::sm120::catalogs())
        .into_iter()
        .filter(|entry| entry.profile == own)
        .collect::<Vec<_>>();
    assert_eq!(
        experts
            .iter()
            .map(|entry| entry.symbol.as_str())
            .collect::<Vec<_>>(),
        crate::sm120::launch::moe::mimo_variants(tp)
            .iter()
            .map(|entry| entry.symbol.as_str())
            .collect::<Vec<_>>()
    );
    for entry in experts {
        let counts = RuntimeCounts {
            m: entry.tokens_capacity,
            batch: 1,
            context: 1,
            routed_rows: entry.query.m_capacity,
            split_k: entry.knobs.split_k_capacity,
        };
        sizes.push(entry.plan(counts).unwrap().workspace_bytes);
        if let Some(reduction) = entry.reduction_plan(counts).unwrap() {
            sizes.push(reduction.workspace_bytes);
        }
    }
    let rows = kernels_sm120::row::catalog(model_mimo26_contract::sm120::catalogs())
        .into_iter()
        .filter(|entry| rank_invariant(entry.profile))
        .collect::<Vec<_>>();
    assert_eq!(
        rows.iter()
            .map(|entry| entry.symbol.as_str())
            .collect::<Vec<_>>(),
        crate::sm120::launch::row::entries_for(tp)
            .map(|entry| entry.symbol.as_str())
            .collect::<Vec<_>>()
    );
    for entry in rows {
        let counts = RuntimeCounts {
            m: 8192,
            batch: 1,
            context: 1,
            routed_rows: 65_536,
            split_k: 1,
        };
        sizes.push(entry.plan(8192, counts).unwrap().workspace_bytes);
    }
    for rows in [32, 8192] {
        sizes.push(
            kernels_sm120::comm::plan(
                &CommQuery::AllReduce {
                    ranks: tp.ranks(),
                    elements: u64::from(rows) * tp.geometry.hidden() as u64,
                    dtype: DType::Bf16,
                    fused_residual: true,
                    fused_rmsnorm: Some(RmsNormEpilogue {
                        kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
                        width: tp.geometry.hidden(),
                        eps: tp.geometry.eps() as f32,
                    }),
                },
                rows,
            )
            .unwrap()
            .workspace_bytes,
        );
    }
    sizes.push(
        kernels_sm120::comm::plan(
            &CommQuery::AllGather {
                ranks: tp.ranks(),
                elements_per_rank: 32 * tp.vocab(),
                dtype: DType::Bf16,
            },
            32,
        )
        .unwrap()
        .workspace_bytes,
    );
    let maximum = sizes.iter().copied().max().expect("nonempty model catalog");
    let declared = model_mimo26_contract::sm120::workspace::workspace_bytes(tp, items).unwrap();
    assert_eq!(maximum, declared, "TP{} items={}", tp.ranks(), items.get());
    assert!(sizes.into_iter().all(|size| size <= declared));
    maximum
}

#[test]
fn every_catalog_plan_fits_the_exact_workspace_maximum() {
    for tp in crate::topology::all() {
        for count in [1, 17, 32, 33, 64, 100, 256] {
            assert!(planner_maximum(tp, Items::new(count).unwrap()) > 0);
        }
    }
}
