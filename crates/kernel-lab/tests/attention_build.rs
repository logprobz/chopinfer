//! CPU-only compilation and launch-plan checks for every model's attention
//! catalog (`kernel_lab::catalogs`).

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use kernel_build::{BuildOptions, Toolchain, build};
use kernel_core::contract::{Argument, AttentionPhase, NumericalContract, RuntimeCounts};
use kernels_sm120::attention::{
    Buffers, CacheLayout, GatedRopeInputs, RopeInputs, Variant, catalog, gated, tuning,
};

fn scratch() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("attention-build-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}

/// The variant `name` of `variants`.
fn named(variants: &[Variant], name: &str) -> Variant {
    *variants
        .iter()
        .find(|variant| variant.name == name)
        .unwrap_or_else(|| panic!("no attention variant {name}"))
}

fn buffers(variant: Variant) -> Buffers {
    let (k_map, v_map) = variant.cache_maps(16384, 2, 3).unwrap();
    Buffers {
        q: 1,
        k_cache: 2,
        v_cache: 3,
        k_page_stride_bytes: k_map.strides_bytes[3],
        v_page_stride_bytes: v_map.strides_bytes[3],
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
        k_scale: 0.5,
        v_scale: 0.25,
    }
}

#[test]
fn planner_covers_catalog_and_capacity_edges() {
    let table = tuning();
    assert_eq!((table.compute_capability, table.sm_count), (120, 188));
    assert_eq!(table.prefill_key_tile, 32);
    assert_eq!(table.prefill_rows, 128);
    assert_eq!(table.prefill_ctas, 188);
    assert_eq!(table.decode_buckets.len(), 9);
    let variants: Vec<_> = catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|v| !kernel_lab::lab_shapes::PROFILES.contains(&v.profile))
        .collect();
    assert_eq!(variants.len(), 26 + 10);
    assert_ne!(
        named(&variants, "std128_bf16_prefill")
            .instantiation()
            .unwrap()
            .key
            .query,
        named(&variants, "std128_softcap_prefill")
            .instantiation()
            .unwrap()
            .key
            .query
    );
    assert_eq!(
        named(&variants, "mimo_swa_prefill")
            .query
            .numerical_contract_id()
            .unwrap()
            .as_str(),
        "attention.ring-kv-bf16.prefill.v1"
    );
    assert_eq!(
        named(&variants, "mimo_dflash_block")
            .query
            .numerical_contract_id()
            .unwrap()
            .as_str(),
        "attention.ring-kv-bf16.verify-bidirectional.v1"
    );
    for variant in variants {
        variant.instantiation().unwrap();
        let verify = variant.query.phase == AttentionPhase::Verify;
        let live = RuntimeCounts {
            m: match variant.query.phase {
                AttentionPhase::Decode => 1,
                AttentionPhase::Verify => 256,
                AttentionPhase::Prefill => 8192,
            },
            batch: 32,
            context: variant.query.max_context_capacity,
            routed_rows: 0,
            split_k: 1,
        };
        let planned = variant.plan(live, buffers(variant)).unwrap();
        let prefill = variant.query.phase == AttentionPhase::Prefill;
        if prefill {
            let tokens = 128 / (variant.query.q_heads / variant.query.kv_heads);
            let ctas = 188_u32.div_ceil(variant.query.kv_heads);
            assert_eq!(
                planned.partial.grid,
                [
                    if variant.layout == CacheLayout::Ring {
                        8192 / tokens + 32
                    } else {
                        ctas
                    },
                    variant.query.kv_heads,
                    1
                ]
            );
            if variant.layout == CacheLayout::Paged {
                assert_eq!(planned.partial_args.0[19], Argument::U32(ctas));
            }
            assert_eq!(planned.partial.block[0], 256);
            assert_eq!(planned.merge.grid, [ctas, variant.query.kv_heads, 1]);
            assert_eq!(planned.merge.block[0], 256);
            assert!(planned.partial_symbol.contains("_direct_"));
            assert_eq!(planned.direct_output, planned.split_capacity == 1);
        } else if verify {
            let group = variant.query.q_heads / variant.query.kv_heads;
            assert_eq!(variant.query_capacity, 8);
            assert_eq!(
                planned.partial.grid,
                [
                    188_u32.div_ceil(variant.query.kv_heads),
                    variant.query.kv_heads,
                    1
                ]
            );
            assert_eq!(planned.partial.block[0], 8 * group * 2);
            assert!(planned.partial.dynamic_shared_bytes > 48 * 1024);
            assert!(planned.partial.dynamic_shared_bytes <= 96 * 1024);
            assert_eq!(
                planned.merge.grid,
                [variant.query.v_dim.div_ceil(64), variant.query.q_heads, 256]
            );
            // At 32 items, windows of at most 128 keys are not split; CTAs write output.
            let whole = variant.query.window_left.is_some_and(|left| left < 128);
            assert_eq!(planned.split_capacity, if whole { 1 } else { 16 });
            assert!(planned.partial_symbol.ends_with("_partial_v1"));
            assert_eq!(planned.direct_output, whole);
        } else if variant.layout == CacheLayout::Ring {
            assert_eq!(planned.partial.grid, [32, variant.query.kv_heads, 1]);
            assert_eq!(planned.partial.cluster, [1, 1, 1]);
            assert_eq!(planned.partial.block[0], 256);
            assert_eq!(planned.partial.dynamic_shared_bytes, 128 * (192 + 128) * 2);
            assert!(planned.partial_symbol.ends_with("_direct_v1"));
            assert!(planned.direct_output);
        } else {
            let ctas = 376_u32.div_ceil(variant.query.kv_heads);
            assert_eq!(planned.partial.grid, [ctas, variant.query.kv_heads, 1]);
            assert_eq!(planned.partial.block[0], 128);
            assert_eq!(planned.partial_args.0[17], Argument::U32(ctas));
            assert_eq!(
                planned.merge.grid,
                [variant.query.v_dim.div_ceil(64), variant.query.q_heads, 32]
            );
            assert_eq!(planned.merge.block[0], 128);
        }
        assert!(planned.partial.pdl && planned.merge.pdl);
        assert_eq!(
            variant.dynamic_shared_bytes(&planned.partial_symbol),
            Some(planned.partial.dynamic_shared_bytes)
        );
        assert_eq!(variant.dynamic_shared_bytes("chop_attn_unknown_v1"), None);
        let slots: u32 = match variant.query.phase {
            AttentionPhase::Decode => 32,
            AttentionPhase::Verify => 256,
            AttentionPhase::Prefill => 8192,
        };
        // Partials only: a paged prefill merge walks the step's aligned
        // query tiles and needs no split records.
        assert_eq!(
            planned.partial.workspace_bytes,
            u64::from(slots)
                * u64::from(variant.query.q_heads)
                * u64::from(planned.split_capacity)
                * u64::from(variant.query.v_dim + 2)
                * 4
        );
        let map_params = if variant.query.phase == AttentionPhase::Prefill {
            2
        } else {
            0
        };
        // Paged decode and verify take the bucket's `split_units` right
        // after `split_target` (partial argument 18, merge argument 10).
        let paged_decode =
            variant.query.phase == AttentionPhase::Decode && variant.layout == CacheLayout::Paged;
        let units = usize::from(paged_decode || verify);
        assert_eq!(
            planned.partial_args.0.len(),
            23 + map_params
                + units
                + if prefill || verify || variant.layout == CacheLayout::Ring {
                    2
                } else {
                    0
                }
        );
        assert_eq!(planned.merge_args.0.len(), 11 + units);
        if paged_decode || verify {
            let tuned = table
                .decode_buckets
                .iter()
                .find(|bucket| bucket.batch == 32)
                .unwrap()
                .split_units;
            assert_eq!(planned.partial_args.0[18], Argument::U32(tuned));
            assert_eq!(planned.merge_args.0[10], Argument::U32(tuned));
        }
        assert_eq!(
            planned.partial_args.0[21 + map_params + units],
            Argument::U32(0.5f32.to_bits())
        );
        assert_eq!(
            planned.partial_args.0[22 + map_params + units],
            Argument::U32(0.25f32.to_bits())
        );
        if variant.layout == CacheLayout::Ring && prefill {
            assert_eq!(planned.split_capacity, 1);
            assert!(planned.direct_output);
        }
        assert!(!planned.writes_cache);
        let rope = RopeInputs {
            qkv: 16,
            theta: 1e4,
            v_multiplier: 0.707,
        };
        let folded = variant.plan_folded(32, live, buffers(variant), rope);
        let gated_rope = GatedRopeInputs {
            qkv: 16,
            q_norm: 17,
            k_norm: 18,
            rope_positions: 19,
            cos_sin: 20,
        };
        let gated_folded = variant.plan_folded_gated(32, live, buffers(variant), gated_rope);
        if let Some(fold) = variant.gated_fold {
            assert!(folded.is_err());
            let folded = gated_folded.unwrap();
            assert!(folded.writes_cache);
            assert_eq!(folded.partial_symbol, variant.rope_symbol());
            assert_eq!(folded.partial, planned.partial);
            let args = &folded.partial_args.0;
            assert_eq!(args.len(), planned.partial_args.0.len() + 5);
            assert_eq!(args[0], Argument::DeviceBuffer(16));
            assert_eq!(args[1..args.len() - 5], planned.partial_args.0[1..]);
            assert_eq!(
                args[args.len() - 5..],
                [
                    Argument::DeviceBuffer(17),
                    Argument::DeviceBuffer(18),
                    Argument::DeviceBuffer(19),
                    Argument::DeviceBuffer(20),
                    Argument::U32(fold.eps.to_bits()),
                ]
            );
            assert_eq!(
                variant.dynamic_shared_bytes(&folded.partial_symbol),
                Some(0)
            );
        } else if variant.rope_fold.is_some() {
            assert!(gated_folded.is_err());
            let folded = folded.unwrap();
            let table = variant
                .plan_folded_table(32, live, buffers(variant), rope, 91)
                .unwrap();
            assert_eq!(table.partial_symbol, variant.rope_table_symbol());
            assert_eq!(table.partial, folded.partial);
            assert_eq!(
                &table.partial_args.0[..folded.partial_args.0.len()],
                &folded.partial_args.0
            );
            assert_eq!(
                table.partial_args.0.last(),
                Some(&Argument::DeviceBuffer(91))
            );
            assert_eq!(
                variant.dynamic_shared_bytes(&table.partial_symbol),
                Some(table.partial.dynamic_shared_bytes)
            );
            assert!(folded.writes_cache);
            assert_eq!(folded.partial_symbol, variant.rope_symbol());
            assert_eq!(folded.partial, planned.partial);
            assert_eq!(folded.direct_output, planned.direct_output);
            let args = &folded.partial_args.0;
            assert_eq!(args.len(), planned.partial_args.0.len() + 2);
            assert_eq!(args[0], Argument::DeviceBuffer(16));
            assert_eq!(args[1..args.len() - 2], planned.partial_args.0[1..]);
            assert_eq!(args[args.len() - 2], Argument::U32(1e4_f32.to_bits()));
            assert_eq!(args[args.len() - 1], Argument::U32(0.707_f32.to_bits()));
            assert_eq!(
                variant.dynamic_shared_bytes(&folded.partial_symbol),
                Some(folded.partial.dynamic_shared_bytes)
            );
            for theta in [1.0, f32::NAN] {
                let rope = RopeInputs { theta, ..rope };
                assert!(
                    variant
                        .plan_folded(32, live, buffers(variant), rope)
                        .is_err()
                );
            }
        } else {
            assert!(folded.is_err());
            assert!(gated_folded.is_err());
        }
        let mut too_large = live;
        too_large.context += 1;
        assert!(variant.plan(too_large, buffers(variant)).is_err());
        if verify {
            for (bucket, mut splits) in [(1, 256), (2, 128), (4, 64), (8, 32), (16, 16), (32, 16)] {
                if bucket == 32 && variant.query.window_left.is_some_and(|left| left < 128) {
                    splits = 1;
                }
                let mut counts = live;
                counts.batch = bucket;
                counts.m = bucket * 8;
                let bucket_plan = variant
                    .plan_for_bucket(bucket, counts, buffers(variant))
                    .unwrap();
                assert_eq!(bucket_plan.merge.grid[2], bucket * 8);
                assert_eq!(bucket_plan.split_capacity, splits);
                assert_eq!(
                    bucket_plan.partial.workspace_bytes,
                    u64::from(bucket * 8)
                        * u64::from(variant.query.q_heads)
                        * u64::from(splits)
                        * u64::from(variant.query.v_dim + 2)
                        * 4
                );
                counts.m += 1;
                assert!(
                    variant
                        .plan_for_bucket(bucket, counts, buffers(variant))
                        .is_err()
                );
            }
        }
        if variant.query.phase == AttentionPhase::Decode {
            for bucket in [1, 2, 4, 8, 16, 32] {
                let mut counts = live;
                counts.batch = bucket;
                let bucket_plan = variant
                    .plan_for_bucket(bucket, counts, buffers(variant))
                    .unwrap();
                assert_eq!(bucket_plan.merge.grid[2], bucket);
                assert_eq!(
                    bucket_plan.split_capacity,
                    variant.split_capacity(bucket).unwrap()
                );
                assert!(bucket_plan.split_capacity <= 512);
                assert!(bucket_plan.partial_symbol.ends_with(
                    if variant.layout == CacheLayout::Ring {
                        "_direct_v1"
                    } else {
                        "_partial_v1"
                    }
                ));
                assert_eq!(
                    bucket_plan.merge.block[0],
                    match bucket {
                        1 => 512,
                        2 | 4 => 256,
                        _ => 128,
                    }
                );
            }
        }
    }
}

#[test]
fn batch_capacity_extends_decode_and_verify_to_256_items() {
    use kernels_sm120::attention::{MAX_BATCH_ROWS, MAX_VERIFY_TOKENS, is_batch_bucket};
    assert_eq!((MAX_BATCH_ROWS, MAX_VERIFY_TOKENS), (256, 256));
    assert!(
        [1, 2, 4, 8, 16, 32, 64, 128, 256]
            .into_iter()
            .all(is_batch_bucket)
    );
    assert!(![0, 3, 48, 96, 512].into_iter().any(is_batch_bucket));
    for variant in catalog(kernel_lab::catalogs()) {
        if variant.query.phase == AttentionPhase::Prefill {
            continue;
        }
        let verify = variant.query.phase == AttentionPhase::Verify;
        let live = |batch: u32| RuntimeCounts {
            m: if verify {
                (batch * variant.query_capacity).min(256)
            } else {
                1
            },
            batch,
            context: 32_768,
            routed_rows: 0,
            split_k: 1,
        };
        // The catalog prepares 32 items; wider buckets need a wider capacity.
        assert_eq!(variant.query.max_batch_capacity, 32, "{}", variant.name);
        assert!(
            variant
                .plan_for_bucket(64, live(64), buffers(variant))
                .is_err()
        );
        assert!(variant.with_batch_capacity(0).is_err());
        assert!(variant.with_batch_capacity(257).is_err());
        let mut beyond = variant;
        beyond.query.max_batch_capacity = 512;
        assert!(beyond.validate().is_err());
        assert_eq!(
            variant
                .with_batch_capacity(100)
                .unwrap()
                .query
                .max_batch_capacity,
            128
        );
        let wide = variant.with_batch_capacity(256).unwrap();
        // Buckets of at most 32 items plan exactly as the 32-item variant:
        // the same grids, shared bytes, workspace and arguments.
        for bucket in [1, 2, 4, 8, 16, 32] {
            let narrow = variant
                .plan_for_bucket(bucket, live(bucket), buffers(variant))
                .unwrap();
            let planned = wide
                .plan_for_bucket(bucket, live(bucket), buffers(variant))
                .unwrap();
            assert_eq!(planned.partial, narrow.partial, "{} {bucket}", variant.name);
            assert_eq!(planned.merge, narrow.merge);
            assert_eq!(planned.partial_args.0, narrow.partial_args.0);
            assert_eq!(planned.merge_args.0, narrow.merge_args.0);
            assert_eq!(planned.partial_symbol, narrow.partial_symbol);
        }
        assert_eq!(
            variant.dynamic_shared_bytes(&variant.instantiation().unwrap().symbol),
            Some(
                variant
                    .plan_for_bucket(32, live(32), buffers(variant))
                    .unwrap()
                    .partial
                    .dynamic_shared_bytes
            )
        );
        for bucket in [64, 128, 256] {
            let planned = wide
                .plan_for_bucket(bucket, live(bucket), buffers(variant))
                .unwrap();
            let q_heads = u64::from(variant.query.q_heads);
            let row_bytes = q_heads * u64::from(variant.query.v_dim + 2) * 4;
            if verify {
                let whole = variant.query.window_left.is_some_and(|left| left < 128);
                assert_eq!(planned.split_capacity, if whole { 1 } else { 16 });
                assert_eq!(planned.direct_output, whole);
                assert_eq!(planned.merge.grid[2], 256);
                assert_eq!(
                    planned.partial.workspace_bytes,
                    256 * row_bytes * u64::from(planned.split_capacity)
                );
                let narrow = variant
                    .plan_for_bucket(32, live(32), buffers(variant))
                    .unwrap()
                    .partial
                    .dynamic_shared_bytes;
                assert_eq!(
                    planned.partial.dynamic_shared_bytes,
                    narrow + 28 * (bucket - 32)
                );
                assert!(planned.partial.dynamic_shared_bytes <= 96 * 1024);
                let mut over = live(bucket);
                over.m = 257;
                assert!(
                    wide.plan_for_bucket(bucket, over, buffers(variant))
                        .is_err()
                );
            } else {
                assert_eq!(planned.merge.grid[2], bucket);
                assert_eq!(planned.merge.block[0], 128);
                assert_eq!(planned.split_capacity, wide.split_capacity(bucket).unwrap());
                if variant.layout == CacheLayout::Paged {
                    assert_eq!(
                        planned.partial.workspace_bytes,
                        u64::from(bucket) * row_bytes * u64::from(planned.split_capacity)
                    );
                    // Wide buckets keep the workspace within twice the 32-row one.
                    assert!(
                        planned.partial.workspace_bytes
                            <= 2 * 32 * row_bytes * u64::from(wide.split_capacity(32).unwrap())
                    );
                } else {
                    assert_eq!(planned.partial.grid[0], bucket);
                    assert!(planned.direct_output);
                }
            }
            let mut over = live(bucket);
            over.batch = bucket + 1;
            assert!(
                wide.plan_for_bucket(bucket, over, buffers(variant))
                    .is_err()
            );
        }
        assert_eq!(
            wide.dynamic_shared_bytes(&wide.instantiation().unwrap().symbol),
            Some(
                wide.plan_for_bucket(256, live(256), buffers(variant))
                    .unwrap()
                    .partial
                    .dynamic_shared_bytes
            )
        );
        for bucket in [48, 512] {
            assert!(wide.split_capacity(bucket).is_err());
            assert!(
                wide.plan_for_bucket(bucket, live(32), buffers(variant))
                    .is_err()
            );
        }
    }
}

#[test]
fn attention_planner_refuses_batches_above_256_items() {
    let refused = |refusal: kernel_core::contract::Refusal| {
        assert!(refusal.0.contains("256-item"), "{}", refusal.0);
    };
    for variant in catalog(kernel_lab::catalogs()) {
        if variant.query.phase == AttentionPhase::Prefill {
            continue;
        }
        let live = |batch: u32| RuntimeCounts {
            m: if variant.query.phase == AttentionPhase::Verify {
                (batch * variant.query_capacity).min(256)
            } else {
                1
            },
            batch,
            context: variant.query.max_context_capacity,
            routed_rows: 0,
            split_k: 1,
        };
        let wide = variant.with_batch_capacity(256).unwrap();
        wide.split_capacity(256).unwrap();
        wide.plan_for_bucket(256, live(256), buffers(variant))
            .unwrap();
        for rows in [257, 512] {
            refused(wide.split_capacity(rows).unwrap_err());
            refused(
                wide.plan_for_bucket(rows, live(rows), buffers(variant))
                    .unwrap_err(),
            );
            refused(
                wide.plan_for_bucket(256, live(rows), buffers(variant))
                    .unwrap_err(),
            );
            refused(variant.with_batch_capacity(rows).unwrap_err());
            let mut wider = variant;
            wider.query.max_batch_capacity = rows;
            refused(wider.validate().unwrap_err());
            refused(wider.plan(live(rows), buffers(variant)).unwrap_err());
        }
    }
}

#[test]
fn planner_refuses_mismatched_cache_and_runtime_layouts() {
    let variants = catalog(kernel_lab::catalogs());
    let paged = named(&variants, "mimo_ga_bf16_prefill");
    let ring = named(&variants, "mimo_swa_prefill");
    let live = RuntimeCounts {
        m: 16,
        batch: 1,
        context: 65,
        routed_rows: 0,
        split_k: 1,
    };
    let mut wrong_map = buffers(paged);
    wrong_map.k_map.as_mut().unwrap().box_dims[2] = 8;
    assert!(paged.plan_for_bucket(1, live, wrong_map).is_err());
    let mut strided = buffers(paged);
    let k_block = strided.k_page_stride_bytes;
    let layer_bytes = k_block + strided.v_page_stride_bytes;
    let page_stride = 3 * layer_bytes;
    let (k_map, v_map) = paged
        .cache_maps_with_strides(16_384, 2, 3, page_stride, page_stride)
        .unwrap();
    strided.k_map = Some(k_map);
    strided.v_map = Some(v_map);
    strided.k_page_stride_bytes = page_stride;
    strided.v_page_stride_bytes = page_stride;
    let plan = paged.plan_for_bucket(1, live, strided.clone()).unwrap();
    assert_eq!(plan.partial_args.0[16], Argument::U64(page_stride));
    assert_eq!(plan.partial_args.0[17], Argument::U64(page_stride));
    strided.k_page_stride_bytes = u64::MAX - 15;
    assert!(paged.plan_for_bucket(1, live, strided).is_err());
    assert!(
        paged
            .cache_maps_with_strides(16_384, 2, 3, k_block - 16, layer_bytes)
            .is_err()
    );
    let mut no_chunk = buffers(ring);
    no_chunk.k_new = None;
    assert!(ring.plan_for_bucket(1, live, no_chunk).is_err());
    let mut no_sink = buffers(ring);
    no_sink.sink = None;
    assert!(ring.plan_for_bucket(1, live, no_sink).is_err());
    let mut invalid_page = paged;
    invalid_page.query.page_size = 128;
    assert!(invalid_page.validate().is_err());
    let mut invalid_ring = ring;
    invalid_ring.query.window_left = Some(128);
    assert!(invalid_ring.validate().is_err());
    // Only decode variants of the MiMo head geometry fold RoPE.
    let mut folded_prefill = paged;
    folded_prefill.rope_fold = Some(64);
    assert!(folded_prefill.validate().is_err());
    let mut wide_rotary = named(&variants, "mimo_ga_bf16_decode");
    assert_eq!(wide_rotary.rope_fold, Some(64));
    wide_rotary.rope_fold = Some(128);
    assert!(wide_rotary.validate().is_err());
    assert!(ring.split_capacity(3).is_err());
    // The gated fold serves paged decode at head 128 or 256. Head 128
    // permits groups up to eight; head 256 also permits group 16.
    let qwen_decode = variants
        .iter()
        .copied()
        .find(|v| v.name == "qwen38_bf16_decode")
        .unwrap();
    assert_eq!(
        qwen_decode.gated_fold,
        Some(model_qwen38_contract::sm120::attention::GATED.fold())
    );
    let mut gated_prefill = variants
        .iter()
        .copied()
        .find(|v| v.name == "qwen38_bf16_prefill")
        .unwrap();
    gated_prefill.gated_fold = Some(model_qwen38_contract::sm120::attention::GATED.fold());
    assert!(gated_prefill.validate().is_err());
    let mut both = qwen_decode;
    both.rope_fold = Some(64);
    assert!(both.validate().is_err());
    let mut group12 = qwen_decode;
    group12.query.q_heads = 12;
    assert!(group12.validate().is_err());
}

#[test]
fn qwen38_variants_match_decision_0084() {
    let variants: Vec<_> = catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|v| v.profile == model_qwen38_contract::sm120::config::PROFILE)
        .collect();
    assert_eq!(variants.len(), 6);
    for variant in &variants {
        let q = variant.query;
        assert_eq!((q.q_heads, q.kv_heads, q.qk_dim, q.v_dim), (6, 1, 256, 256));
        assert_eq!((q.page_size, q.max_context_capacity), (64, 262_144));
        assert_eq!(q.softmax_scale, 0.0625);
        assert!(q.causal && q.window_left.is_none() && !q.sink && q.softcap.is_none());
        assert_eq!(
            variant.gated_fold.is_some(),
            q.phase == AttentionPhase::Decode
        );
    }
    // Block-table rows of 4096 pages.
    let decode = named(&variants, "qwen38_bf16_decode");
    let planned = decode
        .plan_for_bucket(
            1,
            RuntimeCounts {
                m: 1,
                batch: 1,
                context: 262_144,
                routed_rows: 0,
                split_k: 1,
            },
            buffers(decode),
        )
        .unwrap();
    assert_eq!(planned.partial_args.0[13], Argument::U32(4096));
    // TP4's rows, then TP2's.
    assert_eq!(
        gated::catalog(kernel_lab::catalogs())
            .iter()
            .filter(|e| !kernel_lab::lab_shapes::PROFILES.contains(&e.profile))
            .count(),
        2 * 3
    );
    let rows = model_qwen38_contract::sm120::attention::gated_catalog(
        model_qwen38_contract::sm120::config::PROFILE,
    );
    assert_eq!(rows.len(), 3);
    for row in &rows {
        row.contract().unwrap();
        assert_eq!(row.plan(8192).unwrap().grid, [8192, 1, 1]);
        assert!(row.plan(8193).is_err());
        assert!(row.plan(0).is_err());
    }
    assert_eq!(
        rows[0].query.numerical_contract_id().unwrap().as_str(),
        "row.qk-gemma-norm-mrope-kv-write-bf16.v1"
    );
    assert_eq!(
        rows[1].query.numerical_contract_id().unwrap().as_str(),
        "row.qk-gemma-norm-mrope-kv-write-fp8.v1"
    );
    assert_eq!(
        rows[2].query.numerical_contract_id().unwrap().as_str(),
        "row.attention-output-sigmoid-gate.v1"
    );
    let cubins = kernels_sm120::attention::cubin_specs(
        kernel_lab::catalogs(),
        Some(model_qwen38_contract::sm120::config::PROFILE),
    );
    assert_eq!(cubins.len(), 1);
    assert_eq!(cubins[0].name, "attention_qwen38_sm120");
    // Six variants of two entries, two decode folds and three rows.
    assert_eq!(cubins[0].entries.len(), 6 * 2 + 2 + 3);
}

#[test]
fn exact_rows_plans_keep_the_one_row_plan() {
    let live = RuntimeCounts {
        m: 8,
        batch: 1,
        context: 32_768,
        routed_rows: 0,
        split_k: 1,
    };
    let mut checked = 0;
    for variant in catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|v| v.has_rows())
    {
        let one = variant.plan_for_bucket(1, live, buffers(variant)).unwrap();
        let rows = variant.plan_rows(8, live, buffers(variant)).unwrap();
        assert_eq!(rows.partial_symbol, variant.rows_symbol());
        assert_eq!(rows.split_capacity, one.split_capacity);
        if variant.layout == CacheLayout::Ring {
            assert!(variant.plan_rows_shared(8, live, buffers(variant)).is_err());
            continue;
        }
        let shared = variant.plan_rows_shared(8, live, buffers(variant)).unwrap();
        assert_eq!(shared.partial_symbol, variant.rows_shared_symbol());
        assert_eq!(shared.merge_symbol, variant.rows_merge_symbol());
        // Row groups (at most one per row) by the one-row plan's CTAs, four
        // rows of four warps each per CTA.
        assert_eq!(
            shared.partial.grid,
            [8, one.partial.grid[0], one.partial.grid[1]]
        );
        assert_eq!(shared.partial.block, [4 * one.partial.block[0], 1, 1]);
        assert_eq!(shared.partial_args, rows.partial_args);
        assert_eq!(shared.merge, rows.merge);
        assert_eq!(shared.partial.workspace_bytes, rows.partial.workspace_bytes);
        assert_eq!(
            Some(shared.partial.dynamic_shared_bytes),
            variant.dynamic_shared_bytes(&variant.rows_shared_symbol())
        );
        assert!(shared.partial.dynamic_shared_bytes <= 96 * 1024);
        assert!(
            variant
                .plan_rows_shared(17, live, buffers(variant))
                .is_err()
        );
        checked += 1;
    }
    assert_eq!(checked, 2 * 2);
}

#[test]
fn every_attention_entry_builds_and_is_audited() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(
        !toolchain.skipped,
        "SM120 compile coverage requires CUDA on Linux"
    );
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../kernels-sm120/cuda");
    let specs = catalog(kernel_lab::catalogs())
        .into_iter()
        .map(|variant| variant.cubin(&root).unwrap())
        .collect::<Vec<_>>();
    let temp = scratch();
    let built = build(
        &toolchain,
        &specs,
        &temp.join("out"),
        &temp.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    assert_eq!(built.len(), specs.len());
    for cubin in built {
        let variant = catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|v| v.name == cubin.name)
            .unwrap();
        let mut expected = std::collections::BTreeSet::from([
            variant.instantiation().unwrap().symbol,
            format!("chop_attn_{}_merge_v1", variant.name),
        ]);
        if variant.rope_fold.is_some() || variant.gated_fold.is_some() {
            expected.insert(variant.rope_symbol());
        }
        if variant.rope_fold.is_some() {
            expected.insert(variant.rope_table_symbol());
        }
        if variant.has_rows() {
            expected.insert(variant.rows_symbol());
            if variant.layout == CacheLayout::Paged {
                expected.insert(variant.rows_shared_symbol());
                expected.insert(variant.rows_merge_symbol());
            }
        }
        assert_eq!(cubin.entries.len(), expected.len());
        assert_eq!(
            cubin
                .entries
                .iter()
                .map(|entry| entry.symbol.clone())
                .collect::<std::collections::BTreeSet<_>>(),
            expected,
        );
        for entry in &cubin.entries {
            assert_eq!(entry.local_bytes, 0, "{} has local memory", entry.symbol);
            assert!(
                entry.shared_static_bytes
                    + variant.dynamic_shared_bytes(&entry.symbol).unwrap_or(0)
                    <= 101_376,
                "{} exceeds the shared memory of a CTA",
                entry.symbol
            );
            // The computed fallback's libdevice trig frame is already allowed
            // by its declaration. Served table entries must keep a zero frame.
            let stack_limit =
                if variant.rope_fold.is_some() && entry.symbol == variant.rope_symbol() {
                    40
                } else {
                    0
                };
            assert!(
                entry.stack_bytes <= stack_limit,
                "{} stack exceeds {stack_limit}",
                entry.symbol
            );
            println!(
                "{} registers={} smem={} local={} stack={}",
                entry.symbol,
                entry.registers,
                entry.shared_static_bytes,
                entry.local_bytes,
                entry.stack_bytes
            );
        }
        let output = Command::new(toolchain.home.join("bin/cuobjdump"))
            .arg("-sass")
            .arg(&cubin.path)
            .output()
            .unwrap();
        assert!(output.status.success());
        let sass = String::from_utf8(output.stdout).unwrap();
        assert!(
            sass.contains("FADD") || sass.contains("FFMA"),
            "{} has no FP32 arithmetic",
            cubin.name
        );
        assert!(
            sass.contains("HMMA.16816"),
            "{} has no BF16 MMA",
            cubin.name
        );
        assert!(sass.contains("LDSM"), "{} has no ldmatrix", cubin.name);
        if cubin.name.ends_with("decode") {
            assert!(
                sass.contains("LDGSTS"),
                "{} has no asynchronous KV load",
                cubin.name
            );
        }
        if cubin.name.ends_with("prefill") {
            assert!(
                sass.contains(if cubin.name.contains("swa") {
                    "LDGSTS"
                } else {
                    "UTMALDG.4D"
                }),
                "{} has no paged TMA or ring asynchronous load",
                cubin.name
            );
        }
    }
    fs::remove_dir_all(temp).unwrap();
}

/// The compiler's FP8 refusal is the engine's zero-stack audit, not a
/// mathematical head-count restriction. This probe only compiles cubins.
#[test]
fn head_256_fp8_small_groups_fail_the_zero_stack_audit() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped, "the spill regression requires nvcc");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../kernels-sm120/cuda");
    let variants = catalog(kernel_lab::catalogs());
    for group in [2, 4] {
        let mut variant = *variants
            .iter()
            .find(|v| {
                v.query.phase == kernel_core::contract::AttentionPhase::Prefill
                    && v.query.kv_dtype == kernel_core::contract::DType::Fp8E4M3
                    && v.query.qk_dim == 256
                    && v.query.v_dim == 256
                    && v.layout == CacheLayout::Paged
            })
            .unwrap();
        variant.query.q_heads = group;
        let spec = variant.cubin(&root).unwrap();
        assert!(
            spec.entries
                .iter()
                .all(|entry| entry.allowed_local_bytes == 0)
        );
        let temp = scratch();
        let error = build(
            &toolchain,
            &[spec],
            &temp.join("out"),
            &temp.join("kernels.lock"),
            BuildOptions {
                lock_update: true,
                emit_cargo_directives: false,
                profile_line_info: false,
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("STACK 16 exceeds allowed local bytes 0"),
            "group {group}: {error}"
        );
        fs::remove_dir_all(temp).unwrap();
    }
}
