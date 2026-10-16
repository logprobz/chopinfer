//! Shared model geometry.
#[cfg(test)]
pub use model_qwen38_contract::contract::*;
#[cfg(not(test))]
pub use model_qwen38_contract::contract::{
    ANCHOR_ITEMS, BF16_BYTES, BUNDLE_ID, DECODE_BATCH, DECODE_ENTRY_BUCKETS, DECODE_ENTRY_ROWS,
    DRAFT_ITEMS, EXACT_VERIFY, Items, MAX_DRAFT_TOKENS, MAX_VERIFY_ROWS, MAX_VERIFY_TOKENS,
    NGRAM_HOST_VIEW, PAGE_TOKENS, PREFILL_GRANULE, PREFILL_TOKENS, QSA_RING, Qwen, RANK_IMAGE_ROLE,
    RING_STATE_ID, Ranks, Ring, SCORE_ROWS, SCORED_ROWS, Shape, TOKEN_GRANULE, VERIFY_ROW_BUCKETS,
    index_role,
};

/// Cache geometry-owned values independently of the instance rank count.
pub fn per_geometry<T>(
    cache: &'static std::sync::OnceLock<Vec<std::sync::OnceLock<T>>>,
    shape: Shape,
    build: impl FnOnce(&'static model_qwen38_contract::geometry::Geometry) -> T,
) -> &'static T {
    let geometries = model_qwen38_contract::geometry::GEOMETRIES;
    let entries = cache.get_or_init(|| {
        geometries
            .iter()
            .map(|_| std::sync::OnceLock::new())
            .collect()
    });
    let index = geometries
        .iter()
        .position(|g| g.id == shape.geometry.id)
        .expect("registered geometry");
    entries[index].get_or_init(|| build(&geometries[index]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_geometry_matches_decision_0084() {
        assert_eq!(Shape::TP4.linear_qkv, 2560);
        assert_eq!(Shape::TP4.attn_row, 3584);
        assert_eq!(INDEX_ROW, 640);
        assert_eq!(Shape::TP4.vocab_per_rank, 62_080);
        assert_eq!(MAX_PAGES, 4096);
        // 12,288 bytes of KV and 768 of block keys per token.
        assert_eq!(PAGE_BYTES_BF16, 64 * (12_288 + 768));
        assert_eq!(PAGE_BYTES_BF16_MTP, 64 * (13_312 + 832));
        // FP8 halves the main layers' K and V only: 6,144 bytes of KV and
        // 768 of block keys per token, and 1,088 more for the BF16 MTP layer.
        assert_eq!(PAGE_BYTES_FP8, 64 * (6_144 + 768));
        assert_eq!(PAGE_BYTES_FP8_MTP, 64 * (6_144 + 768 + 1_088));
        assert!(
            KV_FORMATS[..2]
                .iter()
                .all(|format| format.valid(FULL_LAYERS))
        );
        assert!(
            KV_FORMATS[2..]
                .iter()
                .all(|format| format.valid(FULL_LAYERS))
        );
        let fp8_mtp = &KV_FORMATS[3];
        assert_eq!(
            page_layer(fp8_mtp, 47),
            (11 * 36_864, &PAGE_PLANES_FP8, true)
        );
        let (offset, planes, fp8) = page_layer(fp8_mtp, MTP_LAYER);
        assert_eq!((offset, fp8), (12 * 36_864, false));
        assert_eq!(offset + planes.iter().sum::<u64>(), PAGE_BYTES_FP8_MTP);
        assert_eq!(
            page_layer(&KV_FORMATS[1], MTP_LAYER).0 + PAGE_LAYER_BYTES_BF16,
            PAGE_BYTES_BF16_MTP
        );
        // 32 slots of the drafter's ring state, as before the served count;
        // 102,912 bytes per slot without the drafter.
        assert_eq!(RING_STATE_SLOT_BYTES_MTP, 111_616);
        assert_eq!(RING_STATE_BYTES_MTP, 3_571_712);
        assert_eq!(RING_STATE_SLOT_BYTES, 102_912);
        assert!(RING_STATE_SLOT_BYTES.is_multiple_of(256));
        let ring = Ring {
            shape: model_qwen38_contract::contract::Shape::flash(
                model_qwen38_contract::contract::Ranks::TP4,
            ),
            layers: FULL_LAYERS,
            slots: 32,
        };
        assert_eq!(
            (ring.key_bytes(), ring.position_bytes(), ring.state_bytes()),
            (131_072, 6_144, 32 * RING_STATE_SLOT_BYTES)
        );
        assert_eq!(
            (DECODE_BATCH, DECODE_ENTRY_ROWS, DRAFT_ITEMS),
            (256, 32, 32)
        );
        assert_eq!(DECODE_ENTRY_BUCKETS.last(), Some(&DECODE_ENTRY_ROWS));
        assert!(VERIFY_ROW_BUCKETS.contains(&MAX_VERIFY_ROWS));
        // 28.3 MB of delta-rule state and 0.55 MB of convolution state.
        assert_eq!(LINEAR_LAYERS * Shape::TP4.delta_entry_bytes, 28_311_552);
        assert_eq!(LINEAR_LAYERS * Shape::TP4.conv_entry_bytes, 552_960);
        assert_eq!(PLE_CONV_BYTES, 184_320);
        assert_eq!(
            Shape::TP4.recurrent_state_layouts().map(|(_, bytes)| bytes),
            [552_960, 28_311_552, 184_448]
        );
        assert!(
            Shape::TP4
                .recurrent_state_layouts()
                .is_sorted_by_key(|(id, _)| *id)
        );
        assert_eq!((FULL_LAYERS, LINEAR_LAYERS), (12, 36));
        assert_eq!((0..LAYERS).filter(|&layer| is_full(layer)).count(), 12);
        assert_eq!((ordinal(47), ordinal(46), ordinal(0)), (11, 35, 0));
        assert_eq!(ordinal(MTP_LAYER), FULL_LAYERS);
    }

    /// docs/topology-qwen.md, "Per-rank shapes".
    #[test]
    fn shapes_per_topology() {
        let row = |s: Shape| {
            [
                s.ranks(),
                s.q_heads,
                s.q_heads * 2 * ATTN_HEAD,
                s.attn_row,
                s.attn_value,
                s.key_heads,
                s.value_heads,
                s.linear_qkv,
                s.linear_value,
                s.linear_ab_stride,
                s.local_experts,
                s.shared_intermediate,
                s.vocab_per_rank,
                s.ngram_local_heads,
            ]
        };
        assert_eq!(
            row(Shape::TP2),
            [
                2, 12, 6144, 6656, 3072, 8, 24, 5120, 3072, 48, 256, 384, 124_160, 8
            ]
        );
        assert_eq!(
            row(Shape::TP4),
            [
                4, 6, 3072, 3584, 1536, 4, 12, 2560, 1536, 24, 128, 256, 62_080, 4
            ]
        );
        let state = |s: Shape| (s.conv_entry_bytes, s.delta_entry_bytes);
        assert_eq!(state(Shape::TP2), (30_720, 1_572_864));
        assert_eq!(state(Shape::TP4), (15_360, 786_432));
        assert_eq!(
            TOPOLOGIES.map(|t| (t.id, t.ranks.get())),
            [("qwen38-flash-tp2", 2), ("qwen38-flash-tp4", 4)]
        );
        assert_eq!(
            (0..4).map(|r| Shape::TP4.kv_head(r)).collect::<Vec<_>>(),
            [0, 0, 1, 1]
        );
        assert_eq!(
            (0..2).map(|r| Shape::TP2.kv_head(r)).collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(Shape::of(Ranks::TP1), None);
        assert_eq!(Shape::of(Ranks::TP8), None);
        assert_eq!([Shape::TP2].map(|s| s.tag()), ["qwen38_tp2"]);
        assert_eq!(Shape::TP4.tag(), "qwen38");
    }
}

/// The runtime shape agrees, at every topology, with what the build
/// compiled: the defines of the per-rank cubins in `kernels.lock` and the
/// shapes of the catalog entries the program launches.
#[cfg(test)]
mod build_agreement {
    use super::*;

    fn defines(cubin: &str) -> Vec<(String, String)> {
        let lock: serde_json::Value =
            serde_json::from_str(include_str!("../kernels.lock")).expect("lock JSON");
        let record = lock
            .as_array()
            .unwrap()
            .iter()
            .find(|record| record["name"] == cubin)
            .unwrap_or_else(|| panic!("no cubin {cubin}"));
        serde_json::from_value(record["defines"].clone()).unwrap()
    }

    #[test]
    fn build_and_runtime_shapes_agree_at_every_topology() {
        for shape in Shape::ALL {
            let profile = shape.profile();
            assert_eq!(
                model_qwen38_contract::sm120::config::profile_shape(profile),
                Some(model_qwen38_contract::sm120::config::RankShape {
                    ranks: shape.ranks()
                })
            );
            let tag = shape.tag();
            let entries: Vec<_> = [
                ("CHOP_QWEN38_LINEAR_LAYERS", LINEAR_LAYERS, ""),
                (
                    "CHOP_QWEN38_MAX_VERIFY_ROWS",
                    u64::from(MAX_VERIFY_ROWS),
                    "",
                ),
                (
                    "CHOP_QWEN38_RING_SLOT_KEY_BYTES",
                    RING_SLOT_KEY_BYTES,
                    "ull",
                ),
                (
                    "CHOP_QWEN38_RING_SLOT_POSITION_BYTES",
                    RING_SLOT_POSITION_BYTES,
                    "ull",
                ),
                (
                    "CHOP_QWEN38_CONV_ENTRY_BYTES",
                    shape.conv_entry_bytes,
                    "ull",
                ),
                (
                    "CHOP_QWEN38_DELTA_ENTRY_BYTES",
                    shape.delta_entry_bytes,
                    "ull",
                ),
                ("CHOP_QWEN38_HIDDEN", u64::from(HIDDEN), ""),
                ("CHOP_QWEN38_STREAMS", u64::from(STREAMS), ""),
                ("CHOP_QWEN38_HC_RANK", u64::from(HC_RANK), ""),
                ("CHOP_QWEN38_PLE_CONV_TAPS", u64::from(PLE_CONV_TAPS), ""),
                (
                    "CHOP_QWEN38_PLE_CONV_DILATION",
                    u64::from(PLE_CONV_DILATION),
                    "",
                ),
                ("CHOP_QWEN38_PLE_CONV_ROWS", u64::from(PLE_CONV_ROWS), ""),
                ("CHOP_QWEN38_NGRAM_EOS", u64::from(NGRAM_EOS), ""),
                ("CHOP_QWEN38_NGRAM_SIZE", u64::from(NGRAM_SIZE), ""),
                ("CHOP_QWEN38_NGRAM_HEADS", u64::from(NGRAM_HEADS), ""),
                (
                    "CHOP_QWEN38_NGRAM_HEADS_PER_ORDER",
                    u64::from(NGRAM_HEADS_PER_ORDER),
                    "",
                ),
                (
                    "CHOP_QWEN38_NGRAM_HEAD_WIDTH",
                    u64::from(NGRAM_HEAD_WIDTH),
                    "",
                ),
                (
                    "CHOP_QWEN38_NGRAM_DEFAULT_LOCAL_HEADS",
                    u64::from(Shape::TP4.ngram_local_heads),
                    "",
                ),
            ]
            .map(|(name, value, suffix)| (name.to_owned(), format!("{value}{suffix}")))
            .into();
            assert_eq!(defines(&format!("{tag}_bundle")), entries);
            let mut speculation = vec![
                ("CHOP_QWEN38_RANKS".to_owned(), shape.ranks().to_string()),
                (
                    "CHOP_QWEN38_NGRAM_LOCAL_HEADS".to_owned(),
                    shape.ngram_local_heads.to_string(),
                ),
                (
                    "CHOP_QWEN38_SHARD_VOCAB".to_owned(),
                    shape.vocab_per_rank.to_string(),
                ),
            ];
            speculation.extend(entries);
            assert_eq!(defines(&format!("{tag}_speculation")), speculation);
            assert_eq!(
                defines(&format!("chop_moe_{tag}_route")),
                [
                    (
                        "CHOP_QWEN38_LOCAL_EXPERTS".into(),
                        shape.local_experts.to_string()
                    ),
                    ("CHOP_QWEN38_GLOBAL_EXPERTS".into(), EXPERTS.to_string()),
                    ("CHOP_QWEN38_TOP_K".into(), TOP_K.to_string()),
                ]
            );
            assert_eq!(
                model_qwen38_contract::sm120::moe::local_experts(profile).unwrap(),
                shape.local_experts
            );
            let conv = model_qwen38_contract::sm120::linear::qwen38_catalog(profile)
                .into_iter()
                .find(|entry| entry.symbol == format!("chop_linear_{tag}_conv_decode_v1"))
                .unwrap();
            assert!(matches!(
                conv.query,
                kernel_core::contract::RecurrentQuery::CausalConv1d { channels, .. }
                    if channels == shape.linear_qkv
            ));
            let lin_in =
                crate::sm120::catalog::qwen::entry(&format!("chop_gemm_{tag}_lin_in_c1_v1"))
                    .unwrap();
            assert_eq!(
                lin_in.segments,
                [
                    shape.linear_qkv,
                    shape.linear_value,
                    shape.value_heads,
                    shape.value_heads
                ]
            );
            let attn_in =
                crate::sm120::catalog::qwen::entry(&format!("chop_gemm_{tag}_attn_in_c1_v1"))
                    .unwrap();
            assert_eq!(
                attn_in.segments,
                [
                    shape.q_heads * 2 * ATTN_HEAD,
                    ATTN_HEAD,
                    ATTN_HEAD,
                    INDEX_ROW
                ]
            );
            let head = crate::sm120::catalog::dense::entry(&format!("chop_gemm_{tag}_head_m1_v1"))
                .unwrap();
            let kernel_core::contract::StaticQuery::Gemm(query) = head.key.query else {
                panic!("head query")
            };
            assert_eq!(query.n, shape.vocab_per_rank);
        }
    }
}
