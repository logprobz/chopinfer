//! The engine domain against the SM120 kernels' qualification and sources:
//! the attention instances, the cluster decode shapes and every cited limit.

use super::*;

fn served_ranks() -> impl Iterator<Item = usize> {
    model_qwen38_contract::contract::SERVED_RANKS
        .into_iter()
        .map(|ranks| ranks as usize)
}

fn served_rank_counts() -> String {
    let mut ranks = served_ranks().collect::<Vec<_>>();
    ranks.sort_unstable();
    ranks
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" and ")
}

#[test]
fn admitted_attention_product_is_closed_over_lab_coverage() {
    use kernels_sm120::attention::qualification as q;
    assert_eq!(
        q::EXACT_VERIFY_ROWS,
        model_qwen38_contract::rank::EXACT_VERIFY_ROWS
    );
    let mut config =
        Qwen38Config::from_bytes(include_bytes!("../../fixtures/config.json")).unwrap();
    for shape in q::HEAD_SHAPES {
        for &(rotary, sections) in q::ROTARY_SECTIONS {
            for &index_heads in q::INDEX_HEADS {
                for &kept in q::KEPT_BLOCKS {
                    for total_q in (4..=64).step_by(4) {
                        let text = &mut config.text_config;
                        text.head_dim = shape.head_dim as usize;
                        text.num_attention_heads = total_q;
                        text.indexer_n_heads = index_heads as usize;
                        text.indexer_budget = (kept * q::BLOCK) as usize;
                        text.rope_parameters.partial_rotary_factor =
                            rotary as f64 / shape.head_dim as f64;
                        text.rope_parameters.mrope_section = sections.map(|s| s as usize).to_vec();
                        let expected = shape.dense_groups.contains(&(total_q as u32 / 4))
                            && served_ranks().all(|rank| {
                                shape.rows.contains(&(total_q as u32 / rank as u32))
                                    && shape
                                        .sparse_groups
                                        .contains(&(total_q as u32 / rank as u32))
                            });
                        assert_eq!(
                            engine_domain(&config, &serde_json::to_value(&config).unwrap()).is_ok(),
                            expected,
                            "d{} r{rotary} i{index_heads} k{kept} Q{total_q}",
                            shape.head_dim
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cluster_decode_domain_checks_both_bucket_groups() {
    let mut config =
        Qwen38Config::from_bytes(include_bytes!("../../fixtures/config.json")).unwrap();
    for (hidden, expected) in [(2304, [3, 3]), (3840, [3, 5]), (5120, [4, 5])] {
        config.text_config.hidden_size = hidden;
        engine_domain(&config, &serde_json::to_value(&config).unwrap()).unwrap();
        let splits = [[1, 2, 4], [8, 16, 32]].map(|buckets| {
            (1..=8)
                .find(|&blocks| {
                    buckets.into_iter().all(|bucket| {
                        kernels_sm120::gemm::fused::cluster_decode_shape(
                            (4 * hidden) as u32,
                            bucket,
                            blocks,
                        )
                        .is_some()
                    })
                })
                .unwrap()
        });
        assert_eq!(splits, expected);
    }
    for hidden in [2816, 3328, 4352] {
        config.text_config.hidden_size = hidden;
        assert_eq!(
            engine_domain(&config, &serde_json::to_value(&config).unwrap())
                .unwrap_err()
                .to_string(),
            format!(
                "geometry unmatched role lin_in: cluster decode K={} has no legal warp-block count in 1..=8 for buckets [1, 2, 4] (at most 8 CTAs and 101376 shared bytes)",
                hidden
            )
        );
    }
}

#[test]
fn engine_limits_agree_with_sources() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../crates/kernels-sm120");
    for (file, text) in [
        (
            "src/gemm/fused.rs",
            "pub fn cluster_decode_shape(k: u32, m_capacity: u32, warp_blocks: u32)",
        ),
        ("src/gemm/fused.rs", "!k.is_multiple_of(128)"),
        ("src/gemm/fused.rs", "blocks / chunk <= 8"),
        (
            "src/gemm/fused.rs",
            "cluster_shared_bytes(tokens, chunk, passes) + STATIC_SHARED <= 101_376",
        ),
        (
            "cuda/attention/sparse_indexer.cuh",
            "static_assert((Heads == 4 || Heads == 8 || Heads == 16) && Dim == 128 &&",
        ),
        (
            "cuda/attention/sparse_indexer.cuh",
            "(Rotary == 32 || Rotary == 64),",
        ),
        (
            "cuda/attention/sparse_indexer.cuh",
            "static constexpr int width = (Heads + 1) * Dim;",
        ),
        ("cuda/attention/sparse.cuh", "static_assert(group <= 16,"),
        (
            "cuda/attention/sparse.cuh",
            "static_assert((Dim == 128 || Dim == 256) && Page % Block == 0);",
        ),
        (
            "cuda/attention/varlen.cuh",
            "static_assert(D == 64 || D == 72, ",
        ),
        (
            "cuda/linear/delta.cuh",
            "static_assert(Dim == 128 && ValueHeads % KeyHeads == 0);",
        ),
        (
            "cuda/linear/delta_prefill.cuh",
            "static_assert(Group <= 3 && Dim % 8 == 0);",
        ),
        (
            "cuda/linear/delta_prefill.cuh",
            "static_assert(Dim == 128 && kPrefillThreads == 256 && kScanRows == 16);",
        ),
        (
            "cuda/linear/delta_prefill.cuh",
            "static_assert(L::kBytes <= 101376);",
        ),
        (
            "src/attention/mod.rs",
            "paged decode key tile partials exceed K tile fold scratch",
        ),
        (
            "src/attention/mod.rs",
            "parts * fold_floats * 4 <= keys * query.qk_dim * 2",
        ),
        (
            "src/attention/mod.rs",
            "![128, 256].contains(&self.query.qk_dim)",
        ),
        ("src/attention/mod.rs", "![32, 64].contains(&fold.rotary)"),
        (
            "cuda/attention/prefill.cuh",
            "static_assert(16 % tile_group == 0,",
        ),
        (
            "cuda/attention/verify.cuh",
            "static_assert(rows % 16 == 0 && rows <= 128",
        ),
        (
            "src/attention/mod.rs",
            "(self.query_capacity * (self.query.q_heads / self.query.kv_heads)) % 16 != 0",
        ),
        (
            "cuda/attention/decode.cuh",
            "D::parts * fold_floats * 4 <= D::k_bytes",
        ),
        ("src/attention/gated.rs", "![32, 64].contains(&rotary_dim)"),
        (
            "cuda/linear/norm.cuh",
            "static_assert(HeadDim == 128 && Width % HeadDim == 0);",
        ),
        (
            "cuda/linear/conv.cuh",
            "static_assert(Channels % 2 == 0 && History > 0);",
        ),
        ("cuda/linear/conv.cuh", "static_assert(History <= kChunk);"),
        ("cuda/linear/common.cuh", "constexpr int kChunk = 64;"),
        (
            "cuda/moe/route_parallel.cuh",
            "static_assert(Experts % 32 == 0 && TopK > 0 && TopK <= 32);",
        ),
        (
            "cuda/moe/route_parallel.cuh",
            "static_assert(LocalExperts >= 32 && LocalExperts <= 512 && LocalExperts % 32 == 0);",
        ),
        (
            "src/moe/parallel.rs",
            "!(32..=512).contains(&local_experts)",
        ),
        ("src/moe/parallel.rs", "!local_experts.is_power_of_two()"),
    ] {
        assert!(
            std::fs::read_to_string(root.join(file))
                .unwrap()
                .contains(text),
            "engine limit changed in {file}: {text}"
        );
    }
    let row_source = std::fs::read_to_string(root.join("src/row/hyper_connection.rs")).unwrap();
    for text in [
        "self.streams != 4",
        "self.rank != 320",
        "self.eps_bits != 1e-6_f32.to_bits()",
        "self.orders != 3",
        "self.heads_per_order != 8",
        "!16_u32.is_multiple_of(local)",
        "channels > 0",
        "channels.is_multiple_of(1024)",
        "kernel: 4,",
        "dilation: 3,",
    ] {
        assert!(
            row_source.contains(text),
            "row engine limit changed: {text}"
        );
    }
    for text in [
        "self.width == 0",
        "!self.width.is_multiple_of(256)",
        "self.head_width == 0",
        "!self.head_width.is_multiple_of(16)",
        "let channels = streams.width * streams.streams;",
    ] {
        assert!(
            row_source.contains(text),
            "row engine limit changed: {text}"
        );
    }
    model_qwen38_contract::contract::Shape::flash(model_qwen38_contract::contract::Ranks::TP4)
        .validate_rows()
        .unwrap();
    assert!(
        std::fs::read_to_string(root.join("cuda/moe/grouped_prefill.cuh"))
            .unwrap()
            .contains(
                "static_assert(Experts >= 32 && Experts <= 256 && (Experts & (Experts - 1)) == 0);"
            )
    );
    assert!(
        std::fs::read_to_string(root.join("src/attention/mod.rs"))
            .unwrap()
            .contains("if group > 8 && group != 16 {")
    );
    assert!(
        std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../bundle/cuda/gemm/qwen38.cuh")
        )
        .unwrap()
        .contains("static_assert(K % 256 == 0 && Rows >= 1 && Rows <= 32);")
    );
    let query_source =
        std::fs::read_to_string(root.join("../kernel-core/src/contract.rs")).unwrap();
    assert!(query_source.contains("!(2..=8).contains(&kernel)"));
    assert!(query_source.contains("eps.is_finite() && eps > 0.0"));
    let sparse_source = std::fs::read_to_string(root.join("src/attention/sparse.rs")).unwrap();
    for text in [
        "![4, 8, 16].contains(&self.index_heads)",
        "self.index_dim != 128",
        "![32, 64].contains(&self.rotary_dim)",
        "self.mrope.iter().sum::<u32>() != self.rotary_dim / 2",
        "![128, 256].contains(&self.head_dim)",
        "const TOPK_MAX_BLOCKS: u32 = 65_536;",
        "self.block < 2",
        "self.ring < self.block",
        "!self.ring.is_multiple_of(self.block)",
        "!self.page_size.is_multiple_of(self.block)",
        "!self.max_context.is_multiple_of(self.block)",
        "self.max_blocks() > TOPK_MAX_BLOCKS",
    ] {
        assert!(sparse_source.contains(text), "QSA limit changed: {text}");
    }
    assert!(
        std::fs::read_to_string(root.join("cuda/attention/gated.cuh"))
            .unwrap()
            .contains("static_assert(q_warps + KvHeads <= 8,")
    );
    assert!(
        std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../bundle/cuda/gemm/qwen38.cuh")
        )
        .unwrap()
        .contains("static_assert(K % 16 == 0 && K <= 512,")
    );
    // The declared index-head and head-dim sets are the attention
    // qualification table's, and their reasons list them.
    let rules: serde_json::Value = serde_json::from_str(GEOMETRIES[0].rules).unwrap();
    let domain = &rules["engine_domain"];
    use kernels_sm120::attention::qualification;
    use model_compiler_sdk::geometry::engine_entries;
    for (path, values, reason) in [
        (
            "text_config.indexer_n_heads",
            qualification::INDEX_HEADS.to_vec(),
            "sparse indexer requires heads in",
        ),
        (
            "text_config.head_dim",
            qualification::HEAD_SHAPES
                .iter()
                .map(|shape| shape.head_dim)
                .collect(),
            "sparse and gated attention require head dim in",
        ),
    ] {
        let listed = values
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(domain[path]["values"], serde_json::json!(values), "{path}");
        assert_eq!(
            domain[path]["reason"],
            format!("{reason} {{{listed}}}"),
            "{path}"
        );
    }
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bundle/src/scratch.rs");
    let rope = std::fs::read_to_string(bundle).unwrap();
    for text in [
        "let pairs = shape.rotary() as usize / 2;",
        "theta: f64::from(shape.rope_theta()),",
        "rule: AngleRule::HfF32AngleBf16CosSin,",
    ] {
        assert!(rope.contains(text), "served RoPE rule changed: {text}");
    }
    let table = std::fs::read_to_string(root.join("../kernel-core/src/rotary.rs")).unwrap();
    for text in [
        "!self.theta.is_finite() || self.theta <= 0.0",
        "!(self.theta as f32).is_finite()",
        "self.theta as f32 <= 0.0",
    ] {
        assert!(table.contains(text), "computed RoPE limit changed: {text}");
    }
    let audit = std::fs::read_to_string(root.join("../kernel-build/src/lib.rs")).unwrap();
    for text in [
        "allowed_local_bytes: 0,",
        "if local_bytes > entry.allowed_local_bytes",
        "if stack_bytes > entry.allowed_local_bytes",
    ] {
        assert!(audit.contains(text), "zero-stack audit changed: {text}");
    }
    // Every cited engine limit still reads as declared.
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    for (path, entry) in engine_entries(domain) {
        if let Some(source) = entry["source"].as_str() {
            let text = std::fs::read_to_string(repository.join(source)).unwrap();
            assert!(
                text.contains(entry["source_text"].as_str().unwrap()),
                "engine limit changed in {source} for {path}"
            );
        }
    }
    // The page and served rank counts are the contract's.
    let rank_multiple = (1..)
        .find(|n| served_ranks().all(|rank| n % rank == 0))
        .unwrap();
    for (path, entry) in engine_entries(domain) {
        match entry["phase"].as_str() {
            Some("after_qsa_context") => {
                assert_eq!(
                    entry["multiple_of"],
                    model_qwen38_contract::contract::PAGE_TOKENS,
                    "{path}"
                );
                assert_eq!(
                    entry["reason"],
                    format!(
                        "context must be a multiple of the {}-token KV page",
                        model_qwen38_contract::contract::PAGE_TOKENS
                    )
                );
            }
            Some("after_linear_groups") => {
                assert_eq!(entry["multiple_of"], rank_multiple, "{path}");
                assert_eq!(
                    entry["reason"],
                    format!(
                        "must be positive and divisible by served rank counts {}",
                        served_rank_counts()
                    )
                );
            }
            _ => {}
        }
    }
    let limits = engine_entries(domain)
        .map(|(path, entry)| {
            let mut limit = entry.as_object().unwrap().clone();
            limit.retain(|key, _| !["reason", "source", "source_text"].contains(&key.as_str()));
            (path, serde_json::Value::Object(limit))
        })
        .collect::<Vec<_>>();
    let page = |phase: &str| serde_json::json!({"multiple_of": 64, "phase": phase});
    let ranks = |path| {
        (
            path,
            serde_json::json!({"multiple_of": 4, "phase": "after_linear_groups"}),
        )
    };
    let rows = |values, phase: &str| serde_json::json!({"values": values, "phase": phase});
    assert_eq!(
        limits,
        vec![
            (
                "text_config.indexer_kv_heads",
                serde_json::json!({"values": [1]})
            ),
            (
                "text_config.indexer_head_dim",
                serde_json::json!({"values": [128]})
            ),
            (
                "text_config.linear_key_head_dim",
                serde_json::json!({"values": [128]})
            ),
            (
                "text_config.linear_value_head_dim",
                serde_json::json!({"values": [128]})
            ),
            (
                "text_config.indexer_n_heads",
                serde_json::json!({"values": [4, 8, 16]})
            ),
            (
                "text_config.head_dim",
                serde_json::json!({"values": [128, 256]})
            ),
            (
                "text_config.linear_conv_kernel_dim",
                serde_json::json!({"min": 2, "max": 8})
            ),
            (
                "text_config.ple_conv_kernel_size",
                serde_json::json!({"min": 2, "max": 8})
            ),
            (
                "text_config.ple_conv_kernel_size",
                rows(serde_json::json!([4]), "after_row_shapes")
            ),
            (
                "text_config.hc_count",
                rows(serde_json::json!([4]), "after_row_widths")
            ),
            (
                "text_config.hc_lowrank",
                serde_json::json!({"min": 16, "max": 512, "multiple_of": 16})
            ),
            (
                "text_config.hc_lowrank",
                rows(serde_json::json!([320]), "after_row_widths")
            ),
            (
                "text_config.num_experts_per_tok",
                serde_json::json!({"max": 32})
            ),
            (
                "text_config.ple_layer_ids",
                serde_json::json!({"max_length": 1})
            ),
            (
                "text_config.ngram_size",
                rows(serde_json::json!([3]), "after_row_widths")
            ),
            (
                "text_config.heads_per_ngram",
                rows(serde_json::json!([8]), "after_row_widths")
            ),
            (
                "text_config.max_position_embeddings",
                page("after_qsa_context")
            ),
            ranks("text_config.hidden_size"),
            ranks("text_config.vocab_size"),
            ranks("text_config.linear_num_key_heads"),
            ranks("text_config.linear_num_value_heads"),
            ranks("text_config.num_attention_heads"),
            ranks("text_config.num_experts"),
            ranks("text_config.ple_embed_dim"),
        ]
    );
}
