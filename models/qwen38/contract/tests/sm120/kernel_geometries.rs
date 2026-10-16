use model_qwen38_contract::sm120::{self, dense::Role, geometry};

#[test]
fn every_profile_builds_its_catalog_without_cross_geometry_entries() {
    let catalogs = sm120::catalogs();
    for owner in geometry::MODELS {
        for profile in owner.model.profiles() {
            let specs = kernels_sm120::gemm::fused::cubin_specs(catalogs, Some(profile));
            assert!(!specs.is_empty());
            if profile.ranks() == Some(4) {
                assert!(specs.iter().any(|spec| {
                    spec.entries
                        .iter()
                        .any(|entry| entry.symbol == owner.hc.up_mix_symbol)
                }));
            }
            for spec in specs {
                assert!(spec.name.starts_with(&format!("gemm_{}_", profile.tag())));
                assert!(spec.entries.iter().all(|entry| {
                    sm120::PROFILES
                        .iter()
                        .filter(|candidate| {
                            entry
                                .symbol
                                .starts_with(&format!("chop_gemm_{}_", candidate.tag()))
                        })
                        .max_by_key(|candidate| candidate.tag().len())
                        == Some(&profile)
                }));
            }
            let shape = geometry::shape(profile).unwrap();
            for role in [
                Role::LinearOutput,
                Role::AttentionOutput,
                Role::PleValue,
                Role::MtpFc,
            ] {
                let (entry_profile, name) = sm120::dense::alias(profile, role);
                let actual = sm120::dense::shapes(entry_profile)
                    .into_iter()
                    .find(|entry| entry.0 == name)
                    .unwrap();
                assert_eq!((actual.1, actual.2, actual.3), role.dimensions(shape));
            }
        }
    }
}

#[test]
fn flash_recipe_defines_remain_exact() {
    for ranks in [2, 4] {
        let legacy = sm120::config::RankShape { ranks };
        let generated = geometry::shape(sm120::config::profile(ranks).unwrap()).unwrap();
        assert_eq!(generated.bundle_defines(), legacy.bundle_defines());
        assert_eq!(
            generated.speculation_defines(),
            legacy.speculation_defines()
        );
    }
}

#[test]
fn attention_catalogs_keep_each_geometries_head_and_rotary_shapes() {
    use kernels_sm120::attention::sparse::Kind;
    let sparse = sm120::attention::sparse_catalog();
    let dense = sm120::attention::catalog();
    for owner in geometry::MODELS {
        let g = owner.geometry;
        for profile in owner.model.profiles() {
            let shape = geometry::shape(profile).unwrap();
            let entries = sparse
                .iter()
                .filter(|e| e.profile == profile)
                .collect::<Vec<_>>();
            assert_eq!(
                entries.len(),
                if profile.ranks() == Some(4) { 12 } else { 6 }
            );
            for entry in entries {
                entry.contract().unwrap();
                assert_eq!(entry.geometry.index_heads, g.index_heads());
                assert_eq!(entry.geometry.head_dim, g.head_dim());
                assert_eq!(entry.geometry.rotary_dim, g.rotary());
                assert_eq!(entry.geometry.theta, g.rope_theta());
                assert_eq!(entry.geometry.mrope, g.mrope());
                assert_eq!(entry.geometry.q_heads, shape.q_heads());
                assert_eq!(entry.geometry.kv_heads, shape.kv_heads());
                if entry.kind == Kind::Query {
                    assert_eq!(entry.threads(), 32 * g.index_heads());
                }
                if entry.name.ends_with("block_scores_prefill") {
                    assert_eq!(
                        entry.kind,
                        Kind::Scores {
                            rows: 256 / g.index_heads(),
                            by_run: false
                        }
                    );
                }
            }
            let gated = sm120::attention::gated_config(profile).unwrap();
            assert_eq!(gated.head_dim, g.head_dim());
            assert_eq!(gated.rotary, g.rotary());
            assert_eq!(
                gated.row_width(),
                (2 * shape.q_heads() + 2 * shape.kv_heads()) * g.head_dim()
            );
            let variants = dense
                .iter()
                .filter(|v| v.profile == profile)
                .collect::<Vec<_>>();
            assert_eq!(
                variants.len(),
                if profile.ranks() == Some(4) { 6 } else { 0 }
            );
            for variant in variants {
                variant.validate().unwrap();
                assert_eq!(
                    (variant.query.qk_dim, variant.query.v_dim),
                    (g.head_dim(), g.head_dim())
                );
                if let Some(fold) = variant.gated_fold {
                    assert_eq!(fold.rotary, g.rotary());
                    assert_eq!(fold.sections, g.mrope());
                }
            }
        }
    }
}

#[test]
fn widened_indexers_select_the_query_and_score_launch_shapes() {
    use kernels_sm120::attention::sparse::{self, Kind};
    for heads in [4, 8, 16] {
        for rotary in [32, 64] {
            let g = sparse::Geometry {
                index_heads: heads,
                head_dim: 128,
                rotary_dim: rotary,
                mrope: if rotary == 32 {
                    [6, 5, 5]
                } else {
                    [11, 11, 10]
                },
                ..sm120::attention::SPARSE
            };
            let entries = sparse::indexer_entries(
                sm120::config::PROFILE,
                g,
                sm120::config::EXACT_VERIFY_ROWS,
            );
            assert_eq!(entries.len(), 6);
            for entry in entries {
                let contract = entry.contract().unwrap();
                if entry.kind == Kind::Query {
                    assert_eq!(entry.plan(8, 8192).unwrap().block, [heads * 32, 1, 1]);
                    assert_eq!(u32::from(contract.key.knobs.warps), heads);
                }
                if entry.name.ends_with("block_scores_prefill") {
                    assert_eq!(
                        entry.kind,
                        Kind::Scores {
                            rows: 256 / heads,
                            by_run: false
                        }
                    );
                    assert_eq!(contract.key.knobs.tile_m, 256);
                }
            }
        }
    }
}

#[test]
fn hc_precompute_uses_each_profiles_geometry_and_source_owner() {
    use kernels_sm120::gemm::fused;
    use kernels_sm120::{Catalogs, Model, ModelCatalog, Profile};
    const OTHER: Model = Model::new(
        kernels_sm120::profile_names!("fixture-hc", "fixture_hc", 4, 2),
        sm120::config::CONFIG,
    );
    const PROFILES: [Profile; 2] = [sm120::config::PROFILE, OTHER.at(4)];
    fn entries() -> Vec<fused::Entry> {
        let flash = sm120::gemm::fused_catalog()
            .into_iter()
            .filter(|e| e.profile == PROFILES[0] && fused::hc_precompute_spec(e).is_ok())
            .collect::<Vec<_>>();
        let mut entries = flash.clone();
        for mut entry in flash {
            entry.profile = PROFILES[1];
            entry.symbol = entry.symbol.replace("qwen38", "fixture_hc");
            entry.hc = Some(fused::HcGeometry {
                rank: 160,
                up_mix_symbol: "chop_gemm_fixture_hc_up_mix_decode_v1",
                ..sm120::gemm::HC
            });
            entry.segments[0] = 160;
            entry.query.n = entry.segments.iter().sum();
            entry.contract().unwrap();
            entries.push(entry);
        }
        entries
    }
    static CATALOG: ModelCatalog = ModelCatalog {
        name: "hc-fixture",
        profiles: &PROFILES,
        gemm_fused: entries,
        gemm_fused_sources: |profile, spec| {
            spec.defines.push(("OWNER".into(), profile.tag().into()))
        },
        ..ModelCatalog::EMPTY
    };
    let catalogs = Catalogs::new(&[&CATALOG]).unwrap();
    for profile in PROFILES {
        let specs = fused::cubin_specs(&catalogs, Some(profile));
        let precompute = specs
            .iter()
            .find(|s| s.name == format!("gemm_{}_hc_precompute_sm120", profile.tag()))
            .unwrap();
        assert_eq!(precompute.entries.len(), 6);
        assert_eq!(
            precompute.defines,
            vec![("OWNER".into(), profile.tag().into())]
        );
        assert!(precompute.entries.iter().all(|e| {
            e.symbol
                .starts_with(&format!("chop_gemm_{}_", profile.tag()))
        }));
        let geometry = entries()
            .into_iter()
            .find(|e| e.profile == profile)
            .unwrap()
            .hc
            .unwrap();
        assert_eq!(
            fused::hc_precompute_shared(
                &catalogs,
                &format!("chop_gemm_{}_hc_up_mix_precomputed_v1", profile.tag())
            ),
            Some(
                fused::hc_up_mix_plan(geometry, 32)
                    .unwrap()
                    .dynamic_shared_bytes
            )
        );
    }
}

#[test]
fn synthetic_config_requires_its_listed_geometry() {
    for (id, bytes) in [
        (
            "synthetic_small",
            include_bytes!("../../../../../fixtures/qwen38/geometries/synthetic-small/config.json")
                .as_slice(),
        ),
        (
            "synthetic_wide",
            include_bytes!("../../../../../fixtures/qwen38/geometries/synthetic-wide/config.json")
                .as_slice(),
        ),
    ] {
        let parsed = model_qwen38_contract::config::Qwen38Config::from_bytes(bytes);
        let document: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        let listed = model_qwen38_contract::geometry::GEOMETRIES
            .iter()
            .find(|g| {
                let expected: serde_json::Value = serde_json::from_str(g.document).unwrap();
                model_qwen38_contract::geometry::COMPILE_FIELDS
                    .iter()
                    .all(|field| {
                        model_compiler_sdk::geometry::field(&document, field)
                            == model_compiler_sdk::geometry::field(&expected, field)
                    })
            });
        if let Some(geometry) = listed {
            let config = parsed.expect("listed synthetic configuration is in the engine domain");
            assert_eq!(config.geometry().unwrap().geometry.id, geometry.id);
            assert_eq!(config.text_config.indexer_budget, 1024);
            assert_eq!(
                config.text_config.vocab_size,
                if id == "synthetic_small" {
                    248832
                } else {
                    248320
                }
            );
        } else {
            assert!(
                parsed
                    .unwrap_err()
                    .to_string()
                    .starts_with("unsupported geometry:")
            );
        }
    }
}

#[test]
fn unlisted_small_geometry_builds_catalog_entries_without_cubins() {
    use kernels_sm120::Model;
    use kernels_sm120::gemm::{bf16_query, fused};
    use model_qwen38_contract::geometry::FLASH;
    let mut g = FLASH;
    g.id = "catalog_fixture";
    g.kernel_tag = "catalog_fixture";
    g.base = false;
    g.pinned_names = false;
    g.tuning = None;
    g.shape.hidden = 1024;
    g.shape.ple.embed_dim = 1024;
    g.shape.linear.key_heads = 8;
    g.shape.linear.value_heads = 24;
    g.shape.moe.experts = 128;
    g.shape.moe.top_k = 8;
    g.shape.moe.intermediate = 512;
    g.shape.moe.shared_intermediate = 512;
    const MODEL: Model = Model::new(
        kernels_sm120::profile_names!("catalog-fixture", "catalog_fixture", 4, 2),
        sm120::config::CONFIG,
    );
    for ranks in [2, 4] {
        let shape = geometry::GeometryShape { geometry: g, ranks };
        let dense = sm120::dense::shapes_for(shape);
        assert!(dense.iter().any(|s| s.0 == "attn_out"));
        assert_eq!(
            sm120::dense::alias_for(shape, Role::PleValue),
            (4, "ple_value")
        );
        assert_eq!(
            sm120::dense::alias_for(shape, Role::MtpFc),
            (4, "ple_value")
        );
        let clusters = sm120::gemm::cluster_shapes(shape, ranks == 4).unwrap();
        assert!(!clusters.iter().any(|s| s.0 == "lin_qkv"));
        assert_eq!(
            clusters.iter().filter(|s| s.0 == "ple_value").count(),
            usize::from(ranks == 4)
        );
        for (name, segments, k, output, swiglu, blocks) in clusters {
            for bucket in model_qwen38_contract::contract::DECODE_ENTRY_BUCKETS {
                let mut query = bf16_query(bucket, segments.iter().sum(), k, output);
                if swiglu {
                    query.epilogues.swiglu =
                        Some(kernel_core::contract::SwigluColumns::ConcatenatedHalves);
                }
                let mut entry = fused::Entry::new(
                    MODEL.at(ranks),
                    &format!("{name}_c{bucket}"),
                    query,
                    fused::cluster_kind(k, bucket, blocks),
                    &segments,
                    kernels_sm120::invariance::Invariance::NONE,
                );
                entry.hc = Some(fused::HcGeometry {
                    width: g.hidden(),
                    ..sm120::gemm::HC
                });
                entry.contract().unwrap();
                assert!(entry.symbol.starts_with("chop_gemm_catalog_fixture_"));
                assert!(fused::hc_precompute_spec(&entry).is_err());
            }
        }
    }
    assert!(
        !model_qwen38_contract::geometry::GEOMETRIES
            .iter()
            .any(|listed| listed.id == g.id)
    );
    assert!(
        geometry::require_base_set("kernel-lab", &[FLASH, g])
            .unwrap_err()
            .contains("catalog_fixture")
    );
    assert!(geometry::require_base_set("release bundle", &[FLASH, g]).is_err());
    geometry::require_base_set("kernel-lab", &[FLASH]).unwrap();
}

#[test]
fn precompute_availability_matches_the_published_producers() {
    for owner in geometry::MODELS {
        let count = sm120::gemm::fused_catalog()
            .iter()
            .filter(|e| {
                e.profile == owner.model.at(4)
                    && kernels_sm120::gemm::fused::hc_precompute_spec(e).is_ok()
            })
            .count();
        assert_eq!(owner.hc_precompute(), count == 4);
    }
    assert!(geometry::MODELS[0].hc_precompute());
}

#[test]
fn synthetic_exclusions_match_its_layer_counts() {
    for source in [
        include_str!("../../../../../fixtures/qwen38/geometries/synthetic-small/config.json"),
        include_str!("../../../../../fixtures/qwen38/geometries/synthetic-wide/config.json"),
    ] {
        let config: serde_json::Value = serde_json::from_str(source).unwrap();
        let modules = config["quantization_config"]["modules_to_not_convert"]
            .as_array()
            .unwrap();
        for (prefix, count) in [
            (
                "model.language_model.layers.",
                config["text_config"]["num_hidden_layers"].as_u64().unwrap(),
            ),
            (
                "model.visual.blocks.",
                config["vision_config"]["depth"].as_u64().unwrap(),
            ),
            (
                "mtp.layers.",
                config["text_config"]["mtp"]["num_hidden_layers"]
                    .as_u64()
                    .unwrap(),
            ),
        ] {
            let indices = modules
                .iter()
                .filter_map(|m| m.as_str().unwrap().strip_prefix(prefix))
                .map(|m| m.split('.').next().unwrap().parse::<u64>().unwrap())
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(indices, (0..count).collect(), "{prefix}");
        }
    }
}
