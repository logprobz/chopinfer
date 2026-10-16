//! Metadata fixtures contain header facts only; no checkpoint payload is read.
#[path = "../projections_support/mod.rs"]
mod projections_support;
use kernel_core::contract::DType;
use model_contract::projections;
use model_mimo26_contract as model_contract;
use projection_spec::bind::{CheckpointMetadata, TensorInfo};
use projection_spec::{Declaration, Geometry};
use projections_support::checkpoint::{assert_header_facts, checkpoint_headers, read_header};
use projections_support::{bind, fixture, owner_scope};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The main checkpoint's header facts, with the DFlash drafter's and the
/// retained audio tokenizer encoder's.
fn full_headers(root: &Path) -> CheckpointMetadata {
    let mut metadata = checkpoint_headers(root);
    let h = read_header(&root.join("dflash"), "dflash_draft_model.safetensors");
    let other = CheckpointMetadata::from_headers(
        &BTreeMap::from([("dflash_draft_model.safetensors".into(), h)]),
        None,
        Value::Null,
    )
    .unwrap();
    for (name, t) in other.tensors {
        assert!(metadata.tensors.insert(name, t).is_none());
    }
    let h = read_header(&root.join("audio_tokenizer"), "model.safetensors");
    let audio = CheckpointMetadata::from_headers(
        &BTreeMap::from([("audio_tokenizer.safetensors".into(), h)]),
        None,
        Value::Null,
    )
    .unwrap();
    for (name, t) in audio.tensors {
        if name.starts_with("encoder.")
            && ![
                "._codebook.cluster_size",
                "._codebook.embed_avg",
                "._codebook.inited",
            ]
            .iter()
            .any(|suffix| name.contains(suffix))
        {
            assert!(metadata.tensors.insert(name, t).is_none());
        }
    }
    metadata
}
#[test]
fn declarations_validate_at_every_served_degree_and_pin_digest() {
    let mut digests = BTreeSet::new();
    for tp in [2, 4] {
        let d = projections::declaration(tp).unwrap();
        assert!(!d.projections.is_empty());
        digests.insert(d.digest());
    }
    assert_eq!(digests.len(), 1);
    assert_eq!(digests.first().unwrap(), digest().trim());
}
#[test]
fn providers_refuse_unknown_fields_ports_and_numerical_identity() {
    let g = projections::geometry(4).unwrap();
    let d = projections::declaration(4).unwrap();
    let p = d
        .projections
        .iter()
        .find(|p| {
            p.composition
                .as_ref()
                .is_some_and(|c| c.implementation.is_some())
        })
        .unwrap();
    let original = p
        .composition
        .as_ref()
        .unwrap()
        .implementation
        .as_ref()
        .unwrap();
    for field in ["unknown", "numerical_contract", "dtype"] {
        let mut v = original.clone();
        match field {
            "unknown" => {
                v.parameters["unexpected"] = Value::Bool(true);
            }
            "numerical_contract" => v.numerical_contract = "wrong.v1".into(),
            _ => v.inputs[0].dtype = projection_spec::DType::F32,
        }
        assert!(
            projections::registry()
                .unwrap()
                .resolve(
                    model_name(),
                    &v,
                    projection_spec::registry::Target::Projection,
                    &g
                )
                .is_err()
        );
    }
    let mut retired = original.clone();
    retired.id = "mimo26/legacy-fragment-experts.v1".into();
    retired.numerical_contract = "mimo26.experts.v1".into();
    retired.parameters = serde_json::json!({
        "source_numerical_contract": "mimo26.source-recipe.v1",
        "experts": 256, "top_k": 8, "scale_order": "block_scale_after_mma",
        "route_weight": "after-down"
    });
    assert!(
        projections::registry()
            .unwrap()
            .resolve(
                model_name(),
                &retired,
                projection_spec::registry::Target::Projection,
                &g
            )
            .unwrap_err()
            .0
            .contains("is not registered")
    );
}

fn model_name() -> &'static str {
    "mimo26"
}
fn digest() -> &'static str {
    include_str!("../fixtures/projections/declaration.sha256")
}
#[test]
fn binds_mopd_main_mtp_and_dflash_with_restarting_scale_grids() {
    let mut m = fixture(include_str!("../fixtures/projections/mopd.json"));
    assert_eq!(m.tensors.len(), 73081 + 63 + 389);
    owner_scope(&mut m);
    for tp in [2, 4] {
        let d = projections::declaration(tp).unwrap();
        let b = bind(&d, &projections::geometry(tp).unwrap(), &m).unwrap();
        let qkv: Vec<_> = b
            .operands
            .iter()
            .filter(|b| b.projection.ends_with(".qkv"))
            .collect();
        assert_eq!(qkv.len(), 51);
        assert!(
            qkv.iter()
                .all(|o| o.imported.parts.len() == 4 && o.part_bindings.len() == 4)
        );
        let context = b
            .assemblies
            .iter()
            .find(|a| a.projection == "dflash.context-kv")
            .unwrap();
        assert_eq!(context.members.len(), 10);
        assert!(context.tensors.iter().all(|t| t.len() == 1));
        for (execution, source, layer) in [
            ("mtp-attention-adapter", "mtp", 1),
            ("mtp-mlp-adapter", "mtp-mlp", 0),
            ("mtp-input-norm-adapter", "mtp-input_layernorm", 0),
            ("mtp-pre-mlp-norm-adapter", "mtp-pre_mlp_layernorm", 1),
            ("mtp-final-norm-adapter", "mtp-final_layernorm", 1),
        ] {
            assert!(b.layer_aliases.iter().any(|a| a.execution_stem == execution
                && a.source_stem == source
                && a.execution_index == layer
                && a.source_index == 0));
        }
        let qkv_views: Vec<_> = b
            .assemblies
            .iter()
            .filter(|a| a.projection == "dflash.qkv")
            .collect();
        assert_eq!(qkv_views.len(), 5);
        assert!(
            qkv_views
                .iter()
                .all(|a| a.members.iter().all(|m| m.inherit_split))
        );
    }
}
#[test]
fn binds_materialized_mtp_layers_but_executes_only_layer_zero() {
    use kernel_core::precision::{ExecutionPhase, format::FormatId};
    let d = projections::declaration(4).unwrap();
    d.admit(
        &projections::geometry(4).unwrap(),
        "mtp.qkv",
        Some(0),
        4,
        ExecutionPhase::Decode,
        FormatId::Fp8Block128,
    )
    .unwrap();
    assert!(
        d.admit(
            &projections::geometry(4).unwrap(),
            "mtp.qkv",
            Some(1),
            4,
            ExecutionPhase::Decode,
            FormatId::Fp8Block128
        )
        .is_err()
    );
}
#[test]
fn refuses_bad_qkv_companions_and_unaccounted_tensors() {
    let mut m = fixture(include_str!("../fixtures/projections/mopd.json"));
    owner_scope(&mut m);
    let d = projections::declaration(4).unwrap();
    let g = projections::geometry(4).unwrap();
    let name = "model.layers.0.self_attn.qkv_proj.weight_scale_inv";
    m.tensors.get_mut(name).unwrap().shape = vec![107, 32];
    assert!(bind(&d, &g, &m).is_err());
    let mut m = fixture(include_str!("../fixtures/projections/mopd.json"));
    owner_scope(&mut m);
    m.tensors.insert(
        "unexpected.weight".into(),
        TensorInfo {
            dtype: "BF16".into(),
            shape: vec![1, 1],
            file: None,
        },
    );
    assert!(
        bind(&d, &g, &m)
            .unwrap_err()
            .0
            .contains("unexpected.weight")
    );
}
#[test]
fn opt_in_full_headers() {
    let Ok(root) = std::env::var("CHOPINFER_PROJECTION_HEADERS_MIMO26") else {
        eprintln!("SKIP: CHOPINFER_PROJECTION_HEADERS_MIMO26 is unset");
        return;
    };
    let mut m = full_headers(Path::new(&root));
    assert_header_facts(
        &m,
        &fixture(include_str!("../fixtures/projections/mopd.json")),
    );
    owner_scope(&mut m);
    let d = projections::declaration(4).unwrap();
    bind(&d, &projections::geometry(4).unwrap(), &m).unwrap();
}

#[test]
fn head_use_implementation_ports_match_output_dtype() {
    let d = projections::declaration(4).unwrap();
    let d = d
        .expanded_profiles(&projections::geometry(4).unwrap())
        .unwrap();
    let head = d.projections.iter().find(|p| p.id == "head").unwrap();
    assert_eq!(head.uses.len(), 38);
    assert_eq!(
        head.uses
            .iter()
            .filter(|u| u.contract.as_ref().unwrap().output_dtype == DType::F32)
            .count(),
        19
    );
}

#[test]
fn sink_placement_and_eh_rank_k_match_production_tp() {
    for tp in [2, 4] {
        let d = projections::declaration(tp).unwrap();
        let g = projections::geometry(tp).unwrap();
        let production = if tp == 2 {
            model_contract::Tp::TP2
        } else {
            model_contract::Tp::TP4
        };
        for rank in 0..tp {
            let topology = projection_spec::derive::Topology { ranks: tp, rank };
            for id in ["swa.sink", "mtp.sink", "dflash.sink"] {
                let side = d.side_tensors.iter().find(|s| s.id == id).unwrap();
                let selection = side.selection(&g, topology).unwrap();
                let width = 64 / u64::from(tp);
                assert_eq!(
                    selection.ranges,
                    vec![u64::from(rank) * width..(u64::from(rank) + 1) * width]
                );
                assert_eq!(selection.shape, vec![width]);
            }
            let eh = d.projections.iter().find(|p| p.id == "mtp.eh").unwrap();
            let owners: Vec<_> = eh.inputs[0]
                .segments
                .iter()
                .filter_map(|segment| {
                    let range = segment.range(&d, &g, topology).unwrap();
                    (!range.is_empty()).then_some((segment, range))
                })
                .collect();
            assert_eq!(owners.len(), 1);
            let (owner, range) = &owners[0];
            let (hidden, offset) = production.mtp_eh_source(rank as usize);
            assert_eq!(
                owner.source,
                if hidden {
                    "normalized-hidden"
                } else {
                    "normalized-embedding"
                }
            );
            assert_eq!(range.start, offset);
            assert_eq!(range.end - range.start, production.mtp_eh_k());
        }
    }
}

#[test]
fn norm_sides_retain_kernel_owned_arithmetic_and_head_weight_placement() {
    for tp in [2, 4] {
        let d = projections::declaration(tp).unwrap();
        for side in d
            .side_tensors
            .iter()
            .filter(|s| s.kind == projection_spec::SideKind::Norm)
        {
            assert!(side.norm.is_none(), "{}", side.id);
            assert_eq!(side.admitted_dtypes(), &[DType::Bf16]);
            assert!(matches!(
                side.consumer,
                projection_spec::Consumer::Kernel { .. }
            ));
            if matches!(side.id.as_str(), "dflash.q_norm" | "dflash.k_norm") {
                assert_eq!(
                    side.consumer,
                    projection_spec::Consumer::Kernel {
                        kernel: side.id.clone()
                    }
                );
                for rank in 0..tp {
                    let selection = side
                        .selection(
                            &projections::geometry(tp).unwrap(),
                            projection_spec::derive::Topology { rank, ranks: tp },
                        )
                        .unwrap();
                    assert_eq!(selection.source_shape, vec![128]);
                    assert_eq!(selection.shape, vec![128]);
                    assert_eq!(
                        selection.ranges,
                        std::iter::once(0..128).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn expert_source_identity_and_native_stage_ownership_remain_explicit() {
    use kernel_core::precision::format::FormatId;
    use projection_spec::derive::{AdapterValue, StageExecution};
    let d = projections::declaration(4).unwrap();
    let ga = d.projections.iter().find(|p| p.id == "ga.qkv").unwrap();
    assert!(ga.composition.as_ref().unwrap().post.is_empty());
    for id in ["moe.experts.gate-up", "moe.experts.down"] {
        let p = d.projections.iter().find(|p| p.id == id).unwrap();
        assert_eq!(p.numerical_contract, "mimo26.source-recipe.v1");
        let composition = p.composition.as_ref().unwrap();
        for format in [FormatId::Mxfp4, FormatId::Fp8Block128, FormatId::Fp8Rows64] {
            assert!(composition.implementation_for(format).is_none());
        }
        assert!(
            !composition
                .numerical_contracts
                .contains_key(&FormatId::Fp8Block128)
        );
        assert!(
            !composition
                .numerical_contracts
                .contains_key(&FormatId::Fp8Rows64)
        );
    }
    let down = d
        .projections
        .iter()
        .find(|p| p.id == "moe.experts.down")
        .unwrap();
    assert_eq!(
        down.composition.as_ref().unwrap().post[0].contract.id,
        "mimo26/legacy-combine.v1"
    );
    let embedding = d
        .projections
        .iter()
        .find(|p| p.id == "dflash.embedding")
        .unwrap();
    let mask = &embedding.composition.as_ref().unwrap().post[0];
    assert_eq!(
        mask.phases,
        vec![kernel_core::precision::ExecutionPhase::Verify]
    );
    let params: projections::MaskParameters =
        serde_json::from_value(mask.contract.parameters.clone()).unwrap();
    for tp in [model_contract::Tp::TP2, model_contract::Tp::TP4] {
        let adapter = params.native_adapter(tp);
        assert_eq!(
            adapter.execution,
            StageExecution::Standalone {
                symbol: model_contract::sm120::bundle_symbol(tp, "dflash_mask_rows")
            }
        );
        for name in ["positions", "prefix_len", "token_batch", "live_tokens"] {
            let AdapterValue::Binding(binding) = &adapter.facts[name] else {
                panic!("native control must be a binding")
            };
            assert_eq!(binding.view, name);
            assert_eq!(binding.dtype, DType::I32);
            assert_eq!(
                binding.interpretation,
                if name == "live_tokens" {
                    Some(projection_spec::contracts::Interpretation::Unsigned)
                } else {
                    None
                }
            );
        }
    }
    let corrupt = &mut serde_json::to_value(params).unwrap();
    corrupt["adapter"]["facts"]["positions"]["value"]["dtype"] = Value::String("F32".into());
    let mut v = mask.contract.clone();
    v.parameters = corrupt.clone();
    assert!(
        projections::registry()
            .unwrap()
            .resolve(
                model_name(),
                &v,
                projection_spec::registry::Target::Stage,
                &projections::geometry(4).unwrap()
            )
            .is_err()
    );
}

fn ordinary_offer_specs(
    d: &Declaration,
    b: &projection_spec::bind::BoundDeclaration,
    g: &Geometry,
    id: &str,
    ranks: u32,
    rank: u32,
) -> Vec<projection_spec::derive::DerivedOperation> {
    use kernel_core::contract::{ActivationFormat, Target};
    use kernel_core::precision::format::{A16Recipe, KernelFamily, registry};
    use kernel_core::precision::{ActivationSpec, ArithmeticSpec, Rounding};
    use projection_spec::derive::{
        GeometryMode, ImportedProjection, InstanceKey, ProjectionContext, SourceMatrix, Topology,
    };
    let p = d.projections.iter().find(|p| p.id == id).unwrap();
    let family = if matches!(p.kind, projection_spec::ProjectionKind::LookupTable { .. }) {
        KernelFamily::Lookup
    } else {
        KernelFamily::Dense
    };
    let first = b.operands.iter().find(|o| o.projection == id).unwrap();
    let mut operands = b
        .operands
        .iter()
        .filter(|o| o.projection == id && o.layer == first.layer)
        .collect::<Vec<_>>();
    operands.sort_by_key(|o| o.operand);
    assert!(p.parts.is_empty());
    assert_eq!(operands.len(), p.operand_count() as usize);
    let weights = operands
        .iter()
        .map(|o| o.imported.weight.clone())
        .collect::<Vec<_>>();
    let numerical = p
        .composition
        .as_ref()
        .and_then(|c| c.numerical_contracts.get(&first.format));
    let recipe = numerical.map_or(A16Recipe::ScaledWeight, |r| r.a16_recipe);
    let numerical_contract = numerical
        .map_or(&p.numerical_contract, |r| &r.numerical_contract)
        .clone();
    let imported = ImportedProjection {
        operation_id: id.into(),
        target: Target::Sm120f,
        family,
        format: first.format,
        geometry: GeometryMode::RankLocal,
        activations: if family == KernelFamily::Lookup {
            Vec::new()
        } else {
            vec![ActivationSpec::bf16(); weights.len()]
        },
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Bf16,
            accumulator: DType::F32,
            output: DType::Bf16,
            decode_order: registry::by_id(first.format)
                .unwrap()
                .decode_order(family, ActivationFormat::Bf16, recipe)
                .unwrap(),
            rounding: Rounding::NearestEven,
            numerical_contract,
        },
        a16_recipe: recipe,
        matrices: weights
            .iter()
            .enumerate()
            .map(|(operand, w)| SourceMatrix {
                weight: w.clone(),
                expert: 0,
                operand: operand as u32,
                source_expert: 0,
                target_rows: 0..w.logical_shape[0],
                source_rows: 0..w.logical_shape[0],
                target_k: 0..w.logical_shape[1],
                source_k: 0..w.logical_shape[1],
            })
            .collect(),
        weights,
        preparation: None,
    };
    p.uses
        .iter()
        .map(|usage| {
            ProjectionContext {
                bound: b,
                source_instance: InstanceKey {
                    projection: id.into(),
                    layer: first.layer,
                },
                selections: None,
                declaration: d,
                projection: p,
                geometry: g,
                topology: Topology { ranks, rank },
                layer: first.layer,
                source_layer: first.layer,
                usage,
            }
            .ordinary(&imported)
            .unwrap_or_else(|e| panic!("{id}/TP{ranks}/rank{rank}: {e}"))
        })
        .collect()
}

#[test]
fn ordinary_declaration_specs_reproduce_dense_catalog_bodies() {
    use kernels_sm120::precision::{KernelOffer, generated};
    use model_contract::capabilities::{self, Projection};
    let committed: Vec<generated::Adoption> =
        serde_json::from_str(include_str!("../../../bundle/projection-adoptions.json")).unwrap();
    let mut m = fixture(include_str!("../fixtures/projections/mopd.json"));
    owner_scope(&mut m);
    let catalog = model_contract::sm120::gemm::catalog();
    let mut comparisons = 0;
    for tp in model_contract::Tp::SUPPORTED {
        let g = projections::geometry(tp.ranks()).unwrap();
        let d = projections::declaration(tp.ranks()).unwrap();
        let b = bind(&d, &g, &m).unwrap();
        let d = d.expanded_profiles(&g).unwrap();
        for (id, projection) in [
            ("ga.output", Projection::Output),
            ("swa.output", Projection::Output),
            ("dense.down", Projection::MlpDown),
            ("dense.mlp.gate-up", Projection::MlpGateUp),
            ("head", Projection::Head { f32: false }),
            ("router", Projection::Router),
        ] {
            for rank in 0..tp.ranks() {
                for derived in ordinary_offer_specs(&d, &b, &g, id, tp.ranks(), rank) {
                    let projection = if id == "head" {
                        Projection::Head {
                            f32: derived.operation.arithmetic.output == DType::F32,
                        }
                    } else {
                        projection
                    };
                    let name = capabilities::sm120::symbol(projection, tp, derived.requested);
                    let entry = catalog.iter().find(|entry| entry.symbol == name).unwrap();
                    let old =
                        KernelOffer::catalog_dense(entry.instantiation.clone(), derived.requested)
                            .unwrap();
                    let fresh = capabilities::sm120::scheduled_offer(
                        &derived.operation,
                        derived.requested,
                        &[],
                    )
                    .unwrap();
                    let adoptions = generated::adopt(&fresh, &old).unwrap();
                    for adoption in &adoptions {
                        assert!(
                            committed
                                .iter()
                                .rev()
                                .find(|old| old.symbol == adoption.symbol)
                                .is_some_and(|old| old.generated == adoption.generated
                                    && old.entry_sha256 == adoption.entry_sha256),
                            "{id}: generated identity changed: {adoption:?}"
                        );
                    }
                    let restored = capabilities::sm120::scheduled_offer(
                        &derived.operation,
                        derived.requested,
                        &adoptions,
                    )
                    .unwrap();
                    assert_eq!(
                        restored.offer.capability().unwrap(),
                        old.capability().unwrap(),
                        "{id}"
                    );
                    assert_eq!(
                        format!("{:?}", restored.offer.handle()),
                        format!("{:?}", old.handle()),
                        "{id}"
                    );
                    comparisons += 1;
                }
            }
        }
    }
    assert!(comparisons > 400);
}

#[test]
fn ordinary_declaration_specs_reproduce_lookup_catalog_identities() {
    use kernels_sm120::precision::generated::{Adoption, comparison};
    use model_contract::capabilities::{self, Projection};
    let committed: Vec<Adoption> =
        serde_json::from_str(include_str!("../../../bundle/projection-adoptions.json")).unwrap();
    let mut m = fixture(include_str!("../fixtures/projections/mopd.json"));
    owner_scope(&mut m);
    let cubins = kernels_sm120::row::cubin_specs(model_contract::sm120::catalogs(), None);
    let mut comparisons = 0;
    for tp in model_contract::Tp::SUPPORTED {
        let g = projections::geometry(tp.ranks()).unwrap();
        let d = projections::declaration(tp.ranks()).unwrap();
        let b = bind(&d, &g, &m).unwrap();
        let d = d.expanded_profiles(&g).unwrap();
        for rank in 0..tp.ranks() {
            for derived in ordinary_offer_specs(&d, &b, &g, "embedding", tp.ranks(), rank) {
                let old =
                    capabilities::capability(Projection::Embedding, tp, derived.requested).unwrap();
                let (capability, adoptions) = comparison::catalog(
                    &derived.operation,
                    old.clone(),
                    &cubins,
                    &BTreeMap::from([(
                        old.stages[0].clone(),
                        kernels_sm120::precision::generated::identity::retained_row(
                            model_contract::sm120::workspace::row::entries_for(tp)
                                .find(|e| e.symbol == old.stages[0])
                                .unwrap(),
                            &old,
                        )
                        .unwrap(),
                    )]),
                )
                .unwrap();
                assert_eq!(capability, old);
                assert!(!adoptions.is_empty());
                for adoption in adoptions {
                    assert!(
                        committed
                            .iter()
                            .rev()
                            .find(|old| old.symbol == adoption.symbol)
                            .is_some_and(|old| old.generated == adoption.generated
                                && old.entry_sha256 == adoption.entry_sha256),
                        "generated lookup identity changed: {adoption:?}"
                    );
                }
                comparisons += 1;
            }
        }
    }
    assert!(comparisons > 100);
}

/// Drafter shapes come from each geometry's DFlash configuration, so a
/// data-only geometry's drafter binds its own tensors.
#[test]
fn dflash_shapes_name_geometry_keys() {
    let declaration: serde_json::Value =
        serde_json::from_str(include_str!("../../projections.json")).unwrap();
    let literal = |shape: &serde_json::Value| {
        shape
            .as_array()
            .unwrap()
            .iter()
            .any(serde_json::Value::is_number)
    };
    for section in ["projections", "side_tensors"] {
        for entry in declaration[section].as_array().unwrap() {
            let id = entry["id"].as_str().unwrap();
            if id.starts_with("dflash.") && id != "dflash.embedding" {
                assert!(!literal(&entry["shape"]), "{id} shape {}", entry["shape"]);
                for member in entry["assembly"].as_array().into_iter().flatten() {
                    assert!(member["rows"]["length"].is_string(), "{id} member rows");
                }
            }
        }
    }
    let stem = declaration["layer_stems"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["id"] == "dflash")
        .unwrap();
    assert_eq!(stem["indices"]["end"], "dflash_layers");
}
