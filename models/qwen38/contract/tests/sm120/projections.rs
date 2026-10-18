//! Metadata fixtures contain header facts only; no checkpoint payload is read.
#[path = "../projections_support/mod.rs"]
mod projections_support;
use kernel_core::contract::DType;
use model_contract::projections;
use model_qwen38_contract as model_contract;
use projection_spec::bind::TensorInfo;
use projection_spec::{Declaration, Geometry};
use projections_support::checkpoint::{assert_header_facts, checkpoint_headers};
use projections_support::exports::{FP8, Layout};
use projections_support::{bind, fixture, owner_scope};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;

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
/// The drafter, its published ports, the MTP layer's execution and the
/// offered formats are the only declaration changes since the previous pin:
/// without them the declaration keeps that digest.
#[test]
fn declaration_without_the_added_fields_keeps_the_previous_digest() {
    let mut d = projections::declaration(4).unwrap();
    let stems: Vec<String> = d.drafters.iter().flat_map(|d| d.stems.clone()).collect();
    d.drafters.clear();
    for p in &mut d.projections {
        p.publishes_as = None;
        p.offers.clear();
    }
    for stem in &mut d.layer_stems {
        if stems.contains(&stem.id) {
            stem.execution = None;
        }
    }
    assert_eq!(
        d.digest(),
        "fcfe35da8e34910b7bfe50ccaf2bd3f218b52ec5870a4383097696f9a9e71497"
    );
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
}

fn model_name() -> &'static str {
    "qwen38"
}
fn digest() -> &'static str {
    include_str!("../fixtures/projections/declaration.sha256")
}

#[test]
fn binds_the_fp8_export_and_both_mixed_layouts_and_keeps_f32_norm_sources() {
    for (name, text, count) in [
        ("fp8", FP8, 152089),
        (
            Layout::Nvfp4RoutedFp8Mtp.name(),
            Layout::Nvfp4RoutedFp8Mtp.text(),
            299212,
        ),
        (
            Layout::Mxfp8DenseNvfp4.name(),
            Layout::Mxfp8DenseNvfp4.text(),
            301593,
        ),
    ] {
        let mut metadata = fixture(text);
        assert_eq!(metadata.tensors.len(), count, "{name}");
        let d = owner_scope(&projections::declaration(4).unwrap(), &mut metadata);
        let b = bind(&d, &projections::geometry(4).unwrap(), &metadata)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(b.operands.len() > 75000);
        use kernel_core::precision::{ExecutionPhase, format::FormatId};
        // The MTP layer executes after the main layers.
        let mtp = projections::geometry(4).unwrap()["layers"];
        for (projection, layer, ranks) in [
            ("linear.a", Some(0), 2),
            ("head", None, 2),
            ("mtp.q", Some(mtp), 4),
            ("mtp.fc-hidden", None, 4),
        ] {
            let admission = d.admit_checkpoint(
                &projections::geometry(4).unwrap(),
                &b,
                projection_spec::Admission {
                    projection,
                    layer,
                    ranks,
                    phase: ExecutionPhase::Decode,
                    format: FormatId::Bf16,
                },
            );
            // Every layout binds TP2, TP4 and MTP: the declaration keeps
            // no checkpoint-conditioned refusal.
            assert!(admission.is_ok(), "{name}: {projection}/TP{ranks}");
        }

        if name == Layout::Mxfp8DenseNvfp4.name() {
            assert_eq!(
                b.side_tensors
                    .iter()
                    .filter(|n| metadata.tensors[*n].dtype == "F32")
                    .count(),
                100
            );
            let f32_views: Vec<_> = b
                .side_views
                .iter()
                .filter(|v| metadata.tensors[&v.tensor].dtype == "F32")
                .collect();
            assert_eq!(f32_views.len(), 100);
            assert!(f32_views.iter().all(|v| v.view.dtype == DType::F32
                && v.view.conversion == projection_spec::contracts::ViewConversion::Identity));
            assert!(
                b.fusion_groups
                    .iter()
                    .filter(|f| f.id == "linear.input-fusion" || f.id == "full.input-fusion")
                    .all(|f| !f.fused)
            );
        }
    }
}
#[test]
fn refuses_unknown_missing_companions_and_partial_encoder_owner() {
    let mut m = fixture(Layout::Mxfp8DenseNvfp4.text());
    let d = owner_scope(&projections::declaration(4).unwrap(), &mut m);
    let g = projections::geometry(4).unwrap();
    let name = "model.language_model.layers.0.linear_attn.in_proj_a.weight_scale";
    m.tensors.remove(name);
    assert!(bind(&d, &g, &m).unwrap_err().0.contains(name));
    let mut m = fixture(FP8);
    let d = owner_scope(&projections::declaration(4).unwrap(), &mut m);
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
fn mixed_topologies_need_no_checkpoint_condition() {
    use kernel_core::precision::{ExecutionPhase, format::FormatId};
    let d = projections::declaration(4).unwrap();
    assert!(
        d.layer_stems
            .iter()
            .flat_map(|s| &s.support)
            .all(|r| r.enabled && r.when.is_none())
    );
    assert!(d.checkpoint_conditions(std::iter::empty()).is_empty());
    let mtp = projections::geometry(4).unwrap()["layers"];
    for (projection, layer) in [("linear.a", Some(0)), ("mtp.q", Some(mtp))] {
        for ranks in [2, 4] {
            for format in [FormatId::Bf16, FormatId::Nvfp4] {
                d.admit(
                    &projections::geometry(4).unwrap(),
                    projection,
                    layer,
                    ranks,
                    ExecutionPhase::Decode,
                    format,
                )
                .unwrap_or_else(|e| panic!("{projection}/TP{ranks}/{format:?}: {e}"));
            }
        }
    }
}
/// The headers of each checkpoint `CHOPINFER_PROJECTION_HEADERS_QWEN38`
/// selects (one checkpoint, or a directory whose children are checkpoints)
/// bind closed coverage. The FP8 export's headers and quantization_config
/// equal its capture. A checkpoint with a synthesized layout's tensor count
/// holds exactly that layout's tensors, dtypes and shapes; its own
/// quantization_config stands for the synthesized rules.
#[test]
fn opt_in_full_headers() {
    let Ok(root) = std::env::var("CHOPINFER_PROJECTION_HEADERS_QWEN38") else {
        eprintln!("SKIP: CHOPINFER_PROJECTION_HEADERS_QWEN38 is unset");
        return;
    };
    let root = Path::new(&root);
    let roots = if root.join("config.json").is_file() {
        vec![root.to_path_buf()]
    } else {
        let mut children: Vec<_> = std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.join("config.json").is_file())
            .collect();
        children.sort();
        children
    };
    assert!(!roots.is_empty(), "{} holds no checkpoint", root.display());
    for root in roots {
        let mut metadata = checkpoint_headers(&root);
        let count = metadata.tensors.len();
        let fp8 = fixture(FP8);
        if count == fp8.tensors.len() {
            assert_header_facts(&metadata, &fp8);
        } else if let Some(mut expected) = Layout::ALL
            .into_iter()
            .map(|layout| fixture(layout.text()))
            .find(|layout| layout.tensors.len() == count)
        {
            expected.quantization_config = metadata.quantization_config.clone();
            assert_header_facts(&metadata, &expected);
        } else {
            eprintln!(
                "{}: {count} tensors match no captured or synthesized layout; binding only",
                root.display()
            );
        }
        let d = owner_scope(&projections::declaration(4).unwrap(), &mut metadata);
        bind(&d, &projections::geometry(4).unwrap(), &metadata)
            .unwrap_or_else(|e| panic!("{}: {e}", root.display()));
    }
}

#[test]
fn a_partial_encoder_owner_cannot_be_omitted() {
    let mut m = fixture(Layout::Mxfp8DenseNvfp4.text());
    m.tensors.remove("model.visual.pos_embed.weight");
    let d = projections::declaration(4).unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| owner_scope(&d, &mut m))).is_err()
    );
}

#[test]
fn kernel_owned_linear_norm_retains_shape_dtype_and_legacy_gate() {
    for tp in [2, 4] {
        let d = projections::declaration(tp).unwrap();
        let side = d
            .side_tensors
            .iter()
            .find(|s| s.id == "linear.norm-norm")
            .unwrap();
        assert!(side.norm.is_none());
        assert_eq!(
            side.shape,
            vec![projection_spec::Dim::Key("linear_value_head".into())]
        );
        assert_eq!(
            side.shape[0]
                .resolve(&projections::geometry(tp).unwrap())
                .unwrap(),
            128
        );
        assert_eq!(side.admitted_dtypes(), &[DType::Bf16]);
        assert_eq!(
            side.consumer,
            projection_spec::Consumer::Kernel {
                kernel: "linear.gated_norm".into()
            }
        );
        assert!(side.placement.is_default());
        let consumer = d
            .projections
            .iter()
            .find(|p| p.id == "linear.output")
            .unwrap();
        let implementation = consumer
            .composition
            .as_ref()
            .unwrap()
            .implementation_for(kernel_core::precision::format::FormatId::Bf16)
            .unwrap();
        assert_eq!(
            implementation.parameters["input_gate"]["projection"],
            "linear.z"
        );
        assert_eq!(
            implementation.parameters["input_gate"]["activation"],
            "sigmoid"
        );
        assert_eq!(
            implementation.parameters["input_gate"]["order"],
            "norm-then-gate"
        );
    }
}

#[test]
fn per_stream_norm_parameters_preserve_separate_affine_slices() {
    let d = projections::declaration(4).unwrap();
    for id in [
        "main.attn.hc-norm",
        "main.mlp.hc-norm",
        "ple.norm-key",
        "ple.norm-query",
        "ple.norm-conv",
    ] {
        let n = d
            .side_tensors
            .iter()
            .find(|s| s.id == id)
            .unwrap()
            .norm
            .as_ref()
            .unwrap();
        let g = projections::geometry(4).unwrap();
        assert_eq!(n.affine, projection_spec::contracts::NormAffine::PerGroup);
        assert_eq!(n.reduction_width.resolve(&g).unwrap(), 2560);
        assert_eq!(n.groups.resolve(&g).unwrap(), 4);
    }
}
#[test]
fn fp8_metadata_binds_at_tp2() {
    let mut metadata = fixture(FP8);
    let d = owner_scope(&projections::declaration(2).unwrap(), &mut metadata);
    bind(&d, &projections::geometry(2).unwrap(), &metadata).unwrap();
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
fn side_placement_matches_rank_local_legacy_shapes() {
    for tp in [2, 4] {
        let d = projections::declaration(tp).unwrap();
        let g = projections::geometry(tp).unwrap();
        for rank in 0..tp {
            let topology = projection_spec::derive::Topology { ranks: tp, rank };
            for id in ["linear.a_log", "linear.dt_bias"] {
                let side = d.side_tensors.iter().find(|s| s.id == id).unwrap();
                let selection = side.selection(&g, topology).unwrap();
                let width = 48 / u64::from(tp);
                assert_eq!(
                    selection.ranges,
                    vec![u64::from(rank) * width..(u64::from(rank) + 1) * width]
                );
                assert_eq!(selection.shape, vec![width]);
                assert_eq!(side.views[0].dtype, DType::F32);
            }
            let conv = d
                .side_tensors
                .iter()
                .find(|s| s.id == "linear.conv")
                .unwrap();
            let selection = conv.selection(&g, topology).unwrap();
            let expected: Vec<_> = [(0, 2048), (2048, 2048), (4096, 6144)]
                .into_iter()
                .map(|(base, width)| {
                    base + u64::from(rank) * width / u64::from(tp)
                        ..base + (u64::from(rank) + 1) * width / u64::from(tp)
                })
                .collect();
            assert_eq!(selection.ranges, expected);
            assert_eq!(selection.shape, vec![10240 / u64::from(tp), 1, 4]);
            let ple = d.side_tensors.iter().find(|s| s.id == "ple.conv").unwrap();
            assert_eq!(
                ple.selection(&g, topology).unwrap().shape,
                vec![10240, 1, 4]
            );
        }
    }
}

#[test]
fn shared_norm_metadata_is_limited_to_matching_consumers() {
    for tp in [2, 4] {
        let d = projections::declaration(tp).unwrap();
        for side in d
            .side_tensors
            .iter()
            .filter(|s| s.kind == projection_spec::SideKind::Norm)
        {
            let shared = side.id.ends_with("hc-norm")
                || side.id.ends_with("mixer.norm")
                || side.id.starts_with("ple.norm-")
                || side.id.starts_with("mtp.pre-fc-");
            assert_eq!(side.norm.is_some(), shared, "{}", side.id);
            if let Some(norm) = &side.norm {
                let expected = if side.id.starts_with("mtp.pre-fc-") {
                    kernel_core::contract::NormReduction::XorThenSerial
                } else {
                    kernel_core::contract::NormReduction::ShuffleDown
                };
                assert_eq!(norm.reduction, expected);
            } else {
                assert_eq!(side.admitted_dtypes(), &[DType::Bf16]);
            }
        }
    }
}

#[test]
fn source_formats_select_legacy_experts_and_linear_output_explicitly() {
    use kernel_core::precision::format::FormatId;
    let d = projections::declaration(4).unwrap();
    for id in [
        "main.experts.gate-up",
        "main.experts.down",
        "mtp.experts.gate-up",
        "mtp.experts.down",
    ] {
        let p = d.projections.iter().find(|p| p.id == id).unwrap();
        let composition = p.composition.as_ref().unwrap();
        let legacy = composition
            .implementation_for(FormatId::Fp8Block128)
            .unwrap();
        assert_eq!(legacy.id, "qwen38/legacy-fragment-experts.v1");
        assert_eq!(
            legacy.numerical_contract,
            composition.numerical_contracts[&FormatId::Fp8Block128].numerical_contract
        );
        for native in [FormatId::Nvfp4, FormatId::Nvfp4WeightOnly] {
            assert!(composition.implementation_for(native).is_none());
        }
    }
    let output = d
        .projections
        .iter()
        .find(|p| p.id == "linear.output")
        .unwrap();
    let c = output.composition.as_ref().unwrap();
    assert_eq!(
        c.implementation_for(FormatId::Bf16).unwrap().id,
        "qwen38/legacy-fused-dense.v1"
    );
    assert!(c.implementation_for(FormatId::Mxfp8).is_none());
    let shards = d
        .projections
        .iter()
        .find(|p| p.id == "ngram")
        .unwrap()
        .lookup
        .as_ref()
        .unwrap()
        .shards
        .resolve(&projections::geometry(4).unwrap())
        .unwrap();
    assert_eq!(shards.len(), 128);
    assert!(shards[127].source.tensor.ends_with("shard_127.weight"));
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
        family: KernelFamily::Dense,
        format: first.format,
        geometry: GeometryMode::RankLocal,
        activations: vec![ActivationSpec::bf16(); weights.len()],
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Bf16,
            accumulator: DType::F32,
            output: DType::Bf16,
            decode_order: registry::by_id(first.format)
                .unwrap()
                .decode_order(KernelFamily::Dense, ActivationFormat::Bf16, recipe)
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
fn ordinary_declaration_specs_reproduce_mixed_offer_bodies() {
    use kernels_sm120::precision::generated;
    use model_contract::precision::DenseRole;
    let committed: Vec<generated::Adoption> =
        serde_json::from_str(include_str!("../../../bundle/projection-adoptions.json")).unwrap();
    let mut padded_linear = BTreeSet::new();
    let mut comparisons = 0;
    for text in [FP8, Layout::Mxfp8DenseNvfp4.text()] {
        let mut m = fixture(text);
        let ranks = model_contract::contract::TOPOLOGY.ranks;
        let g = projections::geometry(ranks.get()).unwrap();
        let d = owner_scope(&projections::declaration(ranks.get()).unwrap(), &mut m);
        let b = bind(&d, &g, &m).unwrap();
        let d = d.expanded_profiles(&g).unwrap();
        let shape = model_contract::contract::Shape::flash(ranks);
        for (id, role) in [
            ("linear.z", DenseRole::LinearZ),
            ("linear.a", DenseRole::LinearA),
            ("linear.b", DenseRole::LinearB),
            ("full.q", DenseRole::AttentionQ),
            ("full.k", DenseRole::AttentionK),
            ("full.v", DenseRole::AttentionV),
            ("full.index", DenseRole::IndexQk),
            ("full.output", DenseRole::AttentionOutput),
            ("linear.output", DenseRole::Output),
            ("main.shared.gate-up", DenseRole::SharedGateUp),
            ("main.shared.down", DenseRole::SharedDown),
        ] {
            let projection = d.projections.iter().find(|p| p.id == id).unwrap();
            let format = b
                .operands
                .iter()
                .find(|o| o.projection == id)
                .unwrap()
                .format;
            if let Some(implementation) = projection
                .composition
                .as_ref()
                .and_then(|c| c.implementation_for(format))
            {
                // A4 owns this fused BF16 composition. The full hand-list
                // comparison continues to cover it until its variant lands.
                assert_eq!(id, "linear.output");
                assert_eq!(implementation.id, "qwen38/legacy-fused-dense.v1");
                continue;
            }
            for rank in 0..shape.ranks() {
                for derived in ordinary_offer_specs(&d, &b, &g, id, shape.ranks(), rank) {
                    let operation = &derived.operation;
                    let weight = operation.weights[0].format;
                    let published = model_contract::sm120::generated::publish(
                        shape,
                        role.projection_id(shape, None),
                        weight,
                        derived.requested,
                    )
                    .unwrap();
                    assert_eq!(
                        operation.matrix_geometry().unwrap(),
                        published.capability.geometry,
                        "{id}"
                    );
                    if matches!(role, DenseRole::LinearA | DenseRole::LinearB)
                        && weight == kernel_core::contract::WeightFormat::Bf16
                    {
                        assert_eq!(operation.matrix_geometry().unwrap().rows, 64);
                        padded_linear.insert((id, rank));
                    }
                    let derived_offer = model_contract::sm120::generated::scheduled_offer(
                        shape,
                        operation,
                        derived.requested,
                        &committed,
                    )
                    .unwrap_or_else(|e| panic!("{id} {weight:?}: {e}"));
                    let capability = derived_offer.offer.capability().unwrap();
                    for (identity, symbol) in
                        derived_offer.identities.iter().zip(&capability.stages)
                    {
                        assert!(
                            committed
                                .iter()
                                .rev()
                                .find(|old| old.symbol == *symbol)
                                .is_some_and(|old| old.generated == identity.generated
                                    && old.entry_sha256 == identity.entry_sha256),
                            "{id}: generated identity changed: {identity:?}"
                        );
                    }
                    assert_eq!(
                        capability.stages[..],
                        published.offer.capability().unwrap().stages[..],
                        "{id}"
                    );
                    assert_eq!(
                        format!("{:?}", derived_offer.offer.handle()),
                        format!("{:?}", published.offer.handle()),
                        "{id}"
                    );
                    comparisons += 1;
                }
            }
        }
    }
    assert!(comparisons > 1000);
    assert_eq!(
        padded_linear.len(),
        2 * model_contract::contract::TOPOLOGY.ranks.get() as usize
    );
}

/// The declared MTP drafter owns exactly what the `mtp` checkpoint prefix
/// names: its stems, projections, fusion groups, side tensors and sources.
#[test]
fn the_declared_drafter_owns_the_mtp_layer() {
    use model_contract::contract::{Ranks, Shape};
    for ranks in [Ranks::TP2, Ranks::TP4] {
        let shape = Shape::flash(ranks);
        let ports = projections::ports::get(shape).unwrap();
        let d = &ports.declaration;
        assert_eq!(d.drafters.len(), 1);
        for p in &d.projections {
            assert_eq!(d.drafter_of(&p.id).is_some(), p.id.starts_with("mtp."));
        }
        for g in &d.fusion_groups {
            assert_eq!(
                d.drafter_of_group(&g.id).is_some(),
                g.id.starts_with("mtp.")
            );
        }
        for s in &d.side_tensors {
            assert_eq!(d.drafter_of_side(s).is_some(), s.id.starts_with("mtp."));
        }
        let stems: BTreeSet<_> = d.drafters[0].stems.iter().cloned().collect();
        let prefixed: BTreeSet<_> = d
            .layer_stems
            .iter()
            .filter(|s| s.template.starts_with("mtp."))
            .map(|s| s.id.clone())
            .collect();
        assert_eq!(stems, prefixed);
        let sources = ports.drafter_sources().unwrap();
        let prefixed: BTreeSet<_> = d
            .source_positions(&ports.geometry)
            .unwrap()
            .into_keys()
            .filter(|name| name.starts_with("mtp."))
            .collect();
        assert_eq!(sources, &prefixed);
        // The prepared component records the declared operations at the
        // MTP layer, in declaration order.
        let layer = u64::from(shape.mtp_layer());
        let declared: Vec<_> = d.drafters[0]
            .operations
            .iter()
            .map(|id| ports.operation(id, Some(layer)).unwrap())
            .collect();
        assert_eq!(
            model_contract::prepared::drafter_operations(shape).unwrap(),
            declared
        );
        // A geometry without the MTP layer omits every drafter member.
        let mut omitted = projections::declaration_for(shape).unwrap();
        omitted.omit_drafter("mtp").unwrap();
        assert!(omitted.drafters.is_empty());
        assert!(
            omitted
                .projections
                .iter()
                .all(|p| !p.id.starts_with("mtp."))
        );
        assert!(
            omitted
                .side_tensors
                .iter()
                .all(|s| !s.id.starts_with("mtp."))
        );
        assert!(
            omitted
                .fusion_groups
                .iter()
                .all(|g| !g.id.starts_with("mtp."))
        );
        assert!(omitted.layer_stems.iter().all(|s| !s.id.starts_with("mtp")));
    }
}

/// The declared offers publish the ports and formats the former constants
/// named: BF16 and MXFP8 at every dense port, NVFP4 experts and lookup.
#[test]
fn declared_offers_keep_the_published_ports() {
    use kernel_core::contract::WeightFormat;
    use model_contract::contract::Shape;
    use model_contract::precision::DenseRole;
    use model_contract::sm120::generated;
    for geometry in model_contract::geometry::GEOMETRIES {
        for topology in model_contract::contract::TOPOLOGIES {
            let shape = Shape::new(geometry, topology.ranks);
            let mut expected: Vec<(&str, WeightFormat)> = Vec::new();
            for role in DenseRole::all_for(shape).unwrap() {
                for weight in [WeightFormat::Bf16, WeightFormat::Mxfp8] {
                    expected.push((role.projection_id(shape, None), weight));
                }
            }
            for id in ["main.experts.gate-up", "main.experts.down", "ngram"] {
                expected.push((id, WeightFormat::Nvfp4));
            }
            assert_eq!(generated::published_ports(shape).unwrap(), expected);
        }
    }
}

/// A declaration that offers another format at a port publishes it with
/// no Rust change: NVFP4 dense generates offers. MXFP8 grouped experts have
/// no shared kernel and are refused naming the port and format.
#[test]
fn offered_formats_are_declaration_data() {
    use kernel_core::contract::WeightFormat;
    use model_contract::contract::{Ranks, Shape};
    use model_contract::precision::uses;
    use model_contract::sm120::generated;
    let shape = Shape::flash(Ranks::TP4);
    let usage = uses().unwrap()[0];
    let committed = projections::ports::get(shape).unwrap();
    let error = generated::generate(committed, "full.q", WeightFormat::Nvfp4, usage)
        .err()
        .unwrap();
    assert!(error.contains("composes no Nvfp4 recipe"), "{error}");
    let offered = |id: &str, format: &str| {
        let d = parsed_offer(id, format).unwrap();
        projections::ports::Ports::with_declaration(shape, d).unwrap()
    };
    let ports = offered("full.q", "nvfp4");
    assert!(
        generated::offered_ports(&ports)
            .unwrap()
            .contains(&("full.q", WeightFormat::Nvfp4))
    );
    let published = generated::generate(&ports, "full.q", WeightFormat::Nvfp4, usage).unwrap();
    assert_eq!(published.capability.weight, WeightFormat::Nvfp4);
    assert!(!published.capability.stages.is_empty());
    // The shared packed grouped engine also takes MXFP8 weights.
    let ports = offered("main.experts.gate-up", "mxfp8");
    assert!(
        generated::offered_ports(&ports)
            .unwrap()
            .contains(&("main.experts.gate-up", WeightFormat::Mxfp8))
    );
    let published =
        generated::generate(&ports, "main.experts.gate-up", WeightFormat::Mxfp8, usage).unwrap();
    assert_eq!(published.capability.weight, WeightFormat::Mxfp8);
    // A format with no parameterless weight, such as block FP8, has no
    // generated offer; its registered implementation publishes it.
    let error = parsed_offer("full.q", "fp8_block128").unwrap_err();
    assert!(
        error.0.contains("has no generated offer weight format"),
        "{error}"
    );
}

/// The committed declaration with `format` also offered at projection `id`,
/// with a fixture recipe.
fn parsed_offer(id: &str, format: &str) -> Result<Declaration, projection_spec::Error> {
    let mut value: Value = serde_json::from_str(projections::DECLARATION).unwrap();
    let p = value["projections"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|p| p["id"] == id)
        .unwrap();
    p["offers"].as_array_mut().unwrap().push(format.into());
    p["composition"]["numerical_contracts"][format] =
        serde_json::json!({"numerical_contract": "fixture.v1", "a16_recipe": "scaled_weight"});
    projection_spec::parse(
        &value.to_string(),
        &projections::geometry(4).unwrap(),
        &projections::registry().unwrap(),
    )
}

/// A format offered only by the later of two equal-geometry ports is still
/// built: the inventory shares offers by kernel identity, not by geometry.
#[test]
fn every_offered_format_is_built_into_the_inventory() {
    use model_contract::contract::{Ranks, Shape};
    use model_contract::sm120::generated;
    use std::cell::RefCell;
    use std::sync::Arc;
    let shape = Shape::flash(Ranks::TP4);
    let ports: &'static _ = Box::leak(Box::new(
        projections::ports::Ports::with_declaration(
            shape,
            parsed_offer("full.v", "nvfp4").unwrap(),
        )
        .unwrap(),
    ));
    assert_eq!(
        ports.geometry("full.k", false).unwrap(),
        ports.geometry("full.v", false).unwrap()
    );
    let published = RefCell::new(Vec::new());
    let inventory = generated::inventory_in(ports, |id, weight, usage| {
        let generated = Arc::new(generated::generate(ports, id, weight, usage)?);
        published
            .borrow_mut()
            .push((id.to_owned(), weight, Arc::clone(&generated)));
        Ok(generated)
    })
    .unwrap();
    let built: BTreeSet<String> = inventory
        .offers
        .iter()
        .flat_map(|offer| offer.capability().unwrap().stages)
        .collect();
    let published = published.into_inner();
    assert!(
        published.iter().any(|(id, weight, _)| id == "full.v"
            && *weight == kernel_core::contract::WeightFormat::Nvfp4)
    );
    for (id, weight, generated) in &published {
        for stage in generated.offer.capability().unwrap().stages {
            assert!(
                built.contains(&stage),
                "{id} {weight:?}: {stage} is not built"
            );
        }
    }
}
