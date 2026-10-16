mod common;
use common::*;
use projection_spec::registry::Registry;
use projection_spec::*;

#[test]
fn shared_profiles_expand_without_repeating_bucket_records() {
    let mut d = declaration();
    d.use_profiles
        .insert("served".into(), d.projections[0].uses.clone());
    for p in &mut d.projections[0..2] {
        p.uses.clear();
        p.use_profile = Some(UseProfileRef::Id("served".into()));
    }
    d.validate(&geometry(), &Registry::default()).unwrap();
    let expanded = d.expanded_profiles(&geometry()).unwrap();
    assert_eq!(
        expanded.projections[0].uses,
        declaration().projections[0].uses
    );
    assert!(d.canonical_json().contains("\"use_profile\":\"served\""));
    assert!(d.projections[0].uses.is_empty());
    let b = d
        .bind(&geometry(), &Registry::default(), &metadata(), &Bf16Rules)
        .unwrap();
    assert_eq!(b.fusion_groups.len(), 8);
    d.projections[0].use_profile = Some(UseProfileRef::Id("missing".into()));
    assert!(
        d.validate(&geometry(), &Registry::default())
            .unwrap_err()
            .0
            .contains("unknown use profile")
    );
}

#[test]
fn named_operands_and_current_layer_assemblies_bind_one_template_per_stem() {
    use projection_spec::contracts::{RowRange, SourceMember};
    let mut d = declaration();
    d.projections[3].operands = vec!["gate".into(), "up".into()];
    let metadata = metadata();
    let mut assembly = d.projections[8].clone();
    assembly.id = "context".into();
    assembly.stem = Some("decoder".into());
    assembly.source = None;
    assembly.shape[0] = Dim::Literal(8);
    assembly.assembly = vec![SourceMember {
        coordinates: Default::default(),
        inherit_split: false,
        projection: "q".into(),
        layer: Some(LayerSelector::current()),
        operand: None,
        rows: RowRange {
            start: Dim::Literal(0),
            length: Dim::Literal(8),
        },
    }];
    d.projections.push(assembly);
    let bound = d
        .bind(&geometry(), &Registry::default(), &metadata, &Bf16Rules)
        .unwrap();
    assert_eq!(bound.assemblies.len(), 2);
    assert_eq!(bound.assemblies[0].layer, Some(0));
    assert_eq!(bound.assemblies[1].tensors[0], vec!["layers.1.q.weight"]);
    assert!(
        bound
            .operands
            .iter()
            .any(|b| b.imported.weight.codes.tensor == "layers.0.feed.gate.weight")
    );
    let offset = LayerSelector::Index(Dim::Expr(Box::new(DimExpr::Sum {
        terms: vec![Dim::Key("current".into()), Dim::Literal(1)],
    })));
    assert_eq!(offset.resolve(&geometry(), Some(0)).unwrap(), 1);
    assert!(
        offset
            .resolve(&geometry(), None)
            .unwrap_err()
            .0
            .contains("owning layer template")
    );
    d.projections[3].operands[1] = "gate".into();
    assert!(
        d.validate(&geometry(), &Registry::default())
            .unwrap_err()
            .0
            .contains("duplicate id")
    );
}

#[test]
fn grouped_norms_validate_reduction_and_affine_storage_without_a_provider() {
    use projection_spec::contracts::{NormAffine, NormContract};
    let mut d = declaration();
    d.side_tensors[0].dtypes.clear();
    d.side_tensors[0].norm = Some(NormContract {
        reduction_width: Dim::Literal(8),
        groups: Dim::Literal(2),
        affine: NormAffine::PerGroup,
        reduction: NormReduction::XorThenSerial,
        epsilon_bits: 1e-6f32.to_bits(),
        one_plus_weight: true,
        weight_dtypes: vec![DType::Bf16, DType::F32],
    });
    d.bind(&geometry(), &Registry::default(), &metadata(), &Bf16Rules)
        .unwrap();
    d.side_tensors[0].norm.as_mut().unwrap().affine = NormAffine::Shared;
    assert!(
        d.validate(&geometry(), &Registry::default())
            .unwrap_err()
            .0
            .contains("shared norm kernel")
    );
    d.side_tensors[0].norm.as_mut().unwrap().reduction_width = Dim::Literal(16);
    assert!(
        d.validate(&geometry(), &Registry::default())
            .unwrap_err()
            .to_string()
            .contains("shared norm kernel")
    );
    d.side_tensors[0].norm.as_mut().unwrap().groups = Dim::Literal(1);
    d.validate(&geometry(), &Registry::default()).unwrap();
    d.side_tensors[0].norm.as_mut().unwrap().epsilon_bits = f32::NAN.to_bits();
    assert!(
        d.validate(&geometry(), &Registry::default())
            .unwrap_err()
            .0
            .contains("epsilon")
    );
}

#[test]
fn heterogeneous_head_segments_own_heads_across_discontiguous_chunks() {
    use projection_spec::contracts::RowRange;
    let mut d = declaration();
    let range = |start, length| RowRange {
        start: Dim::Literal(start),
        length: Dim::Literal(length),
    };
    let segments = vec![
        HeadSegment {
            id: "q".into(),
            heads: Dim::Literal(2),
            head_dim: Dim::Literal(2),
            ranges: vec![range(0, 2), range(4, 2)],
        },
        HeadSegment {
            id: "v".into(),
            heads: Dim::Literal(4),
            head_dim: Dim::Literal(1),
            ranges: vec![range(2, 2), range(6, 2)],
        },
    ];
    d.projections[0].parts.clear();
    d.projections[0].split = SplitClass::SegmentedHeads {
        segments,
        order: HeadGatherOrder::Segments(vec!["q".into(), "v".into()]),
        tile: Dim::Literal(2),
    };
    d.validate(&geometry(), &Registry::default()).unwrap();
    if let SplitClass::SegmentedHeads { segments, .. } = &mut d.projections[0].split {
        segments[1].ranges[0].start = Dim::Literal(1);
    }
    assert!(
        d.validate(&geometry(), &Registry::default())
            .unwrap_err()
            .0
            .contains("overlap or gaps")
    );
}

#[test]
fn arbitrary_rank_side_placement_gathers_before_consumer_conversion() {
    use projection_spec::side::SidePlacement;
    let mut side = declaration().side_tensors[3].clone();
    side.shape = vec![Dim::Literal(5), Dim::Literal(1), Dim::Literal(3)];
    side.placement = SidePlacement {
        axis: 0,
        split: SplitClass::Column {
            tile: Dim::Literal(2),
        },
    };
    let selected = side
        .selection(
            &geometry(),
            projection_spec::derive::Topology { ranks: 2, rank: 1 },
        )
        .unwrap();
    assert_eq!(selected.shape, vec![4, 1, 3]);
    let source = (0..15u16).flat_map(u16::to_le_bytes).collect::<Vec<_>>();
    let mut expected = source[18..30].to_vec();
    expected.resize(24, 0);
    assert_eq!(selected.repack(&source, DType::Bf16).unwrap(), expected);
    side.placement.axis = 3;
    assert!(
        side.selection(
            &geometry(),
            projection_spec::derive::Topology { ranks: 2, rank: 0 }
        )
        .unwrap_err()
        .0
        .contains("axis")
    );
}

#[test]
fn assemblies_declare_coordinate_domain_and_source_split_inheritance() {
    use projection_spec::contracts::{RowCoordinates, RowRange, SourceMember};
    let mut d = declaration();
    let mut p = d.projections[8].clone();
    p.id = "inherited".into();
    p.source = None;
    p.shape[0] = Dim::Literal(8);
    p.assembly = vec![SourceMember {
        coordinates: RowCoordinates::RankLocal,
        inherit_split: true,
        projection: "q".into(),
        layer: Some(LayerSelector::Index(Dim::Literal(0))),
        operand: None,
        rows: RowRange {
            start: Dim::Literal(0),
            length: Dim::Literal(8),
        },
    }];
    d.projections.push(p);
    let bound = d
        .bind(&geometry(), &Registry::default(), &metadata(), &Bf16Rules)
        .unwrap();
    assert_eq!(
        bound.assemblies[0].members[0].coordinates,
        RowCoordinates::RankLocal
    );
    assert!(bound.assemblies[0].members[0].inherit_split);
    d.projections.last_mut().unwrap().assembly[0].rows.length = Dim::Literal(9);
    assert!(
        d.validate(&geometry(), &Registry::default())
            .unwrap_err()
            .0
            .contains("out of bounds")
    );
}

#[test]
fn fusion_segments_overlap_only_within_their_named_output_buffer() {
    use projection_spec::contracts::OutputSegment;
    let mut d = declaration();
    d.fusion_groups[0].outputs = vec![
        OutputSegment {
            buffer: "queries".into(),
            member: "q".into(),
            offset: Dim::Literal(0),
            row_stride: Dim::Literal(8),
            live_columns: Dim::Literal(8),
        },
        OutputSegment {
            buffer: "keys".into(),
            member: "k".into(),
            offset: Dim::Literal(0),
            row_stride: Dim::Literal(8),
            live_columns: Dim::Literal(8),
        },
    ];
    d.validate(&geometry(), &Registry::default()).unwrap();
    d.fusion_groups[0].outputs[1].buffer = "queries".into();
    assert!(
        d.validate(&geometry(), &Registry::default())
            .unwrap_err()
            .0
            .contains("overlap")
    );
}

#[test]
fn integer_ports_preserve_unsigned_bits_without_changing_dtype() {
    use projection_spec::contracts::{Interpretation, Port};
    let mut port = Port {
        id: "state".into(),
        binding: "state".into(),
        dtype: DType::I64,
        interpretation: Some(Interpretation::Unsigned),
        shape: vec![Dim::Literal(4)],
        state: true,
        format: None,
    };
    port.validate(&geometry()).unwrap();
    let wire = serde_json::to_string(&port).unwrap();
    assert!(wire.contains("\"dtype\":\"I64\""));
    assert!(wire.contains("\"interpretation\":\"unsigned\""));
    assert_eq!(serde_json::from_str::<Port>(&wire).unwrap(), port);
    port.dtype = DType::F32;
    assert!(
        port.validate(&geometry())
            .unwrap_err()
            .0
            .contains("I32 or I64")
    );
}

#[test]
fn optional_external_owners_are_absent_or_complete_and_attested() {
    use projection_spec::contracts::ExternalTensor;
    let mut d = declaration();
    for id in ["a", "b"] {
        d.external_tensors.push(ExternalTensor {
            optional: true,
            id: format!("external-{id}"),
            owner: "encoder".into(),
            image_role: "encoder-image".into(),
            source: TensorSource {
                stem: None,
                tensor: format!("encoder.{id}"),
            },
            shape: vec![Dim::Literal(2)],
            dtype: DType::Bf16,
        });
    }
    let mut m = metadata();
    d.bind(&geometry(), &Registry::default(), &m, &Bf16Rules)
        .unwrap();
    m.tensors.insert("encoder.a".into(), info(&[2], "BF16"));
    m.validated_owners.insert(
        "encoder".into(),
        std::collections::BTreeSet::from(["encoder.a".into()]),
    );
    assert!(
        d.bind(&geometry(), &Registry::default(), &m, &Bf16Rules)
            .unwrap_err()
            .0
            .contains("partial external owner")
    );
    m.tensors.insert("encoder.b".into(), info(&[2], "BF16"));
    assert!(
        d.bind(&geometry(), &Registry::default(), &m, &Bf16Rules)
            .unwrap_err()
            .0
            .contains("was not validated")
    );
    m.validated_owners
        .get_mut("encoder")
        .unwrap()
        .insert("encoder.b".into());
    assert_eq!(
        d.bind(&geometry(), &Registry::default(), &m, &Bf16Rules)
            .unwrap()
            .external_tensors
            .len(),
        2
    );
}

#[test]
fn checkpoint_wide_support_is_resolved_before_execution_admission() {
    use kernel_core::precision::{ExecutionPhase, format::FormatId};
    use projection_spec::contracts::SupportRule;
    let mut d = declaration();
    d.layer_stems[1].support = vec![SupportRule {
        when: Some(CheckpointCondition {
            any_format_in: vec!["decoder".into()],
            formats: vec![FormatId::Bf16],
        }),
        ranks: vec![],
        formats: vec![],
        phases: vec![],
        enabled: false,
    }];
    let bound = d
        .bind(&geometry(), &Registry::default(), &metadata(), &Bf16Rules)
        .unwrap();
    // Conditions bound from projection formats alone equal the full binding.
    let operands = bound
        .operands
        .iter()
        .map(|o| (o.projection.as_str(), o.format));
    assert_eq!(d.checkpoint_conditions(operands), bound.conditions);
    // A stem scope matches its projections only in a listed format, and a
    // projection outside the stem does not match it.
    let matches = |d: &Declaration, projection, format| {
        d.checkpoint_conditions([(projection, format)])[0].matches
    };
    assert!(matches(&d, "q", FormatId::Bf16));
    assert!(!matches(&d, "q", FormatId::Mxfp8));
    assert!(!matches(&d, "table", FormatId::Bf16));
    assert!(!matches(&d, "draft", FormatId::Bf16));
    d.layer_stems[1].support[0]
        .when
        .as_mut()
        .unwrap()
        .any_format_in = vec!["table".into()];
    assert!(matches(&d, "table", FormatId::Bf16));
    assert!(!matches(&d, "q", FormatId::Bf16));
    d.layer_stems[1].support[0]
        .when
        .as_mut()
        .unwrap()
        .any_format_in = vec!["decoder".into()];
    let request = Admission {
        projection: "draft",
        layer: Some(0),
        ranks: 2,
        phase: ExecutionPhase::Decode,
        format: FormatId::Bf16,
    };
    assert!(
        d.admit_checkpoint(&geometry(), &bound, request)
            .unwrap_err()
            .0
            .contains("unsupported")
    );
    assert!(
        d.admit(
            &geometry(),
            "draft",
            Some(0),
            2,
            ExecutionPhase::Decode,
            FormatId::Bf16
        )
        .unwrap_err()
        .0
        .contains("checkpoint binding")
    );
    d.layer_stems[1].support[0].when.as_mut().unwrap().formats = vec![FormatId::Fp8Block128];
    let bound = d
        .bind(&geometry(), &Registry::default(), &metadata(), &Bf16Rules)
        .unwrap();
    d.admit_checkpoint(&geometry(), &bound, request).unwrap();
    d.layer_stems[1].support[0]
        .when
        .as_mut()
        .unwrap()
        .any_format_in = vec!["unknown".into()];
    assert!(
        d.validate(&geometry(), &Registry::default())
            .unwrap_err()
            .0
            .contains("unknown scope")
    );
}

#[test]
fn aliases_cross_stems_and_activation_segments_follow_rank_k() {
    use projection_spec::contracts::{ExecutionLayers, LayerAlias};
    use projection_spec::inputs::{ActivationInput, ActivationSegment, ActivationSelection};
    let mut d = declaration();
    d.layer_stems[1].execution = Some(ExecutionLayers {
        indices: vec![Dim::Literal(7)],
        aliases: vec![LayerAlias {
            source_stem: Some("decoder".into()),
            execution_index: Dim::Literal(7),
            source_index: Dim::Literal(1),
        }],
    });
    d.projections[2].inputs = vec![ActivationInput {
        operand: "weight".into(),
        binding: "activation".into(),
        segments: vec![ActivationSegment {
            origin: None,
            length: None,
            source: "hidden".into(),
            selection: ActivationSelection::RankK {
                projection: "out".into(),
                operand: "weight".into(),
            },
        }],
    }];
    let bound = d
        .bind(&geometry(), &Registry::default(), &metadata(), &Bf16Rules)
        .unwrap();
    assert_eq!(bound.layer_aliases[0].source_stem, "decoder");
    assert_eq!(bound.layer_aliases[0].source_index, 1);
    let range = d.projections[2].inputs[0].segments[0]
        .range(
            &d,
            &geometry(),
            projection_spec::derive::Topology { ranks: 2, rank: 1 },
        )
        .unwrap();
    assert_eq!(range, 4..8);
    d.projections[2].operand_splits = vec![projection_spec::contracts::OperandSplit {
        operand: "weight".into(),
        axis: SliceAxis::K,
        tile: Dim::Literal(8),
    }];
    assert_eq!(
        d.projections[2].inputs[0].segments[0]
            .range(
                &d,
                &geometry(),
                projection_spec::derive::Topology { ranks: 2, rank: 0 }
            )
            .unwrap(),
        0..4
    );
    assert_eq!(
        d.projections[2].inputs[0].segments[0]
            .range(
                &d,
                &geometry(),
                projection_spec::derive::Topology { ranks: 2, rank: 1 }
            )
            .unwrap(),
        4..8
    );
    d.layer_stems[1].execution.as_mut().unwrap().aliases[0].source_stem = Some("missing".into());
    assert!(
        d.validate(&geometry(), &Registry::default())
            .unwrap_err()
            .0
            .contains("unknown source stem")
    );
}

#[test]
fn numerical_recipes_select_the_bound_format_without_selecting_an_implementation() {
    use kernel_core::precision::format::{A16Recipe, FormatId};
    let mut d = declaration();
    let composition = serde_json::from_value(serde_json::json!({
        "pre": [], "post": [],
        "numerical_contracts": {
            "bf16": {"numerical_contract": "test-bf16-v2", "a16_recipe": "scaled_weight"},
            "fp8_block128": {"numerical_contract": "test-block-v2", "a16_recipe": "scale_after_block_mma"}
        }
    })).unwrap();
    d.projections[0].composition = Some(composition);
    let bound = d
        .bind(&geometry(), &Registry::default(), &metadata(), &Bf16Rules)
        .unwrap();
    let selected = d.projections[0]
        .composition
        .as_ref()
        .unwrap()
        .numerical_for_sources([bound.operands[0].format])
        .unwrap()
        .unwrap();
    assert_eq!(selected.numerical_contract, "test-bf16-v2");
    assert_eq!(selected.a16_recipe, A16Recipe::ScaledWeight);
    assert!(
        d.projections[0]
            .composition
            .as_ref()
            .unwrap()
            .implementation
            .is_none()
    );
    assert_eq!(
        d.projections[0]
            .composition
            .as_ref()
            .unwrap()
            .numerical_contracts[&FormatId::Fp8Block128]
            .a16_recipe,
        A16Recipe::ScaleAfterBlockMma
    );
    d.projections[0]
        .composition
        .as_mut()
        .unwrap()
        .numerical_contracts
        .get_mut(&FormatId::Bf16)
        .unwrap()
        .numerical_contract
        .clear();
    assert!(
        d.validate(&geometry(), &Registry::default())
            .unwrap_err()
            .0
            .contains("numerical contract")
    );
}

#[test]
fn norm_contracts_refuse_arithmetic_without_a_shared_lowering() {
    use projection_spec::contracts::NormAffine;
    for change in 0..3 {
        let mut d = declaration();
        let side = d
            .side_tensors
            .iter_mut()
            .find(|s| s.norm.is_some())
            .unwrap();
        let norm = side.norm.as_mut().unwrap();
        match change {
            0 => norm.epsilon_bits = 1e-5f32.to_bits(),
            1 => norm.one_plus_weight = false,
            _ => {
                norm.groups = Dim::Literal(4);
                norm.affine = NormAffine::Shared;
            }
        }
        assert!(
            d.validate(&geometry(), &Registry::default())
                .unwrap_err()
                .to_string()
                .contains("shared norm kernel")
        );
    }
}
