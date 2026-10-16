mod common;

use common::*;
use kernel_core::contract::{ActivationFormat, DType};
use kernel_core::precision::ExecutionPhase;
use kernel_core::precision::format::FormatId;
use projection_spec::bind::CheckpointMetadata;
use projection_spec::contracts::*;
use projection_spec::registry::{self, Provider, Registration, Registry, Target};
use projection_spec::{
    Declaration, Dim, Error, Geometry, SideKind, SideTensor, SliceAxis, TensorSource, Variant,
    parse,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    width: u32,
}

fn transform(v: &Variant, g: &Geometry) -> Result<Value, Error> {
    if v.inputs[0].format != Some(FormatId::Fp8Block128)
        || v.outputs[0].format != Some(FormatId::Fp8Rows64)
    {
        return Err(Error("regroup source/effective formats disagree".into()));
    }
    registry::parameters::<Parameters>(v, g)
}

static PROVIDER: Provider = Provider {
    namespace: "fixture",
    registrations: &[
        Registration {
            id: "fixture/regroup.v1",
            numerical_contract: "fixture.copy.v1",
            target: Target::SourceTransform,
            decoder: transform,
        },
        Registration {
            id: "fixture/hash.v1",
            numerical_contract: "fixture.hash.v1",
            target: Target::Stage,
            decoder: registry::parameters::<Parameters>,
        },
        Registration {
            id: "fixture/legacy.v1",
            numerical_contract: "fixture.legacy.v1",
            target: Target::Projection,
            decoder: registry::parameters::<Parameters>,
        },
    ],
};

fn registry() -> Registry {
    let mut r = Registry::default();
    r.register(&PROVIDER).unwrap();
    r
}
fn port(id: &str, state: bool) -> Port {
    Port {
        interpretation: None,
        id: id.into(),
        binding: id.into(),
        dtype: DType::I64,
        shape: vec![Dim::Literal(2)],
        state,
        format: None,
    }
}
fn variant(id: &str, numerical: &str) -> Variant {
    Variant {
        id: id.into(),
        parameters: json!({"width":2}),
        numerical_contract: numerical.into(),
        inputs: vec![port("history", true)],
        outputs: vec![port("history", true)],
    }
}
fn row(start: u64, length: u64) -> RowRange {
    RowRange {
        start: Dim::Literal(start),
        length: Dim::Literal(length),
    }
}
fn verify(d: &Declaration) -> Result<(), Error> {
    d.validate(&geometry(), &registry())
}
fn contract(output: DType) -> UseContract {
    UseContract {
        compiled_capacity: Dim::Literal(256),
        row_multiplier: Dim::Literal(1),
        chunk_rows: None,
        per_item_verify: None,
        aggregate_verify: None,
        output_dtype: output,
        numerical_contract: "fixture.use.v1".into(),
        activation: Some(ActivationFormat::Bf16),
        exact_rows: false,
        support: vec![],
        implementation: None,
    }
}
fn bind(
    d: &Declaration,
    m: &CheckpointMetadata,
) -> Result<projection_spec::bind::BoundDeclaration, Error> {
    d.bind(&geometry(), &registry(), m, &Bf16Rules)
}

#[test]
fn format_ids_round_trip_with_no_string_fallback() {
    for id in [
        FormatId::Bf16,
        FormatId::Fp8PerTensor,
        FormatId::Fp8PerChannel,
        FormatId::Fp8Block128,
        FormatId::Fp8Rows64,
        FormatId::Mxfp8,
        FormatId::Mxfp4,
        FormatId::Nvfp4,
        FormatId::Nvfp4WeightOnly,
    ] {
        let text = serde_json::to_string(&id).unwrap();
        assert_eq!(serde_json::from_str::<FormatId>(&text).unwrap(), id);
    }
    let mut d = declaration();
    d.projections[0].format = Some(FormatId::Nvfp4);
    assert!(
        bind(&d, &metadata())
            .unwrap_err()
            .0
            .contains("plug-in disagrees")
    );
}

#[test]
fn source_transform_is_typed_and_provider_checks_effective_format() {
    let mut d = declaration();
    let mut v = variant("fixture/regroup.v1", "fixture.copy.v1");
    v.inputs[0].format = Some(FormatId::Fp8Block128);
    v.outputs[0].format = Some(FormatId::Fp8Rows64);
    d.projections[0].parts[0].transforms.push(v);
    d.projections[0].parts[0].scale_origin = [128, 0];
    verify(&d).unwrap();
    parse(
        &serde_json::to_string(&d).unwrap(),
        &geometry(),
        &registry(),
    )
    .unwrap();
    let mut bad = d.clone();
    bad.projections[0].parts[0].transforms[0].outputs[0].format = Some(FormatId::Bf16);
    assert!(verify(&bad).unwrap_err().0.contains("effective formats"));
    let mut bad = d.clone();
    bad.projections[0].parts[0].transforms[0].numerical_contract = "fixture.changed.v1".into();
    assert!(verify(&bad).unwrap_err().0.contains("numerical contract"));
    let mut bad = d;
    bad.projections[0].parts[0].transforms[0].parameters["unknown"] = json!(true);
    assert!(verify(&bad).unwrap_err().0.contains("unknown field"));
}

#[test]
fn ordered_stateful_pre_stage_and_explicit_legacy_implementation_validate() {
    let mut d = declaration();
    d.projections[7].composition = Some(Composition {
        implementations: Default::default(),
        numerical_contracts: Default::default(),
        pre: vec![Stage {
            id: "hash_pre".into(),
            contract: variant("fixture/hash.v1", "fixture.hash.v1"),
            phases: vec![
                ExecutionPhase::Prefill,
                ExecutionPhase::Decode,
                ExecutionPhase::Verify,
            ],
            mutates: vec!["history".into()],
        }],
        implementation: Some(variant("fixture/legacy.v1", "fixture.legacy.v1")),
        post: vec![],
    });
    verify(&d).unwrap();
    let mut bad = d.clone();
    bad.projections[7].composition.as_mut().unwrap().pre[0].mutates = vec!["tokens".into()];
    assert!(
        verify(&bad)
            .unwrap_err()
            .0
            .contains("undeclared input state")
    );
    let mut bad = d.clone();
    bad.projections[7].composition.as_mut().unwrap().pre[0]
        .contract
        .outputs
        .clear();
    assert!(verify(&bad).unwrap_err().0.contains("typed inputs/outputs"));
    let mut bad = d;
    bad.projections[7]
        .composition
        .as_mut()
        .unwrap()
        .implementation
        .as_mut()
        .unwrap()
        .numerical_contract = "fixture.other.v1".into();
    assert!(verify(&bad).unwrap_err().0.contains("numerical contract"));
}

#[test]
fn parts_refuse_overlapping_ranges_and_uncovered_head_runs() {
    let mut d = declaration();
    d.projections[0].parts[1].range = Some(row(3, 4));
    assert!(verify(&d).unwrap_err().0.contains("ordered logical rows"));
    let mut d = declaration();
    d.projections[0].parts[0].head_runs = vec![HeadRun {
        id: "query".into(),
        rows: row(0, 4),
        heads: Dim::Literal(1),
        head_dim: Dim::Literal(4),
    }];
    verify(&d).unwrap();
    d.projections[0].parts[0].head_runs[0].rows.length = Dim::Literal(3);
    assert!(verify(&d).unwrap_err().0.contains("whole heads"));
}

#[test]
fn fusion_predicate_applies_per_use_with_separate_or_refuse_fallback() {
    let mut d = declaration();
    for p in &mut d.projections[0..2] {
        for use_ in &mut p.uses {
            let mut c = contract(DType::Bf16);
            if use_.phase == ExecutionPhase::Verify {
                c.per_item_verify = Some(use_.capacity.clone());
                c.aggregate_verify = Some(use_.capacity.clone());
            }
            use_.contract = Some(c);
        }
    }
    d.fusion_groups[0].eligibility = Eligibility::All {
        formats: vec![FormatId::Bf16],
        activation: Some(ActivationFormat::Bf16),
        phases: vec![ExecutionPhase::Decode],
    };
    let bound = bind(&d, &metadata()).unwrap();
    assert!(
        bound
            .fusion_groups
            .iter()
            .all(|g| g.fused == (g.phase == ExecutionPhase::Decode))
    );
    d.fusion_groups[0].fallback = Fallback::Refuse;
    assert!(
        bind(&d, &metadata())
            .unwrap_err()
            .0
            .contains("fallback refuses")
    );
    d.fusion_groups[0].eligibility = Eligibility::All {
        formats: vec![FormatId::Mxfp8],
        activation: None,
        phases: vec![],
    };
    assert!(
        bind(&d, &metadata())
            .unwrap_err()
            .0
            .contains("eligibility failed")
    );
}

#[test]
fn output_segments_refuse_overlap_stride_overflow_and_unknown_members() {
    let mut d = declaration();
    d.fusion_groups[0].outputs = vec![
        OutputSegment {
            buffer: "shared".into(),
            member: "q".into(),
            offset: Dim::Literal(0),
            row_stride: Dim::Literal(16),
            live_columns: Dim::Literal(8),
        },
        OutputSegment {
            buffer: "shared".into(),
            member: "k".into(),
            offset: Dim::Literal(8),
            row_stride: Dim::Literal(16),
            live_columns: Dim::Literal(8),
        },
    ];
    verify(&d).unwrap();
    let mut bad = d.clone();
    bad.fusion_groups[0].outputs[1].offset = Dim::Literal(7);
    assert!(verify(&bad).unwrap_err().0.contains("overlap"));
    let mut bad = d.clone();
    bad.fusion_groups[0].outputs[1].row_stride = Dim::Literal(15);
    assert!(verify(&bad).unwrap_err().0.contains("row stride"));
    let mut bad = d;
    bad.fusion_groups[0].outputs[1].member = "missing".into();
    assert!(verify(&bad).unwrap_err().0.contains("nonmember"));
}

#[test]
fn checkpoint_lookup_shards_and_prime_sized_heads_are_independent_of_tp() {
    let mut d = declaration();
    let p = &mut d.projections[6];
    p.kind = projection_spec::ProjectionKind::LookupTable {};
    p.source = None;
    p.lookup = Some(LookupAssembly {
        companions: Default::default(),
        shards: projection_spec::LookupShards::Explicit(vec![
            LookupShard {
                id: "first".into(),
                source: TensorSource {
                    stem: None,
                    tensor: "table.0".into(),
                },
                rows: row(0, 24),
            },
            LookupShard {
                id: "last".into(),
                source: TensorSource {
                    stem: None,
                    tensor: "table.1".into(),
                },
                rows: row(24, 8),
            },
        ]),
        heads: vec![
            LookupHead {
                id: "h0".into(),
                vocabulary: row(0, 17),
            },
            LookupHead {
                id: "h1".into(),
                vocabulary: row(17, 13),
            },
        ],
        padding_owner: PaddingOwner::LastRank {},
    });
    let mut m = metadata();
    m.tensors.remove("embed.weight");
    m.tensors.insert("table.0".into(), info(&[24, 16], "BF16"));
    m.tensors.insert("table.1".into(), info(&[8, 16], "BF16"));
    let bound = bind(&d, &m).unwrap();
    assert_eq!(
        bound
            .operands
            .iter()
            .filter(|b| b.projection == "table")
            .map(|b| b.shard)
            .collect::<Vec<_>>(),
        [Some(0), Some(1)]
    );
    let mut bad = d.clone();
    bad.projections[6].lookup.as_mut().unwrap().heads[1]
        .vocabulary
        .start = Dim::Literal(16);
    assert!(verify(&bad).unwrap_err().0.contains("head vocabularies"));
    let mut bad = d;
    bad.projections[6]
        .lookup
        .as_mut()
        .unwrap()
        .shards
        .explicit_mut()
        .unwrap()[1]
        .rows
        .start = Dim::Literal(23);
    assert!(verify(&bad).unwrap_err().0.contains("contiguous rows"));
}

#[test]
fn one_source_has_multiple_projection_and_side_consumers_and_derived_assemblies() {
    let mut d = declaration();
    let mut shared = d.projections[8].clone();
    shared.id = "shared_draft".into();
    shared.assembly = vec![SourceMember {
        coordinates: Default::default(),
        inherit_split: false,
        projection: "draft".into(),
        layer: Some(projection_spec::LayerSelector::Index(Dim::Literal(0))),
        operand: None,
        rows: row(0, 16),
    }];
    d.projections.push(shared);
    let mut derived = d.projections[8].clone();
    derived.id = "context".into();
    derived.shape[0] = Dim::Literal(16);
    derived.source.as_mut().unwrap().stem = None;
    derived.source.as_mut().unwrap().tensor = "derived".into();
    derived.assembly = vec![
        SourceMember {
            coordinates: Default::default(),
            inherit_split: false,
            projection: "q".into(),
            layer: Some(projection_spec::LayerSelector::Index(Dim::Literal(0))),
            operand: None,
            rows: row(0, 8),
        },
        SourceMember {
            coordinates: Default::default(),
            inherit_split: false,
            projection: "k".into(),
            layer: Some(projection_spec::LayerSelector::Index(Dim::Literal(1))),
            operand: None,
            rows: row(0, 8),
        },
    ];
    d.projections.push(derived);
    let mut side = d.side_tensors[0].clone();
    side.id = "also_draft".into();
    side.source = d.projections[8].source.clone().unwrap();
    side.shape = vec![Dim::Literal(16), Dim::Literal(16)];
    side.kind = SideKind::Gate;
    side.norm = None;
    side.consumer = "draft".into();
    d.side_tensors.push(side);
    let bound = bind(&d, &metadata()).unwrap();
    assert_eq!(
        bound.assemblies[1].tensors,
        [
            vec!["layers.0.q.weight".to_string()],
            vec!["layers.1.k.weight".to_string()]
        ]
    );
    assert_eq!(
        bound
            .operands
            .iter()
            .filter(|o| o.imported.weight.codes.tensor == "draft.0.proj.weight")
            .count(),
        1
    );
    let mut bad = d;
    bad.projections.last_mut().unwrap().assembly[0].rows.length = Dim::Literal(9);
    assert!(verify(&bad).unwrap_err().0.contains("out of bounds"));
}

#[test]
fn norm_views_widen_exactly_and_refuse_narrowing() {
    let mut d = declaration();
    let side = &mut d.side_tensors[0];
    side.views = vec![ConsumerView {
        consumer: "q".into(),
        dtype: DType::F32,
        conversion: ViewConversion::WidenBf16F32,
    }];
    side.norm = Some(NormContract {
        reduction_width: Dim::Literal(16),
        groups: Dim::Literal(1),
        affine: NormAffine::Shared,
        reduction: projection_spec::NormReduction::ShuffleDown,
        epsilon_bits: 1e-6f32.to_bits(),
        one_plus_weight: true,
        weight_dtypes: vec![DType::Bf16, DType::F32],
    });
    let mut m = metadata();
    m.tensors.get_mut("layers.0.norm.weight").unwrap().dtype = "F32".into();
    let bound = bind(&d, &m).unwrap();
    assert_eq!(
        bound.side_views.iter().filter(|v| v.side == "norm").count(),
        2
    );
    d.side_tensors[0].views[0].dtype = DType::Bf16;
    assert!(verify(&d).unwrap_err().0.contains("narrow or change dtype"));
}

#[test]
fn integer_and_arbitrary_rank_side_state_binds_to_registered_stage() {
    let mut d = declaration();
    d.projections[7].composition = Some(Composition {
        implementations: Default::default(),
        numerical_contracts: Default::default(),
        pre: vec![Stage {
            id: "hash_pre".into(),
            contract: variant("fixture/hash.v1", "fixture.hash.v1"),
            phases: vec![ExecutionPhase::Decode],
            mutates: vec!["history".into()],
        }],
        implementation: None,
        post: vec![],
    });
    d.side_tensors.push(SideTensor {
        placement: Default::default(),
        id: "hash_state".into(),
        source: TensorSource {
            stem: None,
            tensor: "history".into(),
        },
        shape: vec![
            Dim::Literal(2),
            Dim::Literal(2),
            Dim::Literal(1),
            Dim::Literal(1),
            Dim::Literal(1),
            Dim::Literal(1),
        ],
        kind: SideKind::State,
        dtypes: vec![DType::I64],
        consumer: "hash_pre".into(),
        views: vec![],
        norm: None,
    });
    let mut m = metadata();
    m.tensors
        .insert("history".into(), info(&[2, 2, 1, 1, 1, 1], "I64"));
    bind(&d, &m).unwrap();
    m.tensors.get_mut("history").unwrap().dtype = "F32".into();
    assert!(bind(&d, &m).unwrap_err().0.contains("history"));
}

#[test]
fn requested_uses_distinguish_compiled_rows_verify_limits_and_output_contracts() {
    let mut d = declaration();
    let p = &mut d.projections[8];
    let mut bf16 = p.uses[1].clone();
    bf16.contract = Some(contract(DType::Bf16));
    let mut f32 = bf16.clone();
    f32.contract.as_mut().unwrap().output_dtype = DType::F32;
    p.uses.push(bf16);
    p.uses.push(f32);
    let mut verify_use = p.uses[2].clone();
    let mut c = contract(DType::Bf16);
    c.compiled_capacity = Dim::Literal(8);
    c.row_multiplier = Dim::Literal(4);
    c.chunk_rows = Some(Dim::Literal(8));
    c.per_item_verify = Some(Dim::Literal(4));
    c.aggregate_verify = Some(Dim::Literal(256));
    c.exact_rows = true;
    verify_use.contract = Some(c);
    p.uses.push(verify_use);
    verify(&d).unwrap();
    let mut bad = d.clone();
    bad.projections[8]
        .uses
        .last_mut()
        .unwrap()
        .contract
        .as_mut()
        .unwrap()
        .chunk_rows = None;
    assert!(verify(&bad).unwrap_err().0.contains("explicit chunking"));
    let mut bad = d;
    bad.projections[8]
        .uses
        .last_mut()
        .unwrap()
        .contract
        .as_mut()
        .unwrap()
        .per_item_verify = Some(Dim::Literal(512));
    assert!(verify(&bad).unwrap_err().0.contains("aggregate limit"));
}

#[test]
fn materialization_aliases_and_format_topology_refusals_are_declarations() {
    let mut d = declaration();
    d.layer_stems[1].execution = Some(ExecutionLayers {
        indices: vec![Dim::Literal(5)],
        aliases: vec![LayerAlias {
            source_stem: None,
            execution_index: Dim::Literal(5),
            source_index: Dim::Literal(0),
        }],
    });
    d.layer_stems[1].support = vec![
        SupportRule {
            when: None,
            ranks: vec![2, 4],
            formats: vec![],
            phases: vec![],
            enabled: true,
        },
        SupportRule {
            when: None,
            ranks: vec![],
            formats: vec![FormatId::Nvfp4],
            phases: vec![],
            enabled: false,
        },
    ];
    verify(&d).unwrap();
    bind(&d, &metadata()).unwrap();
    d.admit(
        &geometry(),
        "draft",
        Some(5),
        2,
        ExecutionPhase::Decode,
        FormatId::Bf16,
    )
    .unwrap();
    assert!(
        d.admit(
            &geometry(),
            "draft",
            Some(0),
            2,
            ExecutionPhase::Decode,
            FormatId::Bf16
        )
        .is_err()
    );
    assert!(
        d.admit(
            &geometry(),
            "draft",
            Some(5),
            1,
            ExecutionPhase::Decode,
            FormatId::Bf16
        )
        .is_err()
    );
    assert!(
        d.admit(
            &geometry(),
            "draft",
            Some(5),
            4,
            ExecutionPhase::Decode,
            FormatId::Nvfp4
        )
        .is_err()
    );
    d.layer_stems[1].execution.as_mut().unwrap().indices.clear();
    d.layer_stems[1].execution.as_mut().unwrap().aliases.clear();
    d.projections[8].uses.clear();
    verify(&d).unwrap();
    bind(&d, &metadata()).unwrap();
}

#[test]
fn closed_coverage_includes_validated_external_owners() {
    let mut d = declaration();
    d.external_tensors.push(ExternalTensor {
        optional: false,
        id: "encoder_weight".into(),
        owner: "encoder_spec".into(),
        image_role: "vision".into(),
        source: TensorSource {
            stem: None,
            tensor: "encoder.norm".into(),
        },
        shape: vec![Dim::Literal(4)],
        dtype: DType::F32,
    });
    let mut m = metadata();
    m.tensors.insert("encoder.norm".into(), info(&[4], "F32"));
    assert!(
        bind(&d, &m)
            .unwrap_err()
            .0
            .contains("not validated by owner")
    );
    m.validated_owners.insert(
        "encoder_spec".into(),
        std::collections::BTreeSet::from(["encoder.norm".into()]),
    );
    bind(&d, &m).unwrap();
    m.tensors.insert("unowned".into(), info(&[1], "BF16"));
    assert!(
        bind(&d, &m)
            .unwrap_err()
            .0
            .contains("unknown checkpoint tensor `unowned`")
    );
}

#[test]
fn per_operand_splits_refuse_wrong_axis_and_missing_operand() {
    let mut d = declaration();
    d.projections[3].operand_splits = vec![OperandSplit {
        operand: "gate".into(),
        axis: SliceAxis::Rows,
        tile: Dim::Literal(16),
    }];
    verify(&d).unwrap();
    d.projections[3].operand_splits[0].axis = SliceAxis::K;
    assert!(verify(&d).unwrap_err().0.contains("axis disagrees"));
    d.projections[3].operand_splits[0].operand = "missing".into();
    assert!(verify(&d).unwrap_err().0.contains("out of range"));
}

#[test]
fn zero_geometry_offsets_and_strict_contract_fields_are_supported() {
    let mut geometry = geometry();
    geometry.insert("zero".into(), 0);
    assert_eq!(
        Dim::Key("zero".into()).resolve_offset(&geometry).unwrap(),
        0
    );
    assert!(Dim::Key("zero".into()).resolve(&geometry).is_err());
    let expression: Dim =
        serde_json::from_value(json!({"op":"quotient","numerator":0,"denominator":0})).unwrap();
    assert!(
        expression
            .resolve_offset(&geometry)
            .unwrap_err()
            .0
            .contains("divisor")
    );
    assert!(
        serde_json::from_value::<RowRange>(json!({"start":0,"length":1,"unknown":1}))
            .unwrap_err()
            .to_string()
            .contains("unknown field")
    );
    assert!(
        serde_json::from_value::<PaddingOwner>(json!({"owner":"last_rank","rank":1}))
            .unwrap_err()
            .to_string()
            .contains("unknown field")
    );
}

#[test]
fn derived_members_produce_fusion_decisions_and_mixed_output_types_refuse() {
    let mut d = declaration();
    for id in ["derived_q", "derived_k"] {
        let mut p = d.projections[0].clone();
        p.id = id.into();
        p.source.as_mut().unwrap().stem = None;
        p.source.as_mut().unwrap().tensor = "derived".into();
        p.assembly = vec![SourceMember {
            coordinates: Default::default(),
            inherit_split: false,
            projection: if id == "derived_q" { "q" } else { "k" }.into(),
            layer: Some(projection_spec::LayerSelector::Index(Dim::Literal(0))),
            operand: None,
            rows: row(0, 8),
        }];
        d.projections.push(p);
    }
    let mut group = d.fusion_groups[0].clone();
    group.id = "derived_group".into();
    group.members = vec!["derived_q".into(), "derived_k".into()];
    d.fusion_groups.push(group);
    let bound = bind(&d, &metadata()).unwrap();
    assert_eq!(
        bound
            .fusion_groups
            .iter()
            .filter(|g| g.id == "derived_group")
            .count(),
        4
    );
    d.projections[1].uses[0].contract = Some(contract(DType::F32));
    assert!(verify(&d).unwrap_err().0.contains("input contract"));
}

#[test]
fn gate_up_assembly_preserves_distinct_source_operand_selections() {
    let mut d = declaration();
    let mut combined = d.projections[3].clone();
    combined.id = "assembled_gate_up".into();
    combined.source.as_mut().unwrap().stem = None;
    combined.source.as_mut().unwrap().tensor = "derived".into();
    combined.assembly = vec![
        SourceMember {
            coordinates: Default::default(),
            inherit_split: false,
            projection: "feed".into(),
            layer: Some(projection_spec::LayerSelector::Index(Dim::Literal(0))),
            operand: Some("gate".into()),
            rows: row(0, 12),
        },
        SourceMember {
            coordinates: Default::default(),
            inherit_split: false,
            projection: "feed".into(),
            layer: Some(projection_spec::LayerSelector::Index(Dim::Literal(0))),
            operand: Some("up".into()),
            rows: row(0, 12),
        },
    ];
    d.projections.push(combined);
    let bound = bind(&d, &metadata()).unwrap();
    assert_eq!(
        bound.assemblies[0]
            .members
            .iter()
            .map(|m| m.operand.as_deref())
            .collect::<Vec<_>>(),
        [Some("gate"), Some("up")]
    );
    assert_eq!(
        bound.assemblies[0].tensors,
        [
            vec!["layers.0.feed.gate.weight".to_string()],
            vec!["layers.0.feed.up.weight".to_string()]
        ]
    );
}

#[test]
fn assembly_requires_a_layer_for_a_multilayer_source() {
    let mut d = declaration();
    let mut p = d.projections[8].clone();
    p.id = "context".into();
    p.shape[0] = Dim::Literal(8);
    p.source = Some(TensorSource {
        stem: None,
        tensor: "derived".into(),
    });
    p.assembly = vec![SourceMember {
        coordinates: Default::default(),
        inherit_split: false,
        projection: "q".into(),
        layer: None,
        operand: None,
        rows: row(0, 8),
    }];
    d.projections.push(p);
    assert!(verify(&d).unwrap_err().0.contains("explicit source layer"));
    d.projections.last_mut().unwrap().assembly[0].layer =
        Some(projection_spec::LayerSelector::Index(Dim::Literal(0)));
    verify(&d).unwrap();
}

#[test]
fn per_use_implementation_cannot_repeat_the_composition_default() {
    let mut d = declaration();
    let default = variant("fixture/legacy.v1", "fixture.legacy.v1");
    d.projections[0].composition = Some(Composition {
        implementations: Default::default(),
        numerical_contracts: Default::default(),
        implementation: Some(default.clone()),
        ..Default::default()
    });
    let mut c = contract(DType::Bf16);
    c.implementation = Some(default);
    d.projections[0].uses[0].contract = Some(c);
    assert!(
        verify(&d)
            .unwrap_err()
            .0
            .contains("repeats the default implementation")
    );
    d.projections[0].uses[0]
        .contract
        .as_mut()
        .unwrap()
        .implementation = None;
    verify(&d).unwrap();
}

#[test]
fn fusion_segments_match_member_widths_and_one_shared_stride() {
    let mut d = declaration();
    d.fusion_groups[0].outputs = vec![
        OutputSegment {
            buffer: "shared".into(),
            member: "q".into(),
            offset: Dim::Literal(0),
            row_stride: Dim::Literal(16),
            live_columns: Dim::Literal(1),
        },
        OutputSegment {
            buffer: "shared".into(),
            member: "k".into(),
            offset: Dim::Literal(8),
            row_stride: Dim::Literal(16),
            live_columns: Dim::Literal(8),
        },
    ];
    assert!(verify(&d).unwrap_err().0.contains("member output width"));
    d.fusion_groups[0].outputs[0].live_columns = Dim::Literal(8);
    d.fusion_groups[0].outputs[1].row_stride = Dim::Literal(32);
    assert!(verify(&d).unwrap_err().0.contains("shared row stride"));
    d.fusion_groups[0].outputs[1].row_stride = Dim::Literal(16);
    verify(&d).unwrap();
}

#[test]
fn verify_contract_requires_both_limits_and_covers_the_bucket() {
    let mut d = declaration();
    d.projections[2].uses[2].contract = Some(contract(DType::Bf16));
    assert!(verify(&d).unwrap_err().0.contains("needs both"));
    let c = d.projections[2].uses[2].contract.as_mut().unwrap();
    c.per_item_verify = Some(Dim::Literal(1));
    c.aggregate_verify = Some(Dim::Literal(1));
    assert!(verify(&d).unwrap_err().0.contains("requested capacity"));
    d.projections[2].uses[2]
        .contract
        .as_mut()
        .unwrap()
        .aggregate_verify = Some(Dim::Literal(4));
    verify(&d).unwrap();
}

#[test]
fn norm_group_width_matches_its_shared_affine_weight() {
    let mut d = declaration();
    d.side_tensors[0].norm = Some(NormContract {
        reduction_width: Dim::Literal(7),
        groups: Dim::Literal(1),
        affine: NormAffine::Shared,
        reduction: projection_spec::NormReduction::ShuffleDown,
        epsilon_bits: 1e-6f32.to_bits(),
        one_plus_weight: true,
        weight_dtypes: vec![DType::Bf16, DType::F32],
    });
    assert!(verify(&d).unwrap_err().0.contains("affine storage"));
    d.side_tensors[0].norm.as_mut().unwrap().reduction_width = Dim::Literal(16);
    verify(&d).unwrap();
}

#[test]
fn admit_checks_implicit_materialization_and_global_layer_identity() {
    let d = declaration();
    assert!(
        d.admit(
            &geometry(),
            "q",
            Some(99),
            1,
            ExecutionPhase::Prefill,
            FormatId::Bf16
        )
        .unwrap_err()
        .0
        .contains("not materialized")
    );
    assert!(
        d.admit(
            &geometry(),
            "table",
            Some(3),
            1,
            ExecutionPhase::Prefill,
            FormatId::Bf16
        )
        .unwrap_err()
        .0
        .contains("unstemmed")
    );
    d.admit(
        &geometry(),
        "q",
        Some(0),
        1,
        ExecutionPhase::Prefill,
        FormatId::Bf16,
    )
    .unwrap();
    d.admit(
        &geometry(),
        "table",
        None,
        1,
        ExecutionPhase::Prefill,
        FormatId::Bf16,
    )
    .unwrap();
}

#[test]
fn head_group_counts_are_exact_and_run_dimensions_agree() {
    let mut d = declaration();
    d.projections[0].split = projection_spec::SplitClass::HeadGrouped {
        heads: Dim::Literal(1),
        head_dim: Dim::Literal(2),
        tile: Dim::Literal(4),
    };
    assert!(verify(&d).unwrap_err().0.contains("head count"));
    let mut d = declaration();
    d.projections[0].parts[0].head_runs = vec![HeadRun {
        id: "head".into(),
        rows: row(0, 4),
        heads: Dim::Literal(2),
        head_dim: Dim::Literal(2),
    }];
    assert!(verify(&d).unwrap_err().0.contains("head dimension"));
}

#[test]
fn stage_and_external_ids_share_the_declaration_namespace() {
    let mut d = declaration();
    d.external_tensors.push(ExternalTensor {
        optional: false,
        id: "q".into(),
        owner: "encoder".into(),
        image_role: "image".into(),
        source: TensorSource {
            stem: None,
            tensor: "encoder.weight".into(),
        },
        shape: vec![Dim::Literal(2)],
        dtype: DType::Bf16,
    });
    assert!(verify(&d).unwrap_err().0.contains("duplicate id `q`"));
    let mut d = declaration();
    for p in &mut d.projections[0..2] {
        p.composition = Some(Composition {
            implementations: Default::default(),
            numerical_contracts: Default::default(),
            pre: vec![Stage {
                id: "hash".into(),
                contract: variant("fixture/hash.v1", "fixture.hash.v1"),
                phases: vec![ExecutionPhase::Decode],
                mutates: vec![],
            }],
            ..Default::default()
        });
    }
    assert!(verify(&d).unwrap_err().0.contains("duplicate id `hash`"));
}

#[test]
fn concrete_claims_refuse_accidental_stem_overlap_with_both_owners() {
    let mut d = declaration();
    d.layer_stems.push(projection_spec::LayerStem {
        id: "alternate".into(),
        template: "layers.{i}.feed".into(),
        indices: projection_spec::LayerIndices::List {
            indices: vec![Dim::Literal(0)],
        },
        execution: None,
        support: vec![],
    });
    let mut p = d.projections[8].clone();
    p.id = "alternate_projection".into();
    p.source = Some(TensorSource {
        stem: Some("alternate".into()),
        tensor: "{stem}.gate.weight".into(),
    });
    p.shape = [Dim::Key("intermediate".into()), Dim::Key("hidden".into())];
    d.projections.push(p);
    verify(&d).unwrap();
    let error = bind(&d, &metadata()).unwrap_err();
    assert!(
        error
            .0
            .contains("claimed by `feed` and `alternate_projection`"),
        "{error}"
    );
}

#[test]
fn layer_indices_aliases_and_assembly_layers_resolve_geometry_offsets() {
    let mut text: Value = serde_json::from_str(FIXTURE).unwrap();
    text["layer_stems"][1]["indices"] = json!({"kind":"list","indices":["layers"]});
    text["layer_stems"][1]["execution"] = json!({
        "indices":[{"op":"sum","terms":["layers",1]}],
        "aliases":[{"execution_index":{"op":"sum","terms":["layers",1]},"source_index":"layers"}]
    });
    let mut p = text["projections"][8].clone();
    p["id"] = json!("derived");
    p["assembly"] = json!([{"projection":"draft","layer":"layers","operand":null,"rows":{"start":0,"length":"hidden"}}]);
    text["projections"].as_array_mut().unwrap().push(p);
    let d = parse(&text.to_string(), &geometry(), &registry()).unwrap();
    assert_eq!(
        d.layer_stems[1].indices.resolve(&geometry()).unwrap(),
        vec![2]
    );
    d.admit(
        &geometry(),
        "draft",
        Some(3),
        1,
        ExecutionPhase::Decode,
        FormatId::Bf16,
    )
    .unwrap();
    text["layer_stems"][1]["indices"] =
        json!({"kind":"range","start":"layers","end":{"op":"sum","terms":["layers",1]}});
    parse(&text.to_string(), &geometry(), &registry()).unwrap();
}

#[test]
fn shared_kernel_epilogues_refuse_unknown_schema_fields() {
    let mut value = serde_json::to_value(kernel_core::contract::Epilogues::default()).unwrap();
    value["unknown"] = json!(true);
    let error = serde_json::from_value::<kernel_core::contract::Epilogues>(value).unwrap_err();
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn lookup_shards_cannot_repeat_a_checkpoint_source() {
    let mut d = declaration();
    d.projections[6].kind = projection_spec::ProjectionKind::LookupTable {};
    d.projections[6].source = None;
    d.projections[6].lookup = Some(LookupAssembly {
        companions: Default::default(),
        shards: projection_spec::LookupShards::Explicit(vec![
            LookupShard {
                id: "a".into(),
                source: TensorSource {
                    stem: None,
                    tensor: "same.weight".into(),
                },
                rows: row(0, 16),
            },
            LookupShard {
                id: "b".into(),
                source: TensorSource {
                    stem: None,
                    tensor: "same.weight".into(),
                },
                rows: row(16, 16),
            },
        ]),
        heads: vec![LookupHead {
            id: "all".into(),
            vocabulary: row(0, 32),
        }],
        padding_owner: PaddingOwner::LastRank {},
    });
    assert!(
        verify(&d)
            .unwrap_err()
            .0
            .contains("duplicate lookup shard source")
    );
}

#[test]
fn identical_stem_patterns_are_valid_when_concrete_tensor_claims_are_distinct() {
    let mut d = declaration();
    let mut alias = d.layer_stems[1].clone();
    alias.id = "drafter_alias".into();
    d.layer_stems.push(alias);
    d.projections[8].source.as_mut().unwrap().stem = Some("drafter_alias".into());
    d.side_tensors[3].source.stem = Some("drafter_alias".into());
    verify(&d).unwrap();
    bind(&d, &metadata()).unwrap();
}

#[test]
fn compositions_resolve_one_authoritative_numerical_recipe_for_source_formats() {
    use kernel_core::precision::format::A16Recipe;
    let recipe = projection_spec::contracts::NumericalRecipe {
        numerical_contract: "fixture.authoritative.v1".into(),
        a16_recipe: A16Recipe::ScaledWeight,
    };
    let mut c = Composition {
        numerical_contracts: std::collections::BTreeMap::from([
            (FormatId::Bf16, recipe.clone()),
            (FormatId::Fp8Block128, recipe.clone()),
        ]),
        ..Default::default()
    };
    assert_eq!(
        c.numerical_for_sources([FormatId::Bf16, FormatId::Fp8Block128])
            .unwrap(),
        Some(&recipe)
    );
    c.numerical_contracts
        .get_mut(&FormatId::Fp8Block128)
        .unwrap()
        .numerical_contract = "fixture.other.v1".into();
    assert!(
        c.numerical_for_sources([FormatId::Bf16, FormatId::Fp8Block128])
            .unwrap_err()
            .to_string()
            .contains("incompatible numerical recipes")
    );
    c.numerical_contracts.remove(&FormatId::Fp8Block128);
    assert!(
        c.numerical_for_sources([FormatId::Bf16, FormatId::Fp8Block128])
            .is_err()
    );
    assert_eq!(c.numerical_for_sources([]).unwrap(), None);
}
