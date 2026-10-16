use kernel_core::contract::{DType, WeightFormat};
use model_compiler_sdk::quantization::projection::metadata::HeaderCompanionRules;
use projection_spec::bind::{CheckpointMetadata, TensorInfo};
use projection_spec::registry::Registry;
use projection_spec::{Geometry, parse};
use std::collections::BTreeMap;

fn info(shape: &[u64], dtype: &str) -> TensorInfo {
    TensorInfo {
        shape: shape.into(),
        dtype: dtype.into(),
        file: Some("weights.safetensors".into()),
    }
}
fn declaration(rows: u64, k: u64, parts: serde_json::Value) -> projection_spec::Declaration {
    parse(&serde_json::json!({"schema": "chopinfer.projection.v1", "namespace": "synthetic", "layer_stems": [], "projections": [{
        "id": "matrix", "source": {"tensor": "matrix.weight"}, "shape": [rows,k],
        "parts": parts, "kind": {"kind": "dense"}, "split": {"class": "replicate"},
        "epilogues": kernel_core::contract::Epilogues::default(), "numerical_contract": "test-v1", "uses": [{"phase": "Decode", "capacity": 1, "graph": false}]
    }], "fusion_groups": [], "side_tensors": []}).to_string(), &Geometry::new(), &Registry::default()).unwrap()
}
#[test]
fn shared_headers_import_restarted_grids_without_fabricating_a_uniform_bank() {
    let d = declaration(
        258,
        128,
        serde_json::json!([
            {"id":"first", "rows":129,"scale_origin":[0,0]},
            {"id":"second", "rows":129,"scale_origin":[0,0]}
        ]),
    );
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[258, 128], "F8_E4M3")),
            ("matrix.weight_scale_inv".into(), info(&[4, 1], "BF16")),
        ]),
        quantization_config: serde_json::json!({"quant_method":"fp8", "weight_block_size":[128,128],"activation_scheme":"dynamic"}),
        ..Default::default()
    };
    let rules = HeaderCompanionRules::new(&m, None).unwrap();
    let b = d
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    let imported = &b.operands[0].imported;
    assert_eq!(imported.parts.len(), 2);
    assert_eq!(imported.parts[1].codes.offset, 129 * 128);
    let scale = imported.parts[1].local_scale.as_ref().unwrap();
    assert_eq!(scale.tensor.offset, 4);
    assert_eq!(scale.tensor.dtype, DType::Bf16);
    assert_eq!(
        imported.weight.format,
        WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32)
    );
    assert!(imported.weight.validate().is_err());
    let mut wrong = m.clone();
    wrong
        .tensors
        .get_mut("matrix.weight_scale_inv")
        .unwrap()
        .shape = vec![3, 1];
    assert!(
        d.bind(&Geometry::new(), &Registry::default(), &wrong, &rules)
            .unwrap_err()
            .0
            .contains("scale")
    );
}

#[test]
fn shared_imports_keep_calibration_provenance_and_lookup_weight_only_semantics() {
    use projection_spec::bind::{BindingRequest, CompanionRules};
    let mut d = declaration(2, 32, serde_json::json!([]));
    let mut m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[2, 16], "U8")),
            ("matrix.weight_scale".into(), info(&[2, 2], "F8_E4M3")),
            ("matrix.weight_scale_2".into(), info(&[], "F32")),
            ("matrix.input_scale".into(), info(&[], "F32")),
        ]),
        quantization_config: serde_json::json!({"quant_algo":"NVFP4","group_size":16}),
        ..Default::default()
    };
    let rules = HeaderCompanionRules::new(&m, None).unwrap();
    let b = d
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    assert_eq!(
        b.operands[0].imported.companions[0].tensor,
        "matrix.input_scale"
    );
    assert_eq!(
        b.operands[0].imported.companions[0].file.as_deref(),
        Some("weights.safetensors")
    );
    m.tensors.remove("matrix.input_scale");
    d.projections[0].kind = projection_spec::ProjectionKind::LookupTable {};
    let imported = rules
        .import(
            &BindingRequest {
                projection: &d.projections[0],
                tensor: "matrix.weight",
                logical_shape: &[2, 32],
                geometry: &Geometry::new(),
                layer: None,
                operand: None,
                expert: None,
                shard: None,
                head: None,
            },
            &m,
        )
        .unwrap();
    assert!(imported.companions.is_empty());
    assert_eq!(imported.weight.format, WeightFormat::Nvfp4);
}

/// A config group that targets the `Linear` class reaches a dense
/// projection, not a table, and loses to an exclusion.
#[test]
fn a_linear_class_group_reaches_projections_but_not_tables() {
    use projection_spec::bind::{BindingRequest, CompanionRules};
    let nvfp4 =
        serde_json::json!({"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16});
    let config = serde_json::json!({
        "quant_method": "modelopt", "quant_algo": "NVFP4",
        "config_groups": {"group_0": {"weights": nvfp4, "input_activations": nvfp4, "targets": ["Linear"]}},
    });
    let mut d = declaration(2, 32, serde_json::json!([]));
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[2, 16], "U8")),
            ("matrix.weight_scale".into(), info(&[2, 2], "F8_E4M3")),
            ("matrix.weight_scale_2".into(), info(&[], "F32")),
            ("matrix.input_scale".into(), info(&[], "F32")),
        ]),
        quantization_config: config.clone(),
        ..Default::default()
    };
    let rules = HeaderCompanionRules::new(&m, None).unwrap();
    let b = d
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    assert_eq!(b.operands[0].imported.weight.format, WeightFormat::Nvfp4);
    assert_eq!(
        b.operands[0].imported.companions[0].tensor,
        "matrix.input_scale"
    );
    let request = |d: &projection_spec::Declaration| {
        let m = CheckpointMetadata {
            tensors: BTreeMap::from([("matrix.weight".into(), info(&[2, 32], "BF16"))]),
            ..m.clone()
        };
        rules
            .import(
                &BindingRequest {
                    projection: &d.projections[0],
                    tensor: "matrix.weight",
                    logical_shape: &[2, 32],
                    geometry: &Geometry::new(),
                    layer: None,
                    operand: None,
                    expert: None,
                    shard: None,
                    head: None,
                },
                &m,
            )
            .map(|imported| imported.weight.format)
    };
    d.projections[0].kind = projection_spec::ProjectionKind::LookupTable {};
    assert_eq!(request(&d).unwrap(), WeightFormat::Bf16);
    let mut excluded = config;
    excluded["ignore"] = serde_json::json!(["matrix"]);
    let m = CheckpointMetadata {
        quantization_config: excluded,
        ..m
    };
    let rules = HeaderCompanionRules::new(&m, None).unwrap();
    let d = declaration(2, 32, serde_json::json!([]));
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([("matrix.weight".into(), info(&[2, 32], "BF16"))]),
        ..m
    };
    let b = d
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    assert_eq!(b.operands[0].imported.weight.format, WeightFormat::Bf16);
}

/// The header adapter imports under the family's options: a KV cache hint
/// is refused by default and recorded under the record-and-ignore policy.
#[test]
fn header_rules_import_under_the_family_options() {
    use model_compiler_sdk::quantization::import::{ImportOptions, KvHintPolicy};
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([("matrix.weight".into(), info(&[2, 32], "BF16"))]),
        quantization_config: serde_json::json!({
            "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {"other": {"quant_algo": "NVFP4", "group_size": 16}},
            "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
        }),
        ..Default::default()
    };
    let refused = HeaderCompanionRules::new(&m, None).err().unwrap();
    assert!(refused.0.contains("kv_cache_scheme"), "{}", refused.0);
    let options = ImportOptions {
        kv_hint: KvHintPolicy::RecordAndIgnore,
    };
    let rules = HeaderCompanionRules::new_with(&m, None, &options).unwrap();
    let b = declaration(2, 32, serde_json::json!([]))
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    assert_eq!(b.operands[0].imported.weight.format, WeightFormat::Bf16);
}

#[test]
fn explicit_table_companions_preserve_identity_and_format_scope() {
    use kernel_core::precision::format::FormatId;
    use model_compiler_sdk::quantization::projection::metadata::{CompanionNames, HeaderRole};
    let mut d = declaration(4, 16, serde_json::json!([]));
    d.projections[0].source = None;
    d.projections[0].kind = projection_spec::ProjectionKind::LookupTable {};
    d.projections[0].lookup = Some(serde_json::from_value(serde_json::json!({
        "companions":{"default":{"local":"shard","global":"table"},"formats":{"fp8_per_tensor":{"local":"table","global":"table"}}},
        "shards":[
            {"id":"left","source":{"tensor":"table.left.weight"},"rows":{"start":0,"length":2}},
            {"id":"right","source":{"tensor":"table.right.weight"},"rows":{"start":2,"length":2}}
        ],
        "heads":[{"id":"h","vocabulary":{"start":0,"length":4}}],
        "padding_owner":{"owner":"last_rank"}
    })).unwrap());
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("table.left.weight".into(), info(&[2, 16], "F8_E4M3")),
            ("table.right.weight".into(), info(&[2, 16], "F8_E4M3")),
            ("table.weight_scale".into(), info(&[], "F32")),
        ]),
        quantization_config: serde_json::json!({"quant_algo":"FP8"}),
        ..Default::default()
    };
    let mut rules = HeaderCompanionRules::new(&m, None).unwrap();
    rules.roles.insert(
        "matrix".into(),
        HeaderRole {
            format_companions: BTreeMap::from([(
                FormatId::Fp8PerTensor,
                CompanionNames {
                    local: Some("table.weight_scale".into()),
                    global: None,
                },
            )]),
            ..Default::default()
        },
    );
    let b = d
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    assert_eq!(b.operands.len(), 2);
    assert!(
        b.operands
            .iter()
            .all(|b| b.bindings[0].tensor == "table.weight_scale")
    );
}

#[test]
fn independent_frames_honor_grid_origins_and_scalar_storage_order() {
    use model_compiler_sdk::quantization::projection::metadata::{HeaderRole, ScalarSharing};
    let mut d = declaration(
        4,
        32,
        serde_json::json!([
            {"id":"a","rows":2,"scale_origin":[0,0]},
            {"id":"b","rows":2,"scale_origin":[0,0]}
        ]),
    );
    d.projections[0].operands = vec!["gate".into(), "up".into()];
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[8, 16], "U8")),
            ("matrix.weight_scale".into(), info(&[8, 2], "F8_E4M3")),
            ("matrix.weight_scale_2".into(), info(&[2], "F32")),
            ("matrix.input_scale".into(), info(&[4], "F32")),
        ]),
        quantization_config: serde_json::json!({"quant_algo":"NVFP4","group_size":16}),
        ..Default::default()
    };
    let mut rules = HeaderCompanionRules::new(&m, None).unwrap();
    rules.roles.insert(
        "matrix".into(),
        HeaderRole {
            global_sharing: Some(ScalarSharing::Operand),
            calibration_sharing: Some(ScalarSharing::Frame),
            ..Default::default()
        },
    );
    let b = d
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    let parts = &b.operands[0].imported.parts;
    assert_eq!(parts.len(), 4);
    assert_eq!(parts[2].global_scale.as_ref().unwrap().tensor.offset, 4);
    assert_eq!(parts[3].local_scale.as_ref().unwrap().tensor.offset, 12);
    assert_eq!(b.operands[0].imported.companions[0].shape, vec![4]);
    rules.roles.get_mut("matrix").unwrap().global_sharing = Some(ScalarSharing::Shared);
    assert!(
        d.bind(&Geometry::new(), &Registry::default(), &m, &rules)
            .unwrap_err()
            .0
            .contains("cardinality")
    );
}

#[test]
fn header_index_adapter_keeps_file_offsets_separate_from_tensor_views() {
    let d = declaration(2, 4, serde_json::json!([]));
    let headers = BTreeMap::from([(
        "weights.safetensors".into(),
        serde_json::json!({
            "matrix.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[128,144]}
        }),
    )]);
    let index = serde_json::json!({"weight_map":{"matrix.weight":"weights.safetensors"}});
    let m =
        CheckpointMetadata::from_headers(&headers, Some(&index), serde_json::Value::Null).unwrap();
    let rules = HeaderCompanionRules::new(&m, None).unwrap();
    let b = d
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    assert_eq!(b.operands[0].imported.weight.codes.offset, 0);
    assert_eq!(b.operands[0].imported.weight.codes.byte_length, 16);
}

#[test]
fn independent_block_views_keep_nonzero_scale_origins() {
    let d = declaration(
        128,
        128,
        serde_json::json!([
            {"id":"a","rows":64,"scale_origin":[0,0]},
            {"id":"b","rows":64,"scale_origin":[96,0]}
        ]),
    );
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[128, 128], "F8_E4M3")),
            ("matrix.weight_scale".into(), info(&[3, 1], "F32")),
        ]),
        quantization_config: serde_json::json!({"quant_method":"fp8","weight_block_size":[128,128],"activation_scheme":"dynamic"}),
        ..Default::default()
    };
    let rules = HeaderCompanionRules::new(&m, None).unwrap();
    let b = d
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    let scale = b.operands[0].imported.parts[1]
        .local_scale
        .as_ref()
        .unwrap();
    assert_eq!(scale.groups[0].origin, 96);
    assert_eq!(scale.tensor.shape, vec![2, 1]);
    assert_eq!(scale.tensor.offset, 4);
}

#[test]
fn independent_parts_refuse_transposed_code_and_scale_headers() {
    let d = declaration(
        258,
        128,
        serde_json::json!([{"id":"first","rows":129,"scale_origin":[0,0]},
        {"id":"second","rows":129,"scale_origin":[0,0]}]),
    );
    for (codes, scales, accepted) in [
        (vec![258, 128], vec![4, 1], true),
        (vec![128, 258], vec![4, 1], false),
        (vec![258, 128], vec![1, 4], false),
    ] {
        let m = CheckpointMetadata {
            tensors: BTreeMap::from([
                ("matrix.weight".into(), info(&codes, "F8_E4M3")),
                ("matrix.weight_scale_inv".into(), info(&scales, "BF16")),
            ]),
            quantization_config: serde_json::json!({"quant_method":"fp8","weight_block_size":[128,128],"activation_scheme":"dynamic"}),
            ..Default::default()
        };
        let rules = HeaderCompanionRules::new(&m, None).unwrap();
        let result = d.bind(&Geometry::new(), &Registry::default(), &m, &rules);
        assert_eq!(result.is_ok(), accepted, "code {codes:?}, scale {scales:?}");
        if !accepted {
            assert!(result.unwrap_err().to_string().contains("shape"));
        }
    }
}

#[test]
fn scalar_vectors_on_parts_require_the_declared_sharing_shape() {
    use model_compiler_sdk::quantization::projection::metadata::{HeaderRole, ScalarSharing};
    let mut d = declaration(
        4,
        32,
        serde_json::json!([{"id":"a","rows":2,"scale_origin":[0,0]},{"id":"b","rows":2,"scale_origin":[0,0]}]),
    );
    d.projections[0].operands = vec!["gate".into(), "up".into()];
    let mut m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[8, 16], "U8")),
            ("matrix.weight_scale".into(), info(&[8, 2], "F8_E4M3")),
            ("matrix.weight_scale_2".into(), info(&[2], "F32")),
        ]),
        quantization_config: serde_json::json!({"quant_algo":"W4A16_NVFP4","group_size":16}),
        ..Default::default()
    };
    let mut rules = HeaderCompanionRules::new(&m, None).unwrap();
    rules.roles.insert(
        "matrix".into(),
        HeaderRole {
            global_sharing: Some(ScalarSharing::Operand),
            ..Default::default()
        },
    );
    assert!(
        d.bind(&Geometry::new(), &Registry::default(), &m, &rules)
            .is_ok()
    );
    m.tensors.get_mut("matrix.weight_scale_2").unwrap().shape = vec![1, 2];
    assert!(
        d.bind(&Geometry::new(), &Registry::default(), &m, &rules)
            .unwrap_err()
            .to_string()
            .contains("shape")
    );
}

#[test]
fn header_aliases_resolve_only_required_roles_and_refuse_default_conflicts() {
    use model_compiler_sdk::quantization::projection::metadata::{CompanionNames, HeaderRole};
    let d = declaration(128, 128, serde_json::json!([]));
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[128, 128], "F8_E4M3")),
            ("matrix.weight_scale_inv".into(), info(&[1, 1], "BF16")),
        ]),
        quantization_config: serde_json::json!({"quant_method":"fp8","weight_block_size":[128,128],"activation_scheme":"dynamic"}),
        ..Default::default()
    };
    let mut rules = HeaderCompanionRules::new(&m, None).unwrap();
    rules.roles.insert(
        "matrix".into(),
        HeaderRole {
            companions: CompanionNames {
                local: None,
                global: Some("absent.global".into()),
            },
            ..Default::default()
        },
    );
    assert!(
        d.bind(&Geometry::new(), &Registry::default(), &m, &rules)
            .is_ok()
    );
    let mut duplicate = m.clone();
    duplicate
        .tensors
        .insert("alternate.scale".into(), info(&[1, 1], "BF16"));
    rules.roles.get_mut("matrix").unwrap().companions.local = Some("alternate.scale".into());
    assert!(
        d.bind(&Geometry::new(), &Registry::default(), &duplicate, &rules)
            .unwrap_err()
            .to_string()
            .contains("alias conflicts")
    );
}

#[test]
fn header_calibrations_retain_frame_indices_and_scalar_offsets() {
    use kernel_core::precision::ScaleScope;
    use model_compiler_sdk::quantization::projection::metadata::{HeaderRole, ScalarSharing};
    let mut d = declaration(
        4,
        32,
        serde_json::json!([{"id":"a","rows":2,"scale_origin":[0,0]},{"id":"b","rows":2,"scale_origin":[0,0]}]),
    );
    d.projections[0].operands = vec!["gate".into(), "up".into()];
    d.projections[0].kind = projection_spec::ProjectionKind::ExpertBank {
        experts: projection_spec::Dim::Literal(2),
        top_k: projection_spec::Dim::Literal(1),
    };
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[2, 8, 16], "U8")),
            ("matrix.weight_scale".into(), info(&[2, 8, 2], "F8_E4M3")),
            ("matrix.weight_scale_2".into(), info(&[2], "F32")),
            ("matrix.input_scale".into(), info(&[8], "F32")),
        ]),
        quantization_config: serde_json::json!({"quant_algo":"NVFP4","group_size":16}),
        ..Default::default()
    };
    let mut rules = HeaderCompanionRules::new(&m, None).unwrap();
    rules.roles.insert(
        "matrix".into(),
        HeaderRole {
            global_sharing: Some(ScalarSharing::Operand),
            calibration_sharing: Some(ScalarSharing::Frame),
            ..Default::default()
        },
    );
    let bound = d
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    let imported = &bound.operands[0].imported;
    assert_eq!(
        imported.weight.global_scale.as_ref().unwrap().scope,
        ScaleScope::Projection
    );
    assert_eq!(imported.calibrations.len(), 8);
    for (i, calibration) in imported.calibrations.iter().enumerate() {
        assert_eq!(calibration.expert, (i / 4) as u64);
        assert_eq!(calibration.operand, ((i / 2) % 2) as u64);
        assert_eq!(
            calibration.part.as_deref(),
            Some(if i % 2 == 0 { "a" } else { "b" })
        );
        assert_eq!(calibration.scale.tensor.offset, i as u64 * 4);
        assert_eq!(calibration.scale.tensor.byte_length, 4);
    }
    let mut ordinary = declaration(4, 32, serde_json::json!([]));
    ordinary.projections[0].kind = projection_spec::ProjectionKind::ExpertBank {
        experts: projection_spec::Dim::Literal(2),
        top_k: projection_spec::Dim::Literal(1),
    };
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[2, 4, 16], "U8")),
            ("matrix.weight_scale".into(), info(&[2, 4, 2], "F8_E4M3")),
            ("matrix.weight_scale_2".into(), info(&[2], "F32")),
            ("matrix.input_scale".into(), info(&[2], "F32")),
        ]),
        quantization_config: serde_json::json!({"quant_algo":"NVFP4","group_size":16}),
        ..Default::default()
    };
    let rules = HeaderCompanionRules::new(&m, None).unwrap();
    let bound = ordinary
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    assert_eq!(
        bound.operands[0]
            .imported
            .calibrations
            .iter()
            .map(|c| (c.expert, c.scale.tensor.offset))
            .collect::<Vec<_>>(),
        vec![(0, 0), (1, 4)]
    );
}
#[test]
fn explicit_operand_calibration_without_parts_resolves_independent_frames() {
    use model_compiler_sdk::quantization::projection::metadata::{HeaderRole, ScalarSharing};
    let mut d = declaration(2, 32, serde_json::json!([]));
    d.projections[0].operands = vec!["gate".into(), "up".into()];
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[4, 16], "U8")),
            ("matrix.weight_scale".into(), info(&[4, 2], "F8_E4M3")),
            ("matrix.weight_scale_2".into(), info(&[], "F32")),
            ("matrix.input_scale".into(), info(&[2], "F32")),
        ]),
        quantization_config: serde_json::json!({"quant_algo":"NVFP4","group_size":16}),
        ..Default::default()
    };
    let mut rules = HeaderCompanionRules::new(&m, None).unwrap();
    rules.roles.insert(
        "matrix".into(),
        HeaderRole {
            calibration_sharing: Some(ScalarSharing::Operand),
            ..Default::default()
        },
    );
    let bound = d
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    assert_eq!(
        bound.operands[0]
            .imported
            .calibrations
            .iter()
            .map(|c| (c.operand, c.part.clone(), c.scale.tensor.offset))
            .collect::<Vec<_>>(),
        vec![(0, None, 0), (1, None, 4)]
    );
}
#[test]
fn headers_reject_config_changes_and_lookup_calibration() {
    let mut d = declaration(2, 32, serde_json::json!([]));
    let mut m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[2, 16], "U8")),
            ("matrix.weight_scale".into(), info(&[2, 2], "F8_E4M3")),
            ("matrix.weight_scale_2".into(), info(&[], "F32")),
            ("matrix.input_scale".into(), info(&[], "F32")),
        ]),
        quantization_config: serde_json::json!({"quant_algo":"NVFP4","group_size":16}),
        ..Default::default()
    };
    let rules = HeaderCompanionRules::new(&m, None).unwrap();
    let mut changed = m.clone();
    changed.quantization_config = serde_json::json!({"quant_algo":"MXFP8","group_size":32});
    assert!(
        d.bind(&Geometry::new(), &Registry::default(), &changed, &rules)
            .unwrap_err()
            .to_string()
            .contains("config changed")
    );
    d.projections[0].kind = projection_spec::ProjectionKind::LookupTable {};
    assert!(
        d.bind(&Geometry::new(), &Registry::default(), &m, &rules)
            .unwrap_err()
            .to_string()
            .contains("uninterpreted execution companion")
    );
    m.tensors.remove("matrix.input_scale");
    assert!(
        d.bind(&Geometry::new(), &Registry::default(), &m, &rules)
            .is_ok()
    );
}
#[test]
fn independent_mx_frames_preserve_u8_scale_storage_as_ue8m0() {
    let d = declaration(
        4,
        64,
        serde_json::json!([{"id":"a","rows":2,"scale_origin":[0,0]},{"id":"b","rows":2,"scale_origin":[0,0]}]),
    );
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[4, 64], "F8_E4M3")),
            ("matrix.weight_scale".into(), info(&[4, 2], "U8")),
        ]),
        quantization_config: serde_json::json!({"quant_algo":"MXFP8","group_size":32}),
        ..Default::default()
    };
    let rules = HeaderCompanionRules::new(&m, None).unwrap();
    let bound = d
        .bind(&Geometry::new(), &Registry::default(), &m, &rules)
        .unwrap();
    let parts = &bound.operands[0].imported.parts;
    assert_eq!(
        parts[1].local_scale.as_ref().unwrap().tensor.dtype,
        DType::Ue8M0
    );
    assert_eq!(parts[1].local_scale.as_ref().unwrap().tensor.offset, 4);
}
#[test]
fn header_templates_render_all_binding_indices() {
    use model_compiler_sdk::quantization::projection::metadata::{CompanionNames, HeaderRole};
    use projection_spec::bind::{BindingRequest, CompanionRules};
    let mut d = declaration(2, 32, serde_json::json!([]));
    d.projections[0].operands = vec!["gate".into(), "up".into()];
    d.projections[0].kind = projection_spec::ProjectionKind::ExpertBank {
        experts: projection_spec::Dim::Literal(4),
        top_k: projection_spec::Dim::Literal(1),
    };
    let m = CheckpointMetadata {
        tensors: BTreeMap::from([
            ("matrix.weight".into(), info(&[2, 16], "U8")),
            ("local.3.up.2.4.5".into(), info(&[2, 2], "F8_E4M3")),
            ("global.3.up.2.4.5".into(), info(&[], "F32")),
        ]),
        quantization_config: serde_json::json!({"quant_algo":"W4A16_NVFP4","group_size":16}),
        ..Default::default()
    };
    let mut rules = HeaderCompanionRules::new(&m, None).unwrap();
    rules.roles.insert(
        "matrix".into(),
        HeaderRole {
            targets: vec!["target.{i}.{operand}.{expert}.{shard}.{head}".into()],
            companions: CompanionNames {
                local: Some("local.{i}.{operand}.{expert}.{shard}.{head}".into()),
                global: Some("global.{i}.{operand}.{expert}.{shard}.{head}".into()),
            },
            ..Default::default()
        },
    );
    let mut request = BindingRequest {
        projection: &d.projections[0],
        tensor: "matrix.weight",
        logical_shape: &[2, 32],
        geometry: &Geometry::new(),
        layer: Some(3),
        operand: Some(1),
        expert: Some(2),
        shard: Some(4),
        head: Some(5),
    };
    let imported = rules.import(&request, &m).unwrap();
    assert_eq!(
        imported.weight.local_scale.unwrap().tensor.tensor,
        "local.3.up.2.4.5"
    );
    assert_eq!(
        imported.weight.global_scale.unwrap().tensor.tensor,
        "global.3.up.2.4.5"
    );
    request.head = None;
    assert!(
        rules
            .import(&request, &m)
            .unwrap_err()
            .to_string()
            .contains("unresolved")
    );
}
