mod common;

use common::*;
use kernel_core::contract::{DType, ScaleMode, WeightFormat};
use kernel_core::precision::{
    CodeEncoding, GroupAxis, ScaleConvention, ScaleScope, ScaleSpec, TensorSpan,
};
use projection_spec::bind::{
    BindingRequest, CheckpointMetadata, CompanionRules, ImportedOperand, TensorRequirement,
};
use projection_spec::registry::Registry;
use projection_spec::{Error, ProjectionKind};
use serde_json::json;

fn bind(metadata: &CheckpointMetadata) -> Result<projection_spec::bind::BoundDeclaration, Error> {
    declaration().bind(&geometry(), &Registry::default(), metadata, &Bf16Rules)
}

#[test]
fn binds_independent_synthetic_headers_in_declaration_order() {
    let bound = bind(&metadata()).unwrap();
    assert_eq!(bound.operands.len(), 26);
    assert_eq!(bound.side_tensors.len(), 7);
    assert_eq!(
        bound.operands[0].imported.weight.codes.tensor,
        "layers.0.q.weight"
    );
    assert_eq!(
        bound.operands[1].imported.weight.codes.tensor,
        "layers.1.q.weight"
    );
    let feed: Vec<_> = bound
        .operands
        .iter()
        .filter(|o| o.projection == "feed")
        .map(|o| (o.layer, o.operand))
        .collect();
    assert_eq!(
        feed,
        [
            (Some(0), Some(0)),
            (Some(0), Some(1)),
            (Some(1), Some(0)),
            (Some(1), Some(1))
        ]
    );
    let expert = bound
        .operands
        .iter()
        .find(|o| o.projection == "experts")
        .unwrap();
    assert_eq!(expert.imported.weight.logical_shape, [4, 12, 16]);
    let separate = bound
        .operands
        .iter()
        .find(|o| o.projection == "expert_out")
        .unwrap();
    assert_eq!(separate.expert, Some(0));
    assert_eq!(separate.imported.weight.logical_shape, [16, 12]);
    assert_eq!(bound.fusion_groups.len(), 8);
    assert!(
        bound
            .fusion_groups
            .iter()
            .all(|g| g.fused && g.members == ["q", "k"])
    );
}

#[test]
fn missing_and_unknown_tensors_are_named() {
    let mut metadata = metadata();
    metadata.tensors.remove("layers.1.q.weight");
    assert!(bind(&metadata).unwrap_err().0.contains("layers.1.q.weight"));
    metadata
        .tensors
        .insert("layers.1.q.weight".into(), info(&[8, 16], "BF16"));
    metadata
        .tensors
        .insert("surprise.weight".into(), info(&[1], "BF16"));
    assert!(
        bind(&metadata)
            .unwrap_err()
            .0
            .contains("unknown checkpoint tensor `surprise.weight`")
    );
}

#[test]
fn side_dtype_shape_and_missing_roles_are_named() {
    for (shape, dtype, why) in [
        (&[16][..], "F16", "not admitted"),
        (&[15][..], "BF16", "expected"),
    ] {
        let mut metadata = metadata();
        metadata
            .tensors
            .insert("layers.0.norm.weight".into(), info(shape, dtype));
        let error = bind(&metadata).unwrap_err();
        assert!(
            error.0.contains("layers.0.norm.weight") && error.0.contains(why),
            "{error}"
        );
    }
    let mut metadata = metadata();
    metadata.tensors.remove("layers.0.norm.weight");
    assert!(
        bind(&metadata)
            .unwrap_err()
            .0
            .contains("layers.0.norm.weight")
    );
    metadata
        .tensors
        .insert("layers.0.norm.weight".into(), info(&[16], "F32"));
    bind(&metadata).unwrap();
}

#[test]
fn incorrect_weight_metadata_and_quantization_recipe_are_refused() {
    for (shape, dtype) in [(&[7, 16][..], "BF16"), (&[8, 16][..], "F32")] {
        let mut metadata = metadata();
        metadata
            .tensors
            .insert("layers.0.q.weight".into(), info(shape, dtype));
        assert!(bind(&metadata).unwrap_err().0.contains("layers.0.q.weight"));
    }
    let mut metadata = metadata();
    metadata.quantization_config = json!({"recipe":"unsupported"});
    let error = bind(&metadata).unwrap_err();
    assert!(error.0.contains("layers.0.q.weight") && error.0.contains("unsupported quantization"));
}

struct ChunkRules;
impl CompanionRules for ChunkRules {
    fn import(
        &self,
        request: &BindingRequest<'_>,
        metadata: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        let mut imported = bf16(request, metadata)?;
        for part in &request.projection.parts {
            imported.companions.push(TensorRequirement {
                tensor: format!("{}.{}.scale", request.tensor, part.id),
                dtype: "F32".into(),
                shape: vec![part.rows.resolve(request.geometry)?],
                file: Some("scales.safetensors".into()),
            });
        }
        Ok(imported)
    }
}

#[test]
fn chunk_companion_hook_enforces_names_shapes_dtypes_and_index_files() {
    let bind = |m: &CheckpointMetadata| {
        declaration().bind(&geometry(), &Registry::default(), m, &ChunkRules)
    };
    let mut metadata = metadata();
    assert!(
        bind(&metadata)
            .unwrap_err()
            .0
            .contains("layers.0.q.weight.query.scale")
    );
    for layer in 0..2 {
        for part in ["query", "key"] {
            let mut scale = info(&[4], "F32");
            scale.file = Some("scales.safetensors".into());
            metadata
                .tensors
                .insert(format!("layers.{layer}.q.weight.{part}.scale"), scale);
        }
    }
    bind(&metadata).unwrap();
    let tensor = "layers.0.q.weight.key.scale";
    for scale in [info(&[4], "F32"), info(&[3], "F32"), info(&[4], "BF16")] {
        let mut altered = metadata.clone();
        altered.tensors.insert(tensor.into(), scale);
        assert!(bind(&altered).unwrap_err().0.contains(tensor));
    }
}

struct MixedRules;
impl CompanionRules for MixedRules {
    fn import(
        &self,
        request: &BindingRequest<'_>,
        metadata: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        let mut imported = bf16(request, metadata)?;
        if request.projection.id == "k" {
            imported.weight.format = WeightFormat::Fp8PerTensor;
            imported.weight.encoding = CodeEncoding::E4M3;
            imported.weight.codes.dtype = DType::Fp8E4M3;
            imported.weight.codes.byte_length /= 2;
            imported.weight.local_scale = Some(ScaleSpec {
                tensor: TensorSpan {
                    tensor: format!("{}.scale", request.tensor),
                    dtype: DType::F32,
                    shape: vec![1, 1],
                    offset: 0,
                    byte_length: 4,
                },
                scope: ScaleScope::Tensor,
                convention: ScaleConvention::Multiply,
                groups: vec![],
            });
        }
        Ok(imported)
    }
}

fn mixed_metadata() -> CheckpointMetadata {
    let mut metadata = metadata();
    for layer in 0..2 {
        let name = format!("layers.{layer}.k.weight");
        metadata.tensors.get_mut(&name).unwrap().dtype = "F8_E4M3".into();
        metadata
            .tensors
            .insert(format!("{name}.scale"), info(&[1, 1], "F32"));
    }
    metadata
}

#[test]
fn mixed_imported_formats_fall_back_to_ordered_separate_members() {
    let bound = declaration()
        .bind(
            &geometry(),
            &Registry::default(),
            &mixed_metadata(),
            &MixedRules,
        )
        .unwrap();
    assert!(
        bound
            .fusion_groups
            .iter()
            .all(|g| !g.fused && g.members == ["q", "k"])
    );
}

#[test]
fn missing_or_wrong_standard_scale_companions_are_named() {
    for wrong in [
        None,
        Some(info(&[1, 2], "F32")),
        Some(info(&[1, 1], "BF16")),
    ] {
        let mut metadata = mixed_metadata();
        let name = "layers.1.k.weight.scale";
        metadata.tensors.remove(name);
        if let Some(info) = wrong {
            metadata.tensors.insert(name.into(), info);
        }
        let error = declaration()
            .bind(&geometry(), &Registry::default(), &metadata, &MixedRules)
            .unwrap_err();
        assert!(error.0.contains(name), "{error}");
    }
}

#[test]
fn packed_operands_and_unexpanded_lookup_axes_bind_exact_shapes() {
    let mut d = declaration();
    let mut metadata = metadata();
    d.projections[3].source.as_mut().unwrap().tensor = "{stem}.feed.weight".into();
    for layer in 0..2 {
        metadata
            .tensors
            .remove(&format!("layers.{layer}.feed.gate.weight"));
        metadata
            .tensors
            .remove(&format!("layers.{layer}.feed.up.weight"));
        metadata.tensors.insert(
            format!("layers.{layer}.feed.weight"),
            info(&[24, 16], "BF16"),
        );
    }
    d.projections[7].lookup = None;
    d.projections[7].source = Some(projection_spec::TensorSource {
        stem: None,
        tensor: "hash.weight".into(),
    });
    for shard in 0..2 {
        for head in 0..2 {
            metadata
                .tensors
                .remove(&format!("hash.{shard}.{head}.weight"));
        }
    }
    metadata
        .tensors
        .insert("hash.weight".into(), info(&[32, 4], "BF16"));
    let bound = d
        .bind(&geometry(), &Registry::default(), &metadata, &Bf16Rules)
        .unwrap();
    assert_eq!(
        bound
            .operands
            .iter()
            .find(|o| o.projection == "feed")
            .unwrap()
            .imported
            .weight
            .logical_shape,
        [24, 16]
    );
    assert_eq!(
        bound
            .operands
            .iter()
            .find(|o| o.projection == "hash")
            .unwrap()
            .imported
            .weight
            .logical_shape,
        [32, 4]
    );
}

struct BadImportRules(u8);
impl CompanionRules for BadImportRules {
    fn import(
        &self,
        request: &BindingRequest<'_>,
        metadata: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        let mut imported = bf16(request, metadata)?;
        match self.0 {
            0 => imported.weight.logical_shape[0] += 1,
            1 => imported.weight.codes.tensor = "another.weight".into(),
            2 => imported.weight.codes.offset = 1,
            3 => imported.weight.codes.byte_length += 1,
            4 => imported.weight.format = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            5 => {
                imported.companions = vec![TensorRequirement {
                    tensor: request.tensor.into(),
                    dtype: "BF16".into(),
                    shape: vec![8, 16],
                    file: None,
                }]
            }
            _ => unreachable!(),
        }
        Ok(imported)
    }
}

#[test]
fn importer_cannot_change_source_geometry_encoding_or_span_bounds() {
    for scenario in 0..6 {
        let error = declaration()
            .bind(
                &geometry(),
                &Registry::default(),
                &metadata(),
                &BadImportRules(scenario),
            )
            .unwrap_err();
        assert!(error.0.contains("layers.0.q.weight"), "{error}");
    }
}

#[test]
fn undeclared_source_reuse_names_both_claimants() {
    let mut d = declaration();
    d.projections[1].source = d.projections[0].source.clone();
    let error = d
        .bind(&geometry(), &Registry::default(), &metadata(), &Bf16Rules)
        .unwrap_err();
    assert!(error.0.contains("claimed by `q` and `k`"));
    let mut d = declaration();
    d.projections[5].kind = ProjectionKind::ExpertBank {
        experts: projection_spec::Dim::Literal(1_000_001),
        top_k: projection_spec::Dim::Literal(2),
    };
    assert!(
        d.bind(&geometry(), &Registry::default(), &metadata(), &Bf16Rules)
            .unwrap_err()
            .0
            .contains("source instance limit")
    );
}

#[test]
fn header_index_adapter_checks_shapes_offsets_files_duplicates_and_coverage() {
    use std::collections::BTreeMap;
    let header = json!({"a.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"__metadata__":{"format":"pt"}});
    let headers = BTreeMap::from([("weights.safetensors".into(), header.clone())]);
    let index = json!({"weight_map":{"a.weight":"weights.safetensors"}});
    let metadata =
        CheckpointMetadata::from_headers(&headers, Some(&index), json!({"format":"bf16"})).unwrap();
    assert_eq!(metadata.tensors["a.weight"].shape, [2, 3]);
    assert_eq!(
        metadata.tensors["a.weight"].file.as_deref(),
        Some("weights.safetensors")
    );
    assert_eq!(metadata.quantization_config["format"], "bf16");
    CheckpointMetadata::from_headers(&headers, None, json!(null)).unwrap();
    for pointer in [
        "/a.weight/shape",
        "/a.weight/dtype",
        "/a.weight/data_offsets",
    ] {
        let mut changed = header.clone();
        *changed.pointer_mut(pointer).unwrap() = match pointer {
            "/a.weight/shape" => json!([0, 3]),
            "/a.weight/dtype" => json!("BAD"),
            _ => json!([0, 11]),
        };
        let error = CheckpointMetadata::from_headers(
            &BTreeMap::from([("weights.safetensors".into(), changed)]),
            Some(&index),
            json!(null),
        )
        .unwrap_err();
        assert!(error.0.contains("a.weight"), "{error}");
    }
    for index in [
        json!({"weight_map":{"a.weight":"other.safetensors"}}),
        json!({"weight_map":{"a.weight":"weights.safetensors","b":"weights.safetensors"}}),
    ] {
        assert!(CheckpointMetadata::from_headers(&headers, Some(&index), json!(null)).is_err());
    }
    let mut duplicate = headers.clone();
    duplicate.insert("other.safetensors".into(), header);
    assert!(
        CheckpointMetadata::from_headers(&duplicate, None, json!(null))
            .unwrap_err()
            .0
            .contains("multiple headers")
    );
    let overlap = json!({"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}});
    assert!(
        CheckpointMetadata::from_headers(
            &BTreeMap::from([("weights.safetensors".into(), overlap)]),
            None,
            json!(null)
        )
        .unwrap_err()
        .0
        .contains("offsets overlap")
    );
}

struct SharedScaleRules;
impl CompanionRules for SharedScaleRules {
    fn import(
        &self,
        request: &BindingRequest<'_>,
        metadata: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        let mut imported = bf16(request, metadata)?;
        imported.companions.push(TensorRequirement {
            tensor: "shared.scale".into(),
            dtype: "F32".into(),
            shape: vec![1],
            file: None,
        });
        Ok(imported)
    }
}

#[test]
fn shared_companions_are_legal_when_their_roles_agree() {
    let mut metadata = metadata();
    metadata
        .tensors
        .insert("shared.scale".into(), info(&[1], "F32"));
    declaration()
        .bind(
            &geometry(),
            &Registry::default(),
            &metadata,
            &SharedScaleRules,
        )
        .unwrap();
}

struct BlockRules;
impl CompanionRules for BlockRules {
    fn import(
        &self,
        request: &BindingRequest<'_>,
        metadata: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        let mut imported = bf16(request, metadata)?;
        if request.projection.id == "experts" {
            imported.weight.format = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);
            imported.weight.encoding = CodeEncoding::E4M3;
            imported.weight.codes.dtype = DType::Fp8E4M3;
            imported.weight.codes.byte_length /= 2;
            imported.weight.local_scale = Some(ScaleSpec {
                tensor: TensorSpan {
                    tensor: format!("{}.scale", request.tensor),
                    dtype: DType::F32,
                    shape: vec![4, 1, 1],
                    offset: 0,
                    byte_length: 16,
                },
                scope: ScaleScope::Block,
                convention: ScaleConvention::Multiply,
                groups: vec![
                    GroupAxis {
                        axis: 1,
                        size: 128,
                        origin: 0,
                    },
                    GroupAxis {
                        axis: 2,
                        size: 128,
                        origin: 0,
                    },
                ],
            });
        }
        Ok(imported)
    }
}

#[test]
fn uniform_expert_banks_keep_per_expert_scale_grids() {
    let mut metadata = metadata();
    for layer in 0..2 {
        let name = format!("layers.{layer}.experts.weight");
        metadata.tensors.get_mut(&name).unwrap().dtype = "F8_E4M3".into();
        metadata
            .tensors
            .insert(format!("{name}.scale"), info(&[4, 1, 1], "F32"));
    }
    declaration()
        .bind(&geometry(), &Registry::default(), &metadata, &BlockRules)
        .unwrap();
}

struct PackedRules;
impl CompanionRules for PackedRules {
    fn import(
        &self,
        request: &BindingRequest<'_>,
        metadata: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        let mut imported = bf16(request, metadata)?;
        if request.projection.id == "q" {
            use kernel_core::precision::NibbleOrder;
            imported.weight.format = WeightFormat::Nvfp4;
            imported.weight.encoding = CodeEncoding::E2M1(NibbleOrder::EvenLow);
            imported.weight.codes.dtype = DType::U8;
            imported.weight.codes.byte_length /= 2;
            imported.weight.local_scale = Some(ScaleSpec {
                tensor: TensorSpan {
                    tensor: format!("{}.local", request.tensor),
                    dtype: DType::Ue4M3,
                    shape: vec![8, 1],
                    offset: 0,
                    byte_length: 8,
                },
                scope: ScaleScope::Block,
                convention: ScaleConvention::Multiply,
                groups: vec![
                    GroupAxis {
                        axis: 0,
                        size: 1,
                        origin: 0,
                    },
                    GroupAxis {
                        axis: 1,
                        size: 16,
                        origin: 0,
                    },
                ],
            });
            imported.weight.global_scale = Some(ScaleSpec {
                tensor: TensorSpan {
                    tensor: format!("{}.global", request.tensor),
                    dtype: DType::F32,
                    shape: vec![1],
                    offset: 0,
                    byte_length: 4,
                },
                scope: ScaleScope::Projection,
                convention: ScaleConvention::Multiply,
                groups: vec![],
            });
            imported.parts.clear();
            for index in 0..2 {
                let mut part = imported.weight.clone();
                part.logical_shape = vec![4, 16];
                part.codes.shape = vec![4, 8];
                part.codes.offset = index * 32;
                part.codes.byte_length = 32;
                let local = &mut part.local_scale.as_mut().unwrap().tensor;
                local.shape = vec![4, 1];
                local.offset = index * 4;
                local.byte_length = 4;
                imported.parts.push(part);
            }
        }
        Ok(imported)
    }
}

#[test]
fn packed_codes_bind_stored_shapes_and_both_local_and_global_companions() {
    let mut metadata = metadata();
    for layer in 0..2 {
        let name = format!("layers.{layer}.q.weight");
        metadata.tensors.insert(name.clone(), info(&[8, 8], "U8"));
        metadata
            .tensors
            .insert(format!("{name}.local"), info(&[8, 1], "F8_E4M3"));
        metadata
            .tensors
            .insert(format!("{name}.global"), info(&[1], "F32"));
    }
    let bind = |m: &CheckpointMetadata| {
        declaration().bind(&geometry(), &Registry::default(), m, &PackedRules)
    };
    bind(&metadata).unwrap();
    for name in ["layers.0.q.weight.local", "layers.0.q.weight.global"] {
        let mut missing = metadata.clone();
        missing.tensors.remove(name);
        assert!(bind(&missing).unwrap_err().0.contains(name));
    }
    metadata
        .tensors
        .get_mut("layers.0.q.weight.global")
        .unwrap()
        .dtype = "BF16".into();
    assert!(
        bind(&metadata)
            .unwrap_err()
            .0
            .contains("layers.0.q.weight.global")
    );
}

struct IndependentGridRules;
impl CompanionRules for IndependentGridRules {
    fn import(
        &self,
        request: &BindingRequest<'_>,
        metadata: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        let mut imported = bf16(request, metadata)?;
        if request.projection.id == "q" {
            imported.weight.format = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);
            imported.weight.encoding = CodeEncoding::E4M3;
            imported.weight.codes.dtype = DType::Fp8E4M3;
            imported.weight.codes.byte_length /= 2;
            imported.weight.local_scale = Some(ScaleSpec {
                tensor: TensorSpan {
                    tensor: format!("{}.scale", request.tensor),
                    dtype: DType::F32,
                    shape: vec![2, 1],
                    offset: 0,
                    byte_length: 8,
                },
                scope: ScaleScope::Block,
                convention: ScaleConvention::Multiply,
                groups: vec![
                    GroupAxis {
                        axis: 0,
                        size: 128,
                        origin: 0,
                    },
                    GroupAxis {
                        axis: 1,
                        size: 128,
                        origin: 0,
                    },
                ],
            });
            imported.parts.clear();
            for index in 0..2 {
                let mut part = imported.weight.clone();
                part.logical_shape = vec![4, 16];
                part.codes.shape = vec![4, 16];
                part.codes.offset = index * 64;
                part.codes.byte_length = 64;
                let scale = &mut part.local_scale.as_mut().unwrap().tensor;
                scale.shape = vec![1, 1];
                scale.offset = index * 4;
                scale.byte_length = 4;
                imported.parts.push(part);
            }
        }
        Ok(imported)
    }
}

#[test]
fn independently_restarted_scale_grids_bind_as_complete_ordered_views() {
    let mut metadata = metadata();
    for layer in 0..2 {
        let name = format!("layers.{layer}.q.weight");
        metadata.tensors.get_mut(&name).unwrap().dtype = "F8_E4M3".into();
        metadata
            .tensors
            .insert(format!("{name}.scale"), info(&[2, 1], "F32"));
    }
    let d = declaration();
    let bound = d
        .bind(
            &geometry(),
            &Registry::default(),
            &metadata,
            &IndependentGridRules,
        )
        .unwrap();
    assert_eq!(bound.operands[0].part_bindings.len(), 2);
    assert!(bound.operands[0].imported.weight.validate().is_err());
    assert!(
        bound.operands[0]
            .imported
            .parts
            .iter()
            .all(|p| p.validate().is_ok())
    );
    metadata.tensors.remove("layers.1.q.weight.scale");
    assert!(
        d.bind(
            &geometry(),
            &Registry::default(),
            &metadata,
            &IndependentGridRules
        )
        .unwrap_err()
        .0
        .contains("layers.1.q.weight.scale")
    );
}

struct BadGridRules(u8);
impl CompanionRules for BadGridRules {
    fn import(
        &self,
        r: &BindingRequest<'_>,
        m: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        let mut imported = IndependentGridRules.import(r, m)?;
        if !imported.parts.is_empty() {
            match self.0 {
                0 => {
                    imported.parts.pop();
                }
                1 => imported.parts[1].codes.offset -= 1,
                2 => {
                    imported.parts[1]
                        .local_scale
                        .as_mut()
                        .unwrap()
                        .tensor
                        .offset = 0
                }
                3 => imported.parts[0].local_scale.as_mut().unwrap().groups[0].origin = 1,
                _ => unreachable!(),
            }
        }
        Ok(imported)
    }
}

#[test]
fn part_views_refuse_gaps_overlapping_code_ranges_and_wrong_grid_origins() {
    let mut metadata = metadata();
    for layer in 0..2 {
        let name = format!("layers.{layer}.q.weight");
        metadata.tensors.get_mut(&name).unwrap().dtype = "F8_E4M3".into();
        metadata
            .tensors
            .insert(format!("{name}.scale"), info(&[2, 1], "F32"));
    }
    for (scenario, why) in [
        (0, "one declared matrix part"),
        (1, "code range disagrees"),
        (2, "do not cover"),
        (3, "grid origin disagrees"),
    ] {
        let error = declaration()
            .bind(
                &geometry(),
                &Registry::default(),
                &metadata,
                &BadGridRules(scenario),
            )
            .unwrap_err();
        assert!(error.0.contains(why), "{error}");
    }
}

struct PackedOperandRules;
impl CompanionRules for PackedOperandRules {
    fn import(
        &self,
        r: &BindingRequest<'_>,
        m: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        let mut imported = bf16(r, m)?;
        if r.projection.id == "feed" {
            use kernel_core::precision::NibbleOrder;
            imported.weight.format = WeightFormat::Nvfp4;
            imported.weight.encoding = CodeEncoding::E2M1(NibbleOrder::EvenLow);
            imported.weight.codes.dtype = DType::U8;
            imported.weight.codes.byte_length /= 2;
            imported.weight.local_scale = Some(ScaleSpec {
                tensor: TensorSpan {
                    tensor: format!("{}.local", r.tensor),
                    dtype: DType::Ue4M3,
                    shape: vec![24, 1],
                    offset: 0,
                    byte_length: 24,
                },
                scope: ScaleScope::Block,
                convention: ScaleConvention::Multiply,
                groups: vec![
                    GroupAxis {
                        axis: 0,
                        size: 1,
                        origin: 0,
                    },
                    GroupAxis {
                        axis: 1,
                        size: 16,
                        origin: 0,
                    },
                ],
            });
            imported.weight.global_scale = Some(ScaleSpec {
                tensor: TensorSpan {
                    tensor: format!("{}.global", r.tensor),
                    dtype: DType::F32,
                    shape: vec![2],
                    offset: 0,
                    byte_length: 8,
                },
                scope: ScaleScope::Projection,
                convention: ScaleConvention::Multiply,
                groups: vec![],
            });
            for operand in 0..2 {
                let mut part = imported.weight.clone();
                part.logical_shape = vec![12, 16];
                part.codes.shape = vec![12, 8];
                part.codes.offset = operand * 96;
                part.codes.byte_length = 96;
                let local = &mut part.local_scale.as_mut().unwrap().tensor;
                local.shape = vec![12, 1];
                local.offset = operand * 12;
                local.byte_length = 12;
                let global = &mut part.global_scale.as_mut().unwrap().tensor;
                global.shape = vec![1];
                global.offset = operand * 4;
                global.byte_length = 4;
                imported.parts.push(part);
            }
        }
        Ok(imported)
    }
}

#[test]
fn packed_gate_up_retains_independent_global_companions_per_operand() {
    let mut d = declaration();
    d.projections[3].source.as_mut().unwrap().tensor = "{stem}.feed.weight".into();
    let mut m = metadata();
    for layer in 0..2 {
        m.tensors
            .remove(&format!("layers.{layer}.feed.gate.weight"));
        m.tensors.remove(&format!("layers.{layer}.feed.up.weight"));
        let name = format!("layers.{layer}.feed.weight");
        m.tensors.insert(name.clone(), info(&[24, 8], "U8"));
        m.tensors
            .insert(format!("{name}.local"), info(&[24, 1], "F8_E4M3"));
        m.tensors
            .insert(format!("{name}.global"), info(&[2], "F32"));
    }
    let bound = d
        .bind(&geometry(), &Registry::default(), &m, &PackedOperandRules)
        .unwrap();
    let feed = bound
        .operands
        .iter()
        .find(|b| b.projection == "feed")
        .unwrap();
    assert_eq!(
        feed.imported
            .parts
            .iter()
            .map(|p| p.global_scale.as_ref().unwrap().tensor.offset)
            .collect::<Vec<_>>(),
        [0, 4]
    );
    assert_eq!(feed.part_bindings.len(), 2);
}

struct AggregateOnly;
impl CompanionRules for AggregateOnly {
    fn import(
        &self,
        request: &BindingRequest<'_>,
        metadata: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        let mut imported = bf16(request, metadata)?;
        imported.parts.clear();
        Ok(imported)
    }
}

#[test]
fn declared_parts_require_imported_part_views() {
    let error = declaration()
        .bind(
            &geometry(),
            &Registry::default(),
            &metadata(),
            &AggregateOnly,
        )
        .unwrap_err();
    assert!(
        error
            .0
            .contains("declared parts require imported part views"),
        "{error}"
    );
}

struct LookupCompanions {
    local_shared: bool,
    global_shared: bool,
}
impl CompanionRules for LookupCompanions {
    fn import(
        &self,
        request: &BindingRequest<'_>,
        metadata: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        let mut imported = bf16(request, metadata)?;
        if request.projection.id == "table" {
            imported.weight.format = WeightFormat::Nvfp4;
            imported.weight.encoding =
                CodeEncoding::E2M1(kernel_core::precision::NibbleOrder::EvenLow);
            imported.weight.codes.dtype = DType::U8;
            imported.weight.codes.shape = vec![16, 8];
            imported.weight.codes.byte_length = 128;
            let name = |shared, suffix| {
                if shared {
                    format!("table.{suffix}")
                } else {
                    format!("{}.{suffix}", request.tensor)
                }
            };
            imported.weight.local_scale = Some(ScaleSpec {
                tensor: TensorSpan {
                    tensor: name(self.local_shared, "local"),
                    dtype: DType::Ue4M3,
                    shape: vec![16, 1],
                    offset: 0,
                    byte_length: 16,
                },
                scope: ScaleScope::Block,
                convention: ScaleConvention::Multiply,
                groups: vec![
                    GroupAxis {
                        axis: 0,
                        size: 1,
                        origin: 0,
                    },
                    GroupAxis {
                        axis: 1,
                        size: 16,
                        origin: 0,
                    },
                ],
            });
            imported.weight.global_scale = Some(ScaleSpec {
                tensor: TensorSpan {
                    tensor: name(self.global_shared, "global"),
                    dtype: DType::F32,
                    shape: vec![1],
                    offset: 0,
                    byte_length: 4,
                },
                scope: ScaleScope::Projection,
                convention: ScaleConvention::Multiply,
                groups: vec![],
            });
        }
        Ok(imported)
    }
}

#[test]
fn lookup_companion_scopes_enforce_sharing_for_local_and_global_roles() {
    use projection_spec::contracts::*;
    let mut d = declaration();
    d.projections[6].source = None;
    let rows = |start, length| RowRange {
        start: projection_spec::Dim::Literal(start),
        length: projection_spec::Dim::Literal(length),
    };
    d.projections[6].lookup = Some(LookupAssembly {
        companions: Default::default(),
        shards: projection_spec::LookupShards::Explicit(
            (0..2)
                .map(|s| LookupShard {
                    id: format!("s{s}"),
                    source: projection_spec::TensorSource {
                        stem: None,
                        tensor: format!("table.{s}"),
                    },
                    rows: rows(s * 16, 16),
                })
                .collect(),
        ),
        heads: vec![LookupHead {
            id: "all".into(),
            vocabulary: rows(0, 32),
        }],
        padding_owner: PaddingOwner::LastRank {},
    });
    d.projections[6]
        .lookup
        .as_mut()
        .unwrap()
        .companions
        .default
        .global = CompanionScope::Shard;
    for role in ["local", "global"] {
        for scope in [CompanionScope::Shard, CompanionScope::Table] {
            let policy = &mut d.projections[6].lookup.as_mut().unwrap().companions.default;
            if role == "local" {
                policy.local = scope;
            } else {
                policy.global = scope;
            }
            let rules = LookupCompanions {
                local_shared: role == "local" && scope == CompanionScope::Shard,
                global_shared: role == "global" && scope == CompanionScope::Shard,
            };
            let mut m = metadata();
            m.tensors.remove("embed.weight");
            for s in 0..2 {
                m.tensors.insert(format!("table.{s}"), info(&[16, 8], "U8"));
                let local = if rules.local_shared {
                    "table.local".into()
                } else {
                    format!("table.{s}.local")
                };
                let global = if rules.global_shared {
                    "table.global".into()
                } else {
                    format!("table.{s}.global")
                };
                m.tensors.insert(local, info(&[16, 1], "F8_E4M3"));
                m.tensors.insert(global, info(&[1], "F32"));
            }
            let error = d
                .bind(&geometry(), &Registry::default(), &m, &rules)
                .unwrap_err();
            assert!(
                error.0.contains(&format!("lookup {role} companion scope")),
                "{error}"
            );
            // The same concrete sharing becomes valid under its matching scope.
            let valid = if scope == CompanionScope::Shard {
                CompanionScope::Table
            } else {
                CompanionScope::Shard
            };
            let policy = &mut d.projections[6].lookup.as_mut().unwrap().companions.default;
            if role == "local" {
                policy.local = valid;
            } else {
                policy.global = valid;
            }
            d.bind(&geometry(), &Registry::default(), &m, &rules)
                .unwrap();
            d.projections[6].lookup.as_mut().unwrap().companions.default = CompanionScopes {
                local: CompanionScope::Shard,
                global: CompanionScope::Shard,
            };
        }
    }
}

#[test]
fn implicit_consumer_views_keep_the_observed_header_dtype() {
    let mut metadata = metadata();
    metadata
        .tensors
        .get_mut("layers.0.norm.weight")
        .unwrap()
        .dtype = "F32".into();
    let bound = bind(&metadata).unwrap();
    let view = bound
        .side_views
        .iter()
        .find(|v| v.tensor == "layers.0.norm.weight")
        .unwrap();
    assert_eq!(view.view.dtype, DType::F32);
    assert_eq!(view.view.consumer, "q");
    assert_eq!(
        view.view.conversion,
        projection_spec::contracts::ViewConversion::Identity
    );
}

#[test]
fn format_keyed_companion_policy_checks_the_observed_shard_format() {
    use kernel_core::precision::format::FormatId;
    use projection_spec::contracts::*;
    let mut d = declaration();
    d.projections[6].source = None;
    d.projections[6].lookup = Some(LookupAssembly {
        companions: CompanionPolicy {
            formats: std::collections::BTreeMap::from([(
                FormatId::Nvfp4,
                CompanionScopes {
                    local: CompanionScope::Table,
                    global: CompanionScope::Table,
                },
            )]),
            ..Default::default()
        },
        shards: projection_spec::LookupShards::Explicit(
            (0..2)
                .map(|s| LookupShard {
                    id: format!("s{s}"),
                    source: projection_spec::TensorSource {
                        stem: None,
                        tensor: format!("table.{s}"),
                    },
                    rows: RowRange {
                        start: projection_spec::Dim::Literal(s * 16),
                        length: projection_spec::Dim::Literal(16),
                    },
                })
                .collect(),
        ),
        heads: vec![LookupHead {
            id: "all".into(),
            vocabulary: RowRange {
                start: projection_spec::Dim::Literal(0),
                length: projection_spec::Dim::Literal(32),
            },
        }],
        padding_owner: PaddingOwner::LastRank {},
    });
    let mut m = metadata();
    m.tensors.remove("embed.weight");
    for s in 0..2 {
        m.tensors.insert(format!("table.{s}"), info(&[16, 8], "U8"));
    }
    m.tensors
        .insert("table.local".into(), info(&[16, 1], "F8_E4M3"));
    m.tensors.insert("table.global".into(), info(&[1], "F32"));
    let rules = LookupCompanions {
        local_shared: true,
        global_shared: true,
    };
    d.bind(&geometry(), &Registry::default(), &m, &rules)
        .unwrap();
    d.projections[6]
        .lookup
        .as_mut()
        .unwrap()
        .companions
        .formats
        .clear();
    assert!(
        d.bind(&geometry(), &Registry::default(), &m, &rules)
            .unwrap_err()
            .0
            .contains("local companion scope")
    );
}

#[test]
fn materialized_source_names_match_bound_claims_and_instances() {
    use projection_spec::bind::SourceOwner;
    let d = declaration();
    let g = geometry();
    let names = d.source_names(&g).unwrap();
    let bound = bind(&metadata()).unwrap();
    let mut claimed: std::collections::BTreeSet<_> = bound
        .operands
        .iter()
        .map(|o| o.imported.weight.codes.tensor.clone())
        .collect();
    claimed.extend(bound.side_tensors.iter().cloned());
    assert_eq!(
        names
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        claimed
    );
    let feed = d.projections.iter().position(|p| p.id == "feed").unwrap();
    assert_eq!(
        names["layers.1.feed.up.weight"],
        SourceOwner::Projection(feed)
    );
    assert!(matches!(
        names["layers.0.norm.weight"],
        SourceOwner::Side(_)
    ));
    let positions = d.source_positions(&g).unwrap();
    assert!(positions.iter().all(|(name, p)| names[name] == p.owner));
    let up = positions["layers.1.feed.up.weight"];
    assert_eq!((up.layer, up.operand, up.expert), (Some(1), Some(1), None));
    let expert = positions["layers.0.expert.3.out.weight"];
    assert_eq!((expert.layer, expert.expert), (Some(0), Some(3)));
    assert_eq!(positions["hash.1.0.weight"].shard, Some(2));
    assert_eq!(
        (
            positions["embed.weight"].layer,
            positions["embed.weight"].shard
        ),
        (None, None)
    );
    assert_eq!(
        d.instance_sources(&g, "feed", Some(1)).unwrap(),
        ["layers.1.feed.gate.weight", "layers.1.feed.up.weight"]
    );
    assert_eq!(
        d.instance_sources(&g, "expert_out", Some(0)).unwrap(),
        ["layers.0.expert.out.weight"]
    );
    assert_eq!(
        d.instance_sources(&g, "table", None).unwrap(),
        ["embed.weight"]
    );
    let layers = g["layers"];
    assert!(d.instance_sources(&g, "q", Some(layers)).is_err());
    assert!(d.instance_sources(&g, "q", None).is_err());
    assert!(d.instance_sources(&g, "table", Some(0)).is_err());
    assert!(d.instance_sources(&g, "hash", None).is_err());
    assert!(d.instance_sources(&g, "absent", None).is_err());
}

/// A bank's expert index need not be a whole path segment: the bank names
/// its resident tensor without the index and its delimiter.
#[test]
fn expert_placeholders_inside_a_path_segment_name_banks_and_members() {
    let g = geometry();
    let mut d = declaration();
    assert_eq!(
        d.expert_sources(&g, "expert_out", Some(0), 3).unwrap(),
        ["layers.0.expert.3.out.weight"]
    );
    let source = d
        .projections
        .iter_mut()
        .find(|p| p.id == "expert_out")
        .unwrap()
        .source
        .as_mut()
        .unwrap();
    assert!(source.tensor.contains(".expert.{expert}."));
    source.tensor = source
        .tensor
        .replace(".expert.{expert}.", ".expert_{expert}.");
    assert!(
        d.source_names(&g)
            .unwrap()
            .contains_key("layers.0.expert_3.out.weight")
    );
    assert_eq!(
        d.instance_sources(&g, "expert_out", Some(0)).unwrap(),
        ["layers.0.expert.out.weight"]
    );
    assert_eq!(
        d.expert_sources(&g, "expert_out", Some(0), 3).unwrap(),
        ["layers.0.expert_3.out.weight"]
    );
    assert!(d.expert_sources(&g, "table", None, 0).is_err());
}

#[test]
fn default_operation_ids_resolve_to_their_instances() {
    use projection_spec::derive::default_operation_id;
    let d = declaration();
    let feed = d.projections.iter().find(|p| p.id == "feed").unwrap();
    assert_eq!(default_operation_id(feed), "{projection}.{i}.{output}");
    let (p, layer) = d.operation_instance("feed.1.bf16").unwrap();
    assert_eq!((p.id.as_str(), layer), ("feed", Some(1)));
    let table = d.projections.iter().find(|p| p.id == "table").unwrap();
    assert_eq!(default_operation_id(table), "{projection}.{output}");
    let (p, layer) = d.operation_instance("table.f32").unwrap();
    assert_eq!((p.id.as_str(), layer), ("table", None));
    let (p, layer) = d
        .operation_instance("expert_out.0.bf16.alternative")
        .unwrap();
    assert_eq!((p.id.as_str(), layer), ("expert_out", Some(0)));
    assert!(d.operation_instance("feed.bf16").is_err());
    assert!(d.operation_instance("feedback.0.bf16").is_err());
    assert!(d.operation_instance("absent.0.bf16").is_err());
}

#[test]
fn declared_operations_list_every_executed_instance_and_output() {
    let d = declaration();
    let g = geometry();
    let operations = d
        .operations(&g, |p| p.id == "feed" || p.id == "table")
        .unwrap();
    let ids: Vec<_> = operations.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, ["feed.0.bf16", "feed.1.bf16", "table.bf16"]);
    let feed = d.projections.iter().find(|p| p.id == "feed").unwrap();
    assert_eq!(operations[0].1.len(), feed.uses.len());
    assert!(
        operations[0]
            .1
            .iter()
            .zip(&feed.uses)
            .all(|(usage, declared)| usage.phase == declared.phase && usage.graph == declared.graph)
    );
}
