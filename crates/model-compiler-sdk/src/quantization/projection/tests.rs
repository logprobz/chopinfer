use super::*;
use kernel_core::contract::{ActivationFormat, DType, Target, WeightFormat};
use kernel_core::precision::format::{A16Recipe, FormatId, KernelFamily};
use kernel_core::precision::{
    ActivationSpec, ArithmeticSpec, CodeEncoding, DecodeOrder, GroupAxis, Rounding,
    ScaleConvention, ScaleScope, ScaleSpec, TensorSpan, WeightLayout, WeightSpec,
};
use kernel_core::weight_transform::{PackedMatrixSource, PackedOperand, WeightTransform};
use projection_spec::bind::{
    BindingRequest, CheckpointMetadata, CompanionRules, ImportedOperand, TensorInfo,
};
use projection_spec::derive::{
    self, DerivationInput, ImportedProjection, ImportedProjections, InstanceKey, SourceMatrix,
    Topology, hooks::NoVariants,
};
use projection_spec::registry::Registry;

struct Rules(WeightSpec);
impl CompanionRules for Rules {
    fn import(
        &self,
        _: &BindingRequest<'_>,
        _: &CheckpointMetadata,
    ) -> Result<ImportedOperand, projection_spec::Error> {
        Ok(ImportedOperand {
            weight: self.0.clone(),
            parts: vec![],
            companions: vec![],
            calibrations: vec![],
        })
    }
}
fn fixture(scaled: bool) -> (RankDerivation, Vec<Vec<u8>>) {
    let d=projection_spec::parse(r#"{
        "schema":"chopinfer.projection.v1","namespace":"fixture","layer_stems":[],"fusion_groups":[],"side_tensors":[],
        "projections":[{"id":"matrix","source":{"tensor":"matrix.weight"},"shape":[8,16],
        "kind":{"kind":"dense"},"split":{"class":"column","tile":4},
        "epilogues":{"residual_add":false,"bias":false,"gelu":null,"swiglu":null,"output_quantization":null,"weighted_moe_combine":false},
        "numerical_contract":"fixture.numeric.v1","uses":[{"phase":"Prefill","capacity":16,"graph":false},{"phase":"Decode","capacity":1,"graph":true}]}]
    }"#,&Default::default(),&Registry::default()).unwrap();
    let mut payloads: Vec<Vec<u8>> = vec![if scaled {
        (0..128).map(|i| (i % 112) as u8).collect()
    } else {
        (0..128u16).flat_map(u16::to_le_bytes).collect()
    }];
    let mut weight = WeightSpec {
        codes: TensorSpan {
            tensor: "matrix.weight".into(),
            dtype: if scaled { DType::Fp8E4M3 } else { DType::Bf16 },
            shape: vec![8, 16],
            offset: 0,
            byte_length: payloads[0].len() as u64,
        },
        logical_shape: vec![8, 16],
        format: if scaled {
            WeightFormat::Fp8PerChannel
        } else {
            WeightFormat::Bf16
        },
        encoding: if scaled {
            CodeEncoding::E4M3
        } else {
            CodeEncoding::Bf16LittleEndian
        },
        layout: WeightLayout::row_major(),
        local_scale: None,
        global_scale: None,
        transforms: vec![],
    };
    if scaled {
        payloads.push(
            (0..8)
                .flat_map(|i| (i as f32 + 1.0).to_le_bytes())
                .collect(),
        );
        weight.local_scale = Some(ScaleSpec {
            tensor: TensorSpan {
                tensor: "matrix.scale".into(),
                dtype: DType::F32,
                shape: vec![8],
                offset: 0,
                byte_length: 32,
            },
            scope: ScaleScope::Channel,
            convention: ScaleConvention::Multiply,
            groups: vec![GroupAxis {
                axis: 0,
                size: 1,
                origin: 0,
            }],
        });
    }
    let metadata = CheckpointMetadata {
        tensors: std::iter::once(&weight.codes)
            .chain(weight.local_scale.iter().map(|s| &s.tensor))
            .map(|s| {
                (
                    s.tensor.clone(),
                    TensorInfo {
                        dtype: s.dtype.checkpoint_dtype().unwrap().into(),
                        shape: s.shape.clone(),
                        file: None,
                    },
                )
            })
            .collect(),
        ..Default::default()
    };
    let g = Default::default();
    let registry = Registry::default();
    let bound = d
        .bind(&g, &registry, &metadata, &Rules(weight.clone()))
        .unwrap();
    let plugin = registry::by_id(if scaled {
        FormatId::Fp8PerChannel
    } else {
        FormatId::Bf16
    })
    .unwrap();
    let imported = ImportedProjections::from([(
        InstanceKey {
            projection: "matrix".into(),
            layer: None,
        },
        ImportedProjection {
            operation_id: "matrix".into(),
            target: Target::Sm120f,
            family: KernelFamily::Dense,
            format: plugin.id(),
            geometry: derive::GeometryMode::RankLocal,
            weights: vec![weight.clone()],
            activations: vec![ActivationSpec::bf16()],
            arithmetic: ArithmeticSpec {
                weight_operand: DType::Bf16,
                accumulator: DType::F32,
                output: DType::Bf16,
                decode_order: plugin
                    .decode_order(
                        KernelFamily::Dense,
                        ActivationFormat::Bf16,
                        A16Recipe::ScaledWeight,
                    )
                    .unwrap(),
                rounding: Rounding::NearestEven,
                numerical_contract: "fixture.numeric.v1".into(),
            },
            a16_recipe: A16Recipe::ScaledWeight,
            preparation: None,
            matrices: vec![SourceMatrix {
                weight,
                expert: 0,
                operand: 0,
                source_expert: 0,
                target_rows: 0..8,
                source_rows: 0..8,
                target_k: 0..16,
                source_k: 0..16,
            }],
        },
    )]);
    (
        derive::derive(DerivationInput {
            declaration: &d,
            bound: &bound,
            geometry: &g,
            registry: &registry,
            topology: Topology { ranks: 2, rank: 1 },
            imported: &imported,
            variants: &NoVariants,
        })
        .unwrap(),
        payloads,
    )
}

fn execute(record: &WeightRecord, sources: &[&[u8]]) -> Vec<u8> {
    let prepared = record
        .clone()
        .prepare(record.output_bytes().unwrap())
        .unwrap();
    prepared
        .produce_whole(&mut |index, offset, out| {
            out.copy_from_slice(&sources[index][offset as usize..offset as usize + out.len()]);
            Ok(())
        })
        .unwrap()
}

#[test]
fn packed_records_preserve_fields_and_payloads_for_padded_fragments_and_scales() {
    for scaled in [false, true] {
        let (rank, payloads) = fixture(scaled);
        let operation = &rank.operations[0];
        let derived = records(operation).unwrap();
        let matrix = PackedMatrixSource {
            expert: 0,
            projection: 0,
            format: operation.packing.format,
            rows: 8,
            k: 16,
            row_origin: 0,
            k_origin: 0,
            nibble_order: kernel_core::precision::NibbleOrder::EvenLow,
            codes: 0,
            local: scaled.then_some(1),
            local_dtype: scaled.then_some(DType::F32),
            global: None,
            source_expert: 0,
            source_experts: 1,
            rows_kept: std::iter::once(4..8).collect(),
            k_kept: 0..16,
        };
        let expected = WeightRecord {
            transform: WeightTransform::PackedOperand(Box::new(PackedOperand {
                plan: operation.packing.clone(),
                section: PackedSection::Codes,
                decode_order: DecodeOrder::ScaleThenCode,
                matrices: vec![matrix],
            })),
            sources: format::sources(&operation.slices[0].source.weight),
            axis: None,
            ranges: vec![],
            stored_shape: vec![],
        };
        assert_eq!(derived[0].record, expected);
        let inputs = payloads.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let actual = execute(&derived[0].record, &inputs);
        let expected = if scaled {
            payloads[0][64..128].to_vec()
        } else {
            use kernel_core::transform::Transform;
            kernel_core::transform::fragment::FragmentWeightLayout {
                rows: 64,
                k: 16,
                format: kernel_core::transform::fragment::FragmentWeightFormat::Bf16,
            }
            .pack(&{
                let mut rows = payloads[0][128..256].to_vec();
                rows.resize(64 * 16 * 2, 0);
                rows
            })
            .unwrap()
            .bytes
        };
        assert_eq!(actual, expected);
        if scaled {
            assert_eq!(execute(&derived[1].record, &inputs), payloads[1][16..32]);
        }
        assert_eq!(rank_records(&rank).unwrap().len(), derived.len());
    }
}

#[test]
fn legacy_sections_and_provider_records_preserve_existing_transform_identity() {
    let (rank, payloads) = fixture(true);
    let mut operation = rank.operations[0].clone();
    let base = WeightRecord {
        transform: WeightTransform::Copy,
        sources: vec![format::sources(&operation.slices[0].source.weight)[0].clone()],
        axis: Some(0),
        ranges: std::iter::once(4..8).collect(),
        stored_shape: vec![4, 16],
    };
    let record = section_record(
        &operation,
        PackedSection::Codes,
        RecordLayout::Rows,
        base,
        vec!["matrix.weight".into()],
    )
    .unwrap();
    assert_eq!(record.record.transform, WeightTransform::Copy);
    assert_eq!(
        execute(&record.record, &[&payloads[0]]),
        payloads[0][64..128]
    );
    let mut complete = records(&operation).unwrap();
    complete[0] = record;
    operation.variant_records = complete.clone();
    assert_eq!(records(&operation).unwrap(), complete);
    operation.variant_records[0].record.stored_shape = vec![8, 16];
    assert!(
        records(&operation)
            .unwrap_err()
            .to_string()
            .contains("placement")
    );
}

#[test]
fn partial_sources_and_conflicting_resident_records_are_refused() {
    let (mut rank, _) = fixture(false);
    rank.operations[0].slices[0].source.weight.codes.offset = 2;
    assert!(
        records(&rank.operations[0])
            .unwrap_err()
            .to_string()
            .contains("part views")
    );
    let (mut rank, _) = fixture(false);
    rank.operations[1].slices[0].rows = std::iter::once(0..4).collect();
    assert!(
        rank_records(&rank)
            .unwrap_err()
            .to_string()
            .contains("conflicting weight records")
    );
}

#[test]
fn variant_fragments_fill_one_view_and_persist_their_placement() {
    let (mut rank, _) = fixture(false);
    let operation = &mut rank.operations[0];
    let view = operation
        .views
        .iter()
        .find(|v| matches!(v.role, ViewRole::Weight(PackedSection::Codes)))
        .unwrap();
    let bytes = view.bytes / 2;
    let source = format::sources(&operation.slices[0].source.weight)[0].clone();
    operation.variant_records = (0..2)
        .map(|piece| VariantRecord {
            view: view.id.clone(),
            view_offset: piece * bytes,
            section: PackedSection::Codes,
            record: WeightRecord {
                transform: WeightTransform::Copy,
                sources: vec![source.clone()],
                axis: Some(0),
                ranges: std::iter::once(0..8).collect(),
                stored_shape: vec![32, 16],
            },
            tensors: vec!["matrix.weight".into()],
        })
        .collect();
    assert_eq!(records(operation).unwrap().len(), 2);
    let persisted = persistent::records(operation).unwrap();
    assert_eq!(
        persisted.iter().map(|p| p.view_offset).collect::<Vec<_>>(),
        vec![0, bytes]
    );
    assert_eq!(
        serde_json::from_str::<persistent::PersistentRecord>(
            &serde_json::to_string(&persisted[1]).unwrap()
        )
        .unwrap(),
        persisted[1]
    );
    operation.variant_records[1].view_offset = 0;
    assert!(
        records(operation)
            .unwrap_err()
            .to_string()
            .contains("overlap")
    );
    operation.variant_records[1].view_offset = bytes + 1;
    assert!(
        records(operation)
            .unwrap_err()
            .to_string()
            .contains("placement")
    );
    operation.variant_records.pop();
    assert!(records(operation).unwrap_err().to_string().contains("hole"));
}

#[test]
fn prepared_record_descriptors_keep_authenticated_source_offsets() {
    let (rank, payloads) = fixture(false);
    let mut operation = rank.operations[0].clone();
    let baseline = records(&operation).unwrap();
    let expected = execute(&baseline[0].record, &[&payloads[0]]);
    operation.record_sources.clear();
    operation.slices[0].source.weight.codes.offset = 16;
    operation.variant_records = baseline.clone();
    operation.source_preparation = Some(derive::SourcePreparation {
        topology: Topology { ranks: 2, rank: 1 },
        geometry: operation.operation.geometry.unwrap(),
        records: baseline,
    });
    let descriptors = persistent::records(&operation).unwrap();
    let mut storage = vec![0xff; 16];
    storage.extend_from_slice(&payloads[0]);
    let source = &descriptors[0].recipe.sources[0];
    assert_eq!(
        source.offset, 16,
        "preparation must retain its bounded source offset"
    );
    let payload = &storage[source.offset as usize..(source.offset + source.byte_length) as usize];
    assert_eq!(
        execute(&operation.variant_records[0].record, &[payload]),
        expected
    );
    let prepared_rank = RankDerivation {
        operations: vec![operation.clone(), operation.clone()],
        ..Default::default()
    };
    assert_eq!(
        persistent::rank_records(&prepared_rank).unwrap(),
        descriptors
    );
    let mut conflict = prepared_rank.clone();
    conflict.operations[1].slices[0].source.weight.codes.offset += 2;
    assert!(
        persistent::rank_records(&conflict)
            .unwrap_err()
            .to_string()
            .contains("conflicting")
    );
    let mut ambiguous = operation.clone();
    let mut other = ambiguous.slices[0].clone();
    other.source.weight.codes.offset += 2;
    ambiguous.slices.push(other);
    assert!(
        persistent::records(&ambiguous)
            .unwrap_err()
            .to_string()
            .contains("ambiguous")
    );
}

#[test]
fn existing_record_spans_keep_native_scale_dtypes_and_nonzero_offsets() {
    for (format, dtype) in [
        (WeightFormat::Nvfp4, DType::Ue4M3),
        (WeightFormat::Mxfp8, DType::Ue8M0),
    ] {
        let (rank, _) = fixture(true);
        let mut operation = rank.operations[0].clone();
        let packed = format == WeightFormat::Nvfp4;
        let scale_columns = if packed { 2 } else { 1 };
        let weight = &mut operation.slices[0].source.weight;
        weight.format = format;
        weight.logical_shape = vec![8, 32];
        weight.encoding = if packed {
            CodeEncoding::E2M1(kernel_core::precision::NibbleOrder::EvenLow)
        } else {
            CodeEncoding::E4M3
        };
        weight.codes.dtype = if packed { DType::U8 } else { DType::Fp8E4M3 };
        weight.codes.shape = vec![8, if packed { 16 } else { 32 }];
        weight.codes.byte_length = weight.codes.shape.iter().product();
        let scale = weight.local_scale.as_mut().unwrap();
        scale.scope = ScaleScope::Block;
        scale.groups = vec![
            GroupAxis {
                axis: 0,
                size: 1,
                origin: 0,
            },
            GroupAxis {
                axis: 1,
                size: if packed { 16 } else { 32 },
                origin: 0,
            },
        ];
        scale.tensor.dtype = dtype;
        scale.tensor.offset = 64;
        scale.tensor.shape = vec![8, scale_columns];
        scale.tensor.byte_length = 8 * scale_columns;
        weight.global_scale = packed.then(|| ScaleSpec {
            tensor: TensorSpan {
                tensor: "matrix.global".into(),
                dtype: DType::F32,
                shape: vec![],
                offset: 0,
                byte_length: 4,
            },
            scope: ScaleScope::Projection,
            convention: ScaleConvention::Multiply,
            groups: vec![],
        });
        weight.validate().unwrap();
        let sources = format::sources(weight);
        let mut records = vec![];
        for (section, source) in [
            (PackedSection::Codes, sources[0].clone()),
            (PackedSection::LocalScales, sources[1].clone()),
        ] {
            let view = operation
                .views
                .iter_mut()
                .find(|v| v.role == ViewRole::Weight(section))
                .unwrap();
            let (axis, ranges, shape) = if section == PackedSection::Codes {
                (
                    Some(0),
                    std::iter::once(4..8).collect(),
                    vec![4, if packed { 16 } else { 32 }],
                )
            } else {
                (None, vec![], source.shape.clone())
            };
            let record = WeightRecord {
                transform: WeightTransform::Copy,
                sources: vec![source],
                axis,
                ranges,
                stored_shape: shape,
            };
            view.bytes = record.output_bytes().unwrap();
            view.dtype = if section == PackedSection::Codes {
                DType::U8
            } else {
                dtype
            };
            view.shape = vec![view.bytes];
            records.push(VariantRecord {
                view: view.id.clone(),
                view_offset: 0,
                section,
                record,
                tensors: vec![
                    if section == PackedSection::Codes {
                        "matrix.weight"
                    } else {
                        "matrix.scale"
                    }
                    .into(),
                ],
            });
        }
        operation.variant_records = records;
        let descriptor = persistent::records(&operation)
            .unwrap()
            .into_iter()
            .find(|r| r.section == PackedSection::LocalScales)
            .unwrap();
        let sources = descriptor.recipe.sources;
        assert_eq!(sources[0].offset, 64);
        assert_eq!(sources[0].dtype, dtype);
        let mut unknown = operation.clone();
        unknown.variant_records[1].tensors[0] = "unbound.scale".into();
        assert!(
            persistent::records(&unknown)
                .unwrap_err()
                .to_string()
                .contains("authenticated source view")
        );
    }
}

#[test]
fn generic_offset_parts_require_an_explicit_existing_abi_recipe() {
    let (mut rank, _) = fixture(false);
    let operation = &mut rank.operations[0];
    let original = operation.slices[0].clone();
    operation.slices.clear();
    for (source_row, target_row) in [(4, 0), (6, 2)] {
        let mut part = original.clone();
        part.source.weight.logical_shape = vec![2, 16];
        part.source.weight.codes.shape = vec![2, 16];
        part.source.weight.codes.offset = source_row * 32;
        part.source.weight.codes.byte_length = 64;
        part.rows = std::iter::once(0..2).collect();
        part.source.source_rows = 0..2;
        part.source.target_rows = target_row..target_row + 2;
        part.target_rows = std::iter::once(target_row..target_row + 2).collect();
        operation.slices.push(part);
    }
    assert!(
        persistent::records(operation)
            .unwrap_err()
            .to_string()
            .contains("part views")
    );
}

#[test]
fn existing_whole_tensor_records_keep_aggregate_provenance_with_part_slices() {
    let (rank, _) = fixture(true);
    let mut operation = rank.operations[0].clone();
    let whole = records(&operation).unwrap();
    operation.variant_records = whole.clone();
    let weight = &mut operation.slices[0].source.weight;
    weight.logical_shape = vec![4, 16];
    weight.codes.shape = vec![4, 16];
    weight.codes.offset = 64;
    weight.codes.byte_length = 64;
    let scale = weight.local_scale.as_mut().unwrap();
    scale.tensor.shape = vec![4];
    scale.tensor.offset = 16;
    scale.tensor.byte_length = 16;
    let records = persistent::records(&operation).unwrap();
    assert!(
        records
            .iter()
            .all(|r| r.recipe.sources.iter().all(|s| s.offset == 0))
    );
    assert_eq!(records[0].recipe.sources[0].byte_length, 128);
    assert_eq!(records[0].recipe.sources[1].byte_length, 32);
    operation.record_sources.clear();
    assert!(
        persistent::records(&operation)
            .unwrap_err()
            .to_string()
            .contains("authenticated source view")
    );
}

#[test]
fn bound_fragment_policy_preserves_served_encoding_and_exact_bytes() {
    let (rank, payloads) = fixture(false);
    let before = &rank.operations[0];
    let bound = bind_record_policy(
        before.clone(),
        RecordPolicy::Sections(RecordLayout::Fragment),
    )
    .unwrap();
    let actual = records(&bound).unwrap();
    let expected = WeightRecord {
        transform: WeightTransform::Fragment(
            kernel_core::transform::fragment::FragmentWeightFormat::Bf16,
        ),
        sources: format::sources(&before.slices[0].source.weight),
        axis: Some(0),
        ranges: std::iter::once(4..8).collect(),
        stored_shape: vec![64, 16],
    };
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].record, expected);
    assert_eq!(actual[0].view_offset, 0);
    let inputs = payloads.iter().map(Vec::as_slice).collect::<Vec<_>>();
    assert_eq!(
        execute(&actual[0].record, &inputs),
        execute(&records(before).unwrap()[0].record, &inputs)
    );
    assert_eq!(
        persistent::records(&bound).unwrap()[0].recipe.transform,
        kernel_core::weight_transform::FRAGMENT
    );
    assert_eq!(bound.operation, before.operation);
}

#[test]
fn bound_row_policy_emits_code_and_scale_copies_and_refuses_unbound_sources() {
    let (rank, payloads) = fixture(true);
    let bound = bind_record_policy(
        rank.operations[0].clone(),
        RecordPolicy::Sections(RecordLayout::Rows),
    )
    .unwrap();
    let pieces = records(&bound).unwrap();
    assert_eq!(pieces.len(), 2);
    assert_eq!(pieces[0].section, PackedSection::Codes);
    assert_eq!(pieces[0].record.transform, WeightTransform::Copy);
    assert_eq!(pieces[1].record.transform, WeightTransform::CopyScales);
    assert_eq!(
        execute(&pieces[0].record, &[&payloads[0]]),
        payloads[0][64..128]
    );
    assert_eq!(
        execute(&pieces[1].record, &[&payloads[1]]),
        payloads[1][16..32]
    );
    let mut invalid = rank.operations[0].clone();
    invalid.record_sources.clear();
    assert!(
        bind_record_policy(invalid, RecordPolicy::PackedPieces)
            .unwrap_err()
            .to_string()
            .contains("authenticated whole")
    );
    assert!(
        bind_record_policy(bound, RecordPolicy::Packed)
            .unwrap_err()
            .to_string()
            .contains("provider-owned")
    );
}

#[test]
fn packed_piece_policy_keeps_operand_order_offsets_and_resident_identity() {
    let (rank, payloads) = fixture(false);
    let mut op = rank.operations[0].clone();
    let mut up = op.slices[0].clone();
    up.operand = 1;
    up.source.operand = 1;
    up.source.weight.codes.tensor = "up.weight".into();
    op.record_sources.push(up.source.weight.codes.clone());
    op.operation.weights.push(up.source.weight.clone());
    op.operation.activations.push(ActivationSpec::bf16());
    op.slices.push(up);
    op.packing.projections = 2;
    for view in &mut op.views {
        if matches!(view.role, ViewRole::Weight(PackedSection::Codes)) {
            view.bytes *= 2;
            view.shape[0] *= 2;
        }
    }
    let default = records(&op).unwrap();
    assert_eq!(default.len(), 1);
    let bound = bind_record_policy(op.clone(), RecordPolicy::PackedPieces).unwrap();
    let pieces = records(&bound).unwrap();
    assert_eq!(pieces.len(), 2);
    let bytes = pieces[0].record.output_bytes().unwrap();
    assert_eq!(pieces[0].view_offset, 0);
    assert_eq!(pieces[1].view_offset, bytes);
    assert_eq!(pieces[0].tensors, ["matrix.weight"]);
    assert_eq!(pieces[1].tensors, ["up.weight"]);
    let mut result = execute(&pieces[0].record, &[&payloads[0]]);
    result.extend(execute(&pieces[1].record, &[&payloads[0]]));
    assert_eq!(
        result,
        execute(&default[0].record, &[&payloads[0], &payloads[0]])
    );
    let mut bad = op;
    bad.slices[1].operand = 0;
    assert!(bind_record_policy(bad, RecordPolicy::PackedPieces).is_err());
    let mut repeated = rank.clone();
    repeated.operations = vec![bound.clone(), bound.clone()];
    assert_eq!(rank_records(&repeated).unwrap(), pieces);
    repeated.operations[1].variant_records[1].view_offset += 1;
    assert!(rank_records(&repeated).is_err());
}

fn strip_fixture() -> DerivedOperation {
    let (rank, _) = fixture(false);
    let mut op = rank.operations[0].clone();
    op.family = KernelFamily::GroupedExpert;
    op.packing = kernel_core::precision::layout::PackingPlan {
        layout: WeightLayout {
            id: "moe-packed-Absent".into(),
            version: 1,
        },
        format: WeightFormat::Mxfp4,
        experts: 2,
        projections: 2,
        rows: 32,
        k: 256,
        columns: kernel_core::contract::SwigluColumns::ConcatenatedHalves,
        local_scale_dtype: Some(DType::Ue8M0),
    };
    op.logical_geometry = kernel_core::precision::MatrixGeometry { rows: 32, k: 256 };
    op.slices.clear();
    op.record_sources.clear();
    op.views.retain(|v| !matches!(v.role, ViewRole::Weight(_)));
    for (section, dtype, bytes, layout) in [
        (PackedSection::Codes, DType::U8, 16384, "mxfp4_e2m1_rows_v1"),
        (
            PackedSection::LocalScales,
            DType::Ue8M0,
            1024,
            "mxfp4_ue8m0_scale_tiles_16x8_v1",
        ),
    ] {
        op.views.push(derive::RequiredView {
            id: format!("bank.{section:?}"),
            role: ViewRole::Weight(section),
            dtype,
            interpretation: None,
            shape: vec![bytes],
            bytes,
            layout: layout.into(),
        });
    }
    for expert in 0..2 {
        for operand in 0..2 {
            let name = format!("bank.{expert}.{operand}");
            let codes = TensorSpan {
                tensor: format!("{name}.weight"),
                dtype: DType::U8,
                shape: vec![64, 128],
                offset: 0,
                byte_length: 8192,
            };
            let scales = TensorSpan {
                tensor: format!("{name}.scale"),
                dtype: DType::Ue8M0,
                shape: vec![64, 8],
                offset: 0,
                byte_length: 512,
            };
            op.record_sources.extend([codes.clone(), scales.clone()]);
            let weight = WeightSpec {
                codes,
                logical_shape: vec![64, 256],
                format: WeightFormat::Mxfp4,
                encoding: CodeEncoding::E2M1(kernel_core::precision::NibbleOrder::EvenLow),
                layout: WeightLayout::row_major(),
                local_scale: Some(ScaleSpec {
                    tensor: scales,
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
                            size: 32,
                            origin: 0,
                        },
                    ],
                }),
                global_scale: None,
                transforms: vec![],
            };
            op.slices.push(derive::OperandSlice {
                source: SourceMatrix {
                    weight,
                    expert,
                    operand,
                    source_expert: 0,
                    target_rows: 0..64,
                    source_rows: 0..64,
                    target_k: 0..256,
                    source_k: 0..256,
                },
                format: FormatId::Mxfp4,
                rows: std::iter::once(32..64).collect(),
                k: 0..256,
                target_rows: std::iter::once(0..32).collect(),
                target_k: 0..256,
                expert,
                operand,
                geometry: op.logical_geometry,
            });
        }
    }
    op
}

#[test]
fn strip_policy_binds_plugin_layouts_and_exact_expert_major_legacy_records() {
    let op = strip_fixture();
    let bound = bind_record_policy(
        op.clone(),
        RecordPolicy::Sections(RecordLayout::Mxfp4Strips),
    )
    .unwrap();
    let records = records(&bound).unwrap();
    assert_eq!(records.len(), 8);
    for (section, transform, shape, length) in [
        (
            PackedSection::Codes,
            WeightTransform::Mxfp4Moe(kernel_core::weight_transform::Mxfp4Section::Codes),
            vec![32, 128],
            4096,
        ),
        (
            PackedSection::LocalScales,
            WeightTransform::Mxfp4Moe(kernel_core::weight_transform::Mxfp4Section::Scales),
            vec![32, 8],
            256,
        ),
    ] {
        let pieces: Vec<_> = records.iter().filter(|r| r.section == section).collect();
        for (i, piece) in pieces.iter().enumerate() {
            assert_eq!(piece.view_offset, i as u64 * length);
            assert_eq!(piece.record.transform, transform);
            assert_eq!(piece.record.axis, Some(0));
            assert_eq!(
                piece.record.ranges,
                std::iter::once(32..64).collect::<Vec<_>>()
            );
            assert_eq!(piece.record.stored_shape, shape);
            assert_eq!(
                piece.tensors[0],
                format!(
                    "bank.{}.{}.{}",
                    i / 2,
                    i % 2,
                    if section == PackedSection::Codes {
                        "weight"
                    } else {
                        "scale"
                    }
                )
            );
            // Scales stay in their domain, at most 252.
            let source: Vec<_> = (0..piece.record.sources[0].byte_length)
                .map(|n| (n * 37 + (n >> 9)) as u8)
                .map(|b| {
                    if section == PackedSection::Codes {
                        b
                    } else {
                        b % 253
                    }
                })
                .collect();
            let actual = execute(&piece.record, &[&source]);
            let input = &source[(source.len() / 2)..];
            let expected = kernel_core::weight_transform::mxfp4::Mxfp4Strips {
                section: if section == PackedSection::Codes {
                    kernel_core::weight_transform::Mxfp4Section::Codes
                } else {
                    kernel_core::weight_transform::Mxfp4Section::Scales
                },
                rows: 32,
                row_bytes: shape[1] as usize,
            }
            .pack(input)
            .unwrap();
            assert_eq!(actual, expected);
        }
    }
    let layouts: Vec<_> = bound
        .views
        .iter()
        .filter(|v| matches!(v.role, ViewRole::Weight(_)))
        .map(|v| v.layout.as_str())
        .collect();
    assert_eq!(
        layouts,
        [
            "mxfp4_gate_up_moe_fragments_16x64_v1",
            "mxfp4_gate_up_moe_fragment_scales_16x64_v2"
        ]
    );
    assert_eq!(bound.operation, op.operation);
    let mut invalid = op;
    invalid.slices[0].k = 16..256;
    assert!(
        bind_record_policy(invalid, RecordPolicy::Sections(RecordLayout::Mxfp4Strips)).is_err()
    );
}

#[test]
fn packed_piece_scalar_globals_keep_served_copy_gather_and_raw_header_shape() {
    let mut op = strip_fixture();
    op.packing.format = WeightFormat::Nvfp4;
    op.packing.layout.id = "moe-packed-ExpertProjection".into();
    op.packing.local_scale_dtype = Some(DType::Ue4M3);
    op.operation.arithmetic.decode_order = kernel_core::precision::DecodeOrder::ExactScaledOperand;
    op.record_sources.clear();
    for slice in &mut op.slices {
        slice.format = FormatId::Nvfp4;
        let w = &mut slice.source.weight;
        w.format = WeightFormat::Nvfp4;
        let local = w.local_scale.as_mut().unwrap();
        local.tensor.dtype = DType::Ue4M3;
        local.tensor.shape[1] = 16;
        local.tensor.byte_length = 1024;
        local.groups[1].size = 16;
        let global = TensorSpan {
            tensor: format!("{}.global", w.codes.tensor),
            dtype: DType::F32,
            shape: vec![],
            offset: 0,
            byte_length: 4,
        };
        w.global_scale = Some(ScaleSpec {
            tensor: global.clone(),
            scope: ScaleScope::Projection,
            convention: ScaleConvention::Multiply,
            groups: vec![],
        });
        op.record_sources
            .extend([w.codes.clone(), local.tensor.clone(), global]);
    }
    let view = op
        .views
        .iter_mut()
        .find(|v| v.role == ViewRole::Weight(PackedSection::LocalScales))
        .unwrap();
    view.dtype = DType::Ue4M3;
    view.bytes = 2048;
    view.shape = vec![2048];
    let mut global = view.clone();
    global.id = "bank.global".into();
    global.role = ViewRole::Weight(PackedSection::GlobalScales);
    global.dtype = DType::F32;
    global.bytes = 16;
    global.shape = vec![4];
    global.layout = "f32_expert_projection_scales_v1".into();
    op.views.push(global);
    let bound = bind_record_policy(op, RecordPolicy::PackedPieces).unwrap();
    let records = records(&bound).unwrap();
    let globals: Vec<_> = records
        .iter()
        .filter(|r| r.section == PackedSection::GlobalScales)
        .collect();
    assert_eq!(globals.len(), 4);
    for (i, r) in globals.iter().enumerate() {
        assert_eq!(r.view_offset, i as u64 * 4);
        assert_eq!(r.record.transform, WeightTransform::CopyScales);
        assert_eq!(r.record.axis, Some(0));
        assert_eq!(r.record.ranges, vec![0..1]);
        assert_eq!(r.record.stored_shape, vec![1]);
        assert_eq!(r.record.sources[0].shape, Vec::<u64>::new());
        assert_eq!(
            execute(&r.record, &[&1.25_f32.to_le_bytes()]),
            1.25_f32.to_le_bytes()
        );
    }
}
