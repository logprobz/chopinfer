use super::*;
use crate::contract::{MAX_VERIFY_ROWS, PREFILL_TOKENS};
use kernel_core::contract::DType;
use projection_spec::contracts::Interpretation;

#[test]
fn hash_stage_matches_served_lookup_symbols_state_and_phase_addressing() {
    // Both served exports retain per-tensor FP8 n-gram tables. The mixed
    // export serves TP4; the FP8 export serves both declared rank counts.
    for (export, text, ranks) in [
        (
            "fp8",
            include_str!("../../../tests/fixtures/projections/fp8.json"),
            &[2, 4][..],
        ),
        (
            "nvidia",
            include_str!("../../../tests/fixtures/projections/nvidia.json"),
            &[4][..],
        ),
    ] {
        for &tp in ranks {
            let declaration = crate::projections::declaration(tp).unwrap();
            let mut metadata = fixtures::fixture(text);
            fixtures::owner_scope(&declaration, &mut metadata);
            let bound = fixtures::bind(
                &declaration,
                &crate::projections::geometry(tp).unwrap(),
                &metadata,
            )
            .unwrap();
            if export == "nvidia" {
                assert!(
                    bound
                        .operands
                        .iter()
                        .any(|o| o.imported.weight.format == WeightFormat::Nvfp4)
                );
            }
            let table = bound
                .operands
                .iter()
                .find(|o| o.projection == "ngram")
                .unwrap();
            let stage = &declaration
                .projections
                .iter()
                .find(|p| p.id == "ngram")
                .unwrap()
                .composition
                .as_ref()
                .unwrap()
                .pre[0];
            let shape = Shape::of(Ranks::new(tp).unwrap()).unwrap();
            for usage in crate::precision::uses().unwrap() {
                for rank in 0..tp {
                    let descriptor =
                        hash_descriptor(stage, shape, rank, usage, table.imported.weight.format)
                            .unwrap();
                    let current =
                        crate::precision::legacy_lookup_capability(shape, true, usage).unwrap();
                    assert_eq!(
                        descriptor.execution,
                        HashExecution::FusedLookup(current.stages[0].clone()),
                        "{export} TP{tp} rank {rank}"
                    );
                    assert_eq!(
                        (descriptor.table_start, descriptor.table_rows),
                        crate::views::ngram_rows(rank, shape).unwrap()
                    );
                    assert_eq!(descriptor.history_offset_bytes, shape.ple_conv_bytes());
                    assert_eq!(
                        descriptor.history_item_stride_bytes,
                        shape.ple_entry_bytes()
                    );
                    assert_eq!(
                        descriptor.ids_row_stride_bytes,
                        u64::from(shape.ngram_local_heads) * 8
                    );
                    assert_eq!(descriptor.local_heads, shape.ngram_local_heads);
                    assert_eq!(
                        descriptor.addressing,
                        match usage.phase {
                            ExecutionPhase::Prefill => HashAddressing::PrefillContiguous,
                            ExecutionPhase::Decode => HashAddressing::DecodeItemStride,
                            ExecutionPhase::Verify => HashAddressing::VerifyCuRows,
                        }
                    );
                    assert_eq!(
                        descriptor.verify_rows,
                        (usage.phase == ExecutionPhase::Verify).then_some(MAX_VERIFY_ROWS)
                    );
                    let logical = &descriptor.stage;
                    use projection_spec::derive::AdapterValue as Value;
                    let adapter = logical.adapter.as_ref().unwrap();
                    assert_eq!(
                        adapter.execution,
                        projection_spec::derive::StageExecution::Fused {
                            operation: current.stages[0].clone()
                        }
                    );
                    assert_eq!(
                        adapter.facts["table-start"],
                        Value::U64(descriptor.table_start)
                    );
                    assert_eq!(
                        adapter.facts["local-heads"],
                        Value::U64(shape.ngram_local_heads.into())
                    );
                    let Value::Binding(history) = &adapter.facts["history-in"] else {
                        panic!("missing history binding")
                    };
                    assert_eq!(history.offset_bytes, shape.ple_conv_bytes());
                    assert_eq!(history.stride_bytes, Some(shape.ple_entry_bytes()));
                    let Value::Binding(ids) = &adapter.facts["ids"] else {
                        panic!("missing ids binding")
                    };
                    assert_eq!(
                        ids.stride_bytes,
                        Some(u64::from(shape.ngram_local_heads) * 8)
                    );
                    assert_eq!(logical.mutations, ["history"]);
                    assert_eq!(
                        logical
                            .inputs
                            .iter()
                            .map(|v| v.id.as_str())
                            .collect::<Vec<_>>(),
                        ["tokens", "history", "multipliers", "sizes", "offsets"]
                    );
                    assert_eq!(
                        logical
                            .outputs
                            .iter()
                            .map(|v| v.id.as_str())
                            .collect::<Vec<_>>(),
                        ["ids", "history"]
                    );
                    for view in logical.inputs.iter().chain(&logical.outputs) {
                        assert_eq!(view.interpretation, Some(Interpretation::Unsigned));
                    }
                    assert_eq!(logical.inputs[1].dtype, DType::I32);
                    assert_eq!(logical.inputs[1].bytes, 8);
                    assert_eq!(logical.inputs[1].role, ViewRole::State);
                    assert_eq!(logical.outputs[1], logical.inputs[1]);
                    assert_eq!(logical.outputs[0].shape, [u64::from(PREFILL_TOKENS), 16]);
                }
            }
        }
    }
}

#[test]
fn nvfp4_hash_remains_a_separate_stage_and_invalid_adapters_are_refused() {
    let declaration = crate::projections::declaration(4).unwrap();
    let stage = &declaration
        .projections
        .iter()
        .find(|p| p.id == "ngram")
        .unwrap()
        .composition
        .as_ref()
        .unwrap()
        .pre[0];
    let shape = Shape::of(Ranks::TP4).unwrap();
    for usage in crate::precision::uses().unwrap() {
        let descriptor = hash_descriptor(stage, shape, 0, usage, WeightFormat::Nvfp4).unwrap();
        assert_eq!(
            descriptor.execution,
            HashExecution::Standalone(crate::sm120::generated::ngram_hash_symbol(
                shape,
                usage.phase
            ))
        );
        if usage.phase == ExecutionPhase::Verify {
            assert_eq!(descriptor.controls, ["cu_rows", "q_len", "live_items"]);
        }
    }
    let usage = crate::precision::uses().unwrap()[0];
    assert!(hash_descriptor(stage, shape, 4, usage, WeightFormat::Fp8PerTensor).is_err());
    assert!(hash_descriptor(stage, shape, 0, usage, WeightFormat::Mxfp8).is_err());
    let mut corrupt = stage.clone();
    corrupt.mutates.clear();
    assert!(hash_descriptor(&corrupt, shape, 0, usage, WeightFormat::Fp8PerTensor).is_err());
}

#[test]
fn registered_hash_hook_retains_the_descriptor_and_refuses_missing_binding() {
    use kernel_core::precision::format::FormatId;
    use kernel_core::precision::{
        CodeEncoding, ScaleConvention, ScaleScope, ScaleSpec, TensorSpan, WeightLayout, WeightSpec,
    };
    use projection_spec::bind::{BoundDeclaration, BoundOperand, ImportedOperand};
    use projection_spec::derive::{InstanceKey, Topology};
    for tp in [2, 4] {
        let geometry = crate::projections::geometry(tp).unwrap();
        let declaration = crate::projections::declaration(tp)
            .unwrap()
            .expanded_profiles(&geometry)
            .unwrap();
        let projection = declaration
            .projections
            .iter()
            .find(|p| p.id == "ngram")
            .unwrap();
        let stage = &projection.composition.as_ref().unwrap().pre[0];
        let scalar = ScaleSpec {
            tensor: TensorSpan {
                tensor: "table.scale".into(),
                dtype: DType::Bf16,
                shape: vec![],
                offset: 0,
                byte_length: 2,
            },
            scope: ScaleScope::Table,
            convention: ScaleConvention::Multiply,
            groups: vec![],
        };
        let weight = WeightSpec {
            codes: TensorSpan {
                tensor: "table.weight".into(),
                dtype: DType::Fp8E4M3,
                shape: vec![1, 1],
                offset: 0,
                byte_length: 1,
            },
            logical_shape: vec![1, 1],
            format: WeightFormat::Fp8PerTensor,
            encoding: CodeEncoding::E4M3,
            layout: WeightLayout::row_major(),
            local_scale: Some(scalar),
            global_scale: None,
            transforms: vec![],
        };
        weight.validate().unwrap();
        let bound = BoundDeclaration {
            operands: vec![BoundOperand {
                projection: projection.id.clone(),
                layer: Some(1),
                operand: Some(0),
                expert: None,
                shard: Some(0),
                head: None,
                imported: ImportedOperand {
                    weight,
                    parts: vec![],
                    companions: vec![],
                    calibrations: vec![],
                },
                format: FormatId::Fp8PerTensor,
                bindings: vec![],
                part_bindings: vec![],
            }],
            ..Default::default()
        };
        for usage in &projection.uses {
            for rank in 0..tp {
                let context = ProjectionContext {
                    bound: &bound,
                    selections: None,
                    source_instance: InstanceKey {
                        projection: projection.id.clone(),
                        layer: Some(1),
                    },
                    declaration: &declaration,
                    projection,
                    geometry: &geometry,
                    topology: Topology { ranks: tp, rank },
                    layer: Some(1),
                    source_layer: Some(1),
                    usage,
                };
                let actual = Variants.stage(stage, &context).unwrap();
                let shape = Shape::of(Ranks::new(tp).unwrap()).unwrap();
                let expected = hash_descriptor(
                    stage,
                    shape,
                    rank,
                    OperationUse {
                        phase: usage.phase,
                        capacity: usage.capacity.resolve(&geometry).unwrap() as u32,
                        graph: usage.graph,
                    },
                    WeightFormat::Fp8PerTensor,
                )
                .unwrap();
                assert_eq!(actual, expected.stage);
                let empty = BoundDeclaration::default();
                assert!(
                    Variants
                        .stage(
                            stage,
                            &ProjectionContext {
                                bound: &empty,
                                ..context
                            }
                        )
                        .is_err()
                );
            }
        }
    }
}

#[path = "../../../tests/projections_support/mod.rs"]
mod fixtures;
#[path = "prototype.rs"]
mod prototype;

#[test]
fn served_export_implementation_records_match_current_rank_images() {
    use projection_spec::derive::{InstanceKey, ProjectionContext, Topology, ViewRole};
    for (export, text, degrees) in [
        (
            "fp8",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/projections/fp8.json"
            )),
            &[2, 4][..],
        ),
        (
            "nvidia",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/projections/nvidia.json"
            )),
            &[4][..],
        ),
    ] {
        let mut metadata = fixtures::fixture(text);
        let original = crate::projections::declaration(4).unwrap();
        fixtures::owner_scope(&original, &mut metadata);
        for &tp in degrees {
            let geometry = crate::projections::geometry(tp).unwrap();
            let mut declaration = crate::projections::declaration(tp)
                .unwrap()
                .expanded_profiles(&geometry)
                .unwrap();
            // Each projection is compared at its first bound layer.
            let mut metadata = metadata.clone();
            graph::representative_layers(&mut declaration, &mut metadata, &geometry);
            let bound = fixtures::bind(&declaration, &geometry, &metadata).unwrap();
            let shape = Shape::of(Ranks::new(tp).unwrap()).unwrap();
            let mut implementations = 0;
            let mut experts = 0;
            for projection in &declaration.projections {
                // The mixed export is served without the FP8 MTP drafter.
                if export == "nvidia" && projection.id.starts_with("mtp.") {
                    continue;
                }
                let Some(frame) = bound
                    .operands
                    .iter()
                    .find(|b| b.projection == projection.id)
                else {
                    continue;
                };
                let imported = prototype::prototype(projection, frame.layer, &bound, &geometry);
                let Some(variant) = projection
                    .composition
                    .as_ref()
                    .and_then(|c| c.implementation_for(imported.format))
                else {
                    continue;
                };
                let mut scoped = bound.clone();
                scoped
                    .operands
                    .retain(|b| b.projection == projection.id && b.layer == frame.layer);
                let native_catalog = super::legacy::catalog(&imported, shape, &scoped).unwrap();
                let native = crate::placement::ImagePlan::new(&native_catalog, None).unwrap();
                for usage in &projection.uses {
                    for rank in 0..tp {
                        let context = ProjectionContext {
                            bound: &scoped,
                            selections: None,
                            source_instance: InstanceKey {
                                projection: projection.id.clone(),
                                layer: frame.layer,
                            },
                            declaration: &declaration,
                            projection,
                            geometry: &geometry,
                            topology: Topology { ranks: tp, rank },
                            layer: executed_layer(&declaration, &geometry, projection, frame.layer),
                            source_layer: frame.layer,
                            usage,
                        };
                        let derived = Variants
                            .implementation(variant, &context, &imported)
                            .unwrap_or_else(|e| panic!("{export} TP{tp} {}: {e}", projection.id));
                        let index = &native.ranks[rank as usize];
                        for view in derived
                            .views
                            .iter()
                            .filter(|v| matches!(v.role, ViewRole::Weight(_)))
                        {
                            let expected = index.views.iter().find(|v| v.name == view.id).unwrap();
                            assert_eq!(view.bytes, expected.bytes);
                            assert_eq!(view.shape, expected.shape);
                            assert_eq!(view.layout, expected.layout_id);
                        }
                        for record in &derived.variant_records {
                            let expected = native.sources[&record.tensors[0]]
                                .iter()
                                .find(|p| p.rank == rank as usize)
                                .unwrap();
                            let view = &index.views[expected.view];
                            let native_record = crate::weight_plan::fragment_record(
                                &native_catalog.tensors[&record.tensors[0]],
                                expected,
                                &view.layout_id,
                            )
                            .unwrap();
                            assert_eq!(record.record, native_record);
                            assert_eq!(
                                record.record.transform.encode_params().unwrap(),
                                native_record.transform.encode_params().unwrap()
                            );
                            if usage == &projection.uses[0] && record.view_offset == 0 {
                                bytes::assert_bytes(&record.record);
                            }
                            assert_eq!(record.view_offset, expected.offset - view.offset);
                            assert_eq!(
                                record.record.ranges,
                                expected
                                    .slices
                                    .iter()
                                    .map(|(_, r)| r.clone())
                                    .collect::<Vec<_>>()
                            );
                            assert_eq!(record.record.stored_shape, expected.shape);
                            assert_eq!(record.record.output_bytes().unwrap(), expected.bytes);
                        }
                        assert_eq!(derived.operation.weights, imported.weights);
                        let adapter = derived.stages[0].adapter.as_ref().unwrap();
                        let expected = if variant.id.ends_with("legacy-fragment-experts.v1") {
                            let role = if derived.operation.epilogues.swiglu.is_some() {
                                crate::precision::Projection::GateUp
                            } else {
                                crate::precision::Projection::Down
                            };
                            crate::precision::legacy_expert_capability(
                                shape,
                                role,
                                derived.requested,
                            )
                            .unwrap()
                            .stages[0]
                                .clone()
                        } else if projection.id == "linear.output" {
                            crate::precision::DenseRole::Output
                                .legacy_capability(shape, derived.requested)
                                .unwrap()
                                .stages[0]
                                .clone()
                        } else {
                            let suffix = if crate::precision::projection_capacity(derived.requested)
                                .unwrap()
                                <= crate::contract::DECODE_ENTRY_ROWS
                            {
                                "hc_up_decode"
                            } else {
                                "hc_up_prefill"
                            };
                            format!("chop_gemm_{}_{suffix}_v1", shape.geometry.kernel_tag)
                        };
                        assert_eq!(
                            adapter.execution,
                            projection_spec::derive::StageExecution::Fused {
                                operation: expected
                            }
                        );
                        let encoded = serde_json::to_vec(adapter).unwrap();
                        assert_eq!(
                            serde_json::from_slice::<projection_spec::derive::StageAdapter>(
                                &encoded
                            )
                            .unwrap(),
                            *adapter
                        );
                        assert_eq!(
                            derived.operation.arithmetic.decode_order,
                            imported.arithmetic.decode_order
                        );
                    }
                }
                implementations += 1;
                experts += u32::from(variant.id.ends_with("legacy-fragment-experts.v1"));
            }
            assert_eq!(implementations, if export == "fp8" { 11 } else { 4 });
            assert_eq!(experts, if export == "fp8" { 4 } else { 0 });
        }
    }
}

#[path = "bytes.rs"]
mod bytes;

#[test]
fn served_bf16_fusions_match_native_entries_records_and_output_addresses() {
    use projection_spec::derive::{AdapterValue as Value, FusionContext, InstanceKey, Topology};
    for (export, text, degrees) in [
        (
            "fp8",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/projections/fp8.json"
            )),
            &[2, 4][..],
        ),
        (
            "nvidia",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/projections/nvidia.json"
            )),
            &[4][..],
        ),
    ] {
        let mut metadata = fixtures::fixture(text);
        fixtures::owner_scope(&crate::projections::declaration(4).unwrap(), &mut metadata);
        for &tp in degrees {
            let geometry = crate::projections::geometry(tp).unwrap();
            let mut declaration = crate::projections::declaration(tp)
                .unwrap()
                .expanded_profiles(&geometry)
                .unwrap();
            // Each fusion is compared at its first bound layer.
            let mut metadata = metadata.clone();
            graph::representative_layers(&mut declaration, &mut metadata, &geometry);
            let bound = fixtures::bind(&declaration, &geometry, &metadata).unwrap();
            let shape = Shape::of(Ranks::new(tp).unwrap()).unwrap();
            let mut seen = 0;
            for group in &declaration.fusion_groups {
                if export == "nvidia" && group.members[0].starts_with("mtp.") {
                    continue;
                }
                let projection = declaration
                    .projections
                    .iter()
                    .find(|p| p.id == group.members[0])
                    .unwrap();
                let frame = bound
                    .operands
                    .iter()
                    .find(|b| b.projection == projection.id)
                    .unwrap();
                let mut scoped = bound.clone();
                scoped
                    .operands
                    .retain(|b| group.members.contains(&b.projection) && b.layer == frame.layer);
                let imports: Vec<_> = group
                    .members
                    .iter()
                    .map(|id| {
                        let p = declaration
                            .projections
                            .iter()
                            .find(|p| &p.id == id)
                            .unwrap();
                        (p, prototype::prototype(p, frame.layer, &scoped, &geometry))
                    })
                    .collect();
                if imports
                    .iter()
                    .any(|(_, i)| i.format != kernel_core::precision::format::FormatId::Bf16)
                {
                    continue;
                }
                let variant = group
                    .composition
                    .as_ref()
                    .unwrap()
                    .implementation
                    .as_ref()
                    .unwrap();
                let parameters: super::super::InputFusionParameters =
                    super::super::typed(variant).unwrap();
                for usage in &projection.uses {
                    for rank in 0..tp {
                        let context = ProjectionContext {
                            bound: &scoped,
                            selections: None,
                            source_instance: InstanceKey {
                                projection: projection.id.clone(),
                                layer: frame.layer,
                            },
                            declaration: &declaration,
                            projection,
                            geometry: &geometry,
                            topology: Topology { ranks: tp, rank },
                            layer: executed_layer(&declaration, &geometry, projection, frame.layer),
                            source_layer: frame.layer,
                            usage,
                        };
                        let members: Vec<_> = imports
                            .iter()
                            .map(|(p, i)| {
                                let ctx = ProjectionContext {
                                    projection: p,
                                    source_instance: InstanceKey {
                                        projection: p.id.clone(),
                                        layer: frame.layer,
                                    },
                                    ..context.clone()
                                };
                                ctx.ordinary(i).unwrap()
                            })
                            .collect();
                        let fusion = Variants
                            .fusion(
                                variant,
                                &FusionContext {
                                    projection: context,
                                    group,
                                    numerical: None,
                                },
                                &members,
                            )
                            .unwrap_or_else(|e| panic!("{export} TP{tp} {}: {e}", group.id));
                        let requested = members[0].requested;
                        let capacity = crate::precision::projection_capacity(requested).unwrap();
                        let schedule = if capacity <= crate::contract::DECODE_ENTRY_ROWS {
                            vec![parameters.decode_group.clone()]
                        } else {
                            parameters.prefill_groups.clone()
                        };
                        assert_eq!(fusion.stages.len(), schedule.len());
                        assert_eq!(
                            fusion.members,
                            members
                                .iter()
                                .map(|m| m.instance.clone())
                                .collect::<Vec<_>>()
                        );
                        for (stage, ids) in fusion.stages.iter().zip(schedule) {
                            let adapter = stage.adapter.as_ref().unwrap();
                            let symbol = match ids[0].as_str() {
                                "linear.qkv" => crate::precision::DenseRole::LinearQkv
                                    .legacy_capability(shape, requested)
                                    .unwrap()
                                    .stages[0]
                                    .clone(),
                                "linear.z" => crate::precision::DenseRole::LinearZ
                                    .legacy_capability(shape, requested)
                                    .unwrap()
                                    .stages[0]
                                    .clone(),
                                "linear.a" => crate::precision::DenseRole::LinearA
                                    .legacy_capability(shape, requested)
                                    .unwrap()
                                    .stages[0]
                                    .clone(),
                                "full.q" | "mtp.q" => crate::precision::DenseRole::AttentionQ
                                    .legacy_capability(shape, requested)
                                    .unwrap()
                                    .stages[0]
                                    .clone(),
                                "full.index" | "mtp.index" => crate::precision::DenseRole::IndexQk
                                    .legacy_capability(shape, requested)
                                    .unwrap()
                                    .stages[0]
                                    .clone(),
                                id => {
                                    let suffix = if capacity <= crate::contract::DECODE_ENTRY_ROWS {
                                        format!("hc_in_c{capacity}")
                                    } else if id.ends_with("block_inject_weight") {
                                        "hc_inject".into()
                                    } else {
                                        "hc_down_prefill".into()
                                    };
                                    format!("chop_gemm_{}_{suffix}_v1", shape.geometry.kernel_tag)
                                }
                            };
                            assert_eq!(
                                adapter.execution,
                                projection_spec::derive::StageExecution::Fused {
                                    operation: symbol
                                }
                            );
                            assert_eq!(
                                adapter.facts["compiled-capacity"],
                                Value::U64(capacity.into())
                            );
                            let mut offsets = std::collections::BTreeMap::new();
                            for output in group.outputs.iter().filter(|o| ids.contains(&o.member)) {
                                let (offset, stride, columns) = match output.member.as_str() {
                                    "linear.qkv" => (0, shape.linear_qkv, shape.linear_qkv),
                                    "linear.z" => (0, shape.linear_value, shape.linear_value),
                                    "linear.a" => (0, shape.linear_ab_stride, shape.value_heads),
                                    "linear.b" => (
                                        shape.value_heads,
                                        shape.linear_ab_stride,
                                        shape.value_heads,
                                    ),
                                    "full.q" | "mtp.q" => {
                                        (0, shape.attn_row, shape.q_heads * 2 * shape.attn_head())
                                    }
                                    "full.k" | "mtp.k" => (
                                        shape.q_heads * 2 * shape.attn_head(),
                                        shape.attn_row,
                                        shape.attn_head(),
                                    ),
                                    "full.v" | "mtp.v" => (
                                        shape.q_heads * 2 * shape.attn_head() + shape.attn_head(),
                                        shape.attn_row,
                                        shape.attn_head(),
                                    ),
                                    "full.index" | "mtp.index" => {
                                        (0, shape.index_row(), shape.index_row())
                                    }
                                    id if id.ends_with("input_mix_weight_down") => {
                                        (0, shape.hc_rank(), shape.hc_rank())
                                    }
                                    _ => (0, shape.streams(), shape.streams()),
                                };
                                offsets.insert(
                                    output.member.clone(),
                                    Value::List(vec![
                                        Value::Text(output.member.clone()),
                                        Value::Text(output.buffer.clone()),
                                        Value::U64(offset.into()),
                                        Value::U64(stride.into()),
                                        Value::U64(columns.into()),
                                    ]),
                                );
                                let view = stage
                                    .outputs
                                    .iter()
                                    .find(|v| v.id == output.buffer)
                                    .unwrap();
                                assert_eq!(view.shape, [u64::from(capacity), u64::from(stride)]);
                            }
                            let Value::List(actual) = &adapter.facts["outputs"] else {
                                panic!("missing native outputs")
                            };
                            assert_eq!(
                                actual,
                                &group
                                    .outputs
                                    .iter()
                                    .filter(|o| ids.contains(&o.member))
                                    .map(|o| offsets[&o.member].clone())
                                    .collect::<Vec<_>>()
                            );
                            let encoded = serde_json::to_vec(adapter).unwrap();
                            assert_eq!(
                                serde_json::from_slice::<projection_spec::derive::StageAdapter>(
                                    &encoded
                                )
                                .unwrap(),
                                *adapter
                            );
                        }
                        for (operation, member) in fusion.operations.iter().zip(&members) {
                            let mut expected = member.operation.clone();
                            expected.geometry = Some(member.logical_geometry);
                            assert_eq!(operation.operation, expected);
                            let imported = &imports
                                .iter()
                                .find(|(p, _)| p.id == operation.instance.projection)
                                .unwrap()
                                .1;
                            let catalog = super::legacy::catalog(imported, shape, &scoped).unwrap();
                            let native = crate::placement::ImagePlan::new(&catalog, None).unwrap();
                            for record in &operation.variant_records {
                                let piece = native.sources[&record.tensors[0]]
                                    .iter()
                                    .find(|p| p.rank == rank as usize)
                                    .unwrap();
                                let view = &native.ranks[rank as usize].views[piece.view];
                                assert_eq!(
                                    record.record,
                                    crate::weight_plan::fragment_record(
                                        &catalog.tensors[&record.tensors[0]],
                                        piece,
                                        &view.layout_id
                                    )
                                    .unwrap()
                                );
                                assert_eq!(record.view_offset, piece.offset - view.offset);
                                if usage == &projection.uses[0] {
                                    bytes::assert_bytes(&record.record);
                                }
                            }
                        }
                    }
                }
                seen += 1;
            }
            assert_eq!(seen, if export == "fp8" { 7 } else { 4 });
        }
    }
}

#[path = "graph.rs"]
mod graph;

/// The layer that executes a source layer: its stem's execution alias, else
/// the same index.
fn executed_layer(
    declaration: &projection_spec::Declaration,
    geometry: &projection_spec::Geometry,
    projection: &projection_spec::Projection,
    source: Option<u64>,
) -> Option<u64> {
    let source = source?;
    let aliases = projection
        .stem()
        .and_then(|id| declaration.layer_stems.iter().find(|s| s.id == id))
        .and_then(|stem| stem.execution.as_ref())
        .map_or(&[][..], |execution| execution.aliases.as_slice());
    Some(
        aliases
            .iter()
            .find(|a| a.source_index.resolve_offset(geometry).ok() == Some(source))
            .map_or(source, |a| {
                a.execution_index.resolve_offset(geometry).unwrap()
            }),
    )
}
