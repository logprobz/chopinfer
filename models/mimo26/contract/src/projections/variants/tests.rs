use super::*;
use kernel_core::contract::{DType as CodeDType, ScaleMode, WeightFormat};
use kernel_core::precision::{
    CodeEncoding, GroupAxis, ScaleConvention, ScaleScope, ScaleSpec, TensorSpan, WeightLayout,
};

fn source(name: &str, rows: u64, k: u64) -> WeightSpec {
    let scale_rows = rows.div_ceil(128);
    WeightSpec {
        codes: TensorSpan {
            tensor: format!("{name}.weight"),
            dtype: CodeDType::Fp8E4M3,
            shape: vec![4, rows, k],
            offset: 16,
            byte_length: 4 * rows * k,
        },
        logical_shape: vec![4, rows, k],
        format: WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        encoding: CodeEncoding::E4M3,
        layout: WeightLayout::row_major(),
        local_scale: Some(ScaleSpec {
            tensor: TensorSpan {
                tensor: format!("{name}.weight_scale_inv"),
                dtype: CodeDType::F32,
                shape: vec![4, scale_rows, k / 128],
                offset: 32,
                byte_length: 4 * scale_rows * (k / 128) * 4,
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
        }),
        global_scale: None,
        transforms: vec![],
    }
}

#[test]
fn qkv_records_and_scale_bytes_equal_the_retired_loader_for_every_rank() {
    for (projection, stem) in [
        ("ga.qkv", "model.layers.0.self_attn.qkv_proj"),
        ("swa.qkv", "model.layers.1.self_attn.qkv_proj"),
        ("mtp.qkv", "model.mtp.layers.0.self_attn.qkv_proj"),
    ] {
        for tp in Tp::SUPPORTED {
            let declaration = crate::projections::declaration(tp.ranks()).unwrap();
            let variant = &declaration
                .projections
                .iter()
                .find(|p| p.id == projection)
                .unwrap()
                .source_transforms[0];
            let parameters: RegroupParameters =
                serde_json::from_value(variant.parameters.clone()).unwrap();
            let chunk_rows = parameters.chunk_rows(tp).unwrap();
            let original = source(stem, chunk_rows, u64::from(tp.geometry.hidden()));
            let local = original.local_scale.as_ref().unwrap();
            let scale_rows = u64::from(parameters.chunks) * chunk_rows.div_ceil(128);
            let k_blocks = u64::from(tp.geometry.hidden()) / 128;
            let bytes: Vec<u8> = (0..scale_rows * k_blocks)
                .flat_map(|i| (1.0 + (i as f32) / 1024.0).to_le_bytes())
                .collect();
            let global = projection == "ga.qkv";
            let grid = crate::precision::qkv_scale_grid(
                tp.geometry,
                global,
                scale_rows,
                k_blocks as usize,
            )
            .unwrap();
            for rank in 0..tp.count() {
                let scope = format!("loader-qkv {projection} tp{} rank{rank}", tp.ranks());
                let mut lines = retired::Lines::default();
                let derived = qkv(&original, &parameters, tp, rank).unwrap();
                assert_eq!(
                    derived.weight.logical_shape,
                    [tp.qkv_rows(global), u64::from(tp.geometry.hidden())]
                );
                assert_eq!(
                    derived.weight.format,
                    if tp.ranks() == 4 {
                        original.format
                    } else {
                        WeightFormat::Fp8Rows64x128(ScaleMode::ExactF32)
                    }
                );
                let scale = derived.weight.local_scale.as_ref().unwrap();
                assert!(scale.groups.iter().all(|g| g.origin == 0));
                assert_eq!(scale.groups[0].size, if tp.ranks() == 4 { 128 } else { 64 });
                assert_eq!(
                    scale.tensor.offset,
                    if tp.ranks() == 4 {
                        original.local_scale.as_ref().unwrap().tensor.offset
                            + rank as u64
                                * derived
                                    .weight
                                    .local_scale
                                    .as_ref()
                                    .unwrap()
                                    .tensor
                                    .byte_length
                    } else {
                        0
                    }
                );
                assert_eq!(
                    scale.tensor.tensor,
                    if tp.ranks() == 4 {
                        local.tensor.tensor.clone()
                    } else {
                        format!("{}@tp2", local.tensor.tensor)
                    }
                );
                let derived_record = &derived.records[1];
                lines.add(
                    &scope,
                    format!(
                        "tensors={:?} {}",
                        derived_record.tensors,
                        retired::record(&derived_record.record)
                    ),
                );
                let prepared = derived_record
                    .record
                    .clone()
                    .prepare(derived_record.record.output_bytes().unwrap())
                    .unwrap();
                let actual = prepared
                    .produce_whole(&mut |index, offset, out| {
                        assert_eq!(index, 0);
                        out.copy_from_slice(&bytes[offset as usize..offset as usize + out.len()]);
                        Ok(())
                    })
                    .unwrap();
                lines.add(&scope, format!("bytes={}", retired::bytes(&actual)));
                lines.check(&scope);
                let group_rows = if tp.ranks() == 4 { 128 } else { 64 };
                for (row, source_row) in derived
                    .source_rows
                    .iter()
                    .flat_map(|r| r.clone())
                    .enumerate()
                {
                    let scale_row = grid.scale_row(source_row as usize).unwrap();
                    let start = row / group_rows * k_blocks as usize * 4;
                    assert_eq!(
                        &actual[start..start + k_blocks as usize * 4],
                        &bytes[scale_row * k_blocks as usize * 4
                            ..(scale_row + 1) * k_blocks as usize * 4]
                    );
                }
            }
        }
    }
}

#[test]
fn registered_stage_hooks_preserve_ports_and_native_control_contracts() {
    use kernel_core::precision::{ExecutionPhase, OperationUse};
    use projection_spec::bind::BoundDeclaration;
    use projection_spec::derive::hooks::VariantDeriver;
    use projection_spec::derive::{InstanceKey, ProjectionContext, Topology};
    for tp in Tp::SUPPORTED {
        let geometry = crate::projections::geometry(tp.ranks()).unwrap();
        let declaration = crate::projections::declaration(tp.ranks())
            .unwrap()
            .expanded_profiles(&geometry)
            .unwrap();
        let bound = BoundDeclaration::default();
        let mut seen = std::collections::BTreeSet::new();
        for projection in &declaration.projections {
            let Some(composition) = &projection.composition else {
                continue;
            };
            for stage in composition.pre.iter().chain(&composition.post) {
                for usage in &projection.uses {
                    if !stage.phases.contains(&usage.phase) {
                        continue;
                    }
                    for rank in 0..tp.ranks() {
                        let capacity = usage.capacity.resolve(&geometry).unwrap() as u32;
                        let requested = OperationUse {
                            phase: usage.phase,
                            capacity,
                            graph: usage.graph,
                        };
                        let descriptor = stage_descriptor(stage, tp, requested).unwrap();
                        let context = ProjectionContext {
                            bound: &bound,
                            selections: None,
                            source_instance: InstanceKey {
                                projection: projection.id.clone(),
                                layer: Some(0),
                            },
                            declaration: &declaration,
                            projection,
                            geometry: &geometry,
                            topology: Topology {
                                ranks: tp.ranks(),
                                rank,
                            },
                            layer: Some(0),
                            source_layer: Some(0),
                            usage,
                        };
                        assert_eq!(
                            Variants::FLASH.stage(stage, &context).unwrap(),
                            descriptor.stage
                        );
                        let adapter = descriptor.stage.adapter.as_ref().unwrap();
                        let encoded = serde_json::to_vec(adapter).unwrap();
                        assert_eq!(
                            serde_json::from_slice::<projection_spec::derive::StageAdapter>(
                                &encoded
                            )
                            .unwrap(),
                            *adapter
                        );
                        seen.insert(stage.contract.id.as_str());
                        assert_eq!(
                            descriptor.stage.numerical_contract,
                            stage.contract.numerical_contract
                        );
                        assert_eq!(descriptor.stage.mutations, stage.mutates);
                        for (views, ports) in [
                            (&descriptor.stage.inputs, &stage.contract.inputs),
                            (&descriptor.stage.outputs, &stage.contract.outputs),
                        ] {
                            for (view, port) in views.iter().zip(ports) {
                                assert_eq!(view.id, port.binding);
                                assert_eq!(view.dtype, port.dtype);
                                assert_eq!(
                                    view.shape,
                                    port.shape
                                        .iter()
                                        .map(|d| d.resolve(&geometry).unwrap())
                                        .collect::<Vec<_>>()
                                );
                                assert_eq!(
                                    view.bytes,
                                    view.shape.iter().product::<u64>() * dtype_bits(view.dtype) / 8
                                );
                            }
                        }
                        match descriptor.behavior {
                            StageBehavior::Routing { fused, scale_bits } => {
                                assert_eq!(
                                    fused,
                                    usage.phase != ExecutionPhase::Prefill
                                        && crate::sm120::moe::prefer_fused_router(capacity)
                                );
                                assert_eq!(scale_bits, 1.0_f32.to_bits());
                                use projection_spec::derive::AdapterValue as Value;
                                let wide = crate::serving::prefill_route(usage.phase, capacity);
                                let expected = if fused {
                                    format!(
                                        "{}_t{capacity}_v1",
                                        crate::sm120::moe::FUSED_ROUTER_PREFIX
                                    )
                                } else if wide {
                                    format!(
                                        "chop_row_router_{}_sigmoid{}_v1",
                                        tp.geometry.kernel_tag,
                                        tp.geometry.experts()
                                    )
                                } else {
                                    crate::sm120::moe::DECODE_ROUTE_SYMBOL.into()
                                };
                                assert_eq!(adapter.facts["symbol"], Value::Text(expected));
                                assert_eq!(
                                    adapter.facts["scale"],
                                    Value::F32Bits(1.0_f32.to_bits())
                                );
                                let Value::List(permutation) = &adapter.facts["permutation"] else {
                                    panic!("missing permutation metadata")
                                };
                                assert_eq!(permutation.len(), if wide { 4 } else { 0 });
                            }
                            StageBehavior::Combine { parameters } => {
                                assert_eq!(
                                    parameters,
                                    serde_json::json!({"route_order":"ascending-route", "rank_order":"ascending-rank", "route_rounding":"bf16-before-combine", "residual_rounding":"once-after-sum"})
                                );
                            }
                            StageBehavior::Mask {
                                source,
                                view,
                                width,
                                controls,
                            } => {
                                assert_eq!(source, "mask_embedding.pt");
                                assert_eq!(view, "mask_embedding.weight");
                                assert_eq!(width as usize * 2, crate::drafters::MASK_BYTES);
                                assert_eq!(usage.phase, ExecutionPhase::Verify);
                                assert_eq!(
                                    adapter.execution,
                                    projection_spec::derive::StageExecution::Standalone {
                                        symbol: crate::sm120::bundle_symbol(tp, "dflash_mask_rows")
                                    }
                                );
                                assert_eq!(
                                    controls,
                                    ["positions", "prefix_len", "token_batch", "live_tokens"]
                                );
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(seen.len(), 3);
    }
}

#[test]
fn qkv_refuses_changed_scale_origin_convention_and_topology() {
    let declaration = crate::projections::declaration(2).unwrap();
    let variant = &declaration
        .projections
        .iter()
        .find(|p| p.id == "ga.qkv")
        .unwrap()
        .source_transforms[0];
    let parameters: RegroupParameters = serde_json::from_value(variant.parameters.clone()).unwrap();
    let original = source(
        "model.layers.0.self_attn.qkv_proj",
        parameters.chunk_rows(Tp::TP2).unwrap(),
        4096,
    );
    assert!(qkv(&original, &parameters, Tp::TP2, 2).is_err());
    let mut changed = original.clone();
    changed.local_scale.as_mut().unwrap().groups[0].origin = 64;
    changed.validate().unwrap();
    assert!(qkv(&changed, &parameters, Tp::TP2, 0).is_err());
    let mut changed = original;
    changed.local_scale.as_mut().unwrap().convention = ScaleConvention::Reciprocal;
    changed.validate().unwrap();
    assert!(qkv(&changed, &parameters, Tp::TP2, 0).is_err());
}

#[path = "../../../tests/projections_support/mod.rs"]
mod fixtures;
#[path = "prototype.rs"]
mod prototype;
#[path = "../../../tests/projections_support/retired.rs"]
mod retired;

#[test]
fn mopd_transform_hooks_and_dense_records_match_the_retired_loader_at_both_degrees() {
    use projection_spec::derive::hooks::VariantDeriver;
    use projection_spec::derive::{
        InstanceKey, ProjectionContext, SourceTransformResult, Topology, ViewRole,
    };
    let mut metadata = fixtures::fixture(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/projections/mopd.json"
    )));
    fixtures::owner_scope(&mut metadata);
    for tp in Tp::SUPPORTED {
        let geometry = crate::projections::geometry(tp.ranks()).unwrap();
        let declaration = crate::projections::declaration(tp.ranks())
            .unwrap()
            .expanded_profiles(&geometry)
            .unwrap();
        let bound = fixtures::bind(&declaration, &geometry, &metadata).unwrap();
        let mut transforms = 0;
        let mut implementations = 0;
        let scope = format!("hooks tp{}", tp.ranks());
        let mut lines = retired::Lines::default();
        for projection in &declaration.projections {
            let Some(frame) = bound
                .operands
                .iter()
                .find(|b| b.projection == projection.id)
            else {
                continue;
            };
            let imported = prototype::prototype(projection, frame.layer, &bound, &geometry);
            let usage = &projection.uses[0];
            for rank in 0..tp.ranks() {
                let context = ProjectionContext {
                    bound: &bound,
                    selections: None,
                    source_instance: InstanceKey {
                        projection: projection.id.clone(),
                        layer: frame.layer,
                    },
                    declaration: &declaration,
                    projection,
                    geometry: &geometry,
                    topology: Topology {
                        ranks: tp.ranks(),
                        rank,
                    },
                    layer: frame.layer,
                    source_layer: frame.layer,
                    usage,
                };
                for variant in &projection.source_transforms {
                    let result = Variants::FLASH
                        .transform(variant, None, &context, imported.clone())
                        .unwrap();
                    match result {
                        SourceTransformResult::Bypass => assert_eq!(tp.ranks(), 4),
                        SourceTransformResult::Applied(result) => {
                            assert_eq!(tp.ranks(), 2);
                            assert_eq!(result.matrices, imported.matrices);
                            let preparation = result.preparation.unwrap();
                            assert_eq!(preparation.topology, context.topology);
                            assert_eq!(
                                result.weights[0].logical_shape,
                                [preparation.geometry.rows, preparation.geometry.k]
                            );
                            for record in preparation.records {
                                lines.add(
                                    format!("{scope} rank{rank}"),
                                    format!(
                                        "prepared {} {:?} {:?} {}",
                                        projection.id,
                                        record.tensors,
                                        record.section,
                                        retired::record(&record.record)
                                    ),
                                );
                            }
                        }
                    }
                    transforms += 1;
                }
                if let Some(variant) = projection
                    .composition
                    .as_ref()
                    .and_then(|c| c.implementation_for(imported.format))
                {
                    let actual = Variants::FLASH
                        .implementation(variant, &context, &imported)
                        .unwrap();
                    assert_eq!(actual.operation.weights, imported.weights);
                    assert_eq!(actual.operation.arithmetic, imported.arithmetic);
                    assert!(
                        actual
                            .views
                            .iter()
                            .any(|v| matches!(v.role, ViewRole::Weight(_)))
                    );
                    for record in &actual.variant_records {
                        lines.add(
                            format!("{scope} rank{rank}"),
                            format!(
                                "variant {} {:?} {:?} {}",
                                projection.id,
                                record.tensors,
                                record.section,
                                retired::record(&record.record)
                            ),
                        );
                        bytes::assert_bytes(&record.record);
                    }
                    for usage in &projection.uses {
                        let next = Variants::FLASH
                            .implementation(
                                variant,
                                &ProjectionContext {
                                    usage,
                                    ..context.clone()
                                },
                                &imported,
                            )
                            .unwrap();
                        assert_eq!(next.variant_records, actual.variant_records);
                        let adapter = next.stages[0].adapter.as_ref().unwrap();
                        let capacity = if next.requested.phase
                            == kernel_core::precision::ExecutionPhase::Prefill
                            || next.requested.capacity > crate::serving::DECODE_ENTRY_ROWS
                        {
                            crate::serving::PREFILL_TOKENS
                        } else {
                            next.requested.capacity
                        };
                        let symbol = if projection.id.starts_with("dflash.") {
                            let bucket = if capacity == crate::serving::PREFILL_TOKENS {
                                "prefill".into()
                            } else {
                                format!("m{capacity}")
                            };
                            format!(
                                "chop_gemm_{}_dflash_gate_up_{bucket}_v1",
                                crate::topology::kernel_tag(tp)
                            )
                        } else {
                            crate::capabilities::sm120::symbol(
                                crate::capabilities::Projection::MlpGateUp,
                                tp,
                                next.requested,
                            )
                        };
                        assert_eq!(
                            adapter.execution,
                            projection_spec::derive::StageExecution::Fused { operation: symbol }
                        );
                        assert_eq!(
                            adapter.facts["compiled-capacity"],
                            projection_spec::derive::AdapterValue::U64(capacity.into())
                        );
                    }
                    implementations += 1;
                }
            }
        }
        assert_eq!(transforms, 3 * tp.ranks());
        assert_eq!(implementations, 3 * tp.ranks());
        lines.check(&scope);
    }
}

#[path = "bytes.rs"]
mod bytes;

#[test]
fn served_mxfp4_experts_select_the_ordinary_path_without_fp8_registrations() {
    let catalog = crate::sm120::moe::catalog();
    assert!(!catalog.is_empty());
    assert!(
        catalog
            .iter()
            .all(|v| v.query.weight == WeightFormat::Mxfp4)
    );
    let mut metadata = fixtures::fixture(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/projections/mopd.json"
    )));
    fixtures::owner_scope(&mut metadata);
    for tp in Tp::SUPPORTED {
        let geometry = crate::projections::geometry(tp.ranks()).unwrap();
        let declaration = crate::projections::declaration(tp.ranks())
            .unwrap()
            .expanded_profiles(&geometry)
            .unwrap();
        let bound = fixtures::bind(&declaration, &geometry, &metadata).unwrap();
        for projection in declaration
            .projections
            .iter()
            .filter(|p| matches!(p.kind, projection_spec::ProjectionKind::ExpertBank { .. }))
        {
            let frame = bound
                .operands
                .iter()
                .find(|b| b.projection == projection.id)
                .unwrap();
            let imported = prototype::prototype(projection, frame.layer, &bound, &geometry);
            assert_eq!(
                imported.format,
                kernel_core::precision::format::FormatId::Mxfp4
            );
            assert!(
                projection
                    .composition
                    .as_ref()
                    .unwrap()
                    .implementation_for(imported.format)
                    .is_none()
            );
            let composition = projection.composition.as_ref().unwrap();
            assert!(composition.implementations.is_empty());
            assert!(composition.implementation.is_none());
            for format in [
                kernel_core::precision::format::FormatId::Fp8Block128,
                kernel_core::precision::format::FormatId::Fp8Rows64,
            ] {
                assert!(composition.implementation_for(format).is_none());
            }
        }
    }
}

#[path = "graph.rs"]
mod graph;

#[test]
fn review_router_wide_uses_match_served_prefill_routing() {
    use projection_spec::derive::{AdapterValue, StageExecution};
    for tp in crate::Tp::SUPPORTED {
        let declaration = crate::projections::declaration(tp.ranks())
            .unwrap()
            .expanded_profiles(&crate::projections::geometry(tp.ranks()).unwrap())
            .unwrap();
        let router = declaration
            .projections
            .iter()
            .find(|p| p.id == "router")
            .unwrap();
        let stage = &router.composition.as_ref().unwrap().post[0];
        for usage in crate::capabilities::uses()
            .into_iter()
            .filter(|u| crate::serving::prefill_route(u.phase, u.capacity))
        {
            let descriptor = stage_descriptor(stage, tp, usage).unwrap();
            let adapter = descriptor.stage.adapter.unwrap();
            assert_eq!(
                adapter.execution,
                StageExecution::Standalone {
                    symbol: format!(
                        "chop_row_router_{}_sigmoid{}_v1",
                        tp.geometry.kernel_tag,
                        tp.geometry.experts()
                    )
                }
            );
            let AdapterValue::List(permutation) = &adapter.facts["permutation"] else {
                panic!("missing permutation stages")
            };
            let expected: Vec<_> = ["count", "prefix", "scan", "scatter"]
                .into_iter()
                .map(|name| {
                    AdapterValue::Text(format!(
                        "chop_row_permute_tile_{name}_{}_b64_v1",
                        tp.geometry.kernel_tag
                    ))
                })
                .collect();
            assert_eq!(permutation, &expected, "TP{} {:?}", tp.ranks(), usage);
        }
    }
}

#[test]
fn review_tp4_qkv_addresses_follow_the_requested_rank() {
    let declaration = crate::projections::declaration(4).unwrap();
    let projection = declaration
        .projections
        .iter()
        .find(|p| p.id == "swa.qkv")
        .unwrap();
    let parameters: RegroupParameters =
        serde_json::from_value(projection.source_transforms[0].parameters.clone()).unwrap();
    let chunk_rows = parameters.chunk_rows(crate::Tp::TP4).unwrap();
    let original = source("model.layers.1.self_attn.qkv_proj", chunk_rows, 4096);
    let local = original.local_scale.as_ref().unwrap();
    for rank in 0..4 {
        let prepared = qkv(&original, &parameters, crate::Tp::TP4, rank).unwrap();
        assert_eq!(
            prepared.weight.codes.offset,
            original.codes.offset + rank as u64 * chunk_rows * 4096
        );
        assert_eq!(
            prepared.weight.local_scale.as_ref().unwrap().tensor.offset,
            local.tensor.offset + rank as u64 * chunk_rows.div_ceil(128) * 32 * 4
        );
    }
}

#[test]
fn review_prepared_qkv_views_keep_the_native_rank_index_names() {
    use projection_spec::derive::{Topology, ViewRole};
    let mut metadata = fixtures::fixture(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/projections/mopd.json"
    )));
    fixtures::owner_scope(&mut metadata);
    let geometry = crate::projections::geometry(2).unwrap();
    let declaration = crate::projections::declaration(2)
        .unwrap()
        .expanded_profiles(&geometry)
        .unwrap();
    let bound = fixtures::bind(&declaration, &geometry, &metadata).unwrap();
    use projection_spec::derive::hooks::VariantDeriver;
    use projection_spec::derive::{InstanceKey, ProjectionContext, SourceTransformResult};
    let projection = declaration
        .projections
        .iter()
        .find(|p| p.id == "swa.qkv")
        .unwrap();
    let imported = prototype::prototype(projection, Some(1), &bound, &geometry);
    let context = ProjectionContext {
        declaration: &declaration,
        bound: &bound,
        geometry: &geometry,
        projection,
        topology: Topology { ranks: 2, rank: 1 },
        layer: Some(1),
        source_layer: Some(1),
        source_instance: InstanceKey {
            projection: projection.id.clone(),
            layer: Some(1),
        },
        selections: None,
        usage: &projection.uses[0],
    };
    let SourceTransformResult::Applied(imported) = Variants::FLASH
        .transform(&projection.source_transforms[0], None, &context, imported)
        .unwrap()
    else {
        panic!("TP2 must regroup")
    };
    let operation = context.ordinary(&imported).unwrap();
    let names: Vec<_> = operation
        .views
        .iter()
        .filter(|v| matches!(v.role, ViewRole::Weight(_)))
        .map(|v| v.id.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "model.layers.1.self_attn.qkv_proj.weight",
            "model.layers.1.self_attn.qkv_proj.weight_scale_inv"
        ]
    );
}
