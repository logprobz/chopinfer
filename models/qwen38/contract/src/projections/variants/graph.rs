use super::{MIXED, fixtures, prototype};
use projection_spec::derive::{DerivationInput, ImportedProjections, InstanceKey, Topology};
use projection_spec::{Declaration, Dim, Geometry, LayerIndices};
use std::collections::{BTreeMap, BTreeSet};

struct NativeRecords {
    views: BTreeMap<String, (Vec<u64>, u64, String)>,
    records: BTreeMap<String, (kernel_core::weight_transform::WeightRecord, u64)>,
}

fn native_records(
    imported: &projection_spec::derive::ImportedProjection,
    shape: crate::contract::Shape,
    bound: &projection_spec::bind::BoundDeclaration,
    rank: u32,
) -> NativeRecords {
    let catalog = super::super::legacy::catalog(imported, shape, bound).unwrap();
    let native = crate::placement::ImagePlan::new(&catalog, None).unwrap();
    let index = &native.ranks[rank as usize];
    NativeRecords {
        views: index
            .views
            .iter()
            .map(|v| {
                (
                    v.name.clone(),
                    (v.shape.clone(), v.bytes, v.layout_id.clone()),
                )
            })
            .collect(),
        records: native
            .sources
            .iter()
            .filter_map(|(name, pieces)| {
                let piece = pieces.iter().find(|p| p.rank == rank as usize)?;
                let view = &index.views[piece.view];
                Some((
                    name.clone(),
                    (
                        crate::weight_plan::fragment_record(
                            &catalog.tensors[name],
                            piece,
                            &view.layout_id,
                        )
                        .unwrap(),
                        piece.offset - view.offset,
                    ),
                ))
            })
            .collect(),
    }
}

fn imports(
    d: &Declaration,
    b: &projection_spec::bind::BoundDeclaration,
    g: &Geometry,
) -> ImportedProjections {
    let mut result = BTreeMap::new();
    for frame in &b.operands {
        let key = InstanceKey {
            projection: frame.projection.clone(),
            layer: frame.layer,
        };
        if result.contains_key(&key) {
            continue;
        }
        let p = d
            .projections
            .iter()
            .find(|p| p.id == frame.projection)
            .unwrap();
        result.insert(key, prototype::prototype(p, frame.layer, b, g));
    }
    for p in d.projections.iter().filter(|p| !p.assembly.is_empty()) {
        let source = &p.assembly[0];
        let candidate = result
            .iter()
            .find(|(k, _)| k.projection == source.projection)
            .unwrap()
            .1
            .clone();
        let layers = p
            .stem()
            .map(|id| {
                d.layer_stems
                    .iter()
                    .find(|s| s.id == id)
                    .unwrap()
                    .indices
                    .resolve(g)
                    .unwrap()
            })
            .unwrap_or_default();
        for layer in if layers.is_empty() {
            vec![None]
        } else {
            layers.into_iter().map(Some).collect()
        } {
            let mut own = candidate.clone();
            own.operation_id = p.id.clone();
            own.arithmetic.numerical_contract = p.numerical_contract.clone();
            result
                .entry(InstanceKey {
                    projection: p.id.clone(),
                    layer,
                })
                .or_insert(own);
        }
    }
    result
}

/// Layers a served-graph comparison derives. Repeated layers of one stem share
/// their projection, operand, use and fusion declarations, so the default keeps
/// each stem's first, middle and last layer; `Every` is the opt-in full sweep.
#[derive(Clone, Copy, PartialEq)]
enum Layers {
    Representative,
    Every,
}

/// Keep the first, middle and last layer of every projection stem with more,
/// in the declaration and in the checkpoint headers it binds.
pub(super) fn representative_layers(
    d: &mut Declaration,
    m: &mut projection_spec::bind::CheckpointMetadata,
    g: &Geometry,
) {
    // An assembly that reads fixed source layers keeps their whole stem.
    let fixed: BTreeSet<_> = d
        .projections
        .iter()
        .flat_map(|p| &p.assembly)
        .filter(|m| matches!(m.layer, Some(projection_spec::LayerSelector::Index(_))))
        .filter_map(|m| d.projections.iter().find(|p| p.id == m.projection)?.stem())
        .collect();
    let stems: BTreeSet<_> = d
        .projections
        .iter()
        .filter_map(|p| p.stem())
        .filter(|s| !fixed.contains(s))
        .map(str::to_string)
        .collect();
    for stem in d
        .layer_stems
        .iter_mut()
        .filter(|s| s.execution.is_none() && stems.contains(&s.id))
    {
        let all = stem.indices.resolve(g).unwrap();
        if all.len() <= 3 {
            continue;
        }
        let kept = [all[0], all[all.len() / 2], all[all.len() - 1]];
        stem.indices = LayerIndices::List {
            indices: kept.iter().map(|&i| Dim::Literal(i)).collect(),
        };
        let dropped: Vec<_> = all
            .iter()
            .filter(|i| !kept.contains(i))
            .map(|i| format!("{}.", stem.template.replace("{i}", &i.to_string())))
            .collect();
        m.tensors
            .retain(|name, _| !dropped.iter().any(|prefix| name.starts_with(prefix)));
    }
}

/// Ranks and layers that repeat a transform, source and stored shape differ
/// only in the rows they select, which the exact record comparison pins, so
/// the default checks the record's whole payload once per such recipe.
fn payload_recipe(record: &kernel_core::weight_transform::WeightRecord) -> String {
    format!(
        "{:?} {:?} {:?} {:?}",
        record.transform, record.axis, record.sources, record.stored_shape
    )
}

#[test]
fn fp8_tp2_served_graph_matches_native_variant_specs_records_and_stages() {
    assert_eq!(served_graph(false, 2, Layers::Representative, 1805), 14);
}

#[test]
fn fp8_tp4_served_graph_matches_native_variant_specs_records_and_stages() {
    assert_eq!(served_graph(false, 4, Layers::Representative, 1805), 14);
}

#[test]
fn nvfp4_routed_tp4_served_graph_matches_native_variant_specs_records_and_stages() {
    assert_eq!(served_graph(true, 4, Layers::Representative, 1406), 10);
}

#[test]
#[ignore = "every layer of every served graph; about 50 minutes, run with --ignored"]
fn complete_served_graphs_match_native_variant_specs_records_and_stages() {
    served_graph(false, 2, Layers::Every, 16055);
    served_graph(false, 4, Layers::Every, 16055);
    served_graph(true, 4, Layers::Every, 15656);
}

/// Derive and compare one served graph, of the FP8 export or the `mixed`
/// layout; returns the number of record recipes whose whole payload bytes
/// it compared.
fn served_graph(mixed: bool, ranks: u32, layers: Layers, operations: usize) -> usize {
    let (export, text) = if mixed {
        (MIXED.name(), MIXED.text())
    } else {
        ("fp8", fixtures::exports::FP8)
    };
    let mut metadata = fixtures::fixture(text);
    fixtures::owner_scope(&crate::projections::declaration(4).unwrap(), &mut metadata);
    let mut payloads = BTreeSet::new();
    let g = crate::projections::geometry(ranks).unwrap();
    let mut d = crate::projections::declaration(ranks)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    if layers == Layers::Representative {
        representative_layers(&mut d, &mut metadata, &g);
    }
    let mut bound = fixtures::bind(&d, &g, &metadata).unwrap();
    let mut graph = d.clone();
    graph.projections.sort_by_key(|p| p.assembly.is_empty());
    if mixed {
        for drafter in d.drafters.iter() {
            graph.omit_drafter(&drafter.id).unwrap();
        }
    }
    bound
        .operands
        .retain(|b| graph.projections.iter().any(|p| p.id == b.projection));
    bound
        .side_views
        .retain(|b| graph.side_tensors.iter().any(|s| s.id == b.side));
    bound
        .fusion_groups
        .retain(|b| graph.fusion_groups.iter().any(|f| f.id == b.id));
    let imported = imports(&graph, &bound, &g);
    for rank in 0..ranks {
        let result = projection_spec::derive::derive(DerivationInput {
            declaration: &graph,
            bound: &bound,
            geometry: &g,
            registry: &crate::projections::registry().unwrap(),
            topology: Topology { ranks, rank },
            imported: &imported,
            variants: &super::super::Variants,
        })
        .unwrap_or_else(|e| panic!("{export} TP{ranks} rank {rank}: {e}"));
        let mut variants = BTreeSet::new();
        let mut native_plans = BTreeMap::new();
        assert_eq!(result.operations.len(), operations);
        let shape =
            crate::contract::Shape::of(crate::contract::Ranks::new(ranks).unwrap()).unwrap();
        for operation in result
            .operations
            .iter()
            .chain(result.fusions.iter().flat_map(|f| &f.operations))
        {
            if !operation.variant_records.is_empty() {
                assert_eq!(
                    model_compiler_sdk::quantization::projection::records(operation).unwrap(),
                    operation.variant_records
                );
                let native = native_plans
                    .entry(operation.source_instance.clone())
                    .or_insert_with(|| {
                        native_records(&imported[&operation.source_instance], shape, &bound, rank)
                    });
                for view in operation
                    .views
                    .iter()
                    .filter(|v| matches!(v.role, projection_spec::derive::ViewRole::Weight(_)))
                {
                    let expected = &native.views[&view.id];
                    assert_eq!(
                        (view.shape.clone(), view.bytes, view.layout.clone()),
                        expected.clone()
                    );
                }
                for record in &operation.variant_records {
                    let (expected, offset) = &native.records[&record.tensors[0]];
                    assert_eq!(&record.record, expected);
                    assert_eq!(record.view_offset, *offset);
                    assert_eq!(
                        record.record.transform.encode_params().unwrap(),
                        expected.transform.encode_params().unwrap()
                    );
                    let payload = match layers {
                        Layers::Representative => payload_recipe(&record.record),
                        Layers::Every => format!(
                            "{rank} {} {} {}",
                            operation.instance.projection,
                            record.section.index(),
                            record.tensors[0].contains(".up_proj.")
                        ),
                    };
                    if payloads.insert(payload) {
                        super::bytes::assert_whole_bytes(&record.record);
                    }
                }
            }
            if let Some(adapter) = operation.stages.first().and_then(|s| s.adapter.as_ref())
                && adapter.contract == "qwen38/legacy-fused-dense.v1"
                && operation
                    .instance
                    .projection
                    .ends_with("input_mix_weight_up")
            {
                let expected = if operation.requested.phase
                    != kernel_core::precision::ExecutionPhase::Prefill
                    && operation.requested.capacity <= crate::contract::DECODE_ENTRY_ROWS
                {
                    crate::contract::DECODE_ENTRY_ROWS
                } else {
                    crate::contract::PREFILL_TOKENS
                };
                assert_eq!(
                    adapter.facts["native-capacity"],
                    projection_spec::derive::AdapterValue::U64(expected.into())
                );
            }
            if operation.instance.projection == "head" {
                let output = operation.operation.arithmetic.output;
                assert!(matches!(
                    output,
                    kernel_core::contract::DType::Bf16 | kernel_core::contract::DType::F32
                ));
                assert_eq!(
                    operation.operation.matrix_geometry().unwrap().rows,
                    u64::from(shape.vocab() / ranks)
                );
                assert_eq!(
                    operation.operation.arithmetic.decode_order,
                    kernel_core::precision::DecodeOrder::ScaleThenCode
                );
            }
            if operation.instance.projection == "ngram" {
                assert_eq!(operation.operation.geometry, None);
                assert_eq!(
                    operation.operation.matrix_geometry().unwrap().rows,
                    *g.get("ngram_rows").unwrap()
                );
                let native =
                    crate::precision::legacy_lookup_capability(shape, true, operation.requested)
                        .unwrap();
                let adapter = operation.stages[0].adapter.as_ref().unwrap();
                assert_eq!(
                    adapter.execution,
                    projection_spec::derive::StageExecution::Fused {
                        operation: native.stages[0].clone()
                    }
                );
                assert_eq!(
                    operation.operation.arithmetic.decode_order,
                    native.arithmetic.decode_order
                );
                assert_eq!(
                    adapter.facts["table-start"],
                    projection_spec::derive::AdapterValue::U64(
                        crate::views::ngram_rows(rank, shape).unwrap().0
                    )
                );
            }
        }
        for fusion in result.fusions.iter().filter(|f| f.fused) {
            assert!(!fusion.stages.is_empty());
            for member in &fusion.operations {
                assert_eq!(
                    member.operation.matrix_geometry().unwrap(),
                    member.logical_geometry
                );
            }
            for stage in &fusion.stages {
                let adapter = stage.adapter.as_ref().unwrap();
                let projection_spec::derive::AdapterValue::List(outputs) =
                    &adapter.facts["outputs"]
                else {
                    panic!("missing fusion outputs")
                };
                for output in outputs {
                    let projection_spec::derive::AdapterValue::List(fields) = output else {
                        panic!("missing segment")
                    };
                    let projection_spec::derive::AdapterValue::Text(buffer) = &fields[1] else {
                        panic!("missing output buffer")
                    };
                    let projection_spec::derive::AdapterValue::U64(offset) = fields[2] else {
                        panic!("missing offset")
                    };
                    let projection_spec::derive::AdapterValue::U64(stride) = fields[3] else {
                        panic!("missing stride")
                    };
                    let projection_spec::derive::AdapterValue::U64(columns) = fields[4] else {
                        panic!("missing live columns")
                    };
                    let view = stage.outputs.iter().find(|v| &v.id == buffer).unwrap();
                    assert_eq!(*view.shape.last().unwrap(), stride);
                    assert!(offset + columns <= stride);
                    let expected_stride = match buffer.as_str() {
                        "qkv" if fusion.id.starts_with("linear") => shape.linear_qkv,
                        "qkv" => shape.attn_row,
                        "z" => shape.linear_value,
                        "ab" => shape.linear_ab_stride,
                        "index" => shape.index_row(),
                        "down" => shape.hc_rank(),
                        "inject" => shape.streams(),
                        _ => panic!("unknown native buffer"),
                    };
                    assert_eq!(stride, u64::from(expected_stride));
                }
            }
        }
        for stage in result
            .operations
            .iter()
            .flat_map(|o| &o.stages)
            .chain(result.fusions.iter().flat_map(|f| &f.stages))
        {
            if let Some(a) = &stage.adapter {
                let bytes = serde_json::to_vec(a).unwrap();
                assert_eq!(
                    serde_json::from_slice::<projection_spec::derive::StageAdapter>(&bytes)
                        .unwrap(),
                    *a
                );
                variants.insert(a.contract.as_str());
            }
        }
        assert!(variants.contains("qwen38/ngram-hash.v1"));
        assert!(variants.contains("qwen38/legacy-input-fusion.v1"));
        assert!(variants.contains("qwen38/legacy-fused-dense.v1"));
        assert_eq!(
            variants.contains("qwen38/legacy-fragment-experts.v1"),
            !mixed
        );
    }
    payloads.len()
}
