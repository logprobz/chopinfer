use super::{fixtures, prototype, retired};
use projection_spec::derive::{DerivationInput, ImportedProjections, InstanceKey, Topology};
use projection_spec::{Declaration, Dim, Geometry};
use std::collections::{BTreeMap, BTreeSet};

fn assert_mtp_alias(
    alias: &projection_spec::derive::DerivedOperation,
    result: &projection_spec::derive::RankDerivation,
) {
    let source = result
        .operations
        .iter()
        .find(|o| o.instance == alias.source_instance && o.requested == alias.requested)
        .unwrap();
    assert_eq!(alias.source_layer, Some(0));
    assert_eq!(
        alias.instance.layer,
        Some(if alias.instance.projection.starts_with("mtp.attention") {
            1
        } else {
            0
        })
    );
    assert_eq!(alias.operation.arithmetic, source.operation.arithmetic);
    assert_eq!(alias.operation.epilogues, source.operation.epilogues);
    assert_eq!(alias.logical_geometry, source.logical_geometry);
    assert_eq!(alias.packing, source.packing);
    assert_eq!(alias.stages, source.stages);
    // An adapter selects its whole source: the same rows of each source span,
    // which it may split by part, and weight views of the same sizes.
    let windows = |o: &projection_spec::derive::DerivedOperation| {
        let mut rows = BTreeMap::<_, Vec<_>>::new();
        for s in &o.slices {
            let w = &s.source.weight;
            let span = (&w.codes, w.local_scale.as_ref().map(|l| &l.tensor));
            rows.entry(format!("{span:?} {} {:?}", s.expert, s.k))
                .or_default()
                .extend(s.rows.iter().flat_map(Clone::clone));
        }
        rows
    };
    assert_eq!(
        windows(alias),
        windows(source),
        "{} slices",
        alias.instance.projection
    );
    let weights = |o: &projection_spec::derive::DerivedOperation| {
        o.views
            .iter()
            .filter(|v| matches!(v.role, projection_spec::derive::ViewRole::Weight(_)))
            .map(|v| (v.role, v.bytes))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        weights(alias),
        weights(source),
        "{} views",
        alias.instance.projection
    );
    assert_eq!(
        alias.source_preparation.is_some(),
        source.source_preparation.is_some()
    );
    if !alias.variant_records.is_empty() {
        assert_eq!(
            model_compiler_sdk::quantization::projection::records(alias).unwrap(),
            alias.variant_records
        );
    }
}

/// The record source of one checkpoint tensor, from its header.
fn header_source(
    metadata: &projection_spec::bind::CheckpointMetadata,
    name: &str,
) -> kernel_core::weight_transform::SourceTensor {
    let header = &metadata.tensors[name];
    let width = match header.dtype.as_str() {
        "BF16" => 2,
        "F32" => 4,
        "F8_E4M3" => 1,
        other => panic!("unexpected alias source dtype {other}"),
    };
    kernel_core::weight_transform::SourceTensor {
        dtype: header.dtype.clone(),
        shape: header.shape.clone(),
        byte_length: header.shape.iter().product::<u64>() * width,
    }
}

/// Compare an alias without provider records with the retired loader's rank
/// piece, as recorded: its slices, weight views and complete records on every
/// rank and use. Only the payload bytes are sampled.
fn assert_ordinary_alias(
    alias: &projection_spec::derive::DerivedOperation,
    graph: &projection_spec::derive::RankDerivation,
    metadata: &projection_spec::bind::CheckpointMetadata,
    tp: crate::Tp,
    rank: u32,
    bytes: bool,
) {
    use kernel_core::weight_transform::{
        PackedMatrixSource, PackedOperand, PackedSection, WeightRecord, WeightTransform,
    };
    use projection_spec::derive::ViewRole;
    let source = graph
        .operations
        .iter()
        .find(|o| o.instance == alias.source_instance && o.requested == alias.requested)
        .unwrap();
    let weight = &source.operation.weights[0];
    let codes = &weight.codes;
    let local = weight.local_scale.as_ref();
    let label = format!(
        "alias tp{} rank{rank} {} {}",
        tp.ranks(),
        alias.instance.projection,
        retired::usage(&alias.requested)
    );
    let mut lines = retired::Lines::default();
    // A weight view matches its source's.
    let assert_view = |section| {
        let [view, source_view] = [alias, source].map(|o| {
            o.views
                .iter()
                .find(|v| v.role == ViewRole::Weight(section))
                .unwrap_or_else(|| panic!("{} lacks a {section:?} view", o.instance.projection))
        });
        assert_eq!(
            (&view.dtype, &view.shape, &view.layout, view.bytes),
            (
                &source_view.dtype,
                &source_view.shape,
                &source_view.layout,
                source_view.bytes
            ),
            "{} {section:?} view",
            alias.instance.projection
        );
        view.id.clone()
    };
    let codes_view = assert_view(PackedSection::Codes);
    let local_view = local.map(|_| assert_view(PackedSection::LocalScales));
    for view in alias
        .views
        .iter()
        .filter(|v| matches!(v.role, ViewRole::Weight(_)))
    {
        lines.add(&label, format!("view {:?} bytes={}", view.role, view.bytes));
    }
    if alias.instance.projection.ends_with("qkv") {
        assert_eq!(tp.ranks(), 4);
        let mut rows = Vec::new();
        for slice in &alias.slices {
            assert_eq!(slice.source.weight.codes.tensor, codes.tensor);
            assert_eq!(slice.k, 0..4096);
            let span = &slice.source.weight.codes;
            let first = span.offset / (span.byte_length / span.shape[0]);
            rows.extend(
                slice
                    .rows
                    .iter()
                    .flat_map(|r| r.clone().map(|row| first + row)),
            );
        }
        lines.add(&label, format!("rows={rows:?}"));
        assert!(
            model_compiler_sdk::quantization::projection::records(alias)
                .unwrap_err()
                .to_string()
                .contains("part views require")
        );
        lines.check(&label);
        return;
    }
    assert_eq!(alias.slices.len(), 1);
    let slice = &alias.slices[0];
    assert_eq!(slice.source.weight.codes.tensor, codes.tensor);
    let (rows, k) = (codes.shape[0], codes.shape[1]);
    lines.add(
        &label,
        format!("slice rows={:?} k={:?}", slice.rows, slice.k),
    );
    let plan = &alias.packing;
    assert_eq!(
        (plan.format, plan.experts, plan.projections),
        (weight.format, 1, 1)
    );
    lines.add(&label, format!("stored={:?}", [plan.rows, plan.k]));
    let matrix = PackedMatrixSource {
        expert: 0,
        projection: 0,
        format: weight.format,
        rows,
        k,
        row_origin: 0,
        k_origin: 0,
        nibble_order: kernel_core::precision::NibbleOrder::EvenLow,
        codes: 0,
        local: local.map(|_| 1),
        local_dtype: local.map(|l| l.tensor.dtype),
        global: None,
        source_expert: 0,
        source_experts: 1,
        rows_kept: slice.rows.clone(),
        k_kept: slice.k.clone(),
    };
    let mut names = vec![codes.tensor.clone()];
    if let Some(local) = local {
        names.push(local.tensor.tensor.clone());
    }
    let sources: Vec<_> = names.iter().map(|n| header_source(metadata, n)).collect();
    let records = model_compiler_sdk::quantization::projection::records(alias).unwrap();
    assert_eq!(records.len(), names.len());
    let byte_label = format!("alias-bytes{}", label.strip_prefix("alias").unwrap());
    let mut byte_lines = retired::Lines::default();
    for record in records {
        let view = if record.section == PackedSection::Codes {
            &codes_view
        } else {
            assert_eq!(record.section, PackedSection::LocalScales);
            local_view.as_ref().unwrap()
        };
        let expected = WeightRecord {
            transform: WeightTransform::PackedOperand(Box::new(PackedOperand {
                plan: plan.clone(),
                section: record.section,
                decode_order: source.operation.arithmetic.decode_order,
                matrices: vec![matrix.clone()],
            })),
            sources: sources.clone(),
            axis: None,
            ranges: Vec::new(),
            stored_shape: Vec::new(),
        };
        assert_eq!(
            record.record, expected,
            "{} rank {rank} {:?} record",
            alias.instance.projection, record.section
        );
        lines.add(
            &label,
            format!(
                "record {:?} {}",
                record.section,
                retired::record(&record.record)
            ),
        );
        assert_eq!(record.view_offset, 0);
        assert_eq!(record.tensors, names);
        assert_eq!(&record.view, view);
        if bytes {
            byte_lines.add(
                &byte_label,
                format!(
                    "{:?} {}",
                    record.section,
                    retired::bytes(&super::bytes::produced_bytes(&record.record))
                ),
            );
        }
    }
    lines.check(&label);
    if bytes {
        byte_lines.check(&byte_label);
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
/// their projection, operand, use and stage declarations, so the default keeps
/// each stem's first, middle and last layer; `Every` is the opt-in full sweep.
#[derive(Clone, Copy, PartialEq)]
enum Layers {
    Representative,
    Every,
}

/// Keep the first, middle and last layer of every unaliased projection stem
/// with more, in the declaration and in the checkpoint headers it binds.
fn representative_layers(
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
    for stem in d.layer_stems.iter_mut().filter(|s| {
        stems.contains(&s.id) && s.execution.as_ref().is_none_or(|e| e.aliases.is_empty())
    }) {
        let all = stem.indices.resolve(g).unwrap();
        if all.len() <= 3 {
            continue;
        }
        let kept = [all[0], all[all.len() / 2], all[all.len() - 1]];
        let literal: Vec<_> = kept.iter().map(|&i| Dim::Literal(i)).collect();
        if let Some(execution) = &mut stem.execution {
            assert_eq!(execution.indices.len(), all.len());
            execution.indices = literal.clone();
        }
        stem.indices = projection_spec::LayerIndices::List { indices: literal };
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

/// The aliases whose payload bytes a served graph samples.
const ALIASES: [&str; 4] = [
    "mtp.attention-adapter.qkv",
    "mtp.attention-adapter.output",
    "mtp.mlp-adapter.gate-up",
    "mtp.mlp-adapter.down",
];

/// Check one default served-graph test: its exact count of whole-payload
/// recipes, and alias bytes for every alias on the sampled first and last
/// ranks among `checked`.
fn representative_graph(ranks: u32, checked: std::ops::Range<u32>, recipes: usize) {
    let sampled: Vec<_> = [0, ranks - 1]
        .into_iter()
        .filter(|r| checked.contains(r))
        .collect();
    let coverage = served_graph(ranks, checked, Layers::Representative, 1027);
    assert_eq!(coverage.recipes, recipes);
    assert_eq!(
        coverage.alias_bytes,
        ALIASES
            .iter()
            .flat_map(|a| sampled.iter().map(|&r| (a.to_string(), r)))
            .collect::<BTreeSet<_>>()
    );
}

#[test]
fn tp2_rank0_served_graph_matches_native_variant_specs_records_and_stages() {
    representative_graph(2, 0..1, 13);
}

#[test]
fn tp2_rank1_served_graph_matches_native_variant_specs_records_and_stages() {
    representative_graph(2, 1..2, 13);
}

#[test]
fn tp4_ranks01_served_graph_matches_native_variant_specs_records_and_stages() {
    representative_graph(4, 0..2, 3);
}

#[test]
fn tp4_ranks23_served_graph_matches_native_variant_specs_records_and_stages() {
    representative_graph(4, 2..4, 3);
}

/// An interior-rank alias whose K window moved by one scale block, keeping its
/// length, must fail the record comparison that runs without payload bytes.
#[test]
fn shifted_interior_alias_k_windows_fail_the_record_comparison() {
    let mut metadata = fixtures::fixture(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/projections/mopd.json"
    )));
    fixtures::owner_scope(&mut metadata);
    let g = crate::projections::geometry(4).unwrap();
    let mut d = crate::projections::declaration(4)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let mut bound = fixtures::bind(&d, &g, &metadata).unwrap();
    let ids = [
        "mtp.output",
        "mtp.attention-adapter.output",
        "mtp-mlp.down",
        "mtp.mlp-adapter.down",
    ];
    d.projections.retain(|p| ids.contains(&p.id.as_str()));
    for p in &mut d.projections {
        p.uses.truncate(1);
    }
    d.side_tensors.clear();
    d.fusion_groups.clear();
    bound
        .operands
        .retain(|b| ids.contains(&b.projection.as_str()));
    bound
        .assemblies
        .retain(|b| ids.contains(&b.projection.as_str()));
    bound.side_tensors.clear();
    bound.side_views.clear();
    bound.fusion_groups.clear();
    let imported = imports(&d, &bound, &g);
    let mut rejected = 0;
    for rank in [1, 2] {
        let result = projection_spec::derive::derive(DerivationInput {
            declaration: &d,
            bound: &bound,
            geometry: &g,
            registry: &crate::projections::registry().unwrap(),
            topology: Topology { ranks: 4, rank },
            imported: &imported,
            variants: &super::super::Variants::FLASH,
        })
        .unwrap();
        for alias in result
            .operations
            .iter()
            .filter(|o| o.instance.projection.contains("-adapter."))
        {
            assert!(alias.variant_records.is_empty());
            assert_ordinary_alias(alias, &result, &metadata, crate::Tp::TP4, rank, false);
            let mut shifted = alias.clone();
            shifted.slices[0].k.start += 128;
            shifted.slices[0].k.end += 128;
            let check = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                assert_ordinary_alias(&shifted, &result, &metadata, crate::Tp::TP4, rank, false)
            }));
            assert!(
                check.is_err(),
                "{} rank {rank}: shifted K window passed",
                alias.instance.projection
            );
            // The shifted window also fails the record comparison on its own.
            let records = model_compiler_sdk::quantization::projection::records(&shifted).unwrap();
            let original = model_compiler_sdk::quantization::projection::records(alias).unwrap();
            assert!(
                records
                    .iter()
                    .zip(&original)
                    .all(|(a, b)| a.record != b.record)
            );
            rejected += 1;
        }
    }
    assert_eq!(rejected, 4);
}

#[test]
#[ignore = "every layer of both served graphs; about 8 minutes, run with --ignored"]
fn complete_served_graphs_match_native_variant_specs_records_and_stages() {
    for ranks in [2, 4] {
        let coverage = served_graph(ranks, 0..ranks, Layers::Every, 5131);
        assert_eq!(
            coverage.alias_bytes,
            ALIASES
                .iter()
                .flat_map(|a| (0..ranks).map(|r| (a.to_string(), r)))
                .collect::<BTreeSet<_>>()
        );
    }
}

/// What a served-graph comparison sampled.
struct Coverage {
    recipes: usize,
    alias_bytes: BTreeSet<(String, u32)>,
}

/// Derive and compare the served graph of `ranks` on the `checked` ranks.
fn served_graph(
    ranks: u32,
    checked: std::ops::Range<u32>,
    layers: Layers,
    operations: usize,
) -> Coverage {
    let mut metadata = fixtures::fixture(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/projections/mopd.json"
    )));
    fixtures::owner_scope(&mut metadata);
    let mut payloads = BTreeSet::new();
    let mut alias_bytes = BTreeSet::new();
    let g = crate::projections::geometry(ranks).unwrap();
    let mut d = crate::projections::declaration(ranks)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    if layers == Layers::Representative {
        representative_layers(&mut d, &mut metadata, &g);
    }
    let bound = fixtures::bind(&d, &g, &metadata).unwrap();
    let mut graph = d.clone();
    // Resolve registered sources before their declaration order reaches them.
    graph.projections.sort_by_key(|p| p.assembly.is_empty());
    let imported = imports(&graph, &bound, &g);
    for rank in checked {
        let scope = format!(
            "graph {} tp{ranks} rank{rank}",
            match layers {
                Layers::Representative => "representative",
                Layers::Every => "every",
            }
        );
        let mut lines = retired::Lines::default();
        let result = projection_spec::derive::derive(DerivationInput {
            declaration: &graph,
            bound: &bound,
            geometry: &g,
            registry: &crate::projections::registry().unwrap(),
            topology: Topology { ranks, rank },
            imported: &imported,
            variants: &super::super::Variants::FLASH,
        })
        .unwrap_or_else(|e| panic!("TP{ranks} rank {rank}: {e}"));
        assert!(!result.operations.is_empty());
        let mut stages = BTreeSet::new();
        let mut alias_payloads = BTreeSet::new();
        let mut prepared = 0;
        let mut adapters = 0;
        let mut heads = 0;
        for operation in &result.operations {
            let tp = crate::Tp::new(ranks).unwrap();
            if operation.instance.projection.contains("-adapter.") {
                adapters += 1;
                assert_mtp_alias(operation, &result);
                // Records, slices and views are compared on every rank and
                // use; alias payload bytes once per alias, on the first and
                // last rank by default.
                let bytes = (layers == Layers::Every || rank == 0 || rank + 1 == ranks)
                    && alias_payloads.insert(operation.instance.projection.clone());
                if bytes {
                    super::bytes::assert_alias_bytes(operation, &result);
                    alias_bytes.insert((operation.instance.projection.clone(), rank));
                }
                if operation.variant_records.is_empty() {
                    assert_ordinary_alias(operation, &result, &metadata, tp, rank, bytes);
                }
            }
            if let Some(preparation) = &operation.source_preparation {
                assert_eq!(ranks, 2);
                prepared += 1;
                assert_eq!(preparation.topology, Topology { ranks, rank });
                assert_eq!(
                    model_compiler_sdk::quantization::projection::records(operation).unwrap(),
                    preparation.records
                );
            }
            for record in &operation.variant_records {
                let source = &record.record.sources[0];
                let header = &metadata.tensors[&record.tensors[0]];
                assert_eq!(source.shape, header.shape);
                assert_eq!(source.dtype, header.dtype);
                lines.add(
                    &scope,
                    format!(
                        "variant {} {:?} {} {:?} tensors={:?} offset={} {}",
                        operation.instance.projection,
                        operation.instance.layer,
                        retired::usage(&operation.requested),
                        record.section,
                        record.tensors,
                        record.view_offset,
                        retired::record(&record.record)
                    ),
                );
                let payload = match layers {
                    Layers::Representative => payload_recipe(&record.record),
                    Layers::Every => format!(
                        "{rank} {} {}",
                        operation.instance.projection,
                        record.section.index()
                    ),
                };
                if payloads.insert(payload) {
                    super::bytes::assert_whole_bytes(&record.record);
                }
            }
            if operation.instance.projection.ends_with("qkv")
                && operation.instance.projection != "dflash.qkv"
            {
                assert_eq!(operation.source_preparation.is_some(), ranks == 2);
                assert_eq!(
                    operation.operation.weights[0].format,
                    crate::capabilities::Projection::Qkv {
                        global: operation.instance.projection == "ga.qkv"
                    }
                    .weight(tp)
                );
            }
            let role = match operation.instance.projection.as_str() {
                "ga.qkv" => Some(crate::capabilities::Projection::Qkv { global: true }),
                "swa.qkv" | "mtp.qkv" | "mtp.attention-adapter.qkv" => {
                    Some(crate::capabilities::Projection::Qkv { global: false })
                }
                "dense.mlp.gate-up" | "mtp-mlp.mlp.gate-up" | "mtp.mlp-adapter.gate-up" => {
                    Some(crate::capabilities::Projection::MlpGateUp)
                }
                "router" => Some(crate::capabilities::Projection::Router),
                "head" => {
                    heads += 1;
                    Some(crate::capabilities::Projection::Head {
                        f32: operation.operation.arithmetic.output
                            == kernel_core::contract::DType::F32,
                    })
                }
                _ => None,
            };
            if let Some(role) = role {
                let mut expected = crate::precision::offered_operation(
                    operation.operation.id.clone(),
                    role,
                    tp,
                    operation.operation.weights.clone(),
                    operation.operation.activations.clone(),
                    None,
                );
                expected.uses = vec![operation.requested];
                assert_eq!(
                    operation.operation, expected,
                    "{}",
                    operation.instance.projection
                );
            }
            for stage in &operation.stages {
                let adapter = stage.adapter.as_ref().unwrap();
                assert_eq!(
                    serde_json::from_slice::<projection_spec::derive::StageAdapter>(
                        &serde_json::to_vec(adapter).unwrap()
                    )
                    .unwrap(),
                    *adapter
                );
                if adapter.contract == "mimo26/legacy-fused-dense.v1" {
                    let declared = graph
                        .projections
                        .iter()
                        .find(|p| p.id == operation.source_instance.projection)
                        .unwrap()
                        .composition
                        .as_ref()
                        .unwrap()
                        .implementation
                        .as_ref()
                        .unwrap();
                    let parameters: crate::projections::DenseParameters =
                        crate::projections::typed(declared).unwrap();
                    assert_eq!(
                        parameters.scale_order,
                        match operation.operation.arithmetic.decode_order {
                            kernel_core::precision::DecodeOrder::BlockScaleAfterMma => {
                                crate::projections::ScaleOrder::BlockScaleAfterMma
                            }
                            kernel_core::precision::DecodeOrder::ScaleThenCode => {
                                crate::projections::ScaleOrder::ScaleThenCode
                            }
                            order => panic!("unexpected native gate/up decode order: {order:?}"),
                        },
                        "{} declared scale order differs from served arithmetic",
                        operation.instance.projection
                    );
                    let expected = if operation.operation.arithmetic.decode_order
                        == kernel_core::precision::DecodeOrder::BlockScaleAfterMma
                    {
                        "block-scale-after-mma"
                    } else {
                        "scale-then-code"
                    };
                    assert_eq!(
                        adapter.facts["scale-order"],
                        projection_spec::derive::AdapterValue::Text(expected.into())
                    );
                }
                if adapter.contract != "mimo26/legacy-fused-dense.v1" {
                    let p = graph
                        .projections
                        .iter()
                        .find(|p| p.id == operation.instance.projection)
                        .unwrap();
                    let declared = p
                        .composition
                        .as_ref()
                        .unwrap()
                        .pre
                        .iter()
                        .chain(&p.composition.as_ref().unwrap().post)
                        .find(|s| s.id == stage.id)
                        .unwrap();
                    assert_eq!(
                        *stage,
                        super::super::stage_descriptor(declared, tp, operation.requested)
                            .unwrap()
                            .stage
                    );
                }
            }
            for stage in &operation.stages {
                if let Some(adapter) = &stage.adapter {
                    stages.insert(adapter.contract.as_str());
                }
            }
        }
        assert_eq!(stages.len(), 4);
        let qkv_uses = result
            .operations
            .iter()
            .filter(|o| {
                o.instance.projection.ends_with("qkv") && o.instance.projection != "dflash.qkv"
            })
            .count();
        assert_eq!(prepared, if ranks == 2 { qkv_uses } else { 0 });
        assert_eq!(adapters, 4 * 19);
        assert_eq!(result.operations.len(), operations);
        assert_eq!(heads, 38);
        lines.check(&scope);
    }
    Coverage {
        recipes: payloads.len(),
        alias_bytes,
    }
}

/// The bytes of `section` that the served rank record of `output`'s source
/// QKV produces in rank rows `window`.
fn served_window(
    result: &projection_spec::derive::RankDerivation,
    output: &projection_spec::derive::DerivedOperation,
    section: kernel_core::weight_transform::PackedSection,
    window: std::ops::Range<u64>,
) -> Vec<u8> {
    use model_compiler_sdk::quantization::projection::{bind_record_policy, records};
    let source = result
        .operations
        .iter()
        .find(|o| o.instance.projection == "mtp.qkv" && o.requested == output.requested)
        .unwrap();
    let format = kernel_core::precision::format::registry::by_format(source.packing.format)
        .unwrap()
        .id();
    let bound = bind_record_policy(
        source.clone(),
        crate::weight_plan::record_policy(format, source.family),
    )
    .unwrap();
    let records: Vec<_> = records(&bound)
        .unwrap()
        .into_iter()
        .filter(|r| r.section == section)
        .collect();
    let length = records
        .iter()
        .map(|r| r.view_offset + r.record.output_bytes().unwrap())
        .max()
        .unwrap();
    let mut view = vec![0; length as usize];
    for record in &records {
        let bytes = super::bytes::produced_bytes(&record.record);
        view[record.view_offset as usize..record.view_offset as usize + bytes.len()]
            .copy_from_slice(&bytes);
    }
    let rows: u64 = records.iter().map(|r| r.record.stored_shape[0]).sum();
    let row_bytes = length / rows;
    view[(window.start * row_bytes) as usize..(window.end * row_bytes) as usize].to_vec()
}

#[test]
fn registered_rank_local_qkv_windows_match_native_records_and_bytes() {
    use kernel_core::weight_transform::PackedSection;
    let windows: BTreeMap<&str, &str> = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/projections/retired-slicer-qkv-windows.sha256"
    ))
    .lines()
    .filter(|line| !line.starts_with('#'))
    .map(|line| {
        let (digest, label) = line.split_once("  ").unwrap();
        (label, digest)
    })
    .collect();
    let mut metadata = fixtures::fixture(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/projections/mopd.json"
    )));
    fixtures::owner_scope(&mut metadata);
    for ranks in [2, 4] {
        let tp = crate::Tp::new(ranks).unwrap();
        let g = crate::projections::geometry(ranks).unwrap();
        let d = crate::projections::declaration(ranks)
            .unwrap()
            .expanded_profiles(&g)
            .unwrap();
        let full_bound = fixtures::bind(&d, &g, &metadata).unwrap();
        for start in [128, tp.qkv_rows(false) - 128] {
            let mut graph = d.clone();
            graph.projections.retain(|p| p.id == "mtp.qkv");
            graph.side_tensors.clear();
            graph.fusion_groups.clear();
            let mut destination = graph.projections[0].clone();
            destination.id = "registered-rank-local-qkv".into();
            destination.source = None;
            destination.stem = None;
            destination.shape[0] = Dim::Literal(128);
            destination.split = projection_spec::SplitClass::Replicate {};
            destination.parts.clear();
            destination.source_transforms.clear();
            destination.inputs.clear();
            destination.uses.truncate(1);
            destination.assembly = vec![
                serde_json::from_value(serde_json::json!({
                    "projection": "mtp.qkv", "layer": 0, "operand": null,
                    "rows": {"start": start, "length": 128},
                    "coordinates": "rank_local", "inherit_split": true
                }))
                .unwrap(),
            ];
            let mut bound = full_bound.clone();
            let names = bound
                .operands
                .iter()
                .filter(|b| b.projection == "mtp.qkv" && b.layer == Some(0))
                .map(|b| b.imported.weight.codes.tensor.clone())
                .collect();
            bound.assemblies = vec![projection_spec::bind::BoundAssembly {
                projection: destination.id.clone(),
                layer: None,
                members: destination.assembly.clone(),
                tensors: vec![names],
            }];
            bound.operands.retain(|b| b.projection == "mtp.qkv");
            bound.side_tensors.clear();
            bound.side_views.clear();
            bound.fusion_groups.clear();
            graph.projections.insert(0, destination);
            let imported = imports(&graph, &bound, &g);
            for rank in 0..ranks {
                let result = projection_spec::derive::derive(DerivationInput {
                    declaration: &graph,
                    bound: &bound,
                    geometry: &g,
                    registry: &crate::projections::registry().unwrap(),
                    topology: Topology { ranks, rank },
                    imported: &imported,
                    variants: &super::super::Variants::FLASH,
                })
                .unwrap();
                let output = result
                    .operations
                    .iter()
                    .find(|o| o.instance.projection == "registered-rank-local-qkv")
                    .unwrap();
                let scope = format!("qkv-window tp{ranks} rank{rank} start{start}");
                let mut lines = retired::Lines::default();
                assert_eq!(
                    output.logical_geometry,
                    kernel_core::precision::MatrixGeometry { rows: 128, k: 4096 }
                );
                assert_eq!(output.source_preparation.is_some(), ranks == 2);
                assert_eq!(
                    output.operation.weights[0].format,
                    crate::capabilities::Projection::Qkv { global: false }.weight(tp)
                );
                if ranks == 2 {
                    assert_eq!(
                        model_compiler_sdk::quantization::projection::records(output).unwrap(),
                        output.variant_records
                    );
                } else {
                    assert!(output.variant_records.is_empty());
                    assert!(
                        model_compiler_sdk::quantization::projection::records(output)
                            .unwrap_err()
                            .to_string()
                            .contains("part views require")
                    );
                }
                for section in [PackedSection::Codes, PackedSection::LocalScales] {
                    let name = if section == PackedSection::Codes {
                        "model.mtp.layers.0.self_attn.qkv_proj.weight"
                    } else {
                        "model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv"
                    };
                    let unit = if section == PackedSection::Codes {
                        1
                    } else if ranks == 2 {
                        64
                    } else {
                        128
                    };
                    // TP2 emits whole records; TP4's are the authenticated
                    // part addresses of the source rows.
                    if ranks == 2 {
                        let record = output
                            .variant_records
                            .iter()
                            .find(|r| r.section == section)
                            .unwrap();
                        assert_eq!(record.view_offset, 0);
                        assert_eq!(record.tensors, vec![name.to_string()]);
                        lines.add(
                            &scope,
                            format!(
                                "{section:?} {} bytes={}",
                                retired::record(&record.record),
                                retired::bytes(&super::bytes::produced_bytes(&record.record))
                            ),
                        );
                    } else {
                        let mut ranges = Vec::new();
                        for slice in &output.slices {
                            let span = if section == PackedSection::Codes {
                                &slice.source.weight.codes
                            } else {
                                &slice.source.weight.local_scale.as_ref().unwrap().tensor
                            };
                            assert_eq!(span.tensor, name);
                            let row_bytes = span.byte_length / span.shape[0];
                            for rows in &slice.rows {
                                let first = span.offset / row_bytes + rows.start / unit;
                                let last = span.offset / row_bytes + rows.end / unit;
                                ranges.push(first..last);
                            }
                        }
                        lines.add(&scope, format!("{section:?} ranges={ranges:?}"));
                        // The served rank record of the source, at the window,
                        // holds the retired slicer's bytes.
                        let label =
                            format!("qkv-window-bytes tp4 rank{rank} start{start} {section:?}");
                        assert_eq!(
                            retired::bytes(&served_window(
                                &result,
                                output,
                                section,
                                start / unit..(start + 128) / unit
                            )),
                            windows[label.as_str()],
                            "{label}"
                        );
                    }
                }
                lines.check(&scope);
            }
        }
    }
}
