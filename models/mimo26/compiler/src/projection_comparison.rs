//! Freeze comparisons use the full MOPD headers and the production source
//! builder. Rank slices, records, layouts and payloads compare with the
//! retired rank slicer's recorded digests.
#[path = "../../contract/tests/projections_support/mod.rs"]
mod headers;
#[path = "../../contract/tests/projections_support/retired.rs"]
#[allow(dead_code)]
mod retired;

// The synthetic projection payloads every model shares, from the SDK's
// test support sources (outside its module tree).
#[path = "../../../../crates/model-compiler-sdk/test-support/synthetic.rs"]
mod synthetic;

use crate::{catalog::TensorRecord, precision};
use kernel_core::contract::DType;
use kernel_core::precision::format::{A16Recipe, KernelFamily, registry};
use kernel_core::precision::{ActivationSpec, dtype_bits};
use model_compiler_sdk::quantization::{ActivationPolicy, resolve_operation};
use model_mimo26_contract::{
    Tp,
    backend::{Kernels, Selected},
    capabilities::{self, Projection},
    geometry::V26_FLASH,
    projections,
};
use projection_spec::derive::{
    GeometryMode, ImportedProjection, InstanceKey, ProjectionContext, SourceMatrix, Topology,
};
use std::collections::{BTreeMap, BTreeSet};
use synthetic::{SyntheticPayload, fill_source, selected_source_payload, synthetic_payload};

fn stored(name: &str) -> runtime_weights::DType {
    match name {
        "BF16" => runtime_weights::DType::Bf16,
        "F32" => runtime_weights::DType::F32,
        "I64" => runtime_weights::DType::I64,
        "U8" => runtime_weights::DType::U8,
        "F8_E4M3" => runtime_weights::DType::F8E4M3,
        other => panic!("unsupported fixture dtype {other}"),
    }
}

/// How much payload production a served comparison runs. Payload bytes back
/// the record and slice checks that run on every layer, rank and use, so the
/// default compares them once per payload class on the first and last rank,
/// and for the first and last member of an expert bank; `Every` is the opt-in
/// full sweep.
#[derive(Clone, Copy, PartialEq)]
enum Sweep {
    Default,
    Every,
}

impl Sweep {
    fn name(self) -> &'static str {
        match self {
            Sweep::Default => "default",
            Sweep::Every => "every",
        }
    }
}

/// The rank whose payload bytes a comparison checks. Interior ranks select an
/// offset window like the last rank, so the default checks the first and last.
fn payload_rank(rank: u32, ranks: u32, sweep: Sweep) -> Option<u32> {
    (sweep == Sweep::Every || rank == 0 || rank + 1 == ranks).then_some(rank)
}

/// Projections with one weight format, packed layout, row-run count and k
/// window gather and pack alike, so the default compares the smallest of them.
fn payload_class(operation: &projection_spec::derive::DerivedOperation) -> String {
    let slice = &operation.slices[0];
    let weight = &slice.source.weight;
    format!(
        "{:?} {:?} {:?} {} {}",
        weight.format,
        operation.packing.layout,
        operation.packing.columns,
        slice.rows.len(),
        slice.k == (0..weight.logical_shape[1])
    )
}

/// What a served comparison covered: every (layer, projection, rank, use) it
/// derived, its OperationSpec and source selection counts, and the payloads
/// it produced as (projection or bank member, rank).
#[derive(Default)]
struct Coverage {
    keys: BTreeSet<(Option<u64>, String, u32, usize)>,
    specs: usize,
    selections: usize,
    payloads: BTreeSet<(String, u32)>,
}

/// The (layer, projection, rank, use) keys the declaration executes for `ids`
/// on the stem layers in `layers`, independent of the comparison loops.
fn declared_keys(
    ranks: u32,
    ids: &[&str],
    layers: std::ops::Range<u64>,
) -> BTreeSet<(Option<u64>, String, u32, usize)> {
    let g = projections::geometry(ranks).unwrap();
    let d = projections::declaration(ranks)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let mut keys = BTreeSet::new();
    for id in ids {
        let p = d.projections.iter().find(|p| p.id == *id).unwrap();
        let executed: Vec<_> = match p.stem() {
            None => vec![None],
            Some(stem) => {
                let stem = d.layer_stems.iter().find(|s| s.id == stem).unwrap();
                match &stem.execution {
                    Some(execution) => execution
                        .indices
                        .iter()
                        .map(|i| Some(i.resolve_offset(&g).unwrap()))
                        .collect(),
                    None => stem
                        .indices
                        .resolve(&g)
                        .unwrap()
                        .into_iter()
                        .map(Some)
                        .collect(),
                }
            }
        };
        for layer in executed {
            if layer.is_some_and(|l| !layers.contains(&l)) {
                continue;
            }
            for rank in 0..ranks {
                for usage in 0..p.uses.len() {
                    keys.insert((layer, id.to_string(), rank, usage));
                }
            }
        }
    }
    keys
}

const ORDINARY: [&str; 9] = [
    "ga.output",
    "swa.output",
    "dense.down",
    "router",
    "embedding",
    "head",
    "mtp.output",
    "mtp-mlp.down",
    "mtp.eh",
];

/// Check one default ordinary comparison against its declared keys, exact
/// counts and the smallest projection of each payload class, sampled on the
/// first and last rank.
fn ordinary(ranks: u32, count: usize) {
    let coverage = served_ordinary(ranks as usize, Sweep::Default);
    assert_eq!(coverage.keys, declared_keys(ranks, &ORDINARY, 0..u64::MAX));
    assert_eq!(
        (coverage.keys.len(), coverage.specs, coverage.selections),
        (count, count, count)
    );
    assert_eq!(
        coverage.payloads,
        ["ga.output", "dense.down", "router"]
            .iter()
            .flat_map(|id| [(id.to_string(), 0), (id.to_string(), ranks - 1)])
            .collect()
    );
}

#[test]
fn production_imports_preserve_admitted_sources_and_refuse_missing_operands() {
    let m = headers::fixture(include_str!(
        "../../contract/tests/fixtures/projections/mopd.json"
    ));
    let metadata = precision::export_metadata(Some(&m.quantization_config), None).unwrap();
    let mut roles = precision::text_roles(&V26_FLASH, |i| V26_FLASH.is_global(i as u32));
    roles.extend(precision::mtp_roles(&V26_FLASH));
    let tensors: BTreeMap<_, _> = m
        .tensors
        .iter()
        .filter(|(name, _)| name.starts_with("model.") || name.as_str() == "lm_head.weight")
        .map(|(name, info)| {
            let dtype = stored(&info.dtype);
            (
                name.clone(),
                TensorRecord {
                    name: name.clone(),
                    dtype,
                    shape: info.shape.clone(),
                    file: String::new(),
                    file_start: 0,
                    bytes: info.shape.iter().product::<u64>()
                        * match dtype {
                            runtime_weights::DType::Bf16 => 2,
                            runtime_weights::DType::F32 => 4,
                            _ => 1,
                        },
                },
            )
        })
        .collect();
    let admitted = precision::admit(&V26_FLASH, &metadata, &tensors, &roles).unwrap();
    let contract_tensors = tensors.clone();
    for tp in Tp::SUPPORTED {
        let graph =
            projections::imports::derive(tp, tp.ranks() - 1, &contract_tensors, &admitted).unwrap();
        assert!(graph.operations.len() > 3000);
        let specs = graph.operation_specs().unwrap();
        let document = model_mimo26_contract::prepared::document(&graph, tp, &admitted).unwrap();
        assert!(
            document
                .operations
                .iter()
                .any(|s| s.source.id.ends_with(".mid"))
        );
        assert!(
            document
                .operations
                .iter()
                .any(|s| s.source.id.ends_with(".inexact"))
        );
        assert!(document.operations.iter().any(|s| {
            s.reuse.is_some()
                && s.source.activations.iter().any(|a| !a.is_bf16())
                && s.effective
                    .activations
                    .iter()
                    .all(kernel_core::precision::ActivationSpec::is_bf16)
        }));
        let bytes = kernel_core::precision::component::encode(&document).unwrap();
        assert_eq!(
            kernel_core::precision::component::decode(&bytes).unwrap(),
            document
        );
        assert!(
            specs
                .iter()
                .any(|s| s.id.starts_with("head.") && s.arithmetic.output == DType::F32)
        );
        let alias = graph
            .operations
            .iter()
            .find(|o| o.instance.projection == "mtp.attention-adapter.qkv")
            .unwrap();
        assert_eq!(alias.source_layer, Some(0));
        assert!(alias.slices.iter().all(|s| {
            s.source
                .weight
                .codes
                .tensor
                .strip_suffix("@tp2")
                .unwrap_or(&s.source.weight.codes.tensor)
                == "model.mtp.layers.0.self_attn.qkv_proj.weight"
        }));
        let mut missing = admitted.clone();
        missing.remove("model.layers.0.self_attn.qkv_proj.weight");
        assert!(projections::imports::derive(tp, 0, &contract_tensors, &missing).is_err());
    }
}

#[test]
fn tp2_served_ordinary_specs_geometry_and_rank_selections_match_the_current_compiler() {
    ordinary(2, 3876);
}

#[test]
fn tp4_served_ordinary_specs_geometry_and_rank_selections_match_the_current_compiler() {
    ordinary(4, 7752);
}

#[test]
#[ignore = "payload bytes for every projection and rank; about 3 minutes, run with --ignored"]
fn served_ordinary_specs_geometry_and_rank_selections_match_the_current_compiler() {
    for ranks in [2, 4] {
        let coverage = served_ordinary(ranks as usize, Sweep::Every);
        assert_eq!(coverage.keys, declared_keys(ranks, &ORDINARY, 0..u64::MAX));
    }
}

fn served_ordinary(ranks: usize, sweep: Sweep) -> Coverage {
    let mut coverage = Coverage::default();
    let mut m = headers::fixture(include_str!(
        "../../contract/tests/fixtures/projections/mopd.json"
    ));
    headers::owner_scope(&mut m);
    let tensors: BTreeMap<_, _> = m
        .tensors
        .iter()
        .map(|(name, info)| {
            let dtype = stored(&info.dtype);
            let bytes = info.shape.iter().product::<u64>()
                * match dtype {
                    runtime_weights::DType::Bf16 => 2,
                    runtime_weights::DType::F32 => 4,
                    runtime_weights::DType::I64 => 8,
                    _ => 1,
                };
            (
                name.clone(),
                TensorRecord {
                    name: name.clone(),
                    dtype,
                    shape: info.shape.clone(),
                    file: String::new(),
                    file_start: 0,
                    bytes,
                },
            )
        })
        .collect();
    let metadata = precision::export_metadata(Some(&m.quantization_config), None).unwrap();
    let mut roles = precision::text_roles(&V26_FLASH, |i| V26_FLASH.is_global(i as u32));
    roles.extend(precision::mtp_roles(&V26_FLASH));
    let inventory = capabilities::inventory_for(&model_mimo26_contract::geometry::V26_FLASH);
    let inventory = inventory.unwrap();
    let tp = Tp::new(ranks as u32).unwrap();
    let g = projections::geometry(ranks as u32).unwrap();
    let d = projections::declaration(ranks as u32).unwrap();
    let b = headers::bind(&d, &g, &m).unwrap();
    let d = d.expanded_profiles(&g).unwrap();
    let mut payloads: BTreeMap<_, (u64, _)> = BTreeMap::new();
    let recorded = format!("ordinary {} tp{ranks}", sweep.name());
    let mut lines = retired::Lines::default();
    for (id, projection) in [
        ("ga.output", Projection::Output),
        ("swa.output", Projection::Output),
        ("dense.down", Projection::MlpDown),
        ("router", Projection::Router),
        ("embedding", Projection::Embedding),
        ("head", Projection::Head { f32: false }),
        ("mtp.output", Projection::Output),
        ("mtp-mlp.down", Projection::MlpDown),
        ("mtp.eh", Projection::Output),
    ] {
        let p = d.projections.iter().find(|p| p.id == id).unwrap();
        for bound in b.operands.iter().filter(|o| o.projection == id) {
            // All three MTP layers materialize; only source layer zero executes.
            if id.starts_with("mtp") && bound.layer != Some(0) {
                continue;
            }
            let name = &bound.imported.weight.codes.tensor;
            let original = precision::import_role(&metadata, name, &roles[name], &tensors).unwrap();
            assert_eq!(
                original.weight, bound.imported.weight,
                "{id}/{:?}",
                bound.layer
            );
            let format = registry::by_weight(&original.weight).unwrap().id();
            let recipe = p
                .composition
                .as_ref()
                .and_then(|c| c.numerical_contracts.get(&format));
            let scope = projection_spec::bind::BoundDeclaration {
                operands: vec![bound.clone()],
                conditions: b.conditions.clone(),
                layer_aliases: b.layer_aliases.clone(),
                ..Default::default()
            };
            for rank in 0..ranks {
                for (use_index, usage) in p.uses.iter().enumerate() {
                    coverage
                        .keys
                        .insert((bound.layer, id.to_string(), rank as u32, use_index));
                    let projection = if id == "head" {
                        Projection::Head {
                            f32: usage.contract.as_ref().unwrap().output_dtype == DType::F32,
                        }
                    } else {
                        projection
                    };
                    let old = precision::offered_operation(
                        name.strip_suffix(".weight").unwrap().into(),
                        projection,
                        tp,
                        vec![original.weight.clone()],
                        if projection == Projection::Embedding {
                            vec![]
                        } else {
                            vec![original.activation.clone().unwrap()]
                        },
                        None,
                    );
                    let expected =
                        resolve_operation(&old, ActivationPolicy::Bf16, inventory).unwrap();
                    let mut admitted = old.clone();
                    admitted.activations.fill(ActivationSpec::bf16());
                    admitted.arithmetic.weight_operand = DType::Bf16;
                    admitted.arithmetic.decode_order = registry::by_id(format)
                        .unwrap()
                        .decode_order(
                            if projection == Projection::Embedding {
                                KernelFamily::Lookup
                            } else {
                                KernelFamily::Dense
                            },
                            kernel_core::contract::ActivationFormat::Bf16,
                            recipe.map_or(A16Recipe::ScaledWeight, |r| r.a16_recipe),
                        )
                        .unwrap();
                    let imported = ImportedProjection {
                        operation_id: old.id.clone(),
                        target: Selected::TARGET,
                        family: if projection == Projection::Embedding {
                            KernelFamily::Lookup
                        } else {
                            KernelFamily::Dense
                        },
                        format,
                        geometry: GeometryMode::RankLocal,
                        weights: old.weights.clone(),
                        activations: admitted.activations.clone(),
                        arithmetic: admitted.arithmetic.clone(),
                        a16_recipe: recipe.map_or(A16Recipe::ScaledWeight, |r| r.a16_recipe),
                        matrices: vec![SourceMatrix {
                            weight: original.weight.clone(),
                            expert: 0,
                            operand: 0,
                            source_expert: 0,
                            target_rows: 0..original.weight.logical_shape[0],
                            source_rows: 0..original.weight.logical_shape[0],
                            target_k: 0..original.weight.logical_shape[1],
                            source_k: 0..original.weight.logical_shape[1],
                        }],
                        preparation: None,
                    };
                    let derived = ProjectionContext {
                        bound: &scope,
                        source_instance: InstanceKey {
                            projection: id.into(),
                            layer: bound.layer,
                        },
                        selections: None,
                        declaration: &d,
                        projection: p,
                        geometry: &g,
                        topology: Topology {
                            ranks: ranks as u32,
                            rank: rank as u32,
                        },
                        layer: bound.layer,
                        source_layer: bound.layer,
                        usage,
                    }
                    .ordinary(&imported)
                    .unwrap_or_else(|e| panic!("{id}/{:?}/TP{ranks}/rank{rank}: {e}", bound.layer));
                    let mut expected_source = admitted.clone();
                    expected_source.uses = vec![derived.requested];
                    assert_eq!(
                        derived.operation, expected_source,
                        "source {id}/{:?}/TP{ranks}/rank{rank}",
                        bound.layer
                    );
                    let actual =
                        resolve_operation(&derived.operation, ActivationPolicy::Bf16, inventory)
                            .unwrap();
                    let mut expected = expected.selection().effective.clone();
                    expected.uses = vec![derived.requested];
                    assert_eq!(
                        actual.selection().effective,
                        expected,
                        "{id}/{:?}/TP{ranks}/rank{rank}",
                        bound.layer
                    );
                    assert_eq!(
                        derived.operation.matrix_geometry().unwrap(),
                        projection.geometry(tp)
                    );
                    let slice = &derived.slices[0];
                    let image = if id.starts_with("mtp") { "mtp" } else { "text" };
                    let label = format!("{recorded} {image} rank{rank} use{use_index}");
                    lines.add(
                        &label,
                        format!(
                            "slice {id} {:?} rows={:?} k={:?}",
                            bound.layer, slice.rows, slice.k
                        ),
                    );
                    if usage == &p.uses[0] {
                        compare_legacy_records(&derived, &mut lines, &label);
                        let payload_rank = payload_rank(rank as u32, ranks as u32, sweep);
                        if let Some(payload_rank) =
                            payload_rank.filter(|_| !matches!(id, "head" | "embedding"))
                        {
                            let class = match sweep {
                                Sweep::Default => payload_class(&derived),
                                Sweep::Every => format!("{id} {format:?}"),
                            };
                            let bytes = original.weight.codes.byte_length;
                            let key = (class, payload_rank);
                            if payloads.get(&key).is_none_or(|(size, _)| bytes < *size) {
                                payloads.insert(key, (bytes, derived.clone()));
                            }
                        }
                    }
                    coverage.specs += 1;
                    coverage.selections += derived.slices.len();
                }
            }
        }
    }
    for ((_, rank), (_, derived)) in &payloads {
        let (codes, local) = compare_payload(derived);
        lines.add(
            format!("{recorded} payloads"),
            format!(
                "payload {} rank{rank} codes={codes} local={local:?}",
                derived.instance.projection
            ),
        );
        coverage
            .payloads
            .insert((derived.instance.projection.clone(), *rank));
    }
    lines.check(&recorded);
    coverage
}

/// Pack the selected source with the shared builder, check that the served
/// records produce the same sections, and return their digests.
fn compare_payload(
    operation: &projection_spec::derive::DerivedOperation,
) -> (String, Option<String>) {
    let SyntheticPayload {
        payloads,
        packed: actual,
    } = synthetic_payload(operation);
    let weight = &operation.slices[0].source.weight;
    let records = model_compiler_sdk::quantization::projection::records(operation).unwrap();
    for record in records {
        let bytes = record
            .record
            .clone()
            .prepare(record.record.output_bytes().unwrap())
            .unwrap()
            .produce_whole(&mut |index, offset, out| {
                out.copy_from_slice(&payloads[index][offset as usize..offset as usize + out.len()]);
                Ok(())
            })
            .unwrap();
        let expected = match record.section {
            kernel_core::weight_transform::PackedSection::Codes => &actual.codes,
            kernel_core::weight_transform::PackedSection::LocalScales => &actual.local_scales,
            kernel_core::weight_transform::PackedSection::GlobalScales => &actual.global_scales,
        };
        assert_eq!(
            &bytes, expected,
            "persistent payload {}",
            operation.instance.projection
        );
    }
    (
        retired::bytes(&actual.codes),
        weight
            .local_scale
            .as_ref()
            .map(|_| retired::bytes(&actual.local_scales)),
    )
}

fn compare_legacy_records(
    operation: &projection_spec::derive::DerivedOperation,
    lines: &mut retired::Lines,
    label: &str,
) {
    let bound = model_compiler_sdk::quantization::projection::bind_record_policy(
        operation.clone(),
        projection_record_policy(
            registry::by_format(operation.packing.format).unwrap().id(),
            operation.family,
        ),
    )
    .unwrap();
    use kernel_core::weight_transform::PackedSection;
    let records = model_compiler_sdk::quantization::projection::records(&bound).unwrap();
    let weight = &operation.slices[0].source.weight;
    let mut expected_count = 0;
    for (section, span) in [
        (PackedSection::Codes, Some(&weight.codes)),
        (
            PackedSection::LocalScales,
            weight.local_scale.as_ref().map(|s| &s.tensor),
        ),
        (
            PackedSection::GlobalScales,
            weight.global_scale.as_ref().map(|s| &s.tensor),
        ),
    ] {
        let Some(span) = span else {
            continue;
        };
        let actual = records
            .iter()
            .find(|r| r.section == section && r.tensors == [span.tensor.clone()])
            .unwrap_or_else(|| panic!("missing served section {:?} {}", section, span.tensor));
        let view = bound.views.iter().find(|v| v.id == actual.view).unwrap();
        assert_eq!(view.bytes, actual.record.output_bytes().unwrap());
        lines.add(
            label,
            format!(
                "record {} {:?} {section:?} tensors={:?} layout={} view_bytes={} offset={} {}",
                operation.instance.projection,
                operation.instance.layer,
                actual.tensors,
                view.layout,
                view.bytes,
                actual.view_offset,
                retired::record(&actual.record)
            ),
        );
        expected_count += 1;
    }
    lines.add(
        label,
        format!(
            "records {} {:?} {}",
            operation.instance.projection,
            operation.instance.layer,
            records.len()
        ),
    );
    assert_eq!(
        records.len(),
        expected_count,
        "unexpected or missing served sections"
    );
    assert_eq!(
        bound
            .views
            .iter()
            .filter(|v| matches!(v.role, projection_spec::derive::ViewRole::Weight(_)))
            .count(),
        expected_count
    );
}

/// The expert layers of each default bank test; together they cover the bank
/// stem on both degrees, so every layer's OperationSpecs, selections, slices,
/// records and views are compared. Payload bytes run at the first layer.
const BANK_LAYERS: [std::ops::Range<u64>; 3] = [1..17, 17..33, 33..48];

const BANKS: [&str; 2] = ["moe.experts.gate-up", "moe.experts.down"];

#[test]
fn bank_tests_cover_every_expert_layer() {
    for ranks in [2, 4] {
        let covered: BTreeSet<_> = BANK_LAYERS
            .iter()
            .flat_map(|layers| declared_keys(ranks, &BANKS, layers.clone()))
            .collect();
        assert_eq!(covered, declared_keys(ranks, &BANKS, 0..u64::MAX));
        assert_eq!(covered.len(), 47 * 2 * ranks as usize * USES);
    }
}

/// Uses each served projection declares.
const USES: usize = 19;

/// Check one default bank test against its declared keys, exact counts and
/// payload samples: at the first expert layer, the first and last bank member
/// of each section on the first and last rank.
fn banks(ranks: u32, chunk: usize) {
    let layers = BANK_LAYERS[chunk].clone();
    let coverage = served_expert_banks(ranks, layers.clone(), Sweep::Default);
    assert_eq!(coverage.keys, declared_keys(ranks, &BANKS, layers.clone()));
    let count = (layers.end - layers.start) as usize * 2 * ranks as usize * USES;
    assert_eq!(
        (coverage.keys.len(), coverage.specs, coverage.selections),
        (count, count, count / 2 * 256 * 3)
    );
    let mut payloads = BTreeSet::new();
    if chunk == 0 {
        for rank in [0, ranks - 1] {
            for member in ["0.gate_proj", "255.up_proj", "0.down_proj", "255.down_proj"] {
                for section in ["weight", "weight_scale"] {
                    let name = format!("model.layers.1.mlp.experts.{member}.{section}");
                    payloads.insert((name, rank));
                }
            }
        }
    }
    assert_eq!(coverage.payloads, payloads);
}

#[test]
fn tp2_served_expert_banks_match_the_current_compiler_in_layers_1_to_17() {
    banks(2, 0);
}

#[test]
fn tp2_served_expert_banks_match_the_current_compiler_in_layers_17_to_33() {
    banks(2, 1);
}

#[test]
fn tp2_served_expert_banks_match_the_current_compiler_in_layers_33_to_48() {
    banks(2, 2);
}

#[test]
fn tp4_served_expert_banks_match_the_current_compiler_in_layers_1_to_17() {
    banks(4, 0);
}

#[test]
fn tp4_served_expert_banks_match_the_current_compiler_in_layers_17_to_33() {
    banks(4, 1);
}

#[test]
fn tp4_served_expert_banks_match_the_current_compiler_in_layers_33_to_48() {
    banks(4, 2);
}

#[test]
#[ignore = "payload bytes for every bank member and rank; about 4 minutes, run with --ignored"]
fn served_expert_bank_specs_and_every_expert_selection_match_the_current_compiler() {
    for ranks in [2, 4] {
        let coverage = served_expert_banks(ranks, 0..u64::MAX, Sweep::Every);
        assert_eq!(coverage.keys, declared_keys(ranks, &BANKS, 0..u64::MAX));
    }
}

fn served_expert_banks(ranks: u32, layers: std::ops::Range<u64>, sweep: Sweep) -> Coverage {
    use model_compiler_sdk::quantization::bank::concatenate_experts;
    let mut m = headers::fixture(include_str!(
        "../../contract/tests/fixtures/projections/mopd.json"
    ));
    headers::owner_scope(&mut m);
    let tensors: BTreeMap<_, _> = m
        .tensors
        .iter()
        .map(|(name, info)| {
            let dtype = stored(&info.dtype);
            let bits = model_compiler_sdk::quantization::source::source_dtype(dtype)
                .map(dtype_bits)
                .unwrap();
            (
                name.clone(),
                TensorRecord {
                    name: name.clone(),
                    dtype,
                    shape: info.shape.clone(),
                    file: String::new(),
                    file_start: 0,
                    bytes: info.shape.iter().product::<u64>() * bits / 8,
                },
            )
        })
        .collect();
    let metadata = precision::export_metadata(Some(&m.quantization_config), None).unwrap();
    let roles = precision::text_roles(&V26_FLASH, |i| V26_FLASH.is_global(i as u32));
    let inventory =
        capabilities::inventory_for(&model_mimo26_contract::geometry::V26_FLASH).unwrap();
    let mut coverage = Coverage::default();
    let recorded = format!(
        "banks {} tp{ranks} {}",
        sweep.name(),
        if layers.end == u64::MAX {
            "all".to_string()
        } else {
            format!("layers{}-{}", layers.start, layers.end)
        }
    );
    let mut lines = retired::Lines::default();
    let tp = Tp::new(ranks).unwrap();
    let g = projections::geometry(ranks).unwrap();
    let d = projections::declaration(ranks).unwrap();
    let b = headers::bind(&d, &g, &m).unwrap();
    let d = d.expanded_profiles(&g).unwrap();
    for (id, projection) in [
        ("moe.experts.gate-up", Projection::ExpertGateUp),
        ("moe.experts.down", Projection::ExpertDown),
    ] {
        let p = d.projections.iter().find(|p| p.id == id).unwrap();
        let bank_layers: BTreeSet<_> = b
            .operands
            .iter()
            .filter(|o| o.projection == id)
            .map(|o| o.layer)
            .collect();
        let first = *bank_layers.first().unwrap();
        for &layer in bank_layers
            .iter()
            .filter(|l| l.is_some_and(|l| layers.contains(&l)))
        {
            let operands: Vec<_> = b
                .operands
                .iter()
                .filter(|o| o.projection == id && o.layer == layer)
                .cloned()
                .collect();
            let mut banks = Vec::new();
            let mut matrices = Vec::new();
            for operand in 0..p.operand_count() {
                let mut frames: Vec<_> = operands
                    .iter()
                    .filter(|o| o.operand.unwrap_or(0) == u64::from(operand))
                    .collect();
                frames.sort_by_key(|o| o.expert);
                let originals: Vec<_> = frames
                    .iter()
                    .map(|o| {
                        let name = &o.imported.weight.codes.tensor;
                        let source =
                            precision::import_role(&metadata, name, &roles[name], &tensors)
                                .unwrap();
                        assert_eq!(source.weight, o.imported.weight);
                        matrices.push(SourceMatrix {
                            weight: source.weight.clone(),
                            expert: o.expert.unwrap() as u32,
                            operand,
                            source_expert: 0,
                            target_rows: 0..source.weight.logical_shape[0],
                            source_rows: 0..source.weight.logical_shape[0],
                            target_k: 0..source.weight.logical_shape[1],
                            source_k: 0..source.weight.logical_shape[1],
                        });
                        source
                    })
                    .collect();
                banks.push(
                    concatenate_experts(
                        &format!(
                            "model.layers.{}.mlp.experts.{}",
                            layer.unwrap(),
                            projection.operands()[operand as usize]
                        ),
                        &originals,
                    )
                    .unwrap()
                    .operand,
                );
            }
            let id_old = format!(
                "model.layers.{}.mlp.experts.{}",
                layer.unwrap(),
                projection.operands().join("+")
            );
            let old = precision::offered_operation(
                id_old,
                projection,
                tp,
                banks.iter().map(|b| b.weight.clone()).collect(),
                banks
                    .iter()
                    .map(|b| b.activation.clone().unwrap())
                    .collect(),
                Some(V26_FLASH.experts() as usize),
            );
            let expected = resolve_operation(&old, ActivationPolicy::Bf16, inventory).unwrap();
            let mut admitted = old.clone();
            admitted.activations.fill(ActivationSpec::bf16());
            admitted.arithmetic.weight_operand = DType::Bf16;
            admitted.arithmetic.decode_order =
                kernel_core::precision::DecodeOrder::ExactScaledOperand;
            let imported = ImportedProjection {
                operation_id: old.id.clone(),
                target: old.target,
                family: KernelFamily::GroupedExpert,
                format: registry::by_weight(&old.weights[0]).unwrap().id(),
                geometry: GeometryMode::RankLocal,
                weights: old.weights.clone(),
                activations: admitted.activations.clone(),
                arithmetic: admitted.arithmetic.clone(),
                a16_recipe: A16Recipe::ScaledWeight,
                matrices,
                preparation: None,
            };
            let scope = projection_spec::bind::BoundDeclaration {
                operands,
                conditions: b.conditions.clone(),
                ..Default::default()
            };
            for rank in 0..ranks {
                for (use_index, usage) in p.uses.iter().enumerate() {
                    coverage
                        .keys
                        .insert((layer, id.to_string(), rank, use_index));
                    let derived = ProjectionContext {
                        bound: &scope,
                        source_instance: InstanceKey {
                            projection: id.into(),
                            layer,
                        },
                        selections: None,
                        declaration: &d,
                        projection: p,
                        geometry: &g,
                        topology: Topology { ranks, rank },
                        layer,
                        source_layer: layer,
                        usage,
                    }
                    .ordinary(&imported)
                    .unwrap();
                    let mut expected_source = admitted.clone();
                    expected_source.uses = vec![derived.requested];
                    assert_eq!(
                        derived.operation, expected_source,
                        "{id}/{layer:?}/TP{ranks}/rank{rank}"
                    );
                    let actual =
                        resolve_operation(&derived.operation, ActivationPolicy::Bf16, inventory)
                            .unwrap();
                    let mut expected = expected.selection().effective.clone();
                    expected.uses = vec![derived.requested];
                    assert_eq!(actual.selection().effective, expected);
                    assert_eq!(
                        derived.operation.matrix_geometry().unwrap(),
                        projection.geometry(tp)
                    );
                    let label = format!("{recorded} rank{rank} use{use_index}");
                    if usage == &p.uses[0] {
                        let compared = compare_bank_records(
                            &derived,
                            layer == first && payload_rank(rank, ranks, sweep).is_some(),
                            sweep,
                            &mut lines,
                            &label,
                        );
                        coverage
                            .payloads
                            .extend(compared.into_iter().map(|name| (name, rank)));
                    }
                    for slice in &derived.slices {
                        let name = &slice.source.weight.codes.tensor;
                        lines.add(
                            &label,
                            format!("slice {name} rows={:?} k={:?}", slice.rows, slice.k),
                        );
                        coverage.selections += 1;
                    }
                    coverage.specs += 1;
                }
            }
        }
    }
    lines.check(&recorded);
    coverage
}

fn compare_bank_records(
    operation: &projection_spec::derive::DerivedOperation,
    payload: bool,
    sweep: Sweep,
    lines: &mut retired::Lines,
    label: &str,
) -> Vec<String> {
    use kernel_core::weight_transform::PackedSection;
    use model_compiler_sdk::quantization::projection::{bind_record_policy, records};
    let mut compared = Vec::new();
    let bound = bind_record_policy(
        operation.clone(),
        projection_record_policy(
            registry::by_format(operation.packing.format).unwrap().id(),
            operation.family,
        ),
    )
    .unwrap();
    let actual = records(&bound).unwrap();
    let mut count = 0;
    for section in [PackedSection::Codes, PackedSection::LocalScales] {
        let selected: Vec<_> = actual.iter().filter(|r| r.section == section).collect();
        let view = bound
            .views
            .iter()
            .find(|v| v.role == projection_spec::derive::ViewRole::Weight(section))
            .unwrap();
        lines.add(
            label,
            format!(
                "view {} {section:?} layout={} bytes={} pieces={}",
                operation.instance.projection,
                view.layout,
                view.bytes,
                selected.len()
            ),
        );
        let last = selected.len() - 1;
        for (index, record) in selected.iter().enumerate() {
            lines.add(
                label,
                format!(
                    "piece {:?} offset={} {}",
                    record.tensors,
                    record.view_offset,
                    retired::record(&record.record)
                ),
            );
            // Bank members repeat one recipe; the default compares the bytes
            // of the first and last member of each section.
            if payload && (sweep == Sweep::Every || index == 0 || index == last) {
                let bytes = compare_record_bytes(&bound, record, &record.tensors);
                lines.add(label, format!("payload {:?} {bytes}", record.tensors));
                compared.push(record.tensors[0].clone());
            }
            count += 1;
        }
    }
    assert_eq!(count, actual.len());
    compared
}

fn derived_piece_bytes(
    slice: &projection_spec::derive::OperandSlice,
    packing: &kernel_core::precision::layout::PackingPlan,
    section: kernel_core::weight_transform::PackedSection,
) -> Vec<u8> {
    use kernel_core::weight_transform::{Mxfp4Section, PackedSection};
    let matrix = selected_source_payload(slice, section);
    let (data, row_bytes, kind) = match section {
        PackedSection::Codes => (&matrix.codes, packing.k as usize / 2, Mxfp4Section::Codes),
        PackedSection::LocalScales => (
            &matrix.scales,
            packing.k as usize / 32,
            Mxfp4Section::Scales,
        ),
        PackedSection::GlobalScales => panic!("this native bank has no globals"),
    };
    kernel_core::weight_transform::mxfp4::Mxfp4Strips {
        section: kind,
        rows: packing.rows as usize,
        row_bytes,
    }
    .pack(data)
    .unwrap()
}
/// Compare loader execution with separately gathered and packed derivation bytes.
fn compare_record_bytes(
    operation: &projection_spec::derive::DerivedOperation,
    record: &projection_spec::derive::VariantRecord,
    names: &[String],
) -> String {
    let slice = operation
        .slices
        .iter()
        .find(|s| {
            s.source.weight.codes.tensor == names[0]
                || s.source
                    .weight
                    .local_scale
                    .as_ref()
                    .is_some_and(|g| g.tensor.tensor == names[0])
                || s.source
                    .weight
                    .global_scale
                    .as_ref()
                    .is_some_and(|g| g.tensor.tensor == names[0])
        })
        .unwrap();
    let actual = derived_piece_bytes(slice, &operation.packing, record.section);
    assert_eq!(actual.len() as u64, record.record.output_bytes().unwrap());
    let weight = &slice.source.weight;
    let sources = std::iter::once(&weight.codes)
        .chain(weight.local_scale.iter().map(|s| &s.tensor))
        .chain(weight.global_scale.iter().map(|s| &s.tensor))
        .collect::<Vec<_>>();
    let sources = names
        .iter()
        .map(|name| *sources.iter().find(|s| s.tensor == *name).unwrap())
        .collect::<Vec<_>>();
    let expected = record
        .record
        .clone()
        .prepare(record.record.output_bytes().unwrap())
        .unwrap()
        .produce_whole(&mut |index, offset, out| {
            fill_source(sources[index], offset, out);
            Ok(())
        })
        .unwrap();
    assert_eq!(actual, expected, "independent synthetic payload {names:?}");
    retired::bytes(&actual)
}

#[test]
fn unbaselined_dflash_headers_are_bf16_without_quantization_companions() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../contract/tests/fixtures/projections/mopd.json"
    ))
    .unwrap();
    let mut matrices = 0;
    let mut sides = 0;
    for family in fixture["families"].as_array().unwrap() {
        let name = family["tensor"].as_str().unwrap();
        if !name.starts_with("layers.")
            && !matches!(name, "fc.weight" | "hidden_norm.weight" | "norm.weight")
        {
            continue;
        }
        assert_eq!(family["dtype"], "BF16", "{name}");
        assert!(!name.contains("scale"));
        let count: u64 = family["axes"]
            .as_object()
            .unwrap()
            .values()
            .map(|axis| {
                assert_eq!(axis["range"], serde_json::json!([0, 5]));
                5
            })
            .product();
        if family["shape"].as_array().unwrap().len() == 2 {
            matrices += count;
        } else {
            sides += count;
        }
    }
    assert_eq!((matrices, sides), (36, 27));
}

#[test]
fn ordinary_block_scale_policy_selects_the_served_plugin_layout_variant() {
    use kernel_core::precision::format::{FormatId, LayoutVariant};
    use model_compiler_sdk::quantization::projection::RecordPolicy;
    let policy = projection_record_policy(FormatId::Fp8Block128, KernelFamily::Dense);
    let selected = match policy {
        RecordPolicy::SectionsWithLayout { layout, .. } => layout,
        RecordPolicy::Sections(_) => LayoutVariant::Standard,
        _ => panic!("expected legacy section policy"),
    };
    assert_eq!(selected, LayoutVariant::AlternateBlockScaleNames);
}

/// Persistent encoding of the served loader, independent of arithmetic binding.
fn projection_record_policy(
    format: kernel_core::precision::format::FormatId,
    family: kernel_core::precision::format::KernelFamily,
) -> model_compiler_sdk::quantization::projection::RecordPolicy {
    use kernel_core::precision::format::{FormatId, KernelFamily};
    use model_compiler_sdk::quantization::{format::RecordLayout, projection::RecordPolicy};
    if format == FormatId::Fp8Block128 && family == KernelFamily::Dense {
        return RecordPolicy::SectionsWithLayout {
            records: RecordLayout::Fragment,
            layout: kernel_core::precision::format::LayoutVariant::AlternateBlockScaleNames,
        };
    }
    RecordPolicy::Sections(if family == KernelFamily::Lookup {
        RecordLayout::Rows
    } else if format == FormatId::Mxfp4 && family == KernelFamily::GroupedExpert {
        RecordLayout::Mxfp4Strips
    } else {
        RecordLayout::Fragment
    })
}
