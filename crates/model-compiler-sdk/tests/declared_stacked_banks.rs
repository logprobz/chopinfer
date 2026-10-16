//! One stacked fused expert-bank source, `[experts, operands * rows, k]`,
//! imported and derived through the shared adapter: implicit expert and
//! operand coordinates, gate and up row offsets, bank-shaped spans, scalar
//! and per-expert globals and calibrations, and expert-owned and
//! expert-sharded rank selections.
use std::collections::BTreeMap;

use kernel_core::contract::{ActivationFormat, DType};
use kernel_core::precision::{OperationKind, TensorSpan};
use model_compiler_sdk::quantization::declared::{self, Admitted, AdmittedRules, RankInputs};
use model_compiler_sdk::quantization::import::{ExportMetadata, OperandRole, import_operand};
use model_compiler_sdk::quantization::projection::records;
use projection_spec::bind::{CheckpointMetadata, TensorInfo};
use projection_spec::derive::{InstanceKey, RankDerivation, Topology, hooks::NoVariants};
use projection_spec::registry::Registry;
use projection_spec::{Declaration, Geometry, parse};

const EXPERTS: u64 = 4;
const ROWS: u64 = 128;
const K: u64 = 128;

fn declaration(split: &str) -> Declaration {
    let text = format!(
        r#"{{"schema": "chopinfer.projection.v1", "namespace": "fixture",
          "use_profiles": {{"served": [{{"phase": "Decode", "capacity": 1, "graph": false}}]}},
          "layer_stems": [],
          "projections": [{{"id": "pair", "source": {{"tensor": "pair.weight"}}, "shape": [{ROWS}, {K}],
            "operands": ["gate", "up"], "kind": {{"kind": "expert_bank", "experts": {EXPERTS}, "top_k": 2}},
            "split": {split},
            "epilogues": {{"residual_add": false, "bias": false, "gelu": null, "swiglu": "ConcatenatedHalves", "output_quantization": null, "weighted_moe_combine": false}},
            "numerical_contract": "a16.f32.v1", "use_profile": "served"}}],
          "fusion_groups": [], "side_tensors": []}}"#
    );
    let g = Geometry::new();
    parse(&text, &g, &Registry::default())
        .unwrap()
        .expanded_profiles(&g)
        .unwrap()
}

fn span(tensor: &str, dtype: DType, shape: Vec<u64>) -> TensorSpan {
    let elements: u64 = shape.iter().product();
    TensorSpan {
        tensor: tensor.into(),
        dtype,
        shape,
        offset: 0,
        byte_length: if dtype == DType::F32 {
            elements * 4
        } else {
            elements
        },
    }
}

/// The stacked NVFP4 bank, with one global and one calibration per expert
/// when `per_expert` holds and single scalars otherwise.
fn sources(per_expert: bool) -> (CheckpointMetadata, Admitted) {
    let scalars = if per_expert { vec![EXPERTS] } else { vec![] };
    let stacked = 2 * ROWS;
    let tensors: BTreeMap<_, _> = [
        span("pair.weight", DType::U8, vec![EXPERTS, stacked, K / 2]),
        span(
            "pair.weight_scale",
            DType::Fp8E4M3,
            vec![EXPERTS, stacked, K / 16],
        ),
        span("pair.weight_scale_2", DType::F32, scalars.clone()),
        span("pair.input_scale", DType::F32, scalars),
    ]
    .into_iter()
    .map(|t| (t.tensor.clone(), t))
    .collect();
    let document = serde_json::json!({
        "producer": {"name": "modelopt", "version": "fixture"},
        "quantization": {"quant_algo": "MIXED_PRECISION", "group_size": 16, "kv_cache_quant_algo": null,
                         "quantized_layers": {"pair": {"quant_algo": "NVFP4", "group_size": 16}}}
    });
    let mut export = ExportMetadata::default();
    export
        .add_document(
            "hf_quant_config.json",
            &serde_json::to_vec(&document).unwrap(),
        )
        .unwrap();
    let operand = import_operand(
        &export,
        &OperandRole {
            operation: "pair".into(),
            weight: "pair.weight".into(),
            targets: vec!["pair".into()],
            logical_shape: vec![EXPERTS, stacked, K],
            lookup: false,
            shared_scale_stem: None,
            declared: None,
        },
        &tensors,
    )
    .unwrap();
    let metadata = CheckpointMetadata {
        tensors: tensors
            .values()
            .map(|t| {
                (
                    t.tensor.clone(),
                    TensorInfo {
                        dtype: t.dtype.checkpoint_dtype().unwrap().into(),
                        shape: t.shape.clone(),
                        file: Some("model.safetensors".into()),
                    },
                )
            })
            .collect(),
        ..Default::default()
    };
    (metadata, Admitted::from([("pair.weight".into(), operand)]))
}

fn derive(split: &str, per_expert: bool, topology: Topology) -> RankDerivation {
    let declaration = declaration(split);
    let (metadata, admitted) = sources(per_expert);
    declared::derive(RankInputs {
        declaration: &declaration,
        geometry: &Geometry::new(),
        registry: &Registry::default(),
        variants: &NoVariants,
        metadata: &metadata,
        admitted: &admitted,
        topology,
    })
    .unwrap()
}

/// The single row range of a slice.
fn one(rows: &[std::ops::Range<u64>]) -> std::ops::Range<u64> {
    let [range] = rows else {
        panic!("one contiguous row range");
    };
    range.clone()
}

const OWNED: &str = r#"{"class": "expert_owned", "tile": 1}"#;

#[test]
fn a_stacked_bank_imports_every_expert_and_operand_in_place() {
    for per_expert in [false, true] {
        let declaration = declaration(OWNED);
        let (metadata, admitted) = sources(per_expert);
        let g = Geometry::new();
        let bound = declaration
            .bind(
                &g,
                &Registry::default(),
                &metadata,
                &AdmittedRules(&admitted),
            )
            .unwrap();
        let imported = declared::imported(&declaration, &bound, &g, &admitted).unwrap();
        let pair = &imported[&InstanceKey {
            projection: "pair".into(),
            layer: None,
        }];
        let source = &admitted["pair.weight"].weight;
        assert_eq!(pair.weights, [source.clone(), source.clone()]);
        let frames: Vec<_> = pair
            .matrices
            .iter()
            .map(|m| {
                assert_eq!(m.weight, *source);
                assert_eq!((m.target_rows.clone(), m.source_k.clone()), (0..ROWS, 0..K));
                (m.expert, m.operand, m.source_expert, m.source_rows.clone())
            })
            .collect();
        let expected: Vec<_> = (0..EXPERTS)
            .flat_map(|e| {
                (0..2).map(move |o| {
                    (
                        e as u32,
                        o as u32,
                        e,
                        o as u64 * ROWS..(o as u64 + 1) * ROWS,
                    )
                })
            })
            .collect();
        assert_eq!(frames, expected);
        // Each expert's calibration covers both fused operands.
        let calibrations: Vec<_> = bound.operands[0]
            .imported
            .calibrations
            .iter()
            .map(|c| (c.expert, c.operand, c.scale.tensor.offset))
            .collect();
        let offset = |e: u64| if per_expert { 4 * e } else { 0 };
        let expected: Vec<_> = (0..EXPERTS)
            .flat_map(|e| (0..2).map(move |o| (e, o, offset(e))))
            .collect();
        assert_eq!(calibrations, expected);
    }
}

#[test]
fn a_stacked_bank_derives_owned_experts_with_operand_row_offsets() {
    for per_expert in [false, true] {
        for rank in 0..2 {
            let graph = derive(OWNED, per_expert, Topology { ranks: 2, rank });
            let [operation] = &graph.operations[..] else {
                panic!("one operation per use");
            };
            assert_eq!(
                operation.operation.kind,
                OperationKind::Grouped {
                    experts: 2,
                    top_k: 2
                }
            );
            let mut slices: Vec<_> = operation
                .slices
                .iter()
                .map(|s| (s.source.source_expert, s.operand, one(&s.rows), s.k.clone()))
                .collect();
            slices.sort_by_key(|s| (s.0, s.1));
            let owned = 2 * u64::from(rank);
            let expected: Vec<_> = (owned..owned + 2)
                .flat_map(|e| {
                    (0..2u32).map(move |o| {
                        let start = u64::from(o) * ROWS;
                        (e, o, start..start + ROWS, 0..K)
                    })
                })
                .collect();
            assert_eq!(slices, expected);
            assert_eq!(records(operation).unwrap().len(), 3);
        }
    }
}

#[test]
fn a_stacked_bank_derives_sharded_intermediate_rows_of_every_expert() {
    let split = r#"{"class": "expert_sharded", "axis": "rows", "tile": 16}"#;
    for rank in 0..2u64 {
        let graph = derive(
            split,
            true,
            Topology {
                ranks: 2,
                rank: rank as u32,
            },
        );
        let operation = &graph.operations[0];
        let mut slices: Vec<_> = operation
            .slices
            .iter()
            .map(|s| (s.source.source_expert, s.operand, one(&s.rows)))
            .collect();
        slices.sort_by_key(|s| (s.0, s.1));
        let half = ROWS / 2;
        let expected: Vec<_> = (0..EXPERTS)
            .flat_map(|e| {
                (0..2u32).map(move |o| {
                    let start = u64::from(o) * ROWS + rank * half;
                    (e, o, start..start + half)
                })
            })
            .collect();
        assert_eq!(slices, expected);
        assert_eq!(records(operation).unwrap().len(), 3);
    }
}

#[test]
fn admitted_activations_apply_one_stacked_source_to_both_operands() {
    let declaration = declaration(OWNED);
    let (_, admitted) = sources(true);
    let graph = derive(OWNED, true, Topology { ranks: 1, rank: 0 });
    let mut spec = graph.operation_specs().unwrap().remove(0);
    declared::admitted_activations(&mut spec, &admitted, &declaration, &Geometry::new()).unwrap();
    assert_eq!(spec.activations.len(), 2);
    assert!(
        spec.activations
            .iter()
            .all(|a| a.format == ActivationFormat::Nvfp4)
    );
}

/// Separate stacked gate and up sources (`pair.{operand}.weight`, each
/// `[experts, rows, k]`) with distinct scalar globals: the packed layout must
/// keep one global per expert and operand, and executing the records must
/// write each operand's own global.
#[test]
fn separate_stacked_operands_with_distinct_globals_pack_their_own_globals() {
    let text = format!(
        r#"{{"schema": "chopinfer.projection.v1", "namespace": "fixture",
          "use_profiles": {{"served": [{{"phase": "Decode", "capacity": 1, "graph": false}}]}},
          "layer_stems": [],
          "projections": [{{"id": "pair", "source": {{"tensor": "pair.{{operand}}.weight"}}, "shape": [{ROWS}, {K}],
            "operands": ["gate", "up"], "kind": {{"kind": "expert_bank", "experts": {EXPERTS}, "top_k": 2}},
            "split": {OWNED},
            "epilogues": {{"residual_add": false, "bias": false, "gelu": null, "swiglu": "ConcatenatedHalves", "output_quantization": null, "weighted_moe_combine": false}},
            "numerical_contract": "a16.f32.v1", "use_profile": "served"}}],
          "fusion_groups": [], "side_tensors": []}}"#
    );
    let g = Geometry::new();
    let declaration = parse(&text, &g, &Registry::default())
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let document = serde_json::json!({
        "producer": {"name": "modelopt", "version": "fixture"},
        "quantization": {"quant_algo": "MIXED_PRECISION", "group_size": 16, "kv_cache_quant_algo": null,
                         "quantized_layers": {"pair.gate": {"quant_algo": "NVFP4", "group_size": 16},
                                              "pair.up": {"quant_algo": "NVFP4", "group_size": 16}}}
    });
    let mut export = ExportMetadata::default();
    export
        .add_document(
            "hf_quant_config.json",
            &serde_json::to_vec(&document).unwrap(),
        )
        .unwrap();
    let mut tensors = BTreeMap::new();
    let mut payloads = BTreeMap::new();
    for (operand, global) in [("gate", 1.0_f32), ("up", 2.0)] {
        let stem = format!("pair.{operand}");
        for (t, bytes) in [
            (
                span(
                    &format!("{stem}.weight"),
                    DType::U8,
                    vec![EXPERTS, ROWS, K / 2],
                ),
                0x21,
            ),
            (
                span(
                    &format!("{stem}.weight_scale"),
                    DType::Fp8E4M3,
                    vec![EXPERTS, ROWS, K / 16],
                ),
                0x38,
            ),
        ] {
            payloads.insert(t.tensor.clone(), vec![bytes; t.byte_length as usize]);
            tensors.insert(t.tensor.clone(), t);
        }
        for (name, value) in [("weight_scale_2", global), ("input_scale", 1.0)] {
            let t = span(&format!("{stem}.{name}"), DType::F32, vec![]);
            payloads.insert(t.tensor.clone(), value.to_le_bytes().to_vec());
            tensors.insert(t.tensor.clone(), t);
        }
    }
    let metadata = CheckpointMetadata {
        tensors: tensors
            .values()
            .map(|t| {
                (
                    t.tensor.clone(),
                    TensorInfo {
                        dtype: t.dtype.checkpoint_dtype().unwrap().into(),
                        shape: t.shape.clone(),
                        file: Some("model.safetensors".into()),
                    },
                )
            })
            .collect(),
        ..Default::default()
    };
    let mut admitted = Admitted::new();
    for operand in ["gate", "up"] {
        let weight = format!("pair.{operand}.weight");
        let imported = import_operand(
            &export,
            &OperandRole {
                operation: "pair".into(),
                weight: weight.clone(),
                targets: vec![format!("pair.{operand}")],
                logical_shape: vec![EXPERTS, ROWS, K],
                lookup: false,
                shared_scale_stem: None,
                declared: None,
            },
            &tensors,
        )
        .unwrap();
        admitted.insert(weight, imported);
    }
    let graph = declared::derive(RankInputs {
        declaration: &declaration,
        geometry: &g,
        registry: &Registry::default(),
        variants: &NoVariants,
        metadata: &metadata,
        admitted: &admitted,
        topology: Topology { ranks: 1, rank: 0 },
    })
    .unwrap();
    let operation = &graph.operations[0];
    assert_eq!(operation.packing.layout.id, "moe-packed-ExpertProjection");
    for record in records(operation).unwrap() {
        let bytes = record.record.output_bytes().unwrap();
        let produced = record
            .record
            .clone()
            .prepare(bytes)
            .unwrap()
            .produce_whole(&mut |index, offset, buffer: &mut [u8]| {
                let source = &payloads[&record.tensors[index]];
                let start = offset as usize;
                buffer.copy_from_slice(&source[start..start + buffer.len()]);
                Ok(())
            })
            .unwrap();
        if record.section == kernel_core::weight_transform::PackedSection::GlobalScales {
            let globals: Vec<f32> = produced
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let expected: Vec<f32> = (0..EXPERTS).flat_map(|_| [1.0, 2.0]).collect();
            assert_eq!(globals, expected);
        }
    }
}

/// A source-less assembled bank over the stacked bank restores the
/// activation recipe of the member it materializes.
#[test]
fn an_assembly_over_a_stacked_bank_restores_its_member_activations() {
    let mut value = serde_json::to_value(declaration(OWNED)).unwrap();
    let mut assembled = value["projections"][0].clone();
    assembled["id"] = "assembled".into();
    assembled["source"] = serde_json::Value::Null;
    assembled["assembly"] = serde_json::json!([{
        "projection": "pair", "operand": null, "rows": {"start": 0, "length": ROWS},
        "coordinates": "global", "inherit_split": true
    }]);
    value["projections"].as_array_mut().unwrap().push(assembled);
    let g = Geometry::new();
    let registry = Registry::default();
    let declaration = parse(&value.to_string(), &g, &registry)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let (metadata, admitted) = sources(false);
    let graph = declared::derive(RankInputs {
        declaration: &declaration,
        geometry: &g,
        registry: &registry,
        variants: &NoVariants,
        metadata: &metadata,
        admitted: &admitted,
        topology: Topology { ranks: 1, rank: 0 },
    })
    .unwrap();
    let mut spec = graph
        .operation_specs()
        .unwrap()
        .into_iter()
        .find(|o| o.id.starts_with("assembled."))
        .unwrap();
    declared::admitted_activations(&mut spec, &admitted, &declaration, &g).unwrap();
    assert_eq!(
        spec.activations,
        admitted["pair.weight"]
            .activation
            .clone()
            .into_iter()
            .cycle()
            .take(2)
            .collect::<Vec<_>>()
    );
}
