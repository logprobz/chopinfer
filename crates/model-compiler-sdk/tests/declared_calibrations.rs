//! Static input calibrations of admitted operands are claimed through
//! `AdmittedRules` with their frame provenance: a scalar calibration, one
//! scalar per expert of a stacked bank shared by its fused operands, and one
//! calibration tensor shared by two sources.
use std::collections::BTreeMap;

use kernel_core::contract::DType;
use kernel_core::precision::TensorSpan;
use model_compiler_sdk::quantization::declared::{self, Admitted, AdmittedRules, RankInputs};
use model_compiler_sdk::quantization::import::{ExportMetadata, OperandRole, import_operand};
use projection_spec::bind::{BoundDeclaration, CheckpointMetadata, TensorInfo};
use projection_spec::derive::{Topology, hooks::NoVariants};
use projection_spec::registry::Registry;
use projection_spec::{Declaration, Geometry, parse};

const EPILOGUES: &str = r#"{"residual_add": false, "bias": false, "gelu": null, "swiglu": null, "output_quantization": null, "weighted_moe_combine": false}"#;

fn declaration(with_bank: bool) -> String {
    let dense = |id: &str, tensor: &str| {
        format!(
            r#"{{"id": "{id}", "source": {{"tensor": "{tensor}"}}, "shape": [32, 32],
             "kind": {{"kind": "dense"}}, "split": {{"class": "replicate"}},
             "epilogues": {EPILOGUES}, "numerical_contract": "a16.f32.v1", "use_profile": "served"}}"#
        )
    };
    let mut projections = vec![
        dense("dense", "dense.weight"),
        dense("first", "shared.a.weight"),
        dense("second", "shared.b.weight"),
    ];
    if with_bank {
        projections.push(format!(
            r#"{{"id": "bank", "source": {{"tensor": "bank.weight"}}, "shape": [32, 32],
             "operands": ["gate", "up"], "kind": {{"kind": "expert_bank", "experts": 2, "top_k": 1}},
             "split": {{"class": "expert_owned", "tile": 1}},
             "epilogues": {}, "numerical_contract": "a16.f32.v1", "use_profile": "served"}}"#,
            EPILOGUES.replace(r#""swiglu": null"#, r#""swiglu": "ConcatenatedHalves""#)
        ));
    }
    format!(
        r#"{{"schema": "chopinfer.projection.v1", "namespace": "fixture",
          "use_profiles": {{"served": [{{"phase": "Decode", "capacity": 1, "graph": false}}]}},
          "layer_stems": [], "projections": [{}], "fusion_groups": [], "side_tensors": []}}"#,
        projections.join(",")
    )
}

fn span(tensor: &str, dtype: DType, shape: Vec<u64>) -> TensorSpan {
    let elements: u64 = shape.iter().product();
    let bytes = match dtype {
        DType::F32 => elements * 4,
        _ => elements,
    };
    TensorSpan {
        tensor: tensor.into(),
        dtype,
        shape,
        offset: 0,
        byte_length: bytes,
    }
}

/// An NVFP4 source with codes, block scales, globals and input scales;
/// `experts` adds a leading bank axis with one global and calibration per
/// expert.
fn nvfp4(stem: &str, rows: u64, experts: Option<u64>) -> Vec<TensorSpan> {
    let bank = |mut shape: Vec<u64>| {
        if let Some(experts) = experts {
            shape.insert(0, experts);
        }
        shape
    };
    let scalar = experts.map_or(vec![], |e| vec![e]);
    vec![
        span(&format!("{stem}.weight"), DType::U8, bank(vec![rows, 16])),
        span(
            &format!("{stem}.weight_scale"),
            DType::Fp8E4M3,
            bank(vec![rows, 2]),
        ),
        span(
            &format!("{stem}.weight_scale_2"),
            DType::F32,
            scalar.clone(),
        ),
        span(&format!("{stem}.input_scale"), DType::F32, scalar),
    ]
}

struct Fixture {
    metadata: CheckpointMetadata,
    admitted: Admitted,
}

fn fixture(with_bank: bool, bank_calibrations: u64) -> Fixture {
    let mut stems = vec!["dense", "shared.a", "shared.b"];
    if with_bank {
        stems.push("bank");
    }
    let layers: BTreeMap<_, _> = stems
        .iter()
        .map(|s| {
            (
                *s,
                serde_json::json!({"quant_algo": "NVFP4", "group_size": 16}),
            )
        })
        .collect();
    let document = serde_json::json!({
        "producer": {"name": "modelopt", "version": "fixture"},
        "quantization": {"quant_algo": "MIXED_PRECISION", "group_size": 16,
                         "kv_cache_quant_algo": null, "quantized_layers": layers}
    });
    let mut export = ExportMetadata::default();
    export
        .add_document(
            "hf_quant_config.json",
            &serde_json::to_vec(&document).unwrap(),
        )
        .unwrap();
    let mut tensors = BTreeMap::new();
    for stem in ["dense", "shared.a"] {
        for t in nvfp4(stem, 32, None) {
            tensors.insert(t.tensor.clone(), t);
        }
    }
    // The second shared source has no calibration of its own: it reads the
    // first source's input scale.
    for t in nvfp4("shared.b", 32, None) {
        if !t.tensor.ends_with("input_scale") {
            tensors.insert(t.tensor.clone(), t);
        }
    }
    if with_bank {
        for mut t in nvfp4("bank", 64, Some(2)) {
            if t.tensor.ends_with("input_scale") {
                t = span(&t.tensor, DType::F32, vec![bank_calibrations]);
            }
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
    let mut admit = |stem: &str, shape: Vec<u64>, inventory: &BTreeMap<String, TensorSpan>| {
        let weight = format!("{stem}.weight");
        let operand = import_operand(
            &export,
            &OperandRole {
                operation: stem.into(),
                weight: weight.clone(),
                targets: vec![stem.into()],
                logical_shape: shape,
                lookup: false,
                shared_scale_stem: None,
                declared: None,
            },
            inventory,
        )
        .unwrap();
        admitted.insert(weight, operand);
    };
    admit("dense", vec![32, 32], &tensors);
    admit("shared.a", vec![32, 32], &tensors);
    let mut own = tensors.clone();
    own.insert(
        "shared.b.input_scale".into(),
        span("shared.b.input_scale", DType::F32, vec![]),
    );
    admit("shared.b", vec![32, 32], &own);
    if with_bank {
        let mut inventory = tensors.clone();
        // The importer checks a per-expert calibration's cardinality; admit
        // the bank against well-formed metadata and bind against `metadata`.
        inventory.insert(
            "bank.input_scale".into(),
            span("bank.input_scale", DType::F32, vec![2]),
        );
        admit("bank", vec![2, 64, 32], &inventory);
    }
    // The caller admits the second source with the first one's calibration.
    let shared = tensors["shared.a.input_scale"].clone();
    let second = admitted.get_mut("shared.b.weight").unwrap();
    let quantizer = second
        .activation
        .as_mut()
        .and_then(|a| a.quantizer.as_mut())
        .unwrap();
    let kernel_core::precision::Calibration::Static(scale) = &mut quantizer.calibration else {
        panic!("static calibration");
    };
    scale.tensor = shared.clone();
    for name in &mut second.source_tensors {
        if name == "shared.b.input_scale" {
            name.clone_from(&shared.tensor);
        }
    }
    Fixture { metadata, admitted }
}

fn bind(with_bank: bool, bank_calibrations: u64) -> Result<BoundDeclaration, String> {
    let g = Geometry::new();
    let registry = Registry::default();
    let declaration: Declaration = parse(&declaration(with_bank), &g, &registry)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let f = fixture(with_bank, bank_calibrations);
    declaration
        .bind(&g, &registry, &f.metadata, &AdmittedRules(&f.admitted))
        .map_err(|e| e.to_string())
}

fn calibrations(bound: &BoundDeclaration, projection: &str) -> Vec<(u64, u64, String, u64)> {
    bound
        .operands
        .iter()
        .filter(|b| b.projection == projection)
        .flat_map(|b| &b.imported.calibrations)
        .map(|c| {
            (
                c.expert,
                c.operand,
                c.scale.tensor.tensor.clone(),
                c.scale.tensor.offset,
            )
        })
        .collect()
}

#[test]
fn scalar_per_expert_and_shared_calibrations_are_claimed_with_their_frames() {
    let bound = bind(true, 2).unwrap();
    assert_eq!(
        calibrations(&bound, "dense"),
        [(0, 0, "dense.input_scale".to_string(), 0)]
    );
    // Both fused operands of each expert read that expert's scalar.
    let bank = |e: u64, o: u64| (e, o, "bank.input_scale".to_string(), 4 * e);
    assert_eq!(
        calibrations(&bound, "bank"),
        [bank(0, 0), bank(0, 1), bank(1, 0), bank(1, 1)]
    );
    for projection in ["first", "second"] {
        assert_eq!(
            calibrations(&bound, projection),
            [(0, 0, "shared.a.input_scale".to_string(), 0)]
        );
    }
    let companions: Vec<_> = bound
        .operands
        .iter()
        .flat_map(|b| b.imported.companions.iter().map(|c| c.tensor.as_str()))
        .collect();
    assert_eq!(
        companions,
        [
            "dense.input_scale",
            "shared.a.input_scale",
            "shared.a.input_scale",
            "bank.input_scale"
        ]
    );
}

#[test]
fn a_calibration_vector_that_matches_no_frame_is_refused() {
    let error = bind(true, 3).unwrap_err();
    assert!(error.contains("bank.input_scale"), "{error}");
}

#[test]
fn admitted_static_calibrations_derive_through_the_shared_adapter() {
    let g = Geometry::new();
    let registry = Registry::default();
    let declaration = parse(&declaration(false), &g, &registry)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let f = fixture(false, 0);
    let graph = declared::derive(RankInputs {
        declaration: &declaration,
        geometry: &g,
        registry: &registry,
        variants: &NoVariants,
        metadata: &f.metadata,
        admitted: &f.admitted,
        topology: Topology { ranks: 1, rank: 0 },
    })
    .unwrap();
    assert_eq!(graph.operations.len(), 3);
}

/// Binds the fixture after `edit` changes the dense source's admitted
/// calibration span.
fn bind_edited(edit: impl FnOnce(&mut TensorSpan)) -> Result<BoundDeclaration, String> {
    let g = Geometry::new();
    let registry = Registry::default();
    let declaration = parse(&declaration(false), &g, &registry)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let mut f = fixture(false, 0);
    let quantizer = f
        .admitted
        .get_mut("dense.weight")
        .unwrap()
        .activation
        .as_mut()
        .and_then(|a| a.quantizer.as_mut())
        .unwrap();
    let kernel_core::precision::Calibration::Static(scale) = &mut quantizer.calibration else {
        panic!("static calibration");
    };
    edit(&mut scale.tensor);
    declaration
        .bind(&g, &registry, &f.metadata, &AdmittedRules(&f.admitted))
        .map_err(|e| e.to_string())
}

#[test]
fn an_admitted_calibration_span_must_be_its_whole_checkpoint_tensor() {
    assert!(bind_edited(|_| {}).is_ok());
    for (case, edit) in [
        (
            "offset",
            Box::new(|s: &mut TensorSpan| s.offset = 4) as Box<dyn FnOnce(&mut TensorSpan)>,
        ),
        ("length", Box::new(|s: &mut TensorSpan| s.byte_length = 8)),
        (
            "dtype",
            Box::new(|s: &mut TensorSpan| s.dtype = DType::Bf16),
        ),
        ("shape", Box::new(|s: &mut TensorSpan| s.shape = vec![1])),
    ] {
        let Err(error) = bind_edited(edit) else {
            panic!("{case}: an inexact admitted calibration span was bound");
        };
        assert!(error.contains("dense.input_scale"), "{case}: {error}");
    }
}

/// Binds the fixture without bank after `edit` changes it.
fn bind_fixture(edit: impl FnOnce(&mut Fixture)) -> Result<BoundDeclaration, String> {
    let g = Geometry::new();
    let registry = Registry::default();
    let declaration = parse(&declaration(false), &g, &registry)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let mut f = fixture(false, 0);
    edit(&mut f);
    declaration
        .bind(&g, &registry, &f.metadata, &AdmittedRules(&f.admitted))
        .map_err(|e| e.to_string())
}

/// The span of the static calibration `weight`'s admitted activation reads.
fn calibration_span<'a>(f: &'a mut Fixture, weight: &str) -> &'a mut TensorSpan {
    let quantizer = f
        .admitted
        .get_mut(weight)
        .unwrap()
        .activation
        .as_mut()
        .and_then(|a| a.quantizer.as_mut())
        .unwrap();
    let kernel_core::precision::Calibration::Static(scale) = &mut quantizer.calibration else {
        panic!("static calibration");
    };
    &mut scale.tensor
}

#[test]
fn a_calibration_left_out_of_the_shown_metadata_is_not_claimed() {
    let bound = bind_fixture(|f| {
        f.metadata.tensors.remove("dense.input_scale");
    })
    .unwrap();
    assert_eq!(calibrations(&bound, "dense"), []);
    let companions: Vec<_> = bound
        .operands
        .iter()
        .flat_map(|b| b.imported.companions.iter().map(|c| c.tensor.as_str()))
        .collect();
    assert_eq!(companions, ["shared.a.input_scale", "shared.a.input_scale"]);
    // A calibration the caller shows is still claimed with its frames.
    for projection in ["first", "second"] {
        assert_eq!(
            calibrations(&bound, projection),
            [(0, 0, "shared.a.input_scale".to_string(), 0)]
        );
    }
    // And still authenticated as its whole checkpoint tensor.
    let error = bind_fixture(|f| {
        f.metadata.tensors.remove("dense.input_scale");
        calibration_span(f, "shared.a.weight").offset = 4;
    })
    .unwrap_err();
    assert!(error.contains("shared.a.input_scale"), "{error}");
}
