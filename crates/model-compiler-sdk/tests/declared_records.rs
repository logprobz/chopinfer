//! Rank records of a derivation over admitted operands: an assembly that
//! tiles its destination from several sources, an unstemmed assembly, and
//! the rank copies of side tensors.
use std::collections::BTreeMap;

use kernel_core::contract::{DType, WeightFormat};
use kernel_core::precision::{ActivationSpec, CodeEncoding, TensorSpan, WeightLayout, WeightSpec};
use kernel_core::weight_transform::{WeightRecord, WeightTransform};
use model_compiler_sdk::quantization::declared::{self, Admitted, AdmittedRules};
use model_compiler_sdk::quantization::format::RecordLayout;
use model_compiler_sdk::quantization::import::ImportedOperand;
use model_compiler_sdk::quantization::projection::{
    RecordPolicy, bind_record_policy, records, side_records,
};
use projection_spec::bind::{CheckpointMetadata, TensorInfo};
use projection_spec::derive::{DerivationInput, Topology, hooks::NoVariants};
use projection_spec::registry::Registry;
use projection_spec::{Geometry, parse};

const DECLARATION: &str = r#"{
  "schema": "chopinfer.projection.v1",
  "namespace": "fixture",
  "use_profiles": {"served": [{"phase": "Decode", "capacity": 1, "graph": true}]},
  "layer_stems": [
    {"id": "block", "template": "layers.{i}", "indices": {"kind": "range", "start": 0, "end": 2}}
  ],
  "projections": [
    {"id": "q", "source": {"stem": "block", "tensor": "{stem}.q.weight"},
     "shape": [128, 16], "kind": {"kind": "dense"}, "split": {"class": "column", "tile": 4},
     "epilogues": {"residual_add": false, "bias": false, "gelu": null, "swiglu": null, "output_quantization": null, "weighted_moe_combine": false},
     "numerical_contract": "a16.f32.v1", "use_profile": "served"},
    {"id": "k", "source": {"stem": "block", "tensor": "{stem}.k.weight"},
     "shape": [128, 16], "kind": {"kind": "dense"}, "split": {"class": "column", "tile": 2},
     "epilogues": {"residual_add": false, "bias": false, "gelu": null, "swiglu": null, "output_quantization": null, "weighted_moe_combine": false},
     "numerical_contract": "a16.f32.v1", "use_profile": "served"},
    {"id": "qk", "source": {"stem": "block", "tensor": "{stem}.qk.weight"},
     "shape": [256, 16], "kind": {"kind": "dense"}, "split": {"class": "column", "tile": 2},
     "epilogues": {"residual_add": false, "bias": false, "gelu": null, "swiglu": null, "output_quantization": null, "weighted_moe_combine": false},
     "numerical_contract": "a16.f32.v1", "use_profile": "served",
     "assembly": [
       {"projection": "q", "layer": "current", "operand": "weight", "rows": {"start": 0, "length": 128}, "coordinates": "global", "inherit_split": true},
       {"projection": "k", "layer": "current", "operand": "weight", "rows": {"start": 0, "length": 128}, "coordinates": "global", "inherit_split": true}
     ]},
    {"id": "chunked", "source": {"stem": "block", "tensor": "{stem}.chunked.weight"},
     "shape": [128, 16], "kind": {"kind": "dense"}, "split": {"class": "column", "tile": 4},
     "epilogues": {"residual_add": false, "bias": false, "gelu": null, "swiglu": null, "output_quantization": null, "weighted_moe_combine": false},
     "numerical_contract": "a16.f32.v1", "use_profile": "served",
     "parts": [
       {"id": "part-0", "rows": 64, "range": {"start": 0, "length": 64}, "scale_origin": [0, 0]},
       {"id": "part-1", "rows": 64, "range": {"start": 64, "length": 64}, "scale_origin": [0, 0]}
     ]},
    {"id": "context", "source": {"tensor": "context.weight"},
     "shape": [256, 16], "kind": {"kind": "dense"}, "split": {"class": "column", "tile": 2},
     "epilogues": {"residual_add": false, "bias": false, "gelu": null, "swiglu": null, "output_quantization": null, "weighted_moe_combine": false},
     "numerical_contract": "a16.f32.v1", "use_profile": "served",
     "assembly": [
       {"projection": "k", "layer": 0, "operand": null, "rows": {"start": 0, "length": 128}, "coordinates": "global", "inherit_split": true},
       {"projection": "k", "layer": 1, "operand": null, "rows": {"start": 0, "length": 128}, "coordinates": "global", "inherit_split": true}
     ]}
  ],
  "fusion_groups": [],
  "side_tensors": [
    {"id": "norm", "source": {"stem": "block", "tensor": "{stem}.norm.weight"}, "shape": [16],
     "kind": "norm", "consumer": {"kernel": "fixture.norm"}, "dtypes": ["Bf16"]},
    {"id": "sink", "source": {"stem": "block", "tensor": "{stem}.sink"}, "shape": [4],
     "kind": "sink", "dtypes": ["Bf16"], "consumer": "q",
     "placement": {"axis": 0, "split": {"class": "column", "tile": 1}}}
  ],
  "external_tensors": []
}"#;

fn operand(name: &str, shape: [u64; 2]) -> ImportedOperand {
    ImportedOperand {
        weight: WeightSpec {
            codes: TensorSpan {
                tensor: name.into(),
                dtype: DType::Bf16,
                shape: shape.to_vec(),
                offset: 0,
                byte_length: shape[0] * shape[1] * 2,
            },
            logical_shape: shape.to_vec(),
            format: WeightFormat::Bf16,
            encoding: CodeEncoding::Bf16LittleEndian,
            layout: WeightLayout::row_major(),
            local_scale: None,
            global_scale: None,
            transforms: vec![],
        },
        activation: Some(ActivationSpec::bf16()),
        source_tensors: vec![name.into()],
    }
}

fn info(shape: &[u64]) -> TensorInfo {
    TensorInfo {
        dtype: "BF16".into(),
        shape: shape.to_vec(),
        file: Some("weights.safetensors".into()),
    }
}

#[test]
fn assemblies_tile_their_rows_and_sides_copy_their_rank_ranges() {
    let g = Geometry::new();
    let registry = Registry::default();
    let declaration = parse(DECLARATION, &g, &registry)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let mut admitted = Admitted::new();
    let mut metadata = CheckpointMetadata::default();
    for layer in 0..2 {
        for name in ["q", "k"] {
            let rows = 128;
            let tensor = format!("layers.{layer}.{name}.weight");
            admitted.insert(tensor.clone(), operand(&tensor, [rows, 16]));
            metadata.tensors.insert(tensor, info(&[rows, 16]));
        }
        let chunked = format!("layers.{layer}.chunked.weight");
        let mut bank = operand(&chunked, [128, 16]);
        bank.weight.logical_shape = vec![2, 64, 16];
        bank.weight.codes.shape = vec![2, 64, 16];
        admitted.insert(chunked.clone(), bank);
        metadata.tensors.insert(chunked, info(&[128, 16]));
        metadata
            .tensors
            .insert(format!("layers.{layer}.norm.weight"), info(&[16]));
        metadata
            .tensors
            .insert(format!("layers.{layer}.sink"), info(&[4]));
    }
    let bound = declaration
        .bind(&g, &registry, &metadata, &AdmittedRules(&admitted))
        .unwrap();
    let imported = declared::imported(&declaration, &bound, &g, &admitted).unwrap();
    // Each layer of a stemmed assembly reads its own layer's first member.
    for layer in 0..2 {
        let key = projection_spec::derive::InstanceKey {
            projection: "qk".into(),
            layer: Some(layer),
        };
        assert_eq!(
            imported[&key].weights[0].codes.tensor,
            format!("layers.{layer}.q.weight")
        );
    }
    // The unstemmed assembly has one instance.
    assert!(
        imported
            .keys()
            .any(|k| k.projection == "context" && k.layer.is_none())
    );
    let topology = Topology { ranks: 2, rank: 1 };
    let graph = projection_spec::derive::derive(DerivationInput {
        declaration: &declaration,
        bound: &bound,
        geometry: &g,
        registry: &registry,
        topology,
        imported: &imported,
        variants: &NoVariants,
    })
    .unwrap();
    let bound_records = |projection: &str, layer: Option<u64>| {
        let operation = graph
            .operations
            .iter()
            .find(|o| o.instance.projection == projection && o.instance.layer == layer)
            .unwrap();
        records(
            &bind_record_policy(
                operation.clone(),
                RecordPolicy::Sections(RecordLayout::Rows),
            )
            .unwrap(),
        )
        .unwrap()
    };
    let copy = |rows: u64, range: std::ops::Range<u64>| WeightRecord {
        transform: WeightTransform::Copy,
        sources: vec![kernel_core::weight_transform::SourceTensor {
            dtype: "BF16".into(),
            shape: vec![rows, 16],
            byte_length: rows * 32,
        }],
        axis: Some(0),
        ranges: std::iter::once(range.clone()).collect(),
        stored_shape: vec![range.end - range.start, 16],
    };
    // Rank 1 holds q rows 64..128, then k rows 64..128, one record each.
    let qk = bound_records("qk", Some(1));
    assert_eq!(qk.len(), 2);
    assert_eq!(qk[0].tensors, ["layers.1.q.weight"]);
    assert_eq!((qk[0].view_offset, &qk[0].record), (0, &copy(128, 64..128)));
    assert_eq!(qk[1].tensors, ["layers.1.k.weight"]);
    assert_eq!(
        (qk[1].view_offset, &qk[1].record),
        (64 * 32, &copy(128, 64..128))
    );
    // The unstemmed assembly reads each layer's rank rows in member order.
    let context = bound_records("context", None);
    assert_eq!(
        context
            .iter()
            .map(|r| (r.tensors[0].as_str(), r.view_offset, r.record.clone()))
            .collect::<Vec<_>>(),
        [
            ("layers.0.k.weight", 0, copy(128, 64..128)),
            ("layers.1.k.weight", 64 * 32, copy(128, 64..128)),
        ]
    );
    // A rank's part frame gathers from the whole tensor at the part's rows.
    let chunked = bound_records("chunked", Some(0));
    assert_eq!(chunked.len(), 1);
    assert_eq!(chunked[0].record, copy(128, 64..128));
    // Norms copy whole; sinks copy the rank's heads.
    let sides: BTreeMap<_, _> = side_records(&declaration, &bound, &g, topology, &metadata)
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(sides.len(), 4);
    let norm = &sides["layers.0.norm.weight"];
    assert_eq!(norm.axis, Some(0));
    assert_eq!(norm.ranges[..], std::iter::once(0..16).collect::<Vec<_>>());
    assert_eq!(norm.stored_shape, [16]);
    let sink = &sides["layers.1.sink"];
    assert_eq!(sink.ranges[..], std::iter::once(2..4).collect::<Vec<_>>());
    assert_eq!(sink.stored_shape, [2]);
    assert_eq!(sink.output_bytes().unwrap(), 4);
}

/// Independent parts of a packed bank keep its stored code extent: an
/// MXFP4 bank of logical [2, 64, 64] stores [2, 64, 32] bytes of codes.
#[test]
fn admitted_packed_parts_keep_the_stored_k_axis() {
    use kernel_core::precision::{GroupAxis, NibbleOrder, ScaleConvention, ScaleScope, ScaleSpec};
    use projection_spec::bind::{BindingRequest, CompanionRules};
    let g = Geometry::new();
    let registry = Registry::default();
    let declaration = parse(DECLARATION, &g, &registry).unwrap();
    let mut projection = declaration
        .projections
        .iter()
        .find(|p| p.id == "chunked")
        .unwrap()
        .clone();
    projection.shape[1] = projection_spec::Dim::Literal(64);
    let mut bank = operand("chunked.weight", [128, 64]);
    bank.weight.logical_shape = vec![2, 64, 64];
    bank.weight.codes = TensorSpan {
        tensor: "chunked.weight".into(),
        dtype: DType::U8,
        shape: vec![2, 64, 32],
        offset: 0,
        byte_length: 4096,
    };
    bank.weight.format = WeightFormat::Mxfp4;
    bank.weight.encoding = CodeEncoding::E2M1(NibbleOrder::EvenLow);
    bank.weight.local_scale = Some(ScaleSpec {
        tensor: TensorSpan {
            tensor: "chunked.weight_scale".into(),
            dtype: DType::Ue8M0,
            shape: vec![2, 64, 2],
            offset: 0,
            byte_length: 256,
        },
        scope: ScaleScope::Block,
        convention: ScaleConvention::Multiply,
        groups: vec![
            GroupAxis {
                axis: 1,
                size: 1,
                origin: 0,
            },
            GroupAxis {
                axis: 2,
                size: 32,
                origin: 0,
            },
        ],
    });
    bank.weight.validate().unwrap();
    let admitted = Admitted::from([("chunked.weight".into(), bank)]);
    let request = BindingRequest {
        projection: &projection,
        tensor: "chunked.weight",
        logical_shape: &[128, 64],
        geometry: &g,
        layer: Some(0),
        operand: None,
        expert: None,
        shard: None,
        head: None,
    };
    let imported = AdmittedRules(&admitted)
        .import(&request, &CheckpointMetadata::default())
        .unwrap();
    for (index, part) in imported.parts.iter().enumerate() {
        assert_eq!(part.logical_shape, [64, 64]);
        assert_eq!(part.codes.shape, [64, 32]);
        assert_eq!(part.codes.offset, index as u64 * 2048);
    }
    assert_eq!(imported.weight.codes.shape, [128, 32]);
}

/// The bound fixture declaration with `sink` heads per layer and, with
/// `scalar`, an unstemmed F32 scalar side.
fn bound_sides(
    sink: u64,
    scalar: bool,
) -> (
    projection_spec::Declaration,
    projection_spec::bind::BoundDeclaration,
    CheckpointMetadata,
) {
    let g = Geometry::new();
    let registry = Registry::default();
    let mut text = DECLARATION.replace(
        r#""sink", "source": {"stem": "block", "tensor": "{stem}.sink"}, "shape": [4],"#,
        &format!(
            r#""sink", "source": {{"stem": "block", "tensor": "{{stem}}.sink"}}, "shape": [{sink}],"#
        ),
    );
    if scalar {
        text = text.replace(
            r#""side_tensors": ["#,
            r#""side_tensors": [
    {"id": "scalar", "source": {"tensor": "scalar"}, "shape": [], "kind": "norm",
     "consumer": {"kernel": "fixture.norm"}, "dtypes": ["F32"]},"#,
        );
    }
    let declaration = parse(&text, &g, &registry)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let mut admitted = Admitted::new();
    let mut metadata = CheckpointMetadata::default();
    for layer in 0..2 {
        for name in ["q", "k"] {
            let tensor = format!("layers.{layer}.{name}.weight");
            admitted.insert(tensor.clone(), operand(&tensor, [128, 16]));
            metadata.tensors.insert(tensor, info(&[128, 16]));
        }
        let chunked = format!("layers.{layer}.chunked.weight");
        let mut bank = operand(&chunked, [128, 16]);
        bank.weight.logical_shape = vec![2, 64, 16];
        bank.weight.codes.shape = vec![2, 64, 16];
        admitted.insert(chunked.clone(), bank);
        metadata.tensors.insert(chunked, info(&[128, 16]));
        metadata
            .tensors
            .insert(format!("layers.{layer}.norm.weight"), info(&[16]));
        metadata
            .tensors
            .insert(format!("layers.{layer}.sink"), info(&[sink]));
    }
    if scalar {
        metadata.tensors.insert(
            "scalar".into(),
            TensorInfo {
                dtype: "F32".into(),
                shape: vec![],
                file: Some("weights.safetensors".into()),
            },
        );
    }
    let bound = declaration
        .bind(&g, &registry, &metadata, &AdmittedRules(&admitted))
        .unwrap();
    (declaration, bound, metadata)
}

/// A replicated scalar side stores its one element.
#[test]
fn scalar_sides_store_one_element() {
    let (declaration, bound, metadata) = bound_sides(4, true);
    let topology = Topology { ranks: 2, rank: 1 };
    let sides: BTreeMap<_, _> =
        side_records(&declaration, &bound, &Geometry::new(), topology, &metadata)
            .unwrap()
            .into_iter()
            .collect();
    assert_eq!(sides["scalar"].output_bytes().unwrap(), 4);
    assert_eq!(sides["scalar"].stored_shape, [1]);
}

/// A rank past a side tensor's last element stores its zero padding.
#[test]
fn trailing_side_ranks_store_their_padding() {
    let g = Geometry::new();
    let (declaration, bound, metadata) = bound_sides(2, false);
    let topology = Topology { ranks: 4, rank: 3 };
    let selection = declaration
        .side_tensors
        .iter()
        .find(|s| s.id == "sink")
        .unwrap()
        .selection(&g, topology)
        .unwrap();
    assert_eq!(
        selection.repack(&[1, 2, 3, 4], DType::Bf16).unwrap(),
        [0, 0]
    );
    let sides: BTreeMap<_, _> = side_records(&declaration, &bound, &g, topology, &metadata)
        .unwrap()
        .into_iter()
        .collect();
    let sink = &sides["layers.1.sink"];
    assert_eq!(sink.output_bytes().unwrap(), 2);
    assert_eq!(sink.transform, WeightTransform::Inline(vec![0, 0]));
}
