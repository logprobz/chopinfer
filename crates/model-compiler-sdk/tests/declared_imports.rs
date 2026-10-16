//! The shared production importer over admitted operands, on a synthetic
//! declaration with fused operands, a per-expert bank and an unstemmed matrix.
use std::collections::BTreeMap;

use kernel_core::contract::{DType, WeightFormat};
use kernel_core::precision::{ActivationSpec, CodeEncoding, TensorSpan, WeightLayout, WeightSpec};
use model_compiler_sdk::quantization::declared::{self, Admitted, RankInputs};
use model_compiler_sdk::quantization::import::ImportedOperand;
use projection_spec::bind::{CheckpointMetadata, TensorInfo};
use projection_spec::derive::{InstanceKey, Topology, hooks::NoVariants};
use projection_spec::registry::Registry;
use projection_spec::{Geometry, parse};

const DECLARATION: &str = r#"{
  "schema": "chopinfer.projection.v1",
  "namespace": "fixture",
  "use_profiles": {"served": [
    {"phase": "Prefill", "capacity": 8, "graph": false},
    {"phase": "Decode", "capacity": 1, "graph": true}
  ]},
  "layer_stems": [
    {"id": "decoder", "template": "layers.{i}", "indices": {"kind": "range", "start": 0, "end": "layers"}}
  ],
  "projections": [
    {"id": "feed", "source": {"stem": "decoder", "tensor": "{stem}.mlp.{operand}.weight"},
     "shape": ["intermediate", "hidden"], "kind": {"kind": "dense"},
     "split": {"class": "column", "tile": 4},
     "epilogues": {"residual_add": false, "bias": false, "gelu": null, "swiglu": "ConcatenatedHalves", "output_quantization": null, "weighted_moe_combine": false},
     "numerical_contract": "a16.f32.v1", "operands": ["gate", "up"], "use_profile": "served"},
    {"id": "experts", "source": {"stem": "decoder", "tensor": "{stem}.mlp.experts.{expert}.down.weight"},
     "shape": ["hidden", "intermediate"], "kind": {"kind": "expert_bank", "experts": "experts", "top_k": 2},
     "split": {"class": "expert_owned", "tile": 4},
     "epilogues": {"residual_add": false, "bias": false, "gelu": null, "swiglu": null, "output_quantization": null, "weighted_moe_combine": false},
     "numerical_contract": "a16.f32.v1", "use_profile": "served"},
    {"id": "head", "source": {"tensor": "head.weight"},
     "shape": ["vocab", "hidden"], "kind": {"kind": "dense"}, "split": {"class": "replicate"},
     "epilogues": {"residual_add": false, "bias": false, "gelu": null, "swiglu": null, "output_quantization": null, "weighted_moe_combine": false},
     "numerical_contract": "a16.f32.v1", "use_profile": "served"}
  ],
  "fusion_groups": [],
  "side_tensors": [],
  "external_tensors": []
}"#;

/// The declaration with its bank's members named by `member`: identities
/// come from the declared source, not a checkpoint naming convention.
fn declaration(member: &str) -> String {
    DECLARATION.replace(
        "{stem}.mlp.experts.{expert}",
        &format!("{{stem}}.mlp.{member}"),
    )
}

/// Bank member names the tests run with, including an expert index that is
/// not a whole path segment.
const MEMBERS: [&str; 3] = [
    "experts.{expert}",
    "specialists.{expert}",
    "specialists_{expert}",
];

/// `member` naming `expert`.
fn member_name(member: &str, expert: u64) -> String {
    member.replace("{expert}", &expert.to_string())
}

fn geometry() -> Geometry {
    BTreeMap::from([
        ("hidden".into(), 16),
        ("intermediate".into(), 12),
        ("experts".into(), 8),
        ("vocab".into(), 32),
        ("layers".into(), 2),
    ])
}

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

fn sources(member: &str) -> (CheckpointMetadata, Admitted) {
    let mut admitted = BTreeMap::new();
    for layer in 0..2 {
        for part in ["gate", "up"] {
            let name = format!("layers.{layer}.mlp.{part}.weight");
            admitted.insert(name.clone(), operand(&name, [12, 16]));
        }
        for expert in 0..8 {
            let name = format!(
                "layers.{layer}.mlp.{}.down.weight",
                member_name(member, expert)
            );
            admitted.insert(name.clone(), operand(&name, [16, 12]));
        }
    }
    admitted.insert("head.weight".into(), operand("head.weight", [32, 16]));
    let metadata = CheckpointMetadata {
        tensors: admitted
            .iter()
            .map(|(name, o)| {
                (
                    name.clone(),
                    TensorInfo {
                        dtype: "BF16".into(),
                        shape: o.weight.logical_shape.clone(),
                        file: Some("weights.safetensors".into()),
                    },
                )
            })
            .collect(),
        ..Default::default()
    };
    (metadata, admitted)
}

#[test]
fn admitted_operands_derive_fused_bank_and_unstemmed_operations() {
    for member in MEMBERS {
        derive_fused_bank_and_unstemmed_operations(member);
    }
}

fn derive_fused_bank_and_unstemmed_operations(member: &str) {
    let g = geometry();
    let registry = Registry::default();
    let declaration = parse(&declaration(member), &g, &registry)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let (metadata, admitted) = sources(member);
    let bound = declaration
        .bind(
            &g,
            &registry,
            &metadata,
            &declared::AdmittedRules(&admitted),
        )
        .unwrap();
    let imported = declared::imported(&declaration, &bound, &g, &admitted).unwrap();
    assert_eq!(imported.len(), 5);
    let feed = &imported[&InstanceKey {
        projection: "feed".into(),
        layer: Some(1),
    }];
    assert_eq!(feed.operation_id, "{projection}.{i}.{output}");
    assert_eq!(feed.weights.len(), 2);
    assert_eq!(feed.weights[1].codes.tensor, "layers.1.mlp.up.weight");
    let experts = &imported[&InstanceKey {
        projection: "experts".into(),
        layer: Some(0),
    }];
    assert_eq!(experts.weights[0].logical_shape, [8, 16, 12]);
    assert_eq!(experts.matrices.len(), 8);
    assert_eq!(
        imported[&InstanceKey {
            projection: "head".into(),
            layer: None,
        }]
            .operation_id,
        "{projection}.{output}"
    );

    let graph = declared::derive(RankInputs {
        declaration: &declaration,
        geometry: &g,
        registry: &registry,
        variants: &NoVariants,
        metadata: &metadata,
        admitted: &admitted,
        topology: Topology { ranks: 2, rank: 1 },
    })
    .unwrap();
    let ids: std::collections::BTreeSet<_> = graph
        .operation_specs()
        .unwrap()
        .into_iter()
        .map(|o| o.id)
        .collect();
    assert_eq!(
        ids.into_iter().collect::<Vec<_>>(),
        [
            "experts.0.bf16",
            "experts.1.bf16",
            "feed.0.bf16",
            "feed.1.bf16",
            "head.bf16"
        ]
    );

    let mut missing = admitted.clone();
    missing.remove("layers.1.mlp.up.weight");
    assert!(
        declaration
            .bind(&g, &registry, &metadata, &declared::AdmittedRules(&missing))
            .unwrap_err()
            .0
            .contains("source was not admitted")
    );
}

#[test]
fn admitted_activation_recipes_replace_import_activations() {
    for member in MEMBERS {
        activation_recipes_replace_import_activations(member);
    }
}

fn activation_recipes_replace_import_activations(member: &str) {
    use kernel_core::contract::{ActivationFormat, ScaleMode};
    use kernel_core::precision::DecodeOrder;
    let g = geometry();
    let registry = Registry::default();
    let declaration = parse(&declaration(member), &g, &registry)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let (metadata, mut admitted) = sources(member);
    let graph = declared::derive(RankInputs {
        declaration: &declaration,
        geometry: &g,
        registry: &registry,
        variants: &NoVariants,
        metadata: &metadata,
        admitted: &admitted,
        topology: Topology { ranks: 2, rank: 1 },
    })
    .unwrap();
    let operations = graph.operation_specs().unwrap();
    let operation = |id: &str| operations.iter().find(|o| o.id == id).unwrap().clone();
    // BF16 sources keep the derived operation.
    let mut head = operation("head.bf16");
    declared::admitted_activations(&mut head, &admitted, &declaration, &g).unwrap();
    assert_eq!(head, operation("head.bf16"));
    // A bank reads its first expert's recipe and takes the native order.
    let fp8 = ActivationSpec {
        format: ActivationFormat::Fp8Group128(ScaleMode::ExactF32),
        ..ActivationSpec::bf16()
    };
    admitted
        .get_mut(&format!(
            "layers.1.mlp.{}.down.weight",
            member_name(member, 0)
        ))
        .unwrap()
        .activation = Some(fp8.clone());
    let mut experts = operation("experts.1.bf16");
    declared::admitted_activations(&mut experts, &admitted, &declaration, &g).unwrap();
    assert_eq!(experts.activations, [fp8]);
    assert_eq!(experts.arithmetic.weight_operand, DType::Fp8E4M3);
    assert_eq!(
        experts.arithmetic.decode_order,
        DecodeOrder::SuccessiveScalesAfterBlockMma
    );
    // Fused operands must agree, and every source must be admitted.
    admitted
        .get_mut("layers.0.mlp.up.weight")
        .unwrap()
        .activation = Some(ActivationSpec {
        format: ActivationFormat::Mxfp8,
        ..ActivationSpec::bf16()
    });
    let mut feed = operation("feed.0.bf16");
    assert!(declared::admitted_activations(&mut feed, &admitted, &declaration, &g).is_err());
    admitted.remove("head.weight");
    let mut head = operation("head.bf16");
    assert!(declared::admitted_activations(&mut head, &admitted, &declaration, &g).is_err());
}
