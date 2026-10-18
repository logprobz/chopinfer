//! A grouped dense projection over a block-32 FP8 checkpoint tensor, from
//! the importer through derivation to its packed records: each rank's
//! groups are resident as MXFP8 rows, every row with its own block's scales,
//! including a block that straddles two groups.
use std::collections::BTreeMap;

use kernel_core::contract::{DType, WeightFormat};
use kernel_core::precision::TensorSpan;
use kernel_core::precision::execution::BlockScaledExecution;
use kernel_core::weight_transform::{PackedSection, WeightTransform};
use model_compiler_sdk::quantization::declared::{self, Admitted, AdmittedRules};
use model_compiler_sdk::quantization::format::RecordLayout;
use model_compiler_sdk::quantization::import::{ExportMetadata, OperandRole, import_operand};
use model_compiler_sdk::quantization::projection::{RecordPolicy, bind_record_policy, records};
use projection_spec::bind::{CheckpointMetadata, TensorInfo};
use projection_spec::derive::{DerivationInput, Topology, hooks::NoVariants};
use projection_spec::registry::Registry;
use projection_spec::{Geometry, parse};

/// Four groups of 48 rows: the 32-row block of rows 32..64 straddles the
/// first two groups, and the one of rows 128..160 the third and fourth.
const GROUPS: u64 = 4;
const ROWS: u64 = 48;
const K: u64 = 64;

const DECLARATION: &str = r#"{
  "schema": "chopinfer.projection.v1",
  "namespace": "fixture",
  "use_profiles": {"served": [{"phase": "Prefill", "capacity": 16, "graph": false},
                              {"phase": "Decode", "capacity": 1, "graph": true}]},
  "layer_stems": [],
  "projections": [
    {"id": "grouped", "source": {"tensor": "grouped.weight"},
     "shape": [48, 64], "kind": {"kind": "grouped_dense", "groups": 4},
     "split": {"class": "expert_owned", "tile": 1},
     "epilogues": {"residual_add": false, "bias": false, "gelu": null, "swiglu": null, "output_quantization": null, "weighted_moe_combine": false},
     "numerical_contract": "fixture.fold.v1", "format": "fp8_block32", "use_profile": "served"}
  ],
  "fusion_groups": [],
  "side_tensors": [],
  "external_tensors": []
}"#;

fn span(name: &str, dtype: DType, shape: &[u64]) -> TensorSpan {
    TensorSpan {
        tensor: name.into(),
        dtype,
        shape: shape.into(),
        offset: 0,
        byte_length: shape.iter().product(),
    }
}

#[test]
fn block32_groups_derive_and_record_as_mxfp8_rows() {
    let g = Geometry::new();
    let registry = Registry::default();
    let declaration = parse(DECLARATION, &g, &registry)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let mut metadata = ExportMetadata::default();
    metadata
        .add_document(
            "config",
            br#"{"quant_method":"fp8","activation_scheme":"dynamic",
                 "weight_block_size":[32,32],"scale_fmt":"ue8m0"}"#,
        )
        .unwrap();
    let stacked = [GROUPS * ROWS, K];
    let blocks = [stacked[0] / 32, K / 32];
    let tensors: BTreeMap<_, _> = [
        span("grouped.weight", DType::Fp8E4M3, &stacked),
        span("grouped.weight_scale", DType::Ue8M0, &blocks),
    ]
    .into_iter()
    .map(|t| (t.tensor.clone(), t))
    .collect();
    let role = OperandRole {
        operation: "grouped".into(),
        weight: "grouped.weight".into(),
        targets: vec!["grouped".into()],
        logical_shape: stacked.to_vec(),
        lookup: false,
        shared_scale_stem: None,
        declared: None,
    };
    let operand = import_operand(&metadata, &role, &tensors).unwrap();
    assert_eq!(operand.weight.format, WeightFormat::Fp8Block32x32);
    let admitted = Admitted::from([("grouped.weight".into(), operand)]);
    let checkpoint = CheckpointMetadata {
        tensors: [
            ("grouped.weight", "F8_E4M3", stacked.to_vec()),
            ("grouped.weight_scale", "U8", blocks.to_vec()),
        ]
        .into_iter()
        .map(|(name, dtype, shape)| {
            (
                name.into(),
                TensorInfo {
                    dtype: dtype.into(),
                    shape,
                    file: Some("weights.safetensors".into()),
                },
            )
        })
        .collect(),
        ..Default::default()
    };
    let bound = declaration
        .bind(&g, &registry, &checkpoint, &AdmittedRules(&admitted))
        .unwrap();
    let imported = declared::imported(&declaration, &bound, &g, &admitted).unwrap();

    let codes: Vec<u8> = (0..stacked[0] * K)
        .map(|i| ((i * 37 + 5) % 251) as u8 & 0x7e)
        .collect();
    let scales: Vec<u8> = (0..blocks[0] * blocks[1]).map(|i| 110 + i as u8).collect();
    for (ranks, local) in [(1_u32, 4_u64), (2, 2), (4, 1)] {
        let rank = ranks - 1;
        let derivation = projection_spec::derive::derive(DerivationInput {
            declaration: &declaration,
            bound: &bound,
            geometry: &g,
            registry: &registry,
            topology: Topology { ranks, rank },
            imported: &imported,
            variants: &NoVariants,
        })
        .unwrap();
        let operation = &derivation.operations[0];
        assert_eq!(operation.packing.format, WeightFormat::Mxfp8);
        assert_eq!(u64::from(operation.packing.experts), local);
        assert_eq!(
            operation.packing.layout.id,
            if local > 1 {
                "row-major-groups"
            } else {
                "row-major"
            }
        );
        let query = operation
            .fold_gemm_query(BlockScaledExecution::ExactDecodedBf16Mma)
            .unwrap();
        assert_eq!(
            (
                query.groups,
                query.n,
                query.k,
                query.activation_group_stride
            ),
            (local as u32, ROWS as u32, K as u32, K as u32)
        );
        assert_eq!(derivation.operation_specs().is_err(), local > 1);

        // Section copies would store the source grid; the packed operand
        // expands it.
        assert!(
            bind_record_policy(
                operation.clone(),
                RecordPolicy::Sections(RecordLayout::Rows)
            )
            .is_err()
        );
        let first = u64::from(rank) * local * ROWS;
        let expected_codes =
            codes[(first * K) as usize..((first + local * ROWS) * K) as usize].to_vec();
        let expected_scales: Vec<u8> = (first..first + local * ROWS)
            .flat_map(|row| {
                let block = (row / 32 * blocks[1]) as usize;
                scales[block..block + blocks[1] as usize].to_vec()
            })
            .collect();
        for policy in [RecordPolicy::Packed, RecordPolicy::PackedPieces] {
            let bound_operation = bind_record_policy(operation.clone(), policy).unwrap();
            let mut sections = BTreeMap::<usize, Vec<u8>>::new();
            for record in records(&bound_operation).unwrap() {
                let WeightTransform::PackedOperand(packed) = &record.record.transform else {
                    panic!("{policy:?} emits packed operands");
                };
                let sources: Vec<&[u8]> = record
                    .tensors
                    .iter()
                    .map(|name| match name.as_str() {
                        "grouped.weight" => codes.as_slice(),
                        "grouped.weight_scale" => scales.as_slice(),
                        other => panic!("unexpected source {other}"),
                    })
                    .collect();
                let bytes = packed.execute(&record.record, &sources).unwrap();
                let section = sections.entry(record.section.index()).or_default();
                assert_eq!(section.len() as u64, record.view_offset);
                section.extend(bytes);
            }
            assert_eq!(
                sections[&PackedSection::Codes.index()],
                expected_codes,
                "{policy:?}"
            );
            assert_eq!(
                sections[&PackedSection::LocalScales.index()],
                expected_scales,
                "{policy:?}"
            );
        }
    }
}
