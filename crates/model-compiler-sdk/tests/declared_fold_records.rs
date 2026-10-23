//! Block-32 FP8 dense projections recorded with packed fold codes
//! (`RecordPolicy::Mxfp8FoldPieces`): each destination matrix's codes are
//! one `Mxfp8FoldPacked` gather, whose bytes equal
//! `Mxfp8FoldLayout::pack` of the row-major codes section the packed
//! operand records store, and its scales stay packed operand pieces of the
//! same bytes. Covered: a row slice, a K slice (with a final 64-K fold),
//! a grouped projection's groups and a fused pair of operands, at every
//! rank of one, two and four ranks.
use std::collections::BTreeMap;

use kernel_core::contract::DType;
use kernel_core::precision::TensorSpan;
use kernel_core::weight_transform::mxfp8_fold::{LAYOUT_ID, Mxfp8FoldLayout};
use kernel_core::weight_transform::{PackedSection, WeightRecord, WeightTransform};
use model_compiler_sdk::quantization::declared::{self, Admitted, AdmittedRules};
use model_compiler_sdk::quantization::format::RecordLayout;
use model_compiler_sdk::quantization::import::{ExportMetadata, OperandRole, import_operand};
use model_compiler_sdk::quantization::projection::{RecordPolicy, bind_record_policy, records};
use projection_spec::bind::{CheckpointMetadata, TensorInfo};
use projection_spec::derive::hooks::NoVariants;
use projection_spec::derive::{DerivationInput, DerivedOperation, Topology, ViewRole};
use projection_spec::registry::Registry;
use projection_spec::{Geometry, parse};

const EPILOGUES: &str = r#"{"residual_add": false, "bias": false, "gelu": null, "swiglu": null, "output_quantization": null, "weighted_moe_combine": false}"#;

/// A projection id, its source tensors (stacked shape) and its declaration.
type Fixture = (&'static str, Vec<(&'static str, [u64; 2])>, String);

fn projections() -> Vec<Fixture> {
    let declared = |id: &str, source: &str, rest: &str| {
        format!(
            r#"{{"id": "{id}", "source": {{"tensor": "{source}"}}, {rest},
                "epilogues": {EPILOGUES}, "numerical_contract": "fixture.fold.v1",
                "format": "fp8_block32", "use_profile": "served"}}"#
        )
    };
    vec![
        // Each rank keeps a row slice.
        (
            "rows",
            vec![("rows.weight", [64, 192])],
            declared(
                "rows",
                "rows.weight",
                r#""shape": [64, 192], "kind": {"kind": "dense"}, "split": {"class": "column", "tile": 8}"#,
            ),
        ),
        // Each rank keeps a K slice; at four ranks each 320-K slice ends in
        // a 64-K fold.
        (
            "columns",
            vec![("columns.weight", [32, 1280])],
            declared(
                "columns",
                "columns.weight",
                r#""shape": [32, 1280], "kind": {"kind": "dense"}, "split": {"class": "row", "tile": 64}"#,
            ),
        ),
        // Four groups of 48 rows, whole groups per rank.
        (
            "grouped",
            vec![("grouped.weight", [192, 64])],
            declared(
                "grouped",
                "grouped.weight",
                r#""shape": [48, 64], "kind": {"kind": "grouped_dense", "groups": 4}, "split": {"class": "expert_owned", "tile": 1}"#,
            ),
        ),
        // Two fused operands, each its own pieces.
        (
            "pair",
            vec![("pair.w1.weight", [64, 128]), ("pair.w3.weight", [64, 128])],
            declared(
                "pair",
                "pair.{operand}.weight",
                r#""shape": [64, 128], "operands": ["w1", "w3"], "kind": {"kind": "dense"}, "split": {"class": "column", "tile": 8}"#,
            ),
        ),
    ]
}

fn span(name: &str, dtype: DType, shape: &[u64]) -> TensorSpan {
    TensorSpan {
        tensor: name.into(),
        dtype,
        shape: shape.into(),
        offset: 0,
        byte_length: shape.iter().product(),
    }
}

/// Every record's bytes, placed in its section's view.
fn sections(
    operation: &DerivedOperation,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> BTreeMap<usize, Vec<u8>> {
    let mut sections = BTreeMap::<usize, Vec<u8>>::new();
    for record in records(operation).unwrap() {
        let sources: Vec<&[u8]> = record
            .tensors
            .iter()
            .map(|name| payloads[name].as_slice())
            .collect();
        let bytes = execute(&record.record, &sources);
        let section = sections.entry(record.section.index()).or_default();
        assert_eq!(section.len() as u64, record.view_offset);
        section.extend(bytes);
    }
    sections
}

fn execute(record: &WeightRecord, sources: &[&[u8]]) -> Vec<u8> {
    record
        .clone()
        .prepare(record.output_bytes().unwrap())
        .unwrap()
        .produce_whole(&mut |index, offset, out| {
            out.copy_from_slice(&sources[index][offset as usize..offset as usize + out.len()]);
            Ok(())
        })
        .unwrap()
}

#[test]
fn fold_pieces_store_the_packed_operand_codes_in_packed_folds() {
    let g = Geometry::new();
    let registry = Registry::default();
    let fixtures = projections();
    let declaration = format!(
        r#"{{"schema": "chopinfer.projection.v1", "namespace": "fixture",
            "use_profiles": {{"served": [{{"phase": "Prefill", "capacity": 16, "graph": false}},
                                         {{"phase": "Decode", "capacity": 1, "graph": true}}]}},
            "layer_stems": [], "projections": [{}],
            "fusion_groups": [], "side_tensors": [], "external_tensors": []}}"#,
        fixtures
            .iter()
            .map(|(_, _, declared)| declared.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    let declaration = parse(&declaration, &g, &registry)
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
    let mut tensors = BTreeMap::new();
    let mut checkpoint = CheckpointMetadata::default();
    let mut payloads = BTreeMap::new();
    let mut roles = Vec::new();
    for (index, (projection, sources, _)) in fixtures.iter().enumerate() {
        for (name, shape) in sources {
            let scale = format!("{name}_scale");
            let blocks = [shape[0] / 32, shape[1] / 32];
            for (tensor, dtype, header, shape) in [
                (name.to_string(), DType::Fp8E4M3, "F8_E4M3", shape.to_vec()),
                (scale.clone(), DType::Ue8M0, "U8", blocks.to_vec()),
            ] {
                tensors.insert(tensor.clone(), span(&tensor, dtype, &shape));
                checkpoint.tensors.insert(
                    tensor,
                    TensorInfo {
                        dtype: header.into(),
                        shape,
                        file: Some("weights.safetensors".into()),
                    },
                );
            }
            let seed = 7 * index as u64 + name.len() as u64;
            payloads.insert(
                name.to_string(),
                (0..shape[0] * shape[1])
                    .map(|i| ((i * 37 + 5 + seed) % 251) as u8 & 0x7e)
                    .collect::<Vec<u8>>(),
            );
            payloads.insert(
                scale,
                (0..blocks[0] * blocks[1])
                    .map(|i| 110 + ((i + seed) % 23) as u8)
                    .collect::<Vec<u8>>(),
            );
            roles.push(OperandRole {
                operation: projection.to_string(),
                weight: name.to_string(),
                targets: vec![projection.to_string()],
                logical_shape: shape.to_vec(),
                lookup: false,
                shared_scale_stem: None,
                declared: None,
            });
        }
    }
    let admitted: Admitted = roles
        .iter()
        .map(|role| {
            (
                role.weight.clone(),
                import_operand(&metadata, role, &tensors).unwrap(),
            )
        })
        .collect();
    let bound = declaration
        .bind(&g, &registry, &checkpoint, &AdmittedRules(&admitted))
        .unwrap();
    let imported = declared::imported(&declaration, &bound, &g, &admitted).unwrap();
    let mut covered = BTreeMap::<String, u32>::new();
    for ranks in [1_u32, 2, 4] {
        for rank in 0..ranks {
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
            let mut seen = std::collections::BTreeSet::new();
            for operation in derivation.selected_operations() {
                let projection = operation.instance.projection.as_str();
                let at = format!("{projection} tp{ranks} rank {rank}");
                if !seen.insert(projection) {
                    continue;
                }
                let operands = if projection == "pair" { 2 } else { 1 };
                // Today's records: one packed operand per section, or one
                // packed piece per operand.
                let reference = bind_record_policy(
                    operation.clone(),
                    if operands > 1 {
                        RecordPolicy::PackedPieces
                    } else {
                        RecordPolicy::Packed
                    },
                )
                .unwrap();
                let folded =
                    bind_record_policy(operation.clone(), RecordPolicy::Mxfp8FoldPieces).unwrap();
                // Fold codes come with packed operand scales only.
                assert!(
                    bind_record_policy(
                        operation.clone(),
                        RecordPolicy::Sections(RecordLayout::Mxfp8Fold)
                    )
                    .is_err(),
                    "{at}"
                );
                let plan = &operation.packing;
                let layout = Mxfp8FoldLayout {
                    groups: (plan.experts * plan.projections) as usize,
                    n: plan.rows as usize,
                    k: plan.k as usize,
                };
                let expected = sections(&reference, &payloads);
                let actual = sections(&folded, &payloads);
                let codes = PackedSection::Codes.index();
                let scales = PackedSection::LocalScales.index();
                assert_eq!(
                    actual[&codes],
                    layout.pack(&expected[&codes]).unwrap(),
                    "{at}"
                );
                assert_eq!(actual[&scales], expected[&scales], "{at}");
                assert_eq!(actual.len(), expected.len(), "{at}");
                for record in records(&folded).unwrap() {
                    let transform = &record.record.transform;
                    match record.section {
                        PackedSection::Codes => {
                            assert_eq!(*transform, WeightTransform::Mxfp8FoldPacked, "{at}");
                        }
                        _ => assert!(
                            matches!(transform, WeightTransform::PackedOperand(p)
                                if p.section == PackedSection::LocalScales),
                            "{at}"
                        ),
                    }
                }
                // The views keep their bytes; only the codes layout moves.
                for (before, after) in reference.views.iter().zip(&folded.views) {
                    assert_eq!((&before.id, before.bytes), (&after.id, after.bytes), "{at}");
                    let layout = if after.role == ViewRole::Weight(PackedSection::Codes) {
                        LAYOUT_ID.to_owned()
                    } else {
                        before.layout.clone()
                    };
                    assert_eq!(after.layout, layout, "{at}");
                }
                *covered.entry(projection.to_owned()).or_default() += 1;
            }
        }
    }
    assert_eq!(covered.len(), 4, "{covered:?}");
}
