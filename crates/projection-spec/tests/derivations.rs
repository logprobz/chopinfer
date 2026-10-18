mod common;

use common::*;
use kernel_core::contract::{DType, Target};
use kernel_core::precision::format::{A16Recipe, FormatId, registry as formats};
use kernel_core::precision::packed::MatrixView;
use kernel_core::precision::{
    ActivationSpec, ArithmeticSpec, MatrixGeometry, NibbleOrder, OperationKind, OperationSpec,
    Rounding,
};
use projection_spec::bind::{BoundDeclaration, CheckpointMetadata};
use projection_spec::derive::{
    self, DerivationInput, ImportedProjection, ImportedProjections, InstanceKey, RankDerivation,
    SourceMatrix, Topology, ViewRole, hooks::NoVariants,
};
use projection_spec::registry::Registry;
use projection_spec::{
    Declaration, Dim, Geometry, ProjectionKind, SliceAxis, SplitClass, TensorSource,
};

fn fixture(
    split: SplitClass,
    rows: u64,
    k: u64,
    kind: ProjectionKind,
) -> (Declaration, BoundDeclaration, Geometry, ImportedProjections) {
    let mut d = declaration();
    let mut p = d.projections[1].clone();
    p.id = "matrix".into();
    p.source = Some(TensorSource {
        stem: None,
        tensor: "matrix.weight".into(),
    });
    p.shape = [Dim::Literal(rows), Dim::Literal(k)];
    p.split = split;
    p.kind = kind;
    d.projections = vec![p];
    d.fusion_groups.clear();
    d.side_tensors.clear();
    d.layer_stems.clear();
    let g = geometry();
    let experts = match &d.projections[0].kind {
        ProjectionKind::ExpertBank { experts, .. } => experts.resolve(&g).unwrap(),
        _ => 1,
    };
    let groups = match &d.projections[0].kind {
        ProjectionKind::GroupedDense { groups } => Some(groups.resolve(&g).unwrap()),
        _ => None,
    };
    let shape = if let Some(groups) = groups {
        vec![groups * rows, k]
    } else if experts == 1 {
        vec![rows, k]
    } else {
        vec![experts, rows, k]
    };
    let metadata = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([("matrix.weight".into(), info(&shape, "BF16"))]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let bound = d
        .bind(&g, &Registry::default(), &metadata, &Bf16Rules)
        .unwrap();
    let weight = bound.operands[0].imported.weight.clone();
    let family = match d.projections[0].kind {
        ProjectionKind::Dense {} | ProjectionKind::GroupedDense { .. } => {
            kernel_core::precision::format::KernelFamily::Dense
        }
        ProjectionKind::ExpertBank { .. } => {
            kernel_core::precision::format::KernelFamily::GroupedExpert
        }
        ProjectionKind::LookupTable {} => kernel_core::precision::format::KernelFamily::Lookup,
    };
    let imported = ImportedProjection {
        operation_id: "matrix.{output}".into(),
        target: Target::Sm120f,
        family,
        format: FormatId::Bf16,
        geometry: derive::GeometryMode::RankLocal,
        weights: vec![weight.clone()],
        activations: if family == kernel_core::precision::format::KernelFamily::Lookup {
            vec![]
        } else {
            vec![ActivationSpec::bf16()]
        },
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Bf16,
            accumulator: DType::F32,
            output: DType::Bf16,
            decode_order: formats::by_id(FormatId::Bf16)
                .unwrap()
                .decode_order(
                    family,
                    kernel_core::contract::ActivationFormat::Bf16,
                    A16Recipe::ScaledWeight,
                )
                .unwrap(),
            rounding: Rounding::NearestEven,
            numerical_contract: d.projections[0].numerical_contract.clone(),
        },
        a16_recipe: A16Recipe::ScaledWeight,
        preparation: None,
        matrices: match groups {
            Some(groups) => {
                derive::grouped_dense_matrices(&bound.operands[0], groups, rows, k).unwrap()
            }
            None => (0..experts)
                .map(|e| SourceMatrix {
                    weight: weight.clone(),
                    expert: e as u32,
                    operand: 0,
                    source_expert: e,
                    target_rows: 0..rows,
                    source_rows: 0..rows,
                    target_k: 0..k,
                    source_k: 0..k,
                })
                .collect(),
        },
    };
    (
        d,
        bound,
        g,
        ImportedProjections::from([(
            InstanceKey {
                projection: "matrix".into(),
                layer: None,
            },
            imported,
        )]),
    )
}
fn run(
    d: &Declaration,
    b: &BoundDeclaration,
    g: &Geometry,
    i: &ImportedProjections,
    t: Topology,
) -> Result<RankDerivation, projection_spec::Error> {
    derive::derive(DerivationInput {
        declaration: d,
        bound: b,
        geometry: g,
        registry: &Registry::default(),
        topology: t,
        imported: i,
        variants: &NoVariants,
    })
}

#[test]
fn column_geometry_padding_specs_and_payloads_match_independent_rank_slices() {
    let (d, b, g, i) = fixture(
        SplitClass::Column {
            tile: Dim::Literal(4),
        },
        10,
        16,
        ProjectionKind::Dense {},
    );
    for rank in 0..2 {
        let r = run(&d, &b, &g, &i, Topology { ranks: 2, rank }).unwrap();
        assert_eq!(r.operations.len(), 4);
        let op = &r.operations[0];
        assert_eq!(op.logical_geometry, MatrixGeometry { rows: 5, k: 16 });
        assert_eq!(
            op.operation.geometry,
            Some(MatrixGeometry { rows: 64, k: 16 })
        );
        assert_eq!(
            op.slices[0].rows,
            std::iter::once(u64::from(rank) * 5..u64::from(rank) * 5 + 5).collect::<Vec<_>>()
        );
        let import = i.values().next().unwrap();
        let expected = OperationSpec {
            id: "matrix.bf16".into(),
            kind: OperationKind::Dense,
            target: Target::Sm120f,
            weights: import.weights.clone(),
            geometry: Some(MatrixGeometry { rows: 64, k: 16 }),
            activations: import.activations.clone(),
            arithmetic: import.arithmetic.clone(),
            epilogues: d.projections[0].epilogues,
            uses: op.operation.uses.clone(),
        };
        assert_eq!(op.operation, expected);
        assert_eq!(r.operation_specs().unwrap()[0].uses.len(), 4);
        let codes = (0u16..160).flat_map(u16::to_le_bytes).collect::<Vec<_>>();
        let view = MatrixView {
            tensor: "matrix.weight",
            format: kernel_core::contract::WeightFormat::Bf16,
            rows: 10,
            k: 16,
            row_origin: 0,
            k_origin: 0,
            nibble_order: NibbleOrder::EvenLow,
            codes: &codes,
            local: None,
            global: None,
        };
        let packed = derive::payload::slice(&op.slices[0], view).unwrap();
        let mut expected = codes[(rank as usize) * 160..(rank as usize + 1) * 160].to_vec();
        expected.resize(64 * 16 * 2, 0);
        assert_eq!(packed.codes, expected);
        assert!(
            op.views.iter().any(|v| v.role == ViewRole::Output
                && v.shape == vec![128, 64]
                && v.bytes == 16384)
        );
    }
}

#[test]
fn row_k_selection_expert_ownership_and_intermediate_sharding_are_distinct() {
    for (split, kind, logical, physical, experts) in [
        (
            SplitClass::Row {
                tile: Dim::Literal(4),
            },
            ProjectionKind::Dense {},
            MatrixGeometry { rows: 8, k: 5 },
            MatrixGeometry { rows: 64, k: 16 },
            1,
        ),
        (
            SplitClass::ExpertOwned {
                tile: Dim::Literal(1),
            },
            ProjectionKind::ExpertBank {
                experts: Dim::Literal(4),
                top_k: Dim::Literal(1),
            },
            MatrixGeometry { rows: 8, k: 10 },
            MatrixGeometry { rows: 8, k: 16 },
            2,
        ),
        (
            SplitClass::ExpertSharded {
                axis: SliceAxis::Rows,
                tile: Dim::Literal(4),
            },
            ProjectionKind::ExpertBank {
                experts: Dim::Literal(4),
                top_k: Dim::Literal(1),
            },
            MatrixGeometry { rows: 4, k: 10 },
            MatrixGeometry { rows: 4, k: 16 },
            4,
        ),
    ] {
        let (d, b, g, i) = fixture(split, 8, 10, kind);
        let r = run(&d, &b, &g, &i, Topology { ranks: 2, rank: 1 }).unwrap();
        assert_eq!(r.operations[0].logical_geometry, logical);
        assert_eq!(r.operations[0].operation.geometry, Some(physical));
        assert_eq!(r.operations[0].packing.experts, experts);
        assert_eq!(r.operations[0].slices.len(), experts as usize);
        if matches!(d.projections[0].split, SplitClass::ExpertOwned { .. }) {
            assert_eq!(r.operations[0].slices[0].source.source_expert, 2);
            assert_eq!(r.operations[0].slices[0].expert, 0);
        }
    }
}

/// One output row of `op` over the stacked `[groups * rows, k]` source `w`
/// for one row `x` of its input view, as the derivation lays them out:
/// local group `g` reads input columns `g * K..` and writes output columns
/// `g * rows..`, K and rows being the packed extents of one group.
fn execute(op: &derive::DerivedOperation, w: &[f64], k: u64, x: &[f64]) -> Vec<f64> {
    let groups = u64::from(op.dense_groups().unwrap_or(1));
    let (n, width) = (u64::from(op.packing.rows), u64::from(op.packing.k));
    assert_eq!(x.len() as u64, groups * width);
    let mut y = vec![0.0; (groups * n) as usize];
    for slice in &op.slices {
        let group = u64::from(slice.expert);
        let sources = slice.rows.iter().flat_map(|r| r.clone());
        let targets = slice.target_rows.iter().flat_map(|r| r.clone());
        for (source, target) in sources.zip(targets) {
            let mut sum = 0.0;
            for column in slice.k.clone() {
                let input = group * width + slice.target_k.start + column - slice.k.start;
                sum += w[(source * k + column) as usize] * x[input as usize];
            }
            y[(group * n + target) as usize] += sum;
        }
    }
    y
}

#[test]
fn grouped_dense_ranks_own_whole_groups_and_map_onto_the_fold_contract() {
    let (d, b, g, i) = fixture(
        SplitClass::ExpertOwned {
            tile: Dim::Literal(1),
        },
        4,
        32,
        ProjectionKind::GroupedDense {
            groups: Dim::Literal(4),
        },
    );
    for (ranks, groups) in [(1, Some(4)), (2, Some(2)), (4, None)] {
        for rank in 0..ranks {
            let r = run(&d, &b, &g, &i, Topology { ranks, rank }).unwrap();
            let op = &r.operations[0];
            let local = groups.unwrap_or(1);
            assert_eq!(op.dense_groups(), groups);
            assert_eq!(op.packing.experts, local);
            assert_eq!(op.logical_geometry, MatrixGeometry { rows: 4, k: 32 });
            assert_eq!(op.slices.len(), local as usize);
            for (index, slice) in op.slices.iter().enumerate() {
                let group = rank * local + index as u32;
                assert_eq!(slice.expert, index as u32);
                assert_eq!(slice.source.expert, group);
                let first = u64::from(group) * 4;
                assert_eq!(slice.rows, vec![first..first + 4]);
                assert_eq!(slice.target_rows, vec![0..4]);
            }
            let (n, width) = (u64::from(op.packing.rows), u64::from(op.packing.k));
            for (role, shape) in [
                (ViewRole::Input, vec![128, u64::from(local) * width]),
                (ViewRole::Output, vec![128, u64::from(local) * n]),
            ] {
                assert!(
                    op.views.iter().any(|v| v.role == role && v.shape == shape),
                    "{role:?} {shape:?}"
                );
            }
            // BF16 fragments are not the fold GEMM's MXFP8 rows.
            let refusal = op
                .fold_gemm_query(
                    kernel_core::precision::execution::BlockScaledExecution::ExactDecodedBf16Mma,
                )
                .unwrap_err()
                .0;
            assert!(refusal.contains("MXFP8 rows"), "{refusal}");
            // Several groups have no public dense operation kind.
            let selection =
                derive::rank_selection(&d.projections[0], &g, Topology { ranks, rank }, 0).unwrap();
            assert_eq!(
                d.projections[0].operation_kind(&g, &selection).is_err(),
                groups.is_some()
            );
            // One group is the dense matrix it is; more never lower to a
            // dense kernel.
            let specs = r.operation_specs();
            if groups.is_some() {
                let error = specs.unwrap_err().0;
                assert!(error.contains("grouped fold GEMM"), "{error}");
            } else {
                assert_eq!(specs.unwrap()[0].kind, OperationKind::Dense);
            }
        }
    }
    // Groups split over ranks whole, never across a rank boundary.
    assert!(run(&d, &b, &g, &i, Topology { ranks: 8, rank: 0 }).is_err());
    let mut dense_split = d.clone();
    dense_split.projections[0].split = SplitClass::Column {
        tile: Dim::Literal(4),
    };
    assert!(dense_split.validate(&g, &Registry::default()).is_err());
}

#[test]
fn distinct_inputs_per_group_need_the_grouped_form() {
    let (groups, rows, k) = (4u64, 4u64, 32u64);
    let w: Vec<f64> = (0..groups * rows * k)
        .map(|i| (i % 7) as f64 - 3.0)
        .collect();
    // One rank of two holds groups 2 and 3, and its input row holds their
    // two distinct K-wide slices side by side.
    let x: Vec<f64> = (0..2 * k).map(|i| ((i * 5) % 11) as f64 - 5.0).collect();
    let mut reference = vec![];
    for local in 0..2 {
        for row in 0..rows {
            let source = (2 + local) * rows + row;
            reference.push(
                (0..k)
                    .map(|c| w[(source * k + c) as usize] * x[(local * k + c) as usize])
                    .sum::<f64>(),
            );
        }
    }
    let topology = Topology { ranks: 2, rank: 1 };
    let (d, b, g, i) = fixture(
        SplitClass::ExpertOwned {
            tile: Dim::Literal(1),
        },
        rows,
        k,
        ProjectionKind::GroupedDense {
            groups: Dim::Literal(groups),
        },
    );
    let grouped = run(&d, &b, &g, &i, topology).unwrap();
    let op = &grouped.operations[0];
    let n = u64::from(op.packing.rows);
    let y = execute(op, &w, k, &x);
    let live: Vec<f64> = (0..2)
        .flat_map(|local| y[(local * n) as usize..(local * n + rows) as usize].to_vec())
        .collect();
    assert_eq!(live, reference);

    // The same stack as one dense matrix: the rank's rows, but every row
    // reads the first K columns, so the second group sees the first's input.
    let (d, b, g, i) = fixture(
        SplitClass::Column {
            tile: Dim::Literal(rows),
        },
        groups * rows,
        k,
        ProjectionKind::Dense {},
    );
    let dense = run(&d, &b, &g, &i, topology).unwrap();
    let op = &dense.operations[0];
    let y = execute(op, &w, k, &x[..op.packing.k as usize]);
    assert_eq!(y[..(rows as usize)], reference[..(rows as usize)]);
    assert_ne!(
        y[(rows as usize)..(2 * rows as usize)],
        reference[(rows as usize)..]
    );
}

#[test]
fn head_grouped_selection_duplicates_small_head_counts_by_rank() {
    let (d, b, g, i) = fixture(
        SplitClass::HeadGrouped {
            heads: Dim::Literal(2),
            head_dim: Dim::Literal(4),
            tile: Dim::Literal(4),
        },
        8,
        16,
        ProjectionKind::Dense {},
    );
    for rank in 0..4 {
        let r = run(&d, &b, &g, &i, Topology { ranks: 4, rank }).unwrap();
        assert_eq!(r.operations[0].logical_geometry.rows, 4);
        assert_eq!(
            r.operations[0].slices[0].rows[0],
            u64::from(rank / 2) * 4..u64::from(rank / 2) * 4 + 4
        );
    }
}

#[test]
fn imported_overlaps_holes_and_unbound_sources_are_refused() {
    let (d, b, g, i) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    let mut bad = i.clone();
    let import = bad.values_mut().next().unwrap();
    let mut half = import.matrices[0].clone();
    half.target_rows = 0..4;
    half.source_rows = 0..4;
    import.matrices = vec![half.clone(), half];
    assert!(
        run(&d, &b, &g, &bad, Topology { ranks: 1, rank: 0 })
            .unwrap_err()
            .0
            .contains("overlap")
    );
    bad.values_mut().next().unwrap().matrices.pop();
    assert!(
        run(&d, &b, &g, &bad, Topology { ranks: 1, rank: 0 })
            .unwrap_err()
            .0
            .contains("do not cover")
    );
    bad.values_mut().next().unwrap().matrices[0]
        .weight
        .codes
        .tensor = "unbound".into();
    assert!(
        run(&d, &b, &g, &bad, Topology { ranks: 1, rank: 0 })
            .unwrap_err()
            .0
            .contains("authenticated binding")
    );
}

#[test]
fn empty_and_mismatched_payload_views_return_errors() {
    let (d, b, g, i) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    let r = run(&d, &b, &g, &i, Topology { ranks: 1, rank: 0 }).unwrap();
    let mut selection = r.operations[0].slices[0].clone();
    let codes = vec![0; 256];
    let view = MatrixView {
        tensor: "matrix.weight",
        format: kernel_core::contract::WeightFormat::Bf16,
        rows: 8,
        k: 16,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes: &codes,
        local: None,
        global: None,
    };
    selection.rows.clear();
    assert!(
        derive::payload::slice(&selection, view)
            .unwrap_err()
            .0
            .contains("nonempty")
    );
    selection = r.operations[0].slices[0].clone();
    assert!(
        derive::payload::slice(
            &selection,
            MatrixView {
                tensor: "another.weight",
                ..view
            }
        )
        .unwrap_err()
        .0
        .contains("source matrix")
    );
}

#[test]
fn side_views_bind_their_declared_role_even_for_another_consumer() {
    use projection_spec::contracts::{ConsumerView, RowRange, SourceMember, ViewConversion};
    let (mut d, _, g, mut i) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    let mut reused = d.projections[0].clone();
    reused.id = "other".into();
    reused.assembly = vec![SourceMember {
        coordinates: Default::default(),
        inherit_split: false,
        projection: "matrix".into(),
        layer: None,
        operand: None,
        rows: RowRange {
            start: Dim::Literal(0),
            length: Dim::Literal(8),
        },
    }];
    d.projections.push(reused);
    d.side_tensors.push(projection_spec::SideTensor {
        placement: Default::default(),
        id: "side".into(),
        source: TensorSource {
            stem: None,
            tensor: "side.weight".into(),
        },
        shape: vec![Dim::Literal(16)],
        kind: projection_spec::SideKind::Norm,
        dtypes: vec![DType::Bf16],
        consumer: "matrix".into(),
        views: vec![ConsumerView {
            consumer: "other".into(),
            dtype: DType::F32,
            conversion: ViewConversion::WidenBf16F32,
        }],
        norm: None,
    });
    let metadata = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([
            ("matrix.weight".into(), info(&[8, 16], "BF16")),
            ("side.weight".into(), info(&[16], "BF16")),
        ]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let b = d
        .bind(&g, &Registry::default(), &metadata, &Bf16Rules)
        .unwrap();
    let mut reused = i.values().next().unwrap().clone();
    reused.operation_id = "other.{output}".into();
    i.insert(
        InstanceKey {
            projection: "other".into(),
            layer: None,
        },
        reused,
    );
    let result = run(&d, &b, &g, &i, Topology { ranks: 1, rank: 0 }).unwrap();
    assert_eq!(result.side_views.len(), 1);
    assert_eq!(result.side_views[0].dtype, DType::F32);
    assert_eq!(result.side_views[0].bytes, 64);
    d.side_tensors[0].placement = projection_spec::side::SidePlacement {
        axis: 0,
        split: SplitClass::Column {
            tile: Dim::Literal(4),
        },
    };
    let result = run(&d, &b, &g, &i, Topology { ranks: 2, rank: 1 }).unwrap();
    assert_eq!(result.side_views[0].shape, vec![8]);
    assert_eq!(result.side_views[0].dtype, DType::F32);
    assert_eq!(result.side_views[0].bytes, 32);
}

#[test]
fn requested_buckets_preserve_compiled_capacity_multiplier_and_output_contract() {
    use projection_spec::contracts::UseContract;
    let (mut d, b, g, i) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    d.projections[0].uses.truncate(1);
    d.projections[0].uses[0].contract = Some(UseContract {
        compiled_capacity: Dim::Literal(64),
        row_multiplier: Dim::Literal(2),
        chunk_rows: Some(Dim::Literal(64)),
        per_item_verify: None,
        aggregate_verify: None,
        output_dtype: DType::F32,
        numerical_contract: "fixture.f32.v1".into(),
        activation: Some(kernel_core::contract::ActivationFormat::Bf16),
        exact_rows: false,
        support: vec![],
        implementation: None,
    });
    let r = run(&d, &b, &g, &i, Topology { ranks: 1, rank: 0 }).unwrap();
    let op = &r.operations[0];
    assert_eq!(op.requested.capacity, 128);
    assert_eq!(op.compiled_capacity, 64);
    assert_eq!(op.chunk_rows, Some(64));
    assert_eq!(op.row_multiplier, 2);
    assert_eq!(op.operation.id, "matrix.f32");
    assert_eq!(op.operation.arithmetic.numerical_contract, "fixture.f32.v1");
    let output = op
        .views
        .iter()
        .find(|v| v.role == ViewRole::Output)
        .unwrap();
    assert_eq!(output.dtype, DType::F32);
    assert_eq!(output.shape, vec![256, 64]);
    assert_eq!(output.bytes, 65536);
}

#[test]
fn execution_aliases_keep_source_names_and_change_operation_layer_identity() {
    use projection_spec::contracts::{ExecutionLayers, LayerAlias};
    let (mut d, _, g, mut i) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    d.layer_stems.push(projection_spec::LayerStem {
        id: "alias".into(),
        template: "layers.{i}".into(),
        indices: projection_spec::LayerIndices::List {
            indices: vec![Dim::Key("layers".into())],
        },
        execution: Some(ExecutionLayers {
            indices: vec![Dim::Literal(3)],
            aliases: vec![LayerAlias {
                source_stem: None,
                execution_index: Dim::Literal(3),
                source_index: Dim::Key("layers".into()),
            }],
        }),
        support: vec![],
    });
    d.projections[0].source = Some(TensorSource {
        stem: Some("alias".into()),
        tensor: "{stem}.matrix.weight".into(),
    });
    let metadata = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([(
            "layers.2.matrix.weight".into(),
            info(&[8, 16], "BF16"),
        )]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let b = d
        .bind(&g, &Registry::default(), &metadata, &Bf16Rules)
        .unwrap();
    let mut imported = i.pop_first().unwrap().1;
    imported.operation_id = "layers.{i}.matrix.{output}".into();
    imported.weights[0] = b.operands[0].imported.weight.clone();
    imported.matrices[0].weight = imported.weights[0].clone();
    i.insert(
        InstanceKey {
            projection: "matrix".into(),
            layer: Some(2),
        },
        imported,
    );
    let r = run(&d, &b, &g, &i, Topology { ranks: 1, rank: 0 }).unwrap();
    assert_eq!(r.operations[0].instance.layer, Some(3));
    assert_eq!(r.operations[0].source_layer, Some(2));
    assert_eq!(r.operations[0].operation.id, "layers.3.matrix.bf16");
    assert_eq!(
        r.operations[0].slices[0].source.weight.codes.tensor,
        "layers.2.matrix.weight"
    );
}

#[test]
fn explicit_row_family_keeps_row_major_geometry_and_fusion_replacements_select_uses() {
    let (d, b, g, mut i) = fixture(SplitClass::Replicate {}, 8, 10, ProjectionKind::Dense {});
    i.values_mut().next().unwrap().family = kernel_core::precision::format::KernelFamily::Row;
    let mut rank = run(&d, &b, &g, &i, Topology { ranks: 1, rank: 0 }).unwrap();
    assert_eq!(
        rank.operations[0].operation.geometry,
        Some(MatrixGeometry { rows: 8, k: 10 })
    );
    assert_eq!(
        rank.operations[0].packing.layout,
        kernel_core::precision::WeightLayout::row_major()
    );
    let mut replacement = rank.operations[0].clone();
    replacement.operation.id = "fused".into();
    rank.fusions.push(projection_spec::derive::DerivedFusion {
        id: "group".into(),
        layer: None,
        requested: replacement.requested,
        members: vec![replacement.instance.clone()],
        fused: true,
        operations: vec![replacement],
        views: vec![],
        stages: vec![],
    });
    let specs = rank.operation_specs().unwrap();
    assert_eq!(specs.len(), 2);
    assert_eq!(
        specs.iter().find(|s| s.id == "fused").unwrap().uses.len(),
        1
    );
    assert_eq!(
        specs
            .iter()
            .find(|s| s.id == "matrix.bf16")
            .unwrap()
            .uses
            .len(),
        3
    );
}

fn variant_parameters(
    v: &projection_spec::Variant,
    _: &Geometry,
) -> Result<serde_json::Value, projection_spec::Error> {
    Ok(v.parameters.clone())
}
static VARIANTS: projection_spec::registry::Provider = projection_spec::registry::Provider {
    namespace: "fixture",
    registrations: &[projection_spec::registry::Registration {
        id: "fixture/implementation.v1",
        numerical_contract: "fixture.implementation.v1",
        target: projection_spec::registry::Target::Projection,
        decoder: variant_parameters,
    }],
};
struct Implementations(derive::DerivedOperation);
impl derive::hooks::VariantDeriver for Implementations {
    fn implementation(
        &self,
        v: &projection_spec::Variant,
        ctx: &derive::ProjectionContext<'_>,
        _: &ImportedProjection,
    ) -> Result<derive::DerivedOperation, projection_spec::Error> {
        let mut operation = self.0.clone();
        operation.operation.id = format!(
            "implementation.{}",
            v.parameters["schedule"].as_str().unwrap()
        );
        operation.operation.arithmetic.numerical_contract = v.numerical_contract.clone();
        operation.requested = kernel_core::precision::OperationUse {
            phase: ctx.usage.phase,
            capacity: ctx.usage.capacity.resolve(ctx.geometry)? as u32,
            graph: ctx.usage.graph,
        };
        operation.operation.uses = vec![operation.requested];
        let c = ctx.usage.contract.as_ref();
        operation.compiled_capacity =
            c.map(|c| c.compiled_capacity.resolve(ctx.geometry))
                .transpose()?
                .unwrap_or(operation.requested.capacity.into()) as u32;
        operation.row_multiplier = c
            .map(|c| c.row_multiplier.resolve(ctx.geometry))
            .transpose()?
            .unwrap_or(1);
        operation.chunk_rows = c
            .and_then(|c| c.chunk_rows.as_ref())
            .map(|d| d.resolve(ctx.geometry))
            .transpose()?;
        operation.per_item_verify = c
            .and_then(|c| c.per_item_verify.as_ref())
            .map(|d| d.resolve(ctx.geometry))
            .transpose()?;
        operation.aggregate_verify = c
            .and_then(|c| c.aggregate_verify.as_ref())
            .map(|d| d.resolve(ctx.geometry))
            .transpose()?;
        operation.exact_rows = c.is_some_and(|c| c.exact_rows);

        Ok(operation)
    }
}

#[test]
fn registered_implementations_use_the_override_and_missing_behavior_refuses() {
    use projection_spec::contracts::{Composition, Port, UseContract};
    let (mut d, b, g, i) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    let baseline = run(&d, &b, &g, &i, Topology { ranks: 1, rank: 0 })
        .unwrap()
        .operations[0]
        .clone();
    let port = Port {
        interpretation: None,
        id: "rows".into(),
        binding: "rows".into(),
        dtype: DType::Bf16,
        shape: vec![Dim::Literal(8), Dim::Literal(16)],
        state: false,
        format: None,
    };
    let default = projection_spec::Variant {
        id: "fixture/implementation.v1".into(),
        parameters: serde_json::json!({"schedule":"default"}),
        numerical_contract: "fixture.implementation.v1".into(),
        inputs: vec![port.clone()],
        outputs: vec![port],
    };
    d.projections[0].composition = Some(Composition {
        implementations: Default::default(),
        numerical_contracts: Default::default(),
        implementation: Some(default.clone()),
        ..Default::default()
    });
    let mut override_ = default;
    override_.parameters = serde_json::json!({"schedule":"decode"});
    d.projections[0].uses[1].contract = Some(UseContract {
        compiled_capacity: Dim::Literal(1),
        row_multiplier: Dim::Literal(1),
        chunk_rows: None,
        per_item_verify: None,
        aggregate_verify: None,
        output_dtype: DType::Bf16,
        numerical_contract: "fixture.implementation.v1".into(),
        activation: None,
        exact_rows: true,
        support: vec![],
        implementation: Some(override_),
    });
    let mut registry = Registry::default();
    registry.register(&VARIANTS).unwrap();
    let input = |variants| DerivationInput {
        declaration: &d,
        bound: &b,
        geometry: &g,
        registry: &registry,
        topology: Topology { ranks: 1, rank: 0 },
        imported: &i,
        variants,
    };
    assert!(
        derive::derive(input(&NoVariants))
            .unwrap_err()
            .0
            .contains("no implementation behavior")
    );
    let result = derive::derive(input(&Implementations(baseline))).unwrap();
    assert_eq!(result.operations[0].operation.id, "implementation.default");
    assert_eq!(result.operations[1].operation.id, "implementation.decode");
}

#[test]
fn implementation_hooks_preserve_declared_topology_refusals() {
    use projection_spec::contracts::{Composition, Port};
    let (mut d, b, g, i) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    let baseline = run(&d, &b, &g, &i, Topology { ranks: 1, rank: 0 })
        .unwrap()
        .operations[0]
        .clone();
    let port = Port {
        interpretation: None,
        id: "rows".into(),
        binding: "rows".into(),
        dtype: DType::Bf16,
        shape: vec![Dim::Literal(8), Dim::Literal(16)],
        state: false,
        format: None,
    };
    d.projections[0].composition = Some(Composition {
        implementations: Default::default(),
        numerical_contracts: Default::default(),
        implementation: Some(projection_spec::Variant {
            id: "fixture/implementation.v1".into(),
            parameters: serde_json::json!({"schedule":"default"}),
            numerical_contract: "fixture.implementation.v1".into(),
            inputs: vec![port.clone()],
            outputs: vec![port],
        }),
        ..Default::default()
    });
    let mut text = serde_json::to_value(&d).unwrap();
    text["projections"][0]["uses"][0]["contract"] = serde_json::json!({
        "compiled_capacity":128,"row_multiplier":1,"per_item_verify":null,"aggregate_verify":null,
        "output_dtype":"Bf16","numerical_contract":"fixture.implementation.v1","activation":null,"exact_rows":false,
        "support":[{"ranks":[2],"formats":[],"phases":[],"enabled":false}]
    });
    let mut registry = Registry::default();
    registry.register(&VARIANTS).unwrap();
    let d = projection_spec::parse(&text.to_string(), &g, &registry).unwrap();
    let hooks = Implementations(baseline);
    let error = derive::derive(DerivationInput {
        declaration: &d,
        bound: &b,
        geometry: &g,
        registry: &registry,
        topology: Topology { ranks: 2, rank: 0 },
        imported: &i,
        variants: &hooks,
    })
    .unwrap_err();
    assert!(error.0.contains("not supported"));
}

#[test]
fn source_shape_geometry_keeps_none_in_specs_and_derives_rank_geometry_separately() {
    let (d, b, g, mut i) = fixture(
        SplitClass::VocabSharded {
            tile: Dim::Literal(4),
        },
        8,
        16,
        ProjectionKind::LookupTable {},
    );
    i.values_mut().next().unwrap().geometry = derive::GeometryMode::SourceShape;
    let rank = run(&d, &b, &g, &i, Topology { ranks: 2, rank: 1 }).unwrap();
    assert_eq!(rank.operations[0].operation.geometry, None);
    assert_eq!(
        rank.operations[0].operation.matrix_geometry().unwrap(),
        MatrixGeometry { rows: 8, k: 16 }
    );
    assert_eq!(
        rank.operations[0].logical_geometry,
        MatrixGeometry { rows: 4, k: 16 }
    );
    assert_eq!(rank.operations[0].packing.rows, 4);
    assert_eq!(rank.operations[0].slices[0].rows[0], 4..8);
}

#[test]
fn lookup_output_views_use_embedding_width() {
    let (d, b, g, i) = fixture(
        SplitClass::Replicate {},
        8,
        16,
        ProjectionKind::LookupTable {},
    );
    let rank = run(&d, &b, &g, &i, Topology { ranks: 1, rank: 0 }).unwrap();
    let output = rank.operations[0]
        .views
        .iter()
        .find(|v| v.role == ViewRole::Output)
        .unwrap();
    assert_eq!(output.shape, vec![128, 16]);
}

#[test]
fn segmented_head_ownership_preserves_rank_gather_order_and_exact_bytes() {
    use projection_spec::contracts::RowRange;
    use projection_spec::{HeadGatherOrder, HeadSegment};
    let range = |start, length| RowRange {
        start: Dim::Literal(start),
        length: Dim::Literal(length),
    };
    let split = SplitClass::SegmentedHeads {
        segments: vec![
            HeadSegment {
                id: "q".into(),
                heads: Dim::Literal(2),
                head_dim: Dim::Literal(2),
                ranges: vec![range(0, 2), range(4, 2)],
            },
            HeadSegment {
                id: "v".into(),
                heads: Dim::Literal(4),
                head_dim: Dim::Literal(1),
                ranges: vec![range(2, 2), range(6, 2)],
            },
        ],
        order: HeadGatherOrder::Segments(vec!["v".into(), "q".into()]),
        tile: Dim::Literal(2),
    };
    let (d, b, g, i) = fixture(split, 8, 16, ProjectionKind::Dense {});
    let rank = run(&d, &b, &g, &i, Topology { ranks: 2, rank: 1 }).unwrap();
    let selection = &rank.operations[0].slices[0];
    assert_eq!(selection.rows, vec![6..8, 4..6]);
    let codes = (0u16..128).flat_map(u16::to_le_bytes).collect::<Vec<_>>();
    let view = MatrixView {
        tensor: "matrix.weight",
        format: kernel_core::contract::WeightFormat::Bf16,
        rows: 8,
        k: 16,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes: &codes,
        local: None,
        global: None,
    };
    let packed = derive::payload::slice(selection, view).unwrap();
    let mut expected = codes[6 * 32..8 * 32].to_vec();
    expected.extend_from_slice(&codes[4 * 32..6 * 32]);
    expected.resize(64 * 16 * 2, 0);
    assert_eq!(packed.codes, expected);
}

#[test]
fn format_numerical_mapping_controls_the_derived_operation() {
    let (mut d, b, g, i) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    let recipe = projection_spec::contracts::NumericalRecipe {
        numerical_contract: "alternate-bf16-v1".into(),
        a16_recipe: A16Recipe::ScaledWeight,
    };
    d.projections[0].composition = Some(projection_spec::contracts::Composition {
        implementations: Default::default(),
        numerical_contracts: std::collections::BTreeMap::from([(FormatId::Bf16, recipe.clone())]),
        ..Default::default()
    });
    let result = run(&d, &b, &g, &i, Topology { ranks: 1, rank: 0 }).unwrap();
    assert!(
        result
            .operations
            .iter()
            .all(|op| op.operation.arithmetic.numerical_contract == "alternate-bf16-v1")
    );
}

#[test]
fn activation_segments_materialize_rank_k_columns_with_exact_bits_and_zero_padding() {
    use projection_spec::inputs::{ActivationInput, ActivationSegment, ActivationSelection};
    let (mut d, b, g, i) = fixture(
        SplitClass::Row {
            tile: Dim::Literal(8),
        },
        8,
        8,
        ProjectionKind::Dense {},
    );
    d.projections[0].inputs = vec![ActivationInput {
        operand: "weight".into(),
        binding: "hidden-rank".into(),
        segments: vec![ActivationSegment {
            origin: None,
            length: None,
            source: "hidden".into(),
            selection: ActivationSelection::RankK {
                projection: "matrix".into(),
                operand: "weight".into(),
            },
        }],
    }];
    let r = run(&d, &b, &g, &i, Topology { ranks: 2, rank: 1 }).unwrap();
    let op = &r.operations[0];
    let input = &op.inputs[0];
    assert_eq!(input.segments[0].columns, 4..8);
    assert_eq!(input.segments[0].target_columns, 0..4);
    assert_eq!(input.view.shape, vec![128, 16]);
    let bytes = (0..1024u16).flat_map(u16::to_le_bytes).collect::<Vec<_>>();
    let buffers = std::collections::BTreeMap::from([(
        "hidden".into(),
        derive::payload::ActivationBuffer {
            dtype: DType::Bf16,
            rows: 128,
            columns: 8,
            bytes: &bytes,
        },
    )]);
    let packed = derive::payload::input(input, &buffers).unwrap();
    let expected = (0..128)
        .flat_map(|row| {
            bytes[(row * 8 + 4) * 2..(row * 8 + 8) * 2]
                .iter()
                .copied()
                .chain([0; 24])
        })
        .collect::<Vec<_>>();
    assert_eq!(packed, expected);
    d.projections[0].inputs[0].segments[0].selection = ActivationSelection::Rows {
        rows: projection_spec::contracts::RowRange {
            start: Dim::Literal(0),
            length: Dim::Literal(3),
        },
    };
    assert!(
        run(&d, &b, &g, &i, Topology { ranks: 2, rank: 1 })
            .unwrap_err()
            .0
            .contains("weight K ownership")
    );
}

/// Two local groups of a grouped dense input read their own K columns of
/// the whole grouped activation, each into its packed K-wide slot.
#[test]
fn grouped_activation_bindings_cover_every_local_group() {
    use projection_spec::inputs::{ActivationInput, ActivationSegment, ActivationSelection};
    let (mut d, b, g, i) = fixture(
        SplitClass::ExpertOwned {
            tile: Dim::Literal(1),
        },
        4,
        8,
        ProjectionKind::GroupedDense {
            groups: Dim::Literal(4),
        },
    );
    d.projections[0].inputs = vec![ActivationInput {
        operand: "weight".into(),
        binding: "grouped".into(),
        segments: vec![ActivationSegment {
            origin: None,
            length: None,
            source: "heads".into(),
            selection: ActivationSelection::RankK {
                projection: "matrix".into(),
                operand: "weight".into(),
            },
        }],
    }];
    let r = run(&d, &b, &g, &i, Topology { ranks: 2, rank: 1 }).unwrap();
    let op = &r.operations[0];
    // BF16 fragments pad K 8 to 16: each group's slot ends in zeros.
    assert_eq!(op.packing.k, 16);
    let input = &op.inputs[0];
    assert_eq!(input.view.shape, vec![128, 32]);
    assert_eq!(
        input
            .segments
            .iter()
            .map(|s| (s.columns.clone(), s.target_columns.clone()))
            .collect::<Vec<_>>(),
        vec![(16..24, 0..8), (24..32, 16..24)]
    );
    let bytes = (0..128 * 32u16)
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let buffers = std::collections::BTreeMap::from([(
        "heads".into(),
        derive::payload::ActivationBuffer {
            dtype: DType::Bf16,
            rows: 128,
            columns: 32,
            bytes: &bytes,
        },
    )]);
    let packed = derive::payload::input(input, &buffers).unwrap();
    let expected = (0..128)
        .flat_map(|row| {
            let group =
                |g: usize| bytes[(row * 32 + g * 8) * 2..(row * 32 + g * 8 + 8) * 2].to_vec();
            [group(2), vec![0; 16], group(3), vec![0; 16]].concat()
        })
        .collect::<Vec<_>>();
    assert_eq!(packed, expected);
    // A binding that covers only the first group's columns leaves the
    // second uncovered.
    d.projections[0].inputs[0].segments[0].selection = ActivationSelection::Rows {
        rows: projection_spec::contracts::RowRange {
            start: Dim::Literal(16),
            length: Dim::Literal(8),
        },
    };
    assert!(
        run(&d, &b, &g, &i, Topology { ranks: 2, rank: 1 })
            .unwrap_err()
            .0
            .contains("cover live K")
    );
}

#[test]
fn inherited_assemblies_select_source_ownership_once_and_translate_rank_local_rows() {
    use projection_spec::contracts::{RowCoordinates, RowRange, SourceMember};
    let (mut d, _, g, mut i) = fixture(
        SplitClass::Column {
            tile: Dim::Literal(4),
        },
        8,
        16,
        ProjectionKind::Dense {},
    );
    let mut p = d.projections[0].clone();
    p.id = "assembly".into();
    p.source = None;
    p.split = SplitClass::Replicate {};
    p.assembly = vec![SourceMember {
        coordinates: RowCoordinates::Global,
        inherit_split: true,
        projection: "matrix".into(),
        layer: None,
        operand: Some("weight".into()),
        rows: RowRange {
            start: Dim::Literal(0),
            length: Dim::Literal(8),
        },
    }];
    d.projections.push(p);
    let m = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([(
            "matrix.weight".into(),
            info(&[8, 16], "BF16"),
        )]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let b = d.bind(&g, &Registry::default(), &m, &Bf16Rules).unwrap();
    let mut prototype = i.values().next().unwrap().clone();
    prototype.operation_id = "assembly.{output}".into();
    prototype.matrices.clear();
    i.insert(
        InstanceKey {
            projection: "assembly".into(),
            layer: None,
        },
        prototype,
    );
    let r = run(&d, &b, &g, &i, Topology { ranks: 2, rank: 1 }).unwrap();
    let op = r
        .operations
        .iter()
        .find(|o| o.instance.projection == "assembly")
        .unwrap();
    assert_eq!(op.logical_geometry, MatrixGeometry { rows: 4, k: 16 });
    assert_eq!(op.slices[0].rows, vec![4..8]);
    assert_eq!(op.slices[0].source.target_rows, 0..4);
    d.projections[1].shape[0] = Dim::Literal(2);
    d.projections[1].assembly[0].coordinates = RowCoordinates::RankLocal;
    d.projections[1].assembly[0].rows = RowRange {
        start: Dim::Literal(1),
        length: Dim::Literal(2),
    };
    let b = d.bind(&g, &Registry::default(), &m, &Bf16Rules).unwrap();
    let r = run(&d, &b, &g, &i, Topology { ranks: 2, rank: 1 }).unwrap();
    let op = r
        .operations
        .iter()
        .find(|o| o.instance.projection == "assembly")
        .unwrap();
    assert_eq!(op.slices[0].rows, vec![5..7]);
    assert_eq!(op.logical_geometry.rows, 2);
}

fn cross_stem_fixture() -> (Declaration, BoundDeclaration, Geometry, ImportedProjections) {
    use projection_spec::contracts::{ExecutionLayers, LayerAlias};
    let (mut d, _, g, mut imports) =
        fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    d.layer_stems = vec![
        projection_spec::LayerStem {
            id: "main".into(),
            template: "main.{i}".into(),
            indices: projection_spec::LayerIndices::List {
                indices: vec![Dim::Literal(0)],
            },
            execution: None,
            support: vec![],
        },
        projection_spec::LayerStem {
            id: "aux".into(),
            template: "aux.{i}".into(),
            indices: projection_spec::LayerIndices::List {
                indices: vec![Dim::Literal(1)],
            },
            execution: Some(ExecutionLayers {
                indices: vec![Dim::Literal(9)],
                aliases: vec![LayerAlias {
                    source_stem: Some("main".into()),
                    execution_index: Dim::Literal(9),
                    source_index: Dim::Literal(0),
                }],
            }),
            support: vec![],
        },
    ];
    d.projections[0].source = Some(TensorSource {
        stem: Some("main".into()),
        tensor: "{stem}.matrix.weight".into(),
    });
    let mut alias = d.projections[0].clone();
    alias.id = "alias".into();
    alias.source.as_mut().unwrap().stem = Some("aux".into());
    d.projections.push(alias);
    let m = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([
            ("main.0.matrix.weight".into(), info(&[8, 16], "BF16")),
            ("aux.1.matrix.weight".into(), info(&[8, 16], "BF16")),
        ]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let b = d.bind(&g, &Registry::default(), &m, &Bf16Rules).unwrap();
    let prototype = imports.pop_first().unwrap().1;
    for (projection, layer, tensor) in [
        ("matrix", 0, "main.0.matrix.weight"),
        ("alias", 1, "aux.1.matrix.weight"),
    ] {
        let mut i = prototype.clone();
        i.operation_id = "{stem}.{projection}.{output}".into();
        i.weights[0] = b
            .operands
            .iter()
            .find(|o| o.imported.weight.codes.tensor == tensor)
            .unwrap()
            .imported
            .weight
            .clone();
        i.matrices[0].weight = i.weights[0].clone();
        imports.insert(
            InstanceKey {
                projection: projection.into(),
                layer: Some(layer),
            },
            i,
        );
    }
    (d, b, g, imports)
}

#[test]
fn cross_stem_execution_uses_declared_source_names_and_keeps_execution_identity() {
    let (d, b, g, imports) = cross_stem_fixture();
    let r = run(&d, &b, &g, &imports, Topology { ranks: 1, rank: 0 }).unwrap();
    let alias = r
        .operations
        .iter()
        .find(|o| o.instance.projection == "alias")
        .unwrap();
    assert_eq!(alias.instance.layer, Some(9));
    assert_eq!(alias.source_layer, Some(0));
    assert_eq!(alias.operation.id, "aux.9.alias.bf16");
    assert_eq!(
        alias.slices[0].source.weight.codes.tensor,
        "main.0.matrix.weight"
    );
}

/// One dense projection over `layers` stem instances, one binding each.
fn layered_fixture(layers: u64) -> (Declaration, BoundDeclaration, Geometry, ImportedProjections) {
    let (mut d, _, g, mut imports) =
        fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    d.layer_stems = vec![projection_spec::LayerStem {
        id: "main".into(),
        template: "main.{i}".into(),
        indices: projection_spec::LayerIndices::Range {
            start: Dim::Literal(0),
            end: Dim::Literal(layers),
        },
        execution: None,
        support: vec![],
    }];
    d.projections[0].source = Some(TensorSource {
        stem: Some("main".into()),
        tensor: "{stem}.matrix.weight".into(),
    });
    let m = CheckpointMetadata {
        tensors: (0..layers)
            .map(|i| (format!("main.{i}.matrix.weight"), info(&[8, 16], "BF16")))
            .collect(),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let b = d.bind(&g, &Registry::default(), &m, &Bf16Rules).unwrap();
    let prototype = imports.pop_first().unwrap().1;
    for operand in &b.operands {
        let mut i = prototype.clone();
        i.operation_id = "{stem}.{projection}.{output}".into();
        i.weights[0] = operand.imported.weight.clone();
        i.matrices[0].weight = i.weights[0].clone();
        imports.insert(
            InstanceKey {
                projection: "matrix".into(),
                layer: operand.layer,
            },
            i,
        );
    }
    (d, b, g, imports)
}

/// Seconds per derived operation, the fastest of three runs.
fn seconds_per_operation(layers: u64) -> f64 {
    let (d, b, g, imports) = layered_fixture(layers);
    let uses = d.projections[0].uses.len() as u64;
    (0..3)
        .map(|_| {
            let start = std::time::Instant::now();
            let r = run(&d, &b, &g, &imports, Topology { ranks: 1, rank: 0 }).unwrap();
            assert_eq!(r.operations.len() as u64, layers * uses);
            start.elapsed().as_secs_f64() / r.operations.len() as f64
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn derivation_cost_per_operation_does_not_grow_with_graph_size() {
    // Scanning every binding for each operation makes the large graph's
    // per-operation cost several times the small one's.
    let small = seconds_per_operation(128);
    let large = seconds_per_operation(2048);
    assert!(
        large < 2.0 * small,
        "per-operation derivation grew from {small:.2e} s to {large:.2e} s"
    );
}

fn paired_fixture(
    rows: u64,
    k: u64,
) -> (Declaration, BoundDeclaration, Geometry, ImportedProjections) {
    use projection_spec::contracts::{RowRange, SourceMember};
    let (mut d, _, g, mut imports) =
        fixture(SplitClass::Replicate {}, rows, k, ProjectionKind::Dense {});
    let mut p = d.projections[0].clone();
    p.id = "other".into();
    p.source = None;
    p.assembly = vec![SourceMember {
        coordinates: Default::default(),
        inherit_split: false,
        projection: "matrix".into(),
        layer: None,
        operand: None,
        rows: RowRange {
            start: Dim::Literal(0),
            length: Dim::Literal(rows),
        },
    }];
    d.projections.push(p);
    d.fusion_groups = vec![
        serde_json::from_value(
            serde_json::json!({"id":"pair","input":"hidden","members":["matrix","other"]}),
        )
        .unwrap(),
    ];
    let m = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([(
            "matrix.weight".into(),
            info(&[rows, k], "BF16"),
        )]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let b = d.bind(&g, &Registry::default(), &m, &Bf16Rules).unwrap();
    let mut imported = imports.values().next().unwrap().clone();
    imported.operation_id = "other.{output}".into();
    imports.insert(
        InstanceKey {
            projection: "other".into(),
            layer: None,
        },
        imported,
    );
    (d, b, g, imports)
}

#[test]
fn fusion_maps_apply_the_selected_block_a16_recipe_to_replacement_specs() {
    use kernel_core::contract::{ScaleMode, WeightFormat};
    use kernel_core::precision::{
        CodeEncoding, DecodeOrder, GroupAxis, ScaleConvention, ScaleScope, ScaleSpec, TensorSpan,
    };
    let (mut d, mut b, g, mut imports) = paired_fixture(128, 128);
    let recipe = projection_spec::contracts::NumericalRecipe {
        numerical_contract: "fixture.fusion.v1".into(),
        a16_recipe: A16Recipe::ScaleAfterBlockMma,
    };
    d.fusion_groups[0].composition = Some(projection_spec::contracts::Composition {
        implementations: Default::default(),
        numerical_contracts: std::collections::BTreeMap::from([(
            FormatId::Fp8Block128,
            recipe.clone(),
        )]),
        ..Default::default()
    });
    let mut weight = b.operands[0].imported.weight.clone();
    weight.format = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);
    weight.encoding = CodeEncoding::E4M3;
    weight.codes.dtype = DType::Fp8E4M3;
    weight.codes.byte_length /= 2;
    weight.local_scale = Some(ScaleSpec {
        tensor: TensorSpan {
            tensor: "matrix.scale".into(),
            dtype: DType::F32,
            shape: vec![1, 1],
            offset: 0,
            byte_length: 4,
        },
        scope: ScaleScope::Block,
        convention: ScaleConvention::Multiply,
        groups: vec![
            GroupAxis {
                axis: 0,
                size: 128,
                origin: 0,
            },
            GroupAxis {
                axis: 1,
                size: 128,
                origin: 0,
            },
        ],
    });
    b.operands[0].imported.weight = weight.clone();
    b.operands[0].format = FormatId::Fp8Block128;

    for i in imports.values_mut() {
        i.format = FormatId::Fp8Block128;
        i.weights = vec![weight.clone()];
        i.matrices[0].weight = weight.clone();
    }
    let rank = run(&d, &b, &g, &imports, Topology { ranks: 1, rank: 0 }).unwrap();
    assert!(
        rank.selected_operations()
            .all(|op| op.a16_recipe == A16Recipe::ScaleAfterBlockMma
                && op.operation.arithmetic.decode_order == DecodeOrder::BlockScaleAfterMma
                && op.operation.arithmetic.numerical_contract == "fixture.fusion.v1")
    );
}

static HOOKS: projection_spec::registry::Provider = projection_spec::registry::Provider {
    namespace: "fixture",
    registrations: &[
        projection_spec::registry::Registration {
            id: "fixture/implementation.v1",
            numerical_contract: "fixture.implementation.v1",
            target: projection_spec::registry::Target::Projection,
            decoder: variant_parameters,
        },
        projection_spec::registry::Registration {
            id: "fixture/source.v1",
            numerical_contract: "fixture.source.v1",
            target: projection_spec::registry::Target::SourceTransform,
            decoder: variant_parameters,
        },
        projection_spec::registry::Registration {
            id: "fixture/stage.v1",
            numerical_contract: "fixture.stage.v1",
            target: projection_spec::registry::Target::Stage,
            decoder: variant_parameters,
        },
        projection_spec::registry::Registration {
            id: "fixture/fusion.v1",
            numerical_contract: "fixture.fusion.v1",
            target: projection_spec::registry::Target::FusionGroup,
            decoder: variant_parameters,
        },
    ],
};
fn hook_stage(id: &str) -> derive::DerivedStage {
    derive::DerivedStage {
        id: id.into(),
        numerical_contract: "fixture.stage.v1".into(),
        inputs: vec![],
        outputs: vec![],
        mutations: vec![],
        adapter: None,
    }
}
struct HookProbe(std::cell::RefCell<Vec<String>>);
impl derive::hooks::VariantDeriver for HookProbe {
    fn implementation(
        &self,
        variant: &projection_spec::Variant,
        ctx: &derive::ProjectionContext<'_>,
        imported: &ImportedProjection,
    ) -> Result<derive::DerivedOperation, projection_spec::Error> {
        let mut operation = ctx.ordinary(imported)?;
        operation.operation.arithmetic.numerical_contract = variant.numerical_contract.clone();
        operation.stages.push(hook_stage("implementation-stage"));
        Ok(operation)
    }
    fn transform(
        &self,
        v: &projection_spec::Variant,
        part: Option<&projection_spec::Part>,
        ctx: &derive::ProjectionContext<'_>,
        imported: ImportedProjection,
    ) -> Result<derive::SourceTransformResult, projection_spec::Error> {
        assert!(!ctx.bound.operands.is_empty());
        if let Some(part) = part {
            let start = part
                .range
                .as_ref()
                .unwrap()
                .start
                .resolve_offset(ctx.geometry)?;
            let end = start + part.rows.resolve(ctx.geometry)?;
            assert!(
                imported
                    .matrices
                    .iter()
                    .all(|m| m.target_rows.start >= start && m.target_rows.end <= end)
            );
        }
        self.0.borrow_mut().push(v.id.clone());
        Ok(derive::SourceTransformResult::Applied(imported))
    }
    fn stage(
        &self,
        stage: &projection_spec::contracts::Stage,
        ctx: &derive::ProjectionContext<'_>,
    ) -> Result<derive::DerivedStage, projection_spec::Error> {
        self.0.borrow_mut().push(stage.id.clone());
        let ports = |ports: &[projection_spec::contracts::Port]| {
            ports
                .iter()
                .map(|p| {
                    let shape = p
                        .shape
                        .iter()
                        .map(|d| d.resolve(ctx.geometry))
                        .collect::<Result<Vec<_>, _>>()?;
                    let bytes = shape.iter().product::<u64>()
                        * kernel_core::precision::dtype_bits(p.dtype)
                        / 8;
                    Ok(derive::RequiredView {
                        id: p.binding.clone(),
                        role: ViewRole::State,
                        dtype: p.dtype,
                        interpretation: p.interpretation,
                        shape,
                        bytes,
                        layout: "row-major".into(),
                    })
                })
                .collect::<Result<Vec<_>, projection_spec::Error>>()
        };
        Ok(derive::DerivedStage {
            id: stage.id.clone(),
            numerical_contract: stage.contract.numerical_contract.clone(),
            inputs: ports(&stage.contract.inputs)?,
            outputs: ports(&stage.contract.outputs)?,
            mutations: stage.mutates.clone(),
            adapter: None,
        })
    }
    fn fusion(
        &self,
        v: &projection_spec::Variant,
        ctx: &derive::FusionContext<'_>,
        members: &[derive::DerivedOperation],
    ) -> Result<derive::DerivedFusion, projection_spec::Error> {
        self.0.borrow_mut().push(v.id.clone());
        assert_eq!(
            ctx.numerical.unwrap().a16_recipe,
            A16Recipe::ScaleAfterBlockMma
        );
        Ok(derive::DerivedFusion {
            id: ctx.group.id.clone(),
            layer: ctx.projection.layer,
            requested: members[0].requested,
            members: members.iter().map(|m| m.instance.clone()).collect(),
            fused: true,
            operations: members.to_vec(),
            views: vec![],
            stages: vec![hook_stage("fusion-stage")],
        })
    }
}

#[test]
fn registered_hooks_receive_bound_context_and_preserve_ordered_state_and_fusion_recipes() {
    use projection_spec::contracts::{Composition, Interpretation, NumericalRecipe, Port, Stage};
    let (mut d, _, g, imports) = paired_fixture(8, 16);
    let weight_port = Port {
        id: "weight".into(),
        binding: "weight".into(),
        dtype: DType::Bf16,
        interpretation: None,
        shape: vec![Dim::Literal(8), Dim::Literal(16)],
        state: false,
        format: Some(FormatId::Bf16),
    };
    d.projections[0].source_transforms = vec![projection_spec::Variant {
        id: "fixture/source.v1".into(),
        numerical_contract: "fixture.source.v1".into(),
        parameters: serde_json::json!({}),
        inputs: vec![weight_port.clone()],
        outputs: vec![weight_port.clone()],
    }];
    let state = Port {
        id: "history".into(),
        binding: "history".into(),
        dtype: DType::I64,
        interpretation: Some(Interpretation::Unsigned),
        shape: vec![Dim::Literal(2)],
        state: true,
        format: None,
    };
    let stage = Stage {
        id: "before".into(),
        contract: projection_spec::Variant {
            id: "fixture/stage.v1".into(),
            numerical_contract: "fixture.stage.v1".into(),
            parameters: serde_json::json!({}),
            inputs: vec![state.clone()],
            outputs: vec![state],
        },
        phases: vec![kernel_core::precision::ExecutionPhase::Decode],
        mutates: vec!["history".into()],
    };
    d.projections[0].composition = Some(Composition {
        implementations: Default::default(),
        pre: vec![stage.clone()],
        implementation: Some(projection_spec::Variant {
            id: "fixture/implementation.v1".into(),
            numerical_contract: "fixture.implementation.v1".into(),
            parameters: serde_json::json!({}),
            inputs: vec![weight_port.clone()],
            outputs: vec![weight_port.clone()],
        }),
        ..Default::default()
    });
    let recipe = NumericalRecipe {
        numerical_contract: "fixture.fusion.v1".into(),
        a16_recipe: A16Recipe::ScaleAfterBlockMma,
    };
    d.fusion_groups[0].composition = Some(Composition {
        implementations: Default::default(),
        post: vec![Stage {
            id: "after".into(),
            ..stage
        }],
        numerical_contracts: std::collections::BTreeMap::from([(FormatId::Bf16, recipe)]),
        implementation: Some(projection_spec::Variant {
            id: "fixture/fusion.v1".into(),
            numerical_contract: "fixture.fusion.v1".into(),
            parameters: serde_json::json!({}),
            inputs: vec![weight_port.clone()],
            outputs: vec![weight_port],
        }),
        ..Default::default()
    });
    let mut registry = Registry::default();
    registry.register(&HOOKS).unwrap();
    let m = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([(
            "matrix.weight".into(),
            info(&[8, 16], "BF16"),
        )]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let bound = d.bind(&g, &registry, &m, &Bf16Rules).unwrap();
    let probe = HookProbe(Default::default());
    let rank = derive::derive(DerivationInput {
        declaration: &d,
        bound: &bound,
        geometry: &g,
        registry: &registry,
        topology: Topology { ranks: 1, rank: 0 },
        imported: &imports,
        variants: &probe,
    })
    .unwrap();
    assert!(probe.0.borrow().iter().any(|s| s == "fixture/source.v1"));
    let decode = rank
        .fusions
        .iter()
        .find(|f| f.requested.phase == kernel_core::precision::ExecutionPhase::Decode)
        .unwrap();
    assert_eq!(
        decode
            .stages
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>(),
        ["fusion-stage", "after"]
    );
    assert_eq!(
        decode.stages[1].inputs[0].interpretation,
        Some(Interpretation::Unsigned)
    );
    assert_eq!(
        rank.operations
            .iter()
            .find(|o| o.instance.projection == "matrix"
                && o.requested.phase == kernel_core::precision::ExecutionPhase::Decode)
            .unwrap()
            .stages
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>(),
        ["implementation-stage", "before"]
    );
    assert!(
        rank.operations
            .iter()
            .find(|o| o.instance.projection == "matrix"
                && o.requested.phase == kernel_core::precision::ExecutionPhase::Decode)
            .unwrap()
            .stages
            .iter()
            .any(|s| s.id == "before")
    );
}

#[test]
fn per_part_provider_hooks_receive_only_their_declared_target_window() {
    let (mut d, _, g, imports) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    let port = projection_spec::contracts::Port {
        id: "weight".into(),
        binding: "weight".into(),
        dtype: DType::Bf16,
        interpretation: None,
        shape: vec![Dim::Literal(8), Dim::Literal(16)],
        state: false,
        format: Some(FormatId::Bf16),
    };
    let transform = projection_spec::Variant {
        id: "fixture/source.v1".into(),
        numerical_contract: "fixture.source.v1".into(),
        parameters: serde_json::json!({}),
        inputs: vec![port.clone()],
        outputs: vec![port],
    };
    d.projections[0].parts = (0..2)
        .map(|i| projection_spec::Part {
            id: format!("p{i}"),
            rows: Dim::Literal(4),
            heads: None,
            range: Some(projection_spec::contracts::RowRange {
                start: Dim::Literal(i * 4),
                length: Dim::Literal(4),
            }),
            scale_origin: [0, 0],
            head_runs: vec![],
            transforms: vec![transform.clone()],
        })
        .collect();
    let mut registry = Registry::default();
    registry.register(&HOOKS).unwrap();
    let m = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([(
            "matrix.weight".into(),
            info(&[8, 16], "BF16"),
        )]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let bound = d.bind(&g, &registry, &m, &Bf16Rules).unwrap();
    let probe = HookProbe(Default::default());
    let rank = derive::derive(DerivationInput {
        declaration: &d,
        bound: &bound,
        geometry: &g,
        registry: &registry,
        topology: Topology { ranks: 1, rank: 0 },
        imported: &imports,
        variants: &probe,
    })
    .unwrap();
    assert_eq!(rank.operations[0].slices.len(), 2);
    assert_eq!(rank.operations[0].slices[0].target_rows, vec![0..4]);
    assert_eq!(rank.operations[0].slices[1].target_rows, vec![4..8]);
}

#[test]
fn logical_concat_domains_read_only_the_rank_owned_physical_buffer() {
    use projection_spec::inputs::{ActivationInput, ActivationSegment};
    let (mut d, b, g, i) = fixture(
        SplitClass::Row {
            tile: Dim::Literal(8),
        },
        8,
        8,
        ProjectionKind::Dense {},
    );
    let segment = |source: &str, origin: u64| {
        serde_json::from_value::<ActivationSegment>(serde_json::json!({"source":source,"origin":origin,"length":4,"selection":{"select":"rank_k","projection":"matrix","operand":"weight"}})).unwrap()
    };
    d.projections[0].inputs = vec![ActivationInput {
        operand: "weight".into(),
        binding: "eh".into(),
        segments: vec![segment("hidden", 4), segment("embedding", 0)],
    }];
    let rank = run(&d, &b, &g, &i, Topology { ranks: 2, rank: 1 }).unwrap();
    let input = &rank.operations[0].inputs[0];
    assert_eq!(input.segments.len(), 1);
    assert_eq!(input.segments[0].source, "hidden");
    assert_eq!(input.segments[0].columns, 0..4);
    let source = (0..512u16).flat_map(u16::to_le_bytes).collect::<Vec<_>>();
    let buffers = std::collections::BTreeMap::from([(
        "hidden".into(),
        derive::payload::ActivationBuffer {
            dtype: DType::Bf16,
            rows: 128,
            columns: 4,
            bytes: &source,
        },
    )]);
    let actual = derive::payload::input(input, &buffers).unwrap();
    let expected = source
        .chunks_exact(8)
        .flat_map(|row| row.iter().copied().chain([0; 24]))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    d.projections[0].inputs[0].segments[1] = segment("embedding", 4);
    assert!(
        run(&d, &b, &g, &i, Topology { ranks: 2, rank: 1 })
            .unwrap_err()
            .0
            .contains("domains overlap")
    );
}

#[test]
fn format_implementation_maps_dispatch_the_explicit_provider() {
    use projection_spec::contracts::{Composition, Port};
    let (mut d, b, g, i) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    let baseline = run(&d, &b, &g, &i, Topology { ranks: 1, rank: 0 })
        .unwrap()
        .operations[0]
        .clone();
    let port = Port {
        id: "x".into(),
        binding: "x".into(),
        dtype: DType::Bf16,
        interpretation: None,
        shape: vec![Dim::Literal(8), Dim::Literal(16)],
        state: false,
        format: None,
    };
    let variant = projection_spec::Variant {
        id: "fixture/implementation.v1".into(),
        parameters: serde_json::json!({"schedule":"mapped"}),
        numerical_contract: "fixture.implementation.v1".into(),
        inputs: vec![port.clone()],
        outputs: vec![port],
    };
    d.projections[0].composition = Some(Composition {
        implementations: std::collections::BTreeMap::from([(FormatId::Bf16, variant)]),
        ..Default::default()
    });
    let mut registry = Registry::default();
    registry.register(&VARIANTS).unwrap();
    let rank = derive::derive(DerivationInput {
        declaration: &d,
        bound: &b,
        geometry: &g,
        registry: &registry,
        topology: Topology { ranks: 1, rank: 0 },
        imported: &i,
        variants: &Implementations(baseline),
    })
    .unwrap();
    assert!(
        rank.operations
            .iter()
            .all(|o| o.operation.id == "implementation.mapped")
    );
}

#[test]
fn checked_ordinary_helper_retains_selection_and_refuses_bad_context() {
    let (d, b, g, i) = fixture(
        SplitClass::Column {
            tile: Dim::Literal(4),
        },
        8,
        16,
        ProjectionKind::Dense {},
    );
    let t = Topology { ranks: 2, rank: 1 };
    let key = InstanceKey {
        projection: "matrix".into(),
        layer: None,
    };
    let imported = &i[&key];
    let baseline = run(&d, &b, &g, &i, t).unwrap().operations[0].clone();
    let selections = vec![derive::RankSelection {
        rows: std::iter::once(4..8).collect(),
        k: 0..16,
        experts: 0..1,
        logical: MatrixGeometry { rows: 4, k: 16 },
        physical: MatrixGeometry { rows: 4, k: 16 },
    }];
    let mut ctx = derive::ProjectionContext {
        bound: &b,
        source_instance: key,
        selections: Some(&selections),
        declaration: &d,
        projection: &d.projections[0],
        geometry: &g,
        topology: t,
        layer: None,
        source_layer: None,
        usage: &d.projections[0].uses[0],
    };
    assert_eq!(ctx.ordinary(imported).unwrap(), baseline);
    ctx.topology.rank = 2;
    assert!(
        ctx.ordinary(imported)
            .unwrap_err()
            .to_string()
            .contains("topology")
    );
    ctx.topology = t;
    ctx.layer = Some(99);
    assert!(
        ctx.ordinary(imported)
            .unwrap_err()
            .to_string()
            .contains("execution")
    );
    ctx.layer = None;
    let mut forged = imported.clone();
    forged.matrices[0].weight.codes.tensor = "unbound.weight".into();
    assert!(
        ctx.ordinary(&forged)
            .unwrap_err()
            .to_string()
            .contains("authenticated")
    );
}

struct TransformBypass;
impl derive::hooks::VariantDeriver for TransformBypass {
    fn transform(
        &self,
        _: &projection_spec::Variant,
        _: Option<&projection_spec::Part>,
        _: &derive::ProjectionContext<'_>,
        imported: ImportedProjection,
    ) -> Result<derive::SourceTransformResult, projection_spec::Error> {
        let _ = imported;
        Ok(derive::SourceTransformResult::Bypass)
    }
}
#[test]
fn topology_source_transform_bypass_preserves_authenticated_inputs() {
    let (mut d, b, g, i) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    let port = projection_spec::contracts::Port {
        id: "weight".into(),
        binding: "weight".into(),
        dtype: DType::Bf16,
        interpretation: None,
        shape: vec![Dim::Literal(8), Dim::Literal(16)],
        state: false,
        format: Some(FormatId::Bf16),
    };
    d.projections[0].source_transforms = vec![projection_spec::Variant {
        id: "fixture/source.v1".into(),
        parameters: serde_json::json!({}),
        numerical_contract: "fixture.source.v1".into(),
        inputs: vec![port.clone()],
        outputs: vec![projection_spec::contracts::Port {
            format: Some(FormatId::Fp8Rows64),
            ..port
        }],
    }];
    let mut registry = Registry::default();
    registry.register(&HOOKS).unwrap();
    let rank = derive::derive(DerivationInput {
        declaration: &d,
        bound: &b,
        geometry: &g,
        registry: &registry,
        topology: Topology { ranks: 4, rank: 0 },
        imported: &i,
        variants: &TransformBypass,
    })
    .unwrap();
    let imported = i.values().next().unwrap();
    assert!(
        rank.operations
            .iter()
            .all(|op| op.operation.weights == imported.weights)
    );
}

struct PreparedRows {
    corrupt_source: bool,
    wrong_rank: bool,
}
impl derive::hooks::VariantDeriver for PreparedRows {
    fn transform(
        &self,
        _: &projection_spec::Variant,
        _: Option<&projection_spec::Part>,
        ctx: &derive::ProjectionContext<'_>,
        mut imported: ImportedProjection,
    ) -> Result<derive::SourceTransformResult, projection_spec::Error> {
        use kernel_core::weight_transform::{
            PackedSection, SourceTensor, WeightRecord, WeightTransform,
        };
        if ctx.topology.ranks == 4 {
            return Ok(derive::SourceTransformResult::Bypass);
        }
        let source = &imported.matrices[0].weight;
        let row = u64::from(ctx.topology.rank) * 128;
        let mut weight = source.clone();
        weight.format = kernel_core::contract::WeightFormat::Fp8Rows64x128(
            kernel_core::contract::ScaleMode::ExactF32,
        );
        weight.logical_shape = vec![128, 128];
        weight.codes.tensor = "prepared.codes".into();
        weight.codes.shape = vec![128, 128];
        weight.codes.byte_length = 128 * 128;
        let local = weight.local_scale.as_mut().unwrap();
        local.tensor.tensor = "prepared.local".into();
        local.tensor.shape = vec![2, 1];
        local.tensor.byte_length = 8;
        local.groups[0].size = 64;
        let record = |span: &kernel_core::precision::TensorSpan, section, ranges, stored_shape| {
            derive::VariantRecord {
                view: format!(
                    "matrix.bf16.{}",
                    if section == PackedSection::Codes {
                        "codes"
                    } else {
                        "local"
                    }
                ),
                view_offset: 0,
                section,
                record: WeightRecord {
                    transform: WeightTransform::Copy,
                    sources: vec![SourceTensor {
                        dtype: span.dtype.checkpoint_dtype().unwrap().into(),
                        shape: span.shape.clone(),
                        byte_length: span.byte_length,
                    }],
                    axis: Some(0),
                    ranges,
                    stored_shape,
                },
                tensors: vec![span.tensor.clone()],
            }
        };
        let scale = &source.local_scale.as_ref().unwrap().tensor;
        let records = vec![
            record(
                &source.codes,
                PackedSection::Codes,
                std::iter::once(row..row + 128).collect(),
                vec![128, 128],
            ),
            record(
                scale,
                PackedSection::LocalScales,
                vec![row / 128..row / 128 + 1, row / 128..row / 128 + 1],
                vec![2, 1],
            ),
        ];
        imported.preparation = Some(derive::SourcePreparation {
            topology: Topology {
                rank: ctx.topology.rank + u32::from(self.wrong_rank),
                ..ctx.topology
            },
            geometry: MatrixGeometry { rows: 128, k: 128 },
            records,
        });
        imported.format = FormatId::Fp8Rows64;
        imported.weights = vec![weight];
        if self.corrupt_source {
            imported.matrices[0].weight.codes.tensor = "forged.codes".into();
        }
        Ok(derive::SourceTransformResult::Applied(imported))
    }
    fn implementation(
        &self,
        v: &projection_spec::Variant,
        ctx: &derive::ProjectionContext<'_>,
        imported: &ImportedProjection,
    ) -> Result<derive::DerivedOperation, projection_spec::Error> {
        let mut base = ctx.ordinary(imported)?;
        base.operation.arithmetic.numerical_contract = v.numerical_contract.clone();
        // An adapter may replace final records; source preparation still survives.
        base.source_preparation = None;
        Ok(base)
    }
}
#[test]
fn prepared_sources_keep_raw_matrices_and_copy_codes_with_repeated_scale_rows() {
    use kernel_core::precision::{
        CodeEncoding, GroupAxis, ScaleConvention, ScaleScope, ScaleSpec, TensorSpan,
    };
    let (mut d, mut b, g, mut i) = fixture(
        SplitClass::Column {
            tile: Dim::Literal(64),
        },
        256,
        128,
        ProjectionKind::Dense {},
    );
    let mut raw = b.operands[0].imported.weight.clone();
    raw.format = kernel_core::contract::WeightFormat::Fp8Block128x128(
        kernel_core::contract::ScaleMode::ExactF32,
    );
    raw.encoding = CodeEncoding::E4M3;
    raw.codes.dtype = DType::Fp8E4M3;
    raw.codes.byte_length /= 2;
    raw.local_scale = Some(ScaleSpec {
        tensor: TensorSpan {
            tensor: "matrix.scale".into(),
            dtype: DType::F32,
            shape: vec![2, 1],
            offset: 0,
            byte_length: 8,
        },
        scope: ScaleScope::Block,
        convention: ScaleConvention::Multiply,
        groups: vec![
            GroupAxis {
                axis: 0,
                size: 128,
                origin: 0,
            },
            GroupAxis {
                axis: 1,
                size: 128,
                origin: 0,
            },
        ],
    });
    b.operands[0].imported.weight = raw.clone();
    b.operands[0].format = FormatId::Fp8Block128;
    for imported in i.values_mut() {
        imported.format = FormatId::Fp8Block128;
        imported.family = kernel_core::precision::format::KernelFamily::Row;
        imported.weights = vec![raw.clone()];
        imported.matrices[0].weight = raw.clone();
    }
    let port = projection_spec::contracts::Port {
        id: "weight".into(),
        binding: "weight".into(),
        dtype: DType::Fp8E4M3,
        interpretation: None,
        shape: vec![Dim::Literal(256), Dim::Literal(128)],
        state: false,
        format: Some(FormatId::Fp8Block128),
    };
    d.projections[0].source_transforms = vec![projection_spec::Variant {
        id: "fixture/source.v1".into(),
        parameters: serde_json::json!({}),
        numerical_contract: "fixture.source.v1".into(),
        inputs: vec![port.clone()],
        outputs: vec![projection_spec::contracts::Port {
            format: Some(FormatId::Fp8Rows64),
            ..port.clone()
        }],
    }];
    let mut registry = Registry::default();
    registry.register(&HOOKS).unwrap();
    let provider = PreparedRows {
        corrupt_source: false,
        wrong_rank: false,
    };
    let check = |declaration: &Declaration, provider: &PreparedRows, ranks| {
        derive::derive(DerivationInput {
            declaration,
            bound: &b,
            geometry: &g,
            registry: &registry,
            topology: Topology { ranks, rank: 1 },
            imported: &i,
            variants: provider,
        })
    };
    for legacy in [false, true] {
        if legacy {
            d.projections[0].composition = Some(projection_spec::contracts::Composition {
                implementation: Some(projection_spec::Variant {
                    id: "fixture/implementation.v1".into(),
                    parameters: serde_json::json!({}),
                    numerical_contract: "fixture.implementation.v1".into(),
                    inputs: vec![port.clone()],
                    outputs: vec![port.clone()],
                }),
                ..Default::default()
            });
        }
        let rank = check(&d, &provider, 2).unwrap();
        for operation in &rank.operations {
            assert_eq!(operation.slices[0].source.weight, raw);
            assert_eq!(operation.slices[0].format, FormatId::Fp8Block128);
            assert_eq!(
                operation.operation.weights[0].format,
                kernel_core::contract::WeightFormat::Fp8Rows64x128(
                    kernel_core::contract::ScaleMode::ExactF32
                )
            );
            let preparation = operation.source_preparation.as_ref().unwrap();
            assert_eq!(preparation.geometry, operation.operation.geometry.unwrap());
            assert_eq!(preparation.records, operation.variant_records);
            let codes: Vec<u8> = (0..256 * 128).map(|v| (v % 113) as u8).collect();
            let scales = [1.25f32, 2.5f32]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            for record in &preparation.records {
                let source = if record.tensors[0] == "matrix.weight" {
                    &codes
                } else {
                    &scales
                };
                let result = record
                    .record
                    .clone()
                    .prepare(record.record.output_bytes().unwrap())
                    .unwrap()
                    .produce_whole(&mut |_, offset, out| {
                        out.copy_from_slice(&source[offset as usize..offset as usize + out.len()]);
                        Ok(())
                    })
                    .unwrap();
                let expected = if record.tensors[0] == "matrix.weight" {
                    codes[128 * 128..].to_vec()
                } else {
                    scales[4..].repeat(2)
                };
                assert_eq!(result, expected);
            }
        }
        let bypass = check(&d, &provider, 4).unwrap();
        assert!(
            bypass
                .operations
                .iter()
                .all(|o| o.operation.weights[0] == raw && o.source_preparation.is_none())
        );
    }
    assert!(
        check(
            &d,
            &PreparedRows {
                corrupt_source: true,
                wrong_rank: false
            },
            2
        )
        .unwrap_err()
        .to_string()
        .contains("authenticated")
    );
    assert!(
        check(
            &d,
            &PreparedRows {
                corrupt_source: false,
                wrong_rank: true
            },
            2
        )
        .unwrap_err()
        .to_string()
        .contains("topology")
    );
}

struct NativeStage {
    standalone: bool,
    corrupt: u8,
}
impl derive::hooks::VariantDeriver for NativeStage {
    fn stage(
        &self,
        stage: &projection_spec::contracts::Stage,
        ctx: &derive::ProjectionContext<'_>,
    ) -> Result<derive::DerivedStage, projection_spec::Error> {
        use derive::{AdapterValue, ControlBinding, StageAdapter, StageExecution};
        let mut derived =
            derive::hooks::VariantDeriver::stage(&HookProbe(Default::default()), stage, ctx)?;
        let control = |view: &str| {
            AdapterValue::Binding(ControlBinding {
                view: view.into(),
                dtype: DType::I32,
                interpretation: Some(projection_spec::contracts::Interpretation::Unsigned),
                offset_bytes: if self.corrupt == 2 { 1 } else { 0 },
                stride_bytes: Some(4),
            })
        };
        derived.adapter = Some(StageAdapter {
            contract: if self.corrupt == 1 {
                "foreign/stage.v1".into()
            } else {
                stage.contract.id.clone()
            },
            execution: if self.standalone {
                StageExecution::Standalone {
                    symbol: "native_stage".into(),
                }
            } else {
                StageExecution::Fused {
                    operation: "lookup".into(),
                }
            },
            facts: std::collections::BTreeMap::from([
                ("local_heads".into(), AdapterValue::U64(2)),
                ("history_offset_bytes".into(), AdapterValue::U64(64)),
                ("history_item_stride_bytes".into(), AdapterValue::U64(128)),
                ("table_start".into(), AdapterValue::U64(9)),
                ("table_rows".into(), AdapterValue::U64(32)),
                (
                    "addressing".into(),
                    AdapterValue::Text(format!("{:?}", ctx.usage.phase)),
                ),
                (
                    "scale_bits".into(),
                    AdapterValue::F32Bits(1.25f32.to_bits()),
                ),
                ("live_items".into(), control("live_items")),
                ("q_len".into(), control("q_len")),
                ("cu_rows".into(), control("cu_rows")),
                ("positions".into(), control("positions")),
                ("prefix_len".into(), control("prefix_len")),
                ("token_batch".into(), control("token_batch")),
                ("live_tokens".into(), control("live_tokens")),
            ]),
        });
        Ok(derived)
    }
}
#[test]
fn native_stage_adapters_preserve_phase_controls_without_changing_logical_ports() {
    use kernel_core::precision::ExecutionPhase::{Decode, Prefill, Verify};
    use projection_spec::contracts::{Composition, Port, Stage};
    let (mut d, b, g, imports) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    let port = Port {
        id: "history".into(),
        binding: "history".into(),
        dtype: DType::I64,
        interpretation: None,
        shape: vec![Dim::Literal(2)],
        state: true,
        format: None,
    };
    let stage = Stage {
        id: "address".into(),
        phases: vec![Prefill, Decode, Verify],
        mutates: vec!["history".into()],
        contract: projection_spec::Variant {
            id: "fixture/stage.v1".into(),
            numerical_contract: "fixture.stage.v1".into(),
            parameters: serde_json::json!({}),
            inputs: vec![port.clone()],
            outputs: vec![port],
        },
    };
    d.projections[0].composition = Some(Composition {
        pre: vec![stage],
        ..Default::default()
    });
    let mut registry = Registry::default();
    registry.register(&HOOKS).unwrap();
    let run = |provider: &NativeStage| {
        derive::derive(DerivationInput {
            declaration: &d,
            bound: &b,
            geometry: &g,
            registry: &registry,
            topology: Topology { ranks: 2, rank: 1 },
            imported: &imports,
            variants: provider,
        })
    };
    for standalone in [false, true] {
        let provider = NativeStage {
            standalone,
            corrupt: 0,
        };
        let rank = run(&provider).unwrap();
        assert_eq!(rank.operations.len(), d.projections[0].uses.len());
        for operation in rank.operations {
            assert_eq!(operation.stages.len(), 1);
            let resolved = &operation.stages[0];
            assert_eq!(resolved.inputs.len(), 1);
            assert_eq!(resolved.inputs[0].id, "history");
            let payload = resolved
                .adapter
                .as_ref()
                .expect("provider addressing must survive derivation");
            assert_eq!(
                payload.facts["addressing"],
                derive::AdapterValue::Text(format!("{:?}", operation.requested.phase))
            );
            assert_eq!(
                matches!(payload.execution, derive::StageExecution::Standalone { .. }),
                standalone
            );
            assert_eq!(
                serde_json::from_str::<derive::StageAdapter>(
                    &serde_json::to_string(payload).unwrap()
                )
                .unwrap(),
                *payload
            );
        }
    }
    assert!(
        run(&NativeStage {
            standalone: false,
            corrupt: 1
        })
        .unwrap_err()
        .to_string()
        .contains("identity")
    );
    assert!(
        run(&NativeStage {
            standalone: false,
            corrupt: 2
        })
        .unwrap_err()
        .to_string()
        .contains("control binding")
    );
}

#[test]
fn aliased_fusion_eligibility_uses_actual_source_roles_and_keeps_execution_layer() {
    let (mut d, _, g, mut imports) = cross_stem_fixture();
    for index in [0, 1] {
        let mut p = d.projections[index].clone();
        p.id = if index == 0 { "second" } else { "alias_second" }.into();
        p.source.as_mut().unwrap().tensor = "{stem}.second.weight".into();
        d.projections.push(p);
    }
    d.fusion_groups = vec![serde_json::from_value(serde_json::json!({"id":"aliases","input":"hidden",
        "members":["alias","alias_second"], "eligibility":{"predicate":"all","formats":["bf16"],"activation":null,"phases":[]}})).unwrap()];
    let metadata = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([
            ("main.0.matrix.weight".into(), info(&[8, 16], "BF16")),
            ("main.0.second.weight".into(), info(&[8, 16], "BF16")),
            ("aux.1.matrix.weight".into(), info(&[8, 16], "BF16")),
            ("aux.1.second.weight".into(), info(&[8, 16], "BF16")),
        ]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let mut b = d
        .bind(&g, &Registry::default(), &metadata, &Bf16Rules)
        .unwrap();
    for (owner, layer, tensor) in [
        ("second", 0, "main.0.second.weight"),
        ("alias_second", 1, "aux.1.second.weight"),
    ] {
        let mut import = imports.values().next().unwrap().clone();
        import.weights[0] = b
            .operands
            .iter()
            .find(|o| o.imported.weight.codes.tensor == tensor)
            .unwrap()
            .imported
            .weight
            .clone();
        import.matrices[0].weight = import.weights[0].clone();
        imports.insert(
            InstanceKey {
                projection: owner.into(),
                layer: Some(layer),
            },
            import,
        );
    }
    let rank = run(&d, &b, &g, &imports, Topology { ranks: 1, rank: 0 }).unwrap();
    assert!(rank.fusions.iter().all(|f| f.layer == Some(9) && f.fused));
    assert!(
        rank.fusions
            .iter()
            .flat_map(|f| &f.members)
            .all(|m| m.layer == Some(9))
    );
    // The auxiliary materialization remains BF16; execution uses these FP8 sources.
    for operand in b.operands.iter_mut().filter(|o| o.layer == Some(0)) {
        let weight = &mut operand.imported.weight;
        weight.format = kernel_core::contract::WeightFormat::Fp8PerChannel;
        weight.encoding = kernel_core::precision::CodeEncoding::E4M3;
        weight.codes.dtype = DType::Fp8E4M3;
        weight.codes.byte_length /= 2;
        weight.local_scale = Some(kernel_core::precision::ScaleSpec {
            tensor: kernel_core::precision::TensorSpan {
                tensor: format!("{}.scale", weight.codes.tensor),
                dtype: DType::F32,
                shape: vec![8],
                offset: 0,
                byte_length: 32,
            },
            scope: kernel_core::precision::ScaleScope::Channel,
            convention: kernel_core::precision::ScaleConvention::Multiply,
            groups: vec![kernel_core::precision::GroupAxis {
                axis: 0,
                size: 1,
                origin: 0,
            }],
        });
        operand.format = FormatId::Fp8PerChannel;
        let imported = imports
            .get_mut(&InstanceKey {
                projection: operand.projection.clone(),
                layer: Some(0),
            })
            .unwrap();
        imported.format = FormatId::Fp8PerChannel;
        imported.weights = vec![weight.clone()];
        imported.matrices[0].weight = weight.clone();
    }
    assert!(b.fusion_groups.iter().all(|f| f.fused));
    let rank = run(&d, &b, &g, &imports, Topology { ranks: 1, rank: 0 }).unwrap();
    assert!(rank.fusions.iter().all(|f| !f.fused));
    d.fusion_groups[0].fallback = projection_spec::contracts::Fallback::Refuse;
    assert!(
        run(&d, &b, &g, &imports, Topology { ranks: 1, rank: 0 })
            .unwrap_err()
            .to_string()
            .contains("fallback refuses")
    );
}

#[test]
fn generic_fusion_preserves_declared_native_stages() {
    use kernel_core::precision::ExecutionPhase::{Decode, Prefill, Verify};
    use projection_spec::contracts::{Composition, Port, Stage};
    let (mut d, b, g, imports) = paired_fixture(8, 16);
    let port = Port {
        id: "history".into(),
        binding: "history".into(),
        dtype: DType::I64,
        interpretation: None,
        shape: vec![Dim::Literal(2)],
        state: true,
        format: None,
    };
    d.fusion_groups[0].composition = Some(Composition {
        pre: vec![Stage {
            id: "address".into(),
            phases: vec![Prefill, Decode, Verify],
            mutates: vec!["history".into()],
            contract: projection_spec::Variant {
                id: "fixture/stage.v1".into(),
                numerical_contract: "fixture.stage.v1".into(),
                parameters: serde_json::json!({}),
                inputs: vec![port.clone()],
                outputs: vec![port],
            },
        }],
        ..Default::default()
    });
    let mut registry = Registry::default();
    registry.register(&HOOKS).unwrap();
    let rank = derive::derive(DerivationInput {
        declaration: &d,
        bound: &b,
        geometry: &g,
        registry: &registry,
        topology: Topology { ranks: 1, rank: 0 },
        imported: &imports,
        variants: &NativeStage {
            standalone: false,
            corrupt: 0,
        },
    })
    .unwrap();
    assert!(
        rank.fusions
            .iter()
            .all(|f| f.stages.len() == 1 && f.stages[0].adapter.is_some())
    );
}

#[test]
fn rank_local_assembly_rows_are_owned_before_destination_splitting() {
    use projection_spec::contracts::{RowCoordinates, RowRange};
    let (mut d, _, g, imports) = paired_fixture(8, 16);
    d.fusion_groups.clear();
    d.projections[0].split = SplitClass::Column {
        tile: Dim::Literal(4),
    };
    let destination = &mut d.projections[1];
    destination.shape[0] = Dim::Literal(2);
    destination.split = SplitClass::Column {
        tile: Dim::Literal(1),
    };
    destination.assembly[0].coordinates = RowCoordinates::RankLocal;
    destination.assembly[0].rows = RowRange {
        start: Dim::Literal(1),
        length: Dim::Literal(2),
    };
    let m = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([(
            "matrix.weight".into(),
            info(&[8, 16], "BF16"),
        )]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let bound = d.bind(&g, &Registry::default(), &m, &Bf16Rules).unwrap();
    for rank in 0..2 {
        let result = run(&d, &bound, &g, &imports, Topology { ranks: 2, rank }).unwrap();
        let operation = result
            .operations
            .iter()
            .find(|o| o.instance.projection == "other")
            .unwrap();
        let start = 1 + 4 * u64::from(rank);
        assert_eq!(
            operation.slices[0].rows,
            std::iter::once(start..start + 2).collect::<Vec<_>>()
        );
        assert_eq!(operation.logical_geometry.rows, 2);
    }
}

#[test]
fn assembled_projection_applies_the_selected_source_format_recipe() {
    let (mut d, _, g, imports) = paired_fixture(8, 16);
    d.fusion_groups.clear();
    d.projections[1].composition = Some(projection_spec::contracts::Composition {
        numerical_contracts: std::collections::BTreeMap::from([(
            FormatId::Bf16,
            projection_spec::contracts::NumericalRecipe {
                numerical_contract: "fixture.assembly.v1".into(),
                a16_recipe: A16Recipe::ScaledWeight,
            },
        )]),
        ..Default::default()
    });
    let m = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([(
            "matrix.weight".into(),
            info(&[8, 16], "BF16"),
        )]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let bound = d.bind(&g, &Registry::default(), &m, &Bf16Rules).unwrap();
    let result = run(&d, &bound, &g, &imports, Topology { ranks: 2, rank: 1 }).unwrap();
    let assembled = result
        .operations
        .iter()
        .filter(|o| o.instance.projection == "other")
        .collect::<Vec<_>>();
    assert!(!assembled.is_empty());
    for operation in assembled {
        assert_eq!(
            operation.operation.arithmetic.numerical_contract,
            "fixture.assembly.v1"
        );
        assert_eq!(operation.a16_recipe, A16Recipe::ScaledWeight);
    }
}

#[test]
fn assembled_block_recipe_changes_a16_decode_order() {
    use kernel_core::precision::{
        CodeEncoding, DecodeOrder, GroupAxis, ScaleConvention, ScaleScope, ScaleSpec, TensorSpan,
    };
    let (mut d, mut b, g, mut imports) = paired_fixture(128, 128);
    d.fusion_groups.clear();
    let mut weight = b.operands[0].imported.weight.clone();
    weight.format = kernel_core::contract::WeightFormat::Fp8Block128x128(
        kernel_core::contract::ScaleMode::ExactF32,
    );
    weight.encoding = CodeEncoding::E4M3;
    weight.codes.dtype = DType::Fp8E4M3;
    weight.codes.byte_length /= 2;
    weight.local_scale = Some(ScaleSpec {
        tensor: TensorSpan {
            tensor: "matrix.scale".into(),
            dtype: DType::F32,
            shape: vec![1, 1],
            offset: 0,
            byte_length: 4,
        },
        scope: ScaleScope::Block,
        convention: ScaleConvention::Multiply,
        groups: vec![
            GroupAxis {
                axis: 0,
                size: 128,
                origin: 0,
            },
            GroupAxis {
                axis: 1,
                size: 128,
                origin: 0,
            },
        ],
    });
    b.operands[0].format = FormatId::Fp8Block128;
    b.operands[0].imported.weight = weight.clone();
    for import in imports.values_mut() {
        import.format = FormatId::Fp8Block128;
        import.weights = vec![weight.clone()];
        import.matrices[0].weight = weight.clone();
    }
    d.projections[1].composition = Some(projection_spec::contracts::Composition {
        numerical_contracts: std::collections::BTreeMap::from([(
            FormatId::Fp8Block128,
            projection_spec::contracts::NumericalRecipe {
                numerical_contract: "fixture.assembly.block.v1".into(),
                a16_recipe: A16Recipe::ScaleAfterBlockMma,
            },
        )]),
        ..Default::default()
    });
    let rank = run(&d, &b, &g, &imports, Topology { ranks: 1, rank: 0 }).unwrap();
    let operations = rank
        .operations
        .iter()
        .filter(|o| o.instance.projection == "other")
        .collect::<Vec<_>>();
    assert!(!operations.is_empty());
    for op in operations {
        assert_eq!(op.a16_recipe, A16Recipe::ScaleAfterBlockMma);
        assert_eq!(
            op.operation.arithmetic.decode_order,
            DecodeOrder::BlockScaleAfterMma
        );
        assert_eq!(
            op.operation.arithmetic.numerical_contract,
            "fixture.assembly.block.v1"
        );
    }
}

#[test]
fn expert_assemblies_refuse_mixed_inheritance_in_either_member_order() {
    use projection_spec::contracts::{RowRange, SourceMember};
    let (mut d, _, g, _) = fixture(
        SplitClass::ExpertOwned {
            tile: Dim::Literal(1),
        },
        8,
        16,
        ProjectionKind::ExpertBank {
            experts: Dim::Literal(4),
            top_k: Dim::Literal(2),
        },
    );
    let mut p = d.projections[0].clone();
    p.id = "assembly".into();
    p.source = None;
    let member = |start, inherited| SourceMember {
        coordinates: Default::default(),
        inherit_split: inherited,
        projection: "matrix".into(),
        layer: None,
        operand: None,
        rows: RowRange {
            start: Dim::Literal(start),
            length: Dim::Literal(4),
        },
    };
    p.assembly = vec![member(0, false), member(4, true)];
    d.projections.push(p);
    for reverse in [false, true] {
        if reverse {
            d.projections[1].assembly.reverse();
        }
        assert!(
            d.validate(&g, &Registry::default())
                .unwrap_err()
                .to_string()
                .contains("expert inheritance")
        );
    }
}

#[test]
fn expert_assembly_members_keep_one_source_expert_per_local_slot() {
    use projection_spec::contracts::{RowRange, SourceMember};
    for inherited in [false, true] {
        let (mut d, _, g, mut imports) = fixture(
            SplitClass::ExpertOwned {
                tile: Dim::Literal(1),
            },
            8,
            16,
            ProjectionKind::ExpertBank {
                experts: Dim::Literal(4),
                top_k: Dim::Literal(2),
            },
        );
        let mut p = d.projections[0].clone();
        p.id = "assembly".into();
        p.source = None;
        p.assembly = [0, 4]
            .into_iter()
            .map(|start| SourceMember {
                coordinates: Default::default(),
                inherit_split: inherited,
                projection: "matrix".into(),
                layer: None,
                operand: None,
                rows: RowRange {
                    start: Dim::Literal(start),
                    length: Dim::Literal(4),
                },
            })
            .collect();
        d.projections.push(p);
        let m = CheckpointMetadata {
            tensors: std::collections::BTreeMap::from([(
                "matrix.weight".into(),
                info(&[4, 8, 16], "BF16"),
            )]),
            quantization_config: serde_json::json!({"recipe":"synthetic"}),
            ..Default::default()
        };
        let bound = d.bind(&g, &Registry::default(), &m, &Bf16Rules).unwrap();
        let mut prototype = imports.values().next().unwrap().clone();
        prototype.operation_id = "assembly.{output}".into();
        prototype.matrices.clear();
        imports.insert(
            InstanceKey {
                projection: "assembly".into(),
                layer: None,
            },
            prototype,
        );
        let rank = run(&d, &bound, &g, &imports, Topology { ranks: 2, rank: 1 }).unwrap();
        let ops = rank
            .operations
            .iter()
            .filter(|o| o.instance.projection == "assembly")
            .collect::<Vec<_>>();
        assert!(!ops.is_empty());
        for op in ops {
            assert_eq!(op.slices.len(), 4);
            for slice in &op.slices {
                assert_eq!(slice.source.source_expert, u64::from(slice.expert) + 2);
            }
        }
    }
}

#[test]
fn activation_rows_must_match_weight_k_offset_as_well_as_width() {
    use projection_spec::contracts::RowRange;
    use projection_spec::inputs::{ActivationInput, ActivationSegment, ActivationSelection};
    let (mut d, b, g, imports) = fixture(
        SplitClass::Row {
            tile: Dim::Literal(8),
        },
        8,
        8,
        ProjectionKind::Dense {},
    );
    d.projections[0].inputs = vec![ActivationInput {
        operand: "weight".into(),
        binding: "hidden".into(),
        segments: vec![ActivationSegment {
            source: "hidden".into(),
            origin: None,
            length: None,
            selection: ActivationSelection::Rows {
                rows: RowRange {
                    start: Dim::Literal(0),
                    length: Dim::Literal(4),
                },
            },
        }],
    }];
    let t = Topology { ranks: 2, rank: 1 };
    assert!(
        run(&d, &b, &g, &imports, t)
            .unwrap_err()
            .to_string()
            .contains("weight K ownership")
    );
    d.projections[0].inputs[0].segments[0].selection = ActivationSelection::Rows {
        rows: RowRange {
            start: Dim::Literal(4),
            length: Dim::Literal(4),
        },
    };
    assert_eq!(
        run(&d, &b, &g, &imports, t).unwrap().operations[0].inputs[0].segments[0].columns,
        4..8
    );
    let mut other = d.projections[0].clone();
    other.id = "other".into();
    other.inputs.clear();
    other.shape[1] = Dim::Literal(4);
    other.split = SplitClass::Replicate {};
    other.source.as_mut().unwrap().tensor = "other.weight".into();
    d.projections.push(other);
    d.projections[0].inputs[0].segments[0].selection = ActivationSelection::RankK {
        projection: "other".into(),
        operand: "weight".into(),
    };
    assert!(
        run(&d, &b, &g, &imports, t)
            .unwrap_err()
            .to_string()
            .contains("weight K ownership")
    );
}

#[test]
fn source_matrices_cannot_swap_named_operands_in_derive_or_ordinary() {
    let (mut d, _, g, mut imports) = fixture(
        SplitClass::Column {
            tile: Dim::Literal(4),
        },
        8,
        16,
        ProjectionKind::Dense {},
    );
    d.projections[0].operands = vec!["gate".into(), "up".into()];
    d.projections[0].source.as_mut().unwrap().tensor = "matrix.{operand}.weight".into();
    let m = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([
            ("matrix.gate.weight".into(), info(&[8, 16], "BF16")),
            ("matrix.up.weight".into(), info(&[8, 16], "BF16")),
        ]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let b = d.bind(&g, &Registry::default(), &m, &Bf16Rules).unwrap();
    let key = InstanceKey {
        projection: "matrix".into(),
        layer: None,
    };
    let imported = imports.get_mut(&key).unwrap();
    imported.weights = (0..2)
        .map(|n| {
            b.operands
                .iter()
                .find(|b| b.operand == Some(n))
                .unwrap()
                .imported
                .weight
                .clone()
        })
        .collect();
    imported.activations = vec![ActivationSpec::bf16(); 2];
    imported.matrices = (0..2)
        .map(|n| SourceMatrix {
            weight: imported.weights[n].clone(),
            operand: n as u32,
            expert: 0,
            source_expert: 0,
            target_rows: 0..8,
            source_rows: 0..8,
            target_k: 0..16,
            source_k: 0..16,
        })
        .collect();
    let t = Topology { ranks: 2, rank: 0 };
    assert!(run(&d, &b, &g, &imports, t).is_ok());
    let imported = imports.get_mut(&key).unwrap();
    imported.matrices[0].operand = 1;
    imported.matrices[1].operand = 0;
    assert!(
        run(&d, &b, &g, &imports, t)
            .unwrap_err()
            .to_string()
            .contains("binding placement")
    );
    let ctx = derive::ProjectionContext {
        declaration: &d,
        bound: &b,
        projection: &d.projections[0],
        geometry: &g,
        topology: t,
        layer: None,
        source_layer: None,
        source_instance: key.clone(),
        selections: None,
        usage: &d.projections[0].uses[0],
    };
    assert!(
        ctx.ordinary(&imports[&key])
            .unwrap_err()
            .to_string()
            .contains("binding placement")
    );
}
#[test]
fn source_matrices_cannot_reassign_bank_experts() {
    let (d, b, g, mut imports) = fixture(
        SplitClass::ExpertOwned {
            tile: Dim::Literal(1),
        },
        8,
        16,
        ProjectionKind::ExpertBank {
            experts: Dim::Literal(4),
            top_k: Dim::Literal(2),
        },
    );
    imports.values_mut().next().unwrap().matrices[0].source_expert = 1;
    assert!(
        run(&d, &b, &g, &imports, Topology { ranks: 2, rank: 0 })
            .unwrap_err()
            .to_string()
            .contains("binding placement")
    );
}

#[test]
fn source_matrices_keep_row_stacked_operands_and_declared_part_positions() {
    let (mut d, _, g, mut imports) =
        fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    d.projections[0].operands = vec!["gate".into(), "up".into()];
    let metadata = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([(
            "matrix.weight".into(),
            info(&[16, 16], "BF16"),
        )]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let bound = d
        .bind(&g, &Registry::default(), &metadata, &Bf16Rules)
        .unwrap();
    let weight = bound.operands[0].imported.weight.clone();
    let import = imports.values_mut().next().unwrap();
    import.weights = vec![weight.clone(); 2];
    import.activations = vec![ActivationSpec::bf16(); 2];
    import.matrices = (0..2)
        .map(|n| SourceMatrix {
            weight: weight.clone(),
            operand: n,
            expert: 0,
            source_expert: 0,
            target_rows: 0..8,
            source_rows: u64::from(n) * 8..u64::from(n + 1) * 8,
            target_k: 0..16,
            source_k: 0..16,
        })
        .collect();
    let t = Topology { ranks: 1, rank: 0 };
    assert!(run(&d, &bound, &g, &imports, t).is_ok());
    imports.values_mut().next().unwrap().matrices[0].source_rows = 8..16;
    assert!(
        run(&d, &bound, &g, &imports, t)
            .unwrap_err()
            .to_string()
            .contains("binding placement")
    );
    let (mut d, _, g, mut imports) =
        fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    d.projections[0].parts = (0..2)
        .map(|i| projection_spec::Part {
            id: format!("part{i}"),
            rows: Dim::Literal(4),
            heads: None,
            scale_origin: [0, 0],
            range: None,
            head_runs: vec![],
            transforms: vec![],
        })
        .collect();
    let metadata = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([(
            "matrix.weight".into(),
            info(&[8, 16], "BF16"),
        )]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let bound = d
        .bind(&g, &Registry::default(), &metadata, &Bf16Rules)
        .unwrap();
    imports.values_mut().next().unwrap().matrices = bound.operands[0]
        .imported
        .parts
        .iter()
        .enumerate()
        .map(|(i, w)| SourceMatrix {
            weight: w.clone(),
            operand: 0,
            expert: 0,
            source_expert: 0,
            target_rows: i as u64 * 4..(i as u64 + 1) * 4,
            source_rows: 0..4,
            target_k: 0..16,
            source_k: 0..16,
        })
        .collect();
    assert!(run(&d, &bound, &g, &imports, t).is_ok());
    let matrices = &mut imports.values_mut().next().unwrap().matrices;
    matrices[0].target_rows = 4..8;
    matrices[1].target_rows = 0..4;
    assert!(
        run(&d, &bound, &g, &imports, t)
            .unwrap_err()
            .to_string()
            .contains("binding placement")
    );
}

#[test]
fn a_whole_operand_exposed_as_one_part_has_one_authenticated_placement() {
    let (mut d, _, g, imports) = fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    d.projections[0].parts = vec![projection_spec::Part {
        id: "whole".into(),
        rows: Dim::Literal(8),
        heads: None,
        scale_origin: [0, 0],
        range: None,
        head_runs: vec![],
        transforms: vec![],
    }];
    let metadata = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([(
            "matrix.weight".into(),
            info(&[8, 16], "BF16"),
        )]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let bound = d
        .bind(&g, &Registry::default(), &metadata, &Bf16Rules)
        .unwrap();
    assert_eq!(
        bound.operands[0].imported.weight,
        bound.operands[0].imported.parts[0]
    );
    assert!(run(&d, &bound, &g, &imports, Topology { ranks: 1, rank: 0 }).is_ok());
}

struct ChannelParts;
impl projection_spec::bind::CompanionRules for ChannelParts {
    fn import(
        &self,
        r: &projection_spec::bind::BindingRequest<'_>,
        _: &CheckpointMetadata,
    ) -> Result<projection_spec::bind::ImportedOperand, projection_spec::Error> {
        use kernel_core::precision::{
            CodeEncoding, GroupAxis, ScaleConvention, ScaleScope, ScaleSpec, TensorSpan,
            WeightLayout, WeightSpec,
        };
        let weight = WeightSpec {
            codes: TensorSpan {
                tensor: r.tensor.into(),
                dtype: DType::Fp8E4M3,
                shape: vec![8, 16],
                offset: 0,
                byte_length: 128,
            },
            logical_shape: vec![8, 16],
            format: kernel_core::contract::WeightFormat::Fp8PerChannel,
            encoding: CodeEncoding::E4M3,
            layout: WeightLayout::row_major(),
            local_scale: Some(ScaleSpec {
                tensor: TensorSpan {
                    tensor: "matrix.scale".into(),
                    dtype: DType::F32,
                    shape: vec![8],
                    offset: 0,
                    byte_length: 32,
                },
                scope: ScaleScope::Channel,
                convention: ScaleConvention::Multiply,
                groups: vec![GroupAxis {
                    axis: 0,
                    size: 1,
                    origin: 0,
                }],
            }),
            global_scale: None,
            transforms: vec![],
        };
        let parts = (0..2)
            .map(|i| {
                let mut p = weight.clone();
                p.logical_shape = vec![4, 16];
                p.codes.shape = vec![4, 16];
                p.codes.offset = i * 64;
                p.codes.byte_length = 64;
                let scale = p.local_scale.as_mut().unwrap();
                scale.tensor.shape = vec![4];
                scale.tensor.offset = i * 16;
                scale.tensor.byte_length = 16;
                p
            })
            .collect();
        Ok(projection_spec::bind::ImportedOperand {
            weight,
            parts,
            companions: vec![],
            calibrations: vec![],
        })
    }
}
struct WholeReads(bool);
impl derive::hooks::VariantDeriver for WholeReads {
    fn transform(
        &self,
        _: &projection_spec::Variant,
        _: Option<&projection_spec::Part>,
        ctx: &derive::ProjectionContext<'_>,
        mut imported: ImportedProjection,
    ) -> Result<derive::SourceTransformResult, projection_spec::Error> {
        use kernel_core::weight_transform::{
            PackedSection, SourceTensor, WeightRecord, WeightTransform,
        };
        let full = &ctx.bound.operands[0].imported.weight;
        let records = [
            (&full.codes, PackedSection::Codes, "codes"),
            (
                &full.local_scale.as_ref().unwrap().tensor,
                PackedSection::LocalScales,
                "local",
            ),
        ]
        .into_iter()
        .map(|(span, section, suffix)| derive::VariantRecord {
            view: format!("matrix.bf16.{suffix}"),
            view_offset: 0,
            section,
            record: WeightRecord {
                transform: WeightTransform::Copy,
                sources: vec![SourceTensor {
                    dtype: span.dtype.checkpoint_dtype().unwrap().into(),
                    shape: span.shape.clone(),
                    byte_length: span.byte_length,
                }],
                axis: None,
                ranges: vec![],
                stored_shape: span.shape.clone(),
            },
            tensors: vec![if self.0 {
                "outside.tensor".into()
            } else {
                span.tensor.clone()
            }],
        })
        .collect();
        imported.preparation = Some(derive::SourcePreparation {
            topology: ctx.topology,
            geometry: MatrixGeometry { rows: 8, k: 16 },
            records,
        });
        Ok(derive::SourceTransformResult::Applied(imported))
    }
}
#[test]
fn part_framed_preparation_records_can_read_the_bound_whole_tensors() {
    let (mut d, _, g, mut imports) =
        fixture(SplitClass::Replicate {}, 8, 16, ProjectionKind::Dense {});
    d.projections[0].parts = (0..2)
        .map(|i| projection_spec::Part {
            id: format!("part{i}"),
            rows: Dim::Literal(4),
            heads: None,
            range: None,
            scale_origin: [0, 0],
            head_runs: vec![],
            transforms: vec![],
        })
        .collect();
    let port = projection_spec::contracts::Port {
        id: "weight".into(),
        binding: "weight".into(),
        dtype: DType::Fp8E4M3,
        interpretation: None,
        shape: vec![Dim::Literal(8), Dim::Literal(16)],
        state: false,
        format: Some(FormatId::Fp8PerChannel),
    };
    d.projections[0].source_transforms = vec![projection_spec::Variant {
        id: "fixture/source.v1".into(),
        numerical_contract: "fixture.source.v1".into(),
        parameters: serde_json::json!({}),
        inputs: vec![port.clone()],
        outputs: vec![port],
    }];
    let mut registry = Registry::default();
    registry.register(&HOOKS).unwrap();
    let metadata = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([
            ("matrix.weight".into(), info(&[8, 16], "F8_E4M3")),
            ("matrix.scale".into(), info(&[8], "F32")),
        ]),
        ..Default::default()
    };
    let bound = d.bind(&g, &registry, &metadata, &ChannelParts).unwrap();
    let import = imports.values_mut().next().unwrap();
    import.format = FormatId::Fp8PerChannel;
    import.weights = vec![bound.operands[0].imported.weight.clone()];
    import.matrices = bound.operands[0]
        .imported
        .parts
        .iter()
        .enumerate()
        .map(|(i, w)| SourceMatrix {
            weight: w.clone(),
            expert: 0,
            operand: 0,
            source_expert: 0,
            target_rows: i as u64 * 4..(i as u64 + 1) * 4,
            source_rows: 0..4,
            target_k: 0..16,
            source_k: 0..16,
        })
        .collect();
    let run = |provider: &WholeReads| {
        derive::derive(DerivationInput {
            declaration: &d,
            bound: &bound,
            geometry: &g,
            registry: &registry,
            topology: Topology { ranks: 1, rank: 0 },
            imported: &imports,
            variants: provider,
        })
    };
    let rank = run(&WholeReads(false)).unwrap();
    for op in rank.operations {
        assert_eq!(op.slices.len(), 2);
        assert_eq!(op.slices[1].source.weight.codes.offset, 64);
        assert_eq!(op.variant_records[0].record.sources[0].byte_length, 128);
        assert_eq!(op.variant_records[1].record.sources[0].byte_length, 32);
    }
    assert!(
        run(&WholeReads(true))
            .unwrap_err()
            .to_string()
            .contains("authenticated")
    );
}

#[test]
fn dense_bf16_fragment_padding_admits_every_declared_use() {
    let (d, b, g, i) = fixture(
        SplitClass::Column {
            tile: Dim::Literal(16),
        },
        48,
        2560,
        ProjectionKind::Dense {},
    );
    let result = run(&d, &b, &g, &i, Topology { ranks: 4, rank: 0 }).unwrap();
    for operation in result.operations {
        assert_eq!(operation.logical_geometry.rows, 12);
        assert_eq!(operation.operation.matrix_geometry().unwrap().rows, 64);
        assert_eq!(operation.packing.rows, 64);
        assert_eq!(operation.slices[0].geometry.rows, 64);
    }
}

#[test]
fn cross_stem_assembly_alias_resolves_primitive_roles_without_equal_compositions() {
    use projection_spec::contracts::{RowCoordinates, RowRange, SourceMember};
    let (mut d, _, g, mut imports) = cross_stem_fixture();
    d.projections[0].split = SplitClass::Column {
        tile: Dim::Literal(4),
    };
    let alias = &mut d.projections[1];
    alias.source = None;
    alias.stem = Some("aux".into());
    alias.assembly = vec![SourceMember {
        projection: "matrix".into(),
        layer: Some(projection_spec::LayerSelector::Index(Dim::Literal(0))),
        operand: Some("weight".into()),
        rows: RowRange {
            start: Dim::Literal(0),
            length: Dim::Literal(8),
        },
        coordinates: RowCoordinates::Global,
        inherit_split: true,
    }];
    let metadata = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([(
            "main.0.matrix.weight".into(),
            info(&[8, 16], "BF16"),
        )]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let bound = d
        .bind(&g, &Registry::default(), &metadata, &Bf16Rules)
        .unwrap();
    imports
        .get_mut(&InstanceKey {
            projection: "alias".into(),
            layer: Some(1),
        })
        .unwrap()
        .matrices
        .clear();
    for rank in 0..2 {
        let t = Topology { ranks: 2, rank };
        let result = run(&d, &bound, &g, &imports, t).unwrap();
        let alias = result
            .operations
            .iter()
            .find(|o| o.instance.projection == "alias")
            .unwrap();
        assert_eq!(alias.instance.layer, Some(9));
        assert_eq!(alias.source_layer, Some(0));
        assert_eq!(
            alias.source_instance,
            InstanceKey {
                projection: "matrix".into(),
                layer: Some(0)
            }
        );
        assert_eq!(alias.operation.id, "aux.9.alias.bf16");
        assert_eq!(alias.logical_geometry.rows, 4);
        assert_eq!(
            alias.slices[0].rows,
            vec![u64::from(rank) * 4..u64::from(rank + 1) * 4]
        );
        assert_eq!(
            alias.slices[0].source.weight.codes.tensor,
            "main.0.matrix.weight"
        );
    }
}

#[test]
fn registered_rank_local_assembly_uses_one_authenticated_part_representation() {
    use projection_spec::contracts::{Port, RowCoordinates, RowRange};
    let (mut d, _, g, mut imports) = paired_fixture(512, 128);
    d.fusion_groups.clear();
    d.projections[0].split = SplitClass::Column {
        tile: Dim::Literal(128),
    };
    d.projections[0].parts = (0..4)
        .map(|i| projection_spec::Part {
            id: format!("chunk-{i}"),
            rows: Dim::Literal(128),
            heads: None,
            scale_origin: [0, 0],
            range: None,
            head_runs: vec![],
            transforms: vec![],
        })
        .collect();
    let port = Port {
        id: "weight".into(),
        binding: "weight".into(),
        dtype: DType::Bf16,
        interpretation: None,
        shape: vec![Dim::Literal(512), Dim::Literal(128)],
        state: false,
        format: Some(FormatId::Bf16),
    };
    d.projections[0].source_transforms = vec![projection_spec::Variant {
        id: "fixture/source.v1".into(),
        parameters: serde_json::json!({}),
        numerical_contract: "fixture.source.v1".into(),
        inputs: vec![port.clone()],
        outputs: vec![port],
    }];
    let destination = &mut d.projections[1];
    destination.shape[0] = Dim::Literal(128);
    destination.assembly[0].coordinates = RowCoordinates::RankLocal;
    destination.assembly[0].inherit_split = true;
    destination.assembly[0].rows = RowRange {
        start: Dim::Literal(128),
        length: Dim::Literal(128),
    };
    let metadata = CheckpointMetadata {
        tensors: std::collections::BTreeMap::from([(
            "matrix.weight".into(),
            info(&[512, 128], "BF16"),
        )]),
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    };
    let mut registry = Registry::default();
    registry.register(&HOOKS).unwrap();
    let bound = d.bind(&g, &registry, &metadata, &Bf16Rules).unwrap();
    imports
        .get_mut(&InstanceKey {
            projection: "matrix".into(),
            layer: None,
        })
        .unwrap()
        .matrices = bound.operands[0]
        .imported
        .parts
        .iter()
        .enumerate()
        .map(|(i, w)| SourceMatrix {
            weight: w.clone(),
            expert: 0,
            operand: 0,
            source_expert: 0,
            target_rows: i as u64 * 128..(i as u64 + 1) * 128,
            source_rows: 0..128,
            target_k: 0..128,
            source_k: 0..128,
        })
        .collect();
    for rank in 0..2 {
        let result = derive::derive(DerivationInput {
            declaration: &d,
            bound: &bound,
            geometry: &g,
            registry: &registry,
            topology: Topology { ranks: 2, rank },
            imported: &imports,
            variants: &TransformBypass,
        })
        .unwrap();
        let operation = result
            .operations
            .iter()
            .find(|o| o.instance.projection == "other")
            .unwrap();
        assert_eq!(operation.logical_geometry.rows, 128);
        assert_eq!(operation.slices.len(), 1);
        assert_eq!(
            operation.slices[0].source.weight.codes.offset,
            u64::from(rank * 2 + 1) * 128 * 128 * 2
        );
        assert_eq!(operation.slices[0].rows, vec![0..128]);
        assert_eq!(operation.slices[0].source.target_rows, 0..128);
    }
}

fn prepared_assembly_fixture() -> (
    Declaration,
    BoundDeclaration,
    Geometry,
    ImportedProjections,
    Registry,
) {
    use kernel_core::precision::{
        CodeEncoding, GroupAxis, ScaleConvention, ScaleScope, ScaleSpec, TensorSpan,
    };
    let (mut d, mut b, g, mut i) = fixture(
        SplitClass::Column {
            tile: Dim::Literal(64),
        },
        256,
        128,
        ProjectionKind::Dense {},
    );
    let mut raw = b.operands[0].imported.weight.clone();
    raw.format = kernel_core::contract::WeightFormat::Fp8Block128x128(
        kernel_core::contract::ScaleMode::ExactF32,
    );
    raw.encoding = CodeEncoding::E4M3;
    raw.codes.dtype = DType::Fp8E4M3;
    raw.codes.byte_length /= 2;
    raw.local_scale = Some(ScaleSpec {
        tensor: TensorSpan {
            tensor: "matrix.scale".into(),
            dtype: DType::F32,
            shape: vec![2, 1],
            offset: 0,
            byte_length: 8,
        },
        scope: ScaleScope::Block,
        convention: ScaleConvention::Multiply,
        groups: vec![
            GroupAxis {
                axis: 0,
                size: 128,
                origin: 0,
            },
            GroupAxis {
                axis: 1,
                size: 128,
                origin: 0,
            },
        ],
    });
    b.operands[0].imported.weight = raw.clone();
    b.operands[0].format = FormatId::Fp8Block128;
    for imported in i.values_mut() {
        imported.format = FormatId::Fp8Block128;
        imported.family = kernel_core::precision::format::KernelFamily::Row;
        imported.weights = vec![raw.clone()];
        imported.matrices[0].weight = raw.clone();
    }
    let port = projection_spec::contracts::Port {
        id: "weight".into(),
        binding: "weight".into(),
        dtype: DType::Fp8E4M3,
        interpretation: None,
        shape: vec![Dim::Literal(256), Dim::Literal(128)],
        state: false,
        format: Some(FormatId::Fp8Block128),
    };
    d.projections[0].source_transforms = vec![projection_spec::Variant {
        id: "fixture/source.v1".into(),
        parameters: serde_json::json!({}),
        numerical_contract: "fixture.source.v1".into(),
        inputs: vec![port.clone()],
        outputs: vec![projection_spec::contracts::Port {
            format: Some(FormatId::Fp8Rows64),
            ..port.clone()
        }],
    }];
    let mut registry = Registry::default();
    registry.register(&HOOKS).unwrap();
    let mut destination = d.projections[0].clone();
    destination.id = "prepared-window".into();
    destination.source = None;
    destination.parts.clear();
    destination.source_transforms.clear();
    destination.composition = None;
    destination.inputs.clear();
    destination.shape[0] = Dim::Literal(64);
    destination.split = SplitClass::Replicate {};
    destination.assembly=vec![serde_json::from_value(serde_json::json!({"projection":"matrix","layer":null,
        "operand":"weight","rows":{"start":64,"length":64},"coordinates":"rank_local","inherit_split":true})).unwrap()];
    let mut prototype = i.values().next().unwrap().clone();
    prototype.operation_id = "prepared-window.{output}".into();
    i.insert(
        InstanceKey {
            projection: destination.id.clone(),
            layer: None,
        },
        prototype,
    );
    b.assemblies.push(projection_spec::bind::BoundAssembly {
        projection: destination.id.clone(),
        layer: None,
        members: destination.assembly.clone(),
        tensors: vec![vec!["matrix.weight".into()]],
    });
    d.projections.push(destination);
    (d, b, g, i, registry)
}

struct PreparedMember;
impl derive::hooks::VariantDeriver for PreparedMember {
    fn transform(
        &self,
        v: &projection_spec::Variant,
        part: Option<&projection_spec::Part>,
        ctx: &derive::ProjectionContext<'_>,
        imported: ImportedProjection,
    ) -> Result<derive::SourceTransformResult, projection_spec::Error> {
        PreparedRows {
            corrupt_source: false,
            wrong_rank: false,
        }
        .transform(v, part, ctx, imported)
    }
    fn stage(
        &self,
        s: &projection_spec::contracts::Stage,
        _: &derive::ProjectionContext<'_>,
    ) -> Result<derive::DerivedStage, projection_spec::Error> {
        let mut result = hook_stage(&s.id);
        let view = |p: &projection_spec::contracts::Port, role| derive::RequiredView {
            id: p.binding.clone(),
            role,
            dtype: p.dtype,
            interpretation: p.interpretation,
            shape: p
                .shape
                .iter()
                .map(|d| d.resolve(&geometry()).unwrap())
                .collect(),
            bytes: 2,
            layout: "row-major".into(),
        };
        result.inputs = s
            .contract
            .inputs
            .iter()
            .map(|p| view(p, ViewRole::Input))
            .collect();
        result.outputs = s
            .contract
            .outputs
            .iter()
            .map(|p| view(p, ViewRole::Output))
            .collect();
        Ok(result)
    }
}

#[test]
fn assembly_applies_source_preparation_and_stages_before_trimming_rank_rows() {
    let (mut d, b, g, imports, registry) = prepared_assembly_fixture();
    d.projections[0].composition = Some(
        serde_json::from_value(serde_json::json!({"pre":[{"id":"source-stage",
        "contract":{"id":"fixture/stage.v1","parameters":{},"numerical_contract":"fixture.stage.v1",
        "inputs":[{"id":"in","binding":"in","dtype":"Bf16","shape":[1],"state":false}],
        "outputs":[{"id":"out","binding":"out","dtype":"Bf16","shape":[1],"state":false}]},
        "phases":["Decode","Prefill","Verify"],"mutates":[]}],"post":[]}))
        .unwrap(),
    );
    // A source may follow its consumer in declaration order.
    d.projections.reverse();
    for rank in 0..2 {
        let result = derive::derive(DerivationInput {
            declaration: &d,
            bound: &b,
            geometry: &g,
            registry: &registry,
            topology: Topology { ranks: 2, rank },
            imported: &imports,
            variants: &PreparedMember,
        })
        .unwrap();
        let destination = result
            .operations
            .iter()
            .find(|o| o.instance.projection == "prepared-window")
            .unwrap();
        assert!(
            destination.source_preparation.is_some(),
            "source preparation was lost"
        );
        assert_eq!(
            destination.operation.weights[0].format,
            kernel_core::contract::WeightFormat::Fp8Rows64x128(
                kernel_core::contract::ScaleMode::ExactF32
            )
        );
        assert_eq!(
            destination.logical_geometry,
            MatrixGeometry { rows: 64, k: 128 }
        );
        assert_eq!(
            destination.operation.geometry,
            Some(MatrixGeometry { rows: 64, k: 128 })
        );
        assert_eq!(
            destination
                .stages
                .iter()
                .map(|s| s.id.as_str())
                .collect::<Vec<_>>(),
            vec!["source-stage"]
        );
        let prepared = destination.source_preparation.as_ref().unwrap();
        assert_eq!(prepared.records.len(), 2);
        for record in &prepared.records {
            assert!(destination.views.iter().any(|v| v.id == record.view));
            assert_eq!(record.view_offset, 0);
            let local = record.section == kernel_core::weight_transform::PackedSection::LocalScales;
            assert_eq!(
                record.record.stored_shape,
                if local { vec![1, 1] } else { vec![64, 128] }
            );
            let start = if local {
                u64::from(rank)
            } else {
                u64::from(rank) * 128 + 64
            };
            assert_eq!(
                record.record.ranges,
                vec![start..start + if local { 1 } else { 64 }]
            );
            let payload = if local {
                [1.25f32, 2.5f32]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>()
            } else {
                (0..256 * 128).map(|n| (n % 113) as u8).collect()
            };
            let actual = record
                .record
                .clone()
                .prepare(record.record.output_bytes().unwrap())
                .unwrap()
                .produce_whole(&mut |_, offset, out| {
                    out.copy_from_slice(&payload[offset as usize..offset as usize + out.len()]);
                    Ok(())
                })
                .unwrap();
            let width = if local { 4 } else { 128 };
            assert_eq!(
                actual,
                payload[start as usize * width
                    ..start as usize * width + if local { 4 } else { 64 * 128 }]
            );
        }
    }
}

#[test]
fn assembled_source_bypass_keeps_raw_format_and_geometry_at_other_topologies() {
    let (mut d, b, g, imports, registry) = prepared_assembly_fixture();
    d.projections[1].assembly[0].rows.start = Dim::Literal(0);
    let mut bound = b;
    bound.assemblies[0].members = d.projections[1].assembly.clone();
    for rank in 0..4 {
        let result = derive::derive(DerivationInput {
            declaration: &d,
            bound: &bound,
            geometry: &g,
            registry: &registry,
            topology: Topology { ranks: 4, rank },
            imported: &imports,
            variants: &PreparedMember,
        })
        .unwrap();
        let destination = result
            .operations
            .iter()
            .find(|o| o.instance.projection == "prepared-window")
            .unwrap();
        assert!(destination.source_preparation.is_none());
        assert_eq!(
            destination.operation.weights[0].format,
            kernel_core::contract::WeightFormat::Fp8Block128x128(
                kernel_core::contract::ScaleMode::ExactF32
            )
        );
        assert_eq!(destination.logical_geometry.rows, 64);
        assert_eq!(
            destination.slices[0].rows,
            vec![u64::from(rank) * 64..u64::from(rank + 1) * 64]
        );
    }
}

struct PreparedImplementation;
impl derive::hooks::VariantDeriver for PreparedImplementation {
    fn transform(
        &self,
        v: &projection_spec::Variant,
        part: Option<&projection_spec::Part>,
        ctx: &derive::ProjectionContext<'_>,
        imported: ImportedProjection,
    ) -> Result<derive::SourceTransformResult, projection_spec::Error> {
        PreparedMember.transform(v, part, ctx, imported)
    }
    fn implementation(
        &self,
        v: &projection_spec::Variant,
        ctx: &derive::ProjectionContext<'_>,
        imported: &ImportedProjection,
    ) -> Result<derive::DerivedOperation, projection_spec::Error> {
        let mut operation = ctx.ordinary(imported)?;
        operation.operation.arithmetic.numerical_contract = v.numerical_contract.clone();
        operation.stages.push(hook_stage("source-implementation"));
        Ok(operation)
    }
}
#[test]
fn assembly_retains_source_implementation_stages() {
    let (mut d, b, g, mut imports, registry) = prepared_assembly_fixture();
    let port = projection_spec::contracts::Port {
        id: "weight".into(),
        binding: "weight".into(),
        dtype: DType::Fp8E4M3,
        interpretation: None,
        shape: vec![Dim::Literal(128), Dim::Literal(128)],
        state: false,
        format: Some(FormatId::Fp8Rows64),
    };
    d.projections[0].composition = Some(projection_spec::contracts::Composition {
        implementation: Some(projection_spec::Variant {
            id: "fixture/implementation.v1".into(),
            parameters: serde_json::json!({}),
            numerical_contract: "fixture.implementation.v1".into(),
            inputs: vec![port.clone()],
            outputs: vec![port],
        }),
        ..Default::default()
    });
    d.projections[1].numerical_contract = "fixture.implementation.v1".into();
    imports
        .get_mut(&InstanceKey {
            projection: "prepared-window".into(),
            layer: None,
        })
        .unwrap()
        .arithmetic
        .numerical_contract = "fixture.implementation.v1".into();
    let result = derive::derive(DerivationInput {
        declaration: &d,
        bound: &b,
        geometry: &g,
        registry: &registry,
        topology: Topology { ranks: 2, rank: 1 },
        imported: &imports,
        variants: &PreparedImplementation,
    })
    .unwrap();
    let destination = result
        .operations
        .iter()
        .find(|o| o.instance.projection == "prepared-window")
        .unwrap();
    assert_eq!(
        destination
            .stages
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>(),
        vec!["source-implementation"]
    );
    assert!(destination.source_preparation.is_some());
}

#[test]
fn plain_assembly_members_materialize_without_a_source_execution_bucket() {
    let (mut d, b, g, imports) = paired_fixture(8, 16);
    d.fusion_groups.clear();
    d.projections[0]
        .uses
        .retain(|u| u.phase == kernel_core::precision::ExecutionPhase::Verify);
    let result = run(&d, &b, &g, &imports, Topology { ranks: 1, rank: 0 }).unwrap();
    assert_eq!(
        result
            .operations
            .iter()
            .filter(|o| o.instance.projection == "other")
            .count(),
        4
    );
}

#[test]
fn public_operation_id_uses_output_and_refuses_unresolved_layers() {
    let (d, b, g, _) = fixture(SplitClass::Replicate {}, 64, 128, ProjectionKind::Dense {});
    let mut context = derive::ProjectionContext {
        declaration: &d,
        bound: &b,
        geometry: &g,
        projection: &d.projections[0],
        topology: Topology { ranks: 1, rank: 0 },
        layer: Some(7),
        source_layer: Some(7),
        source_instance: InstanceKey {
            projection: "matrix".into(),
            layer: Some(7),
        },
        selections: None,
        usage: &d.projections[0].uses[0],
    };
    assert_eq!(
        context
            .operation_id("{projection}.{i}.{output}", DType::F32)
            .unwrap(),
        "matrix.7.f32"
    );
    assert_eq!(
        context
            .operation_id("{projection}.{i}.{output}", DType::Bf16)
            .unwrap(),
        "matrix.7.bf16"
    );
    context.layer = None;
    assert!(
        context
            .operation_id("{projection}.{i}.{output}", DType::Bf16)
            .is_err()
    );
    assert!(
        context
            .operation_id("{projection}.{unknown}", DType::Bf16)
            .is_err()
    );
}

#[test]
fn projection_kinds_name_their_family_and_rank_operation_kind() {
    use kernel_core::precision::format::KernelFamily;
    let d = declaration();
    let g = geometry();
    let topology = Topology { ranks: 2, rank: 0 };
    let kind = |id: &str| {
        let p = d.projections.iter().find(|p| p.id == id).unwrap();
        let selection = derive::rank_selection(p, &g, topology, 0).unwrap();
        (p.family(), p.operation_kind(&g, &selection).unwrap())
    };
    assert_eq!(kind("q"), (KernelFamily::Dense, OperationKind::Dense));
    assert_eq!(
        kind("experts"),
        (
            KernelFamily::GroupedExpert,
            // The fixture slices each expert's rows, so a rank keeps all four.
            OperationKind::Grouped {
                experts: 4,
                top_k: 2
            }
        )
    );
    assert_eq!(kind("table"), (KernelFamily::Lookup, OperationKind::Lookup));
}
