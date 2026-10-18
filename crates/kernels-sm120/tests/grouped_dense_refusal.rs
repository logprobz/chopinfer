//! A derived grouped dense operation reaches no dense kernel: its spec, fed
//! straight to the generated provider or to capability binding, is refused,
//! while the same spec stated as one dense matrix would have bound.
use kernel_core::contract::{DType, DenseKernel, Epilogues, GemmQuery, Target, WeightFormat};
use kernel_core::precision::execution::ImplementationProvider;
use kernel_core::precision::format::{A16Recipe, FormatId, KernelFamily, registry};
use kernel_core::precision::{
    ActivationSpec, ArithmeticSpec, CodeEncoding, ExecutionPhase, OperationKind, OperationUse,
    Rounding, TensorSpan, WeightLayout, WeightSpec,
};
use kernels_sm120::gemm;
use kernels_sm120::precision::{CompiledInventory, KernelOffer, generated};
use kernels_sm120::row::precision as row;
use projection_spec::bind::{
    BindingRequest, CheckpointMetadata, CompanionRules, ImportedOperand, TensorInfo,
};
use projection_spec::derive::{
    self, DerivationInput, DerivedOperation, ImportedProjection, ImportedProjections, InstanceKey,
    Topology, hooks::NoVariants,
};
use projection_spec::registry::Registry;

/// Four BF16 groups of [64, 128], stacked as one [256, 128] tensor.
const DECLARATION: &str = r#"{
  "schema": "chopinfer.projection.v1",
  "namespace": "fixture",
  "layer_stems": [],
  "projections": [
    {"id": "grouped", "source": {"tensor": "grouped.weight"},
     "shape": [64, 128], "kind": {"kind": "grouped_dense", "groups": 4},
     "split": {"class": "expert_owned", "tile": 1},
     "epilogues": {"residual_add": false, "bias": false, "gelu": null, "swiglu": null, "output_quantization": null, "weighted_moe_combine": false},
     "numerical_contract": "fixture.numeric.v1", "format": "bf16",
     "uses": [{"phase": "Decode", "capacity": 1, "graph": false}]}
  ],
  "fusion_groups": [],
  "side_tensors": [],
  "external_tensors": []
}"#;

fn weight() -> WeightSpec {
    WeightSpec {
        codes: TensorSpan {
            tensor: "grouped.weight".into(),
            dtype: DType::Bf16,
            shape: vec![256, 128],
            offset: 0,
            byte_length: 256 * 128 * 2,
        },
        logical_shape: vec![256, 128],
        format: WeightFormat::Bf16,
        encoding: CodeEncoding::Bf16LittleEndian,
        layout: WeightLayout::row_major(),
        local_scale: None,
        global_scale: None,
        transforms: vec![],
    }
}

struct Rules;
impl CompanionRules for Rules {
    fn import(
        &self,
        _: &BindingRequest<'_>,
        _: &CheckpointMetadata,
    ) -> Result<ImportedOperand, projection_spec::Error> {
        Ok(ImportedOperand {
            weight: weight(),
            parts: vec![],
            companions: vec![],
            calibrations: vec![],
        })
    }
}

fn usage() -> OperationUse {
    OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: 1,
        graph: false,
    }
}

/// The rank's derived operation over all four groups at TP1.
fn derived() -> DerivedOperation {
    let g = Default::default();
    let registry = Registry::default();
    let declaration = projection_spec::parse(DECLARATION, &g, &registry).unwrap();
    let metadata = CheckpointMetadata {
        tensors: [(
            "grouped.weight".to_owned(),
            TensorInfo {
                dtype: "BF16".into(),
                shape: vec![256, 128],
                file: None,
            },
        )]
        .into(),
        ..Default::default()
    };
    let bound = declaration.bind(&g, &registry, &metadata, &Rules).unwrap();
    let plugin = registry::by_id(FormatId::Bf16).unwrap();
    let imported = ImportedProjections::from([(
        InstanceKey {
            projection: "grouped".into(),
            layer: None,
        },
        ImportedProjection {
            operation_id: "grouped".into(),
            target: Target::Sm120f,
            family: KernelFamily::Dense,
            format: FormatId::Bf16,
            geometry: derive::GeometryMode::RankLocal,
            weights: vec![weight()],
            activations: vec![ActivationSpec::bf16()],
            arithmetic: ArithmeticSpec {
                weight_operand: DType::Bf16,
                accumulator: DType::F32,
                output: DType::Bf16,
                decode_order: plugin
                    .decode_order(
                        KernelFamily::Dense,
                        kernel_core::contract::ActivationFormat::Bf16,
                        A16Recipe::ScaledWeight,
                    )
                    .unwrap(),
                rounding: Rounding::NearestEven,
                numerical_contract: "fixture.numeric.v1".into(),
            },
            a16_recipe: A16Recipe::ScaledWeight,
            matrices: derive::grouped_dense_matrices(&bound.operands[0], 4, 64, 128).unwrap(),
            preparation: None,
        },
    )]);
    let derivation = derive::derive(DerivationInput {
        declaration: &declaration,
        bound: &bound,
        geometry: &g,
        registry: &registry,
        topology: Topology { ranks: 1, rank: 0 },
        imported: &imported,
        variants: &NoVariants,
    })
    .unwrap();
    derivation.operations[0].clone()
}

#[test]
fn a_derived_grouped_dense_spec_reaches_no_dense_kernel() {
    let derived = derived();
    let spec = &derived.operation;
    assert_eq!(spec.kind, OperationKind::GroupedDense { groups: 4 });
    spec.validate().unwrap();

    // The generated provider refuses the field itself.
    let error = generated::offer(
        generated::Request {
            operation: spec,
            format: FormatId::Bf16,
            usage: usage(),
            implementation: generated::Implementation::Dense {
                kernel: DenseKernel::FragmentDecode { splits: 1 },
                capacity: 1,
                knobs: None,
            },
        },
        &[],
    )
    .unwrap_err();
    assert!(error.contains("grouped fold GEMM"), "{error}");

    // A dense BF16 kernel of one group's geometry would bind the spec stated
    // as one dense matrix; the grouped spec it refuses by kind.
    let geometry = spec.matrix_geometry().unwrap();
    let offer = KernelOffer::dense(
        gemm::DenseRequest {
            query: GemmQuery {
                m_capacity: 1,
                n: geometry.rows as u32,
                k: geometry.k as u32,
                weight: WeightFormat::Bf16,
                activation: kernel_core::contract::ActivationFormat::Bf16,
                output: DType::Bf16,
                epilogues: Epilogues::default(),
                grouped: None,
            },
            target: Target::Sm120f,
            symbol: "fixture_dense_v1".into(),
            kernel: DenseKernel::FragmentDecode { splits: 1 },
            knobs: None,
        },
        usage(),
        row::Nvfp4Global::Absent,
        "fixture_quant_v1",
    )
    .unwrap();
    let available = offer
        .handle()
        .entries()
        .unwrap()
        .into_iter()
        .map(|e| e.symbol)
        .collect();
    let inventory = CompiledInventory::new(vec![offer], available, "a".repeat(64)).unwrap();
    let mut placeholder = spec.clone();
    placeholder.kind = OperationKind::Dense;
    inventory.bind(&placeholder, usage()).unwrap();
    let error = inventory.bind(spec, usage()).err().unwrap();
    assert!(error.contains("GroupedDense"), "{error}");
}
