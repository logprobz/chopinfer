//! Model-owned projection boundaries and fusion peers over shared recipes.
use super::*;
use kernel_core::precision::OperationUse;
use kernel_core::precision::capability::CapabilityInventory;
use kernel_core::precision::execution::{Binding, ImplementationProvider};
use model_compiler_sdk::quantization::{ActivationPolicy, PreparedOperation, resolve_operation};

pub struct DenseOperation {
    pub source: OperationSpec,
    pub role: DenseRole,
    pub shape: Shape,
    /// All inputs of the existing multi-output fusion retain BF16 weights.
    /// Otherwise even its BF16 peer projections use the shared composition.
    pub legacy: bool,
}

#[cfg(test)]
pub fn dense_roles(layer: u32) -> Result<Vec<DenseRole>> {
    Ok(DenseRole::roles_for(Shape::TP4, layer)?)
}

#[cfg(test)]
fn stems(layer: u32, role: DenseRole) -> Vec<String> {
    role.operand_names(Shape::TP4, layer)
        .unwrap()
        .iter()
        .map(|name| name.trim_end_matches(".weight").into())
        .collect()
}

pub fn dense(
    source: &SourcePrecision,
    layer: u32,
    role: DenseRole,
    shape: Shape,
) -> Result<DenseOperation> {
    if !DenseRole::roles_for(shape, layer)?.contains(&role) {
        return Err(
            format!("layer {layer}: {role:?} is not a Qwen operation at this layer").into(),
        );
    }
    let definitions = super::declaration::Definitions::new(shape)?;
    let declared = definitions.projection(super::declaration::Definitions::dense_id(
        role, shape, layer,
    ))?;
    let names = definitions.names(declared, Some(layer))?;
    let operands = names
        .iter()
        .map(|name| {
            source
                .operands
                .get(name)
                .cloned()
                .ok_or_else(|| format!("missing dense operand {name}").into())
        })
        .collect::<Result<Vec<_>>>()?;
    let legacy = definitions.legacy_dense(source, declared, layer)?;
    if !legacy
        && model_qwen38_contract::projections::support::Support::sources(
            shape,
            source
                .operands
                .iter()
                .map(|(name, o)| (name, o.weight.format)),
        )?
        .admit_mixed(
            &declared.id,
            kernel_core::precision::ExecutionPhase::Decode,
            kernel_core::precision::format::registry::by_weight(&operands[0].weight)?.id(),
        )
        .is_err()
    {
        return Err(UnsuppliedTopology(format!(
            "operation {}: the declaration supplies no mixed dense composition at TP{}",
            names[0],
            shape.ranks(),
        ))
        .into());
    }
    let id = names[0]
        .trim_end_matches(".weight")
        .replace(".gate_proj", ".gate_up_proj");
    let operation = definitions.operation(declared, id, &operands, legacy)?;
    Ok(DenseOperation {
        source: operation,
        role,
        shape,
        legacy,
    })
}

struct DenseProvider<'a> {
    inventory: &'a CapabilityInventory,
    implementation: String,
}

impl ImplementationProvider for DenseProvider<'_> {
    type Handle = usize;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> std::result::Result<Binding<usize>, String> {
        self.inventory.bind_matching(operation, usage, |c| {
            c.implementation == self.implementation
        })
    }
}

impl DenseOperation {
    pub fn resolve(
        &self,
        policy: ActivationPolicy,
        inventory: &CapabilityInventory,
    ) -> Result<PreparedOperation<usize>> {
        let implementation = if self.legacy {
            self.role.legacy_implementation(self.shape)
        } else {
            self.role
                .mixed_implementation_for(self.shape, self.source.weights[0].format)
        };
        Ok(resolve_operation(
            &self.source,
            policy,
            &DenseProvider {
                inventory,
                implementation,
            },
        )?)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use kernel_core::precision::capability::CapabilityDocument;
    use kernel_core::precision::{ActivationSpec, TensorSpan};
    use model_compiler_sdk::quantization::import::{ExportMetadata, OperandRole, import_operand};
    use model_compiler_sdk::quantization::source::TensorInventory;
    use std::collections::BTreeMap;

    pub(in crate::projection_program) fn inventory() -> CapabilityInventory {
        let mut operations = model_qwen38_contract::sm120::generated::all_capabilities().unwrap();
        operations.extend(precision::legacy_capabilities().unwrap());
        let document = CapabilityDocument {
            version: 1,
            operations,
        };
        let entries = document.entries();
        CapabilityInventory::new(document, entries, "0".repeat(64)).unwrap()
    }

    /// Layer 0's linear-attention input fusion: BF16 peers, or MXFP8
    /// `linear.a` when `mixed`.
    pub(in crate::projection_program) fn fixture(mixed: bool) -> SourcePrecision {
        let metadata = if mixed {
            let mut metadata = ExportMetadata::default();
            metadata.add_document("quantization", br#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"model.language_model.layers.0.linear_attn.in_proj_a":{"quant_algo":"MXFP8","group_size":32}}}"#).unwrap();
            metadata
        } else {
            ExportMetadata::default()
        };
        let mut tensors = BTreeMap::new();
        let mut operands = BTreeMap::new();
        for (role, rows) in [
            (DenseRole::LinearQkv, 10240),
            (DenseRole::LinearZ, 6144),
            (DenseRole::LinearA, 48),
            (DenseRole::LinearB, 48),
        ] {
            let stem = &stems(0, role)[0];
            let name = format!("{stem}.weight");
            let mx = mixed && role == DenseRole::LinearA;
            tensors.insert(
                name.clone(),
                TensorSpan {
                    tensor: name.clone(),
                    dtype: if mx { DType::Fp8E4M3 } else { DType::Bf16 },
                    shape: vec![rows, 2560],
                    offset: 0,
                    byte_length: rows * 2560 * if mx { 1 } else { 2 },
                },
            );
            if mx {
                let scale = format!("{stem}.weight_scale");
                tensors.insert(
                    scale.clone(),
                    TensorSpan {
                        tensor: scale,
                        dtype: DType::U8,
                        shape: vec![rows, 80],
                        offset: 0,
                        byte_length: rows * 80,
                    },
                );
            }
            operands.insert(
                name.clone(),
                import_operand(
                    &metadata,
                    &OperandRole {
                        operation: stem.clone(),
                        weight: name,
                        targets: vec![stem.clone()],
                        logical_shape: vec![rows, 2560],
                        lookup: false,
                        shared_scale_stem: None,
                        declared: None,
                    },
                    &tensors,
                )
                .unwrap(),
            );
        }
        SourcePrecision {
            inventory: TensorInventory {
                tensors,
                spans: BTreeMap::new(),
            },
            metadata,
            operands,
        }
    }

    #[test]
    fn mxfp4_bf16_dense_recipe_uses_exact_scaled_operands() {
        let mut source = fixture(true);
        let name = format!("{}.weight", stems(0, DenseRole::LinearA)[0]);
        let operand = source.operands.get_mut(&name).unwrap();
        operand.weight.format = WeightFormat::Mxfp4;
        operand.weight.encoding = kernel_core::precision::CodeEncoding::E2M1(
            kernel_core::precision::NibbleOrder::EvenLow,
        );
        operand.weight.codes.dtype = DType::U8;
        *operand.weight.codes.shape.last_mut().unwrap() /= 2;
        operand.weight.codes.byte_length /= 2;
        operand.activation = Some(ActivationSpec::bf16());
        let op = dense(&source, 0, DenseRole::LinearA, Shape::TP4).unwrap();
        assert_eq!(
            op.source.arithmetic.decode_order,
            DecodeOrder::ExactScaledOperand
        );
    }

    /// The declaration leaves no topology out for any format of one
    /// quantized peer, with and without a quantized operand outside the
    /// fusion: a composition the kernels do not supply is refused when the
    /// operation resolves, not when it is built.
    #[test]
    fn the_declaration_leaves_no_fusion_topology_out() {
        use kernel_core::contract::ScaleMode;
        let peer = format!("{}.weight", stems(0, DenseRole::LinearA)[0]);
        // A quantized operand of another projection (the shared expert).
        let elsewhere = "model.language_model.layers.0.mlp.shared_expert.down_proj.weight";
        for format in [
            WeightFormat::Bf16,
            WeightFormat::Mxfp8,
            WeightFormat::Nvfp4,
            WeightFormat::Mxfp4,
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            WeightFormat::Fp8PerTensor,
        ] {
            for other in [None, Some(WeightFormat::Nvfp4), Some(WeightFormat::Mxfp4)] {
                let mut source = fixture(false);
                source.operands.get_mut(&peer).unwrap().weight.format = format;
                if let Some(other) = other {
                    let mut operand = source.operands[&peer].clone();
                    operand.weight.format = other;
                    source.operands.insert(elsewhere.into(), operand);
                }
                for shape in [Shape::TP2, Shape::TP4] {
                    for role in [DenseRole::LinearQkv, DenseRole::LinearB] {
                        let unsupplied = dense(&source, 0, role, shape)
                            .err()
                            .is_some_and(|e| e.downcast_ref::<UnsuppliedTopology>().is_some());
                        assert!(
                            !unsupplied,
                            "{role:?} with a {format:?} peer and {other:?} elsewhere at TP{}",
                            shape.ranks()
                        );
                    }
                }
            }
        }
    }

    /// The pre-port operation fetched every fusion peer before it judged
    /// the fusion, so a missing peer fails as itself even after a quantized
    /// peer and where the mixed fusion has no composition.
    #[test]
    fn a_missing_fusion_peer_fails_before_the_topology() {
        let mut source = fixture(true);
        let missing = format!("{}.weight", stems(0, DenseRole::LinearB)[0]);
        source.operands.remove(&missing);
        for shape in [Shape::TP2, Shape::TP4] {
            let Err(error) = dense(&source, 0, DenseRole::LinearQkv, shape) else {
                panic!("TP{}: a missing peer must fail", shape.ranks());
            };
            assert!(
                error.downcast_ref::<UnsuppliedTopology>().is_none(),
                "TP{}: {error}",
                shape.ranks()
            );
            assert_eq!(
                error.to_string(),
                format!("missing dense operand {missing}")
            );
        }
    }

    #[test]
    fn one_quantized_peer_uses_explicit_compositions_for_the_whole_fusion() {
        let inventory = inventory();
        let legacy = fixture(false);
        let mixed = fixture(true);
        for role in [
            DenseRole::LinearQkv,
            DenseRole::LinearZ,
            DenseRole::LinearA,
            DenseRole::LinearB,
        ] {
            let old = dense(&legacy, 0, role, Shape::TP4).unwrap();
            let old = old.resolve(ActivationPolicy::Auto, &inventory).unwrap();
            assert!(
                old.selection()
                    .bindings
                    .iter()
                    .any(|b| b.implementation.stages[0].contains("lin_in_c1"))
            );
            for shape in [Shape::TP2, Shape::TP4] {
                assert!(dense(&legacy, 0, role, shape).unwrap().legacy);
                let op = dense(&mixed, 0, role, shape).unwrap();
                assert!(!op.legacy);
                let auto = op.resolve(ActivationPolicy::Auto, &inventory).unwrap();
                let bf16 = op.resolve(ActivationPolicy::Bf16, &inventory).unwrap();
                assert_eq!(auto.selection().effective, bf16.selection().effective);
                assert!(
                    auto.selection()
                        .effective
                        .activations
                        .iter()
                        .all(ActivationSpec::is_bf16)
                );
                // TP4 serves its adopted symbols; TP2 adopts those of the
                // same executable identity and generates the rest.
                let published = model_qwen38_contract::sm120::generated::inventory(shape)
                    .unwrap()
                    .document
                    .operations;
                for binding in &auto.selection().bindings {
                    let stages = &binding.implementation.stages;
                    assert!(
                        shape.ranks() == 2 || stages[0].starts_with("chop_qwen38_precision_"),
                        "{stages:?}"
                    );
                    assert!(published.iter().any(|c| &c.stages == stages), "{stages:?}");
                }
                assert_eq!(
                    op.resolve(ActivationPolicy::Checkpoint, &inventory).is_ok(),
                    role != DenseRole::LinearA
                );
                if matches!(role, DenseRole::LinearA | DenseRole::LinearB) {
                    let id = role.projection_id(shape, Some(0));
                    let rows = if shape.ranks() == 4 {
                        if role == DenseRole::LinearA { 16 } else { 64 }
                    } else {
                        model_qwen38_contract::projections::ports::get(shape)
                            .unwrap()
                            .weight_geometry(id, op.source.weights[0].format)
                            .unwrap()
                            .rows
                    };
                    assert_eq!(
                        auto.selection().effective.matrix_geometry().unwrap().rows,
                        rows,
                        "{role:?} at TP{}",
                        shape.ranks()
                    );
                    // The copy of the rank's value heads into `ab`: a
                    // shape-free entry that takes its columns and strides at
                    // launch.
                    assert!(auto.selection().bindings.iter().all(|b| {
                        b.implementation.stages[1].starts_with("chop_proj_bf16_copy_")
                    }));
                }
            }
        }
        assert!(dense(&mixed, 0, DenseRole::AttentionQ, Shape::TP4).is_err());
    }

    #[test]
    fn real_checkpoint_dense_operations_keep_source_recipes_and_bind_every_use() {
        let Ok(root) = std::env::var("CHOPINFER_QWEN38_CHECKPOINT") else {
            eprintln!("UNRUN: set CHOPINFER_QWEN38_CHECKPOINT for real dense recipe checks");
            return;
        };
        let catalog = crate::catalog::Catalog::load(std::path::Path::new(&root)).unwrap();
        let source = catalog.precision.as_ref().unwrap();
        let inventory = inventory();
        let mut operations = 0;
        for layer in 0..LAYERS {
            for role in dense_roles(layer).unwrap() {
                let op = dense(source, layer, role, Shape::TP4).unwrap();
                let auto = op.resolve(ActivationPolicy::Auto, &inventory).unwrap();
                let bf16 = op.resolve(ActivationPolicy::Bf16, &inventory).unwrap();
                assert_eq!(auto.selection().source, op.source);
                assert_eq!(auto.selection().effective, bf16.selection().effective);
                assert_eq!(
                    auto.selection().bindings.len(),
                    precision::uses().unwrap().len()
                );
                assert_eq!(
                    op.resolve(ActivationPolicy::Checkpoint, &inventory).is_ok(),
                    op.source.activations.iter().all(ActivationSpec::is_bf16)
                );
                operations += 1;
            }
        }
        assert_eq!(operations, 48 * 7);
    }
}
