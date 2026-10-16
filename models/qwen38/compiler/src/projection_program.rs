//! Family-owned operation boundaries over shared imported operand contracts.
use crate::{Result, precision_source::SourcePrecision};
#[cfg(test)]
use kernel_core::contract::SwigluColumns;
use kernel_core::contract::{ActivationFormat, DType, WeightFormat};
use kernel_core::precision::{ArithmeticSpec, DecodeOrder, OperationKind, OperationSpec, Rounding};
use model_compiler_sdk::quantization::bank::{ImportedBank, concatenate_experts};
use model_qwen38_contract::precision::{DenseRole, Projection};
use model_qwen38_contract::{contract::*, precision};

#[cfg(test)]
pub(crate) mod baseline;
mod declaration;
mod dense;
pub use dense::{DenseOperation, dense};
mod lookup;
pub use lookup::{LookupOperation, lookup};
mod text;
pub use text::{SourceProgram, drafter_refusal};
/// MXFP4 expert offers of the SM120 kernels, for the binding tests.
#[cfg(test)]
mod sm120;
#[cfg(test)]
pub(crate) use text::{
    drafter_tests::drafter_records, routed_nvfp4_fixture, routed_nvfp4_fixture_without,
};

/// A topology whose operations the checkpoint's precision supplies no
/// composition for (the kernels compose no such operation at its rank
/// count), or that supplied
/// precision components do not record: a package leaves it out.
#[derive(Debug)]
pub struct UnsuppliedTopology(pub String);

impl std::fmt::Display for UnsuppliedTopology {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for UnsuppliedTopology {}

/// Assemblies retain original tensor spans for streaming packing and source
/// identity. The operation itself has compact bank and calibration references.
pub struct RoutedOperation {
    pub source: OperationSpec,
    pub banks: Vec<ImportedBank>,
}

pub fn routed(
    source: &SourcePrecision,
    layer: u32,
    projection: Projection,
    shape: Shape,
) -> Result<RoutedOperation> {
    // A drafter layer executed after the main layers has its own declared
    // banks, published as the main ones.
    let definitions = declaration::Definitions::new(shape)?;
    let declared = definitions
        .ports
        .executed_at(
            layer,
            match projection {
                Projection::GateUp => "main.experts.gate-up",
                Projection::Down => "main.experts.down",
            },
        )
        .map_err(|_| "routed operation requires a main Qwen layer or the MTP layer")?;
    // Checkpoint members are per expert; the bank concatenates them.
    let member = declared
        .source
        .as_ref()
        .ok_or("routed projection has no source")?;
    let template = definitions.ports.source_name(
        member,
        Some(definitions.ports.source_layer(&declared.id, layer)?),
    )?;
    let mut banks = Vec::new();
    for name in declared
        .operands
        .iter()
        .map(|o| template.replace("{operand}", o))
    {
        let experts = (0..shape.experts())
            .map(|expert| {
                let name = name.replace("{expert}", &expert.to_string());
                source
                    .operands
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| format!("missing routed operand {name}").into())
            })
            .collect::<Result<Vec<_>>>()?;
        let bank_name = name
            .replace(".{expert}", "")
            .trim_end_matches(".weight")
            .to_owned();
        banks.push(concatenate_experts(&bank_name, &experts)?);
    }
    let id = banks[0]
        .operand
        .weight
        .codes
        .tensor
        .trim_end_matches(".weight")
        .replace(".gate_proj", ".gate_up_proj");
    let operands = banks.iter().map(|b| b.operand.clone()).collect::<Vec<_>>();
    let operation = definitions.operation(declared, id, &operands, true)?;
    Ok(RoutedOperation {
        source: operation,
        banks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::precision::capability::{CapabilityDocument, CapabilityInventory};
    use kernel_core::precision::{Calibration, CodeEncoding, NibbleOrder, ScaleScope, TensorSpan};
    use model_compiler_sdk::quantization::import::{ExportMetadata, OperandRole, import_operand};
    use model_compiler_sdk::quantization::source::TensorInventory;
    use model_compiler_sdk::quantization::{ActivationPolicy, resolve_operation};
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn routed_operations_use_the_selected_geometry_top_k() {
        static GEOMETRY: model_qwen38_contract::geometry::Geometry = {
            let mut geometry = model_qwen38_contract::geometry::FLASH;
            geometry.shape.moe.top_k = 8;
            geometry
        };
        let source = nvfp4_source();
        for ranks in [Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&GEOMETRY, ranks);
            for projection in [Projection::GateUp, Projection::Down] {
                let operation = routed(&source, 0, projection, shape).unwrap();
                assert_eq!(
                    operation.source.kind,
                    OperationKind::Grouped {
                        experts: shape.local_experts,
                        top_k: GEOMETRY.top_k(),
                    }
                );
            }
        }
    }

    fn check_routed(source: &SourcePrecision) {
        for projection in [Projection::GateUp, Projection::Down] {
            let op = routed(source, 0, projection, Shape::TP4).unwrap();
            let mut operations = Vec::new();
            for usage in precision::uses().unwrap() {
                operations.push(
                    model_qwen38_contract::sm120::generated::publish(
                        Shape::TP4,
                        if projection == Projection::GateUp {
                            "main.experts.gate-up"
                        } else {
                            "main.experts.down"
                        },
                        WeightFormat::Nvfp4,
                        usage,
                    )
                    .unwrap()
                    .capability
                    .clone(),
                );
                operations.push(
                    precision::legacy_expert_capability(Shape::TP4, projection, usage).unwrap(),
                );
            }
            let document = CapabilityDocument {
                version: 1,
                operations,
            };
            let available = document.entries();
            let provider = CapabilityInventory::new(document, available, "0".repeat(64)).unwrap();
            let auto = resolve_operation(&op.source, ActivationPolicy::Auto, &provider).unwrap();
            let bf16 = resolve_operation(&op.source, ActivationPolicy::Bf16, &provider).unwrap();
            assert_eq!(auto.selection().effective, bf16.selection().effective);
            assert_eq!(
                auto.selection().bindings.len(),
                precision::uses().unwrap().len()
            );
            assert!(auto.selection().reuse.is_some());
            assert!(!auto.selection().fallback_reasons.is_empty());
            assert!(
                resolve_operation(&op.source, ActivationPolicy::Checkpoint, &provider).is_err()
            );
            for bank in &op.banks {
                assert_eq!(bank.operand.weight.logical_shape[0], u64::from(EXPERTS));
                assert!(bank.operand.source_tensors.len() >= EXPERTS as usize * 2);
                if bank.operand.weight.format == WeightFormat::Nvfp4 {
                    assert_eq!(
                        bank.operand.weight.global_scale.as_ref().unwrap().scope,
                        ScaleScope::Expert
                    );
                    let Calibration::Static(input) = &bank
                        .operand
                        .activation
                        .as_ref()
                        .unwrap()
                        .quantizer
                        .as_ref()
                        .unwrap()
                        .calibration
                    else {
                        panic!("source calibration missing")
                    };
                    assert_eq!(input.scope, ScaleScope::Expert);
                    assert_eq!(input.tensor.shape, [u64::from(EXPERTS)]);
                    assert_eq!(
                        bank.assemblies
                            .iter()
                            .find(|a| a.tensor == input.tensor)
                            .unwrap()
                            .parts
                            .len(),
                        EXPERTS as usize
                    );
                }
            }
            let formats: BTreeSet<_> = auto
                .selection()
                .effective
                .activations
                .iter()
                .map(|a| format!("{:?}", a.format))
                .collect();
            assert_eq!(formats, BTreeSet::from(["Bf16".into()]));
        }
    }

    pub(super) fn nvfp4_source() -> SourcePrecision {
        let mut metadata = ExportMetadata::default();
        metadata
            .add_document(
                "quantization",
                br#"{"quant_method":"modelopt","quant_algo":"NVFP4","group_size":16}"#,
            )
            .unwrap();
        let mut tensors = BTreeMap::new();
        let mut operands = BTreeMap::new();
        for expert in 0..EXPERTS {
            for (projection, n, k) in [
                ("gate_proj", EXPERT_INTERMEDIATE, HIDDEN),
                ("up_proj", EXPERT_INTERMEDIATE, HIDDEN),
                ("down_proj", HIDDEN, EXPERT_INTERMEDIATE),
            ] {
                let stem =
                    format!("model.language_model.layers.0.mlp.experts.{expert}.{projection}");
                for (suffix, dtype, shape) in [
                    ("weight", DType::U8, vec![u64::from(n), u64::from(k / 2)]),
                    (
                        "weight_scale",
                        DType::Fp8E4M3,
                        vec![u64::from(n), u64::from(k / 16)],
                    ),
                    ("weight_scale_2", DType::F32, vec![]),
                    ("input_scale", DType::F32, vec![]),
                ] {
                    let name = format!("{stem}.{suffix}");
                    tensors.insert(
                        name.clone(),
                        TensorSpan {
                            tensor: name,
                            dtype,
                            byte_length: shape.iter().product::<u64>()
                                * kernel_core::precision::dtype_bits(dtype)
                                / 8,
                            offset: 0,
                            shape,
                        },
                    );
                }
                let weight = format!("{stem}.weight");
                let operand = import_operand(
                    &metadata,
                    &OperandRole {
                        operation: stem.clone(),
                        weight: weight.clone(),
                        targets: vec![stem],
                        logical_shape: vec![n.into(), k.into()],
                        lookup: false,
                        shared_scale_stem: None,
                        declared: None,
                    },
                    &tensors,
                )
                .unwrap();
                operands.insert(weight, operand);
            }
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

    pub(super) fn mxfp4_source() -> SourcePrecision {
        let mut source = nvfp4_source();
        for operand in source.operands.values_mut() {
            operand.weight.format = WeightFormat::Mxfp4;
            operand.weight.global_scale = None;
            let local = operand.weight.local_scale.as_mut().unwrap();
            local.tensor.dtype = DType::Ue8M0;
            *local.tensor.shape.last_mut().unwrap() /= 2;
            local.tensor.byte_length /= 2;
            local.groups.last_mut().unwrap().size = 32;
            operand.activation = Some(kernel_core::precision::ActivationSpec::bf16());
        }
        source
    }

    #[test]
    fn mxfp4_bf16_source_recipes_bind_without_fallback() {
        let source = mxfp4_source();
        for projection in [Projection::GateUp, Projection::Down] {
            let mut op = routed(&source, 0, projection, Shape::TP4).unwrap().source;
            op.uses.retain(|u| {
                u.capacity == 1 || u.phase == kernel_core::precision::ExecutionPhase::Prefill
            });
            let provider = super::sm120::mxfp4_inventory(&op);
            for policy in [ActivationPolicy::Checkpoint, ActivationPolicy::Auto] {
                let prepared = resolve_operation(&op, policy, &provider).unwrap();
                assert_eq!(
                    prepared.selection().source.arithmetic.decode_order,
                    DecodeOrder::ExactScaledOperand
                );
                assert_eq!(
                    prepared.selection().effective.arithmetic.decode_order,
                    DecodeOrder::ExactScaledOperand
                );
                assert!(prepared.selection().reuse.is_none());
                assert!(prepared.selection().fallback_reasons.is_empty());
            }
        }
    }

    #[test]
    fn bank_contracts_keep_distinct_scales_and_refuse_encoding_mixtures() {
        let mut source = nvfp4_source();
        check_routed(&source);
        let changed = "model.language_model.layers.0.mlp.experts.19.up_proj.weight";
        source.operands.get_mut(changed).unwrap().weight.encoding =
            CodeEncoding::E2M1(NibbleOrder::EvenHigh);
        let error = routed(&source, 0, Projection::GateUp, Shape::TP4)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains(changed) && error.contains("uniform"),
            "{error}"
        );
        // Another projection remains independently preparable.
        routed(&source, 0, Projection::Down, Shape::TP4).unwrap();
        // The fixture has no MTP experts, and no layer follows the MTP layer.
        assert!(routed(&source, MTP_LAYER, Projection::Down, Shape::TP4).is_err());
        assert!(routed(&source, MTP_LAYER + 1, Projection::Down, Shape::TP4).is_err());
    }

    #[test]
    fn real_checkpoint_routed_recipes_bind_all_uses_and_preserve_calibration() {
        let Ok(root) = std::env::var("CHOPINFER_QWEN38_CHECKPOINT") else {
            eprintln!("UNRUN: set CHOPINFER_QWEN38_CHECKPOINT for full routed recipe checks");
            return;
        };
        let catalog = crate::catalog::Catalog::load(std::path::Path::new(&root)).unwrap();
        check_routed(catalog.precision.as_ref().unwrap());
    }
}
