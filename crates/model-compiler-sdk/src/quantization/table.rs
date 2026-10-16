//! Join model-ordered table shards without expanding their packed contents.
use super::bank::{ImportedTable, TensorAssembly};
use super::import::ImportedOperand;
use crate::CompilerError;
use kernel_core::contract::WeightFormat;
use kernel_core::precision::{ScaleScope, TensorSpan, WeightLayout};
use std::collections::BTreeSet;

fn error(name: &str, reason: impl std::fmt::Display) -> CompilerError {
    CompilerError::validation(format!("lookup table {name}: {reason}"))
}

fn rows(
    name: String,
    parts: Vec<TensorSpan>,
    output: &mut Vec<TensorAssembly>,
) -> Result<TensorSpan, CompilerError> {
    let first = parts.first().ok_or_else(|| error(&name, "no table rows"))?;
    let mut shape = first.shape.clone();
    if shape.len() != 2 {
        return Err(error(&name, "table parts require row matrices"));
    }
    shape[0] = 0;
    let mut bytes = 0_u64;
    for part in &parts {
        part.validate().map_err(|e| error(&name, e))?;
        if part.dtype != first.dtype || part.shape.len() != 2 || part.shape[1] != first.shape[1] {
            return Err(error(
                &name,
                format!(
                    "tensor {}: table column encoding differs between shards",
                    part.tensor
                ),
            ));
        }
        shape[0] = shape[0]
            .checked_add(part.shape[0])
            .ok_or_else(|| error(&name, "row count overflow"))?;
        bytes = bytes
            .checked_add(part.byte_length)
            .ok_or_else(|| error(&name, "byte count overflow"))?;
    }
    let tensor = TensorSpan {
        tensor: name.clone(),
        dtype: first.dtype,
        shape,
        offset: 0,
        byte_length: bytes,
    };
    tensor.validate().map_err(|e| error(&name, e))?;
    output.push(TensorAssembly {
        tensor: tensor.clone(),
        parts,
    });
    Ok(tensor)
}

/// Shard order is supplied by the family. Shared scalar scale references must
/// agree; distinct scales need a separately supported table execution contract.
/// The result may exceed the bounded read size and must then be streamed by part.
pub fn concatenate_rows(
    name: &str,
    shards: &[ImportedOperand],
) -> Result<ImportedTable, CompilerError> {
    let first = shards
        .first()
        .ok_or_else(|| error(name, "empty shard list"))?;
    if shards.len() > 65536 {
        return Err(error(name, "shard count exceeds the supported bound"));
    }
    let mut logical_rows = 0_u64;
    for shard in shards {
        let weight = &shard.weight;
        weight.validate().map_err(|e| error(name, e))?;
        if shard.activation.is_some()
            || weight.logical_shape.len() != 2
            || weight.logical_shape[1] != first.weight.logical_shape[1]
            || weight.format != first.weight.format
            || weight.encoding != first.weight.encoding
            || weight.layout != WeightLayout::row_major()
            || !weight.transforms.is_empty()
            || weight.global_scale != first.weight.global_scale
        {
            return Err(error(
                name,
                format!(
                    "tensor {}: table shards require uniform columns, encoding and scalar globals without projection activations or unsupported transforms",
                    weight.codes.tensor
                ),
            ));
        }
        match weight.format {
            WeightFormat::Bf16 => {}
            WeightFormat::Fp8PerTensor if weight.local_scale == first.weight.local_scale => {}
            WeightFormat::Nvfp4 => {
                let local = weight.local_scale.as_ref().expect("validated NVFP4 scale");
                let expected = first
                    .weight
                    .local_scale
                    .as_ref()
                    .expect("validated NVFP4 scale");
                if local.scope != ScaleScope::Block
                    || local.groups != expected.groups
                    || local.convention != expected.convention
                {
                    return Err(error(
                        name,
                        format!(
                            "tensor {}: table scale grouping differs between shards",
                            local.tensor.tensor
                        ),
                    ));
                }
            }
            _ => {
                return Err(error(
                    name,
                    format!(
                        "tensor {}: unsupported table encoding or distinct scalar scales",
                        weight.codes.tensor
                    ),
                ));
            }
        }
        logical_rows = logical_rows
            .checked_add(weight.logical_shape[0])
            .ok_or_else(|| error(name, "logical row count overflow"))?;
    }
    let mut operand = first.clone();
    let mut assemblies = Vec::new();
    operand.weight.logical_shape[0] = logical_rows;
    operand.weight.codes = rows(
        format!("{name}.weight"),
        shards.iter().map(|s| s.weight.codes.clone()).collect(),
        &mut assemblies,
    )?;
    if operand.weight.format == WeightFormat::Nvfp4 {
        operand
            .weight
            .local_scale
            .as_mut()
            .expect("validated scale")
            .tensor = rows(
            format!("{name}.weight_scale"),
            shards
                .iter()
                .map(|s| {
                    s.weight
                        .local_scale
                        .as_ref()
                        .expect("validated scale")
                        .tensor
                        .clone()
                })
                .collect(),
            &mut assemblies,
        )?;
    }
    operand.source_tensors = shards
        .iter()
        .flat_map(|s| s.source_tensors.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    operand.weight.validate().map_err(|e| error(name, e))?;
    Ok(ImportedTable {
        operand,
        assemblies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::import::{ExportMetadata, OperandRole, import_operand};
    use kernel_core::contract::DType;
    use std::collections::BTreeMap;

    fn shards() -> Vec<ImportedOperand> {
        let mut metadata = ExportMetadata::default();
        metadata
            .add_document(
                "quantization",
                br#"{"quant_method":"modelopt","quant_algo":"W4A16_NVFP4","group_size":16}"#,
            )
            .unwrap();
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "vocabulary.weight_scale_2".into(),
            TensorSpan {
                tensor: "vocabulary.weight_scale_2".into(),
                dtype: DType::F32,
                shape: vec![],
                offset: 0,
                byte_length: 4,
            },
        );
        [3_u64, 4]
            .into_iter()
            .enumerate()
            .map(|(i, n)| {
                let stem = format!("vocabulary.shard_{i}");
                for (suffix, dtype, k) in [
                    ("weight", DType::U8, 80),
                    ("weight_scale", DType::Fp8E4M3, 10),
                ] {
                    let name = format!("{stem}.{suffix}");
                    tensors.insert(
                        name.clone(),
                        TensorSpan {
                            tensor: name,
                            dtype,
                            shape: vec![n, k],
                            offset: 0,
                            byte_length: n * k,
                        },
                    );
                }
                import_operand(
                    &metadata,
                    &OperandRole {
                        operation: stem.clone(),
                        weight: format!("{stem}.weight"),
                        targets: vec![stem],
                        logical_shape: vec![n, 160],
                        lookup: true,
                        shared_scale_stem: Some("vocabulary".into()),
                        declared: None,
                    },
                    &tensors,
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn table_shards_preserve_rows_scales_and_source_order() {
        let shards = shards();
        let table = concatenate_rows("lookup", &shards).unwrap();
        assert_eq!(table.operand.weight.logical_shape, [7, 160]);
        assert_eq!(table.operand.weight.codes.shape, [7, 80]);
        assert_eq!(
            table
                .operand
                .weight
                .local_scale
                .as_ref()
                .unwrap()
                .tensor
                .shape,
            [7, 10]
        );
        assert_eq!(
            table.assemblies[0].parts[1].tensor,
            "vocabulary.shard_1.weight"
        );
        let bytes = table
            .read_span(&table.operand.weight.codes, |part| {
                Ok(vec![
                    u8::from(part.tensor.contains("shard_1"));
                    part.byte_length as usize
                ])
            })
            .unwrap();
        assert_eq!(&bytes[239..241], &[0, 1]);
        let mut changed = shards.clone();
        changed[1]
            .weight
            .global_scale
            .as_mut()
            .unwrap()
            .tensor
            .tensor = "different_global".into();
        assert!(
            concatenate_rows("lookup", &changed)
                .unwrap_err()
                .to_string()
                .contains("scalar globals")
        );
        changed = shards;
        changed[1]
            .weight
            .transforms
            .push(kernel_core::precision::RequiredTransform::Unsupported {
                field: "rotation".into(),
                value: "unknown".into(),
            });
        assert!(concatenate_rows("lookup", &changed).is_err());
    }
}
