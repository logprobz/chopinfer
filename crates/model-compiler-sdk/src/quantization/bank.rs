//! Normalize separately exported expert matrices into one uniform bank contract.
//! The family supplies expert order. Assemblies retain every original tensor;
//! packers can stream their parts through the shared physical packer.

use super::import::ImportedOperand;
use crate::CompilerError;
use kernel_core::precision::{Calibration, ScaleScope, ScaleSpec, TensorSpan, WeightLayout};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorAssembly {
    pub tensor: TensorSpan,
    /// Concatenation order is part of interpretation and cache identity.
    pub parts: Vec<TensorSpan>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedAssembly {
    pub operand: ImportedOperand,
    pub assemblies: Vec<TensorAssembly>,
}

pub type ImportedBank = ImportedAssembly;
pub type ImportedTable = ImportedAssembly;

fn error(name: &str, reason: impl std::fmt::Display) -> CompilerError {
    CompilerError::validation(format!("expert bank {name}: {reason}"))
}

fn assemble(
    name: String,
    shape: Vec<u64>,
    parts: Vec<TensorSpan>,
    output: &mut Vec<TensorAssembly>,
) -> Result<TensorSpan, CompilerError> {
    let first = parts
        .first()
        .ok_or_else(|| error(&name, "empty source assembly"))?;
    let mut bytes = 0_u64;
    for part in &parts {
        part.validate().map_err(|e| error(&name, e))?;
        if part.dtype != first.dtype || part.byte_length != first.byte_length {
            return Err(error(&name, "expert encoding or companion sizes differ"));
        }
        bytes = bytes
            .checked_add(part.byte_length)
            .ok_or_else(|| error(&name, "assembly byte overflow"))?;
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

fn scalar_bank(
    name: String,
    scales: Vec<&ScaleSpec>,
    output: &mut Vec<TensorAssembly>,
) -> Result<ScaleSpec, CompilerError> {
    let first = scales[0];
    if scales.iter().any(|s| {
        s.tensor.elements().ok() != Some(1)
            || !s.groups.is_empty()
            || s.convention != first.convention
            || s.tensor.dtype != first.tensor.dtype
    }) {
        return Err(error(
            &name,
            "expected one compatible source scalar per expert",
        ));
    }
    // A common source span is already a proof of equal bits. Distinct spans
    // retain per-expert scope until a separate exact coalescing pass proves it.
    if scales.iter().all(|s| s.tensor == first.tensor) {
        return Ok(first.clone());
    }
    let tensor = assemble(
        name,
        vec![scales.len() as u64],
        scales.iter().map(|s| s.tensor.clone()).collect(),
        output,
    )?;
    Ok(ScaleSpec {
        tensor,
        scope: ScaleScope::Expert,
        convention: first.convention,
        groups: vec![],
    })
}

pub fn concatenate_experts(
    name: &str,
    experts: &[ImportedOperand],
) -> Result<ImportedBank, CompilerError> {
    let first = experts
        .first()
        .ok_or_else(|| error(name, "empty expert list"))?;
    if experts.len() > 65536 {
        return Err(error(name, "expert count exceeds the supported bound"));
    }
    for (i, expert) in experts.iter().enumerate() {
        expert.weight.validate().map_err(|e| error(name, e))?;
        let weight = &expert.weight;
        if weight.logical_shape.len() != 2
            || weight.logical_shape != first.weight.logical_shape
            || weight.format != first.weight.format
            || weight.encoding != first.weight.encoding
            || weight.layout != WeightLayout::row_major()
            || !weight.transforms.is_empty()
        {
            return Err(error(
                name,
                format!(
                    "expert {i}, tensor {}: bank requires uniform matrix geometry and encoding without unsupported transforms",
                    weight.codes.tensor
                ),
            ));
        }
        if let Some(a) = &expert.activation {
            a.validate(&weight.codes.tensor, &[1, weight.logical_shape[1]])
                .map_err(|e| error(name, e))?;
            if !a.transforms.is_empty() {
                return Err(error(
                    name,
                    "activation transforms require a supported bank composition",
                ));
            }
        }
    }
    let mut assemblies = Vec::new();
    let mut operand = first.clone();
    operand.weight.logical_shape.insert(0, experts.len() as u64);
    let mut shape = first.weight.codes.shape.clone();
    shape.insert(0, experts.len() as u64);
    operand.weight.codes = assemble(
        format!("{name}.weight"),
        shape,
        experts.iter().map(|e| e.weight.codes.clone()).collect(),
        &mut assemblies,
    )?;
    if let Some(local) = &first.weight.local_scale {
        let mut scales = Vec::new();
        for expert in experts {
            let actual = expert
                .weight
                .local_scale
                .as_ref()
                .ok_or_else(|| error(name, "missing expert local scale"))?;
            if actual.scope != local.scope
                || actual.groups != local.groups
                || actual.convention != local.convention
            {
                return Err(error(
                    name,
                    "local scale grouping or convention differs between experts",
                ));
            }
            scales.push(actual.tensor.clone());
        }
        if local.scope != ScaleScope::Block {
            return Err(error(
                name,
                "this expert assembly supports block-scaled or unscaled matrices",
            ));
        }
        let mut shape = local.tensor.shape.clone();
        shape.insert(0, experts.len() as u64);
        let mut grouped = local.clone();
        grouped.tensor = assemble(
            format!("{name}.weight_scale"),
            shape,
            scales,
            &mut assemblies,
        )?;
        for group in &mut grouped.groups {
            group.axis += 1;
        }
        operand.weight.local_scale = Some(grouped);
    }
    if first.weight.global_scale.is_some() {
        let scales = experts
            .iter()
            .map(|e| {
                e.weight
                    .global_scale
                    .as_ref()
                    .ok_or_else(|| error(name, "missing expert global scale"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        operand.weight.global_scale = Some(scalar_bank(
            format!("{name}.weight_scale_2"),
            scales,
            &mut assemblies,
        )?);
    }
    if let Some(input) = &first.activation {
        let mut normalized = input.clone();
        if let Some(q) = &mut normalized.quantizer
            && matches!(q.calibration, Calibration::Static(_))
        {
            let mut scales = Vec::new();
            for expert in experts {
                let a = expert
                    .activation
                    .as_ref()
                    .ok_or_else(|| error(name, "missing activation recipe"))?;
                let mut comparable = a.clone();
                let actual = comparable
                    .quantizer
                    .as_mut()
                    .ok_or_else(|| error(name, "activation quantizer differs between experts"))?;
                let Calibration::Static(scale) = &actual.calibration else {
                    return Err(error(
                        name,
                        "activation calibration differs between experts",
                    ));
                };
                scales.push(scale.clone());
                actual.calibration = q.calibration.clone();
                if comparable != *input {
                    return Err(error(name, "activation recipe differs between experts"));
                }
            }
            q.calibration = Calibration::Static(scalar_bank(
                format!("{name}.input_scale"),
                scales.iter().collect(),
                &mut assemblies,
            )?);
        } else if experts.iter().any(|e| e.activation.as_ref() != Some(input)) {
            return Err(error(name, "activation recipe differs between experts"));
        }
        operand.activation = Some(normalized);
    } else if experts.iter().any(|e| e.activation.is_some()) {
        return Err(error(
            name,
            "lookup and projection input contracts cannot share a bank",
        ));
    }
    operand.source_tensors = experts
        .iter()
        .flat_map(|e| e.source_tensors.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    operand.weight.validate().map_err(|e| error(name, e))?;
    Ok(ImportedBank {
        operand,
        assemblies,
    })
}

impl ImportedAssembly {
    /// Bounded convenience for small banks and scalar assemblies. Large bank
    /// packers iterate `parts` and pack one source matrix at a time.
    pub fn read_span(
        &self,
        span: &TensorSpan,
        mut read: impl FnMut(&TensorSpan) -> Result<Vec<u8>, CompilerError>,
    ) -> Result<Vec<u8>, CompilerError> {
        span.validate().map_err(|e| error(&span.tensor, e))?;
        if span.byte_length > 2 * 1024 * 1024 * 1024 {
            return Err(error(
                &span.tensor,
                "assembled read exceeds the 2 GiB source bound",
            ));
        }
        let mut output = Vec::new();
        if let Some(assembly) = self
            .assemblies
            .iter()
            .find(|a| a.tensor.tensor == span.tensor)
        {
            if &assembly.tensor != span {
                return Err(error(
                    &span.tensor,
                    "requested span differs from the assembly",
                ));
            }
            for part in &assembly.parts {
                part.validate().map_err(|e| error(&span.tensor, e))?;
                if part.dtype != span.dtype
                    || part.byte_length > span.byte_length - output.len() as u64
                {
                    return Err(error(
                        &span.tensor,
                        "assembly part exceeds its dtype or byte bounds",
                    ));
                }
                let bytes = read(part)?;
                if bytes.len() as u64 != part.byte_length
                    || output.len() as u64 + part.byte_length > span.byte_length
                {
                    return Err(error(&span.tensor, "assembly part exceeds its bounds"));
                }
                output.extend(bytes);
            }
        } else {
            output = read(span)?;
        }
        if output.len() as u64 != span.byte_length {
            return Err(error(
                &span.tensor,
                "assembled length disagrees with the descriptor",
            ));
        }
        Ok(output)
    }
}
