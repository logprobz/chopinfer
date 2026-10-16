//! Read one logical operand through its owning source inventory. Companions can
//! live in different files; all reads use the supplied bounded tensor spans.

use super::import::ImportedOperand;
use crate::CompilerError;
use kernel_core::contract::{DType, WeightFormat};
use kernel_core::precision::packed::{MatrixView, ScaleBytes};
use kernel_core::precision::{Calibration, CodeEncoding, NibbleOrder, TensorSpan};
use std::collections::BTreeMap;

pub struct LoadedOperand {
    descriptor: ImportedOperand,
    payloads: BTreeMap<String, Vec<u8>>,
}

/// Decode selected lookup rows without reading or expanding a whole embedding
/// table. The callback admits each tensor-relative span against captured files.
pub fn read_lookup_rows(
    descriptor: &ImportedOperand,
    rows: &[u64],
    order: kernel_core::precision::DecodeOrder,
    mut read: impl FnMut(&TensorSpan) -> Result<Vec<u8>, CompilerError>,
) -> Result<Vec<Vec<u16>>, CompilerError> {
    let weight = &descriptor.weight;
    weight.validate()?;
    if descriptor.activation.is_some()
        || !weight.transforms.is_empty()
        || weight.logical_shape.len() != 2
        || weight.layout != kernel_core::precision::WeightLayout::row_major()
        || !matches!(
            weight.format,
            WeightFormat::Bf16 | WeightFormat::Fp8PerTensor | WeightFormat::Nvfp4
        )
        || !order.scales_before_bf16_mma()
    {
        return Err(error(
            &weight.codes.tensor,
            "lookup row decoding requires a supported plain BF16, scalar FP8 or NVFP4 operand and explicit pre-BF16 scale order",
        ));
    }
    let count = weight.logical_shape[0];
    let select = |span: &mut TensorSpan, row: u64| -> Result<(), CompilerError> {
        let bytes = span.byte_length / count;
        span.offset = row
            .checked_mul(bytes)
            .and_then(|n| span.offset.checked_add(n))
            .ok_or_else(|| error(&span.tensor, "row offset overflow"))?;
        span.shape[0] = 1;
        span.byte_length = bytes;
        Ok(())
    };
    rows.iter()
        .map(|&row| {
            if row >= count {
                return Err(error(
                    &weight.codes.tensor,
                    "lookup row exceeds its logical bounds",
                ));
            }
            let mut row_operand = descriptor.clone();
            row_operand.weight.logical_shape[0] = 1;
            select(&mut row_operand.weight.codes, row)?;
            if weight.format == WeightFormat::Nvfp4 {
                let scale = row_operand.weight.local_scale.as_mut().unwrap();
                select(&mut scale.tensor, row)?;
                let axis = scale.groups.iter_mut().find(|g| g.axis == 0).unwrap();
                axis.origin = axis
                    .origin
                    .checked_add(row)
                    .ok_or_else(|| error(&scale.tensor.tensor, "scale row origin overflow"))?;
            }
            let loaded = read_operand(row_operand, &mut read)?;
            let matrix = loaded.matrix(0)?;
            (0..matrix.k)
                .map(|k| matrix.bf16_operand(0, k, order).map_err(Into::into))
                .collect()
        })
        .collect()
}

fn error(tensor: &str, reason: impl std::fmt::Display) -> CompilerError {
    CompilerError::validation(format!("tensor {tensor}: precision payload: {reason}"))
}

pub fn read_operand(
    descriptor: ImportedOperand,
    mut read: impl FnMut(&TensorSpan) -> Result<Vec<u8>, CompilerError>,
) -> Result<LoadedOperand, CompilerError> {
    let weight = &descriptor.weight;
    weight
        .validate()
        .map_err(|e| error(&weight.codes.tensor, e))?;
    let mut spans = vec![&weight.codes];
    spans.extend(
        weight
            .local_scale
            .iter()
            .chain(weight.global_scale.iter())
            .map(|s| &s.tensor),
    );
    if let Some(activation) = &descriptor.activation {
        activation
            .validate(
                &weight.codes.tensor,
                &[1, *weight.logical_shape.last().unwrap()],
            )
            .map_err(|e| error(&weight.codes.tensor, e))?;
        if let Some(q) = &activation.quantizer
            && let Calibration::Static(scale) = &q.calibration
        {
            spans.push(&scale.tensor);
        }
    }
    let mut payloads = BTreeMap::new();
    for span in spans {
        span.validate().map_err(|e| error(&span.tensor, e))?;
        let bytes = read(span)?;
        if bytes.len() as u64 != span.byte_length {
            return Err(error(
                &span.tensor,
                "read length does not match the authenticated span",
            ));
        }
        if payloads.insert(span.tensor.clone(), bytes).is_some() {
            return Err(error(&span.tensor, "ambiguous duplicate operand source"));
        }
    }
    let loaded = LoadedOperand {
        descriptor,
        payloads,
    };
    for expert in 0..loaded.experts() {
        loaded
            .matrix(expert)?
            .validate()
            .map_err(|e| error(&loaded.descriptor.weight.codes.tensor, e))?;
    }
    if let Some(q) = loaded
        .descriptor
        .activation
        .as_ref()
        .and_then(|a| a.quantizer.as_ref())
        && let Calibration::Static(scale) = &q.calibration
    {
        let count = scale
            .tensor
            .elements()
            .map_err(|e| error(&scale.tensor.tensor, e))?;
        if count != 1 && count != loaded.experts() as u64 {
            return Err(error(
                &scale.tensor.tensor,
                "calibration count does not match its expert domain",
            ));
        }
        ScaleBytes {
            dtype: scale.tensor.dtype,
            bytes: &loaded.payloads[&scale.tensor.tensor],
        }
        .validate(count as usize, true)
        .map_err(|e| error(&scale.tensor.tensor, e))?;
    }
    Ok(loaded)
}

impl LoadedOperand {
    pub fn descriptor(&self) -> &ImportedOperand {
        &self.descriptor
    }
    pub fn payloads(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.payloads
    }
    pub fn into_parts(self) -> (ImportedOperand, BTreeMap<String, Vec<u8>>) {
        (self.descriptor, self.payloads)
    }
    pub fn experts(&self) -> usize {
        let shape = &self.descriptor.weight.logical_shape;
        if shape.len() == 3 {
            shape[0] as usize
        } else {
            1
        }
    }

    pub fn matrix(&self, expert: usize) -> Result<MatrixView<'_>, CompilerError> {
        let weight = &self.descriptor.weight;
        let name = weight.codes.tensor.as_str();
        let shape = &weight.logical_shape;
        if expert >= self.experts() {
            return Err(error(name, "expert index out of bounds"));
        }
        let rows = shape[shape.len() - 2] as usize;
        let k = shape[shape.len() - 1] as usize;
        let codes = &self.payloads[name];
        let code_bytes = codes.len() / self.experts();
        let local = weight.local_scale.as_ref().map(|scale| {
            let bytes = &self.payloads[&scale.tensor.tensor];
            let size = bytes.len() / self.experts();
            ScaleBytes {
                dtype: scale.tensor.dtype,
                bytes: &bytes[expert * size..(expert + 1) * size],
            }
        });
        let global = weight.global_scale.as_ref().map(|scale| {
            let bytes = &self.payloads[&scale.tensor.tensor];
            let index = if bytes.len() == 4 { 0 } else { expert * 4 };
            ScaleBytes {
                dtype: DType::F32,
                bytes: &bytes[index..index + 4],
            }
        });
        // The importer admits these source encodings only in row-major form.
        if weight.layout != kernel_core::precision::WeightLayout::row_major() {
            return Err(error(
                name,
                "source payload decoder requires a normalized row-major layout",
            ));
        }
        let nibble_order = match weight.encoding {
            CodeEncoding::E2M1(order) => order,
            _ => NibbleOrder::EvenLow,
        };
        let matrix = MatrixView {
            tensor: name,
            format: weight.format,
            rows,
            k,
            row_origin: weight
                .local_scale
                .as_ref()
                .and_then(|s| {
                    s.groups
                        .iter()
                        .find(|g| usize::from(g.axis) == shape.len() - 2)
                })
                .map_or(Ok(0), |g| usize::try_from(g.origin))
                .map_err(|e| error(name, e))?,
            k_origin: weight
                .local_scale
                .as_ref()
                .and_then(|s| {
                    s.groups
                        .iter()
                        .find(|g| usize::from(g.axis) == shape.len() - 1)
                })
                .map_or(Ok(0), |g| usize::try_from(g.origin))
                .map_err(|e| error(name, e))?,
            nibble_order,
            codes: &codes[expert * code_bytes..(expert + 1) * code_bytes],
            local,
            global,
        };
        if weight.format == WeightFormat::Fp8PerTensor && self.experts() > 1 {
            return Err(error(
                name,
                "per-tensor FP8 bank sharing requires an explicit supported scale layout",
            ));
        }
        Ok(matrix)
    }
}
