//! Format plug-in weight records. Placement and checkpoint naming stay with
//! the caller; records use the existing host transform ABI without new IDs.

use crate::CompilerError;
use kernel_core::contract::DType;
use kernel_core::precision::DecodeOrder;
use kernel_core::precision::format::{FormatId, FormatPlugin};
use kernel_core::precision::layout::PackingPlan;
use kernel_core::transform::fragment::FragmentWeightFormat;
use kernel_core::weight_transform::{
    Mxfp4Section, PackedMatrixSource, PackedOperand, PackedSection, SourceTensor, WeightRecord,
    WeightTransform,
};
use std::ops::Range;

fn error(reason: &str) -> CompilerError {
    CompilerError::validation(format!("format weight records: {reason}"))
}

/// Make one record for each resident section, preserving the caller's bank
/// order and exact source selections. Absent companions produce no records.
pub fn packed_records(
    plugin: &dyn FormatPlugin,
    plan: PackingPlan,
    decode_order: DecodeOrder,
    matrices: Vec<PackedMatrixSource>,
    sources: Vec<SourceTensor>,
) -> Result<Vec<WeightRecord>, CompilerError> {
    plan.validate()?;
    if !plugin.accepts(plan.format) || matrices.iter().any(|m| m.format != plan.format) {
        return Err(error(
            "plug-in disagrees with packing plan or source matrices",
        ));
    }
    let sizes = plan.section_bytes()?;
    let mut records = Vec::new();
    for (section, bytes) in [
        (PackedSection::Codes, sizes.codes),
        (PackedSection::LocalScales, sizes.local_scales),
        (PackedSection::GlobalScales, sizes.global_scales),
    ] {
        if bytes == 0 {
            continue;
        }
        let record = WeightRecord {
            transform: WeightTransform::PackedOperand(Box::new(PackedOperand {
                plan: plan.clone(),
                section,
                decode_order,
                matrices: matrices.clone(),
            })),
            sources: sources.clone(),
            axis: None,
            ranges: vec![],
            stored_shape: vec![],
        };
        record
            .output_bytes()
            .map_err(|e| CompilerError::validation(e.to_string()))?;
        records.push(record);
    }
    Ok(records)
}

/// Legacy single-section layouts that already have dedicated host transforms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecordLayout {
    Rows,
    Fragment,
    Mxfp4Strips,
}

/// Build a legacy gather/fragment record without changing its wire transform.
/// BF16 scales widen only when the physical consumer requests F32. A scale
/// section stored as it is checks each scale while it streams (finite, sign
/// bit clear), as a packed operand's scales are checked.
pub fn section_record(
    plugin: &dyn FormatPlugin,
    section: PackedSection,
    source_dtype: DType,
    launch_dtype: DType,
    layout: RecordLayout,
    mut record: WeightRecord,
) -> Result<WeightRecord, CompilerError> {
    let dtype_ok = match (section, plugin.id()) {
        (PackedSection::Codes, FormatId::F32) => source_dtype == DType::F32,
        (PackedSection::Codes, FormatId::Bf16) => source_dtype == DType::Bf16,
        (PackedSection::Codes, FormatId::Mxfp4 | FormatId::Nvfp4 | FormatId::Nvfp4WeightOnly) => {
            source_dtype == DType::U8
        }
        (PackedSection::Codes, _) => source_dtype == DType::Fp8E4M3,
        (PackedSection::LocalScales, FormatId::Bf16 | FormatId::F32) => false,
        (PackedSection::LocalScales, FormatId::Fp8Block32 | FormatId::Mxfp4 | FormatId::Mxfp8) => {
            matches!(source_dtype, DType::U8 | DType::Ue8M0)
        }
        (PackedSection::LocalScales, FormatId::Nvfp4 | FormatId::Nvfp4WeightOnly) => {
            matches!(source_dtype, DType::Fp8E4M3 | DType::Ue4M3)
        }
        (PackedSection::LocalScales, _) => matches!(source_dtype, DType::Bf16 | DType::F32),
        (PackedSection::GlobalScales, FormatId::Nvfp4 | FormatId::Nvfp4WeightOnly) => {
            source_dtype == DType::F32
        }
        (PackedSection::GlobalScales, _) => false,
    };
    if !dtype_ok
        || layout != RecordLayout::Rows
            && section == PackedSection::Codes
            && source_dtype != launch_dtype
    {
        return Err(error(
            "section dtype disagrees with the format or physical layout",
        ));
    }
    let transform = match (layout, section, plugin.id()) {
        (RecordLayout::Fragment, PackedSection::Codes, FormatId::Bf16) => {
            WeightTransform::Fragment(FragmentWeightFormat::Bf16)
        }
        (
            RecordLayout::Fragment,
            PackedSection::Codes,
            FormatId::Fp8Block128 | FormatId::Fp8Rows64,
        ) => WeightTransform::Fragment(FragmentWeightFormat::Fp8E4M3),
        (RecordLayout::Mxfp4Strips, PackedSection::Codes, FormatId::Mxfp4) => {
            WeightTransform::Mxfp4Moe(Mxfp4Section::Codes)
        }
        (RecordLayout::Mxfp4Strips, PackedSection::LocalScales, FormatId::Mxfp4) => {
            WeightTransform::Mxfp4Moe(Mxfp4Section::Scales)
        }
        (RecordLayout::Fragment, PackedSection::Codes, _) => {
            return Err(error("format has no fragment code transform"));
        }
        (RecordLayout::Rows | RecordLayout::Fragment, _, _) => {
            if source_dtype == DType::Bf16 && launch_dtype == DType::F32 {
                WeightTransform::WidenBf16F32
            } else if source_dtype == launch_dtype
                && section != PackedSection::Codes
                && matches!(
                    source_dtype,
                    DType::F32 | DType::Bf16 | DType::Fp8E4M3 | DType::Ue4M3
                )
            {
                WeightTransform::CopyScales
            } else if source_dtype == launch_dtype {
                WeightTransform::Copy
            } else {
                return Err(error("unsupported section dtype conversion"));
            }
        }
        _ => return Err(error("section has no transform in the selected layout")),
    };
    record.transform = transform;
    record
        .output_bytes()
        .map_err(|e| CompilerError::validation(e.to_string()))?;
    Ok(record)
}

/// Source-space selection retained in a packed record. The plug-in validates
/// the imported format while the repacker remains the authority for payloads.
pub fn matrix_source(
    plugin: &dyn FormatPlugin,
    weight: &kernel_core::precision::WeightSpec,
    rows_kept: Vec<Range<u64>>,
    k_kept: Range<u64>,
) -> Result<PackedMatrixSource, CompilerError> {
    plugin.companions(weight)?;
    if !weight.transforms.is_empty()
        || weight
            .local_scale
            .iter()
            .chain(weight.global_scale.iter())
            .any(|s| s.convention != kernel_core::precision::ScaleConvention::Multiply)
    {
        return Err(error(
            "packed records require multiplicative scales and no unbound transforms",
        ));
    }
    let rank = weight.logical_shape.len();
    let rows = weight.logical_shape[rank - 2];
    let k = weight.logical_shape[rank - 1];
    if rows_kept.is_empty()
        || rows_kept.iter().any(|r| r.is_empty() || r.end > rows)
        || k_kept.is_empty()
        || k_kept.end > k
    {
        return Err(error("selection exceeds the imported operand"));
    }
    if weight.layout != kernel_core::precision::WeightLayout::row_major() {
        return Err(error("matrix source requires imported row-major storage"));
    }
    let origin = |axis| {
        weight
            .local_scale
            .as_ref()
            .and_then(|s| {
                s.groups
                    .iter()
                    .find(|g| usize::from(g.axis) == axis)
                    .map(|g| g.origin)
            })
            .unwrap_or(0)
    };
    Ok(PackedMatrixSource {
        expert: 0,
        projection: 0,
        format: weight.format,
        rows,
        k,
        row_origin: origin(rank - 2),
        k_origin: origin(rank - 1),
        nibble_order: match weight.encoding {
            kernel_core::precision::CodeEncoding::E2M1(order) => order,
            _ => kernel_core::precision::NibbleOrder::EvenLow,
        },
        codes: 0,
        local: weight.local_scale.as_ref().map(|_| 1),
        local_dtype: weight.local_scale.as_ref().map(|s| s.tensor.dtype),
        global: weight.global_scale.as_ref().map(|_| 2),
        source_expert: 0,
        source_experts: if rank == 3 {
            weight.logical_shape[0]
        } else {
            1
        },
        rows_kept,
        k_kept,
    })
}

/// The checkpoint source list for `matrix_source`, in codes/local/global order.
pub fn sources(weight: &kernel_core::precision::WeightSpec) -> Vec<SourceTensor> {
    std::iter::once(&weight.codes)
        .chain(weight.local_scale.iter().map(|s| &s.tensor))
        .chain(weight.global_scale.iter().map(|s| &s.tensor))
        .map(|s| SourceTensor {
            dtype: match s.dtype {
                DType::Bf16 => "BF16",
                DType::F32 => "F32",
                DType::Fp8E4M3 | DType::Ue4M3 => "F8_E4M3",
                _ => "U8",
            }
            .into(),
            shape: s.shape.clone(),
            byte_length: s.byte_length,
        })
        .collect()
}

#[cfg(test)]
mod tests;
