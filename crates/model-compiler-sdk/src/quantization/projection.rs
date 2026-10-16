//! Persistent records for rank-local projection derivations. Source names and
//! existing host transform identifiers remain part of the caller's weight plan.

pub mod metadata;
pub mod persistent;
mod policy;
pub use policy::{RecordPolicy, bind_record_policy};

use crate::CompilerError;
use crate::quantization::format::{self, RecordLayout};
use kernel_core::precision::format::registry;
use kernel_core::weight_transform::{PackedSection, SourceTensor, WeightRecord};
use projection_spec::derive::{DerivedOperation, RankDerivation, VariantRecord, ViewRole};

fn error(reason: &str) -> CompilerError {
    CompilerError::validation(format!("projection records: {reason}"))
}

fn check_record(operation: &DerivedOperation, record: &VariantRecord) -> Result<(), CompilerError> {
    let view = operation
        .views
        .iter()
        .find(|v| v.id == record.view && v.role == ViewRole::Weight(record.section))
        .ok_or_else(|| error("record has no matching declared weight view"))?;
    let bytes = record
        .record
        .output_bytes()
        .map_err(|e| error(&e.to_string()))?;
    let end = record
        .view_offset
        .checked_add(bytes)
        .ok_or_else(|| error("record placement overflows"))?;
    if record.record.sources.len() != record.tensors.len() || bytes == 0 || end > view.bytes {
        return Err(error(
            "record source count or placement disagrees with its view",
        ));
    }
    Ok(())
}

fn check_records(
    operation: &DerivedOperation,
    records: &[VariantRecord],
) -> Result<(), CompilerError> {
    for record in records {
        check_record(operation, record)?;
    }
    for view in operation
        .views
        .iter()
        .filter(|v| matches!(v.role, ViewRole::Weight(_)))
    {
        let mut pieces: Vec<_> = records.iter().filter(|r| r.view == view.id).collect();
        pieces.sort_by_key(|r| r.view_offset);
        let mut cursor = 0;
        for piece in pieces {
            if piece.view_offset != cursor {
                return Err(error("resident view records overlap or leave a hole"));
            }
            cursor = cursor
                .checked_add(
                    piece
                        .record
                        .output_bytes()
                        .map_err(|e| error(&e.to_string()))?,
                )
                .ok_or_else(|| error("record placement overflows"))?;
        }
        if cursor != view.bytes {
            return Err(error("resident view records leave a hole"));
        }
    }
    Ok(())
}

/// Build existing PACKED_OPERAND records, or preserve a provider's explicit
/// legacy records. Each ordinary expert/operand needs one source matrix.
/// Offset part views and multipart matrices require provider-owned assembly.
pub fn records(operation: &DerivedOperation) -> Result<Vec<VariantRecord>, CompilerError> {
    if !operation.variant_records.is_empty() {
        check_records(operation, &operation.variant_records)?;
        return Ok(operation.variant_records.clone());
    }
    let first = operation
        .slices
        .first()
        .ok_or_else(|| error("operation has no source selections"))?;
    let plugin = registry::by_id(first.format)?;
    let mut sources: Vec<SourceTensor> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut matrices = Vec::new();
    for slice in &operation.slices {
        if slice.format != first.format {
            return Err(error("ordinary operands use different format plug-ins"));
        }
        let weight = &slice.source.weight;
        let spans: Vec<_> = std::iter::once(&weight.codes)
            .chain(weight.local_scale.iter().map(|s| &s.tensor))
            .chain(weight.global_scale.iter().map(|s| &s.tensor))
            .collect();
        for span in &spans {
            let partial = operation
                .operation
                .weights
                .iter()
                .flat_map(|w| {
                    std::iter::once(&w.codes)
                        .chain(w.local_scale.iter().map(|s| &s.tensor))
                        .chain(w.global_scale.iter().map(|s| &s.tensor))
                })
                .any(|full| full.tensor == span.tensor && full != *span);
            if span.offset != 0 || partial {
                return Err(error(
                    "part views require an explicit provider record or source assembly",
                ));
            }
        }
        let described = format::sources(weight);
        let mut indices = Vec::new();
        for (span, source) in spans.iter().zip(described) {
            let index = if let Some(index) = names.iter().position(|name| *name == span.tensor) {
                if sources[index] != source {
                    return Err(error(
                        "one checkpoint tensor has conflicting source metadata",
                    ));
                }
                index
            } else {
                let index = names.len();
                names.push(span.tensor.clone());
                sources.push(source);
                index
            };
            indices.push(u32::try_from(index).map_err(|_| error("source count exceeds u32"))?);
        }
        let mut matrix =
            format::matrix_source(plugin, weight, slice.rows.clone(), slice.k.clone())?;
        matrix.expert = slice.expert;
        matrix.projection = slice.operand;
        matrix.source_expert = slice.source.source_expert;
        matrix.codes = indices[0];
        matrix.local = weight.local_scale.as_ref().map(|_| indices[1]);
        matrix.global = weight
            .global_scale
            .as_ref()
            .map(|_| indices[1 + usize::from(weight.local_scale.is_some())]);
        matrices.push(matrix);
    }
    let raw = format::packed_records(
        plugin,
        operation.packing.clone(),
        operation.operation.arithmetic.decode_order,
        matrices,
        sources,
    )?;
    let views: Vec<_> = operation
        .views
        .iter()
        .filter_map(|v| match v.role {
            ViewRole::Weight(section) => Some((v, section)),
            _ => None,
        })
        .collect();
    if views.len() != raw.len() {
        return Err(error(
            "packing sections disagree with required weight views",
        ));
    }
    raw.into_iter()
        .zip(views)
        .map(|(record, (view, section))| {
            let result = VariantRecord {
                view: view.id.clone(),
                view_offset: 0,
                section,
                record,
                tensors: names.clone(),
            };
            check_record(operation, &result)?;
            Ok(result)
        })
        .collect()
}

/// Preserve an explicit legacy gather selection through the format factory.
/// Providers choose the admitted existing layout and its source selection.
pub fn section_record(
    operation: &DerivedOperation,
    section: PackedSection,
    layout: RecordLayout,
    base: WeightRecord,
    tensors: Vec<String>,
) -> Result<VariantRecord, CompilerError> {
    let first = operation
        .slices
        .first()
        .ok_or_else(|| error("operation has no source selections"))?;
    let weight = &first.source.weight;
    let source_dtype = match section {
        PackedSection::Codes => weight.codes.dtype,
        PackedSection::LocalScales => {
            weight
                .local_scale
                .as_ref()
                .ok_or_else(|| error("operation has no local scales"))?
                .tensor
                .dtype
        }
        PackedSection::GlobalScales => {
            weight
                .global_scale
                .as_ref()
                .ok_or_else(|| error("operation has no global scales"))?
                .tensor
                .dtype
        }
    };
    let view = operation
        .views
        .iter()
        .find(|v| v.role == ViewRole::Weight(section))
        .ok_or_else(|| error("section has no required view"))?;
    // Code views hold encoded bits. Their byte storage may use U8 while
    // the checkpoint source retains its FP8 element spelling.
    let launch_dtype = if section == PackedSection::Codes {
        if kernel_core::precision::dtype_bits(source_dtype)
            != kernel_core::precision::dtype_bits(view.dtype)
        {
            return Err(error("code view changes the source element width"));
        }
        source_dtype
    } else {
        view.dtype
    };
    let record = format::section_record(
        registry::by_id(first.format)?,
        section,
        source_dtype,
        launch_dtype,
        layout,
        base,
    )?;
    let result = VariantRecord {
        view: view.id.clone(),
        view_offset: 0,
        section,
        record,
        tensors,
    };
    check_record(operation, &result)?;
    Ok(result)
}

/// Deduplicate resident views across requested buckets. Reusing a view with
/// different records is refused instead of depending on declaration order.
pub fn rank_records(rank: &RankDerivation) -> Result<Vec<VariantRecord>, CompilerError> {
    let mut resident = std::collections::BTreeMap::<String, Vec<VariantRecord>>::new();
    for operation in rank.selected_operations() {
        let mut grouped = std::collections::BTreeMap::<String, Vec<VariantRecord>>::new();
        for record in records(operation)? {
            grouped.entry(record.view.clone()).or_default().push(record);
        }
        for (view, mut pieces) in grouped {
            pieces.sort_by_key(|r| r.view_offset);
            if let Some(previous) = resident.get(&view) {
                if *previous != pieces {
                    return Err(error("one resident view has conflicting weight records"));
                }
            } else {
                resident.insert(view, pieces);
            }
        }
    }
    Ok(resident.into_values().flatten().collect())
}

/// One rank's copy of each bound side tensor: the rank's ranges of its
/// declared placement axis, read from the whole checkpoint tensor. A scalar
/// stores one element, and a rank past the tensor's last element stores its
/// zero padding inline; every record holds its selection's stored bytes.
pub fn side_records(
    declaration: &projection_spec::Declaration,
    bound: &projection_spec::bind::BoundDeclaration,
    geometry: &projection_spec::Geometry,
    topology: projection_spec::derive::Topology,
    metadata: &projection_spec::bind::CheckpointMetadata,
) -> Result<Vec<(String, WeightRecord)>, CompilerError> {
    use kernel_core::weight_transform::WeightTransform;
    let mut result: Vec<(String, WeightRecord)> = Vec::new();
    for view in &bound.side_views {
        if result.iter().any(|(tensor, _)| *tensor == view.tensor) {
            continue;
        }
        let side = declaration
            .side_tensors
            .iter()
            .find(|s| s.id == view.side)
            .ok_or_else(|| error("bound side has no declaration"))?;
        let info = metadata
            .tensors
            .get(&view.tensor)
            .ok_or_else(|| error("bound side has no checkpoint header"))?;
        let width = [
            kernel_core::contract::DType::Bf16,
            kernel_core::contract::DType::F16,
            kernel_core::contract::DType::F32,
            kernel_core::contract::DType::U8,
        ]
        .into_iter()
        .find(|d| d.checkpoint_dtype() == Some(info.dtype.as_str()))
        .map(|d| kernel_core::precision::dtype_bits(d) / 8)
        .ok_or_else(|| error("side tensor dtype has no copy width"))?;
        let selection = side
            .selection(geometry, topology)
            .map_err(|e| error(&e.0))?;
        // The host gathers a scalar as one element.
        let stored_shape = if selection.shape.is_empty() {
            vec![1]
        } else {
            selection.shape
        };
        let bytes = stored_shape.iter().product::<u64>() * width;
        let ranges: Vec<_> = selection
            .ranges
            .into_iter()
            .filter(|range| range.start < range.end)
            .collect();
        let record = if ranges.is_empty() {
            // A rank past the tensor's last element holds only padding.
            WeightRecord {
                transform: WeightTransform::Inline(vec![0; bytes as usize]),
                sources: vec![],
                axis: None,
                ranges: vec![],
                stored_shape,
            }
        } else {
            WeightRecord {
                transform: WeightTransform::Copy,
                sources: vec![SourceTensor {
                    dtype: info.dtype.clone(),
                    shape: info.shape.clone(),
                    byte_length: info.shape.iter().product::<u64>() * width,
                }],
                axis: Some(selection.axis.unwrap_or(0) as u32),
                ranges: ranges.into_iter().collect(),
                stored_shape,
            }
        };
        if record.output_bytes().map_err(|e| error(&e.to_string()))? != bytes {
            return Err(error("side record differs from its rank selection"));
        }
        result.push((view.tensor.clone(), record));
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
