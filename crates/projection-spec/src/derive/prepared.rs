//! Compose existing preparation records without changing their host transform ABI.
use super::assembly::rank_rows;
use super::selection::Selection;
use super::*;
use crate::Error;
use kernel_core::precision::{ScaleScope, dtype_bits};
use kernel_core::weight_transform::WeightTransform;

pub(super) struct Piece {
    expert: u32,
    operand: u32,
    row: u64,
    unit: u64,
    record: VariantRecord,
}

fn local_rows(owned: &[Range<u64>], rows: &Range<u64>) -> Result<Range<u64>, Error> {
    let mut base = 0;
    for range in owned {
        if range.start <= rows.start && rows.end <= range.end {
            return Ok(base + rows.start - range.start..base + rows.end - range.start);
        }
        base += range.end - range.start;
    }
    Err(Error(
        "assembly requests rows outside its prepared source ownership".into(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn crop(
    source: &DerivedOperation,
    owned: &Selection,
    rows: &Range<u64>,
    operand: u32,
    expert: u32,
    target_operand: u32,
    target_expert: u32,
    target_row: u64,
) -> Result<Vec<Piece>, Error> {
    if source.variant_records.is_empty() {
        return Ok(Vec::new());
    }
    let local = local_rows(&owned.rows, rows)?;
    let weight = &source.operation.weights[operand as usize];
    let mut result = Vec::new();
    for record in &source.variant_records {
        let section = record.section;
        let total = source
            .packing
            .section_bytes()
            .map_err(|e| Error(e.to_string()))?;
        let bytes = match section {
            PackedSection::Codes => total.codes,
            PackedSection::LocalScales => total.local_scales,
            PackedSection::GlobalScales => total.global_scales,
        };
        let matrix_bytes =
            bytes / u64::from(source.packing.experts) / u64::from(source.packing.projections);
        if matrix_bytes == 0 {
            return Err(Error(
                "prepared source section has no matrix footprint".into(),
            ));
        }
        let matrix = u64::from(expert - owned.experts.start)
            * u64::from(source.packing.projections)
            + u64::from(operand);
        let begin = matrix * matrix_bytes;
        let end = begin + matrix_bytes;
        let record_bytes = record
            .record
            .output_bytes()
            .map_err(|e| Error(e.to_string()))?;
        if record.view_offset >= end || record.view_offset + record_bytes <= begin {
            continue;
        }
        if record.view_offset < begin || record.view_offset + record_bytes > end {
            return Err(Error(
                "prepared assembly needs separately addressable source matrices".into(),
            ));
        }
        let unit = match section {
            PackedSection::Codes => 1,
            PackedSection::LocalScales => weight
                .local_scale
                .as_ref()
                .and_then(|s| s.groups.iter().find(|g| g.axis == 0))
                .map_or(u64::MAX, |g| u64::from(g.size)),
            PackedSection::GlobalScales => u64::MAX,
        };
        let mut record = record.clone();
        if unit == u64::MAX {
            if local != (0..source.logical_geometry.rows) {
                return Err(Error(
                    "partial prepared assembly cannot reinterpret scalar companions".into(),
                ));
            }
            result.push(Piece {
                expert: target_expert,
                operand: target_operand,
                row: 0,
                unit: 1,
                record,
            });
            continue;
        }
        if local.start % unit != 0 || local.end % unit != 0 {
            return Err(Error("prepared assembly window cuts a scale group".into()));
        }
        let shape = &record.record.stored_shape;
        if shape.len() != 2
            || record.record.axis != Some(0)
            || !matches!(
                record.record.transform,
                WeightTransform::Copy
                    | WeightTransform::CopyScales
                    | WeightTransform::Fragment(_)
                    | WeightTransform::Mxfp4Moe(_)
                    | WeightTransform::WidenBf16F32
            )
        {
            return Err(Error(
                "prepared assembly needs an existing row-gather record".into(),
            ));
        }
        let row_bytes = record_bytes / shape[0];
        let offset = (record.view_offset - begin) / row_bytes;
        let requested = local.start / unit..local.end / unit;
        let first = requested.start.max(offset);
        let last = requested.end.min(offset + shape[0]);
        if first >= last {
            continue;
        }
        record.record.ranges = rank_rows(&record.record.ranges, first - offset..last - offset)?;
        record.record.stored_shape[0] = last - first;
        record.view_offset = 0;
        result.push(Piece {
            expert: target_expert,
            operand: target_operand,
            row: target_row + (first - requested.start) * unit,
            unit,
            record,
        });
    }
    Ok(result)
}

pub(super) fn effective(
    imported: &mut ImportedProjection,
    source: &ImportedProjection,
    selections: &[Selection],
    topology: Topology,
) -> Result<(), Error> {
    if selections
        .iter()
        .any(|s| s.physical != selections[0].physical)
    {
        return Err(Error(
            "prepared assembly operands have different geometry".into(),
        ));
    }
    let padding = kernel_core::precision::format::registry::by_id(source.format)
        .map_err(|e| Error(e.to_string()))?
        .padding_tile(imported.family);
    let geometry = MatrixGeometry {
        rows: super::selection::pad(selections[0].physical.rows, padding.rows as u64)?,
        k: super::selection::pad(selections[0].physical.k, padding.k as u64)?,
    };
    let mut weights = Vec::new();
    for operand in 0..selections.len() {
        let mut weight = source
            .weights
            .get(operand)
            .unwrap_or(&source.weights[0])
            .clone();
        weight.logical_shape = vec![geometry.rows, geometry.k];
        weight.codes.tensor = format!(
            "{}.assembled.{operand}.codes@tp{}-{}",
            imported.operation_id, topology.ranks, topology.rank
        );
        weight.codes.shape = vec![geometry.rows, geometry.k];
        weight.codes.offset = 0;
        weight.codes.byte_length = geometry.rows * geometry.k * dtype_bits(weight.codes.dtype) / 8;
        if let Some(local) = &mut weight.local_scale {
            if local.scope != ScaleScope::Block {
                return Err(Error(
                    "prepared assembly requires explicit block companion geometry".into(),
                ));
            }
            let group = |axis| {
                local
                    .groups
                    .iter()
                    .find(|g| g.axis == axis)
                    .map(|g| u64::from(g.size))
                    .ok_or_else(|| Error("prepared scale axis is missing".into()))
            };
            local.tensor.shape = vec![
                geometry.rows.div_ceil(group(0)?),
                geometry.k.div_ceil(group(1)?),
            ];
            local.tensor.tensor = format!(
                "{}.assembled.{operand}.local@tp{}-{}",
                imported.operation_id, topology.ranks, topology.rank
            );
            local.tensor.offset = 0;
            local.tensor.byte_length =
                local.tensor.shape.iter().product::<u64>() * dtype_bits(local.tensor.dtype) / 8;
            for g in &mut local.groups {
                g.origin = 0;
            }
        }
        weight.validate().map_err(|e| Error(e.to_string()))?;
        weights.push(weight);
    }
    imported.format = source.format;
    imported.weights = weights;
    imported.a16_recipe = source.a16_recipe;
    imported.arithmetic.decode_order = source.arithmetic.decode_order;
    imported.arithmetic.weight_operand = source.arithmetic.weight_operand;
    imported.preparation = Some(SourcePreparation {
        topology,
        geometry,
        records: vec![],
    });
    Ok(())
}

pub(super) fn place(
    pieces: Vec<Piece>,
    imported: &ImportedProjection,
    selections: &[Selection],
) -> Result<Vec<VariantRecord>, Error> {
    let mut records = Vec::new();
    for piece in pieces {
        let rows = selections[piece.operand as usize].physical.rows;
        let row_bytes = piece
            .record
            .record
            .output_bytes()
            .map_err(|e| Error(e.to_string()))?
            / piece.record.record.stored_shape[0];
        let prefix =
            (u64::from(piece.expert) * selections.len() as u64 + u64::from(piece.operand)) * rows;
        let mut record = piece.record;
        record.view_offset = (prefix + piece.row) / piece.unit * row_bytes;
        records.push(record);
    }
    if imported.preparation.is_some() && records.is_empty() {
        return Err(Error("prepared assembly has no source records".into()));
    }
    Ok(records)
}
