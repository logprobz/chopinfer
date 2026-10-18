//! Bind a caller's persistent encoding choice without changing the projection wire.
use super::*;
use kernel_core::contract::WeightFormat;
use kernel_core::precision::format::LayoutVariant;
use kernel_core::precision::{ScaleScope, TensorSpan};
use kernel_core::weight_transform::WeightTransform;
use projection_spec::derive::OperandSlice;
use std::collections::BTreeMap;

/// Model or use policy, selected before persistent view identity is deduplicated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecordPolicy {
    /// One existing PackedOperand record per complete resident section.
    Packed,
    /// Expert-major, operand-major PackedOperand pieces; globals remain copies.
    PackedPieces,
    /// Existing Copy, Fragment or Mxfp4Moe pieces, chosen through the format plug-in.
    Sections(RecordLayout),
    /// A section encoding with an explicit plug-in layout variant.
    SectionsWithLayout {
        records: RecordLayout,
        layout: LayoutVariant,
    },
}

/// Bind the physical view layouts and exact records together. Explicit provider
/// records already own this choice and cannot be replaced by a caller policy.
pub fn bind_record_policy(
    mut operation: DerivedOperation,
    policy: RecordPolicy,
) -> Result<DerivedOperation, CompilerError> {
    if !operation.variant_records.is_empty() || operation.source_preparation.is_some() {
        return Err(error(
            "record policy cannot replace provider-owned preparation or records",
        ));
    }
    // A resident operand in another encoding than its source (a dense 32x32
    // UE8M0 operand's MXFP8 rows) is produced only by the packed operand,
    // which expands it; a section copy would store the source grid.
    let expanded = operation
        .slices
        .iter()
        .any(|s| s.source.weight.format != operation.packing.format);
    if expanded && !matches!(policy, RecordPolicy::Packed | RecordPolicy::PackedPieces) {
        return Err(error(
            "a row-expanded operand is recorded as packed operands, not section copies",
        ));
    }
    if policy == RecordPolicy::Packed {
        operation.variant_records = records(&operation)?;
    } else {
        let pieces = pieces(&operation)?;
        let section_layout = match policy {
            RecordPolicy::Sections(records) => Some((records, None)),
            RecordPolicy::SectionsWithLayout { records, layout } => Some((records, Some(layout))),
            _ => None,
        };
        if let Some((record_layout, explicit)) = section_layout {
            let weight = &pieces[0].source.weight;
            let (family, variant) = match record_layout {
                RecordLayout::Rows => (
                    kernel_core::precision::format::KernelFamily::Row,
                    LayoutVariant::Standard,
                ),
                RecordLayout::Fragment => (
                    kernel_core::precision::format::KernelFamily::Dense,
                    LayoutVariant::Standard,
                ),
                RecordLayout::Mxfp4Strips => (
                    operation.family,
                    LayoutVariant::Mxfp4Strips {
                        fused_gate_up: operation.packing.projections == 2,
                    },
                ),
            };
            let layout = registry::by_id(pieces[0].format)?.layout_for_weight(
                weight,
                family,
                explicit.unwrap_or(variant),
            )?;
            for view in &mut operation.views {
                if let ViewRole::Weight(section) = view.role {
                    view.layout = match section {
                        PackedSection::Codes => layout.codes,
                        PackedSection::LocalScales => layout
                            .local_scales
                            .ok_or_else(|| error("record layout has no local scales"))?,
                        PackedSection::GlobalScales => layout
                            .global_scales
                            .ok_or_else(|| error("record layout has no global scales"))?,
                    }
                    .into();
                }
            }
        }
        operation.variant_records = emit(&operation, &pieces, policy)?;
    }
    check_records(&operation, &operation.variant_records)?;
    // Resolve every source back to its authenticated span before returning.
    persistent::records(&operation)?;
    Ok(operation)
}

/// Append `ranges` to `rows`, joining a range that continues the last one.
fn coalesce(rows: &mut Vec<std::ops::Range<u64>>, ranges: &[std::ops::Range<u64>]) {
    for range in ranges {
        match rows.last_mut() {
            Some(last) if last.end == range.start => last.end = range.end,
            _ => rows.push(range.clone()),
        }
    }
}

/// Whether `slice` fills its whole destination rather than one tile of it.
fn whole_destination(slice: &OperandSlice, operation: &DerivedOperation) -> bool {
    matches!(&slice.target_rows[..], [rows] if *rows == (0..operation.logical_geometry.rows))
}

/// An operation's authenticated whole source tensors by name, in record
/// source order: expert banks read hundreds of them.
struct Wholes<'a>(BTreeMap<&'a str, Vec<&'a TensorSpan>>);

impl<'a> Wholes<'a> {
    fn of(operation: &'a DerivedOperation) -> Self {
        let mut wholes = BTreeMap::<_, Vec<_>>::new();
        for whole in operation.record_sources.iter().filter(|w| w.offset == 0) {
            wholes.entry(whole.tensor.as_str()).or_default().push(whole);
        }
        Self(wholes)
    }

    /// The authenticated whole tensor `span` reads: itself, or the source it
    /// is a byte window of (a part frame such as one row chunk of a tensor).
    fn of_span(&self, span: &TensorSpan) -> Result<&'a TensorSpan, CompilerError> {
        self.0
            .get(span.tensor.as_str())
            .into_iter()
            .flatten()
            .copied()
            .filter(|whole| {
                whole.dtype == span.dtype
                    && span.offset + span.byte_length <= whole.byte_length
                    && (*whole == span || whole.shape[1..] == span.shape[1..])
            })
            .max_by_key(|whole| whole.byte_length)
            .ok_or_else(|| error("piece encoding requires an authenticated whole source tensor"))
    }
}

/// One complete destination matrix per expert and operand. A destination
/// may be tiled in row order by several sources (an assembly's members);
/// consecutive selections of one source gather as one piece.
fn pieces(operation: &DerivedOperation) -> Result<Vec<OperandSlice>, CompilerError> {
    operation.packing.validate()?;
    let mut sorted = operation.slices.clone();
    sorted.sort_by_key(|s| (s.expert, s.operand, s.target_rows.first().map(|r| r.start)));
    let mut pieces: Vec<OperandSlice> = Vec::new();
    for slice in sorted {
        if let Some(last) = pieces.last_mut()
            && (last.expert, last.operand) == (slice.expert, slice.operand)
            && last.source.weight == slice.source.weight
            && last.k == slice.k
        {
            coalesce(&mut last.rows, &slice.rows);
            coalesce(&mut last.target_rows, &slice.target_rows);
            continue;
        }
        let mut merged = slice.clone();
        merged.rows.clear();
        merged.target_rows.clear();
        coalesce(&mut merged.rows, &slice.rows);
        coalesce(&mut merged.target_rows, &slice.target_rows);
        pieces.push(merged);
    }
    let wholes = Wholes::of(operation);
    let incomplete = || error("piece encoding needs complete, ordered destination matrices");
    let mut destination = None;
    let mut cursor = 0;
    let mut matrices = 0u64;
    for slice in &pieces {
        if destination != Some((slice.expert, slice.operand)) {
            if destination.is_some() && cursor != operation.logical_geometry.rows {
                return Err(incomplete());
            }
            if u64::from(slice.expert) * u64::from(operation.packing.projections)
                + u64::from(slice.operand)
                != matrices
            {
                return Err(incomplete());
            }
            destination = Some((slice.expert, slice.operand));
            matrices += 1;
            cursor = 0;
        }
        for target in &slice.target_rows {
            if target.start != cursor {
                return Err(incomplete());
            }
            cursor = target.end;
        }
        let tile = !whole_destination(slice, operation);
        if tile && slice.geometry.rows != operation.logical_geometry.rows {
            return Err(error("a tiled destination cannot carry padding rows"));
        }
        if registry::by_id(slice.format)?
            .resident_format(slice.source.weight.format, operation.family)
            != operation.packing.format
            || slice.geometry.rows != u64::from(operation.packing.rows)
            || slice.geometry.k != u64::from(operation.packing.k)
            || slice.target_k != (0..operation.logical_geometry.k)
        {
            return Err(incomplete());
        }
        let weight = &slice.source.weight;
        registry::by_id(slice.format)?.companions(weight)?;
        for span in std::iter::once(&weight.codes)
            .chain(weight.local_scale.iter().map(|s| &s.tensor))
            .chain(weight.global_scale.iter().map(|s| &s.tensor))
        {
            wholes.of_span(span)?;
        }
    }
    if cursor != operation.logical_geometry.rows
        || matrices
            != u64::from(operation.packing.experts) * u64::from(operation.packing.projections)
    {
        return Err(error(
            "piece encoding needs one complete source matrix per expert/operand",
        ));
    }
    Ok(pieces)
}

fn emit(
    operation: &DerivedOperation,
    pieces: &[OperandSlice],
    policy: RecordPolicy,
) -> Result<Vec<VariantRecord>, CompilerError> {
    let wholes = Wholes::of(operation);
    let mut result = Vec::new();
    let mut cursors = BTreeMap::new();
    let mut scalar_sources: BTreeMap<u8, TensorSpan> = BTreeMap::new();
    for slice in pieces {
        let weight = &slice.source.weight;
        let plugin = registry::by_id(slice.format)?;
        let mut records = if policy == RecordPolicy::PackedPieces {
            let mut plan = operation.packing.clone();
            plan.experts = 1;
            plan.projections = 1;
            let mut matrix =
                format::matrix_source(plugin, weight, slice.rows.clone(), slice.k.clone())?;
            matrix.source_expert = slice.source.source_expert;
            format::packed_records(
                plugin,
                plan,
                operation.operation.arithmetic.decode_order,
                vec![matrix],
                format::sources(weight),
            )?
            .into_iter()
            .filter_map(|r| {
                let WeightTransform::PackedOperand(ref p) = r.transform else {
                    unreachable!("packed_records emits PackedOperand")
                };
                (p.section != PackedSection::GlobalScales).then_some((
                    p.section,
                    r,
                    std::iter::once(&weight.codes)
                        .chain(weight.local_scale.iter().map(|s| &s.tensor))
                        .chain(weight.global_scale.iter().map(|s| &s.tensor))
                        .map(|s| s.tensor.clone())
                        .collect(),
                ))
            })
            .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        for (section, span) in [
            (PackedSection::Codes, Some(&weight.codes)),
            (
                PackedSection::LocalScales,
                weight.local_scale.as_ref().map(|s| &s.tensor),
            ),
            (
                PackedSection::GlobalScales,
                weight.global_scale.as_ref().map(|s| &s.tensor),
            ),
        ] {
            let Some(span) = span else {
                continue;
            };
            if policy == RecordPolicy::PackedPieces && section != PackedSection::GlobalScales {
                continue;
            }
            let view = operation
                .views
                .iter()
                .find(|v| v.role == ViewRole::Weight(section))
                .ok_or_else(|| error("piece has no resident section view"))?;
            let scalar = section == PackedSection::GlobalScales
                && !operation.packing.per_expert_global()
                || section == PackedSection::LocalScales
                    && weight.format == WeightFormat::Fp8PerTensor;
            if scalar {
                if let Some(previous) = scalar_sources.get(&section_index(section)) {
                    if previous != span {
                        return Err(error(
                            "one scalar view requires the same authenticated source, not guessed uniform values",
                        ));
                    }
                    continue;
                }
                scalar_sources.insert(section_index(section), span.clone());
            }
            // A tile of the destination stores its own rows; a whole
            // destination keeps its padded geometry.
            let tile_rows = if whole_destination(slice, operation) {
                slice.geometry.rows
            } else {
                slice.target_rows.iter().map(|r| r.end - r.start).sum()
            };
            let mut base = section_base(slice, section, span, wholes.of_span(span)?, tile_rows)?;
            if policy == RecordPolicy::PackedPieces
                && section == PackedSection::GlobalScales
                && span.shape.iter().product::<u64>() == 1
            {
                // The existing piece loader gathers scalar headers as one row.
                // Keep the raw source rank, which remains authenticated.
                base.axis = Some(0);
                base.ranges = std::iter::once(0..1).collect();
                base.stored_shape = vec![1];
            }
            let layout = match policy {
                RecordPolicy::Sections(layout)
                | RecordPolicy::SectionsWithLayout {
                    records: layout, ..
                } => layout,
                RecordPolicy::PackedPieces => RecordLayout::Rows,
                RecordPolicy::Packed => unreachable!("Packed is bound separately"),
            };
            let launch_dtype = if section == PackedSection::Codes {
                span.dtype
            } else {
                view.dtype
            };
            let record =
                format::section_record(plugin, section, span.dtype, launch_dtype, layout, base)?;
            records.push((section, record, vec![span.tensor.clone()]));
        }
        for (section, record, tensors) in records {
            let view = operation
                .views
                .iter()
                .find(|v| v.role == ViewRole::Weight(section))
                .ok_or_else(|| error("record section has no view"))?;
            let cursor = cursors.entry(section_index(section)).or_insert(0u64);
            let offset = *cursor;
            *cursor = cursor
                .checked_add(record.output_bytes().map_err(|e| error(&e.to_string()))?)
                .ok_or_else(|| error("piece placement overflows"))?;
            result.push(VariantRecord {
                view: view.id.clone(),
                view_offset: offset,
                section,
                record,
                tensors,
            });
        }
    }
    // Resident section order, then the served expert/operand piece order.
    let order: BTreeMap<_, _> = operation
        .views
        .iter()
        .enumerate()
        .map(|(i, v)| (v.id.as_str(), i))
        .collect();
    result.sort_by_key(|r| (order[r.view.as_str()], r.view_offset));
    Ok(result)
}

fn section_base(
    slice: &OperandSlice,
    section: PackedSection,
    span: &TensorSpan,
    whole: &TensorSpan,
    tile_rows: u64,
) -> Result<WeightRecord, CompilerError> {
    let weight = &slice.source.weight;
    let mut source = format::sources(weight)
        .into_iter()
        .find(|s| {
            s.shape == span.shape
                && s.byte_length == span.byte_length
                && (span.dtype.checkpoint_dtype() == Some(s.dtype.as_str())
                    || s.dtype == "U8" && span.dtype == kernel_core::contract::DType::Ue8M0)
        })
        .ok_or_else(|| error("section source metadata disagrees"))?;
    // A part frame gathers from its whole source, from the part's first row.
    let first_row = if whole == span {
        0
    } else {
        source.shape = whole.shape.clone();
        source.byte_length = whole.byte_length;
        let row_bytes = whole.byte_length / whole.shape.first().copied().unwrap_or(1).max(1);
        if row_bytes == 0 || span.offset % row_bytes != 0 {
            return Err(error("a part frame does not start on a source row"));
        }
        span.offset / row_bytes
    };
    let mut base = WeightRecord {
        transform: WeightTransform::Copy,
        sources: vec![source],
        axis: None,
        ranges: vec![],
        stored_shape: span.shape.clone(),
    };
    if section == PackedSection::GlobalScales {
        if span.shape.iter().product::<u64>() != 1 {
            if weight.global_scale.as_ref().unwrap().scope != ScaleScope::Expert {
                return Err(error(
                    "global copy requires a scalar or an expert-owned scalar",
                ));
            }
            base.axis = Some(0);
            base.ranges =
                std::iter::once(slice.source.source_expert..slice.source.source_expert + 1)
                    .collect();
            base.stored_shape[0] = 1;
        }
        return Ok(base);
    }
    if weight.logical_shape.len() != 2 || span.shape.len() > 2 {
        return Err(error(
            "section encoding requires a checkpoint matrix, not a bank or part frame",
        ));
    }
    if section == PackedSection::LocalScales && weight.format == WeightFormat::Fp8PerTensor {
        return Ok(base);
    }
    let (row_group, k_group) = if section == PackedSection::Codes {
        (
            1,
            if matches!(weight.format, WeightFormat::Mxfp4 | WeightFormat::Nvfp4) {
                2
            } else {
                1
            },
        )
    } else {
        let scale = weight.local_scale.as_ref().unwrap();
        if scale.groups.iter().any(|g| g.origin != 0) {
            return Err(error(
                "section encoding needs an aligned, non-restarting scale grid",
            ));
        }
        let group = |axis| {
            scale
                .groups
                .iter()
                .find(|g| g.axis == axis)
                .map_or(u64::MAX, |g| u64::from(g.size))
        };
        (group(0), group(1))
    };
    let grouped = |r: &std::ops::Range<u64>,
                   unit: u64,
                   total: u64|
     -> Result<std::ops::Range<u64>, CompilerError> {
        if r.start % unit != 0 || r.end % unit != 0 && r.end != total {
            return Err(error(
                "section selection cuts a packed element or scale group",
            ));
        }
        Ok(r.start / unit..r.end.div_ceil(unit))
    };
    let k = grouped(&slice.k, k_group, weight.logical_shape[1])?;
    let mut rows = Vec::new();
    for r in &slice.rows {
        let r = grouped(r, row_group, weight.logical_shape[0])?;
        coalesce(
            &mut rows,
            std::slice::from_ref(&(r.start + first_row..r.end + first_row)),
        );
    }
    let full_k = 0..span.shape.last().copied().unwrap_or(1);
    let full_rows = std::iter::once(0..whole.shape[0]).collect::<Vec<_>>();
    let target = if span.shape.len() == 1 {
        vec![tile_rows.div_ceil(row_group)]
    } else {
        vec![
            tile_rows.div_ceil(row_group),
            slice.geometry.k.div_ceil(k_group),
        ]
    };
    if k == full_k || span.shape.len() == 1 {
        base.axis = Some(0);
        base.ranges = rows;
    } else if rows == full_rows {
        base.axis = Some(1);
        base.ranges = std::iter::once(k).collect();
    } else {
        return Err(error(
            "section encoding cannot gather rows and K in one host record",
        ));
    }
    base.stored_shape = target;
    Ok(base)
}

fn section_index(section: PackedSection) -> u8 {
    match section {
        PackedSection::Codes => 0,
        PackedSection::LocalScales => 1,
        PackedSection::GlobalScales => 2,
    }
}
