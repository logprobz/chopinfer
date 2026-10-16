//! Compose declared source selections in logical or post-TP row coordinates.
use super::selection::{self, Selection};
use super::*;
use crate::contracts::RowCoordinates;
use crate::{Dim, Error, ProjectionKind, SplitClass};

fn intersection(a: &Range<u64>, b: &Range<u64>) -> Option<Range<u64>> {
    let start = a.start.max(b.start);
    let end = a.end.min(b.end);
    (start < end).then_some(start..end)
}
pub(super) fn rank_rows(
    rows: &[Range<u64>],
    requested: Range<u64>,
) -> Result<Vec<Range<u64>>, Error> {
    let mut base = 0;
    let mut result = Vec::new();
    for range in rows {
        let end = base + (range.end - range.start);
        if let Some(kept) = intersection(&(base..end), &requested) {
            result.push(range.start + kept.start - base..range.start + kept.end - base);
        }
        base = end;
    }
    if requested.end > base {
        return Err(Error(
            "rank-local assembly rows exceed source ownership".into(),
        ));
    }
    Ok(result)
}
fn inherited_axes(p: &Projection) -> (bool, bool, bool) {
    match p.split {
        SplitClass::Row { .. }
        | SplitClass::ExpertSharded {
            axis: crate::SliceAxis::K,
            ..
        } => (false, true, false),
        SplitClass::ExpertOwned { .. } => (false, false, true),
        SplitClass::Replicate {} => (false, false, false),
        _ => (true, false, false),
    }
}
fn experts(p: &Projection, g: &Geometry) -> Result<u64, Error> {
    match &p.kind {
        ProjectionKind::ExpertBank { experts, .. } => experts.resolve(g),
        _ => Ok(1),
    }
}
fn tile(p: &Projection, g: &Geometry, rows: bool) -> Result<u64, Error> {
    match &p.split {
        SplitClass::Column { tile }
        | SplitClass::HeadGrouped { tile, .. }
        | SplitClass::SegmentedHeads { tile, .. }
        | SplitClass::VocabSharded { tile }
            if rows =>
        {
            tile.resolve(g)
        }
        SplitClass::Row { tile } if !rows => tile.resolve(g),
        SplitClass::ExpertSharded { axis, tile } if rows == (*axis == crate::SliceAxis::Rows) => {
            tile.resolve(g)
        }
        _ => Ok(1),
    }
}

pub(super) struct Prepared {
    pub imported: ImportedProjection,
    pub selections: Vec<Selection>,
    pub records: Vec<VariantRecord>,
    pub stages: Vec<DerivedStage>,
}

pub(super) fn prepare(
    input: &DerivationInput<'_>,
    p: &Projection,
    layer: Option<u64>,
    prototype: &ImportedProjection,
) -> Result<Option<Prepared>, Error> {
    let mut raw = prototype.clone();
    raw.preparation = None;
    prepare_with_sources(input, p, layer, &raw, &std::collections::BTreeMap::new())
}

pub(super) fn prepare_with_sources(
    input: &DerivationInput<'_>,
    p: &Projection,
    layer: Option<u64>,
    prototype: &ImportedProjection,
    sources: &std::collections::BTreeMap<InstanceKey, super::operation::SourceNode>,
) -> Result<Option<Prepared>, Error> {
    if p.assembly.is_empty() {
        return Ok(None);
    }
    let bound = input
        .bound
        .assemblies
        .iter()
        .find(|a| a.projection == p.id && a.layer == layer)
        .ok_or_else(|| Error("assembly has no authenticated binding".into()))?;
    let g = input.geometry;
    let declared_rows = p.shape[0].resolve(g)?;
    let operand_count = p.operand_count() as usize;
    let mut cursors = vec![0u64; operand_count];
    let mut widths = vec![None; operand_count];
    let mut inherited = (false, false, false);
    let mut matrices = Vec::new();
    let mut pending_records = Vec::new();
    let mut stages = Vec::new();
    let mut prepared_source = None;
    let mut global_cursor = 0u64;
    let mut expert_count = experts(p, g)?;
    for (member, names) in bound.members.iter().zip(&bound.tensors) {
        let source = input
            .declaration
            .projections
            .iter()
            .find(|s| s.id == member.projection)
            .ok_or_else(|| Error("assembly source role is missing".into()))?;
        let source_layer = member
            .layer
            .as_ref()
            .map(|s| s.resolve(g, layer))
            .transpose()?;
        let source_key = InstanceKey {
            projection: source.id.clone(),
            layer: source_layer,
        };
        let node = sources.get(&source_key);
        let imported = if let Some(node) = node {
            for stage in &node.derived.stages {
                if !stages.contains(stage) {
                    stages.push(stage.clone());
                }
            }
            if node.imported.preparation.is_some() {
                if prepared_source
                    .as_ref()
                    .is_some_and(|i: &ImportedProjection| i.format != node.imported.format)
                {
                    return Err(Error(
                        "prepared assembly sources have incompatible effective formats".into(),
                    ));
                }
                prepared_source = Some(node.imported.clone());
            }
            &node.imported
        } else {
            let imported = input
                .imported
                .get(&source_key)
                .ok_or_else(|| Error(format!("assembly source `{}` has no import", source.id)))?;
            super::operation::import_admitted(input, source, source_layer, imported)?;
            imported
        };
        let start = member.rows.start.resolve_offset(g)?;
        let end = start
            .checked_add(member.rows.length.resolve(g)?)
            .ok_or_else(|| Error("assembly row extent overflows".into()))?;
        let operands = member
            .operand
            .as_ref()
            .map(|name| {
                source
                    .operand_index(name)
                    .map(|n| vec![n as u32])
                    .ok_or_else(|| Error("unknown assembly operand".into()))
            })
            .transpose()?
            .unwrap_or_else(|| (0..source.operand_count()).collect());
        for operand in operands {
            // A declared member may not silently cross destination operands.
            // Split at the destination boundary while retaining source order.
            let mut requested_start = start;
            while requested_start < end {
                let target_operand = (global_cursor / declared_rows) as usize;
                if target_operand >= operand_count {
                    return Err(Error("assembly exceeds destination operands".into()));
                }
                let length =
                    (end - requested_start).min(declared_rows - global_cursor % declared_rows);
                let requested = requested_start..requested_start + length;
                let owned = selection::select(source, g, input.topology, operand)?;
                let selected = if member.coordinates == RowCoordinates::RankLocal {
                    rank_rows(&owned.rows, requested.clone())?
                } else if member.inherit_split {
                    owned
                        .rows
                        .iter()
                        .filter_map(|r| intersection(r, &requested))
                        .collect()
                } else {
                    vec![requested]
                };
                let source_k = if member.inherit_split {
                    owned.k.clone()
                } else {
                    0..source.shape[1].resolve(g)?
                };
                let width = source_k.end - source_k.start;
                if widths[target_operand].is_some_and(|w| w != width) {
                    return Err(Error(
                        "assembly members have incompatible inherited K ownership".into(),
                    ));
                }
                widths[target_operand] = Some(width);
                let axes = if member.inherit_split {
                    inherited_axes(source)
                } else {
                    (false, false, false)
                };
                // Rank-local rows are already selected, independently of K/expert inheritance.
                inherited.0 |= axes.0 || member.coordinates == RowCoordinates::RankLocal;
                inherited.1 |= axes.1;
                inherited.2 |= axes.2;
                let source_experts = if axes.2 {
                    owned.experts.clone()
                } else {
                    0..u32::try_from(experts(source, g)?)
                        .map_err(|_| Error("assembly experts exceed u32".into()))?
                };
                let n = source_experts.end - source_experts.start;
                if axes.2 {
                    expert_count = n.into();
                }
                if u64::from(n) != expert_count {
                    return Err(Error(
                        "assembly members have incompatible expert ownership".into(),
                    ));
                }
                for range in selected {
                    let target_start = cursors[target_operand];
                    let target_end = target_start
                        .checked_add(range.end - range.start)
                        .ok_or_else(|| Error("assembly target rows overflow".into()))?;
                    for expert in source_experts.clone() {
                        if let Some(node) = node {
                            pending_records.extend(super::prepared::crop(
                                &node.derived,
                                &owned,
                                &range,
                                operand,
                                expert,
                                target_operand as u32,
                                expert - source_experts.start,
                                target_start,
                            )?);
                        }
                        let mut covered = 0u64;
                        for matrix in imported
                            .matrices
                            .iter()
                            .filter(|m| m.operand == operand && m.expert == expert)
                        {
                            if !names.contains(&matrix.weight.codes.tensor) {
                                return Err(Error(
                                    "assembly import is outside its authenticated member".into(),
                                ));
                            }
                            let Some(rows) = intersection(&matrix.target_rows, &range) else {
                                continue;
                            };
                            let Some(k) = intersection(&matrix.target_k, &source_k) else {
                                continue;
                            };
                            covered = covered
                                .checked_add(
                                    (rows.end - rows.start)
                                        .checked_mul(k.end - k.start)
                                        .ok_or_else(|| {
                                            Error("assembly coverage overflows".into())
                                        })?,
                                )
                                .ok_or_else(|| Error("assembly coverage overflows".into()))?;
                            matrices.push(SourceMatrix {
                                weight: matrix.weight.clone(),
                                expert: expert - source_experts.start,
                                operand: target_operand as u32,
                                source_expert: matrix.source_expert,
                                target_rows: target_start + rows.start - range.start
                                    ..target_start + rows.end - range.start,
                                source_rows: matrix.source_rows.start + rows.start
                                    - matrix.target_rows.start
                                    ..matrix.source_rows.start + rows.end
                                        - matrix.target_rows.start,
                                target_k: k.start - source_k.start..k.end - source_k.start,
                                source_k: matrix.source_k.start + k.start - matrix.target_k.start
                                    ..matrix.source_k.start + k.end - matrix.target_k.start,
                            });
                        }
                        if covered != (range.end - range.start) * width {
                            return Err(Error(
                                "assembly source mappings do not cover the declared member".into(),
                            ));
                        }
                    }
                    cursors[target_operand] = target_end;
                }
                requested_start += length;
                global_cursor += length;
            }
        }
    }
    let mut selections = Vec::new();
    for operand in 0..operand_count {
        let rows = cursors[operand];
        let k = widths[operand].ok_or_else(|| Error("assembly operand has no columns".into()))?;
        if rows == 0 || k == 0 {
            return Err(Error("assembly rank owns no live matrix".into()));
        }
        let mut projected = p.clone();
        projected.shape = [Dim::Literal(rows), Dim::Literal(k)];
        projected.parts.clear();
        // Inherited axes have already been selected. Other destination axes
        // retain their own declared TP policy.
        if (inherited.0
            && matches!(
                projected.split,
                SplitClass::Column { .. }
                    | SplitClass::HeadGrouped { .. }
                    | SplitClass::SegmentedHeads { .. }
                    | SplitClass::VocabSharded { .. }
                    | SplitClass::ExpertSharded {
                        axis: crate::SliceAxis::Rows,
                        ..
                    }
            ))
            || (inherited.1
                && matches!(
                    projected.split,
                    SplitClass::Row { .. }
                        | SplitClass::ExpertSharded {
                            axis: crate::SliceAxis::K,
                            ..
                        }
                ))
            || (inherited.2 && matches!(projected.split, SplitClass::ExpertOwned { .. }))
        {
            projected.split = SplitClass::Replicate {};
        }
        if let ProjectionKind::ExpertBank { experts, .. } = &mut projected.kind {
            *experts = Dim::Literal(expert_count);
        }
        let mut selected = selection::select(&projected, g, input.topology, operand as u32)?;
        if inherited.0 {
            selected.physical.rows = selection::pad(selected.logical.rows, tile(p, g, true)?)?;
        }
        if inherited.1 {
            selected.physical.k = selection::pad(selected.logical.k, tile(p, g, false)?)?;
        }
        selections.push(selected);
    }
    let mut imported = prototype.clone();
    imported.matrices = matrices;
    if let Some(source) = prepared_source {
        super::prepared::effective(&mut imported, &source, &selections, input.topology)?;
    }
    let records = super::prepared::place(pending_records, &imported, &selections)?;
    if let Some(preparation) = &mut imported.preparation {
        preparation.records = records.clone();
    }
    Ok(Some(Prepared {
        imported,
        selections,
        records,
        stages,
    }))
}
