//! Authenticate matrix roles and placements, including declared source reuse.
use super::*;
use crate::Error;
use crate::bind::{BoundDeclaration, BoundOperand};

fn direct(p: &Projection, b: &BoundOperand, g: &Geometry) -> Result<Vec<SourceMatrix>, Error> {
    let rows = p.shape[0].resolve(g)?;
    let k = p.shape[1].resolve(g)?;
    let shards = p
        .lookup
        .as_ref()
        .map(|s| s.shards.resolve(g))
        .transpose()?
        .unwrap_or_default();
    let shard = b
        .shard
        .map(|i| {
            shards
                .get(i as usize)
                .ok_or_else(|| Error("binding has no declared shard".into()))
        })
        .transpose()?;
    let row_origin = shard
        .map(|s| s.rows.start.resolve_offset(g))
        .transpose()?
        .unwrap_or(0);
    let frame_rows = shard
        .map(|s| s.rows.length.resolve(g))
        .transpose()?
        .unwrap_or(rows);
    let source_experts = if b.imported.weight.logical_shape.len() == 3 {
        b.imported.weight.logical_shape[0]
    } else {
        1
    };
    if let crate::ProjectionKind::GroupedDense { groups } = &p.kind {
        return grouped_dense(b, groups.resolve(g)?, rows, k);
    }
    let operands = if b.operand.is_some() {
        1
    } else {
        p.operand_count() as u64
    };
    let mut result = Vec::new();
    if b.imported.weight.validate().is_ok() {
        for expert in 0..source_experts {
            for operand in 0..operands {
                let mut physical = operand * frame_rows;
                let mut logical = row_origin;
                let ranges = if p.parts.is_empty() {
                    vec![(row_origin, frame_rows)]
                } else {
                    p.parts
                        .iter()
                        .map(|part| {
                            let length = part.rows.resolve(g)?;
                            let start = part
                                .range
                                .as_ref()
                                .map(|r| r.start.resolve_offset(g))
                                .transpose()?
                                .unwrap_or(logical);
                            logical = start
                                .checked_add(length)
                                .ok_or_else(|| Error("part placement overflows".into()))?;
                            Ok((start, length))
                        })
                        .collect::<Result<Vec<_>, Error>>()?
                };
                for (start, length) in ranges {
                    result.push(SourceMatrix {
                        weight: b.imported.weight.clone(),
                        expert: u32::try_from(b.expert.unwrap_or(expert))
                            .map_err(|_| Error("expert index exceeds u32".into()))?,
                        operand: u32::try_from(b.operand.unwrap_or(operand))
                            .map_err(|_| Error("operand index exceeds u32".into()))?,
                        source_expert: expert,
                        source_rows: physical..physical + length,
                        target_rows: start..start + length,
                        source_k: 0..k,
                        target_k: 0..k,
                    });
                    physical += length;
                }
            }
        }
    }
    let chunks = p.parts.len().max(1) as u64;
    for (index, weight) in b.imported.parts.iter().enumerate() {
        let index = index as u64;
        let part_index = (index % chunks) as usize;
        let part = p.parts.get(part_index);
        let length = part
            .map(|p| p.rows.resolve(g))
            .transpose()?
            .unwrap_or(frame_rows);
        let prefix = p.parts.iter().take(part_index).try_fold(0u64, |n, p| {
            n.checked_add(p.rows.resolve(g)?)
                .ok_or_else(|| Error("part placement overflows".into()))
        })?;
        let start = part
            .and_then(|p| p.range.as_ref())
            .map(|r| r.start.resolve_offset(g))
            .transpose()?
            .unwrap_or(row_origin + prefix);
        let matrix = SourceMatrix {
            weight: weight.clone(),
            expert: u32::try_from(b.expert.unwrap_or(index / (chunks * operands)))
                .map_err(|_| Error("expert index exceeds u32".into()))?,
            operand: u32::try_from(b.operand.unwrap_or((index / chunks) % operands))
                .map_err(|_| Error("operand index exceeds u32".into()))?,
            source_expert: 0,
            source_rows: 0..length,
            target_rows: start..start + length,
            source_k: 0..k,
            target_k: 0..k,
        };
        if !result.contains(&matrix) {
            result.push(matrix);
        }
    }
    Ok(result)
}

/// One source matrix per group of a grouped dense projection: group `g` is
/// rows `g * rows..(g + 1) * rows` of the stacked source, in place.
pub fn grouped_dense(
    b: &BoundOperand,
    groups: u64,
    rows: u64,
    k: u64,
) -> Result<Vec<SourceMatrix>, Error> {
    if groups
        .checked_mul(rows)
        .is_none_or(|stacked| b.imported.weight.logical_shape != [stacked, k])
        || b.expert.is_some()
        || b.operand.is_some_and(|o| o != 0)
        || !b.imported.parts.is_empty()
    {
        return Err(Error(
            "a grouped dense source is one [groups * rows, K] matrix".into(),
        ));
    }
    (0..groups)
        .map(|group| {
            Ok(SourceMatrix {
                weight: b.imported.weight.clone(),
                expert: u32::try_from(group).map_err(|_| Error("group exceeds u32".into()))?,
                operand: 0,
                source_expert: 0,
                source_rows: group * rows..(group + 1) * rows,
                target_rows: 0..rows,
                source_k: 0..k,
                target_k: 0..k,
            })
        })
        .collect()
}

pub(super) fn check(
    declaration: &Declaration,
    bound: &BoundDeclaration,
    p: &Projection,
    g: &Geometry,
    layer: Option<u64>,
    topology: Topology,
    imported: &ImportedProjection,
) -> Result<(), Error> {
    let mut expected = Vec::new();
    if p.assembly.is_empty() {
        let instance = Some((p.id.as_str(), layer));
        for b in super::index::operands(bound, instance, &Default::default()) {
            expected.extend(direct(p, b, g)?);
        }
    } else {
        // Reuse the assembler with authenticated source frames. This checks its
        // placement rules without trusting the caller's assembled coordinates.
        let mut sources = ImportedProjections::new();
        for member in &p.assembly {
            let source = declaration
                .projections
                .iter()
                .find(|s| s.id == member.projection)
                .ok_or_else(|| Error("assembly source role is missing".into()))?;
            let source_layer = member
                .layer
                .as_ref()
                .map(|l| l.resolve(g, layer))
                .transpose()?;
            let key = InstanceKey {
                projection: source.id.clone(),
                layer: source_layer,
            };
            if sources.contains_key(&key) {
                continue;
            }
            let instance = Some((source.id.as_str(), source_layer));
            let operands = super::index::operands(bound, instance, &Default::default());
            let first = operands
                .first()
                .ok_or_else(|| Error("assembly source has no binding".into()))?;
            let mut prototype = imported.clone();
            prototype.format = first.format;
            prototype.weights =
                vec![first.imported.weight.clone(); source.operand_count() as usize];
            prototype.matrices = Vec::new();
            prototype.preparation = None;
            for b in operands {
                // Aggregate and part mappings are alternative authenticated views,
                // not two contributions to the same assembly row coverage.
                let parts = imported
                    .matrices
                    .iter()
                    .any(|m| b.imported.parts.contains(&m.weight));
                prototype
                    .matrices
                    .extend(direct(source, b, g)?.into_iter().filter(|m| {
                        if parts {
                            b.imported.parts.contains(&m.weight)
                        } else {
                            m.weight == b.imported.weight
                        }
                    }));
            }
            sources.insert(key, prototype);
        }
        let registry = Registry::default();
        let input = DerivationInput {
            declaration,
            bound,
            geometry: g,
            registry: &registry,
            topology,
            imported: &sources,
            variants: &hooks::NoVariants,
        };
        expected = super::assembly::prepare(&input, p, layer, imported)?
            .ok_or_else(|| Error("missing assembled placement".into()))?
            .imported
            .matrices;
    }
    let mut by_role = BTreeMap::<(u32, u32, u64), Vec<&SourceMatrix>>::new();
    for admitted in &expected {
        by_role
            .entry((admitted.operand, admitted.expert, admitted.source_expert))
            .or_default()
            .push(admitted);
    }
    for matrix in &imported.matrices {
        let mut covered = 0u64;
        let candidates = by_role.get(&(matrix.operand, matrix.expert, matrix.source_expert));
        for admitted in candidates
            .into_iter()
            .flatten()
            .filter(|m| m.weight == matrix.weight)
        {
            let rows = matrix.source_rows.start.max(admitted.source_rows.start)
                ..matrix.source_rows.end.min(admitted.source_rows.end);
            let k = matrix.source_k.start.max(admitted.source_k.start)
                ..matrix.source_k.end.min(admitted.source_k.end);
            if rows.is_empty() || k.is_empty() {
                continue;
            }
            if matrix.target_rows.start + rows.start - matrix.source_rows.start
                != admitted.target_rows.start + rows.start - admitted.source_rows.start
                || matrix.target_k.start + k.start - matrix.source_k.start
                    != admitted.target_k.start + k.start - admitted.source_k.start
            {
                continue;
            }
            covered = covered
                .checked_add(
                    (rows.end - rows.start)
                        .checked_mul(k.end - k.start)
                        .ok_or_else(|| Error("matrix placement area overflows".into()))?,
                )
                .ok_or_else(|| Error("matrix placement coverage overflows".into()))?;
        }
        if covered
            != (matrix.source_rows.end - matrix.source_rows.start)
                .checked_mul(matrix.source_k.end - matrix.source_k.start)
                .ok_or_else(|| Error("matrix placement area overflows".into()))?
        {
            return Err(Error(format!(
                "source `{}` disagrees with its binding placement (operand {}, expert {})",
                matrix.weight.codes.tensor, matrix.operand, matrix.expert
            )));
        }
    }
    Ok(())
}
