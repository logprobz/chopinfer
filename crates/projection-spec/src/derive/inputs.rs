//! Activation segment views preserve live columns separately from padding.
use super::{DerivedInput, DerivedOperation, InputSegment, ProjectionContext, ViewRole};
use crate::Error;

/// A grouped dense input's segments and padding: local group `j` reads its
/// global group's K columns of each source segment, placed at `j` times the
/// packed K in the view, and the rest of its slot is padding. Every local
/// group's live K is covered exactly once.
fn grouped(
    ctx: &ProjectionContext<'_>,
    operation: &DerivedOperation,
    input: &crate::inputs::ActivationInput,
    operand: usize,
    width: u64,
) -> Result<(Vec<InputSegment>, Vec<std::ops::Range<u64>>), Error> {
    let selected = match ctx.selections {
        Some(selections) => selections
            .get(operand)
            .ok_or_else(|| Error("activation operand has no assembly selection".into()))?
            .clone(),
        None => {
            super::selection::select(ctx.projection, ctx.geometry, ctx.topology, operand as u32)?
        }
    };
    let k = selected.logical.k;
    let packed = u64::from(operation.packing.k);
    let first = u64::from(selected.experts.start);
    let local = u64::from(selected.experts.end) - first;
    if width
        != packed
            .checked_mul(local)
            .ok_or_else(|| Error("activation width overflows".into()))?
    {
        return Err(Error(
            "activation view disagrees with its storage/geometry contract".into(),
        ));
    }
    let mut segments = Vec::new();
    for segment in &input.segments {
        let global = segment.global_range(ctx.declaration, ctx.geometry, ctx.topology)?;
        let origin = segment
            .origin
            .as_ref()
            .map(|d| d.resolve_offset(ctx.geometry))
            .transpose()?
            .unwrap_or(0);
        let source_columns = segment
            .length
            .as_ref()
            .map(|d| d.resolve(ctx.geometry))
            .transpose()?;
        for j in 0..local {
            let start = (first + j) * k;
            let (lo, hi) = (global.start.max(start), global.end.min(start + k));
            if lo >= hi {
                continue;
            }
            segments.push(InputSegment {
                source: segment.source.clone(),
                source_columns,
                columns: lo - origin..hi - origin,
                target_columns: j * packed + lo - start..j * packed + hi - start,
            });
        }
    }
    segments.sort_by_key(|s| s.target_columns.start);
    for pair in segments.windows(2) {
        if pair[0].target_columns.end > pair[1].target_columns.start {
            return Err(Error("activation owner domains overlap".into()));
        }
    }
    for j in 0..local {
        let slot = j * packed..j * packed + k;
        let covered: u64 = segments
            .iter()
            .filter(|s| slot.contains(&s.target_columns.start))
            .map(|s| s.target_columns.end - s.target_columns.start)
            .sum();
        if covered != k {
            return Err(Error(
                "activation segments do not cover live K exactly".into(),
            ));
        }
    }
    let padding = (0..local)
        .filter(|_| k < packed)
        .map(|j| j * packed + k..(j + 1) * packed)
        .collect();
    Ok((segments, padding))
}

pub(super) fn lower(
    ctx: &ProjectionContext<'_>,
    operation: &mut DerivedOperation,
) -> Result<(), Error> {
    for input in &ctx.projection.inputs {
        let operand =
            ctx.projection
                .operand_index(&input.operand)
                .ok_or_else(|| Error("activation operand is missing".into()))? as usize;
        let view = operation
            .views
            .iter()
            .filter(|v| v.role == ViewRole::Input)
            .nth(operand)
            .ok_or_else(|| Error("activation binding has no required input view".into()))?
            .clone();
        let activation = operation
            .operation
            .activations
            .get(operand)
            .ok_or_else(|| Error("activation binding has no storage recipe".into()))?;
        if let crate::ProjectionKind::GroupedDense { .. } = ctx.projection.kind {
            if view.dtype != activation.storage || view.shape.len() != 2 {
                return Err(Error(
                    "activation view disagrees with its storage/geometry contract".into(),
                ));
            }
            let (segments, padding) = grouped(ctx, operation, input, operand, view.shape[1])?;
            operation.inputs.push(DerivedInput {
                operand: operand as u32,
                binding: input.binding.clone(),
                view,
                segments,
                padding,
            });
            continue;
        }
        if view.dtype != activation.storage
            || view.shape.len() != 2
            || view.shape[1] != u64::from(operation.packing.k)
        {
            return Err(Error(
                "activation view disagrees with its storage/geometry contract".into(),
            ));
        }
        let owned_k = match ctx.selections {
            Some(selections) => selections
                .get(operand)
                .ok_or_else(|| Error("activation operand has no assembly selection".into()))?
                .k
                .clone(),
            None => {
                super::selection::select(
                    ctx.projection,
                    ctx.geometry,
                    ctx.topology,
                    operand as u32,
                )?
                .k
            }
        };
        let mut logical_cursor = owned_k.start;
        let domains = input
            .segments
            .iter()
            .any(|s| s.origin.is_some() || s.length.is_some());
        let requested = if domains {
            Some(input.segments[0].requested_range(ctx.declaration, ctx.geometry, ctx.topology)?)
        } else {
            None
        };
        if requested.as_ref().is_some_and(|r| *r != owned_k) {
            return Err(Error(
                "activation selection disagrees with weight K ownership".into(),
            ));
        }
        let mut end = 0u64;
        let mut segments = Vec::new();
        for segment in &input.segments {
            if !domains {
                let logical =
                    segment.requested_range(ctx.declaration, ctx.geometry, ctx.topology)?;
                if logical.start != logical_cursor || logical.end > owned_k.end {
                    return Err(Error(
                        "activation selection disagrees with weight K ownership".into(),
                    ));
                }
                logical_cursor = logical.end;
            }
            let columns = segment.range(ctx.declaration, ctx.geometry, ctx.topology)?;
            if domains
                && (segment.origin.is_none()
                    || segment.length.is_none()
                    || Some(segment.requested_range(
                        ctx.declaration,
                        ctx.geometry,
                        ctx.topology,
                    )?) != requested)
            {
                return Err(Error(
                    "activation owners must declare domains over one logical selection".into(),
                ));
            }
            if columns.is_empty() {
                continue;
            }
            let next = end
                .checked_add(columns.end - columns.start)
                .ok_or_else(|| Error("activation segment extent overflows".into()))?;
            let target_columns = if let Some(requested) = &requested {
                let global = segment.global_range(ctx.declaration, ctx.geometry, ctx.topology)?;
                global.start - requested.start..global.end - requested.start
            } else {
                end..next
            };
            if segments.iter().any(|s: &InputSegment| {
                s.target_columns.start < target_columns.end
                    && target_columns.start < s.target_columns.end
            }) {
                return Err(Error("activation owner domains overlap".into()));
            }
            segments.push(InputSegment {
                source: segment.source.clone(),
                source_columns: segment
                    .length
                    .as_ref()
                    .map(|d| d.resolve(ctx.geometry))
                    .transpose()?,
                columns,
                target_columns,
            });
            end = next;
        }
        segments.sort_by_key(|s| s.target_columns.start);
        if !domains && logical_cursor != owned_k.end {
            return Err(Error(
                "activation selection disagrees with weight K ownership".into(),
            ));
        }
        if end != operation.logical_geometry.k
            || requested.as_ref().is_some_and(|r| r.end - r.start != end)
        {
            return Err(Error(
                "activation segments do not cover live K exactly".into(),
            ));
        }
        operation.inputs.push(DerivedInput {
            operand: operand as u32,
            binding: input.binding.clone(),
            view,
            segments,
            padding: Vec::new(),
        });
    }
    Ok(())
}
