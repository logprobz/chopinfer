//! Activation segment views preserve live columns separately from padding.
use super::{DerivedInput, DerivedOperation, InputSegment, ProjectionContext, ViewRole};
use crate::Error;

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
        });
    }
    Ok(())
}
