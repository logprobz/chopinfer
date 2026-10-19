//! Refuse target bindings that can race a layer's unfinished transfer.
use crate::{dspark::cycle::LayerGather, program::Stage};
use bundle_runtime::{dispatch::DispatchError, pipeline::PreparedLaunch};
use kernel_core::contract::Argument;
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;
use std::ops::Range;

pub(super) fn sections(
    stages: &[Stage],
    layers: impl IntoIterator<Item = u32>,
    partial: bool,
) -> Result<Vec<Range<usize>>, DispatchError> {
    let mut start = 0;
    let mut sections = Vec::new();
    for layer in layers {
        let mut found = stages.iter().enumerate().filter_map(|(i, stage)| {
            matches!(stage, Stage::EngramRows(l) if *l == layer).then_some(i)
        });
        let end = match found.next() {
            Some(end) if end >= start && found.next().is_none() => end,
            None if partial => start,
            _ => return Err(DispatchError::InvalidPlan),
        };
        sections.push(start..end);
        start = end;
    }
    sections.push(start..stages.len());
    Ok(sections)
}

fn disjoint(a: Span, b: Span) -> Result<(), DispatchError> {
    let end = |s: Span| {
        s.offset
            .checked_add(s.length)
            .ok_or(DispatchError::InvalidPlan)
    };
    let (a_end, b_end) = (end(a)?, end(b)?);
    if a.buffer == b.buffer
        && a.length != 0
        && b.length != 0
        && a.offset < b_end
        && b.offset < a_end
    {
        return Err(DispatchError::InvalidPlan);
    }
    Ok(())
}

fn check_launch(op: &PreparedLaunch, pending: &[Span]) -> Result<(), DispatchError> {
    let check = |id| -> Result<(), DispatchError> {
        let bound = op.bindings.get(id)?;
        for &span in pending {
            disjoint(bound, span)?;
        }
        Ok(())
    };
    for arg in &op.arguments.0 {
        match arg {
            Argument::DeviceBuffer(id) => check(*id)?,
            Argument::TensorMap(map) => check(map.base_buffer)?,
            Argument::DevicePointerArray(ids) => {
                for id in ids {
                    check(*id)?;
                }
            }
            _ => (),
        }
    }
    Ok(())
}

pub(super) fn before_join(
    program: &[PreparedLaunch],
    pending: &[LayerGather],
) -> Result<(), DispatchError> {
    let spans = pending
        .iter()
        .flat_map(|layer| std::iter::once(layer.indices).chain(layer.destinations.iter().copied()))
        .collect::<Vec<_>>();
    for op in program {
        check_launch(op, &spans)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::dispatch::BufferBindings;
    use kernel_core::contract::{
        ArgumentList, LaunchPlan, TensorMap, TensorMapElement, TensorMapInterleave,
        TensorMapL2Promotion, TensorMapOobFill, TensorMapSwizzle,
    };
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    #[test]
    fn each_layer_starts_its_own_piece_and_reordering_is_refused() {
        let stages = [Stage::EngramRows(1), Stage::EngramRows(14)];
        assert_eq!(
            sections(&stages, [1, 14], false).unwrap(),
            [0..0, 0..1, 1..2]
        );
        assert!(sections(&stages, [14, 1], false).is_err());
        assert!(sections(&stages, [1, 2], false).is_err());
        assert!(sections(&[Stage::EngramRows(1), Stage::EngramRows(1)], [1], false).is_err());
        assert_eq!(sections(&[], [1, 14], true).unwrap(), [0..0, 0..0, 0..0]);
    }

    #[test]
    fn pending_upload_aliases_and_invalid_bindings_are_refused() {
        let span = |at, len| Span::new(BundleDeviceBufferHandleV1::from_parts(1, 1), at, len);
        let pending = [span(128, 64)];
        for (view, valid) in [
            (span(0, 128), true),
            (span(192, 64), true),
            (span(127, 2), false),
            (span(128, 64), false),
            (span(u64::MAX, 2), false),
        ] {
            let mut bindings = BufferBindings::default();
            let id = bindings.bind(view);
            let mut op = PreparedLaunch {
                symbol: String::new(),
                rows: 1,
                plan: LaunchPlan {
                    grid: [1; 3],
                    block: [32, 1, 1],
                    cluster: [1; 3],
                    dynamic_shared_bytes: 0,
                    workspace_bytes: 0,
                    pdl: true,
                },
                arguments: ArgumentList(vec![Argument::DeviceBuffer(id)]),
                bindings,
            };
            assert_eq!(check_launch(&op, &pending).is_ok(), valid);
            op.arguments = ArgumentList(vec![Argument::DevicePointerArray(vec![id])]);
            assert_eq!(check_launch(&op, &pending).is_ok(), valid);
            op.arguments = ArgumentList(vec![Argument::TensorMap(TensorMap {
                base_buffer: id,
                dims: vec![64, 1],
                strides_bytes: vec![1, 64],
                box_dims: vec![64, 1],
                element_strides: vec![1, 1],
                swizzle: TensorMapSwizzle::None,
                interleave: TensorMapInterleave::None,
                l2_promotion: TensorMapL2Promotion::None,
                oob_fill: TensorMapOobFill::Zero,
                element: TensorMapElement::Fp8E4M3,
            })]);
            assert_eq!(check_launch(&op, &pending).is_ok(), valid);
            op.arguments = ArgumentList(vec![Argument::DeviceBuffer(u64::MAX)]);
            assert!(check_launch(&op, &pending).is_err());
        }
    }
}
