//! Hidden-row copies through the already embedded uint16 layout-copy entry.

use bundle_runtime::dispatch::BufferBindings;
use bundle_runtime::pipeline::PreparedLaunch;
use kernels_sm120::row::copy::SLICE_SYMBOL;
use kernels_sm120::row::precision::{CopyRows, LayoutCopy, RowResources, ShapeBinding};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

use crate::drafter::copies::Run;

pub fn prepare(
    run: Run,
    hidden: u32,
    source: Span,
    destination: Span,
    counts: Span,
) -> Result<PreparedLaunch, String> {
    let stride = |rows: u32| -> Result<CopyRows, String> {
        Ok(CopyRows {
            row_stride: rows
                .checked_mul(hidden)
                .ok_or("hidden copy stride overflows")?,
            column_stride: 1,
            column_offset: 0,
        })
    };
    let copy = LayoutCopy::new(
        hidden,
        run.rows,
        stride(run.source_stride)?,
        stride(run.destination_stride)?,
        SLICE_SYMBOL,
        ShapeBinding::Launch,
        RowResources::SM120F,
    )
    .map_err(|e| e.to_string())?;
    let view = |span: Span, first: u32, bytes: u64| -> Result<Span, String> {
        let offset = u64::from(first)
            .checked_mul(u64::from(hidden) * 2)
            .ok_or("hidden copy offset overflows")?;
        if offset
            .checked_add(bytes)
            .is_none_or(|end| end > span.length)
        {
            return Err("hidden copy exceeds its buffer".into());
        }
        Ok(Span::new(
            span.buffer,
            span.offset
                .checked_add(offset)
                .ok_or("hidden copy view overflows")?,
            bytes,
        ))
    };
    let offset = u64::from(run.rows) * 4;
    if offset + 4 > counts.length {
        return Err("hidden copy exceeds its count table".into());
    }
    let mut bindings = BufferBindings::default();
    let input = bindings.bind(view(source, run.first.staged, copy.source_bytes())?);
    let output = bindings.bind(view(
        destination,
        run.first.retained,
        copy.destination_bytes(),
    )?);
    let live = bindings.bind(Span::new(
        counts.buffer,
        counts
            .offset
            .checked_add(offset)
            .ok_or("hidden copy count offset overflows")?,
        4,
    ));
    Ok(PreparedLaunch {
        symbol: SLICE_SYMBOL.into(),
        rows: run.rows,
        plan: copy.stage().plan(run.rows).map_err(|e| e.to_string())?,
        arguments: copy
            .arguments(input, output, live)
            .map_err(|e| e.to_string())?,
        bindings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::drafter::Retain;
    use kernel_core::contract::Argument;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    #[test]
    fn copy_footprints_and_count_offsets_are_bounded() {
        let span = |id, offset, length| {
            Span::new(
                BundleDeviceBufferHandleV1::from_parts(1, id),
                offset,
                length,
            )
        };
        let run = Run {
            first: Retain {
                phase: 1,
                staged: 0,
                retained: 5,
            },
            rows: 32,
            source_stride: 1,
            destination_stride: 6,
        };
        let source = span(1, 256, 32 * 8192);
        let destination = span(2, 512, 192 * 8192);
        let counts = span(3, 4, 33 * 4);
        let launch = prepare(run, 4096, source, destination, counts).unwrap();
        let Argument::DeviceBuffer(live) = launch.arguments.0[2] else {
            panic!("live rows");
        };
        assert_eq!(launch.bindings.get(live).unwrap(), span(3, 132, 4));
        assert!(
            prepare(
                run,
                4096,
                span(1, 256, source.length - 1),
                destination,
                counts
            )
            .is_err()
        );
        assert!(
            prepare(
                run,
                4096,
                source,
                span(2, 512, destination.length - 1),
                counts
            )
            .is_err()
        );
        assert!(
            prepare(
                run,
                4096,
                source,
                destination,
                span(3, 4, counts.length - 1)
            )
            .is_err()
        );
    }
}
