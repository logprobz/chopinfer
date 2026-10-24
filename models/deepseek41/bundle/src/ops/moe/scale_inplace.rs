//! Rebind a qualified consumer to its declared tile-scale bank without
//! changing any arithmetic, launch dimensions or other pointer bindings.
use super::Lowered;
use kernel_core::contract::Argument;
use runtime_bundle_abi::DeviceBufferSpanV1;

pub(crate) fn bind(
    calls: &mut [Lowered],
    raw: DeviceBufferSpanV1,
    encoded: DeviceBufferSpanV1,
    width: u32,
    minimum: u64,
) -> Result<(), String> {
    if encoded.length < minimum
        || encoded.length > u64::from(u32::MAX)
        || !encoded.offset.is_multiple_of(16)
        || encoded.offset.checked_add(encoded.length).is_none()
    {
        return Err("in-place scale bank is too short, too large or unaligned".into());
    }
    let mut readers = 0;
    for call in calls {
        let launch = &mut call.launch;
        let mut pointers = Vec::new();
        for (at, arg) in launch.arguments.0.iter().enumerate() {
            if let Argument::DeviceBuffer(handle) = arg {
                let span = launch.bindings.get(*handle).map_err(|e| format!("{e:?}"))?;
                let end = span
                    .offset
                    .checked_add(span.length)
                    .ok_or("scale pointer overflows")?;
                let raw_end = raw
                    .offset
                    .checked_add(raw.length)
                    .ok_or("raw scale bank overflows")?;
                if span.buffer == raw.buffer && span.offset < raw_end && raw.offset < end {
                    if span.offset != raw.offset || span.length > raw.length {
                        return Err("scale reader exceeds the raw bank".into());
                    }
                    pointers.push(at);
                }
            }
        }
        let variant = crate::Kernels::moe_scale_variant(&launch.symbol, width);
        if pointers.is_empty() && variant.is_none() {
            continue;
        }
        if pointers != [2] {
            return Err(format!("{} has an undeclared scale pointer", launch.symbol));
        }
        let symbol =
            variant.ok_or_else(|| format!("{} has no encoded scale consumer", launch.symbol))?;
        launch.symbol = symbol.into();
        launch.arguments.0[2] = Argument::DeviceBuffer(launch.bindings.bind(encoded));
        readers += 1;
    }
    if readers == 0 {
        return Err("in-place schedule contains no scale reader".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Bind;
    use kernel_core::backend::Launch;
    use kernel_core::contract::{ArgumentList, LaunchPlan};
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    #[test]
    fn rebinding_refuses_unknown_readers_and_invalid_bank_spans() {
        let raw = DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, 1), 256, 4096);
        let encoded =
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(2, 1), 16, 512);
        let spec = crate::sm120::scale_inplace::cubins().unwrap().remove(0).1;
        let encoded_symbol = &spec.entries[0].symbol;
        let (base, _) = encoded_symbol.rsplit_once("_scale_tiles_w").unwrap();
        let raw_symbol = format!("{base}_v1");
        let call = |symbol: &str, span, positions: &[usize]| {
            let mut bind = Bind::default();
            let handle = bind.view(span).handle;
            let mut args = vec![Argument::U32(7); 4];
            for &at in positions {
                args[at] = Argument::DeviceBuffer(handle);
            }
            bind.lowered(Launch {
                symbol: symbol.into(),
                rows: 8,
                plan: LaunchPlan {
                    grid: [6, 1, 1],
                    block: [128, 1, 1],
                    dynamic_shared_bytes: 0,
                    workspace_bytes: 0,
                    cluster: [1; 3],
                    pdl: true,
                },
                arguments: ArgumentList(args),
                collective: false,
                persistent: false,
            })
        };
        let mut good = vec![call(&raw_symbol, raw, &[2])];
        bind(&mut good, raw, encoded, 1, 512).unwrap();
        assert_eq!(good[0].launch.symbol, *encoded_symbol);
        assert_eq!(good[0].launch.plan.grid, [6, 1, 1]);
        for (symbol, span, positions) in [
            ("unknown_scale_reader", raw, vec![2]),
            (raw_symbol.as_str(), raw, vec![1]),
            (raw_symbol.as_str(), raw, vec![1, 2]),
            (
                "unknown_interior_reader",
                DeviceBufferSpanV1::new(raw.buffer, raw.offset + 16, 128),
                vec![2],
            ),
        ] {
            assert!(bind(&mut [call(symbol, span, &positions)], raw, encoded, 1, 512).is_err());
        }
        for (offset, bytes) in [(16, 511), (1, 512), (16, 1_u64 << 32), (u64::MAX - 15, 512)] {
            let bad = DeviceBufferSpanV1::new(encoded.buffer, offset, bytes);
            assert!(bind(&mut [call(&raw_symbol, raw, &[2])], raw, bad, 1, 512).is_err());
        }
        assert!(bind(&mut [call(&raw_symbol, raw, &[2])], raw, encoded, 3, 512).is_err());
    }
}
