//! Run the MiMo bundle's prepared launches on lab sessions.
//!
//! Spans name lab buffers by slot: `allocate` returns a span over a new
//! buffer whose handle slot is its key in `buffers`.

#![allow(dead_code, reason = "each lab test uses a subset of the helpers")]

use std::collections::{BTreeMap, BTreeSet};

use bundle_mimo26::pipeline::PreparedLaunch;
use kernel_core::contract::Argument;
use kernel_lab::{LabResult, Session};
use runtime_bundle_abi::{BundleDeviceBufferHandleV1 as Handle, DeviceBufferSpanV1 as Span};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument};

/// A zeroed buffer of `bytes`, starting with `data`.
pub fn allocate(
    buffers: &mut BTreeMap<u32, DeviceBuffer>,
    session: &Session,
    bytes: usize,
    data: &[u8],
) -> LabResult<Span> {
    let buffer = session.allocate(bytes)?;
    buffer
        .zero_at_async(0, bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    if !data.is_empty() {
        buffer
            .copy_from_bytes_at(0, data, &session.stream)
            .map_err(|e| e.to_string())?;
    }
    let id = buffers.len() as u32 + 1;
    buffers.insert(id, buffer);
    Ok(Span::new(Handle::from_parts(1, id), 0, bytes as u64))
}

pub fn bf16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Writes `data` at the start of `span`.
pub fn write(
    buffers: &BTreeMap<u32, DeviceBuffer>,
    session: &Session,
    span: Span,
    data: &[u8],
) -> LabResult<()> {
    if data.len() as u64 > span.length {
        return Err("lab write exceeds its span".into());
    }
    buffers[&span.buffer.slot()]
        .copy_from_bytes_at(span.offset as usize, data, &session.stream)
        .map_err(|e| e.to_string())
}

/// Reads `bytes` from the start of `span` after the session's queued work.
pub fn read(
    buffers: &BTreeMap<u32, DeviceBuffer>,
    session: &Session,
    span: Span,
    bytes: usize,
) -> LabResult<Vec<u8>> {
    let mut output = vec![0; bytes];
    buffers[&span.buffer.slot()]
        .copy_to_bytes_at_async(span.offset as usize, &mut output, &session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    Ok(output)
}

/// Resolves every needed symbol on every session before any launch: lazy
/// module loading may synchronize a context waiting for another rank.
pub fn functions(
    sessions: &[Session],
    needed: &BTreeSet<&str>,
) -> LabResult<Vec<BTreeMap<String, Function>>> {
    sessions
        .iter()
        .map(|session| {
            let modules = bundle_mimo26::mtp::program::qualification_cubins()
                .filter(|(_, symbols)| symbols.iter().any(|s| needed.contains(s)))
                .map(|(bytes, symbols)| {
                    Ok((
                        session
                            .context
                            .load_module(bytes)
                            .map_err(|e| e.to_string())?,
                        symbols.iter().copied().collect::<BTreeSet<_>>(),
                    ))
                })
                .collect::<LabResult<Vec<_>>>()?;
            needed
                .iter()
                .map(|&symbol| {
                    let module = &modules
                        .iter()
                        .find(|(_, symbols)| symbols.contains(symbol))
                        .ok_or_else(|| format!("missing bundle cubin for {symbol}"))?
                        .0;
                    Ok((
                        symbol.to_owned(),
                        module.function(symbol).map_err(|e| e.to_string())?,
                    ))
                })
                .collect::<LabResult<BTreeMap<_, _>>>()
        })
        .collect()
}

pub fn launch(
    session: &Session,
    buffers: &BTreeMap<u32, DeviceBuffer>,
    functions: &BTreeMap<String, Function>,
    plan: &PreparedLaunch,
) -> LabResult<()> {
    let span = |id| plan.bindings.get(id).map_err(|e| format!("{e:?}"));
    let buffer = |s: Span| {
        buffers
            .get(&s.buffer.slot())
            .ok_or_else(|| "missing buffer".to_owned())
    };
    let address = |id| {
        let s = span(id)?;
        buffer(s)?
            .device_address_at(s.offset as usize, s.length as usize)
            .map_err(|e| e.to_string())
    };
    let mut maps = BTreeMap::new();
    let mut arrays = BTreeMap::new();
    for (i, arg) in plan.arguments.0.iter().enumerate() {
        match arg {
            Argument::TensorMap(spec) => {
                let s = span(spec.base_buffer)?;
                maps.insert(
                    i,
                    session
                        .context
                        .tensor_map(
                            buffer(s)?,
                            s.offset as usize,
                            s.length as usize,
                            kernel_lab::launch::descriptor(spec)?,
                        )
                        .map_err(|e| e.to_string())?,
                );
            }
            Argument::DevicePointerArray(ids) => {
                arrays.insert(
                    i,
                    ids.iter()
                        .map(|&id| address(id))
                        .collect::<LabResult<Vec<_>>>()?,
                );
            }
            _ => {}
        }
    }
    let args = plan
        .arguments
        .0
        .iter()
        .enumerate()
        .map(|(i, arg)| {
            Ok(match arg {
                Argument::DeviceBuffer(id) => {
                    let s = span(*id)?;
                    KernelArgument::Range {
                        buffer: buffer(s)?,
                        offset: s.offset as usize,
                        byte_len: s.length as usize,
                    }
                }
                Argument::U32(v) => KernelArgument::U32(*v),
                Argument::I32(v) => KernelArgument::I32(*v),
                Argument::U64(v) => KernelArgument::U64(*v),
                Argument::NullDevicePointer => KernelArgument::NullBuffer,
                Argument::TensorMap(_) => KernelArgument::TensorMap(&maps[&i]),
                Argument::DevicePointerArray(_) => KernelArgument::DevicePointerArray(&arrays[&i]),
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let function = &functions[&plan.symbol];
    if plan.plan.dynamic_shared_bytes > 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(plan.plan.dynamic_shared_bytes)
            .map_err(|e| e.to_string())?;
    }
    function
        .launch(
            &session.stream,
            kernel_lab::launch::config(plan.plan)?,
            &args,
        )
        .map_err(|e| format!("{}: {e}", plan.symbol))
}

/// Real instance-style HF tables for the positions used by a GPU fixture.
pub fn hf_rope_tables(
    buffers: &mut BTreeMap<u32, DeviceBuffer>,
    session: &Session,
    positions: usize,
) -> LabResult<[Span; 2]> {
    let mut tables = Vec::new();
    for theta in [1e7, 1e4] {
        let bytes: Vec<u8> = (0..positions)
            .flat_map(|p| {
                (0..32).flat_map(move |pair| {
                    let (c, s) = kernel_core::hf_rope::cos_sin(p, pair, theta);
                    [c as f32, s as f32].into_iter().flat_map(f32::to_le_bytes)
                })
            })
            .collect();
        tables.push(allocate(buffers, session, bytes.len(), &bytes)?);
    }
    Ok(tables.try_into().expect("GA and SWA tables"))
}

/// Refuse a stale computed fallback or an unbound table in a served fixture.
pub fn assert_hf_table_plan(plan: &PreparedLaunch, tables: [Span; 2]) {
    assert!(
        plan.symbol.contains("table"),
        "{} is not a served table entry",
        plan.symbol
    );
    let table = tables[usize::from(!plan.symbol.contains("_ga_"))];
    assert!(
        plan.arguments.0.iter().any(|arg| match arg {
            Argument::DeviceBuffer(id) => plan.bindings.get(*id).ok() == Some(table),
            _ => false,
        }),
        "{} does not bind its HF table",
        plan.symbol
    );
}
