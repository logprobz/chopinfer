//! Concurrent graph replays with distinct input, status, outputs and workspace.
use super::*;
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig};

pub(super) struct Case<'a> {
    pub images: &'a [Vec<u8>],
    pub buffers: &'a [DeviceBuffer],
    pub bindings: &'a Bindings<'a>,
    pub calls: &'a [Launch],
    pub functions: &'a [Function],
    pub baseline_calls: &'a [Launch],
    pub baseline_functions: &'a [Function],
    pub paired: bool,
    pub persistent: Option<(&'a [Launch], &'a [Function], usize)>,
}

pub(super) fn check(session: &Session, case: Case<'_>) -> LabResult<()> {
    let copy = |buffer: &DeviceBuffer, bytes: &[u8]| {
        buffer
            .copy_from_bytes(bytes, &session.stream)
            .map_err(|e| e.to_string())
    };
    let read = |buffer: &DeviceBuffer| -> LabResult<Vec<u8>> {
        let mut bytes = vec![0; buffer.byte_len()];
        buffer
            .copy_to_bytes(&mut bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    };
    // Negating every code gives the second op different fold partials,
    // making an accidental shared workspace observable.
    let other_codes: Vec<u8> = case.images[0].iter().map(|v| v ^ 0x80).collect();
    let other = [
        session.allocate(other_codes.len())?,
        session.allocate(case.images[11].len())?,
        session.allocate(case.images[12].len())?,
        session.allocate(case.images[13].len())?,
        session.allocate(case.images[7].len())?,
    ];
    copy(&other[0], &other_codes)?;
    run(
        session,
        &session.stream,
        case.baseline_calls,
        case.baseline_functions,
        case.bindings,
    )?;
    let expected_a = [read(&case.buffers[4])?, read(&case.buffers[10])?];
    let mut baseline_b = Bindings {
        buffers: case.bindings.buffers.clone(),
    };
    baseline_b.buffers.insert(1, &other[0]);
    run(
        session,
        &session.stream,
        case.baseline_calls,
        case.baseline_functions,
        &baseline_b,
    )?;
    let expected_b = [read(&case.buffers[4])?, read(&case.buffers[10])?];
    let expected_status = read(&case.buffers[6])?;
    let mut bindings_b = Bindings {
        buffers: case.bindings.buffers.clone(),
    };
    for (handle, buffer) in [1, 12, 13, 14, 8].into_iter().zip(&other) {
        bindings_b.buffers.insert(handle, buffer);
    }
    let other_state = case
        .persistent
        .map(|(_, _, index)| session.allocate(case.images[index].len()))
        .transpose()?;
    if let Some((calls, functions, index)) = case.persistent {
        let state = other_state.as_ref().unwrap();
        copy(state, &case.images[index])?;
        bindings_b.buffers.insert(index as u64 + 1, state);
        run(session, &session.stream, calls, functions, &bindings_b)?;
    }
    let graph_a = session
        .capture(|stream| run(session, stream, case.calls, case.functions, case.bindings))?;
    let graph_b =
        session.capture(|stream| run(session, stream, case.calls, case.functions, &bindings_b))?;
    let stream_b = session.context.stream().map_err(|e| e.to_string())?;
    let release = session.context.stream().map_err(|e| e.to_string())?;
    let gate = session.allocate(4)?;
    let gate_fn = session
        .module("probe_sm120")?
        .function("chop_probe_gate_v1")
        .map_err(|e| e.to_string())?;
    let config = kernel_harness::launch::simple_config(
        (1, 1, 1),
        (32, 1, 1),
        0,
        LaunchConfig::NO_CLUSTER,
        false,
    )?;
    for _ in 0..3 {
        for index in [7, 11, 12, 13] {
            copy(&case.buffers[index], &case.images[index])?;
        }
        for (buffer, index) in other[1..].iter().zip([11, 12, 13, 7]) {
            copy(buffer, &case.images[index])?;
        }
        copy(&gate, &0_u32.to_ne_bytes())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let scheduled = (|| -> LabResult<()> {
            gate_fn
                .launch(&session.stream, config, &[KernelArgument::Buffer(&gate)])
                .map_err(|e| e.to_string())?;
            gate_fn
                .launch(&stream_b, config, &[KernelArgument::Buffer(&gate)])
                .map_err(|e| e.to_string())?;
            graph_a.launch(&session.stream).map_err(|e| e.to_string())?;
            graph_b.launch(&stream_b).map_err(|e| e.to_string())?;
            Ok(())
        })();
        // Release both streams even if enqueueing a graph reports an error.
        gate.copy_from_bytes(&1_u32.to_ne_bytes(), &release)
            .map_err(|e| e.to_string())?;
        stream_b.synchronize().map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        scheduled?;
        if let Some((_, _, index)) = case.persistent {
            assert!(read(&case.buffers[index])?.iter().all(|&v| v == 0));
            assert!(read(other_state.as_ref().unwrap())?.iter().all(|&v| v == 0));
        }
        assert_eq!(
            read(&case.buffers[11])?,
            expected_a[0],
            "concurrent output A"
        );
        assert_eq!(read(&other[1])?, expected_b[0], "concurrent output B");
        assert_eq!(
            read(&case.buffers[13])?,
            expected_status,
            "concurrent status A"
        );
        assert_eq!(read(&other[3])?, expected_status, "concurrent status B");
        if case.paired {
            assert_eq!(
                read(&case.buffers[12])?,
                expected_a[1],
                "concurrent paired output A"
            );
            assert_eq!(
                read(&other[2])?,
                expected_b[1],
                "concurrent paired output B"
            );
        }
    }
    Ok(())
}
