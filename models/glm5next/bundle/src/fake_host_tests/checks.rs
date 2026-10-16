//! What the fake host checks of the device calls it serves: an upload, a
//! readback and a device copy move exactly their spans' bytes, with or
//! without the backing; a launch checks its flags before any injected
//! failure, and every buffer its arguments name, inside pointer arrays and
//! tensor maps too. The device answers the batch-limit and residency
//! queries, and its injected copy and launch failures fail the next copy
//! or launch before its checks. While the device's snapshot record is on,
//! uploads, readbacks and device copies land in it. A span whose end
//! overflows lies past its buffer, with or without overflow checks. A
//! handle that fails to be made (an overflowing rank or handle ID) is not
//! counted as acquired.

use runtime_bundle_abi::*;

use super::FakeHost;

/// One rank's stream and two 64-byte buffers of `host`.
fn buffers(
    host: &impl HostServicesBackendV1,
) -> (
    BundleStreamHandleV1,
    BundleDeviceBufferHandleV1,
    BundleDeviceBufferHandleV1,
) {
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let first = host.allocate_device_buffer(pool, stream, 64, 16).unwrap();
    let second = host.allocate_device_buffer(pool, stream, 64, 16).unwrap();
    (stream, first, second)
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn an_upload_fills_exactly_its_destination() {
    let host = FakeHost::new((0, 0));
    let (stream, buffer, _) = buffers(&*host);
    let _ = host.copy_host_to_device(stream, DeviceBufferSpanV1::new(buffer, 0, 16), &[1; 8]);
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn a_readback_fills_exactly_its_destination() {
    let host = FakeHost::new((0, 0));
    let (stream, buffer, _) = buffers(&*host);
    let mut bytes = [0; 8];
    let _ = host.copy_device_to_host(stream, &mut bytes, DeviceBufferSpanV1::new(buffer, 0, 16));
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn a_device_copy_moves_as_many_bytes_as_it_reads() {
    let host = FakeHost::new((0, 0));
    let (stream, first, second) = buffers(&*host);
    let _ = host.copy_device_to_device(
        stream,
        DeviceBufferSpanV1::new(second, 0, 16),
        DeviceBufferSpanV1::new(first, 0, 8),
    );
}

/// The symbol `name` of a module `host` loaded in the first rank's context.
fn symbol(host: &impl HostServicesBackendV1, name: &str) -> BundleCudaSymbolHandleV1 {
    let module = host
        .load_cuda_module(BundleContextHandleV1::from_parts(1, 1), &[])
        .unwrap();
    host.resolve_cuda_symbol(module, name.as_bytes()).unwrap()
}

/// One block of one thread, launched with `flags`.
fn config(flags: u32) -> LaunchConfigV1 {
    LaunchConfigV1 {
        flags,
        grid_x: 1,
        grid_y: 1,
        grid_z: 1,
        block_x: 1,
        block_y: 1,
        block_z: 1,
        shared_memory_bytes: 0,
        cluster: LaunchClusterV1::NONE,
    }
}

/// A buffer slot the host never handed out.
fn unknown() -> DeviceBufferSpanV1 {
    DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, 999), 0, 8)
}

#[test]
#[should_panic(expected = "unknown buffer 999")]
fn a_launch_checks_the_buffers_of_its_pointer_arrays() {
    let host = FakeHost::new((0, 0));
    let (stream, buffer, _) = buffers(&*host);
    let kernel = symbol(&*host, "kernel");
    let peers = vec![
        Some(DeviceBufferSpanV1::new(buffer, 0, 8)),
        None,
        Some(unknown()),
    ];
    let _ = host.launch_cuda_symbol(
        stream,
        kernel,
        config(BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION),
        &[KernelArgumentV1::DevicePointerArray(peers)],
    );
}

#[test]
#[should_panic(expected = "unknown buffer 999")]
fn a_launch_checks_the_buffers_of_its_tensor_maps() {
    let host = FakeHost::new((0, 0));
    let (stream, _, _) = buffers(&*host);
    let kernel = symbol(&*host, "kernel");
    let map = TensorMapV1 {
        buffer: unknown(),
        rank: 1,
        element_type: TensorMapElementV1::U8,
        swizzle: TensorMapSwizzleV1::None,
        interleave: 0,
        l2_promotion: 0,
        oob_fill: 0,
        global_dims: [8, 1, 1, 1, 1],
        global_strides_bytes: [8, 8, 8, 8],
        box_dims: [8, 1, 1, 1, 1],
        element_strides: [1; 5],
    };
    let _ = host.launch_cuda_symbol(
        stream,
        kernel,
        config(BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION),
        &[KernelArgumentV1::TensorMap(map)],
    );
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn a_launch_checks_its_flags_before_an_injected_failure() {
    let host = FakeHost::new((0, 0));
    let (stream, _, _) = buffers(&*host);
    let kernel = symbol(&*host, "kernel");
    *host.fail_launch_on_rank.borrow_mut() = Some(("kernel".into(), 0));
    let _ = host.launch_cuda_symbol(stream, kernel, config(0), &[]);
}

#[test]
fn batch_limits_report_the_devices_scored_rows() {
    let host = FakeHost::new((0, 0));
    host.device.scored_rows.set(17);
    assert_eq!(host.query_batch_limits().unwrap().maximum_scored_rows, 17);
}

/// The device's model answers for the low-latency all-reduce's locked
/// resources at the queried launch.
#[test]
fn residency_answers_from_the_device() {
    let host = FakeHost::new((0, 0));
    buffers(&*host);
    let entry = model_glm5next_contract::sm120::engines::LL_SYMBOL;
    let kernel = symbol(&*host, entry);
    let launch = config(BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION);
    assert_eq!(
        host.query_cluster_residency(kernel, launch),
        Ok(crate::sm120::device::report(
            super::ll_resources()[entry],
            launch
        ))
    );
}

#[test]
fn a_device_copy_fault_fails_the_next_copy_before_its_checks() {
    let host = FakeHost::new((0, 0));
    let (stream, first, second) = buffers(&*host);
    let lines = host.device.journal.borrow().len();
    host.device.fail_copy.set(Some(0));
    assert_eq!(
        host.copy_device_to_device(stream, unknown(), unknown()),
        Err(BundleStatusV1::HostError)
    );
    assert_eq!(host.device.journal.borrow().len(), lines);
    host.copy_device_to_device(
        stream,
        DeviceBufferSpanV1::new(second, 0, 8),
        DeviceBufferSpanV1::new(first, 0, 8),
    )
    .unwrap();
}

#[test]
fn a_device_launch_fault_fails_the_next_launch_before_its_buffer_checks() {
    let host = FakeHost::new((0, 0));
    let (stream, _, _) = buffers(&*host);
    let kernel = symbol(&*host, "kernel");
    let flags = config(BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION);
    let lines = host.device.journal.borrow().len();
    host.device.fail_launch.set(Some(0));
    assert_eq!(
        host.launch_cuda_symbol(
            stream,
            kernel,
            flags,
            &[KernelArgumentV1::DeviceBuffer(unknown())]
        ),
        Err(BundleStatusV1::HostError)
    );
    assert_eq!(host.device.journal.borrow().len(), lines);
    host.launch_cuda_symbol(stream, kernel, flags, &[]).unwrap();
}

#[test]
fn the_snapshot_record_holds_uploads_readbacks_and_copies() {
    let host = FakeHost::new((0, 0));
    let (stream, first, second) = buffers(&*host);
    let (source, destination) = (
        DeviceBufferSpanV1::new(first, 0, 8),
        DeviceBufferSpanV1::new(second, 8, 8),
    );
    *host.device.snapshot_ops.borrow_mut() = Some(Vec::new());
    host.copy_host_to_device(stream, source, &[1; 8]).unwrap();
    host.copy_device_to_host(stream, &mut [0; 8], source)
        .unwrap();
    host.copy_device_to_device(stream, destination, source)
        .unwrap();
    let ops = host.device.snapshot_ops.borrow_mut().take().unwrap();
    let expected = [
        bundle_runtime::fake_host::SnapshotOp::Upload(stream.slot(), source, vec![1; 8]),
        bundle_runtime::fake_host::SnapshotOp::Read(stream.slot(), source),
        bundle_runtime::fake_host::SnapshotOp::Copy(stream.slot(), source, destination),
    ];
    assert_eq!(format!("{ops:?}"), format!("{expected:?}"));
}

#[test]
#[should_panic(expected = "passes its buffer's 64 bytes")]
fn a_span_whose_end_overflows_lies_past_its_buffer() {
    let host = FakeHost::new((0, 0));
    let (stream, buffer, _) = buffers(&*host);
    let _ = host.copy_host_to_device(
        stream,
        DeviceBufferSpanV1::new(buffer, u64::MAX - 3, 8),
        &[1; 8],
    );
}

/// Whether `call` panicked, as it does with overflow checks on; with them
/// off the overflowing value wraps and the call succeeds.
fn panics<T>(call: impl FnOnce() -> T) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)).is_err()
}

#[test]
fn a_context_of_an_overflowing_rank_is_counted_only_once_made() {
    let host = FakeHost::new((0, 0));
    let failed = panics(|| host.select_context(u32::MAX, 0));
    assert_eq!(host.acquired.get(), u32::from(!failed));
}

#[test]
fn an_image_of_an_exhausted_handle_id_is_counted_only_once_made() {
    let host = FakeHost::new((0, 0));
    host.device.next.set(u32::MAX);
    let failed = panics(|| {
        host.open_compiled_image(model_glm5next_contract::direct::RANK_IMAGE_ROLE.as_bytes())
    });
    assert_eq!(host.acquired.get(), u32::from(!failed));
}

#[test]
#[should_panic(expected = "dynamic_shared_memory_bytes <= DYNAMIC_SHARED_BYTES")]
fn a_symbol_configures_at_most_the_devices_shared_memory() {
    let host = FakeHost::new((0, 0));
    buffers(&*host);
    let kernel = symbol(&*host, "kernel");
    let _ = host.configure_cuda_symbol(kernel, 101_377);
}
