use super::*;
use runtime_host::cuda::CudaHostDeviceV1;

fn span(
    buffer: runtime_bundle_abi::BundleDeviceBufferHandleV1,
    offset: u64,
    length: u64,
) -> DeviceBufferSpanV1 {
    DeviceBufferSpanV1::new(buffer, offset, length)
}
fn exercise() {
    let backend = Rc::new(HostServicesV1Backend::new(
        CudaHostDeviceV1::new().unwrap(),
        HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap()),
    ));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let contexts: Vec<_> = (0..2)
        .map(|rank| host.select_context(rank, rank as i32).unwrap())
        .collect();
    let pools: Vec<_> = contexts
        .iter()
        .map(|&context| host.select_memory_pool(context).unwrap())
        .collect();
    let streams: Vec<_> = contexts
        .iter()
        .map(|&context| host.create_stream(context, 0).unwrap())
        .collect();
    let access = host.acquire_peer_access(contexts[0], contexts[1]).unwrap();
    host.set_memory_pool_peer_access(pools[1], contexts[0], true)
        .unwrap();
    let buffers: Vec<_> = pools
        .iter()
        .zip(&streams)
        .map(|(&pool, &stream)| host.allocate_device_buffer(pool, stream, 4096, 8).unwrap())
        .collect();
    let produced = host.create_event(contexts[1]).unwrap();
    for rows in [1_u64, 4, 8] {
        let bytes = rows * 256;
        let source: Vec<_> = (0..bytes)
            .map(|index| ((index + rows) * 17 % 251) as u8)
            .collect();
        host.copy_host_to_device(streams[0], span(buffers[0], 0, 4096), &[0xa5; 4096])
            .unwrap();
        host.copy_host_to_device(streams[1], span(buffers[1], 512, bytes), &source)
            .unwrap();
        host.record_event(produced, streams[1]).unwrap();
        host.wait_event(streams[0], produced).unwrap();
        host.copy_peer_to_peer(
            streams[0],
            contexts[0],
            span(buffers[0], 1024, bytes),
            contexts[1],
            span(buffers[1], 512, bytes),
        )
        .unwrap();
        let mut actual = vec![0; 4096];
        host.copy_device_to_host(streams[0], &mut actual, span(buffers[0], 0, 4096))
            .unwrap();
        assert!(actual[..1024].iter().all(|&byte| byte == 0xa5));
        assert_eq!(actual[1024..1024 + bytes as usize], source);
        assert!(
            actual[1024 + bytes as usize..]
                .iter()
                .all(|&byte| byte == 0xa5)
        );
    }
    // Teardown must also fence a submitted copy that has no result readback.
    host.copy_peer_to_peer(
        streams[0],
        contexts[0],
        span(buffers[0], 1024, 256),
        contexts[1],
        span(buffers[1], 512, 256),
    )
    .unwrap();
    host.release_event(produced).unwrap();
    host.release_device_buffer(buffers[1]).unwrap();
    host.release_device_buffer(buffers[0]).unwrap();
    host.set_memory_pool_peer_access(pools[1], contexts[0], false)
        .unwrap();
    host.release_peer_access(access).unwrap();
    for stream in streams.into_iter().rev() {
        host.release_stream(stream).unwrap();
    }
    for pool in pools.into_iter().rev() {
        host.release_memory_pool(pool).unwrap();
    }
    for context in contexts.into_iter().rev() {
        host.release_context(context).unwrap();
    }
    assert_eq!(backend.live_resource_count(), 0);
    println!("peer result copies: 1/4/8 sampler records, poison guards and pending release passed");
}

#[cfg(feature = "cuda-fake-driver-tests")]
#[test]
fn peer_result_copy_fake_driver_runs_the_two_rank_service_path() {
    runtime_cuda::fake_driver::reset();
    exercise();
    assert_eq!(runtime_cuda::fake_driver::ordering_violations(), 0);
}

#[cfg(not(feature = "cuda-fake-driver-tests"))]
#[test]
fn peer_results_gpu_preserve_sampler_bytes_and_pending_release() {
    let Ok(devices) = std::env::var("CHOPINFER_KERNEL_GPUS") else {
        eprintln!("skipped: CHOPINFER_KERNEL_GPUS is not set");
        return;
    };
    let devices: Vec<_> = devices.split(',').map(str::trim).collect();
    assert!(
        devices.len() >= 2 && devices[0] == "0" && devices[1] == "1",
        "probe needs visible GPUs 0 and 1"
    );
    exercise();
}
