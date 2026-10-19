use super::*;
use runtime_bundle_abi::MappedGatherPlaneV1;
use runtime_host::HostDeviceBackendV1;

fn exercise<D: HostDeviceBackendV1 + 'static>(device: D, delayed: bool, label: &str) {
    let root = temp(label);
    write_checkpoint(&root.0);
    let image = planned(&root.0, (0..2).map(|rank| plan(2, rank)).collect()).unwrap();
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
        .with_planned_compiled_image(Arc::new(image))
        .unwrap();
    let backend = Rc::new(HostServicesV1Backend::new(device, config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let mut ranks = Vec::new();
    for rank in 0..2 {
        let context = host.select_context(rank, rank as i32).unwrap();
        let pool = host.select_memory_pool(context).unwrap();
        let stream = host.create_stream(context, 0).unwrap();
        ranks.push((context, pool, stream));
    }
    let image = host.open_compiled_image(ROLE.as_bytes()).unwrap();
    let ids = host
        .allocate_device_buffer(ranks[0].1, ranks[0].2, 32, 8)
        .unwrap();
    let input = DeviceBufferSpanV1::new(ids, 0, 32);
    let mut outputs = Vec::new();
    let mut planes = Vec::new();
    for (rank, &(_, pool, stream)) in ranks.iter().enumerate() {
        let buffer = host.allocate_device_buffer(pool, stream, 48, 8).unwrap();
        outputs.push(buffer);
        for (view_id, row_bytes, offset, length) in [
            (b"codes".to_vec(), 8, 0, 32),
            (b"scales".to_vec(), 1, 32, 4),
        ] {
            planes.push(MappedGatherPlaneV1 {
                image,
                rank: rank as u32,
                view_id,
                row_bytes,
                index_offset: 8,
                stream,
                destination: DeviceBufferSpanV1::new(buffer, offset, length),
            });
        }
    }
    let invalid = Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument));
    for change in 0..6 {
        let mut bad = planes.clone();
        match change {
            0 => bad[0].index_offset = 3,
            1 => bad[0].row_bytes = 0,
            2 => bad[0].destination.length = 31,
            3 => bad[1].destination.offset = 0,
            4 => bad[0].destination = input,
            _ => bad[0].stream = ranks[1].2,
        }
        assert_eq!(host.create_mapped_gather(ranks[0].2, input, &bad), invalid);
    }
    host.begin_graph_capture(ranks[0].2).unwrap();
    assert_eq!(
        host.create_mapped_gather(ranks[0].2, input, &planes),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    let graph = host.end_graph_capture(ranks[0].2).unwrap();
    host.release_graph(graph).unwrap();
    let gather = host
        .create_mapped_gather(ranks[0].2, input, &planes)
        .unwrap();
    assert_eq!(backend.host_memory_v1().gather_staging_bytes, 32 + 2 * 36);
    assert!(host.release_device_buffer(outputs[1]).is_err());
    host.begin_graph_capture(ranks[0].2).unwrap();
    let refused = Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
    assert_eq!(host.enqueue_mapped_gather(gather, 1), refused);
    assert_eq!(host.wait_mapped_gather(gather, ranks[0].2), refused);
    assert_eq!(host.release_mapped_gather(gather), refused);
    assert_eq!(
        host.complete_mapped_gather(gather),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    let graph = host.end_graph_capture(ranks[0].2).unwrap();
    host.release_graph(graph).unwrap();
    assert!(host.enqueue_mapped_gather(gather, 5).is_err());
    assert!(host.enqueue_mapped_gather(gather, 0).is_err());
    let read = |rank: usize| {
        let mut bytes = [0; 48];
        host.copy_device_to_host(
            ranks[rank].2,
            &mut bytes,
            DeviceBufferSpanV1::new(outputs[rank], 0, 48),
        )
        .unwrap();
        bytes
    };
    // The synthetic driver deliberately returns NOT_READY once per event,
    // even after a stream fence. Exercise the nonblocking completion contract.
    let complete = |gather| {
        for _ in 0..16 {
            match host.complete_mapped_gather(gather) {
                Ok(None) => continue,
                result => return result,
            }
        }
        panic!("prepared gather never completed");
    };
    for picked in [&[7_u32, 1, 255][..], &[19][..]] {
        let words = [999_u32, 999]
            .into_iter()
            .chain(picked.iter().copied())
            .chain([0; 8])
            .take(8)
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        host.copy_host_to_device(ranks[0].2, input, &words).unwrap();
        for (rank, &buffer) in outputs.iter().enumerate() {
            host.copy_host_to_device(
                ranks[rank].2,
                DeviceBufferSpanV1::new(buffer, 0, 48),
                &[0xa5; 48],
            )
            .unwrap();
        }
        host.enqueue_mapped_gather(gather, picked.len() as u64)
            .unwrap();
        // CPU fake deliberately withholds DMA until the consumer dependency.
        if delayed {
            assert_eq!(host.complete_mapped_gather(gather).unwrap(), None);
            assert_eq!(read(0), [0xa5; 48]);
            assert!(host.enqueue_mapped_gather(gather, 1).is_err());
        }
        for (rank, &(_, _, stream)) in ranks.iter().enumerate() {
            host.wait_mapped_gather(gather, stream).unwrap();
            let actual = read(rank);
            let expected_codes = picked
                .iter()
                .flat_map(|&row| codes()[row as usize * 8..][..8].to_vec())
                .collect::<Vec<_>>();
            assert_eq!(&actual[..picked.len() * 8], expected_codes);
            assert!(
                actual[picked.len() * 8..32]
                    .iter()
                    .all(|&byte| byte == 0xa5)
            );
            let expected_scales = picked
                .iter()
                .map(|&row| scales()[rank * 512 + row as usize])
                .collect::<Vec<_>>();
            assert_eq!(&actual[32..32 + picked.len()], expected_scales);
            assert!(actual[32 + picked.len()..].iter().all(|&byte| byte == 0xa5));
        }
        assert!(complete(gather).unwrap().is_some());
    }
    // Device indices past a plane's bound fail at completion, never publish success.
    let mut bad = [0; 32];
    bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    host.copy_host_to_device(ranks[0].2, input, &bad).unwrap();
    host.enqueue_mapped_gather(gather, 1).unwrap();
    for &(_, _, stream) in &ranks {
        host.wait_mapped_gather(gather, stream).unwrap();
        // Match the ordinary result boundary before querying completion.
        host.synchronize_stream(stream).unwrap();
    }
    assert_eq!(
        complete(gather),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    assert!(host.enqueue_mapped_gather(gather, 1).is_err());
    host.release_mapped_gather(gather).unwrap();
    assert_eq!(backend.host_memory_v1().gather_staging_bytes, 0);
    // Cancellation releases pending staging; old generations stay invalid.
    host.copy_host_to_device(ranks[0].2, input, &[0; 32])
        .unwrap();
    let next = host
        .create_mapped_gather(ranks[0].2, input, &planes)
        .unwrap();
    assert_ne!(next, gather);
    assert!(host.enqueue_mapped_gather(gather, 1).is_err());
    host.enqueue_mapped_gather(next, 1).unwrap();
    host.release_mapped_gather(next).unwrap();
    for buffer in outputs.into_iter().rev() {
        host.release_device_buffer(buffer).unwrap();
    }
    host.release_device_buffer(ids).unwrap();
    host.release_compiled_image(image).unwrap();
    for (context, pool, stream) in ranks.into_iter().rev() {
        host.release_stream(stream).unwrap();
        host.release_memory_pool(pool).unwrap();
        host.release_context(context).unwrap();
    }
    assert_eq!(backend.live_resource_count(), 0);
}

#[test]
fn async_gather_waits_for_consumers_reuses_staging_and_refuses_bad_device_ids() {
    exercise(FakeHostDeviceV1::new(), true, "async-fake");
}

#[cfg(feature = "cuda-fake-driver-tests")]
#[test]
fn async_gather_uses_event_ordered_cuda_transfers_and_releases_pinned_storage() {
    runtime_cuda::fake_driver::reset();
    exercise(
        runtime_host::cuda::CudaHostDeviceV1::new().unwrap(),
        false,
        "async-driver",
    );
    assert_eq!(runtime_cuda::fake_driver::ordering_violations(), 0);
}

#[cfg(all(feature = "linux-cuda-host", not(feature = "cuda-fake-driver-tests")))]
#[test]
fn async_gather_gpu_preserves_bytes_across_two_ranks_and_pending_release() {
    let Ok(devices) = std::env::var("CHOPINFER_KERNEL_GPUS") else {
        eprintln!("skipped: CHOPINFER_KERNEL_GPUS is not set");
        return;
    };
    let devices: Vec<i32> = devices
        .split(',')
        .map(|word| word.trim().parse().expect("GPU ordinal"))
        .collect();
    assert!(
        devices.contains(&0) && devices.contains(&1),
        "fixture requires GPUs 0 and 1"
    );
    exercise(
        runtime_host::cuda::CudaHostDeviceV1::new().unwrap(),
        false,
        "async-gpu",
    );
    println!("async gather GPU byte parity, bounds, reuse and pending release passed");
}
