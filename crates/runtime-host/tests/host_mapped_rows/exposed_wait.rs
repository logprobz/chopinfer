//! The step trace's prepared-gather wait: each completed gather adds its
//! longest consumer-stream wait to the step whose execution completes it.
use super::*;
use runtime_bundle_abi::{BundleMappedGatherHandleV1, MappedGatherPlaneV1};
use runtime_engine::profile;

#[test]
fn completed_gathers_add_their_longest_wait_to_the_completing_step() {
    let root = temp("exposed-wait");
    write_checkpoint(&root.0);
    let image = planned(&root.0, (0..2).map(|rank| plan(2, rank)).collect()).unwrap();
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
        .with_planned_compiled_image(Arc::new(image))
        .unwrap();
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
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
    // Two layers, each gathering four code rows onto both ranks' streams.
    // Resources release in reverse creation order, so every buffer precedes
    // the gathers.
    let mut buffers = Vec::new();
    let mut layers = Vec::new();
    for _ in 0..2 {
        let ids = host
            .allocate_device_buffer(ranks[0].1, ranks[0].2, 16, 8)
            .unwrap();
        buffers.push(ids);
        let indices = DeviceBufferSpanV1::new(ids, 0, 16);
        let words = [3_u32, 1, 4, 1]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        host.copy_host_to_device(ranks[0].2, indices, &words)
            .unwrap();
        let mut planes = Vec::new();
        for (rank, &(_, pool, stream)) in ranks.iter().enumerate() {
            let buffer = host.allocate_device_buffer(pool, stream, 32, 8).unwrap();
            buffers.push(buffer);
            planes.push(MappedGatherPlaneV1 {
                image,
                rank: rank as u32,
                view_id: b"codes".to_vec(),
                row_bytes: 8,
                index_offset: 0,
                stream,
                destination: DeviceBufferSpanV1::new(buffer, 0, 32),
            });
        }
        layers.push((indices, planes));
    }
    let gathers = layers
        .iter()
        .map(|(indices, planes)| {
            host.create_mapped_gather(ranks[0].2, *indices, planes)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let join = |gather: BundleMappedGatherHandleV1| {
        host.enqueue_mapped_gather(gather, 4).unwrap();
        for &(_, _, stream) in &ranks {
            host.wait_mapped_gather(gather, stream).unwrap();
        }
    };
    let complete = |gather: BundleMappedGatherHandleV1| {
        host.complete_mapped_gather(gather)
            .unwrap()
            .expect("joined fake gather")
            .exposed_nanoseconds
    };
    let waits = |nanoseconds: Vec<u64>| {
        backend.with_device_mut(|device| device.set_gather_waits(nanoseconds));
    };
    let recorded = profile::mapped_gather_exposed_nanoseconds;
    let synchronizations = backend.with_device(FakeHostDeviceV1::synchronizations);

    // The layers' longest waits fall on different ranks: 5 and 1 us, then 1
    // and 4 us. No stream waited more than 6 us, yet the step reports 9: the
    // sum bounds the critical-path wait from above, it is not a union.
    waits(vec![5_000, 1_000, 1_000, 4_000]);
    let step = profile::TraceTotals::begin();
    join(gathers[0]);
    join(gathers[1]);
    assert_eq!([complete(gathers[0]), complete(gathers[1])], [5_000, 4_000]);
    assert_eq!(recorded(), 9_000);
    drop(step);

    // A gather joined in one step and completed in the next is charged there.
    waits(vec![2_000, 3_000]);
    let step = profile::TraceTotals::begin();
    join(gathers[0]);
    assert_eq!(recorded(), 0);
    drop(step);
    let step = profile::TraceTotals::begin();
    assert_eq!(complete(gathers[0]), 3_000);
    assert_eq!(recorded(), 3_000);
    drop(step);

    // Outside a traced step completion reports the same stats and records
    // nothing.
    waits(vec![7_000]);
    join(gathers[1]);
    assert_eq!(complete(gathers[1]), 7_000);
    assert_eq!(recorded(), 3_000);

    // A traced step without prepared gathers reports none.
    let step = profile::TraceTotals::begin();
    assert_eq!(recorded(), 0);
    drop(step);
    // Recording added no stream synchronization.
    assert_eq!(
        backend.with_device(FakeHostDeviceV1::synchronizations),
        synchronizations
    );

    for gather in gathers.into_iter().rev() {
        host.release_mapped_gather(gather).unwrap();
    }
    for buffer in buffers.into_iter().rev() {
        host.release_device_buffer(buffer).unwrap();
    }
    host.release_compiled_image(image).unwrap();
    for (context, pool, stream) in ranks.into_iter().rev() {
        host.release_stream(stream).unwrap();
        host.release_memory_pool(pool).unwrap();
        host.release_context(context).unwrap();
    }
    assert_eq!(backend.live_resource_count(), 0);
}
