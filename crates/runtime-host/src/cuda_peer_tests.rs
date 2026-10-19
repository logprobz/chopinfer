use super::*;
use runtime_cuda::fake_driver;

struct Fixture {
    device: CudaHostDeviceV1,
    contexts: Vec<DeviceObjectV1>,
    pools: Vec<DeviceObjectV1>,
    streams: Vec<DeviceObjectV1>,
    buffers: Vec<DeviceObjectV1>,
    access: PeerAccessAcquisitionV1,
}
impl Fixture {
    fn new() -> Self {
        fake_driver::reset();
        let mut device = CudaHostDeviceV1::new().unwrap();
        let contexts: Vec<_> = (0..2)
            .map(|rank| device.create_context(rank, rank as i32).unwrap())
            .collect();
        let pools: Vec<_> = contexts
            .iter()
            .map(|&context| device.create_memory_pool(context).unwrap())
            .collect();
        let streams: Vec<_> = contexts
            .iter()
            .map(|&context| device.create_stream(context, 0).unwrap())
            .collect();
        let buffers: Vec<_> = pools
            .iter()
            .zip(&streams)
            .map(|(&pool, &stream)| {
                device
                    .allocate_device_buffer(pool, Some(stream), 2048, 1)
                    .unwrap()
            })
            .collect();
        let access = device
            .acquire_peer_access(contexts[0], contexts[1])
            .unwrap();
        device
            .copy_host_to_device(
                streams[0],
                DeviceRangeV1 {
                    object: buffers[0],
                    offset: 0,
                    length: 2048,
                },
                &[0xa5; 2048],
            )
            .unwrap();
        let bytes: Vec<_> = (0..2048).map(|index| (index % 251) as u8).collect();
        device
            .copy_host_to_device(
                streams[1],
                DeviceRangeV1 {
                    object: buffers[1],
                    offset: 0,
                    length: 2048,
                },
                &bytes,
            )
            .unwrap();
        // Finish only setup uploads, before measuring peer submission.
        device.upload_rings.clear();
        Self {
            device,
            contexts,
            pools,
            streams,
            buffers,
            access,
        }
    }
    fn ranges(&self) -> (DeviceRangeV1, DeviceRangeV1) {
        (
            DeviceRangeV1 {
                object: self.buffers[0],
                offset: 1280,
                length: 256,
            },
            DeviceRangeV1 {
                object: self.buffers[1],
                offset: 256,
                length: 256,
            },
        )
    }
    fn copy(&mut self) -> HostDeviceResultV1<()> {
        let (destination, source) = self.ranges();
        self.device.copy_peer_to_peer(
            self.streams[0],
            self.contexts[0],
            destination,
            self.contexts[1],
            source,
        )
    }
    fn finish(mut self) {
        for buffer in self.buffers {
            self.device.release_device_buffer(buffer).unwrap();
        }
        self.device
            .release_peer_access(self.access.object, self.access.host_acquired)
            .unwrap();
        for stream in self.streams {
            self.device.release_stream(stream).unwrap();
        }
        for pool in self.pools {
            self.device.release_memory_pool(pool).unwrap();
        }
        for context in self.contexts {
            self.device.release_context(context).unwrap();
        }
        assert_eq!(self.device.live_object_count(), 0);
    }
}

#[test]
fn verify_result_peer_copy_preserves_bytes_without_a_submission_fence() {
    let mut f = Fixture::new();
    let (destination, source) = f.ranges();
    assert_eq!(
        f.device
            .copy_device_to_device(f.streams[0], destination, source),
        Err(BundleStatusV1::InvalidArgument)
    );
    let fences = fake_driver::call_count(5);
    f.copy()
        .expect("collect the second verify row from its owner rank");
    assert_eq!(fake_driver::call_count(18), 1);
    assert_eq!(
        fake_driver::call_count(5),
        fences,
        "peer submission must not synchronize the host"
    );
    let mut actual = [0; 2048];
    f.device
        .copy_device_to_host(
            f.streams[0],
            &mut actual,
            DeviceRangeV1 {
                object: f.buffers[0],
                offset: 0,
                length: 2048,
            },
        )
        .unwrap();
    assert!(actual[..1280].iter().all(|&byte| byte == 0xa5));
    assert_eq!(
        actual[1280..1536],
        (256..512)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>()
    );
    assert!(actual[1536..].iter().all(|&byte| byte == 0xa5));
    assert_eq!(fake_driver::ordering_violations(), 0);
    f.finish();
}

#[test]
fn peer_results_keep_both_allocations_until_a_successful_completion_query() {
    let mut f = Fixture::new();
    f.copy().unwrap();
    let fences = fake_driver::call_count(5);
    for &buffer in &f.buffers {
        assert!(matches!(
            f.device.buffer(buffer).unwrap().release(),
            Err(CudaError::ResourceShared(_))
        ));
    }
    fake_driver::fail_next(31, 719);
    assert!(f.device.retire_peer_copies().is_err());
    assert_eq!(f.device.peer_copies.len(), 1);
    f.device.retire_peer_copies().unwrap();
    assert_eq!(
        f.device.peer_copies.len(),
        1,
        "not-ready is not permission to release"
    );
    f.device.retire_peer_copies().unwrap();
    assert!(f.device.peer_copies.is_empty());
    assert_eq!(
        fake_driver::call_count(5),
        fences,
        "completed copies retire without a host fence"
    );
    f.finish();
}

#[test]
fn peer_copy_refuses_wrong_contexts_streams_and_bounds_before_submission() {
    let mut f = Fixture::new();
    let (destination, source) = f.ranges();
    let fences = fake_driver::call_count(5);
    for (stream, dc, dst, sc, src) in [
        (
            f.streams[0],
            f.contexts[1],
            destination,
            f.contexts[1],
            source,
        ),
        (
            f.streams[0],
            f.contexts[0],
            destination,
            f.contexts[0],
            source,
        ),
        (
            f.streams[1],
            f.contexts[0],
            destination,
            f.contexts[1],
            source,
        ),
        (
            f.streams[0],
            f.contexts[0],
            DeviceRangeV1 {
                offset: 2000,
                ..destination
            },
            f.contexts[1],
            source,
        ),
        (
            f.streams[0],
            f.contexts[0],
            destination,
            f.contexts[1],
            DeviceRangeV1 {
                length: 255,
                ..source
            },
        ),
    ] {
        assert!(
            f.device
                .copy_peer_to_peer(stream, dc, dst, sc, src)
                .is_err()
        );
    }
    assert_eq!(fake_driver::call_count(18), 0);
    assert_eq!(fake_driver::call_count(5), fences);
    f.finish();
}

#[test]
fn peer_copy_submission_failures_fence_before_releasing_the_leases() {
    for operation in [18, 30] {
        let mut f = Fixture::new();
        let fences = fake_driver::call_count(5);
        fake_driver::fail_next(operation, 719);
        assert!(f.copy().is_err());
        assert!(f.device.peer_copies.is_empty());
        assert_eq!(fake_driver::call_count(5), fences + 1);
        f.finish();
    }
}

#[test]
fn unfinished_peer_copies_apply_bounded_backpressure() {
    let mut f = Fixture::new();
    let fences = fake_driver::call_count(5);
    for index in 0..=MAX_PENDING_PEER_COPIES {
        if index > 0 {
            fake_driver::fail_next(31, 600);
        }
        f.copy().unwrap();
        assert!(f.device.peer_copies.len() <= MAX_PENDING_PEER_COPIES);
    }
    assert_eq!(f.device.peer_copies.len(), MAX_PENDING_PEER_COPIES);
    assert_eq!(
        fake_driver::call_count(5),
        fences + 1,
        "only a full queue forces completion"
    );
    f.finish();
}
