use super::*;

struct Fixture {
    source: DeviceBuffer,
    targets: Vec<HostTransformTarget>,
    owners: Vec<Stream>,
    transform: HostTransform,
}

fn fixture() -> Fixture {
    ffi::fake_driver_reset();
    let driver = Driver::initialize().unwrap();
    let contexts = [
        driver.retain_primary_context(0).unwrap(),
        driver.retain_primary_context(1).unwrap(),
    ];
    let owners: Vec<_> = contexts
        .iter()
        .map(|context| context.stream().unwrap())
        .collect();
    let source = contexts[0]
        .default_memory_pool()
        .unwrap()
        .allocate(&owners[0], 8)
        .unwrap();
    source
        .copy_from_bytes(&[1, 2, 3, 4, 5, 6, 7, 8], &owners[0])
        .unwrap();
    let targets = contexts
        .iter()
        .zip(&owners)
        .enumerate()
        .map(|(output, (context, stream))| HostTransformTarget {
            buffer: context
                .default_memory_pool()
                .unwrap()
                .allocate(stream, 8)
                .unwrap(),
            offset: 0,
            output,
            source_offset: 0,
            length: 8,
        })
        .collect();
    let transform =
        HostTransform::new(&contexts[0], 8, &[(&owners[0], 8), (&owners[1], 8)]).unwrap();
    Fixture {
        source,
        targets,
        owners,
        transform,
    }
}

fn enqueue(f: &mut Fixture) -> Result<(), CudaError> {
    f.transform.enqueue(
        &f.source,
        0,
        8,
        &f.owners[0],
        &f.targets,
        |input, output| {
            output[0].copy_from_slice(input);
            for (to, from) in output[1].iter_mut().zip(input.iter().rev()) {
                *to = *from;
            }
            Ok(())
        },
    )
}

fn finish(f: &mut Fixture) -> HostTransformStats {
    for index in 0..2 {
        f.transform.wait_on(index, &f.owners[index]).unwrap();
    }
    assert!(f.transform.complete().unwrap().is_none());
    for _ in 0..16 {
        if let Some(stats) = f.transform.complete().unwrap() {
            return stats;
        }
    }
    panic!("fake events never completed");
}

#[test]
fn host_transform_reuses_pinned_storage_and_waits_only_on_events() {
    let mut f = fixture();
    assert_eq!(f.transform.pinned_bytes(), 24);
    let allocated = ffi::fake_driver_call_count(16);
    for _ in 0..2 {
        let drains = ffi::fake_driver_call_count(5);
        enqueue(&mut f).unwrap();
        assert!(enqueue(&mut f).is_err(), "pending storage was reused");
        let stats = finish(&mut f);
        assert_eq!(stats.failure, None);
        assert_eq!(stats.exposed_milliseconds.len(), 2);
        assert!(stats.exposed_milliseconds.iter().all(|ms| *ms > 0.0));
        assert_eq!(ffi::fake_driver_call_count(5), drains);
        assert_eq!(ffi::fake_driver_call_count(16), allocated);
        assert_eq!(ffi::fake_driver_ordering_violations(), 0);
        for (index, expected) in [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]]
            .iter()
            .enumerate()
        {
            let mut actual = [0; 8];
            f.targets[index]
                .buffer
                .copy_to_bytes(&mut actual, &f.owners[index])
                .unwrap();
            assert_eq!(&actual, expected);
        }
    }
    assert_eq!(ffi::fake_driver_call_count(34), 2);
}

#[test]
fn host_transform_rejects_bad_ranges_before_queuing_work() {
    let mut f = fixture();
    assert!(
        f.transform
            .enqueue(&f.source, 1, 8, &f.owners[0], &f.targets, |_, _| Ok(()))
            .is_err()
    );
    f.targets[1].source_offset = 1;
    assert!(enqueue(&mut f).is_err());
    assert_eq!(ffi::fake_driver_call_count(34), 0);
    f.targets[1].source_offset = 0;
    enqueue(&mut f).unwrap();
    finish(&mut f);
}

#[test]
fn host_transform_failed_submission_cannot_reuse_old_ready_events() {
    for failure in [8, 34] {
        let mut f = fixture();
        enqueue(&mut f).unwrap();
        finish(&mut f);
        ffi::fake_driver_fail_next(failure, 719);
        assert!(enqueue(&mut f).is_err());
        assert!(f.transform.complete().is_err());
        assert!(f.transform.wait_on(0, &f.owners[0]).is_err());
        let drains = ffi::fake_driver_call_count(5);
        drop(f);
        assert!(ffi::fake_driver_call_count(5) >= drains + 3);
    }
}

#[test]
fn host_transform_reports_callback_failure_after_upload_completion() {
    let mut f = fixture();
    f.transform
        .enqueue(&f.source, 0, 8, &f.owners[0], &f.targets, |_, _| {
            Err("invalid table index".into())
        })
        .unwrap();
    let stats = finish(&mut f);
    assert_eq!(stats.failure.as_deref(), Some("invalid table index"));
    for index in 0..2 {
        let mut actual = [0xff; 8];
        f.targets[index]
            .buffer
            .copy_to_bytes(&mut actual, &f.owners[index])
            .unwrap();
        assert_eq!(actual, [0; 8]);
    }
}
