//! The package does not allocate verify drafts, so an allocated verify
//! step (`chopbatch-v7`) is refused before any work, and the same step
//! without its allocation runs.
use super::*;
use runtime_bundle_sdk::{SamplingKindV1, VerifyAllocationV1, VerifyCalibrationV1};

#[test]
fn allocated_verify_steps_are_refused_until_the_package_allocates() {
    let (host, mut instance) = create(Shape::new(&FLASH, Ranks::TP4).unwrap());
    let prefill = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(1, false, 3, true, vec![1, 2, 3])],
        vec![BatchAppendV1::new(0, 3, 3, vec![0], None)],
    )
    .with_state_slots(vec![0])
    .encode()
    .unwrap();
    execute(&mut instance, &host, &prefill);
    instance.commit().unwrap();
    let start = state(&instance, 1).end;
    let step = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::verify(1, 4, true, vec![7])],
        vec![BatchAppendV1::new(start, 4, start + 4, vec![0], None)],
    )
    .with_state_slots(vec![0]);
    let calibration = VerifyCalibrationV1::new(&[(1.0, 0.0); 5], &[(1.0, 0.0); 5]).unwrap();
    let allocated = step
        .clone()
        .with_verify_allocation(VerifyAllocationV1::new(
            3,
            1,
            calibration,
            vec![SamplingKindV1::Greedy],
        ))
        .encode()
        .unwrap();
    let mut sink = Sink::default();
    assert_eq!(
        instance.execute(&allocated, &mut sink).err(),
        Some(BundleAbiError::Status(BundleStatusV1::Unsupported))
    );
    execute(&mut instance, &host, &step.encode().unwrap());
    instance.cancel().unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
