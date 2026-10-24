//! Allocated verify steps: an instance prepares their ops and storage only
//! when its host allocates, and until the drafter wires them an allocated
//! verify step (`chopbatch-v7`) is refused before any work, while the same
//! step without its allocation runs.
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

/// The create record's verify_allocation option makes the drafter prepare
/// the allocation and the allocated acceptance, hold their storage on every
/// rank and load their cubins; without it the instance holds none of them.
#[test]
fn an_allocating_host_gets_the_allocation_ops_and_storage() {
    use model_deepseek41_contract::verify_allocation::OPTION;
    use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1};

    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let declared = ModelOptionsV1 {
        schema: CAPABILITY.into(),
        options: shape
            .geometry
            .model_options(shape.geometry.runtime_defaults()),
    };
    let symbols = [
        crate::sm120::allocate_entries::SYMBOL,
        crate::sm120::allocated_acceptance_entries::SYMBOL,
    ];
    for allocates in [false, true] {
        let overrides = if allocates {
            vec![(OPTION.to_owned(), "true".to_owned())]
        } else {
            vec![]
        };
        let resolved = declared.resolve(&overrides.into_iter().collect()).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.drafter.set(true);
        host.speculation.set(Some((5, 32)));
        let record = crate::config::tests::with_options(
            crate::config::tests::record(&shape, host.kv, 4, false),
            &resolved,
        );
        let mut instance = api()
            .create(
                HostServicesTableV1::new(Rc::new(MappedHost::new(Rc::clone(&host)))),
                &record,
            )
            .unwrap_or_else(|e| panic!("drafter create: {e:?} {:?}", host.logs.borrow()));
        assert_eq!(
            crate::instance::dspark_allocation_ranks(instance.handle()),
            allocates.then_some(4)
        );
        for symbol in symbols {
            let resolved = host
                .symbol_names
                .borrow()
                .values()
                .filter(|name| name.as_str() == symbol)
                .count();
            assert_eq!(resolved, if allocates { 4 } else { 0 }, "{symbol}");
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
