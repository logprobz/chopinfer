//! Allocated verify steps (`chopbatch-v7`): an instance prepares their ops
//! and storage only when its host allocates and refuses them otherwise; an
//! allocating instance runs them on the device results path, allocating on
//! every rank after the draft copies and before the hash, computing the
//! width eagerly, gathering the head and feature rows back into the
//! reserved layout and walking the verified drafts.
use super::results::{DeviceSink, metadata};
use super::*;
use runtime_bundle_sdk::{SamplingKindV1, VerifyAllocationV1, VerifyCalibrationV1};

/// An instance of `items` slots whose host allocates.
fn create_allocating(shape: Shape, items: u32) -> (Rc<Host>, BundleInstanceV1) {
    use model_deepseek41_contract::verify_allocation::OPTION;
    use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1};
    let declared = ModelOptionsV1 {
        schema: CAPABILITY.into(),
        options: shape
            .geometry
            .model_options(shape.geometry.runtime_defaults()),
    };
    let resolved = declared
        .resolve(
            &[(OPTION.to_owned(), "true".to_owned())]
                .into_iter()
                .collect(),
        )
        .unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.drafter.set(true);
    host.speculation.set(Some((5, 32)));
    let record = crate::config::tests::with_options(
        crate::config::tests::record(&shape, host.kv, items, false),
        &resolved,
    );
    let instance = api()
        .create(
            HostServicesTableV1::new(Rc::new(MappedHost::new(Rc::clone(&host)))),
            &record,
        )
        .unwrap_or_else(|e| panic!("drafter create: {e:?} {:?}", host.logs.borrow()));
    (host, instance)
}

/// `items` verify items reserving `k` drafts each from position `start`,
/// allocated to `width` rows.
fn allocated(items: u32, start: u64, k: u32, width: u32) -> BatchRequestV1 {
    let calibration = VerifyCalibrationV1::new(&[(1.0, 0.0); 5], &[(1.0, 0.0); 5]).unwrap();
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..items)
            .map(|slot| BatchItemV1::verify(u64::from(slot + 1), k + 1, true, vec![7]))
            .collect(),
        (0..items)
            .map(|slot| {
                BatchAppendV1::new(
                    start,
                    u64::from(k + 1),
                    start + u64::from(k + 1),
                    vec![u64::from(slot)],
                    None,
                )
            })
            .collect(),
    )
    .with_state_slots((0..items).map(u64::from).collect())
    .with_verify_allocation(VerifyAllocationV1::new(
        width,
        1,
        calibration,
        vec![SamplingKindV1::Sampled; items as usize],
    ))
}

/// The position in `step` of the first launch of `symbol` on `stream`.
fn launch_of(step: &[String], stream: u32, symbol: &str) -> Option<usize> {
    step.iter()
        .position(|line| line.starts_with(&format!("launch {stream} {symbol} ")))
}

/// An allocated step runs on every topology: each rank allocates after its
/// draft copies and before rank 0 hashes the computed rows, the target
/// computes the width eagerly (no replay), every rank gathers its head and
/// packed feature rows back into the reserved rows by index, the sampler
/// reads the reserved rows, and the allocated acceptance walks the verified
/// drafts. A host without device results is refused before any work.
#[test]
fn allocated_steps_compute_the_width_and_keep_the_reserved_layout() {
    for shape in crate::shape::topologies() {
        let (items, k, width) = (8_u32, 3_u32, 16_u32);
        let (host, mut instance) = create_allocating(shape, 8);
        for slot in 0..items {
            execute(
                &mut instance,
                &host,
                &super::super::prompt(u64::from(slot + 1), u64::from(slot), 0, 3, true),
            );
            instance.commit().unwrap();
        }
        let request = allocated(items, 3, k, width).encode().unwrap();
        let mut legacy = Sink::default();
        assert_eq!(
            instance.execute(&request, &mut legacy).err(),
            Some(BundleAbiError::Status(BundleStatusV1::Unsupported))
        );
        let mut sink = DeviceSink {
            host: Rc::clone(&host),
            metadata: Some(metadata(items, k, 1)),
            legacy: Sink::default(),
            drafts: Vec::new(),
            samples: Vec::new(),
            completion: None,
        };
        let start = host.journal.borrow().len();
        instance
            .execute(&request, &mut sink)
            .unwrap_or_else(|e| panic!("{e:?} {:?}", host.logs.borrow()));
        assert_eq!(sink.drafts.len(), items as usize);
        assert_eq!(
            sink.samples
                .iter()
                .map(|s| s.logits.logits.row_count)
                .sum::<u32>(),
            items * (k + 1)
        );
        let packet = sink.completion.unwrap();
        assert_eq!(packet.sample_rows, items * (k + 1));
        let journal = host.journal.borrow();
        let step = &journal[start..];
        let streams = host.rank_streams.borrow().clone();
        let hash = launch_of(step, streams[&0], crate::sm120::engram_hash_rows::SYMBOL)
            .expect("rank 0 hashes the computed rows");
        let acceptance = launch_of(
            step,
            streams[&0],
            crate::sm120::allocated_acceptance_entries::SYMBOL,
        )
        .expect("the allocated acceptance walks the verified drafts");
        assert!(
            launch_of(
                step,
                streams[&0],
                crate::sm120::host_acceptance_entries::SYMBOL
            )
            .is_none()
        );
        // The target runs eagerly between the hash and the acceptance.
        assert!(
            !step[hash..acceptance]
                .iter()
                .any(|line| line.starts_with("replay "))
        );
        for rank in 0..shape.ranks() {
            let stream = streams[&rank];
            let allocate = launch_of(step, stream, crate::sm120::allocate_entries::SYMBOL)
                .unwrap_or_else(|| panic!("rank {rank} allocates"));
            assert!(allocate < hash);
            let prefix = format!("launch {stream} ");
            let head = step[hash..acceptance]
                .iter()
                .position(|line| {
                    line.starts_with(&prefix) && line.contains("chop_deepseek41_head_f32")
                })
                .map(|offset| hash + offset)
                .unwrap_or_else(|| panic!("rank {rank} runs its head"));
            // The feature packing's gather and the packed rows' gather into
            // the reserved rows, then the reserved head rows' gather.
            let gathers = step[hash..head]
                .iter()
                .filter(|line| {
                    line.starts_with(&prefix) && line.contains("chop_row_gather_by_index")
                })
                .count();
            assert_eq!(gathers, 3, "rank {rank}");
        }
        drop(journal);
        instance.cancel().unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn allocated_verify_steps_are_refused_without_the_hosts_option() {
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
