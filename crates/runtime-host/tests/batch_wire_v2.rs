use std::cell::RefCell;
use std::rc::Rc;

use runtime_bundle_abi::{BundleAbiError, OwnedLogitsRowV1, ResultSinkLimitsV1};
use runtime_bundle_sdk::{
    BatchAppendCopyV2, BatchAppendV1, BatchItemV1, BatchRequestV1,
    BatchWireVersionV1 as SdkBatchWireVersionV1,
};
use runtime_engine::{ModelBatchItem, ModelExecutor, RequestId};
use runtime_host::{BatchWireCapabilityV1, BundleExecutionV1};
use runtime_kv::{KvGeometry, KvPool, PreparedAppendBatch};
use runtime_scheduler::BatchPhase;

struct RecordingExecution {
    request: Rc<RefCell<Vec<u8>>>,
}

impl BundleExecutionV1 for RecordingExecution {
    fn execute_batch(
        &mut self,
        request: &[u8],
        _limits: ResultSinkLimitsV1,
    ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
        self.request.borrow_mut().extend_from_slice(request);
        Ok(Vec::new())
    }

    fn commit(&mut self) -> Result<(), BundleAbiError> {
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), BundleAbiError> {
        Ok(())
    }

    fn destroy(&mut self) -> Result<(), BundleAbiError> {
        Ok(())
    }

    fn is_poisoned(&self) -> bool {
        false
    }
}

fn cow_plan() -> PreparedAppendBatch {
    let geometry = KvGeometry::new(4, 4).expect("the COW fixture geometry is valid");
    let mut pool = KvPool::new(geometry, 2).expect("the COW fixture pool is valid");
    let source = pool
        .create_sequence(8)
        .expect("the source sequence fits the fixture");
    pool.append(source, 3)
        .expect("the partial source prefix fits the fixture");
    let child = pool
        .fork(source, 8)
        .expect("the child shares the partial source block");
    pool.prepare_append_batch(&[(child, 1)])
        .expect("the child append can be prepared")
}

fn execute_and_record(kv_plan: &PreparedAppendBatch, batch_wire: BatchWireCapabilityV1) -> Vec<u8> {
    let recorded = Rc::new(RefCell::new(Vec::new()));
    let mut executor = runtime_host::BundleModelExecutorV1::new_with_batch_wire(
        RecordingExecution {
            request: Rc::clone(&recorded),
        },
        16,
        1,
        batch_wire,
    );
    let items = [ModelBatchItem::new(
        RequestId::new(7),
        BatchPhase::Decode,
        vec![11],
        false,
    )];
    let transaction = executor
        .execute(&items, kv_plan)
        .expect("the recording execution accepts a rowless request");
    drop(transaction);
    recorded.borrow().clone()
}

fn sdk_request(version: SdkBatchWireVersionV1, kv_plan: &PreparedAppendBatch) -> BatchRequestV1 {
    let append = &kv_plan.appends()[0];
    BatchRequestV1::new(
        version,
        vec![BatchItemV1::new(7, true, 1, false, vec![11])],
        vec![BatchAppendV1::new(
            append.previous_tokens() as u64,
            append.additional_tokens() as u64,
            append.new_tokens() as u64,
            append
                .block_table()
                .iter()
                .map(|block| block.index() as u64)
                .collect(),
            append.copy().map(|copy| {
                BatchAppendCopyV2::new(
                    copy.source().index() as u64,
                    copy.destination().index() as u64,
                    copy.initialized_tokens() as u64,
                )
            }),
        )],
    )
}

#[test]
fn cow_append_requires_a_v2_record_with_copy_metadata() {
    let kv_plan = cow_plan();
    let copy = kv_plan.appends()[0]
        .copy()
        .expect("the prepared append exposes its COW work");
    let recorded = execute_and_record(&kv_plan, BatchWireCapabilityV1::V2CopyOnWrite);
    let expected = sdk_request(SdkBatchWireVersionV1::V2, &kv_plan)
        .encode()
        .expect("the independent SDK codec accepts the V2 fixture");

    assert_eq!(recorded, expected, "host and SDK V2 bytes must match");
    let decoded = BatchRequestV1::decode(&recorded).expect("the host V2 record decodes");
    assert_eq!(
        decoded.appends()[0].copy(),
        Some(BatchAppendCopyV2::new(
            copy.source().index() as u64,
            copy.destination().index() as u64,
            copy.initialized_tokens() as u64,
        ))
    );
}

#[test]
fn mixed_prefill_and_decode_tables_index_one_physical_pool() {
    let mut pool = KvPool::new(KvGeometry::new(4, 8).unwrap(), 2).unwrap();
    let prefill = pool.create_sequence(8).unwrap();
    let decode = pool.create_sequence(8).unwrap();
    pool.append(decode, 3).unwrap();
    let prepared = pool
        .prepare_append_batch(&[(prefill, 4), (decode, 1)])
        .unwrap();
    let items = [
        ModelBatchItem::new(
            RequestId::new(10),
            BatchPhase::Prefill { tokens: 4 },
            vec![1; 4],
            false,
        ),
        ModelBatchItem::new(RequestId::new(11), BatchPhase::Decode, vec![2], false),
    ];
    let recorded = Rc::new(RefCell::new(Vec::new()));
    let mut executor = runtime_host::BundleModelExecutorV1::new_with_batch_wire(
        RecordingExecution {
            request: Rc::clone(&recorded),
        },
        16,
        2,
        BatchWireCapabilityV1::V2CopyOnWrite,
    );
    drop(executor.execute(&items, &prepared).unwrap());
    let decoded = BatchRequestV1::decode(&recorded.borrow()).unwrap();
    assert_eq!(decoded.items().len(), 2);
    assert_eq!(decoded.appends().len(), 2);
    for (wire, append) in decoded.appends().iter().zip(prepared.appends()) {
        assert_eq!(wire.previous_tokens(), append.previous_tokens() as u64);
        assert_eq!(wire.new_tokens(), append.new_tokens() as u64);
        assert_eq!(wire.page_table().len(), append.block_table().len());
        for (page, block) in wire.page_table().iter().zip(append.block_table()) {
            assert_eq!(*page, block.index() as u64);
            assert!(*page < 8);
            assert!(*page * 1024 < 8 * 1024);
        }
    }
}

#[test]
fn v1_capability_refuses_to_drop_cow_metadata() {
    let kv_plan = cow_plan();
    let recorded = Rc::new(RefCell::new(Vec::new()));
    let mut executor = runtime_host::BundleModelExecutorV1::new(
        RecordingExecution {
            request: Rc::clone(&recorded),
        },
        16,
        1,
    );
    let items = [ModelBatchItem::new(
        RequestId::new(7),
        BatchPhase::Decode,
        vec![11],
        false,
    )];

    let error = executor
        .execute(&items, &kv_plan)
        .err()
        .expect("V1 cannot silently discard COW work");

    assert!(error.message().contains("chop.batch-wire.v2"));
    assert!(recorded.borrow().is_empty());
}

#[test]
fn v1_host_encoder_preserves_exact_sdk_byte_parity() {
    let geometry = KvGeometry::new(4, 2).expect("the V1 fixture geometry is valid");
    let mut pool = KvPool::new(geometry, 1).expect("the V1 fixture pool is valid");
    let sequence = pool.create_sequence(8).expect("the V1 sequence fits");
    let kv_plan = pool
        .prepare_append_batch(&[(sequence, 1)])
        .expect("the V1 append fits");
    let recorded = execute_and_record(&kv_plan, BatchWireCapabilityV1::V1);
    let expected = sdk_request(SdkBatchWireVersionV1::V1, &kv_plan)
        .encode()
        .expect("the independent SDK codec accepts the V1 fixture");

    assert_eq!(recorded, expected, "host and SDK V1 bytes must match");
    assert_eq!(&recorded[..12], b"chopbatch-v1");
}

#[test]
fn legacy_execution_retirement_is_a_noop_and_reports_no_callback() {
    let recorded = Rc::new(RefCell::new(Vec::new()));
    let mut executor = runtime_host::BundleModelExecutorV1::new(
        RecordingExecution {
            request: recorded.clone(),
        },
        3,
        1,
    );
    assert!(!executor.supports_request_retirement());
    executor
        .retire_request_v1(RequestId::new(u64::MAX))
        .unwrap();
    assert!(recorded.borrow().is_empty());
}
