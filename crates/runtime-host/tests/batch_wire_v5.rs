//! Explicit 3-axis positions cross the ABI as `chopbatch-v5`, only for a
//! bundle that declared `chop.position-axes.v1`, and only in steps that carry
//! them.

use std::cell::RefCell;
use std::rc::Rc;

use runtime_bundle_abi::{BundleAbiError, OwnedLogitsRowV1, ResultSinkLimitsV1};
use runtime_bundle_sdk::{BATCH_REQUEST_MAGIC_V3, BATCH_REQUEST_MAGIC_V5, BatchRequestV1};
use runtime_engine::{
    Engine, EngineConfig, EngineRequest, MediaConfigV1, MediaInputV1, MediaSpanV1, ModelBatchItem,
    ModelExecutor, PositionAxesV1, PositionSpanV1, RequestId,
};
use runtime_host::{BatchWireCapabilityV1, BundleExecutionV1, BundleModelExecutorV1};
use runtime_kv::{KvGeometry, KvPool};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::{BatchPhase, SchedulerConfig};

const VOCABULARY: usize = 16;

/// Records every request and emits one zero row per logits row it asks for.
#[derive(Default)]
struct RecordingBundle(Rc<RefCell<Vec<Vec<u8>>>>);

impl BundleExecutionV1 for RecordingBundle {
    fn execute_batch(
        &mut self,
        request: &[u8],
        _: ResultSinkLimitsV1,
    ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
        self.0.borrow_mut().push(request.to_vec());
        let batch = BatchRequestV1::decode(request).expect("the host encodes a readable batch");
        Ok(batch
            .items()
            .iter()
            .filter(|item| item.expects_logits())
            .flat_map(|item| {
                let rows = if item.is_verify() {
                    item.phase_tokens() as usize
                } else {
                    1
                };
                std::iter::repeat_n(item.request_id(), rows)
            })
            .map(|id| OwnedLogitsRowV1::new(id, vec![0; VOCABULARY * 4]))
            .collect())
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

fn items() -> Vec<ModelBatchItem> {
    vec![
        ModelBatchItem::new(
            RequestId::new(7),
            BatchPhase::Prefill { tokens: 2 },
            vec![3, 4],
            true,
        )
        .with_position_axes(vec![[0, 0, 0], [1, 1, 2]]),
        ModelBatchItem::new(RequestId::new(8), BatchPhase::Decode, vec![5], true),
    ]
}

fn run(
    wire: BatchWireCapabilityV1,
    capability: bool,
    items: &[ModelBatchItem],
) -> (Result<(), String>, Vec<Vec<u8>>) {
    let mut pool = KvPool::new(KvGeometry::new(4, 8).unwrap(), 2).unwrap();
    let first = pool.create_sequence(8).unwrap();
    let second = pool.create_sequence(8).unwrap();
    let prepared = pool
        .prepare_append_batch(&[(first, 2), (second, 1)])
        .unwrap();
    let recorded = Rc::new(RefCell::new(Vec::new()));
    let mut host = BundleModelExecutorV1::new_with_batch_wire(
        RecordingBundle(Rc::clone(&recorded)),
        VOCABULARY,
        2,
        wire,
    )
    .with_position_axes(capability);
    let result = host
        .execute(items, &prepared)
        .map(drop)
        .map_err(|error| error.message().to_owned());
    let recorded = recorded.borrow().clone();
    (result, recorded)
}

#[test]
fn explicit_positions_ride_v5_around_the_enclosed_v3_record() {
    let (result, recorded) = run(BatchWireCapabilityV1::V3Media, true, &items());
    result.unwrap();
    assert!(recorded[0].starts_with(BATCH_REQUEST_MAGIC_V5));
    let batch = BatchRequestV1::decode(&recorded[0]).unwrap();
    assert_eq!(batch.items()[0].position_axes(), [[0, 0, 0], [1, 1, 2]]);
    assert!(batch.items()[1].position_axes().is_empty());
    // The enclosed record is exactly the step's V3 record without positions.
    let plain = items()
        .into_iter()
        .map(|item| item.with_position_axes(Vec::new()))
        .collect::<Vec<_>>();
    let (_, unwrapped) = run(BatchWireCapabilityV1::V3Media, true, &plain);
    assert!(unwrapped[0].starts_with(BATCH_REQUEST_MAGIC_V3));
    assert_eq!(&recorded[0][16..16 + unwrapped[0].len()], unwrapped[0]);
    // Steps without positions keep their bytes whether or not the bundle
    // could read them.
    let (_, without) = run(BatchWireCapabilityV1::V3Media, false, &plain);
    assert_eq!(unwrapped, without);
}

#[test]
fn explicit_positions_need_the_capability_v3_and_whole_items() {
    let (result, recorded) = run(BatchWireCapabilityV1::V3Media, false, &items());
    assert!(result.unwrap_err().contains("chop.position-axes.v1"));
    assert!(
        recorded.is_empty(),
        "a refused step never reaches the bundle"
    );
    let (result, recorded) = run(BatchWireCapabilityV1::V2PromptRows, true, &items());
    assert!(result.unwrap_err().contains("V3"));
    assert!(recorded.is_empty());
    let mut partial = items();
    partial[0] = partial[0].clone().with_position_axes(vec![[0, 0, 0]]);
    let (result, recorded) = run(BatchWireCapabilityV1::V3Media, true, &partial);
    assert!(result.is_err());
    assert!(recorded.is_empty());
}

#[test]
fn an_engine_step_sequence_sends_every_row_its_rule_position() {
    let vocabulary = ModelVocabulary::new(VOCABULARY).unwrap();
    let config = EngineConfig::new(
        SchedulerConfig::new(4, 64, 32, 5, 1).unwrap(),
        KvGeometry::new(4, 64).unwrap(),
        vocabulary,
        128,
        64,
    )
    .unwrap()
    .with_media(
        MediaConfigV1::new(16, 4)
            .unwrap()
            .with_encoder(1 << 20, 1)
            .unwrap(),
    )
    .with_position_axes();
    let recorded = Rc::new(RefCell::new(Vec::new()));
    let executor = BundleModelExecutorV1::new_with_batch_wire(
        RecordingBundle(Rc::clone(&recorded)),
        VOCABULARY,
        8,
        BatchWireCapabilityV1::V3Media,
    )
    .with_position_axes(true);
    let mut engine = Engine::new(config, executor).unwrap();
    // "a b", a 2 x 3 grid of merged blocks, "c".
    let mut prompt = vec![1, 2];
    prompt.extend([9; 6]);
    prompt.push(3);
    let span = PositionSpanV1 {
        prompt_row: 2,
        rows: 6,
        grid: [1, 2, 3],
    };
    let request = EngineRequest::new(
        RequestId::new(1),
        prompt,
        3,
        SamplingParameters::new(0.0).unwrap(),
        1,
        Vec::new(),
        vocabulary,
    )
    .unwrap()
    .with_media(
        vec![MediaInputV1 {
            key: [1; 32],
            encoder: 0,
            geometry: [1, 4, 6, 0],
            output_rows: 6,
            payload: vec![1_u8; 24].into(),
        }],
        vec![MediaSpanV1 {
            media: 0,
            prompt_row: 2,
            rows: 6,
            media_row: 0,
        }],
    )
    .unwrap()
    .with_position_axes(vec![span])
    .unwrap();
    engine.submit(request).unwrap();
    engine
        .submit(
            EngineRequest::new(
                RequestId::new(2),
                vec![1, 2, 3],
                2,
                SamplingParameters::new(0.0).unwrap(),
                2,
                Vec::new(),
                vocabulary,
            )
            .unwrap(),
        )
        .unwrap();
    engine.run_until_idle(64).unwrap();
    let axes = PositionAxesV1::new(9, 12, &[span]).unwrap();
    let mut seen = Vec::new();
    for bytes in recorded.borrow().iter() {
        let batch = BatchRequestV1::decode(bytes).unwrap();
        for (item, append) in batch.items().iter().zip(batch.appends()) {
            if item.request_id() == 2 {
                assert!(item.position_axes().is_empty());
                continue;
            }
            let first = append.previous_tokens() as usize;
            assert_eq!(
                item.position_axes(),
                axes.rows(first, item.phase_tokens() as usize)
            );
            seen.extend_from_slice(item.position_axes());
        }
    }
    // Nine prompt rows in two chunks, then two generated inputs.
    assert_eq!(
        seen,
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 3],
            [2, 2, 4],
            [2, 3, 2],
            [2, 3, 3],
            [2, 3, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
        ]
    );
}
