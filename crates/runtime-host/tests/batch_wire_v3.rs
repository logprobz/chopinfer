use std::cell::RefCell;
use std::rc::Rc;

use runtime_bundle_abi::{BundleAbiError, OwnedLogitsRowV1, ResultSinkLimitsV1};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
use runtime_engine::{
    DraftSource, DraftWork, MediaEncodeV1, MediaInjectV1, ModelBatchItem, ModelExecutor, RequestId,
};
use runtime_host::{BatchWireCapabilityV1, BundleExecutionV1, BundleModelExecutorV1};
use runtime_kv::{ContextParallelPagesV1, KvGeometry, KvPool, RankPageV1};
use runtime_scheduler::BatchPhase;

/// A context-parallel group reads one wire: each append carries the logical
/// stripe table, which is every rank's own table of local pages, and the
/// bytes are the size-1 encoding of that table. Each group rank applies the
/// ownership rule the create record sent it to find its pages.
#[test]
fn every_context_parallel_rank_reads_its_pages_from_one_wire() {
    let pages = ContextParallelPagesV1::new(2, 4, 4).unwrap();
    let record = |geometry: KvGeometry| {
        let mut pool = KvPool::new(geometry, 1).unwrap();
        let sequence = pool.create_sequence(32).unwrap();
        let prepared = pool.prepare_append_batch(&[(sequence, 11)]).unwrap();
        let item = ModelBatchItem::new(
            RequestId::new(7),
            BatchPhase::Prefill { tokens: 11 },
            (0..11).collect(),
            true,
        );
        let recorded = Rc::new(RefCell::new(Vec::new()));
        let mut host = BundleModelExecutorV1::new_with_batch_wire(
            RecordingExecution(Rc::clone(&recorded), vec![7]),
            16,
            1,
            BatchWireCapabilityV1::V3Speculative,
        );
        drop(host.execute(&[item], &prepared).unwrap());
        let table = prepared.appends()[0]
            .block_table()
            .iter()
            .map(|page| page.index() as u64)
            .collect::<Vec<_>>();
        (recorded.take(), table)
    };
    let (striped, table) = record(pages.geometry(8).unwrap());
    // Two stripes of eight tokens: the same bytes a size-1 pool of
    // eight-token pages sends.
    assert_eq!(table.len(), 2);
    assert_eq!(striped, record(KvGeometry::new(8, 8).unwrap()).0);
    let wire = BatchRequestV1::decode(&striped).unwrap();
    assert_eq!(wire.version(), BatchWireVersionV1::V3);
    let append = &wire.appends()[0];
    assert_eq!(append.page_table(), table.as_slice());
    let rank_pages = |rank| {
        pages
            .rank_pages(
                rank,
                append.page_table().iter().map(|&page| page as usize),
                append.new_tokens() as usize,
            )
            .unwrap()
    };
    let page = |local, first_position, tokens| RankPageV1 {
        local_page: table[local] as usize,
        first_position,
        tokens,
    };
    assert_eq!(rank_pages(0), [page(0, 0, 4), page(1, 8, 3)]);
    assert_eq!(rank_pages(1), [page(0, 4, 4)]);
    assert_eq!(
        (0..2)
            .map(|rank| pages.rank_tokens(rank, 11))
            .sum::<usize>(),
        11
    );
}

struct RecordingExecution(Rc<RefCell<Vec<u8>>>, Vec<u64>);

impl BundleExecutionV1 for RecordingExecution {
    fn execute_batch(
        &mut self,
        request: &[u8],
        _: ResultSinkLimitsV1,
    ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
        self.0.borrow_mut().extend_from_slice(request);
        let logits = vec![0; 16 * 4];
        Ok(self
            .1
            .iter()
            .map(|&id| OwnedLogitsRowV1::new(id, logits.clone()))
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

#[test]
fn host_verify_wire_matches_the_independent_v3_codec() {
    let mut pool = KvPool::new(KvGeometry::new(4, 8).unwrap(), 1).unwrap();
    let sequence = pool.create_sequence(8).unwrap();
    let prepared = pool.prepare_append_batch(&[(sequence, 3)]).unwrap();
    let item = ModelBatchItem::new(
        RequestId::new(7),
        BatchPhase::Verify { draft_tokens: 2 },
        vec![3, 4, 5],
        true,
    )
    .with_draft(DraftWork::new(2, DraftSource::Host));
    let recorded = Rc::new(RefCell::new(Vec::new()));
    let mut host = BundleModelExecutorV1::new_with_batch_wire(
        RecordingExecution(Rc::clone(&recorded), vec![7; 3]),
        16,
        1,
        BatchWireCapabilityV1::V3Speculative,
    );
    drop(host.execute(&[item], &prepared).unwrap());
    let append = &prepared.appends()[0];
    let expected = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::verify(7, 3, false, vec![3, 4, 5])],
        vec![BatchAppendV1::new(
            append.previous_tokens() as u64,
            append.additional_tokens() as u64,
            append.new_tokens() as u64,
            append
                .block_table()
                .iter()
                .map(|page| page.index() as u64)
                .collect(),
            None,
        )],
    )
    .encode()
    .unwrap();
    assert_eq!(*recorded.borrow(), expected);
    let decoded = BatchRequestV1::decode(&expected).unwrap();
    assert!(decoded.items()[0].is_verify());
    assert!(decoded.media().encodes.is_empty());
}

#[test]
fn media_wire_carries_batch_indices_offsets_and_payloads() {
    let mut pool = KvPool::new(KvGeometry::new(4, 8).unwrap(), 2).unwrap();
    let first = pool.create_sequence(8).unwrap();
    let second = pool.create_sequence(8).unwrap();
    let prepared = pool
        .prepare_append_batch(&[(first, 1), (second, 1)])
        .unwrap();
    let items = [
        ModelBatchItem::new(
            RequestId::new(7),
            BatchPhase::Prefill { tokens: 1 },
            vec![3],
            true,
        )
        .with_media(
            vec![MediaEncodeV1 {
                encoder: 0,
                geometry: [1, 2, 3, 4],
                output_rows: 1,
                pages: vec![2],
                payload: [11_u8, 12].to_vec().into(),
            }],
            vec![MediaInjectV1 {
                item_row: 0,
                rows: 1,
                arena_row: 8,
            }],
        ),
        ModelBatchItem::new(
            RequestId::new(8),
            BatchPhase::Prefill { tokens: 1 },
            vec![4],
            true,
        )
        .with_media(
            vec![MediaEncodeV1 {
                encoder: 1,
                geometry: [4, 3, 2, 1],
                output_rows: 1,
                pages: vec![3],
                payload: [13_u8].to_vec().into(),
            }],
            vec![MediaInjectV1 {
                item_row: 0,
                rows: 1,
                arena_row: 12,
            }],
        ),
    ];
    let recorded = Rc::new(RefCell::new(Vec::new()));
    let mut host = BundleModelExecutorV1::new_with_batch_wire(
        RecordingExecution(Rc::clone(&recorded), vec![7, 8]),
        16,
        2,
        BatchWireCapabilityV1::V3Media,
    );
    drop(host.execute(&items, &prepared).unwrap());
    let wire = BatchRequestV1::decode(&recorded.borrow()).unwrap();
    assert_eq!(wire.media().payload, [11, 12, 13]);
    assert_eq!(
        wire.media()
            .encodes
            .iter()
            .map(|record| record.payload_offset)
            .collect::<Vec<_>>(),
        [0, 2]
    );
    assert_eq!(
        wire.media()
            .injects
            .iter()
            .map(|record| record.item_index)
            .collect::<Vec<_>>(),
        [0, 1]
    );
}

#[test]
fn media_records_require_a_media_capable_v3_bundle() {
    let mut pool = KvPool::new(KvGeometry::new(4, 8).unwrap(), 1).unwrap();
    let sequence = pool.create_sequence(8).unwrap();
    let prepared = pool.prepare_append_batch(&[(sequence, 1)]).unwrap();
    let item = ModelBatchItem::new(
        RequestId::new(7),
        BatchPhase::Prefill { tokens: 1 },
        vec![3],
        true,
    )
    .with_media(
        vec![MediaEncodeV1 {
            encoder: 0,
            geometry: [1, 1, 1, 0],
            output_rows: 1,
            pages: vec![0],
            payload: [1_u8].to_vec().into(),
        }],
        vec![MediaInjectV1 {
            item_row: 0,
            rows: 1,
            arena_row: 0,
        }],
    );
    for wire in [
        BatchWireCapabilityV1::V2PromptRows,
        BatchWireCapabilityV1::V3Speculative,
    ] {
        let mut host = BundleModelExecutorV1::new_with_batch_wire(
            RecordingExecution(Rc::new(RefCell::new(Vec::new())), vec![7]),
            16,
            1,
            wire,
        );
        assert!(
            host.execute(std::slice::from_ref(&item), &prepared)
                .is_err()
        );
    }
}
