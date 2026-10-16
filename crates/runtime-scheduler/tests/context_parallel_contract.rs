use runtime_kv::ContextParallelPagesV1;
use runtime_scheduler::{
    AdmissionPolicy, Batch, BatchOutcome, InvalidRequest, RequestId, RequestSpec, Scheduler,
    SchedulerConfig, SubmitError,
};

const PAGES_PER_RANK: usize = 8;
const PAGE_TOKENS: usize = 16;

fn scheduler(size: usize) -> Scheduler {
    let pages = ContextParallelPagesV1::new(size, PAGE_TOKENS, 4).unwrap();
    Scheduler::new(
        SchedulerConfig::new(4, 8, 512, 256, 1)
            .unwrap()
            .with_admission(AdmissionPolicy::Reserved),
        pages.geometry(PAGES_PER_RANK).unwrap(),
    )
    .unwrap()
}

fn commit(s: &mut Scheduler, batch: Batch) {
    let outcomes: Vec<_> = batch
        .items()
        .iter()
        .map(|item| {
            BatchOutcome::success(
                item.request_id(),
                item.phase(),
                usize::from(item.expects_output()),
                false,
            )
        })
        .collect();
    s.commit_batch(batch.id(), &outcomes).unwrap();
    s.audit().unwrap();
}

#[test]
fn reservations_count_stripes_so_capacity_scales_with_the_group() {
    // 199 rows (the last token is emitted, not written) need 13 pages; one
    // rank holds 8.
    let refused = scheduler(1).submit(RequestSpec::new(RequestId::new(1), 100, 100).unwrap());
    assert_eq!(
        refused,
        Err(SubmitError::Invalid(InvalidRequest::ExceedsKvPool {
            maximum: PAGES_PER_RANK * PAGE_TOKENS,
            requested: 199,
        }))
    );

    // At size 2 the same pages per rank hold 256 rows. The reservation is
    // 7 stripes, which is the 7 pages every group rank holds at most.
    let mut s = scheduler(2);
    s.submit(RequestSpec::new(RequestId::new(1), 100, 100).unwrap())
        .unwrap();
    s.admit().unwrap();
    assert_eq!(s.reserved_pages(), 7);
    while let Some(batch) = s.next_batch().unwrap() {
        commit(&mut s, batch);
        assert!(s.kv_stats().used_blocks() <= 7);
    }
    assert_eq!(s.kv_stats().active_sequences(), 0);
    assert_eq!(s.reserved_pages(), 0);
}
