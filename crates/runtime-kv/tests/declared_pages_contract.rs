#![forbid(unsafe_code)]

use runtime_kv::{
    DeclaredPageLayoutError, DeclaredPageLayoutV1, KvRankV1, PagedKvErrorV1, PagedKvPlanV1,
    PhysicalPageIdV1, RankOwnershipQualification, RankPlacementV1, StateLifetimeV1,
};

/// A declared layout owned by this test, not by the shared KV crate.
const BYTES_PER_TOKEN: u32 = 656;
const SEGMENTS_PER_PAGE: u32 = 43;
const PAGE_TOKENS: u32 = 256;
const ALIGNMENT_BYTES: u64 = 16;
const RANK_COUNT: u8 = 4;
const MAX_SEQUENCE_TOKENS: usize = 1_048_576;

fn layout() -> DeclaredPageLayoutV1 {
    DeclaredPageLayoutV1::new(
        BYTES_PER_TOKEN,
        SEGMENTS_PER_PAGE,
        PAGE_TOKENS,
        ALIGNMENT_BYTES,
        RANK_COUNT,
        RankPlacementV1::RankLocal,
        StateLifetimeV1::Request,
        MAX_SEQUENCE_TOKENS,
    )
    .unwrap()
}

fn plan(page_capacity: u64) -> PagedKvPlanV1 {
    PagedKvPlanV1::validate(layout(), page_capacity, 4).unwrap()
}

fn rank(index: u8) -> KvRankV1 {
    plan(1).rank(index).unwrap()
}

#[test]
fn declared_geometry_and_rank_pools_are_exact() {
    let plan = plan(3);
    let layout = layout();
    assert_eq!(layout.bytes_per_token(), 656);
    assert_eq!(layout.page_tokens(), 256);
    assert_eq!(layout.segments_per_page(), 43);
    assert_eq!(layout.rank_count(), 4);
    assert_eq!(layout.alignment_bytes(), 16);
    assert_eq!(layout.rank_placement(), RankPlacementV1::RankLocal);
    assert_eq!(layout.state_lifetime(), StateLifetimeV1::Request);
    assert_eq!(layout.maximum_sequence_tokens(), 1_048_576);
    assert_eq!(layout.maximum_logical_pages(), 4_096);
    assert_eq!(plan.bytes_per_segment_page(), 167_936);
    assert_eq!(plan.bytes_per_page(), 7_221_248);
    assert_eq!(plan.bytes_per_rank_pool(), 21_663_744);
    assert_eq!(plan.total_bytes(), 86_654_976);
    assert_eq!(
        plan.ownership_qualification(),
        RankOwnershipQualification::Unqualified
    );
    let segments = plan.segment_specs();
    assert_eq!(segments.len(), 43);
    for (index, spec) in segments.into_iter().enumerate() {
        assert_eq!(spec.segment(), index);
        assert_eq!(spec.bytes_per_token(), 656);
        assert_eq!(spec.page_tokens(), 256);
        assert_eq!(spec.segment_bytes(), 167_936);
    }

    let ranks = plan.rank_pools();
    assert_eq!(
        ranks
            .iter()
            .map(|pool| pool.rank().index())
            .collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert_eq!(
        ranks
            .iter()
            .map(|pool| pool.base_offset())
            .collect::<Vec<_>>(),
        [0, 21_663_744, 43_327_488, 64_991_232]
    );
}

#[test]
fn a_malformed_declared_layout_is_rejected() {
    let cases = [
        (
            DeclaredPageLayoutV1::new(
                0,
                SEGMENTS_PER_PAGE,
                PAGE_TOKENS,
                ALIGNMENT_BYTES,
                RANK_COUNT,
                RankPlacementV1::RankLocal,
                StateLifetimeV1::Request,
                MAX_SEQUENCE_TOKENS,
            ),
            DeclaredPageLayoutError::ZeroBytesPerToken,
        ),
        (
            DeclaredPageLayoutV1::new(
                BYTES_PER_TOKEN,
                0,
                PAGE_TOKENS,
                ALIGNMENT_BYTES,
                RANK_COUNT,
                RankPlacementV1::RankLocal,
                StateLifetimeV1::Request,
                MAX_SEQUENCE_TOKENS,
            ),
            DeclaredPageLayoutError::ZeroSegmentsPerPage,
        ),
        (
            DeclaredPageLayoutV1::new(
                BYTES_PER_TOKEN,
                SEGMENTS_PER_PAGE,
                0,
                ALIGNMENT_BYTES,
                RANK_COUNT,
                RankPlacementV1::RankLocal,
                StateLifetimeV1::Request,
                MAX_SEQUENCE_TOKENS,
            ),
            DeclaredPageLayoutError::ZeroPageTokens,
        ),
        (
            DeclaredPageLayoutV1::new(
                BYTES_PER_TOKEN,
                SEGMENTS_PER_PAGE,
                PAGE_TOKENS,
                24,
                RANK_COUNT,
                RankPlacementV1::RankLocal,
                StateLifetimeV1::Request,
                MAX_SEQUENCE_TOKENS,
            ),
            DeclaredPageLayoutError::InvalidAlignment {
                alignment_bytes: 24,
            },
        ),
        (
            DeclaredPageLayoutV1::new(
                BYTES_PER_TOKEN,
                SEGMENTS_PER_PAGE,
                PAGE_TOKENS,
                ALIGNMENT_BYTES,
                0,
                RankPlacementV1::RankLocal,
                StateLifetimeV1::Request,
                MAX_SEQUENCE_TOKENS,
            ),
            DeclaredPageLayoutError::RankCountOutOfRange { rank_count: 0 },
        ),
        (
            DeclaredPageLayoutV1::new(
                BYTES_PER_TOKEN,
                SEGMENTS_PER_PAGE,
                PAGE_TOKENS,
                ALIGNMENT_BYTES,
                RANK_COUNT,
                RankPlacementV1::RankLocal,
                StateLifetimeV1::Request,
                0,
            ),
            DeclaredPageLayoutError::ZeroMaximumSequenceTokens,
        ),
        (
            DeclaredPageLayoutV1::new(
                3,
                1,
                1,
                16,
                RANK_COUNT,
                RankPlacementV1::Replicated,
                StateLifetimeV1::Process,
                MAX_SEQUENCE_TOKENS,
            ),
            DeclaredPageLayoutError::UnalignedPage {
                page_bytes: 3,
                alignment_bytes: 16,
            },
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, Err(expected));
    }
}

#[test]
fn validation_rejects_size_overflow_and_invalid_rank() {
    assert_eq!(
        PagedKvPlanV1::validate(layout(), u64::MAX, 1),
        Err(PagedKvErrorV1::ArithmeticOverflow)
    );
    assert_eq!(
        plan(1).rank(RANK_COUNT),
        Err(PagedKvErrorV1::RankOutOfRange { rank: RANK_COUNT })
    );
}

#[test]
fn page_coverage_and_offsets_are_checked() {
    let plan = plan(4);
    assert_eq!(plan.logical_pages_for_tokens(0).unwrap(), 0);
    assert_eq!(plan.logical_pages_for_tokens(256).unwrap(), 1);
    assert_eq!(plan.logical_pages_for_tokens(257).unwrap(), 2);
    assert_eq!(
        plan.logical_pages_for_tokens(MAX_SEQUENCE_TOKENS + 1),
        Err(PagedKvErrorV1::SequenceTokenOutOfRange {
            tokens: MAX_SEQUENCE_TOKENS + 1,
            maximum: MAX_SEQUENCE_TOKENS,
        })
    );

    plan.validate_pages(&[PhysicalPageIdV1::new(3), PhysicalPageIdV1::new(1)])
        .unwrap();
    let first = plan.row_offset(PhysicalPageIdV1::new(3), 0, 0).unwrap();
    let last = plan.row_offset(PhysicalPageIdV1::new(1), 42, 255).unwrap();
    assert_eq!(first % u64::from(BYTES_PER_TOKEN), 0);
    assert_eq!(last % u64::from(BYTES_PER_TOKEN), 0);
    assert_eq!(
        plan.page_offset(PhysicalPageIdV1::new(1)).unwrap(),
        7_221_248
    );
    assert_eq!(
        plan.aggregate_page_offset(rank(3), PhysicalPageIdV1::new(1))
            .unwrap(),
        93_876_224
    );

    assert_eq!(
        plan.row_offset(PhysicalPageIdV1::new(0), 0, PAGE_TOKENS as usize),
        Err(PagedKvErrorV1::TokenInPageOutOfRange {
            token: PAGE_TOKENS as usize,
            page_tokens: PAGE_TOKENS as usize,
        })
    );
    assert_eq!(
        plan.row_offset(PhysicalPageIdV1::new(0), SEGMENTS_PER_PAGE as usize, 0),
        Err(PagedKvErrorV1::SegmentOutOfRange {
            segment: SEGMENTS_PER_PAGE as usize,
            segment_count: SEGMENTS_PER_PAGE as usize,
        })
    );
    assert_eq!(
        plan.page_offset(PhysicalPageIdV1::new(4)),
        Err(PagedKvErrorV1::PageOutOfRange {
            page: PhysicalPageIdV1::new(4),
            capacity: 4,
        })
    );

    assert_eq!(
        plan.validate_pages(&[PhysicalPageIdV1::new(1), PhysicalPageIdV1::new(1)]),
        Err(PagedKvErrorV1::DuplicatePage {
            page: PhysicalPageIdV1::new(1)
        })
    );
}
