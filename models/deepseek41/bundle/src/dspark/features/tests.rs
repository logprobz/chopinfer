use super::*;
use crate::dspark::state::{CommitItem, Kind, Rows, Snapshot};
use crate::dspark::transfers::{retain_copies, snapshot_copies};
use model_deepseek41_contract::geometry::FLASH;
use runtime_bundle_abi::BundleDeviceBufferHandleV1;

fn setup() -> (Config, Layout, Span, Span) {
    let config = Config {
        slots: 3,
        stages: FLASH.dspark_stages(),
        window: FLASH.shape.attention.window,
        max_prefill_rows: 8192,
        context_tokens: FLASH.context_tokens(),
        position_capacity: FLASH.context_tokens(),
    };
    let layout = Layout::new(&FLASH, config).unwrap();
    let retained = span(1, 256, layout.retained_bytes().unwrap());
    let staged = span(2, 1024, 8192 * layout.row_bytes());
    (config, layout, retained, staged)
}

fn span(id: u32, offset: u64, length: u64) -> Span {
    Span::new(
        BundleDeviceBufferHandleV1::from_parts(id, 1),
        offset,
        length,
    )
}

#[test]
fn accepted_rows_bind_to_the_right_slot_across_a_history_wrap() {
    let (config, layout, retained, staged) = setup();
    let bound = layout.bind(retained, staged).unwrap();
    let start = config.history_capacity() - 2;
    assert!(bound.retain(&[]).unwrap().is_empty());
    // Acceptance depths 0..5 retain the bonus row plus 0..5 proposals.
    for kept in 1..=6 {
        let item = CommitItem {
            rows: Rows {
                request: 17,
                slot: 2,
                start,
                kind: Kind::Decode { drafts: 5 },
            },
            kept,
            project_from_staging: false,
            retain_offset: 0,
        };
        let plans = retain_copies(config, item, 7).unwrap();
        let copies = bound.retain(&plans).unwrap();
        let pairs: Vec<_> = copies
            .iter()
            .flat_map(|c| {
                assert_eq!(c.source.buffer, staged.buffer);
                assert_eq!(c.destination.buffer, retained.buffer);
                assert_eq!(c.source.length, c.destination.length);
                (0..c.source.length / layout.row_bytes())
                    .map(|i| {
                        (
                            (c.source.offset - staged.offset) / layout.row_bytes() + i,
                            (c.destination.offset - retained.offset) / layout.row_bytes() + i,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let expected: Vec<_> = (0..u64::from(kept))
            .map(|i| {
                (
                    7 + i,
                    2 * config.history_capacity() + (start + i) % config.history_capacity(),
                )
            })
            .collect();
        assert_eq!(pairs, expected);
    }
}

#[test]
fn snapshot_views_keep_chronological_rows_when_restoring_another_slot() {
    let (config, layout, retained, _) = setup();
    let bound = layout.bind(retained, span(0, 0, 0)).unwrap();
    let end = config.history_capacity() + 3;
    let snapshot = Snapshot {
        start: end - u64::from(config.window),
        end,
        window: config.window,
        stages: config.stages,
    };
    let saved = span(3, 2048, u64::from(config.window) * layout.row_bytes());
    let capture = bound
        .capture(&snapshot_copies(config, 2, snapshot).unwrap(), saved)
        .unwrap();
    let restore = bound
        .restore(&snapshot_copies(config, 0, snapshot).unwrap(), saved)
        .unwrap();
    let mut slots = vec![u64::MAX; 3 * config.history_capacity() as usize];
    for p in snapshot.start..snapshot.end {
        slots[config.history_cell(2, p) as usize] = p;
    }
    let mut compact = vec![u64::MAX; config.window as usize];
    for copy in capture {
        let src = (copy.source.offset - retained.offset) / layout.row_bytes();
        let dst = (copy.destination.offset - saved.offset) / layout.row_bytes();
        let rows = copy.source.length / layout.row_bytes();
        for i in 0..rows {
            compact[(dst + i) as usize] = slots[(src + i) as usize];
        }
    }
    assert_eq!(compact, (snapshot.start..snapshot.end).collect::<Vec<_>>());
    for copy in restore {
        let src = (copy.source.offset - saved.offset) / layout.row_bytes();
        let dst = (copy.destination.offset - retained.offset) / layout.row_bytes();
        let rows = copy.source.length / layout.row_bytes();
        for i in 0..rows {
            slots[(dst + i) as usize] = compact[(src + i) as usize];
        }
    }
    for p in snapshot.start..snapshot.end {
        assert_eq!(slots[config.history_cell(0, p) as usize], p);
    }
    assert!(
        slots[config.history_capacity() as usize..(2 * config.history_capacity()) as usize]
            .iter()
            .all(|&v| v == u64::MAX)
    );
}

#[test]
fn binding_refuses_bad_spans_and_partial_transfer_plans_before_io() {
    let (config, layout, retained, staged) = setup();
    assert!(
        layout
            .bind(span(1, 256, retained.length + layout.row_bytes()), staged)
            .is_err()
    );
    assert!(
        layout
            .bind(retained, span(2, u64::MAX - 1, staged.length))
            .is_err()
    );
    assert!(layout.bind(retained, span(2, 3, staged.length)).is_err());
    assert!(
        layout
            .bind(retained, span(1, 256, layout.row_bytes()))
            .is_err()
    );
    // Shared allocations are allowed when the two role spans are disjoint.
    assert!(
        layout
            .bind(
                retained,
                span(1, retained.offset + retained.length, staged.length)
            )
            .is_ok()
    );
    let bound = layout.bind(retained, staged).unwrap();
    let valid = RetainCopy {
        staged_row: 0,
        history_row: 0,
        rows: 1,
    };
    assert!(
        bound
            .retain(&[
                valid,
                RetainCopy {
                    staged_row: 8192,
                    ..valid
                }
            ])
            .is_err()
    );
    assert!(
        bound
            .retain(&[RetainCopy {
                history_row: config.history_capacity() - 1,
                rows: 2,
                ..valid
            }])
            .is_err()
    );
    let chunk = ProjectionChunk {
        request: 17,
        slot: 2,
        stage: 1,
        position: 7,
        rows: 3,
        source: Source::Retained {
            row: config.history_cell(2, 7),
        },
        ring_row: 2 * u64::from(config.window) + 7,
    };
    let source = bound.projection_source(chunk).unwrap();
    assert_eq!(
        source.offset,
        retained.offset + config.history_cell(2, 7) * layout.row_bytes()
    );
    assert_eq!(source.length, 3 * layout.row_bytes());
    assert!(
        bound
            .projection_source(ProjectionChunk {
                source: Source::Retained { row: 7 },
                ..chunk
            })
            .is_err()
    );
    assert!(
        bound
            .projection_source(ProjectionChunk {
                ring_row: chunk.ring_row + 1,
                ..chunk
            })
            .is_err()
    );
    assert!(
        bound
            .projection_source(ProjectionChunk {
                stage: config.stages,
                ..chunk
            })
            .is_err()
    );
    assert!(
        bound
            .projection_source(ProjectionChunk {
                source: Source::Staged { row: u64::MAX },
                ..chunk
            })
            .is_err()
    );
}

#[test]
fn batched_gathers_copy_retained_rows_into_contiguous_staging() {
    let (config, layout, retained, staged) = setup();
    let bound = layout.bind(retained, staged).unwrap();
    let row = layout.row_bytes();
    let segment = crate::dspark::transfers::GatherSegment {
        request: 17,
        slot: 1,
        position: 300,
        rows: 4,
        history_row: config.history_cell(1, 300),
        staged_row: 6,
    };
    let copy = bound.gather(segment).unwrap();
    assert_eq!(copy.source.buffer, retained.buffer);
    assert_eq!(
        copy.source.offset,
        retained.offset + config.history_cell(1, 300) * row
    );
    assert_eq!(copy.destination.buffer, staged.buffer);
    assert_eq!(copy.destination.offset, staged.offset + 6 * row);
    assert_eq!(copy.source.length, 4 * row);
    assert_eq!(copy.destination.length, 4 * row);
    assert_eq!(
        bound.batch_source(10).unwrap(),
        Span::new(staged.buffer, staged.offset, 10 * row)
    );
    // Another slot's cell, and rows across the history wrap, are refused.
    assert!(
        bound
            .gather(crate::dspark::transfers::GatherSegment {
                history_row: config.history_cell(2, 300),
                ..segment
            })
            .is_err()
    );
    let wrap = config.history_capacity() - 2;
    assert!(
        bound
            .gather(crate::dspark::transfers::GatherSegment {
                position: wrap,
                history_row: config.history_cell(1, wrap),
                ..segment
            })
            .is_err()
    );
}
