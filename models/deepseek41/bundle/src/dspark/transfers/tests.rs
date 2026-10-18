use super::super::state::{Rows, State};
use super::*;
use std::collections::BTreeSet;

fn config(window: u32) -> Config {
    Config {
        slots: 3,
        stages: 3,
        window,
        max_prefill_rows: 8192,
        context_tokens: 1 << 20,
        position_capacity: 1 << 20,
    }
}

fn pass(config: Config, start: u64, count: u32) -> Projection {
    let mut state = State::new(config).unwrap();
    if start > 0 {
        state
            .restore(
                41,
                2,
                Snapshot {
                    start: start.saturating_sub(u64::from(config.window)),
                    end: start,
                    window: config.window,
                    stages: config.stages,
                },
            )
            .unwrap();
        for stage in 0..config.stages {
            let prior = state.projection(41, stage, u32::MAX).unwrap().unwrap();
            state.projected(prior).unwrap();
        }
    }
    state
        .prepare(vec![Rows {
            request: 41,
            slot: 2,
            start,
            kind: Kind::Prefill { rows: count },
        }])
        .unwrap();
    state.begin_execution().unwrap();
    state.executed().unwrap();
    let plan = state.begin_commit(&[]).unwrap();
    state.complete_commit(plan).unwrap();
    state.projection(41, 1, u32::MAX).unwrap().unwrap()
}

fn rows(chunks: &[ProjectionChunk]) -> Vec<(u64, u64, u64)> {
    chunks
        .iter()
        .flat_map(|chunk| {
            let source = match chunk.source {
                Source::Retained { row } | Source::Staged { row } => row,
            };
            (0..u64::from(chunk.rows)).map(move |offset| {
                (
                    chunk.position + offset,
                    source + offset,
                    chunk.ring_row + offset,
                )
            })
        })
        .collect()
}

fn disjoint_chunks(config: Config, chunks: &[ProjectionChunk], max_rows: u32) {
    for chunk in chunks {
        assert!(chunk.rows > 0 && chunk.rows <= max_rows && chunk.rows <= config.window);
        assert_eq!(chunk.slot, 2);
        assert!(chunk.ring_row >= 2 * u64::from(config.window));
        assert!(chunk.ring_row + u64::from(chunk.rows) <= 3 * u64::from(config.window));
        let cells = (chunk.position..chunk.position + u64::from(chunk.rows))
            .map(|position| position % u64::from(config.window))
            .collect::<BTreeSet<_>>();
        assert_eq!(cells.len(), chunk.rows as usize);
    }
}

#[test]
fn retained_projection_splits_at_both_wraps_and_preserves_exact_rows() {
    for window in [1, 3, 8, 128] {
        let config = config(window);
        let history = config.history_capacity();
        for start in [
            0,
            1,
            u64::from(window) - 1,
            u64::from(window),
            u64::from(window) + 1,
            history - 1,
            history,
            history + 1,
            3 * history + 2,
        ] {
            for count in [1, 5, history as u32] {
                let pass = pass(config, start, count);
                for capacity in [1, 3, 8, 1024] {
                    let chunks = retained_projection(config, pass, capacity).unwrap();
                    disjoint_chunks(config, &chunks, capacity);
                    for chunk in &chunks {
                        assert_eq!((chunk.request, chunk.stage), (41, 1));
                        let Source::Retained { row } = chunk.source else {
                            panic!("wrong source")
                        };
                        assert!(row >= 2 * history && row + u64::from(chunk.rows) <= 3 * history);
                    }
                    let expected = (start..start + u64::from(count))
                        .map(|position| {
                            (
                                position,
                                config.history_cell(2, position),
                                2 * u64::from(window) + position % u64::from(window),
                            )
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(rows(&chunks), expected);
                }
            }
        }
    }
}

#[test]
fn unsplit_projection_is_a_duplicate_ring_write_negative_control() {
    let config = config(4);
    let pass = pass(config, 3, 7);
    let naive = (pass.start..pass.start + u64::from(pass.count))
        .map(|position| position % u64::from(config.window))
        .collect::<BTreeSet<_>>();
    assert!(naive.len() < pass.count as usize);
    let chunks = retained_projection(config, pass, 128).unwrap();
    assert!(chunks.len() >= 3);
    disjoint_chunks(config, &chunks, 128);
    assert_eq!(
        rows(&chunks)
            .iter()
            .map(|&(position, _, _)| position)
            .collect::<Vec<_>>(),
        vec![3, 4, 5, 6, 7, 8, 9]
    );
}

fn long_commit(config: Config, start: u64, count: u32) -> CommitItem {
    CommitItem {
        rows: Rows {
            request: 41,
            slot: 2,
            start,
            kind: Kind::Prefill { rows: count },
        },
        kept: count,
        project_from_staging: true,
        retain_offset: count - config.window,
    }
}

#[test]
fn staged_prefill_projects_every_row_in_order_at_large_source_offsets() {
    for window in [3, 8, 128] {
        let config = config(window);
        let item = long_commit(config, u64::from(window) - 1, 4096);
        let source = (1_u64 << 40) + 13;
        for stage in 0..config.stages {
            for capacity in [1, 4, 512] {
                let chunks = staged_projection(config, item, stage, source, capacity).unwrap();
                disjoint_chunks(config, &chunks, capacity);
                assert!(chunks.iter().all(|chunk| chunk.stage == stage));
                let expected = (0..u64::from(item.kept))
                    .map(|offset| {
                        let position = item.rows.start + offset;
                        (
                            position,
                            source + offset,
                            2 * u64::from(window) + position % u64::from(window),
                        )
                    })
                    .collect::<Vec<_>>();
                assert_eq!(rows(&chunks), expected);
            }
        }
    }
}

#[test]
fn retention_copies_only_accepted_rows_for_all_depths() {
    let config = config(3);
    let source = (1_u64 << 39) + 31;
    for start in [0, 7, 8, 9, 10, 18] {
        for accepted in 0..=5 {
            let item = CommitItem {
                rows: Rows {
                    request: 41,
                    slot: 2,
                    start,
                    kind: Kind::Decode { drafts: 5 },
                },
                kept: accepted + 1,
                project_from_staging: false,
                retain_offset: 0,
            };
            let copies = retain_copies(config, item, source).unwrap();
            let actual = copies
                .iter()
                .flat_map(|copy| {
                    (0..u64::from(copy.rows))
                        .map(move |offset| (copy.staged_row + offset, copy.history_row + offset))
                })
                .collect::<Vec<_>>();
            let expected = (0..u64::from(item.kept))
                .map(|offset| (source + offset, config.history_cell(2, start + offset)))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
            assert!(copies.len() <= 2);
        }
    }
}

#[test]
fn long_prefill_retains_only_its_window_and_snapshots_round_trip_across_slots() {
    let config = config(4);
    for end in [4, 9, 10, 11, 20, 21, 64] {
        let snapshot = Snapshot {
            start: end - 4,
            end,
            window: 4,
            stages: 3,
        };
        let capture = snapshot_copies(config, 2, snapshot).unwrap();
        let mut history = vec![u64::MAX; 3 * config.history_capacity() as usize];
        for position in snapshot.start..snapshot.end {
            history[config.history_cell(2, position) as usize] = position;
        }
        let mut saved = vec![u64::MAX; 4];
        for copy in capture {
            for offset in 0..u64::from(copy.rows) {
                saved[(copy.snapshot_row + offset) as usize] =
                    history[(copy.history_row + offset) as usize];
            }
        }
        assert_eq!(saved, (snapshot.start..snapshot.end).collect::<Vec<_>>());
        for copy in snapshot_copies(config, 0, snapshot).unwrap() {
            for offset in 0..u64::from(copy.rows) {
                history[(copy.history_row + offset) as usize] =
                    saved[(copy.snapshot_row + offset) as usize];
            }
        }
        for position in snapshot.start..snapshot.end {
            assert_eq!(history[config.history_cell(0, position) as usize], position);
        }
    }
    let item = long_commit(config, 3, 61);
    let copies = retain_copies(config, item, 100).unwrap();
    let offsets = copies
        .iter()
        .flat_map(|copy| copy.staged_row..copy.staged_row + u64::from(copy.rows))
        .collect::<Vec<_>>();
    assert_eq!(offsets, vec![157, 158, 159, 160]);
}

#[test]
fn short_snapshots_copy_no_uninitialized_history_rows() {
    let config = config(128);
    let snapshot = Snapshot {
        start: 0,
        end: 3,
        window: 128,
        stages: 3,
    };
    assert_eq!(
        snapshot_copies(config, 1, snapshot).unwrap(),
        vec![SnapshotCopy {
            history_row: 134,
            snapshot_row: 0,
            rows: 3
        }]
    );
}

#[test]
fn malformed_transfer_plans_fail_before_any_copy() {
    let config = config(4);
    let pass = pass(config, 3, 7);
    assert!(retained_projection(config, pass, 0).is_err());
    let mut invalid = [pass; 5];
    invalid[0].stage = 3;
    invalid[1].slot = 3;
    invalid[2].count = 0;
    invalid[3].count = 11;
    invalid[4].start = u64::MAX;
    for bad in invalid {
        assert!(retained_projection(config, bad, 128).is_err());
    }
    let item = long_commit(config, 3, 61);
    for bad in [
        CommitItem { kept: 0, ..item },
        CommitItem { kept: 60, ..item },
        CommitItem {
            project_from_staging: false,
            ..item
        },
        CommitItem {
            retain_offset: 0,
            ..item
        },
    ] {
        assert!(retain_copies(config, bad, 0).is_err());
        assert!(staged_projection(config, bad, 0, 0, 128).is_err());
    }
    assert!(retain_copies(config, item, u64::MAX).is_err());
    assert!(staged_projection(config, item, 3, 0, 128).is_err());
    assert!(staged_projection(config, item, 0, u64::MAX, 128).is_err());
    assert!(staged_projection(config, item, 0, 0, 0).is_err());
    let snapshot = Snapshot {
        start: 0,
        end: 3,
        window: 4,
        stages: 3,
    };
    assert!(snapshot_copies(config, 3, snapshot).is_err());
    for bad in [
        Snapshot {
            start: 1,
            ..snapshot
        },
        Snapshot { end: 0, ..snapshot },
        Snapshot {
            stages: 2,
            ..snapshot
        },
        Snapshot {
            window: 3,
            ..snapshot
        },
    ] {
        assert!(snapshot_copies(config, 0, bad).is_err());
    }
}
