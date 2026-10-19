use super::*;

#[test]
fn declared_drafter_state_matches_the_committed_row_layout() {
    use bundle_runtime::state::{Capture, Scope};
    use model_deepseek41_contract::geometry::GEOMETRIES;
    use model_deepseek41_contract::state::{KvStorage, Widths, request_state};
    for geometry in GEOMETRIES {
        let config = Config {
            slots: 3,
            stages: geometry.dspark_stages(),
            window: geometry.shape.attention.window,
            max_prefill_rows: 8192,
            context_tokens: geometry.context_tokens(),
            position_capacity: geometry.context_tokens(),
        };
        let feature_row =
            geometry.shape.dspark.targets.len() as u64 * u64::from(geometry.shape.hidden) * 2;
        for storage in [KvStorage::Native, KvStorage::Bf16] {
            let state = request_state(geometry, storage, true);
            let features = state
                .buffers
                .iter()
                .find(|b| b.name == "dspark.features")
                .unwrap();
            assert_eq!(features.planes, [config.history_capacity() * feature_row]);
            assert_eq!(
                features.scope,
                Scope::Rows {
                    rows: config.history_capacity()
                }
            );
            let ring = state
                .buffers
                .iter()
                .find(|b| b.name == "dspark.ring")
                .unwrap();
            assert_eq!(ring.layers, u64::from(config.stages));
            assert_eq!(
                ring.scope,
                Scope::WindowRing {
                    tokens: u64::from(config.window)
                }
            );
            assert_eq!(ring.capture, Capture::None);
            let planes: Vec<_> = storage
                .ring_cell(Widths::of(geometry))
                .into_iter()
                .filter(|&n| n != 0)
                .map(|n| n * u64::from(config.window))
                .collect();
            assert_eq!(ring.planes, planes);
        }
    }
}

fn config() -> Config {
    Config {
        slots: 3,
        stages: 3,
        window: 4,
        max_prefill_rows: 64,
        context_tokens: 256,
        position_capacity: 256,
    }
}

fn prefill(request: u64, slot: u32, start: u64, rows: u32) -> Rows {
    Rows {
        request,
        slot,
        start,
        kind: Kind::Prefill { rows },
    }
}

fn decode(request: u64, slot: u32, start: u64, drafts: u32) -> Rows {
    Rows {
        request,
        slot,
        start,
        kind: Kind::Decode { drafts },
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Feature {
    position: u64,
    /// Three distinguishable raw attention-input taps; no projected KV is
    /// substituted for them in the retained history.
    taps: [u64; 3],
}

impl Feature {
    fn new(request: u64, position: u64, salt: u64) -> Self {
        Self {
            position,
            taps: [request, 37 + position, 39 + salt],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Event {
    Project {
        slot: u32,
        stage: u32,
        feature: Feature,
        from_staging: bool,
    },
    Retain {
        slot: u32,
        feature: Feature,
    },
}

/// A bounded memory model of the copies and projections the plans require.
/// This checks contents and ordered row coverage, not GPU arithmetic.
struct Memory {
    state: State,
    raw: BTreeMap<u64, Feature>,
    rings: BTreeMap<(u32, u32, u64), Feature>,
    journal: Vec<Event>,
}

impl Memory {
    fn new() -> Self {
        Self {
            state: State::new(config()).unwrap(),
            raw: BTreeMap::new(),
            rings: BTreeMap::new(),
            journal: Vec::new(),
        }
    }

    fn project(&mut self, slot: u32, stage: u32, feature: Feature, from_staging: bool) {
        let cell = feature.position % u64::from(self.state.config().window);
        self.rings.insert((slot, stage, cell), feature);
        self.journal.push(Event::Project {
            slot,
            stage,
            feature,
            from_staging,
        });
    }

    fn catch_up(&mut self, request: u64) {
        for stage in 0..self.state.config().stages {
            while let Some(pass) = self.state.projection(request, stage, 3).unwrap() {
                for position in pass.start..pass.start + u64::from(pass.count) {
                    let feature = self.raw[&self.state.config().history_cell(pass.slot, position)];
                    assert_eq!(
                        feature.position, position,
                        "raw feature was overwritten too early"
                    );
                    self.project(pass.slot, stage, feature, false);
                }
                self.state.projected(pass).unwrap();
            }
        }
    }

    fn run(&mut self, rows: Vec<Rows>, kept: &[(u64, u32)], salt: u64) -> Vec<DraftBlock> {
        self.state.prepare(rows.clone()).unwrap();
        for row in &rows {
            if self.state.context(row.request).is_some() {
                self.catch_up(row.request);
            }
        }
        let blocks = self.state.begin_execution().unwrap();
        let staged = rows
            .iter()
            .map(|row| {
                (
                    row.request,
                    (0..row.kind.target_rows().unwrap())
                        .map(|offset| {
                            Feature::new(row.request, row.start + u64::from(offset), salt)
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        self.state.executed().unwrap();
        let plan = self.state.begin_commit(kept).unwrap();
        self.apply(&plan, &staged);
        self.state.complete_commit(plan).unwrap();
        blocks
    }

    fn apply(&mut self, plan: &CommitPlan, staged: &BTreeMap<u64, Vec<Feature>>) {
        for item in plan.items() {
            let features = &staged[&item.rows.request][..item.kept as usize];
            if item.project_from_staging {
                for stage in 0..self.state.config().stages {
                    for &feature in features {
                        self.project(item.rows.slot, stage, feature, true);
                    }
                }
            }
            for &feature in &features[item.retain_offset as usize..] {
                self.raw.insert(
                    self.state
                        .config()
                        .history_cell(item.rows.slot, feature.position),
                    feature,
                );
                self.journal.push(Event::Retain {
                    slot: item.rows.slot,
                    feature,
                });
            }
        }
    }

    fn positions(&self, slot: u32, stage: u32) -> Vec<u64> {
        self.journal
            .iter()
            .filter_map(|event| match event {
                Event::Project {
                    slot: s,
                    stage: t,
                    feature,
                    ..
                } if (*s, *t) == (slot, stage) => Some(feature.position),
                _ => None,
            })
            .collect()
    }

    fn assert_ring(&self, request: u64, salt: u64) {
        let context = self.state.context(request).unwrap();
        let window = u64::from(self.state.config().window);
        for stage in 0..self.state.config().stages {
            for position in context.end.saturating_sub(window)..context.end {
                assert_eq!(
                    self.rings[&(context.slot, stage, position % window)],
                    Feature::new(request, position, salt)
                );
            }
        }
    }
}

#[test]
fn every_acceptance_depth_retains_and_projects_all_kept_target_rows() {
    for accepted in 0..=5 {
        let mut memory = Memory::new();
        memory.run(vec![prefill(41, 0, 0, 3)], &[], 7);
        memory.run(vec![decode(41, 0, 3, 5)], &[(41, accepted + 1)], 7);
        let end = 4 + u64::from(accepted);
        let context = memory.state.context(41).unwrap();
        assert_eq!(context.end, end);
        assert_eq!(context.projected(), &[3, 3, 3]);
        for event in &memory.journal {
            if let Event::Retain { feature, .. } = event {
                assert!(
                    feature.position < end,
                    "a rejected target row entered history"
                );
            }
        }
        memory.catch_up(41);
        for stage in 0..3 {
            assert_eq!(memory.positions(0, stage), (0..end).collect::<Vec<_>>());
        }
        memory.assert_ring(41, 7);
    }
}

#[test]
fn ar_only_steps_catch_up_and_wrap_without_skipping_rows() {
    let mut memory = Memory::new();
    memory.run(vec![prefill(41, 0, 0, 2)], &[], 7);
    for start in 2..35 {
        let blocks = memory.run(vec![decode(41, 0, start, 0)], &[], 7);
        assert!(blocks.is_empty());
        assert_eq!(memory.state.context(41).unwrap().projected(), &[start; 3]);
    }
    memory.catch_up(41);
    for stage in 0..3 {
        assert_eq!(memory.positions(0, stage), (0..35).collect::<Vec<_>>());
    }
    assert_eq!(
        memory.raw.len(),
        memory.state.config().history_capacity() as usize
    );
    memory.assert_ring(41, 7);
}

#[test]
fn long_prefill_projects_every_row_from_staging_with_bounded_history() {
    let mut memory = Memory::new();
    memory.run(vec![prefill(41, 0, 0, 31)], &[], 7);
    assert_eq!(memory.state.context(41).unwrap().projected(), &[31; 3]);
    assert_eq!(memory.state.context(41).unwrap().retained_start, 27);
    assert_eq!(memory.raw.len(), 4);
    for stage in 0..3 {
        assert_eq!(memory.positions(0, stage), (0..31).collect::<Vec<_>>());
    }
    assert!(
        memory
            .journal
            .iter()
            .filter_map(|event| match event {
                Event::Project { from_staging, .. } => Some(*from_staging),
                _ => None,
            })
            .all(|from_staging| from_staging)
    );
    memory.assert_ring(41, 7);
    memory.run(vec![decode(41, 0, 31, 5)], &[(41, 4)], 7);
    memory.catch_up(41);
    for stage in 0..3 {
        assert_eq!(memory.positions(0, stage), (0..35).collect::<Vec<_>>());
    }
    memory.assert_ring(41, 7);
}

#[test]
fn prefill_at_and_above_history_capacity_preserves_order() {
    let mut memory = Memory::new();
    let capacity = config().history_capacity() as u32;
    memory.run(vec![prefill(41, 0, 0, capacity)], &[], 7);
    assert_eq!(memory.state.context(41).unwrap().projected(), &[0; 3]);
    memory.run(
        vec![prefill(41, 0, u64::from(capacity), capacity + 1)],
        &[],
        7,
    );
    let end = u64::from(2 * capacity + 1);
    assert_eq!(memory.state.context(41).unwrap().projected(), &[end; 3]);
    for stage in 0..3 {
        assert_eq!(memory.positions(0, stage), (0..end).collect::<Vec<_>>());
    }
    memory.assert_ring(41, 7);
}

#[test]
fn cancellation_keeps_raw_history_and_completed_catch_up() {
    for cancel_phase in 0..3 {
        let mut memory = Memory::new();
        memory.run(vec![prefill(41, 0, 0, 8)], &[], 7);
        let raw_before = memory.raw.clone();
        memory.state.prepare(vec![decode(41, 0, 8, 5)]).unwrap();
        memory.catch_up(41);
        let context = memory.state.context(41).unwrap().clone();
        if cancel_phase >= 1 {
            memory.state.begin_execution().unwrap();
        }
        if cancel_phase >= 2 {
            memory.state.executed().unwrap();
        }
        memory.state.cancel().unwrap();
        assert_eq!(memory.state.context(41), Some(&context));
        assert_eq!(memory.raw, raw_before);
        memory.run(vec![decode(41, 0, 8, 5)], &[(41, 2)], 7);
        memory.catch_up(41);
        for stage in 0..3 {
            assert_eq!(memory.positions(0, stage), (0..10).collect::<Vec<_>>());
        }
        memory.assert_ring(41, 7);
    }
    let mut state = State::new(config()).unwrap();
    state.prepare(vec![prefill(41, 0, 0, 3)]).unwrap();
    state.cancel().unwrap();
    state.prepare(vec![prefill(42, 0, 0, 3)]).unwrap();
    assert!(state.context(41).is_none());
}

#[test]
fn every_stage_must_catch_up_before_any_target_execution() {
    let mut memory = Memory::new();
    memory.run(vec![prefill(41, 0, 0, 8)], &[], 7);
    // An AR step defers the rows the drafter has not projected.
    memory.state.prepare(vec![decode(41, 0, 8, 0)]).unwrap();
    assert!(memory.state.begin_execution().unwrap().is_empty());
    memory.state.cancel().unwrap();
    memory.state.prepare(vec![decode(41, 0, 8, 1)]).unwrap();
    for stage in 0..3 {
        assert!(memory.state.begin_execution().is_err());
        let pass = memory.state.projection(41, stage, 64).unwrap().unwrap();
        assert_eq!((pass.start, pass.count), (0, 8));
        memory.state.projected(pass).unwrap();
        assert!(memory.state.projected(pass).is_err());
    }
    assert_eq!(memory.state.begin_execution().unwrap().len(), 1);
    assert!(memory.state.projection(41, 0, 1).is_err());
}

#[test]
fn a_bulk_commit_must_catch_up_before_any_target_execution() {
    let mut state = State::new(config()).unwrap().with_batch_rows(2).unwrap();
    state.prepare(vec![prefill(41, 0, 0, 3)]).unwrap();
    state.begin_execution().unwrap();
    state.executed().unwrap();
    let plan = state.begin_commit(&[]).unwrap();
    state.complete_commit(plan).unwrap();
    assert_eq!(state.context(41).unwrap().bulk(), Some((0, 3)));
    state.prepare(vec![decode(41, 0, 3, 0)]).unwrap();
    assert!(state.begin_execution().is_err());
    // Batched rows never cover a bulk range.
    state.skip_unread(41).unwrap();
    assert_eq!(state.row_projection(41).unwrap(), None);
    for stage in 0..3 {
        let pass = state.projection(41, stage, 2).unwrap().unwrap();
        assert_eq!((pass.start, pass.count), (0, 2));
        state.projected(pass).unwrap();
        let pass = state.projection(41, stage, 2).unwrap().unwrap();
        assert_eq!((pass.start, pass.count), (2, 1));
        state.projected(pass).unwrap();
    }
    assert_eq!(state.context(41).unwrap().bulk(), None);
    assert!(state.begin_execution().unwrap().is_empty());
}

#[test]
fn batched_projections_skip_overwritten_rows_and_reject_stale_passes() {
    let mut state = State::new(config()).unwrap().with_batch_rows(2).unwrap();
    let mut end = 0;
    for rows in [vec![prefill(41, 0, 0, 1)], vec![prefill(41, 0, 1, 1)]]
        .into_iter()
        .chain((2..9).map(|start| vec![decode(41, 0, start, 0)]))
    {
        state.prepare(rows).unwrap();
        state.begin_execution().unwrap();
        state.executed().unwrap();
        let plan = state.begin_commit(&[]).unwrap();
        state.complete_commit(plan).unwrap();
        end += 1;
    }
    assert_eq!(end, 9);
    assert_eq!(state.context(41).unwrap().projected(), &[0; 3]);
    state.prepare(vec![decode(41, 0, 9, 1)]).unwrap();
    // A pass must follow the skip.
    assert!(state.row_projection(41).is_err());
    state.skip_unread(41).unwrap();
    assert_eq!(state.context(41).unwrap().projected(), &[5; 3]);
    let pass = state.row_projection(41).unwrap().unwrap();
    assert_eq!((pass.slot, pass.start, pass.count), (0, 5, 4));
    state.projected_rows(pass).unwrap();
    assert!(state.projected_rows(pass).is_err());
    assert_eq!(state.row_projection(41).unwrap(), None);
    assert_eq!(state.begin_execution().unwrap().len(), 1);
    assert!(state.skip_unread(41).is_err());
}

#[test]
fn budgets_compute_five_positions_and_verify_only_the_emitted_prefix() {
    for budget in 1..=5 {
        let mut memory = Memory::new();
        memory.run(vec![prefill(41, 0, 0, 3)], &[], 7);
        let blocks = memory.run(vec![decode(41, 0, 3, budget)], &[(41, budget + 1)], 7);
        assert_eq!(
            blocks,
            vec![DraftBlock {
                request: 41,
                slot: 0,
                main_rows: 3,
                compute_rows: 5,
                emit_rows: budget,
                verify_rows: budget + 1,
            }]
        );
        assert_eq!(memory.state.context(41).unwrap().end, 4 + u64::from(budget));
    }
}

#[test]
fn snapshot_preemption_and_restore_rebuild_every_stage_in_a_new_slot() {
    let mut memory = Memory::new();
    memory.run(vec![prefill(41, 0, 0, 8)], &[], 7);
    assert!(memory.state.snapshot(41).is_err());
    memory.catch_up(41);
    let snapshot = memory.state.snapshot(41).unwrap();
    assert_eq!(
        snapshot,
        Snapshot {
            start: 4,
            end: 8,
            window: 4,
            stages: 3
        }
    );
    let saved = (snapshot.start..snapshot.end)
        .map(|position| memory.raw[&memory.state.config().history_cell(0, position)])
        .collect::<Vec<_>>();
    memory.state.retire(41).unwrap();
    memory.run(vec![prefill(42, 0, 0, 3)], &[], 99);
    for feature in saved {
        memory.raw.insert(
            memory.state.config().history_cell(1, feature.position),
            feature,
        );
    }
    memory.state.restore(41, 1, snapshot).unwrap();
    assert_eq!(memory.state.context(41).unwrap().projected(), &[4; 3]);
    memory.state.prepare(vec![decode(41, 1, 8, 1)]).unwrap();
    assert!(memory.state.begin_execution().is_err());
    memory.catch_up(41);
    memory.assert_ring(41, 7);
    for stage in 0..3 {
        assert_eq!(memory.positions(1, stage), vec![4, 5, 6, 7]);
    }
    memory.state.begin_execution().unwrap();
    assert!(memory.state.retire(41).is_err());
    memory.state.cancel().unwrap();
    memory.catch_up(42);
    memory.assert_ring(42, 99);
}

#[test]
fn same_request_and_slot_restore_invalidates_old_projection_completions() {
    let mut state = State::new(config()).unwrap();
    let snapshot = Snapshot {
        start: 4,
        end: 8,
        window: 4,
        stages: 3,
    };
    state.restore(41, 0, snapshot).unwrap();
    let stale = state.projection(41, 0, 4).unwrap().unwrap();
    state.retire(41).unwrap();
    state.restore(41, 0, snapshot).unwrap();
    let before = state.clone();
    assert!(state.projected(stale).is_err());
    assert_eq!(state, before);
    state
        .projected(state.projection(41, 0, 4).unwrap().unwrap())
        .unwrap();
    assert_eq!(state.context(41).unwrap().projected(), &[8, 4, 4]);
}

#[test]
fn invalid_acceptance_is_atomic_and_commit_cannot_be_cancelled() {
    let mut memory = Memory::new();
    memory.run(vec![prefill(41, 0, 0, 3), prefill(42, 1, 0, 4)], &[], 7);
    memory
        .state
        .prepare(vec![decode(41, 0, 3, 5), decode(42, 1, 4, 2)])
        .unwrap();
    memory.catch_up(41);
    memory.catch_up(42);
    memory.state.begin_execution().unwrap();
    assert!(memory.state.begin_commit(&[(41, 1), (42, 1)]).is_err());
    memory.state.executed().unwrap();
    let before = memory.state.clone();
    for kept in [
        vec![],
        vec![(41, 1)],
        vec![(42, 1), (41, 1)],
        vec![(41, 0), (42, 1)],
        vec![(41, 7), (42, 1)],
        vec![(41, 1), (42, 4)],
    ] {
        assert!(memory.state.begin_commit(&kept).is_err());
        assert_eq!(memory.state, before);
    }
    let plan = memory.state.begin_commit(&[(41, 6), (42, 1)]).unwrap();
    assert!(memory.state.cancel().is_err());
    assert!(memory.state.begin_commit(&[(41, 6), (42, 1)]).is_err());
    let stale = plan.clone();
    memory.state.complete_commit(plan).unwrap();
    assert_eq!(memory.state.context(41).unwrap().end, 9);
    assert_eq!(memory.state.context(42).unwrap().end, 5);
    assert!(memory.state.complete_commit(stale).is_err());
}

#[test]
fn commit_completion_must_match_the_plan_that_started_its_copies() {
    let mut memory = Memory::new();
    memory.run(vec![prefill(41, 0, 0, 3)], &[], 7);
    memory.state.prepare(vec![decode(41, 0, 3, 5)]).unwrap();
    memory.catch_up(41);
    memory.state.begin_execution().unwrap();
    memory.state.executed().unwrap();
    let mut other = memory.state.clone();
    let correct = memory.state.begin_commit(&[(41, 1)]).unwrap();
    let wrong = other.begin_commit(&[(41, 6)]).unwrap();
    let before = memory.state.clone();
    assert!(memory.state.complete_commit(wrong).is_err());
    assert_eq!(memory.state, before);
    memory.state.complete_commit(correct).unwrap();
    assert_eq!(memory.state.context(41).unwrap().end, 4);
}

#[test]
fn malformed_steps_and_restores_leave_the_state_unchanged() {
    let mut memory = Memory::new();
    memory.run(vec![prefill(41, 0, 0, 3)], &[], 7);
    let before = memory.state.clone();
    for rows in [
        vec![],
        vec![decode(41, 0, 3, 6)],
        vec![decode(41, 1, 3, 0)],
        vec![decode(41, 0, 4, 0)],
        vec![prefill(42, 0, 0, 1)],
        vec![prefill(42, 1, 1, 1)],
        vec![prefill(42, 1, 0, 0)],
        vec![prefill(42, 1, 0, 65)],
        vec![decode(42, 1, 0, 1)],
        vec![decode(41, 0, 3, 0), decode(41, 0, 3, 0)],
        vec![prefill(42, 1, 0, 1), prefill(43, 1, 0, 1)],
        vec![prefill(42, 3, 0, 1)],
        vec![decode(41, 0, u64::MAX, 1)],
    ] {
        assert!(memory.state.prepare(rows).is_err());
        assert_eq!(memory.state, before);
    }
    let valid = Snapshot {
        start: 0,
        end: 3,
        window: 4,
        stages: 3,
    };
    for bad in [
        Snapshot { start: 1, ..valid },
        Snapshot { end: 0, ..valid },
        Snapshot { end: 257, ..valid },
        Snapshot { window: 5, ..valid },
        Snapshot { stages: 1, ..valid },
    ] {
        assert!(memory.state.restore(42, 1, bad).is_err());
        assert_eq!(memory.state, before);
    }
    assert!(memory.state.restore(42, 0, valid).is_err());
    assert!(memory.state.restore(41, 1, valid).is_err());
    assert!(memory.state.restore(42, 3, valid).is_err());
    assert_eq!(memory.state, before);
}

#[test]
fn partial_budget_still_needs_position_capacity_for_the_whole_block() {
    let mut state = State::new(config()).unwrap();
    state
        .restore(
            41,
            0,
            Snapshot {
                start: 250,
                end: 254,
                window: 4,
                stages: 3,
            },
        )
        .unwrap();
    assert_eq!(
        state.prepare(vec![decode(41, 0, 254, 1)]),
        Err("DSpark full block exceeds position capacity")
    );
    state.prepare(vec![decode(41, 0, 254, 0)]).unwrap();
    state.cancel().unwrap();
    let mut extended = State::new(Config {
        position_capacity: 260,
        ..config()
    })
    .unwrap();
    extended
        .restore(
            41,
            0,
            Snapshot {
                start: 250,
                end: 254,
                window: 4,
                stages: 3,
            },
        )
        .unwrap();
    extended.prepare(vec![decode(41, 0, 254, 1)]).unwrap();
}

#[test]
fn invalid_geometry_and_projection_ranges_are_refused() {
    for bad in [
        Config {
            slots: 0,
            ..config()
        },
        Config {
            stages: 0,
            ..config()
        },
        Config {
            window: 0,
            ..config()
        },
        Config {
            window: u32::MAX,
            ..config()
        },
        Config {
            max_prefill_rows: 0,
            ..config()
        },
        Config {
            context_tokens: 0,
            ..config()
        },
        Config {
            position_capacity: 255,
            ..config()
        },
    ] {
        assert!(State::new(bad).is_err());
    }
    let mut state = State::new(config()).unwrap();
    state
        .restore(
            41,
            0,
            Snapshot {
                start: 4,
                end: 8,
                window: 4,
                stages: 3,
            },
        )
        .unwrap();
    assert!(state.projection(41, 3, 1).is_err());
    assert!(state.projection(41, 0, 0).is_err());
    assert!(state.projection(42, 0, 1).is_err());
    let good = state.projection(41, 0, 2).unwrap().unwrap();
    let before = state.clone();
    for bad in [
        Projection { count: 0, ..good },
        Projection { count: 5, ..good },
        Projection { start: 5, ..good },
        Projection { slot: 1, ..good },
        Projection { stage: 3, ..good },
        Projection {
            start: u64::MAX,
            ..good
        },
    ] {
        assert!(state.projected(bad).is_err());
        assert_eq!(state, before);
    }
    state.projected(good).unwrap();
    assert_eq!(state.projection(41, 0, 9).unwrap().unwrap().start, 6);
}

#[test]
fn device_failure_poisons_all_lifecycle_mutations() {
    let mut state = State::new(config()).unwrap();
    state.prepare(vec![prefill(41, 0, 0, 3)]).unwrap();
    state.begin_execution().unwrap();
    state.executed().unwrap();
    let plan = state.begin_commit(&[]).unwrap();
    state.poison();
    assert!(state.is_poisoned());
    assert!(state.complete_commit(plan).is_err());
    assert!(state.cancel().is_err());
    assert!(state.retire(41).is_err());
    assert!(state.snapshot(41).is_err());
    assert!(state.projection(41, 0, 1).is_err());
    assert!(state.prepare(vec![prefill(41, 0, 0, 3)]).is_err());
    assert!(state.begin_execution().is_err());
    assert!(state.executed().is_err());
    assert!(state.begin_commit(&[]).is_err());
    assert!(
        state
            .restore(
                41,
                0,
                Snapshot {
                    start: 0,
                    end: 3,
                    window: 4,
                    stages: 3
                }
            )
            .is_err()
    );
}

/// A pass of one stage, as `projection` issued it: request, stage, start
/// and count.
type Pass = (u64, u32, u64, u32);

/// The serving runtime's order: drafting items and bulk commits catch up,
/// other items defer.
struct Deferred {
    memory: Memory,
    passes: Vec<Pass>,
    batched: Vec<(u64, u64, u32)>,
}

impl Deferred {
    fn new() -> Self {
        let mut memory = Memory::new();
        memory.state = State::new(config()).unwrap().with_batch_rows(2).unwrap();
        Self {
            memory,
            passes: Vec::new(),
            batched: Vec::new(),
        }
    }

    fn catch_up(&mut self, request: u64) {
        let m = &mut self.memory;
        m.state.skip_unread(request).unwrap();
        if let Some(pass) = m.state.row_projection(request).unwrap() {
            assert!(pass.count <= m.state.config().window);
            self.batched.push((request, pass.start, pass.count));
            for stage in 0..m.state.config().stages {
                for position in pass.start..pass.start + u64::from(pass.count) {
                    let feature = m.raw[&m.state.config().history_cell(pass.slot, position)];
                    assert_eq!(feature.position, position);
                    m.project(pass.slot, stage, feature, false);
                }
            }
            m.state.projected_rows(pass).unwrap();
        }
        if m.state.has_bulk(request) {
            for stage in 0..m.state.config().stages {
                while let Some(pass) = m.state.projection(request, stage, 3).unwrap() {
                    self.passes.push((request, stage, pass.start, pass.count));
                    for position in pass.start..pass.start + u64::from(pass.count) {
                        let feature = m.raw[&m.state.config().history_cell(pass.slot, position)];
                        m.project(pass.slot, stage, feature, false);
                    }
                    m.state.projected(pass).unwrap();
                }
            }
        }
        assert!(!m.state.has_bulk(request));
    }

    fn run(&mut self, rows: Vec<Rows>, kept: &[(u64, u32)], salt: u64) -> Vec<DraftBlock> {
        self.memory.state.prepare(rows.clone()).unwrap();
        for row in &rows {
            if self.memory.state.needs_catch_up(row.request)
                && (row.kind.drafts() > 0 || self.memory.state.has_bulk(row.request))
            {
                self.catch_up(row.request);
            }
        }
        let blocks = self.memory.state.begin_execution().unwrap();
        let staged = rows
            .iter()
            .map(|row| {
                (
                    row.request,
                    (0..row.kind.target_rows().unwrap())
                        .map(|offset| {
                            Feature::new(row.request, row.start + u64::from(offset), salt)
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        self.memory.state.executed().unwrap();
        let plan = self.memory.state.begin_commit(kept).unwrap();
        self.memory.apply(&plan, &staged);
        self.memory.state.complete_commit(plan).unwrap();
        blocks
    }
}

/// Eager passes, recorded by wrapping `Memory::catch_up`.
fn eager_passes(memory: &mut Memory, rows: &[Rows], passes: &mut Vec<Pass>) {
    for row in rows {
        if memory.state.context(row.request).is_none() {
            continue;
        }
        for stage in 0..memory.state.config().stages {
            while let Some(pass) = memory.state.projection(row.request, stage, 3).unwrap() {
                passes.push((row.request, stage, pass.start, pass.count));
                for position in pass.start..pass.start + u64::from(pass.count) {
                    let feature =
                        memory.raw[&memory.state.config().history_cell(pass.slot, position)];
                    memory.project(pass.slot, stage, feature, false);
                }
                memory.state.projected(pass).unwrap();
            }
        }
    }
}

#[test]
fn deferred_catch_up_drafts_from_the_eager_rings_and_keeps_wide_passes() {
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = |bound: u32| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % u64::from(bound)) as u32
    };
    let mut eager = Memory::new();
    let mut eager_log = Vec::new();
    let mut deferred = Deferred::new();
    let mut ends = [0_u64; 2];
    // Prefills of one to three rows: three rows are a bulk commit.
    for (request, end) in (41..).zip(&mut ends) {
        let rows = 1 + next(3);
        let step = vec![prefill(request, request as u32 - 41, 0, rows)];
        eager.run(step.clone(), &[], 7);
        deferred.run(step, &[], 7);
        *end = u64::from(rows);
    }
    while ends.iter().all(|&end| end + 6 < config().context_tokens) {
        let mut rows = Vec::new();
        let mut kept = Vec::new();
        for (index, end) in ends.iter().enumerate() {
            let request = 41 + index as u64;
            // Mostly AR steps, so deferred rows outlive the window.
            let drafts = if next(4) == 0 { 1 + next(5) } else { 0 };
            rows.push(decode(request, index as u32, *end, drafts));
            if drafts > 0 {
                kept.push((request, 1 + next(drafts + 1)));
            }
        }
        // Eager: catch everything up before execution, recording passes.
        eager.state.prepare(rows.clone()).unwrap();
        eager_passes(&mut eager, &rows, &mut eager_log);
        eager.state.cancel().unwrap();
        let eager_blocks = eager.run(rows.clone(), &kept, 7);
        let blocks = deferred.run(rows.clone(), &kept, 7);
        assert_eq!(blocks, eager_blocks);
        // Every drafting item reads the eager ring contents.
        for block in &blocks {
            let window = u64::from(config().window);
            for stage in 0..config().stages {
                for position in block.main_rows.saturating_sub(window)..block.main_rows {
                    let cell = (block.slot, stage, position % window);
                    assert_eq!(deferred.memory.rings[&cell], eager.rings[&cell]);
                    assert_eq!(eager.rings[&cell].position, position);
                }
            }
        }
        for (index, end) in ends.iter_mut().enumerate() {
            let request = 41 + index as u64;
            *end = deferred.memory.state.context(request).unwrap().end;
            assert_eq!(*end, eager.state.context(request).unwrap().end);
        }
    }
    // Every pass wider than the batch bound ran identically; the rest ran
    // in batches of at most one window.
    let wide = eager_log
        .iter()
        .filter(|pass| pass.3 > 2)
        .copied()
        .collect::<Vec<_>>();
    assert!(!wide.is_empty());
    assert_eq!(
        deferred
            .passes
            .iter()
            .filter(|pass| pass.3 > 2)
            .copied()
            .collect::<Vec<_>>(),
        wide
    );
    assert!(!deferred.batched.is_empty());
    // Deferral skipped rows the window had already overwritten.
    let projected = deferred
        .memory
        .journal
        .iter()
        .filter(|event| matches!(event, Event::Project { .. }))
        .count();
    let eager_projected = eager
        .journal
        .iter()
        .filter(|event| matches!(event, Event::Project { .. }))
        .count();
    assert!(projected < eager_projected, "{projected} {eager_projected}");
}
