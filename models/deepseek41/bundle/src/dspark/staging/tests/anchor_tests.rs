use super::*;
use crate::dspark::state::{Config, Rows, State};
use model_deepseek41_contract::geometry::FLASH;

fn state() -> State {
    State::new(Config {
        slots: 1,
        stages: 3,
        window: 128,
        max_prefill_rows: 8192,
        context_tokens: 4096,
        position_capacity: 4096,
    })
    .unwrap()
}
fn rows(start: u64, count: u32) -> Rows {
    Rows {
        request: 7,
        slot: 0,
        start,
        kind: Kind::Prefill { rows: count },
    }
}
fn pending(rows: Rows) -> Pending {
    Pending {
        rows: TargetRows {
            rows,
            tokens: (rows.start..rows.start + u64::from(rows.kind.target_rows().unwrap()))
                .map(|p| p as u32)
                .collect(),
        },
        text: None,
    }
}
fn executed(state: &mut State, rows: Rows) {
    state.prepare(vec![rows]).unwrap();
    state.begin_execution().unwrap();
    state.executed().unwrap();
}

#[test]
fn anchor_segments_publish_the_split_history_and_reject_a_wrong_head() {
    let mut staging = Staging::new(&FLASH).unwrap();
    staging.pending.push(pending(rows(0, 789)));
    let mut wrong = state();
    executed(&mut wrong, rows(0, 788));
    let plan = wrong.begin_commit_with_anchors(&[], &[(7, 768)]).unwrap();
    assert_eq!(
        staging.validate_commit(&plan).unwrap_err(),
        "Engram commit has no matching target rows"
    );
    assert!(staging.rows.history(7).is_empty());
    assert_eq!(staging.pending[0].rows.tokens.len(), 789);

    let mut fused = state();
    executed(&mut fused, rows(0, 789));
    let plan = fused.begin_commit_with_anchors(&[], &[(7, 768)]).unwrap();
    let tails = plan.tails().to_vec();
    staging.commit(&plan).unwrap();
    fused.complete_commit(plan).unwrap();
    assert_eq!(staging.pending.len(), 1);
    assert_eq!(staging.pending[0].rows.rows, rows(768, 21));
    assert_eq!(
        staging.pending[0].rows.tokens,
        (768..789).collect::<Vec<_>>()
    );
    let plan = fused.begin_tail_commit(&tails).unwrap();
    staging.commit(&plan).unwrap();
    fused.complete_commit(plan).unwrap();
    assert!(staging.pending.is_empty());

    let mut split = Staging::new(&FLASH).unwrap();
    let mut lifecycle = state();
    for row in [rows(0, 768), rows(768, 21)] {
        executed(&mut lifecycle, row);
        split.pending.push(pending(row));
        let plan = lifecycle.begin_commit(&[]).unwrap();
        split.commit(&plan).unwrap();
        lifecycle.complete_commit(plan).unwrap();
    }
    assert_eq!(staging.rows.history(7), split.rows.history(7));
}
