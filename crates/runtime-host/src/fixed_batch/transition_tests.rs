use super::trace::{Item, Mode, Step, Trace};
use super::transition::{History, VERSION};

fn trace() -> Trace {
    let mut trace = Trace::construct(vec![vec![3; 16], vec![7; 16]], 1024, false);
    trace.version = VERSION;
    trace.steps.truncate(1);
    for item in &mut trace.steps[0].items {
        item.history.clear();
    }
    let mut emitted = [0; 2];
    for (step, (drafts, kept)) in std::iter::repeat_n((0, 1), 270)
        .chain([(1, 2), (3, 4), (1, 1)])
        .enumerate()
    {
        let items = (0..2)
            .map(|index| {
                let slot = (index + step) % 2;
                let item = Item {
                    slot,
                    prefill: false,
                    anchor: false,
                    inputs: vec![11 + slot as u32],
                    drafts,
                    fixed_drafts: Vec::new(),
                    kept,
                    history: Vec::new(),
                    emitted: emitted[slot] + 1,
                };
                emitted[slot] += kept;
                item
            })
            .collect();
        trace.steps.push(Step {
            mode: Mode::Seeded,
            items,
        });
    }
    trace
}

#[test]
fn transition_histories_preserve_accepted_proposals_across_two_wraps_and_resumed_drafts() {
    let trace = trace();
    trace.validate().unwrap();
    assert_eq!(trace.final_tokens(), 16 + 270 + 2 + 4 + 1);
    let mut history = History::new(&trace);
    let mut expected = [Vec::new(), Vec::new()];
    let mut materialized = trace.clone();
    // The old bound uses explicit histories and must agree with the implicit one.
    materialized.version = 3;
    for (index, step) in trace.steps.iter().enumerate() {
        let actual = history.materialize(step).unwrap();
        let mut proposals = Vec::new();
        for item in &actual.items {
            expected[item.slot].extend_from_slice(&item.inputs);
            assert_eq!(item.history, expected[item.slot]);
            if !item.prefill {
                assert_eq!(item.emitted, item.history.len() - 16);
            }
            if item.drafts > 0 {
                let tokens: Vec<_> = (0..item.drafts)
                    .map(|draft| 200 + (index + item.slot + draft) as u32)
                    .collect();
                expected[item.slot].extend_from_slice(&tokens[..item.kept - 1]);
                proposals.push((item.slot as u64 + 1, tokens));
            }
        }
        history = history.prepare_commit(&actual, &proposals).unwrap();
        materialized.steps[index] = actual;
    }
    assert_eq!(trace.metadata_bound(), materialized.metadata_bound());
    assert!(trace.metadata_bound() > 1 << 20);
    let old = Trace::construct(vec![vec![3; 16]], 1024, true);
    old.validate().unwrap();
    let mut too_long = old.clone();
    too_long.steps.push(old.steps[12].clone());
    assert!(too_long.validate().is_err());
}

#[test]
fn transition_validation_rejects_unbounded_or_ambiguous_histories_before_execution() {
    let good = trace();
    let invalid = |change: fn(&mut Trace)| {
        let mut bad = good.clone();
        change(&mut bad);
        assert!(bad.validate().is_err());
    };
    invalid(|t| t.steps.resize(1025, t.steps[1].clone()));
    invalid(|t| t.steps[1].items[0].history.push(3));
    invalid(|t| t.steps[1].items[0].inputs.push(3));
    invalid(|t| t.steps[1].items[0].drafts = usize::MAX);
    invalid(|t| t.steps[1].items[0].emitted += 1);
    invalid(|t| t.steps[1].items[0].inputs[0] = 1024);
    invalid(|t| t.steps[1].items[0].slot = usize::MAX);
    invalid(|t| t.steps[1].items[0].slot = t.steps[1].items[1].slot);
    invalid(|t| t.steps[1].items[0].kept = 0);
    invalid(|t| t.steps[271].items[0].kept = 3);
    invalid(|t| t.steps[271].items[0].fixed_drafts.push(4));
    invalid(|t| t.steps[1].items[0].anchor = true);
    invalid(|t| t.steps[1] = t.steps[0].clone());
    invalid(|t| t.steps[0].items.pop().map(|_| ()).unwrap());
    invalid(|t| t.steps[0].items[0].inputs.resize(8193, 1));
    let mut sparse = good;
    // An absent item stays live; the present item's history still advances.
    sparse.steps.last_mut().unwrap().items.pop();
    sparse.validate().unwrap();
}

#[test]
fn transition_history_preflights_every_proposal_without_mutating_committed_state() {
    let trace = trace();
    let history = History::new(&trace);
    let prefill = history.materialize(&trace.steps[0]).unwrap();
    let history = history.prepare_commit(&prefill, &[]).unwrap();
    let mut step = trace.steps[1].clone();
    for item in &mut step.items {
        item.drafts = 1;
        item.kept = 2;
    }
    let materialized = history.materialize(&step).unwrap();
    let good = vec![(1, vec![53]), (2, vec![61])];
    for proposals in [
        vec![],
        vec![(1, vec![53])],
        vec![(1, vec![53]), (1, vec![61])],
        vec![(1, vec![53, 54]), (2, vec![61])],
        vec![(1, vec![53]), (2, vec![1024])],
        vec![(1, vec![53]), (2, vec![61]), (3, vec![62])],
    ] {
        assert!(history.prepare_commit(&materialized, &proposals).is_err());
        assert_eq!(history.materialize(&step).unwrap(), materialized);
    }
    let mut changed = materialized.clone();
    changed.items[0].history[0] ^= 1;
    assert!(history.prepare_commit(&changed, &good).is_err());
    let next = history.prepare_commit(&materialized, &good).unwrap();
    let resumed = next.materialize(&step).unwrap();
    assert_eq!(resumed.items[0].history[17], 53);
    assert_eq!(resumed.items[1].history[17], 61);
    assert_eq!(history.materialize(&step).unwrap(), materialized);
}
