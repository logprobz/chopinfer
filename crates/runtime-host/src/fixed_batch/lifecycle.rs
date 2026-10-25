//! Explicit request lifetimes for bounded prefix and cancellation replays.
use super::trace::Trace;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const VERSION: u32 = 7;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Boundary {
    /// One-based step number. Actions run while the instance is Ready.
    pub before_step: usize,
    pub actions: Vec<Action>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cancel: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    Capture { slot: usize },
    Retire { slot: usize },
    Restore { slot: usize, from: usize },
}

pub struct Analysis {
    /// Includes provisional rows, even when their step is canceled.
    pub peak_tokens: usize,
    pub histories: Vec<Vec<usize>>,
    pub captures: usize,
}

#[derive(Clone, Copy)]
struct Lifetime {
    tokens: usize,
    prompt: usize,
}

pub fn analyze(trace: &Trace) -> Result<Analysis, String> {
    if trace.version != VERSION
        || !(1..=32).contains(&trace.concurrency)
        || !(1024..=1 << 20).contains(&trace.vocabulary)
        || !(2..=1024).contains(&trace.steps.len())
        || !(1..=16).contains(&trace.lifecycle.len())
    {
        return Err("unsupported lifecycle trace geometry".into());
    }
    let mut previous = 0;
    for boundary in &trace.lifecycle {
        if boundary.before_step <= previous
            || boundary.before_step > trace.steps.len()
            || boundary.actions.len() > 8
            || (boundary.actions.is_empty() && !boundary.cancel)
        {
            return Err("lifecycle boundaries must be ordered, bounded and nonempty".into());
        }
        previous = boundary.before_step;
    }
    let mut live = vec![None::<Lifetime>; trace.concurrency];
    let mut saved = live.clone();
    let mut prefilled = BTreeSet::new();
    let mut counts = [0; 4];
    let mut drafted = false;
    let mut analysis = Analysis {
        peak_tokens: 0,
        histories: Vec::new(),
        captures: 0,
    };
    for (index, step) in trace.steps.iter().enumerate() {
        let boundary = trace.lifecycle.iter().find(|b| b.before_step == index + 1);
        for action in boundary.into_iter().flat_map(|b| &b.actions) {
            match *action {
                Action::Capture { slot } => {
                    let state = live
                        .get(slot)
                        .copied()
                        .flatten()
                        .ok_or("capture has no live prefix")?;
                    if saved[slot].replace(state).is_some() || counts[0] == 8 {
                        return Err("duplicate or excessive saved prefix".into());
                    }
                    counts[0] += 1;
                }
                Action::Retire { slot } => {
                    live.get_mut(slot)
                        .ok_or("unknown retired slot")?
                        .take()
                        .ok_or("retire has no live request")?;
                    counts[1] += 1;
                }
                Action::Restore { slot, from } => {
                    let state = saved
                        .get(from)
                        .copied()
                        .flatten()
                        .ok_or("restore has no saved prefix")?;
                    if live
                        .get_mut(slot)
                        .ok_or("unknown restored slot")?
                        .replace(state)
                        .is_some()
                    {
                        return Err("restore would overwrite a live request".into());
                    }
                    counts[2] += 1;
                }
            }
        }
        let prefill = step.items.first().ok_or("empty lifecycle step")?.prefill;
        if step.items.len() > 32 || step.rows() > 256 {
            return Err("lifecycle step exceeds its row bound".into());
        }
        let mut seen = BTreeSet::new();
        let mut histories = vec![0; trace.concurrency];
        for item in &step.items {
            if item.slot >= trace.concurrency
                || !seen.insert(item.slot)
                || item.prefill != prefill
                || item.anchor
                || item.drafts > 7
                || !item.fixed_drafts.is_empty()
                || !item.history.is_empty()
                || item.inputs.is_empty()
                || item.inputs.len() > 8192
                || item
                    .inputs
                    .iter()
                    .any(|&token| token as usize >= trace.vocabulary)
                || (prefill && (item.drafts != 0 || item.kept != item.inputs.len()))
                || (!prefill
                    && (item.inputs.len() != 1 || !(1..=item.drafts + 1).contains(&item.kept)))
            {
                return Err("invalid lifecycle trace item".into());
            }
            let state = if prefill {
                if live[item.slot].is_some() {
                    return Err("prefill would overwrite a live lifetime".into());
                }
                prefilled.insert(item.slot);
                Lifetime {
                    tokens: 0,
                    prompt: item.inputs.len(),
                }
            } else {
                live[item.slot].ok_or("decode has no live lifetime")?
            };
            let length = state.tokens + item.inputs.len();
            if length.checked_sub(item.emitted) != Some(state.prompt) {
                return Err("lifecycle prompt/generated split drifted".into());
            }
            histories[item.slot] = length;
            analysis.peak_tokens = analysis.peak_tokens.max(length + item.drafts);
            live[item.slot] = if boundary.is_some_and(|b| b.cancel) {
                None
            } else {
                Some(Lifetime {
                    tokens: state.tokens + item.kept,
                    ..state
                })
            };
            drafted |= item.drafts > 0;
        }
        if prefill && step.items.iter().map(|i| i.inputs.len()).sum::<usize>() > 8192 {
            return Err("lifecycle prefill exceeds its token bound".into());
        }
        counts[3] += usize::from(boundary.is_some_and(|b| b.cancel));
        analysis.histories.push(histories);
    }
    if prefilled.len() != trace.concurrency || !drafted || counts.contains(&0) {
        return Err(
            "lifecycle trace must exercise capture, retire, restore, cancellation and drafting"
                .into(),
        );
    }
    analysis.captures = counts[0];
    Ok(analysis)
}

#[cfg(test)]
mod tests {
    use super::super::trace::{Mode, Step};
    use super::super::transition::History;
    use super::*;

    fn fixture() -> Trace {
        let mut trace = Trace::construct(vec![vec![3; 16], vec![7; 16]], 1024, false);
        trace.version = VERSION;
        trace.steps.truncate(1);
        for item in &mut trace.steps[0].items {
            item.history.clear();
        }
        let item = trace.steps[0].items[0].clone();
        let mut lengths = [16; 2];
        for (depth, kept) in std::iter::repeat_n((0, 1), 270).chain([(1, 1), (1, 2), (3, 4)]) {
            trace.steps.push(Step {
                mode: Mode::Seeded,
                items: (0..2)
                    .map(|slot| {
                        let mut item = item.clone();
                        item.slot = slot;
                        item.prefill = false;
                        item.inputs = vec![11 + slot as u32];
                        item.drafts = depth;
                        item.kept = kept;
                        item.emitted = lengths[slot] + 1 - 16;
                        lengths[slot] += kept;
                        item
                    })
                    .collect(),
            });
        }
        let mut cancel = trace.steps.last().unwrap().clone();
        cancel.items.remove(0);
        cancel.items[0].emitted = lengths[1] + 1 - 16;
        trace.steps.push(cancel);
        let canceled = trace.steps.len();
        trace.lifecycle.push(Boundary {
            before_step: canceled,
            actions: vec![Action::Capture { slot: 0 }],
            cancel: true,
        });
        let mut resumed = trace.steps.last().unwrap().clone();
        resumed.items = (0..2)
            .map(|slot| {
                let mut item = resumed.items[0].clone();
                item.slot = slot;
                item.drafts = 1;
                item.kept = 2;
                item.emitted = lengths[0] + 1 - 16;
                item
            })
            .collect();
        trace.steps.push(resumed);
        trace.lifecycle.push(Boundary {
            before_step: canceled + 1,
            actions: vec![
                Action::Retire { slot: 0 },
                Action::Restore { slot: 1, from: 0 },
                Action::Restore { slot: 0, from: 0 },
            ],
            cancel: false,
        });
        trace
    }

    #[test]
    fn lifecycle_lengths_and_histories_survive_cancel_restore_and_slot_swap() {
        let trace = fixture();
        trace.validate().unwrap();
        let analysis = analyze(&trace).unwrap();
        // The canceled D3 has more provisional tokens than the final D1.
        assert_eq!(analysis.peak_tokens, 16 + 270 + 1 + 2 + 4 + 4);
        let mut history = History::new(&trace);
        let mut saved = None;
        let mut slots = crate::recurrent::StateSlotPoolV1::for_test(2);
        for id in 1..=2 {
            slots.allocate(runtime_engine::RequestId::new(id)).unwrap();
        }
        for (index, step) in trace.steps.iter().enumerate() {
            let boundary = trace.lifecycle.iter().find(|b| b.before_step == index + 1);
            for action in boundary.into_iter().flat_map(|b| &b.actions) {
                match *action {
                    Action::Capture { slot } => {
                        saved = Some(history.tokens(slot).unwrap().to_vec())
                    }
                    Action::Retire { slot } => {
                        history.replace(slot, Vec::new()).unwrap();
                        slots.retire(runtime_engine::RequestId::new(slot as u64 + 1));
                    }
                    Action::Restore { slot, .. } => {
                        history.replace(slot, saved.clone().unwrap()).unwrap();
                        assert_eq!(
                            slots
                                .allocate(runtime_engine::RequestId::new(slot as u64 + 1))
                                .unwrap(),
                            1 - slot as u64
                        );
                    }
                }
            }
            let actual = history.materialize(step).unwrap();
            for item in &actual.items {
                assert_eq!(item.history.len(), analysis.histories[index][item.slot]);
            }
            let proposals = actual
                .items
                .iter()
                .filter(|i| i.drafts > 0)
                .map(|i| (i.slot as u64 + 1, vec![67; i.drafts]))
                .collect::<Vec<_>>();
            let next = history.prepare_commit(&actual, &proposals).unwrap();
            if boundary.is_some_and(|b| b.cancel) {
                for item in &actual.items {
                    history.replace(item.slot, Vec::new()).unwrap();
                    slots.retire(runtime_engine::RequestId::new(item.slot as u64 + 1));
                }
            } else {
                history = next;
            }
        }
        assert_eq!(
            history.tokens(0).unwrap().len(),
            history.tokens(1).unwrap().len()
        );
        assert!(trace.metadata_bound() > 1 << 20);
        let mut old = Trace::construct(vec![vec![3; 16]], 1024, false);
        assert!(
            serde_json::to_value(&old)
                .unwrap()
                .get("lifecycle")
                .is_none()
        );
        old.lifecycle = trace.lifecycle;
        assert!(old.validate().unwrap_err().contains("own trace version"));
    }

    #[test]
    fn lifecycle_refuses_stale_lifetimes_and_unbounded_actions() {
        let good = fixture();
        for mutation in 0..9 {
            let mut bad = good.clone();
            match mutation {
                0 => {
                    bad.lifecycle[1].actions.remove(0);
                }
                1 => {
                    bad.lifecycle[0].cancel = false;
                }
                2 => {
                    bad.lifecycle[1].before_step = 1;
                }
                3 => {
                    bad.lifecycle[0].actions[0] = Action::Capture { slot: 33 };
                }
                4 => {
                    bad.lifecycle[1].actions[1] = Action::Restore { slot: 1, from: 1 };
                }
                5 => {
                    bad.steps.last_mut().unwrap().items[0].emitted += 1;
                }
                6 => {
                    bad.lifecycle[0]
                        .actions
                        .resize(9, Action::Capture { slot: 0 });
                }
                7 => {
                    bad.lifecycle[0].actions.push(Action::Capture { slot: 0 });
                }
                _ => {
                    bad.steps[1].items[0].inputs.push(1);
                }
            };
            assert!(bad.validate().is_err(), "mutation {mutation}");
        }
    }
}
