//! Matcher configurations: the stacks of rule states a prefix can be in.
//!
//! A stack is a persistent linked list of frames, so copying a configuration
//! for a fork or a rollback point copies only its top pointers. Two stacks
//! with the same frames compare equal however they were reached, so an
//! ambiguous grammar does not multiply identical stacks.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::automaton::{Automaton, DEAD};
use crate::budget::Budget;
use crate::expr::GrammarError;

/// A call beyond this depth fails the request, including other live paths.
pub(crate) const MAX_DEPTH: u32 = 2048;
/// Includes duplicate visits and pending calls, so epsilon fanout is bounded too.
pub(crate) const MAX_CLOSURE_WORK: usize = 4096;

#[derive(Debug)]
pub(crate) struct Frame {
    pub(crate) state: u32,
    pub(crate) parent: Option<Arc<Frame>>,
    pub(crate) depth: u32,
    hash: u64,
}

impl Frame {
    pub(crate) fn new(state: u32, parent: Option<Arc<Frame>>) -> Arc<Self> {
        let (depth, parent_hash) = parent
            .as_ref()
            .map_or((1, 0), |parent| (parent.depth + 1, parent.hash));
        let hash = (u64::from(state) ^ parent_hash.rotate_left(17))
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(u64::from(depth));
        Arc::new(Self {
            state,
            parent,
            depth,
            hash,
        })
    }
}

fn same_stack(left: &Arc<Frame>, right: &Arc<Frame>) -> bool {
    let mut left = Some(left);
    let mut right = Some(right);
    loop {
        match (left, right) {
            (None, None) => return true,
            (Some(a), Some(b)) => {
                if Arc::ptr_eq(a, b) {
                    return true;
                }
                if a.hash != b.hash || a.state != b.state || a.depth != b.depth {
                    return false;
                }
                left = a.parent.as_ref();
                right = b.parent.as_ref();
            }
            _ => return false,
        }
    }
}

struct StackKey(Arc<Frame>);

impl PartialEq for StackKey {
    fn eq(&self, other: &Self) -> bool {
        same_stack(&self.0, &other.0)
    }
}

impl Eq for StackKey {}

impl Hash for StackKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash.hash(state);
    }
}

/// Every stack a prefix can be in whose top state reads another byte, and
/// whether the prefix is already a complete match.
#[derive(Clone, Debug, Default)]
pub(crate) struct Config {
    pub(crate) stacks: Vec<Arc<Frame>>,
    pub(crate) complete: bool,
    pub(crate) failure: Option<GrammarError>,
}

impl PartialEq for Config {
    fn eq(&self, other: &Self) -> bool {
        self.failure == other.failure
            && self.complete == other.complete
            && self.stacks.len() == other.stacks.len()
            && self
                .stacks
                .iter()
                .zip(&other.stacks)
                .all(|(left, right)| same_stack(left, right))
    }
}

impl Config {
    fn failed(error: GrammarError) -> Self {
        Self {
            failure: Some(error),
            ..Self::default()
        }
    }

    pub(crate) fn initial(automaton: &Automaton) -> Self {
        closure(
            automaton,
            vec![Frame::new(automaton.root_start(), None)],
            None,
        )
    }

    pub(crate) fn initial_budgeted(automaton: &Automaton, budget: &Budget) -> Self {
        closure(
            automaton,
            vec![Frame::new(automaton.root_start(), None)],
            Some(budget),
        )
    }

    pub(crate) fn is_dead(&self) -> bool {
        self.stacks.is_empty() && !self.complete
    }

    /// A configuration of one of this configuration's stacks.
    pub(crate) fn single(stack: &Arc<Frame>) -> Self {
        Self {
            stacks: vec![Arc::clone(stack)],
            complete: false,
            failure: None,
        }
    }

    pub(crate) fn step(&self, automaton: &Automaton, byte: u8) -> Self {
        self.step_with_budget(automaton, byte, None)
    }

    pub(crate) fn step_budgeted(&self, automaton: &Automaton, byte: u8, budget: &Budget) -> Self {
        self.step_with_budget(automaton, byte, Some(budget))
    }

    fn step_with_budget(&self, automaton: &Automaton, byte: u8, budget: Option<&Budget>) -> Self {
        if let Some(budget) = budget
            && let Err(error) = budget.charge(self.stacks.len(), 0)
        {
            return Self::failed(error);
        }
        if self.failure.is_some() {
            return self.clone();
        }
        let seeds = self
            .stacks
            .iter()
            .filter_map(|frame| {
                let next = automaton.next(frame.state, byte);
                (next != DEAD).then(|| Frame::new(next, frame.parent.clone()))
            })
            .collect();
        closure(automaton, seeds, budget)
    }

    pub(crate) fn step_bytes(&self, automaton: &Automaton, bytes: &[u8]) -> Self {
        let mut config = self.clone();
        for &byte in bytes {
            if config.failure.is_some() {
                return config;
            }
            if config.stacks.is_empty() {
                return Self::default();
            }
            config = config.step(automaton, byte);
        }
        config
    }
}

/// Expands seed frames through returns and calls until every stack's top can
/// read a byte.
fn closure(automaton: &Automaton, seeds: Vec<Arc<Frame>>, budget: Option<&Budget>) -> Config {
    if let Some(budget) = budget
        && let Err(error) = budget
            .charge(seeds.len(), 0)
            .and_then(|()| budget.scratch(MAX_CLOSURE_WORK * 64))
    {
        return Config::failed(error);
    }
    let mut seen = HashSet::new();
    let mut pending = seeds;
    let mut stacks = Vec::new();
    let mut complete = false;
    let mut work = pending.len();
    while let Some(frame) = pending.pop() {
        let calls = automaton.state(frame.state).calls.len();
        // Charge before allocating frames or growing the pending stack.
        work = work.saturating_add(1 + calls);
        if work > MAX_CLOSURE_WORK {
            return Config::failed(GrammarError::too_large(format!(
                "grammar matcher work budget exceeded (limit: {MAX_CLOSURE_WORK} units per closure)"
            )));
        }
        if let Some(budget) = budget
            && let Err(error) = budget.charge(1 + calls, 0)
        {
            return Config::failed(error);
        }
        if !seen.insert(StackKey(Arc::clone(&frame))) {
            continue;
        }
        let state = automaton.state(frame.state);
        if state.reads {
            stacks.push(Arc::clone(&frame));
        }
        if state.accepting {
            match &frame.parent {
                None => complete = true,
                Some(parent) => pending.push(Arc::clone(parent)),
            }
        }
        if frame.depth >= MAX_DEPTH && !state.calls.is_empty() {
            return Config::failed(GrammarError::too_large(format!(
                "grammar matcher depth budget exceeded (limit: {MAX_DEPTH} frames)"
            )));
        }
        if frame.depth < MAX_DEPTH {
            for &(rule, ret) in state.calls.iter() {
                let caller = Frame::new(ret, frame.parent.clone());
                pending.push(Frame::new(
                    automaton.rules[rule as usize].start,
                    Some(caller),
                ));
            }
        }
    }
    stacks.sort_by_key(|frame| frame.hash);
    Config {
        stacks,
        complete,
        failure: None,
    }
}
