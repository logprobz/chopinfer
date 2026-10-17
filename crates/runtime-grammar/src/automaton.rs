//! Compiling a grammar to deterministic byte automata with rule calls.
//!
//! Rules that take part in a reference cycle stay separate automata that the
//! matcher calls with a stack. Every other rule reference is expanded in
//! place, once for each state it continues to, so a grammar without recursion
//! becomes one automaton. Each automaton is a subset construction over its
//! Thompson automaton, and states that cannot reach an accepting state are
//! removed, so any live state can still finish.

use std::collections::HashMap;

use crate::budget::Budget;
use crate::expr::{Expr, Grammar, GrammarError, RuleId};
use crate::utf8::scalar_range_sequences;

/// No state: the input so far matches nothing.
pub(crate) const DEAD: u32 = u32::MAX;

pub(crate) const MAX_NFA_STATES: usize = 1 << 21;
pub(crate) const MAX_TABLE_ENTRIES: usize = 1 << 24;
const MAX_REPEAT: u32 = 1 << 16;

#[derive(Debug)]
pub(crate) struct DfaState {
    pub(crate) rule: u32,
    pub(crate) accepting: bool,
    /// Whether any byte leads somewhere.
    pub(crate) reads: bool,
    next: Box<[u32]>,
    /// Calls this state can make: the called rule and the state to return to.
    pub(crate) calls: Box<[(u32, u32)]>,
}

#[derive(Debug)]
pub(crate) struct CompiledRule {
    pub(crate) start: u32,
    classes: Box<[u8; 256]>,
}

/// Every compiled rule of one grammar, with states numbered globally.
#[derive(Debug)]
pub(crate) struct Automaton {
    pub(crate) states: Vec<DfaState>,
    pub(crate) rules: Vec<CompiledRule>,
    pub(crate) root: u32,
    whitespace: Vec<Option<(u32, usize)>>,
}

impl Automaton {
    /// Heap bytes of the compiled states and rules, an upper estimate.
    pub(crate) fn retained_bytes(&self) -> usize {
        use crate::heap::vec_bytes;
        let transitions = self
            .states
            .iter()
            .map(|state| {
                vec_bytes::<u32>(state.next.len()) + vec_bytes::<(u32, u32)>(state.calls.len())
            })
            .sum::<usize>();
        vec_bytes::<DfaState>(self.states.capacity())
            + transitions
            + vec_bytes::<CompiledRule>(self.rules.capacity())
            + self.rules.len() * 256
            + vec_bytes::<Option<(u32, usize)>>(self.whitespace.capacity())
    }

    #[inline]
    pub(crate) fn next(&self, state: u32, byte: u8) -> u32 {
        let record = &self.states[state as usize];
        let class = self.rules[record.rule as usize].classes[byte as usize];
        record.next[class as usize]
    }

    pub(crate) fn state(&self, state: u32) -> &DfaState {
        &self.states[state as usize]
    }

    pub(crate) fn root_start(&self) -> u32 {
        self.rules[self.root as usize].start
    }

    /// Maximum stack growth in one epsilon closure, after a byte is read.
    /// Returns MAX_DEPTH for epsilon cycles, where no smaller bound is proven.
    pub(crate) fn max_call_growth(&self) -> u32 {
        use crate::stack::MAX_DEPTH;

        // A return edge is reachable without reading only when its callee
        // can accept empty input. Solve that property before building edges.
        let mut nullable: Vec<_> = self.states.iter().map(|state| state.accepting).collect();
        loop {
            let mut changed = false;
            for (id, state) in self.states.iter().enumerate() {
                if !nullable[id]
                    && state.calls.iter().any(|&(rule, ret)| {
                        nullable[self.rules[rule as usize].start as usize] && nullable[ret as usize]
                    })
                {
                    nullable[id] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // Calls add one frame; returns to this frame add none. Process the
        // graph from its leaves, retaining the longest weighted path.
        let mut predecessors = vec![Vec::new(); self.states.len()];
        let mut pending = vec![0; self.states.len()];
        for (id, state) in self.states.iter().enumerate() {
            for &(rule, ret) in state.calls.iter() {
                let start = self.rules[rule as usize].start as usize;
                predecessors[start].push((id, 1));
                pending[id] += 1;
                if nullable[start] {
                    predecessors[ret as usize].push((id, 0));
                    pending[id] += 1;
                }
            }
        }
        let mut ready: Vec<_> = pending
            .iter()
            .enumerate()
            .filter_map(|(id, &count)| (count == 0).then_some(id))
            .collect();
        let mut growth = vec![0_u32; self.states.len()];
        let mut visited = 0;
        while let Some(id) = ready.pop() {
            visited += 1;
            for &(parent, cost) in &predecessors[id] {
                growth[parent] = growth[parent].max((growth[id] + cost).min(MAX_DEPTH));
                pending[parent] -= 1;
                if pending[parent] == 0 {
                    ready.push(parent);
                }
            }
        }
        if visited != self.states.len() {
            MAX_DEPTH
        } else {
            growth.into_iter().max().unwrap_or(0)
        }
    }

    /// A finite, nonaccepting whitespace chain with identical exits. Tokens
    /// whose whitespace prefix exceeds its budget are the only differences
    /// from the representative with the longest budget.
    pub(crate) fn whitespace_class(&self, state: u32) -> Option<(u32, usize)> {
        self.whitespace[state as usize]
    }

    pub(crate) fn classification_states(&self, horizon: usize) -> Vec<u32> {
        let mut groups = HashMap::new();
        let mut signatures = Vec::with_capacity(self.states.len());
        for (index, state) in self.states.iter().enumerate() {
            let exits: Vec<_> = (0..=255u8)
                .filter(|byte| !b" \n\t\r".contains(byte))
                .map(|byte| self.next(index as u32, byte))
                .collect();
            let key = (state.accepting, state.calls.clone(), exits);
            let next = groups.len();
            signatures.push(*groups.entry(key).or_insert(next));
        }
        let successor: Vec<_> = (0..self.states.len())
            .map(|index| {
                let next = self.next(index as u32, b' ');
                (next != DEAD
                    && signatures[index] == signatures[next as usize]
                    && b"\n\t\r"
                        .iter()
                        .all(|&byte| self.next(index as u32, byte) == next))
                .then_some(next)
            })
            .collect();
        let mut surviving = vec![true; self.states.len()];
        for _ in 0..horizon {
            let next: Vec<_> = successor
                .iter()
                .map(|next| next.is_some_and(|next| surviving[next as usize]))
                .collect();
            if next == surviving {
                break;
            }
            surviving = next;
        }
        let mut representatives = HashMap::new();
        surviving
            .iter()
            .enumerate()
            .map(|(index, &long)| {
                if long {
                    *representatives
                        .entry(signatures[index])
                        .or_insert(index as u32)
                } else {
                    index as u32
                }
            })
            .collect()
    }

    fn whitespace_classes(&self) -> Vec<Option<(u32, usize)>> {
        let mut groups = HashMap::new();
        let mut signatures = vec![None; self.states.len()];
        for (index, state) in self.states.iter().enumerate() {
            let next = self.next(index as u32, b' ');
            if state.accepting
                || !state.calls.is_empty()
                || !b"\n\t\r"
                    .iter()
                    .all(|&byte| self.next(index as u32, byte) == next)
            {
                continue;
            }
            let exits: Vec<_> = (0..=255u8)
                .filter(|byte| !b" \n\t\r".contains(byte))
                .map(|byte| self.next(index as u32, byte))
                .collect();
            let next = groups.len();
            signatures[index] = Some(*groups.entry(exits).or_insert(next));
        }
        let mut incoming = vec![Vec::new(); self.states.len()];
        let mut budgets = vec![None; self.states.len()];
        let mut pending = std::collections::VecDeque::new();
        for (index, signature) in signatures.iter().enumerate() {
            if signature.is_none() {
                continue;
            }
            let next = self.next(index as u32, b' ');
            if next == DEAD {
                budgets[index] = Some(0);
                pending.push_back(index);
            } else if signatures[next as usize] == *signature {
                incoming[next as usize].push(index);
            }
        }
        while let Some(index) = pending.pop_front() {
            for &source in &incoming[index] {
                budgets[source] = Some(budgets[index].unwrap() + 1);
                pending.push_back(source);
            }
        }
        let mut representatives = vec![(0, 0); groups.len()];
        for (index, budget) in budgets.iter().enumerate() {
            if let Some(budget) = budget {
                let best = &mut representatives[signatures[index].unwrap()];
                if *budget > best.1 {
                    *best = (index as u32, *budget);
                }
            }
        }
        budgets
            .iter()
            .enumerate()
            .map(|(index, budget)| {
                budget.and_then(|budget| {
                    let (representative, longest) = representatives[signatures[index].unwrap()];
                    (longest > 0).then_some((representative, budget))
                })
            })
            .collect()
    }

    /// Whether any compiled rule can be called, which means a match needs a
    /// stack rather than a single state.
    #[cfg(test)]
    pub(crate) fn has_calls(&self) -> bool {
        self.states.iter().any(|state| !state.calls.is_empty())
    }

    #[cfg(test)]
    pub(crate) fn compile(grammar: &Grammar) -> Result<Self, GrammarError> {
        Self::compile_with_budget(grammar, &Budget::default())
    }

    pub(crate) fn compile_with_budget(
        grammar: &Grammar,
        budget: &Budget,
    ) -> Result<Self, GrammarError> {
        budget.charge(
            grammar.rules.len().saturating_mul(grammar.rules.len()),
            grammar.rules.len().saturating_mul(256),
        )?;
        let recursive = recursive_rules(grammar, budget)?;
        let mut compiled_index = HashMap::new();
        let mut order = vec![grammar.root];
        compiled_index.insert(grammar.root, 0_u32);
        for (index, is_recursive) in recursive.iter().enumerate() {
            let id = RuleId(index as u32);
            if *is_recursive && id != grammar.root {
                compiled_index.insert(id, order.len() as u32);
                order.push(id);
            }
        }
        let mut builder = NfaBuilder {
            grammar,
            budget,
            compiled_index: &compiled_index,
            states: Vec::new(),
            expanding: Vec::new(),
            expanded: HashMap::new(),
            optional: HashMap::new(),
        };
        let mut nfas = Vec::with_capacity(order.len());
        for &id in &order {
            nfas.push(builder.build_rule(id)?);
        }
        let nfa = builder.states;
        let mut states = Vec::new();
        let mut rules = Vec::with_capacity(order.len());
        let mut entries = 0_usize;
        for (rule, &built) in nfas.iter().enumerate() {
            let compiled =
                determinize(&nfa, rule as u32, built, &mut states, &mut entries, budget)?;
            rules.push(compiled);
        }
        let mut automaton = Self {
            states,
            rules,
            root: 0,
            whitespace: Vec::new(),
        };
        // Reserve work for partition refinement and its predecessor scans.
        budget.charge(
            entries.saturating_mul(states_log_bound(automaton.states.len())),
            0,
        )?;
        budget.automaton(nfa.len(), entries);
        automaton.prune_unproductive(budget)?;
        automaton.minimize();
        automaton.whitespace = automaton.whitespace_classes();
        budget.report("automaton", grammar);
        Ok(automaton)
    }

    /// Minimize within each rule, treating calls as labelled return edges.
    /// Refining only predecessors of a split avoids a full-table pass for
    /// every character of a bounded repetition.
    fn minimize(&mut self) {
        let mut blocks: Vec<Vec<u32>> = Vec::new();
        let mut initial = HashMap::new();
        let mut owner = Vec::with_capacity(self.states.len());
        let mut offsets = Vec::with_capacity(self.states.len());
        let mut incoming = vec![Vec::new(); self.states.len()];
        for (index, state) in self.states.iter().enumerate() {
            let block = *initial
                .entry((state.rule, state.accepting))
                .or_insert_with(|| {
                    blocks.push(Vec::new());
                    blocks.len() - 1
                });
            owner.push(block);
            offsets.push(blocks[block].len());
            blocks[block].push(index as u32);
            for (label, &next) in state.next.iter().enumerate() {
                if next != DEAD {
                    incoming[next as usize].push((index as u32, label));
                }
            }
            for &(rule, ret) in state.calls.iter() {
                incoming[ret as usize].push((index as u32, 256 + rule as usize));
            }
        }
        let mut pending: std::collections::VecDeque<_> = (0..blocks.len()).collect();
        let mut queued = vec![true; blocks.len()];
        while let Some(splitter) = pending.pop_front() {
            queued[splitter] = false;
            let mut labels: HashMap<usize, Vec<u32>> = HashMap::new();
            for &target in &blocks[splitter] {
                for &(source, label) in &incoming[target as usize] {
                    labels.entry(label).or_default().push(source);
                }
            }
            for sources in labels.values() {
                let mut affected: HashMap<usize, Vec<u32>> = HashMap::new();
                for &source in sources {
                    affected
                        .entry(owner[source as usize])
                        .or_default()
                        .push(source);
                }
                for (block, subset) in affected {
                    if subset.len() == blocks[block].len() {
                        continue;
                    }
                    let new = blocks.len();
                    for &state in &subset {
                        let offset = offsets[state as usize];
                        blocks[block].swap_remove(offset);
                        if let Some(&moved) = blocks[block].get(offset) {
                            offsets[moved as usize] = offset;
                        }
                        owner[state as usize] = new;
                    }
                    for (offset, &state) in subset.iter().enumerate() {
                        offsets[state as usize] = offset;
                    }
                    blocks.push(subset);
                    queued.push(false);
                    let enqueue = if queued[block] || blocks[new].len() <= blocks[block].len() {
                        new
                    } else {
                        block
                    };
                    queued[enqueue] = true;
                    pending.push_back(enqueue);
                }
            }
        }
        let old = std::mem::take(&mut self.states);
        self.states = old
            .into_iter()
            .enumerate()
            .filter_map(|(index, mut state)| {
                (blocks[owner[index]][0] == index as u32).then(|| {
                    for next in state.next.iter_mut() {
                        if *next != DEAD {
                            *next = blocks[owner[*next as usize]][0];
                        }
                    }
                    for (_, ret) in state.calls.iter_mut() {
                        *ret = blocks[owner[*ret as usize]][0];
                    }
                    (index as u32, state)
                })
            })
            .map(|(_, state)| state)
            .collect();
        // States above remain in original representative order.
        let mut representatives: Vec<_> = blocks.iter().map(|block| block[0]).collect();
        representatives.sort_unstable();
        let mut compact = vec![0; owner.len()];
        for (index, &representative) in representatives.iter().enumerate() {
            compact[representative as usize] = index as u32;
        }
        for state in &mut self.states {
            for next in state.next.iter_mut() {
                if *next != DEAD {
                    *next = compact[*next as usize];
                }
            }
            for (_, ret) in state.calls.iter_mut() {
                *ret = compact[*ret as usize];
            }
        }
        for rule in &mut self.rules {
            rule.start = compact[blocks[owner[rule.start as usize]][0] as usize];
        }
    }

    /// Removes every edge into a state that cannot finish its rule, and every
    /// call whose rule cannot finish.
    fn prune_unproductive(&mut self, budget: &Budget) -> Result<(), GrammarError> {
        // A byte edge needs one productive successor. A call needs both its
        // callee's start and its return state. Notify each dependency once.
        #[derive(Clone, Copy)]
        enum Dependency {
            Byte(usize),
            Call(usize),
        }
        let count = self.states.len();
        budget.charge(count, count.saturating_mul(64))?;
        let mut incoming = vec![Vec::new(); count];
        let mut calls = Vec::new();
        let mut productive = vec![false; count];
        let mut pending = Vec::new();
        for (source, state) in self.states.iter().enumerate() {
            budget.charge(1 + state.next.len() + state.calls.len(), 0)?;
            if state.accepting {
                productive[source] = true;
                pending.push(source);
            }
            for &target in state.next.iter().filter(|&&target| target != DEAD) {
                budget.charge(1, 32)?;
                incoming[target as usize].push(Dependency::Byte(source));
            }
            for &(rule, ret) in state.calls.iter() {
                budget.charge(1, 80)?;
                let call = calls.len();
                calls.push((source, 2_u8));
                incoming[self.rules[rule as usize].start as usize].push(Dependency::Call(call));
                incoming[ret as usize].push(Dependency::Call(call));
            }
        }
        while let Some(target) = pending.pop() {
            budget.charge(1, 0)?;
            for dependency in &incoming[target] {
                budget.charge(1, 0)?;
                let source = match *dependency {
                    Dependency::Byte(source) => source,
                    Dependency::Call(id) => {
                        let (source, remaining) = &mut calls[id];
                        *remaining -= 1;
                        if *remaining != 0 {
                            continue;
                        }
                        *source
                    }
                };
                if !std::mem::replace(&mut productive[source], true) {
                    pending.push(source);
                }
            }
        }
        budget.charge(self.rules.len(), self.rules.len())?;
        let rule_productive: Vec<_> = self
            .rules
            .iter()
            .map(|rule| productive[rule.start as usize])
            .collect();
        if !productive[self.root_start() as usize] {
            return Err(GrammarError::invalid("the constraint matches no output"));
        }
        for state in &mut self.states {
            budget.charge(1 + state.next.len() + state.calls.len(), 0)?;
            for next in state.next.iter_mut() {
                if *next != DEAD && !productive[*next as usize] {
                    *next = DEAD;
                }
            }
            state.calls = state
                .calls
                .iter()
                .copied()
                .filter(|&(rule, ret)| rule_productive[rule as usize] && productive[ret as usize])
                .collect();
            state.reads = state.next.iter().any(|&next| next != DEAD);
        }
        Ok(())
    }
}

/// Marks the rules that can reach themselves through references.
fn recursive_rules(grammar: &Grammar, budget: &Budget) -> Result<Vec<bool>, GrammarError> {
    let count = grammar.rules.len();
    let references: Vec<Vec<u32>> = grammar
        .rules
        .iter()
        .map(|rule| {
            let mut found = Vec::new();
            collect_references(rule, &mut found, budget)?;
            budget.charge(found.len().saturating_mul(states_log_bound(found.len())), 0)?;
            found.sort_unstable();
            found.dedup();
            Ok(found)
        })
        .collect::<Result<_, GrammarError>>()?;
    (0..count)
        .map(|start| {
            budget.scratch(count + references[start].len() * 8)?;
            let mut seen = vec![false; count];
            let mut pending = references[start].clone();
            while let Some(rule) = pending.pop() {
                budget.rule_work(1)?;
                budget.charge(1, 0)?;
                if rule as usize == start {
                    return Ok(true);
                }
                if !std::mem::replace(&mut seen[rule as usize], true) {
                    let next = &references[rule as usize];
                    budget.rule_work(next.len())?;
                    budget.charge(next.len(), 0)?;
                    budget.scratch(count + (pending.len() + next.len()) * 8)?;
                    pending.extend(next);
                }
            }
            Ok(false)
        })
        .collect()
}

fn collect_references(
    expr: &Expr,
    found: &mut Vec<u32>,
    budget: &Budget,
) -> Result<(), GrammarError> {
    budget.rule_work(1)?;
    budget.charge(1, 0)?;
    match expr {
        Expr::Rule(id) => {
            budget.charge(0, 8)?;
            found.push(id.0);
        }
        Expr::Seq(items) | Expr::Alt(items) => {
            for item in items {
                collect_references(item, found, budget)?;
            }
        }
        Expr::Repeat { expr, .. } => collect_references(expr, found, budget)?,
        Expr::OrderedSubset {
            items,
            tail,
            separator,
        } => {
            for (item, _) in items {
                collect_references(item, found, budget)?;
            }
            if let Some(tail) = tail {
                collect_references(tail, found, budget)?;
            }
            collect_references(separator, found, budget)?;
        }
        Expr::Empty
        | Expr::Nothing
        | Expr::Literal(_)
        | Expr::Class(_)
        | Expr::TextExcluding(_)
        | Expr::TextUntil { .. } => {}
    }
    Ok(())
}

#[derive(Default)]
struct NfaState {
    epsilon: Vec<u32>,
    bytes: Vec<(u8, u8, u32)>,
    calls: Vec<(u32, u32)>,
}

/// Builds Thompson automata back to front: each expression is built for the
/// state it continues to and returns its start. Two uses of one expression
/// that continue to the same state are one fragment, so an expanded rule, an
/// ordered subset's item or tail, and a run of bounded repetitions are built
/// once for each place they lead to rather than once for each use.
///
/// Each build also returns its [`Unshared`] count, which determinization
/// growth is charged against, so sharing a fragment refuses no grammar the
/// unshared construction admits.
struct NfaBuilder<'a> {
    grammar: &'a Grammar,
    budget: &'a Budget,
    compiled_index: &'a HashMap<RuleId, u32>,
    states: Vec<NfaState>,
    expanding: Vec<RuleId>,
    /// The start of each expanded rule and its unshared count, by the rule
    /// and the state it continues to.
    expanded: HashMap<(RuleId, u32), (u32, Unshared)>,
    /// For a repeated rule, literal or class and the state its repetitions
    /// continue to, `optional[j]` starts up to `j` more of them; with the
    /// unshared count of one of them.
    optional: HashMap<(&'a Expr, u32), (Vec<u32>, Unshared)>,
}

/// The states the construction that builds every use of an expression
/// separately reaches from the expression's start, and whether it reaches
/// the expression's end. That construction gives each sequence a start state
/// and each alternative, class, repetition, marked text and `Nothing` an end
/// state, builds an ordered subset's items and tail once before any entry and
/// once after a separator, and builds every copy of a repetition.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Unshared {
    states: usize,
    finishes: bool,
}

impl Unshared {
    fn new(states: usize, finishes: bool) -> Self {
        Self { states, finishes }
    }

    /// A start state followed by `items` in order: each item is reached only
    /// when every item before it finishes.
    fn sequence(items: impl IntoIterator<Item = Self>) -> Self {
        let mut sequence = Self::new(1, true);
        for item in items {
            if !sequence.finishes {
                break;
            }
            sequence.states = sequence.states.saturating_add(item.states);
            sequence.finishes = item.finishes;
        }
        sequence
    }

    /// `Expr::seq([first, second])`, as that constructor flattens its parts:
    /// a nested sequence contributes its items without its start state.
    fn flattened(first: (&Expr, Self), second: (&Expr, Self)) -> Self {
        let mut items = 0;
        let mut parts = Vec::new();
        for (expr, count) in [first, second] {
            match expr {
                Expr::Empty => {}
                Expr::Nothing => return Self::new(1, false),
                Expr::Seq(inner) => {
                    items += inner.len();
                    parts.push(Self::new(count.states.saturating_sub(1), count.finishes));
                }
                _ => {
                    items += 1;
                    parts.push(count);
                }
            }
        }
        let joined = Self::sequence(parts);
        match items {
            0 => Self::new(1, true),
            1 => Self::new(joined.states.saturating_sub(1), joined.finishes),
            _ => joined,
        }
    }
}

impl<'a> NfaBuilder<'a> {
    fn add(&mut self) -> Result<u32, GrammarError> {
        if self.states.len() >= MAX_NFA_STATES {
            return Err(GrammarError::too_large("the constraint is too large"));
        }
        self.budget.charge(1, 256)?;
        self.states.push(NfaState::default());
        Ok((self.states.len() - 1) as u32)
    }

    fn epsilon(&mut self, from: u32, to: u32) {
        self.states[from as usize].epsilon.push(to);
    }

    /// A rule compiled on its own: its start, its accepting state and the
    /// states its unshared construction reaches, at most the NFA states that
    /// construction could build at all.
    fn build_rule(&mut self, id: RuleId) -> Result<(u32, u32, usize), GrammarError> {
        let end = self.add()?;
        let (start, unshared) = self.expand(id, end)?;
        Ok((start, end, unshared.states.min(MAX_NFA_STATES)))
    }

    fn expand(&mut self, id: RuleId, next: u32) -> Result<(u32, Unshared), GrammarError> {
        let grammar = self.grammar;
        self.expanding.push(id);
        let built = self.build(&grammar.rules[id.0 as usize], next);
        self.expanding.pop();
        built
    }

    /// The start of `expr` continuing to `next`, and its unshared count.
    fn build(&mut self, expr: &'a Expr, next: u32) -> Result<(u32, Unshared), GrammarError> {
        self.budget.charge(1, 0)?;
        match expr {
            Expr::Empty => Ok((next, Unshared::new(1, true))),
            Expr::Nothing => Ok((self.add()?, Unshared::new(1, false))),
            Expr::Literal(bytes) => {
                let mut current = next;
                for &byte in bytes.iter().rev() {
                    let state = self.add()?;
                    self.states[state as usize]
                        .bytes
                        .push((byte, byte, current));
                    current = state;
                }
                Ok((current, Unshared::new(bytes.len() + 1, true)))
            }
            Expr::Class(class) => {
                let first = self.states.len();
                let start = self.add()?;
                let mut sequences = Vec::new();
                for &(lo, hi) in class.ranges() {
                    scalar_range_sequences(lo, hi, &mut sequences);
                }
                let finishes = !sequences.is_empty();
                for ranges in sequences {
                    let mut current = next;
                    for (index, &(lo, hi)) in ranges.iter().enumerate().rev() {
                        let state = if index == 0 { start } else { self.add()? };
                        self.states[state as usize].bytes.push((lo, hi, current));
                        current = state;
                    }
                }
                let states = self.states.len() - first + usize::from(finishes);
                Ok((start, Unshared::new(states, finishes)))
            }
            Expr::Seq(items) => {
                let mut current = next;
                let mut counts = Vec::with_capacity(items.len());
                for item in items.iter().rev() {
                    let (start, count) = self.build(item, current)?;
                    current = start;
                    counts.push(count);
                }
                Ok((current, Unshared::sequence(counts.into_iter().rev())))
            }
            Expr::Alt(items) => {
                let start = self.add()?;
                let mut unshared = Unshared::new(1, false);
                for item in items {
                    let (item_start, count) = self.build(item, next)?;
                    self.epsilon(start, item_start);
                    unshared.states = unshared.states.saturating_add(count.states);
                    unshared.finishes |= count.finishes;
                }
                unshared.states = unshared
                    .states
                    .saturating_add(usize::from(unshared.finishes));
                Ok((start, unshared))
            }
            Expr::Repeat { expr, min, max } => {
                if *min > MAX_REPEAT || max.is_some_and(|max| max > MAX_REPEAT) {
                    return Err(GrammarError::too_large("a repetition bound is too large"));
                }
                if max.is_some_and(|max| max < *min) {
                    return Err(GrammarError::invalid("a repetition has max below min"));
                }
                let (mut start, required, mut copy) = match max {
                    // The last required copy, or a loop state without one,
                    // repeats through the loop.
                    None => {
                        let repeat = self.add()?;
                        self.epsilon(repeat, next);
                        let (item, copy) = self.build(expr, repeat)?;
                        self.epsilon(repeat, item);
                        if *min == 0 {
                            (repeat, 0, copy)
                        } else {
                            (item, min - 1, copy)
                        }
                    }
                    Some(max) => {
                        let (start, copy) = self.optional(expr, max - min, next)?;
                        (start, *min, copy)
                    }
                };
                for _ in 0..required {
                    (start, copy) = self.build(expr, start)?;
                }
                // The unshared construction builds a start state, `min`
                // copies, then a loop state and one copy, or an end state and
                // copies up to `max`, each reached when the one before it
                // finishes.
                let unshared = if *min > 0 && !copy.finishes {
                    Unshared::new(copy.states.saturating_add(1), false)
                } else {
                    let optional = match max {
                        None => 1,
                        Some(max) if copy.finishes => max - min,
                        Some(max) => (max - min).min(1),
                    };
                    let copies = (*min + optional) as usize;
                    Unshared::new(copy.states.saturating_mul(copies).saturating_add(2), true)
                };
                Ok((start, unshared))
            }
            Expr::Rule(id) => {
                if id.0 as usize >= self.grammar.rules.len() {
                    return Err(GrammarError::invalid("a grammar rule is undefined"));
                }
                // Only the root and recursive rules are compiled on their own,
                // and nothing refers to a root that is not recursive.
                if let Some(&compiled) = self.compiled_index.get(id) {
                    let start = self.add()?;
                    self.states[start as usize].calls.push((compiled, next));
                    return Ok((start, Unshared::new(2, true)));
                }
                if self.expanding.contains(id) {
                    return Err(GrammarError::invalid("a grammar rule refers to itself"));
                }
                if let Some(&built) = self.expanded.get(&(*id, next)) {
                    return Ok(built);
                }
                let built = self.expand(*id, next)?;
                self.expanded.insert((*id, next), built);
                Ok(built)
            }
            Expr::TextExcluding(markers) => self.marked_text(markers, None, next),
            Expr::TextUntil { end, avoid } => self.marked_text(avoid, Some(end), next),
            Expr::OrderedSubset {
                items,
                tail,
                separator,
            } => self.ordered_subset(items, tail.as_deref(), separator, next),
        }
    }

    /// A state that admits up to `count` copies of `expr` and then continues
    /// to `next`, and the unshared count of one copy (none when `count` is
    /// 0). The states for fewer copies are a suffix of the states for more,
    /// so every run of a rule, literal or class bounded above with one
    /// continuation shares one chain.
    fn optional(
        &mut self,
        expr: &'a Expr,
        count: u32,
        next: u32,
    ) -> Result<(u32, Unshared), GrammarError> {
        if count == 0 {
            return Ok((next, Unshared::default()));
        }
        let shared = match expr {
            Expr::Rule(_) => true,
            Expr::Literal(bytes) => {
                self.budget.charge(bytes.len(), 0)?;
                true
            }
            Expr::Class(class) => {
                self.budget.charge(class.ranges().len(), 0)?;
                true
            }
            _ => false,
        };
        let (mut chain, mut copy) = if shared {
            self.optional.remove(&(expr, next)).unwrap_or_default()
        } else {
            (Vec::new(), Unshared::default())
        };
        if chain.is_empty() {
            chain.push(next);
        }
        while chain.len() <= count as usize {
            let state = self.add()?;
            self.epsilon(state, next);
            let item;
            (item, copy) = self.build(expr, chain[chain.len() - 1])?;
            self.epsilon(state, item);
            chain.push(state);
        }
        let start = chain[count as usize];
        if shared {
            self.optional.insert((expr, next), (chain, copy));
        }
        Ok((start, copy))
    }

    /// Two chains of states: `before[i]` has written no entry yet, and
    /// `after[i]` has, so a separator comes first. Both enter one fragment of
    /// each item, which continues to `after[i + 1]`, and of the tail, which
    /// continues to the last `after`. Nothing reaches `after[0]`, so it is not
    /// built, and with an empty separator the two chains are one.
    fn ordered_subset(
        &mut self,
        items: &'a [(Expr, bool)],
        tail: Option<&'a Expr>,
        separator: &'a Expr,
        next: u32,
    ) -> Result<(u32, Unshared), GrammarError> {
        let joined = matches!(separator, Expr::Empty);
        // An entry that matches nothing is not written after a separator
        // either, as the unshared construction's `Expr::seq` drops it.
        let separable = |entry: &Expr| !joined && !matches!(entry, Expr::Nothing);
        let mut before = self.add()?;
        let mut after = if joined { before } else { self.add()? };
        self.epsilon(before, next);
        if !joined {
            self.epsilon(after, next);
        }
        // The unshared counts of each entry alone and after a separator.
        let mut tail_counts = None;
        let mut item_counts = vec![(Unshared::default(), Unshared::default()); items.len()];
        if let Some(tail) = tail {
            let (start, count) = self.build(tail, after)?;
            self.epsilon(before, start);
            let mut separator_count = Unshared::new(1, true);
            if separable(tail) {
                let separated;
                (separated, separator_count) = self.build(separator, start)?;
                self.epsilon(after, separated);
            }
            let separated = Unshared::flattened((separator, separator_count), (tail, count));
            tail_counts = Some((count, separated));
        }
        for (index, (item, required)) in items.iter().enumerate().rev() {
            let (start, count) = self.build(item, after)?;
            let item_before = self.add()?;
            self.epsilon(item_before, start);
            if !required {
                self.epsilon(item_before, before);
            }
            let mut separator_count = Unshared::new(1, true);
            let item_after = if joined || index == 0 {
                item_before
            } else {
                let item_after = self.add()?;
                if separable(item) {
                    let separated;
                    (separated, separator_count) = self.build(separator, start)?;
                    self.epsilon(item_after, separated);
                }
                if !required {
                    self.epsilon(item_after, after);
                }
                item_after
            };
            let separated = Unshared::flattened((separator, separator_count), (item, count));
            item_counts[index] = (count, separated);
            before = item_before;
            after = item_after;
        }
        // What the unshared construction reaches: its chains of `before`
        // and `after` states, its end state, and each entry from the chain
        // states that enter it.
        let mut states = 0_usize;
        let mut reached_before = true;
        let mut reached_after = false;
        for (&(count, separated), (_, required)) in item_counts.iter().zip(items) {
            states =
                states.saturating_add(usize::from(reached_before) + usize::from(reached_after));
            let mut next_after = false;
            if reached_before {
                states = states.saturating_add(count.states);
                next_after |= count.finishes;
            }
            if reached_after {
                states = states.saturating_add(separated.states);
                next_after |= separated.finishes;
            }
            reached_before &= !required;
            reached_after = next_after || (reached_after && !required);
        }
        if let Some((count, separated)) = tail_counts {
            if reached_before {
                states = states.saturating_add(count.states);
                reached_after |= count.finishes;
            }
            if reached_after {
                states = states.saturating_add(separated.states);
            }
        }
        let finishes = reached_before || reached_after;
        states = states.saturating_add(
            usize::from(reached_before) + usize::from(reached_after) + usize::from(finishes),
        );
        Ok((before, Unshared::new(states, finishes)))
    }

    /// UTF-8 text that never contains one of the `avoid` markers, and, with
    /// an `end` marker, runs exactly to the first occurrence of that marker:
    /// the product of a UTF-8 validator and one substring matcher per marker.
    /// Occurrences that straddle the text and the end marker count, so a
    /// reader that searches for the first marker splits where this did.
    fn marked_text(
        &mut self,
        avoid: &[Vec<u8>],
        end_marker: Option<&[u8]>,
        next: u32,
    ) -> Result<(u32, Unshared), GrammarError> {
        let markers: Vec<&[u8]> = avoid.iter().map(Vec::as_slice).chain(end_marker).collect();
        if markers.is_empty() || markers.iter().any(|marker| marker.is_empty()) || markers.len() > 4
        {
            return Err(GrammarError::invalid(
                "marked text needs one to four nonempty markers",
            ));
        }
        let failures = markers
            .iter()
            .map(|marker| failure_function(marker))
            .collect::<Vec<_>>();
        let kmp = |index: usize, mut matched: usize, byte: u8| -> usize {
            let marker = markers[index];
            loop {
                if marker[matched] == byte {
                    return matched + 1;
                }
                if matched == 0 {
                    return 0;
                }
                matched = failures[index][matched - 1];
            }
        };
        let end = next;
        let first_state = self.states.len();
        let mut ended_anywhere = false;
        let mut ids: HashMap<(Utf8State, Vec<usize>), u32> = HashMap::new();
        let first = (Utf8State::Boundary, vec![0_usize; markers.len()]);
        let start = self.add()?;
        ids.insert(first.clone(), start);
        let mut pending = vec![first];
        while let Some(key) = pending.pop() {
            let from = ids[&key];
            let (utf8, matched) = key;
            if utf8 == Utf8State::Boundary && end_marker.is_none() {
                self.epsilon(from, end);
                ended_anywhere = true;
            }
            let mut run: Option<(u8, u8, u32)> = None;
            for byte in 0..=255_u8 {
                let mut target = None;
                if let Some(next) = utf8.next(byte) {
                    let matched = matched
                        .iter()
                        .enumerate()
                        .map(|(index, &state)| kmp(index, state, byte))
                        .collect::<Vec<_>>();
                    let avoided = matched[..avoid.len()]
                        .iter()
                        .zip(avoid)
                        .any(|(&state, marker)| state == marker.len());
                    let ended =
                        end_marker.is_some_and(|marker| matched[avoid.len()] == marker.len());
                    if ended && !avoided && next == Utf8State::Boundary {
                        target = Some(end);
                        ended_anywhere = true;
                    } else if !ended && !avoided {
                        let key = (next, matched);
                        target = Some(match ids.get(&key) {
                            Some(&id) => id,
                            None => {
                                let id = self.add()?;
                                ids.insert(key.clone(), id);
                                pending.push(key);
                                id
                            }
                        });
                    }
                }
                run = match (run, target) {
                    (Some((lo, hi, id)), Some(next)) if next == id && hi + 1 == byte => {
                        Some((lo, byte, id))
                    }
                    (previous, next) => {
                        if let Some((lo, hi, id)) = previous {
                            self.states[from as usize].bytes.push((lo, hi, id));
                        }
                        next.map(|id| (byte, byte, id))
                    }
                };
            }
            if let Some((lo, hi, id)) = run {
                self.states[from as usize].bytes.push((lo, hi, id));
            }
        }
        // The unshared construction builds its own end state too.
        let states = self.states.len() - first_state + usize::from(ended_anywhere);
        Ok((start, Unshared::new(states, ended_anywhere)))
    }
}

fn failure_function(pattern: &[u8]) -> Vec<usize> {
    let mut failure = vec![0; pattern.len()];
    let mut length = 0;
    for index in 1..pattern.len() {
        while length > 0 && pattern[index] != pattern[length] {
            length = failure[length - 1];
        }
        if pattern[index] == pattern[length] {
            length += 1;
        }
        failure[index] = length;
    }
    failure
}

/// Where a UTF-8 validator is inside one encoded scalar.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Utf8State {
    Boundary,
    /// Needs this many more bytes in 80..=BF.
    Continuation(u8),
    /// Needs one byte in the range, then this many continuation bytes.
    Restricted(u8, u8, u8),
}

impl Utf8State {
    fn next(self, byte: u8) -> Option<Self> {
        let continuation = |remaining: u8| {
            if remaining == 0 {
                Self::Boundary
            } else {
                Self::Continuation(remaining)
            }
        };
        match self {
            Self::Boundary => match byte {
                0x00..=0x7F => Some(Self::Boundary),
                0xC2..=0xDF => Some(Self::Continuation(1)),
                0xE0 => Some(Self::Restricted(0xA0, 0xBF, 1)),
                0xE1..=0xEC | 0xEE..=0xEF => Some(Self::Continuation(2)),
                0xED => Some(Self::Restricted(0x80, 0x9F, 1)),
                0xF0 => Some(Self::Restricted(0x90, 0xBF, 2)),
                0xF1..=0xF3 => Some(Self::Continuation(3)),
                0xF4 => Some(Self::Restricted(0x80, 0x8F, 2)),
                _ => None,
            },
            Self::Continuation(remaining) => (0x80..=0xBF)
                .contains(&byte)
                .then(|| continuation(remaining - 1)),
            Self::Restricted(lo, hi, remaining) => {
                (lo..=hi).contains(&byte).then(|| continuation(remaining))
            }
        }
    }
}

fn states_log_bound(count: usize) -> usize {
    count.max(1).ilog2() as usize + 1
}

/// Determinizes one rule's automaton, given its start, its accepting state
/// and the states the unshared construction builds for it (see
/// `NfaBuilder`). Growth is charged against that count, which sharing
/// fragments does not reduce.
fn determinize(
    nfa: &[NfaState],
    rule: u32,
    (start, accept, unshared): (u32, u32, usize),
    states: &mut Vec<DfaState>,
    entries: &mut usize,
    budget: &Budget,
) -> Result<CompiledRule, GrammarError> {
    // Reserve traversal, marks and closure buffers before allocating them.
    budget.charge(0, nfa.len().saturating_mul(32))?;
    let reachable = reachable_states(nfa, start);
    let mut boundaries = [false; 257];
    boundaries[0] = true;
    boundaries[256] = true;
    for &state in &reachable {
        for &(lo, hi, _) in &nfa[state as usize].bytes {
            boundaries[lo as usize] = true;
            boundaries[hi as usize + 1] = true;
        }
    }
    let mut classes = Box::new([0_u8; 256]);
    let mut representatives = Vec::new();
    let mut class = 0_usize;
    for byte in 0..256 {
        if boundaries[byte] && byte != 0 {
            class += 1;
        }
        if boundaries[byte] {
            representatives.push(byte as u8);
        }
        classes[byte] = u8::try_from(class).expect("at most 256 byte classes");
    }
    let class_count = representatives.len();

    let base = states.len() as u32;
    let mut marks = vec![0_u32; nfa.len()];
    let mut generation = 0_u32;
    let mut closure = |seeds: &mut Vec<u32>| -> Result<Vec<u32>, GrammarError> {
        generation += 1;
        let mut stack = std::mem::take(seeds);
        for &seed in &stack {
            marks[seed as usize] = generation;
        }
        let mut result = Vec::new();
        while let Some(state) = stack.pop() {
            budget.charge(1 + nfa[state as usize].epsilon.len(), 0)?;
            result.push(state);
            for &next in &nfa[state as usize].epsilon {
                if marks[next as usize] != generation {
                    marks[next as usize] = generation;
                    stack.push(next);
                }
            }
        }
        budget.charge(
            result.len().saturating_mul(states_log_bound(result.len())),
            0,
        )?;
        result.sort_unstable();
        Ok(result)
    };
    let mut ids: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut sets: Vec<Vec<u32>> = Vec::new();
    budget.dfa_growth(1, unshared)?;
    *entries += class_count;
    if *entries > MAX_TABLE_ENTRIES {
        return Err(GrammarError::too_large("the constraint is too large"));
    }
    let initial = closure(&mut vec![start])?;
    budget.charge(1, initial.len() * 16 + 256 + class_count * 64)?;
    ids.insert(initial.clone(), base);
    sets.push(initial);
    let mut index = 0;
    while index < sets.len() {
        let mut scratch = sets[index].len() * 4 + class_count * 24;
        budget.scratch(scratch)?;
        let set = sets[index].clone();
        let mut by_class: Vec<Vec<u32>> = vec![Vec::new(); class_count];
        let mut by_rule: Vec<(u32, Vec<u32>)> = Vec::new();
        for &state in &set {
            let record = &nfa[state as usize];
            for &(lo, hi, target) in &record.bytes {
                for class in classes[lo as usize]..=classes[hi as usize] {
                    budget.charge(1, 0)?;
                    scratch += 8;
                    budget.scratch(scratch)?;
                    by_class[class as usize].push(target);
                }
            }
            for &(called, target) in &record.calls {
                budget.charge(1 + by_rule.len(), 0)?;
                scratch += 64;
                budget.scratch(scratch)?;
                match by_rule.iter_mut().find(|(rule, _)| *rule == called) {
                    Some((_, targets)) => targets.push(target),
                    None => by_rule.push((called, vec![target])),
                }
            }
        }
        let mut intern = |mut seeds: Vec<u32>,
                          ids: &mut HashMap<Vec<u32>, u32>,
                          sets: &mut Vec<Vec<u32>>|
         -> Result<u32, GrammarError> {
            if seeds.is_empty() {
                return Ok(DEAD);
            }
            budget.charge(seeds.len().saturating_mul(states_log_bound(seeds.len())), 0)?;
            seeds.sort_unstable();
            seeds.dedup();
            let target = closure(&mut seeds)?;
            if let Some(&id) = ids.get(&target) {
                return Ok(id);
            }
            let id = base + sets.len() as u32;
            *entries += class_count;
            if *entries > MAX_TABLE_ENTRIES || id == DEAD {
                return Err(GrammarError::too_large("the constraint is too large"));
            }
            budget.dfa_growth(sets.len() + 1, unshared)?;
            let bytes = target.len() * 16 + 256 + class_count * 64;
            // Transition buffers remain live while subsets are interned.
            budget.scratch(scratch.saturating_add(bytes))?;
            budget.charge(1, bytes)?;
            ids.insert(target.clone(), id);
            sets.push(target);
            Ok(id)
        };
        let mut next = Vec::with_capacity(class_count);
        for targets in by_class {
            next.push(intern(targets, &mut ids, &mut sets)?);
        }
        let call_bytes = by_rule.len().saturating_mul(32);
        budget.scratch(scratch.saturating_add(call_bytes))?;
        budget.charge(0, call_bytes)?;
        let mut calls = Vec::with_capacity(by_rule.len());
        for (called, targets) in by_rule {
            calls.push((called, intern(targets, &mut ids, &mut sets)?));
        }
        let accepting = set.binary_search(&accept).is_ok();
        states.push(DfaState {
            rule,
            accepting,
            reads: next.iter().any(|&next| next != DEAD),
            next: next.into_boxed_slice(),
            calls: calls.into_boxed_slice(),
        });
        index += 1;
    }
    Ok(CompiledRule {
        start: base,
        classes,
    })
}

fn reachable_states(nfa: &[NfaState], start: u32) -> Vec<u32> {
    let mut seen = vec![false; nfa.len()];
    let mut pending = vec![start];
    let mut reached = Vec::new();
    seen[start as usize] = true;
    while let Some(state) = pending.pop() {
        reached.push(state);
        let record = &nfa[state as usize];
        let targets = record
            .epsilon
            .iter()
            .copied()
            .chain(record.bytes.iter().map(|&(_, _, target)| target))
            .chain(record.calls.iter().map(|&(_, target)| target));
        for target in targets {
            if !std::mem::replace(&mut seen[target as usize], true) {
                pending.push(target);
            }
        }
    }
    reached
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{CharClass, GrammarBuilder};

    /// Whether a grammar without calls matches the whole input.
    fn matches(automaton: &Automaton, input: &[u8]) -> bool {
        let mut state = automaton.root_start();
        for &byte in input {
            state = automaton.next(state, byte);
            if state == DEAD {
                return false;
            }
        }
        automaton.state(state).accepting
    }

    #[test]
    fn expressions_compile_to_exact_languages() {
        let digits = Expr::Class(CharClass::from_ranges([(u32::from(b'0'), u32::from(b'9'))]));
        let grammar = Grammar::from_expr(Expr::seq([
            Expr::literal("a"),
            digits.clone().repeat(2, Some(3)),
            Expr::alt([Expr::literal("x"), Expr::literal("yz")]).optional(),
        ]));
        let automaton = Automaton::compile(&grammar).unwrap();
        assert!(!automaton.has_calls());
        for good in ["a12", "a123", "a12x", "a999yz"] {
            assert!(matches(&automaton, good.as_bytes()), "{good}");
        }
        for bad in ["a1", "a1234", "a12y", "b12", "a12xx"] {
            assert!(!matches(&automaton, bad.as_bytes()), "{bad}");
        }
    }

    #[test]
    fn unproductive_branches_are_pruned() {
        let grammar = Grammar::from_expr(Expr::alt([
            Expr::literal("ok"),
            Expr::Seq(vec![Expr::literal("no"), Expr::Nothing]),
        ]));
        let automaton = Automaton::compile(&grammar).unwrap();
        let start = automaton.root_start();
        assert_eq!(automaton.next(start, b'n'), DEAD);
        assert_ne!(automaton.next(start, b'o'), DEAD);
    }

    #[test]
    fn excluded_markers_never_appear() {
        let grammar = Grammar::from_expr(Expr::seq([
            Expr::TextExcluding(vec![b"</p>".to_vec()]),
            Expr::literal("</p>"),
        ]));
        let automaton = Automaton::compile(&grammar).unwrap();
        assert!(matches(&automaton, "ab</</p>".as_bytes()));
        assert!(matches(&automaton, "é<</p>".as_bytes()));
        assert!(!matches(&automaton, b"a</p>b</p>"));
        assert!(!matches(&automaton, b"\xff</p>"));
        assert!(!matches(&automaton, b"\xe2\x82</p>"));
        let both = Grammar::from_expr(Expr::TextExcluding(vec![b"<a>".to_vec(), b"<b>".to_vec()]));
        let automaton = Automaton::compile(&both).unwrap();
        assert!(matches(&automaton, b"x<a<b<ab>"));
        assert!(!matches(&automaton, b"x<b>"));
        assert!(!matches(&automaton, b"<<a>"));
        // A marker with a border: the text ends at the first occurrence,
        // including one that straddles the text and the marker.
        let until = Grammar::from_expr(Expr::TextUntil {
            end: b"\n</p>\n".to_vec(),
            avoid: vec![b"<x>".to_vec()],
        });
        let automaton = Automaton::compile(&until).unwrap();
        assert!(matches(&automaton, b"ab\n</p>\n"));
        assert!(matches(&automaton, b"</p>\n\n</p>\n"));
        assert!(matches(&automaton, b"\n</p>\n"));
        assert!(!matches(&automaton, b"</p>\n\n</p>\n</p>\n"));
        assert!(!matches(&automaton, b"ab\n</p>\n\n</p>\n"));
        assert!(!matches(&automaton, b"a<x>\n</p>\n"));
        assert!(!matches(&automaton, b"ab\n</p>"));
    }

    #[test]
    fn closure_growth_includes_nullable_returns_and_bounds_cycles() {
        let mut builder = GrammarBuilder::new();
        let nested = builder.declare();
        let wrapper = builder.declare();
        builder.define(
            nested,
            Expr::alt([
                Expr::literal(""),
                Expr::seq([Expr::literal("("), Expr::Rule(wrapper), Expr::literal(")")]),
            ]),
        );
        builder.define(wrapper, Expr::seq([Expr::Rule(nested), Expr::Rule(nested)]));
        let grammar = builder.finish(wrapper).unwrap();
        let automaton = Automaton::compile(&grammar).unwrap();
        assert_eq!(automaton.max_call_growth(), 2);

        let mut builder = GrammarBuilder::new();
        let recursive = builder.declare();
        builder.define(
            recursive,
            Expr::alt([Expr::literal("x"), Expr::Rule(recursive)]),
        );
        let automaton = Automaton::compile(&builder.finish(recursive).unwrap()).unwrap();
        assert_eq!(automaton.max_call_growth(), crate::stack::MAX_DEPTH);
    }

    #[test]
    fn recursive_rules_become_calls() {
        let mut builder = GrammarBuilder::new();
        let nested = builder.declare();
        builder.define(
            nested,
            Expr::alt([
                Expr::literal("x"),
                Expr::seq([Expr::literal("("), Expr::Rule(nested), Expr::literal(")")]),
            ]),
        );
        let automaton = Automaton::compile(&builder.finish(nested).unwrap()).unwrap();
        assert!(automaton.has_calls());
    }

    type Words = std::collections::BTreeSet<Vec<u8>>;

    const ALPHABET: &[u8] = b"abc";

    /// Every word over `ALPHABET` of at most `limit` bytes.
    fn every_word(limit: usize) -> Vec<Vec<u8>> {
        let mut words = vec![Vec::new()];
        let mut index = 0;
        while index < words.len() {
            if words[index].len() < limit {
                for &byte in ALPHABET {
                    words.push([words[index].as_slice(), &[byte]].concat());
                }
            }
            index += 1;
        }
        words
    }

    fn concat(left: &Words, right: &Words, limit: usize) -> Words {
        let mut words = Words::new();
        for left in left {
            for right in right
                .iter()
                .filter(|right| left.len() + right.len() <= limit)
            {
                words.insert([left.as_slice(), right].concat());
            }
        }
        words
    }

    /// `start` followed by any number of `step`.
    fn star(start: Words, step: &Words, limit: usize) -> Words {
        let mut words = start;
        loop {
            let count = words.len();
            let more = concat(&words, step, limit);
            words.extend(more);
            if words.len() == count {
                return words;
            }
        }
    }

    fn position(word: &[u8], marker: &[u8]) -> Option<usize> {
        word.windows(marker.len())
            .position(|window| window == marker)
    }

    /// The words of at most `limit` bytes that `expr` matches, as its
    /// documentation defines them.
    fn words(expr: &Expr, rules: &[Expr], limit: usize) -> Words {
        let empty = || Words::from([Vec::new()]);
        match expr {
            Expr::Empty => empty(),
            Expr::Nothing => Words::new(),
            Expr::Literal(bytes) => Words::from_iter((bytes.len() <= limit).then(|| bytes.clone())),
            Expr::Class(class) => (0..=127_u8)
                .filter(|&byte| limit > 0 && class.contains(u32::from(byte)))
                .map(|byte| vec![byte])
                .collect(),
            Expr::Seq(items) => items.iter().fold(empty(), |words_so_far, item| {
                concat(&words_so_far, &words(item, rules, limit), limit)
            }),
            Expr::Alt(items) => items
                .iter()
                .flat_map(|item| words(item, rules, limit))
                .collect(),
            Expr::Repeat { expr, min, max } => {
                let one = words(expr, rules, limit);
                let mut exact = empty();
                for _ in 0..*min {
                    exact = concat(&exact, &one, limit);
                }
                match max {
                    None => star(exact, &one, limit),
                    Some(max) => {
                        let mut all = exact.clone();
                        for _ in *min..*max {
                            exact = concat(&exact, &one, limit);
                            all.extend(exact.iter().cloned());
                        }
                        all
                    }
                }
            }
            Expr::Rule(id) => words(&rules[id.0 as usize], rules, limit),
            Expr::TextExcluding(markers) => every_word(limit)
                .into_iter()
                .filter(|word| {
                    markers
                        .iter()
                        .all(|marker| position(word, marker).is_none())
                })
                .collect(),
            Expr::TextUntil { end, avoid } => every_word(limit)
                .into_iter()
                .filter(|word| {
                    word.len() >= end.len()
                        && position(word, end) == Some(word.len() - end.len())
                        && avoid.iter().all(|marker| position(word, marker).is_none())
                })
                .collect(),
            Expr::OrderedSubset {
                items,
                tail,
                separator,
            } => {
                let separator = words(separator, rules, limit);
                let mut after = match tail {
                    None => empty(),
                    Some(tail) => star(
                        empty(),
                        &concat(&separator, &words(tail, rules, limit), limit),
                        limit,
                    ),
                };
                let mut before = match tail {
                    None => empty(),
                    Some(tail) => {
                        let mut words = concat(&words(tail, rules, limit), &after, limit);
                        words.insert(Vec::new());
                        words
                    }
                };
                for (item, required) in items.iter().rev() {
                    let item = words(item, rules, limit);
                    let mut item_before = concat(&item, &after, limit);
                    let mut item_after = concat(&concat(&separator, &item, limit), &after, limit);
                    if !required {
                        item_before.extend(before);
                        item_after.extend(after);
                    }
                    before = item_before;
                    after = item_after;
                }
                before
            }
        }
    }

    struct Random(u64);

    impl Random {
        fn below(&mut self, bound: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % bound
        }

        fn word(&mut self, shortest: u64, longest: u64) -> Vec<u8> {
            let length = shortest + self.below(longest - shortest + 1);
            (0..length)
                .map(|_| ALPHABET[self.below(ALPHABET.len() as u64) as usize])
                .collect()
        }

        /// An expression that may refer to the rules in `rules`.
        fn expr(&mut self, depth: u32, rules: std::ops::Range<u32>) -> Expr {
            if depth == 0 || self.below(3) == 0 {
                return match self.below(8) {
                    0 => Expr::Empty,
                    1 => Expr::Nothing,
                    2 | 3 => Expr::Literal(self.word(0, 2)),
                    4 => Expr::Class(CharClass::from_ranges(
                        ALPHABET
                            .iter()
                            .filter(|_| self.below(2) == 0)
                            .map(|&byte| (u32::from(byte), u32::from(byte))),
                    )),
                    5 if !rules.is_empty() => Expr::Rule(RuleId(
                        rules.start + self.below(u64::from(rules.end - rules.start)) as u32,
                    )),
                    6 => Expr::TextExcluding(vec![self.word(1, 2)]),
                    7 => Expr::TextUntil {
                        end: self.word(1, 2),
                        avoid: (0..self.below(2)).map(|_| self.word(1, 2)).collect(),
                    },
                    _ => Expr::Literal(self.word(1, 1)),
                };
            }
            let some = |count: u64, random: &mut Self| {
                (0..count)
                    .map(|_| random.expr(depth - 1, rules.clone()))
                    .collect::<Vec<_>>()
            };
            match self.below(6) {
                0 => Expr::Seq(some(2 + self.below(2), self)),
                1 => Expr::Alt(some(2 + self.below(2), self)),
                2 => {
                    let min = self.below(3) as u32;
                    let max = (self.below(3) != 0).then(|| min + self.below(3) as u32);
                    Expr::Repeat {
                        expr: Box::new(self.expr(depth - 1, rules)),
                        min,
                        max,
                    }
                }
                3 => {
                    let items = some(self.below(4), self)
                        .into_iter()
                        .map(|item| (item, self.below(3) == 0))
                        .collect();
                    let tail =
                        (self.below(2) == 0).then(|| Box::new(self.expr(depth - 1, rules.clone())));
                    let separator = match self.below(3) {
                        0 => Expr::Empty,
                        1 => Expr::Literal(self.word(1, 1)),
                        _ => self.expr(depth - 1, rules),
                    };
                    Expr::OrderedSubset {
                        items,
                        tail,
                        separator: Box::new(separator),
                    }
                }
                // Bounded runs of one leaf that continue to different
                // states, and to the same one.
                4 => {
                    let leaf = self.expr(0, rules.clone());
                    let run = |random: &mut Self| {
                        let min = random.below(2) as u32;
                        Expr::Repeat {
                            expr: Box::new(leaf.clone()),
                            min,
                            max: Some(min + random.below(3) as u32),
                        }
                    };
                    let first = run(self);
                    let second = run(self);
                    let between = self.expr(depth - 1, rules);
                    if self.below(2) == 0 {
                        Expr::Seq(vec![first, between, second])
                    } else {
                        Expr::Alt(vec![first, Expr::Seq(vec![between, second])])
                    }
                }
                // One expression, then more of it after a separator: both
                // copies continue to the same state.
                _ => {
                    let item = self.expr(depth - 1, rules.clone());
                    let separator = self.expr(depth - 1, rules);
                    Expr::Seq(vec![
                        item.clone(),
                        Expr::Repeat {
                            expr: Box::new(Expr::Seq(vec![separator, item])),
                            min: 0,
                            max: None,
                        },
                    ])
                }
            }
        }
    }

    /// Random grammars of shared rules, repetitions, ordered subsets and
    /// marked text compile within the budgets and admit exactly the words
    /// their definitions do, so building each use of an expression that
    /// continues to the same state once changes no language.
    #[test]
    fn built_automata_match_the_definitions() {
        const LIMIT: usize = 5;
        let candidates = every_word(LIMIT);
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        for case in 0..400 {
            let count = 1 + random.below(3) as u32;
            let rules: Vec<Expr> = (0..count)
                .map(|rule| random.expr(3, rule + 1..count))
                .collect();
            let mut builder = GrammarBuilder::new();
            for rule in &rules {
                builder.rule(rule.clone());
            }
            let grammar = builder.finish(RuleId(0)).unwrap();
            let expected = words(&rules[0], &rules, LIMIT);
            let automaton = match Automaton::compile(&grammar) {
                Ok(automaton) => automaton,
                Err(error) => {
                    assert_eq!(
                        error,
                        GrammarError::invalid("the constraint matches no output"),
                        "{case}: {rules:?}"
                    );
                    assert!(expected.is_empty(), "{case}: {rules:?}");
                    continue;
                }
            };
            assert!(!automaton.has_calls());
            for word in &candidates {
                assert_eq!(
                    matches(&automaton, word),
                    expected.contains(word),
                    "{case}: {:?} in {rules:?}",
                    String::from_utf8_lossy(word)
                );
            }
        }
    }
}
