//! Compiling a grammar to deterministic byte automata with rule calls.
//!
//! Rules that take part in a reference cycle stay separate automata that the
//! matcher calls with a stack. Every other rule reference is expanded in
//! place, so a grammar without recursion becomes one automaton. Each
//! automaton is a subset construction over its Thompson automaton, and states
//! that cannot reach an accepting state are removed, so any live state can
//! still finish.

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
        };
        let mut nfas = Vec::with_capacity(order.len());
        for &id in &order {
            let (start, end) = builder.build_rule(id)?;
            nfas.push((start, end));
        }
        let nfa = builder.states;
        let mut states = Vec::new();
        let mut rules = Vec::with_capacity(order.len());
        let mut entries = 0_usize;
        for (rule, &(start, end)) in nfas.iter().enumerate() {
            let compiled = determinize(
                &nfa,
                rule as u32,
                start,
                end,
                &mut states,
                &mut entries,
                budget,
            )?;
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

struct NfaBuilder<'a> {
    grammar: &'a Grammar,
    budget: &'a Budget,
    compiled_index: &'a HashMap<RuleId, u32>,
    states: Vec<NfaState>,
    expanding: Vec<RuleId>,
}

impl NfaBuilder<'_> {
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

    fn build_rule(&mut self, id: RuleId) -> Result<(u32, u32), GrammarError> {
        self.expanding.push(id);
        let fragment = self.build(&self.grammar.rules[id.0 as usize]);
        self.expanding.pop();
        fragment
    }

    fn build(&mut self, expr: &Expr) -> Result<(u32, u32), GrammarError> {
        self.budget.charge(1, 0)?;
        match expr {
            Expr::Empty => {
                let state = self.add()?;
                Ok((state, state))
            }
            Expr::Nothing => Ok((self.add()?, self.add()?)),
            Expr::Literal(bytes) => {
                let start = self.add()?;
                let mut current = start;
                for &byte in bytes {
                    let next = self.add()?;
                    self.states[current as usize].bytes.push((byte, byte, next));
                    current = next;
                }
                Ok((start, current))
            }
            Expr::Class(class) => {
                let start = self.add()?;
                let end = self.add()?;
                let mut sequences = Vec::new();
                for &(lo, hi) in class.ranges() {
                    scalar_range_sequences(lo, hi, &mut sequences);
                }
                for ranges in sequences {
                    let mut current = start;
                    for (index, &(lo, hi)) in ranges.iter().enumerate() {
                        let next = if index + 1 == ranges.len() {
                            end
                        } else {
                            self.add()?
                        };
                        self.states[current as usize].bytes.push((lo, hi, next));
                        current = next;
                    }
                }
                Ok((start, end))
            }
            Expr::Seq(items) => {
                let start = self.add()?;
                let mut current = start;
                for item in items {
                    let (item_start, item_end) = self.build(item)?;
                    self.epsilon(current, item_start);
                    current = item_end;
                }
                Ok((start, current))
            }
            Expr::Alt(items) => {
                let start = self.add()?;
                let end = self.add()?;
                for item in items {
                    let (item_start, item_end) = self.build(item)?;
                    self.epsilon(start, item_start);
                    self.epsilon(item_end, end);
                }
                Ok((start, end))
            }
            Expr::Repeat { expr, min, max } => {
                if *min > MAX_REPEAT || max.is_some_and(|max| max > MAX_REPEAT) {
                    return Err(GrammarError::too_large("a repetition bound is too large"));
                }
                if max.is_some_and(|max| max < *min) {
                    return Err(GrammarError::invalid("a repetition has max below min"));
                }
                let start = self.add()?;
                let mut current = start;
                for _ in 0..*min {
                    let (item_start, item_end) = self.build(expr)?;
                    self.epsilon(current, item_start);
                    current = item_end;
                }
                match max {
                    None => {
                        let loop_state = self.add()?;
                        self.epsilon(current, loop_state);
                        let (item_start, item_end) = self.build(expr)?;
                        self.epsilon(loop_state, item_start);
                        self.epsilon(item_end, loop_state);
                        Ok((start, loop_state))
                    }
                    Some(max) => {
                        let end = self.add()?;
                        self.epsilon(current, end);
                        for _ in *min..*max {
                            let (item_start, item_end) = self.build(expr)?;
                            self.epsilon(current, item_start);
                            self.epsilon(item_end, end);
                            current = item_end;
                        }
                        Ok((start, end))
                    }
                }
            }
            Expr::Rule(id) => {
                if id.0 as usize >= self.grammar.rules.len() {
                    return Err(GrammarError::invalid("a grammar rule is undefined"));
                }
                // Only the root and recursive rules are compiled on their own,
                // and nothing refers to a root that is not recursive.
                if let Some(&compiled) = self.compiled_index.get(id) {
                    let start = self.add()?;
                    let end = self.add()?;
                    self.states[start as usize].calls.push((compiled, end));
                    return Ok((start, end));
                }
                if self.expanding.contains(id) {
                    return Err(GrammarError::invalid("a grammar rule refers to itself"));
                }
                self.build_rule(*id)
            }
            Expr::TextExcluding(markers) => self.marked_text(markers, None),
            Expr::TextUntil { end, avoid } => self.marked_text(avoid, Some(end)),
            Expr::OrderedSubset {
                items,
                tail,
                separator,
            } => self.ordered_subset(items, tail.as_deref(), separator),
        }
    }

    /// Two chains of states: `before[i]` has emitted nothing yet, `after[i]`
    /// has emitted at least one entry and so needs a separator next. Each
    /// item is built once per chain.
    fn ordered_subset(
        &mut self,
        items: &[(Expr, bool)],
        tail: Option<&Expr>,
        separator: &Expr,
    ) -> Result<(u32, u32), GrammarError> {
        let before = (0..=items.len())
            .map(|_| self.add())
            .collect::<Result<Vec<_>, _>>()?;
        let after = (0..=items.len())
            .map(|_| self.add())
            .collect::<Result<Vec<_>, _>>()?;
        let end = self.add()?;
        for (index, (item, required)) in items.iter().enumerate() {
            let (start, finish) = self.build(item)?;
            self.epsilon(before[index], start);
            self.epsilon(finish, after[index + 1]);
            let (start, finish) = self.build(&Expr::seq([separator.clone(), item.clone()]))?;
            self.epsilon(after[index], start);
            self.epsilon(finish, after[index + 1]);
            if !required {
                self.epsilon(before[index], before[index + 1]);
                self.epsilon(after[index], after[index + 1]);
            }
        }
        let last = items.len();
        self.epsilon(before[last], end);
        self.epsilon(after[last], end);
        if let Some(tail) = tail {
            let (start, finish) = self.build(tail)?;
            self.epsilon(before[last], start);
            self.epsilon(finish, after[last]);
            let (start, finish) = self.build(&Expr::seq([separator.clone(), tail.clone()]))?;
            self.epsilon(after[last], start);
            self.epsilon(finish, after[last]);
        }
        Ok((before[0], end))
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
    ) -> Result<(u32, u32), GrammarError> {
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
        let end = self.add()?;
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
        Ok((start, end))
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

fn determinize(
    nfa: &[NfaState],
    rule: u32,
    start: u32,
    accept: u32,
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
    budget.dfa_growth(1, reachable.len())?;
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
            budget.dfa_growth(sets.len() + 1, reachable.len())?;
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
}
