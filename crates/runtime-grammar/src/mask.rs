//! Token classes cached per grammar position.
//!
//! A position is keyed by the top two frames of a stack and whether they are
//! the whole stack. One scan of the vocabulary trie from a key sorts every
//! token into accepted (it ends inside the key's frames), rejected, or
//! dependent (only a path that returned below the key's frames could take it,
//! so the frames underneath decide). Accepted sets are stored as a bitmask
//! or as a list, whichever is smaller. These depth-independent sets are
//! supersets; the matcher rechecks tokens that can exhaust its actual depth.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::GrammarError;
use crate::automaton::{Automaton, DEAD};
use crate::stack::{Frame, MAX_CLOSURE_WORK, MAX_DEPTH};
use crate::vocab::TokenVocabulary;

/// The cache key of one stack.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct MaskKey {
    pub(crate) top: u32,
    pub(crate) below: Option<u32>,
    /// Whether nothing lies under these frames.
    whole: bool,
}

impl MaskKey {
    pub(crate) fn of(stack: &Frame) -> Self {
        match &stack.parent {
            None => Self {
                top: stack.state,
                below: None,
                whole: true,
            },
            Some(parent) => Self {
                top: stack.state,
                below: Some(parent.state),
                whole: parent.parent.is_none(),
            },
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum TokenSet {
    Words(Box<[u32]>),
    Tokens(Box<[u32]>),
    AllBut(Box<[u32]>),
}

impl TokenSet {
    /// The set of `count` tokens whose bits are set in `bits`.
    fn from_bits(bits: Vec<u32>, count: usize, size: usize) -> Self {
        let words = bits.len();
        let members = |set: bool| {
            bits.iter().enumerate().flat_map(move |(index, &word)| {
                let mut remaining = if set { word } else { !word };
                std::iter::from_fn(move || {
                    if remaining == 0 {
                        return None;
                    }
                    let bit = remaining.trailing_zeros();
                    remaining &= remaining - 1;
                    Some((index * 32) as u32 + bit)
                })
            })
        };
        if count <= words {
            return Self::Tokens(members(true).collect());
        }
        if size - count <= words {
            return Self::AllBut(
                members(false)
                    .filter(|&token| (token as usize) < size)
                    .collect(),
            );
        }
        Self::Words(bits.into_boxed_slice())
    }

    fn or_into(&self, out: &mut [u32], size: usize) {
        match self {
            Self::Words(bits) => {
                for (out, bits) in out.iter_mut().zip(bits.iter()) {
                    *out |= bits;
                }
            }
            Self::Tokens(tokens) => {
                for &token in tokens.iter() {
                    out[token as usize / 32] |= 1 << (token % 32);
                }
            }
            Self::AllBut(rejected) => {
                // `rejected` is sorted, so one pass clears its bits word by word.
                let last = out.len() - 1;
                let mut rejected = rejected.iter().peekable();
                for (index, out) in out.iter_mut().enumerate() {
                    let mut word = if index == last && size % 32 != 0 {
                        (1 << (size % 32)) - 1
                    } else {
                        u32::MAX
                    };
                    while let Some(&&token) = rejected.peek() {
                        if token as usize / 32 != index {
                            break;
                        }
                        word &= !(1 << (token % 32));
                        rejected.next();
                    }
                    *out |= word;
                }
            }
        }
    }
}

/// The classified vocabulary at one key.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct KeyMask {
    accepted: TokenSet,
    pub(crate) dependent: Box<[u32]>,
}

impl KeyMask {
    /// The heap bytes of one shared mask: its `Arc` allocation and payload.
    fn shared_bytes(&self) -> usize {
        crate::heap::arc_bytes::<Self>() + self.bytes()
    }

    fn bytes(&self) -> usize {
        let accepted = match &self.accepted {
            TokenSet::Words(items) | TokenSet::Tokens(items) | TokenSet::AllBut(items) => {
                items.len()
            }
        };
        (accepted + self.dependent.len()) * 4
    }

    pub(crate) fn or_into(&self, out: &mut [u32], size: usize) {
        self.accepted.or_into(out, size);
    }
}

const ROOT: u32 = 0;
const UNKNOWN: u32 = 1;
// A top-frame scan above a caller retains the minimum caller depth. Actual
// remaining-depth checks happen in the matcher after unioning cached masks.
const UNKNOWN_CALLER: u32 = 2;
const DEAD_CONFIG: u32 = 0;
const UNCOMPUTED: u32 = u32::MAX;

/// A key-relative copy of the matcher with interned frames and memoized
/// transitions, so a trie scan costs one table lookup per node once warm.
#[derive(Debug)]
pub(crate) struct Scanner {
    /// (state, parent, depth); entries 0, 1 and 2 stand for the true bottom,
    /// unknown frames, and an unknown caller whose depth is retained.
    frames: Vec<(u32, u32, u32)>,
    frame_ids: HashMap<(u32, u32), u32>,
    configs: Vec<(Box<[u32]>, bool)>,
    config_ids: HashMap<(Box<[u32]>, bool), u32>,
    /// 256 entries per configuration: the next configuration shifted left
    /// once, with whether a path escaped the key in the low bit.
    transitions: Vec<u32>,
    bases: HashMap<(u32, bool), Arc<KeyMask>>,
}

impl Scanner {
    /// Heap bytes this scanner's tables hold, an upper estimate. Base masks
    /// are counted whole even where the distinct set shares them.
    fn bytes(&self) -> usize {
        use crate::heap::{table_bytes, vec_bytes};
        // Each configuration's frames are held twice: by number and as the
        // index key.
        let frames = self
            .configs
            .iter()
            .map(|(frames, _)| 2 * vec_bytes::<u32>(frames.len()))
            .sum::<usize>();
        let bases = self
            .bases
            .values()
            .map(|mask| mask.shared_bytes())
            .sum::<usize>();
        vec_bytes::<(u32, u32, u32)>(self.frames.capacity())
            + table_bytes::<((u32, u32), u32)>(self.frame_ids.capacity())
            + vec_bytes::<(Box<[u32]>, bool)>(self.configs.capacity())
            + table_bytes::<((Box<[u32]>, bool), u32)>(self.config_ids.capacity())
            + frames
            + vec_bytes::<u32>(self.transitions.capacity())
            + table_bytes::<((u32, bool), Arc<KeyMask>)>(self.bases.capacity())
            + bases
    }

    pub(crate) fn new() -> Self {
        let dead: (Box<[u32]>, bool) = (Box::new([]), false);
        let mut config_ids = HashMap::new();
        config_ids.insert(dead.clone(), DEAD_CONFIG);
        Self {
            frames: vec![
                (u32::MAX, ROOT, 0),
                (u32::MAX, UNKNOWN, 0),
                (u32::MAX, UNKNOWN_CALLER, 1),
            ],
            frame_ids: HashMap::new(),
            configs: vec![dead],
            config_ids,
            transitions: vec![UNCOMPUTED; 256],
            bases: HashMap::new(),
        }
    }

    fn frame(&mut self, state: u32, parent: u32) -> u32 {
        if let Some(&id) = self.frame_ids.get(&(state, parent)) {
            return id;
        }
        let depth = self.frames[parent as usize].2 + 1;
        let id = self.frames.len() as u32;
        self.frames.push((state, parent, depth));
        self.frame_ids.insert((state, parent), id);
        id
    }

    fn config(&mut self, frames: Vec<u32>, complete: bool) -> u32 {
        let key = (frames.into_boxed_slice(), complete);
        if let Some(&id) = self.config_ids.get(&key) {
            return id;
        }
        let id = self.configs.len() as u32;
        self.configs.push(key.clone());
        self.config_ids.insert(key, id);
        self.transitions
            .resize(self.transitions.len() + 256, UNCOMPUTED);
        id
    }

    /// The configuration after `seeds`, and whether some path returned
    /// under the key's frames on the way.
    fn closure(
        &mut self,
        automaton: &Automaton,
        seeds: Vec<u32>,
    ) -> Result<(u32, bool), GrammarError> {
        let mut seen = std::collections::HashSet::new();
        let mut pending = seeds;
        let mut reading = Vec::new();
        let mut complete = false;
        let mut escaped = false;
        let mut work = pending.len();
        while let Some(frame) = pending.pop() {
            let calls = automaton.state(self.frames[frame as usize].0).calls.len();
            work = work.saturating_add(1 + calls);
            if work > MAX_CLOSURE_WORK {
                // Classifying a long token must obey the same bound as
                // accepting its bytes. Do not cache a partially explored set.
                return Err(GrammarError::too_large(format!(
                    "grammar scanner work budget exceeded (limit: {MAX_CLOSURE_WORK} units per closure)"
                )));
            }
            if !seen.insert(frame) {
                continue;
            }
            let (state_id, parent, depth) = self.frames[frame as usize];
            let state = automaton.state(state_id);
            if state.reads {
                reading.push(frame);
            }
            if state.accepting {
                match parent {
                    ROOT => complete = true,
                    UNKNOWN | UNKNOWN_CALLER => escaped = true,
                    parent => pending.push(parent),
                }
            }
            if depth >= MAX_DEPTH && !state.calls.is_empty() {
                return Err(GrammarError::too_large(format!(
                    "grammar scanner depth budget exceeded (limit: {MAX_DEPTH} frames)"
                )));
            }
            if depth < MAX_DEPTH {
                for &(rule, ret) in state.calls.iter() {
                    let caller = self.frame(ret, parent);
                    let start = automaton.rules[rule as usize].start;
                    pending.push(self.frame(start, caller));
                }
            }
        }
        reading.sort_unstable();
        Ok((self.config(reading, complete), escaped))
    }

    fn step(
        &mut self,
        automaton: &Automaton,
        config: u32,
        byte: u8,
    ) -> Result<(u32, bool), GrammarError> {
        let value = self.transitions[config as usize * 256 + byte as usize];
        if value != UNCOMPUTED {
            return Ok((value >> 1, value & 1 == 1));
        }
        let seeds = self.configs[config as usize]
            .0
            .iter()
            .filter_map(|&frame| {
                let (state, parent, _) = self.frames[frame as usize];
                let next = automaton.next(state, byte);
                (next != DEAD).then_some((next, parent))
            })
            .collect::<Vec<_>>();
        let seeds = seeds
            .into_iter()
            .map(|(state, parent)| self.frame(state, parent))
            .collect();
        let (next, escaped) = self.closure(automaton, seeds)?;
        self.transitions[config as usize * 256 + byte as usize] = (next << 1) | u32::from(escaped);
        Ok((next, escaped))
    }

    fn start(&mut self, key: MaskKey) -> u32 {
        let top = match key.below {
            None => self.frame(key.top, if key.whole { ROOT } else { UNKNOWN }),
            Some(below) => {
                let bottom = if key.whole { ROOT } else { UNKNOWN };
                let parent = self.frame(below, bottom);
                self.frame(key.top, parent)
            }
        };
        self.config(vec![top], false)
    }

    /// Reuse the tokens that finish inside the top frame for every caller.
    /// Only tokens that can return from that frame need the second frame.
    pub(crate) fn classify(
        &mut self,
        automaton: &Automaton,
        vocabulary: &TokenVocabulary,
        key: MaskKey,
    ) -> Result<KeyMask, GrammarError> {
        let whitespace = automaton.whitespace_class(key.top);
        let base_top = whitespace.map_or(key.top, |(representative, _)| representative);
        let base_key = (base_top, key.below.is_some());
        let base = match self.bases.get(&base_key) {
            Some(base) => Arc::clone(base),
            None => {
                let base = Arc::new(self.scan(
                    automaton,
                    vocabulary,
                    MaskKey {
                        top: base_top,
                        below: None,
                        whole: false,
                    },
                    base_key.1,
                )?);
                self.bases.insert(base_key, Arc::clone(&base));
                base
            }
        };
        let trim = whitespace.filter(|&(_, budget)| budget < vocabulary.whitespace_horizon());
        if base.dependent.is_empty() && trim.is_none() {
            return Ok((*base).clone());
        }
        let start = self.start(key);
        let mut accepted = vec![0; vocabulary.mask_words()];
        base.or_into(&mut accepted, vocabulary.size());
        let mut dependent = Vec::new();
        for &token in base.dependent.iter() {
            let mut config = start;
            let mut escaped = false;
            for &byte in vocabulary.ordinary_bytes(token).expect("ordinary token") {
                let (next, escaped_now) = self.step(automaton, config, byte)?;
                config = next;
                escaped |= escaped_now;
                if config == DEAD_CONFIG {
                    break;
                }
            }
            if config != DEAD_CONFIG {
                accepted[token as usize / 32] |= 1 << (token % 32);
            } else if escaped {
                dependent.push(token);
            }
        }
        if let Some((_, budget)) = trim {
            // These states cannot return or call before exhausting the run,
            // so an overlong whitespace prefix is rejected, never dependent.
            for token in vocabulary.longer_whitespace_prefixes(budget) {
                accepted[token as usize / 32] &= !(1 << (token % 32));
            }
        }
        let count = accepted.iter().map(|word| word.count_ones() as usize).sum();
        Ok(KeyMask {
            accepted: TokenSet::from_bits(accepted, count, vocabulary.size()),
            dependent: dependent.into_boxed_slice(),
        })
    }

    /// Classifies every ordinary token at `key`.
    fn scan(
        &mut self,
        automaton: &Automaton,
        vocabulary: &TokenVocabulary,
        key: MaskKey,
        has_caller: bool,
    ) -> Result<KeyMask, GrammarError> {
        let start = if has_caller {
            let top = self.frame(key.top, UNKNOWN_CALLER);
            self.config(vec![top], false)
        } else {
            self.start(key)
        };
        let mut accepted = vec![0_u32; vocabulary.mask_words()];
        let mut accepted_count = 0;
        let mut dependent = Vec::new();
        let depth = vocabulary.max_depth + 1;
        let mut configs = vec![start; depth];
        let mut escapes = vec![false; depth];
        let nodes = &vocabulary.nodes;
        let mut index = 0;
        while index < nodes.len() {
            let node = nodes[index];
            let level = node.depth as usize;
            let (config, escaped_now) = self.step(automaton, configs[level - 1], node.byte)?;
            let escaped = escapes[level - 1] || escaped_now;
            if config == DEAD_CONFIG {
                if escaped {
                    for inner in &nodes[index..node.end as usize] {
                        dependent.extend_from_slice(
                            &vocabulary.node_tokens
                                [inner.tokens.0 as usize..inner.tokens.1 as usize],
                        );
                    }
                }
                index = node.end as usize;
                continue;
            }
            configs[level] = config;
            escapes[level] = escaped;
            for &token in &vocabulary.node_tokens[node.tokens.0 as usize..node.tokens.1 as usize] {
                accepted[token as usize / 32] |= 1 << (token % 32);
            }
            accepted_count += (node.tokens.1 - node.tokens.0) as usize;
            index += 1;
        }
        dependent.sort_unstable();
        Ok(KeyMask {
            accepted: TokenSet::from_bits(accepted, accepted_count, vocabulary.size()),
            dependent: dependent.into_boxed_slice(),
        })
    }
}

/// Every classified key of one compiled grammar.
#[derive(Debug)]
pub(crate) struct MaskCache {
    masks: RwLock<HashMap<MaskKey, Arc<KeyMask>>>,
    states: Vec<u32>,
    /// Distinct classifications; many positions share one, such as the
    /// characters of a long bounded string.
    distinct: Mutex<HashSet<Arc<KeyMask>>>,
    scanner: Mutex<Scanner>,
    growth: GrowthObserver,
}

/// Told after each classification computed on first use, once no cache lock
/// is held, so an owner can measure the grammar again.
#[derive(Default)]
pub(crate) struct GrowthObserver(OnceLock<Box<dyn Fn() + Send + Sync>>);

impl GrowthObserver {
    pub(crate) fn set(&self, observer: Box<dyn Fn() + Send + Sync>) -> bool {
        self.0.set(observer).is_ok()
    }

    fn notify(&self) {
        if let Some(observer) = self.0.get() {
            observer();
        }
    }
}

impl std::fmt::Debug for GrowthObserver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(if self.0.get().is_some() {
            "GrowthObserver(set)"
        } else {
            "GrowthObserver(unset)"
        })
    }
}

impl MaskCache {
    pub(crate) fn new(automaton: &Automaton, vocabulary: &TokenVocabulary) -> Self {
        Self {
            states: automaton.classification_states(vocabulary.whitespace_horizon()),
            masks: RwLock::new(HashMap::new()),
            distinct: Mutex::new(HashSet::new()),
            scanner: Mutex::new(Scanner::new()),
            growth: GrowthObserver::default(),
        }
    }

    pub(crate) fn growth(&self) -> &GrowthObserver {
        &self.growth
    }

    fn canonical(&self, key: MaskKey) -> MaskKey {
        MaskKey {
            top: self.states[key.top as usize],
            below: key.below.map(|state| self.states[state as usize]),
            whole: key.whole,
        }
    }

    pub(crate) fn get(
        &self,
        automaton: &Automaton,
        vocabulary: &TokenVocabulary,
        key: MaskKey,
    ) -> Result<Arc<KeyMask>, GrammarError> {
        let key = self.canonical(key);
        if let Some(mask) = self.masks.read().expect("mask cache lock").get(&key) {
            return Ok(Arc::clone(mask));
        }
        let mut scanner = self.scanner.lock().expect("mask scanner lock");
        if let Some(mask) = self.masks.read().expect("mask cache lock").get(&key) {
            return Ok(Arc::clone(mask));
        }
        // A refused classification can still have grown the scanner's
        // tables, so the observer hears of both outcomes.
        let classified = scanner.classify(automaton, vocabulary, key).map(|mask| {
            let mask = self.intern(mask);
            self.masks
                .write()
                .expect("mask cache lock")
                .insert(key, Arc::clone(&mask));
            mask
        });
        drop(scanner);
        self.growth.notify();
        classified
    }

    pub(crate) fn insert(&self, key: MaskKey, mask: KeyMask) {
        let mask = self.intern(mask);
        self.masks
            .write()
            .expect("mask cache lock")
            .entry(key)
            .or_insert(mask);
    }

    fn intern(&self, mask: KeyMask) -> Arc<KeyMask> {
        let mut distinct = self.distinct.lock().expect("mask cache lock");
        if let Some(known) = distinct.get(&mask) {
            return Arc::clone(known);
        }
        let mask = Arc::new(mask);
        distinct.insert(Arc::clone(&mask));
        mask
    }

    /// Number of distinct classifications and the bytes they hold.
    pub(crate) fn distinct(&self) -> (usize, usize) {
        let distinct = self.distinct.lock().expect("mask cache lock");
        let bytes = distinct.iter().map(|mask| mask.bytes()).sum();
        (distinct.len(), bytes)
    }

    pub(crate) fn len(&self) -> usize {
        self.masks.read().expect("mask cache lock").len()
    }

    /// Heap bytes held for classifications, an upper estimate: the key
    /// table, the distinct masks, the canonical states and the lazy
    /// scanner's tables.
    pub(crate) fn retained_bytes(&self) -> usize {
        use crate::heap::{table_bytes, vec_bytes};
        let keys = table_bytes::<(MaskKey, Arc<KeyMask>)>(
            self.masks.read().expect("mask cache lock").capacity(),
        );
        let masks = {
            let distinct = self.distinct.lock().expect("mask cache lock");
            table_bytes::<Arc<KeyMask>>(distinct.capacity())
                + distinct
                    .iter()
                    .map(|mask| mask.shared_bytes())
                    .sum::<usize>()
        };
        let scanner = self.scanner.lock().expect("mask scanner lock").bytes();
        keys + masks + vec_bytes::<u32>(self.states.capacity()) + scanner
    }

    /// Every key a stack can have, or `None` when there are more than
    /// `limit`. A stack's top reads a byte; with a single frame it is in the
    /// root rule, and otherwise it is in a called rule above that call's
    /// return state, which is the whole stack's bottom only when the caller
    /// is the root rule, and has frames under it only when the caller can
    /// itself be called.
    pub(crate) fn reachable_keys(
        &self,
        automaton: &Automaton,
        limit: usize,
        canonical: bool,
    ) -> Option<Vec<MaskKey>> {
        let key_for = |key| if canonical { self.canonical(key) } else { key };
        let mut reading = vec![Vec::new(); automaton.rules.len()];
        for (index, state) in automaton.states.iter().enumerate() {
            if state.reads {
                reading[state.rule as usize].push(index as u32);
            }
        }
        let mut called = vec![false; automaton.rules.len()];
        for state in &automaton.states {
            for &(rule, _) in state.calls.iter() {
                called[rule as usize] = true;
            }
        }
        let mut keys = std::collections::HashSet::new();
        let root = automaton.root as usize;
        for &top in &reading[root] {
            keys.insert(key_for(MaskKey {
                top,
                below: None,
                whole: true,
            }));
            if keys.len() > limit {
                return None;
            }
        }
        for state in &automaton.states {
            let caller = state.rule as usize;
            for &(rule, ret) in state.calls.iter() {
                for whole in [true, false] {
                    if (whole && caller != root) || (!whole && !called[caller]) {
                        continue;
                    }
                    for &top in &reading[rule as usize] {
                        keys.insert(key_for(MaskKey {
                            top,
                            below: Some(ret),
                            whole,
                        }));
                        if keys.len() > limit {
                            return None;
                        }
                    }
                }
            }
        }
        let mut keys: Vec<MaskKey> = keys.into_iter().collect();
        keys.sort_unstable_by_key(|key| (key.top, key.below, key.whole));
        Some(keys)
    }
}
