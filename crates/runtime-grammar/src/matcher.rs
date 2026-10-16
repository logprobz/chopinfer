//! Compiled grammars and the per-request state that follows them.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use crate::automaton::Automaton;
use crate::expr::{Grammar, GrammarError};
use crate::mask::{MaskCache, MaskKey, Scanner};
use crate::stack::{Config, MAX_DEPTH};
use crate::vocab::TokenVocabulary;

/// Bound the full vocabulary scans, after equivalent positions share work.
const EAGER_POSITIONS: usize = 1 << 13;
/// Bound enumeration and storage of caller contexts independently of scans.
const EAGER_KEYS: usize = 1 << 16;
/// Committed tokens a matcher can roll back.
pub const MAX_ROLLBACK: usize = 64;

/// A grammar compiled against one vocabulary, shared by every request that
/// uses it.
#[derive(Debug)]
pub struct CompiledGrammar {
    automaton: Automaton,
    compilation_usage: crate::CompileUsage,
    vocabulary: Arc<TokenVocabulary>,
    special: Box<[u32]>,
    cache: MaskCache,
    max_call_growth: u32,
}

impl CompiledGrammar {
    pub fn new(grammar: &Grammar, vocabulary: Arc<TokenVocabulary>) -> Result<Self, GrammarError> {
        let budget = crate::budget::Budget::default();
        let automaton = Automaton::compile_with_budget(grammar, &budget)?;
        let mut special = grammar
            .special_tokens()
            .iter()
            .flat_map(|text| vocabulary.special_with_text(text).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        special.sort_unstable();
        special.dedup();
        let cache = MaskCache::new(&automaton, &vocabulary);
        let max_call_growth = automaton.max_call_growth();
        Ok(Self {
            max_call_growth,
            compilation_usage: budget.usage(),
            automaton,
            vocabulary,
            special: special.into_boxed_slice(),
            cache,
        })
    }

    /// Costs charged by automaton compilation, independent of tokenizer size.
    #[must_use]
    pub fn compilation_usage(&self) -> crate::CompileUsage {
        self.compilation_usage
    }

    #[must_use]
    pub fn vocabulary(&self) -> &Arc<TokenVocabulary> {
        &self.vocabulary
    }

    /// Classifies every position a stack can reach ahead of use, spread
    /// over up to `threads` workers. Grammars with more positions than the
    /// eager scan or caller-context bound fill their cache on first use instead.
    /// Returns the number of classified positions, or an error if any scan
    /// exhausts a resource budget. Failed classifications are never published.
    pub fn precompute(&self, threads: usize) -> Result<usize, GrammarError> {
        let Some(mut keys) = self.cache.reachable_keys(&self.automaton, EAGER_KEYS, true) else {
            return Ok(0);
        };
        let scan_top = |key: &MaskKey| {
            (
                self.automaton
                    .whitespace_class(key.top)
                    .map_or(key.top, |(representative, _)| representative),
                key.below.is_some(),
            )
        };
        keys.sort_unstable_by_key(scan_top);
        let groups: Vec<_> = keys
            .chunk_by(|left, right| scan_top(left) == scan_top(right))
            .collect();
        if groups.len() > EAGER_POSITIONS {
            return Ok(0);
        }
        let threads = threads.clamp(1, groups.len().max(1));
        // Keep caller contexts of one top frame on the same worker, so its
        // vocabulary scan is shared by every context.
        let groups = &groups;
        let classified = AtomicUsize::new(0);
        let failure = Mutex::new(None);
        let work = |worker: usize| {
            let mut scanner = Scanner::new();
            for group in groups.iter().skip(worker).step_by(threads) {
                for &key in *group {
                    if failure.lock().expect("precompute failure lock").is_some() {
                        return;
                    }
                    match scanner.classify(&self.automaton, &self.vocabulary, key) {
                        Ok(mask) => {
                            self.cache.insert(key, mask);
                            classified.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(error) => {
                            *failure.lock().expect("precompute failure lock") = Some(error);
                            return;
                        }
                    }
                }
            }
        };
        if threads == 1 {
            // One worker runs here: a spawned thread would still be exiting,
            // and releasing its own runtime state, after the scope returns.
            work(0);
        } else {
            thread::scope(|scope| {
                for worker in 0..threads {
                    let work = &work;
                    scope.spawn(move || work(worker));
                }
            });
        }
        if let Some(error) = failure.into_inner().expect("precompute failure lock") {
            return Err(error);
        }
        Ok(classified.load(Ordering::Relaxed))
    }

    /// Reachable stack positions before classification sharing, or `None`
    /// when enumeration exceeds `limit`. Intended for compile diagnostics.
    #[must_use]
    pub fn reachable_positions(&self, limit: usize) -> Option<usize> {
        self.cache
            .reachable_keys(&self.automaton, limit, false)
            .map(|keys| keys.len())
    }

    /// Number of classified positions so far.
    #[must_use]
    pub fn cached_positions(&self) -> usize {
        self.cache.len()
    }

    /// Heap bytes this grammar holds now, an upper estimate from the
    /// capacities of what it owns: its own `Arc` allocation, its automaton
    /// and every classification cached so far. The shared vocabulary is not
    /// counted. Classifications filled on first use after `precompute` add to
    /// it; see `observe_growth`.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        crate::heap::arc_bytes::<Self>()
            + self.automaton.retained_bytes()
            + crate::heap::vec_bytes::<u32>(self.special.len())
            + self.cache.retained_bytes()
    }

    /// Calls `observer` after each classification this grammar computes on
    /// first use, with no cache lock held, so its owner can measure it again.
    /// Returns false, keeping the first, when an observer is already set.
    pub fn observe_growth(&self, observer: Box<dyn Fn() + Send + Sync>) -> bool {
        self.cache.growth().set(observer)
    }

    /// Distinct token classifications held for this grammar, and the bytes
    /// they occupy.
    #[must_use]
    pub fn cached_classes(&self) -> (usize, usize) {
        self.cache.distinct()
    }
}

/// Where a match is: the configuration after the tokens accepted so far.
#[derive(Clone, Debug)]
pub struct GrammarMatcher {
    grammar: Arc<CompiledGrammar>,
    config: Config,
    // Exhaustion is terminal for a request, including its speculative forks.
    failed: Arc<OnceLock<GrammarError>>,
}

impl GrammarMatcher {
    #[must_use]
    pub fn new(grammar: Arc<CompiledGrammar>) -> Self {
        let config = Config::initial(&grammar.automaton);
        let failed = Arc::new(OnceLock::new());
        if let Some(error) = &config.failure {
            let _ = failed.set(error.clone());
        }
        Self {
            grammar,
            config,
            failed,
        }
    }

    #[must_use]
    pub fn grammar(&self) -> &Arc<CompiledGrammar> {
        &self.grammar
    }

    /// The terminal resource failure, shared by this request's speculative forks.
    #[must_use]
    pub fn failure(&self) -> Option<&GrammarError> {
        self.failed.get()
    }

    /// Whether the text so far is a complete match.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.failed.get().is_none() && self.config.complete
    }

    /// Whether any byte can follow.
    #[must_use]
    pub fn can_continue(&self) -> bool {
        self.failed.get().is_none() && !self.config.stacks.is_empty()
    }

    fn token_bytes(&self, token_id: u32) -> Option<&[u8]> {
        let vocabulary = &self.grammar.vocabulary;
        vocabulary.ordinary_bytes(token_id).or_else(|| {
            self.grammar
                .special
                .binary_search(&token_id)
                .ok()
                .and_then(|_| vocabulary.special_bytes(token_id))
        })
    }

    fn check(&self) -> Result<(), GrammarError> {
        match self.failed.get() {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn checked_config(&self, next: Config) -> Result<Config, GrammarError> {
        if let Some(error) = &next.failure {
            let _ = self.failed.set(error.clone());
            self.check()?;
        }
        Ok(next)
    }

    /// Whether the bytes of this token continue the match. Exhaustion fails
    /// the request; it is never a grammatical rejection.
    pub fn accepts_token(&self, token_id: u32) -> Result<bool, GrammarError> {
        self.check()?;
        let Some(bytes) = self.token_bytes(token_id) else {
            return Ok(false);
        };
        Ok(!self
            .checked_config(self.config.step_bytes(&self.grammar.automaton, bytes))?
            .is_dead())
    }

    /// Advances over one token's bytes.
    pub fn accept_token(&mut self, token_id: u32) -> Result<(), GrammarError> {
        self.check()?;
        let bytes = self
            .token_bytes(token_id)
            .ok_or(GrammarError::Rejected { token_id })?;
        let next = self.checked_config(self.config.step_bytes(&self.grammar.automaton, bytes))?;
        if next.is_dead() {
            return Err(GrammarError::Rejected { token_id });
        }
        self.config = next;
        Ok(())
    }

    /// Advances over raw bytes; used by tests and callers that feed text.
    pub fn accept_bytes(&mut self, bytes: &[u8]) -> Result<(), GrammarError> {
        self.check()?;
        let next = self.checked_config(self.config.step_bytes(&self.grammar.automaton, bytes))?;
        if next.is_dead() {
            return Err(GrammarError::invalid(
                "bytes do not continue the constraint",
            ));
        }
        self.config = next;
        Ok(())
    }

    /// Sets the bit of every byte-matching token that continues the match.
    /// Stop tokens are the caller's to add. `out` is cleared first.
    pub fn fill_token_bitmask(&self, out: &mut [u32]) -> Result<(), GrammarError> {
        out.fill(0);
        let result = self.fill_token_bitmask_inner(out);
        if let Err(error) = &result {
            out.fill(0);
            let _ = self.failed.set(error.clone());
        }
        result
    }

    fn fill_token_bitmask_inner(&self, out: &mut [u32]) -> Result<(), GrammarError> {
        self.check()?;
        let grammar = &self.grammar;
        let vocabulary = &grammar.vocabulary;
        out.fill(0);
        for stack in &self.config.stacks {
            let mask = grammar
                .cache
                .get(&grammar.automaton, vocabulary, MaskKey::of(stack))?;
            mask.or_into(out, vocabulary.size());
            if mask.dependent.is_empty() {
                continue;
            }
            let single = Config::single(stack);
            for &token in mask.dependent.iter() {
                if out[token as usize / 32] & (1 << (token % 32)) != 0 {
                    continue;
                }
                let bytes = vocabulary
                    .ordinary_bytes(token)
                    .expect("trie tokens have bytes");
                if !self
                    .checked_config(single.step_bytes(&grammar.automaton, bytes))?
                    .is_dead()
                {
                    out[token as usize / 32] |= 1 << (token % 32);
                }
            }
        }
        // Cached scans start at the minimum depth for their top frames.
        // They are a superset at greater depths. A closure can push several
        // frames per byte, including through nullable rules, so use the
        // automaton's proven bound rather than assuming one call per byte.
        if grammar.max_call_growth != 0 {
            let depth = self
                .config
                .stacks
                .iter()
                .map(|stack| stack.depth)
                .max()
                .unwrap_or(0);
            let budget = ((MAX_DEPTH - depth) / grammar.max_call_growth) as usize;
            if budget < vocabulary.max_depth {
                for &token in vocabulary.longer_tokens(budget) {
                    let word = &mut out[token as usize / 32];
                    let bit = 1 << (token % 32);
                    if *word & bit != 0 && !self.accepts_token(token)? {
                        *word &= !bit;
                    }
                }
            }
        }
        for &token in grammar.special.iter() {
            if self.accepts_token(token)? {
                out[token as usize / 32] |= 1 << (token % 32);
            }
        }
        Ok(())
    }
}

/// Two states are equal when they are at the same position of the same
/// compiled grammar; rollback history is not compared.
impl PartialEq for ConstraintState {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.matcher.grammar, &other.matcher.grammar)
            && self.matcher.failed.get() == other.matcher.failed.get()
            && self.matcher.config == other.matcher.config
            && self.gate == other.gate
            && self.stop_tokens == other.stop_tokens
            && self.terminated == other.terminated
    }
}

#[derive(Clone, Debug)]
struct Checkpoint {
    matcher: Config,
    gate: Option<Arc<[u32]>>,
    terminated: bool,
}

/// One request's constraint: an optional gate that must open first, the
/// matcher, and the stop tokens admitted where the grammar may end.
#[derive(Clone, Debug)]
pub struct ConstraintState {
    matcher: GrammarMatcher,
    gate: Option<Arc<[u32]>>,
    stop_tokens: Arc<[u32]>,
    terminated: bool,
    history: VecDeque<Checkpoint>,
}

impl ConstraintState {
    /// `gate`, when given, lists tokens that start the constraint: rows are
    /// unconstrained until one of them is accepted. `stop_tokens` are the
    /// request's stop identifiers.
    #[must_use]
    pub fn new(
        grammar: Arc<CompiledGrammar>,
        gate: Option<Vec<u32>>,
        mut stop_tokens: Vec<u32>,
    ) -> Self {
        stop_tokens.sort_unstable();
        stop_tokens.dedup();
        Self {
            matcher: GrammarMatcher::new(grammar),
            gate: gate.map(|mut tokens| {
                tokens.sort_unstable();
                tokens.dedup();
                tokens.into()
            }),
            stop_tokens: stop_tokens.into(),
            terminated: false,
            history: VecDeque::new(),
        }
    }

    /// A copy at the same position without rollback history, for deriving
    /// the rows after speculative tokens.
    #[must_use]
    pub fn fork(&self) -> Self {
        Self {
            matcher: self.matcher.clone(),
            gate: self.gate.clone(),
            stop_tokens: Arc::clone(&self.stop_tokens),
            terminated: self.terminated,
            history: VecDeque::new(),
        }
    }

    /// Whether the next row is constrained.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.matcher.check().is_err()
            || (self.gate.is_none()
                && !self.terminated
                && (self.matcher.can_continue() || !self.stop_tokens.is_empty()))
    }

    #[must_use]
    pub fn is_terminated(&self) -> bool {
        self.matcher.check().is_ok() && self.terminated
    }

    #[must_use]
    pub fn matcher(&self) -> &GrammarMatcher {
        &self.matcher
    }

    /// Writes the next row's allowed tokens. Returns `Ok(false)`, leaving
    /// `out` untouched, when the row is unconstrained.
    pub fn fill_mask(&self, out: &mut [u32]) -> Result<bool, GrammarError> {
        if let Err(error) = self.matcher.check() {
            out.fill(0);
            return Err(error);
        }
        if !self.is_active() {
            return Ok(false);
        }
        self.matcher.fill_token_bitmask(out)?;
        if self.matcher.is_complete() {
            let size = self.matcher.grammar.vocabulary.size();
            for &token in self.stop_tokens.iter() {
                if (token as usize) < size {
                    out[token as usize / 32] |= 1 << (token % 32);
                }
            }
        }
        if out.iter().all(|word| *word == 0) {
            return Err(GrammarError::NoAllowedToken);
        }
        Ok(true)
    }

    /// Whether the constraint admits `token_id` next.
    pub fn allows(&self, token_id: u32) -> Result<bool, GrammarError> {
        self.matcher.check()?;
        if !self.is_active() {
            return Ok(true);
        }
        if self.stop_tokens.binary_search(&token_id).is_ok() {
            return Ok(self.matcher.is_complete());
        }
        self.matcher.accepts_token(token_id)
    }

    /// Commits one token.
    pub fn accept(&mut self, token_id: u32) -> Result<(), GrammarError> {
        self.matcher.check()?;
        let checkpoint = Checkpoint {
            matcher: self.matcher.config.clone(),
            gate: self.gate.clone(),
            terminated: self.terminated,
        };
        if let Some(gate) = &self.gate {
            if gate.binary_search(&token_id).is_ok() {
                self.gate = None;
                // A gate token that is also the grammar's first text (a tool
                // call marker) is consumed by it.
                if self.matcher.accepts_token(token_id)? {
                    self.matcher.accept_token(token_id)?;
                }
            }
        } else if !self.terminated {
            if self.stop_tokens.binary_search(&token_id).is_ok() {
                if !self.matcher.is_complete() {
                    return Err(GrammarError::Rejected { token_id });
                }
                self.terminated = true;
            } else if !self.matcher.can_continue() && self.stop_tokens.is_empty() {
                self.terminated = true;
            } else {
                self.matcher.accept_token(token_id)?;
            }
        }
        if self.history.len() == MAX_ROLLBACK {
            self.history.pop_front();
        }
        self.history.push_back(checkpoint);
        Ok(())
    }

    /// Undoes the last `count` accepted tokens.
    pub fn rollback(&mut self, count: usize) -> Result<(), GrammarError> {
        if count > self.history.len() {
            return Err(GrammarError::invalid(
                "rollback exceeds the retained history",
            ));
        }
        let mut restored = None;
        for _ in 0..count {
            restored = self.history.pop_back();
        }
        if let Some(checkpoint) = restored {
            self.matcher.config = checkpoint.matcher;
            self.gate = checkpoint.gate;
            self.terminated = checkpoint.terminated;
        }
        Ok(())
    }
}
