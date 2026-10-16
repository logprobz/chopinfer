//! Request grammars compiled against the installed tokenizer's bytes.
//!
//! Each vocabulary trie is built once, on the first grammar that needs it.
//! Compiled grammars are shared and kept for reuse, so a schema that repeats
//! pays compilation and its token classes once.
//!
//! Compilation is bounded across requests, not only within one: requests
//! asking for an equal grammar while it compiles wait for that compilation
//! instead of starting their own, at most `CONCURRENT_COMPILATIONS` distinct
//! grammars compile at once, and the kept grammars are bounded by count and
//! by the bytes they hold: each grammar and its compiled form report an upper
//! estimate of their heap bytes, measured when kept, whenever reused, and
//! whenever a kept grammar grows by classifying on first use. A request
//! cancelled while it waits stops waiting. A compilation that has started
//! runs to its end and is kept, since other requests may be waiting for it.
//!
//! Two vocabularies serve two kinds of grammar. A tool-choice grammar writes
//! calls in the model's own markup and names the marker texts it writes as
//! special tokens; it compiles against the tokenizer's own split of ordinary
//! and special tokens. Every other grammar constrains answer text (a response
//! format or structured output) and names none; it compiles against a
//! vocabulary that also treats every declared boundary token as special.
//! Tokenizers leave markers such as `<tool_call>` and `</think>` ordinary, so
//! an answer grammar would otherwise admit one wherever its bytes fit, inside
//! a JSON string for one, and the completion parser would read the sampled
//! marker as structure: it refuses the completion as invalid model output,
//! or reads the rest of the answer as a tool call.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak};
use std::time::Duration;

use runtime_grammar::{CompiledGrammar, Grammar, TokenVocabulary, VocabularyToken};
use runtime_tokenizer::Tokenizer;

const CACHED_GRAMMARS: usize = 64;
/// Bytes the kept grammars and their keys may hold together; the shared
/// vocabularies are not counted.
const CACHED_GRAMMAR_BYTES: usize = 256 << 20;
/// Distinct grammars compiling at once, each with up to
/// `PRECOMPUTE_THREADS` classification workers.
const CONCURRENT_COMPILATIONS: usize = 2;
const PRECOMPUTE_THREADS: usize = 8;
/// How often a waiting request checks its cancellation.
const CANCELLATION_POLL: Duration = Duration::from_millis(20);
pub(crate) const CANCELLED: &str = "the request was cancelled while its grammar compiled";

pub(crate) struct GrammarServiceV1 {
    tokenizer: Arc<Tokenizer>,
    width: usize,
    /// The declared boundary tokens the tokenizer leaves ordinary, sorted.
    ordinary_boundaries: Vec<u32>,
    /// The tokenizer's own split, for grammars that write the model's markup.
    markup_vocabulary: OnceLock<Result<Arc<TokenVocabulary>, String>>,
    /// The same with `ordinary_boundaries` special, for answer grammars.
    answer_vocabulary: OnceLock<Result<Arc<TokenVocabulary>, String>>,
    compiled: Arc<SharedCompilations<Arc<Grammar>, Arc<CompiledGrammar>>>,
}

/// What one kept grammar holds: its key and its compiled form.
fn kept_bytes(grammar: &Arc<Grammar>, compiled: &Arc<CompiledGrammar>) -> usize {
    grammar.retained_bytes() + compiled.retained_bytes()
}

impl GrammarServiceV1 {
    /// A service for `tokenizer` and logits rows `width` tokens wide, whose
    /// declaration names `boundaries` as its boundary token identifiers.
    pub(crate) fn new(
        tokenizer: Arc<Tokenizer>,
        width: usize,
        boundaries: impl IntoIterator<Item = u32>,
    ) -> Self {
        let mut ordinary_boundaries = boundaries
            .into_iter()
            .filter(|id| {
                tokenizer
                    .token_is_special(*id)
                    .is_ok_and(|special| !special)
            })
            .collect::<Vec<_>>();
        ordinary_boundaries.sort_unstable();
        ordinary_boundaries.dedup();
        Self {
            tokenizer,
            width,
            ordinary_boundaries,
            markup_vocabulary: OnceLock::new(),
            answer_vocabulary: OnceLock::new(),
            compiled: Arc::new(SharedCompilations::new(Limits {
                compilations: CONCURRENT_COMPILATIONS,
                entries: CACHED_GRAMMARS,
                bytes: CACHED_GRAMMAR_BYTES,
            })),
        }
    }

    /// The vocabulary `grammar` compiles against: the tokenizer's own for a
    /// grammar that names special tokens, the answer vocabulary otherwise.
    fn vocabulary(&self, grammar: &Grammar) -> Result<Arc<TokenVocabulary>, String> {
        let (vocabulary, excluded) =
            if grammar.special_tokens().is_empty() && !self.ordinary_boundaries.is_empty() {
                (&self.answer_vocabulary, self.ordinary_boundaries.as_slice())
            } else {
                (&self.markup_vocabulary, &[][..])
            };
        vocabulary
            .get_or_init(|| {
                let mut tokens = Vec::new();
                while let Ok(bytes) = self.tokenizer.token_bytes(tokens.len() as u32) {
                    let id = tokens.len() as u32;
                    let special = self
                        .tokenizer
                        .token_is_special(id)
                        .map_err(|error| error.to_string())?
                        || excluded.binary_search(&id).is_ok();
                    tokens.push(VocabularyToken {
                        bytes: bytes.to_vec(),
                        special,
                    });
                }
                TokenVocabulary::new(self.width, tokens)
                    .map(Arc::new)
                    .map_err(|error| error.to_string())
            })
            .clone()
    }

    /// Compiles `grammar`, or returns the copy compiled for an equal one.
    /// This can take milliseconds; callers run it off the engine thread.
    /// While `cancelled` answers true, a call that is still waiting returns
    /// `CANCELLED`.
    pub(crate) fn compile(
        &self,
        grammar: &Arc<Grammar>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Arc<CompiledGrammar>, String> {
        self.compiled.get_or_compile(
            grammar,
            cancelled,
            || {
                let vocabulary = self.vocabulary(grammar)?;
                let compiled =
                    CompiledGrammar::new(grammar, vocabulary).map_err(|error| error.to_string())?;
                let threads = std::thread::available_parallelism()
                    .map_or(1, usize::from)
                    .min(PRECOMPUTE_THREADS);
                compiled
                    .precompute(threads)
                    .map_err(|error| error.to_string())?;
                let compiled = Arc::new(compiled);
                // Classifying on first use grows a kept grammar; the cache
                // measures it again then, and evicts past its bound.
                let cache = Arc::downgrade(&self.compiled);
                let address = Arc::as_ptr(&compiled) as usize;
                compiled.observe_growth(Box::new(move || {
                    if let Some(cache) = Weak::upgrade(&cache) {
                        cache.regrow(
                            |compiled| Arc::as_ptr(compiled) as usize == address,
                            kept_bytes,
                        );
                    }
                }));
                Ok(compiled)
            },
            kept_bytes,
        )
    }

    #[cfg(test)]
    pub(crate) fn kept_bytes(&self) -> usize {
        self.compiled.lock().map_or(0, |state| state.kept_bytes)
    }
}

#[derive(Clone, Copy, Debug)]
struct Limits {
    compilations: usize,
    entries: usize,
    bytes: usize,
}

/// One compilation in progress, and its result once it ends.
type Flight<V> = Arc<OnceLock<Result<V, String>>>;

/// Results kept most recent first, with the compilations still running.
struct State<K, V> {
    kept: VecDeque<(K, V, usize)>,
    kept_bytes: usize,
    running: Vec<(K, Flight<V>)>,
    #[cfg(test)]
    peak_running: usize,
}

/// A bounded cache whose misses compile once per distinct key.
struct SharedCompilations<K, V> {
    limits: Limits,
    state: Mutex<State<K, V>>,
    changed: Condvar,
    /// Runs once, right after a compilation's result becomes visible to the
    /// requests waiting for it.
    #[cfg(test)]
    published: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl<K: Clone + PartialEq, V: Clone> SharedCompilations<K, V> {
    fn new(limits: Limits) -> Self {
        Self {
            limits,
            state: Mutex::new(State {
                kept: VecDeque::new(),
                kept_bytes: 0,
                running: Vec::new(),
                #[cfg(test)]
                peak_running: 0,
            }),
            changed: Condvar::new(),
            #[cfg(test)]
            published: Mutex::new(None),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, State<K, V>>, String> {
        self.state
            .lock()
            .map_err(|_| "grammar cache is poisoned".to_owned())
    }

    fn wait<'a>(
        &self,
        state: MutexGuard<'a, State<K, V>>,
    ) -> Result<MutexGuard<'a, State<K, V>>, String> {
        self.changed
            .wait_timeout(state, CANCELLATION_POLL)
            .map(|(state, _)| state)
            .map_err(|_| "grammar cache is poisoned".to_owned())
    }

    /// The kept value for `key`; else the result of the running compilation
    /// of an equal key; else `compile`'s result, once a compilation slot is
    /// free. `cancelled` is asked only while the call waits, and waiting ends
    /// with `CANCELLED` when it answers true.
    fn get_or_compile(
        &self,
        key: &K,
        cancelled: &dyn Fn() -> bool,
        compile: impl FnOnce() -> Result<V, String>,
        bytes: impl Fn(&K, &V) -> usize,
    ) -> Result<V, String> {
        let mut state = self.lock()?;
        let flight = loop {
            if let Some(index) = state.kept.iter().position(|(kept, ..)| kept == key) {
                let (kept, value, size) = state.kept.remove(index).expect("index is in range");
                let measured = bytes(&kept, &value);
                state.kept_bytes = state.kept_bytes - size + measured;
                state.kept.push_front((kept, value.clone(), measured));
                self.evict(&mut state);
                return Ok(value);
            }
            if let Some(flight) = state
                .running
                .iter()
                .find(|(running, _)| running == key)
                .map(|(_, flight)| Arc::clone(flight))
            {
                while flight.get().is_none() {
                    if cancelled() {
                        return Err(CANCELLED.to_owned());
                    }
                    state = self.wait(state)?;
                }
                return flight.get().expect("the compilation ended").clone();
            }
            if state.running.len() < self.limits.compilations {
                let flight = Flight::default();
                state.running.push((key.clone(), Arc::clone(&flight)));
                #[cfg(test)]
                {
                    state.peak_running = state.peak_running.max(state.running.len());
                }
                break flight;
            }
            if cancelled() {
                return Err(CANCELLED.to_owned());
            }
            state = self.wait(state)?;
        };
        drop(state);
        // The slot and the flight are given up however the compilation
        // ends, a panic included, so waiters never wait on a lost one.
        let ending = Ending {
            cache: self,
            key,
            flight: &flight,
        };
        let result = compile();
        ending.finish(&result, bytes);
        #[cfg(test)]
        if let Some(published) = self.published.lock().unwrap().take() {
            published();
        }
        result
    }

    /// Measures again the kept value `is` picks, which has grown, and evicts
    /// past the bounds.
    fn regrow(&self, is: impl Fn(&V) -> bool, bytes: impl Fn(&K, &V) -> usize) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let state = &mut *state;
        if let Some(entry) = state.kept.iter_mut().find(|(_, value, _)| is(value)) {
            let measured = bytes(&entry.0, &entry.1);
            state.kept_bytes = state.kept_bytes - entry.2 + measured;
            entry.2 = measured;
            self.evict(state);
        }
    }

    /// Drops the least recently used values past either bound.
    fn evict(&self, state: &mut State<K, V>) {
        while state.kept.len() > self.limits.entries
            || (state.kept_bytes > self.limits.bytes && !state.kept.is_empty())
        {
            let (_, _, size) = state.kept.pop_back().expect("kept is not empty");
            state.kept_bytes -= size;
        }
    }
}

/// Releases one compilation's slot and keeps its value.
struct Ending<'a, K: Clone + PartialEq, V: Clone> {
    cache: &'a SharedCompilations<K, V>,
    key: &'a K,
    flight: &'a Flight<V>,
}

impl<K: Clone + PartialEq, V: Clone> Ending<'_, K, V> {
    /// Keeps a successful value, measured and charged, before any waiting
    /// request can see it: a request classifying with it grows a value the
    /// cache already holds, or one it already turned away.
    fn finish(self, result: &Result<V, String>, bytes: impl Fn(&K, &V) -> usize) {
        let mut state = self.state();
        if let Ok(value) = result {
            let size = bytes(self.key, value);
            if size <= self.cache.limits.bytes {
                state
                    .kept
                    .push_front((self.key.clone(), value.clone(), size));
                state.kept_bytes += size;
                self.cache.evict(&mut state);
            }
        }
        self.publish(state, result.clone());
        std::mem::forget(self);
    }

    fn state(&self) -> MutexGuard<'_, State<K, V>> {
        self.cache
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Gives up the slot and hands `result` to the waiting requests, which
    /// read it under the same lock.
    fn publish(&self, mut state: MutexGuard<'_, State<K, V>>, result: Result<V, String>) {
        state
            .running
            .retain(|(_, flight)| !Arc::ptr_eq(flight, self.flight));
        let _ = self.flight.set(result);
        drop(state);
        self.cache.changed.notify_all();
    }
}

impl<K: Clone + PartialEq, V: Clone> Drop for Ending<'_, K, V> {
    fn drop(&mut self) {
        let state = self.state();
        self.publish(state, Err("grammar compilation did not finish".to_owned()));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread;

    use super::*;

    fn limits(compilations: usize, entries: usize, bytes: usize) -> Limits {
        Limits {
            compilations,
            entries,
            bytes,
        }
    }

    fn kept<V: Clone>(cache: &SharedCompilations<u32, V>) -> Vec<u32> {
        cache
            .lock()
            .unwrap()
            .kept
            .iter()
            .map(|(key, ..)| *key)
            .collect()
    }

    /// A cancellation predicate that acknowledges, once, that its caller is
    /// waiting (the cache asks it only while waiting), and answers `cancel`.
    fn waiting<'a>(
        acknowledge: &'a mpsc::Sender<()>,
        cancel: &'a AtomicBool,
    ) -> impl Fn() -> bool + 'a {
        let sent = AtomicBool::new(false);
        move || {
            if !sent.swap(true, Ordering::SeqCst) {
                acknowledge.send(()).unwrap();
            }
            cancel.load(Ordering::SeqCst)
        }
    }

    /// A compilation that reports it started and runs until released.
    fn held<V>(
        started: &mpsc::Sender<()>,
        release: &Mutex<mpsc::Receiver<()>>,
        result: Result<V, String>,
    ) -> impl FnOnce() -> Result<V, String> {
        let started = started.clone();
        move || {
            started.send(()).unwrap();
            release.lock().unwrap().recv().unwrap();
            result
        }
    }

    /// Equal misses arriving while one compiles wait for it and share it.
    #[test]
    fn equal_misses_share_one_compilation() {
        let cache = SharedCompilations::<u32, Arc<u32>>::new(limits(2, 8, 1 << 20));
        let (started, wait_started) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let released = Mutex::new(released);
        let (acknowledge, acknowledged) = mpsc::channel();
        let never = AtomicBool::new(false);
        let results = thread::scope(|scope| {
            let first = scope.spawn(|| {
                cache.get_or_compile(
                    &1,
                    &|| false,
                    held(&started, &released, Ok(Arc::new(7))),
                    |_, _| 1,
                )
            });
            wait_started.recv().unwrap();
            let waiters = (0..4)
                .map(|_| {
                    let predicate = waiting(&acknowledge, &never);
                    let cache = &cache;
                    scope.spawn(move || {
                        cache.get_or_compile(&1, &predicate, || unreachable!("shared"), |_, _| 1)
                    })
                })
                .collect::<Vec<_>>();
            // Every waiter is inside the wait before the compilation ends.
            for _ in 0..4 {
                acknowledged.recv().unwrap();
            }
            release.send(()).unwrap();
            let mut results = vec![first.join().unwrap().unwrap()];
            for waiter in waiters {
                results.push(waiter.join().unwrap().unwrap());
            }
            results
        });
        assert!(results.iter().all(|value| Arc::ptr_eq(value, &results[0])));
        assert_eq!(cache.lock().unwrap().peak_running, 1);
    }

    /// A failed compilation reaches every request waiting for it, and is
    /// not kept.
    #[test]
    fn a_shared_compilation_failure_reaches_every_waiter() {
        let cache = SharedCompilations::<u32, u32>::new(limits(2, 8, 1 << 20));
        let (started, wait_started) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let released = Mutex::new(released);
        let (acknowledge, acknowledged) = mpsc::channel();
        let never = AtomicBool::new(false);
        let refusal = Err("refused".to_owned());
        thread::scope(|scope| {
            let first = scope.spawn(|| {
                cache.get_or_compile(
                    &1,
                    &|| false,
                    held(&started, &released, refusal.clone()),
                    |_, _| 1,
                )
            });
            wait_started.recv().unwrap();
            let waiters = (0..3)
                .map(|_| {
                    let predicate = waiting(&acknowledge, &never);
                    let cache = &cache;
                    scope.spawn(move || {
                        cache.get_or_compile(&1, &predicate, || unreachable!("shared"), |_, _| 1)
                    })
                })
                .collect::<Vec<_>>();
            for _ in 0..3 {
                acknowledged.recv().unwrap();
            }
            release.send(()).unwrap();
            assert_eq!(first.join().unwrap(), refusal);
            for waiter in waiters {
                assert_eq!(waiter.join().unwrap(), refusal);
            }
        });
        assert!(kept(&cache).is_empty());
    }

    /// Distinct misses compile at most `compilations` at once; the rest wait
    /// for a free slot. A cancelled waiter leaves without compiling, and the
    /// other compiles once a slot is free.
    #[test]
    fn distinct_misses_wait_for_a_compilation_slot_and_cancel() {
        let cache = SharedCompilations::<u32, u32>::new(limits(2, 8, 1 << 20));
        let (started, wait_started) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let released = Mutex::new(released);
        let (acknowledge, acknowledged) = mpsc::channel();
        let (cancel, never) = (AtomicBool::new(false), AtomicBool::new(false));
        thread::scope(|scope| {
            let first = scope.spawn(|| {
                cache.get_or_compile(&1, &|| false, held(&started, &released, Ok(1)), |_, _| 1)
            });
            let second = scope.spawn(|| {
                cache.get_or_compile(&2, &|| false, held(&started, &released, Ok(2)), |_, _| 1)
            });
            wait_started.recv().unwrap();
            wait_started.recv().unwrap();
            let cancelled = {
                let predicate = waiting(&acknowledge, &cancel);
                let cache = &cache;
                scope.spawn(move || {
                    cache.get_or_compile(&3, &predicate, || unreachable!("cancelled"), |_, _| 1)
                })
            };
            let patient = {
                let predicate = waiting(&acknowledge, &never);
                let cache = &cache;
                scope.spawn(move || cache.get_or_compile(&4, &predicate, || Ok(4), |_, _| 1))
            };
            acknowledged.recv().unwrap();
            acknowledged.recv().unwrap();
            cancel.store(true, Ordering::SeqCst);
            assert_eq!(cancelled.join().unwrap(), Err(CANCELLED.to_owned()));
            assert!(!patient.is_finished());
            release.send(()).unwrap();
            release.send(()).unwrap();
            assert_eq!(first.join().unwrap(), Ok(1));
            assert_eq!(second.join().unwrap(), Ok(2));
            assert_eq!(patient.join().unwrap(), Ok(4));
        });
        assert_eq!(cache.lock().unwrap().peak_running, 2);
        assert!(cache.lock().unwrap().running.is_empty());
        let mut finished = kept(&cache);
        finished.sort_unstable();
        assert_eq!(finished, [1, 2, 4]);
    }

    /// A waiter on a shared compilation stops waiting when its own request
    /// is cancelled; another waiter keeps waiting, and the compilation
    /// finishes and is kept.
    #[test]
    fn a_cancelled_waiter_leaves_a_shared_compilation() {
        let cache = SharedCompilations::<u32, u32>::new(limits(1, 8, 1 << 20));
        let (started, wait_started) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let released = Mutex::new(released);
        let (acknowledge, acknowledged) = mpsc::channel();
        let (cancel, never) = (AtomicBool::new(false), AtomicBool::new(false));
        thread::scope(|scope| {
            let first = scope.spawn(|| {
                cache.get_or_compile(&1, &|| false, held(&started, &released, Ok(5)), |_, _| 1)
            });
            wait_started.recv().unwrap();
            let cancelled = {
                let predicate = waiting(&acknowledge, &cancel);
                let cache = &cache;
                scope.spawn(move || {
                    cache.get_or_compile(&1, &predicate, || unreachable!("shared"), |_, _| 1)
                })
            };
            let patient = {
                let predicate = waiting(&acknowledge, &never);
                let cache = &cache;
                scope.spawn(move || {
                    cache.get_or_compile(&1, &predicate, || unreachable!("shared"), |_, _| 1)
                })
            };
            acknowledged.recv().unwrap();
            acknowledged.recv().unwrap();
            cancel.store(true, Ordering::SeqCst);
            assert_eq!(cancelled.join().unwrap(), Err(CANCELLED.to_owned()));
            assert!(!patient.is_finished());
            release.send(()).unwrap();
            assert_eq!(first.join().unwrap(), Ok(5));
            assert_eq!(patient.join().unwrap(), Ok(5));
        });
        assert_eq!(kept(&cache), [1]);
    }

    /// Failures are returned and not kept; a panicking compilation frees its
    /// slot.
    #[test]
    fn failed_compilations_are_not_kept_and_free_their_slot() {
        let cache = SharedCompilations::<u32, u32>::new(limits(1, 8, 1 << 20));
        assert_eq!(
            cache.get_or_compile(&1, &|| false, || Err("refused".to_owned()), |_, _| 1),
            Err("refused".to_owned())
        );
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache.get_or_compile(&2, &|| false, || panic!("compiler bug"), |_, _| 1)
        }));
        assert!(panicked.is_err());
        assert!(kept(&cache).is_empty());
        assert_eq!(
            cache.get_or_compile(&2, &|| false, || Ok(9), |_, _| 1),
            Ok(9)
        );
    }

    /// Kept values are bounded by count and by bytes, least recently used
    /// first, and a value larger than the byte bound is returned but not kept.
    #[test]
    fn kept_grammars_are_bounded_by_count_and_bytes() {
        let cache = SharedCompilations::<u32, usize>::new(limits(2, 3, 100));
        let size = |_: &u32, value: &usize| *value;
        for (key, bytes) in [(1, 30), (2, 30), (3, 30)] {
            cache
                .get_or_compile(&key, &|| false, || Ok(bytes), size)
                .unwrap();
        }
        assert_eq!(kept(&cache), [3, 2, 1]);
        // Reusing 1 makes it the most recent; a fourth entry evicts 2.
        cache
            .get_or_compile(&1, &|| false, || unreachable!("kept"), size)
            .unwrap();
        cache
            .get_or_compile(&4, &|| false, || Ok(30), size)
            .unwrap();
        assert_eq!(kept(&cache), [4, 1, 3]);
        // 60 more bytes pass the 100-byte bound: the two oldest go.
        cache
            .get_or_compile(&5, &|| false, || Ok(60), size)
            .unwrap();
        assert_eq!(kept(&cache), [5, 4]);
        assert_eq!(cache.lock().unwrap().kept_bytes, 90);
        assert_eq!(
            cache.get_or_compile(&6, &|| false, || Ok(101), size),
            Ok(101)
        );
        assert_eq!(kept(&cache), [5, 4]);
    }

    /// A kept value that grows while another is in use is measured when it
    /// grows, and the bound evicts it then; reuse also measures again.
    #[test]
    fn growth_of_any_kept_grammar_is_charged_when_it_grows() {
        let cache = SharedCompilations::<u32, Arc<AtomicUsize>>::new(limits(2, 8, 100));
        let size = |_: &u32, value: &Arc<AtomicUsize>| value.load(Ordering::SeqCst);
        let first = cache
            .get_or_compile(&1, &|| false, || Ok(Arc::new(AtomicUsize::new(10))), size)
            .unwrap();
        let second = cache
            .get_or_compile(&2, &|| false, || Ok(Arc::new(AtomicUsize::new(10))), size)
            .unwrap();
        // The older entry grows without being looked up.
        first.store(95, Ordering::SeqCst);
        cache.regrow(|value| Arc::ptr_eq(value, &first), size);
        assert_eq!(kept(&cache), [2]);
        assert_eq!(cache.lock().unwrap().kept_bytes, 10);
        // Reuse measures again too.
        second.store(40, Ordering::SeqCst);
        cache
            .get_or_compile(&2, &|| false, || unreachable!("kept"), size)
            .unwrap();
        assert_eq!(cache.lock().unwrap().kept_bytes, 40);
    }

    /// A request waiting for a compilation can classify with its result as
    /// soon as it sees it. By then the cache holds and charges the result,
    /// so growth reported at that point still enforces the bound.
    #[test]
    fn a_result_is_charged_before_a_waiter_can_grow_it() {
        let cache = SharedCompilations::<u32, Arc<AtomicUsize>>::new(limits(2, 64, 100));
        let size = |_: &u32, value: &Arc<AtomicUsize>| value.load(Ordering::SeqCst);
        let total = |cache: &SharedCompilations<u32, Arc<AtomicUsize>>| {
            cache
                .lock()
                .unwrap()
                .kept
                .iter()
                .map(|(_, value, _)| value.load(Ordering::SeqCst))
                .sum::<usize>()
        };
        cache
            .get_or_compile(&2, &|| false, || Ok(Arc::new(AtomicUsize::new(10))), size)
            .unwrap();
        let (started, wait_started) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let released = Mutex::new(released);
        let (acknowledge, acknowledged) = mpsc::channel();
        let (published, wait_published) = mpsc::channel();
        let (resume, resumed) = mpsc::channel::<()>();
        let never = AtomicBool::new(false);
        thread::scope(|scope| {
            let leader = scope.spawn(|| {
                cache.get_or_compile(
                    &1,
                    &|| false,
                    held(&started, &released, Ok(Arc::new(AtomicUsize::new(10)))),
                    size,
                )
            });
            wait_started.recv().unwrap();
            let waiter = {
                let predicate = waiting(&acknowledge, &never);
                let cache = &cache;
                scope.spawn(move || {
                    cache.get_or_compile(&1, &predicate, || unreachable!("shared"), size)
                })
            };
            acknowledged.recv().unwrap();
            // The leader stops right after its result becomes visible.
            *cache.published.lock().unwrap() = Some(Box::new(move || {
                published.send(()).unwrap();
                resumed.recv().unwrap();
            }));
            release.send(()).unwrap();
            wait_published.recv().unwrap();
            let shared = waiter.join().unwrap().unwrap();
            shared.store(95, Ordering::SeqCst);
            cache.regrow(|value| Arc::ptr_eq(value, &shared), size);
            resume.send(()).unwrap();
            assert!(Arc::ptr_eq(&leader.join().unwrap().unwrap(), &shared));
        });
        cache
            .get_or_compile(&3, &|| false, || Ok(Arc::new(AtomicUsize::new(10))), size)
            .unwrap();
        let kept = total(&cache);
        assert!(kept <= 100, "kept {kept} of 100 bytes");
        let charged = cache.lock().unwrap().kept_bytes;
        assert_eq!(charged, kept);
    }
}
