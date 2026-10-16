//! `retained_bytes` is an upper bound on what grammars really keep, checked
//! against the allocator. One test, since the live-byte count is global.
// SAFETY-REVIEWED: the only unsafe code is the counting allocator below,
// which forwards to the system allocator.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

use runtime_grammar::{
    CompiledGrammar, GrammarMatcher, JsonOptions, TokenVocabulary, VocabularyToken,
    json_schema_grammar, regex_grammar,
};

struct Counting;

static LIVE: AtomicIsize = AtomicIsize::new(0);

// SAFETY-REVIEWED: every method forwards to the system allocator with the
// caller's pointer and layout unchanged and only updates a counter.
unsafe impl GlobalAlloc for Counting {
    // SAFETY-REVIEWED: forwards the caller's layout contract unchanged.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY-REVIEWED: the caller guarantees a valid nonzero layout.
        let pointer = unsafe { System.alloc(layout) };
        count(pointer, layout.size() as isize)
    }

    // SAFETY-REVIEWED: forwards the caller's layout contract unchanged.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY-REVIEWED: the caller guarantees a valid nonzero layout.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        count(pointer, layout.size() as isize)
    }

    // SAFETY-REVIEWED: forwards a pointer this allocator returned.
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::SeqCst);
        // SAFETY-REVIEWED: the pointer came from System with this layout.
        unsafe { System.dealloc(pointer, layout) }
    }

    // SAFETY-REVIEWED: forwards a pointer this allocator returned.
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY-REVIEWED: the pointer came from System with this layout.
        let next = unsafe { System.realloc(pointer, layout, size) };
        count(next, size as isize - layout.size() as isize)
    }
}

fn count(pointer: *mut u8, bytes: isize) -> *mut u8 {
    if !pointer.is_null() {
        LIVE.fetch_add(bytes, Ordering::SeqCst);
    }
    pointer
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn live() -> isize {
    LIVE.load(Ordering::SeqCst)
}

#[test]
fn retained_bytes_bound_what_the_allocator_keeps() {
    let tokens = (0..256_u16)
        .map(|byte| VocabularyToken {
            bytes: vec![byte as u8],
            special: false,
        })
        .chain(
            ["ab", "abc", "{\"", "\":", "\"}", "12"]
                .iter()
                .map(|text| VocabularyToken {
                    bytes: text.as_bytes().to_vec(),
                    special: false,
                }),
        )
        .collect::<Vec<_>>();
    let width = tokens.len();
    let vocabulary = Arc::new(TokenVocabulary::new(width, tokens).unwrap());
    let schema = serde_json::json!({"type": "object", "properties": {
        "name": {"type": "string", "maxLength": 12},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2}
    }, "required": ["name"]});
    for precompute in [false, true] {
        for source in ["regex", "schema"] {
            let before = live();
            let grammar = Arc::new(if source == "regex" {
                regex_grammar("[a-z]{512}").unwrap()
            } else {
                json_schema_grammar(&schema, JsonOptions::default()).unwrap()
            });
            let kept = live() - before;
            assert!(
                grammar.retained_bytes() as isize >= kept,
                "{source}: grammar reports {} of {kept} bytes",
                grammar.retained_bytes()
            );
            let before = live();
            let compiled =
                Arc::new(CompiledGrammar::new(&grammar, Arc::clone(&vocabulary)).unwrap());
            // One worker runs on this thread. A spawned worker frees its own
            // thread state after it is joined, which this count would see.
            if precompute {
                compiled.precompute(1).unwrap();
            }
            let kept = live() - before;
            let reported = compiled.retained_bytes();
            assert!(
                reported as isize >= kept,
                "{source} precompute={precompute}: compiled reports {reported} of {kept} bytes"
            );
            // Classifications filled on first use are reported, and the
            // observer hears of each.
            let grown = Arc::new(AtomicUsize::new(0));
            let heard = Arc::clone(&grown);
            assert!(compiled.observe_growth(Box::new(move || {
                heard.fetch_add(1, Ordering::SeqCst);
            })));
            let mut bits = vec![0; vocabulary.mask_words()];
            let before = live();
            let mut matcher = GrammarMatcher::new(Arc::clone(&compiled));
            let text: &[u8] = if source == "regex" {
                b"ab"
            } else {
                b"{\"name\":\"ab"
            };
            for &byte in text {
                matcher.fill_token_bitmask(&mut bits).unwrap();
                matcher.accept_token(u32::from(byte)).unwrap();
            }
            matcher.fill_token_bitmask(&mut bits).unwrap();
            drop(matcher);
            let kept = live() - before;
            let after = compiled.retained_bytes();
            assert!(
                (after - reported) as isize >= kept,
                "{source} precompute={precompute}: growth reports {} of {kept} bytes",
                after - reported
            );
            assert_eq!(grown.load(Ordering::SeqCst) > 0, kept > 0);
        }
    }
}
