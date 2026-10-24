//! The prefix registries' host bytes stop growing at their digest cap,
//! checked against the allocator. One test, since the live-byte count is
//! global.
// SAFETY-REVIEWED: the only unsafe code is the counting allocator below,
// which forwards to the system allocator.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicIsize, Ordering};

use runtime_kv::{KvGeometry, KvPool, PREFIX_REGISTRY_DIGESTS_V1, PrefixChainV1};

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

const BLOCK_TOKENS: usize = 64;
/// Full blocks per distinct prompt, as a 32K-token prompt has 512.
const PROMPT_BLOCKS: usize = 512;

/// Distinct prompts, one salt each, whose full blocks are `digests` digests.
fn prompts(digests: usize) -> Vec<PrefixChainV1> {
    let tokens: Arc<[u32]> = (0..(BLOCK_TOKENS * PROMPT_BLOCKS) as u32).collect();
    (0..digests / PROMPT_BLOCKS)
        .map(|salt| {
            PrefixChainV1::salted(
                BLOCK_TOKENS,
                Arc::clone(&tokens),
                Vec::new(),
                Some(&salt.to_le_bytes()),
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn prefix_registries_stop_growing_at_their_digest_cap() {
    let cap = PREFIX_REGISTRY_DIGESTS_V1;
    // Three caps of distinct digests, hashed before anything is counted.
    let chains = prompts(3 * cap);
    let (first, rest) = chains.split_at(cap / PROMPT_BLOCKS);
    let mut pool = KvPool::new(KvGeometry::new(BLOCK_TOKENS, 64).unwrap(), 1).unwrap();
    for registry in ["produced", "observed"] {
        let mut record = |chains: &[PrefixChainV1]| {
            let before = live();
            for chain in chains {
                if registry == "produced" {
                    pool.record_produced(chain, 1, PROMPT_BLOCKS);
                } else {
                    pool.observe_prefix(chain);
                }
            }
            live() - before
        };
        let filled = record(first);
        // Every digest past the cap displaces the oldest one, so the
        // registry keeps what it held at the cap.
        let after = record(rest);
        let per_digest = filled as f64 / cap as f64;
        eprintln!(
            "{registry} registry: {filled} bytes for {cap} digests ({per_digest:.1} per \
             digest), {after} more after {} more",
            2 * cap
        );
        assert!(
            (64.0..=256.0).contains(&per_digest),
            "{registry}: {per_digest} bytes per digest"
        );
        assert!(
            after.unsigned_abs() <= filled.unsigned_abs() / 16,
            "{registry} registry grew by {after} bytes past its cap of {cap} digests"
        );
    }
}
