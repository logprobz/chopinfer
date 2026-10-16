//! The pooled indexer selection fixture shared by the one-split case
//! (`pooled_indexer_conformance.rs`) and the split merge chain
//! (`pooled_indexer_merge_conformance.rs`): integer queries, head weights
//! and pooled keys over shuffled pages of a larger pool, whose candidate
//! keys the host reproduces bit for bit (as `tests/pooled_indexer.rs`
//! does).
#![allow(dead_code)] // Shared by test binaries that use different parts.

use std::ops::Range;

use crate::hot_keys::{bf16_bytes, bf16_values, i32_bytes};
use crate::shapes;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, DType};
use kernel_lab::LabResult;
use kernel_lab::conformance::{Case, Holes, Images, Rows};
use kernels_sm120::pooled_indexer::{Entry, Kind};

pub const HEADS: usize = 32;
pub const DIM: usize = 128;
pub const GROUP: usize = 4;
pub const PAGE: usize = 64;
/// Pools of one page.
pub const PAGE_POOLS: usize = PAGE / GROUP;
/// Candidate keys a split keeps at most (`max_top_k / group`).
pub const MAX_KEPT: usize = 512;
/// The runtime top-k, in tokens: each row keeps four pools.
pub const TOP_K: usize = 16;
/// Pages of the pool, more than the items use, so every page can move up.
pub const POOL_PAGES: usize = 16;
/// A page of pooled keys and 512 decoy bytes.
pub const STRIDE: usize = PAGE_POOLS * DIM * 2 + 512;
/// Pages of distinct codes, whose `c + 1` stays exact in BF16.
pub const CODE_PAGES: usize = 16;

/// The global selection entry.
pub fn entry() -> Entry {
    shapes::lab_entries()
        .into_iter()
        .find(|entry| entry.kind == Kind::Select { ranks: None })
        .expect("a global selection entry")
}

/// Small integers as BF16.
fn integers(values: impl Iterator<Item = i32>) -> Vec<u16> {
    values.map(|value| f32_to_bf16(value as f32)).collect()
}

/// The pooled keys of `pages` pages: pool row `r` of page `p` holds code
/// `c = p % 16 * 16 + r` as `[c + 1, c * 5 % 7 + 1, (c * 3 + d) % 5 ...]`,
/// then decoy bytes.
pub fn pooled_keys(pages: usize) -> Vec<u8> {
    let distinct = pages.min(CODE_PAGES);
    let mut pattern = vec![0x7f; distinct * STRIDE];
    for code in 0..distinct * PAGE_POOLS {
        let key = integers((0..DIM as i32).map(|d| match d {
            0 => code as i32 + 1,
            1 => code as i32 * 5 % 7 + 1,
            _ => (code as i32 * 3 + d) % 5,
        }));
        let at = code / PAGE_POOLS * STRIDE + code % PAGE_POOLS * DIM * 2;
        pattern[at..][..DIM * 2].copy_from_slice(&bf16_bytes(&key));
    }
    pattern
        .chunks_exact(STRIDE)
        .cycle()
        .take(pages)
        .flatten()
        .copied()
        .collect()
}

/// `rows` rows' queries: even heads `[1, h / 2 % 3]`, odd heads `[-1, 1]`,
/// the rest of each head 0, so the dots are small integers of both signs.
pub fn queries(rows: usize) -> Vec<u16> {
    integers((0..rows * HEADS * DIM).map(|at| {
        let (head, d) = (at / DIM % HEADS, at % DIM);
        match (head % 2, d) {
            (0, 0) => 1,
            (0, 1) => (head / 2 % 3) as i32,
            (1, 0) => -1,
            (1, 1) => 1,
            _ => 0,
        }
    }))
}

/// The kernel's candidate key: the score's bits mapped so unsigned order is
/// IEEE order (-0 as +0), then the inverted pool (`candidate_key`).
pub fn candidate(score: f32, pool: usize) -> u64 {
    let bits = if score == 0.0 { 0 } else { score.to_bits() };
    let ordered = if score.is_nan() {
        0
    } else if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    };
    u64::from(ordered) << 32 | u64::from(!(pool as u32))
}

/// The pools row `row` sees.
pub fn visible(images: &Images, row: usize) -> LabResult<usize> {
    Ok((images.i32s("positions")?[row] as usize + 1) / GROUP)
}

/// Row `row`'s candidate keys of `pools`, in pool order, reading each table
/// ID (of tables `max_pages` wide) as `read` maps it. A score is `sum_h w_h
/// relu(dot_h * f32(128^-0.5))` with `w_h = bf16 * f32(32^-0.5)`, F32 FMA
/// from zero in head order; the dots are exact.
pub fn candidates(
    images: &Images,
    row: usize,
    pools: Range<usize>,
    max_pages: usize,
    read: fn(i32) -> i32,
) -> LabResult<Vec<u64>> {
    let scale = (1.0 / (DIM as f64).sqrt()) as f32;
    let head_scale = (1.0 / (HEADS as f64).sqrt()) as f32;
    let q = bf16_values(&images.bytes("q")?[row * HEADS * DIM * 2..][..HEADS * DIM * 2]);
    let weights = images.u16s("head_weights")?;
    let keys = images.bytes("pooled_keys")?;
    let tables = images.i32s("page_tables")?;
    let item = usize::try_from(images.i32s("row_batch")?[row]).map_err(|e| e.to_string())?;
    let w = weights[row * HEADS..][..HEADS]
        .iter()
        .map(|&b| bf16_to_f32(b) * head_scale)
        .collect::<Vec<_>>();
    let mut candidates = Vec::with_capacity(pools.len());
    for pool in pools {
        let page = usize::try_from(read(tables[item * max_pages + pool / PAGE_POOLS]))
            .map_err(|e| e.to_string())?;
        let key = bf16_values(
            keys.get(page * STRIDE + pool % PAGE_POOLS * DIM * 2..)
                .and_then(|bytes| bytes.get(..DIM * 2))
                .ok_or_else(|| format!("page {page} lies past the pool"))?,
        );
        let mut score = 0_f32;
        for (head, &w) in w.iter().enumerate() {
            let query = &q[head * DIM..][..DIM];
            let dot: f64 = query
                .iter()
                .zip(&key)
                .filter(|(q, _)| **q != 0.0)
                .map(|(q, k)| q * k)
                .sum();
            let x = dot as f32 * scale;
            score = w.mul_add(if x > 0.0 { x } else { 0.0 }, score);
        }
        candidates.push(candidate(score, pool));
    }
    Ok(candidates)
}

/// The rows of a selection case: each row's item and position, and the
/// items' page tables of `max_pages` pages.
pub struct Layout {
    pub items: Vec<i32>,
    pub positions: Vec<i32>,
    pub tables: Vec<i32>,
    pub max_pages: usize,
}

/// `case` with the selection's inputs, ID inputs and scalars for `layout`
/// over a pool of `pages` pages, at `splits` splits per row. The page
/// tables rerun with their pages moved to the top of the pool; a -1 page is
/// not a hole the selection skips, so they declare no hole row.
pub fn inputs<'a>(case: Case<'a>, layout: &Layout, pages: usize, splits: u32) -> Case<'a> {
    let rows = layout.items.len();
    let weights = integers((0..rows * HEADS).map(|at| (at / HEADS + at % HEADS) as i32 % 3 + 1));
    case.input(
        "q",
        DType::Bf16,
        (Rows::Live, HEADS * DIM),
        bf16_bytes(&queries(rows)),
    )
    .produced("q")
    .input(
        "head_weights",
        DType::Bf16,
        (Rows::Live, HEADS),
        bf16_bytes(&weights),
    )
    .input(
        "pooled_keys",
        DType::U8,
        (Rows::Fixed(pages), STRIDE),
        pooled_keys(pages),
    )
    .ids(
        "page_tables",
        (
            Rows::Fixed(layout.tables.len() / layout.max_pages),
            layout.max_pages,
        ),
        &layout.tables,
        &[("pooled_keys", STRIDE)],
        Holes::Never,
    )
    .input(
        "row_batch",
        DType::I32,
        (Rows::Live, 1),
        i32_bytes(&layout.items),
    )
    .input(
        "positions",
        DType::I32,
        (Rows::Live, 1),
        i32_bytes(&layout.positions),
    )
    .scalar("max_pages", Argument::U32(layout.max_pages as u32))
    .scalar("page_stride_bytes", Argument::U64(STRIDE as u64))
    .scalar("top_k", Argument::U32(TOP_K as u32))
    .scalar("splits", Argument::U32(splits))
    .scalar("cp_rank", Argument::U32(0))
    .scalar("cp_size", Argument::U32(1))
}
