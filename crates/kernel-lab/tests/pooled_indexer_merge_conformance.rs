//! The pooled indexer's selection split merge (`pooled_indexer_shapes.rs`)
//! through the declarative conformance runner (`kernel_lab::conformance`):
//! the global selection's split kernel chained with its merge kernel
//! ([`SplitMerge`]) and checked as one launch, at a bucket of eight rows,
//! which the planner gives 64 splits. Each row sees 130 to 144 pools, so
//! its first three splits score tiles of up to 64 pools and the rest write
//! zeros.
//!
//! The fixture is the one-split case's (`common/pooled_select.rs`), whose
//! candidate keys the host reproduces bit for bit, so the oracle gives
//! exact bits for both stages:
//!
//! - the split workspace (u64 `[row][split][max_kept]`), an output of the
//!   chain: each split's best `top_k / group` candidate keys of its
//!   tile-aligned pool range descending, then zeros;
//! - the key lists (i32 `[row][2051]`): the row's best pools ascending,
//!   each as its four token positions, then the always-selected tail of the
//!   incomplete pool, then -1. The lists hold -1, so they start from a fill
//!   other than all ones.
//!
//! The pooled keys repeat every 16 pages, and the tables map copies of the
//! page of the best pools into later splits ([`tables`]), so equal scores
//! tie across splits and ties go to the lower pool: rows 0 to 6 select
//! pools from splits 0 and 1, row 7 from splits 0, 1 and 2. A merge that
//! folds only split 0's list fails every run with a live row
//! ([`winners_span_splits_and_a_split_zero_merge_fails`]).
//!
//! The page tables rerun with their pages moved to the top of the pool: the
//! workspace witnesses it (the old IDs over the moved pool score other
//! pools). A -1 page is not a hole the selection skips, so the tables
//! declare no hole row.
//!
//! GPU: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 cargo test
//! -p kernel-lab --release --locked --offline --test
//! pooled_indexer_merge_conformance -- --test-threads=1 --nocapture`

#[path = "../pooled_indexer_shapes.rs"]
mod shapes;

#[path = "common/hot_keys.rs"]
mod hot_keys;

#[path = "common/pooled_select.rs"]
mod pooled_select;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use kernel_core::contract::DType;
use kernel_lab::conformance::{Case, Expected, Images, Rows, Write};
use kernel_lab::lab_entry::SplitMerge;
use kernel_lab::{LabResult, gpu_test, latent_shapes};
use kernels_sm120::pooled_indexer::Entry;
use pooled_select::{
    CODE_PAGES, GROUP, Layout, MAX_KEPT, TOP_K, candidates, entry, inputs, visible,
};

const CAPACITY: usize = 8;
/// Two items of four rows each, at positions whose tails hold 1, 2, 3 and
/// 0 tokens.
const ITEM_ROWS: usize = 4;
const POSITIONS: [i32; ITEM_ROWS] = [520, 533, 546, 575];
/// Pages of each item, which hold its rows' pools; both items use the
/// same pages in different orders.
const ITEM_PAGES: usize = 9;
const MAX_PAGES: usize = ITEM_PAGES + 1;
/// Pools a split's range is aligned to (`Select::tile`).
const TILE: usize = 64;
/// A key list: the top-k tokens and the incomplete pool's tail.
const LIST_WIDTH: usize = 2048 + GROUP - 1;
/// The lists' fill: no list entry has these bits.
const FILL: u8 = 0x5a;
/// Pages of the pool: the codes repeat every [`CODE_PAGES`] pages, so a
/// page's copies hold the same pooled keys.
const PAGES: usize = 3 * CODE_PAGES;
/// The copies of page 8 the tables add.
const COPIES: [i32; 2] = [8 + CODE_PAGES as i32, 8 + 2 * CODE_PAGES as i32];

/// The pools `[begin, end)` of split `split` of `splits` over `pools`
/// (`split_range`).
fn split_range(pools: usize, split: usize, splits: usize) -> (usize, usize) {
    let aligned = pools.div_ceil(splits).div_ceil(TILE) * TILE;
    let begin = pools.min(split * aligned);
    (begin, pools.min(begin + aligned))
}

/// Row `row`'s candidate keys of every pool it sees, refused when it keeps
/// them all unscored.
fn row_candidates(images: &Images, row: usize) -> LabResult<Vec<u64>> {
    let visible = visible(images, row)?;
    if visible <= TOP_K / GROUP {
        return Err(format!(
            "row {row} sees {visible} pools, which it keeps unscored"
        ));
    }
    candidates(images, row, 0..visible, MAX_PAGES, hot_keys::whole)
}

/// The pools of the best `top_k / group` of `keys` ascending; ties go to
/// the lower pool, whose inverted bits rank higher.
fn selected(keys: &[u64]) -> Vec<i32> {
    let mut best = keys.to_vec();
    best.sort_unstable_by(|a, b| b.cmp(a));
    let mut pools = best
        .iter()
        .take(TOP_K / GROUP)
        .map(|&key| !(key as u32) as i32)
        .collect::<Vec<_>>();
    pools.sort_unstable();
    pools
}

/// The key lists at `live`, with `pools(row)` selected in each live row:
/// each pool's four token positions, the incomplete pool's tail, then -1.
fn key_lists(
    images: &Images,
    live: u32,
    mut pools: impl FnMut(usize) -> LabResult<Vec<i32>>,
) -> LabResult<Vec<u8>> {
    let positions = images.i32s("positions")?;
    let mut lists = vec![0_i32; CAPACITY * LIST_WIDTH];
    for row in 0..live as usize {
        let position = positions[row] + 1;
        let base = position / GROUP as i32 * GROUP as i32;
        let list = pools(row)?
            .into_iter()
            .flat_map(|pool| (0..GROUP as i32).map(move |e| pool * GROUP as i32 + e))
            .chain(base..position)
            .chain(std::iter::repeat(-1))
            .take(LIST_WIDTH);
        for (at, entry) in lists[row * LIST_WIDTH..][..LIST_WIDTH].iter_mut().zip(list) {
            *at = entry;
        }
    }
    Ok(lists.iter().flat_map(|entry| entry.to_le_bytes()).collect())
}

/// The workspace and key lists at `live`.
fn oracle(images: &Images, live: u32, splits: usize) -> LabResult<Vec<(String, Expected)>> {
    let kept = TOP_K / GROUP;
    let mut workspace = vec![0_u64; CAPACITY * splits * MAX_KEPT];
    for row in 0..live as usize {
        let all = row_candidates(images, row)?;
        for split in 0..splits {
            let (begin, end) = split_range(all.len(), split, splits);
            let mut best = all[begin..end].to_vec();
            best.sort_unstable_by(|a, b| b.cmp(a));
            best.truncate(kept);
            workspace[(row * splits + split) * MAX_KEPT..][..best.len()].copy_from_slice(&best);
        }
    }
    let lists = key_lists(images, live, |row| {
        Ok(selected(&row_candidates(images, row)?))
    })?;
    Ok(vec![
        (
            "workspace".into(),
            Expected::Bits(workspace.iter().flat_map(|key| key.to_le_bytes()).collect()),
        ),
        ("output".into(), Expected::Bits(lists)),
    ])
}

/// Both items' page tables: pages low in the pool, shuffled two ways, which
/// both put page 8, of the best pools, at logical page 2 (pools 32 to 47,
/// in split 0). Item 0 holds a copy of it at logical page 5 (split 1),
/// item 1 copies at logical pages 7 (split 1) and 8 (split 2).
fn tables() -> Vec<i32> {
    let mut tables = vec![-1; 2 * MAX_PAGES];
    for page in 0..ITEM_PAGES {
        tables[page] = (page * 4 % ITEM_PAGES) as i32;
        tables[MAX_PAGES + page] = ((page * 7 + 3) % ITEM_PAGES) as i32;
    }
    tables[5] = COPIES[0];
    tables[MAX_PAGES + 7] = COPIES[0];
    tables[MAX_PAGES + 8] = COPIES[1];
    tables
}

/// The chain of `entry` and `merge`.
fn case<'a>(entry: &'a Entry, merge: &'a SplitMerge<'a, Entry>) -> Case<'a> {
    let splits = entry.splits(CAPACITY as u32) as usize;
    let layout = Layout {
        items: (0..CAPACITY).map(|row| (row / ITEM_ROWS) as i32).collect(),
        positions: (0..CAPACITY)
            .map(|row| POSITIONS[row % ITEM_ROWS])
            .collect(),
        tables: tables(),
        max_pages: MAX_PAGES,
    };
    inputs(
        Case::new(entry, CAPACITY as u32).then(merge),
        &layout,
        PAGES,
        splits as u32,
    )
    .output(
        "workspace",
        DType::I64,
        (Rows::Live, splits * MAX_KEPT),
        Write::LiveRows,
    )
    .output_with(
        "output",
        DType::I32,
        (Rows::Live, LIST_WIDTH),
        Write::LiveRows,
        &[FILL],
    )
    .oracle(move |images, live| oracle(images, live, splits))
}

#[test]
fn pooled_indexer_merge_case_validates() {
    let entry = entry();
    let merge = SplitMerge(&entry);
    assert_eq!(entry.splits(CAPACITY as u32), 64);
    case(&entry, &merge)
        .validate()
        .expect("selection merge chain");
}

/// Every row's pools fill two tiles and part of a third, so three splits
/// score and the merge folds their lists.
#[test]
fn rows_score_in_three_splits() {
    for position in POSITIONS {
        let pools = (position as usize + 1) / GROUP;
        let scoring = (0..64)
            .filter(|&split| {
                let (begin, end) = split_range(pools, split, 64);
                begin < end
            })
            .count();
        assert_eq!(scoring, 3, "position {position}");
        assert!(pools <= ITEM_PAGES * 16);
    }
}

/// Both items hold page 8 at logical page 2, and its copies where
/// [`tables`] says.
#[test]
fn the_tables_copy_the_best_page_into_later_splits() {
    let tables = tables();
    assert_eq!((tables[2], tables[MAX_PAGES + 2]), (8, 8));
    for copy in COPIES {
        assert_eq!(copy as usize % CODE_PAGES, 8);
        assert!((copy as usize) < PAGES);
    }
    let splits_of = |item: usize| {
        (0..ITEM_PAGES)
            .filter(|&page| tables[item * MAX_PAGES + page] as usize % CODE_PAGES == 8)
            .map(|page| page * 16 / TILE)
            .collect::<Vec<_>>()
    };
    assert_eq!(splits_of(0), [0, 1]);
    assert_eq!(splits_of(1), [0, 1, 2]);
}

/// Every live row's selected pools lie in two or more splits, three in
/// some row, and a merge that folds only split 0's list fails every run
/// with a live row.
#[test]
fn winners_span_splits_and_a_split_zero_merge_fails() {
    let entry = entry();
    let merge = SplitMerge(&entry);
    let splits = entry.splits(CAPACITY as u32) as usize;
    let case = case(&entry, &merge);
    let spans = RefCell::new(Vec::new());
    let refusing = case
        .refusing_runs(|images, live| {
            for row in 0..live as usize {
                let all = row_candidates(images, row)?;
                let owners = selected(&all)
                    .iter()
                    .map(|&pool| {
                        (0..splits)
                            .find(|&split| {
                                let (begin, end) = split_range(all.len(), split, splits);
                                (begin..end).contains(&(pool as usize))
                            })
                            .expect("a split owns every pool")
                    })
                    .collect::<BTreeSet<_>>();
                spans.borrow_mut().push((live, row, owners));
            }
            let lists = key_lists(images, live, |row| {
                let all = row_candidates(images, row)?;
                let (begin, end) = split_range(all.len(), 0, splits);
                Ok(selected(&all[begin..end]))
            })?;
            Ok(vec![("output".into(), lists)])
        })
        .expect("a split-zero merge");
    let spans = spans.into_inner();
    for (live, row, owners) in &spans {
        assert!(owners.len() >= 2, "live {live} row {row}: {owners:?}");
    }
    assert!(spans.iter().any(|(.., owners)| owners.len() == 3));
    assert_eq!(
        refusing,
        ["live 8", "live 1", "live 4", "live 7", "high page_tables"]
    );
}

#[test]
fn pooled_indexer_merge_conforms_gpu() {
    gpu_test(1 << 30, |session| {
        let module = session.module(shapes::CUBIN)?;
        let lab = session.module(&latent_shapes::cubin_name())?;
        let entry = entry();
        let merge = SplitMerge(&entry);
        let evidence = case(&entry, &merge).run(session, &module, Some(&lab))?;
        session.receipt(
            "pooled_indexer_merge_conformance",
            BTreeMap::from([("evidence".into(), evidence)]),
        )?;
        Ok(())
    })
    .expect("pooled indexer selection merge conforms");
}
