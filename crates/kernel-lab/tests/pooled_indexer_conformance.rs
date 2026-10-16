//! The pooled indexer's selection (`pooled_indexer_shapes.rs`) through the
//! declarative conformance runner (`kernel_lab::conformance`): the global
//! selection's split kernel over shuffled pages of pooled keys in a larger
//! pool, with the page tables rerun on their pages moved to the top of the
//! pool. The selection writes each split's best candidate keys and a merge
//! kernel folds the splits into key lists; a case of one entry cannot chain
//! the two, so the case takes a bucket of 376 rows, which the planner gives
//! one split, and checks the split kernel's workspace. Its candidate keys
//! (score bits, then the inverted pool) are exact: integer operands give
//! exact dots, and the host reproduces the kernel's F32 score arithmetic
//! (as `tests/pooled_indexer.rs` does), so the workspace witnesses the
//! high-id rerun: the moved pages give the first run's bits, the old IDs
//! over the moved pool other pools' scores. A -1 page is not a hole the
//! selection skips, so the tables declare no hole row. The case reruns over
//! a pool of [`hot_keys::WIDE_PAGES`] too, whose rerun moves its pages to
//! IDs past 16 bits; the oracle shows that the IDs narrowed to 16 bits
//! change its workspace (`narrowing_checked`). The fixture is
//! `common/pooled_select.rs`; `pooled_indexer_merge_conformance.rs` chains
//! the split kernel with its merge at a bucket of several splits.
//!
//! GPU: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 cargo test
//! -p kernel-lab --release --locked --offline --test
//! pooled_indexer_conformance -- --test-threads=1 --nocapture`

#[path = "../pooled_indexer_shapes.rs"]
mod shapes;

#[path = "common/hot_keys.rs"]
mod hot_keys;

#[path = "common/pooled_select.rs"]
mod pooled_select;

use std::collections::BTreeMap;
use std::sync::atomic::Ordering::Relaxed;

use hot_keys::{HIGHEST, NARROWED, WIDE_PAGES, narrowing_checked};
use kernel_core::contract::DType;
use kernel_lab::conformance::{Case, Expected, Images, Rows, Write};
use kernel_lab::{LabResult, gpu_test, latent_shapes};
use kernels_sm120::pooled_indexer::Entry;
use pooled_select::{
    GROUP, Layout, MAX_KEPT, POOL_PAGES, STRIDE, TOP_K, candidates, entry, inputs, visible,
};
use serde_json::json;

/// The smallest bucket the planner gives one split.
const CAPACITY: usize = 376;
/// Rows of each item, at consecutive positions from `FIRST`.
const ITEM_ROWS: usize = 94;
const FIRST: usize = 98;
const ITEMS: usize = CAPACITY / ITEM_ROWS;
/// Pages of each item, which hold its rows' pools.
const ITEM_PAGES: usize = 3;
const MAX_PAGES: usize = ITEM_PAGES + 1;

/// The split workspace at `live`, reading each table ID as `read` maps it:
/// each row's best `TOP_K / GROUP` candidate keys descending, then zeros.
fn workspace(
    images: &Images,
    live: u32,
    read: fn(i32) -> i32,
) -> LabResult<Vec<(String, Expected)>> {
    let kept = TOP_K / GROUP;
    let mut out = vec![0_u64; CAPACITY * MAX_KEPT];
    for row in 0..live as usize {
        let visible = visible(images, row)?;
        if visible <= kept {
            return Err(format!(
                "row {row} sees {visible} pools, which it keeps unscored"
            ));
        }
        let mut candidates = candidates(images, row, 0..visible, MAX_PAGES, read)?;
        candidates.sort_unstable_by(|a, b| b.cmp(a));
        out[row * MAX_KEPT..][..kept].copy_from_slice(&candidates[..kept]);
    }
    let bytes = out.iter().flat_map(|key| key.to_le_bytes()).collect();
    Ok(vec![("workspace".into(), Expected::Bits(bytes))])
}

/// The case over a pool of `pages` pages.
fn case(entry: &Entry, pages: usize) -> Case<'_> {
    assert_eq!(
        entry.splits(CAPACITY as u32),
        1,
        "the bucket takes one split"
    );
    // Every item's pages, all low in the pool and shuffled.
    let mut tables = vec![-1; ITEMS * MAX_PAGES];
    for item in 0..ITEMS {
        for page in 0..ITEM_PAGES {
            tables[item * MAX_PAGES + page] = ((item * ITEM_PAGES + page) * 5 % 12) as i32;
        }
    }
    let layout = Layout {
        items: (0..CAPACITY).map(|row| (row / ITEM_ROWS) as i32).collect(),
        positions: (0..CAPACITY)
            .map(|row| (FIRST + row % ITEM_ROWS) as i32)
            .collect(),
        tables,
        max_pages: MAX_PAGES,
    };
    inputs(Case::new(entry, CAPACITY as u32), &layout, pages, 1)
        .output(
            "workspace",
            DType::I64,
            (Rows::Live, MAX_KEPT),
            Write::LiveRows,
        )
        .oracle(|images, live| narrowing_checked(images, live, "page_tables", workspace))
}

#[test]
fn pooled_indexer_selection_case_validates() {
    let entry = entry();
    case(&entry, POOL_PAGES)
        .validate()
        .expect("pooled indexer selection case");
}

/// The wide case's high-id rerun moves its 12 pages to the top of the pool,
/// IDs 65624 to 65635, and the IDs narrowed to 16 bits change its
/// workspace.
#[test]
fn wide_pooled_indexer_selection_case_validates() {
    let entry = entry();
    case(&entry, WIDE_PAGES)
        .validate()
        .expect("wide pooled indexer selection case");
    assert!(
        NARROWED.load(Relaxed) > 0,
        "no ID past 16 bits was narrowed"
    );
    assert_eq!(HIGHEST.load(Relaxed), WIDE_PAGES as i32 - 1);
}

#[test]
fn pooled_indexer_selection_conforms_gpu() {
    // The wide case's pool, and 1 GiB for the rest.
    let need = WIDE_PAGES * STRIDE + (1 << 30);
    gpu_test(need as u64, |session| {
        let module = session.module(shapes::CUBIN)?;
        let lab = session.module(&latent_shapes::cubin_name())?;
        let entry = entry();
        for pages in [POOL_PAGES, WIDE_PAGES] {
            let evidence = case(&entry, pages)
                .run(session, &module, Some(&lab))
                .map_err(|e| format!("{pages} pages: {e}"))?;
            session.receipt(
                "pooled_indexer_conformance",
                BTreeMap::from([
                    ("pool_pages".into(), json!(pages)),
                    ("evidence".into(), evidence),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("pooled indexer selection conforms");
}
