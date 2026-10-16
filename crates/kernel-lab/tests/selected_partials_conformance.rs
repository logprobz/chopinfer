//! Selected attention partials (`selected_partials_shapes.rs`) through the
//! declarative conformance runner (`kernel_lab::conformance`): BF16 verify
//! rows past 512 complete blocks, whose selected lists drop some blocks,
//! and a short decode row that attends every token, over shuffled pages of
//! a larger pool, with the block tables rerun on their pages moved to the
//! top of the pool. The keys are one-hot (`common/hot_keys.rs`), so the
//! oracle gives every partial's exact bits, and the maxima and numerators
//! witness the high-id rerun: the moved pages give the first run's bits,
//! the old IDs over the moved pool other keys' scores and values. A -1
//! page traps, so the tables declare no hole row. The case reruns over a
//! pool of [`hot_keys::WIDE_PAGES`] too, whose rerun moves its pages to IDs
//! past 16 bits and byte offsets past 2^31; the oracle shows that the IDs
//! narrowed to 16 bits change its outputs (`narrowing_checked`).
//!
//! GPU: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 cargo test
//! -p kernel-lab --release --locked --offline --test
//! selected_partials_conformance -- --test-threads=1 --nocapture`

#[path = "../selected_partials_shapes.rs"]
mod shapes;

#[path = "common/hot_keys.rs"]
mod hot_keys;

use std::collections::BTreeMap;
use std::sync::atomic::Ordering::Relaxed;

use hot_keys::{
    HIGHEST, NARROWED, PagedShape, WIDE_PAGES, bf16_bytes, i32_bytes, key_row, narrowing_checked,
    partial_outputs, pool, query_row, value_row,
};
use kernel_core::contract::{Argument, DType};
use kernel_lab::conformance::{Case, Holes, Rows, Write};
use kernel_lab::{LabResult, gpu_test, latent_shapes};
use kernels_sm120::selected_partials::Entry;
use serde_json::json;

const HEADS: usize = (shapes::GEOMETRY.q_heads * shapes::GEOMETRY.kv_heads) as usize;
const DIM: usize = shapes::GEOMETRY.head_dim as usize;
const PAGE: usize = shapes::GEOMETRY.page as usize;
const BLOCK: usize = shapes::GEOMETRY.block as usize;
const TOP: usize = shapes::GEOMETRY.top_blocks as usize;
/// Pages of the pool, more than the items use, so every page can move up.
const POOL_PAGES: usize = 40;
/// A page and 512 decoy bytes.
const STRIDE: usize = PAGE * DIM * 2 + 512;
const SHAPE: PagedShape = PagedShape {
    heads: HEADS,
    width: DIM,
    page: PAGE,
    stride: STRIDE,
    scale: 0.0625,
};

/// The verify entry's rows: the last four tokens of a 2112-token item (527
/// and 528 complete blocks), then the last token of a 100-token item.
const ROWS: [(usize, usize); 5] = [(0, 2108), (0, 2109), (0, 2110), (0, 2111), (1, 99)];
const LENGTHS: [usize; 2] = [2112, 100];

fn entry() -> Entry {
    let name = format!("{}_bf16_verify", shapes::PROFILE.tag());
    shapes::lab_entries()
        .into_iter()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("no lab entry {name}"))
}

/// Row `position`'s selected list of `TOP` blocks: every complete block
/// when there are at most `TOP`, else all but blocks 3, 34, 65, ... (as
/// many as exceed `TOP`), ascending; -1 past them.
fn selection(position: usize) -> Vec<i32> {
    let complete = (position + 1) / BLOCK;
    let dropped = complete.saturating_sub(TOP);
    let mut list = (0..complete)
        .filter(|&block| block % 31 != 3 || block / 31 >= dropped)
        .map(|block| block as i32)
        .collect::<Vec<_>>();
    list.resize(TOP, -1);
    list
}

/// The positions a row attends: its selected blocks' tokens, then the
/// incomplete tail.
fn kept(selected: &[i32], position: usize) -> LabResult<Vec<usize>> {
    let complete = (position + 1) / BLOCK;
    let mut kept = Vec::new();
    for &block in &selected[..complete.min(TOP)] {
        let block = usize::try_from(block).map_err(|e| e.to_string())?;
        if block >= complete {
            return Err(format!(
                "block {block} is not complete at position {position}"
            ));
        }
        kept.extend(block * BLOCK..(block + 1) * BLOCK);
    }
    kept.extend(complete * BLOCK..=position);
    Ok(kept)
}

/// The case over K and V pools of `pages` pages.
fn case(entry: &Entry, pages: usize) -> Case<'_> {
    let capacity = ROWS.len();
    // Every item's pages, all low in the pool and shuffled.
    let used = LENGTHS.iter().map(|n| n.div_ceil(PAGE)).sum::<usize>();
    let max_pages = LENGTHS.iter().map(|n| n.div_ceil(PAGE)).max().unwrap_or(0) + 1;
    let mut tables = vec![-1; LENGTHS.len() * max_pages];
    let mut next = 0;
    for (item, length) in LENGTHS.iter().enumerate() {
        for page in 0..length.div_ceil(PAGE) {
            tables[item * max_pages + page] = (next * 11 % used) as i32;
            next += 1;
        }
    }
    // Heads alternate signs, and the signs alternate by row, so heads pick
    // both the largest and the smallest code they see.
    let q = (0..capacity)
        .flat_map(|row| (0..HEADS).flat_map(move |head| query_row(DIM, (row + head) % 2 == 1)))
        .collect::<Vec<_>>();
    let items = ROWS
        .iter()
        .map(|&(item, _)| item as i32)
        .collect::<Vec<_>>();
    let positions = ROWS.iter().map(|&(_, p)| p as i32).collect::<Vec<_>>();
    let selected = ROWS
        .iter()
        .flat_map(|&(_, position)| selection(position))
        .collect::<Vec<_>>();
    let stride = STRIDE as u64;
    Case::new(entry, capacity as u32)
        .input("q", DType::Bf16, (Rows::Live, HEADS * DIM), bf16_bytes(&q))
        .produced("q")
        .input(
            "k_cache",
            DType::U8,
            (Rows::Fixed(pages), STRIDE),
            pool(pages, PAGE, DIM, STRIDE, key_row),
        )
        .input(
            "v_cache",
            DType::U8,
            (Rows::Fixed(pages), STRIDE),
            pool(pages, PAGE, DIM, STRIDE, value_row),
        )
        .ids(
            "block_tables",
            (Rows::Fixed(LENGTHS.len()), max_pages),
            &tables,
            &[("k_cache", STRIDE), ("v_cache", STRIDE)],
            Holes::Never,
        )
        .input("row_batch", DType::I32, (Rows::Live, 1), i32_bytes(&items))
        .input(
            "positions",
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(&positions),
        )
        .input(
            "selected",
            DType::I32,
            (Rows::Live, TOP),
            i32_bytes(&selected),
        )
        .output("out_max", DType::F32, (Rows::Live, HEADS), Write::LiveRows)
        .output("out_den", DType::F32, (Rows::Live, HEADS), Write::LiveRows)
        .output(
            "out_num",
            DType::F32,
            (Rows::Live, HEADS * DIM),
            Write::LiveRows,
        )
        .scalar("max_pages", Argument::U32(max_pages as u32))
        .scalar("k_page_stride_bytes", Argument::U64(stride))
        .scalar("v_page_stride_bytes", Argument::U64(stride))
        .scalar("scale_bits", Argument::U32(SHAPE.scale.to_bits()))
        .scalar("k_scale_bits", Argument::U32(1.0_f32.to_bits()))
        .scalar("v_scale_bits", Argument::U32(1.0_f32.to_bits()))
        .scalar("cp_rank", Argument::U32(0))
        .scalar("cp_size", Argument::U32(1))
        .oracle(move |images, live| {
            narrowing_checked(images, live, "block_tables", |images, live, read| {
                partial_outputs(
                    images,
                    live,
                    capacity,
                    SHAPE,
                    max_pages,
                    read,
                    |images, row, position| {
                        let selected = images.i32s("selected")?;
                        kept(&selected[row * TOP..][..TOP], position)
                    },
                )
            })
        })
}

#[test]
fn selected_partial_case_validates() {
    let entry = entry();
    case(&entry, POOL_PAGES)
        .validate()
        .expect("selected partials case");
}

/// The wide case's high-id rerun moves its 35 pages to the top of the pool,
/// IDs 65601 to 65635, and the IDs narrowed to 16 bits change its outputs.
#[test]
fn wide_selected_partial_case_validates() {
    let entry = entry();
    case(&entry, WIDE_PAGES)
        .validate()
        .expect("wide selected partials case");
    assert!(
        NARROWED.load(Relaxed) > 0,
        "no ID past 16 bits was narrowed"
    );
    assert_eq!(HIGHEST.load(Relaxed), WIDE_PAGES as i32 - 1);
}

#[test]
fn selected_partials_conform_gpu() {
    // The wide case's K and V pools, and 1 GiB for the rest.
    let need = 2 * WIDE_PAGES * STRIDE + (1 << 30);
    gpu_test(need as u64, |session| {
        let module = session.module(shapes::CUBIN)?;
        let lab = session.module(&latent_shapes::cubin_name())?;
        let entry = entry();
        for pages in [POOL_PAGES, WIDE_PAGES] {
            let evidence = case(&entry, pages)
                .run(session, &module, Some(&lab))
                .map_err(|e| format!("{pages} pages: {e}"))?;
            session.receipt(
                "selected_partials_conformance",
                BTreeMap::from([
                    ("pool_pages".into(), json!(pages)),
                    ("evidence".into(), evidence),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("selected partials conform");
}
