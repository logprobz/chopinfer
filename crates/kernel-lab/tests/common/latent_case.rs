//! The latent attention fixture shared by the one-split prefill cases
//! (`latent_attention_conformance.rs`) and the split merge chains
//! (`latent_split_merge_conformance.rs`): four rows of two items over
//! shuffled pages of a larger pool, with one-hot keys
//! (`common/hot_keys.rs`).
#![allow(dead_code)] // Shared by test binaries that use different parts.

use crate::hot_keys::{
    Partials, bf16_bytes, bf16_values, f32_bytes, i32_bytes, key_row, partials, pool, query_row,
};
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, DType};
use kernel_lab::LabResult;
use kernel_lab::conformance::{Case, Expected, Holes, Images, Rows};
use kernel_lab::latent_shapes::{GEOMETRY, PROFILE, catalog};
use kernels_sm120::latent::Entry;

pub const HEADS: usize = GEOMETRY.heads as usize;
pub const LATENT: usize = GEOMETRY.latent as usize;
pub const PAGE: usize = GEOMETRY.page_size as usize;
pub const MAX_KEYS: usize = GEOMETRY.max_keys as usize;
/// Pages of the pool, more than the items use, so every page can move up.
pub const POOL_PAGES: usize = 8;
/// A page and 1024 decoy bytes.
pub const STRIDE: usize = PAGE * LATENT * 2 + 1024;
/// Each item's tokens and pages, all low in the pool.
pub const LENGTHS: [usize; 2] = [150, 40];
pub const TABLES: [&[i32]; 2] = [&[2, 0, 3], &[1]];
pub const MAX_PAGES: usize = 4;
pub const CAPACITY: usize = 4;

/// Each row's item and key list: every token of the first item; every
/// third one with holes between; the second item's tokens in reverse; and
/// two runs of the first item's.
pub fn rows() -> Vec<(usize, Vec<i32>)> {
    let tokens = |range: std::ops::Range<i32>| range.collect::<Vec<_>>();
    vec![
        (0, tokens(0..150)),
        (
            0,
            (0..50)
                .flat_map(|i| [i * 3, -1].into_iter().take(if i % 5 == 0 { 2 } else { 1 }))
                .collect(),
        ),
        (1, tokens(0..40).into_iter().rev().collect()),
        (0, tokens(100..150).into_iter().chain(0..10).collect()),
    ]
}

/// The rows `rows` repeated in order to `capacity` rows.
pub fn repeated(rows: &[(usize, Vec<i32>)], capacity: usize) -> Vec<(usize, Vec<i32>)> {
    rows.iter().cycle().take(capacity).cloned().collect()
}

/// The capacity of the case whose images are `images`: one item per row.
pub fn capacity(images: &Images) -> LabResult<usize> {
    Ok(images.i32s("row_batch")?.len())
}

/// The BF16 attention entry `attn_bf16_{kernel}`, such as `prefill` or
/// `decode_partial`.
pub fn entry(kernel: &str) -> Entry {
    entry_of("bf16", kernel)
}

/// The attention entry `attn_{kv}_{kernel}`, `kv` being `bf16` or `fp8`.
pub fn entry_of(kv: &str, kernel: &str) -> Entry {
    let name = format!("{}_attn_{kv}_{kernel}", PROFILE.tag());
    catalog()
        .into_iter()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("no lab entry {name}"))
}

/// `case` with the fixture's inputs, ID inputs and scalars over a pool of
/// `pages` pages, at `splits` splits per row. The block tables rerun with
/// their pages moved to the top of the pool and the last live key list
/// reruns all holes.
pub fn inputs(case: Case<'_>, splits: u32, pages: usize) -> Case<'_> {
    inputs_of(case, splits, pages, key_row, &rows())
}

/// [`inputs`] with each token's latent row `latent_row(code, width)` and
/// the rows' items and key lists `rows`, one per row of the case's
/// capacity.
pub fn inputs_of<'a>(
    case: Case<'a>,
    splits: u32,
    pages: usize,
    latent_row: fn(usize, usize) -> Vec<u16>,
    rows: &[(usize, Vec<i32>)],
) -> Case<'a> {
    let image = pool(pages, PAGE, LATENT, STRIDE, latent_row);
    inputs_over(case, splits, pages, image, rows, query_row)
}

/// [`inputs_of`] over the pool image `image` of `pages` pages at
/// [`STRIDE`], with each query head's row `query_row(width, negative)`.
pub fn inputs_over<'a>(
    case: Case<'a>,
    splits: u32,
    pages: usize,
    image: Vec<u8>,
    rows: &[(usize, Vec<i32>)],
    query_row: fn(usize, bool) -> Vec<u16>,
) -> Case<'a> {
    let mut tables = vec![-1; LENGTHS.len() * MAX_PAGES];
    for (item, pages) in TABLES.iter().enumerate() {
        assert!(pages.len() * PAGE >= LENGTHS[item]);
        tables[item * MAX_PAGES..][..pages.len()].copy_from_slice(pages);
    }
    let mut lists = vec![-1; rows.len() * MAX_KEYS];
    for (row, (_, list)) in rows.iter().enumerate() {
        lists[row * MAX_KEYS..][..list.len()].copy_from_slice(list);
    }
    let items = rows
        .iter()
        .map(|&(item, _)| item as i32)
        .collect::<Vec<_>>();
    // Heads alternate signs, and the signs alternate by row, so heads pick
    // both the largest and the smallest code they see.
    let q = (0..rows.len())
        .flat_map(|row| (0..HEADS).flat_map(move |head| query_row(LATENT, (row + head) % 2 == 1)))
        .collect::<Vec<_>>();
    case.input(
        "q",
        DType::Bf16,
        (Rows::Live, HEADS * LATENT),
        bf16_bytes(&q),
    )
    .produced("q")
    .input("pool", DType::U8, (Rows::Fixed(pages), STRIDE), image)
    .ids(
        "block_tables",
        (Rows::Fixed(LENGTHS.len()), MAX_PAGES),
        &tables,
        &[("pool", STRIDE)],
        Holes::Never,
    )
    .input("row_batch", DType::I32, (Rows::Live, 1), i32_bytes(&items))
    .ids(
        "key_lists",
        (Rows::Live, MAX_KEYS),
        &lists,
        &[],
        Holes::LastLive,
    )
    .scalar("max_pages", Argument::U32(MAX_PAGES as u32))
    .scalar("page_stride_bytes", Argument::U64(STRIDE as u64))
    .scalar("budget", Argument::U32(MAX_KEYS as u32))
    .scalar("splits", Argument::U32(splits))
    .scalar(
        "scale_bits",
        Argument::U32(GEOMETRY.softmax_scale.to_bits()),
    )
}

/// A fixture's exact partials of a row over some entries of its key list
/// ([`row_partials`]).
pub type RowPartials =
    fn(&Images, usize, std::ops::Range<usize>, fn(i32) -> i32) -> LabResult<Partials>;

/// The exact partials of row `row` over the entries `entries` of its key
/// list, reading each table ID as `read` maps it. Holes are not keys.
pub fn row_partials(
    images: &Images,
    row: usize,
    entries: std::ops::Range<usize>,
    read: fn(i32) -> i32,
) -> LabResult<Partials> {
    let q = bf16_values(images.bytes("q")?);
    let pool = images.bytes("pool")?;
    let tables = images.i32s("block_tables")?;
    let item = usize::try_from(images.i32s("row_batch")?[row]).map_err(|e| e.to_string())?;
    let keys = images.i32s("key_lists")?[row * MAX_KEYS..][entries]
        .iter()
        .filter_map(|&token| usize::try_from(token).ok())
        .map(|token| (read(tables[item * MAX_PAGES + token / PAGE]), token % PAGE))
        .collect::<Vec<_>>();
    let row_q = &q[row * HEADS * LATENT..][..HEADS * LATENT];
    partials(
        row_q,
        HEADS,
        LATENT,
        (pool, pool),
        STRIDE,
        &keys,
        GEOMETRY.softmax_scale,
    )
}

/// The exact outputs at `live` of an entry over every key, reading each
/// table ID as `read` maps it: BF16 `o / l`, or the F32 partials when
/// `partial`.
pub fn outputs(
    images: &Images,
    live: u32,
    partial: bool,
    read: fn(i32) -> i32,
) -> LabResult<Vec<(String, Expected)>> {
    outputs_of(images, live, partial, read, row_partials)
}

/// [`outputs`] from each row's partials `row_partials(images, row,
/// entries, read)`.
pub fn outputs_of(
    images: &Images,
    live: u32,
    partial: bool,
    read: fn(i32) -> i32,
    row_partials: RowPartials,
) -> LabResult<Vec<(String, Expected)>> {
    let capacity = capacity(images)?;
    let mut max = vec![0.0; capacity * HEADS];
    let mut den = vec![0.0; capacity * HEADS];
    let mut num = vec![0.0; capacity * HEADS * LATENT];
    for row in 0..live as usize {
        // Holes and entries past the budget are not keys.
        let partial = row_partials(images, row, 0..MAX_KEYS, read)?;
        max[row * HEADS..][..HEADS].copy_from_slice(&partial.max);
        den[row * HEADS..][..HEADS].copy_from_slice(&partial.den);
        num[row * HEADS * LATENT..][..HEADS * LATENT].copy_from_slice(&partial.num);
    }
    if partial {
        return Ok(vec![
            ("out_max".into(), Expected::Bits(f32_bytes(&max))),
            ("out_den".into(), Expected::Bits(f32_bytes(&den))),
            ("out_num".into(), Expected::Bits(f32_bytes(&num))),
        ]);
    }
    // The denominator is 1, or 0 for an empty list, which writes zeros.
    let out = num
        .iter()
        .enumerate()
        .map(|(at, &o)| {
            let l = den[at / LATENT];
            f32_to_bf16(if l == 0.0 { 0.0 } else { o / l })
        })
        .collect::<Vec<_>>();
    Ok(vec![("out".into(), Expected::Bits(bf16_bytes(&out)))])
}
