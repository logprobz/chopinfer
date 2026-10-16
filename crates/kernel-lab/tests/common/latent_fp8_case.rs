//! The latent attention fixture (`common/latent_case.rs`) over an FP8
//! E4M3 block-128 pool (`LatentKvFormat::Fp8E4M3Block128`) with FP8
//! one-hot keys (`common/hot_fp8.rs`): the same rows, key lists, tables and
//! stride. At [`STRIDE`] a page's 33 KiB of codes and scales leave 32 KiB
//! of decoys, so the wide rerun's byte offsets pass 2^32 as the BF16
//! cases' do.
#![allow(dead_code)] // Shared by test binaries that use different parts.

use crate::hot_fp8::{latent_partials, latent_pool, query_row};
use crate::hot_keys::{Partials, bf16_values};
use crate::latent_case::{
    HEADS, LATENT, MAX_KEYS, MAX_PAGES, PAGE, STRIDE, inputs_over, outputs_of, rows,
};
use kernel_lab::LabResult;
use kernel_lab::conformance::{Case, Expected, Images};
use kernel_lab::latent_shapes::GEOMETRY;

/// `latent_case::inputs` over an FP8 pool of `pages` pages.
pub fn inputs(case: Case<'_>, splits: u32, pages: usize) -> Case<'_> {
    inputs_of(case, splits, pages, &rows())
}

/// [`inputs`] with the rows' items and key lists `rows`, one per row of
/// the case's capacity.
pub fn inputs_of<'a>(
    case: Case<'a>,
    splits: u32,
    pages: usize,
    rows: &[(usize, Vec<i32>)],
) -> Case<'a> {
    let image = latent_pool(pages, PAGE, LATENT, STRIDE);
    inputs_over(case, splits, pages, image, rows, query_row)
}

/// `latent_case::row_partials` over the FP8 pool.
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
    latent_partials(
        &q[row * HEADS * LATENT..][..HEADS * LATENT],
        HEADS,
        LATENT,
        pool,
        STRIDE,
        PAGE,
        &keys,
        GEOMETRY.softmax_scale,
    )
}

/// `latent_case::outputs` over the FP8 pool.
pub fn outputs(
    images: &Images,
    live: u32,
    partial: bool,
    read: fn(i32) -> i32,
) -> LabResult<Vec<(String, Expected)>> {
    outputs_of(images, live, partial, read, row_partials)
}
