//! The wide row-local folds' packed entries against their row-major
//! entries, bit for bit (`kernel_harness::packed_fold`), on the launches
//! their prepared ops return, with a negative control that moves one strip
//! of the packed codes and paired timings at WQ_b's decode fold.

#[path = "packed_staged/support.rs"]
mod support;

use kernel_core::backend::{Counts, Launch, Need, View};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
use kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout;
use kernel_harness::LabResult;
use kernel_harness::packed_fold::{self, Image};
use kernel_lab_families::mxfp8_fold_local_wide_packed as family;
use kernels_sm120::gemm::mxfp8_fold_local_wide::{self as wide, Entry};
use kernels_sm120::gemm::mxfp8_fold_packed;

const CUBINS: &str = "CHOPINFER_MXFP8_FOLD_LOCAL_WIDE_PACKED_CUBINS";
const OUTPUT: u64 = 5;

fn layout(entry: &Entry) -> Mxfp8FoldLayout {
    let q = entry.base.query;
    Mxfp8FoldLayout {
        groups: q.groups as usize,
        n: q.n as usize,
        k: q.k as usize,
    }
}

fn buffers(entry: &Entry, rows: u32) -> Mxfp8FoldGemmBuffers<View> {
    let size = entry.base.bytes(rows);
    let v = |handle, bytes| View { handle, bytes };
    Mxfp8FoldGemmBuffers {
        activations: v(1, size.activations),
        activation_scales: v(2, size.activation_scales),
        weights: v(3, size.weights),
        weight_scales: v(4, size.weight_scales),
        output: v(OUTPUT, size.output),
        live_rows: v(6, 4),
        status: v(7, size.status),
        status_row_offset: 0,
    }
}

/// The launches the entry's prepared op returns at `rows` rows.
fn launches(entry: &Entry, rows: u32) -> Vec<Launch> {
    entry
        .prepare(Need::rows(entry.base.query.m_capacity))
        .unwrap()
        .launches(&buffers(entry, rows), View::EMPTY, Counts::rows(rows))
        .unwrap()
}

fn packed(entry: &Entry, launches: Vec<Launch>) -> Vec<Launch> {
    let symbol = entry.symbol();
    mxfp8_fold_packed::launches(launches, |s| s == symbol)
}

fn row_counts(entry: &Entry) -> [u32; 3] {
    let capacity = entry.base.query.m_capacity;
    [1, capacity / 2 + 1, capacity]
}

/// Every view the launches name. Row 1 has a nonzero status, so the fold
/// fails it.
fn images(entry: &Entry, rows: u32, seed: u64) -> LabResult<Vec<Image>> {
    let b = buffers(entry, rows);
    let mut status = vec![0; b.status.bytes as usize];
    if rows > 1 {
        status[4] = 4;
    }
    Ok(vec![
        Image::data(
            b.activations.handle,
            packed_fold::codes(b.activations.bytes as usize, seed),
        ),
        Image::data(
            b.activation_scales.handle,
            packed_fold::scales(b.activation_scales.bytes as usize, seed + 1),
        ),
        Image::codes(b.weights.handle, layout(entry), seed + 2)?,
        Image::data(
            b.weight_scales.handle,
            packed_fold::scales(b.weight_scales.bytes as usize, seed + 3),
        ),
        Image::data(b.output.handle, vec![0xa5; b.output.bytes as usize]),
        Image::data(b.live_rows.handle, rows.to_le_bytes().to_vec()),
        Image::data(b.status.handle, status),
    ])
}

#[test]
fn packed_lab_entries_differ_from_row_major_only_in_weight_addressing() {
    let entries = wide::packed_lab_entries();
    let spec = family::cubins().pop().unwrap();
    assert_eq!(spec.entries.len(), 2 * entries.len());
    for entry in &entries {
        let row_major = entry.spec();
        let packed_spec = entry.packed_spec().unwrap();
        support::assert_packed_twin(
            &row_major,
            &packed_spec,
            &row_major
                .body
                .replace(">(activations", ",true>(activations"),
        );
        assert!(spec.entries.iter().any(|e| e.symbol == packed_spec.symbol));
        assert!(spec.entries.iter().any(|e| e.symbol == row_major.symbol));
        layout(entry).packed_bytes().unwrap();
        // Renaming keeps the prepared op's plan and arguments.
        let rows = entry.base.query.m_capacity;
        let row_major = launches(entry, rows);
        let renamed = packed(entry, row_major.clone());
        assert_eq!(row_major.len(), 1);
        assert_eq!(renamed[0].symbol, packed_spec.symbol);
        assert_eq!(
            (renamed[0].plan, &renamed[0].arguments),
            (row_major[0].plan, &row_major[0].arguments)
        );
    }
    // Both rings, tiles that overhang N, odd fold counts and groups.
    assert!(entries.iter().any(|e| e.stages == 0));
    assert!(entries.iter().any(|e| e.stages >= 3));
    assert!(entries.iter().any(|e| e.base.query.n % e.tile_n != 0));
    assert!(entries.iter().any(|e| (e.base.query.k / 128) % 2 == 1));
    assert!(
        entries
            .iter()
            .any(|e| e.base.query.groups > 1 && e.base.query.n > 8)
    );
    // Shapes the fold or the packed layout refuse have no packed twin.
    let mut odd = entries[0].clone();
    odd.tile_n = 44;
    assert!(odd.packed_spec().is_err());
    let mut half = entries[0].clone();
    half.base.query.k = 576;
    half.base.query.activation_row_stride = 576;
    half.base.query.activation_group_stride = 576;
    assert!(half.packed_spec().is_err());
}

#[test]
fn packed_wide_folds_equal_row_major_on_gpu() {
    kernel_harness::gpu_test(256 << 20, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let function = |symbol: &str| module.function(symbol).map_err(|e| e.to_string());
        let mut compared = 0;
        for (index, entry) in wide::packed_lab_entries().iter().enumerate() {
            for rows in row_counts(entry) {
                let row_major = launches(entry, rows);
                let images = images(entry, rows, 100 * index as u64 + u64::from(rows))?;
                packed_fold::compare(
                    session,
                    &function,
                    &row_major,
                    &packed(entry, row_major.clone()),
                    &images,
                )
                .map_err(|e| format!("{} rows={rows}: {e}", entry.symbol()))?;
                compared += 1;
            }
        }
        assert_eq!(compared, 21);
        Ok(())
    })
    .unwrap();
}

/// Packed codes from the wrong strip must change the output, so the
/// comparison sees the addressing errors an eight-row oracle cannot.
#[test]
fn packed_wide_comparison_sees_a_moved_strip_on_gpu() {
    kernel_harness::gpu_test(64 << 20, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let function = |symbol: &str| module.function(symbol).map_err(|e| e.to_string());
        let entries = wide::packed_lab_entries();
        // Strips 0 and 1 of one matrix, then the last strip of group 0 and
        // the first of group 1.
        for (n, groups, first) in [(104, 1, 0), (40, 3, 4)] {
            let entry = entries
                .iter()
                .find(|e| (e.base.query.n, e.base.query.groups) == (n, groups))
                .unwrap();
            let rows = entry.base.query.m_capacity;
            let mut images = images(entry, rows, 7)?;
            let strip = 8 * entry.base.query.k as usize;
            let codes = images[2].packed.as_mut().unwrap();
            let (left, right) = codes.split_at_mut((first + 1) * strip);
            left[first * strip..].swap_with_slice(&mut right[..strip]);
            let row_major = launches(entry, rows);
            let error = packed_fold::compare(
                session,
                &function,
                &row_major,
                &packed(entry, row_major.clone()),
                &images,
            )
            .unwrap_err();
            assert!(error.starts_with(&format!("view {OUTPUT} byte")), "{error}");
        }
        Ok(())
    })
    .unwrap();
}

/// Paired timings of WQ_b's 8192 x 1280 fold (the served refilling ring
/// and the pair loop) at the 16- and 32-row decode buckets; a packed entry
/// may be at most 3% slower.
#[test]
#[ignore = "timing: run on an idle GPU"]
fn packed_wide_folds_time_like_row_major_on_gpu() {
    kernel_harness::gpu_test(256 << 20, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let mut slow = Vec::new();
        for entry in wide::packed_lab_entries()
            .iter()
            .filter(|e| e.base.query.n == 8192)
        {
            for rows in [16, 32] {
                let row_major = launches(entry, rows);
                let packed = packed(entry, row_major.clone());
                let ratio = support::time(
                    session,
                    &module,
                    [&row_major, &packed],
                    &images(entry, rows, 5)?,
                )?;
                println!(
                    "{} rows={rows}: packed/row-major {ratio:.4}",
                    entry.symbol()
                );
                if ratio > 1.03 {
                    slow.push((entry.symbol(), rows));
                }
            }
        }
        assert!(slow.is_empty(), "packed entries slower than 1.03: {slow:?}");
        Ok(())
    })
    .unwrap();
}
