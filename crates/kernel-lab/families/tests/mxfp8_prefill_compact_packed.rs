//! The compact prefill tiles' packed entries against their row-major
//! entries, bit for bit (`kernel_harness::packed_fold`), on the launches the
//! `Sm120` backend prepares for registered exact row counts, with a negative
//! control that moves one strip of the packed codes and paired timings at
//! served prefill shapes.

#[path = "packed_staged/support.rs"]
mod support;
#[path = "support/timing_probe.rs"]
mod timing_probe;

use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
use kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout;
use kernel_harness::LabResult;
use kernel_harness::packed_fold::{self, Image};
use kernel_lab_families::mxfp8_prefill_compact_packed as family;
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_fold_packed;
use kernels_sm120::gemm::mxfp8_prefill_compact::{self as compact, Entry};

const CUBINS: &str = "CHOPINFER_MXFP8_PREFILL_COMPACT_PACKED_CUBINS";
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

/// The launches the backend's prefill op returns at `rows` rows with the
/// entry registered for exactly that count, as a model registers it.
fn launches(entry: &Entry, rows: u32) -> Vec<Launch> {
    let q = entry.base.query;
    Sm120::new()
        .with_mxfp8_prefill(entry.base.clone())
        .with_mxfp8_prefill_compact(entry.clone(), rows)
        .mxfp8_prefill_gemm(&q, Need::rows(q.m_capacity))
        .unwrap()
        .launches(&buffers(entry, rows), View::EMPTY, Counts::rows(rows))
        .unwrap()
}

/// `launches` with the compute launch renamed; the admission reads no
/// weight codes.
fn packed(entry: &Entry, rows: u32, launches: Vec<Launch>) -> Vec<Launch> {
    let symbol = entry.symbol(Entry::tile(rows));
    mxfp8_fold_packed::launches(launches, |s| s == symbol)
}

/// 16-, 32- and 64-row tiles, partial tiles and full capacity.
fn row_counts(entry: &Entry) -> Vec<u32> {
    match entry.base.query.m_capacity {
        8192 => vec![1, 513, 8192],
        capacity => vec![1, 13, 100, capacity],
    }
}

/// Every view the launches name. Row 1 holds a NaN code when there are two
/// rows, so the admission fails it.
fn images(entry: &Entry, rows: u32, seed: u64) -> LabResult<Vec<Image>> {
    let b = buffers(entry, rows);
    let mut activations = packed_fold::codes(b.activations.bytes as usize, seed);
    if rows > 1 {
        activations[entry.base.query.activation_row_stride as usize + 4] = 0x7f;
    }
    Ok(vec![
        Image::data(b.activations.handle, activations),
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
        Image::data(b.status.handle, vec![0; b.status.bytes as usize]),
    ])
}

#[test]
fn packed_lab_entries_differ_from_row_major_only_in_weight_addressing() {
    let entries = compact::packed_lab_entries();
    let spec = family::cubins().pop().unwrap();
    assert_eq!(spec.entries.len(), 10 * entries.len());
    for entry in &entries {
        let row_major = entry.specs().unwrap();
        for tile in [16, 32, 64] {
            let served = row_major
                .iter()
                .find(|s| s.symbol == entry.symbol(tile))
                .unwrap();
            let packed_spec = entry.packed_spec(tile).unwrap();
            support::assert_packed_twin(
                served,
                &packed_spec,
                &served.body.replace(">(activations", ",true>(activations"),
            );
            assert!(spec.entries.iter().any(|e| e.symbol == packed_spec.symbol));
            assert!(spec.entries.iter().any(|e| e.symbol == served.symbol));
        }
        layout(entry).packed_bytes().unwrap();
        // Renaming changes only the compute launch's symbol.
        for rows in row_counts(entry) {
            let row_major = launches(entry, rows);
            let renamed = packed(entry, rows, row_major.clone());
            assert_eq!(row_major.len(), 2);
            assert_eq!(renamed[0], row_major[0]);
            assert_eq!(
                renamed[1].symbol,
                entry.packed_spec(Entry::tile(rows)).unwrap().symbol
            );
            assert_eq!(
                (renamed[1].plan, &renamed[1].arguments),
                (row_major[1].plan, &row_major[1].arguments)
            );
        }
    }
    // Final 64-K folds after full ones, K divisible by 128, both stage
    // counts, groups and both outputs; every tile is reached.
    let q = |e: &Entry| e.base.query;
    assert!(entries.iter().any(|e| q(e).k % 128 == 64 && q(e).k > 128));
    assert!(entries.iter().any(|e| q(e).k % 128 == 0));
    assert!(entries.iter().any(|e| e.stages == 2) && entries.iter().any(|e| e.stages == 3));
    assert!(entries.iter().any(|e| q(e).groups > 1 && q(e).n > 8));
    assert!(
        entries
            .iter()
            .any(|e| q(e).output == kernel_core::contract::DType::Bf16)
    );
    assert!(
        entries
            .iter()
            .any(|e| q(e).output == kernel_core::contract::DType::F32)
    );
    for tile in [16, 32, 64] {
        assert!(
            entries
                .iter()
                .any(|e| row_counts(e).iter().any(|&r| Entry::tile(r) == tile))
        );
    }
    // Shapes the packed layout cannot hold have no packed twin.
    let mut odd = entries[0].clone();
    odd.base.query.k = 544;
    assert!(odd.packed_spec(32).is_err());
    let mut narrow = entries[0].clone();
    narrow.base.query.n = 68;
    assert!(narrow.packed_spec(32).is_err());
}

#[test]
fn packed_compact_tiles_equal_row_major_on_gpu() {
    kernel_harness::gpu_test(1 << 30, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let function = |symbol: &str| module.function(symbol).map_err(|e| e.to_string());
        let mut compared = 0;
        for (index, entry) in compact::packed_lab_entries().iter().enumerate() {
            for rows in row_counts(entry) {
                let row_major = launches(entry, rows);
                let images = images(entry, rows, 100 * index as u64 + u64::from(rows))?;
                packed_fold::compare(
                    session,
                    &function,
                    &row_major,
                    &packed(entry, rows, row_major.clone()),
                    &images,
                )
                .map_err(|e| format!("{} rows={rows}: {e}", entry.symbol(Entry::tile(rows))))?;
                compared += 1;
            }
        }
        assert_eq!(compared, 25);
        Ok(())
    })
    .unwrap();
}

/// Packed codes from the wrong strip must change the output, so the
/// comparison sees the addressing errors an eight-row oracle cannot.
#[test]
fn packed_compact_comparison_sees_a_moved_strip_on_gpu() {
    kernel_harness::gpu_test(64 << 20, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let function = |symbol: &str| module.function(symbol).map_err(|e| e.to_string());
        let entries = compact::packed_lab_entries();
        // Strips 0 and 1 of one matrix, then the last strip of group 0 and
        // the first of group 1.
        for (n, groups, first) in [(72, 1, 0), (40, 3, 4)] {
            let entry = entries
                .iter()
                .find(|e| (e.base.query.n, e.base.query.groups) == (n, groups))
                .unwrap();
            let rows = 100;
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
                &packed(entry, rows, row_major.clone()),
                &images,
            )
            .unwrap_err();
            assert!(error.starts_with(&format!("view {OUTPUT} byte")), "{error}");
        }
        Ok(())
    })
    .unwrap();
}

/// Paired timings of wide prefill shapes (N 5120 by K 2048,
/// N 8192 by K 1280 and N 5120 by K 576) at 512, 2048 and 8192 rows; a
/// packed entry may be at most 3% slower.
#[test]
#[ignore = "timing: run on an idle GPU"]
fn packed_compact_tiles_time_like_row_major_on_gpu() {
    timing_probe::timing_test(1 << 30, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let mut slow = Vec::new();
        for entry in compact::packed_lab_entries()
            .iter()
            .filter(|e| e.base.query.m_capacity == 8192)
        {
            for rows in [512, 2048, 8192] {
                let row_major = launches(entry, rows);
                let packed = packed(entry, rows, row_major.clone());
                let ratio = support::time(
                    session,
                    &module,
                    [&row_major, &packed],
                    &images(entry, rows, 5)?,
                )?;
                let symbol = entry.symbol(Entry::tile(rows));
                println!("{symbol} rows={rows}: packed/row-major {ratio:.4}");
                if ratio > 1.03 {
                    slow.push((symbol, rows));
                }
            }
        }
        assert!(slow.is_empty(), "packed entries slower than 1.03: {slow:?}");
        Ok(())
    })
    .unwrap();
}
