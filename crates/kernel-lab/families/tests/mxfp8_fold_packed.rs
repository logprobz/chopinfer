//! The v2 fold's packed entries against its row-major entries, bit for bit
//! (`kernel_harness::packed_fold`), with negative controls that move one
//! strip of the packed codes.

use kernel_core::backend::Launch;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
use kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout;
use kernel_harness::packed_fold::{self, Image};
use kernel_harness::{LabResult, Session};
use kernel_lab_families::mxfp8_fold_packed as family;
use kernels_sm120::gemm::mxfp8_fold::{self, Entry};
use kernels_sm120::gemm::mxfp8_fold_packed;
use runtime_cuda::Module;
use std::path::PathBuf;

const HANDLES: Mxfp8FoldGemmBuffers<u64> = Mxfp8FoldGemmBuffers {
    activations: 1,
    activation_scales: 2,
    weights: 3,
    weight_scales: 4,
    output: 5,
    live_rows: 6,
    status: 7,
    status_row_offset: 0,
};

fn layout(entry: &Entry) -> Mxfp8FoldLayout {
    Mxfp8FoldLayout {
        groups: entry.query.groups as usize,
        n: entry.query.n as usize,
        k: entry.query.k as usize,
    }
}

fn launch(entry: &Entry, symbol: String, rows: u32) -> Launch {
    Launch {
        symbol,
        rows,
        plan: entry.plan(rows).unwrap(),
        arguments: entry.arguments(HANDLES, rows),
        collective: false,
        persistent: false,
    }
}

fn row_counts(entry: &Entry) -> Vec<u32> {
    let capacity = entry.query.m_capacity;
    let mut rows = vec![1, capacity.div_ceil(2), capacity];
    rows.dedup();
    rows
}

/// Every view of one launch. Row 1 holds a NaN code when there are two
/// rows, so the status path runs too.
fn images(entry: &Entry, rows: u32, seed: u64) -> LabResult<Vec<Image>> {
    let sizes = entry.bytes(rows);
    let mut activations = packed_fold::codes(sizes.activations as usize, seed);
    if rows > 1 {
        activations[entry.query.activation_row_stride as usize + 4] = 0x7f;
    }
    Ok(vec![
        Image::data(HANDLES.activations, activations),
        Image::data(
            HANDLES.activation_scales,
            packed_fold::scales(sizes.activation_scales as usize, seed + 1),
        ),
        Image::codes(HANDLES.weights, layout(entry), seed + 2)?,
        Image::data(
            HANDLES.weight_scales,
            packed_fold::scales(sizes.weight_scales as usize, seed + 3),
        ),
        Image::data(HANDLES.output, vec![0xa5; sizes.output as usize]),
        Image::data(HANDLES.live_rows, rows.to_le_bytes().to_vec()),
        Image::data(HANDLES.status, vec![0; sizes.status as usize]),
    ])
}

#[test]
fn packed_lab_entries_differ_from_row_major_only_in_weight_addressing() {
    let entries = mxfp8_fold::packed_lab_entries();
    let spec = family::cubins().pop().unwrap();
    assert_eq!(spec.entries.len(), 2 * entries.len());
    for entry in &entries {
        let row_major = entry.spec();
        let packed = entry.packed_spec().unwrap();
        assert_eq!(packed.symbol, mxfp8_fold_packed::symbol(&entry.symbol));
        assert_eq!(
            packed.body,
            row_major
                .body
                .replace(">(activations", ",true>(activations")
        );
        assert_ne!(packed.body, row_major.body);
        assert_eq!(packed.params, row_major.params);
        assert!(spec.entries.iter().any(|e| e.symbol == packed.symbol));
        assert!(spec.entries.iter().any(|e| e.symbol == entry.symbol));
        layout(entry).packed_bytes().unwrap();
    }
    // Both tilings, a final 64-K fold, several strips and several groups.
    assert!(entries.iter().any(|e| e.decode()) && entries.iter().any(|e| !e.decode()));
    assert!(
        entries
            .iter()
            .any(|e| e.query.k % 128 == 64 && e.query.k > 64)
    );
    assert!(entries.iter().any(|e| e.query.groups > 1 && e.query.n > 8));
    for entry in mxfp8_fold::lab_entries() {
        let fits = entry.query.n % 8 == 0 && entry.query.k % 64 == 0;
        assert_eq!(entry.packed_spec().is_ok(), fits, "{}", entry.symbol);
    }
}

fn load(session: &Session) -> LabResult<Module> {
    let directory = std::env::var_os("CHOPINFER_FOLD_PACKED_CUBINS")
        .map(PathBuf::from)
        .ok_or("CHOPINFER_FOLD_PACKED_CUBINS is required")?;
    kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
        .and_then(|r| r.verify_current())
        .map_err(|e| e.to_string())?;
    let bytes = std::fs::read(directory.join(format!("{}.cubin", family::CUBIN)))
        .map_err(|e| e.to_string())?;
    session.load_cubin(&bytes)
}

#[test]
fn packed_fold_equals_row_major_on_gpu() {
    kernel_harness::gpu_test(256 << 20, |session| {
        let module = load(session)?;
        let function = |symbol: &str| module.function(symbol).map_err(|e| e.to_string());
        let mut compared = 0;
        for (index, entry) in mxfp8_fold::packed_lab_entries().iter().enumerate() {
            let packed = mxfp8_fold_packed::symbol(&entry.symbol);
            for rows in row_counts(entry) {
                let images = images(entry, rows, 100 * index as u64 + u64::from(rows))?;
                packed_fold::compare(
                    session,
                    &function,
                    &[launch(entry, entry.symbol.clone(), rows)],
                    &[launch(entry, packed.clone(), rows)],
                    &images,
                )
                .map_err(|e| format!("{} rows={rows}: {e}", entry.symbol))?;
                compared += 1;
            }
        }
        assert_eq!(compared, 16);
        Ok(())
    })
    .unwrap();
}

/// Packed codes from the wrong strip must change the output, so the
/// comparison sees the addressing errors an eight-row oracle cannot.
#[test]
fn packed_fold_comparison_sees_a_moved_strip_on_gpu() {
    kernel_harness::gpu_test(64 << 20, |session| {
        let module = load(session)?;
        let function = |symbol: &str| module.function(symbol).map_err(|e| e.to_string());
        let entries = mxfp8_fold::packed_lab_entries();
        // Strips 0 and 1 of one matrix, then the last strip of group 0 and
        // the first of group 1.
        for (n, groups, first) in [(64, 1, 0), (16, 2, 1)] {
            let entry = entries
                .iter()
                .find(|e| e.query.n == n && e.query.groups == groups)
                .unwrap();
            let rows = entry.query.m_capacity;
            let mut images = images(entry, rows, 7)?;
            let strip = 8 * entry.query.k as usize;
            let codes = images[2].packed.as_mut().unwrap();
            let (left, right) = codes.split_at_mut((first + 1) * strip);
            left[first * strip..].swap_with_slice(&mut right[..strip]);
            let error = packed_fold::compare(
                session,
                &function,
                &[launch(entry, entry.symbol.clone(), rows)],
                &[launch(
                    entry,
                    mxfp8_fold_packed::symbol(&entry.symbol),
                    rows,
                )],
                &images,
            )
            .unwrap_err();
            assert!(error.starts_with("view 5 byte"), "{error}");
        }
        Ok(())
    })
    .unwrap();
}
