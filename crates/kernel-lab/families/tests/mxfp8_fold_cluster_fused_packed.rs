//! The fused cluster folds' packed entries against their row-major entries,
//! bit for bit (`kernel_harness::packed_fold`), on the launches their
//! prepared ops return, with a negative control that moves one strip of the
//! packed codes and paired timings at the served decode schedules.

#[path = "packed_staged/support.rs"]
mod support;

use kernel_core::backend::{Counts, Launch, Need, View};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
use kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout;
use kernel_harness::LabResult;
use kernel_harness::packed_fold::{self, Image};
use kernel_lab_families::mxfp8_fold_cluster_fused_packed as family;
use kernels_sm120::gemm::mxfp8_fold_cluster_fused::{
    self as fused, Entry, FusedBuffers, Quantized,
};
use kernels_sm120::gemm::{mxfp8_fold, mxfp8_fold_packed};

const CUBINS: &str = "CHOPINFER_MXFP8_FOLD_CLUSTER_FUSED_PACKED_CUBINS";
const OUTPUT: u64 = 5;
const WORKSPACE: u64 = 8;

fn layout(entry: &Entry) -> Mxfp8FoldLayout {
    let q = entry.cluster.base.query;
    Mxfp8FoldLayout {
        groups: q.groups as usize,
        n: q.n as usize,
        k: q.k as usize,
    }
}

fn buffers(entry: &Entry, rows: u32) -> FusedBuffers {
    let size = entry.cluster.base.bytes(rows);
    let q = entry.cluster.base.query;
    let v = |handle, bytes| View { handle, bytes };
    let columns = u64::from(rows) * u64::from(q.n * q.groups);
    FusedBuffers {
        fold: Mxfp8FoldGemmBuffers {
            activations: v(1, size.activations),
            activation_scales: v(2, size.activation_scales),
            weights: v(3, size.weights),
            weight_scales: v(4, size.weight_scales),
            output: v(OUTPUT, size.output),
            live_rows: v(6, 4),
            status: v(7, size.status),
            status_row_offset: 0,
        },
        quantized: entry.quantize.then_some(Quantized {
            codes: v(9, columns),
            scales: v(10, columns / 32),
            flags: v(11, columns / 32),
        }),
        pending: (entry.pending != 0).then_some(v(12, u64::from(rows) * u64::from(entry.pending))),
    }
}

/// The launches the entry's prepared op returns at `rows` rows: the plain
/// op's when the fold has neither a quantized output nor pending flags.
fn launches(entry: &Entry, rows: u32) -> Vec<Launch> {
    let need = Need::rows(entry.cluster.base.query.m_capacity);
    let b = buffers(entry, rows);
    let work = View {
        handle: WORKSPACE,
        bytes: entry.workspace(),
    };
    if entry.quantize || entry.pending != 0 {
        entry
            .prepare(need)
            .unwrap()
            .launches(&b, work, Counts::rows(rows))
    } else {
        entry
            .prepare_plain(need)
            .unwrap()
            .launches(&b.fold, work, Counts::rows(rows))
    }
    .unwrap()
}

fn packed(entry: &Entry, launches: Vec<Launch>) -> Vec<Launch> {
    let symbol = entry.symbol();
    mxfp8_fold_packed::launches(launches, |s| s == symbol)
}

fn row_counts(entry: &Entry) -> [u32; 3] {
    let capacity = entry.cluster.base.query.m_capacity;
    [1, capacity / 2 + 1, capacity]
}

/// Every view the launches name. Row 1 has a nonzero status and row 2 a
/// pending flag, so the admission paths run too.
fn images(entry: &Entry, rows: u32, seed: u64) -> LabResult<Vec<Image>> {
    let b = buffers(entry, rows);
    let f = b.fold;
    let mut status = vec![0; f.status.bytes as usize];
    if rows > 1 {
        status[4] = 4;
    }
    let mut images = vec![
        Image::data(
            f.activations.handle,
            packed_fold::codes(f.activations.bytes as usize, seed),
        ),
        Image::data(
            f.activation_scales.handle,
            packed_fold::scales(f.activation_scales.bytes as usize, seed + 1),
        ),
        Image::codes(f.weights.handle, layout(entry), seed + 2)?,
        Image::data(
            f.weight_scales.handle,
            packed_fold::scales(f.weight_scales.bytes as usize, seed + 3),
        ),
        Image::data(f.output.handle, vec![0xa5; f.output.bytes as usize]),
        Image::data(f.live_rows.handle, rows.to_le_bytes().to_vec()),
        Image::data(f.status.handle, status),
        Image::data(WORKSPACE, vec![0x5a; entry.workspace() as usize]),
    ];
    for view in b
        .quantized
        .iter()
        .flat_map(|o| [o.codes, o.scales, o.flags])
    {
        images.push(Image::data(view.handle, vec![0xa5; view.bytes as usize]));
    }
    if let Some(view) = b.pending {
        let mut flags = vec![0; view.bytes as usize];
        if rows > 2 {
            flags[2 * entry.pending as usize] = 1;
        }
        images.push(Image::data(view.handle, flags));
    }
    Ok(images)
}

#[test]
fn packed_lab_entries_differ_from_row_major_only_in_weight_addressing() {
    let entries = fused::packed_lab_entries();
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
        let rows = entry.cluster.base.query.m_capacity;
        let row_major = launches(entry, rows);
        let renamed = packed(entry, row_major.clone());
        assert_eq!(row_major.len(), 1);
        assert_eq!(renamed[0].symbol, packed_spec.symbol);
        assert_eq!(
            (renamed[0].plan, &renamed[0].arguments),
            (row_major[0].plan, &row_major[0].arguments)
        );
    }
    // Every column tile, both row tiles, groups, the plain fold and the
    // status fold with a quantized output and with pending flags.
    for tile in [16, 32, 64] {
        assert!(entries.iter().any(|e| e.cluster.tile_n == tile));
    }
    for rows in [16, 32] {
        assert!(entries.iter().any(|e| e.cluster.tile_rows == rows));
    }
    assert!(
        entries
            .iter()
            .any(|e| e.cluster.base.query.groups > 1 && e.cluster.base.query.n > 8)
    );
    assert!(entries.iter().any(|e| !e.quantize && e.pending == 0));
    assert!(entries.iter().any(|e| e.quantize));
    assert!(entries.iter().any(|e| e.pending != 0));
    // A shape the fold refuses has no packed twin.
    let mut query: Mxfp8FoldGemmQuery = entries[0].cluster.base.query;
    query.k = 576;
    query.activation_row_stride = 576;
    query.activation_group_stride = 576;
    let mut refused = entries[0].clone();
    refused.cluster.base = mxfp8_fold::Entry::new(query, "refused").unwrap();
    assert!(refused.packed_spec().is_err());
}

#[test]
fn packed_fused_folds_equal_row_major_on_gpu() {
    kernel_harness::gpu_test(256 << 20, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let function = |symbol: &str| module.function(symbol).map_err(|e| e.to_string());
        let mut compared = 0;
        for (index, entry) in fused::packed_lab_entries().iter().enumerate() {
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
        assert_eq!(compared, 39);
        Ok(())
    })
    .unwrap();
}

/// Packed codes from the wrong strip must change the output, so the
/// comparison sees the addressing errors an eight-row oracle cannot.
#[test]
fn packed_fused_comparison_sees_a_moved_strip_on_gpu() {
    kernel_harness::gpu_test(64 << 20, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let function = |symbol: &str| module.function(symbol).map_err(|e| e.to_string());
        let entries = fused::packed_lab_entries();
        // Strips 0 and 1 of one matrix, then the last strip of group 0 and
        // the first of group 1.
        for (n, groups, first) in [(64, 1, 0), (48, 3, 5)] {
            let entry = entries
                .iter()
                .find(|e| (e.cluster.base.query.n, e.cluster.base.query.groups) == (n, groups))
                .unwrap();
            let rows = entry.cluster.base.query.m_capacity;
            let mut images = images(entry, rows, 7)?;
            let strip = 8 * entry.cluster.base.query.k as usize;
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

/// Paired timings of each served WQ_a, WKV, WO_a and WO_b schedule at its
/// decode row count; a packed entry may be at most 3% slower.
#[test]
#[ignore = "timing: run on an idle GPU"]
fn packed_fused_folds_time_like_row_major_on_gpu() {
    kernel_harness::gpu_test(256 << 20, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let mut slow = Vec::new();
        for entry in fused::packed_lab_entries()
            .iter()
            .filter(|e| e.cluster.base.query.n >= 512)
        {
            let rows = entry.cluster.base.query.m_capacity;
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
                slow.push(entry.symbol());
            }
        }
        assert!(slow.is_empty(), "packed entries slower than 1.03: {slow:?}");
        Ok(())
    })
    .unwrap();
}
