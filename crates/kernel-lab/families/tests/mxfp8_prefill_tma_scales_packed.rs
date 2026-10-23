//! The staged-scale TMA prefill's packed entries against its row-major
//! entries, bit for bit (`kernel_harness::packed_fold`), over the launches
//! of the prepared op production runs at its registered row counts, with
//! negative controls that move one strip of the packed codes.

mod prefill_packed;
#[path = "support/timing_probe.rs"]
mod timing_probe;

use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_harness::LabResult;
use kernel_lab_families::mxfp8_prefill_tma_scales_packed as family;
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_fold_packed;
use kernels_sm120::gemm::mxfp8_prefill_tma_scales::Entry;
use prefill_packed::{HANDLES, compare, images, load, move_strip, views};

const CUBINS: &str = "CHOPINFER_MXFP8_PREFILL_TMA_SCALES_PACKED_CUBINS";

/// The row-major launches of the prepared op, with `entry` registered at
/// `rows` as production registers it, and the same launches moved to packed
/// codes.
fn launches(entry: &Entry, rows: u32) -> LabResult<[Vec<Launch>; 2]> {
    let base = &entry.base;
    let op = Sm120::new()
        .with_mxfp8_prefill(base.clone())
        .with_mxfp8_prefill_tma_scales(entry.clone(), rows)
        .mxfp8_prefill_gemm(&base.query, Need::rows(base.query.m_capacity))
        .map_err(|e| format!("{e:?}"))?;
    let row_major = op
        .launches(&views(base, rows), View::EMPTY, Counts::rows(rows))
        .map_err(|e| e.0)?;
    if row_major[1].symbol != entry.symbol() {
        return Err(format!("{rows} rows do not launch {}", entry.symbol()));
    }
    let packed = entry.packed_launches(row_major.clone()).map_err(|e| e.0)?;
    Ok([row_major, packed])
}

#[test]
fn packed_specs_differ_from_row_major_only_in_weight_addressing() {
    let entries = [family::entries(), family::timing_entries()].concat();
    let spec = family::cubins().pop().unwrap();
    assert_eq!(spec.entries.len(), 4 * entries.len());
    for entry in &entries {
        let row_major = entry.specs().unwrap().remove(1);
        let packed = entry.packed_spec().unwrap();
        assert_eq!(packed.symbol, mxfp8_fold_packed::symbol(&row_major.symbol));
        assert_eq!(packed.symbol, entry.packed_symbol());
        assert_eq!(
            packed.body,
            row_major
                .body
                .replace(",false,2>(activation_map", ",false,2,true>(activation_map")
        );
        assert_ne!(packed.body, row_major.body);
        assert_eq!(packed.params, row_major.params);
        assert!(spec.entries.iter().any(|e| e.symbol == packed.symbol));
        assert!(spec.entries.iter().any(|e| e.symbol == row_major.symbol));
    }
    let entries = family::entries();
    assert!(entries.iter().any(|e| e.base.query.k == 128));
    assert!(
        entries
            .iter()
            .any(|e| e.base.query.groups > 1 && e.base.query.n % 128 != 0)
    );
    // Staged scales need complete folds, and packed codes N divisible by 8.
    let mut odd = entries[0].clone();
    odd.base.query.n = 196;
    odd.specs().unwrap();
    assert!(odd.packed_spec().is_err());
    let mut half = entries[0].clone();
    half.base.query.k = 576;
    half.base.query.activation_row_stride = 576;
    half.base.query.activation_group_stride = 576;
    assert!(half.specs().is_err() && half.packed_spec().is_err());
}

#[test]
fn packed_launches_move_only_the_compute_launch() {
    for entry in family::entries() {
        for rows in [1, 200, 256] {
            let [row_major, packed] = launches(&entry, rows).unwrap();
            assert_eq!(row_major.len(), 2);
            assert_eq!(packed[0], row_major[0]);
            assert_eq!(packed[1].symbol, entry.packed_symbol());
            assert_eq!(packed[1].plan, row_major[1].plan);
            let expected =
                kernels_sm120::gemm::mxfp8_prefill::packed_arguments(&entry.base, HANDLES, rows)
                    .unwrap();
            assert_eq!(packed[1].arguments, expected);
        }
    }
}

#[test]
fn packed_tma_scales_equal_row_major_on_gpu() {
    kernel_harness::gpu_test(256 << 20, |session| {
        let module = load(session, CUBINS, family::CUBIN)?;
        let mut compared = 0;
        for (index, entry) in family::entries().iter().enumerate() {
            // A boundary tile only, a full and a boundary tile, and capacity
            // in full tiles.
            for rows in [1, 200, entry.base.query.m_capacity] {
                let [row_major, packed] = launches(entry, rows)?;
                let images = images(&entry.base, rows, 100 * index as u64 + u64::from(rows))?;
                compare(session, &module, &row_major, &packed, &images)
                    .map_err(|e| format!("{} rows={rows}: {e}", entry.symbol()))?;
                compared += 1;
            }
        }
        assert_eq!(compared, 12);
        Ok(())
    })
    .unwrap();
}

/// Packed codes from the wrong strip must change the output, through full
/// tiles and boundary tiles.
#[test]
fn packed_tma_scales_comparison_sees_a_moved_strip_on_gpu() {
    kernel_harness::gpu_test(64 << 20, |session| {
        let module = load(session, CUBINS, family::CUBIN)?;
        let entries = family::entries();
        // Strips 0 and 1 of one matrix, then the last strip of group 0 and
        // the first of group 1.
        for (n, groups, first) in [(200, 1, 0), (128, 2, 15)] {
            let entry = entries
                .iter()
                .find(|e| (e.base.query.n, e.base.query.groups) == (n, groups))
                .unwrap();
            for rows in [1, entry.base.query.m_capacity] {
                let [row_major, packed] = launches(entry, rows)?;
                let mut images = images(&entry.base, rows, 7)?;
                move_strip(&mut images, &entry.base, first);
                let error = compare(session, &module, &row_major, &packed, &images).unwrap_err();
                assert!(error.starts_with("view 5 byte"), "{error}");
            }
        }
        Ok(())
    })
    .unwrap();
}

/// Paired row-major and packed times at the wide shapes and row
/// counts this entry serves. The packed entry must be no slower (each
/// paired ratio at most 1.03).
#[test]
#[ignore = "GPU timing"]
fn packed_tma_scales_are_no_slower_on_gpu() {
    timing_probe::timing_test(1 << 30, |session| {
        let module = load(session, CUBINS, family::CUBIN)?;
        let mut ratios = Vec::new();
        for entry in family::timing_entries() {
            for rows in [4096, 8192] {
                let [row_major, packed] = launches(&entry, rows)?;
                let ratio = prefill_packed::packed_over_row_major(
                    session,
                    &module,
                    &entry.base,
                    rows,
                    &row_major,
                    &packed,
                )?;
                ratios.push((entry.symbol(), rows, ratio));
            }
        }
        for (symbol, rows, ratio) in &ratios {
            assert!(
                *ratio <= 1.03,
                "{symbol} rows={rows}: packed/row-major {ratio}"
            );
        }
        Ok(())
    })
    .unwrap();
}
