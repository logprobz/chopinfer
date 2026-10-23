//! The split prefill's packed entries against its row-major entries, bit
//! for bit (`kernel_harness::packed_fold`), over the launches of the
//! prepared split op production runs, with negative controls that move one
//! strip of the packed codes.

mod prefill_packed;
#[path = "support/timing_probe.rs"]
mod timing_probe;

use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_harness::LabResult;
use kernel_lab_families::mxfp8_prefill_split_packed as family;
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_fold::Entry;
use kernels_sm120::gemm::{mxfp8_fold_packed, mxfp8_prefill, mxfp8_prefill_split as split};
use prefill_packed::{compare, images, load, move_strip, views};

const CUBINS: &str = "CHOPINFER_MXFP8_PREFILL_SPLIT_PACKED_CUBINS";

/// The row-major launches of the prepared split op at `rows`, and the same
/// launches moved to packed codes.
fn launches(entry: &Entry, rows: u32) -> LabResult<[Vec<Launch>; 2]> {
    let op = Sm120::new()
        .with_mxfp8_prefill_split(entry.clone())
        .mxfp8_prefill_gemm(&entry.query, Need::rows(entry.query.m_capacity))
        .map_err(|e| format!("{e:?}"))?;
    let row_major = op
        .launches(&views(entry, rows), View::EMPTY, Counts::rows(rows))
        .map_err(|e| e.0)?;
    let packed = split::packed_launches(entry, row_major.clone()).map_err(|e| e.0)?;
    Ok([row_major, packed])
}

#[test]
fn packed_specs_differ_from_row_major_only_in_weight_addressing() {
    let entries = [family::entries(), family::timing_entries()].concat();
    let spec = family::cubins().pop().unwrap();
    assert_eq!(spec.entries.len(), 8 * entries.len());
    for entry in &entries {
        let row_major = split::specs(entry).unwrap();
        let packed = split::packed_specs(entry).unwrap();
        assert_eq!(packed.len(), 2);
        let replaced = [
            row_major[1]
                .body
                .replace(",false>(activation_map", ",false,true>(activation_map"),
            row_major[2]
                .body
                .replace(">(activations", ",true>(activations"),
        ];
        for ((packed, row_major), body) in packed.iter().zip(&row_major[1..]).zip(replaced) {
            assert_eq!(packed.symbol, mxfp8_fold_packed::symbol(&row_major.symbol));
            assert_eq!(packed.body, body);
            assert_ne!(packed.body, row_major.body);
            assert_eq!(packed.params, row_major.params);
            assert!(spec.entries.iter().any(|e| e.symbol == packed.symbol));
            assert!(spec.entries.iter().any(|e| e.symbol == row_major.symbol));
        }
        let unsplit = mxfp8_prefill::packed_symbol(entry);
        assert!(spec.entries.iter().any(|e| e.symbol == unsplit));
    }
    // A decode-sized entry has no split tail, packed or not.
    let mut decode = family::entries().remove(1);
    decode.query.m_capacity = 8;
    assert!(split::specs(&decode).is_err() && split::packed_specs(&decode).is_err());
    assert!(split::packed_tail_spec(&decode).is_err());
    let mut odd = family::entries().remove(0);
    odd.query.n = 196;
    split::specs(&odd).unwrap();
    assert!(split::packed_specs(&odd).is_err() && split::packed_tail_spec(&odd).is_err());
}

#[test]
fn packed_launches_move_full_tiles_and_tails() {
    for entry in family::entries() {
        for (rows, split_tail) in [(1, false), (129, true), (200, true), (256, false)] {
            let [row_major, packed] = launches(&entry, rows).unwrap();
            assert_eq!(row_major.len(), if split_tail { 3 } else { 2 });
            assert_eq!(packed[0], row_major[0]);
            assert_eq!(
                packed[1].symbol,
                mxfp8_fold_packed::symbol(&row_major[1].symbol)
            );
            assert_eq!(packed[1].plan, row_major[1].plan);
            assert_ne!(packed[1].arguments.0[1], row_major[1].arguments.0[1]);
            assert_eq!(packed[1].arguments.0[2..], row_major[1].arguments.0[2..]);
            if split_tail {
                assert_eq!(
                    packed[2].symbol,
                    split::packed_tail_spec(&entry).unwrap().symbol
                );
                assert_eq!(packed[2].plan, row_major[2].plan);
                assert_eq!(packed[2].arguments, row_major[2].arguments);
            }
        }
    }
}

#[test]
fn packed_split_prefill_equals_row_major_on_gpu() {
    kernel_harness::gpu_test(256 << 20, |session| {
        let module = load(session, CUBINS, family::CUBIN)?;
        let mut compared = 0;
        for (index, entry) in family::entries().iter().enumerate() {
            // Unsplit below one tile, split tails of 1, 72 and 127 rows, and
            // capacity in whole tiles.
            for rows in [1, 129, 200, 255, entry.query.m_capacity] {
                let [row_major, packed] = launches(entry, rows)?;
                let images = images(entry, rows, 100 * index as u64 + u64::from(rows))?;
                compare(session, &module, &row_major, &packed, &images)
                    .map_err(|e| format!("{} rows={rows}: {e}", entry.symbol))?;
                compared += 1;
            }
        }
        assert_eq!(compared, 30);
        Ok(())
    })
    .unwrap();
}

/// Packed codes from the wrong strip must change the output of the full
/// tiles and of the tail.
#[test]
fn packed_split_prefill_comparison_sees_a_moved_strip_on_gpu() {
    kernel_harness::gpu_test(64 << 20, |session| {
        let module = load(session, CUBINS, family::CUBIN)?;
        let entries = family::entries();
        for (n, k, groups, first) in [(200, 576, 1, 0), (128, 256, 2, 15)] {
            let entry = entries
                .iter()
                .find(|e| (e.query.n, e.query.k, e.query.groups) == (n, k, groups))
                .unwrap();
            let rows = 200;
            let [row_major, packed] = launches(entry, rows)?;
            let mut images = images(entry, rows, 7)?;
            move_strip(&mut images, entry, first);
            assert_eq!(row_major.len(), 3);
            // Every launch, the full tiles alone and the tail alone.
            for range in [0..3, 1..2, 2..3] {
                let error = compare(
                    session,
                    &module,
                    &row_major[range.clone()],
                    &packed[range.clone()],
                    &images,
                )
                .unwrap_err();
                assert!(error.starts_with("view 5 byte"), "{range:?}: {error}");
            }
        }
        Ok(())
    })
    .unwrap();
}

/// Paired row-major and packed times of split launches at DeepSeek V4.1
/// prefill shapes. The packed entries must be no slower (each paired ratio
/// at most 1.03).
#[test]
#[ignore = "GPU timing"]
fn packed_split_prefill_is_no_slower_on_gpu() {
    timing_probe::timing_test(1 << 30, |session| {
        let module = load(session, CUBINS, family::CUBIN)?;
        let mut ratios = Vec::new();
        for entry in family::timing_entries() {
            // Whole tiles with a 72-row tail.
            for rows in [584, 2120, 8136] {
                let [row_major, packed] = launches(&entry, rows)?;
                assert_eq!(row_major.len(), 3);
                let ratio = prefill_packed::packed_over_row_major(
                    session, &module, &entry, rows, &row_major, &packed,
                )?;
                ratios.push((entry.symbol.clone(), rows, ratio));
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
