//! The TMA prefill's packed entries against its row-major entries, bit for
//! bit (`kernel_harness::packed_fold`), over the launches of the prepared
//! op production runs, with negative controls that move one strip of the
//! packed codes.

mod prefill_packed;

use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_core::contract::{Argument, DType};
use kernel_harness::LabResult;
use kernel_lab_families::mxfp8_prefill_packed as family;
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_fold::Entry;
use kernels_sm120::gemm::{mxfp8_fold_packed, mxfp8_prefill};
use prefill_packed::{HANDLES, compare, images, load, move_strip, views};

const CUBINS: &str = "CHOPINFER_MXFP8_PREFILL_PACKED_CUBINS";

/// The row-major launches of the prepared op at `rows`, and the same
/// launches moved to packed codes.
fn launches(entry: &Entry, rows: u32) -> LabResult<[Vec<Launch>; 2]> {
    let op = Sm120::new()
        .with_mxfp8_prefill(entry.clone())
        .mxfp8_prefill_gemm(&entry.query, Need::rows(entry.query.m_capacity))
        .map_err(|e| format!("{e:?}"))?;
    let row_major = op
        .launches(&views(entry, rows), View::EMPTY, Counts::rows(rows))
        .map_err(|e| e.0)?;
    let packed = mxfp8_prefill::packed_launches(entry, row_major.clone()).map_err(|e| e.0)?;
    Ok([row_major, packed])
}

#[test]
fn packed_specs_differ_from_row_major_only_in_weight_addressing() {
    let entries = [family::entries(), family::timing_entries()].concat();
    let spec = family::cubins().pop().unwrap();
    assert_eq!(spec.entries.len(), 3 * entries.len());
    for entry in &entries {
        let row_major = mxfp8_prefill::specs(entry).unwrap().remove(1);
        let packed = mxfp8_prefill::packed_spec(entry).unwrap();
        assert_eq!(packed.symbol, mxfp8_fold_packed::symbol(&row_major.symbol));
        assert_eq!(
            packed.body,
            row_major
                .body
                .replace(",false>(activation_map", ",false,true>(activation_map")
        );
        assert_ne!(packed.body, row_major.body);
        assert_eq!(packed.params, row_major.params);
        assert!(spec.entries.iter().any(|e| e.symbol == packed.symbol));
        assert!(spec.entries.iter().any(|e| e.symbol == row_major.symbol));
    }
    // Both step bodies, a final 64-K fold through full tiles and through the
    // boundary path only, partial column tiles, groups and both outputs.
    let q = |e: &Entry| e.query;
    let entries = family::entries();
    assert!(entries.iter().any(|e| q(e).k < 1024 && q(e).k % 128 == 0));
    assert!(entries.iter().any(|e| q(e).k > 1024 && q(e).k % 128 == 0));
    assert!(
        entries
            .iter()
            .any(|e| q(e).k % 128 == 64 && q(e).k > 128 && q(e).activation_row_stride == q(e).k)
    );
    assert!(
        entries
            .iter()
            .any(|e| q(e).k % 128 == 64 && q(e).activation_row_stride > q(e).k && q(e).groups > 1)
    );
    assert!(entries.iter().any(|e| q(e).n % 128 != 0 && q(e).n > 128));
    assert!(entries.iter().any(|e| q(e).groups > 1 && q(e).n % 128 != 0));
    assert!(entries.iter().any(|e| q(e).output == DType::F32));
    assert!(entries.iter().any(|e| q(e).output == DType::Bf16));
}

#[test]
fn packed_entries_refuse_shapes_the_packed_map_cannot_hold() {
    let base = family::entries().remove(0);
    for (n, k) in [(196, 576), (200, 544), (200, 64)] {
        let mut entry = base.clone();
        entry.query.n = n;
        entry.query.k = k;
        entry.query.activation_group_stride = k;
        entry.query.activation_row_stride = k;
        mxfp8_prefill::specs(&entry).unwrap();
        assert!(mxfp8_prefill::packed_spec(&entry).is_err(), "n={n} k={k}");
        assert!(mxfp8_prefill::packed_weight_map(&entry, 3).is_err());
    }
    // A map box is one fold of 16 strips; the extent stays inside the codes.
    for entry in family::entries() {
        let q = entry.query;
        let map = mxfp8_prefill::packed_weight_map(&entry, 3).unwrap();
        assert_eq!(map.box_dims.iter().product::<u32>(), 128 * 128);
        let extent = map.strides_bytes[2] * map.dims[2] + map.strides_bytes[3] * (map.dims[3] - 1);
        assert!(extent <= u64::from(q.groups * q.n * q.k));
        assert_eq!(
            map.strides_bytes[3] * map.dims[3],
            u64::from(q.groups * q.n * q.k)
        );
    }
}

#[test]
fn packed_launches_move_only_the_compute_launch() {
    for entry in family::entries() {
        for rows in [1, 200, 256] {
            let [row_major, packed] = launches(&entry, rows).unwrap();
            assert_eq!(row_major.len(), 2);
            assert_eq!(packed[0], row_major[0]);
            assert_eq!(packed[1].symbol, mxfp8_prefill::packed_symbol(&entry));
            assert_eq!(packed[1].plan, row_major[1].plan);
            let expected = mxfp8_prefill::packed_arguments(&entry, HANDLES, rows).unwrap();
            assert_eq!(packed[1].arguments, expected);
            let Argument::TensorMap(map) = &packed[1].arguments.0[1] else {
                panic!("no packed weight map");
            };
            assert_eq!(map.base_buffer, HANDLES.weights);
            for (i, (a, b)) in packed[1]
                .arguments
                .0
                .iter()
                .zip(&row_major[1].arguments.0)
                .enumerate()
            {
                assert_eq!(a == b, i != 1);
            }
        }
    }
}

#[test]
fn packed_prefill_equals_row_major_on_gpu() {
    kernel_harness::gpu_test(256 << 20, |session| {
        let module = load(session, CUBINS, family::CUBIN)?;
        let mut compared = 0;
        for (index, entry) in family::entries().iter().enumerate() {
            // A boundary tile only, one full tile, a full and a boundary
            // tile, and capacity in full tiles.
            for rows in [1, 128, 200, entry.query.m_capacity] {
                let [row_major, packed] = launches(entry, rows)?;
                let images = images(entry, rows, 100 * index as u64 + u64::from(rows))?;
                compare(session, &module, &row_major, &packed, &images)
                    .map_err(|e| format!("{} rows={rows}: {e}", entry.symbol))?;
                compared += 1;
            }
        }
        assert_eq!(compared, 24);
        Ok(())
    })
    .unwrap();
}

/// Packed codes from the wrong strip must change the output, through full
/// tiles and boundary tiles.
#[test]
fn packed_prefill_comparison_sees_a_moved_strip_on_gpu() {
    kernel_harness::gpu_test(64 << 20, |session| {
        let module = load(session, CUBINS, family::CUBIN)?;
        let entries = family::entries();
        // Strips 0 and 1 of one matrix with a final 64-K fold, then the last
        // strip of group 0 and the first of group 1.
        for (n, k, groups, first) in [(200, 576, 1, 0), (128, 256, 2, 15)] {
            let entry = entries
                .iter()
                .find(|e| (e.query.n, e.query.k, e.query.groups) == (n, k, groups))
                .unwrap();
            for rows in [1, entry.query.m_capacity] {
                let [row_major, packed] = launches(entry, rows)?;
                let mut images = images(entry, rows, 7)?;
                move_strip(&mut images, entry, first);
                let error = compare(session, &module, &row_major, &packed, &images).unwrap_err();
                assert!(error.starts_with("view 5 byte"), "{error}");
            }
        }
        Ok(())
    })
    .unwrap();
}

/// Paired row-major and packed times at DeepSeek V4.1 prefill shapes. The
/// packed entry must be no slower (each paired ratio at most 1.03).
#[test]
#[ignore = "GPU timing"]
fn packed_prefill_is_no_slower_on_gpu() {
    kernel_harness::gpu_test(1 << 30, |session| {
        let module = load(session, CUBINS, family::CUBIN)?;
        let mut ratios = Vec::new();
        for entry in family::timing_entries() {
            for rows in [512, 2048, 8192] {
                let [row_major, packed] = launches(&entry, rows)?;
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
