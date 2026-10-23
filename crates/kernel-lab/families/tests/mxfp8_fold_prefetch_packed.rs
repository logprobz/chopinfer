#[path = "support/packed_row_readers.rs"]
mod support;
#[path = "support/timing_probe.rs"]
mod timing_probe;
use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_harness::packed_fold;
use kernel_lab_families::mxfp8_fold_prefetch_packed as family;
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_fold_packed;
use kernels_sm120::gemm::mxfp8_fold_ticket::Entry;

const CUBINS: &str = "CHOPINFER_MXFP8_FOLD_PREFETCH_PACKED_CUBINS";

fn launches(entry: &Entry, rows: u32) -> Vec<Launch> {
    let backend = Sm120::new().with_mxfp8_fold_ticket(entry.clone());
    let b = support::buffers(entry.query, rows);
    let workspace = View {
        handle: 11,
        bytes: entry.workspace(rows).unwrap(),
    };
    if entry.paired {
        backend
            .mxfp8_fold_gemm_pair(&entry.query, Need::rows(rows))
            .unwrap()
            .launches(&b, workspace, Counts::rows(rows))
            .unwrap()
    } else {
        backend
            .mxfp8_fold_gemm(&entry.query, Need::rows(rows))
            .unwrap()
            .launches(&b[0], workspace, Counts::rows(rows))
            .unwrap()
    }
}

fn packed(entry: &Entry, calls: Vec<Launch>) -> Vec<Launch> {
    mxfp8_fold_packed::launches(calls, |s| s == entry.specs()[1].symbol)
}

#[test]
fn packed_specs_preserve_everything_except_weight_addressing() {
    let entries = family::entries();
    let cubin = family::cubins().pop().unwrap();
    assert_eq!(cubin.entries.len(), entries.len() * 3);
    for entry in entries {
        let row = entry.specs().remove(1);
        let mut expected = row.clone();
        expected.symbol = mxfp8_fold_packed::symbol(&row.symbol);
        expected.body = row.body.replace(">(activations", ",true>(activations");
        let actual = entry.packed_spec().unwrap();
        assert_eq!(actual.symbol, expected.symbol);
        assert_eq!(actual.body, expected.body);
        assert_eq!(actual.params, expected.params);
        assert_eq!(actual.max_threads, expected.max_threads);
        assert_eq!(actual.min_blocks, expected.min_blocks);
        assert_eq!(actual.allowed_local_bytes, expected.allowed_local_bytes);
        assert!(cubin.entries.iter().any(|e| e.symbol == expected.symbol));
        for (n, k) in [(7, 576), (40, 160), (0, 576), (40, 0)] {
            let mut invalid = entry.clone();
            invalid.query.n = n;
            invalid.query.k = k;
            assert!(invalid.packed_spec().is_err());
        }
        for rows in [1, 7, 17, entry.query.m_capacity] {
            let calls = launches(&entry, rows);
            let renamed = packed(&entry, calls.clone());
            for (a, b) in calls.iter().zip(&renamed) {
                assert_eq!(a.plan, b.plan);
                assert_eq!(a.arguments, b.arguments);
            }
        }
    }
}

#[test]
fn packed_equals_row_major_on_gpu() {
    kernel_harness::gpu_test(512 << 20, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let function = |s: &str| module.function(s).map_err(|e| e.to_string());
        for (i, entry) in family::entries().iter().enumerate() {
            for rows in [1, 7, 17, entry.query.m_capacity] {
                let calls = launches(entry, rows);
                let images = support::images(
                    entry.query,
                    rows,
                    entry.paired,
                    entry.workspace(rows).unwrap(),
                    i as u64 * 100 + u64::from(rows),
                )?;
                packed_fold::compare(
                    session,
                    &function,
                    &calls,
                    &packed(entry, calls.clone()),
                    &images,
                )
                .map_err(|e| format!("{} rows={rows}: {e}", entry.symbol))?;
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn packed_comparison_detects_a_moved_strip_on_gpu() {
    kernel_harness::gpu_test(256 << 20, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let function = |s: &str| module.function(s).map_err(|e| e.to_string());
        for entry in family::entries().iter().take(4) {
            let rows = 1;
            let mut images = support::images(
                entry.query,
                rows,
                entry.paired,
                entry.workspace(rows).unwrap(),
                17,
            )?;
            let codes = images
                .iter_mut()
                .find(|i| i.handle == 3)
                .unwrap()
                .packed
                .as_mut()
                .unwrap();
            let strip = entry.query.k as usize * 8;
            let first = if entry.query.groups > 1 {
                entry.query.n as usize / 8 - 1
            } else {
                0
            };
            let (a, b) = codes.split_at_mut((first + 1) * strip);
            a[first * strip..].swap_with_slice(&mut b[..strip]);
            let calls = launches(entry, rows);
            let error = packed_fold::compare(
                session,
                &function,
                &calls,
                &packed(entry, calls.clone()),
                &images,
            )
            .unwrap_err();
            assert!(error.starts_with("view 5 byte"), "{error}");
        }
        Ok(())
    })
    .unwrap();
}

#[test]
#[ignore = "requires an idle GPU and the packed family cubins"]
fn packed_timing_stays_within_three_percent_on_gpu() {
    timing_probe::timing_test(support::TIMING_MEMORY_BYTES, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let function = |s: &str| module.function(s).map_err(|e| e.to_string());
        let mut slow = Vec::new();
        for entry in family::entries().iter().filter(|e| e.query.n >= 5120) {
            // Cover both served windows for N5120/K576; the shared buckets
            // otherwise sample only its 29-32 window.
            let second_window = (entry.query.n == 5120 && entry.query.k == 576).then_some(56);
            for rows in support::TIMING_ROWS.into_iter().chain(second_window).filter(|&rows| rows <= entry.query.m_capacity) {
                let calls = launches(entry, rows);
                let packed = packed(entry, calls.clone());
                let mut images = support::images(
                    entry.query,
                    rows,
                    entry.paired,
                    entry.workspace(rows).unwrap(),
                    7,
                )?;
                support::admit_all(&mut images, 7);
                packed_fold::compare(session, &function, &calls, &packed, &images)?;
                let timing = support::time(session, &module, [&calls, &packed], &images)?;
                let ratio = timing.paired_ratio;
                let served = support::decode_served("prefetch", entry.query, rows, entry.paired);
                println!("{} rows={rows} packed/row-major={ratio:.6} row_major_us={:.6} packed_us={:.6} reps={} served={}",
                    entry.symbol, timing.row_major_us, timing.packed_us, timing.repetitions, if served { "yes" } else { "no" });
                if served && let Some(failure) = timing.failure() {
                    slow.push(format!("{} rows={rows}: {failure}", entry.symbol));
                }
            }
        }
        assert!(
            slow.is_empty(),
            "served packed timing failures:\n{}",
            slow.join("\n")
        );
        Ok(())
    })
    .unwrap();
}
