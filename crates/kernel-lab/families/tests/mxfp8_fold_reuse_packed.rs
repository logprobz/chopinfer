#[path = "support/packed_row_readers.rs"]
mod support;
#[path = "support/timing_probe.rs"]
mod timing_probe;
use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_harness::packed_fold;
use kernel_lab_families::mxfp8_fold_reuse_packed as family;
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_fold_packed;
use kernels_sm120::gemm::mxfp8_fold_reuse::Entry;

const CUBINS: &str = "CHOPINFER_MXFP8_FOLD_REUSE_PACKED_CUBINS";

// Production selects exact unpaired buckets. The device also supports a
// one-row tile and paired banks, which are exercised with its declared ABI.
fn launches(entry: &Entry, rows: u32) -> Vec<Launch> {
    use kernel_core::contract::{Argument as A, ArgumentList};
    let folds = 4;
    let mut exact = entry.clone();
    exact.ticket.query.m_capacity = rows;
    let b = support::buffers(entry.ticket.query, rows);
    let workspace = View {
        handle: 11,
        bytes: entry.workspace(rows).unwrap(),
    };
    if rows != 1 && !entry.ticket.paired {
        return Sm120::new()
            .with_mxfp8_fold_reuse(exact.clone(), folds)
            .mxfp8_fold_gemm(&exact.ticket.query, Need::rows(rows))
            .unwrap()
            .launches(&b[0], workspace, Counts::rows(rows))
            .unwrap();
    }
    let d = |v: View| A::DeviceBuffer(v.handle);
    let second = |v: View| {
        if entry.ticket.paired {
            d(v)
        } else {
            A::NullDevicePointer
        }
    };
    let args = [
        vec![
            d(b[0].activations),
            d(b[0].activation_scales),
            d(workspace),
            d(b[0].live_rows),
            d(b[0].status),
            A::U64(b[0].status_row_offset),
            A::U32(rows),
        ],
        vec![
            d(b[0].activations),
            d(b[0].activation_scales),
            d(b[0].weights),
            d(b[0].weight_scales),
            second(b[1].weights),
            second(b[1].weight_scales),
            d(b[0].output),
            second(b[1].output),
            d(workspace),
            d(b[0].live_rows),
            A::U32(rows),
            A::U32(folds),
        ],
    ];
    let mut plans = entry.plans(rows.max(6), folds).unwrap();
    plans[0].grid[0] = rows;
    entry
        .specs()
        .into_iter()
        .zip(plans)
        .zip(args)
        .map(|((spec, plan), args)| Launch {
            symbol: spec.symbol,
            rows,
            plan,
            arguments: ArgumentList(args),
            collective: false,
            persistent: false,
        })
        .collect()
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
        expected.body = row.body.replace(">(task,arena", ",true>(task,arena");
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
            invalid.ticket.query.n = n;
            invalid.ticket.query.k = k;
            assert!(invalid.packed_spec().is_err());
        }
        for rows in [1, 6, 17, entry.ticket.query.m_capacity] {
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
            for rows in [1, 6, 17, entry.ticket.query.m_capacity] {
                let calls = launches(entry, rows);
                let images = support::images(
                    entry.ticket.query,
                    rows,
                    entry.ticket.paired,
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
                .map_err(|e| format!("{} rows={rows}: {e}", entry.symbol()))?;
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
                entry.ticket.query,
                rows,
                entry.ticket.paired,
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
            let strip = entry.ticket.query.k as usize * 8;
            let first = if entry.ticket.query.groups > 1 {
                entry.ticket.query.n as usize / 8 - 1
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
    timing_probe::timing_test(512 << 20, |session| {
        let module = support::load(session, CUBINS, family::CUBIN)?;
        let function = |s: &str| module.function(s).map_err(|e| e.to_string());
        for entry in family::entries()
            .iter()
            .filter(|e| e.ticket.query.n >= 5120)
        {
            for rows in [1, 6, 16, entry.ticket.query.m_capacity] {
                let calls = launches(entry, rows);
                let packed = packed(entry, calls.clone());
                let mut images = support::images(
                    entry.ticket.query,
                    rows,
                    entry.ticket.paired,
                    entry.workspace(rows).unwrap(),
                    7,
                )?;
                support::admit_all(&mut images, 7);
                packed_fold::compare(session, &function, &calls, &packed, &images)?;
                let ratio = support::time(session, &module, [&calls, &packed], &images)?;
                println!("{} rows={rows} packed/row-major={ratio:.6}", entry.symbol());
                assert!(ratio <= 1.03, "{} rows={rows}: {ratio}", entry.symbol());
            }
        }
        Ok(())
    })
    .unwrap();
}
