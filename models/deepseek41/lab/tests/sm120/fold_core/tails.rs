//! Missing small-row and drafter-tail folds against the current packed selector.
//! The scoped switch changes lab references and candidates only. Serving
//! admission stays fixed until bitwise, memory and timing evidence is collected.
use super::*;
use std::cell::{Cell, RefCell};

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static SCRATCH: RefCell<Vec<View>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn active() -> bool {
    ACTIVE.with(Cell::get)
}

fn scoped<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ACTIVE.with(|v| v.set(self.0));
        }
    }
    let _restore = Restore(ACTIVE.with(|v| v.replace(true)));
    run()
}

pub(super) fn start_case() {
    if active() {
        SCRATCH.with_borrow_mut(Vec::clear);
    }
}

pub(super) fn work(a: &mut Arena<'_>, bytes: u64) -> LabResult<View> {
    let v = a.filled(bytes.max(4), 0)?;
    if active() {
        SCRATCH.with_borrow_mut(|scratch| scratch.push(v));
    }
    Ok(v)
}

pub(super) fn poison(a: &Arena<'_>) -> LabResult<()> {
    if active() && a.session.is_some() {
        SCRATCH.with_borrow(|scratch| {
            for &v in scratch {
                a.fill(v, 0xa5)?;
            }
            Ok(())
        })
    } else {
        Ok(())
    }
}

/// The same packed stream/direct/reuse/ticket selection that serves the
/// missing buckets, including initialization on each launch sequence.
pub(super) fn reference(
    a: &mut Arena<'_>,
    base: &mxfp8_fold::Entry,
    b: &Buffers<View>,
    rows: u32,
) -> LabResult<Vec<Launch>> {
    let packed = packed_views(a, base, b)?;
    let q = sm120::mxfp8_fold::routed_query(base.query, rows);
    if let Ok(op) = backend().mxfp8_fold_gemm_stream(&q, false, Need::rows(rows)) {
        // Persistent counters start at zero once and survive replays.
        // Only disposable workspace is poisoned before every launch.
        let state = a.filled(op.info().bytes_of("state").ok_or("missing fold state")?, 0)?;
        let views = Mxfp8FoldStreamBuffers {
            first: packed,
            second: None,
            packed_weights: [a.alias(packed.weights), View::EMPTY],
            state,
        };
        let scratch = work(a, op.info().workspace)?;
        op.launches(&views, scratch, Counts::rows(rows))
            .map_err(|e| e.to_string())
    } else {
        let op = backend()
            .mxfp8_fold_gemm(&q, Need::rows(rows))
            .map_err(|e| e.to_string())?;
        let scratch = work(a, op.info().workspace)?;
        op.launches(&packed, scratch, Counts::rows(rows))
            .map_err(|e| e.to_string())
    }
}

pub(super) fn ignore_status(a: &mut Arena<'_>, run: &Launch, rows: u32) -> LabResult<Launch> {
    let mut control = run.clone();
    let zeros = a.filled(u64::from(rows + 5) * 256, 0)?;
    control.arguments.0[11] = Argument::DeviceBuffer(zeros.handle);
    if control.arguments.0[8] != Argument::NullDevicePointer {
        control.arguments.0[8] = Argument::DeviceBuffer(zeros.handle);
    }
    Ok(control)
}

pub(super) fn status_build(
    a: &mut Arena<'_>,
    id: &'static str,
    tile: u32,
    rows: u32,
    input: &data::Input,
) -> LabResult<StatusCase> {
    start_case();
    let b = (0..2)
        .map(|_| status_views(a, input, rows))
        .collect::<LabResult<Vec<_>>>()?;
    let base = base_of(input.n, input.k, input.groups);
    let served = reference(a, &base, &b[0], rows)?;
    let packed = packed_views(a, &base, &b[1])?;
    let (run, control) = if id == "attention.q-b" {
        let run = if rows > 32 {
            let entry = sm120::fold_verify_local::entry().map_err(|e| e.to_string())?;
            let mut run = entry.launch(&packed, rows).map_err(|e| e.to_string())?;
            run.symbol = entry.packed_symbol();
            run
        } else {
            let entry = sm120::fold_fused::wide(32).ok_or("missing wide status fold")?;
            let mut run = entry.launch(&packed, rows).map_err(|e| e.to_string())?;
            run.symbol = entry.packed_spec().map_err(|e| e.to_string())?.symbol;
            run
        };
        let mut control = run.clone();
        let zeros = a.filled(u64::from(rows + 2) * 4, 0)?;
        control.arguments.0[8] = Argument::DeviceBuffer(zeros.handle);
        (run, control)
    } else {
        let entry = if tile == 64 {
            sm120::fold_verify::entry(id)
        } else {
            sm120::fold_fused::fused(id, tile)
        }
        .ok_or("missing status fold")?;
        let scratch = work(a, entry.workspace())?;
        let mut run = entry
            .launch(&packed, None, None, scratch, rows)
            .map_err(|e| e.to_string())?;
        run.symbol = entry.packed_spec().map_err(|e| e.to_string())?.symbol;
        let control = ignore_status(a, &run, rows)?;
        (run, control)
    };
    Ok(StatusCase {
        id,
        served,
        candidates: vec![("status".into(), vec![run])],
        control: vec![control],
        b,
    })
}

fn rows() -> Vec<u32> {
    (3..=31)
        .filter(|r| ![4, 8, 16].contains(r))
        .chain([35, 40, 45, 50, 55, 60])
        .collect()
}

fn gpu(s: &Session) -> LabResult<()> {
    let mode = mode()?;
    let group = std::env::var("CHOPINFER_FOLD_TAIL_GROUP").map_err(|e| e.to_string())?;
    if !["q-a", "q-b", "wo", "drafter35", "calibration"].contains(&group.as_str()) {
        return Err("CHOPINFER_FOLD_TAIL_GROUP: q-a, q-b, wo, drafter35 or calibration".into());
    }
    let wide_only = match std::env::var("CHOPINFER_FOLD_TAIL_WIDE_ONLY") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "1" => true,
        _ => return Err("CHOPINFER_FOLD_TAIL_WIDE_ONLY must be unset or 1".into()),
    };
    let counts = if wide_only {
        if !["q-a", "q-b", "wo"].contains(&group.as_str()) {
            return Err("wide-only qualification requires q-a, q-b or wo".into());
        }
        vec![45, 50, 55, 60]
    } else if group == "drafter35" {
        vec![35]
    } else if group == "calibration" {
        vec![40]
    } else {
        rows()
    };
    let mut cases = 0;
    for rows in counts.iter().copied() {
        let tile = if rows <= 16 {
            16
        } else if rows <= 32 {
            32
        } else {
            64
        };
        for rank in 0..4 {
            let mut record = if group == "wo" || group == "calibration" {
                let first = data::synthetic_rank(&base_of(1024, 4096, 2), rows, rank);
                let second = data::synthetic_rank(&base_of(5120, 2048, 1), rows, rank);
                let input = attention_rows(&first);
                wo_case(
                    s,
                    tile,
                    rows,
                    "synthetic",
                    [
                        &first.weights,
                        &first.weight_scales,
                        &second.weights,
                        &second.weight_scales,
                    ],
                    &input,
                    mode,
                )?
            } else {
                let (id, n, k) = if group == "q-b" {
                    ("attention.q-b", 8192, 1280)
                } else {
                    ("attention.q-a", 1280, 5120)
                };
                let input = data::synthetic_rank(&base_of(n, k, 1), rows, rank);
                status_case(s, id, tile, &input, mode)?
            };
            record["rank"] = json!(rank);
            println!("FOLD_TAIL_CASE {record}");
            cases += 1;
        }
    }
    assert_eq!(cases, counts.len() * 4);
    println!(
        "FOLD_TAIL_DONE {}",
        json!({"group":group,"cases":cases,"rows":counts})
    );
    Ok(())
}

#[test]
fn tail_catalog_uses_current_packed_modules() {
    scoped(|| {
        for rows in rows() {
            let tile = if rows <= 16 {
                16
            } else if rows <= 32 {
                32
            } else {
                64
            };
            for &(id, n, k) in &[("attention.q-a", 1280, 5120), ("attention.q-b", 8192, 1280)] {
                let mut a = Arena::new(None);
                let input = data::empty(&base_of(n, k, 1), rows);
                let case = status_build(&mut a, id, tile, rows, &input).unwrap();
                for call in case
                    .served
                    .iter()
                    .chain(&case.candidates[0].1)
                    .chain(&case.control)
                {
                    call.plan.validate_geometry().unwrap();
                    assert!(
                        fold_cubin_for_symbol(&call.symbol).is_some(),
                        "{}",
                        call.symbol
                    );
                }
                if rows == 35 && id == "attention.q-a" {
                    assert!(
                        case.served
                            .iter()
                            .any(|call| call.symbol.contains("ticket_n1280"))
                    );
                }
            }
            {
                let mut a = Arena::new(None);
                let wo = wo_build(&mut a, tile, rows, [&[]; 4], &[]).unwrap();
                assert_eq!(wo.arms.len(), 3);
                for arm in wo.arms {
                    for call in arm
                        .calls
                        .iter()
                        .chain(arm.controls.iter().flatten().flatten())
                    {
                        call.plan.validate_geometry().unwrap();
                        assert!(
                            fold_cubin_for_symbol(&call.symbol).is_some(),
                            "{}",
                            call.symbol
                        );
                    }
                }
            }
        }
        assert_eq!(rows().len(), 32);
    });
}

#[test]
#[ignore = "requires an idle GPU and explicit fold-tail group and mode"]
fn gpu_tails() {
    let cubins = if mode().unwrap() == Mode::Time {
        let bytes = std::fs::read(root().join("cubins/probe_sm120.cubin")).unwrap();
        let lock: Value = serde_json::from_str(include_str!(
            "../../../../../../crates/kernel-lab/kernels.lock"
        ))
        .unwrap();
        assert!(lock.as_array().unwrap().iter().any(
            |r| r["name"] == "probe_sm120" && r["cubin_sha256"] == kernel_build::digest(&bytes)
        ));
        let bytes = Box::leak(bytes.into_boxed_slice());
        kernel_harness::Cubins::new(
            Box::leak(Box::new([("probe_sm120", &bytes[..])])),
            &[(
                "probe_sm120",
                &["chop_probe_flush_v1", "chop_probe_read_v1"],
            )],
        )
    } else {
        kernel_harness::Cubins::NONE
    };
    let result = scoped(|| kernel_harness::gpu_test_with(cubins, 1 << 30, gpu)).unwrap();
    assert!(result.is_some());
}
