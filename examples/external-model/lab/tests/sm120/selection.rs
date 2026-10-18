//! The example's SM120 selection on a GPU, through the SDK's kernel
//! harness: the locked artifacts of a verified build, the ops as the bundle
//! prepares them, their oracles, live-row writes and graph replay.
//! Explicit invocation requires the report even when no GPU is selected.

#[path = "../../../bundle/tests/sm120/cases.rs"]
mod cases;
#[path = "../../../bundle/kernel_cases.rs"]
mod kernel_cases;
#[path = "../../../bundle/src/sm120/own_ops.rs"]
mod own_ops;
#[path = "../../../bundle/src/sm120/selection.rs"]
mod selection;

use example_contract::ops;

use kernel_build::BuildReport;
use kernel_core::backend::{Counts, Launch, View};
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::DType;
use kernel_core::rmsnorm::RmsNormBuffers;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::poison::{Replay, WriteMap};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

fn artifacts() -> BuildReport {
    let path = std::env::var_os("CHOPINFER_MODEL_SELECTION_REPORT").expect(
        "set CHOPINFER_MODEL_SELECTION_REPORT to the exported example's verified build report",
    );
    let report = BuildReport::read(Path::new(&path)).expect("read external selection report");
    let context = report.context.as_ref().expect("model ownership");
    let root = |id| {
        std::fs::canonicalize(&context.sources.iter().find(|s| s.id == id).unwrap().path)
            .expect("reported source root exists")
    };
    let selected = selection::selection(
        &root("external-model"),
        &root("chopinfer").join("crates/kernels-sm120/cuda"),
    )
    .expect("reconstruct model selection");
    cases::check_artifacts(&report, &selected);
    cases::check_refusals(&selected);
    report
}

/// The example's ops as its bundle prepares them on SM120.
fn prepared() -> ops::Ops {
    ops::Ops::prepare(&selection::backend(), &own_ops::Sm120Own).expect("prepare the example's ops")
}

fn view(handle: u64, bytes: usize) -> View {
    View {
        handle,
        bytes: bytes as u64,
    }
}

/// The one launch of a prepared op.
fn single(launches: Vec<Launch>) -> kernel_harness::LabResult<Launch> {
    match <[Launch; 1]>::try_from(launches) {
        Ok([launch]) => Ok(launch),
        Err(launches) => Err(format!("expected one launch, got {}", launches.len())),
    }
}

#[test]
fn exported_contracts_and_model_cases() {
    let report = artifacts();
    for live in kernel_cases::LIVE_ROWS {
        let reference = kernel_cases::rms_reference(live).unwrap();
        assert_eq!(
            reference.map_or(0, |r| r.values.len()),
            live as usize * ops::WIDTH
        );
    }
    assert_eq!(
        report
            .cubins
            .iter()
            .map(|c| c.declarations.len())
            .sum::<usize>(),
        2
    );
    // The ops as the bundle prepares them launch only reported entries.
    let ops = prepared();
    let declared = report
        .cubins
        .iter()
        .flat_map(|c| &c.declarations)
        .map(|d| d.symbol.as_str())
        .collect::<Vec<_>>();
    for (symbol, _) in ops
        .rmsnorm
        .info()
        .symbols
        .iter()
        .chain(&ops.probe.info().symbols)
    {
        assert!(declared.contains(&symbol.as_str()), "{symbol}");
    }
    eprintln!(
        "verified two compiled entries, the prepared ops' symbols, four model-owned RMSNorm cases and incompatible-contract refusals"
    );
}

#[test]
fn exported_gpu_oracles_and_graph_replay() {
    let report = artifacts();
    let result = kernel_harness::gpu_test(16 * 1024 * 1024, |session| {
        let load = |name| {
            let path = &report
                .cubins
                .iter()
                .find(|c| c.name == name)
                .ok_or("missing example cubin")?
                .path;
            session.load_cubin(&std::fs::read(path).map_err(|e| e.to_string())?)
        };
        let ops = prepared();
        let module = load("external_rmsnorm_v1")?;
        let x = session.upload_bf16(&kernel_cases::input())?;
        let weight = session.upload_bf16(&kernel_cases::weights())?;
        let count = ops::WIDTH * ops::CAPACITY as usize;
        let y = session.upload_bf16(&vec![kernel_cases::PADDING; count])?;
        let live = session.allocate(4)?;
        let bindings = Bindings {
            buffers: BTreeMap::from([(1, &x), (2, &weight), (3, &y), (4, &live)]),
        };
        // The graph launches the capacity bucket; the device count selects
        // the live rows.
        let norm = single(
            ops.rmsnorm
                .launches(
                    &RmsNormBuffers {
                        x: view(1, count * 2),
                        weight: view(2, ops::WIDTH * 2),
                        residual: None,
                        y: view(3, count * 2),
                        live_rows: view(4, 4),
                        status: None,
                        status_row_offset: 0,
                    },
                    View::EMPTY,
                    Counts::rows(ops::CAPACITY),
                )
                .map_err(|e| e.to_string())?,
        )?;
        let function = module.function(&norm.symbol).map_err(|e| e.to_string())?;
        let graph = session.capture(|stream| {
            launch(
                &session.context,
                &function,
                stream,
                norm.plan,
                &norm.arguments,
                &bindings,
            )
        })?;
        let mut misses = Vec::new();
        for rows in kernel_cases::LIVE_ROWS {
            let checked = (|| -> kernel_harness::LabResult<()> {
                live.copy_from_bytes(&rows.to_ne_bytes(), &session.stream)
                    .map_err(|e| e.to_string())?;
                let written = rows as usize * ops::WIDTH;
                let mut replay = Replay::new(session).output(
                    "RMSNorm",
                    &y,
                    WriteMap::of(DType::Bf16).prefix(written)?,
                );
                replay.eager(|| {
                    launch(
                        &session.context,
                        &function,
                        &session.stream,
                        norm.plan,
                        &norm.arguments,
                        &bindings,
                    )
                })?;
                for _ in 0..3 {
                    replay.graph(&graph)?;
                }
                let actual = session.download_bf16(&y, count)?;
                let written = rows as usize * ops::WIDTH;
                if actual[written..].iter().any(|&v| v != 0xffff) {
                    return Err("RMSNorm wrote outside live rows".into());
                }
                let mut receipt = BTreeMap::from([
                    ("case".into(), json!(selection::RMS_CASE)),
                    ("live_rows".into(), json!(rows)),
                    ("graph_replays".into(), json!(3)),
                    ("mechanics_only".into(), json!(true)),
                    ("qualified".into(), json!(false)),
                ]);
                replay.record(&mut receipt);
                if let Some(reference) =
                    kernel_cases::rms_reference(rows).map_err(|e| e.to_string())?
                {
                    let values = actual[..written]
                        .iter()
                        .map(|&v| bf16_to_f32(v))
                        .collect::<Vec<_>>();
                    kernel_harness::check_output(&values, &reference, DType::Bf16, &mut receipt)?;
                }
                session.receipt("model-owned-selection", receipt)?;

                Ok(())
            })();
            if let Err(error) = checked {
                eprintln!("live rows={rows}: {error}");
                misses.push(error);
            }
        }
        let module = load("external_probe")?;
        let result = session.allocate(4)?;
        let bindings = Bindings {
            buffers: BTreeMap::from([(1, &result)]),
        };
        for value in kernel_cases::PROBE_VALUES {
            let checked = (|| -> kernel_harness::LabResult<()> {
                let probe = single(
                    ops.probe
                        .launches(
                            &ops::ProbeBuffers {
                                result: view(1, 4),
                                value,
                            },
                            View::EMPTY,
                            Counts::rows(1),
                        )
                        .map_err(|e| e.to_string())?,
                )?;
                let function = module.function(&probe.symbol).map_err(|e| e.to_string())?;
                let graph = session.capture(|stream| {
                    launch(
                        &session.context,
                        &function,
                        stream,
                        probe.plan,
                        &probe.arguments,
                        &bindings,
                    )
                })?;
                let mut replay = Replay::new(session).output(
                    "scalar",
                    &result,
                    WriteMap::of(DType::F32).prefix(1)?,
                );
                replay.eager(|| {
                    launch(
                        &session.context,
                        &function,
                        &session.stream,
                        probe.plan,
                        &probe.arguments,
                        &bindings,
                    )
                })?;
                for _ in 0..3 {
                    replay.graph(&graph)?;
                }
                let actual = f64::from(session.download_f32(&result, 1)?[0]);
                let expected = ops::sigmoid_probe(value);
                if !actual.is_finite() || (actual - expected).abs() > ops::PROBE_ABS_BOUND {
                    return Err(format!(
                        "local sigmoid oracle failed for {value}: {actual} vs {expected}"
                    ));
                }
                let mut receipt = BTreeMap::from([
                    ("case".into(), json!(selection::PROBE_CASE)),
                    ("value".into(), json!(value)),
                    ("actual".into(), json!(actual)),
                    ("expected".into(), json!(expected)),
                    ("bound".into(), json!(ops::PROBE_ABS_BOUND)),
                    ("graph_replays".into(), json!(3)),
                    ("mechanics_only".into(), json!(true)),
                    ("qualified".into(), json!(false)),
                ]);
                replay.record(&mut receipt);
                session.receipt("model-owned-selection", receipt)?;

                Ok(())
            })();
            if let Err(error) = checked {
                eprintln!("value={value}: {error}");
                misses.push(error);
            }
        }
        if misses.is_empty() {
            Ok(())
        } else {
            Err(misses.join("; "))
        }
    })
    .expect("external artifact GPU cases");
    if result.is_none() {
        eprintln!(
            "UNRUN: model-owned GPU oracles, live-row writes and graph replay; CPU binding checks passed"
        );
    }
}
