//! E5: cold-weight HC input, fused up mix and their dependent PDL chain.
//! This measures the shipped kernels; it does not select a new serving path.

use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, DType, LaunchPlan};
use kernel_core::transform::Transform;
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, gpu_test};
use kernels_sm120::gemm::fused as gemm;
use runtime_cuda::{DeviceBuffer, Function, Stream};
use serde_json::json;
use std::collections::BTreeMap;

#[path = "common/evidence.rs"]
#[allow(dead_code)]
mod evidence;

#[path = "common/e5.rs"]
mod diagnostics;

use kernel_lab::poison as replay;
use kernel_lab::poison::{Phase, WriteMap};

fn call_checks<'a>(
    s: &'a Session,
    buffers: &'a [BTreeMap<u64, DeviceBuffer>],
    rows: usize,
    mode: &str,
    workspace: usize,
) -> LabResult<Vec<replay::Checks<'a>>> {
    buffers
        .iter()
        .map(|call| {
            let mut checks = replay::Checks::new(s);
            if mode != "mix" {
                checks = checks
                    .output(
                        "down",
                        &call[&DOWN],
                        WriteMap::of(DType::Bf16).prefix(rows * RANK)?,
                    )
                    .output(
                        "inject",
                        &call[&INJECT],
                        WriteMap::of(DType::Bf16).prefix(rows * 4)?,
                    );
                if workspace != 0 {
                    checks = checks.scratch_map("input partials", &call[&PARTIAL], 0, move |_| {
                        WriteMap::of(DType::F32).byte_prefix(workspace)
                    });
                }
            }
            if mode != "in" {
                checks = checks.output(
                    "mix",
                    &call[&OUTPUT],
                    WriteMap::of(DType::Bf16).prefix(rows * 2560)?,
                );
            }
            Ok(checks)
        })
        .collect()
}

const WIDE: usize = 10240;
const RANK: usize = 320;
const CALLS: usize = 32;
const X: u64 = 1;
const DOWN: u64 = 2;
const INJECT: u64 = 3;
const OUTPUT: u64 = 4;
const LIVE: u64 = 5;
const PARTIAL: u64 = 6;
const SEMAPHORES: u64 = 7;
const DOWN_WEIGHT: u64 = 10;
const INJECT_WEIGHT: u64 = 11;
const UP_WEIGHT: u64 = 12;

fn values(n: usize, seed: usize) -> Vec<u16> {
    (0..n)
        .map(|i| f32_to_bf16((((i * 17 + seed) % 511) as f32 - 255.0) / 4096.0))
        .collect()
}
fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
fn upload(s: &Session, v: &[u8]) -> LabResult<DeviceBuffer> {
    let b = s.allocate(v.len().max(4))?;
    b.copy_from_bytes(v, &s.stream).map_err(|e| e.to_string())?;
    Ok(b)
}
fn weight(rows: usize, k: usize, seed: usize) -> LabResult<Vec<u8>> {
    FragmentWeightLayout {
        rows,
        k,
        format: FragmentWeightFormat::Bf16,
    }
    .pack(&bytes(&values(rows * k, seed)))
    .map(|p| p.bytes)
    .map_err(|e| e.to_string())
}
fn load(s: &Session, symbol: &str, plan: LaunchPlan) -> LabResult<Function> {
    let f = s
        .module("gemm_qwen38_own_sm120")?
        .function(symbol)
        .map_err(|e| e.to_string())?;
    if plan.dynamic_shared_bytes >= 48 * 1024 {
        f.set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
            .map_err(|e| e.to_string())?;
    }
    Ok(f)
}

// Profiling cannot use time()'s host-released gate: profiler serialization
// could wait for that gate before the host can release it. Flush, then launch.
fn flush(s: &Session) -> LabResult<kernel_lab::eviction::Eviction> {
    kernel_lab::eviction::evict(s)
}

#[test]
fn qwen38_hc_e5() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        return;
    }
    let rows: u32 = std::env::var("CHOPINFER_HC_E5_ROWS")
        .expect("E5 rows")
        .parse()
        .expect("integer rows");
    assert!([16, 32].contains(&rows));
    let mode = std::env::var("CHOPINFER_HC_E5_MODE").expect("E5 mode");
    assert!(["in", "mix", "chain"].contains(&mode.as_str()));
    let profile = std::env::var("CHOPINFER_E5_PROFILE").as_deref() == Ok("1");
    gpu_test(2 << 30, |s| {
        evidence::exclusive(s)?;
        let entry = gemm::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|e| e.symbol == format!("chop_gemm_qwen38_hc_in_c{rows}_v1"))
            .ok_or("missing HC input entry")?;
        let in_plan = entry.plan(rows).map_err(|e| e.to_string())?;
        let hc = entry.hc.ok_or("missing HC geometry")?;
        let mix_plan = gemm::hc_up_mix_plan(hc, rows).map_err(|e| e.to_string())?;
        let input = load(s, &entry.symbol, in_plan)?;
        let mix = load(s, hc.up_mix_symbol, mix_plan)?;
        let mut b = BTreeMap::new();
        b.insert(X, upload(s, &bytes(&values(rows as usize * WIDE, 7)))?);
        b.insert(DOWN, upload(s, &bytes(&values(rows as usize * RANK, 11)))?);
        b.insert(INJECT, upload(s, &vec![0; rows as usize * 4 * 2])?);
        b.insert(OUTPUT, upload(s, &vec![0; rows as usize * 2560 * 2])?);
        b.insert(LIVE, upload(s, &rows.to_ne_bytes())?);
        b.insert(
            PARTIAL,
            s.allocate(in_plan.workspace_bytes.max(4) as usize)?,
        );
        b.insert(
            SEMAPHORES,
            upload(s, &vec![0; entry.semaphore_words().max(1) as usize * 4])?,
        );
        let packed = [
            weight(RANK, WIDE, 13)?,
            weight(4, WIDE, 17)?,
            weight(WIDE, RANK, 19)?,
        ];
        let banks = (0..CALLS)
            .map(|_| {
                packed
                    .iter()
                    .map(|p| upload(s, p))
                    .collect::<LabResult<Vec<_>>>()
            })
            .collect::<LabResult<Vec<_>>>()?;
        let in_args = entry
            .arguments(
                rows,
                gemm::Buffers {
                    weights: [DOWN_WEIGHT, INJECT_WEIGHT, 0, 0],
                    activation: X,
                    outputs: [DOWN, INJECT, 0, 0],
                    output_strides: [RANK as u32, 4, 0, 0],
                    partial: PARTIAL,
                    semaphores: SEMAPHORES,
                },
            )
            .map_err(|e| e.to_string())?;
        let mix_args = gemm::hc_up_mix_arguments(UP_WEIGHT, DOWN, X, OUTPUT, LIVE)
            .map_err(|e| e.to_string())?;
        let calls = (0..CALLS)
            .map(|_| {
                let mut call = BTreeMap::new();
                for id in [DOWN, INJECT, OUTPUT] {
                    call.insert(id, upload(s, &kernel_lab::poison::read(s, &b[&id])?)?);
                }
                call.insert(
                    PARTIAL,
                    s.allocate(in_plan.workspace_bytes.max(4) as usize)?,
                );
                call.insert(
                    SEMAPHORES,
                    upload(s, &vec![0; entry.semaphore_words().max(1) as usize * 4])?,
                );
                Ok(call)
            })
            .collect::<LabResult<Vec<BTreeMap<u64, DeviceBuffer>>>>()?;
        let launch_one = |stream: &Stream, bank: usize, only_input: bool| -> LabResult<()> {
            let mut buffers = b.iter().map(|(&id, b)| (id, b)).collect::<BTreeMap<_, _>>();
            buffers.extend(calls[bank].iter().map(|(&id, buffer)| (id, buffer)));
            for (i, weight) in banks[bank].iter().enumerate() {
                buffers.insert(DOWN_WEIGHT + i as u64, weight);
            }
            let bindings = Bindings { buffers };
            if mode != "mix" || only_input {
                launch::launch(&s.context, &input, stream, in_plan, &in_args, &bindings)?;
            }
            if mode != "in" && !only_input {
                launch::launch(&s.context, &mix, stream, mix_plan, &mix_args, &bindings)?;
            }
            Ok(())
        };
        if mode == "mix" {
            let mut producer_checks = call_checks(
                s,
                &calls,
                rows as usize,
                "in",
                in_plan.workspace_bytes as usize,
            )?;
            replay::phase_all(&mut producer_checks, Phase::Eager, false, || {
                (0..CALLS).try_for_each(|bank| launch_one(&s.stream, bank, true))
            })?;
        }
        let run = |stream: &Stream, bank: usize| launch_one(stream, bank, false);
        let graph = s.capture(|stream| (0..CALLS).try_for_each(|i| run(stream, i)))?;
        let mut fields = BTreeMap::from([
            ("rows".into(), json!(rows)),
            ("mode".into(), json!(mode)),
            ("calls_per_graph".into(), json!(CALLS)),
            (
                "weight_bank_bytes".into(),
                json!(packed.iter().map(Vec::len).sum::<usize>() * CALLS),
            ),
            (
                "artifact".into(),
                evidence::identity("gemm_qwen38_own_sm120")?,
            ),
            ("profile_only".into(), json!(profile)),
        ]);
        if profile {
            let _flush = flush(s)?;
            graph.launch(&s.stream).map_err(|e| e.to_string())?;
            s.stream.synchronize().map_err(|e| e.to_string())?;
        } else {
            let mut checks = call_checks(
                s,
                &calls,
                rows as usize,
                &mode,
                in_plan.workspace_bytes as usize,
            )?;
            replay::phase_all(&mut checks, Phase::Eager, false, || {
                (0..CALLS).try_for_each(|bank| run(&s.stream, bank))
            })?;
            for id in [DOWN, INJECT, OUTPUT] {
                if (mode == "mix" && id != OUTPUT) || (mode == "in" && id == OUTPUT) {
                    continue;
                }
                let expected = kernel_lab::poison::read(s, &calls[0][&id])?;
                for (bank, call) in calls.iter().enumerate().skip(1) {
                    if kernel_lab::poison::read(s, &call[&id])? != expected {
                        return Err(format!(
                            "HC eager bank {bank} buffer {id} differs from bank 0"
                        ));
                    }
                }
            }
            replay::phase_all(&mut checks, Phase::Graph, true, || {
                graph.launch(&s.stream).map_err(|e| e.to_string())
            })?;
            replay::phase_all(&mut checks, Phase::Pdl, true, || {
                (0..CALLS).try_for_each(|bank| run(&s.stream, bank))
            })?;
            s.stream.synchronize().map_err(|e| e.to_string())?;
            let result_id = if mode == "in" { DOWN } else { OUTPUT };
            let len = rows as usize * if mode == "in" { RANK } else { 2560 };
            let expected = s.download_bf16(&calls[0][&result_id], len)?;
            let timing = kernel_lab::time(s, 3, 11, |stream| {
                graph.launch(stream).map_err(|e| e.to_string())
            })?;
            replay::phase_all(&mut checks, Phase::Graph, true, || {
                graph.launch(&s.stream).map_err(|e| e.to_string())
            })?;
            let actual = s.download_bf16(&calls[0][&result_id], len)?;
            let mut call_evidence = Vec::new();
            for check in &checks {
                let mut record = BTreeMap::new();
                check.record(&mut record);
                call_evidence.push(record);
            }
            fields.insert("poisoned_calls".into(), json!(call_evidence));
            if actual != expected {
                return Err("E5 graph changed eager output bits".into());
            }
            fields.insert(
                "median_us_per_call".into(),
                json!(f64::from(timing.median_ms) * 1000.0 / CALLS as f64),
            );
            fields.insert(
                "min_us_per_call".into(),
                json!(f64::from(timing.min_ms) * 1000.0 / CALLS as f64),
            );
            fields.insert("eager_graph_bits_equal".into(), json!(true));
            fields.insert(
                "cache_policy".into(),
                json!("selected L2 eviction before graph; 32 distinct weight banks"),
            );
        }
        if mode != "in"
            && let Some(module) = diagnostics::module(s)?
        {
            let trace_bytes = 160 * 7 * 8;
            let traces = (0..CALLS)
                .map(|_| upload(s, &vec![0; trace_bytes]))
                .collect::<LabResult<Vec<_>>>()?;
            // Obtain the shipped reference after any previous timed replay.
            let mut shipped = call_checks(
                s,
                &calls,
                rows as usize,
                &mode,
                in_plan.workspace_bytes as usize,
            )?;
            replay::phase_all(&mut shipped, Phase::Eager, false, || {
                (0..CALLS).try_for_each(|bank| run(&s.stream, bank))
            })?;
            let expected = s.download_bf16(&calls[0][&OUTPUT], rows as usize * 2560)?;
            let mut failures = replay::Failures::default();
            for variant in [
                "control",
                "no_weight_loads",
                "no_bottleneck",
                "no_mma",
                "trace",
                "control_after",
            ] {
                let result = (|| -> LabResult<()> {
                    let symbol = format!(
                        "e5_hc_{}",
                        if variant == "control_after" {
                            "control"
                        } else {
                            variant
                        }
                    );
                    let function = module.function(&symbol).map_err(|e| e.to_string())?;
                    let mut args = mix_args.clone();
                    args.0.push(Argument::DeviceBuffer(20));
                    let diagnostic_run = |stream: &Stream, bank: usize| {
                        let mut buffers =
                            b.iter().map(|(&id, b)| (id, b)).collect::<BTreeMap<_, _>>();
                        buffers.extend(calls[bank].iter().map(|(&id, buffer)| (id, buffer)));
                        for (i, weight) in banks[bank].iter().enumerate() {
                            buffers.insert(DOWN_WEIGHT + i as u64, weight);
                        }
                        buffers.insert(20, &traces[bank]);
                        let bindings = Bindings { buffers };
                        if mode == "chain" {
                            launch::launch(
                                &s.context, &input, stream, in_plan, &in_args, &bindings,
                            )?;
                        }
                        launch::launch(&s.context, &function, stream, mix_plan, &args, &bindings)
                    };
                    let diagnostic_graph =
                        s.capture(|stream| (0..CALLS).try_for_each(|i| diagnostic_run(stream, i)))?;
                    let mut checks = call_checks(
                        s,
                        &calls,
                        rows as usize,
                        &mode,
                        in_plan.workspace_bytes as usize,
                    )?;
                    for (bank, check) in checks.iter_mut().enumerate() {
                        let old = std::mem::replace(check, replay::Checks::new(s));
                        let written = if variant == "trace" { trace_bytes } else { 0 };
                        *check =
                            old.scratch_map("diagnostic stamps", &traces[bank], 0, move |_| {
                                WriteMap::of(DType::I64).byte_prefix(written)
                            });
                    }
                    replay::phase_all(&mut checks, Phase::Eager, false, || {
                        (0..CALLS).try_for_each(|bank| diagnostic_run(&s.stream, bank))
                    })?;
                    let timing = kernel_lab::time(s, 3, 11, |stream| {
                        diagnostic_graph.launch(stream).map_err(|e| e.to_string())
                    })?;
                    replay::phase_all(&mut checks, Phase::Graph, true, || {
                        diagnostic_graph
                            .launch(&s.stream)
                            .map_err(|e| e.to_string())
                    })?;
                    let equal = (0..CALLS)
                        .map(|bank| s.download_bf16(&calls[bank][&OUTPUT], rows as usize * 2560))
                        .collect::<LabResult<Vec<_>>>()?
                        .iter()
                        .all(|actual| actual == &expected);
                    if ["control", "trace", "control_after"].contains(&variant) && !equal {
                        return Err(format!(
                            "HC diagnostic {variant} differs from the shipped kernel"
                        ));
                    }
                    let mut record = BTreeMap::from([
                        ("family".into(), json!("hc")),
                        ("rows".into(), json!(rows)),
                        ("mode".into(), json!(mode)),
                        ("variant".into(), json!(variant)),
                        ("diagnostic_only".into(), json!(true)),
                        ("matches_shipped_bits".into(), json!(equal)),
                        ("calls_per_graph".into(), json!(CALLS)),
                        (
                            "median_us_per_call".into(),
                            json!(f64::from(timing.median_ms) * 1000.0 / CALLS as f64),
                        ),
                        (
                            "min_us_per_call".into(),
                            json!(f64::from(timing.min_ms) * 1000.0 / CALLS as f64),
                        ),
                    ]);
                    record.insert("diagnostic_artifact".into(), diagnostics::artifact()?);
                    let mut call_evidence = Vec::new();
                    for check in &checks {
                        let mut fields = BTreeMap::new();
                        check.record(&mut fields);
                        call_evidence.push(fields);
                    }
                    record.insert("poisoned_calls".into(), json!(call_evidence));
                    if variant == "trace" {
                        let stages = [
                            "entry",
                            "weight_loads_issued",
                            "dependency_wait_done",
                            "bottleneck_ready",
                            "mma_done",
                            "mix_terms_ready",
                            "output_stores_issued",
                        ]
                        .into_iter()
                        .enumerate()
                        .map(|(i, n)| (i, n.to_owned()))
                        .collect::<Vec<_>>();
                        for trace in &traces[1..] {
                            diagnostics::trace(s, trace, 160, 7, &stages)?;
                        }
                        record.insert(
                            "trace".into(),
                            diagnostics::trace(s, &traces[0], 160, 7, &stages)?,
                        );
                    }
                    evidence::receipt(s, "qwen38-e5-diagnostic", record)?;
                    Ok(())
                })();
                failures.check(&format!("HC diagnostic {variant}"), result);
            }
            failures.finish()?;
        }
        evidence::receipt(s, "qwen38-hc-e5", fields.clone())?;
        eprintln!(
            "E5 {}",
            serde_json::to_string(&fields).map_err(|e| e.to_string())?
        );
        Ok(())
    })
    .expect("HC E5")
    .expect("one assigned GPU is required");
}
