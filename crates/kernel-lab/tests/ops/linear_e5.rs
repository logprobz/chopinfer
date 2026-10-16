//! Focused cold-state recurrence and norm measurements for draft recurrence.
//! The source state is immutable; verify checkpoints and outputs must repeat bitwise.
use super::*;

#[path = "../common/evidence.rs"]
#[allow(dead_code)]
mod evidence;

#[path = "../common/e5.rs"]
mod diagnostics;

use kernel_lab::poison::Phase;

fn checkpoint_map(
    batch: usize,
    per: usize,
    slot_bytes: usize,
    element: DType,
) -> LabResult<WriteMap> {
    // linear/conv.cuh:195-237 and delta.cuh:121-185: live verify rows
    // write disjoint checkpoint slots and never update their source slots.
    WriteMap::of(element).byte_ranges(
        (batch..batch * (1 + per)).map(|slot| slot * slot_bytes..(slot + 1) * slot_bytes),
    )
}

fn bank_checks<'a>(
    s: &'a Session,
    ops: &'a [Operands],
    initial: &[u8],
    map: &WriteMap,
    output_bytes: usize,
) -> LabResult<Vec<Replay<'a>>> {
    ops.iter()
        .enumerate()
        .map(|(bank, op)| {
            let mut replay = Replay::new(s).output(
                format!("output bank {bank}"),
                &op.buffers["output"],
                WriteMap::of(DType::Bf16).byte_prefix(output_bytes)?,
            );
            if !initial.is_empty() {
                replay = replay.state(
                    format!("state bank {bank}"),
                    &op.buffers["state"],
                    initial.to_vec(),
                    map.clone(),
                    Vec::new(),
                );
            }
            Ok(replay)
        })
        .collect()
}

fn checked_banks(
    replays: &mut [Replay<'_>],
    phase: Phase,
    launch: impl FnOnce() -> LabResult<()>,
) -> LabResult<()> {
    for replay in replays.iter() {
        replay.prepare(phase)?;
    }
    launch()?;
    let mut misses = Vec::new();
    for (bank, replay) in replays.iter_mut().enumerate() {
        if let Err(error) = replay.finish(phase, true) {
            misses.push(format!("bank {bank}: {error}"));
        }
    }
    if misses.is_empty() {
        Ok(())
    } else {
        Err(misses.join("; "))
    }
}

fn bank_evidence(replays: &[Replay<'_>]) -> Value {
    json!(
        replays
            .iter()
            .map(|replay| {
                let mut fields = BTreeMap::new();
                replay.record(&mut fields);
                fields
            })
            .collect::<Vec<_>>()
    )
}

fn stamp_map(ctas: usize, per: usize, trace: bool) -> LabResult<WriteMap> {
    // The diagnostic delta generator, mark sites 0..20: globaltimer stores
    // slots 0..4, two per live row, and exit slot 20 of each 21-word CTA.
    WriteMap::of(DType::I64).byte_ranges((0..ctas).flat_map(|cta| {
        (0..if trace { 4 + 2 * per } else { 0 })
            .chain(trace.then_some(20))
            .map(move |slot| (cta * 21 + slot) * 8..(cta * 21 + slot + 1) * 8)
    }))
}

fn prepare_stamps(s: &Session, ops: &[Operands]) -> LabResult<()> {
    for op in ops {
        poison::fill_all(s, &op.buffers["e5_stamps"], &[POISON])?;
    }
    Ok(())
}

fn check_stamps(s: &Session, ops: &[Operands], map: &WriteMap) -> LabResult<()> {
    let mut misses = Vec::new();
    for (bank, op) in ops.iter().enumerate() {
        let bytes = poison::read(s, &op.buffers["e5_stamps"])?;
        if let Err(error) = poison::check(
            &format!("trace bank {bank}"),
            &bytes,
            map,
            Fill::Pattern(&[POISON]),
        ) {
            misses.push(error);
        }
    }
    if misses.is_empty() {
        Ok(())
    } else {
        Err(misses.join("; "))
    }
}

fn flush(s: &Session) -> LabResult<kernel_lab::eviction::Eviction> {
    kernel_lab::eviction::evict(s)
}

#[test]
fn linear_e5() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("linear_e5: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    let shape = std::env::var("CHOPINFER_LINEAR_E5_SHAPE").expect("E5 shape");
    let (batch, per) = match shape.as_str() {
        "16x1" => (16_usize, 1_usize),
        "32x1" => (32, 1),
        "16x2" => (16, 2),
        _ => panic!("expected 16x1, 32x1 or 16x2"),
    };
    let mode = std::env::var("CHOPINFER_LINEAR_E5_MODE").expect("E5 linear mode");
    assert!(["conv", "delta", "norm"].contains(&mode.as_str()));
    let profile = std::env::var("CHOPINFER_E5_PROFILE").as_deref() == Ok("1");
    gpu_test(3 << 30, |s| {
        evidence::exclusive(s)?;
        let g = geometry();
        if g.ranks != 4 {
            return Err("E5 uses TP4 per-rank geometry".into());
        }
        let module = module(s)?;
        let rows = batch * per;
        let pool = batch * (1 + per);
        let checkpoint_ids = (0..batch * verify_row_limit() as usize)
            .map(|i| {
                (batch
                    + (i / verify_row_limit() as usize) * per
                    + (i % verify_row_limit() as usize).min(per - 1)) as i32
            })
            .collect::<Vec<_>>();
        let layout = VerifyLayout {
            lengths: vec![per; batch],
            live: batch,
            slot_ids: (0..batch as i32).collect(),
            checkpoint_ids,
            pool,
            compact: false,
            bucket: Some(Bucket {
                rows: rows as u32,
                sequences: batch as u32,
            }),
        };
        let mut rng = Rng::new(0xe51632);
        // These existing f64 gates inspect every live output and checkpoint.
        // They run outside profiling; the profiler measures already checked kernels.
        if !profile {
            match mode.as_str() {
                "delta" => {
                    delta_verify_run(s, &module, format!("e5_delta_{shape}"), &mut rng, &layout)?
                }
                "conv" => {
                    conv_verify_run(s, &module, format!("e5_conv_{shape}"), &mut rng, &layout)?
                }
                _ => gated_norm(s, &module, rows, 0xe5)?,
            }
        }
        let (mut base, state, output_bytes, name) = match mode.as_str() {
            "delta" => {
                let d = DeltaCase::new(&mut rng, rows, pool);
                (
                    d.operands(s, rows)?,
                    bytes_f32(&d.pool),
                    rows * g.norm_width * 2,
                    "delta_verify",
                )
            }
            "conv" => {
                let d = ConvCase::new(&mut rng, rows, pool);
                (
                    d.operands(s, rows)?,
                    bytes_u16(&d.pool),
                    rows * g.channels * 2,
                    "conv_verify",
                )
            }
            _ => {
                let d = NormCase::new(&mut rng, rows);
                (
                    d.operands(s, rows)?,
                    Vec::new(),
                    rows * g.norm_width * 2,
                    "gated_norm",
                )
            }
        };
        if mode != "norm" {
            base.i32(s, "cu_rows", &cu(&layout.lengths))?;
            base.i32(s, "slot_ids", &layout.slot_ids)?;
            base.i32(s, "checkpoint_ids", &layout.checkpoint_ids)?;
            base.u32(s, "live_sequences", batch as u32)?;
        }
        let kernel = Kernel::load(&module, &symbol(name))?.with_bucket(rows as u32, batch as u32);
        // Delta's distinct committed states exceed L2; every call also writes
        // its own checkpoint pool. The small conv/norm cases use one cold call.
        let calls = if mode == "delta" {
            graph_calls(state_bytes(batch))
        } else {
            1
        };
        let mut ops = Vec::with_capacity(calls);
        for _ in 0..calls {
            let mut op = Operands {
                buffers: base.buffers.clone(),
                scalars: base.scalars.clone(),
            };
            if !state.is_empty() {
                op.bytes(s, "state", &state)?;
            }
            op.zero(s, "output", output_bytes)?;
            ops.push(op);
        }
        let (slot_bytes, state_element) = match mode.as_str() {
            "conv" => (HISTORY * g.channels * 2, DType::Bf16),
            "delta" => (g.delta_state * 4, DType::F32),
            _ => (0, DType::F32),
        };
        let state_map = if state.is_empty() {
            WriteMap::of(DType::F32).none()?
        } else {
            checkpoint_map(batch, per, slot_bytes, state_element)?
        };
        let run = |stream: &Stream, op: &Operands| {
            kernel.launch(s, stream, op, rows as u32, batch as u32, true)
        };
        let graph = s.capture(|stream| ops.iter().try_for_each(|op| run(stream, op)))?;
        let mut fields = BTreeMap::from([
            ("shape".into(), json!(shape)),
            ("mode".into(), json!(mode)),
            ("rows".into(), json!(rows)),
            ("sequences".into(), json!(batch)),
            ("calls_per_graph".into(), json!(calls)),
            (
                "source_state_bytes_per_call".into(),
                json!(if mode == "delta" {
                    state_bytes(batch)
                } else {
                    0.0
                }),
            ),
            ("state_pool_bytes_per_call".into(), json!(state.len())),
            (
                "artifact".into(),
                evidence::identity("linear_qwen38_flash_sm120")?,
            ),
            ("profile_only".into(), json!(profile)),
        ]);
        if profile {
            // No host-released timing gate while Nsight Compute serializes launches.
            let _flush = flush(s)?;
            graph.launch(&s.stream).map_err(|e| e.to_string())?;
            s.stream.synchronize().map_err(|e| e.to_string())?;
        } else {
            let mut replays = bank_checks(s, &ops, &state, &state_map, output_bytes)?;
            checked_banks(&mut replays, Phase::Eager, || {
                ops.iter().try_for_each(|op| run(&s.stream, op))
            })?;
            let expected_output = ops[0].read(s, "output", output_bytes)?;
            let expected_state = if state.is_empty() {
                Vec::new()
            } else {
                ops[0].read(s, "state", state.len())?
            };
            let timing = kernel_lab::time(s, 3, 11, |stream| {
                graph.launch(stream).map_err(|e| e.to_string())
            })?;
            for op in &ops {
                if op.read(s, "output", output_bytes)? != expected_output {
                    return Err("linear E5 graph changed eager output bits".into());
                }
                if !state.is_empty() && op.read(s, "state", state.len())? != expected_state {
                    return Err("linear E5 graph changed eager state or checkpoint bits".into());
                }
            }
            checked_banks(&mut replays, Phase::Graph, || {
                graph.launch(&s.stream).map_err(|e| e.to_string())
            })?;
            checked_banks(&mut replays, Phase::Relaunch, || {
                ops.iter().try_for_each(|op| run(&s.stream, op))
            })?;
            checked_banks(&mut replays, Phase::Pdl, || {
                ops.iter().try_for_each(|op| run(&s.stream, op))
            })?;
            fields.insert("poisoned_banks".into(), bank_evidence(&replays));
            fields.insert("eager_graph_bits_equal".into(), json!(true));
            fields.insert(
                "median_us_per_call".into(),
                json!(f64::from(timing.median_ms) * 1000.0 / calls as f64),
            );
            fields.insert(
                "min_us_per_call".into(),
                json!(f64::from(timing.min_ms) * 1000.0 / calls as f64),
            );
            fields.insert(
                "cache_policy".into(),
                json!("selected L2 eviction before graph; distinct state per call"),
            );
        }
        if mode == "delta"
            && let Some(diagnostic_module) = diagnostics::module(s)?
        {
            let ctas = 8 * g.value_heads * batch;
            for op in &mut ops {
                op.zero(s, "e5_stamps", ctas * 21 * 8)?;
            }
            let mut shipped = bank_checks(s, &ops, &state, &state_map, output_bytes)?;
            checked_banks(&mut shipped, Phase::Eager, || {
                ops.iter().try_for_each(|op| run(&s.stream, op))
            })?;
            let expected_output = ops[0].read(s, "output", output_bytes)?;
            let expected_state = ops[0].read(s, "state", state.len())?;
            let mut failures = Vec::new();
            for variant in [
                "control",
                "stream_rows",
                "no_state_loads",
                "no_checkpoints",
                "trace",
                "control_after",
            ] {
                let result = (|| -> LabResult<()> {
                    let symbol = format!(
                        "e5_delta_{}",
                        if variant == "control_after" {
                            "control"
                        } else {
                            variant
                        }
                    );
                    let function = diagnostic_module
                        .function(&symbol)
                        .map_err(|e| e.to_string())?;
                    let diagnostic_run = |stream: &Stream, op: &Operands| {
                        let mut args = kernel
                            .entry
                            .params
                            .iter()
                            .map(|param| match param {
                                EntryParam::DevicePointer { name, .. } => op
                                    .buffers
                                    .get(name.as_str())
                                    .map(runtime_cuda::KernelArgument::Buffer)
                                    .ok_or_else(|| format!("missing {name}")),
                                EntryParam::U32 { name } => op
                                    .scalars
                                    .get(name.as_str())
                                    .copied()
                                    .map(runtime_cuda::KernelArgument::U32)
                                    .ok_or_else(|| format!("missing {name}")),
                                _ => Err("unexpected delta diagnostic argument".into()),
                            })
                            .collect::<LabResult<Vec<_>>>()?;
                        args.push(runtime_cuda::KernelArgument::Buffer(
                            &op.buffers["e5_stamps"],
                        ));
                        let mut plan = kernel
                            .entry
                            .plan(kernel.bucket, rows as u32, batch as u32)
                            .map_err(|e| e.to_string())?;
                        plan.pdl = true;
                        function
                            .launch(stream, launch::config(plan)?, &args)
                            .map_err(|e| e.to_string())
                    };
                    let diagnostic_graph = s.capture(|stream| {
                        ops.iter().try_for_each(|op| diagnostic_run(stream, op))
                    })?;
                    let map = if variant == "no_checkpoints" {
                        WriteMap::of(state_element).none()?
                    } else {
                        state_map.clone()
                    };
                    let mut replays = bank_checks(s, &ops, &state, &map, output_bytes)?;
                    let stamps = stamp_map(ctas, per, variant == "trace")?;
                    prepare_stamps(s, &ops)?;
                    checked_banks(&mut replays, Phase::Eager, || {
                        ops.iter().try_for_each(|op| diagnostic_run(&s.stream, op))
                    })?;
                    check_stamps(s, &ops, &stamps)?;
                    let timing = kernel_lab::time(s, 3, 11, |stream| {
                        diagnostic_graph.launch(stream).map_err(|e| e.to_string())
                    })?;
                    // A fresh execution checks every bank; timing never certifies stores.
                    prepare_stamps(s, &ops)?;
                    checked_banks(&mut replays, Phase::Graph, || {
                        diagnostic_graph
                            .launch(&s.stream)
                            .map_err(|e| e.to_string())
                    })?;
                    check_stamps(s, &ops, &stamps)?;
                    let mut equal = true;
                    for op in &ops {
                        equal &= op.read(s, "output", output_bytes)? == expected_output;
                        equal &= op.read(s, "state", state.len())? == expected_state;
                    }
                    if ["control", "trace", "control_after"].contains(&variant) && !equal {
                        return Err(format!(
                            "delta diagnostic {variant} differs from the shipped kernel"
                        ));
                    }
                    let mut record = BTreeMap::from([
                        ("family".into(), json!("delta")),
                        ("diagnostic_artifact".into(), diagnostics::artifact()?),
                        ("poisoned_banks".into(), bank_evidence(&replays)),
                        ("shape".into(), json!(shape)),
                        ("variant".into(), json!(variant)),
                        ("diagnostic_only".into(), json!(true)),
                        ("matches_shipped_bits".into(), json!(equal)),
                        ("calls_per_graph".into(), json!(calls)),
                        (
                            "median_us_per_call".into(),
                            json!(f64::from(timing.median_ms) * 1000.0 / calls as f64),
                        ),
                        (
                            "min_us_per_call".into(),
                            json!(f64::from(timing.min_ms) * 1000.0 / calls as f64),
                        ),
                    ]);
                    if variant == "trace" {
                        let mut stages = [
                            (0, "entry"),
                            (1, "dependency_wait_done"),
                            (2, "state_loads_issued"),
                            (3, "operand_loads_issued"),
                        ]
                        .into_iter()
                        .map(|(i, n)| (i, n.to_owned()))
                        .collect::<Vec<_>>();
                        for row in 0..per {
                            stages.push((4 + 2 * row, format!("row_{row}_math_done")));
                            stages
                                .push((5 + 2 * row, format!("row_{row}_checkpoint_stores_issued")));
                        }
                        stages.push((20, "exit".into()));
                        let traces = ops
                            .iter()
                            .map(|op| {
                                diagnostics::trace(s, &op.buffers["e5_stamps"], ctas, 21, &stages)
                            })
                            .collect::<LabResult<Vec<_>>>()?;
                        record.insert("trace".into(), traces[0].clone());
                        record.insert("trace_banks".into(), json!(traces));
                    }
                    evidence::receipt(s, "qwen38-e5-diagnostic", record)?;

                    Ok(())
                })();
                if let Err(error) = result {
                    eprintln!("diagnostic {variant}: {error}");
                    failures.push(format!("{variant}: {error}"));
                }
            }
            if !failures.is_empty() {
                return Err(failures.join("; "));
            }
        }
        evidence::receipt(s, "qwen38-linear-e5", fields.clone())?;
        eprintln!(
            "E5 {}",
            serde_json::to_string(&fields).map_err(|e| e.to_string())?
        );
        Ok(())
    })
    .expect("linear E5")
    .expect("one assigned GPU is required");
}

#[test]
fn e5_checkpoint_and_stamp_contract_cpu() {
    let map = checkpoint_map(2, 2, 8, DType::F32).unwrap();
    assert_eq!(map.ranges().len(), 1);
    assert_eq!(map.ranges()[0], 16..48);
    let map = stamp_map(2, 2, true).unwrap();
    assert_eq!(map.written_bytes(), 2 * 9 * 8);
    let mut bytes = vec![POISON; 2 * 21 * 8];
    for range in map.ranges() {
        bytes[range.clone()].fill(1);
    }
    assert!(poison::check("trace", &bytes, &map, Fill::Pattern(&[POISON])).is_ok());
    bytes[20 * 8..21 * 8].fill(POISON);
    assert!(poison::check("missing exit", &bytes, &map, Fill::Pattern(&[POISON])).is_err());
    assert_eq!(stamp_map(2, 2, false).unwrap().written_bytes(), 0);
}
