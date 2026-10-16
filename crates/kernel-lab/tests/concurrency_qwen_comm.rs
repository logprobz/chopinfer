use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, SigmoidMeanHyperConnectionQuery};
use kernel_core::hyper_connection_oracle;
use kernel_lab::{LabResult, Session, check_output, gpu_test_multi, launch};
use model_qwen38_contract::sm120::comm as qwen38;
use runtime_cuda::DeviceBuffer;
use serde_json::json;
use std::collections::BTreeMap;

use kernel_lab::poison::{Phase, Replay, WriteMap};

fn codes(len: usize, seed: usize) -> Vec<u16> {
    (0..len)
        .map(|i| f32_to_bf16((((i * 31 + seed * 73) % 509) as f32 - 254.0) / 128.0))
        .collect()
}
fn decoded(values: &[u16]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(bf16_to_f32(v))).collect()
}
fn bytes(s: &Session, data: &[u8]) -> LabResult<DeviceBuffer> {
    let b = s.allocate(data.len())?;
    b.copy_from_bytes(data, &s.stream)
        .map_err(|e| e.to_string())?;
    Ok(b)
}

const SYMBOLS: [&str; 3] = [
    qwen38::SYMBOL,
    qwen38::SCATTER_SYMBOL,
    qwen38::GATHER_SYMBOL,
];

/// The one-shot entry at the call's full capacity on a state of its own,
/// the oracle both paths must match bit for bit.
fn standalone(call: qwen38::Call) -> Result<Vec<qwen38::Launch>, kernel_core::contract::Refusal> {
    let query = SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        ranks: 4,
        streams: 4,
        width: 2560,
        eps_bits: 1e-6_f32.to_bits(),
        capacity: call.capacity,
    };
    Ok(vec![qwen38::Launch {
        symbol: qwen38::SYMBOL,
        plan: qwen38::plan(query, call.capacity)?,
        arguments: qwen38::standalone_arguments(call)?,
    }])
}

/// Which all-reduce a call runs: the bundle's choice by rows, or the
/// one-shot entry at the full capacity on the separate legacy state.
#[derive(Clone, Copy)]
enum Path {
    Bundle(u32),
    Legacy,
}

fn run_cases(benchmark: bool, large_prefill: bool) {
    gpu_test_multi(
        if benchmark || large_prefill {
            2_u64 << 30
        } else {
            128 << 20
        },
        |sessions| {
            if sessions.len() != 4 {
                return Err("Qwen comm requires exactly four GPUs".into());
            }
            if benchmark {
                for session in sessions {
                    conc::exclusive(session)?;
                }
            }
            let capacity = 257;
            let width = 2560;
            let wide = width * 4;
            let query = SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
                ranks: 4,
                streams: 4,
                width: width as u32,
                eps_bits: 1e-6_f32.to_bits(),
                capacity: capacity as u32,
            };
            let size = qwen38::buffers(query).map_err(|e| e.to_string())?;
            // The one-shot entry at full capacity needs two banks of four
            // capacity-row slots.
            let legacy_inbox = qwen38::standalone_inbox_bytes(4, capacity as u32) as usize;
            let x = codes(capacity * wide, 1);
            let logits = codes(capacity * 4, 2);
            let weight = codes(wide, 3);
            let partials: Vec<_> = (0..4).map(|r| codes(capacity * width, r + 4)).collect();
            let mut fixtures = Vec::new();
            let mut modules = Vec::new();
            for (rank, s) in sessions.iter().enumerate() {
                let mut b = BTreeMap::new();
                for (id, values) in [
                    (100, &partials[rank]),
                    (101, &x),
                    (102, &logits),
                    (103, &weight),
                ] {
                    b.insert(id, s.upload_bf16(values)?);
                }
                for id in [104, 105, 114, 115, 124, 125, 134, 135] {
                    b.insert(id, s.upload_bf16(&vec![0x55aa; capacity * wide])?);
                }
                b.insert(106, s.allocate(size.inbox_bytes as usize)?);
                b.insert(107, bytes(s, &vec![0; size.flags_bytes as usize])?);
                b.insert(108, bytes(s, &vec![0; size.epoch_bytes as usize])?);
                b.insert(109, bytes(s, &[0; 4])?);
                // Live rows of the one-shot call in the mixed graph.
                b.insert(110, bytes(s, &7_u32.to_ne_bytes())?);
                b.insert(116, s.allocate(legacy_inbox)?);
                b.insert(117, bytes(s, &vec![0; size.flags_bytes as usize])?);
                b.insert(118, bytes(s, &vec![0; size.epoch_bytes as usize])?);
                fixtures.push(b);
                modules.push(
                    s.context
                        .load_module(include_bytes!(concat!(
                            env!("OUT_DIR"),
                            "/hc-prefill/comm_qwen38.cubin"
                        )))
                        .map_err(|e| e.to_string())?,
                );
            }
            let function = |rank: usize, symbol: &str| {
                modules[rank].function(symbol).map_err(|e| e.to_string())
            };
            let functions = (0..4)
                .map(|rank| {
                    Ok([
                        function(rank, qwen38::SYMBOL)?,
                        function(rank, qwen38::SCATTER_SYMBOL)?,
                        function(rank, qwen38::GATHER_SYMBOL)?,
                    ])
                })
                .collect::<LabResult<Vec<_>>>()?;
            for s in sessions {
                s.stream.synchronize().map_err(|e| e.to_string())?;
            }
            let set_live = |id: u64, rows: u32| -> LabResult<()> {
                for (r, s) in sessions.iter().enumerate() {
                    fixtures[r][&id]
                        .copy_from_bytes(&rows.to_ne_bytes(), &s.stream)
                        .map_err(|e| e.to_string())?;
                    s.stream.synchronize().map_err(|e| e.to_string())?;
                }
                Ok(())
            };
            // Launches one call: the bundle's path for up to `rows` live rows
            // (from buffer `live`), or the one-shot entry on the legacy state.
            let run_to = |rank: usize,
                          path: Path,
                          live: u64,
                          pdl: bool,
                          destination: Option<(u64, u64)>|
             -> LabResult<()> {
                let mut buffers: BTreeMap<_, _> =
                    fixtures[rank].iter().map(|(&k, v)| (k, v)).collect();
                let legacy = matches!(path, Path::Legacy);
                let (inbox, flags) = if legacy { (116, 117) } else { (106, 107) };
                for (peer, fixture) in fixtures.iter().enumerate() {
                    buffers.insert(peer as u64 + 1, &fixture[&inbox]);
                    buffers.insert(peer as u64 + 11, &fixture[&flags]);
                }
                let call = qwen38::Call {
                    input: 100,
                    residual: 101,
                    logits: 102,
                    weight: 103,
                    residual_out: destination.map_or(if legacy { 114 } else { 104 }, |pair| pair.0),
                    normed: destination.map_or(if legacy { 115 } else { 105 }, |pair| pair.1),
                    inboxes: (1..=4).collect(),
                    flags: (11..=14).collect(),
                    epoch: if legacy { 118 } else { 108 },
                    live_rows: live,
                    capacity: capacity as u32,
                    rank: rank as u32,
                };
                let launches = match path {
                    Path::Legacy => standalone(call),
                    Path::Bundle(rows) => qwen38::launches(query, rows, call),
                }
                .map_err(|e| e.to_string())?;
                for qwen38::Launch {
                    symbol,
                    mut plan,
                    arguments,
                } in launches
                {
                    let entry = SYMBOLS.iter().position(|&s| s == symbol).unwrap();
                    plan.pdl = pdl;
                    launch::launch(
                        &sessions[rank].context,
                        &functions[rank][entry],
                        &sessions[rank].stream,
                        plan,
                        &arguments,
                        &launch::Bindings {
                            buffers: buffers.clone(),
                        },
                    )?;
                }
                Ok(())
            };
            let run = |rank, path, live, pdl| run_to(rank, path, live, pdl, None);
            let run_all = |path: Path| -> LabResult<()> {
                for rank in 0..4 {
                    run(rank, path, 109, false)?;
                }
                for s in sessions {
                    s.stream.synchronize().map_err(|e| e.to_string())?;
                }
                Ok(())
            };
            // Two PDL calls of the bundle's path per graph, after a one-shot
            // call of 7 rows when the path is phased (both protocols on one
            // state).
            let capture = |rows: u32| -> LabResult<Vec<_>> {
                sessions
                    .iter()
                    .enumerate()
                    .map(|(r, s)| {
                        s.capture(|_| {
                            if qwen38::phased(rows) {
                                run_to(r, Path::Bundle(7), 110, true, Some((124, 125)))?;
                            }
                            run(r, Path::Bundle(rows), 109, true)?;
                            run_to(r, Path::Bundle(rows), 109, true, Some((134, 135)))
                        })
                    })
                    .collect()
            };
            // Every worker bucket of the phased plan (8 workers up to 32
            // rows, one more cluster per 32 rows, 128 workers from 481),
            // uneven owner splits, and one-shot calls in between on the same
            // state.
            let shapes = conc::ROWS;
            let mut failures = conc::replay::Failures::default();
            for rows in shapes {
                let result = (|| -> LabResult<()> {
                    set_live(109, rows as u32)?;
                    let mut initial_checks = sessions
                        .iter()
                        .enumerate()
                        .map(|(rank, s)| {
                            let mut replay = Replay::new(s);
                            for id in [104, 105, 114, 115] {
                                replay = replay.output(
                                    format!("destination {id}"),
                                    &fixtures[rank][&id],
                                    WriteMap::of(DType::Bf16).prefix(rows * wide)?,
                                );
                            }
                            Ok(replay)
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    conc::collective_phase(
                        sessions,
                        &mut initial_checks,
                        Phase::Eager,
                        false,
                        || {
                            run_all(Path::Bundle(rows as u32))?;
                            run_all(Path::Legacy)
                        },
                    )?;
                    let mut receipt = BTreeMap::from([
                        ("rows".into(), json!(rows)),
                        ("capacity".into(), json!(capacity)),
                        ("artifact".into(), evidence::identity("comm_qwen38")?),
                        (
                            "symbols".into(),
                            json!(if qwen38::phased(rows as u32) {
                                vec![qwen38::SCATTER_SYMBOL, qwen38::GATHER_SYMBOL]
                            } else {
                                vec![qwen38::SYMBOL]
                            }),
                        ),
                        ("profile".into(), json!("qwen38-flash")),
                    ]);
                    let download = |r: usize, id: u64| -> LabResult<Vec<u16>> {
                        sessions[r].download_bf16(&fixtures[r][&id], capacity * wide)
                    };
                    let before = (0..4)
                        .map(|r| Ok((download(r, 104)?, download(r, 105)?)))
                        .collect::<LabResult<Vec<_>>>()?;
                    for begin in (0..rows).step_by(128) {
                        let end = (begin + 128).min(rows);
                        let reference = hyper_connection_oracle::all_reduce_inject_norm(
                            query,
                            &partials
                                .iter()
                                .map(|p| decoded(&p[begin * width..end * width]))
                                .collect::<Vec<_>>(),
                            &decoded(&x[begin * wide..end * wide]),
                            &decoded(&logits[begin * 4..end * 4]),
                            &decoded(&weight),
                        )
                        .map_err(|e| e.to_string())?;
                        check_output(
                            &before[0].0[begin * wide..end * wide]
                                .iter()
                                .map(|&v| bf16_to_f32(v))
                                .collect::<Vec<_>>(),
                            &reference.residual,
                            DType::Bf16,
                            &mut receipt,
                        )?;
                        check_output(
                            &before[0].1[begin * wide..end * wide]
                                .iter()
                                .map(|&v| bf16_to_f32(v))
                                .collect::<Vec<_>>(),
                            &reference.normed,
                            DType::Bf16,
                            &mut receipt,
                        )?;
                    }
                    for (rank, (residual, normed)) in before.iter().enumerate() {
                        assert_eq!(residual, &before[0].0, "rank {rank} residual");
                        assert_eq!(normed, &before[0].1, "rank {rank} norm");
                        // Bit for bit the one-shot entry's result, and untouched
                        // past the live rows.
                        assert_eq!(
                            residual,
                            &download(rank, 114)?,
                            "rank {rank} residual vs one-shot"
                        );
                        assert_eq!(
                            normed,
                            &download(rank, 115)?,
                            "rank {rank} norm vs one-shot"
                        );
                        // No shape before the full-capacity one writes the
                        // last prepared row.
                        if !large_prefill && rows < capacity {
                            assert!(
                                residual[(capacity - 1) * wide..]
                                    .iter()
                                    .all(|&v| v == 0xffff),
                                "rank {rank} rows {rows} residual tail"
                            );
                            assert!(
                                normed[(capacity - 1) * wide..].iter().all(|&v| v == 0xffff),
                                "rank {rank} rows {rows} norm tail"
                            );
                        }
                    }
                    let mut replays = sessions
                        .iter()
                        .enumerate()
                        .map(|(rank, s)| {
                            let mut replay = Replay::new(s);
                            for (id, live) in [
                                (104, rows),
                                (105, rows),
                                (124, 7),
                                (125, 7),
                                (134, rows),
                                (135, rows),
                            ] {
                                replay = replay.output(
                                    format!("graph destination {id}"),
                                    &fixtures[rank][&id],
                                    WriteMap::of(DType::Bf16).prefix(live * wide)?,
                                );
                            }
                            Ok(replay)
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    let graphs = capture(rows as u32)?;
                    let launch_graphs = || -> LabResult<()> {
                        for (graph, s) in graphs.iter().zip(sessions) {
                            graph.launch(&s.stream).map_err(|e| e.to_string())?;
                        }
                        Ok(())
                    };
                    conc::collective_phase(
                        sessions,
                        &mut replays,
                        Phase::Eager,
                        false,
                        launch_graphs,
                    )?;
                    // The independent one-shot bank gets its own numerical oracle.
                    let reference = hyper_connection_oracle::all_reduce_inject_norm(
                        query,
                        &partials
                            .iter()
                            .map(|p| decoded(&p[..7 * width]))
                            .collect::<Vec<_>>(),
                        &decoded(&x[..7 * wide]),
                        &decoded(&logits[..7 * 4]),
                        &decoded(&weight),
                    )
                    .map_err(|e| e.to_string())?;
                    for (rank, full) in before.iter().enumerate() {
                        for (id, expected) in [(124, &reference.residual), (125, &reference.normed)]
                        {
                            let actual = download(rank, id)?;
                            check_output(
                                &actual[..7 * wide]
                                    .iter()
                                    .map(|&v| bf16_to_f32(v))
                                    .collect::<Vec<_>>(),
                                expected,
                                DType::Bf16,
                                &mut receipt,
                            )?;
                        }
                        for (id, expected) in [(134, &full.0), (135, &full.1)] {
                            assert_eq!(&download(rank, id)?, expected, "second full graph call");
                        }
                    }
                    for _ in 0..3 {
                        conc::collective_phase(
                            sessions,
                            &mut replays,
                            Phase::Graph,
                            true,
                            launch_graphs,
                        )?;
                        for (r, s) in sessions.iter().enumerate() {
                            assert_eq!(
                                s.download_bf16(&fixtures[r][&104], capacity * wide)?,
                                before[r].0,
                                "residual graph/PDL"
                            );
                            assert_eq!(
                                s.download_bf16(&fixtures[r][&105], capacity * wide)?,
                                before[r].1,
                                "norm graph/PDL"
                            );
                        }
                    }
                    conc::collective_phase(
                        sessions,
                        &mut replays,
                        Phase::Pdl,
                        true,
                        launch_graphs,
                    )?;
                    let mut rank_evidence = Vec::new();
                    for replay in &replays {
                        let mut fields = BTreeMap::new();
                        replay.record(&mut fields);
                        rank_evidence.push(fields);
                    }
                    receipt.insert("poisoned_ranks".into(), json!(rank_evidence));
                    receipt.insert("graph_determinism_pdl_rank_agreement".into(), json!(true));
                    receipt.insert("bitwise_one_shot_agreement".into(), json!(true));
                    sessions[0].receipt("qwen38-comm-g1", receipt)?;
                    let graphs = sessions
                        .iter()
                        .enumerate()
                        .map(|(rank, s)| {
                            s.capture(|_| run(rank, Path::Bundle(rows as u32), 109, true))
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    let timing = kernel_lab::time_multi_graph(sessions, &graphs, 1, 3, 9)?;
                    for (r, s) in sessions.iter().enumerate() {
                        assert_eq!(
                            s.download_bf16(&fixtures[r][&104], capacity * wide)?,
                            before[r].0,
                            "timed residual replay"
                        );
                        assert_eq!(
                            s.download_bf16(&fixtures[r][&105], capacity * wide)?,
                            before[r].1,
                            "timed norm replay"
                        );
                    }
                    let mut timed_checks = sessions
                        .iter()
                        .enumerate()
                        .map(|(rank, s)| {
                            Ok(Replay::new(s)
                                .output(
                                    "residual",
                                    &fixtures[rank][&104],
                                    WriteMap::of(DType::Bf16).prefix(rows * wide)?,
                                )
                                .output(
                                    "normed",
                                    &fixtures[rank][&105],
                                    WriteMap::of(DType::Bf16).prefix(rows * wide)?,
                                ))
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    conc::collective_phase(
                        sessions,
                        &mut timed_checks,
                        Phase::Eager,
                        false,
                        || run_all(Path::Bundle(rows as u32)),
                    )?;
                    conc::collective_phase(
                        sessions,
                        &mut timed_checks,
                        Phase::Graph,
                        true,
                        || {
                            for (s, graph) in sessions.iter().zip(&graphs) {
                                graph.launch(&s.stream).map_err(|e| e.to_string())?;
                            }
                            Ok(())
                        },
                    )?;
                    let mut timed_evidence = Vec::new();
                    for replay in &timed_checks {
                        let mut fields = BTreeMap::new();
                        replay.record(&mut fields);
                        timed_evidence.push(fields);
                    }
                    conc::record(
                        &sessions[0],
                        "M3",
                        "qwen_phased_all_reduce",
                        rows,
                        "comm_qwen38",
                        f64::from(timing.critical_path.median_ms) * 1000.0,
                        f64::from(timing.critical_path.min_ms) * 1000.0,
                        BTreeMap::from([
                            ("poisoned_ranks".into(), json!(timed_evidence)),
                            ("calls_per_graph".into(), json!(1)),
                            ("rounds".into(), json!(9)),
                            (
                                "per_rank_median_ms".into(),
                                json!(
                                    timing
                                        .per_rank
                                        .iter()
                                        .map(|t| t.median_ms)
                                        .collect::<Vec<_>>()
                                ),
                            ),
                        ]),
                    )?;

                    Ok(())
                })();
                failures.check(&format!("rows={rows}"), result);
            }
            failures.finish()?;
            Ok(())
        },
    )
    .expect("Qwen TP4 HC G1");
}

#[path = "common/concurrency.rs"]
mod conc;
use conc::evidence;
#[test]
fn m3_qwen_phased_all_reduce() {
    if conc::enabled("m3") {
        assert_eq!(
            kernel_lab::selected_ordinals().unwrap().unwrap().len(),
            4,
            "M3 collective requires TP4"
        );
        run_cases(true, false);
    }
}
