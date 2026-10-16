use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{
    Argument, ArgumentList, CommQuery, DType, LaunchPlan, RmsNormEpilogue,
};
use kernel_core::oracle::check_exact;
use kernel_core::oracle::{NormArithmetic, NormOffset, NormScale, RmsNorm};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::{LabResult, Session, gpu_test_multi, time_multi_graph};
use runtime_cuda::{DeviceBuffer, GraphExec, KernelArgument, LaunchConfig};
use serde_json::json;
use std::collections::BTreeMap;

/// HF's BF16 RMSNorm with eps 1e-6: F32 arithmetic, the normalized value
/// and the output each rounded to BF16.
const NORM: RmsNorm = RmsNorm {
    eps: 1e-6,
    offset: NormOffset::Plain,
    arithmetic: NormArithmetic::F32,
    scale: NormScale::MultiplyReciprocal,
    before_weight: Some(DType::Bf16),
    output: Some(DType::Bf16),
};

const CALLS_PER_GRAPH: usize = 64;
const WARMUP_REPLAYS: usize = 2;
const TIMED_REPLAYS: usize = 5;

/// Poisons each `(rank, buffer)` once every rank is idle: peers write into
/// a rank's outputs, so no rank may still run.
fn poison_ranks<'a>(
    sessions: &[Session],
    buffers: impl IntoIterator<Item = (usize, &'a DeviceBuffer)>,
) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (rank, buffer) in buffers {
        poison::poison(&sessions[rank], &[buffer])?;
    }
    Ok(())
}

/// The bytes of `buffer` after a replay on poisoned outputs: every element
/// of `map` written, every other byte still poison.
fn check_written(
    session: &Session,
    buffer: &DeviceBuffer,
    map: &WriteMap,
    label: &str,
) -> LabResult<Vec<u8>> {
    let bytes = poison::read(session, buffer)?;
    poison::check(label, &bytes, map, Fill::Pattern(&[POISON]))?;
    Ok(bytes)
}

/// One more launch of every rank's timed graph, after which every rank is
/// idle: the replay whose outputs a case checks, on buffers poisoned first.
fn replay(sessions: &[Session], graphs: &[GraphExec]) -> LabResult<()> {
    for (session, graph) in sessions.iter().zip(graphs) {
        graph.launch(&session.stream).map_err(|e| e.to_string())?;
    }
    poison::synchronize(sessions)
}

/// Legacy fused one-shot entries, kept as the baseline for the push path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Variant {
    Plain,
    Serial,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain_one_shot",
            Self::Serial => "one_shot",
        }
    }
}

fn run_launch_floor(sessions: &[Session]) -> LabResult<()> {
    let modules = sessions
        .iter()
        .map(|session| session.module("probe_sm120"))
        .collect::<LabResult<Vec<_>>>()?;
    let functions = modules
        .iter()
        .map(|module| {
            module
                .function("chop_probe_sm120_v1")
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions
        .iter()
        .map(|session| session.allocate(4))
        .collect::<LabResult<Vec<_>>>()?;
    let graphs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            session.capture(|_| {
                for _ in 0..CALLS_PER_GRAPH {
                    functions[rank]
                        .launch(
                            &session.stream,
                            launch::simple_config(
                                (1, 1, 1),
                                (32, 1, 1),
                                0,
                                LaunchConfig::NO_CLUSTER,
                                true,
                            )?,
                            &[KernelArgument::Buffer(&outputs[rank])],
                        )
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let timing = time_multi_graph(
        sessions,
        &graphs,
        CALLS_PER_GRAPH,
        WARMUP_REPLAYS,
        TIMED_REPLAYS,
    )?;
    let divisor = CALLS_PER_GRAPH as f32 / 1000.0;
    for (rank, session) in sessions.iter().enumerate() {
        session.receipt(
            "comm_bench",
            BTreeMap::from([
                ("case".into(), json!("graph_launch_floor")),
                ("rank".into(), json!(rank)),
                ("calls_per_graph".into(), json!(CALLS_PER_GRAPH)),
                ("pdl".into(), json!(true)),
                (
                    "median_us_per_call".into(),
                    json!(timing.critical_path.median_ms / divisor),
                ),
                (
                    "min_us_per_call".into(),
                    json!(timing.critical_path.min_ms / divisor),
                ),
            ]),
        )?;
    }
    Ok(())
}

fn run_local_micro(session: &Session, name: &str) -> LabResult<()> {
    let input = session.upload_bf16(&vec![f32_to_bf16(1.0); 4096])?;
    let output = session.allocate(8192)?;
    let residual = session.upload_bf16(&vec![f32_to_bf16(1.0); 4096])?;
    let weights = session.upload_bf16(&vec![f32_to_bf16(1.0); 4096])?;
    let flags = session.allocate(8)?;
    let epoch = session.allocate(4)?;
    flags
        .copy_from_bytes(&[0; 8], &session.stream)
        .map_err(|e| e.to_string())?;
    epoch
        .copy_from_bytes(&[0; 4], &session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    let module = session.module(if name == "local_copy" {
        "probe_sm120"
    } else {
        "comm_sm120"
    })?;
    let symbol = match name {
        "local_copy" => "chop_probe_copy_vector_v1",
        "local_norm" => "chop_comm_local_norm_bf16_v1",
        "local_flags" => "chop_comm_local_flags_v1",
        _ => return Err("unknown local microbenchmark".into()),
    };
    let function = module.function(symbol).map_err(|e| e.to_string())?;
    let config = launch::simple_config(
        (1, 1, 1),
        (if name == "local_norm" { 1024 } else { 256 }, 1, 1),
        0,
        LaunchConfig::NO_CLUSTER,
        true,
    )?;
    let graph = session.capture(|_| {
        for _ in 0..CALLS_PER_GRAPH {
            let arguments = match name {
                "local_copy" => vec![
                    KernelArgument::Buffer(&input),
                    KernelArgument::Buffer(&output),
                    KernelArgument::U64(2048),
                ],
                "local_norm" => vec![
                    KernelArgument::Buffer(&input),
                    KernelArgument::Buffer(&output),
                    KernelArgument::Buffer(&residual),
                    KernelArgument::Buffer(&weights),
                    KernelArgument::U32(1e-6_f32.to_bits()),
                ],
                "local_flags" => vec![
                    KernelArgument::Buffer(&flags),
                    KernelArgument::Buffer(&epoch),
                ],
                _ => unreachable!(),
            };
            function
                .launch(&session.stream, config, &arguments)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    })?;
    let timing = time_multi_graph(
        std::slice::from_ref(session),
        std::slice::from_ref(&graph),
        CALLS_PER_GRAPH,
        WARMUP_REPLAYS,
        TIMED_REPLAYS,
    )?;
    let expected = match name {
        "local_copy" => Some(vec![f32_to_bf16(1.0); 4096]),
        "local_norm" => Some(
            NORM.apply_bf16(
                &vec![f32_to_bf16(2.0); 4096],
                &vec![f32_to_bf16(1.0); 4096],
                4096,
            )
            .map_err(|e| e.to_string())?,
        ),
        _ => None,
    };
    if let Some(expected) = expected {
        // One more replay on a poisoned output. Both entries write all 4096
        // values: the copy moves 2048 words (probe.cuh `copy_vector`), the
        // norm one row of 4096 (comm.cuh `norm_row_bf16`); the README is
        // silent on these micro entries.
        poison::poison(session, &[&output])?;
        replay(std::slice::from_ref(session), std::slice::from_ref(&graph))?;
        check_written(
            session,
            &output,
            &WriteMap::of(DType::Bf16).prefix(4096)?,
            &format!("{name} output"),
        )?;
        let actual = session.download_bf16(&output, 4096)?;
        check_exact(&actual, &expected)
            .map_err(|index| format!("{name} differs at element {index}"))?;
    } else {
        let mut bytes = [0_u8; 4];
        epoch
            .copy_to_bytes(&mut bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        let expected_calls = ((WARMUP_REPLAYS + TIMED_REPLAYS) * CALLS_PER_GRAPH) as u32;
        if u32::from_ne_bytes(bytes) != expected_calls {
            return Err("local flag epoch did not advance with graph replays".into());
        }
    }
    let divisor = CALLS_PER_GRAPH as f32 / 1000.0;
    session.receipt(
        "comm_bench",
        BTreeMap::from([
            ("case".into(), json!(name)),
            ("ranks".into(), json!(1)),
            ("elements".into(), json!(4096)),
            ("pdl".into(), json!(true)),
            ("calls_per_graph".into(), json!(CALLS_PER_GRAPH)),
            ("correct".into(), json!(true)),
            (
                "median_us_per_call".into(),
                json!(timing.critical_path.median_ms / divisor),
            ),
            (
                "min_us_per_call".into(),
                json!(timing.critical_path.min_ms / divisor),
            ),
        ]),
    )?;
    Ok(())
}

fn run_peer_control(sessions: &[Session], tokens: usize, local: bool) -> LabResult<()> {
    let ranks = if local { 1 } else { sessions.len() };
    let elements = tokens * 4096;
    let bytes = elements * 2;
    let inputs = sessions[..ranks]
        .iter()
        .enumerate()
        .map(|(rank, session)| session.upload_bf16(&vec![f32_to_bf16((rank + 1) as f32); elements]))
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions[..ranks]
        .iter()
        .map(|session| session.allocate(bytes))
        .collect::<LabResult<Vec<_>>>()?;
    let modules = sessions[..ranks]
        .iter()
        .map(|session| session.module("probe_sm120"))
        .collect::<LabResult<Vec<_>>>()?;
    let functions = modules
        .iter()
        .map(|module| {
            module
                .function("chop_probe_p2p_pull_v1")
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let graphs = sessions[..ranks]
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            let source = if ranks == 4 {
                rank ^ 1
            } else {
                (rank + 1) % ranks
            };
            let address = inputs[source]
                .device_address_at(0, bytes)
                .map_err(|e| e.to_string())?;
            let blocks = std::env::var("CHOPINFER_KERNEL_BENCH_BLOCKS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or((bytes / 4).div_ceil(256).min(752));
            let config = launch::simple_config(
                (blocks as u32, 1, 1),
                (256, 1, 1),
                0,
                LaunchConfig::NO_CLUSTER,
                true,
            )?;
            session.capture(|_| {
                for _ in 0..CALLS_PER_GRAPH {
                    functions[rank]
                        .launch(
                            &session.stream,
                            config,
                            &[
                                KernelArgument::Buffer(&outputs[rank]),
                                KernelArgument::DevicePointerArray(&[address]),
                                KernelArgument::U64((bytes / 4) as u64),
                            ],
                        )
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let timing = time_multi_graph(
        &sessions[..ranks],
        &graphs,
        CALLS_PER_GRAPH,
        WARMUP_REPLAYS,
        TIMED_REPLAYS,
    )?;
    // One more replay on poisoned outputs: each rank's pull writes all of
    // its output (probe.cuh `p2p_pull`; the README is silent).
    poison_ranks(&sessions[..ranks], outputs.iter().enumerate())?;
    replay(&sessions[..ranks], &graphs)?;
    let written = WriteMap::of(DType::Bf16).byte_prefix(bytes)?;
    for (rank, session) in sessions[..ranks].iter().enumerate() {
        let source = if ranks == 4 {
            rank ^ 1
        } else {
            (rank + 1) % ranks
        };
        check_written(
            session,
            &outputs[rank],
            &written,
            &format!("peer pull control rank {rank} output"),
        )?;
        let expected = vec![f32_to_bf16((source + 1) as f32); elements];
        let actual = session.download_bf16(&outputs[rank], elements)?;
        check_exact(&actual, &expected)
            .map_err(|index| format!("peer pull control rank {rank} differs at {index}"))?;
        session.receipt(
            "comm_bench",
            BTreeMap::from([
                ("case".into(), json!("peer_pull_control")),
                (
                    "variant".into(),
                    json!(if local {
                        "peer_control_local"
                    } else {
                        "peer_control"
                    }),
                ),
                ("tokens".into(), json!(tokens)),
                ("ranks".into(), json!(ranks)),
                ("rank".into(), json!(rank)),
                ("source_rank".into(), json!(source)),
                (
                    "blocks".into(),
                    json!(
                        std::env::var("CHOPINFER_KERNEL_BENCH_BLOCKS")
                            .ok()
                            .and_then(|value| value.parse::<u32>().ok())
                            .unwrap_or((bytes / 4).div_ceil(256).min(752) as u32)
                    ),
                ),
                ("correct".into(), json!(true)),
                (
                    "median_us_per_call".into(),
                    json!(timing.critical_path.median_ms * 1000.0 / CALLS_PER_GRAPH as f32),
                ),
            ]),
        )?;
    }
    Ok(())
}

/// Push to rank + 1 and pull from rank + 2 at the same time, against each
/// alone, with every rank doing the same.
fn run_push_pull(sessions: &[Session], bytes: usize) -> LabResult<()> {
    let ranks = sessions.len();
    let words = bytes / 4;
    let inputs = sessions
        .iter()
        .map(|session| session.upload_bf16(&vec![f32_to_bf16(1.0); bytes / 2]))
        .collect::<LabResult<Vec<_>>>()?;
    let targets = sessions
        .iter()
        .map(|session| session.allocate(bytes * ranks))
        .collect::<LabResult<Vec<_>>>()?;
    let locals = sessions
        .iter()
        .map(|session| session.allocate(bytes))
        .collect::<LabResult<Vec<_>>>()?;
    let functions = sessions
        .iter()
        .map(|session| {
            session
                .module("probe_sm120")?
                .function("chop_probe_p2p_push_pull_v1")
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    for (mode, first_row, rows) in [
        ("push", 0_u32, 1_u32),
        ("pull", 1, 1),
        ("push_and_pull", 0, 2),
    ] {
        let graphs = sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| {
                let peers = vec![
                    targets[(rank + 1) % ranks]
                        .device_address_at(bytes * rank, bytes)
                        .map_err(|e| e.to_string())?,
                    inputs[(rank + 2) % ranks]
                        .device_address_at(0, bytes)
                        .map_err(|e| e.to_string())?,
                ];
                let config = launch::simple_config(
                    (16, rows, 1),
                    (256, 1, 1),
                    0,
                    LaunchConfig::NO_CLUSTER,
                    true,
                )?;
                session.capture(|_| {
                    for _ in 0..CALLS_PER_GRAPH {
                        functions[rank]
                            .launch(
                                &session.stream,
                                config,
                                &[
                                    KernelArgument::Buffer(&inputs[rank]),
                                    KernelArgument::DevicePointerArray(&peers),
                                    KernelArgument::Buffer(&locals[rank]),
                                    KernelArgument::U64(words as u64),
                                    KernelArgument::U32(first_row),
                                ],
                            )
                            .map_err(|e| e.to_string())?;
                    }
                    Ok(())
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        let timing = time_multi_graph(
            sessions,
            &graphs,
            CALLS_PER_GRAPH,
            WARMUP_REPLAYS,
            TIMED_REPLAYS,
        )?;
        let us = timing.critical_path.median_ms * 1000.0 / CALLS_PER_GRAPH as f32;
        for (rank, session) in sessions.iter().enumerate() {
            session.receipt(
                "comm_bench",
                BTreeMap::from([
                    ("case".into(), json!("peer_push_pull")),
                    ("variant".into(), json!(mode)),
                    ("ranks".into(), json!(ranks)),
                    ("rank".into(), json!(rank)),
                    ("tokens".into(), json!(bytes)),
                    ("median_us_per_call".into(), json!(us)),
                    (
                        "min_us_per_call".into(),
                        json!(timing.critical_path.min_ms * 1000.0 / CALLS_PER_GRAPH as f32),
                    ),
                ]),
            )?;
        }
    }
    Ok(())
}

/// Every rank pushes to one or more peers at once with 16 CTAs per peer.
fn run_push_fanout(sessions: &[Session], bytes: usize) -> LabResult<()> {
    let ranks = sessions.len();
    let words = bytes / 4;
    let inputs = sessions
        .iter()
        .map(|session| session.upload_bf16(&vec![f32_to_bf16(1.0); bytes / 2]))
        .collect::<LabResult<Vec<_>>>()?;
    let targets = sessions
        .iter()
        .map(|session| session.allocate(bytes * ranks))
        .collect::<LabResult<Vec<_>>>()?;
    let functions = sessions
        .iter()
        .map(|session| {
            session
                .module("probe_sm120")?
                .function("chop_probe_p2p_push_fanout_v1")
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    for fanout in [1, ranks - 1] {
        let graphs = sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| {
                // Distinct regions per source keep every write disjoint.
                let peers = (1..=fanout)
                    .map(|step| {
                        targets[(rank + step) % ranks]
                            .device_address_at(bytes * rank, bytes)
                            .map_err(|e| e.to_string())
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                let config = launch::simple_config(
                    (16, fanout as u32, 1),
                    (256, 1, 1),
                    0,
                    LaunchConfig::NO_CLUSTER,
                    true,
                )?;
                session.capture(|_| {
                    for _ in 0..CALLS_PER_GRAPH {
                        functions[rank]
                            .launch(
                                &session.stream,
                                config,
                                &[
                                    KernelArgument::Buffer(&inputs[rank]),
                                    KernelArgument::DevicePointerArray(&peers),
                                    KernelArgument::U64(words as u64),
                                ],
                            )
                            .map_err(|e| e.to_string())?;
                    }
                    Ok(())
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        let timing = time_multi_graph(
            sessions,
            &graphs,
            CALLS_PER_GRAPH,
            WARMUP_REPLAYS,
            TIMED_REPLAYS,
        )?;
        let us = timing.critical_path.median_ms * 1000.0 / CALLS_PER_GRAPH as f32;
        for (rank, session) in sessions.iter().enumerate() {
            session.receipt(
                "comm_bench",
                BTreeMap::from([
                    ("case".into(), json!("peer_push_fanout")),
                    ("ranks".into(), json!(ranks)),
                    ("rank".into(), json!(rank)),
                    ("destinations_at_once".into(), json!(fanout)),
                    ("bytes_per_destination".into(), json!(bytes)),
                    ("ctas_per_destination".into(), json!(16)),
                    ("median_us_per_call".into(), json!(us)),
                    (
                        "sent_gb_per_s_per_rank".into(),
                        json!((bytes * fanout) as f32 / us / 1000.0),
                    ),
                ]),
            )?;
        }
    }
    Ok(())
}

/// The planner-selected v2 fused entry. The residual is updated in place as
/// in the MiMo bundle; the check replays the same chain on the host.
fn run_push(sessions: &[Session], tokens: usize) -> LabResult<()> {
    use kernel_core::codec::bf16_to_f32;
    let ranks = sessions.len();
    let elements = tokens * 4096;
    let query = CommQuery::AllReduce {
        ranks: ranks as u32,
        elements: elements as u64,
        dtype: DType::Bf16,
        fused_residual: true,
        fused_rmsnorm: Some(RmsNormEpilogue {
            kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
            width: 4096,
            eps: 1e-6,
        }),
    };
    let plan = kernels_sm120::comm::plan(&query, tokens as u32).map_err(|e| e.to_string())?;
    if !plan.pdl {
        return Err("comm benchmark requires PDL launches".into());
    }
    let protocol = kernels_sm120::comm::push_protocol(ranks as u32, tokens as u32)
        .map_err(|e| e.to_string())?;
    let sizes = kernels_sm120::comm::push_buffers(ranks as u32, tokens as u32)
        .map_err(|e| e.to_string())?;
    let host_inputs = (0..ranks)
        .map(|rank| {
            (0..elements)
                .map(|index| f32_to_bf16(((index * 7 + rank * 13) % 101) as f32 * 0.001 - 0.05))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let host_residual = (0..elements)
        .map(|index| f32_to_bf16((index % 23) as f32 * 0.05 - 0.5))
        .collect::<Vec<_>>();
    let host_weights = (0..4096)
        .map(|index| f32_to_bf16(0.75 + (index % 5) as f32 * 0.125))
        .collect::<Vec<_>>();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut residuals = Vec::new();
    let mut weights = Vec::new();
    let mut inboxes = Vec::new();
    let mut flags = Vec::new();
    let mut epochs = Vec::new();
    for (rank, session) in sessions.iter().enumerate() {
        inputs.push(session.upload_bf16(&host_inputs[rank])?);
        outputs.push(session.allocate(elements * 2)?);
        residuals.push(session.upload_bf16(&host_residual)?);
        weights.push(session.upload_bf16(&host_weights)?);
        let inbox = session.allocate(sizes.inbox_bytes as usize)?;
        inbox
            .zero_at_async(0, sizes.zeroed_inbox_bytes as usize, &session.stream)
            .map_err(|e| e.to_string())?;
        inboxes.push(inbox);
        let flag_bytes = sizes.flag_bytes.max(4) as usize;
        let flag = session.allocate(flag_bytes)?;
        flag.copy_from_bytes(&vec![0; flag_bytes], &session.stream)
            .map_err(|e| e.to_string())?;
        flags.push(flag);
        let epoch = session.allocate(sizes.epoch_bytes as usize)?;
        epoch
            .copy_from_bytes(&vec![0; sizes.epoch_bytes as usize], &session.stream)
            .map_err(|e| e.to_string())?;
        epochs.push(epoch);
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    let functions = sessions
        .iter()
        .map(|session| {
            session
                .module("comm_sm120")?
                .function("chop_comm_allreduce_oneshot_norm_bf16_v2")
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let handles: Vec<u64> = (0..ranks).map(|rank| rank as u64 + 1).collect();
    let graphs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
            buffers.insert(100, &inputs[rank]);
            buffers.insert(101, &outputs[rank]);
            buffers.insert(102, &epochs[rank]);
            buffers.insert(103, &residuals[rank]);
            buffers.insert(104, &weights[rank]);
            for peer in 0..ranks {
                buffers.insert(peer as u64 + 1, &inboxes[peer]);
                buffers.insert(peer as u64 + 11, &flags[peer]);
            }
            let bindings = Bindings { buffers };
            let args = kernels_sm120::comm::fused_arguments_v2(
                kernels_sm120::comm::FusedWithResidualOutputCall {
                    fused: kernels_sm120::comm::FusedOneShotCall {
                        base: kernels_sm120::comm::OneShotCall {
                            input: 100,
                            output: 101,
                            peer_inboxes: handles.clone(),
                            peer_flags: handles.iter().map(|value| value + 10).collect(),
                            epoch: 102,
                            elements: elements as u64,
                            rank: rank as u32,
                            ranks: ranks as u32,
                        },
                        residual: Some(103),
                        weights: 104,
                        epsilon: 1e-6,
                    },
                    residual_output: 103,
                },
            )
            .map_err(|e| e.to_string())?;
            session.capture(|_| {
                for _ in 0..CALLS_PER_GRAPH {
                    launch::launch(
                        &session.context,
                        &functions[rank],
                        &session.stream,
                        plan,
                        &args,
                        &bindings,
                    )?;
                }
                Ok(())
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let timing = time_multi_graph(
        sessions,
        &graphs,
        CALLS_PER_GRAPH,
        WARMUP_REPLAYS,
        TIMED_REPLAYS,
    )?;
    // One more replay on poisoned outputs, checked below: every call writes
    // all rows of the output (README "Fused decode all-reduce"). The
    // residual is the in-place chain, and its calls count too.
    poison_ranks(sessions, outputs.iter().enumerate())?;
    replay(sessions, &graphs)?;
    let written = WriteMap::of(DType::Bf16).prefix(elements)?;
    let calls = (WARMUP_REPLAYS + TIMED_REPLAYS + 1) * CALLS_PER_GRAPH;
    let sum = (0..elements)
        .map(|index| {
            (0..ranks).fold(0.0_f32, |sum, rank| {
                sum + kernel_core::codec::bf16_to_f32(host_inputs[rank][index])
            })
        })
        .collect::<Vec<_>>();
    let mut expected_residual = host_residual;
    for _ in 0..calls {
        for (value, &add) in expected_residual.iter_mut().zip(&sum) {
            *value = f32_to_bf16(add + bf16_to_f32(*value));
        }
    }
    let expected = NORM
        .apply_bf16(&expected_residual, &host_weights, 4096)
        .map_err(|e| e.to_string())?;
    for (rank, session) in sessions.iter().enumerate() {
        check_written(
            session,
            &outputs[rank],
            &written,
            &format!("push token {tokens} rank {rank} output"),
        )?;
    }
    let first = sessions[0].download_bf16(&outputs[0], elements)?;
    let divisor = CALLS_PER_GRAPH as f32 / 1000.0;
    let per_rank_us: Vec<f32> = timing
        .per_rank
        .iter()
        .map(|sample| sample.median_ms / divisor)
        .collect();
    for (rank, session) in sessions.iter().enumerate() {
        let actual = session.download_bf16(&outputs[rank], elements)?;
        let residual_gate = check_exact(
            &session.download_bf16(&residuals[rank], elements)?,
            &expected_residual,
        );
        let equal = actual == first;
        let near = actual
            .iter()
            .zip(&expected)
            .all(|(&a, &b)| (i32::from(a) - i32::from(b)).abs() <= 1);
        session.receipt(
            "comm_bench",
            BTreeMap::from([
                ("case".into(), json!("bf16_fused_decode_latency")),
                ("variant".into(), json!("push")),
                ("protocol".into(), json!(format!("{protocol:?}"))),
                ("tokens".into(), json!(tokens)),
                ("elements".into(), json!(elements)),
                ("ranks".into(), json!(ranks)),
                ("rank".into(), json!(rank)),
                ("grid".into(), json!(plan.grid[0])),
                ("cluster".into(), json!(plan.cluster[0])),
                ("calls_per_graph".into(), json!(CALLS_PER_GRAPH)),
                ("warmup_replays".into(), json!(WARMUP_REPLAYS)),
                ("timed_replays".into(), json!(TIMED_REPLAYS)),
                ("pdl".into(), json!(plan.pdl)),
                (
                    "median_us_per_call".into(),
                    json!(timing.critical_path.median_ms / divisor),
                ),
                (
                    "min_us_per_call".into(),
                    json!(timing.critical_path.min_ms / divisor),
                ),
                ("per_rank_median_us_per_call".into(), json!(per_rank_us)),
                ("residual_correct".into(), json!(residual_gate.is_ok())),
                ("bitwise_equal_to_rank0".into(), json!(equal)),
                ("output_within_one_bf16_step".into(), json!(near)),
                ("input_pattern".into(), json!("varying_bf16")),
            ]),
        )?;
        residual_gate.map_err(|index| {
            format!("push token {tokens} rank {rank} residual differs at {index}")
        })?;
        if !equal || !near {
            return Err(format!("push token {tokens} rank {rank} output mismatch"));
        }
    }
    Ok(())
}

/// The bundle's fused prefill path for rows above 32: reduce-scatter and the
/// fused gather with residual and RMSNorm, on buffers sized like the MiMo
/// prefill comm state. Bus bandwidth counts 2 (ranks - 1) / ranks of the
/// message per rank, the traffic an all-reduce must move.
fn run_prefill(sessions: &[Session], rows: usize) -> LabResult<()> {
    use kernel_core::codec::bf16_to_f32;
    const STATE_ROWS: usize = 8192;
    let ranks = sessions.len();
    let elements = rows * 4096;
    let calls = (8192 / rows).clamp(2, 16);
    let query = CommQuery::AllReduce {
        ranks: ranks as u32,
        elements: elements as u64,
        dtype: DType::Bf16,
        fused_residual: true,
        fused_rmsnorm: Some(RmsNormEpilogue {
            kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
            width: 4096,
            eps: 1e-6,
        }),
    };
    let mut plan = kernels_sm120::comm::plan(&query, rows as u32).map_err(|e| e.to_string())?;
    // A smaller grid keeps the sending cluster and fewer worker clusters.
    if let Some(grid) = std::env::var("CHOPINFER_KERNEL_BENCH_PREFILL_GRID")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
    {
        plan.grid[0] = grid;
    }
    if kernels_sm120::comm::choose(&query).map_err(|e| e.to_string())?
        != kernels_sm120::comm::Variant::FusedTwoShot
    {
        return Err("prefill bench expects the two-stage path".into());
    }
    let host_inputs = (0..ranks)
        .map(|rank| {
            (0..elements)
                .map(|index| f32_to_bf16(((index * 7 + rank * 13) % 101) as f32 * 0.001 - 0.05))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let host_residual = (0..elements)
        .map(|index| f32_to_bf16((index % 23) as f32 * 0.05 - 0.5))
        .collect::<Vec<_>>();
    let host_weights = (0..4096)
        .map(|index| f32_to_bf16(0.75 + (index % 5) as f32 * 0.125))
        .collect::<Vec<_>>();
    let flag_bytes = 4 * ranks * STATE_ROWS * 4;
    let inbox_bytes = ranks * STATE_ROWS * 4096 * 2;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut residuals = Vec::new();
    let mut weights = Vec::new();
    let mut inboxes = Vec::new();
    let mut flags = Vec::new();
    let mut epochs = Vec::new();
    for (rank, session) in sessions.iter().enumerate() {
        inputs.push(session.upload_bf16(&host_inputs[rank])?);
        outputs.push(session.allocate(elements * 2)?);
        residuals.push(session.upload_bf16(&host_residual)?);
        weights.push(session.upload_bf16(&host_weights)?);
        inboxes.push(session.allocate(inbox_bytes)?);
        let flag = session.allocate(flag_bytes)?;
        flag.copy_from_bytes(&vec![0; flag_bytes], &session.stream)
            .map_err(|e| e.to_string())?;
        flags.push(flag);
        let epoch = session.allocate(STATE_ROWS * 4)?;
        epoch
            .copy_from_bytes(&vec![0; STATE_ROWS * 4], &session.stream)
            .map_err(|e| e.to_string())?;
        epochs.push(epoch);
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    let modules = sessions
        .iter()
        .map(|session| session.module("comm_sm120"))
        .collect::<LabResult<Vec<_>>>()?;
    let handles: Vec<u64> = (0..ranks as u64).map(|rank| rank + 1).collect();
    let flag_handles: Vec<u64> = handles.iter().map(|handle| handle + 10).collect();
    let output_handles: Vec<u64> = handles.iter().map(|handle| handle + 20).collect();
    let mut launches = Vec::new();
    for rank in 0..ranks {
        let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
        buffers.insert(100, &inputs[rank]);
        buffers.insert(101, &outputs[rank]);
        buffers.insert(102, &epochs[rank]);
        buffers.insert(103, &residuals[rank]);
        buffers.insert(104, &weights[rank]);
        for peer in 0..ranks {
            buffers.insert(peer as u64 + 1, &inboxes[peer]);
            buffers.insert(peer as u64 + 11, &flags[peer]);
            buffers.insert(peer as u64 + 21, &outputs[peer]);
        }
        let call = kernels_sm120::comm::OneShotCall {
            input: 100,
            output: 101,
            peer_inboxes: handles.clone(),
            peer_flags: flag_handles.clone(),
            epoch: 102,
            elements: elements as u64,
            rank: rank as u32,
            ranks: ranks as u32,
        };
        let gather = kernels_sm120::comm::round_once_gather_arguments(
            &call,
            output_handles.clone(),
            104,
            1e-6,
            103,
        )
        .map_err(|e| e.to_string())?;
        let scatter = kernels_sm120::comm::round_once_scatter_arguments(call, 103)
            .map_err(|e| e.to_string())?;
        launches.push((Bindings { buffers }, scatter, gather));
    }
    let functions = modules
        .iter()
        .map(|module| {
            Ok((
                module
                    .function(kernels_sm120::comm::PREFILL_ROUND_ONCE_SYMBOL)
                    .map_err(|e| e.to_string())?,
                module
                    .function(kernels_sm120::comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL)
                    .map_err(|e| e.to_string())?,
            ))
        })
        .collect::<LabResult<Vec<_>>>()?;
    let issue = |rank: usize, session: &Session| -> LabResult<()> {
        let (bindings, scatter, gather) = &launches[rank];
        launch::launch(
            &session.context,
            &functions[rank].0,
            &session.stream,
            plan,
            scatter,
            bindings,
        )?;
        launch::launch(
            &session.context,
            &functions[rank].1,
            &session.stream,
            plan,
            gather,
            bindings,
        )
    };
    // Every rank's output ends up whole: the scatter writes the owner's
    // tiles, the gather sends them into every peer's output and normalizes
    // them there (README "Prefill all-reduce"). The residual is the in-place
    // chain, never poisoned.
    let written = WriteMap::of(DType::Bf16).prefix(elements)?;
    // One eager call checks the residual stream against the host exactly.
    poison_ranks(sessions, outputs.iter().enumerate())?;
    for (rank, session) in sessions.iter().enumerate() {
        issue(rank, session)?;
    }
    for session in sessions {
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    for (rank, session) in sessions.iter().enumerate() {
        check_written(
            session,
            &outputs[rank],
            &written,
            &format!("prefill {rows} rank {rank} eager output"),
        )?;
    }
    let expected = (0..elements)
        .map(|index| {
            let sum = (0..ranks).fold(0.0_f32, |sum, rank| {
                sum + bf16_to_f32(host_inputs[rank][index])
            });
            f32_to_bf16(sum + bf16_to_f32(host_residual[index]))
        })
        .collect::<Vec<_>>();
    for (rank, session) in sessions.iter().enumerate() {
        check_exact(
            &session.download_bf16(&residuals[rank], elements)?,
            &expected,
        )
        .map_err(|index| format!("prefill {rows} rank {rank}: residual differs at {index}"))?;
    }
    let graphs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            session.capture(|_| {
                for _ in 0..calls {
                    issue(rank, session)?;
                }
                Ok(())
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let timing = time_multi_graph(sessions, &graphs, calls, 1, TIMED_REPLAYS)?;
    // One more replay on poisoned outputs, compared below.
    poison_ranks(sessions, outputs.iter().enumerate())?;
    replay(sessions, &graphs)?;
    for (rank, session) in sessions.iter().enumerate() {
        check_written(
            session,
            &outputs[rank],
            &written,
            &format!("prefill {rows} rank {rank} replayed output"),
        )?;
    }
    let first = sessions[0].download_bf16(&outputs[0], elements)?;
    let first_residual = sessions[0].download_bf16(&residuals[0], elements)?;
    let us = timing.critical_path.median_ms * 1000.0 / calls as f32;
    let bytes = (elements * 2) as f32;
    let algorithm_gb_s = bytes / us / 1000.0;
    let bus_gb_s = algorithm_gb_s * 2.0 * (ranks - 1) as f32 / ranks as f32;
    for (rank, session) in sessions.iter().enumerate() {
        let equal = session.download_bf16(&outputs[rank], elements)? == first
            && session.download_bf16(&residuals[rank], elements)? == first_residual;
        session.receipt(
            "comm_bench",
            BTreeMap::from([
                ("case".into(), json!("bf16_fused_prefill_bandwidth")),
                ("rows".into(), json!(rows)),
                ("ranks".into(), json!(ranks)),
                ("rank".into(), json!(rank)),
                ("grid".into(), json!(plan.grid[0])),
                ("cluster".into(), json!(plan.cluster[0])),
                ("calls_per_graph".into(), json!(calls)),
                ("timed_replays".into(), json!(TIMED_REPLAYS)),
                ("median_us_per_call".into(), json!(us)),
                (
                    "min_us_per_call".into(),
                    json!(timing.critical_path.min_ms * 1000.0 / calls as f32),
                ),
                ("algorithm_gb_per_s".into(), json!(algorithm_gb_s)),
                ("bus_gb_per_s".into(), json!(bus_gb_s)),
                ("first_call_residual_exact".into(), json!(true)),
                ("bitwise_equal_to_rank0".into(), json!(equal)),
            ]),
        )?;
        if !equal {
            return Err(format!("prefill {rows} rank {rank} differs from rank 0"));
        }
    }
    eprintln!(
        "prefill rows {rows} ranks {ranks} grid {}: {us:.1} us per call, algorithm {algorithm_gb_s:.1} GB/s, bus {bus_gb_s:.1} GB/s",
        plan.grid[0]
    );
    if std::env::var("CHOPINFER_KERNEL_BENCH_OVERLAP").as_deref() == Ok("1") {
        measure_prefill_overlap(sessions, &graphs, rows, plan.grid[0], us * calls as f32)?;
    }
    Ok(())
}

/// The wide push all-reduce of decode and verify graphs above 32 rows
/// against the entries graphs used before it: the phased reduce-scatter and
/// fused gather pair, preceded with `combine` by the MoE row combine that
/// the wide combine entry folds in. Both arms run on one comm state sized
/// like a MiMo verify state of `rows` rows; bus bandwidth counts
/// 2 (ranks - 1) / ranks of the message per rank.
/// `CHOPINFER_KERNEL_BENCH_WIDE_WORKERS` overrides the wide worker CTAs.
fn run_wide(sessions: &[Session], rows: usize, combine: bool) -> LabResult<()> {
    use kernel_build::EntryParam;
    use kernel_core::contract::RuntimeCounts;
    const CALLS: usize = 32;
    let ranks = sessions.len();
    let elements = rows * 4096;
    let depth = if combine { 8 } else { 1 };
    let query = CommQuery::AllReduce {
        ranks: ranks as u32,
        elements: elements as u64,
        dtype: DType::Bf16,
        fused_residual: true,
        fused_rmsnorm: Some(RmsNormEpilogue {
            kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
            width: 4096,
            eps: 1e-6,
        }),
    };
    if kernels_sm120::comm::graph_choose(&query).map_err(|e| e.to_string())?
        != kernels_sm120::comm::Variant::FusedWide
    {
        return Err(format!(
            "{rows} rows at {ranks} ranks do not take the wide entry"
        ));
    }
    let mut wide_plan =
        kernels_sm120::comm::graph_plan(&query, rows as u32).map_err(|e| e.to_string())?;
    if let Some(workers) = std::env::var("CHOPINFER_KERNEL_BENCH_WIDE_WORKERS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
    {
        wide_plan.grid[0] = wide_plan.cluster[0] + workers.next_multiple_of(wide_plan.cluster[0]);
    }
    // Keep the worker count within the target layout.
    kernels_sm120::comm::wide_fits_target_layout(&wide_plan).map_err(|e| e.to_string())?;
    let phased_plan = kernels_sm120::comm::plan(&query, rows as u32).map_err(|e| e.to_string())?;
    let sizes = kernels_sm120::comm::fused_state_buffers(ranks as u32, rows as u32)
        .map_err(|e| e.to_string())?;
    let host_inputs = (0..ranks)
        .map(|rank| {
            (0..elements * depth)
                .map(|index| f32_to_bf16(((index * 7 + rank * 13) % 101) as f32 * 0.001 - 0.05))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let host_residual = (0..elements)
        .map(|index| f32_to_bf16((index % 23) as f32 * 0.05 - 0.5))
        .collect::<Vec<_>>();
    let host_weights = (0..4096)
        .map(|index| f32_to_bf16(0.75 + (index % 5) as f32 * 0.125))
        .collect::<Vec<_>>();
    let zeroed = |session: &Session, bytes: u64| -> LabResult<DeviceBuffer> {
        let buffer = session.allocate(bytes.max(4) as usize)?;
        buffer
            .copy_from_bytes(&vec![0; bytes.max(4) as usize], &session.stream)
            .map_err(|e| e.to_string())?;
        Ok(buffer)
    };
    let mut inputs = Vec::new();
    let mut combined = Vec::new();
    let mut outputs = Vec::new();
    let mut residuals = Vec::new();
    let mut weights = Vec::new();
    let mut inboxes = Vec::new();
    let mut flags = Vec::new();
    let mut epochs = Vec::new();
    let mut topk = Vec::new();
    let mut lives = Vec::new();
    for (rank, session) in sessions.iter().enumerate() {
        inputs.push(session.upload_bf16(&host_inputs[rank])?);
        combined.push(session.allocate(elements * 2)?);
        outputs.push(session.allocate(elements * 2)?);
        residuals.push(session.upload_bf16(&host_residual)?);
        weights.push(session.upload_bf16(&host_weights)?);
        inboxes.push(zeroed(session, sizes.inbox_bytes)?);
        flags.push(zeroed(session, sizes.flag_bytes)?);
        epochs.push(zeroed(session, sizes.epoch_bytes)?);
        topk.push(session.upload_f32(&vec![1.0; rows * 8])?);
        let live = session.allocate(4)?;
        live.copy_from_bytes(&(rows as u32).to_ne_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        lives.push(live);
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    // The MoE row combine the phased arm runs first with `combine`.
    let combine_entry = kernels_sm120::row::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.symbol.contains("combine_mimo4096"))
        .ok_or("combine entry")?;
    let combine_arguments = combine_entry
        .pack_arguments(
            combine_entry
                .params
                .iter()
                .map(|param| match param {
                    EntryParam::DevicePointer { name, .. } => {
                        Ok(Argument::DeviceBuffer(match name.as_str() {
                            "routed" => 100,
                            "weights" => 106,
                            "output" => 107,
                            "live_rows" | "live_tokens" => 105,
                            _ => return Err(format!("unexpected combine pointer {name}")),
                        }))
                    }
                    _ => Err("unexpected combine scalar".to_string()),
                })
                .collect::<LabResult<Vec<_>>>()?,
        )
        .map_err(|e| e.to_string())?;
    // The combine row plans power-of-two row buckets.
    let combine_plan = combine_entry
        .plan(
            (rows as u32).next_power_of_two(),
            RuntimeCounts {
                m: rows as u32,
                batch: rows as u32,
                context: 0,
                routed_rows: rows as u32 * 8,
                split_k: 1,
            },
        )
        .map_err(|e| e.to_string())?;
    let handles: Vec<u64> = (0..ranks as u64).map(|rank| rank + 1).collect();
    let flag_handles: Vec<u64> = handles.iter().map(|handle| handle + 10).collect();
    let output_handles: Vec<u64> = handles.iter().map(|handle| handle + 20).collect();
    let bindings = (0..ranks)
        .map(|rank| {
            let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
            buffers.insert(100, &inputs[rank]);
            buffers.insert(101, &outputs[rank]);
            buffers.insert(102, &epochs[rank]);
            buffers.insert(103, &residuals[rank]);
            buffers.insert(104, &weights[rank]);
            buffers.insert(105, &lives[rank]);
            buffers.insert(106, &topk[rank]);
            buffers.insert(107, &combined[rank]);
            for peer in 0..ranks {
                buffers.insert(peer as u64 + 1, &inboxes[peer]);
                buffers.insert(peer as u64 + 11, &flags[peer]);
                buffers.insert(peer as u64 + 21, &outputs[peer]);
            }
            Bindings { buffers }
        })
        .collect::<Vec<_>>();
    let base = |rank: usize, input: u64| kernels_sm120::comm::OneShotCall {
        input,
        output: 101,
        peer_inboxes: handles.clone(),
        peer_flags: flag_handles.clone(),
        epoch: 102,
        elements: elements as u64,
        rank: rank as u32,
        ranks: ranks as u32,
    };
    let modules = sessions
        .iter()
        .map(|session| session.module("comm_sm120"))
        .collect::<LabResult<Vec<_>>>()?;
    let row_modules = sessions
        .iter()
        .map(|session| {
            session
                .context
                .load_module(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/row_engine.cubin"
                )))
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let mut results = Vec::new();
    for arm in ["wide", "phased"] {
        let mut launches = Vec::new();
        for rank in 0..ranks {
            let mut stages = Vec::new();
            if arm == "wide" {
                let call = kernels_sm120::comm::FusedWithResidualOutputCall {
                    fused: kernels_sm120::comm::FusedOneShotCall {
                        base: base(rank, 100),
                        residual: Some(103),
                        weights: 104,
                        epsilon: 1e-6,
                    },
                    residual_output: 103,
                };
                let (symbol, args) = if combine {
                    (
                        kernels_sm120::comm::WIDE_COMBINE_SYMBOL,
                        kernels_sm120::comm::fused_combine_arguments(call, Some(105)),
                    )
                } else {
                    (
                        kernels_sm120::comm::WIDE_SYMBOL,
                        kernels_sm120::comm::fused_arguments_v2(call),
                    )
                };
                let function = modules[rank].function(symbol).map_err(|e| e.to_string())?;
                stages.push((function, wide_plan, args.map_err(|e| e.to_string())?));
            } else {
                let input = if combine {
                    let function = row_modules[rank]
                        .function(&combine_entry.symbol)
                        .map_err(|e| e.to_string())?;
                    stages.push((function, combine_plan, combine_arguments.clone()));
                    107
                } else {
                    100
                };
                let call = base(rank, input);
                let gather = kernels_sm120::comm::round_once_gather_arguments(
                    &call,
                    output_handles.clone(),
                    104,
                    1e-6,
                    103,
                )
                .map_err(|e| e.to_string())?;
                let scatter = kernels_sm120::comm::round_once_scatter_arguments(call, 103)
                    .map_err(|e| e.to_string())?;
                for (symbol, args) in [
                    (kernels_sm120::comm::PREFILL_ROUND_ONCE_SYMBOL, scatter),
                    (
                        kernels_sm120::comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL,
                        gather,
                    ),
                ] {
                    let function = modules[rank].function(symbol).map_err(|e| e.to_string())?;
                    stages.push((function, phased_plan, args));
                }
            }
            launches.push(stages);
        }
        let issue = |rank: usize, session: &Session| -> LabResult<()> {
            for (function, plan, args) in &launches[rank] {
                launch::launch(
                    &session.context,
                    function,
                    &session.stream,
                    *plan,
                    args,
                    &bindings[rank],
                )?;
            }
            Ok(())
        };
        let graphs = sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| {
                session.capture(|_| (0..CALLS).try_for_each(|_| issue(rank, session)))
            })
            .collect::<LabResult<Vec<_>>>()?;
        let timing = time_multi_graph(sessions, &graphs, CALLS, 1, TIMED_REPLAYS)?;
        // One more replay on poisoned outputs, and on the combine rows the
        // phased arm's scatter reads, compared below. Both arms write all
        // rows of every rank's output (README "Wide all-reduce for graphs
        // above 32 rows" and "Prefill all-reduce").
        poison_ranks(sessions, outputs.iter().enumerate())?;
        if arm == "phased" && combine {
            poison_ranks(sessions, combined.iter().enumerate())?;
        }
        replay(sessions, &graphs)?;
        let written = WriteMap::of(DType::Bf16).prefix(elements)?;
        for (rank, session) in sessions.iter().enumerate() {
            check_written(
                session,
                &outputs[rank],
                &written,
                &format!("{arm} {rows} rows rank {rank} output"),
            )?;
        }
        let first = sessions[0].download_bf16(&outputs[0], elements)?;
        for (rank, session) in sessions.iter().enumerate() {
            if session.download_bf16(&outputs[rank], elements)? != first {
                return Err(format!(
                    "{arm} {rows} rows: rank {rank} differs from rank 0"
                ));
            }
        }
        let us = timing.critical_path.median_ms * 1000.0 / CALLS as f32;
        results.push((arm, us, timing));
    }
    let wide_us = results[0].1;
    let bus =
        |us: f32| (elements * 2) as f32 / us / 1000.0 * 2.0 * (ranks - 1) as f32 / ranks as f32;
    for (rank, session) in sessions.iter().enumerate() {
        for (arm, us, timing) in &results {
            session.receipt(
                "comm_bench",
                BTreeMap::from([
                    ("case".into(), json!("bf16_wide_allreduce")),
                    ("arm".into(), json!(arm)),
                    ("combine".into(), json!(combine)),
                    ("rows".into(), json!(rows)),
                    ("ranks".into(), json!(ranks)),
                    ("rank".into(), json!(rank)),
                    (
                        "grid".into(),
                        json!(if *arm == "wide" {
                            wide_plan.grid[0]
                        } else {
                            phased_plan.grid[0]
                        }),
                    ),
                    ("calls_per_graph".into(), json!(CALLS)),
                    ("timed_replays".into(), json!(TIMED_REPLAYS)),
                    ("median_us_per_call".into(), json!(us)),
                    (
                        "min_us_per_call".into(),
                        json!(timing.critical_path.min_ms * 1000.0 / CALLS as f32),
                    ),
                    ("bus_gb_per_s".into(), json!(bus(*us))),
                    ("wide_speedup".into(), json!(us / wide_us)),
                    ("bitwise_equal_to_rank0".into(), json!(true)),
                ]),
            )?;
        }
    }
    eprintln!(
        "wide rows {rows} ranks {ranks} combine {combine} grid {}: wide {wide_us:.1} us ({:.1} GB/s bus), phased {:.1} us ({:.1} GB/s bus)",
        wide_plan.grid[0],
        bus(wide_us),
        results[1].1,
        bus(results[1].1)
    );
    Ok(())
}

/// Runs the all-reduce graph next to an MMA-bound stand-in for prefill GEMMs
/// on a second stream: one 256-thread CTA per SM holding 80 KiB of shared
/// memory, so no all-reduce CTA can share its SM, as with the packed prefill
/// GEMMs. The stand-in leaves the all-reduce grid's SMs free, as a GEMM
/// sized for overlap would, and runs about 1.5 times as long. Reports each
/// alone and both together, per rank, slowest rank.
fn measure_prefill_overlap(
    sessions: &[Session],
    reduce: &[runtime_cuda::GraphExec],
    rows: usize,
    grid: u32,
    reduce_us: f32,
) -> LabResult<()> {
    const ROUNDS: usize = 5;
    let modules = sessions
        .iter()
        .map(|session| session.module("probe_sm120"))
        .collect::<LabResult<Vec<_>>>()?;
    let streams = sessions
        .iter()
        .map(|session| session.context.stream().map_err(|e| e.to_string()))
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions
        .iter()
        .map(|session| session.allocate(188 * 256 * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let inputs = sessions
        .iter()
        .map(|session| session.upload_f32(&[1.0; 8]))
        .collect::<LabResult<Vec<_>>>()?;
    // Measured about 6260 iterations per millisecond per CTA.
    let iterations = ((reduce_us / 1000.0 * 1.5 * 6260.0) as u32).max(1000);
    let compute_grid = 188_u32.saturating_sub(grid).max(1);
    let compute = sessions
        .iter()
        .enumerate()
        .map(|(rank, _)| {
            let function = modules[rank]
                .function("chop_probe_mma_bf16_v1")
                .map_err(|e| e.to_string())?;
            function
                .set_max_dynamic_shared_memory(80 * 1024)
                .map_err(|e| e.to_string())?;
            let stream = &streams[rank];
            stream.begin_capture().map_err(|e| e.to_string())?;
            function
                .launch(
                    stream,
                    launch::simple_config(
                        (compute_grid, 1, 1),
                        (256, 1, 1),
                        80 * 1024,
                        LaunchConfig::NO_CLUSTER,
                        false,
                    )?,
                    &[
                        KernelArgument::Buffer(&outputs[rank]),
                        KernelArgument::Buffer(&inputs[rank]),
                        KernelArgument::U32(iterations),
                    ],
                )
                .map_err(|e| e.to_string())?;
            stream
                .end_capture()
                .map_err(|e| e.to_string())?
                .instantiate()
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    // Wall time from the first start to the last end on each rank; slowest rank.
    let run = |with_reduce: bool, with_compute: bool| -> LabResult<(f32, f32, f32)> {
        let mut samples = Vec::new();
        for _ in 0..ROUNDS {
            let mut marks = Vec::new();
            for (rank, session) in sessions.iter().enumerate() {
                let events = (0..4)
                    .map(|_| session.context.timed_event().map_err(|e| e.to_string()))
                    .collect::<LabResult<Vec<_>>>()?;
                if with_reduce {
                    events[0]
                        .record(&session.stream)
                        .map_err(|e| e.to_string())?;
                    reduce[rank]
                        .launch(&session.stream)
                        .map_err(|e| e.to_string())?;
                    events[1]
                        .record(&session.stream)
                        .map_err(|e| e.to_string())?;
                }
                if with_compute {
                    events[2]
                        .record(&streams[rank])
                        .map_err(|e| e.to_string())?;
                    compute[rank]
                        .launch(&streams[rank])
                        .map_err(|e| e.to_string())?;
                    events[3]
                        .record(&streams[rank])
                        .map_err(|e| e.to_string())?;
                }
                marks.push(events);
            }
            let mut worst = (0.0_f32, 0.0_f32, 0.0_f32);
            for events in &marks {
                for event in events {
                    event.synchronize().map_err(|e| e.to_string())?;
                }
                let span = |a: usize, b: usize| {
                    events[a].elapsed_ms(&events[b]).map_err(|e| e.to_string())
                };
                let reduce_ms = if with_reduce { span(0, 1)? } else { 0.0 };
                let compute_ms = if with_compute { span(2, 3)? } else { 0.0 };
                let wall = match (with_reduce, with_compute) {
                    (true, true) => {
                        // Latest end minus earliest start, both relative to event 0.
                        let (end_reduce, start_compute, end_compute) =
                            (span(0, 1)?, span(0, 2)?, span(0, 3)?);
                        end_reduce.max(end_compute) - start_compute.min(0.0)
                    }
                    (true, false) => reduce_ms,
                    _ => compute_ms,
                };
                worst = (
                    worst.0.max(reduce_ms),
                    worst.1.max(compute_ms),
                    worst.2.max(wall),
                );
            }
            samples.push(worst);
        }
        samples.sort_by(|a, b| a.2.total_cmp(&b.2));
        Ok(samples[ROUNDS / 2])
    };
    run(true, true)?;
    let (reduce_alone, _, _) = run(true, false)?;
    let (_, compute_alone, _) = run(false, true)?;
    let (reduce_both, compute_both, wall_both) = run(true, true)?;
    let hidden = reduce_alone + compute_alone - wall_both;
    for (rank, session) in sessions.iter().enumerate() {
        session.receipt(
            "comm_bench",
            BTreeMap::from([
                ("case".into(), json!("bf16_prefill_overlap")),
                ("rows".into(), json!(rows)),
                ("ranks".into(), json!(sessions.len())),
                ("rank".into(), json!(rank)),
                ("allreduce_grid".into(), json!(grid)),
                (
                    "compute_stand_in".into(),
                    json!(format!(
                        "chop_probe_mma_bf16_v1, {compute_grid} CTAs, 80 KiB shared"
                    )),
                ),
                ("allreduce_alone_ms".into(), json!(reduce_alone)),
                ("compute_alone_ms".into(), json!(compute_alone)),
                ("allreduce_concurrent_ms".into(), json!(reduce_both)),
                ("compute_concurrent_ms".into(), json!(compute_both)),
                ("wall_concurrent_ms".into(), json!(wall_both)),
                ("hidden_ms".into(), json!(hidden)),
            ]),
        )?;
    }
    eprintln!(
        "overlap rows {rows} grid {grid}: all-reduce {reduce_alone:.3} ms, compute {compute_alone:.3} ms alone; together all-reduce {reduce_both:.3}, compute {compute_both:.3}, wall {wall_both:.3} ms; hidden {hidden:.3} ms"
    );
    Ok(())
}

/// Logits to rank 0 for `rows` rows: the bundle's one `allgather_bf16_v1`
/// launch per row against one `gather_rows_bf16_v1` launch.
fn run_logits(sessions: &[Session], rows: usize) -> LabResult<()> {
    const SHARD: usize = 38144;
    const STEPS: usize = 8;
    let ranks = sessions.len();
    let full = ranks * SHARD;
    let inputs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            session.upload_bf16(&vec![f32_to_bf16(rank as f32 + 1.0); rows * SHARD])
        })
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions
        .iter()
        .map(|session| session.allocate(rows * full * 2))
        .collect::<LabResult<Vec<_>>>()?;
    let zeroed = |bytes: usize| -> LabResult<Vec<DeviceBuffer>> {
        sessions
            .iter()
            .map(|session| {
                let buffer = session.allocate(bytes)?;
                buffer
                    .copy_from_bytes(&vec![0; bytes], &session.stream)
                    .map_err(|e| e.to_string())?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                Ok(buffer)
            })
            .collect()
    };
    let tiles = SHARD.div_ceil(4096);
    let old_flags = zeroed(2 * ranks * tiles * 4)?;
    let old_epochs = zeroed(tiles * 4)?;
    let new_flags = zeroed(kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES as usize)?;
    let new_epochs = zeroed(kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES as usize)?;
    let modules = sessions
        .iter()
        .map(|session| session.module("comm_sm120"))
        .collect::<LabResult<Vec<_>>>()?;
    let address = |buffer: &DeviceBuffer, offset: usize, bytes: usize| {
        buffer
            .device_address_at(offset, bytes)
            .map_err(|e| e.to_string())
    };
    let old_plan = kernels_sm120::comm::plan(
        &CommQuery::AllGather {
            ranks: ranks as u32,
            elements_per_rank: SHARD as u64,
            dtype: DType::Bf16,
        },
        1,
    )
    .map_err(|e| e.to_string())?;
    for batched in [false, true] {
        let graphs = sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| {
                let flags = (0..ranks)
                    .map(|peer| {
                        let buffer = if batched {
                            &new_flags[peer]
                        } else {
                            &old_flags[peer]
                        };
                        address(buffer, 0, buffer.byte_len())
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                let function = modules[rank]
                    .function(if batched {
                        "chop_comm_gather_rows_bf16_v1"
                    } else {
                        "chop_comm_allgather_bf16_v1"
                    })
                    .map_err(|e| e.to_string())?;
                session.capture(|_| {
                    for _ in 0..STEPS {
                        if batched {
                            let outs = (0..ranks)
                                .map(|peer| address(&outputs[peer], 0, rows * full * 2))
                                .collect::<LabResult<Vec<_>>>()?;
                            function
                                .launch(
                                    &session.stream,
                                    launch::config(kernels_sm120::comm::gather_rows_plan())?,
                                    &[
                                        KernelArgument::Buffer(&inputs[rank]),
                                        KernelArgument::DevicePointerArray(&outs),
                                        KernelArgument::DevicePointerArray(&flags),
                                        KernelArgument::Buffer(&new_epochs[rank]),
                                        KernelArgument::NullBuffer,
                                        KernelArgument::U32(rows as u32),
                                        KernelArgument::U64(SHARD as u64),
                                        KernelArgument::U32(rank as u32),
                                        KernelArgument::U32(ranks as u32),
                                        KernelArgument::U32(0),
                                    ],
                                )
                                .map_err(|e| e.to_string())?;
                            continue;
                        }
                        for row in 0..rows {
                            let outs = (0..ranks)
                                .map(|peer| address(&outputs[peer], row * full * 2, full * 2))
                                .collect::<LabResult<Vec<_>>>()?;
                            function
                                .launch(
                                    &session.stream,
                                    launch::config(old_plan)?,
                                    &[
                                        KernelArgument::Range {
                                            buffer: &inputs[rank],
                                            offset: row * SHARD * 2,
                                            byte_len: SHARD * 2,
                                        },
                                        KernelArgument::DevicePointerArray(&outs),
                                        KernelArgument::DevicePointerArray(&flags),
                                        KernelArgument::Buffer(&old_epochs[rank]),
                                        KernelArgument::U64(SHARD as u64),
                                        KernelArgument::U32(rank as u32),
                                        KernelArgument::U32(ranks as u32),
                                        KernelArgument::U32(0),
                                    ],
                                )
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    Ok(())
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        let timing = time_multi_graph(sessions, &graphs, STEPS, 1, TIMED_REPLAYS)?;
        // One more replay on poisoned outputs, checked below: both gathers
        // write only rank 0's output, every row in full (README "Logits
        // gather"; with a target rank the legacy entry writes only to it).
        poison_ranks(sessions, outputs.iter().enumerate())?;
        replay(sessions, &graphs)?;
        for (rank, session) in sessions.iter().enumerate() {
            let map = if rank == 0 {
                WriteMap::of(DType::Bf16).prefix(rows * full)?
            } else {
                WriteMap::of(DType::Bf16).none()?
            };
            check_written(
                session,
                &outputs[rank],
                &map,
                &format!("logits rows {rows} batched {batched} rank {rank} output"),
            )?;
        }
        let gathered = sessions[0].download_bf16(&outputs[0], rows * full)?;
        let correct = (0..rows * ranks).all(|chunk| {
            gathered[chunk * SHARD..(chunk + 1) * SHARD]
                .iter()
                .all(|&bits| bits == f32_to_bf16((chunk % ranks) as f32 + 1.0))
        });
        let us = timing.critical_path.median_ms * 1000.0 / STEPS as f32;
        for (rank, session) in sessions.iter().enumerate() {
            session.receipt(
                "comm_bench",
                BTreeMap::from([
                    ("case".into(), json!("bf16_logits_gather")),
                    (
                        "variant".into(),
                        json!(if batched {
                            "gather_rows"
                        } else {
                            "allgather_per_row"
                        }),
                    ),
                    ("rows".into(), json!(rows)),
                    ("ranks".into(), json!(ranks)),
                    ("rank".into(), json!(rank)),
                    (
                        "launches_per_step".into(),
                        json!(if batched { 1 } else { rows }),
                    ),
                    ("median_us_per_step".into(), json!(us)),
                    ("correct".into(), json!(correct)),
                ]),
            )?;
        }
        eprintln!(
            "logits rows {rows} {}: {us:.1} us per step, correct {correct}",
            if batched {
                "gather_rows"
            } else {
                "per-row allgather"
            }
        );
        if !correct {
            return Err(format!(
                "logits gather rows {rows} batched {batched} mismatch"
            ));
        }
    }
    Ok(())
}

/// F32 logits to rank 0 for `rows` rows with one `gather_rows_f32_v1` launch,
/// the F32 decode head's gather.
fn run_logits_f32(sessions: &[Session], rows: usize) -> LabResult<()> {
    const SHARD: usize = 38144;
    const STEPS: usize = 8;
    let ranks = sessions.len();
    let full = ranks * SHARD;
    let inputs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| session.upload_f32(&vec![rank as f32 + 1.5; rows * SHARD]))
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions
        .iter()
        .map(|session| session.allocate(rows * full * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let zeroed = |bytes: usize| -> LabResult<Vec<DeviceBuffer>> {
        sessions
            .iter()
            .map(|session| {
                let buffer = session.allocate(bytes)?;
                buffer
                    .copy_from_bytes(&vec![0; bytes], &session.stream)
                    .map_err(|e| e.to_string())?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                Ok(buffer)
            })
            .collect()
    };
    let flags = zeroed(kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES as usize)?;
    let epochs = zeroed(kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES as usize)?;
    let address = |buffer: &DeviceBuffer, bytes: usize| {
        buffer
            .device_address_at(0, bytes)
            .map_err(|e| e.to_string())
    };
    // The modules outlive the captured graphs.
    let modules = sessions
        .iter()
        .map(|session| session.module("comm_sm120"))
        .collect::<LabResult<Vec<_>>>()?;
    let graphs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            let function = modules[rank]
                .function("chop_comm_gather_rows_f32_v1")
                .map_err(|e| e.to_string())?;
            let flag_addresses = (0..ranks)
                .map(|peer| address(&flags[peer], flags[peer].byte_len()))
                .collect::<LabResult<Vec<_>>>()?;
            let outs = (0..ranks)
                .map(|peer| address(&outputs[peer], rows * full * 4))
                .collect::<LabResult<Vec<_>>>()?;
            session.capture(|_| {
                for _ in 0..STEPS {
                    function
                        .launch(
                            &session.stream,
                            launch::config(kernels_sm120::comm::gather_rows_plan())?,
                            &[
                                KernelArgument::Buffer(&inputs[rank]),
                                KernelArgument::DevicePointerArray(&outs),
                                KernelArgument::DevicePointerArray(&flag_addresses),
                                KernelArgument::Buffer(&epochs[rank]),
                                KernelArgument::NullBuffer,
                                KernelArgument::U32(rows as u32),
                                KernelArgument::U64(SHARD as u64),
                                KernelArgument::U32(rank as u32),
                                KernelArgument::U32(ranks as u32),
                                KernelArgument::U32(0),
                            ],
                        )
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let timing = time_multi_graph(sessions, &graphs, STEPS, 1, TIMED_REPLAYS)?;
    // One more replay on poisoned outputs, checked below: the gather writes
    // only rank 0's output, every row in full (README "Logits gather").
    poison_ranks(sessions, outputs.iter().enumerate())?;
    replay(sessions, &graphs)?;
    for (rank, session) in sessions.iter().enumerate() {
        let map = if rank == 0 {
            WriteMap::of(DType::F32).prefix(rows * full)?
        } else {
            WriteMap::of(DType::F32).none()?
        };
        check_written(
            session,
            &outputs[rank],
            &map,
            &format!("f32 logits rows {rows} rank {rank} output"),
        )?;
    }
    let gathered = sessions[0].download_f32(&outputs[0], rows * full)?;
    let correct = (0..rows * ranks).all(|chunk| {
        gathered[chunk * SHARD..(chunk + 1) * SHARD]
            .iter()
            .all(|&value| value == (chunk % ranks) as f32 + 1.5)
    });
    let us = timing.critical_path.median_ms * 1000.0 / STEPS as f32;
    for (rank, session) in sessions.iter().enumerate() {
        session.receipt(
            "comm_bench",
            BTreeMap::from([
                ("case".into(), json!("f32_logits_gather")),
                ("variant".into(), json!("gather_rows")),
                ("rows".into(), json!(rows)),
                ("ranks".into(), json!(ranks)),
                ("rank".into(), json!(rank)),
                ("median_us_per_step".into(), json!(us)),
                ("correct".into(), json!(correct)),
            ]),
        )?;
    }
    eprintln!("logits rows {rows} gather_rows f32: {us:.1} us per step, correct {correct}");
    if !correct {
        return Err(format!("f32 logits gather rows {rows} mismatch"));
    }
    Ok(())
}

/// One 16-bit lane of rank `rank`'s logits shard: `row`, lane `lane`, for
/// input pattern `seed`. Every lane differs by rank, row and column, so an
/// F32 value's two halves differ too (splitmix64 of the four indices).
fn qwen38_logits_lane(seed: u64, rank: usize, row: usize, lane: usize) -> u16 {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(((rank as u64) << 48) ^ ((row as u64) << 32) ^ lane as u64);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u16
}

/// Qwen3.8 logits to rank 0 for `rows` rows over its 62,080-column shard with
/// one `gather_rows_bf16_v1` launch: BF16 logits, then F32 logits moved as
/// BF16 pairs of twice the shard, as the bundle gathers them. Each lane
/// carries its rank, row and column, so a repeated row, a permuted column or
/// swapped F32 halves fail the check. The output starts from a sentinel and
/// ends in canary lanes that no rank may write. After timing, rank 0's
/// output is checked, the inputs are rewritten with a second pattern and the
/// graphs replayed, so stale rows from earlier replays fail as well. After
/// each timing one more replay runs on poisoned outputs and is checked too.
fn run_qwen38_logits(sessions: &[Session], rows: usize) -> LabResult<()> {
    const SHARD: usize = 62080;
    const STEPS: usize = 8;
    const CANARY: usize = 64;
    const SENTINEL: u16 = 0xA5C3;
    let ranks = sessions.len();
    let zeroed = |bytes: usize| -> LabResult<Vec<DeviceBuffer>> {
        sessions
            .iter()
            .map(|session| {
                let buffer = session.allocate(bytes)?;
                buffer
                    .copy_from_bytes(&vec![0; bytes], &session.stream)
                    .map_err(|e| e.to_string())?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                Ok(buffer)
            })
            .collect()
    };
    let address = |buffer: &DeviceBuffer, bytes: usize| {
        buffer
            .device_address_at(0, bytes)
            .map_err(|e| e.to_string())
    };
    let bytes = |lanes: &[u16]| {
        lanes
            .iter()
            .flat_map(|lane| lane.to_ne_bytes())
            .collect::<Vec<_>>()
    };
    let modules = sessions
        .iter()
        .map(|session| session.module("comm_sm120"))
        .collect::<LabResult<Vec<_>>>()?;
    for (dtype, shard) in [("bf16", SHARD), ("f32", 2 * SHARD)] {
        let full = ranks * shard;
        let input = |seed: u64, rank: usize| {
            (0..rows * shard)
                .map(|index| qwen38_logits_lane(seed, rank, index / shard, index % shard))
                .collect::<Vec<_>>()
        };
        let inputs = sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| session.upload_bf16(&input(0, rank)))
            .collect::<LabResult<Vec<_>>>()?;
        let outputs = zeroed((rows * full + CANARY) * 2)?;
        let flags = zeroed(kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES as usize)?;
        let epochs = zeroed(kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES as usize)?;
        // Every output lane, gathered rows and canaries, starts as the sentinel.
        let reset = || -> LabResult<()> {
            let sentinel = bytes(&vec![SENTINEL; rows * full + CANARY]);
            for (session, output) in sessions.iter().zip(&outputs) {
                output
                    .copy_from_bytes(&sentinel, &session.stream)
                    .map_err(|e| e.to_string())?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
            }
            Ok(())
        };
        // Rank 0 holds row r as the ranks' shards in rank order, then canaries.
        let check = |seed: u64| -> LabResult<bool> {
            let gathered = sessions[0].download_bf16(&outputs[0], rows * full + CANARY)?;
            let rows_match = (0..rows * full).all(|index| {
                let (row, column) = (index / full, index % full);
                gathered[index] == qwen38_logits_lane(seed, column / shard, row, column % shard)
            });
            Ok(rows_match && gathered[rows * full..].iter().all(|&lane| lane == SENTINEL))
        };
        reset()?;
        let graphs = sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| {
                let function = modules[rank]
                    .function("chop_comm_gather_rows_bf16_v1")
                    .map_err(|e| e.to_string())?;
                let flag_addresses = (0..ranks)
                    .map(|peer| address(&flags[peer], flags[peer].byte_len()))
                    .collect::<LabResult<Vec<_>>>()?;
                let outs = (0..ranks)
                    .map(|peer| address(&outputs[peer], rows * full * 2))
                    .collect::<LabResult<Vec<_>>>()?;
                session.capture(|_| {
                    for _ in 0..STEPS {
                        function
                            .launch(
                                &session.stream,
                                launch::config(kernels_sm120::comm::gather_rows_plan())?,
                                &[
                                    KernelArgument::Buffer(&inputs[rank]),
                                    KernelArgument::DevicePointerArray(&outs),
                                    KernelArgument::DevicePointerArray(&flag_addresses),
                                    KernelArgument::Buffer(&epochs[rank]),
                                    KernelArgument::NullBuffer,
                                    KernelArgument::U32(rows as u32),
                                    KernelArgument::U64(shard as u64),
                                    KernelArgument::U32(rank as u32),
                                    KernelArgument::U32(ranks as u32),
                                    KernelArgument::U32(0),
                                ],
                            )
                            .map_err(|e| e.to_string())?;
                    }
                    Ok(())
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        // One more replay per pattern, on poisoned outputs. Lanes are any
        // u16, 0xffff included, so rank 0's gathered lanes start from the
        // complement of the lanes they must receive (`Fill::Complement`);
        // its canaries and every other rank's output stay poison, since
        // senders write only the target (README "Logits gather"). Whether
        // rank 0 then holds exactly the pattern's lanes.
        let poisoned_replay = |seed: u64| -> LabResult<bool> {
            let lanes = rows * full;
            let mut image = bytes(
                &(0..lanes)
                    .map(|index| {
                        let (row, column) = (index / full, index % full);
                        qwen38_logits_lane(seed, column / shard, row, column % shard)
                    })
                    .collect::<Vec<_>>(),
            );
            image.resize((lanes + CANARY) * 2, POISON);
            let complement = image[..lanes * 2]
                .iter()
                .map(|byte| !byte)
                .collect::<Vec<_>>();
            poison_ranks(sessions, outputs.iter().enumerate())?;
            poison::write(&sessions[0], &outputs[0], 0, &complement)?;
            replay(sessions, &graphs)?;
            let mut exact = false;
            for (rank, (session, output)) in sessions.iter().zip(&outputs).enumerate() {
                let actual = poison::read(session, output)?;
                let label = format!("qwen38 {dtype} logits rows {rows} pattern {seed} rank {rank}");
                if rank == 0 {
                    poison::check(
                        &label,
                        &actual,
                        &WriteMap::of(DType::Bf16).prefix(lanes)?,
                        Fill::Complement {
                            pattern: &[POISON],
                            expected: &image,
                        },
                    )?;
                    exact = actual == image;
                } else {
                    poison::check(
                        &label,
                        &actual,
                        &WriteMap::of(DType::Bf16).none()?,
                        Fill::Pattern(&[POISON]),
                    )?;
                }
            }
            Ok(exact)
        };
        let timing = time_multi_graph(sessions, &graphs, STEPS, 1, TIMED_REPLAYS)?;
        let mut correct = check(0)?;
        correct &= poisoned_replay(0)?;
        // A second pattern through the same graphs: no row may come from the
        // first one.
        for (rank, (session, buffer)) in sessions.iter().zip(&inputs).enumerate() {
            buffer
                .copy_from_bytes(&bytes(&input(1, rank)), &session.stream)
                .map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
        }
        reset()?;
        time_multi_graph(sessions, &graphs, STEPS, 1, 3)?;
        correct &= check(1)?;
        correct &= poisoned_replay(1)?;
        let us = timing.critical_path.median_ms * 1000.0 / STEPS as f32;
        for (rank, session) in sessions.iter().enumerate() {
            session.receipt(
                "comm_bench",
                BTreeMap::from([
                    ("case".into(), json!("qwen38_logits_gather")),
                    ("variant".into(), json!("gather_rows")),
                    ("logits".into(), json!(dtype)),
                    ("rows".into(), json!(rows)),
                    ("ranks".into(), json!(ranks)),
                    ("rank".into(), json!(rank)),
                    ("median_us_per_step".into(), json!(us)),
                    ("correct".into(), json!(correct)),
                ]),
            )?;
        }
        eprintln!("qwen38 logits rows {rows} {dtype}: {us:.1} us per step, correct {correct}");
        if !correct {
            return Err(format!("qwen38 {dtype} logits gather rows {rows} mismatch"));
        }
    }
    Ok(())
}

/// The gather check's lane pattern tells every rank, row, column and F32
/// half apart, so the corruptions it must catch all change the lanes.
#[test]
fn qwen38_logits_lanes_expose_row_column_and_half_swaps() {
    const SHARD: usize = 2 * 62080;
    let lane = |rank, row, column| qwen38_logits_lane(0, rank, row, column);
    let layout = |rows: usize, ranks: usize| {
        (0..rows * ranks * SHARD)
            .map(|index| {
                let (row, column) = (index / (ranks * SHARD), index % (ranks * SHARD));
                lane(column / SHARD, row, column % SHARD)
            })
            .collect::<Vec<_>>()
    };
    let expected = layout(8, 4);
    // Row 0 copied into every row.
    let repeated = (0..expected.len())
        .map(|index| expected[index % (4 * SHARD)])
        .collect::<Vec<_>>();
    assert_ne!(repeated, expected);
    // Two columns of one row exchanged.
    let mut permuted = expected.clone();
    permuted.swap(3 * 4 * SHARD + 10, 3 * 4 * SHARD + 11);
    assert_ne!(permuted, expected);
    // The two halves of every F32 value exchanged.
    let halves = expected
        .chunks_exact(2)
        .flat_map(|pair| [pair[1], pair[0]])
        .collect::<Vec<_>>();
    assert_ne!(halves, expected);
    // Two ranks' shards exchanged.
    let mut ranks = expected.clone();
    ranks[..2 * SHARD].rotate_left(SHARD);
    assert_ne!(ranks, expected);
    // The second pattern differs from the first in almost every lane.
    let differ = (0..SHARD)
        .filter(|&column| {
            qwen38_logits_lane(1, 2, 5, column) != qwen38_logits_lane(0, 2, 5, column)
        })
        .count();
    assert!(differ > SHARD * 99 / 100, "{differ}");
}

/// Decode logits rows exchanged so each rank owns a block of full rows
/// (`chop_comm_exchange_rows_{f32,bf16}_v1`), against `gather_rows` to rank 0.
/// Every element carries its row, rank and column, and each owner's rows are
/// checked in full.
fn run_logits_exchange(sessions: &[Session], rows: usize, bf16: bool) -> LabResult<()> {
    const SHARD: usize = 38144;
    const STEPS: usize = 8;
    let ranks = sessions.len();
    let full = ranks * SHARD;
    let element = if bf16 { 2 } else { 4 };
    // Exact in BF16 (at most eight significant bits) for rows <= 32:
    // (row, rank) and the column's parity.
    let value = |row: usize, rank: usize, column: usize| {
        (row * 4 + rank) as f32 + (column % 2) as f32 * 0.5
    };
    let inputs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            let values: Vec<f32> = (0..rows * SHARD)
                .map(|index| value(index / SHARD, rank, index % SHARD))
                .collect();
            if bf16 {
                session.upload_bf16(&values.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>())
            } else {
                session.upload_f32(&values)
            }
        })
        .collect::<LabResult<Vec<_>>>()?;
    let zeroed = |bytes: usize| -> LabResult<Vec<DeviceBuffer>> {
        sessions
            .iter()
            .map(|session| {
                let buffer = session.allocate(bytes)?;
                buffer
                    .copy_from_bytes(&vec![0; bytes], &session.stream)
                    .map_err(|e| e.to_string())?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                Ok(buffer)
            })
            .collect()
    };
    let modules = sessions
        .iter()
        .map(|session| session.module("comm_sm120"))
        .collect::<LabResult<Vec<_>>>()?;
    let dtype = if bf16 { "bf16" } else { "f32" };
    for exchange in [false, true] {
        let outputs = zeroed(rows * full * element)?;
        let flags = zeroed(if exchange {
            kernels_sm120::comm::EXCHANGE_ROWS_FLAG_BYTES
        } else {
            kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES
        } as usize)?;
        let epochs = zeroed(kernels_sm120::comm::EXCHANGE_ROWS_EPOCH_BYTES as usize)?;
        let graphs = sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| {
                let function = modules[rank]
                    .function(&if exchange {
                        format!("chop_comm_exchange_rows_{dtype}_v1")
                    } else {
                        format!("chop_comm_gather_rows_{dtype}_v1")
                    })
                    .map_err(|e| e.to_string())?;
                let outs = outputs
                    .iter()
                    .map(|buffer| {
                        buffer
                            .device_address_at(0, buffer.byte_len())
                            .map_err(|e| e.to_string())
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                let flag_addresses = flags
                    .iter()
                    .map(|buffer| {
                        buffer
                            .device_address_at(0, buffer.byte_len())
                            .map_err(|e| e.to_string())
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                session.capture(|_| {
                    for _ in 0..STEPS {
                        if exchange {
                            function
                                .launch(
                                    &session.stream,
                                    launch::config(kernels_sm120::comm::exchange_rows_plan())?,
                                    &[
                                        KernelArgument::Buffer(&inputs[rank]),
                                        KernelArgument::DevicePointerArray(&outs),
                                        KernelArgument::DevicePointerArray(&flag_addresses),
                                        KernelArgument::Buffer(&epochs[rank]),
                                        KernelArgument::U32(rows as u32),
                                        KernelArgument::U64(SHARD as u64),
                                        KernelArgument::U32(rank as u32),
                                        KernelArgument::U32(ranks as u32),
                                    ],
                                )
                                .map_err(|e| e.to_string())?;
                        } else {
                            function
                                .launch(
                                    &session.stream,
                                    launch::config(kernels_sm120::comm::gather_rows_plan())?,
                                    &[
                                        KernelArgument::Buffer(&inputs[rank]),
                                        KernelArgument::DevicePointerArray(&outs),
                                        KernelArgument::DevicePointerArray(&flag_addresses),
                                        KernelArgument::Buffer(&epochs[rank]),
                                        KernelArgument::NullBuffer,
                                        KernelArgument::U32(rows as u32),
                                        KernelArgument::U64(SHARD as u64),
                                        KernelArgument::U32(rank as u32),
                                        KernelArgument::U32(ranks as u32),
                                        KernelArgument::U32(0),
                                    ],
                                )
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    Ok(())
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        let timing = time_multi_graph(sessions, &graphs, STEPS, 1, TIMED_REPLAYS)?;
        let us = timing.critical_path.median_ms * 1000.0 / STEPS as f32;
        // One more replay on poisoned outputs, checked below: each owner's
        // rows arrive in full and no other bytes are written (README "Logits
        // row exchange" and "Logits gather"), in elements of the payload.
        poison_ranks(sessions, outputs.iter().enumerate())?;
        replay(sessions, &graphs)?;
        // Each owner holds its rows in full; with the gather, rank 0 owns all.
        let mut correct = true;
        for (owner, session) in sessions.iter().enumerate() {
            let owned = if exchange {
                let range = kernels_sm120::comm::exchange_rows_owned(
                    rows as u32,
                    ranks as u32,
                    owner as u32,
                );
                range.start as usize..range.end as usize
            } else if owner == 0 {
                0..rows
            } else {
                0..0
            };
            check_written(
                session,
                &outputs[owner],
                &WriteMap::of(if bf16 { DType::Bf16 } else { DType::F32 })
                    .rows(full, owned.clone())?,
                &format!("logits rows {rows} {dtype} exchange {exchange} owner {owner} output"),
            )?;
            if owned.is_empty() {
                continue;
            }
            let got: Vec<f32> = if bf16 {
                session
                    .download_bf16(&outputs[owner], rows * full)?
                    .into_iter()
                    .map(kernel_core::codec::bf16_to_f32)
                    .collect()
            } else {
                session.download_f32(&outputs[owner], rows * full)?
            };
            for row in owned {
                for source in 0..ranks {
                    let base = (row * ranks + source) * SHARD;
                    correct &=
                        (0..SHARD).all(|column| got[base + column] == value(row, source, column));
                }
            }
        }
        for (rank, session) in sessions.iter().enumerate() {
            session.receipt(
                "comm_bench",
                BTreeMap::from([
                    ("case".into(), json!("logits_exchange")),
                    (
                        "variant".into(),
                        json!(if exchange {
                            "exchange_rows"
                        } else {
                            "gather_rows"
                        }),
                    ),
                    ("dtype".into(), json!(dtype)),
                    ("rows".into(), json!(rows)),
                    ("ranks".into(), json!(ranks)),
                    ("rank".into(), json!(rank)),
                    ("median_us_per_step".into(), json!(us)),
                    ("correct".into(), json!(correct)),
                ]),
            )?;
        }
        eprintln!(
            "logits rows {rows} {dtype} {}: {us:.1} us per step, correct {correct}",
            if exchange {
                "exchange_rows"
            } else {
                "gather_rows"
            }
        );
        if !correct {
            return Err(format!(
                "logits exchange {exchange} rows {rows} {dtype} mismatch"
            ));
        }
    }
    Ok(())
}

/// The MiMo decode head (32 rows, F32 or BF16 output) followed by the logits
/// gather to rank 0, against the head launched over `chunks` column chunks
/// whose gathers run on a second stream, each after its chunk and during the
/// next. Values are zero; this times data movement and overlap only. The
/// chunked gathers land chunk-major on rank 0, the same bytes as one gather.
fn run_head_overlap(sessions: &[Session], chunks: usize, bf16: bool) -> LabResult<()> {
    const ROWS: u32 = 32;
    const STEPS: usize = 4;
    const WEIGHT: u64 = 1;
    const ACTIVATION: u64 = 2;
    const OUTPUT: u64 = 3;
    const PARTIAL: u64 = 4;
    const SEMAPHORES: u64 = 5;
    let ranks = sessions.len();
    let element = if bf16 { 2 } else { 4 };
    // CHOPINFER_KERNEL_HEAD_SERIAL keeps chunk gathers on the head's stream.
    let serial = std::env::var("CHOPINFER_KERNEL_HEAD_SERIAL").as_deref() == Ok("1");
    // CHOPINFER_KERNEL_HEAD_GRID caps each head launch's CTAs.
    let grid_cap = std::env::var("CHOPINFER_KERNEL_HEAD_GRID")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(u32::MAX);
    let variant = kernels_sm120::gemm::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|item| item.symbol == "chop_gemm_mimo_head_m32_v1")
        .ok_or("missing MiMo head GEMM")?;
    let key = &variant.key;
    let kernel_core::contract::StaticQuery::Gemm(query) = key.query else {
        return Err("MiMo head is not a GEMM".into());
    };
    let (n, k) = (query.n as usize, query.k as usize);
    let tile = usize::from(key.knobs.tile_m);
    let splits = kernels_sm120::gemm::choose_split_k(key).map_err(|e| e.to_string())?;
    let counts = kernel_core::contract::RuntimeCounts {
        m: ROWS,
        batch: ROWS,
        context: 0,
        routed_rows: 0,
        split_k: splits,
    };
    let base_plan = kernels_sm120::gemm::plan(key, counts).map_err(|e| e.to_string())?;
    let maps =
        kernels_sm120::gemm::tensor_maps(key, WEIGHT, ACTIVATION).map_err(|e| e.to_string())?;
    let base_args = kernels_sm120::gemm::arguments(
        key,
        counts,
        maps,
        kernels_sm120::gemm::GemmBuffers {
            output: OUTPUT,
            partial: PARTIAL,
            semaphores: SEMAPHORES,
            weight_scales: None,
            activation_scales: None,
            residual: None,
            weight_global_scale: None,
            activation_global_scale: None,
        },
    )
    .map_err(|e| e.to_string())?;
    // Column chunks of whole tiles; one chunk is the unchunked head.
    let tiles = n / tile;
    let widths = |count: usize| -> Vec<usize> {
        (0..count)
            .map(|chunk| (tiles * (chunk + 1) / count - tiles * chunk / count) * tile)
            .collect()
    };
    let chunk_args = |width: usize| -> LabResult<ArgumentList> {
        let mut args = base_args.clone();
        let at = args
            .0
            .iter()
            .position(|arg| *arg == Argument::U32(n as u32))
            .ok_or("head arguments carry no width")?;
        args.0[at] = Argument::U32(width as u32);
        // live_m, n, k, splits, output_f32: F32 logits as the F32 head writes.
        args.0[at + 3] = Argument::U32(u32::from(!bf16));
        Ok(args)
    };
    let zeroed = |session: &Session, bytes: usize| -> LabResult<DeviceBuffer> {
        let buffer = session.allocate(bytes)?;
        buffer
            .copy_from_bytes(&vec![0; bytes], &session.stream)
            .map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        Ok(buffer)
    };
    let modules = sessions
        .iter()
        .map(|session| {
            Ok((
                session.module("gemm_mimo_sm120")?,
                session.module("comm_sm120")?,
            ))
        })
        .collect::<LabResult<Vec<_>>>()?;
    for count in [1, chunks] {
        let widths = widths(count);
        let mut rank_state = Vec::new();
        for session in sessions {
            let weights = widths
                .iter()
                .map(|width| zeroed(session, width * k * 2))
                .collect::<LabResult<Vec<_>>>()?;
            let outputs = widths
                .iter()
                .map(|width| zeroed(session, ROWS as usize * width * element))
                .collect::<LabResult<Vec<_>>>()?;
            let partials = widths
                .iter()
                .map(|_| zeroed(session, base_plan.workspace_bytes as usize))
                .collect::<LabResult<Vec<_>>>()?;
            let semaphores = widths
                .iter()
                .map(|_| zeroed(session, 1 << 16))
                .collect::<LabResult<Vec<_>>>()?;
            let flags = widths
                .iter()
                .map(|_| {
                    zeroed(
                        session,
                        kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES as usize,
                    )
                })
                .collect::<LabResult<Vec<_>>>()?;
            let epochs = widths
                .iter()
                .map(|_| {
                    zeroed(
                        session,
                        kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES as usize,
                    )
                })
                .collect::<LabResult<Vec<_>>>()?;
            let gathered = zeroed(session, ROWS as usize * n * ranks * element)?;
            let activation = zeroed(session, ROWS as usize * k * 2)?;
            let side = session.context.stream().map_err(|e| e.to_string())?;
            let events = (0..=widths.len())
                .map(|_| session.context.event().map_err(|e| e.to_string()))
                .collect::<LabResult<Vec<_>>>()?;
            rank_state.push((
                weights, outputs, partials, semaphores, flags, epochs, gathered, activation, side,
                events,
            ));
        }
        let graphs = sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| {
                let (gemm_module, comm_module) = &modules[rank];
                let head = gemm_module
                    .function(&variant.symbol)
                    .map_err(|e| e.to_string())?;
                let gather = comm_module
                    .function(if bf16 {
                        "chop_comm_gather_rows_bf16_v1"
                    } else {
                        "chop_comm_gather_rows_f32_v1"
                    })
                    .map_err(|e| e.to_string())?;
                let (
                    weights,
                    outputs,
                    partials,
                    semaphores,
                    _,
                    epochs,
                    _,
                    activation,
                    side,
                    events,
                ) = &rank_state[rank];
                session.capture(|stream| {
                    for _ in 0..STEPS {
                        if count > 1 {
                            events[widths.len()]
                                .record(stream)
                                .map_err(|e| e.to_string())?;
                            side.wait_event(&events[widths.len()])
                                .map_err(|e| e.to_string())?;
                        }
                        let mut offset = 0;
                        for (chunk, &width) in widths.iter().enumerate() {
                            let mut plan = base_plan;
                            plan.grid[0] = plan.grid[0]
                                .min((width / tile * usize::from(splits)) as u32)
                                .min(grid_cap);
                            launch::launch(
                                &session.context,
                                &head,
                                stream,
                                plan,
                                &chunk_args(width)?,
                                &Bindings {
                                    buffers: BTreeMap::from([
                                        (WEIGHT, &weights[chunk]),
                                        (ACTIVATION, activation),
                                        (OUTPUT, &outputs[chunk]),
                                        (PARTIAL, &partials[chunk]),
                                        (SEMAPHORES, &semaphores[chunk]),
                                    ]),
                                },
                            )?;
                            let gather_stream = if count > 1 && !serial {
                                events[chunk].record(stream).map_err(|e| e.to_string())?;
                                side.wait_event(&events[chunk]).map_err(|e| e.to_string())?;
                                side
                            } else {
                                stream
                            };
                            let outs = (0..ranks)
                                .map(|peer| {
                                    rank_state[peer]
                                        .6
                                        .device_address_at(
                                            ROWS as usize * offset * ranks * element,
                                            ROWS as usize * width * ranks * element,
                                        )
                                        .map_err(|e| e.to_string())
                                })
                                .collect::<LabResult<Vec<_>>>()?;
                            let flag_addresses = (0..ranks)
                                .map(|peer| {
                                    let buffer = &rank_state[peer].4[chunk];
                                    buffer
                                        .device_address_at(0, buffer.byte_len())
                                        .map_err(|e| e.to_string())
                                })
                                .collect::<LabResult<Vec<_>>>()?;
                            gather
                                .launch(
                                    gather_stream,
                                    launch::config(kernels_sm120::comm::gather_rows_plan())?,
                                    &[
                                        KernelArgument::Buffer(&outputs[chunk]),
                                        KernelArgument::DevicePointerArray(&outs),
                                        KernelArgument::DevicePointerArray(&flag_addresses),
                                        KernelArgument::Buffer(&epochs[chunk]),
                                        KernelArgument::NullBuffer,
                                        KernelArgument::U32(ROWS),
                                        KernelArgument::U64(width as u64),
                                        KernelArgument::U32(rank as u32),
                                        KernelArgument::U32(ranks as u32),
                                        KernelArgument::U32(0),
                                    ],
                                )
                                .map_err(|e| e.to_string())?;
                            offset += width;
                        }
                        if count > 1 {
                            events[widths.len()]
                                .record(side)
                                .map_err(|e| e.to_string())?;
                            stream
                                .wait_event(&events[widths.len()])
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    Ok(())
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        let timing = time_multi_graph(sessions, &graphs, STEPS, 1, TIMED_REPLAYS)?;
        let us = timing.critical_path.median_ms * 1000.0 / STEPS as f32;
        for (rank, session) in sessions.iter().enumerate() {
            session.receipt(
                "comm_bench",
                BTreeMap::from([
                    ("case".into(), json!("head_gather_overlap")),
                    ("logits".into(), json!(if bf16 { "bf16" } else { "f32" })),
                    ("serial_gathers".into(), json!(serial)),
                    ("grid_cap".into(), json!(grid_cap)),
                    ("chunks".into(), json!(count)),
                    ("chunk_columns".into(), json!(widths)),
                    ("rows".into(), json!(ROWS)),
                    ("rank".into(), json!(rank)),
                    ("median_us_per_step".into(), json!(us)),
                ]),
            )?;
        }
        eprintln!("head and gather, bf16 {bf16}, {count} chunk(s): {us:.1} us per step");
    }
    Ok(())
}

/// Legacy one-shot, one CTA per row with inbox slots and acknowledgments.
/// `Plain` is the unfused all-reduce; `Serial` adds residual and RMSNorm.
fn run_case(sessions: &[Session], tokens: usize, variant: Variant) -> LabResult<()> {
    let ranks = sessions.len();
    let elements = 4096 * tokens;
    let plan = LaunchPlan {
        grid: [tokens as u32, 1, 1],
        block: [256, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    let inputs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| session.upload_bf16(&vec![f32_to_bf16((rank + 1) as f32); elements]))
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions
        .iter()
        .map(|session| session.allocate(elements * 2))
        .collect::<LabResult<Vec<_>>>()?;
    let inboxes = sessions
        .iter()
        .map(|session| session.allocate(ranks * elements * 2))
        .collect::<LabResult<Vec<_>>>()?;
    let flag_bytes = 2 * ranks * tokens * 4;
    let zeroed = |bytes: usize| -> LabResult<Vec<DeviceBuffer>> {
        sessions
            .iter()
            .map(|session| {
                let buffer = session.allocate(bytes)?;
                buffer
                    .copy_from_bytes(&vec![0; bytes], &session.stream)
                    .map_err(|e| e.to_string())?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                Ok(buffer)
            })
            .collect()
    };
    let flags = zeroed(flag_bytes)?;
    let epochs = zeroed(tokens * 4)?;
    let residuals = sessions
        .iter()
        .map(|session| session.upload_bf16(&vec![f32_to_bf16(1.0); elements]))
        .collect::<LabResult<Vec<_>>>()?;
    let weights = sessions
        .iter()
        .map(|session| session.upload_bf16(&vec![f32_to_bf16(1.0); 4096]))
        .collect::<LabResult<Vec<_>>>()?;
    let symbol = match variant {
        Variant::Plain => "chop_comm_allreduce_oneshot_bf16_v1",
        Variant::Serial => "chop_comm_allreduce_oneshot_norm_bf16_v1",
    };
    let functions = sessions
        .iter()
        .map(|session| {
            session
                .module("comm_sm120")?
                .function(symbol)
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let handles: Vec<u64> = (0..ranks as u64).map(|rank| rank + 1).collect();
    let graphs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
            buffers.insert(100, &inputs[rank]);
            buffers.insert(101, &outputs[rank]);
            buffers.insert(102, &epochs[rank]);
            buffers.insert(103, &residuals[rank]);
            buffers.insert(104, &weights[rank]);
            for peer in 0..ranks {
                buffers.insert(peer as u64 + 1, &inboxes[peer]);
                buffers.insert(peer as u64 + 11, &flags[peer]);
            }
            let bindings = Bindings { buffers };
            let base = kernels_sm120::comm::OneShotCall {
                input: 100,
                output: 101,
                peer_inboxes: handles.clone(),
                peer_flags: handles.iter().map(|handle| handle + 10).collect(),
                epoch: 102,
                elements: elements as u64,
                rank: rank as u32,
                ranks: ranks as u32,
            };
            let args = match variant {
                Variant::Plain => kernels_sm120::comm::arguments(base),
                Variant::Serial => {
                    kernels_sm120::comm::fused_arguments(kernels_sm120::comm::FusedOneShotCall {
                        base,
                        residual: Some(103),
                        weights: 104,
                        epsilon: 1e-6,
                    })
                }
            }
            .map_err(|e| e.to_string())?;
            session.capture(|_| {
                for _ in 0..CALLS_PER_GRAPH {
                    launch::launch(
                        &session.context,
                        &functions[rank],
                        &session.stream,
                        plan,
                        &args,
                        &bindings,
                    )?;
                }
                Ok(())
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let timing = time_multi_graph(
        sessions,
        &graphs,
        CALLS_PER_GRAPH,
        WARMUP_REPLAYS,
        TIMED_REPLAYS,
    )?;
    // One more replay on poisoned outputs, checked below: both legacy
    // entries write every element of the rank's own output and only read
    // the residual (comm.cuh `allreduce_oneshot` and
    // `allreduce_oneshot_norm_bf16`; the README is silent).
    poison_ranks(sessions, outputs.iter().enumerate())?;
    replay(sessions, &graphs)?;
    let written = WriteMap::of(DType::Bf16).prefix(elements)?;
    let sum = f32_to_bf16(
        (1..=ranks).sum::<usize>() as f32 + if variant == Variant::Plain { 0.0 } else { 1.0 },
    );
    let expected = match variant {
        Variant::Plain => vec![sum; elements],
        Variant::Serial => NORM
            .apply_bf16(&vec![sum; elements], &vec![f32_to_bf16(1.0); 4096], 4096)
            .map_err(|e| e.to_string())?,
    };
    let divisor = CALLS_PER_GRAPH as f32 / 1000.0;
    for (rank, session) in sessions.iter().enumerate() {
        check_written(
            session,
            &outputs[rank],
            &written,
            &format!("{} token {tokens} rank {rank} output", variant.name()),
        )?;
        let gate = check_exact(&session.download_bf16(&outputs[rank], elements)?, &expected);
        session.receipt(
            "comm_bench",
            BTreeMap::from([
                ("case".into(), json!("bf16_fused_decode_latency")),
                ("variant".into(), json!(variant.name())),
                ("tokens".into(), json!(tokens)),
                ("elements".into(), json!(elements)),
                ("ranks".into(), json!(ranks)),
                ("rank".into(), json!(rank)),
                ("calls_per_graph".into(), json!(CALLS_PER_GRAPH)),
                ("pdl".into(), json!(plan.pdl)),
                (
                    "median_us_per_call".into(),
                    json!(timing.critical_path.median_ms / divisor),
                ),
                (
                    "min_us_per_call".into(),
                    json!(timing.critical_path.min_ms / divisor),
                ),
                ("correct".into(), json!(gate.is_ok())),
                ("input_pattern".into(), json!("constant")),
            ]),
        )?;
        gate.map_err(|index| {
            format!(
                "{} token {tokens} rank {rank} differs at {index}",
                variant.name()
            )
        })?;
    }
    Ok(())
}

#[test]
fn bf16_fused_decode_latency() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        eprintln!("skipping comm latency benchmark: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs");
        return;
    }
    let selected_tokens = std::env::var("CHOPINFER_KERNEL_BENCH_TOKENS")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("integer benchmark token count")
        });
    if let Some(tokens) = selected_tokens {
        assert!(
            [1, 8, 16, 32].contains(&tokens),
            "unsupported benchmark token count"
        );
    }
    let selected_variant = std::env::var("CHOPINFER_KERNEL_BENCH_VARIANT").ok();
    if let Some(name) = selected_variant.as_deref() {
        assert!(
            [
                "launch_floor",
                "local_copy",
                "local_norm",
                "local_flags",
                "local_one_shot",
                "local_plain_one_shot",
                "one_shot",
                "plain_one_shot",
                "peer_control",
                "peer_control_local",
                "push",
                "local_push",
                "prefill",
                "logits",
                "logits_f32",
                "qwen38_logits",
                "head_overlap",
                "logits_exchange",
                "wide",
                "wide_combine",
                "peer_push_fanout",
                "peer_push_pull"
            ]
            .contains(&name),
            "unsupported benchmark variant"
        );
    }
    let tokens = [1, 8, 16, 32]
        .into_iter()
        .filter(|tokens| selected_tokens.is_none_or(|selected| selected == *tokens))
        .collect::<Vec<_>>();
    gpu_test_multi(64 * 1024 * 1024, |sessions| {
        match selected_variant.as_deref() {
            Some("launch_floor") => run_launch_floor(sessions),
            Some(name @ ("peer_control" | "peer_control_local")) => run_peer_control(
                sessions,
                selected_tokens.unwrap_or(32),
                name == "peer_control_local",
            ),
            Some(name @ ("local_copy" | "local_norm" | "local_flags")) => {
                run_local_micro(&sessions[0], name)
            }
            Some("local_one_shot") => run_case(
                &sessions[..1],
                selected_tokens.unwrap_or(1),
                Variant::Serial,
            ),
            Some("local_plain_one_shot") => {
                run_case(&sessions[..1], selected_tokens.unwrap_or(1), Variant::Plain)
            }
            Some("prefill") => {
                let rows = std::env::var("CHOPINFER_KERNEL_BENCH_ROWS")
                    .ok()
                    .map(|value| vec![value.parse::<usize>().expect("integer row count")])
                    .unwrap_or_else(|| vec![512, 2048, 8192]);
                rows.into_iter()
                    .try_for_each(|rows| run_prefill(sessions, rows))
            }
            Some(name @ ("wide" | "wide_combine")) => {
                let rows = std::env::var("CHOPINFER_KERNEL_BENCH_ROWS")
                    .ok()
                    .map(|value| {
                        value
                            .split(',')
                            .map(|rows| rows.trim().parse::<usize>().expect("integer row count"))
                            .collect()
                    })
                    .unwrap_or_else(|| vec![33, 64, 100, 128, 255, 256]);
                rows.into_iter()
                    .try_for_each(|rows| run_wide(sessions, rows, name == "wide_combine"))
            }
            Some("logits") => [1, 8, 32]
                .into_iter()
                .try_for_each(|rows| run_logits(sessions, rows)),
            Some("logits_f32") => [1, 8, 32]
                .into_iter()
                .try_for_each(|rows| run_logits_f32(sessions, rows)),
            Some("qwen38_logits") => [1, 8, 16, 32]
                .into_iter()
                .try_for_each(|rows| run_qwen38_logits(sessions, rows)),
            // CHOPINFER_KERNEL_BENCH_ROWS picks one row count; above 32 rows
            // (verify steps) only F32 values stay exact.
            Some("logits_exchange") => std::env::var("CHOPINFER_KERNEL_BENCH_ROWS")
                .ok()
                .map(|value| vec![value.parse::<usize>().expect("integer row count")])
                .unwrap_or_else(|| vec![1, 2, 5, 8, 17, 32])
                .into_iter()
                .try_for_each(|rows| {
                    run_logits_exchange(sessions, rows, false)?;
                    if rows <= 32 {
                        run_logits_exchange(sessions, rows, true)?;
                    }
                    Ok(())
                }),
            Some("head_overlap") => run_head_overlap(
                sessions,
                std::env::var("CHOPINFER_KERNEL_HEAD_CHUNKS")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(3),
                std::env::var("CHOPINFER_KERNEL_HEAD_BF16").as_deref() == Ok("1"),
            ),
            Some("peer_push_pull") => [65536, 262144]
                .into_iter()
                .try_for_each(|bytes| run_push_pull(sessions, bytes)),
            Some("peer_push_fanout") => [8192, 65536, 262144]
                .into_iter()
                .try_for_each(|bytes| run_push_fanout(sessions, bytes)),
            Some(name @ ("push" | "local_push")) => {
                let ranks = if name == "push" { sessions.len() } else { 1 };
                tokens
                    .iter()
                    .try_for_each(|&tokens| run_push(&sessions[..ranks], tokens))
            }
            Some("one_shot") => tokens
                .iter()
                .try_for_each(|&tokens| run_case(sessions, tokens, Variant::Serial)),
            Some("plain_one_shot") => tokens
                .iter()
                .try_for_each(|&tokens| run_case(sessions, tokens, Variant::Plain)),
            _ => tokens.iter().try_for_each(|&tokens| {
                run_push(sessions, tokens)?;
                run_case(sessions, tokens, Variant::Serial)
            }),
        }
    })
    .expect("multi-GPU comm latency benchmark");
}
