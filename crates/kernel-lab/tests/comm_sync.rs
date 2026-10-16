//! Cross-rank synchronization stress for the comm protocols. The tests load
//! the `comm_sm120_skew` cubin (entries `chop_comm_skew_*`): in every call the rank chosen by the call's
//! epoch lags 20 us at a protocol step (`test_skew` in `comm.cuh`), so its
//! peers run as far ahead as each protocol allows. Calls are queued on every
//! rank without host synchronization, eagerly and as captured graphs, on
//! one comm state whose epochs start near the u32 wrap.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{
    Argument, ArgumentList, CommQuery, DType, LaunchPlan, RmsNormEpilogue,
};
use kernel_core::oracle;
use kernel_core::oracle::{NormArithmetic, NormOffset, NormScale, RmsNorm};
use kernel_lab::launch::{self, Bindings, CollectiveModules, RankCall};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, gpu_test_multi};
use runtime_cuda::DeviceBuffer;
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

const CUBIN: &str = "comm_sm120_skew";

/// The skew cubin's name of a comm entry.
fn skew(symbol: &str) -> String {
    symbol.replacen("chop_comm_", "chop_comm_skew_", 1)
}
const MAX_ROWS: usize = 2048;
const FLAG_BYTES: usize = 4096;
const EPOCH_CELLS: usize = 32;
const REPLAYS: usize = 3;
/// Epoch cells start here, so the calls cross the u32 wrap.
const NEAR_WRAP: u32 = u32::MAX - 20;

/// Deterministic BF16 test values in [-scale, scale].
fn pattern(seed: u64, len: usize, scale: f32) -> Vec<u16> {
    (0..len as u64)
        .map(|index| {
            let mut x = (index ^ seed.rotate_left(17)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            x ^= x >> 29;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x ^= x >> 32;
            f32_to_bf16(((x % 2001) as f32 - 1000.0) / 1000.0 * scale)
        })
        .collect()
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len())?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn synchronize(sessions: &[Session]) -> LabResult<()> {
    for session in sessions {
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    Ok(())
}

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

/// The bytes of `buffer` after launches on poisoned outputs: every element
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

/// One rank's hidden comm state, shared by every call of a sequence, and a
/// separate input and output per call so every call's result is kept.
struct HiddenRank {
    inbox: DeviceBuffer,
    flags: DeviceBuffer,
    epoch: DeviceBuffer,
    residual: DeviceBuffer,
    weights: DeviceBuffer,
    inputs: Vec<DeviceBuffer>,
    outputs: Vec<DeviceBuffer>,
}

/// One call of a sequence: its rows, and whether a decode or verify graph
/// issues it (`graph_choose`: the wide entry above 32 rows) rather than
/// prefill (`choose`: the phased pair above 32 rows).
#[derive(Clone, Copy)]
struct Call {
    rows: usize,
    graph: bool,
}

fn prefill(rows: &[usize]) -> Vec<Call> {
    rows.iter()
        .map(|&rows| Call { rows, graph: false })
        .collect()
}

/// Fused hidden all-reduce `call` with the bundle's entries: the v2 push
/// entry up to 32 rows, the wide entry above in graphs and the phased pair
/// in prefill. Each stage is one collective call of every rank (or of
/// `only_rank`'s device).
fn hidden_call(
    sessions: &[Session],
    modules: &CollectiveModules,
    ranks: &[HiddenRank],
    call: usize,
    shape: Call,
    only_rank: Option<usize>,
) -> LabResult<()> {
    let rows = shape.rows;
    let count = sessions.len();
    let elements = (rows * 4096) as u64;
    let query = CommQuery::AllReduce {
        ranks: count as u32,
        elements,
        dtype: DType::Bf16,
        fused_residual: true,
        fused_rmsnorm: Some(RmsNormEpilogue {
            kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
            width: 4096,
            eps: 1e-6,
        }),
    };
    let (variant, plan) = if shape.graph {
        (
            kernels_sm120::comm::graph_choose(&query),
            kernels_sm120::comm::graph_plan(&query, rows as u32),
        )
    } else {
        (
            kernels_sm120::comm::choose(&query),
            kernels_sm120::comm::plan(&query, rows as u32),
        )
    };
    let (variant, plan) = (
        variant.map_err(|e| e.to_string())?,
        plan.map_err(|e| e.to_string())?,
    );
    let inboxes: Vec<u64> = (0..count as u64).map(|rank| rank + 1).collect();
    let flags: Vec<u64> = (0..count as u64).map(|rank| rank + 11).collect();
    let outputs: Vec<u64> = (0..count as u64).map(|rank| rank + 21).collect();
    let bindings = |rank: usize| {
        let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
        buffers.insert(100, &ranks[rank].inputs[call]);
        buffers.insert(101, &ranks[rank].outputs[call]);
        buffers.insert(102, &ranks[rank].epoch);
        buffers.insert(103, &ranks[rank].residual);
        buffers.insert(104, &ranks[rank].weights);
        for (peer, state) in ranks.iter().enumerate() {
            buffers.insert(peer as u64 + 1, &state.inbox);
            buffers.insert(peer as u64 + 11, &state.flags);
            buffers.insert(peer as u64 + 21, &state.outputs[call]);
        }
        Bindings { buffers }
    };
    let stages = |rank: usize| -> LabResult<Vec<(&str, ArgumentList)>> {
        let base = kernels_sm120::comm::OneShotCall {
            input: 100,
            output: 101,
            peer_inboxes: inboxes.clone(),
            peer_flags: flags.clone(),
            epoch: 102,
            elements,
            rank: rank as u32,
            ranks: count as u32,
        };
        Ok(match variant {
            kernels_sm120::comm::Variant::FusedOneShot
            | kernels_sm120::comm::Variant::FusedWide => vec![(
                if variant == kernels_sm120::comm::Variant::FusedWide {
                    kernels_sm120::comm::WIDE_SYMBOL
                } else {
                    "chop_comm_allreduce_oneshot_norm_bf16_v2"
                },
                kernels_sm120::comm::fused_arguments_v2(
                    kernels_sm120::comm::FusedWithResidualOutputCall {
                        fused: kernels_sm120::comm::FusedOneShotCall {
                            base,
                            residual: Some(103),
                            weights: 104,
                            epsilon: 1e-6,
                        },
                        residual_output: 103,
                    },
                )
                .map_err(|e| e.to_string())?,
            )],
            kernels_sm120::comm::Variant::FusedTwoShot => {
                let gather = kernels_sm120::comm::round_once_gather_arguments(
                    &base,
                    outputs.clone(),
                    104,
                    1e-6,
                    103,
                )
                .map_err(|e| e.to_string())?;
                vec![
                    (
                        kernels_sm120::comm::PREFILL_ROUND_ONCE_SYMBOL,
                        kernels_sm120::comm::round_once_scatter_arguments(base, 103)
                            .map_err(|e| e.to_string())?,
                    ),
                    (
                        kernels_sm120::comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL,
                        gather,
                    ),
                ]
            }
            other => return Err(format!("unexpected fused variant {other:?}")),
        })
    };
    for (stage, (symbol, _)) in stages(0)?.into_iter().enumerate() {
        launch::launch_collective(sessions, modules, &skew(symbol), plan, only_rank, |rank| {
            Ok(RankCall {
                args: stages(rank)?.swap_remove(stage).1,
                bindings: bindings(rank),
            })
        })?;
    }
    Ok(())
}

/// Host replay of one call on the residual chain: the push, wide and phased
/// paths all add the residual to the rank-ordered F32 sum and round once.
fn host_call(inputs: &[Vec<Vec<u16>>], residual: &mut [u16], call: usize, rows: usize) {
    for (index, value) in residual[..rows * 4096].iter_mut().enumerate() {
        let sum = inputs
            .iter()
            .fold(0.0_f32, |sum, rank| sum + bf16_to_f32(rank[call][index]));
        *value = f32_to_bf16(sum + bf16_to_f32(*value));
    }
}

/// The LL flag of `epoch`, as `ll_flag` in `push.cuh`.
fn ll_flag(epoch: u32) -> u32 {
    0xffc0_ffc0 | (epoch & 0x3f) | (((epoch >> 6) & 0x3f) << 16)
}

/// Inbox bytes that look like the first two low-latency calls' data: every
/// 8-byte entry carries the flag of the next call of its parity bank and a
/// payload of 512.0 pairs. The rest of the inbox repeats the first entry.
fn stale_inbox(ranks: usize, bytes: usize) -> Vec<u8> {
    let plane = 2 * 2048;
    (0..bytes / 8)
        .flat_map(|entry| {
            let bank = (entry / (ranks * plane)) as u32 & 1;
            let first = NEAR_WRAP.wrapping_add(1);
            let epoch = if first & 1 == bank {
                first
            } else {
                first.wrapping_add(1)
            };
            ((u64::from(ll_flag(epoch)) << 32) | 0x4400_4400).to_ne_bytes()
        })
        .collect()
}

/// Runs `rows` once eagerly and `REPLAYS` times as one captured graph per
/// rank, without host synchronization between the calls of either group,
/// each group on poisoned outputs, then checks the residual chain exactly
/// and every call's normalized output of the group's last pass on every
/// rank. The inbox starts as `stale_inbox`; unless `skip_init` the
/// low-latency region is then zeroed as `PushBuffers` requires.
fn run_hidden_sequence(sessions: &[Session], label: &str, rows: &[usize]) -> LabResult<()> {
    run_hidden_sequence_from(sessions, label, &prefill(rows), false)
}

fn run_hidden_sequence_from(
    sessions: &[Session],
    label: &str,
    calls: &[Call],
    skip_init: bool,
) -> LabResult<()> {
    let rows: Vec<usize> = calls.iter().map(|call| call.rows).collect();
    let rows = rows.as_slice();
    let count = sessions.len();
    let elements = MAX_ROWS * 4096;
    let host_inputs = (0..count)
        .map(|rank| {
            rows.iter()
                .enumerate()
                .map(|(call, &rows)| pattern((rank * 131 + call) as u64 + 7, rows * 4096, 0.05))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let host_residual = pattern(98, elements, 1.0);
    let host_weights = pattern(9, 4096, 1.0)
        .into_iter()
        .map(|bits| f32_to_bf16(1.0 + 0.5 * bf16_to_f32(bits)))
        .collect::<Vec<_>>();
    let phased = kernels_sm120::comm::phased_buffers(count as u32, elements as u64, DType::Bf16)
        .map_err(|e| e.to_string())?;
    let push = kernels_sm120::comm::push_buffers(count as u32, 32).map_err(|e| e.to_string())?;
    let inbox_bytes = phased.inbox_bytes.max(push.inbox_bytes) as usize;
    let ranks = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            let state = HiddenRank {
                inbox: upload(session, &stale_inbox(count, inbox_bytes))?,
                flags: upload(session, &vec![0; FLAG_BYTES])?,
                epoch: upload(session, &u32_bytes(&[NEAR_WRAP; EPOCH_CELLS]))?,
                residual: session.upload_bf16(&host_residual)?,
                weights: session.upload_bf16(&host_weights)?,
                inputs: host_inputs[rank]
                    .iter()
                    .map(|values| session.upload_bf16(values))
                    .collect::<LabResult<Vec<_>>>()?,
                outputs: rows
                    .iter()
                    .map(|&rows| session.allocate(rows * 4096 * 2))
                    .collect::<LabResult<Vec<_>>>()?,
            };
            if !skip_init {
                state
                    .inbox
                    .zero_at_async(0, push.zeroed_inbox_bytes as usize, &session.stream)
                    .map_err(|e| e.to_string())?;
            }
            session.stream.synchronize().map_err(|e| e.to_string())?;
            Ok(state)
        })
        .collect::<LabResult<Vec<_>>>()?;
    // Module loads synchronize the device, so load before any peer waits.
    let modules = CollectiveModules::load(sessions, CUBIN)?;
    // Checks every rank after `passes` passes over the calls: the residual
    // chain exactly, then each call's output of the last pass. The push,
    // wide and phased entries write all rows of the call's own output
    // (README "Fused decode all-reduce", "Wide all-reduce for graphs above
    // 32 rows" and "Prefill all-reduce").
    let check = |passes: usize, phase: &str| -> LabResult<()> {
        let mut host_residual = host_residual.clone();
        let mut snapshots = Vec::new();
        for pass in 0..passes {
            for (call, &shape) in calls.iter().enumerate() {
                host_call(&host_inputs, &mut host_residual, call, shape.rows);
                if pass + 1 == passes {
                    snapshots.push(host_residual[..shape.rows * 4096].to_vec());
                }
            }
        }
        for (rank, session) in sessions.iter().enumerate() {
            let residual = session.download_bf16(&ranks[rank].residual, elements)?;
            oracle::check_exact(&residual, &host_residual).map_err(|index| {
                format!("{label} {phase} rank {rank}: residual differs at {index}")
            })?;
        }
        for (call, &rows) in rows.iter().enumerate() {
            let written = WriteMap::of(DType::Bf16).prefix(rows * 4096)?;
            for (rank, session) in sessions.iter().enumerate() {
                check_written(
                    session,
                    &ranks[rank].outputs[call],
                    &written,
                    &format!("{label} {phase} call {call} ({rows} rows) rank {rank} output"),
                )?;
            }
            let expected = NORM
                .apply_bf16(&snapshots[call], &host_weights, 4096)
                .map_err(|e| e.to_string())?;
            let first = sessions[0].download_bf16(&ranks[0].outputs[call], rows * 4096)?;
            for (rank, session) in sessions.iter().enumerate() {
                let output = session.download_bf16(&ranks[rank].outputs[call], rows * 4096)?;
                if let Some(index) = output.iter().zip(&first).position(|(a, b)| a != b) {
                    return Err(format!(
                        "{label} {phase} call {call} ({rows} rows) rank {rank}: output differs from rank 0 at {index}"
                    ));
                }
                // Another order of the sum of squares can move a normalized
                // value one BF16 step, two after a weight above one.
                if let Some(index) = output
                    .iter()
                    .zip(&expected)
                    .position(|(&a, &b)| (i32::from(a) - i32::from(b)).abs() > 2)
                {
                    return Err(format!(
                        "{label} {phase} call {call} ({rows} rows) rank {rank}: output {} vs oracle {} at {index}",
                        output[index], expected[index]
                    ));
                }
            }
        }
        Ok(())
    };
    let outputs = || {
        ranks
            .iter()
            .enumerate()
            .flat_map(|(rank, state)| state.outputs.iter().map(move |output| (rank, output)))
    };
    // Each group runs without host synchronization on poisoned outputs. The
    // residual is the in-place chain and, like the epochs, changes with
    // every call; neither is poisoned.
    poison_ranks(sessions, outputs())?;
    for (call, &shape) in calls.iter().enumerate() {
        hidden_call(sessions, &modules, &ranks, call, shape, None)?;
    }
    let graphs = RankReplay::new(sessions, |rank, _| {
        for (call, &shape) in calls.iter().enumerate() {
            hidden_call(sessions, &modules, &ranks, call, shape, Some(rank))?;
        }
        Ok(())
    })?;
    synchronize(sessions)?;
    check(1, "eager")?;
    poison_ranks(sessions, outputs())?;
    for _ in 0..REPLAYS {
        graphs.launch(sessions)?;
    }
    synchronize(sessions)?;
    check(1 + REPLAYS, "graph replays")?;
    for (rank, session) in sessions.iter().enumerate() {
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!(label));
        receipt.insert("ranks".into(), serde_json::json!(count));
        receipt.insert("rank".into(), serde_json::json!(rank));
        receipt.insert("rows".into(), serde_json::json!(rows));
        receipt.insert("graph_replays".into(), serde_json::json!(REPLAYS));
        receipt.insert("first_epoch".into(), serde_json::json!(NEAR_WRAP));
        receipt.insert("correct".into(), serde_json::json!(true));
        session.receipt("comm_sync", receipt)?;
    }
    Ok(())
}

/// Two-shot calls whose row count grows and shrinks: a lagging rank still
/// reads its gather region while a peer's next, larger scatter arrives.
#[test]
fn two_shot_row_count_changes_keep_gather_regions() {
    gpu_test_multi(1024 * 1024 * 1024, |sessions| {
        run_hidden_sequence(
            sessions,
            "two_shot_row_count_changes",
            &[17, 32, 17, 32, 24, 32, 17, 17, 32, 20, 31],
        )
    })
    .expect("two-shot row count changes");
}

/// Calls that switch protocol on one state, as MiMo prefill chunks of any
/// size do: a lagging rank's row CTAs still wait on a flag of the previous
/// protocol while its peers already publish the next one's.
#[test]
fn protocol_switches_keep_every_flag_until_read() {
    gpu_test_multi(1024 * 1024 * 1024, |sessions| {
        run_hidden_sequence(
            sessions,
            "protocol_switches",
            &[
                16, 32, 16, 16, 32, 17, 3, 32, 1, 2, 17, 33, 16, 32, 100, 5, 32, 2, 1, 16,
            ],
        )
    })
    .expect("protocol switches");
}

/// Wide calls whose row count grows and shrinks: a lagging rank still reads
/// its gather slots while a peer's next, larger scatter arrives.
#[test]
fn wide_row_count_changes_keep_gather_regions() {
    let calls: Vec<Call> = [33, 256, 64, 255, 100, 128, 33, 256, 129, 200]
        .into_iter()
        .map(|rows| Call { rows, graph: true })
        .collect();
    gpu_test_multi(1024 * 1024 * 1024, |sessions| {
        if kernels_sm120::comm::wide_max_rows(sessions.len() as u32).map_err(|e| e.to_string())?
            == 0
        {
            let ranks = sessions.len();
            kernel_lab::skip(&format!(
                "wide row count changes: no wide path at {ranks} ranks"
            ))?;
            return Ok(());
        }
        run_hidden_sequence_from(sessions, "wide_row_count_changes", &calls, false)
    })
    .expect("wide row count changes");
}

/// Wide graph calls between the push protocols and phased prefill calls on
/// one state, as a MiMo verify step at 64 to 256 rows follows decode steps
/// and prefill chunks: every protocol's flags and inbox regions survive a
/// lagging rank whose peers already run the next call.
#[test]
fn wide_calls_mix_with_every_protocol() {
    let calls = [
        (33, true),
        (1, false),
        (256, true),
        (17, false),
        (100, false),
        (64, true),
        (2, false),
        (128, true),
        (32, false),
        (255, true),
        (2048, false),
        (40, true),
        (5, false),
        (33, true),
        (16, false),
        (256, true),
    ]
    .map(|(rows, graph)| Call { rows, graph });
    gpu_test_multi(1024 * 1024 * 1024, |sessions| {
        run_hidden_sequence_from(sessions, "wide_protocol_switches", &calls, false)
    })
    .expect("wide protocol switches");
}

/// A slow consumer on `TARGET`'s stream: the skew cubin's spin entry.
fn spin(sessions: &[Session], modules: &CollectiveModules) -> LabResult<()> {
    let plan = LaunchPlan {
        grid: [1, 1, 1],
        block: [32, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    launch::launch_collective(
        sessions,
        modules,
        "chop_comm_skew_spin_v1",
        plan,
        Some(TARGET),
        |_| {
            Ok(RankCall {
                args: ArgumentList(Vec::new()),
                bindings: Bindings {
                    buffers: BTreeMap::new(),
                },
            })
        },
    )
}

const GATHER_CALLS: usize = 6;
/// Rank 0 receives every gather.
const TARGET: usize = 0;

/// Queues `GATHER_CALLS` gathers to rank 0 on every rank without host
/// synchronization (`launch_call(call)` launches one on every rank), on
/// poisoned outputs of `GATHER_CALLS + 1` regions.
/// After each, rank 0 runs a slow consumer and copies the gathered bytes
/// into snapshot `call + 1` of its output, so a sender that overwrites
/// before the consumer ran changes a snapshot. `expected(call)` gives the
/// gathered bytes of a call, in elements of `dtype`.
#[allow(clippy::too_many_arguments)]
fn gather_with_slow_consumer(
    sessions: &[Session],
    modules: &CollectiveModules,
    outputs: &[DeviceBuffer],
    region: usize,
    dtype: DType,
    label: &str,
    launch_call: impl Fn(usize) -> LabResult<()>,
    expected: impl Fn(usize) -> Vec<u8>,
) -> LabResult<()> {
    let output = &outputs[TARGET];
    poison_ranks(sessions, outputs.iter().enumerate())?;
    for call in 0..GATHER_CALLS {
        launch_call(call)?;
        spin(sessions, modules)?;
        output
            .copy_within_async(0, (call + 1) * region, region, &sessions[TARGET].stream)
            .map_err(|e| e.to_string())?;
    }
    synchronize(sessions)?;
    // Senders write only the target: its region 0 in full every call (README
    // "Logits gather"; the legacy entry with a target rank writes only to
    // that rank), and the snapshots fill the others.
    let mut actual = Vec::new();
    for (rank, session) in sessions.iter().enumerate() {
        let map = if rank == TARGET {
            WriteMap::of(dtype).byte_prefix(outputs[rank].byte_len())?
        } else {
            WriteMap::of(dtype).none()?
        };
        let bytes = check_written(
            session,
            &outputs[rank],
            &map,
            &format!("{label}: rank {rank} output"),
        )?;
        if rank == TARGET {
            actual = bytes;
        }
    }
    for call in 0..GATHER_CALLS {
        let snapshot = &actual[(call + 1) * region..(call + 2) * region];
        if let Some(index) = snapshot
            .iter()
            .zip(expected(call))
            .position(|(a, b)| *a != b)
        {
            return Err(format!("{label}: call {call} differs at byte {index}"));
        }
    }
    for (rank, session) in sessions.iter().enumerate() {
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!(label));
        receipt.insert("ranks".into(), serde_json::json!(sessions.len()));
        receipt.insert("rank".into(), serde_json::json!(rank));
        receipt.insert("calls".into(), serde_json::json!(GATHER_CALLS));
        receipt.insert("correct".into(), serde_json::json!(true));
        session.receipt("comm_sync", receipt)?;
    }
    Ok(())
}

/// Input bytes of `rank` in gather `call`.
fn gather_input(rank: usize, call: usize, bytes: usize) -> Vec<u8> {
    pattern((call * 16 + rank) as u64 + 300, bytes / 2, 4.0)
        .into_iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

/// `chop_comm_gather_rows_*_v1` from a fresh state (`last_epoch` None) or
/// from the state a completed call `last_epoch` leaves: epoch cells at it,
/// senders' acknowledgments at the one before, the target's data flags at it.
fn run_row_gather_sequence(
    sessions: &[Session],
    dtype: DType,
    last_epoch: Option<u32>,
) -> LabResult<()> {
    const ROWS: u32 = 8;
    const SHARD: usize = 38144;
    let ranks = sessions.len();
    let element = if dtype == DType::Bf16 { 2 } else { 4 };
    let symbol = if dtype == DType::Bf16 {
        "chop_comm_gather_rows_bf16_v1"
    } else {
        "chop_comm_gather_rows_f32_v1"
    };
    let input_bytes = ROWS as usize * SHARD * element;
    let region = input_bytes * ranks;
    let tagged = |epoch: u32| 0x8000_0000 | epoch;
    let flag_words = kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES as usize / 4;
    let modules = CollectiveModules::load(sessions, CUBIN)?;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut flags = Vec::new();
    let mut epochs = Vec::new();
    for (rank, session) in sessions.iter().enumerate() {
        let mut words = vec![0_u32; flag_words];
        if let Some(last) = last_epoch {
            for cta in 0..8 {
                if rank == TARGET {
                    for source in 0..ranks {
                        words[source * 8 + cta] = tagged(last);
                    }
                } else {
                    words[64 + cta] = tagged(last.wrapping_sub(1));
                }
            }
        }
        flags.push(upload(session, &u32_bytes(&words))?);
        epochs.push(upload(session, &u32_bytes(&[last_epoch.unwrap_or(0); 8]))?);
        inputs.push(
            (0..GATHER_CALLS)
                .map(|call| upload(session, &gather_input(rank, call, input_bytes)))
                .collect::<LabResult<Vec<_>>>()?,
        );
        outputs.push(session.allocate((GATHER_CALLS + 1) * region)?);
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    let plan = kernels_sm120::comm::gather_rows_plan();
    let launch_call = |call: usize| -> LabResult<()> {
        launch::launch_collective(sessions, &modules, &skew(symbol), plan, None, |rank| {
            let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
            buffers.insert(100, &inputs[rank][call]);
            buffers.insert(102, &epochs[rank]);
            for peer in 0..ranks {
                buffers.insert(peer as u64 + 1, &outputs[peer]);
                buffers.insert(peer as u64 + 11, &flags[peer]);
            }
            let args =
                kernels_sm120::comm::gather_rows_arguments(kernels_sm120::comm::GatherRowsCall {
                    input: 100,
                    outputs: (1..=ranks as u64).collect(),
                    flags: (11..11 + ranks as u64).collect(),
                    epoch: 102,
                    row_indices: None,
                    rows: ROWS,
                    shard: SHARD as u64,
                    rank: rank as u32,
                    ranks: ranks as u32,
                    target: TARGET as u32,
                })
                .map_err(|e| e.to_string())?;
            Ok(RankCall {
                args,
                bindings: Bindings { buffers },
            })
        })
    };
    let label = format!(
        "row_gather_{}_{}",
        if dtype == DType::Bf16 { "bf16" } else { "f32" },
        if last_epoch.is_some() {
            "near_wrap"
        } else {
            "first_use"
        }
    );
    gather_with_slow_consumer(
        sessions,
        &modules,
        &outputs,
        region,
        dtype,
        &label,
        launch_call,
        |call| {
            let inputs: Vec<_> = (0..ranks)
                .map(|rank| gather_input(rank, call, input_bytes))
                .collect();
            let row = SHARD * element;
            (0..ROWS as usize)
                .flat_map(|r| (0..ranks).map(move |q| (r, q)))
                .flat_map(|(r, q)| inputs[q][r * row..(r + 1) * row].to_vec())
                .collect()
        },
    )
}

#[test]
fn row_gather_waits_for_the_consumer_from_first_use() {
    gpu_test_multi(512 * 1024 * 1024, |sessions| {
        run_row_gather_sequence(sessions, DType::Bf16, None)?;
        run_row_gather_sequence(sessions, DType::F32, None)
    })
    .expect("row gather from first use");
}

/// Calls with epochs 0xfffffffe, 0xffffffff, 0, 1, 2 and 3: senders used to
/// skip the acknowledgment at 0 and 1 and overwrite rows the target's
/// consumer had not read.
#[test]
fn row_gather_waits_for_the_consumer_across_the_epoch_wrap() {
    gpu_test_multi(512 * 1024 * 1024, |sessions| {
        run_row_gather_sequence(sessions, DType::Bf16, Some(u32::MAX - 2))?;
        run_row_gather_sequence(sessions, DType::F32, Some(u32::MAX - 2))
    })
    .expect("row gather across the epoch wrap");
}

/// The legacy `chop_comm_allgather_bf16_v1` to rank 0, fresh or from the
/// state a completed call `last_epoch` leaves.
fn run_legacy_allgather_sequence(sessions: &[Session], last_epoch: Option<u32>) -> LabResult<()> {
    const ELEMENTS: usize = 2 * 4096 + 100;
    let ranks = sessions.len();
    let tiles = ELEMENTS.div_ceil(4096);
    let input_bytes = ELEMENTS * 2;
    let region = input_bytes * ranks;
    let query = CommQuery::AllGather {
        ranks: ranks as u32,
        elements_per_rank: ELEMENTS as u64,
        dtype: DType::Bf16,
    };
    let plan = kernels_sm120::comm::plan(&query, 1).map_err(|e| e.to_string())?;
    let modules = CollectiveModules::load(sessions, CUBIN)?;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut flags = Vec::new();
    let mut epochs = Vec::new();
    for (rank, session) in sessions.iter().enumerate() {
        // Data words [source][tile] hold the last call, acknowledgments
        // [ranks + source][tile] the one before.
        let mut words = vec![0_u32; 2 * ranks * tiles];
        if let Some(last) = last_epoch {
            words[..ranks * tiles].fill(last);
            words[ranks * tiles..].fill(last.wrapping_sub(1));
        }
        flags.push(upload(session, &u32_bytes(&words))?);
        epochs.push(upload(
            session,
            &u32_bytes(&vec![last_epoch.unwrap_or(0); tiles]),
        )?);
        inputs.push(
            (0..GATHER_CALLS)
                .map(|call| upload(session, &gather_input(rank, call, input_bytes)))
                .collect::<LabResult<Vec<_>>>()?,
        );
        outputs.push(session.allocate((GATHER_CALLS + 1) * region)?);
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    let launch_call = |call: usize| -> LabResult<()> {
        let symbol = skew("chop_comm_allgather_bf16_v1");
        launch::launch_collective(sessions, &modules, &symbol, plan, None, |rank| {
            let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
            buffers.insert(100, &inputs[rank][call]);
            buffers.insert(102, &epochs[rank]);
            for peer in 0..ranks {
                buffers.insert(peer as u64 + 1, &outputs[peer]);
                buffers.insert(peer as u64 + 11, &flags[peer]);
            }
            let args = ArgumentList(vec![
                Argument::DeviceBuffer(100),
                Argument::DevicePointerArray((1..=ranks as u64).collect()),
                Argument::DevicePointerArray((11..11 + ranks as u64).collect()),
                Argument::DeviceBuffer(102),
                Argument::U64(ELEMENTS as u64),
                Argument::U32(rank as u32),
                Argument::U32(ranks as u32),
                Argument::U32(TARGET as u32),
            ]);
            Ok(RankCall {
                args,
                bindings: Bindings { buffers },
            })
        })
    };
    let label = format!(
        "legacy_allgather_{}",
        if last_epoch.is_some() {
            "near_wrap"
        } else {
            "first_use"
        }
    );
    gather_with_slow_consumer(
        sessions,
        &modules,
        &outputs,
        region,
        DType::Bf16,
        &label,
        launch_call,
        |call| {
            (0..ranks)
                .flat_map(|rank| gather_input(rank, call, input_bytes))
                .collect()
        },
    )
}

/// The target used to skip its acknowledgment when the epoch wrapped to
/// zero, and every sender waited for 0xffffffff forever.
#[test]
fn legacy_allgather_acknowledges_across_the_epoch_wrap() {
    gpu_test_multi(256 * 1024 * 1024, |sessions| {
        run_legacy_allgather_sequence(sessions, None)?;
        run_legacy_allgather_sequence(sessions, Some(u32::MAX - 2))
    })
    .expect("legacy all-gather across the epoch wrap");
}

/// Low-latency calls read entries whose bytes carry an upcoming call's flag.
/// Without the zeroed region `PushBuffers` requires, a receiver whose
/// partner lags takes the stale payload for the partner's data; with it,
/// every call matches the oracle.
#[test]
fn low_latency_needs_its_zeroed_region() {
    const ROWS: [usize; 6] = [2, 2, 1, 2, 1, 1];
    gpu_test_multi(1024 * 1024 * 1024, |sessions| {
        if !sessions.len().is_power_of_two() {
            let ranks = sessions.len();
            kernel_lab::skip(&format!(
                "low-latency zeroed region: {ranks} ranks are not a power of two"
            ))?;
            return Ok(());
        }
        match run_hidden_sequence_from(sessions, "low_latency_stale_inbox", &prefill(&ROWS), true) {
            Err(error) if error.contains("differs") || error.contains("oracle") => {}
            Err(error) => return Err(error),
            Ok(()) => return Err("stale LL entries were not taken for data".into()),
        }
        run_hidden_sequence(sessions, "low_latency_zeroed_region", &ROWS)
    })
    .expect("low-latency inbox initialization");
}
