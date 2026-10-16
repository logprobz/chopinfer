//! The wide push all-reduce (`chop_comm_allreduce_wide_*`, 33 to 256 rows in
//! decode and verify graphs) against the host reference, against the push
//! path bit for bit, and with the MoE combine folded in.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{CommQuery, DType, LaunchPlan, RmsNormEpilogue};
use kernel_core::oracle;
use kernel_lab::launch::{self, Bindings, CollectiveModules, RankCall};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, gpu_test_multi};
use runtime_cuda::DeviceBuffer;
use std::collections::BTreeMap;

use kernel_core::oracle::{NormArithmetic, NormOffset, NormScale, RmsNorm};
use kernels_sm120::comm::{
    FusedOneShotCall, FusedWithResidualOutputCall, OneShotCall, Variant, WIDE_COMBINE_SYMBOL,
    WIDE_SYMBOL,
};

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

/// The row counts the package names: odd, even, the smallest and the largest.
const ROWS: [usize; 6] = [33, 64, 100, 128, 255, 256];

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

/// Partial sums whose exponents span 16 binades, so the rank-ordered F32
/// sums round and another summation order would change their bits.
fn partials(seed: u64, len: usize, scale: f32) -> Vec<u16> {
    pattern(seed, len, scale)
        .into_iter()
        .enumerate()
        .map(|(index, bits)| {
            let binade = ((index as u64 * 5 + seed * 3) % 16) as i32 - 8;
            f32_to_bf16(bf16_to_f32(bits) * f32::powi(2.0, binade))
        })
        .collect()
}

fn weights() -> Vec<u16> {
    pattern(7, 4096, 1.0)
        .into_iter()
        .map(|bits| f32_to_bf16(1.0 + 0.5 * bf16_to_f32(bits)))
        .collect()
}

fn fused_query(ranks: usize, rows: usize) -> CommQuery {
    CommQuery::AllReduce {
        ranks: ranks as u32,
        elements: (rows * 4096) as u64,
        dtype: DType::Bf16,
        fused_residual: true,
        fused_rmsnorm: Some(RmsNormEpilogue {
            kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
            width: 4096,
            eps: 1e-6,
        }),
    }
}

/// Row counts from `CHOPINFER_KERNEL_COMM_TOKENS` (comma separated) or the
/// package's list.
fn row_list() -> Vec<usize> {
    std::env::var("CHOPINFER_KERNEL_COMM_TOKENS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|rows| rows.trim().parse().expect("integer row count"))
                .collect()
        })
        .unwrap_or_else(|| ROWS.to_vec())
}

fn zeroed(session: &Session, bytes: usize) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.max(4))?;
    buffer
        .copy_from_bytes(&vec![0; bytes.max(4)], &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

/// One rank's comm state for fused hidden calls of up to `rows` rows in a
/// graph: the planner's sizes, the inbox beyond the low-latency region
/// filled with stale bytes.
struct State {
    inbox: DeviceBuffer,
    flags: DeviceBuffer,
    epoch: DeviceBuffer,
}

fn states(sessions: &[Session], capacity: usize) -> LabResult<Vec<State>> {
    let ranks = sessions.len() as u32;
    let sizes = kernels_sm120::comm::fused_state_buffers(ranks, capacity as u32)
        .map_err(|e| e.to_string())?;
    sessions
        .iter()
        .map(|session| {
            let inbox = session.allocate(sizes.inbox_bytes as usize)?;
            inbox
                .copy_from_bytes(&vec![0x3c; sizes.inbox_bytes as usize], &session.stream)
                .map_err(|e| e.to_string())?;
            inbox
                .zero_at_async(0, sizes.zeroed_inbox_bytes as usize, &session.stream)
                .map_err(|e| e.to_string())?;
            let state = State {
                inbox,
                flags: zeroed(session, sizes.flag_bytes as usize)?,
                epoch: zeroed(session, sizes.epoch_bytes as usize)?,
            };
            session.stream.synchronize().map_err(|e| e.to_string())?;
            Ok(state)
        })
        .collect()
}

/// Handles: 100 input, 101 output, 102 epoch, 103 residual (and residual
/// output), 104 weights, 105 live rows, peer inboxes 1.., peer flags 11...
fn bindings<'a>(
    states: &'a [State],
    rank: usize,
    input: &'a DeviceBuffer,
    output: &'a DeviceBuffer,
    residual: &'a DeviceBuffer,
    weights: &'a DeviceBuffer,
    live: Option<&'a DeviceBuffer>,
) -> Bindings<'a> {
    let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
    buffers.insert(100, input);
    buffers.insert(101, output);
    buffers.insert(102, &states[rank].epoch);
    buffers.insert(103, residual);
    buffers.insert(104, weights);
    if let Some(live) = live {
        buffers.insert(105, live);
    }
    for (peer, state) in states.iter().enumerate() {
        buffers.insert(peer as u64 + 1, &state.inbox);
        buffers.insert(peer as u64 + 11, &state.flags);
    }
    Bindings { buffers }
}

/// The v2 (or combine, with `live`) arguments on the handles of `bindings`.
fn arguments(
    ranks: usize,
    rank: usize,
    rows: usize,
    live: bool,
) -> LabResult<kernel_core::contract::ArgumentList> {
    let handles: Vec<u64> = (0..ranks as u64).map(|rank| rank + 1).collect();
    let call = FusedWithResidualOutputCall {
        fused: FusedOneShotCall {
            base: OneShotCall {
                input: 100,
                output: 101,
                peer_inboxes: handles.clone(),
                peer_flags: handles.iter().map(|value| value + 10).collect(),
                epoch: 102,
                elements: (rows * 4096) as u64,
                rank: rank as u32,
                ranks: ranks as u32,
            },
            residual: Some(103),
            weights: 104,
            epsilon: 1e-6,
        },
        residual_output: 103,
    };
    if live {
        kernels_sm120::comm::fused_combine_arguments(call, Some(105))
    } else {
        kernels_sm120::comm::fused_arguments_v2(call)
    }
    .map_err(|e| e.to_string())
}

/// The graph planner's variant and plan for `rows`; the wide entry above 32.
fn graph_launch(ranks: usize, rows: usize, combine: bool) -> LabResult<(&'static str, LaunchPlan)> {
    let query = fused_query(ranks, rows);
    let plan = kernels_sm120::comm::graph_plan(&query, rows as u32).map_err(|e| e.to_string())?;
    let symbol = match kernels_sm120::comm::graph_choose(&query).map_err(|e| e.to_string())? {
        Variant::FusedWide if combine => WIDE_COMBINE_SYMBOL,
        Variant::FusedWide => WIDE_SYMBOL,
        Variant::FusedOneShot if combine => "chop_comm_allreduce_combine_norm_bf16_v1",
        Variant::FusedOneShot => "chop_comm_allreduce_oneshot_norm_bf16_v2",
        other => return Err(format!("{rows} rows: unexpected graph variant {other:?}")),
    };
    Ok((symbol, plan))
}

/// The comm cubin loaded for collective calls, and `symbol`, the entry to
/// call.
fn collective(sessions: &[Session], symbol: &str) -> LabResult<(CollectiveModules, String)> {
    Ok((
        CollectiveModules::load(sessions, "comm_sm120")?,
        symbol.to_owned(),
    ))
}

fn synchronize(sessions: &[Session]) -> LabResult<()> {
    for session in sessions {
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Poisons each rank's output once every rank is idle.
fn poison_outputs(sessions: &[Session], outputs: &[DeviceBuffer]) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (session, output) in sessions.iter().zip(outputs) {
        poison::poison(session, &[output])?;
    }
    Ok(())
}

/// Every element of `map` written, every other byte of `buffer` still
/// poison.
fn check_written(
    session: &Session,
    buffer: &DeviceBuffer,
    map: &WriteMap,
    label: &str,
) -> LabResult<()> {
    let bytes = poison::read(session, buffer)?;
    poison::check(label, &bytes, map, Fill::Pattern(&[POISON]))
}

fn wide_ranks(sessions: &[Session]) -> LabResult<bool> {
    Ok(
        kernels_sm120::comm::wide_max_rows(sessions.len() as u32).map_err(|e| e.to_string())?
            >= 256,
    )
}

/// Eager calls and graph replays of the wide entry, each adding rank-ordered
/// sums of one of three inputs to the residual in place: the residual chain
/// matches the host exactly, normalized rows stay within two BF16 steps of
/// the HF RMSNorm oracle and match across ranks, and the epoch counts calls.
/// `workers` gives each rank's worker CTAs; empty takes the planner's.
fn run_reference(sessions: &[Session], rows: usize, workers: &[u32]) -> LabResult<()> {
    const DISTINCT: usize = 3;
    const EAGER: usize = 2;
    const CAPTURED: usize = 4;
    const REPLAYS: usize = 2;
    let ranks = sessions.len();
    let elements = rows * 4096;
    let (symbol, plan) = graph_launch(ranks, rows, false)?;
    if symbol != WIDE_SYMBOL {
        return Err(format!("{rows} rows do not take the wide entry"));
    }
    let host_inputs = (0..ranks)
        .map(|rank| {
            (0..DISTINCT)
                .map(|set| partials((rank * DISTINCT + set) as u64, elements, 0.05))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let host_residual = pattern(99, elements, 1.0);
    let host_weights = weights();
    let states = states(sessions, rows)?;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut residuals = Vec::new();
    let mut scales = Vec::new();
    for (rank, session) in sessions.iter().enumerate() {
        inputs.push(
            host_inputs[rank]
                .iter()
                .map(|values| session.upload_bf16(values))
                .collect::<LabResult<Vec<_>>>()?,
        );
        outputs.push(session.allocate(elements * 2)?);
        residuals.push(session.upload_bf16(&host_residual)?);
        scales.push(session.upload_bf16(&host_weights)?);
    }
    synchronize(sessions)?;
    let (modules, entry) = collective(sessions, symbol)?;
    // Rank `rank`'s call of `set` with its worker count.
    let issue = |rank: usize, set: usize| -> LabResult<()> {
        let mut plan = plan;
        if let Some(&workers) = workers.get(rank) {
            plan.grid[0] = plan.cluster[0] + workers.next_multiple_of(plan.cluster[0]);
        }
        // Keep the worker count within the target layout.
        kernels_sm120::comm::wide_fits_target_layout(&plan).map_err(|e| e.to_string())?;
        launch::launch_collective(sessions, &modules, &entry, plan, Some(rank), |rank| {
            Ok(RankCall {
                args: arguments(ranks, rank, rows, false)?,
                bindings: bindings(
                    &states,
                    rank,
                    &inputs[rank][set],
                    &outputs[rank],
                    &residuals[rank],
                    &scales[rank],
                    None,
                ),
            })
        })
    };
    // The wide entry writes all rows of the output (README "Wide all-reduce
    // for graphs above 32 rows": workers normalize every row); the residual
    // is the in-place chain, never poisoned.
    let written = WriteMap::of(DType::Bf16).prefix(elements)?;
    // Checks every rank after the calls of `sets`: outputs written, equal to
    // rank 0's and within two BF16 steps of the oracle, the residual chain
    // exact and the epoch at the call count. `receipts` records the result.
    let check = |phase: &str, sets: &[usize], receipts: bool| -> LabResult<()> {
        let mut expected_residual = host_residual.clone();
        for &set in sets {
            for (index, value) in expected_residual.iter_mut().enumerate() {
                let sum = (0..ranks).fold(0.0_f32, |sum, rank| {
                    sum + bf16_to_f32(host_inputs[rank][set][index])
                });
                *value = f32_to_bf16(sum + bf16_to_f32(*value));
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
                &format!("{rows} rows rank {rank} {phase} output"),
            )?;
        }
        let first = sessions[0].download_bf16(&outputs[0], elements)?;
        for (rank, session) in sessions.iter().enumerate() {
            let actual = session.download_bf16(&outputs[rank], elements)?;
            let residual = session.download_bf16(&residuals[rank], elements)?;
            let mut epoch = [0_u8; 4];
            states[rank]
                .epoch
                .copy_to_bytes(&mut epoch, &session.stream)
                .map_err(|e| e.to_string())?;
            let epoch = u32::from_ne_bytes(epoch);
            let residual_gate = oracle::check_exact(&residual, &expected_residual);
            let cross_rank = actual.iter().zip(&first).position(|(a, b)| a != b);
            // Another order of the sum of squares can move a normalized value
            // one BF16 step, two after a weight above one (as in comm_sync).
            let far = actual
                .iter()
                .zip(&expected)
                .position(|(&a, &b)| (i32::from(a) - i32::from(b)).abs() > 2);
            if receipts {
                let mut receipt = BTreeMap::new();
                receipt.insert(
                    "case".into(),
                    serde_json::json!("bf16_wide_allreduce_reference"),
                );
                receipt.insert("rows".into(), serde_json::json!(rows));
                receipt.insert("ranks".into(), serde_json::json!(ranks));
                receipt.insert("rank".into(), serde_json::json!(rank));
                receipt.insert("grid".into(), serde_json::json!(plan.grid[0]));
                receipt.insert("calls".into(), serde_json::json!(sets.len()));
                receipt.insert("graph_replays".into(), serde_json::json!(REPLAYS));
                receipt.insert(
                    "residual_correct".into(),
                    serde_json::json!(residual_gate.is_ok()),
                );
                receipt.insert(
                    "bitwise_equal_to_rank0".into(),
                    serde_json::json!(cross_rank.is_none()),
                );
                receipt.insert(
                    "output_within_two_bf16_steps".into(),
                    serde_json::json!(far.is_none()),
                );
                receipt.insert("device_epoch".into(), serde_json::json!(epoch));
                session.receipt("comm_wide", receipt)?;
            }
            residual_gate.map_err(|index| {
                format!("{rows} rows rank {rank} {phase}: residual differs at {index}")
            })?;
            if let Some(index) = cross_rank {
                return Err(format!(
                    "{rows} rows rank {rank} {phase}: output differs from rank 0 at {index}"
                ));
            }
            if let Some(index) = far {
                return Err(format!(
                    "{rows} rows rank {rank} {phase}: output {} vs oracle {} at {index}",
                    actual[index], expected[index]
                ));
            }
            if epoch as usize != sets.len() {
                return Err(format!(
                    "{rows} rows rank {rank} {phase}: device epoch {epoch}"
                ));
            }
        }
        Ok(())
    };
    // The eager calls, then the graph replays, each group queued without
    // host synchronization on poisoned outputs.
    let mut sets = Vec::new();
    poison_outputs(sessions, &outputs)?;
    for call in 0..EAGER {
        kernel_lab::topology::each_rank(ranks, |rank| issue(rank, call % DISTINCT))?;
        sets.push(call % DISTINCT);
    }
    let graphs = RankReplay::new(sessions, |rank, _| {
        (0..CAPTURED).try_for_each(|call| issue(rank, (EAGER + call) % DISTINCT))
    })?;
    synchronize(sessions)?;
    check("eager", &sets, false)?;
    poison_outputs(sessions, &outputs)?;
    for _ in 0..REPLAYS {
        graphs.launch(sessions)?;
        sets.extend((0..CAPTURED).map(|call| (EAGER + call) % DISTINCT));
    }
    synchronize(sessions)?;
    check("graph replay", &sets, true)
}

/// Chunks of at most 32 rows that avoid the low-latency protocol, whose
/// two-CTA norm sums squares in another order: every chunk takes the
/// one-shot or the two-shot. The two-shot normalizes a row as the wide
/// workers do; the one-shot sums the squares in another order.
fn push_chunks(rows: usize) -> Vec<std::ops::Range<usize>> {
    let mut chunks = Vec::new();
    let mut first = 0;
    while first < rows {
        let left = rows - first;
        let take = if left > 34 {
            32
        } else if left > 32 {
            left - 3
        } else {
            left
        };
        chunks.push(first..first + take);
        first += take;
    }
    chunks
}

/// One wide call against the v2 push entry on chunks of the same rows, each
/// on its own buffers and a separate comm state, with and without the MoE
/// combine folded in: pre-norm rows must match bit for bit, normalized rows
/// bit for bit against two-shot chunks and within one BF16 step against
/// one-shot chunks.
fn run_against_push(sessions: &[Session], rows: usize, live: Option<usize>) -> LabResult<()> {
    let ranks = sessions.len();
    let combine = live.is_some();
    let depth = if combine { 8 } else { 1 };
    let host_inputs = (0..ranks)
        .map(|rank| partials(rank as u64 + 40, rows * depth * 4096, 0.05))
        .collect::<Vec<_>>();
    let host_residual = pattern(41, rows * 4096, 1.0);
    let host_weights = weights();
    let run = |ranges: &[std::ops::Range<usize>]| -> LabResult<Vec<(Vec<u16>, Vec<u16>)>> {
        let states = states(sessions, rows)?;
        let mut per_rank = vec![(Vec::new(), Vec::new()); ranks];
        for range in ranges {
            let count = range.len();
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            let mut residuals = Vec::new();
            let mut scales = Vec::new();
            let mut lives = Vec::new();
            for (rank, session) in sessions.iter().enumerate() {
                inputs.push(session.upload_bf16(
                    &host_inputs[rank][range.start * depth * 4096..range.end * depth * 4096],
                )?);
                outputs.push(session.allocate(count * 4096 * 2)?);
                residuals.push(
                    session.upload_bf16(&host_residual[range.start * 4096..range.end * 4096])?,
                );
                scales.push(session.upload_bf16(&host_weights)?);
                let chunk_live = live.map_or(count, |live| {
                    live.clamp(range.start, range.end) - range.start
                });
                let buffer = session.allocate(4)?;
                buffer
                    .copy_from_bytes(&(chunk_live as u32).to_ne_bytes(), &session.stream)
                    .map_err(|e| e.to_string())?;
                lives.push(buffer);
            }
            synchronize(sessions)?;
            let (symbol, plan) = graph_launch(ranks, count, combine)?;
            let (modules, entry) = collective(sessions, symbol)?;
            // Every entry writes all `count` rows of the output, the combine
            // entries' rows at or past live with a zero sum (README "rows at
            // or past `*live_rows` contribute zero"; push.cuh
            // `input_vector`).
            let written = WriteMap::of(DType::Bf16).prefix(count * 4096)?;
            poison_outputs(sessions, &outputs)?;
            launch::launch_collective(sessions, &modules, &entry, plan, None, |rank| {
                Ok(RankCall {
                    args: arguments(ranks, rank, count, combine)?,
                    bindings: bindings(
                        &states,
                        rank,
                        &inputs[rank],
                        &outputs[rank],
                        &residuals[rank],
                        &scales[rank],
                        Some(&lives[rank]),
                    ),
                })
            })?;
            synchronize(sessions)?;
            for (rank, session) in sessions.iter().enumerate() {
                check_written(
                    session,
                    &outputs[rank],
                    &written,
                    &format!(
                        "{symbol} rows {}..{} rank {rank} output",
                        range.start, range.end
                    ),
                )?;
            }
            for (rank, session) in sessions.iter().enumerate() {
                per_rank[rank]
                    .0
                    .extend(session.download_bf16(&outputs[rank], count * 4096)?);
                per_rank[rank]
                    .1
                    .extend(session.download_bf16(&residuals[rank], count * 4096)?);
            }
        }
        Ok(per_rank)
    };
    let wide = run(std::slice::from_ref(&(0..rows)))?;
    let push = run(&push_chunks(rows))?;
    let label = if combine { "combine" } else { "plain" };
    // Rows of chunks that take the two-shot, whose norm is the wide one's.
    let two_shot: Vec<bool> = push_chunks(rows)
        .into_iter()
        .flat_map(|range| {
            let protocol = kernels_sm120::comm::push_protocol(ranks as u32, range.len() as u32);
            let two_shot = matches!(protocol, Ok(kernels_sm120::comm::PushProtocol::TwoShot));
            std::iter::repeat_n(two_shot, range.len())
        })
        .collect();
    for rank in 0..ranks {
        let mut one_step = 0_usize;
        let output =
            wide[rank]
                .0
                .iter()
                .zip(&push[rank].0)
                .enumerate()
                .position(|(index, (&a, &b))| {
                    let step = (i32::from(a) - i32::from(b)).abs();
                    one_step += usize::from(step == 1);
                    step > 1 || (step == 1 && two_shot[index / 4096])
                });
        let residual = wide[rank]
            .1
            .iter()
            .zip(&push[rank].1)
            .position(|(a, b)| a != b);
        let cross_rank = wide[rank] != wide[0];
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "case".into(),
            serde_json::json!("bf16_wide_allreduce_against_push"),
        );
        receipt.insert("variant".into(), serde_json::json!(label));
        receipt.insert("rows".into(), serde_json::json!(rows));
        receipt.insert("live_rows".into(), serde_json::json!(live.unwrap_or(rows)));
        receipt.insert("ranks".into(), serde_json::json!(ranks));
        receipt.insert("rank".into(), serde_json::json!(rank));
        receipt.insert(
            "push_chunks".into(),
            serde_json::json!(
                push_chunks(rows)
                    .iter()
                    .map(|range| range.len())
                    .collect::<Vec<_>>()
            ),
        );
        receipt.insert(
            "residual_bitwise_equal_to_push".into(),
            serde_json::json!(residual.is_none()),
        );
        receipt.insert(
            "output_matches_push".into(),
            serde_json::json!(output.is_none()),
        );
        receipt.insert(
            "one_shot_outputs_one_step_apart".into(),
            serde_json::json!(one_step),
        );
        receipt.insert(
            "bitwise_equal_to_rank0".into(),
            serde_json::json!(!cross_rank),
        );
        sessions[rank].receipt("comm_wide", receipt)?;
        if let Some(index) = output {
            return Err(format!(
                "{label} {rows} rows rank {rank}: output differs from push at {index}"
            ));
        }
        if let Some(index) = residual {
            return Err(format!(
                "{label} {rows} rows rank {rank}: residual differs from push at {index}"
            ));
        }
        if cross_rank {
            return Err(format!(
                "{label} {rows} rows rank {rank}: differs from rank 0"
            ));
        }
    }
    Ok(())
}

/// The wide combine entry on preweighted routed rows `[rows][8][4096]`
/// against the wide entry on host-combined rows (zero at and past the live
/// count), after eager calls and graph replays: every output bit matches.
fn run_combine(sessions: &[Session], rows: usize, live: usize) -> LabResult<()> {
    const CALLS: usize = 4;
    let ranks = sessions.len();
    let elements = rows * 4096;
    let routed = (0..ranks)
        .map(|rank| pattern(rank as u64 + 70, elements * 8, 0.02))
        .collect::<Vec<_>>();
    let combined = routed
        .iter()
        .map(|values| {
            (0..elements)
                .map(|index| {
                    let (row, column) = (index / 4096, index % 4096);
                    if row >= live {
                        return 0;
                    }
                    let sum = (0..8).fold(0.0_f32, |sum, k| {
                        sum + bf16_to_f32(values[(row * 8 + k) * 4096 + column])
                    });
                    f32_to_bf16(sum)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let residual = pattern(71, elements, 1.0);
    let host_weights = weights();
    // Both entries write all rows of the output, the combine entry's rows at
    // or past live with a zero sum (README "rows at or past `*live_rows`
    // contribute zero"; push.cuh `input_vector`), so both maps hold all rows.
    let written = WriteMap::of(DType::Bf16).prefix(elements)?;
    let mut finals = Vec::new();
    for fused in [false, true] {
        let (symbol, plan) = graph_launch(ranks, rows, fused)?;
        let states = states(sessions, rows)?;
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut residuals = Vec::new();
        let mut scales = Vec::new();
        let mut lives = Vec::new();
        for (rank, session) in sessions.iter().enumerate() {
            inputs.push(session.upload_bf16(if fused {
                &routed[rank]
            } else {
                &combined[rank]
            })?);
            outputs.push(session.allocate(elements * 2)?);
            residuals.push(session.upload_bf16(&residual)?);
            scales.push(session.upload_bf16(&host_weights)?);
            let count = session.allocate(4)?;
            count
                .copy_from_bytes(&(live as u32).to_ne_bytes(), &session.stream)
                .map_err(|e| e.to_string())?;
            lives.push(count);
        }
        synchronize(sessions)?;
        let (modules, entry) = collective(sessions, symbol)?;
        let issue = |only_rank: Option<usize>| -> LabResult<()> {
            launch::launch_collective(sessions, &modules, &entry, plan, only_rank, |rank| {
                Ok(RankCall {
                    args: arguments(ranks, rank, rows, fused)?,
                    bindings: bindings(
                        &states,
                        rank,
                        &inputs[rank],
                        &outputs[rank],
                        &residuals[rank],
                        &scales[rank],
                        Some(&lives[rank]),
                    ),
                })
            })
        };
        // Every rank's output written and its output and residual equal to
        // rank 0's; rank 0's pair.
        let state = |phase: &str| -> LabResult<(Vec<u16>, Vec<u16>)> {
            let mut state = Vec::new();
            for (rank, session) in sessions.iter().enumerate() {
                check_written(
                    session,
                    &outputs[rank],
                    &written,
                    &format!("combine {fused} {rows} rows live {live} rank {rank} {phase}"),
                )?;
                state.push((
                    session.download_bf16(&outputs[rank], elements)?,
                    session.download_bf16(&residuals[rank], elements)?,
                ));
            }
            if state.iter().any(|item| item != &state[0]) {
                return Err(format!("combine {fused} {rows} rows {phase}: ranks differ"));
            }
            Ok(state.swap_remove(0))
        };
        // The residual is the in-place chain of both groups, never poisoned.
        poison_outputs(sessions, &outputs)?;
        for _ in 0..CALLS / 2 {
            issue(None)?;
        }
        let graphs = RankReplay::new(sessions, |rank, _| {
            (0..CALLS / 2).try_for_each(|_| issue(Some(rank)))
        })?;
        synchronize(sessions)?;
        let eager = state("eager")?;
        poison_outputs(sessions, &outputs)?;
        graphs.launch(sessions)?;
        synchronize(sessions)?;
        finals.push([eager, state("graph replay")?]);
    }
    let equal = finals[0] == finals[1];
    for (rank, session) in sessions.iter().enumerate() {
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "case".into(),
            serde_json::json!("bf16_wide_combine_allreduce"),
        );
        receipt.insert("rows".into(), serde_json::json!(rows));
        receipt.insert("live_rows".into(), serde_json::json!(live));
        receipt.insert("ranks".into(), serde_json::json!(ranks));
        receipt.insert("rank".into(), serde_json::json!(rank));
        receipt.insert(
            "bitwise_equal_to_combine_then_wide".into(),
            serde_json::json!(equal),
        );
        session.receipt("comm_wide", receipt)?;
    }
    if !equal {
        return Err(format!(
            "{rows} rows live {live}: wide combine differs from combine + wide"
        ));
    }
    Ok(())
}

/// Every spinning comm grid fits resident on one device at its largest
/// plan. Wide entries also reach the tuning model's 36 resident clusters.
#[test]
fn spinning_comm_entries_stay_resident_on_distinct_devices() {
    gpu_test_multi(64 * 1024 * 1024, |sessions| {
        let session = &sessions[0];
        let qwen_wide = model_qwen38_contract::sm120::comm::decode::wide_plan(256)
            .map_err(|e| e.to_string())?
            .grid[0];
        let qwen_query = model_qwen38_contract::sm120::comm::query(4);
        let kernel_core::contract::SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm { capacity, .. } =
            qwen_query
        else {
            return Err("Qwen3.8 all-reduce query".into());
        };
        let qwen_phased = model_qwen38_contract::sm120::comm::prefill_plan(qwen_query, capacity)
            .map_err(|e| e.to_string())?
            .grid[0];
        let wide = graph_launch(4, 256, false)?.1.grid[0];
        let phased = kernels_sm120::comm::plan(&fused_query(4, 8192), 8192)
            .map_err(|e| e.to_string())?
            .grid[0];
        let (push_symbol, _) = graph_launch(4, 32, false)?;
        let push = graph_launch(4, 32, false)?.1.grid[0];
        let entries: Vec<(&str, String, u32)> = vec![
            ("comm_sm120", WIDE_SYMBOL.into(), wide),
            ("comm_sm120", WIDE_COMBINE_SYMBOL.into(), wide),
            (
                "comm_sm120",
                kernels_sm120::comm::PREFILL_ROUND_ONCE_SYMBOL.into(),
                phased,
            ),
            (
                "comm_sm120",
                kernels_sm120::comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL.into(),
                phased,
            ),
            ("comm_sm120", push_symbol.into(), push),
            (
                "comm_qwen38",
                model_qwen38_contract::sm120::comm::SCATTER_SYMBOL.into(),
                qwen_phased,
            ),
            (
                "comm_qwen38",
                model_qwen38_contract::sm120::comm::GATHER_SYMBOL.into(),
                qwen_phased,
            ),
        ]
        .into_iter()
        .chain(
            model_qwen38_contract::sm120::comm::decode::WIDE_SYMBOLS
                .iter()
                .map(|&symbol| ("comm_qwen38_decode_wide", symbol.to_owned(), qwen_wide)),
        )
        .collect();
        let sms = session.context.device().multiprocessor_count();
        for (cubin, symbol, grid) in entries {
            let function = session
                .module(cubin)?
                .function(&symbol)
                .map_err(|e| e.to_string())?;
            let resident = function
                .max_active_clusters(
                    runtime_cuda::LaunchConfig {
                        grid: (grid, 1, 1),
                        block: (256, 1, 1),
                        shared_memory_bytes: 0,
                        cluster: (8, 1, 1),
                        programmatic_stream_serialization: false,
                    },
                    false,
                )
                .map_err(|e| e.to_string())?;
            let clusters = grid / 8;
            eprintln!(
                "residency {symbol}: {grid} CTAs ({clusters} clusters), {resident} clusters \
                 resident on {sms} SMs"
            );
            if resident < clusters {
                return Err(format!(
                    "{symbol}: {resident} resident clusters hold fewer than the grid's {clusters} clusters"
                ));
            }
            // The wide entries' two CTAs per SM, as the tuning model counts.
            if symbol.contains("_wide_") && resident < 36 {
                return Err(format!(
                    "{symbol}: {resident} resident clusters, below the tuning model's 36"
                ));
            }
        }
        Ok(())
    })
    .expect("spinning comm entries resident on distinct devices");
}

#[test]
fn bf16_wide_allreduce_matches_the_reference() {
    gpu_test_multi(512 * 1024 * 1024, |sessions| {
        if !wide_ranks(sessions)? {
            let ranks = sessions.len();
            kernel_lab::skip(&format!("wide comm test: no wide path at {ranks} ranks"))?;
            return Ok(());
        }
        row_list()
            .into_iter()
            .try_for_each(|rows| run_reference(sessions, rows, &[]))
    })
    .expect("wide all-reduce against the reference");
}

/// The worker count is a launch knob: ranks agree on the protocol whatever
/// their grids, so rank 0 with 8 workers and the others with 64 give the
/// reference's results.
#[test]
fn bf16_wide_allreduce_takes_any_worker_count_per_rank() {
    gpu_test_multi(512 * 1024 * 1024, |sessions| {
        if !wide_ranks(sessions)? {
            let ranks = sessions.len();
            kernel_lab::skip(&format!("wide comm test: no wide path at {ranks} ranks"))?;
            return Ok(());
        }
        let workers: Vec<u32> = (0..sessions.len())
            .map(|rank| if rank == 0 { 8 } else { 64 })
            .collect();
        [33, 128, 256]
            .into_iter()
            .try_for_each(|rows| run_reference(sessions, rows, &workers))
    })
    .expect("wide all-reduce with per-rank worker counts");
}

#[test]
fn bf16_wide_allreduce_matches_the_push_path() {
    gpu_test_multi(512 * 1024 * 1024, |sessions| {
        if !wide_ranks(sessions)? {
            let ranks = sessions.len();
            kernel_lab::skip(&format!("wide comm test: no wide path at {ranks} ranks"))?;
            return Ok(());
        }
        for rows in row_list() {
            run_against_push(sessions, rows, None)?;
        }
        // The combine fold on both paths, with padded rows.
        for (rows, live) in [(33, 20), (100, 100), (256, 129)] {
            run_against_push(sessions, rows, Some(live))?;
        }
        Ok(())
    })
    .expect("wide all-reduce against the push path");
}

#[test]
fn bf16_wide_combine_allreduce() {
    gpu_test_multi(512 * 1024 * 1024, |sessions| {
        if !wide_ranks(sessions)? {
            let ranks = sessions.len();
            kernel_lab::skip(&format!("wide comm test: no wide path at {ranks} ranks"))?;
            return Ok(());
        }
        for (rows, live) in [
            (33, 33),
            (64, 40),
            (100, 100),
            (128, 1),
            (255, 200),
            (256, 256),
            (64, 0),
        ] {
            run_combine(sessions, rows, live)?;
        }
        Ok(())
    })
    .expect("wide combine all-reduce");
}
