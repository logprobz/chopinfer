use kernel_core::contract::{CommQuery, DType, RmsNormEpilogue};
use kernel_core::oracle;
use kernel_core::oracle::{NormArithmetic, NormOffset, NormScale, RmsNorm};
use kernel_lab::launch::{self, Bindings, CollectiveModules};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, check_output, gpu_test_multi};
use kernels_sm120::invariance::Property;
use runtime_cuda::DeviceBuffer;
use std::collections::{BTreeMap, BTreeSet};

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

/// The bytes of `buffer` after a launch on poisoned outputs: every element
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

fn run(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let elements = 4096_usize;
    let query = CommQuery::AllReduce {
        ranks: ranks as u32,
        elements: elements as u64,
        dtype: DType::F32,
        fused_residual: false,
        fused_rmsnorm: None,
    };
    let plan = kernels_sm120::comm::plan(&query, 1).map_err(|e| e.to_string())?;
    let inputs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| session.upload_f32(&vec![(rank + 1) as f32; elements]))
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions
        .iter()
        .map(|session| session.allocate(elements * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let inboxes = sessions
        .iter()
        .map(|session| session.allocate(ranks * elements * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let flags = sessions
        .iter()
        .map(|session| session.allocate(2 * ranks * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let epochs = sessions
        .iter()
        .map(|session| session.allocate(4))
        .collect::<LabResult<Vec<_>>>()?;
    for (rank, session) in sessions.iter().enumerate() {
        flags[rank]
            .copy_from_bytes(&vec![0; 2 * ranks * 4], &session.stream)
            .map_err(|e| e.to_string())?;
        epochs[rank]
            .copy_from_bytes(&0_u32.to_ne_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
    }
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    let handles: Vec<u64> = (0..ranks).map(|rank| rank as u64 + 1).collect();
    let call = |rank: usize| -> LabResult<launch::RankCall<'_>> {
        let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
        buffers.insert(100, &inputs[rank]);
        buffers.insert(101, &outputs[rank]);
        buffers.insert(102, &epochs[rank]);
        for peer in 0..ranks {
            buffers.insert(peer as u64 + 1, &inboxes[peer]);
            buffers.insert(peer as u64 + 11, &flags[peer]);
        }
        let args = kernels_sm120::comm::arguments(kernels_sm120::comm::OneShotCall {
            input: 100,
            output: 101,
            peer_inboxes: handles.clone(),
            peer_flags: handles.iter().map(|value| value + 10).collect(),
            epoch: 102,
            elements: elements as u64,
            rank: rank as u32,
            ranks: ranks as u32,
        })
        .map_err(|e| e.to_string())?;
        Ok(launch::RankCall {
            args,
            bindings: Bindings { buffers },
        })
    };
    let launch_ranks = |only_rank: Option<usize>| {
        launch::launch_collective(
            sessions,
            &modules,
            "chop_comm_allreduce_oneshot_f32_v1",
            plan,
            only_rank,
            call,
        )
    };
    // Each rank writes every element of its own output; the README is
    // silent on the F32 entries, so the map follows comm.cuh
    // `allreduce_oneshot`.
    let written = WriteMap::of(DType::F32).prefix(elements)?;
    poison_ranks(sessions, outputs.iter().enumerate())?;
    launch_ranks(None)?;
    for session in sessions {
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    let expected = oracle::all_reduce(
        &(0..ranks)
            .map(|rank| vec![(rank + 1) as f64; elements])
            .collect::<Vec<_>>(),
        None,
        None,
    )
    .map_err(|e| e.to_string())?;
    let mut direct = Vec::new();
    for (rank, session) in sessions.iter().enumerate() {
        check_written(
            session,
            &outputs[rank],
            &written,
            &format!("rank {rank} direct F32 output"),
        )?;
        let actual = session.download_f32(&outputs[rank], elements)?;
        direct.push(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!("f32_oneshot_allreduce"));
        receipt.insert("phase".into(), serde_json::json!("direct"));
        let gate = check_output(&actual, &expected, DType::F32, &mut receipt);
        session.receipt("comm", receipt)?;
        gate?;
    }
    let graphs = RankReplay::new(sessions, |rank, _| launch_ranks(Some(rank)))?;
    poison_ranks(sessions, outputs.iter().enumerate())?;
    graphs.launch(sessions)?;
    for session in sessions {
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    for (rank, session) in sessions.iter().enumerate() {
        check_written(
            session,
            &outputs[rank],
            &written,
            &format!("rank {rank} repeated F32 output"),
        )?;
        let actual = session.download_f32(&outputs[rank], elements)?;
        oracle::check_exact(
            &actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            &direct[rank],
        )
        .map_err(|index| format!("rank {rank}: repeated F32 reduction differs at {index}"))?;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!("f32_oneshot_allreduce"));
        receipt.insert("phase".into(), serde_json::json!("graph_replay"));
        let gate = check_output(&actual, &expected, DType::F32, &mut receipt);
        session.receipt("comm", receipt)?;
        gate?;
    }
    Ok(())
}

#[test]
fn f32_oneshot_allreduce() {
    gpu_test_multi(64 * 1024 * 1024, run).expect("multi-GPU all-reduce");
}

/// Deterministic BF16 test values in [-scale, scale].
fn pattern(seed: u64, len: usize, scale: f32) -> Vec<u16> {
    use kernel_core::codec::f32_to_bf16;
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

/// The v2 fused entry: every call adds rank-ordered sums to the residual in
/// place, as the MiMo bundle does. Graph replays advance device epochs only.
/// Each call of a group writes its own output slot (the eager calls slots
/// `0..EAGER`, the captured calls the next `CAPTURED`), so a later call
/// cannot cover a store an earlier one missed.
fn run_fused(sessions: &[Session], tokens: usize) -> LabResult<Vec<u16>> {
    use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
    const DISTINCT: usize = 3;
    const EAGER: usize = 2;
    const CAPTURED: usize = 8;
    const REPLAYS: usize = 3;
    const SLOTS: usize = EAGER + CAPTURED;
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
        return Err("fused all-reduce plan must use PDL".into());
    }
    let sizes = kernels_sm120::comm::push_buffers(ranks as u32, tokens as u32)
        .map_err(|e| e.to_string())?;
    let host_inputs = (0..ranks)
        .map(|rank| {
            (0..DISTINCT)
                .map(|set| pattern((rank * DISTINCT + set) as u64, elements, 0.05))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let host_residual = pattern(99, elements, 1.0);
    let host_weights = pattern(7, 4096, 1.0)
        .into_iter()
        .map(|bits| f32_to_bf16(1.0 + 0.5 * bf16_to_f32(bits)))
        .collect::<Vec<_>>();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut residuals = Vec::new();
    let mut weights = Vec::new();
    let mut inboxes = Vec::new();
    let mut flags = Vec::new();
    let mut epochs = Vec::new();
    for (rank, session) in sessions.iter().enumerate() {
        inputs.push(
            host_inputs[rank]
                .iter()
                .map(|values| session.upload_bf16(values))
                .collect::<LabResult<Vec<_>>>()?,
        );
        outputs.push(
            (0..SLOTS)
                .map(|_| session.allocate(elements * 2))
                .collect::<LabResult<Vec<_>>>()?,
        );
        residuals.push(session.upload_bf16(&host_residual)?);
        weights.push(session.upload_bf16(&host_weights)?);
        // Only the low-latency region must start zeroed.
        let inbox = session.allocate(sizes.inbox_bytes as usize)?;
        inbox
            .copy_from_bytes(&vec![0x3c; sizes.inbox_bytes as usize], &session.stream)
            .map_err(|e| e.to_string())?;
        inbox
            .zero_at_async(0, sizes.zeroed_inbox_bytes as usize, &session.stream)
            .map_err(|e| e.to_string())?;
        inboxes.push(inbox);
        let flag = session.allocate(sizes.flag_bytes.max(4) as usize)?;
        flag.copy_from_bytes(&vec![0; sizes.flag_bytes.max(4) as usize], &session.stream)
            .map_err(|e| e.to_string())?;
        flags.push(flag);
        let epoch = session.allocate(sizes.epoch_bytes as usize)?;
        epoch
            .copy_from_bytes(&vec![0; sizes.epoch_bytes as usize], &session.stream)
            .map_err(|e| e.to_string())?;
        epochs.push(epoch);
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    let handles: Vec<u64> = (0..ranks).map(|rank| rank as u64 + 1).collect();
    let bindings = (0..ranks)
        .map(|rank| {
            let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
            for (set, input) in inputs[rank].iter().enumerate() {
                buffers.insert(200 + set as u64, input);
            }
            for (slot, output) in outputs[rank].iter().enumerate() {
                buffers.insert(300 + slot as u64, output);
            }
            buffers.insert(102, &epochs[rank]);
            buffers.insert(103, &residuals[rank]);
            buffers.insert(104, &weights[rank]);
            for peer in 0..ranks {
                buffers.insert(peer as u64 + 1, &inboxes[peer]);
                buffers.insert(peer as u64 + 11, &flags[peer]);
            }
            Bindings { buffers }
        })
        .collect::<Vec<_>>();
    let arguments = |rank: usize, set: usize, slot: usize| {
        kernels_sm120::comm::fused_arguments_v2(kernels_sm120::comm::FusedWithResidualOutputCall {
            fused: kernels_sm120::comm::FusedOneShotCall {
                base: kernels_sm120::comm::OneShotCall {
                    input: 200 + set as u64,
                    output: 300 + slot as u64,
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
        })
        .map_err(|e| e.to_string())
    };
    let launch_ranks = |set: usize, slot: usize, only_rank: Option<usize>| {
        launch::launch_collective(
            sessions,
            &modules,
            "chop_comm_allreduce_oneshot_norm_bf16_v2",
            plan,
            only_rank,
            |rank| {
                Ok(launch::RankCall {
                    args: arguments(rank, set, slot)?,
                    bindings: Bindings {
                        buffers: bindings[rank].buffers.clone(),
                    },
                })
            },
        )
    };
    // Every call writes all rows of the output (README "Fused decode
    // all-reduce"); the residual is the in-place chain, never poisoned.
    let written = WriteMap::of(DType::Bf16).prefix(elements)?;
    // Checks every rank after the calls of `sets`, the last of which wrote
    // `slots` in order: every slot written and equal to rank 0's, the
    // residual chain exact and the epoch at the call count. The final slot
    // of the last group (`last`) also keeps the original bound of one BF16
    // step from the oracle and records the result; weights above one can
    // move an earlier state's outputs two steps, as comm_sync and comm_wide
    // allow.
    let check = |phase: &str, sets: &[usize], slots: &[usize], last: bool| -> LabResult<Vec<u16>> {
        for (rank, session) in sessions.iter().enumerate() {
            for &slot in slots {
                check_written(
                    session,
                    &outputs[rank][slot],
                    &written,
                    &format!("tokens {tokens} rank {rank} {phase} output slot {slot}"),
                )?;
            }
        }
        for &slot in slots {
            let first = sessions[0].download_bf16(&outputs[0][slot], elements)?;
            for (rank, session) in sessions.iter().enumerate().skip(1) {
                let actual = session.download_bf16(&outputs[rank][slot], elements)?;
                if let Some(index) = actual.iter().zip(&first).position(|(a, b)| a != b) {
                    return Err(format!(
                        "tokens {tokens} rank {rank} {phase} slot {slot}: output differs from rank 0 at {index}"
                    ));
                }
            }
        }
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
        let final_slot = *slots.last().ok_or("a check needs a written slot")?;
        let first = sessions[0].download_bf16(&outputs[0][final_slot], elements)?;
        for (rank, session) in sessions.iter().enumerate() {
            let actual = session.download_bf16(&outputs[rank][final_slot], elements)?;
            let actual_residual = session.download_bf16(&residuals[rank], elements)?;
            let mut epoch_bytes = vec![0_u8; sizes.epoch_bytes as usize];
            epochs[rank]
                .copy_to_bytes(&mut epoch_bytes, &session.stream)
                .map_err(|e| e.to_string())?;
            let epoch = u32::from_ne_bytes(epoch_bytes[..4].try_into().unwrap());
            let residual_gate = oracle::check_exact(&actual_residual, &expected_residual);
            let cross_rank = actual.iter().zip(&first).position(|(a, b)| a != b);
            // The device sums squares in a fixed tree order; allow one BF16 step.
            let far = actual
                .iter()
                .zip(&expected)
                .position(|(&a, &b)| (i32::from(a) - i32::from(b)).abs() > 1)
                .filter(|_| last);
            if last {
                let mut receipt = BTreeMap::new();
                receipt.insert(
                    "case".into(),
                    serde_json::json!("bf16_fused_push_allreduce_residual_hf_rmsnorm"),
                );
                receipt.insert("tokens".into(), serde_json::json!(tokens));
                receipt.insert("ranks".into(), serde_json::json!(ranks));
                receipt.insert(
                    "protocol".into(),
                    serde_json::json!(format!(
                        "{:?}",
                        kernels_sm120::comm::push_protocol(ranks as u32, tokens as u32)
                            .map_err(|e| e.to_string())?
                    )),
                );
                receipt.insert("calls".into(), serde_json::json!(sets.len()));
                receipt.insert("graph_replays".into(), serde_json::json!(REPLAYS));
                receipt.insert("pdl".into(), serde_json::json!(plan.pdl));
                receipt.insert(
                    "residual_correct".into(),
                    serde_json::json!(residual_gate.is_ok()),
                );
                receipt.insert(
                    "bitwise_equal_to_rank0".into(),
                    serde_json::json!(cross_rank.is_none()),
                );
                receipt.insert(
                    "output_within_one_bf16_step".into(),
                    serde_json::json!(far.is_none()),
                );
                receipt.insert("device_epoch".into(), serde_json::json!(epoch));
                session.receipt("comm", receipt)?;
            }
            residual_gate.map_err(|index| {
                format!("tokens {tokens} rank {rank} {phase}: pre-norm residual differs at {index}")
            })?;
            if let Some(index) = cross_rank {
                return Err(format!(
                    "tokens {tokens} rank {rank} {phase}: output differs from rank 0 at {index}"
                ));
            }
            if let Some(index) = far {
                return Err(format!(
                    "tokens {tokens} rank {rank} {phase}: output {} vs oracle {} at {index}",
                    actual[index], expected[index]
                ));
            }
            if epoch as usize != sets.len() {
                return Err(format!(
                    "tokens {tokens} rank {rank} {phase}: device epoch {epoch}"
                ));
            }
        }
        Ok(first)
    };
    let poison_slots = |slots: std::ops::Range<usize>| {
        poison_ranks(
            sessions,
            outputs.iter().enumerate().flat_map(|(rank, buffers)| {
                buffers[slots.clone()]
                    .iter()
                    .map(move |buffer| (rank, buffer))
            }),
        )
    };
    // The eager calls, then each graph replay, queued without host
    // synchronization on poisoned output slots.
    let mut sets = Vec::new();
    poison_slots(0..EAGER)?;
    for call in 0..EAGER {
        launch_ranks(call % DISTINCT, call, None)?;
        sets.push(call % DISTINCT);
    }
    let graphs = RankReplay::new(sessions, |rank, _| {
        for call in 0..CAPTURED {
            launch_ranks((EAGER + call) % DISTINCT, EAGER + call, Some(rank))?;
        }
        Ok(())
    })?;
    poison::synchronize(sessions)?;
    let eager_slots = (0..EAGER).collect::<Vec<_>>();
    check("eager", &sets, &eager_slots, false)?;
    // Poisoned and checked before and after every replay of the same graph.
    let graph_slots = (EAGER..SLOTS).collect::<Vec<_>>();
    let mut first = Vec::new();
    for replay in 0..REPLAYS {
        poison_slots(EAGER..SLOTS)?;
        graphs.launch(sessions)?;
        sets.extend((0..CAPTURED).map(|call| (EAGER + call) % DISTINCT));
        poison::synchronize(sessions)?;
        first = check(
            &format!("graph replay {replay}"),
            &sets,
            &graph_slots,
            replay + 1 == REPLAYS,
        )?;
    }
    Ok(first)
}

/// The combine entry on preweighted routed rows against the v2 entry on
/// host-combined rows: every output must match bitwise after eager calls and
/// graph replays, with and without padded rows.
fn run_fused_combine(sessions: &[Session], tokens: usize, live: usize) -> LabResult<()> {
    use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
    const CALLS: usize = 6;
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
    let sizes = kernels_sm120::comm::push_buffers(ranks as u32, tokens as u32)
        .map_err(|e| e.to_string())?;
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
    let weights = pattern(72, 4096, 1.0)
        .into_iter()
        .map(|bits| f32_to_bf16(1.0 + 0.5 * bf16_to_f32(bits)))
        .collect::<Vec<_>>();
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    // Both arms write every row below `tokens` of the output and the
    // residual: the combine entry's rows at or past live take a zero sum
    // and are still written (README "rows at or past `*live_rows` contribute
    // zero"; push.cuh `input_vector`), so both maps hold all rows.
    let written = WriteMap::of(DType::Bf16).prefix(elements)?;
    let mut finals = Vec::new();
    for fused in [false, true] {
        let zeroed = |bytes: usize| -> LabResult<Vec<DeviceBuffer>> {
            sessions
                .iter()
                .map(|session| {
                    let buffer = session.allocate(bytes.max(4))?;
                    buffer
                        .copy_from_bytes(&vec![0; bytes.max(4)], &session.stream)
                        .map_err(|e| e.to_string())?;
                    Ok(buffer)
                })
                .collect()
        };
        let inboxes = zeroed(sizes.inbox_bytes as usize)?;
        let flags = zeroed(sizes.flag_bytes as usize)?;
        let epochs = zeroed(sizes.epoch_bytes as usize)?;
        let outputs = sessions
            .iter()
            .map(|session| session.allocate(elements * 2))
            .collect::<LabResult<Vec<_>>>()?;
        let mut inputs = Vec::new();
        let mut residuals = Vec::new();
        let mut scales = Vec::new();
        let mut lives = Vec::new();
        for (rank, session) in sessions.iter().enumerate() {
            inputs.push(session.upload_bf16(if fused {
                &routed[rank]
            } else {
                &combined[rank]
            })?);
            residuals.push(session.upload_bf16(&residual)?);
            scales.push(session.upload_bf16(&weights)?);
            let count = session.allocate(4)?;
            count
                .copy_from_bytes(&(live as u32).to_ne_bytes(), &session.stream)
                .map_err(|e| e.to_string())?;
            lives.push(count);
            session.stream.synchronize().map_err(|e| e.to_string())?;
        }
        let symbol = if fused {
            "chop_comm_allreduce_combine_norm_bf16_v1"
        } else {
            "chop_comm_allreduce_oneshot_norm_bf16_v2"
        };
        let handles: Vec<u64> = (0..ranks as u64).map(|rank| rank + 1).collect();
        let call = |rank: usize| -> LabResult<launch::RankCall<'_>> {
            let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
            buffers.insert(100, &inputs[rank]);
            buffers.insert(101, &outputs[rank]);
            buffers.insert(102, &epochs[rank]);
            buffers.insert(103, &residuals[rank]);
            buffers.insert(104, &scales[rank]);
            buffers.insert(105, &lives[rank]);
            for peer in 0..ranks {
                buffers.insert(peer as u64 + 1, &inboxes[peer]);
                buffers.insert(peer as u64 + 11, &flags[peer]);
            }
            let call = kernels_sm120::comm::FusedWithResidualOutputCall {
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
            };
            let args = if fused {
                kernels_sm120::comm::fused_combine_arguments(call, Some(105))
            } else {
                kernels_sm120::comm::fused_arguments_v2(call)
            }
            .map_err(|e| e.to_string())?;
            Ok(launch::RankCall {
                args,
                bindings: Bindings { buffers },
            })
        };
        let issue = |only_rank: Option<usize>| {
            launch::launch_collective(sessions, &modules, symbol, plan, only_rank, call)
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
                    &format!("combine {fused} tokens {tokens} live {live} rank {rank} {phase}"),
                )?;
                state.push((
                    session.download_bf16(&outputs[rank], elements)?,
                    session.download_bf16(&residuals[rank], elements)?,
                ));
            }
            if state.iter().any(|item| item != &state[0]) {
                return Err(format!(
                    "combine {fused} tokens {tokens} {phase}: ranks differ"
                ));
            }
            Ok(state.swap_remove(0))
        };
        // The residual is the in-place chain of both groups, never poisoned.
        poison_ranks(sessions, outputs.iter().enumerate())?;
        for _ in 0..CALLS / 2 {
            issue(None)?;
        }
        let graphs = RankReplay::new(sessions, |rank, _| {
            (0..CALLS / 2).try_for_each(|_| issue(Some(rank)))
        })?;
        poison::synchronize(sessions)?;
        let eager = state("eager")?;
        poison_ranks(sessions, outputs.iter().enumerate())?;
        graphs.launch(sessions)?;
        poison::synchronize(sessions)?;
        finals.push([eager, state("graph replay")?]);
    }
    let equal = finals[0] == finals[1];
    for (rank, session) in sessions.iter().enumerate() {
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "case".into(),
            serde_json::json!("bf16_fused_combine_allreduce"),
        );
        receipt.insert("tokens".into(), serde_json::json!(tokens));
        receipt.insert("live_rows".into(), serde_json::json!(live));
        receipt.insert("ranks".into(), serde_json::json!(ranks));
        receipt.insert("rank".into(), serde_json::json!(rank));
        receipt.insert(
            "bitwise_equal_to_combine_then_v2".into(),
            serde_json::json!(equal),
        );
        session.receipt("comm", receipt)?;
    }
    if !equal {
        return Err(format!(
            "tokens {tokens} live {live}: fused combine differs from combine + v2"
        ));
    }
    Ok(())
}

#[test]
fn bf16_fused_combine_allreduce() {
    gpu_test_multi(256 * 1024 * 1024, |sessions| {
        for (tokens, live) in [
            (1, 1),
            (2, 1),
            (4, 3),
            (8, 8),
            (16, 11),
            (17, 17),
            (32, 32),
            (32, 20),
        ] {
            run_fused_combine(sessions, tokens, live)?;
        }
        Ok(())
    })
    .expect("multi-GPU fused combine all-reduce");
}

/// Padded one-shot rows (live < tokens) raced the combine staging: rows that
/// finish early overwrote combined values the sender had not yet copied to
/// every peer. Each pass issues eager calls and a graph replay and compares
/// every rank bitwise; CHOPINFER_KERNEL_COMM_STRESS sets the pass count.
#[test]
fn bf16_fused_combine_allreduce_stress() {
    let passes = std::env::var("CHOPINFER_KERNEL_COMM_STRESS")
        .ok()
        .map(|value| value.parse::<usize>().expect("integer pass count"))
        .unwrap_or(24);
    gpu_test_multi(256 * 1024 * 1024, |sessions| {
        for pass in 0..passes {
            for (tokens, live) in [(16, 11), (16, 1), (8, 5)] {
                run_fused_combine(sessions, tokens, live)
                    .map_err(|error| format!("pass {pass}: {error}"))?;
            }
        }
        Ok(())
    })
    .expect("multi-GPU fused combine all-reduce stress");
}

#[test]
fn bf16_fused_allreduce() {
    gpu_test_multi(256 * 1024 * 1024, |sessions| {
        let tokens = std::env::var("CHOPINFER_KERNEL_COMM_TOKENS")
            .ok()
            .map(|value| vec![value.parse::<usize>().expect("integer token count")])
            .unwrap_or_else(|| vec![1, 2, 3, 4, 8, 16, 17, 32]);
        for tokens in tokens {
            let first = run_fused(sessions, tokens)?;
            if sessions.len() != 4 {
                let repeated = run_fused(sessions, tokens)?;
                oracle::check_exact(&repeated, &first)
                    .map_err(|index| format!("tokens {tokens}: repeated run differs at {index}"))?;
            }
        }
        Ok(())
    })
    .expect("multi-GPU fused all-reduce");
}

fn run_two_shot(sessions: &[Session]) -> LabResult<()> {
    use kernel_core::contract::{Argument, ArgumentList};
    let ranks = sessions.len();
    let elements = 65537_usize;
    let tiles = elements.div_ceil(4096);
    let query = CommQuery::AllReduce {
        ranks: ranks as u32,
        elements: elements as u64,
        dtype: DType::F32,
        fused_residual: false,
        fused_rmsnorm: None,
    };
    let plan = kernels_sm120::comm::plan(&query, 1).map_err(|e| e.to_string())?;
    let inputs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| session.upload_f32(&vec![(rank + 1) as f32; elements]))
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions
        .iter()
        .map(|session| session.allocate(elements * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let sizes = kernels_sm120::comm::phased_buffers(ranks as u32, elements as u64, DType::F32)
        .map_err(|e| e.to_string())?;
    let inboxes = sessions
        .iter()
        .map(|session| session.allocate(sizes.inbox_bytes as usize))
        .collect::<LabResult<Vec<_>>>()?;
    let flags = sessions
        .iter()
        .map(|session| session.allocate(4 * ranks * tiles * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let epochs = sessions
        .iter()
        .map(|session| session.allocate(tiles * 4))
        .collect::<LabResult<Vec<_>>>()?;
    for (rank, session) in sessions.iter().enumerate() {
        flags[rank]
            .copy_from_bytes(&vec![0; 4 * ranks * tiles * 4], &session.stream)
            .map_err(|e| e.to_string())?;
        epochs[rank]
            .copy_from_bytes(&vec![0; tiles * 4], &session.stream)
            .map_err(|e| e.to_string())?;
    }
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    let peer_handles: Vec<u64> = (0..ranks).map(|rank| rank as u64 + 1).collect();
    let output_handles: Vec<u64> = (0..ranks).map(|rank| rank as u64 + 21).collect();
    // The scatter on every rank, then the gather on every rank.
    let launch_ranks = |only_rank: Option<usize>| -> LabResult<()> {
        let stage = |rank: usize, gather: bool| -> LabResult<launch::RankCall<'_>> {
            let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
            buffers.insert(100, &inputs[rank]);
            buffers.insert(101, &outputs[rank]);
            buffers.insert(102, &epochs[rank]);
            for peer in 0..ranks {
                buffers.insert(peer as u64 + 1, &inboxes[peer]);
                buffers.insert(peer as u64 + 11, &flags[peer]);
                buffers.insert(peer as u64 + 21, &outputs[peer]);
            }
            let args = if gather {
                ArgumentList(vec![
                    Argument::DeviceBuffer(101),
                    Argument::DevicePointerArray(output_handles.clone()),
                    Argument::DevicePointerArray(peer_handles.iter().map(|h| h + 10).collect()),
                    Argument::DeviceBuffer(102),
                    Argument::U64(elements as u64),
                    Argument::U32(rank as u32),
                    Argument::U32(ranks as u32),
                ])
            } else {
                kernels_sm120::comm::arguments(kernels_sm120::comm::OneShotCall {
                    input: 100,
                    output: 101,
                    peer_inboxes: peer_handles.clone(),
                    peer_flags: peer_handles.iter().map(|h| h + 10).collect(),
                    epoch: 102,
                    elements: elements as u64,
                    rank: rank as u32,
                    ranks: ranks as u32,
                })
                .map_err(|e| e.to_string())?
            };
            Ok(launch::RankCall {
                args,
                bindings: Bindings { buffers },
            })
        };
        launch::launch_collective(
            sessions,
            &modules,
            "chop_comm_reduce_scatter_f32_v1",
            plan,
            only_rank,
            |rank| stage(rank, false),
        )?;
        launch::launch_collective(
            sessions,
            &modules,
            "chop_comm_allgather_scattered_f32_v1",
            plan,
            only_rank,
            |rank| stage(rank, true),
        )
    };
    // Every rank's output ends up whole: the scatter writes the owner's
    // tiles and the gather sends them into every peer's output (README
    // "Prefill all-reduce").
    let written = WriteMap::of(DType::F32).prefix(elements)?;
    poison_ranks(sessions, outputs.iter().enumerate())?;
    launch_ranks(None)?;
    for session in sessions {
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    let expected = oracle::all_reduce(
        &(0..ranks)
            .map(|rank| vec![(rank + 1) as f64; elements])
            .collect::<Vec<_>>(),
        None,
        None,
    )
    .map_err(|e| e.to_string())?;
    let mut direct = Vec::new();
    for (rank, session) in sessions.iter().enumerate() {
        check_written(
            session,
            &outputs[rank],
            &written,
            &format!("rank {rank} direct two-shot output"),
        )?;
        let actual = session.download_f32(&outputs[rank], elements)?;
        direct.push(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!("f32_two_shot_allreduce"));
        let gate = check_output(&actual, &expected, DType::F32, &mut receipt);
        session.receipt("comm", receipt)?;
        gate?;
    }
    let graphs = RankReplay::new(sessions, |rank, _| launch_ranks(Some(rank)))?;
    poison_ranks(sessions, outputs.iter().enumerate())?;
    graphs.launch(sessions)?;
    for session in sessions {
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    for (rank, session) in sessions.iter().enumerate() {
        check_written(
            session,
            &outputs[rank],
            &written,
            &format!("rank {rank} repeated two-shot output"),
        )?;
        let actual = session.download_f32(&outputs[rank], elements)?;
        oracle::check_exact(
            &actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            &direct[rank],
        )
        .map_err(|index| format!("rank {rank}: repeated F32 reduction differs at {index}"))?;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!("f32_two_shot_allreduce"));
        receipt.insert("phase".into(), serde_json::json!("graph_replay"));
        let gate = check_output(&actual, &expected, DType::F32, &mut receipt);
        session.receipt("comm", receipt)?;
        gate?;
    }
    Ok(())
}

#[test]
fn f32_two_shot_allreduce() {
    gpu_test_multi(64 * 1024 * 1024, run_two_shot).expect("multi-GPU two-shot all-reduce");
}

/// The phased F32 pair over 2048 rows of 4096 elements, at its planned
/// grid and at narrower ones (the sending cluster and one to four clusters
/// of workers), one after another on the same protocol state: every grid
/// writes every element of every rank's output, each the rank-ordered F32
/// sum of the ranks' inputs, bit for bit; a graph replay at 24 CTAs too.
/// Inputs span 24 binades, so the sums round, and a different order would
/// show.
fn run_two_shot_grids(sessions: &[Session]) -> LabResult<()> {
    use kernel_core::contract::{Argument, ArgumentList};
    let ranks = sessions.len();
    let elements = 2048 * 4096_usize;
    let query = CommQuery::AllReduce {
        ranks: ranks as u32,
        elements: elements as u64,
        dtype: DType::F32,
        fused_residual: false,
        fused_rmsnorm: None,
    };
    let planned = kernels_sm120::comm::plan(&query, 2048).map_err(|e| e.to_string())?;
    let value = |rank: usize, index: usize| {
        let mut x = ((rank * elements + index) as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x ^= x >> 29;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 32;
        let mantissa = ((x % 2001) as f32 - 1000.0) / 1000.0;
        mantissa * 2f32.powi(((x >> 16) % 24) as i32 - 12)
    };
    let values = (0..ranks)
        .map(|rank| {
            (0..elements)
                .map(|index| value(rank, index))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // Rank 0 first, from zero, one F32 rounding per add.
    let expected = (0..elements)
        .map(|index| {
            let mut sum = 0.0_f32;
            for rank in &values {
                sum += rank[index];
            }
            sum.to_bits()
        })
        .collect::<Vec<_>>();
    let inputs = sessions
        .iter()
        .zip(&values)
        .map(|(session, values)| session.upload_f32(values))
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions
        .iter()
        .map(|session| session.allocate(elements * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let sizes = kernels_sm120::comm::phased_buffers(ranks as u32, elements as u64, DType::F32)
        .map_err(|e| e.to_string())?;
    let allocate = |bytes: u64| {
        sessions
            .iter()
            .map(|session| session.allocate(bytes as usize))
            .collect::<LabResult<Vec<_>>>()
    };
    let inboxes = allocate(sizes.inbox_bytes)?;
    let flags = allocate(sizes.flag_bytes)?;
    let epochs = allocate(sizes.epoch_bytes)?;
    for (rank, session) in sessions.iter().enumerate() {
        flags[rank]
            .copy_from_bytes(&vec![0; sizes.flag_bytes as usize], &session.stream)
            .map_err(|e| e.to_string())?;
        epochs[rank]
            .copy_from_bytes(&vec![0; sizes.epoch_bytes as usize], &session.stream)
            .map_err(|e| e.to_string())?;
    }
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    let peer_handles: Vec<u64> = (0..ranks).map(|rank| rank as u64 + 1).collect();
    let output_handles: Vec<u64> = (0..ranks).map(|rank| rank as u64 + 21).collect();
    // The scatter on every rank, then the gather, at `grid` CTAs.
    let launch_ranks = |grid: u32, only_rank: Option<usize>| -> LabResult<()> {
        let plan = kernel_core::contract::LaunchPlan {
            grid: [grid, 1, 1],
            ..planned
        };
        let stage = |rank: usize, gather: bool| -> LabResult<launch::RankCall<'_>> {
            let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
            buffers.insert(100, &inputs[rank]);
            buffers.insert(101, &outputs[rank]);
            buffers.insert(102, &epochs[rank]);
            for peer in 0..ranks {
                buffers.insert(peer as u64 + 1, &inboxes[peer]);
                buffers.insert(peer as u64 + 11, &flags[peer]);
                buffers.insert(peer as u64 + 21, &outputs[peer]);
            }
            let args = if gather {
                ArgumentList(vec![
                    Argument::DeviceBuffer(101),
                    Argument::DevicePointerArray(output_handles.clone()),
                    Argument::DevicePointerArray(peer_handles.iter().map(|h| h + 10).collect()),
                    Argument::DeviceBuffer(102),
                    Argument::U64(elements as u64),
                    Argument::U32(rank as u32),
                    Argument::U32(ranks as u32),
                ])
            } else {
                kernels_sm120::comm::arguments(kernels_sm120::comm::OneShotCall {
                    input: 100,
                    output: 101,
                    peer_inboxes: peer_handles.clone(),
                    peer_flags: peer_handles.iter().map(|h| h + 10).collect(),
                    epoch: 102,
                    elements: elements as u64,
                    rank: rank as u32,
                    ranks: ranks as u32,
                })
                .map_err(|e| e.to_string())?
            };
            Ok(launch::RankCall {
                args,
                bindings: Bindings { buffers },
            })
        };
        launch::launch_collective(
            sessions,
            &modules,
            "chop_comm_reduce_scatter_f32_v1",
            plan,
            only_rank,
            |rank| stage(rank, false),
        )?;
        launch::launch_collective(
            sessions,
            &modules,
            "chop_comm_allgather_scattered_f32_v1",
            plan,
            only_rank,
            |rank| stage(rank, true),
        )
    };
    let written = WriteMap::of(DType::F32).prefix(elements)?;
    let check = |grid: u32, phase: &str| -> LabResult<()> {
        for session in sessions {
            session.stream.synchronize().map_err(|e| e.to_string())?;
        }
        for (rank, session) in sessions.iter().enumerate() {
            check_written(
                session,
                &outputs[rank],
                &written,
                &format!("rank {rank} {phase} output at {grid} CTAs"),
            )?;
            let actual = session.download_f32(&outputs[rank], elements)?;
            oracle::check_exact(
                &actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                &expected,
            )
            .map_err(|index| {
                format!("rank {rank}: {phase} sum at {grid} CTAs differs at {index}")
            })?;
            let mut receipt = BTreeMap::new();
            receipt.insert(
                "case".into(),
                serde_json::json!("f32_two_shot_allreduce_any_worker_grid"),
            );
            receipt.insert("phase".into(), serde_json::json!(phase));
            receipt.insert("grid".into(), serde_json::json!(grid));
            receipt.insert("planned_grid".into(), serde_json::json!(planned.grid[0]));
            receipt.insert("bitwise".into(), serde_json::json!(true));
            session.receipt("comm", receipt)?;
        }
        Ok(())
    };
    for grid in [planned.grid[0], 16, 24, 32, 40, planned.grid[0]] {
        poison_ranks(sessions, outputs.iter().enumerate())?;
        launch_ranks(grid, None)?;
        check(grid, "eager")?;
    }
    let graphs = RankReplay::new(sessions, |rank, _| launch_ranks(24, Some(rank)))?;
    for _ in 0..2 {
        poison_ranks(sessions, outputs.iter().enumerate())?;
        graphs.launch(sessions)?;
        check(24, "graph_replay")?;
    }
    Ok(())
}

#[test]
fn f32_two_shot_allreduce_any_worker_grid() {
    gpu_test_multi(512 * 1024 * 1024, run_two_shot_grids)
        .expect("multi-GPU two-shot all-reduce at narrower grids");
}

fn run_allgather(sessions: &[Session], target: u32) -> LabResult<()> {
    use kernel_core::contract::{Argument, ArgumentList};
    let ranks = sessions.len();
    let elements = 38144_usize;
    let tiles = elements.div_ceil(4096);
    let query = CommQuery::AllGather {
        ranks: ranks as u32,
        elements_per_rank: elements as u64,
        dtype: DType::F32,
    };
    let plan = kernels_sm120::comm::plan(&query, 1).map_err(|e| e.to_string())?;
    let inputs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| session.upload_f32(&vec![(rank + 1) as f32; elements]))
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions
        .iter()
        .map(|session| session.allocate(ranks * elements * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let flags = sessions
        .iter()
        .map(|session| session.allocate(2 * ranks * tiles * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let epochs = sessions
        .iter()
        .map(|session| session.allocate(tiles * 4))
        .collect::<LabResult<Vec<_>>>()?;
    for (rank, session) in sessions.iter().enumerate() {
        flags[rank]
            .copy_from_bytes(&vec![0; 2 * ranks * tiles * 4], &session.stream)
            .map_err(|e| e.to_string())?;
        epochs[rank]
            .copy_from_bytes(&vec![0; tiles * 4], &session.stream)
            .map_err(|e| e.to_string())?;
    }
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    let handles: Vec<u64> = (0..ranks).map(|rank| rank as u64 + 1).collect();
    // A target below the rank count receives every rank's shard, rank-major;
    // no other output is written (README "Buffer sharing and other
    // entries": with a target rank it writes only to that rank).
    let gathered = |rank: usize| target == ranks as u32 || rank as u32 == target;
    let written = WriteMap::of(DType::F32).prefix(ranks * elements)?;
    let untouched = WriteMap::of(DType::F32).none()?;
    poison_ranks(sessions, outputs.iter().enumerate())?;
    launch::launch_collective(
        sessions,
        &modules,
        "chop_comm_allgather_f32_v1",
        plan,
        None,
        |rank| {
            let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
            buffers.insert(100, &inputs[rank]);
            buffers.insert(102, &epochs[rank]);
            for peer in 0..ranks {
                buffers.insert(peer as u64 + 1, &outputs[peer]);
                buffers.insert(peer as u64 + 11, &flags[peer]);
            }
            let args = ArgumentList(vec![
                Argument::DeviceBuffer(100),
                Argument::DevicePointerArray(handles.clone()),
                Argument::DevicePointerArray(handles.iter().map(|h| h + 10).collect()),
                Argument::DeviceBuffer(102),
                Argument::U64(elements as u64),
                Argument::U32(rank as u32),
                Argument::U32(ranks as u32),
                Argument::U32(target),
            ]);
            Ok(launch::RankCall {
                args,
                bindings: Bindings { buffers },
            })
        },
    )?;
    for session in sessions {
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    let expected = oracle::all_gather(
        &(0..ranks)
            .map(|rank| vec![(rank + 1) as f64; elements])
            .collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;
    let reference = oracle::OracleOutput::new(expected.clone(), vec![0.0; expected.len()], 1)
        .map_err(|e| e.to_string())?;
    for (rank, session) in sessions.iter().enumerate() {
        check_written(
            session,
            &outputs[rank],
            if gathered(rank) { &written } else { &untouched },
            &format!("target {target} rank {rank} all-gather output"),
        )?;
        if !gathered(rank) {
            continue;
        }
        let actual = session.download_f32(&outputs[rank], ranks * elements)?;
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "case".into(),
            serde_json::json!(if target == ranks as u32 {
                "f32_allgather_all_ranks"
            } else {
                "f32_allgather_one_rank"
            }),
        );
        let gate = check_output(&actual, &reference, DType::F32, &mut receipt);
        session.receipt("comm", receipt)?;
        gate?;
    }
    Ok(())
}

/// Batched logits gather: listed rows and a leading range, eager and in
/// graphs whose replays see new row lists and new inputs.
fn run_gather_rows(sessions: &[Session], target: u32) -> LabResult<()> {
    const CAPACITY: usize = 40;
    const SHARD: usize = 38144;
    let ranks = sessions.len();
    let full = ranks * SHARD;
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    let host_inputs = |seed: u64| {
        (0..ranks)
            .map(|rank| pattern(seed * 8 + rank as u64, CAPACITY * SHARD, 4.0))
            .collect::<Vec<_>>()
    };
    let mut inputs_now = host_inputs(1);
    let inputs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| session.upload_bf16(&inputs_now[rank]))
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions
        .iter()
        .map(|session| session.allocate(CAPACITY * full * 2))
        .collect::<LabResult<Vec<_>>>()?;
    let flags = sessions
        .iter()
        .map(|session| {
            let bytes = kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES as usize;
            let buffer = session.allocate(bytes)?;
            buffer
                .copy_from_bytes(&vec![0; bytes], &session.stream)
                .map_err(|e| e.to_string())?;
            Ok(buffer)
        })
        .collect::<LabResult<Vec<_>>>()?;
    let epochs = sessions
        .iter()
        .map(|session| {
            let bytes = kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES as usize;
            let buffer = session.allocate(bytes)?;
            buffer
                .copy_from_bytes(&vec![0; bytes], &session.stream)
                .map_err(|e| e.to_string())?;
            Ok(buffer)
        })
        .collect::<LabResult<Vec<_>>>()?;
    let list_bytes = |rows: &[u32]| {
        rows.iter()
            .flat_map(|row| row.to_ne_bytes())
            .collect::<Vec<_>>()
    };
    let lists = sessions
        .iter()
        .map(|session| session.allocate(16))
        .collect::<LabResult<Vec<_>>>()?;
    let set_list = |rows: &[u32]| -> LabResult<()> {
        for (list, session) in lists.iter().zip(sessions) {
            list.copy_from_bytes(&list_bytes(rows), &session.stream)
                .map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    let plan = kernels_sm120::comm::gather_rows_plan();
    let launch_call = |only_rank: Option<usize>, listed: bool, rows: u32| -> LabResult<()> {
        launch::launch_collective(
            sessions,
            &modules,
            "chop_comm_gather_rows_bf16_v1",
            plan,
            only_rank,
            |rank| {
                let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
                buffers.insert(100, &inputs[rank]);
                buffers.insert(102, &epochs[rank]);
                buffers.insert(103, &lists[rank]);
                for peer in 0..ranks {
                    buffers.insert(peer as u64 + 1, &outputs[peer]);
                    buffers.insert(peer as u64 + 11, &flags[peer]);
                }
                let args = kernels_sm120::comm::gather_rows_arguments(
                    kernels_sm120::comm::GatherRowsCall {
                        input: 100,
                        outputs: (1..=ranks as u64).collect(),
                        flags: (11..11 + ranks as u64).collect(),
                        epoch: 102,
                        row_indices: listed.then_some(103),
                        rows,
                        shard: SHARD as u64,
                        rank: rank as u32,
                        ranks: ranks as u32,
                        target,
                    },
                )
                .map_err(|e| e.to_string())?;
                Ok(launch::RankCall {
                    args,
                    bindings: Bindings { buffers },
                })
            },
        )
    };
    let check = |inputs_now: &[Vec<u16>], rows: &[u32], label: &str| -> LabResult<()> {
        let actual =
            sessions[target as usize].download_bf16(&outputs[target as usize], CAPACITY * full)?;
        for &row in rows {
            for (rank, input) in inputs_now.iter().enumerate() {
                let at = (row as usize * ranks + rank) * SHARD;
                let expected = &input[row as usize * SHARD..(row as usize + 1) * SHARD];
                if &actual[at..at + SHARD] != expected {
                    return Err(format!("{label}: row {row} shard {rank} differs"));
                }
            }
        }
        Ok(())
    };
    let sync = || -> LabResult<()> {
        for session in sessions {
            session.stream.synchronize().map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    // Only the target's output is written: every listed or leading row in
    // full, rank q's shard of row r at (r * ranks + q) * shard (README
    // "Logits gather"; senders write only to the target).
    let written = |rows: &[u32], label: &str| -> LabResult<()> {
        let rows = rows
            .iter()
            .map(|&row| row as usize)
            .collect::<BTreeSet<_>>();
        for (rank, session) in sessions.iter().enumerate() {
            let map = if rank == target as usize {
                WriteMap::of(DType::Bf16).rows(full, rows.iter().copied())?
            } else {
                WriteMap::of(DType::Bf16).none()?
            };
            check_written(
                session,
                &outputs[rank],
                &map,
                &format!("{label}: rank {rank} output"),
            )?;
        }
        Ok(())
    };
    set_list(&[5, 0, 17, 3])?;
    poison_ranks(sessions, outputs.iter().enumerate())?;
    launch_call(None, true, 4)?;
    sync()?;
    written(&[5, 0, 17, 3], "listed rows")?;
    check(&inputs_now, &[5, 0, 17, 3], "listed rows")?;
    let untouched =
        sessions[target as usize].download_bf16(&outputs[target as usize], CAPACITY * full)?;
    if untouched[full..2 * full].iter().any(|&bits| bits != 0xffff) {
        return Err("an unlisted row was written".into());
    }
    poison_ranks(sessions, outputs.iter().enumerate())?;
    launch_call(None, false, 32)?;
    sync()?;
    let leading = (0..32).collect::<Vec<_>>();
    written(&leading, "leading rows")?;
    check(&inputs_now, &leading, "leading rows")?;
    let graphs = RankReplay::new(sessions, |rank, _| {
        launch_call(Some(rank), true, 4)?;
        launch_call(Some(rank), false, 8)
    })?;
    for (replay, rows) in [[39_u32, 1, 30, 12], [2, 38, 33, 7]].iter().enumerate() {
        set_list(rows)?;
        inputs_now = host_inputs(2 + replay as u64);
        for (rank, session) in sessions.iter().enumerate() {
            inputs[rank]
                .copy_from_bytes(
                    &inputs_now[rank]
                        .iter()
                        .flat_map(|value| value.to_ne_bytes())
                        .collect::<Vec<_>>(),
                    &session.stream,
                )
                .map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
        }
        poison_ranks(sessions, outputs.iter().enumerate())?;
        graphs.launch(sessions)?;
        sync()?;
        let mut expected_rows = rows.to_vec();
        expected_rows.extend(0..8);
        let label = format!("graph replay {replay}");
        written(&expected_rows, &label)?;
        check(&inputs_now, &expected_rows, &label)?;
    }
    for (rank, session) in sessions.iter().enumerate() {
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!("bf16_gather_rows"));
        receipt.insert("ranks".into(), serde_json::json!(ranks));
        receipt.insert("rank".into(), serde_json::json!(rank));
        receipt.insert("target".into(), serde_json::json!(target));
        receipt.insert("shard".into(), serde_json::json!(SHARD));
        receipt.insert("graph_replays".into(), serde_json::json!(2));
        receipt.insert("correct".into(), serde_json::json!(true));
        session.receipt("comm", receipt)?;
    }
    Ok(())
}

/// The logits row exchange with a late receiver: every other rank launches
/// a call while `late` has not, and must wait for `late`'s acknowledgment
/// instead of overwriting the rows `late` still holds from the previous call.
/// Epochs start next to the wrap, so the calls cover epochs 0xffff_ffff, 0 and
/// 1, where a sender that skips the acknowledgment on small epochs would write
/// early.
fn run_exchange_rows_late_receiver(sessions: &[Session], late: usize) -> LabResult<()> {
    const ROWS: u32 = 8;
    const SHARD: usize = 1024;
    let ranks = sessions.len();
    let full = ranks * SHARD;
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    let late_ranks = [late];
    let value = |call: usize, rank: usize, row: usize, column: usize| {
        (call * 1000 + rank * 100 + row) as f32 + (column % 2) as f32 * 0.5
    };
    let host_input = |call: usize, rank: usize| {
        (0..ROWS as usize * SHARD)
            .map(|index| value(call, rank, index / SHARD, index % SHARD))
            .collect::<Vec<_>>()
    };
    let inputs = sessions
        .iter()
        .map(|session| session.allocate(ROWS as usize * SHARD * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions
        .iter()
        .map(|session| session.upload_f32(&vec![-1.0; ROWS as usize * full]))
        .collect::<LabResult<Vec<_>>>()?;
    let zeroed = |session: &Session, bytes: Vec<u8>| -> LabResult<DeviceBuffer> {
        let buffer = session.allocate(bytes.len())?;
        buffer
            .copy_from_bytes(&bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        Ok(buffer)
    };
    let flags = sessions
        .iter()
        .map(|session| {
            zeroed(
                session,
                vec![0; kernels_sm120::comm::EXCHANGE_ROWS_FLAG_BYTES as usize],
            )
        })
        .collect::<LabResult<Vec<_>>>()?;
    // The next call runs at epoch 0xffff_ffff, then 0 and 1.
    let epochs = sessions
        .iter()
        .map(|session| zeroed(session, 0xffff_fffe_u32.to_ne_bytes().repeat(8)))
        .collect::<LabResult<Vec<_>>>()?;
    let launch_call = |only_rank: usize| -> LabResult<()> {
        launch::launch_collective(
            sessions,
            &modules,
            "chop_comm_exchange_rows_f32_v1",
            kernels_sm120::comm::exchange_rows_plan(),
            Some(only_rank),
            |rank| {
                let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
                buffers.insert(100, &inputs[rank]);
                buffers.insert(102, &epochs[rank]);
                for peer in 0..ranks {
                    buffers.insert(peer as u64 + 1, &outputs[peer]);
                    buffers.insert(peer as u64 + 11, &flags[peer]);
                }
                let args = kernels_sm120::comm::exchange_rows_arguments(
                    kernels_sm120::comm::ExchangeRowsCall {
                        input: 100,
                        outputs: (1..=ranks as u64).collect(),
                        flags: (11..11 + ranks as u64).collect(),
                        epoch: 102,
                        rows: ROWS,
                        shard: SHARD as u64,
                        rank: rank as u32,
                        ranks: ranks as u32,
                    },
                )
                .map_err(|e| e.to_string())?;
                Ok(launch::RankCall {
                    args,
                    bindings: Bindings { buffers },
                })
            },
        )
    };
    let set_inputs = |call: usize| -> LabResult<()> {
        for (rank, session) in sessions.iter().enumerate() {
            let bytes = host_input(call, rank)
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>();
            inputs[rank]
                .copy_from_bytes(&bytes, &session.stream)
                .map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    // Each owner's rows hold every rank's shard from `call`.
    let owned_rows_hold = |owner: usize, call: usize| -> LabResult<bool> {
        let actual = sessions[owner].download_f32(&outputs[owner], ROWS as usize * full)?;
        let owned = kernels_sm120::comm::exchange_rows_owned(ROWS, ranks as u32, owner as u32);
        Ok(owned.into_iter().all(|row| {
            (0..ranks).all(|source| {
                let at = (row as usize * ranks + source) * SHARD;
                (0..SHARD)
                    .all(|column| actual[at + column] == value(call, source, row as usize, column))
            })
        }))
    };
    let sync = || -> LabResult<()> {
        for session in sessions {
            session.stream.synchronize().map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    set_inputs(0)?;
    kernel_lab::topology::each_rank(ranks, launch_call)?;
    sync()?;
    for owner in 0..ranks {
        if !owned_rows_hold(owner, 0)? {
            return Err(format!("call 0: owner {owner}'s rows differ"));
        }
    }
    for call in 1..3 {
        set_inputs(call)?;
        // Every rank's launch is checked before the first goes out.
        kernel_lab::topology::preflight(|| (0..ranks).try_for_each(launch_call))?;
        for rank in (0..ranks).filter(|rank| !late_ranks.contains(rank)) {
            launch_call(rank)?;
        }
        // The late receiver has not acknowledged call `call - 1`, so no
        // sender may have written its rows yet.
        std::thread::sleep(std::time::Duration::from_millis(50));
        if !owned_rows_hold(late, call - 1)? {
            return Err(format!(
                "call {call}: a sender overwrote rank {late}'s rows before its acknowledgment"
            ));
        }
        for &rank in &late_ranks {
            launch_call(rank)?;
        }
        sync()?;
        for owner in 0..ranks {
            if !owned_rows_hold(owner, call)? {
                return Err(format!("call {call}: owner {owner}'s rows differ"));
            }
        }
    }
    for (rank, session) in sessions.iter().enumerate() {
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "case".into(),
            serde_json::json!("f32_exchange_rows_late_receiver"),
        );
        receipt.insert("ranks".into(), serde_json::json!(ranks));
        receipt.insert("rank".into(), serde_json::json!(rank));
        receipt.insert("late".into(), serde_json::json!(late));
        receipt.insert("late_ranks".into(), serde_json::json!(late_ranks));
        receipt.insert("epochs".into(), serde_json::json!(["0xffffffff", "0", "1"]));
        receipt.insert("correct".into(), serde_json::json!(true));
        session.receipt("comm", receipt)?;
    }
    Ok(())
}

#[test]
fn f32_exchange_rows_wait_for_late_receivers() {
    gpu_test_multi(64 * 1024 * 1024, |sessions| {
        run_exchange_rows_late_receiver(sessions, 1)?;
        run_exchange_rows_late_receiver(sessions, 0)
    })
    .expect("multi-GPU row exchange with a late receiver");
}

#[test]
fn bf16_gather_rows() {
    gpu_test_multi(256 * 1024 * 1024, |sessions| {
        run_gather_rows(sessions, 0)?;
        run_gather_rows(sessions, sessions.len() as u32 - 1)
    })
    .expect("multi-GPU row gather");
}

#[test]
fn f32_allgather_all_ranks() {
    gpu_test_multi(64 * 1024 * 1024, |sessions| {
        run_allgather(sessions, sessions.len() as u32)
    })
    .expect("multi-GPU all-gather");
}

#[test]
fn f32_allgather_one_rank() {
    gpu_test_multi(64 * 1024 * 1024, |sessions| run_allgather(sessions, 0))
        .expect("single-target all-gather");
}

/// Per-rank buffers of one MiMo hidden comm state, sized like the bundle's
/// prefill state: every fused call, whatever its row count, shares them.
struct HiddenState {
    inputs: Vec<DeviceBuffer>,
    output: DeviceBuffer,
    residual: DeviceBuffer,
    weights: DeviceBuffer,
    inbox: DeviceBuffer,
    flags: DeviceBuffer,
    epoch: DeviceBuffer,
}

const STATE_ROWS: usize = 8192;

/// Which entries a fused hidden all-reduce runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Hidden {
    /// The v2 entry up to 32 rows, otherwise the phased reduce-scatter and
    /// fused gather stages.
    Planned,
    /// The phased stages at any row count.
    Phased,
    /// The verify entry, whose rows normalize in the low-latency order (up
    /// to 32 rows); `modules` must be the verify cubin's.
    Exact,
}

/// One fused hidden all-reduce of `rows` rows over the ranks.
fn fused_hidden_call(
    sessions: &[Session],
    modules: &CollectiveModules,
    states: &[HiddenState],
    rows: usize,
    set: usize,
    only_rank: Option<usize>,
) -> LabResult<()> {
    fused_hidden_call_with(
        sessions,
        modules,
        states,
        rows,
        set,
        only_rank,
        Hidden::Planned,
    )
}

fn fused_hidden_call_with(
    sessions: &[Session],
    modules: &CollectiveModules,
    states: &[HiddenState],
    rows: usize,
    set: usize,
    only_rank: Option<usize>,
    mode: Hidden,
) -> LabResult<()> {
    let ranks = sessions.len();
    let elements = (rows * 4096) as u64;
    let query = CommQuery::AllReduce {
        ranks: ranks as u32,
        elements,
        dtype: DType::Bf16,
        fused_residual: true,
        fused_rmsnorm: Some(RmsNormEpilogue {
            kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
            width: 4096,
            eps: 1e-6,
        }),
    };
    let mut plan = kernels_sm120::comm::plan(&query, rows as u32).map_err(|e| e.to_string())?;
    let variant = kernels_sm120::comm::choose(&query).map_err(|e| e.to_string())?;
    if mode == Hidden::Phased && variant != kernels_sm120::comm::Variant::FusedTwoShot {
        // The phased stages at 33 rows give the grid; the kernels take any
        // row count.
        let phased = CommQuery::AllReduce {
            ranks: ranks as u32,
            elements: 33 * 4096,
            dtype: DType::Bf16,
            fused_residual: true,
            fused_rmsnorm: Some(RmsNormEpilogue {
                kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
                width: 4096,
                eps: 1e-6,
            }),
        };
        plan = kernels_sm120::comm::plan(&phased, 33).map_err(|e| e.to_string())?;
    }
    if mode == Hidden::Exact && variant != kernels_sm120::comm::Variant::FusedOneShot {
        return Err(format!("the verify entry does not take {rows} rows"));
    }
    let inboxes: Vec<u64> = (0..ranks as u64).map(|rank| rank + 1).collect();
    let flags: Vec<u64> = (0..ranks as u64).map(|rank| rank + 11).collect();
    let outputs: Vec<u64> = (0..ranks as u64).map(|rank| rank + 21).collect();
    let stages = |rank: usize| -> LabResult<Vec<(&'static str, launch::RankCall<'_>)>> {
        let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
        buffers.insert(100, &states[rank].inputs[set]);
        buffers.insert(101, &states[rank].output);
        buffers.insert(102, &states[rank].epoch);
        buffers.insert(103, &states[rank].residual);
        buffers.insert(104, &states[rank].weights);
        for (peer, state) in states.iter().enumerate() {
            buffers.insert(peer as u64 + 1, &state.inbox);
            buffers.insert(peer as u64 + 11, &state.flags);
            buffers.insert(peer as u64 + 21, &state.output);
        }
        let base = kernels_sm120::comm::OneShotCall {
            input: 100,
            output: 101,
            peer_inboxes: inboxes.clone(),
            peer_flags: flags.clone(),
            epoch: 102,
            elements,
            rank: rank as u32,
            ranks: ranks as u32,
        };
        let stages = match (mode, variant) {
            (Hidden::Phased, _) | (_, kernels_sm120::comm::Variant::FusedTwoShot) => {
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
            (_, kernels_sm120::comm::Variant::FusedOneShot) => vec![(
                if mode == Hidden::Exact {
                    kernels_sm120::comm::VERIFY_SYMBOL
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
            other => return Err(format!("unexpected fused variant {other:?}")),
        };
        Ok(stages
            .into_iter()
            .map(|(symbol, args)| {
                let bindings = Bindings {
                    buffers: buffers.clone(),
                };
                (symbol, launch::RankCall { args, bindings })
            })
            .collect())
    };
    // Each stage runs on every rank before the next.
    let symbols = stages(0)?
        .into_iter()
        .map(|(symbol, _)| symbol)
        .collect::<Vec<_>>();
    for (stage, symbol) in symbols.into_iter().enumerate() {
        launch::launch_collective(sessions, modules, symbol, plan, only_rank, |rank| {
            stages(rank)?
                .into_iter()
                .nth(stage)
                .map(|(_, call)| call)
                .ok_or_else(|| "missing stage".to_owned())
        })?;
    }
    Ok(())
}

/// Host replay of the residual chain: every path adds the residual to the
/// rank-ordered F32 sum and rounds once.
fn host_fused_call(inputs: &[Vec<Vec<u16>>], residual: &mut [u16], rows: usize, set: usize) {
    use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
    for (index, value) in residual[..rows * 4096].iter_mut().enumerate() {
        let sum = inputs
            .iter()
            .fold(0.0_f32, |sum, rank| sum + bf16_to_f32(rank[set][index]));
        *value = f32_to_bf16(sum + bf16_to_f32(*value));
    }
}

/// Poisons every rank's hidden output once all ranks are idle. The residual
/// is the in-place chain and keeps its bytes.
fn poison_hidden(sessions: &[Session], states: &[HiddenState]) -> LabResult<()> {
    poison_ranks(
        sessions,
        states.iter().map(|state| &state.output).enumerate(),
    )
}

/// Checks the residual chain and the first `rows` output rows on every rank
/// after calls on poisoned outputs that wrote rows `[0, written_rows)`: the
/// push, decode and phased entries write each call's rows of every rank's
/// output and nothing past them (README "Fused decode all-reduce" and
/// "Prefill all-reduce").
fn check_hidden(
    sessions: &[Session],
    states: &[HiddenState],
    expected_residual: &[u16],
    weights: &[u16],
    rows: usize,
    written_rows: usize,
    label: &str,
) -> LabResult<()> {
    let elements = rows * 4096;
    let written = WriteMap::of(DType::Bf16).prefix(written_rows * 4096)?;
    for (rank, session) in sessions.iter().enumerate() {
        check_written(
            session,
            &states[rank].output,
            &written,
            &format!("{label} rank {rank} output"),
        )?;
    }
    let expected = NORM
        .apply_bf16(&expected_residual[..elements], weights, 4096)
        .map_err(|e| e.to_string())?;
    let first = sessions[0].download_bf16(&states[0].output, elements)?;
    for (rank, session) in sessions.iter().enumerate() {
        let residual = session.download_bf16(&states[rank].residual, STATE_ROWS * 4096)?;
        oracle::check_exact(&residual, expected_residual)
            .map_err(|index| format!("{label} rank {rank}: residual differs at {index}"))?;
        let output = session.download_bf16(&states[rank].output, elements)?;
        if let Some(index) = output.iter().zip(&first).position(|(a, b)| a != b) {
            return Err(format!(
                "{label} rank {rank}: output differs from rank 0 at {index}"
            ));
        }
        // Summing squares in another order can move the normalized value one
        // BF16 step; a weight above one can turn that into two output steps.
        if let Some(index) = output
            .iter()
            .zip(&expected)
            .position(|(&a, &b)| (i32::from(a) - i32::from(b)).abs() > 2)
        {
            return Err(format!(
                "{label} rank {rank}: output {} vs oracle {} at {index}",
                output[index], expected[index]
            ));
        }
    }
    Ok(())
}

/// Every rank's hidden comm state over `inputs[rank]` sets, with the same
/// residual and norm weights on every rank.
fn hidden_states(
    sessions: &[Session],
    host_inputs: &[Vec<Vec<u16>>],
    host_residual: &[u16],
    host_weights: &[u16],
) -> LabResult<Vec<HiddenState>> {
    let ranks = sessions.len();
    let elements = STATE_ROWS * 4096;
    let flag_bytes = 4 * ranks * STATE_ROWS * 4;
    sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            let inbox_bytes = ranks * elements * 2;
            let inbox = session.allocate(inbox_bytes)?;
            inbox
                .copy_from_bytes(&vec![0x3c; inbox_bytes], &session.stream)
                .map_err(|e| e.to_string())?;
            let push =
                kernels_sm120::comm::push_buffers(ranks as u32, 32).map_err(|e| e.to_string())?;
            inbox
                .zero_at_async(0, push.zeroed_inbox_bytes as usize, &session.stream)
                .map_err(|e| e.to_string())?;
            let flags = session.allocate(flag_bytes)?;
            flags
                .copy_from_bytes(&vec![0; flag_bytes], &session.stream)
                .map_err(|e| e.to_string())?;
            let epoch = session.allocate(STATE_ROWS * 4)?;
            epoch
                .copy_from_bytes(&vec![0; STATE_ROWS * 4], &session.stream)
                .map_err(|e| e.to_string())?;
            let state = HiddenState {
                inputs: host_inputs[rank]
                    .iter()
                    .map(|values| session.upload_bf16(values))
                    .collect::<LabResult<Vec<_>>>()?,
                output: session.allocate(elements * 2)?,
                residual: session.upload_bf16(host_residual)?,
                weights: session.upload_bf16(host_weights)?,
                inbox,
                flags,
                epoch,
            };
            session.stream.synchronize().map_err(|e| e.to_string())?;
            Ok(state)
        })
        .collect::<LabResult<Vec<_>>>()
}

/// Consecutive fused calls with different row counts on one comm state, as a
/// MiMo prefill with chunks of several sizes and interleaved decode-sized
/// calls produces, then graph replays of a mixed sequence.
fn run_prefill_sequence(sessions: &[Session]) -> LabResult<Vec<u16>> {
    use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
    const EAGER: [usize; 8] = [33, 2048, 5, 100, 8192, 1, 33, 2];
    const CAPTURED: [usize; 4] = [100, 33, 2, 2048];
    const REPLAYS: usize = 2;
    let ranks = sessions.len();
    let elements = STATE_ROWS * 4096;
    let host_inputs = (0..ranks)
        .map(|rank| {
            (0..2)
                .map(|set| pattern((rank * 2 + set) as u64 + 50, elements, 0.05))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut host_residual = pattern(98, elements, 1.0);
    let host_weights = pattern(9, 4096, 1.0)
        .into_iter()
        .map(|bits| f32_to_bf16(1.0 + 0.5 * bf16_to_f32(bits)))
        .collect::<Vec<_>>();
    let states = hidden_states(sessions, &host_inputs, &host_residual, &host_weights)?;
    // Module loads synchronize the device, so load before any peer waits.
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    for (call, &rows) in EAGER.iter().enumerate() {
        poison_hidden(sessions, &states)?;
        fused_hidden_call(sessions, &modules, &states, rows, call % 2, None)?;
        for session in sessions {
            session.stream.synchronize().map_err(|e| e.to_string())?;
        }
        host_fused_call(&host_inputs, &mut host_residual, rows, call % 2);
        check_hidden(
            sessions,
            &states,
            &host_residual,
            &host_weights,
            rows,
            rows,
            &format!("eager call {call} ({rows} rows)"),
        )?;
        eprintln!("fused sequence: eager call {call} with {rows} rows passed");
    }
    let graphs = RankReplay::new(sessions, |rank, _| {
        for (call, &rows) in CAPTURED.iter().enumerate() {
            fused_hidden_call(sessions, &modules, &states, rows, call % 2, Some(rank))?;
        }
        Ok(())
    })?;
    // The replays run back to back on outputs poisoned once; the residual
    // chain changes with every call.
    poison_hidden(sessions, &states)?;
    for _ in 0..REPLAYS {
        graphs.launch(sessions)?;
        for (call, &rows) in CAPTURED.iter().enumerate() {
            host_fused_call(&host_inputs, &mut host_residual, rows, call % 2);
        }
    }
    for session in sessions {
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    let last = CAPTURED[CAPTURED.len() - 1];
    let widest = CAPTURED.into_iter().max().unwrap_or(0);
    check_hidden(
        sessions,
        &states,
        &host_residual,
        &host_weights,
        last,
        widest,
        "graph replay",
    )?;
    for (rank, session) in sessions.iter().enumerate() {
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "case".into(),
            serde_json::json!("bf16_fused_prefill_row_count_sequence"),
        );
        receipt.insert("ranks".into(), serde_json::json!(ranks));
        receipt.insert("rank".into(), serde_json::json!(rank));
        receipt.insert("eager_rows".into(), serde_json::json!(EAGER));
        receipt.insert("captured_rows".into(), serde_json::json!(CAPTURED));
        receipt.insert("graph_replays".into(), serde_json::json!(REPLAYS));
        receipt.insert("correct".into(), serde_json::json!(true));
        session.receipt("comm", receipt)?;
    }
    sessions[0].download_bf16(&states[0].output, last * 4096)
}

#[test]
fn bf16_fused_prefill_row_count_sequence() {
    gpu_test_multi(1024 * 1024 * 1024, |sessions| {
        let first = run_prefill_sequence(sessions)?;
        if sessions.len() != 4 {
            let repeated = run_prefill_sequence(sessions)?;
            oracle::check_exact(&repeated, &first)
                .map_err(|index| format!("repeated prefill sequence differs at {index}"))?;
        }
        Ok(())
    })
    .expect("multi-GPU fused prefill sequence");
}

/// The phased prefill all-reduce against the host replay and the verify
/// entry. For identical partials and residual its pre-norm rows must be
/// bitwise those of the host's one-rounding replay (every row count,
/// including the 948/960 and 1,823/1,920 prefill halves), and its first
/// rows (up to 32), pre-norm and normalized, bitwise those the verify entry
/// gives the same rows alone: the phased gather normalizes in the
/// low-latency order.
fn run_phased_ll_order(sessions: &[Session]) -> LabResult<()> {
    use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
    const ROWS: [usize; 13] = [1, 2, 7, 17, 32, 33, 100, 948, 960, 1024, 1823, 1920, 8192];
    let ranks = sessions.len();
    let elements = STATE_ROWS * 4096;
    let host_inputs = (0..ranks)
        .map(|rank| {
            (0..2)
                .map(|set| pattern((rank * 2 + set) as u64 + 70, elements, 0.05))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut host_residual = pattern(97, elements, 1.0);
    let host_weights = pattern(8, 4096, 1.0)
        .into_iter()
        .map(|bits| f32_to_bf16(1.0 + 0.5 * bf16_to_f32(bits)))
        .collect::<Vec<_>>();
    let states = hidden_states(sessions, &host_inputs, &host_residual, &host_weights)?;
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    let exact =
        CollectiveModules::load(sessions, model_mimo26_contract::sm120::comm::VERIFY_CUBIN)?;
    let synchronize = || {
        sessions
            .iter()
            .try_for_each(|session| session.stream.synchronize().map_err(|e| e.to_string()))
    };
    let mut cases = Vec::new();
    for (call, &rows) in ROWS.iter().enumerate() {
        let set = call % 2;
        let before = host_residual.clone();
        poison_hidden(sessions, &states)?;
        fused_hidden_call_with(sessions, &modules, &states, rows, set, None, Hidden::Phased)?;
        synchronize()?;
        host_fused_call(&host_inputs, &mut host_residual, rows, set);
        let label = format!("phased call {call} ({rows} rows)");
        check_hidden(
            sessions,
            &states,
            &host_residual,
            &host_weights,
            rows,
            rows,
            &label,
        )?;
        // The verify entry reruns the first rows alone from the residual's
        // pre-call bytes on a poisoned output.
        let alone = rows.min(32);
        let phased = sessions
            .iter()
            .zip(&states)
            .map(|(session, state)| session.download_bf16(&state.output, alone * 4096))
            .collect::<LabResult<Vec<_>>>()?;
        for (session, state) in sessions.iter().zip(&states) {
            state
                .residual
                .copy_from_bytes(
                    &before
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                    &session.stream,
                )
                .map_err(|e| e.to_string())?;
        }
        poison_hidden(sessions, &states)?;
        fused_hidden_call_with(sessions, &exact, &states, alone, set, None, Hidden::Exact)?;
        synchronize()?;
        let written = WriteMap::of(DType::Bf16).prefix(alone * 4096)?;
        for (rank, session) in sessions.iter().enumerate() {
            check_written(
                session,
                &states[rank].output,
                &written,
                &format!("{label} rank {rank} verify entry output"),
            )?;
            let residual = session.download_bf16(&states[rank].residual, alone * 4096)?;
            oracle::check_exact(&residual, &host_residual[..alone * 4096]).map_err(|index| {
                format!("{label} rank {rank}: verify pre-norm row differs at {index}")
            })?;
            let output = session.download_bf16(&states[rank].output, alone * 4096)?;
            oracle::check_exact(&output, &phased[rank]).map_err(|index| {
                format!("{label} rank {rank}: verify normalized row differs at {index}")
            })?;
        }
        // Restore the phased call's residual for the next call.
        for (session, state) in sessions.iter().zip(&states) {
            state
                .residual
                .copy_from_bytes(
                    &host_residual
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                    &session.stream,
                )
                .map_err(|e| e.to_string())?;
        }
        synchronize()?;
        eprintln!(
            "{label}: pre-norm rows bitwise equal to the one-rounding replay; the first {alone} \
             rows bitwise equal to the verify entry's, pre-norm and normalized"
        );
        cases.push(serde_json::json!({
            "rows": rows,
            "prenorm_equal_host_one_rounding": true,
            "rows_equal_verify_entry_alone": alone,
        }));
    }
    for (rank, session) in sessions.iter().enumerate() {
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "case".into(),
            serde_json::json!("bf16_prefill_phased_matches_verify_rows"),
        );
        receipt.insert("ranks".into(), serde_json::json!(ranks));
        receipt.insert("rank".into(), serde_json::json!(rank));
        receipt.insert("cases".into(), serde_json::json!(cases));
        receipt.insert("correct".into(), serde_json::json!(true));
        session.receipt("comm", receipt)?;
    }
    Ok(())
}

#[test]
fn bf16_prefill_phased_matches_verify_rows() {
    gpu_test_multi(1024 * 1024 * 1024, run_phased_ll_order)
        .expect("phased prefill all-reduce in the low-latency order");
}

#[test]
fn ranks_have_distinct_contexts_and_streams() {
    gpu_test_multi(1024 * 1024, |sessions| {
        for (rank, session) in sessions.iter().enumerate() {
            for (peer, other) in sessions.iter().enumerate().skip(rank + 1) {
                if session.context == other.context {
                    return Err(format!("ranks {rank} and {peer} share a context"));
                }
                let same_stream = session
                    .stream
                    .with_raw_handle(|a| other.stream.with_raw_handle(|b| a == b))
                    .map_err(|e| e.to_string())?
                    .map_err(|e| e.to_string())?;
                if same_stream {
                    return Err(format!("ranks {rank} and {peer} share a stream"));
                }
            }
        }
        Ok(())
    })
    .expect("rank context and stream placement");
}

/// The target F32 logits exchange versus Qwen's gather of BF16 pairs, at
/// the actual vocabulary width. Graphs share state across changing buckets.
#[test]
fn qwen38_f32_owned_rows_match_gather_bits() {
    gpu_test_multi(256 * 1024 * 1024, |sessions| {
        if sessions.len() != 4 {
            return Err("this equality test requires four ranks".into());
        }
        let ranks = sessions.len();
        let shard = 248_320 / ranks;
        let capacity = 32;
        let zeroed = |session: &Session, bytes: usize| -> LabResult<DeviceBuffer> {
            let buffer = session.allocate(bytes)?;
            buffer
                .copy_from_bytes(&vec![0; bytes], &session.stream)
                .map_err(|e| e.to_string())?;
            Ok(buffer)
        };
        let inputs = sessions
            .iter()
            .map(|s| s.allocate(capacity * shard * 4))
            .collect::<LabResult<Vec<_>>>()?;
        let outputs = sessions
            .iter()
            .map(|s| s.allocate(capacity * shard * ranks * 4))
            .collect::<LabResult<Vec<_>>>()?;
        let gathered = sessions
            .iter()
            .map(|s| s.allocate(capacity * shard * ranks * 4))
            .collect::<LabResult<Vec<_>>>()?;
        let flags = sessions
            .iter()
            .map(|s| zeroed(s, kernels_sm120::comm::EXCHANGE_ROWS_FLAG_BYTES as usize))
            .collect::<LabResult<Vec<_>>>()?;
        let epochs = sessions
            .iter()
            .map(|s| zeroed(s, kernels_sm120::comm::EXCHANGE_ROWS_EPOCH_BYTES as usize))
            .collect::<LabResult<Vec<_>>>()?;
        let gather_flags = sessions
            .iter()
            .map(|s| zeroed(s, kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES as usize))
            .collect::<LabResult<Vec<_>>>()?;
        let gather_epochs = sessions
            .iter()
            .map(|s| zeroed(s, kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES as usize))
            .collect::<LabResult<Vec<_>>>()?;
        let modules = CollectiveModules::load(sessions, "comm_sm120")?;
        let sync = || -> LabResult<()> {
            for session in sessions {
                session.stream.synchronize().map_err(|e| e.to_string())?;
            }
            Ok(())
        };
        let call = |rank: usize, rows: u32, owned: bool| -> LabResult<launch::RankCall<'_>> {
            let mut buffers = BTreeMap::from([
                (100, &inputs[rank]),
                (
                    101,
                    if owned {
                        &epochs[rank]
                    } else {
                        &gather_epochs[rank]
                    },
                ),
            ]);
            for peer in 0..ranks {
                buffers.insert(
                    1 + peer as u64,
                    if owned {
                        &outputs[peer]
                    } else {
                        &gathered[peer]
                    },
                );
                buffers.insert(
                    11 + peer as u64,
                    if owned {
                        &flags[peer]
                    } else {
                        &gather_flags[peer]
                    },
                );
            }
            let args = if owned {
                kernels_sm120::comm::exchange_rows_arguments(
                    kernels_sm120::comm::ExchangeRowsCall {
                        input: 100,
                        outputs: (1..=4).collect(),
                        flags: (11..15).collect(),
                        epoch: 101,
                        rows,
                        shard: shard as u64,
                        rank: rank as u32,
                        ranks: 4,
                    },
                )
            } else {
                kernels_sm120::comm::gather_rows_arguments(kernels_sm120::comm::GatherRowsCall {
                    input: 100,
                    outputs: (1..=4).collect(),
                    flags: (11..15).collect(),
                    epoch: 101,
                    row_indices: None,
                    rows,
                    shard: 2 * shard as u64,
                    rank: rank as u32,
                    ranks: 4,
                    target: 0,
                })
            };
            Ok(launch::RankCall {
                args: args.map_err(|e| e.to_string())?,
                bindings: Bindings { buffers },
            })
        };
        let launch_ranks = |only_rank: Option<usize>, rows: u32, owned: bool| {
            let (symbol, plan) = if owned {
                (
                    "chop_comm_exchange_rows_f32_v1",
                    kernels_sm120::comm::exchange_rows_plan(),
                )
            } else {
                (
                    "chop_comm_gather_rows_bf16_v1",
                    kernels_sm120::comm::gather_rows_plan(),
                )
            };
            launch::launch_collective(sessions, &modules, symbol, plan, only_rank, |rank| {
                call(rank, rows, owned)
            })
        };
        // Raw F32 payloads, which the gather moves as BF16 pairs: the gather
        // writes rows [0, rows) of rank 0's `gathered` in full, the exchange
        // each owner's block of rows of its `outputs`; no other bytes
        // (README "Logits gather" and "Logits row exchange").
        let write_maps = |rows: u32, owned: bool| -> LabResult<Vec<WriteMap>> {
            (0..ranks)
                .map(|rank| {
                    let block = if owned {
                        kernels_sm120::comm::exchange_rows_owned(rows, 4, rank as u32)
                    } else if rank == 0 {
                        0..rows
                    } else {
                        0..0
                    };
                    WriteMap::of(DType::F32)
                        .rows(ranks * shard, block.start as usize..block.end as usize)
                })
                .collect()
        };
        sync()?;
        let cases = [1, 2, 4, 5, 8, 16, 17, 32, 1, 32, 8];
        for (case, rows) in cases.into_iter().enumerate() {
            for (rank, session) in sessions.iter().enumerate() {
                let bytes = (0..capacity * shard)
                    .flat_map(|index| {
                        // Include signed zero, infinities, and distinct NaN payloads.
                        let bits = match index % 19 {
                            0 => 0x8000_0000,
                            1 => 0x7f80_0000,
                            2 => 0xff80_0000,
                            3 => 0x7fc0_0000 | (index as u32 & 0xffff),
                            _ => ((index + rank * 101 + case * 13) as f32 / 128.0).to_bits(),
                        };
                        bits.to_ne_bytes()
                    })
                    .collect::<Vec<_>>();
                inputs[rank]
                    .copy_from_bytes(&bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
            }
            sync()?;
            for owned in [false, true] {
                let targets = if owned { &outputs } else { &gathered };
                let maps = write_maps(rows, owned)?;
                let kind = if owned { "exchange" } else { "gather" };
                // Every rank's bytes after a launch on poisoned outputs.
                let written = |phase: &str| -> LabResult<Vec<Vec<u8>>> {
                    sessions
                        .iter()
                        .enumerate()
                        .map(|(rank, session)| {
                            check_written(
                                session,
                                &targets[rank],
                                &maps[rank],
                                &format!("case {case} ({rows} rows) {kind} {phase} rank {rank}"),
                            )
                        })
                        .collect()
                };
                poison_ranks(sessions, targets.iter().enumerate())?;
                launch_ranks(None, rows, owned)?;
                sync()?;
                let eager = written("eager")?;
                let graphs =
                    RankReplay::new(sessions, |rank, _| launch_ranks(Some(rank), rows, owned))?;
                for replay in 0..2 {
                    poison_ranks(sessions, targets.iter().enumerate())?;
                    graphs.launch(sessions)?;
                    sync()?;
                    if written(&format!("replay {replay}"))? != eager {
                        return Err(format!(
                            "case {case} ({rows} rows) {kind} replay {replay} differs from eager"
                        ));
                    }
                }
            }
            let elements = rows as usize * ranks * shard;
            let expected = sessions[0].download_f32(&gathered[0], elements)?;
            for (owner, session) in sessions.iter().enumerate() {
                let actual = session.download_f32(&outputs[owner], elements)?;
                for row in kernels_sm120::comm::exchange_rows_owned(rows, 4, owner as u32) {
                    let start = row as usize * ranks * shard;
                    for column in 0..ranks * shard {
                        if actual[start + column].to_bits() != expected[start + column].to_bits() {
                            return Err(format!(
                                "owner {owner}, row {row}, column {column}: exchanged bits differ"
                            ));
                        }
                    }
                }
            }
        }
        for session in sessions {
            session.receipt(
                "comm",
                BTreeMap::from([
                    (
                        "case".into(),
                        serde_json::json!("qwen38_f32_owned_rows_match_gather_bits"),
                    ),
                    ("cases".into(), serde_json::json!(cases.len())),
                    ("vocabulary".into(), serde_json::json!(248320)),
                    ("correct".into(), serde_json::json!(true)),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("Qwen F32 row exchange equals the gather bit for bit");
}

#[path = "common/invariance.rs"]
mod invariance;
#[path = "comm/invariance.rs"]
mod invariance_adapter;

/// The invariance adapters of this test binary, one per op kind.
fn invariance_adapters() -> Vec<Box<dyn invariance::Adapter>> {
    vec![Box::new(invariance_adapter::Comm)]
}

fn invariance_case(property: Property) {
    gpu_test_multi(256 << 20, |sessions| {
        invariance::run(property, &mut invariance_adapters(), sessions)
    })
    .unwrap_or_else(|error| panic!("{property:?} comm invariance: {error}"));
}

#[test]
fn invariance_comm_chunk() {
    invariance_case(Property::Chunk);
}

#[test]
fn invariance_comm_rows() {
    invariance_case(Property::Rows);
}

#[test]
fn invariance_comm_batch() {
    invariance_case(Property::Batch);
}

/// This binary holds exactly the adapters the shared table places in
/// "comm", and they cover every claim of their op kinds.
#[test]
fn invariance_adapters_match_the_table() {
    let held = invariance_adapters()
        .iter()
        .map(|adapter| adapter.kind())
        .collect::<Vec<_>>();
    let placed = invariance::ADAPTERS
        .iter()
        .filter(|(_, _, binary)| *binary == "comm")
        .map(|(kind, _, _)| *kind)
        .collect::<Vec<_>>();
    assert_eq!(held, placed, "comm adapters against ADAPTERS");
    let uncovered = invariance::uncovered()
        .into_iter()
        .filter(|claim| {
            held.iter()
                .any(|kind| claim.contains(&format!(" {kind:?} ")))
        })
        .collect::<Vec<_>>();
    assert!(
        uncovered.is_empty(),
        "claims without an adapter: {uncovered:?}"
    );
}
