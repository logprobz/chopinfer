//! MiMo fixtures mirrored from concurrency_mimo_comm.rs; kernels are unchanged.
use kernel_core::contract::{CommQuery, DType, RmsNormEpilogue};
use kernel_core::oracle;
use kernel_core::oracle::{NormArithmetic, NormOffset, NormScale, RmsNorm};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session};
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
struct HiddenState {
    inputs: Vec<DeviceBuffer>,
    output: DeviceBuffer,
    residual: DeviceBuffer,
    weights: DeviceBuffer,
    inbox: DeviceBuffer,
    flags: DeviceBuffer,
    epoch: DeviceBuffer,
}

const STATE_ROWS: usize = 256;

/// One fused hidden all-reduce as the bundle issues it: the v2 entry up to 32
/// rows, otherwise the reduce-scatter and fused gather stages.
fn fused_hidden_call(
    sessions: &[Session],
    modules: &[runtime_cuda::Module],
    states: &[HiddenState],
    rows: usize,
    set: usize,
    only_rank: Option<usize>,
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
    let plan = kernels_sm120::comm::plan(&query, rows as u32).map_err(|e| e.to_string())?;
    let variant = kernels_sm120::comm::choose(&query).map_err(|e| e.to_string())?;
    let inboxes: Vec<u64> = (0..ranks as u64).map(|rank| rank + 1).collect();
    let flags: Vec<u64> = (0..ranks as u64).map(|rank| rank + 11).collect();
    let outputs: Vec<u64> = (0..ranks as u64).map(|rank| rank + 21).collect();
    for (rank, session) in sessions.iter().enumerate() {
        if only_rank.is_some_and(|only| only != rank) {
            continue;
        }
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
        let bindings = Bindings { buffers };
        let module = &modules[rank];
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
        let stages = match variant {
            kernels_sm120::comm::Variant::FusedOneShot => vec![(
                "chop_comm_allreduce_oneshot_norm_bf16_v2",
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
        };
        for (symbol, args) in stages {
            let function = module.function(symbol).map_err(|e| e.to_string())?;
            launch::launch(
                &session.context,
                &function,
                &session.stream,
                plan,
                &args,
                &bindings,
            )?;
        }
    }
    Ok(())
}

/// Host replay of the residual chain. The push path and the two-stage
/// prefill path both add the residual to the rank-ordered F32 sum and round
/// once.
fn host_fused_call(inputs: &[Vec<Vec<u16>>], residual: &mut [u16], rows: usize, set: usize) {
    use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
    for (index, value) in residual[..rows * 4096].iter_mut().enumerate() {
        let sum = inputs
            .iter()
            .fold(0.0_f32, |sum, rank| sum + bf16_to_f32(rank[set][index]));
        *value = f32_to_bf16(sum + bf16_to_f32(*value));
    }
}

fn check_hidden(
    sessions: &[Session],
    states: &[HiddenState],
    expected_residual: &[u16],
    weights: &[u16],
    rows: usize,
    label: &str,
) -> LabResult<()> {
    let elements = rows * 4096;
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

fn live_bytes(rows: usize) -> LabResult<usize> {
    if rows == 0 || rows > STATE_ROWS {
        return Err("invalid live hidden rows".into());
    }
    Ok(rows * 4096 * 2)
}

#[test]
fn hidden_live_span_contract_cpu() {
    for rows in [1, 32, 256] {
        assert_eq!(live_bytes(rows).unwrap(), rows * 4096 * 2);
    }
    assert!(live_bytes(0).is_err());
    assert!(live_bytes(257).is_err());
}

pub fn measure(
    sessions: &[Session],
    mut run: impl FnMut(&str, usize, &[runtime_cuda::GraphExec]) -> LabResult<()>,
) -> LabResult<()> {
    let elements = STATE_ROWS * 4096;
    let host_inputs: Vec<_> = (0..4)
        .map(|r| vec![pattern(r + 50, elements, 0.05)])
        .collect();
    let initial = pattern(98, elements, 1.0);
    let weights = vec![kernel_core::codec::f32_to_bf16(1.0); 4096];
    let modules = sessions
        .iter()
        .map(|s| s.module("comm_sm120"))
        .collect::<LabResult<Vec<_>>>()?;
    let mut misses = Vec::new();
    for rows in [1, 32, 256] {
        let checked = (|| -> kernel_lab::LabResult<()> {
            // Push uses per-row epochs, phased uses one epoch. Each case owns fresh
            // protocol state sized by its matching planner.
            let layout = if rows <= 32 {
                kernels_sm120::comm::push_buffers(4, rows as u32)
            } else {
                kernels_sm120::comm::phased_buffers(4, (rows * 4096) as u64, DType::Bf16)
            }
            .map_err(|e| e.to_string())?;
            let states = sessions
                .iter()
                .enumerate()
                .map(|(r, s)| {
                    let zero = |n| -> LabResult<DeviceBuffer> {
                        let b = s.allocate(n)?;
                        b.copy_from_bytes(&vec![0; n], &s.stream)
                            .map_err(|e| e.to_string())?;
                        Ok(b)
                    };
                    Ok(HiddenState {
                        inputs: vec![s.upload_bf16(&host_inputs[r][0])?],
                        output: zero(elements * 2)?,
                        residual: s.upload_bf16(&initial)?,
                        weights: s.upload_bf16(&weights)?,
                        // Use this protocol planner's inbox, flags and epoch sizes.
                        inbox: zero(layout.inbox_bytes as usize)?,
                        flags: zero(layout.flag_bytes.max(4) as usize)?,
                        epoch: zero(layout.epoch_bytes as usize)?,
                    })
                })
                .collect::<LabResult<Vec<_>>>()?;
            // Residual is read-modify-write; restore it before this compared eager call.
            let initial_bytes = initial
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>();
            let mut checks = sessions
                .iter()
                .zip(&states)
                .map(|(s, state)| {
                    Ok(kernel_lab::poison::Replay::new(s)
                        .output(
                            "normed",
                            &state.output,
                            kernel_lab::poison::WriteMap::of(DType::Bf16)
                                .byte_prefix(live_bytes(rows)?)?,
                        )
                        .state(
                            "residual",
                            &state.residual,
                            initial_bytes.clone(),
                            kernel_lab::poison::WriteMap::of(DType::Bf16).none()?,
                            std::iter::once(0..live_bytes(rows)?).collect(),
                        ))
                })
                .collect::<LabResult<Vec<_>>>()?;
            kernel_lab::poison::synchronize(sessions)?;
            for replay in &checks {
                replay.prepare(kernel_lab::poison::Phase::Eager)?;
            }
            fused_hidden_call(sessions, &modules, &states, rows, 0, None)?;
            kernel_lab::poison::synchronize(sessions)?;
            let mut misses = Vec::new();
            for (rank, replay) in checks.iter_mut().enumerate() {
                if let Err(error) = replay.finish(kernel_lab::poison::Phase::Eager, true) {
                    misses.push(format!("rank {rank}: {error}"));
                }
            }
            if !misses.is_empty() {
                return Err(misses.join("; "));
            }
            let mut expected = initial.clone();
            host_fused_call(&host_inputs, &mut expected, rows, 0);
            check_hidden(sessions, &states, &expected, &weights, rows, "KVO eager")?;
            let evidence = checks
                .iter()
                .map(|replay| {
                    let mut fields = BTreeMap::new();
                    replay.record(&mut fields);
                    fields
                })
                .collect::<Vec<_>>();
            sessions[0].receipt(
                "offload-compute-eager",
                BTreeMap::from([
                    ("rows".into(), serde_json::json!(rows)),
                    ("poisoned_ranks".into(), serde_json::json!(evidence)),
                ]),
            )?;
            let graphs = sessions
                .iter()
                .enumerate()
                .map(|(r, s)| {
                    s.capture(|_| {
                        for _ in 0..super::CALLS {
                            fused_hidden_call(sessions, &modules, &states, rows, 0, Some(r))?;
                        }
                        Ok(())
                    })
                })
                .collect::<LabResult<Vec<_>>>()?;
            run(
                if rows <= 32 {
                    "mimo_push"
                } else {
                    "mimo_phased"
                },
                rows,
                &graphs,
            )?;

            Ok(())
        })();
        if let Err(error) = checked {
            eprintln!("rows={rows}: {error}");
            misses.push(error);
        }
    }
    if misses.is_empty() {
        Ok(())
    } else {
        Err(misses.join("; "))
    }
}
