//! The Qwen3.8 hyper-connection all-reduce at the selected rank count: 2, 4
//! or 8 ranks (`CHOPINFER_KERNEL_GPUS`, distinct devices). The
//! test names keep their TP4 prefix from before the entries took a rank
//! count. Replays use one captured graph per rank through `RankReplay`.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, LaunchPlan, SigmoidMeanHyperConnectionQuery};
use kernel_core::hyper_connection_oracle;
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, check_output, gpu_test_multi, launch};
use model_qwen38_contract::sm120::comm as qwen38;
use runtime_cuda::DeviceBuffer;
use serde_json::json;
use std::collections::BTreeMap;

fn codes(len: usize, seed: usize) -> Vec<u16> {
    (0..len)
        .map(|i| f32_to_bf16((((i * 31 + seed * 73) % 509) as f32 - 254.0) / 128.0))
        .collect()
}
fn partial_codes(
    len: usize,
    width: usize,
    ranks: usize,
    rank: usize,
    seed: usize,
    cancellation: bool,
) -> Vec<u16> {
    let mut values = codes(len, seed + rank);
    if cancellation {
        // At four/eight ranks the forward sum loses the odd small value,
        // while reverse and a one-rank rotation retain it. Two finite
        // terms commute, so TP2 checks the F32 result without that control.
        let value = match rank {
            0 => 16777216.0,
            1 => (2 * (seed % 4) + 1) as f32,
            2 if ranks >= 4 => -16777216.0,
            _ => 0.0,
        };
        for column in (0..len).step_by(width) {
            values[column] = f32_to_bf16(value);
        }
    }
    values
}

fn ordered_sums(partials: &[Vec<u16>], elements: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(elements * 4);
    for column in 0..elements {
        let mut value = 0.0_f32;
        for source in partials {
            value += bf16_to_f32(source[column]);
        }
        data.extend_from_slice(&value.to_ne_bytes());
    }
    data
}

fn check_sum_bytes(actual: &[u8], expected: &[u8]) -> LabResult<()> {
    if actual != expected {
        return Err("F32 sums differ from the rank-ordered CPU replay".into());
    }
    Ok(())
}

fn check_sums(
    sessions: &[Session],
    fixtures: &[BTreeMap<u64, DeviceBuffer>],
    offset: usize,
    expected: &[u8],
) -> LabResult<()> {
    for (rank, session) in sessions.iter().enumerate() {
        let data = poison::read(session, &fixtures[rank][&106])?;
        check_sum_bytes(&data[offset..offset + expected.len()], expected)?;
    }
    Ok(())
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

/// Poisons buffers `ids` of every rank's fixture once every rank is idle.
fn poison_ids(
    sessions: &[Session],
    fixtures: &[BTreeMap<u64, DeviceBuffer>],
    ids: &[u64],
) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (s, fixture) in sessions.iter().zip(fixtures) {
        let targets = ids.iter().map(|id| &fixture[id]).collect::<Vec<_>>();
        poison::poison(s, &targets)?;
    }
    Ok(())
}

const CANARY_BYTES: usize = 128;
const CANARY: u8 = 0xa5;

fn guarded_flags(s: &Session, length: usize) -> LabResult<DeviceBuffer> {
    let mut data = vec![0; length + CANARY_BYTES];
    data[length..].fill(CANARY);
    bytes(s, &data)
}

fn check_canaries(
    sessions: &[Session],
    fixtures: &[BTreeMap<u64, DeviceBuffer>],
    length: usize,
) -> LabResult<()> {
    for (rank, session) in sessions.iter().enumerate() {
        for id in [107, 117] {
            let data = poison::read(session, &fixtures[rank][&id])?;
            if data[length..] != [CANARY; CANARY_BYTES] {
                return Err(format!("rank {rank} flags {id} overflowed the canary"));
            }
        }
    }
    Ok(())
}

// Only the F32 sum region is rewritten in full by a phased pair. Never
// poison its BF16 inboxes, flag slots, counter or generation.
fn poison_sums(
    sessions: &[Session],
    fixtures: &[BTreeMap<u64, DeviceBuffer>],
    inbox_bytes: usize,
    sum_bytes: usize,
) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (rank, session) in sessions.iter().enumerate() {
        let region = inbox_bytes - sum_bytes..inbox_bytes;
        poison::fill(
            session,
            &fixtures[rank][&106],
            std::slice::from_ref(&region),
            &[POISON],
        )?;
    }
    Ok(())
}

/// The first `rows` rows of `wide` BF16 values of a poisoned output after
/// its calls: the entries write residual and normed rows below the live
/// count only (qwen38.cuh loops over tokens below `*live_rows`), so every
/// byte past them must still hold poison.
fn live_rows(
    s: &Session,
    buffer: &DeviceBuffer,
    rows: usize,
    wide: usize,
    label: &str,
) -> LabResult<Vec<u16>> {
    let bytes = poison::read(s, buffer)?;
    poison::check(
        label,
        &bytes,
        &WriteMap::of(DType::Bf16).prefix(rows * wide)?,
        Fill::Pattern(&[POISON]),
    )?;
    Ok(bytes[..rows * wide * 2]
        .chunks_exact(2)
        .map(|pair| u16::from_ne_bytes([pair[0], pair[1]]))
        .collect())
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
        ranks: call.inboxes.len() as u32,
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

fn run_cases(benchmark: bool, large_prefill: bool, cancellation: bool) {
    gpu_test_multi(
        if benchmark || large_prefill {
            2_u64 << 30
        } else {
            128 << 20
        },
        |sessions| {
            let ranks = rank_count(sessions)?;
            if benchmark {
                for session in sessions {
                    evidence::exclusive(session)?;
                }
            }
            let capacity = if benchmark || large_prefill {
                8192
            } else {
                513
            };
            let width = 2560;
            let wide = width * 4;
            let query = SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
                ranks: ranks as u32,
                streams: 4,
                width: width as u32,
                eps_bits: 1e-6_f32.to_bits(),
                capacity: capacity as u32,
            };
            let size = qwen38::buffers(query).map_err(|e| e.to_string())?;
            // The one-shot entry at full capacity needs two banks of a
            // capacity-row slot per rank.
            let legacy_inbox =
                qwen38::standalone_inbox_bytes(ranks as u32, capacity as u32) as usize;
            let x = codes(capacity * wide, 1);
            let logits = codes(capacity * 4, 2);
            let weight = codes(wide, 3);
            let partials: Vec<_> = (0..ranks)
                .map(|rank| partial_codes(capacity * width, width, ranks, rank, 4, cancellation))
                .collect();
            let mut fixtures = Vec::new();
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
                // Outputs, poisoned before every launch.
                for id in [104, 105, 114, 115] {
                    b.insert(id, s.allocate(capacity * wide * 2)?);
                }
                b.insert(106, s.allocate(size.inbox_bytes as usize)?);
                b.insert(107, guarded_flags(s, size.flags_bytes as usize)?);
                b.insert(108, bytes(s, &vec![0; size.epoch_bytes as usize])?);
                b.insert(109, bytes(s, &[0; 4])?);
                if benchmark {
                    b.insert(111, s.upload_bf16(&codes(capacity * width, rank + 23))?);
                }
                b.insert(116, s.allocate(legacy_inbox)?);
                b.insert(117, guarded_flags(s, size.flags_bytes as usize)?);
                b.insert(118, bytes(s, &vec![0; size.epoch_bytes as usize])?);
                fixtures.push(b);
            }
            let modules = launch::CollectiveModules::load(sessions, "comm_qwen38")?;
            for symbol in SYMBOLS {
                modules.function(symbol)?;
            }
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
            let set_partials = |seed: usize| -> LabResult<Vec<Vec<u16>>> {
                poison::synchronize(sessions)?;
                let partials = (0..ranks)
                    .map(|rank| {
                        partial_codes(capacity * width, width, ranks, rank, seed, cancellation)
                    })
                    .collect::<Vec<_>>();
                for (rank, session) in sessions.iter().enumerate() {
                    let data = partials[rank]
                        .iter()
                        .flat_map(|value| value.to_ne_bytes())
                        .collect::<Vec<_>>();
                    fixtures[rank][&100]
                        .copy_from_bytes(&data, &session.stream)
                        .map_err(|e| e.to_string())?;
                }
                Ok(partials)
            };
            // Launches one call on every rank, or just rank `only`:
            // the bundle's path for up to `rows` live rows (from buffer
            // `live`), or the one-shot entry on the legacy state.
            let run = |only: Option<usize>,
                       path: Path,
                       live: u64,
                       pdl: bool,
                       input: u64|
             -> LabResult<()> {
                let legacy = matches!(path, Path::Legacy);
                let (inbox, flags) = if legacy { (116, 117) } else { (106, 107) };
                let launches = |rank: usize| -> LabResult<Vec<qwen38::Launch>> {
                    let call = qwen38::Call {
                        input,
                        residual: 101,
                        logits: 102,
                        weight: 103,
                        residual_out: if legacy { 114 } else { 104 },
                        normed: if legacy { 115 } else { 105 },
                        inboxes: (1..=ranks as u64).collect(),
                        flags: (11..11 + ranks as u64).collect(),
                        epoch: if legacy { 118 } else { 108 },
                        live_rows: live,
                        capacity: capacity as u32,
                        rank: rank as u32,
                    };
                    match path {
                        Path::Legacy => standalone(call),
                        Path::Bundle(rows) => qwen38::launches(query, rows, call),
                    }
                    .map_err(|e| e.to_string())
                };
                let bindings = |rank: usize| {
                    let mut buffers: BTreeMap<_, _> =
                        fixtures[rank].iter().map(|(&k, v)| (k, v)).collect();
                    for (peer, fixture) in fixtures.iter().enumerate() {
                        buffers.insert(peer as u64 + 1, &fixture[&inbox]);
                        buffers.insert(peer as u64 + 11, &fixture[&flags]);
                    }
                    launch::Bindings { buffers }
                };
                // Each entry runs on every rank before the next.
                for (entry, first) in launches(only.unwrap_or(0))?.into_iter().enumerate() {
                    let plan = LaunchPlan { pdl, ..first.plan };
                    launch::launch_collective(
                        sessions,
                        &modules,
                        first.symbol,
                        plan,
                        only,
                        |rank| {
                            let launch = launches(rank)?
                                .into_iter()
                                .nth(entry)
                                .ok_or("missing entry")?;
                            Ok(launch::RankCall {
                                args: launch.arguments,
                                bindings: bindings(rank),
                            })
                        },
                    )?;
                }
                Ok(())
            };
            let run_all = |path: Path| -> LabResult<()> {
                let outputs = match path {
                    Path::Bundle(_) => [104, 105],
                    Path::Legacy => [114, 115],
                };
                poison_ids(sessions, &fixtures, &outputs)?;
                if !matches!(path, Path::Legacy) {
                    poison_sums(
                        sessions,
                        &fixtures,
                        size.inbox_bytes as usize,
                        capacity * width * 4,
                    )?;
                }
                run(None, path, 109, false, 100)?;
                for s in sessions {
                    s.stream.synchronize().map_err(|e| e.to_string())?;
                }
                check_canaries(sessions, &fixtures, size.flags_bytes as usize)?;
                Ok(())
            };
            let run = &run;
            let replay = |rows: u32| {
                RankReplay::new(sessions, move |r, _| {
                    run(Some(r), Path::Bundle(rows), 109, true, 100)
                })
            };
            if benchmark {
                use kernel_lab::eviction::{Eviction, EvictionMode};
                let (peak, peak_source) = evidence::peak(&sessions[0], "kernel_push")?;
                let mode = EvictionMode::selected()?;
                for rows in [2048_u32, 8192] {
                    set_live(109, rows)?;
                    let graphs = sessions
                        .iter()
                        .enumerate()
                        .map(|(rank, session)| {
                            session.capture(|_| {
                                run(Some(rank), Path::Bundle(rows), 109, true, 100)?;
                                run(Some(rank), Path::Bundle(rows), 109, true, 111)
                            })
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    let evictions = sessions
                        .iter()
                        .map(|session| Eviction::new(session, mode))
                        .collect::<LabResult<Vec<_>>>()?;
                    // The measured rank's stream gate precedes its events.
                    // Peers evict and enqueue before that gate is released;
                    // their scatter waits for the gated rank. Each rank is
                    // measured separately, including its communication waits.
                    for measured in 0..ranks {
                        let timing = kernel_lab::with_timing_label("prefill_pair", || {
                            kernel_lab::time_prepared(
                                &sessions[measured],
                                2,
                                7,
                                mode,
                                |_| Ok(()),
                                |stream| {
                                    for (rank, session) in sessions.iter().enumerate() {
                                        if rank == measured {
                                            continue;
                                        }
                                        evictions[rank].run(&session.stream)?;
                                        session.stream.synchronize().map_err(|e| e.to_string())?;
                                        graphs[rank]
                                            .launch(&session.stream)
                                            .map_err(|e| e.to_string())?;
                                    }
                                    graphs[measured].launch(stream).map_err(|e| e.to_string())
                                },
                            )
                        })?;
                        poison::synchronize(sessions)?;
                        evidence::receipt(
                            &sessions[measured],
                            "prefill-pair",
                            BTreeMap::from([
                                ("rows".into(), json!(rows)),
                                ("ranks".into(), json!(ranks)),
                                ("measured_rank".into(), json!(measured)),
                                ("peak_source".into(), json!(peak_source)),
                                (
                                    "transfer_floor_ms".into(),
                                    json!(
                                        f64::from(rows) * 2560.0 * 6.0 * (ranks - 1) as f64
                                            / ranks as f64
                                            / peak
                                            * 1000.0
                                    ),
                                ),
                                ("calls_per_graph".into(), json!(2)),
                                ("median_ms".into(), json!(timing.median_ms)),
                                ("min_ms".into(), json!(timing.min_ms)),
                                ("artifact".into(), evidence::identity("comm_qwen38")?),
                            ]),
                        )?;
                    }
                }
                return Ok(());
            }
            // Phased worker buckets (one more cluster of workers per 8
            // rows, 128 workers from 121), uneven owner splits, and
            // one-shot calls in between on the same state.
            let shapes: Vec<usize> = if large_prefill {
                vec![33, 65, 121, 129, 481, 512, 2048, 4097, 8192, 32]
            } else {
                (1..=33)
                    .chain([512, 8, 129, 257, 0, 65, 1, 97, 120, 121, 480, 32, 511, 513])
                    .collect()
            };
            for (case, rows) in shapes.into_iter().enumerate() {
                let partials = set_partials(31 + 17 * case)?;
                set_live(109, rows as u32)?;
                run_all(Path::Bundle(rows as u32))?;
                // Poisoned sums before the call make the gather's delivery
                // observable; the one-shot entry's state is separate.
                if rows > 32 {
                    check_sums(
                        sessions,
                        &fixtures,
                        size.inbox_bytes as usize - capacity * width * 4,
                        &ordered_sums(&partials, rows * width),
                    )?;
                }
                run_all(Path::Legacy)?;
                let mut receipt = BTreeMap::from([
                    ("rows".into(), json!(rows)),
                    ("ranks".into(), json!(ranks)),
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
                // Output `id` of rank `r` holds `live` written rows and
                // poison past them; its live rows.
                let written = |r: usize, id: u64, live: usize, what: &str| {
                    live_rows(
                        &sessions[r],
                        &fixtures[r][&id],
                        live,
                        wide,
                        &format!("rows {rows} rank {r} {what} output {id}"),
                    )
                };
                let before = (0..ranks)
                    .map(|r| {
                        Ok((
                            written(r, 104, rows, "bundle")?,
                            written(r, 105, rows, "bundle")?,
                        ))
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                // Cancellation cases test the transport's F32 rank order,
                // rather than the separate mathematical f64 bound.
                for begin in (0..if cancellation { 0 } else { rows }).step_by(128) {
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
                    // Bit for bit the one-shot entry's live rows; both leave
                    // every later row poisoned.
                    assert_eq!(
                        residual,
                        &written(rank, 114, rows, "one-shot")?,
                        "rank {rank} residual vs one-shot"
                    );
                    assert_eq!(
                        normed,
                        &written(rank, 115, rows, "one-shot")?,
                        "rank {rank} norm vs one-shot"
                    );
                }
                let replays = replay(rows as u32)?;
                for iteration in 0..3 {
                    let partials = set_partials(41 + 17 * case + iteration)?;
                    run_all(Path::Legacy)?;
                    poison_ids(sessions, &fixtures, &[104, 105])?;
                    poison_sums(
                        sessions,
                        &fixtures,
                        size.inbox_bytes as usize,
                        capacity * width * 4,
                    )?;
                    replays.launch(sessions)?;
                    poison::synchronize(sessions)?;
                    check_canaries(sessions, &fixtures, size.flags_bytes as usize)?;
                    if rows > 32 {
                        check_sums(
                            sessions,
                            &fixtures,
                            size.inbox_bytes as usize - capacity * width * 4,
                            &ordered_sums(&partials, rows * width),
                        )?;
                    }
                    for rank in 0..ranks {
                        assert_eq!(
                            written(rank, 104, rows, "graph/PDL")?,
                            written(rank, 114, rows, "one-shot")?,
                            "residual graph/PDL"
                        );
                        assert_eq!(
                            written(rank, 105, rows, "graph/PDL")?,
                            written(rank, 115, rows, "one-shot")?,
                            "norm graph/PDL"
                        );
                    }
                }
                receipt.insert("graph_determinism_pdl_rank_agreement".into(), json!(true));
                receipt.insert("bitwise_one_shot_agreement".into(), json!(true));
                sessions[0].receipt("qwen38-comm-g1", receipt)?;
            }
            Ok(())
        },
    )
    .expect("Qwen HC G1");
}

/// One graph per rank runs a chain of calls with changing row counts on one
/// state, both protocols, without host synchronization between calls; a
/// different rank (call % ranks) starts each call late (a single-CTA flush on
/// its stream),
/// and the skew cubin lags the rank chosen by the generation 20 us at the
/// scatter block wait and the gather wait.
/// Every call writes its own outputs, which must match the one-shot entry
/// at full capacity bit for bit on every rank, across two replays. The
/// pair's generation starts just below 2^32.
fn mixed_sequence() {
    const SEQUENCE: [u32; 14] = [0, 1, 2, 32, 33, 1024, 17, 700, 32, 1024, 5, 129, 64, 1];
    let capacity = 1024_usize;
    gpu_test_multi(1_u64 << 30, |sessions| {
        let ranks = rank_count(sessions)?;
        let width = 2560;
        let wide = width * 4;
        let query = SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
            ranks: ranks as u32,
            streams: 4,
            width: width as u32,
            eps_bits: 1e-6_f32.to_bits(),
            capacity: capacity as u32,
        };
        let size = qwen38::buffers(query).map_err(|e| e.to_string())?;
        let x = codes(capacity * wide, 11);
        let logits = codes(capacity * 4, 12);
        let weight = codes(wide, 13);
        // Ids: 101..103 inputs; 106/107/108 state; 116/117/118 legacy state;
        // 114/115 legacy outputs; 200 + 4 i live rows, 201 + 4 i and
        // 202 + 4 i outputs and 203 + 4 i partials of call i; 300 flush scratch.
        let mut fixtures = Vec::new();
        let mut probes = Vec::new();
        for (rank, s) in sessions.iter().enumerate() {
            let mut b = BTreeMap::new();
            for (id, values) in [(101, &x), (102, &logits), (103, &weight)] {
                b.insert(id, s.upload_bf16(values)?);
            }
            b.insert(106, s.allocate(size.inbox_bytes as usize)?);
            b.insert(107, guarded_flags(s, size.flags_bytes as usize)?);
            let mut epoch = vec![0_u8; size.epoch_bytes as usize];
            epoch[8..16].copy_from_slice(&((1_u64 << 32) - 3).to_ne_bytes());
            b.insert(108, bytes(s, &epoch)?);
            b.insert(
                116,
                s.allocate(qwen38::standalone_inbox_bytes(ranks as u32, capacity as u32) as usize)?,
            );
            b.insert(117, guarded_flags(s, size.flags_bytes as usize)?);
            b.insert(118, bytes(s, &vec![0; size.epoch_bytes as usize])?);
            // Outputs, poisoned before every launch.
            for id in [114, 115] {
                b.insert(id, s.allocate(capacity * wide * 2)?);
            }
            for (i, rows) in SEQUENCE.iter().enumerate() {
                let id = 200 + 4 * i as u64;
                b.insert(id, bytes(s, &rows.to_ne_bytes())?);
                b.insert(id + 1, s.allocate(capacity * wide * 2)?);
                b.insert(id + 2, s.allocate(capacity * wide * 2)?);
                b.insert(
                    id + 3,
                    s.upload_bf16(&partial_codes(
                        capacity * width,
                        width,
                        ranks,
                        rank,
                        14 + i * ranks,
                        true,
                    ))?,
                );
            }
            b.insert(300, s.allocate(16 << 20)?);
            b.insert(301, bytes(s, &vec![POISON; capacity * width * 4])?);
            fixtures.push(b);
            probes.push(s.module("probe_sm120")?);
        }
        let comm = launch::CollectiveModules::load(sessions, "comm_qwen38_skew")?;
        for s in sessions {
            s.stream.synchronize().map_err(|e| e.to_string())?;
        }
        // The skew cubin's entries lag 20 us at the phased waits of the rank
        // chosen by the generation. Every function is resolved before any
        // rank submits, so no function load runs while a peer waits.
        let skewed = |name: &str| name.replacen("chop_comm_qwen38_", "chop_comm_qwen38_skew_", 1);
        for name in SYMBOLS {
            comm.function(&skewed(name))?;
        }
        let flushes = probes
            .iter()
            .map(|probe| {
                probe
                    .function("chop_probe_flush_v1")
                    .map_err(|e| e.to_string())
            })
            .collect::<LabResult<Vec<_>>>()?;
        // Call i on every rank, or just rank `only`: the bundle's path
        // into its own outputs, or (legacy) the one-shot entry
        // at full capacity on the legacy state.
        let call = |only: Option<usize>, i: usize, legacy: bool, pdl: bool| -> LabResult<()> {
            let rows = SEQUENCE[i];
            let id = 200 + 4 * i as u64;
            let (inbox, flags) = if legacy { (116, 117) } else { (106, 107) };
            let launches = |rank: usize| -> LabResult<Vec<qwen38::Launch>> {
                let c = qwen38::Call {
                    input: id + 3,
                    residual: 101,
                    logits: 102,
                    weight: 103,
                    residual_out: if legacy { 114 } else { id + 1 },
                    normed: if legacy { 115 } else { id + 2 },
                    inboxes: (1..=ranks as u64).collect(),
                    flags: (11..11 + ranks as u64).collect(),
                    epoch: if legacy { 118 } else { 108 },
                    live_rows: id,
                    capacity: capacity as u32,
                    rank: rank as u32,
                };
                if legacy {
                    standalone(c)
                } else {
                    qwen38::launches(query, rows, c)
                }
                .map_err(|e| e.to_string())
            };
            let bindings = |rank: usize| {
                let mut buffers: BTreeMap<_, _> =
                    fixtures[rank].iter().map(|(&k, v)| (k, v)).collect();
                for (peer, fixture) in fixtures.iter().enumerate() {
                    buffers.insert(peer as u64 + 1, &fixture[&inbox]);
                    buffers.insert(peer as u64 + 11, &fixture[&flags]);
                }
                launch::Bindings { buffers }
            };
            // Each entry runs on every rank before the next.
            for (entry, first) in launches(only.unwrap_or(0))?.into_iter().enumerate() {
                let plan = LaunchPlan { pdl, ..first.plan };
                launch::launch_collective(
                    sessions,
                    &comm,
                    &skewed(first.symbol),
                    plan,
                    only,
                    |rank| {
                        let launch = launches(rank)?
                            .into_iter()
                            .nth(entry)
                            .ok_or("missing entry")?;
                        Ok(launch::RankCall {
                            args: launch.arguments,
                            bindings: bindings(rank),
                        })
                    },
                )?;
            }
            Ok(())
        };
        let delay = |rank: usize| -> LabResult<()> {
            launch::launch_with(
                &flushes[rank],
                &sessions[rank].stream,
                launch::simple_config((1, 1, 1), (256, 1, 1), 0, (1, 1, 1), false)?,
                &[
                    runtime_cuda::KernelArgument::Buffer(&fixtures[rank][&300]),
                    runtime_cuda::KernelArgument::U64(4 << 20),
                ],
            )
        };
        // Expected live rows: the one-shot entry at full capacity, eagerly,
        // on poisoned outputs whose rows past the live ones stay poison.
        let mut expected = Vec::new();
        for (i, rows) in SEQUENCE.iter().enumerate() {
            poison_ids(sessions, &fixtures, &[114, 115])?;
            call(None, i, true, false)?;
            for s in sessions {
                s.stream.synchronize().map_err(|e| e.to_string())?;
            }
            let mut outputs = Vec::new();
            for (rank, s) in sessions.iter().enumerate() {
                let label =
                    |id: u64| format!("call {i} ({rows} rows) rank {rank} one-shot output {id}");
                outputs.push((
                    live_rows(s, &fixtures[rank][&114], *rows as usize, wide, &label(114))?,
                    live_rows(s, &fixtures[rank][&115], *rows as usize, wide, &label(115))?,
                ));
            }
            expected.push(outputs.swap_remove(0));
        }
        // The delayed rank's flush goes before its call.
        let replays = RankReplay::new(sessions, |rank, _| {
            for (i, &rows) in SEQUENCE.iter().enumerate() {
                if rank == i % ranks {
                    delay(i % ranks)?;
                }
                // The previous local gather is complete. A peer cannot
                // send the next sums before receiving this rank's scatter,
                // which follows this bounded copy on the local stream.
                if rows > 32 {
                    fixtures[rank][&106]
                        .copy_range_from_async(
                            size.inbox_bytes as usize - capacity * width * 4,
                            &fixtures[rank][&301],
                            0,
                            capacity * width * 4,
                            &sessions[rank].stream,
                        )
                        .map_err(|e| e.to_string())?;
                }
                call(Some(rank), i, false, true)?;
            }
            Ok(())
        })?;
        // Every call's own outputs, poisoned before each replay; the calls of
        // a replay run without host synchronization.
        let outputs = (0..SEQUENCE.len() as u64)
            .flat_map(|i| [201 + 4 * i, 202 + 4 * i])
            .collect::<Vec<_>>();
        for replay in 0..2 {
            poison_ids(sessions, &fixtures, &outputs)?;
            poison_sums(
                sessions,
                &fixtures,
                size.inbox_bytes as usize,
                capacity * width * 4,
            )?;
            replays.launch(sessions)?;
            for s in sessions {
                s.stream.synchronize().map_err(|e| e.to_string())?;
            }
            check_canaries(sessions, &fixtures, size.flags_bytes as usize)?;
            for (i, (residual, normed)) in expected.iter().enumerate() {
                let id = 200 + 4 * i as u64;
                for (rank, s) in sessions.iter().enumerate() {
                    let context = format!(
                        "replay {replay} call {i} ({} rows) rank {rank}",
                        SEQUENCE[i]
                    );
                    // Live rows match; rows past them stay poison.
                    for (offset, want) in [(1, residual), (2, normed)] {
                        let got = live_rows(
                            s,
                            &fixtures[rank][&(id + offset)],
                            SEQUENCE[i] as usize,
                            wide,
                            &format!("{context} output {offset}"),
                        )?;
                        assert_eq!(&got, want, "{context} output {offset}");
                    }
                }
            }
        }
        let mut generation = [0_u8; 16];
        fixtures[0][&108]
            .copy_to_bytes(&mut generation, &sessions[0].stream)
            .map_err(|e| e.to_string())?;
        let phased = SEQUENCE
            .iter()
            .filter(|&&rows| qwen38::phased(rows))
            .count() as u64;
        assert_eq!(
            u64::from_ne_bytes(generation[8..16].try_into().unwrap()),
            (1_u64 << 32) - 3 + 2 * phased,
            "one generation per phased call"
        );
        let receipt = BTreeMap::from([
            ("sequence".into(), json!(SEQUENCE)),
            ("capacity".into(), json!(capacity)),
            ("artifact".into(), evidence::identity("comm_qwen38")?),
            ("ranks".into(), json!(ranks)),
            ("delayed_rank_per_call".into(), json!("call % ranks")),
            ("skew_ns_at_phased_waits".into(), json!(20000)),
            ("generation_start".into(), json!((1_u64 << 32) - 3)),
            (
                "bitwise_one_shot_agreement_every_call_every_rank".into(),
                json!(true),
            ),
        ]);
        sessions[0].receipt("qwen38-comm-mixed", receipt)?;
        Ok(())
    })
    .expect("Qwen HC mixed sequence");
}

/// The selected rank count, which the entries must support.
fn rank_count(sessions: &[Session]) -> LabResult<usize> {
    let ranks = sessions.len();
    if !qwen38::RANKS.contains(&(ranks as u32)) {
        return Err(format!("Qwen comm takes 2, 4 or 8 ranks, not {ranks}"));
    }
    Ok(ranks)
}

#[path = "common/evidence.rs"]
mod evidence;

#[test]
fn tp4_hyper_connection_g1() {
    run_cases(false, false, false);
}

#[test]
#[ignore = "requires the parent's exclusive KERNEL window"]
fn tp4_hyper_connection_g4() {
    run_cases(true, false, false);
}

#[test]
fn tp4_prefill_g1() {
    run_cases(false, true, false);
}

#[test]
fn tp4_mixed_rows_delayed_rank_g1() {
    mixed_sequence();
}

/// CPU replay: deliberately reversed/rotated F32 sums fail the exact same
/// byte comparison used by the GPU gather checks.
#[test]
fn rank_order_negative_control() {
    for ranks in [4, 8] {
        let mut partials = (0..ranks)
            .map(|rank| partial_codes(1, 1, ranks, rank, 0, true))
            .collect::<Vec<_>>();
        let expected = ordered_sums(&partials, 1);
        assert_eq!(expected, 0.0_f32.to_ne_bytes());
        check_sum_bytes(&expected, &expected).unwrap();
        partials.reverse();
        let reversed = ordered_sums(&partials, 1);
        assert_eq!(reversed, 1.0_f32.to_ne_bytes());
        assert!(check_sum_bytes(&reversed, &expected).is_err());
        partials.reverse();
        partials.rotate_left(1);
        assert!(check_sum_bytes(&ordered_sums(&partials, 1), &expected).is_err());
    }
}

#[test]
fn tp4_rank_order_g1() {
    run_cases(false, false, true);
}
