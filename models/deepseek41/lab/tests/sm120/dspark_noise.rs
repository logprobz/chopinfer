//! The drafter's served coupled noise (`sm120::dspark_noise`) on the GPU:
//! greedy rows, rows past the live count and columns past the live width
//! keep their bits; sampled rows stay within the contract's bound of the
//! f64 oracle (`kernel_core::draft_noise`); and with the served wide argmax
//! partial and the sharded final, every rank's shard of a full vocabulary
//! proposes the CPU coupled proposal, the host sampler's own sample for
//! equal logits, unless the two lie within the bound of a tie. With the
//! probe cubin it also times a draft step's five selections with and
//! without the noise.
use bundle_deepseek41::sm120::{dspark_argmax, dspark_noise, fold_cubin_for_symbol};
use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_core::contract::{DType, RowQuery};
use kernel_core::draft_noise::{self, DraftNoise, DraftNoiseBuffers, NOISE_BYTES, NoiseRecord};
use kernel_core::rows::{ArgmaxPairs, ArgmaxPartialBuffers};
use kernel_harness::eviction::EvictionMode;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::{Cubins, LabResult};
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::draft_noise::SYMBOL;
use runtime_cuda::Stream;
use std::collections::BTreeMap;

const VOCAB: u32 = 32_320;
const RANKS: u32 = 4;
const ROWS: u32 = 8;
/// The most drafting items a timed step holds.
const TIMED_ROWS: u32 = 32;
const PATTERNS: [&str; 4] = ["random", "flat", "peaked", "tie"];
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];

fn view(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}

fn op() -> DraftNoise {
    DraftNoise {
        vocab_per_rank: VOCAB,
        ranks: RANKS,
    }
}

/// The noise launch of `rows` rows of rank `rank`'s shard at `position`:
/// logits 1, width 2, noise 3, live rows 4.
fn noise_launch(rank: u32, rows: u32, position: u32) -> Launch {
    let prepared = Sm120::new()
        .with_draft_noise(dspark_noise::entry())
        .draft_noise(&op(), rank, Need::rows(TIMED_ROWS))
        .unwrap();
    let buffers = DraftNoiseBuffers {
        logits: view(1, u64::from(TIMED_ROWS * VOCAB) * 4),
        live_width: view(2, 4),
        noise: view(3, u64::from(TIMED_ROWS) * NOISE_BYTES as u64),
        live_rows: view(4, 4),
        position,
    };
    let mut launches = prepared
        .launches(&buffers, View::EMPTY, Counts::rows(rows))
        .unwrap();
    assert_eq!(launches.len(), 1);
    launches.remove(0)
}

/// The served wide argmax partial of `rows` rows of rank `rank`'s shard:
/// logits 1, width 2, live rows 4, scores 5, ids 6.
fn argmax_launch(rank: u32, rows: u32) -> Launch {
    let query = RowQuery::ShardedArgmaxPartial {
        vocab_per_rank: VOCAB,
        ranks: RANKS,
        logits: DType::F32,
    };
    let buffers = ArgmaxPartialBuffers {
        logits: view(1, u64::from(TIMED_ROWS * VOCAB) * 4),
        live_width: view(2, 4),
        pairs: ArgmaxPairs {
            scores: view(5, u64::from(TIMED_ROWS) * 4),
            ids: view(6, u64::from(TIMED_ROWS) * 4),
            peer: false,
        },
        live_rows: view(4, 4),
    };
    let mut launches = Sm120::new()
        .with_row_argmax(dspark_argmax::partial())
        .sharded_argmax_partial(&query, rank, Need::rows(TIMED_ROWS))
        .unwrap()
        .launches(&buffers, View::EMPTY, Counts::rows(rows))
        .unwrap();
    assert_eq!(launches.len(), 1);
    launches.remove(0)
}

/// Row `r`'s noise: every fourth row greedy, the others sampled at three
/// temperatures with their own seed and generated index.
fn record(r: u32) -> NoiseRecord {
    if r % 4 == 3 {
        return NoiseRecord::GREEDY;
    }
    NoiseRecord {
        seed: 0x5eed_0000_0000_1000 + u64::from(r),
        generated: 17 * u64::from(r) + 3,
        temperature: [0.6, 1.0, 1.7][r as usize % 3],
    }
}

fn sampled(r: u32) -> NoiseRecord {
    NoiseRecord {
        temperature: 1.0,
        ..record(r % 3)
    }
}

/// One row of `pattern` over `columns`, varied by `seed`.
fn row(pattern: &str, seed: u32, columns: usize) -> Vec<f32> {
    let mut v = kernel_harness::wide_values(columns, 0xd5a7_0000 + u64::from(seed));
    match pattern {
        "random" => {
            for x in &mut v {
                // Logits of a spread distribution, a few nats apart.
                *x = (*x).clamp(-8.0, 8.0);
            }
        }
        "flat" => v.fill(0.0),
        "peaked" => {
            v.iter_mut().for_each(|x| *x = x.clamp(-4.0, 4.0));
            v[(seed as usize * 7919) % columns] = 40.0;
        }
        "tie" => {
            v.iter_mut().for_each(|x| *x = x.clamp(-4.0, 0.0));
            for at in [5, columns / 2, columns - 1] {
                v[at] = 6.0;
            }
        }
        _ => unreachable!("{pattern}"),
    }
    v
}

fn records(rows: u32, of: impl Fn(u32) -> NoiseRecord) -> Vec<u8> {
    (0..rows).flat_map(|r| of(r).encode()).collect()
}

#[test]
fn dspark_noise_cpu_plans_the_served_entry_beside_the_wide_argmax() {
    for rank in 0..RANKS {
        let noise = noise_launch(rank, 5, 3);
        let argmax = argmax_launch(rank, 5);
        assert_eq!(noise.symbol, SYMBOL);
        assert_eq!(noise.plan.grid, [5, VOCAB.div_ceil(2048), 1]);
        assert_eq!(argmax.symbol, dspark_argmax::SYMBOL);
        assert!(noise.plan.pdl && argmax.plan.pdl);
        assert_eq!(
            noise.arguments.0[4..],
            [
                kernel_core::contract::Argument::U32(VOCAB),
                kernel_core::contract::Argument::U32(rank * VOCAB),
                kernel_core::contract::Argument::U32(3),
            ]
        );
    }
    let (cubin, _) = fold_cubin_for_symbol(SYMBOL).expect("the noise entry is embedded");
    assert_eq!(cubin, dspark_noise::CUBIN);
    assert!(fold_cubin_for_symbol(dspark_argmax::SYMBOL).is_some());
    let full = row("random", 1, 64);
    assert!(draft_noise::proposal(&full, &record(0), 2).unwrap() < 64);
}

#[test]
fn dspark_noise_gpu_keeps_greedy_rows_and_couples_sampled_proposals() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("the draft noise checks need an idle GPU").unwrap();
        return;
    }
    let probe = std::env::var_os("CHOPINFER_DSPARK_NOISE_PROBE")
        .map(|path| &*Box::leak(std::fs::read(path).unwrap().into_boxed_slice()));
    let cubins = probe.map_or(Cubins::NONE, |probe| {
        Cubins::new(
            Box::leak(Box::new([("probe_sm120", probe)])),
            &[("probe_sm120", PROBE_SYMBOLS)],
        )
    });
    let result = kernel_harness::gpu_test_with(cubins, 512 << 20, |s| {
        let load = |symbol: &str| -> LabResult<runtime_cuda::Function> {
            let module = s.load_cubin(fold_cubin_for_symbol(symbol).ok_or("unembedded")?.1)?;
            module.function(symbol).map_err(|e| e.to_string())
        };
        let noise_fn = load(SYMBOL)?;
        let argmax_fn = load(dspark_argmax::SYMBOL)?;
        let logits = s.allocate((TIMED_ROWS * VOCAB) as usize * 4)?;
        let width = s.allocate(4)?;
        let noise = s.allocate(TIMED_ROWS as usize * NOISE_BYTES)?;
        let live = s.allocate(4)?;
        let scores = s.allocate(TIMED_ROWS as usize * 4)?;
        let ids = s.allocate(TIMED_ROWS as usize * 4)?;
        let bindings = Bindings {
            buffers: BTreeMap::from([
                (1, &logits),
                (2, &width),
                (3, &noise),
                (4, &live),
                (5, &scores),
                (6, &ids),
            ]),
        };
        let run = |planned: &Launch, function: &runtime_cuda::Function, stream: &Stream| {
            launch(
                &s.context,
                function,
                stream,
                planned.plan,
                &planned.arguments,
                &bindings,
            )
        };
        let upload = |buffer: &runtime_cuda::DeviceBuffer, bytes: &[u8]| {
            buffer
                .copy_from_bytes(bytes, &s.stream)
                .map_err(|e| e.to_string())
        };
        upload(&noise, &records(ROWS, record))?;
        // 1. Every row, column and record against the oracle and the bits.
        let mut cases = 0;
        let mut worst = 0.0_f64;
        for rank in [0, RANKS - 1] {
            for position in [0, 4] {
                let planned = noise_launch(rank, ROWS, position);
                let graph = s.capture(|stream| run(&planned, &noise_fn, stream))?;
                for pattern in PATTERNS {
                    let rows: Vec<Vec<f32>> = (0..ROWS)
                        .map(|r| row(pattern, r + 10 * rank, VOCAB as usize))
                        .collect();
                    let image: Vec<u8> = rows
                        .iter()
                        .flatten()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    for live_width in [VOCAB, 33, VOCAB + 5] {
                        for live_rows in [ROWS, 5, 0] {
                            for captured in [false, true] {
                                upload(&logits, &image)?;
                                upload(&width, &live_width.to_le_bytes())?;
                                upload(&live, &live_rows.to_le_bytes())?;
                                if captured {
                                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                                } else {
                                    run(&planned, &noise_fn, &s.stream)?;
                                }
                                s.stream.synchronize().map_err(|e| e.to_string())?;
                                let got = s.download_f32(&logits, (ROWS * VOCAB) as usize)?;
                                let label = format!(
                                    "rank={rank} position={position} {pattern} \
                                     width={live_width} live={live_rows} graph={captured}"
                                );
                                let columns = live_width.clamp(1, VOCAB) as usize;
                                for (r, input) in rows.iter().enumerate() {
                                    let out = &got[r * VOCAB as usize..(r + 1) * VOCAB as usize];
                                    let noise = record(r as u32);
                                    let kept = r >= live_rows as usize || !noise.sampled();
                                    let changed_from = if kept { 0 } else { columns };
                                    if out[changed_from..]
                                        .iter()
                                        .zip(&input[changed_from..])
                                        .any(|(a, b)| a.to_bits() != b.to_bits())
                                    {
                                        return Err(format!("{label} row {r}: kept bits changed"));
                                    }
                                    if kept {
                                        continue;
                                    }
                                    let reference = draft_noise::reference(
                                        &op(),
                                        rank,
                                        &input[..columns],
                                        &noise,
                                        position,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    for ((&y, &x), &want) in out.iter().zip(input).zip(&reference) {
                                        let bound = draft_noise::bound(x, &noise, want);
                                        let error = (f64::from(y) - want).abs();
                                        if !(error <= bound) {
                                            return Err(format!(
                                                "{label} row {r}: {y} against {want} \
                                                 exceeds {bound}"
                                            ));
                                        }
                                        worst = worst.max(error / bound);
                                    }
                                }
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 2 * 2 * PATTERNS.len() * 3 * 3 * 2);
        eprintln!("DSPARK_NOISE_BOUND cases={cases} worst_bound_use={worst:.6} PASS");
        // 2. Full-vocabulary proposals through the served argmax and final.
        let mut proposals = 0;
        let mut exact = 0;
        let mut near = 0;
        upload(&width, &VOCAB.to_le_bytes())?;
        upload(&live, &ROWS.to_le_bytes())?;
        for pattern in PATTERNS {
            let full: Vec<Vec<f32>> = (0..ROWS)
                .map(|r| row(pattern, 100 + r, (VOCAB * RANKS) as usize))
                .collect();
            for position in 0..5 {
                let mut pairs = vec![Vec::new(); ROWS as usize];
                for rank in 0..RANKS {
                    let shard: Vec<u8> = full
                        .iter()
                        .flat_map(|r| &r[(rank * VOCAB) as usize..((rank + 1) * VOCAB) as usize])
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    upload(&logits, &shard)?;
                    run(&noise_launch(rank, ROWS, position), &noise_fn, &s.stream)?;
                    run(&argmax_launch(rank, ROWS), &argmax_fn, &s.stream)?;
                    s.stream.synchronize().map_err(|e| e.to_string())?;
                    let score = s.download_f32(&scores, ROWS as usize)?;
                    let id = s.download_f32(&ids, ROWS as usize)?;
                    for r in 0..ROWS as usize {
                        pairs[r].push((score[r], id[r].to_bits() as usize));
                    }
                }
                for (r, input) in full.iter().enumerate() {
                    let noise = record(r as u32);
                    let device = kernel_core::oracle::sharded_argmax_final(&pairs[r])
                        .map_err(|e| e.to_string())? as u32;
                    let expected = draft_noise::proposal(input, &noise, position)
                        .map_err(|e| e.to_string())?;
                    proposals += 1;
                    if device == expected {
                        exact += 1;
                        continue;
                    }
                    let whole = DraftNoise {
                        vocab_per_rank: VOCAB * RANKS,
                        ranks: 1,
                    };
                    let values = draft_noise::reference(&whole, 0, input, &noise, position)
                        .map_err(|e| e.to_string())?;
                    let (a, b) = (values[device as usize], values[expected as usize]);
                    let bound = draft_noise::bound(input[device as usize], &noise, a)
                        + draft_noise::bound(input[expected as usize], &noise, b);
                    if (a - b).abs() > bound {
                        return Err(format!(
                            "{pattern} position {position} row {r}: \
                             device {device} ({a}) against {expected} ({b})"
                        ));
                    }
                    near += 1;
                }
            }
        }
        assert_eq!(proposals, PATTERNS.len() * 5 * ROWS as usize);
        eprintln!(
            "DSPARK_NOISE_PROPOSALS proposals={proposals} exact={exact} near_ties={near} PASS"
        );
        // 3. A draft step's five selections, with and without the noise.
        let mut timings = 0;
        if probe.is_some() {
            upload(&logits, &vec![0; (TIMED_ROWS * VOCAB) as usize * 4])?;
            let greedy = |_: u32| NoiseRecord::GREEDY;
            for (label, of) in [
                ("sampled", sampled as fn(u32) -> NoiseRecord),
                ("greedy", greedy),
            ] {
                upload(&noise, &records(TIMED_ROWS, of))?;
                for rows in [1, 4, 8, 16, TIMED_ROWS] {
                    upload(&live, &rows.to_le_bytes())?;
                    let rank = 1;
                    let plain = s.capture(|stream| {
                        for _ in 0..5 {
                            run(&argmax_launch(rank, rows), &argmax_fn, stream)?;
                        }
                        Ok(())
                    })?;
                    let coupled = s.capture(|stream| {
                        for position in 0..5 {
                            run(&noise_launch(rank, rows, position), &noise_fn, stream)?;
                            run(&argmax_launch(rank, rows), &argmax_fn, stream)?;
                        }
                        Ok(())
                    })?;
                    let graphs = [plain, coupled];
                    let timing = kernel_harness::time_ab(
                        s,
                        3,
                        15,
                        [EvictionMode::Read; 2],
                        |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
                    )?;
                    let a = timing.arms[0].median_ms * 1000.;
                    let b = timing.arms[1].median_ms * 1000.;
                    eprintln!(
                        "DSPARK_NOISE_TIMING records={label} rows={rows} argmax_step_us={a:.2} \
                         coupled_step_us={b:.2} added_step_us={:.2} ratio={:.4}",
                        b - a,
                        timing.paired_ratio
                    );
                    timings += 1;
                }
            }
        }
        eprintln!("DSPARK_NOISE_COMPLETE cases={cases} proposals={proposals} timings={timings}");
        Ok(())
    })
    .unwrap();
    assert!(result.is_some());
}
