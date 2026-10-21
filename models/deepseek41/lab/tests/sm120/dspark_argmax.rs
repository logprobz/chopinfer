//! The drafter's 1024-thread argmax partial against the 256-thread entry it
//! replaces and a CPU oracle of the contract's order (NaN below every
//! value, -0 as +0, ties to the lower id), at every DSpark rank shard.
use bundle_deepseek41::sm120::{dspark_argmax, fold_cubin_for_symbol};
use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_core::contract::{DType, RowQuery};
use kernel_core::rows::{ArgmaxPairs, ArgmaxPartialBuffers};
use kernel_harness::eviction::EvictionMode;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::{Cubins, LabResult};
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::argmax;
use runtime_cuda::Stream;
use std::collections::BTreeMap;

const SERIAL: &str = "chop_dspark_argmax_f32_v1";
const VOCAB: u32 = 32_320;
const RANKS: u32 = 4;
const ROWS: u32 = 8;
const PATTERNS: [&str; 7] = [
    "random", "tie", "zeros", "nan_inf", "all_nan", "two_inf", "max_last",
];
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];

fn entries() -> [argmax::Entry; 2] {
    [
        argmax::Entry::new(argmax::Pattern::PartialF32, SERIAL),
        dspark_argmax::partial(),
    ]
}

fn view(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}

/// Each arm's launch of `rows` rows of rank `rank`'s shard.
fn launches(rank: u32, rows: u32) -> Vec<Launch> {
    let query = RowQuery::ShardedArgmaxPartial {
        vocab_per_rank: VOCAB,
        ranks: RANKS,
        logits: DType::F32,
    };
    let buffers = ArgmaxPartialBuffers {
        logits: view(1, u64::from(ROWS * VOCAB) * 4),
        live_width: view(2, 4),
        pairs: ArgmaxPairs {
            scores: view(3, u64::from(ROWS) * 4),
            ids: view(4, u64::from(ROWS) * 4),
            peer: false,
        },
        live_rows: view(5, 4),
    };
    entries()
        .map(|entry| {
            let prepared = Sm120::new()
                .with_row_argmax(entry)
                .sharded_argmax_partial(&query, rank, Need::rows(ROWS))
                .unwrap();
            let mut launch = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            assert_eq!(launch.len(), 1);
            launch.remove(0)
        })
        .into_iter()
        .collect()
}

#[test]
fn dspark_argmax_wide_cpu_plans_the_same_launch_at_1024_threads() {
    for rank in 0..RANKS {
        let [serial, wide] = &launches(rank, 5)[..] else {
            unreachable!()
        };
        assert_eq!(serial.symbol, SERIAL);
        assert_eq!(wide.symbol, dspark_argmax::SYMBOL);
        assert_eq!(
            (serial.plan.block, wide.plan.block),
            ([256, 1, 1], [1024, 1, 1])
        );
        assert_eq!(serial.plan.grid, wide.plan.grid);
        assert_eq!(serial.arguments, wide.arguments);
        assert!(wide.plan.pdl);
    }
    for symbol in [SERIAL, dspark_argmax::SYMBOL] {
        assert!(
            fold_cubin_for_symbol(symbol).is_some(),
            "{symbol} is embedded"
        );
    }
    for pattern in PATTERNS {
        let logits = row(pattern, 3);
        let (_, id) = oracle(&logits, VOCAB, 0);
        assert!((id as usize) < logits.len(), "{pattern}");
    }
}

/// The contract's rank of a logit, by its bits.
fn rank(value: f32) -> u32 {
    let mut bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return 0;
    }
    if bits == 0x8000_0000 {
        bits = 0;
    }
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// The winning logit's bits and global id over the first `width` columns.
fn oracle(row: &[f32], width: u32, offset: u32) -> (u32, i32) {
    let width = width.clamp(1, VOCAB) as usize;
    let mut best = (0_u64, 0_usize);
    for (col, &value) in row[..width].iter().enumerate() {
        let key = (u64::from(rank(value)) << 32) | u64::from(u32::MAX - (offset + col as u32));
        if key > best.0 {
            best = (key, col);
        }
    }
    (row[best.1].to_bits(), (offset + best.1 as u32) as i32)
}

/// One row of `pattern`, varied by `seed`.
fn row(pattern: &str, seed: u32) -> Vec<f32> {
    let n = VOCAB as usize;
    let s = seed as usize;
    let nan =
        |i: usize| f32::from_bits(0x7fc0_0000 | (i as u32 & 0x3f_ffff) | ((i as u32 & 1) << 31));
    let mut v = kernel_harness::wide_values(n, 0x5eed_0000 + u64::from(seed));
    match pattern {
        "random" => {}
        "tie" => {
            // The largest value at several columns, the lowest past column 0.
            for at in [n - 1, 7 + s, n / 2, 1000 + s] {
                v[at] = 1.0e6;
            }
        }
        "zeros" => {
            for (i, x) in v.iter_mut().enumerate() {
                *x = if (i + s) % 3 == 0 { -0.0 } else { 0.0 };
            }
        }
        "nan_inf" => {
            for (i, x) in v.iter_mut().enumerate() {
                *x = nan(i + s);
            }
            v[n - 1 - s] = f32::NEG_INFINITY;
        }
        "all_nan" => {
            for (i, x) in v.iter_mut().enumerate() {
                *x = nan(i * 7 + s);
            }
        }
        "two_inf" => {
            v[n - 2] = f32::INFINITY;
            v[200 + s] = f32::INFINITY;
        }
        "max_last" => v[n - 1] = 3.0e7,
        _ => unreachable!("{pattern}"),
    }
    v
}

fn bytes(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

#[test]
fn dspark_argmax_wide_gpu_matches_the_serial_entry_and_oracle() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("the argmax comparison needs an idle GPU").unwrap();
        return;
    }
    let probe = std::env::var_os("CHOPINFER_DSPARK_ARGMAX_PROBE")
        .map(|path| &*Box::leak(std::fs::read(path).unwrap().into_boxed_slice()));
    let cubins = probe.map_or(Cubins::NONE, |probe| {
        Cubins::new(
            Box::leak(Box::new([("probe_sm120", probe)])),
            &[("probe_sm120", PROBE_SYMBOLS)],
        )
    });
    let result = kernel_harness::gpu_test_with(cubins, 256 << 20, |s| {
        let modules = entries()
            .iter()
            .map(|e| s.load_cubin(fold_cubin_for_symbol(&e.symbol).unwrap().1))
            .collect::<LabResult<Vec<_>>>()?;
        let functions = modules
            .iter()
            .zip(entries())
            .map(|(m, e)| m.function(&e.symbol).map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        let logits = s.allocate((ROWS * VOCAB) as usize * 4)?;
        let width = s.allocate(4)?;
        let live = s.allocate(4)?;
        let pairs = (0..2)
            .map(|_| Ok([s.allocate(ROWS as usize * 4)?, s.allocate(ROWS as usize * 4)?]))
            .collect::<LabResult<Vec<_>>>()?;
        let mut cases = 0;
        let mut timings = Vec::new();
        for rank in [0, RANKS - 1] {
            let planned = launches(rank, ROWS);
            let invoke = |arm: usize, stream: &Stream| {
                let bindings = Bindings {
                    buffers: BTreeMap::from([
                        (1, &logits),
                        (2, &width),
                        (3, &pairs[arm][0]),
                        (4, &pairs[arm][1]),
                        (5, &live),
                    ]),
                };
                launch(
                    &s.context,
                    &functions[arm],
                    stream,
                    planned[arm].plan,
                    &planned[arm].arguments,
                    &bindings,
                )
            };
            let graphs = (0..2)
                .map(|arm| s.capture(|stream| invoke(arm, stream)))
                .collect::<LabResult<Vec<_>>>()?;
            for pattern in PATTERNS {
                let rows: Vec<Vec<f32>> = (0..ROWS).map(|r| row(pattern, r)).collect();
                let image: Vec<u8> = rows.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
                logits
                    .copy_from_bytes(&image, &s.stream)
                    .map_err(|e| e.to_string())?;
                for live_width in [VOCAB, 1, 33, VOCAB - 1, VOCAB + 5] {
                    for live_rows in [ROWS, 1, 5, 0] {
                        width
                            .copy_from_bytes(&live_width.to_le_bytes(), &s.stream)
                            .map_err(|e| e.to_string())?;
                        live.copy_from_bytes(&live_rows.to_le_bytes(), &s.stream)
                            .map_err(|e| e.to_string())?;
                        for graph in [false, true] {
                            for pair in &pairs {
                                for buffer in pair {
                                    buffer
                                        .copy_from_bytes(&bytes([0xa5a5_a5a5; ROWS as usize]), &s.stream)
                                        .map_err(|e| e.to_string())?;
                                }
                            }
                            for (arm, captured) in graphs.iter().enumerate().take(2) {
                                if graph {
                                    captured.launch(&s.stream).map_err(|e| e.to_string())?;
                                } else {
                                    invoke(arm, &s.stream)?;
                                }
                            }
                            s.stream.synchronize().map_err(|e| e.to_string())?;
                            let read = |arm: usize| -> LabResult<Vec<u32>> {
                                Ok(s.download_f32(&pairs[arm][0], ROWS as usize)?
                                    .into_iter()
                                    .map(f32::to_bits)
                                    .chain(
                                        s.download_f32(&pairs[arm][1], ROWS as usize)?
                                            .into_iter()
                                            .map(f32::to_bits),
                                    )
                                    .collect())
                            };
                            let (serial, wide) = (read(0)?, read(1)?);
                            let label = format!(
                                "rank={rank} {pattern} width={live_width} live={live_rows} graph={graph}"
                            );
                            if serial != wide {
                                return Err(format!("{label}: wide pairs differ from serial"));
                            }
                            for r in 0..ROWS as usize {
                                let got = (wide[r], wide[ROWS as usize + r]);
                                if r < live_rows as usize {
                                    let (score, id) = oracle(&rows[r], live_width, rank * VOCAB);
                                    if got != (score, id as u32) {
                                        return Err(format!("{label} row {r}: {got:?} != oracle ({score:#x}, {id})"));
                                    }
                                } else if got != (0xa5a5_a5a5, 0xa5a5_a5a5) {
                                    return Err(format!("{label} row {r}: wrote past the live rows"));
                                }
                            }
                        }
                        cases += 1;
                    }
                }
                eprintln!("DSPARK_ARGMAX_BITWISE rank={rank} pattern={pattern} PASS");
            }
            if probe.is_some() {
                let image: Vec<u8> = (0..ROWS)
                    .flat_map(|r| row("random", r))
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                logits
                    .copy_from_bytes(&image, &s.stream)
                    .map_err(|e| e.to_string())?;
                width
                    .copy_from_bytes(&VOCAB.to_le_bytes(), &s.stream)
                    .map_err(|e| e.to_string())?;
                for live_rows in [1, 5, ROWS] {
                    live.copy_from_bytes(&live_rows.to_le_bytes(), &s.stream)
                        .map_err(|e| e.to_string())?;
                    let timing = kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                        graphs[arm].launch(stream).map_err(|e| e.to_string())
                    })?;
                    eprintln!(
                        "DSPARK_ARGMAX_TIMING rank={rank} rows={live_rows} serial_us={} wide_us={} ratio={}",
                        timing.arms[0].median_ms * 1000.,
                        timing.arms[1].median_ms * 1000.,
                        timing.paired_ratio
                    );
                    timings.push(timing.paired_ratio);
                }
            }
        }
        assert_eq!(cases, 2 * PATTERNS.len() * 5 * 4);
        eprintln!("DSPARK_ARGMAX_COMPLETE cases={cases} timings={}", timings.len());
        Ok(())
    })
    .unwrap();
    assert!(result.is_some());
}
