//! The one-row Markov step against the per-row step it replaces and a CPU
//! emulation of their shared chain (128 fused products in K order from +0,
//! chains folded from +0, then base), at every DSpark vocabulary shard.
use bundle_deepseek41::sm120::{fold_cubin_for_symbol, markov_row};
use kernel_core::backend::{Counts, Launch, Need, View};
use kernel_harness::eviction::EvictionMode;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::{Cubins, LabResult};
use model_deepseek41_contract::geometry::FLASH;
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::dspark::{MarkovStep, MarkovStepBuffers};
use runtime_cuda::Stream;
use std::collections::BTreeMap;

const PER_ROW: &str = "chop_deepseek41_dspark_markov_v1";
const PATTERNS: [&str; 4] = ["random", "edges", "zeros", "large"];
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_gate_release_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];

fn view(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}

/// The per-row step's launch of one row, and the row step's.
fn launches(vocab: u32, aliased: bool) -> [Launch; 2] {
    let rank = FLASH.shape.dspark.markov_rank;
    let op = MarkovStep {
        rank,
        vocab_rows: vocab,
        logits_stride: vocab,
    };
    let prepared = crate::own_ops::Sm120Own::new(&FLASH)
        .dspark_markov_step(&op, Need::rows(8))
        .unwrap();
    let logits = u64::from(vocab) * 4;
    let b = MarkovStepBuffers {
        embedding: view(1, u64::from(rank) * 2),
        weight: view(2, u64::from(vocab) * u64::from(rank) * 2),
        base: view(3, logits),
        output: view(if aliased { 3 } else { 4 }, logits),
        live_rows: view(5, 4),
    };
    let row = prepared
        .launches(&b, View::EMPTY, Counts::rows(1))
        .unwrap()
        .remove(0);
    assert_eq!(row.symbol, markov_row::SYMBOL);
    // The per-row step at one row, as served before the row step.
    let per_row = Launch {
        symbol: PER_ROW.into(),
        plan: kernel_core::contract::LaunchPlan {
            grid: [vocab.div_ceil(128), 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            ..row.plan
        },
        ..row.clone()
    };
    [per_row, row]
}

fn bf16(value: f32) -> u16 {
    kernel_core::codec::f32_to_bf16(value)
}

fn widen(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// The shared chain of one logit.
fn oracle(embedding: &[u16], weight: &[u16], base: f32) -> u32 {
    let mut sum = 0.0_f32;
    for chain in embedding.chunks(128).zip(weight.chunks(128)) {
        let mut partial = 0.0_f32;
        for (&e, &w) in chain.0.iter().zip(chain.1) {
            partial = widen(e).mul_add(widen(w), partial);
        }
        sum += partial;
    }
    (base + sum).to_bits()
}

fn values(pattern: &str, count: usize, seed: u64) -> Vec<u16> {
    let random = kernel_harness::wide_values(count, seed);
    random
        .iter()
        .enumerate()
        .map(|(i, &v)| match pattern {
            "random" => bf16(v * 0.05),
            "edges" => match i % 11 {
                0 => 0x8000,
                1 => 0x7f80,
                2 => 0xff80,
                3 => 0x7fc1,
                4 => 0x0001,
                _ => bf16(v * 0.05),
            },
            "zeros" => {
                if i % 2 == 0 {
                    0x8000
                } else {
                    0
                }
            }
            "large" => bf16(v * 3.0e4),
            _ => unreachable!(),
        })
        .collect()
}

#[test]
fn dspark_markov_row_cpu_plans_the_row_step_for_one_row() {
    for tp in [1, 2, 4] {
        let vocab = FLASH.shape.vocab / tp;
        for aliased in [false, true] {
            let [per_row, row] = launches(vocab, aliased);
            assert_eq!(row.plan.grid, [vocab.div_ceil(64), 1, 1]);
            assert_eq!(row.plan.block, [256, 1, 1]);
            assert!(row.plan.pdl);
            assert_eq!(per_row.arguments, row.arguments);
        }
    }
    for symbol in [PER_ROW, markov_row::SYMBOL] {
        assert!(
            fold_cubin_for_symbol(symbol).is_some(),
            "{symbol} is embedded"
        );
    }
    let e = values("random", 256, 1);
    let w = values("random", 256, 2);
    assert_eq!(oracle(&e, &w, 0.0), oracle(&e, &w, 0.0));
}

#[test]
fn dspark_markov_row_gpu_matches_the_per_row_step_and_oracle() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("the Markov row comparison needs an idle GPU").unwrap();
        return;
    }
    let probe = std::env::var_os("CHOPINFER_DSPARK_MARKOV_PROBE")
        .map(|path| &*Box::leak(std::fs::read(path).unwrap().into_boxed_slice()));
    let cubins = probe.map_or(Cubins::NONE, |probe| {
        Cubins::new(
            Box::leak(Box::new([("probe_sm120", probe)])),
            &[("probe_sm120", PROBE_SYMBOLS)],
        )
    });
    let rank = FLASH.shape.dspark.markov_rank as usize;
    let result = kernel_harness::gpu_test_with(cubins, 1 << 30, |s| {
        let modules = [PER_ROW, markov_row::SYMBOL]
            .iter()
            .map(|symbol| s.load_cubin(fold_cubin_for_symbol(symbol).unwrap().1))
            .collect::<LabResult<Vec<_>>>()?;
        let functions = modules
            .iter()
            .zip([PER_ROW, markov_row::SYMBOL])
            .map(|(m, symbol)| m.function(symbol).map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        let mut cases = 0;
        let mut timings = 0;
        for tp in [4, 2, 1] {
            let vocab = (FLASH.shape.vocab / tp) as usize;
            let embedding = s.allocate(rank * 2)?;
            let weight = s.allocate(vocab * rank * 2)?;
            let base = s.allocate(vocab * 4)?;
            let live = s.allocate(4)?;
            let outputs = [s.allocate(vocab * 4)?, s.allocate(vocab * 4)?];
            let bases = [s.allocate(vocab * 4)?, s.allocate(vocab * 4)?];
            for aliased in [false, true] {
                let planned = launches(vocab as u32, aliased);
                let invoke = |arm: usize, stream: &Stream| {
                    let bindings = Bindings {
                        buffers: BTreeMap::from([
                            (1, &embedding),
                            (2, &weight),
                            (3, if aliased { &bases[arm] } else { &base }),
                            (4, &outputs[arm]),
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
                for (p, pattern) in PATTERNS.iter().enumerate() {
                    let seed = (tp * 16 + p as u32) as u64;
                    let e = values(pattern, rank, seed);
                    let w = values(pattern, vocab * rank, seed + 1000);
                    let b: Vec<f32> = kernel_harness::wide_values(vocab, seed + 2000);
                    let words = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
                    let floats = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
                    embedding.copy_from_bytes(&words(&e), &s.stream).map_err(|e| e.to_string())?;
                    weight.copy_from_bytes(&words(&w), &s.stream).map_err(|e| e.to_string())?;
                    base.copy_from_bytes(&floats(&b), &s.stream).map_err(|e| e.to_string())?;
                    for live_rows in [1_u32, 0, 7] {
                        live.copy_from_bytes(&live_rows.to_le_bytes(), &s.stream)
                            .map_err(|e| e.to_string())?;
                        for graph in [false, true] {
                            let poison = vec![0xa5_u8; vocab * 4];
                            for arm in 0..2 {
                                outputs[arm].copy_from_bytes(&poison, &s.stream).map_err(|e| e.to_string())?;
                                bases[arm].copy_from_bytes(&floats(&b), &s.stream).map_err(|e| e.to_string())?;
                            }
                            for (arm, exec) in graphs.iter().enumerate() {
                                if graph {
                                    exec.launch(&s.stream).map_err(|e| e.to_string())?;
                                } else {
                                    invoke(arm, &s.stream)?;
                                }
                            }
                            s.stream.synchronize().map_err(|e| e.to_string())?;
                            let read = |arm: usize| -> LabResult<Vec<u32>> {
                                let target = if aliased { &bases[arm] } else { &outputs[arm] };
                                Ok(s.download_f32(target, vocab)?.into_iter().map(f32::to_bits).collect())
                            };
                            let (old, new) = (read(0)?, read(1)?);
                            let label = format!("tp={tp} {pattern} aliased={aliased} live={live_rows} graph={graph}");
                            if old != new {
                                let at = old.iter().zip(&new).position(|(a, b)| a != b).unwrap();
                                return Err(format!("{label}: logit {at} {:#x} != per-row {:#x}", new[at], old[at]));
                            }
                            for col in (0..vocab).step_by(97).chain([vocab - 1]) {
                                let expected = if live_rows == 0 {
                                    if aliased { b[col].to_bits() } else { 0xa5a5_a5a5 }
                                } else {
                                    oracle(&e, &w[col * rank..][..rank], b[col])
                                };
                                let got = new[col];
                                let nan = f32::from_bits(expected).is_nan() && f32::from_bits(got).is_nan();
                                if got != expected && !nan {
                                    return Err(format!("{label}: logit {col} {got:#x} != oracle {expected:#x}"));
                                }
                            }
                            cases += 1;
                        }
                    }
                    eprintln!("DSPARK_MARKOV_ROW_BITWISE tp={tp} pattern={pattern} aliased={aliased} PASS");
                }
                if probe.is_some() && !aliased {
                    live.copy_from_bytes(&1_u32.to_le_bytes(), &s.stream).map_err(|e| e.to_string())?;
                    let timing = kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                        graphs[arm].launch(stream).map_err(|e| e.to_string())
                    })?;
                    eprintln!(
                        "DSPARK_MARKOV_ROW_TIMING tp={tp} per_row_us={:.3} row_us={:.3} ratio={:.3}",
                        timing.arms[0].median_ms * 1000.,
                        timing.arms[1].median_ms * 1000.,
                        timing.paired_ratio
                    );
                    timings += 1;
                }
            }
        }
        assert_eq!(cases, 3 * 2 * PATTERNS.len() * 3 * 2);
        eprintln!("DSPARK_MARKOV_ROW_COMPLETE cases={cases} timings={timings}");
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "the comparison ran");
}
