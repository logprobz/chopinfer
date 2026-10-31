//! The staged Markov step (the `dspark_markov_staged` candidate) against
//! the tiled step it replaces at two or more rows, and a CPU emulation of
//! their shared chain (128 fused products in K order from +0, chains
//! folded from +0, then base), at every DSpark vocabulary shard. The
//! staged step reads each item's base in place from the item-major block
//! logits, five rows apart, where the tiled step reads the contiguous copy
//! the row gather made; an aliased case runs both in place on contiguous
//! rows, as the packed heads do.
use bundle_deepseek41::sm120::{fold_cubin_for_symbol, markov_staged, markov_tiled};
use kernel_core::backend::{Counts, Launch, Need, View};
use kernel_harness::eviction::EvictionMode;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::{Cubins, LabResult};
use model_deepseek41_contract::geometry::FLASH;
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::dspark::{MarkovStep, MarkovStepBuffers};
use runtime_cuda::Stream;
use std::collections::BTreeMap;

const BLOCK: usize = 5;
const PATTERNS: [&str; 4] = ["random", "edges", "zeros", "large"];
/// Drafting items of the drafter's forwards: two to twelve, and the most
/// a forward prepares.
const ROWS: [u32; 7] = [2, 3, 5, 8, 9, 12, 32];
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_gate_release_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];

fn view(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}

fn step(vocab: u32) -> MarkovStep {
    MarkovStep {
        rank: FLASH.shape.dspark.markov_rank,
        vocab_rows: vocab,
        logits_stride: vocab,
    }
}

/// The tiled step's launch of `rows` rows over a contiguous base, and the
/// staged step's over the block logits (or, `aliased`, over the same
/// contiguous rows in place).
fn launches(vocab: u32, rows: u32, aliased: bool) -> [Launch; 2] {
    let op = step(vocab);
    let rank = u64::from(op.rank);
    let row = u64::from(vocab) * 4;
    let buffers = |base: View, output: View| MarkovStepBuffers {
        embedding: view(1, u64::from(rows) * rank * 2),
        weight: view(2, u64::from(vocab) * rank * 2),
        base,
        output,
        live_rows: view(5, 4),
    };
    let contiguous = view(3, u64::from(rows) * row);
    let output = if aliased {
        contiguous
    } else {
        view(4, u64::from(rows) * row)
    };
    let tiled = crate::own_ops::Sm120Own::new(&FLASH)
        .dspark_markov_step(&op, Need::rows(32))
        .unwrap()
        .launches(
            &buffers(contiguous, output),
            View::EMPTY,
            Counts::rows(rows),
        )
        .unwrap()
        .remove(0);
    assert_eq!(tiled.symbol, markov_tiled::SYMBOL);
    let (stride, base) = if aliased {
        (vocab, contiguous)
    } else {
        let block = (u64::from(rows - 1) * BLOCK as u64 + 1) * row;
        (BLOCK as u32 * vocab, view(6, block))
    };
    let staged = crate::dspark::markov_staged(&FLASH, &op, stride, Need::rows(32))
        .unwrap()
        .launches(&buffers(base, output), View::EMPTY, Counts::rows(rows))
        .unwrap()
        .remove(0);
    assert_eq!(staged.symbol, markov_staged::SYMBOL);
    [tiled, staged]
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
fn dspark_markov_staged_cpu_plans_two_or_more_rows_and_refuses_other_shapes() {
    for tp in [1, 2, 4] {
        let vocab = FLASH.shape.vocab / tp;
        for rows in ROWS {
            for aliased in [false, true] {
                let [tiled, staged] = launches(vocab, rows, aliased);
                assert_eq!(staged.plan.grid, [vocab.div_ceil(64), 1, 1]);
                assert_eq!(staged.plan.block, [128, 1, 1]);
                assert_eq!(staged.plan.dynamic_shared_bytes, 0);
                assert!(staged.plan.pdl && tiled.plan.pdl);
                assert_eq!(staged.rows, rows);
            }
        }
    }
    let op = step(FLASH.shape.vocab / 4);
    // One row takes the row step, as the served step does.
    let one = crate::dspark::markov_staged(&FLASH, &op, 5 * op.vocab_rows, Need::rows(32))
        .unwrap()
        .launches(
            &MarkovStepBuffers {
                embedding: view(1, 512),
                weight: view(2, u64::from(op.vocab_rows) * 512),
                base: view(3, u64::from(op.vocab_rows) * 4),
                output: view(4, u64::from(op.vocab_rows) * 4),
                live_rows: view(5, 4),
            },
            View::EMPTY,
            Counts::rows(1),
        )
        .unwrap();
    assert_eq!(one[0].symbol, bundle_deepseek41::sm120::markov_row::SYMBOL);
    // A base stride below the vocabulary and another rank are refused.
    assert!(crate::dspark::markov_staged(&FLASH, &op, op.vocab_rows - 1, Need::rows(32)).is_err());
    assert!(
        crate::dspark::markov_staged(
            &FLASH,
            &MarkovStep { rank: 128, ..op },
            op.vocab_rows,
            Need::rows(32)
        )
        .is_err()
    );
    for symbol in [markov_tiled::SYMBOL, markov_staged::SYMBOL] {
        assert!(
            fold_cubin_for_symbol(symbol).is_some(),
            "{symbol} is embedded"
        );
    }
}

/// The weights are the only reads before the dependency wait, each a
/// 16-byte (or, unaligned, 2-byte) read-only load; every other read and
/// every write follows it.
#[test]
fn dspark_markov_staged_cpu_reads_only_weights_before_the_wait() {
    let path = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("markov-staged.cubin");
    std::fs::write(
        &path,
        fold_cubin_for_symbol(markov_staged::SYMBOL).unwrap().1,
    )
    .unwrap();
    let output = std::process::Command::new(
        kernel_build::Toolchain::detect()
            .unwrap()
            .home
            .join("bin/cuobjdump"),
    )
    .arg("--dump-sass")
    .arg(&path)
    .output()
    .unwrap();
    assert!(output.status.success());
    let sass = String::from_utf8(output.stdout).unwrap();
    let body = sass
        .split(&format!("Function : {}", markov_staged::SYMBOL))
        .nth(1)
        .expect("staged Markov symbol");
    let wait = body.find("ACQBULK").expect("dependency wait");
    let (before, after) = body.split_at(wait);
    assert!(before.contains("PREEXIT"));
    let loads: Vec<_> = before.lines().filter(|l| l.contains("LDG.")).collect();
    assert!(!loads.is_empty());
    for line in loads {
        assert!(line.contains(".CONSTANT"), "early non-weight load {line}");
    }
    assert!(!before.contains("STG.") && !before.contains("STS"));
    assert!(after.contains("FFMA") && after.contains("FADD") && after.contains("BAR.SYNC"));
    for forbidden in ["TF32", "HMMA", "HGMMA", "UTMMA", ".FTZ"] {
        assert!(!body.contains(forbidden), "unexpected {forbidden}");
    }
}

#[test]
fn dspark_markov_staged_gpu_matches_the_tiled_step_and_oracle() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("the staged Markov comparison needs an idle GPU").unwrap();
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
        let modules = [markov_tiled::SYMBOL, markov_staged::SYMBOL]
            .iter()
            .map(|symbol| s.load_cubin(fold_cubin_for_symbol(symbol).unwrap().1))
            .collect::<LabResult<Vec<_>>>()?;
        let functions = modules
            .iter()
            .zip([markov_tiled::SYMBOL, markov_staged::SYMBOL])
            .map(|(m, symbol)| m.function(symbol).map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        let most = *ROWS.iter().max().unwrap() as usize;
        let mut cases = 0;
        let mut timings = 0;
        for tp in [4, 2, 1] {
            let vocab = (FLASH.shape.vocab / tp) as usize;
            let embedding = s.allocate(most * rank * 2)?;
            let weight = s.allocate(vocab * rank * 2)?;
            // The block logits from the selected position's first row on.
            let block = s.allocate(most * BLOCK * vocab * 4)?;
            let gathered = s.allocate(most * vocab * 4)?;
            let live = s.allocate(4)?;
            let outputs = [s.allocate(most * vocab * 4)?, s.allocate(most * vocab * 4)?];
            let in_place = [s.allocate(most * vocab * 4)?, s.allocate(most * vocab * 4)?];
            for (p, pattern) in PATTERNS.iter().enumerate() {
                let seed = u64::from(tp * 16) + p as u64;
                let e = values(pattern, most * rank, seed);
                let w = values(pattern, vocab * rank, seed + 1000);
                let b: Vec<f32> = kernel_harness::wide_values(most * BLOCK * vocab, seed + 2000);
                let words = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
                let floats = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
                embedding
                    .copy_from_bytes(&words(&e), &s.stream)
                    .map_err(|e| e.to_string())?;
                weight
                    .copy_from_bytes(&words(&w), &s.stream)
                    .map_err(|e| e.to_string())?;
                for position in [0, BLOCK - 1] {
                    // The row gather's copy: item r's row 5 r + position.
                    let copied: Vec<f32> = (0..most)
                        .flat_map(|r| b[(r * BLOCK + position) * vocab..][..vocab].to_vec())
                        .collect();
                    let shifted = &b[position * vocab..];
                    block
                        .copy_from_bytes(&floats(shifted), &s.stream)
                        .map_err(|e| e.to_string())?;
                    for rows in ROWS {
                        for aliased in [false, true] {
                            let planned = launches(vocab as u32, rows, aliased);
                            let invoke = |arm: usize, stream: &Stream| {
                                let (base, output) = match (aliased, arm) {
                                    (true, _) => (&in_place[arm], &in_place[arm]),
                                    (false, 0) => (&gathered, &outputs[0]),
                                    (false, _) => (&block, &outputs[1]),
                                };
                                let bindings = Bindings {
                                    buffers: BTreeMap::from([
                                        (1, &embedding),
                                        (2, &weight),
                                        (3, base),
                                        (4, output),
                                        (5, &live),
                                        (6, base),
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
                            for live_rows in [rows, rows - 1, 0] {
                                live.copy_from_bytes(&live_rows.to_le_bytes(), &s.stream)
                                    .map_err(|e| e.to_string())?;
                                for graph in [false, true] {
                                    let poison = vec![0xa5_u8; most * vocab * 4];
                                    for arm in 0..2 {
                                        outputs[arm]
                                            .copy_from_bytes(&poison, &s.stream)
                                            .map_err(|e| e.to_string())?;
                                        in_place[arm]
                                            .copy_from_bytes(&floats(&copied), &s.stream)
                                            .map_err(|e| e.to_string())?;
                                    }
                                    gathered
                                        .copy_from_bytes(&floats(&copied), &s.stream)
                                        .map_err(|e| e.to_string())?;
                                    for (arm, exec) in graphs.iter().enumerate() {
                                        if graph {
                                            exec.launch(&s.stream).map_err(|e| e.to_string())?;
                                        } else {
                                            invoke(arm, &s.stream)?;
                                        }
                                    }
                                    s.stream.synchronize().map_err(|e| e.to_string())?;
                                    let read = |arm: usize| -> LabResult<Vec<u32>> {
                                        let target = if aliased {
                                            &in_place[arm]
                                        } else {
                                            &outputs[arm]
                                        };
                                        Ok(s.download_f32(target, most * vocab)?
                                            .into_iter()
                                            .map(f32::to_bits)
                                            .collect())
                                    };
                                    let (old, new) = (read(0)?, read(1)?);
                                    let label = format!(
                                        "tp={tp} {pattern} position={position} rows={rows} \
                                         aliased={aliased} live={live_rows} graph={graph}"
                                    );
                                    if old != new {
                                        let at =
                                            old.iter().zip(&new).position(|(a, b)| a != b).unwrap();
                                        return Err(format!(
                                            "{label}: element {at} {:#x} != tiled {:#x}",
                                            new[at], old[at]
                                        ));
                                    }
                                    // The block logits stay untouched.
                                    let after: Vec<u32> = s
                                        .download_f32(&block, shifted.len())?
                                        .into_iter()
                                        .map(f32::to_bits)
                                        .collect();
                                    if after.iter().zip(shifted).any(|(a, b)| *a != b.to_bits()) {
                                        return Err(format!("{label}: the block logits changed"));
                                    }
                                    for r in 0..most {
                                        for col in (r * 13..vocab).step_by(211).chain([vocab - 1]) {
                                            let got = new[r * vocab + col];
                                            let expected = if (r as u32) < live_rows {
                                                oracle(
                                                    &e[r * rank..][..rank],
                                                    &w[col * rank..][..rank],
                                                    copied[r * vocab + col],
                                                )
                                            } else if aliased {
                                                copied[r * vocab + col].to_bits()
                                            } else {
                                                0xa5a5_a5a5
                                            };
                                            let nan = f32::from_bits(expected).is_nan()
                                                && f32::from_bits(got).is_nan();
                                            if got != expected && !nan {
                                                return Err(format!(
                                                    "{label}: row {r} logit {col} {got:#x} != \
                                                     oracle {expected:#x}"
                                                ));
                                            }
                                        }
                                    }
                                    cases += 1;
                                }
                            }
                            if probe.is_some() && !aliased && position == 0 && p == 0 {
                                live.copy_from_bytes(&rows.to_le_bytes(), &s.stream)
                                    .map_err(|e| e.to_string())?;
                                let timing = kernel_harness::time_ab(
                                    s,
                                    3,
                                    15,
                                    [EvictionMode::Read; 2],
                                    |arm, stream| {
                                        graphs[arm].launch(stream).map_err(|e| e.to_string())
                                    },
                                )?;
                                eprintln!(
                                    "DSPARK_MARKOV_STAGED_TIMING tp={tp} rows={rows} \
                                     tiled_us={:.3} staged_us={:.3} ratio={:.3}",
                                    timing.arms[0].median_ms * 1000.,
                                    timing.arms[1].median_ms * 1000.,
                                    timing.paired_ratio
                                );
                                timings += 1;
                            }
                        }
                    }
                }
                eprintln!("DSPARK_MARKOV_STAGED_BITWISE tp={tp} pattern={pattern} PASS");
            }
        }
        assert_eq!(cases, 3 * PATTERNS.len() * 2 * ROWS.len() * 2 * 3 * 2);
        eprintln!("DSPARK_MARKOV_STAGED_COMPLETE cases={cases} timings={timings}");
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "the comparison ran");
}
