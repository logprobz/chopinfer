//! The staged confidence head (the `dspark_confidence_staged` candidate)
//! against the cooperative entry it replaces and a CPU emulation of their
//! shared chain (128 fused products in K order from +0, chains folded from
//! +0), with poisoned outputs, graphs and a PDL producer that holds the
//! hidden rows, embeddings and live count until after the consumer starts.
use super::{b16, b32, bf16, chain_f32, probe, upload};
use bundle_deepseek41::sm120::{confidence_parallel, confidence_staged, fold_cubin_for_symbol};
use kernel_core::backend::{Counts, Launch, Need, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_harness::eviction::EvictionMode;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::poison::{LiveProducer, Replay, Rewrites, WriteMap};
use kernel_harness::{Cubins, LabResult, Session};
use model_deepseek41_contract::geometry::FLASH;
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::dspark::{ConfidenceHead, ConfidenceHeadBuffers};
use runtime_cuda::Stream;
use std::collections::BTreeMap;

const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_gate_release_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
/// Drafting items of the drafter's forwards, and the most a forward prepares.
const ROWS: [u32; 6] = [1, 2, 5, 8, 12, 32];

fn view(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}

/// The served entry's launch and the staged entry's, over the same views.
fn launches(op: ConfidenceHead, rows: u32) -> [Launch; 2] {
    let buffers = ConfidenceHeadBuffers {
        hidden: view(
            1,
            (u64::from(rows - 1) * u64::from(op.hidden_stride) + u64::from(op.hidden)) * 4,
        ),
        markov: view(2, u64::from(rows * op.markov_rank) * 2),
        weight: view(3, u64::from(op.hidden + op.markov_rank) * 2),
        scores: view(
            4,
            (u64::from(rows - 1) * u64::from(op.score_stride) + 1) * 4,
        ),
        live_rows: view(5, 4),
    };
    let served = crate::own_ops::Sm120Own::new(&FLASH)
        .dspark_confidence(&op, Need::rows(32))
        .unwrap()
        .launches(&buffers, View::EMPTY, Counts::rows(rows))
        .unwrap()
        .remove(0);
    assert_eq!(served.symbol, confidence_parallel::SYMBOL);
    let staged = crate::dspark::confidence_staged(&FLASH, &op, Need::rows(32))
        .unwrap()
        .launches(&buffers, View::EMPTY, Counts::rows(rows))
        .unwrap()
        .remove(0);
    assert_eq!(staged.symbol, confidence_staged::SYMBOL);
    [served, staged]
}

fn served_op(hidden_stride: u32) -> ConfidenceHead {
    ConfidenceHead {
        hidden: FLASH.shape.hidden,
        markov_rank: FLASH.shape.dspark.markov_rank,
        hidden_stride,
        score_stride: 5,
    }
}

#[test]
fn confidence_staged_cpu_plans_every_row_count_and_refuses_partial_chains() {
    for stride in [5 * FLASH.shape.hidden, 5 * FLASH.shape.hidden + 3] {
        for rows in ROWS {
            let [served, staged] = launches(served_op(stride), rows);
            assert_eq!(staged.plan.grid, [rows, 1, 1]);
            assert_eq!(staged.plan.block, [confidence_staged::THREADS, 1, 1]);
            assert_eq!(staged.plan.dynamic_shared_bytes, 0);
            assert!(staged.plan.pdl && served.plan.pdl);
            assert_eq!(staged.arguments, served.arguments);
        }
    }
    let op = served_op(5 * FLASH.shape.hidden);
    // Another geometry's widths and a partial last chain are refused.
    for refused in [
        ConfidenceHead {
            hidden: FLASH.shape.hidden - 1,
            ..op
        },
        ConfidenceHead {
            markov_rank: 128,
            ..op
        },
    ] {
        assert!(crate::dspark::confidence_staged(&FLASH, &refused, Need::rows(32)).is_err());
    }
    for symbol in [confidence_parallel::SYMBOL, confidence_staged::SYMBOL] {
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
fn confidence_staged_cpu_reads_only_weights_before_the_wait() {
    let path =
        std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("confidence-staged.cubin");
    std::fs::write(
        &path,
        fold_cubin_for_symbol(confidence_staged::SYMBOL).unwrap().1,
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
        .split(&format!("Function : {}", confidence_staged::SYMBOL))
        .nth(1)
        .expect("staged confidence symbol");
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

/// Hidden rows of `pattern`: random, a one-ulp product, ordered partials
/// that distinguish a serial outer fold from a reassociated one, signed
/// zeros and subnormals, and infinities and NaNs.
fn hidden_rows(pattern: u32, rows: usize, stride: usize, hidden: usize) -> Vec<f32> {
    let mut x = kernel_harness::wide_values(rows * stride, 31 + u64::from(pattern));
    if pattern == 0 {
        return x;
    }
    for row in 0..rows {
        let x = &mut x[row * stride..][..hidden];
        match pattern {
            1 => {
                x.fill(0.0);
                x[0] = f32::from_bits(0x3f80_8000);
            }
            2 => {
                x.fill(0.0);
                for (i, value) in [16777216.0, 1.0, -16777216.0, 1.0].into_iter().enumerate() {
                    x[i * 128] = value;
                }
            }
            3 => {
                for (i, value) in x.iter_mut().enumerate() {
                    *value = f32::from_bits(
                        [
                            0,
                            0x8000_0000,
                            1,
                            0x8000_0001,
                            0x007f_ffff,
                            0x807f_ffff,
                            0x0080_0000,
                        ][i % 7],
                    );
                }
            }
            4 => {
                for (i, value) in x.iter_mut().enumerate().step_by(97 + row) {
                    *value = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN][i % 3];
                }
            }
            _ => unreachable!(),
        }
    }
    x
}

/// Every case of one operand layout: both entries' bits, the oracle,
/// poisoned stride gaps and dead rows, eager, relaunch, graph and PDL
/// replays, and optional pair timing.
fn compare(
    s: &Session,
    op: ConfidenceHead,
    pattern: u32,
    probe_timing: bool,
) -> LabResult<(usize, usize)> {
    let most = *ROWS.iter().max().unwrap() as usize;
    let (hidden, rank, stride) = (
        op.hidden as usize,
        op.markov_rank as usize,
        op.hidden_stride as usize,
    );
    let x = hidden_rows(pattern, most, stride, hidden);
    let mut m = bf16(most * rank, 47 + u64::from(pattern));
    let mut w = bf16(hidden + rank, 67 + u64::from(pattern));
    if matches!(pattern, 1 | 2) {
        m.fill(0);
        w.fill(f32_to_bf16(1.0));
    }
    let exact: Vec<u32> = (0..most)
        .map(|row| {
            let input: Vec<f32> = x[row * stride..][..hidden]
                .iter()
                .copied()
                .chain(m[row * rank..][..rank].iter().copied().map(bf16_to_f32))
                .collect();
            chain_f32(&input, &w).to_bits()
        })
        .collect();
    if pattern == 2 && exact.iter().any(|&bits| bits != 1.0_f32.to_bits()) {
        return Err("confidence order control does not distinguish reassociation".into());
    }
    let images = [b32(&x), b16(&m)];
    let inputs = images
        .iter()
        .map(|image| upload(s, image))
        .collect::<LabResult<Vec<_>>>()?;
    let held = images
        .iter()
        .map(|image| upload(s, image))
        .collect::<LabResult<Vec<_>>>()?;
    // The producer holds the hidden rows and embeddings, which a
    // predecessor writes; it never rewrites the weights.
    let rewrites = Rewrites::new(s, &held.iter().zip(&inputs).collect::<Vec<_>>())?;
    let weight = upload(s, &b16(&w))?;
    let producer_module = s.load_cubin(probe::cubin())?;
    let producer = LiveProducer::with_function(
        s,
        producer_module
            .function(probe::SYMBOL)
            .map_err(|e| e.to_string())?,
    )?;
    let modules = [confidence_parallel::SYMBOL, confidence_staged::SYMBOL]
        .iter()
        .map(|symbol| s.load_cubin(fold_cubin_for_symbol(symbol).unwrap().1))
        .collect::<LabResult<Vec<_>>>()?;
    let functions = modules
        .iter()
        .zip([confidence_parallel::SYMBOL, confidence_staged::SYMBOL])
        .map(|(m, symbol)| m.function(symbol).map_err(|e| e.to_string()))
        .collect::<LabResult<Vec<_>>>()?;
    let live = s.allocate(4)?;
    let outputs = [s.allocate(most * 5 * 4)?, s.allocate(most * 5 * 4)?];
    let mut cases = 0;
    let mut timings = 0;
    for rows in ROWS {
        let planned = launches(op, rows);
        let invoke = |arm: usize, stream: &Stream| {
            launch(
                &s.context,
                &functions[arm],
                stream,
                planned[arm].plan,
                &planned[arm].arguments,
                &Bindings {
                    buffers: BTreeMap::from([
                        (1, &inputs[0]),
                        (2, &inputs[1]),
                        (3, &weight),
                        (4, &outputs[arm]),
                        (5, &live),
                    ]),
                },
            )
        };
        let graphs = (0..2)
            .map(|arm| s.capture(|stream| invoke(arm, stream)))
            .collect::<LabResult<Vec<_>>>()?;
        let mut live_counts = vec![0, 1, rows.div_ceil(2), rows - 1, rows];
        live_counts.sort_unstable();
        live_counts.dedup();
        for n in live_counts {
            live.copy_from_bytes(&n.to_le_bytes(), &s.stream)
                .map_err(|e| e.to_string())?;
            let mut scores = Vec::new();
            for arm in 0..2 {
                let mut replay = Replay::new(s).output(
                    "confidence",
                    &outputs[arm],
                    WriteMap::of(DType::F32).ranges((0..n as usize).map(|r| r * 5..r * 5 + 1))?,
                );
                replay.eager(|| invoke(arm, &s.stream))?;
                replay.relaunch(|| invoke(arm, &s.stream))?;
                for _ in 0..3 {
                    replay.graph(&graphs[arm])?;
                }
                replay.pdl(
                    || producer.launch_rewriting(s, &live, &rewrites),
                    || invoke(arm, &s.stream),
                    true,
                )?;
                scores.push(
                    s.download_f32(&outputs[arm], most * 5)?
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                );
            }
            let label =
                format!("hidden={hidden} stride={stride} pattern={pattern} rows={rows} live={n}");
            for (row, &expected) in exact.iter().enumerate().take(n as usize) {
                let (served, staged) = (scores[0][row * 5], scores[1][row * 5]);
                if staged != served {
                    return Err(format!(
                        "{label}: row {row} staged {staged:#x} != served {served:#x}"
                    ));
                }
                let nan = f32::from_bits(expected).is_nan() && f32::from_bits(staged).is_nan();
                if staged != expected && !nan {
                    return Err(format!(
                        "{label}: row {row} staged {staged:#x} != chain128 {expected:#x}"
                    ));
                }
            }
            cases += 1;
        }
        if probe_timing && pattern == 0 {
            live.copy_from_bytes(&rows.to_le_bytes(), &s.stream)
                .map_err(|e| e.to_string())?;
            let timing =
                kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                    graphs[arm].launch(stream).map_err(|e| e.to_string())
                })?;
            eprintln!(
                "DSPARK_CONFIDENCE_STAGED_TIMING hidden={hidden} stride={stride} rows={rows} \
                 served_us={:.3} staged_us={:.3} ratio={:.3}",
                timing.arms[0].median_ms * 1000.,
                timing.arms[1].median_ms * 1000.,
                timing.paired_ratio
            );
            timings += 1;
        }
    }
    Ok((cases, timings))
}

#[test]
fn confidence_staged_gpu_matches_the_served_entry_and_oracle() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("the staged confidence comparison needs an idle GPU").unwrap();
        return;
    }
    let probe = std::env::var_os("CHOPINFER_DSPARK_CONFIDENCE_PROBE")
        .map(|path| &*Box::leak(std::fs::read(path).unwrap().into_boxed_slice()));
    let cubins = probe.map_or(Cubins::NONE, |probe| {
        Cubins::new(
            Box::leak(Box::new([("probe_sm120", probe)])),
            &[("probe_sm120", PROBE_SYMBOLS)],
        )
    });
    let result = kernel_harness::gpu_test_with(cubins, 768 << 20, |s| {
        let mut cases = 0;
        let mut timings = 0;
        // The served item-major stride takes the vector loads; an odd
        // stride takes the scalar copy.
        for stride in [5 * FLASH.shape.hidden, 5 * FLASH.shape.hidden + 3] {
            for pattern in 0..5 {
                let (c, t) = compare(s, served_op(stride), pattern, probe.is_some())?;
                cases += c;
                timings += t;
            }
            eprintln!("DSPARK_CONFIDENCE_STAGED_BITWISE stride={stride} PASS");
        }
        assert_eq!(cases, 2 * 5 * 25);
        eprintln!("DSPARK_CONFIDENCE_STAGED_COMPLETE cases={cases} timings={timings}");
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "the comparison ran");
}
