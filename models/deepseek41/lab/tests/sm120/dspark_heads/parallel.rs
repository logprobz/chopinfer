//! Compare the cooperative confidence entry with the retained serial entry.
use super::{b16, b32, bf16, chain_f32, probe, upload};
use bundle_deepseek41::sm120::confidence_parallel;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan};
use kernel_harness::eviction::EvictionMode;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::poison::{LiveProducer, Replay, Rewrites, WriteMap};
use kernel_harness::{Cubins, LabResult, Session};
use runtime_cuda::Stream;
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const REPEATS: usize = 16;
const PROBES: &[&str] = &[
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
    "chop_probe_gate_v1",
];

fn timing_cubins() -> LabResult<Cubins> {
    let root = PathBuf::from(
        std::env::var_os("CHOPINFER_CONFIDENCE_ARTIFACTS")
            .ok_or("staged confidence artifacts required")?,
    );
    let bytes = std::fs::read(root.join("probe_sm120.cubin")).map_err(|e| e.to_string())?;
    let lock: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../../crates/kernel-lab/kernels.lock"
    ))
    .unwrap();
    let record = lock
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "probe_sm120")
        .ok_or("missing timing probe lock")?;
    if record["cubin_sha256"] != kernel_build::digest(&bytes) {
        return Err("confidence timing probe differs from its lock".into());
    }
    let bytes = Box::leak(bytes.into_boxed_slice());
    Ok(Cubins::new(
        Box::leak(Box::new([("probe_sm120", &bytes[..])])),
        &[("probe_sm120", PROBES)],
    ))
}

fn padded(mut image: Vec<u8>) -> Vec<u8> {
    image.resize(image.len().next_multiple_of(4), 0);
    image
}

fn compare(s: &Session, hidden: u32, rank: u32, capacity: u32, pattern: u32) -> LabResult<()> {
    let stride = hidden * 5 + 3;
    let mut x = kernel_harness::wide_values((capacity * stride) as usize, 31);
    let mut m = bf16((capacity * rank) as usize, 47);
    let mut w = bf16((hidden + rank) as usize, 67);
    if pattern != 0 {
        x.fill(0.0);
        m.fill(0);
        w.fill(f32_to_bf16(1.0));
        for row in 0..capacity as usize {
            let x = &mut x[row * stride as usize..][..hidden as usize];
            match pattern {
                1 => x[0] = f32::from_bits(0x3f80_8000),
                2 => {
                    // Ordered partials [2^24, 1, -2^24, 1] distinguish a
                    // serial outer fold from a reassociated reduction tree.
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
                _ => return Err("unknown confidence pattern".into()),
            }
        }
    }
    let exact: Vec<_> = (0..capacity as usize)
        .map(|row| {
            let input = x[row * stride as usize..][..hidden as usize]
                .iter()
                .copied()
                .chain(
                    m[row * rank as usize..][..rank as usize]
                        .iter()
                        .copied()
                        .map(bf16_to_f32),
                )
                .collect::<Vec<_>>();
            chain_f32(&input, &w).to_bits()
        })
        .collect();
    if pattern == 2 && exact.iter().any(|&bits| bits != 1.0_f32.to_bits()) {
        return Err("confidence order control does not distinguish reassociation".into());
    }
    let images = [b32(&x), padded(b16(&m)), padded(b16(&w))];
    let inputs = images
        .iter()
        .map(|image| upload(s, image))
        .collect::<LabResult<Vec<_>>>()?;
    let staged = images
        .iter()
        .map(|image| upload(s, image))
        .collect::<LabResult<Vec<_>>>()?;
    let rewrites = Rewrites::new(s, &staged.iter().zip(&inputs).collect::<Vec<_>>())?;
    let producer_module = s.load_cubin(probe::cubin())?;
    let producer = LiveProducer::with_function(
        s,
        producer_module
            .function(probe::SYMBOL)
            .map_err(|e| e.to_string())?,
    )?;
    let modules = [
        s.load_cubin(super::super::cubin())?,
        s.load_cubin(super::super::parallel_cubin())?,
    ];
    let functions = [
        modules[0].function(crate::dspark_entries::CONFIDENCE),
        modules[1].function(confidence_parallel::SYMBOL),
    ]
    .into_iter()
    .collect::<Result<Vec<_>, _>>()
    .map_err(|e| e.to_string())?;
    let live = s.allocate(4)?;
    let outputs = [
        s.allocate(capacity as usize * 5 * 4)?,
        s.allocate(capacity as usize * 5 * 4)?,
    ];
    let args = ArgumentList(vec![
        Argument::DeviceBuffer(1),
        Argument::DeviceBuffer(2),
        Argument::DeviceBuffer(3),
        Argument::DeviceBuffer(4),
        Argument::DeviceBuffer(5),
        Argument::U32(hidden),
        Argument::U32(rank),
        Argument::U32(stride),
        Argument::U32(5),
    ]);
    let invoke = |arm: usize, stream: &Stream| {
        let plan = LaunchPlan {
            grid: [
                if arm == 0 {
                    capacity.div_ceil(128)
                } else {
                    capacity
                },
                1,
                1,
            ],
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1; 3],
            pdl: true,
        };
        launch(
            &s.context,
            &functions[arm],
            stream,
            plan,
            &args,
            &Bindings {
                buffers: BTreeMap::from([
                    (1, &inputs[0]),
                    (2, &inputs[1]),
                    (3, &inputs[2]),
                    (4, &outputs[arm]),
                    (5, &live),
                ]),
            },
        )
    };
    let graphs = [
        s.capture(|stream| invoke(0, stream))?,
        s.capture(|stream| invoke(1, stream))?,
    ];
    let mut live_counts = vec![0, 1, capacity.div_ceil(2), capacity];
    live_counts.sort_unstable();
    live_counts.dedup();
    for n in live_counts {
        live.copy_from_bytes(&n.to_le_bytes(), &s.stream)
            .map_err(|e| e.to_string())?;
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
            let got = s.download_f32(&outputs[arm], capacity as usize * 5)?;
            for row in 0..n as usize {
                if got[row * 5].to_bits() != exact[row] {
                    return Err(format!(
                        "confidence arm {arm} width {hidden} capacity {capacity} pattern {pattern} row {row} differs from chain128"
                    ));
                }
            }
        }
        s.receipt(
            "confidence-parallel-parity",
            BTreeMap::from([
                ("hidden".into(), json!(hidden)),
                ("rank".into(), json!(rank)),
                ("capacity".into(), json!(capacity)),
                ("live".into(), json!(n)),
                ("pattern".into(), json!(pattern)),
                ("serial_and_parallel_bitwise".into(), json!(true)),
                ("poison_graph_pdl".into(), json!(true)),
            ]),
        )?;
    }
    if hidden == 5120 && pattern == 0 && matches!(capacity, 1 | 8) {
        let graphs = (0..2)
            .map(|arm| {
                s.capture(|stream| {
                    for _ in 0..REPEATS {
                        invoke(arm, stream)?;
                    }
                    Ok(())
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        for mode in [EvictionMode::Read, EvictionMode::Write] {
            let timing = kernel_harness::time_ab(s, 3, 9, [mode; 2], |arm, stream| {
                graphs[arm].launch(stream).map_err(|e| e.to_string())
            })?;
            s.receipt(
                "confidence-parallel-timing",
                BTreeMap::from([
                    ("rows".into(), json!(capacity)),
                    ("eviction".into(), json!(mode.name())),
                    (
                        "serial_us".into(),
                        json!(timing.arms[0].median_ms * 1000. / REPEATS as f32),
                    ),
                    (
                        "parallel_us".into(),
                        json!(timing.arms[1].median_ms * 1000. / REPEATS as f32),
                    ),
                    ("paired_ratio".into(), json!(timing.paired_ratio)),
                ]),
            )?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_CONFIDENCE_ARTIFACTS"]
fn confidence_parallel_matches_serial_bits_poison_graph_and_pdl() {
    let result = kernel_harness::gpu_test_with(timing_cubins().unwrap(), 768 << 20, |s| {
        for hidden in [1024, 5120] {
            for capacity in [1, 8, 32, 129] {
                for pattern in 0..4 {
                    compare(s, hidden, 256, capacity, pattern)?;
                }
            }
        }
        // Cross the hidden/Markov boundary inside a chain, and cross the
        // cooperative block's 128-chain tile without folding padding zeros.
        for hidden in [1023, 16639] {
            compare(s, hidden, 129, 3, 0)?;
        }
        Ok(())
    })
    .unwrap();
    assert!(
        result.is_some(),
        "confidence GPU comparison did not execute"
    );
}
