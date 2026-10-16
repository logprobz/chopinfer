//! F32 side tensors: f64 oracles, BF16 parity, poisoned replay and row invariance.
#[path = "common/norm_weights.rs"]
mod norm_checks;
use norm_checks::cases;

use kernel_core::{
    codec::{bf16_to_f32, f32_to_bf16},
    contract::DType,
    oracle::OracleOutput,
};
use kernel_lab::{
    LabResult, Session,
    poison::{self, Replay, WriteMap},
};
use kernels_sm120::comm::epilogue::{GroupedRmsNorm, NormReduction};
use runtime_cuda::{KernelArgument, LaunchConfig};
use serde_json::json;
use std::collections::BTreeMap;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn weights(n: usize, exact: bool) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let w = match i % 5 {
                0 => -0.9998,
                1 => -1.0002,
                2 => 0.10003,
                3 => -2.003,
                _ => (i % 71) as f32 / 91.0,
            };
            if exact {
                bf16_to_f32(f32_to_bf16(w))
            } else {
                w
            }
        })
        .collect()
}

fn input(ids: &[usize], width: usize) -> Vec<u16> {
    norm_checks::varied_input(ids, width)
}

fn norm_oracle(x: &[f64], w: &[f32], width: usize) -> OracleOutput {
    let mut values = Vec::new();
    let mut sums = Vec::new();
    for (g, group) in x.chunks_exact(width).enumerate() {
        let square = group.iter().map(|v| v * v).sum::<f64>();
        let inv = (square / width as f64 + f64::from(1e-6_f32)).sqrt().recip();
        for (c, &v) in group.iter().enumerate() {
            let value = v * inv * (1.0 + f64::from(w[(g * width + c) % w.len()]));
            values.push(value);
            sums.push(value.abs());
        }
    }
    OracleOutput::new(values, sums, width).unwrap()
}

fn row_run(
    session: &Session,
    shape: GroupedRmsNorm,
    ids: &[usize],
    capacity: u32,
    exact: bool,
) -> LabResult<Vec<u16>> {
    let width = (shape.group_width * shape.groups) as usize;
    let x = input(ids, width);
    let w = weights(width, exact);
    let source = session.upload_bf16(if x.is_empty() { &[0] } else { &x })?;
    let weight = if shape.weight_dtype == DType::F32 {
        session.upload_f32(&w)?
    } else {
        session.upload_bf16(&w.iter().copied().map(f32_to_bf16).collect::<Vec<_>>())?
    };
    let output = session.allocate(capacity as usize * width * 2)?;
    let count = session.allocate(4)?;
    poison::write(session, &count, 0, &(ids.len() as u32).to_le_bytes())?;
    let module = session.module(cases::CUBIN)?;
    let symbol = cases::row_symbol(shape.group_width, shape.groups, shape.weight_dtype);
    let function = module.function(&symbol).map_err(err)?;
    let plan = shape.plan(capacity, ids.len() as u32).map_err(err)?;
    let config = LaunchConfig {
        grid: (plan.grid[0], plan.grid[1], 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
        cluster: LaunchConfig::NO_CLUSTER,
        programmatic_stream_serialization: true,
    };
    let launch = || {
        function
            .launch(
                &session.stream,
                config,
                &[
                    KernelArgument::Buffer(&source),
                    KernelArgument::Buffer(&weight),
                    KernelArgument::Buffer(&output),
                    KernelArgument::Buffer(&count),
                ],
            )
            .map_err(err)
    };
    let mut replay =
        Replay::new(session).output("norm", &output, WriteMap::of(DType::Bf16).prefix(x.len())?);
    replay.eager(launch)?;
    replay.relaunch(launch)?;
    let graph = session.capture(|_| launch())?;
    replay.graph(&graph)?;
    let actual = session.download_bf16(&output, x.len())?;
    let decoded = x
        .iter()
        .map(|&v| f64::from(bf16_to_f32(v)))
        .collect::<Vec<_>>();
    let oracle = norm_oracle(&decoded, &w, shape.group_width as usize);
    let mut receipt = BTreeMap::from([
        ("symbol".into(), json!(symbol)),
        ("rows".into(), json!(ids)),
        ("capacity".into(), json!(capacity)),
        ("bf16_representable_weights".into(), json!(exact)),
    ]);
    kernel_lab::check_output(
        &actual.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
        &oracle,
        DType::Bf16,
        &mut receipt,
    )?;
    replay.record(&mut receipt);
    session.receipt("norm_weights", receipt)?;
    Ok(actual)
}

fn row_case(width: u32, groups: u32) {
    kernel_lab::gpu_test(64 << 20, |session| {
        let shape = GroupedRmsNorm {
            reduction: if groups == 1 {
                NormReduction::XorThenSerial
            } else {
                NormReduction::ShuffleDown
            },
            weight_dtype: DType::F32,
            group_width: width,
            groups,
        };
        let stride = (width * groups) as usize;
        let full = row_run(session, shape, &[0, 1, 2, 3], 8, false)?;
        let permuted = row_run(session, shape, &[3, 1, 0, 2], 4, false)?;
        for (at, id) in [3, 1, 0, 2].into_iter().enumerate() {
            assert_eq!(
                &permuted[at * stride..(at + 1) * stride],
                &full[id * stride..(id + 1) * stride]
            );
        }
        for ids in [&[0][..], &[1, 2][..], &[3][..]] {
            let chunk = row_run(session, shape, ids, ids.len() as u32, false)?;
            assert_eq!(chunk, full[ids[0] * stride..(ids[0] + ids.len()) * stride]);
        }
        row_run(session, shape, &[], 8, false)?;
        let f32 = row_run(session, shape, &[0, 1], 4, true)?;
        let bf16 = row_run(
            session,
            GroupedRmsNorm {
                weight_dtype: DType::Bf16,
                ..shape
            },
            &[0, 1],
            4,
            true,
        )?;
        assert_eq!(
            f32, bf16,
            "representable side tensors must preserve output bits"
        );
        let narrowed = row_run(
            session,
            GroupedRmsNorm {
                weight_dtype: DType::Bf16,
                ..shape
            },
            &[0, 1, 2, 3],
            8,
            true,
        )?;
        assert_ne!(full, narrowed, "fixture must catch narrowing F32 weights");
        session.receipt(
            "norm_weights_invariance",
            BTreeMap::from([
                ("group_width".into(), json!(width)),
                ("groups".into(), json!(groups)),
                ("chunk_bitwise".into(), json!(true)),
                ("rows_bitwise".into(), json!(true)),
                ("batch_bitwise".into(), json!(true)),
                ("bf16_weight_parity".into(), json!(true)),
            ]),
        )?;
        Ok(())
    })
    .unwrap();
}

#[test]
fn grouped_streams_f32() {
    row_case(2560, 4);
}
#[test]
fn across_streams_f32() {
    row_case(10240, 1);
}
#[test]
fn embedding_row_f32() {
    row_case(2560, 1);
}

fn comm_run(
    session: &Session,
    ids: &[usize],
    capacity: u32,
    ranks: usize,
    mode: u32,
) -> LabResult<Vec<u16>> {
    const WIDTH: usize = 2560;
    const WIDE: usize = WIDTH * 4;
    let mut residual = input(ids, WIDE);
    for (row, &id) in ids.iter().enumerate() {
        if id == 4 {
            residual[row * WIDE..(row + 1) * WIDE].fill(0);
        }
    }
    let logits: Vec<_> = ids
        .iter()
        .flat_map(|&r| {
            (0..4).map(move |s| {
                f32_to_bf16(if r == 4 {
                    -400.0
                } else {
                    [-400.0, -8.0, 0.0, 8.0][(r + s) % 4]
                })
            })
        })
        .collect();
    let partials: Vec<_> = ids
        .iter()
        .flat_map(|&row| {
            (0..ranks * WIDTH).map(move |c| {
                if row == 4 {
                    1e30_f32
                } else {
                    (((row * 11 + c * 3) % 67) as f32 - 33.0) / 64.0
                }
            })
        })
        .collect();
    let w = weights(WIDE, false);
    let mut injected = Vec::new();
    for row in 0..ids.len() {
        for stream in 0..4 {
            let z = f64::from(bf16_to_f32(logits[row * 4 + stream])) / 4.0;
            let scale = 2.0 / (1.0 + (-z).exp());
            for c in 0..WIDTH {
                let sum = (0..ranks)
                    .map(|rank| f64::from(partials[(row * ranks + rank) * WIDTH + c]))
                    .sum::<f64>();
                injected.push(
                    f64::from(bf16_to_f32(residual[row * WIDE + stream * WIDTH + c])) + scale * sum,
                );
            }
        }
    }
    let source = session.upload_f32(if partials.is_empty() {
        &[0.0]
    } else {
        &partials
    })?;
    let x = session.upload_bf16(if residual.is_empty() { &[0] } else { &residual })?;
    let l = session.upload_bf16(if logits.is_empty() { &[0] } else { &logits })?;
    let weight = session.upload_f32(&w)?;
    let residual_out = session.allocate(capacity as usize * WIDE * 2)?;
    let output = session.allocate(capacity as usize * WIDE * 2)?;
    let count = session.allocate(4)?;
    poison::write(session, &count, 0, &(ids.len() as u32).to_le_bytes())?;
    let symbol = format!("chop_lab_norm_comm_f32_m{mode}_v1");
    let module = session.module(cases::CUBIN)?;
    let function = module.function(&symbol).map_err(err)?;
    let config = LaunchConfig {
        grid: (capacity, if mode == 2 { 1 } else { 4 }, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
        cluster: LaunchConfig::NO_CLUSTER,
        programmatic_stream_serialization: true,
    };
    let launch = || {
        function
            .launch(
                &session.stream,
                config,
                &[
                    KernelArgument::Buffer(&source),
                    KernelArgument::Buffer(&x),
                    KernelArgument::Buffer(&l),
                    KernelArgument::Buffer(&weight),
                    KernelArgument::Buffer(&residual_out),
                    KernelArgument::Buffer(&output),
                    KernelArgument::Buffer(&count),
                    KernelArgument::U32(ranks as u32),
                ],
            )
            .map_err(err)
    };
    let map = WriteMap::of(DType::Bf16).prefix(ids.len() * WIDE)?;
    let mut replay = Replay::new(session)
        .output("norm", &output, map.clone())
        .output("residual", &residual_out, map);
    replay.eager(launch)?;
    replay.relaunch(launch)?;
    let graph = session.capture(|_| launch())?;
    replay.graph(&graph)?;
    let actual = session.download_bf16(&output, residual.len())?;
    let actual_residual = session.download_bf16(&residual_out, residual.len())?;
    let mut receipt = BTreeMap::from([
        ("symbol".into(), json!(symbol)),
        ("rows".into(), json!(ids)),
        ("ranks_summed_locally".into(), json!(ranks)),
        ("collective_qualified".into(), json!(false)),
    ]);
    let residual_oracle = OracleOutput::new(
        injected.clone(),
        injected.iter().map(|v| v.abs()).collect(),
        ranks,
    )
    .unwrap();
    kernel_lab::check_output(
        &actual_residual
            .iter()
            .copied()
            .map(bf16_to_f32)
            .collect::<Vec<_>>(),
        &residual_oracle,
        DType::Bf16,
        &mut receipt,
    )?;
    kernel_lab::check_output(
        &actual.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
        &norm_oracle(&injected, &w, WIDTH),
        DType::Bf16,
        &mut receipt,
    )?;
    replay.record(&mut receipt);
    session.receipt("norm_weights", receipt)?;
    Ok([actual, actual_residual].concat())
}

#[test]
fn comm_epilogue_f32() {
    kernel_lab::gpu_test(64 << 20, |session| {
        for ranks in [2, 4, 8] {
            let baseline = comm_run(session, &[0, 1, 2], 8, ranks, 0)?;
            for mode in 0..3 {
                assert_eq!(baseline, comm_run(session, &[0, 1, 2], 3, ranks, mode)?);
                let perm = comm_run(session, &[2, 0, 1], 4, ranks, mode)?;
                let stride = 10240;
                for output in 0..2 {
                    for (at, id) in [2, 0, 1].into_iter().enumerate() {
                        let base = output * 3 * stride;
                        assert_eq!(
                            &perm[base + at * stride..base + (at + 1) * stride],
                            &baseline[base + id * stride..base + (id + 1) * stride]
                        );
                    }
                    for id in 0..3 {
                        let chunk = comm_run(session, &[id], 1, ranks, mode)?;
                        assert_eq!(
                            &chunk[output * stride..(output + 1) * stride],
                            &baseline[(output * 3 + id) * stride..(output * 3 + id + 1) * stride]
                        );
                    }
                }
                comm_run(session, &[], 8, ranks, mode)?;
            }
        }
        session.receipt(
            "norm_weights_invariance",
            BTreeMap::from([
                ("epilogue_forms_bitwise".into(), json!(true)),
                ("chunk_bitwise".into(), json!(true)),
                ("rows_bitwise".into(), json!(true)),
                ("batch_bitwise".into(), json!(true)),
                ("collective_qualified".into(), json!(false)),
            ]),
        )?;
        Ok(())
    })
    .unwrap();
}

#[test]
fn oracle_catches_narrowed_weights_and_wrong_groups() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let w = weights(4, false);
    let narrowed = weights(4, true);
    assert_ne!(
        norm_oracle(&x, &w, 2).values,
        norm_oracle(&x, &narrowed, 2).values
    );
    assert_ne!(norm_oracle(&x, &w, 2).values, norm_oracle(&x, &w, 4).values);
    let incorrectly_narrowed = norm_oracle(&x, &narrowed, 2)
        .values
        .iter()
        .map(|&v| f64::from(bf16_to_f32(kernel_core::codec::f64_to_bf16(v))))
        .collect::<Vec<_>>();
    assert!(
        kernel_core::oracle::check_numeric(
            &incorrectly_narrowed,
            &norm_oracle(&x, &w, 2),
            DType::Bf16
        )
        .is_err()
    );
    assert!(norm_oracle(&x, &w, 2).values[0] > 0.0);
    assert_eq!(norm_oracle(&x, &narrowed, 2).values[0], 0.0);
}

#[test]
fn every_instance_has_a_lock_record() {
    let lock: serde_json::Value = serde_json::from_str(include_str!("../kernels.lock")).unwrap();
    for spec in [cases::cubin_spec(), cases::wrong_tree_spec()] {
        let record = lock
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["name"] == spec.name)
            .unwrap();
        assert_eq!(
            record["entries"].as_array().unwrap().len(),
            spec.entries.len()
        );
        for entry in &spec.entries {
            assert!(
                record["entries"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|r| r["symbol"] == entry.symbol)
            );
        }
    }
    let owned_lock: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("OUT_DIR"),
        "/hc-prefill/kernels.lock"
    )))
    .unwrap();
    let legacy = owned_lock
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == cases::LEGACY_CUBIN)
        .unwrap();
    assert_eq!(legacy["entries"].as_array().unwrap().len(), 3);
    for mode in 0..3 {
        let symbol = format!("chop_lab_norm_comm_legacy_bf16_m{mode}_v1");
        assert!(
            legacy["entries"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["symbol"] == symbol)
        );
    }
}

#[test]
fn write_maps_reject_missing_and_inactive_row_writes() {
    let map = WriteMap::of(DType::Bf16).prefix(2).unwrap();
    let fill = [poison::POISON];
    let valid = [0, 0, 0, 0, 255, 255];
    assert!(poison::check("norm", &valid, &map, poison::Fill::Pattern(&fill)).is_ok());
    assert!(poison::check("norm", &[255; 6], &map, poison::Fill::Pattern(&fill)).is_err());
    assert!(poison::check("norm", &[0; 6], &map, poison::Fill::Pattern(&fill)).is_err());
}

#[test]
fn sparse_adversary_exposes_wrong_tree_after_bf16_rounding() {
    for width in [2560, 10240] {
        let (x, w) = norm_checks::sparse_input(width, 1);
        assert_eq!(norm_checks::square_sum(&x, false), 16_777_222.0);
        assert_eq!(norm_checks::square_sum(&x, true), 16_777_216.0);
        assert_ne!(
            norm_checks::cpu_norm(&x, &w, false),
            norm_checks::cpu_norm(&x, &w, true)
        );
    }
}

#[test]
fn wrong_tree_changes_gpu_output_bits() {
    kernel_lab::gpu_test(64 << 20, norm_checks::wrong_tree_gpu).unwrap();
}

#[test]
fn comm_inject_tail_f32() {
    kernel_lab::gpu_test(64 << 20, |session| {
        let expected = comm_run(session, &[4], 2, 2, 0)?;
        assert!(expected.iter().any(|&v| bf16_to_f32(v) != 0.0));
        for mode in 1..3 {
            assert_eq!(expected, comm_run(session, &[4], 2, 2, mode)?);
        }
        Ok(())
    })
    .unwrap();
}
