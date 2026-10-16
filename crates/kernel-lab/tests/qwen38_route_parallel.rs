//! parallel routing: exact selections, weight bits, stable metadata and replay.
use kernel_core::contract::DType;
use kernel_lab::{LabResult, Session, gpu_test, launch, time};
use kernels_sm120::moe;
use model_qwen38_contract::sm120::moe as qwen;
use runtime_cuda::{DeviceBuffer, KernelArgument, Stream};
use serde_json::json;
use std::collections::BTreeMap;

use kernel_lab::poison as replay;
use kernel_lab::poison::{Phase as ReplayPhase, WriteMap};

fn replay_checks<'a>(
    session: &'a Session,
    buffers: &'a [DeviceBuffer],
    geometry: replay::RouteGeometry,
    metadata: bool,
) -> LabResult<replay::Checks<'a>> {
    let checks = replay::Checks::new(session);
    if metadata {
        checks.route(
            geometry,
            replay::RouteBuffers {
                ids: &buffers[1],
                weights: &buffers[2],
                counts: &buffers[3],
                offsets: &buffers[4],
                sorted: &buffers[5],
                padded: &buffers[6],
                active: Some(&buffers[8]),
            },
            &[0x80],
        )
    } else {
        let map = WriteMap::of(DType::I32).prefix(geometry.routes)?;
        Ok(checks
            .dynamic(
                "route IDs",
                &buffers[1],
                {
                    let map = map.clone();
                    move |_| Ok(map.clone())
                },
                &[0x80],
            )
            .dynamic(
                "route weights",
                &buffers[2],
                move |_| Ok(map.clone()),
                &[0x80],
            ))
    }
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len())?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

/// Route outputs and guards. The ticket is checked separately: the one-CTA
/// route has none.
fn snapshot(session: &Session, buffers: &[DeviceBuffer]) -> LabResult<Vec<Vec<u8>>> {
    [1, 2, 3, 4, 5, 6, 8]
        .iter()
        .map(|&i| {
            let mut bytes = vec![0; buffers[i].byte_len()];
            buffers[i]
                .copy_to_bytes(&mut bytes, &session.stream)
                .map_err(|e| e.to_string())?;
            Ok(bytes)
        })
        .collect()
}

/// The ticket: the parallel route's arrival count.
fn ticket_words(session: &Session, ticket: &DeviceBuffer) -> LabResult<Vec<u32>> {
    let mut bytes = vec![0; ticket.byte_len()];
    ticket
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes
        .chunks_exact(4)
        .map(|w| u32::from_ne_bytes(w.try_into().unwrap()))
        .collect())
}

const TICKET_WORDS: usize = (moe::parallel::PARALLEL_ROUTE_TICKET_BYTES / 4) as usize;

#[test]
fn qwen38_parallel_route_is_bitwise_exact() {
    gpu_test(512 << 20, |session| {
        let mut failures = replay::Failures::default();
        for ranks in [4, 2] {
            let profile = model_qwen38_contract::sm120::config::profile(ranks).ok_or("Qwen topology")?;
            let local = qwen::local_experts(profile).map_err(|e| e.0.to_string())?;
            let original = session.module(&qwen::route_cubin_spec_for(profile).unwrap().name)?;
            let parallel = session.module(&qwen::parallel_route_cubin_spec_for(profile).unwrap().name)?;
            for capacity in qwen::DECODE_BUCKETS.into_iter().filter(|&c| qwen::use_parallel_route(c)) {
                let old = qwen::route_entry_for(profile, &format!("decode_route_t{capacity}")).map_err(|e| e.0.to_string())?;
                let new = qwen::route_entry_for(profile, &format!("decode_route_parallel_t{capacity}")).map_err(|e| e.0.to_string())?;
                let old_fn = original.function(&old.symbol).map_err(|e| e.to_string())?;
                let new_fn = parallel.function(&new.symbol).map_err(|e| e.to_string())?;
                let mut buffers = Vec::new();
                // logits, IDs, weights, counts, offsets, sorted routes plus a
                // guard, padded count, live count, active experts, ticket.
                for words in [capacity * 512, capacity * 10, capacity * 10,
                    local, local + 1, capacity * 10 + local * 15 + 32,
                    1, 1, local + 1, TICKET_WORDS as u32] {
                    buffers.push(upload(session, &vec![0; words as usize * 4])?);
                }
                for rank in 0..ranks {
                    let run = |stream: &Stream, candidate: bool| {
                        let (entry, f) = if candidate { (&new, &new_fn) } else { (&old, &old_fn) };
                        let mut args: Vec<_> = buffers[..9].iter().map(KernelArgument::Buffer).collect();
                        if candidate { args.push(KernelArgument::Buffer(&buffers[9])); }
                        args.push(KernelArgument::U32(rank));
                        f.launch(stream, launch::config(entry.plan(capacity, capacity).map_err(|e| e.0.to_string())?)?, &args).map_err(|e| e.to_string())
                    };
                    let graph = session.capture(|stream| run(stream, true))?;
                    let old_graph = session.capture(|stream| run(stream, false))?;
                    // Every live count, descending so replays leave stale rows,
                    // then full capacity again.
                    for live in (0..=capacity).rev().chain([capacity]) {
                        buffers[7].copy_from_bytes(&(live as i32).to_ne_bytes(), &session.stream).map_err(|e| e.to_string())?;
                        for pattern in 0..12 {
                let result = (|| {
                            let values: Vec<f32> = (0..capacity * 512).map(|i| {
                                let expert = i % 512;
                                let row = i / 512;
                                match pattern {
                                    0 => 0.0,
                                    1 => if expert % 2 == 0 { -0.0 } else { 0.0 },
                                    2 => ((expert + row * 13) % 17) as f32 - 8.0,
                                    3 => ((i.wrapping_mul(1_664_525).wrapping_add(1_013_904_223) >> 8) % 8192) as f32 / 256.0 - 16.0,
                                    4 => if expert / local == rank { 1000.0 } else { -1000.0 },
                                    5 => 1.0 + ((expert * 37 + row) % 13) as f32 * f32::EPSILON,
                                    6 => if expert % 13 == 0 { f32::INFINITY } else { expert as f32 },
                                    7 => if expert % 13 == 0 { f32::NAN } else { expert as f32 },
                                    8 => f32::NEG_INFINITY,
                                    9 => f32::NAN,
                                    10 => if expert % 2 == 0 { f32::NEG_INFINITY } else { f32::NAN },
                                    _ => match row % 4 { 0 => f32::NAN, 1 => f32::NEG_INFINITY,
                                        2 => if expert < 3 { f32::INFINITY } else { -1.0 }, _ => expert as f32 },
                                }
                            }).collect();
                            let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
                            buffers[0].copy_from_bytes(&bytes, &session.stream).map_err(|e| e.to_string())?;
                            let geometry = replay::RouteGeometry {
                                routes: live as usize * 10,
                                experts: (rank * local) as usize..((rank + 1) * local) as usize,
                                block: 16, sorted_offset: 0,
                            };
                            let mut checks = replay_checks(session, &buffers, geometry, true)?;
                            checks.phase(ReplayPhase::Eager, false, || run(&session.stream, false))?;
                            let expected = snapshot(session, &buffers)?;
                            for _ in 0..8 {
                                checks.phase(ReplayPhase::Graph, false, || graph.launch(&session.stream).map_err(|e| e.to_string()))?;
                            }
                            // Retain the uninterrupted ticket-reset stress separately.
                            for _ in 0..8 { graph.launch(&session.stream).map_err(|e| e.to_string())?; }
                            let actual = snapshot(session, &buffers)?;
                            if expected != actual {
                                return Err(format!("TP{ranks} rank {rank}, capacity {capacity}, live {live}, pattern {pattern}: route bytes differ"));
                            }
                            if ticket_words(session, &buffers[9])? != [0] {
                                return Err(format!("TP{ranks} rank {rank}, live {live}: arrival count was not reset"));
                            }
                            if pattern == 0 {
                                let ids: Vec<_> = actual[0].chunks_exact(4).map(|b| i32::from_ne_bytes(b.try_into().unwrap())).collect();
                                for row in 0..live as usize {
                                    if ids[row * 10..row * 10 + 10] != [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] {
                                        return Err("tied logits must select experts 0 through 9".into());
                                    }
                                }
                            }

                    Ok(())
                })();
                failures.check(&format!("rank count {ranks}, rank {rank}, capacity {capacity}, live {live}, pattern {pattern}"), result);
            }
                    }
                    let mut receipt = BTreeMap::from([
                        ("correct".into(), json!(true)),
                        ("ranks".into(), json!(ranks)), ("rank".into(), json!(rank)),
                        ("capacity".into(), json!(capacity)), ("exact_with_guards_and_ticket_reset".into(), json!(true)),
                    ]);
                    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() == Ok("1")
                        && ranks == 4 && rank == 1 {
                        let old = time(session, 2, 15, |stream| old_graph.launch(stream).map_err(|e| e.to_string()))?;
                        let new = time(session, 2, 15, |stream| graph.launch(stream).map_err(|e| e.to_string()))?;
                        receipt.insert("one_cta_us".into(), json!(old.median_ms * 1000.0));
                        receipt.insert("parallel_us".into(), json!(new.median_ms * 1000.0));
                    }
                    receipt.insert("correct".into(), json!(failures.is_empty()));
                    session.receipt("c16_small_route", receipt)?;
                }
            }
        }
        failures.finish()
    }).expect("parallel route exactness");
}

#[test]
fn grouped_route_matches_row_router() {
    use kernels_sm120::moe::parallel::{ParallelRoute, RouteScoring};
    gpu_test(512 << 20, |session| {
        let mut failures = replay::Failures::default();
        let module = session.module("route_grouped_lab")?;
        let candidate = module
            .function("chop_route_grouped_parallel_lab_v1")
            .map_err(|e| e.to_string())?;
        let reference = module
            .function("chop_route_grouped_reference_lab_v1")
            .map_err(|e| e.to_string())?;
        let geometry = ParallelRoute {
            experts: 256,
            local_experts: 256,
            top_k: 8,
            padding: 16,
            capacity: 64,
            groups: 8,
            take_groups: 4,
            scoring: RouteScoring::GroupedSoftmax,
        };
        let mut buffers = Vec::new();
        for words in [
            64 * 256,
            64 * 8,
            64 * 8,
            256,
            257,
            64 * 8 + 256 * 15 + 32,
            1,
            1,
            257,
            1,
        ] {
            buffers.push(upload(session, &vec![0; words * 4])?);
        }
        let mut args: Vec<_> = buffers.iter().map(KernelArgument::Buffer).collect();
        args.push(KernelArgument::U32(0));
        let plan = geometry.plan(64).map_err(|e| e.0.to_string())?;
        let mut fixed = plan;
        fixed.grid = [64, 1, 1];
        fixed.block = [256, 1, 1];
        let graph = session.capture(|stream| {
            candidate
                .launch(stream, launch::config(plan)?, &args)
                .map_err(|e| e.to_string())
        })?;
        for live in [64_i32, 33, 1, 0, 63, 64] {
            buffers[7]
                .copy_from_bytes(&live.to_ne_bytes(), &session.stream)
                .map_err(|e| e.to_string())?;
            for pattern in 0..3 {
                let result = (|| {
                    let logits: Vec<u8> = (0..64 * 256)
                        .flat_map(|i| {
                            let value = match pattern {
                                0 => 0.0,
                                1 => ((i * 97 + i / 256 * 13) % 113) as f32 / 8.0 - 7.0,
                                _ => ((i % 256 / 32 + i / 256) % 8) as f32,
                            };
                            value.to_ne_bytes()
                        })
                        .collect();
                    buffers[0]
                        .copy_from_bytes(&logits, &session.stream)
                        .map_err(|e| e.to_string())?;
                    let geometry = replay::RouteGeometry {
                        routes: live as usize * 8,
                        experts: 0..256,
                        block: 16,
                        sorted_offset: 0,
                    };
                    let mut reference_checks =
                        replay_checks(session, &buffers, geometry.clone(), false)?;
                    reference_checks.phase(ReplayPhase::Eager, false, || {
                        reference
                            .launch(&session.stream, launch::config(fixed)?, &args)
                            .map_err(|e| e.to_string())
                    })?;
                    let expected = snapshot(session, &buffers)?;
                    let mut checks = replay_checks(session, &buffers, geometry, true)?;
                    checks.phase(ReplayPhase::Eager, false, || {
                        candidate
                            .launch(&session.stream, launch::config(plan)?, &args)
                            .map_err(|e| e.to_string())
                    })?;
                    checks.phase(ReplayPhase::Graph, false, || {
                        graph.launch(&session.stream).map_err(|e| e.to_string())
                    })?;
                    let actual = snapshot(session, &buffers)?;
                    if actual[..2] != expected[..2] || ticket_words(session, &buffers[9])? != [0] {
                        return Err(format!(
                            "grouped route differs at live={live}, pattern={pattern}"
                        ));
                    }

                    Ok(())
                })();
                failures.check(
                    &format!("grouped route live {live} pattern {pattern}"),
                    result,
                );
            }
        }
        failures.finish()?;
        session.receipt(
            "c16_small_route",
            BTreeMap::from([
                ("case".into(), json!("grouped_8_take_4_rows_0_64")),
                ("correct".into(), json!(true)),
                ("bitwise_ids_and_weights".into(), json!(true)),
            ]),
        )
    })
    .expect("shared grouped scoring matches the existing row router");
}

/// Invalid device states each run in their own process, because a CUDA trap
/// poisons its context. `capacity:live:stale_count`: a live count above
/// capacity, and stale arrival counts of at least the CTA count with every
/// row live, for Qwen's C32 route. A stale
/// count below the CTA count is excluded by the ticket's ownership (zeroed at
/// create, bound by no other launch, one stream), which the bundle fake-host
/// tests check; the kernel cannot detect it without a per-launch wait.
const TRAP_CASES: [&str; 3] = ["32:33:0", "32:32:32", "32:32:100"];

#[test]
fn parallel_route_rejects_invalid_device_state() {
    if std::env::var_os("CHOPINFER_KERNEL_GPU").is_none() {
        kernel_lab::skip("parallel_route_rejects_invalid_device_state: set CHOPINFER_KERNEL_GPU")
            .unwrap();
        return;
    }
    let Ok(case) = std::env::var("CHOPINFER_C16_TRAP_CASE") else {
        for case in TRAP_CASES {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "parallel_route_rejects_invalid_device_state",
                    "--exact",
                    "--nocapture",
                ])
                .env("CHOPINFER_C16_TRAP_CASE", case)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{case}: {}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            eprintln!("{case}: CUDA device trap confirmed");
        }
        return;
    };
    let fields: Vec<&str> = case.split(':').collect();
    let [capacity, live, stale] = fields[..] else {
        panic!("trap case {case}");
    };
    let (capacity, live, stale): (u32, i32, u32) = (
        capacity.parse().unwrap(),
        live.parse().unwrap(),
        stale.parse().unwrap(),
    );
    gpu_test(512 << 20, |session| {
        let rows = capacity as usize;
        let (experts, top_k, local) = (512, 10, 128);
        let logits: Vec<u8> = (0..rows * experts)
            .flat_map(|i| (((i * 7919) % 4093) as f32 / 512.0 - 4.0).to_ne_bytes())
            .collect();
        let ticket = upload(
            session,
            &vec![0; moe::parallel::PARALLEL_ROUTE_TICKET_BYTES as usize],
        )?;
        ticket
            .copy_from_bytes(&stale.to_ne_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        let live_buffer = upload(session, &live.to_ne_bytes())?;
        let mut outputs = Vec::new();
        // IDs, weights, counts, offsets, sorted rows, padded count, active experts.
        for words in [
            rows * top_k,
            rows * top_k,
            local,
            local + 1,
            rows * top_k + local * 15 + 32,
            1,
            local + 1,
        ] {
            outputs.push(upload(session, &vec![0; words * 4])?);
        }
        let logits = upload(session, &logits)?;
        let entry = qwen::route_entry_for(
            model_qwen38_contract::sm120::config::PROFILE,
            &format!("decode_route_parallel_t{capacity}"),
        )
        .map_err(|e| e.0.to_string())?;
        let module = session.module("chop_moe_qwen38_route_parallel")?;
        let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
        let plan = entry
            .plan(capacity, capacity)
            .map_err(|e| e.0.to_string())?;
        let args = [
            KernelArgument::Buffer(&logits),
            KernelArgument::Buffer(&outputs[0]),
            KernelArgument::Buffer(&outputs[1]),
            KernelArgument::Buffer(&outputs[2]),
            KernelArgument::Buffer(&outputs[3]),
            KernelArgument::Buffer(&outputs[4]),
            KernelArgument::Buffer(&outputs[5]),
            KernelArgument::Buffer(&live_buffer),
            KernelArgument::Buffer(&outputs[6]),
            KernelArgument::Buffer(&ticket),
            KernelArgument::U32(0),
        ];
        function
            .launch(&session.stream, launch::config(plan)?, &args)
            .map_err(|e| e.to_string())?;
        let error = session
            .stream
            .synchronize()
            .expect_err("invalid state must trap")
            .to_string();
        // A device trap surfaces as CUDA_ERROR_LAUNCH_FAILED (719) on the
        // current driver and as CUDA_ERROR_ILLEGAL_INSTRUCTION (715) on others.
        if ![
            "CUDA_ERROR_LAUNCH_FAILED (719)",
            "CUDA_ERROR_ILLEGAL_INSTRUCTION (715)",
        ]
        .iter()
        .any(|code| error.contains(code))
        {
            return Err(format!("expected a CUDA device trap, received {error}"));
        }
        Ok(())
    })
    .expect("invalid device route state traps");
}
