//! parallel routing: exact selections, weight bits, stable metadata and replay.
use kernel_core::contract::DType;
use kernel_lab::{LabResult, Session, gpu_test, launch, time};
use kernels_sm120::moe;
use model_qwen38_contract::sm120::{
    config,
    moe::{self as qwen, GLOBAL_EXPERTS, ROUTE_BLOCK, RouteEntry, TOP_K},
};
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
            let catalog = catalog(session, ranks)?;
            let local = catalog.local as u32;
            for (new, new_fn) in catalog.decode.iter().filter(|(entry, _)| entry.parallel.is_some()) {
                let capacity = new.tokens_capacity;
                let (old, old_fn) = catalog.decode.iter()
                    .find(|(entry, _)| entry.parallel.is_none() && entry.tokens_capacity == capacity)
                    .ok_or("one-CTA route")?;
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
                                    // Both zero signs in every lane (experts lane + 32 j).
                                    1 => if (expert / 32 + expert) % 2 == 0 { -0.0 } else { 0.0 },
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
/// row live, for the parallel C32 route. A stale
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
        let routes = catalog(session, 4)?;
        let (entry, function) = routes
            .decode
            .iter()
            .find(|(entry, _)| entry.parallel.is_some() && entry.tokens_capacity == capacity)
            .ok_or("parallel route")?;
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

/// One rank count's route entries, each with its function: the prefill
/// router at four ranks (it serves every rank count), then each decode
/// bucket's one-CTA route and, where served, its parallel route.
struct Catalog {
    local: usize,
    router: Option<(RouteEntry, runtime_cuda::Function)>,
    decode: Vec<(RouteEntry, runtime_cuda::Function)>,
}

fn catalog(session: &Session, ranks: u32) -> LabResult<Catalog> {
    let profile = config::profile(ranks).ok_or("topology")?;
    let local = qwen::local_experts(profile).map_err(|e| e.0.to_string())? as usize;
    let original = session.module(&qwen::route_cubin_spec_for(profile).unwrap().name)?;
    let parallel = session.module(&qwen::parallel_route_cubin_spec_for(profile).unwrap().name)?;
    let mut operations = Vec::new();
    if ranks == 4 {
        operations.push(("router".to_string(), &original));
    }
    for capacity in qwen::DECODE_BUCKETS {
        operations.push((format!("decode_route_t{capacity}"), &original));
        if qwen::use_parallel_route(capacity) {
            operations.push((format!("decode_route_parallel_t{capacity}"), &parallel));
        }
    }
    let mut catalog = Catalog {
        local,
        router: None,
        decode: Vec::new(),
    };
    for (operation, module) in operations {
        let entry = qwen::route_entry_for(profile, &operation).map_err(|e| e.0.to_string())?;
        let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
        if operation == "router" {
            catalog.router = Some((entry, function));
        } else {
            catalog.decode.push((entry, function));
        }
    }
    Ok(catalog)
}

const EXPERTS: usize = GLOBAL_EXPERTS as usize;
const ROUTES: usize = TOP_K as usize;

fn mix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Rows whose ranking turns on signed zeros. A lane holds experts
/// `lane + 32 * j`, so the declared order (`Ties::LowerIndex`: +0 above -0,
/// then the lower expert) differs from a lane's first strict maximum when a
/// -0 precedes an equal +0 in one lane.
fn signed_zero_rows() -> Vec<Vec<f32>> {
    // Experts 1-9 and 32 at +0 rank above expert 0's -0, which shares
    // expert 32's lane.
    let mut reproducer = vec![-1.0; EXPERTS];
    reproducer[0] = -0.0;
    for expert in (1..10).chain([32]) {
        reproducer[expert] = 0.0;
    }
    let mut rows = vec![reproducer];
    // Nine larger logits in other lanes, then the tenth route between a -0
    // and a +0 of one lane, in both orders, at other lanes and slots.
    for (lane, first, second) in [(5, 0, 3), (31, 1, 15), (17, 2, 9), (0, 14, 15)] {
        for negative_first in [true, false] {
            let mut row = vec![-1.0; EXPERTS];
            for slot in 0..9 {
                row[(lane + 1 + slot) % 32 + 32 * slot] = 1.0;
            }
            let (a, b) = if negative_first {
                (-0.0, 0.0)
            } else {
                (0.0, -0.0)
            };
            row[lane + 32 * first] = a;
            row[lane + 32 * second] = b;
            rows.push(row);
        }
    }
    // A few ones among many zeros of both signs.
    for seed in 0..24_u64 {
        rows.push(
            (0..EXPERTS as u64)
                .map(|expert| match mix(seed << 16 | expert) % 64 {
                    0 => 1.0,
                    1..=16 => 0.0,
                    17..=32 => -0.0,
                    _ => -1.0,
                })
                .collect(),
        );
    }
    rows
}

/// Finite logits with no zero of either sign, which every implementation
/// of the declared order ranks the same way.
const ZERO_FREE: [&str; 5] = ["uniform", "skewed", "equal", "local", "epsilon"];

fn zero_free_row(pattern: &str, row: usize) -> Vec<f32> {
    let row64 = row as u64;
    (0..EXPERTS)
        .map(|expert| {
            let e = expert as u64;
            let value = match pattern {
                "uniform" => (mix(row64 << 20 | e) % 8192) as f32 / 256.0 - 16.0,
                // A per-expert popularity plus per-token noise, like the
                // MoE ops tests' skewed logits, so every rank gets routes.
                "skewed" => {
                    let u = (mix(0x5eed_0000 | e) % 2048) as f32 / 1024.0 - 1.0;
                    let noise = (0..3)
                        .map(|draw| (mix(row64 << 24 | e << 2 | draw) % 1024) as f32 / 1024.0)
                        .sum::<f32>();
                    0.8 * u.signum() * u.abs().powf(1.5) + noise
                }
                "equal" => 1.5 + (expert % 3) as f32,
                "local" => {
                    if expert / 128 == row % 4 {
                        1000.0
                    } else {
                        -1000.0
                    }
                }
                _ => 1.0 + ((expert * 37 + row) % 13) as f32 * f32::EPSILON,
            };
            if value == 0.0 { 0.5 } else { value }
        })
        .collect()
}

fn to_bytes(rows: &[Vec<f32>]) -> Vec<u8> {
    rows.iter()
        .flatten()
        .flat_map(|v| v.to_ne_bytes())
        .collect()
}

fn download(session: &Session, buffer: &DeviceBuffer, words: usize) -> LabResult<Vec<i32>> {
    let mut bytes = vec![0; words * 4];
    buffer
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes
        .chunks_exact(4)
        .map(|w| i32::from_ne_bytes(w.try_into().unwrap()))
        .collect())
}

/// A route entry's buffers: logits, IDs, weights, counts, offsets, sorted
/// routes plus a guard, padded count, live count, active experts and the
/// parallel route's ticket. The router reads the first three and the live
/// count.
struct RouteBuffers {
    buffers: Vec<DeviceBuffer>,
    rows: usize,
    local: usize,
}

impl RouteBuffers {
    fn new(session: &Session, rows: usize, local: usize) -> LabResult<Self> {
        let mut buffers = Vec::new();
        for words in [
            rows * EXPERTS,
            rows * ROUTES,
            rows * ROUTES,
            local,
            local + 1,
            rows * ROUTES + local * 15 + 32,
            1,
            1,
            local + 1,
            TICKET_WORDS,
        ] {
            buffers.push(upload(session, &vec![0; words * 4])?);
        }
        Ok(Self {
            buffers,
            rows,
            local,
        })
    }

    fn fill(&self, session: &Session, rows: &[Vec<f32>]) -> LabResult<()> {
        assert_eq!(rows.len(), self.rows);
        self.buffers[0]
            .copy_from_bytes(&to_bytes(rows), &session.stream)
            .map_err(|e| e.to_string())?;
        self.buffers[7]
            .copy_from_bytes(&(rows.len() as i32).to_ne_bytes(), &session.stream)
            .map_err(|e| e.to_string())
    }

    /// Launches `entry` over every row; `rank` filters a decode route.
    fn launch(
        &self,
        stream: &Stream,
        (entry, function): &(RouteEntry, runtime_cuda::Function),
        rank: Option<u32>,
    ) -> LabResult<()> {
        let rows = self.rows as u32;
        let mut args: Vec<_> = match rank {
            None => [0, 1, 2, 7]
                .iter()
                .map(|&i| KernelArgument::Buffer(&self.buffers[i]))
                .collect(),
            Some(_) => self.buffers[..9]
                .iter()
                .map(KernelArgument::Buffer)
                .collect(),
        };
        if entry.parallel.is_some() {
            args.push(KernelArgument::Buffer(&self.buffers[9]));
        }
        if let Some(rank) = rank {
            args.push(KernelArgument::U32(rank));
        }
        let plan = entry.plan(rows, rows).map_err(|e| e.0.to_string())?;
        function
            .launch(stream, launch::config(plan)?, &args)
            .map_err(|e| e.to_string())
    }

    /// IDs and weights, then for a decode route its counts, offsets, sorted
    /// routes up to the padded count, active experts and the padded count.
    fn outputs(&self, session: &Session, metadata: bool) -> LabResult<Vec<Vec<i32>>> {
        let routes = self.rows * ROUTES;
        let mut out = vec![
            download(session, &self.buffers[1], routes)?,
            download(session, &self.buffers[2], routes)?,
        ];
        if metadata {
            let padded = download(session, &self.buffers[6], 1)?;
            let capacity = routes + self.local * 15 + 32;
            out.push(download(session, &self.buffers[3], self.local)?);
            out.push(download(session, &self.buffers[4], self.local + 1)?);
            out.push(download(
                session,
                &self.buffers[5],
                (padded[0].max(0) as usize).min(capacity),
            )?);
            out.push(download(session, &self.buffers[8], self.local + 1)?);
            out.push(padded);
        }
        Ok(out)
    }

    /// A digest of [`Self::outputs`].
    fn digest(&self, session: &Session, metadata: bool) -> LabResult<String> {
        let bytes: Vec<u8> = self
            .outputs(session, metadata)?
            .concat()
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        Ok(kernel_build::digest(&bytes))
    }
}

/// Checks `outputs` against the f64 oracle: IDs exactly, weights within
/// F32 rounding (as the MoE ops tests check routing), and a decode route's
/// local routes and padded stable permutation exactly.
fn check_against_oracle(
    rows: &[Vec<f32>],
    outputs: &[Vec<i32>],
    local: usize,
    rank: Option<u32>,
) -> LabResult<()> {
    use kernel_core::contract::{MoeQuery, RouterKind};
    let mut selections = Vec::new();
    for (row, logits) in rows.iter().enumerate() {
        let scores: Vec<f64> = logits.iter().map(|&v| f64::from(v)).collect();
        let selection = kernel_core::oracle::router(
            &scores,
            None,
            ROUTES,
            RouterKind::SoftmaxTopK { renormalize: true },
        )
        .map_err(|e| e.0.to_string())?;
        for slot in 0..ROUTES {
            let route = row * ROUTES + slot;
            let id = outputs[0][route];
            let weight = f32::from_bits(outputs[1][route] as u32);
            if id < 0 || id as usize != selection.expert_ids[slot] {
                return Err(format!(
                    "row {row} route {slot}: expert {id} against oracle {}",
                    selection.expert_ids[slot]
                ));
            }
            if (f64::from(weight) - selection.weights[slot]).abs() > 2e-6 {
                return Err(format!(
                    "row {row} route {slot}: weight {weight} against oracle {}",
                    selection.weights[slot]
                ));
            }
        }
        selections.push(selection);
    }
    let Some(rank) = rank else {
        return Ok(());
    };
    let routes = kernel_core::moe_oracle::local_expert_routes(
        MoeQuery::LocalRoutes {
            experts: EXPERTS as u32,
            local_experts: local as u32,
            top_k: ROUTES as u32,
            capacity: rows.len() as u32,
        },
        &selections,
        rank as usize,
    )
    .map_err(|e| e.0.to_string())?;
    let block = ROUTE_BLOCK as usize;
    let mut counts = vec![0; local];
    let mut offsets = Vec::new();
    let mut sorted = Vec::new();
    for (expert, count) in counts.iter_mut().enumerate() {
        offsets.push(sorted.len() as i32);
        for (&route, &route_expert) in routes.routes.iter().zip(&routes.experts) {
            if route_expert == expert {
                sorted.push(route as i32);
                *count += 1;
            }
        }
        sorted.resize(sorted.len().div_ceil(block) * block, -1);
    }
    offsets.push(sorted.len() as i32);
    let active: Vec<i32> = (0..local)
        .filter(|&e| counts[e] > 0)
        .map(|e| e as i32)
        .collect();
    let (gpu_counts, gpu_offsets, gpu_sorted, gpu_active, padded) = (
        &outputs[2],
        &outputs[3],
        &outputs[4],
        &outputs[5],
        outputs[6][0],
    );
    if padded as usize != sorted.len()
        || *gpu_offsets != offsets
        || *gpu_sorted != sorted
        || *gpu_counts != counts
        || gpu_active[0] as usize != active.len()
        || gpu_active[1..=active.len()] != active[..]
    {
        return Err("local routes or their permutation differ from the oracle".into());
    }
    Ok(())
}

const DIGEST_ROUTER_ROWS: usize = 64;
const TIMED_ROUTER_ROWS: usize = 2048;
/// Launches per timed graph, so a launch's time resolves below the event
/// timer's tick.
const TIMED_LAUNCHES: usize = 64;

/// Times `TIMED_LAUNCHES` launches of `route` captured in one graph.
fn time_route(
    session: &Session,
    buffers: &RouteBuffers,
    route: &(RouteEntry, runtime_cuda::Function),
    rank: Option<u32>,
) -> LabResult<serde_json::Value> {
    let graph = session.capture(|stream| {
        for _ in 0..TIMED_LAUNCHES {
            buffers.launch(stream, route, rank)?;
        }
        Ok(())
    })?;
    let timing = kernel_lab::time_named(session, &route.0.symbol, 2, 15, |stream| {
        graph.launch(stream).map_err(|e| e.to_string())
    })?;
    let per_launch = |ms: f32| f64::from(ms) * 1000.0 / TIMED_LAUNCHES as f64;
    Ok(json!({"rows": buffers.rows, "launches": TIMED_LAUNCHES,
        "median_us": per_launch(timing.median_ms), "min_us": per_launch(timing.min_ms)}))
}

/// Every route entry of this file's model, the prefill router and, at TP4
/// and TP2 on every rank, each decode bucket's route and the parallel C32
/// route, against the f64 oracle on rows whose order turns on signed zeros.
/// First, every entry's outputs on zero-free rows are digested into a
/// receipt, so builds can be compared bit for bit where the declared order
/// cannot differ; with `CHOPINFER_KERNEL_BENCH=1`, the router at 2048 rows,
/// the C16 route and the parallel C32 route are timed on skewed logits that
/// route to the timed rank. Oracle failures are reported after both receipts.
#[test]
fn routes_rank_signed_zeros_as_the_oracle() {
    gpu_test(512 << 20, |session| {
        let catalogs = [catalog(session, 4)?, catalog(session, 2)?];
        let router = catalogs[0].router.as_ref().ok_or("prefill router")?;
        let mut digests = BTreeMap::new();
        let buffers = RouteBuffers::new(session, DIGEST_ROUTER_ROWS, 128)?;
        for pattern in ZERO_FREE {
            let rows: Vec<_> = (0..DIGEST_ROUTER_ROWS)
                .map(|row| zero_free_row(pattern, row))
                .collect();
            buffers.fill(session, &rows)?;
            buffers.launch(&session.stream, router, None)?;
            digests.insert(
                format!("{} {pattern}", router.0.symbol),
                json!(buffers.digest(session, false)?),
            );
        }
        for (catalog, ranks) in catalogs.iter().zip([4_u32, 2]) {
            for route in &catalog.decode {
                let capacity = route.0.tokens_capacity as usize;
                let buffers = RouteBuffers::new(session, capacity, catalog.local)?;
                for rank in 0..ranks {
                    for pattern in ZERO_FREE {
                        let rows: Vec<_> = (0..capacity)
                            .map(|row| zero_free_row(pattern, row))
                            .collect();
                        buffers.fill(session, &rows)?;
                        buffers.launch(&session.stream, route, Some(rank))?;
                        digests.insert(
                            format!("{} rank {rank} {pattern}", route.0.symbol),
                            json!(buffers.digest(session, true)?),
                        );
                    }
                }
            }
        }
        let count = digests.len();
        session.receipt(
            "route_signed_zero_digests",
            BTreeMap::from([
                ("digests".into(), json!(digests)),
                ("count".into(), json!(count)),
            ]),
        )?;

        if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() == Ok("1") {
            let mut timings = BTreeMap::new();
            let rows: Vec<_> = (0..TIMED_ROUTER_ROWS)
                .map(|row| zero_free_row("skewed", row))
                .collect();
            let buffers = RouteBuffers::new(session, TIMED_ROUTER_ROWS, 128)?;
            buffers.fill(session, &rows)?;
            timings.insert(
                router.0.symbol.clone(),
                time_route(session, &buffers, router, None)?,
            );
            let catalog = &catalogs[0];
            for route in &catalog.decode {
                let capacity = route.0.tokens_capacity as usize;
                if capacity != 16 && route.0.parallel.is_none() {
                    continue;
                }
                let buffers = RouteBuffers::new(session, capacity, catalog.local)?;
                let rows: Vec<_> = (0..capacity)
                    .map(|row| zero_free_row("skewed", row))
                    .collect();
                buffers.fill(session, &rows)?;
                buffers.launch(&session.stream, route, Some(1))?;
                let routed: i32 = buffers.outputs(session, true)?[2].iter().sum();
                if routed == 0 {
                    return Err(format!("{}: the timed rank has no routes", route.0.symbol));
                }
                let mut timing = time_route(session, &buffers, route, Some(1))?;
                timing["rank"] = json!(1);
                timing["local_routes"] = json!(routed);
                timings.insert(route.0.symbol.clone(), timing);
            }
            session.receipt(
                "route_signed_zero_timing",
                BTreeMap::from([("timings".into(), json!(timings))]),
            )?;
        }

        let mut failures = replay::Failures::default();
        let rows = signed_zero_rows();
        let buffers = RouteBuffers::new(session, rows.len(), 128)?;
        buffers.fill(session, &rows)?;
        buffers.launch(&session.stream, router, None)?;
        failures.check(
            &router.0.symbol,
            check_against_oracle(&rows, &buffers.outputs(session, false)?, 128, None),
        );
        for (catalog, ranks) in catalogs.iter().zip([4_u32, 2]) {
            for route in &catalog.decode {
                let capacity = route.0.tokens_capacity as usize;
                let buffers = RouteBuffers::new(session, capacity, catalog.local)?;
                for rank in 0..ranks {
                    for start in (0..rows.len()).step_by(capacity) {
                        let chunk: Vec<_> = (start..start + capacity)
                            .map(|i| rows[i % rows.len()].clone())
                            .collect();
                        buffers.fill(session, &chunk)?;
                        buffers.launch(&session.stream, route, Some(rank))?;
                        failures.check(
                            &format!(
                                "{} rank {rank} rows {start}..{}",
                                route.0.symbol,
                                start + capacity
                            ),
                            check_against_oracle(
                                &chunk,
                                &buffers.outputs(session, true)?,
                                catalog.local,
                                Some(rank),
                            ),
                        );
                    }
                }
            }
        }
        session.receipt(
            "route_signed_zeros",
            BTreeMap::from([
                ("rows".into(), json!(rows.len())),
                ("correct".into(), json!(failures.is_empty())),
            ]),
        )?;
        failures.finish()
    })
    .expect("routes rank signed zeros as the oracle does");
}
