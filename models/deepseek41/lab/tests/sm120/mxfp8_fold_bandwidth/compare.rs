use super::*;
use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldStreamBuffers};
use kernel_harness::launch::Bindings;
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::{mxfp8_fold_decode::Entry as OldEntry, mxfp8_fold_direct::Entry};
use previous::{Weights, pattern, run, scales};
use runtime_cuda::Function;

fn functions(session: &Session, calls: &[Launch]) -> LabResult<Vec<Function>> {
    struct Cached(runtime_cuda::Module);
    let cache = session.state(|| Ok(Cached(session.load_cubin(cubin()?.1)?)))?;
    calls
        .iter()
        .map(|call| {
            let f = if call.symbol.contains("_mxfp8_direct_") {
                cache.0.function(&call.symbol).map_err(|e| e.to_string())?
            } else {
                previous::functions(session, std::slice::from_ref(call))?.remove(0)
            };
            if call.plan.dynamic_shared_bytes > 0 {
                f.set_max_dynamic_shared_memory(call.plan.dynamic_shared_bytes)
                    .map_err(|e| e.to_string())?;
            }
            Ok(f)
        })
        .collect()
}

pub(super) fn case(
    session: &Session,
    entry: &OldEntry,
    rows: u32,
    candidate: &Entry,
    weights: &Weights,
    timed: bool,
    profile: bool,
) -> LabResult<serde_json::Value> {
    let q = entry.query;
    let ranks = previous::topologies(entry);
    let baseline_query = registration::routed_query(q, rows);
    let baseline = catalog::register(
        registration::register(Sm120::new(), &FLASH, ranks[0])?,
        &FLASH,
        ranks[0],
    )?;
    let has_stream = catalog::qualified(&FLASH, baseline_query, rows, entry.paired);
    let old_stream = if has_stream {
        Some(
            baseline
                .mxfp8_fold_gemm_stream(&baseline_query, entry.paired, Need::rows(rows))
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let old_single = if !has_stream && !entry.paired {
        Some(
            baseline
                .mxfp8_fold_gemm(&baseline_query, Need::rows(rows))
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let old_pair = if !has_stream && entry.paired {
        Some(
            baseline
                .mxfp8_fold_gemm_pair(&baseline_query, Need::rows(rows))
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let baseline_workspace = old_stream
        .as_ref()
        .map(|o| o.info().workspace)
        .or_else(|| old_single.as_ref().map(|o| o.info().workspace))
        .or_else(|| old_pair.as_ref().map(|o| o.info().workspace))
        .unwrap();
    let baseline_state = old_stream
        .as_ref()
        .and_then(|o| o.info().bytes_of("state"))
        .unwrap_or(4);
    let mut candidate = candidate.clone();
    candidate.packed.ticket.query = baseline_query;
    let sizes = entry.legacy().map_err(|e| e.to_string())?.bytes(rows);
    const OFFSET: usize = 3;
    let initial_status: Vec<u8> = (0..rows as usize + OFFSET + 2)
        .flat_map(|i| {
            (if i < OFFSET || i >= OFFSET + rows as usize {
                0x4000_u32
            } else {
                0
            })
            .to_ne_bytes()
        })
        .collect();
    let mut images = vec![
        pattern(sizes.activations, 1),
        scales(sizes.activation_scales, 1),
        vec![0; 4],
        vec![0; 4],
        vec![0xa5; sizes.output as usize],
        rows.to_ne_bytes().to_vec(),
        initial_status.clone(),
        vec![0xcd; 4],
        vec![0; 4],
        vec![0; 4],
        vec![0xa5; sizes.output as usize],
        vec![0xa5; sizes.output as usize],
        vec![0xa5; sizes.output as usize],
        initial_status,
        vec![0xcd; baseline_workspace.max(4) as usize],
        vec![0xcd; candidate.packed.packed_bytes() as usize],
        vec![
            0xcd;
            if entry.paired {
                candidate.packed.packed_bytes() as usize
            } else {
                4
            }
        ],
        vec![0xcd; candidate.state_bytes(rows).map_err(|e| e.to_string())? as usize],
        vec![0xcd; baseline_state as usize],
        vec![0; candidate.stamp_bytes(rows).map_err(|e| e.to_string())? as usize],
    ];
    images.push(images[0].clone());
    let buffers = images
        .iter()
        .map(|b| session.allocate(b.len()))
        .collect::<LabResult<Vec<_>>>()?;
    let lengths: Vec<_> = images.iter().map(|b| b.len() as u64).collect();
    let view = |i: usize| View {
        handle: i as u64 + 1,
        bytes: [2, 3, 8, 9]
            .iter()
            .position(|&j| j == i)
            .map(|j| weights.buffers[j].byte_len() as u64)
            .unwrap_or(lengths[i]),
    };
    let first = Mxfp8FoldGemmBuffers {
        activations: view(0),
        activation_scales: view(1),
        weights: view(2),
        weight_scales: view(3),
        output: view(4),
        live_rows: view(5),
        status: view(6),
        status_row_offset: OFFSET as u64,
    };
    let second = Mxfp8FoldGemmBuffers {
        weights: view(8),
        weight_scales: view(9),
        output: view(10),
        ..first
    };
    let new_first = Mxfp8FoldGemmBuffers {
        output: view(11),
        status: view(13),
        ..first
    };
    let new_second = Mxfp8FoldGemmBuffers {
        output: view(12),
        status: view(13),
        ..second
    };
    let stateful = Mxfp8FoldStreamBuffers {
        first: new_first,
        second: entry.paired.then_some(new_second),
        packed_weights: [view(15), if entry.paired { view(16) } else { View::EMPTY }],
        state: view(17),
    };
    let old_stateful = Mxfp8FoldStreamBuffers {
        first,
        second: entry.paired.then_some(second),
        state: view(18),
        ..stateful
    };
    let old_calls = if let Some(op) = &old_stream {
        op.launches(&old_stateful, view(14), Counts::rows(rows))
    } else if let Some(op) = old_single {
        op.launches(&first, view(14), Counts::rows(rows))
    } else {
        old_pair
            .unwrap()
            .launches(&[first, second], view(14), Counts::rows(rows))
    }
    .map_err(|e| e.to_string())?;
    let initialize = [candidate.initialize(&stateful).map_err(|e| e.to_string())?];
    let baseline_init = old_stream
        .as_ref()
        .map(|op| op.initialize(&old_stateful))
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    let new_calls = [candidate
        .launch(&stateful, rows, None)
        .map_err(|e| e.to_string())?];
    let clock_calls = [candidate
        .launch(&stateful, rows, Some(view(19)))
        .map_err(|e| e.to_string())?];
    let old_functions = functions(session, &old_calls)?;
    let new_functions = functions(session, &new_calls)?;
    let init_functions = functions(session, &initialize)?;
    let baseline_init_functions = functions(session, &baseline_init)?;
    let clock_functions = functions(session, &clock_calls)?;
    let mut bindings = Bindings {
        buffers: buffers
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u64 + 1, b))
            .collect(),
    };
    for (i, b) in [2, 3, 8, 9].into_iter().zip(&weights.buffers) {
        bindings.buffers.insert(i + 1, b);
    }
    let copy = |i: usize, bytes: &[u8]| {
        buffers[i]
            .copy_from_bytes(bytes, &session.stream)
            .map_err(|e| e.to_string())
    };
    let upload = |images: &[Vec<u8>]| -> LabResult<()> {
        for (i, bytes) in images.iter().enumerate() {
            if i >= 15 || [2, 3, 8, 9].contains(&i) {
                continue;
            }
            copy(i, bytes)?;
        }
        Ok(())
    };
    let read = |i: usize| -> LabResult<Vec<u8>> {
        let mut bytes = vec![0; buffers[i].byte_len()];
        buffers[i]
            .copy_to_bytes(&mut bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    };
    let equal = || -> LabResult<()> {
        if read(4)? != read(11)?
            || read(6)? != read(13)?
            || (entry.paired && read(10)? != read(12)?)
        {
            return Err(format!("bitwise mismatch {} rows={rows}", candidate.symbol));
        }
        if read(17)?.iter().any(|&b| b != 0) {
            return Err("direct fold state did not self-reset".into());
        }
        if has_stream && read(18)?.iter().any(|&b| b != 0) {
            return Err("baseline stream state did not self-reset".into());
        }
        Ok(())
    };
    upload(&images)?;
    for (i, bytes) in images.iter().enumerate().skip(15) {
        copy(i, bytes)?;
    }
    run(
        session,
        &session.stream,
        &initialize,
        &init_functions,
        &bindings,
    )?;
    run(
        session,
        &session.stream,
        &baseline_init,
        &baseline_init_functions,
        &bindings,
    )?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    let graph = session.capture(|s| run(session, s, &new_calls, &new_functions, &bindings))?;
    for live in [0, 1, rows.saturating_sub(1), rows] {
        images[5] = live.to_ne_bytes().to_vec();
        upload(&images)?;
        run(
            session,
            &session.stream,
            &old_calls,
            &old_functions,
            &bindings,
        )?;
        for _ in 0..2 {
            for i in [11, 12, 13] {
                copy(i, &images[i])?;
            }
            graph.launch(&session.stream).map_err(|e| e.to_string())?;
            equal()?;
        }
        run(
            session,
            &session.stream,
            &new_calls,
            &new_functions,
            &bindings,
        )?;
        equal()?;
    }
    let at = ((rows - 1) * q.activation_row_stride
        + (q.groups - 1) * q.activation_group_stride
        + q.k / 2) as usize;
    let saved = (images[0][at], images[1][at / 32]);
    for (code, scale, prior) in [
        (127, 127, 0),
        (56, 255, 0),
        (126, 254, 0),
        (1, 0, 0),
        (56, 127, 0x80),
        (126, 127, 0),
        (127, 3, 0),
        (255, 246, 0),
        (1, 2, 0),
    ] {
        images[0][at] = code;
        images[1][at / 32] = scale;
        let pos = (OFFSET + rows as usize - 1) * 4;
        for i in [6, 13] {
            images[i][pos..pos + 4].copy_from_slice(&u32::to_ne_bytes(prior));
        }
        upload(&images)?;
        run(
            session,
            &session.stream,
            &old_calls,
            &old_functions,
            &bindings,
        )?;
        run(
            session,
            &session.stream,
            &new_calls,
            &new_functions,
            &bindings,
        )?;
        equal()?;
        for i in [11, 12, 13] {
            copy(i, &images[i])?;
        }
        graph.launch(&session.stream).map_err(|e| e.to_string())?;
        equal()?;
    }
    images[0][at] = saved.0;
    images[1][at / 32] = saved.1;
    let pos = (OFFSET + rows as usize - 1) * 4;
    for i in [6, 13] {
        images[i][pos..pos + 4].fill(0);
    }
    upload(&images)?;
    run(
        session,
        &session.stream,
        &old_calls,
        &old_functions,
        &bindings,
    )?;
    if profile {
        previous::concurrent::check(
            session,
            previous::concurrent::Case {
                images: &images,
                buffers: &buffers,
                bindings: &bindings,
                calls: &new_calls,
                functions: &new_functions,
                baseline_calls: &old_calls,
                baseline_functions: &old_functions,
                paired: entry.paired,
                persistent: Some((&initialize, &init_functions, 17)),
            },
        )?;
        upload(&images)?;
        run(
            session,
            &session.stream,
            &old_calls,
            &old_functions,
            &bindings,
        )?;
        let publish = Launch {
            symbol: kernels_sm120::gemm::mxfp8_fold_direct::PUBLISH.into(),
            plan: LaunchPlan {
                grid: [1, 1, 1],
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1, 1, 1],
                pdl: true,
            },
            arguments: ArgumentList(vec![
                Argument::DeviceBuffer(view(20).handle),
                Argument::DeviceBuffer(view(0).handle),
                Argument::U64(sizes.activations),
                Argument::DeviceBuffer(view(5).handle),
                Argument::U32(rows),
                Argument::U32(100000),
            ]),
            rows,
            collective: false,
            persistent: false,
        };
        let producer_calls = [publish];
        let producer_functions = functions(session, &producer_calls)?;
        copy(0, &vec![127; images[0].len()])?;
        copy(5, &0_u32.to_ne_bytes())?;
        let dependent = session.capture(|s| {
            run(session, s, &producer_calls, &producer_functions, &bindings)?;
            run(session, s, &new_calls, &new_functions, &bindings)
        })?;
        dependent
            .launch(&session.stream)
            .map_err(|e| e.to_string())?;
        equal()?;
    }
    // Poison outputs before the diagnostic variant, so missing stores
    // cannot inherit a successful normal result.
    copy(11, &images[11])?;
    copy(12, &images[12])?;
    copy(19, &images[19])?;
    run(
        session,
        &session.stream,
        &clock_calls,
        &clock_functions,
        &bindings,
    )?;
    equal()?;
    let stamp_bytes = read(19)?;
    let mut cycles = [0_u64; 5];
    let mut maximum = [0_u64; 5];
    let mut last = 0_u64;
    for row in stamp_bytes.chunks_exact(64) {
        let values: Vec<_> = row
            .chunks_exact(8)
            .map(|x| u64::from_ne_bytes(x.try_into().unwrap()))
            .collect();
        if values[7] != 1 {
            return Err("missing direct fold clock record".into());
        }
        last += values[5];
        for i in 0..5 {
            cycles[i] += values[i];
            maximum[i] = maximum[i].max(values[i]);
        }
    }
    if last != u64::from(rows.div_ceil(16)) {
        return Err("incorrect number of final row-tile arrivals".into());
    }
    let mut record = json!({"n":q.n,"k":q.k,"groups":q.groups,"paired":entry.paired,"rows":rows,"ranks":ranks,
        "variant":candidate.schedule.name(),"bitwise":true,"clock_bitwise":true,"boundary_replays":8,
        "admission_cases":9,"admission_graph_replays":9,"concurrent_replays":if profile {3}else{0},"pdl_producer":profile,
        "baseline_stream":has_stream,"baseline_symbols":old_calls.iter().map(|c|&c.symbol).collect::<Vec<_>>(),
        "candidate_symbols":new_calls.iter().map(|c|&c.symbol).collect::<Vec<_>>(),
        "baseline_grid":old_calls.iter().map(|c|c.plan.grid).collect::<Vec<_>>(),
        "candidate_grid":new_calls[0].plan.grid,"candidate_threads":new_calls[0].plan.block[0],
        "candidate_shared_bytes":new_calls[0].plan.dynamic_shared_bytes,
        "state_bytes":candidate.state_bytes(rows).map_err(|e|e.to_string())?,"global_partial_bytes":0,
        "trace_replays":{"baseline":14,"candidate":31,"boundary":8,"admission":9,"warmup_per_mode":2,"rounds_per_mode":5},
        "clock":{"phase_order":["weight_issue","pdl_wait","admission","arithmetic","completion"],
            "summed_cta_cycles":cycles,"maximum_cta_cycles":maximum,
            "scope":"diagnostic lane zero; not exclusive wall time"}});
    if !timed {
        return Ok(record);
    }
    let old_graph = session.capture(|s| run(session, s, &old_calls, &old_functions, &bindings))?;
    let old_ranges = [
        if has_stream { view(15) } else { view(2) },
        view(3),
        if entry.paired {
            if has_stream { view(16) } else { view(8) }
        } else {
            View::EMPTY
        },
        if entry.paired { view(9) } else { View::EMPTY },
    ];
    let new_ranges = [
        view(15),
        view(3),
        if entry.paired { view(16) } else { View::EMPTY },
        if entry.paired { view(9) } else { View::EMPTY },
    ];
    let prefetches = [old_ranges, new_ranges].map(|ranges| {
        kernels_sm120::gemm::mxfp8_fold_stream::prefetch_launch(ranges).map_err(|e| e.to_string())
    });
    let [a, b] = prefetches;
    let prefetched = [a?, b?];
    let prefetch_functions = functions(session, &prefetched)?;
    let mode = kernel_harness::eviction::EvictionMode::Read;
    let bytes = (sizes.weights + sizes.weight_scales) * if entry.paired { 2 } else { 1 };
    for hot in [false, true] {
        let key = if hot { "prefetched" } else { "cold" };
        let label = format!("fold-bandwidth-{}-r{rows}-{key}", candidate.symbol);
        let paired = kernel_harness::with_timing_label(&label, || {
            kernel_harness::time_ab_prepared(
                session,
                2,
                5,
                [mode, mode],
                |arm, s| {
                    if hot {
                        run(
                            session,
                            s,
                            &prefetched[arm..arm + 1],
                            &prefetch_functions[arm..arm + 1],
                            &bindings,
                        )
                    } else {
                        Ok(())
                    }
                },
                |arm, s| {
                    if arm == 0 {
                        old_graph.launch(s).map_err(|e| e.to_string())
                    } else {
                        graph.launch(s).map_err(|e| e.to_string())
                    }
                },
            )
        })?;
        record[key] = json!({"baseline_us":paired.arms[0].median_ms*1000.0,
            "candidate_us":paired.arms[1].median_ms*1000.0,"paired_ratio":paired.paired_ratio,
            "rounds":5,"label":label,"no_regression":paired.paired_ratio<=1.03,
            "effective_tbps":bytes as f64/(f64::from(paired.arms[1].median_ms)*1.0e9),
            "floor_ratio":f64::from(paired.arms[1].median_ms)*1000.0/(bytes as f64/1_800_000.0)});
    }
    record["weight_bytes"] = json!(bytes);
    record["floor_us"] = json!(bytes as f64 / 1_800_000.0);
    Ok(record)
}
