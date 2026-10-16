//! KV offload copy interference, kernels only. Serving steps are not covered here.
#[allow(dead_code)]
#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/kv_offload.rs"]
mod kvo;
#[path = "common/kv_offload_mimo.rs"]
mod mimo;
#[path = "common/kv_offload_qwen.rs"]
mod qwen;
use kernel_lab::{LabResult, Session, gpu_test_multi, launch};
use runtime_cuda::{GraphExec, KernelArgument, LaunchConfig, PinnedHostBuffer};
use serde_json::json;
use std::time::{Duration, Instant};
const CALLS: usize = 512;

fn measure(sessions: &[Session], name: &str, rows: usize, graphs: &[GraphExec]) -> LabResult<()> {
    let page = if name.starts_with("qwen") {
        kvo::PAGES[2].1
    } else {
        kvo::PAGES[0].1
    };
    let api = kvo::Api::open()?;
    let mut arenas = sessions
        .iter()
        .map(|s| kvo::Arena::new(s, page, 16))
        .collect::<LabResult<Vec<_>>>()?;
    for (s, a) in sessions.iter().zip(&mut arenas) {
        a.verify(s, &api, true)?;
    }
    let modules = sessions
        .iter()
        .map(|s| s.module("probe_sm120"))
        .collect::<LabResult<Vec<_>>>()?;
    let gates = modules
        .iter()
        .map(|m| m.function("chop_probe_gate_v1").map_err(|e| e.to_string()))
        .collect::<LabResult<Vec<_>>>()?;
    let flags = sessions
        .iter()
        .map(|s| s.allocate(4))
        .collect::<LabResult<Vec<_>>>()?;
    let releases = sessions
        .iter()
        .map(|s| s.context.stream().map_err(|e| e.to_string()))
        .collect::<LabResult<Vec<_>>>()?;
    // Allocate release staging while every stream is idle. A page-locked
    // allocation after the gates launch can wait for those same gates.
    let release_words = sessions
        .iter()
        .map(|s| {
            let mut word = PinnedHostBuffer::new(&s.context, 4).map_err(|e| e.to_string())?;
            word.as_mut_slice().copy_from_slice(&1_u32.to_ne_bytes());
            Ok(word)
        })
        .collect::<LabResult<Vec<_>>>()?;
    let baseline = kernel_lab::time_multi_graph(sessions, graphs, CALLS, 2, 7)?;
    // Sustain traffic well beyond the idle graph duration, including a
    // substantial slowdown. Keep the arena bounded and reuse it in order.
    let chunks = 256_usize.max(
        (f64::from(baseline.critical_path.median_ms) * 1e-3 * 57e9 * 8.0 / arenas[0].bytes() as f64)
            .ceil() as usize,
    );
    if chunks > 32768 {
        return Err("baseline requires more than the bounded copy budget".into());
    }
    eprintln!(
        "KVO {name} rows {rows}: baseline {} ms, {chunks} chunks",
        baseline.critical_path.median_ms
    );

    for h2d in [true, false] {
        // These are nominal per-rank rate targets. Receipts record achieved
        // rates, including DMA and host submission gaps.
        for cap in [0_u64, 10, 25, 50] {
            let mut all_samples = Vec::new();
            let mut traffic = Vec::new();
            for round in 0..9 {
                for (s, flag) in sessions.iter().zip(&flags) {
                    flag.copy_from_bytes(&0_u32.to_ne_bytes(), &s.stream)
                        .map_err(|e| e.to_string())?;
                }
                // Event creation, like pinned allocation, stays before every gate.
                let times = sessions
                    .iter()
                    .map(|s| {
                        Ok((
                            s.context.timed_event().map_err(|e| e.to_string())?,
                            s.context.timed_event().map_err(|e| e.to_string())?,
                            s.context.timed_event().map_err(|e| e.to_string())?,
                            s.context.timed_event().map_err(|e| e.to_string())?,
                        ))
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                let scheduled = (|| -> LabResult<()> {
                    for (r, s) in sessions.iter().enumerate() {
                        let a = &mut arenas[r];
                        gates[r]
                            .launch(
                                &a.stream,
                                launch::simple_config(
                                    (1, 1, 1),
                                    (32, 1, 1),
                                    0,
                                    LaunchConfig::NO_CLUSTER,
                                    false,
                                )?,
                                &[KernelArgument::Buffer(&flags[r])],
                            )
                            .map_err(|e| e.to_string())?;
                        let (copy_start, _, comm_start, comm_end) = &times[r];
                        copy_start.record(&a.stream).map_err(|e| e.to_string())?;
                        // Submit only the first chunk behind the gate. The
                        // host paces subsequent chunks after releasing all ranks.
                        a.enqueue(&api, true, h2d)?;
                        s.stream.wait_event(copy_start).map_err(|e| e.to_string())?;
                        comm_start.record(&s.stream).map_err(|e| e.to_string())?;
                        graphs[r].launch(&s.stream).map_err(|e| e.to_string())?;
                        comm_end.record(&s.stream).map_err(|e| e.to_string())?;
                    }
                    Ok(())
                })();
                // Release even after a scheduling error, so owners can drain.
                for ((flag, stream), word) in flags.iter().zip(&releases).zip(&release_words) {
                    flag.copy_from_pinned_at_async(0, word, 0, 4, stream)
                        .map_err(|e| e.to_string())?
                        .finish();
                }
                scheduled?;
                let pacing_start = Instant::now();
                for chunk in 1..chunks {
                    if cap != 0 {
                        let target = Duration::from_secs_f64(
                            chunk as f64 * arenas[0].bytes() as f64 / (cap as f64 * 1e9),
                        );
                        if let Some(wait) = target.checked_sub(pacing_start.elapsed()) {
                            std::thread::sleep(wait);
                        }
                    }
                    for a in &mut arenas {
                        a.enqueue(&api, true, h2d)?;
                    }
                }
                for (r, a) in arenas.iter().enumerate() {
                    times[r].1.record(&a.stream).map_err(|e| e.to_string())?;
                }
                let mut critical = 0.0_f64;
                let mut per_rank = Vec::new();
                for (r, (cs, ce, gs, ge)) in times.iter().enumerate() {
                    ce.synchronize().map_err(|e| e.to_string())?;
                    ge.synchronize().map_err(|e| e.to_string())?;
                    let copy_ms = f64::from(cs.elapsed_ms(ce).map_err(|e| e.to_string())?);
                    let graph_ms = f64::from(gs.elapsed_ms(ge).map_err(|e| e.to_string())?);
                    let graph_end_ms = f64::from(cs.elapsed_ms(ge).map_err(|e| e.to_string())?);
                    if graph_end_ms > copy_ms {
                        return Err(format!(
                            "{name} copies ended before all-reduce; increase traffic duration"
                        ));
                    }
                    critical = critical.max(graph_ms);
                    per_rank.push(json!({"rank":r,"copy_ms":copy_ms,"graph_ms":graph_ms,
                        "graph_end_from_copy_start_ms":graph_end_ms,
                        "achieved_gbps":arenas[r].bytes() as f64 * chunks as f64 / copy_ms / 1e6}));
                }
                if round >= 2 {
                    all_samples.push(critical);
                    traffic.push(per_rank);
                }
            }
            let median = kvo::median(&all_samples);
            let baseline_after = kernel_lab::time_multi_graph(sessions, graphs, CALLS, 2, 7)?;
            for s in sessions {
                kvo::record(
                    s,
                    "M3",
                    json!({"kernel":name,"rows":rows,"calls_per_graph":CALLS,
                    "direction":if h2d {"h2d"} else {"d2h"},"cap_gbps":if cap == 0 {None} else {Some(cap)},
                    "pacing":"host submits one page-list burst per rank at absolute bytes/cap intervals after releasing all rank gates; achieved rate includes submission and sleep overhead",
                    "page_bytes":page,"pages_per_chunk":16,"chunks":chunks,"layout":"permuted_even_pages",
                    "copy_streams_per_rank":1,"rounds":7,"warmup":2,
                    "baseline_before_ms":baseline.critical_path.median_ms,"baseline_after_ms":baseline_after.critical_path.median_ms,
                    "interference_critical_path_ms":all_samples,"traffic":traffic,
                    "all_reduce_us":median * 1000.0 / CALLS as f64,
                    "slowdown_ratio":median / f64::from(baseline.critical_path.median_ms),
                    "artifact":evidence::identity(if name.starts_with("qwen") {"comm_qwen38_decode"} else {"comm_sm120"})?,
                    "coverage":"Each rank's copy stream remains active through its full all-reduce graph; pacing includes idle gaps.",
                    "scope":"kernel-only, no serving decode ITL or prefill step qualification"}),
                )?;
            }
        }
    }
    Ok(())
}
#[test]
fn gated_release_uses_preallocated_staging_cpu() {
    let source = include_str!("kv_offload_interference.rs");
    let start = source
        .find(concat!("// Release even after", " a scheduling error"))
        .unwrap();
    let end = source[start..].find("let pacing_start").unwrap() + start;
    let release = &source[start..end];
    assert!(release.contains("copy_from_pinned_at_async"));
    assert!(!release.contains("copy_from_bytes"));
    assert!(!release.contains("PinnedHostBuffer::new"));
}

#[test]
fn m3_all_reduce_interference() {
    if !kvo::enabled() {
        return;
    }
    gpu_test_multi(512 << 20, |sessions| {
        let unique: std::collections::BTreeSet<_> = sessions.iter().map(|s| s.ordinal).collect();
        assert_eq!(unique.len(), 4, "four distinct GPUs required");
        kvo::exclusive(sessions)?;
        let mut failures = Vec::new();
        if let Err(error) = mimo::measure(sessions, |name, rows, graphs| {
            measure(sessions, name, rows, graphs)
        }) {
            failures.push(error);
        }
        if let Err(error) = qwen::measure(sessions, |name, rows, graphs| {
            measure(sessions, name, rows, graphs)
        }) {
            failures.push(error);
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    })
    .expect("M3")
    .expect("four GPUs required");
}
