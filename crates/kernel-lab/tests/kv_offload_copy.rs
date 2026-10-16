//! M1 and M2. Opt-in only, with a bounded pinned footprint below 4 GiB.
#[path = "common/kv_offload.rs"]
mod kvo;
use kernel_lab::{LabResult, Session, gpu_test, gpu_test_multi};
use runtime_cuda::PinnedHostBuffer;
use serde_json::json;
use std::time::Instant;

fn proc_file(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| format!("unavailable: {e}"))
}

#[test]
fn m1_pinned_allocation() {
    if !kvo::enabled() {
        return;
    }
    gpu_test(64 << 20, |s| {
        kvo::exclusive(std::slice::from_ref(s))?;
        let limits = proc_file("/proc/self/limits");
        let mut buffers = Vec::new();
        for bytes in [64 << 20, 256 << 20, 1 << 30, 1 << 30] {
            let before = proc_file("/proc/self/status");
            let start = Instant::now();
            let result = PinnedHostBuffer::new(&s.context, bytes);
            let allocation_ms = start.elapsed().as_secs_f64() * 1000.0;
            let error = result.as_ref().err().map(ToString::to_string);
            let mut touch_ms = None;
            if let Ok(mut buffer) = result {
                let start = Instant::now();
                buffer.as_mut_slice().fill(0x5a);
                touch_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                buffers.push(buffer);
            }
            kvo::record(s, "M1", json!({"allocation_bytes":bytes, "allocation_ms":allocation_ms,
                "success":error.is_none(), "error":error, "touch_ms":touch_ms,
                "cumulative_pinned_bytes":buffers.iter().map(PinnedHostBuffer::len).sum::<usize>(),
                "limits":limits, "status_before":before, "status_after":proc_file("/proc/self/status"),
                "meminfo":proc_file("/proc/meminfo"),
                "operator_item":"128 GiB requires at least 137438953472 bytes plus other locked allocations: shell ulimit -l 134217728 (KiB), systemd LimitMEMLOCK=137438953472 or infinity, docker --ulimit memlock=137438953472:137438953472 or -1:-1; provision host RAM plus 16 GiB reserve. This bounded probe does not qualify 128 GiB or the deploy container.",
                "memlock_interpretation":"Record VmLck and VmRSS deltas; CUDA pinning may not appear in VmLck. No limit exhaustion attempted."}))?;
            if let Some(error) = error { return Err(error); }
        }
        drop(buffers);
        kvo::record(s, "M1-release", json!({"status_after_free":proc_file("/proc/self/status")}))
    }).expect("M1").expect("GPU required");
}

fn copies(sessions: &[Session]) -> LabResult<()> {
    kvo::exclusive(sessions)?;
    let api = kvo::Api::open()?;
    let gates = kvo::CopyGate::new(sessions)?;
    // 16 pages exercise the proposed small chunks; 256 pages model longer
    // swap lists. Scatter permutes distinct even slots in a 2x-sized arena.
    let mut failures = Vec::new();
    for (name, page) in kvo::PAGES {
        for pages in [16, 256] {
            let mut arenas = sessions
                .iter()
                .map(|s| kvo::Arena::new(s, page, pages))
                .collect::<LabResult<Vec<_>>>()?;
            for scattered in [false, true] {
                let mut correct = true;
                for (s, a) in sessions.iter().zip(&mut arenas) {
                    if let Err(error) = a.verify(s, &api, scattered) {
                        eprintln!(
                            "{name} pages={pages} scattered={scattered} rank={}: {error}",
                            s.ordinal
                        );
                        correct = false;
                        failures.push(error);
                    }
                }
                // Mixed direction deliberately uses ONE stream per rank:
                // opposite directions serialize, as the proposed host tier does.
                for direction in ["h2d", "d2h", "alternating_h2d_d2h"] {
                    let mut samples = Vec::new();
                    let mut rank_samples = vec![Vec::new(); sessions.len()];
                    for round in 0..9 {
                        gates.reset()?;
                        let start = Instant::now();
                        let mut events = Vec::new();
                        let scheduled = (|| -> LabResult<()> {
                            for (rank, (s, a)) in sessions.iter().zip(&mut arenas).enumerate() {
                                gates.arm(rank, &a.stream)?;
                                let begin = s.context.timed_event().map_err(|e| e.to_string())?;
                                let end = s.context.timed_event().map_err(|e| e.to_string())?;
                                begin.record(&a.stream).map_err(|e| e.to_string())?;
                                for _ in 0..8 {
                                    a.enqueue(&api, scattered, direction != "d2h")?;
                                    if direction == "alternating_h2d_d2h" {
                                        a.enqueue(&api, scattered, false)?;
                                    }
                                }
                                end.record(&a.stream).map_err(|e| e.to_string())?;
                                events.push((begin, end));
                            }
                            Ok(())
                        })();
                        // Always release already-queued gates before propagating errors.
                        gates.release()?;
                        scheduled?;
                        for a in &arenas {
                            a.sync()?;
                        }
                        let wall = start.elapsed().as_secs_f64() * 1000.0;
                        if round >= 2 {
                            samples.push(wall);
                            for (rank, (start, end)) in events.iter().enumerate() {
                                rank_samples[rank].push(f64::from(
                                    start.elapsed_ms(end).map_err(|e| e.to_string())?,
                                ));
                            }
                        }
                    }
                    let bytes = page
                        * pages
                        * 8
                        * if direction == "alternating_h2d_d2h" {
                            2
                        } else {
                            1
                        };
                    for (rank, s) in sessions.iter().enumerate() {
                        kvo::record(
                            s,
                            "M2",
                            json!({"page_contract":name,"page_bytes":page,"pages":pages,
                            "layout":if scattered {"permuted_even_pages"} else {"contiguous_pages"},
                            "direction":direction,"ranks":sessions.len(),"rank":rank,
                            "rank_devices":sessions.iter().map(|s|s.ordinal).collect::<Vec<_>>(),
                            "copy_api":"cuMemcpyBatchAsync_v2","copy_streams_per_rank":1,
                            "calls_per_sample":if direction == "alternating_h2d_d2h" {16} else {8},
                            "bytes_per_rank_per_sample":bytes,"warmup":2,"rounds":7,
                            "host_wall_ms":samples,"rank_event_ms":rank_samples[rank],
                            "rank_gbps":bytes as f64 / kvo::median(&rank_samples[rank]) / 1e6,
                            "aggregate_host_gbps":bytes as f64 * sessions.len() as f64 / kvo::median(&samples) / 1e6,
                            "timing_note":"All rank queues are gated before release. Host wall includes submission, gate release skew and completion; CUDA events measure queued DMA. Opposite directions serialize on one stream.",
                            "correct":correct,"copy_destinations_complemented":true}),
                        )?;
                    }
                }
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}
#[test]
fn m2_copy_rates_single() {
    if !kvo::enabled() {
        return;
    }
    gpu_test(1 << 30, |s| copies(std::slice::from_ref(s)))
        .expect("M2 single")
        .expect("GPU required");
}
#[test]
fn m2_copy_rates_four() {
    if !kvo::enabled() {
        return;
    }
    gpu_test_multi(1 << 30, |sessions| {
        let unique: std::collections::BTreeSet<_> = sessions.iter().map(|s| s.ordinal).collect();
        assert_eq!(unique.len(), 4, "four distinct GPUs required");
        copies(sessions)
    })
    .expect("M2 four")
    .expect("four GPUs required");
}
#[test]
fn page_layout_contracts() {
    assert_eq!(
        kvo::PAGES.map(|(_, bytes)| bytes),
        [368640, 184320, 835584, 442368]
    );
    for pages in [16, 256] {
        let slots: std::collections::BTreeSet<_> = (0..pages)
            .map(|i| kvo::page_index(i, pages, true))
            .collect();
        assert_eq!(slots.len(), pages);
        assert!(slots.iter().all(|&i| i < 2 * pages && i % 2 == 0));
    }
}

#[test]
fn m1_memlock_probe() {
    if !kvo::enabled() {
        return;
    }
    gpu_test(64 << 20, |s| {
        kvo::exclusive(std::slice::from_ref(s))?;
        let bytes = 16 << 20;
        let start = Instant::now();
        let result = PinnedHostBuffer::new(&s.context, bytes);
        kvo::record(s, "M1-memlock", json!({"allocation_bytes":bytes,
            "allocation_ms":start.elapsed().as_secs_f64()*1000.0,
            "success":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),
            "limits":proc_file("/proc/self/limits"),"status_while_allocated":proc_file("/proc/self/status"),
            "interpretation":"Runner sets soft RLIMIT_MEMLOCK to 1 MiB in this subprocess only. A successful 16 MiB CUDA allocation shows this CUDA path bypasses that soft limit here, not that 128 GiB or a deploy container is qualified."}))
    }).expect("M1 memlock").expect("GPU required");
}
