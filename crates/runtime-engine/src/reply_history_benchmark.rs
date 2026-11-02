//! Explicit CPU-only paired harness. Existing prompt chains and synthetic input
//! tokens are shared setup; every new observer allocation/hash/probe is timed.
use super::*;
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy, Default)]
struct Sample {
    elapsed: f64,
    cpu: f64,
}
fn thread_cpu_ns() -> u64 {
    std::fs::read_to_string("/proc/thread-self/schedstat")
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap()
}
fn timed(mut operation: impl FnMut()) -> Sample {
    let cpu = thread_cpu_ns();
    let started = Instant::now();
    operation();
    let elapsed = started.elapsed().as_nanos() as f64;
    Sample {
        elapsed,
        cpu: thread_cpu_ns().saturating_sub(cpu) as f64,
    }
}
fn distribution(samples: &[Sample], cpu: bool) -> serde_json::Value {
    assert!(!samples.is_empty());
    let mut values = samples
        .iter()
        .map(|s| if cpu { s.cpu } else { s.elapsed })
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let at = |q: f64| values[((values.len() - 1) as f64 * q).ceil() as usize] / 1000.0;
    serde_json::json!({"samples": values.len(), "p50_us": at(0.5), "p95_us": at(0.95),
        "p99_us": at(0.99), "max_us": at(1.), "min_us": at(0.)})
}
fn subtract(a: Sample, b: Sample) -> Sample {
    Sample {
        elapsed: a.elapsed - b.elapsed,
        cpu: a.cpu - b.cpu,
    }
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    prompt: usize,
    reply: usize,
    block: usize,
    concurrency: usize,
    width: usize,
    samples: usize,
    mode: &'static str,
}

fn seed(case: Case, metrics: KvMetrics, tokens: &[u32]) -> Registry {
    let mut registry = Registry::new(metrics).unwrap();
    for index in 0..MAX_COMPLETED {
        let salt = (case.mode == "many_salts").then(|| (index as u64).to_le_bytes());
        let chain = PrefixChainV1::salted(
            case.block,
            Arc::from(&tokens[..case.prompt]),
            vec![],
            salt.as_ref().map(|s| s.as_slice()),
        )
        .unwrap();
        let id = RequestId::new(index as u64 + 1);
        registry.submit(id, &chain, case.reply);
        registry.append(id, &tokens[case.prompt..case.prompt + case.reply]);
        registry.engine_finish(id, FinishReason::Stop);
    }
    assert!(!registry.completed.is_empty());
    assert!(registry.bytes() <= MAX_BYTES);
    registry
}

fn cycle(
    registry: &mut Option<Mutex<Registry>>,
    chain: &PrefixChainV1,
    case: Case,
    serial: u64,
    phases: &mut [Vec<Sample>; 4],
    collect: bool,
) {
    let first = 1000 + serial * case.concurrency as u64;
    let lookup = timed(|| {
        for request in 0..case.concurrency {
            let id = RequestId::new(first + request as u64);
            if let Some(registry) = registry {
                let mut registry = registry.lock().unwrap();
                registry.submit(id, chain, case.reply);
                registry.defer(id);
                registry.prefill(id, 0, chain.tokens().len() - 1, 0);
            } else {
                black_box((id, chain));
            }
        }
    });
    let mut work = lookup;
    let output = [13_u32; 8];
    for offset in (0..case.reply).step_by(case.width) {
        let width = case.width.min(case.reply - offset);
        let last = offset + width == case.reply;
        let sample = timed(|| {
            for request in 0..case.concurrency {
                let id = RequestId::new(first + request as u64);
                if let Some(registry) = registry {
                    let mut guard = registry.lock().unwrap();
                    guard.append(id, &output[..width]);
                    if last {
                        guard.engine_finish(id, FinishReason::Length);
                    }
                    drop(guard);
                    for ordinal in offset..offset + width {
                        registry
                            .lock()
                            .unwrap()
                            .annotate(id, ordinal, ReplyRegion::Content);
                    }
                } else {
                    black_box((id, &output[..width]));
                }
            }
        });
        work.elapsed += sample.elapsed;
        work.cpu += sample.cpu;
        if collect {
            phases[1].push(sample);
        }
    }
    let finish = timed(|| {
        for request in 0..case.concurrency {
            let id = RequestId::new(first + request as u64);
            if let Some(registry) = registry {
                let mut registry = registry.lock().unwrap();
                registry.delivered(id, FinishReason::Length, case.reply);
            } else {
                black_box(id);
            }
        }
    });
    work.elapsed += finish.elapsed;
    work.cpu += finish.cpu;
    if collect {
        phases[0].push(lookup);
        phases[2].push(finish);
        phases[3].push(work);
    }
}

#[test]
#[ignore = "explicit release CPU timing; run in an open CPU window"]
fn reply_observer_cpu_cost() {
    assert!(!cfg!(debug_assertions), "run this harness with --release");
    let cases = [
        Case {
            name: "100k_c1_ar",
            prompt: 100_000,
            reply: 550,
            block: 64,
            concurrency: 1,
            width: 1,
            samples: 1000,
            mode: "match",
        },
        Case {
            name: "160k_c8_verify",
            prompt: 160_000,
            reply: 550,
            block: 64,
            concurrency: 8,
            width: 8,
            samples: 1000,
            mode: "match",
        },
        Case {
            name: "160k_c32_ar",
            prompt: 160_000,
            reply: 550,
            block: 64,
            concurrency: 32,
            width: 1,
            samples: 1000,
            mode: "miss",
        },
        Case {
            name: "160k_c32_verify64",
            prompt: 160_000,
            reply: 550,
            block: 64,
            concurrency: 32,
            width: 2,
            samples: 1000,
            mode: "near_end",
        },
        Case {
            name: "160k_c32_verify192",
            prompt: 160_000,
            reply: 550,
            block: 64,
            concurrency: 32,
            width: 6,
            samples: 1000,
            mode: "match",
        },
        Case {
            name: "160k_c8_salts",
            prompt: 160_000,
            reply: 550,
            block: 64,
            concurrency: 8,
            width: 8,
            samples: 1000,
            mode: "many_salts",
        },
        Case {
            name: "long_reply_c1",
            prompt: 262_144,
            reply: 16_384,
            block: 64,
            concurrency: 1,
            width: 8,
            samples: 1000,
            mode: "near_end",
        },
        Case {
            name: "byte_pressure_c8",
            prompt: 262_144,
            reply: 550,
            block: 4,
            concurrency: 8,
            width: 8,
            samples: 1000,
            mode: "match",
        },
        Case {
            name: "joint_pressure_c8",
            prompt: 28_200,
            reply: 550,
            block: 4,
            concurrency: 8,
            width: 8,
            samples: 1000,
            mode: "match",
        },
    ];
    let mut passed = true;
    for case in cases {
        let mut tokens = vec![13; case.prompt + case.reply + 1];
        let metrics = KvMetrics::default();
        let registry = seed(case, metrics.clone(), &tokens);
        let seeded_entries = registry.completed.len();
        let seeded_bytes = registry.bytes();
        if case.mode == "miss" {
            tokens[0] = 17;
        }
        if case.mode == "near_end" {
            tokens[case.prompt + case.reply - 3] = 17;
        }
        let salt = (case.mode == "many_salts").then(|| 128_u64.to_le_bytes());
        let chain = PrefixChainV1::salted(
            case.block,
            Arc::from(tokens),
            vec![],
            salt.as_ref().map(|s| s.as_slice()),
        )
        .unwrap();
        let mut enabled = Some(Mutex::new(registry));
        let mut disabled = None;
        let mut on: [Vec<Sample>; 4] = std::array::from_fn(|_| Vec::new());
        let mut off: [Vec<Sample>; 4] = std::array::from_fn(|_| Vec::new());
        for serial in 0..case.samples + 8 {
            // Reverse paired order each sample to expose drift.
            if serial % 2 == 0 {
                cycle(
                    &mut enabled,
                    &chain,
                    case,
                    serial as u64,
                    &mut on,
                    serial >= 8,
                );
                cycle(
                    &mut disabled,
                    &chain,
                    case,
                    serial as u64,
                    &mut off,
                    serial >= 8,
                );
            } else {
                cycle(
                    &mut disabled,
                    &chain,
                    case,
                    serial as u64,
                    &mut off,
                    serial >= 8,
                );
                cycle(
                    &mut enabled,
                    &chain,
                    case,
                    serial as u64,
                    &mut on,
                    serial >= 8,
                );
            }
        }
        let differences: [Vec<Sample>; 4] = std::array::from_fn(|i| {
            assert_eq!(on[i].len(), off[i].len());
            on[i]
                .iter()
                .zip(&off[i])
                .map(|(&a, &b)| subtract(a, b))
                .collect()
        });
        // A coincident cycle can complete a batch and admit the next batch.
        let coincident = differences[0]
            .iter()
            .zip(&differences[2])
            .map(|(a, b)| Sample {
                elapsed: a.elapsed + b.elapsed,
                cpu: a.cpu + b.cpu,
            })
            .collect::<Vec<_>>();
        let variability = distribution(&coincident, false);
        let p99 = variability["p99_us"].as_f64().unwrap();
        let maximum = variability["max_us"].as_f64().unwrap();
        // Noisy tails cannot prove the default-on gate. Report the
        // measured tails regardless, and choose off unless every case resolves.
        let resolved = case.samples >= 1000 && maximum.max(p99) < 140.;
        passed &= resolved;
        let phases = ["lookup_prefill_cycle", "commit_cycle", "finalization_cycle", "request_batch_total"]
            .into_iter().enumerate().map(|(i,name)| (name.to_owned(), serde_json::json!({
                "incremental_elapsed": distribution(&differences[i], false),
                "incremental_cpu": distribution(&differences[i], true),
                "enabled_elapsed": distribution(&on[i], false), "disabled_elapsed": distribution(&off[i], false),
            }))).collect::<serde_json::Map<_,_>>();
        let total_ns: f64 = differences[3].iter().map(|s| s.elapsed).sum();
        println!(
            "{}",
            serde_json::json!({"observer_cpu_case": case.name, "prompt": case.prompt,
            "reply": case.reply, "block": case.block, "concurrency": case.concurrency,
            "accepted_width": case.width, "mode": case.mode, "seeded_entries": seeded_entries,
            "seeded_bytes": seeded_bytes, "final_bytes": enabled.as_ref().unwrap().lock().unwrap().bytes(),
            "phases": phases, "coincident_cycle_incremental_elapsed": variability,
            "elapsed_us_per_committed_token": total_ns/1000./(case.samples*case.concurrency*case.reply) as f64,
            "elapsed_us_per_completed_request": total_ns/1000./(case.samples*case.concurrency) as f64,
            "default_on_resolved": resolved,
            "method": "paired alternating CPU-only observer calls; thread schedstat CPU and Instant elapsed; prompt chain shared with disabled baseline; eight warmup batches"})
        );
        assert!(enabled.as_ref().unwrap().lock().unwrap().bytes() <= MAX_BYTES);
    }
    println!(
        "observer_default_on_gate={}",
        if passed {
            "passed"
        } else {
            "unresolved_or_exceeded_default_off"
        }
    );
}
