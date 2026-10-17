//! `--list`: one discovery pass, two polls one interval apart, then a table
//! or a `chopinfer.top.list.v1` JSON document. This host's GPUs are read at
//! each poll, without GPM counters.

use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::discover::Discovery;
use crate::gpu::poll::{Report, Sampler, open};
use crate::gpu::{Gpus, Reading, Status};
use crate::instances::{Instance, Registry, Source, State};
use crate::poll::{Poll, poll};
use crate::targets::Target;
use crate::ui::widgets::{integer, pad, percent, right, sanitize};
use crate::ui::{Figures, state_group};

pub const SCHEMA: &str = "chopinfer.top.list.v1";

/// Finds and polls every instance, and reads this host's GPUs when `gpus`;
/// returns them and when they were read.
pub fn collect(
    targets: Vec<Target>,
    discovery: Option<Discovery>,
    notes: Vec<String>,
    interval: f64,
    bearer: Option<&str>,
    gpus: bool,
) -> (Registry, Gpus, f64) {
    let origin = Instant::now();
    let now = || origin.elapsed().as_secs_f64();
    let mut registry = Registry::new(interval);
    for note in notes {
        registry.note(
            now(),
            "chopinfer-top",
            note,
            crate::instances::Tone::Warning,
        );
    }
    for target in targets {
        registry.add_target(target, now());
    }
    if let Some(mut discovery) = discovery {
        let mut events = Vec::new();
        let found = discovery.scan(now(), &mut events);
        for event in events {
            registry.note(now(), "discovery", event, crate::instances::Tone::Warning);
        }
        registry.reconcile(found, now());
    }
    let mut devices = Gpus::off();
    let nvml = gpus.then(open);
    let mut sampler = match nvml {
        Some(Ok((ref nvml, ref driver, ref found))) => {
            devices = Gpus::new(Status::Opening, false, interval);
            devices.apply(
                Report::Opened {
                    driver: driver.clone(),
                    devices: found.clone(),
                },
                &mut registry,
                now(),
            );
            Some(Sampler::new(nvml, found.clone(), "/proc"))
        }
        Some(Err(ref reason)) => {
            devices = Gpus::new(Status::Opening, false, interval);
            devices.apply(Report::Unavailable(reason.clone()), &mut registry, now());
            None
        }
        None => None,
    };
    for round in 0..2 {
        if round > 0 {
            thread::sleep(Duration::from_secs_f64(interval));
        }
        if let Some(sampler) = sampler.as_mut() {
            let readings = sampler.poll(now(), false);
            devices.apply(Report::Polled(readings), &mut registry, now());
        }
        let jobs = registry
            .visible()
            .into_iter()
            .filter(|instance| !instance.endpoints.is_empty())
            .map(|instance| {
                let endpoints = instance.endpoints.clone();
                let key = matches!(instance.source, Source::Target(_))
                    .then(|| bearer.map(str::to_owned))
                    .flatten();
                (instance.id, endpoints, key)
            })
            .collect::<Vec<_>>();
        let results = thread::scope(|scope| {
            let handles = jobs
                .into_iter()
                .map(|(id, endpoints, key)| {
                    scope.spawn(move || {
                        let mut preferred = 0;
                        let result = poll(&endpoints, &mut preferred, key.as_deref(), || {
                            origin.elapsed().as_secs_f64()
                        });
                        (id, result)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .filter_map(|handle| handle.join().ok())
                .collect::<Vec<(u64, Poll)>>()
        });
        for (id, result) in results {
            registry.apply(id, result, now());
        }
    }
    drop(sampler);
    let read = now();
    (registry, devices, read)
}

/// The instances as a table, then this host's GPUs when they were read.
pub fn table(registry: &Registry, gpus: &Gpus, now: f64) -> String {
    let mut out = format!(
        "{}  {}  {}  {}  {} {} {} {}\n",
        pad("SOURCE", 16),
        pad("ADDRESS", 21),
        pad("MODEL", 20),
        pad("STATE", 13),
        right("DECODE", 8),
        right("PREFILL", 8),
        right("RUN/WAIT", 9),
        right("KV", 5)
    );
    for instance in registry.visible() {
        let figures = Figures::new(instance, crate::stats::Window::Minute);
        let number = |value: Option<f64>| value.map_or_else(|| "-".to_owned(), integer);
        let load = match (figures.running(), figures.waiting()) {
            (Some(running), Some(waiting)) => format!("{}/{}", integer(running), integer(waiting)),
            _ => "-".to_owned(),
        };
        out.push_str(&format!(
            "{}  {}  {}  {}  {} {} {} {}\n",
            pad(&sanitize(&instance.source.label()), 16),
            pad(
                &sanitize(&instance.address().unwrap_or_else(|| "-".into())),
                21
            ),
            pad(&sanitize(instance.name().unwrap_or("-")), 20),
            pad(instance.state(now).label(), 13),
            right(&number(figures.decode()), 8),
            right(&number(figures.prefill()), 8),
            right(&load, 9),
            right(&percent(figures.kv()), 5),
        ));
    }
    if gpus.status == Status::Open && !gpus.devices.is_empty() {
        out.push('\n');
        out.push_str(&gpu_table(registry, gpus));
    }
    out
}

/// This host's GPUs as a table.
fn gpu_table(registry: &Registry, gpus: &Gpus) -> String {
    let mut out = format!(
        "{}  {}  {} {} {} {} {}  {}  {}\n",
        pad("GPU", 3),
        pad("NAME", 20),
        right("TEMP", 5),
        right("FAN", 4),
        right("POWER", 9),
        right("SM MHZ", 6),
        right("VRAM GiB", 11),
        right("UTIL", 4),
        "INSTANCE"
    );
    for (index, device) in gpus.devices.iter().enumerate() {
        let reading = gpus.latest(index);
        let field = |value: Option<String>| value.unwrap_or_else(|| "-".to_owned());
        let gib = |bytes: u64| bytes as f64 / f64::from(1 << 30);
        let instance = gpus
            .processes(index, &registry.instances)
            .into_iter()
            .find_map(|(_, row)| row)
            .and_then(|id| registry.get(id))
            .map_or_else(|| "-".to_owned(), |instance| sanitize(&instance.label()));
        out.push_str(&format!(
            "{}  {}  {} {} {} {} {}  {}  {}\n",
            pad(&device.index.to_string(), 3),
            pad(&sanitize(&device.short_name()), 20),
            right(
                &field(reading.and_then(|r| r.temperature).map(|c| format!("{c}C"))),
                5
            ),
            right(
                &field(reading.and_then(|r| r.fan).map(|f| format!("{f}%"))),
                4
            ),
            right(
                &field(reading.and_then(|r| r.power).map(|watts| {
                    match reading.and_then(|r| r.power_limit) {
                        Some(limit) => format!("{watts:.0}/{limit:.0}W"),
                        None => format!("{watts:.0}W"),
                    }
                })),
                9
            ),
            right(
                &field(
                    reading
                        .and_then(|r| r.sm_mhz)
                        .map(|mhz| integer(f64::from(mhz)))
                ),
                6
            ),
            right(
                &field(
                    reading
                        .and_then(|r| r.memory_used)
                        .zip(device.memory_total)
                        .map(|(used, total)| format!("{:.1}/{:.1}", gib(used), gib(total)))
                ),
                11
            ),
            right(&percent(reading.and_then(Reading::kernels)), 4),
            instance,
        ));
    }
    out
}

/// The instances and GPUs as a `chopinfer.top.list.v1` document.
pub fn document(registry: &Registry, gpus: &Gpus, now: f64) -> Value {
    json!({
        "schema": SCHEMA,
        "instances": registry.visible().into_iter().map(|instance| row(instance, now)).collect::<Vec<_>>(),
        "gpus": gpu_document(registry, gpus, now),
        "events": registry
            .events
            .iter()
            .map(|event| json!({"who": event.who, "what": event.what}))
            .collect::<Vec<_>>(),
    })
}

/// This host's GPUs at `now`: whether they were read, the driver, and each
/// GPU's latest reading with its processes and the instance each one is.
fn gpu_document(registry: &Registry, gpus: &Gpus, now: f64) -> Value {
    let (status, detail) = match &gpus.status {
        Status::Off => ("off", None),
        Status::Opening => ("opening", None),
        Status::Unavailable(reason) => ("unavailable", Some(reason.clone())),
        Status::Open => ("read", None),
    };
    let devices = gpus
        .devices
        .iter()
        .enumerate()
        .map(|(index, device)| {
            let reading = gpus.latest(index);
            let processes = gpus
                .processes(index, &registry.instances)
                .into_iter()
                .map(|(process, row)| {
                    json!({
                        "pid": process.pid,
                        "type": process.kind.label(),
                        "memory_bytes": process.memory,
                        "name": process.owner.name,
                        "instance": row.and_then(|id| registry.get(id)).map(Instance::label),
                    })
                })
                .collect::<Vec<_>>();
            json!({
                "index": device.index,
                "name": device.name,
                "temperature_celsius": reading.and_then(|r| r.temperature),
                "headroom_celsius": reading.and_then(|r| r.headroom),
                "temperature_limit_celsius": device.temperature_limit,
                "fan_percent": reading.and_then(|r| r.fan),
                "power_watts": reading.and_then(|r| r.power),
                "power_limit_watts": reading.and_then(|r| r.power_limit),
                "sm_mhz": reading.and_then(|r| r.sm_mhz),
                "sm_max_mhz": device.sm_max_mhz,
                "memory_mhz": reading.and_then(|r| r.memory_mhz),
                "pstate": reading.and_then(|r| r.pstate),
                "clock_event_reasons": reading.and_then(|r| r.reasons).map(|reasons| reasons.0),
                "limited_by": reading.and_then(|r| r.reasons).and_then(|reasons| reasons.label()),
                "memory_used_bytes": reading.and_then(|r| r.memory_used),
                "memory_total_bytes": device.memory_total,
                "utilization": reading.and_then(|r| r.utilization).map(|(gpu, memory)| json!({
                    "gpu": gpu,
                    "memory": memory,
                })),
                "pcie": json!({
                    "generation": reading.and_then(|r| r.pcie).map(|(generation, _)| generation),
                    "width": reading.and_then(|r| r.pcie).map(|(_, width)| width),
                    "max_generation": device.pcie_max.map(|(generation, _)| generation),
                    "max_width": device.pcie_max.map(|(_, width)| width),
                }),
                "ecc": device.ecc,
                "ecc_uncorrected": reading.and_then(|r| r.ecc_uncorrected),
                "gpm": device.gpm,
                "lost": gpus.lost(index),
                "unreadable": device.unreadable,
                // Null when not read; from an earlier poll when the latest
                // one's process lists failed.
                "processes": gpus.processes_read(index).map(|_| processes),
                "processes_age_seconds": gpus.processes_read(index).map(|at| (now - at).max(0.0)),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "status": status,
        "detail": detail,
        "driver": gpus.driver,
        "devices": devices,
    })
}

fn row(instance: &Instance, now: f64) -> Value {
    let figures = Figures::new(instance, crate::stats::Window::Minute);
    let state = instance.state(now);
    let detail = match &state {
        State::NotReady(reason) | State::Unreachable(reason) => Some(reason.clone()),
        _ => None,
    };
    let (pid, container, container_id, target) = match &instance.source {
        Source::Process { pid, .. } => (Some(*pid), None, None, None),
        Source::Container { name, id, .. } => (None, Some(name.clone()), Some(id.clone()), None),
        Source::Target(target) => (None, None, None, Some(target.display())),
    };
    let identity = instance
        .health
        .as_ref()
        .and_then(|health| health.serving.as_ref())
        .map(|serving| {
            let profile = &serving.profile;
            json!({
                "bundle_id": serving.bundle_id,
                "profile": profile.name,
                "tensor_parallel_size": profile.tensor_parallel_size,
                "context_parallel_size": profile.context_parallel_size,
                "speculation": profile.speculation.as_ref().map(|speculation| json!({
                    "mode": speculation.mode,
                    "tokens": speculation.tokens,
                    "drafter": speculation.drafter,
                })),
                "kv_cache_dtype": profile.kv_cache_dtype,
                "prefix_cache": profile.prefix_cache,
                "max_sequences": profile.max_sequences,
                "served_model_names": serving.served_model_names,
                "startup_nonce": serving.startup_nonce,
                "checkpoint_identity": serving.checkpoint_identity,
                "version": serving.version,
                "source_commit": serving.source_commit,
            })
        });
    let totals = figures.latest().map(|snapshot| {
        let counters = &snapshot.counters;
        let map = |values: &[(String, f64)]| {
            values
                .iter()
                .map(|(key, value)| (key.clone(), json!(value)))
                .collect::<serde_json::Map<_, _>>()
        };
        json!({
            "generation_tokens": counters.generation_tokens,
            "prefill_tokens": counters.prefill_tokens,
            "finished": map(&counters.finished),
            "cancellations": counters.cancellations,
            "errors": counters.errors,
            "refusals": map(&counters.refusals),
            "batches": counters.batches,
        })
    });
    json!({
        "source": instance.source.kind(),
        "pid": pid,
        "container": container,
        "container_id": container_id,
        "target": target,
        "address": instance.address(),
        "state": state.label(),
        "group": state_group(&state),
        "detail": detail,
        "model": instance.name(),
        "also": instance.aliases,
        "uptime_seconds": instance.uptime(now),
        "identity": identity,
        "rates": {
            "decode_tokens_per_second": figures.decode(),
            "prefill_tokens_per_second": figures.prefill(),
        },
        "gauges": {
            "running": figures.running(),
            "waiting": figures.waiting(),
            "kv_usage": figures.kv(),
        },
        "ratios": {
            "prefix_token_hits": figures.token_hits(),
            "draft_acceptance": figures.acceptance(),
        },
        "totals": totals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poll::tests::FakeServer;

    const HEALTH: &str = r#"{"status":"ready","waiting":{"total":0},
        "serving":{"bundle_id":"fixture","startup_nonce":"aa","served_model_names":["fx"],
                   "profile":{"name":null,"tensor_parallel_size":4,"speculation":{"mode":"off","tokens":0,"drafter":null},
                              "kv_cache_dtype":"bf16","prefix_cache":true,"max_sequences":8},
                   "kv_page_tokens":16,"version":"0.1.0","source_commit":null,"checkpoint_identity":null}}"#;

    #[test]
    fn lists_targets_twice_polled_as_a_table_and_a_document() {
        let server = FakeServer::start(
            Some("aa"),
            HEALTH,
            "chop_process_uptime_seconds 10\nvllm:generation_tokens_total 40\nvllm:num_requests_running 1\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0.5\n",
        );
        let dead = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let targets = vec![
            Target::parse(&server.address.to_string()).unwrap(),
            Target::parse(&dead.to_string()).unwrap(),
        ];
        let (registry, gpus, now) = collect(targets, None, vec!["note".into()], 0.5, None, false);
        let text = table(&registry, &gpus, now);
        let lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 3, "{text}");
        assert!(
            lines[1].contains("fx") && lines[1].contains("ready") && lines[1].contains("1/0"),
            "{text}"
        );
        assert!(lines[2].contains("unreachable"), "{text}");
        let document = document(&registry, &gpus, now);
        assert_eq!(document["schema"], SCHEMA);
        assert_eq!(document["gpus"]["status"], "off");
        assert_eq!(document["gpus"]["devices"], serde_json::json!([]));
        let first = &document["instances"][0];
        assert_eq!(first["source"], "target");
        assert_eq!(first["state"], "ready");
        assert_eq!(first["identity"]["bundle_id"], "fixture");
        assert_eq!(first["identity"]["startup_nonce"], "aa");
        assert_eq!(first["totals"]["generation_tokens"], 40.0);
        assert_eq!(first["gauges"]["kv_usage"], 0.5);
        assert_eq!(document["instances"][1]["state"], "unreachable");
        assert_eq!(document["events"][0]["what"], "note");
    }

    #[test]
    fn lists_gpus_with_their_instances() {
        use crate::gpu::{Device, Kind, Owner, Process, Reasons};
        let mut registry = Registry::new(1.0);
        registry.add_target(Target::parse(":8311").unwrap(), 0.0);
        let mut gpus = Gpus::new(Status::Opening, false, 1.0);
        gpus.apply(
            Report::Opened {
                driver: Some("615.71.09".into()),
                devices: vec![Device {
                    index: 0,
                    name: "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition".into(),
                    memory_total: Some(97_887 << 20),
                    sm_max_mhz: Some(3090),
                    pcie_max: Some((5, 16)),
                    temperature_limit: Some(92),
                    ecc: Some(false),
                    gpm: true,
                    unreadable: None,
                }],
            },
            &mut registry,
            0.0,
        );
        let process = |pid, listen: Option<&str>, kind| Process {
            pid,
            memory: Some(1 << 30),
            kind,
            owner: Owner {
                name: Some("chopinfer".into()),
                listen: listen.map(str::to_owned),
                ..Owner::default()
            },
        };
        gpus.apply(
            Report::Polled(vec![Reading {
                temperature: Some(68),
                headroom: Some(24),
                fan: Some(54),
                power: Some(243.6),
                power_limit: Some(250.0),
                sm_mhz: Some(2520),
                reasons: Some(Reasons(Reasons::POWER_CAP)),
                memory_used: Some(88 << 30),
                utilization: Some((0.97, 0.71)),
                pcie: Some((5, 16)),
                at: 1.0,
                processes: Some(vec![
                    process(7, Some("0.0.0.0:8311"), Kind::Compute),
                    process(9, None, Kind::Graphics),
                ]),
                answered: true,
                ..Reading::default()
            }]),
            &mut registry,
            1.0,
        );
        let text = gpu_table(&registry, &gpus);
        assert_eq!(
            text.lines().nth(1),
            Some(
                "0    RTX PRO 6000 Max-Q      68C  54%  244/250W  2,520   88.0/95.6   97%  target 127.0.0.1:8311"
            ),
            "{text}"
        );
        let document = document(&registry, &gpus, 1.0);
        let gpu = &document["gpus"]["devices"][0];
        assert_eq!(document["gpus"]["status"], "read");
        assert_eq!(document["gpus"]["driver"], "615.71.09");
        assert_eq!(gpu["limited_by"], "power cap");
        assert_eq!(gpu["clock_event_reasons"], 4);
        assert_eq!(gpu["utilization"]["gpu"], 0.97);
        assert_eq!(gpu["pcie"]["max_generation"], 5);
        assert_eq!(gpu["processes"][0]["instance"], "target 127.0.0.1:8311");
        assert_eq!(gpu["processes"][1]["type"], "G");
        assert_eq!(gpu["processes"][1]["instance"], serde_json::Value::Null);
        assert_eq!(gpu["processes_age_seconds"], 0.0);
        // A poll whose process lists failed keeps the list before it, and
        // says how old it is; a GPU never listed has none.
        gpus.apply(
            Report::Polled(vec![Reading {
                at: 2.0,
                answered: true,
                ..Reading::default()
            }]),
            &mut registry,
            2.0,
        );
        let document = super::document(&registry, &gpus, 2.5);
        let gpu = &document["gpus"]["devices"][0];
        assert_eq!(gpu["processes"][0]["instance"], "target 127.0.0.1:8311");
        assert_eq!(gpu["processes_age_seconds"], 1.5);
        let mut unread = Gpus::new(Status::Opening, true, 1.0);
        unread.apply(
            Report::Opened {
                driver: None,
                devices: vec![Device::default()],
            },
            &mut registry,
            0.0,
        );
        unread.apply(Report::Polled(vec![Reading::default()]), &mut registry, 1.0);
        let document = super::document(&registry, &unread, 1.0);
        assert_eq!(
            document["gpus"]["devices"][0]["processes"],
            serde_json::Value::Null
        );
    }
}
