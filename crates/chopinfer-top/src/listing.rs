//! `--list`: one discovery pass, two polls one interval apart, then a table
//! or a `chopinfer.top.list.v1` JSON document.

use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::discover::Discovery;
use crate::instances::{Instance, Registry, Source, State};
use crate::poll::{Poll, poll};
use crate::targets::Target;
use crate::ui::widgets::{integer, pad, percent, right, sanitize};
use crate::ui::{Figures, state_group};

pub const SCHEMA: &str = "chopinfer.top.list.v1";

/// Finds and polls every instance; returns the registry and when it was
/// read.
pub fn collect(
    targets: Vec<Target>,
    discovery: Option<Discovery>,
    notes: Vec<String>,
    interval: f64,
    bearer: Option<&str>,
) -> (Registry, f64) {
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
    for round in 0..2 {
        if round > 0 {
            thread::sleep(Duration::from_secs_f64(interval));
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
    let read = now();
    (registry, read)
}

/// The instances as a table.
pub fn table(registry: &Registry, now: f64) -> String {
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
    out
}

/// The instances as a `chopinfer.top.list.v1` document.
pub fn document(registry: &Registry, now: f64) -> Value {
    json!({
        "schema": SCHEMA,
        "instances": registry.visible().into_iter().map(|instance| row(instance, now)).collect::<Vec<_>>(),
        "events": registry
            .events
            .iter()
            .map(|event| json!({"who": event.who, "what": event.what}))
            .collect::<Vec<_>>(),
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
        let (registry, now) = collect(targets, None, vec!["note".into()], 0.5, None);
        let text = table(&registry, now);
        let lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 3, "{text}");
        assert!(
            lines[1].contains("fx") && lines[1].contains("ready") && lines[1].contains("1/0"),
            "{text}"
        );
        assert!(lines[2].contains("unreachable"), "{text}");
        let document = document(&registry, now);
        assert_eq!(document["schema"], SCHEMA);
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
}
