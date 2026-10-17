//! The list, detail and GPU views, rendered from a fixed fleet and fixed
//! GPU readings and compared with golden screens in `tests/screens/`.

use std::path::{Path, PathBuf};

use chopinfer_top::app::{App, View};
use chopinfer_top::discover::Found;
use chopinfer_top::discover::docker::ServeContainer;
use chopinfer_top::discover::proc::ServeProcess;
use chopinfer_top::gpu::poll::Report;
use chopinfer_top::gpu::{Activity, Device, Gpus, Kind, Owner, Process, Reading, Reasons, Status};
use chopinfer_top::health::Health;
use chopinfer_top::instances::Registry;
use chopinfer_top::poll::{Answer, Poll};
use chopinfer_top::prometheus::Histogram;
use chopinfer_top::stats::{Counters, Gauges, Histograms, Snapshot};
use chopinfer_top::targets::Target;
use chopinfer_top::ui;

/// Seconds of history the fleet has when the screens are drawn.
const NOW: f64 = 300.0;

fn histogram(bounds: &[f64], per_second: &[f64], seconds: f64) -> Histogram {
    let mut cumulative = 0.0;
    let mut buckets = Vec::new();
    for (bound, count) in bounds.iter().chain([f64::INFINITY].iter()).zip(per_second) {
        cumulative += count * seconds;
        buckets.push((*bound, cumulative));
    }
    Histogram {
        sum: 0.0,
        count: cumulative,
        buckets,
    }
}

const LATENCY: [f64; 6] = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5];
const LONG: [f64; 6] = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0];
const STEP: [f64; 6] = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5];

/// A busy server `seconds` into its life, generating about `rate` tokens a
/// second with `running` sequences.
fn busy(seconds: f64, rate: f64, running: f64) -> Snapshot {
    let t = seconds;
    // A deterministic wobble so sparklines have shape.
    let wobble = |period: f64| ((t % period) - period / 2.0).abs() / period;
    Snapshot {
        server_seconds: Some(seconds),
        client_seconds: seconds,
        counters: Counters {
            generation_tokens: rate * t + 200.0 * wobble(37.0) * t.min(37.0) / 37.0,
            prefill_tokens: 5200.0 * t + 30_000.0 * (t / 11.0).floor(),
            prefix_hit_tokens: 3700.0 * t,
            prefix_query_tokens: 5200.0 * t,
            prefix_lookups: 6.0 * t,
            prefix_lookup_hits: 5.0 * t,
            prefix_evictions: 0.4 * t,
            prefix_recomputed_seen_tokens: 150.0 * t,
            spec_draft_tokens: 1000.0 * t,
            spec_accepted_tokens: 590.0 * t,
            spec_draft_by_position: vec![200.0 * t, 200.0 * t, 200.0 * t, 200.0 * t, 200.0 * t],
            spec_accepted_by_position: vec![172.0 * t, 148.0 * t, 126.0 * t, 92.0 * t, 52.0 * t],
            speculation_k: vec![
                ("0".to_owned(), 2.0 * t),
                ("3".to_owned(), 11.0 * t),
                ("5".to_owned(), 15.0 * t),
                ("7".to_owned(), 22.0 * t),
            ],
            finished: vec![
                ("stop".to_owned(), (4.6 * t).floor()),
                ("length".to_owned(), (0.5 * t).floor()),
                ("abort".to_owned(), (0.05 * t).floor()),
            ],
            refusals: vec![
                ("capacity".to_owned(), 3.0),
                ("invalid_request".to_owned(), 1.0),
            ],
            cancellations: (0.05 * t).floor(),
            batches: 47.0 * t,
            decode_only_batches: 36.0 * t,
            mixed_batches: 9.0 * t,
            prefill_only_batches: 2.0 * t,
            ..Counters::default()
        },
        gauges: Gauges {
            running: Some(running),
            waiting: Some(2.0),
            kv_usage: Some(0.64),
            prefix_anchors: Some(1204.0),
            prefix_blocks: Some(18_330.0),
            intake_used: Some(6.0 * 1024.0 * 1024.0),
            intake_budget: Some(256.0 * 1024.0 * 1024.0),
            transport_reserved: Some(3.0 * 1024.0 * 1024.0),
            transport_budget: Some(512.0 * 1024.0 * 1024.0),
            worker: Some("ready"),
            poisoned: Some(false),
        },
        histograms: Histograms {
            ttft: Some(histogram(&LATENCY, &[0.0, 1.0, 3.0, 0.6, 0.3, 0.1, 0.0], t)),
            inter_token: Some(histogram(&STEP, &[6.0, 30.0, 4.0, 0.5, 0.1, 0.0, 0.0], t)),
            end_to_end: Some(histogram(&LONG, &[0.2, 0.8, 1.5, 1.6, 0.5, 0.1, 0.0], t)),
            queue: Some(histogram(&LATENCY, &[4.0, 0.4, 0.3, 0.1, 0.0, 0.0, 0.0], t)),
            prefill: Some(histogram(&LATENCY, &[0.5, 2.0, 1.7, 0.6, 0.2, 0.0, 0.0], t)),
            per_output_token: Some(histogram(&STEP, &[3.0, 1.6, 0.3, 0.0, 0.0, 0.0, 0.0], t)),
            step_decode: Some(histogram(&STEP, &[1.0, 30.0, 5.0, 0.0, 0.0, 0.0, 0.0], t)),
            step_mixed: Some(histogram(&STEP, &[0.0, 0.0, 2.0, 6.0, 1.0, 0.0, 0.0], t)),
            step_prefill: Some(histogram(&STEP, &[0.0, 0.0, 0.0, 0.5, 1.5, 0.0, 0.0], t)),
        },
    }
}

fn health(
    name: &str,
    bundle: &str,
    profile: &str,
    ranks: u64,
    speculation: (&str, u64, Option<&str>),
    nonce: &str,
) -> Option<Health> {
    let body = serde_json::json!({
        "status": "ready",
        "waiting": {"total": 2, "pre_backend": 0, "transport": 0, "intake": 0, "preparation": 0,
                    "staging": 0, "storage": 0, "backend": 2},
        "serving": {
            "bundle_id": bundle,
            "checkpoint_identity": "3fa91c07d2aa5e81c0ffee00112233445566778899aabbccddeeff0011223344",
            "startup_nonce": nonce,
            "profile": {
                "name": profile, "tensor_parallel_size": ranks,
                "speculation": {"mode": speculation.0, "tokens": speculation.1, "drafter": speculation.2},
                "kv_cache_dtype": "fp8", "prefix_cache": true, "max_sequences": 32
            },
            "kv_page_tokens": 64,
            "version": "0.1.0-alpha.2",
            "source_commit": "4c1e9a2b7d0f4c1e9a2b7d0f4c1e9a2b7d0f4c1e",
            "served_model_names": [name]
        }
    });
    Health::parse(body.to_string().as_bytes())
}

fn answer(endpoint: &str, nonce: &str, health: Option<Health>, snapshot: Snapshot) -> Poll {
    Poll::Answer(Box::new(Answer {
        endpoint: endpoint.to_owned(),
        nonce: nonce.to_owned(),
        health,
        snapshot: Some(snapshot),
        problem: None,
    }))
}

/// The preview's fleet: a busy MiMo process, a Qwen target that restarted
/// 30 s ago, an idle GLM container, a loading process and an unreachable
/// target.
fn fleet() -> App {
    let mut registry = Registry::new(1.0);
    let qwen = registry.add_target(Target::parse(":8431").unwrap(), 0.0);
    let remote = registry.add_target(Target::parse("192.0.2.10:8310").unwrap(), 0.0);
    let process = |pid: u32, listening: &[&str], hint: &str, age: f64| {
        Found::Process(ServeProcess {
            pid,
            start_ticks: u64::from(pid),
            executable: "chopinfer".to_owned(),
            arguments: format!("chopinfer serve --model /models/{hint} --tensor-parallel-size 2")
                .split_whitespace()
                .map(str::to_owned)
                .collect(),
            model_hint: Some(hint.to_owned()),
            age: Some(age),
            listening: listening
                .iter()
                .map(|address| address.parse().unwrap())
                .collect(),
        })
    };
    registry.reconcile(
        vec![
            process(48211, &["127.0.0.1:8311"], "MiMo-V2.6-Flash", 11_520.0),
            process(51022, &[], "Qwen3.8-Flash", 134.0 - NOW),
            Found::Container(ServeContainer {
                daemon: PathBuf::from("/var/run/docker.sock"),
                id: "4f2a".to_owned(),
                name: "glm-tp4".to_owned(),
                arguments: Vec::new(),
                model_hint: None,
                seen_for: 0.0,
                addresses: vec!["127.0.0.1:8600".parse().unwrap()],
            }),
        ],
        0.0,
    );
    let (mimo, glm) = (3, 5);
    registry.apply(remote, Poll::Failed("connection refused".to_owned()), 1.0);
    let mimo_nonce = "7d2e04b1c9aa7d2e04b1c9aa7d2e04b1";
    let first_qwen = "e19c5f0a42e19c5f0a42e19c5f0a42e1";
    let second_qwen = "5b70c2e9d85b70c2e9d85b70c2e9d85b";
    for second in 0..=NOW as u32 {
        let t = f64::from(second);
        registry.apply(
            mimo,
            answer(
                "127.0.0.1:8311",
                mimo_nonce,
                health(
                    "MiMo-V2.6-Flash",
                    "mimo26",
                    "tp4-dflash",
                    4,
                    ("adaptive", 7, Some("dflash")),
                    mimo_nonce,
                ),
                busy(11_000.0 + t, 1800.0, 24.0),
            ),
            t,
        );
        let (nonce, life) = if t < 270.0 {
            (first_qwen, 5000.0 + t)
        } else {
            (second_qwen, t - 270.0)
        };
        registry.apply(
            qwen,
            answer(
                "127.0.0.1:8431",
                nonce,
                health(
                    "Qwen3.8-Flash",
                    "qwen38",
                    "tp2-mtp",
                    2,
                    ("adaptive", 3, Some("mtp")),
                    nonce,
                ),
                busy(life, 950.0, 11.0),
            ),
            t,
        );
        let mut idle = busy(86_400.0, 0.0, 0.0);
        idle.server_seconds = Some(86_400.0 + t);
        idle.gauges.waiting = Some(0.0);
        idle.gauges.kv_usage = Some(0.02);
        registry.apply(
            glm,
            answer(
                "127.0.0.1:8600",
                "41aa8d3e0741aa8d3e0741aa8d3e0741",
                health(
                    "GLM-5.3-Flash",
                    "glm5next",
                    "tp4-mtp",
                    4,
                    ("adaptive", 3, Some("mtp")),
                    "41aa8d3e0741aa8d3e0741aa8d3e0741",
                ),
                idle,
            ),
            t,
        );
    }
    let mut app = App::new(registry, 1.0, 15.0, vec!["proc", "docker", "--target"]);
    app.selected = Some(mimo);
    app
}

const MIB: u64 = 1 << 20;

/// This host's GPUs for the fleet: four RTX PRO 6000 serving MiMo (process
/// 48211, TP4), a stray python3 beside it on GPU 3, a graphics process on
/// GPU 0, and GPU 2 in a thermal slowdown for the last minute. GPM figures
/// when `counters`.
fn with_gpus(app: &mut App, counters: bool) {
    let mut gpus = Gpus::new(Status::Opening, counters, 1.0);
    let device = |index| Device {
        index,
        name: "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition".to_owned(),
        memory_total: Some(97_887 * MIB),
        sm_max_mhz: Some(3090),
        pcie_max: Some((5, 16)),
        temperature_limit: Some(92),
        ecc: Some(false),
        gpm: true,
    };
    gpus.apply(
        Report::Opened {
            driver: Some("615.71.09".to_owned()),
            devices: (0..4).map(device).collect(),
        },
        &mut app.registry,
        0.0,
    );
    let process = |pid, memory, kind, ticks, name: &str| Process {
        pid,
        memory: Some(memory),
        kind,
        owner: Owner {
            start_ticks: Some(ticks),
            name: Some(name.to_owned()),
            ..Owner::default()
        },
    };
    for second in 0..=NOW as u32 {
        let t = f64::from(second);
        let wobble = ((t % 37.0) - 18.5).abs() / 18.5;
        let reading = |index: u32| {
            let hot = index == 2 && t >= 240.0;
            let mut processes = vec![process(
                48211,
                89_012 * MIB,
                Kind::Compute,
                48211,
                "chopinfer",
            )];
            if index == 3 {
                processes.push(process(51877, 1_434 * MIB, Kind::Compute, 7, "python3"));
            }
            if index == 0 {
                processes.push(process(2210, 96 * MIB, Kind::Graphics, 3, "Xorg"));
            }
            let used = processes
                .iter()
                .filter_map(|process| process.memory)
                .sum::<u64>()
                + 380 * MIB;
            let kernels = if hot { 0.95 } else { 0.82 + 0.08 * wobble };
            Reading {
                at: t,
                temperature: Some(if hot { 90 } else { 68 + index as i32 }),
                headroom: Some(if hot { 2 } else { 24 - index as i32 }),
                fan: Some(if hot { 100 } else { 54 + 2 * index }),
                power: Some(if hot { 214.0 } else { 236.0 + 10.0 * wobble }),
                power_limit: Some(250.0),
                sm_mhz: Some(if hot { 1650 } else { 2520 }),
                memory_mhz: Some(14_001),
                pstate: Some(1),
                reasons: Some(Reasons(if hot {
                    Reasons::SOFTWARE_THERMAL | Reasons::HARDWARE_THERMAL
                } else {
                    Reasons::POWER_CAP
                })),
                memory_used: Some(used),
                utilization: Some((0.97, 0.71)),
                pcie: Some((5, 16)),
                ecc_uncorrected: None,
                processes,
                activity: counters.then_some(Activity {
                    kernels: Some(kernels),
                    sm: Some(0.79),
                    occupancy: Some(0.31),
                    tensor: Some(if hot { 0.30 } else { 0.38 }),
                    dram: Some(if hot { 0.66 } else { 0.82 }),
                    pcie_rx: Some(6.2e9),
                    pcie_tx: Some(6.0e9),
                }),
                lost: false,
            }
        };
        gpus.apply(
            Report::Polled((0..4).map(reading).collect()),
            &mut app.registry,
            t,
        );
    }
    app.gpus = gpus;
}

/// An idle MiMo target with one short burst 55 s before `NOW`: a 76 tok/s
/// decode poll and prefill polls of 1,900 and 4,755 tok/s.
fn burst() -> App {
    let mut registry = Registry::new(1.0);
    let id = registry.add_target(Target::parse(":8311").unwrap(), 0.0);
    let nonce = "7d2e04b1c9aa7d2e04b1c9aa7d2e04b1";
    for second in 0..=NOW as u32 {
        let t = f64::from(second);
        let mut snapshot = busy(11_000.0 + t, 0.0, 0.0);
        snapshot.counters.generation_tokens = if t >= 246.0 { 76.0 } else { 0.0 };
        snapshot.counters.prefill_tokens = if t >= 247.0 {
            6_655.0
        } else if t >= 246.0 {
            1_900.0
        } else {
            0.0
        };
        registry.apply(
            id,
            answer(
                "127.0.0.1:8311",
                nonce,
                health(
                    "MiMo-V2.6-Flash",
                    "mimo26",
                    "tp4-dflash",
                    4,
                    ("adaptive", 7, Some("dflash")),
                    nonce,
                ),
                snapshot,
            ),
            t,
        );
    }
    let mut app = App::new(registry, 1.0, 15.0, vec!["--target"]);
    app.selected = Some(id);
    app
}

fn screens() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/screens")
}

/// Compares a rendered screen with its golden. A mismatch writes the
/// rendered screen under Cargo's target temporary directory to compare.
fn check(name: &str, app: &App, width: u16, height: u16) {
    let rendered = ui::draw(app, width, height, NOW).text();
    let golden = screens().join(format!("{name}.txt"));
    let expected = std::fs::read_to_string(&golden).unwrap_or_default();
    if rendered != expected {
        let actual = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.txt"));
        std::fs::write(&actual, &rendered).unwrap();
        panic!(
            "{name} differs from {}; the rendered screen is {}",
            golden.display(),
            actual.display()
        );
    }
}

#[test]
fn list_view() {
    let app = fleet();
    check("list-132x38", &app, 132, 38);
    // The summary's text fits its columns.
    check("list-100x30", &app, 100, 30);
    check("list-80x24", &app, 80, 24);
}

#[test]
fn a_short_burst_is_drawn_against_a_round_scale() {
    check("list-burst-132x38", &burst(), 132, 38);
}

#[test]
fn detail_view() {
    let mut app = fleet();
    app.view = View::Detail;
    check("detail-132x38", &app, 132, 38);
    // The smallest size that shows every panel.
    check("detail-100x32", &app, 100, 32);
    // Smaller terminals show fewer panels.
    check("detail-132x24", &app, 132, 24);
    check("detail-80x24", &app, 80, 24);
    check("detail-60x24", &app, 60, 24);
    // A loading process shows its status instead of panels.
    app.selected = Some(4);
    check("detail-loading-132x38", &app, 132, 38);
}

#[test]
fn help_overlay() {
    let mut app = fleet();
    app.help = true;
    check("help-132x38", &app, 132, 38);
}

#[test]
fn gpu_view() {
    let mut app = fleet();
    with_gpus(&mut app, true);
    app.view = View::Gpus;
    check("gpus-132x38", &app, 132, 38);
    // The GPU in a thermal slowdown.
    app.gpu = 2;
    check("gpus-100x30", &app, 100, 30);
    app.gpu = 3;
    check("gpus-80x24", &app, 80, 24);
    app.gpu = 0;
    app.help = true;
    check("help-gpus-132x38", &app, 132, 38);
}

#[test]
fn the_gpu_view_without_gpm_or_without_a_driver() {
    let mut app = fleet();
    with_gpus(&mut app, false);
    app.view = View::Gpus;
    check("gpus-nogpm-132x38", &app, 132, 38);
    app.gpus = Gpus::new(
        Status::Unavailable("libnvidia-ml.so.1 could not be loaded".to_owned()),
        true,
        1.0,
    );
    check("gpus-unavailable-100x30", &app, 100, 30);
}

#[test]
fn an_instance_names_its_gpus_and_they_lead_back_to_it() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let press = |app: &mut App, code| app.key(KeyEvent::new(code, KeyModifiers::NONE));
    let mut app = fleet();
    with_gpus(&mut app, true);
    app.view = View::Detail;
    let detail = ui::draw(&app, 132, 38, NOW).text();
    assert!(detail.contains("on GPUs 0-3"), "{detail}");
    app.view = View::List;
    press(&mut app, KeyCode::Char('g'));
    assert_eq!(app.view, View::Gpus);
    for _ in 0..3 {
        press(&mut app, KeyCode::Char('j'));
    }
    assert_eq!(app.gpu, 3);
    app.selected = None;
    press(&mut app, KeyCode::Enter);
    assert_eq!((app.view, app.current().unwrap().id), (View::Detail, 3));
    press(&mut app, KeyCode::Esc);
    assert_eq!(app.view, View::Gpus);
    press(&mut app, KeyCode::Esc);
    assert_eq!(app.view, View::List);
    // From an instance's details, g selects its first GPU.
    app.gpu = 3;
    app.view = View::Detail;
    press(&mut app, KeyCode::Char('g'));
    assert_eq!((app.view, app.gpu), (View::Gpus, 0));
    press(&mut app, KeyCode::Char('g'));
    assert_eq!(app.view, View::Detail);
}

#[test]
fn every_size_draws_without_panicking() {
    let mut app = fleet();
    with_gpus(&mut app, true);
    for view in [View::List, View::Detail, View::Gpus] {
        app.view = view;
        for help in [false, true] {
            app.help = help;
            for width in (40..=200).step_by(7).chain([1024, 4000]) {
                for height in (10..=60).step_by(3) {
                    let frame = ui::draw(&app, width, height, NOW);
                    assert_eq!((frame.width, frame.height), (width, height));
                }
            }
        }
    }
}

/// The GLM container's answer at `NOW` with `health` instead of its own.
fn glm_answer(health: &serde_json::Value) -> Poll {
    let mut idle = busy(86_400.0 + NOW, 0.0, 0.0);
    idle.gauges.waiting = Some(0.0);
    answer(
        "127.0.0.1:8600",
        "41aa8d3e0741aa8d3e0741aa8d3e0741",
        Health::parse(health.to_string().as_bytes()),
        idle,
    )
}

#[test]
fn a_not_ready_reason_is_shown_in_the_summary_header_and_events() {
    let mut app = fleet();
    let glm = 5;
    app.registry.apply(
        glm,
        glm_answer(&serde_json::json!({
            "status": "not_ready",
            "reason": "generation worker is restarting",
            "serving": {"served_model_names": ["GLM-5.3-Flash"]}
        })),
        NOW,
    );
    app.selected = Some(glm);
    let list = ui::draw(&app, 132, 38, NOW).text();
    assert!(
        list.contains("GLM-5.3-Flash  not ready: generation worker is restarting"),
        "{list}"
    );
    assert!(
        list.lines().any(|line| line.contains(" ago ")
            && line.contains("docker glm-tp4")
            && line.contains("not ready: generation worker is restarting")),
        "{list}"
    );
    app.view = View::Detail;
    let detail = ui::draw(&app, 132, 38, NOW).text();
    let header = detail.lines().next().unwrap();
    assert!(
        header.contains("not ready: generation worker is restarting"),
        "{header}"
    );
}

#[test]
fn a_very_long_model_name_is_cut_at_the_edge() {
    let mut app = fleet();
    let glm = 5;
    let name = "M".repeat(70_000);
    app.registry.apply(
        glm,
        glm_answer(&serde_json::json!({
            "status": "ready",
            "serving": {"served_model_names": [name]}
        })),
        NOW,
    );
    app.selected = Some(glm);
    with_gpus(&mut app, true);
    for view in [View::List, View::Detail, View::Gpus] {
        app.view = view;
        for (width, height) in [(132, 32), (1024, 32), (100, 32)] {
            let frame = ui::draw(&app, width, height, NOW);
            assert_eq!(frame.width, width);
        }
    }
}

#[test]
fn a_long_list_scrolls_to_the_selection() {
    let mut app = fleet();
    for port in 0..30 {
        app.registry
            .add_target(Target::parse(&format!(":{}", 9000 + port)).unwrap(), NOW);
    }
    let last = app.rows().last().unwrap().id;
    app.selected = Some(last);
    app.settle(24);
    let list = ui::draw(&app, 80, 24, NOW).text();
    assert!(list.contains("rows 23-35 of 35"), "{list}");
    let selected = list
        .lines()
        .find(|line| line.contains('▌'))
        .unwrap_or_default();
    assert!(selected.contains("127.0.0.1:9029"), "{list}");
}
