//! `eval run --launch` against a fake `serve`: this test binary, run by a
//! script standing in for `chopinfer`, serves `/v1/models` and `/health` on
//! the address after `--listen`.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use runtime_eval::json::sha256_hex;
use runtime_eval::{Exit, Host, Tool, cli};
use serde_json::{Value, json};

const CHECKPOINT: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const NONCE: &str = "0123456789abcdef0123456789abcdef";
/// Set for the fake `serve` to the words it was started with.
const FAKE: &str = "RUNTIME_EVAL_FAKE_SERVE";

/// The fake `serve`: idle unless started by the script [`serve_script`]
/// writes. `--fail` exits at once; `--not-ready N` answers the first N
/// `/health` requests 503.
#[test]
fn fake_serve() {
    let Ok(words) = std::env::var(FAKE) else {
        return;
    };
    let words: Vec<&str> = words.split(' ').collect();
    let after = |flag: &str| {
        words
            .iter()
            .position(|word| *word == flag)
            .map(|index| words[index + 1])
    };
    assert_eq!(words[0], "serve");
    if words.contains(&"--fail") {
        eprintln!("serve: refused the fake bundle");
        std::process::exit(2);
    }
    let not_ready: usize = after("--not-ready").map_or(0, |count| count.parse().unwrap());
    let listener = TcpListener::bind(after("--listen").unwrap()).unwrap();
    println!("fake serve listening on {}", listener.local_addr().unwrap());
    let mut health = 0;
    for stream in listener.incoming() {
        let mut reader = BufReader::new(stream.unwrap());
        let mut request = String::new();
        reader.read_line(&mut request).unwrap();
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap() > 2 {
            line.clear();
        }
        let (status, body) = if request.contains("/v1/models") {
            ("200 OK", json!({"data": [{"id": "Example"}]}))
        } else if health < not_ready {
            health += 1;
            ("503 Service Unavailable", json!({"status": "not_ready"}))
        } else {
            let serving = json!({"checkpoint_identity": CHECKPOINT, "startup_nonce": NONCE});
            ("200 OK", json!({"status": "ready", "serving": serving}))
        };
        let body = body.to_string();
        let response = format!(
            "HTTP/1.1 {status}\r\ncontent-length: {}\r\nx-chopinfer-checkpoint-identity: \
             {CHECKPOINT}\r\nx-chopinfer-startup-nonce: {NONCE}\r\n\r\n{body}",
            body.len()
        );
        let _ = reader.into_inner().write_all(response.as_bytes());
    }
}

/// A script standing in for `chopinfer`: runs [`fake_serve`] with its
/// arguments.
fn serve_script(root: &Path) -> PathBuf {
    let script = root.join("bin/chopinfer");
    std::fs::create_dir_all(script.parent().unwrap()).unwrap();
    let test = std::env::current_exe().unwrap();
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\nexport {FAKE}=\"$*\"\nexec '{}' fake_serve --exact --nocapture \
             --test-threads=1\n",
            test.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    script
}

/// A root with one model declaring an informational check, which a run
/// against a bound server passes without a reference.
fn synthetic_root() -> PathBuf {
    let root = std::env::temp_dir().join(format!("runtime-eval-launch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let model = root.join("models/example");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::create_dir_all(root.join("tools/qual")).unwrap();
    let modules = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/qual/modules.json");
    std::fs::copy(modules, root.join("tools/qual/modules.json")).unwrap();
    let rule = json!({"path": ["passed"], "op": "eq", "value": true, "source": "docs/example.md"});
    let declaration = json!({
        "schema": "chopinfer-qual-model/v1", "model": "example",
        "cohorts": {"quick": {"checks": ["structured"]}},
        "change_classes": {"frontend": [], "quality": [], "service": [], "speculation": [], "prefix": []},
        "checks": {"structured": {"module": "structured", "args": [], "timeout_seconds": 60,
                                  "receipt": "*-s.json", "rules": [rule],
                                  "informational_reason": "not yet"}},
        "inputs": {},
        "qualification": {"teacher_reference_kind": "exact", "teacher_receipt": "teacher",
                          "teacher_capture_schema": "example.capture.v1", "verify_buckets": [2],
                          "chars_per_token": 4.0,
                          "checkpoint": {"lock": "models/example/checkpoint-files.json",
                                         "repository": "org/example", "revision": "abc",
                                         "served_model_name": "Example",
                                         "source_lock_sha256": CHECKPOINT,
                                         "checkpoint_identity": CHECKPOINT}},
        "receipts": {"env": "RUNTIME_EVAL_TEST_UNSET", "path": "qual/example"}
    });
    std::fs::write(model.join("qual.json"), declaration.to_string()).unwrap();
    root
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// `eval run --launch FLAGS` for the synthetic model with the fake serve,
/// writing the report to `ROOT/<name>`; its exit and report.
fn launch(root: &Path, name: &str, options: &[&str], flags: &[&str]) -> (Exit, Option<Value>) {
    let host = Host {
        executable: serve_script(root),
        checkpoint_model: Box::new(|_| Err("no checkpoints".into())),
        serve_listen: None,
    };
    let tool = Tool {
        name: "chopinfer eval".into(),
        version: "test".into(),
        revision: None,
    };
    let model = root.join("models/example");
    let report = root.join(name);
    let mut arguments = vec![
        "run",
        "--model",
        model.to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
    ];
    arguments.extend(options);
    arguments.push("--launch");
    arguments.extend(flags);
    let arguments: Vec<String> = arguments.iter().map(|word| (*word).to_owned()).collect();
    let exit = cli::run_hosted(&tool, &host, &arguments);
    let report = std::fs::read(report.join("report.json"))
        .ok()
        .map(|bytes| runtime_eval::json::parse(&bytes).unwrap());
    (exit, report)
}

/// The launched process has exited and been reaped.
fn gone(launch: &Value) -> bool {
    let pid = launch["pid"].as_u64().unwrap();
    !Path::new(&format!("/proc/{pid}")).exists()
}

// One test runs every launch, so no other test of this binary forks while
// the script is written (which could make it busy at its exec).
#[test]
fn a_launched_server_is_waited_for_checked_and_stopped() {
    let root = synthetic_root();
    let listen = format!("127.0.0.1:{}", free_port());

    // Ready after two health polls, bound, checked and stopped.
    let (exit, report) = launch(
        &root,
        "ready",
        &["--check", "structured"],
        &["--listen", &listen, "--not-ready", "2"],
    );
    let report = report.unwrap();
    assert_eq!(exit, Exit::Pass, "{report}");
    assert_eq!(report["server"]["url"], format!("http://{listen}"));
    let record = &report["server"]["launch"];
    let log = root.join("ready/server.log");
    assert_eq!(record["log"], log.to_str().unwrap());
    let bytes = std::fs::read(&log).unwrap();
    assert_eq!(record["log_sha256"], sha256_hex(&bytes));
    assert!(String::from_utf8_lossy(&bytes).contains("fake serve listening"));
    let command: Vec<&str> = record["command"]
        .as_array()
        .unwrap()
        .iter()
        .map(|word| word.as_str().unwrap())
        .collect();
    assert_eq!(
        command[1..],
        ["serve", "--listen", &listen, "--not-ready", "2"]
    );
    assert!(
        record["exit"].as_str().unwrap().contains("SIGTERM"),
        "{record}"
    );
    assert!(gone(record));

    // A server that exits before it is ready: the environment's failure,
    // with the end of its log.
    let (exit, report) = launch(&root, "fails", &[], &["--listen", &listen, "--fail"]);
    let report = report.unwrap();
    assert_eq!(exit, Exit::Environment);
    let message = report["failure"]["message"].as_str().unwrap();
    assert!(message.contains("exited before it was ready"), "{message}");
    assert!(message.contains("refused the fake bundle"), "{message}");
    assert_eq!(report["exit"], 4);
    assert!(gone(&report["server"]["launch"]));

    // One never ready within the timeout is stopped.
    let (exit, report) = launch(
        &root,
        "slow",
        &["--timeout", "1"],
        &["--listen", &listen, "--not-ready", "1000000"],
    );
    let report = report.unwrap();
    assert_eq!(exit, Exit::Environment);
    let message = report["failure"]["message"].as_str().unwrap();
    assert!(message.contains("not ready within 1 s"), "{message}");
    assert!(gone(&report["server"]["launch"]));

    // A launch needs a listen address with a port, and no --url.
    for (name, options, flags) in [
        ("unlisted", &[][..], &["--fail"][..]),
        ("port", &[][..], &["--listen", "127.0.0.1:0"][..]),
        ("both", &["--url", "http://127.0.0.1:9"][..], &[][..]),
    ] {
        let (exit, report) = launch(&root, name, options, flags);
        assert_eq!(exit, Exit::Usage, "{name}");
        assert!(report.is_none() && !root.join(name).exists(), "{name}");
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn cleanup_stops_descendants_after_the_leader_exits() {
    use runtime_eval::launch::Server;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    let root = std::env::temp_dir().join(format!("runtime-eval-groups-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let script = root.join("server");
    std::fs::write(&script, "#!/bin/sh\n(trap '' TERM; echo $$ > /dev/null; while :; do sleep 1; done) &\necho $! > \"$2\"\nexit 0\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut unrelated = Command::new("sleep").arg("120").spawn().unwrap();
    let mut survivors = Vec::new();
    for mode in ["stop", "drop"] {
        let pid_file = root.join(mode);
        let mut server = Server::start(
            &script,
            &[pid_file.display().to_string()],
            &root.join(format!("{mode}.log")),
        )
        .unwrap()
        // The descendant ignores SIGTERM, so any grace ends in SIGKILL.
        .with_grace(Duration::from_secs(1));
        // The shell creates the file before writing the PID, so wait for a number:
        // an empty read would name /proc//stat, which is /proc/stat.
        let read_pid = || {
            std::fs::read_to_string(&pid_file)
                .ok()
                .map(|text| text.trim().to_owned())
                .filter(|pid| pid.parse::<u32>().is_ok())
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        while read_pid().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        let pid = read_pid().unwrap();
        assert!(
            server
                .wait_ready("http://127.0.0.1:1", Duration::from_secs(1))
                .is_err()
        );
        if mode == "stop" {
            server.stop();
        } else {
            drop(server);
        }
        // A process that exited stays a zombie (Z) until its reaper takes
        // it, and is dead (X) while that reaper releases it.
        let running = || {
            std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
                !stat
                    .split_once(") ")
                    .unwrap()
                    .1
                    .starts_with(['Z', 'X', 'x'])
            })
        };
        // A killed process still has to be scheduled to exit; the bound
        // only keeps a survivor from hanging the test.
        let deadline = Instant::now() + Duration::from_secs(30);
        while running() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        survivors.push((mode, running()));
        // Clean up the regression's own descendant even on the unfixed implementation.
        let _ = Command::new("kill")
            .args(["-KILL", "--", &pid])
            .stderr(Stdio::null())
            .status();
    }
    let unrelated_running = unrelated.try_wait().unwrap().is_none();
    unrelated.kill().unwrap();
    unrelated.wait().unwrap();
    std::fs::remove_dir_all(root).unwrap();
    assert!(unrelated_running);
    assert!(
        survivors.iter().all(|(_, running)| !running),
        "{survivors:?}"
    );
}
