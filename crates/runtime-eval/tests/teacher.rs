//! The teacher_forced capture of `eval run` against a fake server must send
//! the requests and write the receipt and compact capture the retired
//! `tools/qual/teacher.py` did, timing and its informational block aside
//! (tests/data/retired-captures.json). The greedy record of `eval capture`
//! must send one cold greedy request per pinned text and write their tokens.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use runtime_eval::json::{canonical, canonical_sha256, sha256_hex};
use runtime_eval::{Exit, Tool};
use serde_json::{Map, Value, json};

const CHECKPOINT: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const NONCE: &str = "0123456789abcdef0123456789abcdef";

/// One scored position's stored top five: the token first, then four
/// others, each `step` nats below the one before after the first gap.
fn ranked(token: u64, first: f64, gap: f64) -> Vec<Value> {
    let mut pairs = vec![json!([token, first]), json!([token + 100, first - gap])];
    for (index, other) in (101..104).enumerate() {
        pairs.push(json!([token + other, first - gap - 1.0 - index as f64]));
    }
    pairs
}

fn variant(ids: &[u64], from: usize, first: f64, gap: f64) -> Value {
    let positions = &ids[from..];
    let ranked: Vec<_> = positions
        .iter()
        .map(|token| ranked(*token, first, gap))
        .collect();
    let column = |index: usize| {
        ranked
            .iter()
            .map(|row| {
                row.iter()
                    .map(|pair| pair[index].clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    json!({
        "logprob": positions.iter().map(|_| first).collect::<Vec<_>>(),
        "top1": positions,
        "margin": positions.iter().map(|_| gap).collect::<Vec<_>>(),
        "topk_ids": column(0),
        "topk_logits": column(1),
        "topk_logprobs": column(1),
    })
}

/// Two texts in two categories with a gate and a spread variant.
fn reference(checkpoint: &Value) -> Value {
    let texts: Vec<Value> = [
        ("t0", "prose", (1u64..=15).collect::<Vec<_>>(), 1usize),
        ("t1", "code", [1].into_iter().chain(30..40).collect(), 2),
    ]
    .into_iter()
    .map(|(name, category, ids, from)| {
        json!({"name": name, "category": category, "prompt_token_ids": ids, "score_from": from,
               "exact": variant(&ids, from, -0.25, 0.25),
               "bf16": variant(&ids, from, -0.375, 0.125)})
    })
    .collect();
    json!({"schema": "example.teacher.v1",
           "provenance": {"checkpoint": checkpoint, "variants": {"exact": "f64", "bf16": "bf16"}},
           "texts": texts})
}

/// A root with one model whose teacher check runs under both tools.
fn synthetic_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "runtime-eval-teacher-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let model = root.join("models/example");
    let fixtures = root.join("fixtures/example");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::create_dir_all(&fixtures).unwrap();
    std::fs::create_dir_all(root.join("tools/qual")).unwrap();
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::fs::copy(
        repo.join("tools/qual/modules.json"),
        root.join("tools/qual/modules.json"),
    )
    .unwrap();
    let lock = json!({"model.safetensors": {"sha256": "ab", "size": 1}});
    let lock_path = model.join("checkpoint-files.json");
    std::fs::write(&lock_path, lock.to_string()).unwrap();
    let checkpoint = json!({"repository": "org/example", "revision": "abc",
                            "served_model_name": "Example",
                            "source_lock_sha256": sha256_hex(&canonical(&lock)),
                            "checkpoint_identity": CHECKPOINT});
    let bytes = serde_json::to_vec(&reference(&checkpoint)).unwrap();
    std::fs::write(fixtures.join("teacher.json"), &bytes).unwrap();
    std::fs::write(
        fixtures.join("teacher-texts.json"),
        json!({"texts": [{"name": "t0"}, {"name": "t1"}]}).to_string(),
    )
    .unwrap();
    std::fs::write(
        model.join("frontend.json"),
        json!({"reasoning": {"thinking_keywords": ["enable_thinking"]}}).to_string(),
    )
    .unwrap();
    let mut pin = checkpoint.clone();
    // teacher.py reads the lock under its own checkout: give it an absolute path.
    pin["lock"] = json!(lock_path.to_str().unwrap());
    let rule = |path: Value, op: &str, value: Value| json!({"path": path, "op": op, "value": value, "source": "docs/example.md"});
    let declaration = json!({
        "schema": "chopinfer-qual-model/v1", "model": "example",
        "cohorts": {"quick": {"checks": ["teacher"]},
                    "full": {"checks": ["teacher"]}},
        "change_classes": {"frontend": [], "quality": [], "service": [], "speculation": [], "prefix": []},
        "checks": {
            "teacher": {"module": "teacher", "args": ["--url", "{url}", "--model", "{model}"],
                        "cohort_args": {"full": ["--capture-out", "{raw}/teacher-capture.json"]},
                        "timeout_seconds": 60, "receipt": "*-teacher.json",
                        "rules": [rule(json!(["passed"]), "eq", json!(true)),
                                  rule(json!(["top1_disagreement"]), "le", json!(0.1))]}
        },
        "references": {"teacher": {"path": "fixtures/example/teacher.json", "sha256": sha256_hex(&bytes),
                                   "schema": "example.teacher.v1", "variants": ["exact", "bf16"]}},
        "inputs": {"fixture_root": "fixtures/example", "teacher_reference": "fixtures/example/teacher.json",
                   "exact": "fixtures/example/teacher.json",
                   "frontend": "models/example/frontend.json"},
        "qualification": {"teacher_reference_kind": "exact", "teacher_receipt": "teacher",
                          "teacher_capture_schema": "example.capture.v1", "verify_buckets": [2],
                          "chars_per_token": 4.0, "checkpoint": pin},
        "receipts": {"env": "RUNTIME_EVAL_TEST_UNSET", "path": "qual/example"}
    });
    std::fs::write(model.join("qual.json"), declaration.to_string()).unwrap();
    assert_eq!(canonical_sha256(&lock), checkpoint["source_lock_sha256"]);
    root
}

/// The served prompt-logprob row of `token` at `position`: the token
/// leads, slightly off the reference, except token 5, where the
/// reference's runner-up leads: a near tie.
fn served_row(token: u64, position: usize) -> Value {
    let shift = 0.015625 * (position % 4) as f64;
    let flipped = token == 5;
    let (lead, second) = if flipped {
        (token + 100, token)
    } else {
        (token, token + 100)
    };
    let mut row = Map::new();
    row.insert(
        lead.to_string(),
        json!({"logprob": -0.1875 - shift, "rank": 1}),
    );
    row.insert(
        second.to_string(),
        json!({"logprob": -0.4375 - shift, "rank": 2}),
    );
    row.insert(
        (token + 101).to_string(),
        json!({"logprob": -2.5, "rank": 3}),
    );
    Value::Object(row)
}

fn answer(path: &str, body: &Value) -> Value {
    if body.get("cache_salt").is_some() {
        // A greedy request: up to three tokens from the prompt's length,
        // never cached.
        let length = body["prompt"].as_array().unwrap().len() as u64;
        let count = body["max_tokens"].as_u64().unwrap().min(3);
        let ids: Vec<u64> = (0..count).map(|index| length * 10 + index).collect();
        let finish = if length % 2 == 0 { "length" } else { "stop" };
        return json!({"choices": [{"index": 0, "text": " y", "token_ids": ids, "finish_reason": finish}],
                      "usage": {"prompt_tokens": length, "completion_tokens": count,
                                "prompt_tokens_details": {"cached_tokens": 0}}});
    }
    match path {
        "/v1/models" => json!({"data": [{"id": "Example"}]}),
        "/health" => json!({"status": "ready",
                            "serving": {"checkpoint_identity": CHECKPOINT, "startup_nonce": NONCE}}),
        _ => {
            let ids: Vec<u64> = body["prompt"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap())
                .collect();
            let mut rows = vec![Value::Null];
            rows.extend(
                ids.iter()
                    .enumerate()
                    .skip(1)
                    .map(|(position, token)| served_row(*token, position)),
            );
            json!({"choices": [{"index": 0, "text": " x", "finish_reason": "length",
                                "prompt_token_ids": ids, "prompt_logprobs": rows,
                                "logprobs": {"token_logprobs": [null], "tokens": [" x"]}}],
                   "usage": {"prompt_tokens": ids.len(), "completion_tokens": 1}})
        }
    }
}

/// A fake server answering every connection; returns its URL and the
/// request lines and bodies it read.
fn serve() -> (String, Arc<Mutex<Vec<String>>>) {
    serve_answering_as(NONCE)
}

/// A fake server whose answers to requests with a body name the startup
/// nonce `answered`, while /health names its own.
fn serve_answering_as(answered: &'static str) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = requests.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            let mut reader = BufReader::new(stream);
            let mut first = String::new();
            if reader.read_line(&mut first).unwrap_or(0) == 0 {
                continue;
            }
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
                if line == "\r\n" || line.is_empty() {
                    break;
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let path = first
                .split_whitespace()
                .nth(1)
                .unwrap_or_default()
                .to_owned();
            let parsed = if body.is_empty() {
                Value::Null
            } else {
                runtime_eval::json::parse(&body).unwrap()
            };
            if !body.is_empty() {
                seen.lock().unwrap().push(format!(
                    "{path} {}",
                    String::from_utf8_lossy(&canonical(&parsed))
                ));
            }
            let text = answer(&path, &parsed).to_string();
            let nonce = if body.is_empty() { NONCE } else { answered };
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
                 x-chopinfer-checkpoint-identity: {CHECKPOINT}\r\nx-chopinfer-startup-nonce: {nonce}\r\n\
                 connection: close\r\n\r\n{text}",
                text.len()
            );
            let mut stream = reader.into_inner();
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (url, requests)
}

fn read(path: &Path) -> Value {
    runtime_eval::json::parse(&std::fs::read(path).unwrap()).unwrap()
}

/// A receipt without what differs between two runs of the same capture:
/// its stamp, its host load, the per-text latencies and the informational
/// block the retired teacher.py added. Each category also drops the verdict
/// fields variant-spread.v2 added after teacher.py retired (`spread_rule`,
/// `passed_by`, `power`, `paired_reference`); the rest is still what it
/// wrote.
fn comparable(receipt: &Value) -> Value {
    let mut value = receipt.clone();
    let map = value.as_object_mut().unwrap();
    map.remove("at");
    map.remove("informational");
    map.remove("host_load");
    for row in map["rows"].as_array_mut().unwrap() {
        row.as_object_mut().unwrap().remove("latency_seconds");
    }
    if let Some(categories) = map.get_mut("by_category").and_then(Value::as_object_mut) {
        for category in categories.values_mut() {
            let category = category.as_object_mut().unwrap();
            for key in ["spread_rule", "passed_by", "power", "paired_reference"] {
                category.remove(key);
            }
        }
    }
    value
}

/// teacher_arms.py authenticates every input it reads against the reference
/// lock a model declared under the reference root must have: adopt one for
/// the synthetic inputs, as `refs adopt` would.
fn adopt(root: &Path) {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let adopted = Command::new("python3")
        .arg("-B")
        .arg("-c")
        .arg(
            "import json, sys\n\
             from pathlib import Path\n\
             import declaration, refs\n\
             root = Path(sys.argv[1])\n\
             config = declaration.load(root / 'models/example/qual.json')\n\
             snapshot = refs.snapshot(config, root)\n\
             refs.lock_path('example', root).write_text(json.dumps(snapshot))\n",
        )
        .arg(root)
        .env("PYTHONPATH", repo.join("tools/qual"))
        .env("CHOPINFER_QUAL_MODEL_ROOT", root)
        .output()
        .unwrap();
    assert!(
        adopted.status.success(),
        "{}",
        String::from_utf8_lossy(&adopted.stderr)
    );
}

/// A startup log of the fake server at `url` that names its checkpoint and
/// startup nonce.
fn startup_log(root: &Path, url: &str) -> PathBuf {
    let log = root.join("server.log");
    let identity = json!({"checkpoint_identity": CHECKPOINT, "startup_nonce": NONCE});
    let listening = json!({"status": "listening", "address": url.trim_start_matches("http://")});
    std::fs::write(
        &log,
        format!("startup serving_identity: {identity}\n{listening}\n"),
    )
    .unwrap();
    log
}

#[test]
fn the_capture_writes_the_receipt_teacher_py_wrote() {
    same_receipt("same", false, false);
}

#[test]
fn bound_to_the_startup_log_it_writes_the_receipt_teacher_py_wrote() {
    same_receipt("logged", true, false);
}

#[test]
fn in_the_full_cohort_it_writes_the_capture_and_receipt_teacher_py_wrote() {
    same_receipt("captured", true, true);
}

/// What the teacher capture wrote that a run does not vary: the requests
/// the server read, the comparable receipt and the compact capture, each
/// with what a run varies (its startup log and digests, the root and the
/// server's address) replaced by `varying`.
fn invariant(
    sent: &[String],
    receipt: &Value,
    capture: Option<&[u8]>,
    varying: &[(String, &str)],
) -> Value {
    let normal = |text: String| {
        varying
            .iter()
            .fold(text, |text, (from, to)| text.replace(from.as_str(), to))
    };
    let receipt = String::from_utf8(canonical(&comparable(receipt))).unwrap();
    json!({"requests": sent,
           "receipt": runtime_eval::json::parse(normal(receipt).as_bytes()).unwrap(),
           "capture": capture.map(|bytes| normal(String::from_utf8(bytes.to_vec()).unwrap()))})
}

/// `eval run` against the fake server, bound to its startup log when
/// `logged`, and in the full cohort, whose arguments write the compact
/// capture, when `captured`: it sends and writes what the retired
/// teacher.py did.
fn same_receipt(name: &str, logged: bool, captured: bool) {
    let root = synthetic_root(name);
    let (url, requests) = serve();
    let report = root.join("report");
    let log = logged.then(|| startup_log(&root, &url));
    let mut arguments: Vec<String> = [
        "run",
        "--model",
        root.join("models/example").to_str().unwrap(),
        "--url",
        &url,
        "--check",
        "teacher",
        "--report",
        report.to_str().unwrap(),
    ]
    .iter()
    .map(|word| (*word).to_owned())
    .collect();
    // Outside the full cohort, the log is named relative to this working
    // directory, with a leading `./`: the receipt names it as Python's Path
    // does.
    let (log_argument, log_shown) = match &log {
        Some(log) if !captured => {
            let ups = "../".repeat(std::env::current_dir().unwrap().components().count() - 1);
            let below = log.strip_prefix("/").unwrap().to_str().unwrap();
            (
                Some(format!("./{ups}{below}")),
                Some(format!("{ups}{below}")),
            )
        }
        Some(log) => (
            log.to_str().map(String::from),
            log.to_str().map(String::from),
        ),
        None => (None, None),
    };
    arguments.extend(
        log_argument
            .iter()
            .flat_map(|log| ["--server-log".to_owned(), log.clone()]),
    );
    if captured {
        arguments.extend(["--cohort".to_owned(), "full".to_owned()]);
    }
    let tool = Tool {
        name: "chopinfer eval".into(),
        version: "test".into(),
        revision: None,
    };
    let exit = runtime_eval::cli::run(&tool, &arguments);
    let value = read(&report.join("report.json"));
    let check = &value["checks"][0];
    assert_eq!(exit, Exit::Pass, "{check}");
    assert_eq!(check["status"], "pass");
    assert_eq!(check["record"], "records/teacher.json");
    let record_bytes = std::fs::read(report.join("records/teacher.json")).unwrap();
    assert_eq!(check["record_sha256"], sha256_hex(&record_bytes));
    as_python_writes_receipts(&report.join("records/teacher.json"), &record_bytes);
    assert_eq!(check["limits"], check["metrics"]["limits"]);
    assert_eq!(check["spread"], check["metrics"]["reference_spread"]);
    let rust = read(&report.join("records/teacher.json"));
    if let Some(log) = &log {
        assert_eq!(
            rust["server"]["checkpoint"]["server_log"],
            log_shown.as_deref().unwrap()
        );
        assert_eq!(
            rust["server"]["checkpoint"]["server_log_sha256"],
            sha256_hex(&std::fs::read(log).unwrap())
        );
    } else {
        assert!(rust["server"]["checkpoint"].get("server_log").is_none());
    }
    // One near-tie flip of 23 positions, recomputed from the served rows.
    assert_eq!(check["metrics"]["scored"], 23);
    assert_eq!(check["metrics"]["top1_near_ties"], 1);
    assert_eq!(check["metrics"]["top1_decisive_flips"], 0);
    let sent: Vec<String> = requests.lock().unwrap().drain(..).collect();
    assert_eq!(sent.len(), 2);

    let capture = report.join("captures/teacher.json");
    if captured {
        // The capture whose digest the receipt names.
        let bytes = std::fs::read(&capture).unwrap();
        assert_eq!(rust["capture_sha256"], sha256_hex(&bytes));
        let value = read(&capture);
        assert_eq!(value["schema"], "example.capture.v1");
        assert_eq!(value["server"], rust["server"]);
        assert_eq!(value["texts"][0]["margin"][0], 0.25);
    } else {
        assert!(!capture.exists());
        assert!(rust.get("capture_sha256").is_none());
    }
    let mut varying = Vec::new();
    if let Some(log) = &log {
        varying.push((log_shown.clone().unwrap(), "{log}"));
        varying.push((sha256_hex(&std::fs::read(log).unwrap()), "{log_sha256}"));
    }
    if captured {
        varying.push((
            rust["capture_sha256"].as_str().unwrap().to_owned(),
            "{capture_sha256}",
        ));
    }
    varying.push((root.to_str().unwrap().to_owned(), "{root}"));
    varying.push((url.clone(), "{url}"));
    varying.push((url.trim_start_matches("http://").to_owned(), "{address}"));
    let bytes = captured.then(|| std::fs::read(&capture).unwrap());
    as_retired_python_wrote(
        &format!("teacher/{name}"),
        &invariant(&sent, &rust, bytes.as_deref(), &varying),
    );
    std::fs::remove_dir_all(root).unwrap();
}

/// Asserts that `bytes`, the receipt at `path`, are its content serialized
/// as the Python receipts are written (`common.receipt`: `json.dump(value,
/// indent=2, sort_keys=True)` and a newline), so a change of whitespace, key
/// order or number formatting fails here as well as the content digest.
fn as_python_writes_receipts(path: &Path, bytes: &[u8]) {
    let Ok(python) = Command::new("python3").arg("--version").output() else {
        eprintln!("python3 is missing: the receipt serialization cross-check is skipped");
        return;
    };
    assert!(python.status.success());
    let written = Command::new("python3")
        .arg("-B")
        .arg("-c")
        .arg(
            "import json, sys\n\
             value = json.loads(open(sys.argv[1], 'rb').read())\n\
             sys.stdout.buffer.write((json.dumps(value, indent=2, sort_keys=True) + '\\n').encode())\n",
        )
        .arg(path)
        .output()
        .unwrap();
    assert!(
        written.status.success(),
        "{}",
        String::from_utf8_lossy(&written.stderr)
    );
    assert!(
        written.stdout == bytes,
        "{}: the receipt is not serialized as the Python receipts are",
        path.display()
    );
}

/// Asserts that `value`, what a capture wrote that a run does not vary,
/// has the digest tests/data/retired-captures.json records for `name`:
/// what the retired Python capture wrote against the same fake server.
fn as_retired_python_wrote(name: &str, value: &Value) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/retired-captures.json");
    let recorded = runtime_eval::json::parse(&std::fs::read(path).unwrap()).unwrap();
    let digest = canonical_sha256(value);
    if recorded["captures"][name].as_str() != Some(digest.as_str()) {
        let written = std::env::temp_dir().join(format!(
            "runtime-eval-{}-{}.json",
            name.replace('/', "-"),
            std::process::id()
        ));
        std::fs::write(&written, canonical(value)).unwrap();
        panic!(
            "{name}: eval sent or wrote other than the retired Python capture: sha256 {digest}, recorded {}; \
             written to {}",
            recorded["captures"][name],
            written.display()
        );
    }
}

#[test]
fn a_startup_log_of_another_process_fails_the_run() {
    let root = synthetic_root("stale-log");
    let (url, requests) = serve();
    let log = root.join("stale.log");
    let identity = json!({"checkpoint_identity": CHECKPOINT,
                          "startup_nonce": "ffffffffffffffffffffffffffffffff"});
    let listening = json!({"status": "listening", "address": url.trim_start_matches("http://")});
    std::fs::write(
        &log,
        format!("startup serving_identity: {identity}\n{listening}\n"),
    )
    .unwrap();
    let report = root.join("report");
    let arguments: Vec<String> = [
        "run",
        "--model",
        root.join("models/example").to_str().unwrap(),
        "--url",
        &url,
        "--check",
        "teacher",
        "--server-log",
        log.to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
    ]
    .iter()
    .map(|word| (*word).to_owned())
    .collect();
    let tool = Tool {
        name: "chopinfer eval".into(),
        version: "test".into(),
        revision: None,
    };
    assert_eq!(runtime_eval::cli::run(&tool, &arguments), Exit::Fail);
    let value = read(&report.join("report.json"));
    assert_eq!(value["checks"][0]["status"], "fail");
    assert!(
        value["failure"]
            .to_string()
            .contains("is not the log of the live server"),
        "{value}"
    );
    // Refused before any check request.
    assert!(requests.lock().unwrap().is_empty());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_refused_run_leaves_no_capture_a_comparison_of_arms_admits() {
    // Passing numbers, but each scored response names another process.
    let root = synthetic_root("refused-capture");
    let (url, _requests) = serve_answering_as("ffffffffffffffffffffffffffffffff");
    let log = startup_log(&root, &url);
    let report = root.join("report");
    let arguments: Vec<String> = [
        "run",
        "--model",
        root.join("models/example").to_str().unwrap(),
        "--url",
        &url,
        "--check",
        "teacher",
        "--cohort",
        "full",
        "--server-log",
        log.to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
    ]
    .iter()
    .map(|word| (*word).to_owned())
    .collect();
    let tool = Tool {
        name: "chopinfer eval".into(),
        version: "test".into(),
        revision: None,
    };
    assert_eq!(runtime_eval::cli::run(&tool, &arguments), Exit::Fail);
    let value = read(&report.join("report.json"));
    assert_eq!(value["checks"][0]["status"], "insufficient");
    assert!(
        value["failure"]["message"]
            .as_str()
            .unwrap()
            .starts_with("refused: "),
        "{value}"
    );
    let capture = report.join("captures/teacher.json");
    assert!(!report.join("captures").exists());

    let Ok(python) = Command::new("python3").arg("--version").output() else {
        eprintln!("python3 is missing: the cross-check is skipped");
        return;
    };
    assert!(python.status.success());
    adopt(&root);
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    // A comparison of arms does not admit what the refused run left: the
    // receipt names a capture the run never published.
    let admitted = Command::new("python3")
        .arg("-B")
        .arg("-c")
        .arg(
            "import sys\n\
             import declaration, teacher_arms\n\
             config = declaration.load(sys.argv[1])\n\
             teacher_arms.load_arm('candidate', sys.argv[2], sys.argv[3], config)\n",
        )
        .arg(root.join("models/example/qual.json"))
        .arg(report.join("records/teacher.json"))
        .arg(&capture)
        .env("PYTHONPATH", repo.join("tools/qual"))
        .env("CHOPINFER_QUAL_MODEL_ROOT", &root)
        .output()
        .unwrap();
    let refusal = String::from_utf8_lossy(&admitted.stderr);
    assert!(!admitted.status.success());
    assert!(
        refusal.contains("FileNotFoundError") && refusal.contains(capture.to_str().unwrap()),
        "{refusal}"
    );
    std::fs::remove_dir_all(root).unwrap();
}

/// The greedy requests a server read, each without its cold cache salt,
/// which must be fresh.
fn greedy_requests(requests: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    let mut salts = Vec::new();
    requests
        .lock()
        .unwrap()
        .drain(..)
        .map(|line| {
            let (path, body) = line.split_once(' ').unwrap();
            let mut body = runtime_eval::json::parse(body.as_bytes()).unwrap();
            let salt = body.as_object_mut().unwrap().remove("cache_salt").unwrap();
            let salt = salt.as_str().unwrap().to_owned();
            assert!(salt.starts_with("greedy-cold-") && !salts.contains(&salt));
            salts.push(salt);
            format!("{path} {}", String::from_utf8(canonical(&body)).unwrap())
        })
        .collect()
}

#[test]
fn eval_capture_writes_the_greedy_record() {
    let root = synthetic_root("greedy");
    let (url, requests) = serve();
    let log = startup_log(&root, &url);
    let directory = root.join("rust-records");
    let record = [
        "--arm",
        "ar",
        "--k",
        "0",
        "--cold-cache",
        "--concurrency",
        "1",
        "--prompt-tokens",
        "13",
        "--max-tokens",
        "3",
    ];
    let model = root.join("models/example");
    let mut arguments: Vec<String> = [
        "capture",
        "--kind",
        "greedy_pin",
        "--model",
        model.to_str().unwrap(),
        "--url",
        &url,
        "--served-model",
        "Example",
        "--server-log",
        log.to_str().unwrap(),
        "--report",
        directory.to_str().unwrap(),
    ]
    .iter()
    .map(|word| (*word).to_owned())
    .collect();
    arguments.extend(record.iter().map(|word| (*word).to_owned()));
    let tool = Tool {
        name: "chopinfer eval".into(),
        version: "test".into(),
        revision: None,
    };
    assert_eq!(runtime_eval::cli::run(&tool, &arguments), Exit::Pass);
    let written: Vec<PathBuf> = std::fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(written.len(), 1);
    let name = written[0].file_name().unwrap().to_str().unwrap().to_owned();
    assert!(name.ends_with("-greedy-equality-ar.json"), "{name}");
    let rust = read(&written[0]);
    assert!(name.starts_with(rust["at"].as_str().unwrap()));
    assert_eq!(
        runtime_eval::json::receipt_text(&rust),
        std::fs::read(&written[0]).unwrap()
    );
    assert_eq!(
        rust["server"]["checkpoint"]["checkpoint_identity"],
        CHECKPOINT
    );
    assert_eq!(
        rust["server"]["checkpoint"]["server_log"],
        log.to_str().unwrap()
    );
    assert_eq!(rust["server"]["model"], "Example");
    // The whole record but its stamp, server and host load.
    let pinned = std::fs::read(root.join("fixtures/example/teacher.json")).unwrap();
    let mut record = rust.clone();
    let map = record.as_object_mut().unwrap();
    map.remove("at");
    map.remove("server");
    map.remove("host_load");
    assert_eq!(
        record,
        json!({"schema": "chopinfer-qualification/v1", "check": "greedy-equality-ar",
               "mechanics_only": false, "arm": "ar", "k": 0, "concurrency": 1,
               "prompt_tokens": 13, "max_tokens": 3, "cold_cache": true,
               "reference_sha256": sha256_hex(&pinned),
               "requests": {"t0": [130, 131, 132], "t1": [100, 101, 102]},
               "finish_reasons": {"t0": "stop", "t1": "length"}, "passed": true})
    );
    // One cold request per pinned text, its first min(13, len - 1) tokens,
    // each under a fresh salt.
    let body = |prompt: &str| {
        format!(
            "/v1/completions {{\"max_tokens\":3,\"model\":\"Example\",\"prompt\":[{prompt}],\
             \"return_token_ids\":true,\"skip_special_tokens\":false,\"temperature\":0,\"top_p\":1}}"
        )
    };
    assert_eq!(
        greedy_requests(&requests),
        [
            body("1,2,3,4,5,6,7,8,9,10,11,12,13"),
            body("1,30,31,32,33,34,35,36,37,38")
        ]
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn eval_capture_refuses_what_is_not_a_record() {
    let root = synthetic_root("capture-usage");
    let model = root.join("models/example");
    let model = model.to_str().unwrap();
    let tool = Tool {
        name: "chopinfer eval".into(),
        version: "test".into(),
        revision: None,
    };
    let base = [
        "--model",
        model,
        "--url",
        "http://127.0.0.1:9",
        "--report",
        "/nonexistent/r",
    ];
    for words in [
        &["--kind", "teacher_forced", "--arm", "ar"][..],
        &["--kind", "greedy_pin", "--exact", "example", "--arm", "ar"],
        &["--kind", "greedy_pin", "--arm", "ar", "--arm", "k3"],
        &["--kind", "determinism", "--arm", "ar", "--k", "0"],
        &["--kind", "determinism", "--arm"],
        &["--arm", "ar"],
        &[
            "--kind",
            "determinism",
            "--kind",
            "determinism",
            "--arm",
            "ar",
        ],
    ] {
        let mut arguments = vec!["capture"];
        arguments.extend(base);
        arguments.extend(words);
        let arguments: Vec<String> = arguments.iter().map(|word| (*word).to_owned()).collect();
        assert_eq!(
            runtime_eval::cli::run(&tool, &arguments),
            Exit::Usage,
            "{words:?}"
        );
    }
    for missing in ["--url", "--report"] {
        let mut arguments: Vec<String> =
            vec!["capture".into(), "--kind".into(), "determinism".into()];
        let mut words = base.iter();
        while let Some(word) = words.next() {
            let value = words.next().unwrap();
            if *word != missing {
                arguments.extend([(*word).to_owned(), (*value).to_owned()]);
            }
        }
        assert_eq!(
            runtime_eval::cli::run(&tool, &arguments),
            Exit::Usage,
            "{missing}"
        );
    }
    // An unreachable server is the environment's; nothing is written.
    let report = root.join("unreachable");
    let arguments: Vec<String> = [
        "capture",
        "--kind",
        "determinism",
        "--model",
        model,
        "--url",
        "http://127.0.0.1:9",
        "--report",
        report.to_str().unwrap(),
        "--arm",
        "ar",
        "--run-id",
        "r",
    ]
    .iter()
    .map(|word| (*word).to_owned())
    .collect();
    assert_eq!(runtime_eval::cli::run(&tool, &arguments), Exit::Environment);
    assert!(!report.exists());
    std::fs::remove_dir_all(root).unwrap();
}
