//! The determinism kind against `tools/qual/determinism.py`. Records
//! changed one field at a time are rescored by both implementations, which
//! must agree on the status, the error text and the metrics to the bit. The
//! live capture and Python's `record` run against the same fake server and
//! must send the same requests and write the same receipt, as must `eval
//! capture --kind determinism`. The comparisons with Python are skipped
//! when python3 is missing.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use runtime_eval::declaration::{self, Check, Loaded};
use runtime_eval::http::{BoundServer, Client, ServerIdentity};
use runtime_eval::json::{canonical, canonical_sha256, compact, parse};
use runtime_eval::kind::{self, Capture, Kind, KindError};
use runtime_eval::kinds;
use serde_json::{Value, json};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs `script` under python3 with tools/qual importable, `input` on
/// stdin and `environment` set; its stdout, or `None` when python3 is
/// missing.
fn python(script: &str, input: &[u8], environment: &[(&str, &Path)]) -> Option<Vec<u8>> {
    let mut child = match Command::new("python3")
        .envs(environment.iter().copied())
        .arg("-B")
        .arg("-c")
        .arg(format!(
            "import sys\nsys.path.insert(0, {:?})\n{script}",
            root().join("tools/qual").to_str().unwrap()
        ))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("skipped: python3 is missing");
            return None;
        }
        Err(error) => panic!("python3: {error}"),
    };
    child.stdin.take().unwrap().write_all(input).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "python3 failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(output.stdout)
}

fn determinism() -> Box<dyn Kind> {
    kinds::all()
        .into_iter()
        .find(|kind| kind.info().name == "determinism")
        .unwrap()
}

fn check() -> Check {
    Check {
        id: "determinism".into(),
        module: "determinism".into(),
        timeout_seconds: 60.0,
        rules: Vec::new(),
        informational_reason: None,
        declared: json!({}),
    }
}

fn pass_vector() -> Value {
    let file =
        parse(&std::fs::read(root().join("tests/qual-vectors/determinism.json")).unwrap()).unwrap();
    file["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|vector| vector["name"] == "pass")
        .unwrap()
        .clone()
}

/// The pass vector's records with the value at `pointer` of one of them
/// replaced (or removed when `value` is `None`).
fn changed(side: &str, pointer: &str, value: Option<Value>) -> Value {
    let vector = pass_vector();
    let mut case = json!({"receipt": vector["receipt"], "reference": vector["reference"]});
    let record = &mut case[side];
    match value {
        Some(value) => {
            *record
                .pointer_mut(pointer)
                .unwrap_or_else(|| panic!("{pointer}")) = value
        }
        None => {
            let (parent, key) = pointer.rsplit_once('/').unwrap();
            record
                .pointer_mut(parent)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .remove(key)
                .unwrap();
        }
    }
    case
}

/// Rows of the pass vector by position: 0 capital, 4 long, 5
/// long-cold-repeat, 6 long-ids, 7 long-primer-1024, 8 long-hit-1024, 10
/// long-hit-2000.
fn cases() -> Vec<Value> {
    let receipt = |pointer: &str, value: Value| changed("receipt", pointer, Some(value));
    let reference = |pointer: &str, value: Value| changed("reference", pointer, Some(value));
    vec![
        receipt("/arm", json!("spec")),
        receipt("/passed", json!(1)),
        reference("/mechanics_only", json!(true)),
        receipt("/concurrency", json!(true)),
        receipt("/concurrency", json!(1.0)),
        receipt("/concurrency", json!(2)),
        receipt("/sampling/temperature", json!(0.0)),
        receipt("/sampling/top_p", json!(false)),
        receipt("/image/included", json!(0)),
        receipt("/image", json!([])),
        receipt("/image", json!([1])),
        receipt("/request_set/sha256", json!("")),
        receipt("/request_set", json!("names")),
        receipt("/requests", json!({})),
        receipt("/requests", json!("rows")),
        receipt("/requests", json!(5)),
        receipt("/requests/0", json!(null)),
        receipt("/cold_repeat/supported", json!(1)),
        receipt("/prefix_hit/anchors", json!([1024.0, 2000])),
        receipt("/prefix_hit/anchors", json!([1024])),
        receipt("/prefix_hit", json!([0])),
        receipt("/requests/1/cache_salt", json!("salt-capital")),
        receipt("/requests/6/cache_salt", json!("")),
        receipt("/requests/8/cache_salt", json!("salt-long-ids")),
        receipt("/requests/0/finish_reason", json!("abort")),
        receipt("/requests/0/text", json!(null)),
        receipt(
            "/requests/0/usage/prompt_tokens_details/cached_tokens",
            json!(0.0),
        ),
        receipt("/requests/0/usage/prompt_tokens_details", json!(null)),
        receipt("/requests/0/usage/prompt_tokens_details", json!("details")),
        receipt("/requests/0/usage", json!([1])),
        receipt(
            "/requests/8/usage/prompt_tokens_details/cached_tokens",
            json!(true),
        ),
        receipt(
            "/requests/8/usage/prompt_tokens_details/cached_tokens",
            json!(1025),
        ),
        receipt(
            "/requests/8/usage/prompt_tokens_details/cached_tokens",
            json!(1024.0),
        ),
        receipt(
            "/requests/10/usage/prompt_tokens_details/cached_tokens",
            json!(0),
        ),
        receipt("/requests/0/token_ids", json!([1, true])),
        receipt("/requests/0/token_ids", json!([])),
        receipt("/requests/0/usage/completion_tokens", json!(2.0)),
        receipt("/requests/0/usage/completion_tokens", json!(3)),
        receipt("/requests/7/token_ids", json!([1, 2])),
        receipt("/requests/0/tokens", json!([" a"])),
        receipt("/requests/0/logprobs/0", json!(true)),
        receipt("/requests/0/logprobs/0", json!(-1)),
        receipt("/requests/0/top_logprobs/0", json!([[" a", -0.125, 1]])),
        receipt("/requests/0/top_logprobs/0", json!([])),
        receipt("/requests/0/top_logprobs/0/0/0", json!(7)),
        receipt("/requests/4/usage/prompt_tokens", json!(2048)),
        receipt("/requests/4/usage/prompt_tokens", json!(2049.0)),
        receipt("/requests/5/usage/prompt_tokens", json!(2801)),
        receipt("/request_set/long_prompt_tokens", json!(2000)),
        receipt("/request_set/long_prompt_token_ids_sha256", json!(null)),
        receipt("/requests/7/usage/prompt_tokens", json!(1025)),
        receipt("/requests/6/usage/prompt_tokens", json!(2800.0)),
        // Scored differences.
        receipt("/requests/0/logprobs/1", json!(-0.25000000000000006)),
        receipt("/requests/0/logprobs/0", json!(-1.5)),
        receipt("/requests/0/tokens/0", json!(" z")),
        receipt("/requests/0/token_ids/1", json!(9)),
        receipt("/requests/0/text", json!(" az")),
        receipt("/requests/0/reasoning", json!("")),
        receipt("/requests/0/finish_reason", json!("stop")),
        receipt(
            "/requests/0/top_logprobs/1",
            json!([["b", -0.25], ["c", -2.0]]),
        ),
        reference(
            "/requests/0/top_logprobs/1",
            json!([["b", -0.5], ["c", -2.0], ["b", -0.375]]),
        ),
        reference("/requests/5/logprobs/1", json!(-0.0625)),
        receipt("/requests/10/token_ids/1", json!(7)),
        receipt("/requests/10/tokens/1", json!("q")),
        reference("/requests/6/logprobs/0", json!(-0.1)),
        reference("/request_set/sha256", json!("dddd")),
        changed("receipt", "/arm", None),
        changed("receipt", "/requests/0/reasoning", None),
        changed("receipt", "/requests/0/usage/completion_tokens", None),
        changed("receipt", "/requests/4/usage/prompt_tokens", None),
        json!({"receipt": pass_vector()["receipt"], "reference": null}),
    ]
}

/// Python's `kinds.rescore` of each case: status, error and metrics.
const RESCORE: &str = "
import json
import kinds
results = []
for case in json.load(sys.stdin):
    result = kinds.rescore(kinds.KINDS['determinism'], case['receipt'], case['reference'], {})
    results.append({'status': result['status'], 'error': result.get('error'), 'metrics': result['metrics']})
json.dump(results, sys.stdout)
";

#[test]
fn changed_records_rescore_as_python_does() {
    let cases = cases();
    let Some(output) = python(RESCORE, &serde_json::to_vec(&cases).unwrap(), &[]) else {
        return;
    };
    let expected = parse(&output).unwrap();
    assert_eq!(expected.as_array().unwrap().len(), cases.len());
    let kind = determinism();
    let mut statuses = std::collections::BTreeSet::new();
    for (case, expected) in cases.iter().zip(expected.as_array().unwrap()) {
        let reference = Some(&case["reference"]).filter(|value| !value.is_null());
        let result = kind::rescore(kind.as_ref(), &case["receipt"], reference, &check(), false);
        let actual = json!({"status": result.status.as_str(), "error": result.error,
                            "metrics": result.metrics});
        // The canonical text tells ints from floats and holds every float's bits.
        assert_eq!(
            String::from_utf8(canonical(&actual)).unwrap(),
            String::from_utf8(canonical(expected)).unwrap(),
            "{case}"
        );
        statuses.insert(result.status.as_str());
    }
    assert_eq!(
        statuses.into_iter().collect::<Vec<_>>(),
        ["fail", "insufficient", "pass"]
    );
}

#[test]
fn a_check_that_sets_a_module_constant_is_refused() {
    let vector = pass_vector();
    let mut check = check();
    check.declared = json!({"constants": {"determinism": {"MAX_TOKENS": {"value": 8}}}});
    let result = kind::rescore(
        determinism().as_ref(),
        &vector["receipt"],
        Some(&vector["reference"]),
        &check,
        false,
    );
    assert_eq!(result.status.as_str(), "fail");
    assert_eq!(
        result.error.as_deref(),
        Some(
            "checks.determinism.constants.determinism.MAX_TOKENS is not applied by chopinfer eval"
        )
    );
}

const CHECKPOINT: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const NONCE: &str = "0123456789abcdef0123456789abcdef";
const MODEL: &str = "served";
/// `sha256(determinism.image_png())`.
const PNG_SHA256: &str = "b7ab2baf457a7323d0977d8d08ad4cddab4ca402fc4515d58351f40eddeda746";

/// The request lines and bodies a fake server read, in order.
type Log = Arc<Mutex<Vec<(String, Value)>>>;

/// The answer of the fake server, computed from the request: every answer
/// a function of its prompt, a cold salt reporting no cached tokens and a
/// salt seen before reporting its first request's prompt tokens, plus
/// `extra_cached`.
fn answer(path: &str, body: &Value, salts: &mut HashMap<String, u64>, extra_cached: u64) -> Value {
    match path {
        "/v1/models" => return json!({"object": "list", "data": [{"id": MODEL}]}),
        "/health" => {
            return json!({"status": "ok",
                          "serving": {"checkpoint_identity": CHECKPOINT, "startup_nonce": NONCE}});
        }
        "/tokenize" => {
            let ids: Vec<u64> = (0..2900).map(|index| index * 37 % 1000 + 3).collect();
            return json!({"tokens": ids, "count": ids.len(), "max_model_len": 8192});
        }
        _ => {}
    }
    let prompt = match &body["prompt"] {
        Value::String(text) => text.len() as u64 / 4,
        Value::Array(ids) => ids.len() as u64,
        _ => 300,
    };
    let salt = body["cache_salt"].as_str().unwrap().to_owned();
    let cached = salts.get(&salt).copied().unwrap_or(0) + extra_cached;
    salts.entry(salt).or_insert(prompt);
    let count = body["max_tokens"].as_u64().unwrap().min(3);
    let ids: Vec<u64> = (0..count).map(|index| (prompt + index) % 50 + 1).collect();
    let tokens: Vec<String> = ids.iter().map(|id| format!(" t{id}")).collect();
    // Sevenths need all 17 digits to read back.
    let logprobs: Vec<f64> = ids.iter().map(|id| -(*id as f64) / 7.0).collect();
    let usage = json!({"prompt_tokens": prompt, "completion_tokens": count,
                       "total_tokens": prompt + count,
                       "prompt_tokens_details": {"cached_tokens": cached}});
    let choice = if path == "/v1/chat/completions" {
        let content: Vec<Value> = tokens
            .iter()
            .zip(&logprobs)
            .map(|(token, logprob)| {
                json!({"token": token, "logprob": logprob, "bytes": [],
                       "top_logprobs": [{"token": token, "logprob": logprob},
                                        {"token": " alt", "logprob": logprob - 1.5}]})
            })
            .collect();
        json!({"index": 0, "message": {"role": "assistant", "content": tokens.concat(),
                                       "reasoning_content": null},
               "logprobs": {"content": content}, "finish_reason": "length", "token_ids": ids})
    } else {
        let top: Vec<Value> = tokens
            .iter()
            .zip(&logprobs)
            .map(|(token, logprob)| json!({token.as_str(): logprob, " alt": logprob - 1.5}))
            .collect();
        json!({"index": 0, "text": tokens.concat(),
               "logprobs": {"tokens": tokens, "token_logprobs": logprobs, "top_logprobs": top},
               "finish_reason": "length", "token_ids": ids})
    };
    json!({"id": "x", "object": "text_completion", "model": MODEL, "choices": [choice],
           "usage": usage})
}

/// A fake server answering every request on its own connection; its URL
/// and the log of what it read.
fn fake_server(extra_cached: u64) -> (String, Log) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let log = Log::default();
    let written = log.clone();
    std::thread::spawn(move || {
        let mut salts = HashMap::new();
        for stream in listener.incoming() {
            let mut reader = BufReader::new(stream.unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let mut length = 0;
            loop {
                let mut header = String::new();
                reader.read_line(&mut header).unwrap();
                if let Some(value) = header.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
                if header.trim().is_empty() {
                    break;
                }
            }
            let mut bytes = vec![0; length];
            reader.read_exact(&mut bytes).unwrap();
            let body = if bytes.is_empty() {
                Value::Null
            } else {
                parse(&bytes).unwrap()
            };
            let path = line.split(' ').nth(1).unwrap().to_owned();
            let text = answer(&path, &body, &mut salts, extra_cached).to_string();
            written
                .lock()
                .unwrap()
                .push((line.trim_end().to_owned(), body));
            let mut stream = reader.into_inner();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
                 connection: close\r\nx-chopinfer-checkpoint-identity: {CHECKPOINT}\r\n\
                 x-chopinfer-startup-nonce: {NONCE}\r\n\r\n{text}",
                text.len()
            )
            .unwrap();
        }
    });
    (url, log)
}

/// A root whose model declares the determinism check with `args` and a
/// frontend whose thinking switch is `thinking`.
fn declared(name: &str, args: &[&str]) -> Loaded {
    let root = std::env::temp_dir().join(format!(
        "runtime-eval-determinism-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let model = root.join("models/example");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::create_dir_all(root.join("fixtures/example")).unwrap();
    std::fs::create_dir_all(root.join("tools/qual")).unwrap();
    std::fs::copy(
        self::root().join("tools/qual/modules.json"),
        root.join("tools/qual/modules.json"),
    )
    .unwrap();
    let frontend = json!({"reasoning": {"thinking_keywords": ["thinking", "think"]}});
    std::fs::write(
        root.join("fixtures/example/frontend.json"),
        frontend.to_string(),
    )
    .unwrap();
    let lock = json!({"model.safetensors": {"sha256": "ab", "size": 1}});
    std::fs::write(model.join("checkpoint-files.json"), lock.to_string()).unwrap();
    let pin = json!({"repository": "org/example", "revision": "abc",
                     "served_model_name": MODEL, "source_lock_sha256": canonical_sha256(&lock),
                     "checkpoint_identity": CHECKPOINT,
                     "lock": "models/example/checkpoint-files.json"});
    let declaration = json!({
        "schema": "chopinfer-qual-model/v1", "model": "example",
        "cohorts": {"quick": {"checks": ["determinism"]}},
        "change_classes": {"frontend": [], "quality": [], "service": [], "speculation": [], "prefix": []},
        "checks": {"determinism": {"module": "determinism", "args": args, "timeout_seconds": 60,
                                   "receipt": "*-determinism-*.json",
                                   "rules": [{"path": ["passed"], "op": "eq", "value": true,
                                              "source": "docs/check-kinds.md"}]}},
        "inputs": {"frontend": "fixtures/example/frontend.json"},
        "qualification": {"teacher_reference_kind": "exact", "teacher_receipt": "teacher",
                          "teacher_capture_schema": "example.capture.v1", "verify_buckets": [2],
                          "chars_per_token": 4.0, "checkpoint": pin},
        "receipts": {"env": "RUNTIME_EVAL_TEST_UNSET", "path": "qual/example"}
    });
    std::fs::write(model.join("qual.json"), declaration.to_string()).unwrap();
    declaration::load(&model).unwrap()
}

/// The determinism capture of the declared check against `url`, within
/// `budget`.
fn capture(loaded: &Loaded, url: &str, budget: Duration) -> Result<Value, KindError> {
    let mut client = Client::new(url, Duration::from_secs(30), 0).unwrap();
    let server = ServerIdentity {
        url: url.into(),
        model: MODEL.into(),
        models: vec![MODEL.into()],
        health: None,
    };
    let bound = BoundServer {
        checkpoint_identity: CHECKPOINT.into(),
        startup_nonce: NONCE.into(),
        package_identity: None,
        metadata_identity: None,
        server_log: None,
    };
    let mut capture = Capture {
        client: &mut client,
        declaration: loaded,
        check: &loaded.declaration.checks[0],
        server: &server,
        bound: &bound,
        reference: None,
        deadline: Instant::now() + budget,
        host_load_start: runtime_eval::capture::host_load::sample(),
        mechanics: false,
        capture_out: None,
    };
    determinism().capture(&mut capture)
}

/// Python's `determinism.record` against `url`: its receipt.
const RECORD: &str = "
import argparse
import json
import determinism
determinism.answer_control = lambda: {'chat_template_kwargs': {'thinking': False}}
determinism.record(argparse.Namespace(**json.load(sys.stdin)))
";

/// The receipt Python's `record` writes for `arguments` against a fresh
/// fake server, and what that server read; `None` without python3.
fn python_record(name: &str, arguments: Value) -> Option<(Value, Log)> {
    let (url, log) = fake_server(0);
    let receipts = std::env::temp_dir().join(format!(
        "runtime-eval-determinism-receipts-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&receipts);
    let mut arguments = arguments;
    arguments["url"] = json!(url);
    arguments["model"] = Value::Null;
    arguments["timeout"] = json!(30);
    let output = python(
        RECORD,
        arguments.to_string().as_bytes(),
        &[("CHOPINFER_QUAL_RECEIPTS", &receipts)],
    )?;
    let printed = parse(String::from_utf8(output).unwrap().trim().as_bytes()).unwrap();
    let path = printed["receipt"].as_str().unwrap();
    let receipt = parse(&std::fs::read(path).unwrap()).unwrap();
    std::fs::remove_dir_all(&receipts).unwrap();
    Some((receipt, log))
}

/// The POSTs a server read: request line and compact body without its
/// cache salt; and the salts in order.
fn posts(log: &Log) -> (Vec<(String, String)>, Vec<Value>) {
    let mut requests = Vec::new();
    let mut salts = Vec::new();
    for (line, body) in log.lock().unwrap().iter() {
        if !line.starts_with("POST ") {
            continue;
        }
        let mut body = body.clone();
        if let Some(salt) = body.as_object_mut().unwrap().remove("cache_salt") {
            salts.push(salt);
        }
        requests.push((line.clone(), String::from_utf8(compact(&body)).unwrap()));
    }
    (requests, salts)
}

/// Every request's own fresh salt, `determinism-<name>-<uuid4 hex>`, and
/// each hit its primer's; the salts are those the server read.
fn check_salts(receipt: &Value, sent: &[Value]) {
    let rows = receipt["requests"].as_array().unwrap();
    let recorded: Vec<Value> = rows.iter().map(|row| row["cache_salt"].clone()).collect();
    assert_eq!(&recorded, sent);
    let mut own = Vec::new();
    for row in rows {
        let name = row["name"].as_str().unwrap();
        let salt = row["cache_salt"].as_str().unwrap();
        if let Some(anchor) = name.strip_prefix("long-hit-") {
            let primer = rows
                .iter()
                .find(|row| row["name"] == format!("long-primer-{anchor}"))
                .unwrap();
            assert_eq!(salt, primer["cache_salt"], "{name}");
            continue;
        }
        let hex = salt
            .strip_prefix(&format!("determinism-{name}-"))
            .unwrap_or_else(|| panic!("{salt}"));
        assert!(
            hex.len() == 32
                && hex
                    .bytes()
                    .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
                && &hex[12..13] == "4"
                && "89ab".contains(&hex[16..17]),
            "{salt}"
        );
        assert!(!own.contains(&salt), "{salt}");
        own.push(salt);
    }
}

/// A receipt without what differs by run: the time, the server identity
/// (Python's names its health), the host load, and each request's timing
/// and salt.
fn comparable(receipt: &Value) -> Value {
    let mut receipt = receipt.clone();
    let map = receipt.as_object_mut().unwrap();
    map.remove("at");
    map.remove("server");
    map.remove("host_load");
    for row in map["requests"].as_array_mut().unwrap() {
        let row = row.as_object_mut().unwrap();
        row.remove("seconds");
        row.remove("cache_salt");
    }
    receipt
}

fn records_as_python_does(name: &str, image: bool) {
    let mut args = vec!["--arm", "spec", "--run-id", "run-1"];
    if image {
        args.extend(["--image", "--image-reason", "the test serves images"]);
    }
    let loaded = declared(name, &args);
    let (url, log) = fake_server(0);
    let receipt = capture(&loaded, &url, Duration::from_secs(60)).unwrap();
    let (requests, salts) = posts(&log);
    check_salts(&receipt, &salts);
    assert_eq!(receipt["check"], "determinism-spec");
    assert_eq!(receipt["mechanics_only"], false);
    assert_eq!(
        receipt["server"]["checkpoint"]["checkpoint_identity"],
        CHECKPOINT
    );
    assert_eq!(
        receipt["image"]["png_sha256"],
        if image {
            json!(PNG_SHA256)
        } else {
            Value::Null
        }
    );
    assert_eq!(receipt["cold_repeat"]["equal"], true);
    assert_eq!(receipt["prefix_hit"]["equal"], true);
    // Two arms recorded alike pass.
    let kind = determinism();
    let check = &loaded.declaration.checks[0];
    let result = kind::rescore(kind.as_ref(), &receipt, Some(&receipt), check, false);
    assert_eq!(result.status.as_str(), "pass", "{:?}", result.error);
    let arguments = json!({"arm": "spec", "run_id": "run-1", "image": image,
                           "image_reason": if image { json!("the test serves images") } else { Value::Null }});
    let Some((python, python_log)) = python_record(name, arguments) else {
        return;
    };
    let (python_requests, python_salts) = posts(&python_log);
    check_salts(&python, &python_salts);
    assert_eq!(requests, python_requests);
    assert_eq!(
        String::from_utf8(canonical(&comparable(&receipt))).unwrap(),
        String::from_utf8(canonical(&comparable(&python))).unwrap()
    );
    let result = kind::rescore(kind.as_ref(), &receipt, Some(&python), check, false);
    assert_eq!(result.status.as_str(), "pass", "{:?}", result.error);
}

#[test]
fn the_record_sends_and_writes_what_python_does() {
    records_as_python_does("text", false);
}

#[test]
fn the_record_with_the_image_sends_and_writes_what_python_does() {
    records_as_python_does("image", true);
}

#[test]
fn a_record_refuses_cached_cold_requests_and_missing_arguments() {
    let loaded = declared("cached", &["--arm", "spec", "--run-id", "run-1"]);
    let (url, _) = fake_server(5);
    assert_eq!(
        capture(&loaded, &url, Duration::from_secs(60)),
        Err(KindError::Invalid(
            "capital: a cold request requires reported cached_tokens=0, got 5".into()
        ))
    );
    let loaded = declared("unnamed", &["--arm", "spec"]);
    assert_eq!(
        capture(&loaded, &url, Duration::from_secs(60)),
        Err(KindError::Invalid("record needs --arm and --run-id".into()))
    );
}

#[test]
fn an_unreachable_server_or_a_spent_budget_is_the_environments() {
    let loaded = declared("unreachable", &["--arm", "spec", "--run-id", "run-1"]);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    assert!(matches!(
        capture(&loaded, &url, Duration::from_secs(60)),
        Err(KindError::Environment(_))
    ));
    let (url, log) = fake_server(0);
    assert_eq!(
        capture(&loaded, &url, Duration::ZERO),
        Err(KindError::Environment(
            "the check's time budget is exhausted".into()
        ))
    );
    assert!(log.lock().unwrap().is_empty());
}

#[test]
fn eval_capture_writes_the_record_python_writes() {
    let loaded = declared("command", &[]);
    let model = loaded.root.join("models/example");
    let (url, log) = fake_server(0);
    let server_log = loaded.root.join("server.log");
    let identity = json!({"checkpoint_identity": CHECKPOINT, "startup_nonce": NONCE});
    let listening = json!({"status": "listening", "address": url.trim_start_matches("http://")});
    std::fs::write(
        &server_log,
        format!("startup serving_identity: {identity}\n{listening}\n"),
    )
    .unwrap();
    let directory = loaded.root.join("records");
    let arguments: Vec<String> = [
        "capture",
        "--kind",
        "determinism",
        "--model",
        model.to_str().unwrap(),
        "--url",
        &url,
        "--report",
        directory.to_str().unwrap(),
        "--server-log",
        server_log.to_str().unwrap(),
        "--arm",
        "spec",
        "--run-id",
        "run-1",
        "--image",
        "--image-reason",
        "the test serves images",
    ]
    .iter()
    .map(|word| (*word).to_owned())
    .collect();
    let tool = runtime_eval::Tool {
        name: "chopinfer eval".into(),
        version: "test".into(),
        revision: None,
    };
    assert_eq!(
        runtime_eval::cli::run(&tool, &arguments),
        runtime_eval::Exit::Pass
    );
    let written: Vec<PathBuf> = std::fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(written.len(), 1);
    assert!(
        written[0]
            .to_str()
            .unwrap()
            .ends_with("-determinism-spec.json")
    );
    let receipt = parse(&std::fs::read(&written[0]).unwrap()).unwrap();
    let (requests, salts) = posts(&log);
    check_salts(&receipt, &salts);
    assert_eq!(
        receipt["server"]["checkpoint"]["server_log"],
        server_log.to_str().unwrap()
    );
    assert_eq!(receipt["image"]["png_sha256"], PNG_SHA256);
    let arguments = json!({"arm": "spec", "run_id": "run-1", "image": true,
                           "image_reason": "the test serves images"});
    let Some((python, python_log)) = python_record("command", arguments) else {
        return;
    };
    let (python_requests, python_salts) = posts(&python_log);
    check_salts(&python, &python_salts);
    assert_eq!(requests, python_requests);
    assert_eq!(
        String::from_utf8(canonical(&comparable(&receipt))).unwrap(),
        String::from_utf8(canonical(&comparable(&python))).unwrap()
    );
    // A log of another process refuses the record before any request.
    let (url, log) = fake_server(0);
    let stale = loaded.root.join("stale.log");
    std::fs::write(
        &stale,
        format!("startup serving_identity: {identity}\n{listening}\n"),
    )
    .unwrap();
    let arguments: Vec<String> = [
        "capture",
        "--kind",
        "determinism",
        "--model",
        model.to_str().unwrap(),
        "--url",
        &url,
        "--report",
        directory.to_str().unwrap(),
        "--server-log",
        stale.to_str().unwrap(),
        "--arm",
        "ar",
        "--run-id",
        "run-1",
    ]
    .iter()
    .map(|word| (*word).to_owned())
    .collect();
    assert_eq!(
        runtime_eval::cli::run(&tool, &arguments),
        runtime_eval::Exit::Fail
    );
    assert!(posts(&log).0.is_empty());
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
}
