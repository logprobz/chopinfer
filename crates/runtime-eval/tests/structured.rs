//! The structured kind's capture against `tools/qual/structured.py`. The
//! live capture runs against fake servers that answer every case from the
//! same list of outputs (valid, malformed, schema violations, refusals,
//! truncations, tool calls, plain and streamed). A mechanics capture and
//! Python's `structured.main --mechanics-only` must send the same requests
//! and write the same receipt but for timing; every other capture must
//! send and write what the retired Python capture did
//! (tests/data/retired-captures.json). Python's `json.loads` and the port
//! decode the same texts alike. The comparisons with Python are skipped
//! when python3 is missing.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use runtime_eval::declaration::{self, Loaded};
use runtime_eval::http::{BoundServer, Client, ServerIdentity};
use runtime_eval::json::{canonical, canonical_sha256, compact, parse};
use runtime_eval::kind::{self, Capture, Kind, KindError, Status};
use runtime_eval::kinds;
use runtime_eval::pyjson::{self, Object};
use serde_json::{Value, json};

const CHECKPOINT: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const NONCE: &str = "0123456789abcdef0123456789abcdef";
const MODEL: &str = "served";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs `script` under python3 with tools/qual importable, `input` on
/// stdin and `environment` set; its stdout, or `None` when python3 is
/// missing.
fn python(script: &str, input: &[u8], environment: &[(&str, &Path)]) -> Option<Vec<u8>> {
    let mut child = match Command::new("python3")
        .env_remove("CHOPINFER_QUAL_CONFIG")
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

fn structured() -> Box<dyn Kind> {
    kinds::all()
        .into_iter()
        .find(|kind| kind.info().name == "structured")
        .unwrap()
}

/// `structured.self_test`'s valid person.
fn person() -> Value {
    json!({"name": "Ada", "age": 36, "status": "active", "kind": "person",
           "address": {"city": "Paris", "zip": "75001"}, "tags": ["math"], "active": true,
           "nickname": null, "note": null})
}

/// The person with `changes` (a null removes a field).
fn person_with(changes: Value) -> String {
    let mut value = person();
    for (key, change) in changes.as_object().unwrap() {
        match change {
            Value::Null => {
                value.as_object_mut().unwrap().remove(key);
            }
            change => value[key] = change.clone(),
        }
    }
    value.to_string()
}

fn text(content: &str) -> Value {
    json!({"content": content})
}

fn calls(calls: &[(&str, &str)]) -> Value {
    json!({"calls": calls})
}

/// Outputs every case can get: truncations and error statuses.
fn common_outputs() -> Vec<Value> {
    vec![
        json!({"content": "y".repeat(150) + &"z".repeat(150), "finish": "length"}),
        json!({"content": "", "finish": "length"}),
        json!({"status": 400, "body": "{\"error\": \"unsupported schema\"}"}),
        json!({"status": 500, "body": "boom"}),
        json!({"status": 422, "body": "unprocessable"}),
    ]
}

/// Texts a JSON case can get, valid and not.
fn json_outputs() -> Vec<Value> {
    let mut outputs: Vec<Value> = [
        person().to_string(),
        person_with(json!({"nickname": "Countess", "age": 36.0})),
        person_with(json!({"age": 1e2})),
        String::new(),
        "Sure! {\"a\": 1}".into(),
        "{\"name\": \"Ada\",}".into(),
        "{\"name\" \"Ada\"}".into(),
        "[1, 2]".into(),
        "{\"a\": 1} trailing".into(),
        "{\"a\": \"unterminated".into(),
        "{\"a\": \"bad \\q escape\"}".into(),
        "{\"a\": \"\\u12\"}".into(),
        "{\"a\": \"\\u12zz\"}".into(),
        "{\"a\": \"\\ud83d\\ude00\", \"b\": \"\\u00e9\"}".into(),
        "{\"a\": NaN, \"b\": Infinity, \"c\": -Infinity, \"d\": 1e400, \"e\": -0.0}".into(),
        "{\"name\": \"x\", \"name\": \"y\"}".into(),
        "\n\n  {\"name\": \n \"Ada\"".into(),
        "{\"a\":\t\"tab\tinside\"}".into(),
        format!("1{}", "0".repeat(4400)),
        "x".repeat(600),
        "nul".into(),
        "{\"a\": tru}".into(),
        "\u{feff}{}".into(),
        "{\"ä\": \"ü\", \"emoji\": \"😀\"}".into(),
        "-".into(),
        "01".into(),
        "1.".into(),
        "1e+".into(),
        "{\"a\":[1,2,}".into(),
        "{\"a\" : 1 , }".into(),
        "{\"a\":1}}".into(),
        "\"just a string\"".into(),
        "null".into(),
        "-Infinity".into(),
        "{}".into(),
    ]
    .iter()
    .map(|content| text(content))
    .collect();
    for changes in [
        json!({"age": 151}),
        json!({"age": -1}),
        json!({"age": 3.5}),
        json!({"age": true}),
        json!({"age": 12345678901234567890123456789012345678901234567890.0}),
        json!({"status": "gone"}),
        json!({"kind": "robot"}),
        json!({"kind": 1}),
        json!({"address": {"city": "Paris"}}),
        json!({"address": {"city": "Paris", "zip": "1", "x": 0}}),
        json!({"tags": []}),
        json!({"tags": ["a", "b", "c", "d"]}),
        json!({"tags": [1, ["x"]]}),
        json!({"name": "x".repeat(41)}),
        json!({"active": 1}),
        json!({"nickname": 5}),
        json!({"note": "x"}),
        json!({"extra": 1}),
        json!({"age": null, "name": null, "x": 1, "y": 2, "tags": "t", "note": 0}),
    ] {
        outputs.push(text(&person_with(changes)));
    }
    let big = person().to_string().replace(
        "\"age\":36",
        "\"age\":1234567890123456789012345678901234567890123456789",
    );
    outputs.push(text(&big));
    outputs.push(text(
        &person().to_string().replace("\"age\":36", "\"age\":NaN"),
    ));
    outputs.extend(common_outputs());
    outputs
}

/// The outputs of `case`, served in turn.
fn outputs(case: &str) -> Vec<Value> {
    match case {
        "chat_so_choice" => [
            "neutral",
            "positive",
            "Positive",
            "",
            "positive ",
            "\"neutral\"",
            "négatif",
            "it's",
            "a'b\"c",
            "\n",
            "tab\there",
            "\u{7f}del",
            "😀",
        ]
        .iter()
        .map(|content| text(content))
        .chain([text(&"x".repeat(600))])
        .chain(common_outputs())
        .collect(),
        "chat_so_regex" => [
            "555-0123",
            "555-01234",
            "tel 555-0123",
            "٥٥٥-٠١٢٣",
            "555-0123\n",
            "",
            "555‐0123",
            "5550123",
            "１２３-４５６７",
        ]
        .iter()
        .map(|content| text(content))
        .chain(common_outputs())
        .collect(),
        "tool_required" | "tool_named" => vec![
            calls(&[
                (
                    "get_weather",
                    "{\"city\": \"Paris\", \"unit\": \"celsius\"}",
                ),
                ("get_time", "{\"timezone\": \"Asia/Tokyo\"}"),
            ]),
            calls(&[("get_time", "{\"timezone\": \"Asia/Tokyo\"}")]),
            json!({"calls": [], "content": "I cannot call tools."}),
            calls(&[("get_stock", "{}")]),
            calls(&[("get_weather", "{}")]),
            calls(&[("get_weather", "[]")]),
            calls(&[("get_weather", "{")]),
            calls(&[("get_weather", "{\"city\": \"Paris\", \"unit\": \"kelvin\"}")]),
            calls(&[("get_weather", "{\"city\": 5, \"x\": 1}")]),
            calls(&[("get_weather", "{\"city\": \"Paris\", \"unit\": null}")]),
            calls(&[("get_time", "{\"timezone\": \"UTC\"} extra")]),
            calls(&[("get_time", "")]),
            json!({"raw": [{"function": {"name": "get_time"}}]}),
            json!({"raw": [{"type": "function"}]}),
        ]
        .into_iter()
        .chain(common_outputs())
        .collect(),
        _ => json_outputs(),
    }
}

/// One valid output per case.
fn valid(case: &str) -> Vec<Value> {
    vec![match case {
        "chat_so_choice" => text("neutral"),
        "chat_so_regex" => text("555-0123"),
        "tool_required" | "tool_named" => calls(&[("get_time", "{\"timezone\": \"Asia/Tokyo\"}")]),
        _ => text(&person().to_string()),
    }]
}

/// The case a request is, from its body.
fn case_of(path: &str, body: &Value) -> &'static str {
    if path == "/v1/completions" {
        return if body.get("structured_outputs").is_some() {
            "completion_so_json"
        } else {
            "completion_json_schema"
        };
    }
    if body.get("tools").is_some() {
        return if body["tool_choice"] == "required" {
            "tool_required"
        } else {
            "tool_named"
        };
    }
    if let Some(outputs) = body.get("structured_outputs") {
        for (key, case) in [
            ("json", "chat_so_json"),
            ("choice", "chat_so_choice"),
            ("regex", "chat_so_regex"),
            ("json_object", "chat_so_json_object"),
        ] {
            if outputs.get(key).is_some() {
                return case;
            }
        }
    }
    let format = &body["response_format"];
    if format["type"] == "json_object" {
        "chat_json_object"
    } else if body["chat_template_kwargs"]["thinking"] == true {
        "reasoning_json_schema"
    } else if format["json_schema"]["strict"] == true {
        "chat_json_schema_strict"
    } else {
        "chat_json_schema_loose"
    }
}

/// The non-streamed response body of `output`.
fn response(chat: bool, output: &Value) -> Value {
    let content = output.get("content").cloned().unwrap_or(Value::Null);
    let tokens = content
        .as_str()
        .map_or(3, |text| text.chars().count() / 4 + 1);
    let usage = json!({"prompt_tokens": 20, "completion_tokens": tokens,
                       "total_tokens": 20 + tokens});
    let finish = output.get("finish").cloned().unwrap_or(json!("stop"));
    if !chat {
        return json!({"choices": [{"index": 0, "text": content, "finish_reason": finish}],
                      "usage": usage});
    }
    let mut message = json!({"role": "assistant", "content": content});
    if let Some(calls) = output.get("calls").and_then(Value::as_array) {
        let calls: Vec<Value> = calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                json!({"id": format!("call-{index}"), "type": "function",
                       "function": {"name": call[0], "arguments": call[1]}})
            })
            .collect();
        message["tool_calls"] = json!(calls);
    }
    if let Some(raw) = output.get("raw") {
        message["tool_calls"] = raw.clone();
    }
    json!({"choices": [{"index": 0, "message": message, "finish_reason": finish}],
           "usage": usage})
}

/// The streamed events of `output`: the content in pieces of seven
/// characters, each tool call's name then its arguments in pieces, the
/// finish and the usage.
fn events(chat: bool, output: &Value) -> Vec<Value> {
    let whole = response(chat, output);
    let choice = &whole["choices"][0];
    let pieces = |text: &str| -> Vec<String> {
        let characters: Vec<char> = text.chars().collect();
        characters
            .chunks(7)
            .map(|chunk| chunk.iter().collect())
            .collect()
    };
    let mut events = Vec::new();
    if !chat {
        for piece in pieces(choice["text"].as_str().unwrap_or_default()) {
            events.push(json!({"choices": [{"index": 0, "text": piece}]}));
        }
        events.push(json!({"choices": [{"index": 0, "text": "",
                                        "finish_reason": choice["finish_reason"]}]}));
    } else {
        events.push(json!({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}));
        let message = &choice["message"];
        for piece in pieces(message["content"].as_str().unwrap_or_default()) {
            events.push(json!({"choices": [{"index": 0, "delta": {"content": piece}}]}));
        }
        if output.get("raw").is_some() {
            events.push(json!({"choices": [{"index": 0,
                                            "delta": {"tool_calls": message["tool_calls"]}}]}));
        } else if let Some(calls) = message["tool_calls"].as_array() {
            for (index, call) in calls.iter().enumerate() {
                events.push(json!({"choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": index, "id": call["id"], "type": "function",
                     "function": {"name": call["function"]["name"], "arguments": ""}}]}}]}));
                for piece in pieces(call["function"]["arguments"].as_str().unwrap()) {
                    events.push(json!({"choices": [{"index": 0, "delta": {"tool_calls": [
                        {"index": index, "function": {"arguments": piece}}]}}]}));
                }
            }
        }
        events.push(json!({"choices": [{"index": 0, "delta": {},
                                        "finish_reason": choice["finish_reason"]}]}));
    }
    events.push(json!({"choices": [], "usage": whole["usage"]}));
    events
}

type Log = Arc<Mutex<Vec<(String, Value)>>>;
type Outputs = fn(&str) -> Vec<Value>;

/// A fake server answering each case's requests with its outputs in turn
/// (plain and streamed requests each from the first); its URL and the log
/// of what it read.
fn fake_server(outputs: Outputs) -> (String, Log) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let log = Log::default();
    let written = log.clone();
    std::thread::spawn(move || {
        let mut served: HashMap<(&str, bool), usize> = HashMap::new();
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
            written
                .lock()
                .unwrap()
                .push((line.trim_end().to_owned(), body.clone()));
            let (status, kind, text) = match path.as_str() {
                "/v1/models" => (
                    "200 OK",
                    "application/json",
                    json!({"data": [{"id": MODEL}]}).to_string(),
                ),
                "/health" => ("200 OK", "application/json", "{}".into()),
                _ => {
                    let case = case_of(&path, &body);
                    let streamed = body["stream"] == true;
                    let turn = served.entry((case, streamed)).or_default();
                    let list = outputs(case);
                    let output = list[*turn % list.len()].clone();
                    *turn += 1;
                    let chat = path == "/v1/chat/completions";
                    if let Some(status) = output.get("status") {
                        let status = match status.as_u64().unwrap() {
                            400 => "400 Bad Request",
                            422 => "422 Unprocessable Entity",
                            _ => "500 Internal Server Error",
                        };
                        (
                            status,
                            "text/plain",
                            output["body"].as_str().unwrap().into(),
                        )
                    } else if streamed {
                        let mut text = String::new();
                        for event in events(chat, &output) {
                            text.push_str(&format!("data: {event}\n\n"));
                        }
                        text.push_str("data: [DONE]\n\n");
                        ("200 OK", "text/event-stream", text)
                    } else {
                        (
                            "200 OK",
                            "application/json",
                            response(chat, &output).to_string(),
                        )
                    }
                }
            };
            let mut stream = reader.into_inner();
            write!(
                stream,
                "HTTP/1.1 {status}\r\ncontent-type: {kind}\r\ncontent-length: {}\r\n\
                 connection: close\r\nx-chopinfer-checkpoint-identity: {CHECKPOINT}\r\n\
                 x-chopinfer-startup-nonce: {NONCE}\r\n\r\n{text}",
                text.len()
            )
            .unwrap();
        }
    });
    (url, log)
}

/// A root whose model declares the structured check with `args` and
/// `constants`, and the path of its frontend.
fn declared(name: &str, args: &[&str], constants: Value) -> (Loaded, PathBuf) {
    let root = std::env::temp_dir().join(format!(
        "runtime-eval-structured-{name}-{}",
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
    let frontend = root.join("fixtures/example/frontend.json");
    std::fs::write(
        &frontend,
        json!({"reasoning": {"thinking_keywords": ["thinking", "think"]}}).to_string(),
    )
    .unwrap();
    let lock = json!({"model.safetensors": {"sha256": "ab", "size": 1}});
    std::fs::write(model.join("checkpoint-files.json"), lock.to_string()).unwrap();
    let pin = json!({"repository": "org/example", "revision": "abc",
                     "served_model_name": MODEL, "source_lock_sha256": canonical_sha256(&lock),
                     "checkpoint_identity": CHECKPOINT,
                     "lock": "models/example/checkpoint-files.json"});
    let mut declared_args = vec!["--url", "{url}", "--model", "{model}"];
    declared_args.extend(args);
    let declaration = json!({
        "schema": "chopinfer-qual-model/v1", "model": "example",
        "cohorts": {"quick": {"checks": ["structured"]}},
        "change_classes": {"frontend": [], "quality": [], "service": [], "speculation": [], "prefix": []},
        "checks": {"structured": {"module": "structured", "args": declared_args,
                                  "timeout_seconds": 60, "receipt": "*-structured.json",
                                  "constants": constants,
                                  "rules": [{"path": ["passed"], "op": "eq", "value": true,
                                             "source": "docs/check-kinds.md"}]}},
        "inputs": {"frontend": "fixtures/example/frontend.json"},
        "qualification": {"teacher_reference_kind": "exact", "teacher_receipt": "teacher",
                          "teacher_capture_schema": "example.capture.v1", "verify_buckets": [2],
                          "chars_per_token": 4.0, "checkpoint": pin},
        "receipts": {"env": "RUNTIME_EVAL_TEST_UNSET", "path": "qual/example"}
    });
    std::fs::write(model.join("qual.json"), declaration.to_string()).unwrap();
    (declaration::load(&model).unwrap(), frontend)
}

/// The structured capture of the declared check against `url`, within
/// `budget`.
fn capture(
    loaded: &Loaded,
    url: &str,
    budget: Duration,
    mechanics: bool,
) -> Result<Value, KindError> {
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
        mechanics,
        capture_out: None,
    };
    structured().capture(&mut capture)
}

/// Python's `structured.main` (a mechanics capture: `flags` hold
/// --mechanics-only) against `url` with the frontend's thinking keyword and
/// the module constants: its receipt.
const MAIN: &str = "
import json
import common
import structured
spec = json.load(sys.stdin)
structured.thinking_kwargs = lambda enabled: {'thinking': enabled}
allowed = json.loads(open(spec['frontend']).read())['reasoning'].get('allow_disabled', True) is not False
structured.thinking_can_be_disabled = lambda: allowed
structured.answer_kwargs = lambda: {'thinking': False} if allowed else {}
structured.arguments = lambda description, model_facts=False: common.arguments(description)
for name, entry in spec['constants'].get('structured', {}).items():
    setattr(structured, name, entry['value'])
sys.argv = ['structured.py', '--url', spec['url'], '--timeout', '30'] + spec['flags']
try:
    structured.main()
except SystemExit:
    pass
";

/// The receipt Python's `structured.main` writes against a fresh fake
/// server with `outputs`, and what that server read; `None` without
/// python3.
fn python_main(
    name: &str,
    outputs: Outputs,
    flags: &[&str],
    constants: &Value,
    frontend: &Path,
) -> Option<(Value, Log)> {
    let (url, log) = fake_server(outputs);
    let receipts = std::env::temp_dir().join(format!(
        "runtime-eval-structured-receipts-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&receipts);
    let spec = json!({"url": url, "constants": constants, "flags": flags,
                      "frontend": frontend.to_str().unwrap()});
    let output = python(
        MAIN,
        spec.to_string().as_bytes(),
        &[("CHOPINFER_QUAL_RECEIPTS", &receipts)],
    )?;
    let output = String::from_utf8(output).unwrap();
    let printed = parse(output.trim().lines().last().unwrap().as_bytes()).unwrap();
    let path = printed["receipt"].as_str().unwrap();
    let receipt = parse(&std::fs::read(path).unwrap()).unwrap();
    std::fs::remove_dir_all(&receipts).unwrap();
    Some((receipt, log))
}

/// What a server read, but the identity requests Python's client makes:
/// each request line with its compact body (key order as sent).
fn requests(log: &Log) -> Vec<(String, String)> {
    log.lock()
        .unwrap()
        .iter()
        .filter(|(line, _)| !line.starts_with("GET /v1/models") && !line.starts_with("GET /health"))
        .map(|(line, body)| (line.clone(), String::from_utf8(compact(body)).unwrap()))
        .collect()
}

/// A receipt without what differs by run: the time, the server identity
/// (Python's names its health), the host load and what timing gives:
/// latencies, speeds and their summaries.
fn comparable(receipt: &Value) -> Value {
    let mut receipt = receipt.clone();
    let map = receipt.as_object_mut().unwrap();
    map.remove("at");
    map.remove("server");
    map.remove("host_load");
    for row in map["rows"].as_array_mut().unwrap() {
        let row = row.as_object_mut().unwrap();
        row.remove("latency_seconds");
        row.remove("tokens_per_second");
    }
    let summary = map["summary"].as_object_mut().unwrap();
    let mut summaries = Vec::new();
    for (key, value) in summary.iter_mut() {
        if key == "overall" {
            summaries.push(value);
        } else {
            summaries.extend(value.as_object_mut().unwrap().values_mut());
        }
    }
    for summary in summaries {
        let summary = summary.as_object_mut().unwrap();
        for key in [
            "latency_p50_seconds",
            "latency_p90_seconds",
            "mean_tokens_per_second",
        ] {
            summary.remove(key);
        }
    }
    receipt
}

/// What a capture wrote that a run does not vary: the requests the server
/// read and the comparable receipt.
fn invariant(log: &Log, receipt: &Value) -> Value {
    json!({"requests": requests(log), "receipt": comparable(receipt)})
}

/// Asserts that `value`, what a capture wrote that a run does not vary,
/// has the digest tests/data/retired-captures.json records for `name`:
/// what the retired Python capture wrote against the same fake server.
fn as_retired_python_wrote(name: &str, value: &Value) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/retired-captures.json");
    let recorded = parse(&std::fs::read(path).unwrap()).unwrap();
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

/// Captures with `args` against a fake server and compares the requests
/// and the receipt with Python's mechanics capture against another, or
/// outside a mechanics run with what the retired Python capture wrote;
/// returns the receipt.
fn captures_as_python_does(
    name: &str,
    outputs: Outputs,
    args: &[&str],
    constants: Value,
    mechanics: bool,
) -> Value {
    captured_as_python_does(name, outputs, args, constants, mechanics, false)
}

/// As [`captures_as_python_does`]; where `reasons`, the frontend refuses
/// disabled thinking (`allow_disabled: false`).
fn captured_as_python_does(
    name: &str,
    outputs: Outputs,
    args: &[&str],
    constants: Value,
    mechanics: bool,
    reasons: bool,
) -> Value {
    let (loaded, frontend) = declared(name, args, constants.clone());
    if reasons {
        let value = json!({"reasoning": {"thinking_keywords": ["thinking", "think"],
                                         "allow_disabled": false}});
        std::fs::write(&frontend, value.to_string()).unwrap();
    }
    let (url, log) = fake_server(outputs);
    let receipt = capture(&loaded, &url, Duration::from_secs(120), mechanics).unwrap();
    assert_eq!(receipt["check"], "structured");
    assert_eq!(receipt["mechanics_only"], mechanics);
    assert_eq!(
        receipt["server"]["checkpoint"]["checkpoint_identity"],
        CHECKPOINT
    );
    let flags = [args, &["--mechanics-only"]].concat();
    if !mechanics {
        as_retired_python_wrote(&format!("structured/{name}"), &invariant(&log, &receipt));
    } else if let Some((python, python_log)) =
        python_main(name, outputs, &flags, &constants, &frontend)
    {
        assert_eq!(requests(&log), requests(&python_log));
        let (ours, theirs) = (comparable(&receipt), comparable(&python));
        for (ours, theirs) in ours["rows"]
            .as_array()
            .unwrap()
            .iter()
            .zip(theirs["rows"].as_array().unwrap())
        {
            assert_eq!(
                String::from_utf8(canonical(ours)).unwrap(),
                String::from_utf8(canonical(theirs)).unwrap()
            );
        }
        assert_eq!(
            String::from_utf8(canonical(&ours)).unwrap(),
            String::from_utf8(canonical(&theirs)).unwrap()
        );
    }
    receipt
}

fn statuses(receipt: &Value) -> Vec<&str> {
    receipt["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["status"].as_str().unwrap())
        .collect()
}

/// The verdict of a captured receipt under the documented predicates.
fn rescored(receipt: &Value, mechanics: bool) -> kind::Rescored {
    let (loaded, _) = declared("rescore", &[], json!({}));
    let mut check = loaded.declaration.checks[0].clone();
    let rules = json!([
        {"path": ["passed"], "op": "eq", "value": true},
        {"path": ["summary", "overall", "validity_rate"], "op": "ge", "value": 1.0},
        {"path": ["summary", "overall", "refused"], "op": "eq", "value": 0},
        {"path": ["summary", "overall", "truncated"], "op": "eq", "value": 0}
    ]);
    check.rules = rules
        .as_array()
        .unwrap()
        .iter()
        .map(|rule| runtime_eval::declaration::Predicate {
            path: rule["path"].as_array().unwrap().clone(),
            op: rule["op"].as_str().unwrap().into(),
            value: rule["value"].clone(),
            source: "docs/check-kinds.md".into(),
        })
        .collect();
    kind::rescore(structured().as_ref(), receipt, None, &check, mechanics)
}

#[test]
fn every_case_passes_against_a_conforming_server_as_in_python() {
    let args = [
        "--stream",
        "--repeat",
        "2",
        "--seed",
        "7",
        "--temperature",
        "0.5",
        "--max-tokens",
        "64",
        "--reasoning-max-tokens",
        "128",
    ];
    let receipt = captures_as_python_does("pass", valid, &args, json!({}), false);
    assert_eq!(statuses(&receipt), vec!["passed"; 48]);
    assert_eq!(receipt["passed"], true);
    assert_eq!(
        receipt["settings"],
        json!({"cases": kinds::structured::STRUCTURED_CASES, "repeat": 2, "stream": true,
               "max_tokens": 64, "reasoning_max_tokens": 128, "temperature": 0.5, "seed": 7})
    );
    let result = rescored(&receipt, false);
    assert_eq!(result.status, Status::Pass, "{:?}", result.error);
    let metrics = result.metrics.unwrap();
    assert_eq!(metrics["summary"], receipt["summary"]);
}

/// A frontend that refuses disabled thinking: the answer-only chat cases
/// send no switch and the reasoning budget, as in Python.
#[test]
fn answer_only_cases_reason_where_thinking_cannot_be_disabled() {
    let args = ["--max-tokens", "64", "--reasoning-max-tokens", "128"];
    let receipt = captured_as_python_does("reasons", valid, &args, json!({}), false, true);
    assert_eq!(receipt["passed"], true);
    let (loaded, frontend) = declared("reasons-requests", &args, json!({}));
    let value = json!({"reasoning": {"thinking_keywords": ["thinking"], "allow_disabled": false}});
    std::fs::write(&frontend, value.to_string()).unwrap();
    let (url, log) = fake_server(valid);
    capture(&loaded, &url, Duration::from_secs(120), false).unwrap();
    let chats: Vec<Value> = log
        .lock()
        .unwrap()
        .iter()
        .filter(|(line, _)| line.starts_with("POST /v1/chat/completions"))
        .map(|(_, body)| body.clone())
        .collect();
    assert!(!chats.is_empty());
    for body in chats {
        let switch = &body["chat_template_kwargs"];
        if switch == &json!({"thinking": true}) {
            continue;
        }
        assert_eq!(switch, &json!({}), "{body}");
        assert_eq!(body["max_tokens"], 128, "{body}");
    }
}

#[test]
fn outputs_are_refused_truncated_and_validated_as_in_python() {
    let repeat = kinds::structured::STRUCTURED_CASES
        .iter()
        .map(|case| outputs(case).len())
        .max()
        .unwrap()
        .to_string();
    let args = ["--stream", "--repeat", repeat.as_str()];
    let receipt = captures_as_python_does("mixed", outputs, &args, json!({}), false);
    let statuses = statuses(&receipt);
    for status in ["passed", "failed", "refused", "truncated"] {
        assert!(statuses.contains(&status), "{status}");
    }
    assert_eq!(receipt["passed"], false);
    // Stored, the receipt is scored to the summary it was written with.
    let mut stored = receipt.clone();
    stored.as_object_mut().unwrap().remove("errors");
    let result = rescored(&stored, false);
    assert_eq!(result.status, Status::Fail);
    assert_eq!(result.metrics.unwrap()["summary"], receipt["summary"]);
}

#[test]
fn defaults_constants_and_a_mechanics_run_send_what_python_sends() {
    captures_as_python_does("defaults", valid, &[], json!({}), false);
    let constants = json!({"structured": {
        "CHOICES": {"value": ["yes", "no"], "source": "test"},
        "PHONE": {"value": "[0-9]{3}-[0-9]{4}", "source": "test"},
        "PERSON_PROMPT": {"value": "Describe Ada as JSON.", "source": "test"},
        "PERSON": {"value": {"type": "object", "required": ["name"],
                             "properties": {"name": {"type": "string", "maxLength": 2}}},
                   "source": "test"},
        "TOOLS": {"value": [], "source": "test"}}});
    let receipt =
        captures_as_python_does("constants", outputs, &["--repeat", "3"], constants, false);
    assert_eq!(
        receipt["rows"][12]["error"],
        "ValueError: 'neutral' is not an offered choice"
    );
    let receipt = captures_as_python_does(
        "mechanics",
        valid,
        &["--cases", "chat_so_choice,tool_named"],
        json!({}),
        true,
    );
    assert_eq!(
        receipt["skipped"],
        json!(["output validity (8-token outputs)"])
    );
    assert_eq!(statuses(&receipt), vec!["passed", "passed"]);
}

/// Python's `json.loads` of each text: the repr and type of the value, or
/// the error.
const LOADS: &str = "
import json
results = []
for text in json.load(sys.stdin):
    try:
        value = json.loads(text)
        results.append([repr(value) if len(text) < 5000 else None, type(value).__name__])
    except (ValueError, RecursionError) as error:
        results.append([f'{type(error).__name__}: {error}', None])
print(json.dumps(results))
";

#[test]
fn texts_decode_as_python_decodes_them() {
    let mut texts: Vec<String> = json_outputs()
        .iter()
        .filter_map(|output| output["content"].as_str().map(String::from))
        .collect();
    texts.extend(
        [
            "[1, [2, {\"a\": [3]}], -0, 0.5e-3, 1E+2, 2e-400]",
            "{\"a\": 1, \"a\": 2, \"b\": 3}",
            "  \r\n\t[]  ",
            "[\"\\\"\\\\\\/\\b\\f\\n\\r\\t\\u0041\"]",
            "\"\\ud83d\\ude0\"",
            "\"\\ud83d\\u12\"",
            "\"\\",
            "\"abc\\",
            "\"\\u",
            "\"\\u004",
            "\"\\u0041",
            "[1,",
            "[1 2]",
            "{\"a\"",
            "{\"a\" 1}",
            "{1: 2}",
            "{\"a\": 1,",
            "trueish",
            "true",
            "nulll",
            "-NaN",
            "-Infinityx",
            "Infinit",
            "[\n1,\n2,\n]",
            "1.5e",
            "-01",
            "0x10",
            ".5",
            "+1",
            "18446744073709551616",
            "-170141183460469231731687303715884105729",
        ]
        .map(String::from),
    );
    texts.push(format!("-{}", "9".repeat(4301)));
    texts.push("9".repeat(4300));
    texts.push(format!("{}{}", "[".repeat(9997), "]".repeat(9997)));
    texts.push(format!("{}{}", "[".repeat(9998), "]".repeat(9998)));
    texts.push(format!("{}1{}", "{\"a\":".repeat(9998), "}".repeat(9998)));
    let ours: Vec<Value> = texts
        .iter()
        .map(|text| match pyjson::loads(text) {
            Ok(value) => json!([(text.len() < 5000).then(|| value.repr()), value.type_name()]),
            Err(error) => json!([error.to_string(), null]),
        })
        .collect();
    // A lone surrogate escape reads as U+FFFD: Rust strings cannot hold it.
    assert_eq!(
        pyjson::loads("\"\\ud83d\\u0041\\udc00\""),
        Ok(Object::Str("\u{fffd}A\u{fffd}".into()))
    );
    assert!(matches!(
        pyjson::loads("{\"a\": 1, \"a\": 2, \"b\": 3}"),
        Ok(Object::Dict(entries)) if entries.len() == 2 && entries[0].1 == Object::Int(2)
    ));
    let Some(output) = python(LOADS, json!(texts).to_string().as_bytes(), &[]) else {
        return;
    };
    let theirs = parse(String::from_utf8(output).unwrap().trim().as_bytes()).unwrap();
    for ((text, ours), theirs) in texts.iter().zip(&ours).zip(theirs.as_array().unwrap()) {
        let shown: String = text.chars().take(80).collect();
        assert_eq!(ours, theirs, "{shown:?}");
    }
}

#[test]
fn unknown_settings_constants_an_unreachable_server_or_a_spent_budget_are_refused() {
    let (url, log) = fake_server(valid);
    let refused = |args: &[&str], constants: Value| {
        let (loaded, _) = declared("refused", args, constants);
        capture(&loaded, &url, Duration::from_secs(60), false).unwrap_err()
    };
    assert_eq!(
        refused(
            &[],
            json!({"structured": {"TYPES": {"value": 1, "source": "test"}}})
        ),
        KindError::Invalid(
            "checks.structured.constants.structured.TYPES is not applied by chopinfer eval".into()
        )
    );
    assert_eq!(
        refused(&["--cases", "chat_so_choice,nope"], json!({})).to_string(),
        format!(
            "unknown cases ['nope']; choose from {}",
            "['chat_json_object', 'chat_json_schema_strict', 'chat_json_schema_loose', \
             'chat_so_json', 'chat_so_choice', 'chat_so_regex', 'chat_so_json_object', \
             'completion_so_json', 'completion_json_schema', 'tool_required', 'tool_named', \
             'reasoning_json_schema']"
        )
    );
    assert_eq!(
        refused(&["--repeat", "0"], json!({})).to_string(),
        "repeat must be positive"
    );
    assert_eq!(
        refused(&["--repeat", "two"], json!({})).to_string(),
        "argument --repeat: invalid int value: 'two'"
    );
    assert!(log.lock().unwrap().is_empty());
    let (loaded, _) = declared("unreachable", &[], json!({}));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let unreachable = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    assert!(matches!(
        capture(&loaded, &unreachable, Duration::from_secs(60), false),
        Err(KindError::Environment(_))
    ));
    assert_eq!(
        capture(&loaded, &url, Duration::ZERO, false),
        Err(KindError::Environment(
            "the check's time budget is exhausted".into()
        ))
    );
    assert!(log.lock().unwrap().is_empty());
}

#[test]
fn outputs_python_cannot_read_stop_the_run() {
    fn nested(_: &str) -> Vec<Value> {
        vec![json!({"content": format!("{}{}", "[".repeat(10000), "]".repeat(10000))})]
    }
    fn numeric(_: &str) -> Vec<Value> {
        vec![json!({"content": 5})]
    }
    for (outputs, error) in [
        (
            nested as Outputs,
            "RecursionError: maximum recursion depth exceeded while decoding a JSON array from a unicode string",
        ),
        (
            numeric as Outputs,
            "structured.py cannot read the output of type 'int'",
        ),
    ] {
        let (loaded, _) = declared("unreadable", &["--cases", "chat_json_object"], json!({}));
        let (url, _) = fake_server(outputs);
        let result = capture(&loaded, &url, Duration::from_secs(60), false);
        if error.starts_with("Recursion") {
            assert_eq!(result, Err(KindError::Invalid(error.into())));
        } else {
            // A number is an output Python validates and fails, then cannot slice.
            assert_eq!(result.unwrap_err().to_string(), error);
        }
    }
}

#[test]
fn a_late_final_response_cannot_pass() {
    fn late(case: &str) -> Vec<Value> {
        if case == "reasoning_json_schema" {
            std::thread::sleep(Duration::from_millis(900));
        }
        valid(case)
    }
    let (url, log) = fake_server(late);
    let (loaded, _) = declared("late-final", &["--repeat", "1"], json!({}));
    let started = Instant::now();
    let result = capture(&loaded, &url, Duration::from_millis(400), false);
    let elapsed = started.elapsed();
    assert_eq!(log.lock().unwrap().len(), 12);
    assert!(
        matches!(result, Err(KindError::Environment(_))),
        "{result:?}"
    );
    assert!(elapsed < Duration::from_millis(800), "{elapsed:?}");
}
