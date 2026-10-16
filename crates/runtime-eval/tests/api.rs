//! The api kind against `tools/qual/kinds.py` and `tools/qual/api.py`.
//! Receipts changed one field at a time are rescored by both
//! implementations, which must agree on the status, the error text and the
//! metrics. A mechanics capture and Python's `api.main --mechanics-only`
//! run against the same fake server and must send the same requests and
//! write the same receipt. Every other capture must send and write what the
//! retired Python capture did (tests/data/retired-captures.json). The
//! comparisons with Python are skipped when python3 is missing.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use runtime_eval::declaration::{self, Check, Loaded, Predicate};
use runtime_eval::http::{BoundServer, Client, ServerIdentity};
use runtime_eval::json::{canonical, canonical_sha256, compact, parse};
use runtime_eval::kind::{self, Capture, Kind, KindError};
use runtime_eval::kinds;
use serde_json::{Map, Value, json};

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

fn api() -> Box<dyn Kind> {
    kinds::all()
        .into_iter()
        .find(|kind| kind.info().name == "api")
        .unwrap()
}

/// The check the vectors declare: `passed` true.
fn check() -> Check {
    Check {
        id: "api".into(),
        module: "api".into(),
        timeout_seconds: 60.0,
        rules: vec![Predicate {
            path: vec![json!("passed")],
            op: "eq".into(),
            value: json!(true),
            source: "docs/check-kinds.md".into(),
        }],
        informational_reason: None,
        declared: json!({}),
    }
}

fn pass_receipt() -> Value {
    let file = parse(&std::fs::read(root().join("tests/qual-vectors/api.json")).unwrap()).unwrap();
    file["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|vector| vector["name"] == "pass")
        .unwrap()["receipt"]
        .clone()
}

/// The pass receipt with the value at `pointer` replaced, or removed when
/// `value` is `None`.
fn changed(pointer: &str, value: Option<Value>) -> Value {
    let mut receipt = pass_receipt();
    match value {
        Some(value) => {
            *receipt
                .pointer_mut(pointer)
                .unwrap_or_else(|| panic!("{pointer}")) = value
        }
        None => {
            let (parent, key) = pointer.rsplit_once('/').unwrap();
            let parent = receipt.pointer_mut(parent).unwrap();
            match parent {
                Value::Object(map) => {
                    map.shift_remove(key).unwrap();
                }
                Value::Array(items) => {
                    items.remove(key.parse().unwrap());
                }
                _ => panic!("{pointer}"),
            }
        }
    }
    receipt
}

/// Receipts and whether each is rescored as a mechanics run.
fn cases() -> Vec<(Value, bool)> {
    let set = |pointer: &str, value: Value| changed(pointer, Some(value));
    let mut swapped = pass_receipt();
    swapped["rows"].as_array_mut().unwrap().swap(0, 1);
    let mut extra = pass_receipt();
    let last = extra["rows"][39].clone();
    extra["rows"].as_array_mut().unwrap().push(last);
    let mut unpassed = pass_receipt();
    unpassed["rows"][3]["passed"] = json!(false);
    unpassed["passed"] = json!(true);
    let mechanics = set("/mechanics_only", json!(true));
    let mut mechanics_rows = mechanics.clone();
    mechanics_rows["rows"] = json!([{"name": "x", "passed": 1}]);
    vec![
        (pass_receipt(), false),
        (set("/rows/3/passed", json!(false)), false),
        (set("/rows/39/passed", json!(false)), false),
        (unpassed, false),
        (set("/passed", json!(false)), false),
        (changed("/passed", None), false),
        (set("/rows/3/passed", json!(1)), false),
        (set("/rows/3/passed", json!(null)), false),
        (changed("/rows/3/passed", None), false),
        (changed("/rows/3/latency_seconds", None), false),
        (set("/rows/3/name", json!("choice_single")), false),
        (changed("/rows/3/name", None), false),
        (set("/rows/3", json!("row")), false),
        (set("/rows/3", json!([])), false),
        (changed("/rows/3", None), false),
        (swapped, false),
        (extra, false),
        (set("/rows", json!({})), false),
        (set("/rows", json!("rows")), false),
        (set("/rows", json!([])), false),
        (changed("/rows", None), false),
        (
            set(
                "/errors",
                json!([{"name": "tokenize", "error": "HTTP 404"}]),
            ),
            false,
        ),
        (set("/skipped", json!(["coverage"])), false),
        (mechanics.clone(), false),
        (mechanics, true),
        (mechanics_rows, true),
        (set("/stream_text", json!(null)), false),
        (json!([]), false),
    ]
}

/// Python's `kinds.rescore` of each case: status, error and metrics.
const RESCORE: &str = "
import json
import kinds
check = {'rules': [{'path': ['passed'], 'op': 'eq', 'value': True, 'source': 'docs/check-kinds.md'}]}
results = []
for receipt, mechanics in json.load(sys.stdin):
    result = kinds.rescore(kinds.KINDS['api'], receipt, None, check, mechanics)
    results.append({'status': result['status'], 'error': result.get('error'), 'metrics': result['metrics']})
json.dump(results, sys.stdout)
";

#[test]
fn changed_receipts_rescore_as_python_does() {
    let cases = cases();
    let Some(output) = python(RESCORE, &serde_json::to_vec(&cases).unwrap(), &[]) else {
        return;
    };
    let expected = parse(&output).unwrap();
    assert_eq!(expected.as_array().unwrap().len(), cases.len());
    let kind = api();
    let mut statuses = std::collections::BTreeSet::new();
    for ((receipt, mechanics), expected) in cases.iter().zip(expected.as_array().unwrap()) {
        let result = kind::rescore(kind.as_ref(), receipt, None, &check(), *mechanics);
        let actual = json!({"status": result.status.as_str(), "error": result.error,
                            "metrics": result.metrics});
        // The canonical text tells ints from floats and holds every float's bits.
        assert_eq!(
            String::from_utf8(canonical(&actual)).unwrap(),
            String::from_utf8(canonical(expected)).unwrap(),
            "{receipt}"
        );
        statuses.insert(result.status.as_str());
    }
    assert_eq!(
        statuses.into_iter().collect::<Vec<_>>(),
        ["fail", "insufficient", "pass"]
    );
}

const CHECKPOINT: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const NONCE: &str = "0123456789abcdef0123456789abcdef";
const MODEL: &str = "served";
/// The one token "Hello" encodes to.
const HELLO: u64 = 9906;

/// How the fake server misbehaves.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Flaw {
    None,
    /// No choice for the single chat request, streamed tool arguments that
    /// differ from the whole response's, a tool answer cut off by the length
    /// bound and a negative metric.
    Answers,
    /// `/tokenize` and `/detokenize` are absent.
    NoTokenizer,
    /// The frontend always reasons: it declares `allow_disabled: false` and
    /// refuses a chat request that disables thinking with HTTP 400.
    AlwaysReasons,
    /// The frontend declares `allow_disabled: false`, but the server answers
    /// a request that disables thinking.
    AcceptsDisabled,
}

/// The request lines and bodies a fake server read, in order.
type Log = Arc<Mutex<Vec<(String, Value)>>>;

/// One generated choice.
struct Generated {
    index: u64,
    prompt: Vec<u64>,
    ids: Vec<u64>,
    pieces: Vec<String>,
    reasoning: Option<String>,
    tool: bool,
    finish: &'static str,
}

/// The choices a completion or chat request generates, and its prompt
/// tokens: each prompt one token ("Hello" or its ID), a chat prompt five;
/// `max_tokens` tokens finishing at the length bound unless the request
/// stops on a token, a stop string, a tool call or a tool result.
fn generate(chat: bool, body: &Value, flaw: Flaw) -> (Vec<Generated>, u64) {
    let prompts: Vec<Vec<u64>> = if chat {
        vec![vec![1, 2, 3, 4, 5]]
    } else {
        match &body["prompt"] {
            Value::Array(items) if items[0].is_number() => {
                vec![items.iter().map(|id| id.as_u64().unwrap()).collect()]
            }
            Value::Array(items) => vec![vec![HELLO]; items.len()],
            _ => vec![vec![HELLO]],
        }
    };
    let n = body["n"].as_u64().unwrap_or(1);
    if flaw == Flaw::Answers && chat && n == 1 && body["seed"] == 41 {
        return (Vec::new(), 5);
    }
    let limit = body["max_completion_tokens"]
        .as_u64()
        .filter(|_| chat)
        .or(body["max_tokens"].as_u64())
        .unwrap();
    let messages = body["messages"].as_array();
    let tool_result = messages.is_some_and(|messages| messages.last().unwrap()["role"] == "tool");
    let thinking =
        body["chat_template_kwargs"]["thinking"] == true && body["include_reasoning"] != false;
    let mut choices = Vec::new();
    for (number, prompt) in prompts.iter().enumerate() {
        for sample in 0..n {
            let (pieces, reasoning, tool, finish): (Vec<&str>, Option<&str>, bool, &str) =
                if tool_result && flaw == Flaw::Answers {
                    (Vec::new(), Some("The tool says"), false, "length")
                } else if tool_result {
                    (
                        vec!["It is 18", " degrees", " and cloudy."],
                        None,
                        false,
                        "stop",
                    )
                } else if body.get("tools").is_some() {
                    (Vec::new(), None, true, "tool_calls")
                } else if body["stop_token_ids"]
                    .as_array()
                    .is_some_and(|ids| !ids.is_empty())
                {
                    (vec![""], None, false, "stop")
                } else if body.get("stop").is_some() {
                    (vec!["A", " short", " sentence."], None, false, "stop")
                } else {
                    let pieces = [" w"; 1024];
                    (pieces[..limit as usize].to_vec(), None, false, "length")
                };
            let reasoning = reasoning.or(thinking.then_some("17 times 23 is 391."));
            choices.push(Generated {
                index: number as u64 * n + sample,
                prompt: prompt.clone(),
                ids: (0..pieces.len() as u64).map(|at| 100 + at).collect(),
                pieces: pieces.into_iter().map(String::from).collect(),
                reasoning: reasoning.map(String::from),
                tool,
                finish,
            });
        }
    }
    let prompt_tokens = prompts.iter().map(|prompt| prompt.len() as u64).sum();
    (choices, prompt_tokens)
}

fn usage(prompt: u64, completion: u64) -> Value {
    json!({"prompt_tokens": prompt, "completion_tokens": completion,
           "total_tokens": prompt + completion})
}

/// A tool call, its arguments whole or in `part`.
fn tool_call(part: Option<&str>) -> Value {
    match part {
        None => json!({"id": "call-1", "type": "function",
                       "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}),
        Some(part) => json!({"index": 0, "function": {"arguments": part}}),
    }
}

/// A non-streamed completion or chat response.
fn response(chat: bool, body: &Value, flaw: Flaw) -> Value {
    let (generated, prompt_tokens) = generate(chat, body, flaw);
    let ids = body["return_token_ids"] == true;
    let choices: Vec<Value> = generated
        .iter()
        .map(|choice| {
            let mut value = json!({"index": choice.index, "finish_reason": choice.finish});
            if chat {
                let mut message = json!({"role": "assistant", "content": choice.pieces.concat()});
                if choice.tool {
                    message["content"] = Value::Null;
                    message["tool_calls"] = json!([tool_call(None)]);
                }
                if choice.pieces.is_empty() && !choice.tool {
                    message["content"] = Value::Null;
                }
                if let Some(reasoning) = &choice.reasoning {
                    message["reasoning"] = json!(reasoning);
                }
                value["message"] = message;
            } else {
                value["text"] = json!(choice.pieces.concat());
                value["logprobs"] = if body["logprobs"].is_number() {
                    json!({"text_offset": [0], "token_logprobs": [-0.5], "tokens": [" w"],
                           "top_logprobs": [{" w": -0.5}]})
                } else {
                    Value::Null
                };
                if ids {
                    value["prompt_token_ids"] = json!(choice.prompt);
                }
            }
            if ids {
                value["token_ids"] = json!(choice.ids);
            }
            value
        })
        .collect();
    let completion = generated.iter().map(|choice| choice.ids.len() as u64).sum();
    let mut value = json!({"id": "response-1",
                           "object": if chat { "chat.completion" } else { "text_completion" },
                           "model": MODEL, "choices": choices,
                           "usage": usage(prompt_tokens, completion)});
    if chat && ids {
        value["prompt_token_ids"] = json!(generated[0].prompt);
    }
    value
}

/// A streamed completion or chat response's events: one per token of each
/// choice, the last with its finish, usage on every event with continuous
/// usage stats, and a usage-only event with `include_usage`.
fn stream_events(chat: bool, body: &Value, flaw: Flaw) -> Vec<Value> {
    let (generated, prompt_tokens) = generate(chat, body, flaw);
    let ids = body["return_token_ids"] == true;
    let object = if chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    let continuous = body["stream_options"]["continuous_usage_stats"] == true;
    let mut events = Vec::new();
    let mut produced = 0;
    for choice in &generated {
        let mut deltas: Vec<(Value, Option<u64>)> = if choice.tool {
            let first = tool_call(None);
            let mut first = json!({"index": 0, "id": first["id"], "type": "function",
                                   "function": {"name": "get_weather", "arguments": ""}});
            first["function"]["arguments"] = json!("");
            // The arguments arrive whole when the call closes, as the server
            // streams them.
            let arguments = if flaw == Flaw::Answers {
                "{\"city\": \"Rome\"}"
            } else {
                "{\"city\": \"Paris\"}"
            };
            [first, tool_call(Some(arguments))]
                .into_iter()
                .map(|call| (json!({"tool_calls": [call]}), None))
                .collect()
        } else {
            choice
                .pieces
                .iter()
                .zip(&choice.ids)
                .map(|(piece, id)| (json!(piece), Some(*id)))
                .collect()
        };
        if choice.tool {
            deltas.push((json!({}), None));
        }
        let last = deltas.len() - 1;
        for (at, (delta, id)) in deltas.into_iter().enumerate() {
            let mut value = json!({"index": choice.index,
                                   "finish_reason": if at == last { json!(choice.finish) } else { Value::Null }});
            if chat {
                value["delta"] = if delta.is_string() {
                    json!({"content": delta})
                } else {
                    delta
                };
            } else {
                value["text"] = delta;
                if ids && at == 0 {
                    value["prompt_token_ids"] = json!(choice.prompt);
                }
            }
            if let Some(id) = id.filter(|_| ids) {
                value["token_ids"] = json!([id]);
            }
            produced += u64::from(id.is_some());
            let mut event = json!({"id": "stream-1", "object": object, "model": MODEL,
                                   "choices": [value]});
            if chat && ids && at == 0 {
                event["prompt_token_ids"] = json!(choice.prompt);
            }
            if continuous {
                event["usage"] = usage(prompt_tokens, produced);
            }
            events.push(event);
        }
    }
    if body["stream_options"]["include_usage"] == true {
        let mut total = usage(prompt_tokens, produced);
        total["prompt_tokens_details"] = json!({"cached_tokens": 0});
        events.push(
            json!({"id": "stream-1", "object": object, "model": MODEL, "choices": [],
                           "usage": total}),
        );
    }
    events
}

/// The vLLM-compatible metrics; with the `Answers` flaw the KV cache usage
/// is negative.
fn metrics(flaw: Flaw) -> String {
    let labels = "engine=\"0\",model_name=\"served\"";
    let mut lines = vec![
        "# HELP vllm:request_success_total Finished requests.".to_owned(),
        format!("vllm:request_success_total{{{labels},finished_reason=\"stop\"}} 7.0"),
        format!("vllm:prompt_tokens_total{{{labels}}} 70.0"),
        format!("vllm:generation_tokens_total{{{labels}}} 1_400.0"),
        format!("vllm:num_requests_running{{{labels}}} 0.0"),
        format!("vllm:num_requests_waiting{{{labels}}} 0.0"),
        format!(
            "vllm:kv_cache_usage_perc{{{labels}}} {}",
            if flaw == Flaw::Answers {
                "-1.0"
            } else {
                "0.25"
            }
        ),
    ];
    for name in [
        "vllm:time_to_first_token_seconds",
        "vllm:inter_token_latency_seconds",
        "vllm:e2e_request_latency_seconds",
    ] {
        lines.push(format!("{name}_bucket{{{labels},le=\"0.5\"}} 3.0"));
        lines.push(format!("{name}_bucket{{{labels},le=\"+Inf\"}} 7.0"));
        lines.push(format!("{name}_sum{{{labels}}} 1.5e0"));
        lines.push(format!("{name}_count{{{labels}}} 7.0"));
    }
    lines.join("\r\n") + "\n"
}

/// The fake server's answer: status, content type and body.
fn answer(path: &str, body: &Value, flaw: Flaw) -> (&'static str, &'static str, String) {
    let json = |value: Value| ("200 OK", "application/json", value.to_string());
    match path {
        "/v1/models" => json(json!({"object": "list", "data": [{"id": MODEL}]})),
        "/health" => json(json!({"status": "ok"})),
        "/metrics" => ("200 OK", "text/plain; version=0.0.4", metrics(flaw)),
        "/tokenize" | "/detokenize" if flaw == Flaw::NoTokenizer => (
            "404 Not Found",
            "application/json",
            json!({"detail": "Not Found"}).to_string(),
        ),
        "/tokenize" => json(json!({"tokens": [HELLO], "count": 1, "max_model_len": 8192})),
        "/detokenize" => json(json!({"prompt": "Hello"})),
        "/v1/chat/completions"
            if flaw == Flaw::AlwaysReasons && body["chat_template_kwargs"]["thinking"] == false =>
        {
            (
                "400 Bad Request",
                "application/json",
                json!({"error": {"message": "this frontend does not allow reasoning to be disabled"}})
                    .to_string(),
            )
        }
        _ => {
            let chat = path == "/v1/chat/completions";
            if body["stream"] == true {
                let text: String = stream_events(chat, body, flaw)
                    .iter()
                    .map(|event| format!("data: {event}\n\n"))
                    .collect();
                ("200 OK", "text/event-stream", text + "data: [DONE]\n\n")
            } else {
                json(response(chat, body, flaw))
            }
        }
    }
}

/// A fake server answering every request on its own connection; its URL
/// and the log of what it read.
fn fake_server(flaw: Flaw) -> (String, Log) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let log = Log::default();
    let written = log.clone();
    std::thread::spawn(move || {
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
            let (status, kind, text) = answer(&path, &body, flaw);
            written
                .lock()
                .unwrap()
                .push((line.trim_end().to_owned(), body));
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

/// The declared inputs: the completions sample (in the layout holding
/// response bodies by name, or the one holding {status, seconds, body})
/// and the chat renders with their tools-00 conversation.
fn inputs(root: &Path, bodies_by_name: bool) -> Map<String, Value> {
    let logprobs = json!({"choices": [{"index": 0, "text": " the",
        "logprobs": {"text_offset": [0], "token_logprobs": [-0.5], "tokens": [" w"],
                     "top_logprobs": [{" w": -0.5}]}}]});
    let samples = if bodies_by_name {
        json!({"chat_thinking": {}, "completion_logprobs": logprobs})
    } else {
        json!({"completion-logprobs": {"status": 200, "seconds": 0.5, "body": logprobs}})
    };
    let renders = json!([
        {"id": "chat-00", "messages": [{"role": "user", "content": "Hi"}]},
        {"id": "tools-00",
         "messages": [{"role": "system", "content": "Use the tools."},
                      {"role": "user", "content": "What is the weather in Paris?"}],
         "tools": [{"type": "function",
                    "function": {"name": "get_weather", "description": "Current weather.",
                                 "parameters": {"type": "object",
                                                "properties": {"city": {"type": "string"}},
                                                "required": ["city"]}}}]}
    ]);
    let frontend = json!({"reasoning": {"thinking_keywords": ["thinking", "think"]}});
    let mut paths = Map::new();
    for (name, file, value) in [
        ("frontend", "frontend.json", frontend),
        ("api_samples", "api-samples.json", samples),
        ("chat_renders", "chat-renders.json", renders),
    ] {
        let path = root.join("fixtures/example").join(file);
        std::fs::write(&path, value.to_string()).unwrap();
        paths.insert(name.into(), json!(path.to_str().unwrap()));
    }
    paths
}

/// A root whose model declares the api check (with `constants`), and the
/// paths of its declared inputs.
fn declared(name: &str, bodies_by_name: bool, constants: Value) -> (Loaded, Map<String, Value>) {
    let root = std::env::temp_dir().join(format!("runtime-eval-api-{name}-{}", std::process::id()));
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
    let paths = inputs(&root, bodies_by_name);
    let lock = json!({"model.safetensors": {"sha256": "ab", "size": 1}});
    std::fs::write(model.join("checkpoint-files.json"), lock.to_string()).unwrap();
    let pin = json!({"repository": "org/example", "revision": "abc",
                     "served_model_name": MODEL, "source_lock_sha256": canonical_sha256(&lock),
                     "checkpoint_identity": CHECKPOINT,
                     "lock": "models/example/checkpoint-files.json"});
    let declaration = json!({
        "schema": "chopinfer-qual-model/v1", "model": "example",
        "cohorts": {"quick": {"checks": ["api"]}},
        "change_classes": {"frontend": [], "quality": [], "service": [], "speculation": [], "prefix": []},
        "checks": {"api": {"module": "api", "args": ["--url", "{url}", "--model", "{model}"],
                           "timeout_seconds": 60, "receipt": "*-api.json", "constants": constants,
                           "rules": [{"path": ["passed"], "op": "eq", "value": true,
                                      "source": "docs/check-kinds.md"}]}},
        "inputs": {"frontend": "fixtures/example/frontend.json",
                   "api_samples": "fixtures/example/api-samples.json",
                   "chat_renders": "fixtures/example/chat-renders.json"},
        "qualification": {"teacher_reference_kind": "exact", "teacher_receipt": "teacher",
                          "teacher_capture_schema": "example.capture.v1", "verify_buckets": [2],
                          "chars_per_token": 4.0, "checkpoint": pin},
        "receipts": {"env": "RUNTIME_EVAL_TEST_UNSET", "path": "qual/example"}
    });
    std::fs::write(model.join("qual.json"), declaration.to_string()).unwrap();
    (declaration::load(&model).unwrap(), paths)
}

/// The api capture of the declared check against `url`, within `budget`.
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
    api().capture(&mut capture)
}

/// Python's `api.main --mechanics-only` against `url` with the declared
/// inputs, the frontend's thinking keyword and the module constants: its
/// receipt.
const MAIN: &str = "
import json
from pathlib import Path
import api
import common
spec = json.load(sys.stdin)
api.input_path = lambda name: Path(spec['inputs'][name])
api.input_bytes = lambda name: Path(spec['inputs'][name]).read_bytes()
api.thinking_kwargs = lambda enabled: {'thinking': enabled}
api.thinking_can_be_disabled = lambda: json.loads(Path(spec['inputs']['frontend']).read_text())[
    'reasoning'].get('allow_disabled', True) is not False
api.arguments = lambda description, model_facts=False: common.arguments(description)
for name, entry in spec['constants'].get('api', {}).items():
    setattr(api, name, entry['value'])
sys.argv = ['api.py', '--url', spec['url'], '--timeout', '30', '--mechanics-only']
try:
    api.main()
except SystemExit:
    pass
";

/// The receipt Python's mechanics capture writes against a fresh fake
/// server with `flaw`, and what that server read; `None` without python3.
fn python_main(
    name: &str,
    flaw: Flaw,
    inputs: &Map<String, Value>,
    constants: &Value,
) -> Option<(Value, Log)> {
    let (url, log) = fake_server(flaw);
    let receipts = std::env::temp_dir().join(format!(
        "runtime-eval-api-receipts-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&receipts);
    let spec = json!({"url": url, "inputs": inputs, "constants": constants});
    let output = python(
        MAIN,
        spec.to_string().as_bytes(),
        &[("CHOPINFER_QUAL_RECEIPTS", &receipts)],
    )?;
    let printed = parse(String::from_utf8(output).unwrap().trim().as_bytes()).unwrap();
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
/// (Python's names its health), the host load, the latencies and the
/// arrival times of a rejected stream's events (Python's are its monotonic
/// clock's).
fn comparable(receipt: &Value) -> Value {
    let mut receipt = receipt.clone();
    let map = receipt.as_object_mut().unwrap();
    map.remove("at");
    map.remove("server");
    map.remove("host_load");
    for row in map["rows"].as_array_mut().unwrap() {
        row.as_object_mut().unwrap().remove("latency_seconds");
    }
    for error in map["errors"].as_array_mut().unwrap() {
        if let Some(events) = error.get_mut("response").and_then(Value::as_array_mut) {
            for event in events {
                event[0] = Value::Null;
            }
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

/// Captures with `flaw` against a fake server and compares the requests
/// and the receipt with Python's mechanics capture against another, or
/// outside a mechanics run with what the retired Python capture wrote;
/// returns the receipt.
fn captures_as_python_does(
    name: &str,
    flaw: Flaw,
    bodies_by_name: bool,
    constants: Value,
    mechanics: bool,
) -> Value {
    let (loaded, inputs) = declared(name, bodies_by_name, constants.clone());
    if matches!(flaw, Flaw::AlwaysReasons | Flaw::AcceptsDisabled) {
        let frontend = json!({"reasoning": {"thinking_keywords": ["thinking", "think"],
                                            "allow_disabled": false}});
        std::fs::write(inputs["frontend"].as_str().unwrap(), frontend.to_string()).unwrap();
    }
    let (url, log) = fake_server(flaw);
    let receipt = capture(&loaded, &url, Duration::from_secs(60), mechanics).unwrap();
    assert_eq!(receipt["check"], "api");
    assert_eq!(receipt["mechanics_only"], mechanics);
    // The envelope records the host load that labels the latencies.
    assert_eq!(
        receipt["host_load"]["rule"],
        runtime_eval::capture::host_load::RULE
    );
    assert_eq!(receipt["host_load"]["samples"].as_array().unwrap().len(), 2);
    assert_eq!(
        receipt["server"]["checkpoint"]["checkpoint_identity"],
        CHECKPOINT
    );
    if !mechanics {
        as_retired_python_wrote(&format!("api/{name}"), &invariant(&log, &receipt));
    } else if let Some((python, python_log)) = python_main(name, flaw, &inputs, &constants) {
        assert_eq!(requests(&log), requests(&python_log));
        assert_eq!(
            String::from_utf8(canonical(&comparable(&receipt))).unwrap(),
            String::from_utf8(canonical(&comparable(&python))).unwrap()
        );
    }
    receipt
}

fn rows(receipt: &Value) -> Vec<&Value> {
    receipt["rows"].as_array().unwrap().iter().collect()
}

/// A frontend that always reasons must refuse the disabled-thinking case,
/// which then passes on that refusal alone, as in Python.
#[test]
fn a_frontend_that_always_reasons_refuses_disabled_thinking_as_in_python() {
    let receipt = captures_as_python_does("reasons", Flaw::AlwaysReasons, true, json!({}), false);
    let names: Vec<&str> = rows(&receipt)
        .iter()
        .map(|row| row["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, kinds::api::API_CASES);
    let row = rows(&receipt)[35];
    assert_eq!(row["name"], "reasoning_False");
    assert_eq!(row["passed"], true);
    assert!(row["refused"].as_str().unwrap().starts_with("HTTP 400: "));
    assert_eq!(receipt["errors"], json!([]));
    let result = kind::rescore(api().as_ref(), &receipt, None, &check(), false);
    assert_eq!(result.status.as_str(), "pass", "{:?}", result.error);
    // A server that answers the disabled switch fails that case.
    let receipt = captures_as_python_does("accepts", Flaw::AcceptsDisabled, true, json!({}), false);
    assert_eq!(rows(&receipt)[35]["passed"], false);
    assert_eq!(receipt["errors"][0]["name"], "reasoning_False");
    assert_eq!(
        receipt["errors"][0]["error"],
        "accepted a request the frontend refuses"
    );
}

#[test]
fn every_case_passes_against_a_conforming_server_as_in_python() {
    let receipt = captures_as_python_does("pass", Flaw::None, true, json!({}), false);
    let names: Vec<&str> = rows(&receipt)
        .iter()
        .map(|row| row["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, kinds::api::API_CASES);
    assert_eq!(receipt["errors"], json!([]));
    assert_eq!(receipt["passed"], true);
    assert_eq!(
        receipt["stream_text"],
        " w w w w w w w w w w w w w w w w w w w w w w w w"
    );
    for row in rows(&receipt) {
        assert!(row["latency_seconds"].is_f64() || row["name"] == "metrics_compatibility");
    }
    let tool_result = rows(&receipt)[38];
    assert_eq!(
        (
            &tool_result["finish_reason"],
            &tool_result["length_limited"]
        ),
        (&json!("stop"), &json!(false))
    );
    let result = kind::rescore(api().as_ref(), &receipt, None, &check(), false);
    assert_eq!(result.status.as_str(), "pass", "{:?}", result.error);
}

#[test]
fn rejected_answers_keep_their_response_as_in_python() {
    // The other sample layout, and the tool answer's budget set by the check.
    let constants = json!({"api": {"TOOL_RESULT_MAX_TOKENS": {"value": 512, "source": "test"}}});
    let receipt = captures_as_python_does("answers", Flaw::Answers, false, constants, false);
    assert_eq!(receipt["passed"], false);
    let errors: Vec<&str> = receipt["errors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|error| error["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        errors,
        [
            "choice_single_True",
            "multiple_choices_True_False",
            "multiple_choices_True_True",
            "stream_tool_arguments",
            "tool_result",
            "metrics_compatibility"
        ]
    );
    let error = |at: usize| &receipt["errors"][at];
    assert_eq!(error(0)["error"], "missing choice");
    assert_eq!(error(0)["response"]["choices"], json!([]));
    // The single chat case failed: the multiple-choice cases expect no prompt usage.
    assert_eq!(error(1)["error"], "prompt usage duplicated");
    assert_eq!(
        error(3)["error"],
        "streamed tool calls differ from the unstreamed response"
    );
    assert_eq!(
        error(4)["error"],
        "final answer did not use tool result (length limit reached)"
    );
    assert_eq!(
        error(4)["response"]["choices"][0]["message"]["reasoning"],
        "The tool says"
    );
    assert_eq!(
        error(5),
        &json!({"name": "metrics_compatibility",
                "error": "negative metric: vllm:kv_cache_usage_perc{engine=\"0\",model_name=\"served\"} -1.0"})
    );
    let tool_result = rows(&receipt)[38];
    assert_eq!(
        tool_result,
        &json!({"name": "tool_result", "passed": false, "finish_reason": "length",
                "length_limited": true})
    );
    let result = kind::rescore(api().as_ref(), &receipt, None, &check(), false);
    assert_eq!(
        (result.status.as_str(), result.error.as_deref()),
        ("fail", Some("check reported errors"))
    );
}

#[test]
fn a_server_without_a_tokenizer_fails_those_cases_as_in_python() {
    let receipt = captures_as_python_does("tokenizer", Flaw::NoTokenizer, true, json!({}), false);
    let names: Vec<&str> = rows(&receipt)
        .iter()
        .map(|row| row["name"].as_str().unwrap())
        .collect();
    // The batch, detokenize and token stop cases need the token IDs.
    assert_eq!(names.len(), 32);
    assert!(!names.contains(&"detokenize"));
    assert_eq!(
        receipt["errors"],
        json!([{"name": "tokenize", "error": "HTTP 404: {\"detail\":\"Not Found\"}"}])
    );
    let result = kind::rescore(api().as_ref(), &receipt, None, &check(), true);
    assert_eq!(
        (result.status.as_str(), result.error.as_deref()),
        ("fail", Some("check reported errors"))
    );
}

#[test]
fn a_mechanics_run_sends_the_fixed_tool_request_as_in_python() {
    let receipt = captures_as_python_does("mechanics", Flaw::None, true, json!({}), true);
    assert_eq!(receipt["passed"], true);
    assert_eq!(
        receipt["skipped"],
        json!(["model reasoning and tool round trip (fixture context is 128 tokens)"])
    );
    assert_eq!(rows(&receipt).len(), 39);
}

#[test]
fn an_unknown_constant_an_unreachable_server_or_a_spent_budget_is_refused() {
    let constants = json!({"api": {"TOOL_RESULT_MAX_TOKENS": {"value": 512, "source": "test"}}});
    let (mut loaded, _) = declared("refused", true, constants);
    loaded.declaration.checks[0].declared["constants"]["api"]["OTHER"] = json!({"value": 1});
    let (url, log) = fake_server(Flaw::None);
    assert_eq!(
        capture(&loaded, &url, Duration::from_secs(60), false),
        Err(KindError::Invalid(
            "checks.api.constants.api.OTHER is not applied by chopinfer eval".into()
        ))
    );
    let (loaded, _) = declared("unreachable", true, json!({}));
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
