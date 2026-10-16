//! One unchanged host binary runs both external fixtures, each loading its
//! synthetic checkpoint through the fixture's weight plan.
//! Each fixture commits, cancels, and drops an open execution through that binary.
//! Manifest refusals are checked before a native module is opened.

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use runtime_weights::sha256_bytes;
use serde_json::Value;

/// Every bundle this one host binary is driven over, in the order it runs them.
const EXTERNAL_FIXTURES: [FixtureBundle; 2] = [RECURRENT_FIXTURE, TP4_FIXTURE];

const RECURRENT_FIXTURE: FixtureBundle = FixtureBundle {
    package: "bundle-recurrent-fixture",
    builder: "build-recurrent-fixture",
    bundle_id: "fixture.recurrent.tp1.v1",
    fixture: "tp1-recurrent",
    // This fixture's own declared page is the batch it serves, so nothing here
    // names a count for it.
    fixture_tokens: None,
    tail_tokens: None,
    rank_count: 1,
    topology_id: "tp1",
    architecture: "sm_120f",
};
/// TP4 paged-KV fixture with a shorter tail through the same operation.
const TP4_FIXTURE: FixtureBundle = FixtureBundle {
    package: "bundle-tp4-fixture",
    builder: "build-tp4-fixture",
    bundle_id: "fixture.tp4.tp4.v1",
    fixture: "tp4-synthetic",
    // Its declared 32-token page is exactly the the full prefill chunk it serves, so the run
    // takes the page and names no count.
    fixture_tokens: None,
    // The other shape its one operation is lent: a tail shorter than the page,
    // which the same bundle has to serve through the same host.
    tail_tokens: Some(8),
    rank_count: 4,
    topology_id: "tp4",
    architecture: "sm_120f",
};

/// The capability no host declares, used to prove the check is enforced.
const ABSENT_CAPABILITY: &str = "chop.capability.absent.v1";

/// Every scenario this one host binary drives over every emitted bundle.
///
/// A positive run is the first of them and not the whole gate: an external
/// boundary that could not cancel or release would still pass a run that only
/// commits. Each scenario names the status its receipt must report and how many
/// times the host executed the bundle's batch to get there.
const HOST_SCENARIOS: [HostScenario; 3] = [
    HostScenario {
        name: "commit",
        status: "committed",
        executions: 1,
    },
    // The cancelled batch is replayed, so this one executes twice and the host
    // requires the second row to be what the cancelled one produced.
    HostScenario {
        name: "cancel",
        status: "cancelled",
        executions: 2,
    },
    HostScenario {
        name: "drop",
        status: "dropped",
        executions: 1,
    },
];

#[test]
fn two_external_bundles_run_through_one_unchanged_host_binary() {
    let temporary = TempDirectory::new("external-bundle-host");
    let target_directory = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("target"));
    // Each fixture's package and synthetic checkpoint live here.
    let root = temporary.path();

    // Everything below happens after this binary exists and is hashed.
    let host = build_host(&target_directory, temporary.path());
    let host_hash = hash(&host);
    assert_macos_host_path_is_cuda_free(&target_directory);

    let mut emitted = Vec::new();
    let mut identity = None;
    for fixture in EXTERNAL_FIXTURES {
        let bundle = bundle_directory(root, fixture);
        build_fixture(fixture, root, &target_directory);
        assert_unchanged(&host, &host_hash, "building an external fixture bundle");

        // The positive run, the cancellation, and the cleanup are all this one
        // binary's, over the bundle it just emitted.
        for scenario in HOST_SCENARIOS {
            let receipt = run_emitted(&host, root, fixture, scenario);
            assert_scenario_receipt(&receipt, fixture, scenario);
            assert_one_host_ran_everything(&mut identity, &receipt, fixture, scenario);
            assert_unchanged(
                &host,
                &host_hash,
                &format!(
                    "running the {} scenario of {}",
                    scenario.name, fixture.bundle_id
                ),
            );
        }

        assert_named_token_count_is_the_batch(&host, root, fixture);
        assert_unchanged(
            &host,
            &host_hash,
            "refusing an external fixture's declared page as a batch",
        );

        assert_declared_tail_also_runs(&host, root, fixture);
        assert_unchanged(
            &host,
            &host_hash,
            "running an external fixture's shorter tail",
        );

        emitted.push((fixture, bundle));
    }

    assert_legacy_prefill_chunks(&host, root, temporary.path());
    assert_served_media_and_speculation(&host, root, temporary.path());
    assert_unchanged(&host, &host_hash, "serving media and speculative requests");
    assert_served_prefix_cache(&host, root, temporary.path());
    assert_unchanged(&host, &host_hash, "serving prefix cache hits");
    assert_dashboard_lists_a_served_fixture(&target_directory, &host, root, temporary.path());
    assert_unchanged(
        &host,
        &host_hash,
        "listing a served fixture with chopinfer-top",
    );
    reject_unloadable_bundles(&host, &host_hash, temporary.path(), &emitted);
    refuse_serving_without_a_frontend(&host, root, temporary.path(), &emitted);
    assert_unchanged(&host, &host_hash, "refusing a package without a frontend");
    check_frontend_vectors(&host, temporary.path(), &emitted);
    assert_unchanged(&host, &host_hash, "checking a package's frontend vectors");
}

/// The TP4 fixture's own frontend vectors, which `bundle check-frontend`
/// runs against its emitted package.
const TP4_FRONTEND_VECTORS: [(&str, &str); 3] = [
    ("--render", "render-vectors.json"),
    ("--tokenizer", "tokenizer-vectors.json"),
    ("--streaming", "streaming-vectors.json"),
];

/// The unchanged host checks the TP4 package's frontend against the
/// fixture's vectors with no checkpoint, printing one passing line per case,
/// and fails a vector that expects something else, naming the difference.
fn check_frontend_vectors(host: &Path, scratch: &Path, emitted: &[(FixtureBundle, PathBuf)]) {
    let vectors = workspace_root().join("crates/bundle-tp4-fixture/tests/frontend");
    let bundle = bundle_of(emitted, TP4_FIXTURE);
    let check = |files: &[(&str, PathBuf)]| {
        let mut command = Command::new(host);
        command.args(["bundle", "check-frontend", "--bundle"]);
        command.arg(bundle);
        for (flag, file) in files {
            command.arg(flag).arg(file);
        }
        command.output().expect("the external host starts")
    };
    let files: Vec<_> = TP4_FRONTEND_VECTORS
        .iter()
        .map(|(flag, name)| (*flag, vectors.join(name)))
        .collect();
    let checked = check(&files);
    let printed = String::from_utf8_lossy(&checked.stdout);
    assert!(
        checked.status.success(),
        "{printed}\n{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let cases: Vec<_> = printed
        .lines()
        .filter(|line| !line.starts_with("checked "))
        .collect();
    let expected: usize = TP4_FRONTEND_VECTORS
        .iter()
        .map(|(_, name)| {
            let cases: Vec<Value> =
                serde_json::from_slice(&fs::read(vectors.join(name)).unwrap()).unwrap();
            cases.len()
        })
        .sum();
    assert_eq!(cases.len(), expected, "{printed}");
    assert!(
        cases.iter().all(|line| line.starts_with("pass ")),
        "{printed}"
    );
    assert!(printed.contains("all passed"), "{printed}");

    // One streaming vector that expects other content fails the run.
    let mut streaming: Vec<Value> =
        serde_json::from_slice(&fs::read(vectors.join("streaming-vectors.json")).unwrap()).unwrap();
    let wrong = scratch.join("wrong-streaming-vectors.json");
    streaming.truncate(1);
    streaming[0]["expect"]["content"] = Value::from("Goodbye!");
    fs::write(&wrong, serde_json::to_vec(&streaming).unwrap()).unwrap();
    let failed = check(&[("--streaming", wrong)]);
    let printed = String::from_utf8_lossy(&failed.stdout);
    assert!(!failed.status.success(), "{printed}");
    assert!(
        printed.starts_with(&format!(
            "FAIL streaming {}: whole: content differs at byte 0",
            streaming[0]["id"].as_str().unwrap()
        )),
        "{printed}"
    );
    assert!(
        String::from_utf8_lossy(&failed.stderr).contains("1 of 1 frontend vector(s)"),
        "{}",
        String::from_utf8_lossy(&failed.stderr)
    );
}

const ONE_PIXEL_RED_97_PNG_V1: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNIZGAAAAEnAGIdeLkqAAAAAElFTkSuQmCC";

/// How much of a served fixture's log a failing test prints: at most this many
/// of its last lines, read from at most its last [`LOG_TAIL_BYTES`].
const LOG_TAIL_LINES: usize = 80;
const LOG_TAIL_BYTES: u64 = 64 * 1024;

struct ServedFixtureV1 {
    process: Child,
    port: u16,
    log: PathBuf,
}

impl ServedFixtureV1 {
    fn start(
        host: &Path,
        root: &Path,
        scratch: &Path,
        label: &str,
        speculative: bool,
        scripted_miss: bool,
    ) -> Self {
        Self::start_with(host, root, scratch, label, speculative, scripted_miss, &[])
    }

    fn start_with(
        host: &Path,
        root: &Path,
        scratch: &Path,
        label: &str,
        speculative: bool,
        scripted_miss: bool,
        extra: &[&str],
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback test port is available");
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let log = scratch.join(format!("served-{label}.log"));
        let stderr = fs::File::create(&log).unwrap();
        let mut command = Command::new(host);
        command.arg("serve");
        command.args(selection(root, TP4_FIXTURE));
        command.args([
            "--topology",
            "tp4",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--fake-host-services",
            "--max-sequences",
            "1",
            "--prefill-chunk-tokens",
            "32",
            "--max-context-tokens",
            "128",
        ]);
        if speculative {
            command.args(["--speculative-tokens", "3", "--speculation", "fixed"]);
        }
        if scripted_miss {
            command.env("TP4_FIXTURE_SCRIPTED_MISS", "1");
        }
        command.args(extra);
        let process = command
            .stdout(Stdio::null())
            .stderr(stderr)
            .spawn()
            .expect("prebuilt host starts");
        let mut server = Self { process, port, log };
        for _ in 0..200 {
            if let Some(status) = server.process.try_wait().unwrap() {
                // The fixture's drop prints its log tail.
                panic!("prebuilt host exited {status}");
            }
            if server
                .exchange("GET", "/health", "")
                .is_some_and(|(status, _)| status == 200)
            {
                // Health names what this host serves: the fixture, the
                // settings it resolved, and the checkpoint its weight plans
                // and loads logged.
                let (_, body) = server
                    .exchange("GET", "/health", "")
                    .expect("health answers");
                let serving = serde_json::from_str::<Value>(&body).unwrap()["serving"].clone();
                let loads = fs::read_to_string(&server.log)
                    .unwrap()
                    .lines()
                    .filter_map(|line| {
                        line.strip_prefix("startup weight_plan: ")
                            .or_else(|| line.strip_prefix("startup direct_load: "))
                    })
                    .map(|load| serde_json::from_str::<Value>(load).unwrap())
                    .collect::<Vec<_>>();
                assert!(!loads.is_empty(), "{body}");
                for load in &loads {
                    assert_eq!(load["checkpoint_identity"], serving["checkpoint_identity"]);
                }
                // The log's identity line is this process's: the same
                // identity, and the nonce this process drew.
                let logged = fs::read_to_string(&server.log)
                    .unwrap()
                    .lines()
                    .filter_map(|line| line.strip_prefix("startup serving_identity: "))
                    .map(|identity| serde_json::from_str::<Value>(identity).unwrap())
                    .collect::<Vec<_>>();
                let mut live = serving.clone();
                live.as_object_mut().unwrap().remove("served_model_names");
                assert_eq!(logged, [live], "{body}");
                assert_eq!(serving["startup_nonce"].as_str().unwrap().len(), 32);
                assert_eq!(serving["bundle_id"], TP4_FIXTURE.bundle_id, "{body}");
                assert_eq!(serving["package_identity"], Value::Null, "{body}");
                let profile = &serving["profile"];
                assert_eq!(profile["name"], Value::Null, "{body}");
                assert_eq!(profile["tensor_parallel_size"], TP4_FIXTURE.rank_count);
                assert_eq!(profile["max_sequences"], 1, "{body}");
                assert_eq!(
                    profile["speculation"]["mode"],
                    if speculative { "fixed" } else { "off" },
                    "{body}"
                );
                assert_eq!(profile["prefix_cache"], !extra.contains(&"off"), "{body}");
                let (status, body) = server
                    .exchange(
                        "POST",
                        "/tokenize",
                        r#"{"prompt":"hi","add_special_tokens":false}"#,
                    )
                    .expect("tokenize answers");
                assert_eq!(status, 200, "{body}");
                let encoded: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(
                    encoded["count"].as_u64().unwrap() as usize,
                    encoded["tokens"].as_array().unwrap().len()
                );
                let (status, body) = server
                    .exchange(
                        "POST",
                        "/detokenize",
                        &serde_json::json!({"tokens": encoded["tokens"]}).to_string(),
                    )
                    .expect("detokenize answers");
                assert_eq!(status, 200, "{body}");
                assert_eq!(
                    serde_json::from_str::<Value>(&body).unwrap()["prompt"],
                    "hi"
                );
                return server;
            }
            thread::sleep(Duration::from_millis(50));
        }
        panic!("prebuilt host did not become healthy");
    }

    fn exchange(&self, method: &str, path: &str, body: &str) -> Option<(u16, String)> {
        let address = format!("127.0.0.1:{}", self.port);
        let mut stream =
            TcpStream::connect_timeout(&address.parse().unwrap(), Duration::from_millis(300))
                .ok()?;
        stream
            .set_read_timeout(Some(Duration::from_secs(20)))
            .ok()?;
        stream
            .set_write_timeout(Some(Duration::from_secs(20)))
            .ok()?;
        write!(stream, "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).ok()?;
        let mut response = Vec::new();
        stream.read_to_end(&mut response).ok()?;
        let split = response.windows(4).position(|bytes| bytes == b"\r\n\r\n")?;
        let head = std::str::from_utf8(&response[..split]).ok()?;
        let status = head.split_whitespace().nth(1)?.parse().ok()?;
        let mut body = &response[split + 4..];
        let body = if head
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            let mut decoded = Vec::new();
            loop {
                let end = body.windows(2).position(|bytes| bytes == b"\r\n")?;
                let size =
                    usize::from_str_radix(std::str::from_utf8(&body[..end]).ok()?, 16).ok()?;
                body = &body[end + 2..];
                if size == 0 {
                    break;
                }
                decoded.extend_from_slice(body.get(..size)?);
                body = body.get(size + 2..)?;
            }
            decoded
        } else {
            body.to_vec()
        };
        Some((status, String::from_utf8(body).ok()?))
    }

    fn post(&self, body: &Value) -> Value {
        let (status, body) = self
            .exchange("POST", "/v1/chat/completions", &body.to_string())
            .unwrap_or_else(|| panic!("prebuilt host did not answer"));
        assert_eq!(status, 200, "prebuilt host refused request: {body}");
        serde_json::from_str(&body).expect("chat response is JSON")
    }

    fn metric(&self, name: &str) -> u64 {
        let (status, body) = self
            .exchange("GET", "/metrics", "")
            .expect("metrics answer");
        assert_eq!(status, 200);
        body.lines()
            .find_map(|line| line.strip_prefix(&format!("{name} ")))
            .unwrap_or_else(|| panic!("metric {name} absent: {body}"))
            .parse()
            .expect("metric has an integer value")
    }

    fn no_page_leak(&self) {
        for _ in 0..40 {
            if self.metric("chop_kv_blocks_used") == 0
                && self.metric("chop_kv_active_sequences") == 0
            {
                return;
            }
            thread::sleep(Duration::from_millis(50));
        }
        panic!("request left KV pages or sequences live");
    }
}

impl Drop for ServedFixtureV1 {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
        // A failure while this server lives, from startup on, also prints what
        // the host logged: the scratch directory and its log go once the
        // failure unwinds. Nothing here may panic, or the unwind aborts.
        if thread::panicking() {
            eprintln!(
                "served host log {}, last {LOG_TAIL_LINES} lines within {LOG_TAIL_BYTES} bytes:\n{}",
                self.log.display(),
                log_tail(&self.log)
            );
        }
    }
}

/// The last [`LOG_TAIL_LINES`] lines within the last [`LOG_TAIL_BYTES`] of a
/// log, decoded lossily. A log that cannot be read is described, not raised.
fn log_tail(path: &Path) -> String {
    let read = || -> std::io::Result<Vec<u8>> {
        let mut file = fs::File::open(path)?;
        let start = file.metadata()?.len().saturating_sub(LOG_TAIL_BYTES);
        file.seek(SeekFrom::Start(start))?;
        let mut bytes = Vec::new();
        file.take(LOG_TAIL_BYTES).read_to_end(&mut bytes)?;
        // A suffix that starts mid-line drops that partial line, unless it is
        // all one line.
        if start > 0
            && let Some(newline) = bytes.iter().position(|&byte| byte == b'\n')
        {
            bytes.drain(..=newline);
        }
        Ok(bytes)
    };
    match read() {
        Ok(bytes) => {
            let text = String::from_utf8_lossy(&bytes);
            let lines: Vec<_> = text.lines().collect();
            lines[lines.len().saturating_sub(LOG_TAIL_LINES)..].join("\n")
        }
        Err(error) => format!("unreadable: {error}"),
    }
}

#[test]
fn a_served_log_tail_is_bounded_and_never_raises() {
    let temporary = TempDirectory::new("served-log-tail");
    let log = temporary.path().join("served.log");
    assert!(log_tail(&log).starts_with("unreadable: "));
    // A long line, then more lines than the tail keeps.
    let mut text = "x".repeat(1 << 20);
    for line in 0..100 {
        text.push_str(&format!("\nline {line}"));
    }
    fs::write(&log, &text).unwrap();
    let tail = log_tail(&log);
    assert_eq!(tail.lines().count(), LOG_TAIL_LINES);
    assert!(
        tail.starts_with("line 20\n") && tail.ends_with("line 99"),
        "{tail}"
    );
    // One line longer than the byte bound keeps only its last bytes.
    fs::write(&log, format!("{}\u{e9}end", "y".repeat(1 << 20))).unwrap();
    let tail = log_tail(&log);
    assert_eq!(tail.len() as u64, LOG_TAIL_BYTES);
    assert!(tail.ends_with("\u{e9}end"));
}

/// A JSON answer never samples one of the model's output markers, even where
/// its bytes would fit a JSON string. The fixture's markers are ordinary
/// tokens, as `<think>` and `<tool_call>` are in served tokenizers; one
/// sampled inside a string used to reach the completion parser as structure,
/// which refused the completion with HTTP 500 `invalid_model_output`. The
/// biases open the object and a key, then favour the reasoning close marker
/// over everything else the string admits.
fn assert_answer_grammars_never_sample_a_marker(server: &ServedFixtureV1) {
    let byte = |byte: u8| bundle_tp4_fixture::TP4_FIXTURE_BYTE_TOKEN_FIRST_ID_V1 + u32::from(byte);
    let reasoning_close = bundle_tp4_fixture::TP4_FIXTURE_OUTPUT_MARKERS_V1[1];
    let (marker, _) = bundle_tp4_fixture::tp4_published_markers_v1()
        .into_iter()
        .find(|(_, text)| *text == reasoning_close)
        .expect("the reasoning close marker is published");
    for stream in [false, true] {
        let mut request = serde_json::json!({
            "model": "fixture/tp4", "messages": [{"role": "user", "content": "a"}],
            "chat_template_kwargs": {"enable_thinking": false},
            "response_format": {"type": "json_object"},
            "temperature": 0, "max_tokens": 8, "return_token_ids": true, "stream": stream,
            "logit_bias": {
                byte(b'{').to_string(): 50, byte(b'"').to_string(): 40, marker.to_string(): 100,
            },
        });
        if stream {
            request["stream_options"] = serde_json::json!({"include_usage": true});
        }
        let (status, body) = server
            .exchange("POST", "/v1/chat/completions", &request.to_string())
            .unwrap();
        assert_eq!(status, 200, "stream {stream}: {body}");
        let responses: Vec<Value> = if stream {
            body.lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .map(|data| serde_json::from_str(data).unwrap())
                .collect()
        } else {
            vec![serde_json::from_str(&body).unwrap()]
        };
        let mut content = String::new();
        let mut tokens = Vec::new();
        for response in &responses {
            for choice in response["choices"].as_array().unwrap() {
                let message = &choice[if stream { "delta" } else { "message" }];
                content.push_str(message["content"].as_str().unwrap_or(""));
                if let Some(ids) = choice["token_ids"].as_array() {
                    tokens.extend(ids.iter().map(|id| id.as_u64().unwrap()));
                }
            }
        }
        assert!(content.starts_with("{\""), "stream {stream}: {body}");
        assert!(
            !content.contains(reasoning_close),
            "stream {stream}: {body}"
        );
        assert!(
            !tokens.contains(&u64::from(marker)),
            "stream {stream}: {body}"
        );
        assert_eq!(tokens.len(), 8, "stream {stream}: {body}");
    }
}

fn assert_served_byte_tail_buffering(server: &ServedFixtureV1) {
    for byte in [0x80_u8, 0xe2] {
        let token_id = bundle_tp4_fixture::TP4_FIXTURE_BYTE_TOKEN_FIRST_ID_V1 + u32::from(byte);
        for chat in [false, true] {
            for stream in [false, true] {
                let mut request = serde_json::json!({
                    "model": "fixture/tp4", "temperature": 0, "max_tokens": 1,
                    "logit_bias": {token_id.to_string(): 100}, "stream": stream,
                    "return_token_ids": true,
                });
                let path = if chat {
                    request["messages"] = serde_json::json!([{"role": "user", "content": "a"}]);
                    request["logprobs"] = Value::Bool(true);
                    request["top_logprobs"] = Value::from(1);
                    "/v1/chat/completions"
                } else {
                    request["prompt"] = Value::from("a");
                    request["logprobs"] = Value::from(1);
                    "/v1/completions"
                };
                if stream {
                    request["stream_options"] = serde_json::json!({"include_usage": true});
                }
                let (status, body) = server.exchange("POST", path, &request.to_string()).unwrap();
                assert_eq!(
                    status, 200,
                    "byte {byte:02x}, chat {chat}, stream {stream}: {body}"
                );
                let responses: Vec<Value> = if stream {
                    assert!(body.contains("data: [DONE]"), "{body}");
                    body.lines()
                        .filter_map(|line| line.strip_prefix("data: "))
                        .filter(|data| *data != "[DONE]")
                        .map(|data| serde_json::from_str(data).unwrap())
                        .collect()
                } else {
                    vec![serde_json::from_str(&body).unwrap()]
                };
                let mut output = String::new();
                let mut tokens = Vec::new();
                let mut labels = Vec::new();
                let mut finished = false;
                for response in &responses {
                    for choice in response["choices"].as_array().unwrap() {
                        let content = if chat {
                            &choice[if stream { "delta" } else { "message" }]["content"]
                        } else {
                            &choice["text"]
                        };
                        output.push_str(content.as_str().unwrap_or(""));
                        if let Some(ids) = choice["token_ids"].as_array() {
                            tokens.extend(ids.iter().cloned());
                        }
                        if chat {
                            if let Some(rows) = choice["logprobs"]["content"].as_array() {
                                labels.extend(rows.iter().map(|row| row["token"].clone()));
                                // vLLM reports the bytes of the label, which
                                // is empty until the character completes.
                                for row in rows {
                                    assert_eq!(row["bytes"], serde_json::json!([]));
                                }
                            }
                        } else if let Some(rows) = choice["logprobs"]["tokens"].as_array() {
                            labels.extend(rows.iter().cloned());
                        }
                        finished |= choice["finish_reason"] == "length";
                    }
                }
                // Generation never flushes a trailing replacement fragment.
                // Whole-input /detokenize remains a separate lossy operation.
                assert_eq!(output, "", "{body}");
                assert_eq!(
                    tokens,
                    vec![Value::from(token_id)],
                    "forced byte was not sampled: {body}"
                );
                assert_eq!(labels, vec![Value::from("")], "{body}");
                assert!(finished, "{body}");
                assert_eq!(responses.last().unwrap()["usage"]["completion_tokens"], 1);
                assert_eq!(server.exchange("GET", "/health", "").unwrap().0, 200);
                let healthy = server.post(&serde_json::json!({
                    "model": "fixture/tp4", "messages": [{"role": "user", "content": "a"}],
                    "temperature": 0, "max_tokens": 1,
                }));
                assert_eq!(healthy["usage"]["completion_tokens"], 1);
                server.no_page_leak();
            }
        }
    }
}

fn assert_served_media_and_speculation(host: &Path, root: &Path, scratch: &Path) {
    let request = |content: Value| {
        serde_json::json!({
            "model": "fixture/tp4", "messages": [{"role": "user", "content": content}],
            "max_tokens": 10, "temperature": 0, "logprobs": true, "top_logprobs": 5,
        })
    };
    let text = request(Value::from("a"));
    let mut constrained = request(Value::from("a"));
    constrained["structured_outputs"] = serde_json::json!({"regex": "[a-c]{3}-[0-9]"});
    let image = request(serde_json::json!([{"type": "image_url", "image_url": {
        "url": format!("data:image/png;base64,{ONE_PIXEL_RED_97_PNG_V1}")
    }}]));
    let plain = ServedFixtureV1::start(host, root, scratch, "plain", false, false);
    assert_served_byte_tail_buffering(&plain);
    assert_answer_grammars_never_sample_a_marker(&plain);
    for script in ["api.py", "structured.py", "release_gate.py"] {
        let qualification = Command::new("python3")
            .arg("-B")
            .arg(workspace_root().join("tools/qual").join(script))
            .args([
                "--url",
                &format!("http://127.0.0.1:{}", plain.port),
                "--mechanics-only",
                "--timeout",
                "10",
            ])
            .env("CHOPINFER_QUAL_RECEIPTS", scratch.join("api-receipts"))
            .env(
                "CHOPINFER_QUAL_CONFIG",
                workspace_root().join("tools/qual/gates/mechanics.json"),
            )
            .output()
            .expect("API qualification starts");
        assert!(
            qualification.status.success(),
            "{script} mechanics failed: {} {}",
            String::from_utf8_lossy(&qualification.stdout),
            String::from_utf8_lossy(&qualification.stderr)
        );
    }
    let baseline = plain.post(&text);
    let constrained_baseline = plain.post(&constrained);
    let media = plain.post(&image);
    assert_eq!(
        baseline["choices"], media["choices"],
        "fake copy encoder must match text logits row by row"
    );
    assert_eq!(baseline["usage"], media["usage"]);
    plain.no_page_leak();
    drop(plain);

    let speculative = ServedFixtureV1::start(host, root, scratch, "speculative", true, false);
    let accepted = speculative.post(&text);
    assert_eq!(baseline["choices"], accepted["choices"]);
    assert_eq!(baseline["usage"], accepted["usage"]);
    // Verify rows carry their own masks, so a constrained request emits the
    // same tokens with speculation.
    assert_eq!(
        constrained_baseline["choices"],
        speculative.post(&constrained)["choices"]
    );
    assert!(speculative.metric("chop_spec_verify_items_total") > 0);
    assert!(speculative.metric("chop_spec_accepted_tokens_total") > 0);
    speculative.no_page_leak();
    drop(speculative);

    let misses = ServedFixtureV1::start(host, root, scratch, "misses", true, true);
    let scripted = misses.post(&text);
    assert_eq!(baseline["choices"], scripted["choices"]);
    assert_eq!(baseline["usage"], scripted["usage"]);
    assert_eq!(
        constrained_baseline["choices"],
        misses.post(&constrained)["choices"]
    );
    assert!(misses.metric("chop_spec_verify_items_total") > 0);
    assert!(
        misses.metric("chop_spec_draft_tokens_total")
            > misses.metric("chop_spec_accepted_tokens_total")
    );
    misses.no_page_leak();
}

/// `chopinfer-top --list --json` reports a served fixture: found as this
/// user's serve process on Linux, and as an explicit target everywhere, with
/// the identity `/health` reports and the tokens it generated.
fn assert_dashboard_lists_a_served_fixture(
    target_directory: &Path,
    host: &Path,
    root: &Path,
    scratch: &Path,
) {
    run(
        cargo(target_directory).args([
            "build",
            "--locked",
            "--offline",
            "-p",
            "chopinfer-top",
            "--bin",
            "chopinfer-top",
        ]),
        "build chopinfer-top",
    );
    let dashboard = target_directory.join("debug").join("chopinfer-top");
    let server = ServedFixtureV1::start(host, root, scratch, "dashboard", false, false);
    server.post(&serde_json::json!({
        "model": "fixture/tp4", "messages": [{"role": "user", "content": "a"}],
        "max_tokens": 4, "temperature": 0,
    }));
    let (status, health) = server
        .exchange("GET", "/health", "")
        .expect("health answers");
    assert_eq!(status, 200, "{health}");
    let nonce = serde_json::from_str::<Value>(&health).unwrap()["serving"]["startup_nonce"].clone();
    let address = format!("127.0.0.1:{}", server.port);
    let list = |arguments: &[&str]| -> Value {
        let output = run(
            Command::new(&dashboard)
                .args(["--list", "--json", "--no-docker", "--interval", "0.5"])
                .args(arguments)
                .env_remove("CHOPINFER_TOP_TARGETS"),
            "chopinfer-top --list",
        );
        let listing: Value = serde_json::from_str(&output).expect("the listing is JSON");
        assert_eq!(listing["schema"], "chopinfer.top.list.v1", "{output}");
        listing
    };
    let check = |instance: &Value, source: &str| {
        assert_eq!(instance["source"], source, "{instance}");
        assert_eq!(instance["address"], address.as_str(), "{instance}");
        assert_eq!(instance["group"], "ready", "{instance}");
        assert_eq!(
            instance["identity"]["bundle_id"], TP4_FIXTURE.bundle_id,
            "{instance}"
        );
        assert_eq!(instance["identity"]["startup_nonce"], nonce, "{instance}");
        assert!(
            instance["totals"]["generation_tokens"].as_f64().unwrap() > 0.0,
            "{instance}"
        );
    };
    if cfg!(target_os = "linux") {
        // Other tests may serve at the same time; this server is the one
        // with its pid.
        let listing = list(&[]);
        let pid = u64::from(server.process.id());
        let found = listing["instances"]
            .as_array()
            .unwrap()
            .iter()
            .find(|instance| instance["pid"] == pid)
            .unwrap_or_else(|| panic!("process {pid} was not discovered: {listing}"));
        check(found, "proc");
    }
    let listing = list(&["--no-discover", "--target", &address]);
    assert_eq!(
        listing["instances"].as_array().unwrap().len(),
        1,
        "{listing}"
    );
    check(&listing["instances"][0], "target");
}

/// A repeated prompt resumes from the anchor its second use captured, and the
/// completion equals the one a server with `--prefix-cache off` returns.
fn assert_legacy_prefill_chunks(host: &Path, root: &Path, scratch: &Path) {
    let server = ServedFixtureV1::start_with(
        host,
        root,
        scratch,
        "legacy-chunks",
        false,
        false,
        &["--prefix-cache", "off"],
    );
    let request = serde_json::json!({
        "model": "fixture/tp4", "prompt": vec![2; 64],
        "max_tokens": 1, "temperature": 0,
    });
    let (status, body) = server
        .exchange("POST", "/v1/completions", &request.to_string())
        .expect("the legacy server answers");
    assert_eq!(
        status, 200,
        "legacy prefill must retain 32-token chunks: {body}"
    );
    let response: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(response["usage"]["prompt_tokens"], 64);
    assert_eq!(response["usage"]["completion_tokens"], 1);
    server.no_page_leak();
}

fn assert_served_prefix_cache(host: &Path, root: &Path, scratch: &Path) {
    let request = serde_json::json!({
        "model": "fixture/tp4",
        "messages": [{"role": "user", "content": "a prompt long enough to fill one page"}],
        "max_tokens": 1, "temperature": 0, "logprobs": true, "top_logprobs": 5,
    });
    let plain = ServedFixtureV1::start_with(
        host,
        root,
        scratch,
        "prefix-off",
        false,
        false,
        &["--prefix-cache", "off"],
    );
    let baseline = plain.post(&request);
    assert_eq!(
        baseline["usage"]["prompt_tokens_details"]["cached_tokens"], 0,
        "a server without the cache reports no cached tokens"
    );
    drop(plain);

    // The cache is on by default.
    let cached = ServedFixtureV1::start(host, root, scratch, "prefix-on", false, false);
    let prompt_tokens = baseline["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(
        prompt_tokens > 32,
        "the prompt spans more than one page: {prompt_tokens}"
    );
    let mut reused = Vec::new();
    for _ in 0..3 {
        let response = cached.post(&request);
        assert_eq!(baseline["choices"], response["choices"]);
        assert_eq!(
            baseline["usage"]["completion_tokens"],
            response["usage"]["completion_tokens"]
        );
        reused.push(response["usage"]["prompt_tokens_details"]["cached_tokens"].clone());
    }
    assert_eq!(
        reused,
        [0, 32, 32],
        "a repeat hits the start of the final chunk its first use captured: {}",
        cached.exchange("GET", "/metrics", "").unwrap().1
    );
    assert_eq!(cached.metric("chop_prefix_cache_hits_total"), 2);
    assert_eq!(cached.metric("chop_prefix_cache_reused_tokens_total"), 64);
    assert!(cached.metric("chop_prefix_cache_anchors") > 0);
    // vLLM's series count queried and cached prompt tokens; the three
    // requests above are not the only lookups since startup.
    let queried = cached.metric("chop_prefix_cache_queried_tokens_total");
    assert!(queried >= 3 * prompt_tokens, "{queried}");
    let metrics = cached.exchange("GET", "/metrics", "").unwrap().1;
    for (name, value) in [
        ("vllm:prefix_cache_queries_total", queried),
        ("vllm:prefix_cache_hits_total", 64),
    ] {
        let line = format!("{name}{{model_name=\"fixture/tp4\",engine=\"0\"}} {value}");
        assert!(
            metrics.lines().any(|row| row == line),
            "{line} in {metrics}"
        );
    }
    cached.no_page_leak();
}

/// What one scenario's receipt has to say about the run it reports.
///
/// The status is the scenario's own, so a run that quietly committed where it
/// was told to cancel is caught rather than read as a pass. The resource count
/// is the cleanup result: whatever the scenario did, the host owns nothing on
/// the bundle's behalf when the run ends.
fn assert_scenario_receipt(receipt: &Value, fixture: FixtureBundle, scenario: HostScenario) {
    let at = |field: &str| {
        receipt.get(field).unwrap_or_else(|| {
            panic!(
                "the {} receipt names no '{field}': {receipt}",
                scenario.name
            )
        })
    };

    assert_eq!(at("bundle"), &Value::from(fixture.bundle_id));
    assert_eq!(at("scenario"), &Value::from(scenario.name));
    assert_eq!(at("status"), &Value::from(scenario.status));
    assert_eq!(at("ranks"), &Value::from(fixture.rank_count));
    // The identities the bundle declared and the run was given, read back off
    // the receipt rather than assumed from the host's defaults.
    assert_eq!(at("topology"), &Value::from(fixture.topology_id));
    assert_eq!(at("architecture"), &Value::from(fixture.architecture));
    assert_eq!(
        at("live_resources"),
        &Value::from(0),
        "a finished scenario leaves this host owning nothing: {receipt}"
    );
    assert_eq!(at("executions"), &Value::from(scenario.executions));
    // One row per execution, because one row is what each one schedules.
    assert_eq!(at("rows"), &Value::from(scenario.executions));
    assert_eq!(at("request"), &Value::from("external-bundle-host"));

    let tokens = at("prefill_tokens")
        .as_u64()
        .expect("the prefill token count is a number");
    match fixture.fixture_tokens {
        Some(named) => assert_eq!(tokens, named as u64),
        None => assert!(tokens > 0, "a run drives at least one token: {receipt}"),
    }
    assert!(
        at("logits_per_row")
            .as_u64()
            .is_some_and(|logits| logits > 0),
        "a row this host kept has a width: {receipt}"
    );
}

/// Every scenario of every bundle reports the same host running it.
///
/// The binary is hashed after each run, so this is the receipt saying the same
/// thing from the inside: one architecture, one ABI version, and one kind of
/// device services across both packages and all three scenarios.
fn assert_one_host_ran_everything(
    identity: &mut Option<Value>,
    receipt: &Value,
    fixture: FixtureBundle,
    scenario: HostScenario,
) {
    let stated = Value::Array(
        ["architecture", "abi_version", "device"]
            .iter()
            .map(|field| {
                receipt
                    .get(*field)
                    .unwrap_or_else(|| {
                        panic!(
                            "the {} receipt names no '{field}': {receipt}",
                            scenario.name
                        )
                    })
                    .clone()
            })
            .collect(),
    );
    match identity {
        Some(first) => assert_eq!(
            &stated, first,
            "the {} scenario of {} names another host: {receipt}",
            scenario.name, fixture.bundle_id
        ),
        None => *identity = Some(stated),
    }
}

/// A fixture that names a token count is one whose declared page is not it.
///
/// Without this, `--fixture-tokens 32` could be a flag that changes nothing and
/// the run would pass either way. A fixture that names no count has nothing to check here: its declared page
/// is the batch it serves.
fn assert_named_token_count_is_the_batch(host: &Path, root: &Path, fixture: FixtureBundle) {
    if fixture.fixture_tokens.is_none() {
        return;
    }
    let mut command = Command::new(host);
    command.args(["--fixture-request", "external-bundle-host"]);
    command.args(selection(root, fixture));
    command.args(["--topology", fixture.topology_id]);
    if cfg!(target_os = "macos") {
        command.arg("--fake-host-services");
    }
    let taken = command.output().expect("the external host starts");

    assert!(
        !taken.status.success(),
        "{} names a token count, so its declared page must not also be a batch it serves:\n{}",
        fixture.bundle_id,
        String::from_utf8_lossy(&taken.stdout)
    );
}

/// A fixture that declares a shorter tail serves it through the same host.
///
/// A bundle whose one operation declares a fixed operand shape has to be lent
/// that shape whatever the batch carries, so the batch that is not the declared
/// page is the one that says the shape holds. This runs the same emitted
/// bundle again over a tail shorter than its page and reads the count back off
/// the receipt. A fixture that declares no tail has nothing to check here.
fn assert_declared_tail_also_runs(host: &Path, root: &Path, fixture: FixtureBundle) {
    let Some(tail) = fixture.tail_tokens else {
        return;
    };
    let mut command = Command::new(host);
    command.args(["--fixture-request", "external-bundle-host"]);
    if cfg!(target_os = "macos") {
        command.arg("--fake-host-services");
    }
    command.args(selection(root, fixture));
    command.args([
        "--topology",
        fixture.topology_id,
        "--fixture-tokens",
        &tail.to_string(),
    ]);
    let printed = run(
        &mut command,
        &format!("run a {tail}-token tail of {} ", fixture.bundle_id),
    );
    let receipt: Value = serde_json::from_str(printed.trim()).unwrap_or_else(|error| {
        panic!("the tail receipt must be one JSON line: {error}\n{printed}")
    });

    assert_eq!(receipt["bundle"], Value::from(fixture.bundle_id));
    assert_eq!(receipt["status"], Value::from("committed"));
    assert_eq!(
        receipt["prefill_tokens"],
        Value::from(tail),
        "the tail this host drove is not the one it was asked for: {receipt}"
    );
    assert_eq!(
        receipt["live_resources"],
        Value::from(0),
        "a finished tail leaves this host owning nothing: {receipt}"
    );
    assert!(
        receipt["logits_per_row"]
            .as_u64()
            .is_some_and(|logits| logits > 0),
        "a tail emits one row of the same declared width: {receipt}"
    );
}

/// Five things this host refuses before it opens a native module.
///
/// Each subject is a copy of an emitted bundle with one manifest field
/// rewritten, and each copy is otherwise a well-formed package: the schema, ABI
/// version, capability, and artifact-origin cases are refused during
/// verification, and the generic-kernel case is refused by the host's own
/// operation catalog after the package verifies.
fn reject_unloadable_bundles(
    host: &Path,
    host_hash: &str,
    scratch: &Path,
    emitted: &[(FixtureBundle, PathBuf)],
) {
    let tp4 = bundle_of(emitted, TP4_FIXTURE);
    let recurrent = bundle_of(emitted, RECURRENT_FIXTURE);

    for (label, fixture, source, from, to, refusal) in [
        (
            "another-schema",
            TP4_FIXTURE,
            tp4,
            "\"schema\":\"chop-model-bundle-v1\"",
            "\"schema\":\"chop-model-bundle-v2\"".to_owned(),
            "unsupported bundle schema",
        ),
        (
            "another-abi-version",
            TP4_FIXTURE,
            tp4,
            "\"abi_version\":1",
            "\"abi_version\":2".to_owned(),
            "unsupported bundle ABI version",
        ),
        // Anchored on the whole resource-requirements object, because a
        // semantic contract carries a capability list of its own and only the
        // package's requirements are the host's to answer for.
        (
            "an-absent-capability",
            TP4_FIXTURE,
            tp4,
            "\"resource_requirements\":{\"minimum_device_memory_bytes\":4194304,\
             \"required_capabilities\":[\"chop.frontend.v1\",\"chop.media-inputs.v1\",\"chop.prefix-cache.v1\",\"chop.prompt-rows.v2\",\"chop.request-retirement.v1\",\"chop.speculative.v1\",\"chop.weight-plan.v1\"]}",
            format!(
                "\"resource_requirements\":{{\"minimum_device_memory_bytes\":4194304,\
                 \"required_capabilities\":[\"{ABSENT_CAPABILITY}\",\"chop.frontend.v1\",\"chop.media-inputs.v1\",\"chop.prefix-cache.v1\",\"chop.prompt-rows.v2\",\"chop.request-retirement.v1\",\"chop.speculative.v1\",\"chop.weight-plan.v1\"]}}"
            ),
            "requires unsupported host capability",
        ),
        (
            "an-unserved-generic-operation",
            RECURRENT_FIXTURE,
            recurrent,
            "\"chop.numerics.exact-copy.v1\"",
            "\"chop.numerics.other-copy.v1\"".to_owned(),
            "this host cannot run the selected bundle's operation",
        ),
        // The artifact origin is enforced, not decorative: a record that says
        // its code is a component has to be that component, measured.
        (
            "an-implementation-that-is-not-the-file-it-names",
            TP4_FIXTURE,
            tp4,
            "\"origin\":{\"kind\":\"native-module\"}",
            "\"origin\":{\"component\":\"frontend\",\
             \"kind\":\"component\"}"
                .to_owned(),
            "artifact does not match component",
        ),
    ] {
        let derived = scratch.join(format!("rejected-{label}"));
        derive_bundle(source, &derived, from, &to);

        let refused = run_bundle_directory(host, &derived, scratch, fixture, &[]);

        assert!(
            !refused.status.success(),
            "a bundle declaring {label} must be refused, and it ran:\n{}",
            String::from_utf8_lossy(&refused.stdout)
        );
        let said = String::from_utf8_lossy(&refused.stderr);
        assert!(
            said.contains(refusal),
            "a bundle declaring {label} must be refused with '{refusal}', and it said:\n{said}"
        );
        assert_unchanged(host, host_hash, &format!("refusing {label}"));
    }

    // The capability case is the one a host can answer for, so the same
    // directory that was just refused loads once this host declares it. That is
    // what says the refusal was the check working rather than a broken package.
    let declared = scratch.join("rejected-an-absent-capability");
    let accepted = run_bundle_directory(
        host,
        &declared,
        scratch,
        TP4_FIXTURE,
        &["--capability", ABSENT_CAPABILITY],
    );
    assert!(
        accepted.status.success(),
        "a host declaring {ABSENT_CAPABILITY} must load the same bundle:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&accepted.stdout),
        String::from_utf8_lossy(&accepted.stderr)
    );
    assert_unchanged(
        host,
        host_hash,
        "running a bundle whose capability this host declared",
    );
}

/// A package that does not require `chop.frontend.v1` serves no chat
/// frontend: the host has no format of its own, so serving it is refused
/// before its module opens.
fn refuse_serving_without_a_frontend(
    host: &Path,
    root: &Path,
    scratch: &Path,
    emitted: &[(FixtureBundle, PathBuf)],
) {
    let derived = scratch.join("rejected-no-frontend");
    derive_bundle(
        bundle_of(emitted, TP4_FIXTURE),
        &derived,
        "\"chop.frontend.v1\",",
        "",
    );
    let refused = Command::new(host)
        .args(["serve", "--bundle"])
        .arg(&derived)
        .arg("--checkpoint")
        .arg(checkpoint_directory(root, TP4_FIXTURE))
        .args([
            "--topology",
            "tp4",
            "--listen",
            "127.0.0.1:0",
            "--fake-host-services",
            "--max-sequences",
            "1",
            "--prefill-chunk-tokens",
            "32",
            "--max-context-tokens",
            "128",
        ])
        .output()
        .expect("prebuilt host starts");
    let said = String::from_utf8_lossy(&refused.stderr);
    assert!(!refused.status.success(), "{said}");
    assert!(
        said.contains("the package does not require chop.frontend.v1"),
        "{said}"
    );
    assert!(!said.contains("startup module load:"), "{said}");
}

fn bundle_of(emitted: &[(FixtureBundle, PathBuf)], fixture: FixtureBundle) -> &Path {
    emitted
        .iter()
        .find(|(candidate, _)| candidate.bundle_id == fixture.bundle_id)
        .map(|(_, bundle)| bundle.as_path())
        .unwrap_or_else(|| panic!("{} was emitted before it is derived", fixture.bundle_id))
}

/// Copies one emitted bundle and rewrites exactly one manifest field.
///
/// The replacement must occur once and only once, so a manifest that changed
/// shape fails here instead of quietly producing a package that differs
/// somewhere else. The rewritten manifest is still canonical JSON: only a
/// value is replaced, and no key order changes.
fn derive_bundle(source: &Path, destination: &Path, from: &str, to: &str) {
    let _ = fs::remove_dir_all(destination);
    copy_tree(source, destination);
    let manifest_path = destination.join("manifest.json");
    let manifest = fs::read_to_string(&manifest_path).expect("the emitted manifest is UTF-8");
    assert_eq!(
        manifest.matches(from).count(),
        1,
        "'{from}' must appear exactly once in {}",
        manifest_path.display()
    );
    fs::write(&manifest_path, manifest.replace(from, to))
        .expect("the derived manifest is writable");
}

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).expect("the derived bundle directory is creatable");
    for entry in fs::read_dir(source).expect("the emitted bundle directory is readable") {
        let entry = entry.expect("the emitted bundle entry is readable");
        let target = destination.join(entry.file_name());
        if entry
            .file_type()
            .expect("the emitted bundle entry has a type")
            .is_dir()
        {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target).expect("the emitted bundle file is copyable");
        }
    }
}

fn build_host(target_directory: &Path, snapshot_directory: &Path) -> PathBuf {
    let mut command = cargo(target_directory);
    command.args([
        "build",
        "--locked",
        "--offline",
        "-p",
        "runtime-host",
        "--bin",
        "chopinfer-host",
    ]);
    if cfg!(target_os = "macos") {
        command.args(["--no-default-features", "--features", "fake-host-services"]);
    }
    run(&mut command, "build one external-bundle host");

    let built = target_directory.join("debug").join(host_binary_name());
    assert!(built.is_file(), "external-bundle host is missing");
    // Cargo may update its shared target while building either fixture. Run a
    // private copy so the hash still proves every scenario used one binary.
    let host = snapshot_directory.join(host_binary_name());
    fs::copy(&built, &host).expect("snapshot the external-bundle host");
    host
}

/// Emits `fixture`'s package and writes its synthetic checkpoint under `root`.
fn build_fixture(fixture: FixtureBundle, root: &Path, target_directory: &Path) {
    let output = bundle_directory(root, fixture);
    let mut command = cargo(target_directory);
    command.args([
        "run",
        "--locked",
        "--offline",
        "-p",
        fixture.package,
        "--bin",
        fixture.builder,
    ]);
    if cfg!(target_os = "macos") {
        command.args(["--no-default-features", "--features", "fake-host-services"]);
    }
    command.args([
        "--",
        "--fixture",
        fixture.fixture,
        "--output",
        output.to_str().expect("fixture output path is UTF-8"),
        "--checkpoint",
        checkpoint_directory(root, fixture)
            .to_str()
            .expect("fixture checkpoint path is UTF-8"),
    ]);
    run(&mut command, "build an external fixture bundle");
    assert!(output.is_dir(), "external fixture bundle is missing");
}

fn bundle_directory(root: &Path, fixture: FixtureBundle) -> PathBuf {
    root.join(fixture.bundle_id)
}

fn checkpoint_directory(root: &Path, fixture: FixtureBundle) -> PathBuf {
    root.join(format!("{}.checkpoint", fixture.bundle_id))
}

/// The emitted bundle and the checkpoint its weight plan loads.
fn selection(root: &Path, fixture: FixtureBundle) -> [String; 4] {
    [
        "--bundle".to_owned(),
        bundle_directory(root, fixture).display().to_string(),
        "--checkpoint".to_owned(),
        checkpoint_directory(root, fixture).display().to_string(),
    ]
}

/// Runs one scenario of one emitted bundle, and reads the receipt it printed.
fn run_emitted(host: &Path, root: &Path, fixture: FixtureBundle, scenario: HostScenario) -> Value {
    let mut command = host_command(host, fixture);
    command.args(selection(root, fixture));
    command.args(["--scenario", scenario.name]);
    let printed = run(
        &mut command,
        &format!(
            "run the {} scenario of an emitted bundle through the external host",
            scenario.name
        ),
    );
    serde_json::from_str(printed.trim()).unwrap_or_else(|error| {
        panic!(
            "the {} receipt must be one JSON line: {error}\n{printed}",
            scenario.name
        )
    })
}

/// Runs one bundle directory through the host over `fixture`'s checkpoint.
fn run_bundle_directory(
    host: &Path,
    bundle: &Path,
    root: &Path,
    fixture: FixtureBundle,
    extra: &[&str],
) -> std::process::Output {
    let mut command = host_command(host, fixture);
    command.args(["--bundle", bundle.to_str().expect("bundle path is UTF-8")]);
    command.args([
        "--checkpoint",
        checkpoint_directory(root, fixture)
            .to_str()
            .expect("fixture checkpoint path is UTF-8"),
    ]);
    command.args(extra);
    command.output().expect("the external host starts")
}

/// The conformance invocation both run paths share.
///
/// The topology is the one the bundle declares, named rather than inferred from
/// being its only one. A fixture may supply an explicit token count.
fn host_command(host: &Path, fixture: FixtureBundle) -> Command {
    let mut command = Command::new(host);
    command.args(["--fixture-request", "external-bundle-host"]);
    command.args(["--topology", fixture.topology_id]);
    if let Some(tokens) = fixture.fixture_tokens {
        command.args(["--fixture-tokens", &tokens.to_string()]);
    }
    if cfg!(target_os = "macos") {
        command.arg("--fake-host-services");
    }
    command
}

fn assert_macos_host_path_is_cuda_free(target_directory: &Path) {
    if !cfg!(target_os = "macos") {
        return;
    }

    let mut packages = vec!["runtime-host"];
    packages.extend(EXTERNAL_FIXTURES.iter().map(|fixture| fixture.package));
    for package in packages {
        let mut command = cargo(target_directory);
        command.args([
            "tree",
            "--locked",
            "--offline",
            "-p",
            package,
            "--prefix",
            "none",
            "--no-default-features",
            "--features",
            "fake-host-services",
        ]);
        let output = command
            .output()
            .expect("macOS fake-host dependency audit starts");
        assert!(
            output.status.success(),
            "macOS fake-host dependency audit failed for {package}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let tree = String::from_utf8(output.stdout).expect("dependency audit is UTF-8");
        assert!(
            !tree.lines().any(|line| line.starts_with("runtime-cuda ")),
            "macOS fake-host path compiles runtime-cuda through {package}: {tree}"
        );
    }
}

fn cargo(target_directory: &Path) -> Command {
    let executable = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(executable);
    command
        .current_dir(workspace_root())
        .env("CARGO_TARGET_DIR", target_directory);
    command
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs one command that has to succeed, and returns what it printed.
fn run(command: &mut Command, label: &str) -> String {
    let output = command.output().expect("fixture command starts");
    assert!(
        output.status.success(),
        "{label} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("the command printed UTF-8")
}

fn hash(path: &Path) -> String {
    // The debug host is about 100 MB. Hashing it repeatedly in this debug
    // test costs more than the bundle scenarios. Keep the same full SHA-256
    // check, using a system implementation when one is available.
    for (tool, args) in [("sha256sum", &[][..]), ("shasum", &["-a", "256"][..])] {
        let output = match Command::new(tool).args(args).arg(path).output() {
            Ok(output) => output,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => panic!("{tool} could not hash {}: {error}", path.display()),
        };
        assert!(
            output.status.success(),
            "{tool} failed for {}",
            path.display()
        );
        let printed = String::from_utf8(output.stdout).expect("SHA-256 output is UTF-8");
        let digest = printed
            .split_whitespace()
            .next()
            .expect("SHA-256 digest exists");
        assert!(digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        return digest.to_ascii_lowercase();
    }
    sha256_bytes(&fs::read(path).expect("host binary is readable"))
}

fn assert_unchanged(path: &Path, expected: &str, action: &str) {
    assert_eq!(
        hash(path),
        expected,
        "host binary changed after {action}: {}",
        path.display()
    );
}

fn host_binary_name() -> &'static str {
    #[cfg(windows)]
    {
        "chopinfer-host.exe"
    }
    #[cfg(not(windows))]
    {
        "chopinfer-host"
    }
}

#[derive(Clone, Copy)]
struct FixtureBundle {
    package: &'static str,
    /// The program that emits this bundle's package.
    ///
    /// Each bundle names its own, because one target directory holds one file
    /// per binary name and this gate builds every bundle into one of them.
    builder: &'static str,
    bundle_id: &'static str,
    fixture: &'static str,
    fixture_tokens: Option<usize>,
    /// A second batch this bundle serves, shorter than its declared page.
    tail_tokens: Option<usize>,
    /// The ranks this fixture's one topology declares, which the receipt says.
    rank_count: u32,
    /// The topology every run of this bundle names, and the receipt reports.
    topology_id: &'static str,
    /// The architecture its declared target is qualified for.
    architecture: &'static str,
}

/// One scenario the host binary drives, and what its receipt must report.
#[derive(Clone, Copy)]
struct HostScenario {
    /// What `--scenario` names.
    name: &'static str,
    /// What the receipt calls a run that ended this way.
    status: &'static str,
    /// How many times the host executed the bundle's batch.
    executions: usize,
}

struct TempDirectory {
    path: PathBuf,
}

impl TempDirectory {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);

        let path = std::env::temp_dir().join(format!(
            "{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).expect("fixture temporary directory is creatable");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
