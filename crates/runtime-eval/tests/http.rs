//! The HTTP client against a local fake server.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::thread::JoinHandle;
use std::time::Duration;

use runtime_eval::declaration::CheckpointPin;
use runtime_eval::http::{BindError, Client, Endpoint, HttpError};
use serde_json::{Value, json};

const CHECKPOINT: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const OTHER: &str = "5555555555555555555555555555555555555555555555555555555555555555";
const NONCE: &str = "0123456789abcdef0123456789abcdef";
const OTHER_NONCE: &str = "ffffffffffffffffffffffffffffffff";
/// A canned response that closes the connection without answering.
const DROP: &str = "";

/// Serves one canned response per connection, in order, then stops
/// listening; returns the requests it read (head and body).
fn serve(responses: Vec<String>) -> (String, JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let mut requests = Vec::new();
        for response in responses {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream);
            let mut head = String::new();
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
                head.push_str(&line);
                if line == "\r\n" {
                    break;
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            requests.push(head + &String::from_utf8(body).unwrap());
            let mut stream = reader.into_inner();
            stream.write_all(response.as_bytes()).unwrap();
        }
        requests
    });
    (url, handle)
}

/// A response with `headers` (name, value) after the content headers.
fn answer(status: &str, body: &str, headers: &[(&str, &str)]) -> String {
    let extra: String = headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect();
    format!(
        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n{extra}\r\n{body}",
        body.len()
    )
}

fn identified(status: &str, body: &str, checkpoint: &str, nonce: &str) -> String {
    answer(
        status,
        body,
        &[
            ("x-chopinfer-checkpoint-identity", checkpoint),
            ("x-chopinfer-startup-nonce", nonce),
        ],
    )
}

fn ok(body: &Value) -> String {
    identified("200 OK", &body.to_string(), CHECKPOINT, NONCE)
}

fn models() -> Value {
    json!({"object": "list", "data": [{"id": "Example"}, {"id": "Other"}]})
}

fn health(checkpoint: &str, nonce: &str) -> Value {
    json!({"status": "ready", "serving": {"checkpoint_identity": checkpoint, "startup_nonce": nonce,
                                          "package_identity": null, "metadata_identity": null}})
}

fn pin() -> CheckpointPin {
    CheckpointPin {
        lock: "models/example/checkpoint-files.json".into(),
        repository: "org/example".into(),
        revision: "abc".into(),
        served_model_name: "Example".into(),
        source_lock_sha256: CHECKPOINT.into(),
        checkpoint_identity: CHECKPOINT.into(),
        registry: None,
    }
}

fn connect(url: &str) -> Client {
    Client::new(url, Duration::from_secs(5), 0)
        .unwrap()
        .with_api_key(Some("secret".into()))
}

/// Binds `client` to the pinned checkpoint.
fn bind(client: &mut Client) -> runtime_eval::http::BoundServer {
    let identity = client.identity(None).unwrap();
    client.bind(&identity, &pin()).unwrap()
}

/// The identity reads of a healthy bound server.
fn healthy() -> Vec<String> {
    vec![ok(&models()), ok(&health(CHECKPOINT, NONCE))]
}

#[test]
fn endpoints_parse() {
    let endpoint = Endpoint::parse("http://127.0.0.1:8101/v1/").unwrap();
    assert_eq!(
        (
            endpoint.host.as_str(),
            endpoint.port,
            endpoint.prefix.as_str()
        ),
        ("127.0.0.1", 8101, "/v1")
    );
    let endpoint = Endpoint::parse("http://[::1]:9/").unwrap();
    assert_eq!((endpoint.host.as_str(), endpoint.port), ("::1", 9));
    assert_eq!(Endpoint::parse("http://localhost").unwrap().port, 80);
    for bad in [
        "https://example.org",
        "http://:80",
        "http://host:x",
        "http://[::1",
    ] {
        assert!(Endpoint::parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn identity_binds_and_confirms_the_live_server() {
    let mut responses = healthy();
    responses.push(ok(&json!({"choices": []})));
    responses.extend(healthy());
    let (url, server) = serve(responses);
    let mut client = connect(&url);
    let identity = client.identity(Some("Example")).unwrap();
    assert_eq!(identity.models, ["Example", "Other"]);
    let bound = client.bind(&identity, &pin()).unwrap();
    assert_eq!(bound.startup_nonce, NONCE);
    let body = json!({"model": "Example", "prompt": [1, 2], "max_tokens": 1});
    let (value, _) = client.json("/v1/completions", Some(&body)).unwrap();
    assert_eq!(value, json!({"choices": []}));
    client.confirm_live("Example", &bound).unwrap();
    let requests = server.join().unwrap();
    assert!(requests[0].starts_with("GET /v1/models HTTP/1.1\r\n"));
    assert!(requests[0].contains("Authorization: Bearer secret\r\n"));
    assert!(requests[2].starts_with("POST /v1/completions HTTP/1.1\r\n"));
    assert!(requests[2].ends_with(&body.to_string()));
}

#[test]
fn a_server_with_another_identity_fails_closed() {
    // Another checkpoint in /health.
    let (url, server) = serve(vec![ok(&models()), ok(&health(OTHER, NONCE))]);
    let mut client = connect(&url);
    let identity = client.identity(None).unwrap();
    assert_eq!(identity.model, "Example");
    assert!(matches!(
        client.bind(&identity, &pin()),
        Err(BindError::Mismatch(_))
    ));
    server.join().unwrap();

    // No identity in /health.
    let (url, server) = serve(vec![ok(&models()), ok(&json!({"status": "ready"}))]);
    let mut client = connect(&url);
    let identity = client.identity(None).unwrap();
    assert!(matches!(
        client.bind(&identity, &pin()),
        Err(BindError::Unidentified(_))
    ));
    server.join().unwrap();

    // /health names the pin but a response's headers name another process.
    let (url, server) = serve(vec![
        identified("200 OK", &models().to_string(), OTHER, NONCE),
        ok(&health(CHECKPOINT, NONCE)),
    ]);
    let mut client = connect(&url);
    let identity = client.identity(None).unwrap();
    assert!(matches!(
        client.bind(&identity, &pin()),
        Err(BindError::Mismatch(_))
    ));
    server.join().unwrap();

    // A later response another process wrote.
    let mut responses =
        identity_responses_with(identified("200 OK", "{}", CHECKPOINT, OTHER_NONCE));
    responses.extend(healthy());
    let (url, server) = serve(responses);
    let mut client = connect(&url);
    let bound = bind(&mut client);
    client.json("/v1/completions", Some(&json!({}))).unwrap();
    assert!(matches!(
        client.confirm_live("Example", &bound),
        Err(BindError::Mismatch(_))
    ));
    server.join().unwrap();

    // The server restarted while scoring.
    let mut responses = healthy();
    responses.extend([ok(&models()), ok(&health(CHECKPOINT, OTHER_NONCE))]);
    let (url, server) = serve(responses);
    let mut client = connect(&url);
    let bound = bind(&mut client);
    assert!(matches!(
        client.confirm_live("Example", &bound),
        Err(BindError::Mismatch(_))
    ));
    server.join().unwrap();
}

fn identity_responses_with(response: String) -> Vec<String> {
    let mut responses = healthy();
    responses.push(response);
    responses
}

#[test]
fn a_repeated_identity_header_fails_closed() {
    let names = [
        "x-chopinfer-checkpoint-identity",
        "x-chopinfer-startup-nonce",
    ];
    let values = [(CHECKPOINT, OTHER), (NONCE, OTHER_NONCE)];
    for (index, name) in names.iter().enumerate() {
        let (good, bad) = values[index];
        let other_name = names[1 - index];
        let other_value = values[1 - index].0;
        for (first, second) in [(good, bad), (bad, good), (good, good)] {
            let headers = [(*name, first), (*name, second), (other_name, other_value)];
            for (kind, response) in [
                ("ordinary", answer("200 OK", "{}", &headers)),
                (
                    "error",
                    answer("400 Bad Request", "{\"error\": \"no\"}", &headers),
                ),
                (
                    "streamed",
                    answer("200 OK", "data: {\"n\": 1}\n\ndata: [DONE]\n\n", &headers),
                ),
            ] {
                let mut responses = identity_responses_with(response);
                responses.extend(healthy());
                let (url, server) = serve(responses);
                let mut client = connect(&url);
                let bound = bind(&mut client);
                match kind {
                    "streamed" => drop(client.stream("/v1/completions", &json!({"stream": true}))),
                    _ => drop(client.json("/v1/completions", Some(&json!({})))),
                }
                let confirmed = client.confirm_live("Example", &bound);
                assert!(
                    matches!(confirmed, Err(BindError::Mismatch(_))),
                    "{name} {first} {second} {kind}: {confirmed:?}"
                );
                server.join().unwrap();
            }
        }
    }
    // At binding, too.
    let (url, server) = serve(vec![
        ok(&models()),
        answer(
            "200 OK",
            &health(CHECKPOINT, NONCE).to_string(),
            &[
                ("x-chopinfer-checkpoint-identity", CHECKPOINT),
                ("x-chopinfer-startup-nonce", NONCE),
                ("x-chopinfer-startup-nonce", NONCE),
            ],
        ),
    ]);
    let mut client = connect(&url);
    let identity = client.identity(None).unwrap();
    assert!(matches!(
        client.bind(&identity, &pin()),
        Err(BindError::Mismatch(_))
    ));
    server.join().unwrap();
}

#[test]
fn responses_without_identity_headers_leave_results_unidentified() {
    let mut responses = identity_responses_with(answer("200 OK", "{}", &[]));
    responses.extend(healthy());
    let (url, server) = serve(responses);
    let mut client = connect(&url);
    let bound = bind(&mut client);
    client.json("/v1/completions", Some(&json!({}))).unwrap();
    assert!(matches!(
        client.confirm_live("Example", &bound),
        Err(BindError::Unidentified(_))
    ));
    server.join().unwrap();
}

#[test]
fn a_health_read_keeps_transport_status_and_body_failures_apart() {
    // The connection drops, and the server is gone for any retry.
    let (url, server) = serve(vec![ok(&models()), DROP.into()]);
    let mut retrying = Client::new(&url, Duration::from_secs(2), 1).unwrap();
    let read = retrying.identity(None);
    server.join().unwrap();
    assert!(matches!(read, Err(HttpError::Transport(_))), "{read:?}");

    // An error status or a body that is not JSON leaves the server unidentified.
    for response in [
        identified("503 Service Unavailable", "{}", CHECKPOINT, NONCE),
        identified("200 OK", "not json", CHECKPOINT, NONCE),
    ] {
        let (url, server) = serve(vec![ok(&models()), response]);
        let identity = connect(&url).identity(None).unwrap();
        assert_eq!(identity.health, None);
        server.join().unwrap();
    }

    // An unserved model is its own error.
    let (url, server) = serve(vec![ok(&models())]);
    assert!(matches!(
        connect(&url).identity(Some("Absent")),
        Err(HttpError::Unavailable(_))
    ));
    server.join().unwrap();
}

#[test]
fn chunked_streams_and_errors() {
    let events = "data: {\"n\": 1}\n\ndata: {\"n\": 2}\n\ndata: [DONE]\n\n";
    let chunked: String = events
        .as_bytes()
        .chunks(7)
        .map(|chunk| {
            format!(
                "{:x}\r\n{}\r\n",
                chunk.len(),
                String::from_utf8_lossy(chunk)
            )
        })
        .collect();
    let streamed = format!(
        "HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\nx-chopinfer-startup-nonce: {NONCE}\r\n\r\n{chunked}0\r\n\r\n"
    );
    let refused = identified("400 Bad Request", "{\"error\": \"no\"}", CHECKPOINT, NONCE);
    let (url, server) = serve(vec![streamed, refused]);
    let mut client = connect(&url);
    let (events, _) = client
        .stream("/v1/completions", &json!({"stream": true}))
        .unwrap();
    let values: Vec<_> = events.into_iter().map(|(_, value)| value).collect();
    assert_eq!(values, [json!({"n": 1}), json!({"n": 2})]);
    match client.json("/v1/completions", Some(&json!({}))) {
        Err(HttpError::Status { status: 400, body }) => assert!(body.contains("no")),
        other => panic!("{other:?}"),
    }
    server.join().unwrap();
    assert!(client.responders().contains(&(None, Some(NONCE.into()))));
}

#[test]
fn an_unreachable_server_is_a_transport_error() {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut client = Client::new(
        &format!("http://127.0.0.1:{port}"),
        Duration::from_secs(2),
        0,
    )
    .unwrap();
    assert!(matches!(
        client.json("/health", None),
        Err(HttpError::Transport(_))
    ));
}

#[test]
fn retry_backoff_cannot_outlive_the_check_budget() {
    use std::time::Instant;
    let (url, server) = serve(vec![answer("503 Service Unavailable", "busy", &[])]);
    let mut client = Client::new(&url, Duration::from_secs(5), 10).unwrap();
    let started = Instant::now();
    client.set_deadline(Some(started + Duration::from_millis(100)));
    let result = client.request("GET", "/test", None);
    assert!(matches!(result, Err(HttpError::Transport(_))), "{result:?}");
    assert!(started.elapsed() < Duration::from_millis(400));
    assert_eq!(server.join().unwrap().len(), 1);
}

#[test]
fn model_options_refuse_unqualified_or_incomplete_identity() {
    let (url, server) = serve(healthy());
    let mut client = connect(&url);
    let identity = client.identity(None).unwrap();
    server.join().unwrap();
    for marker in [json!(false), json!(null), json!("true")] {
        let mut changed = identity.clone();
        let serving = &mut changed.health.as_mut().unwrap()["serving"];
        serving["model_options"] = json!({"budget": 8});
        serving["model_options_qualified"] = marker;
        assert!(client.bind(&changed, &pin()).is_err());
    }
    let mut defaults = identity.clone();
    defaults.health.as_mut().unwrap()["serving"]["model_options"] = json!({"budget": 8});
    defaults.health.as_mut().unwrap()["serving"]["model_options_qualified"] = json!(true);
    assert!(client.bind(&defaults, &pin()).is_ok());
    assert!(client.bind(&identity, &pin()).is_ok());
}

/// A startup log as the host writes it: the checkpoint records, then the
/// listening record of `address`.
fn startup_log(identity: &str, nonce: &str, address: &str) -> String {
    format!(
        "startup direct_package: {}\nstartup serving_identity: {}\nstartup direct_load: {}\n{}\n",
        json!({"package_identity": "p", "checkpoint_identity": identity}),
        json!({"checkpoint_identity": identity, "startup_nonce": nonce}),
        json!({"checkpoint_identity": identity}),
        json!({"status": "listening", "address": address})
    )
}

/// `common.served_checkpoint` of `log` and `url`: its result, or its
/// refusal's message; `None` when python3 is missing.
fn python_served_checkpoint(log: &std::path::Path, url: &str) -> Option<Value> {
    let tools = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/qual");
    // The log as a tool's `type=Path` argument hands it over.
    let script = "import json, sys\nfrom pathlib import Path\nimport common\n\
                  try:\n    print(json.dumps({'ok': common.served_checkpoint(Path(sys.argv[1]), sys.argv[2])}))\n\
                  except ValueError as error:\n    print(json.dumps({'error': str(error)}))\n";
    let output = match std::process::Command::new("python3")
        .arg("-B")
        .arg("-c")
        .arg(script)
        .arg(log)
        .arg(url)
        .env("PYTHONPATH", tools)
        .env_remove("CHOPINFER_QUAL_CONFIG")
        .output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("skipped: python3 is missing");
            return None;
        }
        Err(error) => panic!("python3: {error}"),
    };
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(serde_json::from_slice(&output.stdout).unwrap())
}

#[test]
fn a_startup_log_binds_as_common_reads_it() {
    use runtime_eval::http::served_checkpoint;
    let directory = std::env::temp_dir().join(format!("runtime-eval-log-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let url = "http://127.0.0.1:8101";
    let good = startup_log(CHECKPOINT, NONCE, "127.0.0.1:8101");
    let other_record = format!(
        "startup weight_plan: {}\n",
        json!({"checkpoint_identity": OTHER})
    );
    let second_nonce = format!(
        "startup serving_identity: {}\n",
        json!({"checkpoint_identity": CHECKPOINT, "startup_nonce": OTHER_NONCE})
    );
    let cases: Vec<(&str, String, bool)> = vec![
        ("good", good.clone(), true),
        ("crlf", good.replace('\n', "\r\n"), true),
        (
            "other-address",
            startup_log(CHECKPOINT, NONCE, "127.0.0.1:8102"),
            false,
        ),
        (
            "no-listening",
            good.lines().take(3).collect::<Vec<_>>().join("\n"),
            false,
        ),
        (
            "later-listening",
            format!(
                "{good}{}\n",
                json!({"status": "listening", "address": "127.0.0.1:9"})
            ),
            false,
        ),
        ("two-checkpoints", format!("{other_record}{good}"), false),
        ("two-nonces", format!("{second_nonce}{good}"), false),
        (
            "bad-nonce",
            startup_log(CHECKPOINT, "xyz", "127.0.0.1:8101"),
            false,
        ),
        (
            "no-records",
            format!(
                "{}\n",
                json!({"status": "listening", "address": "127.0.0.1:8101"})
            ),
            false,
        ),
        (
            "unrelated-lines",
            format!("startup other: {{}}\nnoise\n{good}"),
            true,
        ),
    ];
    // Rust and Python read `log` alike: the same result or refusal.
    let same = |name: &str, log: &std::path::Path, text: &str, binds: bool| {
        let rust = served_checkpoint(log, url);
        assert_eq!(rust.is_ok(), binds, "{name}: {rust:?}");
        match (&rust, python_served_checkpoint(log, url)) {
            (_, None) => {}
            (Ok(logged), Some(python)) => {
                assert_eq!(logged.checkpoint_identity, CHECKPOINT, "{name}");
                assert_eq!(logged.startup_nonce, NONCE, "{name}");
                assert_eq!(
                    logged.log.sha256,
                    runtime_eval::json::sha256_hex(text.as_bytes())
                );
                assert_eq!(
                    python["ok"],
                    json!({"checkpoint_identity": logged.checkpoint_identity,
                           "startup_nonce": logged.startup_nonce,
                           "server_log": logged.log.path,
                           "server_log_sha256": logged.log.sha256}),
                    "{name}"
                );
            }
            (Err(error), Some(python)) => {
                assert_eq!(python["error"], error.to_string(), "{name}")
            }
        }
        rust.ok().map(|logged| logged.log.path)
    };
    let texts: Vec<(String, String)> = cases
        .iter()
        .map(|(name, text, _)| (name.to_string(), text.clone()))
        .collect();
    for (name, text, binds) in cases {
        let log = directory.join(format!("{name}.log"));
        std::fs::write(&log, &text).unwrap();
        same(name, &log, &text, binds);
    }
    // Relative and unnormalized arguments name the same files from this
    // working directory; both record them as Python's Path names them.
    let ups = "../".repeat(std::env::current_dir().unwrap().components().count() - 1);
    let below = directory.strip_prefix("/").unwrap().to_str().unwrap();
    let text = |name: &str| &texts.iter().find(|(seen, _)| seen == name).unwrap().1;
    for (argument, name, shown) in [
        (
            format!("./{ups}{below}/good.log"),
            "good",
            Some(format!("{ups}{below}/good.log")),
        ),
        (
            format!("{ups}.//{below}/./good.log"),
            "good",
            Some(format!("{ups}{below}/good.log")),
        ),
        (
            format!("/{}//good.log", directory.display()),
            "good",
            Some(format!("/{}/good.log", directory.display())),
        ),
        (
            format!("./{ups}{below}/other-address.log"),
            "other-address",
            None,
        ),
    ] {
        let log = std::path::Path::new(&argument);
        let recorded = same(&argument, log, text(name), shown.is_some());
        assert_eq!(recorded, shown, "{argument}");
    }

    // Bound to /health's identity, the log must name the same process.
    let log = directory.join("good.log");
    let (served, server) = serve(healthy());
    let mut client = connect(&served);
    let bound = bind(&mut client);
    server.join().unwrap();
    let address = served.trim_start_matches("http://");
    std::fs::write(&log, startup_log(CHECKPOINT, NONCE, address)).unwrap();
    let logged = bound.clone().with_log(&log, &served).unwrap();
    let evidence = logged.to_json();
    assert_eq!(evidence["server_log"], log.to_str().unwrap());
    assert_eq!(
        evidence["server_log_sha256"],
        runtime_eval::json::sha256_hex(&std::fs::read(&log).unwrap())
    );
    let keys: Vec<&String> = evidence.as_object().unwrap().keys().collect();
    assert_eq!(
        keys,
        [
            "checkpoint_identity",
            "package_identity",
            "metadata_identity",
            "startup_nonce",
            "server_log",
            "server_log_sha256"
        ]
    );
    for (identity, nonce) in [(CHECKPOINT, OTHER_NONCE), (OTHER, NONCE)] {
        std::fs::write(&log, startup_log(identity, nonce, address)).unwrap();
        assert!(
            matches!(bound.clone().with_log(&log, &served), Err(BindError::Mismatch(message))
                     if message.contains("is not the log of the live server")),
        );
    }
    assert!(matches!(
        bound.with_log(&directory.join("missing.log"), &served),
        Err(BindError::Unidentified(_))
    ));
    std::fs::remove_dir_all(directory).unwrap();
}
