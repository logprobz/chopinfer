//! `eval list | run | verify-references` on a synthetic root and a fake
//! server, and `verify-references` on the shipped declarations.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

use runtime_eval::cli;
use runtime_eval::json::{canonical_sha256, sha256_hex};
use runtime_eval::{Exit, Tool};
use serde_json::{Value, json};

const CHECKPOINT: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const OTHER_CHECKPOINT: &str = "6666666666666666666666666666666666666666666666666666666666666666";
const NONCE: &str = "0123456789abcdef0123456789abcdef";

fn tool() -> Tool {
    Tool {
        name: "chopinfer eval".into(),
        version: "test".into(),
        revision: None,
    }
}

fn run(arguments: &[&str]) -> Exit {
    let arguments: Vec<String> = arguments.iter().map(|word| (*word).to_owned()).collect();
    cli::run(&tool(), &arguments)
}

fn rule() -> Value {
    json!({"path": ["passed"], "op": "eq", "value": true, "source": "docs/example.md"})
}

/// A root with one model whose teacher reference and lock verify.
fn synthetic_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("runtime-eval-cli-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let model = root.join("models/example");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::create_dir_all(root.join("fixtures/example")).unwrap();
    std::fs::create_dir_all(root.join("tools/qual")).unwrap();
    let modules = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/qual/modules.json");
    std::fs::copy(modules, root.join("tools/qual/modules.json")).unwrap();
    let lock = json!({"model.safetensors": {"sha256": "ab", "size": 1}});
    std::fs::write(model.join("checkpoint-files.json"), lock.to_string()).unwrap();
    let checkpoint = json!({"repository": "org/example", "revision": "abc",
                            "served_model_name": "Example", "source_lock_sha256": canonical_sha256(&lock),
                            "checkpoint_identity": CHECKPOINT});
    let reference = json!({"schema": "example.teacher.v1",
                           "provenance": {"checkpoint": checkpoint, "variants": {"exact": "f64", "bf16": "bf16"}},
                           "texts": [{"name": "t0", "category": "prose", "prompt_token_ids": [1, 2, 3],
                                      "score_from": 1, "exact": {"logprob": [-0.5, -0.25]},
                                      "bf16": {"logprob": [-0.5, -0.3]}}]});
    let bytes = serde_json::to_vec(&reference).unwrap();
    std::fs::write(root.join("fixtures/example/teacher.json"), &bytes).unwrap();
    let mut pin = checkpoint.clone();
    pin["lock"] = json!("models/example/checkpoint-files.json");
    let declaration = json!({
        "schema": "chopinfer-qual-model/v1", "model": "example",
        "cohorts": {"quick": {"checks": ["teacher", "api", "soak"]},
                    "full": {"checks": ["teacher", "api", "structured", "soak"]}},
        "change_classes": {"frontend": [], "quality": [], "service": [], "speculation": [], "prefix": []},
        "checks": {
            "teacher": {"module": "teacher", "args": [], "timeout_seconds": 60, "receipt": "*-teacher.json",
                        "rules": [rule()]},
            "api": {"module": "api", "args": [], "timeout_seconds": 60, "receipt": "*-api.json", "rules": [rule()]},
            "structured": {"module": "structured", "args": [], "timeout_seconds": 60, "receipt": "*-s.json",
                           "rules": [rule()], "informational_reason": "not yet"},
            "soak": {"module": "soak", "args": [], "timeout_seconds": 60, "receipt": "*-soak.json",
                     "rules": [rule()]}
        },
        "references": {"teacher": {"path": "fixtures/example/teacher.json", "sha256": sha256_hex(&bytes),
                                   "schema": "example.teacher.v1", "variants": ["exact", "bf16"]}},
        "inputs": {},
        "qualification": {"teacher_reference_kind": "exact", "teacher_receipt": "teacher",
                          "teacher_capture_schema": "example.capture.v1", "verify_buckets": [2],
                          "chars_per_token": 4.0, "checkpoint": pin},
        "receipts": {"env": "RUNTIME_EVAL_TEST_UNSET", "path": "qual/example"}
    });
    std::fs::write(model.join("qual.json"), declaration.to_string()).unwrap();
    root
}

fn respond(body: &Value) -> String {
    let body = body.to_string();
    format!(
        "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nx-chopinfer-checkpoint-identity: {CHECKPOINT}\r\n\
         x-chopinfer-startup-nonce: {NONCE}\r\n\r\n{body}",
        body.len()
    )
}

fn serve(responses: Vec<String>) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        for response in responses {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap() > 2 {
                line.clear();
            }
            // An empty response drops the connection unanswered.
            let mut stream = reader.into_inner();
            if !response.is_empty() {
                stream.write_all(response.as_bytes()).unwrap();
                let _ = stream.read(&mut [0; 1]);
            }
        }
    });
    (url, handle)
}

fn identity(checkpoint: &str) -> Vec<String> {
    vec![
        respond(&json!({"data": [{"id": "Example"}]})),
        respond(
            &json!({"status": "ready", "serving": {"checkpoint_identity": checkpoint,
                                                       "startup_nonce": NONCE}}),
        ),
    ]
}

/// The identity reads of a run bound to the pin: binding, then confirming.
fn bound_run() -> Vec<String> {
    [identity(CHECKPOINT), identity(CHECKPOINT)].concat()
}

/// Runs `eval run` against `url` for the synthetic model with `arguments`,
/// writing the report to `ROOT/<name>`; its exit and report.
fn run_report(root: &Path, url: &str, name: &str, arguments: &[&str]) -> (Exit, Value) {
    let model = root.join("models/example");
    let directory = root.join(name);
    let mut words = vec![
        "run",
        "--model",
        model.to_str().unwrap(),
        "--url",
        url,
        "--report",
        directory.to_str().unwrap(),
    ];
    words.extend(arguments);
    let exit = run(&words);
    (exit, report(&directory))
}

fn report(directory: &Path) -> Value {
    let bytes = std::fs::read(directory.join("report.json")).unwrap();
    runtime_eval::json::parse(&bytes).unwrap()
}

#[test]
fn usage_and_declaration_errors_exit_2() {
    let root = synthetic_root("usage");
    let model = root.join("models/example");
    let model = model.to_str().unwrap();
    assert_eq!(run(&[]), Exit::Usage);
    assert_eq!(run(&["frobnicate", "--model", model]), Exit::Usage);
    assert_eq!(run(&["list"]), Exit::Usage);
    assert_eq!(run(&["list", "--model", model, "--launch"]), Exit::Usage);
    assert_eq!(run(&["run", "--model", model]), Exit::Usage);
    assert_eq!(
        run(&[
            "run",
            "--model",
            model,
            "--url",
            "http://127.0.0.1:9",
            "--suite",
            "full"
        ]),
        Exit::Usage
    );
    assert_eq!(
        run(&[
            "run",
            "--model",
            model,
            "--url",
            "http://127.0.0.1:9",
            "--check",
            "soak"
        ]),
        Exit::Usage
    );
    assert_eq!(
        run(&[
            "run",
            "--model",
            model,
            "--url",
            "http://127.0.0.1:9",
            "--check",
            "absent"
        ]),
        Exit::Usage
    );
    // --cohort names the cohort of checks named with --check only.
    let url = "http://127.0.0.1:9";
    for words in [
        &["--cohort", "full"][..],
        &["--suite", "accuracy", "--cohort", "full"],
        &["--check", "teacher", "--cohort", "accuracy"],
        &["--check", "teacher", "--cohort", "full", "--cohort", "full"],
        &["--check", "teacher", "--server-log"],
    ] {
        let mut arguments = vec!["run", "--model", model, "--url", url];
        arguments.extend(words);
        assert_eq!(run(&arguments), Exit::Usage, "{words:?}");
    }
    // The receipts directory needs its variable: --report is required.
    assert_eq!(
        run(&["run", "--model", model, "--url", "http://127.0.0.1:9"]),
        Exit::Usage
    );
    let declaration = root.join("models/example/qual.json");
    let mut value: Value = serde_json::from_slice(&std::fs::read(&declaration).unwrap()).unwrap();
    value["colour"] = json!("red");
    std::fs::write(&declaration, value.to_string()).unwrap();
    assert_eq!(run(&["list", "--model", model]), Exit::Usage);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn listing_and_verifying_references() {
    let root = synthetic_root("verify");
    let model = root.join("models/example");
    let model = model.to_str().unwrap();
    assert_eq!(run(&["list", "--model", model, "--json"]), Exit::Pass);
    assert_eq!(run(&["verify-references", "--model", model]), Exit::Pass);
    std::fs::write(root.join("fixtures/example/teacher.json"), b"{}").unwrap();
    assert_eq!(
        run(&["verify-references", "--model", model, "--json"]),
        Exit::Insufficient
    );
    assert_eq!(run(&["list", "--model", model]), Exit::Pass);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn external_reference_paths_preserve_digest_and_missing_data_checks() {
    let root = synthetic_root("external-reference");
    let model = root.join("models/example");
    let path = model.join("qual.json");
    let mut value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let external = root.with_extension("reference-data");
    std::fs::create_dir_all(&external).unwrap();
    let reference = external.join("teacher.json");
    std::fs::rename(root.join("fixtures/example/teacher.json"), &reference).unwrap();
    value["references"]["teacher"]["path"] = json!({
        "env": "CHOPINFER_TEST_EXTERNAL_REFERENCE", "default": external, "path": "teacher.json"
    });
    std::fs::write(&path, value.to_string()).unwrap();
    let loaded = runtime_eval::declaration::load(&model).unwrap();
    let listed = runtime_eval::files::files(&loaded).unwrap();
    assert!(!listed.iter().any(|file| file.ends_with("teacher.json")));
    let model = model.to_str().unwrap();
    assert_eq!(
        run(&["verify-references", "--model", model, "--json"]),
        Exit::Pass
    );
    let bytes = std::fs::read(&reference).unwrap();
    std::fs::write(&reference, b"{}").unwrap();
    assert_eq!(
        run(&["verify-references", "--model", model]),
        Exit::Insufficient
    );
    std::fs::write(&reference, bytes).unwrap();
    value["references"]["teacher"]["path"]
        .as_object_mut()
        .unwrap()
        .remove("default");
    std::fs::write(&path, value.to_string()).unwrap();
    assert_eq!(
        run(&["verify-references", "--model", model]),
        Exit::Insufficient
    );
    std::fs::remove_dir_all(root).unwrap();
    std::fs::remove_dir_all(external).unwrap();
}

#[test]
fn floor_only_capture_requires_an_authenticated_binding() {
    use runtime_eval::kind::{Kind, KindError};
    let root = synthetic_root("floor-only-unbound");
    let model = root.join("models/example");
    let path = model.join("qual.json");
    let mut value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["checks"]["teacher"]["constants"] = json!({"teacher": {
        "ALLOW_FLOOR_ONLY": {"value": true, "source": "test"}
    }});
    std::fs::write(&path, value.to_string()).unwrap();
    let loaded = runtime_eval::declaration::load(&model).unwrap();
    let check = loaded.declaration.check("teacher").unwrap();
    let pin = loaded.declaration.reference("teacher").unwrap();
    let verified = runtime_eval::references::pinned(&loaded, pin, None).unwrap();
    let kind = runtime_eval::kinds::teacher::TeacherForced;
    assert!(
        matches!(kind.reference(check, Some(&verified)), Err(KindError::Insufficient(message))
        if message.contains("authenticated producer and workload binding"))
    );
    assert!(
        matches!(kind.reference(check, None), Err(KindError::Insufficient(message))
        if message.contains("pinned reference"))
    );
    value["checks"]["teacher"]["constants"]["teacher"]["ALLOW_FLOOR_ONLY"]["value"] = json!(1);
    std::fs::write(&path, value.to_string()).unwrap();
    assert!(runtime_eval::declaration::load(&model).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

/// Adds the variant `other` to a synthetic root: another checkpoint with
/// its own lock (kept outside the model directory's top level), reference
/// and reference lock.
fn add_variant(root: &Path) {
    let model = root.join("models/example");
    std::fs::create_dir_all(model.join("other")).unwrap();
    std::fs::create_dir_all(root.join("fixtures/example/other")).unwrap();
    let lock = json!({"model.safetensors": {"sha256": "cd", "size": 2}});
    std::fs::write(model.join("other/checkpoint-files.json"), lock.to_string()).unwrap();
    let mut value: Value =
        serde_json::from_slice(&std::fs::read(model.join("qual.json")).unwrap()).unwrap();
    let mut reference: Value =
        serde_json::from_slice(&std::fs::read(root.join("fixtures/example/teacher.json")).unwrap())
            .unwrap();
    let checkpoint = &mut reference["provenance"]["checkpoint"];
    checkpoint["repository"] = json!("org/other");
    checkpoint["checkpoint_identity"] = json!(OTHER_CHECKPOINT);
    checkpoint["source_lock_sha256"] = json!(canonical_sha256(&lock));
    let mut pin = checkpoint.clone();
    pin["lock"] = json!("models/example/other/checkpoint-files.json");
    let bytes = serde_json::to_vec(&reference).unwrap();
    std::fs::write(root.join("fixtures/example/other/teacher.json"), &bytes).unwrap();
    value["variant"] = json!("other");
    value["qualification"]["checkpoint"] = pin;
    value["references"]["teacher"]["path"] = json!("fixtures/example/other/teacher.json");
    value["references"]["teacher"]["sha256"] = json!(sha256_hex(&bytes));
    std::fs::write(model.join("qual.other.json"), value.to_string()).unwrap();
    let references = json!({"schema": "chopinfer-qual-references/v1", "model": "example",
                            "entries": {}, "objects": {}});
    std::fs::write(
        model.join("qual-references.other.lock.json"),
        references.to_string(),
    )
    .unwrap();
}

#[test]
fn a_variant_is_selected_by_name() {
    let root = synthetic_root("variant");
    add_variant(&root);
    let model = root.join("models/example");
    let model = model.to_str().unwrap();
    for command in ["list", "verify-references", "files"] {
        assert_eq!(
            run(&[command, "--model", model, "--variant", "other"]),
            Exit::Pass,
            "{command}"
        );
        let root = root.to_str().unwrap();
        assert_eq!(
            run(&[command, "--package", root, "--variant", "other"]),
            Exit::Pass,
            "{command}"
        );
    }
    let loaded = runtime_eval::declaration::load_variant(Path::new(model), Some("other")).unwrap();
    assert_eq!(loaded.declaration.variant.as_deref(), Some("other"));
    assert_eq!(
        runtime_eval::files::files(&loaded).unwrap(),
        [
            "fixtures/example/other/teacher.json",
            "models/example/other/checkpoint-files.json",
            "models/example/qual-references.other.lock.json",
            "models/example/qual.other.json",
            "tools/qual/modules.json",
        ]
    );
    // A run reads the variant's checkpoint pin: a server of the base
    // checkpoint is refused before any check.
    let (url, server) = serve(identity(CHECKPOINT));
    let (exit, report) = run_report(
        &root,
        &url,
        "variant-run",
        &["--variant", "other", "--check", "api"],
    );
    server.join().unwrap();
    assert_eq!(exit, Exit::Fail);
    assert_eq!(report["checks"][0]["status"], "fail");
    let declared = std::fs::read(root.join("models/example/qual.other.json")).unwrap();
    assert_eq!(report["declaration_sha256"], json!(sha256_hex(&declared)));
    for refused in [
        &["list", "--model", model, "--variant", "absent"][..],
        &["list", "--model", model, "--variant", "../example/qual"],
        &[
            "list",
            "--model",
            model,
            "--variant",
            "other",
            "--variant",
            "other",
        ],
        &["list", "--model", model, "--variant"],
    ] {
        assert_eq!(run(refused), Exit::Usage, "{refused:?}");
    }
    // The variant's declaration must name it.
    let path = root.join("models/example/qual.other.json");
    let mut value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["variant"] = json!("another");
    std::fs::write(&path, value.to_string()).unwrap();
    assert_eq!(
        run(&["list", "--model", model, "--variant", "other"]),
        Exit::Usage
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn the_shipped_references_verify() {
    use runtime_eval::declaration::{self, Input};
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut variants = 0;
    for entry in std::fs::read_dir(models).unwrap() {
        let directory = entry.unwrap().path();
        if !directory.join("qual.json").is_file() {
            continue;
        }
        let named = declaration::variants(&directory).unwrap();
        variants += named.len();
        for variant in std::iter::once(None).chain(named.iter().map(|name| Some(name.as_str()))) {
            // A workload kept outside the repository is read only where its
            // variable is set; without it the reference is not evidence.
            let declaration = declaration::load_variant(&directory, variant)
                .unwrap()
                .declaration;
            let unset = declaration.references.iter().any(|pin| {
                matches!(&pin.workload, Some(Input::Environment { env, .. })
                         if std::env::var_os(env).is_none())
            });
            let mut arguments = vec!["verify-references", "--model", directory.to_str().unwrap()];
            arguments.extend(
                variant
                    .map(|variant| ["--variant", variant])
                    .into_iter()
                    .flatten(),
            );
            assert_eq!(
                run(&arguments),
                if unset {
                    Exit::Insufficient
                } else {
                    Exit::Pass
                },
                "{arguments:?}"
            );
        }
    }
    assert!(variants > 0, "no shipped model declares a variant");
}

#[test]
fn a_run_reports_every_selected_check() {
    let root = synthetic_root("run");
    let model = root.join("models/example");
    let (url, server) = serve(bound_run());
    let directory = root.join("report-accuracy");
    let exit = run(&[
        "run",
        "--model",
        model.to_str().unwrap(),
        "--url",
        &url,
        "--suite",
        "accuracy",
        "--report",
        directory.to_str().unwrap(),
    ]);
    server.join().unwrap();
    // The synthetic model declares no frozen teacher texts and none of the
    // api check's inputs: the run has insufficient evidence.
    assert_eq!(exit, Exit::Insufficient);
    let value = report(&directory);
    assert_eq!(value["schema"], "chopinfer.qual-report.v1");
    assert_eq!(value["model"]["checkpoint_identity"], CHECKPOINT);
    assert_eq!(value["exit"], 3);
    let statuses: Vec<(&str, &str)> = value["checks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|check| {
            (
                check["id"].as_str().unwrap(),
                check["status"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        statuses,
        [
            ("teacher", "insufficient"),
            ("api", "insufficient"),
            ("structured", "informational")
        ]
    );
    assert!(directory.join("summary.md").is_file() && directory.join("records").is_dir());
    // A report directory is never reused.
    assert_eq!(
        run(&[
            "run",
            "--model",
            model.to_str().unwrap(),
            "--url",
            &url,
            "--report",
            directory.to_str().unwrap()
        ]),
        Exit::Usage
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_run_against_another_checkpoint_fails() {
    let root = synthetic_root("mismatch");
    let model = root.join("models/example");
    let other = "5555555555555555555555555555555555555555555555555555555555555555";
    let (url, server) = serve(identity(other));
    let directory = root.join("report");
    let exit = run(&[
        "run",
        "--model",
        model.to_str().unwrap(),
        "--url",
        &url,
        "--check",
        "api",
        "--report",
        directory.to_str().unwrap(),
        "--json",
    ]);
    server.join().unwrap();
    assert_eq!(exit, Exit::Fail);
    assert_eq!(report(&directory)["checks"][0]["status"], "fail");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_missing_reference_is_insufficient_and_an_unreachable_server_is_environmental() {
    let root = synthetic_root("environment");
    let model = root.join("models/example");
    std::fs::remove_file(root.join("fixtures/example/teacher.json")).unwrap();
    let (url, server) = serve(bound_run());
    let directory = root.join("report");
    let exit = run(&[
        "run",
        "--model",
        model.to_str().unwrap(),
        "--url",
        &url,
        "--check",
        "teacher",
        "--report",
        directory.to_str().unwrap(),
    ]);
    server.join().unwrap();
    assert_eq!(exit, Exit::Insufficient);
    assert_eq!(report(&directory)["checks"][0]["status"], "insufficient");

    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let directory = root.join("unreachable");
    let exit = run(&[
        "run",
        "--model",
        model.to_str().unwrap(),
        "--url",
        &format!("http://127.0.0.1:{port}"),
        "--check",
        "api",
        "--timeout",
        "2",
        "--report",
        directory.to_str().unwrap(),
    ]);
    assert_eq!(exit, Exit::Environment);
    assert_eq!(report(&directory)["exit"], 4);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn identity_failures_fail_the_run_whatever_its_checks() {
    let root = synthetic_root("identity");
    let other = "5555555555555555555555555555555555555555555555555555555555555555";
    // An informational-only selection against another checkpoint fails.
    let (url, server) = serve(identity(other));
    let (exit, value) = run_report(&root, &url, "informational", &["--check", "structured"]);
    server.join().unwrap();
    assert_eq!(exit, Exit::Fail);
    assert_eq!(
        (value["passed"].as_bool(), value["exit"].as_u64()),
        (Some(false), Some(1))
    );
    assert_eq!(value["checks"][0]["status"], "informational");
    assert_eq!(value["failure"]["exit"], 1);

    // A server that names no identity leaves the run without evidence.
    let unidentified = vec![
        respond(&json!({"data": [{"id": "Example"}]})),
        respond(&json!({"status": "ready"})),
    ];
    let (url, server) = serve(unidentified);
    let (exit, _) = run_report(&root, &url, "unidentified", &["--check", "structured"]);
    server.join().unwrap();
    assert_eq!(exit, Exit::Insufficient);

    // A server that restarts between binding and confirming fails the run.
    let restarted = [
        identity(CHECKPOINT),
        vec![
            respond(&json!({"data": [{"id": "Example"}]})),
            respond(&json!({"status": "ready", "serving": {"checkpoint_identity": CHECKPOINT,
                                                          "startup_nonce": "ffffffffffffffffffffffffffffffff"}})),
        ],
    ]
    .concat();
    let (url, server) = serve(restarted);
    let (exit, value) = run_report(&root, &url, "restarted", &["--check", "structured"]);
    server.join().unwrap();
    assert_eq!(exit, Exit::Fail);
    assert!(
        value["failure"]["message"]
            .as_str()
            .unwrap()
            .contains("changed while scoring")
    );

    // A model the server does not serve fails it too.
    let (url, server) = serve(vec![respond(&json!({"data": [{"id": "Example"}]}))]);
    let (exit, _) = run_report(
        &root,
        &url,
        "unserved",
        &["--served-model", "Absent", "--check", "api"],
    );
    server.join().unwrap();
    assert_eq!(exit, Exit::Fail);

    // An empty selection is not evidence.
    let declaration = root.join("models/example/qual.json");
    let mut value: Value = serde_json::from_slice(&std::fs::read(&declaration).unwrap()).unwrap();
    value["cohorts"]["quick"]["checks"] = json!(["soak"]);
    std::fs::write(&declaration, value.to_string()).unwrap();
    let (url, server) = serve(bound_run());
    let (exit, value) = run_report(&root, &url, "empty", &[]);
    server.join().unwrap();
    assert_eq!(exit, Exit::Insufficient);
    assert_eq!(value["checks"], json!([]));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_health_transport_failure_is_environmental() {
    let root = synthetic_root("health");
    // /v1/models answers, then /health drops and the server is gone.
    let (url, server) = serve(vec![
        respond(&json!({"data": [{"id": "Example"}]})),
        String::new(),
    ]);
    let (exit, value) = run_report(
        &root,
        &url,
        "dropped",
        &["--check", "api", "--timeout", "2"],
    );
    server.join().unwrap();
    assert_eq!(exit, Exit::Environment);
    assert_eq!(value["failure"]["exit"], 4);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn timeouts_out_of_range_are_usage_errors() {
    let root = synthetic_root("timeout");
    let model = root.join("models/example");
    let model = model.to_str().unwrap();
    for timeout in ["1e300", "0", "1e-12", "-1", "nan", "inf"] {
        assert_eq!(
            run(&[
                "run",
                "--model",
                model,
                "--url",
                "http://127.0.0.1:9",
                "--timeout",
                timeout
            ]),
            Exit::Usage,
            "{timeout}"
        );
    }
    let declaration = root.join("models/example/qual.json");
    let mut value: Value = serde_json::from_slice(&std::fs::read(&declaration).unwrap()).unwrap();
    value["checks"]["api"]["timeout_seconds"] = json!(1e300);
    std::fs::write(&declaration, value.to_string()).unwrap();
    let report = root.join("report");
    assert_eq!(
        run(&[
            "run",
            "--model",
            model,
            "--url",
            "http://127.0.0.1:9",
            "--check",
            "api",
            "--report",
            report.to_str().unwrap()
        ]),
        Exit::Usage
    );
    assert!(!report.exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_package_or_an_install_root_names_its_model() {
    let root = synthetic_root("package");
    let model = root.join("models/example");
    let profiles = json!({"profiles": {"tp1": {"bundle_id": "example.v1"}}});
    std::fs::write(model.join("profiles.json"), profiles.to_string()).unwrap();
    let package = root.join("packages/example");
    std::fs::create_dir_all(&package).unwrap();
    std::fs::write(package.join("manifest.json"), r#"{"id": "example.v1"}"#).unwrap();
    for directory in [&root, &package] {
        let directory = directory.to_str().unwrap();
        assert_eq!(run(&["list", "--package", directory]), Exit::Pass);
        assert_eq!(
            run(&["verify-references", "--package", directory]),
            Exit::Pass
        );
    }
    let (model, package) = (model.to_str().unwrap(), package.to_str().unwrap());
    assert_eq!(
        run(&["list", "--model", model, "--package", package]),
        Exit::Usage
    );
    let manifest = root.join("packages/example/manifest.json");
    std::fs::write(manifest, r#"{"id": "other.v1"}"#).unwrap();
    assert_eq!(run(&["list", "--package", package]), Exit::Usage);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn declaration_validates_a_model_and_refuses_with_exit_2() {
    let root = synthetic_root("declaration");
    let model = root.join("models/example");
    let model = model.to_str().unwrap();
    assert_eq!(run(&["declaration", "--model", model]), Exit::Pass);
    assert_eq!(
        run(&["declaration", "--model", model, "--variant", "absent"]),
        Exit::Usage
    );
    assert_eq!(
        run(&["declaration", "--model", model, "--root", model]),
        Exit::Usage
    );
    assert_eq!(run(&["declaration", "--json"]), Exit::Usage);
    let path = root.join("models/example/qual.json");
    let mut value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["cohorts"]["quick"]["checks"] = json!(["undeclared"]);
    std::fs::write(&path, value.to_string()).unwrap();
    assert_eq!(run(&["declaration", "--model", model]), Exit::Usage);
}
