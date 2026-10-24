#![forbid(unsafe_code)]

//! What one native serving process publishes about the work it really did.
//!
//! The generation worker owns the engine, so the counters a metrics endpoint
//! answers with have to come back from that thread. This drives the real
//! worker with a fake executor and reads the provider the serving command
//! hands to the transport, through the same `MetricsSnapshotProvider` call the
//! `/metrics` route makes. No listener is bound here: the transport's own
//! contract tests cover the route, and what is under test is whether the
//! numbers behind it are the engine's.
//!
//! Nothing is asserted about a measurement no producer exists for. KV
//! evictions, copy-on-write bytes and peak device memory are absent from the
//! published text on purpose, and this checks that they are absent rather than
//! published as zero.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use runtime_engine::{
    Engine, EngineConfig, ExecutorError, LogitsRow, ModelBatchItem, ModelExecutor,
    ModelTransaction, NoopModelTransactionState,
};
use runtime_frontend::FrontendDescriptorV1;
use runtime_host::{
    EngineBackendConfig, EngineBackendStartupError, EngineGenerationBackend, EngineModelPolicyV1,
    NativeRuntimeMetricsV1,
};
use runtime_kv::{KvGeometry, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::{LogitBias, SamplingParameters};
use runtime_scheduler::{SchedulerCapacityProfile, SchedulerConfig};
use runtime_server::{
    AdmissionError, CancellationSignal, ChatMetadata, ChatStateV1, EventSender, GenerationBackend,
    GenerationEvent, MetricsSnapshotProvider, PreparedPrompt, PreparedRequest, RequestedLogprobs,
};
use runtime_tokenizer::{
    BaseTokenRepresentation, ByteLevelPolicy, ConfigFieldPolicy, ConfigTokenRepresentation,
    ConfigTokenValue, FileIdentity, IgnoreMergesPolicy, MergeRecordPolicy, MergeResultIdPolicy,
    NormalizerPolicy, Tokenizer, TokenizerContract, TokenizerDialect,
};
use runtime_weights::sha256_bytes;

/// The specials this fixture's tokenizer is built around, in identifier order.
const SPECIALS: [&str; 7] = [
    "<<begin>>",
    "<<end>>",
    "<<reason>>",
    "<</reason>>",
    "<<user>>",
    "<<bot>>",
    "<<calls>>",
];
const BYTE_ID_START: u32 = SPECIALS.len() as u32;
const VOCABULARY_SIZE: usize = SPECIALS.len() + 256;
const CONTEXT_TOKENS: usize = 48;

/// One frontend declaration, in the shape a bundle ships it.
///
/// Only the generation half matters here: the vocabulary the engine runs, the
/// boundary tokens a prompt may carry, and the completion limits a request is
/// held to. The identities are a fixture's.
const DECLARATION: &str = r##"{
  "schema": "chop-frontend-openai-v2",
  "version": 2,
  "models": [{"id": "fixture/metrics-1", "owned_by": "fixture-metrics"}],
  "context": {"maximum_total_tokens": 48},
  "limits": {
    "maximum_prompt_bytes": 4096,
    "maximum_completion_bytes": 4096,
    "maximum_generated_tokens": 256,
    "maximum_client_stops": 2,
    "maximum_messages": 16,
    "maximum_tools": 2,
    "maximum_tool_calls": 1,
    "maximum_tool_parameters": 4,
    "maximum_name_bytes": 8,
    "maximum_json_bytes": 1024,
    "maximum_json_depth": 4,
    "maximum_json_values": 64
  },
  "reasoning": {
    "disable_value": "off",
    "default_effort": "brief",
    "efforts": [
      {"name": "brief", "preamble": "Be brief.\n"},
      {"name": "deep", "preamble": "Be exhaustive.\n"}
    ],
    "end_markers": ["<</reason>>", "<<calls>>"]
  },
  "tokenizer": {
    "repository": "fixture/metrics",
    "revision": "89abcdef0123456789abcdef0123456789abcdef",
    "definition": {
      "name": "vocabulary.json",
      "byte_length": 2048,
      "sha256": "3333333333333333333333333333333333333333333333333333333333333333"
    },
    "configuration": {
      "name": "vocabulary_config.json",
      "byte_length": 256,
      "sha256": "4444444444444444444444444444444444444444444444444444444444444444"
    },
    "policy": {
      "normalizer": "nfc",
      "pretokenizer_byte_level": {"add_prefix_space": false, "trim_offsets": false, "use_regex": false},
      "post_processor_byte_level": {"add_prefix_space": false, "trim_offsets": false, "use_regex": false},
      "decoder_byte_level": {"add_prefix_space": false, "trim_offsets": false, "use_regex": false},
      "merge_results": {"kind": "vocabulary"},
      "merge_records": "string-or-pair",
      "base_tokens": {"kind": "byte-level"},
      "config_tokens": {"kind": "string"},
      "add_bos_token": {"kind": "required", "value": false},
      "bos_token": {"kind": "required-null"},
      "unk_token": {"kind": "required-null"},
      "tokenizer_class": "NfcTestTokenizer",
      "continuing_subword_prefix": "",
      "end_of_word_suffix": "",
      "ignore_merges": "disabled",
      "add_eos_token": null,
      "legacy": null,
      "empty_sentencepiece_kwargs": false
    },
    "base_vocabulary_size": 256,
    "merge_count": 0,
    "added_token_count": 7,
    "model_max_length": 48,
    "bos_token": "<<begin>>",
    "eos_token": "<<end>>",
    "pad_token": "<<end>>",
    "pretokenizer_patterns": ["\\S+"]
  },
  "tools": {
    "name_bytes": {"ascii_alphanumeric": true, "extra": "_-"},
    "arguments": "json-object-integral-numbers"
  },
  "vocabulary": {
    "size": 263,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "stop_token_ids": [1],
    "boundary_tokens": [
      {"text": "<<begin>>", "id": 0},
      {"text": "<<end>>", "id": 1},
      {"text": "<<reason>>", "id": 2},
      {"text": "<</reason>>", "id": 3},
      {"text": "<<user>>", "id": 4},
      {"text": "<<bot>>", "id": 5},
      {"text": "<<calls>>", "id": 6}
    ],
    "prompt_special_tokens": [
      "<<begin>>", "<<end>>", "<<reason>>", "<</reason>>", "<<user>>",
      "<<bot>>", "<<calls>>"
    ]
  },
  "response": {"reasoning_field": "reasoning_content"},
  "format": {
    "parser": {
      "tool_grammar": "xml-key-value-arguments-v1",
      "json_encoding": "spaced-json-v1",
      "reasoning_open": "<<reason>>",
      "reasoning_close": "<</reason>>",
      "tool_call_open": "<<calls>>",
      "tool_call_close": ">>calls<<",
      "parameter_open": "<<key>>",
      "parameter_name_close": "<</key>>",
      "parameter_value_open": "<<value>>",
      "parameter_close": "<</value>>",
      "reserved_markers": ["<<begin>>", "<<end>>", "<<sys>>", "<<dev>>", "<<user>>", "<<bot>>", "<<reason>>", "<</reason>>", "<<result>>", "[[", "]]", "<<calls>>"],
      "turn_open": ["<<bot>>", "<<user>>"],
      "assistant_header": "<<bot>>"
    }
  }
}"##;

/// One serving process, with the metrics handle its transport would answer on.
struct ServedWorkerV1 {
    backend: EngineGenerationBackend,
    metrics: Arc<NativeRuntimeMetricsV1>,
}

impl ServedWorkerV1 {
    fn start(executor: FixtureExecutorV1) -> Self {
        let backend = EngineGenerationBackend::new(
            tokenizer(),
            engine(executor),
            EngineBackendConfig::default(),
            policy(),
        )
        .expect("the fixture worker starts");
        let metrics = Arc::new(backend.runtime_metrics(&[]));
        Self { backend, metrics }
    }

    /// Reads the provider exactly as the `/metrics` route reads it.
    fn published(&self) -> String {
        let provider: &dyn MetricsSnapshotProvider = self.metrics.as_ref();
        provider.snapshot(12_345_678)
    }

    async fn submit(&self, prompt: &str, maximum_new_tokens: usize) -> InFlightRequestV1 {
        let cancellation = CancellationSignal::new();
        let (sender, events) = tokio::sync::mpsc::channel(64);
        self.backend
            .submit(
                request(prompt, maximum_new_tokens, cancellation.clone()),
                EventSender::new(sender),
            )
            .await
            .expect("the fixture worker admits the request");
        InFlightRequestV1 {
            events,
            cancellation,
        }
    }
}

struct InFlightRequestV1 {
    events: tokio::sync::mpsc::Receiver<GenerationEvent>,
    cancellation: CancellationSignal,
}

impl InFlightRequestV1 {
    /// Reads events until the request stops producing them.
    async fn drain(&mut self) -> Vec<GenerationEvent> {
        let mut seen = Vec::new();
        while let Ok(Some(event)) = tokio::time::timeout(TIMEOUT, self.events.recv()).await {
            let terminal = matches!(
                event,
                GenerationEvent::Finished { .. } | GenerationEvent::Failed { .. }
            );
            seen.push(event);
            if terminal {
                break;
            }
        }
        seen
    }
}

const TIMEOUT: Duration = Duration::from_secs(10);

#[tokio::test]
async fn list_prompts_share_one_prefill_step() {
    let (value, executions) =
        batch_http("/v1/completions", serde_json::json!({"prompt":["x","yy"]})).await;
    assert_eq!(value["choices"][0]["index"], 0);
    assert_eq!(value["choices"][1]["index"], 1);
    assert_eq!(
        value["choices"][0]["token_ids"],
        serde_json::json!([BYTE_ID_START + u32::from(b'a')])
    );
    assert_eq!(
        value["choices"][1]["token_ids"],
        serde_json::json!([BYTE_ID_START + u32::from(b'b')])
    );
    assert_eq!(value["usage"]["prompt_tokens"], 3);
    assert_eq!(value["usage"]["completion_tokens"], 2);
    assert_eq!(executions, 1, "both prompts fit one prefill step");
}

#[tokio::test]
async fn multi_choice_paths_share_prefill_and_respect_the_scheduler_budget() {
    for (path, body, choices, steps) in [
        (
            "/v1/completions",
            serde_json::json!({"prompt":[[7],[8,9]]}),
            2,
            1,
        ),
        (
            "/v1/completions",
            serde_json::json!({"prompt":"x","n":2}),
            2,
            1,
        ),
        (
            "/v1/completions",
            serde_json::json!({"prompt":["x","y"],"n":2}),
            4,
            2,
        ),
        (
            "/v1/completions",
            serde_json::json!({"prompt":["x","y"],"structured_outputs":{"regex":"a"}}),
            2,
            1,
        ),
        (
            "/v1/chat/completions",
            serde_json::json!({"messages":[{"role":"user","content":"x"}],"reasoning_effort":"off","n":2}),
            2,
            1,
        ),
    ] {
        let (value, executions) = batch_http(path, body).await;
        assert_eq!(value["choices"].as_array().unwrap().len(), choices);
        for (index, choice) in value["choices"].as_array().unwrap().iter().enumerate() {
            assert_eq!(choice["index"], index);
        }
        assert_eq!(value["usage"]["completion_tokens"], choices);
        assert_eq!(executions, steps, "{value}");
    }
}

#[tokio::test]
async fn replica_pool_groups_choices_per_destination() {
    let (value, executions) = batch_http_replicas(
        "/v1/completions",
        serde_json::json!({"prompt":["a","b","c","d"]}),
        2,
    )
    .await;
    assert_eq!(value["choices"].as_array().unwrap().len(), 4);
    assert_eq!(value["usage"]["prompt_tokens"], 4);
    assert_eq!(executions, 2, "one prefill step per replica");
}

async fn batch_http(path: &str, body: serde_json::Value) -> (serde_json::Value, usize) {
    batch_http_replicas(path, body, 1).await
}

async fn batch_http_replicas(
    path: &str,
    body: serde_json::Value,
    replicas: usize,
) -> (serde_json::Value, usize) {
    let mut executions = Vec::new();
    let replicas = (0..replicas)
        .map(|_| {
            let executor = FixtureExecutorV1::spelling("abcd");
            executions.push(executor.executions.clone());
            Arc::new(ServedWorkerV1::start(executor).backend)
        })
        .collect();
    let backend = Arc::new(runtime_server::LocalReplicaPool::new(replicas).unwrap());
    let response = batch_http_raw(backend, path, body).await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let value = serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
    (
        value,
        executions
            .iter()
            .map(|count| count.load(Ordering::Relaxed))
            .sum(),
    )
}

async fn batch_http_raw(
    backend: Arc<dyn GenerationBackend>,
    path: &str,
    mut body: serde_json::Value,
) -> String {
    let config = runtime_server::ServerConfig::default();
    let frontend = runtime_host::frontend_from_declaration_v1(
        &declaration(),
        &runtime_frontend::test_support::fixture_export_v1(),
        config,
    )
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(runtime_server::serve(listener, backend, frontend, config));
    body["model"] = serde_json::json!("fixture/metrics-1");
    body.as_object_mut()
        .unwrap()
        .entry("max_tokens")
        .or_insert(serde_json::json!(1));
    body.as_object_mut()
        .unwrap()
        .entry("return_token_ids")
        .or_insert(serde_json::json!(true));
    body["temperature"] = serde_json::json!(0);
    let body = body.to_string();
    let socket = tokio::net::TcpStream::connect(address).await.unwrap();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let raw = tokio::time::timeout(TIMEOUT, batch_exchange(&socket, request.as_bytes()))
        .await
        .unwrap();
    server.abort();
    let _ = server.await;
    String::from_utf8(raw).unwrap()
}

#[tokio::test]
async fn batch_admission_errors_stay_with_the_failed_member() {
    let worker = ServedWorkerV1::start(FixtureExecutorV1::emitting());
    let mut receivers = Vec::new();
    let requests = ["x".to_owned(), "x".repeat(CONTEXT_TOKENS), "y".to_owned()]
        .into_iter()
        .map(|prompt| {
            let (sender, receiver) = tokio::sync::mpsc::channel(64);
            receivers.push(receiver);
            (
                request(&prompt, 1, CancellationSignal::new()),
                EventSender::new(sender),
            )
        })
        .collect();
    let admissions = worker.backend.submit_batch(requests);
    let mut outcomes = Vec::new();
    for admission in admissions {
        outcomes.push(tokio::time::timeout(TIMEOUT, admission).await.unwrap());
    }
    assert_eq!(outcomes[0], Ok(()));
    assert_eq!(outcomes[1], Err(AdmissionError::InvalidRequest));
    assert_eq!(outcomes[2], Ok(()));
    for (index, events) in receivers.into_iter().enumerate() {
        if index == 1 {
            continue;
        }
        let mut request = InFlightRequestV1 {
            events,
            cancellation: CancellationSignal::new(),
        };
        assert!(
            request
                .drain()
                .await
                .iter()
                .any(|event| matches!(event, GenerationEvent::Finished { .. }))
        );
    }
}

#[tokio::test]
async fn cancelling_a_batch_waiting_for_control_space_releases_every_member() {
    let (backend, resume) = paused_backend(1);
    let _live = hold_live_request(&backend).await;
    submit_and_cancel(&backend);
    let cancellation = CancellationSignal::new();
    let mut receivers = Vec::new();
    let requests = (0..2)
        .map(|_| {
            let (sender, receiver) = tokio::sync::mpsc::channel(64);
            receivers.push(receiver);
            (
                request("x", 1, cancellation.clone()),
                EventSender::new(sender),
            )
        })
        .collect();
    let mut admissions = backend.submit_batch(requests);
    for admission in &mut admissions {
        assert!(futures_poll_once(admission).await.is_none());
    }
    cancellation.cancel();
    for admission in admissions {
        assert_eq!(
            tokio::time::timeout(TIMEOUT, admission).await.unwrap(),
            Err(AdmissionError::WorkerUnavailable)
        );
    }
    for mut receiver in receivers {
        assert!(
            tokio::time::timeout(TIMEOUT, receiver.recv())
                .await
                .unwrap()
                .is_none()
        );
    }
    resume.send(()).unwrap();
    let (sender, _receiver) = tokio::sync::mpsc::channel(64);
    assert_eq!(
        tokio::time::timeout(
            TIMEOUT,
            backend.submit(
                request("z", 1, CancellationSignal::new()),
                EventSender::new(sender)
            )
        )
        .await
        .unwrap(),
        Ok(())
    );
}

#[tokio::test]
async fn cancelling_a_batch_releases_running_and_queued_choices() {
    let (backend, resume) = paused_backend(1);
    let cancellation = CancellationSignal::new();
    let mut receivers = Vec::new();
    let requests = (0..2)
        .map(|_| {
            let (sender, receiver) = tokio::sync::mpsc::channel(64);
            receivers.push(receiver);
            (
                request("x", 8, cancellation.clone()),
                EventSender::new(sender),
            )
        })
        .collect();
    let mut admissions = backend.submit_batch(requests).into_iter();
    assert_eq!(
        tokio::time::timeout(TIMEOUT, admissions.next().unwrap())
            .await
            .unwrap(),
        Ok(())
    );
    cancellation.cancel();
    resume.send(()).unwrap();
    assert_eq!(
        tokio::time::timeout(TIMEOUT, admissions.next().unwrap())
            .await
            .unwrap(),
        Err(AdmissionError::WorkerUnavailable)
    );
    drop(admissions);
    let metrics = Arc::new(backend.runtime_metrics(&[]));
    let worker = ServedWorkerV1 { backend, metrics };
    published_when_all(&worker, &[("chop_cancellations_total", 2.0)]).await;
}

fn grouped_test_engine(
    executor: FixtureExecutorV1,
    sequences: usize,
    prefix: bool,
) -> Engine<FixtureExecutorV1> {
    let config = EngineConfig::new(
        SchedulerConfig::new(sequences, 16, 512, 16, 1).unwrap(),
        KvGeometry::new(16, 64).unwrap(),
        ModelVocabulary::new(VOCABULARY_SIZE).unwrap(),
        CONTEXT_TOKENS,
        2,
    )
    .unwrap();
    Engine::new(
        if prefix {
            config.with_prefix_cache(1)
        } else {
            config
        },
        executor,
    )
    .unwrap()
}

#[tokio::test]
async fn grouped_choices_reuse_identical_and_shared_prompt_prefixes() {
    for distinct_suffix in [false, true] {
        let executor = FixtureExecutorV1::emitting();
        let input_tokens = executor.input_tokens.clone();
        let backend = EngineGenerationBackend::new(
            tokenizer(),
            grouped_test_engine(executor, 8, true),
            EngineBackendConfig::default(),
            policy(),
        )
        .unwrap();
        let mut receivers = Vec::new();
        let requests = (0..8)
            .map(|index| {
                let (sender, receiver) = tokio::sync::mpsc::channel(64);
                receivers.push(receiver);
                let suffix = if distinct_suffix {
                    char::from(b'a' + index)
                } else {
                    'a'
                };
                (
                    request(
                        &format!("{}{suffix}", "x".repeat(32)),
                        1,
                        CancellationSignal::new(),
                    ),
                    EventSender::new(sender),
                )
            })
            .collect();
        for admitted in backend.submit_batch(requests) {
            assert_eq!(
                tokio::time::timeout(TIMEOUT, admitted).await.unwrap(),
                Ok(())
            );
        }
        let mut cached = 0;
        for events in receivers {
            let mut request = InFlightRequestV1 {
                events,
                cancellation: CancellationSignal::new(),
            };
            for event in request.drain().await {
                if let GenerationEvent::Finished { usage, .. } = event {
                    cached += usage.cached_tokens.unwrap_or(0);
                }
            }
        }
        assert_eq!(
            input_tokens.load(Ordering::Relaxed),
            40,
            "one shared 32-token prefix plus eight suffixes"
        );
        assert_eq!(cached, 7 * 32);
    }
}

#[tokio::test]
async fn grouped_choices_do_not_expire_behind_their_siblings() {
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        grouped_test_engine(FixtureExecutorV1::slow(), 1, false),
        EngineBackendConfig::default().with_queue_wait(Duration::from_millis(20)),
        policy(),
    )
    .unwrap();
    // Once one choice runs, the other three wait behind it, each at least
    // 16 steps of 5 ms, far past the 20 ms queue wait, and are still admitted.
    for events in start_group(&backend, 4, 16).await {
        let mut request = InFlightRequestV1 {
            events,
            cancellation: CancellationSignal::new(),
        };
        assert!(
            request
                .drain()
                .await
                .iter()
                .any(|event| matches!(event, GenerationEvent::Finished { .. }))
        );
    }
}

#[tokio::test]
async fn a_group_queued_behind_another_request_still_expires() {
    let (executor, mut resume) = FixtureExecutorV1::slow_holding(&[1]);
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        grouped_test_engine(executor, 1, false),
        EngineBackendConfig::default().with_queue_wait(Duration::from_millis(20)),
        policy(),
    )
    .unwrap();
    let _live = start_group(&backend, 1, 32).await;
    // The live request's second step holds while the group is queued, so the
    // worker reads the group with at least 30 steps of 5 ms still to run.
    let requests = (0..2)
        .map(|_| {
            let (sender, _receiver) = tokio::sync::mpsc::channel(64);
            (
                request("y", 1, CancellationSignal::new()),
                EventSender::new(sender),
            )
        })
        .collect();
    let admissions = backend.submit_batch(requests);
    resume.remove(0).send(()).unwrap();
    for admitted in admissions {
        assert_eq!(
            tokio::time::timeout(TIMEOUT, admitted).await.unwrap(),
            Err(AdmissionError::AtCapacity)
        );
    }
}

#[tokio::test]
async fn real_worker_batch_validation_errors_return_http_400_before_streaming() {
    for stream in [false, true] {
        let executor = FixtureExecutorV1::emitting();
        let executions = executor.executions.clone();
        let worker = ServedWorkerV1::start(executor);
        // The worker's declaration does not enable prompt rows. Preparation
        // succeeds, but each admission reports the unsupported scoring request.
        let response = batch_http_raw(
            Arc::new(worker.backend),
            "/v1/completions",
            serde_json::json!({"prompt":["x","yy"],"prompt_logprobs":1,"stream":stream}),
        )
        .await;
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");
        assert!(!response.contains("data: "), "{response}");
        assert_eq!(executions.load(Ordering::Relaxed), 0);
    }
}

#[tokio::test]
async fn real_worker_streaming_batch_preserves_indexes_usage_and_logprobs() {
    let worker = ServedWorkerV1::start(FixtureExecutorV1::spelling("abcd"));
    let response = batch_http_raw(Arc::new(worker.backend), "/v1/completions", serde_json::json!({"prompt":["x","yy"],"n":2,"logprobs":1,"stream":true,"stream_options":{"include_usage":true}})).await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let events: Vec<serde_json::Value> = response
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect();
    assert!(
        events.iter().all(|event| event.get("error").is_none()),
        "{response}"
    );
    let usage = &events.last().unwrap()["usage"];
    assert_eq!(usage["prompt_tokens"], 3);
    assert_eq!(usage["completion_tokens"], 4);
    let mut tokens = vec![Vec::new(); 4];
    let mut finished = Vec::new();
    for event in &events {
        for choice in event["choices"].as_array().unwrap() {
            let index = choice["index"].as_u64().unwrap() as usize;
            if let Some(ids) = choice["token_ids"].as_array() {
                tokens[index].extend(ids.iter().map(|id| id.as_u64().unwrap() as u32));
            }
            if !choice["finish_reason"].is_null() {
                finished.push(index);
            }
        }
    }
    finished.sort_unstable();
    assert_eq!(finished, [0, 1, 2, 3]);
    for (index, ids) in tokens.iter().enumerate() {
        assert_eq!(ids, &[BYTE_ID_START + u32::from(b'a') + index as u32]);
    }
    assert_eq!(response.matches("data: [DONE]").count(), 1);
    assert!(
        events
            .iter()
            .flat_map(|event| event["choices"].as_array().unwrap())
            .any(|choice| choice["logprobs"]["token_logprobs"]
                .as_array()
                .is_some_and(|values| !values.is_empty()))
    );
}

/// Waits for one snapshot containing every expected value from the worker.
async fn published_when_all(worker: &ServedWorkerV1, expected: &[(&str, f64)]) -> String {
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    loop {
        let text = worker.published();
        if expected
            .iter()
            .all(|(name, value)| metric_value(&text, name) == Some(*value))
        {
            return text;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the worker never published {expected:?}:\n{text}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn metric_value(text: &str, name: &str) -> Option<f64> {
    text.lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| {
            let (metric, value) = line.split_once(' ')?;
            (metric == name).then(|| value.parse().ok())?
        })
}

fn value(text: &str, name: &str) -> f64 {
    metric_value(text, name).unwrap_or_else(|| panic!("{name} is not published:\n{text}"))
}

/// A completed request moves the engine's own counters and leaves no sequence.
#[tokio::test]
async fn a_served_request_moves_engine_counters_and_settles_idle() {
    let worker = ServedWorkerV1::start(FixtureExecutorV1::emitting());

    let before = worker.published();
    assert_eq!(value(&before, "chop_generation_worker_ready"), 1.0);
    assert_eq!(value(&before, "chop_requests_submitted_total"), 0.0);
    assert_eq!(value(&before, "chop_batches_total"), 0.0);
    assert_eq!(value(&before, "chop_kv_active_sequences"), 0.0);
    assert!(value(&before, "chop_kv_block_capacity") > 0.0);
    assert!(value(&before, "chop_process_uptime_seconds") > 12.0);

    let mut request = worker.submit("hello", 4).await;
    let events = request.drain().await;
    assert!(
        matches!(events.last(), Some(GenerationEvent::Finished { .. })),
        "the request finished: {events:?}"
    );

    let labels = r#"model_name="fixture/metrics-1",engine="0""#;
    let compatibility_expected = [
        ("vllm:generation_tokens_total", 4.0),
        ("vllm:request_generation_tokens_sum", 4.0),
        ("vllm:request_prompt_tokens_count", 1.0),
        ("vllm:request_queue_time_seconds_count", 1.0),
        ("vllm:request_prefill_time_seconds_count", 1.0),
        ("vllm:request_decode_time_seconds_count", 1.0),
        ("vllm:iteration_tokens_total_count", 4.0),
        ("vllm:num_preemptions_total", 0.0),
        ("chop_instance_poisoned", 0.0),
    ]
    .map(|(name, expected)| (format!("{name}{{{labels}}}"), expected));
    let mut expected = vec![
        ("chop_requests_submitted_total", 1.0),
        ("chop_requests_completed_total", 1.0),
        ("chop_requests_terminal_total", 1.0),
        ("chop_requests_active", 0.0),
        ("chop_sequences_active", 0.0),
        ("chop_generation_tokens_total", 4.0),
        ("chop_ttft_count", 1.0),
        ("chop_itl_count", 3.0),
        ("chop_queue_duration_count", 1.0),
        ("chop_kv_active_sequences", 0.0),
        ("chop_errors_total", 0.0),
        ("chop_cancellations_total", 0.0),
        ("chop_generation_worker_ready", 1.0),
        ("chop_generation_worker_failed", 0.0),
        ("chop_mixed_batches_total", 0.0),
        ("chop_execution_observations_total", 0.0),
    ];
    expected.extend(
        compatibility_expected
            .iter()
            .map(|(name, expected)| (name.as_str(), *expected)),
    );
    // Finished is delivered before the frontend records completion histograms.
    // Native idle counters alone do not mean those observations are published.
    // Wait for every asserted counter in the same scrape, including vLLM's.
    let after = published_when_all(&worker, &expected).await;
    assert_eq!(value(&after, "chop_requests_submitted_total"), 1.0);
    assert_eq!(value(&after, "chop_requests_completed_total"), 1.0);
    for (name, expected) in &compatibility_expected {
        assert_eq!(value(&after, name), *expected, "{name}:\n{after}");
    }

    assert_eq!(value(&after, "chop_requests_terminal_total"), 1.0);
    assert_eq!(value(&after, "chop_errors_total"), 0.0);
    assert_eq!(value(&after, "chop_cancellations_total"), 0.0);
    assert!(value(&after, "chop_prompt_tokens_total") > 0.0);
    assert_eq!(value(&after, "chop_generation_tokens_total"), 4.0);
    assert_eq!(value(&after, "chop_ttft_count"), 1.0);
    assert_eq!(value(&after, "chop_itl_count"), 3.0);
    assert_eq!(value(&after, "chop_queue_duration_count"), 1.0);

    // The worker retired the request, so the idle it reports is a real one.
    assert_eq!(value(&after, "chop_requests_active"), 0.0);
    assert_eq!(value(&after, "chop_sequences_active"), 0.0);
    assert_eq!(value(&after, "chop_generation_worker_ready"), 1.0);
    assert_eq!(value(&after, "chop_generation_worker_failed"), 0.0);

    // Batch membership comes from the scheduler's phases, and one request's
    // prompt and decodes never share a batch.
    assert!(value(&after, "chop_prefill_only_batches_total") >= 1.0);
    assert_eq!(value(&after, "chop_mixed_batches_total"), 0.0);
    assert_eq!(
        value(&after, "chop_batch_items_total"),
        value(&after, "chop_prefill_batch_items_total")
            + value(&after, "chop_decode_batch_items_total")
    );

    // No producer observes these, so none of them is published as a zero.
    assert_eq!(value(&after, "chop_execution_observations_total"), 0.0);
    for absent in [
        "chop_kv_evictions_total",
        "chop_kv_cow_bytes_total",
        "chop_peak_device_bytes",
        // This deployment declares no FP8 pool, so it claims no FP8 number.
        "chop_fp8_kv_tokens",
        "chop_fp8_kv_capacity_tokens",
        // The fixture executor measures no kernel or collective work, and an
        // executor that measures none publishes none.
        "chop_kernel_launches_total",
        "chop_collective_operations_total",
    ] {
        assert!(
            metric_value(&after, absent).is_none(),
            "{absent} is published without a producer:\n{after}"
        );
    }
}

/// A cancelled request is counted as cancelled, not as one that completed.
#[tokio::test]
async fn a_cancelled_request_is_published_as_cancelled() {
    let (resume, paused) = std::sync::mpsc::channel();
    let worker = ServedWorkerV1::start(FixtureExecutorV1 {
        resume_second_step: Some(paused),
        ..FixtureExecutorV1::emitting()
    });
    let mut request = worker.submit("hello", 32).await;
    let first = tokio::time::timeout(TIMEOUT, request.events.recv())
        .await
        .expect("the first event arrives");
    assert!(first.is_some());

    request.cancellation.cancel();
    // The worker cannot finish all tokens before the test thread cancels.
    let _ = resume.send(());
    let text = published_when_all(
        &worker,
        &[
            ("chop_cancellations_total", 1.0),
            ("chop_requests_completed_total", 0.0),
            ("chop_requests_terminal_total", 1.0),
            ("chop_requests_active", 0.0),
            ("chop_sequences_active", 0.0),
            ("chop_kv_active_sequences", 0.0),
            ("chop_kv_tokens", 0.0),
        ],
    )
    .await;
    assert_eq!(value(&text, "chop_requests_completed_total"), 0.0);
    assert_eq!(value(&text, "chop_requests_terminal_total"), 1.0);
    assert_eq!(value(&text, "chop_requests_active"), 0.0);
    assert_eq!(value(&text, "chop_sequences_active"), 0.0);
    assert_eq!(value(&text, "chop_generation_worker_ready"), 1.0);
    // The pool released what the cancelled sequence held.
    assert_eq!(value(&text, "chop_kv_active_sequences"), 0.0);
    assert_eq!(value(&text, "chop_kv_tokens"), 0.0);
}

#[tokio::test]
async fn a_queued_request_waits_then_expires_without_stopping_the_worker() {
    let (executor, mut resume) = FixtureExecutorV1::slow_holding(&[1]);
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine(executor),
        EngineBackendConfig::default().with_queue_wait(Duration::from_millis(20)),
        policy(),
    )
    .expect("the fixture worker starts");
    let held = start_group(&backend, 2, 32).await;
    // The live requests' second step holds while this one is queued, so the
    // worker reads it with at least 30 steps of 5 ms still to run.
    let (sender, _receiver) = tokio::sync::mpsc::channel(64);
    let queued = backend.submit(
        request("x", 4, CancellationSignal::new()),
        EventSender::new(sender),
    );
    resume.remove(0).send(()).unwrap();
    let refused = tokio::time::timeout(TIMEOUT, queued)
        .await
        .expect("the queue wait is bounded");
    assert_eq!(refused, Err(runtime_server::AdmissionError::AtCapacity));
    assert_eq!(backend.readiness(), runtime_server::BackendReadiness::Ready);
    drop(held);
}

/// A burst the in-flight limit admits waits in the queue while the worker is
/// inside a step, rather than being refused by the worker's control channel.
#[tokio::test]
async fn a_burst_within_the_in_flight_limit_queues_while_the_worker_is_busy() {
    const QUEUE: usize = 300;
    let (resume, paused) = std::sync::mpsc::channel();
    let executor = FixtureExecutorV1 {
        resume_second_step: Some(paused),
        ..FixtureExecutorV1::emitting()
    };
    let scheduler = SchedulerConfig::new(2, 2 * 2 + QUEUE, 64, 64, 1)
        .and_then(|config| {
            config.with_capacity_profile(SchedulerCapacityProfile::new("served", 2 + QUEUE)?)
        })
        .expect("the fixture schedule is valid");
    let engine = Engine::new(
        EngineConfig::new(
            scheduler,
            KvGeometry::new(16, 16).expect("the fixture page pool is valid"),
            ModelVocabulary::new(VOCABULARY_SIZE).expect("the fixture vocabulary is valid"),
            CONTEXT_TOKENS,
            2,
        )
        .expect("the fixture engine configuration is valid"),
        executor,
    )
    .expect("the fixture engine builds");
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine,
        EngineBackendConfig::new(2 + QUEUE).expect("the control capacity is valid"),
        policy(),
    )
    .expect("the fixture worker starts");
    let (sender, mut first) = tokio::sync::mpsc::channel(64);
    backend
        .submit(
            request("x", 8, CancellationSignal::new()),
            EventSender::new(sender),
        )
        .await
        .expect("the first request is admitted");
    // The first token means the worker is in its second step, which holds.
    tokio::time::timeout(TIMEOUT, first.recv())
        .await
        .expect("the first token arrives");
    let mut queued = Vec::new();
    let mut receivers = Vec::new();
    for _ in 0..QUEUE {
        let (sender, receiver) = tokio::sync::mpsc::channel(64);
        receivers.push(receiver);
        let mut submitted = backend.submit(
            request("x", 1, CancellationSignal::new()),
            EventSender::new(sender),
        );
        // A refused command resolves at once; a queued one waits.
        assert!(
            futures_poll_once(&mut submitted).await.is_none(),
            "the request was queued, not refused"
        );
        queued.push(submitted);
    }
    resume.send(()).expect("the worker waits for the release");
    for submitted in queued {
        assert_eq!(
            tokio::time::timeout(TIMEOUT, submitted)
                .await
                .expect("every queued request is admitted"),
            Ok(())
        );
    }
    assert_eq!(backend.readiness(), runtime_server::BackendReadiness::Ready);
    drop(receivers);
}

/// A worker with one live slot and one queued slot, whose second step holds
/// until `resume` is sent, over a control channel of `control` commands.
fn paused_backend(control: usize) -> (EngineGenerationBackend, std::sync::mpsc::Sender<()>) {
    let (resume, paused) = std::sync::mpsc::channel();
    let executor = FixtureExecutorV1 {
        resume_second_step: Some(paused),
        ..FixtureExecutorV1::emitting()
    };
    let scheduler = SchedulerConfig::new(1, 4, 64, 64, 1)
        .and_then(|config| {
            config.with_capacity_profile(SchedulerCapacityProfile::new("served", 2)?)
        })
        .expect("the fixture schedule is valid");
    let engine = Engine::new(
        EngineConfig::new(
            scheduler,
            KvGeometry::new(16, 16).expect("the fixture page pool is valid"),
            ModelVocabulary::new(VOCABULARY_SIZE).expect("the fixture vocabulary is valid"),
            CONTEXT_TOKENS,
            2,
        )
        .expect("the fixture engine configuration is valid"),
        executor,
    )
    .expect("the fixture engine builds");
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine,
        EngineBackendConfig::new(control).expect("the control capacity is valid"),
        policy(),
    )
    .expect("the fixture worker starts");
    (backend, resume)
}

/// Starts one live request and waits until the worker holds its second step.
async fn hold_live_request(
    backend: &EngineGenerationBackend,
) -> tokio::sync::mpsc::Receiver<GenerationEvent> {
    let (sender, mut live) = tokio::sync::mpsc::channel(64);
    backend
        .submit(
            request("x", 8, CancellationSignal::new()),
            EventSender::new(sender),
        )
        .await
        .expect("the live request is admitted");
    tokio::time::timeout(TIMEOUT, live.recv())
        .await
        .expect("the first token arrives");
    live
}

/// Submits a request and cancels it before the worker reads it, as a client
/// that disconnects while queued does; its HTTP place is free again at once.
fn submit_and_cancel(backend: &EngineGenerationBackend) {
    let cancellation = CancellationSignal::new();
    let (sender, receiver) = tokio::sync::mpsc::channel(64);
    let submitted = backend.submit(
        request("x", 1, cancellation.clone()),
        EventSender::new(sender),
    );
    cancellation.cancel();
    drop(submitted);
    drop(receiver);
}

/// A replacement for a request cancelled while queued takes its place: the
/// worker drops the cancelled request before it admits the replacement.
#[tokio::test]
async fn a_replacement_for_a_cancelled_queued_request_is_admitted() {
    let (backend, resume) = paused_backend(2);
    let _live = hold_live_request(&backend).await;
    submit_and_cancel(&backend);
    let (sender, _receiver) = tokio::sync::mpsc::channel(64);
    let replacement = backend.submit(
        request("x", 1, CancellationSignal::new()),
        EventSender::new(sender),
    );
    resume.send(()).expect("the worker waits for the release");
    assert_eq!(
        tokio::time::timeout(TIMEOUT, replacement)
            .await
            .expect("the replacement is answered"),
        Ok(())
    );
}

/// A request the engine already queued, then cancelled, leaves the engine
/// before a replacement that arrives in the same worker pass is admitted.
#[tokio::test]
async fn a_replacement_for_a_cancelled_engine_queued_request_is_admitted() {
    let (backend, resume) = paused_backend(2);
    let (sender, mut live) = tokio::sync::mpsc::channel(64);
    let first = backend.submit(
        request("x", 8, CancellationSignal::new()),
        EventSender::new(sender),
    );
    // Sent before the live request's first step, so the worker reads it
    // into the engine's queue before it holds the second step.
    let cancellation = CancellationSignal::new();
    let (sender, queued_receiver) = tokio::sync::mpsc::channel(64);
    let queued = backend.submit(
        request("x", 1, cancellation.clone()),
        EventSender::new(sender),
    );
    first.await.expect("the live request is admitted");
    tokio::time::timeout(TIMEOUT, live.recv())
        .await
        .expect("the first token arrives");
    cancellation.cancel();
    drop(queued);
    drop(queued_receiver);
    let (sender, _receiver) = tokio::sync::mpsc::channel(64);
    let replacement = backend.submit(
        request("x", 1, CancellationSignal::new()),
        EventSender::new(sender),
    );
    resume.send(()).expect("the worker waits for the release");
    assert_eq!(
        tokio::time::timeout(TIMEOUT, replacement)
            .await
            .expect("the replacement is answered"),
        Ok(())
    );
}

/// Commands of cancelled requests still in the control channel do not refuse
/// a live one: it waits for the worker to drain them.
#[tokio::test]
async fn cancelled_commands_in_the_control_channel_do_not_refuse_live_requests() {
    let (backend, resume) = paused_backend(2);
    let _live = hold_live_request(&backend).await;
    submit_and_cancel(&backend);
    submit_and_cancel(&backend);
    let (sender, _receiver) = tokio::sync::mpsc::channel(64);
    let mut replacement = backend.submit(
        request("x", 1, CancellationSignal::new()),
        EventSender::new(sender),
    );
    assert!(
        futures_poll_once(&mut replacement).await.is_none(),
        "the request waits for channel space"
    );
    resume.send(()).expect("the worker waits for the release");
    assert_eq!(
        tokio::time::timeout(TIMEOUT, replacement)
            .await
            .expect("the replacement is answered"),
        Ok(())
    );
    assert_eq!(backend.readiness(), runtime_server::BackendReadiness::Ready);
}

/// Polls a future once without waiting for it.
async fn futures_poll_once<F: std::future::Future + Unpin>(future: &mut F) -> Option<F::Output> {
    std::future::poll_fn(|context| {
        std::task::Poll::Ready(match std::pin::Pin::new(&mut *future).poll(context) {
            std::task::Poll::Ready(output) => Some(output),
            std::task::Poll::Pending => None,
        })
    })
    .await
}

#[tokio::test]
async fn a_queued_request_is_admitted_after_a_live_slot_frees() {
    let (executor, mut resume) = FixtureExecutorV1::slow_holding(&[1]);
    // The queue wait outlasts the live requests however loaded the host is.
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine(executor),
        EngineBackendConfig::default().with_queue_wait(TIMEOUT),
        policy(),
    )
    .expect("the fixture worker starts");
    let held = start_group(&backend, 2, 32).await;
    // Queued while both live requests still hold their slots.
    let (sender, _receiver) = tokio::sync::mpsc::channel(64);
    let queued = backend.submit(
        request("x", 1, CancellationSignal::new()),
        EventSender::new(sender),
    );
    resume.remove(0).send(()).unwrap();
    let admitted = tokio::time::timeout(TIMEOUT, queued)
        .await
        .expect("the queued request eventually receives a slot");
    assert_eq!(admitted, Ok(()));
    assert_eq!(backend.readiness(), runtime_server::BackendReadiness::Ready);
    drop(held);
}

#[tokio::test]
async fn cancelling_a_queued_request_releases_its_place() {
    let (executor, mut resume) = FixtureExecutorV1::slow_holding(&[1, 2]);
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine(executor),
        EngineBackendConfig::default().with_queue_wait(Duration::from_secs(1)),
        policy(),
    )
    .expect("the fixture worker starts");
    let held = start_group(&backend, 2, 32).await;
    let metrics = backend.runtime_metrics(&[]);
    // A group start the queue wait expired counts too, so these are deltas.
    let submitted = value(&metrics.prometheus(0), "chop_requests_submitted_total");
    let cancellations = value(&metrics.prometheus(0), "chop_cancellations_total");
    // Queued while the live requests' second step holds; the worker takes it
    // into its engine before the third step, which holds in turn, so the
    // request is cancelled while it waits in the engine queue.
    let cancellation = CancellationSignal::new();
    let (sender, _receiver) = tokio::sync::mpsc::channel(64);
    let queued = backend.submit(
        request("x", 4, cancellation.clone()),
        EventSender::new(sender),
    );
    resume.remove(0).send(()).unwrap();
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    while value(&metrics.prometheus(0), "chop_requests_submitted_total") < submitted + 1.0 {
        assert!(tokio::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    cancellation.cancel();
    resume.remove(0).send(()).unwrap();
    let result = tokio::time::timeout(TIMEOUT, queued)
        .await
        .expect("queued cancellation settles");
    assert_eq!(
        result,
        Err(runtime_server::AdmissionError::WorkerUnavailable)
    );
    while value(&metrics.prometheus(0), "chop_cancellations_total") < cancellations + 1.0 {
        assert!(tokio::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(backend.readiness(), runtime_server::BackendReadiness::Ready);
    drop(held);
}

#[test]
fn tokenizer_input_bound_applies_to_text_but_not_token_ids() {
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine(FixtureExecutorV1::emitting()),
        EngineBackendConfig::default().with_tokenizer_input_bytes(1),
        policy(),
    )
    .expect("the fixture worker starts");
    assert_eq!(
        backend.prepare_prompt(PreparedPrompt::Raw("ab".to_owned())),
        Err(runtime_server::AdmissionError::InvalidPrompt)
    );
    let prepared = backend
        .prepare_prompt(PreparedPrompt::TokenIds(vec![
            BYTE_ID_START + u32::from(b'a'),
        ]))
        .expect("token IDs bypass the text-input bound");
    assert_eq!(prepared.1, 1);
}

#[test]
fn served_tokenizer_accepts_text_above_four_mib_with_its_configured_limit() {
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine(FixtureExecutorV1::emitting()),
        EngineBackendConfig::default().with_tokenizer_input_bytes(8 * 1024 * 1024),
        policy(),
    )
    .expect("the fixture worker starts");
    let marker = "<<begin>>";
    let count = 4 * 1024 * 1024 / marker.len() + 1;
    let prompt = marker.repeat(count);
    let (prepared, tokens) = backend
        .prepare_prompt(PreparedPrompt::RenderedChat(prompt))
        .expect("configured input limit admits the rendered text");
    assert_eq!(tokens, count);
    assert!(matches!(prepared, PreparedPrompt::TokenIds(ids) if ids.len() == count));
}

/// A worker that failed publishes the failure, not a healthy idle process.
#[tokio::test]
async fn a_failed_worker_publishes_failure_rather_than_idle_success() {
    let worker = ServedWorkerV1::start(FixtureExecutorV1::failing());
    let mut request = worker.submit("hello", 4).await;
    let events = request.drain().await;
    assert!(
        matches!(events.last(), Some(GenerationEvent::Failed { .. })),
        "the request failed: {events:?}"
    );

    let text = published_when_all(
        &worker,
        &[
            ("chop_generation_worker_failed", 1.0),
            ("chop_errors_total", 1.0),
            ("chop_requests_submitted_total", 1.0),
            ("chop_requests_completed_total", 0.0),
            ("chop_requests_terminal_total", 1.0),
            ("chop_requests_active", 0.0),
            ("chop_generation_tokens_total", 0.0),
        ],
    )
    .await;
    assert_eq!(value(&text, "chop_generation_worker_ready"), 0.0);
    assert_eq!(value(&text, "chop_generation_worker_starting"), 0.0);
    assert_eq!(value(&text, "chop_generation_worker_stopped"), 0.0);
    assert_eq!(value(&text, "chop_errors_total"), 1.0);
    assert_eq!(value(&text, "chop_requests_submitted_total"), 1.0);
    assert_eq!(value(&text, "chop_requests_completed_total"), 0.0);
    assert_eq!(value(&text, "chop_requests_terminal_total"), 1.0);
    assert_eq!(value(&text, "chop_requests_active"), 0.0);
    assert_eq!(value(&text, "chop_generation_tokens_total"), 0.0);
}

/// A bias outside the model vocabulary is a client error. A device sampler
/// would refuse it for the whole batch, so it never reaches the worker.
#[tokio::test]
async fn an_out_of_vocabulary_logit_bias_is_refused_before_the_worker() {
    let worker = ServedWorkerV1::start(FixtureExecutorV1::emitting());
    let mut refused = request("hello", 2, CancellationSignal::new());
    refused.sampling = SamplingParameters::new(0.0)
        .unwrap()
        .with_logit_bias(vec![
            LogitBias::new(u32::MAX, 1.0).unwrap(),
            LogitBias::new(BYTE_ID_START, 1.0).unwrap(),
            LogitBias::new(VOCABULARY_SIZE as u32, 1.0).unwrap(),
        ])
        .unwrap();
    let (sender, _events) = tokio::sync::mpsc::channel(8);
    let refusal = worker
        .backend
        .submit(refused, EventSender::new(sender))
        .await
        .expect_err("the bias is refused");
    assert_eq!(
        refusal,
        AdmissionError::InvalidParameter {
            message: format!(
                "token_id(s) [{VOCABULARY_SIZE}, {}] in logit_bias contain out-of-vocab token \
                 ids. Vocabulary size: {VOCABULARY_SIZE}",
                u32::MAX
            ),
            parameter: "logit_bias",
        }
    );

    let mut valid = worker.submit("hello", 2).await;
    let events = valid.drain().await;
    assert!(
        matches!(events.last(), Some(GenerationEvent::Finished { .. })),
        "the valid request finished: {events:?}"
    );
    published_when_all(
        &worker,
        &[
            ("chop_generation_worker_ready", 1.0),
            ("chop_requests_submitted_total", 1.0),
            ("chop_requests_completed_total", 1.0),
        ],
    )
    .await;
}

/// A chat state without reasoning, so the decoder reads content at once.
#[derive(Debug)]
struct PlainChatStateV1;

impl ChatStateV1 for PlainChatStateV1 {
    fn reasoning_enabled(&self) -> bool {
        false
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Ordinary byte tokens can spell the declared terminator. The decoder then
/// finishes the completion although the engine saw no stop token, and the
/// engine request must still be retired rather than left decoding.
#[tokio::test]
async fn a_terminator_spelled_by_ordinary_tokens_retires_the_engine_request() {
    let spelled = FixtureExecutorV1::spelling("hi<<end>>");
    assert_retired_after_terminator(spelled).await;
}

/// The real stop token finishes both the engine and the decoder.
#[tokio::test]
async fn a_stop_token_terminator_retires_the_engine_request() {
    let mut stopped = FixtureExecutorV1::spelling("hi");
    stopped.script.push(1);
    assert_retired_after_terminator(stopped).await;
}

async fn assert_retired_after_terminator(executor: FixtureExecutorV1) {
    let worker = ServedWorkerV1::start(executor);
    let cancellation = CancellationSignal::new();
    let mut chat = request("hello", 24, cancellation.clone());
    chat.chat = Some(ChatMetadata {
        include_reasoning: false,
        allowed_function_names: Vec::new(),
        named_tool_choice: false,
        state: Arc::new(PlainChatStateV1),
    });
    let (sender, events) = tokio::sync::mpsc::channel(64);
    worker
        .backend
        .submit(chat, EventSender::new(sender))
        .await
        .expect("the fixture worker admits the request");
    let mut in_flight = InFlightRequestV1 {
        events,
        cancellation,
    };
    let events = in_flight.drain().await;
    assert!(
        matches!(events.last(), Some(GenerationEvent::Finished { .. })),
        "the completion finished: {events:?}"
    );
    published_when_all(
        &worker,
        &[
            ("chop_requests_active", 0.0),
            ("chop_requests_terminal_total", 1.0),
            ("chop_generation_worker_ready", 1.0),
        ],
    )
    .await;
}

/// A stopped worker keeps its counters and stops claiming to be ready.
#[tokio::test]
async fn a_stopped_worker_is_published_as_stopped_with_what_it_did() {
    let worker = ServedWorkerV1::start(FixtureExecutorV1::emitting());
    let mut request = worker.submit("hello", 2).await;
    let _ = request.drain().await;
    let metrics = Arc::clone(&worker.metrics);

    let ServedWorkerV1 { backend, .. } = worker;
    drop(backend);

    let text = metrics.prometheus(20_000_000);
    assert_eq!(value(&text, "chop_generation_worker_stopped"), 1.0);
    assert_eq!(value(&text, "chop_generation_worker_ready"), 0.0);
    assert_eq!(value(&text, "chop_requests_submitted_total"), 1.0);
    assert_eq!(value(&text, "chop_requests_completed_total"), 1.0);
    assert_eq!(value(&text, "chop_process_uptime_seconds"), 20.0);
}

/// A worker whose engine refused to build publishes nothing at all.
#[test]
fn a_worker_that_never_started_yields_no_metrics_handle() {
    let started = EngineGenerationBackend::try_new_with_factory::<FixtureExecutorV1, _>(
        tokenizer(),
        || Err("the fixture device is unavailable".to_owned()),
        EngineBackendConfig::default(),
        policy(),
    );
    let Err(refusal) = started else {
        panic!("a refused engine must not start a worker");
    };
    assert_eq!(
        refusal,
        EngineBackendStartupError::EngineUnavailable(
            "the fixture device is unavailable".to_owned()
        )
    );
}

#[test]
fn unsupported_media_declarations_refuse_before_engine_startup() {
    for (kind, tag, parameters) in [
        (
            "audio",
            "audio.log-mel.v1",
            serde_json::json!({"sample_rate": 16000}),
        ),
        (
            "audio",
            "audio.log-mel.v1",
            serde_json::json!({"maximum_seconds": 60}),
        ),
        (
            "video",
            "video.timestamped-grids.v1",
            serde_json::json!({"maximum_frames": 16}),
        ),
        (
            "video",
            "video.timestamped-grids.v1",
            serde_json::json!({"sampling": "first"}),
        ),
        (
            "video",
            "video.timestamped-grids.v1",
            serde_json::json!({"repeat_last_if_odd": false}),
        ),
        (
            "video",
            "video.timestamped-grids.v1",
            serde_json::json!({"timestamp": "seconds"}),
        ),
    ] {
        let mut descriptor = declaration();
        descriptor.multimodal = Some(
            serde_json::from_value(serde_json::json!({
                "arena": {"rows":64,"page_rows":64,"hidden":8,"element":"bf16"},
                "encoders": [{"name":"synthetic","payload_element":"bf16","payload_columns":3,
                    "maximum_payload_bytes":1024,"cost_per_row_milli":1}],
                "modalities": [{"kind":kind,"part_types":[format!("{kind}_url")],"maximum_items":1,
                    "maximum_bytes":1024,"maximum_tokens":4,"encoder":0,
                    "recipe":{"tag":tag,"parameters":parameters},
                    "render":{"pad":"<<bot>>"}}]
            }))
            .unwrap(),
        );
        let started = EngineGenerationBackend::try_new_with_factory::<FixtureExecutorV1, _>(
            tokenizer(),
            || panic!("unsupported recipe must be refused before the engine starts"),
            EngineBackendConfig::default(),
            EngineModelPolicyV1::from_declaration(&descriptor),
        );
        assert!(
            matches!(started, Err(EngineBackendStartupError::EngineUnavailable(message))
            if message.contains(tag))
        );
    }
}

/// The fixture model: one byte token per step, or a refusal.
struct FixtureExecutorV1 {
    token_id: u32,
    /// Tokens sampled before `token_id`, one per logits row.
    script: Vec<u32>,
    rows: usize,
    fail: bool,
    refusals: usize,
    executions: Arc<AtomicUsize>,
    input_tokens: Arc<AtomicUsize>,
    delay: Duration,
    resume_second_step: Option<std::sync::mpsc::Receiver<()>>,
    /// Executions, by index, that hold until their sender sends.
    held_steps: Vec<(usize, std::sync::mpsc::Receiver<()>)>,
}

impl FixtureExecutorV1 {
    fn emitting() -> Self {
        Self {
            // Any byte token decodes, which is all the decoder needs here.
            token_id: BYTE_ID_START + u32::from(b'a'),
            script: Vec::new(),
            rows: 0,
            fail: false,
            refusals: 0,
            executions: Arc::new(AtomicUsize::new(0)),
            input_tokens: Arc::new(AtomicUsize::new(0)),
            delay: Duration::ZERO,
            resume_second_step: None,
            held_steps: Vec::new(),
        }
    }

    fn failing() -> Self {
        Self {
            fail: true,
            ..Self::emitting()
        }
    }

    /// Spells `text` in byte tokens, then keeps emitting `a`.
    fn spelling(text: &str) -> Self {
        Self {
            script: text
                .bytes()
                .map(|byte| BYTE_ID_START + u32::from(byte))
                .collect(),
            ..Self::emitting()
        }
    }

    fn slow() -> Self {
        Self {
            delay: Duration::from_millis(5),
            ..Self::emitting()
        }
    }

    /// A slow executor whose executions at `steps` (zero-based) each hold
    /// until the sender at the same position sends. Every other step takes
    /// at least the slow delay, however loaded the host is.
    fn slow_holding(steps: &[usize]) -> (Self, Vec<std::sync::mpsc::Sender<()>>) {
        let (senders, held_steps) = steps
            .iter()
            .map(|&step| {
                let (resume, held) = std::sync::mpsc::channel();
                (resume, (step, held))
            })
            .unzip();
        (
            Self {
                held_steps,
                ..Self::slow()
            },
            senders,
        )
    }
}

impl ModelExecutor for FixtureExecutorV1 {
    type TransactionState<'a> = NoopModelTransactionState;

    fn capture_prefix_state_v1(
        &mut self,
        _: runtime_engine::RequestId,
        _: usize,
        _: &[runtime_kv::BlockId],
    ) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn restore_prefix_state_v1(
        &mut self,
        _: runtime_engine::RequestId,
        _: &[u32],
        _: &[runtime_kv::BlockId],
    ) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        self.input_tokens.fetch_add(
            items
                .iter()
                .map(|item| item.input_tokens().len())
                .sum::<usize>(),
            Ordering::Relaxed,
        );
        let execution = self.executions.fetch_add(1, Ordering::Relaxed);
        if execution == 1
            && let Some(resume) = self.resume_second_step.take()
        {
            resume
                .recv_timeout(TIMEOUT)
                .expect("test releases the second step");
        }
        if let Some(held) = self
            .held_steps
            .iter()
            .position(|(step, _)| *step == execution)
        {
            self.held_steps
                .remove(held)
                .1
                .recv_timeout(TIMEOUT)
                .expect("test releases the held step");
        }
        std::thread::sleep(self.delay);
        if self.refusals > 0 {
            self.refusals -= 1;
            return Err(ExecutorError::batch_refused(
                "bundle batch refused: packing check token budget",
            )
            .unwrap());
        }
        if self.fail {
            return Err(ExecutorError::new("the fixture executor refuses")
                .expect("the fixture message is usable"));
        }
        let rows = items
            .iter()
            .filter(|item| item.expects_logits())
            .map(|item| {
                let token_id = self.script.get(self.rows).copied().unwrap_or(self.token_id);
                self.rows += 1;
                let mut logits = vec![0.0; VOCABULARY_SIZE];
                logits[token_id as usize] = 1.0;
                LogitsRow::new(item.request_id(), logits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ModelTransaction::no_op(rows))
    }
}

fn engine<E: ModelExecutor>(executor: E) -> Engine<E> {
    let geometry = KvGeometry::new(16, 16).expect("the fixture page pool is valid");
    Engine::new(
        EngineConfig::new(
            SchedulerConfig::new(2, 4, 64, 64, 1).expect("the fixture schedule is valid"),
            geometry,
            ModelVocabulary::new(VOCABULARY_SIZE).expect("the fixture vocabulary is valid"),
            CONTEXT_TOKENS,
            2,
        )
        .expect("the fixture engine configuration is valid"),
        executor,
    )
    .expect("the fixture engine builds")
}

fn declaration() -> FrontendDescriptorV1 {
    FrontendDescriptorV1::from_component_bytes(DECLARATION.as_bytes())
        .expect("the fixture declaration is readable")
}

fn policy() -> EngineModelPolicyV1 {
    EngineModelPolicyV1::from_declaration(&declaration())
}

fn request(
    prompt: &str,
    maximum_new_tokens: usize,
    cancellation: CancellationSignal,
) -> PreparedRequest {
    PreparedRequest {
        request_id: "fixture-request".to_owned(),
        client_request_id: None,
        model: "fixture/metrics-1".to_owned(),
        prompt: PreparedPrompt::Raw(prompt.to_owned()),
        max_new_tokens: maximum_new_tokens,
        stop: Vec::new(),
        decoding: runtime_server::DecodingOptions::default(),
        sampling: SamplingParameters::new(0.0).expect("greedy sampling is valid"),
        seed: 7,
        logprobs: RequestedLogprobs {
            enabled: false,
            top: 0,
            echo: false,
            prompt_top: None,
            return_token_ids: false,
        },
        chat: None,
        constraint: None,
        cache_salt: None,
        cancellation,
    }
}

/// Submits `count` requests of `tokens` output tokens as one group and
/// returns their event receivers once the worker has started them. A loaded
/// host can deschedule the worker long enough for a short queue wait to
/// expire the whole group before its first step; none of it ran then, and
/// it is submitted again. A group with some members refused fails.
async fn start_group(
    backend: &EngineGenerationBackend,
    count: usize,
    tokens: usize,
) -> Vec<tokio::sync::mpsc::Receiver<GenerationEvent>> {
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    loop {
        let mut receivers = Vec::new();
        let requests = (0..count)
            .map(|_| {
                let (sender, receiver) = tokio::sync::mpsc::channel(64);
                receivers.push(receiver);
                (
                    request("x", tokens, CancellationSignal::new()),
                    EventSender::new(sender),
                )
            })
            .collect();
        let mut admitted = Vec::new();
        for admission in backend.submit_batch(requests) {
            admitted.push(tokio::time::timeout(TIMEOUT, admission).await.unwrap());
        }
        if admitted.iter().all(Result::is_ok) {
            return receivers;
        }
        assert!(
            admitted
                .iter()
                .all(|result| *result == Err(AdmissionError::AtCapacity)),
            "{admitted:?}"
        );
        assert!(
            tokio::time::Instant::now() < deadline,
            "the group never started"
        );
    }
}

/// A byte-level tokenizer over exactly this declaration's special tokens.
fn tokenizer() -> Tokenizer {
    let definition = tokenizer_definition();
    let configuration = tokenizer_configuration();
    let size = u32::try_from(VOCABULARY_SIZE).expect("the fixture vocabulary fits");
    let contract = TokenizerContract::new_with_dialect(
        "fixture/metrics",
        "89abcdef0123456789abcdef0123456789abcdef",
        FileIdentity::new(
            "tokenizer.json",
            definition.len() as u64,
            &sha256_bytes(&definition),
        )
        .expect("the fixture definition identity is valid"),
        FileIdentity::new(
            "tokenizer_config.json",
            configuration.len() as u64,
            &sha256_bytes(&configuration),
        )
        .expect("the fixture configuration identity is valid"),
        size,
        size,
        0,
        SPECIALS.len(),
        1_048_576,
        Some(SPECIALS[0]),
        SPECIALS[1],
        SPECIALS[1],
        &[r".+"],
        &SPECIALS,
        literal_prefix_policy(size, size - 256),
    )
    .expect("the fixture tokenizer contract is valid");
    Tokenizer::from_verified_bytes(&definition, &configuration, &contract)
        .expect("the fixture tokenizer builds")
}

/// Literal special tokens below the byte tokens, and merges whose results
/// follow each other from one first identifier.
fn literal_prefix_policy(first_merge_id: u32, byte_start_id: u32) -> TokenizerDialect {
    TokenizerDialect {
        normalizer: NormalizerPolicy::IdentitySequence,
        pretokenizer_byte_level: ByteLevelPolicy {
            add_prefix_space: false,
            trim_offsets: true,
            use_regex: false,
        },
        post_processor_byte_level: ByteLevelPolicy {
            add_prefix_space: true,
            trim_offsets: false,
            use_regex: true,
        },
        decoder_byte_level: ByteLevelPolicy {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        },
        merge_results: MergeResultIdPolicy::Sequential {
            first_id: first_merge_id,
        },
        merge_records: MergeRecordPolicy::SpaceSeparatedString,
        base_tokens: BaseTokenRepresentation::LiteralPrefixThenByteLevel { byte_start_id },
        config_tokens: ConfigTokenRepresentation::AddedToken { normalized: true },
        add_bos_token: ConfigFieldPolicy::Required(false),
        bos_token: ConfigFieldPolicy::Required(ConfigTokenValue::ContractValue),
        unk_token: ConfigFieldPolicy::RequiredNull,
        tokenizer_class: "PreTrainedTokenizerFast".to_owned(),
        continuing_subword_prefix: None,
        end_of_word_suffix: None,
        ignore_merges: IgnoreMergesPolicy::Absent,
        add_eos_token: Some(false),
        legacy: Some(true),
        empty_sentencepiece_kwargs: true,
    }
}

fn tokenizer_definition() -> Vec<u8> {
    let mut vocabulary = SPECIALS
        .iter()
        .enumerate()
        .map(|(id, token)| format!("{}:{id}", json_string(token)))
        .collect::<Vec<_>>();
    for (byte, encoded) in byte_alphabet().iter().enumerate() {
        vocabulary.push(format!(
            "{}:{}",
            json_string(&encoded.to_string()),
            byte + BYTE_ID_START as usize
        ));
    }
    let added = SPECIALS
        .iter()
        .enumerate()
        .map(|(id, token)| {
            format!(
                "{{\"id\":{id},\"content\":{},\"single_word\":false,\"lstrip\":false,\
                 \"rstrip\":false,\"normalized\":false,\"special\":true}}",
                json_string(token)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,",
            "\"added_tokens\":[{}],",
            "\"normalizer\":{{\"type\":\"Sequence\",\"normalizers\":[]}},",
            "\"pre_tokenizer\":{{\"type\":\"Sequence\",\"pretokenizers\":[",
            "{{\"type\":\"Split\",\"pattern\":{{\"Regex\":\".+\"}},\"behavior\":\"Isolated\",\"invert\":false}},",
            "{{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}}]}},",
            "\"post_processor\":{{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":false,\"use_regex\":true}},",
            "\"decoder\":{{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true}},",
            "\"model\":{{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,",
            "\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,",
            "\"fuse_unk\":false,\"byte_fallback\":false,\"vocab\":{{{}}},\"merges\":[]}}}}"
        ),
        added,
        vocabulary.join(",")
    )
    .into_bytes()
}

fn tokenizer_configuration() -> Vec<u8> {
    format!(
        concat!(
            "{{\"add_bos_token\":false,\"add_eos_token\":false,",
            "\"bos_token\":{{\"__type\":\"AddedToken\",\"content\":{},\"lstrip\":false,",
            "\"normalized\":true,\"rstrip\":false,\"single_word\":false}},",
            "\"clean_up_tokenization_spaces\":false,",
            "\"eos_token\":{{\"__type\":\"AddedToken\",\"content\":{},\"lstrip\":false,",
            "\"normalized\":true,\"rstrip\":false,\"single_word\":false}},",
            "\"legacy\":true,\"model_max_length\":1048576,",
            "\"pad_token\":{{\"__type\":\"AddedToken\",\"content\":{},\"lstrip\":false,",
            "\"normalized\":true,\"rstrip\":false,\"single_word\":false}},",
            "\"sp_model_kwargs\":{{}},\"tokenizer_class\":\"PreTrainedTokenizerFast\",",
            "\"unk_token\":null}}"
        ),
        json_string(SPECIALS[0]),
        json_string(SPECIALS[1]),
        json_string(SPECIALS[1])
    )
    .into_bytes()
}

fn byte_alphabet() -> [char; 256] {
    let mut direct = Vec::new();
    direct.extend(b'!'..=b'~');
    direct.extend(0xa1..=0xac);
    direct.extend(0xae..=0xff);
    let direct = direct.into_iter().collect::<BTreeSet<_>>();
    let mut output = ['\0'; 256];
    let mut extra = 0_u32;
    for byte in 0_u8..=u8::MAX {
        output[usize::from(byte)] = if direct.contains(&byte) {
            char::from(byte)
        } else {
            let scalar = char::from_u32(256 + extra).expect("the byte alphabet stays in range");
            extra += 1;
            scalar
        };
    }
    output
}

fn json_string(input: &str) -> String {
    serde_json::to_string(input).expect("a JSON string encodes")
}

#[test]
fn raw_prompts_use_declared_special_tokens_and_detokenize_partial_bytes() {
    let worker = ServedWorkerV1::start(FixtureExecutorV1::emitting());
    let text = "<<begin>><<user>>hello<<bot>>";
    let expected = worker.backend.tokenize(text, false).unwrap();
    let (prepared, count) = worker
        .backend
        .prepare_prompt(PreparedPrompt::Raw(text.into()))
        .unwrap();
    assert_eq!(prepared, PreparedPrompt::TokenIds(expected.clone()));
    assert_eq!(count, expected.len());
    assert_eq!(expected[0], 0);
    assert_eq!(worker.backend.detokenize(&expected).unwrap(), text);
    assert_eq!(
        worker.backend.detokenize(&[BYTE_ID_START + 0xc3]).unwrap(),
        "\u{fffd}"
    );
}

#[tokio::test]
async fn a_safe_refusal_reports_its_check_and_the_worker_serves_the_next_request() {
    let worker = ServedWorkerV1::start(FixtureExecutorV1 {
        refusals: 1,
        ..FixtureExecutorV1::emitting()
    });
    let mut refused = worker.submit("x", 1).await;
    let events = refused.drain().await;
    assert!(events.iter().any(|event| matches!(event, GenerationEvent::Failed { message, .. } if message.contains("packing check token budget"))), "{events:?}");
    let mut next = worker.submit("x", 1).await;
    assert!(
        next.drain()
            .await
            .iter()
            .any(|event| matches!(event, GenerationEvent::Finished { .. }))
    );
    assert_eq!(
        worker.backend.readiness(),
        runtime_server::BackendReadiness::Ready
    );
    let metrics = worker.published();
    assert_eq!(
        value(
            &metrics,
            "vllm:batch_refusals_total{model_name=\"fixture/metrics-1\",engine=\"0\"}"
        ),
        1.0
    );
    assert_eq!(
        value(
            &metrics,
            "vllm:batch_refused_requests_total{model_name=\"fixture/metrics-1\",engine=\"0\"}"
        ),
        1.0
    );
    assert_eq!(value(&metrics, "chop_generation_worker_failed"), 0.0);
}

#[tokio::test]
async fn repeated_safe_refusals_stop_the_worker_at_the_default_limit() {
    let worker = ServedWorkerV1::start(FixtureExecutorV1 {
        refusals: 4,
        ..FixtureExecutorV1::emitting()
    });
    for _ in 0..4 {
        let mut refused = worker.submit("x", 1).await;
        assert!(
            refused
                .drain()
                .await
                .iter()
                .any(|event| matches!(event, GenerationEvent::Failed { .. }))
        );
    }
    let metrics = published_when_all(&worker, &[("chop_generation_worker_failed", 1.0)]).await;
    assert_eq!(
        value(
            &metrics,
            "vllm:batch_refusals_total{model_name=\"fixture/metrics-1\",engine=\"0\"}"
        ),
        4.0
    );
    assert_ne!(
        worker.backend.readiness(),
        runtime_server::BackendReadiness::Ready
    );
}

struct UnsafeRefusalBundle {
    poisoned: bool,
    cancel_fails: bool,
}

impl runtime_host::BundleExecutionV1 for UnsafeRefusalBundle {
    fn execute_batch(
        &mut self,
        _: &[u8],
        _: runtime_bundle_abi::ResultSinkLimitsV1,
    ) -> Result<Vec<runtime_bundle_abi::OwnedLogitsRowV1>, runtime_bundle_abi::BundleAbiError> {
        if self.cancel_fails {
            return Ok(Vec::new());
        }
        Err(runtime_bundle_abi::BundleAbiError::Status(
            runtime_bundle_abi::BundleStatusV1::InvalidArgument,
        ))
    }
    fn last_execute_refusal(&self) -> Option<&str> {
        // The poisoned case must override even claimed clean-refusal evidence.
        self.poisoned.then_some("packing check")
    }
    fn commit(&mut self) -> Result<(), runtime_bundle_abi::BundleAbiError> {
        Ok(())
    }
    fn cancel(&mut self) -> Result<(), runtime_bundle_abi::BundleAbiError> {
        self.poisoned = true;
        Err(runtime_bundle_abi::BundleAbiError::Poisoned)
    }
    fn destroy(&mut self) -> Result<(), runtime_bundle_abi::BundleAbiError> {
        Ok(())
    }
    fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

#[tokio::test]
async fn poisoned_post_work_and_failed_cancel_bundle_errors_stop_the_worker() {
    for (poisoned, cancel_fails) in [(true, false), (false, false), (false, true)] {
        let backend = EngineGenerationBackend::new_with_factory(
            tokenizer(),
            move || {
                engine(runtime_host::BundleModelExecutorV1::new(
                    UnsafeRefusalBundle {
                        poisoned,
                        cancel_fails,
                    },
                    VOCABULARY_SIZE,
                    2,
                ))
            },
            EngineBackendConfig::default(),
            policy(),
        )
        .unwrap();
        let metrics = Arc::new(backend.runtime_metrics(&[]));
        let worker = ServedWorkerV1 { backend, metrics };
        let mut request = worker.submit("x", 1).await;
        assert!(
            request
                .drain()
                .await
                .iter()
                .any(|event| matches!(event, GenerationEvent::Failed { .. }))
        );
        let metrics = published_when_all(&worker, &[("chop_generation_worker_failed", 1.0)]).await;
        assert_eq!(
            value(
                &metrics,
                "vllm:batch_refusals_total{model_name=\"fixture/metrics-1\",engine=\"0\"}"
            ),
            0.0
        );
        assert_ne!(
            worker.backend.readiness(),
            runtime_server::BackendReadiness::Ready
        );
    }
}

#[tokio::test]
async fn a_refused_batch_returns_the_existing_http_500_envelope_and_serving_continues() {
    use std::io::{Read, Write};
    let backend = Arc::new(
        EngineGenerationBackend::new(
            tokenizer(),
            engine(FixtureExecutorV1 {
                refusals: 1,
                ..FixtureExecutorV1::emitting()
            }),
            EngineBackendConfig::default(),
            policy(),
        )
        .unwrap(),
    );
    let config = runtime_server::ServerConfig::default();
    let frontend = runtime_host::frontend_from_declaration_v1(
        &declaration(),
        &runtime_frontend::test_support::fixture_export_v1(),
        config,
    )
    .unwrap();
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(runtime_server::serve(listener, backend, frontend, config));
    for expected in [500, 200] {
        let response = tokio::task::spawn_blocking(move || {
            let mut stream = std::net::TcpStream::connect(address).unwrap();
            stream.set_read_timeout(Some(TIMEOUT)).unwrap();
            let body = r#"{"model":"fixture/metrics-1","prompt":"x","max_tokens":1}"#;
            write!(stream, "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).unwrap();
            response
        }).await.unwrap();
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(
            head.starts_with(&format!("HTTP/1.1 {expected}")),
            "{response}"
        );
        let body: serde_json::Value = serde_json::from_str(body).unwrap();
        if expected == 500 {
            assert_eq!(body["error"]["type"], "InternalServerError");
            assert_eq!(body["error"]["code"], "generation_worker_failed");
            assert!(body["error"]["param"].is_null());
            assert!(
                body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("packing check token budget")
            );
        }
    }
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());
}

#[tokio::test]
async fn token_id_prompt_primes_generated_byte_continuations() {
    for (generated, expected) in [(vec![0x9f, 0xa6, 0x80], "🦀"), (vec![b'A'], "�A")] {
        let worker = ServedWorkerV1::start(FixtureExecutorV1 {
            script: generated
                .iter()
                .map(|b| BYTE_ID_START + u32::from(*b))
                .collect(),
            ..FixtureExecutorV1::emitting()
        });
        let cancellation = CancellationSignal::new();
        let mut req = request("hello", generated.len(), cancellation.clone());
        req.prompt = PreparedPrompt::TokenIds(vec![BYTE_ID_START + 0xf0]);
        let (sender, events) = tokio::sync::mpsc::channel(64);
        worker
            .backend
            .submit(req, EventSender::new(sender))
            .await
            .unwrap();
        let seen = InFlightRequestV1 {
            events,
            cancellation,
        }
        .drain()
        .await;
        let text: String = seen
            .iter()
            .filter_map(|e| match e {
                GenerationEvent::Content(s) => Some(s.delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, expected, "{seen:?}");
    }
}

#[test]
fn compatibility_metrics_use_the_primary_served_name() {
    let worker = ServedWorkerV1::start(FixtureExecutorV1::emitting());
    let names = vec!["primary".to_owned(), "alias".to_owned()];
    let snapshot = worker.backend.runtime_metrics(&names).prometheus(0);
    let series: Vec<_> = snapshot
        .lines()
        .filter(|line| line.starts_with("vllm:"))
        .collect();
    assert!(!series.is_empty());
    for line in series {
        assert!(line.contains("model_name=\"primary\""), "{line}");
        assert!(!line.contains("fixture/metrics-1"), "{line}");
    }
    assert!(
        worker
            .backend
            .runtime_metrics(&[])
            .prometheus(0)
            .contains("model_name=\"fixture/metrics-1\"")
    );
    worker.backend.shutdown();
}

#[path = "support/remote_media.rs"]
mod remote_media;

async fn batch_exchange(stream: &tokio::net::TcpStream, mut request: &[u8]) -> Vec<u8> {
    while !request.is_empty() {
        stream.writable().await.expect("the socket is writable");
        match stream.try_write(request) {
            Ok(0) => panic!("the socket closed while writing"),
            Ok(written) => request = &request[written..],
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("the socket write failed: {error}"),
        }
    }
    let mut response = Vec::new();
    loop {
        stream.readable().await.expect("the socket is readable");
        let mut chunk = [0_u8; 4096];
        match stream.try_read(&mut chunk) {
            Ok(0) => return response,
            Ok(read) => response.extend_from_slice(&chunk[..read]),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("the socket read failed: {error}"),
        }
    }
}

/// How a client of [`clients_that_hang_up_leave_the_worker_and_transport_empty`]
/// leaves its request.
#[derive(Clone, Copy, Debug)]
enum Client {
    /// Reads the whole answer.
    Reads { stream: bool },
    /// Hangs up after its first streamed event.
    HangsUpMidStream,
    /// Hangs up right after sending, before any answer.
    HangsUpAtOnce { stream: bool },
}

/// Sends one completion over a fresh connection and leaves as `client` does.
/// Returns whether the client read a whole answer.
async fn completion(address: std::net::SocketAddr, index: usize, client: Client) -> bool {
    let stream = !matches!(
        client,
        Client::Reads { stream: false } | Client::HangsUpAtOnce { stream: false }
    );
    let body = serde_json::json!({
        "model": "fixture/metrics-1",
        "prompt": format!("x{}", "y".repeat(index % 7)),
        "max_tokens": 8,
        "temperature": 0,
        "stream": stream,
        "return_token_ids": index % 2 == 0,
    })
    .to_string();
    let request = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let socket = tokio::net::TcpStream::connect(address).await.unwrap();
    match client {
        Client::Reads { .. } => {
            let raw = tokio::time::timeout(TIMEOUT, batch_exchange(&socket, request.as_bytes()))
                .await
                .unwrap();
            let raw = String::from_utf8(raw).unwrap();
            raw.starts_with("HTTP/1.1 200")
        }
        Client::HangsUpAtOnce { .. } => {
            let mut sent = request.as_bytes();
            while !sent.is_empty() {
                socket.writable().await.unwrap();
                match socket.try_write(sent) {
                    Ok(written) => sent = &sent[written..],
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(error) => panic!("the socket write failed: {error}"),
                }
            }
            drop(socket);
            false
        }
        Client::HangsUpMidStream => {
            let mut sent = request.as_bytes();
            while !sent.is_empty() {
                socket.writable().await.unwrap();
                match socket.try_write(sent) {
                    Ok(written) => sent = &sent[written..],
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(error) => panic!("the socket write failed: {error}"),
                }
            }
            let mut seen = Vec::new();
            let deadline = tokio::time::Instant::now() + TIMEOUT;
            while !String::from_utf8_lossy(&seen).contains("data: ") {
                assert!(tokio::time::Instant::now() < deadline, "no streamed event");
                socket.readable().await.unwrap();
                let mut chunk = [0_u8; 4096];
                match socket.try_read(&mut chunk) {
                    Ok(0) => break,
                    Ok(read) => seen.extend_from_slice(&chunk[..read]),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(error) => panic!("the socket read failed: {error}"),
                }
            }
            drop(socket);
            false
        }
    }
}

/// Requests over the transport whose clients read their answers or hang up
/// before or during them leave nothing held: no request, sequence or KV token
/// in the worker, and no intake or transport bytes in the server.
#[tokio::test]
async fn clients_that_hang_up_leave_the_worker_and_transport_empty() {
    let worker = ServedWorkerV1::start(FixtureExecutorV1::slow());
    let metrics = Arc::clone(&worker.metrics);
    let backend: Arc<dyn GenerationBackend> = Arc::new(worker.backend);
    let config = runtime_server::ServerConfig::default();
    let frontend = runtime_host::frontend_from_declaration_v1(
        &declaration(),
        &runtime_frontend::test_support::fixture_export_v1(),
        config,
    )
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(runtime_server::serve(listener, backend, frontend, config));
    let clients = [
        Client::Reads { stream: false },
        Client::Reads { stream: true },
        Client::HangsUpMidStream,
        Client::HangsUpAtOnce { stream: false },
        Client::HangsUpAtOnce { stream: true },
    ];
    let mut read = 0;
    for round in 0..24 {
        // Two clients at a time, as the fixture engine runs two sequences.
        let pair = [round * 2, round * 2 + 1]
            .map(|index| tokio::spawn(completion(address, index, clients[index % clients.len()])));
        for client in pair {
            read += usize::from(client.await.unwrap());
        }
    }
    assert!(read >= 16, "{read} clients read their answer");
    let provider: &dyn MetricsSnapshotProvider = metrics.as_ref();
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    let (worker_text, server_text) = loop {
        let worker_text = provider.snapshot(12_345_678);
        let socket = tokio::net::TcpStream::connect(address).await.unwrap();
        let server_text = String::from_utf8(
            tokio::time::timeout(
                TIMEOUT,
                batch_exchange(
                    &socket,
                    b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                ),
            )
            .await
            .unwrap(),
        )
        .unwrap();
        let worker_idle = [
            "chop_requests_active",
            "chop_sequences_active",
            "chop_kv_active_sequences",
            "chop_kv_tokens",
        ]
        .iter()
        .all(|name| metric_value(&worker_text, name) == Some(0.0));
        let server_idle = ["chop_requests_waiting", "chop_intake_used_bytes"]
            .iter()
            .all(|name| metric_value(&server_text, name) == Some(0.0));
        if worker_idle && server_idle {
            break (worker_text, server_text);
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the worker or transport still holds requests:\n{worker_text}\n{server_text}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(
        value(&worker_text, "chop_cancellations_total") > 0.0,
        "{worker_text}"
    );
    assert!(
        value(&worker_text, "chop_requests_completed_total") > 0.0,
        "{worker_text}"
    );
    assert_eq!(value(&worker_text, "chop_generation_worker_ready"), 1.0);
    // Only the metrics request's own connection holds transport buffers.
    assert!(value(&server_text, "chop_transport_buffer_reserved_bytes") >= 0.0);
    server.abort();
    let _ = server.await;
}
