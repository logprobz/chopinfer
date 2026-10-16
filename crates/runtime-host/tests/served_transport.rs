#![forbid(unsafe_code)]

//! Serving one selected bundle's declared frontend over the real transport.
//!
//! What `serve` binds is `runtime-server`, and what it answers with comes out
//! of the package the selection verified. This drives that half end to end: a
//! package is written, verified, and read for its declaration through the same
//! two functions the serving command calls, and the transport is then bound on
//! a loopback port and asked what it serves.
//!
//! The generation half needs a bundle whose native module loads, which is what
//! the external-host acceptance path provides. Nothing here fakes a module:
//! the backend below stands in for generation only, and every served identity,
//! limit, and context number is the installed declaration's.

use std::fs;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use runtime_bundle::{
    BundleComponentV1, BundleGraphBucketV1, BundleHostCompatibilityV1, BundleKvLayoutV1,
    BundleManifestV1, BundleOpaqueId, BundlePackageV1, BundleRankPlacementV1, BundleRelativePathV1,
    BundleResourceRequirementsV1, BundleStateLayoutV1, BundleStateLifetimeV1, BundleTargetV1,
    BundleTopologyV1,
};
use runtime_frontend::test_support::fixture_export_v1;
use runtime_host::{
    declaration_from_package_v1, deployment_from_package_v1, frontend_from_declaration_v1,
};
use runtime_server::{
    AdmissionError, BackendReadiness, BoxFuture, EventSender, GenerationBackend, PreparedPrompt,
    PreparedRequest, ServerConfig, ServerFrontendV1, serve,
};
use runtime_weights::sha256_bytes;
use serde_json::Value;
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

const BUNDLE_ID: &str = "fixture.served.v1";
const ARCHITECTURE: &str = "sm_120f";
const FRONTEND_ROLE: &str = "chop.frontend.openai.v1";
const FRONTEND_PATH: &str = "components/frontend.json";
const MODEL: &str = "fixture/served-3";
const DECLARED_CONTEXT_TOKENS: usize = 48;

/// The frontend declaration this fixture bundle ships.
///
/// Every served answer below is somewhere in here: the identity `/v1/models`
/// lists, the owner it names, and the context a request is charged against.
const DECLARATION: &str = r##"{
  "schema": "chop-frontend-openai-v2",
  "version": 2,
  "models": [{"id": "fixture/served-3", "owned_by": "fixture-served"}],
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
    "repository": "fixture/served",
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
    "base_vocabulary_size": 300,
    "merge_count": 8,
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
    "size": 777,
    "bos_token_id": 5,
    "eos_token_id": 6,
    "stop_token_ids": [6],
    "boundary_tokens": [
      {"text": "<<begin>>", "id": 5},
      {"text": "<<end>>", "id": 6},
      {"text": "<<user>>", "id": 8},
      {"text": "<<bot>>", "id": 9},
      {"text": "<<reason>>", "id": 10},
      {"text": "<</reason>>", "id": 11},
      {"text": "<<calls>>", "id": 12}
    ],
    "prompt_special_tokens": ["<<begin>>", "<<end>>", "<<reason>>", "<</reason>>"]
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

/// The transport binds and answers out of the installed declaration.
///
/// Nothing about this deployment was compiled in. The identity it lists, the
/// owner beside it, and the context it charges against all came from the
/// component the package authenticated.
#[tokio::test]
async fn a_bound_transport_serves_the_declaration_its_bundle_shipped() {
    let fixture = FixtureV1::new();
    let package = fixture.verified();

    // The two functions the serving command calls, in the order it calls them.
    let declaration = declaration_from_package_v1(&package).expect("the declaration reads");
    let frontend =
        frontend_from_declaration_v1(&declaration, &fixture_export_v1(), ServerConfig::default())
            .expect("the declaration serves");
    assert_eq!(frontend.maximum_total_tokens(), DECLARED_CONTEXT_TOKENS);

    let server = TestServerV1::start(frontend).await;

    let health = server.get("/health").await;
    assert_eq!(health.status, 200, "{}", health.body);

    assert_eq!(server.get("/ping").await.status, 200);
    let version = server.get("/version").await;
    assert_eq!(version.status, 200);
    assert!(version.json()["version"].is_string());
    let models = server.get("/v1/models").await;
    assert_eq!(models.status, 200, "{}", models.body);
    let listed = models.json();
    let entries = listed["data"].as_array().expect("models are listed");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["id"], MODEL);
    assert_eq!(entries[0]["owned_by"], "fixture-served");
    assert_eq!(entries[0]["max_model_len"], DECLARED_CONTEXT_TOKENS);

    // A model this bundle never declared is not served by this deployment.
    let refused = server
        .post(
            "/v1/completions",
            r#"{"model":"fixture/absent","prompt":"hello","max_tokens":1}"#,
        )
        .await;
    assert_eq!(refused.status, 404, "{}", refused.body);
}

/// A package that does not require `chop.frontend.v1` declares no frontend
/// export, and this host has no format of its own to serve in its place.
#[test]
fn a_package_without_the_frontend_capability_is_not_deployed() {
    let fixture = FixtureV1::new();
    let package = fixture.verified();
    let error = deployment_from_package_v1(&package, &fixture_export_v1(), ServerConfig::default())
        .err()
        .expect("the package declares no frontend")
        .to_string();
    assert_eq!(
        error,
        "the package does not require chop.frontend.v1, so it serves no chat frontend; rebuild \
         it with a compiler that exports one"
    );
}

/// A deployment may serve less context than the bundle declared, never more.
#[tokio::test]
async fn a_deployment_narrows_the_declared_context_and_never_widens_it() {
    let fixture = FixtureV1::new();
    let package = fixture.verified();
    let declaration = declaration_from_package_v1(&package).expect("the declaration reads");

    let narrowed = frontend_from_declaration_v1(
        &declaration,
        &fixture_export_v1(),
        ServerConfig::default().with_maximum_context_tokens(16),
    )
    .expect("a narrowed deployment serves");
    let widened = frontend_from_declaration_v1(
        &declaration,
        &fixture_export_v1(),
        ServerConfig::default().with_maximum_context_tokens(DECLARED_CONTEXT_TOKENS * 4),
    )
    .expect("a widened deployment serves");

    assert_eq!(narrowed.maximum_total_tokens(), 16);
    assert_eq!(widened.maximum_total_tokens(), DECLARED_CONTEXT_TOKENS);
}

/// One bundle directory carrying one declared frontend component.
struct FixtureV1 {
    root: TemporaryDirectoryV1,
}

impl FixtureV1 {
    fn new() -> Self {
        let root = TemporaryDirectoryV1::new("runtime-host-served-transport");
        let package = root.path().join(BUNDLE_ID);
        fs::create_dir_all(package.join("components"))
            .expect("the fixture component directory is creatable");
        fs::write(package.join(FRONTEND_PATH), DECLARATION.as_bytes())
            .expect("the fixture declaration is writable");
        fs::write(
            package.join("manifest.json"),
            manifest()
                .to_canonical_json()
                .expect("the fixture manifest serializes"),
        )
        .expect("the fixture manifest is writable");
        Self { root }
    }

    fn verified(&self) -> BundlePackageV1 {
        BundlePackageV1::verify_directory(
            self.root.path().join(BUNDLE_ID),
            &BundleHostCompatibilityV1::new(ARCHITECTURE, 1).expect("the host descriptor is valid"),
        )
        .expect("the fixture package verifies")
    }
}

fn id(value: &str) -> BundleOpaqueId {
    BundleOpaqueId::new(value).expect("the fixture identifier is valid")
}

fn manifest() -> BundleManifestV1 {
    BundleManifestV1::new(
        id(BUNDLE_ID),
        BundleTargetV1::new(
            vec![id(ARCHITECTURE)],
            vec![BundleTopologyV1::new(id("tp1"), 1).expect("tp1 is valid")],
        )
        .expect("the fixture target is valid"),
        vec![
            BundleComponentV1::new(
                id("frontend"),
                id(FRONTEND_ROLE),
                id("chop.frontend.json.v1"),
                BundleRelativePathV1::new(FRONTEND_PATH).expect("the fixture path is relative"),
                DECLARATION.len() as u64,
                sha256_bytes(DECLARATION.as_bytes()),
            )
            .expect("the fixture component is valid"),
        ],
        BundleResourceRequirementsV1::new(1, Vec::new()).expect("the requirements are valid"),
        BundleKvLayoutV1::new(
            id("fixture.kv.v1"),
            64,
            64,
            1,
            64,
            1,
            1024,
            BundleRankPlacementV1::Replicated,
            BundleStateLifetimeV1::Transaction,
        )
        .expect("the fixture KV layout is valid"),
        BundleStateLayoutV1::new(
            id("fixture.state.v1"),
            64,
            64,
            BundleStateLifetimeV1::Request,
        )
        .expect("the fixture state layout is valid"),
        vec![
            BundleGraphBucketV1::new(id("bucket.one"), vec![id("frontend")], Vec::new())
                .expect("the fixture bucket is valid"),
        ],
        Vec::new(),
        Vec::new(),
        None,
    )
    .expect("the fixture manifest is valid")
}

/// A backend that admits nothing.
///
/// Generation belongs to the loaded bundle, and this test has no module to
/// load. What it stands for is the transport's other half being present and
/// ready, so the routes that answer out of the declaration can be reached.
struct ReadyBackendV1;

impl GenerationBackend for ReadyBackendV1 {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }

    fn maximum_sequence_tokens(&self) -> usize {
        DECLARED_CONTEXT_TOKENS
    }

    fn prompt_token_count(&self, prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
        let text = match prompt {
            PreparedPrompt::Raw(text) | PreparedPrompt::RenderedChat(text) => text,
            PreparedPrompt::TokenIds(_) => return Err(AdmissionError::InvalidPrompt),
            PreparedPrompt::MediaChat { .. } | PreparedPrompt::MediaTokens { .. } => {
                return Err(AdmissionError::InvalidPrompt);
            }
        };
        Ok(text.split_whitespace().count().max(1))
    }

    fn submit(
        &self,
        _request: PreparedRequest,
        _events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        Box::pin(async { Err(AdmissionError::WorkerUnavailable) })
    }
}

struct TestServerV1 {
    address: SocketAddr,
}

impl TestServerV1 {
    async fn start(frontend: Arc<dyn ServerFrontendV1>) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("a loopback port is available");
        let address = listener.local_addr().expect("the listener has an address");
        tokio::spawn(async move {
            let _ = serve(
                listener,
                Arc::new(ReadyBackendV1),
                frontend,
                ServerConfig::default(),
            )
            .await;
        });
        Self { address }
    }

    async fn get(&self, path: &str) -> ResponseV1 {
        self.request(&format!(
            "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        ))
        .await
    }

    async fn post(&self, path: &str, body: &str) -> ResponseV1 {
        self.request(&format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ))
        .await
    }

    async fn request(&self, request: &str) -> ResponseV1 {
        let stream = TcpStream::connect(self.address)
            .await
            .expect("the server accepts connections");
        let raw = timeout(
            Duration::from_secs(5),
            exchange(&stream, request.as_bytes()),
        )
        .await
        .expect("the response arrives");
        let text = String::from_utf8_lossy(&raw).into_owned();
        let (head, body) = text.split_once("\r\n\r\n").unwrap_or((text.as_str(), ""));
        ResponseV1 {
            status: head
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|code| code.parse().ok())
                .unwrap_or(0),
            body: body.to_owned(),
        }
    }
}

async fn exchange(stream: &TcpStream, mut request: &[u8]) -> Vec<u8> {
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

struct ResponseV1 {
    status: u16,
    body: String,
}

impl ResponseV1 {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("the response body is JSON")
    }
}

/// A scratch directory this test owns and removes.
struct TemporaryDirectoryV1 {
    path: PathBuf,
}

impl TemporaryDirectoryV1 {
    fn new(prefix: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "{prefix}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("the scratch directory is creatable");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryDirectoryV1 {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
