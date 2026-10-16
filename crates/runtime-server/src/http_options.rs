//! Deployment HTTP policy, independent of generation admission.
use sha2::{Digest, Sha256};
use std::sync::Arc;

use axum::{
    Router,
    extract::{Request, State},
    http::{HeaderValue, Method, header},
    middleware::Next,
    response::{IntoResponse, Response},
};

/// Stores only a digest and never formats credential material.
#[derive(Clone, Eq, PartialEq)]
pub struct ApiKey([u8; 32]);

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey([redacted])")
    }
}

impl ApiKey {
    pub fn new(value: &str) -> Self {
        Self(Sha256::digest(value.as_bytes()).into())
    }

    fn matches(&self, value: &str) -> bool {
        let candidate: [u8; 32] = Sha256::digest(value.as_bytes()).into();
        // Fixed-length digests, with every byte visited regardless of mismatch.
        // Black boxes prevent replacing this reduction with an early-exit comparison.
        let mut difference = 0_u8;
        for (left, right) in self.0.iter().zip(candidate) {
            difference |= std::hint::black_box(*left) ^ std::hint::black_box(right);
        }
        std::hint::black_box(difference) == 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpOptions {
    pub served_model_names: Vec<String>,
    pub api_key: Option<ApiKey>,
    pub enable_request_id_headers: bool,
    /// Install browser CORS policy only after explicit operator opt-in.
    pub enable_cors: bool,
    pub allowed_origins: Vec<String>,
    pub allowed_methods: Vec<String>,
    pub allowed_headers: Vec<String>,
    pub allow_credentials: bool,
    /// What `/health` reports this deployment serves. Without it, health
    /// reports readiness alone.
    pub serving_identity: Option<crate::ServingIdentity>,
}

impl Default for HttpOptions {
    fn default() -> Self {
        Self {
            served_model_names: Vec::new(),
            api_key: None,
            enable_request_id_headers: false,
            enable_cors: false,
            allowed_origins: vec!["*".into()],
            allowed_methods: vec!["*".into()],
            allowed_headers: vec!["*".into()],
            allow_credentials: false,
            serving_identity: None,
        }
    }
}

impl HttpOptions {
    pub fn validate(&self) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        for name in &self.served_model_names {
            if name.trim().is_empty()
                || name.len() > 512
                || name.chars().any(char::is_control)
                || !seen.insert(name)
            {
                return Err("served model names must be unique, nonempty, at most 512 bytes, and contain no control characters".into());
            }
        }
        for value in self
            .allowed_origins
            .iter()
            .chain(&self.allowed_methods)
            .chain(&self.allowed_headers)
        {
            HeaderValue::from_str(value).map_err(|_| "invalid CORS header value")?;
        }
        for method in &self.allowed_methods {
            if method != "*" {
                Method::from_bytes(method.as_bytes()).map_err(|_| "invalid allowed method")?;
            }
        }
        for name in &self.allowed_headers {
            if name != "*" {
                header::HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| "invalid allowed header")?;
            }
        }
        Ok(())
    }

    pub fn layer(self, router: Router) -> Router {
        router.layer(axum::middleware::from_fn_with_state(
            Arc::new(self),
            http_policy,
        ))
    }
}

fn wildcard(values: &[String]) -> bool {
    values.iter().any(|value| value == "*")
}

fn put(response: &mut Response, name: &'static str, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        response.headers_mut().insert(name, value);
    }
}

#[derive(Clone)]
pub(crate) struct HttpRequestId(pub(crate) String);

async fn http_policy(
    State(options): State<Arc<HttpOptions>>,
    mut request: Request,
    next: Next,
) -> Response {
    let id = options.enable_request_id_headers.then(|| {
        request
            .headers()
            .get("x-request-id")
            .filter(|value| {
                let bytes = value.as_bytes();
                !bytes.is_empty()
                    && bytes.len() <= 512
                    && bytes.iter().all(|byte| (0x20..=0x7e).contains(byte))
            })
            .cloned()
            .unwrap_or_else(|| {
                HeaderValue::from_str(&crate::response::next_request_id())
                    .expect("generated request ID is ASCII")
            })
    });
    if let Some(value) = id.as_ref().and_then(|value| value.to_str().ok()) {
        request
            .extensions_mut()
            .insert(HttpRequestId(value.to_owned()));
    }
    let mut response = cors(State(options.clone()), request, next).await;
    if let Some(id) = id {
        response.headers_mut().insert("x-request-id", id);
    }
    // Every response names the process that wrote it, so a client can tie
    // each scored response, not only `/health`, to the identity it bound.
    if let Some(identity) = &options.serving_identity {
        put(&mut response, STARTUP_NONCE_HEADER, &identity.startup_nonce);
        if let Some(checkpoint) = &identity.checkpoint_identity {
            put(&mut response, CHECKPOINT_IDENTITY_HEADER, checkpoint);
        }
    }
    response
}

/// The response header carrying the serving identity's startup nonce.
pub const STARTUP_NONCE_HEADER: &str = "x-chopinfer-startup-nonce";
/// The response header carrying the serving identity's checkpoint identity.
pub const CHECKPOINT_IDENTITY_HEADER: &str = "x-chopinfer-checkpoint-identity";

async fn cors(State(options): State<Arc<HttpOptions>>, request: Request, next: Next) -> Response {
    if !options.enable_cors {
        return authenticate(&options, request, next).await;
    }
    let Some(origin) = request.headers().get(header::ORIGIN).cloned() else {
        return authenticate(&options, request, next).await;
    };
    let all_origins = wildcard(&options.allowed_origins);
    let allowed_origin = all_origins
        || options
            .allowed_origins
            .iter()
            .any(|allowed| origin == allowed.as_str());
    let preflight = request.method() == Method::OPTIONS
        && request
            .headers()
            .contains_key(header::ACCESS_CONTROL_REQUEST_METHOD);
    let cookie = request.headers().contains_key(header::COOKIE);
    let mut response = if preflight {
        let method = request
            .headers()
            .get(header::ACCESS_CONTROL_REQUEST_METHOD)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        let methods = if wildcard(&options.allowed_methods) {
            "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT".to_owned()
        } else {
            options.allowed_methods.join(", ")
        };
        let allowed_method = methods.split(", ").any(|allowed| allowed == method);
        let requested_headers = request
            .headers()
            .get(header::ACCESS_CONTROL_REQUEST_HEADERS)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        let mut headers = vec![
            "Accept".to_owned(),
            "Accept-Language".into(),
            "Content-Language".into(),
            "Content-Type".into(),
        ];
        headers.extend(options.allowed_headers.iter().cloned());
        let allowed_headers = wildcard(&options.allowed_headers)
            || requested_headers.split(',').all(|name| {
                name.trim().is_empty()
                    || headers
                        .iter()
                        .any(|allowed| allowed.eq_ignore_ascii_case(name.trim()))
            });
        let mut failures = Vec::new();
        if !allowed_origin {
            failures.push("origin");
        }
        if !allowed_method {
            failures.push("method");
        }
        if !allowed_headers {
            failures.push("headers");
        }
        let mut response = if failures.is_empty() {
            "OK".into_response()
        } else {
            (
                axum::http::StatusCode::BAD_REQUEST,
                format!("Disallowed CORS {}", failures.join(", ")),
            )
                .into_response()
        };
        put(&mut response, "access-control-allow-methods", &methods);
        put(&mut response, "access-control-max-age", "600");
        put(
            &mut response,
            "access-control-allow-headers",
            &if wildcard(&options.allowed_headers) {
                requested_headers.to_owned()
            } else {
                headers.join(", ")
            },
        );
        response
    } else {
        authenticate(&options, request, next).await
    };
    if options.allow_credentials {
        put(&mut response, "access-control-allow-credentials", "true");
    }
    let explicit_origin =
        !all_origins || (preflight && options.allow_credentials) || (!preflight && cookie);
    if allowed_origin {
        response.headers_mut().insert(
            header::ACCESS_CONTROL_ALLOW_ORIGIN,
            if explicit_origin {
                origin
            } else {
                HeaderValue::from_static("*")
            },
        );
    }
    if explicit_origin {
        response
            .headers_mut()
            .append(header::VARY, HeaderValue::from_static("Origin"));
    }
    response
}

async fn authenticate(options: &HttpOptions, request: Request, next: Next) -> Response {
    if let Some(key) = &options.api_key {
        let guarded = ["/v1", "/v2", "/inference", "/cohere"]
            .iter()
            .any(|prefix| request.uri().path().starts_with(prefix));
        let authorized = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split_once(' '))
            .is_some_and(|(scheme, token)| {
                scheme.eq_ignore_ascii_case("bearer") && key.matches(token)
            });
        if guarded && request.method() != Method::OPTIONS && !authorized {
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({"error": "Unauthorized"})),
            )
                .into_response();
        }
    }
    next.run(request).await
}
