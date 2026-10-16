//! What `/health` reports: readiness, waiting requests and the serving
//! identity (`ServingIdentity` and `ServingProfile` in
//! `runtime-server/src/identity.rs`).
//!
//! Every field is optional here. A server that reports less still appears,
//! with `-` where a field is missing.

use serde_json::Value;

/// The header every chopinfer response carries; an endpoint without it is
/// not treated as a chopinfer server.
pub const NONCE_HEADER: &str = "x-chopinfer-startup-nonce";

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Health {
    /// `ready` or `not_ready`.
    pub status: String,
    pub reason: Option<String>,
    pub waiting_total: Option<u64>,
    /// Waiting requests by stage, in the order reported.
    pub waiting_stages: Vec<(String, u64)>,
    pub serving: Option<Serving>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Serving {
    pub served_model_names: Vec<String>,
    pub bundle_id: Option<String>,
    pub startup_nonce: Option<String>,
    pub checkpoint_identity: Option<String>,
    pub version: Option<String>,
    pub source_commit: Option<String>,
    pub kv_page_tokens: Option<u64>,
    pub profile: Profile,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Profile {
    pub name: Option<String>,
    pub tensor_parallel_size: Option<u64>,
    pub context_parallel_size: Option<u64>,
    pub speculation: Option<Speculation>,
    pub kv_cache_dtype: Option<String>,
    pub prefix_cache: Option<bool>,
    pub max_sequences: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Speculation {
    /// `off`, `fixed` or `adaptive`.
    pub mode: String,
    pub tokens: u64,
    pub drafter: Option<String>,
}

const STAGES: [&str; 6] = [
    "transport",
    "intake",
    "preparation",
    "staging",
    "storage",
    "backend",
];

impl Health {
    /// Parses a `/health` body; `None` when it is not a JSON object.
    pub fn parse(body: &[u8]) -> Option<Self> {
        let value: Value = serde_json::from_slice(body).ok()?;
        let object = value.as_object()?;
        let text = |value: &Value| value.as_str().map(str::to_owned);
        let waiting = &value["waiting"];
        Some(Self {
            status: text(&value["status"]).unwrap_or_default(),
            reason: text(&value["reason"]),
            waiting_total: waiting["total"]
                .as_u64()
                .or_else(|| value["queue_depth"].as_u64()),
            waiting_stages: STAGES
                .iter()
                .filter_map(|stage| Some(((*stage).to_owned(), waiting[*stage].as_u64()?)))
                .collect(),
            serving: object
                .get("serving")
                .filter(|serving| serving.is_object())
                .map(|serving| {
                    let profile = &serving["profile"];
                    let speculation = &profile["speculation"];
                    Serving {
                        served_model_names: serving["served_model_names"]
                            .as_array()
                            .map(|names| names.iter().filter_map(text).collect())
                            .unwrap_or_default(),
                        bundle_id: text(&serving["bundle_id"]),
                        startup_nonce: text(&serving["startup_nonce"]),
                        checkpoint_identity: text(&serving["checkpoint_identity"]),
                        version: text(&serving["version"]),
                        source_commit: text(&serving["source_commit"]),
                        kv_page_tokens: serving["kv_page_tokens"].as_u64(),
                        profile: Profile {
                            name: text(&profile["name"]),
                            tensor_parallel_size: profile["tensor_parallel_size"].as_u64(),
                            context_parallel_size: profile["context_parallel_size"].as_u64(),
                            speculation: speculation["mode"].as_str().map(|mode| Speculation {
                                mode: mode.to_owned(),
                                tokens: speculation["tokens"].as_u64().unwrap_or(0),
                                drafter: text(&speculation["drafter"]),
                            }),
                            kv_cache_dtype: text(&profile["kv_cache_dtype"]),
                            prefix_cache: profile["prefix_cache"].as_bool(),
                            max_sequences: profile["max_sequences"].as_u64(),
                        },
                    }
                }),
        })
    }

    pub fn ready(&self) -> bool {
        self.status == "ready"
    }

    /// The first served model name, the name the server is shown by.
    pub fn model_name(&self) -> Option<&str> {
        self.serving
            .as_ref()
            .and_then(|serving| serving.served_model_names.first())
            .map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A body in the layout `health_member` serializes.
    const READY: &str = r#"{
        "status": "ready",
        "transport": {"buffer_budget_bytes": 536870912, "buffer_reserved_bytes": 1048576},
        "waiting": {"total": 3, "pre_backend": 1, "transport": 0, "intake": 1, "preparation": 0,
                    "staging": 0, "storage": 0, "backend": 2},
        "queue_depth": 3,
        "modalities": ["text"],
        "serving": {
            "bundle_id": "chopinfer.fixture.tp4",
            "package_identity": null,
            "checkpoint_identity": "3fa91c07d2aa",
            "metadata_identity": null,
            "startup_nonce": "7d2e04b1c9aa7d2e04b1c9aa7d2e04b1",
            "profile": {
                "name": "tp4-mtp",
                "tensor_parallel_size": 4,
                "context_parallel_size": 2,
                "speculation": {"mode": "adaptive", "tokens": 3, "drafter": "mtp"},
                "kv_cache_dtype": "fp8",
                "prefix_cache": true,
                "max_sequences": 32
            },
            "kv_page_tokens": 64,
            "version": "0.1.0-alpha.2",
            "source_commit": "4c1e9a2",
            "served_model_names": ["Qwen3.8-Flash", "alias"]
        }
    }"#;

    #[test]
    fn reads_the_nested_serving_identity() {
        let health = Health::parse(READY.as_bytes()).unwrap();
        assert!(health.ready());
        assert_eq!(health.waiting_total, Some(3));
        assert_eq!(health.waiting_stages.len(), 6);
        assert_eq!(health.waiting_stages[5], ("backend".to_owned(), 2));
        assert_eq!(health.model_name(), Some("Qwen3.8-Flash"));
        let serving = health.serving.unwrap();
        assert_eq!(serving.bundle_id.as_deref(), Some("chopinfer.fixture.tp4"));
        assert_eq!(serving.kv_page_tokens, Some(64));
        assert_eq!(
            serving.profile,
            Profile {
                name: Some("tp4-mtp".to_owned()),
                tensor_parallel_size: Some(4),
                context_parallel_size: Some(2),
                speculation: Some(Speculation {
                    mode: "adaptive".to_owned(),
                    tokens: 3,
                    drafter: Some("mtp".to_owned()),
                }),
                kv_cache_dtype: Some("fp8".to_owned()),
                prefix_cache: Some(true),
                max_sequences: Some(32),
            }
        );
    }

    #[test]
    fn missing_and_null_fields_are_absent_not_errors() {
        let mut value: Value = serde_json::from_str(READY).unwrap();
        value["serving"]["profile"]["name"] = Value::Null;
        value["serving"]["checkpoint_identity"] = Value::Null;
        value["serving"]["profile"]
            .as_object_mut()
            .unwrap()
            .remove("context_parallel_size");
        value["serving"]["profile"]["speculation"]["drafter"] = Value::Null;
        let health = Health::parse(value.to_string().as_bytes()).unwrap();
        let serving = health.serving.unwrap();
        assert_eq!(serving.profile.name, None);
        assert_eq!(serving.checkpoint_identity, None);
        assert_eq!(serving.profile.context_parallel_size, None);
        assert_eq!(serving.profile.speculation.unwrap().drafter, None);

        let starting =
            Health::parse(br#"{"status":"not_ready","reason":"worker starting","queue_depth":0}"#)
                .unwrap();
        assert!(!starting.ready());
        assert_eq!(starting.reason.as_deref(), Some("worker starting"));
        assert_eq!(starting.waiting_total, Some(0));
        assert!(starting.serving.is_none());
        assert_eq!(starting.model_name(), None);
        assert!(Health::parse(b"[1]").is_none());
        assert!(Health::parse(b"not json").is_none());
    }
}
