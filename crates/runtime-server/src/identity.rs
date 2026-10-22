//! What a deployment serves, as its host established it at startup.
use serde::Serialize;

/// The package, checkpoint and settings one server serves. The host fills it
/// once at startup from the values it logs there; `/health` reports it with
/// the served model names and computes nothing per request.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ServingIdentity {
    /// The bundle manifest's identifier.
    pub bundle_id: String,
    /// SHA-256 of a direct package's canonical manifest.
    pub package_identity: Option<String>,
    /// A direct load's checkpoint identity: the SHA-256 of every safetensors
    /// file's name, length and header.
    pub checkpoint_identity: Option<String>,
    /// SHA-256 of the other checkpoint files a direct package was selected for.
    pub metadata_identity: Option<String>,
    /// A random value drawn once per server process and also written to its
    /// startup log, so a client can tell that log from one another process
    /// left at the same address.
    pub startup_nonce: String,
    pub profile: ServingProfile,
    /// Tokens of one KV page on one rank: the package's KV layout. A
    /// context-parallel group of `profile.context_parallel_size` ranks finds
    /// shared prefixes and captures prompt ends in stripes of one page per
    /// group rank, so a client can tell which prefix hits to expect.
    pub kv_page_tokens: u32,
    /// The chopinfer version.
    pub version: String,
    /// The Git commit the binary was built from, when known.
    pub source_commit: Option<String>,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub model_options: std::collections::BTreeMap<String, serde_json::Value>,
    /// None for packages without options; false for any non-default value or
    /// a default the package has not qualified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_options_qualified: Option<bool>,
}

/// The serving profile a deployment named, if any, and the settings it
/// serves with after explicit options override the profile's defaults.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ServingProfile {
    pub name: Option<String>,
    pub tensor_parallel_size: usize,
    /// Ranks per context-parallel group, 1 when every rank holds every KV
    /// page. Reported when the served topology declares context-parallel
    /// groups, so a bundle without them reports what it always did.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_parallel_size: Option<usize>,
    pub speculation: ServingSpeculation,
    /// The resolved KV cache element, `bf16`, `fp8` or `mixed`. A `mixed`
    /// element's planes are in `kv_cache_planes`.
    pub kv_cache_dtype: String,
    /// The planes of a mixed-plane element, each in its own code; absent for
    /// a single-code element.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub kv_cache_planes: Vec<ServingKvPlane>,
    /// Whether prefixes are cached: enabled, and declared by the bundle.
    pub prefix_cache: bool,
    pub max_sequences: usize,
}

/// One plane of a mixed-plane KV element, as the package declares it.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ServingKvPlane {
    pub name: String,
    /// The stored code: `bf16`, `fp8_e4m3` or `fp4_e2m1`.
    pub code: String,
    /// The block scales of a scaled code.
    pub scale: Option<ServingKvScale>,
    /// Bytes one token stores in this plane, codes and scales together.
    pub bytes_per_token: u64,
}

/// One scale of `format` (`e4m3` or `ue8m0`) for each `block` values.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ServingKvScale {
    pub format: String,
    pub block: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ServingSpeculation {
    /// `off`, `fixed` or `adaptive`.
    pub mode: String,
    /// Most draft tokens per item; zero when off.
    pub tokens: usize,
    /// The bundle's declared drafter kind, when it drafts.
    pub drafter: Option<String>,
}

/// The `serving` member of a health response: `identity` and the model names
/// `/v1/models` lists.
pub(crate) fn health_member(
    identity: &ServingIdentity,
    models: &[crate::FrontendModelV1],
) -> serde_json::Value {
    let mut value = serde_json::to_value(identity).expect("an identity serializes");
    value["served_model_names"] = models.iter().map(|model| model.id.clone()).collect();
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_context_parallel_size_is_reported_only_where_groups_are_declared() {
        let profile = |context_parallel_size| ServingProfile {
            tensor_parallel_size: 4,
            context_parallel_size,
            ..ServingProfile::default()
        };
        let json = |size| serde_json::to_value(profile(size)).unwrap();
        assert!(json(None).get("context_parallel_size").is_none());
        assert_eq!(json(Some(1))["context_parallel_size"], 1);
        assert_eq!(json(Some(2))["context_parallel_size"], 2);
        let text = serde_json::to_string(&profile(Some(4))).unwrap();
        assert!(
            text.contains(r#""tensor_parallel_size":4,"context_parallel_size":4,"#),
            "{text}"
        );
    }

    #[test]
    fn kv_planes_are_reported_only_for_a_mixed_plane_element() {
        let single = serde_json::to_value(ServingProfile {
            kv_cache_dtype: "bf16".into(),
            ..ServingProfile::default()
        })
        .unwrap();
        assert!(single.get("kv_cache_planes").is_none(), "{single}");
        let mixed = serde_json::to_value(ServingProfile {
            kv_cache_dtype: "mixed".into(),
            kv_cache_planes: vec![
                ServingKvPlane {
                    name: "compressed".into(),
                    code: "fp4_e2m1".into(),
                    scale: Some(ServingKvScale {
                        format: "e4m3".into(),
                        block: 16,
                    }),
                    bytes_per_token: 18,
                },
                ServingKvPlane {
                    name: "rope".into(),
                    code: "bf16".into(),
                    scale: None,
                    bytes_per_token: 8,
                },
            ],
            ..ServingProfile::default()
        })
        .unwrap();
        assert_eq!(
            mixed["kv_cache_planes"],
            serde_json::json!([
                {"name": "compressed", "code": "fp4_e2m1",
                 "scale": {"format": "e4m3", "block": 16}, "bytes_per_token": 18},
                {"name": "rope", "code": "bf16", "scale": null, "bytes_per_token": 8}
            ])
        );
    }
}
