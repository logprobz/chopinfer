//! Common package records. Model IDs, paths, placement and limits stay caller data.
use crate::{ComponentContentV1, ComponentDraftV1, NativeModuleDraftV1, StagedContentV1};
use runtime_bundle::{
    BundleError, BundleGraphBucketV1, BundleKvDtypeV1, BundleKvFormatV1, BundleKvLayoutV1,
    BundleOpaqueId, BundleRankPlacementV1, BundleRelativePathV1, BundleStateLifetimeV1,
};

pub fn whole_component(
    component: &str,
    role: &str,
    format: &str,
    path: &str,
    content: StagedContentV1,
) -> Result<ComponentDraftV1, BundleError> {
    Ok(ComponentDraftV1::new(
        BundleOpaqueId::new(component)?,
        BundleOpaqueId::new(role)?,
        BundleOpaqueId::new(format)?,
        BundleRelativePathV1::new(path)?,
        ComponentContentV1::Whole(content),
    ))
}

pub fn frontend(content: StagedContentV1) -> Result<ComponentDraftV1, BundleError> {
    whole_component(
        "frontend",
        "chop.frontend.openai.v1",
        "chop.frontend.json.v1",
        "components/frontend.json",
        content,
    )
}

/// The model's serving profiles (its `profiles.json`), carried whole: a bare
/// `serve --bundle DIR --profile NAME` resolves NAME from them. Refused
/// unless it is a `chopinfer-serve-profiles-v2` document; the host checks
/// each profile against the package when it applies one.
pub fn serve_profiles(bytes: &[u8]) -> Result<ComponentDraftV1, BundleError> {
    use runtime_bundle::deployment::{
        DEPLOYMENT_COMPONENT_MAX_BYTES, SERVE_PROFILES_COMPONENT_V1, SERVE_PROFILES_FORMAT_V1,
        SERVE_PROFILES_PATH_V1, SERVE_PROFILES_ROLE_V1,
    };
    let document: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| BundleError::validation(format!("serving profiles: {e}")))?;
    if bytes.len() as u64 > DEPLOYMENT_COMPONENT_MAX_BYTES
        || document["schema"] != "chopinfer-serve-profiles-v2"
        || !document["profiles"].is_object()
    {
        return Err(BundleError::validation(
            "serving profiles must be a bounded chopinfer-serve-profiles-v2 document",
        ));
    }
    whole_component(
        SERVE_PROFILES_COMPONENT_V1,
        SERVE_PROFILES_ROLE_V1,
        SERVE_PROFILES_FORMAT_V1,
        SERVE_PROFILES_PATH_V1,
        StagedContentV1::bytes(bytes.to_vec()),
    )
}

/// The model's deployment policy (its `deploy/policy.json`, see
/// [`crate::deployment_policy`]), carried whole for the image entrypoint to
/// validate its environment against. Refused unless it passes that
/// module's checks.
pub fn deployment_policy(bytes: &[u8]) -> Result<ComponentDraftV1, BundleError> {
    use runtime_bundle::deployment::{
        DEPLOYMENT_COMPONENT_MAX_BYTES, DEPLOYMENT_POLICY_COMPONENT_V1,
        DEPLOYMENT_POLICY_FORMAT_V1, DEPLOYMENT_POLICY_PATH_V1, DEPLOYMENT_POLICY_ROLE_V1,
    };
    if bytes.len() as u64 > DEPLOYMENT_COMPONENT_MAX_BYTES {
        return Err(BundleError::validation(
            "the deployment policy exceeds its bound",
        ));
    }
    crate::deployment_policy::parse(bytes).map_err(BundleError::validation)?;
    whole_component(
        DEPLOYMENT_POLICY_COMPONENT_V1,
        DEPLOYMENT_POLICY_ROLE_V1,
        DEPLOYMENT_POLICY_FORMAT_V1,
        DEPLOYMENT_POLICY_PATH_V1,
        StagedContentV1::bytes(bytes.to_vec()),
    )
}

pub fn tokenizer_definition(content: StagedContentV1) -> Result<ComponentDraftV1, BundleError> {
    whole_component(
        "tokenizer-definition",
        "chop.tokenizer.definition.v1",
        "chop.tokenizer.json.v1",
        "components/tokenizer.json",
        content,
    )
}

pub fn tokenizer_configuration(content: StagedContentV1) -> Result<ComponentDraftV1, BundleError> {
    whole_component(
        "tokenizer-configuration",
        "chop.tokenizer.configuration.v1",
        "chop.tokenizer.json.v1",
        "components/tokenizer_config.json",
        content,
    )
}

/// Powers of two up to the package's batch limit, in ascending order.
pub fn decode_buckets(limit: u32) -> impl Iterator<Item = u32> {
    (0..limit.checked_ilog2().map_or(0, |log| log + 1)).map(|shift| 1 << shift)
}

pub fn graph_buckets(
    components: &[ComponentDraftV1],
    limit: u32,
    decode_name: impl Fn(u32) -> String,
    prefill_name: &str,
) -> Result<Vec<BundleGraphBucketV1>, BundleError> {
    graph_buckets_for(components, decode_buckets(limit), decode_name, prefill_name)
}

/// Use the bundle's bucket enumeration when it provides one.
pub fn graph_buckets_for(
    components: &[ComponentDraftV1],
    decode: impl IntoIterator<Item = u32>,
    decode_name: impl Fn(u32) -> String,
    prefill_name: &str,
) -> Result<Vec<BundleGraphBucketV1>, BundleError> {
    let ids: Vec<_> = components.iter().map(|c| c.id().clone()).collect();
    let mut buckets = decode
        .into_iter()
        .map(|batch| {
            BundleGraphBucketV1::new(
                BundleOpaqueId::new(decode_name(batch))?,
                ids.clone(),
                Vec::new(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    buckets.push(BundleGraphBucketV1::new(
        BundleOpaqueId::new(prefill_name)?,
        ids,
        Vec::new(),
    )?);
    Ok(buckets)
}

/// A rank-local paged KV layout held for a transaction: `page_tokens`-token
/// pages of `page_bytes` in one segment, 256-byte aligned, for sequences of
/// up to `context_tokens`.
pub fn paged_kv_layout(
    layout_id: &str,
    page_bytes: u64,
    page_tokens: u64,
    context_tokens: u64,
) -> Result<BundleKvLayoutV1, BundleError> {
    BundleKvLayoutV1::new(
        BundleOpaqueId::new(layout_id)?,
        page_bytes,
        256,
        u32::try_from(page_tokens).map_err(|_| BundleError::validation("KV page tokens"))?,
        page_bytes / page_tokens,
        1,
        context_tokens,
        BundleRankPlacementV1::RankLocal,
        BundleStateLifetimeV1::Transaction,
    )
}

/// The KV formats of a bundle that runs BF16 and FP8 E4M3 KV, the host
/// picking one at startup with `--kv-cache-dtype`.
pub fn bf16_and_fp8_kv(bf16: BundleKvLayoutV1, fp8: BundleKvLayoutV1) -> Vec<BundleKvFormatV1> {
    vec![
        BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, bf16),
        BundleKvFormatV1::new(BundleKvDtypeV1::Fp8E4m3, fp8),
    ]
}

pub fn native_module(content: StagedContentV1) -> Result<NativeModuleDraftV1, BundleError> {
    Ok(NativeModuleDraftV1::new(
        BundleRelativePathV1::new("native/module.bin")?,
        content,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batch_limit_bounds_buckets_without_overflow() {
        assert!(decode_buckets(0).next().is_none());
        assert_eq!(decode_buckets(1).collect::<Vec<_>>(), [1]);
        assert_eq!(decode_buckets(6).collect::<Vec<_>>(), [1, 2, 4]);
        assert_eq!(decode_buckets(u32::MAX).last(), Some(1 << 31));
    }

    #[test]
    fn serving_profiles_must_be_a_profiles_document() {
        let profiles = br#"{"schema":"chopinfer-serve-profiles-v2","profiles":{}}"#;
        let draft = serve_profiles(profiles).unwrap();
        assert_eq!(draft.id().as_str(), "serve-profiles");
        for refused in [
            &br#"{"schema":"chopinfer-serve-profiles-v1","profiles":{}}"#[..],
            br#"{"schema":"chopinfer-serve-profiles-v2"}"#,
            b"not JSON",
        ] {
            assert!(serve_profiles(refused).is_err());
        }
    }
}
