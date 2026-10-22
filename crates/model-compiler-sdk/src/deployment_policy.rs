//! A model's deployment policy (`chopinfer-deployment-policy-v1`): the
//! container environment variables the model adds to the shared ones, and
//! the drafters its packages may declare. The compiler carries it in every
//! package ([`crate::package_draft::deployment_policy`]) and the family
//! helper reports it (`chopinfer-model-<m> deployment-policy`), so the image
//! entrypoint (`deploy/docker/deployment.py`) validates a model's knobs
//! without a file of its own. That reader checks the same rules.
//!
//! Every variable is named without its `CHOPINFER_` prefix and starts with
//! one of the policy's `knob_prefixes`, and no prefix covers a shared
//! variable ([`SHARED_VARIABLES`]), so a policy never redefines one. A
//! deployment refuses a variable under another installed model's prefix.
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA: &str = "chopinfer-deployment-policy-v1";

/// The image's shared variables (`SHARED` in `deploy/docker/deployment.py`,
/// which a test compares with this list). A policy may neither declare one
/// nor own a prefix one starts with, so no policy changes a shared
/// constraint.
pub const SHARED_VARIABLES: &[&str] = &[
    "API_KEY",
    "BATCH_REFUSAL_MAX_CONSECUTIVE",
    "BATCH_REFUSAL_MAX_IN_WINDOW",
    "BATCH_REFUSAL_WINDOW_SECONDS",
    "BINARY",
    "BUNDLE_FORCE_FULL_VERIFY",
    "BUNDLE_VERIFY_CACHE_DIR",
    "DEBUG_REFERENCE_TOKENS",
    "DRAFT_COST",
    "DRAIN_TIMEOUT_MS",
    "EXPECTED_BUNDLE_ID",
    "GPU_MEMORY_UTILIZATION",
    "INSTALL_ROOT",
    "KV_CACHE_DTYPE",
    "KV_MEMORY_FRACTION",
    "LISTEN",
    "MAX_BATCH_TOKENS",
    "MAX_CONTEXT_TOKENS",
    "MAX_SEQUENCES",
    "MODALITIES",
    "MODEL",
    "MODEL_CACHE",
    "MODEL_OPTIONS",
    "MODULE",
    "PACKAGE",
    "PREFILL_CHUNK_TOKENS",
    "QUEUE_DEPTH",
    "REQUEST_TRACE",
    "SAMPLING_CAPTURE",
    "SAMPLING_CAPTURE_ROWS",
    "SAMPLING_TIMING",
    "SERVE_ARGS",
    "SPECULATION",
    "SPECULATIVE_TOKENS",
    "STEP_PROFILE_CAPACITY",
    "STEP_PROFILE_EVERY",
    "TAPS_BYTES",
    "TAPS_DIR",
    "TAPS_FROM",
    "TAPS_LAYERS",
    "TAPS_SCOPE_JSON",
    "TAPS_SITES",
    "TENSOR_PARALLEL_SIZE",
    "VERIFY_MANIFEST",
    "VERIFY_MODEL",
    "VIDEO_FRAME_CHECKSUMS",
    "VIDEO_INPUT",
];

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeploymentPolicyV1 {
    pub schema: String,
    /// The bundle ID of the packages this policy governs.
    pub bundle_id: String,
    /// The variable name prefixes the model owns, such as `M3_`.
    pub knob_prefixes: Vec<String>,
    /// The drafter kinds a package of this bundle may declare.
    pub drafters: Vec<String>,
    /// Variables that take any value that is not empty, such as a path.
    #[serde(default)]
    pub paths: Vec<String>,
    /// Variables that take 0 or 1.
    #[serde(default)]
    pub booleans: Vec<String>,
    /// Variables that are either unset or 1.
    #[serde(default)]
    pub presence: Vec<String>,
    /// Integer variables and their inclusive bounds.
    #[serde(default)]
    pub numbers: BTreeMap<String, [u64; 2]>,
    /// Integer variables that must also be a multiple of a number.
    #[serde(default)]
    pub multiples: BTreeMap<String, u64>,
    /// Per-layer tap variables, which a serving image refuses.
    #[serde(default)]
    pub taps: Vec<String>,
    /// Variables that apply only with one of the listed drafters.
    #[serde(default)]
    pub drafter_only: BTreeMap<String, Vec<String>>,
    /// Variables that turn off a group of others.
    #[serde(default)]
    pub disabled_by: Vec<DisabledByV1>,
}

/// While `variable` is `value`, every variable starting with `prefix` is
/// refused as not applying "when `reason`".
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DisabledByV1 {
    pub variable: String,
    pub value: String,
    pub prefix: String,
    pub reason: String,
}

fn variable(name: &str) -> bool {
    name.bytes().next().is_some_and(|b| b.is_ascii_uppercase())
        && name
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
}

fn identifier(name: &str, extra: &[u8]) -> bool {
    name.bytes()
        .next()
        .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || extra.contains(&b))
}

/// Any JSON value, refused when an object anywhere in it names a member
/// twice. serde would keep the last of two map members silently, where the
/// image's reader refuses the document.
struct UniqueMembers;

impl<'de> Deserialize<'de> for UniqueMembers {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(UniqueMembers)
    }
}

impl<'de> serde::de::Visitor<'de> for UniqueMembers {
    type Value = Self;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a JSON value")
    }
    fn visit_bool<E>(self, _: bool) -> Result<Self, E> {
        Ok(Self)
    }
    fn visit_i64<E>(self, _: i64) -> Result<Self, E> {
        Ok(Self)
    }
    fn visit_u64<E>(self, _: u64) -> Result<Self, E> {
        Ok(Self)
    }
    fn visit_f64<E>(self, _: f64) -> Result<Self, E> {
        Ok(Self)
    }
    fn visit_str<E>(self, _: &str) -> Result<Self, E> {
        Ok(Self)
    }
    fn visit_unit<E>(self) -> Result<Self, E> {
        Ok(Self)
    }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut items: A) -> Result<Self, A::Error> {
        while items.next_element::<Self>()?.is_some() {}
        Ok(Self)
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut members: A) -> Result<Self, A::Error> {
        let mut seen = BTreeSet::new();
        while let Some(name) = members.next_key::<String>()? {
            if !seen.insert(name.clone()) {
                return Err(serde::de::Error::custom(format!("member {name} repeats")));
            }
            members.next_value::<Self>()?;
        }
        Ok(Self)
    }
}

/// Parses and checks one policy document of at most
/// [`runtime_bundle::deployment::DEPLOYMENT_COMPONENT_MAX_BYTES`].
pub fn parse(bytes: &[u8]) -> Result<DeploymentPolicyV1, String> {
    if bytes.len() as u64 > runtime_bundle::deployment::DEPLOYMENT_COMPONENT_MAX_BYTES {
        return Err("deployment policy: it exceeds its bound".into());
    }
    serde_json::from_slice::<UniqueMembers>(bytes)
        .map_err(|e| format!("deployment policy: {e}"))?;
    let policy: DeploymentPolicyV1 =
        serde_json::from_slice(bytes).map_err(|e| format!("deployment policy: {e}"))?;
    policy.check()?;
    Ok(policy)
}

impl DeploymentPolicyV1 {
    fn check(&self) -> Result<(), String> {
        let refuse = |reason: String| Err(format!("deployment policy: {reason}"));
        if self.schema != SCHEMA {
            return refuse(format!("schema must be {SCHEMA}"));
        }
        if !identifier(&self.bundle_id, b".-") {
            return refuse(format!("invalid bundle ID {:?}", self.bundle_id));
        }
        let prefixes: BTreeSet<_> = self.knob_prefixes.iter().collect();
        if prefixes.is_empty()
            || prefixes.len() != self.knob_prefixes.len()
            || !self
                .knob_prefixes
                .iter()
                .all(|prefix| variable(prefix) && prefix.ends_with('_'))
        {
            return refuse("knob prefixes must be distinct names ending in _".into());
        }
        if let Some(name) = SHARED_VARIABLES.iter().find(|name| {
            self.knob_prefixes
                .iter()
                .any(|prefix| name.starts_with(prefix.as_str()))
        }) {
            return refuse(format!("a knob prefix covers the shared variable {name}"));
        }
        let drafters: BTreeSet<_> = self.drafters.iter().collect();
        if drafters.len() != self.drafters.len()
            || !self.drafters.iter().all(|kind| identifier(kind, b"_-"))
        {
            return refuse("drafters must be distinct drafter kinds".into());
        }
        let owned = |name: &str| {
            variable(name)
                && self
                    .knob_prefixes
                    .iter()
                    .any(|prefix| name.starts_with(prefix.as_str()))
        };
        let declared: Vec<&String> = self
            .paths
            .iter()
            .chain(&self.booleans)
            .chain(&self.presence)
            .chain(self.numbers.keys())
            .chain(&self.taps)
            .collect();
        let unique: BTreeSet<_> = declared.iter().collect();
        if unique.len() != declared.len() {
            return refuse("a variable is declared twice".into());
        }
        if let Some(name) = declared.iter().find(|name| !owned(name)) {
            return refuse(format!("{name} does not start with a knob prefix"));
        }
        if let Some(name) = declared
            .iter()
            .find(|name| SHARED_VARIABLES.contains(&name.as_str()))
        {
            return refuse(format!("{name} is a shared variable"));
        }
        if let Some((name, _)) = self.numbers.iter().find(|(_, [low, high])| low > high) {
            return refuse(format!("{name} has an empty range"));
        }
        for (name, multiple) in &self.multiples {
            if *multiple == 0 || !self.numbers.contains_key(name) {
                return refuse(format!(
                    "{name} must be a declared number and multiple of 1 or more"
                ));
            }
        }
        let settable = |name: &str| {
            unique.iter().any(|known| known.as_str() == name)
                && !self.taps.iter().any(|tap| tap == name)
        };
        for (name, kinds) in &self.drafter_only {
            if !settable(name)
                || kinds.is_empty()
                || kinds.iter().collect::<BTreeSet<_>>().len() != kinds.len()
                || !kinds.iter().all(|kind| drafters.contains(kind))
            {
                return refuse(format!(
                    "{name} must be a declared variable of declared drafters"
                ));
            }
        }
        for rule in &self.disabled_by {
            if !settable(&rule.variable)
                || !owned(&rule.prefix)
                || rule.value.is_empty()
                || rule.reason.is_empty()
            {
                return refuse(format!(
                    "{} must disable a prefix the model owns, with a value and reason",
                    rule.variable
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> serde_json::Value {
        serde_json::json!({
            "schema": SCHEMA, "bundle_id": "fixture.v1", "knob_prefixes": ["FIXTURE_"],
            "drafters": ["mtp"], "booleans": ["FIXTURE_FLAG", "FIXTURE_GROUP"],
            "presence": ["FIXTURE_TRACE"], "numbers": {"FIXTURE_COUNT": [0, 8]},
            "multiples": {"FIXTURE_COUNT": 2}, "taps": ["FIXTURE_DUMP"],
            "drafter_only": {"FIXTURE_FLAG": ["mtp"]},
            "disabled_by": [{"variable": "FIXTURE_GROUP", "value": "0",
                             "prefix": "FIXTURE_GROUP_", "reason": "the group is off"}],
        })
    }

    #[test]
    fn a_policy_declares_only_variables_under_its_own_prefixes() {
        assert!(parse(policy().to_string().as_bytes()).is_ok());
        let minimal = serde_json::json!({
            "schema": SCHEMA, "bundle_id": "fixture.v1", "knob_prefixes": ["FIXTURE_"],
            "drafters": [],
        });
        assert!(parse(minimal.to_string().as_bytes()).is_ok());
        for (pointer, value) in [
            ("/schema", serde_json::json!("v0")),
            ("/bundle_id", serde_json::json!("../x")),
            ("/knob_prefixes", serde_json::json!([])),
            ("/knob_prefixes", serde_json::json!(["FIXTURE"])),
            ("/drafters", serde_json::json!(["mtp", "mtp"])),
            ("/booleans", serde_json::json!(["MAX_SEQUENCES"])),
            ("/presence", serde_json::json!(["FIXTURE_FLAG"])),
            ("/numbers/FIXTURE_COUNT", serde_json::json!([9, 8])),
            ("/multiples/FIXTURE_COUNT", serde_json::json!(0)),
            ("/drafter_only/FIXTURE_FLAG", serde_json::json!(["dflash"])),
            ("/drafter_only/FIXTURE_FLAG", serde_json::json!([])),
            ("/disabled_by/0/prefix", serde_json::json!("OTHER_")),
            ("/disabled_by/0/variable", serde_json::json!("FIXTURE_DUMP")),
        ] {
            let mut value_of = policy();
            *value_of.pointer_mut(pointer).unwrap() = value;
            assert!(parse(value_of.to_string().as_bytes()).is_err(), "{pointer}");
        }
        let mut extra = policy();
        extra["unknown"] = serde_json::json!(1);
        assert!(parse(extra.to_string().as_bytes()).is_err());
    }

    /// The documents both readers must accept or refuse alike: this one and
    /// the image's (`deploy/docker/deployment.py`, run on the same file by
    /// tests/test_deploy_entrypoint.py).
    #[test]
    fn the_shared_policy_cases_resolve_as_the_image_reader_does() {
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../../../tests/deployment-policy-cases.json"))
                .unwrap();
        assert!(cases.len() > 10);
        for case in cases {
            let mut document = case["document"].as_str().unwrap().to_owned();
            if let Some(padding) = case["padding"].as_u64() {
                let end = document.rfind('}').unwrap();
                document.insert_str(end, &" ".repeat(padding as usize));
            }
            let bytes = match case["prefix"].as_str() {
                Some("bom") => [&b"\xef\xbb\xbf"[..], document.as_bytes()].concat(),
                _ => document.into_bytes(),
            };
            let parsed = parse(&bytes);
            assert_eq!(
                parsed.is_ok(),
                case["valid"] == true,
                "{}: {parsed:?}",
                case["name"]
            );
            assert_eq!(
                crate::package_draft::deployment_policy(&bytes).is_ok(),
                case["valid"] == true,
                "{}",
                case["name"]
            );
        }
    }
}
