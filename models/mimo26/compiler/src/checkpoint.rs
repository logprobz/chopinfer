//! Checkpoint identities for preparation. Registration is not qualification.
//! A checkpoint no registered lock matches is served under a local identity
//! named by its own metadata ([`Registry::with_local`]).
use model_compiler_sdk::preparation::tokenizer::TokenizerSemantics;
use model_compiler_sdk::provenance::SourceLock;
use runtime_bundle::checkpoint_binding::FileDigest;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::OnceLock;

/// The registry name and repository of a local identity, and the prefix of
/// its served name.
pub const LOCAL: &str = "local";

/// The metadata a verifier authenticates before it reads a tensor header,
/// with the most bytes a local identity measures of each.
pub const METADATA: [(&str, u64); 4] = [
    ("config.json", 1 << 20),
    ("model.safetensors.index.json", 64 << 20),
    ("tokenizer.json", 64 << 20),
    ("tokenizer_config.json", 1 << 20),
];

/// The official release's tokenizer configuration, which a local identity's
/// must state ([`TokenizerSemantics`]).
const TOKENIZER_SEMANTICS: &[u8] = include_bytes!("../fixtures/tokenizer-semantics.json");

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub repository: String,
    pub revision: String,
    pub served_model_name: String,
    #[serde(default)]
    pub fp8_kv_scales: Option<Vec<[f32; 2]>>,
    #[serde(default)]
    pub fp8_kv_geometry: Option<String>,
    #[serde(default)]
    pub frontend: Option<serde_json::Value>,
}

impl Identity {
    /// Whether this is a local identity ([`Registry::with_local`]).
    pub fn is_local(&self) -> bool {
        self.repository == LOCAL
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryData {
    default: String,
    checkpoints: BTreeMap<String, Identity>,
}

#[derive(Clone)]
pub struct Registry {
    default: String,
    checkpoints: BTreeMap<String, Identity>,
    locks: BTreeMap<String, BTreeMap<String, FileDigest>>,
    /// The tokenizer configuration a local identity's must state.
    tokenizer: TokenizerSemantics,
}

pub fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        Registry::parse(
            include_str!("../../checkpoints.json"),
            include_str!("../../checkpoint-files.json"),
        )
        .expect("checkpoint registry and locks")
    })
}

pub fn identity(name: &str) -> Result<&'static Identity, String> {
    registry().identity(name)
}

impl Registry {
    pub fn parse(data: &str, locks: &str) -> crate::MimoResult<Self> {
        let data: RegistryData = serde_json::from_str(data)?;
        let registry = Self {
            default: data.default,
            checkpoints: data.checkpoints,
            locks: serde_json::from_str(locks)?,
            tokenizer: TokenizerSemantics::parse(TOKENIZER_SEMANTICS)?,
        };
        registry.identity(&registry.default)?;
        let mut triples = BTreeSet::new();
        for (name, identity) in &registry.checkpoints {
            if name == LOCAL || identity.is_local() {
                return Err(
                    format!("checkpoint {name}: {LOCAL} names an unregistered checkpoint").into(),
                );
            }
            if !triples.insert((
                &identity.repository,
                &identity.revision,
                &identity.served_model_name,
            )) {
                return Err(format!("duplicate checkpoint identity pins: {name}").into());
            }
            if !registry.locks.contains_key(name) {
                return Err(format!("checkpoint {name} has no source lock").into());
            }
            if let Some(scales) = &identity.fp8_kv_scales {
                let geometry = model_mimo26_contract::kv_calibration::geometry_id(
                    identity.fp8_kv_geometry.as_deref(),
                );
                let geometry = model_mimo26_contract::geometry::GEOMETRIES.iter()
                    .find(|g| g.id == geometry)
                    .ok_or_else(|| format!("checkpoint {name}: FP8 KV calibration has unknown compiled geometry {geometry}"))?;
                model_mimo26_contract::kv_calibration::validate(scales, geometry.global_layers())
                    .map_err(|e| format!("checkpoint {name}: {e}"))?;
            }
            if let Some(value) = &identity.frontend {
                let object = value
                    .as_object()
                    .ok_or("frontend override must be an object")?;
                if object.values().any(|value| !value.is_object()) {
                    return Err("frontend override blocks must be objects".into());
                }
                if object
                    .keys()
                    .any(|key| !matches!(key.as_str(), "tokenizer" | "vocabulary"))
                {
                    return Err(
                        "frontend override must contain only tokenizer or vocabulary".into(),
                    );
                }
            }
        }
        Ok(registry)
    }

    /// Resolve only registered calibration, and refuse a table absent from the build.
    pub fn fp8_kv_layout(
        &self,
        identity: &Identity,
        tp: model_mimo26_contract::Tp,
    ) -> Result<Option<String>, String> {
        let identity = self.identity(self.name(identity)?)?;
        let Some(scales) = &identity.fp8_kv_scales else {
            return Ok(None);
        };
        let geometry =
            model_mimo26_contract::kv_calibration::geometry_id(identity.fp8_kv_geometry.as_deref());
        if geometry != tp.geometry.id {
            return Err(format!(
                "FP8 KV calibration geometry {geometry} differs from checkpoint geometry {}",
                tp.geometry.id
            ));
        }
        let id = model_mimo26_contract::kv_calibration::layout_id(scales);
        if !model_mimo26_contract::serving::kv_formats(tp)
            .iter()
            .any(|format| format.layout_id == id.as_bytes() && format.scales == scales)
        {
            return Err(format!(
                "FP8 KV calibration layout {id} is not compiled; rebuild from checkpoints.json"
            ));
        }
        Ok(Some(id))
    }

    fn known(&self) -> String {
        self.checkpoints
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn identity(&self, name: &str) -> Result<&Identity, String> {
        self.checkpoints.get(name).ok_or_else(|| {
            format!(
                "unknown MiMo checkpoint identity {name}; known identities: {}",
                self.known()
            )
        })
    }

    pub(crate) fn name(&self, identity: &Identity) -> Result<&str, String> {
        self.checkpoints
            .iter()
            .find(|(_, pin)| {
                pin.repository == identity.repository
                    && pin.revision == identity.revision
                    && pin.served_model_name == identity.served_model_name
            })
            .map(|(name, _)| name.as_str())
            .ok_or_else(|| {
                format!(
                    "checkpoint identity is not pinned; known identities: {}",
                    self.known()
                )
            })
    }

    pub fn identities(&self) -> impl Iterator<Item = (&str, &Identity)> {
        std::iter::once((self.default.as_str(), &self.checkpoints[&self.default])).chain(
            self.checkpoints
                .iter()
                .filter(|(name, _)| *name != &self.default)
                .map(|(name, identity)| (name.as_str(), identity)),
        )
    }

    pub(crate) fn source_lock(&self, identity: &Identity) -> crate::MimoResult<SourceLock> {
        let name = self.name(identity)?;
        Ok(SourceLock::new(
            self.locks
                .get(name)
                .ok_or("checkpoint has no source lock")?
                .clone(),
        ))
    }

    /// This registry, with local identities checked against `semantics`
    /// instead of the official release's tokenizer configuration.
    #[must_use]
    pub fn with_tokenizer_semantics(mut self, semantics: TokenizerSemantics) -> Self {
        self.tokenizer = semantics;
        self
    }

    /// The verifier of the checkpoint at `root`: under the first registered
    /// identity whose lock its metadata matches, else under its local
    /// identity ([`Registry::with_local`]).
    pub fn verifier(
        &self,
        root: &Path,
    ) -> crate::MimoResult<crate::checkpoint_provenance::Verifier> {
        let mut refusals = Vec::new();
        for (name, identity) in self.identities() {
            match crate::checkpoint_provenance::Verifier::with_registry(root, identity, self) {
                Ok(verifier) => return Ok(verifier),
                Err(error) => refusals.push(format!("{name}: {error}")),
            }
        }
        let local = self.with_local(root).and_then(|registry| {
            crate::checkpoint_provenance::Verifier::with_registry(
                root,
                registry.identity(LOCAL)?,
                &registry,
            )
        });
        local.map_err(|error| {
            refusals.push(format!("{LOCAL}: {error}"));
            format!(
                "not a supported MiMo V2.6 Flash checkpoint ({})",
                refusals.join("; ")
            )
            .into()
        })
    }

    /// This registry with the local identity of the checkpoint at `root`,
    /// for a checkpoint no registered lock matches: a lock of its own
    /// [`METADATA`], measured here and authenticated again by its verifier,
    /// once its tokenizer configuration states the official one's
    /// ([`TokenizerSemantics`]). The identity is named by that lock's
    /// SHA-256: repository [`LOCAL`], the digest's first 40 hex digits as
    /// its revision, and `local-<digest>` as its served name. It has no FP8
    /// KV calibration, so it serves BF16 KV only.
    pub fn with_local(&self, root: &Path) -> crate::MimoResult<Self> {
        let mut pins = BTreeMap::new();
        let mut configuration = Vec::new();
        for (name, limit) in METADATA {
            let bytes = read_metadata(root, name, limit)?;
            pins.insert(
                name.to_owned(),
                FileDigest {
                    byte_length: bytes.len() as u64,
                    sha256: runtime_weights::sha256_bytes(&bytes),
                },
            );
            if name == "tokenizer_config.json" {
                configuration = bytes;
            }
        }
        let template = match root.join("chat_template.jinja").try_exists()? {
            true => Some(read_metadata(root, "chat_template.jinja", 1 << 20)?),
            false => None,
        };
        self.tokenizer.check(
            &serde_json::from_slice(&configuration)?,
            template.as_deref(),
        )?;
        let digest = SourceLock::new(pins.clone()).sha256();
        let mut registry = self.clone();
        registry.checkpoints.insert(
            LOCAL.into(),
            Identity {
                repository: LOCAL.into(),
                revision: digest[..40].into(),
                served_model_name: format!("{LOCAL}-{digest}"),
                fp8_kv_scales: None,
                fp8_kv_geometry: None,
                frontend: None,
            },
        );
        registry.locks.insert(LOCAL.into(), pins);
        Ok(registry)
    }
}

/// The bytes of the checkpoint file `name`, refused past `limit` bytes or
/// when the file changes while it is read.
pub(crate) fn read_metadata(root: &Path, name: &str, limit: u64) -> crate::MimoResult<Vec<u8>> {
    let source = crate::catalog::open_source(&root.join(name))?;
    let length = source.metadata()?.len();
    if length > limit {
        return Err(format!("{name} exceeds {limit} bytes").into());
    }
    let mut bytes = vec![0; usize::try_from(length)?];
    source.read_exact_at(&mut bytes, 0)?;
    source.verify()?;
    Ok(bytes)
}

pub fn identities() -> impl Iterator<Item = (&'static str, &'static Identity)> {
    registry().identities()
}

pub fn default_identity() -> &'static Identity {
    identity(&registry().default).expect("default checkpoint identity")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_default_keeps_pinned_rl_alternative() {
        assert_eq!(default_identity().served_model_name, "MiMo-V2.6-Flash-MOPD");
        assert_eq!(
            identity("mopd").unwrap().revision,
            "2479e2d0029eca9a34cc7e7f55a121925f81908e"
        );
        assert!(identity("unknown").is_err());
        let declared: serde_json::Value =
            serde_json::from_slice(crate::frontend::COMPONENT).unwrap();
        assert_eq!(
            declared["models"][0]["id"],
            default_identity().served_model_name
        );
        assert_eq!(
            declared["tokenizer"]["revision"],
            default_identity().revision
        );
        for (_, identity) in identities() {
            let bytes = crate::frontend::component_for_flash(
                identity,
                model_mimo26_contract::encoders::Encoders::ALL,
            )
            .unwrap();
            let descriptor =
                runtime_frontend_contract::FrontendDescriptorV1::from_component_bytes(&bytes)
                    .unwrap();
            assert_eq!(descriptor.tokenizer.revision, identity.revision);
        }
    }
}
