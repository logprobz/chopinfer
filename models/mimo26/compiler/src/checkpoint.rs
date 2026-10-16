//! Checkpoint identities for preparation. Registration is not qualification.
use model_compiler_sdk::provenance::SourceLock;
use runtime_bundle::checkpoint_binding::FileDigest;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

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
        };
        registry.identity(&registry.default)?;
        let mut triples = BTreeSet::new();
        for (name, identity) in &registry.checkpoints {
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

    pub fn verifier(
        &self,
        root: &std::path::Path,
    ) -> crate::MimoResult<crate::checkpoint_provenance::Verifier> {
        let mut refusals = Vec::new();
        for (name, identity) in self.identities() {
            match crate::checkpoint_provenance::Verifier::with_registry(root, identity, self) {
                Ok(verifier) => return Ok(verifier),
                Err(error) => refusals.push(format!("{name}: {error}")),
            }
        }
        Err(format!(
            "not a pinned MiMo V2.6 Flash checkpoint ({})",
            refusals.join("; ")
        )
        .into())
    }
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
