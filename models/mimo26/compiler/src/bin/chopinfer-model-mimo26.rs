//! Installed family helper for MiMo V2.6 Flash. It serves a pinned
//! checkpoint by loading it directly (`chopinfer serve --model DIR`): the
//! package it emits carries no rank images, and it prepares none.

use model_compiler_sdk::files::Result;
use model_compiler_sdk::preparation::direct::{
    DirectEmission, default_profile, installed_native_module, native_module_inputs,
    native_module_staged,
};
use model_compiler_sdk::preparation::helper::{self, Provider};
use model_mimo26::checkpoint_provenance::Verifier;
use model_mimo26_contract::Tp;
use runtime_bundle::preparation::PrepareDirectRequest;
use std::path::{Path, PathBuf};

// The packager's draft.
#[path = "mimo26_package/package.rs"]
mod package;

// Package tests use a native-module object fixture.
#[cfg(test)]
#[path = "../test_native.rs"]
mod test_native;

struct MimoFamily;

/// The serving profiles the package path serves MiMo with.
const PROFILES: &str = include_str!("../../../profiles.json");

impl MimoFamily {
    /// The native module installed beside this helper or in the adjacent
    /// release `lib` directory.
    fn native_module() -> Result<PathBuf> {
        installed_native_module("libbundle_mimo26.so", "MiMo")
    }

    /// The default serving profile: DFlash when the checkpoint carries its
    /// drafter, then MTP, where the rank count has that profile, else
    /// autoregressive.
    fn profile(root: &Path, tp: Tp) -> Result<runtime_bundle::preparation::DirectProfile> {
        use model_mimo26::catalog::{Catalog, DrafterKind};
        let config = Self::verifier(root)?.config()?;
        let tp = Tp::of_geometry(config.geometry()?, tp.ranks())?;
        let mut drafters = Vec::new();
        for (name, kind) in [("dflash", DrafterKind::Dflash), ("mtp", DrafterKind::Mtp)] {
            if Catalog::load_drafter_with_config(root, kind, &config).is_ok() {
                drafters.push(name);
            }
        }
        drafters.push("none");
        let profile = default_profile(PROFILES, tp.ranks(), &drafters)?;
        Self::profile_geometry(profile, tp)
    }

    fn profile_geometry(
        mut profile: runtime_bundle::preparation::DirectProfile,
        tp: Tp,
    ) -> Result<runtime_bundle::preparation::DirectProfile> {
        profile.profile["topology"] = format!("mimo-{}-tp{}", tp.geometry.id, tp.ranks()).into();
        if tp.geometry.id != model_mimo26_contract::geometry::V26_FLASH.id {
            profile.name = format!("{}-{}", tp.geometry.id, profile.name);
            profile.profile["status"] = "experimental".into();
            profile.profile["qualification"] =
                format!("unqualified geometry {}", tp.geometry.id).into();
            profile.profile["evidence"] =
                serde_json::json!(["models/mimo26/geometries/supported.json"]);
            profile.profile["qualified_max_sequences"] = serde_json::Value::Null;
        }
        Ok(profile)
    }

    /// The verifier of the checkpoint at `root`: the registered identity
    /// whose source lock its config, index and tokenizer files match, else
    /// its local identity.
    fn verifier(root: &Path) -> Result<Verifier> {
        Self::verifier_with_registry(root, model_mimo26::checkpoint::registry())
    }

    fn verifier_with_registry(
        root: &Path,
        registry: &model_mimo26::checkpoint::Registry,
    ) -> Result<Verifier> {
        registry.verifier(root)
    }
}

impl Provider for MimoFamily {
    fn name(&self) -> &str {
        "chopinfer-model-mimo26"
    }

    fn deployment_policy(&self) -> Option<&str> {
        Some(package::DEPLOYMENT_POLICY)
    }

    fn profiles(&self) -> Vec<String> {
        Tp::SUPPORTED
            .map(|tp| format!("direct-text-ar-tp{}", tp.ranks()))
            .to_vec()
    }

    fn names_direct_profile(&self) -> bool {
        true
    }
    fn direct_profile(
        &self,
        root: &Path,
        request: &PrepareDirectRequest,
    ) -> Result<runtime_bundle::preparation::DirectProfile> {
        Self::profile(root, Tp::new(request.tensor_parallel_size)?)
    }

    fn probe(&self, path: &Path, ranks: u32) -> Result<()> {
        Tp::new(ranks)?;
        let config = Self::verifier(path)?.config()?;
        Tp::of_geometry(config.geometry()?, ranks)?;
        Ok(())
    }

    fn direct_modalities(
        &self,
        root: &Path,
        request: &PrepareDirectRequest,
    ) -> Result<Option<std::collections::BTreeSet<runtime_bundle::preparation::Modality>>> {
        let config = Self::verifier(root)?.config()?;
        let catalog = model_mimo26::catalog::Catalog::load(root, &config)?;
        Ok(Some(model_mimo26::encoders::served_modalities(
            request,
            catalog.geometry,
            &catalog.encoders,
        )?))
    }

    /// Besides the checkpoint, the request and this helper, its emission
    /// reads only the native module it packages, and no environment.
    fn direct_inputs(
        &self,
        root: &Path,
        _request: &PrepareDirectRequest,
    ) -> Result<Option<Vec<u8>>> {
        let config = Self::verifier(root)?.config()?;
        let geometry = config.geometry()?;
        let mut inputs: serde_json::Value =
            serde_json::from_slice(&native_module_inputs(&Self::native_module()?, &[])?)?;
        inputs["geometry"] = serde_json::json!({"id": geometry.id, "document": geometry.document});
        Ok(Some(serde_json::to_vec(&inputs)?))
    }

    fn emit_direct(
        &self,
        root: &Path,
        request: &PrepareDirectRequest,
        checkpoint_identity: &str,
        output: &Path,
    ) -> Result<DirectEmission> {
        self.emit_direct_with_inputs(root, request, checkpoint_identity, output, None)
    }

    /// The package the package path serves for this checkpoint and rank
    /// count, which serves every supported rank count: the text image, the
    /// drafter of the rank count's default serving profile (DFlash when the
    /// checkpoint carries its drafter, then MTP, where the rank count has
    /// that profile) and the vision and audio encoders when the checkpoint
    /// has them, all planned components the weight plan supplies. With emission inputs, the native module is only the one
    /// they identify.
    fn emit_direct_with_inputs(
        &self,
        root: &Path,
        request: &PrepareDirectRequest,
        _checkpoint_identity: &str,
        output: &Path,
        inputs: Option<&[u8]>,
    ) -> Result<DirectEmission> {
        use model_mimo26::catalog::{Catalog, DrafterKind};
        let tp = Tp::new(request.tensor_parallel_size)?;
        if let Some(policy) = &request.activation_policy {
            return Err(format!(
                "MiMo packages have one fixed precision; --activation-policy {policy} does not apply"
            )
            .into());
        }
        let verifier = Self::verifier(root)?;
        let config = verifier.config()?;
        let catalog = Catalog::load(root, &config)?;
        let encoders =
            model_mimo26::encoders::served(request, catalog.geometry, &catalog.encoders)?;
        let profile = Self::profile(root, tp)?;
        let drafter = match profile.profile["speculation"]["drafter"].as_str() {
            Some("dflash") => Some(DrafterKind::Dflash),
            Some("mtp") => Some(DrafterKind::Mtp),
            _ => None,
        };
        if verifier.identity().is_local() {
            verifier
                .registry()
                .check_tokenizer(root, verifier.identity(), config.geometry()?)?;
        }
        let [tokenizer_definition, tokenizer_configuration] =
            package::PackageAssets::pinned_tokenizer_with_registry(
                root,
                verifier.identity(),
                verifier.registry(),
            )?;
        let assets = package::PackageAssets {
            tokenizer_definition,
            tokenizer_configuration,
            native_module: native_module_staged(&Self::native_module()?, inputs)?,
            drafter,
            encoders,
            precision: model_mimo26::catalog::precision_components(root, drafter).map_err(
                |error| model_compiler_sdk::CompilerError::validation(error.to_string()),
            )?,
        };
        let bytes = package::planned_rank_bytes(root, &assets)?;
        Ok(DirectEmission {
            package: package::emit_verified_direct_package(assets, &bytes, output, &verifier)?,
            profile,
        })
    }
}

fn main() -> Result<()> {
    helper::run(&MimoFamily, std::env::args().skip(1))
}

#[cfg(test)]
#[path = "../../tests/support/registry.rs"]
mod registry_fixture;

#[cfg(test)]
mod tests {
    use super::*;
    use model_compiler_sdk::StagedContentV1;
    use model_mimo26_contract::backend::{Kernels, Selected};
    use model_mimo26_contract::encoders::Encoders;

    #[test]
    fn experimental_geometry_profile_passes_runtime_host_schema() {
        let fixture = registry_fixture::Fixture::new();
        let mut geometry = model_mimo26_contract::geometry::V26_FLASH;
        geometry.id = "profile-test";
        let mut tp = Tp::TP2;
        tp.geometry = Box::leak(Box::new(geometry));
        let original = default_profile(PROFILES, tp.ranks(), &["none"]).unwrap();
        let profile = MimoFamily::profile_geometry(original, tp).unwrap();
        assert_eq!(profile.profile["status"], "experimental");
        assert_eq!(
            profile.profile["evidence"],
            serde_json::json!(["models/mimo26/geometries/supported.json"])
        );
        assert_eq!(
            profile.profile["qualification"],
            "unqualified geometry profile-test"
        );
        let path = fixture.root.join("profiles.json");
        std::fs::write(&path, serde_json::to_vec(&serde_json::json!({
            "schema": "chopinfer-serve-profiles-v2", "model": "mimo26", "served_model_name": "test",
            "profiles": {profile.name: profile.profile}
        })).unwrap()).unwrap();
        let command = runtime_host::parse_host_command_v1(&[
            "profiles".into(),
            "list".into(),
            path.to_str().unwrap().into(),
        ])
        .unwrap();
        let runtime_host::HostCommandV1::Display { text } = command else {
            panic!("profile inspection")
        };
        assert!(text.contains("experimental"));
    }

    #[test]
    fn unlisted_authenticated_geometry_is_refused_before_tensor_headers() {
        let mut fixture = registry_fixture::Fixture::new();
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture.root.join("config.json")).unwrap())
                .unwrap();
        config["max_position_embeddings"] = 524288.into();
        let bytes = serde_json::to_vec(&config).unwrap();
        std::fs::write(fixture.root.join("config.json"), &bytes).unwrap();
        fixture.locks["third"]["config.json"] = serde_json::json!({"byte_length": bytes.len(), "sha256": runtime_weights::sha256_bytes(&bytes)});
        std::fs::remove_file(fixture.root.join("weight.safetensors")).unwrap();
        let error = match MimoFamily::verifier_with_registry(&fixture.root, &fixture.registry()) {
            Ok(_) => panic!("unlisted geometry"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("unsupported geometry:"), "{error}");
        assert!(
            error.contains("max_position_embeddings expected 1048576, received 524288"),
            "{error}"
        );
        assert!(error.contains("supported ids: v26-flash"), "{error}");
    }

    #[test]
    fn helper_advertises_only_the_authenticated_calibration() {
        for format in model_mimo26_contract::serving::kv_formats(Tp::TP4)
            .into_iter()
            .filter(|f| f.fp8())
        {
            let mut fixture = registry_fixture::Fixture::new();
            fixture.data["checkpoints"]["third"]["fp8_kv_scales"] =
                serde_json::json!(format.scales);
            let registry = fixture.registry();
            let verifier = MimoFamily::verifier_with_registry(&fixture.root, &registry).unwrap();
            let [tokenizer_definition, tokenizer_configuration] =
                package::PackageAssets::pinned_tokenizer_with_registry(
                    &fixture.root,
                    verifier.identity(),
                    verifier.registry(),
                )
                .unwrap();
            let assets = package::PackageAssets {
                tokenizer_definition,
                tokenizer_configuration,
                native_module: StagedContentV1::file(crate::test_native::fixture()),
                drafter: None,
                encoders: Encoders::NONE,
                precision: package::fixture_precision(),
            };
            let emitted = package::emit_verified_direct_package(
                assets,
                &Tp::SUPPORTED.map(|tp| (tp, 1 << 30)),
                &fixture.root.join("package"),
                &verifier,
            )
            .unwrap();
            for tp in Tp::SUPPORTED {
                let manifest = emitted
                    .manifest()
                    .for_topology(model_mimo26_contract::topology::topology_id(tp))
                    .unwrap();
                let ids: Vec<_> = manifest
                    .kv_formats()
                    .iter()
                    .map(|f| f.layout().id().as_str())
                    .collect();
                assert_eq!(
                    ids,
                    [
                        "mimo.ga-paged-bf16.v1",
                        std::str::from_utf8(format.layout_id).unwrap()
                    ]
                );
            }
        }
    }

    #[test]
    fn helper_admits_and_packages_a_data_only_identity() {
        let fixture = registry_fixture::Fixture::new();
        let registry = fixture.registry();
        let verifier = MimoFamily::verifier_with_registry(&fixture.root, &registry).unwrap();
        assert_eq!(verifier.identity().served_model_name, "third");
        let [tokenizer_definition, tokenizer_configuration] =
            package::PackageAssets::pinned_tokenizer_with_registry(
                &fixture.root,
                verifier.identity(),
                verifier.registry(),
            )
            .unwrap();
        let assets = package::PackageAssets {
            tokenizer_definition,
            tokenizer_configuration,
            native_module: StagedContentV1::file(crate::test_native::fixture()),
            drafter: None,
            encoders: Encoders::NONE,
            precision: package::fixture_precision(),
        };
        let output = fixture.root.join("package");
        let rank_bytes = Tp::SUPPORTED.map(|tp| (tp, 1 << 30));
        let emitted =
            package::emit_verified_direct_package(assets, &rank_bytes, &output, &verifier).unwrap();
        let host = runtime_bundle::BundleHostCompatibilityV1::new(Selected::ARCHITECTURE, 1)
            .unwrap()
            .with_capabilities(
                emitted
                    .manifest()
                    .resource_requirements()
                    .required_capabilities()
                    .iter()
                    .cloned(),
            );
        runtime_bundle::BundlePackageV1::verify_directory(&output, &host).unwrap();
        for tp in Tp::SUPPORTED {
            let manifest = emitted
                .manifest()
                .for_topology(model_mimo26_contract::topology::topology_id(tp))
                .unwrap();
            assert_eq!(manifest.kv_formats().len(), 1);
            let refusal = runtime_host::resolve_kv_cache_dtype_v1(
                &manifest,
                runtime_host::KvCacheDtypeV1::Fp8,
            )
            .unwrap_err();
            assert!(
                refusal.contains("declares no fp8 KV layout; it supports bf16"),
                "{refusal}"
            );
        }
        let binding: runtime_bundle::checkpoint_binding::CheckpointBinding =
            serde_json::from_slice(
                &std::fs::read(output.join("components/checkpoint-binding.json")).unwrap(),
            )
            .unwrap();
        assert_eq!(binding.repository, "fixture/third");
        assert_eq!(binding.revision, "0123456789abcdef0123456789abcdef01234567");
        assert_eq!(binding.source_lock_sha256, verifier.source_lock_sha256());
        assert_eq!(
            std::fs::read(output.join("components/frontend.json")).unwrap(),
            registry
                .component_for_flash(registry.identity("third").unwrap(), Encoders::NONE)
                .unwrap()
        );
        let component = std::fs::read(output.join("components/frontend.json")).unwrap();
        let descriptor =
            runtime_frontend_contract::FrontendDescriptorV1::from_component_bytes(&component)
                .unwrap();
        assert_eq!(
            runtime_frontend_contract::tokenizer_contract_v1(&descriptor).unwrap(),
            registry
                .tokenizer_contract_for_flash(verifier.identity())
                .unwrap()
        );
    }

    /// A checkpoint whose index no registered lock pins emits a package
    /// bound to its local identity, which passes full package verification
    /// and serves BF16 KV only.
    #[test]
    fn helper_admits_and_packages_a_local_checkpoint() {
        let mut fixture = registry_fixture::Fixture::new();
        fixture.state_tokenizer_semantics();
        std::fs::write(
            fixture.root.join("model.safetensors.index.json"),
            br#"{"metadata":{"total_size":2,"format":"local"},"weight_map":{"a":"weight.safetensors"}}"#,
        )
        .unwrap();
        let registry = fixture.registry();
        let verifier = MimoFamily::verifier_with_registry(&fixture.root, &registry).unwrap();
        let identity = verifier.identity();
        assert!(identity.is_local());
        let [tokenizer_definition, tokenizer_configuration] =
            package::PackageAssets::pinned_tokenizer_with_registry(
                &fixture.root,
                identity,
                verifier.registry(),
            )
            .unwrap();
        let assets = package::PackageAssets {
            tokenizer_definition,
            tokenizer_configuration,
            native_module: StagedContentV1::file(crate::test_native::fixture()),
            drafter: None,
            encoders: Encoders::NONE,
            precision: package::fixture_precision(),
        };
        let output = fixture.root.join("package");
        let rank_bytes = Tp::SUPPORTED.map(|tp| (tp, 1 << 30));
        let emitted =
            package::emit_verified_direct_package(assets, &rank_bytes, &output, &verifier).unwrap();
        let host = runtime_bundle::BundleHostCompatibilityV1::new(Selected::ARCHITECTURE, 1)
            .unwrap()
            .with_capabilities(
                emitted
                    .manifest()
                    .resource_requirements()
                    .required_capabilities()
                    .iter()
                    .cloned(),
            );
        runtime_bundle::BundlePackageV1::verify_directory(&output, &host).unwrap();
        for tp in Tp::SUPPORTED {
            let manifest = emitted
                .manifest()
                .for_topology(model_mimo26_contract::topology::topology_id(tp))
                .unwrap();
            let ids: Vec<_> = manifest
                .kv_formats()
                .iter()
                .map(|f| f.layout().id().as_str())
                .collect();
            assert_eq!(ids, ["mimo.ga-paged-bf16.v1"]);
        }
        let binding: runtime_bundle::checkpoint_binding::CheckpointBinding =
            serde_json::from_slice(
                &std::fs::read(output.join("components/checkpoint-binding.json")).unwrap(),
            )
            .unwrap();
        let digest = verifier.source_lock_sha256();
        assert_eq!(binding.repository, "local");
        assert_eq!(binding.revision, digest[..40]);
        assert_eq!(binding.served_model_name, format!("local-{digest}"));
        assert_eq!(binding.source_lock_sha256, digest);
        let component = std::fs::read(output.join("components/frontend.json")).unwrap();
        let descriptor =
            runtime_frontend_contract::FrontendDescriptorV1::from_component_bytes(&component)
                .unwrap();
        assert_eq!(descriptor.tokenizer.repository, binding.repository);
        assert_eq!(descriptor.tokenizer.revision, binding.revision);
    }
}
