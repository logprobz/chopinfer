//! The shared preparation provider. `probe` routes a checkpoint by its
//! `config.json` alone, before any header or weight is read. A direct
//! emission or profile request first admits the checkpoint in full,
//! `--activation-policy` included. The profile request then selects the
//! checkpoint layout's serving profile; the emission checks every text
//! tensor header the load's plans read, then writes the package of the
//! checkpoint in its layout (`emit`): its pinned frontend, the
//! checkpoint's tokenizer at the pinned revision (every export's is the
//! block-FP8 one's), the precision component, the planned rank images, the
//! planned encoder image when it serves images, and the installed native
//! module.
use model_compiler_sdk::files::Result;
use model_compiler_sdk::preparation::direct::DirectEmission;
use model_compiler_sdk::preparation::helper::Provider;
use model_compiler_sdk::quantization::ActivationPolicy;
use model_glm5next_contract::admission::{self, Matched};
use model_glm5next_contract::quantization::{
    FP8_BLOCK128, LAYOUT_OPTION, Layout, NVFP4_DENSE_ROUTED, NVFP4_ROUTED,
};
use runtime_bundle::preparation::{DirectProfile, Modality, PrepareDirectRequest};
use runtime_bundle_abi::weight_plan::WeightPlanRequestV1;
use std::collections::BTreeSet;
use std::path::Path;

/// The checkpoint's `config.json`, opened as the host opens it (a regular
/// file in the checkpoint, or a confined HF snapshot link to its blob) and
/// read within the admitted bound.
fn config(root: &Path) -> Result<Vec<u8>> {
    crate::checkpoint_headers::document(root, "config.json", admission::CONFIG_BYTES_MAX as u64)
}

/// The match of the `config.json` the header request `checkpoint` read.
fn admit_config(checkpoint: &WeightPlanRequestV1) -> Result<(Matched, serde_json::Value)> {
    let bytes = checkpoint
        .document("config.json")
        .ok_or("the checkpoint has no config.json")?;
    let config = admission::config(bytes)?;
    Ok((admission::admit(&config)?, config))
}

/// The modalities the checkpoint `checkpoint` describes (its headers and
/// documents) provides: text, and images when it stores a vision tower.
/// Video and audio are never served.
pub fn provided_modalities(checkpoint: &WeightPlanRequestV1) -> Result<BTreeSet<Modality>> {
    let mut provided = BTreeSet::from([Modality::Text]);
    if crate::encoder_image::has_tower(checkpoint)? {
        provided.insert(Modality::Image);
    }
    Ok(provided)
}

/// The modalities the package serves for `request` of the checkpoint
/// `checkpoint` describes, through the shared rule over
/// [`provided_modalities`]: `auto` and `all` serve text and, with a vision
/// tower, images; a list naming video or audio, or images without a tower,
/// is refused, and so is one without text. Serving images checks the tower
/// against the vision declaration, so the profile request, the modality
/// resolution and the emission refuse a tower the encoder cannot read
/// alike.
pub fn served_modalities(
    checkpoint: &WeightPlanRequestV1,
    request: &PrepareDirectRequest,
) -> Result<BTreeSet<Modality>> {
    let served = request.served_modalities(&provided_modalities(checkpoint)?)?;
    if !served.contains(&Modality::Text) {
        return Err("--modalities must include text, which every glm5next package serves".into());
    }
    if served.contains(&Modality::Image) {
        model_glm5next_contract::encoder_plan::check_tower(checkpoint)?;
    }
    Ok(served)
}

/// An admitted checkpoint: its match, the config it was admitted from (read
/// once, and the version the package is emitted from), the modalities its
/// package serves and the headers they were resolved from.
pub struct Admitted {
    pub matched: Matched,
    pub config: serde_json::Value,
    pub served: BTreeSet<Modality>,
    pub checkpoint: WeightPlanRequestV1,
}

/// Admit the checkpoint at `root` for `request`: its config, the rank count,
/// the modalities it serves (`served_modalities`, from its headers) and the
/// activation policy against its quantization layout. Every projection runs
/// BF16 activations, so `checkpoint` is refused for an export that declares
/// quantized activations (the NVFP4 export's static NVFP4 inputs), and
/// `auto` and `bf16` run its weights under BF16 activations.
pub fn admit(root: &Path, request: &PrepareDirectRequest) -> Result<Admitted> {
    admission::admit_bytes(&config(root)?)?;
    admission::check_ranks(request.tensor_parallel_size)?;
    let checkpoint = crate::checkpoint_headers::request(root, request.tensor_parallel_size)?;
    let (matched, config) = admit_config(&checkpoint)?;
    let served = served_modalities(&checkpoint, request)?;
    let policy = ActivationPolicy::from_precedence(request.activation_policy.as_deref(), None)?;
    matched.quantization.check_policy(policy)?;
    Ok(Admitted {
        matched,
        config,
        served,
        checkpoint,
    })
}

/// The encoder components a package serving `served` of the checkpoint
/// `checkpoint` describes carries: the planned encoder image when it serves
/// images (its tower checked), none otherwise.
pub fn encoder_components(
    checkpoint: &WeightPlanRequestV1,
    served: &BTreeSet<Modality>,
) -> Result<Vec<model_compiler_sdk::ComponentDraftV1>> {
    if !served.contains(&Modality::Image) {
        return Ok(Vec::new());
    }
    Ok(vec![crate::encoder_image::planned(checkpoint)?])
}

/// The installed native module beside the helper.
const NATIVE_MODULE: &str = "libbundle_glm5next.so";

/// The checkpoint files of the pinned block-FP8 revision, whose chat
/// template the frontend renders.
const PINNED_FILES: &str = include_str!("../../checkpoint-files.json");

// The serving profiles of every checkpoint layout.
use crate::package::PROFILES;

/// The serving profile `name` for a checkpoint of `layout`, refused when it
/// does not exist or its `model_options` assert another layout
/// ([`LAYOUT_OPTION`], which the package's manifest also constrains).
pub fn serving_profile(layout: &Layout, name: &str) -> Result<DirectProfile> {
    let file: serde_json::Value = serde_json::from_str(PROFILES)?;
    let profile = file["profiles"]
        .get(name)
        .ok_or_else(|| format!("glm5next declares no serving profile {name}"))?;
    let served = Layout::named(&profile["model_options"][LAYOUT_OPTION])?;
    if served != layout {
        return Err(format!(
            "serving profile {name} sets {LAYOUT_OPTION}={}, but the checkpoint's layout is {}",
            served.id, layout.id
        )
        .into());
    }
    Ok(DirectProfile {
        name: name.into(),
        profile: profile.clone(),
    })
}

/// The profile a direct package of a checkpoint of `layout` carries: MTP
/// with three draft tokens at context-parallel size 1, `tp4-mtp` for the
/// block-FP8 checkpoint, `tp4-mtp-nvfp4` for local-inference-lab's NVFP4
/// export and `tp4-mtp-nvfp4-nvidia` for NVIDIA's. The autoregressive and
/// context-parallel profiles are chosen by name.
pub fn select_profile(layout: &Layout) -> Result<DirectProfile> {
    let name = [
        (&FP8_BLOCK128, "tp4-mtp"),
        (&NVFP4_ROUTED, "tp4-mtp-nvfp4"),
        (&NVFP4_DENSE_ROUTED, "tp4-mtp-nvfp4-nvidia"),
    ]
    .into_iter()
    .find_map(|(own, name)| (own == layout).then_some(name))
    .ok_or_else(|| {
        format!(
            "glm5next selects no serving profile for the {} checkpoint",
            layout.id
        )
    })?;
    serving_profile(layout, name)
}

/// The checkpoint file `name` at `root`, refused unless it is the pinned
/// revision's (its length and SHA-256 in `checkpoint-files.json`).
fn pinned_file(root: &Path, name: &str, pins: &str) -> Result<Vec<u8>> {
    use sha2::{Digest, Sha256};
    let pins: serde_json::Value = serde_json::from_str(pins)?;
    let pin = &pins[name];
    let limit = pin["byte_length"]
        .as_u64()
        .ok_or_else(|| format!("{name} has no pinned length"))?;
    let bytes = crate::checkpoint_headers::document(root, name, limit)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if pin["byte_length"].as_u64() != Some(bytes.len() as u64) || pin["sha256"] != digest.as_str() {
        return Err(format!(
            "{name} is not the pinned revision's ({} bytes, sha256 {digest}); the frontend renders that revision's chat template",
            bytes.len()
        )
        .into());
    }
    Ok(bytes)
}

/// Emits into `output` the package of the admitted checkpoint at `root`
/// with the native module at `module`, and the serving profile it carries:
/// the frontend, the pinned tokenizer, the precision component of the
/// checkpoint's lowered projections, the planned rank images (with the MTP
/// drafter's views when the profile drafts), the planned encoder image when
/// it serves images, and the module.
pub fn emit(
    root: &Path,
    request: &PrepareDirectRequest,
    module: &Path,
    output: &Path,
) -> Result<DirectEmission> {
    emit_with_pins(root, request, module, output, PINNED_FILES)
}

// The public entry point always supplies the compiled revision pins. Tests
// use small tokenizer stand-ins with their own pins through the same reader.
fn emit_with_pins(
    root: &Path,
    request: &PrepareDirectRequest,
    module: &Path,
    output: &Path,
    pins: &str,
) -> Result<DirectEmission> {
    use model_compiler_sdk::{StagedContentV1, package_draft};
    use model_glm5next_contract::contract::Shape;
    let admitted = admit(root, request)?;
    let matched = admitted.matched;
    let layout = matched.quantization.layout;
    let profile = select_profile(layout)?;
    let precision =
        model_glm5next_contract::prepared::checkpoint_component(layout, &admitted.config)?;
    let mut components = vec![
        package_draft::frontend(StagedContentV1::bytes(crate::frontend::COMPONENT))?,
        package_draft::tokenizer_definition(StagedContentV1::bytes(pinned_file(
            root,
            "tokenizer.json",
            pins,
        )?))?,
        package_draft::tokenizer_configuration(StagedContentV1::bytes(pinned_file(
            root,
            "tokenizer_config.json",
            pins,
        )?))?,
        crate::package::precision(precision, layout)?,
    ];
    components.extend(encoder_components(&admitted.checkpoint, &admitted.served)?);
    // Every text tensor the plans read at load, checked before the package
    // is written.
    model_glm5next_contract::direct::check_text_headers(
        &admitted.checkpoint,
        &matched,
        request.tensor_parallel_size,
    )?;
    let drafter = match profile.profile["speculation"]["drafter"].as_str() {
        Some("mtp") => crate::package::Drafter::default(),
        _ => crate::package::Drafter::None,
    };
    // Admission refused every rank count but 4.
    let shape = Shape::new(matched.geometry, Shape::TP4.ranks);
    let draft = crate::package::draft(
        shape,
        layout,
        matched.runtime,
        components,
        StagedContentV1::file(module),
        drafter,
    )?;
    let package = model_compiler_sdk::emit_bundle_directory_v1(&draft, output)?;
    Ok(DirectEmission { package, profile })
}

pub struct Glm5NextFamily;

impl Provider for Glm5NextFamily {
    fn name(&self) -> &str {
        "chopinfer-model-glm5next"
    }
    fn deployment_policy(&self) -> Option<&str> {
        Some(crate::package::DEPLOYMENT_POLICY)
    }
    /// Every checkpoint layout's profiles, which the emission serves.
    fn profiles(&self) -> Vec<String> {
        let file: serde_json::Value =
            serde_json::from_str(PROFILES).expect("the serving profiles parse");
        file["profiles"]
            .as_object()
            .map(|profiles| {
                profiles
                    .iter()
                    .filter(|(_, profile)| {
                        Layout::named(&profile["model_options"][LAYOUT_OPTION]).is_ok()
                    })
                    .map(|(name, _)| name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }
    fn probe(&self, path: &Path, ranks: u32) -> Result<()> {
        admission::admit_bytes(&config(path)?)?;
        admission::check_ranks(ranks)?;
        Ok(())
    }
    fn emit_direct(
        &self,
        root: &Path,
        request: &PrepareDirectRequest,
        _checkpoint_identity: &str,
        output: &Path,
    ) -> Result<DirectEmission> {
        let module = model_compiler_sdk::preparation::direct::installed_native_module(
            NATIVE_MODULE,
            "GLM 5.3 Flash",
        )?;
        emit(root, request, &module, output)
    }
    fn names_direct_profile(&self) -> bool {
        true
    }
    /// The served set for every selection, the keywords resolved
    /// (`served_modalities`, from the checkpoint's headers), so a keyword and
    /// the list it serves share a published package.
    fn direct_modalities(
        &self,
        root: &Path,
        request: &PrepareDirectRequest,
    ) -> Result<Option<BTreeSet<Modality>>> {
        admission::admit_bytes(&config(root)?)?;
        let checkpoint = crate::checkpoint_headers::request(root, request.tensor_parallel_size)?;
        admit_config(&checkpoint)?;
        served_modalities(&checkpoint, request).map(Some)
    }
    fn direct_profile(&self, root: &Path, request: &PrepareDirectRequest) -> Result<DirectProfile> {
        select_profile(admit(root, request)?.matched.quantization.layout)
    }
}

#[cfg(test)]
#[path = "test_native.rs"]
mod test_native;

#[cfg(test)]
mod tests {
    use super::*;
    use model_glm5next_contract::quantization::LAYOUTS;

    fn probe(config: &serde_json::Value, ranks: u32) -> std::result::Result<(), String> {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "glm5next-probe-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(config).unwrap(),
        )
        .unwrap();
        let result = Glm5NextFamily
            .probe(&root, ranks)
            .map_err(|e| e.to_string());
        std::fs::remove_dir_all(root).unwrap();
        result
    }

    fn config() -> serde_json::Value {
        serde_json::from_slice(include_bytes!("../../contract/fixtures/config.json")).unwrap()
    }

    /// A checkpoint directory holding `config` and the named `files` of
    /// `source`, the emission request with `policy` for it, and a stand-in
    /// native module; removed when dropped.
    struct Checkpoint {
        root: std::path::PathBuf,
        request: PrepareDirectRequest,
        module: std::path::PathBuf,
        pins: Option<String>,
    }

    impl Checkpoint {
        fn new(config: &[u8], policy: Option<&str>, source: Option<&Path>, files: &[&str]) -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "glm5next-emit-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join("config.json"), config).unwrap();
            for name in files {
                std::fs::copy(source.unwrap().join(name), root.join(name)).unwrap();
            }
            let module = root.join(NATIVE_MODULE);
            std::fs::write(&module, b"module").unwrap();
            let request = PrepareDirectRequest {
                path: root.clone(),
                cache_root: root.join("cache"),
                tensor_parallel_size: 4,
                activation_policy: policy.map(str::to_owned),
                modalities: Default::default(),
            };
            Self {
                root,
                request,
                module,
                pins: None,
            }
        }

        /// Catalog headers at their logical lengths, small tokenizer
        /// stand-ins pinned before any mutation, and a native object fixture.
        fn bounded(layout: &Layout, config: &[u8]) -> Self {
            use sha2::{Digest, Sha256};
            let mut checkpoint = Self::new(config, None, None, &[]);
            let mut pins = serde_json::Map::new();
            for name in ["tokenizer.json", "tokenizer_config.json"] {
                let bytes = b"{}\n";
                std::fs::write(checkpoint.root.join(name), bytes).unwrap();
                pins.insert(
                    name.into(),
                    serde_json::json!({
                        "byte_length": bytes.len(),
                        "sha256": format!("{:x}", Sha256::digest(bytes)),
                    }),
                );
            }
            checkpoint.pins = Some(serde_json::to_string(&pins).unwrap());
            let headers = model_glm5next_contract::direct::test_support::checkpoint_request(
                layout,
                config,
                0,
                |_| {},
            );
            let checkpoint = checkpoint.with_request(headers);
            std::fs::copy(test_native::fixture(), &checkpoint.module).unwrap();
            checkpoint
        }

        fn with_request(self, mut headers: WeightPlanRequestV1) -> Self {
            headers
                .documents
                .retain(|document| document.name != "config.json");
            crate::checkpoint_headers::tests::write_headers(&self.root, &headers);
            self
        }

        /// Writes the headers of a checkpoint file holding every catalog
        /// tensor, after `edit`, and the weight map; no tensor bytes.
        fn with_headers(
            self,
            edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
        ) -> Self {
            let mut request = model_glm5next_contract::direct::test_support::request(0, edit);
            // The edited tensors laid back to back again, so the file's
            // spans stay contiguous.
            let mut header: serde_json::Map<String, serde_json::Value> =
                serde_json::from_slice(&request.files[0].header).unwrap();
            let mut offset = 0;
            for tensor in header.values_mut() {
                let span = &tensor["data_offsets"];
                let bytes = span[1].as_u64().unwrap() - span[0].as_u64().unwrap();
                tensor["data_offsets"] = serde_json::json!([offset, offset + bytes]);
                offset += bytes;
            }
            request.files[0].header = serde_json::to_vec(&header).unwrap();
            crate::checkpoint_headers::tests::index_headers(&mut request);
            request
                .documents
                .retain(|document| document.name != "config.json");
            crate::checkpoint_headers::tests::write_headers(&self.root, &request);
            self
        }

        fn emit(&self) -> Result<DirectEmission> {
            if let Some(pins) = &self.pins {
                return emit_with_pins(
                    &self.root,
                    &self.request,
                    &self.module,
                    &self.root.join("out"),
                    pins,
                );
            }
            emit(
                &self.root,
                &self.request,
                &self.module,
                &self.root.join("out"),
            )
        }
    }

    /// Always runs a bounded case in the ordinary library suite. Configured
    /// exports add real metadata and pinned tokenizer coverage. An explicit
    /// native-module override is verified as supplied, never skipped.
    fn emission_checkpoints(variable: &str, layout: &Layout, config: &[u8]) -> Vec<Checkpoint> {
        let mut checkpoints = vec![Checkpoint::bounded(layout, config)];
        if let Some(source) = std::env::var_os(variable) {
            let source = Path::new(&source);
            let config = std::fs::read(source.join("config.json")).unwrap();
            let checkpoint = Checkpoint::new(
                &config,
                None,
                Some(source),
                &["tokenizer.json", "tokenizer_config.json"],
            );
            let headers = crate::checkpoint_headers::request(source, 4).unwrap();
            let lengths: Vec<_> = headers
                .files
                .iter()
                .map(|file| (file.name.clone(), file.byte_length))
                .collect();
            let checkpoint = checkpoint.with_request(headers);
            for (name, byte_length) in lengths {
                assert_eq!(
                    std::fs::metadata(checkpoint.root.join(name)).unwrap().len(),
                    byte_length,
                    "the sparse export fixture retains the original logical length",
                );
            }
            std::fs::copy(test_native::module(), &checkpoint.module).unwrap();
            checkpoints.push(checkpoint);
        }
        eprintln!(
            "{}: 1 bounded case, {} configured export case(s)",
            layout.id,
            checkpoints.len() - 1
        );
        checkpoints
    }

    impl Drop for Checkpoint {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    /// A direct emission request with `policy` for a checkpoint holding
    /// only `config`: the emission's error, after checking the profile
    /// request admits the same way (and selects a profile when admitted).
    fn emit_error(config: &[u8], policy: Option<&str>) -> String {
        let checkpoint = Checkpoint::new(config, policy, None, &[]);
        let error = checkpoint.emit().err().unwrap().to_string();
        let profile = Glm5NextFamily.direct_profile(&checkpoint.root, &checkpoint.request);
        if let Err(profile) = profile {
            assert_eq!(profile.to_string(), error);
        }
        error
    }

    fn direct_profile(config: &[u8]) -> String {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "glm5next-profile-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), config).unwrap();
        let request = PrepareDirectRequest {
            path: root.clone(),
            cache_root: root.join("cache"),
            tensor_parallel_size: 4,
            activation_policy: None,
            modalities: Default::default(),
        };
        let profile = Glm5NextFamily.direct_profile(&root, &request);
        std::fs::remove_dir_all(root).unwrap();
        profile.unwrap().name
    }

    /// Each checkpoint selects its own layout's MTP profile, and a profile
    /// of another layout is refused for it.
    #[test]
    fn the_checkpoint_layout_selects_its_serving_profile() {
        let nvfp4: &[u8] = include_bytes!("../../contract/fixtures/nvfp4-config.json");
        let nvidia: &[u8] = include_bytes!("../../contract/fixtures/nvidia-nvfp4-config.json");
        let fp8: &[u8] = include_bytes!("../../contract/fixtures/config.json");
        assert_eq!(direct_profile(fp8), "tp4-mtp");
        assert_eq!(direct_profile(nvfp4), "tp4-mtp-nvfp4");
        assert_eq!(direct_profile(nvidia), "tp4-mtp-nvfp4-nvidia");
        for layout in [&NVFP4_ROUTED, &NVFP4_DENSE_ROUTED] {
            let selected = select_profile(layout).unwrap();
            assert_eq!(selected.profile["speculation"]["drafter"], "mtp");
            assert_eq!(selected.profile["speculation"]["tokens"], 3);
            assert!(selected.profile.get("context_parallel_size").is_none());
            assert_eq!(selected.profile["status"], "qualified");
            assert_eq!(selected.profile["qualified_max_sequences"], 32);
        }

        let file: serde_json::Value = serde_json::from_str(PROFILES).unwrap();
        let mut names: Vec<_> = file["profiles"].as_object().unwrap().keys().collect();
        names.sort();
        assert_eq!(
            names,
            [
                "tp4-ar",
                "tp4-mtp",
                "tp4-mtp-dcp2",
                "tp4-mtp-dcp4",
                "tp4-mtp-k5",
                "tp4-mtp-nvfp4",
                "tp4-mtp-nvfp4-nvidia"
            ]
        );
        for name in names {
            let own =
                Layout::named(&file["profiles"][name.as_str()]["model_options"][LAYOUT_OPTION])
                    .unwrap();
            assert_eq!(serving_profile(own, name).unwrap().name, *name);
            for other in LAYOUTS.iter().filter(|other| *other != own) {
                assert_eq!(
                    serving_profile(other, name).unwrap_err().to_string(),
                    format!(
                        "serving profile {name} sets {LAYOUT_OPTION}={}, but the checkpoint's \
                         layout is {}",
                        own.id, other.id
                    )
                );
            }
        }
        // The NVIDIA export is refused the local-inference-lab profile.
        assert_eq!(
            serving_profile(&NVFP4_DENSE_ROUTED, "tp4-mtp-nvfp4")
                .unwrap_err()
                .to_string(),
            "serving profile tp4-mtp-nvfp4 sets checkpoint.quantization_layout=nvfp4-routed, but \
             the checkpoint's layout is nvfp4-dense-routed"
        );
        assert_eq!(
            serving_profile(&FP8_BLOCK128, "tp2-mtp")
                .unwrap_err()
                .to_string(),
            "glm5next declares no serving profile tp2-mtp"
        );
    }

    #[test]
    fn the_activation_policy_is_admitted_against_the_quantization_layout() {
        let nvfp4: &[u8] = include_bytes!("../../contract/fixtures/nvfp4-config.json");
        let fp8: &[u8] = include_bytes!("../../contract/fixtures/config.json");
        // auto (the default) and bf16 run BF16 activations: admitted, then
        // each emission reads the checkpoint's tokenizer, which these
        // checkpoints lack.
        for policy in [None, Some("auto"), Some("bf16")] {
            for config in [nvfp4, fp8] {
                let error = emit_error(config, policy);
                assert!(error.contains("tokenizer.json"), "{policy:?}: {error}");
            }
        }
        assert_eq!(
            emit_error(nvfp4, Some("checkpoint")),
            "unsupported activation policy checkpoint: \
             model.language_model.layers.3.mlp.experts.0.gate_proj declares Nvfp4Static \
             activations and glm5next runs BF16 activations only; use auto or bf16"
        );
        // The block-FP8 export declares dynamic FP8 activations.
        assert!(
            emit_error(fp8, Some("checkpoint"))
                .starts_with("unsupported activation policy checkpoint: quantization_config"),
        );
        assert!(emit_error(nvfp4, Some("w4a4")).contains("expected auto, checkpoint or bf16"));
    }

    /// Each admitted layout's catalog includes a vision tower, so its
    /// bounded checkpoint and each configured export serve images by default.
    #[test]
    fn the_admitted_checkpoints_serve_images_by_default() {
        for (variable, layout, config) in [
            (
                "CHOPINFER_GLM5NEXT_CHECKPOINT",
                &FP8_BLOCK128,
                include_bytes!("../../contract/fixtures/config.json").as_slice(),
            ),
            (
                "CHOPINFER_GLM5NEXT_NVFP4_CHECKPOINT",
                &NVFP4_ROUTED,
                include_bytes!("../../contract/fixtures/nvfp4-config.json").as_slice(),
            ),
            (
                "CHOPINFER_GLM5NEXT_NVIDIA_NVFP4_CHECKPOINT",
                &NVFP4_DENSE_ROUTED,
                include_bytes!("../../contract/fixtures/nvidia-nvfp4-config.json").as_slice(),
            ),
        ] {
            for checkpoint in emission_checkpoints(variable, layout, config) {
                assert_eq!(
                    Glm5NextFamily
                        .direct_modalities(&checkpoint.root, &checkpoint.request)
                        .unwrap(),
                    Some(BTreeSet::from([Modality::Text, Modality::Image])),
                    "{}",
                    layout.id,
                );
            }
        }
    }

    #[test]
    fn probe_routes_the_pinned_config_at_tp4() {
        probe(&config(), 4).unwrap();
        assert_eq!(
            probe(&config(), 2).unwrap_err(),
            "unsupported tensor parallel size 2: glm5next serves [4]"
        );
    }

    #[test]
    fn probe_refuses_another_family_and_an_unlisted_geometry_before_headers() {
        let mut other = config();
        other["model_type"] = "qwen38_flash".into();
        assert!(
            probe(&other, 4)
                .unwrap_err()
                .starts_with("unsupported model_type:")
        );
        let mut changed = config();
        changed["text_config"]["hidden_size"] = 5120.into();
        let error = probe(&changed, 4).unwrap_err();
        assert!(
            error.starts_with("unsupported geometry:") && error.contains("text_config.hidden_size"),
            "{error}"
        );
    }

    /// A unique directory under the test temporary directory.
    fn scratch(kind: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "glm5next-{kind}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// The config is read as the host reads checkpoint files: a link out of
    /// the checkpoint and a config past its bound are refused, and an HF
    /// snapshot's link to its blob is followed.
    #[test]
    fn probe_reads_the_config_confined_and_bounded() {
        let fp8: &[u8] = include_bytes!("../../contract/fixtures/config.json");
        let base = scratch("confined");
        let linked = base.join("linked");
        std::fs::create_dir_all(&linked).unwrap();
        std::fs::write(base.join("elsewhere.json"), fp8).unwrap();
        std::os::unix::fs::symlink("../elsewhere.json", linked.join("config.json")).unwrap();
        let error = Glm5NextFamily.probe(&linked, 4).unwrap_err().to_string();
        assert!(error.contains("links are permitted only"), "{error}");
        let oversized = base.join("oversized");
        std::fs::create_dir_all(&oversized).unwrap();
        let mut padded = fp8.to_vec();
        padded.resize(admission::CONFIG_BYTES_MAX + 1, b' ');
        std::fs::write(oversized.join("config.json"), padded).unwrap();
        let error = Glm5NextFamily.probe(&oversized, 4).unwrap_err().to_string();
        assert!(error.contains("exceeds limit"), "{error}");
        let hub = base.join("models--zai-org--GLM");
        let snapshot = hub.join("snapshots").join("rev");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(hub.join("blobs")).unwrap();
        std::fs::write(hub.join("blobs").join("config-blob"), fp8).unwrap();
        std::os::unix::fs::symlink("../../blobs/config-blob", snapshot.join("config.json"))
            .unwrap();
        Glm5NextFamily.probe(&snapshot, 4).unwrap();
        std::fs::remove_dir_all(base).unwrap();
    }

    /// A FIFO in place of the config is refused without waiting for a
    /// writer.
    #[test]
    fn probe_refuses_a_fifo_config() {
        let root = scratch("fifo");
        let status = std::process::Command::new("mkfifo")
            .arg(root.join("config.json"))
            .status()
            .unwrap();
        assert!(status.success());
        let error = Glm5NextFamily.probe(&root, 4).unwrap_err().to_string();
        assert!(error.contains("regular file"), "{error}");
        std::fs::remove_dir_all(root).unwrap();
    }

    /// A text tensor whose offsets leave its shard, or overlap another
    /// tensor's, refuses the emission before any package is written, as the
    /// host's checkpoint opener would refuse the load.
    #[test]
    fn emission_refuses_offsets_outside_the_shard_or_overlapping() {
        const KV_B: &str = "model.language_model.layers.3.self_attn.kv_b_proj.weight";
        let fp8: &[u8] = include_bytes!("../../contract/fixtures/config.json");
        for (case, expected) in [("outside", "gap"), ("overlap", "overlap")] {
            let checkpoint = Checkpoint::new(fp8, None, None, &[]).with_headers(|_| {});
            let path = checkpoint.root.join("model.safetensors");
            // The sparse file's header only; its payload is never read.
            use std::io::Read;
            let mut file = std::fs::File::open(&path).unwrap();
            let length = file.metadata().unwrap().len();
            let mut prefix = [0_u8; 8];
            file.read_exact(&mut prefix).unwrap();
            let mut bytes = vec![0; u64::from_le_bytes(prefix) as usize];
            file.read_exact(&mut bytes).unwrap();
            let payload = length - 8 - bytes.len() as u64;
            let mut header: serde_json::Map<String, serde_json::Value> =
                serde_json::from_slice(&bytes).unwrap();
            let offsets = &header[KV_B]["data_offsets"];
            let span = offsets[1].as_u64().unwrap() - offsets[0].as_u64().unwrap();
            let start = if case == "outside" { payload + 128 } else { 0 };
            header[KV_B]["data_offsets"] = serde_json::json!([start, start + span]);
            let header = serde_json::to_vec(&header).unwrap();
            crate::checkpoint_headers::tests::write_sparse(
                &path,
                &header,
                8 + header.len() as u64 + payload,
            );
            let error = checkpoint.emit().err().unwrap().to_string();
            assert!(
                error.starts_with("model.safetensors: ") && error.contains(expected),
                "{case}: {error}"
            );
            assert!(!checkpoint.root.join("out").exists());
        }
    }

    /// Missing and incompatible text tensors refuse publication. The same
    /// bounded fixture first proves that complete headers emit a package.
    #[test]
    fn emission_refuses_text_headers_the_plan_would_refuse() {
        const KV_B: &str = "model.language_model.layers.3.self_attn.kv_b_proj.weight";
        let fp8: &[u8] = include_bytes!("../../contract/fixtures/config.json");
        Checkpoint::bounded(&FP8_BLOCK128, fp8).emit().unwrap();
        type Edit = fn(&mut serde_json::Map<String, serde_json::Value>);
        let cases: [(Edit, &str); 2] = [
            (
                |header| {
                    header.remove(KV_B);
                },
                "kv_b_proj.weight is not in the weight map",
            ),
            (
                |header| header[KV_B]["dtype"] = serde_json::json!("F16"),
                "is F16 in model.safetensors, the catalog BF16",
            ),
        ];
        for (edit, expected) in cases {
            let checkpoint = Checkpoint::bounded(&FP8_BLOCK128, fp8).with_headers(edit);
            let error = checkpoint.emit().err().unwrap().to_string();
            assert!(error.contains(expected), "{error}");
            assert!(!checkpoint.root.join("out").exists());
        }
    }

    /// Both bounded and configured block-FP8 metadata emit the pinned frontend,
    /// precision and media components. Text-only emission omits the encoder;
    /// same-size tokenizer corruption and an oversized tokenizer fail distinctly.
    #[test]
    fn the_fp8_checkpoint_emits_its_package() {
        for mut checkpoint in emission_checkpoints(
            "CHOPINFER_GLM5NEXT_CHECKPOINT",
            &FP8_BLOCK128,
            include_bytes!("../../contract/fixtures/config.json"),
        ) {
            let emitted = checkpoint.emit().unwrap();
            assert_eq!(emitted.profile.name, "tp4-mtp");
            let out = checkpoint.root.join("out");
            let precision = std::fs::read(out.join(crate::package::PRECISION_PATH)).unwrap();
            model_glm5next_contract::prepared::decode(&precision).unwrap();
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(out.join("manifest.json")).unwrap()).unwrap();
            let roles: Vec<_> = manifest["components"]
                .as_array()
                .unwrap()
                .iter()
                .map(|component| component["role"].as_str().unwrap().to_owned())
                .collect();
            assert!(roles.contains(&model_glm5next_contract::prepared::COMPONENT_ROLE.to_owned()));
            // The package carries the model's profiles and deployment policy
            // byte for byte.
            for (path, file) in [
                (
                    runtime_bundle::deployment::SERVE_PROFILES_PATH_V1,
                    "profiles.json",
                ),
                (
                    runtime_bundle::deployment::DEPLOYMENT_POLICY_PATH_V1,
                    "deploy/policy.json",
                ),
            ] {
                let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
                assert_eq!(
                    std::fs::read(out.join(path)).unwrap(),
                    std::fs::read(model.join(file)).unwrap()
                );
            }
            let media = |manifest: &serde_json::Value| {
                let encoder = manifest["planned_components"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|component| component["id"] == crate::encoder_image::IMAGE_COMPONENT);
                let capability = manifest["resource_requirements"]["required_capabilities"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|capability| {
                        capability == model_glm5next_contract::encoder::MEDIA_CAPABILITY
                    });
                assert_eq!(encoder, capability);
                encoder
            };
            assert!(media(&manifest));
            checkpoint.request.modalities = selection(&[Modality::Text]);
            std::fs::remove_dir_all(&out).unwrap();
            checkpoint.emit().unwrap();
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(out.join("manifest.json")).unwrap()).unwrap();
            assert!(!media(&manifest));
            // Another revision's tokenizer configuration.
            std::fs::remove_dir_all(&out).unwrap();
            let path = checkpoint.root.join("tokenizer_config.json");
            let mut bytes = std::fs::read(&path).unwrap();
            bytes[0] ^= 1;
            std::fs::write(&path, bytes).unwrap();
            let error = checkpoint.emit().err().unwrap().to_string();
            assert!(error.contains("not the pinned revision's"), "{error}");
            assert!(!out.exists());
            let mut oversized = std::fs::read(&path).unwrap();
            oversized.push(b'\n');
            std::fs::write(&path, oversized).unwrap();
            let error = checkpoint.emit().err().unwrap().to_string();
            assert!(
                error.contains("tokenizer_config.json: checkpoint metadata exceeds limit"),
                "{error}"
            );
            assert!(!out.exists());
        }
    }

    /// Both NVFP4 layouts emit their own profile and precision document,
    /// using bounded fixtures plus each configured export's sparse headers.
    #[test]
    fn each_nvfp4_checkpoint_emits_its_text_package() {
        for (variable, layout, config, name) in [
            (
                "CHOPINFER_GLM5NEXT_NVFP4_CHECKPOINT",
                &NVFP4_ROUTED,
                include_bytes!("../../contract/fixtures/nvfp4-config.json").as_slice(),
                "tp4-mtp-nvfp4",
            ),
            (
                "CHOPINFER_GLM5NEXT_NVIDIA_NVFP4_CHECKPOINT",
                &NVFP4_DENSE_ROUTED,
                include_bytes!("../../contract/fixtures/nvidia-nvfp4-config.json").as_slice(),
                "tp4-mtp-nvfp4-nvidia",
            ),
        ] {
            for checkpoint in emission_checkpoints(variable, layout, config) {
                let emitted = checkpoint.emit().unwrap();
                assert_eq!(emitted.profile.name, name);
                assert_eq!(
                    emitted.profile.profile["model_options"][LAYOUT_OPTION],
                    layout.id
                );
                let out = checkpoint.root.join("out");
                let precision = std::fs::read(out.join(crate::package::PRECISION_PATH)).unwrap();
                model_glm5next_contract::prepared::decode_for(&precision, layout).unwrap();
                assert!(model_glm5next_contract::prepared::decode(&precision).is_err());
            }
        }
    }

    /// Every layout's profiles are the helper's.
    #[test]
    fn the_helper_describes_every_layout_s_profiles() {
        let mut profiles = Glm5NextFamily.profiles();
        profiles.sort();
        assert_eq!(
            profiles,
            [
                "tp4-ar",
                "tp4-mtp",
                "tp4-mtp-dcp2",
                "tp4-mtp-dcp4",
                "tp4-mtp-k5",
                "tp4-mtp-nvfp4",
                "tp4-mtp-nvfp4-nvidia"
            ]
        );
    }

    /// The modalities each request resolves to on `checkpoint`, by the
    /// profile request, the modality resolution and the emission alike
    /// (admitted, the emission then reads the absent tokenizer), or the one
    /// refusal all three give.
    fn resolved(checkpoint: &Checkpoint) -> std::result::Result<BTreeSet<Modality>, String> {
        let profile = Glm5NextFamily.direct_profile(&checkpoint.root, &checkpoint.request);
        let modalities = Glm5NextFamily.direct_modalities(&checkpoint.root, &checkpoint.request);
        let emitted = checkpoint.emit().err().unwrap().to_string();
        match modalities {
            Ok(served) => {
                assert_eq!(profile.unwrap().name, "tp4-mtp");
                assert!(emitted.contains("tokenizer.json"), "{emitted}");
                Ok(served.unwrap())
            }
            Err(error) => {
                let error = error.to_string();
                assert_eq!(profile.err().unwrap().to_string(), error);
                assert_eq!(emitted, error);
                Err(error)
            }
        }
    }

    fn selection(modalities: &[Modality]) -> runtime_bundle::preparation::ModalitySelection {
        runtime_bundle::preparation::ModalitySelection::Only(modalities.iter().copied().collect())
    }

    /// A checkpoint with a vision tower serves images: `auto`, `all` and
    /// `text,image` resolve to text and image, `text` to text alone, and a
    /// list naming video or audio, or no text, is refused before the
    /// checkpoint's tokenizer is read.
    #[test]
    fn a_checkpoint_with_a_vision_tower_serves_images_and_refuses_video() {
        use runtime_bundle::preparation::ModalitySelection;
        let fp8: &[u8] = include_bytes!("../../contract/fixtures/config.json");
        let both = BTreeSet::from([Modality::Text, Modality::Image]);
        for (modalities, expected) in [
            (ModalitySelection::Auto, both.clone()),
            (ModalitySelection::All, both.clone()),
            (selection(&[Modality::Text, Modality::Image]), both.clone()),
            (
                selection(&[Modality::Text]),
                BTreeSet::from([Modality::Text]),
            ),
        ] {
            let mut checkpoint = Checkpoint::new(fp8, None, None, &[]).with_headers(|_| {});
            checkpoint.request.modalities = modalities;
            assert_eq!(resolved(&checkpoint).unwrap(), expected);
        }
        for unserved in [Modality::Video, Modality::Audio] {
            let mut checkpoint = Checkpoint::new(fp8, None, None, &[]).with_headers(|_| {});
            checkpoint.request.modalities = selection(&[Modality::Text, unserved]);
            assert_eq!(
                resolved(&checkpoint).unwrap_err(),
                format!(
                    "--modalities names {}, which this checkpoint does not provide; it provides \
                     text,image",
                    unserved.name()
                )
            );
        }
        let mut checkpoint = Checkpoint::new(fp8, None, None, &[]).with_headers(|_| {});
        checkpoint.request.modalities = selection(&[Modality::Image]);
        assert!(
            resolved(&checkpoint)
                .unwrap_err()
                .contains("must include text")
        );
    }

    /// A checkpoint without a vision tower serves text alone, and refuses
    /// images; one whose tower the encoder cannot read is refused images
    /// (by default too) and still serves text.
    #[test]
    fn images_need_a_tower_the_vision_declaration_binds() {
        use runtime_bundle::preparation::ModalitySelection;
        let fp8: &[u8] = include_bytes!("../../contract/fixtures/config.json");
        let text = BTreeSet::from([Modality::Text]);
        let towerless = || {
            Checkpoint::new(fp8, None, None, &[])
                .with_headers(|header| header.retain(|name, _| !name.starts_with("model.visual.")))
        };
        for modalities in [ModalitySelection::Auto, ModalitySelection::All] {
            let mut checkpoint = towerless();
            checkpoint.request.modalities = modalities;
            assert_eq!(resolved(&checkpoint).unwrap(), text);
        }
        // A stale shard of vision tensors the weight map does not name.
        let checkpoint = towerless();
        let mut orphan = model_glm5next_contract::direct::test_support::request(0, |_| {});
        orphan.files[0].name = "stale-vision.safetensors".into();
        orphan.documents.clear();
        crate::checkpoint_headers::tests::write_headers(&checkpoint.root, &orphan);
        assert_eq!(resolved(&checkpoint).unwrap(), text);
        let mut checkpoint = towerless();
        checkpoint.request.modalities = selection(&[Modality::Text, Modality::Image]);
        assert_eq!(
            resolved(&checkpoint).unwrap_err(),
            "--modalities names image, which this checkpoint does not provide; it provides text"
        );
        // A checkpoint directory holding no safetensors file stores no
        // tower either.
        let checkpoint = Checkpoint::new(fp8, None, None, &[]);
        assert_eq!(resolved(&checkpoint).unwrap(), text);
        let broken = || {
            Checkpoint::new(fp8, None, None, &[]).with_headers(|header| {
                header.remove("model.visual.merger.down_proj.weight");
            })
        };
        let error = resolved(&broken()).unwrap_err();
        assert!(error.contains("merger.down_proj.weight"), "{error}");
        let mut checkpoint = broken();
        checkpoint.request.modalities = selection(&[Modality::Text]);
        assert_eq!(resolved(&checkpoint).unwrap(), text);
    }

    /// A package serving images carries the planned encoder image, which
    /// makes it require media inputs (the host then serves the frontend's
    /// image modality) and adds the encoder's buffers to each rank's device
    /// memory; a text package carries neither.
    #[test]
    fn an_image_package_plans_the_encoder_and_requires_media_inputs() {
        use model_compiler_sdk::{StagedContentV1, package_draft};
        use model_glm5next_contract::contract::Shape;
        use model_glm5next_contract::geometry::RuntimeValues;
        let checkpoint = model_glm5next_contract::direct::test_support::request(0, |_| {});
        let manifest = |served: &[Modality]| {
            let mut components = vec![
                package_draft::frontend(StagedContentV1::bytes(crate::frontend::COMPONENT))
                    .unwrap(),
            ];
            components.extend(
                encoder_components(&checkpoint, &served.iter().copied().collect()).unwrap(),
            );
            let draft = crate::package::draft(
                Shape::TP4,
                &FP8_BLOCK128,
                RuntimeValues { index_topk: 2048 },
                components,
                StagedContentV1::bytes(b"module".to_vec()),
                crate::package::Drafter::default(),
            )
            .unwrap();
            model_compiler_sdk::plan_bundle_manifest_v1(&draft).unwrap()
        };
        let image = manifest(&[Modality::Text, Modality::Image]);
        let text = manifest(&[Modality::Text]);
        let planned = |manifest: &runtime_bundle::BundleManifestV1| {
            let value = serde_json::to_value(manifest).unwrap();
            value["planned_components"]
                .as_array()
                .unwrap()
                .iter()
                .map(|component| component["id"].as_str().unwrap().to_owned())
                .collect::<Vec<_>>()
        };
        assert!(planned(&image).contains(&crate::encoder_image::IMAGE_COMPONENT.to_owned()));
        assert!(!planned(&text).contains(&crate::encoder_image::IMAGE_COMPONENT.to_owned()));
        assert!(runtime_host::video::media_inputs_required_v1(&image));
        assert!(!runtime_host::video::media_inputs_required_v1(&text));
        assert_eq!(
            image.resource_requirements().minimum_device_memory_bytes(),
            text.resource_requirements().minimum_device_memory_bytes()
                + model_glm5next_contract::encoder::buffer_bytes()
        );
    }
}
