//! The shared preparation provider. `probe` routes a checkpoint by its
//! `config.json` alone, before any header or weight is read. A direct
//! emission or profile request first admits the checkpoint in full,
//! `--activation-policy` included. The profile request then selects the
//! checkpoint layout's serving profile; the emission checks every text
//! tensor header the load's plans read, then writes the package of the
//! checkpoint in its layout (`emit`): the checkpoint's own tokenizer files
//! with the frontend that names them (the declared one for the pinned
//! revision's files), the precision component, the planned rank images,
//! the planned encoder image when it serves images, and the installed
//! native module.
use model_compiler_sdk::files::Result;
use model_compiler_sdk::preparation::direct::DirectEmission;
use model_compiler_sdk::preparation::helper::Provider;
use model_compiler_sdk::quantization::ActivationPolicy;
use model_glm5next_contract::admission::{self, Matched};
use model_glm5next_contract::quantization::{LAYOUT_OPTION, Layout};
use runtime_bundle::preparation::{DirectProfile, Modality, PrepareDirectRequest};
use runtime_bundle_abi::weight_plan::WeightPlanRequestV1;
use std::collections::{BTreeMap, BTreeSet};
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

/// An admitted checkpoint: its match, its layout (its class encodings with
/// its headers' side tensor dtypes), the config it was admitted from (read
/// once, and the version the package is emitted from), the modalities its
/// package serves and the headers they were resolved from.
pub struct Admitted {
    pub matched: Matched,
    pub layout: &'static Layout,
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
    let layout = model_glm5next_contract::direct::resolve_layout(&checkpoint, &matched)?;
    let served = served_modalities(&checkpoint, request)?;
    let policy = ActivationPolicy::from_precedence(request.activation_policy.as_deref(), None)?;
    matched.quantization.check_policy(policy)?;
    Ok(Admitted {
        matched,
        layout,
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

/// The checkpoint files of the pinned block-FP8 revision, whose tokenizer
/// the declared frontend names.
const PINNED_FILES: &str = include_str!("../../checkpoint-files.json");

// The serving profiles of every checkpoint layout.
use crate::package::PROFILES;

/// The serving profile `name` for a checkpoint of `layout`, refused when it
/// does not exist or its `model_options` assert another layout
/// ([`LAYOUT_OPTION`], which the package's manifest also constrains). The
/// shipped profiles assert none, so each serves every layout.
pub fn serving_profile(layout: &Layout, name: &str) -> Result<DirectProfile> {
    let file: serde_json::Value = serde_json::from_str(PROFILES)?;
    let profile = file["profiles"]
        .get(name)
        .ok_or_else(|| format!("glm5next declares no serving profile {name}"))?;
    if let Some(asserted) = profile["model_options"].get(LAYOUT_OPTION) {
        let served = Layout::named(asserted)?;
        if served != layout {
            return Err(format!(
                "serving profile {name} sets {LAYOUT_OPTION}={}, but the checkpoint's layout is {}",
                served.id, layout.id
            )
            .into());
        }
    }
    Ok(DirectProfile {
        name: name.into(),
        profile: profile.clone(),
    })
}

/// The profile a direct package of a checkpoint of `layout` carries:
/// `tp4-mtp`, MTP with three draft tokens at context-parallel size 1, for
/// every admitted layout, since admission requires a served encoding for
/// the MTP layer's experts as for every other class. The autoregressive and
/// context-parallel profiles are chosen by name.
pub fn select_profile(layout: &Layout) -> Result<DirectProfile> {
    serving_profile(layout, "tp4-mtp")
}

/// The official release's tokenizer configuration, which another
/// checkpoint's must state ([`TokenizerSemantics`]).
const TOKENIZER_SEMANTICS: &[u8] = include_bytes!("../fixtures/tokenizer-semantics.json");

/// The most bytes read of a checkpoint's tokenizer definition, and of its
/// tokenizer configuration or chat template.
const DEFINITION_BYTES: u64 = 64 << 20;
const CONFIGURATION_BYTES: u64 = 1 << 20;

/// Whether `bytes` are the pinned revision's file `name` (its length and
/// SHA-256 in `pins`, `checkpoint-files.json`).
fn pinned(pins: &serde_json::Value, name: &str, bytes: &[u8]) -> bool {
    pins[name]["byte_length"].as_u64() == Some(bytes.len() as u64)
        && pins[name]["sha256"] == runtime_weights::sha256_bytes(bytes).as_str()
}

/// The frontend, `tokenizer.json` and `tokenizer_config.json` the package
/// of the checkpoint at `root` carries. The pinned revision's tokenizer
/// files keep the declared frontend. Other files are served when their
/// configuration and chat template state the pinned revision's
/// ([`TokenizerSemantics`]), under a frontend that names their content
/// (tokenizer repository `local` at a content revision) and builds from
/// them as a serving host builds it.
fn tokenizer(root: &Path, pins: &str) -> Result<[Vec<u8>; 3]> {
    use model_compiler_sdk::preparation::tokenizer::{self, TokenizerSemantics};
    let pins: serde_json::Value = serde_json::from_str(pins)?;
    let read = |name: &str, limit| crate::checkpoint_headers::document(root, name, limit);
    let definition = read("tokenizer.json", DEFINITION_BYTES)?;
    let configuration = read("tokenizer_config.json", CONFIGURATION_BYTES)?;
    if pinned(&pins, "tokenizer.json", &definition)
        && pinned(&pins, "tokenizer_config.json", &configuration)
    {
        return Ok([
            crate::frontend::COMPONENT.to_vec(),
            definition,
            configuration,
        ]);
    }
    let mut files = BTreeMap::from([
        ("tokenizer.json".to_owned(), definition),
        ("tokenizer_config.json".to_owned(), configuration),
    ]);
    if !std::fs::symlink_metadata(root.join("chat_template.jinja"))
        .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
    {
        files.insert(
            "chat_template.jinja".into(),
            read("chat_template.jinja", CONFIGURATION_BYTES)?,
        );
    }
    let config: serde_json::Value = serde_json::from_slice(&files["tokenizer_config.json"])
        .map_err(|e| format!("tokenizer_config.json: {e}"))?;
    TokenizerSemantics::parse(TOKENIZER_SEMANTICS)?
        .check(&config, files.get("chat_template.jinja").map(Vec::as_slice))?;
    let revision = tokenizer::content_revision(&files)?;
    let definition = files.remove("tokenizer.json").expect("read above");
    let configuration = files.remove("tokenizer_config.json").expect("read above");
    let mut frontend: serde_json::Value = serde_json::from_slice(crate::frontend::COMPONENT)?;
    tokenizer::rewrite_tokenizer(
        &mut frontend,
        &definition,
        &configuration,
        "local",
        &revision,
    )?;
    let component = serde_json::to_vec(&frontend)?;
    let descriptor =
        runtime_frontend_contract::FrontendDescriptorV1::from_component_bytes(&component)?;
    runtime_frontend_contract::tokenizer_from_components_v1(
        &descriptor,
        &definition,
        &configuration,
    )?;
    Ok([component, definition, configuration])
}

/// Emits into `output` the package of the admitted checkpoint at `root`
/// with the native module at `module`, and the serving profile it carries:
/// the checkpoint's tokenizer files and the frontend that names them
/// ([`tokenizer`]), the precision component of the
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
    let layout = admitted.layout;
    let profile = select_profile(layout)?;
    let precision =
        model_glm5next_contract::prepared::checkpoint_component(layout, &admitted.config)?;
    let [frontend, definition, configuration] = tokenizer(root, pins)?;
    let mut components = vec![
        package_draft::frontend(StagedContentV1::bytes(frontend))?,
        package_draft::tokenizer_definition(StagedContentV1::bytes(definition))?,
        package_draft::tokenizer_configuration(StagedContentV1::bytes(configuration))?,
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
    /// The profiles the emission serves: each that asserts no layout, or
    /// an admitted one.
    fn profiles(&self) -> Vec<String> {
        let file: serde_json::Value =
            serde_json::from_str(PROFILES).expect("the serving profiles parse");
        file["profiles"]
            .as_object()
            .map(|profiles| {
                profiles
                    .iter()
                    .filter(|(_, profile)| {
                        profile["model_options"]
                            .get(LAYOUT_OPTION)
                            .is_none_or(|layout| Layout::named(layout).is_ok())
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
        select_profile(admit(root, request)?.layout)
    }
}

#[cfg(test)]
#[path = "test_native.rs"]
mod test_native;

#[cfg(test)]
mod tests {
    use super::*;
    use model_glm5next_contract::catalog::DType;
    use model_glm5next_contract::quantization::{
        FP8_BLOCK128, FP8_NVFP4_ROUTED, NVFP4_DENSE_ROUTED, NVFP4_ROUTED, representative_layouts,
        test_support::{config_of, extra_checkpoints},
    };

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

    /// The real metadata of the checkpoint at `source`: its config and
    /// tokenizer files, its headers in sparse files of their logical
    /// lengths, and the installed native module.
    fn real_checkpoint(source: &Path) -> Checkpoint {
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
                "the sparse checkpoint fixture retains the original logical length",
            );
        }
        std::fs::copy(test_native::module(), &checkpoint.module).unwrap();
        checkpoint
    }

    /// Always runs a bounded case in the ordinary library suite. The
    /// configured official checkpoint (`CHOPINFER_GLM5NEXT_CHECKPOINT`, for
    /// its layout) adds real metadata and pinned tokenizer coverage. An
    /// explicit native-module override is verified as supplied, never
    /// skipped.
    fn emission_checkpoints(layout: &Layout, config: &[u8]) -> Vec<Checkpoint> {
        let mut checkpoints = vec![Checkpoint::bounded(layout, config)];
        if *layout == FP8_BLOCK128
            && let Some(source) = std::env::var_os("CHOPINFER_GLM5NEXT_CHECKPOINT")
        {
            checkpoints.push(real_checkpoint(Path::new(&source)));
        }
        eprintln!(
            "{}: 1 bounded case, {} configured case(s)",
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

    /// A checkpoint holding `config` and catalog headers in the layout of
    /// its classes (BF16 convolutions, F32 mHC vectors), with no tokenizer,
    /// and the emission request with `policy` for it.
    fn headers_only(config: &[u8], policy: Option<&str>) -> Checkpoint {
        let classes = admission::admit_bytes(config).unwrap().quantization.classes;
        let layout =
            model_glm5next_contract::quantization::find(classes, DType::Bf16, DType::F32).unwrap();
        let headers = model_glm5next_contract::direct::test_support::checkpoint_request(
            layout,
            config,
            0,
            |_| {},
        );
        Checkpoint::new(config, policy, None, &[]).with_request(headers)
    }

    /// A direct emission request with `policy` for a checkpoint holding
    /// `config` and its headers: the emission's error, after checking the
    /// profile request admits the same way (and selects a profile when
    /// admitted).
    fn emit_error(config: &[u8], policy: Option<&str>) -> String {
        let checkpoint = headers_only(config, policy);
        let error = checkpoint.emit().err().unwrap().to_string();
        let profile = Glm5NextFamily.direct_profile(&checkpoint.root, &checkpoint.request);
        if let Err(profile) = profile {
            assert_eq!(profile.to_string(), error);
        }
        error
    }

    fn direct_profile(config: &[u8]) -> String {
        let checkpoint = headers_only(config, None);
        Glm5NextFamily
            .direct_profile(&checkpoint.root, &checkpoint.request)
            .unwrap()
            .name
    }

    /// Every layout selects the MTP profile, and every shipped profile
    /// serves every layout: none asserts one.
    #[test]
    fn every_layout_selects_the_mtp_profile() {
        for layout in representative_layouts() {
            assert_eq!(direct_profile(&config_of(layout)), "tp4-mtp");
        }
        for layout in representative_layouts() {
            let selected = select_profile(layout).unwrap();
            assert_eq!(selected.name, "tp4-mtp");
            assert_eq!(selected.profile["speculation"]["drafter"], "mtp");
            assert_eq!(selected.profile["speculation"]["tokens"], 3);
            assert!(selected.profile.get("context_parallel_size").is_none());
            assert!(selected.profile.get("model_options").is_none());
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
                "tp4-mtp-k5"
            ]
        );
        for name in names {
            assert!(
                file["profiles"][name.as_str()]
                    .get("model_options")
                    .is_none()
            );
            for layout in representative_layouts() {
                assert_eq!(serving_profile(layout, name).unwrap().name, *name);
            }
        }
        assert_eq!(
            serving_profile(&FP8_BLOCK128, "tp2-mtp")
                .unwrap_err()
                .to_string(),
            "glm5next declares no serving profile tp2-mtp"
        );
    }

    #[test]
    fn the_activation_policy_is_admitted_against_the_quantization_layout() {
        let nvfp4 = &config_of(&NVFP4_ROUTED)[..];
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
    /// bounded checkpoint and the configured official one serve images by
    /// default.
    #[test]
    fn the_admitted_checkpoints_serve_images_by_default() {
        for layout in representative_layouts() {
            for checkpoint in emission_checkpoints(layout, &config_of(layout)) {
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

    /// Both bounded and configured block-FP8 metadata emit the declared
    /// frontend, precision and media components. Text-only emission omits
    /// the encoder; a tokenizer configuration whose chat template is not the
    /// pinned revision's, and an oversized one, fail distinctly.
    #[test]
    fn the_fp8_checkpoint_emits_its_package() {
        for mut checkpoint in emission_checkpoints(
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
            // The pinned revision's tokenizer files keep the declared frontend.
            assert_eq!(
                std::fs::read(out.join("components/frontend.json")).unwrap(),
                crate::frontend::COMPONENT
            );
            // A tokenizer configuration with another chat template.
            std::fs::remove_dir_all(&out).unwrap();
            let path = checkpoint.root.join("tokenizer_config.json");
            std::fs::write(&path, br#"{"chat_template": "{{ edited }}"}"#).unwrap();
            let error = checkpoint.emit().err().unwrap().to_string();
            assert!(
                error.contains("unsupported chat template semantics"),
                "{error}"
            );
            assert!(!out.exists());
            std::fs::write(&path, vec![b' '; (CONFIGURATION_BYTES + 1) as usize]).unwrap();
            let error = checkpoint.emit().err().unwrap().to_string();
            assert!(
                error.contains("tokenizer_config.json: checkpoint metadata exceeds limit"),
                "{error}"
            );
            assert!(!out.exists());
        }
    }

    /// The pinned revision's tokenizer, its configuration serialized anew
    /// with another provenance, states the same meaning: its package carries
    /// those files under a frontend that names their content, which builds
    /// the declared tokenizer. Needs `CHOPINFER_GLM5NEXT_CHECKPOINT`.
    #[test]
    fn a_tokenizer_of_the_same_meaning_is_served_by_its_content() {
        let Some(source) = std::env::var_os("CHOPINFER_GLM5NEXT_CHECKPOINT") else {
            eprintln!("CHOPINFER_GLM5NEXT_CHECKPOINT unset: skipped");
            return;
        };
        let source = Path::new(&source);
        let mut checkpoint = real_checkpoint(source);
        checkpoint.request.modalities = selection(&[Modality::Text]);
        let path = checkpoint.root.join("tokenizer_config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["name_or_path"] = "local-fine-tune".into();
        config["is_local"] = false.into();
        let configuration = serde_json::to_vec_pretty(&config).unwrap();
        std::fs::write(&path, &configuration).unwrap();
        // The chat template is the pinned revision's file.
        let error = checkpoint.emit().err().unwrap().to_string();
        assert!(error.contains("no chat template"), "{error}");
        std::fs::copy(
            source.join("chat_template.jinja"),
            checkpoint.root.join("chat_template.jinja"),
        )
        .unwrap();
        checkpoint.emit().unwrap();
        let out = checkpoint.root.join("out");
        let frontend: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join("components/frontend.json")).unwrap())
                .unwrap();
        let declared: serde_json::Value =
            serde_json::from_slice(crate::frontend::COMPONENT).unwrap();
        let tokenizer = &frontend["tokenizer"];
        assert_eq!(tokenizer["repository"], "local");
        assert_ne!(tokenizer["revision"], declared["tokenizer"]["revision"]);
        assert_eq!(tokenizer["revision"].as_str().unwrap().len(), 64);
        assert_eq!(
            tokenizer["configuration"]["sha256"],
            runtime_weights::sha256_bytes(&configuration)
        );
        assert_eq!(tokenizer["definition"], declared["tokenizer"]["definition"]);
        assert_eq!(
            tokenizer["merge_count"],
            declared["tokenizer"]["merge_count"]
        );
        assert_eq!(frontend["models"], declared["models"]);
        assert_eq!(
            std::fs::read(out.join("components/tokenizer_config.json")).unwrap(),
            configuration
        );
        // Another chat template is refused.
        std::fs::remove_dir_all(&out).unwrap();
        std::fs::write(checkpoint.root.join("chat_template.jinja"), b"{{ edited }}").unwrap();
        let error = checkpoint.emit().err().unwrap().to_string();
        assert!(
            error.contains("unsupported chat template semantics"),
            "{error}"
        );
    }

    /// The text package of `checkpoint`, admitted in `layout`: the MTP
    /// profile, its own precision document, and a manifest that declares
    /// its layout.
    fn assert_text_package(checkpoint: &Checkpoint, layout: &Layout) {
        {
            {
                let emitted = checkpoint.emit().unwrap();
                assert_eq!(emitted.profile.name, "tp4-mtp");
                let out = checkpoint.root.join("out");
                let precision = std::fs::read(out.join(crate::package::PRECISION_PATH)).unwrap();
                model_glm5next_contract::prepared::decode_for(&precision, layout).unwrap();
                // The component holds the lowered projections only, which a
                // mix whose latent attention, shared experts and dense MLPs
                // are block FP8 lowers as the block-FP8 layout does.
                let fp8 = model_glm5next_contract::quantization::Encoding::Fp8Block128;
                assert_eq!(
                    model_glm5next_contract::prepared::decode(&precision).is_ok(),
                    layout.projections == fp8 && layout.dense == fp8,
                    "{}",
                    layout.id
                );
                let manifest: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(out.join("manifest.json")).unwrap())
                        .unwrap();
                let option = manifest["model_options"]["options"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|option| option["name"] == LAYOUT_OPTION)
                    .unwrap();
                assert_eq!(
                    option["definition"]["allowed_values"],
                    serde_json::json!([layout.id])
                );
            }
        }
    }

    /// Each mixed representative layout emits its text package from a
    /// bounded checkpoint of its synthetic config.
    #[test]
    fn each_mixed_checkpoint_emits_its_text_package() {
        for layout in [&NVFP4_ROUTED, &NVFP4_DENSE_ROUTED, &FP8_NVFP4_ROUTED] {
            for checkpoint in emission_checkpoints(layout, &config_of(layout)) {
                assert_text_package(&checkpoint, layout);
            }
        }
    }

    /// Each checkpoint of `CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS`, of any
    /// layout, admits in the layout its config and headers resolve to and
    /// emits its text package; skipped when none is named.
    #[test]
    fn extra_checkpoints_emit_their_packages() {
        let extra = extra_checkpoints();
        if extra.is_empty() {
            eprintln!("CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS unset: skipped");
        }
        for source in extra {
            let mut checkpoint = real_checkpoint(&source);
            let layout = admit(&checkpoint.root, &checkpoint.request).unwrap().layout;
            eprintln!("{}: layout {}", source.display(), layout.id);
            checkpoint.request.modalities = selection(&[Modality::Text]);
            assert_text_package(&checkpoint, layout);
        }
    }

    /// The shipped profiles are the helper's.
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
                "tp4-mtp-k5"
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
        // tower either, so its modalities resolve to text; the profile
        // request and the emission refuse it, since its headers fix its
        // layout and it has none.
        let checkpoint = Checkpoint::new(fp8, None, None, &[]);
        assert_eq!(
            Glm5NextFamily
                .direct_modalities(&checkpoint.root, &checkpoint.request)
                .unwrap(),
            Some(text.clone())
        );
        let error = Glm5NextFamily
            .direct_profile(&checkpoint.root, &checkpoint.request)
            .unwrap_err()
            .to_string();
        assert!(error.contains("model.safetensors.index.json"), "{error}");
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
