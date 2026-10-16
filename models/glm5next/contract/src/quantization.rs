//! The checkpoint encodings glm5next admits. A layout fixes the stored
//! encoding of four projection classes: latent attention with the shared
//! experts, the dense MLPs, the routed experts of the main layers, and the
//! routed experts of the MTP layers. Every other projection is stored in
//! BF16. It also fixes the dtypes of two side tensor families no
//! quantization config declares (the KDA convolutions and the mHC base and
//! scale), since exporters store them differently.
//!
//! An HF block-FP8 export (`quant_method: fp8`) is the base layout. A
//! ModelOpt export is read through the shared importer: each catalog
//! projection's recipe must agree within its class, and the classes must
//! form an admitted layout. Every projection runs BF16 activations (A16), so
//! the `checkpoint` activation policy refuses an export that declares
//! quantized activations, and `auto` and `bf16` run its weights under BF16
//! activations (W4A16 for NVFP4).
//!
//! The shared importer reads HF's single-algorithm ModelOpt `config.json`
//! (NVIDIA's export), whose config groups target the `Linear` class: every
//! role glm5next imports is marked a `Linear`. It records the export's
//! static FP8 KV cache hint (`kv_cache_scheme`) and imports nothing from it,
//! since the latent KV format is the serving `--kv-cache-dtype`; a recorded
//! hint is accepted for the `nvfp4-dense-routed` layout only.
use crate::catalog::{DType, Group, Placement};
use crate::geometry::Geometry;
use kernel_core::contract::WeightFormat;
use model_compiler_sdk::quantization::ActivationPolicy;
use model_compiler_sdk::quantization::import::{
    ActivationRecipe, ExportMetadata, ImportOptions, KvHintPolicy, LINEAR_CLASS_ROLE,
    ROUTED_EXPERT_ROLE,
};
use serde_json::Value;

/// A stored projection encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// FP8 E4M3 with an F32 `weight_scale_inv` per 128 x 128 block.
    Fp8Block128,
    Bf16,
    /// E2M1 pairs in U8, an E4M3 `weight_scale` per 16 K values and an F32
    /// `weight_scale_2`. The export's static `input_scale` is never read.
    Nvfp4,
    /// FP8 E4M3 with an E8M0 `weight_scale` (U8) per 32 K values.
    Mxfp8,
}

impl Encoding {
    fn imported(format: WeightFormat) -> Option<Self> {
        Some(match format {
            WeightFormat::Fp8Block128x128(_) => Self::Fp8Block128,
            WeightFormat::Bf16 => Self::Bf16,
            WeightFormat::Nvfp4 => Self::Nvfp4,
            WeightFormat::Mxfp8 => Self::Mxfp8,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub id: &'static str,
    /// Latent attention and shared expert projections.
    pub projections: Encoding,
    /// The dense MLPs of the first layers.
    pub dense: Encoding,
    /// Routed experts of the main layers.
    pub routed: Encoding,
    /// Routed experts of the MTP layers.
    pub mtp_routed: Encoding,
    /// The KDA q, k and v convolution weights.
    pub kda_conv: DType,
    /// The mHC `hc_*_base` and `hc_*_scale` vectors.
    pub hc_vectors: DType,
}

/// zai-org's block-FP8 export.
pub const FP8_BLOCK128: Layout = Layout {
    id: "fp8-block128",
    projections: Encoding::Fp8Block128,
    dense: Encoding::Fp8Block128,
    routed: Encoding::Fp8Block128,
    mtp_routed: Encoding::Fp8Block128,
    kda_conv: DType::Bf16,
    hc_vectors: DType::F32,
};

/// NVFP4 routed experts, MXFP8 MTP experts and BF16 elsewhere
/// (local-inference-lab's ModelOpt export).
pub const NVFP4_ROUTED: Layout = Layout {
    id: "nvfp4-routed",
    projections: Encoding::Bf16,
    dense: Encoding::Bf16,
    routed: Encoding::Nvfp4,
    mtp_routed: Encoding::Mxfp8,
    kda_conv: DType::Bf16,
    hc_vectors: DType::F32,
};

/// NVFP4 dense MLPs and routed experts, BF16 MTP experts and BF16
/// elsewhere, F32 KDA convolutions and BF16 mHC vectors (NVIDIA's ModelOpt
/// export).
pub const NVFP4_DENSE_ROUTED: Layout = Layout {
    id: "nvfp4-dense-routed",
    projections: Encoding::Bf16,
    dense: Encoding::Nvfp4,
    routed: Encoding::Nvfp4,
    mtp_routed: Encoding::Bf16,
    kda_conv: DType::F32,
    hc_vectors: DType::Bf16,
};

pub const LAYOUTS: &[Layout] = &[FP8_BLOCK128, NVFP4_ROUTED, NVFP4_DENSE_ROUTED];

/// The `chop.model-options.v1` option naming the checkpoint's admitted
/// layout. A package declares it with that layout's id as the one allowed
/// value, so a serving profile that sets it is refused against a package of
/// the other layout, and create reads it.
pub const LAYOUT_OPTION: &str = "checkpoint.quantization_layout";

impl Layout {
    /// The [`LAYOUT_OPTION`] a package of this layout declares.
    pub fn model_option(&self) -> runtime_bundle::model_options::ModelOptionV1 {
        runtime_bundle::model_options::ModelOptionV1 {
            name: LAYOUT_OPTION.into(),
            definition: runtime_bundle::model_options::OptionDefinitionV1::String {
                default: self.id.into(),
                allowed_values: vec![self.id.into()],
            },
            default_is_qualified: true,
        }
    }

    /// The admitted layout a [`LAYOUT_OPTION`] value names.
    pub fn named(value: &Value) -> Result<&'static Self, String> {
        LAYOUTS
            .iter()
            .find(|layout| value.as_str() == Some(layout.id))
            .ok_or_else(|| format!("unsupported {LAYOUT_OPTION} {value}"))
    }
}

/// A checkpoint's admitted layout and the activation recipe it declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Quantization {
    pub layout: &'static Layout,
    /// The first quantized activation recipe the export declares and the
    /// module that declares it.
    pub declared_activation: Option<(String, ActivationRecipe)>,
}

impl Quantization {
    /// Resolve `--activation-policy`: `auto` and `bf16` run BF16
    /// activations; `checkpoint` refuses a declared quantized activation
    /// recipe, which no glm5next engine implements.
    pub fn check_policy(&self, policy: ActivationPolicy) -> Result<(), String> {
        match (policy, &self.declared_activation) {
            (ActivationPolicy::Checkpoint, Some((module, recipe))) => Err(format!(
                "unsupported activation policy checkpoint: {module} declares {recipe:?} \
                 activations and glm5next runs BF16 activations only; use auto or bf16"
            )),
            _ => Ok(()),
        }
    }
}

/// The projection class of a base-catalog projection weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Projections,
    Dense,
    Routed,
    MtpRouted,
    Unquantized,
}

impl Class {
    const fn name(self) -> &'static str {
        match self {
            Self::Projections => "latent attention and shared expert projections",
            Self::Dense => "dense MLP projections",
            Self::Routed => "routed experts",
            Self::MtpRouted => "MTP routed experts",
            Self::Unquantized => "unquantized tensors",
        }
    }
    fn of(self, layout: &Layout) -> Encoding {
        match self {
            Self::Projections => layout.projections,
            Self::Dense => layout.dense,
            Self::Routed => layout.routed,
            Self::MtpRouted => layout.mtp_routed,
            Self::Unquantized => Encoding::Bf16,
        }
    }
}

/// Read the layout of `config`'s `quantization_config`, naming the module
/// whose recipe no admitted layout stores.
pub fn read(g: &Geometry, config: &Value) -> Result<Quantization, String> {
    let quantization = config.get("quantization_config");
    let metadata = export_metadata(quantization)?;
    let method = quantization.and_then(|q| q.get("quant_method"));
    let read = match method.and_then(Value::as_str) {
        Some("fp8") => Ok(Quantization {
            layout: &FP8_BLOCK128,
            declared_activation: metadata
                .default
                .and_then(|recipe| recipe.activation)
                .filter(|&activation| activation != ActivationRecipe::Bf16)
                .map(|activation| ("quantization_config".into(), activation)),
        }),
        Some("modelopt") => imported(g, &metadata),
        _ => Err(format!(
            "unsupported quantization_config.quant_method: expected \"fp8\" or \"modelopt\", received {}",
            method.map_or_else(|| "absent".into(), Value::to_string)
        )),
    }?;
    // The KV hint is ignored for NVIDIA's export only.
    if metadata.kv_hint.is_some() && *read.layout != NVFP4_DENSE_ROUTED {
        return Err(format!(
            "unsupported quantization_config.kv_cache_scheme: glm5next ignores a KV cache hint \
             only for the {} layout",
            NVFP4_DENSE_ROUTED.id
        ));
    }
    Ok(read)
}

/// The shared importer's reading of `quantization_config`. A KV cache hint
/// is recorded, not imported: [`read`] admits it for one layout.
pub fn export_metadata(quantization: Option<&Value>) -> Result<ExportMetadata, String> {
    let options = ImportOptions {
        kv_hint: KvHintPolicy::RecordAndIgnore,
    };
    ExportMetadata::from_documents_with(quantization, None, &options)
        .map_err(|error| format!("unsupported quantization_config: {error}"))
}

/// Every base-catalog projection's recipe through the shared importer. A
/// tensor the program binds as a side tensor (norms, the latent engine's
/// `kv_b_proj`) must not be converted: an explicit rule or exclusion that
/// quantizes it is refused, while a default or a `Linear` class group does
/// not reach it. Its stored dtype is the layout's,
/// and the headers are checked against it.
fn imported(g: &Geometry, metadata: &ExportMetadata) -> Result<Quantization, String> {
    let mut found: [Option<Encoding>; 5] = [None; 5];
    let mut declared_activation = None;
    for spec in crate::catalog::catalog(g) {
        let Some(stem) = spec.name.strip_suffix(".weight") else {
            continue;
        };
        if matches!(
            crate::views::binding(g, &spec.name),
            Some(crate::views::Binding::Side(_))
        ) {
            metadata
                .require_unconverted(&spec.name)
                .map_err(|error| format!("unsupported quantization_config: {error}"))?;
            continue;
        }
        let class = match (spec.dtype, spec.placement, spec.group) {
            (DType::F8E4M3, Placement::Expert { .. }, Group::Mtp) => Class::MtpRouted,
            (DType::F8E4M3, Placement::Expert { .. }, _) => Class::Routed,
            (DType::F8E4M3, ..) if dense(stem) => Class::Dense,
            (DType::F8E4M3, ..) => Class::Projections,
            _ => Class::Unquantized,
        };
        // Every catalog weight outside the side tensors is marked a
        // `Linear`, so a config group targeting the class reaches what a
        // top-level default reaches; exclusions still win, and a weight no
        // layout stores quantized is refused below.
        let mut targets = vec![stem.to_owned(), LINEAR_CLASS_ROLE.into()];
        if matches!(class, Class::Routed | Class::MtpRouted) {
            targets.push(ROUTED_EXPERT_ROLE.into());
        }
        let recipe = metadata
            .recipe(&targets, None)
            .map_err(|error| format!("unsupported quantization_config: {error}"))?;
        let encoding = Encoding::imported(recipe.weight)
            .filter(|&encoding| LAYOUTS.iter().any(|layout| class.of(layout) == encoding))
            .ok_or_else(|| {
                let mut admitted = Vec::new();
                for encoding in LAYOUTS.iter().map(|l| class.of(l)) {
                    if !admitted.contains(&encoding) {
                        admitted.push(encoding);
                    }
                }
                format!(
                    "unsupported quantization_config: {stem} is {:?}; glm5next stores {} as one of {admitted:?}",
                    recipe.weight,
                    class.name()
                )
            })?;
        let slot = &mut found[class as usize];
        if slot.is_some_and(|first| first != encoding) {
            return Err(format!(
                "unsupported quantization_config: {stem} is {encoding:?}, other {} are {:?}",
                class.name(),
                slot.unwrap()
            ));
        }
        *slot = Some(encoding);
        if declared_activation.is_none()
            && let Some(activation) = recipe.activation
            && activation != ActivationRecipe::Bf16
        {
            declared_activation = Some((stem.to_owned(), activation));
        }
    }
    let [projections, dense, routed, mtp_routed, _] = found.map(|e| e.unwrap_or(Encoding::Bf16));
    let classes = (projections, dense, routed, mtp_routed);
    let layout = LAYOUTS
        .iter()
        .find(|l| (l.projections, l.dense, l.routed, l.mtp_routed) == classes)
        .ok_or_else(|| {
            format!(
                "unsupported quantization_config layout: projections {projections:?}, dense \
                 {dense:?}, routed experts {routed:?}, MTP routed experts {mtp_routed:?}; glm5next \
                 admits {:?}",
                LAYOUTS.iter().map(|l| l.id).collect::<Vec<_>>()
            )
        })?;
    Ok(Quantization {
        layout,
        declared_activation,
    })
}

/// Whether `stem` is a dense MLP projection (not a routed or shared expert).
pub(crate) fn dense(stem: &str) -> bool {
    ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .any(|part| stem.ends_with(&format!(".mlp.{part}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::{admit, admit_bytes};

    const NVFP4_CONFIG: &[u8] = include_bytes!("../fixtures/nvfp4-config.json");

    fn nvfp4_config() -> Value {
        serde_json::from_slice(NVFP4_CONFIG).unwrap()
    }

    /// The export's config with `quantized_layers` (and the matching
    /// config group targets) changed by `change`.
    fn relayered(change: impl FnOnce(&mut serde_json::Map<String, Value>)) -> Value {
        let mut config = nvfp4_config();
        let q = &mut config["quantization_config"];
        q.as_object_mut().unwrap().remove("config_groups");
        change(q["quantized_layers"].as_object_mut().unwrap());
        config
    }

    const LAYERS: &str = "model.language_model.layers";

    #[test]
    fn the_nvfp4_export_selects_flash_and_its_layout() {
        let matched = admit_bytes(NVFP4_CONFIG).unwrap();
        assert_eq!(matched.geometry.id, "flash");
        assert_eq!(matched.quantization.layout, &NVFP4_ROUTED);
        assert_eq!(
            matched.quantization.declared_activation,
            Some((
                format!("{LAYERS}.3.mlp.experts.0.gate_proj"),
                ActivationRecipe::Nvfp4Static
            ))
        );
    }

    const NVIDIA_CONFIG: &[u8] = include_bytes!("../fixtures/nvidia-nvfp4-config.json");

    /// NVIDIA's export: a default NVFP4 recipe for every `Linear` with
    /// exclusions, its KV hint ignored.
    #[test]
    fn the_nvidia_export_selects_nvfp4_dense_and_routed() {
        let matched = admit_bytes(NVIDIA_CONFIG).unwrap();
        assert_eq!(matched.geometry.id, "flash");
        assert_eq!(matched.quantization.layout, &NVFP4_DENSE_ROUTED);
        assert_eq!(
            matched.quantization.declared_activation,
            Some((
                format!("{LAYERS}.0.mlp.gate_proj"),
                ActivationRecipe::Nvfp4Static
            ))
        );
        let quantization = matched.quantization;
        quantization.check_policy(ActivationPolicy::Auto).unwrap();
        quantization.check_policy(ActivationPolicy::Bf16).unwrap();
        assert!(
            quantization
                .check_policy(ActivationPolicy::Checkpoint)
                .is_err()
        );
        // Only the KV hint is dropped: an unknown field still refuses.
        let mut config: Value = serde_json::from_slice(NVIDIA_CONFIG).unwrap();
        config["quantization_config"]["unknown_field"] = 1.into();
        let error = admit(&config).unwrap_err();
        assert!(
            error.starts_with("unsupported quantization_config: ")
                && error.contains("unknown_field"),
            "{error}"
        );
        // Without its exclusions the default reaches the latent attention.
        let mut config: Value = serde_json::from_slice(NVIDIA_CONFIG).unwrap();
        config["quantization_config"]["ignore"]
            .as_array_mut()
            .unwrap()
            .retain(|pattern| pattern != &format!("{LAYERS}.3.self_attn*"));
        assert!(
            admit(&config).unwrap_err().starts_with(&format!(
                "unsupported quantization_config: {LAYERS}.3.self_attn.q_a_proj is Nvfp4;"
            )),
            "{}",
            admit(&config).unwrap_err()
        );
    }

    /// The `Linear` group's activation recipe is the export's: a top-level
    /// algorithm that disagrees with it is refused, so `checkpoint` cannot
    /// lose the group's static NVFP4 inputs.
    #[test]
    fn a_linear_group_must_agree_with_the_top_level_algorithm() {
        let mut config: Value = serde_json::from_slice(NVIDIA_CONFIG).unwrap();
        config["quantization_config"]["quant_algo"] = "W4A16_NVFP4".into();
        assert_eq!(
            admit(&config).unwrap_err(),
            "unsupported quantization_config: quantization metadata \
             config.json/quantization_config/config_groups/group_0: declares EncodingRecipe { \
             weight: Nvfp4, activation: Some(Nvfp4Static) }, which disagrees with quant_algo \
             W4A16_NVFP4"
        );
        let mut config: Value = serde_json::from_slice(NVIDIA_CONFIG).unwrap();
        config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["dynamic"] =
            true.into();
        assert!(admit(&config).unwrap_err().starts_with(
            "unsupported quantization_config: quantization metadata \
                     config.json/quantization_config/config_groups/group_0:"
        ));
    }

    /// A `Linear` class group beside a path-targeted group is not the
    /// normalized form; it is refused rather than imported, where its
    /// static NVFP4 inputs would be lost and `checkpoint` would pass.
    #[test]
    fn a_class_group_outside_the_all_linear_form_is_refused() {
        let mut config: Value = serde_json::from_slice(NVIDIA_CONFIG).unwrap();
        let q = &mut config["quantization_config"];
        q.as_object_mut().unwrap().remove("kv_cache_scheme");
        q["group_size"] = 16.into();
        q["quant_algo"] = "W4A16_NVFP4".into();
        let weights = q["config_groups"]["group_0"]["weights"].clone();
        q["config_groups"]["group_1"] = serde_json::json!({
            "weights": weights,
            "targets": [format!("{LAYERS}.0.mlp.gate_proj")],
        });
        assert_eq!(
            admit(&config).unwrap_err(),
            "unsupported quantization_config: quantization metadata \
             config.json/quantization_config/config_groups/group_0: a group targeting a module \
             class cannot be combined with groups or layers targeting module paths"
        );
        // Any module class, in a mixed-precision document too.
        let mut config = nvfp4_config();
        config["quantization_config"]["config_groups"]["group_nvfp4_routed_experts"]["targets"] =
            serde_json::json!(["Embedding"]);
        assert!(admit(&config).unwrap_err().starts_with(
            "unsupported quantization_config: quantization metadata \
             config.json/quantization_config/config_groups/group_nvfp4_routed_experts:"
        ));
    }

    /// A rule that converts a tensor the program reads as a side tensor is
    /// refused, whether a norm or the latent engine's `kv_b_proj`.
    #[test]
    fn explicit_rules_on_side_tensors_are_refused() {
        for side in ["3.input_layernorm", "3.self_attn.kv_b_proj"] {
            let config = relayered(|layers| {
                layers.insert(
                    format!("{LAYERS}.{side}"),
                    serde_json::json!({"quant_algo": "NVFP4", "group_size": 16}),
                );
            });
            let error = admit(&config).unwrap_err();
            assert!(
                error.starts_with("unsupported quantization_config: ") && error.contains(side),
                "{error}"
            );
        }
    }

    /// A KV cache hint is ignored for NVIDIA's layout only, and only the
    /// static FP8 one it carries.
    #[test]
    fn kv_cache_hints_are_refused_outside_the_nvidia_layout() {
        let hints = [
            serde_json::json!({"dynamic": false, "num_bits": 8, "type": "float"}),
            serde_json::json!({"dynamic": true, "num_bits": 4, "type": "int"}),
            false.into(),
        ];
        let base: Value =
            serde_json::from_slice(include_bytes!("../fixtures/config.json")).unwrap();
        for config in [base, nvfp4_config()] {
            for hint in &hints {
                let mut config = config.clone();
                config["quantization_config"]["kv_cache_scheme"] = hint.clone();
                let error = admit(&config).unwrap_err();
                assert!(error.contains("kv_cache_scheme"), "{hint}: {error}");
            }
        }
        for hint in &hints[1..] {
            let mut config: Value = serde_json::from_slice(NVIDIA_CONFIG).unwrap();
            config["quantization_config"]["kv_cache_scheme"] = hint.clone();
            assert!(
                admit(&config).unwrap_err().starts_with(
                    "unsupported quantization_config: quantization metadata \
                         config.json/quantization_config/kv_cache_scheme: unsupported KV cache \
                         hint "
                ),
                "{hint}"
            );
        }
    }

    #[test]
    fn the_base_export_is_block_fp8() {
        let matched = admit_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        assert_eq!(matched.quantization.layout, &FP8_BLOCK128);
    }

    #[test]
    fn auto_and_bf16_run_bf16_activations_and_checkpoint_refuses() {
        let quantization = admit_bytes(NVFP4_CONFIG).unwrap().quantization;
        quantization.check_policy(ActivationPolicy::Auto).unwrap();
        quantization.check_policy(ActivationPolicy::Bf16).unwrap();
        assert_eq!(
            quantization
                .check_policy(ActivationPolicy::Checkpoint)
                .unwrap_err(),
            format!(
                "unsupported activation policy checkpoint: {LAYERS}.3.mlp.experts.0.gate_proj \
                 declares Nvfp4Static activations and glm5next runs BF16 activations only; \
                 use auto or bf16"
            )
        );
        // A weight-only export (W4A16_NVFP4) declares no quantized
        // activation and admits all three.
        let weight_only = Quantization {
            layout: &NVFP4_ROUTED,
            declared_activation: None,
        };
        for policy in [
            ActivationPolicy::Auto,
            ActivationPolicy::Bf16,
            ActivationPolicy::Checkpoint,
        ] {
            weight_only.check_policy(policy).unwrap();
        }
    }

    #[test]
    fn unknown_layouts_are_refused_by_module() {
        // MXFP8 main-layer experts.
        let config = relayered(|layers| {
            layers[&format!("{LAYERS}.3.mlp.experts")] =
                serde_json::json!({"quant_algo": "MXFP8", "group_size": 32});
        });
        assert_eq!(
            admit(&config).unwrap_err(),
            format!(
                "unsupported quantization_config: {LAYERS}.3.mlp.experts.0.gate_proj is Mxfp8; \
                 glm5next stores routed experts as one of [Fp8Block128, Nvfp4]"
            )
        );
        // A partly quantized expert bank.
        let config = relayered(|layers| {
            layers.remove(&format!("{LAYERS}.44.mlp.experts"));
        });
        assert!(admit(&config).unwrap_err().starts_with(&format!(
            "unsupported quantization_config: {LAYERS}.44.mlp.experts.0.gate_proj is Bf16;"
        )));
        // A quantized dense MLP or MTP expert format no layout stores.
        let config = relayered(|layers| {
            layers.insert(
                format!("{LAYERS}.0.mlp"),
                serde_json::json!({"quant_algo": "MXFP8", "group_size": 32}),
            );
        });
        assert!(admit(&config).unwrap_err().starts_with(&format!(
            "unsupported quantization_config: {LAYERS}.0.mlp.gate_proj is Mxfp8;"
        )));
        let config = relayered(|layers| {
            layers[&format!("{LAYERS}.45.mlp.experts")] =
                serde_json::json!({"quant_algo": "NVFP4", "group_size": 16});
        });
        assert!(admit(&config).unwrap_err().starts_with(&format!(
            "unsupported quantization_config: {LAYERS}.45.mlp.experts.0.gate_proj is Nvfp4;"
        )));
        // A quantized norm or router.
        let config = relayered(|layers| {
            layers.insert(
                format!("{LAYERS}.10.mlp.gate"),
                serde_json::json!({"quant_algo": "NVFP4", "group_size": 16}),
            );
        });
        assert!(
            admit(&config)
                .unwrap_err()
                .contains(&format!("{LAYERS}.10.mlp.gate"))
        );
        // Classes that are each admitted but form no layout.
        let config = relayered(|layers| {
            layers[&format!("{LAYERS}.45.mlp.experts")] =
                serde_json::json!({"quant_algo": "FP8_BLOCK_SCALES", "group_size": 128});
        });
        assert_eq!(
            admit(&config).unwrap_err(),
            "unsupported quantization_config layout: projections Bf16, dense Bf16, routed experts \
             Nvfp4, MTP routed experts Fp8Block128; glm5next admits [\"fp8-block128\", \
             \"nvfp4-routed\", \"nvfp4-dense-routed\"]"
        );
        let mut config = nvfp4_config();
        config["quantization_config"]["quant_method"] = "awq".into();
        assert_eq!(
            admit(&config).unwrap_err(),
            "unsupported quantization_config.quant_method: expected one of \
             [\"fp8\",\"modelopt\"], received \"awq\""
        );
    }
}
