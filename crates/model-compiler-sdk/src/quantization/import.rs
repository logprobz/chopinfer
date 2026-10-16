//! Supported export interpretation. Tensor roles and logical geometry come from
//! the family; companion naming, encodings and activation recipes do not.

use crate::CompilerError;
use kernel_core::contract::{ActivationFormat, DType, ScaleMode, WeightFormat};
use kernel_core::precision::{
    ActivationSpec, Calibration, Clipping, CodeEncoding, DynamicScale, GroupAxis, NibbleOrder,
    QuantizerSpec, Rounding, ScaleConvention, ScaleScope, ScaleSpec, TensorSpan, WeightLayout,
    WeightSpec,
};
use serde::{
    Deserialize, Serialize,
    de::{self, MapAccess, Visitor},
};
use std::collections::BTreeMap;
use std::fmt;

pub const IMPORTER_VERSION: &str = "hf-modelopt-precision-v2";
pub const MAX_EXPORT_METADATA_BYTES: usize = 16 * 1024 * 1024;
/// The optional ModelOpt metadata document beside `config.json`.
pub const HF_QUANT_CONFIG: &str = "hf_quant_config.json";
/// A family adds this to an operand's `targets` to mark a routed expert
/// projection. It names a role, not a module path: export rules and
/// exclusions never match it. Only an export's expert storage declaration
/// (`ExportMetadata::expert_storage`) reads it.
pub const ROUTED_EXPERT_ROLE: &str = "@routed-expert";
/// A family adds this to an operand's `targets` to mark a projection that is
/// an instance of the `Linear` module class. It names a role, not a module
/// path: only a config group that targets the class (ModelOpt's `targets:
/// ["Linear"]`) reaches it, and exclusions never match it.
pub const LINEAR_CLASS_ROLE: &str = "@class:Linear";
const CLASS_ROLE_PREFIX: &str = "@class:";
/// The module classes a config group may target.
const TARGET_CLASSES: &[&str] = &["Linear"];

/// How an import treats an export's KV cache quantization hint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum KvHintPolicy {
    /// Refuse any KV cache hint: a non-null `kv_cache_quant_algo` or any
    /// `kv_cache_scheme`.
    #[default]
    Refuse,
    /// Record a supported hint in [`ExportMetadata::kv_hint`] and import
    /// nothing from it: the family selects KV cache precision as a serving
    /// policy. Every other hint form is still refused.
    RecordAndIgnore,
}

/// Family choices for one import. The default reads exactly what
/// [`ExportMetadata::from_documents`] reads.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImportOptions {
    pub kv_hint: KvHintPolicy,
}

/// A KV cache quantization hint an export carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum KvHint {
    /// Static FP8 KV: `kv_cache_scheme` `{"dynamic": false, "num_bits": 8,
    /// "type": "float"}` or `kv_cache_quant_algo: "FP8"`.
    Fp8Static,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ActivationRecipe {
    Bf16,
    Fp8Dynamic,
    Mxfp8Dynamic,
    Nvfp4Static,
    Nvfp4Dynamic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EncodingRecipe {
    pub weight: WeightFormat,
    /// An alias describing only weights cannot invent an activation recipe.
    pub activation: Option<ActivationRecipe>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExportRule {
    pub target: String,
    pub recipe: EncodingRecipe,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct ExportMetadata {
    pub rules: Vec<ExportRule>,
    pub exclusions: Vec<String>,
    /// HF's additional conversion targets include non-linear roles such as
    /// tables. This is not a whitelist for otherwise quantized projections.
    #[serde(default)]
    pub additional_targets: Vec<String>,
    pub default: Option<EncodingRecipe>,
    /// The weight storage an HF FP8 export declares (`store_dtype`) for its
    /// converted routed expert projections in place of the default's block
    /// FP8 weights. The default's activation recipe still applies to them.
    #[serde(default)]
    pub expert_storage: Option<WeightFormat>,
    /// Config groups that target a module class rather than module paths,
    /// keyed by class. A class rule reaches only roles that carry the
    /// class's marker ([`LINEAR_CLASS_ROLE`]), and an exclusion wins over
    /// it, as ModelOpt's `ignore` does.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub class_rules: Vec<ExportRule>,
    /// The KV cache hint an import under [`KvHintPolicy::RecordAndIgnore`]
    /// recorded. Nothing else reads it into a recipe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_hint: Option<KvHint>,
    /// Digest the original bytes, including relevant metadata presence. Producer
    /// provenance is recorded but never executed or used as an admission list.
    pub documents: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Producer {
    name: String,
    version: String,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Document {
    producer: Option<Producer>,
    quantization: Option<Box<Document>>,
    quant_method: Option<String>,
    quant_algo: Option<String>,
    group_size: Option<u32>,
    /// Any JSON value, so a value of the wrong type is refused naming the
    /// field; `null` is no hint.
    kv_cache_quant_algo: Option<serde_json::Value>,
    /// `config.json`'s KV cache hint. Any value, `null` included, is a hint.
    #[serde(default, deserialize_with = "present")]
    kv_cache_scheme: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "unique_map")]
    config_groups: BTreeMap<String, Group>,
    #[serde(default, deserialize_with = "unique_map")]
    quantized_layers: BTreeMap<String, Layer>,
    #[serde(default)]
    ignore: Vec<String>,
    #[serde(default)]
    exclude_modules: Vec<String>,
    #[serde(default)]
    modules_to_not_convert: Vec<String>,
    #[serde(default)]
    modules_to_convert: Vec<String>,
    activation_scheme: Option<String>,
    weight_per_tensor: Option<bool>,
    act_per_tensor: Option<bool>,
    weight_block_size: Option<[u32; 2]>,
    /// The HF FP8 dialect's code format and its unconverted modules.
    fmt: Option<String>,
    #[serde(default)]
    ignored_layers: Vec<String>,
    /// The HF FP8 dialect's routed expert weight storage.
    store_dtype: Option<String>,
    mxfp4_block_size: Option<u32>,
}

fn present<'de, D>(deserializer: D) -> Result<Option<serde_json::Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde_json::Value::deserialize(deserializer).map(Some)
}

fn unique_map<'de, D, T>(deserializer: D) -> Result<BTreeMap<String, T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Unique<T>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>> Visitor<'de> for Unique<T> {
        type Value = BTreeMap<String, T>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("an object with unique target keys")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut values = BTreeMap::new();
            while let Some((key, value)) = map.next_entry::<String, T>()? {
                if values.insert(key.clone(), value).is_some() {
                    return Err(de::Error::custom(format!("duplicate target {key}")));
                }
            }
            Ok(values)
        }
    }
    deserializer.deserialize_map(Unique(std::marker::PhantomData))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Group {
    weights: Quantizer,
    input_activations: Option<Quantizer>,
    targets: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Quantizer {
    dynamic: bool,
    num_bits: u32,
    #[serde(rename = "type")]
    kind: String,
    group_size: Option<u32>,
    weight_block_size: Option<[u32; 2]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Layer {
    quant_algo: String,
    group_size: Option<u32>,
}

fn error(context: &str, reason: impl fmt::Display) -> CompilerError {
    CompilerError::validation(format!("quantization metadata {context}: {reason}"))
}

impl EncodingRecipe {
    pub fn bf16() -> Self {
        Self {
            weight: WeightFormat::Bf16,
            activation: Some(ActivationRecipe::Bf16),
        }
    }

    fn merge(self, other: Self, context: &str) -> Result<Self, CompilerError> {
        if self.weight != other.weight
            || matches!((self.activation, other.activation), (Some(a), Some(b)) if a != b)
        {
            return Err(error(
                context,
                format!("conflicting precision assignments {self:?} and {other:?}"),
            ));
        }
        Ok(Self {
            weight: self.weight,
            activation: self.activation.or(other.activation),
        })
    }
}

fn algorithm(layer: &Layer, context: &str) -> Result<EncodingRecipe, CompilerError> {
    let (recipe, group) = named_algorithm(&layer.quant_algo, context)?;
    if layer.group_size != group {
        return Err(error(context, "group_size disagrees with encoding"));
    }
    Ok(recipe)
}

/// A ModelOpt algorithm's recipe and the group size it requires.
fn named_algorithm(
    name: &str,
    context: &str,
) -> Result<(EncodingRecipe, Option<u32>), CompilerError> {
    let (weight, activation, group) = match name {
        "NVFP4" => (
            WeightFormat::Nvfp4,
            Some(ActivationRecipe::Nvfp4Static),
            Some(16),
        ),
        "W4A16_NVFP4" => (WeightFormat::Nvfp4, Some(ActivationRecipe::Bf16), Some(16)),
        "MXFP8" => (
            WeightFormat::Mxfp8,
            Some(ActivationRecipe::Mxfp8Dynamic),
            Some(32),
        ),
        // These aliases specify weight packing. The explicit config group owns
        // activations; notably FP8_PB_WO exports can declare dynamic FP8 inputs.
        "FP8_PB_WO" | "FP8_BLOCK_SCALES" => (
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            None,
            Some(128),
        ),
        "FP8" => (WeightFormat::Fp8PerTensor, None, None),
        other => return Err(error(context, format!("unsupported algorithm {other}"))),
    };
    Ok((EncodingRecipe { weight, activation }, group))
}

/// A bare module class (`Linear`, `Embedding`) rather than a module path.
fn is_class(target: &str) -> bool {
    !target.contains(['.', '*']) && target.starts_with(|c: char| c.is_ascii_uppercase())
}

fn group_recipe(group: &Group, context: &str) -> Result<EncodingRecipe, CompilerError> {
    let q = &group.weights;
    if q.dynamic || q.kind != "float" {
        return Err(error(
            context,
            "weights require a static floating-point encoding",
        ));
    }
    let weight = match (q.num_bits, q.group_size) {
        (4, Some(16)) => WeightFormat::Nvfp4,
        (8, Some(32)) => WeightFormat::Mxfp8,
        (8, Some(128)) => WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        (8, None) => WeightFormat::Fp8PerTensor,
        _ => return Err(error(context, "unsupported weight num_bits/group_size")),
    };
    if let Some(block) = q.weight_block_size {
        let expected = match weight {
            WeightFormat::Mxfp8 => [1, 32],
            WeightFormat::Nvfp4 => [1, 16],
            WeightFormat::Fp8Block128x128(_) => [128, 128],
            _ => [0, 0],
        };
        if block != expected {
            return Err(error(context, "weight_block_size conflicts with encoding"));
        }
    }
    let activation = match &group.input_activations {
        None => ActivationRecipe::Bf16,
        Some(a) if a.kind != "float" || a.weight_block_size.is_some() => {
            return Err(error(context, "unsupported activation encoding fields"));
        }
        Some(a) => match (a.num_bits, a.group_size, a.dynamic) {
            (4, Some(16), false) => ActivationRecipe::Nvfp4Static,
            (4, Some(16), true) => ActivationRecipe::Nvfp4Dynamic,
            (8, Some(32), true) => ActivationRecipe::Mxfp8Dynamic,
            (8, Some(128), true) => ActivationRecipe::Fp8Dynamic,
            _ => {
                return Err(error(
                    context,
                    "unsupported activation num_bits/group_size/dynamic contract",
                ));
            }
        },
    };
    Ok(EncodingRecipe {
        weight,
        activation: Some(activation),
    })
}

impl ExportMetadata {
    /// A checkpoint's export metadata: `config.json`'s `quantization_config`
    /// (absent: every weight BF16) and the optional `hf_quant_config.json`.
    pub fn from_documents(
        quantization_config: Option<&serde_json::Value>,
        hf_quant_config: Option<&[u8]>,
    ) -> Result<Self, CompilerError> {
        Self::from_documents_with(
            quantization_config,
            hf_quant_config,
            &ImportOptions::default(),
        )
    }

    /// [`Self::from_documents`] under the family's `options`.
    pub fn from_documents_with(
        quantization_config: Option<&serde_json::Value>,
        hf_quant_config: Option<&[u8]>,
        options: &ImportOptions,
    ) -> Result<Self, CompilerError> {
        let mut metadata = Self::default();
        if let Some(config) = quantization_config {
            let bytes = serde_json::to_vec(config)
                .map_err(|e| error("config.json/quantization_config", e))?;
            metadata.add_document_with("config.json/quantization_config", &bytes, options)?;
        }
        if let Some(bytes) = hf_quant_config {
            metadata.add_document_with(HF_QUANT_CONFIG, bytes, options)?;
        }
        Ok(metadata)
    }

    /// Requires that no module rule converts the tensor `name`: a norm, bias,
    /// sink or state tensor keeps its own dtype.
    pub fn require_unconverted(&self, name: &str) -> Result<(), CompilerError> {
        if let Some(stem) = name.strip_suffix(".weight") {
            self.recipe(&[stem.into()], Some(EncodingRecipe::bf16()))?;
        }
        Ok(())
    }

    /// Add a quantization_config object or an hf_quant_config document. The
    /// caller retains source files through the existing preparation interfaces.
    pub fn add_document(&mut self, name: &str, bytes: &[u8]) -> Result<(), CompilerError> {
        self.add_document_with(name, bytes, &ImportOptions::default())
    }

    /// [`Self::add_document`] under the family's `options`.
    pub fn add_document_with(
        &mut self,
        name: &str,
        bytes: &[u8],
        options: &ImportOptions,
    ) -> Result<(), CompilerError> {
        if bytes.len() > MAX_EXPORT_METADATA_BYTES
            || name.is_empty()
            || self.documents.contains_key(name)
        {
            return Err(error(
                name,
                "duplicate document name or metadata size exceeds bound",
            ));
        }
        let document: Document = serde_json::from_slice(bytes).map_err(|e| error(name, e))?;
        // Parse transactionally: a refused document does not leave partial rules.
        let mut next = self.clone();
        next.add_parsed(name, document, options)?;
        next.documents
            .insert(name.into(), runtime_weights::sha256_bytes(bytes));
        *self = next;
        Ok(())
    }

    fn add_parsed(
        &mut self,
        context: &str,
        document: Document,
        options: &ImportOptions,
    ) -> Result<(), CompilerError> {
        if let Some(producer) = &document.producer
            && (producer.name != "modelopt" || producer.version.is_empty())
        {
            return Err(error(context, "unsupported producer schema"));
        }
        if let Some(nested) = document.quantization {
            if document.quant_method.is_some()
                || document.quant_algo.is_some()
                || !document.config_groups.is_empty()
                || !document.quantized_layers.is_empty()
                || document.group_size.is_some()
                || !document.ignore.is_empty()
                || !document.exclude_modules.is_empty()
                || !document.modules_to_not_convert.is_empty()
                || !document.modules_to_convert.is_empty()
                || document.activation_scheme.is_some()
                || document.weight_per_tensor.is_some()
                || document.act_per_tensor.is_some()
                || document.weight_block_size.is_some()
                || document.kv_cache_quant_algo.is_some()
                || document.kv_cache_scheme.is_some()
                || document.fmt.is_some()
                || !document.ignored_layers.is_empty()
                || document.store_dtype.is_some()
                || document.mxfp4_block_size.is_some()
            {
                return Err(error(
                    context,
                    "nested and flat execution metadata cannot be mixed",
                ));
            }
            return self.add_parsed(context, *nested, options);
        }
        // KV is a separate policy. A non-null declaration must be interpreted by
        // its own importer, rather than silently being treated as projection data.
        self.add_kv_hint(
            context,
            document.kv_cache_quant_algo.as_ref(),
            document.kv_cache_scheme.as_ref(),
            options,
        )?;
        if document.quant_method.as_deref() == Some("fp8") {
            // Per-tensor flags and the code format are optional in this
            // dialect; when present they must restate the block recipe.
            if document.activation_scheme.as_deref() != Some("dynamic")
                || document.weight_per_tensor == Some(true)
                || document.act_per_tensor == Some(true)
                || document.weight_block_size != Some([128, 128])
                || document.fmt.as_deref().is_some_and(|fmt| fmt != "e4m3")
                || document.quant_algo.is_some()
                || document.group_size.is_some()
                || !document.config_groups.is_empty()
                || !document.quantized_layers.is_empty()
            {
                return Err(error(
                    context,
                    "HF FP8 requires block-128 E4M3 weights and dynamic group-128 activations",
                ));
            }
            let storage = match (document.store_dtype.as_deref(), document.mxfp4_block_size) {
                (None, None) => None,
                (Some("mxfp4"), Some(32)) => Some(WeightFormat::Mxfp4),
                _ => {
                    return Err(error(
                        context,
                        "expert storage requires store_dtype mxfp4 with mxfp4_block_size 32",
                    ));
                }
            };
            self.set_default(
                EncodingRecipe {
                    weight: WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
                    activation: Some(ActivationRecipe::Fp8Dynamic),
                },
                context,
            )?;
            if let Some(storage) = storage {
                if self.expert_storage.is_some_and(|old| old != storage) {
                    return Err(error(context, "conflicting expert storage declarations"));
                }
                self.expert_storage = Some(storage);
            }
        } else {
            if document
                .quant_method
                .as_deref()
                .is_some_and(|m| m != "modelopt")
                || document.activation_scheme.is_some()
                || document.weight_per_tensor.is_some()
                || document.act_per_tensor.is_some()
                || document.weight_block_size.is_some()
                || !document.modules_to_convert.is_empty()
                || document.fmt.is_some()
                || !document.ignored_layers.is_empty()
                || document.store_dtype.is_some()
                || document.mxfp4_block_size.is_some()
            {
                return Err(error(context, "unsupported exporter fields"));
            }
            let class_form = class_form(&document, context)?;
            match document.quant_algo.as_deref() {
                Some("MIXED_PRECISION") => {
                    if document.config_groups.is_empty() && document.quantized_layers.is_empty() {
                        return Err(error(
                            context,
                            "mixed precision requires per-operation declarations",
                        ));
                    }
                    if document.group_size.is_some_and(|v| v != 16) {
                        return Err(error(
                            context,
                            "unsupported mixed-export group_size default",
                        ));
                    }
                }
                Some(algorithm_name) => {
                    let (_, group) = named_algorithm(algorithm_name, context)?;
                    // ModelOpt's config.json can state the group size only in
                    // its config groups.
                    let stated = document.group_size.is_some()
                        || group.is_none()
                        || document.config_groups.is_empty();
                    let group_size = if stated {
                        document.group_size
                    } else {
                        groups_size(&document.config_groups, algorithm_name, group, context)?
                    };
                    let recipe = algorithm(
                        &Layer {
                            quant_algo: algorithm_name.into(),
                            group_size,
                        },
                        context,
                    )?;
                    if !stated || class_form {
                        for (name, group) in &document.config_groups {
                            let group_context = format!("{context}/config_groups/{name}");
                            let declared = group_recipe(group, &group_context)?;
                            if recipe.merge(declared, &group_context).is_err() {
                                return Err(error(
                                    &group_context,
                                    format!(
                                        "declares {declared:?}, which disagrees with quant_algo \
                                         {algorithm_name}"
                                    ),
                                ));
                            }
                        }
                    }
                    // In the class form the top-level algorithm restates the
                    // class groups; it is not a default for every module.
                    if !class_form {
                        self.set_default(recipe, context)?;
                    }
                }
                None if !document.config_groups.is_empty()
                    || !document.quantized_layers.is_empty() => {}
                None => return Err(error(context, "missing export precision declaration")),
            }
        }
        self.exclusions.extend(document.ignore);
        self.exclusions.extend(document.exclude_modules);
        self.exclusions.extend(document.modules_to_not_convert);
        self.exclusions.extend(document.ignored_layers);
        self.additional_targets.extend(document.modules_to_convert);
        for (name, group) in document.config_groups {
            let recipe = group_recipe(&group, &format!("{context}/config_groups/{name}"))?;
            if group.targets.is_empty() {
                return Err(error(&name, "target list must not be empty"));
            }
            for target in group.targets {
                if is_class(&target) {
                    add_rule(&mut self.class_rules, target, recipe, context)?;
                } else {
                    self.add_rule(target, recipe, context)?;
                }
            }
        }
        for (target, layer) in document.quantized_layers {
            let recipe = algorithm(&layer, &format!("{context}/{target}"))?;
            self.add_rule(target, recipe, context)?;
        }
        if let Some(class) = self.class_rules.first()
            && (!self.rules.is_empty() || self.default.is_some())
        {
            return Err(error(
                context,
                format!(
                    "a group targeting module class {} cannot be combined with module path \
                     rules or a default algorithm",
                    class.target
                ),
            ));
        }
        if self.rules.len() > 65536
            || self.class_rules.len() > 65536
            || self.exclusions.len() > 65536
            || self.additional_targets.len() > 65536
        {
            return Err(error(context, "too many export targets"));
        }
        for pattern in self.exclusions.iter().chain(&self.additional_targets) {
            validate_pattern(pattern).map_err(|e| error(context, e))?;
        }
        self.exclusions.sort();
        self.exclusions.dedup();
        self.additional_targets.sort();
        self.additional_targets.dedup();
        Ok(())
    }

    fn set_default(&mut self, recipe: EncodingRecipe, context: &str) -> Result<(), CompilerError> {
        self.default = Some(match self.default {
            Some(old) => old.merge(recipe, context)?,
            None => recipe,
        });
        Ok(())
    }

    fn add_rule(
        &mut self,
        target: String,
        recipe: EncodingRecipe,
        context: &str,
    ) -> Result<(), CompilerError> {
        add_rule(&mut self.rules, target, recipe, context)
    }

    /// Record the KV cache hints of one document under `options`.
    fn add_kv_hint(
        &mut self,
        context: &str,
        algorithm: Option<&serde_json::Value>,
        scheme: Option<&serde_json::Value>,
        options: &ImportOptions,
    ) -> Result<(), CompilerError> {
        let algorithm = match algorithm {
            None => None,
            Some(serde_json::Value::String(algorithm)) => Some(algorithm.as_str()),
            Some(other) => {
                return Err(error(
                    &format!("{context}/kv_cache_quant_algo"),
                    format!("expected a string or null, received {other}"),
                ));
            }
        };
        if options.kv_hint == KvHintPolicy::Refuse {
            if algorithm.is_some() {
                return Err(error(
                    context,
                    "KV quantization metadata requires an independent supported KV policy",
                ));
            }
            if scheme.is_some() {
                return Err(error(
                    &format!("{context}/kv_cache_scheme"),
                    "KV quantization metadata requires an independent supported KV policy",
                ));
            }
            return Ok(());
        }
        let mut hints = Vec::new();
        match algorithm {
            None => {}
            Some("FP8") => hints.push(KvHint::Fp8Static),
            Some(other) => {
                return Err(error(
                    &format!("{context}/kv_cache_quant_algo"),
                    format!("unsupported KV cache hint {other}; only FP8 is recorded"),
                ));
            }
        }
        if let Some(scheme) = scheme {
            let fp8 = serde_json::json!({"dynamic": false, "num_bits": 8, "type": "float"});
            if *scheme != fp8 {
                return Err(error(
                    &format!("{context}/kv_cache_scheme"),
                    format!("unsupported KV cache hint {scheme}; only {fp8} is recorded"),
                ));
            }
            hints.push(KvHint::Fp8Static);
        }
        for hint in hints {
            if self.kv_hint.is_some_and(|old| old != hint) {
                return Err(error(context, "conflicting KV cache hints"));
            }
            self.kv_hint = Some(hint);
        }
        Ok(())
    }

    pub fn recipe(
        &self,
        targets: &[String],
        hint: Option<EncodingRecipe>,
    ) -> Result<EncodingRecipe, CompilerError> {
        let routed = targets.iter().any(|target| target == ROUTED_EXPERT_ROLE);
        let classes: Vec<&str> = targets
            .iter()
            .filter_map(|target| target.strip_prefix(CLASS_ROLE_PREFIX))
            .collect();
        let targets: Vec<&str> = targets
            .iter()
            .map(String::as_str)
            .filter(|target| {
                *target != ROUTED_EXPERT_ROLE && !target.starts_with(CLASS_ROLE_PREFIX)
            })
            .collect();
        if targets.is_empty() {
            return Err(error(
                "operand",
                "model must provide at least one source target",
            ));
        }
        let context = targets.join(", ");
        let mut found: Option<EncodingRecipe> = None;
        for rule in &self.rules {
            if targets
                .iter()
                .any(|target| target_matches(&rule.target, target))
            {
                found = Some(match found {
                    Some(old) => old.merge(rule.recipe, &context)?,
                    None => rule.recipe,
                });
            }
        }
        let mut class: Option<EncodingRecipe> = None;
        for rule in &self.class_rules {
            if classes.contains(&rule.target.as_str()) {
                class = Some(match class {
                    Some(old) => old.merge(rule.recipe, &context)?,
                    None => rule.recipe,
                });
            }
        }
        let excluded = self
            .exclusions
            .iter()
            .any(|pattern| targets.iter().any(|t| target_matches(pattern, t)));
        let additional = self
            .additional_targets
            .iter()
            .any(|pattern| targets.iter().any(|t| target_matches(pattern, t)));
        // An exclusion wins over a class rule, as ModelOpt's `ignore` does.
        if (found.is_some() || additional) && excluded {
            return Err(error(&context, "target is both quantized and excluded"));
        }
        // Expert storage replaces only the default's weights: an explicit
        // rule, exclusion or family declaration keeps its own recipe.
        let stored = self.default.map(|default| match self.expert_storage {
            Some(weight) if routed => EncodingRecipe { weight, ..default },
            _ => default,
        });
        let declared = found.or_else(|| {
            if excluded {
                Some(EncodingRecipe::bf16())
            } else {
                class.or(hint).or(stored)
            }
        });
        let recipe = declared.unwrap_or_else(EncodingRecipe::bf16);
        if additional && recipe.weight == WeightFormat::Bf16 {
            return Err(error(
                &context,
                "modules_to_convert requires a supported quantized encoding for this role",
            ));
        }
        if let Some(hint) = hint {
            recipe.merge(hint, &context)
        } else {
            Ok(recipe)
        }
    }
}

fn add_rule(
    rules: &mut Vec<ExportRule>,
    target: String,
    recipe: EncodingRecipe,
    context: &str,
) -> Result<(), CompilerError> {
    validate_pattern(&target).map_err(|e| error(context, e))?;
    if let Some(existing) = rules.iter_mut().find(|r| r.target == target) {
        existing.recipe = existing
            .recipe
            .merge(recipe, &format!("{context}/{target}"))?;
    } else {
        rules.push(ExportRule { target, recipe });
    }
    Ok(())
}

/// Whether `document`'s config groups target a module class (ModelOpt's
/// `targets: ["Linear"]`). Only the `Linear` class is admitted, and a class
/// group cannot sit beside a group or layer that targets module paths:
/// either would leave a declared recipe unapplied.
fn class_form(document: &Document, context: &str) -> Result<bool, CompilerError> {
    let mut class_group = None;
    let mut path_group = false;
    for (name, group) in &document.config_groups {
        let group_context = format!("{context}/config_groups/{name}");
        let classes: Vec<&String> = group.targets.iter().filter(|t| is_class(t)).collect();
        if let Some(class) = classes
            .iter()
            .find(|c| !TARGET_CLASSES.contains(&c.as_str()))
        {
            return Err(error(
                &group_context,
                format!("module class {class} is not a supported target; only Linear is"),
            ));
        }
        if classes.is_empty() {
            path_group = true;
        } else if classes.len() != group.targets.len() {
            return Err(error(
                &group_context,
                "a group cannot target both a module class and module paths",
            ));
        } else {
            class_group.get_or_insert(group_context);
        }
    }
    match class_group {
        Some(group_context) if path_group || !document.quantized_layers.is_empty() => Err(error(
            &group_context,
            "a group targeting a module class cannot be combined with groups or layers \
             targeting module paths",
        )),
        class_group => Ok(class_group.is_some()),
    }
}

/// The group size `algorithm` requires, which ModelOpt's config.json form
/// states only in its config groups: every group's weights must state it.
fn groups_size(
    groups: &BTreeMap<String, Group>,
    algorithm: &str,
    required: Option<u32>,
    context: &str,
) -> Result<Option<u32>, CompilerError> {
    for (name, group) in groups {
        let stated = group.weights.group_size;
        if stated != required {
            return Err(error(
                &format!("{context}/config_groups/{name}"),
                format!(
                    "weights group_size {stated:?} disagrees with quant_algo {algorithm}, which \
                     requires {required:?}"
                ),
            ));
        }
    }
    Ok(required)
}

fn validate_pattern(pattern: &str) -> Result<(), &'static str> {
    if pattern.is_empty()
        || pattern.len() > 1024
        || pattern.chars().any(char::is_control)
        || pattern.contains(['?', '[', ']', '\\'])
    {
        Err("target pattern supports literal module paths and '*' only")
    } else {
        Ok(())
    }
}

/// Literal module targets include their child modules. '*' is the only glob
/// operator, so an exporter cannot turn this matcher into arbitrary regex work.
fn target_matches(pattern: &str, target: &str) -> bool {
    if !pattern.contains('*') {
        return target == pattern
            || target
                .strip_prefix(pattern)
                .is_some_and(|s| s.starts_with('.'));
    }
    let p = pattern.as_bytes();
    let t = target.as_bytes();
    let (mut pi, mut ti, mut star, mut retry) = (0, 0, None, 0);
    while ti < t.len() {
        if pi < p.len() && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star = Some(pi);
            pi += 1;
            retry = ti;
        } else if let Some(s) = star {
            retry += 1;
            ti = retry;
            pi = s + 1;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

/// The family supplies source role aliases and table ownership, not scale math.
pub struct OperandRole {
    pub operation: String,
    pub weight: String,
    pub targets: Vec<String>,
    pub logical_shape: Vec<u64>,
    pub lookup: bool,
    /// A sharded table can share a table-global companion. Its stem is a source
    /// role supplied by the model, independent of the quantized encoding.
    pub shared_scale_stem: Option<String>,
    pub declared: Option<EncodingRecipe>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedOperand {
    pub weight: WeightSpec,
    pub activation: Option<ActivationSpec>,
    pub source_tensors: Vec<String>,
}

/// The checkpoint tensors the importer reads as a weight's companion
/// scales, by role.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompanionNames {
    /// The local scale's accepted names, in the importer's preference
    /// order; empty when the format has no local scale.
    pub local: Vec<String>,
    /// The global scale's name, when the format has one.
    pub global: Option<String>,
}

/// The companion names [`import_operand`] reads for `weight` stored as
/// `format`. `shared` names the module whose per-tensor scales the weight
/// reuses; it defaults to the weight's own module. A block FP8 weight
/// prepared onto 64-row strips keeps its source's scale names.
pub fn companion_names(
    format: WeightFormat,
    weight: &str,
    shared: Option<&str>,
) -> Result<CompanionNames, CompilerError> {
    let context = format!("tensor {weight}");
    let stem = weight
        .strip_suffix(".weight")
        .ok_or_else(|| error(&context, "supported export weights end in .weight"))?;
    let shared = shared.unwrap_or(stem);
    Ok(match format {
        WeightFormat::Bf16 => CompanionNames::default(),
        WeightFormat::Fp8PerTensor => CompanionNames {
            local: vec![format!("{shared}.weight_scale")],
            global: None,
        },
        WeightFormat::Fp8Block128x128(_) | WeightFormat::Fp8Rows64x128(_) => CompanionNames {
            local: vec![
                format!("{stem}.weight_scale_inv"),
                format!("{stem}.weight_scale"),
            ],
            global: None,
        },
        WeightFormat::Mxfp8 | WeightFormat::Mxfp4 => CompanionNames {
            local: vec![format!("{stem}.weight_scale")],
            global: None,
        },
        WeightFormat::Nvfp4 => CompanionNames {
            local: vec![format!("{stem}.weight_scale")],
            global: Some(format!("{shared}.weight_scale_2")),
        },
        _ => return Err(error(&context, "unsupported source weight layout")),
    })
}

pub fn import_operand(
    metadata: &ExportMetadata,
    role: &OperandRole,
    tensors: &BTreeMap<String, TensorSpan>,
) -> Result<ImportedOperand, CompilerError> {
    import_operand_with_origins(metadata, role, tensors, [0, 0])
}

/// Validate an independent matrix view at its declared scale-grid origins.
pub(crate) fn import_operand_with_origins(
    metadata: &ExportMetadata,
    role: &OperandRole,
    tensors: &BTreeMap<String, TensorSpan>,
    origins: [u64; 2],
) -> Result<ImportedOperand, CompilerError> {
    let context = format!("operation {}, tensor {}", role.operation, role.weight);
    // Lookup inputs are integer indices. Preserve the export's weight recipe,
    // but its projection activation recipe cannot apply to this operation.
    let declared = role.declared.map(|mut recipe| {
        if role.lookup {
            recipe.activation = None;
        }
        recipe
    });
    let recipe = metadata.recipe(&role.targets, declared)?;
    let stem = role
        .weight
        .strip_suffix(".weight")
        .ok_or_else(|| error(&context, "supported export weights end in .weight"))?;
    let read = |name: &str| {
        tensors
            .get(name)
            .cloned()
            .ok_or_else(|| error(&context, format!("missing companion tensor {name}")))
    };
    let codes = read(&role.weight)?;
    let rank = role.logical_shape.len();
    if !(2..=3).contains(&rank) {
        return Err(error(&context, "expected logical matrix or bank geometry"));
    }
    let scalar = |name: String, scope| -> Result<ScaleSpec, CompilerError> {
        Ok(ScaleSpec {
            tensor: read(&name)?,
            scope,
            convention: ScaleConvention::Multiply,
            groups: vec![],
        })
    };
    let names = companion_names(
        recipe.weight,
        &role.weight,
        role.shared_scale_stem.as_deref(),
    )
    .map_err(|_| error(&context, "unsupported source weight layout"))?;
    let (encoding, local_scale, global_scale) = match recipe.weight {
        WeightFormat::Bf16 => (CodeEncoding::Bf16LittleEndian, None, None),
        WeightFormat::Fp8PerTensor => (
            CodeEncoding::E4M3,
            Some(scalar(
                names.local[0].clone(),
                if role.lookup {
                    ScaleScope::Table
                } else {
                    ScaleScope::Tensor
                },
            )?),
            None,
        ),
        format => {
            let (br, bk) = match format {
                WeightFormat::Nvfp4 => (1, 16),
                WeightFormat::Mxfp8 | WeightFormat::Mxfp4 => (1, 32),
                WeightFormat::Fp8Block128x128(_) => (128, 128),
                _ => return Err(error(&context, "unsupported source weight layout")),
            };
            let present: Vec<_> = names
                .local
                .iter()
                .filter(|name| tensors.contains_key(*name))
                .collect();
            if present.len() > 1 {
                return Err(error(&context, "conflicting scale companions"));
            }
            let name = present
                .first()
                .copied()
                .unwrap_or(names.local.last().unwrap());
            let mut tensor = read(name)?;
            // ModelOpt serializes UE8M0 scale codes as UINT8 safetensors bytes.
            if matches!(format, WeightFormat::Mxfp8 | WeightFormat::Mxfp4)
                && tensor.dtype == DType::U8
            {
                tensor.dtype = DType::Ue8M0;
            }
            let local = ScaleSpec {
                tensor,
                scope: ScaleScope::Block,
                convention: ScaleConvention::Multiply,
                groups: vec![
                    GroupAxis {
                        axis: (rank - 2) as u8,
                        size: br,
                        origin: origins[0],
                    },
                    GroupAxis {
                        axis: (rank - 1) as u8,
                        size: bk,
                        origin: origins[1],
                    },
                ],
            };
            let global = if let Some(global) = names.global {
                let read_global = read(&global)?;
                Some(scalar(
                    global,
                    if role.lookup {
                        ScaleScope::Table
                    } else if rank == 3
                        && read_global.elements().map_err(|e| error(&context, e))? != 1
                    {
                        ScaleScope::Expert
                    } else {
                        ScaleScope::Projection
                    },
                )?)
            } else {
                None
            };
            let encoding = if matches!(format, WeightFormat::Nvfp4 | WeightFormat::Mxfp4) {
                CodeEncoding::E2M1(NibbleOrder::EvenLow)
            } else {
                CodeEncoding::E4M3
            };
            (encoding, Some(local), global)
        }
    };
    let weight = WeightSpec {
        codes,
        logical_shape: role.logical_shape.clone(),
        format: recipe.weight,
        encoding,
        layout: WeightLayout::row_major(),
        local_scale,
        global_scale,
        transforms: vec![],
    };
    weight.validate().map_err(|e| error(&context, e))?;
    let mut source_tensors = vec![role.weight.clone()];
    source_tensors.extend(
        weight
            .local_scale
            .iter()
            .chain(weight.global_scale.iter())
            .map(|s| s.tensor.tensor.clone()),
    );
    let activation = if role.lookup {
        None
    } else {
        let a = recipe.activation.ok_or_else(|| {
            error(
                &context,
                "weight-only algorithm alias has no declared activation recipe",
            )
        })?;
        if a == ActivationRecipe::Bf16 {
            Some(ActivationSpec::bf16())
        } else {
            let (format, compute, group, calibration) = match a {
                ActivationRecipe::Fp8Dynamic => (
                    ActivationFormat::Fp8Group128(ScaleMode::ExactF32),
                    DType::Fp8E4M3,
                    128,
                    Calibration::Dynamic(DynamicScale::GroupAmax(ScaleMode::ExactF32)),
                ),
                ActivationRecipe::Mxfp8Dynamic => (
                    ActivationFormat::Mxfp8,
                    DType::Fp8E4M3,
                    32,
                    Calibration::Dynamic(DynamicScale::MxPowerOfTwo),
                ),
                ActivationRecipe::Nvfp4Dynamic => (
                    ActivationFormat::Nvfp4,
                    DType::Fp4E2M1,
                    16,
                    Calibration::Dynamic(DynamicScale::Nvfp4TensorAmax),
                ),
                ActivationRecipe::Nvfp4Static => {
                    let scale = scalar(
                        format!("{stem}.input_scale"),
                        if rank == 3
                            && read(&format!("{stem}.input_scale"))?
                                .elements()
                                .map_err(|e| error(&context, e))?
                                != 1
                        {
                            ScaleScope::Expert
                        } else {
                            ScaleScope::Projection
                        },
                    )?;
                    let count = scale.tensor.elements().map_err(|e| error(&context, e))?;
                    if count != 1 && (rank != 3 || count != role.logical_shape[0]) {
                        return Err(error(
                            &context,
                            "input_scale cardinality must be scalar or one per expert",
                        ));
                    }
                    if scale.tensor.dtype != DType::F32 {
                        return Err(error(
                            &context,
                            "NVFP4 input_scale must be a supplied F32 global, not an amax",
                        ));
                    }
                    source_tensors.push(scale.tensor.tensor.clone());
                    (
                        ActivationFormat::Nvfp4,
                        DType::Fp4E2M1,
                        16,
                        Calibration::Static(scale),
                    )
                }
                ActivationRecipe::Bf16 => unreachable!(),
            };
            Some(ActivationSpec {
                format,
                // Projection boundaries carry ordinary BF16 buffers. The
                // declared quantizer explicitly produces the compute operand.
                storage: DType::Bf16,
                compute,
                transforms: vec![],
                quantizer: Some(QuantizerSpec {
                    calibration,
                    groups: vec![GroupAxis {
                        axis: 1,
                        size: group,
                        origin: 0,
                    }],
                    rounding: Rounding::NearestEven,
                    clipping: Clipping::FiniteSaturate,
                }),
            })
        }
    };
    if let Some(a) = &activation {
        a.validate(&role.weight, &[1, *role.logical_shape.last().unwrap()])
            .map_err(|e| error(&context, e))?;
    }
    // An unconsumed execution companion is not an invitation to silently discard
    // a calibration or input transform when the declared recipe is incomplete.
    for suffix in [
        "input_scale",
        "pre_quant_scale",
        "input_scale_inv",
        "weight_zero_point",
    ] {
        let name = format!("{stem}.{suffix}");
        if tensors.contains_key(&name) && !source_tensors.contains(&name) {
            return Err(error(
                &context,
                format!("uninterpreted execution companion {name}"),
            ));
        }
    }
    Ok(ImportedOperand {
        weight,
        activation,
        source_tensors,
    })
}
