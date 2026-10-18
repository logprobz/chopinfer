//! The checkpoint encodings glm5next admits. A layout fixes the stored
//! encoding of four projection classes: latent attention with the shared
//! experts, the dense MLPs, the routed experts of the main layers, and the
//! routed experts of the MTP layers. Every other projection is stored in
//! BF16. It also fixes the dtypes of two side tensor families no
//! quantization config declares (the KDA convolutions and the mHC base and
//! scale), since exporters store them differently.
//!
//! The admitted layouts are every combination of the encodings the
//! projection declaration (`projections.json`) supports for each class with
//! BF16 or F32 storage of each side tensor family, which the engine widens
//! to F32 at load ([`layouts`]). No layout is named after an exporter: a
//! checkpoint is admitted by what it stores. Its class encodings come from
//! `config.json` ([`read`]) and its side tensor dtypes from its headers
//! ([`resolve`]).
//!
//! An HF block-FP8 export (`quant_method: fp8`) stores every class in block
//! FP8. A ModelOpt export is read through the shared importer: each catalog
//! projection's recipe must agree within its class, and each class's
//! encoding must be one the declaration supports for it. Every projection
//! runs BF16 activations (A16), so the `checkpoint` activation policy
//! refuses an export that declares quantized activations, and `auto` and
//! `bf16` run its weights under BF16 activations (W4A16 for NVFP4).
//!
//! The shared importer reads HF's single-algorithm ModelOpt `config.json`,
//! whose config groups target the `Linear` class: every role glm5next
//! imports is marked a `Linear`. It records an export's static FP8 KV cache
//! hint (`kv_cache_scheme`) and imports nothing from it, since the latent KV
//! format is the serving `--kv-cache-dtype`.
use crate::catalog::{DType, Group, Placement};
use crate::geometry::Geometry;
use kernel_core::contract::WeightFormat;
use kernel_core::precision::format::FormatId;
use model_compiler_sdk::quantization::ActivationPolicy;
use model_compiler_sdk::quantization::import::{
    ActivationRecipe, ExportMetadata, ImportOptions, KvHintPolicy, LINEAR_CLASS_ROLE,
    ROUTED_EXPERT_ROLE,
};
use serde_json::Value;
use std::sync::LazyLock;

/// A stored projection encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

    /// The encoding a declaration support rule's format names.
    fn declared(format: FormatId) -> Option<Self> {
        Some(match format {
            FormatId::Fp8Block128 => Self::Fp8Block128,
            FormatId::Bf16 => Self::Bf16,
            FormatId::Nvfp4 => Self::Nvfp4,
            FormatId::Mxfp8 => Self::Mxfp8,
            _ => return None,
        })
    }

    /// Its name in a layout id (`fp8` is block-128 FP8, the only FP8
    /// encoding glm5next stores).
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Fp8Block128 => "fp8",
            Self::Bf16 => "bf16",
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp8 => "mxfp8",
        }
    }
}

/// The encodings of the four projection classes, which an export's
/// `quantization_config` fixes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Classes {
    /// Latent attention and shared expert projections.
    pub projections: Encoding,
    /// The dense MLPs of the first layers.
    pub dense: Encoding,
    /// Routed experts of the main layers.
    pub routed: Encoding,
    /// Routed experts of the MTP layers.
    pub mtp_routed: Encoding,
}

impl Classes {
    /// Every class in block FP8, as an HF block-FP8 export stores them.
    pub const BLOCK_FP8: Self = Self {
        projections: Encoding::Fp8Block128,
        dense: Encoding::Fp8Block128,
        routed: Encoding::Fp8Block128,
        mtp_routed: Encoding::Fp8Block128,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// `attn-<e>.dense-<e>.experts-<e>.mtp-<e>.conv-<d>.hc-<d>`, the
    /// encoding tags of the classes in field order, then the side tensor
    /// dtypes ([`Layout::derived_id`]).
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

/// Block FP8 in every class with BF16 convolutions and F32 mHC vectors:
/// the official block-FP8 export's layout.
pub const FP8_BLOCK128: Layout = Layout {
    id: "attn-fp8.dense-fp8.experts-fp8.mtp-fp8.conv-bf16.hc-f32",
    projections: Encoding::Fp8Block128,
    dense: Encoding::Fp8Block128,
    routed: Encoding::Fp8Block128,
    mtp_routed: Encoding::Fp8Block128,
    kda_conv: DType::Bf16,
    hc_vectors: DType::F32,
};

/// NVFP4 routed experts, MXFP8 MTP experts and BF16 elsewhere: a
/// representative mixed layout.
pub const NVFP4_ROUTED: Layout = Layout {
    id: "attn-bf16.dense-bf16.experts-nvfp4.mtp-mxfp8.conv-bf16.hc-f32",
    projections: Encoding::Bf16,
    dense: Encoding::Bf16,
    routed: Encoding::Nvfp4,
    mtp_routed: Encoding::Mxfp8,
    kda_conv: DType::Bf16,
    hc_vectors: DType::F32,
};

/// NVFP4 dense MLPs and routed experts, BF16 MTP experts and BF16
/// elsewhere, F32 KDA convolutions and BF16 mHC vectors: a representative
/// mixed layout.
pub const NVFP4_DENSE_ROUTED: Layout = Layout {
    id: "attn-bf16.dense-nvfp4.experts-nvfp4.mtp-bf16.conv-f32.hc-bf16",
    projections: Encoding::Bf16,
    dense: Encoding::Nvfp4,
    routed: Encoding::Nvfp4,
    mtp_routed: Encoding::Bf16,
    kda_conv: DType::F32,
    hc_vectors: DType::Bf16,
};

/// Block-FP8 latent attention, shared experts, dense MLPs and MTP experts
/// with NVFP4 routed experts: a representative mix of quantized classes.
pub const FP8_NVFP4_ROUTED: Layout = Layout {
    id: "attn-fp8.dense-fp8.experts-nvfp4.mtp-fp8.conv-bf16.hc-f32",
    projections: Encoding::Fp8Block128,
    dense: Encoding::Fp8Block128,
    routed: Encoding::Nvfp4,
    mtp_routed: Encoding::Fp8Block128,
    kda_conv: DType::Bf16,
    hc_vectors: DType::F32,
};

/// The dtypes a side tensor family is admitted in. The engine reads both
/// families in F32 and widens BF16 at load.
pub const SIDE_DTYPES: [DType; 2] = [DType::Bf16, DType::F32];

/// The declaration's use profiles of each class's projections.
const CLASS_PROFILES: [(Class, &[&str]); 4] = [
    (Class::Projections, &["fp8", "fp8-f32"]),
    (Class::Dense, &["dense-mlp", "dense-mlp-f32"]),
    (Class::Routed, &["experts"]),
    (Class::MtpRouted, &["mtp-experts"]),
];

/// The encodings `projections.json` supports for each class, in
/// [`CLASS_PROFILES`] order: those every use of every profile of the class
/// supports at TP4 (an enabled rule with no phase selector).
fn declared_formats() -> Result<[Vec<Encoding>; 4], String> {
    let ranks = crate::contract::Shape::TP4.ranks();
    let declaration = crate::projections::declaration(ranks).map_err(|e| e.to_string())?;
    let mut formats: [Vec<Encoding>; 4] = Default::default();
    for (slot, (class, profiles)) in formats.iter_mut().zip(CLASS_PROFILES) {
        let mut common: Option<Vec<Encoding>> = None;
        for profile in profiles {
            let uses = declaration
                .use_profiles
                .get(*profile)
                .ok_or_else(|| format!("projections.json declares no use profile {profile}"))?;
            for usage in uses {
                let contract = usage
                    .contract
                    .as_ref()
                    .ok_or_else(|| format!("a use of profile {profile} declares no contract"))?;
                let mut supported = Vec::new();
                for rule in &contract.support {
                    if !rule.enabled || !rule.phases.is_empty() || rule.when.is_some() {
                        continue;
                    }
                    if !rule.ranks.is_empty() && !rule.ranks.contains(&ranks) {
                        continue;
                    }
                    for &format in &rule.formats {
                        let encoding = Encoding::declared(format).ok_or_else(|| {
                            format!(
                                "use profile {profile} supports {format:?} for the {}, which no \
                                 glm5next encoding stores",
                                class.name()
                            )
                        })?;
                        if !supported.contains(&encoding) {
                            supported.push(encoding);
                        }
                    }
                }
                common = Some(match common {
                    None => supported,
                    Some(common) => common
                        .into_iter()
                        .filter(|encoding| supported.contains(encoding))
                        .collect(),
                });
            }
        }
        let common = common.unwrap_or_default();
        if common.is_empty() {
            return Err(format!(
                "projections.json supports no encoding for every use of the {}",
                class.name()
            ));
        }
        *slot = common;
    }
    Ok(formats)
}

/// The encodings each class is admitted in, in [`CLASS_PROFILES`] order.
fn class_formats() -> &'static [Vec<Encoding>; 4] {
    static FORMATS: LazyLock<[Vec<Encoding>; 4]> = LazyLock::new(|| {
        declared_formats().expect("the pinned projection declaration admits its class encodings")
    });
    &FORMATS
}

/// The encodings the declaration supports for `class`.
fn formats_of(class: Class) -> &'static [Encoding] {
    let index = CLASS_PROFILES
        .iter()
        .position(|(own, _)| *own == class)
        .expect("every quantized class has use profiles");
    &class_formats()[index]
}

/// Every admitted layout: each class in each encoding the declaration
/// supports for it, and each side tensor family in each of
/// [`SIDE_DTYPES`].
pub fn layouts() -> &'static [Layout] {
    static LAYOUTS: LazyLock<Vec<Layout>> = LazyLock::new(|| {
        let mut layouts = Vec::new();
        for &projections in formats_of(Class::Projections) {
            for &dense in formats_of(Class::Dense) {
                for &routed in formats_of(Class::Routed) {
                    for &mtp_routed in formats_of(Class::MtpRouted) {
                        for kda_conv in SIDE_DTYPES {
                            for hc_vectors in SIDE_DTYPES {
                                let classes = Classes {
                                    projections,
                                    dense,
                                    routed,
                                    mtp_routed,
                                };
                                layouts.push(Layout::of(classes, kda_conv, hc_vectors));
                            }
                        }
                    }
                }
            }
        }
        layouts
    });
    &LAYOUTS
}

/// Layouts that store every class in every encoding the declaration
/// supports for it, in declaration order: layout `i` takes each class's
/// `i`-th encoding, or its last for a class with fewer. Each takes BF16
/// KDA convolutions and F32 mHC vectors, except that the last of several
/// takes the other dtype of each. A lowered projection's format depends only
/// on its class, so these hold every (projection, format) pair of
/// [`layouts`] in as many layouts as the longest class has encodings, and
/// kernel unions are built over them rather than over every layout.
pub fn covering_layouts() -> &'static [&'static Layout] {
    static COVERING: LazyLock<Vec<&'static Layout>> = LazyLock::new(|| {
        let formats = class_formats();
        let count = formats.iter().map(Vec::len).max().unwrap_or(0);
        (0..count)
            .map(|i| {
                let pick = |class: usize| formats[class][i.min(formats[class].len() - 1)];
                let classes = Classes {
                    projections: pick(0),
                    dense: pick(1),
                    routed: pick(2),
                    mtp_routed: pick(3),
                };
                let (kda_conv, hc_vectors) = if count > 1 && i == count - 1 {
                    (DType::F32, DType::Bf16)
                } else {
                    (DType::Bf16, DType::F32)
                };
                find(classes, kda_conv, hc_vectors)
                    .expect("a covering layout is an admitted layout")
            })
            .collect()
    });
    &COVERING
}

/// The layouts full serving plans are checked in: the [`covering_layouts`]
/// and [`FP8_NVFP4_ROUTED`], which mixes block FP8 with NVFP4 routed
/// experts as none of them does.
pub fn representative_layouts() -> Vec<&'static Layout> {
    let mut layouts = covering_layouts().to_vec();
    layouts.push(
        Layout::named(&Value::from(FP8_NVFP4_ROUTED.id))
            .expect("the representative mix is an admitted layout"),
    );
    layouts
}

/// The admitted layout of `classes` with the side tensor dtypes
/// `kda_conv` and `hc_vectors`, or a refusal naming the class or family
/// whose encoding the declaration does not support.
pub fn find(
    classes: Classes,
    kda_conv: DType,
    hc_vectors: DType,
) -> Result<&'static Layout, String> {
    for (class, encoding) in [
        (Class::Projections, classes.projections),
        (Class::Dense, classes.dense),
        (Class::Routed, classes.routed),
        (Class::MtpRouted, classes.mtp_routed),
    ] {
        if !formats_of(class).contains(&encoding) {
            return Err(format!(
                "unsupported quantization_config layout: the {} are {encoding:?}; glm5next \
                 stores them as one of {:?}",
                class.name(),
                formats_of(class)
            ));
        }
    }
    for (family, dtype) in [
        ("KDA convolution weights", kda_conv),
        ("mHC base and scale vectors", hc_vectors),
    ] {
        if !SIDE_DTYPES.contains(&dtype) {
            return Err(format!(
                "unsupported checkpoint layout: the {family} are {}; glm5next reads them as \
                 one of {:?}",
                dtype.name(),
                SIDE_DTYPES.map(DType::name)
            ));
        }
    }
    let id = Layout::derived_id(classes, kda_conv, hc_vectors);
    layouts()
        .iter()
        .find(|layout| layout.id == id)
        .ok_or_else(|| format!("glm5next admits no layout {id}"))
}

/// The `chop.model-options.v1` option naming the checkpoint's admitted
/// layout. A package declares it with that layout's id as the one allowed
/// value, so a serving profile that sets it is refused against a package of
/// another layout, and create reads it.
pub const LAYOUT_OPTION: &str = "checkpoint.quantization_layout";

impl Layout {
    /// The layout of `classes` with the side tensor dtypes `kda_conv` and
    /// `hc_vectors`, its id derived ([`Layout::derived_id`]) and kept for the
    /// process.
    fn of(classes: Classes, kda_conv: DType, hc_vectors: DType) -> Self {
        let id = Self::derived_id(classes, kda_conv, hc_vectors);
        Self {
            id: String::leak(id),
            projections: classes.projections,
            dense: classes.dense,
            routed: classes.routed,
            mtp_routed: classes.mtp_routed,
            kda_conv,
            hc_vectors,
        }
    }

    /// The id of a layout: each class's encoding tag and each side tensor
    /// family's dtype, in a fixed order.
    pub fn derived_id(classes: Classes, kda_conv: DType, hc_vectors: DType) -> String {
        let dtype = |dtype: DType| dtype.name().to_ascii_lowercase();
        format!(
            "attn-{}.dense-{}.experts-{}.mtp-{}.conv-{}.hc-{}",
            classes.projections.tag(),
            classes.dense.tag(),
            classes.routed.tag(),
            classes.mtp_routed.tag(),
            dtype(kda_conv),
            dtype(hc_vectors),
        )
    }

    /// The encodings of its four projection classes.
    pub const fn classes(&self) -> Classes {
        Classes {
            projections: self.projections,
            dense: self.dense,
            routed: self.routed,
            mtp_routed: self.mtp_routed,
        }
    }

    /// Whether every class is stored in block FP8, whatever the side tensor
    /// dtypes: such an instance runs none of the other formats' kernels.
    pub fn block_fp8(&self) -> bool {
        self.classes() == Classes::BLOCK_FP8
    }

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
        layouts()
            .iter()
            .find(|layout| value.as_str() == Some(layout.id))
            .ok_or_else(|| format!("unsupported {LAYOUT_OPTION} {value}"))
    }
}

/// A checkpoint's class encodings and the activation recipe it declares,
/// from its `config.json`. [`resolve`] completes its layout from its
/// headers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Quantization {
    pub classes: Classes,
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

/// The side tensor family a checkpoint tensor belongs to, when its stored
/// dtype is the checkpoint's choice: the KDA q, k and v convolution weights,
/// or the mHC base and scale vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideFamily {
    KdaConv,
    HcVectors,
}

impl SideFamily {
    /// The family of tensor `name`, if any.
    pub fn of(name: &str) -> Option<Self> {
        let kda = ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"]
            .iter()
            .any(|suffix| name.ends_with(&format!("self_attn.{suffix}")));
        let hc = [
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_scale",
        ]
        .iter()
        .any(|suffix| name.ends_with(&format!(".{suffix}")));
        match (kda, hc) {
            (true, _) => Some(Self::KdaConv),
            (_, true) => Some(Self::HcVectors),
            _ => None,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::KdaConv => "KDA convolution weights",
            Self::HcVectors => "mHC base and scale vectors",
        }
    }
}

/// The admitted layout of a checkpoint of geometry `g` whose config gave
/// `quantization`: its class encodings, with the dtype its headers store
/// for each side tensor family. `dtype_of` returns a catalog tensor's header
/// dtype. Every tensor of a family must store the same admitted dtype.
pub fn resolve(
    g: &Geometry,
    quantization: &Quantization,
    mut dtype_of: impl FnMut(&str) -> Result<String, String>,
) -> Result<&'static Layout, String> {
    let mut found: [Option<(String, String)>; 2] = [None, None];
    for spec in crate::catalog::checkpoint_catalog(g, &FP8_BLOCK128) {
        let Some(family) = SideFamily::of(&spec.name) else {
            continue;
        };
        let dtype = dtype_of(&spec.name)?;
        let slot = &mut found[family as usize];
        match slot {
            Some((first, tensor)) if *first != dtype => {
                return Err(format!(
                    "{} is {dtype}, but {tensor} is {first}: the {} must share one dtype",
                    spec.name,
                    family.name()
                ));
            }
            Some(_) => {}
            None => *slot = Some((dtype, spec.name.clone())),
        }
    }
    let dtype = |family: SideFamily| -> Result<DType, String> {
        let (name, tensor) = found[family as usize]
            .as_ref()
            .ok_or_else(|| format!("the checkpoint stores no {}", family.name()))?;
        SIDE_DTYPES
            .into_iter()
            .find(|dtype| dtype.name() == name)
            .ok_or_else(|| {
                format!(
                    "unsupported checkpoint layout: {tensor} is {name}; glm5next reads the {} \
                     as one of {:?}",
                    family.name(),
                    SIDE_DTYPES.map(DType::name)
                )
            })
    };
    find(
        quantization.classes,
        dtype(SideFamily::KdaConv)?,
        dtype(SideFamily::HcVectors)?,
    )
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

    /// The encodings this class is admitted in.
    fn admitted(self) -> &'static [Encoding] {
        match self {
            Self::Unquantized => &[Encoding::Bf16],
            class => formats_of(class),
        }
    }
}

/// Read the class encodings of `config`'s `quantization_config`, naming the
/// module whose recipe the declaration does not support for its class.
pub fn read(g: &Geometry, config: &Value) -> Result<Quantization, String> {
    let quantization = config.get("quantization_config");
    let metadata = export_metadata(quantization)?;
    let method = quantization.and_then(|q| q.get("quant_method"));
    match method.and_then(Value::as_str) {
        Some("fp8") => Ok(Quantization {
            classes: Classes::BLOCK_FP8,
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
    }
}

/// The shared importer's reading of `quantization_config`. A static FP8 KV
/// cache hint is recorded, not imported: the latent KV format is the
/// serving `--kv-cache-dtype`.
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
        let Some(class) = class_of(g, &spec, stem) else {
            metadata
                .require_unconverted(&spec.name)
                .map_err(|error| format!("unsupported quantization_config: {error}"))?;
            continue;
        };
        // Every catalog weight outside the side tensors is marked a
        // `Linear`, so a config group targeting the class reaches what a
        // top-level default reaches; exclusions still win, and a weight
        // stored in an encoding its class does not admit is refused below.
        let mut targets = vec![stem.to_owned(), LINEAR_CLASS_ROLE.into()];
        if matches!(class, Class::Routed | Class::MtpRouted) {
            targets.push(ROUTED_EXPERT_ROLE.into());
        }
        let recipe = metadata
            .recipe(&targets, None)
            .map_err(|error| format!("unsupported quantization_config: {error}"))?;
        let encoding = Encoding::imported(recipe.weight)
            .filter(|encoding| class.admitted().contains(encoding))
            .ok_or_else(|| {
                format!(
                    "unsupported quantization_config: {stem} is {:?}; glm5next stores {} as one of {:?}",
                    recipe.weight,
                    class.name(),
                    class.admitted()
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
    Ok(Quantization {
        classes: Classes {
            projections,
            dense,
            routed,
            mtp_routed,
        },
        declared_activation,
    })
}

/// Whether `stem` is a dense MLP projection (not a routed or shared expert).
pub(crate) fn dense(stem: &str) -> bool {
    ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .any(|part| stem.ends_with(&format!(".mlp.{part}")))
}

/// The class of base-catalog weight `spec` of geometry `g` with weight stem
/// `stem`, or `None` for a side tensor the program binds unconverted.
fn class_of(g: &Geometry, spec: &crate::catalog::TensorSpec, stem: &str) -> Option<Class> {
    if matches!(
        crate::views::binding(g, &spec.name),
        Some(crate::views::Binding::Side(_))
    ) {
        return None;
    }
    Some(match (spec.dtype, spec.placement, spec.group) {
        (DType::F8E4M3, Placement::Expert { .. }, Group::Mtp) => Class::MtpRouted,
        (DType::F8E4M3, Placement::Expert { .. }, _) => Class::Routed,
        (DType::F8E4M3, ..) if dense(stem) => Class::Dense,
        (DType::F8E4M3, ..) => Class::Projections,
        _ => Class::Unquantized,
    })
}

/// The variable naming further checkpoints for tests that read real
/// headers, beside the official one in `CHOPINFER_GLM5NEXT_CHECKPOINT`:
/// directories separated by `:`, of any admitted layout. Each is checked
/// against what its own layout derives, not against pinned values of a
/// particular export.
pub const EXTRA_CHECKPOINTS: &str = "CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS";

/// Test helpers of other crates: the synthetic configs and the extra
/// checkpoints a run names.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    pub use super::synthetic::{config_of, linear_class_config, modelopt_config};

    /// The directories [`super::EXTRA_CHECKPOINTS`] names, none when it is
    /// unset.
    pub fn extra_checkpoints() -> Vec<std::path::PathBuf> {
        std::env::var_os(super::EXTRA_CHECKPOINTS)
            .map(|paths| {
                std::env::split_paths(&paths)
                    .filter(|path| !path.as_os_str().is_empty())
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Synthetic checkpoint configs: the official block-FP8 config with its
/// quantization block rewritten for any class mix, in either form of
/// ModelOpt's metadata the importer reads. Tests admit mixed layouts from
/// them, so no exporter's files are needed.
pub mod synthetic {
    use super::*;

    /// The official block-FP8 `config.json`.
    pub const BLOCK_FP8_CONFIG: &[u8] = include_bytes!("../fixtures/config.json");

    /// A config of `layout`'s class mix: the official config for block FP8,
    /// ModelOpt's single-algorithm form for the NVFP4 dense and routed mix,
    /// and its mixed-precision form for every other mix, so tests read both
    /// forms the importer admits.
    pub fn config_of(layout: &Layout) -> Vec<u8> {
        if layout.classes() == NVFP4_DENSE_ROUTED.classes() {
            linear_class_config(layout.classes())
        } else {
            modelopt_config(layout.classes())
        }
    }

    /// The `quantized_layers` rule and the config group name of `encoding`,
    /// `None` for BF16. Block FP8 has no rule: that algorithm name leaves
    /// the activations to its group.
    fn rule(encoding: Encoding) -> Option<(Option<Value>, &'static str)> {
        Some(match encoding {
            Encoding::Nvfp4 => (
                Some(serde_json::json!({"quant_algo": "NVFP4", "group_size": 16})),
                "group_nvfp4",
            ),
            Encoding::Mxfp8 => (
                Some(serde_json::json!({"quant_algo": "MXFP8", "group_size": 32})),
                "group_mxfp8",
            ),
            Encoding::Fp8Block128 => (None, "group_fp8_block128"),
            Encoding::Bf16 => return None,
        })
    }

    /// The config group of `encoding` over `targets`: its weights and the
    /// activations the official recipe of that format declares (static
    /// NVFP4 inputs, dynamic MXFP8, dynamic block-128 FP8).
    fn group(encoding: Encoding, targets: &[String]) -> Value {
        let float = |bits: u32, group: u32, dynamic: bool| {
            serde_json::json!({
                "dynamic": dynamic, "group_size": group, "num_bits": bits, "type": "float"
            })
        };
        let (weights, inputs) = match encoding {
            Encoding::Nvfp4 => (float(4, 16, false), float(4, 16, false)),
            Encoding::Mxfp8 => (float(8, 32, false), float(8, 32, true)),
            Encoding::Fp8Block128 => (float(8, 128, false), float(8, 128, true)),
            Encoding::Bf16 => unreachable!("BF16 modules are left unconverted"),
        };
        serde_json::json!({"input_activations": inputs, "targets": targets, "weights": weights})
    }

    /// The module a `quantized_layers` key names for weight stem `stem`: a
    /// layer's whole routed or shared expert bank, else the projection.
    fn module(stem: &str) -> String {
        for bank in [".mlp.experts.", ".mlp.shared_experts."] {
            if let Some(at) = stem.find(bank) {
                return stem[..at + bank.len() - 1].to_owned();
            }
        }
        stem.to_owned()
    }

    /// The official config with a ModelOpt mixed-precision
    /// `quantization_config` that stores each class of `classes` in its
    /// encoding: a config group per encoding outside BF16 over its modules,
    /// and a matching `quantized_layers` rule per NVFP4 or MXFP8 module.
    /// Every class in block FP8 gives the official config itself.
    pub fn modelopt_config(classes: Classes) -> Vec<u8> {
        if classes == Classes::BLOCK_FP8 {
            return BLOCK_FP8_CONFIG.to_vec();
        }
        let mut config: Value =
            serde_json::from_slice(BLOCK_FP8_CONFIG).expect("the official config parses");
        let g = &crate::geometry::FLASH;
        let mut layers = serde_json::Map::new();
        let mut targets: Vec<(Encoding, Vec<String>)> = Vec::new();
        for spec in crate::catalog::catalog(g) {
            let Some(stem) = spec.name.strip_suffix(".weight") else {
                continue;
            };
            let encoding = match class_of(g, &spec, stem) {
                Some(Class::Projections) => classes.projections,
                Some(Class::Dense) => classes.dense,
                Some(Class::Routed) => classes.routed,
                Some(Class::MtpRouted) => classes.mtp_routed,
                _ => continue,
            };
            let Some((layer_rule, _)) = rule(encoding) else {
                continue;
            };
            let module = module(stem);
            let at = match targets.iter().position(|(own, _)| *own == encoding) {
                Some(at) => at,
                None => {
                    targets.push((encoding, Vec::new()));
                    targets.len() - 1
                }
            };
            if !targets[at].1.contains(&module) {
                targets[at].1.push(module.clone());
            }
            if let Some(layer_rule) = layer_rule {
                layers.insert(module, layer_rule);
            }
        }
        let groups: serde_json::Map<String, Value> = targets
            .iter()
            .map(|(encoding, modules)| {
                let (_, name) = rule(*encoding).expect("a quantized encoding");
                (name.to_owned(), group(*encoding, modules))
            })
            .collect();
        config["quantization_config"] = serde_json::json!({
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "config_groups": groups,
            "ignore": [],
            "quantized_layers": layers,
        });
        let mut bytes = serde_json::to_vec_pretty(&config).expect("config JSON");
        bytes.push(b'\n');
        bytes
    }

    /// The official config with ModelOpt's single-algorithm form of
    /// `classes`, whose every class outside BF16 must be NVFP4: one config
    /// group of W4A4 NVFP4 (static inputs) targeting the `Linear` class, an
    /// `ignore` list of every module outside those classes (the head,
    /// embedding and vision tower, each main layer's attention, and its
    /// router and shared experts when those stay BF16, and each MTP layer
    /// whole when its experts stay BF16), and a static FP8 KV cache hint.
    pub fn linear_class_config(classes: Classes) -> Vec<u8> {
        let quantized = [
            classes.projections,
            classes.dense,
            classes.routed,
            classes.mtp_routed,
        ];
        assert!(
            quantized
                .iter()
                .all(|&e| matches!(e, Encoding::Nvfp4 | Encoding::Bf16))
                && classes.projections == Encoding::Bf16,
            "the single-algorithm form stores NVFP4 in the dense MLPs and routed experts only"
        );
        let mut config: Value =
            serde_json::from_slice(BLOCK_FP8_CONFIG).expect("the official config parses");
        let g = &crate::geometry::FLASH;
        let prefix = "model.language_model";
        let mut ignore = vec![
            "lm_head".to_owned(),
            format!("{prefix}.embed_tokens"),
            "model.visual*".to_owned(),
        ];
        for layer in 0..g.layers() {
            ignore.push(format!("{prefix}.layers.{layer}.self_attn*"));
            if g.mlp_kind(layer) == crate::geometry::MlpKind::Dense {
                if classes.dense == Encoding::Bf16 {
                    ignore.push(format!("{prefix}.layers.{layer}.mlp*"));
                }
            } else {
                ignore.push(format!("{prefix}.layers.{layer}.mlp.gate"));
                ignore.push(format!("{prefix}.layers.{layer}.mlp.shared_experts*"));
                if classes.routed == Encoding::Bf16 {
                    ignore.push(format!("{prefix}.layers.{layer}.mlp.experts*"));
                }
            }
        }
        assert_eq!(
            classes.mtp_routed,
            Encoding::Bf16,
            "an MTP layer is excluded whole"
        );
        for layer in g.layers()..g.layers() + g.mtp_layers() {
            ignore.push(format!("{prefix}.layers.{layer}*"));
        }
        let nvfp4 = serde_json::json!({
            "dynamic": false, "num_bits": 4, "type": "float", "group_size": 16
        });
        config["quantization_config"] = serde_json::json!({
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {
                    "input_activations": nvfp4,
                    "weights": nvfp4,
                    "targets": ["Linear"],
                },
            },
            "ignore": ignore,
            "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
        });
        let mut bytes = serde_json::to_vec_pretty(&config).expect("config JSON");
        bytes.push(b'\n');
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::{admit, admit_bytes};

    /// The synthetic mixed-precision config of the NVFP4-routed mix.
    fn routed_config() -> Vec<u8> {
        synthetic::config_of(&NVFP4_ROUTED)
    }

    /// The synthetic single-algorithm config of the NVFP4 dense and routed
    /// mix.
    fn dense_routed_config() -> Vec<u8> {
        synthetic::config_of(&NVFP4_DENSE_ROUTED)
    }

    fn nvfp4_config() -> Value {
        serde_json::from_slice(&routed_config()).unwrap()
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
    fn the_mixed_precision_config_selects_flash_and_its_classes() {
        let matched = admit_bytes(&routed_config()).unwrap();
        assert_eq!(matched.geometry.id, "flash");
        assert_eq!(matched.quantization.classes, NVFP4_ROUTED.classes());
        assert_eq!(
            matched.quantization.declared_activation,
            Some((
                format!("{LAYERS}.3.mlp.experts.0.gate_proj"),
                ActivationRecipe::Nvfp4Static
            ))
        );
    }

    /// ModelOpt's single-algorithm form: a default NVFP4 recipe for every
    /// `Linear` with exclusions, its KV hint ignored.
    #[test]
    fn the_single_algorithm_config_selects_nvfp4_dense_and_routed() {
        let matched = admit_bytes(&dense_routed_config()).unwrap();
        assert_eq!(matched.geometry.id, "flash");
        assert_eq!(matched.quantization.classes, NVFP4_DENSE_ROUTED.classes());
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
        let mut config: Value = serde_json::from_slice(&dense_routed_config()).unwrap();
        config["quantization_config"]["unknown_field"] = 1.into();
        let error = admit(&config).unwrap_err();
        assert!(
            error.starts_with("unsupported quantization_config: ")
                && error.contains("unknown_field"),
            "{error}"
        );
        // Without its exclusions the default reaches the latent attention.
        let mut config: Value = serde_json::from_slice(&dense_routed_config()).unwrap();
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
        let mut config: Value = serde_json::from_slice(&dense_routed_config()).unwrap();
        config["quantization_config"]["quant_algo"] = "W4A16_NVFP4".into();
        assert_eq!(
            admit(&config).unwrap_err(),
            "unsupported quantization_config: quantization metadata \
             config.json/quantization_config/config_groups/group_0: declares EncodingRecipe { \
             weight: Nvfp4, activation: Some(Nvfp4Static) }, which disagrees with quant_algo \
             W4A16_NVFP4"
        );
        let mut config: Value = serde_json::from_slice(&dense_routed_config()).unwrap();
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
        let mut config: Value = serde_json::from_slice(&dense_routed_config()).unwrap();
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
        config["quantization_config"]["config_groups"]["group_nvfp4"]["targets"] =
            serde_json::json!(["Embedding"]);
        assert!(admit(&config).unwrap_err().starts_with(
            "unsupported quantization_config: quantization metadata \
             config.json/quantization_config/config_groups/group_nvfp4:"
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

    /// A static FP8 KV cache hint is recorded and ignored for every layout,
    /// and every other hint is refused.
    #[test]
    fn kv_cache_hints_are_recorded_only_when_static_fp8() {
        let hints = [
            serde_json::json!({"dynamic": false, "num_bits": 8, "type": "float"}),
            serde_json::json!({"dynamic": true, "num_bits": 4, "type": "int"}),
            false.into(),
        ];
        let base: Value =
            serde_json::from_slice(include_bytes!("../fixtures/config.json")).unwrap();
        let dense_routed: Value = serde_json::from_slice(&dense_routed_config()).unwrap();
        for config in [base, nvfp4_config(), dense_routed] {
            let classes = admit(&config).unwrap().quantization.classes;
            let mut hinted = config.clone();
            hinted["quantization_config"]["kv_cache_scheme"] = hints[0].clone();
            assert_eq!(admit(&hinted).unwrap().quantization.classes, classes);
            for hint in &hints[1..] {
                let mut config = config.clone();
                config["quantization_config"]["kv_cache_scheme"] = hint.clone();
                let error = admit(&config).unwrap_err();
                assert!(error.contains("kv_cache_scheme"), "{hint}: {error}");
            }
        }
    }

    #[test]
    fn the_base_export_is_block_fp8() {
        let matched = admit_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        assert_eq!(matched.quantization.classes, Classes::BLOCK_FP8);
    }

    #[test]
    fn auto_and_bf16_run_bf16_activations_and_checkpoint_refuses() {
        let quantization = admit_bytes(&routed_config()).unwrap().quantization;
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
            classes: NVFP4_ROUTED.classes(),
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
    fn encodings_a_class_does_not_admit_are_refused_by_module() {
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
        // Classes that are each admitted form a layout together, whichever
        // mix they are.
        let config = relayered(|layers| {
            layers[&format!("{LAYERS}.45.mlp.experts")] =
                serde_json::json!({"quant_algo": "FP8_BLOCK_SCALES", "group_size": 128});
        });
        assert_eq!(
            admit(&config).unwrap().quantization.classes,
            Classes {
                projections: Encoding::Bf16,
                dense: Encoding::Bf16,
                routed: Encoding::Nvfp4,
                mtp_routed: Encoding::Fp8Block128,
            }
        );
        let mut config = nvfp4_config();
        config["quantization_config"]["quant_method"] = "awq".into();
        assert_eq!(
            admit(&config).unwrap_err(),
            "unsupported quantization_config.quant_method: expected one of \
             [\"fp8\",\"modelopt\"], received \"awq\""
        );
    }

    /// Each class is admitted in the encodings its declared uses support,
    /// in declaration order.
    #[test]
    fn class_encodings_come_from_the_declaration() {
        use Encoding::*;
        assert_eq!(
            class_formats(),
            &[
                vec![Fp8Block128, Bf16],
                vec![Fp8Block128, Bf16, Nvfp4],
                vec![Fp8Block128, Nvfp4],
                vec![Fp8Block128, Mxfp8, Bf16],
            ]
        );
    }

    /// The table holds every combination once, each id derived from its
    /// encodings and found again by name; the named layouts are entries of
    /// it.
    #[test]
    fn the_layout_table_is_every_combination_with_derived_ids() {
        let table = layouts();
        assert_eq!(table.len(), 2 * 3 * 2 * 3 * 2 * 2);
        let ids: std::collections::BTreeSet<_> = table.iter().map(|layout| layout.id).collect();
        assert_eq!(ids.len(), table.len());
        for layout in table {
            assert_eq!(
                layout.id,
                Layout::derived_id(layout.classes(), layout.kda_conv, layout.hc_vectors)
            );
            assert_eq!(Layout::named(&Value::from(layout.id)).unwrap(), layout);
        }
        for named in [
            FP8_BLOCK128,
            NVFP4_ROUTED,
            NVFP4_DENSE_ROUTED,
            FP8_NVFP4_ROUTED,
        ] {
            assert_eq!(
                find(named.classes(), named.kda_conv, named.hc_vectors).unwrap(),
                &named
            );
        }
        assert!(Layout::named(&Value::from("fp8-block128")).is_err());
        assert!(FP8_BLOCK128.block_fp8() && !FP8_NVFP4_ROUTED.block_fp8());
    }

    /// The covering layouts are the block-FP8 layout and the two mixes that
    /// hold each class's other encodings, in that order; the representative
    /// layouts add the block-FP8 and NVFP4 mix.
    #[test]
    fn covering_and_representative_layouts() {
        let covering: Vec<_> = covering_layouts().iter().map(|layout| layout.id).collect();
        assert_eq!(
            covering,
            [FP8_BLOCK128.id, NVFP4_ROUTED.id, NVFP4_DENSE_ROUTED.id]
        );
        let representative: Vec<_> = representative_layouts()
            .iter()
            .map(|layout| layout.id)
            .collect();
        assert_eq!(
            representative,
            [
                FP8_BLOCK128.id,
                NVFP4_ROUTED.id,
                NVFP4_DENSE_ROUTED.id,
                FP8_NVFP4_ROUTED.id
            ]
        );
    }

    /// The headers' side tensor dtypes complete the config's classes; a
    /// family stored in two dtypes, or in one the engine does not read, is
    /// refused naming a tensor.
    #[test]
    fn side_tensor_dtypes_resolve_from_the_headers() {
        let matched = admit_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        let g = matched.geometry;
        let stored = |conv: &'static str, hc: &'static str| {
            move |name: &str| -> Result<String, String> {
                Ok(match SideFamily::of(name) {
                    Some(SideFamily::KdaConv) => conv,
                    Some(SideFamily::HcVectors) => hc,
                    None => panic!("{name} is no side tensor"),
                }
                .into())
            }
        };
        let layout = resolve(g, &matched.quantization, stored("BF16", "F32")).unwrap();
        assert_eq!(layout, &FP8_BLOCK128);
        let layout = resolve(g, &matched.quantization, stored("F32", "BF16")).unwrap();
        assert_eq!(
            layout.id,
            "attn-fp8.dense-fp8.experts-fp8.mtp-fp8.conv-f32.hc-bf16"
        );
        assert!(layout.block_fp8());
        let error = resolve(g, &matched.quantization, stored("F16", "F32")).unwrap_err();
        assert!(
            error.starts_with("unsupported checkpoint layout: ") && error.contains("is F16"),
            "{error}"
        );
        let mut first = true;
        let error = resolve(g, &matched.quantization, |name: &str| {
            Ok(match SideFamily::of(name) {
                Some(SideFamily::KdaConv) if std::mem::take(&mut first) => "F32",
                Some(SideFamily::KdaConv) => "BF16",
                _ => "F32",
            }
            .into())
        })
        .unwrap_err();
        assert!(error.contains("must share one dtype"), "{error}");
    }

    /// The synthetic config of any admitted class mix admits to that mix.
    #[test]
    fn synthetic_configs_admit_their_class_mix() {
        for layout in layouts()
            .iter()
            .filter(|layout| (layout.kda_conv, layout.hc_vectors) == (DType::Bf16, DType::F32))
        {
            let config = synthetic::modelopt_config(layout.classes());
            let matched = admit_bytes(&config).unwrap_or_else(|e| panic!("{}: {e}", layout.id));
            assert_eq!(
                matched.quantization.classes,
                layout.classes(),
                "{}",
                layout.id
            );
        }
    }
}
