//! The projection header fixtures: the captured FP8 export, and two mixed
//! layouts derived from it at test time by per-encoding rules. Each is the
//! JSON text `checkpoint::fixture` reads.
//!
//! Every weight keeps its logical shape. MXFP8 stores E4M3 codes with one
//! UE8M0 byte per 32 K values; NVFP4 stores E2M1 pairs with one E4M3 scale
//! per 16 K values and an F32 global scale (one per lookup table) and,
//! outside weight-only rules, an F32 input scale; block FP8 stores E4M3
//! codes with one BF16 inverse scale per 128 x 128 block.
#![allow(dead_code)]

use serde_json::{Map, Value, json};
use std::sync::OnceLock;

/// The FP8 export's header families and quantization_config.
pub(crate) const FP8: &str = include_str!("../fixtures/projections/fp8.json");

/// A mixed layout of the FP8 export's tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    /// NVFP4 routed experts with input scales, per-tensor FP8 lookup
    /// tables, block FP8 MTP experts, BF16 everywhere else, and no vision
    /// tower. Its quantization_config names each rule twice: as a
    /// quantization group and as a quantized layer.
    Nvfp4RoutedFp8Mtp,
    /// MXFP8 attention, linear-attention and shared-expert projections,
    /// NVFP4 routed experts and lookup tables, weight-only NVFP4 MTP
    /// experts, F32 hyper-connection and PLE norm weights (100 of them),
    /// and the BF16 vision tower.
    Mxfp8DenseNvfp4,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Encoding {
    Fp8Block,
    Fp8Table,
    Mxfp8,
    Nvfp4,
    Nvfp4WeightOnly,
    Nvfp4Table,
}

const LAYERS: &str = "model.language_model.layers.";

/// The main-layer dense projections the MXFP8 layout quantizes.
const MXFP8_DENSE: [&str; 13] = [
    ".linear_attn.in_proj_a",
    ".linear_attn.in_proj_b",
    ".linear_attn.in_proj_qkv",
    ".linear_attn.in_proj_z",
    ".linear_attn.out_proj",
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".self_attn.indexer.index_qk_proj",
    ".mlp.shared_expert.gate_proj",
    ".mlp.shared_expert.up_proj",
    ".mlp.shared_expert.down_proj",
];

impl Layout {
    pub(crate) const ALL: [Layout; 2] = [Layout::Nvfp4RoutedFp8Mtp, Layout::Mxfp8DenseNvfp4];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Layout::Nvfp4RoutedFp8Mtp => "nvfp4-routed-fp8-mtp",
            Layout::Mxfp8DenseNvfp4 => "mxfp8-dense-nvfp4",
        }
    }

    /// The layout's fixture text, synthesized once per process.
    pub(crate) fn text(self) -> &'static str {
        static TEXTS: [OnceLock<String>; 2] = [OnceLock::new(), OnceLock::new()];
        TEXTS[self as usize].get_or_init(|| synthesize(self))
    }

    /// The encoding of the weight `module` (a family name less `.weight`),
    /// or None for BF16.
    fn encoding(self, module: &str) -> Option<Encoding> {
        let main = module.starts_with(LAYERS);
        if module.contains(".ngram_embedding.shard_") {
            return Some(match self {
                Layout::Nvfp4RoutedFp8Mtp => Encoding::Fp8Table,
                Layout::Mxfp8DenseNvfp4 => Encoding::Nvfp4Table,
            });
        }
        if module.contains(".mlp.experts.") {
            return Some(match (self, main) {
                (_, true) => Encoding::Nvfp4,
                (Layout::Nvfp4RoutedFp8Mtp, false) => Encoding::Fp8Block,
                (Layout::Mxfp8DenseNvfp4, false) => Encoding::Nvfp4WeightOnly,
            });
        }
        (self == Layout::Mxfp8DenseNvfp4 && main && MXFP8_DENSE.iter().any(|p| module.ends_with(p)))
            .then_some(Encoding::Mxfp8)
    }

    /// Whether the layout stores the BF16 norm weight `tensor` as F32.
    fn f32_norm(self, tensor: &str) -> bool {
        self == Layout::Mxfp8DenseNvfp4
            && tensor.starts_with("model.language_model.")
            && (tensor.ends_with(".hc_norm.weight") || tensor.contains(".ple.norm_"))
    }
}

impl Encoding {
    /// The quantized layer rule ModelOpt writes for this encoding.
    fn rule(self) -> Value {
        match self {
            Encoding::Fp8Block => json!({"quant_algo": "FP8_PB_WO", "group_size": 128}),
            Encoding::Fp8Table => json!({"quant_algo": "FP8"}),
            Encoding::Mxfp8 => json!({"quant_algo": "MXFP8", "group_size": 32}),
            Encoding::Nvfp4 | Encoding::Nvfp4Table => {
                json!({"quant_algo": "NVFP4", "group_size": 16})
            }
            Encoding::Nvfp4WeightOnly => json!({"quant_algo": "W4A16_NVFP4", "group_size": 16}),
        }
    }
}

fn tensor(family: &Value) -> &str {
    family["tensor"].as_str().unwrap()
}

/// The indices each `{axisN}` token of `axes` takes.
fn indices(axis: &Value) -> Vec<u64> {
    if let Some([start, end]) = axis["range"].as_array().map(Vec::as_slice) {
        (start.as_u64().unwrap()..end.as_u64().unwrap()).collect()
    } else {
        let list = axis["indices"].as_array().unwrap();
        list.iter().map(|i| i.as_u64().unwrap()).collect()
    }
}

/// `template` with each axis token it names replaced by each index.
fn expand(template: &str, axes: &Value) -> Vec<String> {
    let mut names = vec![template.to_owned()];
    for (token, axis) in axes.as_object().unwrap() {
        if template.contains(token.as_str()) {
            names = names
                .iter()
                .flat_map(|name| {
                    indices(axis)
                        .into_iter()
                        .map(move |i| name.replace(token.as_str(), &i.to_string()))
                })
                .collect();
        }
    }
    names
}

/// The layer indices of the main-layer family whose name ends in `suffix`.
fn layers(families: &[Value], suffix: &str) -> Vec<u64> {
    let family = families
        .iter()
        .find(|f| tensor(f).starts_with(LAYERS) && tensor(f).ends_with(suffix))
        .unwrap();
    indices(&family["axes"]["{axis0}"])
}

fn synthesize(layout: Layout) -> String {
    let fp8: Value = serde_json::from_str(FP8).unwrap();
    let source = fp8["families"].as_array().unwrap();
    let table = source
        .iter()
        .find_map(|f| tensor(f).split_once(".shard_").map(|(t, _)| t.to_owned()))
        .unwrap();
    let table_encoding = layout.encoding(&format!("{table}.shard_")).unwrap();
    let mut families = Vec::new();
    let mut rules = Map::new();
    for family in source {
        let name = tensor(family);
        if layout == Layout::Nvfp4RoutedFp8Mtp && name.starts_with("model.visual.") {
            continue;
        }
        // Block FP8 inverse scales follow their weights below.
        if name.ends_with(".weight_scale_inv") {
            continue;
        }
        let mut add = |name: String, dtype: &str, shape: Vec<u64>| {
            families.push(json!({"tensor": name, "dtype": dtype, "shape": shape,
                "axes": family["axes"]}));
        };
        // The lookup table's companion: per-tensor FP8 keeps the FP8
        // export's BF16 scale, NVFP4 holds an F32 global scale.
        if name == format!("{table}.weight_scale") {
            match table_encoding {
                Encoding::Fp8Table => add(name.into(), "BF16", vec![1]),
                _ => add(format!("{table}.weight_scale_2"), "F32", vec![1]),
            }
            continue;
        }
        let Some((module, encoding)) = name
            .strip_suffix(".weight")
            .and_then(|m| Some((m, layout.encoding(m)?)))
        else {
            let mut family = family.clone();
            if layout.f32_norm(name) {
                assert_eq!(family["dtype"], "BF16", "{name}");
                family["dtype"] = "F32".into();
            }
            families.push(family);
            continue;
        };
        let shape: Vec<u64> = serde_json::from_value(family["shape"].clone()).unwrap();
        let [rows, k] = shape[..] else {
            panic!("{name}: a quantized weight has two dimensions");
        };
        let part = |suffix: &str| format!("{module}{suffix}");
        match encoding {
            Encoding::Fp8Block => {
                add(part(".weight"), "F8_E4M3", vec![rows, k]);
                add(
                    part(".weight_scale_inv"),
                    "BF16",
                    vec![rows.div_ceil(128), k.div_ceil(128)],
                );
            }
            Encoding::Fp8Table => add(part(".weight"), "F8_E4M3", vec![rows, k]),
            Encoding::Mxfp8 => {
                add(part(".weight"), "F8_E4M3", vec![rows, k]);
                add(part(".weight_scale"), "U8", vec![rows, k / 32]);
            }
            Encoding::Nvfp4 | Encoding::Nvfp4WeightOnly | Encoding::Nvfp4Table => {
                add(part(".weight"), "U8", vec![rows, k / 2]);
                add(part(".weight_scale"), "F8_E4M3", vec![rows, k / 16]);
                if encoding != Encoding::Nvfp4Table {
                    add(part(".weight_scale_2"), "F32", vec![]);
                }
                if encoding == Encoding::Nvfp4 {
                    add(part(".input_scale"), "F32", vec![]);
                }
            }
        }
        // Rules name the expert bank and the whole lookup table.
        let rule_module = if let Some((layer, _)) = module.split_once(".mlp.experts.") {
            format!("{layer}.mlp.experts")
        } else if module.contains(".shard_") {
            table.clone()
        } else {
            module.to_owned()
        };
        for module in expand(&rule_module, &family["axes"]) {
            rules.insert(module, encoding.rule());
        }
    }
    let quantization_config = match layout {
        Layout::Nvfp4RoutedFp8Mtp => {
            let mut groups = Map::new();
            for (algo, weights, inputs) in [
                (
                    "NVFP4",
                    json!({"num_bits": 4, "group_size": 16}),
                    Some(false),
                ),
                (
                    "FP8_PB_WO",
                    json!({"num_bits": 8, "group_size": 128}),
                    Some(true),
                ),
                ("FP8", json!({"num_bits": 8}), None),
            ] {
                let mut weights = weights;
                weights["dynamic"] = false.into();
                weights["type"] = "float".into();
                let mut targets: Vec<&String> = rules
                    .iter()
                    .filter(|(_, rule)| rule["quant_algo"] == algo)
                    .map(|(module, _)| module)
                    .collect();
                targets.sort();
                let mut group = json!({"weights": weights, "targets": targets});
                if let Some(dynamic) = inputs {
                    let mut inputs = group["weights"].clone();
                    inputs["dynamic"] = dynamic.into();
                    group["input_activations"] = inputs;
                }
                groups.insert(format!("group_{}", groups.len()), group);
            }
            // Every BF16 module of each main layer, by its layer kind.
            let linear = layers(source, ".linear_attn.in_proj_a.weight");
            let mut ignore: Vec<String> = [
                "lm_head",
                "model.language_model.embed_tokens",
                "model.language_model.hyper_connection_mixer*",
                "model.visual*",
            ]
            .map(String::from)
            .to_vec();
            for layer in layers(source, ".mlp.gate.weight") {
                let attention = if linear.contains(&layer) {
                    "linear_attn*"
                } else {
                    "self_attn*"
                };
                for module in [
                    "attn_hyper_connection*",
                    attention,
                    "mlp.gate",
                    "mlp.shared_expert*",
                    "mlp.shared_expert_gate",
                    "mlp_hyper_connection*",
                ] {
                    ignore.push(format!("{LAYERS}{layer}.{module}"));
                }
            }
            ignore.sort();
            json!({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                "config_groups": groups, "ignore": ignore, "quantized_layers": rules})
        }
        Layout::Mxfp8DenseNvfp4 => json!({
            "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": null,
            "ignore": ["lm_head", "model.visual.*", "mtp.fc_embedding", "mtp.fc_hidden",
                "mtp.pre_fc_norm_*", "mtp.hyper_connection_mixer*", "mtp.layers.0.self_attn*",
                "mtp.layers.0.attn_hyper_connection*", "mtp.layers.0.mlp_hyper_connection*",
                "mtp.layers.0.mlp.gate", "mtp.layers.0.mlp.shared_expert*"],
            "quantized_layers": rules,
        }),
    };
    json!({"families": families, "quantization_config": quantization_config}).to_string()
}
