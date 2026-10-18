//! Mixed checkpoint layouts for CPU tests, synthesized from the FP8
//! export's configuration and header families (see
//! `projection_headers::exports`). A fixture keeps the checkpoint geometry
//! and selects a few roles of each encoding; its quantization rules cover
//! only those roles and declare no `weight_block_size`.
use serde_json::Value;
use std::sync::OnceLock;

/// The projection header fixtures, which the projection comparisons share.
#[path = "../../contract/tests/projections_support/mod.rs"]
pub(crate) mod headers;

pub(crate) use headers::exports::Layout;

/// The selected roles, as tensor name prefixes: a hyper-connection norm,
/// a linear-attention projection, one routed expert, one lookup shard with
/// its table's global scale, and one MTP expert.
const SELECTED: [&str; 6] = [
    "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight",
    "model.language_model.layers.0.linear_attn.in_proj_a.",
    "model.language_model.layers.0.mlp.experts.0.gate_proj.",
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.",
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
    "mtp.layers.0.mlp.experts.0.gate_proj.",
];

/// A selected tensor: its name, dtype and shape.
pub(crate) type Tensor = (String, String, Vec<u64>);

/// The layout's selected tensors, in name order.
pub(crate) fn tensors(layout: Layout) -> &'static [Tensor] {
    static TENSORS: [OnceLock<Vec<Tensor>>; 2] = [OnceLock::new(), OnceLock::new()];
    TENSORS[layout as usize].get_or_init(|| {
        headers::fixture(layout.text())
            .tensors
            .into_iter()
            .filter(|(name, _)| SELECTED.iter().any(|p| name.starts_with(p)))
            .map(|(name, info)| (name, info.dtype, info.shape))
            .collect()
    })
}

fn selects(module_pattern: &str, layout: Layout) -> bool {
    tensors(layout).iter().any(|(name, _, _)| {
        module_pattern
            .strip_suffix('*')
            .map_or(name.starts_with(&format!("{module_pattern}.")), |prefix| {
                name.starts_with(prefix)
            })
    })
}

/// config.json's quantization_config: the layout's rules for the selected
/// roles.
pub(crate) fn quantization_config(layout: Layout) -> Value {
    let full: Value = serde_json::from_str(layout.text()).unwrap();
    let mut config = full["quantization_config"].clone();
    let rules = config["quantized_layers"].as_object_mut().unwrap();
    rules.retain(|module, _| selects(module, layout));
    let ignore = config["ignore"].as_array_mut().unwrap();
    ignore.retain(|pattern| selects(pattern.as_str().unwrap(), layout));
    if let Some(groups) = config
        .get_mut("config_groups")
        .and_then(Value::as_object_mut)
    {
        for group in groups.values_mut() {
            let targets = group["targets"].as_array_mut().unwrap();
            targets.retain(|module| selects(module.as_str().unwrap(), layout));
        }
        groups.retain(|_, group| !group["targets"].as_array().unwrap().is_empty());
    }
    config
}

/// The checkpoint's config.json: the FP8 export's, with the layout's
/// quantization_config.
pub(crate) fn config(layout: Layout) -> Value {
    let mut config: Value = serde_json::from_str(include_str!("../fixtures/config.json")).unwrap();
    config["quantization_config"] = quantization_config(layout);
    config
}

/// The checkpoint's hf_quant_config.json. The NVFP4-routed layout nests
/// its rules under `quantization`, names the block FP8 rule
/// `FP8_BLOCK_SCALES` and lists its exclusions as `exclude_modules`; the
/// MXFP8 layout repeats config.json's rules.
pub(crate) fn hf_quant_config(layout: Layout) -> Value {
    let config = quantization_config(layout);
    match layout {
        Layout::Nvfp4RoutedFp8Mtp => {
            let mut rules = config["quantized_layers"].clone();
            for rule in rules.as_object_mut().unwrap().values_mut() {
                if rule["quant_algo"] == "FP8_PB_WO" {
                    rule["quant_algo"] = "FP8_BLOCK_SCALES".into();
                }
            }
            serde_json::json!({"quantization": {
                "quant_algo": config["quant_algo"], "kv_cache_quant_algo": null,
                "group_size": 16, "exclude_modules": config["ignore"],
                "quantized_layers": rules,
            }})
        }
        Layout::Mxfp8DenseNvfp4 => config,
    }
}
