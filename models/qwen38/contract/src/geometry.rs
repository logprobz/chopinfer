//! Geometry generated from the supported execution configurations.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub id: &'static str,
    pub base: bool,
    pub pinned_names: bool,
    pub tuning: Option<TuningData>,
    pub kernel_tag: &'static str,
    pub shape: Shape,
    pub semantics: Semantics,
    pub limits: RuntimeLimits,
    pub runtime_fields: &'static [RuntimeField],
    pub rules: &'static str,
    pub document: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TuningData {
    pub dense: &'static str,
    pub moe: &'static str,
    pub schedules: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Linear,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeLimits {
    pub indexer_budget: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeField {
    pub name: &'static str,
    pub overridable: bool,
    pub minimum: u32,
    pub maximum: u32,
    pub qualified: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeValues {
    pub indexer_budget: usize,
}

const fn count(length: usize) -> u32 {
    assert!(length <= u32::MAX as usize, "geometry list exceeds u32");
    length as u32
}

impl Geometry {
    /// Check the runtime settings declared by this geometry.
    pub fn check_runtime(&self, config: &serde_json::Value) -> Result<(), String> {
        for field in self.runtime_fields {
            let value = model_compiler_sdk::geometry::field(config, field.name);
            let number = value.and_then(serde_json::Value::as_u64);
            if number.is_none_or(|n| n < u64::from(field.minimum) || n > u64::from(field.maximum)) {
                return Err(format!(
                    "unsupported runtime {}: expected {}..={}, received {}",
                    field.name,
                    field.minimum,
                    field.maximum,
                    value.unwrap_or(&serde_json::Value::Null)
                ));
            }
            if !field.overridable && number != Some(u64::from(field.maximum)) {
                return Err(format!(
                    "unsupported runtime {}: expected compiled limit {}, received {}",
                    field.name,
                    field.maximum,
                    value.unwrap()
                ));
            }
        }
        Ok(())
    }

    /// Declare only fields supported at create, using checkpoint defaults.
    pub fn model_options(
        &self,
        checkpoint_runtime: RuntimeValues,
    ) -> Vec<runtime_bundle::model_options::ModelOptionV1> {
        use runtime_bundle::model_options::{ModelOptionV1, OptionDefinitionV1};
        self.runtime_fields
            .iter()
            .filter(|field| field.overridable)
            .map(|field| {
                let default = match field.name {
                    "text_config.indexer_budget" => {
                        i64::try_from(checkpoint_runtime.indexer_budget)
                            .expect("validated checkpoint runtime value")
                    }
                    _ => panic!("unsupported runtime field {}", field.name),
                };
                ModelOptionV1 {
                    name: field.name.into(),
                    definition: OptionDefinitionV1::Integer {
                        default,
                        minimum: i64::from(field.minimum),
                        maximum: i64::from(field.maximum),
                    },
                    default_is_qualified: default == i64::from(field.qualified),
                }
            })
            .collect()
    }

    pub const fn eos_token_id(&self) -> u32 {
        self.shape.tokens.eos
    }
    pub const fn index_kv_heads(&self) -> u32 {
        self.shape.indexer.kv_heads
    }
    pub const fn ngram_parts(&self) -> u32 {
        self.shape.ngram.parts
    }
    pub const fn ple_layer_ids(&self) -> &'static [u32] {
        self.shape.ple.layer_ids
    }
    pub const fn language_model_only(&self) -> bool {
        self.semantics.language_model_only
    }
    pub const fn layer_types(&self) -> &'static [LayerKind] {
        self.shape.layers
    }
    pub const fn mtp_layer_types(&self) -> &'static [LayerKind] {
        self.shape.mtp.layers
    }
    pub const fn hidden(&self) -> u32 {
        self.shape.hidden
    }
    pub const fn layers(&self) -> u32 {
        count(self.shape.layers.len())
    }
    pub const fn streams(&self) -> u32 {
        self.shape.hc.streams
    }
    pub const fn hc_rank(&self) -> u32 {
        self.shape.hc.rank
    }
    pub const fn linear_key_heads(&self) -> u32 {
        self.shape.linear.key_heads
    }
    pub const fn linear_value_heads(&self) -> u32 {
        self.shape.linear.value_heads
    }
    pub const fn linear_head(&self) -> u32 {
        self.shape.linear.key_dim
    }
    pub const fn linear_value_head(&self) -> u32 {
        self.shape.linear.value_dim
    }
    pub const fn conv_taps(&self) -> u32 {
        self.shape.linear.conv_taps
    }
    pub const fn attention_q_heads(&self) -> u32 {
        self.shape.attention.heads
    }
    pub const fn attention_kv_heads(&self) -> u32 {
        self.shape.attention.kv_heads
    }
    pub const fn head_dim(&self) -> u32 {
        self.shape.attention.head_dim
    }
    pub const fn rotary(&self) -> u32 {
        self.shape.attention.rotary
    }
    pub const fn index_heads(&self) -> u32 {
        self.shape.indexer.heads
    }
    pub const fn index_dim(&self) -> u32 {
        self.shape.indexer.dim
    }
    pub const fn qsa_block(&self) -> u32 {
        self.shape.indexer.compress_ratio
    }
    pub const fn experts(&self) -> u32 {
        self.shape.moe.experts
    }
    pub const fn top_k(&self) -> u32 {
        self.shape.moe.top_k
    }
    pub const fn expert_intermediate(&self) -> u32 {
        self.shape.moe.intermediate
    }
    pub const fn shared_expert_intermediate(&self) -> u32 {
        self.shape.moe.shared_intermediate
    }
    pub const fn ngram_orders(&self) -> u32 {
        self.shape.ngram.orders
    }
    pub const fn ngram_heads_per_order(&self) -> u32 {
        self.shape.ngram.heads_per_order
    }
    pub const fn ngram_vocab_base(&self) -> u32 {
        self.shape.ngram.vocab_base
    }
    pub const fn ngram_alignment(&self) -> u32 {
        self.shape.ngram.alignment
    }
    pub const fn ple_embed_dim(&self) -> u32 {
        self.shape.ple.embed_dim
    }
    pub const fn ple_conv_dilation(&self) -> u32 {
        self.ngram_orders()
    }
    pub const fn vision_heads(&self) -> u32 {
        self.shape.vision.heads
    }
    pub const fn ple_conv_taps(&self) -> u32 {
        self.shape.ple.conv_taps
    }
    pub const fn vocab(&self) -> u32 {
        self.shape.vocab
    }
    pub const fn mtp_layers(&self) -> u32 {
        count(self.shape.mtp.layers.len())
    }
    pub const fn vision_blocks(&self) -> u32 {
        self.shape.vision.depth
    }
    pub const fn vision_hidden(&self) -> u32 {
        self.shape.vision.hidden
    }
    pub const fn vision_intermediate(&self) -> u32 {
        self.shape.vision.intermediate
    }
    pub const fn vision_positions(&self) -> u32 {
        self.shape.vision.positions
    }
    pub const fn vision_out(&self) -> u32 {
        self.shape.vision.out
    }
    pub const fn vision_patch(&self) -> u32 {
        self.shape.vision.patch
    }
    pub const fn vision_temporal_patch(&self) -> u32 {
        self.shape.vision.temporal_patch
    }
    pub const fn vision_channels(&self) -> u32 {
        self.shape.vision.channels
    }
    pub const fn vision_merge(&self) -> u32 {
        self.shape.vision.merge
    }
    pub const fn context_tokens(&self) -> u64 {
        self.shape.context as u64
    }
    pub const fn ple_layer(&self) -> u32 {
        self.shape.ple.layer_ids[0] - 1
    }
    pub const fn eps(&self) -> f32 {
        self.shape.attention.eps as f32
    }
    pub const fn rope_theta(&self) -> f32 {
        self.shape.attention.theta as f32
    }
    pub const fn mrope(&self) -> [u32; 3] {
        let s = self.shape.attention.mrope;
        [s[0], s[1], s[2]]
    }
    pub const fn top_blocks(&self) -> u32 {
        self.limits.indexer_budget / self.qsa_block()
    }
    pub const fn ngram_heads(&self) -> u32 {
        self.ngram_heads_per_order() * (self.ngram_orders() - 1)
    }
    pub const fn ngram_head_width(&self) -> u32 {
        self.shape.ple.embed_dim / self.ngram_heads()
    }
    pub const fn softmax_scale(&self) -> f32 {
        let x = self.head_dim() as f64;
        let mut root = x;
        let mut i = 0;
        while i < 32 {
            root = (root + x / root) * 0.5;
            i += 1;
        }
        (1.0 / root) as f32
    }
    /// Whether a registered layer is full attention; absent layers return false.
    pub const fn is_full(&self, layer: u32) -> bool {
        if layer < self.layers() {
            matches!(self.shape.layers[layer as usize], LayerKind::Full)
        } else {
            let index = (layer - self.layers()) as usize;
            index < self.shape.mtp.layers.len()
                && matches!(self.shape.mtp.layers[index], LayerKind::Full)
        }
    }
    /// Index within a layer kind; absent layers return zero.
    pub const fn ordinal(&self, layer: u32) -> u64 {
        if layer >= self.layers() + self.mtp_layers() {
            return 0;
        }
        let full = self.is_full(layer);
        let mut count = 0;
        let mut i = 0;
        while i < layer {
            if self.is_full(i) == full {
                count += 1;
            }
            i += 1;
        }
        count
    }
    pub const fn full_layers(&self) -> u64 {
        let mut count = 0;
        let mut i = 0;
        while i < self.layers() {
            if self.is_full(i) {
                count += 1;
            }
            i += 1;
        }
        count
    }
}

/// A geometry's routed and shared experts: expert parallel, the shared
/// expert's columns split and padded to 128.
pub const fn moe(g: Geometry) -> kernel_core::rank_shape::MoeConfig {
    kernel_core::rank_shape::MoeConfig {
        experts: g.experts(),
        top_k: g.top_k(),
        intermediate: g.expert_intermediate(),
        split: kernel_core::rank_shape::ExpertSplit::Experts,
        shared_intermediate: g.shared_expert_intermediate(),
        shared_tile: 128,
    }
}

/// The configuration values of a geometry the build lists
/// ([`MODEL_CONFIGS`]): every decoder layer, and the MTP drafter's, is MoE.
macro_rules! geometry_config {
    ($geometry:expr) => {{
        const G: Geometry = $geometry;
        kernel_core::rank_shape::ModelConfig {
            hidden: G.hidden(),
            vocab: G.vocab(),
            blocks: &[kernel_core::rank_shape::BlockClass {
                name: "moe",
                layers: (0, G.layers()),
                block: kernel_core::rank_shape::Block::Moe(moe(G)),
            }],
            linear: Some(kernel_core::rank_shape::LinearAttentionConfig {
                key_heads: G.linear_key_heads(),
                value_heads: G.linear_value_heads(),
                head: G.linear_head(),
                conv_taps: G.conv_taps(),
            }),
            ngram_heads: G.ngram_heads(),
        }
    }};
}

include!(concat!(env!("OUT_DIR"), "/geometries.rs"));

/// The configuration values `geometry`'s per-rank shapes split.
pub fn model_config(geometry: &Geometry) -> kernel_core::rank_shape::ModelConfig {
    let index = GEOMETRIES
        .iter()
        .position(|listed| listed.id == geometry.id)
        .expect("listed geometry");
    MODEL_CONFIGS[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_admission_uses_the_declared_minimum_and_override_flag() {
        let fields = [RuntimeField {
            name: "text_config.indexer_budget",
            overridable: false,
            minimum: 2,
            maximum: 4,
            qualified: 4,
        }];
        let geometry = Geometry {
            runtime_fields: Box::leak(Box::new(fields)),
            ..FLASH
        };
        for value in [0, 1, 5] {
            assert_eq!(
                geometry
                    .check_runtime(&serde_json::json!({"text_config":{"indexer_budget":value}}))
                    .unwrap_err(),
                format!(
                    "unsupported runtime text_config.indexer_budget: expected 2..=4, received {value}"
                )
            );
        }
        assert_eq!(
            geometry
                .check_runtime(&serde_json::json!({"text_config":{"indexer_budget":2}}))
                .unwrap_err(),
            "unsupported runtime text_config.indexer_budget: expected compiled limit 4, received 2"
        );
        assert!(
            geometry
                .check_runtime(&serde_json::json!({"text_config":{"indexer_budget":4}}))
                .is_ok()
        );
        let fields = [RuntimeField {
            overridable: true,
            ..fields[0]
        }];
        let geometry = Geometry {
            runtime_fields: Box::leak(Box::new(fields)),
            ..FLASH
        };
        assert!(
            geometry
                .check_runtime(&serde_json::json!({"text_config":{"indexer_budget":2}}))
                .is_ok()
        );
    }

    #[test]
    fn model_options_use_checkpoint_defaults_and_compiled_limits() {
        use runtime_bundle::model_options::OptionDefinitionV1;
        let runtime = RuntimeValues {
            indexer_budget: 2048,
        };
        assert_eq!(FLASH.runtime_fields.len(), 1);
        assert!(!FLASH.runtime_fields[0].overridable);
        assert!(FLASH.model_options(runtime).is_empty());
        let fields = [RuntimeField {
            overridable: true,
            ..FLASH.runtime_fields[0]
        }];
        let geometry = Geometry {
            runtime_fields: Box::leak(Box::new(fields)),
            ..FLASH
        };
        for (default, qualified) in [(2048, true), (1024, false)] {
            let options = geometry.model_options(RuntimeValues {
                indexer_budget: default,
            });
            assert_eq!(options.len(), 1);
            assert_eq!(options[0].name, "text_config.indexer_budget");
            assert_eq!(
                options[0].definition,
                OptionDefinitionV1::Integer {
                    default: default as i64,
                    minimum: 1,
                    maximum: 2048,
                }
            );
            assert_eq!(options[0].default_is_qualified, qualified);
        }
    }

    #[test]
    fn derived_flash_dimensions_match() {
        assert_eq!(crate::contract::EPS.to_bits(), 1e-6_f32.to_bits());
        assert_eq!(crate::contract::ROPE_THETA.to_bits(), 1e7_f32.to_bits());
        assert_eq!(crate::contract::PLE_LAYER, 1);
        assert_eq!(crate::contract::PLE_CONV_ROWS, 9);
        assert_eq!(crate::encoder::MLP_PADDED, 4352);
        assert_eq!(crate::encoder::PATCH_COLUMNS, 1536);
        assert_eq!(crate::contract::INDEX_ROW, 640);
        assert_eq!(crate::contract::KV_HEADS, 1);
        assert!(!FLASH.is_full(FLASH.layers() + FLASH.mtp_layers()));
        assert_eq!(FLASH.ordinal(FLASH.layers() + FLASH.mtp_layers()), 0);
        assert_eq!(FLASH.softmax_scale().to_bits(), 0.0625_f32.to_bits());
        assert_eq!(FLASH.ngram_heads(), 16);
        assert_eq!(FLASH.ngram_head_width(), 160);
        assert_eq!(FLASH.top_blocks(), 512);
        for layer in 0..=FLASH.layers() {
            assert_eq!(FLASH.is_full(layer), layer % 4 == 3 || layer == 48);
            assert_eq!(
                FLASH.ordinal(layer),
                if FLASH.is_full(layer) {
                    (layer / 4) as u64
                } else {
                    (layer - layer / 4) as u64
                }
            );
        }
    }

    #[test]
    fn flash_matches_pinned_execution_fields() {
        let pinned: serde_json::Value =
            serde_json::from_str(include_str!("../../compiler/fixtures/config.json")).unwrap();
        let generated: serde_json::Value = serde_json::from_str(FLASH.document).unwrap();
        assert_eq!(FLASH.id, "flash");
        assert_eq!(FLASH.kernel_tag, "qwen38");
        assert_eq!(GEOMETRIES[0], FLASH);
        assert_eq!(FLASH.shape.layers.len(), FLASH.layers() as usize);
        assert_eq!(FLASH.shape.mtp.layers.len(), FLASH.mtp_layers() as usize);
        assert_eq!(FLASH.shape.attention.rotary, 64);
        assert_eq!(FLASH.limits.indexer_budget, 2048);
        assert_eq!(
            FLASH.shape.attention.eps.to_bits(),
            pinned["text_config"]["rms_norm_eps"]
                .as_f64()
                .unwrap()
                .to_bits()
        );
        fn compare(path: &str, actual: &serde_json::Value, expected: &serde_json::Value) {
            if let Some(object) = actual.as_object() {
                for (key, value) in object {
                    compare(&format!("{path}.{key}"), value, &expected[key]);
                }
            } else if path != ".text_config.norm_topk_prob" {
                assert_eq!(actual, expected, "{path}");
            }
        }
        compare("", &generated, &pinned);
    }
}
