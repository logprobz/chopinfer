//! Geometry generated from the supported execution configurations.
//!
//! Every generated field is fixed at compile time: kernel instances and the
//! tensor catalog derive from it. Runtime fields (`index_topk`) stay bounded
//! by the geometry's compiled limit and are offered as model options.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub id: &'static str,
    /// Whether the geometry comes from `models/glm5next/geometries`.
    pub base: bool,
    pub kernel_tag: &'static str,
    pub shape: Shape,
    pub semantics: Semantics,
    pub limits: RuntimeLimits,
    pub runtime_fields: &'static [RuntimeField],
    pub rules: &'static str,
    pub document: &'static str,
}

/// The token mixer of a main layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Per-channel-decay delta rule (KDA).
    Linear,
    /// Latent attention over the indexer's pooled-key selection.
    Sparse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpKind {
    Dense,
    Moe,
}

/// Compiled maxima of the runtime fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeLimits {
    pub index_topk: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeField {
    pub name: &'static str,
    pub overridable: bool,
    pub minimum: u32,
    pub maximum: u32,
    /// Accepted values are multiples of this (`index_topk` keeps whole
    /// pools of `text_config.index_kpool` keys).
    pub step: u32,
    /// The geometry's own value, which qualification runs at.
    pub qualified: u32,
}

/// A checkpoint's runtime field values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeValues {
    pub index_topk: u32,
}

const fn count(length: usize) -> u32 {
    assert!(length <= u32::MAX as usize, "geometry list exceeds u32");
    length as u32
}

impl Geometry {
    /// Check the runtime settings declared by this geometry and read them.
    pub fn check_runtime(&self, config: &serde_json::Value) -> Result<RuntimeValues, String> {
        self.runtime_values(|field| model_compiler_sdk::geometry::field(config, field.name))
    }

    /// Check resolved `chop.model-options.v1` values (the host's resolution
    /// of [`Self::model_options`]) and read them; an option absent from
    /// `values` keeps the checkpoint's value.
    pub fn resolve_model_options(
        &self,
        checkpoint: RuntimeValues,
        values: &std::collections::BTreeMap<String, serde_json::Value>,
    ) -> Result<RuntimeValues, String> {
        let defaults = self.model_options(checkpoint);
        if let Some(name) = values
            .keys()
            .find(|name| !defaults.iter().any(|option| &option.name == *name))
        {
            return Err(format!("unknown model option {name}"));
        }
        let mut owned = std::collections::BTreeMap::new();
        for field in self.runtime_fields {
            let value = match values.get(field.name) {
                Some(value) if field.overridable => value.clone(),
                _ => match field.name {
                    "text_config.index_topk" => checkpoint.index_topk.into(),
                    _ => return Err(format!("unsupported runtime field {}", field.name)),
                },
            };
            owned.insert(field.name, value);
        }
        self.runtime_values(|field| owned.get(field.name))
    }

    /// Each runtime field within its range, a multiple of its step, and
    /// equal to the qualified value unless it is overridable.
    fn runtime_values<'a>(
        &self,
        lookup: impl Fn(&RuntimeField) -> Option<&'a serde_json::Value>,
    ) -> Result<RuntimeValues, String> {
        let mut values = RuntimeValues {
            index_topk: self.limits.index_topk,
        };
        for field in self.runtime_fields {
            let value = lookup(field);
            let number = value
                .and_then(serde_json::Value::as_u64)
                .filter(|&n| {
                    n >= u64::from(field.minimum)
                        && n <= u64::from(field.maximum)
                        && n.is_multiple_of(u64::from(field.step))
                })
                .ok_or_else(|| {
                    format!(
                        "unsupported runtime {}: expected a multiple of {} in {}..={}, received {}",
                        field.name,
                        field.step,
                        field.minimum,
                        field.maximum,
                        value.unwrap_or(&serde_json::Value::Null)
                    )
                })?;
            if !field.overridable && number != u64::from(field.qualified) {
                return Err(format!(
                    "unsupported runtime {}: expected {}, received {number}",
                    field.name, field.qualified
                ));
            }
            match field.name {
                "text_config.index_topk" => values.index_topk = number as u32,
                _ => return Err(format!("unsupported runtime field {}", field.name)),
            }
        }
        Ok(values)
    }

    /// The `chop.model-options.v1` options of the overridable runtime
    /// fields, defaulting to the checkpoint's values.
    pub fn model_options(
        &self,
        checkpoint: RuntimeValues,
    ) -> Vec<runtime_bundle::model_options::ModelOptionV1> {
        use runtime_bundle::model_options::{ModelOptionV1, OptionDefinitionV1};
        self.runtime_fields
            .iter()
            .filter(|field| field.overridable)
            .map(|field| {
                let default = match field.name {
                    "text_config.index_topk" => checkpoint.index_topk,
                    _ => panic!("unsupported runtime field {}", field.name),
                };
                ModelOptionV1 {
                    name: field.name.into(),
                    definition: OptionDefinitionV1::Integer {
                        default: i64::from(default),
                        minimum: i64::from(field.minimum),
                        maximum: i64::from(field.maximum),
                    },
                    default_is_qualified: default == field.qualified,
                }
            })
            .collect()
    }

    pub const fn hidden(&self) -> u32 {
        self.shape.hidden
    }
    pub const fn vocab(&self) -> u32 {
        self.shape.vocab
    }
    /// Main (non-MTP) layers.
    pub const fn layers(&self) -> u32 {
        count(self.shape.layers.len())
    }
    pub const fn mtp_layers(&self) -> u32 {
        self.shape.mtp_layers
    }
    pub const fn layer_kind(&self, layer: u32) -> LayerKind {
        self.shape.layers[layer as usize]
    }
    pub const fn mlp_kind(&self, layer: u32) -> MlpKind {
        self.shape.mlp_layers[layer as usize]
    }
    pub const fn count_layers(&self, kind: LayerKind) -> u32 {
        let mut count = 0;
        let mut i = 0;
        while i < self.shape.layers.len() {
            if self.shape.layers[i] as u8 == kind as u8 {
                count += 1;
            }
            i += 1;
        }
        count
    }
    pub const fn count_mlp(&self, kind: MlpKind) -> u32 {
        let mut count = 0;
        let mut i = 0;
        while i < self.shape.mlp_layers.len() {
            if self.shape.mlp_layers[i] as u8 == kind as u8 {
                count += 1;
            }
            i += 1;
        }
        count
    }
    /// Width of the hyper-connection residual: one hidden row per stream.
    pub const fn hc_width(&self) -> u32 {
        self.shape.hc.streams * self.shape.hidden
    }
    /// Rows of each mHC mixing projection: pre and post per stream, plus the
    /// stream-by-stream combination.
    pub const fn hc_mix(&self) -> u32 {
        (2 + self.shape.hc.streams) * self.shape.hc.streams
    }
    /// Channels of each KDA projection (heads x head dim).
    pub const fn linear_width(&self) -> u32 {
        self.shape.linear.heads * self.shape.linear.head_dim
    }
    pub const fn qk_head(&self) -> u32 {
        self.shape.latent.qk_nope + self.shape.latent.qk_rope
    }
    /// Rows of the latent up-projection: a key and a value per head.
    pub const fn kv_up_rows(&self) -> u32 {
        self.shape.latent.heads * (self.shape.latent.qk_nope + self.shape.latent.v_head)
    }
    pub const fn vision_head_dim(&self) -> u32 {
        self.shape.vision.hidden / self.shape.vision.heads
    }
    pub const fn context_tokens(&self) -> u64 {
        self.shape.context as u64
    }
    pub const fn eps(&self) -> f32 {
        self.shape.eps as f32
    }
    /// Pooled keys the indexer keeps at the compiled limit.
    pub const fn kept_pools(&self) -> u32 {
        self.limits.index_topk / self.shape.indexer.pool
    }
}

include!(concat!(env!("OUT_DIR"), "/geometries.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_layer_counts() {
        assert_eq!(GEOMETRIES[0], FLASH);
        assert!(GEOMETRIES[0].base);
        assert_eq!(FLASH.kernel_tag, "glm5next");
        assert_eq!(FLASH.layers(), 45);
        assert_eq!(FLASH.count_layers(LayerKind::Linear), 34);
        assert_eq!(FLASH.count_layers(LayerKind::Sparse), 11);
        assert_eq!(FLASH.count_mlp(MlpKind::Dense), 3);
        assert_eq!(FLASH.count_mlp(MlpKind::Moe), 42);
        assert_eq!(FLASH.mtp_layers(), 1);
        for layer in 0..FLASH.layers() {
            let sparse = layer % 4 == 3;
            assert_eq!(FLASH.layer_kind(layer) == LayerKind::Sparse, sparse);
            assert_eq!(FLASH.mlp_kind(layer) == MlpKind::Dense, layer < 3);
        }
        assert_eq!(FLASH.hc_width(), 16384);
        assert_eq!(FLASH.hc_mix(), 24);
        assert_eq!(FLASH.linear_width(), 8192);
        assert_eq!(FLASH.qk_head(), 256);
        assert_eq!(FLASH.kv_up_rows(), 32768);
        assert_eq!(FLASH.vision_head_dim(), 64);
        assert_eq!(FLASH.kept_pools(), 512);
        assert_eq!(FLASH.eps().to_bits(), 1e-5_f32.to_bits());
        assert_eq!(FLASH.shape.linear.gate_lower_bound, -5.0);
        assert_eq!(FLASH.shape.hc.eps, 1e-6);
        assert_eq!(FLASH.shape.moe.routed_scale, 2.5);
        assert_eq!(FLASH.shape.tokens.eos, &[154820, 154827, 154829]);
    }

    #[test]
    fn index_topk_is_a_model_option_up_to_the_compiled_limit() {
        use runtime_bundle::model_options::OptionDefinitionV1;
        let config: serde_json::Value = serde_json::from_str(FLASH.document).unwrap();
        let runtime = FLASH.check_runtime(&config).unwrap();
        assert_eq!(runtime, RuntimeValues { index_topk: 2048 });
        let options = FLASH.model_options(runtime);
        assert_eq!(options.len(), 1);
        assert_eq!(options[0].name, "text_config.index_topk");
        assert_eq!(
            options[0].definition,
            OptionDefinitionV1::Integer {
                default: 2048,
                minimum: 4,
                maximum: 2048
            }
        );
        assert!(options[0].default_is_qualified);
        let mut over = config;
        over["text_config"]["index_topk"] = 4096.into();
        assert_eq!(
            FLASH.check_runtime(&over).unwrap_err(),
            "unsupported runtime text_config.index_topk: expected a multiple of 4 in 4..=2048, received 4096"
        );
    }

    #[test]
    fn resolved_index_topk_options_keep_whole_pools() {
        use std::collections::BTreeMap;
        let checkpoint = RuntimeValues { index_topk: 2048 };
        let resolve = |value: u32| {
            FLASH.resolve_model_options(
                checkpoint,
                &BTreeMap::from([("text_config.index_topk".to_owned(), value.into())]),
            )
        };
        for top_k in [4, 2048] {
            assert_eq!(resolve(top_k), Ok(RuntimeValues { index_topk: top_k }));
        }
        for top_k in [5, 2047] {
            assert_eq!(
                resolve(top_k).unwrap_err(),
                format!(
                    "unsupported runtime text_config.index_topk: expected a multiple of 4 in 4..=2048, received {top_k}"
                )
            );
        }
        assert_eq!(
            FLASH.resolve_model_options(checkpoint, &BTreeMap::new()),
            Ok(checkpoint)
        );
        assert!(
            FLASH
                .resolve_model_options(
                    checkpoint,
                    &BTreeMap::from([("text_config.hc_mult".to_owned(), 2.into())])
                )
                .is_err()
        );
    }
}
