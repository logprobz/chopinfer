//! Geometry generated from the supported execution configurations
//! (`models/deepseek41/geometries`, and the directories
//! `CHOPINFER_DEEPSEEK41_GEOMETRY_DIRS` lists). Every field is fixed at
//! compile time: kernel instances and the tensor catalog derive from it.
//! Runtime selection counts stay within those compiled limits.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub id: &'static str,
    /// Whether the geometry comes from `models/deepseek41/geometries`.
    pub base: bool,
    pub kernel_tag: &'static str,
    pub shape: Shape,
    pub rules: &'static str,
    pub document: &'static str,
}

/// How a layer finds its compressed keys and its top-k selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerRole {
    /// Sliding-window attention only (compress ratio 0).
    Window,
    /// Owns a compressor, index keys and an indexer.
    Full,
    /// Reads the latest source's compressed KV and top-k selection.
    Reuse,
    /// Reads the latest source's compressed KV and index keys, and selects
    /// with its own index query and weights.
    Reindex,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeField {
    pub name: &'static str,
    pub overridable: bool,
    pub minimum: u32,
    pub maximum: u32,
    /// The geometry's own value, which qualification runs at.
    pub qualified: u32,
}

/// A checkpoint's runtime field values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeValues {
    pub index_topk: u32,
    pub candidate_topk_blocks: u32,
    pub indexer_mma: bool,
    pub indexer_prefill_tile: bool,
}

const fn count(length: usize) -> u32 {
    assert!(length <= u32::MAX as usize, "geometry list exceeds u32");
    length as u32
}

const fn contains(ids: &[u32], id: u32) -> bool {
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == id {
            return true;
        }
        i += 1;
    }
    false
}

impl Geometry {
    /// The compiled selection limit and its qualified checkpoint value.
    pub const fn runtime_fields(&self) -> [RuntimeField; 2] {
        [
            RuntimeField {
                name: "text_config.candidate_topk_blocks",
                overridable: true,
                minimum: 1,
                maximum: self.shape.candidates.top_blocks,
                qualified: self.shape.candidates.top_blocks,
            },
            RuntimeField {
                name: "text_config.index_topk",
                overridable: true,
                minimum: 1,
                maximum: self.shape.indexer.top_k,
                qualified: self.shape.indexer.top_k,
            },
        ]
    }

    pub const fn native_indexer_supported(&self) -> bool {
        self.shape.indexer.heads == 32
            && self.shape.indexer.dim == 128
            && self.shape.candidates.block == 8
    }

    pub const fn runtime_defaults(&self) -> RuntimeValues {
        RuntimeValues {
            index_topk: self.shape.indexer.top_k,
            candidate_topk_blocks: self.shape.candidates.top_blocks,
            indexer_mma: self.native_indexer_supported(),
            indexer_prefill_tile: false,
        }
    }

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
        crate::prefill_profile::Profile::read(values)?;
        let defaults = self.model_options(checkpoint);
        if let Some(name) = values
            .keys()
            .find(|name| !defaults.iter().any(|option| &option.name == *name))
        {
            return Err(format!("unknown model option {name}"));
        }
        let mma = match values.get("text_config.indexer_mma") {
            None => checkpoint.indexer_mma,
            Some(value) => value
                .as_bool()
                .ok_or("text_config.indexer_mma requires a boolean")?,
        };
        if mma && !self.native_indexer_supported() {
            return Err("text_config.indexer_mma is unsupported by this geometry".into());
        }
        let prefill_tile = match values.get("text_config.indexer_prefill_tile") {
            None => checkpoint.indexer_prefill_tile,
            Some(value) => value
                .as_bool()
                .ok_or("text_config.indexer_prefill_tile requires a boolean")?,
        };
        if prefill_tile && (!mma || self != &FLASH) {
            return Err("text_config.indexer_prefill_tile requires native Flash scoring".into());
        }
        let mut owned = std::collections::BTreeMap::new();
        for field in self.runtime_fields() {
            let value = match values.get(field.name) {
                Some(value) if field.overridable => value.clone(),
                _ => match field.name {
                    "text_config.index_topk" => checkpoint.index_topk.into(),
                    "text_config.candidate_topk_blocks" => checkpoint.candidate_topk_blocks.into(),
                    _ => return Err(format!("unsupported runtime field {}", field.name)),
                },
            };
            owned.insert(field.name, value);
        }
        let mut runtime = self.runtime_values(|field| owned.get(field.name))?;
        runtime.indexer_mma = mma;
        runtime.indexer_prefill_tile = prefill_tile;
        Ok(runtime)
    }

    /// Each runtime field within its range and
    /// equal to the qualified value unless it is overridable.
    fn runtime_values<'a>(
        &self,
        lookup: impl Fn(&RuntimeField) -> Option<&'a serde_json::Value>,
    ) -> Result<RuntimeValues, String> {
        let mut values = RuntimeValues {
            index_topk: self.shape.indexer.top_k,
            candidate_topk_blocks: self.shape.candidates.top_blocks,
            indexer_mma: self.native_indexer_supported(),
            indexer_prefill_tile: false,
        };
        for field in self.runtime_fields() {
            let value = lookup(&field);
            let number = value
                .and_then(serde_json::Value::as_u64)
                .filter(|&n| n >= u64::from(field.minimum) && n <= u64::from(field.maximum))
                .ok_or_else(|| {
                    format!(
                        "unsupported runtime {}: expected {}..={}, received {}",
                        field.name,
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
                "text_config.candidate_topk_blocks" => values.candidate_topk_blocks = number as u32,
                _ => return Err(format!("unsupported runtime field {}", field.name)),
            }
        }
        // The newest candidate block is pinned even when only its first
        // position is visible. Keep enough finite positions in that case;
        // the reference refuses filling top-k with filtered -inf scores.
        let minimum_positions = u64::from(values.candidate_topk_blocks - 1)
            * u64::from(self.shape.candidates.block)
            + 1;
        if u64::from(values.index_topk) > minimum_positions {
            return Err(format!(
                "unsupported runtime text_config.candidate_topk_blocks: {} blocks of {} must retain at least text_config.index_topk={} positions when the newest block holds one position",
                values.candidate_topk_blocks, self.shape.candidates.block, values.index_topk
            ));
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
        let mut options: Vec<_> = self
            .runtime_fields()
            .into_iter()
            .filter(|field| field.overridable)
            .map(|field| {
                let default = match field.name {
                    "text_config.index_topk" => checkpoint.index_topk,
                    "text_config.candidate_topk_blocks" => checkpoint.candidate_topk_blocks,
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
            .chain(crate::prefill_profile::declarations())
            .collect();
        options.push(ModelOptionV1 {
            name: "text_config.indexer_mma".into(),
            definition: OptionDefinitionV1::Boolean {
                default: checkpoint.indexer_mma,
            },
            default_is_qualified: true,
        });
        options.push(ModelOptionV1 {
            name: "text_config.indexer_prefill_tile".into(),
            definition: OptionDefinitionV1::Boolean { default: false },
            default_is_qualified: true,
        });
        options.sort_by(|a, b| a.name.cmp(&b.name));
        options
    }

    /// Backbone layers, without the DSpark stages.
    pub const fn layers(&self) -> u32 {
        self.shape.layers
    }
    pub const fn dspark_stages(&self) -> u32 {
        self.shape.dspark.stages
    }
    /// The compress ratio of backbone layer or DSpark stage `layer`
    /// (stages follow the backbone).
    pub const fn compress_ratio(&self, layer: u32) -> u32 {
        self.shape.compress_ratios[layer as usize]
    }
    pub const fn layer_role(&self, layer: u32) -> LayerRole {
        if self.compress_ratio(layer) == 0 {
            LayerRole::Window
        } else if contains(self.shape.csa.kv_sources, layer) {
            LayerRole::Full
        } else if contains(self.shape.csa.index_sources, layer) {
            LayerRole::Reindex
        } else {
            LayerRole::Reuse
        }
    }
    pub const fn is_engram_layer(&self, layer: u32) -> bool {
        contains(self.shape.engram.layers, layer)
    }
    pub const fn engram_layers(&self) -> u32 {
        count(self.shape.engram.layers.len())
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
    /// Latent channels without RoPE.
    pub const fn nope_dim(&self) -> u32 {
        self.shape.attention.head_dim - self.shape.attention.rope_dim
    }
    /// Input width of each output group's `wo_a` block.
    pub const fn o_group_width(&self) -> u32 {
        self.shape.attention.heads * self.shape.attention.head_dim / self.shape.attention.o_groups
    }
    pub const fn context_tokens(&self) -> u64 {
        self.shape.context as u64
    }
}

include!(concat!(env!("OUT_DIR"), "/geometries.rs"));

#[cfg(test)]
#[path = "geometry_catalog.rs"]
mod catalog;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_rejects_geometries_with_the_same_admission_key() {
        let flash: serde_json::Value = serde_json::from_str(FLASH.document).unwrap();
        let existing = [("flash", flash.clone())];
        for (top_k, blocks) in [(256, 2048), (1024, 2048), (512, 1024), (256, 1024)] {
            let mut collision = flash.clone();
            collision["text_config"]["index_topk"] = top_k.into();
            collision["text_config"]["candidate_topk_blocks"] = blocks.into();
            assert_eq!(
                catalog::check_unique("flash_collision", &collision, &existing, COMPILE_FIELDS),
                Err("duplicate geometry admission key flash_collision: matches flash".into())
            );
        }
        // Match admission's numerical equality, not JSON number spelling.
        let mut collision = flash.clone();
        collision["text_config"]["rope_theta"] = 10000.into();
        assert!(
            catalog::check_unique("same_numbers", &collision, &existing, COMPILE_FIELDS).is_err()
        );
        assert!(catalog::check_unique("flash", &flash, &[], COMPILE_FIELDS).is_ok());
        for path in ["text_config.hidden_size", "dtype"] {
            let mut distinct = flash.clone();
            if path == "dtype" {
                distinct["dtype"] = "float32".into();
            } else {
                distinct["text_config"]["hidden_size"] = 4096.into();
            }
            assert!(
                catalog::check_unique("distinct", &distinct, &existing, COMPILE_FIELDS).is_ok(),
                "{path}"
            );
        }
    }

    #[test]
    fn candidate_pool_bounds_include_the_partial_newest_block() {
        use std::collections::BTreeMap;
        let checkpoint = FLASH.runtime_defaults();
        let resolve = |top_k: u32, blocks: u32| {
            FLASH.resolve_model_options(
                checkpoint,
                &BTreeMap::from([
                    ("text_config.index_topk".into(), top_k.into()),
                    ("text_config.candidate_topk_blocks".into(), blocks.into()),
                ]),
            )
        };
        for (top_k, blocks) in [(1, 1), (9, 2), (256, 33), (512, 65), (512, 2048)] {
            let values = resolve(top_k, blocks).unwrap();
            assert_eq!(values.candidate_topk_blocks, blocks);
            let options = FLASH.model_options(values);
            assert_eq!(options[2].name, "text_config.candidate_topk_blocks");
            assert_eq!(options[2].default_is_qualified, blocks == 2048);
        }
        for (top_k, blocks) in [(2, 1), (10, 2), (256, 32), (512, 64)] {
            let error = resolve(top_k, blocks).unwrap_err();
            assert!(
                error.contains("when the newest block holds one position"),
                "{error}"
            );
        }
        for value in [
            serde_json::json!(0),
            serde_json::json!(2049),
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!(true),
        ] {
            let error = FLASH
                .resolve_model_options(
                    checkpoint,
                    &BTreeMap::from([("text_config.candidate_topk_blocks".into(), value.clone())]),
                )
                .unwrap_err();
            assert_eq!(
                error,
                format!(
                    "unsupported runtime text_config.candidate_topk_blocks: expected 1..=2048, received {value}"
                )
            );
        }
    }

    #[test]
    fn indexer_mma_defaults_on_and_simt_remains_selectable() {
        use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1, OptionDefinitionV1};
        use std::collections::BTreeMap;
        let defaults = FLASH.runtime_defaults();
        let options = FLASH.model_options(defaults);
        let option = options
            .iter()
            .find(|o| o.name == "text_config.indexer_mma")
            .unwrap();
        assert_eq!(
            option.definition,
            OptionDefinitionV1::Boolean { default: true }
        );
        assert!(option.default_is_qualified);
        let declaration = ModelOptionsV1 {
            schema: CAPABILITY.into(),
            options,
        };
        let resolved = declaration.resolve(&BTreeMap::new()).unwrap();
        assert_eq!(resolved["text_config.indexer_mma"], serde_json::json!(true));
        assert_eq!(
            FLASH.resolve_model_options(defaults, &resolved).unwrap(),
            defaults
        );
        let simt = declaration
            .resolve(&BTreeMap::from([(
                "text_config.indexer_mma".into(),
                "false".into(),
            )]))
            .unwrap();
        assert!(!declaration.is_qualified(&simt));
        assert_eq!(
            FLASH.resolve_model_options(defaults, &simt).unwrap(),
            RuntimeValues {
                indexer_mma: false,
                ..defaults
            }
        );
        for value in [
            serde_json::json!(0),
            serde_json::json!("false"),
            serde_json::Value::Null,
        ] {
            assert!(
                FLASH
                    .resolve_model_options(
                        defaults,
                        &BTreeMap::from([("text_config.indexer_mma".into(), value)])
                    )
                    .unwrap_err()
                    .contains("requires a boolean")
            );
        }
    }

    #[test]
    fn prefill_tile_requires_explicit_unqualified_native_opt_in() {
        use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1};
        use std::collections::BTreeMap;
        let defaults = FLASH.runtime_defaults();
        assert!(!defaults.indexer_prefill_tile);
        let declaration = ModelOptionsV1 {
            schema: CAPABILITY.into(),
            options: FLASH.model_options(defaults),
        };
        let resolved = declaration
            .resolve(&BTreeMap::from([(
                "text_config.indexer_prefill_tile".into(),
                "true".into(),
            )]))
            .unwrap();
        assert!(!declaration.is_qualified(&resolved));
        assert!(
            FLASH
                .resolve_model_options(defaults, &resolved)
                .unwrap()
                .indexer_prefill_tile
        );
        let mut invalid = resolved;
        invalid.insert("text_config.indexer_mma".into(), false.into());
        assert!(FLASH.resolve_model_options(defaults, &invalid).is_err());
        invalid.insert("text_config.indexer_mma".into(), true.into());
        for value in [
            serde_json::json!(1),
            serde_json::json!("true"),
            serde_json::Value::Null,
        ] {
            invalid.insert("text_config.indexer_prefill_tile".into(), value);
            assert!(FLASH.resolve_model_options(defaults, &invalid).is_err());
        }
    }

    #[test]
    fn index_topk_options_are_bounded_by_the_compiled_selection() {
        use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1, OptionDefinitionV1};
        use std::collections::BTreeMap;
        let mut config: serde_json::Value = serde_json::from_str(FLASH.document).unwrap();
        let checkpoint = FLASH.check_runtime(&config).unwrap();
        assert_eq!(checkpoint.index_topk, 512);
        let options = FLASH.model_options(checkpoint);
        assert_eq!(options.len(), 6);
        assert_eq!(options[3].name, "text_config.index_topk");
        assert_eq!(options[4].name, "text_config.indexer_mma");
        assert_eq!(
            options[3].definition,
            OptionDefinitionV1::Integer {
                default: 512,
                minimum: 1,
                maximum: 512
            }
        );
        assert!(options[2].default_is_qualified);
        assert_eq!(
            FLASH
                .resolve_model_options(checkpoint, &BTreeMap::new())
                .unwrap(),
            checkpoint
        );
        for top_k in [1, 255, 256, 511, 512] {
            let values = BTreeMap::from([("text_config.index_topk".to_owned(), top_k.into())]);
            assert_eq!(
                FLASH
                    .resolve_model_options(checkpoint, &values)
                    .unwrap()
                    .index_topk,
                top_k
            );
            config["text_config"]["index_topk"] = top_k.into();
            let runtime = FLASH.check_runtime(&config).unwrap();
            assert_eq!(
                FLASH.model_options(runtime)[3].default_is_qualified,
                top_k == 512
            );
        }
        for value in [
            serde_json::json!(0),
            serde_json::json!(513),
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!(true),
        ] {
            let error = FLASH
                .resolve_model_options(
                    checkpoint,
                    &BTreeMap::from([("text_config.index_topk".to_owned(), value.clone())]),
                )
                .unwrap_err();
            assert_eq!(
                error,
                format!(
                    "unsupported runtime text_config.index_topk: expected 1..=512, received {value}"
                )
            );
        }
        let declaration = ModelOptionsV1 {
            schema: CAPABILITY.into(),
            options,
        };
        let defaults = declaration.resolve(&BTreeMap::new()).unwrap();
        assert!(declaration.is_qualified(&defaults));
        let changed = declaration
            .resolve(&BTreeMap::from([(
                "text_config.index_topk".into(),
                "256".into(),
            )]))
            .unwrap();
        assert!(!declaration.is_qualified(&changed));
        assert!(
            FLASH
                .resolve_model_options(checkpoint, &BTreeMap::from([("unknown".into(), 1.into())]))
                .is_err()
        );
    }

    #[test]
    fn flash_layer_schedule() {
        assert_eq!(GEOMETRIES[0], FLASH);
        const { assert!(FLASH.base) };
        assert_eq!(FLASH.kernel_tag, "deepseek41");
        assert_eq!(FLASH.layers(), 40);
        assert_eq!(FLASH.dspark_stages(), 3);
        for layer in 0..43 {
            let expected = match layer {
                0 | 1 | 40.. => LayerRole::Window,
                2 | 8 | 14 | 20 => LayerRole::Full,
                24 | 28 | 32 | 36 => LayerRole::Reindex,
                _ => LayerRole::Reuse,
            };
            assert_eq!(FLASH.layer_role(layer), expected, "layer {layer}");
            let ratio = match layer {
                2..=19 => 2,
                20..=39 => 1,
                _ => 0,
            };
            assert_eq!(FLASH.compress_ratio(layer), ratio, "layer {layer}");
        }
        assert_eq!(FLASH.shape.candidates.source, 20);
        assert!(FLASH.is_engram_layer(1) && FLASH.is_engram_layer(14));
        assert_eq!(FLASH.engram_layers(), 2);
        assert_eq!(FLASH.shape.engram.rows, &[384_006_168, 384_016_682]);
        assert_eq!(FLASH.hc_width(), 20480);
        assert_eq!(FLASH.hc_mix(), 24);
        assert_eq!(FLASH.nope_dim(), 448);
        assert_eq!(FLASH.o_group_width(), 4096);
        assert_eq!(FLASH.context_tokens(), 1 << 20);
        assert_eq!(FLASH.shape.eps, 1e-20);
        assert_eq!(FLASH.shape.hc.eps, 1e-6);
        assert_eq!(FLASH.shape.moe.routed_scale, 1.5);
        assert_eq!(FLASH.shape.rope.compress_theta, 160000.0);
        assert_eq!(FLASH.shape.dspark.targets, &[37, 38, 39]);
        assert_eq!(FLASH.shape.tokens.image, 129264);
        // A geometry recipe appends its own alias after the official one.
        assert_eq!(ARCHITECTURE_ALIASES[0], "DeepseekV41ForCausalLM");
    }

    #[test]
    fn the_engram_layout_reproduces_the_checkpoint_rows() {
        let s = FLASH.shape.engram;
        let rows = crate::structure::engram_rows(
            s.layers.len(),
            s.vocab.into(),
            s.max_ngram.into(),
            s.heads.into(),
        );
        assert_eq!(
            rows,
            s.rows.iter().map(|&r| u64::from(r)).collect::<Vec<_>>()
        );
    }
}
