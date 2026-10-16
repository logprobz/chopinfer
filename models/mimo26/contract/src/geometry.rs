//! Typed execution geometry generated from supported configuration data.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Global,
    Sliding,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub base: bool,
    pub pinned_names: bool,
    pub encoders: bool,
    pub id: &'static str,
    pub kernel_tag: &'static str,
    pub shape: Shape,
    pub semantics: Semantics,
    pub tokens: TokenIds,
    pub limits: RuntimeLimits,
    pub document: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedForwardKind {
    Dense,
    Moe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layer {
    pub attention: LayerKind,
    pub feed_forward: FeedForwardKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionLayerKind {
    Full,
    Window,
    MergeGroupWindow,
}

// Accessor views preserve the existing result types without storing duplicate dimensions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionConfig {
    pub depth: u32,
    pub hidden_size: u32,
    pub in_chans: u32,
    pub intermediate_size: u32,
    pub num_heads: u32,
    pub num_key_value_heads: u32,
    pub num_query_groups: u32,
    pub out_hidden_size: u32,
    pub patch_size: u32,
    pub spatial_merge_size: u32,
    pub spatial_patch_size: u32,
    pub temporal_patch_size: u32,
    pub tokens_per_second: u32,
    pub visual_token_window_size: u32,
    pub vit_window_attn_types: &'static [VisionLayerKind],
    pub window_size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioConfig {
    pub audio_channels: u32,
    pub audio_segment_size: u32,
    pub group_size: u32,
    pub input_local_attn_heads: u32,
    pub input_local_dim: u32,
    pub input_local_head_dim: u32,
    pub input_local_intermediate_size: u32,
    pub input_local_layers: u32,
    pub out_hidden_size: u32,
    pub projection_layers: u32,
    pub rope_theta: f64,
    pub speech_vocab_size: u32,
    pub speech_zeroemb_idx: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioTokenizerConfig {
    pub avg_pooler: u32,
    pub codebook_size: &'static [u32],
    pub d_model: u32,
    pub encoder_attention_heads: u32,
    pub encoder_attn_window_size: &'static [u32],
    pub encoder_ffn_dim: u32,
    pub encoder_layers: u32,
    pub encoder_skip_layer_id: u32,
    pub fmax: Option<u32>,
    pub fmin: u32,
    pub hop_length: u32,
    pub hybrid_block_size: u32,
    pub kernel_size: u32,
    pub max_audio_seconds: u32,
    pub n_mels: u32,
    pub nfft: u32,
    pub num_quantizers: u32,
    pub rope_theta: f64,
    pub sampling_rate: u32,
    pub stride_size: u32,
    pub swa_per_block: u32,
    pub window_size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DflashConfig {
    pub target_layer_ids: &'static [u32],
    pub mask_token_id: u32,
    pub attention_value_scale: f64,
    pub attention_sink_bias: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dflash {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub v_head_dim: u32,
    pub block_size: u32,
    pub dflash_config: DflashConfig,
    pub layer_types: &'static [LayerKind],
    pub sliding_window: u32,
    pub is_causal: bool,
    pub target_hidden_size: u32,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub rotary_dim: u32,
}

const fn count(length: usize) -> u32 {
    assert!(length <= u32::MAX as usize, "geometry list exceeds u32");
    length as u32
}

impl Geometry {
    pub const fn sliding_layers(&self) -> usize {
        self.layers() as usize - self.global_layers()
    }
    /// Number of global layers before this target layer; absent layers return zero.
    pub const fn global_ordinal(&self, layer: u32) -> usize {
        if layer >= self.layers() {
            return 0;
        }
        let mut ordinal = 0;
        let mut i = 0;
        while i < layer {
            if self.is_global(i) {
                ordinal += 1;
            }
            i += 1;
        }
        ordinal
    }
    pub const fn swa_head_dim(&self) -> u32 {
        self.shape.sliding.head_dim
    }
    pub const fn swa_value_head_dim(&self) -> u32 {
        self.shape.sliding.value_dim
    }
    pub const fn tap_count(&self) -> usize {
        self.shape.dflash.target_layer_ids.len()
    }
    pub const fn global_layers(&self) -> usize {
        let mut count = 0;
        let mut layer = 0;
        while layer < self.layers() {
            if self.is_global(layer) {
                count += 1;
            }
            layer += 1;
        }
        count
    }
    pub const fn global_layer_ids<const N: usize>(&self) -> [u32; N] {
        assert!(N == self.global_layers());
        let mut out = [0; N];
        let mut layer = 0;
        let mut index = 0;
        while layer < self.layers() {
            if self.is_global(layer) {
                out[index] = layer;
                index += 1;
            }
            layer += 1;
        }
        out
    }
    pub const fn dense_layer_end(&self) -> u32 {
        let mut layer = 0;
        while layer < self.layers() && !self.is_moe(layer) {
            layer += 1;
        }
        let end = layer;
        while layer < self.layers() {
            assert!(self.is_moe(layer));
            layer += 1;
        }
        end
    }
    pub const fn is_moe(&self, layer: u32) -> bool {
        layer < self.layers()
            && matches!(
                self.shape.text.layers[layer as usize].feed_forward,
                FeedForwardKind::Moe
            )
    }
    pub const fn dflash(&self) -> Dflash {
        Dflash {
            hidden_size: self.shape.text.hidden,
            intermediate_size: self.shape.dflash.intermediate_size,
            num_hidden_layers: count(self.shape.dflash.layers.len()),
            num_attention_heads: self.shape.dflash.num_attention_heads,
            num_key_value_heads: self.shape.dflash.num_key_value_heads,
            head_dim: self.shape.dflash.head_dim,
            v_head_dim: self.shape.dflash.v_head_dim,
            block_size: self.shape.dflash.block_size,
            dflash_config: DflashConfig {
                target_layer_ids: self.shape.dflash.target_layer_ids,
                mask_token_id: self.tokens.dflash_mask,
                attention_value_scale: self.semantics.dflash.attention_value_scale,
                attention_sink_bias: self.semantics.dflash.attention_sink_bias,
            },
            layer_types: self.shape.dflash.layers,
            sliding_window: self.shape.dflash.sliding_window,
            is_causal: self.semantics.dflash.is_causal,
            target_hidden_size: self.shape.text.hidden,
            rope_theta: self.shape.dflash.rope_theta,
            rms_norm_eps: self.semantics.dflash.rms_norm_eps,
            rotary_dim: self.shape.dflash.rotary_dim,
        }
    }
    /// HF's `qk_channels` default; `vision_config` is matched whole, so a
    /// config that sets it is refused.
    pub const fn vision_head_dim(&self) -> u32 {
        64
    }
    pub const fn vision(&self) -> VisionConfig {
        VisionConfig {
            depth: count(self.shape.vision.vit_window_attn_types.len()),
            hidden_size: self.shape.vision.hidden_size,
            in_chans: self.shape.vision.in_chans,
            intermediate_size: self.shape.vision.intermediate_size,
            num_heads: self.shape.vision.num_heads,
            num_key_value_heads: self.shape.vision.num_key_value_heads,
            num_query_groups: self.shape.vision.num_query_groups,
            out_hidden_size: self.hidden(),
            patch_size: self.shape.vision.patch_size,
            spatial_merge_size: self.shape.vision.spatial_merge_size,
            spatial_patch_size: self.shape.vision.spatial_patch_size,
            temporal_patch_size: self.shape.vision.temporal_patch_size,
            tokens_per_second: self.shape.vision.tokens_per_second,
            visual_token_window_size: self.shape.vision.visual_token_window_size,
            vit_window_attn_types: self.shape.vision.vit_window_attn_types,
            window_size: self.shape.vision.window_size,
        }
    }
    pub const fn audio(&self) -> AudioConfig {
        AudioConfig {
            audio_channels: self.shape.audio.audio_channels,
            audio_segment_size: self.shape.audio.audio_segment_size,
            group_size: self.shape.audio.group_size,
            input_local_attn_heads: self.shape.audio.input_local_attn_heads,
            input_local_dim: self.shape.audio.input_local_dim,
            input_local_head_dim: self.shape.audio.input_local_head_dim,
            input_local_intermediate_size: self.shape.audio.input_local_intermediate_size,
            input_local_layers: self.shape.audio.input_local_layers,
            out_hidden_size: self.hidden(),
            projection_layers: self.shape.audio.projection_layers,
            rope_theta: self.shape.audio.rope_theta,
            speech_vocab_size: self.shape.audio.speech_vocab_size,
            speech_zeroemb_idx: self.shape.audio.speech_zeroemb_idx,
        }
    }
    pub const fn audio_tokenizer(&self) -> AudioTokenizerConfig {
        AudioTokenizerConfig {
            avg_pooler: self.shape.audio_tokenizer.avg_pooler,
            codebook_size: self.shape.audio_tokenizer.codebook_size,
            d_model: self.shape.audio_tokenizer.d_model,
            encoder_attention_heads: self.shape.audio_tokenizer.encoder_attention_heads,
            encoder_attn_window_size: self.shape.audio_tokenizer.encoder_attn_window_size,
            encoder_ffn_dim: self.shape.audio_tokenizer.encoder_ffn_dim,
            encoder_layers: self.shape.audio_tokenizer.encoder_layers,
            encoder_skip_layer_id: self.shape.audio_tokenizer.encoder_skip_layer_id,
            fmax: self.shape.audio_tokenizer.fmax,
            fmin: self.shape.audio_tokenizer.fmin,
            hop_length: self.shape.audio_tokenizer.hop_length,
            hybrid_block_size: self.shape.audio_tokenizer.hybrid_block_size,
            kernel_size: self.shape.audio_tokenizer.kernel_size,
            max_audio_seconds: self.limits.max_audio_seconds,
            n_mels: self.shape.audio_tokenizer.n_mels,
            nfft: self.shape.audio_tokenizer.nfft,
            num_quantizers: self.shape.audio_tokenizer.num_quantizers,
            rope_theta: self.shape.audio_tokenizer.rope_theta,
            sampling_rate: self.shape.audio_tokenizer.sampling_rate,
            stride_size: self.shape.audio_tokenizer.stride_size,
            swa_per_block: self.shape.audio_tokenizer.swa_per_block,
            window_size: self.shape.audio_tokenizer.window_size,
        }
    }
    pub const fn hidden(&self) -> u32 {
        self.shape.text.hidden
    }
    pub const fn vocab(&self) -> u32 {
        self.shape.text.vocab
    }
    pub const fn layers(&self) -> u32 {
        count(self.shape.text.layers.len())
    }
    pub const fn rotary(&self) -> u32 {
        self.shape.global.rotary
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
    pub const fn dense_intermediate(&self) -> u32 {
        self.shape.text.dense_intermediate
    }
    pub const fn attention_q_heads(&self) -> u32 {
        self.shape.global.heads
    }
    pub const fn attention_kv_heads(&self) -> u32 {
        self.shape.global.kv_heads
    }
    pub const fn swa_q_heads(&self) -> u32 {
        self.shape.sliding.heads
    }
    pub const fn swa_kv_heads(&self) -> u32 {
        self.shape.sliding.kv_heads
    }
    pub const fn head_dim(&self) -> u32 {
        self.shape.global.head_dim
    }
    pub const fn value_head_dim(&self) -> u32 {
        self.shape.global.value_dim
    }
    pub const fn context_tokens(&self) -> u32 {
        self.limits.context
    }
    pub const fn window(&self) -> u32 {
        self.shape.sliding.window
    }
    pub const fn eps(&self) -> f64 {
        self.semantics.eps
    }
    pub const fn rope_theta(&self) -> f64 {
        self.shape.global.theta
    }
    pub const fn swa_rope_theta(&self) -> f64 {
        self.shape.sliding.theta
    }
    pub const fn mtp_layers(&self) -> u32 {
        self.shape.text.mtp_layers
    }
    /// MTP and absent layers use sliding attention.
    pub const fn is_global(&self, layer: u32) -> bool {
        layer < self.layers()
            && matches!(
                self.shape.text.layers[layer as usize].attention,
                LayerKind::Global
            )
    }
}
include!(concat!(env!("OUT_DIR"), "/geometries.rs"));

/// Match authenticated execution config bytes after metadata has been read.
pub fn match_config(config: &serde_json::Value) -> Result<&'static Geometry, String> {
    match_component(config, crate::geometry_checks::Component::All)
}

fn match_component(
    config: &serde_json::Value,
    component: crate::geometry_checks::Component,
) -> Result<&'static Geometry, String> {
    use crate::geometry_checks::Component;
    let ranks: Vec<_> = crate::Tp::SUPPORTED.iter().map(|tp| tp.ranks()).collect();
    let paths: Vec<_> = COMPILE_FIELDS
        .iter()
        .copied()
        .filter(|path| match component {
            Component::All => true,
            Component::Main => !path.starts_with("dflash.") && *path != "audio_tokenizer_config",
            Component::Dflash => path.starts_with("dflash."),
        })
        .collect();
    let geometries: Vec<_> = GEOMETRIES
        .iter()
        .map(|g| {
            (
                g.id,
                serde_json::from_str::<serde_json::Value>(g.document).unwrap(),
            )
        })
        .collect();
    // Report semantic, structural and engine refusals before the closed comparison.
    let mut valid = Vec::new();
    let mut refusals = Vec::new();
    for (index, rules) in GEOMETRY_RULES.iter().enumerate() {
        let rules: serde_json::Value = serde_json::from_str(rules).unwrap();
        match crate::geometry_checks::validate(config, &rules, &ranks, component) {
            Ok(()) => {
                valid.push(index);
                refusals.push(None);
            }
            Err(error) => refusals.push(Some(error)),
        }
    }
    if valid.is_empty() {
        return Err(refusals.into_iter().flatten().next().unwrap());
    }
    if component != Component::Dflash {
        model_compiler_sdk::quantization::import::ExportMetadata::from_documents(
            config.get("quantization_config").filter(|q| !q.is_null()), None,
        ).map_err(|error| format!("unsupported quantization_config: expected supported export recipe, received {}: {error}", config["quantization_config"]))?;
    }
    let index = match_documents(config, &geometries, &paths)?;
    if !valid.contains(&index) {
        return Err(refusals[index].clone().unwrap());
    }
    Ok(&GEOMETRIES[index])
}

fn match_documents(
    config: &serde_json::Value,
    geometries: &[(&str, serde_json::Value)],
    paths: &[&str],
) -> Result<usize, String> {
    use model_compiler_sdk::geometry::{field, match_geometry};
    let mut differences = Vec::new();
    for (index, (id, document)) in geometries.iter().enumerate() {
        let candidate = [(*id, document.clone())];
        let Some(path) = paths
            .iter()
            .find(|path| match_geometry(config, &candidate, &[path]).is_err())
        else {
            return Ok(index);
        };
        let mut name = (*path).to_owned();
        let mut actual = field(config, path);
        let mut expected = field(document, path);
        if ["hybrid_layer_pattern", "moe_layer_freq"].contains(path)
            && let (Some(a), Some(b)) = (
                actual.and_then(serde_json::Value::as_array),
                expected.and_then(serde_json::Value::as_array),
            )
            && a.len() == b.len()
            && let Some(i) = a.iter().zip(b).position(|(a, b)| a != b)
        {
            name = format!("{path}[{i}]");
            actual = Some(&a[i]);
            expected = Some(&b[i]);
        }
        let display = |value: Option<&serde_json::Value>| {
            value.map_or_else(|| "absent".into(), serde_json::Value::to_string)
        };
        differences.push(format!(
            "{id}: {name} expected {}, received {}",
            display(expected),
            display(actual)
        ));
    }
    Err(format!(
        "unsupported geometry: {}; supported ids: {}",
        differences.join("; "),
        geometries
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Match only target fields, including whole encoder objects.
pub fn match_main(value: &serde_json::Value) -> Result<&'static Geometry, String> {
    match_component(value, crate::geometry_checks::Component::Main)
}

/// Match only drafter fields. Cross-component rules are checked during generation.
pub fn match_dflash(value: &serde_json::Value) -> Result<&'static Geometry, String> {
    match_component(
        &serde_json::json!({"dflash": value}),
        crate::geometry_checks::Component::Dflash,
    )
}

/// Validate the drafter against the already selected target geometry.
/// Several geometries may share exactly the same drafter fields.
pub fn match_dflash_for(
    geometry: &'static Geometry,
    value: &serde_json::Value,
) -> Result<&'static Geometry, String> {
    let index = GEOMETRIES
        .iter()
        .position(|g| g.id == geometry.id)
        .or_else(|| {
            GEOMETRIES
                .iter()
                .position(|g| g.document == geometry.document)
        })
        .ok_or_else(|| format!("unlisted target geometry {}", geometry.id))?;
    let rules = serde_json::from_str(GEOMETRY_RULES[index]).unwrap();
    let ranks: Vec<_> = crate::Tp::SUPPORTED.iter().map(|tp| tp.ranks()).collect();
    crate::geometry_checks::validate(
        &serde_json::json!({"dflash": value}),
        &rules,
        &ranks,
        crate::geometry_checks::Component::Dflash,
    )?;
    let document = serde_json::from_str(geometry.document).unwrap();
    let paths: Vec<_> = COMPILE_FIELDS
        .iter()
        .copied()
        .filter(|path| path.starts_with("dflash."))
        .collect();
    match_documents(
        &serde_json::json!({"dflash": value}),
        &[(geometry.id, document)],
        &paths,
    )?;
    Ok(geometry)
}

/// Target and drafter must select the same execution geometry.
pub fn require_same(target: &Geometry, draft: &Geometry) -> Result<(), String> {
    if target.id != draft.id {
        return Err(format!(
            "unsupported drafter geometry: expected {}, received {}",
            target.id, draft.id
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const ROTARY: u32 = V26_FLASH.rotary();
    const GLOBAL: bool = V26_FLASH.is_global(5);
    #[test]
    fn shared_drafter_fields_match_the_selected_target() {
        let mut geometry = V26_FLASH;
        geometry.id = "shared-drafter-test";
        let geometry = Box::leak(Box::new(geometry));
        let config: serde_json::Value = serde_json::from_str(geometry.document).unwrap();
        assert_eq!(
            match_dflash_for(geometry, &config["dflash"]).unwrap().id,
            geometry.id
        );
    }

    #[test]
    fn one_difference_per_geometry_and_first_pattern_index() {
        let geometries = [
            ("first", serde_json::json!({"hybrid_layer_pattern":[0,1]})),
            ("second", serde_json::json!({"hybrid_layer_pattern":[1,0]})),
        ];
        assert_eq!(
            match_documents(
                &serde_json::json!({"hybrid_layer_pattern":[1,1]}),
                &geometries,
                &["hybrid_layer_pattern"]
            )
            .unwrap_err(),
            "unsupported geometry: first: hybrid_layer_pattern[0] expected 0, received 1; second: hybrid_layer_pattern[1] expected 0, received 1; supported ids: first, second"
        );
    }

    #[test]
    fn validate_every_listed_geometry() {
        for g in GEOMETRIES {
            let value = serde_json::from_str(g.document).unwrap();
            assert_eq!(match_config(&value).unwrap().id, g.id);
        }
        assert_eq!(ROTARY, 64);
        let value: serde_json::Value = serde_json::from_str(V26_FLASH.document).unwrap();
        assert_eq!(GLOBAL, value["hybrid_layer_pattern"][5] == 0);
        assert_eq!(V26_FLASH.dflash().rotary_dim, 64);
        assert_eq!(V26_FLASH.kernel_tag, "mimo");
    }
    #[test]
    fn main_requires_target_fields_and_preserves_encoder_pins() {
        let mut config: serde_json::Value = serde_json::from_str(V26_FLASH.document).unwrap();
        config.as_object_mut().unwrap().remove("dflash");
        config
            .as_object_mut()
            .unwrap()
            .remove("audio_tokenizer_config");
        assert_eq!(match_main(&config).unwrap().id, "v26-flash");
        config["vision_config"]["extra"] = true.into();
        assert!(match_main(&config).unwrap_err().contains("vision_config"));
        config["vision_config"]
            .as_object_mut()
            .unwrap()
            .remove("extra");
        config
            .as_object_mut()
            .unwrap()
            .remove("num_nextn_predict_layers");
        assert!(
            match_main(&config)
                .unwrap_err()
                .contains("num_nextn_predict_layers")
        );
    }

    #[test]
    fn refuse_semantic_structural_and_engine_drift() {
        let original: serde_json::Value = serde_json::from_str(V26_FLASH.document).unwrap();
        let rules = serde_json::from_str(SUPPORTED).unwrap();
        for (pointer, value) in [
            ("/head_dim", serde_json::json!(128)),
            ("/rope_theta", serde_json::json!(12345)),
            ("/hybrid_layer_pattern/0", serde_json::json!(2)),
            (
                "/rope_parameters/partial_rotary_factor",
                serde_json::json!(0.5),
            ),
            (
                "/dflash/dflash_config/target_layer_ids/0",
                serde_json::json!(48),
            ),
            (
                "/attention_projection_layout",
                serde_json::json!("separate"),
            ),
        ] {
            let mut config = original.clone();
            *config.pointer_mut(pointer).unwrap() = value;
            let error = crate::geometry_checks::validate(
                &config,
                &rules,
                &[2, 4],
                crate::geometry_checks::Component::All,
            )
            .unwrap_err();
            assert!(
                error.contains("expected") && error.contains("received"),
                "{error}"
            );
            assert!(match_config(&config).is_err());
        }
    }
}
