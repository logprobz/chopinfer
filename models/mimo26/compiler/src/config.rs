use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::MimoResult;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MiMoConfig {
    pub model_type: String,
    pub moe_router_dtype: String,
    #[serde(flatten)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architectures: Option<Vec<String>>,
    pub num_nextn_predict_layers: u32,
    pub sliding_window_size: u32,
    pub rope_parameters: serde_json::Value,
    pub dtype: String,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub attention_projection_layout: String,
    pub attention_chunk_size: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub swa_num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub swa_num_key_value_heads: usize,
    pub head_dim: usize,
    pub swa_head_dim: usize,
    pub v_head_dim: usize,
    pub swa_v_head_dim: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: Option<usize>,
    pub num_experts_per_tok: usize,
    pub hybrid_layer_pattern: Vec<u8>,
    pub moe_layer_freq: Vec<u8>,
    pub layernorm_epsilon: f64,
    pub max_position_embeddings: usize,
    pub eos_token_id: usize,
    pub sliding_window: usize,
    pub rope_theta: f64,
    pub swa_rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub attention_value_scale: f64,
    pub add_full_attention_sink_bias: bool,
    pub add_swa_attention_sink_bias: bool,
    pub tie_word_embeddings: bool,
    pub scoring_func: String,
    pub topk_method: String,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: Option<f64>,
    /// The export's precision metadata, interpreted by the shared importer
    /// (`crate::precision`).
    pub quantization_config: Option<serde_json::Value>,
    pub vision_config: serde_json::Value,
    pub audio_config: serde_json::Value,
    pub processor_config: serde_json::Value,
}

impl MiMoConfig {
    pub fn load(root: &Path) -> MimoResult<Self> {
        let config: Self = serde_json::from_value(model_compiler_sdk::metadata::read(
            root,
            "config.json",
            1024 * 1024,
        )?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> MimoResult<()> {
        self.geometry()?;
        Ok(())
    }

    pub fn geometry(&self) -> MimoResult<&'static model_mimo26_contract::geometry::Geometry> {
        Ok(model_mimo26_contract::geometry::match_main(
            &serde_json::to_value(self)?,
        )?)
    }

    pub fn is_global(&self, layer: usize) -> bool {
        self.hybrid_layer_pattern[layer] == 0
    }
}

#[cfg(test)]
pub(crate) fn fixture_config() -> MiMoConfig {
    MiMoConfig::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
        .expect("portable MiMo config fixture")
}

/// The engine domain against the SM120 kernels' shared specializations.
#[cfg(test)]
mod sm120;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_returns_the_geometry_it_can_serve() {
        let config = fixture_config();
        let geometry = config.geometry().unwrap();
        assert_eq!(geometry.id, "v26-flash");
        assert_eq!(config.vocab_size, geometry.vocab() as usize);
        assert_eq!(config.hidden_size, geometry.hidden() as usize);
    }

    #[test]
    fn portable_config_admits_the_text_and_encoder_profile() {
        let config = fixture_config();
        assert_eq!((0..48).filter(|&layer| config.is_global(layer)).count(), 9);
        assert_eq!(
            config
                .moe_layer_freq
                .iter()
                .filter(|&&moe| moe == 1)
                .count(),
            47
        );
    }

    #[test]
    fn portable_config_rejects_geometry_and_layer_drift() {
        let original: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/config.json")).unwrap();
        for (pointer, value) in [
            ("/hidden_size", serde_json::json!(2048)),
            ("/num_key_value_heads", serde_json::json!(8)),
            ("/vocab_size", serde_json::json!(152577)),
            ("/hybrid_layer_pattern/5", serde_json::json!(1)),
            ("/moe_layer_freq/0", serde_json::json!(1)),
        ] {
            let mut changed = original.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            let config: MiMoConfig = serde_json::from_value(changed).unwrap();
            assert!(config.validate().is_err(), "accepted drift at {pointer}");
        }
    }

    #[test]
    fn refuse_mtp_and_rotary_drift_precisely() {
        let original: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/config.json")).unwrap();
        for (pointer, value) in [
            ("/num_nextn_predict_layers", serde_json::json!(4)),
            ("/rope_parameters/rope_theta", serde_json::json!(12345)),
            ("/sliding_window_size", serde_json::json!(256)),
        ] {
            let mut changed = original.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            let config: MiMoConfig = serde_json::from_value(changed).unwrap();
            let error = config.validate().unwrap_err().to_string();
            assert!(
                error.contains("expected") && error.contains("received"),
                "{error}"
            );
        }
    }

    /// Encoder and processor profiles are part of the matched geometry, so
    /// drift in any of them is refused.
    #[test]
    fn rejects_profile_drift() {
        let original: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/config.json")).unwrap();
        for (pointer, value) in [
            ("/vision_config/depth", serde_json::json!(27)),
            ("/audio_config/input_local_layers", serde_json::json!(5)),
            ("/processor_config/patch_size", serde_json::json!(8)),
        ] {
            let mut changed = original.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            let config: MiMoConfig = serde_json::from_value(changed).unwrap();
            assert!(config.validate().is_err(), "accepted drift at {pointer}");
        }
    }

    #[test]
    fn engine_domain_refuses_unqualified_routing_and_rotary_shapes() {
        use model_mimo26_contract::geometry::{V26_FLASH, match_dflash, match_main};
        let original: serde_json::Value = serde_json::from_str(V26_FLASH.document).unwrap();
        for (field, value, expected) in [
            (
                "n_routed_experts",
                128,
                "unsupported n_routed_experts: expected [256] (specialized fragment routing shape), received 128",
            ),
            (
                "num_experts_per_tok",
                4,
                "unsupported num_experts_per_tok: expected [8] (specialized fragment routing shape), received 4",
            ),
            (
                "hidden_size",
                2048,
                "unsupported hidden_size: expected [4096] (specialized fragment hidden width, fused comm FUSED_EPILOGUE width 4096 and RoutedCombine<8>), received 2048",
            ),
            (
                "moe_intermediate_size",
                4096,
                "unsupported moe_intermediate_size: expected [2048] (specialized fragment intermediate width at every supported rank count), received 4096",
            ),
            (
                "rope_theta",
                12345,
                "unsupported rope_theta: expected [10000,10000000] (qualified shared rotary arithmetic), received 12345",
            ),
            (
                "swa_rope_theta",
                12345,
                "unsupported swa_rope_theta: expected [10000,10000000] (qualified shared rotary arithmetic), received 12345",
            ),
        ] {
            let mut config = original.clone();
            config[field] = serde_json::json!(value);
            if field == "hidden_size" {
                config["vision_config"]["out_hidden_size"] = serde_json::json!(value);
                config["audio_config"]["out_hidden_size"] = serde_json::json!(value);
            }
            if field == "rope_theta" {
                config["rope_parameters"]["rope_theta"] = serde_json::json!(value);
            }
            assert_eq!(match_main(&config).unwrap_err(), expected);
        }
        let mut config = original.clone();
        config["partial_rotary_factor"] = serde_json::json!(0.5);
        config["rope_parameters"]["partial_rotary_factor"] = serde_json::json!(0.5);
        assert_eq!(
            match_main(&config).unwrap_err(),
            "unsupported rotary_dim: expected [64] (folded RoPE needs 64 rotary of 192 K and 128 V dimensions), received 96.0"
        );
        let mut draft = original["dflash"].clone();
        draft["partial_rotary_factor"] = serde_json::json!(0.25);
        assert_eq!(
            match_dflash(&draft).unwrap_err(),
            "unsupported dflash.rotary_dim: expected [64] (qualified shared rotary arithmetic), received 32.0"
        );
        draft = original["dflash"].clone();
        draft["rope_theta"] = serde_json::json!(12345);
        assert_eq!(
            match_dflash(&draft).unwrap_err(),
            "unsupported dflash.rope_theta: expected [10000] (qualified shared rotary arithmetic), received 12345"
        );
        draft["rope_theta"] = serde_json::json!(10000000);
        assert_eq!(
            match_dflash(&draft).unwrap_err(),
            "unsupported dflash.rope_theta: expected [10000] (qualified shared rotary arithmetic), received 10000000"
        );
    }

    #[test]
    fn checkpoint_config() {
        let Ok(root) = std::env::var("CHOPINFER_MIMO_CHECKPOINT") else {
            eprintln!("SKIP: set CHOPINFER_MIMO_CHECKPOINT for MiMo config test");
            return;
        };
        MiMoConfig::load(Path::new(&root)).unwrap();
    }
}
