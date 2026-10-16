//! Checkpoint-pinned multimodal configuration. The full objects are compared,
//! including fields not consumed by the first encoder kernels, so drift fails closed.

use std::io::Read;
use std::path::Path;

use serde_json::Value;

use crate::MimoResult;
use crate::catalog::open_source;
use crate::config::MiMoConfig;

pub fn load_audio_tokenizer(root: &Path) -> MimoResult<Value> {
    let path = root.join("audio_tokenizer/config.json");
    let mut file = open_source(&path)?;
    if file.metadata()?.len() > 1024 * 1024 {
        return Err("audio tokenizer config exceeds 1 MiB".into());
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let actual: Value = serde_json::from_slice(&bytes)?;
    let geometry = MiMoConfig::load(root)?.geometry()?;
    let expected: Value = serde_json::from_str(geometry.document)?;
    if model_compiler_sdk::geometry::match_geometry(
        &serde_json::json!({"audio_tokenizer_config": actual}),
        &[(geometry.id, expected)],
        &["audio_tokenizer_config"],
    )
    .is_err()
    {
        return Err(format!(
            "audio tokenizer config differs from geometry {}",
            geometry.id
        )
        .into());
    }
    Ok(actual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_pins_have_one_execution_source() {
        let legacy: Value = serde_json::from_str(include_str!("checkpoint_config.json")).unwrap();
        let geometry: Value =
            serde_json::from_str(model_mimo26_contract::geometry::V26_FLASH.document).unwrap();
        for (key, value) in legacy.as_object().unwrap() {
            assert_eq!(value, &geometry[key], "encoder pin {key}");
        }
    }

    #[test]
    fn real_audio_tokenizer_config() {
        let Ok(root) = std::env::var("CHOPINFER_MIMO_CHECKPOINT") else {
            return;
        };
        load_audio_tokenizer(Path::new(&root)).unwrap();
    }
}
