//! What the compiler's integration tests share: a compiled geometry's
//! `config.json` as the official export writes it, and its synthetic
//! checkpoint (`model_deepseek41::synthetic`) written sparse past the
//! header.
#![allow(dead_code)]
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use model_deepseek41::synthetic;
use model_deepseek41_contract::geometry::Geometry;
use serde_json::{Value, json};

pub fn config(geometry: &Geometry) -> Value {
    let mut config: Value = serde_json::from_str(geometry.document).unwrap();
    config["architectures"] = json!(["DeepseekV41ForCausalLM"]);
    config["model_type"] = json!("deepseek_v41");
    config["transformers_version"] = json!("5.6.0");
    config["text_config"]["model_type"] = json!("deepseek_v41_text");
    config["text_config"]["initializer_range"] = json!(0.02);
    config["text_config"]["use_cache"] = json!(true);
    config["text_config"]["attention_dropout"] = json!(0.0);
    config["vision_config"]["model_type"] = json!("deepseek_v41_vision");
    config["quantization_config"] = json!({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4"
    });
    config
}

/// A new directory holding `geometry`'s synthetic checkpoint, its file
/// sparse past the header.
pub fn sparse(geometry: &Geometry) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "deepseek41-sparse-{}-{}-{}",
        geometry.id,
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    let tensors = synthetic::tensors(geometry).unwrap();
    let (header, index, payload) = synthetic::header(&tensors).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    let file = root.join(synthetic::FILE);
    std::fs::write(&file, &bytes).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&file)
        .unwrap()
        .set_len(8 + header.len() as u64 + payload)
        .unwrap();
    std::fs::write(root.join("model.safetensors.index.json"), index).unwrap();
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&config(geometry)).unwrap(),
    )
    .unwrap();
    root
}
