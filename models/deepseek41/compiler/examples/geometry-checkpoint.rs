//! A checkpoint of a compiled geometry with deterministic random weights,
//! for the geometry recipe and the synthetic M0 run: `config.json` as given,
//! the tokenizer files, and every tensor the projection declaration names
//! at the admitted geometry (`model_deepseek41::synthetic`). The written
//! headers then bind to the declaration with no unknown or missing tensor.
//!
//! Usage: geometry-checkpoint CONFIG TOKENIZER_DIR OUTPUT [SEED]
use model_deepseek41::{checkpoint, synthetic};
use model_deepseek41_contract::admission;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: geometry-checkpoint CONFIG TOKENIZER_DIR OUTPUT [SEED]".into());
    }
    let seed = match args.get(3) {
        Some(seed) => seed
            .to_str()
            .and_then(|s| s.parse().ok())
            .ok_or("SEED is an unsigned integer")?,
        None => 0,
    };
    let config_bytes = fs::read(&args[0]).map_err(|e| e.to_string())?;
    let config = serde_json::from_slice(&config_bytes).map_err(|e| e.to_string())?;
    let geometry = admission::admit(&config)?;
    let output = PathBuf::from(&args[2]);
    fs::create_dir(&output).map_err(|e| format!("{}: {e}", output.display()))?;
    fs::write(output.join("config.json"), &config_bytes).map_err(|e| e.to_string())?;
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        fs::copy(PathBuf::from(&args[1]).join(name), output.join(name))
            .map_err(|e| format!("{name}: {e}"))?;
    }
    let tensors = synthetic::tensors(geometry)?;
    let bytes = synthetic::write(&output, &tensors, seed)?;
    let bound = checkpoint::bind(&output, synthetic::external_names(geometry)?)?;
    println!(
        "geometry {}: {} tensors, {bytes} payload bytes, seed {seed}; {} operands and {} side tensors bound",
        geometry.id,
        tensors.len(),
        bound.operands.len(),
        bound.side_tensors.len()
    );
    Ok(())
}
