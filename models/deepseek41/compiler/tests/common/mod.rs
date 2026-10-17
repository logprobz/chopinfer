//! Helpers shared by the tokenizer tests.
use std::path::PathBuf;

/// The checkpoint (or tokenizer) directory, or `None` with a notice.
pub fn checkpoint() -> Option<PathBuf> {
    let dir = std::env::var_os("CHOPINFER_DEEPSEEK_TOKENIZER_DIR")
        .or_else(|| std::env::var_os("CHOPINFER_DEEPSEEK_CHECKPOINT"))
        .map(PathBuf::from);
    if dir.is_none() {
        eprintln!(
            "CHOPINFER_DEEPSEEK_TOKENIZER_DIR and CHOPINFER_DEEPSEEK_CHECKPOINT are unset: skipped"
        );
    }
    dir
}

/// Every added token of the checkpoint's tokenizer, which HF's
/// `encode(add_special_tokens=False)` recognizes in text.
pub fn added_tokens(dir: &std::path::Path) -> Vec<String> {
    let definition: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("tokenizer.json")).unwrap()).unwrap();
    definition["added_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|token| token["content"].as_str().unwrap().to_owned())
        .collect()
}
