//! The frozen Q-0 prompt set (`$CHOPINFER_DATA/qwen38-refs/prompts`).

use std::path::Path;

use serde::Deserialize;

use crate::Result;

#[derive(Clone, Debug, Deserialize)]
pub struct Prompt {
    pub name: String,
    pub token_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct PromptSet {
    teacher: Vec<Prompt>,
    long: Vec<Prompt>,
}

/// The teacher and long prompts of the frozen Q-0 prompt set, in file order.
pub fn load_prompts(path: &Path) -> Result<Vec<Prompt>> {
    let set: PromptSet = serde_json::from_slice(&std::fs::read(path)?)?;
    Ok(set.teacher.into_iter().chain(set.long).collect())
}
