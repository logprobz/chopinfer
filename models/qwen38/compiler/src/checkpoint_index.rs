use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
pub struct Checkpoint {
    pub name: String,
    pub geometry: String,
    pub checkpoint_env: String,
    pub tokenizer_env: String,
    pub tokenizer_cases: usize,
    pub chat_cases: usize,
}

pub fn checkpoints() -> Vec<Checkpoint> {
    let checkpoints: Vec<Checkpoint> =
        serde_json::from_str(include_str!("../fixtures/checkpoints/index.json")).unwrap();
    assert!(!checkpoints.is_empty());
    checkpoints
}

impl Checkpoint {
    pub fn execution_geometry(&self) -> &'static model_qwen38_contract::geometry::Geometry {
        model_qwen38_contract::geometry::GEOMETRIES
            .iter()
            .find(|geometry| geometry.id == self.geometry)
            .unwrap_or_else(|| panic!("unknown checkpoint geometry: {}", self.geometry))
    }

    pub fn checkpoint_root(&self) -> Option<PathBuf> {
        std::env::var_os(&self.checkpoint_env).map(Into::into)
    }

    pub fn fixture<T: serde::de::DeserializeOwned>(&self, name: &str) -> T {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../compiler/fixtures/checkpoints")
            .join(&self.name);
        serde_json::from_slice(&std::fs::read(root.join(name)).unwrap()).unwrap()
    }

    pub fn root(&self) -> Option<PathBuf> {
        std::env::var_os(&self.tokenizer_env)
            .or_else(|| std::env::var_os(&self.checkpoint_env))
            .map(Into::into)
    }
}
