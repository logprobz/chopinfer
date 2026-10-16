//! Model-owned metadata for Qwen3.8 Flash, with no GPU execution.
pub mod config;
mod source;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub const REVISION: &str = "236dfdf285828023ca3bcd3f37366c58a3469b13";
pub mod catalog;
pub mod checkpoint;
pub mod encoders;
pub mod frontend;
pub mod local_frontend;
pub mod oracle;
pub mod package;
pub mod placement;
pub mod precision_packing;
pub mod precision_source;
pub mod preparation;
pub mod projection_program;
pub mod tokenizer;

#[cfg(test)]
mod test_native;

#[cfg(test)]
#[allow(dead_code)]
mod checkpoint_tests;

#[cfg(test)]
#[allow(dead_code)]
mod checkpoint_index;

#[cfg(test)]
mod projection_comparison;
