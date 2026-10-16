#![forbid(unsafe_code)]

pub mod catalog;
pub mod checkpoint;
pub mod checkpoint_provenance;
pub mod config;
pub mod encoders;
pub mod frontend;
pub mod oracle;
pub mod precision;
pub mod rank_bytes;
pub mod tokenizer;

pub type MimoResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[cfg(test)]
mod projection_comparison;
