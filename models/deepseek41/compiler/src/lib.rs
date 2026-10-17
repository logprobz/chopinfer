//! The DeepSeek V4.1 Flash compiler: checkpoint admission, the tokenizer
//! contract and the package. The `chopinfer-model-deepseek41` helper joins
//! it with the packager.
#![forbid(unsafe_code)]
pub mod checkpoint;
pub mod tokenizer;

pub use model_deepseek41_contract::BUNDLE_ID;
