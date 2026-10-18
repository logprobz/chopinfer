//! The DeepSeek V4.1 Flash compiler: checkpoint admission, the tokenizer
//! contract, the package draft and the preparation provider the
//! `chopinfer-model-deepseek41` helper serves.
#![forbid(unsafe_code)]
pub mod checkpoint;
pub mod encoder_image;
pub mod frontend;
pub mod kv;
pub mod package;
pub mod preparation;
pub mod synthetic;
pub mod tokenizer;

pub use model_deepseek41_contract::BUNDLE_ID;
