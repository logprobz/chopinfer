//! The DeepSeek V4.1 Flash preparation helper: `serve --model` probes it
//! with a checkpoint's `config.json`, then asks it for the direct package
//! (`model_deepseek41::preparation`), which carries the native module
//! installed beside it (`libbundle_deepseek41.so`).
#![forbid(unsafe_code)]

use model_compiler_sdk::files::Result;
use model_compiler_sdk::preparation::helper;
use model_deepseek41::preparation::DeepSeek41Family;

fn main() -> Result<()> {
    helper::run(&DeepSeek41Family, std::env::args().skip(1))
}
