//! The GLM 5.3 Flash preparation helper: `serve --model` probes it with a
//! checkpoint's `config.json`, then asks it for the direct package
//! (`model_glm5next::preparation`), which carries the native module
//! installed beside it (`libbundle_glm5next.so`).
#![forbid(unsafe_code)]

use model_compiler_sdk::files::Result;
use model_compiler_sdk::preparation::helper;
use model_glm5next::preparation::Glm5NextFamily;

fn main() -> Result<()> {
    helper::run(&Glm5NextFamily, std::env::args().skip(1))
}
