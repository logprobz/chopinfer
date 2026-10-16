//! Installed adapter for the shared preparation API and protocol.
fn main() -> model_qwen38::Result<()> {
    model_compiler_sdk::preparation::helper::run(
        &model_qwen38::preparation::QwenFamily,
        std::env::args().skip(1),
    )
}
