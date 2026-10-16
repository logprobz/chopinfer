//! Tokenizer assets pinned to the published checkpoint, verified before parsing.
use runtime_tokenizer::{FileIdentity, Tokenizer, TokenizerContract, TokenizerDialect};
use std::path::Path;

/// Added markers allowed in rendered prompts. Ordinary added tokens retain their artifact flags.
pub const PROMPT_TOKENS: &[&str] = &[
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
    "<|audio_pad|>",
];

/// The component policy this model's frontend declaration carries.
pub fn tokenizer_dialect() -> TokenizerDialect {
    runtime_frontend_contract::tokenizer_dialect_v1(&crate::frontend::descriptor().tokenizer.policy)
}

pub fn tokenizer_contract() -> runtime_tokenizer::Result<TokenizerContract> {
    TokenizerContract::new_with_dialect(
        "Qwen/Qwen3.8-Flash-Next-FP8",
        crate::REVISION,
        FileIdentity::new(
            "tokenizer.json",
            12809320,
            "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
        )?,
        FileIdentity::new(
            "tokenizer_config.json",
            17928,
            "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
        )?,
        248044,
        248077,
        247587,
        33,
        262144,
        None,
        "<|im_end|>",
        "<|endoftext|>",
        &[
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ],
        PROMPT_TOKENS,
        tokenizer_dialect(),
    )
}

pub fn load_tokenizer(root: &Path) -> crate::Result<Tokenizer> {
    let definition = crate::source::read(root, "tokenizer.json", 32 * 1024 * 1024)?;
    let configuration = crate::source::read(root, "tokenizer_config.json", 64 * 1024)?;
    Ok(Tokenizer::from_verified_bytes(
        &definition,
        &configuration,
        &tokenizer_contract()?,
    )?)
}
