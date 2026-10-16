//! Self-contained tokenizer facts for the synthetic TP4 fixture.

pub const TP4_WIDTH: u32 = 4;
pub const TP4_TEXT_PROMPT_SPECIAL_TOKENS: [&str; 16] = [
    "[gMASK]",
    "<sop>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
];
pub const TP4_PRETOKENIZER_PATTERN: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

#[derive(Clone, Copy)]
pub struct FixtureTokenizerMetadata;
impl FixtureTokenizerMetadata {
    pub const fn config_vocabulary_size(self) -> u32 {
        154_880
    }
    pub const fn bos_prefix(self) -> &'static [&'static str; 2] {
        &["[gMASK]", "<sop>"]
    }
    pub const fn bos_prefix_ids(self) -> &'static [u32; 2] {
        &[154_822, 154_824]
    }
    pub const fn eos_token(self) -> &'static str {
        "<|endoftext|>"
    }
    pub const fn eos_token_id(self) -> u32 {
        154_820
    }
    pub const fn pad_token(self) -> &'static str {
        "<|endoftext|>"
    }
}
pub const fn tokenizer_metadata() -> FixtureTokenizerMetadata {
    FixtureTokenizerMetadata
}
