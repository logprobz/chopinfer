//! The conversation a chat format renders, as the toolkit reads it from a
//! request ([`crate::toolkit::chat_message_v1`]).

use crate::request::{
    FrontendContentPartV1, FrontendJsonV1, FrontendRoleV1, FrontendToolCallV1, FrontendToolV1,
};

/// Whether a rendered prompt asks the model to reason before it answers.
pub use runtime_bundle_abi::frontend_document::ModeV1 as PromptModeV1;

/// One assistant turn a conversation already contains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssistantTurnV1 {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<FrontendToolCallV1>,
}

/// One tool result a conversation already contains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolResultV1 {
    pub tool_call_id: String,
    pub content: String,
}

/// One message of a conversation, in the roles a prompt renders.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatMessageV1 {
    System(String),
    Developer(String),
    User(String),
    UserParts(Vec<FrontendContentPartV1>),
    Assistant(AssistantTurnV1),
    Tool(ToolResultV1),
    WithTools {
        role: FrontendRoleV1,
        content: String,
        tools: Vec<FrontendJsonV1>,
    },
}

/// Everything one render reads besides the declaration itself, as a
/// format's renderer receives it from the toolkit.
pub struct ChatRenderV1<'a> {
    pub mode: PromptModeV1,
    pub preamble: &'a str,
    pub messages: &'a [ChatMessageV1],
    pub continue_final_message: bool,
    pub add_generation_prompt: bool,
    pub tools: &'a [FrontendToolV1],
    pub response_format: Option<&'a FrontendJsonV1>,
    /// Whether reasoning before the last user query renders; only a
    /// format that reads the `preserve_thinking` keyword drops it, and one
    /// that reads `clear_thinking` empties it.
    pub preserve_thinking: bool,
}
