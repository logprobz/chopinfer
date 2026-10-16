//! The reasoning controls a format reads: which mode and effort a request
//! selects, whether earlier reasoning renders, and whether a prompt leaves
//! reasoning open.

use crate::chat::PromptModeV1;
use crate::descriptor::{
    FrontendEffortUseV1, FrontendReasoningEffortV1, FrontendReasoningPolicyV1,
};
use crate::format::FormatRenderRequestV1;
use crate::request::{FrontendErrorV1, FrontendJsonV1};

/// Whether the last reasoning marker in `prompt` closes reasoning: a
/// reasoning close that no reasoning open follows. This is the test vLLM's
/// thinking parsers make before a structured output applies
/// (`is_reasoning_end`, which reads a prompt back to its last reasoning
/// start or end token). A prompt with neither marker leaves reasoning as it
/// was. The markers are read without the whitespace a template writes
/// around them, and an empty close closes nothing.
#[must_use]
fn closes_reasoning_v1(open: &str, close: &str, prompt: &str) -> bool {
    let (open, close) = (open.trim(), close.trim());
    if close.is_empty() {
        return false;
    }
    prompt
        .rfind(close)
        .is_some_and(|start| open.is_empty() || !prompt[start + close.len()..].contains(open))
}

/// Whether a rendered prompt leaves reasoning open, so the completion begins
/// in reasoning: a thinking prompt does, except a continued final assistant
/// turn whose reasoning the prompt already closed (`closes_reasoning_v1`),
/// as the reference parsers read a prompt.
#[must_use]
pub fn reasoning_open_v1(
    mode: PromptModeV1,
    continue_final_message: bool,
    open: &str,
    close: &str,
    prompt: &str,
) -> bool {
    mode == PromptModeV1::Thinking
        && !(continue_final_message && closes_reasoning_v1(open, close, prompt))
}

/// The top-level `reasoning_effort` values the OpenAI protocol accepts, as
/// vLLM's request schema lists them.
const PROTOCOL_EFFORTS_V1: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// What this model's reasoning controls mean together.
///
/// The controls reach the template as vLLM passes them: a top-level effort
/// replaces the nested one and, unless the client sent `enable_thinking`,
/// sets it to whether the effort differs from the disabling value. The
/// declaration names the keywords its template reads and when it reads the
/// effort, so a control the template ignores changes nothing.
pub fn select_mode_v1<'a>(
    policy: &'a FrontendReasoningPolicyV1,
    request: &FormatRenderRequestV1<'_>,
) -> Result<(PromptModeV1, &'a FrontendReasoningEffortV1), FrontendErrorV1> {
    let top = request.reasoning_effort.as_deref();
    if let Some(value) = top
        && !PROTOCOL_EFFORTS_V1.contains(&value)
        && value != policy.disable_value
        && policy.effort(value).is_none()
    {
        let mut accepted = PROTOCOL_EFFORTS_V1.to_vec();
        for name in std::iter::once(policy.disable_value.as_str()).chain(policy.effort_names()) {
            if !accepted.contains(&name) {
                accepted.push(name);
            }
        }
        return Err(FrontendErrorV1::new(
            format!(
                "top-level reasoning_effort must be {}",
                alternatives_v1(&accepted)
            ),
            Some("reasoning_effort"),
        ));
    }
    let enable_thinking = request
        .enable_thinking
        .or(top.map(|value| value != policy.disable_value));
    let switch = policy
        .thinking_keywords
        .iter()
        .find_map(|keyword| match keyword.as_str() {
            "enable_thinking" => enable_thinking,
            "thinking" => request.thinking,
            // A checked declaration names only the keywords above.
            _ => None,
        });
    let mode = match switch {
        Some(false) if !policy.allow_disabled => return Err(disabled_v1()),
        Some(false) => PromptModeV1::Chat,
        Some(true) => PromptModeV1::Thinking,
        None if policy.default_enabled => PromptModeV1::Thinking,
        None => PromptModeV1::Chat,
    };
    let default = policy
        .effort(&policy.default_effort)
        .expect("a validated declaration names a declared default effort");
    let named = top.or(request.nested_reasoning_effort.as_deref());
    let effort = match (mode, policy.effort_use, named) {
        (PromptModeV1::Thinking, FrontendEffortUseV1::WhenThinking, Some(name)) => {
            policy.effort(name).ok_or_else(|| {
                FrontendErrorV1::new(
                    format!(
                        "reasoning_effort must be {}",
                        alternatives_v1(&policy.effort_names())
                    ),
                    Some("reasoning_effort"),
                )
            })?
        }
        _ => default,
    };
    Ok((mode, effort))
}

/// Whether reasoning before the last user message renders, by the keyword
/// the format reads: `clear_thinking` when `reads_clear_thinking`, otherwise
/// `preserve_thinking`. `clear_thinking` reaches here as the client sent it,
/// so a format that reads it refuses anything but a boolean, and one that
/// does not ignores it whatever it holds.
pub fn preserve_thinking_v1(
    reads_clear_thinking: bool,
    request: &FormatRenderRequestV1<'_>,
) -> Result<bool, FrontendErrorV1> {
    if !reads_clear_thinking {
        return Ok(request.preserve_thinking.unwrap_or(true));
    }
    match &request.clear_thinking {
        None => Ok(true),
        Some(FrontendJsonV1::Bool(clear)) => Ok(!clear),
        Some(_) => Err(FrontendErrorV1::new(
            "chat_template_kwargs.clear_thinking must be a boolean",
            Some("chat_template_kwargs"),
        )),
    }
}

/// Writes a declared value list the way a refusal message reads it.
fn alternatives_v1(values: &[&str]) -> String {
    match values {
        [] => String::new(),
        [only] => (*only).to_owned(),
        [first, second] => format!("{first} or {second}"),
        [head @ .., last] => format!("{}, or {last}", head.join(", ")),
    }
}

fn disabled_v1() -> FrontendErrorV1 {
    FrontendErrorV1::new(
        "this frontend does not allow reasoning to be disabled",
        Some("reasoning_effort"),
    )
}
