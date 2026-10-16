//! Writing a prompt under the declared byte bound, with its media segments.
//!
//! A renderer appends every piece through [`append_v1`], which charges it
//! against `maximum_prompt_bytes` before it is written. A media item renders
//! as a private placeholder ([`media_placeholder_v1`]) that no client text can
//! carry ([`screen_media_text_v1`]); [`split_media_segments_v1`] then cuts the
//! finished prompt into text and media segments and charges each item's pad.

use crate::chat::{ChatMessageV1, ChatRenderV1};
use crate::descriptor::{FormatDescriptorV1, FrontendMediaKindV1};
use crate::request::{
    FrontendContentPartV1, FrontendJsonV1, FrontendMediaPartKindV1, FrontendMediaPartV1,
    FrontendToolV1, PromptSegmentV1,
};
use crate::toolkit::json::{field_v1, resource_v1};

// These separators are never emitted by the tokenizer. Client text that
// holds them is refused before they serve as private boundaries.
const OPEN: &str = "\u{e000}chopinfer-media:";
const CLOSE: &str = ":\u{e001}";

/// Appends `value` to `output`, refusing a prompt longer than `maximum`.
pub fn append_v1(output: &mut String, value: &str, maximum: usize) -> Result<(), String> {
    let length = output
        .len()
        .checked_add(value.len())
        .ok_or_else(|| resource_v1("prompt length overflow"))?;
    if length > maximum {
        return Err(resource_v1("prompt exceeds the byte limit"));
    }
    output
        .try_reserve(value.len())
        .map_err(|_| resource_v1("prompt allocation failed"))?;
    output.push_str(value);
    Ok(())
}

/// Checks one name against the charset and length this model declared.
pub fn validate_name_v1(
    descriptor: &FormatDescriptorV1,
    name: &str,
    context: &str,
) -> Result<(), String> {
    let charset = &descriptor.tools.name_bytes;
    let maximum = descriptor.limits.maximum_name_bytes;
    if name.is_empty() || name.len() > maximum || !name.bytes().all(|byte| charset.permits(byte)) {
        return Err(field_v1(format!(
            "{context} name must match {} and fit {maximum} bytes",
            charset.describe()
        )));
    }
    Ok(())
}

/// The placeholder a renderer writes for the media item at `index` of the
/// media list it returns.
#[must_use]
pub fn media_placeholder_v1(index: usize) -> String {
    format!("{OPEN}{index}{CLOSE}")
}

/// Whether `text` holds a media placeholder boundary or a declared media
/// pad, which no client text may carry.
#[must_use]
pub fn reserved_media_text_v1(descriptor: &FormatDescriptorV1, text: &str) -> bool {
    text.contains(OPEN)
        || text.contains(CLOSE)
        || descriptor.multimodal.as_ref().is_some_and(|declaration| {
            declaration
                .modalities
                .iter()
                .any(|modality| text.contains(&modality.render.pad))
        })
}

/// Refuses a render whose client text anywhere holds a media boundary or
/// pad ([`reserved_media_text_v1`]): every message, tool and tool object.
pub fn screen_media_text_v1(
    descriptor: &FormatDescriptorV1,
    render: &ChatRenderV1<'_>,
    tool_objects: &[FrontendJsonV1],
) -> Result<(), String> {
    let reserved = |text: &str| reserved_media_text_v1(descriptor, text);
    let check_json = |value: &FrontendJsonV1| json_has_string_v1(value, &reserved);
    let check_tool = |tool: &FrontendToolV1| {
        reserved(&tool.name)
            || tool.description.as_deref().is_some_and(reserved)
            || check_json(&tool.parameters)
    };
    let invalid = render.tools.iter().any(check_tool)
        || tool_objects.iter().any(check_json)
        || render.response_format.is_some_and(check_json)
        || render.messages.iter().any(|message| match message {
            ChatMessageV1::System(text)
            | ChatMessageV1::Developer(text)
            | ChatMessageV1::User(text) => reserved(text),
            ChatMessageV1::UserParts(parts) => parts.iter().any(|part| match part {
                FrontendContentPartV1::Text(text)
                | FrontendContentPartV1::ToolReference { name: text } => reserved(text),
                FrontendContentPartV1::Media(_) => false,
            }),
            ChatMessageV1::Assistant(turn) => {
                turn.content
                    .iter()
                    .chain(turn.reasoning_content.iter())
                    .any(|text| reserved(text))
                    || turn.tool_calls.iter().any(|call| {
                        reserved(&call.id)
                            || reserved(&call.name)
                            || reserved(&call.arguments)
                            || serde_json::from_str::<FrontendJsonV1>(&call.arguments)
                                .is_ok_and(|value| check_json(&value))
                    })
            }
            ChatMessageV1::Tool(tool) => reserved(&tool.content) || reserved(&tool.tool_call_id),
            ChatMessageV1::WithTools { content, tools, .. } => {
                reserved(content) || tools.iter().any(check_json)
            }
        });
    if invalid {
        return Err(field_v1(
            "reserved media boundary or placeholder in client text",
        ));
    }
    Ok(())
}

/// The declared kind of one media part, refusing a part the declaration
/// does not serve: no media at all, or not this kind and part type.
pub fn media_kind_v1(
    descriptor: &FormatDescriptorV1,
    item: &FrontendMediaPartV1,
) -> Result<FrontendMediaKindV1, String> {
    let declaration = descriptor
        .multimodal
        .as_ref()
        .ok_or_else(|| field_v1("this bundle does not support media inputs"))?;
    let kind = media_part_kind_v1(item.kind);
    if !declaration
        .modalities
        .iter()
        .any(|entry| entry.kind == kind && entry.part_types.contains(&item.part_type))
    {
        return Err(field_v1("media part type is not declared by this bundle"));
    }
    Ok(kind)
}

/// Cuts a rendered prompt into text and media segments at the placeholders
/// of `media`, in order, and returns the prompt with each placeholder
/// replaced by its kind's pad, the segments and the media. The prompt with
/// pads must fit `maximum_prompt_bytes`.
pub fn split_media_segments_v1(
    descriptor: &FormatDescriptorV1,
    prompt: String,
    media: Vec<FrontendMediaPartV1>,
) -> Result<(String, Vec<PromptSegmentV1>, Vec<FrontendMediaPartV1>), String> {
    if prompt.matches(OPEN).count() != media.len() || prompt.matches(CLOSE).count() != media.len() {
        return Err(field_v1("unexpected media boundary in rendered prompt"));
    }
    if media.is_empty() {
        return Ok((prompt.clone(), vec![PromptSegmentV1::Text(prompt)], media));
    }
    let mut segments = Vec::new();
    let mut debug_prompt = String::new();
    let mut rest = prompt.as_str();
    for (index, item) in media.iter().enumerate() {
        let marker = media_placeholder_v1(index);
        let (before, after) = rest
            .split_once(&marker)
            .ok_or_else(|| field_v1("media render boundary was lost"))?;
        segments.push(PromptSegmentV1::Text(before.to_owned()));
        debug_prompt.push_str(before);
        segments.push(PromptSegmentV1::Media(index));
        let kind = media_part_kind_v1(item.kind);
        let pad = &descriptor
            .multimodal
            .as_ref()
            .expect("media declaration checked")
            .modalities
            .iter()
            .find(|entry| entry.kind == kind)
            .expect("modality checked")
            .render
            .pad;
        debug_prompt.push_str(pad);
        rest = after;
    }
    segments.push(PromptSegmentV1::Text(rest.to_owned()));
    debug_prompt.push_str(rest);
    if debug_prompt.len() > descriptor.limits.maximum_prompt_bytes {
        return Err(resource_v1(
            "rendered media prompt exceeds maximum_prompt_bytes",
        ));
    }
    Ok((debug_prompt, segments, media))
}

const fn media_part_kind_v1(kind: FrontendMediaPartKindV1) -> FrontendMediaKindV1 {
    match kind {
        FrontendMediaPartKindV1::Image => FrontendMediaKindV1::Image,
        FrontendMediaPartKindV1::Audio => FrontendMediaKindV1::Audio,
        FrontendMediaPartKindV1::Video => FrontendMediaKindV1::Video,
    }
}

fn json_has_string_v1(value: &FrontendJsonV1, matches: &impl Fn(&str) -> bool) -> bool {
    match value {
        FrontendJsonV1::String(text) => matches(text),
        FrontendJsonV1::Array(items) => items.iter().any(|item| json_has_string_v1(item, matches)),
        FrontendJsonV1::Object(entries) => entries
            .iter()
            .any(|(key, value)| matches(key) || json_has_string_v1(value, matches)),
        _ => false,
    }
}
