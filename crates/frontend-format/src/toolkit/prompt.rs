//! Writing a prompt under the declared byte bound, with its media segments.
//!
//! A renderer appends every piece through [`append_v1`], which charges it
//! against `maximum_prompt_bytes` before it is written. A media item renders
//! as a private placeholder ([`media_placeholder_v1`]) that no client text can
//! carry ([`screen_media_text_v1`]); [`split_media_segments_v1`] then cuts the
//! finished prompt into text and media segments and charges each item's pad.
//! A renderer whose declaration sets `vocabulary.explicit_special_tokens`
//! writes a [`MarkedPromptV1`] instead, which names its special tokens, and
//! cuts it with [`split_marked_media_segments_v1`].

use std::ops::Range;

use crate::chat::{ChatMessageV1, ChatRenderV1};
use crate::descriptor::{FormatDescriptorV1, FrontendMediaKindV1, FrontendToolGrammarV1};
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

/// A prompt that names its special tokens: what a renderer writes when its
/// declaration sets `vocabulary.explicit_special_tokens`.
///
/// The renderer writes each special token its template writes with
/// [`MarkedPromptV1::push_special`] and every other piece, client text
/// included, with [`MarkedPromptV1::push_plain`], so the spelling of a
/// special token in client text stays plain text. Each push charges
/// `maximum_prompt_bytes` as [`append_v1`] does.
/// [`split_marked_media_segments_v1`] then cuts the finished prompt into
/// segments.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MarkedPromptV1 {
    text: String,
    /// The byte ranges of the special tokens in `text`, ascending, disjoint
    /// and not empty.
    specials: Vec<Range<usize>>,
}

impl MarkedPromptV1 {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            text: String::new(),
            specials: Vec::new(),
        }
    }

    /// Appends plain text, refusing a prompt longer than `maximum`.
    pub fn push_plain(&mut self, text: &str, maximum: usize) -> Result<(), String> {
        append_v1(&mut self.text, text, maximum)
    }

    /// Appends one special token by its text, refusing an empty token and
    /// a prompt longer than `maximum`.
    pub fn push_special(&mut self, token: &str, maximum: usize) -> Result<(), String> {
        if token.is_empty() {
            return Err(field_v1("a prompt special token cannot be empty"));
        }
        self.specials
            .try_reserve(1)
            .map_err(|_| resource_v1("prompt allocation failed"))?;
        let start = self.text.len();
        append_v1(&mut self.text, token, maximum)?;
        self.specials.push(start..self.text.len());
        Ok(())
    }

    /// Appends `other` with its special tokens, refusing a prompt longer
    /// than `maximum`.
    pub fn push_prompt(&mut self, other: &Self, maximum: usize) -> Result<(), String> {
        self.specials
            .try_reserve(other.specials.len())
            .map_err(|_| resource_v1("prompt allocation failed"))?;
        let offset = self.text.len();
        append_v1(&mut self.text, &other.text, maximum)?;
        self.specials.extend(
            other
                .specials
                .iter()
                .map(|special| special.start + offset..special.end + offset),
        );
        Ok(())
    }

    /// The prompt's text, special tokens included.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The byte ranges of the special tokens in [`Self::text`], ascending.
    #[must_use]
    pub fn specials(&self) -> &[Range<usize>] {
        &self.specials
    }
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

/// Checks one parameter name: any text the tool-call parser reads back as
/// it was written, as `encoding.py` reads any `name="..."`. It is not
/// empty, fits `maximum_name_bytes`, has no whitespace around it (the parser
/// trims a name), and holds no reserved marker and none of the bare
/// tool-call markers. Nor may a marker the parser stops at while it reads a
/// name begin inside it in the text a name close writes after it
/// ([`name_stops_v1`]): such a marker would end the name early, or the call.
/// Unlike a function name, it need not be made of the declared name bytes.
pub fn validate_parameter_name_v1(
    descriptor: &FormatDescriptorV1,
    name: &str,
    context: &str,
) -> Result<(), String> {
    let parser = &descriptor.parser;
    let maximum = descriptor.limits.maximum_name_bytes;
    let marked = [
        &parser.tool_call_open,
        &parser.tool_call_close,
        &parser.function_open,
        &parser.function_name_close,
        &parser.function_close,
        &parser.parameter_open,
        &parser.parameter_name_close,
        &parser.parameter_json_name_close,
        &parser.parameter_close,
    ]
    .into_iter()
    .map(|marker| marker.trim())
    .chain(parser.reserved_markers.iter().map(String::as_str))
    .any(|marker| !marker.is_empty() && name.contains(marker));
    let (closes, stops) = name_stops_v1(descriptor);
    // A stop that begins inside the name, whole in the name and its close or
    // running past them, is where the parser would end the name instead.
    let cut = closes.iter().any(|close| {
        let text = format!("{name}{close}");
        (0..name.len()).any(|start| {
            let rest = &text.as_bytes()[start..];
            stops
                .iter()
                .any(|stop| rest.starts_with(stop.as_bytes()) || stop.as_bytes().starts_with(rest))
        })
    });
    if name.is_empty() || name.len() > maximum || name.trim() != name || marked || cut {
        return Err(field_v1(format!(
            "{context} name must fit {maximum} bytes, without surrounding whitespace or a \
             tool-call marker"
        )));
    }
    Ok(())
}

/// The name closes a parameter name may be written before in `descriptor`'s
/// tool grammar, as declared, and the markers the parser stops at while it
/// reads a name: each name close, the marker that ends the call around it
/// (a function close, or a key-value grammar's call close), bare, and the
/// declared stop texts.
fn name_stops_v1(descriptor: &FormatDescriptorV1) -> (Vec<&str>, Vec<&str>) {
    let parser = &descriptor.parser;
    let (closes, ends) = match parser.tool_grammar {
        FrontendToolGrammarV1::XmlFunctionParametersV1 => (
            vec![parser.parameter_name_close.as_str()],
            vec![parser.function_close.as_str()],
        ),
        FrontendToolGrammarV1::XmlTypedParametersV1 => (
            vec![
                parser.parameter_name_close.as_str(),
                parser.parameter_json_name_close.as_str(),
            ],
            vec![parser.function_close.as_str()],
        ),
        FrontendToolGrammarV1::XmlKeyValueArgumentsV1 => (
            vec![parser.parameter_name_close.as_str()],
            vec![parser.tool_call_close.as_str()],
        ),
    };
    let vocabulary = &descriptor.vocabulary;
    let terminators = vocabulary
        .boundary_tokens
        .iter()
        .filter(|token| vocabulary.stop_token_ids.contains(&token.id))
        .map(|token| token.text.as_str());
    let stops = closes
        .iter()
        .chain(&ends)
        .map(|marker| marker.trim())
        .chain(terminators)
        .filter(|marker| !marker.is_empty())
        .collect();
    (closes, stops)
}

/// Checks one function name: a name as [`validate_name_v1`] checks it, or,
/// where the declaration qualifies names (`tools.namespaces`), a namespace
/// and a name joined by the separator, each part such a name, so the whole
/// fits `maximum_function_name_bytes`.
pub fn validate_function_name_v1(
    descriptor: &FormatDescriptorV1,
    name: &str,
    context: &str,
) -> Result<(), String> {
    match descriptor.tools.split_function_name(name) {
        (Some(namespace), bare) => {
            validate_name_v1(descriptor, namespace, context)?;
            validate_name_v1(descriptor, bare, context)
        }
        (None, _) => validate_name_v1(descriptor, name, context),
    }
}

/// The placeholder a renderer writes for the media item at `index` of the
/// media list it returns.
#[must_use]
pub fn media_placeholder_v1(index: usize) -> String {
    format!("{OPEN}{index}{CLOSE}")
}

/// Whether `text` holds a media placeholder boundary, which no client text
/// may carry, or a declared media pad. A pad is reserved too, unless the
/// declaration sets `vocabulary.explicit_special_tokens`: there client text
/// is plain text, so a pad's spelling in it is ordinary text.
#[must_use]
pub fn reserved_media_text_v1(descriptor: &FormatDescriptorV1, text: &str) -> bool {
    text.contains(OPEN)
        || text.contains(CLOSE)
        || (!descriptor.vocabulary.explicit_special_tokens
            && descriptor.multimodal.as_ref().is_some_and(|declaration| {
                declaration
                    .modalities
                    .iter()
                    .any(|modality| text.contains(&modality.render.pad))
            }))
}

/// Refuses a render whose client text anywhere holds a media boundary, or a
/// pad where pads are reserved ([`reserved_media_text_v1`]): every message,
/// tool and tool object.
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
        debug_prompt.push_str(media_pad_v1(descriptor, item));
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

/// Cuts a marked prompt into text, media and special segments, and returns
/// what [`split_media_segments_v1`] returns: the prompt with each
/// placeholder replaced by its kind's pad, which must fit
/// `maximum_prompt_bytes`, the segments and the media.
///
/// Each special token becomes a special segment and must spell a declared
/// boundary token. The plain text around them is cut at the placeholders of
/// `media`, in order; a placeholder lies whole in plain text. Adjacent plain
/// text is one text segment. A media segment has a text segment on each
/// side, empty or not, as [`split_media_segments_v1`] writes them; a text
/// segment between special segments, or at either end next to one, is
/// written only when it holds text. A prompt with no special token so
/// splits exactly as [`split_media_segments_v1`] splits its text.
pub fn split_marked_media_segments_v1(
    descriptor: &FormatDescriptorV1,
    prompt: MarkedPromptV1,
    media: Vec<FrontendMediaPartV1>,
) -> Result<(String, Vec<PromptSegmentV1>, Vec<FrontendMediaPartV1>), String> {
    let MarkedPromptV1 { text, specials } = prompt;
    if text.matches(OPEN).count() != media.len() || text.matches(CLOSE).count() != media.len() {
        return Err(field_v1("unexpected media boundary in rendered prompt"));
    }
    // Every segment, with a text segment between each two others and at
    // both ends, empty or not.
    let mut full = Vec::new();
    let mut padded = String::new();
    let mut placed = 0;
    let mut from = 0;
    let ends = specials
        .iter()
        .map(|special| special.start)
        .chain(std::iter::once(text.len()));
    for (index, end) in ends.enumerate() {
        let mut rest = &text[from..end];
        while let Some(item) = media.get(placed) {
            let Some((before, after)) = rest.split_once(&media_placeholder_v1(placed)) else {
                break;
            };
            full.push(PromptSegmentV1::Text(before.to_owned()));
            padded.push_str(before);
            full.push(PromptSegmentV1::Media(placed));
            padded.push_str(media_pad_v1(descriptor, item));
            placed += 1;
            rest = after;
        }
        full.push(PromptSegmentV1::Text(rest.to_owned()));
        padded.push_str(rest);
        let Some(special) = specials.get(index) else {
            break;
        };
        let token = &text[special.clone()];
        if !descriptor
            .vocabulary
            .boundary_tokens
            .iter()
            .any(|boundary| boundary.text == token)
        {
            return Err(field_v1(format!(
                "rendered special token '{token}' is not a declared boundary token"
            )));
        }
        full.push(PromptSegmentV1::Special(token.to_owned()));
        padded.push_str(token);
        from = special.end;
    }
    if placed != media.len() {
        return Err(field_v1("media render boundary was lost"));
    }
    if padded.len() > descriptor.limits.maximum_prompt_bytes {
        return Err(resource_v1(
            "rendered media prompt exceeds maximum_prompt_bytes",
        ));
    }
    let media_at = |index: usize| matches!(full.get(index), Some(PromptSegmentV1::Media(_)));
    let written = (0..full.len())
        .map(|index| match &full[index] {
            PromptSegmentV1::Text(run) => {
                !run.is_empty()
                    || full.len() == 1
                    || media_at(index + 1)
                    || index.checked_sub(1).is_some_and(media_at)
            }
            _ => true,
        })
        .collect::<Vec<_>>();
    let segments = full
        .into_iter()
        .zip(written)
        .filter_map(|(segment, written)| written.then_some(segment))
        .collect();
    Ok((padded, segments, media))
}

/// The pad of the placed media part `item`'s kind.
fn media_pad_v1<'a>(descriptor: &'a FormatDescriptorV1, item: &FrontendMediaPartV1) -> &'a str {
    let kind = media_part_kind_v1(item.kind);
    &descriptor
        .multimodal
        .as_ref()
        .expect("media declaration checked")
        .modalities
        .iter()
        .find(|entry| entry.kind == kind)
        .expect("modality checked")
        .render
        .pad
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
