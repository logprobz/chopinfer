//! The model-neutral contract between the transport and one model's frontend.
//!
//! The transport parses and validates OpenAI requests, buffers and accounts for
//! generation, and writes OpenAI responses. It knows no tokenizer, no chat
//! template, no completion grammar, and no vocabulary. Everything a served model
//! decides reaches it through this contract: which model identities exist, how
//! long a context may be, how a conversation renders into a prompt, and whether
//! a tool call the backend emitted is one that model could have produced.
//!
//! A frontend supplies those answers from bundle data or from its own code. The
//! transport never inspects what it returns beyond the bounds declared here.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

/// One JSON value from a request, with object keys in their arrival order.
///
/// Order is preserved because a rendered prompt can depend on it, and the
/// transport must not silently reorder what a client sent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrontendJsonV1 {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}

impl FrontendJsonV1 {
    /// The same value as `serde_json` holds it, keys still in arrival order.
    #[must_use]
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            Self::Null => serde_json::Value::Null,
            Self::Bool(value) => serde_json::Value::Bool(*value),
            Self::Number(number) => serde_json::Value::Number(number.clone()),
            Self::String(text) => serde_json::Value::String(text.clone()),
            Self::Array(values) => {
                serde_json::Value::Array(values.iter().map(Self::to_json_value).collect())
            }
            Self::Object(entries) => serde_json::Value::Object(
                entries
                    .iter()
                    .map(|(name, value)| (name.clone(), value.to_json_value()))
                    .collect(),
            ),
        }
    }
}

impl<'de> Deserialize<'de> for FrontendJsonV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(OrderedVisitorV1 {
            duplicates: DuplicateKeysV1::Refuse,
        })
    }
}

/// One JSON value read as [`FrontendJsonV1`] reads one, except that a key
/// named more than once in an object, at any depth, is kept once, in the
/// place it first appeared, with the last value: what `serde_json` and
/// Python's `json` read. Numbers keep their digits, and a key that spells
/// `serde_json`'s private number marker stays an ordinary key.
pub(crate) struct LastValueJsonV1(pub(crate) FrontendJsonV1);

impl<'de> Deserialize<'de> for LastValueJsonV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer
            .deserialize_any(OrderedVisitorV1 {
                duplicates: DuplicateKeysV1::KeepLast,
            })
            .map(Self)
    }
}

/// What an ordered read does with a key an object names twice.
#[derive(Clone, Copy)]
enum DuplicateKeysV1 {
    Refuse,
    KeepLast,
}

struct OrderedVisitorV1 {
    duplicates: DuplicateKeysV1,
}

impl<'de> Visitor<'de> for OrderedVisitorV1 {
    type Value = FrontendJsonV1;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(FrontendJsonV1::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(FrontendJsonV1::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(FrontendJsonV1::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(FrontendJsonV1::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(FrontendJsonV1::Number(value.into()))
    }

    fn visit_i128<E: de::Error>(self, value: i128) -> Result<Self::Value, E> {
        serde_json::Number::from_i128(value)
            .map(FrontendJsonV1::Number)
            .ok_or_else(|| E::custom("integer out of range"))
    }

    fn visit_u128<E: de::Error>(self, value: u128) -> Result<Self::Value, E> {
        serde_json::Number::from_u128(value)
            .map(FrontendJsonV1::Number)
            .ok_or_else(|| E::custom("integer out of range"))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(FrontendJsonV1::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_string(value.to_owned())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(FrontendJsonV1::String(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut items = Vec::new();
        while let Some(item) = sequence.next_element_seed(OrderedSeedV1 {
            duplicates: self.duplicates,
        })? {
            items.push(item);
        }
        Ok(FrontendJsonV1::Array(items))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        // serde_json carries numbers outside its primitive visitor callbacks
        // through this single-field representation when arbitrary precision is on.
        let sized_map = map.size_hint().is_some();
        let first = map.next_key::<OrderedKeyV1>()?;
        if !sized_map && first.as_ref().is_some_and(|key| key.number) {
            let text = map.next_value::<String>()?;
            if map.next_key::<String>()?.is_some() {
                return Err(de::Error::custom("invalid JSON number representation"));
            }
            let number: serde_json::Number = text.parse().map_err(de::Error::custom)?;
            if text.contains(['.', 'e', 'E']) && number.as_f64().is_none_or(|v| !v.is_finite()) {
                return Err(de::Error::custom("non-finite JSON number"));
            }
            return Ok(FrontendJsonV1::Number(number));
        }
        let mut entries: Vec<(String, FrontendJsonV1)> = Vec::new();
        // Each key's place in `entries`.
        let mut seen = std::collections::HashMap::new();
        let mut next = first.map(|key| key.text);
        while let Some(key) = next {
            if entries.len() >= crate::json_limits::MAX_OBJECT_ENTRIES {
                return Err(de::Error::custom("JSON object entry limit exceeded"));
            }
            // Refuse a repeated key before reading its value, so the refusal
            // names this key at its own position.
            let place = seen.get(&key).copied();
            if place.is_some() && matches!(self.duplicates, DuplicateKeysV1::Refuse) {
                return Err(de::Error::custom(format!("duplicate field `{key}`")));
            }
            let value = map.next_value_seed(OrderedSeedV1 {
                duplicates: self.duplicates,
            })?;
            match place {
                None => {
                    seen.insert(key.clone(), entries.len());
                    entries.push((key, value));
                }
                Some(place) => entries[place].1 = value,
            }
            next = map.next_key::<String>()?;
        }
        Ok(FrontendJsonV1::Object(entries))
    }
}

// The number wrapper forwards option deserialization directly to a borrowed
// string. Real JSON map keys visit_some instead. Owned Value keys visit_string,
// and borrowed Value maps expose a size hint. Keep ordinary keys ordinary even
// when they spell serde_json's private marker.
struct OrderedKeyV1 {
    text: String,
    number: bool,
}

impl<'de> Deserialize<'de> for OrderedKeyV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl<'de> Visitor<'de> for KeyVisitor {
            type Value = OrderedKeyV1;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an object key or internal number marker")
            }
            fn visit_some<D: Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> Result<Self::Value, D::Error> {
                String::deserialize(deserializer).map(|text| OrderedKeyV1 {
                    text,
                    number: false,
                })
            }
            fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
                Ok(OrderedKeyV1 {
                    text: text.to_owned(),
                    number: text == "$serde_json::private::Number",
                })
            }
            fn visit_string<E: de::Error>(self, text: String) -> Result<Self::Value, E> {
                Ok(OrderedKeyV1 {
                    text,
                    number: false,
                })
            }
        }
        deserializer.deserialize_option(KeyVisitor)
    }
}

struct OrderedSeedV1 {
    duplicates: DuplicateKeysV1,
}

impl<'de> DeserializeSeed<'de> for OrderedSeedV1 {
    type Value = FrontendJsonV1;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(OrderedVisitorV1 {
            duplicates: self.duplicates,
        })
    }
}

/// One model identity this deployment serves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendModelV1 {
    pub id: String,
    pub owned_by: String,
}

impl FrontendModelV1 {
    #[must_use]
    pub fn new(id: impl Into<String>, owned_by: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            owned_by: owned_by.into(),
        }
    }
}

/// The role one conversation message carried.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrontendRoleV1 {
    System,
    Developer,
    User,
    Assistant,
    Tool,
    /// A role the frontend declared it renders
    /// ([`ServerFrontendV1::extra_roles`]), by name.
    Declared(String),
}

/// One tool call a conversation already contains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendToolCallV1 {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// One message of a conversation, after transport-level shape validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendMessageV1 {
    pub role: FrontendRoleV1,
    pub content: Option<String>,
    /// Ordered content parts when the client supplied an array.
    pub parts: Option<Vec<FrontendContentPartV1>>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<FrontendToolCallV1>,
    pub tool_call_id: Option<String>,
    pub tools: Vec<FrontendJsonV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrontendContentPartV1 {
    Text(String),
    Media(FrontendMediaPartV1),
    /// A tool message's `tool_reference` part, by the tool name it carries.
    /// The transport admits one only for a frontend that reads them
    /// ([`ServerFrontendV1::tool_references`]).
    ToolReference {
        name: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontendMediaPartKindV1 {
    Image,
    Audio,
    Video,
}

impl FrontendMediaPartKindV1 {
    /// Every kind, in the order served modalities are reported.
    pub const ALL: [Self; 3] = [Self::Image, Self::Video, Self::Audio];

    /// The modality's name, as `--modalities` and `/health` spell it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

/// The modalities `frontend` serves, as `/health` and `/v1/models` report
/// them: text, then each media kind it accepts in
/// [`FrontendMediaPartKindV1::ALL`] order.
pub(crate) fn served_modalities(frontend: &(impl ServerFrontendV1 + ?Sized)) -> Vec<&'static str> {
    let media = frontend.served_media();
    std::iter::once("text")
        .chain(
            FrontendMediaPartKindV1::ALL
                .into_iter()
                .filter(|kind| media.contains(kind))
                .map(FrontendMediaPartKindV1::name),
        )
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendMediaPartV1 {
    pub kind: FrontendMediaPartKindV1,
    pub part_type: String,
    /// A data URL, canonical local path, or base64 input_audio payload.
    pub source: String,
    pub format: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromptSegmentV1 {
    Text(String),
    Media(usize),
}

/// One function tool a request offered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendToolV1 {
    pub name: String,
    pub description: Option<String>,
    pub parameters: FrontendJsonV1,
    pub parameters_present: bool,
    pub strict: Option<bool>,
}

/// Whether a request permitted tool calls, or forced them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrontendToolChoiceV1 {
    Auto,
    None,
    /// At least one call to an offered function.
    Required,
    /// Exactly one call to the named function.
    Named(String),
}

/// Everything one chat request offers a frontend for rendering.
///
/// The reasoning controls arrive as the strings the client sent, because what
/// they mean is the frontend's decision, not the transport's.
#[derive(Clone, Debug)]
pub struct FrontendChatRequestV1 {
    pub messages: Vec<FrontendMessageV1>,
    pub continue_final_message: bool,
    pub add_generation_prompt: bool,
    pub tools: Vec<FrontendToolV1>,
    pub tool_choice: FrontendToolChoiceV1,
    pub response_format: Option<FrontendJsonV1>,
    pub reasoning_effort: Option<String>,
    /// The `enable_thinking` template keyword, when the client sent one.
    pub enable_thinking: Option<bool>,
    /// The `thinking` template keyword, when the client sent one.
    pub thinking: Option<bool>,
    pub nested_reasoning_effort: Option<String>,
    /// The `preserve_thinking` template keyword, when the client sent one.
    pub preserve_thinking: Option<bool>,
    /// The `clear_thinking` template keyword as the client sent it. Only a
    /// frontend whose template reads it interprets or refuses the value.
    pub clear_thinking: Option<FrontendJsonV1>,
    /// The opaque `add_vision_id` keyword, interpreted only by declared media labels.
    pub add_vision_id: Option<FrontendJsonV1>,
    /// Each offered tool's function object as the client sent it, in order,
    /// when this frontend reads them; otherwise empty.
    pub tool_objects: Vec<FrontendJsonV1>,
    /// The client's whole `chat_template_kwargs` object: every key in the
    /// order sent, with its value unchanged, except that a typed keyword
    /// above sent as null is absent, as the transport reads it. A key the
    /// client named more than once, at any depth, appears once, where it
    /// first appeared, with the last value sent, as `serde_json` and
    /// Python's `json` read it. `None` when the client sent none, or sent an
    /// array, which the typed read takes positionally into the typed
    /// keywords above. A frontend reads whichever keys its template reads,
    /// so a model's own keyword needs no transport edit.
    pub chat_template_kwargs: Option<FrontendJsonV1>,
}

/// The `chat_template_kwargs` keywords this transport reads into typed
/// fields of [`FrontendChatRequestV1`].
const TYPED_TEMPLATE_KEYWORDS_V1: [&str; 6] = [
    "enable_thinking",
    "thinking",
    "reasoning_effort",
    "preserve_thinking",
    "clear_thinking",
    "add_vision_id",
];

/// The client's `chat_template_kwargs` value as a frontend receives it
/// ([`FrontendChatRequestV1::chat_template_kwargs`]): an object keeps every
/// key in order with its value unchanged, except a typed keyword sent as
/// null, which the transport reads as absent. Any other value forwards no
/// object: null is none, and an array, which the typed read takes
/// positionally, reaches the frontend as the typed keywords it set.
#[must_use]
pub fn forwarded_template_kwargs_v1(value: FrontendJsonV1) -> Option<FrontendJsonV1> {
    match value {
        FrontendJsonV1::Null => None,
        FrontendJsonV1::Object(entries) => Some(FrontendJsonV1::Object(
            entries
                .into_iter()
                .filter(|(name, value)| {
                    !(*value == FrontendJsonV1::Null
                        && TYPED_TEMPLATE_KEYWORDS_V1.contains(&name.as_str()))
                })
                .collect(),
        )),
        _ => None,
    }
}

/// What a frontend decided about one rendered chat, carried to its backend.
///
/// The transport reads only [`Self::reasoning_enabled`]. The frontend that
/// produced the state recovers its own type through [`Self::as_any`].
pub trait ChatStateV1: Any + fmt::Debug + Send + Sync + 'static {
    /// Whether a reasoning delta is legitimate for this rendered prompt.
    fn reasoning_enabled(&self) -> bool;

    fn as_any(&self) -> &dyn Any;
}

/// One rendered chat prompt and what it permits.
#[derive(Clone, Debug)]
pub struct FrontendRenderedChatV1 {
    pub prompt: String,
    pub segments: Vec<PromptSegmentV1>,
    pub media: Vec<FrontendMediaPartV1>,
    pub allowed_function_names: Vec<String>,
    pub state: Arc<dyn ChatStateV1>,
}

/// One frontend refusal, with the request parameter it blames if there is one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendErrorV1 {
    pub message: String,
    pub parameter: Option<String>,
}

impl FrontendErrorV1 {
    #[must_use]
    pub fn new(message: impl Into<String>, parameter: Option<&str>) -> Self {
        Self {
            message: message.into(),
            parameter: parameter.map(str::to_owned),
        }
    }
}

impl fmt::Display for FrontendErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for FrontendErrorV1 {}

// Keep trait declarations and alias delegation in one list. A new method,
// including one with a default, is forwarded automatically. Unsupported
// signatures fail compilation instead of silently inheriting a default.
macro_rules! frontend_with_alias_delegation {
    (@method $(#[$attr:meta])* fn $name:ident(&$receiver:ident $(, $arg:ident: $ty:ty)*) -> $result:ty $body:block) => {
        $(#[$attr])* fn $name(&$receiver $(, $arg: $ty)*) -> $result $body
    };
    (@method $(#[$attr:meta])* fn $name:ident(&$receiver:ident $(, $arg:ident: $ty:ty)*) -> $result:ty) => {
        $(#[$attr])* fn $name(&$receiver $(, $arg: $ty)*) -> $result;
    };
    ($( $(#[$attr:meta])* fn $name:ident(&$receiver:ident $(, $arg:ident: $ty:ty)* $(,)?) -> $result:ty $( $body:block )? $(;)? )*) => {
        /// Everything one served model tells the transport about itself.
        pub trait ServerFrontendV1: Send + Sync + 'static {
            /// Every API identity, most canonical first.
            fn models(&self) -> &[FrontendModelV1];
            $( frontend_with_alias_delegation!(@method $(#[$attr])* fn $name(&$receiver $(, $arg: $ty)*) -> $result $( $body )?); )*
        }
        impl ServerFrontendV1 for crate::aliases::AliasedFrontend {
            fn models(&self) -> &[FrontendModelV1] { &self.models }
            $( $(#[$attr])* fn $name(&$receiver $(, $arg: $ty)*) -> $result {
                $receiver.inner.$name($($arg),*)
            } )*
        }
    };
}

frontend_with_alias_delegation! {
    #[cfg(test)]
    fn delegation_probe(&self) -> usize { 0 }
    /// Sampling controls from the selected model's generation configuration.
    fn sampling_defaults(&self) -> runtime_sampling::SamplingDefaults {
        runtime_sampling::SamplingDefaults::default()
    }

    /// Whether this installed bundle declares media inputs.
    fn supports_media(&self) -> bool {
        false
    }

    /// Whether tool results may carry images, subject to served modalities
    /// and the same request limits as user images.
    fn tool_result_images(&self) -> bool {
        false
    }

    /// The encoded image formats the served model accepts, by name (`png`,
    /// `jpeg`, `gif`, `webp`).
    fn image_formats(&self) -> Vec<String> {
        vec!["png".to_owned(), "jpeg".to_owned()]
    }

    /// Whether images may also arrive as OpenAI `file` parts whose
    /// `file_data` is an image data URL.
    fn image_file_parts(&self) -> bool {
        false
    }

    /// The media kinds this server accepts, of [`FrontendMediaPartKindV1::ALL`];
    /// text is always served. The default accepts every kind when the bundle
    /// declares media inputs.
    fn served_media(&self) -> Vec<FrontendMediaPartKindV1> {
        if self.supports_media() {
            FrontendMediaPartKindV1::ALL.to_vec()
        } else {
            Vec::new()
        }
    }

    /// Whether `--modalities` (auto, all or a list) selected the modalities
    /// this server serves, so refusals name the flag. Capability alone does
    /// not imply that the flag applied.
    fn has_selected_modalities(&self) -> bool {
        false
    }

    /// Operator opt-in for HTTP(S) sources; the host enforces destination policy.
    fn allows_remote_media(&self) -> bool {
        false
    }

    /// Operator opt-in root for file URLs. The default refuses local files.
    fn allowed_local_media_path(&self) -> Option<&std::path::Path> {
        None
    }

    /// The most image parts one request may carry, across all of its
    /// messages, or `None` when this server accepts no images. `/v1/models`
    /// reports it, and the frontend refuses a request that exceeds it.
    fn maximum_images_per_request(&self) -> Option<usize> {
        None
    }
    /// The response field this frontend uses for assistant reasoning.
    fn reasoning_field(&self) -> &'static str {
        "reasoning_content"
    }
    /// Whether an assistant history turn with tool calls falls back to
    /// `reasoning_content` when this frontend's reasoning field is absent.
    fn tool_call_reasoning_fallback(&self) -> bool {
        false
    }
    /// Whether every assistant history turn falls back to
    /// `reasoning_content` when this frontend's reasoning field is absent.
    fn reasoning_content_fallback(&self) -> bool {
        false
    }
    /// Whether this frontend renders each offered tool's function object as
    /// the client sent it, so the transport carries the objects whole.
    fn reads_tool_objects(&self) -> bool {
        false
    }
    /// The message roles this frontend renders besides system, developer,
    /// user, assistant and tool. The transport admits a message of each
    /// that carries only string content.
    fn extra_roles(&self) -> &[String] {
        &[]
    }
    /// Whether this frontend renders `tool_reference` content parts in tool
    /// messages. The transport admits them there only when it does.
    fn tool_references(&self) -> bool {
        false
    }
    /// Whether thinking switches by the top-level `thinking` object rather
    /// than the template keywords. The transport then reads that object,
    /// whose `type` is `enabled`, its alias `adaptive`, or `disabled`,
    /// ignores the thinking keywords and the effort in
    /// `chat_template_kwargs`, and passes the object's switch as both
    /// thinking keywords, typed and in the forwarded object, which no
    /// longer carries the client's. An explicit switch overrides the
    /// effort, so an effort of `none` under an enabled object reaches the
    /// frontend as no effort.
    fn reads_thinking_object(&self) -> bool {
        false
    }
    /// Whether decoding is constrained only for a JSON schema response
    /// format marked `strict`, an explicit `structured_outputs` constraint
    /// and a required or named tool choice. A `json_object` response
    /// format, a schema without `strict: true` and a strict tool under
    /// `auto` then leave decoding unconstrained.
    fn constrains_strict_only(&self) -> bool {
        false
    }
    /// Whether `required` or a named tool choice serves as `auto` when the
    /// request offers no tools, rather than being refused.
    fn forced_choice_without_tools_is_auto(&self) -> bool {
        false
    }
    /// The separator of a qualified function name (`namespace`, separator,
    /// name), for a frontend whose prompt format qualifies a function in a
    /// namespace; `None` when names are not qualified. The transport then
    /// reads an offered tool's `namespace` (on the tool, else on its
    /// function: a name, or an object with a `name` and an optional
    /// `description`), names the tool by its qualified name, puts the
    /// namespace description before the function's on its own line, and
    /// forwards the function object so changed without the namespace.
    fn tool_namespace_separator(&self) -> Option<&str> {
        None
    }
    /// The largest prompt-plus-completion token count this model serves.
    fn maximum_total_tokens(&self) -> usize;

    /// The largest tool-call group one response may carry.
    fn maximum_tool_calls(&self) -> usize;

    /// Renders one conversation into the prompt this model expects.
    fn render_chat(
        &self,
        request: FrontendChatRequestV1,
    ) -> Result<FrontendRenderedChatV1, FrontendErrorV1>;

    /// Checks one tool call a backend emitted against this model's grammar.
    ///
    /// The transport already checked that the name is one the request allowed
    /// and that the group is within bounds. This checks what only the model can:
    /// that the name is well formed and the arguments parse as this model
    /// serializes them.
    fn validate_tool_call(&self, name: &str, arguments: &str) -> Result<(), FrontendErrorV1>;

    /// The grammar a tool choice imposes on the output, written in this
    /// model's own tool-call format, or `None` when the choice imposes none.
    ///
    /// A frontend that cannot express forced calls refuses them, which is
    /// what this default does.
    fn tool_constraint(
        &self,
        _tools: &[FrontendToolV1],
        choice: &FrontendToolChoiceV1,
    ) -> Result<Option<runtime_grammar::Grammar>, FrontendErrorV1> {
        match choice {
            FrontendToolChoiceV1::Auto | FrontendToolChoiceV1::None => Ok(None),
            FrontendToolChoiceV1::Required | FrontendToolChoiceV1::Named(_) => {
                Err(FrontendErrorV1::new(
                    "forced tool choices are not supported by this model's template",
                    Some("tool_choice"),
                ))
            }
        }
    }
}
