//! The one component a bundle ships to declare its OpenAI frontend.
//!
//! The host reads the model-neutral sections: which identities the deployment
//! answers to, how long a context may be, what the reasoning controls mean,
//! which tool names and arguments a call may carry, which token identifiers
//! the vocabulary pins, the tokenizer, the media inputs and the response
//! fields. The chat format itself (the template and the media labels it
//! writes) is the `format` object, which the host bounds and passes to the
//! format implementation unread. So an unchanged host and an unchanged
//! transport serve a different model by installing a different bundle.
//!
//! Parsing is closed: the byte length is bounded before anything is read, the
//! schema and version are checked before the body is deserialized, every field
//! is known, and every declared bound is checked against a ceiling this crate
//! fixes. There is no migration and no default for a missing section.

use std::collections::BTreeSet;
use std::fmt;
use std::str;

use serde::de::{Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::tokenizer::tokenizer_contract_v1;

/// The only schema this crate reads.
pub const FRONTEND_OPENAI_V2_SCHEMA: &str = "chop-frontend-openai-v2";
/// The only version of that schema this crate reads.
pub const FRONTEND_OPENAI_V2_VERSION: u32 = 2;
/// The component role a host resolves this declaration through.
pub const FRONTEND_COMPONENT_ROLE_V1: &str = "chop.frontend.openai.v1";
/// The component role carrying the tokenizer definition this declaration pins.
pub const TOKENIZER_DEFINITION_ROLE_V1: &str = "chop.tokenizer.definition.v1";
/// The component role carrying the tokenizer configuration it pins.
pub const TOKENIZER_CONFIGURATION_ROLE_V1: &str = "chop.tokenizer.configuration.v1";
/// The largest declaration this crate will read at all.
pub const FRONTEND_COMPONENT_MAX_BYTES: usize = 256 * 1024;
/// The largest `format` object, as compact JSON.
pub const FRONTEND_FORMAT_MAX_BYTES: usize = 192 * 1024;

const fn true_v1() -> bool {
    true
}

fn enable_thinking_v1() -> Vec<String> {
    vec!["enable_thinking".to_owned()]
}

const MAX_MODELS: usize = 16;
const MAX_EFFORTS: usize = 16;
const MAX_MARKERS: usize = 128;
const MAX_MARKER_BYTES: usize = 8 * 1024;
const MAX_IDENTIFIER_BYTES: usize = 128;
const MAX_TOTAL_TOKENS: usize = 16 * 1024 * 1024;
const MAX_PROMPT_BYTES: usize = 64 * 1024 * 1024;
/// One media item's encoder payload: the preprocessed rows the host sends,
/// which can exceed the compressed input (a 16 MP image as F32 patch rows
/// is 384 MiB).
const MAX_MEDIA_PAYLOAD_BYTES: usize = 512 * 1024 * 1024;
const MAX_COMPLETION_BYTES: usize = 64 * 1024 * 1024;
const MAX_GENERATED_TOKENS: usize = 16 * 1024 * 1024;
const MAX_CLIENT_STOPS: usize = 16;
const MAX_VOCABULARY_SIZE: u32 = 4 * 1024 * 1024;
const MAX_MESSAGES: usize = 65_536;
const MAX_TOOLS: usize = 4_096;
const MAX_TOOL_CALLS: usize = 4_096;
const MAX_TOOL_PARAMETERS: usize = 4_096;
const MAX_NAME_BYTES: usize = 1_024;
const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;
// The JSON reader refuses more nesting than this on its own, so a declaration
// that asked for more would name a bound nothing could ever reach.
const MAX_JSON_DEPTH: usize = 128;
const MAX_JSON_VALUES: usize = 4 * 1024 * 1024;

/// Why one frontend component was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrontendComponentErrorV1 {
    Empty,
    TooLarge { bytes: usize, maximum: usize },
    NotUtf8,
    Syntax(String),
    Schema { actual: String },
    Version { actual: u32 },
    Invalid(String),
}

impl fmt::Display for FrontendComponentErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("the frontend component is empty"),
            Self::TooLarge { bytes, maximum } => write!(
                formatter,
                "the frontend component is {bytes} bytes and the limit is {maximum}"
            ),
            Self::NotUtf8 => formatter.write_str("the frontend component is not UTF-8"),
            Self::Syntax(message) => write!(formatter, "invalid frontend component: {message}"),
            Self::Schema { actual } => write!(
                formatter,
                "the frontend component declares schema '{actual}' and this host reads only \
                 '{FRONTEND_OPENAI_V2_SCHEMA}'"
            ),
            Self::Version { actual } => write!(
                formatter,
                "the frontend component declares version {actual} and this host reads only \
                 {FRONTEND_OPENAI_V2_VERSION}"
            ),
            Self::Invalid(message) => {
                write!(formatter, "invalid frontend declaration: {message}")
            }
        }
    }
}

impl std::error::Error for FrontendComponentErrorV1 {}

/// One model identity the deployment answers to.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendModelDeclarationV1 {
    pub id: String,
    pub owned_by: String,
}

/// The context this model serves.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendContextPolicyV1 {
    pub maximum_total_tokens: usize,
}

/// One reasoning effort this model accepts, and the preamble it renders.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendReasoningEffortV1 {
    pub name: String,
    pub preamble: String,
}

/// When the template reads the reasoning effort.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrontendEffortUseV1 {
    /// Only while thinking, refusing an effort it does not declare.
    #[default]
    WhenThinking,
    /// Never: any effort leaves the prompt as the default effort renders it.
    Ignored,
}

/// Which request controls switch thinking.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrontendThinkingSwitchV1 {
    /// The `chat_template_kwargs` keywords and the effort, as vLLM passes
    /// them to a template.
    #[default]
    TemplateKeywords,
    /// The top-level `thinking` object, whose `type` is `enabled` (or its
    /// alias `adaptive`) or `disabled`, then the top-level effort: an
    /// explicit object overrides the effort, an effort of `none` disables
    /// thinking and any other enables it. The thinking keywords and the
    /// effort in `chat_template_kwargs` are ignored.
    ThinkingObject,
}

impl FrontendThinkingSwitchV1 {
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// What this model's reasoning controls mean.
///
/// The controls reach the template as vLLM passes them: a top-level effort
/// replaces the nested one, and unless the client sent `enable_thinking`,
/// that keyword is whether the effort differs from `disable_value`. The
/// template switches thinking by the first of `thinking_keywords` the request
/// carries, reads the effort as `effort_use` says, and a request that asks for
/// reasoning without naming an effort gets `default_effort`. The host checks
/// this section; the format interprets the keywords.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendReasoningPolicyV1 {
    pub disable_value: String,
    pub default_effort: String,
    pub efforts: Vec<FrontendReasoningEffortV1>,
    /// Whether an omitted reasoning control selects the default effort.
    #[serde(default)]
    pub default_enabled: bool,
    /// Whether clients may select chat mode by disabling reasoning.
    ///
    /// Missing preserves the original schema rule: disabling is accepted.
    #[serde(default = "true_v1")]
    pub allow_disabled: bool,
    /// The template keywords that switch thinking, first taking precedence.
    #[serde(default = "enable_thinking_v1")]
    pub thinking_keywords: Vec<String>,
    #[serde(default)]
    pub effort_use: FrontendEffortUseV1,
    /// Which request controls switch thinking, which the transport reads
    /// into the keywords above.
    #[serde(default, skip_serializing_if = "FrontendThinkingSwitchV1::is_default")]
    pub switch: FrontendThinkingSwitchV1,
    /// The texts that end a reasoning section, such as its close marker
    /// and a tool-call open that ends it implicitly. Each is a declared
    /// boundary token, so a constraint that waits for reasoning to end waits
    /// for one of their identifiers.
    pub end_markers: Vec<String>,
}

impl FrontendReasoningPolicyV1 {
    #[must_use]
    pub fn effort(&self, name: &str) -> Option<&FrontendReasoningEffortV1> {
        self.efforts.iter().find(|effort| effort.name == name)
    }

    /// The declared efforts as one client-facing list, for a refusal message.
    #[must_use]
    pub fn effort_names(&self) -> Vec<&str> {
        self.efforts
            .iter()
            .map(|effort| effort.name.as_str())
            .collect()
    }
}

/// Which bytes a tool or function name may contain.
///
/// The set is ASCII: the letters and digits when `ascii_alphanumeric` holds,
/// and each byte of `extra`, which are ASCII punctuation, so the set reads
/// back as one bracket expression such as `[A-Za-z0-9_-]`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendNameBytesV1 {
    pub ascii_alphanumeric: bool,
    pub extra: String,
}

impl FrontendNameBytesV1 {
    #[must_use]
    pub fn permits(&self, byte: u8) -> bool {
        (self.ascii_alphanumeric && byte.is_ascii_alphanumeric())
            || self.extra.as_bytes().contains(&byte)
    }

    /// The set as a bracket expression, for a refusal message.
    #[must_use]
    pub fn describe(&self) -> String {
        let letters = if self.ascii_alphanumeric {
            "A-Za-z0-9"
        } else {
            ""
        };
        format!("[{letters}{}]", self.extra)
    }
}

/// The shape a tool call's arguments must take.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrontendToolArgumentsV1 {
    /// A JSON object.
    JsonObject,
    /// A JSON object whose numbers are integral, for a format that cannot
    /// write a fractional number.
    JsonObjectIntegralNumbers,
}

/// What a well-formed tool call looks like for this model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendToolPolicyV1 {
    pub name_bytes: FrontendNameBytesV1,
    pub arguments: FrontendToolArgumentsV1,
    /// How a tool choice changes the render, for a model whose request
    /// mapping says so. Absent, every choice renders every offered tool and
    /// a forced choice's constraint alone does the forcing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choice: Option<FrontendToolChoicePolicyV1>,
}

/// The longest prefill a tool choice policy may declare.
pub const FRONTEND_TOOL_CHOICE_PREFILL_MAX_BYTES: usize = 1024;

/// What the host does around the render for each tool choice. The
/// constraint of a forced choice still does the forcing; a prefill is the
/// opening of the calls it forces, which the prompt then ends with and the
/// completion continues.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendToolChoicePolicyV1 {
    /// Whether `none` renders without the offered tools.
    pub none_omits_tools: bool,
    /// Whether a named choice renders only the tool it names.
    pub named_renders_its_tool_only: bool,
    /// What follows the rendered prompt for `required` and a named choice;
    /// empty for nothing.
    #[serde(default)]
    pub forced_prefill: String,
    /// Whether `required` and a named choice are refused in thinking mode.
    pub forced_refused_in_thinking: bool,
    /// Whether `required` and a named choice serve as `auto` when the
    /// request offers no tools, which the transport otherwise refuses.
    #[serde(default)]
    pub forced_without_tools_is_auto: bool,
}

/// The response field assistant reasoning is returned in, which history
/// turns also carry it in.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontendReasoningFieldV1 {
    Reasoning,
    ReasoningContent,
}

impl FrontendReasoningFieldV1 {
    /// The field's name in a request or response body.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Reasoning => "reasoning",
            Self::ReasoningContent => "reasoning_content",
        }
    }
}

/// How this model's responses read.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendResponseV1 {
    pub reasoning_field: FrontendReasoningFieldV1,
    /// Which requests constrain the response.
    #[serde(default, skip_serializing_if = "FrontendConstraintScopeV1::is_default")]
    pub constraint_scope: FrontendConstraintScopeV1,
}

/// Which requests the transport constrains decoding for.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrontendConstraintScopeV1 {
    /// Every structured output a request asks for, and a forced tool choice
    /// or a strict tool, as vLLM serves them.
    #[default]
    Requested,
    /// Only a JSON schema response format marked `strict`, an explicit
    /// `structured_outputs` constraint, and a required or named tool
    /// choice. A `json_object` response format, a schema without `strict:
    /// true` and a strict tool under `auto` leave decoding unconstrained,
    /// for a model whose vendor API renders them and constrains nothing.
    StrictOnly,
}

impl FrontendConstraintScopeV1 {
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Every bound the renderer charges one request against.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendLimitsV1 {
    pub maximum_prompt_bytes: usize,
    pub maximum_completion_bytes: usize,
    pub maximum_generated_tokens: usize,
    pub maximum_client_stops: usize,
    pub maximum_messages: usize,
    pub maximum_tools: usize,
    pub maximum_tool_calls: usize,
    pub maximum_tool_parameters: usize,
    pub maximum_name_bytes: usize,
    pub maximum_json_bytes: usize,
    pub maximum_json_depth: usize,
    pub maximum_json_values: usize,
}

/// One boundary token, and the identifier the vocabulary gives it.
///
/// A decoder needs both halves: a completion that carries one of these
/// identifiers must decode to exactly this text, or the model emitted a
/// boundary under an identifier that does not mean it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendBoundaryTokenV1 {
    pub text: String,
    pub id: u32,
}

/// The token identifiers this model's vocabulary pins.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendVocabularyV1 {
    pub size: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    /// Every identifier that ends generation, most canonical first.
    pub stop_token_ids: Vec<u32>,
    /// Every boundary this vocabulary gives a single identifier.
    pub boundary_tokens: Vec<FrontendBoundaryTokenV1>,
    /// Exactly the special tokens a rendered prompt may carry.
    ///
    /// Every one of these is also a declared boundary token, so nothing a
    /// prompt is allowed to carry is outside the set a decoder polices.
    pub prompt_special_tokens: Vec<String>,
}

/// One file this declaration pins by exact identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendFileIdentityV1 {
    pub name: String,
    pub byte_length: u64,
    pub sha256: String,
}

/// How the tokenizer definition's normalizer must read.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrontendTokenizerNormalizerV1 {
    /// A `Sequence` normalizer with no members.
    IdentitySequence,
    Nfc,
    /// A JSON `null` normalizer.
    Null,
}

/// The settings one byte-level component of the definition must carry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendByteLevelPolicyV1 {
    pub add_prefix_space: bool,
    pub trim_offsets: bool,
    pub use_regex: bool,
}

/// Which identifier each merge produces.
///
/// The tagged choices that carry no data are empty struct variants rather
/// than unit variants, so a field written beside their tag is refused as
/// unknown instead of being dropped. The same holds for each tagged choice
/// below.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum FrontendMergeResultIdsV1 {
    /// Merge results follow each other in rank order from `first_id`.
    Sequential { first_id: u32 },
    /// Merge results are whatever the vocabulary assigns.
    Vocabulary {},
}

/// How one merge record is written in the definition.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrontendMergeRecordsV1 {
    SpaceSeparatedString,
    StringPairArray,
    StringOrPair,
}

/// How base vocabulary entries spell their bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum FrontendBaseTokensV1 {
    /// Entries below `byte_start_id` are literal text; the rest are byte-level.
    LiteralPrefixThenByteLevel {
        byte_start_id: u32,
    },
    ByteLevel {},
}

/// How the configuration writes a special token.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum FrontendConfigTokensV1 {
    AddedToken { normalized: bool },
    String {},
}

/// What one configuration field must hold.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum FrontendConfigFieldV1<T> {
    Required { value: T },
    RequiredNull {},
    Absent {},
}

/// Which text a required configuration token must carry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum FrontendConfigTokenValueV1 {
    /// The value this tokenizer section declares for the same token.
    ContractValue {},
    Literal {
        text: String,
    },
}

/// Whether the definition's BPE model carries `ignore_merges`, and its value.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrontendIgnoreMergesV1 {
    Absent,
    Disabled,
    /// A pre-tokenized piece that is itself a vocabulary entry skips merging.
    WholePiece,
}

/// The component policies the pinned artifacts are checked and read with.
///
/// Each member is one choice the reader already implements, so a declaration
/// composes a reader this host has rather than describing a parser of its
/// own. Every member is required, and none has a default: an optional member
/// is written as `null` when the artifact must not carry it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendTokenizerPolicyV1 {
    pub normalizer: FrontendTokenizerNormalizerV1,
    pub pretokenizer_byte_level: FrontendByteLevelPolicyV1,
    pub post_processor_byte_level: FrontendByteLevelPolicyV1,
    pub decoder_byte_level: FrontendByteLevelPolicyV1,
    pub merge_results: FrontendMergeResultIdsV1,
    pub merge_records: FrontendMergeRecordsV1,
    pub base_tokens: FrontendBaseTokensV1,
    pub config_tokens: FrontendConfigTokensV1,
    pub add_bos_token: FrontendConfigFieldV1<bool>,
    pub bos_token: FrontendConfigFieldV1<FrontendConfigTokenValueV1>,
    pub unk_token: FrontendConfigFieldV1<FrontendConfigTokenValueV1>,
    pub tokenizer_class: String,
    #[serde(deserialize_with = "Option::deserialize")]
    pub continuing_subword_prefix: Option<String>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub end_of_word_suffix: Option<String>,
    pub ignore_merges: FrontendIgnoreMergesV1,
    #[serde(deserialize_with = "Option::deserialize")]
    pub add_eos_token: Option<bool>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub legacy: Option<bool>,
    /// Whether the configuration carries an empty `sp_model_kwargs` object
    /// (true) or no such field (false).
    pub empty_sentencepiece_kwargs: bool,
}

/// The tokenizer this model is served with.
///
/// The bytes live in their own components, so what a host needs from here is
/// the identity to check them against and the numbers the reader is built
/// with. Nothing is read from a path: a caller hands over bytes the manifest
/// already authenticated, and the reader checks the identity again.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendTokenizerV1 {
    pub repository: String,
    pub revision: String,
    pub definition: FrontendFileIdentityV1,
    pub configuration: FrontendFileIdentityV1,
    pub policy: FrontendTokenizerPolicyV1,
    pub base_vocabulary_size: u32,
    pub merge_count: usize,
    pub added_token_count: usize,
    /// Decodable tokenizer entries when the model carries padded logits rows.
    ///
    /// Missing preserves the original schema rule: tokenizer and logits widths match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocabulary_size: Option<u32>,
    pub model_max_length: usize,
    pub bos_token: Option<String>,
    pub eos_token: String,
    pub pad_token: String,
    pub pretokenizer_patterns: Vec<String>,
}

/// A model-neutral declaration of the media inputs a bundle can encode.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendMultimodalV1 {
    pub arena: FrontendMediaArenaV1,
    pub encoders: Vec<FrontendMediaEncoderV1>,
    pub modalities: Vec<FrontendMediaModalityV1>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FrontendMediaElementV1 {
    Bf16,
    F32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendMediaArenaV1 {
    pub rows: u32,
    pub page_rows: u32,
    pub hidden: u32,
    pub element: FrontendMediaElementV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendMediaEncoderV1 {
    pub name: String,
    pub payload_element: FrontendMediaElementV1,
    pub payload_columns: u32,
    pub maximum_payload_bytes: usize,
    pub cost_per_row_milli: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FrontendMediaKindV1 {
    Image,
    Audio,
    Video,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendMediaGridRenderV1 {
    pub prefix: Vec<String>,
    pub suffix: Vec<String>,
    pub timestamp: String,
}

/// What the host expands for one media item: the pad its rows replace,
/// and for video the grid it writes around each frame group. The labels a
/// format writes around the item are the format's own.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendMediaRenderV1 {
    pub pad: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid: Option<FrontendMediaGridRenderV1>,
}

/// A named media recipe with parameters interpreted by runtime-media.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendMediaRecipeV1 {
    pub tag: String,
    pub parameters: serde_json::Value,
}

/// How the rows of a media span take RoPE positions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FrontendPositionRuleV1 {
    /// Each span row takes (t, h, w) grid coordinates, in merged blocks,
    /// offset by the running position; later rows resume past the grid's
    /// larger side. Requires the `chop.position-axes.v1` bundle capability.
    #[serde(rename = "grid-3axis.v1")]
    Grid3AxisV1,
}

/// How a media item's span holds its encoder rows, when it holds more than
/// them. The encoder writes every row of the span into the arena, so the
/// host splices one pad run of [`FrontendMediaModalityV1::span_rows`] rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FrontendMediaLayoutV1 {
    /// The encoder declaration's `layout.learned-delimiters.v1`: per
    /// temporal grid of `h x w` merged cells, a learned start row, each grid
    /// row's `w` cells followed by a learned newline row, and a learned end
    /// row, `h * (w + 1) + 2` rows. Images only, at sequential positions.
    #[serde(rename = "learned-delimiters.v1")]
    LearnedDelimitersV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendMediaModalityV1 {
    pub kind: FrontendMediaKindV1,
    pub part_types: Vec<String>,
    pub maximum_items: u16,
    pub maximum_bytes: usize,
    pub maximum_tokens: u32,
    pub recipe: FrontendMediaRecipeV1,
    pub encoder: u16,
    pub render: FrontendMediaRenderV1,
    /// Absent: media rows take consecutive 1D positions like text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positions: Option<FrontendPositionRuleV1>,
    /// Absent: an item's span is its encoder rows, one per merged cell.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<FrontendMediaLayoutV1>,
}

impl FrontendMediaModalityV1 {
    /// The rows an item's span takes in the prompt, from its merged grid
    /// `[t, h, w]` (cells after the encoder's merge): `t * h * w`, or under
    /// [`FrontendMediaLayoutV1::LearnedDelimitersV1`] `t * (h * (w + 1) + 2)`.
    /// None on overflow.
    pub fn span_rows(&self, merged: [u64; 3]) -> Option<u64> {
        let [t, h, w] = merged;
        match self.layout {
            None => t.checked_mul(h)?.checked_mul(w),
            Some(FrontendMediaLayoutV1::LearnedDelimitersV1) => {
                t.checked_mul(h.checked_mul(w.checked_add(1)?)?.checked_add(2)?)
            }
        }
    }
}

/// Everything one served model tells the transport about itself.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendDescriptorV1 {
    pub schema: String,
    pub version: u32,
    pub models: Vec<FrontendModelDeclarationV1>,
    pub context: FrontendContextPolicyV1,
    pub limits: FrontendLimitsV1,
    pub reasoning: FrontendReasoningPolicyV1,
    pub tokenizer: FrontendTokenizerV1,
    pub tools: FrontendToolPolicyV1,
    pub vocabulary: FrontendVocabularyV1,
    #[serde(default)]
    pub generation_config: runtime_sampling::SamplingDefaults,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multimodal: Option<FrontendMultimodalV1>,
    pub response: FrontendResponseV1,
    /// The chat format, which only the format implementation reads: an
    /// object of at most [`FRONTEND_FORMAT_MAX_BYTES`] as compact JSON.
    pub format: serde_json::Value,
}

/// Any JSON value, read only to refuse an object that names a member twice
/// at any depth.
struct UniqueMembersV1;

impl<'de> Deserialize<'de> for UniqueMembersV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(UniqueMembersVisitorV1)
    }
}

struct UniqueMembersVisitorV1;

impl<'de> Visitor<'de> for UniqueMembersVisitorV1 {
    type Value = UniqueMembersV1;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, _: bool) -> Result<UniqueMembersV1, E> {
        Ok(UniqueMembersV1)
    }

    fn visit_i64<E>(self, _: i64) -> Result<UniqueMembersV1, E> {
        Ok(UniqueMembersV1)
    }

    fn visit_u64<E>(self, _: u64) -> Result<UniqueMembersV1, E> {
        Ok(UniqueMembersV1)
    }

    fn visit_f64<E>(self, _: f64) -> Result<UniqueMembersV1, E> {
        Ok(UniqueMembersV1)
    }

    fn visit_str<E>(self, _: &str) -> Result<UniqueMembersV1, E> {
        Ok(UniqueMembersV1)
    }

    fn visit_unit<E>(self) -> Result<UniqueMembersV1, E> {
        Ok(UniqueMembersV1)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut items: A) -> Result<UniqueMembersV1, A::Error> {
        while items.next_element::<UniqueMembersV1>()?.is_some() {}
        Ok(UniqueMembersV1)
    }

    // A lossless number arrives as a one-member object, which passes.
    fn visit_map<A: MapAccess<'de>>(self, mut members: A) -> Result<UniqueMembersV1, A::Error> {
        let mut seen = BTreeSet::new();
        while let Some(name) = members.next_key::<String>()? {
            if !seen.insert(name.clone()) {
                return Err(A::Error::custom(format!("duplicate member `{name}`")));
            }
            members.next_value::<UniqueMembersV1>()?;
        }
        Ok(UniqueMembersV1)
    }
}

/// The prefix that decides whether the body is worth deserializing at all.
#[derive(Deserialize)]
struct FrontendHeadV1 {
    schema: String,
    version: u32,
}

impl FrontendDescriptorV1 {
    /// Reads one frontend component a host already authenticated.
    ///
    /// The caller supplies bytes it verified against the installed bundle's
    /// manifest. This adds no second lookup and reads nothing else.
    pub fn from_component_bytes(bytes: &[u8]) -> Result<Self, FrontendComponentErrorV1> {
        if bytes.is_empty() {
            return Err(FrontendComponentErrorV1::Empty);
        }
        if bytes.len() > FRONTEND_COMPONENT_MAX_BYTES {
            return Err(FrontendComponentErrorV1::TooLarge {
                bytes: bytes.len(),
                maximum: FRONTEND_COMPONENT_MAX_BYTES,
            });
        }
        Self::from_declaration_bytes(bytes)
    }

    /// Reads a declaration as [`Self::from_component_bytes`] does, without
    /// the component file's size cap: for bytes written back from a
    /// descriptor that was already read and validated, whose JSON may be
    /// longer than the file it came from (serialization restores omitted
    /// defaults). Every other check applies.
    pub fn from_declaration_bytes(bytes: &[u8]) -> Result<Self, FrontendComponentErrorV1> {
        if bytes.is_empty() {
            return Err(FrontendComponentErrorV1::Empty);
        }
        let text = str::from_utf8(bytes).map_err(|_| FrontendComponentErrorV1::NotUtf8)?;
        // The schema and version decide whether this crate may read the body at
        // all, so they are checked before any field of it is deserialized.
        let head: FrontendHeadV1 = serde_json::from_str(text)
            .map_err(|error| FrontendComponentErrorV1::Syntax(error.to_string()))?;
        if head.schema != FRONTEND_OPENAI_V2_SCHEMA {
            return Err(FrontendComponentErrorV1::Schema {
                actual: head.schema,
            });
        }
        if head.version != FRONTEND_OPENAI_V2_VERSION {
            return Err(FrontendComponentErrorV1::Version {
                actual: head.version,
            });
        }
        // The typed sections refuse a member named twice, but the opaque ones
        // read into a `Value`, which keeps only the last copy; so the text is
        // checked whole first.
        serde_json::from_str::<UniqueMembersV1>(text)
            .map_err(|error| FrontendComponentErrorV1::Syntax(error.to_string()))?;
        let descriptor: Self = serde_json::from_str(text)
            .map_err(|error| FrontendComponentErrorV1::Syntax(error.to_string()))?;
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Checks every rule this schema fixes.
    ///
    /// Reading a component runs this, and so does building a frontend, because
    /// the fields are public and a caller can assemble a descriptor that never
    /// came from a component at all.
    pub fn validate(&self) -> Result<(), FrontendComponentErrorV1> {
        self.generation_config
            .validate()
            .map_err(|error| invalid(format!("invalid generation config: {error}")))?;
        self.validate_models()?;
        self.validate_context()?;
        self.validate_limits()?;
        self.validate_vocabulary()?;
        self.validate_reasoning()?;
        self.validate_tools()?;
        self.validate_tokenizer()?;
        self.validate_multimodal()?;
        self.validate_format()
    }

    /// The format is the format implementation's to read; the host only
    /// bounds it.
    fn validate_format(&self) -> Result<(), FrontendComponentErrorV1> {
        if !self.format.is_object() {
            return Err(invalid("format must be an object"));
        }
        let bytes = serde_json::to_vec(&self.format)
            .map_err(|error| invalid(format!("format does not serialize: {error}")))?
            .len();
        if bytes > FRONTEND_FORMAT_MAX_BYTES {
            return Err(invalid(format!(
                "format is {bytes} bytes and the limit is {FRONTEND_FORMAT_MAX_BYTES}"
            )));
        }
        Ok(())
    }

    fn validate_tools(&self) -> Result<(), FrontendComponentErrorV1> {
        let extra = &self.tools.name_bytes.extra;
        // Each extra byte is punctuation a bracket expression writes as
        // itself: no bracket, caret or backslash, and a hyphen only last.
        for (index, byte) in extra.bytes().enumerate() {
            if !byte.is_ascii_punctuation()
                || matches!(byte, b'[' | b']' | b'^' | b'\\')
                || (byte == b'-' && index + 1 != extra.len())
                || extra.as_bytes()[..index].contains(&byte)
            {
                return Err(invalid(format!(
                    "tool name byte set extra '{extra}' must be distinct ASCII punctuation \
                     other than [, ], ^ and \\, with any - last"
                )));
            }
        }
        if !self.tools.name_bytes.ascii_alphanumeric && extra.is_empty() {
            return Err(invalid("the tool name byte set is empty"));
        }
        if let Some(choice) = &self.tools.choice {
            if choice.forced_prefill.len() > FRONTEND_TOOL_CHOICE_PREFILL_MAX_BYTES {
                return Err(invalid(format!(
                    "tools.choice forced_prefill is {} bytes and the limit is \
                     {FRONTEND_TOOL_CHOICE_PREFILL_MAX_BYTES}",
                    choice.forced_prefill.len()
                )));
            }
            // A prefill opens calls after the answer begins, which a prompt
            // that leaves reasoning open has not reached.
            if !choice.forced_prefill.is_empty() && !choice.forced_refused_in_thinking {
                return Err(invalid(
                    "tools.choice forced_prefill needs forced_refused_in_thinking",
                ));
            }
        }
        Ok(())
    }

    fn validate_multimodal(&self) -> Result<(), FrontendComponentErrorV1> {
        let Some(media) = &self.multimodal else {
            return Ok(());
        };
        if media.arena.element != FrontendMediaElementV1::Bf16
            || media.arena.rows == 0
            || media.arena.page_rows == 0
            || media.arena.hidden == 0
            || media.arena.rows % media.arena.page_rows != 0
            || media.arena.rows > 1_048_576
        {
            return Err(invalid("invalid multimodal arena shape or element"));
        }
        if media.encoders.is_empty()
            || media.encoders.len() > 32
            || media.modalities.is_empty()
            || media.modalities.len() > 3
        {
            return Err(invalid(
                "multimodal encoder or modality count is out of range",
            ));
        }
        let mut names = BTreeSet::new();
        for encoder in &media.encoders {
            bounded_identifier(&encoder.name, "media encoder name")?;
            if !names.insert(encoder.name.as_str())
                || encoder.payload_columns == 0
                || encoder.maximum_payload_bytes == 0
                || encoder.maximum_payload_bytes > MAX_MEDIA_PAYLOAD_BYTES
                || encoder.cost_per_row_milli == 0
            {
                return Err(invalid("invalid or duplicate media encoder"));
            }
        }
        let mut kinds = BTreeSet::new();
        let mut pads = BTreeSet::new();
        for modality in &media.modalities {
            let (kind, tags, allowed) = match modality.kind {
                FrontendMediaKindV1::Image => (
                    "image",
                    &[
                        "image.patch-grid.v1",
                        "image.zero-padded-token-budget.v1",
                        "image.centered-pad.v1",
                    ][..],
                    &["image_url"][..],
                ),
                FrontendMediaKindV1::Audio => (
                    "audio",
                    &["audio.log-mel.v1"][..],
                    &["input_audio", "audio_url"][..],
                ),
                FrontendMediaKindV1::Video => (
                    "video",
                    &["video.timestamped-grids.v1"][..],
                    &["video_url"][..],
                ),
            };
            if !kinds.insert(kind)
                || modality.encoder as usize >= media.encoders.len()
                || modality.maximum_items == 0
                || modality.maximum_bytes == 0
                || modality.maximum_bytes > MAX_PROMPT_BYTES
                || modality.maximum_tokens == 0
                || modality.maximum_tokens as usize > self.context.maximum_total_tokens
                || modality.part_types.is_empty()
                || modality.part_types.len() > allowed.len()
                || modality
                    .part_types
                    .iter()
                    .any(|part| !allowed.contains(&part.as_str()))
                || !tags.contains(&modality.recipe.tag.as_str())
                || !modality.recipe.parameters.is_object()
            {
                return Err(invalid(format!("invalid {kind} modality declaration")));
            }
            if modality.recipe.tag == "image.centered-pad.v1" {
                modality.validate_centered_pad_v1().map_err(invalid)?;
                if media
                    .modalities
                    .iter()
                    .any(|m| m.kind == FrontendMediaKindV1::Video)
                {
                    return Err(invalid("image.centered-pad.v1 has no video form"));
                }
            }
            let pad = &modality.render.pad;
            bounded_marker(pad, "media marker")?;
            if pad.is_empty() || !self.vocabulary.prompt_special_tokens.contains(pad) {
                return Err(invalid(format!(
                    "media marker '{pad}' is not a prompt special token"
                )));
            }
            if !pads.insert(modality.render.pad.as_str()) {
                return Err(invalid("media pad markers must be distinct"));
            }
            if let Some(grid) = &modality.render.grid {
                if modality.kind != FrontendMediaKindV1::Video || grid.timestamp != "mm:ss" {
                    return Err(invalid("invalid media grid rendering"));
                }
                for token in grid.prefix.iter().chain(grid.suffix.iter()) {
                    bounded_marker(token, "media grid marker")?;
                    if !self.vocabulary.prompt_special_tokens.contains(token) {
                        return Err(invalid("media grid marker is not a prompt special token"));
                    }
                }
            } else if modality.kind == FrontendMediaKindV1::Video {
                return Err(invalid("video modality requires grid rendering"));
            }
            if modality.positions == Some(FrontendPositionRuleV1::Grid3AxisV1)
                && (modality.kind != FrontendMediaKindV1::Image
                    || !modality
                        .recipe
                        .parameters
                        .get("merge")
                        .and_then(serde_json::Value::as_u64)
                        .is_some_and(|merge| (1..=u64::from(u16::MAX)).contains(&merge)))
            {
                return Err(invalid(
                    "grid-3axis.v1 positions need an image modality with a positive merge",
                ));
            }
            if modality.layout == Some(FrontendMediaLayoutV1::LearnedDelimitersV1)
                && (modality.kind != FrontendMediaKindV1::Image || modality.positions.is_some())
            {
                return Err(invalid(
                    "learned-delimiters.v1 spans need an image modality at sequential positions",
                ));
            }
        }
        Ok(())
    }

    /// Checks the tokenizer this declaration pins.
    ///
    /// The rules split in two. The bounds this schema fixes are checked here,
    /// because they are about what a declaration may say. Everything else is
    /// the reader's own rule, and rather than restate any of it this builds the
    /// exact contract the reader would build. Restating it is how the two
    /// drifted before: a declaration read here and then failed at deployment,
    /// with the bundle already selected and nothing to fall back to.
    fn validate_tokenizer(&self) -> Result<(), FrontendComponentErrorV1> {
        let tokenizer = &self.tokenizer;
        bounded_identifier(&tokenizer.repository, "tokenizer repository")?;
        for (file, name) in [
            (&tokenizer.definition, "tokenizer definition"),
            (&tokenizer.configuration, "tokenizer configuration"),
        ] {
            bounded_identifier(&file.name, name)?;
        }
        if tokenizer.definition.name == tokenizer.configuration.name {
            return Err(invalid(
                "the tokenizer definition and configuration must be different files",
            ));
        }
        if tokenizer.pretokenizer_patterns.len() > MAX_MARKERS {
            return Err(invalid(format!(
                "{} pretokenizer patterns exceed the limit of {MAX_MARKERS}",
                tokenizer.pretokenizer_patterns.len()
            )));
        }
        for pattern in &tokenizer.pretokenizer_patterns {
            if pattern.is_empty() {
                return Err(invalid("a pretokenizer pattern cannot be empty"));
            }
            bounded_marker(pattern, "pretokenizer pattern")?;
        }
        for (token, name) in [
            (Some(&tokenizer.eos_token), "eos_token"),
            (Some(&tokenizer.pad_token), "pad_token"),
            (tokenizer.bos_token.as_ref(), "bos_token"),
        ] {
            let Some(token) = token else { continue };
            if token.is_empty() {
                return Err(invalid(format!("{name} cannot be empty")));
            }
            bounded_marker(token, name)?;
        }
        let policy = &tokenizer.policy;
        bounded_identifier(&policy.tokenizer_class, "tokenizer_class")?;
        for (affix, name) in [
            (
                &policy.continuing_subword_prefix,
                "continuing_subword_prefix",
            ),
            (&policy.end_of_word_suffix, "end_of_word_suffix"),
        ] {
            if let Some(affix) = affix {
                bounded_marker(affix, name)?;
            }
        }
        for (field, name) in [
            (&policy.bos_token, "bos_token policy"),
            (&policy.unk_token, "unk_token policy"),
        ] {
            if let FrontendConfigFieldV1::Required {
                value: FrontendConfigTokenValueV1::Literal { text },
            } = field
            {
                if text.is_empty() {
                    return Err(invalid(format!("{name} literal cannot be empty")));
                }
                bounded_marker(text, name)?;
            }
        }
        // A deployment may serve less context than its tokenizer can read, as
        // one that reserves headroom does. Serving more would advertise a
        // context the tokenizer refuses the moment a client fills it.
        if self.context.maximum_total_tokens > tokenizer.model_max_length {
            return Err(invalid(format!(
                "maximum_total_tokens {} exceeds the tokenizer's model_max_length {}",
                self.context.maximum_total_tokens, tokenizer.model_max_length
            )));
        }
        // The reader reads this declaration and nothing else, so building its
        // contract here cannot lead back into validation.
        tokenizer_contract_v1(self).map_err(FrontendComponentErrorV1::Invalid)?;
        Ok(())
    }

    fn validate_models(&self) -> Result<(), FrontendComponentErrorV1> {
        if self.schema != FRONTEND_OPENAI_V2_SCHEMA {
            return Err(FrontendComponentErrorV1::Schema {
                actual: self.schema.clone(),
            });
        }
        if self.version != FRONTEND_OPENAI_V2_VERSION {
            return Err(FrontendComponentErrorV1::Version {
                actual: self.version,
            });
        }
        if self.models.is_empty() {
            return Err(invalid("at least one model identity is required"));
        }
        if self.models.len() > MAX_MODELS {
            return Err(invalid(format!(
                "{} model identities exceed the limit of {MAX_MODELS}",
                self.models.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for model in &self.models {
            bounded_identifier(&model.id, "model id")?;
            bounded_identifier(&model.owned_by, "model owner")?;
            if !seen.insert(model.id.as_str()) {
                return Err(invalid(format!(
                    "two model identities share the id '{}'",
                    model.id
                )));
            }
        }
        Ok(())
    }

    fn validate_context(&self) -> Result<(), FrontendComponentErrorV1> {
        bounded_count(
            self.context.maximum_total_tokens,
            MAX_TOTAL_TOKENS,
            "maximum_total_tokens",
        )
    }

    fn validate_limits(&self) -> Result<(), FrontendComponentErrorV1> {
        let limits = self.limits;
        bounded_count(
            limits.maximum_prompt_bytes,
            MAX_PROMPT_BYTES,
            "maximum_prompt_bytes",
        )?;
        bounded_count(
            limits.maximum_completion_bytes,
            MAX_COMPLETION_BYTES,
            "maximum_completion_bytes",
        )?;
        bounded_count(
            limits.maximum_generated_tokens,
            MAX_GENERATED_TOKENS,
            "maximum_generated_tokens",
        )?;
        bounded_count(
            limits.maximum_client_stops,
            MAX_CLIENT_STOPS,
            "maximum_client_stops",
        )?;
        bounded_count(limits.maximum_messages, MAX_MESSAGES, "maximum_messages")?;
        bounded_count(limits.maximum_tools, MAX_TOOLS, "maximum_tools")?;
        bounded_count(
            limits.maximum_tool_calls,
            MAX_TOOL_CALLS,
            "maximum_tool_calls",
        )?;
        bounded_count(
            limits.maximum_tool_parameters,
            MAX_TOOL_PARAMETERS,
            "maximum_tool_parameters",
        )?;
        bounded_count(
            limits.maximum_name_bytes,
            MAX_NAME_BYTES,
            "maximum_name_bytes",
        )?;
        bounded_count(
            limits.maximum_json_bytes,
            MAX_JSON_BYTES,
            "maximum_json_bytes",
        )?;
        bounded_count(
            limits.maximum_json_depth,
            MAX_JSON_DEPTH,
            "maximum_json_depth",
        )?;
        bounded_count(
            limits.maximum_json_values,
            MAX_JSON_VALUES,
            "maximum_json_values",
        )
    }

    fn validate_reasoning(&self) -> Result<(), FrontendComponentErrorV1> {
        let reasoning = &self.reasoning;
        bounded_identifier(&reasoning.disable_value, "disable_value")?;
        bounded_identifier(&reasoning.default_effort, "default_effort")?;
        if reasoning.efforts.is_empty() {
            return Err(invalid("at least one reasoning effort is required"));
        }
        if reasoning.efforts.len() > MAX_EFFORTS {
            return Err(invalid(format!(
                "{} reasoning efforts exceed the limit of {MAX_EFFORTS}",
                reasoning.efforts.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for effort in &reasoning.efforts {
            bounded_identifier(&effort.name, "reasoning effort name")?;
            bounded_marker(&effort.preamble, "reasoning effort preamble")?;
            if effort.name == reasoning.disable_value {
                return Err(invalid(format!(
                    "reasoning effort '{}' collides with the value that disables reasoning",
                    effort.name
                )));
            }
            if !seen.insert(effort.name.as_str()) {
                return Err(invalid(format!(
                    "two reasoning efforts share the name '{}'",
                    effort.name
                )));
            }
        }
        if reasoning.thinking_keywords.len() > MAX_EFFORTS {
            return Err(invalid(format!(
                "{} thinking keywords exceed the limit of {MAX_EFFORTS}",
                reasoning.thinking_keywords.len()
            )));
        }
        for (index, keyword) in reasoning.thinking_keywords.iter().enumerate() {
            bounded_identifier(keyword, "thinking keyword")?;
            if reasoning.thinking_keywords[..index].contains(keyword) {
                return Err(invalid(format!(
                    "thinking keyword {keyword:?} is declared twice"
                )));
            }
        }
        if reasoning.switch == FrontendThinkingSwitchV1::ThinkingObject
            && reasoning.thinking_keywords.is_empty()
        {
            return Err(invalid(
                "reasoning switch thinking-object needs a thinking keyword to carry it",
            ));
        }
        if reasoning.end_markers.len() > MAX_MARKERS {
            return Err(invalid(format!(
                "{} reasoning end markers exceed the limit of {MAX_MARKERS}",
                reasoning.end_markers.len()
            )));
        }
        for (index, marker) in reasoning.end_markers.iter().enumerate() {
            if !self
                .vocabulary
                .boundary_tokens
                .iter()
                .any(|token| token.text == *marker)
            {
                return Err(invalid(format!(
                    "reasoning end marker '{marker}' is not a declared boundary token"
                )));
            }
            if reasoning.end_markers[..index].contains(marker) {
                return Err(invalid(format!(
                    "reasoning end marker '{marker}' is declared twice"
                )));
            }
        }
        if reasoning.effort(&reasoning.default_effort).is_none() {
            return Err(invalid(format!(
                "default reasoning effort '{}' is not a declared effort",
                reasoning.default_effort
            )));
        }
        Ok(())
    }

    fn validate_vocabulary(&self) -> Result<(), FrontendComponentErrorV1> {
        let vocabulary = &self.vocabulary;
        if vocabulary.size == 0 || vocabulary.size > MAX_VOCABULARY_SIZE {
            return Err(invalid(format!(
                "vocabulary size {} is not in 1..={MAX_VOCABULARY_SIZE}",
                vocabulary.size
            )));
        }
        for (id, name) in [
            (vocabulary.bos_token_id, "bos_token_id"),
            (vocabulary.eos_token_id, "eos_token_id"),
        ] {
            if id >= vocabulary.size {
                return Err(invalid(format!(
                    "{name} {id} is outside the vocabulary of {}",
                    vocabulary.size
                )));
            }
        }
        if vocabulary.stop_token_ids.is_empty() {
            return Err(invalid("at least one stop token identifier is required"));
        }
        if vocabulary.stop_token_ids.len() > MAX_MARKERS {
            return Err(invalid(format!(
                "{} stop token identifiers exceed the limit of {MAX_MARKERS}",
                vocabulary.stop_token_ids.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for id in &vocabulary.stop_token_ids {
            if *id >= vocabulary.size {
                return Err(invalid(format!(
                    "stop token {id} is outside the vocabulary of {}",
                    vocabulary.size
                )));
            }
            if !seen.insert(*id) {
                return Err(invalid(format!("stop token {id} is declared twice")));
            }
        }
        if vocabulary.boundary_tokens.is_empty() {
            return Err(invalid("at least one boundary token is required"));
        }
        if vocabulary.boundary_tokens.len() > MAX_MARKERS {
            return Err(invalid(format!(
                "{} boundary tokens exceed the limit of {MAX_MARKERS}",
                vocabulary.boundary_tokens.len()
            )));
        }
        let mut texts = BTreeSet::new();
        let mut ids = BTreeSet::new();
        for boundary in &vocabulary.boundary_tokens {
            if boundary.text.is_empty() {
                return Err(invalid("a boundary token cannot be empty"));
            }
            bounded_marker(&boundary.text, "boundary token")?;
            if boundary.id >= vocabulary.size {
                return Err(invalid(format!(
                    "boundary token '{}' has identifier {} outside the vocabulary of {}",
                    boundary.text, boundary.id, vocabulary.size
                )));
            }
            if !texts.insert(boundary.text.as_str()) {
                return Err(invalid(format!(
                    "boundary token '{}' is declared twice",
                    boundary.text
                )));
            }
            if !ids.insert(boundary.id) {
                return Err(invalid(format!(
                    "boundary token identifier {} is declared twice",
                    boundary.id
                )));
            }
        }
        if vocabulary.prompt_special_tokens.len() > MAX_MARKERS {
            return Err(invalid(format!(
                "{} prompt special tokens exceed the limit of {MAX_MARKERS}",
                vocabulary.prompt_special_tokens.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for token in &vocabulary.prompt_special_tokens {
            if token.is_empty() {
                return Err(invalid("a prompt special token cannot be empty"));
            }
            bounded_marker(token, "prompt special token")?;
            if !seen.insert(token.as_str()) {
                return Err(invalid(format!(
                    "prompt special token '{token}' is declared twice"
                )));
            }
            // A prompt may carry only boundaries this vocabulary names, so a
            // decoder can always check the identifier one arrives under.
            if !texts.contains(token.as_str()) {
                return Err(invalid(format!(
                    "prompt special token '{token}' is not a declared boundary token"
                )));
            }
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> FrontendComponentErrorV1 {
    FrontendComponentErrorV1::Invalid(message.into())
}

fn bounded_identifier(value: &str, name: &str) -> Result<(), FrontendComponentErrorV1> {
    if value.is_empty() {
        return Err(invalid(format!("{name} cannot be empty")));
    }
    if value.len() > MAX_IDENTIFIER_BYTES {
        return Err(invalid(format!(
            "{name} is {} bytes and the limit is {MAX_IDENTIFIER_BYTES}",
            value.len()
        )));
    }
    Ok(())
}

fn bounded_marker(value: &str, name: &str) -> Result<(), FrontendComponentErrorV1> {
    if value.len() > MAX_MARKER_BYTES {
        return Err(invalid(format!(
            "{name} is {} bytes and the limit is {MAX_MARKER_BYTES}",
            value.len()
        )));
    }
    Ok(())
}

fn bounded_count(value: usize, maximum: usize, name: &str) -> Result<(), FrontendComponentErrorV1> {
    if value == 0 || value > maximum {
        return Err(invalid(format!("{name} {value} is not in 1..={maximum}")));
    }
    Ok(())
}
