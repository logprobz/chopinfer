//! Library-direct output against document-path output.
//!
//! The transport's frontend and the host's decoder reach a format through
//! the `chop.frontend.v1` documents. These pairs run the same calls both
//! ways, once calling a model's chat format directly and once through its
//! export's in-process loopback ([`PairedFrontendV1::chat_format`]), assert
//! that every answer agrees, and return the document path's answer. A
//! decoder pair parses with the shared parser called directly and through a
//! chat format's export.

use std::path::Path;
use std::sync::Arc;

use runtime_server::{
    DecodingOptions, FrontendChatRequestV1, FrontendErrorV1, FrontendJsonV1,
    FrontendMediaPartKindV1, FrontendMessageV1, FrontendModelV1, FrontendRenderedChatV1,
    FrontendRoleV1, FrontendToolCallV1, FrontendToolChoiceV1, FrontendToolV1, SamplingDefaults,
    ServerFrontendV1,
};
use serde_json::Value;

use crate::{
    BundleChatStateV1, BundleFrontendV1, BundleParserSourceV1, CompletionDecoderV1,
    CompletionUpdateV1, DecoderLimitsV1, FrontendComponentErrorV1, FrontendDescriptorV1,
    GeneratedTokenV1, PromptModeV1,
};
use frontend_format::test_support::chat::Golden;

/// One frontend twice: a chat format called directly, and through the
/// documents. It serves as the document path, and every render, constraint
/// and tool-call check asserts that both agree.
#[derive(Clone, Debug)]
pub struct PairedFrontendV1 {
    library: BundleFrontendV1,
    document: BundleFrontendV1,
}

impl PairedFrontendV1 {
    /// A model's chat format twice: `F` called directly, and through its
    /// in-process export (`frontend_format::chat_format_export_v1`), both
    /// over `descriptor`, a `format.parser` declaration.
    pub fn chat_format<F: frontend_format::ChatFormatV1>(
        descriptor: FrontendDescriptorV1,
    ) -> Result<Self, FrontendComponentErrorV1> {
        Ok(Self {
            library: BundleFrontendV1::library_chat_format::<F>(descriptor.clone())?,
            document: BundleFrontendV1::with_export(
                descriptor,
                &frontend_format::chat_format_export_v1::<F>(),
            )?,
        })
    }

    /// The document path.
    #[must_use]
    pub const fn document(&self) -> &BundleFrontendV1 {
        &self.document
    }
}

/// Asserts that two rendered chats carry the same prompt and decisions.
fn assert_same_render(
    library: &Result<FrontendRenderedChatV1, FrontendErrorV1>,
    document: &Result<FrontendRenderedChatV1, FrontendErrorV1>,
) {
    match (library, document) {
        (Ok(library), Ok(document)) => {
            assert_eq!(library.prompt, document.prompt, "rendered prompt");
            assert_eq!(library.segments, document.segments, "prompt segments");
            assert_eq!(library.media, document.media, "rendered media");
            assert_eq!(
                library.allowed_function_names, document.allowed_function_names,
                "allowed functions"
            );
            let library = BundleChatStateV1::from_state(&library.state).expect("library state");
            let document = BundleChatStateV1::from_state(&document.state).expect("document state");
            assert_eq!(library.mode(), document.mode(), "mode");
            assert_eq!(
                library.reasoning_effort(),
                document.reasoning_effort(),
                "effort"
            );
            assert_eq!(
                library.reasoning_open(),
                document.reasoning_open(),
                "reasoning open"
            );
            assert_eq!(library.tools(), document.tools(), "tools");
            assert_eq!(
                library.extracts_tool_calls(),
                document.extracts_tool_calls(),
                "tool extraction"
            );
            assert!(library.parser_source().is_none());
            assert!(document.parser_source().is_some());
        }
        (Err(library), Err(document)) => assert_eq!(library, document, "render refusal"),
        (library, document) => panic!(
            "the paths disagree: library {:?}, document {:?}",
            library.as_ref().map(|rendered| &rendered.prompt),
            document.as_ref().map(|rendered| &rendered.prompt)
        ),
    }
}

impl ServerFrontendV1 for PairedFrontendV1 {
    fn models(&self) -> &[FrontendModelV1] {
        self.document.models()
    }

    fn sampling_defaults(&self) -> SamplingDefaults {
        self.document.sampling_defaults()
    }

    fn supports_media(&self) -> bool {
        self.document.supports_media()
    }

    fn served_media(&self) -> Vec<FrontendMediaPartKindV1> {
        self.document.served_media()
    }

    fn has_selected_modalities(&self) -> bool {
        self.document.has_selected_modalities()
    }

    fn allows_remote_media(&self) -> bool {
        self.document.allows_remote_media()
    }

    fn allowed_local_media_path(&self) -> Option<&Path> {
        self.document.allowed_local_media_path()
    }

    fn reasoning_field(&self) -> &'static str {
        self.document.reasoning_field()
    }

    fn tool_call_reasoning_fallback(&self) -> bool {
        self.document.tool_call_reasoning_fallback()
    }

    fn reasoning_content_fallback(&self) -> bool {
        self.document.reasoning_content_fallback()
    }

    fn reads_tool_objects(&self) -> bool {
        self.document.reads_tool_objects()
    }

    fn extra_roles(&self) -> &[String] {
        self.document.extra_roles()
    }

    fn tool_references(&self) -> bool {
        self.document.tool_references()
    }

    fn maximum_total_tokens(&self) -> usize {
        self.document.maximum_total_tokens()
    }

    fn maximum_tool_calls(&self) -> usize {
        self.document.maximum_tool_calls()
    }

    fn render_chat(
        &self,
        request: FrontendChatRequestV1,
    ) -> Result<FrontendRenderedChatV1, FrontendErrorV1> {
        let library = self.library.render_chat(request.clone());
        let document = self.document.render_chat(request);
        assert_same_render(&library, &document);
        document
    }

    fn validate_tool_call(&self, name: &str, arguments: &str) -> Result<(), FrontendErrorV1> {
        let library = self.library.validate_tool_call(name, arguments);
        let document = self.document.validate_tool_call(name, arguments);
        assert_eq!(library, document, "tool call check");
        document
    }

    fn tool_constraint(
        &self,
        tools: &[FrontendToolV1],
        choice: &FrontendToolChoiceV1,
    ) -> Result<Option<runtime_grammar::Grammar>, FrontendErrorV1> {
        let library = self.library.tool_constraint(tools, choice);
        let document = self.document.tool_constraint(tools, choice);
        // A grammar can be megabytes; a disagreement names only its kind.
        let describe =
            |answer: &Result<Option<runtime_grammar::Grammar>, FrontendErrorV1>| match answer {
                Ok(Some(_)) => "a grammar".to_owned(),
                Ok(None) => "no grammar".to_owned(),
                Err(error) => format!("refusal {error:?}"),
            };
        assert!(
            library == document,
            "tool constraint: library {}, document {}",
            describe(&library),
            describe(&document)
        );
        document
    }
}

/// One decoder twice: the shared parser called directly, and one opened
/// through the documents. Every push and finish asserts that both report
/// the same updates or refusal, and returns the document path's.
///
/// A document parser takes nothing after a refusal, which a library parser
/// does, so after the first refusal the pair reports the library path's
/// answers without comparing them.
pub struct PairedDecoderV1 {
    library: CompletionDecoderV1,
    document: CompletionDecoderV1,
    refused: bool,
}

impl PairedDecoderV1 {
    #[must_use]
    pub fn new(
        descriptor: Arc<FrontendDescriptorV1>,
        mode: PromptModeV1,
        limits: DecoderLimitsV1,
    ) -> Self {
        Self::pair(
            CompletionDecoderV1::new(Arc::clone(&descriptor), mode, limits),
            CompletionDecoderV1::new(Arc::clone(&descriptor), mode, limits),
            &descriptor,
        )
    }

    pub fn with_client_stops(
        descriptor: Arc<FrontendDescriptorV1>,
        mode: PromptModeV1,
        limits: DecoderLimitsV1,
        stops: &[String],
    ) -> Result<Self, String> {
        let library =
            CompletionDecoderV1::with_client_stops(Arc::clone(&descriptor), mode, limits, stops);
        let document =
            CompletionDecoderV1::with_client_stops(Arc::clone(&descriptor), mode, limits, stops);
        assert_eq!(library.is_ok(), document.is_ok(), "client stops");
        Ok(Self::pair(library?, document?, &descriptor))
    }

    fn pair(
        library: CompletionDecoderV1,
        mut document: CompletionDecoderV1,
        descriptor: &FrontendDescriptorV1,
    ) -> Self {
        document.set_parser_source(
            BundleParserSourceV1::from_declaration(descriptor)
                .expect("a decoded declaration opens a chat format's export"),
        );
        Self {
            library,
            document,
            refused: false,
        }
    }

    pub fn set_decoding_options(&mut self, options: DecodingOptions, raw: bool) {
        self.library.set_decoding_options(options.clone(), raw);
        self.document.set_decoding_options(options, raw);
    }

    pub fn set_tool_schemas(&mut self, schemas: &[FrontendToolV1]) {
        self.library.set_tool_schemas(schemas);
        self.document.set_tool_schemas(schemas);
    }

    pub fn set_tool_extraction(&mut self, extract: bool) {
        self.library.set_tool_extraction(extract);
        self.document.set_tool_extraction(extract);
    }

    pub fn set_minimum_new_tokens(&mut self, minimum: usize) {
        self.library.set_minimum_new_tokens(minimum);
        self.document.set_minimum_new_tokens(minimum);
    }

    pub fn prime_prompt_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        let library = self.library.prime_prompt_bytes(bytes);
        let document = self.document.prime_prompt_bytes(bytes);
        assert_eq!(library, document, "prompt priming");
        document
    }

    pub fn push(&mut self, token: GeneratedTokenV1) -> Result<Vec<CompletionUpdateV1>, String> {
        let library = self.library.push(token.clone());
        let document = self.document.push(token);
        self.agree(library, document)
    }

    pub fn push_engine_token(
        &mut self,
        token: GeneratedTokenV1,
    ) -> Result<Vec<CompletionUpdateV1>, String> {
        let library = self.library.push_engine_token(token.clone());
        let document = self.document.push_engine_token(token);
        self.agree(library, document)
    }

    pub fn finish_length(&mut self) -> Result<Vec<CompletionUpdateV1>, String> {
        let library = self.library.finish_length();
        let document = self.document.finish_length();
        self.agree(library, document)
    }

    pub fn finish_stop(&mut self) -> Result<Vec<CompletionUpdateV1>, String> {
        let library = self.library.finish_stop();
        let document = self.document.finish_stop();
        self.agree(library, document)
    }

    fn agree(
        &mut self,
        library: Result<Vec<CompletionUpdateV1>, String>,
        document: Result<Vec<CompletionUpdateV1>, String>,
    ) -> Result<Vec<CompletionUpdateV1>, String> {
        if self.refused {
            return library;
        }
        assert_eq!(library, document, "the decoder paths disagree");
        self.refused = document.is_err();
        document
    }
}

/// One captured chat as the transport hands it to `frontend`: messages,
/// tools, and the template keywords, with tool-call arguments as
/// `arguments` writes them and history reasoning read from `reasoning_key`.
/// The client's function objects travel when the frontend's open answer
/// says its render reads them.
#[must_use]
pub fn golden_request(
    frontend: &impl ServerFrontendV1,
    golden: &Golden,
    reasoning_key: &str,
    arguments: fn(&Value) -> String,
) -> FrontendChatRequestV1 {
    let kwargs = &golden.chat_template_kwargs;
    let flag = |name: &str| kwargs.get(name).and_then(Value::as_bool);
    let json = |name: &str| {
        kwargs
            .get(name)
            .filter(|value| !value.is_null())
            .map(|value| serde_json::from_value::<FrontendJsonV1>(value.clone()).unwrap())
    };
    let reads_objects = frontend.reads_tool_objects();
    FrontendChatRequestV1 {
        messages: golden
            .messages
            .iter()
            .map(|message| golden_message(message, reasoning_key, arguments))
            .collect(),
        continue_final_message: false,
        add_generation_prompt: true,
        tools: golden.tools.iter().map(golden_tool).collect(),
        tool_choice: FrontendToolChoiceV1::Auto,
        response_format: None,
        reasoning_effort: None,
        enable_thinking: flag("enable_thinking"),
        thinking: flag("thinking"),
        nested_reasoning_effort: kwargs
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .map(str::to_owned),
        preserve_thinking: flag("preserve_thinking"),
        clear_thinking: json("clear_thinking"),
        add_vision_id: json("add_vision_id"),
        tool_objects: if reads_objects {
            golden
                .tools
                .iter()
                .map(|tool| serde_json::from_value(tool["function"].clone()).unwrap())
                .collect()
        } else {
            Vec::new()
        },
        // The whole object, as the transport forwards it.
        chat_template_kwargs: runtime_server::forwarded_template_kwargs_v1(
            serde_json::from_value(kwargs.clone()).unwrap(),
        ),
    }
}

/// One captured tool, as the transport hands it to a frontend.
fn golden_tool(value: &Value) -> FrontendToolV1 {
    let function = &value["function"];
    FrontendToolV1 {
        name: function["name"].as_str().unwrap().to_owned(),
        description: function["description"].as_str().map(str::to_owned),
        parameters: serde_json::from_value(function["parameters"].clone()).unwrap(),
        parameters_present: !function["parameters"].is_null(),
        strict: function["strict"].as_bool(),
    }
}

fn golden_message(
    value: &Value,
    reasoning_key: &str,
    arguments: fn(&Value) -> String,
) -> FrontendMessageV1 {
    let role = match value["role"].as_str().expect("message role") {
        "system" => FrontendRoleV1::System,
        "developer" => FrontendRoleV1::Developer,
        "user" => FrontendRoleV1::User,
        "assistant" => FrontendRoleV1::Assistant,
        "tool" => FrontendRoleV1::Tool,
        other => panic!("unknown role: {other}"),
    };
    FrontendMessageV1 {
        role,
        content: value["content"].as_str().map(str::to_owned),
        parts: None,
        reasoning_content: value[reasoning_key].as_str().map(str::to_owned),
        tool_calls: value["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|call| FrontendToolCallV1 {
                id: call["id"].as_str().unwrap_or_default().to_owned(),
                name: call["function"]["name"].as_str().unwrap().to_owned(),
                arguments: arguments(&call["function"]["arguments"]),
            })
            .collect(),
        tool_call_id: value["tool_call_id"].as_str().map(str::to_owned),
        tools: value["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|tool| serde_json::from_value(tool.clone()).unwrap())
            .collect(),
    }
}
