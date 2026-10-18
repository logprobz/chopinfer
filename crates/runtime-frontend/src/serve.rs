//! The frontend the transport serves, built from one bundle's declaration.
//!
//! Nothing here names a model. Every answer the transport asks for comes from
//! the installed component: the identities `/v1/models` lists, the context a
//! request is charged against, what the reasoning controls mean, the prompt a
//! conversation renders into, and whether a tool call the backend emitted is
//! one this model could have produced.

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use runtime_bundle_abi::frontend::FrontendExportV1;
use runtime_bundle_abi::frontend_document::OpenedV1;
use runtime_server::{
    ChatStateV1, FrontendChatRequestV1, FrontendErrorV1, FrontendJsonV1, FrontendMediaPartKindV1,
    FrontendModelV1, FrontendRenderedChatV1, FrontendToolChoiceV1, FrontendToolV1, PromptSegmentV1,
    ServerFrontendV1,
};

use crate::checks::{integral_numbers_v1, parse_json_value_v1, validate_name_v1};
use crate::decode::PromptModeV1;
use crate::descriptor::{
    FrontendComponentErrorV1, FrontendConstraintScopeV1, FrontendDescriptorV1, FrontendMediaKindV1,
    FrontendThinkingSwitchV1, FrontendToolArgumentsV1,
};
use crate::format::{FormatRenderRequestV1, FormatRendererV1};
use crate::native::{BundleParserSourceV1, NativeFormatV1};

/// What this frontend decided about one rendered chat.
///
/// The transport reads only whether reasoning is legitimate. A backend paired
/// with the same installed bundle recovers the mode and the effort name from
/// here, because this frontend produced them, and opens the completion's
/// parser from the same frontend.
#[derive(Clone, Debug)]
pub struct BundleChatStateV1 {
    mode: PromptModeV1,
    reasoning_effort: String,
    reasoning_open: bool,
    tools: Vec<runtime_server::FrontendToolV1>,
    extract_tool_calls: bool,
    /// What the prompt ends with that the completion continues: a forced
    /// tool choice's declared prefill, or empty.
    completion_prefix: String,
    /// The frontend that rendered the prompt and the state its render
    /// returned; `None` for a frontend that calls a chat format directly.
    parser: Option<BundleParserSourceV1>,
}

/// Two states are equal when they record the same decisions and parser
/// state, whichever frontend instance made them.
impl PartialEq for BundleChatStateV1 {
    fn eq(&self, other: &Self) -> bool {
        self.mode == other.mode
            && self.reasoning_effort == other.reasoning_effort
            && self.reasoning_open == other.reasoning_open
            && self.tools == other.tools
            && self.extract_tool_calls == other.extract_tool_calls
            && self.completion_prefix == other.completion_prefix
            && self.parser.as_ref().map(BundleParserSourceV1::state)
                == other.parser.as_ref().map(BundleParserSourceV1::state)
    }
}

impl Eq for BundleChatStateV1 {}

impl BundleChatStateV1 {
    #[must_use]
    pub const fn mode(&self) -> PromptModeV1 {
        self.mode
    }

    #[must_use]
    pub fn reasoning_effort(&self) -> &str {
        &self.reasoning_effort
    }

    /// Whether the rendered prompt leaves reasoning open, so the completion
    /// begins in reasoning.
    #[must_use]
    pub const fn reasoning_open(&self) -> bool {
        self.reasoning_open
    }

    #[must_use]
    pub fn tools(&self) -> &[runtime_server::FrontendToolV1] {
        &self.tools
    }

    /// Whether tool calls are read out of the completion.
    ///
    /// Only a request that offers a function and does not choose none can
    /// get a call; otherwise tool-shaped output is markup, as the reference
    /// suppresses extraction.
    #[must_use]
    pub const fn extracts_tool_calls(&self) -> bool {
        self.extract_tool_calls
    }

    /// The text the prompt ends with that the completion's parser and a
    /// forced choice's constraint read as the completion's start: the
    /// declared prefill of a forced tool choice, or empty.
    #[must_use]
    pub fn completion_prefix(&self) -> &str {
        &self.completion_prefix
    }

    /// Where the completion's parser opens: the frontend that rendered the
    /// prompt, with the state its render returned.
    #[must_use]
    pub const fn parser_source(&self) -> Option<&BundleParserSourceV1> {
        self.parser.as_ref()
    }

    /// Recovers this state from the chat metadata the transport carried.
    ///
    /// The state came from this frontend, so a value of another type means the
    /// deployment paired this backend with a different frontend.
    #[must_use]
    pub fn from_state(state: &Arc<dyn ChatStateV1>) -> Option<&Self> {
        state.as_any().downcast_ref::<Self>()
    }
}

impl ChatStateV1 for BundleChatStateV1 {
    fn reasoning_enabled(&self) -> bool {
        self.mode == PromptModeV1::Thinking
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The frontend one installed bundle declared.
#[derive(Clone, Debug)]
pub struct BundleFrontendV1 {
    descriptor: Arc<FrontendDescriptorV1>,
    /// The format, which renders prompts and tool constraints.
    renderer: Arc<dyn FormatRendererV1>,
    /// The same format, which parses completions; `None` when a test's
    /// renderer calls a chat format directly.
    parser: Option<BundleParserSourceV1>,
    /// What the format reads of a request besides the typed fields.
    opened: OpenedV1,
    models: Vec<FrontendModelV1>,
    maximum_total_tokens: usize,
    allowed_local_media_path: Option<PathBuf>,
    remote_media: bool,
    /// The declared media kinds this deployment accepts.
    served_media: Vec<FrontendMediaPartKindV1>,
    selected_modalities: bool,
    /// The most image parts one request may carry, across its messages.
    maximum_images: usize,
}

impl BundleFrontendV1 {
    /// Builds the frontend from one component a host already authenticated,
    /// served by `export`: the package's own frontend, opened over the same
    /// bytes. Every request and answer crosses as the export's documents, and
    /// each answer is checked against the declaration read here.
    pub fn from_export(
        export: &FrontendExportV1,
        bytes: &[u8],
    ) -> Result<Self, FrontendComponentErrorV1> {
        let descriptor = Arc::new(FrontendDescriptorV1::from_component_bytes(bytes)?);
        let format = NativeFormatV1::from_export(export, Arc::clone(&descriptor), Some(bytes))?;
        Ok(Self::with_format(descriptor, Arc::new(format)))
    }

    /// Builds the frontend from one declaration, checking it first, served
    /// by `export` opened over the declaration written back as JSON.
    ///
    /// A descriptor's fields are public, so one can be assembled in code
    /// rather than read from a component. Serving it without checking would
    /// let a deployment reach the renderer with markers it never reserved, so
    /// the same rules a component is held to are enforced here too. A
    /// deployment serves [`Self::from_export`], which opens the export over
    /// the package's own bytes.
    pub fn with_export(
        descriptor: FrontendDescriptorV1,
        export: &FrontendExportV1,
    ) -> Result<Self, FrontendComponentErrorV1> {
        descriptor.validate()?;
        let descriptor = Arc::new(descriptor);
        let format = NativeFormatV1::from_export(export, Arc::clone(&descriptor), None)?;
        Ok(Self::with_format(descriptor, Arc::new(format)))
    }

    fn with_format(descriptor: Arc<FrontendDescriptorV1>, format: Arc<NativeFormatV1>) -> Self {
        let parser = BundleParserSourceV1::new(Arc::clone(&format));
        Self::assemble(descriptor, format, Some(parser))
    }

    /// The frontend over a model's chat format `F` called directly, without
    /// the documents: the library path its export
    /// (`frontend_format::chat_format_export_v1`) is compared against.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn library_chat_format<F: frontend_format::ChatFormatV1>(
        descriptor: FrontendDescriptorV1,
    ) -> Result<Self, FrontendComponentErrorV1> {
        let renderer = Arc::new(crate::loopback::LibraryRendererV1::chat_format::<F>(
            &descriptor,
        )?);
        let descriptor = Arc::new(descriptor);
        Ok(Self::assemble(descriptor, renderer, None))
    }

    fn assemble(
        descriptor: Arc<FrontendDescriptorV1>,
        renderer: Arc<dyn FormatRendererV1>,
        parser: Option<BundleParserSourceV1>,
    ) -> Self {
        let models = descriptor
            .models
            .iter()
            .map(|model| FrontendModelV1::new(model.id.clone(), model.owned_by.clone()))
            .collect();
        let maximum_total_tokens = descriptor.context.maximum_total_tokens;
        let served_media = descriptor
            .multimodal
            .iter()
            .flat_map(|multimodal| &multimodal.modalities)
            .map(|modality| match modality.kind {
                FrontendMediaKindV1::Image => FrontendMediaPartKindV1::Image,
                FrontendMediaKindV1::Audio => FrontendMediaPartKindV1::Audio,
                FrontendMediaKindV1::Video => FrontendMediaPartKindV1::Video,
            })
            .collect();
        let maximum_images = descriptor
            .multimodal
            .iter()
            .flat_map(|multimodal| &multimodal.modalities)
            .find(|modality| modality.kind == FrontendMediaKindV1::Image)
            .map_or(0, |modality| usize::from(modality.maximum_items));
        Self {
            opened: renderer.opened(),
            renderer,
            parser,
            descriptor,
            models,
            maximum_total_tokens,
            allowed_local_media_path: None,
            remote_media: false,
            served_media,
            selected_modalities: false,
            maximum_images,
        }
    }

    /// Where a completion this frontend did not render, such as a raw
    /// completion, opens its parser.
    #[must_use]
    pub fn parser_source(&self) -> Option<BundleParserSourceV1> {
        self.parser.clone()
    }

    /// Serves a stricter context limit than the declared one.
    ///
    /// A deployment may serve less than the bundle declared, never more: a
    /// larger value narrows to the declaration rather than widening past it,
    /// because the bundle is what knows the context this model has.
    #[must_use]
    pub fn with_maximum_total_tokens(mut self, maximum_total_tokens: usize) -> Self {
        self.maximum_total_tokens =
            maximum_total_tokens.min(self.descriptor.context.maximum_total_tokens);
        self
    }

    #[must_use]
    pub fn descriptor(&self) -> &FrontendDescriptorV1 {
        &self.descriptor
    }

    #[must_use]
    pub fn with_remote_media(mut self, enabled: bool) -> Self {
        self.remote_media = enabled;
        self
    }

    /// Accepts only the declared media kinds `kinds` names: a deployment
    /// may serve fewer kinds than the bundle declares (a package without the
    /// media capability), never more.
    #[must_use]
    pub fn with_served_media(mut self, kinds: &[FrontendMediaPartKindV1]) -> Self {
        self.served_media.retain(|kind| kinds.contains(kind));
        self
    }

    /// Record that `--modalities` selected the served modalities, so
    /// refusals can name the flag without attributing package limits to it.
    #[must_use]
    pub fn with_selected_modalities(mut self, selected: bool) -> Self {
        self.selected_modalities = selected;
        self
    }

    /// Accepts at most `maximum` image parts per request instead of the
    /// declared count.
    #[must_use]
    pub fn with_maximum_images(mut self, maximum: usize) -> Self {
        self.maximum_images = maximum;
        self
    }

    #[must_use]
    pub fn with_allowed_local_media_path(mut self, path: PathBuf) -> Self {
        self.allowed_local_media_path = Some(path);
        self
    }
}

impl ServerFrontendV1 for BundleFrontendV1 {
    fn sampling_defaults(&self) -> runtime_server::SamplingDefaults {
        self.descriptor.generation_config
    }

    fn supports_media(&self) -> bool {
        self.descriptor.multimodal.is_some()
    }

    fn served_media(&self) -> Vec<FrontendMediaPartKindV1> {
        self.served_media.clone()
    }

    fn has_selected_modalities(&self) -> bool {
        self.selected_modalities
    }

    fn allows_remote_media(&self) -> bool {
        self.remote_media
    }

    fn allowed_local_media_path(&self) -> Option<&Path> {
        self.allowed_local_media_path.as_deref()
    }

    fn maximum_images_per_request(&self) -> Option<usize> {
        self.served_media
            .contains(&FrontendMediaPartKindV1::Image)
            .then_some(self.maximum_images)
    }
    fn reasoning_field(&self) -> &'static str {
        self.descriptor.response.reasoning_field.name()
    }
    fn tool_call_reasoning_fallback(&self) -> bool {
        self.opened.tool_call_reasoning_fallback
    }
    fn reads_tool_objects(&self) -> bool {
        self.opened.reads_tool_objects
    }
    fn extra_roles(&self) -> &[String] {
        &self.opened.extra_roles
    }
    fn tool_references(&self) -> bool {
        self.opened.tool_references
    }
    fn reasoning_content_fallback(&self) -> bool {
        self.opened.reasoning_content_fallback
    }
    fn reads_thinking_object(&self) -> bool {
        self.descriptor.reasoning.switch == FrontendThinkingSwitchV1::ThinkingObject
    }
    fn constrains_strict_only(&self) -> bool {
        self.descriptor.response.constraint_scope == FrontendConstraintScopeV1::StrictOnly
    }
    fn forced_choice_without_tools_is_auto(&self) -> bool {
        self.descriptor
            .tools
            .choice
            .as_ref()
            .is_some_and(|choice| choice.forced_without_tools_is_auto)
    }
    fn models(&self) -> &[FrontendModelV1] {
        &self.models
    }

    fn maximum_total_tokens(&self) -> usize {
        self.maximum_total_tokens
    }

    fn maximum_tool_calls(&self) -> usize {
        self.descriptor.limits.maximum_tool_calls
    }

    fn render_chat(
        &self,
        request: FrontendChatRequestV1,
    ) -> Result<FrontendRenderedChatV1, FrontendErrorV1> {
        if let Some(multimodal) = &self.descriptor.multimodal {
            for message in &request.messages {
                let strings =
                    message
                        .content
                        .iter()
                        .chain(
                            message
                                .parts
                                .iter()
                                .flatten()
                                .filter_map(|part| match part {
                                    runtime_server::FrontendContentPartV1::Text(text) => Some(text),
                                    _ => None,
                                }),
                        );
                // A tool name a format may write is client text too.
                let references = message
                    .parts
                    .iter()
                    .flatten()
                    .filter_map(|part| match part {
                        runtime_server::FrontendContentPartV1::ToolReference { name } => Some(name),
                        _ => None,
                    });
                for text in strings.chain(references) {
                    if multimodal
                        .modalities
                        .iter()
                        .any(|modality| text.contains(&modality.render.pad))
                    {
                        return Err(FrontendErrorV1::new(
                            "client text contains a media placeholder",
                            Some("messages"),
                        ));
                    }
                }
            }
            // One count over every message, before any item's other checks,
            // so a request with too many images learns that first. Clients
            // read this wording and `param: "image"` as "send fewer images",
            // so no other refusal uses them.
            if let Some(image) = multimodal
                .modalities
                .iter()
                .find(|modality| modality.kind == FrontendMediaKindV1::Image)
            {
                let carried = request
                    .messages
                    .iter()
                    .flat_map(|message| message.parts.iter().flatten())
                    .filter(|part| match part {
                        runtime_server::FrontendContentPartV1::Media(media) => {
                            image.part_types.contains(&media.part_type)
                        }
                        runtime_server::FrontendContentPartV1::Text(_)
                        | runtime_server::FrontendContentPartV1::ToolReference { .. } => false,
                    })
                    .count();
                if carried > self.maximum_images {
                    return Err(FrontendErrorV1::new(
                        format!(
                            "At most {} image(s) may be provided in one request; this request carried {carried}.",
                            self.maximum_images
                        ),
                        Some("image"),
                    ));
                }
            }
            let mut counts = vec![0_usize; multimodal.modalities.len()];
            for media in request
                .messages
                .iter()
                .flat_map(|message| message.parts.iter().flatten())
                .filter_map(|part| match part {
                    runtime_server::FrontendContentPartV1::Media(media) => Some(media),
                    _ => None,
                })
            {
                let index = multimodal
                    .modalities
                    .iter()
                    .position(|entry| entry.part_types.contains(&media.part_type))
                    .ok_or_else(|| {
                        FrontendErrorV1::new(
                            "media part type is not declared by this bundle",
                            Some("messages"),
                        )
                    })?;
                counts[index] += 1;
                let modality = &multimodal.modalities[index];
                if modality.kind != FrontendMediaKindV1::Image
                    && counts[index] > usize::from(modality.maximum_items)
                {
                    return Err(FrontendErrorV1::new(
                        format!(
                            "{} exceeds maximum_items {}",
                            media.part_type, modality.maximum_items
                        ),
                        Some("messages"),
                    ));
                }
                if !media.source.starts_with("file://")
                    && !media.source.starts_with("http://")
                    && !media.source.starts_with("https://")
                    && media.source.len()
                        > modality
                            .maximum_bytes
                            .saturating_mul(4)
                            .saturating_div(3)
                            .saturating_add(256)
                {
                    return Err(FrontendErrorV1::new(
                        format!(
                            "{} exceeds maximum_bytes {}",
                            media.part_type, modality.maximum_bytes
                        ),
                        Some("messages"),
                    ));
                }
                if let Some(path) = media.source.strip_prefix("file://") {
                    let bytes = std::fs::metadata(path)
                        .map_err(|_| {
                            FrontendErrorV1::new(
                                "local media file is unavailable",
                                Some("messages"),
                            )
                        })?
                        .len();
                    if bytes > modality.maximum_bytes as u64 {
                        return Err(FrontendErrorV1::new(
                            format!(
                                "{} exceeds maximum_bytes {}",
                                media.part_type, modality.maximum_bytes
                            ),
                            Some("messages"),
                        ));
                    }
                }
            }
        }
        let FrontendChatRequestV1 {
            messages,
            continue_final_message,
            add_generation_prompt,
            tools,
            tool_choice,
            response_format,
            reasoning_effort,
            enable_thinking,
            thinking,
            nested_reasoning_effort,
            preserve_thinking,
            clear_thinking,
            add_vision_id,
            tool_objects,
            chat_template_kwargs,
        } = request;
        // Every choice renders every offered tool, as the reference template
        // does by default; a forced choice's constraint does the forcing, and
        // none only takes away the calls a completion may make. A declared
        // tool choice policy may omit tools for none, keep only a named
        // choice's tool, and end a forced choice's prompt with a prefill.
        let policy = self.descriptor.tools.choice.as_ref();
        let forced = matches!(
            tool_choice,
            FrontendToolChoiceV1::Required | FrontendToolChoiceV1::Named(_)
        );
        let keep = |tool: &runtime_server::FrontendToolV1| match (&tool_choice, policy) {
            (FrontendToolChoiceV1::None, Some(policy)) => !policy.none_omits_tools,
            (FrontendToolChoiceV1::Named(name), Some(policy))
                if policy.named_renders_its_tool_only =>
            {
                tool.name == *name
            }
            _ => true,
        };
        let kept: Vec<bool> = tools.iter().map(keep).collect();
        let rendered_tools: Vec<runtime_server::FrontendToolV1> = tools
            .iter()
            .zip(&kept)
            .filter(|(_, keep)| **keep)
            .map(|(tool, _)| tool.clone())
            .collect();
        // The function objects sent follow the offered tools one for one.
        let tool_objects = if tool_objects.len() == tools.len() {
            tool_objects
                .into_iter()
                .zip(&kept)
                .filter_map(|(object, keep)| keep.then_some(object))
                .collect()
        } else {
            tool_objects
        };
        let rendered = self.renderer.render(FormatRenderRequestV1 {
            messages,
            continue_final_message,
            add_generation_prompt,
            tools: &rendered_tools,
            tool_objects,
            response_format,
            reasoning_effort,
            enable_thinking,
            thinking,
            nested_reasoning_effort,
            preserve_thinking,
            clear_thinking,
            add_vision_id,
            chat_template_kwargs,
        })?;
        let mut rendered = rendered;
        let mut completion_prefix = String::new();
        if forced && let Some(policy) = policy {
            if policy.forced_refused_in_thinking && rendered.mode == PromptModeV1::Thinking {
                return Err(FrontendErrorV1::new(
                    "thinking mode does not support a required or named tool_choice",
                    Some("tool_choice"),
                ));
            }
            completion_prefix.clone_from(&policy.forced_prefill);
            rendered.prompt.push_str(&completion_prefix);
            match rendered.segments.last_mut() {
                Some(PromptSegmentV1::Text(text)) => text.push_str(&completion_prefix),
                _ if completion_prefix.is_empty() => {}
                _ => rendered
                    .segments
                    .push(PromptSegmentV1::Text(completion_prefix.clone())),
            }
        }
        let extract_tool_calls = tool_choice != FrontendToolChoiceV1::None && !tools.is_empty();
        let allowed_function_names = if extract_tool_calls {
            tools.iter().map(|tool| tool.name.clone()).collect()
        } else {
            Vec::new()
        };
        Ok(FrontendRenderedChatV1 {
            prompt: rendered.prompt,
            segments: rendered.segments,
            media: rendered.media,
            allowed_function_names,
            state: Arc::new(BundleChatStateV1 {
                mode: rendered.mode,
                reasoning_effort: rendered.effort,
                reasoning_open: rendered.reasoning_open,
                tools,
                extract_tool_calls,
                completion_prefix,
                parser: self
                    .parser
                    .as_ref()
                    .map(|parser| parser.with_state(rendered.parser_state)),
            }),
        })
    }

    fn tool_constraint(
        &self,
        tools: &[FrontendToolV1],
        choice: &FrontendToolChoiceV1,
    ) -> Result<Option<runtime_grammar::Grammar>, FrontendErrorV1> {
        self.renderer.tool_constraint(tools, choice)
    }

    fn validate_tool_call(&self, name: &str, arguments: &str) -> Result<(), FrontendErrorV1> {
        validate_name_v1(&self.descriptor, name, "function")
            .map_err(|_| FrontendErrorV1::new("backend tool call name is invalid", None))?;
        let arguments = parse_json_value_v1(arguments, self.descriptor.limits)
            .map_err(|_| FrontendErrorV1::new("backend tool call arguments are invalid", None))?;
        if !matches!(arguments, FrontendJsonV1::Object(_)) {
            return Err(FrontendErrorV1::new(
                "backend tool call arguments must be a JSON object",
                None,
            ));
        }
        match self.descriptor.tools.arguments {
            FrontendToolArgumentsV1::JsonObject => Ok(()),
            FrontendToolArgumentsV1::JsonObjectIntegralNumbers
                if integral_numbers_v1(&arguments) =>
            {
                Ok(())
            }
            FrontendToolArgumentsV1::JsonObjectIntegralNumbers => Err(FrontendErrorV1::new(
                "backend tool call arguments are invalid",
                None,
            )),
        }
    }
}
