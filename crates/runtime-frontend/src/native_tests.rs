//! Conformance of the host half: a fake bundle frontend answers each kind of
//! bad document, and the host refuses every one.

use runtime_bundle_abi::frontend::{FrontendParserV1, FrontendV1};

use super::*;
use crate::decode::{
    CompletionDecoderV1, CompletionUpdateV1, DecoderLimitsV1, GeneratedTokenV1, test_common,
};
use crate::descriptor::{
    FrontendMediaArenaV1, FrontendMediaElementV1, FrontendMediaEncoderV1, FrontendMediaModalityV1,
    FrontendMediaRecipeV1, FrontendMediaRenderV1, FrontendMultimodalV1,
};

/// A bundle frontend whose component names the answers it gives.
struct FakeFrontendV1 {
    scenario: String,
}

struct FakeParserV1 {
    scenario: String,
    pushes: usize,
}

impl FrontendV1 for FakeFrontendV1 {
    type Parser = FakeParserV1;

    fn open(component: &[u8]) -> Result<(Self, Vec<u8>), Vec<u8>> {
        let scenario = String::from_utf8(component.to_vec()).unwrap();
        let opened = match scenario.as_str() {
            "malformed-open" => b"opened".to_vec(),
            "reads-everything" => document::OpenedV1 {
                reads_tool_objects: true,
                reasoning_content_fallback: true,
                tool_call_reasoning_fallback: true,
                extra_roles: Vec::new(),
                tool_references: true,
            }
            .encode(),
            _ => match scenario.strip_prefix("roles:") {
                // Declares the comma-separated roles it names.
                Some(roles) => document::OpenedV1 {
                    extra_roles: roles.split(',').map(str::to_owned).collect(),
                    ..document::OpenedV1::default()
                }
                .encode(),
                None => document::OpenedV1::default().encode(),
            },
        };
        Ok((Self { scenario }, opened))
    }

    fn render(&self, request: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
        if self.scenario == "echo-kwargs" {
            // Refuses with the template keywords document it received.
            let request = document::RenderRequestV1::decode(request).unwrap();
            let kwargs = request
                .chat_template_kwargs
                .unwrap_or_else(|| "none".into());
            return Err(document::RefusalV1::new(kwargs, None).encode());
        }
        if self.scenario.starts_with("roles:") {
            // Refuses with the roles of the messages it received.
            let request = document::RenderRequestV1::decode(request).unwrap();
            let roles: Vec<_> = request
                .messages
                .iter()
                .map(|message| message.role.as_str())
                .collect();
            return Err(document::RefusalV1::new(roles.join(","), None).encode());
        }
        let rendered = |segments: Vec<document::SegmentV1>, effort: &str| {
            document::RenderedV1 {
                segments,
                mode: document::ModeV1::Thinking,
                effort: effort.into(),
                reasoning_open: true,
                parser_state: vec![9],
            }
            .encode()
        };
        let text = |text: &str| document::SegmentV1::Text(text.into());
        let media = document::SegmentV1::Media;
        let limit = descriptor().limits.maximum_prompt_bytes;
        let pads = |count: usize| count * "[A]".len();
        Ok(match self.scenario.as_str() {
            "good" => rendered(
                vec![text("x"), media(0), text("y"), media(1), text("z")],
                "high",
            ),
            "media-twice" => rendered(vec![media(0), media(0)], "low"),
            "media-out-of-order" => rendered(vec![media(1), media(0)], "low"),
            "media-dropped" => rendered(vec![text("x"), media(1), text("z")], "low"),
            "media-extra" => rendered(vec![media(0), media(1), media(2)], "low"),
            "undeclared-effort" => rendered(vec![media(0), media(1)], "extreme"),
            "media-prompt-at-the-limit" => rendered(
                vec![text(&"p".repeat(limit - pads(2))), media(0), media(1)],
                "low",
            ),
            "media-prompt-past-the-limit" => rendered(
                vec![text(&"p".repeat(limit - pads(2) + 1)), media(0), media(1)],
                "low",
            ),
            "prompt-at-the-limit" => rendered(vec![text(&"p".repeat(limit))], "low"),
            "prompt-past-the-limit" => rendered(vec![text(&"p".repeat(limit + 1))], "low"),
            "prompt-past-the-sink" => rendered(vec![text(&"p".repeat(2 << 20))], "low"),
            "state-past-the-limit" => document::RenderedV1 {
                segments: vec![],
                mode: document::ModeV1::Chat,
                effort: "low".into(),
                reasoning_open: false,
                parser_state: vec![9; document::FRONTEND_PARSER_STATE_BYTES_MAX_V1 + 1],
            }
            .encode(),
            "reasoning-open-in-chat" => document::RenderedV1 {
                segments: vec![],
                mode: document::ModeV1::Chat,
                effort: "low".into(),
                reasoning_open: true,
                parser_state: vec![],
            }
            .encode(),
            "malformed" => b"not a document".to_vec(),
            "refuse" => {
                return Err(document::RefusalV1::new("no tools here", Some("tools")).encode());
            }
            "refuse-text" => return Err(b"frontend sink layout 2".to_vec()),
            "refuse-garbage" => return Err(b"\xff".to_vec()),
            "panic" => panic!("render bug"),
            other => panic!("no render scenario {other}"),
        })
    }

    fn constraint(&self, request: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
        match document::ConstraintRequestV1::decode(request).unwrap() {
            document::ConstraintRequestV1::ToolChoice { .. } => match self.scenario.as_str() {
                "good" => Ok(Vec::new()),
                "constraint-past-the-limit" => Ok(runtime_grammar::Grammar::from_expr(
                    runtime_grammar::Expr::literal(&"g".repeat(2 << 20)),
                )
                .encode_wire()),
                _ => Ok(b"CHOPGRAMnot a grammar".to_vec()),
            },
        }
    }

    fn open_parser(&self, request: &[u8]) -> Result<Self::Parser, Vec<u8>> {
        let request = document::ParserOpenV1::decode(request).unwrap();
        assert_eq!(request.parser_state, [9]);
        if self.scenario == "refuse-open" {
            return Err(document::RefusalV1::new("no parser", None).encode());
        }
        Ok(FakeParserV1 {
            scenario: self.scenario.clone(),
            pushes: 0,
        })
    }
}

impl FrontendParserV1 for FakeParserV1 {
    fn push(&mut self, text: &[u8], _last: bool) -> Result<Vec<u8>, Vec<u8>> {
        document::PushV1::decode(text).unwrap();
        let push = self.pushes;
        self.pushes += 1;
        let case = PARSER_CASES
            .iter()
            .find(|case| case.name == self.scenario)
            .unwrap_or_else(|| panic!("no parser scenario {}", self.scenario));
        (case.answers)(push)
    }
}

/// The fixture declaration with one image modality whose pad is `[A]`.
fn descriptor() -> Arc<FrontendDescriptorV1> {
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(test_common::KEY_VALUE.as_bytes()).unwrap();
    descriptor.multimodal = Some(FrontendMultimodalV1 {
        arena: FrontendMediaArenaV1 {
            rows: 64,
            page_rows: 64,
            hidden: 8,
            element: FrontendMediaElementV1::Bf16,
        },
        encoders: vec![FrontendMediaEncoderV1 {
            name: "image".into(),
            payload_element: FrontendMediaElementV1::Bf16,
            payload_columns: 4,
            maximum_payload_bytes: 64,
            cost_per_row_milli: 1,
        }],
        modalities: vec![FrontendMediaModalityV1 {
            kind: FrontendMediaKindV1::Image,
            part_types: vec!["image_url".into()],
            maximum_items: 2,
            maximum_bytes: 64,
            maximum_tokens: 8,
            recipe: FrontendMediaRecipeV1 {
                tag: "image.patch-grid.v1".into(),
                parameters: serde_json::json!({}),
            },
            encoder: 0,
            render: FrontendMediaRenderV1 {
                pad: "[A]".into(),
                grid: None,
            },
            positions: None,
            layout: None,
        }],
    });
    Arc::new(descriptor)
}

fn format(scenario: &str) -> NativeFormatV1 {
    let export = FrontendExportV1::in_process::<FakeFrontendV1>();
    NativeFormatV1::open(&export, scenario.as_bytes(), descriptor()).unwrap()
}

fn image(source: &str) -> FrontendContentPartV1 {
    FrontendContentPartV1::Media(FrontendMediaPartV1 {
        kind: FrontendMediaPartKindV1::Image,
        part_type: "image_url".into(),
        source: source.into(),
        format: None,
    })
}

fn render(scenario: &str) -> Result<FormatRenderedV1, FrontendErrorV1> {
    render_parts(
        scenario,
        vec![
            FrontendContentPartV1::Text("look".into()),
            image("data:first"),
            image("data:second"),
        ],
    )
}

fn render_parts(
    scenario: &str,
    parts: Vec<FrontendContentPartV1>,
) -> Result<FormatRenderedV1, FrontendErrorV1> {
    render_with_kwargs(scenario, parts, None, None)
}

fn render_with_kwargs(
    scenario: &str,
    parts: Vec<FrontendContentPartV1>,
    preserve_thinking: Option<bool>,
    chat_template_kwargs: Option<FrontendJsonV1>,
) -> Result<FormatRenderedV1, FrontendErrorV1> {
    format(scenario).render(FormatRenderRequestV1 {
        messages: vec![FrontendMessageV1 {
            role: FrontendRoleV1::User,
            content: Some("look".into()),
            parts: Some(parts),
            reasoning_content: None,
            tool_calls: vec![],
            tool_call_id: None,
            tools: vec![],
        }],
        continue_final_message: false,
        add_generation_prompt: true,
        tools: &[],
        tool_objects: vec![],
        response_format: None,
        reasoning_effort: None,
        enable_thinking: None,
        thinking: None,
        nested_reasoning_effort: None,
        preserve_thinking,
        clear_thinking: None,
        add_vision_id: None,
        chat_template_kwargs,
    })
}

/// The render document carries the client's whole template keywords object,
/// a key no transport types among them, as sent; a request assembled
/// without one carries the typed keywords it set.
#[test]
fn the_whole_template_keywords_object_reaches_the_render_document() {
    let parts = || vec![FrontendContentPartV1::Text("look".into())];
    let kwargs: FrontendJsonV1 = serde_json::from_str(
        r#"{"model_mode":"deep","enable_thinking":true,"budget":12.50,"x":[null]}"#,
    )
    .unwrap();
    let sent = |preserve, kwargs| {
        render_with_kwargs("echo-kwargs", parts(), preserve, kwargs)
            .unwrap_err()
            .message
    };
    assert_eq!(
        sent(None, Some(kwargs)),
        r#"{"model_mode":"deep","enable_thinking":true,"budget":12.50,"x":[null]}"#
    );
    assert_eq!(sent(Some(true), None), r#"{"preserve_thinking":true}"#);
    assert_eq!(sent(None, None), "none");
}

#[test]
fn the_open_answer_names_what_the_format_reads() {
    assert_eq!(format("good").opened(), document::OpenedV1::default());
    assert_eq!(
        format("reads-everything").opened(),
        document::OpenedV1 {
            reads_tool_objects: true,
            reasoning_content_fallback: true,
            tool_call_reasoning_fallback: true,
            extra_roles: Vec::new(),
            tool_references: true,
        }
    );
    let export = FrontendExportV1::in_process::<FakeFrontendV1>();
    let error = NativeFormatV1::open(&export, b"malformed-open", descriptor())
        .map(|_| ())
        .unwrap_err();
    assert!(
        error.starts_with("the bundle frontend answered a malformed open: "),
        "{error}"
    );
}

/// An open answer may declare extra roles within the document's bounds; a
/// render carries such a role as its name.
#[test]
fn declared_extra_roles_are_checked_and_carried_by_name() {
    let opened = format("roles:observation,ipython").opened();
    assert_eq!(opened.extra_roles, ["observation", "ipython"]);
    let message = |role| FrontendMessageV1 {
        role,
        content: Some("seen".into()),
        parts: None,
        reasoning_content: None,
        tool_calls: vec![],
        tool_call_id: None,
        tools: vec![],
    };
    let error = format("roles:observation")
        .render(FormatRenderRequestV1 {
            messages: vec![
                message(FrontendRoleV1::User),
                message(FrontendRoleV1::Declared("observation".into())),
            ],
            continue_final_message: false,
            add_generation_prompt: true,
            tools: &[],
            tool_objects: vec![],
            response_format: None,
            reasoning_effort: None,
            enable_thinking: None,
            thinking: None,
            nested_reasoning_effort: None,
            preserve_thinking: None,
            clear_thinking: None,
            add_vision_id: None,
            chat_template_kwargs: None,
        })
        .unwrap_err();
    assert_eq!(error.message, "user,observation");

    let most: Vec<_> = (0..document::FRONTEND_EXTRA_ROLES_MAX_V1)
        .map(|index| format!("r{index}"))
        .collect();
    let longest = "r".repeat(document::FRONTEND_ROLE_NAME_BYTES_MAX_V1);
    for roles in [most.join(","), longest.clone(), "a_b-C9".into()] {
        let scenario = format!("roles:{roles}");
        assert_eq!(format(&scenario).opened().extra_roles.join(","), roles);
    }
    let export = FrontendExportV1::in_process::<FakeFrontendV1>();
    for (roles, refusal) in [
        (
            format!("{},r16", most.join(",")),
            "it declares 17 extra roles, past the 16 a frontend may".to_owned(),
        ),
        (
            format!("{longest}r"),
            format!("extra role \"{longest}r\" is not a role name"),
        ),
        (
            String::new(),
            "extra role \"\" is not a role name".to_owned(),
        ),
        (
            "ob servation".into(),
            "extra role \"ob servation\" is not a role name".into(),
        ),
        (
            "observation,tool".into(),
            "extra role \"tool\" is a common role".into(),
        ),
        (
            "observation,observation".into(),
            "extra role \"observation\" is declared twice".into(),
        ),
    ] {
        let scenario = format!("roles:{roles}");
        let error = NativeFormatV1::open(&export, scenario.as_bytes(), descriptor())
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            error,
            format!("the bundle frontend answered a malformed open: {refusal}"),
            "{roles}"
        );
    }
}

#[test]
fn a_render_answer_rebuilds_the_prompt_with_the_request_media() {
    let rendered = render("good").unwrap();
    assert_eq!(rendered.prompt, "x[A]y[A]z");
    assert_eq!(
        rendered.segments,
        [
            PromptSegmentV1::Text("x".into()),
            PromptSegmentV1::Media(0),
            PromptSegmentV1::Text("y".into()),
            PromptSegmentV1::Media(1),
            PromptSegmentV1::Text("z".into()),
        ]
    );
    let sources: Vec<_> = rendered.media.iter().map(|media| &media.source).collect();
    assert_eq!(sources, ["data:first", "data:second"]);
    assert_eq!(rendered.mode, PromptModeV1::Thinking);
    assert_eq!(rendered.effort, "high");
    assert!(rendered.reasoning_open);
    assert_eq!(rendered.parser_state, [9]);
}

/// A format may render a user message without its media parts, as one that
/// renders a message carrying message-level tools as plain content does, so
/// a render may leave request parts unplaced.
#[test]
fn a_render_may_leave_request_media_unplaced() {
    let rendered = render("media-dropped").unwrap();
    assert_eq!(rendered.prompt, "x[A]z");
    assert_eq!(
        rendered.segments,
        [
            PromptSegmentV1::Text("x".into()),
            PromptSegmentV1::Media(0),
            PromptSegmentV1::Text("z".into()),
        ]
    );
    let sources: Vec<_> = rendered.media.iter().map(|media| &media.source).collect();
    assert_eq!(sources, ["data:second"]);
}

#[test]
fn bad_render_answers_are_refused() {
    for scenario in [
        "media-twice",
        "media-out-of-order",
        "media-extra",
        "undeclared-effort",
        "reasoning-open-in-chat",
        "malformed",
    ] {
        let error = render(scenario).unwrap_err();
        assert!(
            error
                .message
                .starts_with("the bundle frontend answered a malformed render"),
            "{scenario}: {}",
            error.message
        );
        assert_eq!(error.parameter, None);
    }
    assert_eq!(
        render("refuse").unwrap_err(),
        FrontendErrorV1::new("no tools here", Some("tools"))
    );
    assert_eq!(
        render("refuse-text").unwrap_err(),
        FrontendErrorV1::new("frontend sink layout 2", None)
    );
    assert!(
        render("refuse-garbage")
            .unwrap_err()
            .message
            .starts_with("the bundle frontend refused with a malformed document")
    );
    assert_eq!(
        render("panic").unwrap_err(),
        FrontendErrorV1::new("the bundle frontend panicked", None)
    );
}

/// A prompt past the declared length is refused as the toolkit's prompt
/// writer refuses it, and an answer longer than any prompt it allows is refused
/// before the host copies it.
#[test]
fn prompts_past_the_declared_length_are_refused() {
    let text = || vec![FrontendContentPartV1::Text("look".into())];
    let too_long = FrontendErrorV1::new(
        resource_v1("prompt exceeds the byte limit"),
        Some("messages"),
    );
    let media_too_long = FrontendErrorV1::new(
        resource_v1("rendered media prompt exceeds maximum_prompt_bytes"),
        Some("messages"),
    );
    let limit = descriptor().limits.maximum_prompt_bytes;
    assert_eq!(
        render_parts("prompt-at-the-limit", text())
            .unwrap()
            .prompt
            .len(),
        limit
    );
    assert_eq!(
        render_parts("prompt-past-the-limit", text()).unwrap_err(),
        too_long
    );
    assert_eq!(
        render_parts("prompt-past-the-sink", text()).unwrap_err(),
        too_long
    );
    assert_eq!(
        render("media-prompt-at-the-limit").unwrap().prompt.len(),
        limit
    );
    assert_eq!(
        render("media-prompt-past-the-limit").unwrap_err(),
        media_too_long
    );
    assert_eq!(render("prompt-past-the-sink").unwrap_err(), media_too_long);
    let state = render("state-past-the-limit").unwrap_err();
    assert!(
        state.message.starts_with(
            "the bundle frontend answered a malformed render: frontend document: parser state"
        ),
        "{}",
        state.message
    );
}

/// A constraint longer than any the declared tool grammars build from the
/// request is refused before the host copies it.
#[test]
fn a_constraint_past_its_limit_is_refused() {
    let error = format("constraint-past-the-limit")
        .tool_constraint(&[], &FrontendToolChoiceV1::Required)
        .unwrap_err();
    assert!(
        error.message.starts_with("the bundle frontend emitted a ")
            && error.message.ends_with(" byte limit"),
        "{}",
        error.message
    );
}

#[test]
fn constraint_answers_are_checked() {
    assert_eq!(
        format("good").tool_constraint(&[], &FrontendToolChoiceV1::Required),
        Ok(None)
    );
    assert!(
        format("bad")
            .tool_constraint(&[], &FrontendToolChoiceV1::Required)
            .unwrap_err()
            .message
            .starts_with("the bundle frontend answered a malformed constraint")
    );
}

/// One parser scenario: the options it opens with, the pushes the host
/// makes, and the answers the fake gives them.
struct ParserCaseV1 {
    name: &'static str,
    options: fn(&mut FormatParserOptionsV1),
    /// Each push's released bytes, whether bytes are held and whether it
    /// is the last.
    pushes: &'static [(&'static [u8], bool, bool)],
    answers: fn(usize) -> Result<Vec<u8>, Vec<u8>>,
    /// What the last push is refused with, or `None` when every push is
    /// answered.
    refusal: Option<&'static str>,
}

fn events(events: Vec<document::EventV1<'static>>, cursor: u64) -> Result<Vec<u8>, Vec<u8>> {
    Ok(document::EventsV1 { events, cursor }.encode())
}

fn finish(reason: document::FinishReasonV1) -> document::EventV1<'static> {
    document::EventV1::Finish {
        reason,
        completion: None,
    }
}

fn calls(calls: &[(&'static str, &'static str)]) -> Vec<document::CallV1<'static>> {
    calls
        .iter()
        .map(|(name, arguments)| document::CallV1 { name, arguments })
        .collect()
}

fn delta(
    index: u32,
    name: Option<&'static str>,
    arguments: &'static str,
) -> document::DeltaV1<'static> {
    document::DeltaV1 {
        index,
        name,
        arguments,
    }
}

fn chat(_: &mut FormatParserOptionsV1) {}

fn thinking(options: &mut FormatParserOptionsV1) {
    options.mode = PromptModeV1::Thinking;
    options.reasoning_open = true;
}

fn streaming(options: &mut FormatParserOptionsV1) {
    options.stream_tool_calls = true;
}

fn no_tools(options: &mut FormatParserOptionsV1) {
    options.extract_tools = false;
}

const CALL: &[u8] = b"<call>";
const ONE: (&str, &str) = ("lookup", r#"{"city":"Paris"}"#);

/// Every kind of bad parser answer, and two good ones.
const PARSER_CASES: &[ParserCaseV1] = &[
    ParserCaseV1 {
        name: "good-text",
        options: thinking,
        pushes: &[
            (b"plan</t>an", false, false),
            (b"plan</t>answer", false, true),
        ],
        answers: |push| match push {
            0 => events(
                vec![
                    document::EventV1::Reasoning(0..4),
                    document::EventV1::Content(8..10),
                ],
                10,
            ),
            _ => events(
                vec![
                    document::EventV1::Content(10..14),
                    finish(document::FinishReasonV1::Length),
                ],
                14,
            ),
        },
        refusal: None,
    },
    ParserCaseV1 {
        name: "good-calls",
        options: streaming,
        pushes: &[(CALL, false, false), (CALL, false, true)],
        answers: |push| match push {
            0 => events(
                vec![document::EventV1::ToolCallDeltas(vec![
                    delta(0, Some("lookup"), r#"{"city":"#),
                    delta(0, None, r#""Paris"}"#),
                ])],
                0,
            ),
            _ => events(
                vec![
                    document::EventV1::ToolCalls {
                        range: 0..6,
                        calls: calls(&[ONE]),
                    },
                    document::EventV1::Finish {
                        reason: document::FinishReasonV1::ToolCalls,
                        completion: Some(document::CompletionV1 {
                            content: "",
                            reasoning_content: None,
                            tool_calls: calls(&[ONE]),
                        }),
                    },
                ],
                6,
            ),
        },
        refusal: None,
    },
    ParserCaseV1 {
        name: "events-past-their-limit",
        options: chat,
        pushes: &[(b"x", false, true)],
        answers: |_| {
            events(
                vec![
                    document::EventV1::Content(0..1),
                    document::EventV1::Finish {
                        reason: document::FinishReasonV1::Stop,
                        completion: Some(document::CompletionV1 {
                            content: "x".repeat(1 << 20).leak(),
                            reasoning_content: None,
                            tool_calls: vec![],
                        }),
                    },
                ],
                1,
            )
        },
        refusal: Some("malformed completion: the bundle frontend emitted a 1048"),
    },
    ParserCaseV1 {
        name: "malformed-events",
        options: chat,
        pushes: &[(b"a", false, false)],
        answers: |_| Ok(b"garbage".to_vec()),
        refusal: Some("malformed completion: the format answered a malformed event document"),
    },
    ParserCaseV1 {
        name: "reasoning-in-chat",
        options: chat,
        pushes: &[(b"abc", false, false)],
        answers: |_| events(vec![document::EventV1::Reasoning(0..3)], 3),
        refusal: Some("malformed completion: the format reported reasoning in chat mode"),
    },
    ParserCaseV1 {
        name: "range-past-bytes",
        options: chat,
        pushes: &[(b"abc", false, false)],
        answers: |_| events(vec![document::EventV1::Content(0..4)], 3),
        refusal: Some("malformed completion: the format reported a range"),
    },
    ParserCaseV1 {
        name: "overlapping-ranges",
        options: chat,
        pushes: &[(b"abcd", false, false)],
        answers: |_| {
            events(
                vec![
                    document::EventV1::Content(0..3),
                    document::EventV1::Content(2..4),
                ],
                4,
            )
        },
        refusal: Some("malformed completion: the format reported a range"),
    },
    ParserCaseV1 {
        name: "range-before-an-earlier-push",
        options: chat,
        pushes: &[(b"ab", false, false), (b"abcd", false, false)],
        answers: |push| match push {
            0 => events(vec![document::EventV1::Content(0..2)], 2),
            _ => events(vec![document::EventV1::Content(1..4)], 4),
        },
        refusal: Some("malformed completion: the format reported a range"),
    },
    ParserCaseV1 {
        name: "range-splitting-a-two-byte-character",
        options: chat,
        pushes: &[("\u{e9}".as_bytes(), false, true)],
        answers: |_| {
            events(
                vec![
                    document::EventV1::Content(0..1),
                    document::EventV1::Content(1..2),
                    finish(document::FinishReasonV1::Stop),
                ],
                2,
            )
        },
        refusal: Some("malformed completion: the format reported a range that splits a UTF-8"),
    },
    ParserCaseV1 {
        name: "range-starting-inside-a-two-byte-character",
        options: chat,
        pushes: &[("a\u{e9}".as_bytes(), false, false)],
        answers: |_| events(vec![document::EventV1::Content(2..3)], 3),
        refusal: Some("malformed completion: the format reported a range that splits a UTF-8"),
    },
    ParserCaseV1 {
        name: "range-splitting-a-three-byte-character",
        options: thinking,
        pushes: &[("\u{20ac}".as_bytes(), false, false)],
        answers: |_| events(vec![document::EventV1::Reasoning(0..2)], 2),
        refusal: Some("malformed completion: the format reported a range that splits a UTF-8"),
    },
    ParserCaseV1 {
        name: "range-splitting-a-four-byte-character",
        options: chat,
        pushes: &[("\u{1f600}".as_bytes(), false, false)],
        answers: |_| events(vec![document::EventV1::Content(0..3)], 3),
        refusal: Some("malformed completion: the format reported a range that splits a UTF-8"),
    },
    ParserCaseV1 {
        name: "tool-call-range-ending-inside-a-character",
        options: chat,
        pushes: &[("<call>\u{1f600}".as_bytes(), false, false)],
        answers: |_| {
            events(
                vec![document::EventV1::ToolCalls {
                    range: 0..7,
                    calls: calls(&[ONE]),
                }],
                7,
            )
        },
        refusal: Some("malformed completion: the format reported a range that splits a UTF-8"),
    },
    ParserCaseV1 {
        name: "whole-character-ranges",
        options: thinking,
        pushes: &[("a\u{e9}\u{20ac}\u{1f600}<call>".as_bytes(), false, true)],
        answers: |_| {
            events(
                vec![
                    document::EventV1::Reasoning(0..1),
                    document::EventV1::Reasoning(1..3),
                    document::EventV1::Content(3..6),
                    document::EventV1::Content(6..10),
                    document::EventV1::ToolCalls {
                        range: 10..16,
                        calls: calls(&[ONE]),
                    },
                    finish(document::FinishReasonV1::ToolCalls),
                ],
                16,
            )
        },
        refusal: None,
    },
    // Replacement decoding reads each maximal ill-formed subpart as one
    // U+FFFD, so a range may end between such runs but not inside one.
    ParserCaseV1 {
        name: "range-splitting-an-ill-formed-run",
        options: chat,
        pushes: &[(b"\xe2\x82x", false, false)],
        answers: |_| events(vec![document::EventV1::Content(0..1)], 1),
        refusal: Some("malformed completion: the format reported a range that splits a UTF-8"),
    },
    ParserCaseV1 {
        name: "range-splitting-a-truncated-four-byte-run",
        options: chat,
        pushes: &[(b"\xf0\x9f\x98x", false, false)],
        answers: |_| events(vec![document::EventV1::Content(0..2)], 2),
        refusal: Some("malformed completion: the format reported a range that splits a UTF-8"),
    },
    ParserCaseV1 {
        name: "range-splitting-a-held-ill-formed-run",
        options: chat,
        pushes: &[(b"\xe2\x82", true, false)],
        answers: |_| events(vec![document::EventV1::Content(0..1)], 1),
        refusal: Some("malformed completion: the format reported a range that splits a UTF-8"),
    },
    ParserCaseV1 {
        name: "cursor-inside-an-ill-formed-run",
        options: chat,
        pushes: &[(b"\xe2\x82x", false, false)],
        answers: |_| events(vec![], 1),
        refusal: Some(
            "malformed completion: the format moved its cursor inside a UTF-8 character or ill-formed byte run",
        ),
    },
    // A lone continuation, a three-byte lead cut short, two invalid leads
    // and a four-byte lead truncated at the end of the final push: each
    // range covers one whole decoding unit.
    ParserCaseV1 {
        name: "ranges-around-invalid-bytes",
        options: chat,
        pushes: &[(b"\x80a\xe2\x82x\xc0\xc0\xff\xf0\x9f\x98", false, true)],
        answers: |_| {
            events(
                vec![
                    document::EventV1::Content(0..1),
                    document::EventV1::Content(1..2),
                    document::EventV1::Content(2..4),
                    document::EventV1::Content(4..5),
                    document::EventV1::Content(5..6),
                    document::EventV1::Content(6..7),
                    document::EventV1::Content(7..8),
                    document::EventV1::Content(8..11),
                    finish(document::FinishReasonV1::Stop),
                ],
                11,
            )
        },
        refusal: None,
    },
    ParserCaseV1 {
        name: "cursor-back",
        options: chat,
        pushes: &[(b"ab", false, false), (b"abcd", false, false)],
        answers: |push| events(vec![], if push == 0 { 2 } else { 1 }),
        refusal: Some("malformed completion: the format moved its cursor"),
    },
    // Also the native answers of the cursor and client-stop regression
    // below, push by push.
    ParserCaseV1 {
        name: "cursor-inside-a-character",
        options: chat,
        pushes: &[("\u{e9}".as_bytes(), false, false)],
        answers: |push| match push {
            0 => events(vec![], 1),
            1 => events(vec![document::EventV1::Content(0..1)], 1),
            2 => events(vec![document::EventV1::Content(2..4)], 4),
            _ => events(vec![finish(document::FinishReasonV1::Length)], 4),
        },
        refusal: Some(
            "malformed completion: the format moved its cursor inside a UTF-8 character or ill-formed byte run",
        ),
    },
    ParserCaseV1 {
        name: "cursor-past-bytes",
        options: chat,
        pushes: &[(b"ab", false, false)],
        answers: |_| events(vec![], 3),
        refusal: Some("malformed completion: the format moved its cursor"),
    },
    ParserCaseV1 {
        name: "call-name-outside-charset",
        options: chat,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            events(
                vec![document::EventV1::ToolCalls {
                    range: 0..6,
                    calls: calls(&[("look up", "{}")]),
                }],
                6,
            )
        },
        refusal: Some("malformed completion: the format named a tool outside"),
    },
    ParserCaseV1 {
        name: "call-arguments-not-an-object",
        options: chat,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            events(
                vec![document::EventV1::ToolCalls {
                    range: 0..6,
                    calls: calls(&[("lookup", "[1]")]),
                }],
                6,
            )
        },
        refusal: Some("malformed completion: the format delivered tool-call arguments"),
    },
    ParserCaseV1 {
        name: "call-arguments-malformed",
        options: chat,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            events(
                vec![document::EventV1::ToolCalls {
                    range: 0..6,
                    calls: calls(&[("lookup", r#"{"a":}"#)]),
                }],
                6,
            )
        },
        refusal: Some("malformed completion: the format delivered tool-call arguments"),
    },
    // The transport charges arguments against the declared limits when it
    // checks the call, as before, so the parser passes them on.
    ParserCaseV1 {
        name: "call-arguments-past-the-limits",
        options: chat,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            events(
                vec![document::EventV1::ToolCalls {
                    range: 0..6,
                    // The fixture declares at most 4096 JSON bytes.
                    calls: calls(&[(
                        "lookup",
                        String::leak(format!(r#"{{"a":"{}"}}"#, "x".repeat(4096))),
                    )]),
                }],
                6,
            )
        },
        refusal: None,
    },
    ParserCaseV1 {
        name: "too-many-calls",
        options: chat,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            // The fixture declares at most 4 calls.
            events(
                vec![document::EventV1::ToolCalls {
                    range: 0..6,
                    calls: calls(&[ONE, ONE, ONE, ONE, ONE]),
                }],
                6,
            )
        },
        refusal: Some("malformed completion: the format delivered more tool calls"),
    },
    ParserCaseV1 {
        name: "empty-call-group",
        options: chat,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            events(
                vec![document::EventV1::ToolCalls {
                    range: 0..6,
                    calls: vec![],
                }],
                6,
            )
        },
        refusal: Some("malformed completion: the format delivered an empty tool-call group"),
    },
    ParserCaseV1 {
        name: "calls-without-extraction",
        options: no_tools,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            events(
                vec![document::EventV1::ToolCalls {
                    range: 0..6,
                    calls: calls(&[ONE]),
                }],
                6,
            )
        },
        refusal: Some("malformed completion: the format delivered tool calls the request"),
    },
    ParserCaseV1 {
        name: "deltas-without-streaming",
        options: chat,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            events(
                vec![document::EventV1::ToolCallDeltas(vec![delta(
                    0,
                    Some("lookup"),
                    "{",
                )])],
                0,
            )
        },
        refusal: Some("malformed completion: the format streamed tool calls the request"),
    },
    ParserCaseV1 {
        name: "deltas-that-do-not-add-up",
        options: streaming,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            events(
                vec![
                    document::EventV1::ToolCallDeltas(vec![delta(0, Some("lookup"), "{}")]),
                    document::EventV1::ToolCalls {
                        range: 0..6,
                        calls: calls(&[ONE]),
                    },
                ],
                6,
            )
        },
        refusal: Some("malformed completion: the format's tool-call deltas do not add up"),
    },
    ParserCaseV1 {
        name: "delta-for-a-delivered-call",
        options: streaming,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            events(
                vec![
                    document::EventV1::ToolCallDeltas(vec![delta(
                        0,
                        Some("lookup"),
                        r#"{"city":"Paris"}"#,
                    )]),
                    document::EventV1::ToolCalls {
                        range: 0..6,
                        calls: calls(&[ONE]),
                    },
                    document::EventV1::ToolCallDeltas(vec![delta(0, None, " ")]),
                ],
                6,
            )
        },
        refusal: Some("malformed completion: the format streamed a delta outside"),
    },
    ParserCaseV1 {
        name: "delta-skipping-a-call",
        options: streaming,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            events(
                vec![document::EventV1::ToolCallDeltas(vec![delta(
                    1,
                    Some("lookup"),
                    "{",
                )])],
                0,
            )
        },
        refusal: Some("malformed completion: the format streamed a delta outside"),
    },
    ParserCaseV1 {
        name: "delta-renaming-a-call",
        options: streaming,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            events(
                vec![document::EventV1::ToolCallDeltas(vec![
                    delta(0, Some("lookup"), "{"),
                    delta(0, Some("other"), "}"),
                ])],
                0,
            )
        },
        refusal: Some("malformed completion: the format named a tool call on other than"),
    },
    ParserCaseV1 {
        name: "delta-name-outside-charset",
        options: streaming,
        pushes: &[(CALL, false, false)],
        answers: |_| {
            events(
                vec![document::EventV1::ToolCallDeltas(vec![delta(
                    0,
                    Some("look up"),
                    "{",
                )])],
                0,
            )
        },
        refusal: Some("malformed completion: the format named a tool outside"),
    },
    ParserCaseV1 {
        name: "event-after-finish",
        options: chat,
        pushes: &[(b"ab", false, false)],
        answers: |_| {
            events(
                vec![
                    finish(document::FinishReasonV1::Stop),
                    document::EventV1::Content(0..2),
                ],
                2,
            )
        },
        refusal: Some("malformed completion: the format reported an event after finishing"),
    },
    ParserCaseV1 {
        name: "finish-twice",
        options: chat,
        pushes: &[(b"ab", false, false)],
        answers: |_| {
            events(
                vec![
                    finish(document::FinishReasonV1::Stop),
                    finish(document::FinishReasonV1::Stop),
                ],
                2,
            )
        },
        refusal: Some("malformed completion: the format reported an event after finishing"),
    },
    ParserCaseV1 {
        name: "finish-before-held-bytes",
        options: chat,
        pushes: &[(b"ab", true, false)],
        answers: |_| events(vec![finish(document::FinishReasonV1::Stop)], 2),
        refusal: Some("malformed completion: the format finished before the end"),
    },
    // Nothing is held, so only the cursor shows the bytes are unread.
    ParserCaseV1 {
        name: "finish-before-the-cursor-reaches-the-bytes",
        options: chat,
        pushes: &[(b"abc", false, false)],
        answers: |_| events(vec![finish(document::FinishReasonV1::Stop)], 0),
        refusal: Some("malformed completion: the format finished before reading every"),
    },
    ParserCaseV1 {
        name: "final-finish-before-the-cursor-reaches-the-bytes",
        options: chat,
        pushes: &[(b"ab", false, false), (b"abc", false, true)],
        answers: |push| match push {
            0 => events(vec![document::EventV1::Content(0..2)], 2),
            _ => events(vec![finish(document::FinishReasonV1::Length)], 2),
        },
        refusal: Some("malformed completion: the format finished before reading every"),
    },
    ParserCaseV1 {
        name: "finish-with-an-undelivered-streamed-call",
        options: streaming,
        pushes: &[(b"abc", false, true)],
        answers: |_| {
            events(
                vec![
                    document::EventV1::ToolCallDeltas(vec![delta(0, Some("lookup"), "{")]),
                    finish(document::FinishReasonV1::Stop),
                ],
                3,
            )
        },
        refusal: Some("malformed completion: the format finished with a streamed tool call"),
    },
    ParserCaseV1 {
        name: "later-finish-with-an-undelivered-streamed-call",
        options: streaming,
        pushes: &[(b"ab", false, false), (b"abc", false, true)],
        answers: |push| match push {
            0 => events(
                vec![document::EventV1::ToolCallDeltas(vec![delta(
                    0,
                    Some("lookup"),
                    "{",
                )])],
                0,
            ),
            _ => events(vec![finish(document::FinishReasonV1::Stop)], 3),
        },
        refusal: Some("malformed completion: the format finished with a streamed tool call"),
    },
    ParserCaseV1 {
        name: "no-finish-on-the-final-push",
        options: chat,
        pushes: &[(b"ab", false, true)],
        answers: |_| events(vec![document::EventV1::Content(0..2)], 2),
        refusal: Some("malformed completion: the format did not finish on the final push"),
    },
    ParserCaseV1 {
        name: "finish-reason-without-calls",
        options: chat,
        pushes: &[(b"ab", false, true)],
        answers: |_| events(vec![finish(document::FinishReasonV1::ToolCalls)], 2),
        refusal: Some("malformed completion: the format's finish reason disagrees"),
    },
    ParserCaseV1 {
        name: "completion-disagreeing-with-calls",
        options: chat,
        pushes: &[(b"ab", false, true)],
        answers: |_| {
            events(
                vec![document::EventV1::Finish {
                    reason: document::FinishReasonV1::Stop,
                    completion: Some(document::CompletionV1 {
                        content: "ab",
                        reasoning_content: None,
                        tool_calls: calls(&[ONE]),
                    }),
                }],
                2,
            )
        },
        refusal: Some("malformed completion: the format's completion disagrees"),
    },
    ParserCaseV1 {
        name: "completion-reasoning-in-chat",
        options: chat,
        pushes: &[(b"ab", false, true)],
        answers: |_| {
            events(
                vec![document::EventV1::Finish {
                    reason: document::FinishReasonV1::Stop,
                    completion: Some(document::CompletionV1 {
                        content: "ab",
                        reasoning_content: Some(""),
                        tool_calls: vec![],
                    }),
                }],
                2,
            )
        },
        refusal: Some("malformed completion: the format reported reasoning in chat mode"),
    },
    ParserCaseV1 {
        name: "refusal-in-a-decoder-family",
        options: chat,
        pushes: &[(b"ab", false, false)],
        answers: |_| {
            Err(document::RefusalV1::new("conversation resource limit: too many", None).encode())
        },
        refusal: Some("conversation resource limit: too many"),
    },
    ParserCaseV1 {
        name: "refusal-outside-the-families",
        options: chat,
        pushes: &[(b"ab", false, false)],
        answers: |_| Err(document::RefusalV1::new("no", None).encode()),
        refusal: Some("malformed completion: no"),
    },
    ParserCaseV1 {
        name: "panic",
        options: chat,
        pushes: &[(b"ab", false, false)],
        answers: |_| panic!("parser bug"),
        refusal: Some("malformed completion: decoder panicked"),
    },
];

fn options() -> FormatParserOptionsV1 {
    FormatParserOptionsV1 {
        mode: PromptModeV1::Chat,
        reasoning_open: false,
        extract_tools: true,
        stream_tool_calls: false,
        raw: false,
        ignore_eos: false,
        engine_tokens: true,
        tool_schemas: vec![],
    }
}

#[test]
fn bad_parser_answers_are_refused_and_end_the_parser() {
    for case in PARSER_CASES {
        let mut options = options();
        (case.options)(&mut options);
        let mut parser = format(case.name).open_parser(options, &[9]).unwrap();
        let mut outcome = Ok(Vec::new());
        for (completion, held, last) in case.pushes {
            outcome = parser.push(FormatInputV1 {
                completion,
                held: *held,
                last: *last,
                stop_floor: Some(0),
            });
            if outcome.is_err() {
                break;
            }
        }
        match (case.refusal, outcome) {
            (None, Ok(_)) => {}
            (Some(refusal), Err(error)) => {
                assert!(error.starts_with(refusal), "{}: {error}", case.name);
                // A parser that failed takes nothing more.
                assert_eq!(
                    parser.push(FormatInputV1 {
                        completion: b"abcdefgh",
                        held: false,
                        last: false,
                        stop_floor: None,
                    }),
                    Err(sequence_v1("bytes pushed after the format parser failed")),
                    "{}",
                    case.name
                );
            }
            (refusal, outcome) => panic!("{}: expected {refusal:?}, got {outcome:?}", case.name),
        }
    }
}

/// A cursor inside a character would let the host's client-stop scan read
/// the character's tail as replacement text matching a stop, release its
/// lead byte alone and publish U+FFFD where the model generated "\u{e9}".
#[test]
fn a_cursor_inside_a_character_is_refused_before_a_stop_scan_reads_it() {
    let descriptor = descriptor();
    let mut decoder = CompletionDecoderV1::with_client_stops(
        Arc::clone(&descriptor),
        PromptModeV1::Chat,
        DecoderLimitsV1::declared(&descriptor),
        &["\u{fffd}ax".to_owned()],
    )
    .unwrap();
    decoder.set_parser_source(
        BundleParserSourceV1::new(Arc::new(format("cursor-inside-a-character")))
            .with_state(vec![9]),
    );
    let mut content = String::new();
    let mut outcome = Ok(());
    for (ordinal, text) in ["\u{e9}", "a", "y"].into_iter().enumerate() {
        match decoder.push(GeneratedTokenV1::new(
            ordinal as u64,
            200,
            text.as_bytes().to_vec(),
        )) {
            Ok(updates) => content.extend(updates.into_iter().filter_map(|update| match update {
                CompletionUpdateV1::Content(text) => Some(text.text),
                _ => None,
            })),
            Err(error) => {
                outcome = Err(error);
                break;
            }
        }
    }
    if outcome.is_ok() {
        outcome = decoder.finish_length().map(|_| ());
    }
    assert!(!content.contains('\u{fffd}'), "{content:?}");
    assert_eq!(
        outcome,
        Err(malformed_v1(
            "the format moved its cursor inside a UTF-8 character or ill-formed byte run"
        ))
    );
}

/// A boundary is refused exactly where cutting the bytes there changes their
/// replacement decoding: every split of every short string over bytes that
/// lead, continue and break UTF-8 sequences, then seeded longer strings.
#[test]
fn a_refused_boundary_is_one_that_changes_the_decoding() {
    const BYTES: [u8; 16] = [
        b'x', 0x80, 0x82, 0x9f, 0xa9, 0xbf, 0xc0, 0xc3, 0xe0, 0xe2, 0xed, 0xf0, 0xf4, 0xf5, 0xff,
        0x98,
    ];
    let check = |bytes: &[u8]| {
        for offset in 1..bytes.len() {
            let (head, tail) = bytes.split_at(offset);
            let changes = String::from_utf8_lossy(head) + String::from_utf8_lossy(tail)
                != String::from_utf8_lossy(bytes);
            assert_eq!(
                splits_decoding_unit_v1(bytes, offset),
                changes,
                "{bytes:02x?} at {offset}"
            );
        }
    };
    for length in 2..=4u32 {
        for mut index in 0..BYTES.len().pow(length) {
            let bytes: Vec<u8> = (0..length)
                .map(|_| {
                    let byte = BYTES[index % BYTES.len()];
                    index /= BYTES.len();
                    byte
                })
                .collect();
            check(&bytes);
        }
    }
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..20_000 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let length = 5 + (seed >> 60) as usize;
        let bytes: Vec<u8> = (0..length)
            .map(|position| BYTES[(seed >> (position * 4 % 56)) as usize % BYTES.len()])
            .collect();
        check(&bytes);
    }
}

#[test]
fn good_parser_answers_become_format_events() {
    let mut options = options();
    streaming(&mut options);
    let mut parser = format("good-calls").open_parser(options, &[9]).unwrap();
    let input = |last| FormatInputV1 {
        completion: CALL,
        held: false,
        last,
        stop_floor: Some(0),
    };
    assert_eq!(
        parser.push(input(false)).unwrap(),
        [FormatEventV1::ToolCallDeltas(vec![
            FormatToolCallDeltaV1 {
                index: 0,
                name: Some("lookup".into()),
                arguments: r#"{"city":"#.into(),
            },
            FormatToolCallDeltaV1 {
                index: 0,
                name: None,
                arguments: r#""Paris"}"#.into(),
            },
        ])]
    );
    assert_eq!(parser.cursor(), 0);
    let call = ParsedToolCallV1 {
        name: ONE.0.into(),
        arguments: ONE.1.into(),
    };
    assert_eq!(
        parser.push(input(true)).unwrap(),
        [
            FormatEventV1::ToolCalls {
                calls: vec![call.clone()],
                range: 0..6,
            },
            FormatEventV1::Finish {
                reason: CompletionFinishV1::ToolCalls,
                completion: Some(AssistantCompletionV1 {
                    content: String::new(),
                    reasoning_content: None,
                    tool_calls: vec![call],
                }),
            },
        ]
    );
    assert_eq!(parser.cursor(), 6);
    assert_eq!(
        parser.push(input(false)),
        Err(malformed_v1("bytes received after terminal completion"))
    );
}

#[test]
fn a_parser_open_refusal_reads_as_the_format_worded_it() {
    let error = format("refuse-open")
        .open_parser(options(), &[9])
        .map(|_| ())
        .unwrap_err();
    assert_eq!(error, malformed_v1("no parser"));
}

#[test]
fn json_values_cross_as_text_that_reads_back_the_same() {
    for text in [
        r#"{"b":1,"a":[true,false,null],"c":{"z":"\u0000\"\\é"}}"#,
        r#"[1.50,2.5e-3,-0,12345678901234567890123,0.1]"#,
        r#""plain""#,
    ] {
        let value: FrontendJsonV1 = serde_json::from_str(text).unwrap();
        let crossed: FrontendJsonV1 = serde_json::from_str(&json_text_v1(&value)).unwrap();
        assert_eq!(crossed, value, "{text}");
    }
}

#[test]
fn tool_call_arguments_are_checked_by_json_syntax_alone() {
    let deep = |depth: usize| format!("{{\"a\":{}1{}}}", "[".repeat(depth), "]".repeat(depth));
    for text in [
        "{}",
        " { } ",
        r#"{"city": "Paris", "n": -0.5e+3, "ok": [true, false, null, {}], "u": "\u00e9\n"}"#,
        r#"{"a":1,"a":2}"#,
        r#"{"x":"é\"\\\/"}"#,
    ]
    .into_iter()
    .map(str::to_owned)
    .chain([deep(100), deep(100_000)])
    {
        assert!(json_object_text_v1(&text), "{text}");
    }
    for text in [
        "",
        "[]",
        "1",
        "\"s\"",
        "{",
        "{\"a\"}",
        "{\"a\":}",
        "{\"a\":1,}",
        "{,}",
        "{a:1}",
        "{\"a\":01}",
        "{\"a\":1.}",
        "{\"a\":tru}",
        "{\"a\":\"\u{1}\"}",
        "{\"a\":\"\\x\"}",
        "{}{}",
        "{\"a\":[1 2]}",
        "{\"a\":[1,]}",
        "{\"a\":1]",
        "{\"a\":\"\\u12\"}",
    ] {
        assert!(!json_object_text_v1(text), "{text}");
    }
    // Wherever serde_json reads an object, so does the check, so it never
    // refuses arguments the transport would accept: every shared case's
    // arguments, and each of them with one byte removed or doubled.
    let cases: serde_json::Value =
        serde_json::from_str(frontend_format::test_support::FUNCTION_PARAMETER_CASES).unwrap();
    let mut texts = vec![
        r#"{"a":[{"b":{}}],"c":"d"}"#.to_owned(),
        r#"{"n":1e5,"m":-0, "s": "\ud83d\ude00"}"#.to_owned(),
    ];
    for case in cases["cases"].as_array().unwrap() {
        for call in case["calls"].as_array().unwrap() {
            texts.push(call[1].as_str().unwrap().to_owned());
        }
    }
    for text in texts {
        let bytes = text.as_bytes();
        let mut variants = vec![bytes.to_vec()];
        for at in 0..bytes.len() {
            variants.push([&bytes[..at], &bytes[at + 1..]].concat());
            variants.push([&bytes[..=at], &bytes[at..]].concat());
        }
        for variant in variants {
            let Ok(variant) = String::from_utf8(variant) else {
                continue;
            };
            if serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&variant).is_ok()
            {
                assert!(json_object_text_v1(&variant), "{variant}");
            }
        }
    }
}
