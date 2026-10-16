#![forbid(unsafe_code)]
//! What `bundle check-frontend` refuses, shown with model-owned formats
//! that misbehave over this fixture's declaration and tokenizer.
//!
//! Each format is a `FrontendV1` exported in process, as a package exports
//! its own, and each vector passes or fails the checks a served response
//! applies.

use bundle_tp4_fixture::{
    Tp4FixtureFormatV1, tp4_fixture_frontend_declaration_v1,
    tp4_fixture_tokenizer_configuration_v1, tp4_fixture_tokenizer_definition_v1,
};
use runtime_bundle_abi::frontend::{FrontendExportV1, FrontendParserV1, FrontendV1};
use runtime_bundle_abi::frontend_document as document;
use runtime_frontend::{FrontendDescriptorV1, tokenizer_from_components_v1};
use runtime_host::{FrontendCheckV1, read_render_vectors_v1, read_streaming_vectors_v1};

/// This fixture's id for a byte: its byte tokens follow its literal ones.
fn byte(value: u8) -> u32 {
    154_564 + u32::from(value)
}

/// The fixture's end of sequence, its one stop token.
const EOS: u32 = 154_820;

/// A format that renders every request in chat mode and parses with `P`.
struct Format<P>(std::marker::PhantomData<fn() -> P>);

trait OpenParser: FrontendParserV1 + Sized {
    fn open(request: &document::ParserOpenV1) -> Self;
}

impl<P: OpenParser> FrontendV1 for Format<P> {
    type Parser = P;

    fn open(_: &[u8]) -> Result<(Self, Vec<u8>), Vec<u8>> {
        Ok((
            Self(std::marker::PhantomData),
            document::OpenedV1::default().encode(),
        ))
    }

    fn render(&self, _: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
        Ok(document::RenderedV1 {
            segments: vec![document::SegmentV1::Text("x".into())],
            mode: document::ModeV1::Chat,
            effort: "standard".into(),
            reasoning_open: false,
            parser_state: Vec::new(),
        }
        .encode())
    }

    fn constraint(&self, _: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
        Ok(Vec::new())
    }

    fn open_parser(&self, request: &[u8]) -> Result<P, Vec<u8>> {
        Ok(P::open(&document::ParserOpenV1::decode(request).unwrap()))
    }
}

/// Streams a call under one name, then delivers it under another with the
/// same arguments.
struct RenamingParser {
    stream: bool,
}

impl OpenParser for RenamingParser {
    fn open(request: &document::ParserOpenV1) -> Self {
        Self {
            stream: request.stream_tool_calls,
        }
    }
}

impl FrontendParserV1 for RenamingParser {
    fn push(&mut self, input: &[u8], _: bool) -> Result<Vec<u8>, Vec<u8>> {
        let pushed = document::PushV1::decode(input).unwrap();
        let mut events = Vec::new();
        if self.stream {
            events.push(document::EventV1::ToolCallDeltas(vec![document::DeltaV1 {
                index: 0,
                name: Some("streamed_name"),
                arguments: "{}",
            }]));
        }
        events.push(document::EventV1::ToolCalls {
            range: 0..1,
            calls: vec![document::CallV1 {
                name: "final_name",
                arguments: "{}",
            }],
        });
        events.push(document::EventV1::Finish {
            reason: document::FinishReasonV1::ToolCalls,
            completion: None,
        });
        Ok(document::EventsV1 {
            events,
            cursor: pushed.keep + pushed.suffix.len() as u64,
        }
        .encode())
    }
}

/// Settles nothing until the final push, then delivers every byte as
/// content at once.
struct WithholdingParser;

impl OpenParser for WithholdingParser {
    fn open(_: &document::ParserOpenV1) -> Self {
        Self
    }
}

impl FrontendParserV1 for WithholdingParser {
    fn push(&mut self, input: &[u8], last: bool) -> Result<Vec<u8>, Vec<u8>> {
        let pushed = document::PushV1::decode(input).unwrap();
        // The host sends only new bytes; every byte so far is unsettled.
        let end = pushed.keep + pushed.suffix.len() as u64;
        let mut events = Vec::new();
        if last {
            if end > 0 {
                events.push(document::EventV1::Content(0..end));
            }
            events.push(document::EventV1::Finish {
                reason: document::FinishReasonV1::Stop,
                completion: None,
            });
        }
        Ok(document::EventsV1 {
            events,
            cursor: if last { end } else { 0 },
        }
        .encode())
    }
}

/// A check of `F` over this fixture's declaration and tokenizer.
fn check<F: FrontendV1>() -> FrontendCheckV1 {
    let component = tp4_fixture_frontend_declaration_v1();
    let declaration = FrontendDescriptorV1::from_component_bytes(&component).unwrap();
    let tokenizer = tokenizer_from_components_v1(
        &declaration,
        &tp4_fixture_tokenizer_definition_v1(),
        &tp4_fixture_tokenizer_configuration_v1(),
    )
    .unwrap();
    FrontendCheckV1::from_component_v1(&FrontendExportV1::in_process::<F>(), &component, tokenizer)
        .unwrap()
}

/// A served stream refuses final calls that do not repeat the calls it
/// streamed, so a check refuses them in the streamed delivery too.
#[test]
fn a_final_call_that_renames_its_streamed_call_fails() {
    let vectors = read_streaming_vectors_v1(
        format!(
            r#"[{{"id":"renamed","request":{{"messages":[{{"role":"user","content":"Hi"}}],
              "tools":[{{"type":"function","function":{{"name":"final_name",
                "parameters":{{"type":"object","properties":{{}}}}}}}}]}},
              "token_ids":[{}],
              "expect":{{"tool_calls":[{{"name":"final_name","arguments":"{{}}"}}],
                         "finish_reason":"tool_calls"}}}}]"#,
            byte(b'x')
        )
        .as_bytes(),
    )
    .unwrap();
    let refused = check::<Format<RenamingParser>>()
        .check_streaming_v1(&vectors[0])
        .unwrap_err();
    assert!(
        refused.starts_with("streamed: the transport refused the completion")
            && refused.ends_with("streamed tool arguments differ from final calls"),
        "{refused}"
    );
}

/// Output the format holds back until the end still adds up to the
/// expected completion, so only a vector's steps see that it was withheld.
#[test]
fn withheld_output_fails_only_the_steps_it_was_owed_to() {
    let tokens = [byte(b'H'), byte(b'i'), byte(b'!'), EOS];
    let vector = |steps: &str| {
        read_streaming_vectors_v1(
            format!(
                r#"[{{"id":"withheld","request":{{"messages":[{{"role":"user","content":"Hi"}}]}},
                  "token_ids":{tokens:?},
                  "expect":{{"content":"Hi!","finish_reason":"stop"}}{steps}}}]"#
            )
            .as_bytes(),
        )
        .unwrap()
        .remove(0)
    };
    let check = check::<Format<WithholdingParser>>();
    check.check_streaming_v1(&vector("")).unwrap();
    let stepped = vector(&format!(
        r#","steps":[{{"token_ids":[{}],"content":"H"}},{{"token_ids":[{}],"content":"i"}},
             {{"token_ids":[{}],"content":"!"}},{{"token_ids":[{EOS}]}}]"#,
        tokens[0], tokens[1], tokens[2]
    ));
    assert_eq!(
        check.check_streaming_v1(&stepped).unwrap_err(),
        r#"whole: step 0 content differs at byte 0: expected "H", got """#
    );
}

/// The fixture's own render vectors pass through its own export, and so do
/// its streaming vectors under default request settings, which skip special
/// tokens: the reasoning close and the tool-call markers reach the parser
/// because they are not special, and the stop token, whose bytes the host
/// withholds, ends a tool call.
#[test]
fn the_fixture_vectors_pass_through_its_own_export() {
    let check = check::<frontend_format::ChatFormatExportV1<Tp4FixtureFormatV1>>();
    let renders = read_render_vectors_v1(include_bytes!("frontend/render-vectors.json"))
        .expect("the fixture's render vectors read");
    assert!(!renders.is_empty());
    for vector in &renders {
        check
            .check_render_v1(vector)
            .unwrap_or_else(|error| panic!("{}: {error}", vector.id));
    }
    let vectors = read_streaming_vectors_v1(include_bytes!("frontend/streaming-vectors.json"))
        .expect("the fixture's streaming vectors read");
    for id in [
        "reasoning-closed-by-its-marker",
        "content-then-call-closed-by-its-stop",
        "reasoning-then-call-closed-by-its-stop",
    ] {
        assert!(vectors.iter().any(|vector| vector.id == id), "{id}");
    }
    for vector in &vectors {
        check
            .check_streaming_v1(vector)
            .unwrap_or_else(|error| panic!("{}: {error}", vector.id));
    }
}
