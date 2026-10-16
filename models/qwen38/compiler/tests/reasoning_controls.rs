//! What the reasoning controls of one request mean for this template, as
//! vLLM hands them to it: a top-level effort replaces the nested one and
//! sets `enable_thinking` unless the client sent it, and the template reads
//! only the keywords it names.

use frontend_format::FrontendEffortUseV1;
use model_qwen38::frontend::descriptor;
use model_qwen38_contract::frontend::Qwen38Format;
use runtime_bundle_abi::frontend::FrontendCallErrorV1;
use runtime_bundle_abi::frontend_document::RefusalV1;
use runtime_frontend::test_support::documents::PairedFrontendV1;
use runtime_frontend::test_support::reasoning::ask;
use runtime_frontend::{BundleChatStateV1, PromptModeV1};
use runtime_frontend_contract::{FrontendDescriptorV1, FrontendReasoningPolicyV1};
use runtime_server::ServerFrontendV1;

type Case<'a> = (
    &'a [(&'a str, bool)],
    Option<&'a str>,
    Option<&'a str>,
    Result<bool, ()>,
);

/// Each case renders thinking (true) or not (false), or is refused because
/// the template raises for its effort.
#[test]
fn reasoning_controls_mean_what_the_template_reads() {
    let descriptor = descriptor();
    let frontend = PairedFrontendV1::chat_format::<Qwen38Format>(descriptor.clone()).unwrap();
    let cases: &[Case<'_>] = &[
        // The template reads the effort only while thinking.
        (&[("enable_thinking", false)], Some("low"), None, Ok(false)),
        (&[("enable_thinking", false)], None, Some("low"), Ok(false)),
        (&[("enable_thinking", false)], None, Some("high"), Ok(false)),
        (&[("enable_thinking", true)], Some("none"), None, Err(())),
        (&[], None, Some("high"), Err(())),
        (&[], Some("high"), None, Err(())),
        (&[], Some("low"), None, Ok(true)),
        (&[], Some("none"), None, Ok(false)),
        (&[], Some("banana"), None, Err(())),
        // The template does not read `thinking`.
        (&[("thinking", false)], None, None, Ok(true)),
        (
            &[("thinking", true), ("enable_thinking", false)],
            None,
            None,
            Ok(false),
        ),
    ];
    for (keywords, top, nested, expected) in cases {
        let label = format!("{keywords:?} top {top:?} nested {nested:?}");
        let rendered = frontend.render_chat(ask(keywords, *top, *nested));
        match expected {
            Err(()) => {
                let error = rendered.expect_err(&label);
                assert_eq!(
                    error.parameter.as_deref(),
                    Some("reasoning_effort"),
                    "{label}"
                );
            }
            Ok(thinking) => {
                let rendered = rendered.unwrap_or_else(|error| panic!("{label}: {error}"));
                let state = BundleChatStateV1::from_state(&rendered.state).unwrap();
                // The template's generation prompts (lines 163-170).
                let (mode, prefix) = if *thinking {
                    (PromptModeV1::Thinking, "<|im_start|>assistant\n<think>\n")
                } else {
                    (
                        PromptModeV1::Chat,
                        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    )
                };
                assert_eq!(state.mode(), mode, "{label}");
                assert!(rendered.prompt.ends_with(prefix), "{label}");
                if !*thinking {
                    for effort in &descriptor.reasoning.efforts {
                        if !effort.preamble.is_empty() {
                            assert!(!rendered.prompt.contains(&effort.preamble), "{label}");
                        }
                    }
                }
            }
        }
    }
    // A top-level effort replaces the nested one, which then selects nothing.
    let low = frontend
        .render_chat(ask(&[], Some("low"), Some("medium")))
        .unwrap()
        .prompt;
    let effort = descriptor.reasoning.effort("low").unwrap();
    assert!(low.contains(&effort.preamble), "{low}");
}

/// The export opens over the shipped components, and refuses a reasoning
/// policy the template does not follow (lines 46 and 165): one that does not
/// think by default, refuses `enable_thinking: false`, or ignores the effort.
#[test]
fn the_export_opens_only_over_the_templates_reasoning_policy() {
    let export = frontend_format::chat_format_export_v1::<Qwen38Format>;
    let shipped = serde_json::to_vec(&descriptor()).unwrap();
    for component in [shipped, model_qwen38::frontend::multimodal_component()] {
        export()
            .open(&component)
            .map(|_| ())
            .unwrap_or_else(|error| panic!("the shipped component opens: {error}"));
    }
    let edits: [fn(&mut FrontendReasoningPolicyV1); 3] = [
        |reasoning| reasoning.default_enabled = false,
        |reasoning| reasoning.allow_disabled = false,
        |reasoning| reasoning.effort_use = FrontendEffortUseV1::Ignored,
    ];
    for edit in edits {
        let mut declared = descriptor();
        edit(&mut declared.reasoning);
        // The host reads the changed declaration; the format refuses it.
        let component = serde_json::to_vec(&declared).unwrap();
        FrontendDescriptorV1::from_component_bytes(&component).unwrap();
        let error = export().open(&component).map(|_| ()).unwrap_err();
        let FrontendCallErrorV1::Refused(refusal) = error else {
            panic!("not a refusal: {error}");
        };
        let refusal = RefusalV1::decode(&refusal).unwrap();
        assert!(
            refusal.message.contains("thinks by default"),
            "{}",
            refusal.message
        );
    }
}
