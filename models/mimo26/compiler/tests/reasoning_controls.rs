//! What the reasoning controls of one request mean for this template, as
//! vLLM hands them to it: a top-level effort replaces the nested one and
//! sets `enable_thinking` unless the client sent it, and the template reads
//! only the keywords it names.

use frontend_format::FrontendEffortUseV1;
use model_mimo26::frontend::{component_for_flash, descriptor};
use model_mimo26_contract::encoders::Encoders;
use model_mimo26_contract::frontend::Mimo26Format;
use runtime_bundle_abi::frontend::FrontendCallErrorV1;
use runtime_bundle_abi::frontend_document::RefusalV1;
use runtime_frontend_contract::{FrontendDescriptorV1, FrontendReasoningPolicyV1};
#[path = "support/frontend.rs"]
mod frontend_support;
use frontend_support::frontend;
use runtime_frontend::test_support::reasoning::ask;
use runtime_frontend::{BundleChatStateV1, PromptModeV1};
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
    let descriptor = frontend_format::FormatDescriptorV1::new(descriptor()).unwrap();
    // The generation prompt of each mode (template lines 90-95).
    let (thinking_prefix, chat_prefix) = (
        "<|im_start|>assistant\n",
        "<|im_start|>assistant\n<think></think>",
    );
    let frontend = frontend();
    let cases: &[Case<'_>] = &[
        // The template never reads the effort.
        (&[("enable_thinking", false)], Some("low"), None, Ok(false)),
        (
            &[("enable_thinking", false)],
            None,
            Some("default"),
            Ok(false),
        ),
        (&[("enable_thinking", false)], None, Some("low"), Ok(false)),
        (&[("enable_thinking", true)], Some("none"), None, Ok(true)),
        (&[], None, Some("high"), Ok(true)),
        (&[], Some("low"), None, Ok(true)),
        (&[], Some("max"), None, Ok(true)),
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
                let (mode, prefix) = if *thinking {
                    (PromptModeV1::Thinking, thinking_prefix)
                } else {
                    (PromptModeV1::Chat, chat_prefix)
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
}

/// The export opens over the component of every media set a package may
/// serve, and refuses a reasoning policy the template does not follow (line
/// 92): one that reads another keyword, does not think by default, refuses
/// `enable_thinking: false`, reads the effort, or declares a preamble.
#[test]
fn the_export_opens_only_over_the_templates_reasoning_policy() {
    let export = frontend_format::chat_format_export_v1::<Mimo26Format>;
    for encoders in Encoders::SETS {
        let component =
            component_for_flash(model_mimo26::checkpoint::default_identity(), encoders).unwrap();
        export()
            .open(&component)
            .map(|_| ())
            .unwrap_or_else(|error| panic!("{encoders:?} opens: {error}"));
    }
    let edits: [fn(&mut FrontendReasoningPolicyV1); 5] = [
        |reasoning| reasoning.thinking_keywords = vec!["thinking".into()],
        |reasoning| reasoning.default_enabled = false,
        |reasoning| reasoning.allow_disabled = false,
        |reasoning| reasoning.effort_use = FrontendEffortUseV1::WhenThinking,
        |reasoning| reasoning.efforts[0].preamble = "Think briefly.".into(),
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
            refusal.message.contains("unless enable_thinking is false"),
            "{}",
            refusal.message
        );
    }
}
