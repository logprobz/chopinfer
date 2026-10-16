#[allow(dead_code)]
#[path = "../src/checkpoint_index.rs"]
mod checkpoint_index;
use frontend_format::PromptModeV1;
use frontend_format::test_support::chat::Golden;
use model_qwen38::{config, local_frontend, tokenizer};
use model_qwen38_contract::frontend::Qwen38Format;
use runtime_frontend::test_support::documents::{PairedFrontendV1, golden_request};
use runtime_server::{FrontendRoleV1, ServerFrontendV1};
mod support;

use model_qwen38::frontend::descriptor;
use model_qwen38::tokenizer::PROMPT_TOKENS;

/// Every captured chat renders the HF prompt and token ids through the
/// bundle's chat format, called directly and through its export.
#[test]
fn captured_chat_prompts_match_text_and_tokens() {
    for checkpoint in support::checkpoints() {
        let tokenizer = checkpoint.tokenizer();
        let descriptor = descriptor();
        let frontend = PairedFrontendV1::chat_format::<Qwen38Format>(descriptor.clone()).unwrap();
        let mut goldens: Vec<Golden> = checkpoint.fixture("chat-goldens.json");
        assert_eq!(goldens.len(), checkpoint.chat_cases);
        assert_eq!(
            checkpoint
                .fixture::<Vec<serde_json::Value>>("tokenizer-golden.json")
                .len(),
            checkpoint.tokenizer_cases
        );
        goldens.extend(checkpoint.fixture::<Vec<Golden>>("chat-integers.json"));
        goldens.extend(checkpoint.fixture::<Vec<Golden>>("chat-floats.json"));
        for golden in goldens {
            let arguments: fn(&serde_json::Value) -> String =
                |args| serde_json::to_string(args).unwrap();
            let mut request = golden_request(&frontend, &golden, "reasoning_content", arguments);
            // A capture leaves out empty content and tool call IDs.
            for message in &mut request.messages {
                message.content.get_or_insert_default();
                if message.role == FrontendRoleV1::Tool {
                    message.tool_call_id.get_or_insert_default();
                }
            }
            let text = frontend
                .render_chat(request)
                .unwrap_or_else(|e| panic!("{}: {e}", golden.id))
                .prompt;
            assert_eq!(text, golden.rendered_text, "{}", golden.id);
            if let Some(tokenizer) = &tokenizer {
                assert_eq!(
                    tokenizer.encode(&text, PROMPT_TOKENS).unwrap(),
                    golden.token_ids,
                    "{}",
                    golden.id
                );
            }
        }
    }
}

#[test]
fn descriptor_tokenizer_and_text_scope() {
    let d = descriptor();
    assert!(d.multimodal.is_none());
    assert_eq!(d.context.maximum_total_tokens, 262144);
    assert_eq!(d.vocabulary.stop_token_ids, [248046, 248044]);
    assert_eq!(
        runtime_frontend::tokenizer_contract_v1(&d).unwrap(),
        model_qwen38::tokenizer::tokenizer_contract().unwrap()
    );
}

#[test]
fn canonical_reasoning_and_multiple_xml_calls_decode() {
    let d = frontend_format::FormatDescriptorV1::new(descriptor()).unwrap();
    let completion = concat!(
        "plan\n</think>\n\nchecking\n\n",
        "<tool_call>\n<function=lookup>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n",
        "<tool_call>\n<function=lookup>\n<parameter=city>\n北京\n</parameter>\n</function>\n</tool_call><|im_end|>"
    );
    let parsed =
        frontend_format::parse_completion_v1(&d, completion, PromptModeV1::Thinking).unwrap();
    assert_eq!(parsed.reasoning_content.as_deref(), Some("plan\n"));
    assert_eq!(parsed.content, "\n\nchecking\n\n");
    assert_eq!(parsed.tool_calls.len(), 2);
    assert_eq!(parsed.tool_calls[0].arguments, r#"{"city": "Paris"}"#);
    assert_eq!(parsed.tool_calls[1].arguments, r#"{"city": "北京"}"#);
    assert!(
        frontend_format::parse_completion_v1(&d, "answer<|endoftext|>", PromptModeV1::Chat).is_ok()
    );
}

#[test]
fn streaming_calls_preserve_newlines_and_separators_at_every_byte_split() {
    use runtime_frontend::test_support::documents::PairedDecoderV1;
    use runtime_frontend::{CompletionUpdateV1, DecoderLimitsV1, GeneratedTokenV1};
    use std::sync::Arc;
    let text = concat!(
        "<tool_call>\n<function=lookup>\n<parameter=city>\n\nParis\n\n</parameter>\n</function>\n</tool_call>\n",
        "<tool_call>\n<function=lookup>\n<parameter=city>\n北京\n</parameter>\n</function>\n</tool_call><|im_end|>"
    );
    let d = Arc::new(descriptor());
    let expected = runtime_frontend::parse_completion_v1(&d, text, PromptModeV1::Chat).unwrap();
    for split in 0..=text.len() {
        let mut decoder = PairedDecoderV1::new(
            Arc::clone(&d),
            PromptModeV1::Chat,
            DecoderLimitsV1::declared(&d),
        );
        decoder.set_decoding_options(
            runtime_server::DecodingOptions {
                stream_tool_calls: true,
                ..Default::default()
            },
            false,
        );
        let mut streamed = [String::new(), String::new()];
        let mut finished = false;
        for (index, bytes) in [&text.as_bytes()[..split], &text.as_bytes()[split..]]
            .into_iter()
            .filter(|bytes| !bytes.is_empty())
            .enumerate()
        {
            for update in decoder
                .push(GeneratedTokenV1::new(index as u64, 42, bytes.to_vec()))
                .unwrap()
            {
                match update {
                    CompletionUpdateV1::ToolCallDeltas(deltas) => {
                        for delta in deltas {
                            streamed[delta.index].push_str(&delta.arguments);
                        }
                    }
                    CompletionUpdateV1::Finished(end) => {
                        assert_eq!(end.completion.as_ref(), Some(&expected));
                        finished = true;
                    }
                    _ => {}
                }
            }
        }
        assert!(finished, "split {split}");
        for (index, call) in expected.tool_calls.iter().enumerate() {
            assert_eq!(streamed[index], call.arguments, "split {split}");
        }
    }
}
