//! Post-end bytes use the served decoder and the pinned frontend export,
//! without loading a checkpoint or a tokenizer.

use std::sync::Arc;

use model_deepseek41::frontend::{COMPONENT, descriptor};
use model_deepseek41_contract::frontend::DeepSeekV41Format;
use model_deepseek41_contract::frontend::encoding::{
    ASSISTANT, BOS, DSML, EOS, THINKING_END, USER,
};
use runtime_frontend::{
    BundleParserSourceV1, CompletionDecoderV1, CompletionFinishV1, CompletionUpdateV1,
    DecoderLimitsV1, GeneratedTokenV1, PromptModeV1,
};

const RESERVED: &str = "malformed completion: completion contains an unexpected reserved boundary";

fn decoder(ignore_eos: bool) -> CompletionDecoderV1 {
    let declared = Arc::new(descriptor());
    let mut decoder = CompletionDecoderV1::new(
        Arc::clone(&declared),
        PromptModeV1::Thinking,
        DecoderLimitsV1::declared(&declared),
    );
    decoder.set_parser_source(
        BundleParserSourceV1::from_export(
            &frontend_format::chat_format_export_v1::<DeepSeekV41Format>(),
            COMPONENT,
        )
        .unwrap(),
    );
    decoder.set_decoding_options(
        runtime_server::DecodingOptions {
            ignore_eos,
            ..Default::default()
        },
        false,
    );
    decoder.set_reasoning_open(true);
    decoder.set_tool_extraction(false);
    decoder
}

fn token(ordinal: usize, text: &str) -> GeneratedTokenV1 {
    let id = descriptor()
        .vocabulary
        .boundary_tokens
        .iter()
        .find(|boundary| boundary.text == text)
        .map_or(200, |boundary| boundary.id);
    GeneratedTokenV1::new(ordinal as u64, id, text.as_bytes().to_vec())
        .with_special(matches!(text, EOS | BOS))
}

fn calls() -> Vec<&'static str> {
    vec![
        "\n\n<",
        DSML,
        " calls>\n<",
        DSML,
        " invoke name=\"f\">\n</",
        DSML,
        " invoke>\n</",
        DSML,
        " calls>",
    ]
}

#[test]
fn ignored_eos_makes_investigated_tails_answer_text() {
    for tail in [
        vec![DSML],
        vec!["plain text"],
        vec![USER, "question", ASSISTANT, "answer"],
        vec![THINKING_END],
        calls(),
    ] {
        // Cover both a closed reasoning section and EOS while still thinking.
        for close_reasoning in [true, false] {
            let mut decoder = decoder(true);
            let mut parts = vec!["why"];
            if close_reasoning {
                parts.extend([THINKING_END, "answer"]);
            }
            parts.push(EOS);
            let tail_ordinal = parts.len();
            parts.extend(tail.iter().copied());
            let mut updates = Vec::new();
            for (ordinal, part) in parts.iter().enumerate() {
                updates.extend(
                    decoder
                        .push_engine_token(token(ordinal, part))
                        .unwrap_or_else(|error| panic!("tail {tail:?}, token {ordinal}: {error}")),
                );
            }
            updates.extend(decoder.finish_length().unwrap());
            let mut content = String::new();
            let mut reasoning = String::new();
            let mut tail_tokens = Vec::new();
            let mut finishes = Vec::new();
            for update in updates {
                match update {
                    CompletionUpdateV1::Content(part) => {
                        content.push_str(&part.text);
                        tail_tokens.extend(
                            part.contributions
                                .into_iter()
                                .filter(|each| each.ordinal >= tail_ordinal as u64)
                                .map(|each| each.ordinal),
                        );
                    }
                    CompletionUpdateV1::Reasoning(part) => reasoning.push_str(&part.text),
                    CompletionUpdateV1::Finished(done) => finishes.push(done),
                    other => panic!("post-end markup produced {other:?}"),
                }
            }
            assert_eq!(reasoning, "why");
            assert_eq!(
                content,
                format!(
                    "{}{}",
                    if close_reasoning { "answer" } else { "" },
                    tail.concat()
                )
            );
            assert_eq!(
                tail_tokens,
                (tail_ordinal as u64..parts.len() as u64).collect::<Vec<_>>()
            );
            assert_eq!(finishes.len(), 1);
            assert_eq!(finishes[0].reason, CompletionFinishV1::Length);
            assert_eq!(finishes[0].generated_token_count, parts.len());
            assert!(finishes[0].completion.is_none());
        }
    }
}

#[test]
fn markup_before_eos_keeps_its_parser_rules() {
    for ignore_eos in [false, true] {
        let mut decoder = decoder(ignore_eos);
        decoder.push_engine_token(token(0, THINKING_END)).unwrap();
        assert_eq!(
            decoder.push_engine_token(token(1, DSML)).unwrap_err(),
            RESERVED
        );
    }
    let mut decoder = decoder(false);
    let mut parts = vec![THINKING_END];
    parts.extend(calls());
    parts.push(EOS);
    let updates: Vec<_> = parts
        .iter()
        .enumerate()
        .flat_map(|(ordinal, part)| decoder.push_engine_token(token(ordinal, part)).unwrap())
        .collect();
    assert!(matches!(
        updates.last(),
        Some(CompletionUpdateV1::Finished(done))
            if done.reason == CompletionFinishV1::Stop && done.generated_token_count == parts.len()
    ));
    assert!(!updates.iter().any(|update| matches!(
        update,
        CompletionUpdateV1::ToolCalls(_) | CompletionUpdateV1::ToolCallDeltas(_)
    )));
}
