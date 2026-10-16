//! Shared streaming decoder cases, instantiated by each frontend owner.

/// Exercise a frontend with its tokenizer and authenticated whitespace fixture.
#[macro_export]
macro_rules! streaming_text_tests {
    ($model:ident, $tokenizer_env:expr, $whitespace:expr) => {
        mod cases {
            use $model as model;
            const TOKENIZER_ENV: &str = $tokenizer_env;
            use runtime_frontend::{
                CompletionDecoderV1, CompletionFinishV1, CompletionUpdateV1, DecoderLimitsV1,
                GeneratedTokenV1, PromptModeV1,
            };
            use runtime_server::DecodingOptions;
            use std::{
                path::Path,
                sync::{Arc, OnceLock},
            };

            fn tokenizer() -> Option<Arc<runtime_tokenizer::Tokenizer>> {
                // These cases exercise decoding against one immutable authenticated fixture.
                // Share its vocabulary instead of parsing it again in every parallel test.
                static TOKENIZER: OnceLock<Option<Arc<runtime_tokenizer::Tokenizer>>> =
                    OnceLock::new();
                TOKENIZER
                    .get_or_init(|| {
                        let root = std::env::var(TOKENIZER_ENV)
                            .or_else(|_| {
                                std::env::var(
                                    TOKENIZER_ENV.replace("_CHECKPOINT", "_TOKENIZER_DIR"),
                                )
                            })
                            .ok()?;
                        Some(Arc::new(
                            model::tokenizer::load_tokenizer(Path::new(&root)).unwrap(),
                        ))
                    })
                    .clone()
            }

            fn decoder(stops: &[&str], stream: bool, include: bool) -> CompletionDecoderV1 {
                let d = Arc::new(model::frontend::descriptor());
                let limits = DecoderLimitsV1::declared(&d);
                let mut decoder = if stops.is_empty() {
                    CompletionDecoderV1::new(d, PromptModeV1::Chat, limits)
                } else {
                    CompletionDecoderV1::with_client_stops(
                        d,
                        PromptModeV1::Chat,
                        limits,
                        &stops.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                    )
                    .unwrap()
                };
                decoder.set_decoding_options(
                    DecodingOptions {
                        stream_tool_calls: stream,
                        include_stop_str_in_output: include,
                        ..Default::default()
                    },
                    true,
                );
                decoder
            }

            #[derive(Default, Debug)]
            struct Output {
                text: String,
                reasoning: String,
                deltas: Vec<String>,
                reason: Option<CompletionFinishV1>,
                count: usize,
            }
            impl Output {
                fn collect(&mut self, updates: Vec<CompletionUpdateV1>) {
                    for update in updates {
                        match update {
                            CompletionUpdateV1::Content(s) => {
                                self.text.push_str(&s.text);
                                self.deltas.push(s.text);
                            }
                            CompletionUpdateV1::Reasoning(s) => self.reasoning.push_str(&s.text),
                            CompletionUpdateV1::Finished(f) => {
                                self.reason = Some(f.reason);
                                self.count = f.generated_token_count;
                            }
                            _ => {}
                        }
                    }
                }
            }
            fn run(
                t: &runtime_tokenizer::Tokenizer,
                d: &mut CompletionDecoderV1,
                ids: &[u32],
            ) -> Output {
                let mut out = Output::default();
                for (i, id) in ids.iter().enumerate() {
                    out.collect(
                        d.push_engine_token(
                            GeneratedTokenV1::new(
                                i as u64,
                                *id,
                                t.token_bytes(*id).unwrap().to_vec(),
                            )
                            .with_special(t.token_is_special(*id).unwrap()),
                        )
                        .unwrap(),
                    );
                    if out.reason.is_some() {
                        break;
                    }
                }
                if out.reason.is_none() {
                    out.collect(d.finish_length().unwrap());
                }
                out
            }

            #[test]
            fn completed_stops_use_completion_position_then_request_order() {
                let Some(t) = tokenizer() else { return };
                for stream in [false, true] {
                    for include in [false, true] {
                        for (stops, ids, expected, count) in [
                            (
                                vec!["abcde", "bc"],
                                vec![64, 65, 66, 67, 68, 55],
                                if include { "abc" } else { "a" },
                                3,
                            ),
                            (
                                vec!["abcde", "bc"],
                                vec![64, 65, 66],
                                if include { "abc" } else { "a" },
                                3,
                            ),
                            (
                                vec!["bc", "abc"],
                                vec![64, 65, 66, 55],
                                if include { "abc" } else { "a" },
                                3,
                            ),
                        ] {
                            let out = run(&t, &mut decoder(&stops, stream, include), &ids);
                            assert_eq!(out.text, expected, "{stops:?} {out:?}");
                            assert_eq!(out.reason, Some(CompletionFinishV1::Stop));
                            assert_eq!(out.count, count);
                        }
                    }
                }
            }

            #[test]
            fn stops_are_applied_before_reasoning_and_tool_markers() {
                let Some(t) = tokenizer() else { return };
                let d = Arc::new(model::frontend::descriptor());
                let allowed: Vec<&str> = d
                    .vocabulary
                    .prompt_special_tokens
                    .iter()
                    .map(String::as_str)
                    .collect();
                for stream in [false, true] {
                    for include in [false, true] {
                        for (source, stop, reasoning, content, count) in [
                            (
                                "plan</think>answer",
                                "plan</think>ans",
                                if include { "plan" } else { "" },
                                if include { "ans" } else { "" },
                                3,
                            ),
                            (
                                "<think>plan</think>answer",
                                "think",
                                if include { "<think" } else { "<" },
                                "",
                                1,
                            ),
                        ] {
                            let ids = t.encode(source, &allowed).unwrap();
                            let mut dec = CompletionDecoderV1::with_client_stops(
                                d.clone(),
                                PromptModeV1::Thinking,
                                DecoderLimitsV1::declared(&d),
                                &[stop.to_owned()],
                            )
                            .unwrap();
                            dec.set_decoding_options(
                                DecodingOptions {
                                    stream_tool_calls: stream,
                                    include_stop_str_in_output: include,
                                    skip_special_tokens: false,
                                    ..Default::default()
                                },
                                false,
                            );
                            let out = run(&t, &mut dec, &ids);
                            assert_eq!(out.reasoning, reasoning, "{source} {out:?}");
                            assert_eq!(out.text, content, "{source} {out:?}");
                            assert_eq!(out.reason, Some(CompletionFinishV1::Stop));
                            assert_eq!(out.count, count);
                        }
                    }
                }
            }

            #[test]
            fn unfinished_and_replacement_tails_stay_buffered_at_finish() {
                let Some(t) = tokenizer() else { return };
                let replacement = t.encode("�", &[]).unwrap();
                for stream in [false, true] {
                    for (ids, expected, deltas) in [
                        (vec![172], "", vec![]),
                        (vec![32, 172], "A", vec!["A"]),
                        (vec![187], "", vec![]),
                        (replacement.clone(), "", vec![]),
                        (vec![187, 32], "�A", vec!["�A"]),
                        ([replacement.clone(), vec![32]].concat(), "�A", vec!["�A"]),
                        (vec![172, 253, 99, 222], "🦀", vec!["🦀"]),
                    ] {
                        for eos in [false, true] {
                            let mut ids = ids.clone();
                            if eos {
                                ids.push(
                                    model::frontend::descriptor().vocabulary.stop_token_ids[0],
                                );
                            }
                            let out = run(&t, &mut decoder(&[], stream, false), &ids);
                            assert_eq!(out.text, expected, "{ids:?} {out:?}");
                            assert_eq!(out.deltas, deltas, "{ids:?}");
                            assert_eq!(
                                out.reason,
                                Some(if eos {
                                    CompletionFinishV1::Stop
                                } else {
                                    CompletionFinishV1::Length
                                })
                            );
                        }
                    }
                }
            }

            #[test]
            fn tokenizer_flags_control_added_token_visibility() {
                let Some(t) = tokenizer() else { return };
                let d = Arc::new(model::frontend::descriptor());
                for stream in [false, true] {
                    for skip in [false, true] {
                        for token in &d.vocabulary.boundary_tokens {
                            let mut dec = decoder(&[], stream, false);
                            dec.set_decoding_options(
                                DecodingOptions {
                                    skip_special_tokens: skip,
                                    ignore_eos: true,
                                    stream_tool_calls: stream,
                                    ..Default::default()
                                },
                                true,
                            );
                            let out = run(&t, &mut dec, &[token.id, 55]);
                            let expected = if skip && t.token_is_special(token.id).unwrap() {
                                "X".to_owned()
                            } else {
                                format!("{}X", token.text)
                            };
                            assert_eq!(out.text, expected, "{} skip={skip}", token.text);
                            assert_eq!(out.count, 2);
                        }
                        for marker in ["<|im_start|>", "<|image_pad|>", "<|fim_prefix|>", "<think>"]
                        {
                            let token = d
                                .vocabulary
                                .boundary_tokens
                                .iter()
                                .find(|b| b.text == marker)
                                .unwrap();
                            let mut dec = decoder(&[], stream, false);
                            dec.set_decoding_options(
                                DecodingOptions {
                                    skip_special_tokens: skip,
                                    stream_tool_calls: stream,
                                    ..Default::default()
                                },
                                false,
                            );
                            let out = run(&t, &mut dec, &[token.id, 55]);
                            assert_eq!(
                                out.text,
                                if skip && t.token_is_special(token.id).unwrap() {
                                    "X".to_owned()
                                } else {
                                    format!("{marker}X")
                                }
                            );
                        }
                    }
                }
                for (marker, stop, expected, reason) in [
                    (
                        "<|fim_prefix|>",
                        "fim_prefix",
                        "<|",
                        CompletionFinishV1::Stop,
                    ),
                    ("<|im_start|>", "im_start", "X", CompletionFinishV1::Length),
                ] {
                    let token = d
                        .vocabulary
                        .boundary_tokens
                        .iter()
                        .find(|b| b.text == marker)
                        .unwrap();
                    let out = run(&t, &mut decoder(&[stop], false, false), &[token.id, 55]);
                    assert_eq!(out.text, expected);
                    assert_eq!(out.reason, Some(reason));
                }
            }

            #[test]
            fn prompt_context_completes_bytes_without_counting_prompt_tokens() {
                let Some(t) = tokenizer() else { return };
                for stream in [false, true] {
                    for (prompt, ids, expected) in [
                        (vec![172], vec![253, 99, 222], "🦀"),
                        (vec![172], vec![32], "�A"),
                        (vec![32, 172], vec![253, 99, 222], "🦀"),
                        (vec![32], vec![33], "B"),
                    ] {
                        let mut dec = decoder(&[], stream, false);
                        dec.prime_prompt_bytes(&t.decode_bytes(&prompt).unwrap())
                            .unwrap();
                        let out = run(&t, &mut dec, &ids);
                        assert_eq!(out.text, expected);
                        assert_eq!(out.count, ids.len());
                    }
                    let special = model::frontend::descriptor()
                        .vocabulary
                        .boundary_tokens
                        .iter()
                        .find(|b| b.text == "<|im_start|>")
                        .unwrap()
                        .id;
                    let mut dec = decoder(&[], stream, false);
                    dec.prime_prompt_bytes(&[0xf0]).unwrap();
                    let out = run(&t, &mut dec, &[special, 253, 99, 222]);
                    assert_eq!(out.text, "🦀");
                    assert_eq!(out.count, 4);
                    let mut dec = decoder(&[], stream, false);
                    dec.prime_prompt_bytes(&[0xf0]).unwrap();
                    let out = run(
                        &t,
                        &mut dec,
                        &[model::frontend::descriptor().vocabulary.stop_token_ids[0]],
                    );
                    assert_eq!(out.text, "");
                    assert_eq!(out.count, 1);
                }
            }

            #[test]
            fn token_strings_retain_byte_alphabet_and_added_token_spellings() {
                let Some(t) = tokenizer() else { return };
                let d = model::frontend::descriptor();
                let allowed: Vec<&str> = d
                    .vocabulary
                    .prompt_special_tokens
                    .iter()
                    .map(String::as_str)
                    .collect();
                for (text, expected) in [
                    (" ", vec!["Ġ"]),
                    ("\n", vec!["Ċ"]),
                    ("🫨", vec!["ðŁ", "«", "¨"]),
                    ("<think>", vec!["<think>"]),
                    ("<|im_start|>", vec!["<|im_start|>"]),
                ] {
                    let ids = t.encode(text, &allowed).unwrap();
                    let actual: Vec<&str> =
                        ids.iter().map(|id| t.token_string(*id).unwrap()).collect();
                    assert_eq!(actual, expected);
                }
                assert!(t.token_string(u32::MAX).is_err());
                assert_eq!(
                    String::from_utf8_lossy(&t.decode_bytes(&[172]).unwrap()),
                    "�"
                );
            }

            #[test]
            fn adjacent_tokens_preserve_byte_level_spacing_for_both_options() {
                let Some(t) = tokenizer() else { return };
                let d = model::frontend::descriptor();
                let allowed: Vec<&str> = d
                    .vocabulary
                    .prompt_special_tokens
                    .iter()
                    .map(String::as_str)
                    .collect();
                for text in ["<|fim_prefix|><|fim_middle|>", "<|im_start|><|image_pad|>"] {
                    let ids = t.encode(text, &allowed).unwrap();
                    for spaces in [false, true] {
                        for skip in [false, true] {
                            for raw in [false, true] {
                                for stream in [false, true] {
                                    let mut dec = decoder(&[], stream, false);
                                    dec.set_decoding_options(
                                        DecodingOptions {
                                            spaces_between_special_tokens: spaces,
                                            skip_special_tokens: skip,
                                            stream_tool_calls: stream,
                                            ..Default::default()
                                        },
                                        raw,
                                    );
                                    let out = run(&t, &mut dec, &ids);
                                    assert_eq!(
                                        out.text,
                                        if skip && text.starts_with("<|im_start|>") {
                                            ""
                                        } else {
                                            text
                                        }
                                    );
                                    assert_eq!(out.count, ids.len());
                                }
                            }
                        }
                    }
                }
            }

            #[test]
            fn long_whitespace_matches_hf_through_the_input_limit() {
                let Some(t) = tokenizer() else { return };
                // Captured with HF tokenizers 0.23.2 and both authenticated artifacts.
                let cases: serde_json::Value = serde_json::from_str($whitespace).unwrap();
                for case in cases[TOKENIZER_ENV].as_array().unwrap() {
                    let mut input = " ".repeat(case["spaces"].as_u64().unwrap() as usize);
                    input.push_str(case["suffix"].as_str().unwrap());
                    let expected: Vec<u32> = case["runs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|run| {
                            std::iter::repeat_n(
                                run[0].as_u64().unwrap() as u32,
                                run[1].as_u64().unwrap() as usize,
                            )
                        })
                        .collect();
                    assert_eq!(t.encode(&input, &[]).unwrap(), expected, "{case}");
                }
            }

            #[test]
            fn stops_withhold_all_channels_across_every_marker_split() {
                let d = Arc::new(model::frontend::descriptor());
                for (source, stop, mode, prefix) in [
                    (
                        "plan</think>answer",
                        "plan</think>ans",
                        PromptModeV1::Thinking,
                        "",
                    ),
                    (
                        "plan</think>answer",
                        "/think",
                        PromptModeV1::Thinking,
                        "plan<",
                    ),
                    (
                        "before<tool_call><function=ping></function></tool_call>",
                        "before<tool_call>",
                        PromptModeV1::Chat,
                        "",
                    ),
                ] {
                    for stream in [false, true] {
                        for split in 1..source.len() {
                            let mut dec = CompletionDecoderV1::with_client_stops(
                                d.clone(),
                                mode,
                                DecoderLimitsV1::declared(&d),
                                &[stop.to_owned()],
                            )
                            .unwrap();
                            dec.set_decoding_options(
                                DecodingOptions {
                                    stream_tool_calls: stream,
                                    ..Default::default()
                                },
                                false,
                            );
                            let mut out = Output::default();
                            for (i, bytes) in
                                [&source.as_bytes()[..split], &source.as_bytes()[split..]]
                                    .iter()
                                    .enumerate()
                            {
                                let updates = dec
                                    .push_engine_token(GeneratedTokenV1::new(
                                        i as u64,
                                        42,
                                        bytes.to_vec(),
                                    ))
                                    .unwrap();
                                assert!(!updates.iter().any(|u| matches!(
                                    u,
                                    CompletionUpdateV1::ToolCalls(_)
                                        | CompletionUpdateV1::ToolCallDeltas(_)
                                )));
                                out.collect(updates);
                                if out.reason.is_some() {
                                    break;
                                }
                            }
                            assert_eq!(
                                format!("{}{}", out.reasoning, out.text),
                                prefix,
                                "{source} split={split}"
                            );
                            assert_eq!(out.reason, Some(CompletionFinishV1::Stop));
                        }
                    }
                }
                // Exhausting the budget releases pending text, including partial markup.
                for end in 1.."plan</think>ans".len() {
                    let source = &"plan</think>answer"[..end];
                    let mut dec = CompletionDecoderV1::with_client_stops(
                        d.clone(),
                        PromptModeV1::Thinking,
                        DecoderLimitsV1::declared(&d),
                        &["plan</think>ans".to_owned()],
                    )
                    .unwrap();
                    let mut out = Output::default();
                    out.collect(
                        dec.push_engine_token(GeneratedTokenV1::new(
                            0,
                            42,
                            source.as_bytes().to_vec(),
                        ))
                        .unwrap(),
                    );
                    assert!(out.reasoning.is_empty() && out.text.is_empty());
                    out.collect(dec.finish_length().unwrap());
                    if end < "plan</think>".len() {
                        assert_eq!(out.reasoning, source);
                        assert!(out.text.is_empty());
                    } else {
                        assert_eq!(out.reasoning, "plan");
                        assert_eq!(out.text, &source["plan</think>".len()..]);
                    }
                    assert_eq!(out.reason, Some(CompletionFinishV1::Length));
                }
            }
        }
    };
}
