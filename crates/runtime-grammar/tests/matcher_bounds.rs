//! Bound recursive matching both when committing bytes and scanning tokens.
#![cfg(target_os = "linux")]

mod common;

use common::bounded;
use runtime_grammar::{GrammarError, GrammarMatcher, JsonOptions, json_schema_grammar};
use serde_json::json;

#[test]
fn ambiguous_recursion_is_bounded() {
    bounded("ambiguous_recursion_is_bounded", || {
        let schema = json!({"$defs": {
            "a": {"type":"array", "items":{"$ref":"#/$defs/u"}},
            "b": {"type":"array", "items":{"$ref":"#/$defs/u"}, "minItems":1},
            "u": {"anyOf":[{"$ref":"#/$defs/a"},{"$ref":"#/$defs/b"}]}
        }, "$ref":"#/$defs/u"});
        let grammar = json_schema_grammar(&schema, JsonOptions::default()).unwrap();
        let vocabulary = std::sync::Arc::new(
            runtime_grammar::TokenVocabulary::new(
                2,
                vec![
                    runtime_grammar::VocabularyToken {
                        bytes: vec![b'['; 32],
                        special: false,
                    },
                    runtime_grammar::VocabularyToken {
                        bytes: vec![b'['],
                        special: false,
                    },
                ],
            )
            .unwrap(),
        );
        let compiled = common::compile(&grammar, &vocabulary);
        let mut matcher = GrammarMatcher::new(compiled);
        for _ in 0..32 {
            match matcher.accept_bytes(b"[") {
                Err(GrammarError::TooLarge(_)) => return,
                Err(error) => panic!("unexpected refusal: {error}"),
                Ok(()) => match matcher.fill_token_bitmask(&mut [0; 16]) {
                    Err(GrammarError::TooLarge(_)) => return,
                    Err(error) => panic!("unexpected mask failure: {error}"),
                    Ok(()) => {}
                },
            }
        }
        panic!("ambiguous recursion must hit the matcher budget");
    });
}

fn compile_schema(
    schema: serde_json::Value,
    extra: &[&[u8]],
) -> std::sync::Arc<runtime_grammar::CompiledGrammar> {
    let mut tokens: Vec<_> = (0..=255)
        .map(|byte| runtime_grammar::VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect();
    tokens.extend(extra.iter().map(|bytes| runtime_grammar::VocabularyToken {
        bytes: bytes.to_vec(),
        special: false,
    }));
    let vocabulary =
        std::sync::Arc::new(runtime_grammar::TokenVocabulary::new(tokens.len(), tokens).unwrap());
    common::compile(
        &json_schema_grammar(&schema, JsonOptions::default()).unwrap(),
        &vocabulary,
    )
}

#[test]
fn initial_exhaustion_never_disables_a_constraint() {
    let compiled = compile_schema(
        json!({"$defs":{"u":{"anyOf":[{"$ref":"#/$defs/u"},{"type":"null"}]}},"$ref":"#/$defs/u"}),
        &[],
    );
    for stops in [vec![], vec![255]] {
        for gate in [None, Some(vec![254])] {
            let mut state =
                runtime_grammar::ConstraintState::new(compiled.clone(), gate, stops.clone());
            let mut mask = vec![u32::MAX; compiled.vocabulary().mask_words()];
            assert!(matches!(
                state.fill_mask(&mut mask),
                Err(GrammarError::TooLarge(_))
            ));
            assert!(mask.iter().all(|word| *word == 0));
            assert!(state.is_active());
            assert!(!state.is_terminated());
            assert!(matches!(
                state.allows(u32::from(b'x')),
                Err(GrammarError::TooLarge(_))
            ));
            assert!(matches!(state.accept(254), Err(GrammarError::TooLarge(_))));
        }
    }
}

#[test]
fn scanner_exhaustion_fails_instead_of_removing_a_valid_token() {
    bounded(
        "scanner_exhaustion_fails_instead_of_removing_a_valid_token",
        || {
            // Both the 32-byte prefix and the one-byte prefix extend to nested
            // arrays ending with an empty array, which the first branch admits.
            let long = vec![b'['; 32];
            let compiled = compile_schema(
                json!({"$defs": {
            "a": {"type":"array", "items":{"$ref":"#/$defs/u"}},
            "b": {"type":"array", "items":{"$ref":"#/$defs/u"}, "minItems":1},
            "u": {"anyOf":[{"$ref":"#/$defs/a"},{"$ref":"#/$defs/b"}]}
        }, "$ref":"#/$defs/u"}),
                &[&long],
            );
            for precompute in [false, true] {
                if precompute {
                    assert!(matches!(
                        compiled.precompute(2),
                        Err(GrammarError::TooLarge(_))
                    ));
                }
                let state = runtime_grammar::ConstraintState::new(compiled.clone(), None, vec![]);
                let mut mask = vec![u32::MAX; compiled.vocabulary().mask_words()];
                assert!(matches!(
                    state.fill_mask(&mut mask),
                    Err(GrammarError::TooLarge(_))
                ));
                assert!(mask.iter().all(|word| *word == 0));
            }
        },
    );
}

#[test]
fn depth_exhaustion_is_an_error_even_with_other_valid_paths() {
    bounded(
        "depth_exhaustion_is_an_error_even_with_other_valid_paths",
        || {
            for (schema, prefix) in [
                (
                    json!({"$defs":{"u":{"anyOf":[{"type":"null"},{"type":"array","items":{"$ref":"#/$defs/u"}}]}},"$ref":"#/$defs/u"}),
                    &b"["[..],
                ),
                (
                    json!({"$defs":{"u":{"type":"object","properties":{"a":{"$ref":"#/$defs/u"},"b":{"type":"null"}}}},"$ref":"#/$defs/u"}),
                    &b"{\"a\":"[..],
                ),
            ] {
                // Every finite prefix has a completion: close all empty containers.
                let compiled = compile_schema(schema, &[]);
                let mut matcher = GrammarMatcher::new(compiled);
                for count in 0..4096 {
                    match matcher.accept_bytes(prefix) {
                        Ok(()) => assert!(count < 4095, "depth bound was not enforced"),
                        Err(GrammarError::TooLarge(_)) => break,
                        Err(error) => panic!("a valid recursive prefix was rejected: {error}"),
                    }
                }
            }
        },
    );
}

#[test]
fn activating_a_gate_cannot_hide_scanner_exhaustion() {
    let long = vec![b'['; 32];
    let compiled = compile_schema(
        json!({"$defs": {
        "a": {"type":"array", "items":{"$ref":"#/$defs/u"}},
        "b": {"type":"array", "items":{"$ref":"#/$defs/u"}, "minItems":1},
        "u": {"anyOf":[{"$ref":"#/$defs/a"},{"$ref":"#/$defs/b"}]}
    }, "$ref":"#/$defs/u"}),
        &[&long],
    );
    for gate in [254, u32::from(b'[')] {
        let mut state =
            runtime_grammar::ConstraintState::new(compiled.clone(), Some(vec![gate]), vec![]);
        let mut mask = vec![0; compiled.vocabulary().mask_words()];
        assert_eq!(state.fill_mask(&mut mask), Ok(false));
        state.accept(gate).unwrap();
        assert!(matches!(
            state.fill_mask(&mut mask),
            Err(GrammarError::TooLarge(_))
        ));
        assert!(matches!(
            state.allows(u32::from(b'[')),
            Err(GrammarError::TooLarge(_))
        ));
        assert!(matches!(
            state.accept(u32::from(b'[')),
            Err(GrammarError::TooLarge(_))
        ));
    }
    let matcher = GrammarMatcher::new(compiled);
    assert!(matches!(
        matcher.accepts_token(256),
        Err(GrammarError::TooLarge(_))
    ));
    assert!(matches!(
        matcher.accepts_token(u32::from(b'[')),
        Err(GrammarError::TooLarge(_))
    ));
}

#[test]
fn scanner_budget_diagnostics_survive_forks_and_rollback() {
    for (schema, length, expected) in [
        (
            json!({"$defs": {
            "a": {"type":"array", "items":{"$ref":"#/$defs/u"}},
            "b": {"type":"array", "items":{"$ref":"#/$defs/u"}, "minItems":1},
            "u": {"anyOf":[{"$ref":"#/$defs/a"},{"$ref":"#/$defs/b"}]}
        }, "$ref":"#/$defs/u"}),
            32,
            "grammar scanner work budget exceeded (limit: 4096 units per closure)",
        ),
        (
            json!({"$defs":{"u":{"anyOf":[{"type":"null"},{"type":"array","items":{"$ref":"#/$defs/u"}}]}},"$ref":"#/$defs/u"}),
            2048,
            "grammar scanner depth budget exceeded (limit: 2048 frames)",
        ),
    ] {
        let compiled = compile_schema(schema, &[&vec![b'['; length]]);
        let mut state = runtime_grammar::ConstraintState::new(compiled.clone(), None, vec![]);
        state.accept(u32::from(b'[')).unwrap();
        let fork = state.fork();
        let expected = GrammarError::TooLarge(expected.to_owned());
        let mut mask = vec![u32::MAX; compiled.vocabulary().mask_words()];
        assert_eq!(fork.fill_mask(&mut mask), Err(expected.clone()));
        assert!(mask.iter().all(|word| *word == 0));
        state.rollback(1).unwrap();
        assert_eq!(state.matcher().failure(), Some(&expected));
        assert_eq!(state.fill_mask(&mut mask), Err(expected.clone()));
        assert_eq!(state.accept(u32::from(b'[')), Err(expected));
        // A fresh request sharing this grammar has no inherited failure.
        assert_eq!(
            GrammarMatcher::new(compiled).accepts_token(u32::from(b'[')),
            Ok(true)
        );
    }
}
