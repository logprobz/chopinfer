//! Constrained decoding against this model's real vocabulary: exact masks
//! and the cost of producing them.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use model_mimo26::tokenizer::load_tokenizer;
use runtime_grammar::{
    CompiledGrammar, ConstraintState, GrammarMatcher, JsonOptions, TokenVocabulary,
    VocabularyToken, json_object_grammar, json_schema_grammar,
};
use serde_json::json;

#[path = "support/structured_cases.rs"]
mod structured_cases;

const LOGITS_WIDTH: usize = 152_576;

fn tokenizer() -> Option<runtime_tokenizer::Tokenizer> {
    let directory = std::env::var("CHOPINFER_MIMO_TOKENIZER_DIR")
        .or_else(|_| std::env::var("CHOPINFER_MIMO_CHECKPOINT"))
        .ok()?;
    let directory = Path::new(&directory);
    directory
        .join("tokenizer.json")
        .exists()
        .then(|| load_tokenizer(directory).expect("load checkpoint tokenizer"))
}

fn vocabulary() -> Option<Arc<TokenVocabulary>> {
    let directory = std::env::var("CHOPINFER_MIMO_TOKENIZER_DIR")
        .or_else(|_| std::env::var("CHOPINFER_MIMO_CHECKPOINT"))
        .ok()?;
    let directory = Path::new(&directory);
    if !directory.join("tokenizer.json").exists() {
        return None;
    }
    let tokenizer = load_tokenizer(directory).expect("load checkpoint tokenizer");
    let mut tokens = Vec::new();
    while let Ok(bytes) = tokenizer.token_bytes(tokens.len() as u32) {
        let special = tokenizer
            .token_is_special(tokens.len() as u32)
            .expect("in range");
        tokens.push(VocabularyToken {
            bytes: bytes.to_vec(),
            special,
        });
    }
    Some(Arc::new(
        TokenVocabulary::new(LOGITS_WIDTH, tokens).expect("vocabulary"),
    ))
}

fn allowed(mask: &[u32]) -> Vec<u32> {
    (0..LOGITS_WIDTH as u32)
        .filter(|&token| mask[token as usize / 32] & (1 << (token % 32)) != 0)
        .collect()
}

#[test]
fn masks_are_exact_and_cheap_with_the_model_vocabulary() {
    let Some(vocabulary) = vocabulary() else {
        eprintln!("checkpoint tokenizer is absent, skipping constrained decoding budget");
        return;
    };
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "tags": {"type": "array", "items": {"enum": ["red", "green", "blue"]}},
            "extra": {"type": "object"}
        },
        "required": ["name", "age"]
    });
    let cases = [
        (
            "json_schema",
            json_schema_grammar(&schema, JsonOptions::default()).unwrap(),
            r#"{"name": "Ada \"L\" é", "age": 36, "tags": ["red"], "extra": {"k": [1, {"z": null}]}}"#,
        ),
        (
            "json_object",
            json_object_grammar(JsonOptions::default()),
            r#"{"a": {"b": [1, 2, {"c": "x"}]}, "d": true}"#,
        ),
    ];
    for (name, grammar, text) in cases {
        let started = Instant::now();
        let compiled = Arc::new(CompiledGrammar::new(&grammar, Arc::clone(&vocabulary)).unwrap());
        let compile_time = started.elapsed();
        let started = Instant::now();
        let precomputed = compiled.precompute(8).unwrap();
        let precompute_time = started.elapsed();
        let mut matcher = GrammarMatcher::new(Arc::clone(&compiled));
        let mut mask = vec![0_u32; vocabulary.mask_words()];
        let mut first_visits = std::time::Duration::ZERO;
        let mut warm = std::time::Duration::ZERO;
        let mut rows = 0_u32;
        for (index, character) in text.char_indices() {
            let started = Instant::now();
            matcher.fill_token_bitmask(&mut mask).unwrap();
            first_visits += started.elapsed();
            let started = Instant::now();
            matcher.fill_token_bitmask(&mut mask).unwrap();
            warm += started.elapsed();
            rows += 1;
            // Exactness on a sample of positions, against a token-by-token check.
            if index % 7 == 0 {
                let expected: Vec<u32> = (0..LOGITS_WIDTH as u32)
                    .filter(|&token| matcher.accepts_token(token).unwrap())
                    .collect();
                assert_eq!(allowed(&mask), expected, "{name} at {index}");
            }
            let mut buffer = [0; 4];
            matcher
                .accept_bytes(character.encode_utf8(&mut buffer).as_bytes())
                .unwrap_or_else(|_| panic!("{name} refused {character:?} at {index}"));
        }
        assert!(matcher.is_complete());
        eprintln!(
            "{name}: compile {compile_time:?}, precomputed {precomputed} positions in \
             {precompute_time:?}, first visits {:?}/row, warm {:?}/row over {rows} rows, \
             {} cached positions",
            first_visits / rows,
            warm / rows,
            compiled.cached_positions()
        );
    }
    // A request-shaped state: every row after the first costs a warm mask.
    let compiled = Arc::new(
        CompiledGrammar::new(
            &json_schema_grammar(&schema, JsonOptions::default()).unwrap(),
            vocabulary,
        )
        .unwrap(),
    );
    compiled.precompute(8).unwrap();
    let state = ConstraintState::new(compiled, None, vec![151_645]);
    let mut mask = vec![0_u32; LOGITS_WIDTH.div_ceil(32)];
    let started = Instant::now();
    for _ in 0..32 {
        state.fill_mask(&mut mask).unwrap();
    }
    eprintln!("32 warm rows: {:?}", started.elapsed());
}

/// The qualification cases replayed token by token: after the compile-time
/// precompute every row is a cache hit, admits the model's next token, and
/// equals a token-by-token check where sampled.
#[test]
fn qualification_rows_are_precomputed_and_exact() {
    let Some(tokenizer) = tokenizer() else {
        eprintln!("checkpoint tokenizer is absent, skipping constrained qualification rows");
        return;
    };
    let vocabulary = structured_cases::vocabulary(&tokenizer, LOGITS_WIDTH);
    let stop = tokenizer
        .encode(structured_cases::STOP_TEXT, &[structured_cases::STOP_TEXT])
        .unwrap()[0];
    for case in structured_cases::cases() {
        if case.name.starts_with("json_") {
            serde_json::from_str::<serde_json::Value>(case.text)
                .expect("qualification golden is JSON");
        }
        let mut tokens = tokenizer
            .encode(case.text, &structured_cases::SPECIALS)
            .unwrap();
        tokens.push(stop);
        let compiled =
            Arc::new(CompiledGrammar::new(&case.grammar, Arc::clone(&vocabulary)).unwrap());
        let positions = compiled.precompute(8).unwrap();
        assert!(positions > 0, "{}", case.name);
        let mut state = ConstraintState::new(Arc::clone(&compiled), None, vec![stop]);
        let mut mask = vec![0_u32; vocabulary.mask_words()];
        for (step, &token) in tokens.iter().enumerate() {
            assert!(
                state.fill_mask(&mut mask).unwrap(),
                "{} step {step}",
                case.name
            );
            assert_eq!(
                compiled.cached_positions(),
                positions,
                "{} step {step}",
                case.name
            );
            assert!(
                mask[token as usize / 32] & (1 << (token % 32)) != 0,
                "{} step {step} refuses {token}",
                case.name
            );
            if step % 16 == 0 {
                let expected: Vec<u32> = (0..LOGITS_WIDTH as u32)
                    .filter(|&candidate| state.allows(candidate).unwrap())
                    .collect();
                assert_eq!(allowed(&mask), expected, "{} step {step}", case.name);
            }
            state.accept(token).unwrap();
        }
        assert!(state.is_terminated(), "{}", case.name);
    }
}

/// Qualification compilation and its margin must not depend on checkpoint availability.
#[test]
fn qualification_grammars_have_compile_headroom_without_checkpoint() {
    let vocabulary = Arc::new(
        TokenVocabulary::new(
            1,
            vec![VocabularyToken {
                bytes: b"x".to_vec(),
                special: false,
            }],
        )
        .unwrap(),
    );
    for case in structured_cases::cases() {
        let compiled = CompiledGrammar::new(&case.grammar, vocabulary.clone())
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        let usage = compiled.compilation_usage();
        assert!(usage.headroom() >= 10.0, "{}: {usage:?}", case.name);
    }
}
