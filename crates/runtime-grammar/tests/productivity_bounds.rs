//! Compilation must finish or refuse, without changing the accepted language.
#![cfg(target_os = "linux")]
mod common;
use common::bounded;
use runtime_grammar::{CompiledGrammar, GrammarError, choice_grammar};

#[test]
fn long_choice_productivity_analysis_is_bounded() {
    bounded("long_choice_productivity_analysis_is_bounded", || {
        let text = "a".repeat(100_000);
        let grammar = choice_grammar(std::slice::from_ref(&text)).unwrap();
        match CompiledGrammar::new(&grammar, common::adversarial_vocabulary()) {
            Ok(compiled) => {
                let mut matcher =
                    runtime_grammar::GrammarMatcher::new(std::sync::Arc::new(compiled));
                matcher.accept_bytes(text.as_bytes()).unwrap();
                assert!(matcher.is_complete());
                assert!(matcher.accept_bytes(b"a").is_err());
            }
            Err(GrammarError::TooLarge(_)) => {}
            Err(error) => panic!("unexpected compilation failure: {error}"),
        }
    });
}
