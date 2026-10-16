//! Model-neutral constrained decoding.
//!
//! A constraint (JSON, a JSON Schema, a regular expression, a fixed choice,
//! or a grammar a frontend built) compiles to byte automata. A vocabulary's
//! token bytes are arranged in a trie once per tokenizer. Per request, a
//! matcher follows committed tokens and produces the bitmask of tokens that
//! may come next. See docs/serving-request-surface.md.

#![forbid(unsafe_code)]

mod automaton;
mod budget;
mod expr;
mod heap;
mod json;
mod mask;
mod matcher;
mod range;
mod regex;
mod stack;
mod utf8;
mod vocab;
mod wire;

pub use budget::CompileUsage;
pub use expr::{CharClass, Expr, Grammar, GrammarBuilder, GrammarError, MAX_SCALAR, RuleId};
pub use json::{JsonOptions, SchemaCompiler, json_object_grammar, json_schema_grammar};
pub use matcher::{CompiledGrammar, ConstraintState, GrammarMatcher, MAX_ROLLBACK};
pub use vocab::{TokenVocabulary, VocabularyToken};
pub use wire::{GRAMMAR_WIRE_LAYOUT_V1, GRAMMAR_WIRE_NESTING_MAX_V1};

/// A grammar for whole outputs matching `pattern` in the supported subset.
pub fn regex_grammar(pattern: &str) -> Result<Grammar, GrammarError> {
    regex::regex_expr(pattern, &Expr::Class).map(Grammar::from_expr)
}

/// A grammar for exactly one of `choices`.
pub fn choice_grammar(choices: &[String]) -> Result<Grammar, GrammarError> {
    if choices.is_empty() {
        return Err(GrammarError::Invalid(
            "choice needs at least one string".to_owned(),
        ));
    }
    if choices.len() > budget::MAX_ALTERNATIVES {
        return Err(GrammarError::too_large("too many grammar choices"));
    }
    let budget = budget::Budget::default();
    for choice in choices {
        budget.charge(
            choice.len() + 1,
            choice.len().saturating_mul(4).saturating_add(128),
        )?;
    }
    Ok(Grammar::from_expr(Expr::alt(
        choices.iter().map(|choice| Expr::literal(choice)),
    )))
}
