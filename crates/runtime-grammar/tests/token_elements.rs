//! Token elements: a grammar position that only one vocabulary token, by
//! its identifier, fills, whatever other tokens spell its text, and that
//! token nowhere else.

use std::sync::Arc;

use runtime_grammar::{
    CharClass, CompiledGrammar, ConstraintState, Expr, GRAMMAR_TOKENS_MAX, Grammar, GrammarError,
    GrammarMatcher, PrefixPart, TokenVocabulary, VocabularyToken,
};

const STOP: u32 = 0;
/// `<M>`, which the tokenizer leaves ordinary.
const ORDINARY: u32 = 1;
/// `<N>`, which the tokenizer marks special.
const SPECIAL: u32 = 2;
/// The token of byte `b` is `BYTE + b`.
const BYTE: u32 = 3;
/// `<M`, an ordinary piece of `<M>`.
const PIECE: u32 = BYTE + 256;
const SIZE: usize = PIECE as usize + 1;

fn vocabulary() -> Arc<TokenVocabulary> {
    let mut tokens = vec![
        VocabularyToken {
            bytes: b"</s>".to_vec(),
            special: true,
        },
        VocabularyToken {
            bytes: b"<M>".to_vec(),
            special: false,
        },
        VocabularyToken {
            bytes: b"<N>".to_vec(),
            special: true,
        },
    ];
    tokens.extend((0..=255_u8).map(|byte| VocabularyToken {
        bytes: vec![byte],
        special: false,
    }));
    tokens.push(VocabularyToken {
        bytes: b"<M".to_vec(),
        special: false,
    });
    Arc::new(TokenVocabulary::new(SIZE, tokens).unwrap())
}

fn byte(byte: u8) -> u32 {
    BYTE + u32::from(byte)
}

fn compile(grammar: &Grammar) -> Arc<CompiledGrammar> {
    Arc::new(CompiledGrammar::new(grammar, vocabulary()).unwrap())
}

fn mask(matcher: &GrammarMatcher) -> Vec<u32> {
    let mut mask = vec![0; matcher.grammar().vocabulary().mask_words()];
    matcher.fill_token_bitmask(&mut mask).unwrap();
    (0..SIZE as u32)
        .filter(|&token| mask[token as usize / 32] & (1 << (token % 32)) != 0)
        .collect()
}

/// Text, then `[`, the ordinary-spelled token `<M>` by its ID, `]` and the
/// special token `<N>` by its ID.
fn grammar() -> Grammar {
    Grammar::from_expr(Expr::seq([
        Expr::Class(CharClass::any()).star(),
        Expr::literal("["),
        Expr::Token(ORDINARY),
        Expr::literal("]"),
        Expr::Token(SPECIAL),
    ]))
}

/// The element admits its token and nothing that spells it; the token,
/// ordinary or special, is admitted nowhere else, though its bytes fit the
/// text before it.
#[test]
fn a_token_element_admits_only_its_token() {
    let compiled = compile(&grammar());
    let mut matcher = GrammarMatcher::new(Arc::clone(&compiled));
    let text = mask(&matcher);
    assert!(text.contains(&byte(b'<')) && text.contains(&PIECE));
    assert!(!text.contains(&ORDINARY) && !text.contains(&SPECIAL));
    assert!(!matcher.accepts_token(ORDINARY).unwrap());
    for token in [byte(b'x'), byte(b'[')] {
        matcher.accept_token(token).unwrap();
    }
    // After `[` the text may go on, or the token follow.
    let after = mask(&matcher);
    assert!(after.contains(&ORDINARY) && after.contains(&byte(b'<')));
    // `<M>` spelled by ordinary tokens is text, after which `]` is text too
    // and `<N>` cannot follow.
    let mut spelled = matcher.clone();
    for token in [byte(b'<'), byte(b'M'), byte(b'>'), byte(b']')] {
        spelled.accept_token(token).unwrap();
    }
    assert!(!spelled.accepts_token(SPECIAL).unwrap());
    matcher.accept_token(ORDINARY).unwrap();
    assert_eq!(mask(&matcher), [byte(b']')]);
    matcher.accept_token(byte(b']')).unwrap();
    assert_eq!(mask(&matcher), [SPECIAL]);
    matcher.accept_token(SPECIAL).unwrap();
    assert!(matcher.is_complete());
    // Bytes alone never fill the element.
    let mut bytes = GrammarMatcher::new(compiled);
    assert!(bytes.accept_bytes(b"x[<M>]<N>").is_ok());
    assert!(!bytes.is_complete());
}

/// A special token the grammar also admits by its text still matches only
/// at its element, not in text its bytes fit.
#[test]
fn a_named_token_admitted_by_text_matches_only_at_its_element() {
    let grammar = Grammar::from_expr(Expr::seq([
        Expr::Class(CharClass::any()).star(),
        Expr::literal("|"),
        Expr::Token(SPECIAL),
    ]))
    .with_special_tokens([b"<N>".to_vec()]);
    let mut matcher = GrammarMatcher::new(compile(&grammar));
    assert!(!mask(&matcher).contains(&SPECIAL));
    matcher.accept_token(byte(b'|')).unwrap();
    assert!(mask(&matcher).contains(&SPECIAL));
    matcher.accept_token(SPECIAL).unwrap();
    assert!(matcher.is_complete());
}

/// An ordinary token holding byte 0xFF, which no UTF-8 text holds, never
/// matches, and fed bytes may not hold it, so neither can spell an
/// element's escape.
#[test]
fn ordinary_tokens_cannot_spell_an_element() {
    let mut tokens = vec![
        VocabularyToken {
            bytes: b"<M>".to_vec(),
            special: false,
        },
        VocabularyToken {
            bytes: vec![0xFF],
            special: false,
        },
    ];
    tokens.extend((0..=255_u8).map(|byte| VocabularyToken {
        bytes: vec![byte, b'!'],
        special: false,
    }));
    let vocabulary = Arc::new(TokenVocabulary::new(258, tokens).unwrap());
    let compiled = CompiledGrammar::new(&Grammar::from_expr(Expr::Token(0)), vocabulary).unwrap();
    let matcher = GrammarMatcher::new(Arc::new(compiled));
    for token in 1..258 {
        assert!(!matcher.accepts_token(token).unwrap(), "{token}");
    }
    assert!(matcher.accepts_token(0).unwrap());
    assert!(matcher.clone().accept_bytes(&[0xFF, 0]).is_err());
}

/// Compilation refuses a token outside the vocabulary, more distinct tokens
/// than a grammar may name, and a literal holding byte 0xFF.
#[test]
fn compilation_refuses_what_an_element_cannot_be() {
    let refused = |grammar: Grammar| CompiledGrammar::new(&grammar, vocabulary()).unwrap_err();
    assert!(matches!(
        refused(Grammar::from_expr(Expr::Token(SIZE as u32))),
        GrammarError::Invalid(message) if message.contains("outside its vocabulary")
    ));
    let named = |count: u32| {
        Grammar::from_expr(Expr::alt((0..count).map(|id| {
            Expr::seq([
                Expr::literal(&id.to_string()),
                Expr::Token(id % SIZE as u32),
            ])
        })))
    };
    assert!(CompiledGrammar::new(&named(GRAMMAR_TOKENS_MAX as u32), vocabulary()).is_ok());
    assert!(matches!(
        refused(named(GRAMMAR_TOKENS_MAX as u32 + 1)),
        GrammarError::TooLarge(message) if message.contains("more than 256 tokens")
    ));
    assert!(matches!(
        refused(Grammar::from_expr(Expr::Literal(vec![b'a', 0xFF]))),
        GrammarError::Invalid(message) if message.contains("0xFF")
    ));
}

/// A prefix's token part is read as the token where the grammar names it
/// and as its text where it does not; its text alone does not fill an
/// element.
#[test]
fn a_prefix_reads_a_named_token_as_the_token() {
    let compiled = compile(&Grammar::from_expr(Expr::seq([
        Expr::literal("["),
        Expr::Token(ORDINARY),
        Expr::literal("]<N>"),
    ])));
    let state = ConstraintState::new(Arc::clone(&compiled), None, vec![STOP])
        .with_prefix_parts(&[PrefixPart::Text(b"["), PrefixPart::Token(ORDINARY, b"<M>")])
        .unwrap();
    assert!(state.allows(byte(b']')).unwrap());
    assert!(
        ConstraintState::new(Arc::clone(&compiled), None, vec![STOP])
            .with_prefix(b"[<M>")
            .is_err()
    );
    // `<N>` is not named here, so its part is read as its text.
    let state = ConstraintState::new(compiled, None, vec![STOP])
        .with_prefix_parts(&[
            PrefixPart::Text(b"["),
            PrefixPart::Token(ORDINARY, b"<M>"),
            PrefixPart::Text(b"]"),
            PrefixPart::Token(SPECIAL, b"<N>"),
        ])
        .unwrap();
    assert!(state.allows(STOP).unwrap());
    assert!(state.matcher().is_complete());
}
