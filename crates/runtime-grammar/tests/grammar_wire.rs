//! The grammar wire: round trips, identical masks after decoding, and the
//! documents decoding refuses.
mod common;

use std::sync::Arc;

use common::{Rng, STOP, adversarial_vocabulary, allowed, compile};
use runtime_grammar::{
    CharClass, CompiledGrammar, ConstraintState, Expr, GRAMMAR_WIRE_LAYOUT_V1,
    GRAMMAR_WIRE_NESTING_MAX_V1, Grammar, GrammarBuilder, GrammarError, JsonOptions, MAX_SCALAR,
    RuleId, SchemaCompiler, TokenVocabulary, VocabularyToken, choice_grammar, json_object_grammar,
    json_schema_grammar, regex_grammar,
};
use serde_json::{Value, json};

/// Every schema fixture, built as the qualification corpus builds it.
fn fixture_grammars() -> Vec<(String, Grammar)> {
    let mut cases: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/qualification.json")).unwrap();
    cases.extend(
        serde_json::from_str::<Vec<Value>>(include_str!("fixtures/golden_schemas.json")).unwrap(),
    );
    let mut grammars = Vec::new();
    for case in cases {
        for any_whitespace in [true, false] {
            let options = JsonOptions {
                any_whitespace,
                ..JsonOptions::default()
            };
            let mut builder = GrammarBuilder::new();
            let mut compiler = SchemaCompiler::new(&mut builder, &case["schema"], options);
            let expr = compiler.schema(&case["schema"]).unwrap();
            let root = builder.rule(expr);
            let name = format!("{} any_whitespace={any_whitespace}", case["name"]);
            grammars.push((name, builder.finish(root).unwrap()));
        }
    }
    grammars
}

/// A tool-call shape over every expression kind, with special tokens.
fn tool_grammar() -> Grammar {
    let parameters = json!({"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer", "minimum": 1}}, "required": ["city"]});
    let mut builder = GrammarBuilder::new();
    let mut compiler = SchemaCompiler::new(
        &mut builder,
        &parameters,
        JsonOptions {
            any_whitespace: false,
            ..JsonOptions::default()
        },
    );
    let days = compiler.schema(&parameters["properties"]["days"]).unwrap();
    let parameter = |name: &str, value: Expr| {
        Expr::seq([
            Expr::literal("<p="),
            Expr::literal(name),
            Expr::literal(">"),
            value,
            Expr::literal("</p>"),
        ])
    };
    let nested = builder.declare();
    builder.define(
        nested,
        Expr::alt([
            Expr::literal("x"),
            Expr::Nothing,
            Expr::seq([Expr::literal("("), Expr::Rule(nested), Expr::literal(")")]),
        ]),
    );
    let body = Expr::OrderedSubset {
        items: vec![
            (
                parameter("city", Expr::TextExcluding(vec![b"</p>".to_vec()])),
                true,
            ),
            (parameter("days", days), false),
            (parameter("note", Expr::Rule(nested)), false),
        ],
        tail: Some(Box::new(parameter(
            "extra",
            Expr::Class(CharClass::of("ab")).repeat(1, Some(3)),
        ))),
        separator: Box::new(Expr::Empty),
    };
    let call = Expr::seq([
        Expr::literal("<call>"),
        body,
        Expr::TextUntil {
            end: b"</call>".to_vec(),
            avoid: vec![b"<call>".to_vec(), b"<p=".to_vec()],
        },
    ]);
    let root = builder.rule(call.plus());
    builder
        .finish(root)
        .unwrap()
        .with_special_tokens([b"<call>".to_vec(), b"</s>".to_vec()])
}

fn constructed_grammars() -> Vec<(String, Grammar)> {
    let mut grammars = vec![
        ("tool".to_owned(), tool_grammar()),
        (
            "object".to_owned(),
            json_object_grammar(JsonOptions::default()),
        ),
        (
            "compact object".to_owned(),
            json_object_grammar(JsonOptions {
                any_whitespace: false,
                max_whitespace_cnt: None,
            }),
        ),
        (
            "unbounded whitespace".to_owned(),
            json_object_grammar(JsonOptions {
                any_whitespace: true,
                max_whitespace_cnt: None,
            }),
        ),
        (
            "choice".to_owned(),
            choice_grammar(&["red".to_owned(), "green".to_owned(), "é😀".to_owned()]).unwrap(),
        ),
        (
            "every scalar".to_owned(),
            Grammar::from_expr(Expr::Class(CharClass::any()).star()),
        ),
        (
            "empty class".to_owned(),
            Grammar::from_expr(Expr::seq([
                Expr::literal("a"),
                Expr::Class(CharClass::default()).optional(),
            ])),
        ),
    ];
    for pattern in [r"^\d{3}-\d{4}$", r"(?:ab|cd)+[^x\n]?|\w\s\.", r"a{2,}b{,3}"] {
        grammars.push((pattern.to_owned(), regex_grammar(pattern).unwrap()));
    }
    for schema in [
        json!({"type": "array", "items": {"$ref": "#"}, "maxItems": 3}),
        json!({"anyOf": [{"type": "string", "format": "date-time"}, {"type": "number", "minimum": 0}]}),
        json!({"type": "object", "additionalProperties": {"type": "integer"}, "properties": {"a": {"enum": ["x", 1, null]}}}),
    ] {
        grammars.push((
            schema.to_string(),
            json_schema_grammar(&schema, JsonOptions::default()).unwrap(),
        ));
    }
    grammars
}

/// Nesting depth of `expr`, collecting the rules it refers to.
fn depth(expr: &Expr, rules: &mut Vec<RuleId>) -> usize {
    let children: Vec<&Expr> = match expr {
        Expr::Seq(items) | Expr::Alt(items) => items.iter().collect(),
        Expr::Repeat { expr, .. } => vec![expr],
        Expr::OrderedSubset {
            items,
            tail,
            separator,
        } => items
            .iter()
            .map(|(item, _)| item)
            .chain(tail.as_deref())
            .chain([&**separator])
            .collect(),
        Expr::Rule(id) => {
            rules.push(*id);
            Vec::new()
        }
        _ => Vec::new(),
    };
    1 + children
        .into_iter()
        .map(|child| depth(child, rules))
        .max()
        .unwrap_or(0)
}

/// The deepest nesting of any rule reachable from the root.
fn grammar_depth(grammar: &Grammar) -> usize {
    let mut pending = vec![grammar.root()];
    let mut seen = std::collections::HashSet::new();
    let mut deepest = 0;
    while let Some(id) = pending.pop() {
        if seen.insert(id) {
            deepest = deepest.max(depth(grammar.rule(id), &mut pending));
        }
    }
    deepest
}

fn header(root: u32, rules: u32) -> Vec<u8> {
    let mut document = b"CHOPGRAM".to_vec();
    document.extend_from_slice(&GRAMMAR_WIRE_LAYOUT_V1.to_le_bytes());
    document.extend_from_slice(&root.to_le_bytes());
    document.extend_from_slice(&rules.to_le_bytes());
    document
}

fn literal(text: &[u8]) -> Vec<u8> {
    let mut expr = vec![2];
    expr.extend_from_slice(&(text.len() as u64).to_le_bytes());
    expr.extend_from_slice(text);
    expr
}

fn class(ranges: &[(u32, u32)]) -> Vec<u8> {
    let mut expr = vec![3];
    expr.extend_from_slice(&(ranges.len() as u32).to_le_bytes());
    for (lo, hi) in ranges {
        expr.extend_from_slice(&lo.to_le_bytes());
        expr.extend_from_slice(&hi.to_le_bytes());
    }
    expr
}

fn list(kind: u8, items: &[Vec<u8>]) -> Vec<u8> {
    let mut expr = vec![kind];
    expr.extend_from_slice(&(items.len() as u32).to_le_bytes());
    for item in items {
        expr.extend_from_slice(item);
    }
    expr
}

fn repeat(min: u32, bounded: u8, max: u32, item: &[u8]) -> Vec<u8> {
    let mut expr = vec![6];
    expr.extend_from_slice(&min.to_le_bytes());
    expr.push(bounded);
    expr.extend_from_slice(&max.to_le_bytes());
    expr.extend_from_slice(item);
    expr
}

fn rule(id: u32) -> Vec<u8> {
    let mut expr = vec![7];
    expr.extend_from_slice(&id.to_le_bytes());
    expr
}

/// A document of `rules` with the given root and special token texts.
fn document(root: u32, rules: &[Vec<u8>], specials: &[&[u8]]) -> Vec<u8> {
    let mut document = header(root, rules.len() as u32);
    for rule in rules {
        document.extend_from_slice(rule);
    }
    document.extend_from_slice(&(specials.len() as u32).to_le_bytes());
    for text in specials {
        document.extend_from_slice(&(text.len() as u64).to_le_bytes());
        document.extend_from_slice(text);
    }
    document
}

fn invalid(bytes: &[u8]) -> String {
    match Grammar::decode_wire(bytes) {
        Err(GrammarError::Invalid(message)) => message,
        other => panic!("expected an invalid document, got {other:?}"),
    }
}

fn too_large(bytes: &[u8]) -> String {
    match Grammar::decode_wire(bytes) {
        Err(GrammarError::TooLarge(message)) => message,
        other => panic!("expected a document past a budget, got {other:?}"),
    }
}

#[test]
fn grammars_round_trip() {
    let grammars = fixture_grammars()
        .into_iter()
        .chain(constructed_grammars())
        .collect::<Vec<_>>();
    assert!(grammars.len() > 20);
    for (name, grammar) in grammars {
        let encoded = grammar.encode_wire();
        let decoded =
            Grammar::decode_wire(&encoded).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(decoded, grammar, "{name}");
        assert_eq!(decoded.special_tokens(), grammar.special_tokens(), "{name}");
        assert_eq!(decoded.encode_wire(), encoded, "{name}");
    }
}

/// Decoded grammars compile to the same masks, step by step along random
/// walks, as the grammars they encode.
#[test]
fn decoded_grammars_compile_to_identical_masks() {
    let vocabulary = adversarial_vocabulary();
    let mut grammars = constructed_grammars();
    grammars.extend(fixture_grammars().into_iter().step_by(3));
    for (name, grammar) in grammars {
        let decoded = Grammar::decode_wire(&grammar.encode_wire()).unwrap();
        let local = compile(&grammar, &vocabulary);
        let remote = compile(&decoded, &vocabulary);
        for seed in 0..4 {
            let mut rng = Rng::new(seed);
            let mut states = [&local, &remote]
                .map(|compiled| ConstraintState::new(Arc::clone(compiled), None, vec![STOP]));
            let mut masks = [(); 2].map(|()| vec![0_u32; vocabulary.mask_words()]);
            for _ in 0..48 {
                for (state, mask) in states.iter().zip(&mut masks) {
                    assert!(state.fill_mask(mask).unwrap());
                }
                assert_eq!(masks[0], masks[1], "{name}");
                let tokens = allowed(&masks[0], vocabulary.size());
                let token = tokens[rng.below(tokens.len())];
                for state in &mut states {
                    state.accept(token).unwrap();
                }
                if token == STOP {
                    break;
                }
            }
        }
    }
}

/// Grammars the schema compiler builds around a pattern of nested starred
/// groups, as deep as its nesting bound admits, decode, and the deepest of
/// each shape compiles on a 2 MiB stack.
#[test]
fn the_depth_bound_admits_every_local_grammar_that_compiles() {
    let starred =
        json!({"type": "string", "pattern": format!("{}x{}", "(a".repeat(64), ")*".repeat(64))});
    let wraps: [fn(Value) -> Value; 3] = [
        |inner| json!({"type": "object", "properties": {"a": inner}, "required": ["a"]}),
        |inner| json!({"type": "object", "additionalProperties": inner}),
        |inner| json!({"type": "object", "properties": {"a": {"type": "array", "items": inner, "maxItems": 1}}, "required": ["a"]}),
    ];
    let mut deepest = 0;
    for wrap in wraps {
        for leaf in [json!({"type": "string"}), starred.clone()] {
            let mut schema = leaf;
            let mut last = None;
            loop {
                let grammar = match json_schema_grammar(&schema, JsonOptions::default()) {
                    Ok(grammar) => grammar,
                    Err(error) => {
                        assert!(error.to_string().contains("nesting depth"), "{error}");
                        break;
                    }
                };
                assert_eq!(
                    Grammar::decode_wire(&grammar.encode_wire()).unwrap(),
                    grammar
                );
                deepest = deepest.max(grammar_depth(&grammar));
                last = Some(grammar);
                schema = wrap(schema);
            }
            let last = last.unwrap();
            let depth = grammar_depth(&last);
            assert!(compiles_on_a_2_mib_stack(&last), "depth {depth}");
        }
    }
    assert!(deepest > 128, "{deepest}");
}

/// A chain of `count` rules, each only a reference to the next, which
/// compilation inlines: compile depth `count` with no deep expression.
fn rule_chain(count: u32) -> Vec<u8> {
    let rules = (0..count)
        .map(|id| {
            if id + 1 < count {
                rule(id + 1)
            } else {
                literal(b"x")
            }
        })
        .collect::<Vec<_>>();
    document(0, &rules, &[])
}

/// `count` ordered subsets, each holding the next as its one required item,
/// around an empty leaf: compile depth `2 * count + 1`.
fn subset_chain(count: usize) -> Vec<u8> {
    let mut expr = vec![0];
    for _ in 0..count {
        let mut outer = list(10, &[]);
        outer[1] = 1;
        outer.push(1);
        outer.extend_from_slice(&expr);
        outer.extend([0, 0]);
        expr = outer;
    }
    document(0, &[expr], &[])
}

/// Compile depth counts inlined rules and ordered subsets twice, and a
/// document one level past the bound, 513, is refused however it gets
/// there.
#[test]
fn documents_past_the_compile_depth_bound_are_refused() {
    assert_eq!(GRAMMAR_WIRE_NESTING_MAX_V1, 512);
    assert!(Grammar::decode_wire(&rule_chain(512)).is_ok());
    let message = too_large(&rule_chain(513));
    assert!(message.contains("compiles 513 levels deep"), "{message}");
    assert!(Grammar::decode_wire(&subset_chain(255)).is_ok());
    let message = too_large(&subset_chain(256));
    assert!(message.contains("compiles 513 levels deep"), "{message}");
    // A rule that refers to itself is compiled on its own and called, so
    // recursion adds no depth.
    let recursive = document(
        0,
        &[list(
            5,
            &[literal(b"x"), list(4, &[literal(b"("), rule(0)])],
        )],
        &[],
    );
    assert!(Grammar::decode_wire(&recursive).is_ok());
}

/// The deepest documents the bound admits, of every shape that recurses
/// most per level, compile on a 2 MiB thread stack.
#[test]
fn documents_at_the_bound_compile_on_a_2_mib_stack() {
    let nested_alternatives = {
        let mut expr = literal(b"x");
        for _ in 1..GRAMMAR_WIRE_NESTING_MAX_V1 {
            expr = list(5, &[expr, literal(b"y")]);
        }
        document(0, &[expr], &[])
    };
    let documents = [
        nested_alternatives,
        rule_chain(GRAMMAR_WIRE_NESTING_MAX_V1 as u32),
        subset_chain((GRAMMAR_WIRE_NESTING_MAX_V1 - 1) / 2),
    ];
    let grammars = documents
        .iter()
        .map(|document| Grammar::decode_wire(document).unwrap())
        .collect::<Vec<_>>();
    let vocabulary = adversarial_vocabulary();
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || {
            for grammar in &grammars {
                // Compiling may refuse a budget; it must not exhaust the stack.
                let _ = CompiledGrammar::new(grammar, Arc::clone(&vocabulary));
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn decoding_refuses_malformed_documents() {
    let valid = document(0, &[literal(b"a")], &[]);
    assert_eq!(
        Grammar::decode_wire(&valid).unwrap(),
        Grammar::from_expr(Expr::literal("a"))
    );

    let mut magic = valid.clone();
    magic[0] = b'X';
    assert!(invalid(&magic).contains("not a grammar wire document"));
    for layout in [0, GRAMMAR_WIRE_LAYOUT_V1 + 1] {
        let mut other = valid.clone();
        other[8..12].copy_from_slice(&layout.to_le_bytes());
        match Grammar::decode_wire(&other) {
            Err(GrammarError::Unsupported(message)) => {
                assert!(message.contains(&format!("layout {layout}")), "{message}");
            }
            other => panic!("{other:?}"),
        }
    }

    let encoded = tool_grammar().encode_wire();
    for cut in 0..encoded.len() {
        assert!(Grammar::decode_wire(&encoded[..cut]).is_err(), "{cut}");
    }
    let mut trailing = encoded.clone();
    trailing.push(0);
    assert!(invalid(&trailing).contains("trailing bytes"));

    for kind in [11, 255] {
        assert!(invalid(&document(0, &[vec![kind]], &[])).contains("unknown expression kind"));
    }
    // Flags are 0 or 1.
    assert!(invalid(&document(0, &[repeat(0, 2, 1, &literal(b"a"))], &[])).contains("flag"));
    let mut subset = list(10, &[]);
    subset.push(2);
    assert!(invalid(&document(0, &[subset], &[])).contains("flag"));
    let mut required = list(10, &[]);
    required[1] = 1;
    required.extend([2, 0, 1, 0]);
    assert!(invalid(&document(0, &[required], &[])).contains("flag"));

    // References stay inside the rules.
    assert!(invalid(&document(1, &[literal(b"a")], &[])).contains("root"));
    assert!(invalid(&document(0, &[], &[])).contains("root"));
    assert!(invalid(&document(0, &[rule(1)], &[])).contains("rule reference"));
    assert!(
        invalid(&document(0, &[list(4, &[rule(0), rule(2)]), rule(0)], &[]))
            .contains("rule reference")
    );
    // A rule may refer to itself or to a later rule.
    assert!(
        Grammar::decode_wire(&document(
            0,
            &[list(4, &[rule(0), rule(1)]), literal(b"x")],
            &[]
        ))
        .is_ok()
    );

    // Classes are sorted, disjoint, non-adjacent scalar ranges.
    for ranges in [
        &[(5, 6), (1, 2)][..],
        &[(1, 5), (3, 8)],
        &[(1, 2), (3, 4)],
        &[(5, 1)],
        &[(0, MAX_SCALAR + 1)],
        &[(MAX_SCALAR + 1, MAX_SCALAR + 2)],
    ] {
        assert!(
            invalid(&document(0, &[class(ranges)], &[])).contains("class"),
            "{ranges:?}"
        );
    }
    let surrogates = [(0x41, 0x5a), (0xd800, 0xdfff), (MAX_SCALAR, MAX_SCALAR)];
    assert_eq!(
        Grammar::decode_wire(&document(0, &[class(&surrogates)], &[])).unwrap(),
        Grammar::from_expr(Expr::Class(CharClass::from_ranges(surrogates)))
    );

    // Repetitions.
    assert!(
        invalid(&document(0, &[repeat(3, 1, 2, &literal(b"a"))], &[])).contains("max below min")
    );
    assert!(invalid(&document(0, &[repeat(0, 0, 2, &literal(b"a"))], &[])).contains("unbounded"));

    // Special token texts do not repeat.
    assert!(invalid(&document(0, &[literal(b"a")], &[b"<s>", b"<s>"])).contains("special token"));

    // Counts past the remaining bytes are truncation, not allocation.
    let mut huge = header(0, u32::MAX);
    huge.push(0);
    assert!(invalid(&huge).contains("ends early"));
    let mut huge_class = vec![3];
    huge_class.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(invalid(&document(0, &[huge_class], &[])).contains("ends early"));
    let mut huge_literal = vec![2];
    huge_literal.extend_from_slice(&u64::MAX.to_le_bytes());
    assert!(invalid(&document(0, &[huge_literal], &[])).contains("ends early"));
}

#[test]
fn decoding_is_bounded_by_nesting_and_not_by_the_construction_budget() {
    let nested = |levels: usize| {
        let mut expr = vec![0];
        for _ in 1..levels {
            let mut outer = list(4, &[]);
            outer[1] = 1;
            outer.extend_from_slice(&expr);
            expr = outer;
        }
        document(0, &[expr], &[])
    };
    let deepest = Grammar::decode_wire(&nested(GRAMMAR_WIRE_NESTING_MAX_V1)).unwrap();
    assert_eq!(grammar_depth(&deepest), GRAMMAR_WIRE_NESTING_MAX_V1);
    assert!(too_large(&nested(GRAMMAR_WIRE_NESTING_MAX_V1 + 1)).contains("nests"));

    // Past the construction budget's expression bytes (one long literal,
    // many small nodes, many texts), a document still decodes, and compiles
    // or not as the same grammar built locally does: compilation applies
    // its own budgets. The two that compile were refused when decoding
    // charged the construction budget.
    let vocabulary = Arc::new(
        TokenVocabulary::new(
            256,
            (0..=255_u8)
                .map(|byte| VocabularyToken {
                    bytes: vec![byte],
                    special: false,
                })
                .collect(),
        )
        .unwrap(),
    );
    let texts: Vec<Vec<u8>> = (0..1 << 20)
        .map(|index: u32| format!("marker{index}").into_bytes())
        .collect();
    let texts: Vec<&[u8]> = texts.iter().map(Vec::as_slice).collect();
    for (document, refusal) in [
        (
            document(0, &[literal(&vec![b'a'; 33 << 20])], &[]),
            Some("the constraint is too large"),
        ),
        (document(0, &[list(5, &vec![vec![0]; 600_000])], &[]), None),
        (document(0, &[literal(b"a")], &texts), None),
    ] {
        let decoded = Grammar::decode_wire(&document).unwrap();
        assert_eq!(decoded.encode_wire(), document);
        let compiled = CompiledGrammar::new(&decoded, Arc::clone(&vocabulary));
        assert_eq!(
            compiled.err().map(|error| error.to_string()).as_deref(),
            refusal
        );
    }
}

/// A root that refers to a middle rule 32,000 times, which refers to an
/// empty leaf 32,000 times, decodes: the depth analysis does not rescan a
/// rule per reference (the crate's unit test bounds the work it charges).
#[test]
fn widely_shared_rules_decode() {
    let width = 32_000;
    let document = document(
        0,
        &[
            list(4, &vec![rule(1); width]),
            list(4, &vec![rule(2); width]),
            vec![0],
        ],
        &[],
    );
    let grammar = Grammar::decode_wire(&document).unwrap();
    assert_eq!(grammar.encode_wire(), document);
}

/// Compiles `grammar` on a 2 MiB thread stack, as a request thread would.
fn compiles_on_a_2_mib_stack(grammar: &Grammar) -> bool {
    let grammar = grammar.clone();
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || CompiledGrammar::new(&grammar, adversarial_vocabulary()).is_ok())
        .unwrap()
        .join()
        .unwrap()
}

/// Local grammars past the old 288 bound that compile and run locally on
/// a 2 MiB stack decode, and their decoded copies compile alike.
#[test]
fn local_grammars_that_compile_decode_and_compile_alike() {
    let mut schema =
        json!({"type": "string", "pattern": format!("{}x{}", "(a".repeat(64), ")*".repeat(64))});
    for _ in 0..80 {
        schema = json!({"type": "array", "items": schema, "maxItems": 1});
    }
    let arrays = json_schema_grammar(&schema, JsonOptions::default()).unwrap();
    let mut builder = GrammarBuilder::new();
    let rules = (0..289).map(|_| builder.declare()).collect::<Vec<_>>();
    for pair in rules.windows(2) {
        builder.define(pair[0], Expr::Rule(pair[1]));
    }
    builder.define(rules[288], Expr::literal("x"));
    let chain = builder.finish(rules[0]).unwrap();
    for (name, local) in [("arrays", arrays), ("rule chain", chain)] {
        assert!(compiles_on_a_2_mib_stack(&local), "{name} compiles locally");
        let decoded = Grammar::decode_wire(&local.encode_wire())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(decoded, local, "{name}");
        assert!(
            compiles_on_a_2_mib_stack(&decoded),
            "{name} compiles decoded"
        );
        let vocabulary = adversarial_vocabulary();
        let masks = [&local, &decoded].map(|grammar| {
            let state = ConstraintState::new(compile(grammar, &vocabulary), None, vec![STOP]);
            let mut mask = vec![0_u32; vocabulary.mask_words()];
            assert!(state.fill_mask(&mut mask).unwrap());
            mask
        });
        assert_eq!(masks[0], masks[1], "{name}");
    }
}
