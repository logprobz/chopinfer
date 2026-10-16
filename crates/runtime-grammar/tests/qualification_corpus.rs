//! Qualification schemas must compile without a checkpoint or a large vocabulary.
use runtime_grammar::{
    CompiledGrammar, GrammarBuilder, JsonOptions, SchemaCompiler, TokenVocabulary, VocabularyToken,
};
use serde_json::Value;
use std::sync::Arc;
#[test]
fn qualification_schemas_compile() {
    let mut cases: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/qualification.json")).unwrap();
    cases.extend(
        serde_json::from_str::<Vec<Value>>(include_str!("fixtures/golden_schemas.json")).unwrap(),
    );
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
    for case in cases {
        let options = JsonOptions {
            any_whitespace: case["compact"] != true,
            ..JsonOptions::default()
        };
        let mut builder = GrammarBuilder::new();
        let mut compiler = SchemaCompiler::new(&mut builder, &case["schema"], options);
        let expr = compiler.schema(&case["schema"]).unwrap();
        let usage = compiler.compilation_usage();
        assert!(
            usage.headroom() >= 10.0,
            "{} construction: {usage:?}",
            case["name"]
        );
        let root = builder.rule(expr);
        let grammar = builder.finish(root).unwrap();
        let compiled = CompiledGrammar::new(&grammar, vocabulary.clone())
            .unwrap_or_else(|error| panic!("{}: {error}", case["name"]));
        let usage = compiled.compilation_usage();
        eprintln!(
            "{}: automaton {usage:?}, headroom {}",
            case["name"],
            usage.headroom()
        );
        assert!(
            usage.headroom() >= 10.0,
            "{} compilation: {usage:?}",
            case["name"]
        );
    }
}
