//! Adversarial request probes run in child processes with hard resource limits.
#![cfg(target_os = "linux")]

mod common;
use common::bounded;

use runtime_grammar::{
    CompiledGrammar, GrammarError, JsonOptions, choice_grammar, json_schema_grammar, regex_grammar,
};
use serde_json::json;

#[test]
fn nested_enum_narrowing_is_bounded() {
    bounded("nested_enum_narrowing_is_bounded", || {
        let mut schema = json!({"type":"string"});
        for _ in 0..20 {
            schema = json!({"enum":["v0","v1"], "properties":{"a":schema}});
        }
        // G1 refuses the empty intersection instead of restoring the strings.
        assert_eq!(
            json_schema_grammar(&schema, JsonOptions::default()).unwrap_err(),
            GrammarError::Invalid("enum has no value satisfying its sibling assertions".to_owned()),
        );
    });
}

#[test]
fn large_choices_are_refused_before_expansion() {
    bounded("large_choices_are_refused_before_expansion", || {
        let choices: Vec<_> = (0..1_000_000).map(|i| format!("v{i}")).collect();
        assert!(matches!(
            choice_grammar(&choices),
            Err(GrammarError::TooLarge(_))
        ));
    });
}

#[test]
fn dfa_subsets_have_a_memory_budget() {
    bounded("dfa_subsets_have_a_memory_budget", || {
        let grammar = regex_grammar("[ab]*a[ab]{20}").unwrap();
        assert!(matches!(
            CompiledGrammar::new(&grammar, common::adversarial_vocabulary()),
            Err(GrammarError::TooLarge(_))
        ));
    });
}

#[test]
fn schema_budget_errors_are_not_widened() {
    bounded("schema_budget_errors_are_not_widened", || {
        for schema in [
            json!({"enum": ["a", "b"], "type": "string", "pattern": "[ab]*a[ab]{20}"}),
            json!({"type": "string", "pattern": "a".repeat(65_537)}),
            json!({"enum": (0..16_385).collect::<Vec<_>>()}),
        ] {
            assert!(matches!(
                json_schema_grammar(&schema, JsonOptions::default()),
                Err(GrammarError::TooLarge(_))
            ));
        }
    });
}

#[test]
fn choice_deduplication_preserves_first_occurrence_order() {
    let choices = vec!["b".to_owned(), "a".to_owned(), "b".to_owned()];
    assert_eq!(
        choice_grammar(&choices).unwrap(),
        choice_grammar(&choices[..2]).unwrap()
    );
}

#[test]
fn recursive_enum_narrowing_shares_the_depth_budget() {
    bounded("recursive_enum_narrowing_shares_the_depth_budget", || {
        let schema = json!({"$defs": {"a": {
            "enum": ["x", "y"], "properties": {"a": {"$ref": "#/$defs/a"}}
        }}, "$ref": "#/$defs/a"});
        assert!(matches!(
            json_schema_grammar(&schema, JsonOptions::default()),
            Err(GrammarError::TooLarge(_))
        ));
    });
}

#[test]
fn rule_graph_analysis_obeys_the_work_budget() {
    bounded("rule_graph_analysis_obeys_the_work_budget", || {
        use runtime_grammar::{Expr, GrammarBuilder};
        let mut builder = GrammarBuilder::new();
        let rules: Vec<_> = (0..4000).map(|_| builder.declare()).collect();
        for (index, &rule) in rules.iter().enumerate() {
            let next = &rules[index + 1..rules.len().min(index + 251)];
            let expr = if next.is_empty() {
                Expr::literal("x")
            } else {
                Expr::alt(next.iter().copied().map(Expr::Rule))
            };
            builder.define(rule, expr);
        }
        let grammar = builder.finish(rules[0]).unwrap();
        assert!(matches!(
            CompiledGrammar::new(&grammar, common::adversarial_vocabulary()),
            Err(GrammarError::TooLarge(_))
        ));
    });
}

#[test]
fn typed_enums_charge_actual_matching_work() {
    bounded("typed_enums_charge_actual_matching_work", || {
        let values: Vec<_> = (0..16_000).map(|index| format!("v{index}")).collect();
        let typed = json_schema_grammar(
            &json!({"enum":values, "type":"string"}),
            JsonOptions::default(),
        )
        .unwrap();
        let untyped = json_schema_grammar(&json!({"enum":values}), JsonOptions::default()).unwrap();
        assert_eq!(typed, untyped);
    });
}

#[test]
fn schema_depth_exhaustion_never_widens_the_language() {
    bounded("schema_depth_exhaustion_never_widens_the_language", || {
        let mut schema = json!({"const": 1});
        for _ in 0..129 {
            schema = json!({"type":"object", "properties":{"a":schema}});
        }
        assert!(matches!(
            json_schema_grammar(&schema, JsonOptions::default()),
            Err(GrammarError::TooLarge(_))
        ));
    });
}

#[test]
fn nested_array_expansion_is_budgeted_before_cloning() {
    bounded("nested_array_expansion_is_budgeted_before_cloning", || {
        let mut schema = json!({"const": 1});
        for _ in 0..40 {
            schema = json!({"type":"array", "items":schema});
        }
        assert!(matches!(
            json_schema_grammar(&schema, JsonOptions::default()),
            Err(GrammarError::TooLarge(_))
        ));
    });
}

#[test]
fn schema_depth_counts_calls_across_narrowing_compilers() {
    bounded(
        "schema_depth_counts_calls_across_narrowing_compilers",
        || {
            let mut definitions = serde_json::Map::new();
            for index in 0..32 {
                definitions.insert(format!("n{index}"), json!({"type":"object", "properties":{"a":{"$ref":format!("#/$defs/n{}", (index+1)%32)}}}));
            }
            definitions
                .get_mut("n0")
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("enum".into(), json!(["x", "y"]));
            let schema = json!({"$defs": definitions, "$ref":"#/$defs/n0"});
            assert!(matches!(
                json_schema_grammar(&schema, JsonOptions::default()),
                Err(GrammarError::TooLarge(_))
            ));
        },
    );
}

#[test]
fn container_enum_narrowing_shares_compile_budget() {
    bounded("container_enum_narrowing_shares_compile_budget", || {
        let values: Vec<_> = (0..4096).map(|i| json!({"b":i,"a":"x"})).collect();
        let schema = json!({"enum":values,"type":"object","properties":{
            "a":{"type":"string","pattern":"[a-z]{1,64}"},"b":{"type":"integer"}
        },"required":["a","b"],"additionalProperties":false});
        assert!(matches!(
            json_schema_grammar(&schema, JsonOptions::default()),
            Err(GrammarError::TooLarge(_))
        ));
    });
}
