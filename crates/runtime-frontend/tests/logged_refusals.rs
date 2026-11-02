//! Fixed decoder refusals must be reviewed into the logging allowlist.

use runtime_frontend::loggable_refusal_v1;

/// Reads calls with one quoted literal, across whitespace and an optional
/// trailing comma. Formatted messages and other expressions stay withheld.
fn literal_arguments(source: &str, function: &str) -> Vec<String> {
    let mut found = Vec::new();
    for (at, _) in source.match_indices(function) {
        if source[..at].ends_with(|ch: char| ch.is_alphanumeric() || ch == '_') {
            continue;
        }
        let Some(argument) = source[at + function.len()..].trim_start().strip_prefix('(') else {
            continue;
        };
        let argument = argument.trim_start();
        if !argument.starts_with('"') {
            continue;
        }
        let mut strings = serde_json::Deserializer::from_str(argument).into_iter::<String>();
        let message = strings
            .next()
            .unwrap()
            .expect("fixed refusal uses a plain string literal");
        let rest = argument[strings.byte_offset()..].trim_start();
        let rest = rest.strip_prefix(',').unwrap_or(rest).trim_start();
        if rest.starts_with(')') {
            found.push(message);
        }
    }
    found
}

#[test]
fn every_fixed_decoder_refusal_is_loggable() {
    let sources = [
        ("decode.rs", include_str!("../src/decode.rs")),
        ("native.rs", include_str!("../src/native.rs")),
        (
            "format_decode.rs",
            include_str!("../../frontend-format/src/format_decode.rs"),
        ),
        (
            "chatml_decode.rs",
            include_str!("../../frontend-format/src/chatml_decode.rs"),
        ),
        (
            "marks.rs",
            include_str!("../../frontend-format/src/marks.rs"),
        ),
    ];
    let constructors = [
        ("malformed_v1", "malformed completion: "),
        ("sequence_v1", "invalid conversation sequence: "),
        ("resource_v1", "conversation resource limit: "),
        ("field_v1", "invalid conversation field: "),
    ];
    let mut checked = 0;
    for (name, source) in sources {
        for (function, family) in constructors {
            for literal in literal_arguments(source, function) {
                let message = format!("{family}{literal}");
                assert_eq!(
                    loggable_refusal_v1(&message),
                    message,
                    "{name}: add the fixed refusal to LOGGED_REFUSALS_V1"
                );
                checked += 1;
            }
        }
        if name == "decode.rs" {
            for literal in literal_arguments(source, "refuse_late") {
                let message = format!("invalid conversation sequence: {literal}");
                assert_eq!(loggable_refusal_v1(&message), message);
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "the guard must read fixed refusal calls");
}

#[test]
fn literal_calls_allow_line_breaks_but_no_formatted_arguments() {
    let source = r#"
        malformed_v1("one")
        malformed_v1(
            "two with \"quotes\"",
        )
        malformed_v1 ( "three" )
        malformed_v1(format!("formatted {value}"))
        malformed_v1(message)
        malformed_v1("an expression".to_owned())
        not_malformed_v1("different function")
    "#;
    assert_eq!(
        literal_arguments(source, "malformed_v1"),
        ["one", "two with \"quotes\"", "three"]
    );
}
