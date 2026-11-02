//! Fixed decoder refusals must be reviewed into the logging allowlist.

use runtime_frontend::loggable_refusal_v1;

/// Reads calls with one quoted literal, across whitespace and an optional
/// trailing comma. Formatted messages and other expressions stay withheld.
/// Raw literals and comments need parser support before they can be checked.
fn literal_arguments(source: &str, function: &str) -> Vec<String> {
    let parts: Vec<_> = function.split_whitespace().collect();
    let (&name, rest) = parts.split_first().unwrap();
    let mut found = Vec::new();
    for (at, _) in source.match_indices(name) {
        if source[..at].ends_with(|ch: char| ch.is_alphanumeric() || ch == '_') {
            continue;
        }
        let suffix = rest
            .iter()
            .try_fold(&source[at + name.len()..], |suffix, part| {
                suffix.trim_start().strip_prefix(*part)
            });
        let Some(argument) = suffix.and_then(|suffix| suffix.trim_start().strip_prefix('(')) else {
            continue;
        };
        let argument = argument.trim_start();
        assert!(
            !argument
                .strip_prefix('r')
                .is_some_and(|raw| raw.trim_start_matches('#').starts_with('"')),
            "{function}: the guard cannot read a raw string literal",
        );
        assert_no_comment(argument, function);
        if !argument.starts_with('"') {
            continue;
        }
        let mut strings = serde_json::Deserializer::from_str(argument).into_iter::<String>();
        let message = strings
            .next()
            .unwrap()
            .expect("fixed refusal uses a plain string literal");
        let rest = argument[strings.byte_offset()..].trim_start();
        assert_no_comment(rest, function);
        let rest = rest.strip_prefix(',').unwrap_or(rest).trim_start();
        assert_no_comment(rest, function);
        if rest.starts_with(')') {
            found.push(message);
        }
    }
    found
}

fn assert_no_comment(text: &str, function: &str) {
    assert!(
        !text.starts_with("//") && !text.starts_with("/*"),
        "{function}: the guard cannot read a commented argument",
    );
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
        if name == "decode.rs" {
            for function in ["refuse_late", "misuse = Some"] {
                for literal in literal_arguments(source, function) {
                    let message = format!("invalid conversation sequence: {literal}");
                    assert_eq!(
                        loggable_refusal_v1(&message),
                        message,
                        "{name}: add the {function} literal to LOGGED_REFUSALS_V1",
                    );
                    checked += 1;
                }
            }
        }
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

#[test]
fn misuse_assignments_allow_whitespace_between_tokens() {
    let source = r#"
        self.misuse = Some("one");
        self.misuse
            = Some (
                "two",
            );
        self.misuse=Some("three");
        self.misuse = Some(message);
        other_misuse = Some("different field");
    "#;
    assert_eq!(
        literal_arguments(source, "misuse = Some"),
        ["one", "two", "three"]
    );
}

#[test]
fn unreadable_literal_arguments_fail_loudly() {
    for source in [
        r#"malformed_v1(r"raw")"#,
        r##"malformed_v1(r#"raw"#)"##,
        r#"malformed_v1(/* explanation */ "commented")"#,
        "malformed_v1(// explanation\n\"commented\")",
        r#"malformed_v1("commented" /* explanation */)"#,
        r#"malformed_v1("commented", /* explanation */)"#,
    ] {
        assert!(
            std::panic::catch_unwind(|| literal_arguments(source, "malformed_v1")).is_err(),
            "guard silently skipped {source}",
        );
    }
}
