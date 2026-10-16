#![forbid(unsafe_code)]
//! Every row of `python_semantics.json`, which `tools/refrun/python_semantics.py`
//! wrote from Python, reproduced by the toolkit.

use frontend_format::toolkit::{
    TojsonOptionsV1, json_to_string_v1, python_capitalize_v1, python_rstrip_v1, python_split_v1,
    python_strip_v1, tojson_v1,
};
use frontend_format::{FrontendJsonEncodingV1, FrontendJsonV1};
use serde_json::Value;

fn document() -> Value {
    serde_json::from_str(include_str!("python_semantics.json")).unwrap()
}

fn rows<'a>(document: &'a Value, section: &str) -> &'a [Value] {
    let rows = document[section].as_array().unwrap();
    assert!(!rows.is_empty(), "{section} has rows");
    rows
}

fn text<'a>(row: &'a Value, key: &str) -> &'a str {
    row[key].as_str().unwrap()
}

/// A JSON value input, read as the request reader reads it.
fn value(row: &Value) -> FrontendJsonV1 {
    serde_json::from_str(text(row, "input")).unwrap()
}

fn options(arguments: &Value) -> TojsonOptionsV1 {
    let mut options = TojsonOptionsV1::default();
    for (name, argument) in arguments.as_object().unwrap() {
        match name.as_str() {
            "ensure_ascii" => options.ensure_ascii = argument.as_bool().unwrap(),
            "sort_keys" => options.sort_keys = argument.as_bool().unwrap(),
            "indent" => {
                options.indent = Some(match argument {
                    Value::Number(spaces) => " ".repeat(spaces.as_u64().unwrap() as usize),
                    indent => indent.as_str().unwrap().to_owned(),
                });
            }
            "separators" => {
                let pair = argument.as_array().unwrap();
                options.separators = Some((
                    pair[0].as_str().unwrap().to_owned(),
                    pair[1].as_str().unwrap().to_owned(),
                ));
            }
            other => panic!("unknown tojson argument {other}"),
        }
    }
    options
}

#[test]
fn the_rows_come_from_a_recorded_python() {
    let document = document();
    assert_eq!(document["schema"], "chop-python-semantics-v1");
    assert!(document["python"].as_str().unwrap().starts_with("3."));
    assert!(matches!(
        document["interpreter"].as_str(),
        Some("CHOPINFER_REFRUN_PYTHON" | "python3")
    ));
}

#[test]
fn tojson_matches_json_dumps_under_every_argument_set() {
    let document = document();
    for row in rows(&document, "tojson") {
        let options = options(&row["args"]);
        let input = value(row);
        assert_eq!(
            tojson_v1(&input, &options),
            text(row, "output"),
            "{} {}",
            row["input"],
            row["args"]
        );
        // HF's default arguments are the declared Python JSON encoding.
        if options == TojsonOptionsV1::default() {
            assert_eq!(
                json_to_string_v1(&input, FrontendJsonEncodingV1::PythonJsonV1, "test").unwrap(),
                text(row, "output")
            );
        }
    }
}

/// `json.dumps` writes a float as its `repr`, in `tojson` and in the
/// declared Python JSON encoding alike.
#[test]
fn floats_write_python_repr() {
    let document = document();
    for row in rows(&document, "float_repr") {
        let input = value(row);
        assert!(matches!(input, FrontendJsonV1::Number(_)));
        assert_eq!(
            tojson_v1(&input, &TojsonOptionsV1::default()),
            text(row, "output"),
            "{}",
            row["input"]
        );
        assert_eq!(
            json_to_string_v1(&input, FrontendJsonEncodingV1::PythonJsonV1, "test").unwrap(),
            text(row, "output"),
            "{}",
            row["input"]
        );
    }
}

#[test]
fn strip_matches_python_whitespace() {
    let document = document();
    for row in rows(&document, "strip") {
        let input = text(row, "input");
        assert_eq!(python_strip_v1(input), text(row, "strip"), "{input:?}");
        assert_eq!(python_rstrip_v1(input), text(row, "rstrip"), "{input:?}");
    }
}

#[test]
fn split_matches_python_split_on_a_separator() {
    let document = document();
    for row in rows(&document, "split") {
        let expected: Vec<&str> = row["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|piece| piece.as_str().unwrap())
            .collect();
        assert_eq!(
            python_split_v1(text(row, "input"), text(row, "separator")),
            expected
        );
    }
}

#[test]
fn capitalize_matches_python_titlecase_and_final_sigma() {
    let document = document();
    for row in rows(&document, "capitalize") {
        let input = text(row, "input");
        assert_eq!(
            python_capitalize_v1(input),
            text(row, "output"),
            "{input:?}"
        );
    }
}
