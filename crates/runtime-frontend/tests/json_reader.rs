#![forbid(unsafe_code)]
//! The transport's lossless JSON reader reads the shared vectors as the
//! declared format's copy does.

use frontend_format::test_support::json_reader::assert_json_reader_cases;
use runtime_server::FrontendJsonV1;
use serde_json::{Value, json};

fn tagged(value: &FrontendJsonV1) -> Value {
    match value {
        FrontendJsonV1::Null => Value::Null,
        FrontendJsonV1::Bool(flag) => Value::Bool(*flag),
        FrontendJsonV1::Number(number) => json!({"number": number.to_string()}),
        FrontendJsonV1::String(text) => json!({"string": text}),
        FrontendJsonV1::Array(items) => {
            json!({"array": items.iter().map(tagged).collect::<Vec<_>>()})
        }
        FrontendJsonV1::Object(entries) => json!({"object": entries
            .iter()
            .map(|(name, value)| json!([name, tagged(value)]))
            .collect::<Vec<_>>()}),
    }
}

#[test]
fn the_shared_json_reader_vectors_read_alike() {
    assert_json_reader_cases(tagged, FrontendJsonV1::to_json_value);
}
