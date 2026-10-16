#![forbid(unsafe_code)]
//! The key-value tool grammar: `<tool_call>name<arg_key>k</arg_key>
//! <arg_value>v</arg_value></tool_call>`, with no function element, raw
//! string values and JSON for every other value. A declaration names the
//! markers its grammar reads and no others.
#[allow(dead_code)]
mod common;

use frontend_format::{FormatDescriptorV1, FrontendToolGrammarV1};

fn descriptor() -> FormatDescriptorV1 {
    FormatDescriptorV1::from_component_bytes(common::KEY_VALUE.as_bytes()).unwrap()
}

#[test]
fn the_grammar_requires_its_own_markers() {
    let valid = descriptor();
    let mut function = valid.clone();
    function.parser.function_open = "<f>".into();
    assert!(
        function
            .validate()
            .unwrap_err()
            .to_string()
            .contains("function_open must be empty")
    );
    let mut value = valid.clone();
    value.parser.parameter_value_open = " ".into();
    assert!(
        value
            .validate()
            .unwrap_err()
            .to_string()
            .contains("parameter_value_open cannot be blank")
    );
    let mut other = valid;
    let parser = &mut other.parser;
    parser.tool_grammar = FrontendToolGrammarV1::XmlFunctionParametersV1;
    parser.function_open = "<f>".into();
    parser.function_name_close = ">".into();
    parser.function_close = "</f>".into();
    assert!(
        other
            .validate()
            .unwrap_err()
            .to_string()
            .contains("only by the key-value tool grammar")
    );
    other.parser.parameter_value_open.clear();
    other.validate().unwrap();
}
