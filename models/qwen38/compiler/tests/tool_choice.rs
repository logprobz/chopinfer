use model_qwen38_contract::frontend::Qwen38Format;
use runtime_frontend::test_support::documents::PairedFrontendV1;
use runtime_frontend::test_support::tool_choice;

fn frontend() -> PairedFrontendV1 {
    PairedFrontendV1::chat_format::<Qwen38Format>(model_qwen38::frontend::descriptor()).unwrap()
}

#[test]
fn only_an_offered_function_can_be_called() {
    tool_choice::only_an_offered_function_can_be_called(&frontend());
}

#[test]
fn every_tool_choice_renders_the_offered_tools() {
    tool_choice::every_tool_choice_renders_the_offered_tools(&frontend(), "<tools>");
}

#[test]
fn tool_references_are_refused() {
    tool_choice::tool_references_are_refused(&frontend());
}

/// A forced call's enumeration and constant values, which the grammar writes
/// as they are, read back through this model's own parser declaration,
/// whose wrappers carry the template's whitespace.
#[test]
fn forced_literal_values_read_back() {
    let descriptor = frontend_format::FormatDescriptorV1::from_component_bytes(
        model_qwen38::frontend::COMPONENT,
    )
    .unwrap();
    frontend_format::test_support::forced_literals::assert_forced_literals_read_back(&descriptor);
}

/// A forced call's additional-property names exclude its property names
/// without outgrowing the grammar budgets, even for two or three tools of
/// 100 properties named as long as this model's declaration permits.
#[test]
fn long_property_names_compile() {
    let descriptor = frontend_format::FormatDescriptorV1::from_component_bytes(
        model_qwen38::frontend::COMPONENT,
    )
    .unwrap();
    frontend_format::test_support::forced_literals::assert_long_property_names_compile(&descriptor);
}
