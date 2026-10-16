//! Check the shipped declaration against the host's implemented recipes.
#[test]
fn shipped_media_declaration_is_supported() {
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("../src/frontend.json")).unwrap();
    let section: serde_json::Value =
        serde_json::from_str(include_str!("../src/frontend-multimodal.json")).unwrap();
    value["multimodal"] = section["multimodal"].clone();
    let descriptor: runtime_frontend::FrontendDescriptorV1 = serde_json::from_value(value).unwrap();
    runtime_host::validate_media_recipes_v1(&descriptor).unwrap();
}
