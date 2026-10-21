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

/// The shipped image recipe serves `load_image`'s policy, and every file of
/// runtime-media's image decode corpus decodes as `load_image` did in this
/// model's capture (`fixtures/qwen38/image-decode.json`).
#[test]
fn image_decode_matches_the_captured_reference_loader() {
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("../src/frontend.json")).unwrap();
    let section: serde_json::Value =
        serde_json::from_str(include_str!("../src/frontend-multimodal.json")).unwrap();
    value["multimodal"] = section["multimodal"].clone();
    let descriptor: runtime_frontend::FrontendDescriptorV1 = serde_json::from_value(value).unwrap();
    let image = descriptor
        .multimodal
        .as_ref()
        .unwrap()
        .modalities
        .iter()
        .find(|modality| modality.kind == runtime_frontend::FrontendMediaKindV1::Image)
        .unwrap();
    let capture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../fixtures/qwen38/image-decode.json"
    );
    let checked = runtime_host::image_decode::check_image_decode_v1(
        &image.recipe,
        std::path::Path::new(capture),
    );
    assert!(checked >= 100, "{checked} files");
}
