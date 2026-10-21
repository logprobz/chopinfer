//! Check the shipped declaration against the host's implemented recipes.
#[test]
fn shipped_media_declaration_is_supported() {
    let value: serde_json::Value =
        serde_json::from_str(include_str!("../src/frontend.json")).unwrap();
    let descriptor: runtime_frontend::FrontendDescriptorV1 = serde_json::from_value(value).unwrap();
    runtime_host::validate_media_recipes_v1(&descriptor).unwrap();
}

/// The shipped image recipe serves `load_image`'s policy, and every file of
/// runtime-media's image decode corpus decodes as `load_image` did in this
/// model's capture (`fixtures/mimo26/image-decode.json`).
#[test]
fn image_decode_matches_the_captured_reference_loader() {
    let descriptor = model_mimo26::frontend::descriptor();
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
        "/../../../fixtures/mimo26/image-decode.json"
    );
    let checked = runtime_host::image_decode::check_image_decode_v1(
        &image.recipe,
        std::path::Path::new(capture),
    );
    assert!(checked >= 100, "{checked} files");
}

#[test]
fn every_listed_geometry_selection_passes_the_host_recipe_check() {
    use model_mimo26_contract::{encoders::Encoders, geometry};
    let registry = model_mimo26::checkpoint::registry();
    for g in geometry::GEOMETRIES {
        for selection in Encoders::SETS {
            if selection.any() && !g.encoders {
                continue;
            }
            let bytes = registry
                .component_for_geometry(model_mimo26::checkpoint::default_identity(), selection, g)
                .unwrap();
            let descriptor =
                runtime_frontend::FrontendDescriptorV1::from_component_bytes(&bytes).unwrap();
            runtime_host::validate_media_recipes_v1(&descriptor)
                .unwrap_or_else(|error| panic!("{} {selection:?}: {error}", g.id));
        }
    }
}

#[test]
fn unsupported_audio_is_refused_before_frontend_emission() {
    use model_mimo26_contract::{encoders::Encoders, geometry::V26_FLASH};
    for (hop, rate, segment, field, declared) in [
        (160, 24_000, 6_000, "maximum_seconds", 400),
        (480, 24_000, 6_000, "maximum_seconds", 1200),
        (160, 16_000, 6_000, "sample_rate", 16000),
        (240, 24_000, 3_000, "segment_frames", 3000),
    ] {
        let mut g = V26_FLASH;
        g.id = "unsupported-audio";
        g.shape.audio_tokenizer.hop_length = hop;
        g.shape.audio_tokenizer.sampling_rate = rate;
        g.shape.audio.audio_segment_size = segment;
        let g = Box::leak(Box::new(g));
        for selection in Encoders::SETS {
            let emitted = model_mimo26::checkpoint::registry().component_for_geometry(
                model_mimo26::checkpoint::default_identity(),
                selection,
                g,
            );
            if selection.audio {
                let error = emitted.unwrap_err().to_string();
                let expected = runtime_frontend_contract::audio_log_mel_v1_fixed_parameters()
                    [field]
                    .as_u64()
                    .unwrap();
                assert!(
                    error.contains(&format!("requires {field}={expected}")),
                    "{error}"
                );
                assert!(error.contains(&format!("declares {declared}")), "{error}");
            } else {
                let descriptor =
                    runtime_frontend::FrontendDescriptorV1::from_component_bytes(&emitted.unwrap())
                        .unwrap();
                runtime_host::validate_media_recipes_v1(&descriptor).unwrap();
            }
        }
    }
}
