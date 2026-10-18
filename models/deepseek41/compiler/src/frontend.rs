//! The declared chat frontend.

use model_deepseek41_contract::geometry::Geometry;
use runtime_frontend_contract::FrontendDescriptorV1;

/// The frontend component, rendered by `model_deepseek41_contract::frontend`
/// and checked against the vendor's encoder pairs by
/// `tests/frontend_goldens.rs`. Its `tokenizer` and `vocabulary` sections
/// are `tokenizer-contract.json`'s.
pub const COMPONENT: &[u8] = include_bytes!("frontend.json");

/// The image declaration a package with the encoder image adds to its
/// frontend as its `multimodal` section: the media arena and the encoder
/// as `model_deepseek41_contract::encoder` declares them, and the image
/// modality, prepared by the vendor processor's centered pad (patch 14,
/// 3 x 3 cells, at least 295,936 pixels, at most 1,024 tokens with the
/// learned delimiters, gray 127 padding, values over 255 normalized by mean
/// and std 0.5) and rendered as the encoder's image placeholder.
pub const MULTIMODAL: &[u8] = include_bytes!("frontend-multimodal.json");

pub fn descriptor() -> FrontendDescriptorV1 {
    FrontendDescriptorV1::from_component_bytes(COMPONENT)
        .expect("the pinned frontend declaration is valid")
}

/// The flash geometry's frontend with the image declaration.
pub fn multimodal_descriptor() -> FrontendDescriptorV1 {
    let bytes = multimodal_component_for(&model_deepseek41_contract::geometry::FLASH)
        .expect("the pinned multimodal frontend declaration is valid");
    FrontendDescriptorV1::from_component_bytes(&bytes)
        .expect("the pinned multimodal frontend declaration is valid")
}

/// The frontend component of a package of `geometry` that serves images:
/// [`component_for`] with [`MULTIMODAL`]'s section as its `multimodal`
/// member. Only a geometry whose vision tower the encoder declaration pins
/// serves images.
pub fn multimodal_component_for(geometry: &Geometry) -> Result<Vec<u8>, String> {
    if !model_deepseek41_contract::encoder::serves(geometry) {
        return Err(format!(
            "geometry {}'s vision tower is not the encoder declaration's: it serves no images",
            geometry.id
        ));
    }
    let mut component: serde_json::Value =
        serde_json::from_slice(&component_for(geometry)?).map_err(|e| e.to_string())?;
    let mut section: serde_json::Value =
        serde_json::from_slice(MULTIMODAL).map_err(|e| e.to_string())?;
    component
        .as_object_mut()
        .ok_or("the frontend component is not an object")?
        .insert("multimodal".into(), section["multimodal"].take());
    let bytes = serde_json::to_vec_pretty(&component).map_err(|e| e.to_string())?;
    let descriptor = FrontendDescriptorV1::from_component_bytes(&bytes)
        .map_err(|e| format!("the image frontend of geometry {}: {e}", geometry.id))?;
    runtime_frontend_contract::tokenizer_contract_v1(&descriptor)
        .map_err(|e| format!("the image frontend of geometry {}: {e}", geometry.id))?;
    Ok(bytes)
}

/// The frontend component of a package of `geometry`: [`COMPONENT`] with
/// the geometry's logits width as its vocabulary size and its context as
/// the total and generated token limits, the tokenizer's own limits kept.
/// The flash geometry's is [`COMPONENT`] itself.
pub fn component_for(geometry: &Geometry) -> Result<Vec<u8>, String> {
    let mut component: serde_json::Value =
        serde_json::from_slice(COMPONENT).map_err(|e| e.to_string())?;
    let original = component.clone();
    component["vocabulary"]["size"] = geometry.shape.vocab.into();
    component["context"]["maximum_total_tokens"] = geometry.context_tokens().into();
    component["limits"]["maximum_generated_tokens"] = geometry.context_tokens().into();
    if component == original {
        return Ok(COMPONENT.to_vec());
    }
    let bytes = serde_json::to_vec_pretty(&component).map_err(|e| e.to_string())?;
    let descriptor = FrontendDescriptorV1::from_component_bytes(&bytes)
        .map_err(|e| format!("the frontend of geometry {}: {e}", geometry.id))?;
    runtime_frontend_contract::tokenizer_contract_v1(&descriptor)
        .map_err(|e| format!("the frontend of geometry {}: {e}", geometry.id))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The serving defaults plan V6 fixes:
    /// temperature and top_p 1.0, thinking on at effort high, and the 1M
    /// context the checkpoint's `max_position_embeddings` gives. The
    /// checkpoint ships no generation_config.json. An omitted max_tokens
    /// fills the remaining context, as the transport does for every model.
    #[test]
    fn the_declaration_holds_the_planned_defaults() {
        let descriptor = descriptor();
        let sampling = descriptor.generation_config;
        assert_eq!(
            (sampling.temperature, sampling.top_p),
            (Some(1.0), Some(1.0))
        );
        assert_eq!(
            (sampling.top_k, sampling.min_p, sampling.repetition_penalty),
            (None, None, None)
        );
        assert!(descriptor.reasoning.default_enabled);
        assert_eq!(descriptor.reasoning.default_effort, "high");
        assert_eq!(descriptor.context.maximum_total_tokens, 1_048_576);
    }

    /// The image declaration matches the encoder contract: the arena's rows,
    /// pages and hidden, the encoder's payload, and the recipe's token
    /// budget and cell; the text frontend declares no media.
    #[test]
    fn the_image_section_matches_the_encoder_contract() {
        use model_deepseek41_contract::encoder::{
            ARENA_HIDDEN, ARENA_ROWS, MAX_PAYLOAD_BYTES, PAGE_ROWS, PATCH_COLUMNS, declarations,
        };
        assert!(descriptor().multimodal.is_none());
        let media = multimodal_descriptor().multimodal.unwrap();
        assert_eq!(
            (media.arena.rows, media.arena.page_rows, media.arena.hidden),
            (ARENA_ROWS as u32, PAGE_ROWS as u32, ARENA_HIDDEN)
        );
        let [encoder] = &media.encoders[..] else {
            panic!("one encoder");
        };
        assert_eq!(encoder.payload_columns, PATCH_COLUMNS);
        assert_eq!(encoder.maximum_payload_bytes as u64, MAX_PAYLOAD_BYTES);
        let [image] = &media.modalities[..] else {
            panic!("one modality");
        };
        assert_eq!(image.render.pad, "<｜deepseek_image｜>");
        let declared = serde_json::to_value(&declarations().unwrap()[0].input.recipes).unwrap();
        let parameters = &image.recipe.parameters;
        assert_eq!(declared[0]["tag"], image.recipe.tag.as_str());
        for key in ["patch", "downsample", "maximum_tokens"] {
            assert_eq!(declared[0][key], parameters[key], "{key}");
        }
        assert_eq!(image.maximum_tokens, 1024);
        // Every item of a full request fits the arena at once.
        assert!(u64::from(image.maximum_items) * 1024 <= ARENA_ROWS);
        // Only the multimodal member differs from the text component.
        let mut merged: serde_json::Value = serde_json::from_slice(
            &multimodal_component_for(&model_deepseek41_contract::geometry::FLASH).unwrap(),
        )
        .unwrap();
        merged.as_object_mut().unwrap().remove("multimodal");
        assert_eq!(
            merged,
            serde_json::from_slice::<serde_json::Value>(COMPONENT).unwrap()
        );
    }

    /// Each compiled geometry's frontend states its own logits width and
    /// context and keeps everything else of the declaration; flash's is the
    /// declaration's bytes.
    #[test]
    fn each_geometry_s_frontend_states_its_limits() {
        use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES};
        assert_eq!(component_for(&FLASH).unwrap(), COMPONENT);
        let declared: serde_json::Value = serde_json::from_slice(COMPONENT).unwrap();
        for geometry in GEOMETRIES {
            let bytes = component_for(geometry).unwrap();
            let frontend = FrontendDescriptorV1::from_component_bytes(&bytes).unwrap();
            assert_eq!(frontend.vocabulary.size, geometry.shape.vocab);
            assert_eq!(
                frontend.context.maximum_total_tokens as u64,
                geometry.context_tokens()
            );
            assert_eq!(
                frontend.limits.maximum_generated_tokens as u64,
                geometry.context_tokens()
            );
            let mut rest: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            let mut expected = declared.clone();
            for value in [&mut rest, &mut expected] {
                value["vocabulary"]["size"] = 0.into();
                value["context"]["maximum_total_tokens"] = 0.into();
                value["limits"]["maximum_generated_tokens"] = 0.into();
            }
            assert_eq!(rest, expected, "{}", geometry.id);
        }
    }
}
