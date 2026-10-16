//! The checkpoint's declared chat frontend.

use runtime_frontend_contract::FrontendDescriptorV1;

pub const COMPONENT: &[u8] = include_bytes!("frontend.json");

/// The image input section (media contract, "Qwen3.8 Flash") that a package
/// with the encoder images adds to `COMPONENT` as its `multimodal` section.
/// The chat format writes the labels around each image itself.
pub const MULTIMODAL: &[u8] = include_bytes!("frontend-multimodal.json");

/// Bind model limits to the matched geometry while retaining tokenizer artifact limits.
pub(crate) fn component_for_geometry(
    bytes: &[u8],
    geometry: &model_qwen38_contract::geometry::Geometry,
) -> crate::Result<Vec<u8>> {
    let mut component: serde_json::Value = serde_json::from_slice(bytes)?;
    let original = component.clone();
    component["vocabulary"]["size"] = geometry.vocab().into();
    component["context"]["maximum_total_tokens"] = geometry.context_tokens().into();
    component["limits"]["maximum_generated_tokens"] = geometry.context_tokens().into();
    if component == original {
        return Ok(bytes.to_vec());
    }
    let bytes = serde_json::to_vec(&component)?;
    let descriptor = FrontendDescriptorV1::from_component_bytes(&bytes)?;
    runtime_frontend_contract::tokenizer_contract_v1(&descriptor)?;
    Ok(bytes)
}

pub fn descriptor() -> FrontendDescriptorV1 {
    FrontendDescriptorV1::from_component_bytes(COMPONENT)
        .expect("the pinned frontend declaration is valid")
}

/// `COMPONENT` with `MULTIMODAL`'s section as its last member, `multimodal`,
/// as pretty JSON: the frontend component of a package with image inputs.
pub fn multimodal_component() -> Vec<u8> {
    multimodal_component_for(
        &model_qwen38_contract::geometry::FLASH,
        crate::encoders::ImagePositions::Grid3Axis,
    )
}

/// `multimodal_component`, with the image modality's position rule removed
/// for `Sequential`.
#[cfg(test)]
pub fn multimodal_component_with(positions: crate::encoders::ImagePositions) -> Vec<u8> {
    multimodal_component_for(&model_qwen38_contract::geometry::FLASH, positions)
}

pub fn multimodal_component_for(
    geometry: &model_qwen38_contract::geometry::Geometry,
    positions: crate::encoders::ImagePositions,
) -> Vec<u8> {
    let mut root: serde_json::Value =
        serde_json::from_slice(COMPONENT).expect("the pinned frontend is JSON");
    let mut input: serde_json::Value =
        serde_json::from_slice(MULTIMODAL).expect("the pinned multimodal section is JSON");
    let mut section = input["multimodal"].take();
    if geometry.id != model_qwen38_contract::geometry::FLASH.id {
        let rows =
            model_qwen38_contract::encoder::MAX_IMAGE_PIXELS / geometry.vision_patch().pow(2);
        let columns = geometry.vision_channels()
            * geometry.vision_temporal_patch()
            * geometry.vision_patch().pow(2);
        section["arena"]["hidden"] = geometry.vision_out().into();
        section["encoders"][0]["name"] =
            format!("{}-vision", geometry.kernel_tag.replace('_', "-")).into();
        section["encoders"][0]["payload_columns"] = columns.into();
        section["encoders"][0]["maximum_payload_bytes"] =
            (u64::from(rows) * u64::from(columns) * 4).into();
        section["modalities"][0]["maximum_tokens"] = (rows / geometry.vision_merge().pow(2)).into();
        let parameters = &mut section["modalities"][0]["recipe"]["parameters"];
        parameters["factor"] = (geometry.vision_patch() * geometry.vision_merge()).into();
        parameters["patch"] = geometry.vision_patch().into();
        parameters["temporal_patch"] = geometry.vision_temporal_patch().into();
        parameters["merge"] = geometry.vision_merge().into();
    }
    if positions == crate::encoders::ImagePositions::Sequential {
        for modality in section["modalities"]
            .as_array_mut()
            .expect("the pinned section lists modalities")
        {
            modality
                .as_object_mut()
                .expect("a modality is an object")
                .remove("positions");
        }
    }
    root.as_object_mut()
        .expect("the pinned frontend is an object")
        .insert("multimodal".into(), section);
    let mut bytes = serde_json::to_vec_pretty(&root).expect("JSON values serialize");
    bytes.push(b'\n');
    bytes
}

pub fn multimodal_descriptor() -> FrontendDescriptorV1 {
    FrontendDescriptorV1::from_component_bytes(&multimodal_component())
        .expect("the pinned multimodal frontend declaration is valid")
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_qwen38_contract::encoder::{
        ARENA_ROWS, MAX_PAYLOAD_BYTES, PAGE_ROWS, PATCH_COLUMNS, VIT_OUT,
    };
    use runtime_frontend_contract::{
        FrontendMediaElementV1, FrontendMediaKindV1, FrontendPositionRuleV1,
    };

    #[test]
    fn geometry_limits_keep_the_tokenizer_admission_checks() {
        let mut geometry = model_qwen38_contract::geometry::FLASH;
        assert_eq!(
            component_for_geometry(COMPONENT, &geometry).unwrap(),
            COMPONENT
        );
        geometry.shape.vocab = 248076;
        assert_eq!(
            component_for_geometry(COMPONENT, &geometry)
                .unwrap_err()
                .to_string(),
            "invalid frontend declaration: boundary token '<|audio_pad|>' has identifier 248076 outside the vocabulary of 248076"
        );
        geometry.shape.vocab = model_qwen38_contract::geometry::FLASH.vocab();
        geometry.shape.context = 524288;
        assert!(
            component_for_geometry(COMPONENT, &geometry)
                .unwrap_err()
                .to_string()
                .contains(
                    "maximum_total_tokens 524288 exceeds the tokenizer's model_max_length 262144"
                )
        );
    }

    #[test]
    fn multimodal_section_matches_the_encoder_contract() {
        for geometry in model_qwen38_contract::geometry::GEOMETRIES {
            let bytes =
                multimodal_component_for(geometry, crate::encoders::ImagePositions::Grid3Axis);
            let frontend = FrontendDescriptorV1::from_component_bytes(&bytes).unwrap();
            let media = frontend.multimodal.unwrap();
            assert_eq!(media.arena.hidden, geometry.vision_out());
            let declaration = model_qwen38_contract::encoder::declarations_for(geometry).unwrap();
            assert_eq!(media.encoders[0].name, declaration[0].name);
            assert_eq!(
                media.encoders[0].payload_columns,
                geometry.vision_channels()
                    * geometry.vision_temporal_patch()
                    * geometry.vision_patch().pow(2)
            );
        }
        let merged = multimodal_descriptor();
        let media = merged.multimodal.as_ref().unwrap();
        assert_eq!(u64::from(media.arena.rows), ARENA_ROWS);
        assert_eq!(u64::from(media.arena.page_rows), PAGE_ROWS);
        assert_eq!(media.arena.hidden, VIT_OUT);
        assert_eq!(media.arena.element, FrontendMediaElementV1::Bf16);
        let [encoder] = media.encoders.as_slice() else {
            panic!("one encoder");
        };
        assert_eq!(encoder.name, "qwen38-vision");
        // The tower keeps F32 activations from its F32 pixel rows.
        assert_eq!(encoder.payload_element, FrontendMediaElementV1::F32);
        assert_eq!(encoder.payload_columns, PATCH_COLUMNS);
        assert_eq!(encoder.maximum_payload_bytes as u64, MAX_PAYLOAD_BYTES);
        let [image] = media.modalities.as_slice() else {
            panic!("one modality");
        };
        assert_eq!(image.kind, FrontendMediaKindV1::Image);
        assert_eq!(image.maximum_items, 64);
        // Images stream through the arena chunk by chunk, so it bounds no
        // count; one largest image must fit it.
        assert!(u64::from(image.maximum_tokens) <= ARENA_ROWS);
        assert_eq!(image.positions, Some(FrontendPositionRuleV1::Grid3AxisV1));
        // Everything else is the text frontend, unchanged.
        let mut text = merged.clone();
        text.multimodal = None;
        assert_eq!(text, descriptor());
        assert!(descriptor().multimodal.is_none());
        assert_eq!(multimodal_component(), multimodal_component());
    }

    /// Every frontend component the compiler writes opens under the bundle's
    /// export: the text component, and the multimodal one for each geometry
    /// in both image-position modes, as packaged.
    #[test]
    fn every_component_opens_under_the_bundle_export() {
        use crate::encoders::ImagePositions;
        let export = model_qwen38_contract::frontend::export();
        let open = |bytes: &[u8]| {
            runtime_frontend::BundleFrontendV1::from_export(&export, bytes)
                .unwrap_or_else(|error| panic!("{error}"))
        };
        for geometry in model_qwen38_contract::geometry::GEOMETRIES {
            let text = open(&component_for_geometry(COMPONENT, geometry).unwrap());
            assert!(text.descriptor().multimodal.is_none());
            for positions in [ImagePositions::Grid3Axis, ImagePositions::Sequential] {
                let bytes = multimodal_component_for(geometry, positions);
                let frontend = open(&component_for_geometry(&bytes, geometry).unwrap());
                let modalities = &frontend
                    .descriptor()
                    .multimodal
                    .as_ref()
                    .unwrap()
                    .modalities;
                assert_eq!(
                    modalities[0].positions.is_some(),
                    positions == ImagePositions::Grid3Axis
                );
                assert!(frontend.descriptor().format.get("media").is_none());
            }
        }
    }
}
