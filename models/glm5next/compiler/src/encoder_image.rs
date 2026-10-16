//! The replicated vision encoder image (`glm5next.encoder-rank-images.v1`)
//! as a package plans it: a planned component with no file, whose images
//! and per-rank indexes the bundle's weight plan supplies from the
//! checkpoint (`model_glm5next_contract::encoder_plan`), each index naming
//! its rank's planned identity. A package plans it only for a checkpoint
//! whose vision tower the declaration binds, and only when it serves
//! images; the component brings the package's media inputs
//! (`crate::package::draft`).

use model_compiler_sdk::ComponentDraftV1;
use model_glm5next_contract::encoder::views::IMAGE_ROLE;
use model_glm5next_contract::encoder_plan;
use runtime_bundle::{BundleError, BundleOpaqueId};
use runtime_bundle_abi::weight_plan::WeightPlanRequestV1;

pub const IMAGE_FORMAT: &str = "glm5next.encoder-rank-image.v1";
pub const IMAGE_COMPONENT: &str = "encoder-rank-images";

/// The package component: the encoder image as a planned component (role
/// and format, no file).
pub fn component() -> Result<ComponentDraftV1, BundleError> {
    ComponentDraftV1::planned(
        BundleOpaqueId::new(IMAGE_COMPONENT)?,
        BundleOpaqueId::new(std::str::from_utf8(IMAGE_ROLE).expect("ASCII role"))?,
        BundleOpaqueId::new(IMAGE_FORMAT)?,
    )
}

/// Whether the checkpoint `request` describes stores a vision tensor (one
/// under the declaration's prefix): one its weight map assigns, or, for a
/// checkpoint without a weight map, one in any file's header. A checkpoint
/// without one serves text alone. A shard the weight map does not name
/// never decides it.
pub fn has_tower(request: &WeightPlanRequestV1) -> Result<bool, String> {
    const INDEX: &str = "model.safetensors.index.json";
    let declarations = model_glm5next_contract::encoder::declarations().map_err(|e| e.0)?;
    let prefix = declarations[0].checkpoint.prefix.as_str();
    if let Some(bytes) = request.document(INDEX) {
        let index = model_glm5next_contract::admission::document(INDEX, bytes, bytes.len())?;
        let map = index["weight_map"]
            .as_object()
            .ok_or_else(|| format!("{INDEX} has no weight_map"))?;
        return Ok(map.keys().any(|name| name.starts_with(prefix)));
    }
    for file in &request.files {
        let serde_json::Value::Object(header) = model_glm5next_contract::admission::document(
            &format!("{}: header", file.name),
            &file.header,
            file.header.len(),
        )?
        else {
            return Err(format!("{}: header is not an object", file.name));
        };
        if header.keys().any(|name| name.starts_with(prefix)) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The component for the checkpoint `request` describes (its headers and
/// documents), after checking its vision tower against the declaration.
pub fn planned(request: &WeightPlanRequestV1) -> Result<ComponentDraftV1, String> {
    encoder_plan::check_tower(request)?;
    component().map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_compiler_sdk::{StagedContentV1, package_draft};
    use model_glm5next_contract::contract::Shape;
    use model_glm5next_contract::direct::test_support;
    use model_glm5next_contract::geometry::RuntimeValues;
    use model_glm5next_contract::quantization::FP8_BLOCK128;

    #[test]
    fn a_checked_tower_plans_the_encoder_image_in_the_package() {
        let component = planned(&test_support::request(0, |_| {})).unwrap();
        let components = vec![
            package_draft::frontend(StagedContentV1::bytes(crate::frontend::COMPONENT)).unwrap(),
            component,
        ];
        let draft = crate::package::draft(
            Shape::TP4,
            &FP8_BLOCK128,
            RuntimeValues { index_topk: 2048 },
            components,
            StagedContentV1::bytes(b"module".to_vec()),
            crate::package::Drafter::default(),
        )
        .unwrap();
        let manifest = model_compiler_sdk::plan_bundle_manifest_v1(&draft).unwrap();
        let manifest = serde_json::to_value(&manifest).unwrap();
        let planned = manifest["planned_components"].as_array().unwrap();
        let encoder = planned
            .iter()
            .find(|component| component["id"] == IMAGE_COMPONENT)
            .unwrap();
        assert_eq!(encoder["role"], std::str::from_utf8(IMAGE_ROLE).unwrap());
        assert_eq!(encoder["format"], IMAGE_FORMAT);
        // The text image stays planned beside it.
        assert!(
            planned
                .iter()
                .any(|component| component["id"] == crate::package::RANK_IMAGE_COMPONENT)
        );
        // The encoder image brings image inputs.
        let required = manifest["resource_requirements"]["required_capabilities"]
            .as_array()
            .unwrap();
        assert!(
            required
                .iter()
                .any(|capability| capability == model_glm5next_contract::encoder::MEDIA_CAPABILITY)
        );
    }

    /// A checkpoint stores the tower when its weight map assigns any tensor
    /// under the vision prefix, checked or not; one without any serves text
    /// alone, whatever a shard the map does not name holds. Without a
    /// weight map the headers decide.
    #[test]
    fn the_weight_map_decides_whether_the_tower_is_present() {
        use crate::checkpoint_headers::tests::index_headers;
        let indexed = |edit: fn(&mut serde_json::Map<String, serde_json::Value>)| {
            let mut request = test_support::request(0, edit);
            index_headers(&mut request);
            request
        };
        assert!(has_tower(&indexed(|_| {})).unwrap());
        assert!(
            has_tower(&indexed(|header| {
                header.remove("model.visual.merger.down_proj.weight");
            }))
            .unwrap()
        );
        let mut text =
            indexed(|header| header.retain(|name, _| !name.starts_with("model.visual.")));
        assert!(!has_tower(&text).unwrap());
        // A stale shard of vision tensors the weight map does not name.
        let mut orphan = test_support::request(0, |_| {}).files.remove(0);
        orphan.name = "stale-vision.safetensors".into();
        text.files.push(orphan);
        assert!(!has_tower(&text).unwrap());
        text.documents
            .retain(|document| document.name == "config.json");
        assert!(has_tower(&text).unwrap());
    }

    #[test]
    fn a_tower_the_declaration_does_not_bind_plans_no_encoder_image() {
        let missing = test_support::request(0, |header| {
            header.remove("model.visual.merger.down_proj.weight");
        });
        let error = planned(&missing).unwrap_err();
        assert!(error.contains("merger.down_proj.weight"), "{error}");
    }
}
