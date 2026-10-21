//! The replicated vision encoder image (`deepseek41.encoder-rank-images.v1`)
//! as a package plans it: a planned component with no file, whose image and
//! per-rank indexes the bundle's weight plan supplies from the checkpoint
//! (`model_deepseek41_contract::encoder_plan`), each index naming its rank's
//! planned identity. A package plans it only for a checkpoint that stores
//! the vision tower the declaration pins, and only when it serves images;
//! the component brings the package's image inputs (`crate::package::draft`).

use std::path::Path;

use model_compiler_sdk::ComponentDraftV1;
use model_deepseek41_contract::encoder::{self, views::IMAGE_ROLE};
use model_deepseek41_contract::geometry::Geometry;
use runtime_bundle::{BundleError, BundleOpaqueId};

pub const IMAGE_FORMAT: &str = "deepseek41.encoder-rank-image.v1";
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

/// Whether the checkpoint at `root` of `geometry` provides images: the
/// geometry's vision tower is the one the declaration pins, and the weight
/// map names a tensor the declaration reads. A synthetic geometry's
/// smaller tower, or a checkpoint without one, serves text alone.
pub fn has_tower(root: &Path, geometry: &Geometry) -> Result<bool, String> {
    if !encoder::serves(geometry) {
        return Ok(false);
    }
    has_tower_in(&crate::checkpoint::metadata(root)?, geometry)
}

pub(crate) fn has_tower_in(
    metadata: &projection_spec::bind::CheckpointMetadata,
    geometry: &Geometry,
) -> Result<bool, String> {
    if !encoder::serves(geometry) {
        return Ok(false);
    }
    let tensors = encoder::tensors().map_err(|e| e.0)?;
    Ok(tensors
        .iter()
        .any(|name| metadata.tensors.contains_key(name)))
}

/// The component for the checkpoint at `root` served at `ranks`, after
/// planning rank 0's encoder image from its headers, which checks every
/// tensor the declaration reads against it.
pub fn planned(root: &Path, ranks: u32) -> Result<ComponentDraftV1, String> {
    let request = crate::checkpoint::request(root, ranks, 0)?;
    model_deepseek41_contract::encoder_plan::plan(&request)?;
    component().map_err(|error| error.to_string())
}

/// The device bytes one rank holds for the encoder beyond the text image:
/// the replicated encoder image and every encoder buffer at the declared
/// capacity, the media arena and the prefill's arena source rows included.
pub fn device_bytes() -> Result<u64, String> {
    Ok(encoder::views::image_bytes()? + encoder::buffer_bytes().map_err(|e| e.0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_component_is_the_planned_encoder_image() {
        let component = component().unwrap();
        assert_eq!(component.id().as_str(), IMAGE_COMPONENT);
        // The image and the buffers: hundreds of MiB, below one GiB each.
        let bytes = device_bytes().unwrap();
        assert!(bytes > encoder::views::image_bytes().unwrap(), "{bytes}");
        assert!(bytes < 4 << 30, "{bytes}");
    }
}
