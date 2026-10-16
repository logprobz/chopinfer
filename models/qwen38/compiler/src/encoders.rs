//! The replicated vision encoder images (media contract, "Qwen3.8 Flash"):
//! the checkpoint's `model.visual.*` tensors as row-major BF16 at the
//! offsets `model_qwen38_contract::encoder::views::layout` fixes, one identical
//! image per rank. A package plans them: after checking the checkpoint's
//! vision tower it declares the image as a planned component, and the weight
//! plan supplies the images and their indexes from the checkpoint.
use crate::{Result, catalog::Catalog};
use model_compiler_sdk::ComponentDraftV1;
use model_qwen38_contract::encoder::views::{self, IMAGE_ROLE, Stored};
use runtime_bundle::{BundleError, BundleOpaqueId};
use runtime_weights::DType;

pub const IMAGE_FORMAT: &str = "qwen38.encoder-rank-image.v1";
pub const IMAGE_COMPONENT: &str = "encoder-rank-images";

/// Checks every source tensor of `layout` against the catalog: BF16 with
/// exactly the shape the stored view is taken from.
fn check_sources(catalog: &Catalog, layout: &[Stored]) -> Result<()> {
    if let Some(source) = &catalog.precision {
        source.require_bf16_vision()?;
    }
    for view in layout {
        let tensor = catalog
            .tensors
            .get(&view.name)
            .ok_or_else(|| format!("checkpoint lacks {}", view.name))?;
        if tensor.dtype != DType::Bf16
            || tensor.shape != view.source_shape
            || tensor.bytes != 2 * view.source_shape.iter().product::<u64>()
        {
            return Err(
                format!("{} is not BF16 of shape {:?}", view.name, view.source_shape).into(),
            );
        }
    }
    Ok(())
}

/// The encoder part of a package: the weight plan supplies the images and
/// their indexes.
pub struct EncoderAssets {
    pub positions: ImagePositions,
    /// Set only by [`EncoderAssets::planned`]: a package without rank images
    /// advertises image inputs only for a checkpoint whose vision tower was
    /// checked.
    pub checked_tower: Option<CheckedTower>,
}

/// Proof that a checkpoint's catalog holds every tensor of the encoder
/// layout, BF16 of the shape its view is taken from, which the weight plan
/// copies. Only [`EncoderAssets::planned`] makes one.
#[derive(Clone, Copy, Debug)]
pub struct CheckedTower(());

impl EncoderAssets {
    /// The encoders of a package without rank images, after checking the
    /// checkpoint's vision tower: the weight plan supplies the images and
    /// indexes from that checkpoint.
    pub fn planned(catalog: &Catalog, positions: ImagePositions) -> Result<Self> {
        check_sources(catalog, &views::layout_for(catalog.geometry))?;
        Ok(Self {
            positions,
            checked_tower: Some(CheckedTower(())),
        })
    }
}

/// The RoPE positions image rows take in the language model.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ImagePositions {
    /// HF's `get_rope_index`: (T, H, W) grid positions (`grid-3axis.v1`).
    #[default]
    Grid3Axis,
    /// Every axis takes the token index, as the pinned vLLM does.
    Sequential,
}

/// The package component: the encoder image as a planned component (role
/// and format, no file) whose images and indexes the weight plan supplies.
pub(crate) fn component() -> std::result::Result<ComponentDraftV1, BundleError> {
    ComponentDraftV1::planned(
        BundleOpaqueId::new(IMAGE_COMPONENT)?,
        BundleOpaqueId::new(std::str::from_utf8(IMAGE_ROLE).expect("ASCII role"))?,
        BundleOpaqueId::new(IMAGE_FORMAT)?,
    )
}

#[cfg(test)]
#[path = "encoders_tests.rs"]
mod tests;
