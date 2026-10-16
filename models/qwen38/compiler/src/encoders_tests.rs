use super::*;
use crate::catalog::{Placement, ScaleLayout, TensorRecord};

/// A catalog holding exactly the vision tower the encoder layout copies:
/// BF16 tensors of each view's source shape.
fn vision_catalog() -> Catalog {
    let tensors = views::layout_flash()
        .into_iter()
        .map(|view| {
            let bytes = 2 * view.source_shape.iter().product::<u64>();
            let record = TensorRecord {
                name: view.name.clone(),
                dtype: DType::Bf16,
                shape: view.source_shape,
                file: "model-00001.safetensors".into(),
                file_start: 0,
                bytes,
                placement: Placement::Replicated,
                scale: ScaleLayout::None,
            };
            (view.name, record)
        })
        .collect();
    Catalog {
        geometry: &model_qwen38_contract::geometry::FLASH,
        tensors,
        ranks: 4,
        precision: None,
    }
}

#[test]
fn a_complete_vision_tower_plans_checked_encoders() {
    let assets = EncoderAssets::planned(&vision_catalog(), ImagePositions::Sequential).unwrap();
    assert!(assets.checked_tower.is_some());
    assert_eq!(assets.positions, ImagePositions::Sequential);
    let component = component().unwrap();
    assert_eq!(component.id().as_str(), IMAGE_COMPONENT);
}

/// A missing tensor, a wrong shape or a wrong dtype refuses the encoders.
#[test]
fn a_wrong_or_incomplete_vision_tower_is_refused() {
    let bias = "model.visual.blocks.0.mlp.linear_fc1.bias";
    let refusal = |catalog: &Catalog| {
        EncoderAssets::planned(catalog, ImagePositions::Grid3Axis)
            .map(drop)
            .unwrap_err()
            .to_string()
    };

    let mut missing = vision_catalog();
    missing.tensors.remove(bias).unwrap();
    assert!(
        refusal(&missing).contains(&format!("checkpoint lacks {bias}")),
        "{}",
        refusal(&missing)
    );

    let mut shape = vision_catalog();
    shape.tensors.get_mut(bias).unwrap().shape.push(1);
    assert!(refusal(&shape).contains("is not BF16 of shape"));

    let mut dtype = vision_catalog();
    let tensor = dtype.tensors.get_mut(bias).unwrap();
    tensor.dtype = DType::F32;
    tensor.bytes *= 2;
    assert!(refusal(&dtype).contains("is not BF16 of shape"));
}
