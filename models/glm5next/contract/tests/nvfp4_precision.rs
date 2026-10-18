//! The precision component of the lowered projections over the NVFP4 mixes
//! (`prepared::document`), from synthetic configs: the NVFP4-routed mix keeps
//! every lowered projection BF16; the NVFP4 dense and routed mix stores the
//! dense MLP down projections (layers 0-2) in NVFP4 with their tensor global
//! (`weight_scale_2`), and their static input scales are ignored
//! (activations stay BF16).
use kernel_core::contract::WeightFormat;
use model_glm5next_contract::quantization::synthetic::config_of;
use model_glm5next_contract::quantization::{self, Layout, NVFP4_DENSE_ROUTED, NVFP4_ROUTED};
use model_glm5next_contract::{precision, prepared, sm120};

fn document(
    layout: &Layout,
    config: &[u8],
) -> kernel_core::precision::execution::PrecisionDocument {
    let config: serde_json::Value = serde_json::from_slice(config).unwrap();
    let metadata = quantization::export_metadata(config.get("quantization_config")).unwrap();
    let document = prepared::document(layout, &metadata, 0).unwrap();
    prepared::check(&document, layout).unwrap();
    document
}

#[test]
fn each_nvfp4_mix_derives_its_lowered_projections() {
    let routed = document(&NVFP4_ROUTED, &config_of(&NVFP4_ROUTED));
    let dense_routed = document(&NVFP4_DENSE_ROUTED, &config_of(&NVFP4_DENSE_ROUTED));
    assert_eq!(routed.operations.len(), dense_routed.operations.len());
    let formats = |document: &kernel_core::precision::execution::PrecisionDocument| {
        document
            .operations
            .iter()
            .filter(|s| s.effective.weights[0].format != WeightFormat::Bf16)
            .map(|s| s.source.id.clone())
            .collect::<Vec<_>>()
    };
    assert!(formats(&routed).is_empty());
    assert_eq!(
        formats(&dense_routed),
        ["dense.down.0.f32", "dense.down.1.f32", "dense.down.2.f32"]
    );
    for selection in dense_routed
        .operations
        .iter()
        .filter(|s| s.source.id.starts_with("dense.down."))
    {
        let [weight] = selection.effective.weights.as_slice() else {
            panic!("{}", selection.source.id);
        };
        let layer = &selection.source.id["dense.down.".len().."dense.down.".len() + 1];
        let stem = format!("model.language_model.layers.{layer}.mlp.down_proj");
        assert_eq!(weight.codes.tensor, format!("{stem}.weight"));
        assert_eq!(
            weight.local_scale.as_ref().unwrap().tensor.tensor,
            format!("{stem}.weight_scale")
        );
        assert_eq!(
            weight.global_scale.as_ref().unwrap().tensor.tensor,
            format!("{stem}.weight_scale_2")
        );
        // The rank's K share of the 12,288-wide intermediate.
        let geometry = selection.effective.geometry.unwrap();
        assert_eq!((geometry.rows, geometry.k), (4096, 3072));
    }
}

/// The shared tiled GEMM offers NVFP4 dense down projections only with
/// their tensor global.
#[test]
fn nvfp4_offers_carry_a_tensor_global() {
    let (usage, _) = precision::uses("dense.down").unwrap()[0];
    let operation = sm120::precision::domain("dense.down", WeightFormat::Nvfp4, usage).unwrap();
    let global = operation.weights[0].global_scale.as_ref().unwrap();
    assert_eq!(global.scope, kernel_core::precision::ScaleScope::Tensor);
    assert!(global.tensor.shape.is_empty());
    sm120::precision::offer("dense.down", WeightFormat::Nvfp4, usage).unwrap();
}

/// Every rank stores each NVFP4 mix's lowered projections as their offers
/// read them (`prepared::resident`): the NVFP4 dense down codes as the
/// rank's [4096, 1536] bytes, their UE4M3 scales [4096, 192] and the scalar
/// global each rank holds whole.
#[test]
fn every_rank_stores_the_nvfp4_mixes_lowered_projections() {
    use model_glm5next_contract::contract::Shape;
    use model_glm5next_contract::direct::{Stored, checkpoint_layout};
    use std::collections::BTreeMap;
    for layout in [&NVFP4_ROUTED, &NVFP4_DENSE_ROUTED] {
        let document = document(layout, &config_of(layout));
        for rank in 0..Shape::TP4.ranks() {
            let entries = checkpoint_layout(Shape::TP4, layout, rank).unwrap();
            let stored: BTreeMap<&str, &Stored> = entries
                .iter()
                .map(|entry| (entry.view.name.as_str(), entry))
                .collect();
            for selection in &document.operations {
                let resident = prepared::resident(selection, &stored)
                    .unwrap_or_else(|e| panic!("{} rank {rank}: {e}", layout.id));
                let nvfp4 = selection.effective.weights[0].format == WeightFormat::Nvfp4;
                assert_eq!(resident.global.is_some(), nvfp4, "{}", selection.source.id);
                if nvfp4 {
                    let bytes = |view: &Option<(String, u64)>| view.as_ref().unwrap().1;
                    assert_eq!(resident.codes.1, 4096 * 1536);
                    assert_eq!(bytes(&resident.scales), 4096 * 192);
                    assert_eq!(bytes(&resident.global), 4);
                }
            }
        }
    }
}
