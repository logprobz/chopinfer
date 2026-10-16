use model_mimo26_contract::geometry::V26_FLASH;

#[test]
fn mtp_and_absent_layers_use_sliding_attention() {
    for layer in V26_FLASH.layers()..V26_FLASH.layers() + V26_FLASH.mtp_layers() {
        assert!(!V26_FLASH.is_global(layer));
    }
    assert!(!V26_FLASH.is_global(V26_FLASH.layers() + V26_FLASH.mtp_layers()));
    assert!(!V26_FLASH.is_global(u32::MAX));
}

const MTP_IS_GLOBAL: bool = V26_FLASH.is_global(V26_FLASH.layers());
const GLOBAL_COUNT: usize = V26_FLASH.global_layers();
const SLIDING_COUNT: usize = V26_FLASH.sliding_layers();
const LAST_GLOBAL_ORDINAL: usize = V26_FLASH.global_ordinal(47);
const TAPS: usize = V26_FLASH.tap_count();

#[test]
fn typed_patterns_and_views_match_the_authenticated_document() {
    use model_mimo26_contract::geometry::{FeedForwardKind, LayerKind, VisionLayerKind};
    let config: serde_json::Value = serde_json::from_str(V26_FLASH.document).unwrap();
    assert_eq!((MTP_IS_GLOBAL, GLOBAL_COUNT), (false, 9));
    assert_eq!(GLOBAL_COUNT + SLIDING_COUNT, V26_FLASH.layers() as usize);
    assert_eq!(LAST_GLOBAL_ORDINAL, 8);
    assert_eq!(TAPS, 5);
    assert_eq!(V26_FLASH.global_ordinal(u32::MAX), 0);
    assert_eq!(V26_FLASH.swa_head_dim(), 192);
    assert_eq!(V26_FLASH.swa_value_head_dim(), 128);
    for (i, layer) in V26_FLASH.shape.text.layers.iter().enumerate() {
        assert_eq!(
            matches!(layer.attention, LayerKind::Global),
            config["hybrid_layer_pattern"][i] == 0
        );
        assert_eq!(
            matches!(layer.feed_forward, FeedForwardKind::Moe),
            config["moe_layer_freq"][i] == 1
        );
    }
    let vision = V26_FLASH.vision();
    assert_eq!(vision.depth as usize, vision.vit_window_attn_types.len());
    for (i, kind) in vision.vit_window_attn_types.iter().enumerate() {
        let expected = match config["vision_config"]["vit_window_attn_types"][i]
            .as_i64()
            .unwrap()
        {
            -1 => VisionLayerKind::Full,
            0 => VisionLayerKind::Window,
            1 => VisionLayerKind::MergeGroupWindow,
            _ => unreachable!(),
        };
        assert_eq!(*kind, expected);
    }
    let draft = V26_FLASH.dflash();
    assert_eq!(draft.hidden_size, V26_FLASH.hidden());
    assert_eq!(draft.target_hidden_size, V26_FLASH.hidden());
    assert_eq!(draft.num_hidden_layers as usize, draft.layer_types.len());
    assert!(
        draft
            .layer_types
            .iter()
            .all(|kind| *kind == LayerKind::Sliding)
    );
    assert_eq!(
        draft.block_size as u64,
        config["dflash"]["dflash_config"]["block_size"]
            .as_u64()
            .unwrap()
    );
}
