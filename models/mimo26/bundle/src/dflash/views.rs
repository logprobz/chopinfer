//! Bind validated drafter views to device spans.
use crate::index::RankIndex;
use model_mimo26_contract::Tp;
pub use model_mimo26_contract::dflash::*;
pub(crate) use model_mimo26_contract::drafters::fields;
use runtime_bundle_abi::DeviceBufferSpanV1;
use std::collections::BTreeMap;

/// The drafter's spans inside its uploaded image, plus the target embedding
/// and vocabulary head it shares.
pub fn spans(
    index: &RankIndex,
    image: DeviceBufferSpanV1,
    target: &BTreeMap<String, DeviceBufferSpanV1>,
    tp: Tp,
) -> Result<BTreeMap<String, DeviceBufferSpanV1>, &'static str> {
    validate(index, tp)?;
    if image.length != index.image_bytes {
        return Err("DFlash image span length mismatch");
    }
    let mut result = BTreeMap::new();
    for key in ["model.embed_tokens.weight", "lm_head.weight"] {
        result.insert(
            key.into(),
            *target
                .get(key)
                .ok_or("DFlash target shared weight missing")?,
        );
    }
    for (name, _, _) in fields(tp) {
        let view = &index.views[&name];
        let offset = image
            .offset
            .checked_add(view.offset)
            .ok_or("DFlash span offset overflow")?;
        result.insert(
            name,
            DeviceBufferSpanV1::new(image.buffer, offset, view.bytes),
        );
    }
    Ok(result)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::index::View;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    pub(crate) fn index(rank: u32) -> RankIndex {
        index_at(rank, Tp::TP4)
    }

    pub(crate) fn index_at(rank: u32, tp: Tp) -> RankIndex {
        let mut views = BTreeMap::new();
        let mut offset = 0;
        for (name, shape, layout) in fields(tp) {
            let bytes = shape.iter().product::<u64>() * 2;
            views.insert(
                name,
                View {
                    offset,
                    bytes,
                    dtype: "BF16".into(),
                    shape: shape.to_vec(),
                    layout_id: layout.into(),
                },
            );
            offset += bytes.next_multiple_of(4096);
        }
        RankIndex {
            rank,
            image: format!("rank{rank}.img"),
            image_bytes: offset,
            sha256: Some("a".repeat(64)),
            direct_identity: None,
            views,
        }
    }

    #[test]
    fn spans_share_target_ends_and_keep_image_offsets() {
        let index = index(1);
        assert!(is_dflash(&index));
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let image = DeviceBufferSpanV1::new(buffer, 8192, index.image_bytes);
        let target = BTreeMap::from([
            (
                "model.embed_tokens.weight".into(),
                DeviceBufferSpanV1::new(buffer, 0, 4096),
            ),
            (
                "lm_head.weight".into(),
                DeviceBufferSpanV1::new(buffer, 4096, 4096),
            ),
        ]);
        let spans = spans(&index, image, &target, Tp::TP4).unwrap();
        assert_eq!(spans.len(), 52);
        let view = &index.views["layers.3.self_attn.qkv_proj.weight"];
        let span = spans["layers.3.self_attn.qkv_proj.weight"];
        assert_eq!(
            (span.offset, span.length),
            (8192 + view.offset, 2560 * 4096 * 2)
        );
        assert_eq!(spans["lm_head.weight"], target["lm_head.weight"]);
        let short = DeviceBufferSpanV1::new(buffer, 0, index.image_bytes - 4096);
        assert!(super::spans(&index, short, &target, Tp::TP4).is_err());
    }

    #[test]
    fn malformed_drafter_images_fail_closed() {
        let good = index(0);
        assert!(validate(&good, Tp::TP4).is_ok());
        let mut bad = good.clone();
        bad.views.remove("layers.4.mlp.down_proj.weight");
        assert!(validate(&bad, Tp::TP4).is_err());
        let mut bad = good.clone();
        bad.views.get_mut("fc.weight").unwrap().shape = vec![5120, 4096];
        assert!(validate(&bad, Tp::TP4).is_err());
        let mut bad = good.clone();
        bad.views.get_mut("norm.weight").unwrap().layout_id = "bf16_mma_fragments_16x16_v1".into();
        assert!(validate(&bad, Tp::TP4).is_err());
        let mut bad = good.clone();
        bad.views
            .get_mut("layers.0.self_attn.q_norm.weight")
            .unwrap()
            .offset = 0;
        assert!(validate(&bad, Tp::TP4).is_err());
        let mut bad = good;
        bad.views.remove("mask_embedding.weight");
        assert!(!is_dflash(&bad));
    }

    #[test]
    fn each_topology_validates_only_its_own_shapes() {
        for tp in Tp::SUPPORTED {
            let index = index_at(tp.ranks() - 1, tp);
            assert!(validate(&index, tp).is_ok());
            let qkv = &index.views["layers.0.self_attn.qkv_proj.weight"];
            assert_eq!(qkv.shape, [tp.dflash_qkv_rows(), 4096]);
            assert_eq!(index.views["fc.weight"].shape, [4096, tp.dflash_fc_k()]);
            for other in Tp::SUPPORTED.into_iter().filter(|&other| other != tp) {
                assert!(validate(&index_at(0, other), tp).is_err());
            }
        }
        assert!(validate(&index_at(2, Tp::TP4), Tp::TP2).is_err());
    }
}
