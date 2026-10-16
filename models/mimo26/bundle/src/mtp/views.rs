//! Bind validated drafter views to device spans.
use crate::index::RankIndex;
use model_mimo26_contract::Tp;
pub(crate) use model_mimo26_contract::drafters::mtp_fields;
pub use model_mimo26_contract::mtp::*;
use runtime_bundle_abi::DeviceBufferSpanV1;
use std::collections::BTreeMap;

pub fn aliases(
    index: &RankIndex,
    image: DeviceBufferSpanV1,
    target: &BTreeMap<String, DeviceBufferSpanV1>,
    tp: Tp,
) -> Result<BTreeMap<String, DeviceBufferSpanV1>, &'static str> {
    validate(index, tp)?;
    if image.length != index.image_bytes {
        return Err("MTP image span length mismatch");
    }
    let mut result = BTreeMap::new();
    for key in ["model.embed_tokens.weight", "lm_head.weight"] {
        result.insert(
            key.into(),
            *target.get(key).ok_or("MTP target shared weight missing")?,
        );
    }
    for (suffix, _, _, _) in mtp_fields(tp) {
        let view = &index.views[&format!("model.mtp.layers.0.{suffix}")];
        let offset = image
            .offset
            .checked_add(view.offset)
            .ok_or("MTP span offset overflow")?;
        let span = DeviceBufferSpanV1::new(image.buffer, offset, view.bytes);
        result.insert(format!("model.mtp.layers.0.{suffix}"), span);
        let alias = if suffix.starts_with("self_attn.") {
            Some(format!("model.layers.1.{suffix}"))
        } else if suffix.starts_with("mlp.") {
            Some(format!("model.layers.0.{suffix}"))
        } else {
            match suffix {
                "input_layernorm.weight" => {
                    Some("model.layers.0.post_attention_layernorm.weight".into())
                }
                "pre_mlp_layernorm.weight" => {
                    Some("model.layers.1.post_attention_layernorm.weight".into())
                }
                "final_layernorm.weight" => Some("model.layers.1.input_layernorm.weight".into()),
                _ => None,
            }
        };
        if let Some(alias) = alias {
            result.insert(alias, span);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::View;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    fn index() -> RankIndex {
        index_at(Tp::TP4)
    }

    fn index_at(tp: Tp) -> RankIndex {
        let mut views = BTreeMap::new();
        let mut offset = 0;
        for layer in 0..3 {
            for (suffix, shape, dtype, layout) in mtp_fields(tp) {
                let bytes = shape.iter().product::<u64>()
                    * match dtype {
                        "F32" => 4,
                        "BF16" => 2,
                        _ => 1,
                    };
                views.insert(
                    format!("model.mtp.layers.{layer}.{suffix}"),
                    View {
                        offset,
                        bytes,
                        dtype: dtype.into(),
                        shape: shape.to_vec(),
                        layout_id: layout.into(),
                    },
                );
                offset += bytes.next_multiple_of(4096);
            }
        }
        RankIndex {
            rank: 0,
            image: "rank0.img".into(),
            image_bytes: offset,
            sha256: Some("a".repeat(64)),
            direct_identity: None,
            views,
        }
    }

    #[test]
    fn aliases_use_only_layer_zero_and_preserve_image_offsets() {
        let index = index();
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let image = DeviceBufferSpanV1::new(buffer, 4096, index.image_bytes);
        let target = BTreeMap::from([
            (
                "model.embed_tokens.weight".into(),
                DeviceBufferSpanV1::new(buffer, 0, 8192),
            ),
            (
                "lm_head.weight".into(),
                DeviceBufferSpanV1::new(buffer, 8192, 8192),
            ),
        ]);
        let mapped = aliases(&index, image, &target, Tp::TP4).unwrap();
        for (alias, original) in [
            (
                "model.layers.1.self_attn.qkv_proj.weight",
                "self_attn.qkv_proj.weight",
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "input_layernorm.weight",
            ),
            (
                "model.layers.1.post_attention_layernorm.weight",
                "pre_mlp_layernorm.weight",
            ),
            (
                "model.layers.1.input_layernorm.weight",
                "final_layernorm.weight",
            ),
        ] {
            let view = &index.views[&format!("model.mtp.layers.0.{original}")];
            assert_eq!(mapped[alias].offset, image.offset + view.offset);
            assert_eq!(mapped[alias].length, view.bytes);
        }
        assert_eq!(mapped["lm_head.weight"], target["lm_head.weight"]);
    }

    #[test]
    fn malformed_drafter_images_fail_closed() {
        let good = index();
        assert!(validate(&good, Tp::TP4).is_ok());
        let mut bad = good.clone();
        bad.views.remove("model.mtp.layers.0.hnorm.weight");
        assert!(validate(&bad, Tp::TP4).is_err());
        let mut bad = good.clone();
        bad.views
            .get_mut("model.mtp.layers.0.eh_proj.weight")
            .unwrap()
            .layout_id = "bf16_row_major_v1".into();
        assert!(validate(&bad, Tp::TP4).is_err());
        let mut bad = good.clone();
        bad.views
            .get_mut("model.mtp.layers.0.hnorm.weight")
            .unwrap()
            .offset = 0;
        assert!(validate(&bad, Tp::TP4).is_err());
        let mut bad = good;
        bad.views
            .get_mut("model.mtp.layers.0.eh_proj.weight")
            .unwrap()
            .shape = vec![2048, 4096];
        assert!(validate(&bad, Tp::TP4).is_err());
    }

    #[test]
    fn each_topology_validates_only_its_own_shapes() {
        for tp in Tp::SUPPORTED {
            let index = index_at(tp);
            assert!(validate(&index, tp).is_ok());
            let eh = &index.views["model.mtp.layers.2.eh_proj.weight"];
            assert_eq!(eh.shape, [4096, tp.mtp_eh_k()]);
            let scale = &index.views["model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv"];
            assert_eq!(scale.layout_id, tp.qkv_scale_layout());
            for other in Tp::SUPPORTED.into_iter().filter(|&other| other != tp) {
                assert!(validate(&index_at(other), tp).is_err());
            }
        }
    }
}
