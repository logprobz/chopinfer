//! Named projection views from the MiMo per-rank materialization index.

use std::collections::BTreeMap;

use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::index::RankIndex;

fn exact(
    views: &BTreeMap<String, DeviceBufferSpanV1>,
    name: &str,
    bytes: u64,
) -> Result<(), &'static str> {
    let Some(view) = views.get(name) else {
        return Err("MiMo rank image is missing a required view");
    };
    if view.length != bytes {
        return Err("MiMo rank image has an incorrect view length");
    }
    Ok(())
}

/// Check the full view set before the bundle instance becomes visible:
/// every view of `index`, the rank index `validate_text` admitted against
/// the text contract, at its exact length.
pub fn validate_complete(
    views: &BTreeMap<String, DeviceBufferSpanV1>,
    index: &RankIndex,
) -> Result<(), &'static str> {
    for (name, view) in &index.views {
        exact(views, name, view.bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::*;
    use crate::topology::Tp;

    /// The declared contract carries the engine layout of layer 0's FP8
    /// MLP (128 x 128 F32 block scales) and of every MoE layer's MXFP4
    /// experts (one scale byte per 32 K elements), and a short view refuses.
    #[test]
    fn declared_mlp_views_match_the_engine_layout() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        for tp in Tp::SUPPORTED {
            let contracts = model_mimo26_contract::text_views(tp);
            let bytes = |name: &str| contracts[name].bytes;
            let (hidden, dense, expert, experts) = (
                u64::from(tp.geometry.hidden()),
                tp.dense_intermediate(),
                tp.expert_intermediate(),
                u64::from(tp.geometry.experts()),
            );
            let stem = "model.layers.0.mlp";
            assert_eq!(
                bytes(&format!("{stem}.gate_up_proj.weight")),
                2 * dense * hidden
            );
            assert_eq!(
                bytes(&format!("{stem}.gate_up_proj.weight_scale_inv")),
                2 * dense / 128 * (hidden / 128) * 4
            );
            assert_eq!(bytes(&format!("{stem}.down_proj.weight")), hidden * dense);
            assert_eq!(
                bytes(&format!("{stem}.down_proj.weight_scale_inv")),
                hidden / 128 * (dense / 128) * 4
            );
            for layer in 1..tp.geometry.layers() {
                let stem = format!("model.layers.{layer}.mlp.experts");
                for (suffix, expected) in [
                    ("gate_up_proj.weight", 2 * expert * hidden / 2),
                    ("gate_up_proj.weight_scale", 2 * expert * hidden / 32),
                    ("down_proj.weight", hidden * expert / 2),
                    ("down_proj.weight_scale", hidden * expert / 32),
                ] {
                    assert_eq!(bytes(&format!("{stem}.{suffix}")), experts * expected);
                }
            }
            let mut offset = 0;
            let index = RankIndex {
                rank: 0,
                image: "rank0.img".into(),
                image_bytes: 0,
                sha256: Some("a".repeat(64)),
                direct_identity: None,
                views: contracts
                    .iter()
                    .map(|(name, c)| {
                        let view = crate::index::View {
                            offset,
                            bytes: c.bytes,
                            dtype: c.dtype.into(),
                            shape: c.shape.clone(),
                            layout_id: c.layout_id.into(),
                        };
                        offset += c.bytes.next_multiple_of(4096);
                        (name.clone(), view)
                    })
                    .collect(),
            };
            let index = RankIndex {
                image_bytes: offset,
                ..index
            };
            crate::index::validate_text(&index, tp).unwrap();
            let mut views: BTreeMap<_, _> = contracts
                .iter()
                .map(|(name, c)| (name.clone(), DeviceBufferSpanV1::new(buffer, 0, c.bytes)))
                .collect();
            assert!(validate_complete(&views, &index).is_ok());
            let name = "model.layers.1.mlp.experts.down_proj.weight";
            views.get_mut(name).unwrap().length -= 1;
            assert!(validate_complete(&views, &index).is_err());
            views.remove(name);
            assert!(validate_complete(&views, &index).is_err());
        }
    }
}
