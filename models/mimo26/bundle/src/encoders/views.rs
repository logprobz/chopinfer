//! Bind authenticated encoder views to device spans.
pub use model_mimo26_contract::encoder_index::*;
use runtime_bundle_abi::DeviceBufferSpanV1;
use std::collections::BTreeMap;

pub fn spans(
    index: &EncoderIndex,
    image: DeviceBufferSpanV1,
) -> Result<BTreeMap<String, DeviceBufferSpanV1>, String> {
    if image.length != index.image_bytes {
        return Err("encoder image allocation length mismatch".into());
    }
    index
        .views
        .iter()
        .map(|(name, view)| {
            let offset = image
                .offset
                .checked_add(view.offset)
                .ok_or("encoder span overflow")?;
            Ok((
                name.clone(),
                DeviceBufferSpanV1::new(image.buffer, offset, view.bytes),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn required_view_set_matches_mm9_catalog_count() {
        let views = model_mimo26_contract::encoders::expected_flash()
            .into_keys()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(views.len(), 848);
        assert!(views.contains("visual.patch_embed.proj.weight"));
        assert!(views.contains("encoder.quantizer.vq.layers.19._codebook.embed"));
    }
}
