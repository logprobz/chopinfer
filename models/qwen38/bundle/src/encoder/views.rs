//! Device spans over the shared encoder image contract.
pub use model_qwen38_contract::encoder::views::*;
use runtime_bundle_abi::DeviceBufferSpanV1;
use std::collections::BTreeMap;

/// The device span of every view inside the uploaded image.
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
