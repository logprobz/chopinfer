//! Validate the DFlash rank image and resolve its per-rank spans.

use crate::geometry::V26_FLASH as GEOMETRY;

use crate::index::RankIndex;

pub const LAYERS: u32 = GEOMETRY.dflash().num_hidden_layers;

use crate::Tp;
pub(crate) use crate::drafters::{fields, model_fields};
/// Whether a drafter rank index holds the DFlash drafter rather than MTP.
pub fn is_dflash(index: &RankIndex) -> bool {
    index.views.contains_key("mask_embedding.weight")
}

/// Refuses a drafter `config.json` other than the supported DFlash profile.
/// The file may carry trailing commas, which are removed before parsing.
pub fn validate_config(bytes: &[u8]) -> Result<&'static crate::geometry::Geometry, String> {
    let value: serde_json::Value = serde_json::from_slice(&without_trailing_commas(bytes))
        .map_err(|error| format!("DFlash config: {error}"))?;
    crate::geometry::match_dflash(&value)
}

pub fn validate_config_for(
    geometry: &'static crate::geometry::Geometry,
    bytes: &[u8],
) -> Result<&'static crate::geometry::Geometry, String> {
    let value: serde_json::Value = serde_json::from_slice(&without_trailing_commas(bytes))
        .map_err(|error| format!("DFlash config: {error}"))?;
    crate::geometry::match_dflash_for(geometry, &value)
}

fn without_trailing_commas(bytes: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(bytes.len());
    let mut quoted = false;
    let mut escaped = false;
    for (index, &byte) in bytes.iter().enumerate() {
        if quoted {
            result.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else if byte == b'"' {
            quoted = true;
            result.push(byte);
        } else if byte == b','
            && bytes[index + 1..]
                .iter()
                .find(|next| !next.is_ascii_whitespace())
                .is_some_and(|next| matches!(*next, b'}' | b']'))
        {
            continue;
        } else {
            result.push(byte);
        }
    }
    result
}

pub fn validate(index: &RankIndex, tp: Tp) -> Result<(), &'static str> {
    if index.rank >= tp.ranks()
        || index.views.len()
            != model_fields(tp).len() + tp.geometry.dflash().num_hidden_layers as usize * 9
    {
        return Err("DFlash image must contain exactly the drafter tensors");
    }
    let mut extents = Vec::new();
    for (name, shape, layout) in fields(tp) {
        let view = index.views.get(&name).ok_or("DFlash tensor missing")?;
        let bytes = shape.iter().product::<u64>() * 2;
        if view.shape != shape
            || view.dtype != "BF16"
            || view.layout_id != layout
            || view.bytes != bytes
        {
            return Err("DFlash tensor geometry or layout mismatch");
        }
        let end = view
            .offset
            .checked_add(bytes)
            .filter(|&end| end <= index.image_bytes)
            .ok_or("DFlash tensor exceeds image")?;
        extents.push((view.offset, end));
    }
    extents.sort_unstable();
    if extents.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err("DFlash tensor extents overlap");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admit_generated_drafter_and_refuse_geometry_drift() {
        let config: serde_json::Value = serde_json::from_str(GEOMETRY.document).unwrap();
        let original = config["dflash"].clone();
        validate_config(&serde_json::to_vec(&original).unwrap()).unwrap();
        for (pointer, value) in [
            ("/rope_theta", serde_json::json!(12345)),
            ("/layer_types/0", serde_json::json!("full_attention")),
            ("/dflash_config/target_layer_ids/0", serde_json::json!(48)),
        ] {
            let mut changed = original.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            let error = validate_config(&serde_json::to_vec(&changed).unwrap()).unwrap_err();
            assert!(
                error.contains("expected") && error.contains("received"),
                "{error}"
            );
        }
    }

    #[test]
    fn only_commas_outside_strings_before_closers_are_removed() {
        let input = br#"{"text":"comma, }", "values":[1,2, ], }"#;
        let value: serde_json::Value =
            serde_json::from_slice(&without_trailing_commas(input)).unwrap();
        assert_eq!(value["text"], "comma, }");
        assert_eq!(value["values"], serde_json::json!([1, 2]));
    }
}
