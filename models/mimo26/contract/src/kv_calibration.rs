//! Checkpoint-owned FP8 KV calibration identities.
use sha2::{Digest, Sha256};

/// Shipped MOPD/RL scales use `kernel_core::kv_fp8::calibrated_scale` with
/// 16 times the f64-oracle amax over 33 teacher texts, separately for K and V
/// in `program::GLOBAL_LAYERS` order. The observed amax was 14 to 28 times
/// the scale and 16 to 32 times below saturation. At scale 1.0 the V of
/// layers 0 and 5 (RMS 0.006 and 0.013) would be mostly subnormal.
/// The old .v1 layout stored unscaled codes.
pub const LEGACY_SCALES: [[f32; 2]; 9] = [
    [0.25, 0.00390625],
    [0.5, 0.0078125],
    [0.5, 0.0625],
    [0.5, 0.0625],
    [1.0, 0.125],
    [1.0, 0.5],
    [1.0, 0.25],
    [1.0, 1.0],
    [1.0, 2.0],
];
pub const LEGACY_LAYOUT: &str = "mimo.ga-paged-fp8.v2";
pub const DEFAULT_GEOMETRY: &str = "v26-flash";

pub fn geometry_id(explicit: Option<&str>) -> &str {
    explicit.unwrap_or(DEFAULT_GEOMETRY)
}

pub fn validate(scales: &[[f32; 2]], global_layers: usize) -> Result<(), String> {
    if scales.len() != global_layers {
        return Err(format!(
            "FP8 KV calibration layer count: expected {global_layers}, received {}",
            scales.len()
        ));
    }
    if scales.iter().flatten().any(|s| !s.is_finite() || *s <= 0.0) {
        return Err("FP8 KV calibration scales must be finite and positive".into());
    }
    Ok(())
}

/// Hash the kernel-facing f32 bits in global-layer K/V order, independently
/// of JSON formatting or checkpoint names. Keep the shipped table's ID.
pub fn layout_id(scales: &[[f32; 2]]) -> String {
    if scales == LEGACY_SCALES {
        return LEGACY_LAYOUT.into();
    }
    let mut digest = Sha256::new();
    for scale in scales.iter().flatten() {
        digest.update(scale.to_bits().to_le_bytes());
    }
    format!("{LEGACY_LAYOUT}.{:x}", digest.finalize())
}

#[derive(Debug, PartialEq)]
pub struct Calibration {
    pub geometry: String,
    pub layout_id: String,
    pub scales: Vec<[f32; 2]>,
}

/// Validate and deduplicate the build's registry against compiled geometries.
pub fn tables(
    checkpoints: &serde_json::Value,
    geometries: &std::collections::BTreeMap<String, usize>,
) -> Result<Vec<Calibration>, String> {
    let mut tables = std::collections::BTreeMap::new();
    for (name, identity) in checkpoints["checkpoints"]
        .as_object()
        .ok_or("checkpoint registry must contain checkpoints")?
    {
        let Some(value) = identity.get("fp8_kv_scales").filter(|v| !v.is_null()) else {
            continue;
        };
        let explicit = identity
            .get("fp8_kv_geometry")
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_str()
                    .ok_or("FP8 KV calibration geometry must be a string")
            })
            .transpose()?;
        let geometry = geometry_id(explicit);
        let layers = geometries.get(geometry).ok_or_else(|| {
            format!(
                "checkpoint {name}: FP8 KV calibration has unknown compiled geometry {geometry}"
            )
        })?;
        let scales: Vec<[f32; 2]> = serde_json::from_value(value.clone())
            .map_err(|e| format!("checkpoint {name}: invalid FP8 KV calibration: {e}"))?;
        validate(&scales, *layers).map_err(|e| format!("checkpoint {name}: {e}"))?;
        let id = layout_id(&scales);
        let key = (geometry.to_owned(), id.clone());
        if let Some(previous) = tables.get(&key) {
            if previous != &scales {
                return Err("FP8 KV calibration digest collision".into());
            }
        } else {
            tables.insert(key, scales);
        }
    }
    Ok(tables
        .into_iter()
        .map(|((geometry, layout_id), scales)| Calibration {
            geometry,
            layout_id,
            scales,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_tables_are_data_only_and_geometry_scoped() {
        let mut registry = serde_json::json!({"checkpoints": {
            "mopd": {"fp8_kv_scales": LEGACY_SCALES},
            "rl": {"fp8_kv_scales": LEGACY_SCALES}
        }});
        let mut geometries = std::collections::BTreeMap::from([(DEFAULT_GEOMETRY.into(), 9)]);
        let original = tables(&registry, &geometries).unwrap();
        assert_eq!(original.len(), 1);
        assert_eq!(original[0].layout_id, LEGACY_LAYOUT);
        let mut other = LEGACY_SCALES;
        other[0][0] = 0.125;
        registry["checkpoints"]["extra"] = serde_json::json!({"fp8_kv_scales": other});
        let added = tables(&registry, &geometries).unwrap();
        assert_eq!(added.len(), 2);
        assert!(added.contains(&original[0]));
        assert_eq!(
            added.iter().find(|v| v.scales == other).unwrap().layout_id,
            layout_id(&other)
        );
        assert_ne!(layout_id(&other), LEGACY_LAYOUT);
        registry["checkpoints"]["extra"]["fp8_kv_geometry"] = "small".into();
        assert!(
            tables(&registry, &geometries)
                .unwrap_err()
                .contains("unknown compiled geometry")
        );
        geometries.insert("small".into(), 1);
        assert!(
            tables(&registry, &geometries)
                .unwrap_err()
                .contains("layer count")
        );
        registry["checkpoints"]["extra"]["fp8_kv_scales"] = serde_json::json!([[0.125, 0.25]]);
        assert_eq!(tables(&registry, &geometries).unwrap().len(), 2);
        for scale in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            assert!(validate(&[[scale, 1.0]], 1).is_err());
        }
    }
}
