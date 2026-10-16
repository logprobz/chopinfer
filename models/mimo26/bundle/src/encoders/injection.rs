//! Validate V3 media records and map prefill rows to arena rows.

#[cfg(test)]
use model_mimo26_contract::geometry::V26_FLASH;

use runtime_bundle_sdk::BatchMediaSectionV3;

use bundle_runtime::media::MediaError;
pub use bundle_runtime::media::MediaStepPlan;

use crate::inputs::PhaseInputs;

pub fn plan_for(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
    media: &BatchMediaSectionV3,
    prefill: &PhaseInputs,
    item_count: usize,
) -> Result<MediaStepPlan, &'static str> {
    for record in &media.encodes {
        validate_record_for(geometry, record)?;
    }
    super::arena::for_geometry(geometry)?
        .plan(media, prefill, item_count)
        .map_err(|error| match error {
            MediaError::Encode | MediaError::Pages => "invalid MiMo media encode record",
            MediaError::Inject => "invalid MiMo media inject record",
            MediaError::NotPrefill => "media injection names a non-prefill item",
            MediaError::OutsideItem => "media injection exceeds its item rows",
            MediaError::Overlap => "media injections overlap",
            MediaError::StrayPayload => "media payload has no encode record",
        })
}

#[cfg(test)]
mod tests {
    use runtime_bundle_sdk::{BatchMediaEncodeV3, BatchMediaInjectV3};

    use super::super::arena::PAYLOAD_LIMITS;

    use super::*;

    #[test]
    fn injects_only_own_prefill_rows_in_fixed_order() {
        let prefill = PhaseInputs {
            item_order: vec![0, 2],
            q_start: vec![0, 4],
            q_len: vec![4, 2],
            token_ids: vec![0; 6],
            ..PhaseInputs::default()
        };
        let mut media = BatchMediaSectionV3 {
            injects: vec![BatchMediaInjectV3 {
                item_index: 0,
                item_row: 1,
                rows: 2,
                arena_row: 64,
            }],
            ..BatchMediaSectionV3::default()
        };
        assert_eq!(
            plan(&media, &prefill, 3).unwrap().source_rows,
            [-1, 64, 65, -1, -1, -1]
        );
        media.injects.push(BatchMediaInjectV3 {
            item_index: 0,
            item_row: 2,
            rows: 1,
            arena_row: 66,
        });
        assert!(plan(&media, &prefill, 3).is_err());
        media.injects[1].item_index = 1;
        assert!(plan(&media, &prefill, 3).is_err());
    }

    #[test]
    fn encode_payload_limits_follow_the_selected_encoder() {
        let bytes = PAYLOAD_LIMITS[0];
        let mut media = BatchMediaSectionV3 {
            encodes: vec![BatchMediaEncodeV3 {
                encoder: 0,
                geometry: [16, 32, 32, 0],
                output_rows: 4096,
                pages: (0..64).collect(),
                payload_offset: 0,
                payload_bytes: bytes,
            }],
            payload: vec![0; bytes as usize],
            ..BatchMediaSectionV3::default()
        };
        let prefill = PhaseInputs::default();
        assert!(plan(&media, &prefill, 0).is_ok());
        media.encodes[0].encoder = 1;
        assert!(plan(&media, &prefill, 0).is_err());
        media.encodes[0].encoder = 2;
        assert!(plan(&media, &prefill, 0).is_err());
    }

    #[test]
    fn encode_pages_are_exclusive_but_injections_can_share() {
        let record = |pages: Vec<u64>, rows, offset| BatchMediaEncodeV3 {
            encoder: 0,
            geometry: [1, 2, rows * 2, 0],
            output_rows: rows,
            pages,
            payload_offset: offset,
            payload_bytes: u64::from(rows) * 24576,
        };
        let mut media = BatchMediaSectionV3 {
            encodes: vec![record(vec![7, 8], 65, 0), record(vec![9], 1, 65 * 24576)],
            payload: vec![0; 66 * 24576],
            ..BatchMediaSectionV3::default()
        };
        let prefill = PhaseInputs {
            item_order: vec![0, 1],
            q_start: vec![0, 1],
            q_len: vec![1, 1],
            token_ids: vec![0; 2],
            ..PhaseInputs::default()
        };
        media.injects = (0..2)
            .map(|item_index| BatchMediaInjectV3 {
                item_index,
                item_row: 0,
                rows: 1,
                arena_row: 7 * 64,
            })
            .collect();
        assert_eq!(plan(&media, &prefill, 2).unwrap().source_rows, [448, 448]);
        for page in [7, 8] {
            media.encodes[1].pages = vec![page];
            assert!(plan(&media, &prefill, 2).is_err(), "page {page} aliases");
        }
    }

    #[test]
    fn rejects_every_encoder_geometry_before_planning_injection() {
        let good = BatchMediaEncodeV3 {
            encoder: 0,
            geometry: [1, 2, 2, 0],
            output_rows: 1,
            pages: vec![0],
            payload_offset: 0,
            payload_bytes: 24576,
        };
        let malformed = [
            (0, [1, 3, 2, 0], 1, 36864),
            (0, [1, 2, 2, 0], 2, 24576),
            (0, [1, 2, 2, 0], 1, 24577),
            (1, [2, 80, 0, 0], 1, 640),
            (1, [60002, 128, 0, 0], 3751, 30721024),
        ];
        for (encoder, geometry, output_rows, payload_bytes) in malformed {
            let bad = BatchMediaEncodeV3 {
                encoder,
                geometry,
                output_rows,
                payload_bytes,
                pages: (1..=u64::from(output_rows).div_ceil(64)).collect(),
                payload_offset: 24576,
            };
            let media = BatchMediaSectionV3 {
                encodes: vec![good.clone(), bad],
                payload: vec![0; 24576 + payload_bytes as usize],
                ..Default::default()
            };
            assert!(
                plan(&media, &PhaseInputs::default(), 0).is_err(),
                "{geometry:?}"
            );
        }
    }
}

/// Validate the wire record against the declaration's recipes and row trace.
pub fn validate_record_for(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
    record: &runtime_bundle_sdk::BatchMediaEncodeV3,
) -> Result<(), &'static str> {
    let declaration = model_mimo26_contract::encoders::declarations(geometry)
        .map_err(|_| "MiMo encoder declarations are invalid")?
        .get(record.encoder as usize)
        .ok_or("unknown MiMo encoder")?;
    let item = declaration
        .admit_record(record.geometry)
        .map_err(|_| "MiMo media record outside its declaration's recipes")?;
    let bytes = item
        .rows
        .checked_mul(u64::from(declaration.input.columns))
        .and_then(|bytes| bytes.checked_mul(declaration.input.payload.bytes()))
        .ok_or("encoder payload geometry overflow")?;
    let trace = declaration
        .trace(item)
        .map_err(|_| "invalid encoder row geometry")?;
    let rows = trace.last().ok_or("empty encoder trace")?.rows;
    if bytes != record.payload_bytes || rows != u64::from(record.output_rows) {
        return Err("encoder payload or output geometry mismatch");
    }
    Ok(())
}

#[cfg(test)]
mod record_tests {
    use super::*;
    /// Unused geometry fields must be zero: the record is refused as a
    /// whole, not read as its leading fields.
    #[test]
    fn records_with_unused_geometry_fields_are_refused() {
        let record = |encoder, geometry: [u32; 4], output_rows, payload_bytes| {
            runtime_bundle_sdk::BatchMediaEncodeV3 {
                encoder,
                geometry,
                output_rows,
                pages: vec![0],
                payload_offset: 0,
                payload_bytes,
            }
        };
        let vision = |rest| record(0, [1, 34, 32, rest], 272, 1088 * 1536 * 4);
        let audio = |c, rest| record(1, [1595, 128, c, rest], 100, 1595 * 128 * 4);
        assert!(validate_record(&vision(0)).is_ok());
        assert!(validate_record(&audio(0, 0)).is_ok());
        for bad in [vision(1), audio(1, 0), audio(0, 1)] {
            assert!(validate_record(&bad).is_err(), "{:?}", bad.geometry);
        }
    }
}

#[cfg(test)]
pub fn plan(
    media: &BatchMediaSectionV3,
    prefill: &PhaseInputs,
    item_count: usize,
) -> Result<MediaStepPlan, &'static str> {
    plan_for(&V26_FLASH, media, prefill, item_count)
}
#[cfg(test)]
pub fn validate_record(
    record: &runtime_bundle_sdk::BatchMediaEncodeV3,
) -> Result<(), &'static str> {
    validate_record_for(&V26_FLASH, record)
}
