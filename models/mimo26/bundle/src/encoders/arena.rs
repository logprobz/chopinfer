//! The 32,768-row media arena, 8,192-row encoder runs and the payload
//! limits; the encoders' regions are their buffer plan's
//! (`super::runtime`).

#[cfg(test)]
use model_mimo26_contract::geometry::V26_FLASH;

pub const ROWS: u64 = model_mimo26_contract::encoders::ARENA_ROWS;
// Policy: media pages follow frontend.json multimodal.arena.page_rows.
pub const PAGE_ROWS: u64 = 64;
#[cfg(test)]
pub const HIDDEN: u64 = V26_FLASH.hidden() as u64;
pub const ELEMENT_BYTES: u64 = 2;
#[cfg(test)]
pub const ARENA_BYTES: u64 = ROWS * HIDDEN * ELEMENT_BYTES;
pub const CHUNK_ROWS: u64 = model_mimo26_contract::encoders::RUN_ROWS;
#[cfg(test)]
pub const PAYLOAD_LIMITS: [u64; 2] = [100_663_296, 30_720_512];
#[cfg(test)]
pub const MAX_PAYLOAD_BYTES: u64 = PAYLOAD_LIMITS[0];
/// The arena as the shared media validation sees it.
#[cfg(test)]
pub const ARENA: bundle_runtime::media::MediaArena = bundle_runtime::media::MediaArena {
    rows: ROWS,
    page_rows: PAGE_ROWS,
    payload_limits: &PAYLOAD_LIMITS,
};

pub fn for_geometry(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
) -> Result<bundle_runtime::media::MediaArena, &'static str> {
    type Limits = Vec<(&'static str, Result<[u64; 2], encoder_spec::Error>)>;
    static LIMITS: std::sync::OnceLock<Limits> = std::sync::OnceLock::new();
    let limits = LIMITS.get_or_init(|| {
        model_mimo26_contract::geometry::GEOMETRIES
            .iter()
            .map(|g| (g.id, model_mimo26_contract::encoders::payload_limits(g)))
            .collect()
    });
    let (_, payload_limits) = limits
        .iter()
        .find(|(id, _)| *id == geometry.id)
        .ok_or("unlisted encoder geometry")?;
    let payload_limits = payload_limits
        .as_ref()
        .map_err(|_| "invalid encoder payload recipes")?;
    Ok(bundle_runtime::media::MediaArena {
        rows: ROWS,
        page_rows: PAGE_ROWS,
        payload_limits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_and_staging_cover_declared_geometry() {
        assert_eq!(
            PAYLOAD_LIMITS,
            model_mimo26_contract::encoders::payload_limits(&V26_FLASH).unwrap()
        );
        let plan =
            super::super::runtime::plan(model_mimo26_contract::encoders::Encoders::ALL).unwrap();
        let bytes = |name| plan.region(name).unwrap().bytes;
        assert_eq!(plan.regions.len(), 17);
        assert_eq!(bytes("staging"), 64 << 20);
        assert_eq!(bytes("arena"), 256 << 20);
        let descriptor: serde_json::Value =
            serde_json::from_str(include_str!("../../../compiler/src/frontend.json")).unwrap();
        let maximum_payload = descriptor["multimodal"]["encoders"]
            .as_array()
            .unwrap()
            .iter()
            .map(|encoder| encoder["maximum_payload_bytes"].as_u64().unwrap())
            .max()
            .unwrap();
        assert!(bytes("payload") >= maximum_payload);
        for (encoder, limit) in descriptor["multimodal"]["encoders"]
            .as_array()
            .unwrap()
            .iter()
            .zip(PAYLOAD_LIMITS)
        {
            assert_eq!(encoder["maximum_payload_bytes"].as_u64(), Some(limit));
        }
        let media = &descriptor["multimodal"];
        for modality in media["modalities"].as_array().unwrap() {
            let p = &modality["recipe"]["parameters"];
            let payload = match modality["kind"].as_str().unwrap() {
                "video" => {
                    p["maximum_frames"]
                        .as_u64()
                        .unwrap()
                        .div_ceil(p["image"]["temporal_patch"].as_u64().unwrap())
                        * p["maximum_pixels_per_frame"].as_u64().unwrap()
                        / p["image"]["patch"].as_u64().unwrap().pow(2)
                        * 1536
                        * 4
                }
                _ => continue,
            };
            let encoder = modality["encoder"].as_u64().unwrap() as usize;
            assert!(PAYLOAD_LIMITS[encoder] >= payload);
            assert!(bytes("payload") >= payload);
        }
        assert_eq!(bytes("source-rows"), CHUNK_ROWS * 8);
        assert!(plan.regions.iter().all(|region| region.bytes > 0));
    }
    #[test]
    fn arena_chunk_writes_follow_noncontiguous_pages() {
        assert_eq!(
            ARENA.runs(&[9, 2, 7], 150, 60, 80).unwrap(),
            [
                bundle_runtime::media::ArenaRun {
                    source_row: 0,
                    arena_row: 636,
                    rows: 4
                },
                bundle_runtime::media::ArenaRun {
                    source_row: 4,
                    arena_row: 128,
                    rows: 64
                },
                bundle_runtime::media::ArenaRun {
                    source_row: 68,
                    arena_row: 448,
                    rows: 12
                },
            ]
        );
        assert!(ARENA.runs(&[1, 1], 100, 0, 100).is_err());
        assert!(ARENA.runs(&[1, 2], 100, 99, 2).is_err());
    }
}
