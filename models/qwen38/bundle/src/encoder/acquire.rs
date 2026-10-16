//! Acquisition of the encoder image and the shared encoder runtime: the
//! image's views on every rank, and the vision declaration lowered onto its
//! buffer plan, every slot prepared on the target's encoder backend
//! (`Target::encoder`).

use std::collections::BTreeMap;

use bundle_runtime::acquire;
use bundle_runtime::encoder::{Config, Encoder, Shared as SharedRegistered};
use bundle_runtime::media::MediaArena;
use bundle_runtime::resources::{Held, allocate_span};
use runtime_bundle_abi::{
    BundleAbiError, BundleContextHandleV1, BundleMemoryPoolHandleV1, BundleStreamHandleV1,
    DeviceBufferSpanV1 as Span, HostServicesV1,
};

#[cfg(test)]
use super::MAX_PAYLOAD_BYTES;
use super::{ARENA_ROWS, PAGE_ROWS, views};
use crate::contract::PREFILL_TOKENS;

/// The arena and the one encoder (index 0) as the media records see them.
#[cfg(test)]
pub const ARENA: MediaArena = MediaArena {
    rows: ARENA_ROWS,
    page_rows: PAGE_ROWS,
    payload_limits: &[MAX_PAYLOAD_BYTES],
};

/// The runtime's settings: its log prefix, the diagnostic capture manifests'
/// schema, the arena and the prefill rows.
#[cfg(test)]
pub const CONFIG: Config = Config {
    label: "Qwen3.8",
    capture_schema: "chopinfer.qwen38.encoder-capture.v1",
    arena: ARENA,
    prefill_rows: PREFILL_TOKENS as u64,
    registered: &SharedRegistered,
    staging_bytes: 0,
};

fn config(shape: crate::contract::Shape) -> Config {
    static LIMITS: std::sync::OnceLock<Vec<std::sync::OnceLock<[u64; 1]>>> =
        std::sync::OnceLock::new();
    let payload_limits = crate::contract::per_geometry(&LIMITS, shape, |g| {
        [model_qwen38_contract::encoder::max_payload_bytes(g)]
    });
    Config {
        label: "Qwen3.8",
        capture_schema: "chopinfer.qwen38.encoder-capture.v1",
        arena: MediaArena {
            rows: ARENA_ROWS,
            page_rows: PAGE_ROWS,
            payload_limits,
        },
        prefill_rows: PREFILL_TOKENS as u64,
        registered: &SharedRegistered,
        staging_bytes: 0,
    }
}

pub(crate) fn arena_for(shape: crate::contract::Shape) -> MediaArena {
    config(shape).arena
}

/// Whether the create configuration names the encoder: its image role and
/// every rank's index role, or none of them (`None` for a partial set).
pub fn configured(roles: &[Vec<u8>], ranks: u32) -> Option<bool> {
    let image = roles.iter().any(|role| role == views::IMAGE_ROLE);
    let indexes = (0..ranks)
        .filter(|&rank| {
            roles
                .iter()
                .any(|role| *role == views::index_role(rank).as_bytes())
        })
        .count() as u32;
    match (image, indexes) {
        (true, count) if count == ranks => Some(true),
        (false, 0) => Some(false),
        _ => None,
    }
}

/// Checks each rank's index against the layout and the image identity the
/// host serves, opens the encoder image on every rank, and creates the
/// runtime over its views with each rank's regions.
pub(crate) fn acquire(
    shape: crate::contract::Shape,
    target: &dyn crate::target::Target,
    services: &HostServicesV1,
    ranks: &[(
        BundleContextHandleV1,
        BundleMemoryPoolHandleV1,
        BundleStreamHandleV1,
    )],
    held: &mut Vec<Held>,
) -> Result<Encoder, BundleAbiError> {
    // Each rank's index, bound to the image the host serves for that rank
    // (its file's digest, or its plan's identity) before the image is opened.
    let log = acquire::Log {
        services,
        bundle: "Qwen3.8 encoder",
    };
    let images = acquire::RankImages {
        role: views::IMAGE_ROLE,
        index_role: &views::index_role,
        max_index_bytes: 1 << 20,
    };
    let (image, indexes) = acquire::rank_images(
        log,
        images,
        ranks.len() as u32,
        |rank, bytes| {
            views::parse_for(bytes, rank, shape.geometry).map_err(|error| {
                let _ = services.log(format!("Qwen3.8 encoder index: {error}").as_bytes());
                BundleAbiError::InvalidRange("Qwen3.8 encoder index")
            })
        },
        |index| acquire::Identity {
            image_bytes: index.image_bytes,
            sha256: index.sha256.as_deref(),
            direct_identity: index.direct_identity.as_deref(),
        },
        held,
    )?;
    let arenas = acquire::upload_arenas(log, image, ranks, held)?;
    let mut weights: Vec<BTreeMap<String, Span>> = Vec::with_capacity(ranks.len());
    for (rank, (arena, index)) in arenas.into_iter().zip(indexes).enumerate() {
        let span = services.resolve_compiled_image_view(image, arena, rank as u32, b"")?;
        weights.push(
            views::spans(&index, span.buffer)
                .map_err(|_| BundleAbiError::InvalidRange("Qwen3.8 encoder image view"))?,
        );
    }
    let invalid = || BundleAbiError::InvalidRange("the Qwen3.8 vision declaration is invalid");
    let [vision] = super::declarations_for(shape.geometry).map_err(|_| invalid())?;
    let kernels = target.encoder().map_err(|_| invalid())?;
    // The one encoder's slots.
    let resolve = |_: usize, stage, part, block| (kernels.slots)(stage, part, block);
    Encoder::with_backend(
        services,
        &[Some(vision)],
        config(shape),
        kernels.backend.as_ref(),
        &resolve,
        weights,
        |rank, bytes| {
            let (_, pool, stream) = ranks[rank];
            allocate_span(services, pool, stream, bytes, held)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::super::{MERGE, PATCH_COLUMNS, VIT_OUT};
    use super::*;

    #[test]
    fn constants_match_the_frontend_descriptor() {
        let section: serde_json::Value = serde_json::from_str(include_str!(
            "../../../compiler/src/frontend-multimodal.json"
        ))
        .unwrap();
        let descriptor = &section["multimodal"];
        let arena = &descriptor["arena"];
        assert_eq!(arena["rows"].as_u64(), Some(ARENA.rows));
        assert_eq!(arena["page_rows"].as_u64(), Some(ARENA.page_rows));
        assert_eq!(arena["hidden"].as_u64(), Some(u64::from(VIT_OUT)));
        let encoders = descriptor["encoders"].as_array().unwrap();
        assert_eq!(encoders.len(), ARENA.payload_limits.len());
        assert_eq!(
            encoders[0]["maximum_payload_bytes"].as_u64(),
            Some(MAX_PAYLOAD_BYTES)
        );
        assert_eq!(
            encoders[0]["payload_columns"].as_u64(),
            Some(u64::from(PATCH_COLUMNS))
        );
        let recipe = &descriptor["modalities"][0]["recipe"]["parameters"];
        assert_eq!(recipe["merge"].as_u64(), Some(u64::from(MERGE)));
    }

    #[test]
    fn roles_name_the_whole_encoder_or_none() {
        let mut roles = vec![b"qwen38.rank-images.v1".to_vec()];
        assert_eq!(configured(&roles, 4), Some(false));
        roles.push(views::IMAGE_ROLE.to_vec());
        assert_eq!(configured(&roles, 4), None);
        roles.extend((0..4).map(|rank| views::index_role(rank).into_bytes()));
        assert_eq!(configured(&roles, 4), Some(true));
        roles.remove(1);
        assert_eq!(configured(&roles, 4), None);
    }
}
