//! A keeper's load: every rank image the instance's weight plans produce,
//! uploaded through the host services a bundle's own uploads call, into
//! memory another process can map.

use runtime_bundle_abi::{BundleStatusV1, HostServicesBackendV1, RankArenaUploadV1};

use super::KeptImageV1;
use crate::device::HostDeviceBackendV1;
use crate::services::HostServicesV1Backend;

/// The stream kind of the keeper's upload streams, the host's own.
const UPLOAD_STREAM_KIND: u32 = 0;

fn failed(what: &'static str) -> impl Fn(BundleStatusV1) -> String {
    move |status| {
        format!(
            "the keeper's {what} failed: {}",
            status.name().unwrap_or("unknown status")
        )
    }
}

/// Loads every planned rank image for the ranks on `device_ordinals`, one
/// upload per image across all ranks, and returns each rank's arena with
/// the descriptor that lends it, holding its memory. Every arena and image
/// is released, so dropping `backend` afterwards releases every context
/// while the memory stays. `backend` must take its arenas from
/// [`super::ArenaSourceV1::Lend`].
pub(crate) fn load_v1<D: HostDeviceBackendV1>(
    backend: &HostServicesV1Backend<D>,
    device_ordinals: &[i32],
) -> Result<Vec<KeptImageV1>, String> {
    let mut ranks = Vec::with_capacity(device_ordinals.len());
    for (rank, &ordinal) in device_ordinals.iter().enumerate() {
        let rank = u32::try_from(rank).map_err(|_| "too many ranks".to_owned())?;
        let context = backend
            .select_context(rank, ordinal)
            .map_err(failed("context selection"))?;
        let pool = backend
            .select_memory_pool(context)
            .map_err(failed("memory pool selection"))?;
        let stream = backend
            .create_stream(context, UPLOAD_STREAM_KIND)
            .map_err(failed("stream creation"))?;
        ranks.push((rank, context, pool, stream));
    }
    let roles = backend.planned_image_roles_v1();
    if roles.is_empty() {
        return Err("the package plans no rank image from the checkpoint".to_owned());
    }
    let mut opened = Vec::with_capacity(roles.len());
    for role in roles {
        let image = backend
            .open_compiled_image(&role)
            .map_err(failed("image open"))?;
        let uploads: Vec<RankArenaUploadV1> = ranks
            .iter()
            .map(|&(rank, context, pool, stream)| RankArenaUploadV1 {
                context,
                pool,
                stream,
                image,
                rank,
            })
            .collect();
        let arenas = backend
            .upload_rank_arenas(&uploads)
            .map_err(failed("upload"))?;
        opened.push((image, arenas));
    }
    // Newest first, as the host releases everything.
    let mut kept = Vec::new();
    for (image, arenas) in opened.into_iter().rev() {
        for arena in arenas.into_iter().rev() {
            kept.push(backend.hold_rank_arena_v1(arena).map_err(failed("hold"))?);
        }
        backend
            .release_compiled_image(image)
            .map_err(failed("image release"))?;
    }
    kept.reverse();
    Ok(kept)
}
