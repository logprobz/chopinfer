//! The deepseek41 vision encoder: its replicated image and rank indexes, run
//! by the shared encoder runtime (`bundle_runtime::encoder`) from the vision
//! declaration, every slot prepared on the instance's backend
//! (`crate::Kernels`, where `sm120::encoder` registers the catalog), over
//! its media arena, with the prefill's injection of its rows.
//!
//! The program calls this module: create acquires the encoder when the
//! record names its image and indexes (`configured`, `acquire`), loading its
//! cubins with each entry's dynamic shared memory (`shared_bytes`); each
//! step runs it before the step opens, and the eager prefill injects its
//! rows into the embedding; the router selects those rows with `bias_vl`
//! and the Engram gate closes on them (`image_rows`, `text_rows`).

use std::collections::BTreeMap;

use bundle_runtime::acquire::{self, Log, RankHandles};
use bundle_runtime::encoder::{Config, Encoder, Shared as SharedRegistered, SlotOp};
use bundle_runtime::index::encoder::EncoderIndex;
use bundle_runtime::media::MediaArena;
use bundle_runtime::resources::{Held, allocate_span};
use kernels_sm120::encoder::{Shared, Slot};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1 as Span, HostServicesV1};

pub use model_deepseek41_contract::encoder::*;

use crate::sm120::encoder::kernels;

const LABEL: &str = "DeepSeek V4.1 Flash";

/// The arena and the one encoder (index 0) as the media records see them.
pub const ARENA: MediaArena = MediaArena {
    rows: ARENA_ROWS,
    page_rows: PAGE_ROWS,
    payload_limits: &[MAX_PAYLOAD_BYTES],
};

/// The runtime's settings: its log prefix, the diagnostic capture
/// manifests' schema, the arena and the prefill rows.
pub const CONFIG: Config = Config {
    label: LABEL,
    capture_schema: "chopinfer.deepseek41.encoder-capture.v1",
    arena: ARENA,
    prefill_rows: PREFILL_TOKENS as u64,
    registered: &SharedRegistered,
    staging_bytes: 0,
};

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

/// The dynamic shared memory an encoder entry `symbol` launches with, for
/// the module load; `None` for a symbol that is not the encoder's.
pub fn shared_bytes(symbol: &str) -> Option<u32> {
    kernels()
        .ok()?
        .iter()
        .find(|entry| entry.symbol == symbol)
        .map(|entry| {
            entry
                .dynamic_shared_bytes()
                .unwrap_or(0)
                .max(acquire::DEFAULT_SHARED_BYTES)
        })
}

/// Per live prefill row, whether it is an image row (1) or a text row (0):
/// an image row is one the step's media plan overrides with an arena row
/// (`MediaStepPlan::source_rows`; -1 keeps the token embedding). The encoder
/// writes an image's learned start, newline and end rows into the arena
/// with its cells, so every row of the image's span is one, as the vendor's
/// image mask (`token_types >= 0`) marks them. The router selects these
/// rows with `bias_vl` (`RowBiasRouterQuery::bias_select`).
pub fn image_rows(source_rows: &[i64]) -> Vec<u8> {
    source_rows.iter().map(|&row| u8::from(row >= 0)).collect()
}

/// The Engram gate's `text` flag per live prefill row: the complement of
/// [`image_rows`], so the gate closes (0) on every image row, as the
/// vendor's `engram_mask = ~image_mask`.
pub fn text_rows(source_rows: &[i64]) -> Vec<u8> {
    source_rows.iter().map(|&row| u8::from(row < 0)).collect()
}

/// The device span of every view inside the uploaded image.
pub fn spans(index: &EncoderIndex, image: Span) -> Result<BTreeMap<String, Span>, String> {
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
            Ok((name.clone(), Span::new(image.buffer, offset, view.bytes)))
        })
        .collect()
}

/// The runtime over each rank's encoder image `weights`, every slot
/// prepared on `kernels`' backend from the query its catalog entry was
/// generated from, every rank's regions allocated with `allocate`.
pub(crate) fn create(
    services: &HostServicesV1,
    kernels: &crate::Kernels,
    weights: Vec<BTreeMap<String, Span>>,
    allocate: &mut dyn FnMut(usize, u64) -> Result<Span, BundleAbiError>,
) -> Result<Encoder, BundleAbiError> {
    let invalid = |_| BundleAbiError::InvalidRange("the deepseek41 vision declaration is invalid");
    let [vision] = declarations().map_err(invalid)?;
    let slots = kernels_sm120::encoder::slots(vision, &Shared).map_err(invalid)?;
    let resolve = |_: usize, stage: usize, part: &'static str, block: Option<u32>| {
        resolve(&slots, Slot { stage, part, block })
    };
    Encoder::with_backend(
        services,
        &[Some(vision)],
        CONFIG,
        kernels.backend.as_ref(),
        &resolve,
        weights,
        allocate,
    )
}

/// Each slot's query, which the backend serves from the catalog entry of
/// its code.
fn resolve(slots: &[(Slot, kernels_sm120::encoder::Entry)], slot: Slot) -> Result<SlotOp, String> {
    let (_, entry) = slots
        .iter()
        .find(|(known, _)| *known == slot)
        .ok_or_else(|| format!("the vision catalog has no slot {slot:?}"))?;
    Ok(SlotOp::Query(entry.query))
}

/// Checks each rank's index against the layout and the image identity the
/// host serves, opens the encoder image on every rank, and creates the
/// runtime over its views with each rank's regions.
pub(crate) fn acquire(
    services: &HostServicesV1,
    kernels: &crate::Kernels,
    ranks: &[RankHandles],
    held: &mut Vec<Held>,
) -> Result<Encoder, BundleAbiError> {
    let weights = image_views(services, ranks, held)?;
    create(services, kernels, weights, &mut |rank, bytes| {
        let (_, pool, stream) = ranks[rank];
        allocate_span(services, pool, stream, bytes, held)
    })
}

/// The views of the encoder image on every rank: each rank's index, bound
/// to the image the host serves for that rank (its plan's identity, or its
/// file's digest) before the image is opened, then the image's spans.
fn image_views(
    services: &HostServicesV1,
    ranks: &[RankHandles],
    held: &mut Vec<Held>,
) -> Result<Vec<BTreeMap<String, Span>>, BundleAbiError> {
    let log = Log {
        services,
        bundle: "DeepSeek V4.1 Flash encoder",
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
            views::parse(bytes, rank).map_err(|error| {
                let _ = services.log(format!("{LABEL} encoder index: {error}").as_bytes());
                BundleAbiError::InvalidRange("deepseek41 encoder index")
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
            spans(&index, span.buffer)
                .map_err(|_| BundleAbiError::InvalidRange("deepseek41 encoder image view"))?,
        );
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use bundle_runtime::encoder::Lowered;
    use encoder_spec::buffers::{self, Serving};

    use super::*;

    #[test]
    fn roles_name_the_whole_encoder_or_none() {
        let mut roles = vec![
            model_deepseek41_contract::serving::RANK_IMAGE_ROLE
                .as_bytes()
                .to_vec(),
        ];
        assert_eq!(configured(&roles, 4), Some(false));
        roles.push(views::IMAGE_ROLE.to_vec());
        assert_eq!(configured(&roles, 4), None);
        roles.extend((0..4).map(|rank| views::index_role(rank).into_bytes()));
        assert_eq!(configured(&roles, 4), Some(true));
        roles.remove(1);
        assert_eq!(configured(&roles, 4), None);
    }

    /// The encoder cubins embed with the bundle's, holding every catalog
    /// entry, and each loads with at least its planned shared memory.
    #[test]
    fn the_encoder_cubins_embed_every_entry() {
        let embedded: Vec<&str> = crate::CUBINS
            .iter()
            .filter(|cubin| crate::sm120::encoder::is_cubin(cubin.name))
            .flat_map(|cubin| cubin.symbols.iter().copied())
            .collect();
        let mut catalog: Vec<&str> = kernels()
            .unwrap()
            .iter()
            .map(|entry| entry.symbol.as_str())
            .collect();
        let mut sorted = embedded.clone();
        sorted.sort_unstable();
        catalog.sort_unstable();
        assert_eq!(sorted, catalog);
        for entry in kernels().unwrap() {
            let planned = entry.dynamic_shared_bytes().unwrap_or(0);
            assert!(shared_bytes(&entry.symbol).unwrap() >= planned);
        }
        assert_eq!(shared_bytes("chop_row_not_an_encoder_entry"), None);
    }

    /// Every slot of the declaration prepares on the instance's backend at
    /// TP4, the rows' and the encoder's entries registered together, and
    /// the launch list runs the catalog's entries.
    #[test]
    fn every_slot_prepares_on_the_instance_backend() {
        let shape = crate::shape::Shape::new(
            &model_deepseek41_contract::geometry::FLASH,
            bundle_runtime::geometry::Ranks::new(4).unwrap(),
        )
        .unwrap();
        let instance = crate::Kernels::new(&shape).unwrap();
        let [vision] = declarations().unwrap();
        let plan = buffers::plan(
            &[vision],
            u64::from(vision.input.capacity_rows),
            Serving {
                arena_rows: ARENA_ROWS,
                prefill_rows: u64::from(PREFILL_TOKENS),
                staging_bytes: 0,
            },
        )
        .unwrap();
        let slots = kernels_sm120::encoder::slots(vision, &Shared).unwrap();
        let lowered = Lowered::with_backend(
            vision,
            &plan,
            instance.backend.as_ref(),
            &|stage, part, block| resolve(&slots, Slot { stage, part, block }),
            &SharedRegistered,
            PREFILL_TOKENS,
        )
        .unwrap();
        let catalog: std::collections::BTreeSet<&str> = kernels()
            .unwrap()
            .iter()
            .map(|entry| entry.symbol.as_str())
            .collect();
        let launched: std::collections::BTreeSet<&str> = lowered.symbols().collect();
        assert!(launched.is_subset(&catalog), "{launched:?}");
        assert!(
            launched
                .iter()
                .any(|symbol| symbol.starts_with("chop_enc_merge_space_to_depth_w1024_g9_"))
        );
    }

    #[test]
    fn image_and_text_rows_follow_the_arena_sources() {
        let sources = [-1, 0, 1, -1, 7, -1];
        assert_eq!(image_rows(&sources), [0, 1, 1, 0, 1, 0]);
        assert_eq!(text_rows(&sources), [1, 0, 0, 1, 0, 1]);
        assert!(image_rows(&[]).is_empty());
    }
}
