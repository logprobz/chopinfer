//! The deepseek41 vision encoder: its replicated image and rank indexes, run
//! by the shared encoder runtime (`bundle_runtime::encoder`) from the vision
//! declaration, every slot prepared on the instance's backend
//! (`Kernels::encoder`, which the target's module implements), over its
//! media arena, with the prefill's injection of its rows.
//!
//! The program calls this module: create acquires the encoder when the
//! record names its image and indexes (`configured`, `acquire`), loading its
//! cubins with each entry's dynamic shared memory
//! (`Kernels::encoder_shared_bytes`); each step runs it before the step
//! opens, and the eager prefill injects its rows into the embedding; the
//! router selects those rows with `bias_vl` and the Engram gate closes on
//! them (`image_rows`, `text_rows`).

use std::collections::BTreeMap;

use bundle_runtime::acquire::{self, Log, RankHandles};
use bundle_runtime::encoder::{Config, Encoder, Shared as SharedRegistered};
use bundle_runtime::index::encoder::EncoderIndex;
use bundle_runtime::media::MediaArena;
use bundle_runtime::resources::{Held, allocate_span};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1 as Span, HostServicesV1};
use runtime_bundle_sdk::taps::{TapCompare, TapDType, TapSharding, TapSpec};

pub use model_deepseek41_contract::encoder::*;

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

/// The encoder's taps (`models/deepseek41/taps.json`): the aligner's image
/// cells in BF16 from rank 0's media arena (`enc.vision.out`, without the
/// learned delimiter rows, as the reference taps the aligner), and every
/// transformer block's input and output rows as the residual stream the
/// runtime reads back (`Encoder::block_taps`), blocks numbered across the
/// transformers.
pub fn taps() -> Result<Vec<TapSpec>, String> {
    let shape = tap_shape().map_err(|error| error.0)?;
    let tap = |name: String, cols, dtype| TapSpec {
        name,
        layer: None,
        detail: false,
        cols,
        dtype,
        sharding: TapSharding::Replicated,
        compare: TapCompare::Dense,
    };
    let dtype = if shape.f32 {
        TapDType::F32
    } else {
        TapDType::Bf16
    };
    let mut specs = vec![tap("enc.vision.out".into(), ARENA_HIDDEN, TapDType::Bf16)];
    let mut block = 0;
    for (width, blocks) in shape.transformers {
        specs.push(tap(format!("enc.vision.B{block:02}.in"), width, dtype));
        for _ in 0..blocks {
            specs.push(tap(format!("enc.vision.B{block:02}.out"), width, dtype));
            block += 1;
        }
    }
    Ok(specs)
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

/// The encoder of an instance serving images, and each rank's text flags
/// of the prefill rows (U8, one a row), which a prefill that injects
/// encoder rows uploads for the masked Engram gate.
pub(crate) struct Media {
    pub(crate) encoder: Encoder,
    pub(crate) text: Vec<Span>,
}

/// Loads the encoder's cubins on every rank into `symbols`, each entry with
/// its planned dynamic shared memory; acquires the encoder over its image
/// and each rank's text flags of `rows` rows.
pub(crate) fn acquire_media(
    services: &HostServicesV1,
    kernels: &crate::Kernels,
    ranks: &[RankHandles],
    symbols: &mut [BTreeMap<String, runtime_bundle_abi::BundleCudaSymbolHandleV1>],
    rows: u64,
    held: &mut Vec<Held>,
) -> Result<Media, BundleAbiError> {
    let log = Log {
        services,
        bundle: "DeepSeek V4.1 Flash encoder",
    };
    for (&(context, _, _), resolved) in ranks.iter().zip(symbols.iter_mut()) {
        for cubin in crate::CUBINS
            .iter()
            .filter(|cubin| crate::Kernels::encoder_cubin(cubin.name))
        {
            acquire::load_module(
                log,
                context,
                (cubin.name, cubin.bytes),
                cubin.symbols.iter().copied(),
                |symbol| {
                    crate::Kernels::encoder_shared_bytes(symbol).ok_or(
                        BundleAbiError::InvalidRange("unplanned deepseek41 encoder symbol"),
                    )
                },
                resolved,
                held,
            )?;
        }
    }
    let encoder = acquire(services, kernels, ranks, held)?;
    let text = ranks
        .iter()
        .map(|&(_, pool, stream)| allocate_span(services, pool, stream, rows, held))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Media { encoder, text })
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
    kernels.encoder(services, weights, &mut |rank, bytes| {
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

    #[test]
    fn image_and_text_rows_follow_the_arena_sources() {
        let sources = [-1, 0, 1, -1, 7, -1];
        assert_eq!(image_rows(&sources), [0, 1, 1, 0, 1, 0]);
        assert_eq!(text_rows(&sources), [1, 0, 0, 1, 0, 1]);
        assert!(image_rows(&[]).is_empty());
    }
}
