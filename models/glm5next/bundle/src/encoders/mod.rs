//! The glm5next vision encoder: its replicated image and rank indexes, run
//! by the shared encoder runtime (`bundle_runtime::encoder`) from the
//! vision declaration, every slot prepared on the target's backend
//! (`Target::encoder`), over its media arena, with the prefill's injection
//! of its rows.
//!
//! The encoder's two cubins embed with the bundle's, in their own lock
//! family (`FAMILY`), and load only for an instance that serves the
//! encoder (`load`, `Target::load_encoder`). Create loads and acquires it when the record names
//! its image and indexes (`configured`); each step runs it before the
//! step opens, and the eager prefill injects its rows into the embedding.
//! A package serving images plans the encoder image and requires the media
//! capability, so the host names the encoder and sends media steps.

use std::collections::BTreeMap;

use bundle_runtime::acquire::{self, Log, RankHandles};
use bundle_runtime::encoder::{Config, Encoder, Shared as SharedRegistered};
use bundle_runtime::index::encoder::EncoderIndex;
use bundle_runtime::media::MediaArena;
use bundle_runtime::resources::{Held, allocate_span};
use encoder_spec::components::StageOp;
use encoder_spec::policy::Precision;
use model_glm5next_contract::contract::PREFILL_TOKENS;
use runtime_bundle_abi::{
    BundleAbiError, BundleCudaSymbolHandleV1, DeviceBufferSpanV1 as Span, HostServicesV1,
};
use runtime_bundle_sdk::taps::{TapCompare, TapDType, TapSharding, TapSpec};

use crate::target::Target;

pub use model_glm5next_contract::encoder::*;

#[cfg(test)]
mod fake_host_tests;

const LABEL: &str = "GLM 5.3 Flash";

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
    capture_schema: "chopinfer.glm5next.encoder-capture.v1",
    arena: ARENA,
    prefill_rows: PREFILL_TOKENS as u64,
    registered: &SharedRegistered,
    staging_bytes: 0,
};

/// The encoder's taps (`models/glm5next/taps.json`): the merged output rows
/// in BF16 from rank 0's media arena, and every transformer block's input
/// and output rows as the F32 residual stream the runtime reads back
/// (`Encoder::block_taps`), blocks numbered across the transformers.
pub fn taps() -> Result<Vec<TapSpec>, String> {
    let [vision] = declarations().map_err(|error| error.0)?;
    let tap = |name: String, cols, dtype| TapSpec {
        name,
        layer: None,
        detail: false,
        cols,
        dtype,
        sharding: TapSharding::Replicated,
        compare: TapCompare::Dense,
    };
    let dtype = match vision.precision {
        Precision::F32Pairs => TapDType::F32,
        Precision::Bf16Legacy => TapDType::Bf16,
    };
    let mut specs = vec![tap(
        "enc.vision.out".into(),
        vision.output.op.params().hidden,
        TapDType::Bf16,
    )];
    let mut block = 0;
    for stage in &vision.stages {
        let StageOp::Transformer(transformer) = &stage.op else {
            continue;
        };
        let params = transformer.params();
        specs.push(tap(
            format!("enc.vision.B{block:02}.in"),
            params.width,
            dtype,
        ));
        for _ in 0..params.blocks {
            specs.push(tap(
                format!("enc.vision.B{block:02}.out"),
                params.width,
                dtype,
            ));
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

/// Loads the encoder's cubins on every rank (`Target::load_encoder`).
pub fn load(
    target: &dyn Target,
    log: Log,
    ranks: &[RankHandles],
    held: &mut Vec<Held>,
) -> Result<Vec<BTreeMap<String, BundleCudaSymbolHandleV1>>, BundleAbiError> {
    target.load_encoder(log, ranks, held)
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

/// Checks each rank's index against the layout and the image identity the
/// host serves, opens the encoder image on every rank, and creates the
/// runtime over its views with each rank's regions on the target's encoder
/// kernels (`Target::encoder`).
pub fn acquire(
    target: &dyn Target,
    services: &HostServicesV1,
    ranks: &[RankHandles],
    held: &mut Vec<Held>,
) -> Result<Encoder, BundleAbiError> {
    let weights = image_views(services, ranks, held)?;
    target.encoder(services, weights, &mut |rank, bytes| {
        let (_, pool, stream) = ranks[rank];
        allocate_span(services, pool, stream, bytes, held)
    })
}

/// The views of the encoder image on every rank: each rank's index, bound
/// to the image the host serves for that rank (its file's digest, or its
/// plan's identity) before the image is opened, then the image's spans.
pub(crate) fn image_views(
    services: &HostServicesV1,
    ranks: &[RankHandles],
    held: &mut Vec<Held>,
) -> Result<Vec<BTreeMap<String, Span>>, BundleAbiError> {
    let log = Log {
        services,
        bundle: "GLM 5.3 Flash encoder",
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
                BundleAbiError::InvalidRange("glm5next encoder index")
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
                .map_err(|_| BundleAbiError::InvalidRange("glm5next encoder image view"))?,
        );
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_the_frontend_descriptor() {
        let frontend: serde_json::Value =
            serde_json::from_str(include_str!("../../../compiler/src/frontend.json")).unwrap();
        let descriptor = &frontend["multimodal"];
        let arena = &descriptor["arena"];
        assert_eq!(arena["rows"].as_u64(), Some(ARENA.rows));
        assert_eq!(arena["page_rows"].as_u64(), Some(ARENA.page_rows));
        assert_eq!(arena["hidden"].as_u64(), Some(u64::from(ARENA_HIDDEN)));
        let encoders = descriptor["encoders"].as_array().unwrap();
        assert_eq!(encoders.len(), ARENA.payload_limits.len());
        assert_eq!(encoders[0]["name"], "glm5next-vision");
        assert_eq!(
            encoders[0]["maximum_payload_bytes"].as_u64(),
            Some(MAX_PAYLOAD_BYTES)
        );
        assert_eq!(
            encoders[0]["payload_columns"].as_u64(),
            Some(u64::from(PATCH_COLUMNS))
        );
        let [vision] = declarations().unwrap();
        assert_eq!(vision.input.columns, PATCH_COLUMNS);
        assert_eq!(vision.output.op.params().hidden, ARENA_HIDDEN);
    }

    #[test]
    fn roles_name_the_whole_encoder_or_none() {
        let mut roles = vec![
            model_glm5next_contract::direct::RANK_IMAGE_ROLE
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

    /// The declaration declares every encoder tap as the runtime writes it,
    /// and every encoder entry matches one: 24 blocks of the 1024-wide F32
    /// residual stream and the 4096-wide BF16 merged rows.
    #[test]
    fn taps_json_declares_every_encoder_tap() {
        let declaration =
            runtime_bundle_sdk::taps::Declaration::parse(include_str!("../../../taps.json"))
                .unwrap();
        let specs = taps().unwrap();
        assert_eq!(specs.len(), 1 + 1 + 24);
        let mut used = vec![false; declaration.entries().len()];
        for spec in &specs {
            used[declaration.check(spec, 4).unwrap()] = true;
        }
        for (entry, used) in declaration.entries().iter().zip(used) {
            let name = entry["name"].as_str().unwrap();
            assert_eq!(used, name.starts_with("enc."), "{name}");
        }
    }
}
