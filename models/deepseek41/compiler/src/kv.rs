//! The serving KV formats of the `deepseek41.v1` package: the native
//! trained formats, the package's preferred layout, and BF16 holding the
//! same values, a comparison layout that is never a served default.
//!
//! A page holds, for each KV source in layer order, one compressed latent
//! and one index key per `ratio` tokens. Natively a latent is E2M1 with an
//! E4M3 scale per 16 values and an index key E2M1 with a UE8M0 scale per 32;
//! the BF16 layout stores each value's dequantized product, which BF16 holds
//! exactly. The window rings are per-request state, not page planes, and
//! BF16 rings make that state larger, so the BF16 format carries its own.

use model_deepseek41_contract::geometry::Geometry;
use runtime_bundle::{
    BundleError, BundleKvCodeV1, BundleKvDtypeV1, BundleKvFormatV1, BundleKvLayoutV1,
    BundleKvPlaneV1, BundleKvScaleFormatV1, BundleKvScaleV1, BundleOpaqueId, BundleStateLayoutV1,
};

/// Native latent values per E4M3 scale.
const LATENT_BLOCK: u32 = 16;
/// Native index-key values per UE8M0 scale.
const INDEX_BLOCK: u32 = 32;

/// The values one token stores in the pages: of every KV source, a latent
/// and an index key per `ratio` tokens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenValues {
    pub latent: u32,
    pub index: u32,
}

impl TokenValues {
    pub fn of(geometry: &Geometry) -> Result<Self, BundleError> {
        let (latent, index) = (
            geometry.shape.attention.head_dim,
            geometry.shape.indexer.dim,
        );
        let mut values = Self {
            latent: 0,
            index: 0,
        };
        for &layer in geometry.shape.csa.kv_sources {
            let ratio = geometry.compress_ratio(layer);
            if ratio == 0 || !latent.is_multiple_of(ratio) || !index.is_multiple_of(ratio) {
                return Err(BundleError::validation(format!(
                    "KV source {layer} compresses at ratio {ratio}, which does not divide its \
                     {latent} latent and {index} index values"
                )));
            }
            values.latent += latent / ratio;
            values.index += index / ratio;
        }
        Ok(values)
    }

    /// Bytes a token takes in the native pages: two E2M1 codes a byte and
    /// one scale byte per block.
    #[must_use]
    pub const fn native_bytes(self) -> u64 {
        (self.latent / 2 + self.latent / LATENT_BLOCK + self.index / 2 + self.index / INDEX_BLOCK)
            as u64
    }

    /// Bytes a token takes in the BF16 pages.
    #[must_use]
    pub const fn bf16_bytes(self) -> u64 {
        2 * (self.latent as u64 + self.index as u64)
    }
}

fn id(value: &str) -> Result<BundleOpaqueId, BundleError> {
    BundleOpaqueId::new(value)
}

/// The native element's planes, `compressed` (the latents) and `index`
/// (the index keys), by the values each token stores in them. Where a page
/// places them is the bundle's: by source layer, then stream.
pub fn native_planes(geometry: &Geometry) -> Result<Vec<BundleKvPlaneV1>, BundleError> {
    let values = TokenValues::of(geometry)?;
    Ok(vec![
        BundleKvPlaneV1::new(
            id("compressed")?,
            BundleKvCodeV1::Fp4E2m1,
            values.latent,
            Some(BundleKvScaleV1::new(
                BundleKvScaleFormatV1::E4m3,
                LATENT_BLOCK,
            )?),
        )?,
        BundleKvPlaneV1::new(
            id("index")?,
            BundleKvCodeV1::Fp4E2m1,
            values.index,
            Some(BundleKvScaleV1::new(
                BundleKvScaleFormatV1::Ue8m0,
                INDEX_BLOCK,
            )?),
        )?,
    ])
}

/// The package's serving KV formats at one topology: the native layout
/// `native` (the preferred `kv_layout`, run with the manifest's state) and
/// the BF16 layout `bf16`, run with `bf16_state`, the request state sized
/// for BF16 rings.
pub fn kv_formats(
    geometry: &Geometry,
    native: BundleKvLayoutV1,
    bf16: BundleKvLayoutV1,
    bf16_state: BundleStateLayoutV1,
) -> Result<Vec<BundleKvFormatV1>, BundleError> {
    let needed = TokenValues::of(geometry)?.bf16_bytes();
    if bf16.bytes_per_token() < needed {
        return Err(BundleError::validation(format!(
            "the BF16 KV layout {} allocates {} bytes per token; its values need {needed}",
            bf16.id(),
            bf16.bytes_per_token()
        )));
    }
    Ok(vec![
        BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, bf16).with_state_layout(bf16_state)?,
        BundleKvFormatV1::mixed_planes(native_planes(geometry)?, native)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::FLASH;
    use runtime_bundle::{BundleRankPlacementV1, BundleStateLifetimeV1};

    fn layout(name: &str, page_bytes: u64, alignment: u64) -> BundleKvLayoutV1 {
        BundleKvLayoutV1::new(
            id(name).unwrap(),
            page_bytes,
            alignment,
            64,
            page_bytes / 64,
            1,
            FLASH.context_tokens(),
            BundleRankPlacementV1::RankLocal,
            BundleStateLifetimeV1::Transaction,
        )
        .unwrap()
    }

    /// The model card's 890 bytes of global KV per token natively and 3,200
    /// in BF16: per token, 3 ratio-2 sources and 1 ratio-1 source of 512
    /// latent and 128 index values.
    #[test]
    fn flash_tokens_take_the_cards_bytes() {
        let values = TokenValues::of(&FLASH).unwrap();
        assert_eq!(
            values,
            TokenValues {
                latent: 1280,
                index: 320
            }
        );
        assert_eq!(values.native_bytes(), 890);
        assert_eq!(values.bf16_bytes(), 3200);
        let planes = native_planes(&FLASH).unwrap();
        let total: u64 = planes
            .iter()
            .map(BundleKvPlaneV1::values)
            .map(u64::from)
            .sum();
        assert_eq!(total, 1600);
        // compressed: 640 code and 80 scale bytes; index: 160 and 10.
        assert_eq!(
            planes
                .iter()
                .map(BundleKvPlaneV1::bytes_per_token)
                .collect::<Vec<_>>(),
            [720, 170]
        );
        assert_eq!(
            planes
                .iter()
                .map(BundleKvPlaneV1::describe)
                .collect::<Vec<_>>(),
            ["compressed fp4_e2m1/e4m3x16", "index fp4_e2m1/ue8m0x32"]
        );
    }

    #[test]
    fn the_package_declares_native_and_bf16_formats() {
        let state = BundleStateLayoutV1::new(
            id("deepseek41.request-state.v1").unwrap(),
            5_668_864,
            256,
            BundleStateLifetimeV1::Request,
        )
        .unwrap();
        let bf16 = layout("deepseek41.csa2-paged-bf16.v1", 204_800, 256);
        // The native page as declared (890 bytes a token) and padded to a
        // multiple of 256 (892); both hold the planes.
        for native in [
            layout("deepseek41.csa2-paged-native.v1", 56_960, 64),
            layout("deepseek41.csa2-paged-native.v1", 57_088, 256),
        ] {
            let formats = kv_formats(&FLASH, native.clone(), bf16.clone(), state.clone()).unwrap();
            assert_eq!(
                formats
                    .iter()
                    .map(BundleKvFormatV1::dtype)
                    .collect::<Vec<_>>(),
                [BundleKvDtypeV1::Bf16, BundleKvDtypeV1::MixedPlanes]
            );
            assert_eq!(formats[0].state_layout(), Some(&state));
            assert_eq!(formats[1].state_layout(), None);
            assert_eq!(formats[1].layout(), &native);
        }
        // A page short of a token's values is refused, natively by the
        // planes and in BF16 here.
        let short = layout("deepseek41.csa2-paged-native.v1", 56_896, 64);
        assert!(kv_formats(&FLASH, short, bf16, state.clone()).is_err());
        let native = layout("deepseek41.csa2-paged-native.v1", 56_960, 64);
        let short = layout("deepseek41.csa2-paged-bf16.v1", 196_608, 256);
        let error = kv_formats(&FLASH, native, short, state)
            .unwrap_err()
            .to_string();
        assert!(error.contains("its values need 3200"), "{error}");
    }
}
