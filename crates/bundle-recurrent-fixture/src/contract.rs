//! Everything this fixture declares about itself, in one place.
//!
//! The emitted manifest and the native module read the same identifiers, sizes,
//! and bytes from here, so a package that installs is described by the same
//! declaration the module inside it was written against. Nothing here is opaque
//! to this fixture and nothing here is meaningful to a host: every identifier is
//! a bounded opaque string the manifest carries and the host never interprets.

use runtime_bundle::{
    BundleError, BundleKvLayoutV1, BundleOpaqueId, BundleRankPlacementV1, BundleStateLayoutV1,
    BundleStateLifetimeV1, BundleTopologyV1,
};

use crate::recurrent::{RECURRENT_LANES_V1, RECURRENT_STATE_BYTES_V1, RecurrentLanesV1};

/// The identifier this bundle installs and is selected by.
pub const RECURRENT_BUNDLE_ID_V1: &str = "fixture.recurrent.tp1.v1";

/// The name `build-recurrent-fixture --fixture` accepts for this bundle.
pub const RECURRENT_FIXTURE_NAME_V1: &str = "tp1-recurrent";

/// The one device architecture this fixture is emitted for.
pub const RECURRENT_ARCHITECTURE_V1: &str = "sm_120f";

/// The one topology this fixture serves: a single rank, and no collective.
pub const RECURRENT_TOPOLOGY_ID_V1: &str = "tp1";

/// How many ranks that topology has.
pub const RECURRENT_RANK_COUNT_V1: u32 = 1;

/// The component that carries this fixture's program image.
pub const RECURRENT_PROGRAM_COMPONENT_V1: &str = "program";

/// The role a host resolves that component by.
pub const RECURRENT_PROGRAM_ROLE_V1: &str = "fixture.recurrent.program.v1";

/// The opaque format of that component.
pub const RECURRENT_PROGRAM_FORMAT_V1: &str = "fixture.recurrent.image.v1";

/// The component that carries this fixture's frontend vocabulary declaration.
pub const RECURRENT_FRONTEND_COMPONENT_V1: &str = "frontend";

/// The role a host resolves that component by.
pub const RECURRENT_FRONTEND_ROLE_V1: &str = "fixture.recurrent.frontend.v1";

/// The opaque format of that component.
pub const RECURRENT_FRONTEND_FORMAT_V1: &str = "fixture.recurrent.vocabulary.v1";

/// Where that component sits inside the package.
pub const RECURRENT_FRONTEND_PATH_V1: &str = "frontend/vocabulary.json";

/// Where the native module sits inside the package.
///
/// The extension is deliberately neutral: the loader stages its own private
/// image and names that copy itself, so nothing loads this path directly.
pub const RECURRENT_NATIVE_MODULE_PATH_V1: &str = "native/module.bin";

/// The one graph bucket this fixture declares.
pub const RECURRENT_GRAPH_BUCKET_V1: &str = "recurrent-step";

/// The stage identifier this fixture selects its one stream for.
pub const RECURRENT_STAGE_ID_V1: &[u8] = b"recurrent-step";

/// The stream kind this fixture asks for. It orders every transfer it makes.
pub const RECURRENT_STREAM_KIND_V1: u32 = 0;

/// The identifier of the persistent state this fixture keeps.
pub const RECURRENT_STATE_LAYOUT_ID_V1: &str = "fixture.recurrent.state.v1";

/// The alignment the persistent state is allocated at.
pub const RECURRENT_STATE_ALIGNMENT_V1: u64 = 64;

/// The identifier of the KV page layout this fixture declares and never uses.
///
/// The canonical manifest carries one KV layout for every bundle. This fixture
/// carries recurrent state instead, so the layout it declares is the smallest
/// well-formed one and no physical KV pool is ever allocated from it.
pub const RECURRENT_KV_LAYOUT_ID_V1: &str = "fixture.recurrent.kv.v1";

/// How many tokens one declared KV page holds.
pub const RECURRENT_KV_PAGE_TOKENS_V1: u32 = 8;

/// How many bytes one declared KV page holds per token.
pub const RECURRENT_KV_BYTES_PER_TOKEN_V1: u64 = 4;

/// How many equally shaped segments one declared KV page holds.
pub const RECURRENT_KV_SEGMENTS_PER_PAGE_V1: u32 = 1;

/// How many bytes one declared KV page holds in total.
pub const RECURRENT_KV_PAGE_BYTES_V1: u64 = RECURRENT_KV_BYTES_PER_TOKEN_V1
    * RECURRENT_KV_PAGE_TOKENS_V1 as u64
    * RECURRENT_KV_SEGMENTS_PER_PAGE_V1 as u64;

/// The alignment one declared KV page is allocated at.
pub const RECURRENT_KV_ALIGNMENT_V1: u64 = 32;

/// A second KV layout this module runs: the same pages at one byte per
/// element, as an FP8 layout halves a BF16 one. The canonical
/// package declares only [`RECURRENT_KV_LAYOUT_ID_V1`]; a test that adds this
/// one to the manifest's `kv_formats` sees the host select it.
pub const RECURRENT_KV_FP8_LAYOUT_ID_V1: &str = "fixture.recurrent.kv-fp8.v1";

/// How many bytes one page of the FP8 layout holds.
pub const RECURRENT_KV_FP8_PAGE_BYTES_V1: u64 = RECURRENT_KV_PAGE_BYTES_V1 / 2;

/// The longest sequence this fixture declares it can hold.
pub const RECURRENT_MAXIMUM_SEQUENCE_TOKENS_V1: u64 = 64;

/// The device memory a host must have before this fixture will load.
pub const RECURRENT_MINIMUM_DEVICE_MEMORY_BYTES_V1: u64 = 1 << 20;

/// The alignment this fixture allocates its own device buffers at.
pub const RECURRENT_BUFFER_ALIGNMENT_V1: u64 = 256;

/// How wide the declared vocabulary is, and therefore how wide one logits row is.
///
/// This is small on purpose. The row a caller receives is exactly this many
/// 32-bit logits, and the program image carries the same number so the module
/// reads its width from the package rather than from a constant compiled in.
pub const RECURRENT_VOCABULARY_V1: u32 = 64;

/// The most prompt tokens one execution may carry.
pub const RECURRENT_MAXIMUM_TOKENS_V1: u32 = 64;

/// The widest vocabulary the native module will accept from a program image.
///
/// The image is authenticated, so this is not a trust boundary. It is a bound:
/// a width past it would size an allocation this fixture never intends to make.
pub const RECURRENT_VOCABULARY_BOUND_V1: u32 = 4096;

/// The most prompt tokens the native module will accept from a program image.
pub const RECURRENT_TOKEN_BOUND_V1: u32 = 4096;

/// The operation half of the key this fixture's own custom operation carries.
pub const RECURRENT_STEP_OPERATION_V1: &str = "fixture.recurrent.step";

/// The variant half of that key.
pub const RECURRENT_STEP_VARIANT_V1: &str = "tp1.v1";

/// The named view that carries this fixture's program geometry.
pub const RECURRENT_GEOMETRY_VIEW_V1: &[u8] = b"program-geometry";

/// The named view that carries this fixture's initial recurrent state.
pub const RECURRENT_STATE_SEED_VIEW_V1: &[u8] = b"state-seed";

/// The alignment the program image and its rank image are placed at.
pub const RECURRENT_PROGRAM_IMAGE_ALIGNMENT_V1: u64 = 64;

/// How many bytes the whole program image holds for its one rank.
pub const RECURRENT_PROGRAM_IMAGE_BYTES_V1: u64 = 128;

/// Where the geometry view starts inside the rank image.
pub const RECURRENT_GEOMETRY_OFFSET_V1: u64 = 0;

/// How many bytes the geometry view holds.
pub const RECURRENT_GEOMETRY_BYTES_V1: u64 = 16;

/// The alignment the geometry view is placed at.
pub const RECURRENT_GEOMETRY_ALIGNMENT_V1: u64 = 16;

/// Where the state seed view starts inside the rank image.
pub const RECURRENT_STATE_SEED_OFFSET_V1: u64 = 64;

/// The one topology this fixture serves.
pub fn recurrent_topology_v1() -> Result<BundleTopologyV1, BundleError> {
    BundleTopologyV1::new(id(RECURRENT_TOPOLOGY_ID_V1)?, RECURRENT_RANK_COUNT_V1)
}

/// The persistent recurrent state this fixture carries instead of a KV cache.
///
/// The lanes outlive every transaction and every request: they are the model,
/// and they live as long as the instance that holds them.
pub fn recurrent_state_layout_v1() -> Result<BundleStateLayoutV1, BundleError> {
    BundleStateLayoutV1::new(
        id(RECURRENT_STATE_LAYOUT_ID_V1)?,
        RECURRENT_STATE_BYTES_V1,
        RECURRENT_STATE_ALIGNMENT_V1,
        BundleStateLifetimeV1::Process,
    )
}

/// The KV page layout this fixture declares and never allocates.
///
/// A V1 manifest carries one for every bundle. This is the smallest well-formed
/// layout that still describes something a host could allocate, and no physical
/// KV pool is ever created from it: the recurrent state is what this fixture
/// carries forward instead.
pub fn recurrent_kv_layout_v1() -> Result<BundleKvLayoutV1, BundleError> {
    BundleKvLayoutV1::new(
        id(RECURRENT_KV_LAYOUT_ID_V1)?,
        RECURRENT_KV_PAGE_BYTES_V1,
        RECURRENT_KV_ALIGNMENT_V1,
        RECURRENT_KV_PAGE_TOKENS_V1,
        RECURRENT_KV_BYTES_PER_TOKEN_V1,
        RECURRENT_KV_SEGMENTS_PER_PAGE_V1,
        RECURRENT_MAXIMUM_SEQUENCE_TOKENS_V1,
        BundleRankPlacementV1::RankLocal,
        BundleStateLifetimeV1::Transaction,
    )
}

/// The FP8 layout of [`RECURRENT_KV_FP8_LAYOUT_ID_V1`].
pub fn recurrent_kv_fp8_layout_v1() -> Result<BundleKvLayoutV1, BundleError> {
    BundleKvLayoutV1::new(
        id(RECURRENT_KV_FP8_LAYOUT_ID_V1)?,
        RECURRENT_KV_FP8_PAGE_BYTES_V1,
        RECURRENT_KV_FP8_PAGE_BYTES_V1,
        RECURRENT_KV_PAGE_TOKENS_V1,
        RECURRENT_KV_BYTES_PER_TOKEN_V1 / 2,
        RECURRENT_KV_SEGMENTS_PER_PAGE_V1,
        RECURRENT_MAXIMUM_SEQUENCE_TOKENS_V1,
        BundleRankPlacementV1::RankLocal,
        BundleStateLifetimeV1::Transaction,
    )
}

/// Builds one bounded opaque identifier this fixture declares.
pub(crate) fn id(value: &str) -> Result<BundleOpaqueId, BundleError> {
    BundleOpaqueId::new(value)
}

/// The four words the program image declares about itself.
///
/// A host never reads these: it resolves the view the module asks for and hands
/// back a bounded span. The module reads them and refuses a package whose image
/// disagrees with the state layout the same manifest declared.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramGeometryV1 {
    pub vocabulary: u32,
    pub maximum_tokens: u32,
    pub state_bytes: u32,
    pub lanes: u32,
}

impl ProgramGeometryV1 {
    /// The geometry this fixture emits into its own program image.
    #[must_use]
    pub const fn declared() -> Self {
        Self {
            vocabulary: RECURRENT_VOCABULARY_V1,
            maximum_tokens: RECURRENT_MAXIMUM_TOKENS_V1,
            state_bytes: RECURRENT_STATE_BYTES_V1 as u32,
            lanes: RECURRENT_LANES_V1 as u32,
        }
    }

    #[must_use]
    pub fn to_bytes(self) -> Vec<u8> {
        [
            self.vocabulary,
            self.maximum_tokens,
            self.state_bytes,
            self.lanes,
        ]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
    }

    /// Reads the geometry out of the exact bytes the view span holds.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != RECURRENT_GEOMETRY_BYTES_V1 as usize {
            return None;
        }
        let mut words = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]));
        Some(Self {
            vocabulary: words.next()?,
            maximum_tokens: words.next()?,
            state_bytes: words.next()?,
            lanes: words.next()?,
        })
    }
}

/// The lanes this fixture's program image seeds a fresh instance with.
#[must_use]
pub fn state_seed_lanes_v1() -> RecurrentLanesV1 {
    let mut lanes = [0_u32; RECURRENT_LANES_V1];
    for (index, lane) in lanes.iter_mut().enumerate() {
        *lane = 0x9E37_79B9_u32.wrapping_mul(index as u32 + 1);
    }
    lanes
}

/// The whole program image, for the one rank this fixture's topology has.
///
/// The layout is fixed: the geometry words, padding up to the state alignment,
/// then the seed lanes. Both named views span exactly what they describe, and
/// the emitter measures the file it wrote rather than trusting these numbers.
#[must_use]
pub fn program_image_bytes_v1() -> Vec<u8> {
    let mut image = vec![0_u8; RECURRENT_PROGRAM_IMAGE_BYTES_V1 as usize];
    let geometry = ProgramGeometryV1::declared().to_bytes();
    let geometry_start = RECURRENT_GEOMETRY_OFFSET_V1 as usize;
    image[geometry_start..geometry_start + geometry.len()].copy_from_slice(&geometry);
    let seed = crate::recurrent::lanes_to_bytes_v1(&state_seed_lanes_v1());
    let seed_start = RECURRENT_STATE_SEED_OFFSET_V1 as usize;
    image[seed_start..seed_start + seed.len()].copy_from_slice(&seed);
    image
}

/// The frontend vocabulary this fixture declares, as the bytes it ships.
///
/// This is the fixture's own declaration format, not the served OpenAI frontend
/// declaration: this bundle ships no tokenizer artifacts and is driven through
/// the conformance host rather than a transport. What it does carry is the one
/// thing execution depends on, a vocabulary of exactly
/// [`RECURRENT_VOCABULARY_V1`] tokens, which is also the width the program image
/// declares and the width of every row this fixture emits.
pub const RECURRENT_FRONTEND_DECLARATION_V1: &str = r#"{
  "schema": "fixture-recurrent-frontend-v1",
  "version": 1,
  "model": "fixture-recurrent-tp1",
  "vocabulary": {
    "size": 64,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "stop_token_ids": [1]
  },
  "tokenizer": {
    "kind": "fixture-identity-byte-v1",
    "bytes_per_token": 1,
    "maximum_total_tokens": 64
  }
}
"#;

#[cfg(test)]
mod tests {
    use crate::recurrent::{RECURRENT_STATE_BYTES_V1, lanes_from_bytes_v1};

    use super::{
        ProgramGeometryV1, RECURRENT_FRONTEND_DECLARATION_V1, RECURRENT_GEOMETRY_BYTES_V1,
        RECURRENT_GEOMETRY_OFFSET_V1, RECURRENT_MAXIMUM_SEQUENCE_TOKENS_V1,
        RECURRENT_MAXIMUM_TOKENS_V1, RECURRENT_PROGRAM_IMAGE_BYTES_V1,
        RECURRENT_STATE_SEED_OFFSET_V1, RECURRENT_VOCABULARY_V1, program_image_bytes_v1,
        state_seed_lanes_v1,
    };

    /// Both views span bytes the image actually holds, and carry what they say.
    #[test]
    fn the_program_image_carries_the_geometry_and_the_seed_its_views_name() {
        let image = program_image_bytes_v1();

        assert_eq!(image.len(), RECURRENT_PROGRAM_IMAGE_BYTES_V1 as usize);
        let geometry_start = RECURRENT_GEOMETRY_OFFSET_V1 as usize;
        let geometry_end = geometry_start + RECURRENT_GEOMETRY_BYTES_V1 as usize;
        assert_eq!(
            ProgramGeometryV1::from_bytes(&image[geometry_start..geometry_end]),
            Some(ProgramGeometryV1::declared())
        );
        let seed_start = RECURRENT_STATE_SEED_OFFSET_V1 as usize;
        let seed_end = seed_start + RECURRENT_STATE_BYTES_V1 as usize;
        assert!(seed_end <= image.len(), "the seed view must fit the image");
        assert_eq!(
            lanes_from_bytes_v1(&image[seed_start..seed_end]),
            Some(state_seed_lanes_v1())
        );
    }

    /// The declaration a deployment would read and the image the module reads
    /// have to agree, or the row a caller receives is not the declared width.
    #[test]
    fn the_declared_vocabulary_is_small_and_matches_the_program_image() {
        let declared = ProgramGeometryV1::declared();

        assert_eq!(declared.vocabulary, RECURRENT_VOCABULARY_V1);
        assert!(
            declared.vocabulary <= 1024,
            "this fixture declares a small vocabulary on purpose, and declared {}",
            declared.vocabulary
        );
        assert!(
            RECURRENT_FRONTEND_DECLARATION_V1
                .contains(&format!("\"size\": {RECURRENT_VOCABULARY_V1}")),
            "the shipped declaration must name the vocabulary the image declares"
        );
        assert!(
            RECURRENT_FRONTEND_DECLARATION_V1.contains(&format!(
                "\"maximum_total_tokens\": {RECURRENT_MAXIMUM_TOKENS_V1}"
            )),
            "the shipped declaration must name the prompt bound the image declares"
        );
        assert_eq!(
            u64::from(RECURRENT_MAXIMUM_TOKENS_V1),
            RECURRENT_MAXIMUM_SEQUENCE_TOKENS_V1,
            "the prompt bound and the declared sequence bound are one number"
        );
    }
}
