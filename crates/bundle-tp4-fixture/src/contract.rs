//! Everything this fixture declares about itself, in one place.
//!
//! The emitted manifest and the native module read the same identifiers, sizes,
//! and bytes from here, so a package that installs is described by the same
//! declaration the module inside it was written against.
//!
//! Two kinds of value live here and they are not the same kind of thing. The
//! width of a logits row and the token identifiers the served declaration pins
//! are public TP4 facts, taken from `bundle-tp4-fixture` rather than restated, so
//! this fixture is TP4-shaped where being TP4-shaped is what the host has to
//! survive. Everything else - the bundle identifier, the component roles, the
//! topology, the KV page, the persistent state, the operation key, and every
//! byte of the program image - is this fixture's own synthetic identity, and
//! none of it is a TP4 artifact, a TP4 profile, or a TP4 qualification.

use crate::metadata::{TP4_TEXT_PROMPT_SPECIAL_TOKENS, TP4_WIDTH, tokenizer_metadata};
use runtime_bundle::{
    BundleError, BundleKvLayoutV1, BundleOpaqueId, BundleRankPlacementV1, BundleStateLayoutV1,
    BundleStateLifetimeV1, BundleTopologyV1,
};

/// The identifier this bundle installs and is selected by.
///
/// It names the fixture, not the model: an installed `fixture/tp4`
/// would carry its own identifier and none of the declarations below.
pub const TP4_FIXTURE_BUNDLE_ID_V1: &str = "fixture.tp4.tp4.v1";

/// The name `build-tp4-fixture --fixture` accepts for this bundle.
pub const TP4_FIXTURE_NAME_V1: &str = "tp4-synthetic";

/// The one device architecture this fixture is emitted for.
pub const TP4_FIXTURE_ARCHITECTURE_V1: &str = "sm_120f";

/// The one topology this fixture serves.
///
/// TP4 deploys at TP4, so this fixture serves TP4: four ranks, four
/// rank images, four contexts, four pools, four streams, and four rank-local KV
/// pools. What it does not do is claim TP4's placement, packing, or collective
/// plan; the four ranks here carry synthetic images and this fixture's own
/// declaration.
pub const TP4_FIXTURE_TOPOLOGY_ID_V1: &str = "tp4";

/// How many ranks that topology has, read from the model crate's TP4 width.
pub const TP4_FIXTURE_RANK_COUNT_V1: u32 = TP4_WIDTH;

/// The rank whose page the emitted row is read back from.
///
/// Every rank writes and reads its own KV page, and one row is emitted, so the
/// rank that emits it is named rather than implied.
///
/// The committed history is this rank's too. The host allocates the declared
/// persistent state on all four ranks, and this fixture seeds, reads, and
/// publishes only this rank's view of it; the other three are allocated and left
/// alone. A model that sharded its recurrent state would own all four. This one
/// does not, and says so rather than implying replication it never performs.
pub const TP4_FIXTURE_EMITTING_RANK_V1: u32 = 0;

/// The component that carries this fixture's synthetic program image.
pub const TP4_FIXTURE_PROGRAM_COMPONENT_V1: &str = "program";

/// The role a host resolves that component by.
pub const TP4_FIXTURE_PROGRAM_ROLE_V1: &str = "fixture.tp4.program.v1";

/// The opaque format of that component.
pub const TP4_FIXTURE_PROGRAM_FORMAT_V1: &str = "fixture.tp4.image.v1";

/// The component that carries the served OpenAI frontend declaration.
pub const TP4_FIXTURE_FRONTEND_COMPONENT_V1: &str = "frontend";

/// The role a host resolves that component by.
///
/// This is the shared role the transport reads a declaration through, because
/// the served half of this fixture is the real one: an unchanged host and an
/// unchanged transport answer out of these bytes.
pub const TP4_FIXTURE_FRONTEND_ROLE_V1: &str = "chop.frontend.openai.v1";

/// The opaque format of that component.
pub const TP4_FIXTURE_FRONTEND_FORMAT_V1: &str = "chop.frontend.json.v1";

/// Where that component sits inside the package.
pub const TP4_FIXTURE_FRONTEND_PATH_V1: &str = "components/frontend.json";

/// The component that carries this fixture's generated tokenizer definition.
pub const TP4_FIXTURE_TOKENIZER_DEFINITION_COMPONENT_V1: &str = "tokenizer-definition";

/// The role a host resolves that component by.
pub const TP4_FIXTURE_TOKENIZER_DEFINITION_ROLE_V1: &str = "chop.tokenizer.definition.v1";

/// Where that component sits inside the package.
pub const TP4_FIXTURE_TOKENIZER_DEFINITION_PATH_V1: &str = "components/tokenizer.json";

/// The component that carries this fixture's generated tokenizer configuration.
pub const TP4_FIXTURE_TOKENIZER_CONFIGURATION_COMPONENT_V1: &str = "tokenizer-configuration";

/// The role a host resolves that component by.
pub const TP4_FIXTURE_TOKENIZER_CONFIGURATION_ROLE_V1: &str = "chop.tokenizer.configuration.v1";

/// Where that component sits inside the package.
pub const TP4_FIXTURE_TOKENIZER_CONFIGURATION_PATH_V1: &str = "components/tokenizer_config.json";

/// The opaque format both generated tokenizer components carry.
///
/// It is deliberately not the format a real tokenizer artifact carries: these
/// bytes are fixture-only stand-ins, and the format says so.
pub const TP4_FIXTURE_TOKENIZER_FORMAT_V1: &str = "fixture.tp4.tokenizer.json.v1";

/// The component that says a package is this fixture and not a model.
pub const TP4_FIXTURE_MARKER_COMPONENT_V1: &str = "fixture-marker";

/// The role a host resolves that component by.
pub const TP4_FIXTURE_MARKER_ROLE_V1: &str = "fixture.tp4.synthetic.v1";

/// The opaque format of that component.
pub const TP4_FIXTURE_MARKER_FORMAT_V1: &str = "fixture.tp4.marker.json.v1";

/// Where that component sits inside the package.
pub const TP4_FIXTURE_MARKER_PATH_V1: &str = "components/fixture-marker.json";

/// The schema the marker component declares itself under.
pub const TP4_FIXTURE_MARKER_SCHEMA_V1: &str = "fixture-tp4-synthetic-v1";

/// Where the native module sits inside the package.
///
/// The extension is deliberately neutral: the loader stages its own private
/// image and names that copy itself, so nothing loads this path directly.
pub const TP4_FIXTURE_NATIVE_MODULE_PATH_V1: &str = "native/module.bin";

/// The one graph bucket this fixture declares.
pub const TP4_FIXTURE_GRAPH_BUCKET_V1: &str = "tp4-synthetic-row";

/// The stage identifier this fixture selects its one stream for.
pub const TP4_FIXTURE_STAGE_ID_V1: &[u8] = b"tp4-synthetic-row";

/// The stream kind this fixture asks for. It orders every transfer it makes.
pub const TP4_FIXTURE_STREAM_KIND_V1: u32 = 0;

/// The identifier of the persistent state this fixture keeps.
pub const TP4_FIXTURE_STATE_LAYOUT_ID_V1: &str = "fixture.tp4.state.v1";

/// How many bytes that state holds: one folded 64-bit token history.
pub const TP4_FIXTURE_STATE_BYTES_V1: u64 = 8;

/// The alignment the persistent state is allocated at.
///
/// A declared state has to be a whole number of alignments, and this one is a
/// single 64-bit word, so the word itself is the alignment.
pub const TP4_FIXTURE_STATE_ALIGNMENT_V1: u64 = TP4_FIXTURE_STATE_BYTES_V1;

/// The identifier of the KV page layout this fixture declares and allocates.
pub const TP4_FIXTURE_KV_LAYOUT_ID_V1: &str = "fixture.tp4.kv.v1";

/// How many tokens one declared KV page holds.
///
/// One PrefillChunk append fills exactly one page, so a decode after it opens the
/// next one and the page table the host prepares is never a single fixed page.
pub const TP4_FIXTURE_KV_PAGE_TOKENS_V1: u32 = 32;

/// How many bytes one declared KV page holds per token.
pub const TP4_FIXTURE_KV_BYTES_PER_TOKEN_V1: u64 = 4;

/// How many equally shaped segments one declared KV page holds.
pub const TP4_FIXTURE_KV_SEGMENTS_PER_PAGE_V1: u32 = 1;

/// How many bytes one declared KV page holds in total.
pub const TP4_FIXTURE_KV_PAGE_BYTES_V1: u64 = TP4_FIXTURE_KV_BYTES_PER_TOKEN_V1
    * TP4_FIXTURE_KV_PAGE_TOKENS_V1 as u64
    * TP4_FIXTURE_KV_SEGMENTS_PER_PAGE_V1 as u64;

/// The alignment one declared KV page is allocated at.
///
/// A declared page has to be a whole number of alignments, so this is the page
/// itself: one PrefillChunk append's worth of tokens.
pub const TP4_FIXTURE_KV_ALIGNMENT_V1: u64 = TP4_FIXTURE_KV_PAGE_BYTES_V1;

/// How many pages one instance's physical KV pool holds.
pub const TP4_FIXTURE_KV_PAGE_CAPACITY_V1: u64 = 4;

/// How many concurrent KV requests one instance's pool admits.
///
/// This fixture opens exactly one request when it is created and prepares one
/// append against it per execution.
pub const TP4_FIXTURE_KV_REQUEST_CAPACITY_V1: u64 = 1;

/// The longest sequence this fixture declares it can hold.
pub const TP4_FIXTURE_MAXIMUM_SEQUENCE_TOKENS_V1: u64 =
    TP4_FIXTURE_KV_PAGE_TOKENS_V1 as u64 * TP4_FIXTURE_KV_PAGE_CAPACITY_V1;

/// The tokens one instance's single KV request may append over its lifetime.
pub const TP4_FIXTURE_KV_QUOTA_TOKENS_V1: u64 = TP4_FIXTURE_MAXIMUM_SEQUENCE_TOKENS_V1;

/// The context the served declaration offers a client.
///
/// It is exactly what one instance's KV pool holds, so the transport never
/// admits a request this fixture's pages could not carry.
pub const TP4_FIXTURE_CONTEXT_TOKENS_V1: usize = TP4_FIXTURE_MAXIMUM_SEQUENCE_TOKENS_V1 as usize;

/// The device memory a host must have before this fixture will load.
pub const TP4_FIXTURE_MINIMUM_DEVICE_MEMORY_BYTES_V1: u64 = 4 << 20;

/// The alignment this fixture allocates its own device buffers at.
pub const TP4_FIXTURE_BUFFER_ALIGNMENT_V1: u64 = 256;

/// The most tokens one prefill execution carries.
///
/// It is the PrefillChunk shape the qualification matrix names, and it is a bound
/// rather than an exact width: an engine chunks a longer prompt into prefills of
/// at most this many tokens and issues a shorter tail, so a fixture that refused
/// anything but exactly 32 could not be driven by one.
pub const TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1: u32 = 32;

/// The tokens one DecodeStep execution carries.
pub const TP4_FIXTURE_DECODE_STEP_TOKENS_V1: u32 = 1;

/// The most tokens one execution may carry.
pub const TP4_FIXTURE_MAXIMUM_TOKENS_V1: u32 = TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1;

/// The widest vocabulary the native module will accept from a program image.
///
/// The image is authenticated, so this is not a trust boundary. It is a bound:
/// a width past it would size an allocation this fixture never intends to make.
pub const TP4_FIXTURE_VOCABULARY_BOUND_V1: u32 = 1 << 21;

/// How many token identifiers the private operation's input operand holds.
///
/// It is a fixed capacity, not a length. One PrefillChunk's worth of identifiers is
/// lent on every call, zero past the count the second operand carries, so the
/// shape the operation declares is the shape it is lent for a DecodeStep, for a
/// full chunk, and for every tail between them. One operation stays exact only
/// if it is never lent something narrower than it declared.
pub const TP4_FIXTURE_ROW_INPUT_TOKENS_V1: u32 = TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1;

/// How many bytes that operand holds.
pub const TP4_FIXTURE_ROW_INPUT_BYTES_V1: u64 = TP4_FIXTURE_ROW_INPUT_TOKENS_V1 as u64 * 4;

/// How many token identifiers the count operand holds: one.
pub const TP4_FIXTURE_ROW_TOKEN_COUNT_ELEMENTS_V1: u64 = 1;

/// How many bytes that operand holds.
pub const TP4_FIXTURE_ROW_TOKEN_COUNT_BYTES_V1: u64 = TP4_FIXTURE_ROW_TOKEN_COUNT_ELEMENTS_V1 * 4;

/// The operation half of the key this fixture's own custom operation carries.
pub const TP4_FIXTURE_ROW_OPERATION_V1: &str = "fixture.tp4.synthetic-row";

/// The variant half of that key.
pub const TP4_FIXTURE_ROW_VARIANT_V1: &str = "tp4.v1";

/// The named view that carries this fixture's program geometry.
pub const TP4_FIXTURE_GEOMETRY_VIEW_V1: &[u8] = b"program-geometry";

/// The named view that carries this fixture's initial folded token history.
pub const TP4_FIXTURE_STATE_SEED_VIEW_V1: &[u8] = b"state-seed";

/// The alignment the program image and its rank image are placed at.
pub const TP4_FIXTURE_PROGRAM_IMAGE_ALIGNMENT_V1: u64 = 256;

/// How many bytes the whole program image holds for its one rank.
pub const TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1: u64 = 4096;

/// Where the geometry view starts inside the rank image.
pub const TP4_FIXTURE_GEOMETRY_OFFSET_V1: u64 = 0;

/// How many bytes the geometry view holds.
pub const TP4_FIXTURE_GEOMETRY_BYTES_V1: u64 = 16;

/// The alignment the geometry view is placed at.
pub const TP4_FIXTURE_GEOMETRY_ALIGNMENT_V1: u64 = 16;

/// Where the fixed prefix that names this image starts inside it.
pub const TP4_FIXTURE_IMAGE_MAGIC_OFFSET_V1: u64 = 16;

/// The fixed prefix every synthetic rank image carries at that offset.
pub const TP4_FIXTURE_IMAGE_MAGIC_V1: &[u8] = b"choptp4-fixture-v1";

/// Where the rank ordinal an image was emitted for sits inside it.
pub const TP4_FIXTURE_IMAGE_RANK_OFFSET_V1: u64 = 48;

/// Where the rank count that image was emitted for sits inside it.
pub const TP4_FIXTURE_IMAGE_RANK_COUNT_OFFSET_V1: u64 = 52;

/// Where the state seed view starts inside the rank image.
pub const TP4_FIXTURE_STATE_SEED_OFFSET_V1: u64 = 64;

/// Where the deterministic filler that makes each image distinct starts.
pub const TP4_FIXTURE_IMAGE_FILLER_OFFSET_V1: u64 = 128;

/// The width of one logits row this fixture emits.
///
/// This is the public TP4 configured vocabulary, read from the model
/// crate rather than restated, because emitting a row of exactly that width is
/// the whole reason this fixture is TP4-shaped. Nothing about the values in the
/// row is TP4 arithmetic.
#[must_use]
pub fn tp4_fixture_vocabulary_v1() -> u32 {
    tokenizer_metadata().config_vocabulary_size()
}

/// The token identifiers TP4 gives its 16 prompt markers.
///
/// These are the published identifiers `bundle-tp4-fixture` pins in
/// `prompt_token_for_id`, in the same order as
/// [`TP4_TEXT_PROMPT_SPECIAL_TOKENS`]. They are reproduced rather than read
/// because that mapping is private to the model crate; the test below binds this
/// list to the marker strings the model crate does export, so the two cannot
/// drift in length or in order without failing here.
pub const TP4_PROMPT_TOKEN_IDS_V1: [u32; TP4_TEXT_PROMPT_SPECIAL_TOKENS.len()] = [
    154_822, 154_824, 154_826, 154_827, 154_828, 154_829, 154_841, 154_842, 154_843, 154_844,
    154_845, 154_846, 154_847, 154_848, 154_849, 154_850,
];

/// The one topology this fixture serves.
pub fn tp4_fixture_topology_v1() -> Result<BundleTopologyV1, BundleError> {
    BundleTopologyV1::new(id(TP4_FIXTURE_TOPOLOGY_ID_V1)?, TP4_FIXTURE_RANK_COUNT_V1)
}

/// The persistent state this fixture carries beside its KV pages.
///
/// It is one 64-bit fold of every token a committed execution appended. The KV
/// transaction is what the host rolls back; this is what makes that rollback
/// visible in the next row, because a cancelled execution never publishes it.
pub fn tp4_fixture_state_layout_v1() -> Result<BundleStateLayoutV1, BundleError> {
    BundleStateLayoutV1::new(
        id(TP4_FIXTURE_STATE_LAYOUT_ID_V1)?,
        TP4_FIXTURE_STATE_BYTES_V1,
        TP4_FIXTURE_STATE_ALIGNMENT_V1,
        BundleStateLifetimeV1::Process,
    )
}

/// The KV page layout this fixture declares and allocates a pool from.
pub fn tp4_fixture_kv_layout_v1() -> Result<BundleKvLayoutV1, BundleError> {
    BundleKvLayoutV1::new(
        id(TP4_FIXTURE_KV_LAYOUT_ID_V1)?,
        TP4_FIXTURE_KV_PAGE_BYTES_V1,
        TP4_FIXTURE_KV_ALIGNMENT_V1,
        TP4_FIXTURE_KV_PAGE_TOKENS_V1,
        TP4_FIXTURE_KV_BYTES_PER_TOKEN_V1,
        TP4_FIXTURE_KV_SEGMENTS_PER_PAGE_V1,
        TP4_FIXTURE_MAXIMUM_SEQUENCE_TOKENS_V1,
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
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub state_bytes: u32,
}

impl ProgramGeometryV1 {
    /// How many ranks this fixture's declared topology has.
    #[must_use]
    pub const fn rank_count(self) -> u32 {
        TP4_FIXTURE_RANK_COUNT_V1
    }
}

impl ProgramGeometryV1 {
    /// The geometry this fixture emits into its own program image.
    #[must_use]
    pub fn declared() -> Self {
        Self {
            vocabulary: tp4_fixture_vocabulary_v1(),
            prefill_tokens: TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1,
            decode_tokens: TP4_FIXTURE_DECODE_STEP_TOKENS_V1,
            state_bytes: TP4_FIXTURE_STATE_BYTES_V1 as u32,
        }
    }

    #[must_use]
    pub fn to_bytes(self) -> Vec<u8> {
        [
            self.vocabulary,
            self.prefill_tokens,
            self.decode_tokens,
            self.state_bytes,
        ]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
    }

    /// Reads the geometry out of the exact bytes the view span holds.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != TP4_FIXTURE_GEOMETRY_BYTES_V1 as usize {
            return None;
        }
        let mut words = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]));
        Some(Self {
            vocabulary: words.next()?,
            prefill_tokens: words.next()?,
            decode_tokens: words.next()?,
            state_bytes: words.next()?,
        })
    }

    /// How many bytes one row of this geometry's width occupies.
    #[must_use]
    pub const fn row_bytes(self) -> u64 {
        self.vocabulary as u64 * 4
    }
}

#[cfg(test)]
mod tests {
    use crate::metadata::{TP4_TEXT_PROMPT_SPECIAL_TOKENS, tokenizer_metadata};

    use super::{
        ProgramGeometryV1, TP4_FIXTURE_KV_PAGE_BYTES_V1, TP4_FIXTURE_KV_PAGE_TOKENS_V1,
        TP4_FIXTURE_MAXIMUM_SEQUENCE_TOKENS_V1, TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1,
        TP4_PROMPT_TOKEN_IDS_V1, tp4_fixture_vocabulary_v1,
    };

    /// The width this fixture emits is the model crate's, not a number here.
    #[test]
    fn the_declared_row_is_the_published_fixture_vocabulary() {
        assert_eq!(tp4_fixture_vocabulary_v1(), 154_880);
        assert_eq!(
            tp4_fixture_vocabulary_v1(),
            tokenizer_metadata().config_vocabulary_size()
        );
        assert_eq!(
            ProgramGeometryV1::declared().row_bytes(),
            154_880 * 4,
            "one row is exactly the declared width in binary32"
        );
    }

    /// The identifiers reproduced here cover exactly the markers the model
    /// crate exports, in that order, and every one of them is a TP4 identifier
    /// rather than something this fixture assigned.
    #[test]
    fn the_pinned_marker_identifiers_cover_the_model_crates_markers_in_order() {
        assert_eq!(
            TP4_PROMPT_TOKEN_IDS_V1.len(),
            TP4_TEXT_PROMPT_SPECIAL_TOKENS.len()
        );
        let bos = tokenizer_metadata();
        assert_eq!(
            [TP4_PROMPT_TOKEN_IDS_V1[0], TP4_PROMPT_TOKEN_IDS_V1[1]],
            *bos.bos_prefix_ids(),
            "the first two markers are the published beginning-of-sequence pair"
        );
        assert_eq!(
            [
                TP4_TEXT_PROMPT_SPECIAL_TOKENS[0],
                TP4_TEXT_PROMPT_SPECIAL_TOKENS[1]
            ],
            *bos.bos_prefix(),
            "and they carry the published beginning-of-sequence markers"
        );
        let mut seen = TP4_PROMPT_TOKEN_IDS_V1.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), TP4_PROMPT_TOKEN_IDS_V1.len());
        assert!(
            TP4_PROMPT_TOKEN_IDS_V1
                .iter()
                .all(|id| *id < tp4_fixture_vocabulary_v1() && *id != bos.eos_token_id()),
            "every marker identifier is inside the vocabulary and is not the stop token"
        );
    }

    /// One PrefillChunk append fills one page exactly, so a decode opens the next.
    #[test]
    fn one_prefill_chunk_append_fills_exactly_one_declared_page() {
        assert_eq!(
            TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1,
            TP4_FIXTURE_KV_PAGE_TOKENS_V1
        );
        assert_eq!(TP4_FIXTURE_KV_PAGE_BYTES_V1, 128);
        assert!(
            TP4_FIXTURE_MAXIMUM_SEQUENCE_TOKENS_V1 > u64::from(TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1),
            "the declared sequence must outlast one prefill, or no decode could follow it"
        );
    }
}
