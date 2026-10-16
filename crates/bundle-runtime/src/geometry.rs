//! The serving policy and model dimensions shared bundle mechanics use, and
//! the rank and item counts an instance runs on.

/// Most tensor-parallel ranks of one instance: the bound of every per-rank
/// array. It equals the comm kernels' most ranks of a collective
/// (`kernel_core::contract::MAX_COMM_RANKS`), which every rank's pointer
/// array argument is sized by.
pub const MAX_RANKS: u32 = 8;

/// The rank counts the shared runtime serves.
pub const SUPPORTED_RANKS: [u32; 4] = [1, 2, 4, 8];

/// A supported tensor-parallel rank count: 1, 2, 4 or 8.
///
/// The count is a runtime value: a bundle reads it from the host's create
/// record (see [`crate::config::decode`]), and graph caches and rank runs are
/// checked against it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Ranks(u32);

impl Ranks {
    pub const TP1: Self = Self(1);
    pub const TP2: Self = Self(2);
    pub const TP4: Self = Self(4);
    pub const TP8: Self = Self(8);

    /// `count` when it is in [`SUPPORTED_RANKS`].
    pub const fn new(count: u32) -> Option<Self> {
        match count {
            1 | 2 | 4 | 8 => Some(Self(count)),
            _ => None,
        }
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// The count as a length, for per-rank arrays.
    pub const fn count(self) -> usize {
        self.0 as usize
    }

    /// Whether `rank` is one of these ranks.
    pub const fn contains(self, rank: u32) -> bool {
        rank < self.0
    }
}

/// Most items one step of any bundle may carry: the package maximum a
/// manifest may declare (`runtime_bundle::BundleBatchLimitsV1`), and the bound
/// of every per-item array a kernel variant is built for.
pub const MAX_ITEMS: u32 = 256;

/// The items an instance serves at once: its slot count, and the most items
/// in one step. A runtime value from the host's `--max-sequences`, carried in
/// the create record (see [`crate::config::CreateConfig::items`]) and bounded
/// by the bundle's [`BundleGeometry::DECODE_BATCH`].
///
/// Slot tables and per-slot buffers hold [`Items::get`] entries; per-row
/// arrays a graph bucket reads hold [`Items::capacity`], the largest bucket.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Items(u32);

impl Items {
    /// `count` items, from 1 to [`MAX_ITEMS`].
    pub const fn new(count: u32) -> Option<Self> {
        if count == 0 || count > MAX_ITEMS {
            None
        } else {
            Some(Self(count))
        }
    }

    /// `count` items when `G` runs that many.
    pub const fn within<G: BundleGeometry>(count: u32) -> Option<Self> {
        if count > G::DECODE_BATCH {
            None
        } else {
            Self::new(count)
        }
    }

    /// The most items `G` runs: what an instance serves when its host names
    /// no count. A geometry whose `DECODE_BATCH` is not a power of two from 1
    /// to [`MAX_ITEMS`] fails to compile wherever this is used, so the
    /// package maximum is always its own largest bucket.
    pub const fn package<G: BundleGeometry>() -> Self {
        const {
            assert!(
                G::DECODE_BATCH >= 1
                    && G::DECODE_BATCH <= MAX_ITEMS
                    && G::DECODE_BATCH.is_power_of_two(),
                "DECODE_BATCH must be a power of two from 1 to MAX_ITEMS"
            )
        };
        Self(G::DECODE_BATCH)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// The count as a length, for per-slot tables.
    pub const fn count(self) -> usize {
        self.0 as usize
    }

    /// Whether `slot` is one of these items' slots.
    pub const fn contains(self, slot: u32) -> bool {
        slot < self.0
    }

    /// The largest graph bucket: the count rounded up to a power of two, at
    /// most [`MAX_ITEMS`].
    pub const fn capacity(self) -> u32 {
        self.0.next_power_of_two()
    }

    /// The graph bucket that holds `live` items, or None for no items or
    /// more than this count.
    pub const fn bucket(self, live: u32) -> Option<u32> {
        if live == 0 || live > self.0 {
            None
        } else {
            Some(live.next_power_of_two())
        }
    }

    /// Every decode graph bucket, smallest first: the powers of two up to
    /// [`Self::capacity`].
    pub fn buckets(self) -> impl Iterator<Item = u32> {
        (0..=self.capacity().trailing_zeros()).map(|shift| 1 << shift)
    }
}

/// Validated model dimensions passed to shared runtime mechanics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelLimits {
    vocab: u32,
    context_tokens: u64,
    page_tokens: u64,
    max_pages: u32,
}

/// Invalid vocabulary, context, or block-table geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LimitsError {
    ZeroVocabulary,
    ZeroContext,
    ZeroPageSize,
    PartialPage,
    TooManyPages,
}

impl ModelLimits {
    pub const fn new(
        vocab: u32,
        context_tokens: u64,
        page_tokens: u64,
    ) -> Result<Self, LimitsError> {
        if vocab == 0 {
            return Err(LimitsError::ZeroVocabulary);
        }
        if context_tokens == 0 {
            return Err(LimitsError::ZeroContext);
        }
        if page_tokens == 0 {
            return Err(LimitsError::ZeroPageSize);
        }
        if context_tokens % page_tokens != 0 {
            return Err(LimitsError::PartialPage);
        }
        let pages = context_tokens / page_tokens;
        if pages > u32::MAX as u64 {
            return Err(LimitsError::TooManyPages);
        }
        Ok(Self {
            vocab,
            context_tokens,
            page_tokens,
            max_pages: pages as u32,
        })
    }

    /// The full vocabulary: logits row width and exclusive token id bound.
    pub const fn vocab(self) -> u32 {
        self.vocab
    }

    /// The geometry's maximum context, which sizes block-table rows.
    /// The create record may select a smaller served context.
    pub const fn context_tokens(self) -> u64 {
        self.context_tokens
    }

    pub const fn page_tokens(self) -> u64 {
        self.page_tokens
    }

    /// Entries of one item's block-table row.
    pub const fn max_pages(self) -> u32 {
        self.max_pages
    }
}

/// A bundle's serving policy. Model dimensions come from [`ModelLimits`];
/// rank and item counts come from runtime [`Ranks`] and [`Items`].
pub trait BundleGeometry {
    /// Tokens per serving KV page.
    const PAGE_TOKENS: u64;
    /// Most tokens in one step's prefill phase.
    const PREFILL_TOKENS: u32;
    /// Most items in one step the bundle's kernels and buffers run: its
    /// package maximum, a power of two up to [`MAX_ITEMS`] (checked at
    /// compile time by [`Items::package`]). An instance serves a runtime
    /// [`Items`] count up to it.
    const DECODE_BATCH: u32;
    /// Most rows one verify item may carry.
    const MAX_VERIFY_ROWS: u32;
    /// Most decode-phase rows in one verify step, and the largest verify
    /// graph token bucket.
    const VERIFY_TOKENS: u32 = Self::DECODE_BATCH;
    /// Optional narrow-instance limit: an instance serving at most this
    /// many items uses this many verify rows. Wider instances use the
    /// full verify limit, so AR siblings leave room for drafted rows.
    const NARROW_VERIFY_TOKENS: Option<u32> = None;
    /// The instance's narrow limit. A bundle may opt into its declared
    /// wider geometry at process start, before allocating any graph inputs.
    fn narrow_verify_tokens() -> Option<u32> {
        Self::NARROW_VERIFY_TOKENS
    }
    /// Prefill token arrays are sized in whole multiples of this many tokens,
    /// so a kernel that reads a padded tail stays inside its own array.
    const TOKEN_GRANULE: u32;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{BundleGeometry, Items, MAX_ITEMS, MAX_RANKS, Ranks, SUPPORTED_RANKS};

    pub(crate) const LIMITS: super::ModelLimits =
        match super::ModelLimits::new(152_576, 1_048_576, 64) {
            Ok(limits) => limits,
            Err(_) => panic!("invalid model limits"),
        };

    #[test]
    fn model_limits_reject_invalid_geometry() {
        use super::{LimitsError, ModelLimits};
        for (vocab, context, page, error) in [
            (0, 64, 64, LimitsError::ZeroVocabulary),
            (1, 0, 64, LimitsError::ZeroContext),
            (1, 64, 0, LimitsError::ZeroPageSize),
            (1, 100, 64, LimitsError::PartialPage),
            (1, 64 * (u32::MAX as u64 + 2), 64, LimitsError::TooManyPages),
        ] {
            assert_eq!(ModelLimits::new(vocab, context, page), Err(error));
        }
        let largest = ModelLimits::new(1, 64 * u32::MAX as u64, 64).unwrap();
        assert_eq!(largest.max_pages(), u32::MAX);
        assert_eq!(largest.context_tokens(), 64 * u32::MAX as u64);
        assert_eq!(largest.vocab(), 1);
    }

    /// Step limits for the shared crate's synthetic tests. The name
    /// predates the runtime rank count; the geometry holds no ranks.
    pub(crate) struct Tp4;

    impl BundleGeometry for Tp4 {
        const PAGE_TOKENS: u64 = 64;
        const PREFILL_TOKENS: u32 = 8192;
        const DECODE_BATCH: u32 = 32;
        const MAX_VERIFY_ROWS: u32 = 8;
        const TOKEN_GRANULE: u32 = 256;
    }

    /// Tp4 whose verify steps carry eight rows for every item.
    pub(crate) struct Wide;

    impl BundleGeometry for Wide {
        const PAGE_TOKENS: u64 = Tp4::PAGE_TOKENS;
        const PREFILL_TOKENS: u32 = Tp4::PREFILL_TOKENS;
        const DECODE_BATCH: u32 = Tp4::DECODE_BATCH;
        const MAX_VERIFY_ROWS: u32 = Tp4::MAX_VERIFY_ROWS;
        const VERIFY_TOKENS: u32 = 256;
        const TOKEN_GRANULE: u32 = Tp4::TOKEN_GRANULE;
    }

    /// Tp4 whose package runs the most items any bundle may.
    pub(crate) struct Max;

    impl BundleGeometry for Max {
        const PAGE_TOKENS: u64 = Tp4::PAGE_TOKENS;
        const PREFILL_TOKENS: u32 = Tp4::PREFILL_TOKENS;
        const DECODE_BATCH: u32 = MAX_ITEMS;
        const MAX_VERIFY_ROWS: u32 = Tp4::MAX_VERIFY_ROWS;
        const TOKEN_GRANULE: u32 = Tp4::TOKEN_GRANULE;
    }

    /// Max whose verify steps carry fewer rows than its largest buckets.
    pub(crate) struct Narrow;

    impl BundleGeometry for Narrow {
        const PAGE_TOKENS: u64 = Tp4::PAGE_TOKENS;
        const PREFILL_TOKENS: u32 = Tp4::PREFILL_TOKENS;
        const DECODE_BATCH: u32 = MAX_ITEMS;
        const MAX_VERIFY_ROWS: u32 = Tp4::MAX_VERIFY_ROWS;
        const VERIFY_TOKENS: u32 = 16;
        const TOKEN_GRANULE: u32 = Tp4::TOKEN_GRANULE;
    }

    /// Every item count an instance may serve.
    pub(crate) fn every_item_count() -> impl Iterator<Item = Items> {
        (1..=MAX_ITEMS).map(|count| Items::new(count).unwrap())
    }

    #[test]
    fn item_counts_are_bounded_by_the_package() {
        assert_eq!(Items::new(0), None);
        assert_eq!(Items::new(MAX_ITEMS + 1), None);
        assert_eq!(Items::package::<Tp4>().get(), 32);
        assert_eq!(Items::package::<Max>().get(), MAX_ITEMS);
        assert_eq!(Items::within::<Tp4>(33), None);
        assert_eq!(Items::within::<Tp4>(0), None);
        assert_eq!(Items::within::<Max>(MAX_ITEMS + 1), None);
        for items in every_item_count() {
            let count = items.get();
            assert_eq!(Items::within::<Max>(count), Some(items));
            assert_eq!(Items::within::<Tp4>(count).is_some(), count <= 32);
            assert_eq!(items.count(), count as usize);
            assert!(items.contains(count - 1) && !items.contains(count));
            let capacity = items.capacity();
            assert!(capacity.is_power_of_two() && capacity >= count && capacity < 2 * count);
            assert!(capacity <= MAX_ITEMS);
            let buckets = items.buckets().collect::<Vec<_>>();
            assert_eq!(buckets.first(), Some(&1));
            assert_eq!(buckets.last(), Some(&capacity));
            assert!(buckets.windows(2).all(|pair| pair[1] == 2 * pair[0]));
            // Every live count up to the item count has a bucket, and the
            // bucket is one of the listed ones; none above it.
            for live in 1..=count {
                let bucket = items.bucket(live).unwrap();
                assert!(bucket >= live && buckets.contains(&bucket));
            }
            assert_eq!(items.bucket(0), None);
            assert_eq!(items.bucket(count + 1), None);
        }
        assert!(Items::package::<Tp4>().buckets().eq([1, 2, 4, 8, 16, 32]));
        assert!(
            Items::package::<Max>()
                .buckets()
                .eq([1, 2, 4, 8, 16, 32, 64, 128, 256])
        );
        assert!(
            Items::new(48)
                .unwrap()
                .buckets()
                .eq([1, 2, 4, 8, 16, 32, 64])
        );
    }

    #[test]
    fn block_table_rows_cover_the_context() {
        assert_eq!(LIMITS.max_pages(), 16_384);
    }

    #[test]
    fn rank_counts_are_the_supported_set() {
        let accepted = (0..=2 * MAX_RANKS)
            .filter_map(Ranks::new)
            .map(Ranks::get)
            .collect::<Vec<_>>();
        assert_eq!(accepted, SUPPORTED_RANKS);
        assert_eq!(SUPPORTED_RANKS.iter().max(), Some(&MAX_RANKS));
        for (ranks, count) in [Ranks::TP1, Ranks::TP2, Ranks::TP4, Ranks::TP8]
            .into_iter()
            .zip(SUPPORTED_RANKS)
        {
            assert_eq!(Ranks::new(count), Some(ranks));
            assert_eq!(ranks.count(), count as usize);
            assert!(ranks.contains(count - 1) && !ranks.contains(count));
        }
    }
}
