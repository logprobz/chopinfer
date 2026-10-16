//! Media inputs: encoder records, placeholder injection and the media arena.
//!
//! The executor runs each media item's encoder into rows of a device arena
//! and overwrites the item's placeholder rows with them. The engine owns the
//! arena the way it owns KV pages: it allocates pages per item, keeps
//! encoded items resident by content key as an encoder cache, and pins an
//! item while a request's prefill still needs its rows.

use std::collections::BTreeMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};

/// The most encoders one configuration declares.
pub const MAX_MEDIA_ENCODERS: usize = 8;

/// A media item's payload bytes, shared by every copy of the item from
/// preparation to retirement. The owner (for example bytes held under the
/// host's memory budget) drops with the last copy.
#[derive(Clone)]
pub struct MediaPayloadV1(Arc<dyn AsRef<[u8]> + Send + Sync>);

impl MediaPayloadV1 {
    /// The bytes `owner` holds, kept alive until the last copy drops.
    #[must_use]
    pub fn new(owner: impl AsRef<[u8]> + Send + Sync + 'static) -> Self {
        Self(Arc::new(owner))
    }
}

impl From<Vec<u8>> for MediaPayloadV1 {
    fn from(bytes: Vec<u8>) -> Self {
        Self::new(bytes)
    }
}

impl AsRef<[u8]> for MediaPayloadV1 {
    fn as_ref(&self) -> &[u8] {
        (*self.0).as_ref()
    }
}

impl std::ops::Deref for MediaPayloadV1 {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_ref()
    }
}

impl PartialEq for MediaPayloadV1 {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl Eq for MediaPayloadV1 {}

impl std::fmt::Debug for MediaPayloadV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MediaPayloadV1({} bytes)", self.len())
    }
}

/// One preprocessed media item of a request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaInputV1 {
    /// Content key of the recipe and payload. `EngineRequest::with_media`
    /// replaces it with [`MediaInputV1::identity`], which both the prefix
    /// cache and the encoder cache use.
    pub key: [u8; 32],
    pub encoder: u16,
    pub geometry: [u32; 4],
    pub output_rows: u32,
    pub payload: MediaPayloadV1,
}

impl MediaInputV1 {
    /// Everything the encoder consumes: the content key with the encoder,
    /// geometry and output rows. Equal payloads can differ in geometry (a
    /// 64x128 and a 128x64 image can pack to the same patch bytes), so a key
    /// without them could serve one item's rows or KV for another. The
    /// encoder index names one encoder of the process's single bundle.
    #[must_use]
    pub fn identity(&self) -> [u8; 32] {
        let mut sha = Sha256::new();
        sha.update(b"chopinfer.media-identity.v2");
        sha.update(self.key);
        sha.update(self.encoder.to_le_bytes());
        for value in self.geometry {
            sha.update(value.to_le_bytes());
        }
        sha.update(self.output_rows.to_le_bytes());
        sha.finalize().into()
    }
}

/// One contiguous placeholder run of a request's prompt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaSpanV1 {
    pub media: u16,
    pub prompt_row: u32,
    pub rows: u32,
    pub media_row: u32,
}

/// An encoder run a batch item asks the executor for.
///
/// Media row `r` lands at arena row `pages[r / page_rows] * page_rows +
/// r % page_rows`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaEncodeV1 {
    pub encoder: u16,
    pub geometry: [u32; 4],
    pub output_rows: u32,
    pub pages: Vec<u64>,
    pub payload: MediaPayloadV1,
}

/// Input rows of a batch item that take arena rows instead of token
/// embeddings. A run never crosses an arena page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaInjectV1 {
    pub item_row: u32,
    pub rows: u32,
    pub arena_row: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MediaEncoderLimitsV1 {
    maximum_payload_bytes: u64,
    cost_per_row_milli: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaConfigError {
    ZeroArenaPages,
    ZeroPageRows,
    ZeroPayloadBytes,
    TooManyEncoders,
}

/// The arena geometry and encoder limits an executor declares.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaConfigV1 {
    arena_pages: usize,
    page_rows: u32,
    encoders: [MediaEncoderLimitsV1; MAX_MEDIA_ENCODERS],
    encoder_count: usize,
}

impl MediaConfigV1 {
    pub const fn new(arena_pages: usize, page_rows: u32) -> Result<Self, MediaConfigError> {
        if arena_pages == 0 {
            return Err(MediaConfigError::ZeroArenaPages);
        }
        if page_rows == 0 {
            return Err(MediaConfigError::ZeroPageRows);
        }
        Ok(Self {
            arena_pages,
            page_rows,
            encoders: [MediaEncoderLimitsV1 {
                maximum_payload_bytes: 0,
                cost_per_row_milli: 0,
            }; MAX_MEDIA_ENCODERS],
            encoder_count: 0,
        })
    }

    /// Declares the next encoder index. Each encoded row costs
    /// `cost_per_row_milli / 1000` tokens of the step budget.
    pub const fn with_encoder(
        mut self,
        maximum_payload_bytes: u64,
        cost_per_row_milli: u32,
    ) -> Result<Self, MediaConfigError> {
        if maximum_payload_bytes == 0 {
            return Err(MediaConfigError::ZeroPayloadBytes);
        }
        if self.encoder_count == MAX_MEDIA_ENCODERS {
            return Err(MediaConfigError::TooManyEncoders);
        }
        self.encoders[self.encoder_count] = MediaEncoderLimitsV1 {
            maximum_payload_bytes,
            cost_per_row_milli,
        };
        self.encoder_count += 1;
        Ok(self)
    }

    #[must_use]
    pub const fn arena_pages(self) -> usize {
        self.arena_pages
    }

    #[must_use]
    pub const fn page_rows(self) -> u32 {
        self.page_rows
    }

    #[must_use]
    pub const fn encoder_count(self) -> usize {
        self.encoder_count
    }

    pub(crate) const fn pages_for(self, rows: u32) -> usize {
        rows.div_ceil(self.page_rows) as usize
    }

    /// Token-equivalent budget cost of encoding `rows` rows.
    pub(crate) fn encode_cost(self, encoder: u16, rows: u32) -> usize {
        let milli =
            u64::from(rows) * u64::from(self.encoders[usize::from(encoder)].cost_per_row_milli);
        usize::try_from(milli.div_ceil(1000)).unwrap_or(usize::MAX)
    }

    /// Checks one request's items against the declared encoders and returns
    /// the most arena pages one prefill step of it can cover: the pages of
    /// the items overlapping any one `granule` of prompt rows, the smallest
    /// chunk a prefill takes, counting each content key once. Items stream
    /// through the arena chunk by chunk, so only this must fit it.
    pub(crate) fn step_pages(
        self,
        media: &[MediaInputV1],
        spans: &[MediaSpanV1],
        granule: usize,
    ) -> Result<usize, MediaRequestError> {
        for item in media {
            let Some(limits) = self.encoders[..self.encoder_count].get(usize::from(item.encoder))
            else {
                return Err(MediaRequestError::UnknownEncoder);
            };
            if item.payload.len() as u64 > limits.maximum_payload_bytes {
                return Err(MediaRequestError::PayloadTooLarge);
            }
        }
        // Each span covers a run of granule windows; sweep the windows in
        // order, keeping the pages of the distinct items open at each.
        let granule = granule.max(1);
        let mut events = Vec::with_capacity(spans.len() * 2);
        for span in spans {
            let item = media
                .get(usize::from(span.media))
                .ok_or(MediaRequestError::UnknownMedia)?;
            let first = span.prompt_row as usize / granule;
            let last = (span.prompt_row as usize + span.rows as usize).saturating_sub(1) / granule;
            // Closings sort before openings at the same window.
            events.push((first, 1u8, item.key, item.output_rows));
            events.push((last + 1, 0u8, item.key, item.output_rows));
        }
        events.sort_unstable_by_key(|&(window, opens, ..)| (window, opens));
        let mut open = BTreeMap::<[u8; 32], u32>::new();
        let (mut pages, mut most) = (0usize, 0usize);
        for (_, opens, key, rows) in events {
            let count = open.entry(key).or_default();
            if opens == 1 {
                *count += 1;
                if *count == 1 {
                    pages += self.pages_for(rows);
                    most = most.max(pages);
                }
            } else {
                *count -= 1;
                if *count == 0 {
                    pages -= self.pages_for(rows);
                }
            }
        }
        Ok(most)
    }

    /// The arena pages of the distinct items `media` that a prefill chunk of
    /// prompt rows `start..end` covers, resident or not: the step pins each.
    pub(crate) fn chunk_pages(
        self,
        media: &[MediaInputV1],
        spans: &[MediaSpanV1],
        start: usize,
        end: usize,
    ) -> usize {
        let mut covered = std::collections::HashSet::<[u8; 32]>::new();
        let mut pages = 0usize;
        for span in spans {
            let (from, to) = (
                span.prompt_row as usize,
                (span.prompt_row + span.rows) as usize,
            );
            if to <= start || from >= end {
                continue;
            }
            let Some(item) = media.get(usize::from(span.media)) else {
                continue;
            };
            if covered.insert(item.key) {
                pages = pages.saturating_add(self.pages_for(item.output_rows));
            }
        }
        pages
    }
}

/// Why a request's media or spans were refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaRequestError {
    /// The engine was configured without media support.
    Unsupported,
    TooManyItems,
    EmptyPayload,
    ZeroOutputRows,
    UnknownEncoder,
    PayloadTooLarge,
    UnknownMedia,
    EmptySpan,
    SpanOutsideItem,
    SpanOutsidePrompt,
    SpansOutOfOrder,
}

/// Checks the media items and spans of a request with `prompt_tokens` tokens.
pub(crate) fn validate_media(
    prompt_tokens: usize,
    media: &[MediaInputV1],
    spans: &[MediaSpanV1],
) -> Result<(), MediaRequestError> {
    if media.len() > usize::from(u16::MAX) + 1 {
        return Err(MediaRequestError::TooManyItems);
    }
    for item in media {
        if item.payload.is_empty() {
            return Err(MediaRequestError::EmptyPayload);
        }
        if item.output_rows == 0 {
            return Err(MediaRequestError::ZeroOutputRows);
        }
    }
    let mut previous_end = 0u64;
    for span in spans {
        let item = media
            .get(usize::from(span.media))
            .ok_or(MediaRequestError::UnknownMedia)?;
        if span.rows == 0 {
            return Err(MediaRequestError::EmptySpan);
        }
        if u64::from(span.media_row) + u64::from(span.rows) > u64::from(item.output_rows) {
            return Err(MediaRequestError::SpanOutsideItem);
        }
        let start = u64::from(span.prompt_row);
        let end = start + u64::from(span.rows);
        if start < previous_end {
            return Err(MediaRequestError::SpansOutOfOrder);
        }
        if end > prompt_tokens as u64 {
            return Err(MediaRequestError::SpanOutsidePrompt);
        }
        previous_end = end;
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ArenaEntry {
    pages: Vec<u64>,
    pins: u32,
    last_use: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArenaChange {
    Allocated([u8; 32]),
    Pinned([u8; 32]),
}

/// The result of pinning one item for a step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ArenaPlacement {
    pub(crate) pages: Vec<u64>,
    /// The item was not resident and must be encoded this step.
    pub(crate) encode: bool,
    /// Unpinned entries evicted to make room.
    pub(crate) evicted: usize,
}

/// Host ownership of the executor's media arena pages.
///
/// Pins last one step: a step pins every item its prefill chunks cover and
/// releases them when it commits, so an item a later chunk still covers may
/// be evicted in between and encoded again. Eviction takes unpinned entries
/// no live prefill still needs first, then the least recently used.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaArenaV1 {
    page_rows: u32,
    capacity: usize,
    free: Vec<u64>,
    entries: BTreeMap<[u8; 32], ArenaEntry>,
    /// Live prefills that have yet to pass each item, by content key.
    wanted: BTreeMap<[u8; 32], u32>,
    tick: u64,
    step: Vec<ArenaChange>,
}

impl MediaArenaV1 {
    #[must_use]
    pub fn new(config: MediaConfigV1) -> Self {
        Self {
            page_rows: config.page_rows,
            capacity: config.arena_pages,
            free: (0..config.arena_pages as u64).rev().collect(),
            entries: BTreeMap::new(),
            wanted: BTreeMap::new(),
            tick: 0,
            step: Vec::new(),
        }
    }

    #[must_use]
    pub fn resident(&self, key: &[u8; 32]) -> bool {
        self.entries.contains_key(key)
    }

    #[must_use]
    pub fn free_pages(&self) -> usize {
        self.free.len()
    }

    pub(crate) fn pages(&self, key: &[u8; 32]) -> Option<&[u64]> {
        self.entries.get(key).map(|entry| entry.pages.as_slice())
    }

    /// Pins an item for the current step, allocating its pages when it is not
    /// resident. `None` means the unpinned entries cannot free enough pages.
    pub(crate) fn pin(&mut self, key: [u8; 32], output_rows: u32) -> Option<ArenaPlacement> {
        self.tick += 1;
        let tick = self.tick;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.pins += 1;
            entry.last_use = tick;
            self.step.push(ArenaChange::Pinned(key));
            return Some(ArenaPlacement {
                pages: entry.pages.clone(),
                encode: false,
                evicted: 0,
            });
        }
        let needed = output_rows.div_ceil(self.page_rows) as usize;
        let mut evicted = 0;
        while self.free.len() < needed {
            let victim = self
                .entries
                .iter()
                .filter(|(_, entry)| entry.pins == 0)
                .min_by_key(|(key, entry)| (self.wanted.contains_key(*key), entry.last_use))
                .map(|(key, _)| *key)?;
            let entry = self.entries.remove(&victim).expect("victim is resident");
            self.free.extend(entry.pages.into_iter().rev());
            evicted += 1;
        }
        let pages: Vec<u64> = (0..needed)
            .map(|_| self.free.pop().expect("pages are free"))
            .collect();
        self.entries.insert(
            key,
            ArenaEntry {
                pages: pages.clone(),
                pins: 1,
                last_use: tick,
            },
        );
        self.step.push(ArenaChange::Allocated(key));
        Some(ArenaPlacement {
            pages,
            encode: true,
            evicted,
        })
    }

    /// Keeps the current step's pins and allocations.
    pub(crate) fn commit_step(&mut self) {
        self.step.clear();
    }

    /// Frees the current step's allocations and drops its pins. Entries it
    /// evicted stay evicted.
    pub(crate) fn rollback_step(&mut self) {
        while let Some(change) = self.step.pop() {
            match change {
                ArenaChange::Allocated(key) => {
                    if let Some(entry) = self.entries.remove(&key) {
                        self.free.extend(entry.pages.into_iter().rev());
                    }
                }
                ArenaChange::Pinned(key) => {
                    if let Some(entry) = self.entries.get_mut(&key) {
                        entry.pins = entry.pins.saturating_sub(1);
                    }
                }
            }
        }
    }

    /// Records that a live prefill has yet to pass the item `key`.
    pub(crate) fn want(&mut self, key: [u8; 32]) {
        *self.wanted.entry(key).or_default() += 1;
    }

    /// Records that a live prefill passed the item `key`, or retired.
    pub(crate) fn unwant(&mut self, key: &[u8; 32]) -> bool {
        match self.wanted.get_mut(key) {
            Some(count) if *count > 1 => {
                *count -= 1;
                true
            }
            Some(_) => {
                self.wanted.remove(key);
                true
            }
            None => false,
        }
    }

    /// Whether the wanted counts equal `expected`, the counts live requests
    /// hold.
    pub(crate) fn wanted_matches(&self, expected: &BTreeMap<[u8; 32], u32>) -> bool {
        self.wanted == *expected
    }

    /// Releases one committed pin.
    pub(crate) fn unpin(&mut self, key: &[u8; 32]) -> bool {
        match self.entries.get_mut(key) {
            Some(entry) if entry.pins > 0 => {
                entry.pins -= 1;
                true
            }
            _ => false,
        }
    }

    /// Checks page ownership and that each entry's pins equal `expected`,
    /// the pins live requests hold, with zero for keys it does not name.
    pub(crate) fn audit(&self, expected: &BTreeMap<[u8; 32], u32>) -> bool {
        let mut owner = vec![false; self.capacity];
        let pages = self
            .free
            .iter()
            .chain(self.entries.values().flat_map(|entry| entry.pages.iter()));
        for page in pages {
            let Some(slot) = owner.get_mut(*page as usize) else {
                return false;
            };
            if *slot {
                return false;
            }
            *slot = true;
        }
        self.step.is_empty()
            && owner.iter().all(|owned| *owned)
            && self
                .entries
                .iter()
                .all(|(key, entry)| expected.get(key).copied().unwrap_or(0) == entry.pins)
            && expected.keys().all(|key| self.entries.contains_key(key))
    }
}

/// Splits `rows` media rows from `media_row` into page-bounded inject runs.
pub(crate) fn inject_runs(
    pages: &[u64],
    page_rows: u32,
    item_row: u32,
    media_row: u32,
    rows: u32,
    runs: &mut Vec<MediaInjectV1>,
) {
    let mut done = 0;
    while done < rows {
        let row = media_row + done;
        let page = pages[(row / page_rows) as usize];
        let offset = row % page_rows;
        let count = (rows - done).min(page_rows - offset);
        runs.push(MediaInjectV1 {
            item_row: item_row + done,
            rows: count,
            arena_row: page * u64::from(page_rows) + u64::from(offset),
        });
        done += count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn arena(pages: usize) -> MediaArenaV1 {
        MediaArenaV1::new(MediaConfigV1::new(pages, 4).unwrap())
    }

    fn pins(entries: &[(u8, u32)]) -> BTreeMap<[u8; 32], u32> {
        entries
            .iter()
            .map(|(value, pins)| (key(*value), *pins))
            .collect()
    }

    #[test]
    fn resident_items_are_hits_and_unpinned_items_are_evicted_oldest_first() {
        let mut arena = arena(4);
        let first = arena.pin(key(1), 8).unwrap();
        assert!(first.encode);
        assert_eq!(first.pages.len(), 2);
        let second = arena.pin(key(2), 5).unwrap();
        assert_eq!(second.pages.len(), 2);
        arena.commit_step();
        assert!(arena.audit(&pins(&[(1, 1), (2, 1)])));

        let hit = arena.pin(key(1), 8).unwrap();
        assert!(!hit.encode);
        assert_eq!(hit.pages, first.pages);
        arena.commit_step();
        // Every page is pinned, so a third item finds no room.
        assert!(arena.pin(key(3), 1).is_none());

        assert!(arena.unpin(&key(1)));
        assert!(arena.unpin(&key(1)));
        assert!(arena.unpin(&key(2)));
        arena.commit_step();
        // Key 2 was used less recently than key 1.
        let third = arena.pin(key(3), 1).unwrap();
        assert_eq!(third.evicted, 1);
        assert!(!arena.resident(&key(2)));
        assert!(arena.resident(&key(1)));
        arena.commit_step();
        assert!(arena.audit(&pins(&[(1, 0), (3, 1)])));
    }

    #[test]
    fn rollback_frees_allocations_and_drops_pins() {
        let mut arena = arena(4);
        arena.pin(key(1), 4).unwrap();
        arena.commit_step();
        arena.pin(key(1), 4).unwrap();
        arena.pin(key(2), 8).unwrap();
        arena.rollback_step();
        assert!(!arena.resident(&key(2)));
        assert_eq!(arena.free_pages(), 3);
        assert!(arena.audit(&pins(&[(1, 1)])));
    }

    #[test]
    fn inject_runs_split_at_page_boundaries() {
        let mut runs = Vec::new();
        inject_runs(&[7, 2, 9], 4, 10, 3, 6, &mut runs);
        assert_eq!(
            runs,
            [
                MediaInjectV1 {
                    item_row: 10,
                    rows: 1,
                    arena_row: 31
                },
                MediaInjectV1 {
                    item_row: 11,
                    rows: 4,
                    arena_row: 8
                },
                MediaInjectV1 {
                    item_row: 15,
                    rows: 1,
                    arena_row: 36
                },
            ]
        );
    }

    #[test]
    fn spans_are_checked_against_items_and_the_prompt() {
        let item = MediaInputV1 {
            key: key(1),
            encoder: 0,
            geometry: [1, 2, 2, 0],
            output_rows: 4,
            payload: vec![0u8; 16].into(),
        };
        let span = |media, prompt_row, rows, media_row| MediaSpanV1 {
            media,
            prompt_row,
            rows,
            media_row,
        };
        let media = std::slice::from_ref(&item);
        assert_eq!(
            validate_media(8, media, &[span(0, 2, 2, 0), span(0, 5, 2, 2)]),
            Ok(())
        );
        for (spans, error) in [
            (vec![span(1, 0, 1, 0)], MediaRequestError::UnknownMedia),
            (vec![span(0, 0, 0, 0)], MediaRequestError::EmptySpan),
            (vec![span(0, 0, 3, 2)], MediaRequestError::SpanOutsideItem),
            (vec![span(0, 7, 2, 0)], MediaRequestError::SpanOutsidePrompt),
            (
                vec![span(0, 2, 2, 0), span(0, 3, 1, 2)],
                MediaRequestError::SpansOutOfOrder,
            ),
        ] {
            assert_eq!(validate_media(8, media, &spans), Err(error));
        }
        let config = MediaConfigV1::new(4, 4)
            .unwrap()
            .with_encoder(16, 1500)
            .unwrap();
        let twice = [item.clone(), item.clone()];
        let spans = [span(0, 0, 4, 0), span(1, 4, 4, 0)];
        assert_eq!(config.step_pages(&twice, &spans, 8), Ok(1));
        assert_eq!(config.encode_cost(0, 3), 5);
        let large = MediaInputV1 {
            payload: vec![0u8; 17].into(),
            ..item
        };
        assert_eq!(
            config.step_pages(&[large], &[span(0, 0, 4, 0)], 1),
            Err(MediaRequestError::PayloadTooLarge)
        );
    }

    /// A request needs the pages of its largest step, the items overlapping
    /// one granule of prompt rows, not the pages of all its items.
    #[test]
    fn a_request_needs_the_pages_of_its_largest_step_not_its_total() {
        let config = MediaConfigV1::new(8, 4)
            .unwrap()
            .with_encoder(64, 1000)
            .unwrap();
        let item = |value, output_rows| MediaInputV1 {
            key: key(value),
            encoder: 0,
            geometry: [1, 2, 2, 0],
            output_rows,
            payload: vec![0u8; 16].into(),
        };
        let span = |media, prompt_row, rows| MediaSpanV1 {
            media,
            prompt_row,
            rows,
            media_row: 0,
        };
        // Two, three and one pages; the fourth repeats the first.
        let media = [item(1, 8), item(2, 12), item(3, 4), item(1, 8)];
        let spans = [
            span(0, 0, 8),
            span(1, 8, 12),
            span(2, 64, 4),
            span(3, 70, 8),
        ];
        assert_eq!(config.step_pages(&media, &spans, 64), Ok(5));
        assert_eq!(config.step_pages(&media, &spans, 1), Ok(3));
        assert_eq!(config.step_pages(&media, &spans, 128), Ok(6));
        assert_eq!(config.chunk_pages(&media, &spans, 0, 9), 5);
        assert_eq!(config.chunk_pages(&media, &spans, 60, 72), 3);
        assert_eq!(config.chunk_pages(&media, &spans, 20, 64), 0);
        assert_eq!(config.chunk_pages(&media, &spans, 0, 100), 6);
    }

    /// Eviction takes items no live prefill still needs before the least
    /// recently used one.
    #[test]
    fn eviction_spares_items_a_live_prefill_still_needs() {
        let mut arena = arena(4);
        arena.pin(key(1), 8).unwrap();
        arena.pin(key(2), 8).unwrap();
        arena.commit_step();
        assert!(arena.unpin(&key(1)) && arena.unpin(&key(2)));
        arena.want(key(1));
        let third = arena.pin(key(3), 8).unwrap();
        assert_eq!(third.evicted, 1);
        assert!(arena.resident(&key(1)));
        assert!(!arena.resident(&key(2)));
        arena.commit_step();
        let wanted = pins(&[(1, 1)]);
        assert!(arena.wanted_matches(&wanted));
        assert!(arena.unwant(&key(1)));
        assert!(!arena.unwant(&key(1)));
        assert!(arena.wanted_matches(&BTreeMap::new()));
    }
}
