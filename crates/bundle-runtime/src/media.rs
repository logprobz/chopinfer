//! The media section of the V3 batch wire: validation of its
//! encode and inject records against a bundle's arena, the prefill rows each
//! inject overrides, and the page runs an encoder writes its output through.
//!
//! The arena is `rows` rows carved into pages of `page_rows`; the bundle
//! allocates it at create time and the host owns its pages. Arena row `r` of
//! an encode lands at `pages[r / page_rows] * page_rows + r % page_rows`.

use std::collections::{BTreeMap, BTreeSet};

use runtime_bundle_abi::{
    BundleAbiError, BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1,
};
use runtime_bundle_sdk::taps::EncoderSegment;
use runtime_bundle_sdk::{BatchMediaEncodeV3, BatchMediaSectionV3, BatchRequestV1};

use crate::inputs::PhaseInputs;

/// A bundle's media arena and the payload limit of each encoder index.
/// Encoder indexes past `payload_limits` are refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaArena {
    pub rows: u64,
    pub page_rows: u64,
    pub payload_limits: &'static [u64],
}

/// What a step's media section asks of the bundle: the encodes to run and,
/// per live prefill row, the arena row that overrides its embedding (-1
/// keeps the token embedding).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaStepPlan {
    pub source_rows: Vec<i64>,
    pub encodes: Vec<BatchMediaEncodeV3>,
}

impl MediaStepPlan {
    /// Whether any prefill row reads the arena.
    pub fn injects(&self) -> bool {
        self.source_rows.iter().any(|&row| row >= 0)
    }
}

/// `count` output rows of an encoder, starting at `source_row` of its output
/// chunk, that land at consecutive arena rows inside one page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArenaRun {
    pub source_row: u32,
    pub arena_row: u64,
    pub rows: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaError {
    /// An encode names an unknown encoder, exceeds its payload limit or the
    /// section's payload, or its pages do not cover its rows once.
    Encode,
    /// An inject is empty, names an item past the batch or leaves the arena
    /// or its page.
    Inject,
    /// An inject names an item that is not in the prefill phase.
    NotPrefill,
    /// An inject reads past its item's rows.
    OutsideItem,
    /// Two injects override the same row.
    Overlap,
    /// A payload without any encode or inject record.
    StrayPayload,
    /// Arena pages or rows an encoder writes are invalid.
    Pages,
}

impl std::fmt::Display for MediaError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Encode => "invalid media encode record",
            Self::Inject => "invalid media inject record",
            Self::NotPrefill => "media injection names a non-prefill item",
            Self::OutsideItem => "media injection exceeds its item rows",
            Self::Overlap => "media injections overlap",
            Self::StrayPayload => "media payload has no encode record",
            Self::Pages => "invalid encoder arena pages",
        })
    }
}

impl MediaArena {
    fn pages_valid(&self, pages: &[u64], output_rows: u32) -> bool {
        pages.len() == u64::from(output_rows).div_ceil(self.page_rows) as usize
            && pages.iter().all(|&page| page < self.rows / self.page_rows)
            && pages.iter().collect::<BTreeSet<_>>().len() == pages.len()
    }

    /// Validates `media` for a step whose prefill phase is `prefill` and whose
    /// batch holds `item_count` items, and maps its injects to prefill rows.
    pub fn plan(
        &self,
        media: &BatchMediaSectionV3,
        prefill: &PhaseInputs,
        item_count: usize,
    ) -> Result<MediaStepPlan, MediaError> {
        let mut sources = vec![-1_i64; prefill.live_tokens() as usize];
        // Two encodes of one step must not write the same arena page.
        let mut written = BTreeSet::new();
        for encode in &media.encodes {
            if !encode.pages.iter().all(|&page| written.insert(page)) {
                return Err(MediaError::Encode);
            }
            if encode.output_rows == 0
                || u64::from(encode.output_rows) > self.rows
                || encode.payload_bytes == 0
                || self
                    .payload_limits
                    .get(encode.encoder as usize)
                    .is_none_or(|&limit| encode.payload_bytes > limit)
                || encode
                    .payload_offset
                    .checked_add(encode.payload_bytes)
                    .is_none_or(|end| end > media.payload.len() as u64)
                || !self.pages_valid(&encode.pages, encode.output_rows)
            {
                return Err(MediaError::Encode);
            }
        }
        for inject in &media.injects {
            if inject.rows == 0
                || inject.item_index as usize >= item_count
                || inject.arena_row >= self.rows
                || inject.arena_row % self.page_rows + u64::from(inject.rows) > self.page_rows
                || inject
                    .arena_row
                    .checked_add(u64::from(inject.rows))
                    .is_none_or(|end| end > self.rows)
            {
                return Err(MediaError::Inject);
            }
            let Some(packed) = prefill
                .item_order
                .iter()
                .position(|&item| item == inject.item_index as usize)
            else {
                return Err(MediaError::NotPrefill);
            };
            let start = prefill.q_start[packed] as usize + inject.item_row as usize;
            if inject
                .item_row
                .checked_add(inject.rows)
                .is_none_or(|end| end > prefill.q_len[packed] as u32)
                || start
                    .checked_add(inject.rows as usize)
                    .is_none_or(|end| end > sources.len())
            {
                return Err(MediaError::OutsideItem);
            }
            for (relative, source) in sources[start..start + inject.rows as usize]
                .iter_mut()
                .enumerate()
            {
                if *source != -1 {
                    return Err(MediaError::Overlap);
                }
                *source = (inject.arena_row + relative as u64) as i64;
            }
        }
        if media.encodes.is_empty() && media.injects.is_empty() && !media.payload.is_empty() {
            return Err(MediaError::StrayPayload);
        }
        Ok(MediaStepPlan {
            source_rows: sources,
            encodes: media.encodes.clone(),
        })
    }

    /// Splits output rows `[first, first + rows)` of an encode with
    /// `output_rows` rows over `pages` at page boundaries. Each run's
    /// `source_row` counts from `first`.
    pub fn runs(
        &self,
        pages: &[u64],
        output_rows: u32,
        first: u32,
        rows: u32,
    ) -> Result<Vec<ArenaRun>, MediaError> {
        if output_rows == 0
            || u64::from(output_rows) > self.rows
            || rows == 0
            || first.checked_add(rows).is_none_or(|end| end > output_rows)
            || !self.pages_valid(pages, output_rows)
        {
            return Err(MediaError::Pages);
        }
        let mut runs = Vec::new();
        let mut copied = 0;
        while copied < rows {
            let row = u64::from(first + copied);
            let page_offset = row % self.page_rows;
            let count = (self.page_rows - page_offset).min(u64::from(rows - copied)) as u32;
            runs.push(ArenaRun {
                source_row: copied,
                arena_row: pages[(row / self.page_rows) as usize] * self.page_rows + page_offset,
                rows: count,
            });
            copied += count;
        }
        Ok(runs)
    }
}

/// A batch item as [`Occurrences::step`] reads it.
#[derive(Clone, Copy, Debug)]
pub struct OccurrenceItem<'a> {
    pub request: u64,
    /// For a prefill item, the prompt rows earlier steps held; `None` for a
    /// decode or verify item, whose prompt is complete.
    pub prefill: Option<u64>,
    /// The item's input tokens this step.
    pub tokens: &'a [u32],
}

impl<'a> OccurrenceItem<'a> {
    /// Every item of `batch`.
    pub fn of(batch: &'a BatchRequestV1) -> Vec<Self> {
        batch
            .items()
            .iter()
            .zip(batch.appends())
            .map(|(item, append)| Self {
                request: item.request_id(),
                prefill: (!item.is_decode() && !item.is_verify()).then(|| append.previous_tokens()),
                tokens: item.input_tokens(),
            })
            .collect()
    }
}

/// One inject of a step as [`Occurrences`] names it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OccurrenceRead {
    /// The encode whose rows it reads (an `Occurrences` serial) and that
    /// encode's encoder index.
    pub encode: u64,
    pub encoder: u32,
    pub arena_row: u64,
    /// Its request, media kind and item, and its rows from their row of the
    /// encode (`index`); `start` unset.
    pub segment: EncoderSegment,
    /// Whether it begins its occurrence.
    pub begins: bool,
}

/// What [`Occurrences::step`] found in one step: every inject of rows of a
/// known encode, per item in item order and in the item's row order, and
/// the serial of each of the step's encodes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OccurrenceStep {
    pub reads: Vec<OccurrenceRead>,
    pub encodes: Vec<u64>,
}

impl OccurrenceStep {
    /// Where an encoder tap's rows are (`runtime_bundle_sdk::taps`): the
    /// step's injects of rows of `encoder`'s encodes, one segment each, and
    /// the arena runs holding them, whose `source_row` counts from the tap's
    /// first row. A row two injects read is in the tap once per inject.
    pub fn encoder_tap(&self, encoder: u32) -> (Vec<EncoderSegment>, Vec<ArenaRun>) {
        let (mut segments, mut runs) = (Vec::<EncoderSegment>::new(), Vec::new());
        for read in self.reads.iter().filter(|read| read.encoder == encoder) {
            let start = segments.last().map_or(0, |s| s.start + s.rows);
            segments.push(EncoderSegment {
                start,
                ..read.segment
            });
            runs.push(ArenaRun {
                source_row: start,
                arena_row: read.arena_row,
                rows: read.segment.rows,
            });
        }
        (segments, runs)
    }
}

/// The media occurrences of each request's prompt, carried across its
/// prefill steps for diagnostic taps (`runtime_bundle_sdk::taps`): an
/// occurrence is a request's `item`-th media item of a `kind`, counted in
/// prompt order over the whole prompt, as a reference runner counts them.
///
/// Each step's encodes take a serial, and every arena page maps to the
/// encode that last wrote it, so an inject is named by the encode and row
/// it reads whether that encode ran this step, in an earlier chunk of the
/// request or for another request (a resident item). An inject continues
/// the request's occurrence that read the encode's rows just before; any
/// other inject begins the next occurrence of the kind `kinds` gives the
/// placeholder token at its first row. A prompt is counted from its first
/// step (no earlier rows); a prompt first seen later (a prefix cache hit,
/// an unseen retry) or reading a row of no known kind or encode is not
/// counted, and its segments name no request. A decode or verify item ends
/// its prompt.
///
/// It also keeps rows read back for an encode (`keep`, such as its
/// transformer blocks) while any page of the encode is unwritten since.
#[derive(Debug, Default)]
pub struct Occurrences {
    serials: u64,
    /// Per arena page: the encode that last wrote it and the page's index in it.
    pages: BTreeMap<u64, (u64, u32)>,
    /// Per encode with a page still mapped: its encoder index and output rows.
    encodes: BTreeMap<u64, (u32, u32)>,
    prompts: BTreeMap<u64, Prompt>,
    pending: Option<BTreeMap<u64, Prompt>>,
    kept: BTreeMap<u64, Vec<(String, Vec<u8>)>>,
}

#[derive(Clone, Debug, Default)]
struct Prompt {
    /// The prompt rows its steps held.
    rows: u64,
    /// Counted from its first row.
    counted: bool,
    /// Occurrences begun per kind.
    counts: BTreeMap<Option<&'static str>, u32>,
    /// Per occurrence that may continue: its encode, next row, kind and item.
    open: Vec<(u64, u32, Option<&'static str>, u32)>,
}

impl Occurrences {
    /// Names the injects of `media`, a step of `items` over `arena`, and
    /// stages each prompt's occurrences until `finish` commits the step.
    /// Arena writes and kept encoder rows survive cancellation.
    pub fn step(
        &mut self,
        arena: &MediaArena,
        media: &BatchMediaSectionV3,
        items: &[OccurrenceItem<'_>],
        kinds: &[(u32, &'static str)],
    ) -> Result<OccurrenceStep, MediaError> {
        self.pending = None;
        let mut prompts = self.prompts.clone();
        let mut step = OccurrenceStep::default();
        for record in &media.encodes {
            if !arena.pages_valid(&record.pages, record.output_rows) {
                return Err(MediaError::Pages);
            }
            self.serials += 1;
            step.encodes.push(self.serials);
            self.encodes
                .insert(self.serials, (record.encoder, record.output_rows));
            for (index, &page) in record.pages.iter().enumerate() {
                self.pages.insert(page, (self.serials, index as u32));
            }
        }
        let mapped = self
            .pages
            .values()
            .map(|&(encode, _)| encode)
            .collect::<BTreeSet<_>>();
        self.encodes.retain(|encode, _| mapped.contains(encode));
        self.kept.retain(|encode, _| mapped.contains(encode));
        if media
            .injects
            .iter()
            .any(|inject| inject.item_index as usize >= items.len())
        {
            return Err(MediaError::Inject);
        }
        for (index, item) in items.iter().enumerate() {
            let Some(previous) = item.prefill else {
                prompts.remove(&item.request);
                continue;
            };
            let prompt = prompts.entry(item.request).or_default();
            if previous == 0 {
                *prompt = Prompt {
                    counted: true,
                    ..Prompt::default()
                };
            } else if !prompt.counted || prompt.rows != previous {
                *prompt = Prompt {
                    rows: previous,
                    ..Prompt::default()
                };
            }
            let mut injects = media
                .injects
                .iter()
                .filter(|inject| inject.item_index as usize == index)
                .collect::<Vec<_>>();
            injects.sort_by_key(|inject| inject.item_row);
            for inject in injects {
                let located = self.pages.get(&(inject.arena_row / arena.page_rows));
                let Some(&(encode, page)) = located else {
                    prompt.counted = false;
                    continue;
                };
                let (encoder, output_rows) = self.encodes[&encode];
                let row =
                    page * arena.page_rows as u32 + (inject.arena_row % arena.page_rows) as u32;
                let kind = item.tokens.get(inject.item_row as usize).and_then(|token| {
                    kinds
                        .iter()
                        .find(|(placeholder, _)| placeholder == token)
                        .map(|&(_, kind)| kind)
                });
                if kind.is_none() {
                    prompt.counted = false;
                }
                let next = row + inject.rows;
                let continued = prompt
                    .open
                    .iter_mut()
                    .find(|(e, expected, k, _)| *e == encode && *expected == row && *k == kind);
                let (occurrence, begins) = match continued {
                    Some(open) => {
                        open.1 = next;
                        (open.3, false)
                    }
                    None => {
                        let count = prompt.counts.entry(kind).or_insert(0);
                        prompt.open.push((encode, next, kind, *count));
                        *count += 1;
                        (*count - 1, true)
                    }
                };
                prompt.open.retain(|&(_, next, _, _)| next < output_rows);
                step.reads.push(OccurrenceRead {
                    encode,
                    encoder,
                    arena_row: inject.arena_row,
                    segment: EncoderSegment {
                        request: Some(item.request),
                        kind,
                        item: occurrence,
                        start: 0,
                        index: row,
                        rows: inject.rows,
                    },
                    begins,
                });
            }
            prompt.rows = previous + item.tokens.len() as u64;
            if !prompt.counted {
                for read in step.reads.iter_mut().rev() {
                    if read.segment.request != Some(item.request) {
                        break;
                    }
                    read.segment.request = None;
                }
            }
        }
        self.pending = Some(prompts);
        Ok(step)
    }

    /// Commits the staged prompt history, or discards it on cancellation.
    pub fn finish(&mut self, commit: bool) {
        if let Some(prompts) = self.pending.take()
            && commit
        {
            self.prompts = prompts;
        }
    }

    /// Forgets a retired request, including any staged copy of its history.
    /// Resident encodes and their kept rows belong to the arena, not the request.
    pub fn retire(&mut self, request: u64) {
        self.prompts.remove(&request);
        if let Some(prompts) = &mut self.pending {
            prompts.remove(&request);
        }
    }

    /// Keeps `bytes`, rows named `name` read back for `encode`, while the
    /// encode's pages are mapped.
    pub fn keep(&mut self, encode: u64, name: String, bytes: Vec<u8>) {
        if self.encodes.contains_key(&encode) {
            self.kept.entry(encode).or_default().push((name, bytes));
        }
    }

    /// Whether `encode` still has an arena page mapped: a later step may
    /// read its rows.
    pub fn resident(&self, encode: u64) -> bool {
        self.encodes.contains_key(&encode)
    }

    /// The rows kept for `encode`, in the order they were kept.
    pub fn kept(&self, encode: u64) -> &[(String, Vec<u8>)] {
        self.kept.get(&encode).map_or(&[], Vec::as_slice)
    }
}

/// Copies `runs` of `arena` (rows of `row_bytes`), in order, to host
/// memory after the work queued on `stream`.
pub fn read_runs(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    arena: DeviceBufferSpanV1,
    runs: &[ArenaRun],
    row_bytes: u64,
) -> Result<Vec<u8>, BundleAbiError> {
    let rows = runs.iter().map(|run| u64::from(run.rows)).sum::<u64>();
    let mut bytes = vec![0; (rows * row_bytes) as usize];
    services.synchronize_stream(stream)?;
    for run in runs {
        let start = (u64::from(run.source_row) * row_bytes) as usize;
        let length = u64::from(run.rows) * row_bytes;
        services.copy_device_to_host(
            stream,
            &mut bytes[start..start + length as usize],
            DeviceBufferSpanV1::new(
                arena.buffer,
                arena.offset + run.arena_row * row_bytes,
                length,
            ),
        )?;
    }
    services.synchronize_stream(stream)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use runtime_bundle_sdk::BatchMediaInjectV3;

    use super::*;

    const ARENA: MediaArena = MediaArena {
        rows: 1024,
        page_rows: 64,
        payload_limits: &[4096, 1024],
    };

    fn prefill() -> PhaseInputs {
        PhaseInputs {
            item_order: vec![0, 2],
            q_start: vec![0, 4],
            q_len: vec![4, 2],
            token_ids: vec![0; 6],
            ..PhaseInputs::default()
        }
    }

    fn inject(item_index: u32, item_row: u32, rows: u32, arena_row: u64) -> BatchMediaInjectV3 {
        BatchMediaInjectV3 {
            item_index,
            item_row,
            rows,
            arena_row,
        }
    }

    fn encode(encoder: u32, output_rows: u32, pages: Vec<u64>, bytes: u64) -> BatchMediaSectionV3 {
        BatchMediaSectionV3 {
            encodes: vec![BatchMediaEncodeV3 {
                encoder,
                geometry: [1, 2, 2, 0],
                output_rows,
                pages,
                payload_offset: 0,
                payload_bytes: bytes,
            }],
            payload: vec![0; bytes as usize],
            ..BatchMediaSectionV3::default()
        }
    }

    #[test]
    fn injects_map_to_their_own_prefill_rows() {
        let mut media = BatchMediaSectionV3 {
            injects: vec![inject(0, 1, 2, 64), inject(2, 0, 1, 130)],
            ..BatchMediaSectionV3::default()
        };
        let plan = ARENA.plan(&media, &prefill(), 3).unwrap();
        assert_eq!(plan.source_rows, [-1, 64, 65, -1, 130, -1]);
        assert!(plan.injects());
        assert!(
            !ARENA
                .plan(&Default::default(), &prefill(), 3)
                .unwrap()
                .injects()
        );
        media.injects.push(inject(0, 2, 1, 66));
        assert_eq!(ARENA.plan(&media, &prefill(), 3), Err(MediaError::Overlap));
        media.injects[2] = inject(1, 0, 1, 66);
        assert_eq!(
            ARENA.plan(&media, &prefill(), 3),
            Err(MediaError::NotPrefill)
        );
        media.injects[2] = inject(2, 1, 2, 66);
        assert_eq!(
            ARENA.plan(&media, &prefill(), 3),
            Err(MediaError::OutsideItem)
        );
        media.injects[2] = inject(3, 0, 1, 66);
        assert_eq!(ARENA.plan(&media, &prefill(), 3), Err(MediaError::Inject));
        // A run may not cross a page or leave the arena.
        media.injects[2] = inject(0, 3, 2, 63);
        assert_eq!(ARENA.plan(&media, &prefill(), 3), Err(MediaError::Inject));
        media.injects[2] = inject(0, 3, 1, 1024);
        assert_eq!(ARENA.plan(&media, &prefill(), 3), Err(MediaError::Inject));
        media.injects[2] = inject(0, 3, 0, 0);
        assert_eq!(ARENA.plan(&media, &prefill(), 3), Err(MediaError::Inject));
    }

    #[test]
    fn encodes_follow_their_encoder_limit_and_cover_their_rows_once() {
        let empty = PhaseInputs::default();
        assert!(
            ARENA
                .plan(&encode(0, 65, vec![3, 0], 4096), &empty, 0)
                .is_ok()
        );
        for media in [
            encode(1, 65, vec![3, 0], 4096),
            encode(2, 65, vec![3, 0], 16),
            encode(0, 65, vec![3], 16),
            encode(0, 65, vec![3, 3], 16),
            encode(0, 65, vec![3, 16], 16),
            encode(0, 1025, vec![0; 17], 16),
            encode(0, 0, vec![], 16),
        ] {
            assert_eq!(ARENA.plan(&media, &empty, 0), Err(MediaError::Encode));
        }
        // Two encodes of one step may not share a page.
        let mut shared = encode(0, 65, vec![3, 0], 4096);
        let mut second = shared.encodes[0].clone();
        second.pages = vec![5, 3];
        shared.encodes.push(second);
        assert_eq!(ARENA.plan(&shared, &empty, 0), Err(MediaError::Encode));
        shared.encodes[1].pages = vec![5, 6];
        assert!(ARENA.plan(&shared, &empty, 0).is_ok());
        let mut past = encode(0, 1, vec![0], 16);
        past.encodes[0].payload_offset = 1;
        assert_eq!(ARENA.plan(&past, &empty, 0), Err(MediaError::Encode));
        past.encodes.clear();
        assert_eq!(ARENA.plan(&past, &empty, 0), Err(MediaError::StrayPayload));
    }

    #[test]
    fn runs_follow_noncontiguous_pages() {
        assert_eq!(
            ARENA.runs(&[9, 2, 7], 150, 60, 80).unwrap(),
            [
                ArenaRun {
                    source_row: 0,
                    arena_row: 636,
                    rows: 4
                },
                ArenaRun {
                    source_row: 4,
                    arena_row: 128,
                    rows: 64
                },
                ArenaRun {
                    source_row: 68,
                    arena_row: 448,
                    rows: 12
                },
            ]
        );
        assert_eq!(ARENA.runs(&[1, 1], 100, 0, 100), Err(MediaError::Pages));
        assert_eq!(ARENA.runs(&[1, 2], 100, 99, 2), Err(MediaError::Pages));
        assert_eq!(ARENA.runs(&[1, 16], 100, 0, 1), Err(MediaError::Pages));
        assert_eq!(ARENA.runs(&[1, 2], 100, 0, 0), Err(MediaError::Pages));
    }

    const IMAGE: u32 = 1;
    const VIDEO: u32 = 2;
    const AUDIO: u32 = 3;
    const KINDS: &[(u32, &str)] = &[(IMAGE, "image"), (VIDEO, "video"), (AUDIO, "audio")];

    fn segment(
        request: Option<u64>,
        (kind, item): (&'static str, u32),
        start: u32,
        index: u32,
        rows: u32,
    ) -> EncoderSegment {
        EncoderSegment {
            request,
            kind: Some(kind),
            item,
            start,
            index,
            rows,
        }
    }

    fn run(source_row: u32, arena_row: u64, rows: u32) -> ArenaRun {
        ArenaRun {
            source_row,
            arena_row,
            rows,
        }
    }

    fn encoded(encoder: u32, output_rows: u32, pages: Vec<u64>) -> BatchMediaEncodeV3 {
        BatchMediaEncodeV3 {
            encoder,
            geometry: [1, 2, 2, 0],
            output_rows,
            pages,
            payload_offset: 0,
            payload_bytes: 1,
        }
    }

    /// Placeholder tokens: `runs` of (rows, token).
    fn tokens(runs: &[(usize, u32)]) -> Vec<u32> {
        runs.iter()
            .flat_map(|&(rows, token)| std::iter::repeat_n(token, rows))
            .collect()
    }

    /// One step of a fresh `Occurrences`: each item a first prefill chunk
    /// of its request, or with `None` a request with earlier rows it never
    /// saw.
    fn first_step(media: &BatchMediaSectionV3, items: &[(Option<u64>, &[u32])]) -> OccurrenceStep {
        let items = items
            .iter()
            .enumerate()
            .map(|(index, &(request, tokens))| OccurrenceItem {
                request: request.unwrap_or(1000 + index as u64),
                prefill: Some(if request.is_some() { 0 } else { 5 }),
                tokens,
            })
            .collect::<Vec<_>>();
        Occurrences::default()
            .step(&ARENA, media, &items, KINDS)
            .unwrap()
    }

    fn chunk<'a>(request: u64, previous: u64, tokens: &'a [u32]) -> OccurrenceItem<'a> {
        OccurrenceItem {
            request,
            prefill: Some(previous),
            tokens,
        }
    }

    /// (request, kind, item, index, rows, begins) of a read.
    type Named = (Option<u64>, &'static str, u32, u32, u32, bool);

    fn named(step: &OccurrenceStep) -> Vec<Named> {
        step.reads
            .iter()
            .map(|read| {
                let s = read.segment;
                (
                    s.request,
                    s.kind.unwrap(),
                    s.item,
                    s.index,
                    s.rows,
                    read.begins,
                )
            })
            .collect()
    }

    #[test]
    fn encoder_taps_count_each_requests_media_in_its_prompt_order() {
        // Item 1 shows a 70-row image (pages 3 and 1, injected in two runs)
        // then a 2-row one; item 0 a 3-row image and a 4-row audio clip.
        let media = BatchMediaSectionV3 {
            encodes: vec![
                encoded(0, 70, vec![3, 1]),
                encoded(1, 4, vec![0]),
                encoded(0, 2, vec![5]),
                encoded(0, 3, vec![6]),
                encoded(0, 1, vec![7]),
            ],
            injects: vec![
                inject(1, 64, 6, 64),
                inject(1, 0, 64, 192),
                inject(0, 3, 4, 0),
                inject(0, 0, 3, 384),
                inject(1, 70, 2, 320),
            ],
            ..BatchMediaSectionV3::default()
        };
        let first = tokens(&[(3, IMAGE), (4, AUDIO)]);
        let second = tokens(&[(72, IMAGE)]);
        let items = [(Some(40), &first[..]), (Some(41), &second[..])];
        let (segments, runs) = first_step(&media, &items).encoder_tap(0);
        assert_eq!(
            segments,
            [
                segment(Some(40), ("image", 0), 0, 0, 3),
                segment(Some(41), ("image", 0), 3, 0, 64),
                segment(Some(41), ("image", 0), 67, 64, 6),
                segment(Some(41), ("image", 1), 73, 0, 2),
            ]
        );
        assert_eq!(
            runs,
            [
                run(0, 384, 3),
                run(3, 192, 64),
                run(67, 64, 6),
                run(73, 320, 2)
            ]
        );
        let items = [(Some(40), &first[..]), (None, &second[..])];
        let (segments, runs) = first_step(&media, &items).encoder_tap(1);
        assert_eq!(segments, [segment(Some(40), ("audio", 0), 0, 0, 4)]);
        assert_eq!(runs, [run(0, 0, 4)]);
        assert_eq!(first_step(&media, &items).encoder_tap(2), (vec![], vec![]));
    }

    /// Each occurrence begins where an inject does not continue it; every
    /// read names the encode's serial.
    #[test]
    fn occurrences_begin_where_no_read_continues_them() {
        let media = BatchMediaSectionV3 {
            encodes: vec![
                encoded(0, 70, vec![3, 1]),
                encoded(1, 4, vec![0]),
                encoded(0, 2, vec![5]),
                encoded(0, 3, vec![6]),
                encoded(0, 1, vec![7]),
            ],
            injects: vec![
                inject(1, 64, 6, 64),
                inject(1, 0, 64, 192),
                inject(0, 3, 4, 0),
                inject(0, 0, 3, 384),
                inject(1, 70, 2, 320),
            ],
            ..BatchMediaSectionV3::default()
        };
        let first = tokens(&[(3, IMAGE), (4, AUDIO)]);
        let second = tokens(&[(72, IMAGE)]);
        let step = first_step(&media, &[(Some(40), &first), (Some(41), &second)]);
        assert_eq!(step.encodes, [1, 2, 3, 4, 5]);
        assert_eq!(
            named(&step),
            [
                (Some(40), "image", 0, 0, 3, true),
                (Some(40), "audio", 0, 0, 4, true),
                (Some(41), "image", 0, 0, 64, true),
                (Some(41), "image", 0, 64, 6, false),
                (Some(41), "image", 1, 0, 2, true),
            ]
        );
        let encodes = step
            .reads
            .iter()
            .map(|read| read.encode)
            .collect::<Vec<_>>();
        assert_eq!(encodes, [4, 2, 1, 1, 3]);
    }

    /// A request's later chunks keep its request and its prompt-wide items:
    /// encoding first starts after a text-only chunk, an image is cut by a
    /// chunk boundary, and a later request reads a resident encode. A
    /// prompt first seen after its first row, or after its decode, is not
    /// counted. Rows kept for an encode last until its page is rewritten.
    #[test]
    fn occurrences_carry_across_prefill_chunks() {
        let mut occurrences = Occurrences::default();
        let step = |occurrences: &mut Occurrences,
                    media: &BatchMediaSectionV3,
                    items: &[OccurrenceItem<'_>]| {
            let step = occurrences.step(&ARENA, media, items, KINDS).unwrap();
            occurrences.finish(true);
            (named(&step), step.encodes)
        };
        // Request 7: 10 text rows, image A (4 rows), 2 text rows, image B
        // (6 rows), 3 text rows, image C (2 rows), over three chunks.
        let text = tokens(&[(10, 0)]);
        assert_eq!(
            step(
                &mut occurrences,
                &BatchMediaSectionV3::default(),
                &[chunk(7, 0, &text)]
            ),
            (vec![], vec![])
        );
        let second = tokens(&[(4, IMAGE), (2, 0), (3, IMAGE)]);
        let media = BatchMediaSectionV3 {
            encodes: vec![encoded(0, 4, vec![2]), encoded(0, 6, vec![3])],
            injects: vec![inject(0, 0, 4, 128), inject(0, 6, 3, 192)],
            ..BatchMediaSectionV3::default()
        };
        assert_eq!(
            step(&mut occurrences, &media, &[chunk(7, 10, &second)]),
            (
                vec![
                    (Some(7), "image", 0, 0, 4, true),
                    (Some(7), "image", 1, 0, 3, true),
                ],
                vec![1, 2]
            )
        );
        occurrences.keep(1, "blocks".into(), vec![1, 2]);
        let third = tokens(&[(3, IMAGE), (3, 0), (2, IMAGE)]);
        let fresh = tokens(&[(1, 0), (4, IMAGE)]);
        let media = BatchMediaSectionV3 {
            encodes: vec![encoded(0, 2, vec![4])],
            injects: vec![
                inject(0, 0, 3, 195),
                inject(0, 6, 2, 256),
                inject(1, 1, 4, 128),
                inject(2, 0, 4, 128),
            ],
            ..BatchMediaSectionV3::default()
        };
        let (reads, encodes) = step(
            &mut occurrences,
            &media,
            &[
                chunk(7, 19, &third),
                chunk(8, 0, &fresh),
                chunk(9, 20, &second),
            ],
        );
        assert_eq!(encodes, [3]);
        assert_eq!(
            reads,
            [
                (Some(7), "image", 1, 3, 3, false),
                (Some(7), "image", 2, 0, 2, true),
                (Some(8), "image", 0, 0, 4, true),
                (None, "image", 0, 0, 4, true),
            ]
        );
        assert_eq!(occurrences.kept(1), [("blocks".to_owned(), vec![1, 2])]);
        assert!((1..=3).all(|encode| occurrences.resident(encode)));
        // Request 7 decodes; a later prefill of it is not counted. Page 2
        // is rewritten, so encode 1's rows go.
        let media = BatchMediaSectionV3 {
            encodes: vec![encoded(0, 1, vec![2])],
            injects: vec![inject(1, 0, 1, 128)],
            ..BatchMediaSectionV3::default()
        };
        let decode = OccurrenceItem {
            request: 7,
            prefill: None,
            tokens: &[5],
        };
        let image = tokens(&[(1, IMAGE)]);
        assert_eq!(
            step(&mut occurrences, &media, &[decode, chunk(7, 30, &image)]),
            (vec![(None, "image", 0, 0, 1, true)], vec![4])
        );
        assert!(occurrences.kept(1).is_empty());
        assert!(!occurrences.resident(1));
        assert!((2..=4).all(|encode| occurrences.resident(encode)));
    }

    #[test]
    fn cancelled_prompt_changes_leave_committed_history_and_resident_rows() {
        let mut occurrences = Occurrences::default();
        let first = BatchMediaSectionV3 {
            encodes: vec![encoded(0, 2, vec![2])],
            injects: vec![inject(0, 0, 1, 128)],
            ..BatchMediaSectionV3::default()
        };
        let items = [chunk(7, 0, &[IMAGE])];
        occurrences.step(&ARENA, &first, &items, KINDS).unwrap();
        assert!(occurrences.prompts.is_empty());
        occurrences.finish(true);
        let next = BatchMediaSectionV3 {
            encodes: vec![encoded(0, 1, vec![3])],
            injects: vec![inject(0, 0, 1, 129), inject(0, 1, 1, 192)],
            ..BatchMediaSectionV3::default()
        };
        let items = [chunk(7, 1, &[IMAGE, IMAGE])];
        let attempted = occurrences.step(&ARENA, &next, &items, KINDS).unwrap();
        occurrences.keep(2, "blocks".into(), vec![1, 2]);
        occurrences.finish(false);
        assert_eq!(occurrences.prompts[&7].rows, 1);
        // Both encodes remain resident after cancellation; retry reads them.
        let retry = BatchMediaSectionV3 {
            encodes: vec![],
            ..next
        };
        let retried = occurrences.step(&ARENA, &retry, &items, KINDS).unwrap();
        assert_eq!(named(&attempted), named(&retried));
        assert_eq!(
            named(&retried),
            vec![
                (Some(7), "image", 0, 1, 1, false),
                (Some(7), "image", 1, 0, 1, true),
            ]
        );
        assert_eq!(occurrences.kept(2), [("blocks".to_owned(), vec![1, 2])]);
        occurrences.finish(true);
        assert_eq!(occurrences.prompts[&7].rows, 3);
    }

    #[test]
    fn retirement_releases_prompt_history_without_releasing_resident_rows() {
        let mut occurrences = Occurrences::default();
        let media = BatchMediaSectionV3 {
            encodes: vec![encoded(0, 2, vec![2])],
            injects: vec![inject(0, 0, 1, 128)],
            ..BatchMediaSectionV3::default()
        };
        occurrences
            .step(&ARENA, &media, &[chunk(7, 0, &[IMAGE])], KINDS)
            .unwrap();
        occurrences.finish(true);
        occurrences.keep(1, "blocks".into(), vec![1, 2]);
        let empty = BatchMediaSectionV3::default();
        // Another request can be open while the bundle retires request 7.
        occurrences
            .step(&ARENA, &empty, &[chunk(8, 0, &[11])], KINDS)
            .unwrap();
        occurrences.retire(7);
        occurrences.finish(true);
        assert!(!occurrences.prompts.contains_key(&7));
        occurrences.retire(8);
        assert!(occurrences.prompts.is_empty());
        // Repeated text-only preemptions must not accumulate prompt entries.
        for request in 10..74 {
            occurrences
                .step(&ARENA, &empty, &[chunk(request, 0, &[11])], KINDS)
                .unwrap();
            occurrences.finish(true);
            occurrences
                .step(&ARENA, &empty, &[chunk(request, 1, &[12])], KINDS)
                .unwrap();
            occurrences.finish(false);
            occurrences.retire(request);
            occurrences.retire(request);
            assert!(occurrences.prompts.is_empty());
            assert!(occurrences.pending.is_none());
        }
        // A cancelled first step has no committed history to retain either.
        occurrences
            .step(&ARENA, &empty, &[chunk(9, 0, &[11])], KINDS)
            .unwrap();
        occurrences.finish(false);
        occurrences.retire(9);
        assert!(occurrences.prompts.is_empty());
        let resident = BatchMediaSectionV3 {
            injects: vec![inject(0, 0, 1, 129), inject(1, 0, 2, 128)],
            ..BatchMediaSectionV3::default()
        };
        let step = occurrences
            .step(
                &ARENA,
                &resident,
                &[chunk(7, 1, &[IMAGE]), chunk(8, 0, &[IMAGE, IMAGE])],
                KINDS,
            )
            .unwrap();
        assert_eq!(
            named(&step),
            vec![
                (None, "image", 0, 1, 1, true),
                (Some(8), "image", 0, 0, 2, true),
            ]
        );
        assert_eq!(occurrences.kept(1), [("blocks".to_owned(), vec![1, 2])]);
    }

    #[test]
    fn encoder_taps_repeat_rows_shared_by_requests() {
        // One 70-row encode: items 0 and 2 inject all of it, item 1 rows
        // 5..9 (a request with earlier steps: not counted), item 3 none.
        let media = BatchMediaSectionV3 {
            encodes: vec![encoded(0, 70, vec![3, 1])],
            injects: vec![
                inject(0, 0, 64, 192),
                inject(1, 0, 4, 197),
                inject(2, 65, 6, 64),
                inject(2, 1, 64, 192),
                inject(0, 64, 6, 64),
            ],
            ..BatchMediaSectionV3::default()
        };
        let image = tokens(&[(71, IMAGE)]);
        let items = [
            (Some(7), &image[..]),
            (None, &image[..]),
            (Some(8), &image[..]),
            (Some(9), &image[..]),
        ];
        let step = first_step(&media, &items);
        // Each request's occurrence begins once.
        let begun = step.reads.iter().filter(|read| read.begins);
        assert_eq!(
            begun.map(|read| read.segment.request).collect::<Vec<_>>(),
            [Some(7), None, Some(8)]
        );
        let (segments, runs) = step.encoder_tap(0);
        assert_eq!(
            segments,
            [
                segment(Some(7), ("image", 0), 0, 0, 64),
                segment(Some(7), ("image", 0), 64, 64, 6),
                segment(None, ("image", 0), 70, 5, 4),
                segment(Some(8), ("image", 0), 74, 0, 64),
                segment(Some(8), ("image", 0), 138, 64, 6),
            ]
        );
        let sources = runs
            .iter()
            .map(|run| (run.source_row, run.arena_row, run.rows));
        assert_eq!(
            sources.collect::<Vec<_>>(),
            [
                (0, 192, 64),
                (64, 64, 6),
                (70, 197, 4),
                (74, 192, 64),
                (138, 64, 6)
            ]
        );
    }

    #[test]
    fn a_request_showing_an_image_twice_holds_each_occurrence() {
        let media = BatchMediaSectionV3 {
            encodes: vec![encoded(0, 1, vec![0])],
            injects: vec![inject(0, 0, 1, 0), inject(0, 1, 1, 0)],
            ..BatchMediaSectionV3::default()
        };
        let image = tokens(&[(2, IMAGE)]);
        let step = first_step(&media, &[(Some(7), &image)]);
        assert!(
            step.reads
                .iter()
                .all(|read| read.begins && read.encode == 1)
        );
        let (segments, runs) = step.encoder_tap(0);
        assert_eq!(
            segments,
            [
                segment(Some(7), ("image", 0), 0, 0, 1),
                segment(Some(7), ("image", 1), 1, 0, 1),
            ]
        );
        assert_eq!(runs, [run(0, 0, 1), run(1, 0, 1)]);
    }

    #[test]
    fn requests_sharing_images_in_other_orders_count_their_own() {
        // Request 7 shows image A (page 0) then B (page 1); request 8 B then A.
        let media = BatchMediaSectionV3 {
            encodes: vec![encoded(0, 1, vec![0]), encoded(0, 1, vec![1])],
            injects: vec![
                inject(0, 0, 1, 0),
                inject(0, 1, 1, 64),
                inject(1, 0, 1, 64),
                inject(1, 1, 1, 0),
            ],
            ..BatchMediaSectionV3::default()
        };
        let image = tokens(&[(2, IMAGE)]);
        let items = [(Some(7), &image[..]), (Some(8), &image[..])];
        let (segments, runs) = first_step(&media, &items).encoder_tap(0);
        assert_eq!(
            segments,
            [
                segment(Some(7), ("image", 0), 0, 0, 1),
                segment(Some(7), ("image", 1), 1, 0, 1),
                segment(Some(8), ("image", 0), 2, 0, 1),
                segment(Some(8), ("image", 1), 3, 0, 1),
            ]
        );
        let arena_rows = runs.iter().map(|run| run.arena_row).collect::<Vec<_>>();
        assert_eq!(arena_rows, [0, 64, 64, 0]);
    }

    #[test]
    fn interleaved_images_and_video_are_counted_per_kind() {
        // Image A, a video injected as two timestamped spans of one encode,
        // then image B; the wire lists the encodes in another order.
        let media = BatchMediaSectionV3 {
            encodes: vec![
                encoded(0, 8, vec![1]),
                encoded(0, 1, vec![2]),
                encoded(0, 2, vec![9]),
            ],
            injects: vec![
                inject(0, 7, 4, 68),
                inject(0, 11, 1, 128),
                inject(0, 0, 2, 576),
                inject(0, 2, 4, 64),
            ],
            ..BatchMediaSectionV3::default()
        };
        let prompt = tokens(&[(2, IMAGE), (4, VIDEO), (1, 0), (4, VIDEO), (1, IMAGE)]);
        let (segments, runs) = first_step(&media, &[(Some(5), &prompt)]).encoder_tap(0);
        assert_eq!(
            segments,
            [
                segment(Some(5), ("image", 0), 0, 0, 2),
                segment(Some(5), ("video", 0), 2, 0, 4),
                segment(Some(5), ("video", 0), 6, 4, 4),
                segment(Some(5), ("image", 1), 10, 0, 1),
            ]
        );
        let arena_rows = runs.iter().map(|run| run.arena_row).collect::<Vec<_>>();
        assert_eq!(arena_rows, [576, 64, 68, 128]);
    }

    #[test]
    fn items_reading_earlier_rows_or_unknown_tokens_are_not_counted() {
        // Item 0 also injects page 4, which no encode of this step wrote;
        // item 2's injected rows hold no placeholder of a known kind.
        let media = BatchMediaSectionV3 {
            encodes: vec![encoded(0, 2, vec![0])],
            injects: vec![
                inject(0, 0, 3, 256),
                inject(0, 3, 2, 0),
                inject(1, 0, 2, 0),
                inject(2, 0, 2, 0),
            ],
            ..BatchMediaSectionV3::default()
        };
        let image = tokens(&[(5, IMAGE)]);
        let text = tokens(&[(2, 99)]);
        let items = [
            (Some(5), &image[..]),
            (Some(6), &image[..]),
            (Some(7), &text[..]),
        ];
        let (segments, _) = first_step(&media, &items).encoder_tap(0);
        let unknown = EncoderSegment {
            request: None,
            kind: None,
            item: 0,
            start: 4,
            index: 0,
            rows: 2,
        };
        assert_eq!(
            segments,
            [
                segment(None, ("image", 0), 0, 0, 2),
                segment(Some(6), ("image", 0), 2, 0, 2),
                unknown,
            ]
        );
    }
}
