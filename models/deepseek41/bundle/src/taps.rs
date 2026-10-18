//! DeepSeek V4.1's taps (`runtime_bundle_sdk::taps`, declared in
//! `models/deepseek41/taps.json`), written only by a bundle built with the
//! `diagnostic-taps` feature, with `CHOPINFER_TAPS_DIR` set at create.
//!
//! Encoder taps: each step whose prefill injects image rows writes the image
//! cells each request injects (`enc.vision.out`, read from rank 0's media
//! arena after the encoder ran, without the learned start, newline and end
//! rows, numbered in raster order as the reference's aligner output) and the
//! transformer blocks of each image occurrence it begins
//! (`enc.vision.B<nn>.in` and `.out`, read back by the shared encoder
//! runtime, `Encoder::block_taps`). Rows are named by request, media kind
//! and the item's place in its prompt, carried across the prompt's prefill
//! chunks (`bundle_runtime::media::Occurrences`).
//!
//! Text taps (`text`): while the sites filter selects one, every prefill
//! runs stage by stage across the ranks, and after each stage its selected
//! taps' rows are read back from rank 0's scratch; decode and verify graph
//! buckets are not tapped. `tools/qual/tapdiff.py` compares the steps with
//! the reference runner's taps (`models/deepseek41/reference.py`, `taps`).
//!
//! One tap step per tapped phase: a prefill's encoder rows are written into
//! its text step when text taps are on, else into a step of their own. A
//! commit publishes the steps, and a cancel or a failed execute removes
//! them. A serving bundle refuses to create while a taps variable is set.

#[cfg(feature = "diagnostic-taps")]
mod text;

#[cfg(feature = "diagnostic-taps")]
pub use enabled::StepTaps;
#[cfg(not(feature = "diagnostic-taps"))]
pub use serving::StepTaps;
#[cfg(feature = "diagnostic-taps")]
pub use text::specs as text_specs;

/// The model name each step's `capture.json` records.
#[cfg(feature = "diagnostic-taps")]
pub const MODEL: &str = "deepseek41";

#[cfg(not(feature = "diagnostic-taps"))]
mod serving {
    use bundle_runtime::encoder::Encoder;
    use bundle_runtime::inputs::PhaseInputs;
    use runtime_bundle_abi::{BundleStreamHandleV1, HostServicesV1};

    use crate::config::Configuration;

    /// A serving build's taps: none, and a create that asks for them is
    /// refused.
    #[derive(Debug, Default)]
    pub struct StepTaps;

    impl StepTaps {
        /// Refuses a create while a taps variable is set.
        pub fn from_env(_config: &Configuration) -> Result<Self, String> {
            match runtime_bundle_sdk::taps::Taps::requested(None) {
                Some(variable) => Err(format!(
                    "{variable} is set, but this DeepSeek V4.1 bundle was built without the \
                     diagnostic-taps feature; unset it to serve"
                )),
                None => Ok(Self),
            }
        }

        /// Whether prefills are tapped: never.
        pub fn text(&self) -> bool {
            false
        }

        pub fn selects_blocks(&self) -> bool {
            false
        }

        pub fn step(
            &mut self,
            _services: &HostServicesV1,
            _stream: BundleStreamHandleV1,
            _encoder: &Encoder,
            _request: &[u8],
            _prefill: &PhaseInputs,
        ) -> Result<(), String> {
            Ok(())
        }

        pub fn finish(&mut self, _commit: bool) -> Result<(), String> {
            Ok(())
        }

        pub fn discard(&mut self, _encoder: &Encoder) -> Result<(), String> {
            Ok(())
        }

        pub fn retire(&mut self, _request: u64) {}
    }
}

#[cfg(feature = "diagnostic-taps")]
mod enabled {
    use std::collections::BTreeMap;

    use bundle_runtime::encoder::{BlockRows, Encoder};
    use bundle_runtime::inputs::PhaseInputs;
    use bundle_runtime::media::{
        ArenaRun, OccurrenceItem, OccurrenceRead, OccurrenceStep, Occurrences,
    };
    use runtime_bundle_abi::{BundleStreamHandleV1, HostServicesV1};
    use runtime_bundle_sdk::BatchCodecV1;
    use runtime_bundle_sdk::taps::{EncoderRows, EncoderSegment, TapError, TapSpec, TapStep, Taps};

    use crate::buffers::RankBuffers;
    use crate::config::Configuration;
    use crate::encoders::ARENA;
    use crate::program::Stage;
    use crate::shape::Shape;

    /// The tower's output rows.
    const OUTPUT: &str = "enc.vision.out";

    /// The taps the environment configures, the encoder's taps, whether the
    /// filter selects a text tap, the image occurrences of every prompt and
    /// the cell grid of every encode they read, a prefill's encoder rows
    /// waiting for its text step, the phase step being captured and the
    /// steps a commit or cancel ends.
    #[derive(Default)]
    pub struct StepTaps {
        taps: Option<Taps>,
        specs: Vec<TapSpec>,
        text: bool,
        kinds: Vec<(u32, &'static str)>,
        side: u32,
        occurrences: Occurrences,
        grids: BTreeMap<u64, Grid>,
        encoders: Vec<EncoderRows>,
        open: Option<TapStep>,
        pending: Vec<TapStep>,
    }

    /// An image's cells: `rows` rows of `columns`, laid out with the
    /// learned delimiters as `rows * (columns + 1) + 2` encoder rows.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(super) struct Grid {
        pub(super) rows: u32,
        pub(super) columns: u32,
    }

    impl Grid {
        /// The grid of an encode of `[h, w]` patches merged `side x side`
        /// that writes `output_rows` rows, which must be its span.
        pub(super) fn of(patches: [u32; 2], side: u32, output_rows: u32) -> Result<Self, String> {
            let grid = Self {
                rows: patches[0].div_ceil(side),
                columns: patches[1].div_ceil(side),
            };
            let span = u64::from(grid.rows) * (u64::from(grid.columns) + 1) + 2;
            if span != u64::from(output_rows) {
                return Err(format!(
                    "an encode of {patches:?} patches writes {output_rows} rows, not its span of {span}"
                ));
            }
            Ok(grid)
        }

        /// The cell at span row `row`, None for a delimiter row.
        pub(super) fn cell(self, row: u32) -> Option<u32> {
            let at = row.checked_sub(1)?;
            let (line, column) = (at / (self.columns + 1), at % (self.columns + 1));
            (line < self.rows && column < self.columns).then_some(line * self.columns + column)
        }
    }

    /// Drops the grid of every encode no arena page maps any more: no later
    /// step reads its rows.
    pub(super) fn keep_resident(grids: &mut BTreeMap<u64, Grid>, occurrences: &Occurrences) {
        grids.retain(|&encode, _| occurrences.resident(encode));
    }

    /// The image cells of `reads`, the injects of the vision encoder's rows
    /// in tap order, each read's encode laid out as `grids` gives: one
    /// segment per run of consecutive cells, numbered as cells, and the
    /// arena runs holding them, whose `source_row` counts from the tap's
    /// first row.
    pub(super) fn cells(
        reads: &[OccurrenceRead],
        grids: &BTreeMap<u64, Grid>,
    ) -> Result<(Vec<EncoderSegment>, Vec<ArenaRun>), String> {
        let (mut segments, mut runs) = (Vec::<EncoderSegment>::new(), Vec::<ArenaRun>::new());
        let mut start = 0;
        for read in reads {
            let grid = grids
                .get(&read.encode)
                .ok_or("an inject reads an encode of no known grid")?;
            let mut last: Option<u32> = None;
            for offset in 0..read.segment.rows {
                let Some(cell) = grid.cell(read.segment.index + offset) else {
                    last = None;
                    continue;
                };
                let arena_row = read.arena_row + u64::from(offset);
                match (last, segments.last_mut(), runs.last_mut()) {
                    (Some(previous), Some(segment), Some(run)) if previous + 1 == cell => {
                        segment.rows += 1;
                        run.rows += 1;
                    }
                    _ => {
                        segments.push(EncoderSegment {
                            start,
                            index: cell,
                            rows: 1,
                            ..read.segment
                        });
                        runs.push(ArenaRun {
                            source_row: start,
                            arena_row,
                            rows: 1,
                        });
                    }
                }
                last = Some(cell);
                start += 1;
            }
        }
        Ok((segments, runs))
    }

    impl StepTaps {
        /// The taps the environment configures, if any, for `config`'s
        /// instance; refuses a filter that selects no tap. The create
        /// record's producer is the host's startup nonce each step's
        /// `capture.json` names.
        pub fn from_env(config: &Configuration) -> Result<Self, String> {
            let Some(taps) = Taps::from_env(None)? else {
                return Ok(Self::default());
            };
            let taps = taps
                .with_model(super::MODEL)
                .with_producer(config.create.tap_producer.clone());
            let specs = if config.media {
                crate::encoders::taps()?
            } else {
                Vec::new()
            };
            let text = super::text::specs(&config.shape);
            let every = specs.iter().chain(&text).cloned().collect::<Vec<_>>();
            taps.check_layers(&every)?;
            taps.check_sites(&every)?;
            let text = text.iter().any(|spec| taps.selected(spec));
            Ok(Self {
                taps: Some(taps),
                specs,
                text,
                kinds: vec![(config.shape.geometry.shape.tokens.image, "image")],
                side: crate::encoders::tap_shape()
                    .map_err(|error| error.0)?
                    .cell_side,
                ..Self::default()
            })
        }

        /// Whether prefills are tapped, running stage by stage.
        pub fn text(&self) -> bool {
            self.text
        }

        /// Whether a block tap is selected: the encoder runtime reads its
        /// blocks back only then.
        pub fn selects_blocks(&self) -> bool {
            self.taps.as_ref().is_some_and(|taps| {
                self.specs
                    .iter()
                    .any(|spec| spec.name != OUTPUT && taps.selected(spec))
            })
        }

        /// Takes the step `request`'s encoder taps after `encoder` ran: the
        /// image cells each prefill item injects and the blocks of each
        /// occurrence it begins, over the `prefill` rows. With text taps
        /// they wait for the prefill's text step (`begin`); else they are a
        /// tap step of their own that stays pending until `finish`. A step
        /// injecting nothing writes none. Steps left pending by an execute
        /// that failed are removed.
        pub fn step(
            &mut self,
            services: &HostServicesV1,
            stream: BundleStreamHandleV1,
            encoder: &Encoder,
            request: &[u8],
            prefill: &PhaseInputs,
        ) -> Result<(), String> {
            if self.taps.is_none() {
                return Ok(());
            }
            self.abandon()?;
            let Some(taps) = self.taps.as_mut() else {
                return Ok(());
            };
            let batch = BatchCodecV1::decode(request).map_err(|error| format!("{error:?}"))?;
            let step = self
                .occurrences
                .step(
                    &ARENA,
                    batch.media(),
                    &OccurrenceItem::of(&batch),
                    &self.kinds,
                )
                .map_err(|error| error.to_string())?;
            for (serial, record) in step.encodes.iter().zip(&batch.media().encodes) {
                let patches = [record.geometry[1], record.geometry[2]];
                self.grids
                    .insert(*serial, Grid::of(patches, self.side, record.output_rows)?);
            }
            keep_resident(&mut self.grids, &self.occurrences);
            let mut rows = block_rows(
                taps,
                &self.specs,
                &mut self.occurrences,
                &step,
                encoder.take_block_rows(),
            )?;
            let output = self
                .specs
                .iter()
                .find(|spec| spec.name == OUTPUT)
                .ok_or("taps.json declares no enc.vision.out")?;
            if taps.selected(output) {
                let reads = step
                    .reads
                    .iter()
                    .filter(|read| read.encoder == 0)
                    .copied()
                    .collect::<Vec<_>>();
                let (segments, runs) = cells(&reads, &self.grids)?;
                if !segments.is_empty() {
                    let arena = encoder.ranks[0].region("arena")?;
                    let row_bytes = u64::from(output.cols) * output.dtype.bytes();
                    let bytes =
                        bundle_runtime::media::read_runs(services, stream, arena, &runs, row_bytes)
                            .map_err(|error| error.to_string())?;
                    rows.insert(
                        0,
                        EncoderRows {
                            spec: output.clone(),
                            bytes,
                            segments,
                        },
                    );
                }
            }
            if rows.is_empty() {
                return Ok(());
            }
            if self.text {
                self.encoders = rows;
                return Ok(());
            }
            let program = "Prefill";
            let live = prefill.live_tokens();
            let dir = taps.next_step().map_err(|error| error.to_string())?;
            let mut capture = taps
                .begin(
                    &dir,
                    program,
                    encoder.ranks.len() as u32,
                    live,
                    &prefill.positions,
                )
                .map_err(|error| error.to_string())?;
            for tap in rows {
                capture.encoder(tap).map_err(|error| error.to_string())?;
            }
            capture
                .open(&prefill.tap_items(program, live, capture.rows(), taps.from()))
                .map_err(|error| error.to_string())?;
            self.pending.push(capture);
            Ok(())
        }

        /// Opens the tap step of a phase of `program` over `inputs` on
        /// `ranks` ranks, whose stages, in order, are `stages`: the
        /// prefill's encoder rows (`step`), then every selected text tap of
        /// each stage, declared in the order `capture` writes them.
        pub fn begin(
            &mut self,
            program: &str,
            ranks: u32,
            inputs: &PhaseInputs,
            shape: &Shape,
            stages: &[Stage],
        ) -> Result<(), String> {
            let Some(taps) = self.taps.as_mut() else {
                return Ok(());
            };
            if let Some(stale) = self.open.take() {
                stale.cancel().map_err(|error| error.to_string())?;
            }
            let live = inputs.live_tokens();
            let dir = taps.next_step().map_err(|error| error.to_string())?;
            let mut capture = taps
                .begin(&dir, program, ranks, live, &inputs.positions)
                .map_err(|error| error.to_string())?;
            for rows in std::mem::take(&mut self.encoders) {
                capture.encoder(rows).map_err(|error| error.to_string())?;
            }
            for &stage in stages {
                for tap in super::text::taps(stage, shape) {
                    if taps.selected(&tap.spec) {
                        capture
                            .declare(&tap.spec)
                            .map_err(|error| error.to_string())?;
                    }
                }
            }
            capture
                .open(&inputs.tap_items(program, live, capture.rows(), taps.from()))
                .map_err(|error| error.to_string())?;
            self.open = Some(capture);
            Ok(())
        }

        /// Copies the open step's rows of `stage`'s selected taps from
        /// `buffers`, each rank's on its stream, once the work queued there
        /// is done.
        pub fn capture(
            &mut self,
            services: &HostServicesV1,
            streams: &[BundleStreamHandleV1],
            buffers: &[RankBuffers],
            shape: &Shape,
            stage: Stage,
        ) -> Result<(), TapError> {
            let (Some(taps), Some(capture)) = (&self.taps, &mut self.open) else {
                return Ok(());
            };
            for tap in super::text::taps(stage, shape) {
                if !taps.selected(&tap.spec) {
                    continue;
                }
                for rank in 0..tap.spec.sharding.ranks(streams.len() as u32) {
                    let index = rank as usize;
                    let buffer = tap.buffer.span(&buffers[index].scratch);
                    capture.capture(services, streams[index], &tap.spec, rank, buffer)?;
                }
            }
            Ok(())
        }

        /// Leaves the open step pending until `finish`.
        pub fn end(&mut self) {
            if let Some(capture) = self.open.take() {
                self.pending.push(capture);
            }
        }

        /// Publishes the pending steps on a commit and removes them on a
        /// cancel, with a step a failed execute left open, and commits or
        /// drops the step's occurrences.
        pub fn finish(&mut self, commit: bool) -> Result<(), String> {
            self.occurrences.finish(commit);
            self.encoders.clear();
            let mut result = match self.open.take() {
                Some(stale) => stale.cancel(),
                None => Ok(()),
            };
            for capture in std::mem::take(&mut self.pending) {
                let ended = if commit {
                    capture.finish()
                } else {
                    capture.cancel()
                };
                result = result.and(ended);
            }
            result.map_err(|error| error.to_string())
        }

        /// Drops what a failed `encoder` run left: the block rows it read
        /// back before failing, whose record indices the next run reuses,
        /// and the steps a failed execute left (`abandon`).
        pub fn discard(&mut self, encoder: &Encoder) -> Result<(), String> {
            drop(encoder.take_block_rows());
            self.abandon()
        }

        /// Removes, never publishes, what a failed execute left: its
        /// encoder rows and its open and pending steps.
        pub fn abandon(&mut self) -> Result<(), String> {
            self.encoders.clear();
            let mut result = match self.open.take() {
                Some(stale) => stale.cancel(),
                None => Ok(()),
            };
            for stale in std::mem::take(&mut self.pending) {
                result = result.and(stale.cancel());
            }
            result.map_err(|error| error.to_string())
        }

        /// Forgets a retired request's occurrences.
        pub fn retire(&mut self, request: u64) {
            self.occurrences.retire(request);
        }
    }

    /// The block taps `taps` selects of each occurrence `step` begins: the
    /// encoder runtime's block rows of the step's encodes are kept per
    /// encode (`Occurrences::keep`), and each occurrence gets its encode's
    /// whole rows, in the item's row order, under its request, kind and
    /// item, whichever step or request ran the encode.
    fn block_rows(
        taps: &Taps,
        specs: &[TapSpec],
        occurrences: &mut Occurrences,
        step: &OccurrenceStep,
        rows: Vec<BlockRows>,
    ) -> Result<Vec<EncoderRows>, String> {
        let spec = |name: &str| {
            specs
                .iter()
                .find(|spec| spec.name == name && name != OUTPUT)
                .ok_or_else(|| format!("{name} is not a declared block tap"))
        };
        for block in rows {
            let spec = spec(&block.name)?;
            if !taps.selected(spec) {
                continue;
            }
            if block.bytes.len() as u64 % (u64::from(spec.cols) * spec.dtype.bytes()) != 0 {
                return Err(format!("{}: a partial row", block.name));
            }
            let encode = step
                .encodes
                .get(block.record)
                .ok_or_else(|| format!("{}: rows of no encode", block.name))?;
            occurrences.keep(*encode, block.name, block.bytes);
        }
        let mut out: Vec<EncoderRows> = Vec::new();
        for read in step.reads.iter().filter(|read| read.begins) {
            for (name, bytes) in occurrences.kept(read.encode) {
                let spec = spec(name)?;
                let rows = bytes.len() as u64 / (u64::from(spec.cols) * spec.dtype.bytes());
                let at = match out.iter().position(|tap| tap.spec.name == *name) {
                    Some(at) => at,
                    None => {
                        out.push(EncoderRows {
                            spec: spec.clone(),
                            bytes: Vec::new(),
                            segments: Vec::new(),
                        });
                        out.len() - 1
                    }
                };
                let tap = &mut out[at];
                let start = tap.rows();
                tap.segments.push(EncoderSegment {
                    start,
                    index: 0,
                    rows: rows as u32,
                    ..read.segment
                });
                tap.bytes.extend_from_slice(bytes);
            }
        }
        Ok(out)
    }
}

#[cfg(all(test, feature = "diagnostic-taps"))]
mod tests {
    use std::collections::BTreeMap;

    use bundle_runtime::media::{ArenaRun, OccurrenceRead};
    use runtime_bundle_sdk::taps::EncoderSegment;

    use bundle_runtime::media::{OccurrenceItem, Occurrences};
    use runtime_bundle_sdk::{BatchMediaEncodeV3, BatchMediaSectionV3};

    use super::enabled::{Grid, cells, keep_resident};

    fn read(encode: u64, arena_row: u64, index: u32, rows: u32) -> OccurrenceRead {
        OccurrenceRead {
            encode,
            encoder: 0,
            arena_row,
            segment: EncoderSegment {
                request: Some(7),
                kind: Some("image"),
                item: 0,
                start: 0,
                index,
                rows,
            },
            begins: index == 0,
        }
    }

    /// An image of 31 x 51 patches has 11 rows of 17 cells, a span of 200
    /// rows; its delimiters are no cells.
    #[test]
    fn the_grid_numbers_the_cells_between_the_delimiters() {
        let grid = Grid::of([31, 51], 3, 200).unwrap();
        assert_eq!(
            grid,
            Grid {
                rows: 11,
                columns: 17
            }
        );
        assert!(Grid::of([31, 51], 3, 201).is_err());
        assert_eq!(grid.cell(0), None);
        assert_eq!(grid.cell(1), Some(0));
        assert_eq!(grid.cell(17), Some(16));
        assert_eq!(grid.cell(18), None);
        assert_eq!(grid.cell(19), Some(17));
        assert_eq!(grid.cell(197), Some(186));
        assert_eq!(grid.cell(198), None);
        assert_eq!(grid.cell(199), None);
    }

    /// A whole span read gives one segment per cell row, numbered as cells,
    /// at the arena rows past each delimiter; a span read in two chunks
    /// gives the same cells.
    #[test]
    fn cells_skip_the_delimiter_rows() {
        let grids = BTreeMap::from([(
            3,
            Grid {
                rows: 2,
                columns: 3,
            },
        )]);
        let (segments, runs) = cells(&[read(3, 100, 0, 10)], &grids).unwrap();
        assert_eq!(
            segments
                .iter()
                .map(|s| (s.start, s.index, s.rows))
                .collect::<Vec<_>>(),
            [(0, 0, 3), (3, 3, 3)]
        );
        assert_eq!(
            runs,
            [
                ArenaRun {
                    source_row: 0,
                    arena_row: 101,
                    rows: 3
                },
                ArenaRun {
                    source_row: 3,
                    arena_row: 105,
                    rows: 3
                }
            ]
        );
        let (split, _) = cells(&[read(3, 100, 0, 3), read(3, 200, 3, 7)], &grids).unwrap();
        assert_eq!(
            split
                .iter()
                .map(|s| (s.start, s.index, s.rows))
                .collect::<Vec<_>>(),
            [(0, 0, 2), (2, 2, 1), (3, 3, 3)]
        );
        assert!(cells(&[read(4, 0, 0, 10)], &grids).is_err());
    }

    /// A grid lasts while its encode has an arena page mapped: an encode
    /// that rewrites the page drops the earlier one's.
    #[test]
    fn grids_follow_the_resident_encodes() {
        let mut occurrences = Occurrences::default();
        let mut grids = BTreeMap::new();
        let grid = Grid {
            rows: 2,
            columns: 3,
        };
        for _ in 0..2 {
            let media = BatchMediaSectionV3 {
                encodes: vec![BatchMediaEncodeV3 {
                    encoder: 0,
                    geometry: [1, 6, 9, 0],
                    output_rows: 10,
                    pages: vec![2],
                    payload_offset: 0,
                    payload_bytes: 0,
                }],
                ..BatchMediaSectionV3::default()
            };
            let item = OccurrenceItem {
                request: 7,
                prefill: Some(0),
                tokens: &[5],
            };
            let step = occurrences
                .step(&crate::encoders::ARENA, &media, &[item], &[(1, "image")])
                .unwrap();
            occurrences.finish(true);
            for &serial in &step.encodes {
                grids.insert(serial, grid);
            }
            keep_resident(&mut grids, &occurrences);
        }
        assert_eq!(grids, BTreeMap::from([(2, grid)]));
    }
}
