//! GLM 5.3 Flash's taps (`runtime_bundle_sdk::taps`, declared in
//! `models/glm5next/taps.json`), written only by a bundle built with the
//! `diagnostic-taps` feature, with `CHOPINFER_TAPS_DIR` set at create.
//!
//! Encoder taps: each step whose prefill injects image rows writes the rows
//! each request injects (`enc.vision.out`, read from rank 0's media arena
//! after the encoder ran) and the transformer blocks of each image
//! occurrence it begins (`enc.vision.B<nn>.in` and `.out`, read back by the
//! shared encoder runtime, `Encoder::block_taps`, each item's rows in their
//! own order). Rows are named by request, media kind and the item's place
//! in its prompt, carried across the prompt's prefill chunks
//! (`bundle_runtime::media::Occurrences`).
//! `tools/qual/encoder_parity.py` compares them with the encoder
//! reference.
//!
//! Text taps (`text`): while the sites filter selects one, every target
//! prefill runs eagerly (never a captured prefill program), stage by stage
//! across the ranks, and after each stage its selected taps' rows are read
//! back from rank 0's scratch (`Instance::launch_tapped`); an eager verify
//! phase does too. Decode and verify graph buckets and the MTP drafter's
//! runs are not tapped. Text taps read the context-parallel size 1
//! program's buffers at its unfused sublayer boundaries, so they refuse a
//! context-parallel instance and the Sinkhorn mHC comm epilogue.
//! `tools/qual/text_layers.py` compares them with the reference runner's
//! taps.
//!
//! One tap step per phase: a prefill's encoder rows are written into its
//! text step when text taps are on, else into a step of their own. A
//! commit publishes the steps and a cancel or a failed execute removes
//! them. A serving bundle refuses to create while a taps variable is set.

#[cfg(feature = "diagnostic-taps")]
mod text;

#[cfg(feature = "diagnostic-taps")]
pub use text::specs as text_specs;

#[cfg(feature = "diagnostic-taps")]
pub use enabled::StepTaps;

#[cfg(not(feature = "diagnostic-taps"))]
pub use serving::StepTaps;

/// The model name each step's `capture.json` records.
#[cfg(feature = "diagnostic-taps")]
pub const MODEL: &str = "glm5next";

#[cfg(not(feature = "diagnostic-taps"))]
mod serving {
    use bundle_runtime::inputs::PhaseInputs;
    use runtime_bundle_abi::{BundleStreamHandleV1, HostServicesV1};

    use crate::config::Configuration;
    use bundle_runtime::encoder::Encoder;

    /// A serving build's taps: none, and a create that asks for them is
    /// refused.
    #[derive(Debug, Default)]
    pub struct StepTaps;

    impl StepTaps {
        /// Refuses a create while a taps variable is set.
        pub fn from_env(_config: &Configuration) -> Result<Self, String> {
            match runtime_bundle_sdk::taps::Taps::requested(None) {
                Some(variable) => Err(format!(
                    "{variable} is set, but this GLM 5.3 Flash bundle was built without the \
                     diagnostic-taps feature; unset it to serve"
                )),
                None => Ok(Self),
            }
        }

        pub fn selects_blocks(&self) -> bool {
            false
        }

        /// Whether text taps are written: never.
        pub fn text(&self) -> bool {
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

        pub fn abandon(&mut self) -> Result<(), String> {
            Ok(())
        }

        pub fn retire(&mut self, _request: u64) {}
    }
}

#[cfg(feature = "diagnostic-taps")]
mod enabled {
    use bundle_runtime::encoder::BlockRows;
    use bundle_runtime::inputs::PhaseInputs;
    use bundle_runtime::media::{OccurrenceItem, OccurrenceStep, Occurrences};
    use model_glm5next_contract::contract::Shape;
    use runtime_bundle_abi::{BundleStreamHandleV1, HostServicesV1};
    use runtime_bundle_sdk::BatchCodecV1;
    use runtime_bundle_sdk::taps::{EncoderRows, EncoderSegment, TapError, TapSpec, TapStep, Taps};

    use crate::buffers::RankBuffers;
    use crate::config::Configuration;
    use crate::encoders::ARENA;
    use crate::program::Stage;
    use bundle_runtime::encoder::Encoder;

    /// The one tower's output rows.
    const OUTPUT: &str = "enc.vision.out";

    /// The taps the environment configures, the image occurrences of every
    /// prompt, a prefill's encoder rows waiting for its text step, the
    /// phase step being captured and the steps a commit or cancel ends.
    #[derive(Default)]
    pub struct StepTaps {
        taps: Option<Taps>,
        /// The encoder's taps.
        specs: Vec<TapSpec>,
        /// Whether the filter selects a text tap.
        text: bool,
        kinds: Vec<(u32, &'static str)>,
        occurrences: Occurrences,
        encoders: Vec<EncoderRows>,
        open: Option<TapStep>,
        pending: Vec<TapStep>,
    }

    impl StepTaps {
        /// The taps the environment configures, if any, for `config`'s
        /// instance, whose prompts mark images with the image token;
        /// refuses a filter that selects no tap, and text taps on a
        /// context-parallel instance or with the comm epilogue. The
        /// create record's producer is the host's startup nonce each
        /// step's `capture.json` names.
        pub fn from_env(config: &Configuration) -> Result<Self, String> {
            let shape = config.shape;
            let specs = crate::encoders::taps()?;
            let Some(taps) = Taps::from_env(None)? else {
                return Ok(Self::default());
            };
            let taps = taps
                .with_model(super::MODEL)
                .with_producer(config.create.tap_producer.clone());
            let text = super::text::specs(shape);
            let every = specs.iter().chain(&text).cloned().collect::<Vec<_>>();
            taps.check_layers(&every)?;
            taps.check_sites(&every)?;
            let text = text.iter().any(|spec| taps.selected(spec));
            if text && config.context_parallel.size() > 1 {
                return Err(
                    "text taps read the context-parallel size 1 program; serve with \
                            --context-parallel-size 1 or select only enc.* taps"
                        .into(),
                );
            }
            if text && config.comm_epilogue {
                return Err(format!(
                    "text taps read the unfused sublayer boundaries; unset {} or select only \
                     enc.* taps",
                    model_glm5next_contract::options::COMM_EPILOGUE_OPTION
                ));
            }
            Ok(Self {
                taps: Some(taps),
                specs,
                text,
                kinds: vec![(shape.geometry.shape.tokens.image, "image")],
                ..Self::default()
            })
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

        /// Whether text taps are written: every target prefill then runs
        /// through `begin`, `capture` and `end`.
        pub fn text(&self) -> bool {
            self.text
        }

        /// Takes the step `request`'s encoder taps after `encoder` ran: the
        /// rows each prefill item injects and the blocks of each
        /// occurrence it begins, over the `prefill` rows. With text taps
        /// they wait for the prefill's text step (`begin`); else they are
        /// a tap step of their own that stays pending until `finish`. A
        /// step injecting nothing writes none. Steps left pending by an
        /// execute that failed are removed.
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
                let (segments, runs) = step.encoder_tap(0);
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
        /// prefill's encoder rows (`step`), then every selected text tap
        /// of each stage, declared in the order `capture` writes them.
        pub fn begin(
            &mut self,
            program: &str,
            ranks: u32,
            inputs: &PhaseInputs,
            shape: Shape,
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

        /// Copies the open step's rows of `stage`'s selected text taps from
        /// `buffers`, each rank's on its stream, once the work queued there
        /// is done.
        pub fn capture(
            &mut self,
            services: &HostServicesV1,
            streams: &[BundleStreamHandleV1],
            buffers: &[RankBuffers],
            shape: Shape,
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
                    capture.capture(
                        services,
                        streams[index],
                        &tap.spec,
                        rank,
                        buffers[index].get(tap.buffer),
                    )?;
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
        /// cancel, and commits or drops the step's occurrences.
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
