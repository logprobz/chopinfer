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
    use bundle_runtime::inputs::PhaseInputs;
    use bundle_runtime::media::{OccurrenceItem, Occurrences};
    use bundle_runtime::taps as shared;
    use model_glm5next_contract::contract::Shape;
    use runtime_bundle_abi::{BundleStreamHandleV1, HostServicesV1};
    use runtime_bundle_sdk::BatchCodecV1;
    use runtime_bundle_sdk::taps::{EncoderRows, TapError, TapSpec, TapStep, Taps};

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
            let text = shared::selects_text(&taps, &specs, &text)?;
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
            shared::selects_blocks(self.taps.as_ref(), &self.specs, OUTPUT)
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
            let mut rows = shared::block_rows(
                taps,
                OUTPUT,
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
            let capture = shared::begin(
                taps,
                "Prefill",
                encoder.ranks.len() as u32,
                prefill,
                &mut rows,
                std::iter::empty(),
            )?;
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
            let capture = shared::begin(
                taps,
                program,
                ranks,
                inputs,
                &mut self.encoders,
                stages.iter().flat_map(|&stage| {
                    super::text::taps(stage, shape)
                        .into_iter()
                        .map(|tap| tap.spec)
                }),
            )?;
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
                shared::capture(taps, capture, services, streams, &tap.spec, |index| {
                    buffers[index].get(tap.buffer)
                })?;
            }
            Ok(())
        }

        /// Leaves the open step pending until `finish`.
        pub fn end(&mut self) {
            shared::end(&mut self.open, &mut self.pending);
        }

        /// Publishes the pending steps on a commit and removes them on a
        /// cancel, and commits or drops the step's occurrences.
        pub fn finish(&mut self, commit: bool) -> Result<(), String> {
            self.occurrences.finish(commit);
            self.encoders.clear();
            shared::finish(&mut self.open, &mut self.pending, commit)
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
            shared::finish(&mut self.open, &mut self.pending, false)
        }

        /// Forgets a retired request's occurrences.
        pub fn retire(&mut self, request: u64) {
            self.occurrences.retire(request);
        }
    }
}
