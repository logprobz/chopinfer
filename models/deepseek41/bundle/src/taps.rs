//! DeepSeek V4.1's taps (`runtime_bundle_sdk::taps`, declared in
//! `models/deepseek41/taps.json`), written only by a bundle built with the
//! `diagnostic-taps` feature, with `CHOPINFER_TAPS_DIR` set at create.
//!
//! While the sites filter selects a tap, every prefill runs stage by stage
//! across the ranks, and after each stage its selected taps' rows are read
//! back from rank 0's scratch (`text`); decode and verify graph buckets are
//! not tapped. `tools/qual/tapdiff.py` compares the steps with the
//! reference runner's taps (`models/deepseek41/reference.py`, `taps`).
//!
//! One tap step per tapped phase: a commit publishes the steps, and a
//! cancel or a failed execute removes them. A serving bundle refuses to
//! create while a taps variable is set.

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

        pub fn finish(&mut self, _commit: bool) -> Result<(), String> {
            Ok(())
        }
    }
}

#[cfg(feature = "diagnostic-taps")]
mod enabled {
    use bundle_runtime::inputs::PhaseInputs;
    use runtime_bundle_abi::{BundleStreamHandleV1, HostServicesV1};
    use runtime_bundle_sdk::taps::{TapError, TapStep, Taps};

    use crate::buffers::RankBuffers;
    use crate::config::Configuration;
    use crate::program::Stage;
    use crate::shape::Shape;

    /// The taps the environment configures, whether the filter selects
    /// any, the phase step being captured and the steps a commit or cancel
    /// ends.
    #[derive(Default)]
    pub struct StepTaps {
        taps: Option<Taps>,
        text: bool,
        open: Option<TapStep>,
        pending: Vec<TapStep>,
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
            let specs = super::text::specs(&config.shape);
            taps.check_layers(&specs)?;
            taps.check_sites(&specs)?;
            let text = specs.iter().any(|spec| taps.selected(spec));
            Ok(Self {
                taps: Some(taps),
                text,
                ..Self::default()
            })
        }

        /// Whether prefills are tapped, running stage by stage.
        pub fn text(&self) -> bool {
            self.text
        }

        /// Opens the tap step of a phase of `program` over `inputs` on
        /// `ranks` ranks, whose stages, in order, are `stages`: every
        /// selected tap of each stage, declared in the order `capture`
        /// writes them.
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
        /// cancel, with a step a failed execute left open.
        pub fn finish(&mut self, commit: bool) -> Result<(), String> {
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

        /// Removes, never publishes, what a failed execute left.
        pub fn abandon(&mut self) -> Result<(), String> {
            self.finish(false)
        }
    }
}
