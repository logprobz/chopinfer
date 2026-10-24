//! The synchronous boundary between drafting and target verification.
//!
//! The host supplies prepared model operations and pinned Engram staging.
//! This module orders them: committed-feature catch-up of the drafting
//! items and of bulk commits, full draft block, bounded ID readback, CPU
//! hash/gather, upload, then target verification. Confidence is read only
//! for diagnostics and never selects serving rows.

use super::state::{
    self, BLOCK_ROWS, CommitPlan, DraftBlock, Kind, Projection, RowProjection, Rows, Snapshot,
    State,
};

const WIDTH: usize = BLOCK_ROWS as usize;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error<E> {
    State(&'static str),
    Host(E),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Input {
    pub rows: Rows,
    /// All prefill or host verification tokens, or only the bonus for decode.
    pub tokens: Vec<u32>,
    /// The target sampler's noise inputs for the item's drafts.
    pub noise: Noise,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftInput {
    pub block: DraftBlock,
    pub bonus: u32,
    pub noise: Noise,
}

/// The target sampler's noise inputs for one drafting item
/// (`kernel_core::draft_noise::NoiseRecord`): the request's seed, the
/// generated index of the token the target samples at draft position zero
/// (the item's first verify row) and the temperature's bits, zero for a
/// greedy item. Draft position `j` draws at `generated + j`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Noise {
    pub seed: u64,
    pub generated: u64,
    pub temperature: u64,
}

impl Noise {
    /// Drafts the argmax of the item's logits.
    pub const GREEDY: Noise = Noise {
        seed: 0,
        generated: 0,
        temperature: 0,
    };

    /// The noise of an item the host samples at `temperature`; a
    /// temperature outside the host sampler's range drafts greedily.
    pub fn sampled(seed: u64, generated: u64, temperature: f64) -> Noise {
        let record = kernel_core::draft_noise::NoiseRecord {
            seed,
            generated,
            temperature,
        };
        if temperature > 0.0 && record.validate().is_ok() {
            Noise {
                seed,
                generated,
                temperature: temperature.to_bits(),
            }
        } else {
            Noise::GREEDY
        }
    }

    pub fn record(self) -> kernel_core::draft_noise::NoiseRecord {
        kernel_core::draft_noise::NoiseRecord {
            seed: self.seed,
            generated: self.generated,
            temperature: f64::from_bits(self.temperature),
        }
    }

    /// The noise of the item's next cycle after `emitted` tokens: the
    /// accepted drafts and the target's correction or bonus token.
    pub fn after(self, emitted: u64) -> Noise {
        if self == Noise::GREEDY {
            return self;
        }
        Noise {
            generated: self.generated.wrapping_add(emitted),
            ..self
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetRows {
    pub rows: Rows,
    /// For a draft budget K: the bonus token followed by the first K drafts.
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DraftTokens {
    pub request: u64,
    pub tokens: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    NotEmitted,
    Accepted,
    /// Requires explicit verifier evidence; a retained-row count alone
    /// cannot establish rejection.
    Rejected,
    /// The commit callback does not say whether verification matched this
    /// proposal, stopped on it, or never reached it.
    Unknown,
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Diagnostic {
    pub request: u64,
    pub main_rows: u64,
    pub proposed: [u32; WIDTH],
    /// Raw F32 score bits preserve even non-finite diagnostic values without
    /// using them to accept, reject or trim a proposal.
    pub confidence_bits: [u32; WIDTH],
    pub outcomes: [Outcome; WIDTH],
}

/// Model program hooks. Device operations use the bundle's prepared backend
/// and OwnOps; host failures are terminal. Hash/gather works on provisional
/// token history and preserves masks for non-text target rows. It must not
/// publish request history before `commit_features` succeeds.
pub trait Host {
    type Error;
    type Gathered;

    /// Project one bulk pass of one stage, as an eager catch-up would.
    fn project_committed(&mut self, pass: Projection) -> Result<(), Self::Error>;
    /// Project the rows of every pass through every stage, in launches of
    /// at most the state's batch rows that may mix requests.
    fn project_rows(&mut self, passes: &[RowProjection]) -> Result<(), Self::Error>;
    /// Before a step whose items all decode and some draft, with each
    /// item's first target token: the host may start reading those rows'
    /// Engram table rows while the drafter runs. It never changes results.
    fn prefetch_engram(&mut self, _inputs: &[Input]) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Observe completed catch-up before a draft can change provisional rings.
    /// This diagnostic must not advance state or request another catch-up.
    #[cfg(feature = "state-capture")]
    fn observe_before_draft(
        &mut self,
        _state: &State,
        _inputs: &[Input],
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Compute all five positions, sequential Markov sampling and the
    /// confidence head, regardless of each block's emitted prefix length.
    fn draft(&mut self, inputs: &[DraftInput]) -> Result<(), Self::Error>;
    /// Copy exactly five int32 IDs per drafting item, after drafting finishes.
    fn read_draft_ids(&mut self, ids: &mut [[u32; WIDTH]]) -> Result<(), Self::Error>;
    fn read_confidence(&mut self, scores: &mut [[f32; WIDTH]]) -> Result<(), Self::Error>;
    /// An optional device target path, after draft production. A successful
    /// result means target verification and its output validation finished.
    /// All five IDs must be checked before publishing any logits or drafts.
    /// None means no work was issued and the ordinary host path is required.
    fn device_target(
        &mut self,
        _inputs: &[Input],
        _drafts: &[DraftInput],
    ) -> Result<Option<Vec<[u32; WIDTH]>>, Self::Error> {
        Ok(None)
    }
    /// Retain the returned target rows for commit after device execution.
    fn retain_target_rows(&mut self, _rows: &[TargetRows]) -> Result<(), Self::Error> {
        Ok(())
    }

    fn hash_and_gather(&mut self, rows: &[TargetRows]) -> Result<Self::Gathered, Self::Error>;
    fn upload_engram(&mut self, gathered: Self::Gathered) -> Result<(), Self::Error>;
    /// Run every target row and stage its raw attention-input features.
    fn target(&mut self, rows: &[TargetRows]) -> Result<(), Self::Error>;
    /// Apply the entire validated plan on every rank, including direct
    /// staged projections for long prefill chunks, before releasing staging.
    fn commit_features(&mut self, plan: &CommitPlan) -> Result<(), Self::Error>;
    /// Diagnostics never change the serving result, even if persistence fails.
    fn record_diagnostics(&mut self, diagnostics: &[Diagnostic]);
}

pub struct Runtime {
    state: State,
    vocab: u32,
    projection_rows: u32,
    diagnostics: Vec<Diagnostic>,
}

impl Runtime {
    pub fn new(state: State, vocab: u32, projection_rows: u32) -> state::Result<Self> {
        if vocab == 0 || projection_rows == 0 || state.is_pending() || state.is_poisoned() {
            return Err("invalid DSpark execution setup");
        }
        Ok(Self {
            state,
            vocab,
            projection_rows,
            diagnostics: Vec::new(),
        })
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    /// The private preview completed these exact projections on every rank.
    /// A real host commit must first produce the identical committed state.
    pub(crate) fn adopt_projections(
        &mut self,
        expected: &State,
        passes: &[state::RowProjection],
    ) -> state::Result<()> {
        if &self.state != expected || self.state.is_pending() {
            return Err("DSpark successor commit differs");
        }
        let mut projected = self.state.clone();
        for pass in passes {
            projected.skip_unread(pass.request)?;
            projected.projected_rows(*pass)?;
        }
        self.state = projected;
        Ok(())
    }

    fn host_result<T, E>(&mut self, result: Result<T, E>) -> Result<T, Error<E>> {
        result.map_err(|error| {
            self.state.poison();
            Error::Host(error)
        })
    }

    /// Also used before a prefix snapshot while there is no target step.
    pub fn catch_up<H: Host>(&mut self, host: &mut H, request: u64) -> Result<(), Error<H::Error>> {
        self.catch_up_all(host, &[request])
    }

    /// Batched launches project every request's readable deferred rows,
    /// then each bulk commit runs the per-stage passes of an eager catch-up.
    fn catch_up_all<H: Host>(
        &mut self,
        host: &mut H,
        requests: &[u64],
    ) -> Result<(), Error<H::Error>> {
        let mut passes = Vec::new();
        for &request in requests {
            self.state.skip_unread(request).map_err(Error::State)?;
            if let Some(pass) = self.state.row_projection(request).map_err(Error::State)? {
                passes.push(pass);
            }
        }
        if !passes.is_empty() {
            self.host_result(host.project_rows(&passes))?;
            for pass in passes {
                self.state.projected_rows(pass).map_err(Error::State)?;
            }
        }
        for &request in requests {
            if !self.state.has_bulk(request) {
                continue;
            }
            for stage in 0..self.state.config().stages {
                while let Some(pass) = self
                    .state
                    .projection(request, stage, self.projection_rows)
                    .map_err(Error::State)?
                {
                    self.host_result(host.project_committed(pass))?;
                    self.state.projected(pass).map_err(Error::State)?;
                }
            }
        }
        Ok(())
    }

    pub fn execute<H: Host>(
        &mut self,
        host: &mut H,
        inputs: &[Input],
    ) -> Result<Vec<DraftTokens>, Error<H::Error>> {
        // Reject bad caller input before opening the transaction or doing IO.
        for input in inputs {
            let known = match input.rows.kind {
                Kind::Prefill { rows } | Kind::Verify { rows } => rows as usize,
                Kind::Decode { .. } => 1,
            };
            if input.tokens.len() != known || input.tokens.iter().any(|&token| token >= self.vocab)
            {
                return Err(Error::State("invalid DSpark input tokens"));
            }
        }
        self.state
            .prepare(inputs.iter().map(|input| input.rows).collect())
            .map_err(Error::State)?;
        self.diagnostics.clear();
        // The device target's steps: each item's first row is known now.
        if inputs.iter().any(|input| input.rows.kind.drafts() > 0)
            && inputs
                .iter()
                .all(|input| matches!(input.rows.kind, Kind::Decode { .. }))
        {
            self.host_result(host.prefetch_engram(inputs))?;
        }
        // Items that do not draft defer their rows to their next drafting
        // step; a bulk commit catches up now, as it would eagerly.
        let catching_up = inputs
            .iter()
            .map(|input| input.rows.request)
            .zip(inputs.iter().map(|input| input.rows.kind.drafts() > 0))
            .filter(|&(request, drafting)| {
                self.state.needs_catch_up(request) && (drafting || self.state.has_bulk(request))
            })
            .map(|(request, _)| request)
            .collect::<Vec<_>>();
        self.catch_up_all(host, &catching_up)?;
        #[cfg(feature = "state-capture")]
        if inputs.iter().any(|input| input.rows.kind.drafts() > 0) {
            let observed = host.observe_before_draft(&self.state, inputs);
            self.host_result(observed)?;
        }
        let blocks = self.state.begin_execution().map_err(Error::State)?;
        let drafting = inputs
            .iter()
            .enumerate()
            .filter(|(_, input)| input.rows.kind.drafts() > 0)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let mut target = inputs
            .iter()
            .map(|input| TargetRows {
                rows: input.rows,
                tokens: input.tokens.clone(),
            })
            .collect::<Vec<_>>();
        let mut emitted = Vec::with_capacity(blocks.len());
        let mut device_target = false;
        if !blocks.is_empty() {
            let draft_inputs = blocks
                .iter()
                .zip(&drafting)
                .map(|(&block, &index)| DraftInput {
                    block,
                    bonus: inputs[index].tokens[0],
                    noise: inputs[index].noise,
                })
                .collect::<Vec<_>>();
            self.host_result(host.draft(&draft_inputs))?;
            let ids = match self.host_result(host.device_target(inputs, &draft_inputs))? {
                Some(ids) => {
                    device_target = true;
                    ids
                }
                None => {
                    let mut ids = vec![[u32::MAX; WIDTH]; blocks.len()];
                    self.host_result(host.read_draft_ids(&mut ids))?;
                    ids
                }
            };
            if ids.len() != blocks.len() {
                self.state.poison();
                return Err(Error::State(
                    "DSpark device result count differs from its blocks",
                ));
            }
            if ids.iter().flatten().any(|&token| token >= self.vocab) {
                self.state.poison();
                return Err(Error::State(
                    "DSpark device draft ID is outside the vocabulary",
                ));
            }
            for ((block, &index), proposed) in blocks.iter().zip(&drafting).zip(ids) {
                let count = block.emit_rows as usize;
                target[index].tokens.extend_from_slice(&proposed[..count]);
                emitted.push(DraftTokens {
                    request: block.request,
                    tokens: proposed[..count].to_vec(),
                });
                let mut outcomes = [Outcome::NotEmitted; WIDTH];
                outcomes[..count].fill(Outcome::Unknown);
                self.diagnostics.push(Diagnostic {
                    request: block.request,
                    main_rows: block.main_rows,
                    proposed,
                    confidence_bits: [0; WIDTH],
                    outcomes,
                });
            }
        }
        if device_target {
            self.host_result(host.retain_target_rows(&target))?;
        } else {
            let gathered = self.host_result(host.hash_and_gather(&target))?;
            self.host_result(host.upload_engram(gathered))?;
            self.host_result(host.target(&target))?;
        }
        if !blocks.is_empty() {
            let mut scores = vec![[f32::NAN; WIDTH]; blocks.len()];
            self.host_result(host.read_confidence(&mut scores))?;
            for (diagnostic, confidence) in self.diagnostics.iter_mut().zip(scores) {
                diagnostic.confidence_bits = confidence.map(f32::to_bits);
            }
        }
        self.state.executed().map_err(Error::State)?;
        Ok(emitted)
    }

    pub fn commit<H: Host>(
        &mut self,
        host: &mut H,
        kept: &[(u64, u32)],
    ) -> Result<Vec<Diagnostic>, Error<H::Error>> {
        let plan = self.state.begin_commit(kept).map_err(Error::State)?;
        self.host_result(host.commit_features(&plan))?;
        self.state.complete_commit(plan).map_err(Error::State)?;
        for diagnostic in &mut self.diagnostics {
            let count = kept
                .iter()
                .find(|&&(request, _)| request == diagnostic.request)
                .expect("validated drafting acceptance")
                .1;
            // These leading matches are certain. The next emitted token
            // may be an accepted stop token, a token-limit boundary or a
            // rejection; the callback reports none of those distinctions.
            // Leave it and later proposals unknown for calibration.
            diagnostic.outcomes[..count as usize - 1].fill(Outcome::Accepted);
        }
        let diagnostics = std::mem::take(&mut self.diagnostics);
        host.record_diagnostics(&diagnostics);
        Ok(diagnostics)
    }

    pub fn cancel<H: Host>(&mut self, host: &mut H) -> Result<Vec<Diagnostic>, Error<H::Error>> {
        self.state.cancel().map_err(Error::State)?;
        for diagnostic in &mut self.diagnostics {
            for outcome in &mut diagnostic.outcomes {
                if *outcome != Outcome::NotEmitted {
                    *outcome = Outcome::Cancelled;
                }
            }
        }
        let diagnostics = std::mem::take(&mut self.diagnostics);
        host.record_diagnostics(&diagnostics);
        Ok(diagnostics)
    }

    /// The owner first calls state().validate_restore and copies the raw
    /// snapshot. A failed copy must call poison(), never restore_after_copy.
    pub fn restore_after_copy(
        &mut self,
        request: u64,
        slot: u32,
        snapshot: Snapshot,
    ) -> state::Result<()> {
        self.state.restore(request, slot, snapshot)
    }

    /// After a successful snapshot copy (preemption), or final retirement.
    pub fn retire(&mut self, request: u64) -> state::Result<()> {
        self.state.retire(request)
    }

    pub fn poison(&mut self) {
        self.state.poison();
    }
}

#[cfg(test)]
mod tests;
