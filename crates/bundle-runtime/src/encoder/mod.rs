//! The shared encoder runtime: a bundle's encoder declarations
//! (`encoder-spec`) lowered at create to one canonical launch list each
//! over the regions of their buffer plan and the views of the encoder image
//! (`lower`), run for every media record of a step by the recorded driver.
//!
//! - **Replicated.** Every rank runs every record. The record's program is
//!   recorded once against rank 0's regions, weights and symbols, relocated
//!   to every rank and submitted in round-robin upload windows
//!   (`recorded`, `submission`); every rank's stream drains before the next
//!   record, and a failure drains every stream it touched. With
//!   `SERIAL_ENV=1` each rank records and runs its own program, one rank
//!   after another. `PARALLEL_ENV` names the parallel policy; only
//!   `replicated` is implemented.
//! - **A record's program.** One program covers every run of a record.
//!   For a declaration of one run per item the payload and the side inputs
//!   (live row counts, sequence bounds, rotary rows, position taps; `side`)
//!   upload directly to their regions, then the launch list runs. For a
//!   staged one (`Declaration::staged`) the
//!   payload uploads directly and every side input is staged in command
//!   order: each run uploads its live row count (again as it changes) and
//!   its sequence bounds, then its stages' commands, and the first stage
//!   reads the run's rows of the payload. After each run its output rows
//!   copy to the record's arena pages.
//! - **Diagnostics.** `capture::CAPTURE_ENV` and `capture::ARENA_ENV`
//!   write captures from the arena rows each record wrote.
//!
//! The bundle states each slot of a declaration (stage, part, block) as the
//! query its catalog entry was generated from, or as its own prepared op for
//! a registered stage (`Encoder::with_backend`). The runtime prepares every
//! slot once on the bundle's `kernel_core::backend::Backend` when it lowers
//! the declaration, and binds each launch's buffers by role per record. The
//! queries come from the walk that generates the backend's encoder catalogs.

pub mod capture;
pub mod lower;
pub mod probe;
pub mod registered;
pub mod side;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use encoder_spec::Declaration;
use encoder_spec::buffers::{self, BufferPlan, Serving};
use encoder_spec::geometry::{Geometry, Media};
use kernel_core::backend::{Backend, Counts, Launch, View};
use kernel_core::contract::Refusal;
use kernel_core::multimodal::EmbeddingOverrideBuffers;
use runtime_bundle_abi::{
    BundleAbiError, BundleCudaSymbolHandleV1, BundleStatusV1, BundleStreamHandleV1,
    DeviceBufferSpanV1 as Span, HostServicesV1,
};
use runtime_bundle_sdk::{BatchCodecV1, BatchMediaEncodeV3, BatchRequestV1};

use crate::dispatch::{self, BufferBindings};
use crate::inputs::PhaseInputs;
use crate::media::{MediaArena, MediaStepPlan};
use crate::pipeline::{Collectives, PreparedLaunch};
use crate::recorded::{Program, Relocations};
use crate::resources::upload_bytes;
use crate::submission::{self, Timings};

pub use lower::{Lowered, Named, SlotOp, Slots};
pub use registered::{Registered, Shared};

/// Selects the parallel policy (`replicated`, the default and the only one
/// implemented).
pub const PARALLEL_ENV: &str = "CHOPINFER_ENCODER_PARALLEL";
/// `1` runs each rank's program alone, one rank after another.
pub const SERIAL_ENV: &str = "CHOPINFER_ENCODER_SERIAL";

/// How the ranks run a record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    /// One program relocated to every rank.
    Relocated,
    /// Each rank's own program, one rank after another (`SERIAL_ENV=1`).
    Serial,
}

impl Mode {
    /// The mode `PARALLEL_ENV` and `SERIAL_ENV` select.
    pub fn from_env(
        parallel: Option<std::ffi::OsString>,
        serial: Option<std::ffi::OsString>,
    ) -> Result<Self, String> {
        if let Some(policy) = parallel
            && policy != "replicated"
        {
            return Err(format!(
                "{PARALLEL_ENV}={}: only the replicated policy is implemented",
                policy.to_string_lossy()
            ));
        }
        Ok(if serial.as_deref() == Some("1".as_ref()) {
            Self::Serial
        } else {
            Self::Relocated
        })
    }
}

/// What a bundle states about its encoders besides their declarations.
#[derive(Clone, Copy)]
pub struct Config {
    /// The log lines' prefix.
    pub label: &'static str,
    /// The schema the capture manifests name.
    pub capture_schema: &'static str,
    pub arena: MediaArena,
    /// Rows of one prefill step: one i64 arena source row each.
    pub prefill_rows: u64,
    /// The host side of the bundle's registered variants.
    pub registered: &'static dyn Registered,
    /// Staging of the recorded uploads (staged declarations stage them all).
    pub staging_bytes: u64,
}

/// One transformer block's output rows of one record, or a quantizer
/// stage's codes, read back on rank 0 with `Encoder::block_taps` (or
/// `code_taps`): the item's
/// rows in their own order, as the stage stores them (F32 residual rows
/// under `f32-pairs.v1`, or u32 codes), every run's in run order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockRows {
    /// The record's index among the step's encodes.
    pub record: usize,
    pub encoder: usize,
    /// `enc.<tower>.B<nn>.out`, blocks numbered across the declaration's
    /// transformers, `B<nn>.in` for a transformer's first block, or
    /// `enc.<tower>.x.codes`.
    pub name: String,
    pub bytes: Vec<u8>,
}

/// One rank's regions and encoder image views.
pub struct Rank {
    /// Every region of the buffer plan, in allocation order, with the stage
    /// that owns it (`buffers::Region::owner`).
    pub regions: Vec<(&'static str, Option<usize>, Span)>,
    pub weights: BTreeMap<String, Span>,
    inject: lower::Inject,
    arena_rows: u64,
    /// This rank's index and the encoder's probe (`probe`).
    index: usize,
    probe: Option<Arc<probe::Probe>>,
}

impl Rank {
    /// The shared region `name`.
    pub fn region(&self, name: &str) -> Result<Span, String> {
        self.owned(name, None)
    }

    /// The region `name` that `owner` owns, or the shared one for `None`.
    pub fn owned(&self, name: &str, owner: Option<usize>) -> Result<Span, String> {
        self.regions
            .iter()
            .find(|&&(region, of, _)| region == name && of == owner)
            .map(|&(_, _, span)| span)
            .ok_or_else(|| format!("no encoder region `{name}` of {owner:?}"))
    }

    /// The prefill launch that overrides `rows` embedding rows of `embed`
    /// by their arena rows (`source-rows`; -1 keeps the token's row).
    pub fn inject(
        &self,
        embed: Span,
        live_tokens: Span,
        rows: u32,
    ) -> Result<PreparedLaunch, Refusal> {
        self.inject_part(embed, live_tokens, 0, rows)
    }

    /// Inject a contiguous part using its offset in the whole source-row map.
    pub fn inject_part(
        &self,
        embed: Span,
        live_tokens: Span,
        first_row: u32,
        rows: u32,
    ) -> Result<PreparedLaunch, Refusal> {
        let source = self.source_rows(first_row, rows)?;
        let arena = self
            .region("arena")
            .map_err(|_| Refusal("encoder region missing"))?;
        let mut bindings = BufferBindings::default();
        let mut view = |span: Span| View {
            handle: bindings.bind(span),
            bytes: span.length,
        };
        let buffers = EmbeddingOverrideBuffers {
            activation: view(embed),
            arena: view(arena),
            source: view(source),
            live_rows: view(live_tokens),
            arena_rows: self.arena_rows,
        };
        let [launch] = <[Launch; 1]>::try_from(self.inject.launches(
            &buffers,
            View::EMPTY,
            Counts::rows(rows),
        )?)
        .map_err(|_| Refusal("an embedding override is one launch"))?;
        if let Some(probe) = &self.probe {
            probe.line(self.index, &launch.symbol, launch.rows);
        }
        Ok(PreparedLaunch {
            symbol: launch.symbol,
            rows: launch.rows,
            plan: launch.plan,
            arguments: launch.arguments,
            bindings,
        })
    }

    /// The source rows of a prefill part from `first_row`, `rows` long.
    fn source_rows(&self, first_row: u32, rows: u32) -> Result<Span, Refusal> {
        let source = self
            .region("source-rows")
            .map_err(|_| Refusal("encoder region missing"))?;
        let skip = u64::from(first_row) * 8;
        if skip + u64::from(rows) * 8 > source.length {
            return Err(Refusal("encoder source rows exceed prepared capacity"));
        }
        Ok(Span::new(
            source.buffer,
            source.offset + skip,
            source.length - skip,
        ))
    }
}

/// Where a launch of a run binds its operands: the rank's regions and
/// views, the declaration, the run and the output rows.
type Place<'a> = (&'a Rank, &'a Lowered, &'a encoder_spec::geometry::Run, Span);

/// The live sequences of transformer `stage` in a run's side inputs.
fn sequences(side: &side::Side, stage: usize) -> Result<u32, String> {
    side.sequences
        .iter()
        .find(|(of, _)| *of == stage)
        .map(|&(_, sequences)| sequences)
        .ok_or_else(|| "a transformer without sequences".to_owned())
}

/// Why a step's encodes failed, by what may still be in flight.
pub(super) enum Fault {
    /// Nothing the step queued is in flight: a refusal on the host, a file
    /// system error, or a submission that drained every stream it touched.
    Drained(String),
    /// A device call outside the recorded submission failed, and may have
    /// queued work: every stream drains, and one that does not poisons.
    Device(String),
    /// A submission left a stream undrained: the instance is poisoned.
    Undrained(String),
}

/// A record admitted by its declaration: the declaration's index, the
/// item's geometry and the merge side of the recipe that admitted it.
struct Admitted {
    encoder: usize,
    geometry: Geometry,
    merge: u32,
}

/// A bundle's encoders on every rank.
pub struct Encoder {
    pub config: Config,
    pub plan: BufferPlan,
    /// Each declaration lowered, in the frontend's encoder order; `None`
    /// for an encoder the package leaves out, whose records are refused.
    pub encoders: Vec<Option<Lowered>>,
    pub ranks: Vec<Rank>,
    pub mode: Mode,
    /// `capture::CAPTURE_ENV`.
    pub capture: Option<PathBuf>,
    /// Set once a capture's file system failure disabled `capture`.
    capture_disabled: std::sync::atomic::AtomicBool,
    /// `capture::ARENA_ENV`.
    pub arena_capture: Option<PathBuf>,
    /// Whether rank 0 reads every transformer block's output rows back
    /// (`take_block_rows`), for a diagnostic bundle's taps.
    pub block_taps: bool,
    /// Whether rank 0 reads each record's quantizer codes back
    /// (`take_block_rows`, `enc.<tower>.x.codes`), for a diagnostic bundle's
    /// taps.
    pub code_taps: bool,
    block_rows: std::sync::Mutex<Vec<BlockRows>>,
    probe: Option<Arc<probe::Probe>>,
}

impl Encoder {
    /// Lowers `declarations` (the frontend's encoders, in order) onto one
    /// buffer plan, with each declaration's stages prepared on `backend`:
    /// `slots` resolves (declaration, stage, part, block) to the slot's
    /// query or the model's own op. Reads the diagnostics' variables, checks
    /// every launch once on every rank against the rank's image views at its
    /// full rows, and allocates every rank's regions in plan order with
    /// `allocate(rank, bytes)`.
    pub fn with_backend(
        services: &HostServicesV1,
        declarations: &[Option<&Declaration>],
        config: Config,
        backend: &dyn Backend,
        slots: &dyn Fn(usize, usize, &'static str, Option<u32>) -> Result<SlotOp, String>,
        weights: Vec<BTreeMap<String, Span>>,
        allocate: impl FnMut(usize, u64) -> Result<Span, BundleAbiError>,
    ) -> Result<Self, BundleAbiError> {
        let prefill_rows = u32::try_from(config.prefill_rows)
            .map_err(|_| BundleAbiError::InvalidRange("encoder prefill rows exceed 32 bits"))?;
        let lower = |index: usize, declaration: &Declaration, plan: &BufferPlan| {
            Lowered::with_backend(
                declaration,
                plan,
                backend,
                &|stage, part, block| slots(index, stage, part, block),
                config.registered,
                prefill_rows,
            )
        };
        Self::create(services, declarations, config, &lower, weights, allocate)
    }

    fn create(
        services: &HostServicesV1,
        declarations: &[Option<&Declaration>],
        config: Config,
        lower_one: &dyn Fn(usize, &Declaration, &BufferPlan) -> Result<Lowered, String>,
        weights: Vec<BTreeMap<String, Span>>,
        mut allocate: impl FnMut(usize, u64) -> Result<Span, BundleAbiError>,
    ) -> Result<Self, BundleAbiError> {
        let refuse = |error: String, what: &'static str| {
            let _ = services.log(format!("{} encoder: {error}", config.label).as_bytes());
            BundleAbiError::InvalidRange(what)
        };
        let mode = Mode::from_env(std::env::var_os(PARALLEL_ENV), std::env::var_os(SERIAL_ENV))
            .map_err(|error| refuse(error, "encoder parallel policy"))?;
        let lower = || {
            let serving = Serving {
                arena_rows: config.arena.rows,
                prefill_rows: config.prefill_rows,
                staging_bytes: config.staging_bytes,
            };
            let planned: Vec<&Declaration> = declarations.iter().flatten().copied().collect();
            let rows = planned
                .iter()
                .map(|declaration| u64::from(declaration.input.capacity_rows))
                .max()
                .unwrap_or(0);
            let plan = buffers::plan(&planned, rows, serving).map_err(|error| error.0)?;
            let encoders = declarations
                .iter()
                .enumerate()
                .map(|(index, declaration)| {
                    declaration
                        .map(|declaration| lower_one(index, declaration, &plan))
                        .transpose()
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok::<_, String>((plan, encoders))
        };
        let (plan, encoders) = lower().map_err(|error| refuse(error, "encoder plan"))?;
        let inject = encoders
            .iter()
            .flatten()
            .next()
            .map(|lowered| lowered.inject.clone())
            .ok_or_else(|| refuse("no lowered declaration".into(), "encoder plan"))?;
        let probe = probe::Probe::configured(config.label);
        let mut ranks = Vec::with_capacity(weights.len());
        for (rank, weights) in weights.into_iter().enumerate() {
            for lowered in encoders.iter().flatten() {
                for launch in &lowered.launches {
                    for operand in launch.operands() {
                        if let lower::Operand::Weight(name) = &operand
                            && !weights.contains_key(name)
                        {
                            return Err(refuse(
                                format!("rank {rank} has no encoder view {name}"),
                                "encoder image view",
                            ));
                        }
                    }
                }
                lowered.check(&weights).map_err(|error| {
                    refuse(format!("rank {rank}: {error}"), "encoder image view")
                })?;
            }
            let regions = plan
                .regions
                .iter()
                .map(|region| Ok((region.name, region.owner, allocate(rank, region.bytes)?)))
                .collect::<Result<Vec<_>, BundleAbiError>>()?;
            ranks.push(Rank {
                regions,
                weights,
                inject: inject.clone(),
                arena_rows: config.arena.rows,
                index: rank,
                probe: probe.clone(),
            });
        }
        Ok(Self {
            config,
            plan,
            encoders,
            ranks,
            mode,
            capture: std::env::var_os(capture::CAPTURE_ENV).map(PathBuf::from),
            capture_disabled: Default::default(),
            arena_capture: std::env::var_os(capture::ARENA_ENV).map(PathBuf::from),
            block_taps: false,
            code_taps: false,
            block_rows: Default::default(),
            probe,
        })
    }

    /// The lowered declarations of the encoders the package plans.
    pub fn lowered(&self) -> impl Iterator<Item = &Lowered> {
        self.encoders.iter().flatten()
    }

    /// The block rows the last `run` read back (`block_taps`).
    pub fn take_block_rows(&self) -> Vec<BlockRows> {
        self.block_rows
            .lock()
            .map(|mut rows| std::mem::take(&mut *rows))
            .unwrap_or_default()
    }

    /// Bytes of one arena row: the language model's BF16 hidden.
    fn row_bytes(&self) -> u64 {
        self.lowered().next().map_or(0, Lowered::row_bytes)
    }

    /// The record's declaration and geometry, or why its declaration
    /// refuses it.
    fn admit(&self, record: &BatchMediaEncodeV3) -> Result<Admitted, String> {
        let encoder = record.encoder as usize;
        let declaration = &self
            .encoders
            .get(encoder)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("no encoder {encoder}"))?
            .declaration;
        let mut refusals = Vec::new();
        for recipe in &declaration.input.recipes {
            let item = match recipe.admit_record(record.geometry) {
                Ok(item) => item,
                Err(error) => {
                    refusals.push(error);
                    continue;
                }
            };
            let media = match item.grid {
                Some(grid) => Media::Grid(grid),
                None => Media::Frames(item.rows),
            };
            let geometry = declaration
                .geometry(recipe.tag(), media)
                .map_err(|error| error.0)?;
            // The rows the span places, after the output layout: the
            // encoder's own rows under `layout.tokens.v1`, with the learned
            // delimiter rows under `layout.learned-delimiters.v1`.
            let output = geometry.rows[declaration.stages.len()].rows;
            let payload = item.rows
                * u64::from(declaration.input.columns)
                * declaration.input.payload.bytes();
            if u64::from(record.output_rows) != output || record.payload_bytes != payload {
                return Err(format!(
                    "{}: a record of {:?} has {} output rows and {} payload bytes, not {output} \
                     and {payload}",
                    declaration.name, record.geometry, record.output_rows, record.payload_bytes
                ));
            }
            return Ok(Admitted {
                encoder,
                geometry,
                merge: recipe.merge(),
            });
        }
        Err(format!("{}: {}", declaration.name, refusals.join("; ")))
    }

    /// Rank 0's regions, views and symbols bound to each rank's.
    fn relocations(
        &self,
        symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
    ) -> Result<Vec<Relocations>, String> {
        let canonical = &self.ranks[0];
        self.ranks
            .iter()
            .zip(symbols)
            .map(|(rank, rank_symbols)| {
                let mut bindings = Relocations::default();
                for ((_, _, from), (_, _, to)) in canonical.regions.iter().zip(&rank.regions) {
                    bindings.span(*from, *to)?;
                }
                for (name, &from) in &canonical.weights {
                    let to = rank
                        .weights
                        .get(name)
                        .ok_or_else(|| format!("missing rank view {name}"))?;
                    bindings.span(from, *to)?;
                }
                for (name, &from) in &symbols[0] {
                    let to = rank_symbols
                        .get(name)
                        .ok_or_else(|| format!("missing rank symbol {name}"))?;
                    bindings.symbol(from, *to);
                }
                Ok(bindings)
            })
            .collect()
    }

    /// The program of `record` on `rank`: every run's uploads, commands and
    /// copies of its output rows to the record's arena pages, at the rows the
    /// layout places them (with its learned rows) for the item whose rows
    /// after the last stage are `encoded`. `sides` holds each run's side
    /// inputs.
    #[allow(clippy::too_many_arguments)]
    fn program<'p>(
        &self,
        rank: usize,
        symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
        lowered: &Lowered,
        record: &BatchMediaEncodeV3,
        encoded: encoder_spec::components::Item,
        runs: &[encoder_spec::geometry::Run],
        payload: &'p [u8],
        sides: &'p [side::Side],
    ) -> Result<Program<'p>, String> {
        let regions = &self.ranks[rank];
        let start = |span: Span, bytes: u64| {
            if bytes > span.length {
                return Err("encoder upload exceeds its region".to_owned());
            }
            Ok(Span::new(span.buffer, span.offset, bytes))
        };
        let payload_region = regions.region(lowered.payload)?;
        // Staged uploads go through staging.
        let mut program = if lowered.staged {
            Program::new(regions.region("staging")?)
        } else {
            Program::direct()
        };
        // The payload precedes every command, so it uploads directly and
        // takes no staging.
        if lowered.staged {
            program.upload_direct(start(payload_region, payload.len() as u64)?, payload)?;
            for (name, bytes) in &lowered.record {
                program.upload(start(regions.region(name)?, bytes.len() as u64)?, bytes)?;
            }
        } else {
            let [side] = sides else {
                return Err("a record of several runs of an unstaged declaration".into());
            };
            program.upload_direct(start(payload_region, payload.len() as u64)?, payload)?;
            for ((name, owner), bytes) in &side.uploads {
                let region = regions.owned(name, *owner)?;
                program.upload_direct(start(region, bytes.len() as u64)?, bytes)?;
            }
        }
        let output = start(regions.region(lowered.output.0)?, lowered.output.1)?;
        let row_bytes = lowered.row_bytes();
        let arena = regions.region("arena")?;
        for (run, side) in runs.iter().zip(sides) {
            let first = run.steps.first().ok_or("a run without stages")?;
            // The live row count slot the device holds (staged declarations).
            let mut count = None;
            let mut stage_count = |program: &mut Program, slot: usize| -> Result<(), String> {
                if lowered.staged && count != Some(slot) {
                    let span = start(regions.region("count")?, 4)?;
                    program.upload(span, &side.counts[slot].to_le_bytes())?;
                    count = Some(slot);
                }
                Ok(())
            };
            stage_count(&mut program, lowered.stage_counts[first.stage][0])?;
            if side.bounds.len() > 1 {
                return Err("two transformers of one run share the sequence bounds".into());
            }
            for (_, bounds) in &side.bounds {
                let span = start(regions.region("sequences")?, bounds.len() as u64)?;
                program.upload(span, bounds)?;
            }
            for step in &run.steps {
                for op in &lowered.ops[step.stage] {
                    let launch = match op {
                        lower::Op::Launch(index) => &lowered.launches[*index],
                        lower::Op::Upload(input, region) => {
                            let bytes = side
                                .staged
                                .get(input)
                                .ok_or_else(|| format!("no side input {input:?}"))?;
                            let span = start(regions.region(region)?, bytes.len() as u64)?;
                            program.upload(span, bytes)?;
                            continue;
                        }
                        lower::Op::Copy {
                            from,
                            to,
                            row_bytes,
                        } => {
                            let bytes = step.rows[0] * row_bytes;
                            program.copy(
                                start(regions.region(from)?, bytes)?,
                                start(regions.region(to)?, bytes)?,
                            );
                            continue;
                        }
                        lower::Op::Tap {
                            name,
                            region,
                            row_bytes,
                        } => {
                            if self.block_taps && rank == 0 {
                                let bytes = step.rows[0] * row_bytes;
                                program.tap(start(regions.region(region)?, bytes)?, name.clone());
                            }
                            continue;
                        }
                        lower::Op::CodesTap {
                            name,
                            region,
                            row_bytes,
                            run_output,
                        } => {
                            if self.code_taps && rank == 0 {
                                let region = regions.region(region)?;
                                let (first, rows) = match run_output {
                                    true => (run.output.start, run.output.end - run.output.start),
                                    false => (0, step.rows[1]),
                                };
                                let (offset, bytes) = (first * row_bytes, rows * row_bytes);
                                if offset + bytes > region.length {
                                    return Err("tapped codes exceed their region".into());
                                }
                                let span = Span::new(region.buffer, region.offset + offset, bytes);
                                program.tap(span, name.clone());
                            }
                            continue;
                        }
                        lower::Op::Gather {
                            from,
                            to,
                            row_bytes,
                        } => {
                            let sources = run.gather.as_deref().ok_or("a gather without rows")?;
                            let (from, to) = (regions.region(from)?, regions.region(to)?);
                            let mut row = 0;
                            while row < sources.len() {
                                let mut rows = 1;
                                while row + rows < sources.len()
                                    && sources[row + rows] == sources[row] + rows as u64
                                {
                                    rows += 1;
                                }
                                let bytes = rows as u64 * row_bytes;
                                let span = |region: Span, first: u64| {
                                    let offset = first * row_bytes;
                                    if offset + bytes > region.length {
                                        return Err("gathered rows exceed their region");
                                    }
                                    Ok(Span::new(region.buffer, region.offset + offset, bytes))
                                };
                                program.copy(span(from, sources[row])?, span(to, row as u64)?);
                                row += rows;
                            }
                            continue;
                        }
                    };
                    let slot = launch.count;
                    let live = side.counts[slot];
                    let place = (regions, lowered, run, output);
                    stage_count(&mut program, slot)?;
                    let ready = &lowered.prepared[launch.op].1;
                    // The op is prepared for the declaration's rows.
                    let capacity = lowered.declaration.input.capacity_rows;
                    for (resolved, symbol, rows) in
                        self.resolve_op(symbols, place, (ready, &launch.roles), side, live)?
                    {
                        let recorded = crate::recorded::Rows {
                            live: rows,
                            capacity,
                        };
                        program.launch(resolved, &symbol, recorded);
                        probe::launch(&self.probe, &symbol, rows);
                    }
                }
            }
            // Only runs through the last stage write output rows.
            if run.steps.last().map(|step| step.stage) != Some(lowered.declaration.stages.len() - 1)
            {
                continue;
            }
            if let Some(delimiters) = &lowered.delimiters {
                let copies = lower::placed_copies(
                    &lowered.declaration.output.op,
                    encoded,
                    run.output.start..run.output.end,
                )?;
                for placed in copies {
                    let (source, rows) = match placed.source {
                        lower::PlacedSource::Output { first, rows } => (
                            Span::new(
                                output.buffer,
                                output.offset + first * row_bytes,
                                rows * row_bytes,
                            ),
                            rows,
                        ),
                        lower::PlacedSource::Learned(delimiter) => {
                            let name = &delimiters[match delimiter {
                                encoder_spec::placement::Delimiter::Start => 0,
                                encoder_spec::placement::Delimiter::Newline => 1,
                                encoder_spec::placement::Delimiter::End => 2,
                            }];
                            let view = *regions
                                .weights
                                .get(name)
                                .ok_or_else(|| format!("no encoder view {name}"))?;
                            (view, 1)
                        }
                    };
                    let placed_rows =
                        u32::try_from(placed.placed).map_err(|_| "placed rows exceed 32 bits")?;
                    let mut offset = 0;
                    for copy in self
                        .config
                        .arena
                        .runs(&record.pages, record.output_rows, placed_rows, rows as u32)
                        .map_err(|error| error.to_string())?
                    {
                        let bytes = u64::from(copy.rows) * row_bytes;
                        program.copy(
                            Span::new(source.buffer, source.offset + offset, bytes),
                            Span::new(
                                arena.buffer,
                                arena.offset + copy.arena_row * row_bytes,
                                bytes,
                            ),
                        );
                        offset += bytes;
                    }
                }
                continue;
            }
            let rows = (run.output.end - run.output.start) as u32;
            for copy in self
                .config
                .arena
                .runs(
                    &record.pages,
                    record.output_rows,
                    run.output.start as u32,
                    rows,
                )
                .map_err(|error| error.to_string())?
            {
                let bytes = u64::from(copy.rows) * row_bytes;
                program.copy(
                    Span::new(
                        output.buffer,
                        output.offset + u64::from(copy.source_row) * row_bytes,
                        bytes,
                    ),
                    Span::new(
                        arena.buffer,
                        arena.offset + copy.arena_row * row_bytes,
                        bytes,
                    ),
                );
            }
        }
        Ok(program)
    }

    /// The span `operand` binds in `run` over `regions`, none for an
    /// operand that is no buffer.
    fn span(
        (regions, lowered, run, output): Place,
        operand: &lower::Operand,
    ) -> Result<Option<Span>, String> {
        Ok(match operand {
            lower::Operand::Region(name) => Some(regions.region(name)?),
            lower::Operand::Side(name, stage) => Some(regions.owned(name, Some(*stage))?),
            lower::Operand::Output => Some(output),
            lower::Operand::Count(slot) => {
                let counts = regions.region("counts")?;
                Some(Span::new(
                    counts.buffer,
                    counts.offset + 4 * *slot as u64,
                    4,
                ))
            }
            lower::Operand::Weight(name) => Some(
                *regions
                    .weights
                    .get(name)
                    .ok_or_else(|| format!("no encoder view {name}"))?,
            ),
            lower::Operand::Payload => {
                let payload = regions.region(lowered.payload)?;
                let row = lowered.payload_row_bytes;
                let bytes = (run.input.end - run.input.start) * row;
                let offset = run.input.start * row;
                if offset + bytes > payload.length {
                    return Err("a run's payload rows exceed their region".into());
                }
                Some(Span::new(payload.buffer, payload.offset + offset, bytes))
            }
            lower::Operand::RunOutput(name, row_bytes) => {
                let region = regions.region(name)?;
                let offset = run.output.start * row_bytes;
                let bytes = (run.output.end - run.output.start) * row_bytes;
                if offset + bytes > region.length {
                    return Err("a run's output rows exceed their region".into());
                }
                Some(Span::new(region.buffer, region.offset + offset, bytes))
            }
            lower::Operand::Null | lower::Operand::Sequences(_) | lower::Operand::U32(_) => None,
        })
    }

    /// The launches of a prepared op over `live` rows of `run`, its roles
    /// bound to `regions`, with each launch's symbol and rows.
    fn resolve_op(
        &self,
        symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
        place: Place,
        (op, roles): (&lower::Ready, &lower::Roles),
        side: &side::Side,
        live: u32,
    ) -> Result<Vec<(dispatch::Resolved, String, u32)>, String> {
        let sequences = match roles {
            lower::Roles::Attention(_, stage) => sequences(side, *stage)?,
            _ => 0,
        };
        let mut bindings = BufferBindings::default();
        let launches = op.launches(roles, (live, sequences), &mut |operand| {
            let span = Self::span(place, operand)?.ok_or("an encoder role binds no buffer")?;
            Ok(View {
                handle: bindings.bind(span),
                bytes: span.length,
            })
        })?;
        launches
            .into_iter()
            .map(|launch| {
                if launch.collective || !launch.plan.pdl {
                    return Err(format!("{}: not a PDL launch of one rank", launch.symbol));
                }
                let resolved = dispatch::resolve(
                    symbols,
                    &launch.symbol,
                    launch.plan,
                    &launch.arguments,
                    &bindings.with_route(op.route(live, &launch)),
                )
                .map_err(|error| format!("{}: {error:?}", launch.symbol))?;
                Ok((resolved, launch.symbol, launch.rows))
            })
            .collect()
    }

    /// Runs the step's encodes on every rank and uploads each rank's prefill
    /// source rows. Returns whether the prefill injects any row. A malformed
    /// media section or a record its declaration refuses is refused before
    /// any device work. A device failure drains every stream; a stream that
    /// does not drain poisons `collectives`.
    pub fn run(
        &self,
        services: &HostServicesV1,
        streams: &[BundleStreamHandleV1],
        symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
        request: &[u8],
        prefill: &PhaseInputs,
        collectives: &Collectives,
    ) -> Result<bool, BundleStatusV1> {
        let label = self.config.label;
        let batch = BatchCodecV1::decode(request).map_err(|_| BundleStatusV1::InvalidArgument)?;
        let media = batch.media();
        let plan = self
            .config
            .arena
            .plan(media, prefill, batch.items().len())
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        let records = plan
            .encodes
            .iter()
            .map(|record| self.admit(record))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                let _ = services.log(format!("{label} encoder record: {error}").as_bytes());
                BundleStatusV1::InvalidArgument
            })?;
        let fault = self.encode(services, streams, symbols, &batch, &plan, &records);
        let error = match fault {
            Ok(injects) => return Ok(injects),
            Err(Fault::Drained(error)) => error,
            Err(Fault::Device(error)) => {
                // Drain every stream, each even after another fails.
                let undrained = streams
                    .iter()
                    .filter(|&&stream| services.synchronize_stream(stream).is_err())
                    .count();
                if undrained > 0 {
                    collectives.poison();
                }
                error
            }
            Err(Fault::Undrained(error)) => {
                collectives.poison();
                error
            }
        };
        let _ = services.log(format!("{label} encoder execution: {error}").as_bytes());
        Err(BundleStatusV1::InternalError)
    }

    /// The device work of `run`: the records' programs, the captures and
    /// the source rows.
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        services: &HostServicesV1,
        streams: &[BundleStreamHandleV1],
        symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
        batch: &BatchRequestV1,
        plan: &MediaStepPlan,
        records: &[Admitted],
    ) -> Result<bool, Fault> {
        let label = self.config.label;
        let media = batch.media();
        let failed = Fault::Drained;
        // A stream that did not drain may still use the regions the next
        // step writes.
        let submitted = |failure: submission::Failure<String>| match failure.undrained {
            true => Fault::Undrained(failure.error),
            false => Fault::Drained(failure.error),
        };
        let row_bytes = self.row_bytes();
        // A capture's file system failure disables it, and logs once; a
        // failed readback is a device fault.
        let disable = |error: String| {
            use std::sync::atomic::Ordering::Relaxed;
            if !self.capture_disabled.swap(true, Relaxed) {
                let message = format!("{label} encoder capture disabled: {error}");
                let _ = services.log(message.as_bytes());
            }
        };
        let kept = |result: Result<(), capture::Failed>| match result {
            Ok(()) => Ok(true),
            Err(capture::Failed::Device(error)) => Err(Fault::Device(error)),
            Err(capture::Failed::File(error)) => {
                disable(error);
                Ok(false)
            }
        };
        let enabled = !self
            .capture_disabled
            .load(std::sync::atomic::Ordering::Relaxed);
        let mut capture = match self
            .capture
            .as_ref()
            .filter(|_| enabled && !media.is_empty())
        {
            Some(dir) => match capture::Capture::begin(dir, row_bytes) {
                Ok(capture) => Some(capture),
                Err(failed) => {
                    kept(Err(failed))?;
                    None
                }
            },
            None => None,
        };
        // A record's rows, and the rows its joining stage gathered.
        let encoded = |capture: &mut Option<capture::Capture>,
                       record,
                       admitted: &Admitted|
         -> Result<(), Fault> {
            let Some(open) = capture else {
                return Ok(());
            };
            let rank = &self.ranks[0];
            let lowered = self.encoders[admitted.encoder]
                .as_ref()
                .expect("an admitted record's encoder is lowered");
            let joined = match lowered.joined {
                Some((region, stage)) => {
                    let span = rank.region(region).map_err(Fault::Drained)?;
                    let rows = admitted.geometry.rows[stage].rows;
                    let row_bytes = lowered
                        .declaration
                        .inputs()
                        .map_err(|e| Fault::Drained(e.0))?[stage + 1]
                        .data
                        .row_bytes();
                    Some((
                        Span::new(span.buffer, span.offset, rows * row_bytes),
                        row_bytes,
                    ))
                }
                None => None,
            };
            let arena = rank.region("arena").map_err(Fault::Drained)?;
            let result = open.encode(
                services,
                streams[0],
                (arena, self.config.arena),
                record,
                joined,
            );
            if !kept(result)? {
                *capture = None;
            }
            Ok(())
        };
        let relocations = match self.mode {
            Mode::Relocated if !records.is_empty() => self.relocations(symbols).map_err(failed)?,
            _ => Vec::new(),
        };
        for (index, (record, admitted)) in plan.encodes.iter().zip(records).enumerate() {
            let started = Instant::now();
            let start = record.payload_offset as usize;
            let payload = &media.payload[start..start + record.payload_bytes as usize];
            let lowered = self.encoders[admitted.encoder]
                .as_ref()
                .expect("an admitted record's encoder is lowered");
            let runs = &admitted.geometry.runs;
            let sides = runs
                .iter()
                .map(|run| {
                    side::inputs(
                        lowered,
                        self.config.registered,
                        &admitted.geometry,
                        admitted.merge,
                        run,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(failed)?;
            let program = |rank: usize| {
                // Launches of a program that failed to build never ran.
                if let Some(probe) = &self.probe {
                    probe.flush([]);
                }
                let encoded = admitted.geometry.rows[lowered.declaration.stages.len() - 1];
                self.program(
                    rank,
                    &symbols[rank],
                    lowered,
                    record,
                    encoded,
                    runs,
                    payload,
                    &sides,
                )
            };
            let mut timings = Timings::default();
            let mut prepare = Duration::ZERO;
            // Rank 0's block taps, in command order.
            let mut tapped = Vec::new();
            let result = match self.mode {
                Mode::Relocated => {
                    let prepared = Instant::now();
                    let program = program(0).map_err(failed)?;
                    if let Some(probe) = &self.probe {
                        probe.flush(0..self.ranks.len());
                    }
                    let ranks = relocations
                        .iter()
                        .zip(streams)
                        .enumerate()
                        .map(|(rank, (bindings, &stream))| {
                            program.bind_relocated(services, stream, bindings, rank == 0)
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(failed)?;
                    prepare += prepared.elapsed();
                    let submitted = submission::submit_timed(&ranks, || None, &mut timings);
                    tapped = ranks[0].take_taps();
                    submitted
                }
                Mode::Serial => (0..self.ranks.len()).try_for_each(|rank| {
                    let program = program(rank).map_err(|error| submission::Failure {
                        error,
                        undrained: false,
                    })?;
                    if let Some(probe) = &self.probe {
                        probe.flush([rank]);
                    }
                    let bound = program.bind(services, streams[rank]);
                    let submitted = submission::submit(std::slice::from_ref(&bound), || None);
                    tapped.extend(bound.take_taps());
                    submitted
                }),
            };
            result.map_err(submitted)?;
            if self.block_taps || self.code_taps {
                // Each block's rows of every run, in run order: the item's rows.
                let mut taps: Vec<(String, Vec<u8>)> = Vec::new();
                for (name, bytes) in tapped {
                    match taps.iter_mut().find(|(known, _)| *known == name) {
                        Some((_, rows)) => rows.extend(bytes),
                        None => taps.push((name, bytes)),
                    }
                }
                let mut sink = self
                    .block_rows
                    .lock()
                    .map_err(|_| failed("tap lock".into()))?;
                sink.extend(taps.into_iter().map(|(name, bytes)| BlockRows {
                    record: index,
                    encoder: admitted.encoder,
                    name,
                    bytes,
                }));
            }
            encoded(&mut capture, record, admitted)?;
            let _ = services.log(
                format!(
                    "{label} encoder rows={} patches={} encoder={} runs={} elapsed_ms={:.3} \
                     prepare_ms={:.3} submit_ms={:.3} drain_ms={:.3} ranks={} mode={:?}",
                    record.output_rows,
                    admitted.geometry.input.rows,
                    admitted.encoder,
                    runs.len(),
                    started.elapsed().as_secs_f64() * 1000.0,
                    prepare.as_secs_f64() * 1000.0,
                    timings.submit.as_secs_f64() * 1000.0,
                    timings.drain.as_secs_f64() * 1000.0,
                    self.ranks.len(),
                    self.mode,
                )
                .as_bytes(),
            );
        }
        if let Some(capture) = capture {
            let arena = self.ranks[0].region("arena").map_err(failed)?;
            kept(capture.finish(
                services,
                streams[0],
                (arena, self.config.arena),
                batch,
                self.config.capture_schema,
            ))?;
        }
        if let Some(root) = self.arena_capture.as_ref().filter(|_| !records.is_empty()) {
            let ranks = self
                .ranks
                .iter()
                .zip(streams)
                .map(|(rank, &stream)| Ok((stream, rank.region("arena")?)))
                .collect::<Result<Vec<_>, String>>()
                .map_err(failed)?;
            capture::arena(
                services,
                root,
                self.config.arena,
                row_bytes,
                &plan.encodes,
                &ranks,
            )
            .map_err(Fault::Device)?;
        }
        if !plan.injects() {
            return Ok(false);
        }
        let sources = plan
            .source_rows
            .iter()
            .flat_map(|row| row.to_le_bytes())
            .collect::<Vec<_>>();
        for (rank, &stream) in self.ranks.iter().zip(streams) {
            let span = rank.region("source-rows").map_err(failed)?;
            upload_bytes(services, stream, span, &sources)
                .map_err(|error| Fault::Device(error.to_string()))?;
        }
        Ok(true)
    }
}
