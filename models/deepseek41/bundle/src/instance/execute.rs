//! One step: pack the batch, open the ledger's step, and for its eager
//! prefill, then its decode or verify phase, upload the phase inputs and
//! issue each rank's lowered program; then emit the logits rows.
//!
//! The program is complete only once every stage lowers: until then
//! execute refuses with `Unsupported` before the step opens, naming the
//! first unlowered stage. The fake-host tests run the lowered stages alone
//! (`Instance::partial`) to drive the step through the ledger.

use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::graphs::GraphKey;
use bundle_runtime::inputs::{PackedStep, PhaseInputs};
use bundle_runtime::ledger::Item;
use bundle_runtime::logits;
use bundle_runtime::pipeline::{
    self, PoisonOnUnwind, PreparedLaunch, Program, Unit, check_collectives, collective_segments,
    launch_segments,
};
use bundle_runtime::resources::{status, upload_bytes};
use bundle_runtime::upload::{DeviceInputs, InputLayout};
use model_deepseek41_contract::serving::{DeepSeek, MAX_VERIFY_ROWS, SCORED_ROWS};
use runtime_bundle_abi::{
    BUNDLE_LOGITS_ELEMENT_V1_F32, BundleAbiError, BundleStatusV1, BundleStreamHandleV1,
    ExecuteResultSinkV1,
};
use runtime_bundle_sdk::BatchCodecV1;

use super::Instance;
use crate::lower::Lowering;
use crate::program::{Phase, Stage};

/// Phase-arena indexes (`Instance::arenas`).
const PREFILL: usize = 0;
const DECODE: usize = 1;
const VERIFY: usize = 2;

/// One phase of a step, planned before the step opens.
struct Run {
    inputs: PhaseInputs,
    arena: usize,
    layout: InputLayout,
    /// Rows the head scores.
    head_rows: u32,
    /// A decode or verify phase's graph bucket: (tokens, items), items 0
    /// for decode. An eager prefill has none.
    graph: Option<(u32, u32)>,
    /// Each rank's launches and whether each waits for its peers: empty
    /// for a rank whose graph of the bucket was captured when the step was
    /// planned, which replays it instead.
    programs: Vec<Vec<PreparedLaunch>>,
    collectives: Vec<Vec<bool>>,
}

/// A phase, its arena index, its input layout and its graph key.
type PhasePlan = (Phase, usize, InputLayout, Option<(u32, u32)>);

/// Each rank's launches and whether each waits for its peers.
type Programs = (Vec<Vec<PreparedLaunch>>, Vec<Vec<bool>>);

/// The ledger items of `phase`, in its item order.
fn items(phase: &PhaseInputs) -> Vec<Item> {
    (0..phase.request_ids.len())
        .map(|item| Item {
            request: phase.request_ids[item],
            slot: u32::try_from(phase.slot_ids[item]).ok(),
            start: phase.prefix_len[item] as u64,
            rows: phase.q_len.get(item).copied().unwrap_or(1) as u32,
            verify: phase.verify && phase.verify_rows.get(item).copied().unwrap_or(false),
        })
        .collect()
}

impl Instance {
    /// Packs a step's prefill and decode or verify items on the host's
    /// state slots. Media and bundle drafts wait for the encoder and the
    /// DSpark drafter.
    fn pack(&self, request: &[u8]) -> Result<PackedStep, BundleStatusV1> {
        let batch = BatchCodecV1::decode(request).map_err(|_| BundleStatusV1::InvalidArgument)?;
        if !batch.media().is_empty() {
            return Err(BundleStatusV1::Unsupported);
        }
        let step = PackedStep::pack_on_host_slots::<DeepSeek>(
            super::model_limits(&self.config),
            &batch,
            self.kv_pages,
            self.verify,
            self.items(),
            1,
        )?;
        if step.bundle_drafts(&batch).iter().any(|&drafts| drafts > 0) {
            return Err(BundleStatusV1::Unsupported);
        }
        Ok(step)
    }

    /// The phase of `inputs`, its arena index, its input layout and its
    /// graph key: (tokens, items) of a verify bucket, (bucket, 0) of a
    /// decode bucket, none for an eager phase.
    fn phase_of(&self, inputs: &PhaseInputs) -> Result<PhasePlan, BundleStatusV1> {
        let items = self.items();
        let live = inputs.live_batch();
        if !inputs.decode {
            let layout =
                InputLayout::prefill_within::<DeepSeek>(inputs, inputs.logit_rows().len(), items);
            let phase = Phase::Prefill {
                rows: inputs.live_tokens(),
            };
            Ok((phase, PREFILL, layout, None))
        } else if inputs.verify {
            let rows = bundle_runtime::graphs::verify_bucket_within::<DeepSeek>(
                inputs.live_tokens(),
                items,
            )
            .ok_or(BundleStatusV1::InvalidArgument)?;
            // A graph's item bucket holds the live items and, at most
            // `MAX_VERIFY_ROWS` rows each, the token bucket. An instance
            // too narrow for the padded tokens runs the phase eagerly at
            // its live rows and items instead.
            let held = rows.div_ceil(MAX_VERIFY_ROWS).next_power_of_two();
            let live_bucket = items.bucket(live).ok_or(BundleStatusV1::InvalidArgument)?;
            let layout = InputLayout::verify_within::<DeepSeek>(items);
            let items_bucket = live_bucket.max(held);
            if items_bucket > items.capacity() {
                let phase = Phase::Verify {
                    rows: inputs.live_tokens(),
                    items: live_bucket,
                };
                return Ok((phase, VERIFY, layout, None));
            }
            let phase = Phase::Verify {
                rows,
                items: items_bucket,
            };
            Ok((phase, VERIFY, layout, Some((rows, items_bucket))))
        } else {
            let bucket = items.bucket(live).ok_or(BundleStatusV1::InvalidArgument)?;
            Ok((
                Phase::Decode(bucket),
                DECODE,
                InputLayout::decode_within(items),
                Some((bucket, 0)),
            ))
        }
    }

    /// Each rank's launches of `phase` and whether each waits for its
    /// peers: every lowered stage, in program order. An unlowered stage
    /// refuses the step unless the instance runs the lowered stages alone.
    /// Only the ranks `lower` marks are lowered; the others get empty
    /// programs.
    fn program(
        &self,
        phase: Phase,
        arena: usize,
        layout: &InputLayout,
        lower: &[bool],
    ) -> Result<Programs, BundleStatusV1> {
        let stages = crate::program::stages(self.config.shape.geometry, phase);
        let mut programs = Vec::new();
        let mut collectives = Vec::new();
        for (rank, buffers) in self.buffers.iter().enumerate() {
            if !lower.get(rank).copied().unwrap_or(true) {
                programs.push(Vec::new());
                collectives.push(Vec::new());
                continue;
            }
            let inputs = DeviceInputs::within(self.arenas[arena][rank], layout);
            let lowering = Lowering {
                shape: &self.config.shape,
                ops: &self.ops,
                buffers,
                inputs: &inputs,
                rank: rank as u32,
            };
            let mut launches = Vec::new();
            let mut waits = Vec::new();
            for &stage in &stages {
                let lowered = match lowering.stage(phase, stage) {
                    Ok(Some(lowered)) => lowered,
                    Ok(None) if self.partial => continue,
                    Ok(None) => {
                        self.log(&format!("stage {stage:?} is not lowered yet"));
                        return Err(BundleStatusV1::Unsupported);
                    }
                    Err(error) => {
                        self.log(&error);
                        return Err(BundleStatusV1::InternalError);
                    }
                };
                for lowered in lowered {
                    waits.push(lowered.collective);
                    launches.push(lowered.launch);
                }
            }
            programs.push(launches);
            collectives.push(waits);
        }
        Ok((programs, collectives))
    }

    /// Plans one phase of the step: every rank's program.
    fn plan_run(&self, inputs: PhaseInputs) -> Result<Run, BundleStatusV1> {
        let (phase, arena, layout, graph) = self.phase_of(&inputs)?;
        let (head_rows, capacity) = if arena == PREFILL {
            (inputs.logit_rows().len() as u32, SCORED_ROWS)
        } else {
            (phase.rows(), crate::buffers::head_rows())
        };
        if head_rows > capacity {
            self.log(&format!(
                "{head_rows} scored rows exceed the head's {capacity} rows"
            ));
            return Err(BundleStatusV1::Unsupported);
        }
        let lower = self.missing_ranks(graph);
        let (programs, collectives) = self.program(phase, arena, &layout, &lower)?;
        Ok(Run {
            inputs,
            arena,
            layout,
            head_rows,
            graph,
            programs,
            collectives,
        })
    }

    /// Which ranks need their program lowered: every rank of an eager run,
    /// and of a graph run the ranks whose graph of the bucket is not
    /// captured yet.
    fn missing_ranks(&self, graph: Option<(u32, u32)>) -> Vec<bool> {
        let ranks = self.buffers.len();
        let Some((bucket, items)) = graph else {
            return vec![true; ranks];
        };
        (0..ranks)
            .map(|rank| {
                !self.graphs.contains(GraphKey {
                    rank: rank as u32,
                    bucket,
                    items,
                })
            })
            .collect()
    }

    /// Uploads `run`'s packed inputs and the changed block-table rows to
    /// every rank's phase arena.
    fn upload(&mut self, run: &Run) -> Result<(), BundleAbiError> {
        // Only a prefill gathers its logits rows; decode rows are its items.
        let rows = if run.arena == PREFILL {
            run.inputs.logit_rows()
        } else {
            Vec::new()
        };
        let bytes = run
            .layout
            .pack(&run.inputs, &rows)
            .ok_or(BundleAbiError::InvalidRange(
                "deepseek41 inputs exceed their arena",
            ))?;
        let writes = self.mirrors[run.arena].update(&run.inputs.block_tables);
        let mirror = &self.mirrors[run.arena];
        let uploaded = (|| {
            for (rank, &(_, _, stream)) in self.ranks.iter().enumerate() {
                let phase = self.arenas[run.arena][rank];
                upload_bytes(&self.services, stream, phase.arena, &bytes)?;
                mirror.upload(&self.services, stream, phase.block_tables, &writes)?;
            }
            Ok(())
        })();
        if uploaded.is_err() {
            self.mirrors[run.arena].forget();
        }
        uploaded
    }

    /// Issues `run`'s uploaded inputs' programs on every rank as one unit
    /// of collective work: eagerly, every rank up to each collective in
    /// turn, or by replaying a decode or verify phase's graphs, each rank's
    /// program captured at its first step of the bucket before any rank
    /// launches. A failure or panic once a rank's share was queued poisons
    /// the instance: its peers would wait forever at the collectives that
    /// rank queued.
    fn launch(&mut self, run: &mut Run) -> Result<(), BundleAbiError> {
        let streams = self
            .ranks
            .iter()
            .map(|&(_, _, stream)| stream)
            .collect::<Vec<_>>();
        let mut programs = std::mem::take(&mut run.programs);
        let mut collectives = std::mem::take(&mut run.collectives);
        if run.graph.is_none() {
            return self
                .launch_eager(&streams, &programs, &collectives)
                .map_err(|error| self.dispatch_failed(error));
        }
        pipeline::run_ranks(
            &self.services,
            &streams,
            &self.symbols,
            run.graph,
            &mut self.graphs,
            &self.collectives,
            |rank| {
                Ok(Program {
                    launches: std::mem::take(&mut programs[rank]),
                    collective: std::mem::take(&mut collectives[rank]),
                })
            },
            |_| {},
        )
        .map_err(|error| self.dispatch_failed(error))
    }

    fn dispatch_failed(&self, error: DispatchError) -> BundleAbiError {
        self.log(&format!("dispatch failed: {error:?}"));
        match error {
            DispatchError::Host(error) => error,
            _ => BundleAbiError::Status(BundleStatusV1::InternalError),
        }
    }

    /// Issues an eager run's segments on every rank, up to each collective
    /// in turn. Ranks whose programs differ in collectives are refused
    /// before anything is queued.
    fn launch_eager(
        &self,
        streams: &[BundleStreamHandleV1],
        programs: &[Vec<PreparedLaunch>],
        collectives: &[Vec<bool>],
    ) -> Result<(), DispatchError> {
        if self.collectives.is_poisoned() {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState).into());
        }
        if programs.len() != streams.len() || self.symbols.len() != streams.len() {
            return Err(DispatchError::InvalidPlan);
        }
        if let Err(error) = check_collectives(programs, collectives) {
            self.log(&error);
            return Err(DispatchError::InvalidPlan);
        }
        let segments = programs
            .iter()
            .zip(collectives)
            .map(|(program, waits)| collective_segments(program, waits))
            .collect::<Vec<_>>();
        let _unwinding = PoisonOnUnwind(&self.collectives);
        let mut started = vec![false; streams.len()];
        launch_segments(
            &self.collectives,
            &mut Unit::default(),
            &mut started,
            &segments,
            |rank, launches| {
                pipeline::launch_all(&self.services, streams[rank], &self.symbols[rank], launches)
            },
            |_| {},
            |_| Ok(()),
        )
    }

    /// Every logits row of the step as device views in the original item
    /// order: a prefill's rows on rank 0, decode and verify rows on the
    /// rank that owns their block, each emitted once its owner completed.
    fn emit_logits(
        &self,
        runs: &[Run],
        sink: &mut ExecuteResultSinkV1,
    ) -> Result<(), BundleAbiError> {
        let empty = PhaseInputs::default();
        let prefill = runs.iter().find(|run| run.arena == PREFILL);
        let decode = runs.iter().find(|run| run.arena != PREFILL);
        let owners = self
            .ranks
            .iter()
            .zip(&self.buffers)
            .map(|(&(context, _, _), buffers)| (context, buffers.scratch.decode_logits))
            .collect::<Vec<_>>();
        let rank0 = &self.buffers[0].scratch;
        let views = logits::ordered_views_owned(
            super::model_limits(&self.config),
            self.ranks[0].0,
            logits::PhaseOutput {
                inputs: prefill.map_or(&empty, |run| &run.inputs),
                buffer: rank0.prefill_logits,
                element_type: BUNDLE_LOGITS_ELEMENT_V1_F32,
                compact: true,
            },
            logits::PhaseOutput {
                inputs: decode.map_or(&empty, |run| &run.inputs),
                buffer: rank0.decode_logits,
                element_type: BUNDLE_LOGITS_ELEMENT_V1_F32,
                compact: false,
            },
            decode.map(|run| logits::RowOwners {
                block: run.head_rows.div_ceil(owners.len() as u32),
                ranks: &owners,
            }),
        )?;
        // The host reads a view as soon as it arrives: every rank that owns
        // one completes its queued work first.
        let owners = views
            .iter()
            .map(|view| view.rank)
            .collect::<std::collections::BTreeSet<_>>();
        for rank in owners {
            let &(_, _, stream) = self
                .ranks
                .get(rank as usize)
                .ok_or(BundleAbiError::InvalidRange("deepseek41 logits owner"))?;
            self.services.synchronize_stream(stream)?;
        }
        for view in views {
            sink.emit_device_view(view)?;
        }
        Ok(())
    }

    pub(super) fn execute_step(
        &mut self,
        request: &[u8],
        sink: &mut ExecuteResultSinkV1,
    ) -> BundleStatusV1 {
        if self.ledger.step().is_some() {
            return BundleStatusV1::InvalidState;
        }
        let step = match self.pack(request) {
            Ok(step) => step,
            Err(error) => return error,
        };
        let mut items = items(&step.prefill);
        items.extend(self::items(&step.decode));
        let mut runs = Vec::new();
        for inputs in [step.prefill, step.decode] {
            if inputs.request_ids.is_empty() {
                continue;
            }
            match self.plan_run(inputs) {
                Ok(run) => runs.push(run),
                Err(error) => return error,
            }
        }
        if self.ledger.preview(&items).is_err() || self.ledger.begin(&items, Vec::new()).is_err() {
            return BundleStatusV1::InvalidArgument;
        }
        let issued = self.issue_step(&mut runs, sink);
        if let Err(error) = issued {
            self.log(&format!("execute failed: {error}"));
            // Work queued behind an unmatched collective never runs; a
            // poisoned instance leaves its streams and slots alone.
            if self.collectives.is_poisoned() {
                self.log(
                    "execute left collectives unmatched across ranks; the instance is poisoned",
                );
                return status(error);
            }
            let _ = self.ledger.cancel();
            return status(error);
        }
        BundleStatusV1::Success
    }

    /// Issues the step: the prefill, then the decode or verify phase, then
    /// the logits.
    fn issue_step(
        &mut self,
        runs: &mut [Run],
        sink: &mut ExecuteResultSinkV1,
    ) -> Result<(), BundleAbiError> {
        for run in runs.iter_mut() {
            self.upload(run)?;
            self.launch(run)?;
        }
        self.emit_logits(runs, sink)
    }
}

/// The stage enum is the program's; nothing here names a stage directly.
#[allow(dead_code)]
const _: Option<Stage> = None;
