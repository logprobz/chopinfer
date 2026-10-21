//! One step at context-parallel size 1: pack the batch, open the ledger's
//! step, and for its eager prefill, then its decode or verify phase, upload
//! the phase inputs and the KDA side inputs and issue each rank's lowered
//! program. An eager prefill runs its indexer stages in pieces of at most
//! 2048 rows (`crate::step::pieces`), each with its own inputs in the
//! rank's piece arena. Commit copies each verify item's last kept
//! checkpoint into its slot; cancel leaves the slots the step advanced
//! invalid, and the ledger retires them.
//!
//! The program is complete only once every stage lowers: until then
//! execute refuses with `Unsupported` before the step opens, naming the
//! first unlowered stage. The fake-host tests run the lowered stages alone
//! (`Instance::partial`) to drive the step through the ledger.

use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::drafter::{DraftError, Pending};
use bundle_runtime::graphs::{GraphCache, GraphKey};
use bundle_runtime::inputs::{PackedStep, PhaseInputs};
use bundle_runtime::logits;
use bundle_runtime::pipeline::{
    PoisonOnUnwind, PreparedLaunch, Program, Unit, check_collectives, collective_segments,
    launch_segments,
};
use bundle_runtime::resources::{status, upload_bytes};
use bundle_runtime::upload::{DeviceInputs, InputLayout, PhaseArena};
use kernel_core::backend::{Counts, Need, Refused};
use model_glm5next_contract::contract::{
    ANCHOR_ITEMS, Glm, MAX_VERIFY_ROWS, PREFILL_GRANULE, latent_format,
};
use model_glm5next_contract::own_ops::OwnOps;
use model_glm5next_contract::own_ops::state::{StateCopy, StateCopyBuffers};
use runtime_bundle_abi::{
    BUNDLE_LOGITS_ELEMENT_V1_F32, BundleAbiError, BundleStatusV1, DeviceBufferSpanV1,
    ExecuteResultSinkV1,
};
use runtime_bundle_sdk::BatchCodecV1;

use super::prefill_graphs::{Key as PrefillKey, Room};
use super::{Instance, NAME};
use crate::buffers::{Buffer, kda_checkpoint_stride};
use crate::drafter::Plan as DraftPlan;
use crate::lower::{Lowering, Piece};
use crate::program::{DraftPass, Phase, Stage};
use crate::step::PIECE_ROWS;
use crate::submit::launch_all;

mod dma;

/// Phase-arena indexes (`Instance::arenas`).
const PREFILL: usize = 0;
const DECODE: usize = 1;
const VERIFY: usize = 2;

pub(super) use crate::step::MAX_PIECES;

/// The input layout of one indexer piece.
pub(super) fn piece_layout(items: bundle_runtime::geometry::Items) -> InputLayout {
    InputLayout::within(PIECE_ROWS, 1, items)
}

/// Bytes of one piece's region in a rank's piece arena.
pub(super) fn piece_stride(items: bundle_runtime::geometry::Items) -> u64 {
    piece_layout(items).end.next_multiple_of(256)
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// One phase of a step, planned before the step opens.
struct Run {
    inputs: PhaseInputs,
    arena: usize,
    layout: InputLayout,
    /// The ledger index of the phase's first item.
    first: usize,
    /// Rows the head scores.
    head_rows: u32,
    /// The phase the programs lower.
    phase: Phase,
    /// A decode or verify phase's graph bucket: (tokens, items), items 0
    /// for decode. An eager prefill has none.
    graph: Option<(u32, u32)>,
    /// An eager prefill's key among the captured prefill programs
    /// (`super::prefill_graphs`); none for a prefill that stays eager.
    prefill: Option<PrefillKey>,
    /// An eager prefill's indexer pieces and their first rows.
    pieces: Vec<(u32, PhaseInputs)>,
    /// Each rank's launches, each launch's stage and whether it waits for
    /// its peers: empty for a rank whose graph of the bucket was captured
    /// when the step was planned (`lowered` false), which replays it
    /// instead.
    programs: Vec<Vec<PreparedLaunch>>,
    stages: Vec<Vec<Stage>>,
    collectives: Vec<Vec<bool>>,
    /// Whether each rank's program was lowered.
    lowered: Vec<bool>,
    /// A drafter run's pass; `None` for the target's phases.
    draft: Option<DraftPass>,
    inject: bool,
    /// A drafter run's own inputs: its rows' source codes, its draft rows
    /// and their count, and a draft step's shared-selection rows.
    draft_side: Vec<(Buffer, Vec<i32>)>,
}

/// A drafter run's own inputs (`Run::draft_side`): each row's (hidden,
/// token) source codes; the draft rows' run rows, drafting items and
/// verify rows, `DRAFT_ITEMS` each; their count; and in a draft step, each
/// item's row of the extend pass (`first`), whose selection it reuses.
fn draft_side(
    run: &crate::drafter::Run,
    first: Option<&crate::drafter::Run>,
) -> Result<Vec<(Buffer, Vec<i32>)>, &'static str> {
    let capacity = model_glm5next_contract::contract::DRAFT_ITEMS as usize;
    if run.drafts.len() > capacity {
        return Err("more drafts than the drafter's rows");
    }
    let (hidden, tokens) = run.codes();
    let codes = hidden
        .iter()
        .zip(&tokens)
        .flat_map(|(&hidden, &token)| [hidden, token])
        .collect();
    let mut rows = vec![0; 3 * capacity];
    for (index, draft) in run.drafts.iter().enumerate() {
        rows[index] = draft.row as i32;
        rows[capacity + index] = draft.item as i32;
        rows[2 * capacity + index] = draft.target;
    }
    let mut side = vec![
        (Buffer::DraftCodes, codes),
        (Buffer::DraftRows, rows),
        (Buffer::DraftCount, vec![run.drafts.len() as i32]),
    ];
    if let Some(first) = first.filter(|first| !std::ptr::eq(*first, run)) {
        let shared = run
            .drafts
            .iter()
            .map(|draft| {
                first
                    .drafts
                    .iter()
                    .find(|extend| extend.item == draft.item)
                    .map(|extend| extend.row as i32)
                    .ok_or("a draft step's item has no extend row")
            })
            .collect::<Result<_, _>>()?;
        side.push((Buffer::DraftShareRows, shared));
    }
    Ok(side)
}

/// A phase, its arena index, its input layout and its graph key.
type PhasePlan = (Phase, usize, InputLayout, Option<(u32, u32)>);

/// Each rank's launches, each launch's stage and whether it waits for its
/// peers.
type Programs = (Vec<Vec<PreparedLaunch>>, Vec<Vec<Stage>>, Vec<Vec<bool>>);

/// A step's drafter runs (`crate::drafter::Plan`): the extend pass and the
/// draft steps before the target, the extend pass over a prefill's
/// completed rows after it.
#[derive(Default)]
struct DraftRuns {
    before: Vec<Run>,
    prefill: Option<Run>,
}

impl Instance {
    pub(super) fn log(&self, message: &str) {
        let _ = self.services.log(format!("{NAME}: {message}").as_bytes());
    }

    /// Packs a step's prefill and decode or verify items on the host's
    /// state slots.
    /// The step, and per decode-phase item the drafts it asks the bundle
    /// for (zero unless the MTP drafter drafts it).
    fn pack(&self, request: &[u8]) -> Result<(PackedStep, Vec<u32>), BundleStatusV1> {
        let batch = BatchCodecV1::decode(request).map_err(|_| BundleStatusV1::InvalidArgument)?;
        if !batch.media().is_empty() && self.encoder.is_none() {
            return Err(BundleStatusV1::Unsupported);
        }
        // A context-parallel group's tables list one local page per stripe.
        let step = PackedStep::pack_on_host_slots::<Glm>(
            self.config.shape.model_limits(),
            &batch,
            self.kv_pages,
            self.verify,
            self.items(),
            u64::from(self.config.context_parallel.size()),
        )?;
        let drafts = step.bundle_drafts(&batch);
        Ok((step, drafts))
    }

    /// The phase of `inputs`, its arena index, its input layout and its
    /// graph key: (tokens, items) of a verify bucket, (bucket, 0) of a
    /// decode bucket, none for an eager phase.
    fn phase_of(&self, inputs: &PhaseInputs) -> Result<PhasePlan, BundleStatusV1> {
        let items = self.items();
        let live = inputs.live_batch();
        if !inputs.decode {
            let layout =
                InputLayout::prefill_within::<Glm>(inputs, inputs.logit_rows().len(), items);
            let phase = Phase::Prefill {
                rows: inputs.live_tokens(),
            };
            Ok((phase, PREFILL, layout, None))
        } else if inputs.verify {
            let rows =
                bundle_runtime::graphs::verify_bucket_within::<Glm>(inputs.live_tokens(), items)
                    .ok_or(BundleStatusV1::InvalidArgument)?;
            // A graph's item bucket holds the live items and, at most
            // `MAX_VERIFY_ROWS` rows each, the token bucket. An instance
            // too narrow for the padded tokens (one served slot and a
            // six-row item padded to eight) runs the phase eagerly at its
            // live rows and items instead.
            let held = rows.div_ceil(MAX_VERIFY_ROWS).next_power_of_two();
            let live_bucket = items.bucket(live).ok_or(BundleStatusV1::InvalidArgument)?;
            let layout = InputLayout::verify_within::<Glm>(items);
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

    /// Each rank's indexer pieces of a prefill: the piece's region of the
    /// rank's piece arena, and the block-table rows from its first item on.
    fn rank_pieces(
        &self,
        draft: bool,
        run_inputs: &PhaseInputs,
        pieces: &[(u32, PhaseInputs)],
        rank: usize,
    ) -> Vec<Piece> {
        let items = self.items();
        let layout = piece_layout(items);
        let stride = piece_stride(items);
        let row_bytes = 4 * u64::from(self.config.shape.model_limits().max_pages());
        let tables = self.phase_arenas(draft)[PREFILL][rank].block_tables;
        let arena = self.piece_arenas[rank];
        pieces
            .iter()
            .enumerate()
            .map(|(index, (first, piece))| {
                let item = run_inputs
                    .item_order
                    .iter()
                    .position(|&order| order == piece.item_order[0])
                    .unwrap_or(0) as u64;
                let region = DeviceBufferSpanV1::new(
                    arena.buffer,
                    arena.offset + index as u64 * stride,
                    layout.end,
                );
                let block_tables = DeviceBufferSpanV1::new(
                    tables.buffer,
                    tables.offset + item * row_bytes,
                    tables.length - item * row_bytes,
                );
                Piece {
                    first: *first,
                    rows: piece.live_tokens(),
                    inputs: DeviceInputs::within(
                        PhaseArena {
                            arena: region,
                            block_tables,
                        },
                        &layout,
                    ),
                }
            })
            .collect()
    }

    /// Each rank's launches of `phase`: every lowered stage, in program
    /// order. An unlowered stage refuses the step unless the instance runs
    /// the lowered stages alone. With `inject`, an eager prefill overrides
    /// its image rows of the embedding by the encoder's arena rows, once
    /// the all-reduce assembled it and before the streams expand it. Only
    /// the ranks `lower` marks are lowered; the others get empty programs.
    #[allow(clippy::too_many_arguments)]
    fn program(
        &self,
        phase: Phase,
        arena: usize,
        layout: &InputLayout,
        run_inputs: &PhaseInputs,
        pieces: &[(u32, PhaseInputs)],
        head_rows: u32,
        draft: Option<DraftPass>,
        inject: bool,
        lower: &[bool],
    ) -> Result<Programs, BundleStatusV1> {
        let media = match (inject, &self.encoder) {
            (false, _) => None,
            (true, Some(encoder)) if arena == PREFILL && draft.is_none() => Some(encoder),
            (true, _) => {
                self.log("only the target's eager prefill injects encoder rows");
                return Err(BundleStatusV1::InternalError);
            }
        };
        if !lower.contains(&true) {
            let ranks = self.buffers.len();
            return Ok((
                (0..ranks).map(|_| Vec::new()).collect(),
                vec![Vec::new(); ranks],
                vec![Vec::new(); ranks],
            ));
        }
        let shape = self.config.shape;
        let kv = latent_format(&self.config.format);
        let checkpoint = self.config.layout;
        let records = match draft {
            Some(pass) => crate::launch::drafter_records_for(shape, checkpoint, kv, phase, pass),
            None => crate::launch::records_for(shape, checkpoint, kv, phase),
        }
        .map_err(|error| {
            self.log(&error);
            BundleStatusV1::InternalError
        })?;
        let peers = self
            .buffers
            .iter()
            .map(|buffers| buffers.comm.clone())
            .collect::<Vec<_>>();
        let mut programs = Vec::new();
        let mut stages = Vec::new();
        let mut collectives = Vec::new();
        for (rank, buffers) in self.buffers.iter().enumerate() {
            if !lower.get(rank).copied().unwrap_or(true) {
                programs.push(Vec::new());
                stages.push(Vec::new());
                collectives.push(Vec::new());
                continue;
            }
            hooks::lowered();
            let inputs =
                DeviceInputs::within(self.phase_arenas(draft.is_some())[arena][rank], layout);
            let pieces = if arena == PREFILL && pieces.len() > 1 {
                self.rank_pieces(draft.is_some(), run_inputs, pieces, rank)
            } else {
                Vec::new()
            };
            let lowering = Lowering {
                shape,
                format: &self.config.format,
                buffers,
                inputs: &inputs,
                ops: &self.ops,
                rank: rank as u32,
                index_topk: self.config.runtime.index_topk,
                pieces: &pieces,
                row_offset: 0,
                peers: &peers,
                head_rows,
                context_parallel: self.context_parallel.as_ref(),
                draft,
                verify_tokens: draft.map(|_| {
                    let layout = InputLayout::verify_within::<Glm>(self.items());
                    DeviceInputs::within(self.arenas[VERIFY][rank], &layout).token_ids
                }),
                projections: self
                    .projections
                    .as_ref()
                    .map(|projections| projections.rank(rank)),
            };
            let mut launches = Vec::new();
            let mut launch_stages = Vec::new();
            let mut waits = Vec::new();
            // One stage's launches in `phase` through `lowering`, each with
            // the facts its op states; none for an unlowered stage of an
            // instance that runs the lowered stages alone.
            let lower = |lowering: &Lowering<'_>, phase: Phase, stage: Stage| match lowering
                .lower_facts(phase, stage, self.config.comm_epilogue)
            {
                Ok(Some(lowered)) => Ok(lowered),
                Ok(None) if self.partial => Ok(Vec::new()),
                Ok(None) => {
                    let family = records
                        .iter()
                        .find(|record| record.stage == stage)
                        .map_or("?", |record| record.op.family());
                    self.log(&format!("stage {stage:?} ({family}) is not lowered yet"));
                    Err(BundleStatusV1::Unsupported)
                }
                Err(error) => {
                    self.log(&error);
                    Err(BundleStatusV1::InternalError)
                }
            };
            // `stage`'s `lowered` launches appended, with whether each waits
            // for its peers.
            let append = |launches: &mut Vec<PreparedLaunch>,
                          launch_stages: &mut Vec<Stage>,
                          waits: &mut Vec<bool>,
                          stage: Stage,
                          lowered: Vec<crate::ops::Lowered>| {
                launch_stages.extend(std::iter::repeat_n(stage, lowered.len()));
                waits.extend(lowered.iter().map(|lowered| lowered.collective));
                launches.extend(crate::ops::launches(lowered));
            };
            for record in &records {
                let lowered = lower(&lowering, phase, record.stage)?;
                append(
                    &mut launches,
                    &mut launch_stages,
                    &mut waits,
                    record.stage,
                    lowered,
                );
                if let (Some(encoder), Stage::EmbeddingReduce) = (media, record.stage) {
                    let inject = encoder.ranks[rank].inject_part(
                        buffers.get(Buffer::Reduced),
                        inputs.live_tokens,
                        0,
                        phase.rows(),
                    );
                    let inject = inject.map_err(|refusal| {
                        self.log(&format!("encoder inject: {}", refusal.0));
                        BundleStatusV1::InternalError
                    })?;
                    append(
                        &mut launches,
                        &mut launch_stages,
                        &mut waits,
                        record.stage,
                        vec![crate::ops::Lowered::local(inject)],
                    );
                }
            }
            self.check_phased_calls(&waits)?;
            programs.push(launches);
            stages.push(launch_stages);
            collectives.push(waits);
        }
        Ok((programs, stages, collectives))
    }

    /// Refuses a rank's program of more collective calls (`waits`, one per
    /// launch) than `Target::phased_calls_per_run`, which bounds the
    /// phased all-reduce calls `quiesce_phased` counts on: every phased
    /// call is a collective.
    fn check_phased_calls(&self, waits: &[bool]) -> Result<(), BundleStatusV1> {
        let bound = self.target.phased_calls_per_run();
        let calls = waits.iter().filter(|&&waits| waits).count() as u64;
        if calls > bound {
            self.log(&format!(
                "a program of {calls} collective calls passes the {bound} phased all-reduce calls \
                 its state's reset counts on"
            ));
            return Err(BundleStatusV1::InternalError);
        }
        Ok(())
    }

    /// Plans one phase of the step: its pieces and every rank's program.
    fn plan_run(
        &self,
        inputs: PhaseInputs,
        first: usize,
        inject: bool,
    ) -> Result<Run, BundleStatusV1> {
        let (phase, arena, layout, graph) = self.phase_of(&inputs)?;
        let pieces = if arena == PREFILL {
            crate::step::pieces(&inputs, PREFILL_GRANULE).map_err(|error| {
                self.log(&error);
                BundleStatusV1::InvalidArgument
            })?
        } else {
            Vec::new()
        };
        let head_rows = if arena == PREFILL {
            inputs.logit_rows().len() as u32
        } else {
            phase.rows()
        };
        let capacity = if arena == PREFILL {
            crate::buffers::SCORED_ROWS
        } else {
            crate::buffers::head_rows()
        };
        if head_rows > capacity {
            self.log(&format!(
                "{head_rows} scored rows exceed the head's {capacity} rows"
            ));
            return Err(BundleStatusV1::Unsupported);
        }
        let prefill = self.prefill_key(arena, &inputs, &layout, &pieces, head_rows, None, inject);
        let lowered = self.missing_ranks(graph, None, prefill.as_ref());
        let (programs, stages, collectives) = self.program(
            phase, arena, &layout, &inputs, &pieces, head_rows, None, inject, &lowered,
        )?;
        Ok(Run {
            inputs,
            arena,
            layout,
            first,
            head_rows,
            phase,
            graph,
            prefill,
            pieces,
            programs,
            stages,
            collectives,
            lowered,
            draft: None,
            inject,
            draft_side: Vec::new(),
        })
    }

    /// The graph cache a run of `draft` replays from: the target's, the
    /// drafter's extend pass's or its draft steps'.
    fn graph_cache(&self, draft: Option<DraftPass>) -> &GraphCache<Glm> {
        match draft {
            None => &self.graphs,
            Some(DraftPass::Extend) => &self.draft_graphs[0],
            Some(DraftPass::Step) => &self.draft_graphs[1],
        }
    }

    /// Which ranks need their program lowered: every rank of an eager run
    /// whose prefill programs are not captured, and of a graph run the
    /// ranks whose graph of the bucket is not captured yet. A captured rank
    /// replays its graph, so its program would only be dropped
    /// (`pipeline::run_ranks` asks only for missing ranks' launches).
    fn missing_ranks(
        &self,
        graph: Option<(u32, u32)>,
        draft: Option<DraftPass>,
        prefill: Option<&PrefillKey>,
    ) -> Vec<bool> {
        let ranks = self.buffers.len();
        let Some((bucket, items)) = graph else {
            let captured = prefill.is_some_and(|key| self.prefill_graphs.contains(key));
            return vec![!captured; ranks];
        };
        let cache = self.graph_cache(draft);
        (0..ranks)
            .map(|rank| {
                !cache.contains(GraphKey {
                    rank: rank as u32,
                    bucket,
                    items,
                })
            })
            .collect()
    }

    /// Lowers the programs of `run`'s ranks that miss their graph now but
    /// were not lowered when the step was planned (their graph was
    /// released since, or a captured prefill program was evicted by an
    /// earlier run's capture), so every rank `run_ranks` captures, or an
    /// eager prefill launches, has its launches.
    fn lower_missing(&self, run: &mut Run) -> Result<(), BundleAbiError> {
        let lower = self
            .missing_ranks(run.graph, run.draft, run.prefill.as_ref())
            .iter()
            .zip(&run.lowered)
            .map(|(&missing, &lowered)| missing && !lowered)
            .collect::<Vec<_>>();
        if !lower.contains(&true) {
            return Ok(());
        }
        // Only an eager prefill injects encoder rows.
        let (programs, stages, collectives) = self
            .program(
                run.phase,
                run.arena,
                &run.layout,
                &run.inputs,
                &run.pieces,
                run.head_rows,
                run.draft,
                false,
                &lower,
            )
            .map_err(BundleAbiError::Status)?;
        for (rank, ((program, launch_stages), waits)) in programs
            .into_iter()
            .zip(stages)
            .zip(collectives)
            .enumerate()
        {
            if lower[rank] {
                run.programs[rank] = program;
                run.stages[rank] = launch_stages;
                run.collectives[rank] = waits;
                run.lowered[rank] = true;
            }
        }
        Ok(())
    }

    /// The target's input arenas, or the drafter's.
    fn phase_arenas(&self, draft: bool) -> &[Vec<PhaseArena>; 3] {
        if draft {
            &self.draft_arenas
        } else {
            &self.arenas
        }
    }

    /// Plans one drafter run of `pass`: a draft step at its item bucket; an
    /// extend pass at its verify bucket when its rows fit one, else eagerly
    /// as a prefill (a chunk's completed rows, or many catch-up rows). The
    /// drafter's argmax reads its own logits shards, so its head scores no
    /// rows for the sink.
    fn plan_draft_run(
        &self,
        run: &crate::drafter::Run,
        pass: DraftPass,
        first: Option<&crate::drafter::Run>,
    ) -> Result<Run, BundleStatusV1> {
        let mut inputs = run.inputs();
        if pass == DraftPass::Extend {
            let fits = inputs.live_tokens() <= crate::buffers::head_rows()
                && inputs
                    .q_len
                    .iter()
                    .all(|&rows| rows as u32 <= MAX_VERIFY_ROWS);
            if fits && inputs.live_tokens() > inputs.live_batch() {
                inputs.verify = true;
                inputs.verify_rows.iter_mut().for_each(|row| *row = true);
            } else if !fits || inputs.live_tokens() > inputs.live_batch() {
                inputs.decode = false;
            }
        }
        let (phase, arena, layout, graph) = self.phase_of(&inputs)?;
        // Its head and argmax score the phase's draft rows.
        if run.drafts.len() > crate::program::draft_rows(phase) as usize {
            self.log(&format!(
                "{} drafts exceed the {:?} drafter run's draft rows",
                run.drafts.len(),
                phase
            ));
            return Err(BundleStatusV1::InternalError);
        }
        let pieces = if arena == PREFILL {
            crate::step::pieces(&inputs, PREFILL_GRANULE).map_err(|error| {
                self.log(&error);
                BundleStatusV1::InvalidArgument
            })?
        } else {
            Vec::new()
        };
        let prefill = self.prefill_key(arena, &inputs, &layout, &pieces, 0, Some(pass), false);
        let lowered = self.missing_ranks(graph, Some(pass), prefill.as_ref());
        let (programs, stages, collectives) = self.program(
            phase,
            arena,
            &layout,
            &inputs,
            &pieces,
            0,
            Some(pass),
            false,
            &lowered,
        )?;
        Ok(Run {
            inputs,
            arena,
            layout,
            first: 0,
            head_rows: 0,
            phase,
            graph,
            prefill,
            pieces,
            programs,
            stages,
            collectives,
            lowered,
            draft: Some(pass),
            inject: false,
            draft_side: draft_side(run, first).map_err(|error| {
                self.log(error);
                BundleStatusV1::InternalError
            })?,
        })
    }

    /// Every drafter run of `plan`.
    fn plan_draft_runs(&self, plan: &DraftPlan) -> Result<DraftRuns, BundleStatusV1> {
        let mut runs = DraftRuns::default();
        for (index, run) in plan.before.iter().enumerate() {
            let pass = if index == 0 {
                DraftPass::Extend
            } else {
                DraftPass::Step
            };
            runs.before
                .push(self.plan_draft_run(run, pass, plan.before.first())?);
        }
        if let Some(run) = &plan.prefill {
            runs.prefill = Some(self.plan_draft_run(run, DraftPass::Extend, None)?);
        }
        Ok(runs)
    }

    /// The KDA entry of each anchor item's spare in `inputs`, by request:
    /// anchor item `k` (`crate::step::anchors`) takes the `k`th spare after
    /// the KDA buffers' own entries and checkpoints, the same in both.
    fn anchor_spares(&self, inputs: &PhaseInputs) -> Result<Vec<(u64, i32)>, String> {
        let state = &self
            .buffers
            .first()
            .ok_or("glm5next holds no rank buffers")?
            .state;
        crate::step::anchors(inputs, PREFILL_GRANULE)?
            .into_iter()
            .enumerate()
            .map(|(anchor, (request, _))| {
                let anchor = anchor as u32;
                let entry = state.anchor_entry("kda.state", anchor)?;
                if state.anchor_entry("kda.conv", anchor)? != entry {
                    return Err("the KDA buffers' anchor spares differ".to_owned());
                }
                let entry = i32::try_from(entry).map_err(|_| "a KDA spare entry overflows")?;
                Ok((request, entry))
            })
            .collect()
    }

    /// The KDA side inputs of `run`, as (buffer, values) uploads.
    fn side(&self, run: &Run) -> Result<Vec<(Buffer, Vec<i32>)>, BundleAbiError> {
        if run.arena == PREFILL {
            let invalid = |error: String| {
                self.log(&error);
                BundleAbiError::InvalidRange("glm5next prefill inputs")
            };
            let spares = self.anchor_spares(&run.inputs).map_err(invalid)?;
            let inputs = crate::step::prefill_inputs(&run.inputs, &spares).map_err(invalid)?;
            return Ok(vec![
                (Buffer::CuSeqlens, inputs.cu_seqlens.clone()),
                (Buffer::CuRows, inputs.cu_seqlens),
                (Buffer::Initial, inputs.initial),
                (Buffer::AnchorIds, inputs.anchor_ids),
            ]);
        }
        let shape = self.config.shape;
        let rows = (0..run.inputs.request_ids.len())
            .map(|item| run.inputs.q_len.get(item).copied().unwrap_or(1) as u32)
            .collect::<Vec<_>>();
        let step = self
            .ledger
            .step()
            .ok_or(BundleAbiError::InvalidRange("no open glm5next step"))?;
        // The indexer tail checkpoints the most verify rows of an item.
        let tail_stride = MAX_VERIFY_ROWS;
        let kda_stride = kda_checkpoint_stride(shape)
            .map_err(|_| BundleAbiError::InvalidRange("glm5next KDA checkpoint stride"))?;
        let state = &self.buffers[0].state;
        let side = crate::step::side_inputs(
            &state.state,
            state.extent,
            step,
            run.first,
            &rows,
            kda_stride,
            tail_stride,
        )
        .map_err(|error| {
            self.log(&error);
            BundleAbiError::InvalidRange("glm5next side inputs")
        })?;
        Ok(vec![
            (Buffer::CuSeqlens, side.cu_rows.clone()),
            (Buffer::CuRows, side.cu_rows),
            (Buffer::CheckpointIds, side.checkpoint_ids),
            (Buffer::TailCheckpointIds, side.tail_checkpoint_ids),
        ])
    }

    /// Uploads `run`'s inputs, block tables, pieces and KDA side inputs to
    /// every rank.
    /// Uploads a target phase's side tables again on every rank (`side`):
    /// an eager prefill issued after the phase's inputs went up rewrote
    /// the scratch tables both phases bind.
    fn upload_side(&self, run: &Run) -> Result<(), BundleAbiError> {
        let side = self.side(run)?;
        for (rank, &(_, _, stream)) in self.ranks.iter().enumerate() {
            let buffers = &self.buffers[rank];
            for (buffer, values) in &side {
                upload_bytes(
                    &self.services,
                    stream,
                    buffers.get(*buffer),
                    &i32_bytes(values),
                )?;
            }
        }
        Ok(())
    }

    fn upload(&mut self, run: &Run) -> Result<(), BundleAbiError> {
        // The drafter's layer has no KDA, and writes its indexer tail in
        // place (`Lowering::index_key_write`).
        let side = if run.draft.is_some() {
            run.draft_side.clone()
        } else {
            self.side(run)?
        };
        let draft = run.draft.is_some();
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
                "glm5next inputs exceed their arena",
            ))?;
        let items = self.items();
        let pieces = if run.pieces.len() > 1 {
            run.pieces
                .iter()
                .map(|(_, piece)| piece_layout(items).pack(piece, &[]))
                .collect::<Option<Vec<_>>>()
                .ok_or(BundleAbiError::InvalidRange(
                    "glm5next piece inputs exceed their arena",
                ))?
        } else {
            Vec::new()
        };
        let mirror = if draft {
            &mut self.draft_mirrors[run.arena]
        } else {
            &mut self.mirrors[run.arena]
        };
        let writes = mirror.update(&run.inputs.block_tables);
        let mirror = if draft {
            &self.draft_mirrors[run.arena]
        } else {
            &self.mirrors[run.arena]
        };
        let uploaded = (|| {
            for (rank, &(_, _, stream)) in self.ranks.iter().enumerate() {
                let phase = self.phase_arenas(draft)[run.arena][rank];
                upload_bytes(&self.services, stream, phase.arena, &bytes)?;
                mirror.upload(&self.services, stream, phase.block_tables, &writes)?;
                let arena = self.piece_arenas[rank];
                let stride = piece_stride(items);
                for (index, bytes) in pieces.iter().enumerate() {
                    let region = DeviceBufferSpanV1::new(
                        arena.buffer,
                        arena.offset + index as u64 * stride,
                        stride,
                    );
                    upload_bytes(&self.services, stream, region, bytes)?;
                }
                let buffers = &self.buffers[rank];
                for (buffer, values) in &side {
                    upload_bytes(
                        &self.services,
                        stream,
                        buffers.get(*buffer),
                        &i32_bytes(values),
                    )?;
                }
            }
            Ok(())
        })();
        if uploaded.is_err() {
            if draft {
                self.draft_mirrors[run.arena].forget();
            } else {
                self.mirrors[run.arena].forget();
            }
        }
        uploaded
    }

    /// Uploads and issues `run` on every rank as one unit of collective
    /// work: eagerly, every rank up to each collective in turn
    /// (`launch_eager`), or by replaying a decode or verify phase's graphs
    /// (`bundle_runtime::pipeline::run_ranks`), each rank's program
    /// captured at its first step of the bucket before any rank launches.
    /// Every launch reads its inputs, and every row, copy, attention, KDA,
    /// indexer and MoE launch its live counts, from the phase arena the step
    /// uploaded, so one capture serves every step of the bucket. The dense
    /// GEMMs (the projections, the pooling gate, the head) and the F32
    /// all-reduce are the exception: the shared GEMM ABI takes the bucket's
    /// row count as a scalar fixed at capture, so they compute and write
    /// every row of the bucket, the inactive ones from stale inputs, and the
    /// all-reduce sums every bucket row. Their rows are independent (the
    /// all-reduce sums element by element), and a dense GEMM may feed
    /// another whole bucket (the KDA gates' down then up projections, a
    /// dense or shared gate-up then its down), but each such row-local chain
    /// ends at a consumer that reads only live rows by the device count (a
    /// copy, a row kernel, an engine, or the rounding after the all-reduce),
    /// or at the logits exchange, which moves the bucket's rows and whose
    /// inactive rows the sink never reads; so an inactive row never reaches
    /// a live one (`glm5next_bundle_kernels.rs`,
    /// `bucket_rows_stay_out_of_live_rows`). A
    /// failure or panic once a rank's share was queued poisons the instance:
    /// its peers would wait forever at the collectives that rank queued.
    fn issue(&mut self, run: &mut Run) -> Result<(), BundleAbiError> {
        self.upload(run)?;
        self.launch(run)
    }

    /// Issues `run`'s uploaded inputs' programs on every rank (`issue`).
    /// A decode, verify or drafter graph run while a release of every
    /// graph is owed (`PrefillGraphs::owes`: a full release failed, so the
    /// ledger holds graphs no cache names, and every lookup is gone, so the
    /// run would capture) first pays it, as a prefill capture does
    /// (`Room::Every`), so nothing captures above those graphs; a failed
    /// release fails the run before it captures anything. Its ranks then
    /// lower what the release took (`lower_missing`).
    fn launch(&mut self, run: &mut Run) -> Result<(), BundleAbiError> {
        let streams = self
            .ranks
            .iter()
            .map(|&(_, _, stream)| stream)
            .collect::<Vec<_>>();
        if run.graph.is_some() && self.prefill_graphs.owes() {
            self.log("a release of every graph is owed; releasing every graph before the run");
            self.invalidate_graphs(&streams)
                .map_err(|error| self.dispatch_failed(error))?;
        }
        if run.graph.is_some() || run.prefill.is_some() {
            self.lower_missing(run)?;
        }
        let mut programs = std::mem::take(&mut run.programs);
        let stages = std::mem::take(&mut run.stages);
        let mut collectives = std::mem::take(&mut run.collectives);
        self.quiesce_phased(&streams)?;
        if dma::eligible(self, run) {
            let planned =
                dma::Plan::prepare(self, run.phase.rows(), &programs, &stages, &collectives)
                    .map_err(|error| {
                        self.log(&error);
                        BundleAbiError::InvalidRange("glm5next prefill DMA plan")
                    })?;
            return planned
                .execute(self, &streams)
                .map_err(|error| self.dispatch_failed(error));
        }
        if let Some(key) = &run.prefill {
            return self
                .launch_prefill(&streams, key, &programs, &stages, &collectives)
                .map_err(|error| self.dispatch_failed(error));
        }
        #[cfg(feature = "diagnostic-taps")]
        if run.graph.is_none() && run.draft.is_none() && self.taps.text() {
            return self.launch_tapped(&streams, run, &programs, &stages, &collectives);
        }
        if run.graph.is_none() {
            return self
                .launch_eager(&streams, &programs, &stages, &collectives)
                .map_err(|error| self.dispatch_failed(error));
        }
        let replayed = match run.graph {
            Some((bucket, items)) if self.shared_overlap.serves(bucket) => self
                .run_overlapped(run.draft, (bucket, items), &programs, &stages)
                .map_err(|error| self.dispatch_failed(error)),
            _ => crate::submit::run_ranks(
                &self.services,
                &streams,
                &self.symbols,
                run.graph,
                match run.draft {
                    None => &mut self.graphs,
                    Some(DraftPass::Extend) => &mut self.draft_graphs[0],
                    Some(DraftPass::Step) => &mut self.draft_graphs[1],
                },
                &self.collectives,
                |rank| {
                    Ok(Program {
                        launches: std::mem::take(&mut programs[rank]),
                        collective: std::mem::take(&mut collectives[rank]),
                    })
                },
                hooks::submitted,
            )
            .map_err(|error| self.dispatch_failed(error)),
        };
        if replayed.is_ok() && self.sync_stages {
            let label = format!("{:?} graph {:?}", run.draft, run.graph);
            for (rank, &stream) in streams.iter().enumerate() {
                self.sync_stage(&label, rank, stream)?;
            }
        }
        replayed
    }

    /// `pipeline::run_ranks` of a graph run whose MoE sublayers overlap the
    /// shared gate/up (`crate::shared_overlap`): every rank missing its
    /// graph of the bucket captures its program over its compute and side
    /// streams, in units the graph pieces never split, before any rank
    /// launches; then every rank replays piece-major, as one `Unit`.
    fn run_overlapped(
        &mut self,
        draft: Option<DraftPass>,
        (bucket, items): (u32, u32),
        programs: &[Vec<PreparedLaunch>],
        stages: &[Vec<Stage>],
    ) -> Result<(), DispatchError> {
        if self.collectives.is_poisoned() {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState).into());
        }
        let ranks = self.ranks.len();
        if self.side_lanes.len() != ranks || self.symbols.len() != ranks {
            return Err(DispatchError::InvalidPlan);
        }
        let graphs = match draft {
            None => &mut self.graphs,
            Some(DraftPass::Extend) => &mut self.draft_graphs[0],
            Some(DraftPass::Step) => &mut self.draft_graphs[1],
        };
        for (rank, lanes) in self.side_lanes.iter().enumerate() {
            let key = GraphKey {
                rank: rank as u32,
                bucket,
                items,
            };
            if graphs.contains(key) {
                continue;
            }
            let (program, stages) = (&programs[rank], &stages[rank]);
            if program.is_empty() || program.len() != stages.len() {
                return Err(DispatchError::InvalidPlan);
            }
            let units = crate::shared_overlap::units(stages).map_err(|error| {
                let _ = self
                    .services
                    .log(format!("{NAME}: shared gate/up overlap: {error}").as_bytes());
                DispatchError::InvalidPlan
            })?;
            hooks::overlapped(rank, (draft, (bucket, items)), &units, stages);
            let symbols = &self.symbols[rank];
            graphs.capture(&self.services, lanes.compute, key, units.len(), |range| {
                units[range]
                    .iter()
                    .try_for_each(|unit| lanes.issue(&self.services, symbols, program, unit))
            })?;
        }
        let streams = self
            .side_lanes
            .iter()
            .map(|lanes| lanes.compute)
            .collect::<Vec<_>>();
        hooks::overlapped_replay((draft, (bucket, items)));
        let _unwinding = PoisonOnUnwind(&self.collectives);
        let replayed = Unit::default().record(
            &self.collectives,
            graphs
                .replay_piece_major(&self.services, &streams, (bucket, items), hooks::submitted)
                .map_err(DispatchError::from),
        )?;
        if replayed {
            Ok(())
        } else {
            Err(DispatchError::InvalidPlan)
        }
    }

    /// Before a run past `Target::phased_reset_runs` runs since the last
    /// reset (or create), zeroes every rank's phased flags and epoch (of
    /// the BF16 partials' state and the MoE sublayers' F32 one) while every
    /// rank is idle: every rank's stream finished (no
    /// phased call runs, so no peer writes a flag), then each rank's
    /// zeroing, then every rank's stream again (no rank's next call writes
    /// a peer's flags before that peer's zeroing ran). The epochs restart
    /// from zero on every rank alike, as at create, so no flag tag repeats
    /// between resets (the phased state's tag period) at any sequence of
    /// entries, row counts, eager runs and graph replays, which read the
    /// state on the device. A failed wait or zeroing poisons the instance:
    /// the ranks' states may then disagree.
    fn quiesce_phased(
        &mut self,
        streams: &[runtime_bundle_abi::BundleStreamHandleV1],
    ) -> Result<(), BundleAbiError> {
        if !self.buffers.iter().any(|buffers| buffers.comm.phased) {
            return Ok(());
        }
        if self.phased_runs < self.phased_reset_runs {
            self.phased_runs += 1;
            return Ok(());
        }
        let zeroed = (|| {
            for &stream in streams {
                self.services.synchronize_stream(stream)?;
            }
            for (buffers, &stream) in self.buffers.iter().zip(streams) {
                let comm = &buffers.comm;
                for state in [&comm.phased_state, &comm.rounded_phased_state] {
                    self.services.zero_device_buffer(stream, state.flags)?;
                    self.services.zero_device_buffer(stream, state.epoch)?;
                }
            }
            for &stream in streams {
                self.services.synchronize_stream(stream)?;
            }
            Ok(())
        })();
        if let Err(error) = zeroed {
            self.collectives.poison();
            self.log(&format!(
                "zeroing the phased all-reduce's state failed: {error}; the instance is poisoned"
            ));
            return Err(error);
        }
        self.log(&format!(
            "zeroed every rank's phased all-reduce flags and epoch after {} runs",
            self.phased_runs
        ));
        self.phased_runs = 1;
        Ok(())
    }

    /// A failed submission's status: the host's own error, or the dispatch
    /// refusal's, which is logged.
    fn dispatch_failed(&self, error: DispatchError) -> BundleAbiError {
        match error {
            DispatchError::Host(error) => error,
            error => {
                self.log(&format!("issue failed: {error:?}"));
                BundleAbiError::Status(bundle_runtime::dispatch::status(error))
            }
        }
    }

    /// Waits for `rank`'s queued work (`super::SYNC_STAGES`), logging
    /// `label` before the wait, so a hang names what it waits for, and the
    /// wait's time after it. A failed wait poisons the instance: work is
    /// queued on every rank, and nothing established that it finished.
    fn sync_stage(
        &self,
        label: &str,
        rank: usize,
        stream: runtime_bundle_abi::BundleStreamHandleV1,
    ) -> Result<(), BundleAbiError> {
        self.log(&format!("stage sync: rank {rank} waits for {label}"));
        let started = std::time::Instant::now();
        if let Err(error) = self.services.synchronize_stream(stream) {
            self.collectives.poison();
            self.log(&format!(
                "stage sync: rank {rank} failed waiting for {label}: {error}; the instance is \
                 poisoned"
            ));
            return Err(error);
        }
        self.log(&format!(
            "stage sync: rank {rank} finished {label} in {:.3}s",
            started.elapsed().as_secs_f64()
        ));
        Ok(())
    }

    /// The key of an eager prefill's programs among the captured ones
    /// (`super::prefill_graphs::Key`), when they may be captured: not for a
    /// decode or verify phase, a prefill that injects encoder rows (the
    /// inject reads the encoder's state of the step), the stage-sync
    /// diagnostics (which synchronize between eager segments) or text taps
    /// (which read every stage's buffers). A context-parallel instance's
    /// programs are captured with their group launches: every group state
    /// (flags, epochs, outputs) lives on the device and is fixed at create,
    /// the launches' spans and scalars follow from the key (the pieces' rows
    /// and offsets pick each launch's bucket and views), and the replay
    /// issues each rank's launches in program order, so every state sees the
    /// same calls in the same order as the eager run's, as the decode
    /// graphs' group launches do.
    #[allow(clippy::too_many_arguments)]
    fn prefill_key(
        &self,
        arena: usize,
        inputs: &PhaseInputs,
        layout: &InputLayout,
        pieces: &[(u32, PhaseInputs)],
        head_rows: u32,
        draft: Option<DraftPass>,
        inject: bool,
    ) -> Option<PrefillKey> {
        if arena != PREFILL || inject || self.sync_stages || self.taps.text() {
            return None;
        }
        if draft.is_none() && dma::eligible_inputs(self, inputs) {
            // The first DMA policy is eager. Cached graphs must not bypass
            // its transport on repeated prefill shapes.
            return None;
        }
        // The pieces' first rows, rows and first items, as `rank_pieces`
        // binds them; one piece binds none.
        let pieces = match pieces.len() {
            0 | 1 => Vec::new(),
            _ => pieces
                .iter()
                .map(|(first, piece)| {
                    let item = inputs
                        .item_order
                        .iter()
                        .position(|&order| order == piece.item_order[0])
                        .unwrap_or(0) as u64;
                    (*first, piece.live_tokens(), item)
                })
                .collect(),
        };
        Some(PrefillKey {
            draft,
            rows: inputs.live_tokens(),
            head_rows,
            layout: *layout,
            pieces,
        })
    }

    /// Issues an eager prefill whose programs may be captured
    /// (`super::prefill_graphs`): a replay of its captured programs, else
    /// eagerly (`launch_eager`) the first time its key runs, and on the
    /// second a capture of every rank's program, then its replay. A
    /// request's last prompt chunk (`PrefillGraphs::pins`) captures at its
    /// first run.
    fn launch_prefill(
        &mut self,
        streams: &[runtime_bundle_abi::BundleStreamHandleV1],
        key: &PrefillKey,
        programs: &[Vec<PreparedLaunch>],
        stages: &[Vec<Stage>],
        collectives: &[Vec<bool>],
    ) -> Result<(), DispatchError> {
        if !self.prefill_graphs.contains(key) {
            if !self.prefill_graphs.pins(key) && !self.prefill_graphs.repeats(key) {
                return self.launch_eager(streams, programs, stages, collectives);
            }
            if self.collectives.is_poisoned() {
                return Err(BundleAbiError::Status(BundleStatusV1::InvalidState).into());
            }
            if let Err(error) = check_collectives(programs, collectives) {
                self.log(&error);
                return Err(DispatchError::InvalidPlan);
            }
            match self.prefill_graphs.room(&self.graphs, key) {
                Room::Free => {}
                Room::Newest => self.release_newest_programs(streams)?,
                Room::Every => self.invalidate_graphs(streams)?,
            }
            let started = std::time::Instant::now();
            let captured = self.prefill_graphs.capture(
                &self.services,
                streams,
                &self.symbols,
                &self.graphs,
                key.clone(),
                programs,
            );
            if let Err(error) = captured {
                // The ledger owns what the capture took; free it now.
                if let Err(released) = self.invalidate_graphs(streams) {
                    self.log(&format!(
                        "releasing the graphs after a failed prefill capture: {released:?}"
                    ));
                }
                return Err(error);
            }
            self.log(&format!(
                "prefill graph capture: {} rows, {} scored, {:?}, {} pieces, {} launches on \
                 rank 0, {:.3}s{}",
                key.rows,
                key.head_rows,
                key.draft,
                key.pieces.len().max(1),
                programs[0].len(),
                started.elapsed().as_secs_f64(),
                if self.prefill_graphs.pins(key) {
                    ", pinned"
                } else {
                    ""
                }
            ));
        }
        let _unwinding = PoisonOnUnwind(&self.collectives);
        self.prefill_graphs.replay(
            &self.services,
            streams,
            &self.collectives,
            key,
            hooks::submitted,
        )
    }

    /// Captures the programs of a request's last prompt chunk at every
    /// length up to the granule (`PrefillGraphs::pins`), as create does
    /// before any other graph exists, and returns how many it captured.
    /// Each is lowered from a one-item chunk of fresh rows scoring its
    /// last: equal keys lower to equal launches, so a request's last chunk
    /// of that length replays the program whatever its tokens, slot and
    /// committed rows. Nothing runs. A failed capture returns its error,
    /// its graphs and the earlier programs' left to the ledger.
    pub(super) fn capture_last_chunks(&mut self) -> Result<usize, String> {
        use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
        let streams = self
            .ranks
            .iter()
            .map(|&(_, _, stream)| stream)
            .collect::<Vec<_>>();
        let mut captured = 0;
        for rows in 1..=PREFILL_GRANULE {
            let item = BatchItemV1::new(0, false, rows, true, vec![0; rows as usize]);
            let append = BatchAppendV1::new(0, u64::from(rows), u64::from(rows), vec![0], None);
            let request = BatchRequestV1::new(BatchWireVersionV1::V3, vec![item], vec![append])
                .with_state_slots(vec![0])
                .encode()
                .map_err(|error| format!("{rows} rows: {error:?}"))?;
            let (step, _) = self
                .pack(&request)
                .map_err(|error| format!("{rows} rows: {error:?}"))?;
            let run = self
                .plan_run(step.prefill, 0, false)
                .map_err(|error| format!("{rows} rows: {error:?}"))?;
            let Some(key) = run.prefill else {
                return Err(format!("{rows} rows: no prefill key"));
            };
            if !self.prefill_graphs.pins(&key) || self.prefill_graphs.contains(&key) {
                continue;
            }
            check_collectives(&run.programs, &run.collectives)
                .map_err(|error| format!("{rows} rows: {error}"))?;
            self.prefill_graphs
                .capture(
                    &self.services,
                    &streams,
                    &self.symbols,
                    &self.graphs,
                    key,
                    &run.programs,
                )
                .map_err(|error| format!("{rows} rows: {error:?}"))?;
            captured += 1;
        }
        Ok(captured)
    }

    /// Releases every graph of the instance once every rank's queued work
    /// is done (`wait_before_release`): the host releases graphs only
    /// newest first, so the prefill store makes room under a decode or
    /// drafter graph (`Room::Every`), and drops a failed capture's graphs,
    /// only by releasing them all. Decode and verify buckets and prefill
    /// programs capture again at their next use; runs this step planned
    /// without lowering lower their programs at launch (`lower_missing`).
    /// A failed wait poisons the instance; a failed release after every
    /// wait stays retryable.
    fn invalidate_graphs(
        &mut self,
        streams: &[runtime_bundle_abi::BundleStreamHandleV1],
    ) -> Result<(), DispatchError> {
        self.wait_before_release(streams, "every graph")?;
        self.log("releasing every graph of the instance for the prefill programs");
        bundle_runtime::entry::Bundle::release_graphs(self)?;
        Ok(())
    }

    /// Releases the prefill programs above every decode and drafter graph
    /// (`Room::Newest`) once every rank's queued work is done, as
    /// `invalidate_graphs` does, keeping every older graph; with none to
    /// release, or when a release fails (its forgotten program's graphs
    /// stay in the ledger), releases every graph. A failure of that too
    /// fails the step, and every later capture first retries it
    /// (`PrefillGraphs::room`). Released programs run eagerly again
    /// (`lower_missing` lowers a run this step planned to replay).
    fn release_newest_programs(
        &mut self,
        streams: &[runtime_bundle_abi::BundleStreamHandleV1],
    ) -> Result<(), DispatchError> {
        self.wait_before_release(streams, "the newest prefill programs")?;
        let released = match self.prefill_graphs.release_newest(&self.services) {
            Ok(released) => released,
            Err(error) => {
                self.log(&format!(
                    "releasing the newest prefill programs failed: {error}; releasing every graph"
                ));
                0
            }
        };
        if released == 0 {
            return self.invalidate_graphs(streams);
        }
        self.log(&format!(
            "released the {released} newest prefill programs, keeping every older graph"
        ));
        Ok(())
    }

    /// Waits for every rank's queued work before graphs are released (a
    /// replay queued earlier in the step may still run). Without a
    /// confirmed wait nothing established that it finished, so a failed
    /// wait poisons the instance, as a failed stage sync does
    /// (`sync_stage`), which keeps its graphs and resources.
    fn wait_before_release(
        &self,
        streams: &[runtime_bundle_abi::BundleStreamHandleV1],
        what: &str,
    ) -> Result<(), DispatchError> {
        for (rank, &stream) in streams.iter().enumerate() {
            if let Err(error) = self.services.synchronize_stream(stream) {
                self.collectives.poison();
                self.log(&format!(
                    "rank {rank} failed waiting before releasing {what}: {error}; the instance \
                     is poisoned"
                ));
                return Err(error.into());
            }
        }
        Ok(())
    }

    /// Issues an eager run's programs collective by collective: each rank's
    /// launches up to and including its next collective, rank after rank,
    /// then the next such segment. A stream holds a bounded number of
    /// pending launches, and a rank spins at a collective until every peer
    /// reaches it; queueing one rank's whole program first would leave the
    /// host blocked in a launch call on that rank's full stream while it
    /// spins at its first collective, before any peer has work (bring-up
    /// 2's hang). Ranks whose programs differ in collectives are refused
    /// before anything is queued. A failure or panic once a segment is
    /// queued poisons the instance, as `run_ranks` does.
    fn launch_eager(
        &self,
        streams: &[runtime_bundle_abi::BundleStreamHandleV1],
        programs: &[Vec<PreparedLaunch>],
        stages: &[Vec<Stage>],
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
        let depth = segments.iter().map(Vec::len).max().unwrap_or(0);
        let round = |segment: usize| -> Result<(), DispatchError> {
            if self.sync_stages {
                for (rank, launches) in segments.iter().enumerate() {
                    let Some(launches) = launches.get(segment) else {
                        continue;
                    };
                    // The segment's launch range on this rank, and its stages.
                    let start = segments[rank][..segment]
                        .iter()
                        .map(|s| s.len())
                        .sum::<usize>();
                    let end = start + launches.len();
                    let named = stages.get(rank).and_then(|stages| stages.get(start..end));
                    let (first, last) = match named {
                        Some([first, .., last]) => (Some(first), Some(last)),
                        Some([only]) => (Some(only), Some(only)),
                        _ => (None, None),
                    };
                    let label = format!(
                        "eager segment {segment} of {depth}, launches {start}..{end}, \
                         {first:?} to {last:?} ({})",
                        launches[launches.len() - 1].symbol
                    );
                    self.sync_stage(&label, rank, streams[rank])?;
                }
            }
            Ok(())
        };
        launch_segments(
            &self.collectives,
            &mut Unit::default(),
            &mut started,
            &segments,
            |rank, launches| {
                launch_all(&self.services, streams[rank], &self.symbols[rank], launches)
            },
            hooks::submitted,
            round,
        )
    }

    /// Issues an eager target run's programs stage by stage for its text
    /// taps (`crate::taps`): each stage's launches on every rank, collective
    /// by collective as `launch_eager` queues them, then the stage's
    /// selected taps read back on each rank's stream, so every rank queued
    /// the stage's collectives before any rank waits. The ranks' launches must follow the phase's stages in
    /// program order (a rank may lower a stage to none); otherwise, or
    /// when their collectives differ, the run is refused before anything
    /// is queued. A failure or panic once a stage is queued poisons the
    /// instance, as `launch_eager` does.
    #[cfg(feature = "diagnostic-taps")]
    fn launch_tapped(
        &mut self,
        streams: &[runtime_bundle_abi::BundleStreamHandleV1],
        run: &Run,
        programs: &[Vec<PreparedLaunch>],
        stages: &[Vec<Stage>],
        collectives: &[Vec<bool>],
    ) -> Result<(), BundleAbiError> {
        let refused = |instance: &Self, error: DispatchError| instance.dispatch_failed(error);
        if self.collectives.is_poisoned() {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
        }
        if programs.len() != streams.len()
            || stages.len() != streams.len()
            || self.symbols.len() != streams.len()
            || programs.iter().zip(stages).any(|(p, s)| p.len() != s.len())
        {
            return Err(refused(self, DispatchError::InvalidPlan));
        }
        let shape = self.config.shape;
        let order = crate::program::stages(shape, run.phase);
        // Each rank's launch range of each stage of `order`.
        let mut walk = vec![vec![0..0; streams.len()]; order.len()];
        for (rank, stages) in stages.iter().enumerate() {
            let mut at = 0;
            for (ranges, &stage) in walk.iter_mut().zip(&order) {
                let end = at + stages[at..].iter().take_while(|&&s| s == stage).count();
                ranges[rank] = at..end;
                at = end;
            }
            if at != stages.len() {
                self.log(&format!(
                    "rank {rank}'s launch {at} ({:?}) is out of the phase's stage order",
                    stages[at]
                ));
                return Err(refused(self, DispatchError::InvalidPlan));
            }
        }
        if let Err(error) = check_collectives(programs, collectives) {
            self.log(&error);
            return Err(refused(self, DispatchError::InvalidPlan));
        }
        let program = match run.phase {
            Phase::Prefill { .. } => "Prefill",
            Phase::Decode(_) => "Decode",
            Phase::Verify { .. } => "Verify",
        };
        let ranks = streams.len() as u32;
        if let Err(error) = self.taps.begin(program, ranks, &run.inputs, shape, &order) {
            self.log(&format!("taps: {error}"));
            return Err(BundleAbiError::Status(BundleStatusV1::InternalError));
        }
        let _unwinding = PoisonOnUnwind(&self.collectives);
        let mut unit = Unit::default();
        let mut started = vec![false; streams.len()];
        for (&stage, ranges) in order.iter().zip(&walk) {
            // The stage's launches collective by collective, as
            // `launch_eager` queues a program.
            let segments = ranges
                .iter()
                .enumerate()
                .map(|(rank, range)| {
                    collective_segments(
                        &programs[rank][range.clone()],
                        &collectives[rank][range.clone()],
                    )
                })
                .collect::<Vec<_>>();
            launch_segments(
                &self.collectives,
                &mut unit,
                &mut started,
                &segments,
                |rank, launches| {
                    launch_all(&self.services, streams[rank], &self.symbols[rank], launches)
                },
                hooks::submitted,
                |_| Ok(()),
            )
            .map_err(|error| refused(self, error))?;
            let captured = self
                .taps
                .capture(&self.services, streams, &self.buffers, shape, stage);
            if let Err(error) = captured {
                // Work is queued on every rank, and nothing established
                // that it finished.
                self.collectives.poison();
                return Err(match error {
                    runtime_bundle_sdk::taps::TapError::Host(error) => error,
                    runtime_bundle_sdk::taps::TapError::Io(error) => {
                        self.log(&format!("taps: {stage:?}: {error}"));
                        BundleAbiError::Status(BundleStatusV1::InternalError)
                    }
                });
            }
        }
        self.taps.end();
        Ok(())
    }

    /// One step, its eager prefill before its decode or verify phase:
    /// refused before the ledger opens it unless it can run, cancelled
    /// through the ledger if issuing it fails.
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
            .map(|(&(context, _, _), buffers)| (context, buffers.comm.decode_logits))
            .collect::<Vec<_>>();
        let rank0 = &self.buffers[0].comm;
        let views = logits::ordered_views_owned(
            self.config.shape.model_limits(),
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
        // one (rank 0 for a prefill's rows, each decode or verify block's
        // owner) completes its queued work first, after all of its peers
        // queued theirs, so no row is still being written.
        let owners = views
            .iter()
            .map(|view| view.rank)
            .collect::<std::collections::BTreeSet<_>>();
        for rank in owners {
            let &(_, _, stream) = self
                .ranks
                .get(rank as usize)
                .ok_or(BundleAbiError::InvalidRange("glm5next logits owner"))?;
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
        let (step, drafts) = match self.pack(request) {
            Ok(packed) => packed,
            Err(error) => return error,
        };
        // The drafter's rows of the step, planned before it opens.
        let draft_plan = match &self.drafter {
            Some(drafter) => match drafter.plan(&step.prefill, &step.decode, &drafts) {
                Ok(plan) => Some(plan),
                Err(error) => {
                    self.log(&format!("the MTP drafter cannot run this step: {error:?}"));
                    return match error {
                        DraftError::History => BundleStatusV1::InvalidState,
                        DraftError::Step => BundleStatusV1::InvalidArgument,
                    };
                }
            },
            None => None,
        };
        let mut draft_runs = match &draft_plan {
            Some(plan) => match self.plan_draft_runs(plan) {
                Ok(runs) => runs,
                Err(error) => return error,
            },
            None => DraftRuns::default(),
        };
        // The encoder runs before the transaction opens: a failure leaves
        // no step to cancel, and the prefill's injects read what it wrote.
        let streams = self
            .ranks
            .iter()
            .map(|&(_, _, stream)| stream)
            .collect::<Vec<_>>();
        let inject = match &self.encoder {
            Some(encoder) => match encoder.run(
                &self.services,
                &streams,
                &self.symbols,
                request,
                &step.prefill,
                &self.collectives,
            ) {
                Ok(inject) => inject,
                Err(error) => {
                    if let Err(taps) = self.taps.discard(encoder) {
                        self.log(&format!("encoder taps: {taps}"));
                    }
                    return error;
                }
            },
            None => false,
        };
        // A diagnostic build writes the step's encoder rows (`crate::taps`).
        if let Some(encoder) = &self.encoder
            && let Err(error) =
                self.taps
                    .step(&self.services, streams[0], encoder, request, &step.prefill)
        {
            self.log(&format!("encoder taps: {error}"));
            return BundleStatusV1::InternalError;
        }
        // Each prompt-ending prefill item that keeps its state at its
        // anchor takes the next anchor spares.
        let anchors = match crate::step::anchors(&step.prefill, PREFILL_GRANULE) {
            Ok(anchors) if anchors.len() <= ANCHOR_ITEMS as usize => anchors,
            Ok(anchors) => {
                self.log(&format!(
                    "{} anchor items exceed the package's {ANCHOR_ITEMS}",
                    anchors.len()
                ));
                return BundleStatusV1::InvalidArgument;
            }
            Err(error) => {
                self.log(&error);
                return BundleStatusV1::InvalidArgument;
            }
        };
        // The drafter's retained rows of an anchor start at its frontier,
        // which must lie in the item's rows (`anchor_retains`).
        if self.drafter.is_some() {
            let shape = self.config.shape;
            for &(request, tokens) in &anchors {
                let prefix = (0..step.prefill.request_ids.len())
                    .find(|&item| step.prefill.request_ids[item] == request)
                    .map_or(0, |item| step.prefill.prefix_len[item] as u64);
                if crate::drafter::anchor_frontier(shape, tokens) < prefix {
                    self.log(&format!(
                        "request {request}'s anchor at {tokens} lies too close to its first row"
                    ));
                    return BundleStatusV1::InvalidArgument;
                }
            }
        }
        let mut items = crate::step::items(&step.prefill);
        let prefill_items = items.len();
        items.extend(crate::step::items(&step.decode));
        let mut runs = Vec::new();
        for (inputs, first, inject) in [
            (step.prefill, 0, inject),
            (step.decode, prefill_items, false),
        ] {
            if inputs.request_ids.is_empty() {
                continue;
            }
            match self.plan_run(inputs, first, inject) {
                Ok(run) => runs.push(run),
                Err(error) => return error,
            }
        }
        if self.ledger.preview(&items).is_err() || self.ledger.begin(&items, anchors).is_err() {
            return BundleStatusV1::InvalidArgument;
        }
        let issued = self.issue_step(&mut runs, &mut draft_runs, draft_plan.as_ref(), sink);
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
            self.pending = None;
            let _ = self.ledger.cancel();
            if let Err(taps) = self.taps.abandon() {
                self.log(&format!("taps: {taps}"));
            }
            return status(error);
        }
        BundleStatusV1::Success
    }

    /// Issues the step: the target's decode or verify inputs first (the
    /// drafter writes its drafts into them), the drafter's extend pass and
    /// draft steps, the target's prefill, the drafter's extend pass over the
    /// prefill's completed rows, then the target's decode or verify phase.
    /// Each target phase stages its hidden rows for commit to retain. The
    /// drafts reach the sink before the logits.
    fn issue_step(
        &mut self,
        runs: &mut [Run],
        draft_runs: &mut DraftRuns,
        plan: Option<&DraftPlan>,
        sink: &mut ExecuteResultSinkV1,
    ) -> Result<(), BundleAbiError> {
        let (prefill, decode): (Vec<_>, Vec<_>) =
            runs.iter_mut().partition(|run| run.arena == PREFILL);
        let mut decode = decode.into_iter().next();
        if let Some(run) = &decode {
            self.upload(run)?;
        }
        for run in &mut draft_runs.before {
            self.issue(run)?;
        }
        let prefilled = !prefill.is_empty();
        for run in prefill {
            self.issue(run)?;
            self.stage(run)?;
        }
        if let Some(run) = &mut draft_runs.prefill {
            self.issue(run)?;
        }
        if let Some(run) = &mut decode {
            // The prefill rewrote the KDA row offsets and checkpoint tables
            // the decode or verify phase reads; its inputs, with the
            // drafter's tokens, stay as they are.
            if prefilled {
                self.upload_side(run)?;
            }
            self.launch(run)?;
            self.stage(run)?;
        }
        if let Some(plan) = plan {
            let mut pending = plan.pending();
            if let Some(run) = &decode {
                self.emit_drafts(run, plan, &mut pending, sink)?;
            }
            self.pending = Some(pending);
        }
        self.emit_logits(runs, sink)
    }

    /// Copies a target phase's hidden rows that commit may retain into the
    /// drafter's staging on every rank: each prefill item's last row, every
    /// decode or verify row. The drafter reads `h_p` as HF's MTP does: the
    /// mean of the streams after `model.norm` (the final norm's output),
    /// which its own `hnorm` then normalizes again.
    fn stage(&self, run: &Run) -> Result<(), BundleAbiError> {
        if self.drafter.is_none() {
            return Ok(());
        }
        let shape = self.config.shape;
        let row = |span, row| crate::buffers::DrafterBuffers::row(shape, span, row);
        for (rank, &(_, _, stream)) in self.ranks.iter().enumerate() {
            let buffers = &self.buffers[rank];
            let drafter = buffers
                .drafter
                .ok_or(BundleAbiError::InvalidRange("glm5next drafter buffers"))?;
            let normed = buffers.get(Buffer::Normed);
            if run.arena == PREFILL {
                for item in 0..run.inputs.request_ids.len() {
                    let last = run.inputs.q_start[item] + run.inputs.q_len[item] - 1;
                    self.services.copy_device_to_device(
                        stream,
                        row(drafter.staged[0], item as u32),
                        row(normed, last as u32),
                    )?;
                }
                for (from, to, rows) in self.anchor_retains(run, rank)? {
                    let bytes = u64::from(rows) * u64::from(shape.profile.hidden) * 2;
                    let span = |span: DeviceBufferSpanV1| {
                        DeviceBufferSpanV1::new(span.buffer, span.offset, bytes)
                    };
                    self.services.copy_device_to_device(
                        stream,
                        span(to),
                        span(row(normed, from)),
                    )?;
                }
            } else {
                let rows = u64::from(run.inputs.live_tokens());
                let bytes = rows * u64::from(shape.profile.hidden) * 2;
                let span = |span: DeviceBufferSpanV1| {
                    DeviceBufferSpanV1::new(span.buffer, span.offset, bytes)
                };
                self.services.copy_device_to_device(
                    stream,
                    span(drafter.staged[1]),
                    span(normed),
                )?;
            }
        }
        Ok(())
    }

    /// The retained rows a prefill run's anchor items keep in their
    /// spares on rank `rank`, as (first row of the run, first spare row,
    /// rows) per run of the ring: the hidden rows from the anchor's drafter
    /// frontier (`crate::drafter::anchor_frontier`) to the anchor, which a
    /// continuation from the anchor reads, each at its position modulo the
    /// ring.
    fn anchor_retains(
        &self,
        run: &Run,
        rank: usize,
    ) -> Result<Vec<(u32, DeviceBufferSpanV1, u32)>, BundleAbiError> {
        let shape = self.config.shape;
        let invalid = |error: String| {
            self.log(&error);
            BundleAbiError::InvalidRange("glm5next anchor retained rows")
        };
        let step = self
            .ledger
            .step()
            .ok_or(BundleAbiError::InvalidRange("no open glm5next step"))?;
        let state = &self.buffers[rank].state;
        let retained = state.retained().map_err(invalid)?;
        let ring = u64::from(crate::drafter::RETAINED_ROWS);
        let inputs = &run.inputs;
        let mut retains = Vec::new();
        for (anchor, &(request, tokens)) in step.anchors.iter().enumerate() {
            let Some(item) = (0..inputs.request_ids.len()).find(|&item| {
                inputs.request_ids[item] == request
                    && inputs.anchors.get(item).copied().unwrap_or(false)
            }) else {
                continue;
            };
            let frontier = crate::drafter::anchor_frontier(shape, tokens);
            let prefix = inputs.prefix_len[item] as u64;
            if frontier < prefix || tokens - frontier > ring {
                return Err(invalid(format!(
                    "request {request}'s anchor at {tokens} keeps rows from {frontier}, \
                     outside its rows from {prefix} or its ring"
                )));
            }
            let entry = state
                .anchor_entry("mtp.retained", anchor as u32)
                .map_err(invalid)?;
            let mut position = frontier;
            while position < tokens {
                let rows = (ring - position % ring).min(tokens - position);
                let spare_row = entry * ring + position % ring;
                let to = crate::buffers::DrafterBuffers::row(shape, retained, spare_row as u32);
                let from = inputs.q_start[item] as u64 + (position - prefix);
                retains.push((from as u32, to, rows as u32));
                position += rows;
            }
        }
        Ok(retains)
    }

    /// Each drafting item's drafts, which the drafter's argmax wrote into
    /// the verify inputs on the device: read back from rank 0, emitted, and
    /// kept for commit's token history.
    fn emit_drafts(
        &self,
        run: &Run,
        plan: &DraftPlan,
        pending: &mut Pending,
        sink: &mut ExecuteResultSinkV1,
    ) -> Result<(), BundleAbiError> {
        if plan.drafting.is_empty() {
            return Ok(());
        }
        let tokens = DeviceInputs::within(self.arenas[run.arena][0], &run.layout).token_ids;
        let count = run.inputs.token_ids.len();
        let mut bytes = vec![0; 4 * count];
        let span = DeviceBufferSpanV1::new(tokens.buffer, tokens.offset, bytes.len() as u64);
        self.services
            .copy_device_to_host(self.ranks[0].2, &mut bytes, span)?;
        let device = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect::<Vec<_>>();
        let vocab = self.config.shape.geometry.vocab();
        for &(request, item, drafts) in &plan.drafting {
            let start = run.inputs.q_start[item] as usize;
            let rows = &device[start + 1..start + 1 + drafts as usize];
            if rows.iter().any(|&token| token >= vocab) {
                return Err(BundleAbiError::InvalidRange(
                    "an MTP draft outside the vocabulary",
                ));
            }
            sink.emit_draft_tokens(request, rows)?;
            let mut kept = vec![run.inputs.token_ids[start]];
            kept.extend_from_slice(rows);
            pending
                .set_tokens(request, &kept)
                .map_err(|_| BundleAbiError::InvalidRange("an MTP draft's item"))?;
        }
        Ok(())
    }

    /// Before the ledger commits `plan`, keeping `kept` rows of each verify
    /// item: copies each one's last kept checkpoint into its slot on every
    /// rank, in the target's layers (the drafter's indexer tail follows its
    /// frontier in place), then commits the drafter.
    pub(super) fn prepare_commit(
        &mut self,
        kept: &[(u64, u32)],
        plan: &bundle_runtime::ledger::CommitPlan,
    ) -> Result<(), BundleStatusV1> {
        let Some(step) = self.ledger.step().cloned() else {
            return Err(BundleStatusV1::InvalidState);
        };
        let state = &self.buffers[0].state;
        let copies =
            match crate::step::commit_copies(&state.state, state.extent, plan.items(), |item| {
                step.verify_index(item)
            }) {
                Ok(copies) => copies,
                Err(error) => {
                    self.log(&error);
                    return Err(BundleStatusV1::InternalError);
                }
            };
        for (rank, &(_, _, stream)) in self.ranks.iter().enumerate() {
            for copy in &copies {
                if let Err(error) = self.copy_layers(rank, stream, copy) {
                    self.log(&format!("commit failed: {error}"));
                    return Err(status(error));
                }
            }
        }
        if let Err(error) = self.commit_drafter(kept) {
            self.log(&format!("drafter commit failed: {error}"));
            return Err(status(error));
        }
        Ok(())
    }

    /// Commit's state copies on the own kernels: one op per checkpointing
    /// buffer over its entry's bytes in the layers commit copies, the
    /// layers one layer stride apart in the source and destination.
    pub(super) fn prepare_state_copies(
        own: &dyn OwnOps,
        pools: &crate::buffers::StatePools,
    ) -> Result<super::StateCopies, Refused> {
        crate::step::checkpoint_layers(&pools.state)
            .into_iter()
            .map(|(index, layers)| {
                let (entry, stride) = pools.entry_layers(index, 0);
                let op = StateCopy {
                    bytes: entry.length,
                    layers: u32::try_from(layers)
                        .map_err(|_| Refused::Rule("a state buffer's layers".into()))?,
                    source_stride: stride,
                    destination_stride: stride,
                };
                let prepared = own.state_copy(&op, Need::rows(1))?;
                Ok((index, op, prepared))
            })
            .collect()
    }

    /// Copies `copy`'s entry in every layer it names on `rank` with one
    /// launch of its buffer's state copy, byte for byte as a host copy per
    /// layer would, on the rank's stream.
    fn copy_layers(
        &self,
        rank: usize,
        stream: runtime_bundle_abi::BundleStreamHandleV1,
        copy: &crate::step::EntryCopy,
    ) -> Result<(), BundleAbiError> {
        let invalid = || BundleAbiError::InvalidRange("glm5next state copy");
        let (_, layers, op) = self
            .state_copies
            .iter()
            .find(|(index, layers, _)| {
                *index == copy.buffer && u64::from(layers.layers) == copy.layers
            })
            .ok_or_else(invalid)?;
        let pools = &self.buffers[rank].state;
        let (from, _) = pools.entry_layers(copy.buffer, copy.from);
        let (to, _) = pools.entry_layers(copy.buffer, copy.to);
        let spans = StateCopyBuffers {
            source: DeviceBufferSpanV1::new(from.buffer, from.offset, layers.source_span()),
            destination: DeviceBufferSpanV1::new(to.buffer, to.offset, layers.destination_span()),
        };
        let launches =
            crate::ops::own::launch(op, spans, Counts::rows(1)).map_err(|_| invalid())?;
        for launch in crate::ops::launches(launches) {
            crate::submit::launch(
                &self.services,
                stream,
                &self.symbols[rank],
                &launch.symbol,
                launch.plan,
                &launch.arguments,
                &launch.bindings,
            )
            .map_err(|error| self.dispatch_failed(error))?;
        }
        Ok(())
    }

    /// Commits the drafter's rows of the open step: each frontier moves,
    /// and every rank copies the staged hidden rows commit retains into
    /// their slots' retained rows.
    fn commit_drafter(&mut self, kept: &[(u64, u32)]) -> Result<(), BundleAbiError> {
        let (Some(drafter), Some(pending)) = (&mut self.drafter, self.pending.take()) else {
            return Ok(());
        };
        let retains = drafter
            .commit(&pending, |request| {
                kept.iter()
                    .find(|&&(id, _)| id == request)
                    .map(|&(_, rows)| rows)
            })
            .map_err(|_| BundleAbiError::InvalidRange("the MTP drafter's commit"))?;
        let shape = self.config.shape;
        let row = |span, row| crate::buffers::DrafterBuffers::row(shape, span, row);
        for (rank, &(_, _, stream)) in self.ranks.iter().enumerate() {
            let buffers = self.buffers[rank]
                .drafter
                .ok_or(BundleAbiError::InvalidRange("glm5next drafter buffers"))?;
            let retained = self.buffers[rank]
                .state
                .retained()
                .map_err(|_| BundleAbiError::InvalidRange("glm5next retained rows"))?;
            for retain in &retains {
                self.services.copy_device_to_device(
                    stream,
                    row(retained, retain.retained),
                    row(buffers.staged[retain.phase], retain.staged),
                )?;
            }
        }
        Ok(())
    }
}

/// A test's panic once a chosen rank's share of a unit is queued.
pub(crate) mod hooks {
    #[cfg(test)]
    thread_local! {
        static PANIC_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    }

    /// Panics once `rank`'s share of the next unit is queued.
    #[cfg(test)]
    pub(crate) fn panic_after(rank: usize) {
        PANIC_AFTER.with(|cell| cell.set(Some(rank)));
    }

    #[cfg(test)]
    thread_local! {
        static LOWERED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Rank programs lowered on this thread so far.
    #[cfg(test)]
    pub(crate) fn lowered_programs() -> usize {
        LOWERED.with(|cell| cell.get())
    }

    /// Counts one rank's program lowering.
    pub(super) fn lowered() {
        #[cfg(test)]
        LOWERED.with(|cell| cell.set(cell.get() + 1));
    }

    /// The graph cache and bucket of an overlapped run: its drafter pass
    /// (none for the target's) and its (bucket, items).
    pub(crate) type RunKey = (Option<crate::program::DraftPass>, (u32, u32));

    /// One rank's overlapped capture: its rank, run key, units and launch
    /// stages.
    #[cfg(test)]
    #[derive(Clone, Debug)]
    pub(crate) struct Overlapped {
        pub rank: usize,
        pub key: RunKey,
        pub units: Vec<Vec<crate::shared_overlap::Op>>,
        pub stages: Vec<crate::program::Stage>,
    }

    #[cfg(test)]
    thread_local! {
        static OVERLAPPED: std::cell::RefCell<Vec<Overlapped>> =
            const { std::cell::RefCell::new(Vec::new()) };
        static REPLAYED: std::cell::RefCell<Vec<RunKey>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Every overlapped capture on this thread since the last call, in
    /// capture order.
    #[cfg(test)]
    pub(crate) fn overlapped_captures() -> Vec<Overlapped> {
        OVERLAPPED.with(|cell| cell.take())
    }

    /// The key of every overlapped run that reached its replay on this
    /// thread since the last call, in order.
    #[cfg(test)]
    pub(crate) fn overlapped_replays() -> Vec<RunKey> {
        REPLAYED.with(|cell| cell.take())
    }

    /// Notes the units of `rank`'s overlapped capture of `stages` for `key`.
    #[cfg_attr(not(test), allow(unused_variables))]
    pub(super) fn overlapped(
        rank: usize,
        key: RunKey,
        units: &[Vec<crate::shared_overlap::Op>],
        stages: &[crate::program::Stage],
    ) {
        #[cfg(test)]
        OVERLAPPED.with(|cell| {
            cell.borrow_mut().push(Overlapped {
                rank,
                key,
                units: units.to_vec(),
                stages: stages.to_vec(),
            })
        });
    }

    /// Notes an overlapped run of `key` about to replay.
    #[cfg_attr(not(test), allow(unused_variables))]
    pub(super) fn overlapped_replay(key: RunKey) {
        #[cfg(test)]
        REPLAYED.with(|cell| cell.borrow_mut().push(key));
    }

    #[cfg_attr(not(test), allow(unused_variables))]
    pub(super) fn submitted(rank: usize) {
        #[cfg(test)]
        if PANIC_AFTER.with(|cell| cell.get()) == Some(rank) {
            PANIC_AFTER.with(|cell| cell.set(None));
            panic!("a test panic after rank {rank} queued its share");
        }
    }
}
