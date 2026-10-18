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
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::serving::{DeepSeek, MAX_VERIFY_ROWS, SCORED_ROWS};
use runtime_bundle_abi::{
    BUNDLE_LOGITS_ELEMENT_V1_F32, BundleAbiError, BundleStatusV1, BundleStreamHandleV1,
    ExecuteResultSinkV1,
};
use runtime_bundle_sdk::BatchCodecV1;

use super::Instance;
use super::media::PrefillMedia;
use crate::engram_rows::Hashed;
use crate::lower::{KvExtent, Lowering, TailUpload, tail_upload};
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
    /// Each rank's stage of each launch, in program order.
    #[cfg_attr(not(feature = "diagnostic-taps"), allow(dead_code))]
    stages: Vec<Vec<Stage>>,
    /// The phase the programs run.
    #[cfg_attr(not(feature = "diagnostic-taps"), allow(dead_code))]
    phase: Phase,
    /// Each rank's compressor tail entries, uploaded with the inputs.
    tails: Vec<TailUpload>,
    /// The window write's inputs, uploaded to every rank.
    window: WindowUpload,
    /// The selections' per-row inputs, uploaded to every rank.
    index: IndexUpload,
    /// The Engram hash ids of the rows, whose table rows the host gathers
    /// on every rank.
    engram: Option<Hashed>,
    /// A prefill's image and text flags, when its rows read the encoder's
    /// arena (`super::media`).
    media: Option<PrefillMedia>,
}

/// What a step's index selections read besides their rows: each row's tag,
/// unique to the step and row; each row's visible compressed count at the
/// candidate source; and each row block's live count, the dense blocks'
/// then the compact blocks'.
#[derive(Debug, Default, PartialEq)]
struct IndexUpload {
    tags: Vec<u64>,
    visible: Vec<i32>,
    live: Vec<u32>,
}

/// The selections' inputs of step `step`'s `inputs` in an instance of `g`.
fn index_upload(g: &Geometry, step: u64, inputs: &PhaseInputs) -> IndexUpload {
    use crate::ops::indexer::{FULL_BLOCK, REINDEX_BLOCK};
    let rows = inputs.positions.len();
    let ratio = g.compress_ratio(g.shape.candidates.source).max(1);
    let live = inputs.live_tokens();
    let mut upload = IndexUpload {
        tags: (0..rows as u64).map(|row| (step << 32) | row).collect(),
        visible: inputs
            .positions
            .iter()
            .map(|&position| (position + 1) / ratio as i32)
            .collect(),
        live: Vec::new(),
    };
    for size in [FULL_BLOCK, REINDEX_BLOCK] {
        upload.live.extend(
            (0..crate::buffers::blocks(size))
                .map(|block| live.saturating_sub(block * size).min(size)),
        );
    }
    upload
}

/// What a step's window write reads besides its rows: each live row's
/// state slot, and each slot's last position in the step (-1 for a slot
/// without rows). A prefill's staging also reads each item's last chunk
/// position and staged block start (`window - 1` positions before its
/// first, at least 0), the items themselves, and each slot's last ring
/// position before the step: its item's first position less one, -1 for a
/// slot without an item.
#[derive(Debug, Default, PartialEq)]
struct WindowUpload {
    rows: Vec<i32>,
    last: Vec<i32>,
    stage_last: Vec<i32>,
    stage_base: Vec<i32>,
    stage_items: Vec<i32>,
    ring_last: Vec<i32>,
}

/// The window write inputs of `inputs` over `slots` state slots, for a
/// window of `window` positions.
fn window_upload(inputs: &PhaseInputs, slots: u32, window: u32) -> Result<WindowUpload, String> {
    let mut upload = WindowUpload {
        rows: Vec::with_capacity(inputs.token_batch.len()),
        last: vec![-1; slots as usize],
        ring_last: vec![-1; slots as usize],
        ..WindowUpload::default()
    };
    for &item in &inputs.token_batch {
        let slot = usize::try_from(item)
            .ok()
            .and_then(|item| inputs.slot_ids.get(item))
            .copied()
            .filter(|&slot| u32::try_from(slot).is_ok_and(|slot| slot < slots))
            .ok_or_else(|| format!("a row's item {item} has no state slot"))?;
        upload.rows.push(slot);
    }
    for (item, &slot) in inputs.slot_ids.iter().enumerate() {
        let first = inputs.prefix_len[item];
        let rows = inputs.q_len.get(item).copied().unwrap_or(1);
        let last = first + rows - 1;
        upload.stage_last.push(last);
        upload.stage_base.push((first - window as i32 + 1).max(0));
        upload.stage_items.push(item as i32);
        let Ok(index) = usize::try_from(slot) else {
            continue;
        };
        if let Some(entry) = upload.last.get_mut(index) {
            *entry = last;
        }
        if let Some(entry) = upload.ring_last.get_mut(index) {
            *entry = first - 1;
        }
    }
    Ok(upload)
}

/// A phase, its arena index, its input layout and its graph key.
type PhasePlan = (Phase, usize, InputLayout, Option<(u32, u32)>);

/// Each rank's launches, whether each waits for its peers, and the stage
/// of each.
type Programs = (Vec<Vec<PreparedLaunch>>, Vec<Vec<bool>>, Vec<Vec<Stage>>);

/// `values` as little-endian bytes.
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

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
    /// state slots. Media waits for an instance serving the encoder, bundle
    /// drafts for the DSpark drafter.
    fn pack(&self, request: &[u8]) -> Result<PackedStep, BundleStatusV1> {
        let batch = BatchCodecV1::decode(request).map_err(|_| BundleStatusV1::InvalidArgument)?;
        if !batch.media().is_empty() && self.media.is_none() {
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
        if self.dspark.is_none() && step.bundle_drafts(&batch).iter().any(|&drafts| drafts > 0) {
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
    /// programs. The head scores `head_rows` rows.
    fn program(
        &self,
        phase: Phase,
        arena: usize,
        layout: &InputLayout,
        lower: &[bool],
        head_rows: u32,
        images: bool,
    ) -> Result<Programs, BundleStatusV1> {
        let stages = crate::program::stages(self.config.shape.geometry, phase);
        let mut programs = Vec::new();
        let mut collectives = Vec::new();
        let mut stage_lists = Vec::new();
        for (rank, buffers) in self.buffers.iter().enumerate() {
            if !lower.get(rank).copied().unwrap_or(true) {
                programs.push(Vec::new());
                collectives.push(Vec::new());
                stage_lists.push(Vec::new());
                continue;
            }
            let inputs = DeviceInputs::within(self.arenas[arena][rank], layout);
            let lowering = Lowering {
                shape: &self.config.shape,
                ops: &self.ops,
                buffers,
                peers: &self.buffers,
                inputs: &inputs,
                rank: rank as u32,
                kv: self.kv_extent(),
                head_rows,
            };
            let mut launches = Vec::new();
            let mut waits = Vec::new();
            let mut of = Vec::new();
            for &stage in &stages {
                if self.dspark.is_some() && stage == Stage::HeadCollapse {
                    for lowered in self.dspark_pack_features(rank, phase).map_err(|error| {
                        self.log(&error);
                        BundleStatusV1::InternalError
                    })? {
                        waits.push(lowered.collective);
                        launches.push(lowered.launch);
                        of.push(stage);
                    }
                }
                // A prefill of image rows closes their Engram gates.
                let lowered = match (self.dspark.is_some(), images, stage) {
                    (true, _, Stage::DsparkFeature(layer)) => {
                        self.dspark_feature(&lowering, phase, layer).map(Some)
                    }
                    (true, _, Stage::EngramRows(layer)) => {
                        self.dspark_engram(&lowering, phase, layer)
                    }
                    (_, true, Stage::EngramGate(layer)) => self
                        .text_flags(rank)
                        .and_then(|text| lowering.engram_gate_masked(phase, layer, text)),
                    _ => lowering.stage(phase, stage),
                };
                let lowered = match lowered {
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
                    of.push(stage);
                }
                // The image rows' embedding from the encoder's arena.
                if images && stage == Stage::EmbeddingReduce {
                    let inject = self
                        .inject(
                            rank,
                            crate::buffers::rows(buffers.scratch.embedded, phase.rows()),
                            inputs.live_tokens,
                            phase.rows(),
                        )
                        .map_err(|error| {
                            self.log(&error);
                            BundleStatusV1::InternalError
                        })?;
                    waits.push(false);
                    launches.push(inject);
                    of.push(stage);
                }
            }
            programs.push(launches);
            collectives.push(waits);
            stage_lists.push(of);
        }
        Ok((programs, collectives, stage_lists))
    }

    /// Plans one phase of the step: every rank's program, a prefill's over
    /// the image rows `media` flags.
    fn plan_run(
        &self,
        inputs: PhaseInputs,
        media: Option<PrefillMedia>,
    ) -> Result<Run, BundleStatusV1> {
        let (phase, arena, layout, graph) = self.phase_of(&inputs)?;
        // Feature and Engram staging offsets depend on the preceding
        // prefill's live rows. Keep this first adapter eager.
        let graph = if self.dspark.is_some() { None } else { graph };
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
        if media.is_some() && arena != PREFILL {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let (programs, collectives, stages) =
            self.program(phase, arena, &layout, &lower, head_rows, media.is_some())?;
        let mut tails = Vec::new();
        for buffers in &self.buffers {
            let upload = tail_upload(
                self.config.shape.geometry,
                self.extent,
                &inputs,
                phase,
                &buffers.state,
            )
            .map_err(|error| {
                self.log(&error);
                BundleStatusV1::InvalidArgument
            })?;
            tails.extend(upload);
        }
        let window = self.config.shape.geometry.shape.attention.window;
        let window = window_upload(&inputs, self.extent.bucket, window).map_err(|error| {
            self.log(&error);
            BundleStatusV1::InvalidArgument
        })?;
        let engram = match self.engram.as_ref().filter(|_| self.dspark.is_none()) {
            Some(engram) => Some(engram.hash(&inputs).map_err(|error| {
                self.log(&format!("Engram rows: {error}"));
                BundleStatusV1::InvalidArgument
            })?),
            None => None,
        };
        let index = index_upload(self.config.shape.geometry, self.steps.get(), &inputs);
        self.steps.set(self.steps.get() + 1);
        Ok(Run {
            index,
            engram,
            media,
            window,
            inputs,
            arena,
            layout,
            head_rows,
            graph,
            programs,
            collectives,
            stages,
            tails,
            phase,
        })
    }

    /// The serving pages and request state the stages address.
    pub(super) fn kv_extent(&self) -> KvExtent {
        let limits = super::model_limits(&self.config);
        KvExtent {
            storage: self.config.kv.storage,
            pages: u32::try_from(self.kv_pages).unwrap_or(u32::MAX),
            extent: self.extent,
            sequences: self.items().get(),
            max_pages: limits.max_pages(),
        }
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
                let scratch = &self.buffers[rank].scratch;
                // The quantizers and folds OR flags into each row's status
                // and keep a flagged row zero: clear every planned row once
                // per phase, before any of them launches.
                self.services
                    .zero_device_buffer(stream, scratch.row_status)?;
                let index = &run.index;
                // Only a prefill stages its window.
                let staging = |values: &[i32]| match run.arena {
                    PREFILL => i32_bytes(values),
                    _ => Vec::new(),
                };
                for (span, bytes) in [
                    (scratch.window_rows, i32_bytes(&run.window.rows)),
                    (scratch.window_last, i32_bytes(&run.window.last)),
                    (scratch.index_visible, i32_bytes(&index.visible)),
                    (scratch.stage_last, staging(&run.window.stage_last)),
                    (scratch.stage_base, staging(&run.window.stage_base)),
                    (scratch.stage_items, staging(&run.window.stage_items)),
                    (scratch.ring_last, staging(&run.window.ring_last)),
                    (
                        scratch.row_tags,
                        index
                            .tags
                            .iter()
                            .flat_map(|tag| tag.to_le_bytes())
                            .collect(),
                    ),
                    (
                        scratch.block_live,
                        index
                            .live
                            .iter()
                            .flat_map(|live| live.to_le_bytes())
                            .collect(),
                    ),
                ] {
                    if !bytes.is_empty() {
                        upload_bytes(&self.services, stream, span, &bytes)?;
                    }
                }
                if let (Some(engram), Some(hashed), Some(image)) =
                    (&self.engram, &run.engram, self.image)
                {
                    engram
                        .gather(
                            &self.services,
                            stream,
                            image,
                            rank as u32,
                            &self.config.shape,
                            scratch,
                            hashed,
                        )
                        .map_err(|error| match error {
                            crate::engram_rows::Error::Host(error) => error,
                            crate::engram_rows::Error::Refused(reason) => {
                                self.log(&format!("Engram rows: {reason}"));
                                BundleAbiError::InvalidRange("deepseek41 Engram staging")
                            }
                        })?;
                }
                if let Some(media) = &run.media {
                    self.upload_flags(rank, media)?;
                }
                if let Some(tails) = run.tails.get(rank) {
                    let scratch = &self.buffers[rank].scratch;
                    for (span, ids) in [
                        (scratch.tail_ids, &tails.ids),
                        (scratch.tail_checkpoints, &tails.checkpoints),
                    ] {
                        if !ids.is_empty() {
                            upload_bytes(&self.services, stream, span, &i32_bytes(ids))?;
                        }
                    }
                    for &(destination, source) in &tails.seeds {
                        self.services
                            .copy_device_to_device(stream, destination, source)?;
                    }
                }
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
        #[cfg(feature = "diagnostic-taps")]
        if self.taps.text() && run.arena == PREFILL {
            return self.launch_tapped(&streams, run, &programs, &collectives);
        }
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

    /// Refuses `run`'s phase when a quantizer or fold flagged a live row:
    /// reads each rank's row status of the live rows back once the phase's
    /// work is done. Each rank quantizes and folds its own slices, so every
    /// rank is read.
    fn check_row_status(&self, run: &Run) -> Result<(), BundleAbiError> {
        let rows = run.inputs.live_tokens();
        if rows == 0 {
            return Ok(());
        }
        let mut status = vec![0_u8; rows as usize * 4];
        for (rank, (&(_, _, stream), buffers)) in self.ranks.iter().zip(&self.buffers).enumerate() {
            let span = crate::buffers::rows(buffers.scratch.row_status, rows);
            self.services.synchronize_stream(stream)?;
            self.services
                .copy_device_to_host(stream, &mut status, span)?;
            self.services.synchronize_stream(stream)?;
            let failed = status.chunks_exact(4).position(|word| word != [0; 4]);
            if let Some(row) = failed {
                let item = run.inputs.token_batch.get(row).copied().unwrap_or(-1);
                let request = usize::try_from(item)
                    .ok()
                    .and_then(|item| run.inputs.request_ids.get(item));
                self.log(&format!(
                    "a quantized input on rank {rank} was invalid in row {row} \
                     (request {request:?}); the step is refused"
                ));
                return Err(BundleAbiError::Status(BundleStatusV1::BundleError));
            }
        }
        Ok(())
    }

    /// Refuses `run`'s phase when an index selection met a non-finite score
    /// in a live row: reads every index source's status of the live rows
    /// back from rank 0 once the phase's work is done. The selections are
    /// replicated, so rank 0's status is every rank's.
    fn check_index_status(&self, run: &Run) -> Result<(), BundleAbiError> {
        let rows = run.inputs.live_tokens();
        if rows == 0 {
            return Ok(());
        }
        let (Some(&(_, _, stream)), Some(buffers)) = (self.ranks.first(), self.buffers.first())
        else {
            return Ok(());
        };
        let sources = self.config.shape.geometry.shape.csa.index_sources;
        let mut status = vec![0_u8; rows as usize * 4];
        self.services.synchronize_stream(stream)?;
        for (source, &layer) in sources.iter().enumerate() {
            let span = crate::buffers::index_status(buffers.scratch.index_status, source, rows);
            self.services
                .copy_device_to_host(stream, &mut status, span)?;
            self.services.synchronize_stream(stream)?;
            let failed = status.chunks_exact(4).position(|word| word != [0; 4]);
            if let Some(row) = failed {
                let item = run.inputs.token_batch.get(row).copied().unwrap_or(-1);
                let request = usize::try_from(item)
                    .ok()
                    .and_then(|item| run.inputs.request_ids.get(item));
                self.log(&format!(
                    "the index selection of layer {layer} met a non-finite score in row {row} \
                     (request {request:?}); the step is refused"
                ));
                return Err(BundleAbiError::Status(BundleStatusV1::BundleError));
            }
        }
        Ok(())
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
    pub(super) fn launch_eager(
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
        // Taps a failed execute left are removed, never published.
        #[cfg(feature = "diagnostic-taps")]
        if let Err(error) = self.taps.abandon() {
            self.log(&format!("taps: {error}"));
            return BundleStatusV1::InternalError;
        }
        let step = match self.pack(request) {
            Ok(step) => step,
            Err(error) => return error,
        };
        if self.dspark.is_some() {
            let mut rows = items(&step.prefill);
            rows.extend(items(&step.decode));
            if self.ledger.preview(&rows).is_err() {
                return BundleStatusV1::InvalidArgument;
            }
            return self.execute_dspark(request, sink, step);
        }
        self.execute_packed(request, sink, step)
    }

    pub(super) fn execute_packed(
        &mut self,
        request: &[u8],
        sink: &mut ExecuteResultSinkV1,
        step: PackedStep,
    ) -> BundleStatusV1 {
        let mut items = items(&step.prefill);
        items.extend(self::items(&step.decode));
        // The encoder runs before the step opens: a failure leaves no step
        // to cancel, and the prefill's injects read what it wrote.
        let media = match self.encode(request, &step.prefill) {
            Ok(media) => media,
            Err(error) => {
                if let Some(media) = &self.media
                    && let Err(taps) = self.taps.discard(&media.encoder)
                {
                    self.log(&format!("encoder taps: {taps}"));
                }
                return error;
            }
        };
        // A diagnostic build writes the step's encoder rows (`crate::taps`).
        if let Some(encoder) = self.media.as_ref().map(|media| &media.encoder)
            && let Err(error) = self.taps.step(
                &self.services,
                self.ranks[0].2,
                encoder,
                request,
                &step.prefill,
            )
        {
            self.log(&format!("encoder taps: {error}"));
            return BundleStatusV1::InternalError;
        }
        let mut runs = Vec::new();
        for (inputs, media) in [(step.prefill, media), (step.decode, None)] {
            if inputs.request_ids.is_empty() {
                continue;
            }
            match self.plan_run(inputs, media) {
                Ok(run) => runs.push(run),
                Err(error) => return error,
            }
        }
        if self.ledger.preview(&items).is_err() || self.ledger.begin(&items, Vec::new()).is_err() {
            return BundleStatusV1::InvalidArgument;
        }
        if let Some(engram) = self.engram.as_mut().filter(|_| self.dspark.is_none()) {
            engram.begin(runs.iter().map(|run| &run.inputs));
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
            if let Some(engram) = self.engram.as_mut().filter(|_| self.dspark.is_none()) {
                engram.cancelled(&self.ledger);
            }
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
            self.dspark_pack_counts(run.phase, run.inputs.live_tokens())?;
            self.launch(run)?;
            if run.media.is_some() {
                self.clear_flags()?;
            }
            self.check_index_status(run)?;
            self.check_row_status(run)?;
        }
        self.emit_logits(runs, sink)
    }
}

#[cfg(feature = "diagnostic-taps")]
impl Instance {
    /// Issues an eager prefill stage by stage across the ranks, each
    /// stage's launches collective by collective as `launch_eager` queues
    /// a program, and copies the stage's selected taps once its work is
    /// done (`crate::taps`).
    fn launch_tapped(
        &mut self,
        streams: &[BundleStreamHandleV1],
        run: &Run,
        programs: &[Vec<PreparedLaunch>],
        collectives: &[Vec<bool>],
    ) -> Result<(), BundleAbiError> {
        let refused = |instance: &Self, error: DispatchError| instance.dispatch_failed(error);
        if self.collectives.is_poisoned() {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
        }
        let stages = &run.stages;
        if programs.len() != streams.len()
            || stages.len() != streams.len()
            || self.symbols.len() != streams.len()
            || programs.iter().zip(stages).any(|(p, s)| p.len() != s.len())
        {
            return Err(refused(self, DispatchError::InvalidPlan));
        }
        let shape = self.config.shape;
        let order = crate::program::stages(shape.geometry, run.phase);
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
        let ranks = streams.len() as u32;
        if let Err(error) = self
            .taps
            .begin("Prefill", ranks, &run.inputs, &shape, &order)
        {
            self.log(&format!("taps: {error}"));
            return Err(BundleAbiError::Status(BundleStatusV1::InternalError));
        }
        let _unwinding = PoisonOnUnwind(&self.collectives);
        let mut unit = Unit::default();
        let mut started = vec![false; streams.len()];
        for (&stage, ranges) in order.iter().zip(&walk) {
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
                    pipeline::launch_all(
                        &self.services,
                        streams[rank],
                        &self.symbols[rank],
                        launches,
                    )
                },
                |_| {},
                |_| Ok(()),
            )
            .map_err(|error| refused(self, error))?;
            let captured = self
                .taps
                .capture(&self.services, streams, &self.buffers, &shape, stage);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::FLASH;

    /// Two items: a three-row prefill on slot 2 from position 5, and a
    /// one-row item on slot 0 at position 9.
    fn inputs() -> PhaseInputs {
        PhaseInputs {
            positions: vec![5, 6, 7, 9],
            token_batch: vec![0, 0, 0, 1],
            slot_ids: vec![2, 0],
            prefix_len: vec![5, 9],
            q_len: vec![3, 1],
            ..PhaseInputs::default()
        }
    }

    /// Each row names its item's slot; each slot of a row its item's last
    /// position, and every other slot -1.
    #[test]
    fn the_window_write_addresses_each_rows_slot() {
        let upload = window_upload(&inputs(), 4, 4).unwrap();
        assert_eq!(upload.rows, [2, 2, 2, 0]);
        assert_eq!(upload.last, [9, -1, 7, -1]);
        assert!(window_upload(&inputs(), 2, 4).is_err());
        // The staging: each item's last chunk position, its staged block
        // from `window - 1` positions back, and each slot's ring snapshot.
        assert_eq!(upload.stage_last, [7, 9]);
        assert_eq!(upload.stage_base, [2, 6]);
        assert_eq!(upload.stage_items, [0, 1]);
        assert_eq!(upload.ring_last, [8, -1, 4, -1]);
        let upload = window_upload(&inputs(), 4, 128).unwrap();
        assert_eq!(upload.stage_base, [0, 0]);
    }

    /// Tags differ between the rows of a step and between steps; each
    /// block's live count is its share of the live rows.
    #[test]
    fn index_tags_are_unique_and_blocks_count_their_live_rows() {
        let mut inputs = inputs();
        inputs.positions = (0..130).collect();
        inputs.token_batch = vec![0; 130];
        inputs.token_ids = vec![1; 130];
        inputs.slot_ids = vec![0];
        inputs.prefix_len = vec![0];
        inputs.q_len = vec![130];
        let first = index_upload(&FLASH, 1, &inputs);
        let second = index_upload(&FLASH, 2, &inputs);
        let mut tags = first.tags.clone();
        tags.extend(&second.tags);
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(tags.len(), 260);
        // Layer 20 compresses each position, so row r sees r + 1.
        assert_eq!(first.visible[..3], [1, 2, 3]);
        let dense = crate::buffers::blocks(crate::ops::indexer::FULL_BLOCK) as usize;
        assert_eq!(first.live[..4], [64, 64, 2, 0]);
        assert_eq!(first.live[dense..], [130, 0]);
    }
}
