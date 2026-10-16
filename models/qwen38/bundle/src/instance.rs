//! Host resource acquisition, the step driver and the ABI entry bodies.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread::LocalKey;

use bundle_runtime::acquire::{self, Log};
use bundle_runtime::entry::{Bundle, Failed};
use bundle_runtime::kv::KvFormat;
use bundle_runtime::resources::{Held, allocate_span, status, upload_bytes};
use bundle_runtime::table::InstanceTable;
use kernel_core::contract::Refusal;
use model_qwen38_contract::own_ops::copies::CopyEntriesBuffers;
#[cfg(test)]
use runtime_bundle_abi::BundleHandleV1;
use runtime_bundle_abi::{
    BundleAbiError, BundleContextHandleV1, BundleCudaSymbolHandleV1, BundleEventHandleV1,
    BundleStatusV1, BundleStreamHandleV1, DeviceBufferSpanV1, ExecuteResultSinkV1, HostServicesV1,
};
use runtime_bundle_sdk::BatchCodecV1;

use crate::CUBINS;
use crate::contract::*;
use crate::graphs::GraphCache;
use crate::inputs::{PackedStep, PhaseInputs, VerifyLimits};
use crate::launch;
use crate::lower::{ItemRows, RankStage};
use crate::pipeline::PreparedLaunch;
use crate::program::{self, Phase};
use crate::scratch::{RankScratch, Served, Speculation};
use crate::side::{ItemEntries, SideInputs, SideLayout};
use crate::state::Open;
use crate::target::Target;
use crate::upload::{BlockTableMirror, DeviceInputs, InputLayout, PhaseArena};
use bundle_runtime::geometry::Items;
use bundle_runtime::ledger::{Ledger, LedgerError, Placement, Rules};
use bundle_runtime::transaction::{self, End, Ended, Failure, Retire, Stage, Steps};

pub(crate) mod hc;
mod logits;
mod switches;
#[cfg(test)]
pub(crate) use hc::{with_hc_padded_mix, with_hc_precompute};
pub(crate) use switches::Switches;
#[path = "prefix_state.rs"]
mod prefix_state;
mod speculation;
#[cfg(test)]
pub(crate) use logits::{FOR_TEST as ROW_OWNERS_FOR_TEST, with_row_owners};

const PREFILL: usize = 0;
const DECODE: usize = 1;

const HALVES: [usize; 2] = [2, 3];

struct Overlap {
    config: bundle_runtime::lanes::Config,
    streams: Vec<BundleStreamHandleV1>,
    events: Vec<Vec<BundleEventHandleV1>>,
    histories: Vec<DeviceBufferSpanV1>,
}

struct Configuration {
    /// The package topology's per-rank sizes.
    shape: Shape,
    /// One device ordinal per rank.
    ordinals: Vec<i32>,
    layout: Layout,
    /// The package names the vision encoder's image and indexes.
    media: bool,
    precision: bool,
    /// The sequences the instance serves (the host's `--max-sequences`).
    items: Items,
    /// The environment switches, read once here.
    switches: Switches,
    /// The GPU target the instance runs on.
    target: Arc<dyn Target>,
    /// The pipelined prefill's lanes, if any (`Target::lanes`), read with
    /// the switches.
    overlap: Option<bundle_runtime::lanes::Config>,
    /// The serving process's startup nonce, which tap steps record.
    tap_producer: Option<String>,
    /// The context-parallel group size: 1 unless the create record names
    /// another (`chop.context-parallel.v1`, sent only above 1).
    context_parallel: u32,
}

/// The pages and ring a package declared: the main full layers, or those
/// and the MTP drafter's layer, in BF16 or with FP8 main layers.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    kv: KvFormat,
    /// Full-attention layers of a page and of the ring.
    kv_layers: u64,
    drafter: bool,
}

impl Layout {
    /// The layout a package declares for its geometry.
    fn of(shape: Shape, layout_id: &[u8]) -> Option<Self> {
        shape.kv_formats().into_iter().find_map(|kv| {
            let drafter = model_qwen38_contract::state::has_mtp_layer(&kv);
            (layout_id == kv.layout_id).then_some(Self {
                kv,
                drafter,
                kv_layers: if drafter {
                    shape.full_layers_mtp()
                } else {
                    shape.full_layers()
                },
            })
        })
    }

    fn drafter(self) -> bool {
        self.drafter
    }

    /// The ring of this layout's full layers for `items` served items: one
    /// slot per item of the largest graph bucket, the state slots the host
    /// allocates (`chop.state-per-item.v1`), or with the drafter the 32 slots
    /// its package declares.
    const fn ring(self, shape: Shape, items: Items) -> Ring {
        Ring {
            shape,
            layers: self.kv_layers,
            slots: items.capacity() as u64,
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only minimum row count for the overlap path.
    pub(crate) static FORCE_OVERLAP_MIN: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
    /// Test-only overlap selection, independent of the process environment.
    pub(crate) static FORCE_OVERLAP: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// Tests select eager execution or graph replay without changing the environment.
    pub(crate) static FORCE_EAGER: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// Every launch the stages lower on this thread: its rank, symbol and
    /// the rows its plan processes (`PreparedLaunch::rows`), for the
    /// invariance journal.
    pub(crate) static PLANNED: std::cell::RefCell<Vec<(usize, String, u32)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether tests force eager execution.
pub(crate) fn eager() -> bool {
    #[cfg(test)]
    if let Some(eager) = FORCE_EAGER.with(std::cell::Cell::get) {
        return eager;
    }
    false
}

/// The GPU target of an instance of `shape`: SM120's, whose own kernels
/// read the padded up mix layout when `padded_mix`.
pub(crate) fn target(shape: Shape, padded_mix: bool) -> Arc<dyn Target> {
    Arc::new(crate::sm120::own_ops::Sm120Own::new(shape, padded_mix))
}

fn hc_available(shape: Shape) -> bool {
    crate::GEOMETRY_HC_PRECOMPUTE
        .iter()
        .any(|(id, available)| *id == shape.geometry.id && *available)
}

fn decode_configuration(bytes: &[u8], log: impl Fn(&str)) -> Result<Configuration, BundleStatusV1> {
    let mut reader = runtime_bundle_sdk::BatchReaderV1::new("bundle configuration", bytes);
    reader
        .expect_magic(bundle_runtime::config::MAGIC)
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    let length = reader
        .count("topology identifier")
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    let id = reader
        .take(length)
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    let id = std::str::from_utf8(id).map_err(|_| BundleStatusV1::InvalidArgument)?;
    let shape = Shape::from_topology(id).ok_or(BundleStatusV1::Unsupported)?;
    let config = bundle_runtime::config::decode_with_context_parallel::<Qwen>(
        bytes,
        &[shape.topology()],
        model_qwen38_contract::context_parallel::sizes(shape),
        log,
    )?;
    let context_parallel = config.context_parallel.size();
    config.validate_context(shape.model_limits())?;
    let ranks = shape.ranks();
    let has_image = config.roles.iter().any(|role| role == RANK_IMAGE_ROLE);
    let has_indexes = (0..ranks).all(|rank| {
        config
            .roles
            .iter()
            .any(|role| *role == index_role(rank).as_bytes())
    });
    let media =
        crate::encoder::configured(&config.roles, ranks).ok_or(BundleStatusV1::Unsupported)?;
    let (kv, state, items) = (config.kv, config.state, config.items);
    let layout = Layout::of(shape, &kv.layout_id).ok_or(BundleStatusV1::Unsupported)?;
    if kv.page_bytes != layout.kv.page_bytes
        || kv.page_tokens != PAGE_TOKENS
        || kv.ranks != u64::from(ranks)
        || kv.lifetime != 0
        || kv.alignment < 256
        || !kv.alignment.is_power_of_two()
        || state.state_id != RING_STATE_ID
        || state.bytes != layout.ring(shape, items).state_bytes()
        || state.alignment < 256
        || !state.alignment.is_power_of_two()
        || !has_image
        || !has_indexes
    {
        return Err(BundleStatusV1::Unsupported);
    }
    // The most sequences a rank of this shape fits (MTP packages declare
    // fewer, the count their accuracy is qualified for).
    if items.get()
        > model_qwen38_contract::package::maximum_sequences(
            shape,
            false,
            model_qwen38_contract::package::Encoders { vision: media },
        )
    {
        return Err(BundleStatusV1::InvalidArgument);
    }
    if config.ordinals.len() != shape.ranks.count() {
        return Err(BundleStatusV1::InvalidArgument);
    }
    let switches = Switches::read(shape);
    let target = target(shape, switches.hc_padded_mix);
    // Prefill lanes would run two pieces' group collectives at once over
    // one group state, so a group of 2 or more prefills in one lane.
    let overlap = if context_parallel > 1 {
        None
    } else {
        target.lanes()
    };
    Ok(Configuration {
        shape,
        ordinals: config.ordinals,
        layout,
        media,
        precision: config
            .roles
            .iter()
            .any(|role| role == model_qwen38_contract::prepared::COMPONENT_ROLE.as_bytes()),
        items,
        switches,
        target,
        overlap,
        tap_producer: config.tap_producer,
        context_parallel,
    })
}

pub(crate) struct Instance {
    precision: Option<crate::precision::Prepared>,
    services: HostServicesV1,
    /// The package topology's per-rank sizes and kernel tag.
    shape: Shape,
    /// Tests can run decode and verify eagerly instead of through graphs.
    eager: bool,
    held: Vec<Held>,
    views: Vec<BTreeMap<String, DeviceBufferSpanV1>>,
    contexts: Vec<BundleContextHandleV1>,
    streams: Vec<BundleStreamHandleV1>,
    /// Per rank, the QSA raw-key ring: the live half, then the saved half.
    rings: Vec<DeviceBufferSpanV1>,
    kv_views: Vec<DeviceBufferSpanV1>,
    kv_page_capacity: u64,
    scratch: Vec<RankScratch>,
    exchange: Vec<crate::logits::ExchangeRowsState>,
    /// Per phase and rank, the shared input arena and its block table.
    arenas: [Vec<PhaseArena>; 4],
    /// Per phase and rank, the side input arena.
    side_arenas: [Vec<DeviceBufferSpanV1>; 4],
    block_tables: [BlockTableMirror; 4],
    overlap: Option<Overlap>,
    symbols: Vec<BTreeMap<String, BundleCudaSymbolHandleV1>>,
    graphs: GraphCache,
    /// The sequences the instance serves, and what its buffers hold for
    /// them.
    items: Items,
    served: Served,
    /// Decode graphs above 32 rows take the wide all-reduce up to its cap
    /// (unless `CHOPINFER_QWEN38_WIDE_COMM=0`).
    wide_comm: bool,
    hc_precompute: bool,
    hc_padded_mix: bool,
    /// The environment switches create read.
    switches: Switches,
    /// The GPU target the instance runs on, built with the switches.
    target: Arc<dyn Target>,
    /// Every op the instance launches through `kernel_core::backend`,
    /// prepared at create.
    ops: Option<crate::ops::Ops>,
    /// The state copies outside lowering, prepared at create.
    copies: Option<crate::ops::Copies>,
    /// Whether each norm's weights are F32, resolved with the ops.
    norm_dtypes: crate::lower::NormDtypes,
    /// Set once work reached some ranks but not their peers; see
    /// `pipeline::Collectives`.
    collectives: crate::pipeline::Collectives,
    /// The context-parallel groups, above size 1 only.
    context_parallel: Option<crate::context_parallel::ContextParallel>,
    /// Which request each host-named slot holds, and the open step.
    ledger: Ledger,
    linear_entries: crate::state::EntryMap,
    ple_entries: crate::state::EntryMap,
    /// What the open step saved and staged.
    open: Option<Open>,
    /// The pages and ring the package declared.
    layout: Layout,
    /// What the host enabled: verify rows and the MTP drafter.
    speculation: Speculation,
    verify_limits: VerifyLimits,
    drafter: Option<speculation::DrafterRuntime>,
    /// Per rank, index lists commit and staging copies upload.
    index_arenas: Vec<DeviceBufferSpanV1>,
    /// The open step's phase layouts, which commit reads again.
    step: Option<StepInputs>,
    /// Activation taps (`dump.rs`), when the environment asks for them.
    dump: Option<runtime_bundle_sdk::taps::Taps>,
    /// This step's encoder rows, read for the taps of its prefill.
    tap_encoders: Vec<runtime_bundle_sdk::taps::EncoderRows>,
    /// Each prompt's media occurrences across its prefill steps, for taps.
    tap_occurrences: bundle_runtime::media::Occurrences,
    /// Captures awaiting the transaction outcome.
    pending_taps: Vec<runtime_bundle_sdk::taps::TapStep>,
    /// The vision encoder, when the package ships it.
    encoder: Option<crate::encoder::Encoder>,
}

#[derive(Clone, Copy)]
struct PhaseDevice {
    inputs: DeviceInputs,
    side: SideInputs,
}

#[derive(Clone)]
struct StepInputs {
    /// Per phase with items, its per-rank device inputs.
    phases: [Option<Vec<PhaseDevice>>; 2],
    /// The decode phase ran verify rows: its PLE state is in checkpoints.
    verify: bool,
    /// Decode phase (slot, rows) per item, for the kept PLE checkpoints.
    decode_items: Vec<(u32, u32, u64)>,
    decode_entries: Vec<ItemEntries>,
}

/// The rows of one phase's items as the lowering sees them. A decode phase
/// runs its whole bucket, one row per item.
fn item_rows(phase: &PhaseInputs, program: Phase) -> Vec<ItemRows> {
    match program {
        Phase::Decode(bucket) => (0..bucket)
            .map(|start| ItemRows { start, rows: 1 })
            .collect(),
        // Verify rows start on the device; only the item bucket is fixed.
        Phase::Verify { items, .. } => (0..items).map(|_| ItemRows { start: 0, rows: 0 }).collect(),
        Phase::Prefill => phase
            .q_start
            .iter()
            .zip(&phase.q_len)
            .map(|(&start, &rows)| ItemRows {
                start: start as u32,
                rows: rows as u32,
            })
            .collect(),
    }
}

/// How one phase's rank programs are planned.
struct PhasePlan {
    program: Phase,
    rows: u32,
    items: Vec<ItemRows>,
    head_rows: u32,
    scored: bool,
    context: u32,
    /// A prefill whose rows take media arena rows.
    inject: bool,
    /// A prefill with an item that keeps its state at an anchor.
    anchored: bool,
}

impl PhasePlan {
    /// The plan of `phase` for an instance serving `served` items: decode and
    /// verify phases run the graph bucket of their items or rows.
    fn new(
        shape: Shape,
        phase: &PhaseInputs,
        served: Items,
    ) -> Result<Option<Self>, BundleStatusV1> {
        if phase.live_batch() == 0 {
            return Ok(None);
        }
        let invalid = BundleStatusV1::InvalidArgument;
        if phase.verify {
            let rows =
                bundle_runtime::graphs::verify_bucket_within::<Qwen>(phase.live_tokens(), served)
                    .ok_or(invalid)?;
            let program = Phase::Verify {
                rows,
                items: served.bucket(phase.live_batch()).ok_or(invalid)?,
            };
            return Ok(Some(Self {
                program,
                rows,
                items: item_rows(phase, program),
                head_rows: rows,
                scored: false,
                context: shape.context_tokens() as u32,
                inject: false,
                anchored: false,
            }));
        }
        if phase.decode {
            let bucket = served.bucket(phase.live_batch()).ok_or(invalid)?;
            let program = Phase::Decode(bucket);
            return Ok(Some(Self {
                program,
                rows: bucket,
                items: item_rows(phase, program),
                head_rows: bucket,
                scored: false,
                context: shape.context_tokens() as u32,
                inject: false,
                anchored: false,
            }));
        }
        Ok(Some(Self {
            program: Phase::Prefill,
            rows: phase.live_tokens(),
            items: item_rows(phase, Phase::Prefill),
            head_rows: phase.logit_rows().len() as u32,
            scored: phase.prompt_rows.iter().any(|&row| row),
            context: phase.seq_lens.iter().copied().max().unwrap_or(1) as u32,
            inject: false,
            anchored: phase.anchors.iter().any(|&anchor| anchor),
        }))
    }
}

impl Instance {
    /// The QSA raw-key ring's layout.
    fn ring(&self) -> Ring {
        self.layout.ring(self.shape, self.items)
    }

    /// Rank `rank`'s live ring; the saved copy follows it.
    fn live_ring(&self, rank: usize) -> DeviceBufferSpanV1 {
        let ring = self.rings[rank];
        DeviceBufferSpanV1::new(ring.buffer, ring.offset, self.ring().bytes())
    }

    /// The shared and side input layouts of one phase. Decode and verify
    /// graphs bind one layout, for the largest bucket of either; prefill
    /// packs around its live tokens.
    fn phase_layouts(&self, phase: &PhaseInputs) -> (InputLayout, SideLayout) {
        let served = self.served;
        if phase.decode {
            (
                InputLayout::verify_within::<Qwen>(self.items),
                SideLayout::for_shape(self.shape, served.rows, served.items),
            )
        } else {
            let layout =
                InputLayout::prefill_within::<Qwen>(phase, phase.logit_rows().len(), self.items);
            let side = SideLayout::for_shape(self.shape, layout.tokens as u32, served.items);
            (layout, side)
        }
    }

    fn rank_stage<'a>(
        &'a self,
        plan: &'a PhasePlan,
        device: PhaseDevice,
        rank: usize,
    ) -> RankStage<'a> {
        RankStage {
            precision: self.precision.as_ref(),
            norm_dtypes: &self.norm_dtypes,
            shape: self.shape,
            rank: rank as u32,
            phase: plan.program,
            rows: plan.rows,
            items: &plan.items,
            head_rows: plan.head_rows,
            scored: plan.scored,
            anchored: plan.anchored,
            context: plan.context,
            views: &self.views[rank],
            inputs: device.inputs,
            side: device.side,
            peers: &self.scratch,
            kv_pool: self.kv_views[rank],
            kv: self.layout.kv,
            ring: self.live_ring(rank),
            ring_layout: self.ring(),
            pool_entries: self.speculation.pool_entries(self.served.slots),
            wide_comm: self.wide_comm,
            hc_precompute: self.hc_precompute,
            hc_padded_mix: self.hc_padded_mix,
            drafter: None,
            exchange: (!self.exchange.is_empty()).then_some(&self.exchange),
            media: self
                .encoder
                .as_ref()
                .filter(|_| plan.inject)
                .map(|encoder| &encoder.ranks[rank]),
            context_parallel: self.context_parallel.as_ref(),
            ops: self.ops.as_ref(),
            target: self.target.as_ref(),
        }
    }

    fn lower(
        &self,
        plan: &PhasePlan,
        device: PhaseDevice,
        rank: usize,
        stage: program::Stage,
    ) -> Result<Vec<PreparedLaunch>, BundleStatusV1> {
        self.lower_with_facts(plan, device, rank, stage)
            .map(crate::ops::launches)
    }

    /// [`Instance::lower`], with each launch's facts for the pipelined
    /// prefill.
    fn lower_with_facts(
        &self,
        plan: &PhasePlan,
        device: PhaseDevice,
        rank: usize,
        stage: program::Stage,
    ) -> Result<Vec<crate::ops::Lowered>, BundleStatusV1> {
        let binding = self.rank_stage(plan, device, rank);
        let (lowered, facts) = match binding.lower_with_facts(stage, 0) {
            Ok(lowered) => {
                let (launches, facts) = crate::ops::unzip(lowered);
                (Ok(launches), facts)
            }
            Err(refusal) => (Err(refusal), Vec::new()),
        };
        #[cfg(test)]
        if let Ok(launches) = &lowered {
            PLANNED.with(|planned| {
                planned.borrow_mut().extend(
                    launches
                        .iter()
                        .map(|launch| (rank, launch.symbol.clone(), launch.rows)),
                )
            });
        }
        lowered
            .map_err(|Refusal(reason)| {
                let _ = self.services.log(
                    format!(
                        "Qwen3.8 rank {rank} {:?} stage {stage:?}: {reason}",
                        plan.program
                    )
                    .as_bytes(),
                );
                BundleStatusV1::Unsupported
            })
            .map(|launches| crate::ops::zip(launches, facts))
    }

    /// Every launch of one rank's program for a phase.
    fn plan_rank(
        &self,
        plan: &PhasePlan,
        device: PhaseDevice,
        rank: usize,
    ) -> Result<crate::pipeline::Program, BundleStatusV1> {
        let mut lowered = Vec::new();
        for stage in program::stages_for(self.shape, plan.program) {
            lowered.extend(self.lower_with_facts(plan, device, rank, stage)?);
        }
        Ok(crate::ops::program(lowered))
    }

    /// Eager prefill, one stage at a time across the ranks, so the ranks
    /// advance together and the device starts while the host lowers the rest.
    /// Decode runs this way too while dumping.
    fn run_eager(
        &mut self,
        plan: &PhasePlan,
        devices: &[PhaseDevice],
        inputs: &PhaseInputs,
    ) -> Result<(), BundleStatusV1> {
        let dump_dir = match self
            .dump
            .as_mut()
            .map(runtime_bundle_sdk::taps::Taps::next_step)
        {
            Some(Ok(dir)) => Some(dir),
            Some(Err(_)) => return Err(BundleStatusV1::InternalError),
            None => None,
        };
        let rows = match plan.program {
            Phase::Decode(_) => plan.items.len() as u32,
            _ => plan.rows,
        };
        let encoders = std::mem::take(&mut self.tap_encoders);
        let mut capture = match (&self.dump, &dump_dir) {
            (Some(dump), Some(dir)) => Some(
                crate::dump::begin(dump, dir, plan.program, rows, inputs, self.shape, encoders)
                    .map_err(|_| BundleStatusV1::InternalError)?,
            ),
            _ => None,
        };
        for stage in program::stages_for(self.shape, plan.program) {
            // Every rank is lowered before any is submitted.
            let lowered = devices
                .iter()
                .enumerate()
                .map(|(rank, &device)| self.lower(plan, device, rank, stage))
                .collect::<Result<Vec<_>, _>>()?;
            let mut unit = crate::pipeline::Unit::default();
            for (rank, launches) in lowered.iter().enumerate() {
                if launches.is_empty() {
                    continue;
                }
                unit.record(
                    &self.collectives,
                    crate::pipeline::launch_all(
                        &self.services,
                        self.streams[rank],
                        &self.symbols[rank],
                        launches,
                    ),
                )
                .map_err(crate::dispatch::status)?;
            }
            if let (Some(dump), Some(capture)) = (&self.dump, &mut capture) {
                self.dump_stage(dump, capture, stage)?;
            }
        }
        if let Some(capture) = capture {
            self.pending_taps.push(capture);
        }
        Ok(())
    }

    fn run_overlap(
        &mut self,
        plan: &PhasePlan,
        devices: &[PhaseDevice],
        whole: &PhaseInputs,
        entries: &[ItemEntries],
        at: u32,
    ) -> Result<bool, BundleStatusV1> {
        use bundle_runtime::lanes::{Executor, Lane, Mark};
        let ends = entries
            .iter()
            .map(|entry| entry.anchor.map(|anchor| anchor.end))
            .collect::<Vec<_>>();
        let Some(halves) = whole.split_at_with_rows(at, &ends) else {
            return self.run_eager(plan, devices, whole).map(|()| false);
        };
        let halves = [halves.0, halves.1];
        let mut part_devices = Vec::new();
        let mut part_plans = Vec::new();
        let mut part_scratch = Vec::new();
        let mut originals = Vec::new();
        for (half, phase) in halves.iter().enumerate() {
            let indices = phase
                .request_ids
                .iter()
                .map(|id| {
                    whole
                        .request_ids
                        .iter()
                        .position(|whole| whole == id)
                        .ok_or(BundleStatusV1::InvalidArgument)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let part_entries = indices
                .iter()
                .enumerate()
                .map(|(item, &original)| {
                    let mut entry = entries[original];
                    if phase.prefix_len[item] > whole.prefix_len[original] {
                        entry.committed = entry.working;
                    }
                    if !phase.anchors[item] {
                        entry.anchor = None;
                    }
                    entry
                })
                .collect::<Vec<_>>();
            part_devices.push(
                self.upload_phase(HALVES[half], phase, &part_entries)
                    .map_err(status)?,
            );
            let mut part = PhasePlan::new(self.shape, phase, self.items)?
                .ok_or(BundleStatusV1::InvalidArgument)?;
            // The final mixer must cover every whole-chunk head/retained row.
            part.head_rows = plan.head_rows;
            // Preserve the whole program's context-dependent QSA grid.
            part.context = plan.context;
            part.inject = plan.inject;
            part_plans.push(part);
            let start = if half == 0 { 0 } else { at };
            let overlap = self.overlap.as_ref().ok_or(BundleStatusV1::InvalidState)?;
            part_scratch.push(
                self.scratch
                    .iter()
                    .enumerate()
                    .map(|(rank, scratch)| {
                        scratch.overlap_part(
                            start,
                            indices[0],
                            (half == 1).then_some(overlap.histories[rank]),
                            self.shape,
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            originals.push(indices);
        }
        let dump_dir = match self
            .dump
            .as_mut()
            .map(runtime_bundle_sdk::taps::Taps::next_step)
        {
            Some(Ok(dir)) => Some(dir),
            Some(Err(_)) => return Err(BundleStatusV1::InternalError),
            None => None,
        };
        let encoders = std::mem::take(&mut self.tap_encoders);
        let capture = match (&self.dump, &dump_dir) {
            (Some(dump), Some(dir)) => Some(
                crate::dump::begin(
                    dump,
                    dir,
                    plan.program,
                    plan.rows,
                    whole,
                    self.shape,
                    encoders,
                )
                .map_err(|_| BundleStatusV1::InternalError)?,
            ),
            _ => None,
        };
        struct Walk<'a> {
            instance: &'a mut Instance,
            whole: &'a PhaseInputs,
            plan: &'a PhasePlan,
            devices: &'a [PhaseDevice],
            halves: &'a [PhaseInputs; 2],
            plans: &'a [PhasePlan],
            part_devices: &'a [Vec<PhaseDevice>],
            scratch: &'a [Vec<RankScratch>],
            originals: &'a [Vec<usize>],
            at: u32,
            capture: Option<runtime_bundle_sdk::taps::TapStep>,
        }
        impl Walk<'_> {
            fn stream(&self, rank: usize, lane: Lane) -> BundleStreamHandleV1 {
                match lane {
                    Lane::Compute => self.instance.streams[rank],
                    Lane::Comm => {
                        self.instance
                            .overlap
                            .as_ref()
                            .expect("overlap acquired")
                            .streams[rank]
                    }
                }
            }
            fn event(&self, lane: Lane, mark: Mark, record: bool) -> Result<(), BundleStatusV1> {
                for rank in 0..self.instance.streams.len() {
                    let event = self
                        .instance
                        .overlap
                        .as_ref()
                        .expect("overlap acquired")
                        .events[rank][mark.index(program::OVERLAP_PARTS)];
                    let stream = self.stream(rank, lane);
                    if record {
                        self.instance.services.record_event(event, stream)
                    } else {
                        self.instance.services.wait_event(stream, event)
                    }
                    .map_err(status)?;
                }
                Ok(())
            }
        }
        impl Executor<program::Stage> for Walk<'_> {
            type Error = BundleStatusV1;
            fn launch(
                &mut self,
                stage: program::Stage,
                part: Option<usize>,
                lane: Lane,
            ) -> Result<(), Self::Error> {
                if stage == program::Stage::NgramGather && part == Some(1) {
                    // Half 0 has completed its PLE group on this stream.
                    // Continuing items read that output; later items read
                    // their original input. Keep input and output disjoint.
                    for rank in 0..self.instance.streams.len() {
                        for (item, &original) in self.originals[1].iter().enumerate() {
                            let continued =
                                self.halves[1].prefix_len[item] > self.whole.prefix_len[original];
                            let staging = self.instance.scratch[rank].ple_staging[0];
                            let source = if continued {
                                staging.output
                            } else {
                                staging.input
                            };
                            let bytes = self.instance.shape.ple_entry_bytes();
                            let destination = self.scratch[1][rank].ple_staging[0].input;
                            self.instance
                                .services
                                .copy_device_to_device(
                                    self.instance.streams[rank],
                                    DeviceBufferSpanV1::new(
                                        destination.buffer,
                                        destination.offset + item as u64 * bytes,
                                        bytes,
                                    ),
                                    DeviceBufferSpanV1::new(
                                        source.buffer,
                                        source.offset + original as u64 * bytes,
                                        bytes,
                                    ),
                                )
                                .map_err(status)?;
                        }
                    }
                }
                let mut lowered = Vec::new();
                for rank in 0..self.instance.streams.len() {
                    let (mut launches, facts) = if let Some(part) = part {
                        let mut binding = self.instance.rank_stage(
                            &self.plans[part],
                            self.part_devices[part][rank],
                            rank,
                        );
                        binding.peers = &self.scratch[part];
                        let first_row = if part == 1 { self.at } else { 0 };
                        let (lowered, facts) = match binding.lower_with_facts(stage, first_row) {
                            Ok(lowered) => {
                                let (launches, facts) = crate::ops::unzip(lowered);
                                (Ok(launches), facts)
                            }
                            Err(refusal) => (Err(refusal), Vec::new()),
                        };
                        lowered
                            .map_err(|refusal| {
                                let _ = self.instance.services.log(
                                format!(
                                    "Qwen3.8 overlap rank {rank} part {part} stage {stage:?}: {}",
                                    refusal.0
                                )
                                .as_bytes(),
                            );
                                BundleStatusV1::Unsupported
                            })
                            .map(|launches| (launches, facts))?
                    } else {
                        crate::ops::unzip(self.instance.lower_with_facts(
                            self.plan,
                            self.devices[rank],
                            rank,
                            stage,
                        )?)
                    };
                    let config = &self
                        .instance
                        .overlap
                        .as_ref()
                        .expect("overlap acquired")
                        .config;
                    // Each launch's grid moves by the facts of the op that
                    // made it, as the symbol rule moved it.
                    for (launch, facts) in launches.iter_mut().zip(facts) {
                        #[cfg(test)]
                        crate::sm120::overlap::compare_with_symbols(
                            self.instance.shape,
                            launch,
                            facts,
                            lane,
                            config,
                        );
                        self.instance.target.adjust(launch, facts, lane, config);
                    }
                    #[cfg(test)]
                    if part.is_some() {
                        PLANNED.with(|planned| {
                            planned.borrow_mut().extend(
                                launches
                                    .iter()
                                    .map(|launch| (rank, launch.symbol.clone(), launch.rows)),
                            )
                        });
                    }
                    lowered.push(launches);
                }
                let mut unit = crate::pipeline::Unit::default();
                for (rank, launches) in lowered.iter().enumerate() {
                    if !launches.is_empty() {
                        unit.record(
                            &self.instance.collectives,
                            crate::pipeline::launch_all(
                                &self.instance.services,
                                self.stream(rank, lane),
                                &self.instance.symbols[rank],
                                launches,
                            ),
                        )
                        .map_err(crate::dispatch::status)?;
                    }
                }
                if let (Some(dump), Some(capture)) = (&self.instance.dump, &mut self.capture) {
                    for tap in crate::dump::taps(stage, self.instance.shape) {
                        if !dump.selected(&tap.spec) {
                            continue;
                        }
                        for rank in 0..tap.spec.sharding.ranks(self.instance.shape.ranks()) {
                            let index = rank as usize;
                            let stream = match lane {
                                Lane::Compute => self.instance.streams[index],
                                Lane::Comm => {
                                    self.instance
                                        .overlap
                                        .as_ref()
                                        .expect("overlap acquired")
                                        .streams[index]
                                }
                            };
                            let result = if let Some(part) = part {
                                capture.capture_part(
                                    &self.instance.services,
                                    stream,
                                    &tap.spec,
                                    rank,
                                    (tap.buffer)(&self.scratch[part][index]),
                                    if part == 0 { 0 } else { self.at },
                                    self.halves[part].live_tokens(),
                                )
                            } else {
                                capture.capture(
                                    &self.instance.services,
                                    stream,
                                    &tap.spec,
                                    rank,
                                    (tap.buffer)(&self.instance.scratch[index]),
                                )
                            };
                            result.map_err(|error| match error {
                                runtime_bundle_sdk::taps::TapError::Host(error) => status(error),
                                runtime_bundle_sdk::taps::TapError::Io(_) => {
                                    BundleStatusV1::InternalError
                                }
                            })?;
                        }
                    }
                }
                Ok(())
            }
            fn record(&mut self, lane: Lane, mark: Mark) -> Result<(), Self::Error> {
                self.event(lane, mark, true)
            }
            fn wait(&mut self, lane: Lane, mark: Mark) -> Result<(), Self::Error> {
                self.event(lane, mark, false)
            }
        }
        let schedule = program::overlap_schedule(self.shape);
        let mut walk = Walk {
            instance: self,
            whole,
            plan,
            devices,
            halves: &halves,
            plans: &part_plans,
            part_devices: &part_devices,
            scratch: &part_scratch,
            originals: &originals,
            at,
            capture,
        };
        bundle_runtime::lanes::execute(schedule, &mut walk)?;
        if let Some(capture) = walk.capture {
            self.pending_taps.push(capture);
        }
        Ok(true)
    }

    fn drain_overlap(&self) {
        let Some(overlap) = &self.overlap else {
            return;
        };
        bundle_runtime::lanes::drain(
            self.collectives.is_poisoned(),
            overlap.streams.iter().copied(),
            self.streams.iter().copied(),
            |stream| self.services.synchronize_stream(stream),
            |error| {
                let _ = self.services.log(
                    format!("Qwen3.8 overlap drain failed: {error}; instance poisoned").as_bytes(),
                );
                self.collectives.poison();
            },
        );
    }

    /// Copies `stage`'s selected taps of the step's rows to its directory.
    fn dump_stage(
        &self,
        dump: &runtime_bundle_sdk::taps::Taps,
        capture: &mut runtime_bundle_sdk::taps::TapStep,
        stage: program::Stage,
    ) -> Result<(), BundleStatusV1> {
        for tap in crate::dump::taps(stage, self.shape) {
            if !dump.selected(&tap.spec) {
                continue;
            }
            for rank in 0..tap.spec.sharding.ranks(self.shape.ranks()) {
                let buffer = (tap.buffer)(&self.scratch[rank as usize]);
                capture
                    .capture(
                        &self.services,
                        self.streams[rank as usize],
                        &tap.spec,
                        rank,
                        buffer,
                    )
                    .map_err(|error| match error {
                        runtime_bundle_sdk::taps::TapError::Host(error) => status(error),
                        runtime_bundle_sdk::taps::TapError::Io(_) => BundleStatusV1::InternalError,
                    })?;
            }
        }
        Ok(())
    }

    /// Decode replays the bucket's graphs, capturing any rank that lacks
    /// them.
    fn run_decode(
        &mut self,
        plan: &PhasePlan,
        devices: &[PhaseDevice],
    ) -> Result<(), BundleStatusV1> {
        let (bucket, items) = match plan.program {
            Phase::Decode(bucket) => (bucket, 0),
            Phase::Verify { rows, items } => (rows, items),
            Phase::Prefill => return Err(BundleStatusV1::InternalError),
        };
        // Plan only the ranks whose bucket graph is missing; every rank is
        // captured before any launches (see `pipeline::run_ranks`).
        let mut lowered = devices
            .iter()
            .enumerate()
            .map(|(rank, &device)| {
                let key = crate::graphs::GraphKey {
                    rank: rank as u32,
                    bucket,
                    items,
                };
                if self.graphs.contains(key) {
                    Ok(None)
                } else {
                    self.plan_rank(plan, device, rank).map(Some)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        crate::pipeline::run_ranks(
            &self.services,
            &self.streams,
            &self.symbols,
            Some((bucket, items)),
            &mut self.graphs,
            &self.collectives,
            |rank| {
                lowered[rank]
                    .take()
                    .ok_or(crate::dispatch::DispatchError::InvalidPlan)
            },
            |_| {},
        )
        .map_err(crate::dispatch::status)
    }

    /// Uploads one phase's shared and side inputs to every rank.
    fn upload_phase(
        &mut self,
        index: usize,
        phase: &PhaseInputs,
        entries: &[ItemEntries],
    ) -> Result<Vec<PhaseDevice>, BundleAbiError> {
        let (layout, side_layout) = self.phase_layouts(phase);
        let rows = if phase.decode {
            Vec::new()
        } else {
            phase.logit_rows()
        };
        let bytes = layout
            .pack(phase, &rows)
            .ok_or(BundleAbiError::InvalidRange(
                "Qwen3.8 inputs exceed their arena",
            ))?;
        let side_bytes = side_layout
            .pack(phase, entries)
            .ok_or(BundleAbiError::InvalidRange(
                "Qwen3.8 side inputs exceed their arena",
            ))?;
        let writes = self.block_tables[index].update(&phase.block_tables);
        let uploaded = (|| {
            for rank in 0..self.shape.ranks.count() {
                let stream = self.streams[rank];
                let arena = self.arenas[index][rank];
                upload_bytes(&self.services, stream, arena.arena, &bytes)?;
                upload_bytes(
                    &self.services,
                    stream,
                    self.side_arenas[index][rank],
                    &side_bytes,
                )?;
                self.block_tables[index].upload(
                    &self.services,
                    stream,
                    arena.block_tables,
                    &writes,
                )?;
            }
            Ok(())
        })();
        if let Err(error) = uploaded {
            self.block_tables[index].forget();
            return Err(error);
        }
        Ok((0..self.shape.ranks.count())
            .map(|rank| PhaseDevice {
                inputs: DeviceInputs::within(self.arenas[index][rank], &layout),
                side: SideInputs::within(self.side_arenas[index][rank], &side_layout),
            })
            .collect())
    }

    /// Packs a step and, per decode item, the drafts the bundle writes.
    fn pack(&self, request: &[u8]) -> Result<(PackedStep, Vec<u32>), BundleStatusV1> {
        let batch = BatchCodecV1::decode(request).map_err(|_| BundleStatusV1::InvalidArgument)?;
        if batch
            .items()
            .iter()
            .flat_map(|item| item.position_axes())
            .flatten()
            .copied()
            .any(|position| !crate::side::valid_rope_position(self.shape, position))
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if !batch.media().is_empty() && self.encoder.is_none() {
            return Err(BundleStatusV1::Unsupported);
        }
        // A context-parallel group's tables list one local page per stripe.
        let step = PackedStep::pack_on_host_slots::<Qwen>(
            self.shape.model_limits(),
            &batch,
            self.kv_page_capacity,
            self.verify_limits,
            self.items,
            u64::from(self.context_parallel.as_ref().map_or(1, |cp| cp.size)),
        )?;
        let drafts = step.bundle_drafts(&batch);
        // The drafter follows every token, so it drafts every verify item.
        if self.drafter.is_some()
            && step
                .decode
                .verify_rows
                .iter()
                .zip(&drafts)
                .any(|(&verify, &count)| verify && count == 0)
        {
            let _ = self
                .services
                .log(b"Qwen3.8 bundle verifies only the drafts its MTP drafter wrote");
            return Err(BundleStatusV1::Unsupported);
        }
        // Scored prompt rows take one F32 logits row each.
        if step.prefill.prompt_rows.iter().any(|&row| row)
            && step.prefill.logit_rows().len() > SCORED_ROWS as usize
        {
            let _ = self.services.log(
                format!("Qwen3.8 bundle scores at most {SCORED_ROWS} prompt rows per step")
                    .as_bytes(),
            );
            return Err(BundleStatusV1::Unsupported);
        }
        Ok((step, drafts))
    }

    fn execute_step(&mut self, request: &[u8], sink: &mut ExecuteResultSinkV1) -> BundleStatusV1 {
        if self.ledger.step().is_some() {
            return BundleStatusV1::InvalidState;
        }
        let (step, drafts) = match self.pack(request) {
            Ok(step) => step,
            Err(error) => return error,
        };
        let items = crate::state::items([&step.prefill, &step.decode]);
        match self.ledger.preview(&items) {
            Ok(_) => {}
            Err(LedgerError::Open) => return BundleStatusV1::InvalidState,
            Err(_) => return BundleStatusV1::InvalidArgument,
        }
        if !crate::state::fits_checkpoints(&step.decode, self.speculation.verify_rows) {
            return BundleStatusV1::InvalidState;
        }
        let (kept_at, anchors) = match self.plan_anchors(&step.prefill) {
            Ok(anchors) => anchors,
            Err(error) => return error,
        };
        let mut plans = match (
            PhasePlan::new(self.shape, &step.prefill, self.items),
            PhasePlan::new(self.shape, &step.decode, self.items),
        ) {
            (Ok(prefill), Ok(decode)) => [prefill, decode],
            (Err(error), _) | (_, Err(error)) => return error,
        };
        let drafter_plan = match &self.drafter {
            Some(runtime) => match runtime.state.plan(&step.prefill, &step.decode, &drafts) {
                Ok(plan) => Some(plan),
                Err(_) => return BundleStatusV1::InvalidArgument,
            },
            None => None,
        };
        // Encoders run before the transaction opens and the ring is saved,
        // so a failure needs no restore.
        if let Some(encoder) = &self.encoder {
            let inject = match encoder.run(
                &self.services,
                &self.streams,
                &self.symbols,
                request,
                &step.prefill,
                &self.collectives,
            ) {
                Ok(inject) => inject,
                Err(error) => return error,
            };
            if let Some(plan) = &mut plans[PREFILL] {
                plan.inject = inject;
            }
            // Every step of a tapped instance carries each prompt's media
            // occurrences, so a later chunk's rows keep their request and
            // items.
            if self.dump.is_some() {
                let read = runtime_bundle_sdk::BatchCodecV1::decode(request)
                    .map_err(|_| "undecodable batch request".to_owned())
                    .and_then(|batch| {
                        let items = bundle_runtime::media::OccurrenceItem::of(&batch);
                        self.tap_occurrences
                            .step(
                                &crate::encoder::arena_for(self.shape),
                                batch.media(),
                                &items,
                                crate::dump::kinds_for(self.shape),
                            )
                            .map_err(|error| error.to_string())
                    })
                    .and_then(|step| {
                        let arena = encoder.ranks[0].region("arena")?;
                        crate::dump::encoder_rows(
                            self.shape,
                            &self.services,
                            self.streams[0],
                            arena,
                            &step,
                        )
                    });
                match read {
                    Ok(rows) => self.tap_encoders = rows,
                    Err(error) => {
                        let _ = self
                            .services
                            .log(format!("Qwen3.8 encoder taps: {error}").as_bytes());
                        return BundleStatusV1::InternalError;
                    }
                }
            }
        }
        let Ok(opened) = self.ledger.begin(&items, anchors) else {
            return BundleStatusV1::InvalidState;
        };
        // Verify items take checkpoints in the step's verify order.
        let verify_rows = self.speculation.verify_rows;
        let mut entries = [
            crate::state::entries(
                opened,
                0,
                &step.prefill,
                &self.linear_entries,
                verify_rows,
                &self.ple_entries,
            ),
            crate::state::entries(
                opened,
                step.prefill.request_ids.len(),
                &step.decode,
                &self.linear_entries,
                verify_rows,
                &self.ple_entries,
            ),
        ];
        for (entry, anchor) in entries[PREFILL].iter_mut().zip(kept_at) {
            entry.anchor = anchor;
        }
        self.open = Some(Open {
            ple: [plans[PREFILL].is_some(), plans[DECODE].is_some()],
            drafter: drafter_plan.as_ref().map(crate::drafter::Plan::pending),
            ..Open::default()
        });
        let executed = self.run_step(&step, &entries, &plans, drafter_plan.as_ref(), sink);
        if let Err(error) = executed {
            self.drain_overlap();
            // Restore copies would queue behind a collective that never
            // completes; a poisoned instance leaves its streams alone.
            if self.collectives.is_poisoned() {
                let _ = self.services.log(
                    b"Qwen3.8 execute left collectives unmatched across ranks; the instance is poisoned",
                );
                return error;
            }
            if let Some(open) = self.open.as_mut() {
                open.cancel_only = true;
            }
            let restored = transaction::cancel(self);
            return if restored == BundleStatusV1::Success {
                error
            } else {
                // V1 retains a transaction by reporting success with unusable
                // output. The host can now cancel through its Pending handle.
                BundleStatusV1::Success
            };
        }
        BundleStatusV1::Success
    }

    /// Gives each prefill item that keeps its state at an anchor an anchor
    /// item, in item order, and returns where each item keeps it and the
    /// anchor items' (request, anchor tokens).
    ///
    /// The anchor is the last `PREFILL_GRANULE` multiple before the item's
    /// unaligned end, and the item starts on a multiple at least one granule
    /// before it: the linear kernels keep the state after the item's last
    /// whole delta-rule chunk, and the PLE, ring and retained rows of the
    /// anchor lie within the item's rows. Only a prompt-ending item keeps an
    /// anchor, at most `ANCHOR_ITEMS` a step.
    #[allow(clippy::type_complexity)]
    fn plan_anchors(
        &self,
        phase: &PhaseInputs,
    ) -> Result<(Vec<Option<crate::side::Anchor>>, Vec<(u64, u64)>), BundleStatusV1> {
        let granule = PREFILL_GRANULE as i32;
        let mut kept_at = vec![None; phase.request_ids.len()];
        let mut anchors = Vec::new();
        for (item, entry) in kept_at.iter_mut().enumerate() {
            if !phase.anchors.get(item).copied().unwrap_or(false) {
                continue;
            }
            let (start, end) = (
                phase.prefix_len[item],
                phase.prefix_len[item] + phase.q_len[item],
            );
            let anchor = end / granule * granule;
            if phase.decode
                || !phase.emits_row[item]
                || start % granule != 0
                || end % granule == 0
                || anchor - start < granule
                || anchors.len() >= ANCHOR_ITEMS as usize
            {
                return Err(BundleStatusV1::InvalidArgument);
            }
            let slot = anchors.len() as u32;
            *entry = Some(crate::side::Anchor {
                entry: self.speculation.anchor_entry(self.served.slots, slot) as i32,
                slot: slot as i32,
                end: anchor,
            });
            anchors.push((phase.request_ids[item], anchor as u64));
        }
        Ok((kept_at, anchors))
    }

    fn run_step(
        &mut self,
        step: &PackedStep,
        entries: &[Vec<ItemEntries>; 2],
        plans: &[Option<PhasePlan>; 2],
        drafter: Option<&crate::drafter::Plan>,
        sink: &mut ExecuteResultSinkV1,
    ) -> Result<(), BundleStatusV1> {
        if plans[PREFILL].is_some() {
            // A prefill chunk rewrites ring rows of its slot's open block;
            // cancel restores them from this copy.
            let bytes = self.ring().bytes();
            for rank in 0..self.shape.ranks.count() {
                let ring = self.rings[rank];
                self.services
                    .copy_device_to_device(
                        self.streams[rank],
                        DeviceBufferSpanV1::new(ring.buffer, ring.offset + bytes, bytes),
                        DeviceBufferSpanV1::new(ring.buffer, ring.offset, bytes),
                    )
                    .map_err(status)?;
            }
            if let Some(open) = self.open.as_mut() {
                open.ring_saved = true;
            }
        } else if speculation::drafted(drafter) {
            // Catch-up and chained drafts write up to fourteen positions of
            // the drafter's ring past its open block.
            self.copy_drafter_ring(false)?;
            if let Some(open) = self.open.as_mut() {
                open.drafter_ring_saved = true;
            }
        }
        let mut devices: [Option<Vec<PhaseDevice>>; 2] = [None, None];
        for (index, phase) in [&step.prefill, &step.decode].into_iter().enumerate() {
            if plans[index].is_some() {
                devices[index] = Some(
                    self.upload_phase(index, phase, &entries[index])
                        .map_err(status)?,
                );
            }
        }
        let decode = &step.decode;
        self.step = Some(StepInputs {
            phases: devices.clone(),
            verify: decode.verify,
            decode_entries: entries[DECODE].clone(),
            decode_items: decode
                .slot_ids
                .iter()
                .zip(&decode.q_len)
                .zip(&decode.request_ids)
                .map(|((&slot, &rows), &request)| (slot as u32, rows as u32, request))
                .collect(),
        });
        // The drafter catches up and drafts before the target verifies.
        if let Some(plan) = drafter {
            for run in &plan.before {
                self.run_drafter(run, devices[DECODE].as_deref())?;
            }
        }
        if let (Some(plan), Some(devices)) = (&plans[PREFILL], &devices[PREFILL]) {
            let split = self
                .overlap
                .as_ref()
                .filter(|overlap| plan.rows >= overlap.config.min_tokens)
                .and_then(|_| {
                    let ends = entries[PREFILL]
                        .iter()
                        .map(|entry| entry.anchor.map(|anchor| anchor.end))
                        .collect::<Vec<_>>();
                    program::overlap_split(&step.prefill, &ends)
                });
            let used_split = if let Some(at) = split {
                self.run_overlap(plan, devices, &step.prefill, &entries[PREFILL], at)?
                    .then_some(at)
            } else {
                self.run_eager(plan, devices, &step.prefill)?;
                None
            };
            if self.switches.overlap_evidence {
                let capture = self
                    .pending_taps
                    .last()
                    .ok_or(BundleStatusV1::InvalidArgument)?;
                runtime_bundle_sdk::diagnostic::write_new(
                    &capture.dir().join("overlap.json"),
                    &serde_json::json!({"rows": plan.rows, "split": used_split}),
                )
                .map_err(|_| BundleStatusV1::InternalError)?;
                crate::dump::OverlapEvidence {
                    services: &self.services,
                    streams: &self.streams,
                    shape: self.shape,
                    inputs: &step.prefill,
                    entries: &entries[PREFILL],
                    scratch: &self.scratch,
                    kv: &self.kv_views,
                    rings: &self.rings,
                    ring: self.ring(),
                    pool_entries: self.speculation.pool_entries(self.served.slots),
                    page_bytes: self.layout.kv.page_bytes,
                    head_rows: plan.head_rows,
                }
                .write(capture.dir())
                .map_err(|reason| {
                    let _ = self
                        .services
                        .log(format!("Qwen3.8 overlap evidence: {reason}").as_bytes());
                    BundleStatusV1::InternalError
                })?;
            }
            if let Some(drafter) = drafter {
                self.stage_target_rows(PREFILL, &step.prefill)?;
                if let Some(run) = &drafter.prefill {
                    self.run_drafter(run, None)?;
                }
            }
        }
        if let (Some(plan), Some(devices)) = (&plans[DECODE], &devices[DECODE]) {
            if self.dump.is_some() || self.eager {
                self.run_eager(plan, devices, &step.decode)?;
            } else {
                self.run_decode(plan, devices)?;
            }
            if drafter.is_some() {
                self.stage_target_rows(DECODE, decode)?;
            }
        }
        // Every owner must finish before the host samples its full rows.
        let owners = if !self.exchange.is_empty() && plans[DECODE].is_some() {
            self.streams.len()
        } else {
            1
        };
        for &stream in &self.streams[..owners] {
            self.services.synchronize_stream(stream).map_err(status)?;
        }
        if let (Some(plan), Some(devices)) = (drafter, &devices[DECODE]) {
            self.emit_drafts(step, plan, devices, sink)?;
        }
        let views = self.logit_views(step, plans[DECODE].as_ref())?;
        for view in views {
            sink.emit_device_view(view).map_err(status)?;
        }
        Ok(())
    }

    /// Publishes committed captures and removes cancelled captures.
    fn finish_taps(&mut self, commit: bool) -> BundleStatusV1 {
        self.tap_occurrences.finish(commit);
        self.tap_encoders.clear();
        for capture in std::mem::take(&mut self.pending_taps) {
            let result = if commit {
                capture.finish()
            } else {
                capture.cancel()
            };
            if result.is_err() {
                return BundleStatusV1::InternalError;
            }
        }
        BundleStatusV1::Success
    }

    fn restore_ring(&self, open: &Open) -> Result<(), BundleStatusV1> {
        if open.drafter_ring_saved {
            return self.copy_drafter_ring(true);
        }
        if !open.ring_saved {
            return Ok(());
        }
        let bytes = self.ring().bytes();
        for rank in 0..self.shape.ranks.count() {
            let ring = self.rings[rank];
            self.services
                .copy_device_to_device(
                    self.streams[rank],
                    DeviceBufferSpanV1::new(ring.buffer, ring.offset, bytes),
                    DeviceBufferSpanV1::new(ring.buffer, ring.offset + bytes, bytes),
                )
                .map_err(status)?;
        }
        Ok(())
    }

    /// Copies each phase's staged PLE state back to its slots; a verify
    /// phase's kept checkpoint ownership is exchanged by `commit_speculation`.
    fn commit_ple(&self, open: &Open, step: Option<&StepInputs>) -> Result<(), BundleStatusV1> {
        let Some(step) = step else {
            return Ok(());
        };
        for (index, devices) in step.phases.iter().enumerate() {
            let (true, Some(devices)) = (open.ple[index], devices) else {
                continue;
            };
            if index == DECODE && step.verify {
                continue;
            }
            let copy = &self
                .copies
                .as_ref()
                .ok_or(BundleStatusV1::InternalError)?
                .ple_commit;
            for (rank, device) in devices.iter().enumerate() {
                let scratch = &self.scratch[rank];
                let spans = [
                    scratch.ple_staging[index].output,
                    scratch.ple_pool,
                    device.side.item_ids,
                    device.side.ple_working,
                    device.inputs.live_batch,
                ];
                let mut bind = crate::ops::Bind::default();
                let buffers = CopyEntriesBuffers {
                    src: bind.view(spans[0]),
                    dst: bind.view(spans[1]),
                    src_ids: bind.view(spans[2]),
                    dst_ids: bind.view(spans[3]),
                    live_items: bind.view(spans[4]),
                };
                // Every served item's entry.
                let items = self.served.items;
                let launches = bind.launch(
                    copy,
                    &buffers,
                    kernel_core::backend::Counts::rows(1).with_items(items, 0),
                );
                let launches = launches.map_err(|_| BundleStatusV1::InternalError)?;
                crate::pipeline::launch_all(
                    &self.services,
                    self.streams[rank],
                    &self.symbols[rank],
                    &launches,
                )
                .map_err(crate::dispatch::status)?;
            }
        }
        Ok(())
    }
}

thread_local! {
    static INSTANCES: RefCell<InstanceTable<Instance>> = const { RefCell::new(InstanceTable::new()) };
}

/// What `acquire` resolves, moved into the instance once all of it exists.
#[derive(Default)]
struct Acquired {
    precision: Option<crate::precision::Prepared>,
    views: Vec<BTreeMap<String, DeviceBufferSpanV1>>,
    contexts: Vec<BundleContextHandleV1>,
    streams: Vec<BundleStreamHandleV1>,
    rings: Vec<DeviceBufferSpanV1>,
    kv_views: Vec<DeviceBufferSpanV1>,
    kv_page_capacity: u64,
    scratch: Vec<RankScratch>,
    exchange: Vec<crate::logits::ExchangeRowsState>,
    arenas: [Vec<PhaseArena>; 4],
    side_arenas: [Vec<DeviceBufferSpanV1>; 4],
    symbols: Vec<BTreeMap<String, BundleCudaSymbolHandleV1>>,
    index_arenas: Vec<DeviceBufferSpanV1>,
    drafter: Option<speculation::DrafterRuntime>,
    encoder: Option<crate::encoder::Encoder>,
    overlap: Option<Overlap>,
    context_parallel: Vec<crate::context_parallel::RankState>,
}

fn acquire(
    services: &HostServicesV1,
    config: &Configuration,
    speculation: Speculation,
    target_graphs: &GraphCache,
    held: &mut Vec<Held>,
    instance: &mut Acquired,
) -> Result<(), BundleAbiError> {
    let log = Log {
        services,
        bundle: "Qwen3.8",
    };
    if config.precision {
        let bytes = acquire::read_component(
            log,
            model_qwen38_contract::prepared::COMPONENT_ROLE,
            kernel_core::precision::component::MAX_COMPONENT_BYTES as u64,
            held,
        )?;
        let precision =
            crate::precision::Prepared::decode(&bytes, config.shape, config.target.as_ref())
                .map_err(|error| {
                    let _ = services.log(error.as_bytes());
                    BundleAbiError::InvalidRange("Qwen prepared operations")
                })?;
        if speculation.drafter
            && let Err(error) =
                model_qwen38_contract::projections::support::mtp(config.shape, precision.document())
        {
            let _ = services.log(error.0.as_bytes());
            return Err(BundleAbiError::InvalidRange(
                "mixed Qwen MTP execution has no prepared contract",
            ));
        }
        instance.precision = Some(precision);
    }
    // Each rank's index, bound to the image the host serves for that rank
    // (its file's digest, or its plan's identity) before the image is opened.
    let admit = |rank, bytes: &[u8]| -> Result<_, BundleAbiError> {
        let index = crate::views::parse(bytes, rank).map_err(|error| {
            let _ = services.log(format!("Qwen3.8 rank index: {error}").as_bytes());
            BundleAbiError::InvalidRange("Qwen3.8 rank index")
        })?;
        let validation = if let Some(precision) = &instance.precision {
            model_qwen38_contract::prepared::validate_views(
                &index,
                config.shape,
                precision.document(),
            )
        } else {
            crate::views::validate(&index, config.shape)
        };
        validation
            .and_then(|()| {
                instance
                    .precision
                    .as_mut()
                    .map_or(Ok(()), |precision| precision.bind_norms(&index))
            })
            .map_err(|error| {
                let _ = services.log(error.as_bytes());
                BundleAbiError::InvalidRange("Qwen3.8 rank image views")
            })?;
        if speculation.drafter {
            // A recorded MTP layer's views in their prepared encodings; any
            // precision admits the declared F32 MTP hyper-connection norms.
            let validation = match &instance.precision {
                Some(precision)
                    if model_qwen38_contract::prepared::records_drafter(
                        precision.document(),
                        config.shape,
                    )
                    .unwrap_or(false) =>
                {
                    model_qwen38_contract::prepared::validate_drafter_views(
                        &index,
                        config.shape,
                        precision.document(),
                    )
                }
                Some(_) => model_qwen38_contract::norms::validate_drafter(&index, config.shape),
                None => crate::views::validate_drafter(&index, config.shape),
            };
            validation.map_err(|error| {
                let _ = services.log(error.as_bytes());
                BundleAbiError::InvalidRange("Qwen3.8 MTP drafter views")
            })?;
        }
        Ok(index)
    };
    let images = acquire::RankImages {
        role: RANK_IMAGE_ROLE,
        index_role: &index_role,
        max_index_bytes: 64 << 20,
    };
    let (image, indexes) = acquire::rank_images(
        log,
        images,
        config.ordinals.len() as u32,
        admit,
        acquire::identity,
        held,
    )?;
    let ranks = acquire::open_ranks(log, &config.ordinals, held)?;
    let arenas = acquire::upload_arenas(log, image, &ranks, held)?;
    let ngram = crate::views::ngram_view_for(config.shape);
    for (rank, (arena, index)) in arenas.into_iter().zip(&indexes).enumerate() {
        let rank = rank as u32;
        // The rank's n-gram rows: in pinned host memory when the package
        // declares the table as a host view, else in the arena.
        let (start, rows) =
            crate::views::ngram_rows(rank, config.shape).map_err(BundleAbiError::InvalidRange)?;
        let placements = [acquire::Placement {
            name: &ngram,
            host_view: Some(NGRAM_HOST_VIEW),
            rows: Some(acquire::RowShard { start, rows }),
        }];
        let placed =
            acquire::place_views(services, image, arena, index, &placements).map_err(|error| {
                match error {
                    acquire::PlaceError::Host(error) => error,
                    acquire::PlaceError::Layout(message) => {
                        let _ = services.log(message.as_bytes());
                        BundleAbiError::InvalidRange("Qwen3.8 rank arena length")
                    }
                }
            })?;
        instance.views.push(acquire::spans(placed));
    }
    if config.media {
        instance.encoder = Some(log.named(
            "vision encoder",
            crate::encoder::acquire(config.shape, config.target.as_ref(), services, &ranks, held),
        )?);
    }
    acquire::grant_peers(services, &ranks, held)?;
    let shape = config.shape;
    let (items, served) = (config.items, Served::new(config.items));
    let workspace = config
        .target
        .workspace_bytes(served)
        .map_err(|_| BundleAbiError::InvalidRange("Qwen3.8 planner workspace"))?
        .max(instance.precision.as_ref().map_or(0, |p| p.workspace_bytes));
    instance.overlap = config.overlap.map(|config| Overlap {
        config,
        streams: Vec::new(),
        events: Vec::new(),
        histories: Vec::new(),
    });
    for (context, pool, stream) in &ranks {
        if let Some(overlap) = &mut instance.overlap {
            let comm = bundle_runtime::lanes::SideLane::create(
                services,
                *context,
                program::OVERLAP_EVENTS,
                held,
            )?;
            overlap.streams.push(comm.stream);
            overlap.events.push(comm.events);
            overlap.histories.push(allocate_span(
                services,
                *pool,
                *stream,
                u64::from(served.items) * shape.ple_entry_bytes(),
                held,
            )?);
        }
        for (index, (arena_bytes, side_bytes)) in [
            (
                InputLayout::prefill_capacity_within::<Qwen>(items),
                SideLayout::for_shape(shape, PREFILL_TOKENS, served.items).end,
            ),
            (
                InputLayout::verify_within::<Qwen>(items).end,
                SideLayout::for_shape(shape, served.rows, served.items).end,
            ),
            (
                InputLayout::prefill_capacity_within::<Qwen>(items),
                SideLayout::for_shape(shape, PREFILL_TOKENS, served.items).end,
            ),
            (
                InputLayout::prefill_capacity_within::<Qwen>(items),
                SideLayout::for_shape(shape, PREFILL_TOKENS, served.items).end,
            ),
        ]
        .into_iter()
        .enumerate()
        .take(if config.overlap.is_some() { 4 } else { 2 })
        {
            let arena = allocate_span(services, *pool, *stream, arena_bytes, held)?;
            let block_tables = allocate_span(
                services,
                *pool,
                *stream,
                BlockTableMirror::device_bytes(shape.model_limits(), items),
                held,
            )?;
            instance.arenas[index].push(PhaseArena {
                arena,
                block_tables,
            });
            instance.side_arenas[index]
                .push(allocate_span(services, *pool, *stream, side_bytes, held)?);
        }
        let comm = config.target.comm_bytes(served);
        let mut scratch = RankScratch::allocate_with_overlap(
            &comm,
            workspace,
            speculation,
            config.layout.kv.page_bytes,
            shape,
            served,
            config.overlap.is_some(),
            config.context_parallel,
            |bytes| allocate_span(services, *pool, *stream, bytes, held),
        )?;
        if let Some(precision) = &instance.precision
            && precision.temporary_bytes != 0
        {
            scratch.precision_temporary = Some(allocate_span(
                services,
                *pool,
                *stream,
                precision.temporary_bytes,
                held,
            )?);
        }
        instance.index_arenas.push(allocate_span(
            services,
            *pool,
            *stream,
            speculation::INDEX_ARENA_BYTES,
            held,
        )?);
        for span in scratch.zeroed(&comm) {
            services.zero_device_buffer(*stream, span)?;
        }
        upload_bytes(
            services,
            *stream,
            scratch.cos_sin,
            crate::scratch::cos_sin_table_for(shape),
        )?;
        instance.scratch.push(scratch);
        // The context-parallel state the group writes: after peer access.
        if config.context_parallel > 1 {
            // The contract's capacity plan counts the same bytes.
            let bytes =
                model_qwen38_contract::context_parallel::rank_bytes(shape, config.context_parallel)
                    .map_err(|error| {
                        let _ = services.log(error.as_bytes());
                        BundleAbiError::InvalidRange("Qwen3.8 context-parallel state")
                    })?;
            let state = crate::context_parallel::RankState::allocate(&bytes, |bytes| {
                allocate_span(services, *pool, *stream, bytes, held)
            })?;
            for span in state.zeroed() {
                services.zero_device_buffer(*stream, span)?;
            }
            instance.context_parallel.push(state);
        }
        if let Some(precision) = instance.precision.as_mut() {
            precision
                .bind_rank(&instance.views[instance.scratch.len() - 1])
                .map_err(|error| {
                    let _ = services.log(error.as_bytes());
                    BundleAbiError::InvalidRange("Qwen bound projection weights")
                })?;
        }
        if config.switches.row_owned_logits {
            let exchange = crate::logits::ExchangeRowsState::allocate(&mut |bytes| {
                allocate_span(services, *pool, *stream, bytes, held)
            })?;
            for span in [exchange.flags, exchange.epoch] {
                services.zero_device_buffer(*stream, span)?;
            }
            instance.exchange.push(exchange);
        }
        let mut symbols = BTreeMap::new();
        let uses = crate::target::Uses {
            graph_rows: served.rows,
            hc_precompute: config.switches.hc_precompute,
            hc_padded_mix: config.switches.hc_padded_mix,
            context_parallel: config.context_parallel > 1,
        };
        // Shared cubins include TP4's rank-invariant entries. Parallel
        // routing and other rank-specific cubins load only for their topology.
        // The target leaves the modules of unused entries unloaded.
        for cubin in CUBINS.iter().filter(|cubin| {
            crate::GEOMETRY_CUBINS
                .iter()
                .any(|(name, geometry, ranks)| {
                    *name == cubin.name
                        && (*geometry == "*" || *geometry == shape.geometry.id)
                        && (*ranks == 0 || *ranks == shape.ranks())
                })
                && (!crate::precision::mixed_module(cubin.name)
                    || instance.precision.as_ref().is_some_and(|p| {
                        cubin
                            .symbols
                            .iter()
                            .any(|s| p.shared_memory.contains_key(*s))
                    }))
                // F32 norm-weight forms load only for an image that stores
                // F32 norm weights.
                && (!crate::F32_NORM_CUBINS.contains(&cubin.name)
                    || instance
                        .precision
                        .as_ref()
                        .is_some_and(|p| !p.norms.is_empty()))
                && config.target.loads(cubin.name, uses)
        }) {
            let precision = instance.precision.as_ref();
            let wanted = cubin.symbols.iter().copied().filter(|symbol| {
                (!crate::precision::mixed_module(cubin.name)
                    || precision.is_some_and(|p| p.shared_memory.contains_key(*symbol)))
                    && config.target.loads_symbol(symbol, uses)
            });
            let shared = |symbol: &str| match precision.and_then(|p| p.shared_memory.get(symbol)) {
                Some(shared) => Ok(*shared),
                None => config
                    .target
                    .dynamic_shared_bytes(symbol)
                    .map_err(|_| BundleAbiError::InvalidRange("Qwen3.8 symbol shared memory")),
            };
            acquire::load_module(
                log,
                *context,
                (cubin.name, cubin.bytes),
                wanted,
                shared,
                &mut symbols,
                held,
            )?;
        }
        instance.symbols.push(symbols);
    }
    if speculation.drafter {
        let pools = ranks
            .iter()
            .map(|&(_, pool, stream)| (pool, stream))
            .collect::<Vec<_>>();
        instance.drafter = Some(speculation::DrafterRuntime::allocate(
            shape,
            services,
            &pools,
            target_graphs,
            items,
            config.switches.mtp_eager,
            held,
        )?);
    }
    let layout = config.layout;
    let ring = layout.ring(shape, items);
    instance.rings =
        acquire::zeroed_state(services, RING_STATE_ID, ring.state_bytes(), &ranks, held)?;
    (instance.kv_page_capacity, instance.kv_views) = acquire::serving_kv(
        log,
        layout.kv.layout_id,
        layout.kv.page_bytes,
        shape.ranks(),
    )?;
    acquire::synchronize(log, &ranks)?;
    instance.contexts = ranks.iter().map(|(context, _, _)| *context).collect();
    instance.streams = ranks.into_iter().map(|(_, _, stream)| stream).collect();
    Ok(())
}

fn create(services: HostServicesV1, bytes: &[u8]) -> Result<Instance, Failed<Instance>> {
    let config = decode_configuration(bytes, |reason| {
        let _ = services.log(format!("Qwen3.8 create failed: {reason}").as_bytes());
    })?;
    if !crate::context_parallel::serves(config.context_parallel) {
        let _ = services.log(
            b"Qwen3.8 create failed: context-parallel attention awaits its GPU check; \
              serve with --context-parallel-size 1",
        );
        return Err(BundleStatusV1::Unsupported.into());
    }
    if let Err(error) = bundle_runtime::config::check_scored_rows(
        &services,
        "Qwen3.8",
        crate::contract::SCORED_ROWS,
    ) {
        return Err(error.into());
    }
    // A package with the MTP drafter drafts in the bundle; without it the
    // host drafts.
    let verify_limits = match services.query_speculation() {
        Ok(enabled) if enabled.maximum_draft_tokens == 0 => VerifyLimits::NONE,
        Ok(enabled) => VerifyLimits {
            bundle_drafter: config.layout.drafter(),
            ..VerifyLimits::enabled_within::<Qwen>(
                enabled.maximum_draft_tokens,
                enabled.maximum_verify_tokens,
                config.items,
            )
        },
        Err(error) => return Err(status(error).into()),
    };
    let speculation = Speculation {
        verify_rows: verify_limits.rows_per_item,
        drafter: verify_limits.bundle_drafter,
    };
    let dump = match crate::dump::from_env_for(config.shape) {
        Ok(dump) => dump.map(|taps| taps.with_producer(config.tap_producer.clone())),
        Err(error) => {
            let _ = services.log(error.as_bytes());
            return Err(BundleStatusV1::InvalidArgument.into());
        }
    };
    let mut instance = Instance {
        precision: None,
        services,
        shape: config.shape,
        eager: eager(),
        held: Vec::new(),
        views: Vec::new(),
        contexts: Vec::new(),
        streams: Vec::new(),
        rings: Vec::new(),
        kv_views: Vec::new(),
        kv_page_capacity: 0,
        scratch: Vec::new(),
        exchange: Vec::new(),
        arenas: Default::default(),
        side_arenas: Default::default(),
        block_tables: std::array::from_fn(|_| {
            BlockTableMirror::new(config.shape.model_limits(), config.items)
        }),
        overlap: None,
        symbols: Vec::new(),
        graphs: GraphCache::with_items(config.shape.ranks, config.items),
        items: config.items,
        served: Served::new(config.items),
        switches: config.switches,
        target: config.target.clone(),
        ops: None,
        copies: None,
        norm_dtypes: crate::lower::NormDtypes::new(config.shape, None),
        wide_comm: config.switches.wide_comm,
        hc_precompute: config.switches.hc_precompute,
        hc_padded_mix: config.switches.hc_padded_mix,
        collectives: Default::default(),
        context_parallel: None,
        // The linear pools invalidate the slots a cancelled step touched.
        ledger: Ledger::new(
            config.items.get(),
            Placement::Host,
            Rules::of(&model_qwen38_contract::state::request_state(
                config.shape,
                &config.layout.kv,
                speculation.drafter,
            )),
            config.shape.context_tokens(),
        ),
        linear_entries: crate::state::EntryMap::new(
            config.items.get(),
            if speculation.verify_rows > 1 {
                config.items.get().min(DRAFT_ITEMS) * speculation.verify_rows
            } else {
                0
            },
        ),
        ple_entries: crate::state::EntryMap::new(
            config.items.get(),
            if speculation.verify_rows > 0 {
                config.items.capacity() * MAX_VERIFY_ROWS
            } else {
                0
            },
        ),
        open: None,
        layout: config.layout,
        speculation,
        verify_limits,
        drafter: None,
        index_arenas: Vec::new(),
        step: None,
        dump,
        tap_encoders: Vec::new(),
        tap_occurrences: Default::default(),
        pending_taps: Vec::new(),
        encoder: None,
    };
    let _ = instance.services.log(
        format!(
            "Qwen3.8 HC precompute: {} (buckets 16/32), padded mix: {}",
            if instance.hc_precompute { "on" } else { "off" },
            if instance.hc_precompute && instance.hc_padded_mix {
                "on"
            } else {
                "off"
            },
        )
        .as_bytes(),
    );
    let mut acquired = Acquired::default();
    if let Err(error) = acquire(
        &instance.services,
        &config,
        speculation,
        &instance.graphs,
        &mut instance.held,
        &mut acquired,
    ) {
        let _ = instance
            .services
            .log(format!("Qwen3.8 create failed: {error}").as_bytes());
        let result = status(error);
        return Err(Failed::releasing(instance, result));
    }
    instance.views = acquired.views;
    instance.precision = acquired.precision;
    instance.contexts = acquired.contexts;
    instance.streams = acquired.streams;
    instance.rings = acquired.rings;
    instance.kv_views = acquired.kv_views;
    instance.kv_page_capacity = acquired.kv_page_capacity;
    instance.scratch = acquired.scratch;
    instance.exchange = acquired.exchange;
    instance.arenas = acquired.arenas;
    instance.side_arenas = acquired.side_arenas;
    instance.symbols = acquired.symbols;
    instance.index_arenas = acquired.index_arenas;
    instance.drafter = acquired.drafter;
    instance.encoder = acquired.encoder;
    instance.overlap = acquired.overlap;
    instance.context_parallel =
        (config.context_parallel > 1).then_some(crate::context_parallel::ContextParallel {
            size: config.context_parallel,
            ranks: acquired.context_parallel,
        });
    // Each rank has its own device and must keep its spinning comm grids
    // resident (`bundle_runtime::residency`). An MPS client may share
    // SMs with other processes. The wide all-reduce falls back to the general
    // entry when it may not fit; any other shortfall fails create.
    let f32_norms = instance
        .precision
        .as_ref()
        .is_some_and(|precision| !precision.norms.is_empty());
    let lanes = instance.overlap.as_ref().map(|overlap| &overlap.config);
    let residency = instance
        .target
        .spinning_entries(instance.served.rows, f32_norms, lanes)
        .map_err(|refusal| format!("Qwen3.8 comm entries: {}", refusal.0))
        .and_then(|entries| {
            bundle_runtime::residency::check_entries(
                &instance.services,
                "Qwen3.8",
                &config.ordinals,
                &instance.symbols,
                &entries,
                instance.wide_comm,
                bundle_runtime::residency::unchecked_from_environment(),
            )
        });
    match residency {
        Ok(wide) => instance.wide_comm = wide,
        Err(error) => {
            let _ = instance
                .services
                .log(format!("Qwen3.8 create failed: {error}").as_bytes());
            return Err(Failed::releasing(instance, BundleStatusV1::Unsupported));
        }
    }
    // Every op is prepared once, after residency settled the collectives,
    // and each norm site's dtype with them.
    instance.norm_dtypes = crate::lower::NormDtypes::new(
        config.shape,
        instance
            .precision
            .as_ref()
            .map(|precision| &precision.norms),
    );
    let prepared = crate::ops::Ops::prepare(
        instance.target.as_ref(),
        config.shape,
        instance.served,
        f32_norms,
        config.context_parallel,
        instance.wide_comm,
        instance.hc_precompute,
    )
    .and_then(|ops| {
        let copies = crate::ops::Copies::prepare(
            instance.target.own(),
            config.shape,
            instance.served,
            speculation::INDEX_ENTRIES as u32,
            instance.prefix_copies(),
        )?;
        Ok((ops, copies))
    });
    match prepared {
        Ok((ops, copies)) => {
            instance.ops = Some(ops);
            instance.copies = Some(copies);
        }
        Err(refused) => {
            let _ = instance
                .services
                .log(format!("Qwen3.8 create failed: an op is refused: {refused}").as_bytes());
            return Err(Failed::releasing(instance, BundleStatusV1::Unsupported));
        }
    }
    // Runners record which all-reduce served the buckets above 32 rows.
    if instance.served.rows > DECODE_ENTRY_ROWS {
        let _ = instance.services.log(
            format!(
                "Qwen3.8 decode graphs above {DECODE_ENTRY_ROWS} rows: wide all-reduce {} ({}={})",
                if instance.wide_comm { "on" } else { "off" },
                launch::WIDE_COMM,
                u8::from(instance.switches.wide_comm)
            )
            .as_bytes(),
        );
    }
    Ok(instance)
}

impl bundle_runtime::table::Live for Instance {
    fn collectives(&self) -> &crate::pipeline::Collectives {
        &self.collectives
    }
}

/// Replaces the pool format's per-layer scales, so tests can see them reach
/// the launches.
#[cfg(test)]
pub(crate) fn force_kv_scales(handle: BundleHandleV1, scales: &'static [[f32; 2]]) {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("live instance")
            .layout
            .kv
            .scales = scales;
    });
}

impl bundle_runtime::prefix::PrefixInstance for Instance {
    fn state_pages(&self) -> Result<u64, BundleStatusV1> {
        self.prefix_state_pages()
    }

    fn capture(
        &mut self,
        request: &runtime_bundle_abi::PrefixStateRequestV1,
    ) -> Result<(), BundleStatusV1> {
        self.capture_prefix(request)
    }

    /// Restore needs the request's state slot, which only the slot tail
    /// names.
    fn restore(
        &mut self,
        request: &runtime_bundle_abi::PrefixStateRequestV1,
        slot: Option<u64>,
    ) -> Result<(), BundleStatusV1> {
        let slot = slot.ok_or(BundleStatusV1::Unsupported)?;
        self.restore_prefix(request, slot)
    }
}

impl Bundle for Instance {
    fn instances() -> &'static LocalKey<RefCell<InstanceTable<Self>>> {
        &INSTANCES
    }

    fn create(services: HostServicesV1, configuration: &[u8]) -> Result<Self, Failed<Self>> {
        create(services, configuration)
    }

    fn resources(&mut self) -> (&HostServicesV1, &mut Vec<Held>) {
        (&self.services, &mut self.held)
    }

    fn release_graphs(&mut self) -> Result<(), BundleAbiError> {
        self.graphs.release_all(&self.services)?;
        match self.drafter.as_mut() {
            Some(runtime) => runtime.release_graphs(&self.services),
            None => Ok(()),
        }
    }

    fn execute(&mut self, request: &[u8], sink: &mut ExecuteResultSinkV1) -> BundleStatusV1 {
        self.execute_step(request, sink)
    }

    fn commit(&mut self) -> BundleStatusV1 {
        transaction::commit(self)
    }

    fn commit_accepted(
        &mut self,
        records: &[runtime_bundle_abi::BundleKeptRowsV1],
    ) -> BundleStatusV1 {
        transaction::commit_accepted(self, records)
    }

    fn cancel(&mut self) -> BundleStatusV1 {
        transaction::cancel(self)
    }

    fn retire_request(&mut self, request_id: u64) -> BundleStatusV1 {
        transaction::retire(self, request_id)
    }
}

/// Commits the open step, keeping each verify item's named rows and every
/// other item whole, or cancels it (restoring the rings it saved). Ledger
/// refusal is read-only. Success closes consistent state; failures during
/// history/checkpoint copies, retained-row uploads/copies, ring restoration
/// or tap finalization are terminal through the ABI client.
impl Steps for Instance {
    const RETIRE: Retire = Retire::UnlessHeld;

    fn ledger(&mut self) -> &mut Ledger {
        &mut self.ledger
    }

    fn status(&self, failure: Failure) -> BundleStatusV1 {
        match failure {
            Failure::VerifyItems | Failure::NotOpen | Failure::StepOpen => {
                BundleStatusV1::InvalidState
            }
            Failure::Ledger(Stage::Retire, _) => BundleStatusV1::InvalidState,
            Failure::Ledger(_, LedgerError::StalePlan) => BundleStatusV1::InternalError,
            Failure::Ledger(Stage::Plan, LedgerError::NotOpen | LedgerError::Open) => {
                BundleStatusV1::InvalidState
            }
            Failure::Ledger(Stage::Plan, _) => BundleStatusV1::InvalidArgument,
            Failure::Ledger(Stage::Apply | Stage::Cancel, _) => BundleStatusV1::InvalidState,
        }
    }

    fn begin(&mut self, _: End) -> Result<(), BundleStatusV1> {
        match self.open {
            Some(_) => Ok(()),
            None => Err(BundleStatusV1::InvalidState),
        }
    }

    fn prepare(
        &mut self,
        _: &[(u64, u32)],
        plan: &bundle_runtime::ledger::CommitPlan,
    ) -> Result<(), BundleStatusV1> {
        let Some(open) = self.open.clone() else {
            return Err(BundleStatusV1::InvalidState);
        };
        if open.cancel_only {
            return Err(BundleStatusV1::InvalidState);
        }
        // Keep the step until success. A host failure may leave partial state;
        // the ABI client poisons the instance and must not retry this callback.
        let step = self.step.clone();
        let result = self
            .commit_ple(&open, step.as_ref())
            .and_then(|()| self.commit_speculation(step.as_ref(), &open, plan));
        self.host_failure(End::Commit, result)
    }

    fn restore(&mut self, _: &bundle_runtime::ledger::Step) -> Result<(), BundleStatusV1> {
        let Some(open) = self.open.as_ref() else {
            return Err(BundleStatusV1::InvalidState);
        };
        let result = self.restore_ring(open);
        self.host_failure(End::Cancel, result)
    }

    fn ended(&mut self, _: Ended<'_>) -> Result<(), BundleStatusV1> {
        self.step = None;
        self.open = None;
        Ok(())
    }

    fn finish_taps(&mut self, commit: bool) -> BundleStatusV1 {
        Instance::finish_taps(self, commit)
    }

    fn retired(&mut self, request: u64) {
        self.tap_occurrences.retire(request);
    }
}

impl Instance {
    /// Logs a commit's or cancel's host failure.
    fn host_failure(
        &self,
        end: End,
        result: Result<(), BundleStatusV1>,
    ) -> Result<(), BundleStatusV1> {
        result.map_err(|error| {
            let _ = self.services.log(
                format!(
                    "Qwen3.8 {} failed: {error:?}",
                    if end.commits() { "commit" } else { "cancel" }
                )
                .as_bytes(),
            );
            BundleStatusV1::HostError
        })
    }
}

#[cfg(test)]
pub(crate) fn pending_step(handle: BundleHandleV1) -> Option<bool> {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .map(|instance| instance.ledger.step().is_some())
    })
}

#[cfg(test)]
pub(crate) fn owner(handle: BundleHandleV1, slot: u32) -> Option<u64> {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .and_then(|instance| instance.ledger.holder(slot))
            .map(|(request, _)| request)
    })
}

/// The full layers of the pages and ring a package of KV layout
/// `layout_id` declares, for the declared-state tests.
#[cfg(test)]
pub(crate) fn kv_layers(layout_id: &[u8]) -> Option<u64> {
    Layout::of(Shape::TP4, layout_id).map(|layout| layout.kv_layers)
}

/// Rank `rank`'s convolution, delta-rule and PLE pools and its live ring,
/// for the prefix snapshot tests.
#[cfg(test)]
pub(crate) fn state_views(handle: BundleHandleV1, rank: usize) -> [DeviceBufferSpanV1; 5] {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("live instance");
        let scratch = &instance.scratch[rank];
        let ring = instance.live_ring(rank);
        [
            scratch.conv_pool,
            scratch.delta_pool,
            scratch.ple_pool,
            ring,
            scratch.drafter.map_or(ring, |drafter| drafter.retained),
        ]
    })
}

/// Every launch of one rank's decode program at `bucket`, for the launch
/// plan tests.
#[cfg(test)]
pub(crate) fn plan_probe(
    handle: BundleHandleV1,
    phase: &PhaseInputs,
    rank: usize,
) -> Result<Vec<PreparedLaunch>, BundleStatusV1> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("live instance");
        let plan = PhasePlan::new(instance.shape, phase, instance.items)?
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let index = usize::from(phase.decode);
        let (layout, side) = instance.phase_layouts(phase);
        let device = PhaseDevice {
            inputs: DeviceInputs::within(instance.arenas[index][rank], &layout),
            side: SideInputs::within(instance.side_arenas[index][rank], &side),
        };
        instance
            .plan_rank(&plan, device, rank)
            .map(|program| program.launches)
    })
}

/// Per rank: its stream, the buffer holding the last program's final X, and
/// each phase's PLE staging input, for the fake host's device model.
#[cfg(test)]
pub(crate) fn model_spans(
    handle: BundleHandleV1,
) -> Vec<(u32, [DeviceBufferSpanV1; 2], [DeviceBufferSpanV1; 3])> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("live instance");
        (0..instance.shape.ranks.count())
            .map(|rank| {
                let scratch = &instance.scratch[rank];
                (
                    instance.streams[rank].slot(),
                    [scratch.x[0], scratch.x[crate::lower::TARGET_X]],
                    [
                        scratch.ple_staging[0].input,
                        scratch.ple_staging[1].input,
                        instance
                            .overlap
                            .as_ref()
                            .map_or(scratch.ple_staging[0].input, |overlap| {
                                overlap.histories[rank]
                            }),
                    ],
                )
            })
            .collect()
    })
}

/// The two dedicated overlap input arenas per rank, for the CPU model.
#[cfg(test)]
pub(crate) fn model_overlap_arenas(handle: BundleHandleV1) -> Vec<[u32; 2]> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("live instance");
        if instance.overlap.is_none() {
            return Vec::new();
        }
        (0..instance.shape.ranks.count())
            .map(|rank| HALVES.map(|part| instance.arenas[part][rank].arena.buffer.slot()))
            .collect()
    })
}

/// Per rank: `local_logits`, `local_scored`, `gathered_scored` and
/// `gathered_decode`, the logits buffers the heads and gathers bind.
#[cfg(test)]
pub(crate) fn logits_spans(handle: BundleHandleV1) -> Vec<[DeviceBufferSpanV1; 4]> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("live instance");
        instance
            .scratch
            .iter()
            .map(|scratch| {
                [
                    scratch.local_logits,
                    scratch.local_scored,
                    scratch.gathered_scored,
                    scratch.gathered_decode,
                ]
            })
            .collect()
    })
}

/// The physical entry of `slot` and entries per pool plane, including
/// after cancellation invalidates the logical owner.
#[cfg(test)]
pub(crate) fn committed_entry(handle: BundleHandleV1, slot: u32) -> Option<(i32, u64)> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("live instance");
        // A logical slot keeps its physical entry across retirement and reuse.
        if slot >= instance.served.slots {
            return None;
        }
        Some((
            instance.linear_entries.committed(slot),
            instance.speculation.pool_entries(instance.served.slots),
        ))
    })
}

#[cfg(test)]
pub(crate) fn ple_entry(handle: BundleHandleV1, slot: u32) -> i32 {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("live instance")
            .ple_entries
            .committed(slot)
    })
}

/// Rank 0's pools: linear convolution, delta rule and PLE.
#[cfg(test)]
pub(crate) fn pools(handle: BundleHandleV1) -> [DeviceBufferSpanV1; 3] {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let scratch = &table.get_mut(handle).expect("live instance").scratch[0];
        [scratch.conv_pool, scratch.delta_pool, scratch.ple_pool]
    })
}

/// Taps `taps` from the next step on, as a create with them would.
#[cfg(test)]
pub(crate) fn set_taps(handle: BundleHandleV1, taps: runtime_bundle_sdk::taps::Taps) {
    assert_eq!(
        INSTANCES.with(|table| {
            bundle_runtime::table::with_instance(table, handle, |instance| {
                instance.dump = Some(taps);
                BundleStatusV1::Success
            })
        }),
        BundleStatusV1::Success
    );
}

/// Changes the live instance's encoder settings (capture, mode).
#[cfg(test)]
pub(crate) fn with_encoder(
    handle: BundleHandleV1,
    change: impl FnOnce(&mut crate::encoder::Encoder),
) {
    assert_eq!(
        INSTANCES.with(|table| {
            bundle_runtime::table::with_instance(table, handle, |instance| {
                change(instance.encoder.as_mut().unwrap());
                BundleStatusV1::Success
            })
        }),
        BundleStatusV1::Success
    );
}

#[cfg(test)]
pub(crate) fn set_test_tail_window(handle: BundleHandleV1, window: Option<u64>) {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .unwrap()
            .ledger
            .set_test_tail_window(window);
    });
}

#[cfg(test)]
type TestScratchSuffix = (
    DeviceBufferSpanV1,
    DeviceBufferSpanV1,
    Option<DeviceBufferSpanV1>,
    DeviceBufferSpanV1,
);

#[cfg(test)]
pub(crate) fn test_scratch_suffix(handle: BundleHandleV1) -> Vec<TestScratchSuffix> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).unwrap();
        instance
            .scratch
            .iter()
            .zip(&instance.index_arenas)
            .map(|(scratch, index)| {
                (
                    scratch.prefix_pages,
                    scratch.route_ticket,
                    scratch.precision_temporary,
                    *index,
                )
            })
            .collect()
    })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum TestExchangeFault {
    Before,
    After,
    StalePle { slot: u32, checkpoint: i32 },
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TestEntryState {
    pub ledger_open: bool,
    pub linear_slots: Vec<i32>,
    pub linear_workspace: Vec<i32>,
    pub ple_slots: Vec<i32>,
    pub ple_workspace: Vec<i32>,
}

#[cfg(test)]
thread_local! {
    static TEST_EXCHANGE_FAULT: std::cell::Cell<Option<TestExchangeFault>> = const { std::cell::Cell::new(None) };
    static TEST_EXCHANGE_OBSERVED: RefCell<Option<TestEntryState>> = const { RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_test_exchange_fault(fault: TestExchangeFault) {
    TEST_EXCHANGE_FAULT.set(Some(fault));
    TEST_EXCHANGE_OBSERVED.with(|observed| *observed.borrow_mut() = None);
}

#[cfg(test)]
pub(crate) fn test_exchange_observation() -> Option<TestEntryState> {
    TEST_EXCHANGE_FAULT.set(None);
    TEST_EXCHANGE_OBSERVED.with(|observed| observed.borrow_mut().take())
}

#[cfg(test)]
pub(crate) fn test_entry_state(handle: BundleHandleV1) -> TestEntryState {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .unwrap()
            .test_entry_state()
    })
}

#[cfg(test)]
impl Instance {
    fn test_entry_state(&self) -> TestEntryState {
        let linear = if self.speculation.verify_rows > 1 {
            self.served.slots.min(DRAFT_ITEMS) * self.speculation.verify_rows
        } else {
            0
        };
        let ple = if self.speculation.verify_rows > 0 {
            self.served.items * MAX_VERIFY_ROWS
        } else {
            0
        };
        TestEntryState {
            ledger_open: self.ledger.step().is_some(),
            linear_slots: (0..self.served.slots)
                .map(|slot| self.linear_entries.committed(slot))
                .collect(),
            linear_workspace: (0..linear)
                .map(|index| self.linear_entries.checkpoint(index as usize))
                .collect(),
            ple_slots: (0..self.served.slots)
                .map(|slot| self.ple_entries.committed(slot))
                .collect(),
            ple_workspace: (0..ple)
                .map(|index| self.ple_entries.checkpoint(index as usize))
                .collect(),
        }
    }

    fn test_before_entry_exchanges(&mut self) -> Result<(), BundleStatusV1> {
        let Some(fault) = TEST_EXCHANGE_FAULT.get() else {
            return Ok(());
        };
        if let TestExchangeFault::StalePle { slot, checkpoint } = fault {
            self.ple_entries
                .prepare(&[(slot, checkpoint)])
                .unwrap()
                .commit(&mut self.ple_entries)
                .unwrap();
        }
        if !matches!(fault, TestExchangeFault::After) {
            TEST_EXCHANGE_OBSERVED
                .with(|observed| *observed.borrow_mut() = Some(self.test_entry_state()));
        }
        if matches!(fault, TestExchangeFault::Before) {
            Err(BundleStatusV1::InternalError)
        } else {
            Ok(())
        }
    }

    fn test_after_entry_exchanges(&self) -> Result<(), BundleStatusV1> {
        if matches!(TEST_EXCHANGE_FAULT.get(), Some(TestExchangeFault::After)) {
            TEST_EXCHANGE_OBSERVED
                .with(|observed| *observed.borrow_mut() = Some(self.test_entry_state()));
            Err(BundleStatusV1::InternalError)
        } else {
            Ok(())
        }
    }
}
