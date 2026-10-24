//! The `glm5next.v1` instance behind the shared ABI entry bodies.
//!
//! Create acquires, in order: the rank contexts, the planned text image and
//! its rank indexes (authenticated by the plan's identity), the placed
//! views, peer access, each rank's phase input arenas and scratch, the
//! host's per-item indexer tail state, the embedded modules, the KDA state
//! pools of the declared recurrent state layouts (with their checkpoints
//! and anchor spares) and the serving KV pool, which the host sizes from
//! what they leave.
//!
//! Execute runs a step through the ledger (`execute`): its eager prefill
//! (replayed from its captured programs once its exact shape repeats,
//! `prefill_graphs`), then its decode or verify phase as captured graphs
//! per bucket, refusing until every stage of the program lowers. The drafter follows. Create
//! serves context-parallel sizes 2 and 4 (`crate::context_parallel::SERVED`).

mod execute;
mod prefill_graphs;
mod prefix;

#[cfg(test)]
pub(crate) use prefill_graphs::last_chunks_in_tests;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::thread::LocalKey;

use bundle_runtime::acquire::{self, Log, RankHandles};
use bundle_runtime::drafter::Pending;
use bundle_runtime::entry::{Bundle, Failed};
use bundle_runtime::geometry::Items;
use bundle_runtime::graphs::GraphCache;
use bundle_runtime::inputs::VerifyLimits;
use bundle_runtime::ledger::{CommitPlan, Ledger, Placement, Rules};
use bundle_runtime::pipeline::Collectives;
use bundle_runtime::resources::{Held, allocate_span, status};
use bundle_runtime::state::Extent;
use bundle_runtime::table::{InstanceTable, Live};
use bundle_runtime::transaction::{self, End, Failure, Retire, Stage, Steps};
use bundle_runtime::upload::{BlockTableMirror, InputLayout, PhaseArena};
use model_glm5next_contract::contract::{Glm, INDEXER_TAIL_STATE_ID, MAX_VERIFY_ROWS, Shape};
use model_glm5next_contract::direct::{INDEX_SCHEMA, RANK_IMAGE_ROLE, index_role};
use runtime_bundle_abi::{
    BundleAbiError, BundleCudaSymbolHandleV1, BundleKeptRowsV1, BundleStatusV1,
    ExecuteResultSinkV1, HostServicesV1,
};

use crate::buffers::{CommState, RankBuffers, StatePools, allocate_scratch_named};
use crate::config::{Configuration, decode};

const NAME: &str = "GLM 5.3 Flash";

thread_local! {
    static INSTANCES: RefCell<InstanceTable<Instance>> = const { RefCell::new(InstanceTable::new()) };
}

pub struct Instance {
    services: HostServicesV1,
    held: Vec<Held>,
    collectives: Collectives,
    config: Configuration,
    ranks: Vec<RankHandles>,
    /// Per rank, every embedded entry's symbol.
    symbols: Vec<BTreeMap<String, BundleCudaSymbolHandleV1>>,
    /// Per phase (prefill, decode, verify) and rank, the shared input
    /// arena and its block table.
    arenas: [Vec<PhaseArena>; 3],
    buffers: Vec<RankBuffers>,
    kv_pages: u64,
    ledger: Ledger,
    verify: VerifyLimits,
    /// The prefill, decode and verify phases' block tables as the device
    /// holds them.
    mirrors: [BlockTableMirror<Glm>; 3],
    /// Per rank, the inputs of an eager prefill's indexer pieces.
    piece_arenas: Vec<runtime_bundle_abi::DeviceBufferSpanV1>,
    /// The GPU target the instance runs on.
    target: std::sync::Arc<dyn crate::target::Target>,
    /// The dynamic shared memory of each entry the instance's modules hold.
    shared_bytes: BTreeMap<String, u32>,
    /// The ops prepared on the target's backend (`crate::ops`), shared by
    /// the instances of one plan.
    ops: std::sync::Arc<crate::ops::Ops>,
    /// Run the lowered stages alone, for the fake-host tests of the step.
    partial: bool,
    /// Each rank's decode and verify graphs by bucket.
    graphs: GraphCache<Glm>,
    /// The target's and the drafter's captured eager prefill programs, by
    /// their exact shape.
    prefill_graphs: prefill_graphs::PrefillGraphs,
    /// The context-parallel groups' state above size 1.
    context_parallel: Option<crate::context_parallel::ContextParallel>,
    /// The MTP drafter's rows per slot, when the instance runs it (an MTP
    /// KV format with speculation enabled), and the open step's commit.
    drafter: Option<crate::drafter::Drafter>,
    pending: Option<Pending>,
    /// Per drafter phase (prefill, decode, verify) and rank, its input
    /// arena, and the block tables as the device holds them.
    draft_arenas: [Vec<PhaseArena>; 3],
    draft_mirrors: [BlockTableMirror<Glm>; 3],
    /// Each rank's drafter graphs: the extend pass's by verify bucket, the
    /// draft step's by item bucket.
    draft_graphs: [GraphCache<Glm>; 2],
    /// The vision encoder, when the create record names its components.
    encoder: Option<bundle_runtime::encoder::Encoder>,
    /// The projections' stores, when the package carries their precision
    /// component.
    projections: Option<crate::projections::Projections>,
    /// Bring-up diagnostics (`SYNC_STAGES`): every rank synchronizes after
    /// each eager segment and graph replay, logging what it waits for.
    sync_stages: bool,
    /// The encoder and text taps a diagnostic build writes
    /// (`crate::taps`).
    taps: crate::taps::StepTaps,
    /// Which decode and verify graph buckets run each MoE sublayer's shared
    /// gate/up on a side stream (`crate::shared_overlap`).
    shared_overlap: crate::shared_overlap::Settings,
    /// Each rank's side stream and events; none without the overlap.
    side_lanes: Vec<crate::shared_overlap::Lanes>,
    /// Private allocation owners and inboxes for prefill's F32 MoE DMA.
    dma: Vec<bundle_runtime::dma::Rank>,
    prefill_dma: bool,
    /// Runs in the phased all-reduce's scope issued since every rank's
    /// phased flags and epoch were last zeroed, and the count at which the
    /// next such run zeroes them first (`Target::phased_reset_runs`; tests
    /// lower it).
    phased_runs: u64,
    phased_reset_runs: u64,
    /// Commit's copy of each checkpointing buffer's entry in the layers it
    /// copies, by the buffer's index (`step::checkpoint_layers`), prepared
    /// on the own kernels once the request state is allocated.
    state_copies: StateCopies,
}

/// The GPU target of an instance of `shape`: SM120's.
pub(crate) fn target(shape: Shape) -> std::sync::Arc<dyn crate::target::Target> {
    std::sync::Arc::new(crate::sm120::own_ops::Sm120Own::new(shape))
}

/// Commit's state copies: each checkpointing buffer's index, its copy and
/// the op prepared for it.
type StateCopies = Vec<(
    usize,
    model_glm5next_contract::own_ops::state::StateCopy,
    kernel_core::backend::Prepared<
        model_glm5next_contract::own_ops::state::StateCopyBuffers<kernel_core::backend::View>,
    >,
)>;

/// The bring-up switch that makes every rank synchronize after each
/// segment of an eager run (its launches up to a collective) and each
/// graph replay, logging the segment's stages, the rank and the wait
/// before and after it, so a hang names its stage. Off unless `1`; it
/// serializes the ranks' work, so it is for diagnosis only.
pub const SYNC_STAGES: &str = "CHOPINFER_GLM5NEXT_SYNC_STAGES";

/// The diagnostic that lowers the phased all-reduce's reset bound
/// (`Target::phased_reset_runs`) to a positive number of runs at serve
/// start, so a short run zeroes every rank's phased state while idle as
/// often as it likes and shows the served bits unchanged by it. Values
/// above the bound, zero and non-numbers keep the bound.
pub const PHASED_RESET_RUNS: &str = "CHOPINFER_GLM5NEXT_PHASED_RESET_RUNS";

impl Live for Instance {
    fn collectives(&self) -> &Collectives {
        &self.collectives
    }
}

/// Verify rows per item the host's speculation asks for (the drafts and
/// the bonus row; none without speculation, as the request state's extent
/// counts them), and the verify limits of an instance serving `items`.
fn verify_rows(
    services: &HostServicesV1,
    items: Items,
) -> Result<(u32, VerifyLimits), BundleStatusV1> {
    let speculation = services.query_speculation().map_err(status)?;
    let rows = match speculation.maximum_draft_tokens {
        0 => 0,
        drafts => drafts + 1,
    };
    if rows > MAX_VERIFY_ROWS {
        return Err(BundleStatusV1::Unsupported);
    }
    let limits = VerifyLimits::enabled_within::<Glm>(
        speculation.maximum_draft_tokens,
        speculation.maximum_verify_tokens,
        items,
    );
    Ok((rows, limits))
}

impl Instance {
    fn acquire(&mut self, verify_rows: u32) -> Result<(), BundleAbiError> {
        let services = &self.services;
        let held = &mut self.held;
        let log = Log {
            services,
            bundle: NAME,
        };
        let shape = self.config.shape;
        let create = &self.config.create;
        let items = create.items;
        // Every rank's index, admitted against this build's plan and bound
        // to the image the host serves for that rank, before any rank
        // context opens or the image is opened or uploaded.
        let layout = self.config.layout;
        let admit = |rank: u32, bytes: &[u8]| {
            let index =
                bundle_runtime::index::parse(bytes, rank, INDEX_SCHEMA).map_err(|error| {
                    let _ = services.log(format!("{NAME} rank index: {error}").as_bytes());
                    BundleAbiError::InvalidRange("glm5next rank index")
                })?;
            // Every view this build binds for the package's checkpoint
            // layout, where its plan put it.
            let expected = crate::rank_layout::layout(shape, layout, rank).map_err(|error| {
                let _ = services.log(error.as_bytes());
                BundleAbiError::InvalidRange("glm5next rank layout")
            })?;
            if expected.len() != index.views.len()
                || expected.iter().any(|entry| {
                    index.views.get(&entry.view.name).is_none_or(|view| {
                        view.offset != entry.offset
                            || view.bytes != entry.bytes()
                            || view.layout_id != entry.layout_id
                    })
                })
            {
                return Err(BundleAbiError::InvalidRange(
                    "glm5next rank index differs from this build's plan",
                ));
            }
            Ok((index, expected))
        };
        let images = acquire::RankImages {
            role: RANK_IMAGE_ROLE.as_bytes(),
            index_role: &index_role,
            max_index_bytes: 64 << 20,
        };
        let (image, indexes) = acquire::rank_images(
            log,
            images,
            create.ordinals.len() as u32,
            admit,
            |(index, _)| acquire::identity(index),
            held,
        )?;
        let ranks = acquire::open_ranks(log, &create.ordinals, held)?;
        let rank_arenas = acquire::upload_arenas(log, image, &ranks, held)?;
        let mut views = Vec::new();
        let mut stored = Vec::new();
        for (arena, (index, expected)) in rank_arenas.into_iter().zip(indexes) {
            let placed =
                acquire::place_views(services, image, arena, &index, &[]).map_err(|error| {
                    match error {
                        acquire::PlaceError::Host(error) => error,
                        acquire::PlaceError::Layout(message) => {
                            let _ = services.log(message.as_bytes());
                            BundleAbiError::InvalidRange("glm5next rank arena length")
                        }
                    }
                })?;
            views.push(acquire::spans(placed));
            stored.push(Vec::clone(&expected));
        }
        acquire::grant_peers(services, &ranks, held)?;
        if self.prefill_dma {
            self.dma = bundle_runtime::dma::acquire(
                services,
                &ranks,
                u64::from(model_glm5next_contract::contract::PREFILL_TOKENS)
                    * u64::from(shape.profile.hidden),
                held,
            )?;
        }
        let layouts = [
            InputLayout::prefill_capacity_within::<Glm>(items),
            InputLayout::decode_within(items).end,
            InputLayout::verify_within::<Glm>(items).end,
        ];
        let tables = BlockTableMirror::<Glm>::device_bytes(shape.model_limits(), items);
        let format = self.config.format;
        let extent = Extent {
            served: items.get(),
            bucket: items.capacity(),
            verify_rows,
        };
        let mut scratch = Vec::new();
        let mut comms = Vec::new();
        let cp_size = self.config.context_parallel.size();
        let mut cp_states = Vec::new();
        let mut drafters = Vec::new();
        for (rank, &(_, pool, stream)) in ranks.iter().enumerate() {
            for (phase, bytes) in layouts.into_iter().enumerate() {
                let arena = allocate_span(services, pool, stream, bytes, held)?;
                let block_tables = allocate_span(services, pool, stream, tables, held)?;
                self.arenas[phase].push(PhaseArena {
                    arena,
                    block_tables,
                });
            }
            if self.drafter.is_some() {
                for (phase, bytes) in layouts.into_iter().enumerate() {
                    let arena = allocate_span(services, pool, stream, bytes, held)?;
                    let block_tables = allocate_span(services, pool, stream, tables, held)?;
                    self.draft_arenas[phase].push(PhaseArena {
                        arena,
                        block_tables,
                    });
                }
            }
            let pieces = u64::from(execute::MAX_PIECES) * execute::piece_stride(items);
            self.piece_arenas
                .push(allocate_span(services, pool, stream, pieces, held)?);
            let buffers =
                allocate_scratch_named(shape, self.drafter.is_some(), |buffer, bytes| {
                    let owner = if matches!(
                        buffer,
                        crate::buffers::Buffer::Partial | crate::buffers::Buffer::Reduced
                    ) {
                        self.dma.get(rank).map_or(stream, |dma| dma.owner)
                    } else {
                        stream
                    };
                    allocate_span(services, pool, owner, bytes, held)
                })?
                .map_err(|error| {
                    let _ = services.log(error.as_bytes());
                    BundleAbiError::InvalidRange("glm5next scratch")
                })?;
            // The split mixer's tickets start at zero; each launch leaves
            // them zero.
            services.zero_device_buffer(stream, buffers[&crate::buffers::Buffer::MixTickets])?;
            // The shared expert's zero residual in the decode and verify
            // graphs, never written.
            services.zero_device_buffer(stream, buffers[&crate::buffers::Buffer::ZeroRows])?;
            scratch.push(buffers);
            // The drafter's staging and its retained rows (request state
            // the KV sizing does not reserve).
            drafters.push(match self.drafter {
                Some(_) => {
                    let retained = StatePools::retained_bytes(shape, &format, extent)
                        .ok_or(BundleAbiError::InvalidRange("glm5next retained rows"))?;
                    let retained = allocate_span(services, pool, stream, retained, held)?;
                    services.zero_device_buffer(stream, retained)?;
                    let staging = crate::buffers::DrafterBuffers::allocate(shape, |bytes| {
                        allocate_span(services, pool, stream, bytes, held)
                    })?;
                    // The live width of the logits shard the drafter's
                    // argmax reads: the whole shard.
                    services.copy_host_to_device(
                        stream,
                        staging.vocab_width,
                        &shape.profile.vocab_rows.to_le_bytes(),
                    )?;
                    services.copy_host_to_device(
                        stream,
                        staging.copy_rows,
                        &crate::drafter::copies::counts(),
                    )?;
                    Some((staging, retained))
                }
                None => None,
            });
            // The all-reduce state peers write: after peer access.
            let reduced = scratch
                .last()
                .map(|buffers: &BTreeMap<_, _>| buffers[&crate::buffers::Buffer::Reduced])
                .ok_or(BundleAbiError::InvalidRange("glm5next scratch"))?;
            let regions = self.ops.comm_regions();
            let draft = self.drafter.is_some().then(|| self.ops.draft_regions());
            let comm = CommState::allocate(shape, draft, reduced, regions, |bytes| {
                allocate_span(services, pool, stream, bytes, held)
            })?
            .map_err(|error| {
                let _ = services.log(error.as_bytes());
                BundleAbiError::InvalidRange("glm5next comm state")
            })?;
            for span in comm.zeroed() {
                services.zero_device_buffer(stream, span)?;
            }
            comms.push(comm);
            // The context-parallel state the group writes: after peer access.
            if cp_size > 1 {
                // The contract's capacity plan counts the same bytes.
                let bytes = model_glm5next_contract::context_parallel::rank_bytes(
                    model_glm5next_contract::context_parallel::Dims::of(shape),
                    cp_size,
                )
                .map_err(|error| {
                    let _ = services.log(error.as_bytes());
                    BundleAbiError::InvalidRange("glm5next context-parallel state")
                })?;
                let state = crate::context_parallel::RankState::allocate(&bytes, |bytes| {
                    allocate_span(services, pool, stream, bytes, held)
                })?;
                for span in state.zeroed() {
                    services.zero_device_buffer(stream, span)?;
                }
                cp_states.push(state);
            }
        }
        self.context_parallel = (cp_size > 1).then_some(crate::context_parallel::ContextParallel {
            size: cp_size,
            ranks: cp_states,
        });
        let tails = acquire::zeroed_state(
            services,
            INDEXER_TAIL_STATE_ID.as_bytes(),
            create.state.bytes,
            &ranks,
            held,
        )?;
        let shared = &self.shared_bytes;
        let uses = crate::target::Uses {
            layout: self.config.layout,
            context_parallel: cp_size > 1,
            comm_epilogue: self.config.comm_epilogue,
            drafter: self.drafter.is_some(),
            prefill_dma: self.prefill_dma,
        };
        for &(context, _, _) in &ranks {
            let mut resolved = BTreeMap::new();
            // The modules the target loads for what the instance runs.
            for cubin in crate::CUBINS
                .iter()
                .filter(|cubin| self.target.loads(cubin.name, uses))
            {
                acquire::load_module(
                    log,
                    context,
                    (cubin.name, cubin.bytes),
                    cubin.symbols.iter().copied(),
                    |symbol| {
                        shared
                            .get(symbol)
                            .copied()
                            .ok_or(BundleAbiError::InvalidRange("unplanned glm5next symbol"))
                    },
                    &mut resolved,
                    held,
                )?;
            }
            self.symbols.push(resolved);
        }
        // Every rank's scale views pass their payload admission.
        crate::payload::admit(
            services,
            self.ops.own.kernels(),
            &ranks,
            &self.symbols,
            &views,
            &stored,
            held,
        )?;
        // The low-latency all-reduce entries' CTAs spin on their
        // counterparts on the peer ranks, so they serve only when every
        // rank's device keeps each one's largest grid resident; otherwise
        // every rank reduces their rows through the phased one-shots
        // (`lower/reduce.rs`).
        match low_latency_admission(
            services,
            self.target.as_ref(),
            &create.ordinals,
            &self.symbols,
        ) {
            Ok(fitted) => {
                let _ =
                    services.log(format!("{NAME} low-latency all-reduce on: {fitted}").as_bytes());
            }
            Err(reason) => {
                let _ = services.log(
                    format!(
                        "{NAME} low-latency all-reduce off, the phased one-shot serves its rows: \
                         {reason}"
                    )
                    .as_bytes(),
                );
                for comm in &mut comms {
                    comm.low_latency = false;
                }
            }
        }
        // The phased all-reduce entries' CTAs spin likewise: they serve
        // only when every rank's device keeps their largest grids
        // resident; otherwise every rank reduces their rows through the
        // comm engine's composition.
        match phased_admission(
            services,
            self.target.as_ref(),
            &create.ordinals,
            &self.symbols,
        ) {
            Ok(fitted) => {
                let _ = services.log(format!("{NAME} phased all-reduce on: {fitted}").as_bytes());
            }
            Err(reason) => {
                let _ = services.log(
                    format!(
                        "{NAME} phased all-reduce off, the round row, the phased pair and the round \
                         row serve its rows: {reason}"
                    )
                    .as_bytes(),
                );
                for comm in &mut comms {
                    comm.phased = false;
                }
            }
        }
        // The vision encoder's cubins and working memory, before the KV
        // pool takes what is left.
        if self.config.media {
            let symbols = crate::encoders::load(self.target.as_ref(), log, &ranks, held)?;
            for (resolved, symbols) in self.symbols.iter_mut().zip(symbols) {
                resolved.extend(symbols);
            }
            let mut encoder =
                crate::encoders::acquire(self.target.as_ref(), services, &ranks, held)?;
            encoder.block_taps = self.taps.selects_blocks();
            self.encoder = Some(encoder);
        }
        // The projections' store, when the package carries its precision
        // component, and their scratch, before the KV pool.
        let role = crate::projections::COMPONENT_ROLE.as_bytes();
        if self.config.create.roles.iter().any(|named| named == role) {
            let bytes =
                acquire::read_component(log, crate::projections::COMPONENT_ROLE, 256 << 20, held)?;
            let invalid = |error: String| {
                let _ = services.log(format!("{NAME} precision component: {error}").as_bytes());
                BundleAbiError::InvalidRange("glm5next precision component")
            };
            // Every dense use is prepared on the instance's backend.
            let (backend, _) = self.target.backend(self.config.layout).map_err(invalid)?;
            let asks = crate::ops::Asks::default();
            let authenticated = crate::projections::authenticate(
                &bytes,
                self.config.layout,
                backend.as_ref(),
                &asks,
            )
            .map_err(invalid)?;
            let mut stores = Vec::with_capacity(ranks.len());
            let mut scratch = Vec::with_capacity(ranks.len());
            let mut side_scratch = Vec::with_capacity(ranks.len());
            for ((&(_, pool, stream), views), layout) in ranks.iter().zip(&views).zip(&stored) {
                stores.push(
                    crate::projections::bind(&authenticated, layout, views).map_err(invalid)?,
                );
                // The rank's scratch, then the side stream's.
                for scratch in [&mut scratch, &mut side_scratch] {
                    let rank =
                        crate::projections::RankScratch::allocate(&authenticated, |bytes| {
                            allocate_span(services, pool, stream, bytes, held)
                        })?;
                    for span in rank.zeroed() {
                        services.zero_device_buffer(stream, span)?;
                    }
                    scratch.push(rank);
                }
            }
            self.projections = Some(crate::projections::Projections {
                instances: crate::projections::instances(&authenticated).map_err(invalid)?,
                authenticated,
                stores,
                scratch,
                side_scratch,
                asked: asks.asked(),
            });
        }
        // The KDA pools, their checkpoints and every anchor spare, last
        // before the KV pool: the package declares its recurrent state
        // preallocated, so the host reserves nothing for it and sizes the
        // pool from what they leave.
        let mut states = Vec::with_capacity(ranks.len());
        for (rank, (drafter, tail)) in drafters.iter().zip(tails).enumerate() {
            let (_, pool, stream) = ranks[rank];
            let mut pools = Vec::new();
            let retained = drafter.map(|(_, retained)| retained);
            let state = StatePools::new(
                shape,
                &format,
                extent,
                self.verify.rows_per_step,
                retained,
                tail,
                |bytes| {
                    let span = allocate_span(services, pool, stream, bytes, held)?;
                    pools.push(span);
                    Ok::<_, BundleAbiError>(span)
                },
            )?
            .map_err(|error| {
                let _ = services.log(error.as_bytes());
                BundleAbiError::InvalidRange("glm5next request state")
            })?;
            for span in pools {
                services.zero_device_buffer(stream, span)?;
            }
            states.push(state);
        }
        let (pages, pools) =
            acquire::serving_kv(log, format.layout_id, format.page_bytes, shape.ranks())?;
        self.kv_pages = pages;
        for (rank, ((scratch, drafter), (views, (kv_pool, state)))) in scratch
            .into_iter()
            .zip(drafters)
            .zip(views.into_iter().zip(pools.into_iter().zip(states)))
            .enumerate()
        {
            self.buffers.push(RankBuffers {
                scratch,
                comm: comms[rank].clone(),
                state,
                views,
                kv_pool,
                drafter: drafter.map(|(staging, _)| staging),
            });
        }
        let own = self.ops.own.kernels();
        self.state_copies =
            Self::prepare_state_copies(own, &self.buffers[0].state).map_err(|refused| {
                let _ = services.log(format!("{NAME} state copies: {refused}").as_bytes());
                BundleAbiError::InvalidRange("glm5next state copies")
            })?;
        // The shared gate/up's side streams and events, last, so an instance
        // without the overlap holds what it did before it existed.
        if self.shared_overlap.on() {
            for &(context, _, stream) in &ranks {
                let side = bundle_runtime::lanes::SideLane::create(
                    services,
                    context,
                    crate::shared_overlap::EVENTS,
                    held,
                )?;
                let &[fork, join] = side.events.as_slice() else {
                    unreachable!("every event was created")
                };
                self.side_lanes.push(crate::shared_overlap::Lanes {
                    compute: stream,
                    side: side.stream,
                    fork,
                    join,
                });
            }
        }
        acquire::synchronize(log, &ranks)?;
        for dma in &self.dma {
            services.synchronize_stream(dma.owner)?;
        }
        self.ranks = ranks;
        Ok(())
    }
}

/// What each rank's device keeps resident of each low-latency all-reduce
/// entry's largest grid (`Target::low_latency`) when every one keeps them
/// resident for the ranks that share it (`bundle_runtime::residency`), else
/// why the entries must not serve.
///
/// Every CTA of the entry spins until its counterparts on the peer ranks
/// have sent, so a grid that is only partly resident on one device can hold
/// the others forever. Fails closed: a report the device cannot vouch for
/// (none, an SM-limited context, an MPS client), a shortfall or a failed
/// query turns the entry off on every rank. PDL: the entry triggers its
/// dependents first, so a dependent grid launches only once every CTA of
/// the entry has started and cannot take the slots it needs; the grid the
/// entry waits for never waits on it, so it finishes and frees its slots.
fn low_latency_admission(
    services: &HostServicesV1,
    target: &dyn crate::target::Target,
    ordinals: &[i32],
    symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
) -> Result<String, String> {
    let mut admitted = Vec::new();
    for entry in target.low_latency()? {
        admitted.push(spinning_admission(
            services,
            ordinals,
            symbols,
            &[entry.symbol],
            entry.plan,
        )?);
    }
    Ok(admitted.join("; "))
}

/// [`low_latency_admission`] of the phased all-reduce's served entries,
/// each at its largest plan (`Target::phased`).
fn phased_admission(
    services: &HostServicesV1,
    target: &dyn crate::target::Target,
    ordinals: &[i32],
    symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
) -> Result<String, String> {
    let mut admitted = Vec::new();
    for entry in target.phased()? {
        admitted.push(spinning_admission(
            services,
            ordinals,
            symbols,
            &[entry.symbol],
            entry.plan,
        )?);
    }
    Ok(admitted.join("; "))
}

/// What each rank's device keeps resident of each of `entries` launched at
/// `plan` when every one keeps the grids of the ranks that share it
/// resident, else why the entries must not serve.
fn spinning_admission(
    services: &HostServicesV1,
    ordinals: &[i32],
    symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
    entries: &[&str],
    plan: kernel_core::contract::LaunchPlan,
) -> Result<String, String> {
    use bundle_runtime::residency::{self, Verdict};
    let grid = plan.grid.iter().product::<u32>();
    let mut admitted = Vec::new();
    for &entry in entries {
        let mut fitted = Vec::new();
        for (rank, symbols) in symbols.iter().enumerate() {
            let Some(&symbol) = symbols.get(entry) else {
                return Err(format!("rank {rank} did not load {entry}"));
            };
            let report = match residency::query(services, symbol, plan) {
                Ok(report) => report,
                Err(error) => return Err(format!("rank {rank} {entry} residency: {error}")),
            };
            let sharing = residency::ranks_on_device(ordinals, rank);
            let reason = match residency::check(&plan, sharing, &report) {
                Verdict::Fits => {
                    fitted.push(format!(
                        "rank {rank} {} of {} CTAs",
                        u64::from(sharing) * u64::from(grid),
                        report.max_active_clusters
                    ));
                    continue;
                }
                Verdict::Unknown(reason) => reason,
                Verdict::Mps => "the process may be an MPS client",
                Verdict::Short => "the device keeps fewer CTAs resident than the grids need",
            };
            return Err(format!(
                "rank {rank}: {entry} ({grid} CTAs, {sharing} grid(s) on the device, {} \
                 resident, {} of {} SMs): {reason}",
                report.max_active_clusters,
                report.usable_multiprocessors,
                report.multiprocessor_count,
            ));
        }
        admitted.push(format!("{entry} needed of resident: {}", fitted.join(", ")));
    }
    Ok(admitted.join("; "))
}

fn create(services: HostServicesV1, bytes: &[u8]) -> Result<Instance, Failed<Instance>> {
    // Test builds record the encoder's launches beside the bundle's own.
    #[cfg(test)]
    crate::submit::observe_programs();
    let log = |reason: &str| {
        let _ = services.log(format!("{NAME} create failed: {reason}").as_bytes());
    };
    let config = decode(bytes, log)?;
    let prefill_dma = crate::prefill_dma::requested(&config);
    if prefill_dma && (config.context_parallel.size() != 1 || config.comm_epilogue) {
        log("prefill DMA currently requires CP1 without the comm epilogue");
        return Err(BundleStatusV1::Unsupported.into());
    }
    let taps = crate::taps::StepTaps::from_env(&config).map_err(|error| {
        log(&error);
        BundleStatusV1::InvalidArgument
    })?;
    if !crate::context_parallel::serves(config.context_parallel.size()) {
        log(
            "context-parallel attention awaits its GPU check; serve with --context-parallel-size 1",
        );
        return Err(BundleStatusV1::Unsupported.into());
    }
    let items = config.create.items;
    let (verify_rows, mut verify) = verify_rows(&services, items)?;
    // The MTP drafter runs over the KV formats with its layer, when the
    // host enables speculation.
    verify.bundle_drafter = model_glm5next_contract::contract::has_mtp_layer(&config.format)
        && verify.rows_per_item > 1;
    // The head scores at most this many rows per phase.
    bundle_runtime::config::check_scored_rows(&services, NAME, crate::buffers::SCORED_ROWS)?;
    let shape = config.shape;
    let state =
        model_glm5next_contract::state::request_state(shape, &config.format, verify.bundle_drafter);
    let target = target(shape);
    let phased_reset_runs = target.phased_reset_runs();
    let shared_bytes = target.shared_bytes(config.layout).map_err(|error| {
        log(&error);
        BundleStatusV1::InternalError
    })?;
    // The ops, prepared once per process on the target's backend.
    let plan = crate::ops::Plan {
        shape,
        layout: config.layout,
        index_topk: config.runtime.index_topk,
        context_parallel: config.context_parallel.size(),
    };
    let ops = crate::ops::Ops::shared(target.as_ref(), plan).map_err(|refused| {
        log(&refused.to_string());
        BundleStatusV1::Unsupported
    })?;
    let limits = shape.model_limits();
    let ledger = Ledger::new(
        items.get(),
        Placement::Host,
        Rules::of(&state),
        config
            .create
            .context_tokens
            .unwrap_or(shape.geometry.context_tokens()),
    );
    // One ledger owns every graph in capture order (the drafter's caches
    // and the captured prefill programs share it), so releasing it frees
    // them newest first, as the host requires.
    let graphs = GraphCache::with_items(shape.ranks, items);
    let draft_graphs = [GraphCache::sharing(&graphs), GraphCache::sharing(&graphs)];
    let mut instance = Instance {
        services,
        held: Vec::new(),
        collectives: Collectives::default(),
        config,
        ranks: Vec::new(),
        symbols: Vec::new(),
        arenas: Default::default(),
        buffers: Vec::new(),
        kv_pages: 0,
        ledger,
        verify,
        mirrors: [
            BlockTableMirror::new(limits, items),
            BlockTableMirror::new(limits, items),
            BlockTableMirror::new(limits, items),
        ],
        piece_arenas: Vec::new(),
        graphs,
        prefill_graphs: Default::default(),
        target,
        shared_bytes,
        ops,
        partial: false,
        context_parallel: None,
        drafter: verify
            .bundle_drafter
            .then(|| crate::drafter::Drafter::new(items.get())),
        pending: None,
        draft_arenas: Default::default(),
        draft_mirrors: [
            BlockTableMirror::new(limits, items),
            BlockTableMirror::new(limits, items),
            BlockTableMirror::new(limits, items),
        ],
        draft_graphs,
        encoder: None,
        projections: None,
        sync_stages: std::env::var_os(SYNC_STAGES).is_some_and(|value| value == "1"),
        taps,
        shared_overlap: crate::shared_overlap::Settings::from_env(),
        side_lanes: Vec::new(),
        dma: Vec::new(),
        prefill_dma,
        phased_runs: 0,
        phased_reset_runs,
        state_copies: Vec::new(),
    };
    if let Some(runs) = std::env::var(PHASED_RESET_RUNS)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&runs| runs > 0 && runs < instance.phased_reset_runs)
    {
        instance.phased_reset_runs = runs;
        instance.log(&format!(
            "{PHASED_RESET_RUNS}={runs}: every rank's phased all-reduce state is zeroed while idle \
             every {runs} decode and verify runs"
        ));
    }
    if instance.config.comm_epilogue {
        instance.log(&format!(
            "{}=true: the sublayer all-reduces run the Sinkhorn mHC comm epilogue",
            model_glm5next_contract::options::COMM_EPILOGUE_OPTION
        ));
    }
    if instance.shared_overlap.on() {
        let buckets = match instance.shared_overlap.max_rows {
            u32::MAX => "every decode and verify graph bucket runs".to_owned(),
            rows => format!("decode and verify graph buckets of at most {rows} rows run"),
        };
        instance.log(&format!(
            "{} on (the default; 0 turns it off): {buckets} each MoE sublayer's shared gate/up \
             on a side stream beside the routed experts",
            crate::shared_overlap::SWITCH,
        ));
    } else {
        instance.log(&format!(
            "{}=0: every decode and verify graph bucket runs its shared gate/up on one stream",
            crate::shared_overlap::SWITCH,
        ));
    }
    if instance.sync_stages {
        instance.log(&format!(
            "{SYNC_STAGES}=1: every rank synchronizes after each eager segment and graph \
             replay (bring-up diagnostics)"
        ));
    }
    if let Err(error) = instance.acquire(verify_rows) {
        let _ = instance
            .services
            .log(format!("{NAME} create failed: {error}").as_bytes());
        let result = status(error);
        return Err(Failed::releasing(instance, result));
    }
    let _ = instance.services.log(
        format!(
            "{NAME}: {} KV pages per rank, {} items",
            instance.kv_pages,
            instance.items().get()
        )
        .as_bytes(),
    );
    // The last chunks' programs, before any other graph (a context-parallel
    // group's, and any under the stage-sync or tap diagnostics, run as
    // their prefills do: captured at their first run, or eagerly).
    if prefill_graphs::last_chunks_from_env()
        && instance.config.context_parallel.size() == 1
        && !instance.sync_stages
        && !instance.taps.text()
    {
        let started = std::time::Instant::now();
        match instance.capture_last_chunks() {
            Ok(captured) => instance.log(&format!(
                "{}=1 (off by default): captured the programs of a request's last prompt chunk \
                 at {captured} lengths in {:.3}s",
                prefill_graphs::LAST_CHUNKS,
                started.elapsed().as_secs_f64()
            )),
            Err(error) => {
                // They capture at their first run instead.
                instance.log(&format!(
                    "capturing the last chunks' programs failed: {error}; releasing every graph"
                ));
                if let Err(error) = Bundle::release_graphs(&mut instance) {
                    instance.log(&format!("releasing every graph failed: {error}"));
                }
            }
        }
    }
    Ok(instance)
}

impl Instance {
    fn items(&self) -> Items {
        self.config.create.items
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

    /// Releases every graph through the ledger they share, newest first;
    /// a graph whose release fails stays owned for the next call, which
    /// the prefill store owes until one succeeds.
    fn release_graphs(&mut self) -> Result<(), BundleAbiError> {
        if !self.dma.is_empty() {
            // Peer-copy owner joins do not cover subsequent compute reads.
            // A failed drain must leave all private buffers and graphs owned.
            for stream in self
                .ranks
                .iter()
                .map(|r| r.2)
                .chain(self.side_lanes.iter().map(|lanes| lanes.side))
            {
                self.services.synchronize_stream(stream)?;
            }
        }
        // The drafter's caches and the prefill programs see the shared
        // ledger's new epoch as released.
        self.prefill_graphs.forget();
        let released = self.graphs.release_all(&self.services);
        if released.is_err() {
            self.prefill_graphs.owe_release();
        }
        released
    }

    fn execute(&mut self, request: &[u8], sink: &mut ExecuteResultSinkV1) -> BundleStatusV1 {
        self.execute_step(request, sink)
    }

    /// A step with verify items commits through `commit_accepted`.
    fn commit(&mut self) -> BundleStatusV1 {
        transaction::commit(self)
    }

    fn commit_accepted(&mut self, records: &[BundleKeptRowsV1]) -> BundleStatusV1 {
        transaction::commit_accepted(self, records)
    }

    /// Decode and verify advance the KDA and tail state in place, so the
    /// ledger retires the requests the step touched.
    fn cancel(&mut self) -> BundleStatusV1 {
        transaction::cancel(self)
    }

    fn retire_request(&mut self, request_id: u64) -> BundleStatusV1 {
        transaction::retire(self, request_id)
    }
}

/// Commits the open step, copying each verify item's last kept checkpoint
/// into its slot and committing the drafter, or cancels it. A ledger
/// refusal changes nothing; a failed copy or tap finish is terminal through
/// the ABI client.
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
            Failure::Ledger(Stage::Plan, _) => BundleStatusV1::InvalidArgument,
            Failure::Ledger(Stage::Apply | Stage::Cancel | Stage::Retire, _) => {
                BundleStatusV1::InvalidState
            }
        }
    }

    fn begin(&mut self, end: End) -> Result<(), BundleStatusV1> {
        if end == End::Cancel {
            // The drafter keeps its frontier: rows a cancelled step wrote
            // past it are rewritten when it reaches them.
            self.pending = None;
        }
        Ok(())
    }

    fn prepare(&mut self, kept: &[(u64, u32)], plan: &CommitPlan) -> Result<(), BundleStatusV1> {
        self.prepare_commit(kept, plan)
    }

    /// Publishes or removes the step's taps (`crate::taps`).
    fn finish_taps(&mut self, commit: bool) -> BundleStatusV1 {
        match self.taps.finish(commit) {
            Ok(()) => BundleStatusV1::Success,
            Err(error) => {
                self.log(&format!("taps: {error}"));
                BundleStatusV1::InternalError
            }
        }
    }

    /// Taps retire the request even when the ledger refuses it.
    fn before_retire(&mut self, request: u64) -> Result<(), BundleStatusV1> {
        self.taps.retire(request);
        Ok(())
    }
}

/// Runs `f` on the live instance `handle`, for the fake-host tests.
#[cfg(test)]
fn with_instance<R>(
    handle: runtime_bundle_abi::BundleHandleV1,
    f: impl FnOnce(&mut Instance) -> R,
) -> Option<R> {
    INSTANCES.with(|table| table.borrow_mut().get_mut(handle).map(f))
}

/// Panics once a rank's share of the next unit is queued.
#[cfg(test)]
pub(crate) use execute::hooks::{
    Overlapped, RunKey, lowered_programs, overlapped_captures, overlapped_replays, panic_after,
};

/// Rank `rank`'s input spans, which no launch writes during a step: every
/// phase's input arena and block tables, and the piece arena.
#[cfg(test)]
pub(crate) fn input_spans(
    handle: runtime_bundle_abi::BundleHandleV1,
    rank: usize,
) -> Vec<runtime_bundle_abi::DeviceBufferSpanV1> {
    with_instance(handle, |instance| {
        let mut spans = instance
            .arenas
            .iter()
            .flat_map(|arenas| [arenas[rank].arena, arenas[rank].block_tables])
            .collect::<Vec<_>>();
        spans.push(instance.piece_arenas[rank]);
        spans
    })
    .expect("a live instance")
}

/// Each rank's request-state buffers, by the snapshot table's buffer index,
/// and the table of `slot`'s snapshot.
#[cfg(test)]
pub(crate) fn snapshot_layout(
    handle: runtime_bundle_abi::BundleHandleV1,
    slot: u32,
) -> (
    Vec<Vec<runtime_bundle_abi::DeviceBufferSpanV1>>,
    bundle_runtime::snapshot::Table,
) {
    with_instance(handle, |instance| {
        let spans = instance
            .buffers
            .iter()
            .map(|buffers| buffers.state.spans.clone())
            .collect();
        (spans, instance.snapshot_table(slot).unwrap())
    })
    .expect("a live instance")
}

/// One rank's request-state buffers and separate spare buffers, by the
/// snapshot table's buffer index.
#[cfg(test)]
pub(crate) type HeldState = (
    Vec<runtime_bundle_abi::DeviceBufferSpanV1>,
    Vec<Option<runtime_bundle_abi::DeviceBufferSpanV1>>,
);

/// Each rank's [`HeldState`] and the table of anchor item `anchor`'s
/// snapshot.
#[cfg(test)]
pub(crate) fn anchor_layout(
    handle: runtime_bundle_abi::BundleHandleV1,
    anchor: u32,
) -> (Vec<HeldState>, bundle_runtime::snapshot::Table) {
    with_instance(handle, |instance| {
        let spans = instance
            .buffers
            .iter()
            .map(|buffers| (buffers.state.spans.clone(), buffers.state.spares.clone()))
            .collect();
        let pools = &instance.buffers[0].state;
        let table = bundle_runtime::snapshot::Table::new(
            &pools.state,
            pools.extent,
            bundle_runtime::state::Holder::Anchor(anchor),
        )
        .unwrap();
        (spans, table)
    })
    .expect("a live instance")
}

/// The entry anchor item `anchor`'s spares take in state buffer `name` on
/// rank 0 (`StatePools::anchor_entry`).
#[cfg(test)]
pub(crate) fn anchor_entry(
    handle: runtime_bundle_abi::BundleHandleV1,
    name: &str,
    anchor: u32,
) -> u64 {
    with_instance(handle, |instance| {
        instance.buffers[0]
            .state
            .anchor_entry(name, anchor)
            .unwrap()
    })
    .expect("a live instance")
}

/// Makes `slot`'s drafter state that of `tokens` committed with `frontier`
/// drafter rows run, as a restore would.
#[cfg(test)]
pub(crate) fn drafter_restore(
    handle: runtime_bundle_abi::BundleHandleV1,
    slot: u32,
    tokens: &[u32],
    frontier: u64,
) {
    with_instance(handle, |instance| {
        instance
            .drafter
            .as_mut()
            .expect("a drafting instance")
            .restore(slot, tokens, frontier)
            .unwrap()
    })
    .expect("a live instance")
}

/// Rank `rank`'s scratch buffer `buffer`.
#[cfg(test)]
pub(crate) fn scratch_span(
    handle: runtime_bundle_abi::BundleHandleV1,
    rank: usize,
    buffer: crate::buffers::Buffer,
) -> runtime_bundle_abi::DeviceBufferSpanV1 {
    with_instance(handle, |instance| instance.buffers[rank].get(buffer)).expect("a live instance")
}

/// Rank `rank`'s projection scratch and side scratch (`projections::Projections`).
#[cfg(test)]
pub(crate) fn projection_scratch_spans(
    handle: runtime_bundle_abi::BundleHandleV1,
    rank: usize,
) -> [Vec<runtime_bundle_abi::DeviceBufferSpanV1>; 2] {
    with_instance(handle, |instance| {
        let projections = instance
            .projections
            .as_ref()
            .expect("an instance with projections");
        [&projections.scratch[rank], &projections.side_scratch[rank]].map(|scratch| {
            vec![
                scratch.workspace,
                scratch.semaphores,
                scratch.temporary,
                scratch.unused,
            ]
        })
    })
    .expect("a live instance")
}

/// Rank `rank`'s drafter input spans, which no launch writes during a
/// step: the drafter's phase arenas and block tables.
#[cfg(test)]
pub(crate) fn draft_input_spans(
    handle: runtime_bundle_abi::BundleHandleV1,
    rank: usize,
) -> Vec<runtime_bundle_abi::DeviceBufferSpanV1> {
    with_instance(handle, |instance| {
        instance
            .draft_arenas
            .iter()
            .filter_map(|arenas| arenas.get(rank))
            .flat_map(|arena| [arena.arena, arena.block_tables])
            .collect()
    })
    .expect("a live instance")
}

/// Each rank's side stream and events of the shared gate/up overlap.
#[cfg(test)]
pub(crate) fn side_lanes(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> Vec<crate::shared_overlap::Lanes> {
    with_instance(handle, |instance| instance.side_lanes.clone()).expect("a live instance")
}

/// Rank `rank`'s drafter staging (`DrafterBuffers::staged`).
#[cfg(test)]
pub(crate) fn drafter_staged(
    handle: runtime_bundle_abi::BundleHandleV1,
    rank: usize,
) -> [runtime_bundle_abi::DeviceBufferSpanV1; 2] {
    with_instance(handle, |instance| {
        instance.buffers[rank]
            .drafter
            .expect("a drafting instance")
            .staged
    })
    .expect("a live instance")
}

/// Rank `rank`'s indexer tail: its layer `layer`, and that layer's entry
/// of `slot`, or of checkpoint `row` of `slot` when `row` is given.
#[cfg(test)]
pub(crate) fn indexer_tail(
    handle: runtime_bundle_abi::BundleHandleV1,
    rank: usize,
    layer: u64,
    slot: u32,
    row: Option<u32>,
) -> [runtime_bundle_abi::DeviceBufferSpanV1; 2] {
    with_instance(handle, |instance| {
        let pools = &instance.buffers[rank].state;
        let index = pools
            .state
            .buffers
            .iter()
            .position(|buffer| buffer.name == "indexer.tail")
            .expect("an indexer tail");
        let entry = match row {
            Some(row) => pools.state.buffers[index]
                .checkpoint(pools.extent, slot, row)
                .expect("a tail checkpoint"),
            None => u64::from(slot),
        };
        [
            pools.layer("indexer.tail", layer).unwrap(),
            pools.entry(index, layer, entry),
        ]
    })
    .expect("a live instance")
}

/// Rank `rank`'s projection weights of operation `id` at `usage`, and its
/// placed view `view`, when the instance holds a projection store.
#[cfg(test)]
pub(crate) fn projection_weights(
    handle: runtime_bundle_abi::BundleHandleV1,
    rank: usize,
    id: &str,
    usage: kernel_core::precision::OperationUse,
) -> Option<bundle_runtime::projection::Weights> {
    with_instance(handle, |instance| {
        let projections = instance.projections.as_ref()?;
        projections.stores[rank]
            .handle(id, usage)
            .ok()
            .map(|handle| handle.weights())
    })
    .expect("a live instance")
}

/// The stage symbols rank `rank`'s store launches for operation `id` at
/// `usage`, when the instance holds a projection store.
#[cfg(test)]
pub(crate) fn projection_stages(
    handle: runtime_bundle_abi::BundleHandleV1,
    rank: usize,
    id: &str,
    usage: kernel_core::precision::OperationUse,
) -> Vec<String> {
    with_instance(handle, |instance| {
        let projections = instance.projections.as_ref().expect("a projection store");
        projections.stores[rank]
            .handle(id, usage)
            .expect("a bound use")
            .identity()
            .stages
            .clone()
    })
    .expect("a live instance")
}

/// Rank `rank`'s placed view `name`.
#[cfg(test)]
pub(crate) fn view_span(
    handle: runtime_bundle_abi::BundleHandleV1,
    rank: usize,
    name: &str,
) -> runtime_bundle_abi::DeviceBufferSpanV1 {
    with_instance(handle, |instance| {
        instance.buffers[rank].view(name).unwrap()
    })
    .expect("a live instance")
}

/// The MTP drafter's frontier of `slot`, when the instance runs it.
#[cfg(test)]
pub(crate) fn drafter_frontier(
    handle: runtime_bundle_abi::BundleHandleV1,
    slot: u32,
) -> Option<u64> {
    with_instance(handle, |instance| {
        instance
            .drafter
            .as_ref()
            .map(|drafter| drafter.frontier(slot))
    })
    .expect("a live instance")
}

/// Synchronizes every rank after each eager segment and graph replay, as
/// `SYNC_STAGES` does.
/// Releases the instance's graphs, as destroy does first.
#[cfg(test)]
pub(crate) fn release_graphs(handle: runtime_bundle_abi::BundleHandleV1) {
    with_instance(handle, |instance| {
        bundle_runtime::entry::Bundle::release_graphs(instance)
    })
    .expect("a live instance")
    .expect("released graphs");
}

/// Releases the instance's graphs, as destroy does first, returning a
/// failure instead of panicking.
#[cfg(test)]
pub(crate) fn try_release_graphs(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> Result<(), BundleAbiError> {
    with_instance(handle, |instance| {
        bundle_runtime::entry::Bundle::release_graphs(instance)
    })
    .expect("a live instance")
}

#[cfg(test)]
pub(crate) fn sync_stages(handle: runtime_bundle_abi::BundleHandleV1) {
    with_instance(handle, |instance| instance.sync_stages = true).expect("a live instance");
}

/// Makes a request's last prompt chunk run eagerly at its first run, as
/// any other prefill shape does, for the tests of the eager path and of
/// the journals it issues.
#[cfg(test)]
pub(crate) fn keep_last_chunks_eager(handle: runtime_bundle_abi::BundleHandleV1) {
    with_instance(handle, |instance| {
        instance.prefill_graphs.keep_last_chunks_eager()
    })
    .expect("a live instance");
}

/// Makes the instance zero every rank's phased all-reduce state after
/// `runs` runs in its scope instead of `Target::phased_reset_runs`.
#[cfg(test)]
pub(crate) fn phased_reset_runs(handle: runtime_bundle_abi::BundleHandleV1, runs: u64) {
    with_instance(handle, |instance| instance.phased_reset_runs = runs).expect("a live instance");
}

/// The span of rank `rank`'s encoder region `name`.
#[cfg(all(test, feature = "diagnostic-taps"))]
pub(crate) fn encoder_region(
    handle: runtime_bundle_abi::BundleHandleV1,
    rank: usize,
    name: &str,
) -> runtime_bundle_abi::DeviceBufferSpanV1 {
    with_instance(handle, |instance| {
        instance.encoder.as_ref().expect("an encoder").ranks[rank]
            .region(name)
            .unwrap()
    })
    .expect("a live instance")
}

/// Issues the lowered stages alone, skipping those not lowered yet.
#[cfg(test)]
pub(crate) fn run_lowered_stages_alone(handle: runtime_bundle_abi::BundleHandleV1) {
    with_instance(handle, |instance| instance.partial = true).expect("a live instance");
}

/// Commits `tokens` tokens of `request` in `slot`, as the prefill that is
/// not lowered yet would.
#[cfg(test)]
pub(crate) fn seed(
    handle: runtime_bundle_abi::BundleHandleV1,
    request: u64,
    slot: u32,
    tokens: u32,
) {
    with_instance(handle, |instance| {
        let item = bundle_runtime::ledger::Item {
            request,
            slot: Some(slot),
            start: 0,
            rows: tokens,
            verify: false,
        };
        instance.ledger.begin(&[item], Vec::new()).unwrap();
        instance.ledger.commit(&[]).unwrap();
    })
    .expect("a live instance");
}

/// What each of the instance's ops and projections asked at create and was
/// granted, and which spinning all-reduce entries create admitted (rank
/// 0's).
#[cfg(test)]
pub(crate) fn asked(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> (Vec<crate::ops::Asked>, crate::ops::reduce::Admission) {
    with_instance(handle, |instance| {
        let comm = &instance.buffers[0].comm;
        let admission = crate::ops::reduce::Admission {
            low_latency: comm.low_latency,
            phased: comm.phased,
        };
        let mut asked = instance.ops.asked.clone();
        if let Some(projections) = &instance.projections {
            asked.extend(projections.asked.iter().cloned());
        }
        (asked, admission)
    })
    .expect("a live instance")
}

/// The (request, tokens) slot `slot` holds, and whether a step is open.
#[cfg(test)]
pub(crate) fn ledger_state(
    handle: runtime_bundle_abi::BundleHandleV1,
    slot: u32,
) -> (Option<(u64, u64)>, bool) {
    with_instance(handle, |instance| {
        (
            instance.ledger.holder(slot),
            instance.ledger.step().is_some(),
        )
    })
    .expect("a live instance")
}

/// Bytes of one indexer piece's packed inputs.
#[cfg(test)]
pub(crate) fn piece_bytes_for_tests() -> u64 {
    execute::piece_layout(Items::new(model_glm5next_contract::contract::DECODE_BATCH).unwrap()).end
}
