//! The `deepseek41.v1` instance behind the shared ABI entry bodies.
//!
//! Create decodes the record, prepares every op once on the target's
//! kernels (`crate::Kernels`), then acquires in order: every rank's index
//! (admitted and authenticated before the image opens), the rank contexts,
//! the image upload and its placed views, peer access, each rank's phase
//! input arenas and scratch, the host's per-item request state, the
//! embedded modules and last the serving KV pool, which the host sizes from
//! what they leave.
//!
//! Execute runs a step through the ledger (`execute`): its eager prefill,
//! then its decode or verify phase as captured graphs per bucket. Every
//! state buffer rolls back by position (`model_deepseek41_contract::state`),
//! so commit only moves the ledger, and cancel leaves the slots the step
//! touched invalid, which the ledger retires. The Engram history of each
//! request is the host's (`engram`): a commit adds the kept tokens, and a
//! freed slot or a retire forgets it.

pub(crate) mod ar_engram;
mod dspark;
mod engram;
pub(crate) mod execute;
mod fold_stream;
mod indexer_graphs;
mod media;
mod memory;
pub(crate) mod prefill;
mod prefill_cache;
pub(crate) mod prefill_engram;
mod prefill_reduce;
mod prefix;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::thread::LocalKey;

use bundle_runtime::acquire::{self, Log, RankHandles};
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::entry::{Bundle, Failed};
use bundle_runtime::geometry::{Items, ModelLimits};
use bundle_runtime::graphs::GraphCache;
use bundle_runtime::inputs::VerifyLimits;
use bundle_runtime::ledger::{Ledger, Placement, Rules};
use bundle_runtime::pipeline::Collectives;
use bundle_runtime::resources::{Held, allocate_span, status};
use bundle_runtime::state::{Extent, RequestState};
use bundle_runtime::table::{InstanceTable, Live};
use bundle_runtime::transaction::{self, Ended, Failure, Retire, Stage, Steps};
use bundle_runtime::upload::{BlockTableMirror, InputLayout, PhaseArena};
use model_deepseek41_contract::serving::{
    DeepSeek, INDEX_SCHEMA, MAX_VERIFY_ROWS, PAGE_TOKENS, RANK_IMAGE_ROLE, REQUEST_STATE_ID,
    index_role,
};
use runtime_bundle_abi::{
    BundleAbiError, BundleCompiledImageHandleV1, BundleCudaSymbolHandleV1, BundleKeptRowsV1,
    BundleStatusV1, ExecuteResultSinkV1, HostServicesV1,
};

use crate::buffers::{LogitsComm, RankBuffers, Scratch, StateSpans};
use crate::config::{Configuration, decode};
use crate::ops::Ops;

pub(crate) const NAME: &str = crate::ops::NAME;

thread_local! {
    static INSTANCES: RefCell<InstanceTable<Instance>> = const { RefCell::new(InstanceTable::new()) };
}

#[cfg(test)]
thread_local! {
    /// Instances created on this thread run the stages their kernels lower
    /// and skip the rest, for the fake-host tests of the lifecycle while
    /// engines are pending.
    pub(crate) static PARTIAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether instances created now run only their lowered stages.
fn partial() -> bool {
    #[cfg(test)]
    return PARTIAL.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

pub struct Instance {
    services: std::rc::Rc<HostServicesV1>,
    held: Vec<Held>,
    collectives: Collectives,
    config: Configuration,
    /// Immutable U32 cells indexed by their value, only for the opt-in profile.
    prefill_counts: Vec<runtime_bundle_abi::DeviceBufferSpanV1>,
    prefill_boundaries: Vec<runtime_bundle_sdk::prefill::Boundary>,
    skip_prefill: bool,
    ranks: Vec<RankHandles>,
    /// AR logits readiness, acquired before scratch and graph captures.
    result_ready: Vec<runtime_bundle_abi::BundleEventHandleV1>,
    /// Per rank, every loaded entry's symbol.
    symbols: Vec<BTreeMap<String, BundleCudaSymbolHandleV1>>,
    /// Per phase (prefill, decode, verify) and rank, the input arena and
    /// its block table.
    arenas: [Vec<PhaseArena>; 3],
    buffers: Vec<RankBuffers>,
    kv_pages: u64,
    ledger: Ledger,
    verify: VerifyLimits,
    /// The prefill, decode and verify phases' block tables as the device
    /// holds them.
    mirrors: [BlockTableMirror<DeepSeek>; 3],
    ops: Ops,
    state: RequestState,
    /// What the state buffers are sized for.
    extent: Extent,
    /// Steps planned so far: each row's index tag holds its step's.
    steps: std::cell::Cell<u64>,
    /// Some rank's router bias choice may hold image rows: set before a
    /// prefill uploads its image flags, cleared once every rank's flags
    /// are zero again. While it is clear, every rank's flags are zero.
    image_flags: std::cell::Cell<bool>,
    overlap: Vec<execute::overlap::Rank>,
    dma: Vec<execute::dma::Rank>,
    /// DMA issues each peer copy on its source's stream.
    dma_push: bool,
    layer_overlap: bool,
    prefill_reduce_runs: u64,
    /// Per-layer gather IDs, reserved before serving KV sizing.
    prefill_engram_inputs: Vec<runtime_bundle_abi::DeviceBufferSpanV1>,
    prefill_engram: Option<prefill_engram::Gathers>,
    /// AR decode's per-layer Engram uploads, when switched on.
    ar_engram: Option<ar_engram::State>,
    /// AR steps' one status readback, when switched on.
    ar_status: Option<execute::ar_status::Packet>,
    /// Kept upper-skip plans of short prompts.
    plans: execute::plan_cache::Cache,
    /// Run the lowered stages alone (`PARTIAL`).
    partial: bool,
    /// Each rank's decode and verify graphs by bucket.
    graphs: GraphCache<DeepSeek>,
    graph_bank: indexer_graphs::Bank,
    indexer_banks: BTreeMap<indexer_graphs::Bank, indexer_graphs::Pair>,
    shared_overlap: crate::shared_overlap::Settings,
    side_lanes: Vec<crate::shared_overlap::Lanes>,
    /// A full draft program for each exact item count, sharing graph ownership.
    draft_graphs: BTreeMap<u32, GraphCache<DeepSeek>>,
    #[cfg(any(feature = "diagnostic-taps", feature = "state-capture"))]
    state_capture_pages: BTreeMap<u64, Vec<i32>>,
    #[cfg(any(feature = "diagnostic-taps", feature = "state-capture"))]
    state_capture_root: Option<std::path::PathBuf>,
    #[cfg(feature = "state-capture")]
    transition_capture_bytes: std::cell::Cell<u64>,
    /// Per-layer taps of prefills (`crate::taps`), when the environment
    /// asks for them.
    taps: crate::taps::StepTaps,
    /// The Engram rows of a geometry with Engram layers.
    engram: Option<engram::Engram>,
    dspark: Option<crate::dspark::device::Device>,
    /// The compiled image, whose mapped views the host gathers rows of.
    image: Option<BundleCompiledImageHandleV1>,
    /// The kernels the encoder is created on, held from create to acquire
    /// by an instance serving images.
    encoder_kernels: Option<crate::Kernels>,
    /// The vision encoder of an instance serving images.
    pub(super) media: Option<crate::encoders::Media>,
    /// What create allocated on each rank, logged once create completes.
    memory: memory::Ledger,
}

impl Live for Instance {
    fn collectives(&self) -> &Collectives {
        &self.collectives
    }
}

/// The model limits of an instance of `config`: the vocabulary, the
/// geometry's context and the page size.
pub(crate) fn model_limits(config: &Configuration) -> ModelLimits {
    ModelLimits::new(
        config.shape.dims().vocab,
        config.shape.geometry.context_tokens(),
        PAGE_TOKENS,
    )
    .expect("the geometry's limits are valid")
}

/// The input layout of a verify step of an instance of `shape` serving
/// `items`: the verify rows the host asked for (`Shape::verify_rows`), so an
/// instance asked for the default rows lays its inputs out as before.
pub(crate) fn verify_layout(shape: &crate::shape::Shape, items: Items) -> InputLayout {
    InputLayout::within(shape.verify_rows, 1, items)
}

/// Verify rows per item the host's speculation asks for (the drafts and
/// the bonus row; none without speculation), and the verify limits of an
/// instance serving `items`.
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
    let limits = VerifyLimits::enabled_within::<DeepSeek>(
        speculation.maximum_draft_tokens,
        speculation.maximum_verify_tokens,
        items,
    );
    Ok((rows, limits))
}

impl Instance {
    fn items(&self) -> Items {
        self.config.create.items
    }

    pub(super) fn log(&self, message: &str) {
        let _ = self.services.log(format!("{NAME}: {message}").as_bytes());
    }

    /// Names the one-row decode twins the instance serves, so a served run
    /// shows what it measured.
    fn log_one_row_twins(&self) {
        for line in crate::Kernels::one_row_twin_lines(&self.config.shape) {
            self.log(&line);
        }
    }

    fn acquire(&mut self) -> Result<(), BundleAbiError> {
        let services = &self.services;
        let held = &mut self.held;
        let log = Log {
            services,
            bundle: NAME,
        };
        let shape = self.config.shape;
        let create = &self.config.create;
        let items = create.items;
        // Every rank's index, parsed under the model's schema and bound to
        // the image the host serves for that rank, before any rank context
        // opens or the image is opened or uploaded.
        // Each weight view an exact F32 projection, a dense projection or the
        // head reads is stored in the layout its ops read.
        let ops = &self.ops;
        let drafter = self.dspark.as_ref();
        let admit = |rank: u32, bytes: &[u8]| {
            let index =
                bundle_runtime::index::parse(bytes, rank, INDEX_SCHEMA).map_err(|error| {
                    let _ = services.log(format!("{NAME} rank index: {error}").as_bytes());
                    BundleAbiError::InvalidRange("deepseek41 rank index")
                })?;
            ops.check_weight_layouts(&shape, &index)
                .and_then(|()| drafter.map_or(Ok(()), |d| d.check_weight_layouts(&index)))
                .and_then(|()| crate::buffers::encoded_scales(&shape, &index).map(|_| ()))
                .map_err(|error| {
                    let _ = services.log(format!("{NAME} rank {rank} index: {error}").as_bytes());
                    BundleAbiError::InvalidRange("deepseek41 weight view layout")
                })?;
            Ok(index)
        };
        let images = acquire::RankImages {
            role: RANK_IMAGE_ROLE.as_bytes(),
            index_role: &index_role,
            max_index_bytes: 64 << 20,
        };
        let (image, indexes) =
            acquire::rank_images(log, images, shape.ranks(), admit, acquire::identity, held)?;
        let ranks = acquire::open_ranks(log, &create.ordinals, held)?;
        let mut memory = memory::Ledger::open(ranks.iter().map(|&(_, pool, _)| pool));
        if self
            .dspark
            .as_ref()
            .is_none_or(|d| d.cycle.undrafted_status)
            && crate::device_results::enabled(shape, self.config.kv.storage)
        {
            for &(context, _, _) in &ranks {
                let event = services.create_event(context)?;
                held.push(Held::Event(event));
                self.result_ready.push(event);
            }
        }
        let rank_arenas = acquire::upload_arenas(log, image, &ranks, held)?;
        self.image = Some(image);
        let mut views = Vec::new();
        for (arena, index) in rank_arenas.into_iter().zip(&indexes) {
            // The Engram tables the plan maps, which the host gathers rows
            // of and which are not placed.
            let mapped = match &mut self.engram {
                Some(engram) => engram.mapped(shape.geometry, index).map_err(|error| {
                    let _ = services.log(format!("{NAME}: {error}").as_bytes());
                    BundleAbiError::InvalidRange("deepseek41 Engram tables")
                })?,
                None => Vec::new(),
            };
            let mapped: Vec<&str> = mapped.iter().map(String::as_str).collect();
            if let Some(engram) = &self.engram {
                engram.register_resident(services, image, index.rank)?;
            }
            let placed = acquire::place_views_mapped(services, image, arena, index, &[], &mapped)
                .map_err(|error| match error {
                acquire::PlaceError::Host(error) => error,
                acquire::PlaceError::Layout(message) => {
                    let _ = services.log(message.as_bytes());
                    BundleAbiError::InvalidRange("deepseek41 rank arena length")
                }
            })?;
            views.push(acquire::spans(placed));
        }
        acquire::grant_peers(services, &ranks, held)?;
        if self.ops.prefill_dma.enabled() {
            self.dma =
                execute::dma::acquire(services, &ranks, self.config.shape.dims().hidden, held)?;
            self.dma_push = crate::ops::dma_push_requested()
                && execute::dma::pushes(services, &ranks, &self.dma)?;
            let copies = if self.dma_push {
                "pushed by each source"
            } else {
                "pulled by each destination"
            };
            let _ = services.log(format!("{NAME}: prefill DMA copies: {copies}").as_bytes());
        }
        memory.take("dma");
        let limits = model_limits(&self.config);
        let layouts = [
            InputLayout::prefill_capacity_within::<DeepSeek>(items),
            InputLayout::decode_within(items).end,
            verify_layout(&self.config.shape, items).end,
        ];
        let tables = BlockTableMirror::<DeepSeek>::device_bytes(limits, items);
        let mut scratch = Vec::new();
        let mut encoded_scales = Vec::new();
        let mut comm = Vec::new();
        let mut logits_comm = Vec::new();
        let mut mix_scratch = Vec::new();
        let mut op_spans = Vec::new();
        // The one-hot `pre` the first sublayer collapses with, on every
        // row; held until the uploads complete.
        let first_pre: Vec<u8> = model_deepseek41_contract::hc::first_pre(shape.geometry)
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .cycle()
            .take(crate::buffers::scratch_rows() as usize * shape.dims().hc.streams as usize * 4)
            .collect();
        for (rank, &(context, pool, stream)) in ranks.iter().enumerate() {
            for (phase, bytes) in layouts.into_iter().enumerate() {
                let arena = allocate_span(services, pool, stream, bytes, held)?;
                let block_tables = allocate_span(services, pool, stream, tables, held)?;
                self.arenas[phase].push(PhaseArena {
                    arena,
                    block_tables,
                });
            }
            memory.take("inputs");
            if self.config.prefill.skip {
                let counts: Vec<u8> = (0..=model_deepseek41_contract::serving::PREFILL_TOKENS)
                    .flat_map(u32::to_le_bytes)
                    .collect();
                let span = allocate_span(services, pool, stream, counts.len() as u64, held)?;
                services.copy_host_to_device(stream, span, &counts)?;
                self.prefill_counts.push(span);
            }
            memory.take("prefill_counts");
            let mut declared = Scratch::of_storage(&shape, self.config.kv.storage);
            // The raw banks the experts read where the rank image stores a
            // routed scale bank encoded: both of a layer's, gate/up then down.
            let encoded = crate::buffers::encoded_scales(&shape, &indexes[rank])
                .map_err(|_| BundleAbiError::InvalidRange("deepseek41 weight view layout"))?;
            if !encoded.is_empty() {
                let bytes = crate::buffers::expert_scale_bytes(&shape)
                    .map_err(|_| BundleAbiError::InvalidRange("deepseek41 expert scale bank"))?;
                declared.expert_scales = Some(kernel_core::allocation::Allocation::bytes(bytes));
            }
            encoded_scales.push(encoded);
            if let Some(drafter) = &self.dspark {
                drafter.reserve_engram(&mut declared);
            }
            // Route only the peer-visible planes to the private allocation
            // owner; arithmetic and copies keep their explicit
            // compute/communication streams.
            let regions = crate::buffers::regions();
            let bound = bundle_runtime::scratch::bind_named(declared, regions, |name, bytes| {
                let owner = if matches!(name, "partial" | "reduced" | "embedded") {
                    self.dma.get(rank).map_or(stream, |dma| dma.owner)
                } else {
                    stream
                };
                allocate_span(services, pool, owner, bytes, held)
            })?;
            memory.add(rank, "scratch", bound.bytes);
            let mut parts = bound.regions.clone();
            let aliased: u64 = parts.iter().map(|&(_, bytes)| bytes).sum();
            parts.push(("unaliased", bound.bytes - aliased));
            memory.detail(rank, "scratch", parts);
            let bound = bound.spans;
            if crate::ops::indexer::prefill::enabled(&shape, self.config.kv.storage) {
                let counts: Vec<u8> = (0..=model_deepseek41_contract::serving::PREFILL_TOKENS)
                    .flat_map(u32::to_le_bytes)
                    .collect();
                services.copy_host_to_device(
                    stream,
                    crate::buffers::index_prefill_counts(bound.block_live)?,
                    &counts,
                )?;
            }
            services.copy_host_to_device(stream, bound.first_pre, &first_pre)?;
            scratch.push(bound);
            // The mixer scratch the mix-collapse lists; its tickets start
            // zeroed. A drafter's forwards above the decode batch split
            // their mixer over the same scratch.
            let bytes = self
                .ops
                .hc
                .as_ref()
                .map(|ops| ops.scratch())
                .unwrap_or_default()
                .most(
                    self.dspark
                        .as_ref()
                        .and_then(|drafter| drafter.ops.hc.as_ref())
                        .map(|ops| ops.draft_scratch())
                        .unwrap_or_default(),
                );
            let partials = bytes
                .partials
                .map(|bytes| allocate_span(services, pool, stream, bytes, held))
                .transpose()?;
            let tickets = match bytes.tickets {
                Some(bytes) => {
                    let span = allocate_span(services, pool, stream, bytes, held)?;
                    services.zero_device_buffer(stream, span)?;
                    Some(span)
                }
                None => None,
            };
            mix_scratch.push((partials, tickets));
            memory.take("mix");
            // The spans the ops size: the sorted routed rows and the
            // backend workspace (at least one byte, so every rank binds a
            // span), and the routes' bias choice, zeroed: text rows.
            let sorted = self
                .ops
                .moe
                .as_ref()
                .map_or(0, |ops| ops.sorted_rows_bytes());
            let sorted = sorted.max(
                self.dspark
                    .as_ref()
                    .and_then(|d| d.ops.moe.as_ref())
                    .map_or(0, |ops| ops.sorted_rows_bytes()),
            );
            let sorted = allocate_span(services, pool, stream, sorted.max(4), held)?;
            memory.take("sorted_rows");
            let workspace = self
                .ops
                .workspace()
                .max(self.dspark.as_ref().map_or(0, |d| d.workspace()))
                .max(1)
                .next_multiple_of(256);
            let workspace = allocate_span(services, pool, stream, workspace, held)?;
            memory.take("workspace");
            let shared_workspace = self
                .ops
                .shared_workspace()
                .max(self.dspark.as_ref().map_or(0, |d| d.ops.shared_workspace()))
                .max(1)
                .next_multiple_of(256);
            let shared_workspace = allocate_span(services, pool, stream, shared_workspace, held)?;
            memory.take("shared_workspace");
            if let Some(drafter) = &mut self.dspark {
                drafter.allocate(services, (context, pool, stream), held)?;
            }
            memory.take("dspark");
            if let Some(bound) = scratch.last() {
                services.zero_device_buffer(stream, bound.bias_select)?;
                services.zero_device_buffer(stream, bound.index_status)?;
            }
            // The compressed RoPE table, which the modules fill below.
            let table = self
                .ops
                .compressor
                .as_ref()
                .map_or(4, |ops| ops.table_bytes());
            let table = allocate_span(services, pool, stream, table, held)?;
            memory.take("rope_table");
            // One admitted eager item uses this arena at any serving capacity.
            // Reserve it before the host sizes its KV pool.
            let prefill_cache = if crate::Kernels::prefill_cache_eligible(
                shape.geometry,
                shape.ranks(),
                self.config.kv.storage,
            ) {
                let bytes = crate::Kernels::prefill_cache_layout(
                    crate::buffers::staged_slots(&shape),
                    64,
                    crate::Kernels::PREFILL_CACHE_PAGES,
                )
                .map_err(|_| BundleAbiError::InvalidRange("prefill cache extent"))?
                .bytes;
                Some(crate::buffers::PrefillCache {
                    data: allocate_span(services, pool, stream, bytes, held)?,
                    tables: allocate_span(
                        services,
                        pool,
                        stream,
                        u64::from(limits.max_pages()) * 4,
                        held,
                    )?,
                })
            } else {
                None
            };
            memory.take("prefill_cache");
            op_spans.push((sorted, workspace, table, shared_workspace, prefill_cache));
            // The all-reduce's comm state, which the peers write: zeroed
            // before any step, after peer access.
            let regions = self.ops.reduce.as_ref().map_or(&[][..], |ops| ops.state());
            let mut spans = Vec::with_capacity(regions.len());
            for &bytes in regions {
                let span = allocate_span(services, pool, stream, bytes, held)?;
                services.zero_device_buffer(stream, span)?;
                spans.push(span);
            }
            comm.push(spans);
            memory.take("comm");
            // The logits moves' comm state, likewise.
            let logits = match self.ops.logits.as_ref().map(|ops| ops.state()) {
                Some([exchange_flags, exchange_epoch, gather_flags, gather_epoch]) => {
                    let mut zeroed = |bytes| -> Result<_, BundleAbiError> {
                        let span = allocate_span(services, pool, stream, bytes, held)?;
                        services.zero_device_buffer(stream, span)?;
                        Ok(span)
                    };
                    let exchange_flags = zeroed(exchange_flags)?;
                    let exchange_epoch = zeroed(exchange_epoch)?;
                    let gather_flags = zeroed(gather_flags)?;
                    let gather_epoch = zeroed(gather_epoch)?;
                    Some(LogitsComm {
                        exchange_flags,
                        exchange_epoch,
                        gather_flags,
                        gather_epoch,
                    })
                }
                None => None,
            };
            logits_comm.push(logits);
            memory.take("logits_comm");
        }
        // The window rings and compressor tails of every bucket item, which
        // the host allocates per state slot; zeroed, as a new request's
        // rings start empty.
        let state_views = acquire::zeroed_state(
            services,
            REQUEST_STATE_ID.as_bytes(),
            create.state.bytes,
            &ranks,
            held,
        )?;
        let extent = self.extent;
        let target = model_deepseek41_contract::state::request_state(
            shape.geometry,
            self.config.kv.storage,
            false,
        );
        let state_views = state_views
            .into_iter()
            .zip(&ranks)
            .map(|(view, &(_, pool, stream))| {
                let mut bound = StateSpans::new(&target, extent, view)?;
                if let Some(drafter) = &self.dspark {
                    drafter.extend_state(services, pool, stream, extent, &mut bound, held)?;
                }
                Ok(bound)
            })
            .collect::<Result<Vec<_>, BundleAbiError>>()?;
        memory.take("dspark");
        let mut shared = self.ops.shared_bytes();
        if let Some(drafter) = &self.dspark {
            for (entry, bytes) in drafter.shared_bytes() {
                let most = shared.entry(entry).or_default();
                *most = (*most).max(bytes);
            }
        }
        for (rank, &(context, pool, stream)) in ranks.iter().enumerate() {
            let mut resolved = BTreeMap::new();
            // The cubins of this geometry and the shared ones.
            for cubin in crate::CUBINS
                .iter()
                .filter(|cubin| {
                    crate::serves(cubin.name, &shape)
                        || self.dspark.as_ref().is_some_and(|d| d.allocation.is_some())
                            && crate::Kernels::allocation_cubin(cubin.name)
                })
                .filter(|cubin| crate::Kernels::loads_indexer(cubin.name, self.config.kv.storage))
                .filter(|cubin| {
                    crate::Kernels::loads_latent_attention(
                        cubin.name,
                        &shape,
                        self.config.kv.storage,
                    )
                })
                .filter(|cubin| {
                    cubin.name != "comm_rounded_phased_deepseek41_sm120"
                        || self
                            .ops
                            .reduce
                            .as_ref()
                            .is_some_and(|ops| ops.prefill_prepared() && !ops.bf16_partials())
                })
                .filter(|cubin| {
                    cubin.name != "comm_dma_sum_deepseek41_sm120" || !self.dma.is_empty()
                })
                .filter(|cubin| {
                    // BF16-partial prefills send rounded partials over DMA.
                    cubin.name != "comm_dma_bf16_deepseek41_sm120"
                        || !self.dma.is_empty()
                            && self
                                .ops
                                .reduce
                                .as_ref()
                                .is_some_and(|ops| ops.bf16_partials())
                })
                .filter(|cubin| {
                    !cubin.name.starts_with("comm_bf16_partials_")
                        || self
                            .ops
                            .reduce
                            .as_ref()
                            .is_some_and(|ops| ops.bf16_partials())
                })
                .filter(|cubin| {
                    self.dspark.is_some()
                        || !crate::Kernels::drafter_only(cubin.name, shape.geometry)
                })
                .filter(|cubin| {
                    crate::Kernels::loads_scale_kernels(
                        cubin.name,
                        encoded_scales[rank].values().next().map(|e| e.format),
                        self.ops.moe.as_ref().map_or(
                            model_deepseek41_contract::weight_plan::RoutedLayout::Rows,
                            crate::ops::moe::MoeOps::gate_up_layout,
                        ),
                    )
                })
                .filter(|cubin| {
                    !cubin.name.starts_with("position_candidates_runtime_")
                        || shape.runtime.candidate_topk_blocks
                            != shape.geometry.shape.candidates.top_blocks
                })
            {
                acquire::load_module(
                    log,
                    context,
                    (cubin.name, cubin.bytes),
                    cubin.symbols.iter().copied(),
                    |symbol| Ok(shared.get(symbol).copied().unwrap_or(0)),
                    &mut resolved,
                    held,
                )?;
            }
            // Every served position's compressed RoPE rows, computed once.
            if let Some(ops) = &self.ops.compressor {
                let fill = ops.fill_table(op_spans[rank].2).map_err(|refusal| {
                    let _ = services
                        .log(format!("{NAME}: compressed RoPE table: {refusal}").as_bytes());
                    BundleAbiError::InvalidRange("deepseek41 compressed RoPE table")
                })?;
                for lowered in fill {
                    let launch = lowered.launch;
                    bundle_runtime::dispatch::launch(
                        services,
                        stream,
                        &resolved,
                        &launch.symbol,
                        launch.plan,
                        &launch.arguments,
                        &launch.bindings,
                    )
                    .map_err(|error| match error {
                        DispatchError::Host(error) => error,
                        error => {
                            let _ = services.log(
                                format!("{NAME}: compressed RoPE table: {error:?}").as_bytes(),
                            );
                            BundleAbiError::InvalidRange("deepseek41 compressed RoPE table")
                        }
                    })?;
                }
            }
            // Flags follow the weight view's name, including drafter stage
            // remapping. The shared head is classified exactly once.
            let weights: Vec<_> = views[rank]
                .iter()
                .map(|(name, &span)| (name.clone(), span))
                .collect();
            for (name, weight) in weights {
                let ops = if name.starts_with("mtp.") {
                    self.dspark.as_ref().map(|drafter| &drafter.ops)
                } else {
                    Some(&self.ops)
                };
                let op = ops.and_then(|ops| {
                    if name == "head.weight" {
                        ops.head.as_ref().map(|head| &head.project)
                    } else {
                        ops.exact_f32
                            .as_ref()
                            .and_then(|dense| dense.weight_op(&name))
                    }
                });
                let Some(op) = op else { continue };
                let Some(bytes) = op.info().bytes_of("scratch.weight_domain") else {
                    continue;
                };
                let flags = allocate_span(services, pool, stream, bytes, held)?;
                let launches =
                    crate::ops::exact_f32::initialize(op, weight, flags).map_err(|error| {
                        let _ = services.log(error.to_string().as_bytes());
                        BundleAbiError::InvalidRange("F32 weight-domain initialization")
                    })?;
                for lowered in launches {
                    let launch = lowered.launch;
                    bundle_runtime::dispatch::launch(
                        services,
                        stream,
                        &resolved,
                        &launch.symbol,
                        launch.plan,
                        &launch.arguments,
                        &launch.bindings,
                    )
                    .map_err(|error| {
                        let _ = services
                            .log(format!("F32 weight-domain initialization: {error:?}").as_bytes());
                        BundleAbiError::InvalidRange("F32 weight-domain initialization")
                    })?;
                }
                views[rank].insert(crate::ops::exact_f32::domain_name(&name), flags);
            }
            memory.take("weight_domain");
            let folds = fold_stream::initialize(
                &self.ops,
                self.dspark.as_ref(),
                services,
                pool,
                stream,
                &mut views[rank],
                held,
            )?;
            memory.add(rank, "fold_state", folds.state);
            memory.add(rank, "fold_workspace", folds.workspace);
            self.symbols.push(resolved);
        }
        // The vision encoder's cubins, image and working memory, before the
        // KV pool takes what is left.
        if let Some(kernels) = self.encoder_kernels.take() {
            let mut media = crate::encoders::acquire_media(
                services,
                &kernels,
                &ranks,
                &mut self.symbols,
                u64::from(crate::buffers::scratch_rows()),
                held,
            )?;
            media.encoder.block_taps = self.taps.selects_blocks();
            self.media = Some(media);
        }
        memory.take("encoder");
        if prefill_engram::enabled(self.dspark.is_some()).map_err(BundleAbiError::Status)?
            && services.has_async_mapped_gather()
        {
            let layers = self
                .engram
                .as_ref()
                .and_then(|engram| engram.tables())
                .map_or(0, <[_]>::len);
            if layers > 0 {
                let engram = &shape.geometry.shape.engram;
                let bytes = u64::from(model_deepseek41_contract::serving::PREFILL_TOKENS)
                    * u64::from((engram.max_ngram - 1) * engram.heads)
                    * 4;
                let (_, pool, stream) = ranks[0];
                self.prefill_engram_inputs = (0..layers)
                    .map(|_| allocate_span(services, pool, stream, bytes, held))
                    .collect::<Result<_, _>>()?;
            }
        }
        memory.take("prefill_engram");
        let (pages, pools) = acquire::serving_kv(
            log,
            self.config.kv.layout_id,
            self.config.page_bytes,
            shape.ranks(),
        )?;
        self.kv_pages = pages;
        memory.sized();
        for (
            (
                ((((scratch, encoded_scales), views), (kv_pool, state)), (comm, logits)),
                (mix_partials, mix_tickets),
            ),
            (sorted_rows, workspace, rope_compressed, shared_workspace, prefill_cache),
        ) in scratch
            .into_iter()
            .zip(encoded_scales)
            .zip(views)
            .zip(pools.into_iter().zip(state_views))
            .zip(comm.into_iter().zip(logits_comm))
            .zip(mix_scratch)
            .zip(op_spans)
        {
            self.buffers.push(RankBuffers {
                scratch,
                state,
                views,
                kv_pool,
                comm,
                logits,
                sorted_rows,
                workspace,
                shared_workspace,
                prefill_cache,
                mix_partials,
                mix_tickets,
                rope_compressed,
                encoded_scales,
            });
        }
        if let Some(drafter) = &mut self.dspark {
            drafter.bind_stages(&self.buffers).map_err(|error| {
                let _ = services.log(error.as_bytes());
                BundleAbiError::InvalidRange("DSpark stage bindings")
            })?;
        }
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
        if execute::overlap::enabled() && self.ops.reduce.is_some() {
            self.overlap = execute::overlap::acquire(services, &ranks, held)?;
        }
        memory.take("overlap");
        self.memory = memory;
        // Allocations on a private stream must exist before either active
        // lane can use them. No peer-visible plane has a create-time producer.
        for dma in &self.dma {
            services.synchronize_stream(dma.owner)?;
        }
        acquire::synchronize(log, &ranks)?;
        self.ranks = ranks;
        Ok(())
    }
}

fn create(services: HostServicesV1, bytes: &[u8]) -> Result<Instance, Failed<Instance>> {
    let log = |reason: &str| {
        let _ = services.log(format!("{NAME} create failed: {reason}").as_bytes());
    };
    let mut config = decode(bytes, log)?;
    let taps = crate::taps::StepTaps::from_env(&config).map_err(|error| {
        log(&error);
        BundleStatusV1::InvalidArgument
    })?;
    let items = config.create.items;
    let (verify_rows, mut verify) = verify_rows(&services, items)?;
    // Every op is prepared for the verify rows the host asks for.
    config.shape = config.shape.with_verify_rows(verify.rows_per_step);
    if verify_rows > 0 && config.kv.storage == model_deepseek41_contract::state::KvStorage::Bf16 {
        log("BF16 KV supports autoregressive serving only; set --speculative-tokens 0");
        return Err(BundleStatusV1::Unsupported.into());
    }
    // The head scores at most this many rows per phase.
    bundle_runtime::config::check_scored_rows(
        &services,
        NAME,
        model_deepseek41_contract::serving::SCORED_ROWS,
    )?;
    let shape = config.shape;
    let has_drafter = verify_rows > 0
        && config.kv.storage == model_deepseek41_contract::state::KvStorage::Native
        && shape.geometry.dspark_stages() > 0;
    let state = model_deepseek41_contract::state::request_state(
        shape.geometry,
        config.kv.storage,
        has_drafter,
    );
    if config.create.state.bytes
        != u64::from(items.capacity())
            * crate::config::slot_state_bytes(&shape, config.kv.storage, false)
    {
        log("the per-item state bytes differ from the model's state declaration");
        return Err(BundleStatusV1::Unsupported.into());
    }
    // The window write reads each slot's last position from one scratch
    // row a slot.
    if items.capacity() > crate::buffers::scratch_rows() {
        log("the instance holds more state slots than the window write addresses");
        return Err(BundleStatusV1::Unsupported.into());
    }
    // Every op, prepared once on the target's kernels.
    let kernels = crate::Kernels::new_for_kv(&shape, config.kv.storage).map_err(|error| {
        log(&error);
        BundleStatusV1::Unsupported
    })?;
    let ops = Ops::prepare(&kernels, &shape, config.kv.storage, config.prefill);
    let dspark = if has_drafter {
        Some(
            crate::dspark::device::Device::new(
                &kernels,
                shape,
                items,
                config
                    .create
                    .context_tokens
                    .unwrap_or(shape.geometry.context_tokens()),
                config.dspark_draft_chunk_items as usize,
                if config.dspark_coupled_drafts {
                    crate::lower::dspark::DraftMode::Sampled
                } else {
                    crate::lower::dspark::DraftMode::Greedy
                },
                config.verify_allocation,
            )
            .map_err(|error| {
                log(&error);
                BundleStatusV1::Unsupported
            })?,
        )
    } else {
        None
    };
    verify.bundle_drafter = has_drafter;
    let media = config.media;
    let engram = engram::Engram::new(shape.geometry, media).map_err(|error| {
        log(&error);
        BundleStatusV1::InternalError
    })?;
    let plans = execute::plan_cache::Cache::from_env().map_err(|error| {
        log(&error);
        BundleStatusV1::InvalidArgument
    })?;
    let partial = partial();
    if !partial && let Some(missing) = ops.missing.first() {
        log(missing);
        return Err(BundleStatusV1::Unsupported.into());
    }
    let limits = model_limits(&config);
    let ledger = Ledger::new(
        items.get(),
        Placement::Host,
        Rules::of(&state),
        config
            .create
            .context_tokens
            .unwrap_or(limits.context_tokens()),
    );
    let mut instance = Instance {
        services: std::rc::Rc::new(services),
        held: Vec::new(),
        collectives: Collectives::default(),
        graphs: GraphCache::with_items(config.create.ranks, items),
        graph_bank: indexer_graphs::Bank::default(),
        indexer_banks: BTreeMap::new(),
        shared_overlap: crate::shared_overlap::Settings::from_env(),
        side_lanes: Vec::new(),
        draft_graphs: BTreeMap::new(),
        #[cfg(any(feature = "diagnostic-taps", feature = "state-capture"))]
        state_capture_pages: BTreeMap::new(),
        #[cfg(any(feature = "diagnostic-taps", feature = "state-capture"))]
        state_capture_root: std::env::var_os("CHOPINFER_DSPARK_STATE_DIR").map(Into::into),
        #[cfg(feature = "state-capture")]
        transition_capture_bytes: std::cell::Cell::new(0),
        config,
        prefill_counts: Vec::new(),
        prefill_boundaries: Vec::new(),
        skip_prefill: false,
        ranks: Vec::new(),
        result_ready: Vec::new(),
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
        ops,
        state,
        extent: Extent {
            served: items.get(),
            bucket: items.capacity(),
            verify_rows,
        },
        steps: std::cell::Cell::new(0),
        image_flags: std::cell::Cell::new(false),
        overlap: Vec::new(),
        dma: Vec::new(),
        dma_push: false,
        layer_overlap: execute::overlap::layers_enabled(),
        prefill_reduce_runs: 0,
        prefill_engram_inputs: Vec::new(),
        prefill_engram: None,
        ar_engram: None,
        ar_status: None,
        plans,
        partial,
        taps,
        engram,
        dspark,
        image: None,
        encoder_kernels: media.then_some(kernels),
        media: None,
        memory: memory::Ledger::default(),
    };
    if let Err(error) = instance.acquire() {
        instance.log(&format!("create failed: {error}"));
        let result = status(error);
        return Err(Failed::releasing(instance, result));
    }
    instance.admit_prefill_reduce();
    if let Err(error) = instance.admit_bf16_partials() {
        instance.log(&error);
        return Err(Failed::releasing(instance, BundleStatusV1::Unsupported));
    }
    instance.memory.take("other");
    if let Err(error) = instance.prepare_dspark_cycle() {
        instance.log(&format!(
            "create failed preparing DSpark device cycle: {error}"
        ));
        return Err(Failed::releasing(instance, status(error)));
    }
    instance.memory.take("dspark_cycle");
    if let Err(error) = instance.prepare_prefill_engram() {
        instance.log(&format!("create failed preparing prefill Engram: {error}"));
        return Err(Failed::releasing(instance, status(error)));
    }
    instance.memory.take("prefill_engram");
    if let Err(error) = instance.prepare_ar_engram() {
        instance.log(&format!("create failed preparing AR Engram: {error}"));
        return Err(Failed::releasing(instance, status(error)));
    }
    if let Err(error) = instance.prepare_ar_status() {
        instance.log(&format!(
            "create failed preparing the AR status packet: {error}"
        ));
        return Err(Failed::releasing(instance, status(error)));
    }
    instance.memory.take("ar_engram");
    instance.log_one_row_twins();
    if let Err(error) = instance.configure_dspark_graphs() {
        instance.log(&format!("create failed configuring DSpark graphs: {error}"));
        return Err(Failed::releasing(instance, status(error)));
    }
    instance.memory.take("dspark_graphs");
    if let Err(error) = instance.check_dspark_allocation() {
        instance.log(&format!(
            "create failed checking the verify allocation: {error}"
        ));
        return Err(Failed::releasing(instance, BundleStatusV1::BundleError));
    }
    for line in instance.memory.lines() {
        let _ = instance
            .services
            .log(format!("startup memory: {line}").as_bytes());
    }
    instance.log(&format!(
        "{} KV pages per rank, {} items; ops prepared on {} and {}",
        instance.kv_pages,
        instance.items().get(),
        instance.ops.backends[0],
        instance.ops.backends[1],
    ));
    Ok(instance)
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
        execute::dma::drain(self)?;
        self.graphs.release_all(&self.services)?;
        // AR's gathers are the newest: the host releases newest first.
        self.release_ar_engram()?;
        if let Some(gathers) = &mut self.prefill_engram {
            gathers.release(&self.services)?;
        }
        if let Some(device) = &mut self.dspark {
            // The early gathers follow the layers' at create.
            if let Some(early) = &mut device.cycle.early {
                while let Some(&gather) = early.gathers.last() {
                    self.services.release_mapped_gather(gather)?;
                    early.gathers.pop();
                }
            }
            while let Some(layer) = device.cycle.layers.last() {
                self.services.release_mapped_gather(layer.gather)?;
                device.cycle.layers.pop();
            }
        }
        Ok(())
    }

    fn execute(&mut self, request: &[u8], sink: &mut ExecuteResultSinkV1) -> BundleStatusV1 {
        if self.ledger.step().is_some() {
            return BundleStatusV1::InvalidState;
        }
        self.skip_prefill = false;
        self.prefill_boundaries.clear();
        self.execute_step(request, sink)
    }

    /// A step with verify items commits through `commit_accepted`.
    fn commit(&mut self) -> BundleStatusV1 {
        transaction::commit(self)
    }

    fn commit_accepted(&mut self, records: &[BundleKeptRowsV1]) -> BundleStatusV1 {
        transaction::commit_accepted(self, records)
    }

    /// The rings and tails advance in place, so the ledger retires the
    /// requests a cancelled step touched.
    fn cancel(&mut self) -> BundleStatusV1 {
        transaction::cancel(self)
    }

    fn retire_request(&mut self, request_id: u64) -> BundleStatusV1 {
        transaction::retire(self, request_id)
    }
}

/// Commits the open step or cancels it. Every state buffer rolls back by
/// position, so a commit moves only the ledger; a ledger refusal changes
/// nothing.
impl Steps for Instance {
    const RETIRE: Retire = Retire::UnlessHeld;

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

    fn ledger(&mut self) -> &mut Ledger {
        &mut self.ledger
    }

    fn prepare(
        &mut self,
        _kept: &[(u64, u32)],
        plan: &bundle_runtime::ledger::CommitPlan,
    ) -> Result<(), BundleStatusV1> {
        self.dspark_commit(plan)
    }

    fn restore(&mut self, _step: &bundle_runtime::ledger::Step) -> Result<(), BundleStatusV1> {
        self.dspark_cancel()
    }

    /// Taps retire the request even when the ledger refuses it; a drafter
    /// with a pending or poisoned runtime refuses the retire.
    fn before_retire(&mut self, request: u64) -> Result<(), BundleStatusV1> {
        self.taps.retire(request);
        if self.dspark.as_ref().is_some_and(|device| {
            device
                .runtime
                .as_ref()
                .is_none_or(|runtime| runtime.state().is_pending() || runtime.state().is_poisoned())
        }) {
            return Err(BundleStatusV1::InvalidState);
        }
        #[cfg(feature = "diagnostic-taps")]
        self.capture_dspark_state(request).map_err(|error| {
            self.log(&error);
            BundleStatusV1::InternalError
        })?;
        #[cfg(all(feature = "state-capture", not(feature = "diagnostic-taps")))]
        self.state_capture_pages.remove(&request);
        Ok(())
    }

    /// The Engram history takes the rows a commit kept, and forgets a
    /// request whose slot a cancel freed.
    fn ended(&mut self, ended: Ended<'_>) -> Result<(), BundleStatusV1> {
        #[cfg(feature = "state-capture")]
        if let Ended::Committed(committed) = ended {
            self.capture_transition_commit(committed).map_err(|error| {
                self.log(&error);
                BundleStatusV1::HostError
            })?;
        }
        #[cfg(feature = "diagnostic-taps")]
        if matches!(ended, Ended::Committed(_)) {
            self.capture_profile_chunks().map_err(|error| {
                self.log(&error);
                BundleStatusV1::HostError
            })?;
        }
        if self.dspark.is_some() {
            if let Ended::Cancelled(step) = ended {
                for item in &step.items {
                    #[cfg(any(feature = "diagnostic-taps", feature = "state-capture"))]
                    self.state_capture_pages.remove(&item.request);
                    self.dspark_forget(item.request);
                }
            }
            return Ok(());
        }
        let Some(engram) = &mut self.engram else {
            return Ok(());
        };
        match ended {
            Ended::Committed(committed) => engram.commit(committed).map_err(|error| {
                let _ = self
                    .services
                    .log(format!("{NAME}: Engram commit: {error}").as_bytes());
                BundleStatusV1::InternalError
            }),
            Ended::Cancelled(_) => {
                engram.cancelled(&self.ledger);
                Ok(())
            }
        }
    }

    fn retired(&mut self, request: u64) {
        self.dspark_forget(request);
        if let Some(engram) = &mut self.engram {
            engram.forget(request);
        }
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
}

/// The (request, tokens) slot `slot` holds, and whether a step is open.
#[cfg(test)]
pub(crate) fn ledger_state(
    handle: runtime_bundle_abi::BundleHandleV1,
    slot: u32,
) -> (Option<(u64, u64)>, bool) {
    INSTANCES
        .with(|table| {
            table.borrow_mut().get_mut(handle).map(|instance| {
                (
                    instance.ledger.holder(slot),
                    instance.ledger.step().is_some(),
                )
            })
        })
        .expect("a live instance")
}

#[cfg(test)]
pub(crate) fn dspark_context(
    handle: runtime_bundle_abi::BundleHandleV1,
    request: u64,
) -> Option<crate::dspark::state::Context> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle)?;
        instance
            .dspark
            .as_ref()?
            .runtime
            .as_ref()?
            .state()
            .context(request)
            .cloned()
    })
}

/// Whether the instance's drafter holds the allocated verify steps' ops
/// and storage: one storage record per rank.
#[cfg(test)]
pub(crate) fn dspark_allocation_ranks(handle: runtime_bundle_abi::BundleHandleV1) -> Option<usize> {
    INSTANCES.with(|table| {
        Some(
            table
                .borrow_mut()
                .get_mut(handle)?
                .dspark
                .as_ref()?
                .allocation
                .as_ref()?
                .ranks
                .len(),
        )
    })
}

/// The drafting items the instance's drafter runs per forward.
#[cfg(test)]
pub(crate) fn dspark_draft_chunk_items(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> Option<usize> {
    INSTANCES.with(|table| {
        Some(
            table
                .borrow_mut()
                .get_mut(handle)?
                .dspark
                .as_ref()?
                .draft_chunk_items,
        )
    })
}

/// The DSpark early gathers' host-hashed ids and their rows per layer.
#[cfg(test)]
pub(crate) fn dspark_early_ids(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> Option<(runtime_bundle_abi::DeviceBufferSpanV1, u32)> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let cycle = &table.get_mut(handle)?.dspark.as_ref()?.cycle;
        Some((cycle.early.as_ref()?.ids, cycle.hash.shape.stride))
    })
}

/// The DSpark device hash's ids and their rows per layer.
#[cfg(test)]
pub(crate) fn dspark_hash_ids(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> Option<(runtime_bundle_abi::DeviceBufferSpanV1, u32)> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let cycle = &table.get_mut(handle)?.dspark.as_ref()?.cycle;
        Some((cycle.memory?.ids, cycle.hash.shape.stride))
    })
}

/// Each rank's drafter noise plane and the instance's draft mode.
#[cfg(test)]
pub(crate) fn dspark_noise(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> Option<(
    Vec<runtime_bundle_abi::DeviceBufferSpanV1>,
    crate::lower::dspark::DraftMode,
)> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let device = table.get_mut(handle)?.dspark.as_ref()?;
        Some((
            device.ranks.iter().map(|rank| rank.noise).collect(),
            device.mode,
        ))
    })
}

#[cfg(test)]
pub(crate) use dspark::boundary::for_test as dspark_boundary_for_test;
#[cfg(test)]
pub(crate) use dspark::drafter_launches_at_depth_for_test as dspark_drafter_launches_at_depth;
#[cfg(test)]
pub(crate) use dspark::drafter_launches_for_test as dspark_drafter_launches;
#[cfg(test)]
pub(crate) use dspark::prefetch::for_test as dspark_prefetch_for_test;

#[cfg(test)]
pub(crate) fn fold_views_for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> Vec<BTreeMap<String, runtime_bundle_abi::DeviceBufferSpanV1>> {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("a live instance")
            .buffers
            .iter()
            .map(|buffers| {
                buffers
                    .views
                    .iter()
                    .filter(|(name, _)| name.contains(".fold_"))
                    .map(|(name, &span)| (name.clone(), span))
                    .collect()
            })
            .collect()
    })
}

/// Marks an instance's router flags as possibly holding image rows, as a
/// prefill does between uploading and clearing them.
#[cfg(test)]
pub(crate) fn set_image_flags_for_test(handle: runtime_bundle_abi::BundleHandleV1, set: bool) {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("instance")
            .image_flags
            .set(set);
    });
}

#[cfg(all(test, any(feature = "diagnostic-taps", feature = "state-capture")))]
pub(crate) fn capture_dspark_state_for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
    request: u64,
    root: &std::path::Path,
) -> Result<(), String> {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("a live instance")
            .capture_dspark_state_to(request, root)
    })
}

#[cfg(all(test, any(feature = "diagnostic-taps", feature = "state-capture")))]
pub(crate) fn set_dspark_state_capture_root_for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
    root: Option<std::path::PathBuf>,
) {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("a live instance")
            .state_capture_root = root;
    });
}

#[cfg(all(test, any(feature = "diagnostic-taps", feature = "state-capture")))]
pub(crate) fn dspark_state_capture_requests_for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> Vec<u64> {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("a live instance")
            .state_capture_pages
            .keys()
            .copied()
            .collect()
    })
}

/// Each rank's bound scratch, by declaration path.
#[cfg(test)]
pub(crate) fn scratch_for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> Vec<Vec<(String, runtime_bundle_abi::DeviceBufferSpanV1)>> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("a live instance");
        instance
            .buffers
            .iter()
            .map(|buffers| {
                let mut named = Vec::new();
                kernel_core::allocation::Traversal::visit_named(
                    &buffers.scratch,
                    &mut |name, &span| named.push((name.to_owned(), span)),
                );
                named
            })
            .collect()
    })
}

/// Keep allocation identities fixed while comparing the two attention routes.
#[cfg(test)]
pub(crate) fn prefill_cache_for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
    enabled: bool,
) -> Vec<runtime_bundle_abi::DeviceBufferSpanV1> {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("a live instance");
        instance
            .buffers
            .iter_mut()
            .filter_map(|buffers| {
                let cache = buffers.prefill_cache?;
                if !enabled {
                    buffers.prefill_cache = None;
                }
                Some(cache.data)
            })
            .collect()
    })
}

#[cfg(test)]
pub(crate) fn dspark_step_trace_for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
    enabled: bool,
) {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("a live instance")
            .dspark
            .as_mut()
            .expect("enabled drafter")
            .step_trace = enabled;
    });
}

#[cfg(test)]
pub(crate) fn retention_metadata_for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> Vec<runtime_bundle_abi::DeviceBufferSpanV1> {
    INSTANCES.with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("a live instance")
            .dspark
            .as_ref()
            .map(|device| {
                device
                    .ranks
                    .iter()
                    .filter_map(|rank| rank.retention_segments)
                    .collect()
            })
            .unwrap_or_default()
    })
}

#[cfg(test)]
pub(crate) fn dspark_graphs_for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
    capacity: Option<usize>,
) -> bundle_runtime::graphs::GraphStats {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("a live instance");
        if let Some(capacity) = capacity {
            instance.graphs.set_capacity(capacity).unwrap();
        }
        instance.graphs.stats()
    })
}

/// Releases every graph in the instance's ledger, so its next steps
/// capture afresh.
#[cfg(test)]
pub(crate) fn release_graphs_for_test(handle: runtime_bundle_abi::BundleHandleV1) {
    INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("a live instance");
        instance.graphs.release_all(&instance.services).unwrap();
    })
}
