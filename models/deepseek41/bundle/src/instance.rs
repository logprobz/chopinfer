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

mod dspark;
mod engram;
mod execute;
mod media;
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
    ranks: Vec<RankHandles>,
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
    /// Run the lowered stages alone (`PARTIAL`).
    partial: bool,
    /// Each rank's decode and verify graphs by bucket.
    graphs: GraphCache<DeepSeek>,
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
        // Each weight view an exact F32 projection or the head reads is
        // stored in the layout its op reads.
        let ops = &self.ops;
        let admit = |rank: u32, bytes: &[u8]| {
            let index =
                bundle_runtime::index::parse(bytes, rank, INDEX_SCHEMA).map_err(|error| {
                    let _ = services.log(format!("{NAME} rank index: {error}").as_bytes());
                    BundleAbiError::InvalidRange("deepseek41 rank index")
                })?;
            ops.check_weight_layouts(shape.geometry, &index)
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
        let limits = model_limits(&self.config);
        let layouts = [
            InputLayout::prefill_capacity_within::<DeepSeek>(items),
            InputLayout::decode_within(items).end,
            InputLayout::verify_within::<DeepSeek>(items).end,
        ];
        let tables = BlockTableMirror::<DeepSeek>::device_bytes(limits, items);
        let mut scratch = Vec::new();
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
        for &(_, pool, stream) in &ranks {
            for (phase, bytes) in layouts.into_iter().enumerate() {
                let arena = allocate_span(services, pool, stream, bytes, held)?;
                let block_tables = allocate_span(services, pool, stream, tables, held)?;
                self.arenas[phase].push(PhaseArena {
                    arena,
                    block_tables,
                });
            }
            let mut declared = Scratch::of_storage(&shape, self.config.kv.storage);
            if let Some(drafter) = &self.dspark {
                drafter.reserve_engram(&mut declared);
            }
            let bound = bundle_runtime::scratch::bind(declared, |bytes| {
                allocate_span(services, pool, stream, bytes, held)
            })?;
            services.copy_host_to_device(stream, bound.first_pre, &first_pre)?;
            scratch.push(bound);
            // The mixer scratch the mix-collapse lists; its tickets start
            // zeroed.
            let bytes = self
                .ops
                .hc
                .as_ref()
                .map(|ops| ops.scratch())
                .unwrap_or_default();
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
            let workspace = self
                .ops
                .workspace()
                .max(self.dspark.as_ref().map_or(0, |d| d.workspace()))
                .max(1)
                .next_multiple_of(256);
            let workspace = allocate_span(services, pool, stream, workspace, held)?;
            if let Some(drafter) = &mut self.dspark {
                drafter.allocate(services, pool, stream, held)?;
            }
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
            op_spans.push((sorted, workspace, table));
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
        let mut shared = self.ops.shared_bytes();
        if let Some(drafter) = &self.dspark {
            for (entry, bytes) in drafter.shared_bytes() {
                let most = shared.entry(entry).or_default();
                *most = (*most).max(bytes);
            }
        }
        for (rank, &(context, _, stream)) in ranks.iter().enumerate() {
            let mut resolved = BTreeMap::new();
            // The cubins of this geometry and the shared ones.
            let geometry = shape.geometry.id;
            for cubin in crate::CUBINS
                .iter()
                .filter(|cubin| crate::serves(cubin.name, geometry))
                .filter(|cubin| {
                    self.dspark.is_some()
                        || !crate::Kernels::drafter_only(cubin.name, shape.geometry)
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
            self.symbols.push(resolved);
        }
        // The vision encoder's cubins, image and working memory, before the
        // KV pool takes what is left.
        if let Some(kernels) = self.encoder_kernels.take() {
            self.media = Some(crate::encoders::acquire_media(
                services,
                &kernels,
                &ranks,
                &mut self.symbols,
                u64::from(crate::buffers::scratch_rows()),
                held,
            )?);
        }
        let (pages, pools) = acquire::serving_kv(
            log,
            self.config.kv.layout_id,
            self.config.page_bytes,
            shape.ranks(),
        )?;
        self.kv_pages = pages;
        for (
            ((((scratch, views), (kv_pool, state)), (comm, logits)), (mix_partials, mix_tickets)),
            (sorted_rows, workspace, rope_compressed),
        ) in scratch
            .into_iter()
            .zip(views)
            .zip(pools.into_iter().zip(state_views))
            .zip(comm.into_iter().zip(logits_comm))
            .zip(mix_scratch)
            .zip(op_spans)
        {
            self.buffers.push(RankBuffers {
                scratch,
                state: StateSpans::new(&self.state, extent, state)?,
                views,
                kv_pool,
                comm,
                logits,
                sorted_rows,
                workspace,
                mix_partials,
                mix_tickets,
                rope_compressed,
            });
        }
        if let Some(drafter) = &mut self.dspark {
            drafter.bind_stages(&self.buffers).map_err(|error| {
                let _ = services.log(error.as_bytes());
                BundleAbiError::InvalidRange("DSpark stage bindings")
            })?;
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
    let config = decode(bytes, log)?;
    let taps = crate::taps::StepTaps::from_env(&config).map_err(|error| {
        log(&error);
        BundleStatusV1::InvalidArgument
    })?;
    let items = config.create.items;
    let (verify_rows, mut verify) = verify_rows(&services, items)?;
    // The head scores at most this many rows per phase.
    bundle_runtime::config::check_scored_rows(
        &services,
        NAME,
        model_deepseek41_contract::serving::SCORED_ROWS,
    )?;
    let shape = config.shape;
    let has_drafter = shape.geometry.dspark_stages() > 0
        && config.create.state.bytes
            == u64::from(items.capacity())
                * crate::config::slot_state_bytes(&shape, config.kv.storage, true);
    let state = model_deepseek41_contract::state::request_state(
        shape.geometry,
        config.kv.storage,
        has_drafter,
    );
    if config.create.state.bytes
        != u64::from(items.capacity())
            * crate::config::slot_state_bytes(&shape, config.kv.storage, has_drafter)
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
    let kernels = crate::Kernels::new(&shape).map_err(|error| {
        log(&error);
        BundleStatusV1::Unsupported
    })?;
    let ops = Ops::prepare(&kernels, &shape, config.kv.storage);
    let dspark = if has_drafter {
        if config.kv.storage != model_deepseek41_contract::state::KvStorage::Native {
            log("DSpark requires the native KV layout");
            return Err(BundleStatusV1::Unsupported.into());
        }
        Some(
            crate::dspark::device::Device::new(
                &kernels,
                shape,
                items,
                config
                    .create
                    .context_tokens
                    .unwrap_or(shape.geometry.context_tokens()),
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
        ops,
        state,
        extent: Extent {
            served: items.get(),
            bucket: items.capacity(),
            verify_rows,
        },
        steps: std::cell::Cell::new(0),
        partial,
        taps,
        engram,
        dspark,
        image: None,
        encoder_kernels: media.then_some(kernels),
        media: None,
    };
    if let Err(error) = instance.acquire() {
        instance.log(&format!("create failed: {error}"));
        let result = status(error);
        return Err(Failed::releasing(instance, result));
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
        self.graphs.release_all(&self.services)
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

    fn before_retire(&mut self, _request: u64) -> Result<(), BundleStatusV1> {
        if self.dspark.as_ref().is_some_and(|device| {
            device
                .runtime
                .as_ref()
                .is_none_or(|runtime| runtime.state().is_pending() || runtime.state().is_poisoned())
        }) {
            return Err(BundleStatusV1::InvalidState);
        }
        Ok(())
    }

    /// The Engram history takes the rows a commit kept, and forgets a
    /// request whose slot a cancel freed.
    fn ended(&mut self, ended: Ended<'_>) -> Result<(), BundleStatusV1> {
        if self.dspark.is_some() {
            if let Ended::Cancelled(step) = ended {
                for item in &step.items {
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
