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
//! touched invalid, which the ledger retires.

mod execute;
mod prefix;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::thread::LocalKey;

use bundle_runtime::acquire::{self, Log, RankHandles};
use bundle_runtime::entry::{Bundle, Failed};
use bundle_runtime::geometry::{Items, ModelLimits};
use bundle_runtime::graphs::GraphCache;
use bundle_runtime::inputs::VerifyLimits;
use bundle_runtime::ledger::{Ledger, Placement, Rules};
use bundle_runtime::pipeline::Collectives;
use bundle_runtime::resources::{Held, allocate_span, status};
use bundle_runtime::state::{Extent, RequestState};
use bundle_runtime::table::{InstanceTable, Live};
use bundle_runtime::transaction::{self, Failure, Retire, Stage, Steps};
use bundle_runtime::upload::{BlockTableMirror, InputLayout, PhaseArena};
use model_deepseek41_contract::serving::{
    DeepSeek, INDEX_SCHEMA, MAX_VERIFY_ROWS, PAGE_TOKENS, RANK_IMAGE_ROLE, REQUEST_STATE_ID,
    index_role,
};
use runtime_bundle_abi::{
    BundleAbiError, BundleCudaSymbolHandleV1, BundleKeptRowsV1, BundleStatusV1,
    ExecuteResultSinkV1, HostServicesV1,
};

use crate::buffers::{RankBuffers, Scratch, StateSpans};
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
    services: HostServicesV1,
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
    /// Run the lowered stages alone (`PARTIAL`).
    partial: bool,
    /// Each rank's decode and verify graphs by bucket.
    graphs: GraphCache<DeepSeek>,
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
        let admit = |rank: u32, bytes: &[u8]| {
            bundle_runtime::index::parse(bytes, rank, INDEX_SCHEMA).map_err(|error| {
                let _ = services.log(format!("{NAME} rank index: {error}").as_bytes());
                BundleAbiError::InvalidRange("deepseek41 rank index")
            })
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
        let mut views = Vec::new();
        for (arena, index) in rank_arenas.into_iter().zip(&indexes) {
            let placed =
                acquire::place_views(services, image, arena, index, &[]).map_err(|error| {
                    match error {
                        acquire::PlaceError::Host(error) => error,
                        acquire::PlaceError::Layout(message) => {
                            let _ = services.log(message.as_bytes());
                            BundleAbiError::InvalidRange("deepseek41 rank arena length")
                        }
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
        for &(_, pool, stream) in &ranks {
            for (phase, bytes) in layouts.into_iter().enumerate() {
                let arena = allocate_span(services, pool, stream, bytes, held)?;
                let block_tables = allocate_span(services, pool, stream, tables, held)?;
                self.arenas[phase].push(PhaseArena {
                    arena,
                    block_tables,
                });
            }
            scratch.push(bundle_runtime::scratch::bind(
                Scratch::of(&shape),
                |bytes| allocate_span(services, pool, stream, bytes, held),
            )?);
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
        for &(context, _, _) in &ranks {
            let mut resolved = BTreeMap::new();
            for cubin in crate::CUBINS {
                acquire::load_module(
                    log,
                    context,
                    (cubin.name, cubin.bytes),
                    cubin.symbols.iter().copied(),
                    |_| Ok(acquire::DEFAULT_SHARED_BYTES),
                    &mut resolved,
                    held,
                )?;
            }
            self.symbols.push(resolved);
        }
        let (pages, pools) = acquire::serving_kv(
            log,
            self.config.kv.layout_id,
            self.config.page_bytes,
            shape.ranks(),
        )?;
        self.kv_pages = pages;
        for ((scratch, views), (kv_pool, state)) in scratch
            .into_iter()
            .zip(views)
            .zip(pools.into_iter().zip(state_views))
        {
            self.buffers.push(RankBuffers {
                scratch,
                state: StateSpans::new(&self.state, extent, state)?,
                views,
                kv_pool,
            });
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
    let items = config.create.items;
    let (verify_rows, verify) = verify_rows(&services, items)?;
    // The head scores at most this many rows per phase.
    bundle_runtime::config::check_scored_rows(
        &services,
        NAME,
        model_deepseek41_contract::serving::SCORED_ROWS,
    )?;
    let shape = config.shape;
    // The DSpark drafter's state joins with the drafter.
    let state =
        model_deepseek41_contract::state::request_state(shape.geometry, config.kv.storage, false);
    if config.create.state.bytes
        != u64::from(items.capacity())
            * crate::config::slot_state_bytes(&shape, config.kv.storage, false)
    {
        log(
            "the per-item state the host allocates is the drafter's, which the bundle cannot run yet",
        );
        return Err(BundleStatusV1::Unsupported.into());
    }
    // Every op, prepared once on the target's kernels.
    let kernels = crate::Kernels::new(&shape).map_err(|error| {
        log(&error);
        BundleStatusV1::Unsupported
    })?;
    let ops = Ops::prepare(&kernels, &shape);
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
        services,
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
        partial,
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
}
