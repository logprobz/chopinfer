//! The instance lifecycle behind this fixture's one native entry point.
//!
//! Instances live in a thread-local table because a host loads one bundle at
//! process start on one thread and keeps it there. Each handle carries a slot
//! and a generation this table issued, so a handle from a destroyed instance, or
//! one this fixture never issued, is refused before it reaches a resource.
//!
//! One instance serves the whole declared TP4 topology. It opens a context, a
//! memory pool, a stage stream, and a rank arena for each of the four ranks,
//! authenticates each rank's uploaded image against the bytes this build can
//! regenerate, and allocates one rank-local KV pool across all four. Every
//! execution writes its tokens into the pages the host's own page table places
//! them in, on every rank, and reads them back; the one row it emits is read
//! back from the emitting rank, whose view of the declared persistent state also
//! holds the committed history.
//!
//! One execution at a time. This instance holds one open transaction and refuses
//! a second, so it serves one sequence and proves nothing about concurrent ones.
//!
//! Not every scheduled step emits. A prompt longer than one PrefillChunk arrives as
//! full chunks the engine expects no row from, then one authenticated tail it
//! does. A chunk runs the same model side and stages the same history; it just
//! has no row to hand back, and this instance hands none back rather than
//! computing one nothing asked for.
//!
//! The transaction has two halves and both of them are the host's. One is the KV
//! append the host prepares, commits, and rolls back through its own paged pool;
//! the other is the folded token history this instance publishes into the
//! persistent state the manifest declared. An execution opens the append, writes
//! every rank's pages, computes what it owes, and stages the history that append
//! would produce. `commit` publishes the history and then commits the append;
//! `cancel` cancels the append and publishes nothing, so a cancelled execution
//! leaves this instance exactly where the last commit left it and replaying it
//! produces the same row. A commit the host refuses in either half puts the
//! history back and leaves the execution open for its owner to finish.
//!
//! `destroy` is not a rollback. An instance with an execution still open is
//! refused, because a caller that has not finalized its transaction has not
//! finished with the instance, and silently cancelling for it would report a
//! transaction as resolved that its owner never resolved. The refusal changes
//! nothing, so a cancel or a commit on the same handle still works and a destroy
//! after that succeeds.

use std::cell::RefCell;

use runtime_bundle_abi::{
    BundleAbiError, BundleCompiledImageHandleV1, BundleContextHandleV1, BundleCreateResultV1,
    BundleDeviceBufferHandleV1, BundleHandleV1, BundleKeptRowsV1, BundleKvRequestHandleV1,
    BundleKvTransactionHandleV1, BundleMemoryPoolHandleV1, BundlePhysicalKvHandleV1,
    BundlePrefixStateInfoV1, BundleRankArenaHandleV1, BundleStateHandleV1, BundleStatusV1,
    BundleStreamHandleV1, CustomCompletionV1, CustomOperandV1, DeviceBufferSpanV1,
    ExecuteResultSinkV1, HostServicesV1, PhysicalKvLayoutV1, PhysicalKvRequestV1,
    PrefixStateRequestV1, StateRequestV1,
};
use runtime_weights::sha256_bytes;

use crate::configuration::BundleConfigurationV1;
use crate::contract::{
    ProgramGeometryV1, TP4_FIXTURE_BUFFER_ALIGNMENT_V1, TP4_FIXTURE_DECODE_STEP_TOKENS_V1,
    TP4_FIXTURE_EMITTING_RANK_V1, TP4_FIXTURE_GEOMETRY_BYTES_V1, TP4_FIXTURE_GEOMETRY_VIEW_V1,
    TP4_FIXTURE_KV_BYTES_PER_TOKEN_V1, TP4_FIXTURE_KV_PAGE_BYTES_V1,
    TP4_FIXTURE_KV_PAGE_CAPACITY_V1, TP4_FIXTURE_KV_PAGE_TOKENS_V1, TP4_FIXTURE_KV_QUOTA_TOKENS_V1,
    TP4_FIXTURE_KV_REQUEST_CAPACITY_V1, TP4_FIXTURE_MARKER_ROLE_V1, TP4_FIXTURE_MAXIMUM_TOKENS_V1,
    TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1, TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1,
    TP4_FIXTURE_PROGRAM_ROLE_V1, TP4_FIXTURE_RANK_COUNT_V1, TP4_FIXTURE_ROW_OPERATION_V1,
    TP4_FIXTURE_ROW_TOKEN_COUNT_BYTES_V1, TP4_FIXTURE_STAGE_ID_V1, TP4_FIXTURE_STATE_SEED_VIEW_V1,
    TP4_FIXTURE_STREAM_KIND_V1, TP4_FIXTURE_TOPOLOGY_ID_V1, TP4_FIXTURE_VOCABULARY_BOUND_V1,
};
use crate::operation::{
    StagedRowV1, TP4_FIXTURE_ROW_OPERATION_ID_V1, Tp4RowOperationV1, bind_tp4_row_v1,
};
use crate::synthetic::{
    TokenHistoryV1, fold_tokens_v1, history_from_bytes_v1, history_to_bytes_v1,
    program_image_digest_v1, state_seed_history_v1,
};
use runtime_bundle_sdk::{BatchCodecV1, BatchRequestV1};

/// The whole rank image, which the host selects with an empty view identifier.
const COMPILED_IMAGE_ARENA_VIEW_V1: &[u8] = b"";

/// The KV operation word the host declared for a token append.
///
/// The payload is this word, then the additional token count as eight
/// little-endian bytes. Nothing in it names a model.
const KV_APPEND_OPERATION_V1: &[u8] = b"append-tokens-v1";

/// One host resource this instance holds, in the order it acquired them.
///
/// The host releases in strict reverse order, so this is a stack: the newest
/// resource is the only one that may be given back next.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HeldV1 {
    Context(BundleContextHandleV1),
    MemoryPool(BundleMemoryPoolHandleV1),
    Stream(BundleStreamHandleV1),
    CompiledImage(BundleCompiledImageHandleV1),
    RankArena(BundleRankArenaHandleV1),
    DeviceBuffer(BundleDeviceBufferHandleV1),
    State(BundleStateHandleV1),
    PhysicalKv(BundlePhysicalKvHandleV1),
    KvRequest(BundleKvRequestHandleV1),
    PageKv(BundlePhysicalKvHandleV1),
    PageRequest(BundleKvRequestHandleV1),
}

fn release_one_v1(services: &HostServicesV1, held: HeldV1) -> Result<(), BundleAbiError> {
    match held {
        HeldV1::Context(context) => services.release_context(context),
        HeldV1::MemoryPool(pool) => services.release_memory_pool(pool),
        HeldV1::Stream(stream) => services.release_stream(stream),
        HeldV1::CompiledImage(image) => services.release_compiled_image(image),
        HeldV1::RankArena(arena) => services.release_rank_arena(arena),
        HeldV1::DeviceBuffer(buffer) => services.release_device_buffer(buffer),
        HeldV1::State(state) => services.release_state(state),
        HeldV1::PhysicalKv(kv) => services.release_physical_kv(kv),
        HeldV1::KvRequest(request) => services.release_kv_request(request),
        HeldV1::PageKv(kv) => services.release_physical_kv(kv),
        HeldV1::PageRequest(request) => services.release_kv_request(request),
    }
}

/// Releases every held resource newest first, keeping whatever is refused.
///
/// A host that refuses one release leaves that resource and every older one on
/// the stack, so the caller holds exactly what the host still owns and the same
/// release can be retried later.
fn release_all_v1(services: &HostServicesV1, held: &mut Vec<HeldV1>) -> Result<(), BundleStatusV1> {
    while let Some(resource) = held.last().copied() {
        release_one_v1(services, resource).map_err(status_of_v1)?;
        held.pop();
    }
    Ok(())
}

/// One rank's own host resources, in the order this instance acquired them.
#[derive(Clone, Copy, Debug)]
struct RankHandlesV1 {
    context: BundleContextHandleV1,
    pool: BundleMemoryPoolHandleV1,
    stream: BundleStreamHandleV1,
    arena: BundleRankArenaHandleV1,
}

/// The handles one created instance keeps for its whole life.
///
/// The rank fields are the emitting rank's: the buffers the one emitted row
/// passes through are allocated from its pool and ordered against its stream.
/// Every rank's own handles live in [`Tp4ResourcesV1::ranks`].
#[derive(Clone, Copy, Debug)]
struct AcquiredHandlesV1 {
    context: BundleContextHandleV1,
    stream: BundleStreamHandleV1,
    /// The fixed-capacity token operand the private operation reads.
    input: BundleDeviceBufferHandleV1,
    /// The operand that says how many of those token slots are live.
    token_count: BundleDeviceBufferHandleV1,
    logits: BundleDeviceBufferHandleV1,
    state: BundleStateHandleV1,
    /// The request every execution appends against: the fixture's own KV
    /// accounting. Its pool numbers its own pages and zeroes each one it
    /// newly reserves.
    kv_request: BundleKvRequestHandleV1,
    /// A request over another physical KV, only ever resolved for page views:
    /// the pages the host places tokens and snapshots in. It never appends,
    /// so the accounting pool's zeroing cannot erase them.
    page_request: BundleKvRequestHandleV1,
    geometry: ProgramGeometryV1,
    state_bytes: u64,
    input_bytes: u64,
    token_count_bytes: u64,
    row_bytes: u64,
}

/// The append and the history one execution is holding open.
#[derive(Clone, Debug)]
struct PendingExecutionV1 {
    transaction: BundleKvTransactionHandleV1,
    /// The history a commit publishes.
    history: TokenHistoryV1,
    /// The history that was committed before this execution ran.
    previous_history: TokenHistoryV1,
    request_id: u64,
    verify_tokens: Option<Vec<u32>>,
}

/// Everything one instance holds from the host, and its open transaction.
struct Tp4ResourcesV1 {
    handles: AcquiredHandlesV1,
    /// Every rank's own resources, in rank order.
    ranks: Vec<RankHandlesV1>,
    held: Vec<HeldV1>,
    pending: Option<PendingExecutionV1>,
    current_request_id: Option<u64>,
    /// Every rank image's digest, in rank order, as authenticated at create.
    image_digests: Vec<String>,
}

/// Why an acquisition failed, and what the host would not take back.
struct AcquireFailureV1 {
    status: BundleStatusV1,
    unreleased: Vec<HeldV1>,
}

/// One created instance: its service table, its resources, and its operation.
struct InstanceV1 {
    services: HostServicesV1,
    resources: Tp4ResourcesV1,
    operation: Tp4RowOperationV1,
}

struct SlotV1 {
    generation: u32,
    instance: Option<InstanceV1>,
}

/// A generation-checked table of this fixture's live instances.
struct InstanceTableV1 {
    slots: Vec<SlotV1>,
    next_generation: u32,
}

impl InstanceTableV1 {
    const fn new() -> Self {
        Self {
            slots: Vec::new(),
            next_generation: 1,
        }
    }

    fn next_generation(&mut self) -> Option<u32> {
        let generation = self.next_generation;
        // Generation zero is the invalid handle, so it is never issued and an
        // exhausted counter refuses rather than wrapping onto a live handle.
        self.next_generation = self.next_generation.checked_add(1)?;
        Some(generation)
    }

    /// Takes one instance and returns its handle, or hands the instance back.
    ///
    /// A refusal must not consume the instance: the resources it acquired are
    /// still live, and the caller is the only thing left that can release them.
    fn insert(&mut self, instance: Box<InstanceV1>) -> Result<BundleHandleV1, Box<InstanceV1>> {
        let free = self
            .slots
            .iter()
            .position(|slot| slot.instance.is_none())
            .unwrap_or(self.slots.len());
        let Some(slot) = free
            .checked_add(1)
            .and_then(|slot| u32::try_from(slot).ok())
        else {
            return Err(instance);
        };
        let Some(generation) = self.next_generation() else {
            return Err(instance);
        };
        let occupied = SlotV1 {
            generation,
            instance: Some(*instance),
        };
        if free == self.slots.len() {
            self.slots.push(occupied);
        } else {
            self.slots[free] = occupied;
        }
        // A slot ordinal is one-based, so slot zero stays the invalid handle.
        Ok(BundleHandleV1::from_parts(generation, slot))
    }

    fn position(&self, handle: BundleHandleV1) -> Option<usize> {
        if !handle.has_generation() {
            return None;
        }
        let index = usize::try_from(handle.slot()).ok()?.checked_sub(1)?;
        let slot = self.slots.get(index)?;
        if slot.generation != handle.generation() || slot.instance.is_none() {
            return None;
        }
        Some(index)
    }

    fn get_mut(&mut self, handle: BundleHandleV1) -> Option<&mut InstanceV1> {
        let index = self.position(handle)?;
        self.slots[index].instance.as_mut()
    }

    fn remove(&mut self, handle: BundleHandleV1) -> Option<InstanceV1> {
        let index = self.position(handle)?;
        self.slots[index].instance.take()
    }
}

thread_local! {
    static INSTANCES_V1: RefCell<InstanceTableV1> = const { RefCell::new(InstanceTableV1::new()) };
}

// Resources the host refused to take back, kept for a later attempt. These
// belong to no live instance, so no handle names them and nothing can use them.
// Dropping them would leak them silently instead, so they are held here and
// every later lifecycle call tries the release again.
thread_local! {
    static UNRELEASED_V1: RefCell<Vec<(HostServicesV1, Vec<HeldV1>)>> =
        const { RefCell::new(Vec::new()) };
}

/// Maps one ABI refusal onto the status this fixture reports for it.
fn status_of_v1(error: BundleAbiError) -> BundleStatusV1 {
    match error {
        // The host answered with a status of its own, which is the answer.
        BundleAbiError::Status(status) => status,
        // This fixture's own operation refused, and the code is its vocabulary.
        BundleAbiError::CustomOperation(_) => BundleStatusV1::BundleError,
        // A service this fixture needs is one the host does not offer.
        BundleAbiError::MissingHostService(_) => BundleStatusV1::Unsupported,
        // Everything else is a table or a value the host produced.
        _ => BundleStatusV1::HostError,
    }
}

fn log_v1(services: &HostServicesV1, message: &str) {
    // A host that refuses a diagnostic changes nothing about the operation.
    let _ = services.log(message.as_bytes());
}

/// Retries every release the host has previously refused.
///
/// This runs at the start of each lifecycle call because a host that recovers
/// says so only by accepting a release, and there is nothing else to ask.
fn sweep_unreleased_v1() {
    let mut held = UNRELEASED_V1.with(|held| std::mem::take(&mut *held.borrow_mut()));
    held.retain_mut(
        |(services, resources)| match release_all_v1(services, resources) {
            Ok(()) => {
                log_v1(
                    services,
                    "TP4 fixture returned resources the host had previously refused",
                );
                false
            }
            Err(_) => true,
        },
    );
    UNRELEASED_V1.with(|slot| slot.borrow_mut().extend(held));
}

/// Reports how many refused releases this fixture is still holding.
#[must_use]
pub fn unreleased_resource_count_v1() -> usize {
    UNRELEASED_V1.with(|held| held.borrow().len())
}

/// Creates one instance for the topology the host configured.
pub fn create_v1(services: HostServicesV1, configuration: &[u8]) -> BundleCreateResultV1 {
    sweep_unreleased_v1();
    match create_instance_v1(services, configuration) {
        Ok(handle) => BundleCreateResultV1::success(handle),
        Err(status) => BundleCreateResultV1::failure(status),
    }
}

fn create_instance_v1(
    services: HostServicesV1,
    configuration: &[u8],
) -> Result<BundleHandleV1, BundleStatusV1> {
    let configuration = match BundleConfigurationV1::decode(configuration) {
        Ok(configuration) => configuration,
        Err(error) => {
            log_v1(
                &services,
                &format!("TP4 fixture refused its configuration: {error}"),
            );
            return Err(BundleStatusV1::InvalidArgument);
        }
    };
    // Nothing is acquired before the declaration is checked, so a host this
    // fixture does not serve costs it no resource at all.
    if let Err(refusal) = check_configuration_v1(&configuration) {
        log_v1(
            &services,
            &format!("TP4 fixture refused this host: {}", refusal.reason),
        );
        return Err(refusal.status);
    }
    let resources = match acquire_v1(&services, &configuration) {
        Ok(resources) => resources,
        Err(failure) => {
            log_v1(
                &services,
                "TP4 fixture could not acquire what its declaration asks for",
            );
            if !failure.unreleased.is_empty() {
                return Err(retain_unreleased_v1(
                    services,
                    failure.unreleased,
                    failure.status,
                ));
            }
            return Err(failure.status);
        }
    };
    log_v1(&services, &created_message_v1(&resources));
    log_v1(&services, &authenticated_images_message_v1(&resources));
    log_v1(&services, &claims_message_v1());
    let instance = Box::new(InstanceV1 {
        services,
        resources,
        operation: Tp4RowOperationV1::new(),
    });
    // A table that cannot issue a handle would otherwise drop every resource
    // this instance just acquired, so the acquisition is undone first. A table
    // an outer lifecycle call is holding cannot issue one either, and that is
    // the same refusal rather than a panic.
    match INSTANCES_V1.with(|table| match table.try_borrow_mut() {
        Ok(mut table) => table.insert(instance),
        Err(_) => Err(instance),
    }) {
        Ok(handle) => Ok(handle),
        Err(mut instance) => {
            log_v1(
                &instance.services,
                "TP4 fixture exhausted its instance table and is releasing what it \
                 acquired",
            );
            match release_all_v1(&instance.services, &mut instance.resources.held) {
                Ok(()) => Err(BundleStatusV1::ResourceExhausted),
                Err(error) => {
                    log_v1(
                        &instance.services,
                        &format!(
                            "TP4 fixture could not release what it acquired: the host \
                             answered status {}",
                            error.0
                        ),
                    );
                    let InstanceV1 {
                        services,
                        resources,
                        ..
                    } = *instance;
                    Err(retain_unreleased_v1(
                        services,
                        resources.held,
                        BundleStatusV1::ResourceExhausted,
                    ))
                }
            }
        }
    }
}

fn retain_unreleased_v1(
    services: HostServicesV1,
    held: Vec<HeldV1>,
    status: BundleStatusV1,
) -> BundleStatusV1 {
    log_v1(
        &services,
        "TP4 fixture is holding the resources the host refused to take back, and will \
         try again on its next lifecycle call",
    );
    UNRELEASED_V1.with(|held_resources| held_resources.borrow_mut().push((services, held)));
    status
}

/// Why one host configuration is not one this fixture serves.
struct ConfigurationRefusalV1 {
    status: BundleStatusV1,
    reason: String,
}

/// Checks the declaration the host encoded before anything is acquired.
fn check_configuration_v1(
    configuration: &BundleConfigurationV1,
) -> Result<(), ConfigurationRefusalV1> {
    let refuse = |status, reason: String| Err(ConfigurationRefusalV1 { status, reason });
    if configuration.topology_id != TP4_FIXTURE_TOPOLOGY_ID_V1.as_bytes() {
        return refuse(
            BundleStatusV1::Unsupported,
            format!(
                "this fixture serves topology {TP4_FIXTURE_TOPOLOGY_ID_V1} and the host \
                 configured '{}'",
                String::from_utf8_lossy(&configuration.topology_id)
            ),
        );
    }
    if configuration.rank_count() != TP4_FIXTURE_RANK_COUNT_V1 as usize {
        return refuse(
            BundleStatusV1::Unsupported,
            format!(
                "this fixture serves exactly {TP4_FIXTURE_RANK_COUNT_V1} ranks and the host \
                 offered {}",
                configuration.rank_count()
            ),
        );
    }
    if !configuration.declares_component_role(TP4_FIXTURE_PROGRAM_ROLE_V1.as_bytes()) {
        return refuse(
            BundleStatusV1::NotFound,
            format!("the package declares no component for role {TP4_FIXTURE_PROGRAM_ROLE_V1}"),
        );
    }
    // The marker role is the gate. A package that does not declare itself this
    // fixture is not one, however its other bytes happen to be shaped, and this
    // module refuses to run a synthetic row on its behalf.
    if !configuration.declares_component_role(TP4_FIXTURE_MARKER_ROLE_V1.as_bytes()) {
        return refuse(
            BundleStatusV1::Unsupported,
            format!(
                "the package declares no {TP4_FIXTURE_MARKER_ROLE_V1} component, so it is not \
                 this fixture and this module computes nothing for it"
            ),
        );
    }
    if configuration.state.byte_length == 0 {
        return refuse(
            BundleStatusV1::Unsupported,
            "the package declares no persistent state, which is where a committed history lives"
                .to_owned(),
        );
    }
    // Where one token lands is arithmetic over the declared page, so a host that
    // configured a different page is one this module would place tokens in the
    // wrong bytes of. It refuses instead of computing against another geometry.
    if configuration.kv.page_tokens != u64::from(TP4_FIXTURE_KV_PAGE_TOKENS_V1)
        || configuration.kv.page_bytes != TP4_FIXTURE_KV_PAGE_BYTES_V1
    {
        return refuse(
            BundleStatusV1::Unsupported,
            format!(
                "this fixture places tokens in {TP4_FIXTURE_KV_PAGE_TOKENS_V1}-token pages of \
                 {TP4_FIXTURE_KV_PAGE_BYTES_V1} bytes and the host configured {}-token pages of \
                 {} bytes",
                configuration.kv.page_tokens, configuration.kv.page_bytes
            ),
        );
    }
    Ok(())
}

/// Acquires everything one instance needs, unwinding whatever it took first.
fn acquire_v1(
    services: &HostServicesV1,
    configuration: &BundleConfigurationV1,
) -> Result<Tp4ResourcesV1, AcquireFailureV1> {
    let mut held = Vec::new();
    match acquire_handles_v1(services, configuration, &mut held) {
        Ok(acquired) => Ok(Tp4ResourcesV1 {
            handles: acquired.handles,
            ranks: acquired.ranks,
            held,
            pending: None,
            current_request_id: None,
            image_digests: acquired.image_digests,
        }),
        Err(status) => match release_all_v1(services, &mut held) {
            Ok(()) => Err(AcquireFailureV1 {
                status,
                unreleased: Vec::new(),
            }),
            Err(_) => Err(AcquireFailureV1 {
                status,
                unreleased: held,
            }),
        },
    }
}

/// Everything one acquisition produced, before it becomes an instance.
struct AcquiredV1 {
    handles: AcquiredHandlesV1,
    ranks: Vec<RankHandlesV1>,
    image_digests: Vec<String>,
}

/// The whole acquisition, in the order the host must see it released back.
///
/// The compiled image opens first and is therefore released last, because every
/// rank arena is a view of it. Each rank then takes its own context, pool,
/// stream, and arena, so a host that refuses part way through hands back exactly
/// the ranks that were opened and nothing more.
fn acquire_handles_v1(
    services: &HostServicesV1,
    configuration: &BundleConfigurationV1,
    held: &mut Vec<HeldV1>,
) -> Result<AcquiredV1, BundleStatusV1> {
    let image = services
        .open_compiled_image(TP4_FIXTURE_PROGRAM_ROLE_V1.as_bytes())
        .map_err(status_of_v1)?;
    held.push(HeldV1::CompiledImage(image));

    let mut ranks = Vec::with_capacity(configuration.rank_count());
    for (rank, ordinal) in configuration.device_ordinals.iter().copied().enumerate() {
        let rank = u32::try_from(rank).map_err(|_| BundleStatusV1::InvalidArgument)?;
        let context = services
            .select_context(rank, ordinal)
            .map_err(status_of_v1)?;
        held.push(HeldV1::Context(context));
        let pool = services.select_memory_pool(context).map_err(status_of_v1)?;
        held.push(HeldV1::MemoryPool(pool));
        // One stage, so one stream per rank: every transfer this instance makes
        // on that rank is ordered against the same one.
        let stream = services
            .select_stage_stream(
                context,
                rank,
                TP4_FIXTURE_STAGE_ID_V1,
                TP4_FIXTURE_STREAM_KIND_V1,
            )
            .map_err(status_of_v1)?;
        held.push(HeldV1::Stream(stream));
        let arena = services
            .upload_rank_arena(context, pool, stream, image, rank)
            .map_err(status_of_v1)?;
        held.push(HeldV1::RankArena(arena));
        ranks.push(RankHandlesV1 {
            context,
            pool,
            stream,
            arena,
        });
    }

    let image_digests = authenticate_images_v1(services, image, &ranks)?;

    let emitting = *ranks
        .get(TP4_FIXTURE_EMITTING_RANK_V1 as usize)
        .ok_or(BundleStatusV1::InvalidArgument)?;

    // The width of a row and the length of a prompt are the package's to
    // declare, so they are read out of the image the host just uploaded rather
    // than compiled into this module.
    let geometry_span = services
        .resolve_compiled_image_view(
            image,
            emitting.arena,
            TP4_FIXTURE_EMITTING_RANK_V1,
            TP4_FIXTURE_GEOMETRY_VIEW_V1,
        )
        .map_err(status_of_v1)?;
    if geometry_span.buffer.length != TP4_FIXTURE_GEOMETRY_BYTES_V1 {
        return Err(BundleStatusV1::InvalidArgument);
    }
    let mut geometry_bytes = vec![0_u8; TP4_FIXTURE_GEOMETRY_BYTES_V1 as usize];
    services
        .copy_device_to_host(emitting.stream, &mut geometry_bytes, geometry_span.buffer)
        .map_err(status_of_v1)?;
    services
        .synchronize_stream(emitting.stream)
        .map_err(status_of_v1)?;
    let geometry =
        ProgramGeometryV1::from_bytes(&geometry_bytes).ok_or(BundleStatusV1::InvalidArgument)?;
    check_geometry_v1(geometry, configuration.state.byte_length)?;

    let state_bytes = configuration.state.byte_length;
    // The token operand is one whole PrefillChunk of identifiers whatever a batch
    // carries, because that is the shape the manifest declares for it.
    let input_bytes = u64::from(geometry.prefill_tokens) * 4 * configuration.scratch_multiplier;
    let token_count_bytes = TP4_FIXTURE_ROW_TOKEN_COUNT_BYTES_V1;
    let row_bytes = geometry.row_bytes();
    let input = services
        .allocate_device_buffer(
            emitting.pool,
            emitting.stream,
            input_bytes,
            TP4_FIXTURE_BUFFER_ALIGNMENT_V1,
        )
        .map_err(status_of_v1)?;
    held.push(HeldV1::DeviceBuffer(input));
    let token_count = services
        .allocate_device_buffer(
            emitting.pool,
            emitting.stream,
            token_count_bytes,
            TP4_FIXTURE_BUFFER_ALIGNMENT_V1,
        )
        .map_err(status_of_v1)?;
    held.push(HeldV1::DeviceBuffer(token_count));
    let logits = services
        .allocate_device_buffer(
            emitting.pool,
            emitting.stream,
            row_bytes,
            TP4_FIXTURE_BUFFER_ALIGNMENT_V1,
        )
        .map_err(status_of_v1)?;
    held.push(HeldV1::DeviceBuffer(logits));

    let state = services
        .allocate_state(&StateRequestV1 {
            flags: 0,
            byte_length: state_bytes,
            alignment: configuration.state.alignment,
            state_id: configuration.state.state_id.clone(),
        })
        .map_err(status_of_v1)?;
    held.push(HeldV1::State(state));

    // The initial history comes out of the package, not out of this module: the
    // seed is a view of the same authenticated image, copied across on the
    // device without ever passing through host memory.
    let seed_span = services
        .resolve_compiled_image_view(
            image,
            emitting.arena,
            TP4_FIXTURE_EMITTING_RANK_V1,
            TP4_FIXTURE_STATE_SEED_VIEW_V1,
        )
        .map_err(status_of_v1)?;
    let state_span = services
        .resolve_state_view(state, TP4_FIXTURE_EMITTING_RANK_V1)
        .map_err(status_of_v1)?;
    if seed_span.buffer.length != state_bytes || state_span.length != state_bytes {
        return Err(BundleStatusV1::InvalidArgument);
    }
    services
        .copy_device_to_device(emitting.stream, state_span, seed_span.buffer)
        .map_err(status_of_v1)?;
    services
        .synchronize_stream(emitting.stream)
        .map_err(status_of_v1)?;

    // The pages the host places tokens and snapshots in, by its own page
    // numbers. Held below the accounting request, which retirement replaces.
    let page_kv = services
        .allocate_physical_kv(&physical_kv_request_v1(configuration))
        .map_err(status_of_v1)?;
    held.push(HeldV1::PageKv(page_kv));
    let page_request = services
        .create_kv_request(page_kv, TP4_FIXTURE_KV_QUOTA_TOKENS_V1)
        .map_err(status_of_v1)?;
    held.push(HeldV1::PageRequest(page_request));

    // The paged pool this fixture's transaction is made of, rank-local across
    // the whole topology, and the one request every execution appends against.
    // The host does not return a released request's sequence to its pool, so
    // the request is opened once here.
    let kv = services
        .allocate_physical_kv(&physical_kv_request_v1(configuration))
        .map_err(status_of_v1)?;
    held.push(HeldV1::PhysicalKv(kv));
    let kv_request = services
        .create_kv_request(kv, TP4_FIXTURE_KV_QUOTA_TOKENS_V1)
        .map_err(status_of_v1)?;
    held.push(HeldV1::KvRequest(kv_request));

    Ok(AcquiredV1 {
        handles: AcquiredHandlesV1 {
            context: emitting.context,
            stream: emitting.stream,
            input,
            token_count,
            logits,
            state,
            kv_request,
            page_request,
            geometry,
            state_bytes,
            input_bytes,
            token_count_bytes,
            row_bytes,
        },
        ranks,
        image_digests,
    })
}

/// Builds the exact physical KV request the installed manifest declared.
fn physical_kv_request_v1(configuration: &BundleConfigurationV1) -> PhysicalKvRequestV1 {
    PhysicalKvRequestV1 {
        flags: 0,
        layout: PhysicalKvLayoutV1 {
            flags: 0,
            rank_count: configuration.kv.rank_count,
            page_bytes: configuration.kv.page_bytes,
            page_alignment: configuration.kv.page_alignment,
            page_tokens: configuration.kv.page_tokens,
            state_lifetime: configuration.kv.state_lifetime,
            layout_id: configuration.kv.layout_id.clone(),
        },
        page_capacity: TP4_FIXTURE_KV_PAGE_CAPACITY_V1,
        request_capacity: TP4_FIXTURE_KV_REQUEST_CAPACITY_V1,
    }
}

/// Reads every uploaded rank image back and refuses one this build did not make.
///
/// The row this fixture emits is a function of these digests, so authenticating
/// each rank is what ties the row to bytes the host verified rather than to
/// whatever happened to be installed under the program role. Every rank is read
/// over its own stream, so a topology missing one is a refusal rather than a row
/// computed from three ranks and a gap.
fn authenticate_images_v1(
    services: &HostServicesV1,
    image: BundleCompiledImageHandleV1,
    ranks: &[RankHandlesV1],
) -> Result<Vec<String>, BundleStatusV1> {
    if ranks.len() != TP4_FIXTURE_RANK_COUNT_V1 as usize {
        return Err(BundleStatusV1::Unsupported);
    }
    let mut digests = Vec::with_capacity(ranks.len());
    for (rank, handles) in ranks.iter().enumerate() {
        let rank = u32::try_from(rank).map_err(|_| BundleStatusV1::InvalidArgument)?;
        let span = services
            .resolve_compiled_image_view(image, handles.arena, rank, COMPILED_IMAGE_ARENA_VIEW_V1)
            .map_err(status_of_v1)?;
        if span.buffer.length != TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1 {
            return Err(BundleStatusV1::Unsupported);
        }
        let mut image_bytes = vec![0_u8; TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1 as usize];
        services
            .copy_device_to_host(handles.stream, &mut image_bytes, span.buffer)
            .map_err(status_of_v1)?;
        services
            .synchronize_stream(handles.stream)
            .map_err(status_of_v1)?;
        let digest = sha256_bytes(&image_bytes);
        if digest != program_image_digest_v1(rank) {
            log_v1(
                services,
                &format!(
                    "TP4 fixture refused a rank {rank} program image this build cannot \
                     regenerate"
                ),
            );
            return Err(BundleStatusV1::Unsupported);
        }
        digests.push(digest);
    }
    Ok(digests)
}

/// Refuses a program image that does not describe the state the manifest does.
fn check_geometry_v1(geometry: ProgramGeometryV1, state_bytes: u64) -> Result<(), BundleStatusV1> {
    if u64::from(geometry.state_bytes) != state_bytes {
        return Err(BundleStatusV1::Unsupported);
    }
    if geometry.vocabulary == 0 || geometry.vocabulary > TP4_FIXTURE_VOCABULARY_BOUND_V1 {
        return Err(BundleStatusV1::Unsupported);
    }
    if geometry.prefill_tokens != TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1
        || geometry.decode_tokens != TP4_FIXTURE_DECODE_STEP_TOKENS_V1
    {
        return Err(BundleStatusV1::Unsupported);
    }
    Ok(())
}

fn created_message_v1(resources: &Tp4ResourcesV1) -> String {
    let handles = resources.handles;
    format!(
        "TP4 fixture instance created for topology {TP4_FIXTURE_TOPOLOGY_ID_V1} across \
         {} ranks, a {}-wide vocabulary read from its own program image, {} bytes of committed \
         history, and one rank-local paged KV request; it authenticated {} rank images and emits \
         its row from rank {TP4_FIXTURE_EMITTING_RANK_V1}",
        resources.ranks.len(),
        handles.geometry.vocabulary,
        handles.state_bytes,
        resources.image_digests.len()
    )
}

/// The digests this instance authenticated, as one auditable line.
fn authenticated_images_message_v1(resources: &Tp4ResourcesV1) -> String {
    let digests = resources
        .image_digests
        .iter()
        .enumerate()
        .map(|(rank, digest)| format!("rank {rank} {digest}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("TP4 fixture authenticated its rank images: {digests}")
}

fn claims_message_v1() -> String {
    format!(
        "TP4 fixture computes no TP4 arithmetic, qualifies no NVFP4 route, implements no \
         TP4 operation, and executes no authenticated payload; every row it emits comes from its \
         one bundle-private operation {TP4_FIXTURE_ROW_OPERATION_V1}"
    )
}

/// What the bundle custom operation did, named as the manifest names it.
fn custom_operation_message_v1(tokens: usize, decode: bool) -> String {
    let phase = if decode { "DecodeStep" } else { "PrefillChunk" };
    format!(
        "TP4 fixture ran bundle-private operation {TP4_FIXTURE_ROW_OPERATION_V1} for \
         one {phase} of {tokens} tokens"
    )
}

/// What a chunk the engine expects no row from did.
fn staged_chunk_message_v1(tokens: usize) -> String {
    format!(
        "TP4 fixture staged one prefill chunk of {tokens} tokens the engine expects no \
         row from, so it ran no operation and emitted none"
    )
}

/// Runs one scheduled batch and emits the row the private operation produced.
///
/// The transaction opens once everything that could refuse the batch has
/// refused, and it is rolled back before this call returns if anything after it
/// fails, so a failed execute leaves nothing a later commit or cancel would have
/// to name.
pub fn execute_v1(
    handle: BundleHandleV1,
    request: &[u8],
    sink: &mut ExecuteResultSinkV1,
) -> BundleStatusV1 {
    // Synthetic process tests control a batch after startup without a GPU.
    #[cfg(feature = "fake-host-services")]
    if let Some(directory) = std::env::var_os("TP4_FIXTURE_CONTROL_DIR") {
        let directory = std::path::PathBuf::from(directory);
        let _ = std::fs::write(directory.join("entered"), b"execute");
        if let Ok(delay) = std::fs::read_to_string(directory.join("delay-ms"))
            && let Ok(delay) = delay.trim().parse::<u64>()
        {
            std::thread::sleep(std::time::Duration::from_millis(delay.min(10_000)));
        }
        if directory.join("fail").exists() {
            return BundleStatusV1::InvalidState;
        }
    }
    match run_execution_v1(handle, request, sink) {
        Ok(()) => BundleStatusV1::Success,
        Err(status) => status,
    }
}

fn run_execution_v1(
    handle: BundleHandleV1,
    request: &[u8],
    sink: &mut ExecuteResultSinkV1,
) -> Result<(), BundleStatusV1> {
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            // A host service that calls back into this fixture finds the table
            // the outer call holds. Refusing here leaves that call's pending
            // execution alone, which panicking on the borrow would not.
            return Err(BundleStatusV1::InvalidState);
        };
        let Some(instance) = table.get_mut(handle) else {
            return Err(BundleStatusV1::StaleHandle);
        };
        let InstanceV1 {
            services,
            resources,
            operation,
        } = instance;
        if resources.pending.is_some() {
            log_v1(
                services,
                "TP4 fixture refused an execution while one is already pending",
            );
            return Err(BundleStatusV1::InvalidState);
        }
        let request = match BatchCodecV1::decode(request) {
            Ok(request) => request,
            Err(error) => {
                log_v1(
                    services,
                    &format!("TP4 fixture refused its batch request: {error}"),
                );
                return Err(BundleStatusV1::InvalidArgument);
            }
        };
        let step = scheduled_step_v1(services, &request, resources.handles.geometry)?;
        resources.current_request_id = Some(step.request_id);
        run_step_v1(services, resources, operation, &step, sink)
    })
}

/// One stretch of an append that lands inside one page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PageRunV1 {
    /// The physical page the host's table places this stretch's logical page at.
    physical_page: u64,
    /// Where inside that page the stretch starts, in bytes.
    byte_offset: u64,
    /// Where the stretch starts inside the append's own tokens.
    first_token: usize,
    /// How many tokens it carries.
    token_count: usize,
}

/// The one item this fixture runs, once the batch has been checked.
#[derive(Clone)]
struct ScheduledStepV1 {
    request_id: u64,
    decode: bool,
    /// Whether the engine scheduled this item to produce a logits row.
    ///
    /// A prompt longer than one chunk arrives as full chunks the engine expects
    /// nothing back from, then one tail it does. Both run the model side; only
    /// the tail emits.
    emits_row: bool,
    prompt_rows: bool,
    verify: bool,
    drafts: Vec<u32>,
    tokens: Vec<u32>,
    /// Where in the pages the host placed this append the tokens go.
    runs: Vec<PageRunV1>,
}

/// Splits one append into the stretches its page table places it in.
///
/// The wire page table is the complete ordered logical block table this append
/// commits to, so the token at sequence position `p` belongs to the page named
/// by entry `p / page_tokens`, at byte `(p % page_tokens) * bytes_per_token`
/// inside it. An append that starts part way through a page and runs past its
/// end is therefore two stretches, and this is where that split happens. A
/// logical page the table does not name is `None`: it is a placement this
/// instance does not own, not one to guess at.
fn page_runs_v1(
    previous_tokens: u64,
    token_count: usize,
    page_table: &[u64],
) -> Option<Vec<PageRunV1>> {
    let page_tokens = u64::from(TP4_FIXTURE_KV_PAGE_TOKENS_V1);
    let bytes_per_token = TP4_FIXTURE_KV_BYTES_PER_TOKEN_V1;
    let mut runs: Vec<PageRunV1> = Vec::new();
    for index in 0..token_count {
        let position = previous_tokens.checked_add(index as u64)?;
        let logical = usize::try_from(position / page_tokens).ok()?;
        let byte_offset = (position % page_tokens).checked_mul(bytes_per_token)?;
        let physical_page = *page_table.get(logical)?;
        let continues = runs.last().is_some_and(|run| {
            run.physical_page == physical_page
                && run.byte_offset + run.token_count as u64 * bytes_per_token == byte_offset
        });
        if continues {
            runs.last_mut()?.token_count += 1;
        } else {
            runs.push(PageRunV1 {
                physical_page,
                byte_offset,
                first_token: index,
                token_count: 1,
            });
        }
    }
    Some(runs)
}

/// Checks the one item this fixture runs and returns the step it schedules.
fn scheduled_step_v1(
    services: &HostServicesV1,
    request: &BatchRequestV1,
    geometry: ProgramGeometryV1,
) -> Result<ScheduledStepV1, BundleStatusV1> {
    let refuse = |reason: &str| {
        log_v1(
            services,
            &format!("TP4 fixture refused the scheduled batch: {reason}"),
        );
        Err(BundleStatusV1::InvalidArgument)
    };
    let ([item], [append]) = (request.items(), request.appends()) else {
        return refuse(&format!(
            "it carries {} items and {} appends, and one step runs exactly one of each",
            request.items().len(),
            request.appends().len()
        ));
    };
    if !item.is_verify() && item.phase_tokens() as usize != item.input_tokens().len() {
        return refuse("its declared phase length is not the number of tokens it carries");
    }
    // A decode is exactly one token by construction. A prefill is a chunk of at
    // most the declared PrefillChunk width: an engine issues full chunks and one
    // shorter tail, so a bound is what this fixture can actually be driven with.
    if item.is_verify() {
        if item.phase_tokens() < 2 || item.phase_tokens() > 4 || !item.expects_logits() {
            return refuse("verify needs two to four emitted rows");
        }
    } else if item.is_decode() {
        if item.input_tokens().len() != geometry.decode_tokens as usize {
            return refuse(&format!(
                "it carries {} tokens and one DecodeStep carries {}",
                item.input_tokens().len(),
                geometry.decode_tokens
            ));
        }
    } else if item.input_tokens().is_empty()
        || item.input_tokens().len() > geometry.prefill_tokens as usize
    {
        return refuse(&format!(
            "it carries {} tokens and one prefill chunk carries 1 to {}",
            item.input_tokens().len(),
            geometry.prefill_tokens
        ));
    }
    let drafts = if item.is_verify() && item.bundle_drafts() {
        if item.input_tokens().len() != 1 {
            return refuse("bundle-drafted verify needs one input token");
        }
        let miss = std::env::var_os("TP4_FIXTURE_SCRIPTED_MISS").is_some()
            && append.previous_tokens() % 2 == 0;
        (0..item.phase_tokens() - 1)
            .map(|index| {
                if miss && index == 0 {
                    crate::synthetic::TP4_FIXTURE_PEAK_TOKEN_ID_V1 - 1
                } else {
                    crate::synthetic::TP4_FIXTURE_PEAK_TOKEN_ID_V1
                }
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut tokens = item.input_tokens().to_vec();
    tokens.extend_from_slice(&drafts);
    if !request.media().is_empty() {
        if item.is_verify() {
            return refuse("media injection is a prefill operation");
        }
        let ([encode], [inject]) = (
            request.media().encodes.as_slice(),
            request.media().injects.as_slice(),
        ) else {
            return refuse("fixture copy encoder needs one encode and one injection");
        };
        let begin =
            usize::try_from(encode.payload_offset).map_err(|_| BundleStatusV1::InvalidArgument)?;
        let end = begin
            .checked_add(
                usize::try_from(encode.payload_bytes)
                    .map_err(|_| BundleStatusV1::InvalidArgument)?,
            )
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let bytes = request
            .media()
            .payload
            .get(begin..end)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if encode.encoder != 0
            || encode.output_rows != 1
            || encode.pages.len() != 1
            || encode.geometry != [1, 1, 1, 0]
            || bytes.len() != 6
            || inject.item_index != 0
            || inject.rows != 1
            || encode.pages[0].checked_mul(4) != Some(inject.arena_row)
        {
            return refuse("fixture copy encoder received an invalid geometry or placement");
        }
        let row = tokens
            .get_mut(inject.item_row as usize)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let value = f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16);
        if !value.is_finite() || value.fract() != 0.0 || !(0.0..256.0).contains(&value) {
            return refuse("fixture copy encoder payload is not a token identifier");
        }
        *row = crate::tokenizer::TP4_FIXTURE_BYTE_TOKEN_FIRST_ID_V1 + value as u32;
    }
    if tokens.len() > TP4_FIXTURE_MAXIMUM_TOKENS_V1 as usize {
        return refuse("it carries more tokens than one execution may");
    }
    if append.additional_tokens() != tokens.len() as u64 {
        return refuse("its prepared append does not carry the tokens the item does");
    }
    let Some(new_tokens) = append
        .previous_tokens()
        .checked_add(append.additional_tokens())
        .filter(|new_tokens| *new_tokens == append.new_tokens())
    else {
        return refuse("its prepared append does not end where its own token counts say it does");
    };
    // The page table is the whole sequence's, so it has exactly one entry per
    // page the sequence occupies once this append commits. A shorter one has
    // nothing to place the tokens past its last page in, and this fixture
    // refuses rather than folding them back into a page it was given.
    let pages = new_tokens.div_ceil(u64::from(TP4_FIXTURE_KV_PAGE_TOKENS_V1));
    if append.page_table().len() as u64 != pages {
        return refuse(&format!(
            "its page table names {} pages and a sequence of {new_tokens} tokens occupies {pages}",
            append.page_table().len()
        ));
    }
    let Some(runs) = page_runs_v1(append.previous_tokens(), tokens.len(), append.page_table())
    else {
        return refuse("its page table does not place every token this append carries");
    };
    Ok(ScheduledStepV1 {
        request_id: item.request_id(),
        decode: item.is_decode(),
        emits_row: item.expects_logits(),
        prompt_rows: item.prompt_rows(),
        verify: item.is_verify(),
        drafts,
        tokens,
        runs,
    })
}

/// Opens the append, runs the model side, and emits whatever it owes.
fn run_step_v1(
    services: &HostServicesV1,
    resources: &mut Tp4ResourcesV1,
    operation: &mut Tp4RowOperationV1,
    step: &ScheduledStepV1,
    sink: &mut ExecuteResultSinkV1,
) -> Result<(), BundleStatusV1> {
    let previous_history = read_history_v1(services, resources)?;
    let transaction = services
        .prepare_kv(
            resources.handles.kv_request,
            &kv_append_operation_v1(step.tokens.len() as u64),
        )
        .map_err(status_of_v1)?;
    match run_model_step_v1(services, resources, operation, step, previous_history, sink) {
        Ok(history) => {
            resources.pending = Some(PendingExecutionV1 {
                transaction,
                history,
                previous_history,
                request_id: step.request_id,
                verify_tokens: step.verify.then(|| step.tokens.clone()),
            });
            Ok(())
        }
        Err(status) => {
            // The append this call opened is the newest resource, so it is the
            // one the host takes back first, and rolling it back here is what
            // leaves nothing pending behind a refused execution.
            if let Err(refused) = services.cancel_kv(transaction) {
                log_v1(
                    services,
                    &format!(
                        "TP4 fixture could not roll back the append behind a refused \
                         execution: {refused}"
                    ),
                );
            }
            Err(status)
        }
    }
}

/// Reads the committed history out of the host's state allocation.
fn read_history_v1(
    services: &HostServicesV1,
    resources: &Tp4ResourcesV1,
) -> Result<TokenHistoryV1, BundleStatusV1> {
    let handles = resources.handles;
    let state_span = services
        .resolve_state_view(handles.state, TP4_FIXTURE_EMITTING_RANK_V1)
        .map_err(status_of_v1)?;
    let mut state_bytes = vec![0_u8; handles.state_bytes as usize];
    services
        .copy_device_to_host(handles.stream, &mut state_bytes, state_span)
        .map_err(status_of_v1)?;
    services
        .synchronize_stream(handles.stream)
        .map_err(status_of_v1)?;
    history_from_bytes_v1(&state_bytes).ok_or(BundleStatusV1::InvalidState)
}

/// Runs one scheduled step's model side and emits the row it owes, if any.
///
/// Both halves of a chunked prompt reach here. A full chunk the engine expects
/// nothing back from stages its tokens and its history exactly as a tail does,
/// and returns without a row rather than inventing one; the tail then computes
/// against the history that chunk committed.
fn run_model_step_v1(
    services: &HostServicesV1,
    resources: &mut Tp4ResourcesV1,
    operation: &mut Tp4RowOperationV1,
    step: &ScheduledStepV1,
    history: TokenHistoryV1,
    sink: &mut ExecuteResultSinkV1,
) -> Result<TokenHistoryV1, BundleStatusV1> {
    stage_kv_pages_v1(services, resources, step)?;
    if step.verify {
        if !step.drafts.is_empty() {
            sink.emit_draft_tokens(step.request_id, &step.drafts)
                .map_err(status_of_v1)?;
        }
        let mut final_history = history;
        for &token in &step.tokens {
            let mut one = step.clone();
            one.tokens = vec![token];
            final_history = emit_row_v1(services, resources, operation, &one, final_history, sink)?;
        }
        return Ok(final_history);
    }
    if step.prompt_rows {
        let mut final_history = history;
        for count in 1..=step.tokens.len() {
            let mut prefix = step.clone();
            prefix.tokens.truncate(count);
            final_history = emit_row_v1(services, resources, operation, &prefix, history, sink)?;
        }
        return Ok(final_history);
    }
    if !step.emits_row {
        log_v1(services, &staged_chunk_message_v1(step.tokens.len()));
        // The same fold the operation would have returned, over the same tokens.
        // Nothing else about the step differs, so the tail after it reads what a
        // row-emitting chunk would have left.
        return Ok(fold_tokens_v1(history, &step.tokens));
    }
    emit_row_v1(services, resources, operation, step, history, sink)
}

/// Writes each token of the append into the page the host placed it in.
///
/// Every rank holds its own copy of the same placement, so the whole append is
/// written and read back on all four. A page a rank's pool does not have is a
/// refusal, not a guess, and a rank that does not read back what it was given is
/// this fixture's own defect reported as one.
fn stage_kv_pages_v1(
    services: &HostServicesV1,
    resources: &Tp4ResourcesV1,
    step: &ScheduledStepV1,
) -> Result<(), BundleStatusV1> {
    let handles = resources.handles;
    // The same bytes go to every rank, so they are written out once here rather
    // than rebuilt four times.
    let staged = step
        .runs
        .iter()
        .map(|run| Ok((*run, run_bytes_v1(step, *run)?)))
        .collect::<Result<Vec<_>, BundleStatusV1>>()?;
    for (rank, rank_handles) in resources.ranks.iter().enumerate() {
        let rank = u32::try_from(rank).map_err(|_| BundleStatusV1::InvalidArgument)?;
        for (run, run_bytes) in &staged {
            let page_span = services
                .resolve_kv_view(
                    handles.page_request,
                    rank,
                    &kv_page_view_id_v1(run.physical_page),
                )
                .map_err(status_of_v1)?;
            let end = run
                .byte_offset
                .checked_add(run_bytes.len() as u64)
                .ok_or(BundleStatusV1::InvalidArgument)?;
            if page_span.length < end {
                log_v1(
                    services,
                    &format!(
                        "TP4 fixture refused an append whose rank {rank} page {} cannot \
                         hold the tokens placed in it",
                        run.physical_page
                    ),
                );
                return Err(BundleStatusV1::InvalidArgument);
            }
            let offset = page_span
                .offset
                .checked_add(run.byte_offset)
                .ok_or(BundleStatusV1::InvalidArgument)?;
            let page_write =
                DeviceBufferSpanV1::new(page_span.buffer, offset, run_bytes.len() as u64);
            services
                .copy_host_to_device(rank_handles.stream, page_write, run_bytes)
                .map_err(status_of_v1)?;
            let mut readback = vec![0_u8; run_bytes.len()];
            services
                .copy_device_to_host(rank_handles.stream, &mut readback, page_write)
                .map_err(status_of_v1)?;
            services
                .synchronize_stream(rank_handles.stream)
                .map_err(status_of_v1)?;
            if readback != *run_bytes {
                log_v1(
                    services,
                    &format!(
                        "TP4 fixture read back something other than the tokens it wrote \
                         into rank {rank}'s KV page {}",
                        run.physical_page
                    ),
                );
                return Err(BundleStatusV1::BundleError);
            }
        }
    }
    Ok(())
}

/// The tokens of one stretch, as the little-endian bytes its page holds.
fn run_bytes_v1(step: &ScheduledStepV1, run: PageRunV1) -> Result<Vec<u8>, BundleStatusV1> {
    let end = run
        .first_token
        .checked_add(run.token_count)
        .ok_or(BundleStatusV1::InvalidArgument)?;
    let tokens = step
        .tokens
        .get(run.first_token..end)
        .ok_or(BundleStatusV1::InvalidArgument)?;
    Ok(tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect())
}

/// Computes the row for one step and delivers it.
///
/// The destination row is zeroed first, so a row a caller receives is one this
/// execution wrote. The operation is lent the shape the manifest declares: one
/// PrefillChunk's capacity of identifiers, zero past the live ones, how many of
/// them are live, and the row.
fn emit_row_v1(
    services: &HostServicesV1,
    resources: &mut Tp4ResourcesV1,
    operation: &mut Tp4RowOperationV1,
    step: &ScheduledStepV1,
    history: TokenHistoryV1,
    sink: &mut ExecuteResultSinkV1,
) -> Result<TokenHistoryV1, BundleStatusV1> {
    let handles = resources.handles;
    let staged = StagedRowV1 {
        image_digests: resources.image_digests.clone(),
        history,
        tokens: step.tokens.clone(),
        vocabulary: handles.geometry.vocabulary,
        input_bytes: handles.input_bytes,
        token_count_bytes: handles.token_count_bytes,
        output_bytes: handles.row_bytes,
    };
    let input_span = DeviceBufferSpanV1::new(handles.input, 0, handles.input_bytes);
    let count_span = DeviceBufferSpanV1::new(handles.token_count, 0, handles.token_count_bytes);
    let logits_span = DeviceBufferSpanV1::new(handles.logits, 0, handles.row_bytes);
    services
        .copy_host_to_device(handles.stream, input_span, &staged.input_operand_bytes())
        .map_err(status_of_v1)?;
    services
        .copy_host_to_device(
            handles.stream,
            count_span,
            &staged.token_count_operand_bytes(),
        )
        .map_err(status_of_v1)?;
    services
        .zero_device_buffer(handles.stream, logits_span)
        .map_err(status_of_v1)?;

    operation.stage(staged);
    let outcome = services.run_custom_operation(
        handles.context,
        handles.stream,
        CustomCompletionV1::Synchronous,
        TP4_FIXTURE_ROW_OPERATION_ID_V1,
        &[
            CustomOperandV1::read(input_span),
            CustomOperandV1::read(count_span),
            CustomOperandV1::write(logits_span),
        ],
        &bind_tp4_row_v1(operation),
    );
    let staged = operation.take_result();
    let result = match (outcome, staged) {
        (Ok(()), Some(result)) => result,
        (outcome, _) => {
            log_v1(
                services,
                &format!(
                    "TP4 fixture private operation produced no result: {}",
                    describe_operation_outcome_v1(&outcome)
                ),
            );
            return Err(match outcome {
                Ok(()) => BundleStatusV1::BundleError,
                Err(error) => status_of_v1(error),
            });
        }
    };

    // The operation computed the row but addresses no memory itself, so the
    // write operand it declared is filled here, through the host.
    services
        .copy_host_to_device(handles.stream, logits_span, &result.logits)
        .map_err(status_of_v1)?;
    log_v1(
        services,
        &custom_operation_message_v1(step.tokens.len(), step.decode),
    );

    let mut row = vec![0_u8; handles.row_bytes as usize];
    services
        .copy_device_to_host(handles.stream, &mut row, logits_span)
        .map_err(status_of_v1)?;
    services
        .synchronize_stream(handles.stream)
        .map_err(status_of_v1)?;

    // Delivering the row is the last thing that can fail, so an execution that
    // refuses after this point does not exist: nothing is pending until here.
    sink.emit(step.request_id, &row).map_err(|error| {
        log_v1(
            services,
            &format!("TP4 fixture could not deliver its logits row: {error}"),
        );
        status_of_v1(error)
    })?;
    Ok(result.history)
}

fn describe_operation_outcome_v1(outcome: &Result<(), BundleAbiError>) -> String {
    match outcome {
        Ok(()) => "the host reported success and the operation staged nothing".to_owned(),
        Err(BundleAbiError::CustomOperation(code)) => format!(
            "it {} (code {code})",
            crate::operation::tp4_row_code_name_v1(*code)
        ),
        Err(error) => format!("the host refused the call: {error}"),
    }
}

/// Encodes one KV append operation for `prepare_kv`.
fn kv_append_operation_v1(additional_tokens: u64) -> Vec<u8> {
    let mut operation = KV_APPEND_OPERATION_V1.to_vec();
    operation.extend_from_slice(&additional_tokens.to_le_bytes());
    operation
}

/// Encodes one physical page index for `resolve_kv_view`.
fn kv_page_view_id_v1(physical_page: u64) -> Vec<u8> {
    physical_page.to_le_bytes().to_vec()
}

/// Commits the pending execution's history and then its KV append.
///
/// The history is published first because it is the half this instance can undo:
/// if the host then refuses the append, the previous history is written back and
/// the execution stays pending, so the same commit can be made again.
pub fn commit_v1(handle: BundleHandleV1) -> BundleStatusV1 {
    sweep_unreleased_v1();
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        let Some(pending) = instance.resources.pending.clone() else {
            log_v1(
                &instance.services,
                "TP4 fixture refused a commit with no execution pending",
            );
            return BundleStatusV1::InvalidState;
        };
        if let Err(status) =
            publish_history_v1(&instance.services, &instance.resources, pending.history)
        {
            // A refused publication may have landed in part, so the previous
            // history is written back before the caller is told the commit did
            // not happen. A cancel after this reads what the last commit left.
            if let Err(refused) = publish_history_v1(
                &instance.services,
                &instance.resources,
                pending.previous_history,
            ) {
                log_v1(
                    &instance.services,
                    &format!(
                        "TP4 fixture could not restore the history behind a refused \
                         publication: the host answered status {}",
                        refused.0
                    ),
                );
            }
            log_v1(
                &instance.services,
                "TP4 fixture could not publish its committed history, so the execution \
                 stays pending",
            );
            return status;
        }
        match instance.services.commit_kv(pending.transaction) {
            Ok(()) => {
                instance.resources.pending = None;
                BundleStatusV1::Success
            }
            Err(error) => {
                // The append did not commit, so the history this instance just
                // published is ahead of it and has to be put back.
                if let Err(refused) = publish_history_v1(
                    &instance.services,
                    &instance.resources,
                    pending.previous_history,
                ) {
                    log_v1(
                        &instance.services,
                        &format!(
                            "TP4 fixture published a history the host then refused to \
                             commit, and could not restore the previous one: status {}",
                            refused.0
                        ),
                    );
                }
                log_v1(
                    &instance.services,
                    "TP4 fixture could not commit its KV append, so the execution stays \
                     pending",
                );
                status_of_v1(error)
            }
        }
    })
}

/// Commits only the verified input prefix selected by the engine.
pub fn commit_accepted_v1(handle: BundleHandleV1, records: &[BundleKeptRowsV1]) -> BundleStatusV1 {
    let prepared = INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        let Some(pending) = instance.resources.pending.as_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let [record] = records else {
            return BundleStatusV1::InvalidArgument;
        };
        let Some(tokens) = pending.verify_tokens.as_ref() else {
            return BundleStatusV1::InvalidArgument;
        };
        let kept = record.kept_rows as usize;
        if record.request_id != pending.request_id || kept == 0 || kept > tokens.len() {
            return BundleStatusV1::InvalidArgument;
        }
        if kept < tokens.len() {
            if let Err(error) = instance.services.cancel_kv(pending.transaction) {
                return status_of_v1(error);
            }
            let transaction = match instance.services.prepare_kv(
                instance.resources.handles.kv_request,
                &kv_append_operation_v1(kept as u64),
            ) {
                Ok(transaction) => transaction,
                Err(error) => {
                    instance.resources.pending = None;
                    return status_of_v1(error);
                }
            };
            pending.transaction = transaction;
            pending.history = fold_tokens_v1(pending.previous_history, &tokens[..kept]);
            pending.verify_tokens = Some(tokens[..kept].to_vec());
        }
        BundleStatusV1::Success
    });
    if prepared != BundleStatusV1::Success {
        return prepared;
    }
    commit_v1(handle)
}

/// Writes one history into the emitting rank's view of the declared state.
///
/// The host allocates that state on every rank. This fixture owns one rank's
/// view of it and leaves the other three alone, which is what the emitting-rank
/// constant records.
fn publish_history_v1(
    services: &HostServicesV1,
    resources: &Tp4ResourcesV1,
    history: TokenHistoryV1,
) -> Result<(), BundleStatusV1> {
    let handles = resources.handles;
    let state_span = services
        .resolve_state_view(handles.state, TP4_FIXTURE_EMITTING_RANK_V1)
        .map_err(status_of_v1)?;
    services
        .copy_host_to_device(handles.stream, state_span, &history_to_bytes_v1(history))
        .map_err(status_of_v1)?;
    services
        .synchronize_stream(handles.stream)
        .map_err(status_of_v1)
}

/// Cancels the pending execution, leaving the committed history as it was.
///
/// Only a commit publishes the history, so a cancel has nothing to undo on that
/// half: it rolls the host's append back and the next execution reads exactly
/// what the last commit left.
pub fn cancel_v1(handle: BundleHandleV1) -> BundleStatusV1 {
    sweep_unreleased_v1();
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        match cancel_pending_v1(instance) {
            Some(Ok(())) => BundleStatusV1::Success,
            Some(Err(status)) => status,
            None => {
                log_v1(
                    &instance.services,
                    "TP4 fixture refused a cancel with no execution pending",
                );
                BundleStatusV1::InvalidState
            }
        }
    })
}

/// Rolls back whatever execution is open, if one is.
///
/// A refused rollback leaves the execution pending, so the transaction is still
/// named by something rather than lost, and the same cancel can be made again.
fn cancel_pending_v1(instance: &mut InstanceV1) -> Option<Result<(), BundleStatusV1>> {
    let pending = instance.resources.pending.clone()?;
    match instance.services.cancel_kv(pending.transaction) {
        Ok(()) => {
            instance.resources.pending = None;
            log_v1(
                &instance.services,
                &format!(
                    "TP4 fixture rolled back the execution of request {}, so its \
                     committed history and its KV pages are unchanged",
                    pending.request_id
                ),
            );
            Some(Ok(()))
        }
        Err(error) => {
            log_v1(
                &instance.services,
                "TP4 fixture could not roll back its KV append, so the execution stays \
                 pending",
            );
            Some(Err(status_of_v1(error)))
        }
    }
}

/// Releases one completed request's KV state and prepares the slot for reuse.
pub fn retire_request_v1(handle: BundleHandleV1, request_id: u64) -> BundleStatusV1 {
    sweep_unreleased_v1();
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        let resources = &mut instance.resources;
        if resources.current_request_id != Some(request_id) {
            return BundleStatusV1::Success;
        }
        if resources.pending.is_some() {
            return BundleStatusV1::InvalidState;
        }
        let Some(kv) = resources.held.iter().find_map(|held| match held {
            HeldV1::PhysicalKv(kv) => Some(*kv),
            _ => None,
        }) else {
            return BundleStatusV1::InvalidState;
        };
        if let Err(error) = instance
            .services
            .release_kv_request(resources.handles.kv_request)
        {
            return status_of_v1(error);
        }
        resources
            .held
            .retain(|held| !matches!(held, HeldV1::KvRequest(_)));
        let new_request = match instance
            .services
            .create_kv_request(kv, TP4_FIXTURE_KV_QUOTA_TOKENS_V1)
        {
            Ok(request) => request,
            Err(error) => return status_of_v1(error),
        };
        resources.handles.kv_request = new_request;
        resources.held.push(HeldV1::KvRequest(new_request));
        if let Err(status) =
            publish_history_v1(&instance.services, resources, state_seed_history_v1())
        {
            return status;
        }
        resources.current_request_id = None;
        BundleStatusV1::Success
    })
}

/// The bytes one prefix snapshot occupies at the start of its page: the
/// folded history, then the committed token count it covers.
const PREFIX_SNAPSHOT_BYTES_V1: u64 = 16;

/// Reports that one snapshot of this fixture's state fits one KV page.
pub fn describe_prefix_state_v1(handle: BundleHandleV1) -> BundlePrefixStateInfoV1 {
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundlePrefixStateInfoV1::failure(BundleStatusV1::InvalidState);
        };
        if table.get_mut(handle).is_none() {
            return BundlePrefixStateInfoV1::failure(BundleStatusV1::StaleHandle);
        }
        BundlePrefixStateInfoV1::success(1)
    })
}

/// Resolves the emitting rank's view of the snapshot page a record names.
fn prefix_snapshot_span_v1(
    services: &HostServicesV1,
    resources: &Tp4ResourcesV1,
    request: &PrefixStateRequestV1,
) -> Result<DeviceBufferSpanV1, BundleStatusV1> {
    let [page] = request.pages.as_slice() else {
        return Err(BundleStatusV1::InvalidArgument);
    };
    let span = services
        .resolve_kv_view(
            resources.handles.page_request,
            TP4_FIXTURE_EMITTING_RANK_V1,
            &kv_page_view_id_v1(*page),
        )
        .map_err(status_of_v1)?;
    if span.length < PREFIX_SNAPSHOT_BYTES_V1 {
        return Err(BundleStatusV1::InvalidState);
    }
    Ok(DeviceBufferSpanV1::new(
        span.buffer,
        span.offset,
        PREFIX_SNAPSHOT_BYTES_V1,
    ))
}

/// Writes the current request's committed history into its snapshot page.
///
/// The history is this fixture's only state the paged KV does not hold, so it
/// is the whole snapshot.
pub fn capture_prefix_state_v1(
    handle: BundleHandleV1,
    request: PrefixStateRequestV1,
) -> BundleStatusV1 {
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        let (services, resources) = (&instance.services, &instance.resources);
        if resources.pending.is_some()
            || resources.current_request_id != Some(request.request_id)
            || !request.token_ids.is_empty()
        {
            return BundleStatusV1::InvalidState;
        }
        let result = (|| {
            let span = prefix_snapshot_span_v1(services, resources, &request)?;
            let history = read_history_v1(services, resources)?;
            let mut bytes = history_to_bytes_v1(history);
            bytes.extend_from_slice(&request.tokens.to_le_bytes());
            let stream = resources.handles.stream;
            services
                .copy_host_to_device(stream, span, &bytes)
                .map_err(status_of_v1)?;
            services.synchronize_stream(stream).map_err(status_of_v1)
        })();
        result.err().unwrap_or(BundleStatusV1::Success)
    })
}

/// Starts a request from a snapshot, as if it had committed `token_ids`.
///
/// The snapshot must be the fold of exactly those tokens, so a host that names
/// the wrong page or the wrong prefix is refused rather than served.
pub fn restore_prefix_state_v1(
    handle: BundleHandleV1,
    request: PrefixStateRequestV1,
) -> BundleStatusV1 {
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        let InstanceV1 {
            services,
            resources,
            ..
        } = instance;
        if resources.pending.is_some()
            || resources.current_request_id.is_some()
            || request.token_ids.len() as u64 != request.tokens
            || request.tokens == 0
        {
            return BundleStatusV1::InvalidState;
        }
        let result = (|| {
            let span = prefix_snapshot_span_v1(services, resources, &request)?;
            let mut bytes = vec![0_u8; PREFIX_SNAPSHOT_BYTES_V1 as usize];
            let stream = resources.handles.stream;
            services
                .copy_device_to_host(stream, &mut bytes, span)
                .map_err(status_of_v1)?;
            services.synchronize_stream(stream).map_err(status_of_v1)?;
            let history = history_from_bytes_v1(&bytes[..8]).ok_or(BundleStatusV1::InvalidState)?;
            let tokens = u64::from_le_bytes(
                bytes[8..16]
                    .try_into()
                    .map_err(|_| BundleStatusV1::InvalidState)?,
            );
            if tokens != request.tokens
                || history != fold_tokens_v1(state_seed_history_v1(), &request.token_ids)
            {
                log_v1(
                    services,
                    "TP4 fixture refused a prefix snapshot that is not the fold of its prefix",
                );
                return Err(BundleStatusV1::InvalidArgument);
            }
            // The request's own KV accounting starts where the prefix ends.
            let transaction = services
                .prepare_kv(
                    resources.handles.kv_request,
                    &kv_append_operation_v1(request.tokens),
                )
                .map_err(status_of_v1)?;
            if let Err(error) = services.commit_kv(transaction) {
                let _ = services.cancel_kv(transaction);
                return Err(status_of_v1(error));
            }
            publish_history_v1(services, resources, history)
        })();
        match result {
            Ok(()) => {
                resources.current_request_id = Some(request.request_id);
                BundleStatusV1::Success
            }
            Err(status) => status,
        }
    })
}

/// Releases one instance and every host resource it still holds.
///
/// An instance with an execution still open is refused. Destroying one would
/// have to resolve a transaction its owner has not resolved, and reporting it as
/// finished either way is a claim this fixture will not make on the caller's
/// behalf. The refusal touches nothing: the execution stays pending, so a cancel
/// or a commit on the same handle still works and a destroy after that succeeds.
///
/// The instance is removed only once the host has taken everything back: a
/// refused release leaves it in the table holding exactly what it could not hand
/// over, so a later destroy of the same handle resumes at that point.
pub fn destroy_v1(handle: BundleHandleV1) -> BundleStatusV1 {
    sweep_unreleased_v1();
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        if let Some(pending) = instance.resources.pending.as_ref() {
            log_v1(
                &instance.services,
                &format!(
                    "TP4 fixture refused a destroy while the execution of request {} is \
                     still open, and rolled nothing back",
                    pending.request_id
                ),
            );
            return BundleStatusV1::InvalidState;
        }
        match release_all_v1(&instance.services, &mut instance.resources.held) {
            Ok(()) => {
                table.remove(handle);
                BundleStatusV1::Success
            }
            Err(status) => {
                log_v1(
                    &instance.services,
                    "TP4 fixture could not release its host resources, and keeps exactly \
                     what the host still owns",
                );
                status
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::{BundleHandleV1, BundleStatusV1};

    use crate::contract::{
        ProgramGeometryV1, TP4_FIXTURE_STATE_BYTES_V1, TP4_FIXTURE_VOCABULARY_BOUND_V1,
    };

    use crate::contract::{TP4_FIXTURE_KV_BYTES_PER_TOKEN_V1, TP4_FIXTURE_KV_PAGE_TOKENS_V1};

    use super::{
        KV_APPEND_OPERATION_V1, PageRunV1, cancel_v1, check_geometry_v1, commit_v1, destroy_v1,
        kv_append_operation_v1, kv_page_view_id_v1, page_runs_v1,
    };

    #[test]
    fn a_handle_this_fixture_never_issued_is_refused_by_every_lifecycle_call() {
        for handle in [
            BundleHandleV1::from_raw(0),
            BundleHandleV1::from_parts(1, 1),
            BundleHandleV1::from_parts(9, 9),
        ] {
            assert_eq!(commit_v1(handle), BundleStatusV1::StaleHandle);
            assert_eq!(cancel_v1(handle), BundleStatusV1::StaleHandle);
            assert_eq!(destroy_v1(handle), BundleStatusV1::StaleHandle);
        }
    }

    #[test]
    fn a_program_image_that_disagrees_with_the_declaration_is_refused() {
        let declared = ProgramGeometryV1::declared();

        assert_eq!(
            check_geometry_v1(declared, TP4_FIXTURE_STATE_BYTES_V1),
            Ok(())
        );
        for wrong in [
            ProgramGeometryV1 {
                state_bytes: declared.state_bytes / 2,
                ..declared
            },
            ProgramGeometryV1 {
                vocabulary: 0,
                ..declared
            },
            ProgramGeometryV1 {
                vocabulary: TP4_FIXTURE_VOCABULARY_BOUND_V1 + 1,
                ..declared
            },
            ProgramGeometryV1 {
                prefill_tokens: declared.prefill_tokens + 1,
                ..declared
            },
            ProgramGeometryV1 {
                decode_tokens: 0,
                ..declared
            },
        ] {
            assert_eq!(
                check_geometry_v1(wrong, TP4_FIXTURE_STATE_BYTES_V1),
                Err(BundleStatusV1::Unsupported),
                "{wrong:?} must not describe this fixture"
            );
        }
    }

    /// The two encodings the host declared, written the way the host reads them.
    #[test]
    fn the_kv_operation_and_page_identifier_carry_what_the_host_declared() {
        let operation = kv_append_operation_v1(32);

        assert!(operation.starts_with(KV_APPEND_OPERATION_V1));
        assert_eq!(
            operation[KV_APPEND_OPERATION_V1.len()..],
            32_u64.to_le_bytes()
        );
        assert_eq!(kv_page_view_id_v1(3), 3_u64.to_le_bytes());
        assert_ne!(
            kv_page_view_id_v1(0),
            Vec::<u8>::new(),
            "a page identifier is never the empty one that selects the whole pool"
        );
    }

    /// Where an append lands is its own position in the sequence read against
    /// the whole page table, not the first entry of it.
    #[test]
    fn every_token_is_placed_by_its_logical_page_and_its_offset_inside_it() {
        let page = u64::from(TP4_FIXTURE_KV_PAGE_TOKENS_V1);
        let table = [7_u64, 5, 9];

        // A whole page from its start is one stretch at offset zero.
        assert_eq!(
            page_runs_v1(0, page as usize, &table),
            Some(vec![PageRunV1 {
                physical_page: 7,
                byte_offset: 0,
                first_token: 0,
                token_count: page as usize,
            }])
        );
        // A decode at the first position of the second page reaches the second
        // entry of the table, and starts at that page's own offset zero.
        assert_eq!(
            page_runs_v1(page, 1, &table),
            Some(vec![PageRunV1 {
                physical_page: 5,
                byte_offset: 0,
                first_token: 0,
                token_count: 1,
            }])
        );
        // A decode part way through a page starts part way through it.
        assert_eq!(
            page_runs_v1(page + 3, 1, &table),
            Some(vec![PageRunV1 {
                physical_page: 5,
                byte_offset: 3 * TP4_FIXTURE_KV_BYTES_PER_TOKEN_V1,
                first_token: 0,
                token_count: 1,
            }])
        );
        // An append that runs past a page boundary is two stretches, split at
        // the boundary, each in the page its own tokens belong to.
        assert_eq!(
            page_runs_v1(page - 2, 4, &table),
            Some(vec![
                PageRunV1 {
                    physical_page: 7,
                    byte_offset: (page - 2) * TP4_FIXTURE_KV_BYTES_PER_TOKEN_V1,
                    first_token: 0,
                    token_count: 2,
                },
                PageRunV1 {
                    physical_page: 5,
                    byte_offset: 0,
                    first_token: 2,
                    token_count: 2,
                },
            ])
        );
    }

    /// A page table that does not reach the tokens it has to place is refused
    /// rather than folded back onto a page it does name.
    #[test]
    fn a_page_table_that_does_not_reach_every_token_places_none_of_them() {
        let page = u64::from(TP4_FIXTURE_KV_PAGE_TOKENS_V1);

        assert_eq!(page_runs_v1(page - 2, 4, &[7]), None);
        assert_eq!(page_runs_v1(page, 1, &[7]), None);
        assert_eq!(page_runs_v1(0, 1, &[]), None);
        assert_eq!(page_runs_v1(u64::MAX, 2, &[7]), None);
    }
}
