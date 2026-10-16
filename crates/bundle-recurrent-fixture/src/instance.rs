//! The instance lifecycle behind this fixture's one native entry point.
//!
//! Instances live in a thread-local table because a host loads one bundle at
//! process start on one thread and keeps it there. Each handle carries a slot
//! and a generation this table issued, so a handle from a destroyed instance, or
//! one this fixture never issued, is refused before it reaches a resource.
//!
//! The transaction is the recurrent state, not a KV append. An execution reads
//! the committed lanes out of the host's state allocation, folds the batch's
//! tokens into them, and stages the result. `commit` is the only thing that
//! writes the persistent state, so `cancel` needs to undo nothing: the lanes a
//! cancelled execution computed are dropped and the next execution reads exactly
//! what the last commit left.

use std::cell::RefCell;
use std::collections::BTreeMap;

use runtime_bundle::{
    GENERIC_COPY_F32_ALIGNMENT_V1, GENERIC_COPY_F32_OPERATION_V1, GENERIC_COPY_F32_PRIMITIVE_V1,
    generic_copy_f32_bytes_v1,
};
use runtime_bundle_abi::{
    BundleAbiError, BundleCompiledImageHandleV1, BundleContextHandleV1, BundleCreateResultV1,
    BundleDeviceBufferHandleV1, BundleHandleV1, BundleMemoryPoolHandleV1, BundleRankArenaHandleV1,
    BundleStateHandleV1, BundleStatusV1, BundleStreamHandleV1, CustomCompletionV1, CustomOperandV1,
    DeviceBufferSpanV1, ExecuteResultSinkV1, HostServicesV1, StateRequestV1,
};

use crate::configuration::BundleConfigurationV1;
use crate::contract::{
    ProgramGeometryV1, RECURRENT_BUFFER_ALIGNMENT_V1, RECURRENT_GEOMETRY_BYTES_V1,
    RECURRENT_GEOMETRY_VIEW_V1, RECURRENT_KV_FP8_LAYOUT_ID_V1, RECURRENT_KV_FP8_PAGE_BYTES_V1,
    RECURRENT_KV_LAYOUT_ID_V1, RECURRENT_KV_PAGE_BYTES_V1, RECURRENT_PROGRAM_ROLE_V1,
    RECURRENT_RANK_COUNT_V1, RECURRENT_STAGE_ID_V1, RECURRENT_STATE_SEED_VIEW_V1,
    RECURRENT_STEP_OPERATION_V1, RECURRENT_STREAM_KIND_V1, RECURRENT_TOKEN_BOUND_V1,
    RECURRENT_TOPOLOGY_ID_V1, RECURRENT_VOCABULARY_BOUND_V1,
};
use crate::operation::{
    RECURRENT_STEP_OPERATION_ID_V1, RecurrentStepOperationV1, StagedStepV1, bind_recurrent_step_v1,
};
use crate::recurrent::{
    RECURRENT_LANES_V1, RecurrentLanesV1, fold_tokens_v1, lanes_from_bytes_v1, lanes_to_bytes_v1,
    logits_row_v1,
};
use runtime_bundle_sdk::{BatchCodecV1, BatchRequestV1};

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

/// The handles one created instance keeps for its whole life.
#[derive(Clone, Copy, Debug)]
struct AcquiredHandlesV1 {
    context: BundleContextHandleV1,
    stream: BundleStreamHandleV1,
    input: BundleDeviceBufferHandleV1,
    /// Where the custom operation writes the row it computed.
    ///
    /// This is not the buffer a caller's row is read from. The declared generic
    /// copy is what moves the row from here into `logits`, so a host that
    /// cannot perform that copy leaves this instance with nothing to emit.
    step_output: BundleDeviceBufferHandleV1,
    logits: BundleDeviceBufferHandleV1,
    state: BundleStateHandleV1,
    geometry: ProgramGeometryV1,
    state_bytes: u64,
    row_bytes: u64,
    /// Page bytes of the declared KV layout the host selected; prefix
    /// snapshots fill whole pages of it.
    kv_page_bytes: u64,
}

/// The lanes one execution computed, waiting for a commit or a cancel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingStepV1 {
    lanes: RecurrentLanesV1,
    request_id: u64,
}

/// Everything one instance holds from the host, and its open transaction.
struct RecurrentResourcesV1 {
    handles: AcquiredHandlesV1,
    held: Vec<HeldV1>,
    pending: Option<PendingStepV1>,
    requests: BTreeMap<u64, RequestStateV1>,
    pending_requests: Option<Vec<RequestCheckpointV1>>,
}

#[derive(Clone, Debug)]
struct RequestStateV1 {
    slot: Option<u64>,
    tokens: u64,
    lanes: RecurrentLanesV1,
}

struct RequestCheckpointV1 {
    request_id: u64,
    slot: u64,
    previous_tokens: u64,
    input_count: usize,
    verify: bool,
    rows: Vec<RecurrentLanesV1>,
}

/// Why an acquisition failed, and what the host would not take back.
struct AcquireFailureV1 {
    status: BundleStatusV1,
    unreleased: Vec<HeldV1>,
}

/// One created instance: its service table, its resources, and its operation.
struct InstanceV1 {
    services: HostServicesV1,
    resources: RecurrentResourcesV1,
    operation: RecurrentStepOperationV1,
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
                    "recurrent fixture returned resources the host had previously refused",
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
                &format!("recurrent fixture refused its configuration: {error}"),
            );
            return Err(BundleStatusV1::InvalidArgument);
        }
    };
    // Nothing is acquired before the declaration is checked, so a host this
    // fixture does not serve costs it no resource at all.
    if let Err(refusal) = check_configuration_v1(&configuration) {
        log_v1(
            &services,
            &format!("recurrent fixture refused this host: {}", refusal.reason),
        );
        return Err(refusal.status);
    }
    let resources = match acquire_v1(&services, &configuration) {
        Ok(resources) => resources,
        Err(failure) => {
            log_v1(
                &services,
                "recurrent fixture could not acquire what its declaration asks for",
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
    log_v1(&services, &created_message_v1(&resources.handles));
    log_v1(&services, &kv_message_v1(&configuration));
    log_v1(&services, &operations_message_v1());
    let instance = Box::new(InstanceV1 {
        services,
        resources,
        operation: RecurrentStepOperationV1::new(),
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
                "recurrent fixture exhausted its instance table and is releasing what it acquired",
            );
            match release_all_v1(&instance.services, &mut instance.resources.held) {
                Ok(()) => Err(BundleStatusV1::ResourceExhausted),
                Err(error) => {
                    log_v1(
                        &instance.services,
                        &format!(
                            "recurrent fixture could not release what it acquired: the host \
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
        "recurrent fixture is holding the resources the host refused to take back, and will try \
         again on its next lifecycle call",
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
    if configuration.topology_id != RECURRENT_TOPOLOGY_ID_V1.as_bytes() {
        return refuse(
            BundleStatusV1::Unsupported,
            format!(
                "this fixture serves topology {RECURRENT_TOPOLOGY_ID_V1} and the host configured \
                 '{}'",
                String::from_utf8_lossy(&configuration.topology_id)
            ),
        );
    }
    if configuration.rank_count() != RECURRENT_RANK_COUNT_V1 as usize {
        return refuse(
            BundleStatusV1::Unsupported,
            format!(
                "this fixture serves exactly {RECURRENT_RANK_COUNT_V1} rank and the host offered {}",
                configuration.rank_count()
            ),
        );
    }
    if !configuration.declares_component_role(RECURRENT_PROGRAM_ROLE_V1.as_bytes()) {
        return refuse(
            BundleStatusV1::NotFound,
            format!("the package declares no component for role {RECURRENT_PROGRAM_ROLE_V1}"),
        );
    }
    if configuration.state.byte_length == 0 {
        return refuse(
            BundleStatusV1::Unsupported,
            "the package declares no persistent state, which is the whole of this fixture's model"
                .to_owned(),
        );
    }
    Ok(())
}

/// Acquires everything one instance needs, unwinding whatever it took first.
fn acquire_v1(
    services: &HostServicesV1,
    configuration: &BundleConfigurationV1,
) -> Result<RecurrentResourcesV1, AcquireFailureV1> {
    let mut held = Vec::new();
    match acquire_handles_v1(services, configuration, &mut held) {
        Ok(handles) => Ok(RecurrentResourcesV1 {
            handles,
            held,
            pending: None,
            requests: BTreeMap::new(),
            pending_requests: None,
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

/// The whole acquisition, in the order the host must see it released back.
fn acquire_handles_v1(
    services: &HostServicesV1,
    configuration: &BundleConfigurationV1,
    held: &mut Vec<HeldV1>,
) -> Result<AcquiredHandlesV1, BundleStatusV1> {
    let ordinal = *configuration
        .device_ordinals
        .first()
        .ok_or(BundleStatusV1::InvalidArgument)?;
    let context = services.select_context(0, ordinal).map_err(status_of_v1)?;
    held.push(HeldV1::Context(context));
    let pool = services.select_memory_pool(context).map_err(status_of_v1)?;
    held.push(HeldV1::MemoryPool(pool));
    // One stage, so one stream: every transfer this instance makes is ordered
    // against the same one, and the host hands back the same stream each time.
    let stream = services
        .select_stage_stream(context, 0, RECURRENT_STAGE_ID_V1, RECURRENT_STREAM_KIND_V1)
        .map_err(status_of_v1)?;
    held.push(HeldV1::Stream(stream));

    let image = services
        .open_compiled_image(RECURRENT_PROGRAM_ROLE_V1.as_bytes())
        .map_err(status_of_v1)?;
    held.push(HeldV1::CompiledImage(image));
    let arena = services
        .upload_rank_arena(context, pool, stream, image, 0)
        .map_err(status_of_v1)?;
    held.push(HeldV1::RankArena(arena));

    // The width of a row and the length of a prompt are the package's to
    // declare, so they are read out of the image the host just uploaded rather
    // than compiled into this module.
    let geometry_span = services
        .resolve_compiled_image_view(image, arena, 0, RECURRENT_GEOMETRY_VIEW_V1)
        .map_err(status_of_v1)?;
    if geometry_span.buffer.length != RECURRENT_GEOMETRY_BYTES_V1 {
        return Err(BundleStatusV1::InvalidArgument);
    }
    let mut geometry_bytes = vec![0_u8; RECURRENT_GEOMETRY_BYTES_V1 as usize];
    services
        .copy_device_to_host(stream, &mut geometry_bytes, geometry_span.buffer)
        .map_err(status_of_v1)?;
    services.synchronize_stream(stream).map_err(status_of_v1)?;
    let geometry =
        ProgramGeometryV1::from_bytes(&geometry_bytes).ok_or(BundleStatusV1::InvalidArgument)?;
    check_geometry_v1(geometry, configuration.state.byte_length)?;

    let state_bytes = configuration.state.byte_length;
    let input_bytes = u64::from(geometry.maximum_tokens) * 4;
    let row_bytes = generic_copy_f32_bytes_v1(geometry.vocabulary);
    let input = services
        .allocate_device_buffer(pool, stream, input_bytes, RECURRENT_BUFFER_ALIGNMENT_V1)
        .map_err(status_of_v1)?;
    held.push(HeldV1::DeviceBuffer(input));
    // Two buffers of one width: what the custom operation writes, and what the
    // declared generic copy moves it into. They are separate allocations
    // because the copy's operands are declared disjoint, and because a row read
    // out of the destination is a row that copy actually produced.
    let step_output = services
        .allocate_device_buffer(pool, stream, row_bytes, GENERIC_COPY_F32_ALIGNMENT_V1)
        .map_err(status_of_v1)?;
    held.push(HeldV1::DeviceBuffer(step_output));
    let logits = services
        .allocate_device_buffer(pool, stream, row_bytes, GENERIC_COPY_F32_ALIGNMENT_V1)
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

    // The initial lanes come out of the package, not out of this module: the
    // seed is a view of the same authenticated image, copied across on the
    // device without ever passing through host memory.
    let seed_span = services
        .resolve_compiled_image_view(image, arena, 0, RECURRENT_STATE_SEED_VIEW_V1)
        .map_err(status_of_v1)?;
    let state_span = services
        .resolve_state_view(state, 0)
        .map_err(status_of_v1)?;
    if seed_span.buffer.length != state_bytes || state_span.length != state_bytes {
        return Err(BundleStatusV1::InvalidArgument);
    }
    services
        .copy_device_to_device(stream, state_span, seed_span.buffer)
        .map_err(status_of_v1)?;
    services.synchronize_stream(stream).map_err(status_of_v1)?;

    // The host declares one of this module's two layouts; any other is one
    // whose pages the snapshots would not fit.
    let kv_page_bytes = [
        (RECURRENT_KV_LAYOUT_ID_V1, RECURRENT_KV_PAGE_BYTES_V1),
        (
            RECURRENT_KV_FP8_LAYOUT_ID_V1,
            RECURRENT_KV_FP8_PAGE_BYTES_V1,
        ),
    ]
    .into_iter()
    .find(|(id, _)| id.as_bytes() == configuration.kv.layout_id.as_slice())
    .map(|(_, bytes)| bytes)
    .filter(|&bytes| bytes == configuration.kv.page_bytes)
    .ok_or(BundleStatusV1::Unsupported)?;
    // Older direct ABI tests do not configure a serving pool. Serving callers
    // require it through the manifest; only an absent service is optional here.
    match services.acquire_serving_kv(&configuration.kv.layout_id) {
        Ok(_) | Err(BundleAbiError::Status(BundleStatusV1::Unsupported)) => {}
        Err(error) => return Err(status_of_v1(error)),
    }

    Ok(AcquiredHandlesV1 {
        context,
        stream,
        input,
        step_output,
        logits,
        state,
        geometry,
        state_bytes,
        row_bytes,
        kv_page_bytes,
    })
}

/// Refuses a program image that does not describe the state the manifest does.
fn check_geometry_v1(geometry: ProgramGeometryV1, state_bytes: u64) -> Result<(), BundleStatusV1> {
    if u64::from(geometry.state_bytes) != state_bytes
        || geometry.lanes as usize != RECURRENT_LANES_V1
        || u64::from(geometry.lanes) * 4 != state_bytes
    {
        return Err(BundleStatusV1::Unsupported);
    }
    if geometry.vocabulary == 0 || geometry.vocabulary > RECURRENT_VOCABULARY_BOUND_V1 {
        return Err(BundleStatusV1::Unsupported);
    }
    if geometry.maximum_tokens == 0 || geometry.maximum_tokens > RECURRENT_TOKEN_BOUND_V1 {
        return Err(BundleStatusV1::Unsupported);
    }
    Ok(())
}

fn created_message_v1(handles: &AcquiredHandlesV1) -> String {
    format!(
        "recurrent fixture instance created for topology {RECURRENT_TOPOLOGY_ID_V1} with \
         {RECURRENT_RANK_COUNT_V1} rank, {} bytes of persistent state across {} lanes, and a \
         {}-wide vocabulary read from its own program image",
        handles.state_bytes, handles.geometry.lanes, handles.geometry.vocabulary
    )
}

fn kv_message_v1(configuration: &BundleConfigurationV1) -> String {
    format!(
        "recurrent fixture carries persistent state instead of a KV cache, so the declared page \
         layout '{}' is never allocated and no physical KV pool exists; the canonical manifest \
         declares it as {RECURRENT_KV_LAYOUT_ID_V1}",
        String::from_utf8_lossy(&configuration.kv.layout_id)
    )
}

fn operations_message_v1() -> String {
    format!(
        "recurrent fixture requires one bundle custom operation {RECURRENT_STEP_OPERATION_V1}, \
         which only this package implements, and one host generic operation \
         {GENERIC_COPY_F32_OPERATION_V1} whose primitive is \
         {GENERIC_COPY_F32_PRIMITIVE_V1}; every execution performs both, so a host with no \
         candidate for that primitive emits no row"
    )
}

/// What the bundle custom operation did, named as the manifest names it.
fn custom_operation_message_v1(tokens: usize) -> String {
    format!(
        "recurrent fixture ran bundle custom operation {RECURRENT_STEP_OPERATION_V1} over {tokens} \
         tokens into its step-output operand"
    )
}

/// What the declared generic operation did, named as the manifest names it.
fn generic_operation_message_v1(row_bytes: u64) -> String {
    format!(
        "recurrent fixture ran host generic operation {GENERIC_COPY_F32_OPERATION_V1} through \
         primitive {GENERIC_COPY_F32_PRIMITIVE_V1}, copying {row_bytes} bytes from its \
         step-output operand into the logits it emits"
    )
}

/// What a host that would not perform that operation left this step with.
fn generic_copy_refusal_message_v1(error: &BundleAbiError) -> String {
    format!(
        "recurrent fixture could not run host generic operation \
         {GENERIC_COPY_F32_OPERATION_V1} through primitive \
         {GENERIC_COPY_F32_PRIMITIVE_V1}, so it emits no row: {error}"
    )
}

/// Runs one scheduled batch and emits the row the recurrent step produced.
///
/// The transaction opens last. Everything that can refuse this execution refuses
/// before the pending step is recorded, so a failed execute leaves nothing a
/// later commit or cancel would have to name.
pub fn execute_v1(
    handle: BundleHandleV1,
    request: &[u8],
    sink: &mut ExecuteResultSinkV1,
) -> BundleStatusV1 {
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
        if resources.pending.is_some() || resources.pending_requests.is_some() {
            log_v1(
                services,
                "recurrent fixture refused an execution while one is already pending",
            );
            return Err(BundleStatusV1::InvalidState);
        }
        let request = match BatchCodecV1::decode(request) {
            Ok(request) => request,
            Err(error) => {
                log_v1(
                    services,
                    &format!("recurrent fixture refused its batch request: {error}"),
                );
                return Err(BundleStatusV1::InvalidArgument);
            }
        };
        if request.state_slots().is_some() {
            return run_request_batch_v1(services, resources, operation, &request, sink);
        }
        let step = scheduled_step_v1(services, &request, resources.handles.geometry)?;
        if step.prompt_rows {
            for count in 1..=step.tokens.len() {
                if let Err(status) = run_step_v1(
                    services,
                    resources,
                    operation,
                    &ScheduledStepV1 {
                        request_id: step.request_id,
                        tokens: step.tokens[..count].to_vec(),
                        prompt_rows: true,
                        initial: None,
                        emit: true,
                    },
                    sink,
                ) {
                    resources.pending = None;
                    return Err(status);
                }
            }
            Ok(())
        } else {
            run_step_v1(services, resources, operation, &step, sink)
        }
    })
}

/// The one item this fixture runs, once the batch has been checked.
struct ScheduledStepV1 {
    request_id: u64,
    tokens: Vec<u32>,
    prompt_rows: bool,
    initial: Option<RecurrentLanesV1>,
    emit: bool,
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
            &format!("recurrent fixture refused the scheduled batch: {reason}"),
        );
        Err(BundleStatusV1::InvalidArgument)
    };
    let [item] = request.items() else {
        return refuse(&format!(
            "it carries {} items and one recurrent step runs exactly one",
            request.items().len()
        ));
    };
    // The host prepares one append per item even where nothing appends to it,
    // and a request that does not is one this fixture did not read whole.
    if request.appends().len() != request.items().len() {
        return refuse("it carries a different number of appends and items");
    }
    if !item.expects_logits() {
        return refuse("it expects no logits, and a step with no row to emit is not one to run");
    }
    if item.input_tokens().is_empty() {
        return refuse("it carries no input tokens");
    }
    if item.input_tokens().len() > geometry.maximum_tokens as usize {
        return refuse(&format!(
            "it carries {} tokens and the program image declares at most {}",
            item.input_tokens().len(),
            geometry.maximum_tokens
        ));
    }
    if item.phase_tokens() as usize != item.input_tokens().len() {
        return refuse("its declared phase length is not the number of tokens it carries");
    }
    // A recurrent step is the same fold either way, so a decode is served as
    // well as a prefill; a decode step is one token by construction.
    if item.is_decode() && item.input_tokens().len() != 1 {
        return refuse("a decode step carries exactly one token");
    }
    Ok(ScheduledStepV1 {
        request_id: item.request_id(),
        tokens: item.input_tokens().to_vec(),
        prompt_rows: item.prompt_rows(),
        initial: None,
        emit: true,
    })
}

/// Reads the committed lanes, runs both declared operations, and emits the row.
///
/// The order is the contract. The destination is zeroed first, so a row read
/// out of it is a row something wrote this step. The bundle custom operation
/// then computes into the intermediate operand, and the declared generic copy
/// moves that operand into the destination. Only the destination is read, so a
/// host whose catalog cannot perform the copy emits nothing, and nothing is
/// pending until the row has been delivered.
fn run_step_v1(
    services: &HostServicesV1,
    resources: &mut RecurrentResourcesV1,
    operation: &mut RecurrentStepOperationV1,
    step: &ScheduledStepV1,
    sink: &mut ExecuteResultSinkV1,
) -> Result<(), BundleStatusV1> {
    let tokens = step.tokens.as_slice();
    let handles = resources.handles;
    let state_span = services
        .resolve_state_view(handles.state, 0)
        .map_err(status_of_v1)?;
    let mut state_bytes = vec![0_u8; handles.state_bytes as usize];
    services
        .copy_device_to_host(handles.stream, &mut state_bytes, state_span)
        .map_err(status_of_v1)?;
    // The lanes are read here, so the transfer that fetched them has to have
    // landed here rather than at the end of the step.
    services
        .synchronize_stream(handles.stream)
        .map_err(status_of_v1)?;
    let lanes = step
        .initial
        .unwrap_or(lanes_from_bytes_v1(&state_bytes).ok_or(BundleStatusV1::InvalidState)?);

    // The tokens go into the operand the operation declares, so the operation
    // is lent the same bytes a native one would read.
    let token_bytes = tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect::<Vec<u8>>();
    let input_span = DeviceBufferSpanV1::new(handles.input, 0, token_bytes.len() as u64);
    let step_output_span = DeviceBufferSpanV1::new(handles.step_output, 0, handles.row_bytes);
    let logits_span = DeviceBufferSpanV1::new(handles.logits, 0, handles.row_bytes);
    services
        .copy_host_to_device(handles.stream, input_span, &token_bytes)
        .map_err(status_of_v1)?;
    // Nothing but the declared copy writes the destination, and it starts this
    // step empty, so a row that reaches a caller cannot be one a skipped copy
    // left behind.
    services
        .zero_device_buffer(handles.stream, logits_span)
        .map_err(status_of_v1)?;

    operation.stage(StagedStepV1 {
        lanes,
        tokens: tokens.to_vec(),
        vocabulary: handles.geometry.vocabulary,
        input_bytes: input_span.length,
        output_bytes: step_output_span.length,
    });
    let outcome = services.run_custom_operation(
        handles.context,
        handles.stream,
        CustomCompletionV1::Synchronous,
        RECURRENT_STEP_OPERATION_ID_V1,
        &[
            CustomOperandV1::read(input_span),
            CustomOperandV1::write(step_output_span),
        ],
        &bind_recurrent_step_v1(operation),
    );
    let staged = operation.take_result();
    let result = match (outcome, staged) {
        (Ok(()), Some(result)) => result,
        (outcome, _) => {
            log_v1(
                services,
                &format!(
                    "recurrent fixture custom operation produced no result: {}",
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
        .copy_host_to_device(handles.stream, step_output_span, &result.logits)
        .map_err(status_of_v1)?;
    log_v1(services, &custom_operation_message_v1(tokens.len()));

    // The second declared operation, which this package does not implement: an
    // exact F32 copy of the step's output into the logits a caller receives.
    // The host performs it through its own device-to-device copy service, and a
    // host that refuses leaves this execution with no row and nothing pending.
    services
        .copy_device_to_device(handles.stream, logits_span, step_output_span)
        .map_err(|error| {
            log_v1(services, &generic_copy_refusal_message_v1(&error));
            status_of_v1(error)
        })?;
    log_v1(services, &generic_operation_message_v1(handles.row_bytes));

    let mut row = vec![0_u8; handles.row_bytes as usize];
    services
        .copy_device_to_host(handles.stream, &mut row, logits_span)
        .map_err(status_of_v1)?;
    services
        .synchronize_stream(handles.stream)
        .map_err(status_of_v1)?;

    // Delivering the row is the last thing that can fail, so an execution that
    // refuses after this point does not exist: nothing is pending until here.
    if step.emit {
        sink.emit(step.request_id, &row).map_err(|error| {
            log_v1(
                services,
                &format!("recurrent fixture could not deliver its logits row: {error}"),
            );
            status_of_v1(error)
        })?;
    }
    resources.pending = Some(PendingStepV1 {
        lanes: result.lanes,
        request_id: step.request_id,
    });
    Ok(())
}

fn describe_operation_outcome_v1(outcome: &Result<(), BundleAbiError>) -> String {
    match outcome {
        Ok(()) => "the host reported success and the operation staged nothing".to_owned(),
        Err(BundleAbiError::CustomOperation(code)) => format!(
            "it {} (code {code})",
            crate::operation::recurrent_step_code_name_v1(*code)
        ),
        Err(error) => format!("the host refused the call: {error}"),
    }
}

/// Commits the pending step by publishing its lanes as the persistent state.
pub fn commit_v1(handle: BundleHandleV1) -> BundleStatusV1 {
    sweep_unreleased_v1();
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        if instance.resources.pending_requests.is_some() {
            return commit_requests_v1(&mut instance.resources, None);
        }
        let Some(pending) = instance.resources.pending.take() else {
            log_v1(
                &instance.services,
                "recurrent fixture refused a commit with no execution pending",
            );
            return BundleStatusV1::InvalidState;
        };
        match publish_state_v1(&instance.services, &instance.resources, &pending) {
            Ok(()) => BundleStatusV1::Success,
            Err(status) => {
                // The state was not published, so the step is still pending and
                // the same commit can be made again.
                instance.resources.pending = Some(pending);
                log_v1(
                    &instance.services,
                    "recurrent fixture could not publish its recurrent state, so the step stays \
                     pending",
                );
                status
            }
        }
    })
}

fn publish_state_v1(
    services: &HostServicesV1,
    resources: &RecurrentResourcesV1,
    pending: &PendingStepV1,
) -> Result<(), BundleStatusV1> {
    let handles = resources.handles;
    let state_span = services
        .resolve_state_view(handles.state, 0)
        .map_err(status_of_v1)?;
    services
        .copy_host_to_device(
            handles.stream,
            state_span,
            &lanes_to_bytes_v1(&pending.lanes),
        )
        .map_err(status_of_v1)?;
    services
        .synchronize_stream(handles.stream)
        .map_err(status_of_v1)
}

/// Cancels the pending step, leaving the persistent state exactly as it was.
///
/// Only a commit writes the state, so a cancel has nothing to undo: it drops
/// the lanes the execution computed and the next execution reads what the last
/// commit left. That is why this cannot fail for a live handle.
pub fn cancel_v1(handle: BundleHandleV1) -> BundleStatusV1 {
    sweep_unreleased_v1();
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        if instance.resources.pending_requests.take().is_some() {
            return BundleStatusV1::Success;
        }
        match instance.resources.pending.take() {
            Some(pending) => {
                log_v1(
                    &instance.services,
                    &format!(
                        "recurrent fixture rolled back the step of request {}, so its persistent \
                         state is unchanged",
                        pending.request_id
                    ),
                );
                BundleStatusV1::Success
            }
            None => {
                log_v1(
                    &instance.services,
                    "recurrent fixture refused a cancel with no execution pending",
                );
                BundleStatusV1::InvalidState
            }
        }
    })
}

/// Releases one instance and every host resource it still holds.
///
/// A pending step is refused rather than rolled back. V1 allows at most one
/// transaction per instance, and closing it is the caller's call to make: a
/// destroy that rolled one back would decide the outcome of a transaction the
/// caller still owns, and would report as released an instance whose step
/// nobody chose to abandon. Nothing is touched on that path, so the same handle
/// still commits or cancels afterwards and destroys once it is closed.
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
        if instance.resources.pending_requests.is_some() {
            return BundleStatusV1::InvalidState;
        }
        if let Some(pending) = instance.resources.pending.as_ref() {
            let request_id = pending.request_id;
            log_v1(
                &instance.services,
                &format!(
                    "recurrent fixture refused to destroy an instance with the step of request \
                     {request_id} still pending"
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
                    "recurrent fixture could not release its host resources, and keeps exactly \
                     what the host still owns",
                );
                status
            }
        }
    })
}

/// Checkpoints contain the state after each input, never a recomputed prefix at commit.
fn run_request_batch_v1(
    services: &HostServicesV1,
    resources: &mut RecurrentResourcesV1,
    operation: &mut RecurrentStepOperationV1,
    request: &BatchRequestV1,
    sink: &mut ExecuteResultSinkV1,
) -> Result<(), BundleStatusV1> {
    let slots = request
        .state_slots()
        .ok_or(BundleStatusV1::InvalidArgument)?;
    if request.items().is_empty() || request.items().len() != request.appends().len() {
        return Err(BundleStatusV1::InvalidArgument);
    }
    let mut pending = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    let result = (|| {
        for ((item, append), &slot) in request.items().iter().zip(request.appends()).zip(slots) {
            let mut tokens = item.input_tokens().to_vec();
            let planned = item.phase_tokens() as usize;
            if !ids.insert(item.request_id())
                || tokens.is_empty()
                || planned > resources.handles.geometry.maximum_tokens as usize
                || (item.is_verify() && planned > 4)
                || (item.bundle_drafts() && (!item.is_verify() || tokens.len() != 1 || planned < 2))
                || (!item.bundle_drafts() && planned != tokens.len())
                || append.additional_tokens() != planned as u64
                || append.previous_tokens().checked_add(planned as u64) != Some(append.new_tokens())
                || (item.is_decode() && tokens.len() != 1)
                || resources
                    .requests
                    .iter()
                    .any(|(&id, state)| id != item.request_id() && state.slot == Some(slot))
            {
                return Err(BundleStatusV1::InvalidArgument);
            }
            let initial = match resources.requests.get(&item.request_id()) {
                Some(state)
                    if state.tokens == append.previous_tokens()
                        && state.slot.is_none_or(|owned| owned == slot) =>
                {
                    Some(state.lanes)
                }
                None if append.previous_tokens() == 0 => None,
                _ => return Err(BundleStatusV1::InvalidState),
            };
            if item.bundle_drafts() {
                // The synthetic drafter starts from the committed lanes,
                // including lanes recovered by a prefix restore. It never
                // reconstructs those lanes from the engine's history.
                let mut lanes =
                    fold_tokens_v1(&initial.ok_or(BundleStatusV1::InvalidState)?, &tokens);
                while tokens.len() < planned {
                    let row = logits_row_v1(&lanes, resources.handles.geometry.vocabulary);
                    let logits: Vec<_> = row
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().expect("f32 row")))
                        .collect();
                    let mut best = 0;
                    for token in 1..logits.len() {
                        if logits[token] > logits[best] {
                            best = token;
                        }
                    }
                    tokens.push(best as u32);
                    lanes = fold_tokens_v1(&lanes, &[best as u32]);
                }
                sink.emit_draft_tokens(item.request_id(), &tokens[1..])
                    .map_err(status_of_v1)?;
            }
            let mut rows = Vec::new();
            let first = if item.is_verify() || item.prompt_rows() {
                1
            } else {
                tokens.len()
            };
            for count in first..=tokens.len() {
                run_step_v1(
                    services,
                    resources,
                    operation,
                    &ScheduledStepV1 {
                        request_id: item.request_id(),
                        tokens: tokens[..count].to_vec(),
                        prompt_rows: false,
                        initial,
                        emit: item.is_verify()
                            || item.prompt_rows()
                            || (item.expects_logits() && count == tokens.len()),
                    },
                    sink,
                )?;
                if !item.is_verify() {
                    rows.clear();
                }
                rows.push(
                    resources
                        .pending
                        .take()
                        .ok_or(BundleStatusV1::InvalidState)?
                        .lanes,
                );
            }
            pending.push(RequestCheckpointV1 {
                request_id: item.request_id(),
                slot,
                previous_tokens: append.previous_tokens(),
                input_count: tokens.len(),
                verify: item.is_verify(),
                rows,
            });
        }
        Ok(())
    })();
    resources.pending = None;
    result?;
    resources.pending_requests = Some(pending);
    Ok(())
}

fn commit_requests_v1(
    resources: &mut RecurrentResourcesV1,
    kept: Option<&[runtime_bundle_abi::BundleKeptRowsV1]>,
) -> BundleStatusV1 {
    let Some(pending) = &resources.pending_requests else {
        return BundleStatusV1::InvalidState;
    };
    let verifies = pending
        .iter()
        .filter(|item| item.verify)
        .collect::<Vec<_>>();
    let records = kept.unwrap_or(&[]);
    if records.len() != verifies.len()
        || records.iter().zip(&verifies).any(|(record, item)| {
            record.request_id != item.request_id
                || record.kept_rows == 0
                || record.kept_rows as usize > item.rows.len()
        })
    {
        return BundleStatusV1::InvalidArgument;
    }
    let mut records = records.iter();
    for item in resources
        .pending_requests
        .take()
        .expect("validated pending batch")
    {
        let count = if item.verify {
            records.next().expect("validated verify record").kept_rows as usize
        } else {
            item.input_count
        };
        let lanes = if item.verify {
            item.rows[count - 1]
        } else {
            item.rows[0]
        };
        resources.requests.insert(
            item.request_id,
            RequestStateV1 {
                slot: Some(item.slot),
                tokens: item.previous_tokens + count as u64,
                lanes,
            },
        );
    }
    BundleStatusV1::Success
}

pub fn commit_accepted_v1(
    handle: BundleHandleV1,
    kept: &[runtime_bundle_abi::BundleKeptRowsV1],
) -> BundleStatusV1 {
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        commit_requests_v1(&mut instance.resources, Some(kept))
    })
}

pub fn retire_request_v1(handle: BundleHandleV1, request_id: u64) -> BundleStatusV1 {
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        if instance.resources.pending_requests.is_some() || instance.resources.pending.is_some() {
            return BundleStatusV1::InvalidState;
        }
        instance.resources.requests.remove(&request_id);
        BundleStatusV1::Success
    })
}

/// A snapshot holds the committed lanes then their token count, padded to
/// whole serving-pool pages.
const SNAPSHOT_STATE_BYTES: usize = crate::recurrent::RECURRENT_STATE_BYTES_V1 as usize;

/// Pages of `page_bytes` one snapshot fills.
const fn snapshot_pages_v1(page_bytes: u64) -> usize {
    (SNAPSHOT_STATE_BYTES + 8).div_ceil(page_bytes as usize)
}

pub fn describe_prefix_state_v1(
    handle: BundleHandleV1,
) -> runtime_bundle_abi::BundlePrefixStateInfoV1 {
    use runtime_bundle_abi::BundlePrefixStateInfoV1 as Info;
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return Info::failure(BundleStatusV1::InvalidState);
        };
        let Some(instance) = table.get_mut(handle) else {
            return Info::failure(BundleStatusV1::StaleHandle);
        };
        if instance.resources.pending.is_some() || instance.resources.pending_requests.is_some() {
            return Info::failure(BundleStatusV1::InvalidState);
        }
        Info::success(snapshot_pages_v1(instance.resources.handles.kv_page_bytes) as u64)
    })
}

fn snapshot_spans_v1(
    services: &HostServicesV1,
    pages: &[u64],
    bytes: u64,
) -> Result<Vec<DeviceBufferSpanV1>, BundleStatusV1> {
    if pages.len() != snapshot_pages_v1(bytes)
        || pages
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != pages.len()
    {
        return Err(BundleStatusV1::InvalidArgument);
    }
    let pool = services.resolve_serving_kv_pool(0).map_err(status_of_v1)?;
    pages
        .iter()
        .map(|page| {
            let offset = page
                .checked_mul(bytes)
                .filter(|offset| {
                    offset
                        .checked_add(bytes)
                        .is_some_and(|end| end <= pool.length)
                })
                .ok_or(BundleStatusV1::InvalidArgument)?;
            Ok(DeviceBufferSpanV1::new(
                pool.buffer,
                pool.offset + offset,
                bytes,
            ))
        })
        .collect()
}

pub fn capture_prefix_state_v1(
    handle: BundleHandleV1,
    request: runtime_bundle_abi::PrefixStateRequestV1,
) -> BundleStatusV1 {
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        let resources = &instance.resources;
        if resources.pending.is_some()
            || resources.pending_requests.is_some()
            || !request.token_ids.is_empty()
        {
            return BundleStatusV1::InvalidState;
        }
        let Some(state) = resources
            .requests
            .get(&request.request_id)
            .filter(|state| state.tokens == request.tokens)
        else {
            return BundleStatusV1::InvalidState;
        };
        let result = (|| {
            let page_bytes = resources.handles.kv_page_bytes;
            let spans = snapshot_spans_v1(&instance.services, &request.pages, page_bytes)?;
            let mut bytes = lanes_to_bytes_v1(&state.lanes);
            bytes.extend_from_slice(&state.tokens.to_le_bytes());
            bytes.resize(spans.len() * page_bytes as usize, 0);
            for (span, chunk) in spans
                .into_iter()
                .zip(bytes.chunks_exact(page_bytes as usize))
            {
                instance
                    .services
                    .copy_host_to_device(resources.handles.stream, span, chunk)
                    .map_err(status_of_v1)?;
            }
            instance
                .services
                .synchronize_stream(resources.handles.stream)
                .map_err(status_of_v1)
        })();
        result.err().unwrap_or(BundleStatusV1::Success)
    })
}

pub fn restore_prefix_state_v1(
    handle: BundleHandleV1,
    request: runtime_bundle_abi::PrefixStateRequestV1,
) -> BundleStatusV1 {
    restore_prefix_state_into_v1(handle, request, None)
}

/// Restores into the host's slot at once, as a device-state bundle must.
pub fn restore_prefix_state_in_slot_v1(
    handle: BundleHandleV1,
    request: runtime_bundle_abi::PrefixStateRequestV1,
    state_slot: u64,
) -> BundleStatusV1 {
    restore_prefix_state_into_v1(handle, request, Some(state_slot))
}

fn restore_prefix_state_into_v1(
    handle: BundleHandleV1,
    request: runtime_bundle_abi::PrefixStateRequestV1,
    state_slot: Option<u64>,
) -> BundleStatusV1 {
    INSTANCES_V1.with(|table| {
        let Ok(mut table) = table.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        let Some(instance) = table.get_mut(handle) else {
            return BundleStatusV1::StaleHandle;
        };
        let resources = &mut instance.resources;
        if resources.pending.is_some()
            || resources.pending_requests.is_some()
            || resources.requests.contains_key(&request.request_id)
            || request.token_ids.len() as u64 != request.tokens
            || state_slot.is_some_and(|slot| {
                resources
                    .requests
                    .values()
                    .any(|state| state.slot == Some(slot))
            })
        {
            return BundleStatusV1::InvalidState;
        }
        let result = (|| {
            let page_bytes = resources.handles.kv_page_bytes;
            let spans = snapshot_spans_v1(&instance.services, &request.pages, page_bytes)?;
            let mut bytes = vec![0_u8; spans.len() * page_bytes as usize];
            for (span, chunk) in spans
                .into_iter()
                .zip(bytes.chunks_exact_mut(page_bytes as usize))
            {
                instance
                    .services
                    .copy_device_to_host(resources.handles.stream, chunk, span)
                    .map_err(status_of_v1)?;
            }
            instance
                .services
                .synchronize_stream(resources.handles.stream)
                .map_err(status_of_v1)?;
            let lanes = lanes_from_bytes_v1(&bytes[..SNAPSHOT_STATE_BYTES])
                .ok_or(BundleStatusV1::InvalidState)?;
            if u64::from_le_bytes(
                bytes[SNAPSHOT_STATE_BYTES..SNAPSHOT_STATE_BYTES + 8]
                    .try_into()
                    .expect("fixed boundary"),
            ) != request.tokens
            {
                return Err(BundleStatusV1::InvalidArgument);
            }
            // Without the state-slot tail the slot ID arrives in the first
            // batch; restore is keyed by request.
            resources.requests.insert(
                request.request_id,
                RequestStateV1 {
                    slot: state_slot,
                    tokens: request.tokens,
                    lanes,
                },
            );
            Ok(())
        })();
        result.err().unwrap_or(BundleStatusV1::Success)
    })
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::{BundleHandleV1, BundleStatusV1};

    use crate::contract::{ProgramGeometryV1, RECURRENT_TOKEN_BOUND_V1};
    use crate::recurrent::RECURRENT_STATE_BYTES_V1;

    use super::{cancel_v1, check_geometry_v1, commit_v1, destroy_v1};

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
    fn a_program_image_that_disagrees_with_the_state_layout_is_refused() {
        let declared = ProgramGeometryV1::declared();

        assert_eq!(
            check_geometry_v1(declared, RECURRENT_STATE_BYTES_V1),
            Ok(())
        );
        for wrong in [
            ProgramGeometryV1 {
                state_bytes: declared.state_bytes / 2,
                ..declared
            },
            ProgramGeometryV1 {
                lanes: declared.lanes + 1,
                ..declared
            },
            ProgramGeometryV1 {
                vocabulary: 0,
                ..declared
            },
            ProgramGeometryV1 {
                maximum_tokens: RECURRENT_TOKEN_BOUND_V1 + 1,
                ..declared
            },
        ] {
            assert_eq!(
                check_geometry_v1(wrong, RECURRENT_STATE_BYTES_V1),
                Err(BundleStatusV1::Unsupported),
                "{wrong:?} must not describe this fixture's state"
            );
        }
    }
}
