//! The sole ABI pointer boundary for the V1 bundle contract.
//!
//! Native code owns every ABI pointer and keeps it valid only for the call that
//! received it. Every read and write of such a pointer happens in this module,
//! inside a generated trampoline or the host-service adapter. Code on either
//! side of the boundary sees owned Rust values or borrows that end with the
//! call, so no pointer-backed record outlives its callback and no caller writes
//! unsafe code.

// SAFETY-REVIEWED: this module contains the ABI trampoline pointer boundary.
#![allow(unsafe_code)]

use std::cell::RefCell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::rc::Rc;
use std::slice;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::host_services::{
    AuthenticatedComponentMetadataV1, CollectiveRequestV1, CompiledImageSpanV1,
    CustomAddressSpaceV1, CustomCompletionV1, CustomOperandV1, DeviceBufferSpanV1,
    HostServiceResultV1, HostServicesBackendV1, KernelArgumentV1, LaunchConfigV1, LentLaunchV1,
    LentRangeV1, LentStreamV1, LogitsRowSinkV1, MappedRowGatherV1, OwnedDraftTokensV1,
    OwnedLogitsRowV1, OwnedLogitsViewV1, PhysicalKvLayoutV1, PhysicalKvRequestV1,
    RankArenaUploadV1, ResultSinkLimitsV1, StateRequestV1, TensorMapV1,
};
use crate::{
    BUNDLE_ABI_VERSION_V1, BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT,
    BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS, BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_BUFFER,
    BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY, BUNDLE_KERNEL_ARGUMENT_V1_I32,
    BUNDLE_KERNEL_ARGUMENT_V1_NULL_DEVICE_POINTER, BUNDLE_KERNEL_ARGUMENT_V1_OPAQUE_BYTES,
    BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP, BUNDLE_KERNEL_ARGUMENT_V1_U32,
    BUNDLE_KERNEL_ARGUMENT_V1_U64, BundleAbiError, BundleAuthenticatedComponentHandleV1,
    BundleAuthenticatedComponentMetadataV1, BundleByteViewV1, BundleCollectiveHandleV1,
    BundleCollectiveRequestV1, BundleCommitAcceptedFnV1, BundleCompiledImageHandleV1,
    BundleContextHandleV1, BundleCreateFnV1, BundleCreateResultV1, BundleCudaModuleHandleV1,
    BundleCudaSymbolHandleV1, BundleCustomLaunchV1, BundleCustomOperandV1,
    BundleCustomOperationFnV1, BundleDeviceBufferHandleV1, BundleDeviceBufferViewV1,
    BundleDevicePointerArrayV1, BundleDraftTokensV1, BundleEmitLogitsRowFnV1, BundleEventHandleV1,
    BundleExecuteFnV1, BundleExecuteResultSinkV1, BundleGraphHandleV1, BundleHandleV1,
    BundleHostServicesV1, BundleKeptRowsV1, BundleKernelArgumentV1, BundleKvRequestHandleV1,
    BundleKvTransactionHandleV1, BundleLaunchConfigV1, BundleLogitsRowV1, BundleLogitsViewV1,
    BundleMappedRowGatherV1, BundleMemoryPoolHandleV1, BundleMutByteViewV1,
    BundlePeerAccessHandleV1, BundlePhysicalKvHandleV1, BundlePhysicalKvRequestV1,
    BundleRankArenaHandleV1, BundleRankArenaUploadV1, BundleRawApiV1, BundleRawBatchLimitsV1,
    BundleRawClusterResidencyV1, BundleRawCompiledImageViewV1, BundleRawSpeculationV1,
    BundleStateHandleV1, BundleStateRequestV1, BundleStatusV1, BundleStreamHandleV1,
    BundleTensorMapV1, ExecuteResultSinkV1, charge_result_capacity_rows_v1,
    charge_result_capacity_v1, status_result, struct_size_v1,
    validate_authenticated_component_metadata_v1, validate_batch_limits_v1, validate_byte_view_v1,
    validate_cluster_residency_v1, validate_collective_request_v1, validate_compiled_image_view_v1,
    validate_custom_operand_struct_size_v1, validate_custom_operands_v1,
    validate_device_buffer_view_v1, validate_device_pointer_array_v1, validate_draft_tokens_v1,
    validate_execute_logits_row_v1, validate_execute_logits_view_v1,
    validate_execute_result_sink_v1, validate_host_services_v1, validate_kept_rows_v1,
    validate_kernel_argument_struct_size_v1, validate_kernel_arguments_v1,
    validate_launch_config_v1, validate_mut_byte_view_v1, validate_physical_kv_request_v1,
    validate_speculation_v1, validate_state_request_v1, validate_tensor_map_v1,
};

/// Converts a Rust panic into the V1 status code before an ABI callback returns.
pub fn catch_panic_status_v1(callback: impl FnOnce() -> BundleStatusV1) -> BundleStatusV1 {
    catch_unwind(AssertUnwindSafe(callback)).unwrap_or(BundleStatusV1::Panic)
}

/// Converts a Rust panic into a failed V1 create result before an ABI callback returns.
pub fn catch_panic_create_result_v1(
    callback: impl FnOnce() -> BundleCreateResultV1,
) -> BundleCreateResultV1 {
    catch_unwind(AssertUnwindSafe(callback))
        .unwrap_or_else(|_| BundleCreateResultV1::failure(BundleStatusV1::Panic))
}

/// Converts a Rust panic into a null API pointer before the native entry returns.
pub fn catch_panic_api_pointer_v1(
    callback: impl FnOnce() -> *const BundleRawApiV1,
) -> *const BundleRawApiV1 {
    catch_unwind(AssertUnwindSafe(callback)).unwrap_or(std::ptr::null())
}

/// Maps callback-input validation failures to a V1 status and catches a panic
/// before it can cross the native ABI boundary.
pub fn catch_panic_callback_status_v1(
    callback: impl FnOnce() -> Result<BundleStatusV1, BundleAbiError>,
) -> BundleStatusV1 {
    catch_panic_status_v1(|| match callback() {
        Ok(status) => status,
        Err(error) => callback_error_status_v1(error),
    })
}

fn callback_error_status_v1(error: BundleAbiError) -> BundleStatusV1 {
    match error {
        BundleAbiError::Status(status) => status,
        BundleAbiError::AbiVersion { .. } => BundleStatusV1::AbiMismatch,
        BundleAbiError::ExecuteResultCapacity => BundleStatusV1::ResourceExhausted,
        // The operation's own code is the bundle's, not this host's.
        BundleAbiError::CustomOperation(_) => BundleStatusV1::BundleError,
        // A host that reported a failure it cannot name has broken the
        // contract, which is its fault rather than an argument's.
        BundleAbiError::MissingCustomOperationStatus => BundleStatusV1::HostError,
        BundleAbiError::Lifecycle(_) | BundleAbiError::Poisoned => BundleStatusV1::InvalidState,
        BundleAbiError::StructSize { .. }
        | BundleAbiError::StructSizeExact { .. }
        | BundleAbiError::MissingApiCallback(_)
        | BundleAbiError::MissingHostService(_)
        | BundleAbiError::MissingExecuteResultCallback
        | BundleAbiError::InvalidByteView
        | BundleAbiError::InvalidMutableByteView
        | BundleAbiError::InvalidReserved { .. }
        | BundleAbiError::InvalidAlignment { .. }
        | BundleAbiError::InvalidRange(_)
        | BundleAbiError::InvalidKernelArgumentKind(_)
        | BundleAbiError::MissingKernelArgumentPayload(_)
        | BundleAbiError::UnselectedKernelArgumentPayload(_)
        | BundleAbiError::InvalidDevicePointerArraySlots(_)
        | BundleAbiError::InvalidCustomOperandFlags(_)
        | BundleAbiError::InvalidCustomAddressSpace(_)
        | BundleAbiError::InvalidCustomLaunch(_)
        | BundleAbiError::InvalidCustomCompletion(_)
        | BundleAbiError::MissingCustomOperationCallback
        | BundleAbiError::InvalidTensorMapElement(_)
        | BundleAbiError::InvalidTensorMapSwizzle(_)
        | BundleAbiError::InvalidTensorMapGeometry(_)
        | BundleAbiError::InvalidLaunchFlags(_)
        | BundleAbiError::InvalidLaunchCluster(_)
        | BundleAbiError::LengthOutOfRange { .. }
        | BundleAbiError::InvalidPointerLength(_)
        | BundleAbiError::NullPointer(_)
        | BundleAbiError::MisalignedPointer(_)
        | BundleAbiError::InvalidLogitsRow
        | BundleAbiError::InvalidLogitsView
        | BundleAbiError::InvalidDraftTokens
        | BundleAbiError::InvalidKeptRows
        | BundleAbiError::InvalidHandle => BundleStatusV1::InvalidArgument,
    }
}

/// Converts an ABI `u64` length to a Rust slice length without truncation.
pub fn checked_u64_to_usize_v1(
    surface: &'static str,
    length: u64,
) -> Result<usize, BundleAbiError> {
    let length = usize::try_from(length).map_err(|_| BundleAbiError::LengthOutOfRange {
        surface,
        actual: length,
    })?;
    if length > isize::MAX as usize {
        return Err(BundleAbiError::LengthOutOfRange {
            surface,
            actual: u64::try_from(length).expect("usize value fits u64"),
        });
    }
    Ok(length)
}

#[derive(Default)]
struct ExecuteObservationV1 {
    active: AtomicBool,
    service_called: AtomicBool,
    diagnostic: Mutex<Option<String>>,
}

trait BackendOwnerV1 {
    fn reference_count(&self) -> usize;
}

struct HostServiceContextV1<B> {
    backend: Rc<B>,
    observation: Arc<ExecuteObservationV1>,
}

impl<B> BackendOwnerV1 for HostServiceContextV1<B> {
    fn reference_count(&self) -> usize {
        Rc::strong_count(&self.backend)
    }
}

/// A complete V1 service table backed by one safe host implementation.
///
/// The table owns its backend and the C-layout record the bundle receives, so
/// both stay at a fixed address for the lifetime of the instance that uses them.
pub struct HostServicesTableV1 {
    raw: Box<BundleHostServicesV1>,
    caller: HostServicesV1,
    backend: Box<dyn BackendOwnerV1>,
    observation: Arc<ExecuteObservationV1>,
}

impl HostServicesTableV1 {
    /// Builds the C-layout table whose callbacks decode every ABI pointer and
    /// then call `backend`.
    #[must_use]
    pub fn new<B: HostServicesBackendV1 + 'static>(backend: Rc<B>) -> Self {
        let observation = Arc::new(ExecuteObservationV1::default());
        let backend = Box::new(HostServiceContextV1 {
            backend,
            observation: Arc::clone(&observation),
        });
        let user_data = ptr::from_ref(backend.as_ref()).cast::<c_void>().cast_mut();
        let raw = Box::new(BundleHostServicesV1 {
            struct_size: struct_size_v1::<BundleHostServicesV1>(),
            abi_version: BUNDLE_ABI_VERSION_V1,
            user_data,
            log: Some(log_trampoline_v1::<B>),
            select_context: Some(select_context_trampoline_v1::<B>),
            release_context: Some(release_context_trampoline_v1::<B>),
            select_memory_pool: Some(select_memory_pool_trampoline_v1::<B>),
            set_memory_pool_peer_access: Some(set_memory_pool_peer_access_trampoline_v1::<B>),
            release_memory_pool: Some(release_memory_pool_trampoline_v1::<B>),
            allocate_device_buffer: Some(allocate_device_buffer_trampoline_v1::<B>),
            release_device_buffer: Some(release_device_buffer_trampoline_v1::<B>),
            open_compiled_image: Some(open_compiled_image_trampoline_v1::<B>),
            upload_rank_arena: Some(upload_rank_arena_trampoline_v1::<B>),
            upload_rank_arenas: Some(upload_rank_arenas_trampoline_v1::<B>),
            release_rank_arena: Some(release_rank_arena_trampoline_v1::<B>),
            resolve_compiled_image_view: Some(resolve_compiled_image_view_trampoline_v1::<B>),
            release_compiled_image: Some(release_compiled_image_trampoline_v1::<B>),
            open_authenticated_component: Some(open_authenticated_component_trampoline_v1::<B>),
            inspect_authenticated_component: Some(
                inspect_authenticated_component_trampoline_v1::<B>,
            ),
            read_authenticated_component: Some(read_authenticated_component_trampoline_v1::<B>),
            release_authenticated_component: Some(
                release_authenticated_component_trampoline_v1::<B>,
            ),
            acquire_peer_access: Some(acquire_peer_access_trampoline_v1::<B>),
            release_peer_access: Some(release_peer_access_trampoline_v1::<B>),
            create_stream: Some(create_stream_trampoline_v1::<B>),
            select_stage_stream: Some(select_stage_stream_trampoline_v1::<B>),
            synchronize_stream: Some(synchronize_stream_trampoline_v1::<B>),
            release_stream: Some(release_stream_trampoline_v1::<B>),
            create_event: Some(create_event_trampoline_v1::<B>),
            record_event: Some(record_event_trampoline_v1::<B>),
            wait_event: Some(wait_event_trampoline_v1::<B>),
            synchronize_event: Some(synchronize_event_trampoline_v1::<B>),
            release_event: Some(release_event_trampoline_v1::<B>),
            copy_host_to_device: Some(copy_host_to_device_trampoline_v1::<B>),
            copy_device_to_host: Some(copy_device_to_host_trampoline_v1::<B>),
            copy_device_to_device: Some(copy_device_to_device_trampoline_v1::<B>),
            copy_peer_to_peer: Some(copy_peer_to_peer_trampoline_v1::<B>),
            zero_device_buffer: Some(zero_device_buffer_trampoline_v1::<B>),
            write_device_pointer_table: Some(write_device_pointer_table_trampoline_v1::<B>),
            load_cuda_module: Some(load_cuda_module_trampoline_v1::<B>),
            resolve_cuda_symbol: Some(resolve_cuda_symbol_trampoline_v1::<B>),
            configure_cuda_symbol: Some(configure_cuda_symbol_trampoline_v1::<B>),
            launch_cuda_symbol: Some(launch_cuda_symbol_trampoline_v1::<B>),
            release_cuda_symbol: Some(release_cuda_symbol_trampoline_v1::<B>),
            release_cuda_module: Some(release_cuda_module_trampoline_v1::<B>),
            allocate_physical_kv: Some(allocate_physical_kv_trampoline_v1::<B>),
            create_kv_request: Some(create_kv_request_trampoline_v1::<B>),
            prepare_kv: Some(prepare_kv_trampoline_v1::<B>),
            commit_kv: Some(commit_kv_trampoline_v1::<B>),
            cancel_kv: Some(cancel_kv_trampoline_v1::<B>),
            resolve_kv_view: Some(resolve_kv_view_trampoline_v1::<B>),
            release_kv_request: Some(release_kv_request_trampoline_v1::<B>),
            release_physical_kv: Some(release_physical_kv_trampoline_v1::<B>),
            allocate_state: Some(allocate_state_trampoline_v1::<B>),
            resolve_state_view: Some(resolve_state_view_trampoline_v1::<B>),
            release_state: Some(release_state_trampoline_v1::<B>),
            create_collective: Some(create_collective_trampoline_v1::<B>),
            execute_collective: Some(execute_collective_trampoline_v1::<B>),
            release_collective: Some(release_collective_trampoline_v1::<B>),
            run_custom_operation: Some(run_custom_operation_trampoline_v1::<B>),
            begin_graph_capture: Some(begin_graph_capture_trampoline_v1::<B>),
            end_graph_capture: Some(end_graph_capture_trampoline_v1::<B>),
            launch_graph: Some(launch_graph_trampoline_v1::<B>),
            release_graph: Some(release_graph_trampoline_v1::<B>),
            acquire_serving_kv: Some(acquire_serving_kv_trampoline_v1::<B>),
            resolve_serving_kv_pool: Some(resolve_serving_kv_pool_trampoline_v1::<B>),
            query_speculation_v1: Some(query_speculation_trampoline_v1::<B>),
            query_batch_limits_v1: Some(query_batch_limits_trampoline_v1::<B>),
            query_cluster_residency_v1: Some(query_cluster_residency_trampoline_v1::<B>),
            gather_mapped_rows_v1: Some(gather_mapped_rows_trampoline_v1::<B>),
        });
        // SAFETY-REVIEWED: this table builds every callback itself and owns the backend they read.
        let checked = unsafe { HostServicesV1::new(*raw) };
        let caller = checked.expect("an adapter-built V1 table is complete");
        Self {
            raw,
            caller,
            backend,
            observation,
        }
    }

    /// Borrows the C-layout record for structural checks.
    #[must_use]
    pub fn services(&self) -> &BundleHostServicesV1 {
        &self.raw
    }

    /// Borrows a safe caller for this table, valid while the table lives.
    #[must_use]
    pub fn caller(&self) -> &HostServicesV1 {
        &self.caller
    }

    /// Counts the owners of the backend allocation this table points at.
    #[must_use]
    pub fn backend_reference_count(&self) -> usize {
        self.backend.reference_count()
    }

    pub(crate) fn begin_execute(&self) {
        self.observation
            .service_called
            .store(false, Ordering::Release);
        *self
            .observation
            .diagnostic
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
        self.observation.active.store(true, Ordering::Release);
    }

    pub(crate) fn end_execute(&self) -> (bool, Option<String>) {
        self.observation.active.store(false, Ordering::Release);
        let diagnostic = self
            .observation
            .diagnostic
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        (
            self.observation.service_called.load(Ordering::Acquire),
            diagnostic,
        )
    }

    pub(crate) fn raw_pointer(&self) -> *const BundleHostServicesV1 {
        ptr::from_ref::<BundleHostServicesV1>(&self.raw)
    }
}

fn context_from_user_data_v1<'a, B: HostServicesBackendV1>(
    user_data: *mut c_void,
) -> Result<&'a HostServiceContextV1<B>, BundleAbiError> {
    let context = user_data.cast::<HostServiceContextV1<B>>().cast_const();
    validate_input_pointer_v1(context, "host service context")?;
    // SAFETY-REVIEWED: every table points at its own live, fixed-address context allocation.
    Ok(unsafe { &*context })
}

fn backend_from_user_data_v1<'a, B: HostServicesBackendV1>(
    user_data: *mut c_void,
) -> Result<&'a B, BundleAbiError> {
    let context = context_from_user_data_v1::<B>(user_data)?;
    // Treat every non-logging service conservatively, including failed calls
    // and future services. Recovery never depends on a device operation list.
    if context.observation.active.load(Ordering::Acquire) {
        context
            .observation
            .service_called
            .store(true, Ordering::Release);
    }
    Ok(&context.backend)
}

const fn failed_host_status_v1(status: BundleStatusV1) -> BundleStatusV1 {
    if status.is_success() {
        BundleStatusV1::HostError
    } else {
        status
    }
}

fn host_unit_status_v1(result: HostServiceResultV1<()>) -> BundleStatusV1 {
    match result {
        Ok(()) => BundleStatusV1::Success,
        Err(status) => failed_host_status_v1(status),
    }
}

/// One host-service result value and the C-layout record that carries it back.
trait AbiOutputV1 {
    type Raw: Copy;

    fn into_raw(self) -> Self::Raw;
}

macro_rules! identity_abi_output_v1 {
    ($($type:ty),* $(,)?) => {
        $(
            impl AbiOutputV1 for $type {
                type Raw = Self;

                fn into_raw(self) -> Self::Raw {
                    self
                }
            }
        )*
    };
}

identity_abi_output_v1!(
    BundleCollectiveHandleV1,
    BundleAuthenticatedComponentHandleV1,
    BundleCompiledImageHandleV1,
    BundleContextHandleV1,
    BundleCudaModuleHandleV1,
    BundleCudaSymbolHandleV1,
    BundleDeviceBufferHandleV1,
    BundleEventHandleV1,
    BundleGraphHandleV1,
    BundleKvRequestHandleV1,
    BundleKvTransactionHandleV1,
    BundleMemoryPoolHandleV1,
    BundlePeerAccessHandleV1,
    BundlePhysicalKvHandleV1,
    BundleRankArenaHandleV1,
    BundleStateHandleV1,
    BundleStreamHandleV1,
    BundleRawSpeculationV1,
    BundleRawBatchLimitsV1,
    BundleRawClusterResidencyV1,
);

impl AbiOutputV1 for DeviceBufferSpanV1 {
    type Raw = BundleDeviceBufferViewV1;

    fn into_raw(self) -> Self::Raw {
        self.to_view()
    }
}

impl AbiOutputV1 for u64 {
    type Raw = u64;

    fn into_raw(self) -> Self::Raw {
        self
    }
}

impl AbiOutputV1 for CompiledImageSpanV1 {
    type Raw = BundleRawCompiledImageViewV1;

    fn into_raw(self) -> Self::Raw {
        self.to_view()
    }
}

impl AbiOutputV1 for AuthenticatedComponentMetadataV1 {
    type Raw = BundleAuthenticatedComponentMetadataV1;

    fn into_raw(self) -> Self::Raw {
        self.to_raw()
    }
}

fn finish_output_v1<T: AbiOutputV1>(
    output: *mut T::Raw,
    result: HostServiceResultV1<T>,
) -> Result<BundleStatusV1, BundleAbiError> {
    match result {
        Ok(value) => {
            write_output_v1(output, value.into_raw())?;
            Ok(BundleStatusV1::Success)
        }
        Err(status) => Ok(failed_host_status_v1(status)),
    }
}

macro_rules! status_service_trampoline_v1 {
    ($name:ident, $method:ident, ($($argument:ident: $type:ty),* $(,)?)) => {
        // SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
        unsafe extern "C" fn $name<B: HostServicesBackendV1>(
            user_data: *mut c_void,
            $($argument: $type,)*
        ) -> BundleStatusV1 {
            catch_panic_callback_status_v1(|| {
                let backend = backend_from_user_data_v1::<B>(user_data)?;
                Ok(host_unit_status_v1(backend.$method($($argument),*)))
            })
        }
    };
}

macro_rules! output_service_trampoline_v1 {
    (
        $name:ident,
        $method:ident,
        $output:ty,
        ($($argument:ident: $type:ty),* $(,)?)
    ) => {
        // SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
        unsafe extern "C" fn $name<B: HostServicesBackendV1>(
            user_data: *mut c_void,
            $($argument: $type,)*
            output: *mut $output,
        ) -> BundleStatusV1 {
            catch_panic_callback_status_v1(|| {
                let backend = backend_from_user_data_v1::<B>(user_data)?;
                validate_output_pointer_v1(output, concat!(stringify!($method), " output"))?;
                finish_output_v1(output, backend.$method($($argument),*))
            })
        }
    };
}

status_service_trampoline_v1!(
    release_context_trampoline_v1,
    release_context,
    (context: BundleContextHandleV1)
);
status_service_trampoline_v1!(
    release_memory_pool_trampoline_v1,
    release_memory_pool,
    (pool: BundleMemoryPoolHandleV1)
);
status_service_trampoline_v1!(
    release_device_buffer_trampoline_v1,
    release_device_buffer,
    (buffer: BundleDeviceBufferHandleV1)
);
status_service_trampoline_v1!(
    release_rank_arena_trampoline_v1,
    release_rank_arena,
    (arena: BundleRankArenaHandleV1)
);
status_service_trampoline_v1!(
    release_compiled_image_trampoline_v1,
    release_compiled_image,
    (image: BundleCompiledImageHandleV1)
);
status_service_trampoline_v1!(
    release_authenticated_component_trampoline_v1,
    release_authenticated_component,
    (component: BundleAuthenticatedComponentHandleV1)
);
status_service_trampoline_v1!(
    release_peer_access_trampoline_v1,
    release_peer_access,
    (peer_access: BundlePeerAccessHandleV1)
);
status_service_trampoline_v1!(
    synchronize_stream_trampoline_v1,
    synchronize_stream,
    (stream: BundleStreamHandleV1)
);
status_service_trampoline_v1!(
    release_stream_trampoline_v1,
    release_stream,
    (stream: BundleStreamHandleV1)
);
status_service_trampoline_v1!(
    record_event_trampoline_v1,
    record_event,
    (event: BundleEventHandleV1, stream: BundleStreamHandleV1)
);
status_service_trampoline_v1!(
    wait_event_trampoline_v1,
    wait_event,
    (stream: BundleStreamHandleV1, event: BundleEventHandleV1)
);
status_service_trampoline_v1!(
    synchronize_event_trampoline_v1,
    synchronize_event,
    (event: BundleEventHandleV1)
);
status_service_trampoline_v1!(
    release_event_trampoline_v1,
    release_event,
    (event: BundleEventHandleV1)
);
status_service_trampoline_v1!(
    configure_cuda_symbol_trampoline_v1,
    configure_cuda_symbol,
    (
        symbol: BundleCudaSymbolHandleV1,
        dynamic_shared_memory_bytes: u32,
    )
);
status_service_trampoline_v1!(
    release_cuda_symbol_trampoline_v1,
    release_cuda_symbol,
    (symbol: BundleCudaSymbolHandleV1)
);
status_service_trampoline_v1!(
    release_cuda_module_trampoline_v1,
    release_cuda_module,
    (module: BundleCudaModuleHandleV1)
);
status_service_trampoline_v1!(
    commit_kv_trampoline_v1,
    commit_kv,
    (transaction: BundleKvTransactionHandleV1)
);
status_service_trampoline_v1!(
    cancel_kv_trampoline_v1,
    cancel_kv,
    (transaction: BundleKvTransactionHandleV1)
);
status_service_trampoline_v1!(
    release_kv_request_trampoline_v1,
    release_kv_request,
    (request: BundleKvRequestHandleV1)
);
status_service_trampoline_v1!(
    release_physical_kv_trampoline_v1,
    release_physical_kv,
    (kv: BundlePhysicalKvHandleV1)
);
status_service_trampoline_v1!(
    release_state_trampoline_v1,
    release_state,
    (state: BundleStateHandleV1)
);
status_service_trampoline_v1!(
    release_collective_trampoline_v1,
    release_collective,
    (collective: BundleCollectiveHandleV1)
);
status_service_trampoline_v1!(begin_graph_capture_trampoline_v1, begin_graph_capture, (stream: BundleStreamHandleV1));
output_service_trampoline_v1!(end_graph_capture_trampoline_v1, end_graph_capture, BundleGraphHandleV1, (stream: BundleStreamHandleV1));
status_service_trampoline_v1!(launch_graph_trampoline_v1, launch_graph, (stream: BundleStreamHandleV1, graph: BundleGraphHandleV1));
status_service_trampoline_v1!(release_graph_trampoline_v1, release_graph, (graph: BundleGraphHandleV1));

output_service_trampoline_v1!(
    select_context_trampoline_v1,
    select_context,
    BundleContextHandleV1,
    (rank: u32, device_ordinal: i32)
);
output_service_trampoline_v1!(
    select_memory_pool_trampoline_v1,
    select_memory_pool,
    BundleMemoryPoolHandleV1,
    (context: BundleContextHandleV1)
);
output_service_trampoline_v1!(
    allocate_device_buffer_trampoline_v1,
    allocate_device_buffer,
    BundleDeviceBufferHandleV1,
    (
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        byte_length: u64,
        alignment: u64,
    )
);
output_service_trampoline_v1!(
    upload_rank_arena_trampoline_v1,
    upload_rank_arena,
    BundleRankArenaHandleV1,
    (
        context: BundleContextHandleV1,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        image: BundleCompiledImageHandleV1,
        rank: u32,
    )
);

// SAFETY-REVIEWED: both arrays belong to the caller for this one callback; all
// elements are copied before the safe backend is invoked.
unsafe extern "C" fn upload_rank_arenas_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    uploads: *const BundleRankArenaUploadV1,
    count: u64,
    output: *mut BundleRankArenaHandleV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let count = checked_slice_count_v1::<BundleRankArenaUploadV1>("rank arena uploads", count)?;
        if count == 0 || count > 64 {
            return Err(BundleAbiError::InvalidRange("rank arena uploads"));
        }
        validate_input_pointer_v1(uploads, "rank arena uploads")?;
        validate_output_pointer_v1(output, "rank arena upload outputs")?;
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        // SAFETY-REVIEWED: the caller guarantees count readable records for this callback.
        let records = unsafe { slice::from_raw_parts(uploads, count) };
        let mut requests = Vec::with_capacity(count);
        for record in records {
            if record.reserved != 0 {
                return Err(BundleAbiError::InvalidReserved {
                    surface: "rank arena upload",
                    actual: record.reserved,
                });
            }
            requests.push(RankArenaUploadV1 {
                context: record.context,
                pool: record.pool,
                stream: record.stream,
                image: record.image,
                rank: record.rank,
            });
        }
        let arenas = match backend.upload_rank_arenas(&requests) {
            Ok(arenas) if arenas.len() == count => arenas,
            Ok(_) => return Ok(BundleStatusV1::HostError),
            Err(status) => return Ok(failed_host_status_v1(status)),
        };
        // SAFETY-REVIEWED: the caller guarantees count writable handles for this callback.
        unsafe { ptr::copy_nonoverlapping(arenas.as_ptr(), output, count) };
        Ok(BundleStatusV1::Success)
    })
}
output_service_trampoline_v1!(
    acquire_peer_access_trampoline_v1,
    acquire_peer_access,
    BundlePeerAccessHandleV1,
    (
        owner_context: BundleContextHandleV1,
        peer_context: BundleContextHandleV1,
    )
);
output_service_trampoline_v1!(
    create_stream_trampoline_v1,
    create_stream,
    BundleStreamHandleV1,
    (context: BundleContextHandleV1, stream_kind: u32)
);
output_service_trampoline_v1!(
    create_event_trampoline_v1,
    create_event,
    BundleEventHandleV1,
    (context: BundleContextHandleV1)
);
output_service_trampoline_v1!(
    create_kv_request_trampoline_v1,
    create_kv_request,
    BundleKvRequestHandleV1,
    (kv: BundlePhysicalKvHandleV1, quota_tokens: u64)
);
output_service_trampoline_v1!(
    resolve_state_view_trampoline_v1,
    resolve_state_view,
    BundleDeviceBufferViewV1,
    (state: BundleStateHandleV1, rank: u32)
);

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn log_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    message: BundleByteViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let context = context_from_user_data_v1::<B>(user_data)?;
        let result = with_byte_view_v1(message, |message| {
            if context.observation.active.load(Ordering::Acquire) {
                let mut diagnostic: String =
                    String::from_utf8_lossy(&message[..message.len().min(512)])
                        .chars()
                        .map(|ch| if ch.is_control() { ' ' } else { ch })
                        .collect();
                let mut end = diagnostic.len().min(512);
                while !diagnostic.is_char_boundary(end) {
                    end -= 1;
                }
                diagnostic.truncate(end);
                *context
                    .observation
                    .diagnostic
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(diagnostic);
            }
            context.backend.log(message)
        })?;
        Ok(host_unit_status_v1(result))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn set_memory_pool_peer_access_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    pool: BundleMemoryPoolHandleV1,
    peer_context: BundleContextHandleV1,
    enabled: u32,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        Ok(host_unit_status_v1(backend.set_memory_pool_peer_access(
            pool,
            peer_context,
            enabled != 0,
        )))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn open_compiled_image_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    component_role: BundleByteViewV1,
    out_image: *mut BundleCompiledImageHandleV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_image, "open_compiled_image output")?;
        let result = with_byte_view_v1(component_role, |component_role| {
            backend.open_compiled_image(component_role)
        })?;
        finish_output_v1(out_image, result)
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn open_authenticated_component_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    component_role: BundleByteViewV1,
    out_component: *mut BundleAuthenticatedComponentHandleV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_component, "open_authenticated_component output")?;
        let result = with_byte_view_v1(component_role, |component_role| {
            backend.open_authenticated_component(component_role)
        })?;
        finish_output_v1(out_component, result)
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and output for this callback.
unsafe extern "C" fn inspect_authenticated_component_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    component: BundleAuthenticatedComponentHandleV1,
    out_metadata: *mut BundleAuthenticatedComponentMetadataV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_metadata, "inspect_authenticated_component output")?;
        finish_output_v1(
            out_metadata,
            backend.inspect_authenticated_component(component),
        )
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and mutable views for this callback.
unsafe extern "C" fn read_authenticated_component_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    component: BundleAuthenticatedComponentHandleV1,
    role_destination: BundleMutByteViewV1,
    bytes_destination: BundleMutByteViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        let result = with_mut_byte_view_v1(role_destination, |role_destination| {
            with_mut_byte_view_v1(bytes_destination, |bytes_destination| {
                backend.read_authenticated_component(component, role_destination, bytes_destination)
            })
        })??;
        Ok(host_unit_status_v1(result))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn resolve_compiled_image_view_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    image: BundleCompiledImageHandleV1,
    arena: BundleRankArenaHandleV1,
    rank: u32,
    view_id: BundleByteViewV1,
    out_view: *mut BundleRawCompiledImageViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_view, "resolve_compiled_image_view output")?;
        let result = with_byte_view_v1(view_id, |view_id| {
            backend.resolve_compiled_image_view(image, arena, rank, view_id)
        })?;
        finish_output_v1(out_view, result)
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data, view and gather record for this callback.
unsafe extern "C" fn gather_mapped_rows_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    stream: BundleStreamHandleV1,
    destination: BundleDeviceBufferViewV1,
    gather: *const BundleMappedRowGatherV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        let destination = checked_device_buffer_span_v1(destination)?;
        with_input_record_v1(gather, "mapped row gather", |gather| {
            if gather.struct_size != struct_size_v1::<BundleMappedRowGatherV1>() {
                return Err(BundleAbiError::StructSizeExact {
                    surface: "mapped row gather",
                    expected: struct_size_v1::<BundleMappedRowGatherV1>(),
                    actual: gather.struct_size,
                });
            }
            let result =
                with_u32_array_v1(gather.rows, gather.row_count, "mapped rows", |rows| {
                    with_byte_view_v1(gather.view_id, |view_id| {
                        backend.gather_mapped_rows(
                            stream,
                            destination,
                            MappedRowGatherV1 {
                                image: gather.image,
                                rank: gather.rank,
                                view_id,
                                row_bytes: gather.row_bytes,
                                rows,
                            },
                        )
                    })
                })??;
            Ok(host_unit_status_v1(result))
        })
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn select_stage_stream_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    context: BundleContextHandleV1,
    rank: u32,
    stage_id: BundleByteViewV1,
    stream_kind: u32,
    out_stream: *mut BundleStreamHandleV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_stream, "select_stage_stream output")?;
        let result = with_byte_view_v1(stage_id, |stage_id| {
            backend.select_stage_stream(context, rank, stage_id, stream_kind)
        })?;
        finish_output_v1(out_stream, result)
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn copy_host_to_device_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    stream: BundleStreamHandleV1,
    destination: BundleDeviceBufferViewV1,
    source: BundleByteViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        let destination = checked_device_buffer_span_v1(destination)?;
        let result = with_byte_view_v1(source, |source| {
            backend.copy_host_to_device(stream, destination, source)
        })?;
        Ok(host_unit_status_v1(result))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn copy_device_to_host_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    stream: BundleStreamHandleV1,
    destination: BundleMutByteViewV1,
    source: BundleDeviceBufferViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        let source = checked_device_buffer_span_v1(source)?;
        let result = with_mut_byte_view_v1(destination, |destination| {
            backend.copy_device_to_host(stream, destination, source)
        })?;
        Ok(host_unit_status_v1(result))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn copy_device_to_device_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    stream: BundleStreamHandleV1,
    destination: BundleDeviceBufferViewV1,
    source: BundleDeviceBufferViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        let destination = checked_device_buffer_span_v1(destination)?;
        let source = checked_device_buffer_span_v1(source)?;
        Ok(host_unit_status_v1(backend.copy_device_to_device(
            stream,
            destination,
            source,
        )))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn copy_peer_to_peer_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    stream: BundleStreamHandleV1,
    destination_context: BundleContextHandleV1,
    destination: BundleDeviceBufferViewV1,
    source_context: BundleContextHandleV1,
    source: BundleDeviceBufferViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        let destination = checked_device_buffer_span_v1(destination)?;
        let source = checked_device_buffer_span_v1(source)?;
        Ok(host_unit_status_v1(backend.copy_peer_to_peer(
            stream,
            destination_context,
            destination,
            source_context,
            source,
        )))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn zero_device_buffer_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    stream: BundleStreamHandleV1,
    destination: BundleDeviceBufferViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        let destination = checked_device_buffer_span_v1(destination)?;
        Ok(host_unit_status_v1(
            backend.zero_device_buffer(stream, destination),
        ))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data, views, and table record for this callback.
unsafe extern "C" fn write_device_pointer_table_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    stream: BundleStreamHandleV1,
    destination: BundleDeviceBufferViewV1,
    table: *const BundleDevicePointerArrayV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        let destination = checked_device_buffer_span_v1(destination)?;
        let slots = with_input_record_v1(table, "device pointer table", |table| {
            owned_device_pointer_slots_v1(&table)
        })?;
        Ok(host_unit_status_v1(backend.write_device_pointer_table(
            stream,
            destination,
            &slots,
        )))
    })
}

/// Reads one validated by-value pointer array into the slots it names.
fn owned_device_pointer_slots_v1(
    array: &BundleDevicePointerArrayV1,
) -> Result<Vec<Option<DeviceBufferSpanV1>>, BundleAbiError> {
    validate_device_pointer_array_v1(*array)?;
    Ok(array.slots[..array.slot_count as usize]
        .iter()
        .map(|slot| {
            if slot.buffer.raw == 0 {
                None
            } else {
                Some(DeviceBufferSpanV1::from_view(*slot))
            }
        })
        .collect())
}

/// Builds one validated by-value pointer array from the slots a caller names.
fn raw_device_pointer_array_v1(
    slots: &[Option<DeviceBufferSpanV1>],
) -> Result<BundleDevicePointerArrayV1, BundleAbiError> {
    let slot_count = u32::try_from(slots.len())
        .map_err(|_| BundleAbiError::InvalidDevicePointerArraySlots(u32::MAX))?;
    if slot_count == 0 || slot_count > BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS {
        return Err(BundleAbiError::InvalidDevicePointerArraySlots(slot_count));
    }
    let mut array = BundleDevicePointerArrayV1::empty();
    array.slot_count = slot_count;
    for (position, slot) in slots.iter().enumerate() {
        if let Some(span) = slot {
            array.slots[position] = span.to_view();
        }
    }
    validate_device_pointer_array_v1(array)?;
    Ok(array)
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn load_cuda_module_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    context: BundleContextHandleV1,
    module_bytes: BundleByteViewV1,
    out_module: *mut BundleCudaModuleHandleV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_module, "load_cuda_module output")?;
        let result = with_byte_view_v1(module_bytes, |module_bytes| {
            backend.load_cuda_module(context, module_bytes)
        })?;
        finish_output_v1(out_module, result)
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn resolve_cuda_symbol_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    module: BundleCudaModuleHandleV1,
    symbol_name: BundleByteViewV1,
    out_symbol: *mut BundleCudaSymbolHandleV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_symbol, "resolve_cuda_symbol output")?;
        let result = with_byte_view_v1(symbol_name, |symbol_name| {
            backend.resolve_cuda_symbol(module, symbol_name)
        })?;
        finish_output_v1(out_symbol, result)
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn launch_cuda_symbol_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    stream: BundleStreamHandleV1,
    symbol: BundleCudaSymbolHandleV1,
    config: *const BundleLaunchConfigV1,
    arguments: *const BundleKernelArgumentV1,
    argument_count: u64,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        let config = owned_launch_config_v1(config)?;
        let arguments = owned_kernel_arguments_v1(arguments, argument_count)?;
        Ok(host_unit_status_v1(
            backend.launch_cuda_symbol(stream, symbol, config, &arguments),
        ))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn allocate_physical_kv_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    request: *const BundlePhysicalKvRequestV1,
    out_kv: *mut BundlePhysicalKvHandleV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_kv, "allocate_physical_kv output")?;
        let request = owned_physical_kv_request_v1(request)?;
        finish_output_v1(out_kv, backend.allocate_physical_kv(&request))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn prepare_kv_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    request: BundleKvRequestHandleV1,
    operation: BundleByteViewV1,
    out_transaction: *mut BundleKvTransactionHandleV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_transaction, "prepare_kv output")?;
        let result = with_byte_view_v1(operation, |operation| {
            backend.prepare_kv(request, operation)
        })?;
        finish_output_v1(out_transaction, result)
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn resolve_kv_view_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    request: BundleKvRequestHandleV1,
    rank: u32,
    view_id: BundleByteViewV1,
    out_view: *mut BundleDeviceBufferViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_view, "resolve_kv_view output")?;
        let result = with_byte_view_v1(view_id, |view_id| {
            backend.resolve_kv_view(request, rank, view_id)
        })?;
        finish_output_v1(out_view, result)
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and view for this callback.
unsafe extern "C" fn acquire_serving_kv_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    layout_id: BundleByteViewV1,
    out_page_capacity: *mut u64,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_page_capacity, "serving KV capacity output")?;
        let result = with_byte_view_v1(layout_id, |id| backend.acquire_serving_kv(id))?;
        finish_output_v1(out_page_capacity, result)
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and output for this callback.
unsafe extern "C" fn resolve_serving_kv_pool_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    rank: u32,
    out_view: *mut BundleDeviceBufferViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_view, "serving KV view output")?;
        finish_output_v1(out_view, backend.resolve_serving_kv_pool(rank))
    })
}

// SAFETY-REVIEWED: this callback validates its output pointer before writing one fixed record.
unsafe extern "C" fn query_speculation_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    out: *mut BundleRawSpeculationV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out, "speculation output")?;
        let config = backend
            .query_speculation()
            .map_err(BundleAbiError::Status)?;
        validate_speculation_v1(config)?;
        finish_output_v1(out, Ok(config))
    })
}

// SAFETY-REVIEWED: this callback validates its output pointer before writing one fixed record.
unsafe extern "C" fn query_batch_limits_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    out: *mut BundleRawBatchLimitsV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out, "batch limits output")?;
        let limits = backend
            .query_batch_limits()
            .map_err(BundleAbiError::Status)?;
        validate_batch_limits_v1(limits)?;
        finish_output_v1(out, Ok(limits))
    })
}

// SAFETY-REVIEWED: this callback copies its config and validates its output pointer before writing one fixed record.
unsafe extern "C" fn query_cluster_residency_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    symbol: BundleCudaSymbolHandleV1,
    config: *const BundleLaunchConfigV1,
    out: *mut BundleRawClusterResidencyV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out, "cluster residency output")?;
        let config = owned_launch_config_v1(config)?;
        let residency = backend
            .query_cluster_residency(symbol, config)
            .map_err(BundleAbiError::Status)?;
        validate_cluster_residency_v1(residency)?;
        finish_output_v1(out, Ok(residency))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn allocate_state_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    request: *const BundleStateRequestV1,
    out_state: *mut BundleStateHandleV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_state, "allocate_state output")?;
        let request = owned_state_request_v1(request)?;
        finish_output_v1(out_state, backend.allocate_state(&request))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn create_collective_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    request: *const BundleCollectiveRequestV1,
    out_collective: *mut BundleCollectiveHandleV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_collective, "create_collective output")?;
        let request = owned_collective_request_v1(request)?;
        finish_output_v1(out_collective, backend.create_collective(&request))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn execute_collective_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    collective: BundleCollectiveHandleV1,
    stream: BundleStreamHandleV1,
    operation: BundleByteViewV1,
    arguments: *const BundleKernelArgumentV1,
    argument_count: u64,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        let arguments = owned_kernel_arguments_v1(arguments, argument_count)?;
        let result = with_byte_view_v1(operation, |operation| {
            backend.execute_collective(collective, stream, operation, &arguments)
        })?;
        Ok(host_unit_status_v1(result))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn run_custom_operation_trampoline_v1<B: HostServicesBackendV1>(
    user_data: *mut c_void,
    context: BundleContextHandleV1,
    stream: BundleStreamHandleV1,
    flags: u32,
    operation_id: BundleByteViewV1,
    operands: *const BundleCustomOperandV1,
    operand_count: u64,
    operation: Option<BundleCustomOperationFnV1>,
    operation_user_data: *mut c_void,
    out_operation_status: *mut i32,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let backend = backend_from_user_data_v1::<B>(user_data)?;
        validate_output_pointer_v1(out_operation_status, "run_custom_operation output")?;
        // A completion policy this version cannot honour is refused before any
        // handle is resolved, because honouring it wrongly is the one mistake
        // no later check would catch.
        let completion = CustomCompletionV1::from_flags(flags)?;
        let entry = operation.ok_or(BundleAbiError::MissingCustomOperationCallback)?;
        // SAFETY-REVIEWED: the entry is the caller's own, and only the invoke below calls it.
        let operation = unsafe { CustomOperationV1::new(entry, operation_user_data) };
        let operands = owned_custom_operands_v1(operands, operand_count)?;
        let result = with_byte_view_v1(operation_id, |operation_id| {
            backend.run_custom_operation(
                context,
                stream,
                completion,
                operation_id,
                &operands,
                &operation,
            )
        })?;
        match result {
            // The callback ran, so its own code goes back whatever it was, and
            // a nonzero one is the bundle's failure rather than the host's.
            Ok(code) => {
                write_output_v1(out_operation_status, code)?;
                Ok(if code == 0 {
                    BundleStatusV1::Success
                } else {
                    BundleStatusV1::BundleError
                })
            }
            Err(status) => Ok(failed_host_status_v1(status)),
        }
    })
}

/// One bundle-supplied custom operation, callable for the length of one call.
///
/// The value is a native entry and the state it was registered with. The
/// lifetime is that state's: an operation cannot outlive the storage its entry
/// will be handed, and invoking it is the only thing that ever happens to it.
pub struct CustomOperationV1<'state> {
    entry: BundleCustomOperationFnV1,
    user_data: *mut c_void,
    state: PhantomData<&'state mut c_void>,
}

impl CustomOperationV1<'_> {
    /// Captures one native custom-operation entry and its own `user_data`.
    ///
    /// # Safety
    ///
    /// `entry` must honour the V1 custom-operation contract for `user_data`:
    /// it reads only the record it is given, retains nothing after it returns,
    /// and lets no exception cross the boundary. `user_data` must point at live
    /// storage of the type that entry reads, for the whole captured lifetime.
    // SAFETY-REVIEWED: capturing a caller-owned native entry is the documented unsafe step.
    pub const unsafe fn new(entry: BundleCustomOperationFnV1, user_data: *mut c_void) -> Self {
        Self {
            entry,
            user_data,
            state: PhantomData,
        }
    }

    pub(crate) const fn entry(&self) -> BundleCustomOperationFnV1 {
        self.entry
    }

    pub(crate) const fn user_data(&self) -> *mut c_void {
        self.user_data
    }

    /// Invokes the operation with the raw values one host lends it.
    ///
    /// The record and both arrays live in this frame and die with the call, so
    /// an operation that retained either has nothing afterwards. A Rust panic
    /// raised inside the callback is caught here rather than unwinding into
    /// native code, and it is reported as a panic rather than as a code.
    pub fn invoke(&self, launch: LentLaunchV1<'_>) -> HostServiceResultV1<i32> {
        let addresses = launch
            .ranges
            .iter()
            .map(|range| range.address)
            .collect::<Vec<u64>>();
        let lengths = launch
            .ranges
            .iter()
            .map(|range| range.length)
            .collect::<Vec<u64>>();
        let count = match u64::try_from(launch.ranges.len()) {
            Ok(count) => count,
            Err(_) => return Err(BundleStatusV1::InvalidArgument),
        };
        let record = BundleCustomLaunchV1 {
            struct_size: struct_size_v1::<BundleCustomLaunchV1>(),
            address_space: launch.address_space.code(),
            operand_count: count,
            stream: launch.stream.raw(),
            // Both arrays are null exactly when the count is zero, which is the
            // same pointer-length rule every other V1 array carries.
            operand_addresses: if count == 0 {
                ptr::null()
            } else {
                addresses.as_ptr()
            },
            operand_lengths: if count == 0 {
                ptr::null()
            } else {
                lengths.as_ptr()
            },
        };
        let entry = self.entry;
        let user_data = self.user_data;
        catch_unwind(AssertUnwindSafe(|| {
            // SAFETY-REVIEWED: the record and both arrays outlive this call and the entry is the bundle's own.
            unsafe { entry(user_data, &raw const record) }
        }))
        .map_err(|_| BundleStatusV1::Panic)
    }
}

fn owned_custom_operands_v1(
    operands: *const BundleCustomOperandV1,
    operand_count: u64,
) -> Result<Vec<CustomOperandV1>, BundleAbiError> {
    with_custom_operands_v1(operands, operand_count, |operands| {
        operands
            .iter()
            .map(|operand| {
                CustomOperandV1::from_flags(
                    DeviceBufferSpanV1::from_view(operand.buffer),
                    operand.flags,
                )
            })
            .collect::<Vec<CustomOperandV1>>()
    })
}

/// Exposes validated custom operands only while `callback` runs.
fn with_custom_operands_v1<T>(
    operands: *const BundleCustomOperandV1,
    operand_count: u64,
    callback: impl FnOnce(&[BundleCustomOperandV1]) -> T,
) -> Result<T, BundleAbiError> {
    let count =
        checked_slice_count_v1::<BundleCustomOperandV1>("custom operand array", operand_count)?;
    if count == 0 {
        if !operands.is_null() {
            return Err(BundleAbiError::InvalidPointerLength("custom operand array"));
        }
        return Ok(callback(&[]));
    }
    validate_input_pointer_v1(operands, "custom operand array")?;
    // The array has no stride of its own, so the first record's declared length
    // is what says whether the rest are where this reader will look for them.
    // It is read on its own, out of the four bytes every version of this record
    // opens with, before anything advances by one V1 record.
    // SAFETY-REVIEWED: a nonnull aligned operand array has at least one record, whose leading struct_size this reads and nothing else.
    validate_custom_operand_struct_size_v1(unsafe { ptr::read(operands.cast::<u32>()) })?;
    // SAFETY-REVIEWED: the ABI callback guarantees a readable operand array for its duration.
    let copied = unsafe { slice::from_raw_parts(operands, count) }.to_vec();
    validate_custom_operands_v1(&copied)?;
    Ok(callback(&copied))
}

fn checked_device_buffer_span_v1(
    view: BundleDeviceBufferViewV1,
) -> Result<DeviceBufferSpanV1, BundleAbiError> {
    validate_device_buffer_view_v1(view)?;
    Ok(DeviceBufferSpanV1::from_view(view))
}

fn owned_launch_config_v1(
    config: *const BundleLaunchConfigV1,
) -> Result<LaunchConfigV1, BundleAbiError> {
    with_input_record_v1(config, "launch config", |config| {
        validate_launch_config_v1(config)?;
        Ok(LaunchConfigV1::from_config(config))
    })
}

fn owned_kernel_arguments_v1(
    arguments: *const BundleKernelArgumentV1,
    argument_count: u64,
) -> Result<Vec<KernelArgumentV1>, BundleAbiError> {
    with_kernel_arguments_v1(arguments, argument_count, |arguments| {
        arguments
            .iter()
            .map(|argument| owned_kernel_argument_v1(*argument))
            .collect::<Result<Vec<KernelArgumentV1>, BundleAbiError>>()
    })?
}

/// Reads one validated kernel argument, and the payload its kind names.
///
/// [`validate_kernel_argument_v1`] has already settled the record's length, its
/// kind, and that the payload this kind selects is the only one present, so the
/// pointer read here is nonnull for exactly the kinds that read one. What the
/// pointer leads to is validated after it is read, exactly as it was when this
/// record carried the payload by value.
fn owned_kernel_argument_v1(
    argument: BundleKernelArgumentV1,
) -> Result<KernelArgumentV1, BundleAbiError> {
    match argument.kind {
        BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_BUFFER => Ok(KernelArgumentV1::DeviceBuffer(
            DeviceBufferSpanV1::from_view(argument.buffer),
        )),
        BUNDLE_KERNEL_ARGUMENT_V1_U64 => Ok(KernelArgumentV1::ScalarU64(argument.scalar_u64)),
        BUNDLE_KERNEL_ARGUMENT_V1_U32 => Ok(KernelArgumentV1::ScalarU32(argument.scalar_u32)),
        BUNDLE_KERNEL_ARGUMENT_V1_NULL_DEVICE_POINTER => Ok(KernelArgumentV1::NullDevicePointer),
        BUNDLE_KERNEL_ARGUMENT_V1_OPAQUE_BYTES => Ok(KernelArgumentV1::OpaqueBytes(
            copy_byte_view_v1(argument.opaque)?,
        )),
        BUNDLE_KERNEL_ARGUMENT_V1_I32 => Ok(KernelArgumentV1::ScalarI32(argument.scalar_i32)),
        BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP => {
            with_input_record_v1(argument.tensor_map, "tensor map", |map| {
                validate_tensor_map_v1(map)?;
                Ok(KernelArgumentV1::TensorMap(TensorMapV1::from_raw(map)?))
            })
        }
        BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY => {
            with_input_record_v1(argument.pointer_array, "device pointer array", |array| {
                Ok(KernelArgumentV1::DevicePointerArray(
                    owned_device_pointer_slots_v1(&array)?,
                ))
            })
        }
        kind => Err(BundleAbiError::InvalidKernelArgumentKind(kind)),
    }
}

fn owned_physical_kv_request_v1(
    request: *const BundlePhysicalKvRequestV1,
) -> Result<PhysicalKvRequestV1, BundleAbiError> {
    with_input_record_v1(request, "physical KV request", |request| {
        validate_physical_kv_request_v1(request)?;
        Ok(PhysicalKvRequestV1 {
            flags: request.flags,
            layout: PhysicalKvLayoutV1 {
                flags: request.layout.flags,
                rank_count: request.layout.rank_count,
                page_bytes: request.layout.page_bytes,
                page_alignment: request.layout.page_alignment,
                page_tokens: request.layout.page_tokens,
                state_lifetime: request.layout.state_lifetime,
                layout_id: copy_byte_view_v1(request.layout.layout_id)?,
            },
            page_capacity: request.page_capacity,
            request_capacity: request.request_capacity,
        })
    })
}

fn owned_state_request_v1(
    request: *const BundleStateRequestV1,
) -> Result<StateRequestV1, BundleAbiError> {
    with_input_record_v1(request, "state request", |request| {
        validate_state_request_v1(request)?;
        Ok(StateRequestV1 {
            flags: request.flags,
            byte_length: request.byte_length,
            alignment: request.alignment,
            state_id: copy_byte_view_v1(request.state_id)?,
        })
    })
}

fn owned_collective_request_v1(
    request: *const BundleCollectiveRequestV1,
) -> Result<CollectiveRequestV1, BundleAbiError> {
    with_input_record_v1(request, "collective request", |request| {
        validate_collective_request_v1(request)?;
        Ok(CollectiveRequestV1 {
            flags: request.flags,
            rank_count: request.rank_count,
            semantic_contract: copy_byte_view_v1(request.semantic_contract)?,
        })
    })
}

/// A callable host-service table checked for V1 version, size, and completeness.
///
/// A generated create trampoline hands one to the bundle it dispatches, and the
/// service methods on it are safe: they build every view and output record from
/// Rust values the caller already owns. Capturing the raw table is the unsafe
/// step, because only the ABI contract can promise it outlives this value.
pub struct HostServicesV1 {
    raw: BundleHostServicesV1,
}

impl HostServicesV1 {
    /// Uploads all rank arenas together when supported by the host.
    pub fn upload_rank_arenas(
        &self,
        uploads: &[RankArenaUploadV1],
    ) -> Result<Vec<BundleRankArenaHandleV1>, BundleAbiError> {
        if uploads.is_empty() || uploads.len() > 64 {
            return Err(BundleAbiError::InvalidRange("rank arena uploads"));
        }
        let available = (self.raw.struct_size as usize)
            >= std::mem::offset_of!(BundleHostServicesV1, upload_rank_arenas)
                + size_of::<Option<crate::BundleUploadRankArenasFnV1>>();
        if available && let Some(service) = self.raw.upload_rank_arenas {
            let inputs: Vec<_> = uploads
                .iter()
                .map(|upload| BundleRankArenaUploadV1 {
                    context: upload.context,
                    pool: upload.pool,
                    stream: upload.stream,
                    image: upload.image,
                    rank: upload.rank,
                    reserved: 0,
                })
                .collect();
            let mut arenas = vec![BundleRankArenaHandleV1::from_raw(0); uploads.len()];
            // SAFETY-REVIEWED: both arrays remain live for the captured callback and have matching lengths.
            status_result(unsafe {
                service(
                    self.raw.user_data,
                    inputs.as_ptr(),
                    inputs.len() as u64,
                    arenas.as_mut_ptr(),
                )
            })?;
            for arena in &arenas {
                arena.validate()?;
            }
            return Ok(arenas);
        }
        let mut arenas = Vec::with_capacity(uploads.len());
        for upload in uploads {
            match self.upload_rank_arena(
                upload.context,
                upload.pool,
                upload.stream,
                upload.image,
                upload.rank,
            ) {
                Ok(arena) => arenas.push(arena),
                Err(error) => {
                    for arena in arenas.into_iter().rev() {
                        let _ = self.release_rank_arena(arena);
                    }
                    return Err(error);
                }
            }
        }
        Ok(arenas)
    }
    /// Checks a raw V1 table and captures it for safe service calls.
    ///
    /// # Safety
    ///
    /// The raw table, every callback in it, and its `user_data` must stay valid
    /// for the lifetime of the returned value, and each callback must honour
    /// the V1 contract for the arguments this crate builds.
    // SAFETY-REVIEWED: capturing a caller-owned service table is the documented unsafe step.
    pub unsafe fn new(raw: BundleHostServicesV1) -> Result<Self, BundleAbiError> {
        validate_host_services_v1(&raw)?;
        Ok(Self { raw })
    }
}

fn required_service_v1<T>(service: Option<T>, name: &'static str) -> Result<T, BundleAbiError> {
    service.ok_or(BundleAbiError::MissingHostService(name))
}

macro_rules! status_service_call_v1 {
    ($method:ident, ($($argument:ident: $type:ty),* $(,)?)) => {
        pub fn $method(&self, $($argument: $type),*) -> Result<(), BundleAbiError> {
            let service = required_service_v1(self.raw.$method, stringify!($method))?;
            // SAFETY-REVIEWED: the captured table stays live and this call passes only scalars and handles.
            status_result(unsafe { service(self.raw.user_data, $($argument),*) })
        }
    };
}

macro_rules! output_service_call_v1 {
    ($method:ident, $output:ty, ($($argument:ident: $type:ty),* $(,)?)) => {
        pub fn $method(&self, $($argument: $type),*) -> Result<$output, BundleAbiError> {
            let service = required_service_v1(self.raw.$method, stringify!($method))?;
            let mut output = <$output>::from_raw(0);
            // SAFETY-REVIEWED: the captured table stays live and the output is this frame's own storage.
            status_result(unsafe { service(self.raw.user_data, $($argument,)* &mut output) })?;
            output.validate()?;
            Ok(output)
        }
    };
}

/// The safe service calls a bundle makes through one checked host table.
///
/// Every method builds the C views, records, and output storage from values the
/// caller already owns, makes the raw call here inside the reviewed boundary,
/// then validates the returned handle or span. A bundle never holds an ABI
/// pointer and never writes unsafe code.
impl HostServicesV1 {
    pub fn log(&self, message: &[u8]) -> Result<(), BundleAbiError> {
        let service = required_service_v1(self.raw.log, "log")?;
        // SAFETY-REVIEWED: the captured table stays live and the view borrows the caller's message.
        status_result(unsafe { service(self.raw.user_data, BundleByteViewV1::from_bytes(message)) })
    }

    pub fn set_memory_pool_peer_access(
        &self,
        pool: BundleMemoryPoolHandleV1,
        peer_context: BundleContextHandleV1,
        enabled: bool,
    ) -> Result<(), BundleAbiError> {
        let service = required_service_v1(
            self.raw.set_memory_pool_peer_access,
            "set_memory_pool_peer_access",
        )?;
        // SAFETY-REVIEWED: the captured table stays live and this call passes only scalars and handles.
        status_result(unsafe {
            service(self.raw.user_data, pool, peer_context, u32::from(enabled))
        })
    }

    pub fn open_compiled_image(
        &self,
        component_role: &[u8],
    ) -> Result<BundleCompiledImageHandleV1, BundleAbiError> {
        let service = required_service_v1(self.raw.open_compiled_image, "open_compiled_image")?;
        let mut output = BundleCompiledImageHandleV1::from_raw(0);
        // SAFETY-REVIEWED: the view borrows the caller's role bytes and the output is this frame's storage.
        status_result(unsafe {
            service(
                self.raw.user_data,
                BundleByteViewV1::from_bytes(component_role),
                &mut output,
            )
        })?;
        output.validate()?;
        Ok(output)
    }

    pub fn open_authenticated_component(
        &self,
        component_role: &[u8],
    ) -> Result<BundleAuthenticatedComponentHandleV1, BundleAbiError> {
        let service = required_service_v1(
            self.raw.open_authenticated_component,
            "open_authenticated_component",
        )?;
        let mut output = BundleAuthenticatedComponentHandleV1::from_raw(0);
        // SAFETY-REVIEWED: the role view and output storage live for this call.
        status_result(unsafe {
            service(
                self.raw.user_data,
                BundleByteViewV1::from_bytes(component_role),
                &mut output,
            )
        })?;
        output.validate()?;
        Ok(output)
    }

    pub fn inspect_authenticated_component(
        &self,
        component: BundleAuthenticatedComponentHandleV1,
    ) -> Result<AuthenticatedComponentMetadataV1, BundleAbiError> {
        let service = required_service_v1(
            self.raw.inspect_authenticated_component,
            "inspect_authenticated_component",
        )?;
        let mut output = BundleAuthenticatedComponentMetadataV1::empty();
        // SAFETY-REVIEWED: the output storage lives for this call.
        status_result(unsafe { service(self.raw.user_data, component, &mut output) })?;
        validate_authenticated_component_metadata_v1(output)?;
        Ok(AuthenticatedComponentMetadataV1::new(
            output.role_length,
            output.byte_length,
            output.sha256,
        ))
    }

    pub fn read_authenticated_component(
        &self,
        component: BundleAuthenticatedComponentHandleV1,
        role_destination: &mut [u8],
        bytes_destination: &mut [u8],
    ) -> Result<(), BundleAbiError> {
        let service = required_service_v1(
            self.raw.read_authenticated_component,
            "read_authenticated_component",
        )?;
        // SAFETY-REVIEWED: both exclusive destinations live for this call.
        status_result(unsafe {
            service(
                self.raw.user_data,
                component,
                BundleMutByteViewV1::from_bytes(role_destination),
                BundleMutByteViewV1::from_bytes(bytes_destination),
            )
        })
    }

    pub fn resolve_compiled_image_view(
        &self,
        image: BundleCompiledImageHandleV1,
        arena: BundleRankArenaHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> Result<CompiledImageSpanV1, BundleAbiError> {
        let service = required_service_v1(
            self.raw.resolve_compiled_image_view,
            "resolve_compiled_image_view",
        )?;
        let mut output = BundleRawCompiledImageViewV1::empty();
        // SAFETY-REVIEWED: the view borrows the caller's identifier and the output is this frame's storage.
        status_result(unsafe {
            service(
                self.raw.user_data,
                image,
                arena,
                rank,
                BundleByteViewV1::from_bytes(view_id),
                &mut output,
            )
        })?;
        checked_compiled_image_span_v1(output)
    }

    pub fn select_stage_stream(
        &self,
        context: BundleContextHandleV1,
        rank: u32,
        stage_id: &[u8],
        stream_kind: u32,
    ) -> Result<BundleStreamHandleV1, BundleAbiError> {
        let service = required_service_v1(self.raw.select_stage_stream, "select_stage_stream")?;
        let mut output = BundleStreamHandleV1::from_raw(0);
        // SAFETY-REVIEWED: the view borrows the caller's stage identifier and the output is this frame's storage.
        status_result(unsafe {
            service(
                self.raw.user_data,
                context,
                rank,
                BundleByteViewV1::from_bytes(stage_id),
                stream_kind,
                &mut output,
            )
        })?;
        output.validate()?;
        Ok(output)
    }

    pub fn copy_host_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> Result<(), BundleAbiError> {
        let service = required_service_v1(self.raw.copy_host_to_device, "copy_host_to_device")?;
        let destination = checked_device_buffer_view_v1(destination)?;
        // SAFETY-REVIEWED: the view borrows the caller's source bytes for the length of this call.
        status_result(unsafe {
            service(
                self.raw.user_data,
                stream,
                destination,
                BundleByteViewV1::from_bytes(source),
            )
        })
    }

    pub fn copy_device_to_host(
        &self,
        stream: BundleStreamHandleV1,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> Result<(), BundleAbiError> {
        let service = required_service_v1(self.raw.copy_device_to_host, "copy_device_to_host")?;
        let source = checked_device_buffer_view_v1(source)?;
        // SAFETY-REVIEWED: the view borrows the caller's exclusive destination for the length of this call.
        status_result(unsafe {
            service(
                self.raw.user_data,
                stream,
                BundleMutByteViewV1::from_bytes(destination),
                source,
            )
        })
    }

    pub fn copy_device_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> Result<(), BundleAbiError> {
        let service = required_service_v1(self.raw.copy_device_to_device, "copy_device_to_device")?;
        let destination = checked_device_buffer_view_v1(destination)?;
        let source = checked_device_buffer_view_v1(source)?;
        // SAFETY-REVIEWED: the captured table stays live and this call passes only pointer-free records.
        status_result(unsafe { service(self.raw.user_data, stream, destination, source) })
    }

    pub fn copy_peer_to_peer(
        &self,
        stream: BundleStreamHandleV1,
        destination_context: BundleContextHandleV1,
        destination: DeviceBufferSpanV1,
        source_context: BundleContextHandleV1,
        source: DeviceBufferSpanV1,
    ) -> Result<(), BundleAbiError> {
        let service = required_service_v1(self.raw.copy_peer_to_peer, "copy_peer_to_peer")?;
        let destination = checked_device_buffer_view_v1(destination)?;
        let source = checked_device_buffer_view_v1(source)?;
        // SAFETY-REVIEWED: the captured table stays live and this call passes only pointer-free records.
        status_result(unsafe {
            service(
                self.raw.user_data,
                stream,
                destination_context,
                destination,
                source_context,
                source,
            )
        })
    }

    pub fn zero_device_buffer(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
    ) -> Result<(), BundleAbiError> {
        let service = required_service_v1(self.raw.zero_device_buffer, "zero_device_buffer")?;
        let destination = checked_device_buffer_view_v1(destination)?;
        // SAFETY-REVIEWED: the captured table stays live and this call passes only pointer-free records.
        status_result(unsafe { service(self.raw.user_data, stream, destination) })
    }

    pub fn write_device_pointer_table(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        slots: &[Option<DeviceBufferSpanV1>],
    ) -> Result<(), BundleAbiError> {
        let service = required_service_v1(
            self.raw.write_device_pointer_table,
            "write_device_pointer_table",
        )?;
        let destination = checked_device_buffer_view_v1(destination)?;
        let table = raw_device_pointer_array_v1(slots)?;
        // SAFETY-REVIEWED: the table record lives in this frame for the call.
        status_result(unsafe { service(self.raw.user_data, stream, destination, &raw const table) })
    }

    pub fn load_cuda_module(
        &self,
        context: BundleContextHandleV1,
        module_bytes: &[u8],
    ) -> Result<BundleCudaModuleHandleV1, BundleAbiError> {
        let service = required_service_v1(self.raw.load_cuda_module, "load_cuda_module")?;
        let mut output = BundleCudaModuleHandleV1::from_raw(0);
        // SAFETY-REVIEWED: the view borrows the caller's module bytes and the output is this frame's storage.
        status_result(unsafe {
            service(
                self.raw.user_data,
                context,
                BundleByteViewV1::from_bytes(module_bytes),
                &mut output,
            )
        })?;
        output.validate()?;
        Ok(output)
    }

    pub fn resolve_cuda_symbol(
        &self,
        module: BundleCudaModuleHandleV1,
        symbol_name: &[u8],
    ) -> Result<BundleCudaSymbolHandleV1, BundleAbiError> {
        let service = required_service_v1(self.raw.resolve_cuda_symbol, "resolve_cuda_symbol")?;
        let mut output = BundleCudaSymbolHandleV1::from_raw(0);
        // SAFETY-REVIEWED: the view borrows the caller's symbol name and the output is this frame's storage.
        status_result(unsafe {
            service(
                self.raw.user_data,
                module,
                BundleByteViewV1::from_bytes(symbol_name),
                &mut output,
            )
        })?;
        output.validate()?;
        Ok(output)
    }

    pub fn launch_cuda_symbol(
        &self,
        stream: BundleStreamHandleV1,
        symbol: BundleCudaSymbolHandleV1,
        config: LaunchConfigV1,
        arguments: &[KernelArgumentV1],
    ) -> Result<(), BundleAbiError> {
        let service = required_service_v1(self.raw.launch_cuda_symbol, "launch_cuda_symbol")?;
        let config = config.to_config();
        validate_launch_config_v1(config)?;
        let arguments = RawKernelArgumentsV1::new(arguments)?;
        let (pointer, count) = kernel_argument_array_v1(&arguments.arguments)?;
        // SAFETY-REVIEWED: the config, the argument array, and every payload it names live in this frame.
        status_result(unsafe {
            service(self.raw.user_data, stream, symbol, &config, pointer, count)
        })
    }

    pub fn allocate_physical_kv(
        &self,
        request: &PhysicalKvRequestV1,
    ) -> Result<BundlePhysicalKvHandleV1, BundleAbiError> {
        let service = required_service_v1(self.raw.allocate_physical_kv, "allocate_physical_kv")?;
        let request = raw_physical_kv_request_v1(request)?;
        let mut output = BundlePhysicalKvHandleV1::from_raw(0);
        // SAFETY-REVIEWED: the record lives in this frame and borrows the caller's identifier bytes.
        status_result(unsafe { service(self.raw.user_data, &request, &mut output) })?;
        output.validate()?;
        Ok(output)
    }

    pub fn acquire_serving_kv(&self, layout_id: &[u8]) -> Result<u64, BundleAbiError> {
        if (self.raw.struct_size as usize)
            < std::mem::offset_of!(BundleHostServicesV1, acquire_serving_kv)
                + std::mem::size_of_val(&self.raw.acquire_serving_kv)
        {
            return Err(BundleAbiError::MissingHostService("acquire_serving_kv"));
        }
        let service = required_service_v1(self.raw.acquire_serving_kv, "acquire_serving_kv")?;
        let mut capacity = 0;
        // SAFETY-REVIEWED: the view borrows the identifier and the output lives in this frame.
        status_result(unsafe {
            service(
                self.raw.user_data,
                BundleByteViewV1::from_bytes(layout_id),
                &mut capacity,
            )
        })?;
        if capacity == 0 {
            return Err(BundleAbiError::InvalidRange("serving KV capacity"));
        }
        Ok(capacity)
    }

    pub fn resolve_serving_kv_pool(&self, rank: u32) -> Result<DeviceBufferSpanV1, BundleAbiError> {
        if (self.raw.struct_size as usize)
            < std::mem::offset_of!(BundleHostServicesV1, resolve_serving_kv_pool)
                + std::mem::size_of_val(&self.raw.resolve_serving_kv_pool)
        {
            return Err(BundleAbiError::MissingHostService(
                "resolve_serving_kv_pool",
            ));
        }
        let service =
            required_service_v1(self.raw.resolve_serving_kv_pool, "resolve_serving_kv_pool")?;
        let mut output = BundleDeviceBufferViewV1::empty();
        // SAFETY-REVIEWED: the output lives in this frame and the service is synchronous.
        status_result(unsafe { service(self.raw.user_data, rank, &mut output) })?;
        checked_device_buffer_span_v1(output)
    }

    pub fn query_speculation(&self) -> Result<BundleRawSpeculationV1, BundleAbiError> {
        let end = std::mem::offset_of!(BundleHostServicesV1, query_speculation_v1)
            + std::mem::size_of_val(&self.raw.query_speculation_v1);
        if (self.raw.struct_size as usize) < end {
            return Ok(BundleRawSpeculationV1::disabled());
        }
        let Some(service) = self.raw.query_speculation_v1 else {
            return Ok(BundleRawSpeculationV1::disabled());
        };
        let mut result = BundleRawSpeculationV1::disabled();
        // SAFETY-REVIEWED: the returned fixed record is this frame's own storage.
        status_result(unsafe { service(self.raw.user_data, &mut result) })?;
        validate_speculation_v1(result)?;
        Ok(result)
    }

    /// The batch limits the host schedules; unreported by an older host.
    pub fn query_batch_limits(&self) -> Result<BundleRawBatchLimitsV1, BundleAbiError> {
        let end = std::mem::offset_of!(BundleHostServicesV1, query_batch_limits_v1)
            + std::mem::size_of_val(&self.raw.query_batch_limits_v1);
        if (self.raw.struct_size as usize) < end {
            return Ok(BundleRawBatchLimitsV1::unreported());
        }
        let Some(service) = self.raw.query_batch_limits_v1 else {
            return Ok(BundleRawBatchLimitsV1::unreported());
        };
        let mut result = BundleRawBatchLimitsV1::unreported();
        // SAFETY-REVIEWED: the returned fixed record is this frame's own storage.
        status_result(unsafe { service(self.raw.user_data, &mut result) })?;
        validate_batch_limits_v1(result)?;
        Ok(result)
    }

    /// What the host's driver reports about keeping `symbol`'s clusters
    /// resident at `config`'s shape; unreported by an older host.
    pub fn query_cluster_residency(
        &self,
        symbol: BundleCudaSymbolHandleV1,
        config: LaunchConfigV1,
    ) -> Result<BundleRawClusterResidencyV1, BundleAbiError> {
        let end = std::mem::offset_of!(BundleHostServicesV1, query_cluster_residency_v1)
            + std::mem::size_of_val(&self.raw.query_cluster_residency_v1);
        if (self.raw.struct_size as usize) < end {
            return Ok(BundleRawClusterResidencyV1::unreported());
        }
        let Some(service) = self.raw.query_cluster_residency_v1 else {
            return Ok(BundleRawClusterResidencyV1::unreported());
        };
        let config = config.to_config();
        validate_launch_config_v1(config)?;
        let mut result = BundleRawClusterResidencyV1::unreported();
        // SAFETY-REVIEWED: the config and the fixed output record are this frame's own storage.
        status_result(unsafe { service(self.raw.user_data, symbol, &config, &mut result) })?;
        validate_cluster_residency_v1(result)?;
        Ok(result)
    }

    /// Whether the host fills the host-mapped gather tail
    /// (`chop.host-mapped-gather.v1`).
    #[must_use]
    pub fn has_host_mapped_gather(&self) -> bool {
        let end = std::mem::offset_of!(BundleHostServicesV1, gather_mapped_rows_v1)
            + std::mem::size_of_val(&self.raw.gather_mapped_rows_v1);
        (self.raw.struct_size as usize) >= end && self.raw.gather_mapped_rows_v1.is_some()
    }

    /// Copies rows of one host-mapped view of `gather.image` for
    /// `gather.rank` into `destination`, ordered on `stream`
    /// (`chop.host-mapped-gather.v1`): row `i` of `destination` is row
    /// `gather.rows[i]` of the view. The host has read the rows when this
    /// returns, so `gather.rows` may be reused at once.
    pub fn gather_mapped_rows(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        gather: MappedRowGatherV1<'_>,
    ) -> Result<(), BundleAbiError> {
        if !self.has_host_mapped_gather() {
            return Err(BundleAbiError::MissingHostService("gather_mapped_rows_v1"));
        }
        let service = required_service_v1(self.raw.gather_mapped_rows_v1, "gather_mapped_rows_v1")?;
        let destination = checked_device_buffer_view_v1(destination)?;
        let record = BundleMappedRowGatherV1 {
            struct_size: struct_size_v1::<BundleMappedRowGatherV1>(),
            rank: gather.rank,
            image: gather.image,
            view_id: BundleByteViewV1::from_bytes(gather.view_id),
            row_bytes: gather.row_bytes,
            rows: if gather.rows.is_empty() {
                ptr::null()
            } else {
                gather.rows.as_ptr()
            },
            row_count: gather.rows.len() as u64,
        };
        // SAFETY-REVIEWED: the record borrows the caller's identifier and rows for the length of this call.
        status_result(unsafe { service(self.raw.user_data, stream, destination, &record) })
    }

    pub fn prepare_kv(
        &self,
        request: BundleKvRequestHandleV1,
        operation: &[u8],
    ) -> Result<BundleKvTransactionHandleV1, BundleAbiError> {
        let service = required_service_v1(self.raw.prepare_kv, "prepare_kv")?;
        let mut output = BundleKvTransactionHandleV1::from_raw(0);
        // SAFETY-REVIEWED: the view borrows the caller's operation bytes and the output is this frame's storage.
        status_result(unsafe {
            service(
                self.raw.user_data,
                request,
                BundleByteViewV1::from_bytes(operation),
                &mut output,
            )
        })?;
        output.validate()?;
        Ok(output)
    }

    pub fn resolve_kv_view(
        &self,
        request: BundleKvRequestHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> Result<DeviceBufferSpanV1, BundleAbiError> {
        let service = required_service_v1(self.raw.resolve_kv_view, "resolve_kv_view")?;
        let mut output = BundleDeviceBufferViewV1::empty();
        // SAFETY-REVIEWED: the view borrows the caller's identifier and the output is this frame's storage.
        status_result(unsafe {
            service(
                self.raw.user_data,
                request,
                rank,
                BundleByteViewV1::from_bytes(view_id),
                &mut output,
            )
        })?;
        checked_device_buffer_span_v1(output)
    }

    pub fn allocate_state(
        &self,
        request: &StateRequestV1,
    ) -> Result<BundleStateHandleV1, BundleAbiError> {
        let service = required_service_v1(self.raw.allocate_state, "allocate_state")?;
        let request = raw_state_request_v1(request)?;
        let mut output = BundleStateHandleV1::from_raw(0);
        // SAFETY-REVIEWED: the record lives in this frame and borrows the caller's identifier bytes.
        status_result(unsafe { service(self.raw.user_data, &request, &mut output) })?;
        output.validate()?;
        Ok(output)
    }

    pub fn resolve_state_view(
        &self,
        state: BundleStateHandleV1,
        rank: u32,
    ) -> Result<DeviceBufferSpanV1, BundleAbiError> {
        let service = required_service_v1(self.raw.resolve_state_view, "resolve_state_view")?;
        let mut output = BundleDeviceBufferViewV1::empty();
        // SAFETY-REVIEWED: the captured table stays live and the output is this frame's own storage.
        status_result(unsafe { service(self.raw.user_data, state, rank, &mut output) })?;
        checked_device_buffer_span_v1(output)
    }

    pub fn create_collective(
        &self,
        request: &CollectiveRequestV1,
    ) -> Result<BundleCollectiveHandleV1, BundleAbiError> {
        let service = required_service_v1(self.raw.create_collective, "create_collective")?;
        let request = raw_collective_request_v1(request)?;
        let mut output = BundleCollectiveHandleV1::from_raw(0);
        // SAFETY-REVIEWED: the record lives in this frame and borrows the caller's contract bytes.
        status_result(unsafe { service(self.raw.user_data, &request, &mut output) })?;
        output.validate()?;
        Ok(output)
    }

    pub fn execute_collective(
        &self,
        collective: BundleCollectiveHandleV1,
        stream: BundleStreamHandleV1,
        operation: &[u8],
        arguments: &[KernelArgumentV1],
    ) -> Result<(), BundleAbiError> {
        let service = required_service_v1(self.raw.execute_collective, "execute_collective")?;
        let arguments = RawKernelArgumentsV1::new(arguments)?;
        let (pointer, count) = kernel_argument_array_v1(&arguments.arguments)?;
        // SAFETY-REVIEWED: the views, the argument array, and every payload it names live in this frame.
        status_result(unsafe {
            service(
                self.raw.user_data,
                collective,
                stream,
                BundleByteViewV1::from_bytes(operation),
                pointer,
                count,
            )
        })
    }

    /// Asks the host to run one operation this bundle supplies itself.
    ///
    /// The bundle names its operands as spans the host already issued and hands
    /// over one native entry. Everything about the call the host owns stays the
    /// host's: it validates, waits on the stream, establishes the context,
    /// lends raw values for the length of the callback, waits again for an
    /// operation that said it may enqueue, and releases nothing early. A
    /// refusal means the callback never ran, and a wait it could not complete
    /// before the callback is one of those refusals.
    ///
    /// A nonzero code from the operation itself comes back as
    /// [`BundleAbiError::CustomOperation`], which is the bundle's own
    /// vocabulary rather than the host's.
    pub fn run_custom_operation(
        &self,
        context: BundleContextHandleV1,
        stream: BundleStreamHandleV1,
        completion: CustomCompletionV1,
        operation_id: &[u8],
        operands: &[CustomOperandV1],
        operation: &CustomOperationV1<'_>,
    ) -> Result<(), BundleAbiError> {
        let service = required_service_v1(self.raw.run_custom_operation, "run_custom_operation")?;
        let raw_operands = raw_custom_operands_v1(operands)?;
        let (pointer, count) = custom_operand_array_v1(&raw_operands)?;
        // The storage starts at the code that means "no code of its own", not
        // at success. A host that reports a bundle failure and writes nothing
        // then reads back as a fault the operation could not describe, rather
        // than as an operation that succeeded and failed at the same time.
        let mut operation_status: i32 = BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT;
        // SAFETY-REVIEWED: the operand array and the status live in this frame for the call.
        let status = unsafe {
            service(
                self.raw.user_data,
                context,
                stream,
                completion.flags(),
                BundleByteViewV1::from_bytes(operation_id),
                pointer,
                count,
                Some(operation.entry()),
                operation.user_data(),
                &raw mut operation_status,
            )
        };
        if status == BundleStatusV1::BundleError {
            // Zero is the operation's word for success, so it cannot also be
            // the code behind a bundle failure. A host that reports one that
            // way has told this caller nothing it can act on.
            if operation_status == 0 {
                return Err(BundleAbiError::MissingCustomOperationStatus);
            }
            return Err(BundleAbiError::CustomOperation(operation_status));
        }
        status_result(status)
    }

    status_service_call_v1!(release_context, (context: BundleContextHandleV1));
    status_service_call_v1!(release_memory_pool, (pool: BundleMemoryPoolHandleV1));
    status_service_call_v1!(
        release_device_buffer,
        (buffer: BundleDeviceBufferHandleV1)
    );
    status_service_call_v1!(release_rank_arena, (arena: BundleRankArenaHandleV1));
    status_service_call_v1!(
        release_compiled_image,
        (image: BundleCompiledImageHandleV1)
    );
    status_service_call_v1!(
        release_authenticated_component,
        (component: BundleAuthenticatedComponentHandleV1)
    );
    status_service_call_v1!(
        release_peer_access,
        (peer_access: BundlePeerAccessHandleV1)
    );
    status_service_call_v1!(synchronize_stream, (stream: BundleStreamHandleV1));
    status_service_call_v1!(release_stream, (stream: BundleStreamHandleV1));
    status_service_call_v1!(
        record_event,
        (
            event: BundleEventHandleV1,
            stream: BundleStreamHandleV1,
        )
    );
    status_service_call_v1!(
        wait_event,
        (
            stream: BundleStreamHandleV1,
            event: BundleEventHandleV1,
        )
    );
    status_service_call_v1!(synchronize_event, (event: BundleEventHandleV1));
    status_service_call_v1!(release_event, (event: BundleEventHandleV1));
    status_service_call_v1!(
        configure_cuda_symbol,
        (
            symbol: BundleCudaSymbolHandleV1,
            dynamic_shared_memory_bytes: u32,
        )
    );
    status_service_call_v1!(
        release_cuda_symbol,
        (symbol: BundleCudaSymbolHandleV1)
    );
    status_service_call_v1!(
        release_cuda_module,
        (module: BundleCudaModuleHandleV1)
    );
    status_service_call_v1!(commit_kv, (transaction: BundleKvTransactionHandleV1));
    status_service_call_v1!(cancel_kv, (transaction: BundleKvTransactionHandleV1));
    status_service_call_v1!(
        release_kv_request,
        (request: BundleKvRequestHandleV1)
    );
    status_service_call_v1!(release_physical_kv, (kv: BundlePhysicalKvHandleV1));
    status_service_call_v1!(release_state, (state: BundleStateHandleV1));
    status_service_call_v1!(
        release_collective,
        (collective: BundleCollectiveHandleV1)
    );
    status_service_call_v1!(begin_graph_capture, (stream: BundleStreamHandleV1));
    status_service_call_v1!(launch_graph, (stream: BundleStreamHandleV1, graph: BundleGraphHandleV1));
    status_service_call_v1!(release_graph, (graph: BundleGraphHandleV1));

    output_service_call_v1!(
        select_context,
        BundleContextHandleV1,
        (rank: u32, device_ordinal: i32)
    );
    output_service_call_v1!(
        select_memory_pool,
        BundleMemoryPoolHandleV1,
        (context: BundleContextHandleV1)
    );
    output_service_call_v1!(
        allocate_device_buffer,
        BundleDeviceBufferHandleV1,
        (
            pool: BundleMemoryPoolHandleV1,
            stream: BundleStreamHandleV1,
            byte_length: u64,
            alignment: u64,
        )
    );
    output_service_call_v1!(
        upload_rank_arena,
        BundleRankArenaHandleV1,
        (
            context: BundleContextHandleV1,
            pool: BundleMemoryPoolHandleV1,
            stream: BundleStreamHandleV1,
            image: BundleCompiledImageHandleV1,
            rank: u32,
        )
    );
    output_service_call_v1!(
        acquire_peer_access,
        BundlePeerAccessHandleV1,
        (
            owner_context: BundleContextHandleV1,
            peer_context: BundleContextHandleV1,
        )
    );
    output_service_call_v1!(
        create_stream,
        BundleStreamHandleV1,
        (context: BundleContextHandleV1, stream_kind: u32)
    );
    output_service_call_v1!(
        create_event,
        BundleEventHandleV1,
        (context: BundleContextHandleV1)
    );
    output_service_call_v1!(end_graph_capture, BundleGraphHandleV1, (stream: BundleStreamHandleV1));
    output_service_call_v1!(
        create_kv_request,
        BundleKvRequestHandleV1,
        (kv: BundlePhysicalKvHandleV1, quota_tokens: u64)
    );
}

fn checked_device_buffer_view_v1(
    span: DeviceBufferSpanV1,
) -> Result<BundleDeviceBufferViewV1, BundleAbiError> {
    let view = span.to_view();
    validate_device_buffer_view_v1(view)?;
    Ok(view)
}

fn checked_compiled_image_span_v1(
    view: BundleRawCompiledImageViewV1,
) -> Result<CompiledImageSpanV1, BundleAbiError> {
    validate_compiled_image_view_v1(view)?;
    Ok(CompiledImageSpanV1::new(
        view.rank,
        DeviceBufferSpanV1::from_view(view.buffer),
        view.alignment,
    ))
}

/// One call's argument records, and the payload records they point at.
///
/// A V1 argument names its large payload rather than carrying it, so the record
/// a callee reads holds that payload's address. The address has to stay valid
/// for the length of the call and no longer, which is exactly this value's life:
/// the payload vectors below own every record an argument points at, and moving
/// this value moves the vectors rather than the storage behind them.
struct RawKernelArgumentsV1 {
    arguments: Vec<BundleKernelArgumentV1>,
    tensor_maps: Vec<BundleTensorMapV1>,
    pointer_arrays: Vec<BundleDevicePointerArrayV1>,
}

impl RawKernelArgumentsV1 {
    fn new(arguments: &[KernelArgumentV1]) -> Result<Self, BundleAbiError> {
        let mut built = Self {
            arguments: Vec::with_capacity(arguments.len()),
            tensor_maps: Vec::new(),
            pointer_arrays: Vec::new(),
        };
        // The payloads are collected first and pointed at afterwards. Taking an
        // address from a vector that is still growing would name storage the
        // next push could move; once every payload is in place, neither vector
        // grows again and the addresses are the ones the callee reads.
        for argument in arguments {
            let mut raw = BundleKernelArgumentV1::empty();
            match argument {
                KernelArgumentV1::DeviceBuffer(span) => {
                    raw.kind = BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_BUFFER;
                    raw.buffer = span.to_view();
                }
                KernelArgumentV1::NullDevicePointer => {
                    raw.kind = BUNDLE_KERNEL_ARGUMENT_V1_NULL_DEVICE_POINTER;
                }
                KernelArgumentV1::ScalarU32(value) => {
                    raw.kind = BUNDLE_KERNEL_ARGUMENT_V1_U32;
                    raw.scalar_u32 = *value;
                }
                KernelArgumentV1::ScalarU64(value) => {
                    raw.kind = BUNDLE_KERNEL_ARGUMENT_V1_U64;
                    raw.scalar_u64 = *value;
                }
                KernelArgumentV1::OpaqueBytes(bytes) => {
                    raw.kind = BUNDLE_KERNEL_ARGUMENT_V1_OPAQUE_BYTES;
                    raw.opaque = BundleByteViewV1::from_bytes(bytes);
                }
                KernelArgumentV1::ScalarI32(value) => {
                    raw.kind = BUNDLE_KERNEL_ARGUMENT_V1_I32;
                    raw.scalar_i32 = *value;
                }
                KernelArgumentV1::TensorMap(map) => {
                    let map = map.to_raw();
                    validate_tensor_map_v1(map)?;
                    raw.kind = BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP;
                    built.tensor_maps.push(map);
                }
                KernelArgumentV1::DevicePointerArray(slots) => {
                    raw.kind = BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY;
                    built
                        .pointer_arrays
                        .push(raw_device_pointer_array_v1(slots)?);
                }
            }
            built.arguments.push(raw);
        }
        let mut maps = built.tensor_maps.iter();
        let mut arrays = built.pointer_arrays.iter();
        for raw in &mut built.arguments {
            match raw.kind {
                BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP => {
                    raw.tensor_map = ptr::from_ref(
                        maps.next()
                            .expect("one map was collected for each tensor-map argument"),
                    );
                }
                BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY => {
                    raw.pointer_array = ptr::from_ref(
                        arrays
                            .next()
                            .expect("one array was collected for each pointer-array argument"),
                    );
                }
                _ => {}
            }
        }
        validate_kernel_arguments_v1(&built.arguments)?;
        Ok(built)
    }
}

fn raw_custom_operands_v1(
    operands: &[CustomOperandV1],
) -> Result<Vec<BundleCustomOperandV1>, BundleAbiError> {
    let operands = operands
        .iter()
        .map(|operand| BundleCustomOperandV1 {
            struct_size: struct_size_v1::<BundleCustomOperandV1>(),
            flags: operand.flags(),
            buffer: operand.span.to_view(),
        })
        .collect::<Vec<BundleCustomOperandV1>>();
    validate_custom_operands_v1(&operands)?;
    Ok(operands)
}

fn custom_operand_array_v1(
    operands: &[BundleCustomOperandV1],
) -> Result<(*const BundleCustomOperandV1, u64), BundleAbiError> {
    if operands.is_empty() {
        return Ok((ptr::null(), 0));
    }
    let count = u64::try_from(operands.len()).map_err(|_| BundleAbiError::LengthOutOfRange {
        surface: "custom operand array",
        actual: u64::MAX,
    })?;
    Ok((operands.as_ptr(), count))
}

fn kernel_argument_array_v1(
    arguments: &[BundleKernelArgumentV1],
) -> Result<(*const BundleKernelArgumentV1, u64), BundleAbiError> {
    if arguments.is_empty() {
        return Ok((ptr::null(), 0));
    }
    let count = u64::try_from(arguments.len()).map_err(|_| BundleAbiError::LengthOutOfRange {
        surface: "kernel argument array",
        actual: u64::MAX,
    })?;
    Ok((arguments.as_ptr(), count))
}

fn raw_physical_kv_request_v1(
    request: &PhysicalKvRequestV1,
) -> Result<BundlePhysicalKvRequestV1, BundleAbiError> {
    let mut raw = BundlePhysicalKvRequestV1::empty();
    raw.flags = request.flags;
    raw.layout.flags = request.layout.flags;
    raw.layout.rank_count = request.layout.rank_count;
    raw.layout.page_bytes = request.layout.page_bytes;
    raw.layout.page_alignment = request.layout.page_alignment;
    raw.layout.page_tokens = request.layout.page_tokens;
    raw.layout.state_lifetime = request.layout.state_lifetime;
    raw.layout.layout_id = BundleByteViewV1::from_bytes(&request.layout.layout_id);
    raw.page_capacity = request.page_capacity;
    raw.request_capacity = request.request_capacity;
    validate_physical_kv_request_v1(raw)?;
    Ok(raw)
}

fn raw_state_request_v1(request: &StateRequestV1) -> Result<BundleStateRequestV1, BundleAbiError> {
    let mut raw = BundleStateRequestV1::empty();
    raw.flags = request.flags;
    raw.byte_length = request.byte_length;
    raw.alignment = request.alignment;
    raw.state_id = BundleByteViewV1::from_bytes(&request.state_id);
    validate_state_request_v1(raw)?;
    Ok(raw)
}

fn raw_collective_request_v1(
    request: &CollectiveRequestV1,
) -> Result<BundleCollectiveRequestV1, BundleAbiError> {
    let mut raw = BundleCollectiveRequestV1::empty();
    raw.flags = request.flags;
    raw.rank_count = request.rank_count;
    raw.semantic_contract = BundleByteViewV1::from_bytes(&request.semantic_contract);
    validate_collective_request_v1(raw)?;
    Ok(raw)
}

/// Calls one bundle's create entry with a pinned table and borrowed configuration.
pub(crate) fn invoke_bundle_create_v1(
    create: BundleCreateFnV1,
    host_services: &HostServicesTableV1,
    configuration: &[u8],
) -> BundleCreateResultV1 {
    // SAFETY-REVIEWED: the table pins a complete live record and the view borrows the caller's bytes.
    unsafe {
        create(
            host_services.raw_pointer(),
            BundleByteViewV1::from_bytes(configuration),
        )
    }
}

/// Calls one bundle's execute entry with a result sink installed for that call.
pub(crate) fn invoke_bundle_execute_v1<S: LogitsRowSinkV1>(
    execute: BundleExecuteFnV1,
    handle: BundleHandleV1,
    request: &[u8],
    sink: &mut S,
) -> Result<BundleStatusV1, BundleAbiError> {
    with_result_sink_v1(sink, |result_sink| {
        validate_execute_result_sink_v1(result_sink)?;
        // SAFETY-REVIEWED: the sink lives for this call and the view borrows the caller's request bytes.
        Ok(unsafe { execute(handle, BundleByteViewV1::from_bytes(request), result_sink) })
    })
}

/// Delivers one row to the host sink a native execute call installed.
pub(crate) fn invoke_emit_logits_row_v1(
    sink: &BundleExecuteResultSinkV1,
    row: BundleLogitsRowV1,
) -> Result<(), BundleAbiError> {
    let emit_logits_row = required_execute_result_callback_v1(sink.emit_logits_row)?;
    // SAFETY-REVIEWED: the native execute call owns this sink and the row borrows the bundle's bytes.
    status_result(unsafe { emit_logits_row(sink.user_data, row) })
}

pub(crate) fn invoke_emit_logits_view_v1(
    sink: &BundleExecuteResultSinkV1,
    view: BundleLogitsViewV1,
) -> Result<(), BundleAbiError> {
    if sink.struct_size < struct_size_v1::<BundleExecuteResultSinkV1>() {
        return Err(BundleAbiError::MissingExecuteResultCallback);
    }
    let emit = sink
        .emit_logits_view
        .ok_or(BundleAbiError::MissingExecuteResultCallback)?;
    // SAFETY-REVIEWED: the native execute call owns this sink and the view is a fixed-width record.
    status_result(unsafe { emit(sink.user_data, view) })
}

pub(crate) fn invoke_emit_draft_tokens_v1(
    sink: &BundleExecuteResultSinkV1,
    drafts: BundleDraftTokensV1,
) -> Result<(), BundleAbiError> {
    let end = std::mem::offset_of!(BundleExecuteResultSinkV1, emit_draft_tokens)
        + std::mem::size_of_val(&sink.emit_draft_tokens);
    if (sink.struct_size as usize) < end {
        return Err(BundleAbiError::MissingExecuteResultCallback);
    }
    let emit = sink
        .emit_draft_tokens
        .ok_or(BundleAbiError::MissingExecuteResultCallback)?;
    // SAFETY-REVIEWED: the execute call owns this sink and drafts borrows this frame's bytes.
    status_result(unsafe { emit(sink.user_data, drafts) })
}

pub(crate) fn invoke_bundle_commit_accepted_v1(
    callback: BundleCommitAcceptedFnV1,
    handle: BundleHandleV1,
    records: &[BundleKeptRowsV1],
) -> Result<(), BundleAbiError> {
    let count = u32::try_from(records.len()).map_err(|_| BundleAbiError::InvalidKeptRows)?;
    let pointer = if records.is_empty() {
        ptr::null()
    } else {
        records.as_ptr()
    };
    // SAFETY-REVIEWED: the callback borrows the checked fixed-width slice only for this call.
    status_result(unsafe { callback(handle, pointer, count) })
}

pub(crate) fn invoke_bundle_describe_prefix_state_v1(
    callback: crate::BundleDescribePrefixStateFnV1,
    handle: BundleHandleV1,
) -> crate::BundlePrefixStateInfoV1 {
    callback(handle)
}

/// Most token IDs one prefix-state record may carry.
pub const BUNDLE_PREFIX_STATE_V1_MAX_TOKENS: u64 = 1 << 24;
/// Most snapshot pages one prefix-state record may name.
pub const BUNDLE_PREFIX_STATE_V1_MAX_PAGES: u64 = 1 << 20;

pub(crate) fn invoke_bundle_prefix_state_v1(
    callback: crate::BundlePrefixStateFnV1,
    handle: BundleHandleV1,
    request_id: u64,
    tokens: u64,
    token_ids: &[u32],
    pages: &[u64],
) -> Result<(), BundleAbiError> {
    with_prefix_state_record_v1(request_id, tokens, token_ids, pages, |record| {
        // SAFETY-REVIEWED: the record's views borrow the caller's byte vectors for the call.
        unsafe { callback(handle, record) }
    })
}

pub(crate) fn invoke_bundle_restore_prefix_state_in_slot_v1(
    callback: crate::BundleRestorePrefixStateInSlotFnV1,
    handle: BundleHandleV1,
    token_ids: &[u32],
    pages: &[u64],
    request_id: u64,
    state_slot: u64,
) -> Result<(), BundleAbiError> {
    with_prefix_state_record_v1(
        request_id,
        token_ids.len() as u64,
        token_ids,
        pages,
        |record| {
            // SAFETY-REVIEWED: the record's views borrow the caller's byte vectors for the call.
            unsafe { callback(handle, record, state_slot) }
        },
    )
}

/// Builds one prefix-state record over this frame's byte vectors for `call`.
fn with_prefix_state_record_v1(
    request_id: u64,
    tokens: u64,
    token_ids: &[u32],
    pages: &[u64],
    call: impl FnOnce(crate::BundlePrefixStateV1) -> BundleStatusV1,
) -> Result<(), BundleAbiError> {
    if tokens > BUNDLE_PREFIX_STATE_V1_MAX_TOKENS
        || pages.len() as u64 > BUNDLE_PREFIX_STATE_V1_MAX_PAGES
    {
        return Err(BundleAbiError::LengthOutOfRange {
            surface: "prefix state",
            actual: tokens.max(pages.len() as u64),
        });
    }
    let token_bytes = token_ids
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect::<Vec<_>>();
    let page_bytes = pages
        .iter()
        .flat_map(|page| page.to_le_bytes())
        .collect::<Vec<_>>();
    let record = crate::BundlePrefixStateV1 {
        struct_size: crate::struct_size_v1::<crate::BundlePrefixStateV1>(),
        reserved: 0,
        request_id,
        tokens,
        token_ids: BundleByteViewV1::from_bytes(&token_bytes),
        pages: BundleByteViewV1::from_bytes(&page_bytes),
    };
    status_result(call(record))
}

/// One decoded prefix-state record, owned by the bundle's handler.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixStateRequestV1 {
    pub request_id: u64,
    pub tokens: u64,
    /// The prefix token IDs of a restore; empty for a capture.
    pub token_ids: Vec<u32>,
    pub pages: Vec<u64>,
}

/// Decodes a prefix-state record before calling a bundle's safe handler.
///
/// # Safety
/// The native caller owns the record's two readable byte views for this call.
// SAFETY-REVIEWED: this is the reviewed byte-view boundary for the prefix-state tail.
pub unsafe fn dispatch_bundle_prefix_state_v1(
    handle: BundleHandleV1,
    state: crate::BundlePrefixStateV1,
    handler: impl FnOnce(BundleHandleV1, PrefixStateRequestV1) -> BundleStatusV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        if state.struct_size != crate::struct_size_v1::<crate::BundlePrefixStateV1>()
            || state.reserved != 0
            || state.tokens > BUNDLE_PREFIX_STATE_V1_MAX_TOKENS
            || state.pages.length > BUNDLE_PREFIX_STATE_V1_MAX_PAGES * 8
            || state.pages.length % 8 != 0
            || (state.token_ids.length != 0 && state.token_ids.length != state.tokens * 4)
        {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument));
        }
        let token_ids = with_byte_view_v1(state.token_ids, |bytes| {
            bytes
                .chunks_exact(4)
                .map(|lane| u32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]))
                .collect::<Vec<_>>()
        })?;
        let pages = with_byte_view_v1(state.pages, |bytes| {
            bytes
                .chunks_exact(8)
                .map(|lane| {
                    u64::from_le_bytes([
                        lane[0], lane[1], lane[2], lane[3], lane[4], lane[5], lane[6], lane[7],
                    ])
                })
                .collect::<Vec<_>>()
        })?;
        Ok(handler(
            handle,
            PrefixStateRequestV1 {
                request_id: state.request_id,
                tokens: state.tokens,
                token_ids,
                pages,
            },
        ))
    })
}

/// Converts a Rust panic into a failed prefix-state description.
pub fn catch_panic_prefix_state_info_v1(
    callback: impl FnOnce() -> crate::BundlePrefixStateInfoV1,
) -> crate::BundlePrefixStateInfoV1 {
    catch_unwind(AssertUnwindSafe(callback))
        .unwrap_or_else(|_| crate::BundlePrefixStateInfoV1::failure(BundleStatusV1::Panic))
}

/// Decodes the exact kept-row array before calling a bundle's safe handler.
///
/// # Safety
/// The native caller owns a readable, aligned array of `record_count` elements.
// SAFETY-REVIEWED: this is the reviewed pointer boundary for the commit tail.
pub unsafe fn dispatch_bundle_commit_accepted_v1(
    handle: BundleHandleV1,
    records: *const BundleKeptRowsV1,
    record_count: u32,
    handler: impl FnOnce(BundleHandleV1, &[BundleKeptRowsV1]) -> BundleStatusV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        if record_count > 1 << 20 {
            return Err(BundleAbiError::LengthOutOfRange {
                surface: "kept-row array",
                actual: u64::from(record_count),
            });
        }
        let count =
            checked_slice_count_v1::<BundleKeptRowsV1>("kept-row array", u64::from(record_count))?;
        if count == 0 {
            if !records.is_null() {
                return Err(BundleAbiError::InvalidPointerLength("kept-row array"));
            }
            return Ok(handler(handle, &[]));
        }
        validate_input_pointer_v1(records, "kept-row array")?;
        // SAFETY-REVIEWED: the native callback promises the fixed-width array remains readable.
        let rows = unsafe { slice::from_raw_parts(records, count) };
        for row in rows {
            validate_kept_rows_v1(*row)?;
        }
        Ok(handler(handle, rows))
    })
}

fn required_execute_result_callback_v1(
    callback: Option<BundleEmitLogitsRowFnV1>,
) -> Result<BundleEmitLogitsRowFnV1, BundleAbiError> {
    callback.ok_or(BundleAbiError::MissingExecuteResultCallback)
}

/// The host-side result channel one execute call installs for a bundle.
struct ResultSinkStateV1<'a, S: LogitsRowSinkV1> {
    sink: &'a mut S,
    limits: ResultSinkLimitsV1,
    emitted_rows: u64,
    emitted_logits: u64,
    device_cycle: Option<EncodedDeviceCycleV1>,
}

/// Installs a native result sink around `sink` for exactly one `callback` call.
pub(crate) fn with_result_sink_v1<S: LogitsRowSinkV1, T>(
    sink: &mut S,
    callback: impl FnOnce(BundleExecuteResultSinkV1) -> T,
) -> T {
    let limits = sink.limits();
    let state = RefCell::new(ResultSinkStateV1 {
        sink,
        limits,
        emitted_rows: 0,
        emitted_logits: 0,
        device_cycle: None,
    });
    let raw = BundleExecuteResultSinkV1 {
        struct_size: struct_size_v1::<BundleExecuteResultSinkV1>(),
        reserved: 0,
        user_data: ptr::from_ref(&state).cast::<c_void>().cast_mut(),
        maximum_rows: limits.maximum_rows,
        maximum_logits_per_row: limits.maximum_logits_per_row,
        maximum_total_logits: limits.maximum_total_logits,
        emit_logits_row: Some(emit_logits_row_trampoline_v1::<S>),
        emit_logits_view: Some(emit_logits_view_trampoline_v1::<S>),
        emit_draft_tokens: Some(emit_draft_tokens_trampoline_v1::<S>),
    };
    callback(raw)
}

// SAFETY-REVIEWED: a V1 caller supplies its own live user_data and views for this callback.
unsafe extern "C" fn emit_logits_row_trampoline_v1<S: LogitsRowSinkV1>(
    user_data: *mut c_void,
    row: BundleLogitsRowV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let state = result_sink_state_from_user_data_v1::<S>(user_data)?;
        let row = copy_logits_row_v1(row)?;
        let mut state = state
            .try_borrow_mut()
            .map_err(|_| BundleAbiError::Lifecycle("reentrant result sink"))?;
        let state = &mut *state;
        charge_result_capacity_v1(
            state.limits,
            &mut state.emitted_rows,
            &mut state.emitted_logits,
            row.logit_count(),
        )?;
        Ok(host_unit_status_v1(state.sink.emit_row(row)))
    })
}

// SAFETY-REVIEWED: a V1 caller supplies live user_data and a fixed-width device view.
unsafe extern "C" fn emit_logits_view_trampoline_v1<S: LogitsRowSinkV1>(
    user_data: *mut c_void,
    view: BundleLogitsViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        validate_execute_logits_view_v1(view)?;
        let state = result_sink_state_from_user_data_v1::<S>(user_data)?;
        let mut state = state
            .try_borrow_mut()
            .map_err(|_| BundleAbiError::Lifecycle("reentrant result sink"))?;
        let state = &mut *state;
        charge_result_capacity_rows_v1(
            state.limits,
            &mut state.emitted_rows,
            &mut state.emitted_logits,
            u64::from(view.row_count),
            u64::from(view.live_width),
        )?;
        Ok(host_unit_status_v1(
            state.sink.emit_view(OwnedLogitsViewV1::new(view)),
        ))
    })
}

// SAFETY-REVIEWED: the native execute call owns the draft record until this callback returns.
unsafe extern "C" fn emit_draft_tokens_trampoline_v1<S: LogitsRowSinkV1>(
    user_data: *mut c_void,
    record: BundleDraftTokensV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        validate_draft_tokens_v1(record)?;
        let bytes = copy_byte_view_v1(record.token_bytes)?;
        let tokens = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("validated draft token width")))
            .collect();
        let state = result_sink_state_from_user_data_v1::<S>(user_data)?;
        let mut state = state
            .try_borrow_mut()
            .map_err(|_| BundleAbiError::Lifecycle("reentrant result sink"))?;
        Ok(host_unit_status_v1(state.sink.emit_draft_tokens(
            OwnedDraftTokensV1 {
                request_id: record.request_id,
                tokens,
            },
        )))
    })
}

fn result_sink_state_from_user_data_v1<'a, S: LogitsRowSinkV1>(
    user_data: *mut c_void,
) -> Result<&'a RefCell<ResultSinkStateV1<'a, S>>, BundleAbiError> {
    let state = user_data
        .cast::<RefCell<ResultSinkStateV1<'a, S>>>()
        .cast_const();
    validate_input_pointer_v1(state, "execute result sink")?;
    // SAFETY-REVIEWED: only the enclosing execute call installs this pointer, and it outlives the call.
    Ok(unsafe { &*state })
}

/// Decodes the native create arguments for one bundle implementation.
///
/// # Safety
///
/// The caller must pass the exact arguments one native `create` call received.
/// For a nonempty configuration view, and for a non-null host-service table,
/// the native caller must keep that storage readable until this call returns,
/// and must keep the service table and its `user_data` valid for the lifetime
/// of the instance the bundle returns.
// SAFETY-REVIEWED: this entry owns the create pointer decode for a generated trampoline.
pub unsafe fn dispatch_bundle_create_v1(
    host_services: *const BundleHostServicesV1,
    configuration: BundleByteViewV1,
    create: impl FnOnce(HostServicesV1, &[u8]) -> BundleCreateResultV1,
) -> BundleCreateResultV1 {
    catch_panic_create_result_v1(|| {
        let capture_services = |raw| {
            // SAFETY-REVIEWED: the native create contract keeps this table live for the instance.
            unsafe { HostServicesV1::new(raw) }
        };
        let services = match copy_host_services_v1(host_services).and_then(capture_services) {
            Ok(services) => services,
            Err(error) => {
                return BundleCreateResultV1::failure(callback_error_status_v1(error));
            }
        };
        match with_byte_view_v1(configuration, |configuration| {
            create(services, configuration)
        }) {
            Ok(result) => result,
            Err(error) => BundleCreateResultV1::failure(callback_error_status_v1(error)),
        }
    })
}

fn copy_host_services_v1(
    pointer: *const BundleHostServicesV1,
) -> Result<BundleHostServicesV1, BundleAbiError> {
    validate_input_pointer_v1(pointer, "host services")?;
    // SAFETY-REVIEWED: the aligned table pointer exposes its first size field.
    let actual = unsafe { (*pointer).struct_size };
    let minimum = std::mem::offset_of!(BundleHostServicesV1, acquire_serving_kv) as u32;
    if actual < minimum {
        return Err(BundleAbiError::StructSize {
            surface: "host services",
            expected: minimum,
            actual,
        });
    }
    let mut copied = BundleHostServicesV1::empty();
    let length = (actual as usize).min(std::mem::size_of::<BundleHostServicesV1>());
    // SAFETY-REVIEWED: the size prefix bounds the readable bytes, and copied owns a full table.
    unsafe {
        ptr::copy_nonoverlapping(
            pointer.cast::<u8>(),
            (&mut copied as *mut BundleHostServicesV1).cast::<u8>(),
            length,
        );
    }
    Ok(copied)
}

/// Decodes the native execute arguments for one bundle implementation.
///
/// # Safety
///
/// The caller must pass the exact arguments one native `execute` call received.
/// For a nonempty request view, the native caller must keep that storage
/// readable until this call returns, and must keep the result sink and its
/// `user_data` valid for the same period.
// SAFETY-REVIEWED: this entry owns the execute pointer decode for a generated trampoline.
pub unsafe fn dispatch_bundle_execute_v1(
    handle: BundleHandleV1,
    request: BundleByteViewV1,
    result_sink: BundleExecuteResultSinkV1,
    execute: impl FnOnce(BundleHandleV1, &[u8], &mut ExecuteResultSinkV1) -> BundleStatusV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let mut result_sink = ExecuteResultSinkV1::new(result_sink)?;
        with_byte_view_v1(request, |request| {
            execute(handle, request, &mut result_sink)
        })
    })
}

/// Decodes the lent values one native custom-operation entry received.
///
/// This is the callback half of the custom-operation boundary, for an operation
/// written in Rust rather than in native code. It checks the record, rebuilds
/// the lent ranges as a borrowed slice, hands both to a safe handler with the
/// operation's own state, and contains a panic as
/// [`BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT`] rather than letting it
/// unwind into the host. The slice ends with the call, which is the whole of
/// what the operation is lent.
///
/// # Safety
///
/// The caller must pass the exact arguments one native custom-operation call
/// received, and `operation_user_data` must point to a live `S` that no other
/// borrow of this call names.
// SAFETY-REVIEWED: this entry owns the custom-operation pointer decode for a generated callback.
pub unsafe fn dispatch_custom_operation_v1<S>(
    operation_user_data: *mut c_void,
    launch: *const BundleCustomLaunchV1,
    handler: impl FnOnce(&mut S, LentLaunchV1<'_>) -> i32,
) -> i32 {
    let decoded = catch_unwind(AssertUnwindSafe(|| -> Result<i32, BundleAbiError> {
        let state = operation_user_data.cast::<S>();
        validate_output_pointer_v1(state, "custom operation state")?;
        validate_input_pointer_v1(launch, "custom launch")?;
        // SAFETY-REVIEWED: the host guarantees one readable aligned launch record for this call.
        let record = unsafe { ptr::read(launch) };
        let address_space = validate_custom_launch_v1(record)?;
        let count = checked_slice_count_v1::<u64>("custom launch operands", record.operand_count)?;
        // SAFETY-REVIEWED: the state pointer is the operation's own live storage for this call.
        let state = unsafe { &mut *state };
        if count == 0 {
            return Ok(handler(
                state,
                LentLaunchV1 {
                    address_space,
                    stream: LentStreamV1::from_raw(record.stream),
                    ranges: &[],
                },
            ));
        }
        // Both arrays are read as `u64`, so both have to be non-null and
        // aligned for one before either is turned into a slice. A record that
        // fails this is refused rather than read at a misaligned address.
        validate_input_pointer_v1(record.operand_addresses, "custom launch addresses")?;
        validate_input_pointer_v1(record.operand_lengths, "custom launch lengths")?;
        // Both arrays are checked non-null and aligned just above, and the host
        // guarantees `operand_count` readable elements in each for this call.
        // SAFETY-REVIEWED: the pointers are validated and the count is the record's own.
        let (addresses, lengths) = unsafe {
            (
                slice::from_raw_parts(record.operand_addresses, count),
                slice::from_raw_parts(record.operand_lengths, count),
            )
        };
        let ranges = addresses
            .iter()
            .zip(lengths)
            .map(|(address, length)| checked_lent_range_v1(*address, *length))
            .collect::<Result<Vec<LentRangeV1>, BundleAbiError>>()?;
        Ok(handler(
            state,
            LentLaunchV1 {
                address_space,
                stream: LentStreamV1::from_raw(record.stream),
                ranges: &ranges,
            },
        ))
    }));
    match decoded {
        Ok(Ok(code)) => code,
        // A record this entry cannot read and a panic it contained are the same
        // thing to the host: the operation could not report a code of its own.
        Ok(Err(_)) | Err(_) => BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT,
    }
}

/// Checks one lent launch record before anything reads through its arrays.
///
/// The answer is the address space the record named, because deciding that is
/// part of deciding whether the record is readable at all: it fixes whether a
/// stream must be there. Everything a caller could get wrong about the shape of
/// the record is settled here, and nothing is read through it until it is.
pub(crate) fn validate_custom_launch_v1(
    record: BundleCustomLaunchV1,
) -> Result<CustomAddressSpaceV1, BundleAbiError> {
    if record.struct_size < struct_size_v1::<BundleCustomLaunchV1>() {
        return Err(BundleAbiError::StructSize {
            surface: "custom launch",
            expected: struct_size_v1::<BundleCustomLaunchV1>(),
            actual: record.struct_size,
        });
    }
    // Both arrays are null exactly when the count is zero, and a record that
    // pairs them any other way is one no length can be trusted from.
    let empty = record.operand_count == 0;
    if record.operand_addresses.is_null() != empty || record.operand_lengths.is_null() != empty {
        return Err(BundleAbiError::InvalidPointerLength(
            "custom launch operands",
        ));
    }
    let address_space = CustomAddressSpaceV1::from_code(record.address_space)?;
    // The address space says what the stream is. A device lend orders work on
    // the host's own stream, so there has to be one; a host lend has no stream
    // at all, and a record that carries one is not the record it claims to be.
    let has_stream = !record.stream.is_null();
    if has_stream != matches!(address_space, CustomAddressSpaceV1::Device) {
        return Err(BundleAbiError::InvalidCustomLaunch("stream"));
    }
    Ok(address_space)
}

/// Checks one lent range before an operation is allowed to use it.
///
/// A zero address is no memory, a zero length is no range, and a range whose
/// end does not fit is one no operation can bound its own work by. Each of
/// those is refused rather than handed on.
fn checked_lent_range_v1(address: u64, length: u64) -> Result<LentRangeV1, BundleAbiError> {
    if address == 0 {
        return Err(BundleAbiError::InvalidCustomLaunch("operand address"));
    }
    if length == 0 {
        return Err(BundleAbiError::InvalidCustomLaunch("operand length"));
    }
    address
        .checked_add(length)
        .ok_or(BundleAbiError::InvalidCustomLaunch("operand range"))?;
    Ok(LentRangeV1::new(address, length))
}

/// Exposes an immutable byte view only while `callback` runs.
pub(crate) fn with_byte_view_v1<T>(
    view: BundleByteViewV1,
    callback: impl FnOnce(&[u8]) -> T,
) -> Result<T, BundleAbiError> {
    validate_byte_view_v1(view)?;
    let length = checked_u64_to_usize_v1("byte view", view.length)?;
    if length == 0 {
        return Ok(callback(&[]));
    }
    // SAFETY-REVIEWED: the ABI callback guarantees a readable byte range for its duration.
    let bytes = unsafe { slice::from_raw_parts(view.data, length) };
    Ok(callback(bytes))
}

/// Exposes a caller's `u32` array only while `callback` runs: null exactly
/// when empty, aligned otherwise.
fn with_u32_array_v1<T>(
    values: *const u32,
    count: u64,
    surface: &'static str,
    callback: impl FnOnce(&[u32]) -> T,
) -> Result<T, BundleAbiError> {
    let count = checked_slice_count_v1::<u32>(surface, count)?;
    if count == 0 {
        if !values.is_null() {
            return Err(BundleAbiError::InvalidPointerLength(surface));
        }
        return Ok(callback(&[]));
    }
    validate_input_pointer_v1(values, surface)?;
    // SAFETY-REVIEWED: the ABI callback guarantees a readable aligned array of `count` values for its duration.
    Ok(callback(unsafe { slice::from_raw_parts(values, count) }))
}

/// Copies an immutable byte view while the native ABI callback owns it.
pub(crate) fn copy_byte_view_v1(view: BundleByteViewV1) -> Result<Vec<u8>, BundleAbiError> {
    with_byte_view_v1(view, <[u8]>::to_vec)
}

/// Exposes a mutable byte view only while `callback` runs.
pub(crate) fn with_mut_byte_view_v1<T>(
    view: BundleMutByteViewV1,
    callback: impl FnOnce(&mut [u8]) -> T,
) -> Result<T, BundleAbiError> {
    validate_mut_byte_view_v1(view)?;
    let length = checked_u64_to_usize_v1("mutable byte view", view.length)?;
    if length == 0 {
        let mut empty: [u8; 0] = [];
        return Ok(callback(&mut empty));
    }
    // SAFETY-REVIEWED: the ABI callback guarantees an exclusive writable byte range for its duration.
    let bytes = unsafe { slice::from_raw_parts_mut(view.data, length) };
    Ok(callback(bytes))
}

/// Copies one validated logits row into host-owned Rust storage.
pub(crate) fn copy_logits_row_v1(
    row: BundleLogitsRowV1,
) -> Result<OwnedLogitsRowV1, BundleAbiError> {
    validate_execute_logits_row_v1(row)?;
    let logit_bits = copy_byte_view_v1(row.logit_bits)?;
    Ok(OwnedLogitsRowV1::new(row.request_id, logit_bits))
}

/// Exposes validated kernel arguments only while `callback` runs.
fn with_kernel_arguments_v1<T>(
    arguments: *const BundleKernelArgumentV1,
    argument_count: u64,
    callback: impl FnOnce(&[BundleKernelArgumentV1]) -> T,
) -> Result<T, BundleAbiError> {
    let count =
        checked_slice_count_v1::<BundleKernelArgumentV1>("kernel argument array", argument_count)?;
    if count == 0 {
        if !arguments.is_null() {
            return Err(BundleAbiError::InvalidPointerLength(
                "kernel argument array",
            ));
        }
        return Ok(callback(&[]));
    }
    validate_input_pointer_v1(arguments, "kernel argument array")?;
    // The array has no stride of its own, so the first record's declared length
    // is what says whether the rest are where this reader will look for them.
    // It is read on its own, out of the four bytes every version of this record
    // opens with, before anything advances by one V1 record: an array whose
    // stride is not this one is refused without being walked.
    // SAFETY-REVIEWED: a nonnull aligned argument array has at least one record, whose leading struct_size this reads and nothing else.
    validate_kernel_argument_struct_size_v1(unsafe { ptr::read(arguments.cast::<u32>()) })?;
    let copied_arguments = copy_kernel_arguments_v1(arguments, count);
    validate_kernel_arguments_v1(&copied_arguments)?;
    Ok(callback(&copied_arguments))
}

fn checked_slice_count_v1<T>(surface: &'static str, count: u64) -> Result<usize, BundleAbiError> {
    let count = checked_u64_to_usize_v1(surface, count)?;
    let byte_length =
        count
            .checked_mul(size_of::<T>())
            .ok_or(BundleAbiError::LengthOutOfRange {
                surface,
                actual: u64::try_from(count).expect("usize value fits u64"),
            })?;
    if byte_length > isize::MAX as usize {
        return Err(BundleAbiError::LengthOutOfRange {
            surface,
            actual: u64::try_from(count).expect("usize value fits u64"),
        });
    }
    Ok(count)
}

fn with_input_record_v1<T: Copy, R>(
    pointer: *const T,
    surface: &'static str,
    callback: impl FnOnce(T) -> Result<R, BundleAbiError>,
) -> Result<R, BundleAbiError> {
    validate_input_pointer_v1(pointer, surface)?;
    // SAFETY-REVIEWED: the ABI callback guarantees one readable aligned input record for its duration.
    let record = unsafe { ptr::read(pointer) };
    callback(record)
}

fn copy_kernel_arguments_v1(
    arguments: *const BundleKernelArgumentV1,
    count: usize,
) -> Vec<BundleKernelArgumentV1> {
    // SAFETY-REVIEWED: the ABI callback guarantees a readable kernel argument array for its duration.
    unsafe { slice::from_raw_parts(arguments, count) }.to_vec()
}

fn write_output_v1<T: Copy>(output: *mut T, value: T) -> Result<(), BundleAbiError> {
    validate_output_pointer_v1(output, "callback output")?;
    // SAFETY-REVIEWED: the ABI callback guarantees aligned writable output storage for this value.
    unsafe { ptr::write(output, value) };
    Ok(())
}

fn validate_input_pointer_v1<T>(
    pointer: *const T,
    surface: &'static str,
) -> Result<(), BundleAbiError> {
    if pointer.is_null() {
        return Err(BundleAbiError::NullPointer(surface));
    }
    if !(pointer as usize).is_multiple_of(align_of::<T>()) {
        return Err(BundleAbiError::MisalignedPointer(surface));
    }
    Ok(())
}

fn validate_output_pointer_v1<T>(
    pointer: *mut T,
    surface: &'static str,
) -> Result<(), BundleAbiError> {
    if pointer.is_null() {
        return Err(BundleAbiError::NullPointer(surface));
    }
    if !(pointer as usize).is_multiple_of(align_of::<T>()) {
        return Err(BundleAbiError::MisalignedPointer(surface));
    }
    Ok(())
}

/// Exports the sole V1 entry point and panic-containing ABI callback trampolines.
///
/// Each supplied expression must be a safe Rust function that receives only
/// checked Rust values: `create` takes the validated host-service table and the
/// configuration bytes, `execute` takes the instance handle, the request bytes,
/// and a bounded result sink, and the transaction callbacks take the instance
/// handle. The macro owns every pointer decode and catches a panic before it
/// can return to native code, so a bundle contains no unsafe code.
///
/// The retirement form is `with_request_retirement BASE, EXTENDED, create,
/// execute, commit, cancel, destroy, retire_request`. It publishes both a copied
/// V1 prefix and the complete extension. The entry point returns the extension's
/// base, whose size covers the full table. The safe retirement handler takes an
/// instance handle and a request ID, and its panics are contained like finalization.
///
/// The state-slot form is `with_prefix_state_slot BASE, EXTENDED, create,
/// execute, commit, cancel, destroy, retire_request, [commit_accepted],
/// describe, capture, restore, restore_in_slot`. The brackets hold the
/// `commit_accepted` handler, or nothing for a bundle without speculation,
/// which exports a null callback. `restore_in_slot` takes the handle, the
/// decoded record and the host's state slot.
#[macro_export]
macro_rules! export_bundle_api_v1 {
    ($api:ident, $create:expr, $execute:expr, $commit:expr, $cancel:expr, $destroy:expr $(,)?) => {
        $crate::export_bundle_api_v1!(@callbacks $create, $execute, $commit, $cancel, $destroy);

        pub static $api: $crate::BundleRawApiV1 = $crate::BundleRawApiV1 {
            struct_size: ::core::mem::size_of::<$crate::BundleRawApiV1>() as u32,
            abi_version: $crate::BUNDLE_ABI_VERSION_V1,
            create: Some(__bundle_create_trampoline_v1),
            execute: Some(__bundle_execute_trampoline_v1),
            commit: Some(__bundle_commit_trampoline_v1),
            cancel: Some(__bundle_cancel_trampoline_v1),
            destroy: Some(__bundle_destroy_trampoline_v1),
        };

        // SAFETY-REVIEWED: the entry catches panics and returns only a static ABI table pointer.
        #[allow(unsafe_code)]
        #[unsafe(no_mangle)]
        pub extern "C" fn chop_bundle_get_api_v1() -> *const $crate::BundleRawApiV1 {
            $crate::catch_panic_api_pointer_v1(|| &$api)
        }
    };
    (with_request_retirement $api:ident, $extended:ident,
        $create:expr, $execute:expr, $commit:expr, $cancel:expr, $destroy:expr,
        $retire_request:expr $(,)?) => {
        $crate::export_bundle_api_v1!(@callbacks $create, $execute, $commit, $cancel, $destroy);

        extern "C" fn __bundle_retire_request_trampoline_v1(
            handle: $crate::BundleHandleV1,
            request_id: u64,
        ) -> $crate::BundleStatusV1 {
            $crate::catch_panic_status_v1(|| ($retire_request)(handle, request_id))
        }

        pub static $extended: $crate::BundleApiWithRequestRetirementV1 =
            $crate::BundleApiWithRequestRetirementV1 {
                base: $crate::BundleRawApiV1 {
                    struct_size: ::core::mem::size_of::<
                        $crate::BundleApiWithRequestRetirementV1,
                    >() as u32,
                    abi_version: $crate::BUNDLE_ABI_VERSION_V1,
                    create: Some(__bundle_create_trampoline_v1),
                    execute: Some(__bundle_execute_trampoline_v1),
                    commit: Some(__bundle_commit_trampoline_v1),
                    cancel: Some(__bundle_cancel_trampoline_v1),
                    destroy: Some(__bundle_destroy_trampoline_v1),
                },
                retire_request_v1: Some(__bundle_retire_request_trampoline_v1),
            };

        // A copied prefix preserves static callers without exposing trailing memory.
        pub static $api: $crate::BundleRawApiV1 = $extended.base;

        // SAFETY-REVIEWED: the entry returns the base of a static extended ABI table.
        #[allow(unsafe_code)]
        #[unsafe(no_mangle)]
        pub extern "C" fn chop_bundle_get_api_v1() -> *const $crate::BundleRawApiV1 {
            $crate::catch_panic_api_pointer_v1(|| &$extended.base)
        }
    };
    (with_speculation $api:ident, $extended:ident,
        $create:expr, $execute:expr, $commit:expr, $cancel:expr, $destroy:expr,
        $retire_request:expr, $commit_accepted:expr $(,)?) => {
        $crate::export_bundle_api_v1!(@callbacks $create, $execute, $commit, $cancel, $destroy);

        extern "C" fn __bundle_retire_request_trampoline_v1(
            handle: $crate::BundleHandleV1,
            request_id: u64,
        ) -> $crate::BundleStatusV1 {
            $crate::catch_panic_status_v1(|| ($retire_request)(handle, request_id))
        }

        // SAFETY-REVIEWED: the generated entry hands its pointer to the reviewed array decoder.
        #[allow(unsafe_code)]
        unsafe extern "C" fn __bundle_commit_accepted_trampoline_v1(
            handle: $crate::BundleHandleV1,
            records: *const $crate::BundleKeptRowsV1,
            record_count: u32,
        ) -> $crate::BundleStatusV1 {
            let callback = $commit_accepted;
            // SAFETY-REVIEWED: the native caller lends this fixed-width array for the callback.
            unsafe { $crate::dispatch_bundle_commit_accepted_v1(handle, records, record_count, callback) }
        }

        pub static $extended: $crate::BundleApiWithSpeculationV1 =
            $crate::BundleApiWithSpeculationV1 {
                base: $crate::BundleApiWithRequestRetirementV1 {
                    base: $crate::BundleRawApiV1 {
                        struct_size: ::core::mem::size_of::<$crate::BundleApiWithSpeculationV1>() as u32,
                        abi_version: $crate::BUNDLE_ABI_VERSION_V1,
                        create: Some(__bundle_create_trampoline_v1),
                        execute: Some(__bundle_execute_trampoline_v1),
                        commit: Some(__bundle_commit_trampoline_v1),
                        cancel: Some(__bundle_cancel_trampoline_v1),
                        destroy: Some(__bundle_destroy_trampoline_v1),
                    },
                    retire_request_v1: Some(__bundle_retire_request_trampoline_v1),
                },
                commit_accepted_v1: Some(__bundle_commit_accepted_trampoline_v1),
            };
        pub static $api: $crate::BundleRawApiV1 = $extended.base.base;

        // SAFETY-REVIEWED: the entry returns the base of a static extended ABI table.
        #[allow(unsafe_code)]
        #[unsafe(no_mangle)]
        pub extern "C" fn chop_bundle_get_api_v1() -> *const $crate::BundleRawApiV1 {
            $crate::catch_panic_api_pointer_v1(|| &$extended.base.base)
        }
    };
    (with_prefix_cache $api:ident, $extended:ident,
        $create:expr, $execute:expr, $commit:expr, $cancel:expr, $destroy:expr,
        $retire_request:expr, $commit_accepted:expr,
        $describe_prefix_state:expr, $capture_prefix_state:expr, $restore_prefix_state:expr $(,)?) => {
        $crate::export_bundle_api_v1!(@callbacks $create, $execute, $commit, $cancel, $destroy);

        extern "C" fn __bundle_retire_request_trampoline_v1(
            handle: $crate::BundleHandleV1,
            request_id: u64,
        ) -> $crate::BundleStatusV1 {
            $crate::catch_panic_status_v1(|| ($retire_request)(handle, request_id))
        }

        // SAFETY-REVIEWED: the generated entry hands its pointer to the reviewed array decoder.
        #[allow(unsafe_code)]
        unsafe extern "C" fn __bundle_commit_accepted_trampoline_v1(
            handle: $crate::BundleHandleV1,
            records: *const $crate::BundleKeptRowsV1,
            record_count: u32,
        ) -> $crate::BundleStatusV1 {
            let callback = $commit_accepted;
            // SAFETY-REVIEWED: the native caller lends this fixed-width array for the callback.
            unsafe { $crate::dispatch_bundle_commit_accepted_v1(handle, records, record_count, callback) }
        }

        extern "C" fn __bundle_describe_prefix_state_trampoline_v1(
            handle: $crate::BundleHandleV1,
        ) -> $crate::BundlePrefixStateInfoV1 {
            $crate::catch_panic_prefix_state_info_v1(|| ($describe_prefix_state)(handle))
        }

        // SAFETY-REVIEWED: the generated entry hands its record to the reviewed byte-view decoder.
        #[allow(unsafe_code)]
        unsafe extern "C" fn __bundle_capture_prefix_state_trampoline_v1(
            handle: $crate::BundleHandleV1,
            state: $crate::BundlePrefixStateV1,
        ) -> $crate::BundleStatusV1 {
            let callback = $capture_prefix_state;
            // SAFETY-REVIEWED: the native caller lends the record's byte views for the callback.
            unsafe { $crate::dispatch_bundle_prefix_state_v1(handle, state, callback) }
        }

        // SAFETY-REVIEWED: the generated entry hands its record to the reviewed byte-view decoder.
        #[allow(unsafe_code)]
        unsafe extern "C" fn __bundle_restore_prefix_state_trampoline_v1(
            handle: $crate::BundleHandleV1,
            state: $crate::BundlePrefixStateV1,
        ) -> $crate::BundleStatusV1 {
            let callback = $restore_prefix_state;
            // SAFETY-REVIEWED: the native caller lends the record's byte views for the callback.
            unsafe { $crate::dispatch_bundle_prefix_state_v1(handle, state, callback) }
        }

        pub static $extended: $crate::BundleApiWithPrefixCacheV1 =
            $crate::BundleApiWithPrefixCacheV1 {
                base: $crate::BundleApiWithSpeculationV1 {
                    base: $crate::BundleApiWithRequestRetirementV1 {
                        base: $crate::BundleRawApiV1 {
                            struct_size: ::core::mem::size_of::<$crate::BundleApiWithPrefixCacheV1>() as u32,
                            abi_version: $crate::BUNDLE_ABI_VERSION_V1,
                            create: Some(__bundle_create_trampoline_v1),
                            execute: Some(__bundle_execute_trampoline_v1),
                            commit: Some(__bundle_commit_trampoline_v1),
                            cancel: Some(__bundle_cancel_trampoline_v1),
                            destroy: Some(__bundle_destroy_trampoline_v1),
                        },
                        retire_request_v1: Some(__bundle_retire_request_trampoline_v1),
                    },
                    commit_accepted_v1: Some(__bundle_commit_accepted_trampoline_v1),
                },
                describe_prefix_state_v1: Some(__bundle_describe_prefix_state_trampoline_v1),
                capture_prefix_state_v1: Some(__bundle_capture_prefix_state_trampoline_v1),
                restore_prefix_state_v1: Some(__bundle_restore_prefix_state_trampoline_v1),
            };
        pub static $api: $crate::BundleRawApiV1 = $extended.base.base.base;

        // SAFETY-REVIEWED: the entry returns the base of a static extended ABI table.
        #[allow(unsafe_code)]
        #[unsafe(no_mangle)]
        pub extern "C" fn chop_bundle_get_api_v1() -> *const $crate::BundleRawApiV1 {
            $crate::catch_panic_api_pointer_v1(|| &$extended.base.base.base)
        }
    };
    (with_prefix_state_slot $api:ident, $extended:ident,
        $create:expr, $execute:expr, $commit:expr, $cancel:expr, $destroy:expr,
        $retire_request:expr, [$($commit_accepted:expr)?],
        $describe_prefix_state:expr, $capture_prefix_state:expr, $restore_prefix_state:expr,
        $restore_prefix_state_in_slot:expr $(,)?) => {
        $crate::export_bundle_api_v1!(@callbacks $create, $execute, $commit, $cancel, $destroy);

        extern "C" fn __bundle_retire_request_trampoline_v1(
            handle: $crate::BundleHandleV1,
            request_id: u64,
        ) -> $crate::BundleStatusV1 {
            $crate::catch_panic_status_v1(|| ($retire_request)(handle, request_id))
        }

        $(
            // SAFETY-REVIEWED: the generated entry hands its pointer to the reviewed array decoder.
            #[allow(unsafe_code)]
            unsafe extern "C" fn __bundle_commit_accepted_trampoline_v1(
                handle: $crate::BundleHandleV1,
                records: *const $crate::BundleKeptRowsV1,
                record_count: u32,
            ) -> $crate::BundleStatusV1 {
                let callback = $commit_accepted;
                // SAFETY-REVIEWED: the native caller lends this fixed-width array for the callback.
                unsafe { $crate::dispatch_bundle_commit_accepted_v1(handle, records, record_count, callback) }
            }
        )?

        extern "C" fn __bundle_describe_prefix_state_trampoline_v1(
            handle: $crate::BundleHandleV1,
        ) -> $crate::BundlePrefixStateInfoV1 {
            $crate::catch_panic_prefix_state_info_v1(|| ($describe_prefix_state)(handle))
        }

        // SAFETY-REVIEWED: the generated entry hands its record to the reviewed byte-view decoder.
        #[allow(unsafe_code)]
        unsafe extern "C" fn __bundle_capture_prefix_state_trampoline_v1(
            handle: $crate::BundleHandleV1,
            state: $crate::BundlePrefixStateV1,
        ) -> $crate::BundleStatusV1 {
            let callback = $capture_prefix_state;
            // SAFETY-REVIEWED: the native caller lends the record's byte views for the callback.
            unsafe { $crate::dispatch_bundle_prefix_state_v1(handle, state, callback) }
        }

        // SAFETY-REVIEWED: the generated entry hands its record to the reviewed byte-view decoder.
        #[allow(unsafe_code)]
        unsafe extern "C" fn __bundle_restore_prefix_state_trampoline_v1(
            handle: $crate::BundleHandleV1,
            state: $crate::BundlePrefixStateV1,
        ) -> $crate::BundleStatusV1 {
            let callback = $restore_prefix_state;
            // SAFETY-REVIEWED: the native caller lends the record's byte views for the callback.
            unsafe { $crate::dispatch_bundle_prefix_state_v1(handle, state, callback) }
        }

        // SAFETY-REVIEWED: the generated entry hands its record to the reviewed byte-view decoder.
        #[allow(unsafe_code)]
        unsafe extern "C" fn __bundle_restore_prefix_state_in_slot_trampoline_v1(
            handle: $crate::BundleHandleV1,
            state: $crate::BundlePrefixStateV1,
            state_slot: u64,
        ) -> $crate::BundleStatusV1 {
            let callback = $restore_prefix_state_in_slot;
            // SAFETY-REVIEWED: the native caller lends the record's byte views for the callback.
            unsafe {
                $crate::dispatch_bundle_prefix_state_v1(handle, state, |handle, request| {
                    callback(handle, request, state_slot)
                })
            }
        }

        pub static $extended: $crate::BundleApiWithPrefixStateSlotV1 =
            $crate::BundleApiWithPrefixStateSlotV1 {
                base: $crate::BundleApiWithPrefixCacheV1 {
                    base: $crate::BundleApiWithSpeculationV1 {
                        base: $crate::BundleApiWithRequestRetirementV1 {
                            base: $crate::BundleRawApiV1 {
                                struct_size: ::core::mem::size_of::<$crate::BundleApiWithPrefixStateSlotV1>() as u32,
                                abi_version: $crate::BUNDLE_ABI_VERSION_V1,
                                create: Some(__bundle_create_trampoline_v1),
                                execute: Some(__bundle_execute_trampoline_v1),
                                commit: Some(__bundle_commit_trampoline_v1),
                                cancel: Some(__bundle_cancel_trampoline_v1),
                                destroy: Some(__bundle_destroy_trampoline_v1),
                            },
                            retire_request_v1: Some(__bundle_retire_request_trampoline_v1),
                        },
                        commit_accepted_v1: $crate::export_bundle_api_v1!(@commit_accepted $($commit_accepted)?),
                    },
                    describe_prefix_state_v1: Some(__bundle_describe_prefix_state_trampoline_v1),
                    capture_prefix_state_v1: Some(__bundle_capture_prefix_state_trampoline_v1),
                    restore_prefix_state_v1: Some(__bundle_restore_prefix_state_trampoline_v1),
                },
                restore_prefix_state_in_slot_v1: Some(__bundle_restore_prefix_state_in_slot_trampoline_v1),
            };
        pub static $api: $crate::BundleRawApiV1 = $extended.base.base.base.base;

        // SAFETY-REVIEWED: the entry returns the base of a static extended ABI table.
        #[allow(unsafe_code)]
        #[unsafe(no_mangle)]
        pub extern "C" fn chop_bundle_get_api_v1() -> *const $crate::BundleRawApiV1 {
            $crate::catch_panic_api_pointer_v1(|| &$extended.base.base.base.base)
        }
    };
    (@commit_accepted) => {
        None
    };
    (@commit_accepted $commit_accepted:expr) => {
        Some(__bundle_commit_accepted_trampoline_v1)
    };
    (@callbacks $create:expr, $execute:expr, $commit:expr, $cancel:expr, $destroy:expr) => {
        // SAFETY-REVIEWED: this is the only unsafe item the macro expands into a bundle crate.
        #[allow(unsafe_code)]
        // SAFETY-REVIEWED: the generated create entry owns the V1 pointer decode for its caller.
        unsafe extern "C" fn __bundle_create_trampoline_v1(
            host_services: *const $crate::BundleHostServicesV1,
            configuration: $crate::BundleByteViewV1,
        ) -> $crate::BundleCreateResultV1 {
            let create = $create;
            // SAFETY-REVIEWED: these are the exact arguments the native create call supplied.
            unsafe { $crate::dispatch_bundle_create_v1(host_services, configuration, create) }
        }

        // SAFETY-REVIEWED: this is the only unsafe item the macro expands into a bundle crate.
        #[allow(unsafe_code)]
        // SAFETY-REVIEWED: the generated execute entry owns the V1 pointer decode for its caller.
        unsafe extern "C" fn __bundle_execute_trampoline_v1(
            handle: $crate::BundleHandleV1,
            request: $crate::BundleByteViewV1,
            result_sink: $crate::BundleExecuteResultSinkV1,
        ) -> $crate::BundleStatusV1 {
            let execute = $execute;
            // SAFETY-REVIEWED: these are the exact arguments the native execute call supplied.
            unsafe { $crate::dispatch_bundle_execute_v1(handle, request, result_sink, execute) }
        }

        extern "C" fn __bundle_commit_trampoline_v1(
            handle: $crate::BundleHandleV1,
        ) -> $crate::BundleStatusV1 {
            $crate::catch_panic_status_v1(|| ($commit)(handle))
        }

        extern "C" fn __bundle_cancel_trampoline_v1(
            handle: $crate::BundleHandleV1,
        ) -> $crate::BundleStatusV1 {
            $crate::catch_panic_status_v1(|| ($cancel)(handle))
        }

        extern "C" fn __bundle_destroy_trampoline_v1(
            handle: $crate::BundleHandleV1,
        ) -> $crate::BundleStatusV1 {
            $crate::catch_panic_status_v1(|| ($destroy)(handle))
        }
    };
}

/// Defines one custom operation whose handler is ordinary safe Rust.
///
/// The macro expands to the C entry a host invokes and to a safe constructor
/// named `$name`, which borrows the operation's own state and returns the
/// [`CustomOperationV1`] to hand the host. The entry decodes the lent record,
/// borrows that state, and calls `$handler` with both; a panic is contained as
/// [`BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT`] before it can reach
/// native code. The borrow is what keeps the state alive: an operation cannot
/// outlive the value its entry will be handed, and the crate that writes the
/// handler contains no unsafe code.
#[macro_export]
macro_rules! export_custom_operation_v1 {
    ($name:ident, $state:ty, $handler:expr $(,)?) => {
        /// Binds this operation's generated entry to one live state value.
        // SAFETY-REVIEWED: this is the only unsafe item the macro expands into its caller.
        #[allow(unsafe_code)]
        fn $name(state: &mut $state) -> $crate::CustomOperationV1<'_> {
            // SAFETY-REVIEWED: the generated entry owns the V1 pointer decode for its host.
            unsafe extern "C" fn entry(
                operation_user_data: *mut ::core::ffi::c_void,
                launch: *const $crate::BundleCustomLaunchV1,
            ) -> i32 {
                let handler = $handler;
                // SAFETY-REVIEWED: these are the exact arguments the native call supplied.
                unsafe {
                    $crate::dispatch_custom_operation_v1::<$state>(
                        operation_user_data,
                        launch,
                        handler,
                    )
                }
            }

            // SAFETY-REVIEWED: that entry reads exactly this state type, and the borrow outlives the value.
            unsafe { $crate::CustomOperationV1::new(entry, ::core::ptr::from_mut(state).cast()) }
        }
    };
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::ptr;
    use std::rc::Rc;

    use super::{
        HostServicesTableV1, HostServicesV1, dispatch_custom_operation_v1, with_result_sink_v1,
    };
    use crate::host_services::{
        HostServiceResultV1, LentLaunchV1, LogitsRowSinkV1, OwnedDraftTokensV1, OwnedLogitsRowV1,
        OwnedLogitsViewV1, RankArenaUploadV1, ResultSinkLimitsV1, UnsupportedHostServicesBackendV1,
    };
    use crate::retain_host_services_for_process_lifetime_v1;
    use crate::{
        BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT, BundleAbiError,
        BundleAuthenticatedComponentHandleV1, BundleByteViewV1, BundleContextHandleV1,
        BundleCreateResultV1, BundleCudaSymbolHandleV1, BundleDeviceBufferHandleV1,
        BundleExecuteResultSinkV1, BundleHandleV1, BundleHostServicesV1, BundleKeptRowsV1,
        BundleKernelArgumentV1, BundleLaunchConfigV1, BundleLogitsRowV1, BundleLogitsViewV1,
        BundleMutByteViewV1, BundlePhysicalKvHandleV1, BundleRawBatchLimitsV1,
        BundleRawSpeculationV1, BundleStatusV1, BundleStreamHandleV1, ExecuteResultSinkV1,
    };

    fn panic_create(_host_services: HostServicesV1, _configuration: &[u8]) -> BundleCreateResultV1 {
        panic!("create panic")
    }

    fn panic_execute(
        _handle: BundleHandleV1,
        _request: &[u8],
        _result_sink: &mut ExecuteResultSinkV1,
    ) -> BundleStatusV1 {
        panic!("execute panic")
    }

    fn panic_status(_handle: BundleHandleV1) -> BundleStatusV1 {
        panic!("status panic")
    }

    crate::export_bundle_api_v1!(
        TEST_API,
        panic_create,
        panic_execute,
        panic_status,
        panic_status,
        panic_status,
    );

    #[test]
    fn generated_callbacks_contain_panics() {
        let handle = BundleHandleV1::from_parts(1, 1);
        let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
        let create = TEST_API.create.expect("test create callback exists");
        let execute = TEST_API.execute.expect("test execute callback exists");
        // SAFETY-REVIEWED: the table pins a live record and the empty views carry no pointer.
        let create_result = unsafe { create(table.raw_pointer(), BundleByteViewV1::empty()) };
        let create_status = create_result.status;
        // SAFETY-REVIEWED: the fixture sink and the empty request view live for this call.
        let execute_status =
            unsafe { execute(handle, BundleByteViewV1::empty(), fixture_result_sink()) };

        assert_eq!(create_status, BundleStatusV1::Panic);
        assert_eq!(execute_status, BundleStatusV1::Panic);
        assert_eq!(
            TEST_API.commit.expect("test commit callback exists")(handle),
            BundleStatusV1::Panic
        );
        assert_eq!(
            TEST_API.cancel.expect("test cancel callback exists")(handle),
            BundleStatusV1::Panic
        );
        assert_eq!(
            TEST_API.destroy.expect("test destroy callback exists")(handle),
            BundleStatusV1::Panic
        );
        assert!(!chop_bundle_get_api_v1().is_null());
    }

    #[test]
    fn create_rejects_a_null_service_table_before_the_bundle_runs() {
        let create = TEST_API.create.expect("test create callback exists");
        // SAFETY-REVIEWED: a null table is the rejected input this check exercises.
        let result = unsafe { create(std::ptr::null(), BundleByteViewV1::empty()) };

        assert_eq!(result.status, BundleStatusV1::InvalidArgument);
        assert!(!result.handle.has_generation());
    }

    #[test]
    fn service_trampolines_reject_invalid_pointers_from_a_native_caller() {
        let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
        let services = *table.services();
        let user_data = services.user_data;
        let select_context = services.select_context.expect("context service exists");
        let allocate_physical_kv = services
            .allocate_physical_kv
            .expect("physical KV service exists");
        let launch = services.launch_cuda_symbol.expect("launch service exists");
        let log = services.log.expect("log service exists");
        let open_component = services
            .open_authenticated_component
            .expect("authenticated component service exists");
        let inspect_component = services
            .inspect_authenticated_component
            .expect("authenticated component inspection exists");
        let read_component = services
            .read_authenticated_component
            .expect("authenticated component read exists");
        let mut context = BundleContextHandleV1::from_raw(0);
        let mut kv = BundlePhysicalKvHandleV1::from_raw(0);
        let config = BundleLaunchConfigV1 {
            grid_x: 1,
            grid_y: 1,
            grid_z: 1,
            block_x: 1,
            block_y: 1,
            block_z: 1,
            ..BundleLaunchConfigV1::empty()
        };
        let arguments = [BundleKernelArgumentV1::empty()];

        // SAFETY-REVIEWED: each call below passes the rejected pointer shape this check exercises.
        unsafe {
            assert_eq!(
                select_context(user_data, 2, 3, ptr::null_mut()),
                BundleStatusV1::InvalidArgument,
                "a null output pointer is rejected before the backend runs"
            );
            assert_eq!(
                select_context(ptr::null_mut(), 2, 3, &mut context),
                BundleStatusV1::InvalidArgument,
                "a null user_data pointer is rejected before the backend runs"
            );
            assert_eq!(
                allocate_physical_kv(user_data, ptr::null(), &mut kv),
                BundleStatusV1::InvalidArgument,
                "a null input record is rejected before the backend runs"
            );
            assert_eq!(
                launch(
                    user_data,
                    BundleStreamHandleV1::from_parts(1, 1),
                    BundleCudaSymbolHandleV1::from_parts(1, 1),
                    &config,
                    arguments.as_ptr(),
                    0,
                ),
                BundleStatusV1::InvalidArgument,
                "a nonempty pointer with a zero count is an invalid pair"
            );
            assert_eq!(
                log(
                    user_data,
                    BundleByteViewV1 {
                        data: ptr::null(),
                        length: 4,
                    },
                ),
                BundleStatusV1::InvalidArgument,
                "a null view with a nonzero length is an invalid pair"
            );
            assert_eq!(
                open_component(user_data, BundleByteViewV1::empty(), ptr::null_mut(),),
                BundleStatusV1::InvalidArgument,
                "a null authenticated component output is rejected"
            );
            assert_eq!(
                inspect_component(
                    user_data,
                    BundleAuthenticatedComponentHandleV1::from_parts(1, 1),
                    ptr::null_mut(),
                ),
                BundleStatusV1::InvalidArgument,
                "a null authenticated metadata output is rejected"
            );
            assert_eq!(
                read_component(
                    user_data,
                    BundleAuthenticatedComponentHandleV1::from_parts(1, 1),
                    BundleMutByteViewV1 {
                        data: ptr::null_mut(),
                        length: 1,
                    },
                    BundleMutByteViewV1::empty(),
                ),
                BundleStatusV1::InvalidArgument,
                "an invalid authenticated component destination is rejected"
            );
        }
    }

    /// An array element with a foreign stride is refused before it is walked.
    ///
    /// This is the failure the exact-length rule exists for. A caller whose
    /// kernel-argument record grew a tail has an array stride longer than this
    /// reader's, so stepping by one V1 record would put the second element
    /// inside that caller's first. The first record's declared length is read on
    /// its own, out of the four bytes every version of the record opens with,
    /// and a length that is not this one's refuses the call before anything
    /// advances by a record at all.
    #[test]
    fn an_argument_array_whose_stride_is_not_this_one_is_refused_before_it_is_walked() {
        let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
        let services = *table.services();
        let user_data = services.user_data;
        let launch = services.launch_cuda_symbol.expect("launch service exists");
        let config = BundleLaunchConfigV1 {
            grid_x: 1,
            grid_y: 1,
            grid_z: 1,
            block_x: 1,
            block_y: 1,
            block_z: 1,
            cluster_x: 1,
            cluster_y: 1,
            cluster_z: 1,
            ..BundleLaunchConfigV1::empty()
        };
        let exact = crate::struct_size_v1::<BundleKernelArgumentV1>();

        let mut first = BundleKernelArgumentV1::empty();
        first.kind = crate::BUNDLE_KERNEL_ARGUMENT_V1_U32;
        let second = first;
        // Two records this reader could otherwise walk, whose first one declares
        // the longer length a grown tail would have given the whole array.
        first.struct_size = exact + 8;
        let grown = [first, second];
        let exactly = [second, second];

        // SAFETY-REVIEWED: each call passes a live two-element array and a live config record.
        unsafe {
            assert_eq!(
                launch(
                    user_data,
                    BundleStreamHandleV1::from_parts(1, 1),
                    BundleCudaSymbolHandleV1::from_parts(1, 1),
                    &config,
                    grown.as_ptr(),
                    2,
                ),
                BundleStatusV1::InvalidArgument,
                "an element declaring a length this array is not strided by is refused"
            );
            // The same two elements at this version's own length reach the
            // backend, which is what says the refusal above was the length and
            // not the array.
            assert_eq!(
                launch(
                    user_data,
                    BundleStreamHandleV1::from_parts(1, 1),
                    BundleCudaSymbolHandleV1::from_parts(1, 1),
                    &config,
                    exactly.as_ptr(),
                    2,
                ),
                BundleStatusV1::Unsupported,
                "the same array at the exact V1 length reaches the backend"
            );
        }
    }

    #[test]
    fn a_custom_operation_without_an_entry_or_an_output_never_runs() {
        let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
        let services = *table.services();
        let user_data = services.user_data;
        let run = services
            .run_custom_operation
            .expect("custom operation service exists");
        let context = BundleContextHandleV1::from_parts(1, 1);
        let stream = BundleStreamHandleV1::from_parts(1, 1);
        let operands = [crate::BundleCustomOperandV1::empty()];
        let mut status = 0_i32;

        // SAFETY-REVIEWED: each call below passes the rejected shape this check exercises.
        unsafe {
            assert_eq!(
                run(
                    user_data,
                    context,
                    stream,
                    0,
                    BundleByteViewV1::empty(),
                    ptr::null(),
                    0,
                    None,
                    ptr::null_mut(),
                    &raw mut status,
                ),
                BundleStatusV1::InvalidArgument,
                "an operation with no entry is refused before anything is resolved"
            );
            assert_eq!(
                run(
                    user_data,
                    context,
                    stream,
                    0,
                    BundleByteViewV1::empty(),
                    ptr::null(),
                    0,
                    Some(refusing_custom_entry),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ),
                BundleStatusV1::InvalidArgument,
                "a null status output is refused before the operation runs"
            );
            assert_eq!(
                run(
                    user_data,
                    context,
                    stream,
                    0,
                    BundleByteViewV1::empty(),
                    operands.as_ptr(),
                    0,
                    Some(refusing_custom_entry),
                    ptr::null_mut(),
                    &raw mut status,
                ),
                BundleStatusV1::InvalidArgument,
                "a nonempty operand pointer with a zero count is an invalid pair"
            );
        }
        assert_eq!(status, 0, "no refusal writes an operation status");
    }

    // SAFETY-REVIEWED: this fixture entry reads nothing and exists to prove it is never called.
    unsafe extern "C" fn refusing_custom_entry(
        _user_data: *mut c_void,
        _launch: *const crate::BundleCustomLaunchV1,
    ) -> i32 {
        unreachable!("a refused custom operation is never invoked")
    }

    /// An operand array with a foreign stride is refused before it is walked.
    ///
    /// The custom operand is the second array element this ABI has, and
    /// `run_custom_operation` carries no stride for it either. A caller whose
    /// operand record grew a tail would have a longer array stride, so stepping
    /// by one V1 record would put the second operand inside that caller's
    /// first, and the span the host then resolved and alias checked would be
    /// built from bytes that mean something else. The first record's declared
    /// length is read on its own and refuses the call before anything advances.
    ///
    /// The control is the same two operands at the exact V1 length: they reach
    /// the backend, which is what says the refusal was the length rather than
    /// the array. This backend answers every service unsupported, so reaching it
    /// is the furthest a well formed call gets here.
    #[test]
    fn an_operand_array_whose_stride_is_not_this_one_is_refused_before_it_is_walked() {
        let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
        let services = *table.services();
        let user_data = services.user_data;
        let run = services
            .run_custom_operation
            .expect("custom operation service exists");
        let context = BundleContextHandleV1::from_parts(1, 1);
        let stream = BundleStreamHandleV1::from_parts(1, 1);
        let exact = crate::struct_size_v1::<crate::BundleCustomOperandV1>();

        let mut first = crate::BundleCustomOperandV1::empty();
        first.buffer = crate::BundleDeviceBufferViewV1 {
            buffer: crate::BundleDeviceBufferHandleV1::from_parts(1, 1),
            offset: 0,
            length: 64,
            ..crate::BundleDeviceBufferViewV1::empty()
        };
        let second = first;
        let mut status = 0_i32;

        for declared in [exact + 8, exact - 1] {
            let mut grown = first;
            grown.struct_size = declared;
            let operands = [grown, second];
            // SAFETY-REVIEWED: the operand array lives in this frame for the length of the call.
            let refusal = unsafe {
                run(
                    user_data,
                    context,
                    stream,
                    0,
                    BundleByteViewV1::empty(),
                    operands.as_ptr(),
                    2,
                    Some(refusing_custom_entry),
                    ptr::null_mut(),
                    &raw mut status,
                )
            };
            assert_eq!(
                refusal,
                BundleStatusV1::InvalidArgument,
                "an operand declaring {declared} rather than {exact} must be refused"
            );
            assert_eq!(status, 0, "a refused operand array writes no status");
        }

        let exactly = [first, second];
        // SAFETY-REVIEWED: the operand array lives in this frame for the length of the call.
        let reached = unsafe {
            run(
                user_data,
                context,
                stream,
                0,
                BundleByteViewV1::empty(),
                exactly.as_ptr(),
                2,
                Some(refusing_custom_entry),
                ptr::null_mut(),
                &raw mut status,
            )
        };
        assert_eq!(
            reached,
            BundleStatusV1::Unsupported,
            "the same operands at the exact V1 length reach the backend"
        );
    }

    /// A launch record over two `u64` arrays, at whatever alignment is asked.
    ///
    /// The arrays live in one oversized buffer so a misaligned view of them can
    /// be built without allocating at an address no allocator would return.
    struct HostileLaunchV1 {
        storage: [u64; 8],
        skew: usize,
    }

    impl HostileLaunchV1 {
        const fn new(skew: usize) -> Self {
            Self {
                storage: [0x1000, 0x2000, 64, 64, 0, 0, 0, 0],
                skew,
            }
        }

        /// The address and length arrays, offset by this record's skew.
        fn arrays(&self) -> (*const u64, *const u64) {
            let base = ptr::from_ref(&self.storage).cast::<u8>();
            // SAFETY-REVIEWED: both offsets stay inside the storage above, and neither is read here.
            let (addresses, lengths) = unsafe { (base.add(self.skew), base.add(16 + self.skew)) };
            (addresses.cast::<u64>(), lengths.cast::<u64>())
        }

        fn record(&self, stream: *mut c_void) -> crate::BundleCustomLaunchV1 {
            let (addresses, lengths) = self.arrays();
            crate::BundleCustomLaunchV1 {
                address_space: crate::BUNDLE_CUSTOM_ADDRESS_SPACE_V1_DEVICE,
                operand_count: 2,
                stream,
                operand_addresses: addresses,
                operand_lengths: lengths,
                ..crate::BundleCustomLaunchV1::empty()
            }
        }
    }

    #[test]
    fn a_misaligned_or_contradictory_launch_record_is_refused_before_it_is_read() {
        let mut state = 0_u32;
        let handler = |state: &mut u32, launch: LentLaunchV1<'_>| {
            *state += 1;
            i32::try_from(launch.ranges.len()).unwrap_or(0)
        };
        let state_pointer = ptr::from_mut(&mut state).cast::<c_void>();
        let stream = ptr::from_mut(&mut state).cast::<c_void>();
        let aligned = HostileLaunchV1::new(0);
        let misaligned = HostileLaunchV1::new(1);

        let readable = aligned.record(stream);
        let skewed = misaligned.record(stream);
        let streamless = aligned.record(ptr::null_mut());
        // SAFETY-REVIEWED: each call below passes the exact record shape this check exercises.
        unsafe {
            assert_eq!(
                dispatch_custom_operation_v1::<u32>(state_pointer, &raw const readable, handler),
                2,
                "an aligned record with two device ranges is the readable one"
            );

            // One byte of skew makes both arrays unreadable as `u64`, and the
            // entry refuses rather than reading at that address.
            assert_eq!(
                dispatch_custom_operation_v1::<u32>(state_pointer, &raw const skewed, handler),
                BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT,
                "a misaligned operand array is refused rather than read"
            );

            // The lengths array alone misaligned is refused for the same reason.
            let mut lengths_only = aligned.record(stream);
            lengths_only.operand_lengths = misaligned.arrays().1;
            assert_eq!(
                dispatch_custom_operation_v1::<u32>(
                    state_pointer,
                    &raw const lengths_only,
                    handler,
                ),
                BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT,
                "a misaligned length array is refused rather than read"
            );

            // A device lend with no stream, and a host lend that carries one,
            // are both records that contradict what they claim to be.
            assert_eq!(
                dispatch_custom_operation_v1::<u32>(state_pointer, &raw const streamless, handler),
                BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT,
                "a device lend without a stream is refused"
            );
            let mut host_space = aligned.record(stream);
            host_space.address_space = crate::BUNDLE_CUSTOM_ADDRESS_SPACE_V1_HOST;
            assert_eq!(
                dispatch_custom_operation_v1::<u32>(state_pointer, &raw const host_space, handler),
                BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT,
                "a host lend that carries a stream is refused"
            );

            // A range with no address, no length, or an end that does not fit
            // is not a range an operation can bound its own work by.
            for storage in [
                [0_u64, 0x2000, 64, 64, 0, 0, 0, 0],
                [0x1000, 0x2000, 0, 64, 0, 0, 0, 0],
                [u64::MAX, 0x2000, 64, 64, 0, 0, 0, 0],
            ] {
                let hostile = HostileLaunchV1 { storage, skew: 0 };
                let record = hostile.record(stream);
                assert_eq!(
                    dispatch_custom_operation_v1::<u32>(state_pointer, &raw const record, handler),
                    BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT,
                    "a range of {storage:?} is refused rather than lent"
                );
            }
        }

        assert_eq!(state, 1, "only the readable record reached the handler");
    }

    #[test]
    fn an_unknown_completion_flag_is_refused_before_anything_is_resolved() {
        let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
        let services = *table.services();
        let run = services
            .run_custom_operation
            .expect("custom operation service exists");
        let mut status = 0_i32;

        // SAFETY-REVIEWED: the entry below is never reached, which is what this check proves.
        let refused = unsafe {
            run(
                services.user_data,
                BundleContextHandleV1::from_parts(1, 1),
                BundleStreamHandleV1::from_parts(1, 1),
                crate::BUNDLE_CUSTOM_OPERATION_FLAGS_V1_KNOWN | 0x8000_0000,
                BundleByteViewV1::empty(),
                ptr::null(),
                0,
                Some(refusing_custom_entry),
                ptr::null_mut(),
                &raw mut status,
            )
        };

        assert_eq!(refused, BundleStatusV1::InvalidArgument);
        assert_eq!(status, 0, "a refusal writes no operation status");
    }

    /// A host that reports a bundle failure and writes no code for it.
    // SAFETY-REVIEWED: this fixture writes nothing through any pointer it receives.
    unsafe extern "C" fn silent_bundle_error_service(
        _user_data: *mut c_void,
        _context: BundleContextHandleV1,
        _stream: BundleStreamHandleV1,
        _flags: u32,
        _operation_id: BundleByteViewV1,
        _operands: *const crate::BundleCustomOperandV1,
        _operand_count: u64,
        _operation: Option<crate::BundleCustomOperationFnV1>,
        _operation_user_data: *mut c_void,
        _out_operation_status: *mut i32,
    ) -> BundleStatusV1 {
        BundleStatusV1::BundleError
    }

    /// A host that reports a bundle failure and calls it success at the same time.
    // SAFETY-REVIEWED: the out parameter is the one the caller's own frame supplied.
    unsafe extern "C" fn zero_bundle_error_service(
        _user_data: *mut c_void,
        _context: BundleContextHandleV1,
        _stream: BundleStreamHandleV1,
        _flags: u32,
        _operation_id: BundleByteViewV1,
        _operands: *const crate::BundleCustomOperandV1,
        _operand_count: u64,
        _operation: Option<crate::BundleCustomOperationFnV1>,
        _operation_user_data: *mut c_void,
        out_operation_status: *mut i32,
    ) -> BundleStatusV1 {
        // SAFETY-REVIEWED: the safe caller passes storage from its own live frame.
        unsafe { ptr::write(out_operation_status, 0) };
        BundleStatusV1::BundleError
    }

    #[test]
    fn a_bundle_failure_without_an_operation_code_is_not_read_as_success() {
        let mut state = 0_u32;
        let operation = fixture_state_operation(&mut state);
        let call = |entry| {
            let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
            let mut raw = *table.services();
            raw.run_custom_operation = Some(entry);
            // SAFETY-REVIEWED: the table stays live for this call and only its custom entry changed.
            let services = unsafe { HostServicesV1::new(raw) }.expect("the table is complete");
            let answer = services.run_custom_operation(
                BundleContextHandleV1::from_parts(1, 1),
                BundleStreamHandleV1::from_parts(1, 1),
                crate::CustomCompletionV1::Synchronous,
                b"fixture",
                &[],
                &operation,
            );
            retain_host_services_for_process_lifetime_v1(table);
            answer
        };

        // A host that writes nothing leaves the storage at the one code that
        // means "no code of its own", so the failure never reads as success.
        assert_eq!(
            call(silent_bundle_error_service),
            Err(BundleAbiError::CustomOperation(
                BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT
            ))
        );
        // Zero is the operation's word for success, so a failure that carries
        // it is one this caller refuses to read as either.
        assert_eq!(
            call(zero_bundle_error_service),
            Err(BundleAbiError::MissingCustomOperationStatus)
        );
        assert_eq!(state, 0, "no hostile answer invoked the operation");
    }

    crate::export_custom_operation_v1!(fixture_state_operation, u32, |state: &mut u32,
                                                                      _launch: LentLaunchV1<
        '_,
    >| {
        *state += 1;
        0
    });

    #[test]
    fn a_custom_operation_entry_refuses_a_record_it_cannot_read() {
        let mut state = 0_u32;
        let handler = |state: &mut u32, launch: LentLaunchV1<'_>| {
            *state += 1;
            i32::try_from(launch.ranges.len()).unwrap_or(0)
        };
        let mut record = crate::BundleCustomLaunchV1 {
            address_space: crate::BUNDLE_CUSTOM_ADDRESS_SPACE_V1_HOST,
            ..crate::BundleCustomLaunchV1::empty()
        };
        let state_pointer = ptr::from_mut(&mut state).cast::<c_void>();

        // SAFETY-REVIEWED: each call below passes the exact record shape this check exercises.
        unsafe {
            assert_eq!(
                dispatch_custom_operation_v1::<u32>(state_pointer, &raw const record, handler),
                0,
                "an empty operand list is a record this entry can read"
            );
            let mut mismatched = record;
            mismatched.operand_count = 2;
            assert_eq!(
                dispatch_custom_operation_v1::<u32>(state_pointer, &raw const mismatched, handler),
                BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT,
                "a count with no arrays behind it is refused rather than read"
            );
            record.struct_size = 0;
            assert_eq!(
                dispatch_custom_operation_v1::<u32>(state_pointer, &raw const record, handler),
                BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT,
                "a record older than V1 is refused rather than read"
            );
            assert_eq!(
                dispatch_custom_operation_v1::<u32>(state_pointer, ptr::null(), handler),
                BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT,
                "a null record is refused rather than read"
            );
            assert_eq!(
                dispatch_custom_operation_v1::<u32>(ptr::null_mut(), &raw const record, handler),
                BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT,
                "a null state is refused rather than borrowed"
            );
        }
        assert_eq!(state, 1, "only the readable record reached the handler");
    }

    #[test]
    fn the_service_record_keeps_its_address_when_the_table_moves() {
        let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
        let record = table.raw_pointer();
        assert!(std::ptr::eq(table.services(), record));

        let moved = table;

        assert_eq!(moved.raw_pointer(), record);
        assert_eq!(moved.backend_reference_count(), 1);
    }

    #[derive(Debug, Default)]
    struct CountingSinkV1 {
        rows: u64,
        views: u64,
        drafts: Vec<OwnedDraftTokensV1>,
    }

    impl LogitsRowSinkV1 for CountingSinkV1 {
        fn limits(&self) -> ResultSinkLimitsV1 {
            ResultSinkLimitsV1::new(1, 2, 2)
        }

        fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
            self.rows += 1;
            Ok(())
        }

        fn emit_view(&mut self, _view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
            self.views += 1;
            Ok(())
        }

        fn emit_draft_tokens(&mut self, drafts: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
            self.drafts.push(drafts);
            Ok(())
        }
    }

    #[test]
    fn the_host_sink_rejects_rows_past_its_declared_limits() {
        let logit_bits = [0_u8, 0, 128, 63, 0, 0, 0, 64];
        let mut sink = CountingSinkV1::default();

        with_result_sink_v1(&mut sink, |raw| {
            let emit = raw.emit_logits_row.expect("result sink callback exists");
            let row =
                BundleLogitsRowV1::from_logit_bits(9, &logit_bits).expect("fixture row is valid");
            // SAFETY-REVIEWED: the sink is live for this closure and the row borrows local bytes.
            unsafe {
                assert_eq!(emit(raw.user_data, row), BundleStatusV1::Success);
                assert_eq!(
                    emit(raw.user_data, row),
                    BundleStatusV1::ResourceExhausted,
                    "a native bundle cannot exceed the row limit the host declared"
                );
            }
        });

        assert_eq!(sink.rows, 1);
    }

    #[test]
    fn device_view_callback_enforces_capacity_without_copying_logits() {
        let mut sink = CountingSinkV1::default();
        with_result_sink_v1(&mut sink, |raw| {
            let emit = raw.emit_logits_view.expect("device view callback exists");
            let mut view = BundleLogitsViewV1::empty();
            view.context = BundleContextHandleV1::from_raw(0x1_0000_0001);
            view.buffer.buffer = BundleDeviceBufferHandleV1::from_raw(0x1_0000_0002);
            view.buffer.length = 8;
            view.element_type = crate::BUNDLE_LOGITS_ELEMENT_V1_F32;
            view.row_stride_elements = 2;
            view.row_count = 1;
            view.live_width = 2;
            // SAFETY-REVIEWED: the sink is live and the view contains only validated handles.
            unsafe {
                assert_eq!(emit(raw.user_data, view), BundleStatusV1::Success);
                assert_eq!(emit(raw.user_data, view), BundleStatusV1::ResourceExhausted);
            }
        });
        assert_eq!(sink.views, 1);
    }

    #[test]
    fn draft_sink_tail_copies_tokens_and_refuses_a_short_table() {
        let mut sink = CountingSinkV1::default();
        with_result_sink_v1(&mut sink, |raw| {
            ExecuteResultSinkV1::new(raw)
                .unwrap()
                .emit_draft_tokens(41, &[7, 8])
                .unwrap();
            let mut old = raw;
            old.struct_size =
                std::mem::offset_of!(BundleExecuteResultSinkV1, emit_draft_tokens) as u32;
            assert_eq!(
                ExecuteResultSinkV1::new(old)
                    .unwrap()
                    .emit_draft_tokens(41, &[9]),
                Err(BundleAbiError::MissingExecuteResultCallback)
            );
        });
        assert_eq!(
            sink.drafts,
            vec![OwnedDraftTokensV1 {
                request_id: 41,
                tokens: vec![7, 8]
            }]
        );
    }

    #[test]
    fn older_host_service_tail_reports_no_batch_limits() {
        let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
        assert_eq!(
            table.caller().query_batch_limits().unwrap(),
            BundleRawBatchLimitsV1::unreported()
        );
        let mut old = *table.services();
        old.struct_size = std::mem::offset_of!(BundleHostServicesV1, query_batch_limits_v1) as u32;
        // SAFETY-REVIEWED: the table and backend remain live for this copied prefix.
        let caller = unsafe { HostServicesV1::new(old) }.unwrap();
        assert_eq!(
            caller.query_batch_limits().unwrap(),
            BundleRawBatchLimitsV1::unreported()
        );
    }

    #[test]
    fn older_host_service_tail_reports_no_cluster_residency() {
        let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
        let config = crate::LaunchConfigV1 {
            flags: 0,
            grid_x: 16,
            grid_y: 1,
            grid_z: 1,
            block_x: 256,
            block_y: 1,
            block_z: 1,
            shared_memory_bytes: 0,
            cluster: crate::LaunchClusterV1 { x: 8, y: 1, z: 1 },
        };
        let symbol = BundleCudaSymbolHandleV1::from_parts(1, 1);
        assert_eq!(
            table
                .caller()
                .query_cluster_residency(symbol, config)
                .unwrap(),
            crate::BundleRawClusterResidencyV1::unreported()
        );
        let mut old = *table.services();
        old.struct_size =
            std::mem::offset_of!(BundleHostServicesV1, query_cluster_residency_v1) as u32;
        // SAFETY-REVIEWED: the table and backend remain live for this copied prefix.
        let caller = unsafe { HostServicesV1::new(old) }.unwrap();
        assert_eq!(
            caller.query_cluster_residency(symbol, config).unwrap(),
            crate::BundleRawClusterResidencyV1::unreported()
        );
    }

    /// A table without the host-mapped gather tail reports the service
    /// missing instead of calling it.
    #[test]
    fn older_host_service_tail_reports_no_host_mapped_gather() {
        let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
        assert!(table.caller().has_host_mapped_gather());
        let stream = BundleStreamHandleV1::from_parts(1, 1);
        let destination =
            crate::DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, 1), 0, 8);
        let gather = crate::MappedRowGatherV1 {
            image: crate::BundleCompiledImageHandleV1::from_parts(1, 1),
            rank: 0,
            view_id: b"rows",
            row_bytes: 8,
            rows: &[3],
        };
        // The current table reaches the backend, which refuses.
        assert_eq!(
            table
                .caller()
                .gather_mapped_rows(stream, destination, gather),
            Err(BundleAbiError::Status(BundleStatusV1::Unsupported))
        );
        let mut old = *table.services();
        old.struct_size = std::mem::offset_of!(BundleHostServicesV1, gather_mapped_rows_v1) as u32;
        let mut absent = *table.services();
        absent.gather_mapped_rows_v1 = None;
        for raw in [old, absent] {
            // SAFETY-REVIEWED: the table and backend remain live for this copied prefix.
            let caller = unsafe { HostServicesV1::new(raw) }.unwrap();
            assert!(!caller.has_host_mapped_gather());
            assert_eq!(
                caller.gather_mapped_rows(stream, destination, gather),
                Err(BundleAbiError::MissingHostService("gather_mapped_rows_v1"))
            );
        }
    }

    #[test]
    fn older_host_service_tail_reports_speculation_disabled() {
        let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
        assert_eq!(
            table.caller().query_speculation().unwrap(),
            BundleRawSpeculationV1::disabled()
        );
        let mut old = *table.services();
        old.struct_size = std::mem::offset_of!(BundleHostServicesV1, query_speculation_v1) as u32;
        // SAFETY-REVIEWED: the table and backend remain live for this copied prefix.
        let caller = unsafe { HostServicesV1::new(old) }.unwrap();
        assert_eq!(
            caller.query_speculation().unwrap(),
            BundleRawSpeculationV1::disabled()
        );
    }

    #[test]
    fn kept_row_dispatch_refuses_an_unbounded_count_before_reading_a_pointer() {
        let callback = |_: BundleHandleV1, _: &[BundleKeptRowsV1]| BundleStatusV1::Success;
        // SAFETY-REVIEWED: the count is refused before the deliberately null pointer is read.
        let status = unsafe {
            super::dispatch_bundle_commit_accepted_v1(
                BundleHandleV1::from_parts(1, 1),
                ptr::null(),
                (1 << 20) + 1,
                callback,
            )
        };
        assert_eq!(status, BundleStatusV1::InvalidArgument);
    }

    fn fixture_result_sink() -> BundleExecuteResultSinkV1 {
        let mut sink = BundleExecuteResultSinkV1::empty();
        sink.maximum_rows = 1;
        sink.maximum_logits_per_row = 1;
        sink.maximum_total_logits = 1;
        sink.emit_logits_row = Some(discard_logits_row);
        sink
    }

    // SAFETY-REVIEWED: the service writes one handle to caller-owned output storage.
    unsafe extern "C" fn old_rank_upload(
        _user_data: *mut c_void,
        _context: BundleContextHandleV1,
        _pool: crate::BundleMemoryPoolHandleV1,
        _stream: BundleStreamHandleV1,
        _image: crate::BundleCompiledImageHandleV1,
        _rank: u32,
        output: *mut crate::BundleRankArenaHandleV1,
    ) -> BundleStatusV1 {
        // SAFETY-REVIEWED: the safe wrapper supplies live, aligned output storage.
        unsafe { ptr::write(output, crate::BundleRankArenaHandleV1::from_parts(1, 1)) };
        BundleStatusV1::Success
    }

    #[test]
    fn batch_upload_falls_back_when_the_host_table_has_no_batch_tail() {
        let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
        let mut raw = *table.services();
        raw.struct_size =
            std::mem::offset_of!(crate::BundleHostServicesV1, upload_rank_arenas) as u32;
        raw.upload_rank_arenas = None;
        raw.upload_rank_arena = Some(old_rank_upload);
        // SAFETY-REVIEWED: the table and its backend stay live for the duration of this test.
        let host = unsafe { HostServicesV1::new(raw) }.unwrap();
        let arenas = host
            .upload_rank_arenas(&[RankArenaUploadV1 {
                context: BundleContextHandleV1::from_parts(1, 1),
                pool: crate::BundleMemoryPoolHandleV1::from_parts(1, 1),
                stream: BundleStreamHandleV1::from_parts(1, 1),
                image: crate::BundleCompiledImageHandleV1::from_parts(1, 1),
                rank: 0,
            }])
            .unwrap();
        assert_eq!(
            arenas,
            vec![crate::BundleRankArenaHandleV1::from_parts(1, 1)]
        );
    }

    extern "C" fn discard_logits_row(
        _user_data: *mut std::ffi::c_void,
        _row: crate::BundleLogitsRowV1,
    ) -> BundleStatusV1 {
        BundleStatusV1::Success
    }
}

/// The bundle side of `chop_bundle_weight_plan_v1`: decodes the request,
/// runs `plan` and emits the encoded plan, an empty document when `plan`
/// declines the image, or the refusal text through the sink's error callback.
///
/// # Safety
/// The native caller owns `request` and `sink` for the call, as the V1 header
/// states.
// SAFETY-REVIEWED: this is the reviewed pointer boundary for the weight plan export.
pub unsafe fn dispatch_weight_plan_v1(
    request: BundleByteViewV1,
    sink: crate::weight_plan::BundleWeightPlanSinkV1,
    plan: impl FnOnce(
        &crate::weight_plan::WeightPlanRequestV1,
    ) -> Result<Option<crate::weight_plan::WeightPlanV1>, String>,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        if sink.struct_size != struct_size_v1::<crate::weight_plan::BundleWeightPlanSinkV1>() {
            return Err(BundleAbiError::StructSizeExact {
                surface: "weight plan sink",
                expected: struct_size_v1::<crate::weight_plan::BundleWeightPlanSinkV1>(),
                actual: sink.struct_size,
            });
        }
        let emit = sink
            .emit
            .ok_or(BundleAbiError::NullPointer("weight plan emit"))?;
        let refuse = |message: String| {
            if let Some(error) = sink.emit_error {
                let bytes = message.into_bytes();
                // SAFETY-REVIEWED: the host lent this callback and context for the call.
                let _ = unsafe { error(sink.context, BundleByteViewV1::from_bytes(&bytes)) };
            }
            BundleStatusV1::BundleError
        };
        // A host of another layout writes a request this build would
        // misread; refuse it before reading a byte.
        if sink.layout != crate::weight_plan::WEIGHT_PLAN_LAYOUT_V1 {
            return Ok(refuse(
                crate::weight_plan::layout_mismatch("sink", sink.layout).to_string(),
            ));
        }
        let request = match with_byte_view_v1(request, |bytes| {
            crate::weight_plan::WeightPlanRequestV1::decode(bytes)
        })? {
            Ok(request) => request,
            Err(error) => return Ok(refuse(error.to_string())),
        };
        let encoded = match plan(&request) {
            Ok(Some(plan)) => plan.encode(),
            Ok(None) => Vec::new(),
            Err(message) => return Ok(refuse(message)),
        };
        // SAFETY-REVIEWED: the host lent this callback and context for the call.
        Ok(unsafe { emit(sink.context, BundleByteViewV1::from_bytes(&encoded)) })
    })
}

/// The bundle side of `chop_bundle_weight_transform_v1`: exposes the call's
/// parameters, sources and output to `transform` for its duration.
///
/// # Safety
/// The native caller owns `call` and every view it names for the call.
// SAFETY-REVIEWED: this is the reviewed pointer boundary for the bundle transform export.
pub unsafe fn dispatch_weight_transform_v1(
    call: *const crate::weight_plan::BundleWeightTransformCallV1,
    transform: impl FnOnce(u32, &[u8], &[&[u8]], u64, &mut [u8]) -> Result<(), String>,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        validate_input_pointer_v1(call, "weight transform call")?;
        // SAFETY-REVIEWED: the host lends one aligned call record for the duration.
        let call = unsafe { *call };
        if call.struct_size != struct_size_v1::<crate::weight_plan::BundleWeightTransformCallV1>() {
            return Err(BundleAbiError::StructSizeExact {
                surface: "weight transform call",
                expected: struct_size_v1::<crate::weight_plan::BundleWeightTransformCallV1>(),
                actual: call.struct_size,
            });
        }
        let count = checked_slice_count_v1::<BundleByteViewV1>(
            "weight transform sources",
            call.source_count,
        )?;
        let views: &[BundleByteViewV1] = if count == 0 {
            &[]
        } else {
            validate_input_pointer_v1(call.sources, "weight transform sources")?;
            // SAFETY-REVIEWED: the host lends `source_count` views for the duration.
            unsafe { slice::from_raw_parts(call.sources, count) }
        };
        let mut sources = Vec::with_capacity(views.len());
        for view in views {
            validate_byte_view_v1(*view)?;
            let length = checked_u64_to_usize_v1("weight transform source", view.length)?;
            sources.push(if length == 0 {
                &[][..]
            } else {
                // SAFETY-REVIEWED: each lent source view stays readable for the call.
                unsafe { slice::from_raw_parts(view.data, length) }
            });
        }
        let result = with_byte_view_v1(call.params, |params| {
            with_mut_byte_view_v1(call.output, |output| {
                transform(call.transform, params, &sources, call.output_offset, output)
            })
        })??;
        Ok(match result {
            Ok(()) => BundleStatusV1::Success,
            Err(_) => BundleStatusV1::BundleError,
        })
    })
}

/// Collects what a bundle's plan export emits.
struct WeightPlanCollectorV1 {
    plan: Option<Vec<u8>>,
    error: Option<String>,
}

// SAFETY-REVIEWED: the host passes this callback its own collector as the context.
unsafe extern "C" fn collect_weight_plan_v1(
    context: *mut c_void,
    bytes: BundleByteViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        // SAFETY-REVIEWED: the host passes its own live collector as the context.
        let collector = unsafe { &mut *context.cast::<WeightPlanCollectorV1>() };
        if collector.plan.is_some() {
            return Ok(BundleStatusV1::InvalidState);
        }
        collector.plan = Some(copy_byte_view_v1(bytes)?);
        Ok(BundleStatusV1::Success)
    })
}

// SAFETY-REVIEWED: the host passes this callback its own collector as the context.
unsafe extern "C" fn collect_weight_plan_error_v1(
    context: *mut c_void,
    bytes: BundleByteViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        // SAFETY-REVIEWED: the host passes its own live collector as the context.
        let collector = unsafe { &mut *context.cast::<WeightPlanCollectorV1>() };
        let bytes = copy_byte_view_v1(bytes)?;
        collector.error = Some(String::from_utf8_lossy(&bytes[..bytes.len().min(4096)]).into());
        Ok(BundleStatusV1::Success)
    })
}

/// The host side of `chop_bundle_weight_plan_v1`: the plan, or `None` when
/// the bundle declines the image.
pub(crate) fn call_weight_plan_v1(
    plan: crate::weight_plan::BundleWeightPlanFnV1,
    request: &crate::weight_plan::WeightPlanRequestV1,
) -> Result<Option<crate::weight_plan::WeightPlanV1>, String> {
    let encoded = request.encode();
    let mut collector = WeightPlanCollectorV1 {
        plan: None,
        error: None,
    };
    let sink = crate::weight_plan::BundleWeightPlanSinkV1 {
        struct_size: struct_size_v1::<crate::weight_plan::BundleWeightPlanSinkV1>(),
        layout: crate::weight_plan::WEIGHT_PLAN_LAYOUT_V1,
        context: ptr::from_mut(&mut collector).cast(),
        emit: Some(collect_weight_plan_v1),
        emit_error: Some(collect_weight_plan_error_v1),
    };
    // SAFETY-REVIEWED: the resolved export has the declared signature; the
    // request bytes and collector outlive the call.
    let status = unsafe { plan(BundleByteViewV1::from_bytes(&encoded), sink) };
    match (status.is_success(), collector.plan) {
        (true, Some(bytes)) if bytes.is_empty() => Ok(None),
        (true, Some(bytes)) => crate::weight_plan::WeightPlanV1::decode(&bytes)
            .map(Some)
            .map_err(|error| error.to_string()),
        (true, None) => Err("the bundle returned no weight plan".into()),
        (false, _) => Err(collector.error.unwrap_or_else(|| {
            format!(
                "the bundle refused the weight plan: {}",
                status.name().unwrap_or("unknown status")
            )
        })),
    }
}

/// The host side of `chop_bundle_weight_transform_v1`.
pub fn call_weight_transform_v1(
    transform_fn: crate::weight_plan::BundleWeightTransformFnV1,
    transform: u32,
    params: &[u8],
    sources: &[&[u8]],
    output_offset: u64,
    output: &mut [u8],
) -> Result<(), String> {
    let views: Vec<_> = sources
        .iter()
        .map(|source| BundleByteViewV1::from_bytes(source))
        .collect();
    let call = crate::weight_plan::BundleWeightTransformCallV1 {
        struct_size: struct_size_v1::<crate::weight_plan::BundleWeightTransformCallV1>(),
        transform,
        params: BundleByteViewV1::from_bytes(params),
        sources: if views.is_empty() {
            ptr::null()
        } else {
            views.as_ptr()
        },
        source_count: views.len() as u64,
        output_offset,
        output: BundleMutByteViewV1::from_bytes(output),
    };
    // SAFETY-REVIEWED: the resolved export has the declared signature; the
    // call record and every view it names outlive the call.
    let status = unsafe { transform_fn(&raw const call) };
    if status.is_success() {
        Ok(())
    } else {
        Err(format!(
            "bundle weight transform {transform} failed: {}",
            status.name().unwrap_or("unknown status")
        ))
    }
}

/// Exports `chop_bundle_weight_plan_v1` from a bundle over a safe planner
/// `fn(&WeightPlanRequestV1) -> Result<Option<WeightPlanV1>, String>`, where
/// `None` declines the requested image.
#[macro_export]
macro_rules! export_weight_plan_v1 {
    ($plan:expr $(,)?) => {
        /// # Safety
        /// The host lends the request and sink for the call.
        // SAFETY-REVIEWED: the export hands its views to the reviewed decoder.
        #[allow(unsafe_code)]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn chop_bundle_weight_plan_v1(
            request: $crate::BundleByteViewV1,
            sink: $crate::weight_plan::BundleWeightPlanSinkV1,
        ) -> $crate::BundleStatusV1 {
            // SAFETY-REVIEWED: the native caller lends both for this call.
            unsafe { $crate::dispatch_weight_plan_v1(request, sink, $plan) }
        }
    };
}

/// Exports `chop_bundle_weight_transform_v1` over a safe transform
/// `fn(u32, &[u8], &[&[u8]], u64, &mut [u8]) -> Result<(), String>`.
#[macro_export]
macro_rules! export_weight_transform_v1 {
    ($transform:expr $(,)?) => {
        /// # Safety
        /// The host lends the call record and its views for the call.
        // SAFETY-REVIEWED: the export hands its record to the reviewed decoder.
        #[allow(unsafe_code)]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn chop_bundle_weight_transform_v1(
            call: *const $crate::weight_plan::BundleWeightTransformCallV1,
        ) -> $crate::BundleStatusV1 {
            // SAFETY-REVIEWED: the native caller lends the record for this call.
            unsafe { $crate::dispatch_weight_transform_v1(call, $transform) }
        }
    };
}

#[cfg(test)]
mod weight_plan_tests {
    use super::*;
    use crate::weight_plan::{
        BundleWeightPlanSinkV1, BundleWeightTransformCallV1, WeightPlanRequestV1, WeightPlanV1,
    };

    fn answer(request: &WeightPlanRequestV1) -> Result<Option<WeightPlanV1>, String> {
        if request.rank == 1 {
            return Err("rank 1 is refused".into());
        }
        if request.image_role != "fixture.rank-images.v1" {
            return Ok(None);
        }
        Ok(Some(WeightPlanV1 {
            ranks: request.ranks,
            rank: request.rank,
            image_role: request.image_role.clone(),
            alignment: 64,
            arena_bytes: 128,
            views: vec![],
            components: vec![],
            records: vec![],
        }))
    }

    // SAFETY-REVIEWED: a test export over the reviewed decoder.
    unsafe extern "C" fn plan_export(
        request: BundleByteViewV1,
        sink: BundleWeightPlanSinkV1,
    ) -> BundleStatusV1 {
        // SAFETY-REVIEWED: the test host lends both for the call.
        unsafe { dispatch_weight_plan_v1(request, sink, answer) }
    }

    // SAFETY-REVIEWED: a test export over the reviewed decoder.
    unsafe extern "C" fn transform_export(
        call: *const BundleWeightTransformCallV1,
    ) -> BundleStatusV1 {
        // SAFETY-REVIEWED: the test host lends the record for the call.
        unsafe {
            dispatch_weight_transform_v1(call, |id, params, sources, offset, output| {
                if id != 0x8000_0001 || params != b"p" || sources.len() != 2 {
                    return Err("unexpected call".into());
                }
                for (index, byte) in output.iter_mut().enumerate() {
                    *byte = sources[0][offset as usize + index] ^ sources[1][0];
                }
                Ok(())
            })
        }
    }

    #[test]
    fn plans_and_refusals_cross_the_export() {
        let mut request = WeightPlanRequestV1 {
            image_role: "fixture.rank-images.v1".into(),
            ranks: 2,
            rank: 0,
            ..WeightPlanRequestV1::default()
        };
        let plan = call_weight_plan_v1(plan_export, &request).unwrap().unwrap();
        assert_eq!((plan.rank, plan.arena_bytes), (0, 128));
        let other = WeightPlanRequestV1 {
            image_role: "fixture.encoder-images.v1".into(),
            ..request.clone()
        };
        assert_eq!(call_weight_plan_v1(plan_export, &other).unwrap(), None);
        request.rank = 1;
        assert_eq!(
            call_weight_plan_v1(plan_export, &request).unwrap_err(),
            "rank 1 is refused"
        );
    }

    /// A host of another document layout is refused by name, through the
    /// sink, before the bundle reads its request.
    #[test]
    fn a_sink_of_another_layout_is_refused_before_decoding() {
        let request = WeightPlanRequestV1 {
            image_role: "fixture.rank-images.v1".into(),
            ranks: 2,
            rank: 0,
            ..WeightPlanRequestV1::default()
        }
        .encode();
        for layout in [0, 1, crate::weight_plan::WEIGHT_PLAN_LAYOUT_V1 + 1] {
            let mut collector = WeightPlanCollectorV1 {
                plan: None,
                error: None,
            };
            let sink = BundleWeightPlanSinkV1 {
                struct_size: struct_size_v1::<BundleWeightPlanSinkV1>(),
                layout,
                context: ptr::from_mut(&mut collector).cast(),
                emit: Some(collect_weight_plan_v1),
                emit_error: Some(collect_weight_plan_error_v1),
            };
            // SAFETY-REVIEWED: the request bytes and collector outlive the call.
            let status = unsafe { plan_export(BundleByteViewV1::from_bytes(&request), sink) };
            assert!(!status.is_success());
            assert!(collector.plan.is_none());
            let error = collector.error.expect("a named refusal");
            assert!(error.contains(&format!("sink layout {layout}")), "{error}");
        }
    }

    #[test]
    fn bundle_transforms_fill_the_requested_part() {
        let source = [1_u8, 2, 3, 4, 5, 6];
        let mut output = [0_u8; 3];
        call_weight_transform_v1(
            transform_export,
            0x8000_0001,
            b"p",
            &[&source, &[0xf0]],
            2,
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [0xf3, 0xf4, 0xf5]);
        assert!(
            call_weight_transform_v1(
                transform_export,
                0x8000_0002,
                b"p",
                &[&source],
                0,
                &mut output
            )
            .is_err()
        );
    }
}

/// A frontend sink the bundle checked: its size and both callbacks.
struct FrontendSinkCallsV1 {
    layout: u32,
    context: *mut c_void,
    emit: crate::BundleFrontendEmitFnV1,
    emit_error: crate::BundleFrontendEmitFnV1,
}

fn frontend_sink_v1(
    sink: crate::frontend::BundleFrontendSinkV1,
) -> Result<FrontendSinkCallsV1, BundleAbiError> {
    if sink.struct_size != struct_size_v1::<crate::frontend::BundleFrontendSinkV1>() {
        return Err(BundleAbiError::StructSizeExact {
            surface: "frontend sink",
            expected: struct_size_v1::<crate::frontend::BundleFrontendSinkV1>(),
            actual: sink.struct_size,
        });
    }
    Ok(FrontendSinkCallsV1 {
        layout: sink.layout,
        context: sink.context,
        emit: sink
            .emit
            .ok_or(BundleAbiError::NullPointer("frontend emit"))?,
        emit_error: sink
            .emit_error
            .ok_or(BundleAbiError::NullPointer("frontend emit_error"))?,
    })
}

impl FrontendSinkCallsV1 {
    fn answer(&self, document: &[u8]) -> BundleStatusV1 {
        // SAFETY-REVIEWED: the host lent this callback and context for the call.
        unsafe { (self.emit)(self.context, BundleByteViewV1::from_bytes(document)) }
    }

    fn refuse(&self, document: &[u8]) -> BundleStatusV1 {
        // SAFETY-REVIEWED: the host lent this callback and context for the call.
        let _ = unsafe { (self.emit_error)(self.context, BundleByteViewV1::from_bytes(document)) };
        BundleStatusV1::BundleError
    }

    /// A host of another layout writes requests this build would misread,
    /// so the call is refused, by name, before a byte is read.
    fn refuse_layout(&self) -> Option<BundleStatusV1> {
        (self.layout != crate::frontend::FRONTEND_LAYOUT_V1).then(|| {
            self.refuse(
                format!(
                    "frontend sink layout {}; this build reads layout {}",
                    self.layout,
                    crate::frontend::FRONTEND_LAYOUT_V1
                )
                .as_bytes(),
            )
        })
    }
}

/// A bundle parser and whether it may take another push: not after the
/// final push, a refusal or a panic.
struct FrontendParserSlotV1<P> {
    parser: P,
    done: bool,
    /// Dropped after the parser, even when the parser's drop panics.
    _open: OpenParserV1,
}

/// A bundle frontend and the count of parsers open on it, which its close
/// checks.
struct FrontendSlotV1<F> {
    frontend: F,
    parsers: Arc<AtomicUsize>,
}

/// Counts one open parser of a frontend until it drops.
struct OpenParserV1(Arc<AtomicUsize>);

impl OpenParserV1 {
    fn new(parsers: &Arc<AtomicUsize>) -> Self {
        parsers.fetch_add(1, Ordering::AcqRel);
        Self(Arc::clone(parsers))
    }
}

impl Drop for OpenParserV1 {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

// SAFETY-REVIEWED: the bundle side of the table's `open`; the host lends the views and output for the call.
unsafe extern "C" fn frontend_open_v1<F: crate::frontend::FrontendV1>(
    component: BundleByteViewV1,
    sink: crate::frontend::BundleFrontendSinkV1,
    frontend: *mut *mut crate::frontend::BundleFrontendV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let sink = frontend_sink_v1(sink)?;
        if let Some(status) = sink.refuse_layout() {
            return Ok(status);
        }
        validate_output_pointer_v1(frontend, "frontend output")?;
        match with_byte_view_v1(component, F::open)? {
            Ok((opened, answer)) => {
                // The answer crosses first, so a host that refuses it never
                // receives the frontend, which drops here.
                let status = sink.answer(&answer);
                if status != BundleStatusV1::Success {
                    return Ok(status);
                }
                let opened = Box::into_raw(Box::new(FrontendSlotV1 {
                    frontend: opened,
                    parsers: Arc::new(AtomicUsize::new(0)),
                }))
                .cast();
                write_output_v1(frontend, opened)?;
                Ok(BundleStatusV1::Success)
            }
            Err(document) => Ok(sink.refuse(&document)),
        }
    })
}

// SAFETY-REVIEWED: the bundle side of `close`; the host passes a frontend this table opened, once.
unsafe extern "C" fn frontend_close_v1<F: crate::frontend::FrontendV1>(
    frontend: *mut crate::frontend::BundleFrontendV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let slot = frontend.cast::<FrontendSlotV1<F>>();
        validate_input_pointer_v1(slot.cast_const(), "frontend")?;
        // SAFETY-REVIEWED: `open` boxed this slot, and it stays open until this close succeeds.
        if unsafe { &*slot }.parsers.load(Ordering::Acquire) != 0 {
            return Err(BundleAbiError::Lifecycle(
                "a frontend close while a parser opened on it is open",
            ));
        }
        // SAFETY-REVIEWED: `open` boxed this slot, and the host closes it once.
        drop(unsafe { Box::from_raw(slot) });
        Ok(BundleStatusV1::Success)
    })
}

/// A frontend's render or constraint: an answer or a refusal document.
type FrontendDocumentCallV1<F> = fn(&F, &[u8]) -> Result<Vec<u8>, Vec<u8>>;

/// Render and constraint: one document in, one answer or refusal out.
fn dispatch_frontend_document_v1<F: crate::frontend::FrontendV1>(
    frontend: *const crate::frontend::BundleFrontendV1,
    request: BundleByteViewV1,
    sink: crate::frontend::BundleFrontendSinkV1,
    answer: FrontendDocumentCallV1<F>,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let sink = frontend_sink_v1(sink)?;
        if let Some(status) = sink.refuse_layout() {
            return Ok(status);
        }
        validate_input_pointer_v1(frontend.cast::<FrontendSlotV1<F>>(), "frontend")?;
        // SAFETY-REVIEWED: the host passes an open frontend of this table; `F: Sync`
        // lets concurrent render and constraint calls share it.
        let frontend = unsafe { &(*frontend.cast::<FrontendSlotV1<F>>()).frontend };
        match with_byte_view_v1(request, |request| answer(frontend, request))? {
            Ok(document) => Ok(sink.answer(&document)),
            Err(document) => Ok(sink.refuse(&document)),
        }
    })
}

// SAFETY-REVIEWED: the bundle side of `render`; the host lends the frontend, view and sink for the call.
unsafe extern "C" fn frontend_render_v1<F: crate::frontend::FrontendV1>(
    frontend: *const crate::frontend::BundleFrontendV1,
    request: BundleByteViewV1,
    sink: crate::frontend::BundleFrontendSinkV1,
) -> BundleStatusV1 {
    dispatch_frontend_document_v1::<F>(frontend, request, sink, F::render)
}

// SAFETY-REVIEWED: the bundle side of `constraint`; the host lends the frontend, view and sink for the call.
unsafe extern "C" fn frontend_constraint_v1<F: crate::frontend::FrontendV1>(
    frontend: *const crate::frontend::BundleFrontendV1,
    request: BundleByteViewV1,
    sink: crate::frontend::BundleFrontendSinkV1,
) -> BundleStatusV1 {
    dispatch_frontend_document_v1::<F>(frontend, request, sink, F::constraint)
}

// SAFETY-REVIEWED: the bundle side of `parser_open`; the host lends the frontend, view, sink and output for the call.
unsafe extern "C" fn frontend_parser_open_v1<F: crate::frontend::FrontendV1>(
    frontend: *const crate::frontend::BundleFrontendV1,
    request: BundleByteViewV1,
    sink: crate::frontend::BundleFrontendSinkV1,
    parser: *mut *mut crate::frontend::BundleFrontendParserV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let sink = frontend_sink_v1(sink)?;
        if let Some(status) = sink.refuse_layout() {
            return Ok(status);
        }
        validate_input_pointer_v1(frontend.cast::<FrontendSlotV1<F>>(), "frontend")?;
        validate_output_pointer_v1(parser, "frontend parser output")?;
        // SAFETY-REVIEWED: the host passes an open frontend of this table.
        let frontend = unsafe { &*frontend.cast::<FrontendSlotV1<F>>() };
        match with_byte_view_v1(request, |request| frontend.frontend.open_parser(request))? {
            Ok(opened) => {
                let slot = FrontendParserSlotV1 {
                    parser: opened,
                    done: false,
                    _open: OpenParserV1::new(&frontend.parsers),
                };
                write_output_v1(parser, Box::into_raw(Box::new(slot)).cast())?;
                Ok(BundleStatusV1::Success)
            }
            Err(document) => Ok(sink.refuse(&document)),
        }
    })
}

// SAFETY-REVIEWED: the bundle side of `parser_push`; the host lends the parser, view and sink for the call.
unsafe extern "C" fn frontend_parser_push_v1<F: crate::frontend::FrontendV1>(
    parser: *mut crate::frontend::BundleFrontendParserV1,
    text: BundleByteViewV1,
    flags: u32,
    sink: crate::frontend::BundleFrontendSinkV1,
) -> BundleStatusV1 {
    use crate::frontend::{FRONTEND_PUSH_FLAG_V1_FINAL, FrontendParserV1};
    catch_panic_callback_status_v1(|| {
        let slot = parser.cast::<FrontendParserSlotV1<F::Parser>>();
        validate_input_pointer_v1(slot.cast_const(), "frontend parser")?;
        // SAFETY-REVIEWED: the host passes an open parser of this table and uses
        // it from one thread at a time, so this borrow is exclusive.
        let slot = unsafe { &mut *slot };
        if slot.done {
            return Err(BundleAbiError::Lifecycle(
                "a frontend push after the final push or a failure",
            ));
        }
        // Every way this push can fail, a panic included, leaves this set;
        // only an answered push before the final one clears it.
        slot.done = true;
        let sink = frontend_sink_v1(sink)?;
        if let Some(status) = sink.refuse_layout() {
            return Ok(status);
        }
        if flags & !crate::frontend::FRONTEND_PUSH_FLAGS_V1_KNOWN != 0 {
            return Err(BundleAbiError::InvalidReserved {
                surface: "frontend push flags",
                actual: flags,
            });
        }
        let last = flags & FRONTEND_PUSH_FLAG_V1_FINAL != 0;
        match with_byte_view_v1(text, |text| slot.parser.push(text, last))? {
            Ok(document) => {
                let status = sink.answer(&document);
                slot.done = last || !status.is_success();
                Ok(status)
            }
            Err(document) => Ok(sink.refuse(&document)),
        }
    })
}

// SAFETY-REVIEWED: the bundle side of `parser_close`; the host passes a parser this table opened, once.
unsafe extern "C" fn frontend_parser_close_v1<F: crate::frontend::FrontendV1>(
    parser: *mut crate::frontend::BundleFrontendParserV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        let slot = parser.cast::<FrontendParserSlotV1<F::Parser>>();
        validate_input_pointer_v1(slot.cast_const(), "frontend parser")?;
        // SAFETY-REVIEWED: `parser_open` boxed this slot, and the host closes it once.
        drop(unsafe { Box::from_raw(slot) });
        Ok(BundleStatusV1::Success)
    })
}

/// The table [`crate::export_frontend_v1!`] exports for `F`: every entry is
/// a trampoline that checks its pointers and sink and catches panics.
#[must_use]
pub const fn frontend_table_v1<F: crate::frontend::FrontendV1>()
-> crate::frontend::BundleFrontendTableV1 {
    crate::frontend::BundleFrontendTableV1 {
        struct_size: struct_size_v1::<crate::frontend::BundleFrontendTableV1>(),
        layout: crate::frontend::FRONTEND_LAYOUT_V1,
        open: Some(frontend_open_v1::<F>),
        close: Some(frontend_close_v1::<F>),
        render: Some(frontend_render_v1::<F>),
        constraint: Some(frontend_constraint_v1::<F>),
        parser_open: Some(frontend_parser_open_v1::<F>),
        parser_push: Some(frontend_parser_push_v1::<F>),
        parser_close: Some(frontend_parser_close_v1::<F>),
    }
}

/// Exports `chop_bundle_frontend_api_v1` from a bundle over a type that
/// implements [`crate::frontend::FrontendV1`].
#[macro_export]
macro_rules! export_frontend_v1 {
    ($frontend:ty $(,)?) => {
        /// The bundle's chat frontend table.
        // SAFETY-REVIEWED: the export returns a static table of the reviewed trampolines.
        #[allow(unsafe_code)]
        #[unsafe(no_mangle)]
        pub extern "C" fn chop_bundle_frontend_api_v1()
        -> *const $crate::frontend::BundleFrontendTableV1 {
            static TABLE: $crate::frontend::BundleFrontendTableV1 =
                $crate::frontend::frontend_table_v1::<$frontend>();
            &raw const TABLE
        }
    };
}

/// The checked entries of a bundle's frontend table.
///
/// The entries are code in the bundle's module. An export read from a loaded
/// library holds that module, and so does every handle opened from it, so no
/// entry is called after the module unmaps.
#[derive(Clone)]
pub struct FrontendExportV1 {
    /// The module the entries live in, absent for an export linked into this
    /// program.
    module: Option<Arc<crate::loader::DynamicLibraryV1>>,
    open: crate::BundleFrontendOpenFnV1,
    close: crate::BundleFrontendCloseFnV1,
    render: crate::BundleFrontendDocumentFnV1,
    constraint: crate::BundleFrontendDocumentFnV1,
    parser_open: crate::BundleFrontendParserOpenFnV1,
    parser_push: crate::BundleFrontendParserPushFnV1,
    parser_close: crate::BundleFrontendParserCloseFnV1,
}

impl std::fmt::Debug for FrontendExportV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FrontendExportV1")
            .field("loaded_module", &self.module.is_some())
            .finish_non_exhaustive()
    }
}

impl FrontendExportV1 {
    /// The export of `F`'s table, as [`crate::export_frontend_v1!`] builds
    /// it, called in this process: an in-process loopback of the ABI.
    #[must_use]
    pub fn in_process<F: crate::frontend::FrontendV1>() -> Self {
        Self::from_table(frontend_table_v1::<F>(), None).expect("the crate's own table is complete")
    }

    /// Reads the table `api` returns and checks its size, layout and
    /// entries. A loaded bundle's export comes from
    /// [`crate::BundleLibraryV1::frontend_v1`] instead, which keeps its module
    /// mapped, and an in-process frontend from [`Self::in_process`].
    ///
    /// # Safety
    /// `api` must return null or a pointer to a readable table that opens
    /// with its `u32` struct_size and, when that size is V1's, is a whole
    /// [`crate::frontend::BundleFrontendTableV1`]; its entries must keep the
    /// contract of `include/chop_bundle_v1.h`; and the code they name must
    /// stay loaded while this export or any handle opened from it lives. A
    /// safe function may return any pointer, so this cannot be safe:
    ///
    /// ```compile_fail,E0133
    /// use runtime_bundle_abi::frontend::{BundleFrontendTableV1, FrontendExportV1};
    /// extern "C" fn dangling() -> *const BundleFrontendTableV1 {
    ///     std::ptr::dangling()
    /// }
    /// let _ = FrontendExportV1::from_api(dangling);
    /// ```
    // SAFETY-REVIEWED: an unsafe constructor; its callers state the requirements above.
    pub unsafe fn from_api(api: crate::BundleFrontendApiFnV1) -> Result<Self, BundleAbiError> {
        // SAFETY-REVIEWED: the caller states the requirements above.
        unsafe { Self::from_module(api, None) }
    }

    /// As [`Self::from_api`], holding `module`, the loaded module `api` is in.
    ///
    /// # Safety
    /// As [`Self::from_api`]; holding `module` keeps the code loaded.
    // SAFETY-REVIEWED: the reviewed reader of a frontend table, under the requirements above.
    pub(crate) unsafe fn from_module(
        api: crate::BundleFrontendApiFnV1,
        module: Option<Arc<crate::loader::DynamicLibraryV1>>,
    ) -> Result<Self, BundleAbiError> {
        use crate::frontend::BundleFrontendTableV1;
        // SAFETY-REVIEWED: the caller guarantees the export returns null or a readable table.
        let table = unsafe { api() };
        validate_input_pointer_v1(table, "frontend table")?;
        // A table of another size is refused before more than its first
        // field is read.
        // SAFETY-REVIEWED: every table version opens with its u32 struct_size.
        let struct_size = unsafe { ptr::read(table.cast::<u32>()) };
        if struct_size != struct_size_v1::<BundleFrontendTableV1>() {
            return Err(BundleAbiError::StructSizeExact {
                surface: "frontend table",
                expected: struct_size_v1::<BundleFrontendTableV1>(),
                actual: struct_size,
            });
        }
        // SAFETY-REVIEWED: the table states exactly this record's size.
        Self::from_table(unsafe { ptr::read(table) }, module)
    }

    /// Checks a table's layout and entries.
    fn from_table(
        table: crate::frontend::BundleFrontendTableV1,
        module: Option<Arc<crate::loader::DynamicLibraryV1>>,
    ) -> Result<Self, BundleAbiError> {
        if table.layout != crate::frontend::FRONTEND_LAYOUT_V1 {
            return Err(BundleAbiError::AbiVersion {
                surface: "frontend table layout",
                actual: table.layout,
            });
        }
        let entry = |name| BundleAbiError::MissingApiCallback(name);
        Ok(Self {
            module,
            open: table.open.ok_or(entry("frontend open"))?,
            close: table.close.ok_or(entry("frontend close"))?,
            render: table.render.ok_or(entry("frontend render"))?,
            constraint: table.constraint.ok_or(entry("frontend constraint"))?,
            parser_open: table.parser_open.ok_or(entry("frontend parser_open"))?,
            parser_push: table.parser_push.ok_or(entry("frontend parser_push"))?,
            parser_close: table.parser_close.ok_or(entry("frontend parser_close"))?,
        })
    }

    /// Opens a frontend over the authenticated frontend component: the
    /// handle, and the open answer document of at most
    /// [`crate::frontend::FRONTEND_OPENED_BYTES_MAX_V1`] bytes.
    pub fn open(
        &self,
        component: &[u8],
    ) -> Result<(FrontendHandleV1, Vec<u8>), crate::frontend::FrontendCallErrorV1> {
        let mut opened: *mut crate::frontend::BundleFrontendV1 = ptr::null_mut();
        let maximum = crate::frontend::FRONTEND_OPENED_BYTES_MAX_V1;
        let (status, collector) = call_frontend_v1(Some(maximum), |sink| {
            // SAFETY-REVIEWED: a checked entry; the component, sink and output outlive the call.
            unsafe {
                (self.open)(
                    BundleByteViewV1::from_bytes(component),
                    sink,
                    &raw mut opened,
                )
            }
        });
        let answer = match opened_frontend_v1(status, collector, opened, true) {
            Ok(answer) => answer,
            Err(error) => {
                if status.is_success() && !opened.is_null() {
                    // SAFETY-REVIEWED: a checked entry; the bundle reported this frontend
                    // open, and the host closes it once as it refuses the call.
                    let _ = unsafe { (self.close)(opened) };
                }
                return Err(error);
            }
        };
        let frontend = ptr::NonNull::new(opened).ok_or(
            crate::frontend::FrontendCallErrorV1::Contract("open returned no frontend"),
        )?;
        let handle = FrontendHandleV1 {
            inner: Arc::new(FrontendInnerV1 {
                export: self.clone(),
                frontend: Some(frontend),
            }),
        };
        Ok((handle, answer))
    }
}

/// Collects what one frontend call emits.
struct FrontendCollectorV1 {
    /// The longest answer the host takes from this call, or `None` for a
    /// parser open, which answers through its output pointer: an answer it
    /// emits is recorded, never copied, and fails the open. A refusal may be up to
    /// [`crate::frontend::FRONTEND_REFUSAL_BYTES_MAX_V1`].
    maximum: Option<usize>,
    /// A document past its limit, refused before a byte of it was copied.
    oversized: Option<(u64, usize)>,
    answer: Option<Vec<u8>>,
    refusal: Option<Vec<u8>>,
    /// The first invalid sink call: a second document of either kind, a
    /// malformed view or a document past its limit. It fails the whole call,
    /// and every later sink call is refused unread.
    broken: Option<&'static str>,
}

impl FrontendCollectorV1 {
    const fn with_maximum(maximum: Option<usize>) -> Self {
        Self {
            maximum,
            oversized: None,
            answer: None,
            refusal: None,
            broken: None,
        }
    }

    /// Takes one document from the sink, as the answer or the refusal.
    fn take(&mut self, document: BundleByteViewV1, refusal: bool) -> BundleStatusV1 {
        if self.broken.is_some() {
            return BundleStatusV1::InvalidState;
        }
        if self.answer.is_some() || self.refusal.is_some() {
            self.broken = Some("a call emitted twice");
            return BundleStatusV1::InvalidState;
        }
        let maximum = match (refusal, self.maximum) {
            (true, _) => crate::frontend::FRONTEND_REFUSAL_BYTES_MAX_V1,
            (false, Some(maximum)) => maximum,
            (false, None) => {
                // Checked as a view, never copied: the open fails anyway.
                if with_byte_view_v1(document, |_| ()).is_err() {
                    self.broken = Some("a call emitted a malformed view");
                    return BundleStatusV1::InvalidArgument;
                }
                self.answer = Some(Vec::new());
                return BundleStatusV1::Success;
            }
        };
        if usize::try_from(document.length).map_or(true, |length| length > maximum) {
            self.oversized = Some((document.length, maximum));
            self.broken = Some("a call emitted a document past its byte limit");
            return BundleStatusV1::InvalidArgument;
        }
        match copy_byte_view_v1(document) {
            Ok(bytes) if refusal => self.refusal = Some(bytes),
            Ok(bytes) => self.answer = Some(bytes),
            Err(_) => {
                self.broken = Some("a call emitted a malformed view");
                return BundleStatusV1::InvalidArgument;
            }
        }
        BundleStatusV1::Success
    }
}

// SAFETY-REVIEWED: the host passes this callback its own collector as the context.
unsafe extern "C" fn collect_frontend_answer_v1(
    context: *mut c_void,
    document: BundleByteViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        // SAFETY-REVIEWED: the host passes its own live collector as the context.
        let collector = unsafe { &mut *context.cast::<FrontendCollectorV1>() };
        Ok(collector.take(document, false))
    })
}

// SAFETY-REVIEWED: the host passes this callback its own collector as the context.
unsafe extern "C" fn collect_frontend_refusal_v1(
    context: *mut c_void,
    document: BundleByteViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        // SAFETY-REVIEWED: the host passes its own live collector as the context.
        let collector = unsafe { &mut *context.cast::<FrontendCollectorV1>() };
        Ok(collector.take(document, true))
    })
}

/// Runs one frontend call with a sink over a fresh collector that takes an
/// answer of at most `maximum` bytes.
fn call_frontend_v1(
    maximum: Option<usize>,
    call: impl FnOnce(crate::frontend::BundleFrontendSinkV1) -> BundleStatusV1,
) -> (BundleStatusV1, FrontendCollectorV1) {
    let mut collector = FrontendCollectorV1::with_maximum(maximum);
    let sink = crate::frontend::BundleFrontendSinkV1 {
        struct_size: struct_size_v1::<crate::frontend::BundleFrontendSinkV1>(),
        layout: crate::frontend::FRONTEND_LAYOUT_V1,
        context: ptr::from_mut(&mut collector).cast(),
        emit: Some(collect_frontend_answer_v1),
        emit_error: Some(collect_frontend_refusal_v1),
    };
    let status = call(sink);
    (status, collector)
}

/// A failed call's error: a sink violation first, whatever the status
/// (a panic included), then the refusal document when the bundle refused
/// with one, otherwise how it failed.
fn failed_frontend_call_v1(
    status: BundleStatusV1,
    collector: FrontendCollectorV1,
) -> crate::frontend::FrontendCallErrorV1 {
    use crate::frontend::FrontendCallErrorV1;
    if let Some(violation) = sink_violation_v1(&collector) {
        return violation;
    }
    if status == BundleStatusV1::Panic {
        return FrontendCallErrorV1::Panicked;
    }
    match (status, collector.refusal) {
        (BundleStatusV1::BundleError, Some(document)) => FrontendCallErrorV1::Refused(document),
        (status, _) => FrontendCallErrorV1::Status(status),
    }
}

/// The first invalid sink call, as the call's error.
fn sink_violation_v1(
    collector: &FrontendCollectorV1,
) -> Option<crate::frontend::FrontendCallErrorV1> {
    use crate::frontend::FrontendCallErrorV1;
    if let Some((length, maximum)) = collector.oversized {
        return Some(FrontendCallErrorV1::Oversized { length, maximum });
    }
    collector.broken.map(FrontendCallErrorV1::Contract)
}

/// The answer of a render, constraint or push.
fn frontend_answer_v1(
    status: BundleStatusV1,
    collector: FrontendCollectorV1,
) -> Result<Vec<u8>, crate::frontend::FrontendCallErrorV1> {
    use crate::frontend::FrontendCallErrorV1;
    if !status.is_success() {
        return Err(failed_frontend_call_v1(status, collector));
    }
    if let Some(violation) = sink_violation_v1(&collector) {
        return Err(violation);
    }
    if collector.refusal.is_some() {
        return Err(FrontendCallErrorV1::Contract(
            "a successful call emitted other than one answer",
        ));
    }
    collector.answer.ok_or(FrontendCallErrorV1::Contract(
        "a successful call emitted no answer",
    ))
}

/// Checks an open or parser open, which returns its object through its
/// output pointer: an open with exactly one answer document, which this
/// returns, and a parser open with none. A failed call whose bundle wrote a
/// pointer anyway is not trusted with it.
fn opened_frontend_v1<T>(
    status: BundleStatusV1,
    collector: FrontendCollectorV1,
    opened: *mut T,
    answers: bool,
) -> Result<Vec<u8>, crate::frontend::FrontendCallErrorV1> {
    use crate::frontend::FrontendCallErrorV1;
    if !status.is_success() {
        return Err(failed_frontend_call_v1(status, collector));
    }
    if let Some(violation) = sink_violation_v1(&collector) {
        return Err(violation);
    }
    let answer = if answers {
        if collector.refusal.is_some() {
            return Err(FrontendCallErrorV1::Contract(
                "a successful call emitted other than one answer",
            ));
        }
        collector.answer.ok_or(FrontendCallErrorV1::Contract(
            "a successful open emitted no answer",
        ))?
    } else {
        if collector.answer.is_some() || collector.refusal.is_some() {
            return Err(FrontendCallErrorV1::Contract(
                "a successful open emitted a document",
            ));
        }
        Vec::new()
    };
    if opened.is_null() {
        return Err(FrontendCallErrorV1::Contract(
            "a successful open returned null",
        ));
    }
    Ok(answer)
}

fn closed_frontend_v1(status: BundleStatusV1) -> Result<(), crate::frontend::FrontendCallErrorV1> {
    match status {
        BundleStatusV1::Success => Ok(()),
        BundleStatusV1::Panic => Err(crate::frontend::FrontendCallErrorV1::Panicked),
        status => Err(crate::frontend::FrontendCallErrorV1::Status(status)),
    }
}

/// An open bundle frontend, closed once its handle and every parser opened
/// from it are gone.
struct FrontendInnerV1 {
    export: FrontendExportV1,
    frontend: Option<ptr::NonNull<crate::frontend::BundleFrontendV1>>,
}

// SAFETY-REVIEWED: the header lets render, constraint and parser_open run concurrently
// on one frontend from any thread, and close runs once, from whichever thread drops the
// last reference, after every parser (each holds one).
unsafe impl Send for FrontendInnerV1 {}
// SAFETY-REVIEWED: as above; shared references only reach render, constraint and parser_open.
unsafe impl Sync for FrontendInnerV1 {}

impl FrontendInnerV1 {
    fn pointer(&self) -> *const crate::frontend::BundleFrontendV1 {
        self.frontend
            .expect("a frontend stays open while referenced")
            .as_ptr()
            .cast_const()
    }

    fn close(&mut self) -> Result<(), crate::frontend::FrontendCallErrorV1> {
        let Some(frontend) = self.frontend.take() else {
            return Ok(());
        };
        // SAFETY-REVIEWED: a checked entry, called once for a frontend `open` returned
        // and after every parser opened on it closed, since each parser holds a reference.
        closed_frontend_v1(unsafe { (self.export.close)(frontend.as_ptr()) })
    }
}

impl Drop for FrontendInnerV1 {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// A bundle frontend the host opened. Render and constraint may be called
/// from any thread; the frontend closes once this handle and every parser
/// opened from it are dropped.
pub struct FrontendHandleV1 {
    inner: Arc<FrontendInnerV1>,
}

/// Names the handle's export without the frontend's pointer.
impl std::fmt::Debug for FrontendHandleV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FrontendHandleV1")
            .field("export", &self.inner.export)
            .finish_non_exhaustive()
    }
}

impl FrontendHandleV1 {
    fn document(
        &self,
        entry: crate::BundleFrontendDocumentFnV1,
        request: &[u8],
        maximum: usize,
    ) -> Result<Vec<u8>, crate::frontend::FrontendCallErrorV1> {
        let (status, collector) = call_frontend_v1(Some(maximum), |sink| {
            // SAFETY-REVIEWED: a checked entry of this frontend's table; the frontend is
            // open, and the request and sink outlive the call.
            unsafe {
                entry(
                    self.inner.pointer(),
                    BundleByteViewV1::from_bytes(request),
                    sink,
                )
            }
        });
        frontend_answer_v1(status, collector)
    }

    /// The rendered prompt document for a chat request document, of at
    /// most `maximum` bytes: a longer one is refused before it is copied.
    pub fn render(
        &self,
        request: &[u8],
        maximum: usize,
    ) -> Result<Vec<u8>, crate::frontend::FrontendCallErrorV1> {
        self.document(self.inner.export.render, request, maximum)
    }

    /// The request's constraint document, empty when it has none, of at
    /// most `maximum` bytes: a longer one is refused before it is copied.
    pub fn constraint(
        &self,
        request: &[u8],
        maximum: usize,
    ) -> Result<Vec<u8>, crate::frontend::FrontendCallErrorV1> {
        self.document(self.inner.export.constraint, request, maximum)
    }

    /// Opens a parser of one request's output.
    pub fn open_parser(
        &self,
        request: &[u8],
    ) -> Result<FrontendParserHandleV1, crate::frontend::FrontendCallErrorV1> {
        let mut opened: *mut crate::frontend::BundleFrontendParserV1 = ptr::null_mut();
        // A parser open answers through its output pointer, never a document.
        let (status, collector) = call_frontend_v1(None, |sink| {
            // SAFETY-REVIEWED: a checked entry of this frontend's table; the frontend is
            // open, and the request, sink and output outlive the call.
            unsafe {
                (self.inner.export.parser_open)(
                    self.inner.pointer(),
                    BundleByteViewV1::from_bytes(request),
                    sink,
                    &raw mut opened,
                )
            }
        });
        if let Err(error) = opened_frontend_v1(status, collector, opened, false) {
            if status.is_success() && !opened.is_null() {
                // SAFETY-REVIEWED: a checked entry; the bundle reported this parser
                // open, and the host closes it once as it refuses the call.
                let _ = unsafe { (self.inner.export.parser_close)(opened) };
            }
            return Err(error);
        }
        Ok(FrontendParserHandleV1 {
            parser: ptr::NonNull::new(opened),
            finished: false,
            frontend: Arc::clone(&self.inner),
        })
    }

    /// Closes the frontend now and reports how its close went. While a
    /// parser opened from it is still open, the close waits for that parser
    /// and this reports a contract error.
    pub fn close(self) -> Result<(), crate::frontend::FrontendCallErrorV1> {
        match Arc::try_unwrap(self.inner) {
            Ok(mut inner) => inner.close(),
            Err(_) => Err(crate::frontend::FrontendCallErrorV1::Contract(
                "a frontend closed while a parser is open",
            )),
        }
    }
}

/// One request's parser. Pushes take `&mut self`, so one thread at a time
/// uses it; it closes when dropped.
pub struct FrontendParserHandleV1 {
    parser: Option<ptr::NonNull<crate::frontend::BundleFrontendParserV1>>,
    /// Set by the final push and by any failed push; the handle then refuses
    /// pushes without calling the bundle.
    finished: bool,
    /// Keeps the frontend open until this parser closes.
    frontend: Arc<FrontendInnerV1>,
}

// SAFETY-REVIEWED: the header lets a parser move between threads while one thread
// at a time uses it, which `&mut self` on every call enforces.
unsafe impl Send for FrontendParserHandleV1 {}

/// States whether the parser is finished or closed without its pointer.
impl std::fmt::Debug for FrontendParserHandleV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FrontendParserHandleV1")
            .field("finished", &self.finished)
            .field("closed", &self.parser.is_none())
            .finish_non_exhaustive()
    }
}

impl FrontendParserHandleV1 {
    /// The event document for the next output text, of at most `maximum`
    /// bytes (a longer one is refused before it is copied); `last` marks
    /// the request's final push. After the final push or any failure the
    /// parser refuses further pushes.
    pub fn push(
        &mut self,
        text: &[u8],
        last: bool,
        maximum: usize,
    ) -> Result<Vec<u8>, crate::frontend::FrontendCallErrorV1> {
        if self.finished {
            return Err(crate::frontend::FrontendCallErrorV1::ParserFinished);
        }
        let parser = self.parser.expect("a parser stays open until it is closed");
        let flags = if last {
            crate::frontend::FRONTEND_PUSH_FLAG_V1_FINAL
        } else {
            0
        };
        let (status, collector) = call_frontend_v1(Some(maximum), |sink| {
            // SAFETY-REVIEWED: a checked entry; the parser is open and exclusively borrowed,
            // and the text and sink outlive the call.
            unsafe {
                (self.frontend.export.parser_push)(
                    parser.as_ptr(),
                    BundleByteViewV1::from_bytes(text),
                    flags,
                    sink,
                )
            }
        });
        let answer = frontend_answer_v1(status, collector);
        self.finished = last || answer.is_err();
        answer
    }

    /// Closes the parser and reports how its close went.
    pub fn close(mut self) -> Result<(), crate::frontend::FrontendCallErrorV1> {
        self.close_parser()
    }

    fn close_parser(&mut self) -> Result<(), crate::frontend::FrontendCallErrorV1> {
        let Some(parser) = self.parser.take() else {
            return Ok(());
        };
        // SAFETY-REVIEWED: a checked entry, called once for a parser `parser_open` returned,
        // while the frontend it was opened on is still open.
        closed_frontend_v1(unsafe { (self.frontend.export.parser_close)(parser.as_ptr()) })
    }
}

impl Drop for FrontendParserHandleV1 {
    fn drop(&mut self) {
        let _ = self.close_parser();
    }
}

#[cfg(test)]
mod frontend_tests {
    use super::*;
    use crate::frontend::{
        BundleFrontendParserV1, BundleFrontendSinkV1, BundleFrontendTableV1, BundleFrontendV1,
        FRONTEND_LAYOUT_V1, FRONTEND_PUSH_FLAG_V1_FINAL, FrontendCallErrorV1, FrontendParserV1,
        FrontendV1,
    };

    struct Echo;

    /// Aligned to 8, so a misaligned parser pointer is detectable.
    struct EchoParser {
        pushes: u64,
    }

    impl FrontendV1 for Echo {
        type Parser = EchoParser;

        fn open(component: &[u8]) -> Result<(Self, Vec<u8>), Vec<u8>> {
            Ok((Self, component.to_vec()))
        }

        fn render(&self, request: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
            Ok(request.to_vec())
        }

        fn constraint(&self, request: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
            Ok(request.to_vec())
        }

        fn open_parser(&self, _request: &[u8]) -> Result<EchoParser, Vec<u8>> {
            Ok(EchoParser { pushes: 0 })
        }
    }

    impl FrontendParserV1 for EchoParser {
        fn push(&mut self, text: &[u8], _last: bool) -> Result<Vec<u8>, Vec<u8>> {
            self.pushes += 1;
            Ok(text.to_vec())
        }
    }

    crate::export_frontend_v1!(Echo);

    /// The export of one of this module's static test tables.
    fn export_of(api: crate::BundleFrontendApiFnV1) -> Result<FrontendExportV1, BundleAbiError> {
        // SAFETY-REVIEWED: every table here is a static of this program whose entries
        // are this module's trampolines or test entries written to the header's contract.
        unsafe { FrontendExportV1::from_api(api) }
    }

    fn table() -> BundleFrontendTableV1 {
        // SAFETY-REVIEWED: the export returns its static table.
        unsafe { *chop_bundle_frontend_api_v1() }
    }

    /// A sink over a fresh collector, changed by `edit`, and what it
    /// collected once `call` returns.
    fn with_sink(
        edit: impl FnOnce(&mut BundleFrontendSinkV1),
        call: impl FnOnce(BundleFrontendSinkV1) -> BundleStatusV1,
    ) -> (BundleStatusV1, FrontendCollectorV1) {
        let mut collector = FrontendCollectorV1::with_maximum(Some(1 << 20));
        let mut sink = BundleFrontendSinkV1 {
            struct_size: struct_size_v1::<BundleFrontendSinkV1>(),
            layout: FRONTEND_LAYOUT_V1,
            context: ptr::from_mut(&mut collector).cast(),
            emit: Some(collect_frontend_answer_v1),
            emit_error: Some(collect_frontend_refusal_v1),
        };
        edit(&mut sink);
        let status = call(sink);
        (status, collector)
    }

    fn open(table: &BundleFrontendTableV1) -> *mut BundleFrontendV1 {
        let mut frontend = ptr::null_mut();
        let (status, _) = with_sink(
            |_| {},
            // SAFETY-REVIEWED: the component, sink and output outlive the call.
            |sink| unsafe {
                table.open.unwrap()(BundleByteViewV1::empty(), sink, &raw mut frontend)
            },
        );
        assert_eq!(status, BundleStatusV1::Success);
        frontend
    }

    fn render(
        table: &BundleFrontendTableV1,
        frontend: *const BundleFrontendV1,
        request: BundleByteViewV1,
        edit: impl FnOnce(&mut BundleFrontendSinkV1),
    ) -> (BundleStatusV1, FrontendCollectorV1) {
        // SAFETY-REVIEWED: the request and sink outlive the call; the frontend is open or deliberately invalid.
        with_sink(edit, |sink| unsafe {
            table.render.unwrap()(frontend, request, sink)
        })
    }

    #[test]
    fn every_entry_checks_its_sink_before_the_request() {
        let table = table();
        let frontend = open(&table);
        let request = BundleByteViewV1::from_bytes(b"chat");
        let (status, collector) = render(&table, frontend, request, |_| {});
        assert_eq!(status, BundleStatusV1::Success);
        assert_eq!(collector.answer.as_deref(), Some(&b"chat"[..]));

        let edits: [fn(&mut BundleFrontendSinkV1); 3] = [
            |sink| sink.struct_size -= 8,
            |sink| sink.emit = None,
            |sink| sink.emit_error = None,
        ];
        for edit in edits {
            let (status, collector) = render(&table, frontend, request, edit);
            assert_eq!(status, BundleStatusV1::InvalidArgument);
            assert!(collector.answer.is_none() && collector.refusal.is_none());
        }
        for layout in [0, FRONTEND_LAYOUT_V1 + 1] {
            let (status, collector) =
                render(&table, frontend, request, |sink| sink.layout = layout);
            assert_eq!(status, BundleStatusV1::BundleError);
            assert!(collector.answer.is_none());
            let refusal = String::from_utf8(collector.refusal.unwrap()).unwrap();
            assert!(
                refusal.contains(&format!("sink layout {layout}")),
                "{refusal}"
            );
        }
        // SAFETY-REVIEWED: the frontend is open and closed once.
        assert_eq!(
            unsafe { table.close.unwrap()(frontend) },
            BundleStatusV1::Success
        );
    }

    #[test]
    fn null_misaligned_and_malformed_arguments_are_refused() {
        let table = table();
        let frontend = open(&table);
        let request = BundleByteViewV1::from_bytes(b"chat");
        let invalid = BundleStatusV1::InvalidArgument;
        assert_eq!(render(&table, ptr::null(), request, |_| {}).0, invalid);
        let dangling = BundleByteViewV1 {
            data: ptr::null(),
            length: 4,
        };
        assert_eq!(render(&table, frontend, dangling, |_| {}).0, invalid);
        let (status, _) = with_sink(
            |_| {},
            // SAFETY-REVIEWED: the output pointer is deliberately null and refused before use.
            |sink| unsafe { table.open.unwrap()(request, sink, ptr::null_mut()) },
        );
        assert_eq!(status, invalid);
        let (status, _) = with_sink(
            |_| {},
            // SAFETY-REVIEWED: the output pointer is deliberately null and refused before use.
            |sink| unsafe { table.parser_open.unwrap()(frontend, request, sink, ptr::null_mut()) },
        );
        assert_eq!(status, invalid);
        let misaligned = ptr::without_provenance_mut::<BundleFrontendParserV1>(1);
        for parser in [ptr::null_mut(), misaligned] {
            let (status, _) = with_sink(
                |_| {},
                // SAFETY-REVIEWED: the parser pointer is deliberately invalid and refused before use.
                |sink| unsafe { table.parser_push.unwrap()(parser, request, 0, sink) },
            );
            assert_eq!(status, invalid);
            // SAFETY-REVIEWED: the parser pointer is deliberately invalid and refused before use.
            assert_eq!(unsafe { table.parser_close.unwrap()(parser) }, invalid);
        }
        // SAFETY-REVIEWED: the frontend pointer is deliberately null and refused before use.
        assert_eq!(unsafe { table.close.unwrap()(ptr::null_mut()) }, invalid);

        let mut parser = ptr::null_mut();
        let (status, _) = with_sink(
            |_| {},
            // SAFETY-REVIEWED: the frontend is open; the request, sink and output outlive the call.
            |sink| unsafe { table.parser_open.unwrap()(frontend, request, sink, &raw mut parser) },
        );
        assert_eq!(status, BundleStatusV1::Success);
        let push = |flags| {
            with_sink(
                |_| {},
                // SAFETY-REVIEWED: the parser is open; the text and sink outlive the call.
                |sink| unsafe { table.parser_push.unwrap()(parser, request, flags, sink) },
            )
        };
        assert_eq!(push(0).1.answer.as_deref(), Some(&b"chat"[..]));
        assert_eq!(push(FRONTEND_PUSH_FLAG_V1_FINAL | 4).0, invalid);
        // A push refused for its flags ends the parser like any failure.
        assert_eq!(
            push(FRONTEND_PUSH_FLAG_V1_FINAL).0,
            BundleStatusV1::InvalidState
        );
        // SAFETY-REVIEWED: the parser and then its frontend are open and closed once.
        unsafe {
            assert_eq!(table.parser_close.unwrap()(parser), BundleStatusV1::Success);
            assert_eq!(table.close.unwrap()(frontend), BundleStatusV1::Success);
        }
    }

    static WRONG_SIZE: BundleFrontendTableV1 = BundleFrontendTableV1 {
        struct_size: 8,
        ..frontend_table_v1::<Echo>()
    };
    static WRONG_LAYOUT: BundleFrontendTableV1 = BundleFrontendTableV1 {
        layout: FRONTEND_LAYOUT_V1 + 1,
        ..frontend_table_v1::<Echo>()
    };
    static MISSING_ENTRY: BundleFrontendTableV1 = BundleFrontendTableV1 {
        parser_push: None,
        ..frontend_table_v1::<Echo>()
    };

    extern "C" fn wrong_size() -> *const BundleFrontendTableV1 {
        &raw const WRONG_SIZE
    }

    extern "C" fn wrong_layout() -> *const BundleFrontendTableV1 {
        &raw const WRONG_LAYOUT
    }

    extern "C" fn missing_entry() -> *const BundleFrontendTableV1 {
        &raw const MISSING_ENTRY
    }

    extern "C" fn no_table() -> *const BundleFrontendTableV1 {
        ptr::null()
    }

    #[test]
    fn tables_of_another_size_or_layout_or_with_a_null_entry_are_refused() {
        assert!(export_of(chop_bundle_frontend_api_v1).is_ok());
        assert_eq!(
            export_of(wrong_size).unwrap_err(),
            BundleAbiError::StructSizeExact {
                surface: "frontend table",
                expected: struct_size_v1::<BundleFrontendTableV1>(),
                actual: 8,
            }
        );
        assert_eq!(
            export_of(wrong_layout).unwrap_err(),
            BundleAbiError::AbiVersion {
                surface: "frontend table layout",
                actual: FRONTEND_LAYOUT_V1 + 1,
            }
        );
        assert_eq!(
            export_of(missing_entry).unwrap_err(),
            BundleAbiError::MissingApiCallback("frontend parser_push")
        );
        assert_eq!(
            export_of(no_table).unwrap_err(),
            BundleAbiError::NullPointer("frontend table")
        );
    }

    // SAFETY-REVIEWED: a broken test entry that answers success without emitting.
    unsafe extern "C" fn silent(
        _frontend: *const BundleFrontendV1,
        _request: BundleByteViewV1,
        _sink: BundleFrontendSinkV1,
    ) -> BundleStatusV1 {
        BundleStatusV1::Success
    }

    // SAFETY-REVIEWED: a broken test entry that emits twice through the lent sink.
    unsafe extern "C" fn twice(
        _frontend: *const BundleFrontendV1,
        request: BundleByteViewV1,
        sink: BundleFrontendSinkV1,
    ) -> BundleStatusV1 {
        // SAFETY-REVIEWED: the host lent the sink and request for the call.
        unsafe {
            sink.emit.unwrap()(sink.context, request);
            sink.emit.unwrap()(sink.context, request)
        }
    }

    // SAFETY-REVIEWED: a broken test entry that refuses without a document.
    unsafe extern "C" fn bare_refusal(
        _frontend: *const BundleFrontendV1,
        _request: BundleByteViewV1,
        _sink: BundleFrontendSinkV1,
    ) -> BundleStatusV1 {
        BundleStatusV1::BundleError
    }

    // SAFETY-REVIEWED: a broken test entry that answers but opens nothing.
    unsafe extern "C" fn open_nothing(
        component: BundleByteViewV1,
        sink: BundleFrontendSinkV1,
        _frontend: *mut *mut BundleFrontendV1,
    ) -> BundleStatusV1 {
        // SAFETY-REVIEWED: the host lent the sink and view for the call.
        unsafe { sink.emit.unwrap()(sink.context, component) }
    }

    static BROKEN: BundleFrontendTableV1 = BundleFrontendTableV1 {
        render: Some(silent),
        constraint: Some(twice),
        ..frontend_table_v1::<Echo>()
    };
    static REFUSES_BARE: BundleFrontendTableV1 = BundleFrontendTableV1 {
        render: Some(bare_refusal),
        ..frontend_table_v1::<Echo>()
    };
    static OPENS_NOTHING: BundleFrontendTableV1 = BundleFrontendTableV1 {
        open: Some(open_nothing),
        ..frontend_table_v1::<Echo>()
    };

    extern "C" fn broken() -> *const BundleFrontendTableV1 {
        &raw const BROKEN
    }

    extern "C" fn refuses_bare() -> *const BundleFrontendTableV1 {
        &raw const REFUSES_BARE
    }

    extern "C" fn opens_nothing() -> *const BundleFrontendTableV1 {
        &raw const OPENS_NOTHING
    }

    #[test]
    fn a_bundle_that_breaks_the_call_contract_is_named() {
        let frontend = export_of(broken).unwrap().open(b"").unwrap().0;
        assert_eq!(
            frontend.render(b"chat", 1 << 20).unwrap_err(),
            FrontendCallErrorV1::Contract("a successful call emitted no answer")
        );
        assert_eq!(
            frontend.constraint(b"chat", 1 << 20).unwrap_err(),
            FrontendCallErrorV1::Contract("a call emitted twice")
        );
        frontend.close().unwrap();
        let frontend = export_of(refuses_bare).unwrap().open(b"").unwrap().0;
        assert_eq!(
            frontend.render(b"chat", 1 << 20).unwrap_err(),
            FrontendCallErrorV1::Status(BundleStatusV1::BundleError)
        );
        assert_eq!(
            export_of(opens_nothing).unwrap().open(b"").err(),
            Some(FrontendCallErrorV1::Contract(
                "a successful open returned null"
            ))
        );
    }

    type SinkEditV1 = fn(&mut BundleFrontendSinkV1);

    /// A parser takes no push after a failed one, whatever failed: the
    /// sink, the layout, the flags or the text view.
    #[test]
    fn the_bundle_parser_takes_no_push_after_a_failed_one() {
        let table = table();
        let frontend = open(&table);
        let text = BundleByteViewV1::from_bytes(b"text");
        let breaks: [(SinkEditV1, u32, BundleByteViewV1); 5] = [
            (|sink| sink.struct_size -= 8, 0, text),
            (|sink| sink.emit = None, 0, text),
            (|sink| sink.layout += 1, 0, text),
            (|_| {}, 2, text),
            (
                |_| {},
                0,
                BundleByteViewV1 {
                    data: ptr::null(),
                    length: 4,
                },
            ),
        ];
        for (edit, flags, view) in breaks {
            let mut parser = ptr::null_mut();
            let (status, _) = with_sink(
                |_| {},
                // SAFETY-REVIEWED: the frontend is open; the request, sink and output outlive the call.
                |sink| unsafe { table.parser_open.unwrap()(frontend, text, sink, &raw mut parser) },
            );
            assert_eq!(status, BundleStatusV1::Success);
            let (status, collector) = with_sink(
                edit,
                // SAFETY-REVIEWED: the parser is open; the view and sink outlive the call or are refused.
                |sink| unsafe { table.parser_push.unwrap()(parser, view, flags, sink) },
            );
            assert!(!status.is_success());
            assert!(collector.answer.is_none());
            let (status, collector) = with_sink(
                |_| {},
                // SAFETY-REVIEWED: the parser is open; the text and sink outlive the call.
                |sink| unsafe { table.parser_push.unwrap()(parser, text, 0, sink) },
            );
            assert_eq!(status, BundleStatusV1::InvalidState);
            assert!(collector.answer.is_none());
            // SAFETY-REVIEWED: the parser is open and closed once.
            assert_eq!(
                unsafe { table.parser_close.unwrap()(parser) },
                BundleStatusV1::Success
            );
        }
        // SAFETY-REVIEWED: the frontend is open and closed once.
        assert_eq!(
            unsafe { table.close.unwrap()(frontend) },
            BundleStatusV1::Success
        );
    }

    static LENIENT_PUSHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static LENIENT_PARSER: u64 = 0;

    // SAFETY-REVIEWED: a lenient test entry that writes the host's output pointer.
    unsafe extern "C" fn lenient_parser_open(
        _frontend: *const BundleFrontendV1,
        _request: BundleByteViewV1,
        _sink: BundleFrontendSinkV1,
        parser: *mut *mut BundleFrontendParserV1,
    ) -> BundleStatusV1 {
        // SAFETY-REVIEWED: the host lends a writable output pointer for the call.
        unsafe { *parser = (&raw const LENIENT_PARSER).cast_mut().cast() };
        BundleStatusV1::Success
    }

    // SAFETY-REVIEWED: a lenient test entry that answers every push, even after the final one.
    unsafe extern "C" fn lenient_push(
        _parser: *mut BundleFrontendParserV1,
        text: BundleByteViewV1,
        _flags: u32,
        sink: BundleFrontendSinkV1,
    ) -> BundleStatusV1 {
        LENIENT_PUSHES.fetch_add(1, Ordering::SeqCst);
        // SAFETY-REVIEWED: the host lent the sink and text for the call.
        unsafe {
            if text.length == 6 {
                sink.emit_error.unwrap()(sink.context, text);
                return BundleStatusV1::BundleError;
            }
            sink.emit.unwrap()(sink.context, text)
        }
    }

    // SAFETY-REVIEWED: a lenient test entry with nothing to release.
    unsafe extern "C" fn lenient_parser_close(
        _parser: *mut BundleFrontendParserV1,
    ) -> BundleStatusV1 {
        BundleStatusV1::Success
    }

    static LENIENT: BundleFrontendTableV1 = BundleFrontendTableV1 {
        parser_open: Some(lenient_parser_open),
        parser_push: Some(lenient_push),
        parser_close: Some(lenient_parser_close),
        ..frontend_table_v1::<Echo>()
    };

    extern "C" fn lenient() -> *const BundleFrontendTableV1 {
        &raw const LENIENT
    }

    /// The host handle refuses a push after the final push or a failed one
    /// itself, without calling a bundle that would accept it.
    #[test]
    fn the_host_parser_takes_no_push_after_the_final_or_a_failed_one() {
        let frontend = export_of(lenient).unwrap().open(b"").unwrap().0;
        let pushes = || LENIENT_PUSHES.load(Ordering::SeqCst);
        let mut parser = frontend.open_parser(b"").unwrap();
        assert_eq!(parser.push(b"a", false, 1 << 20).unwrap(), b"a");
        assert_eq!(parser.push(b"b", true, 1 << 20).unwrap(), b"b");
        let before = pushes();
        for (text, last) in [(&b"c"[..], false), (b"", true)] {
            assert_eq!(
                parser.push(text, last, 1 << 20).unwrap_err(),
                FrontendCallErrorV1::ParserFinished
            );
        }
        assert_eq!(pushes(), before);

        let mut parser = frontend.open_parser(b"").unwrap();
        assert!(matches!(
            parser.push(b"refuse", false, 1 << 20),
            Err(FrontendCallErrorV1::Refused(_))
        ));
        let before = pushes();
        assert_eq!(
            parser.push(b"c", false, 1 << 20).unwrap_err(),
            FrontendCallErrorV1::ParserFinished
        );
        assert_eq!(pushes(), before);
    }

    // SAFETY-REVIEWED: a test entry that misuses the lent sink as its request names.
    unsafe extern "C" fn misuse(
        _frontend: *const BundleFrontendV1,
        request: BundleByteViewV1,
        sink: BundleFrontendSinkV1,
    ) -> BundleStatusV1 {
        // SAFETY-REVIEWED: the host lent the request for the call.
        let script = unsafe { slice::from_raw_parts(request.data, request.length as usize) };
        let answer = BundleByteViewV1::from_bytes(b"answer");
        let malformed = BundleByteViewV1 {
            data: ptr::null(),
            length: 4,
        };
        for &step in &script[..script.len() - 1] {
            let (emit, view) = match step {
                b'a' => (sink.emit, answer),
                b'e' => (sink.emit_error, answer),
                b'm' => (sink.emit, malformed),
                _ => (sink.emit_error, malformed),
            };
            // SAFETY-REVIEWED: the host lent the sink for the call; malformed views are deliberate.
            let _ = unsafe { emit.unwrap()(sink.context, view) };
        }
        match script[script.len() - 1] {
            b'S' => BundleStatusV1::Success,
            b'P' => BundleStatusV1::Panic,
            _ => BundleStatusV1::BundleError,
        }
    }

    static MISUSES: BundleFrontendTableV1 = BundleFrontendTableV1 {
        render: Some(misuse),
        ..frontend_table_v1::<Echo>()
    };

    extern "C" fn misuses() -> *const BundleFrontendTableV1 {
        &raw const MISUSES
    }

    /// One invalid sink call fails the whole call, whatever follows it.
    #[test]
    fn one_invalid_sink_call_fails_the_whole_call() {
        let frontend = export_of(misuses).unwrap().open(b"").unwrap().0;
        assert_eq!(frontend.render(b"aS", 1 << 20).unwrap(), b"answer");
        assert_eq!(
            frontend.render(b"eB", 1 << 20).unwrap_err(),
            FrontendCallErrorV1::Refused(b"answer".to_vec())
        );
        for script in [
            "maS", "mS", "xaS", "xeB", "meB", "eaS", "eaB", "aeS", "aeB", "aaS", "eeB", "amS",
        ] {
            assert!(
                frontend.render(script.as_bytes(), 1 << 20).is_err(),
                "{script} must fail the call"
            );
        }
        assert_eq!(
            frontend.render(b"maS", 1 << 20).unwrap_err(),
            FrontendCallErrorV1::Contract("a call emitted a malformed view")
        );
        assert_eq!(
            frontend.render(b"xeB", 1 << 20).unwrap_err(),
            FrontendCallErrorV1::Contract("a call emitted a malformed view")
        );
    }

    /// A recorded sink violation outranks the status the entry returns,
    /// a panic included.
    #[test]
    fn a_sink_violation_outranks_a_panic_status() {
        let frontend = export_of(misuses).unwrap().open(b"").unwrap().0;
        assert_eq!(
            frontend.render(b"P", 1 << 20).unwrap_err(),
            FrontendCallErrorV1::Panicked
        );
        for script in ["mP", "aaP", "xP"] {
            assert!(
                matches!(
                    frontend.render(script.as_bytes(), 1 << 20).unwrap_err(),
                    FrontendCallErrorV1::Contract(_)
                ),
                "{script}"
            );
        }
    }

    static CLEANUP_CLOSES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static CLEANUP_OPENED: u64 = 0;

    // SAFETY-REVIEWED: a test entry that opens, but answers twice through the sink.
    unsafe extern "C" fn open_and_emit(
        component: BundleByteViewV1,
        sink: BundleFrontendSinkV1,
        frontend: *mut *mut BundleFrontendV1,
    ) -> BundleStatusV1 {
        // SAFETY-REVIEWED: the host lent the sink, view and output for the call.
        unsafe {
            sink.emit.unwrap()(sink.context, component);
            sink.emit.unwrap()(sink.context, component);
            *frontend = (&raw const CLEANUP_OPENED).cast_mut().cast();
        }
        BundleStatusV1::Success
    }

    // SAFETY-REVIEWED: a test entry that opens without an answer.
    unsafe extern "C" fn open_silently(
        _component: BundleByteViewV1,
        _sink: BundleFrontendSinkV1,
        frontend: *mut *mut BundleFrontendV1,
    ) -> BundleStatusV1 {
        // SAFETY-REVIEWED: the host lent the output for the call.
        unsafe {
            *frontend = (&raw const CLEANUP_OPENED).cast_mut().cast();
        }
        BundleStatusV1::Success
    }

    // SAFETY-REVIEWED: a test entry that counts closes of the static it opened.
    unsafe extern "C" fn count_close(frontend: *mut BundleFrontendV1) -> BundleStatusV1 {
        assert_eq!(frontend.cast_const().cast(), &raw const CLEANUP_OPENED);
        CLEANUP_CLOSES.fetch_add(1, Ordering::SeqCst);
        BundleStatusV1::Success
    }

    // SAFETY-REVIEWED: a test entry that opens a parser, but emits a malformed view first.
    unsafe extern "C" fn parser_open_and_emit(
        _frontend: *const BundleFrontendV1,
        _request: BundleByteViewV1,
        sink: BundleFrontendSinkV1,
        parser: *mut *mut BundleFrontendParserV1,
    ) -> BundleStatusV1 {
        let malformed = BundleByteViewV1 {
            data: ptr::null(),
            length: 1,
        };
        // SAFETY-REVIEWED: the host lent the sink and output for the call.
        unsafe {
            sink.emit.unwrap()(sink.context, malformed);
            *parser = (&raw const CLEANUP_OPENED).cast_mut().cast();
        }
        BundleStatusV1::Success
    }

    // SAFETY-REVIEWED: a test entry that counts closes of the static it opened.
    unsafe extern "C" fn count_parser_close(parser: *mut BundleFrontendParserV1) -> BundleStatusV1 {
        assert_eq!(parser.cast_const().cast(), &raw const CLEANUP_OPENED);
        CLEANUP_CLOSES.fetch_add(1, Ordering::SeqCst);
        BundleStatusV1::Success
    }

    static OPENS_AND_EMITS: BundleFrontendTableV1 = BundleFrontendTableV1 {
        open: Some(open_and_emit),
        close: Some(count_close),
        ..frontend_table_v1::<Echo>()
    };
    static OPENS_SILENTLY: BundleFrontendTableV1 = BundleFrontendTableV1 {
        open: Some(open_silently),
        close: Some(count_close),
        ..frontend_table_v1::<Echo>()
    };
    static PARSER_OPENS_AND_EMITS: BundleFrontendTableV1 = BundleFrontendTableV1 {
        parser_open: Some(parser_open_and_emit),
        parser_close: Some(count_parser_close),
        ..frontend_table_v1::<Echo>()
    };

    extern "C" fn opens_and_emits() -> *const BundleFrontendTableV1 {
        &raw const OPENS_AND_EMITS
    }

    extern "C" fn opens_silently() -> *const BundleFrontendTableV1 {
        &raw const OPENS_SILENTLY
    }

    extern "C" fn parser_opens_and_emits() -> *const BundleFrontendTableV1 {
        &raw const PARSER_OPENS_AND_EMITS
    }

    /// An open or parser open the bundle reports successful, but whose sink
    /// calls the host refused or that gave no answer it owed, fails, and the
    /// host closes what was opened.
    #[test]
    fn a_refused_successful_open_is_closed() {
        let closes = || CLEANUP_CLOSES.load(Ordering::SeqCst);
        let before = closes();
        assert_eq!(
            export_of(opens_and_emits).unwrap().open(b"component").err(),
            Some(FrontendCallErrorV1::Contract("a call emitted twice"))
        );
        assert_eq!(closes(), before + 1);
        assert_eq!(
            export_of(opens_silently).unwrap().open(b"component").err(),
            Some(FrontendCallErrorV1::Contract(
                "a successful open emitted no answer"
            ))
        );
        let before = before + 1;
        assert_eq!(closes(), before + 1);
        let frontend = export_of(parser_opens_and_emits)
            .unwrap()
            .open(b"")
            .unwrap()
            .0;
        assert_eq!(
            frontend.open_parser(b"request").err(),
            Some(FrontendCallErrorV1::Contract(
                "a call emitted a malformed view"
            ))
        );
        assert_eq!(closes(), before + 2);
    }

    // SAFETY-REVIEWED: a test sink callback that refuses every document.
    unsafe extern "C" fn refuse_document(
        _context: *mut c_void,
        _document: BundleByteViewV1,
    ) -> BundleStatusV1 {
        BundleStatusV1::HostError
    }

    /// When the host's emit callback fails, the entry returns that status,
    /// and a parser whose answer was not taken takes no further push.
    #[test]
    fn a_failed_emit_fails_the_entry_and_ends_the_parser() {
        let table = table();
        let frontend = open(&table);
        let request = BundleByteViewV1::from_bytes(b"chat");
        let (status, _) = render(&table, frontend, request, |sink| {
            sink.emit = Some(refuse_document);
        });
        assert_eq!(status, BundleStatusV1::HostError);
        let mut parser = ptr::null_mut();
        let (status, _) = with_sink(
            |_| {},
            // SAFETY-REVIEWED: the frontend is open; the request, sink and output outlive the call.
            |sink| unsafe { table.parser_open.unwrap()(frontend, request, sink, &raw mut parser) },
        );
        assert_eq!(status, BundleStatusV1::Success);
        let push = |edit: fn(&mut BundleFrontendSinkV1)| {
            with_sink(
                edit,
                // SAFETY-REVIEWED: the parser is open; the text and sink outlive the call.
                |sink| unsafe { table.parser_push.unwrap()(parser, request, 0, sink) },
            )
            .0
        };
        assert_eq!(
            push(|sink| sink.emit = Some(refuse_document)),
            BundleStatusV1::HostError
        );
        assert_eq!(push(|_| {}), BundleStatusV1::InvalidState);
        // SAFETY-REVIEWED: the parser and then its frontend are open and closed once.
        unsafe {
            assert_eq!(table.parser_close.unwrap()(parser), BundleStatusV1::Success);
            assert_eq!(table.close.unwrap()(frontend), BundleStatusV1::Success);
        }
    }

    /// The bundle refuses to close a frontend while a parser opened on it is
    /// open, and the frontend stays usable until it is closed after them.
    #[test]
    fn a_frontend_with_an_open_parser_refuses_to_close() {
        let table = table();
        let frontend = open(&table);
        let request = BundleByteViewV1::from_bytes(b"chat");
        let open_parser = || {
            let mut parser = ptr::null_mut();
            let (status, _) = with_sink(
                |_| {},
                // SAFETY-REVIEWED: the frontend is open; the request, sink and output outlive the call.
                |sink| unsafe {
                    table.parser_open.unwrap()(frontend, request, sink, &raw mut parser)
                },
            );
            assert_eq!(status, BundleStatusV1::Success);
            parser
        };
        let parsers = [open_parser(), open_parser()];
        // A refused parser open counts nothing.
        let mut refused = ptr::null_mut();
        let (status, _) = with_sink(
            |sink| sink.layout += 1,
            // SAFETY-REVIEWED: the frontend is open; the request, sink and output outlive the call.
            |sink| unsafe { table.parser_open.unwrap()(frontend, request, sink, &raw mut refused) },
        );
        assert_eq!(status, BundleStatusV1::BundleError);
        for parser in parsers {
            // SAFETY-REVIEWED: the frontend is open; its close is refused while a parser is.
            assert_eq!(
                unsafe { table.close.unwrap()(frontend) },
                BundleStatusV1::InvalidState
            );
            assert_eq!(
                render(&table, frontend, request, |_| {})
                    .1
                    .answer
                    .as_deref(),
                Some(&b"chat"[..])
            );
            // SAFETY-REVIEWED: each parser is open and closed once.
            assert_eq!(
                unsafe { table.parser_close.unwrap()(parser) },
                BundleStatusV1::Success
            );
        }
        // SAFETY-REVIEWED: the frontend is open, with no parser, and closed once.
        assert_eq!(
            unsafe { table.close.unwrap()(frontend) },
            BundleStatusV1::Success
        );
    }
}

// The device-result extension keeps the legacy by-value sink unchanged.
use crate::device_results::{
    BundleDeviceCompletionV1, BundleDeviceCycleViewV1, BundleDeviceDraftsV1,
    BundleDeviceResultSinkV1, BundleDeviceSamplingRequestV1, BundleExecuteDeviceResultsFnV1,
    BundleReadyLogitsViewV1, DEVICE_CYCLE_ITEM_BYTES_V1, DeviceCycleMetadataV1,
};

struct EncodedDeviceCycleV1 {
    epoch: u64,
    items: Vec<u8>,
    stops: Vec<u8>,
}

pub(crate) fn invoke_bundle_device_results_v1<S: LogitsRowSinkV1>(
    execute: BundleExecuteDeviceResultsFnV1,
    handle: BundleHandleV1,
    request: &[u8],
    sink: &mut S,
) -> Result<BundleStatusV1, BundleAbiError> {
    let metadata = match sink.device_cycle_metadata() {
        Ok(metadata) => {
            metadata
                .validate(sink.limits())
                .map_err(BundleAbiError::Status)?;
            Some(EncodedDeviceCycleV1 {
                epoch: metadata.cycle_epoch,
                items: metadata.encode_items(),
                stops: metadata
                    .termination_stop_ids
                    .iter()
                    .flat_map(|id| id.to_le_bytes())
                    .collect(),
            })
        }
        Err(BundleStatusV1::Unsupported) => None,
        Err(status) => return Err(BundleAbiError::Status(status)),
    };
    with_result_sink_v1(sink, |base| {
        result_sink_state_from_user_data_v1::<S>(base.user_data)?
            .borrow_mut()
            .device_cycle = metadata;
        let raw = BundleDeviceResultSinkV1 {
            struct_size: struct_size_v1::<BundleDeviceResultSinkV1>(),
            abi_version: BUNDLE_ABI_VERSION_V1,
            base,
            describe_device_cycle_v1: Some(describe_device_cycle_trampoline_v1::<S>),
            declare_input_dependency_v1: Some(declare_input_dependency_trampoline_v1::<S>),
            emit_device_drafts_v1: Some(emit_device_drafts_trampoline_v1::<S>),
            sample_device_logits_v1: Some(sample_device_logits_trampoline_v1::<S>),
            emit_ready_logits_view_v1: Some(emit_ready_logits_trampoline_v1::<S>),
            emit_device_completion_v1: Some(emit_device_completion_trampoline_v1::<S>),
        };
        // SAFETY-REVIEWED: request, table, metadata and user_data live through this exact call.
        Ok(unsafe { execute(handle, BundleByteViewV1::from_bytes(request), &raw) })
    })
}

/// Decode one borrowed record with a size prefix, without reading a short tail.
/// # Safety
/// The pointer is the borrowed record supplied by the native callback. Its
/// header describes the readable bytes for this call.
// SAFETY-REVIEWED: called only by the typed extension entry and callbacks below.
unsafe fn read_device_result_record_v1<T: Copy>(pointer: *const T) -> Result<T, BundleAbiError> {
    validate_input_pointer_v1(pointer, "device result record")?;
    // SAFETY-REVIEWED: a native record supplies at least its fixed size prefix.
    let size = unsafe { pointer.cast::<u32>().read() };
    if (size as usize) < std::mem::size_of::<T>() {
        return Err(BundleAbiError::Status(BundleStatusV1::Unsupported));
    }
    // SAFETY-REVIEWED: the declared size covers T and the native caller lends that record.
    Ok(unsafe { pointer.read() })
}

/// Decode the optional execute entry without extending the legacy signature.
/// # Safety
/// The native caller lends the request, extended sink and its user_data for
/// this call. The sink's size prefix describes its readable bytes.
// SAFETY-REVIEWED: owns the pointer decode for the generated extension entry.
pub unsafe fn dispatch_bundle_execute_device_results_v1(
    handle: BundleHandleV1,
    request: BundleByteViewV1,
    sink: *const BundleDeviceResultSinkV1,
    execute: impl FnOnce(BundleHandleV1, &[u8], &mut ExecuteResultSinkV1) -> BundleStatusV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        // SAFETY-REVIEWED: these are the native entry's borrowed record and bytes.
        let table = unsafe { read_device_result_record_v1(sink)? };
        table.validate()?;
        let mut sink = ExecuteResultSinkV1::new(table.base)?.with_device_results(table)?;
        with_byte_view_v1(request, |request| execute(handle, request, &mut sink))
    })
}

pub(crate) fn device_results_describe(
    table: &BundleDeviceResultSinkV1,
    limits: ResultSinkLimitsV1,
) -> Result<DeviceCycleMetadataV1, BundleAbiError> {
    let callback = table
        .describe_device_cycle_v1
        .ok_or(BundleAbiError::MissingExecuteResultCallback)?;
    let mut raw = BundleDeviceCycleViewV1::empty();
    // SAFETY-REVIEWED: the enclosing execute owns the table and this output record.
    status_result(unsafe { callback(table.base.user_data, &mut raw) })?;
    if raw.struct_size != struct_size_v1::<BundleDeviceCycleViewV1>()
        || raw.reserved != 0
        || raw.items.length / DEVICE_CYCLE_ITEM_BYTES_V1 as u64 > limits.maximum_rows
        || raw.items.length > (1 << 20) * DEVICE_CYCLE_ITEM_BYTES_V1 as u64
        || raw.termination_stop_ids.length > (1 << 20) * 4
    {
        return Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument));
    }
    // The getter retains immutable storage through the enclosing execute.
    let items = copy_byte_view_v1(raw.items)?;
    let stops = copy_byte_view_v1(raw.termination_stop_ids)?;
    DeviceCycleMetadataV1::decode(raw.cycle_epoch, &items, &stops, limits)
        .map_err(BundleAbiError::Status)
}

pub(crate) fn device_results_dependency(
    table: &BundleDeviceResultSinkV1,
    dependency: u32,
) -> Result<(), BundleAbiError> {
    let callback = table
        .declare_input_dependency_v1
        .ok_or(BundleAbiError::MissingExecuteResultCallback)?;
    // SAFETY-REVIEWED: the enclosing execute owns the table and its user_data.
    status_result(unsafe { callback(table.base.user_data, dependency) })
}

macro_rules! invoke_device_result_record {
    ($name:ident, $field:ident, $record:ty) => {
        pub(crate) fn $name(
            table: &BundleDeviceResultSinkV1,
            record: $record,
        ) -> Result<(), BundleAbiError> {
            let callback = table
                .$field
                .ok_or(BundleAbiError::MissingExecuteResultCallback)?;
            // SAFETY-REVIEWED: table and borrowed fixed-width record live through the call.
            status_result(unsafe { callback(table.base.user_data, &record) })
        }
    };
}
invoke_device_result_record!(
    device_results_drafts,
    emit_device_drafts_v1,
    BundleDeviceDraftsV1
);
invoke_device_result_record!(
    device_results_sample,
    sample_device_logits_v1,
    BundleDeviceSamplingRequestV1
);
invoke_device_result_record!(
    device_results_ready,
    emit_ready_logits_view_v1,
    BundleReadyLogitsViewV1
);
invoke_device_result_record!(
    device_results_completion,
    emit_device_completion_v1,
    BundleDeviceCompletionV1
);

// SAFETY-REVIEWED: the execute owns user_data and the caller lends one writable output record.
unsafe extern "C" fn describe_device_cycle_trampoline_v1<S: LogitsRowSinkV1>(
    user_data: *mut c_void,
    output: *mut BundleDeviceCycleViewV1,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        validate_output_pointer_v1(output, "device cycle metadata")?;
        let state = result_sink_state_from_user_data_v1::<S>(user_data)?;
        let state = state
            .try_borrow()
            .map_err(|_| BundleAbiError::Lifecycle("reentrant result sink"))?;
        let Some(metadata) = &state.device_cycle else {
            return Ok(BundleStatusV1::Unsupported);
        };
        let raw = BundleDeviceCycleViewV1 {
            cycle_epoch: metadata.epoch,
            items: BundleByteViewV1::from_bytes(&metadata.items),
            termination_stop_ids: BundleByteViewV1::from_bytes(&metadata.stops),
            ..BundleDeviceCycleViewV1::empty()
        };
        // SAFETY-REVIEWED: output is writable; the backing vectors stay immutable through execute.
        unsafe {
            output.write(raw);
        }
        Ok(BundleStatusV1::Success)
    })
}

// SAFETY-REVIEWED: user_data is installed by this execute's sink wrapper.
unsafe extern "C" fn declare_input_dependency_trampoline_v1<S: LogitsRowSinkV1>(
    user_data: *mut c_void,
    dependency: u32,
) -> BundleStatusV1 {
    catch_panic_callback_status_v1(|| {
        if dependency > 1 {
            return Ok(BundleStatusV1::Unsupported);
        }
        let state = result_sink_state_from_user_data_v1::<S>(user_data)?;
        let mut state = state
            .try_borrow_mut()
            .map_err(|_| BundleAbiError::Lifecycle("reentrant result sink"))?;
        Ok(host_unit_status_v1(
            state.sink.declare_input_dependency(dependency),
        ))
    })
}

macro_rules! receive_device_result_record {
    ($name:ident, $record:ty, $validate:expr, $handler:expr) => {
        // SAFETY-REVIEWED: the enclosing execute lends this record and installs user_data.
        unsafe extern "C" fn $name<S: LogitsRowSinkV1>(
            user_data: *mut c_void,
            pointer: *const $record,
        ) -> BundleStatusV1 {
            catch_panic_callback_status_v1(|| {
                // SAFETY-REVIEWED: the native callback supplied this typed, size-prefixed record.
                let record = unsafe { read_device_result_record_v1(pointer)? };
                ($validate)(record)?;
                let state = result_sink_state_from_user_data_v1::<S>(user_data)?;
                let mut state = state
                    .try_borrow_mut()
                    .map_err(|_| BundleAbiError::Lifecycle("reentrant result sink"))?;
                ($handler)(&mut *state, record)
            })
        }
    };
}
receive_device_result_record!(
    emit_device_drafts_trampoline_v1,
    BundleDeviceDraftsV1,
    |record: BundleDeviceDraftsV1| record.validate().map_err(BundleAbiError::Status),
    |state: &mut ResultSinkStateV1<'_, S>, record| Ok(host_unit_status_v1(
        state.sink.emit_device_drafts(record)
    ))
);
receive_device_result_record!(
    sample_device_logits_trampoline_v1,
    BundleDeviceSamplingRequestV1,
    |record: BundleDeviceSamplingRequestV1| record.validate(),
    |state: &mut ResultSinkStateV1<'_, S>, record: BundleDeviceSamplingRequestV1| {
        charge_result_capacity_rows_v1(
            state.limits,
            &mut state.emitted_rows,
            &mut state.emitted_logits,
            u64::from(record.logits.logits.row_count),
            u64::from(record.logits.logits.live_width),
        )?;
        Ok(host_unit_status_v1(state.sink.sample_device_logits(record)))
    }
);
receive_device_result_record!(
    emit_ready_logits_trampoline_v1,
    BundleReadyLogitsViewV1,
    |record: BundleReadyLogitsViewV1| record.validate(),
    |state: &mut ResultSinkStateV1<'_, S>, record: BundleReadyLogitsViewV1| {
        charge_result_capacity_rows_v1(
            state.limits,
            &mut state.emitted_rows,
            &mut state.emitted_logits,
            u64::from(record.logits.row_count),
            u64::from(record.logits.live_width),
        )?;
        Ok(host_unit_status_v1(
            state.sink.emit_ready_logits_view(record),
        ))
    }
);
receive_device_result_record!(
    emit_device_completion_trampoline_v1,
    BundleDeviceCompletionV1,
    |record: BundleDeviceCompletionV1| record.validate().map_err(BundleAbiError::Status),
    |state: &mut ResultSinkStateV1<'_, S>, record: BundleDeviceCompletionV1| {
        if u64::from(record.sample_rows) > state.limits.maximum_rows {
            return Err(BundleAbiError::ExecuteResultCapacity);
        }
        Ok(host_unit_status_v1(
            state.sink.emit_device_completion(record),
        ))
    }
);

/// Export the optional execute entry without changing the legacy API table.
#[macro_export]
macro_rules! export_bundle_device_results_v1 {
    ($api:ident, $execute:expr $(,)?) => {
        // SAFETY-REVIEWED: the generated entry delegates its borrowed pointers to the ABI decoder.
        #[allow(unsafe_code)]
        unsafe extern "C" fn __bundle_execute_device_results_v1(
            handle: $crate::BundleHandleV1, request: $crate::BundleByteViewV1,
            sink: *const $crate::device_results::BundleDeviceResultSinkV1,
        ) -> $crate::BundleStatusV1 {
            // SAFETY-REVIEWED: these are the exact arguments received from the native host.
            let execute = $execute;
            // SAFETY-REVIEWED: the native entry lends its request and sink for this call.
            unsafe { $crate::dispatch_bundle_execute_device_results_v1(handle, request, sink, execute) }
        }
        pub static $api: $crate::device_results::BundleDeviceResultsApiV1 = $crate::device_results::BundleDeviceResultsApiV1 {
            struct_size: ::core::mem::size_of::<$crate::device_results::BundleDeviceResultsApiV1>() as u32,
            abi_version: $crate::BUNDLE_ABI_VERSION_V1,
            execute_device_results_v1: Some(__bundle_execute_device_results_v1),
        };
        // SAFETY-REVIEWED: this export returns a live immutable static table with the declared C layout.
        #[allow(unsafe_code)]
        #[unsafe(no_mangle)]
        pub extern "C" fn chop_bundle_get_device_results_api_v1() -> *const $crate::device_results::BundleDeviceResultsApiV1 {
            &$api
        }
    };
}

#[cfg(test)]
mod device_result_hook_tests {
    use super::*;
    use crate::BUNDLE_LOGITS_ELEMENT_V1_F32;
    use crate::device_results::*;

    #[derive(Default)]
    struct Sink {
        calls: Vec<&'static str>,
    }
    impl LogitsRowSinkV1 for Sink {
        fn limits(&self) -> ResultSinkLimitsV1 {
            ResultSinkLimitsV1::new(4, 8, 32)
        }
        fn emit_row(&mut self, _: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
            self.calls.push("legacy");
            Ok(())
        }
        fn device_results_enabled(&self) -> bool {
            true
        }
        fn device_cycle_metadata(&self) -> HostServiceResultV1<DeviceCycleMetadataV1> {
            Ok(DeviceCycleMetadataV1 {
                cycle_epoch: 9,
                items: vec![DeviceCycleItemV1 {
                    request_id: 11,
                    slot_generation: 2,
                    acceptance: BundleAcceptanceInputV1 {
                        cycle_epoch: 9,
                        remaining_tokens: 4,
                        first_sample_row: 0,
                        draft_item_index: 0,
                        draft_tokens: 1,
                        stop_offset: 0,
                        stop_count: 1,
                        reserved: 0,
                    },
                }],
                termination_stop_ids: vec![3],
            })
        }
        fn declare_input_dependency(&mut self, _: u32) -> HostServiceResultV1<()> {
            self.calls.push("dependency");
            Ok(())
        }
        fn emit_device_drafts(&mut self, _: BundleDeviceDraftsV1) -> HostServiceResultV1<()> {
            self.calls.push("drafts");
            Ok(())
        }
        fn sample_device_logits(
            &mut self,
            _: BundleDeviceSamplingRequestV1,
        ) -> HostServiceResultV1<()> {
            self.calls.push("sample");
            Ok(())
        }
        fn emit_ready_logits_view(
            &mut self,
            _: BundleReadyLogitsViewV1,
        ) -> HostServiceResultV1<()> {
            self.calls.push("ready");
            Ok(())
        }
        fn emit_device_completion(
            &mut self,
            _: BundleDeviceCompletionV1,
        ) -> HostServiceResultV1<()> {
            self.calls.push("completion");
            Ok(())
        }
    }
    fn buffer(slot: u32, length: u64) -> BundleDeviceBufferViewV1 {
        BundleDeviceBufferViewV1 {
            buffer: BundleDeviceBufferHandleV1::from_parts(1, slot),
            length,
            ..BundleDeviceBufferViewV1::empty()
        }
    }
    fn ready(rows: u32) -> BundleReadyLogitsViewV1 {
        BundleReadyLogitsViewV1 {
            struct_size: struct_size_v1::<BundleReadyLogitsViewV1>(),
            flags: 0,
            cycle_epoch: 9,
            ready: BundleEventHandleV1::from_parts(1, 3),
            logits: BundleLogitsViewV1 {
                context: BundleContextHandleV1::from_parts(1, 1),
                element_type: BUNDLE_LOGITS_ELEMENT_V1_F32,
                buffer: buffer(2, u64::from(rows) * 8 * 4),
                row_stride_elements: 8,
                row_count: rows,
                live_width: 8,
                ..BundleLogitsViewV1::empty()
            },
        }
    }
    fn run_extension(
        _: BundleHandleV1,
        request: &[u8],
        sink: &mut ExecuteResultSinkV1,
    ) -> BundleStatusV1 {
        assert!(sink.device_results_enabled());
        let cycle = sink.device_cycle_metadata().unwrap();
        assert_eq!(cycle.cycle_epoch, 9);
        assert_eq!(cycle.items[0].slot_generation, 2);
        assert_eq!(cycle.termination_stop_ids, [3]);
        if request == b"capacity" {
            assert_eq!(
                sink.emit_ready_logits_view(ready(5)),
                Err(BundleAbiError::ExecuteResultCapacity)
            );
            assert_eq!(sink.emitted_rows(), 0);
            return BundleStatusV1::Success;
        }
        if request == b"ready" {
            sink.emit_ready_logits_view(ready(2)).unwrap();
            assert_eq!(sink.emitted_rows(), 2);
            return BundleStatusV1::Success;
        }
        sink.declare_input_dependency(DEVICE_RESULTS_DEVICE_FEEDBACK_V1)
            .unwrap();
        sink.emit_device_drafts(BundleDeviceDraftsV1 {
            struct_size: struct_size_v1::<BundleDeviceDraftsV1>(),
            flags: 0,
            cycle_epoch: 9,
            request_id: 11,
            slot_generation: 2,
            requested_tokens: 1,
            computed_tokens: 5,
            token_stride_bytes: 4,
            proposals: BundleReadyDeviceViewV1 {
                context: BundleContextHandleV1::from_parts(1, 1),
                rank: 0,
                reserved: 0,
                buffer: buffer(7, 20),
                ready: BundleEventHandleV1::from_parts(1, 8),
            },
            confidence: BundleDeviceBufferViewV1::empty(),
        })
        .unwrap();
        sink.sample_device_logits(BundleDeviceSamplingRequestV1 {
            struct_size: struct_size_v1::<BundleDeviceSamplingRequestV1>(),
            reserved: 0,
            logits: ready(2),
            output: BundleDeviceBufferViewV1 {
                offset: 32,
                ..buffer(4, 512)
            },
            done: BundleEventHandleV1::from_parts(1, 5),
        })
        .unwrap();
        sink.emit_device_completion(BundleDeviceCompletionV1 {
            struct_size: struct_size_v1::<BundleDeviceCompletionV1>(),
            flags: 0,
            cycle_epoch: 9,
            item_count: 1,
            sample_rows: 2,
            acceptance_offset: 0,
            samples_offset: 32,
            packet: BundleReadyDeviceViewV1 {
                context: BundleContextHandleV1::from_parts(1, 1),
                rank: 0,
                reserved: 0,
                buffer: buffer(4, 544),
                ready: BundleEventHandleV1::from_parts(1, 5),
            },
        })
        .unwrap();
        assert_eq!(
            sink.emitted_rows(),
            2,
            "completion must not charge the same rows twice"
        );
        BundleStatusV1::Success
    }
    crate::export_bundle_device_results_v1!(TEST_DEVICE_API, run_extension);

    #[test]
    fn native_device_hooks_preserve_metadata_and_charge_rows_once() {
        for (request, expected) in [
            (
                &b"cycle"[..],
                vec!["dependency", "drafts", "sample", "completion"],
            ),
            (&b"ready"[..], vec!["ready"]),
            (&b"capacity"[..], vec![]),
        ] {
            let mut sink = Sink::default();
            let execute = TEST_DEVICE_API.validate().unwrap();
            assert_eq!(
                invoke_bundle_device_results_v1(
                    execute,
                    BundleHandleV1::from_parts(1, 1),
                    request,
                    &mut sink
                )
                .unwrap(),
                BundleStatusV1::Success
            );
            assert_eq!(sink.calls, expected);
        }
    }

    #[test]
    fn legacy_sink_does_not_expose_device_callbacks() {
        let mut sink = Sink::default();
        with_result_sink_v1(&mut sink, |raw| {
            let sink = ExecuteResultSinkV1::new(raw).unwrap();
            assert!(!sink.device_results_enabled());
            assert_eq!(
                sink.device_cycle_metadata(),
                Err(BundleAbiError::Status(BundleStatusV1::Unsupported))
            );
        });
        assert!(sink.calls.is_empty());
    }

    // SAFETY-REVIEWED: the fixture forwards only a local size prefix to test short-record refusal.
    unsafe extern "C" fn short_sink(
        handle: BundleHandleV1,
        request: BundleByteViewV1,
        _: *const BundleDeviceResultSinkV1,
    ) -> BundleStatusV1 {
        #[repr(C, align(8))]
        struct Header {
            bytes: u32,
            version: u32,
        }
        let header = Header {
            bytes: 8,
            version: BUNDLE_ABI_VERSION_V1,
        };
        // SAFETY-REVIEWED: the advertised eight bytes are live; the decoder must not read a tail.
        unsafe {
            dispatch_bundle_execute_device_results_v1(
                handle,
                request,
                ptr::from_ref(&header).cast(),
                |_, _, _| panic!("short sink reached bundle"),
            )
        }
    }

    // SAFETY-REVIEWED: the input table lives for the call; the fixture copies it before removing one callback.
    unsafe extern "C" fn partial_sink(
        handle: BundleHandleV1,
        request: BundleByteViewV1,
        raw: *const BundleDeviceResultSinkV1,
    ) -> BundleStatusV1 {
        // SAFETY-REVIEWED: invoke_bundle_device_results_v1 supplied a complete live table.
        let mut raw = unsafe { raw.read() };
        raw.sample_device_logits_v1 = None;
        // SAFETY-REVIEWED: this local full table lives through the dispatch.
        unsafe {
            dispatch_bundle_execute_device_results_v1(handle, request, &raw, |_, _, _| {
                panic!("partial sink reached bundle")
            })
        }
    }

    #[test]
    fn short_and_partial_device_sinks_are_refused_before_callbacks() {
        for execute in [short_sink as BundleExecuteDeviceResultsFnV1, partial_sink] {
            let mut sink = Sink::default();
            assert_eq!(
                invoke_bundle_device_results_v1(
                    execute,
                    BundleHandleV1::from_parts(1, 1),
                    b"",
                    &mut sink
                )
                .unwrap(),
                BundleStatusV1::Unsupported
            );
            assert!(sink.calls.is_empty());
        }
        let short = crate::device_results::BundleDeviceResultsApiV1 {
            struct_size: 8,
            ..TEST_DEVICE_API
        };
        assert!(short.validate().is_err());
        let missing = crate::device_results::BundleDeviceResultsApiV1 {
            execute_device_results_v1: None,
            ..TEST_DEVICE_API
        };
        assert!(missing.validate().is_err());
    }
}
