//! The model-independent host that runs one installed bundle.
//!
//! The host owns every device and KV resource, hands the bundle only
//! generation-checked handles, and translates the engine's model transaction
//! into the C ABI in one adapter. Nothing here names a model: the manifest
//! supplies the topology, component roles, page layout, and state layout.
//!
//! Device work reaches a driver through [`HostDeviceBackendV1`]. The
//! `fake-host-services` backend performs its copies and reductions in host
//! memory and refuses a kernel launch rather than reporting work it did not do;
//! it pulls in no driver, so a macOS host and the default authoring build
//! compile no CUDA. The real driver lives behind `linux-cuda-host`, whose
//! dependency is Linux-only.
//!
//! Serving one selected bundle is here too, because a served deployment is
//! read out of the same package the host loads: [`deployment_from_package_v1`]
//! turns that package's declared components into the frontend the transport
//! serves and the tokenizer its prompts are encoded with, and
//! [`EngineGenerationBackend`] runs the engine over the loaded bundle on one
//! worker thread. The transport itself stays in `runtime-server`, which this
//! crate does not extend.

#![deny(unsafe_code)]

mod catalog;
pub mod checkpoint_verify;
mod command;
#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
pub mod cuda;
mod deployment;
mod device;
mod device_memory;
pub mod direct;
mod executor;
#[cfg(feature = "fake-host-services")]
pub mod fake;
mod frontend_check;
#[cfg(feature = "test-support")]
pub mod frontend_corpus;
mod generation;
mod grammar;
mod handles;
pub mod host_rows;
mod install;
mod kernel;
mod kv_cache_dtype;
mod media_cache;
mod media_memory;
pub mod offload;
mod overlap;
pub mod page_cache;
pub mod preparation;
mod probe;
mod profiles;
mod reference_forcing;
#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
mod sampling;
mod sampling_dispatch;
#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
mod sampling_readback;
mod sampling_records;
mod sampling_workspace;
mod services;
mod signals;
mod startup;
mod topology;
pub mod video;
mod watchdog;
pub mod weight_keeper;

pub use topology::DeviceSelectionV1;

pub use catalog::{
    HOST_GENERIC_COPY_F32_MAX_ELEMENTS_V1, HOST_GENERIC_COPY_F32_WIDTHS_V1, HostGenericCatalogV1,
    HostGenericImplementationV1, host_generic_copy_f32_widths_v1,
    host_serves_generic_copy_f32_width_v1,
};
pub use command::{
    CHECK_FRONTEND_USAGE_V1, DEFAULT_HOST_ARCHITECTURE_V1, DEFAULT_LISTEN_ADDRESS_V1,
    DEFAULT_SERVED_BATCH_TOKENS_V1, DEFAULT_SERVED_SEQUENCES_V1, DRIVER_SERVICES_V1,
    DeveloperServingEnvelopeV1, DeveloperServingPlanV1, DeviceServicesV1, DirectPackageIdentityV1,
    FixtureRequestV1, FixtureRunReceiptV1, FixtureScenarioReportV1, FixtureScenarioV1,
    FrontendVectorFilesV1, HOST_COMMAND_USAGE_V1, HOST_MEMORY_SERVICES_V1, HostCommandErrorV1,
    HostCommandV1, MAX_FIXTURE_LOGITS_PER_ROW_V1, ServingOptionsV1,
    developer_serving_engine_config_v1, drive_fixture_scenario_v1, parse_bundle_command_v1,
    parse_conformance_host_command_v1, parse_host_command_v1, resolve_developer_serving_plan_v1,
    run_command_line_v1, run_host_command_v1, run_host_command_with_surface_v1,
};
pub use deployment::{
    BundleDeploymentErrorV1, BundleDeploymentV1, configure_media_body_limit_v1,
    configure_remote_media_v1, configure_served_images_v1, declaration_from_package_v1,
    deployment_from_bundle_v1, deployment_from_package_v1, frontend_export_v1,
    frontend_from_component_v1, frontend_from_declaration_v1, media_body_bytes_v1,
};
pub use device::{
    COLLECTIVE_SUM_BF16_V1, COLLECTIVE_SUM_F32_V1, CollectiveBindingV1,
    DEVICE_POINTER_TABLE_ENTRY_BYTES_V1, DeviceArgumentV1, DeviceCustomOperandV1, DeviceObjectV1,
    DeviceRangeV1, DeviceTensorMapV1, GENERIC_COLLECTIVE_CONTRACTS_V1, GenericCollectivePlanV1,
    GenericCollectiveV1, GraphCaptureErrorV1, HeldDeviceMemoryV1, HostDeviceBackendV1,
    HostDeviceResultV1, LaunchDispatchV1, PeerAccessAcquisitionV1,
    assert_collective_ordering_contract_v1, assert_generic_collective_ordering_contract_v1,
    assert_host_transfer_contract_v1, bf16_bits_from_f32_v1, f32_from_bf16_bits_v1,
    launch_cluster_v1, launch_dispatch_v1, plan_generic_collective_v1,
};
pub use device_memory::{DeviceMemoryReportV1, RankDeviceMemoryV1, WARMUP_HEADROOM_BYTES_V1};
pub use executor::{
    BATCH_PHASE_DECODE_V1, BATCH_PHASE_PREFILL_V1, BATCH_REQUEST_MAGIC_V1, BATCH_REQUEST_MAGIC_V2,
    BATCH_WIRE_V2_CAPABILITY_V1, BatchEmissionsV1, BatchWireCapabilityV1, BundleExecutionV1,
    BundleModelExecutorV1, BundleTransactionStateV1, DeviceLogitsSamplerV1,
    POSITION_AXES_CAPABILITY_V1, TransactionOutcomeV1, batch_wire_capability_v1,
    position_axes_capability_v1,
};
pub use frontend_check::{
    ExpectedCompletionV1, ExpectedRefusalV1, ExpectedRenderV1, ExpectedToolCallV1,
    FrontendCaseOutcomeV1, FrontendCheckV1, FrontendVectorKindV1, RenderVectorV1,
    StreamingVectorV1, TokenizerVectorV1, VectorRequestV1, read_render_vectors_v1,
    read_streaming_vectors_v1, read_tokenizer_vectors_v1,
};
pub use generation::{
    CENTERED_PAD_TAG_V1, EngineBackendConfig, EngineBackendConfigError, EngineBackendStartupError,
    EngineGenerationBackend, EngineModelPolicyV1, NativeRuntimeMetricsV1, TOKEN_BUDGET_TAG_V1,
    WorkerWatchdogV1, audio_recipe_v1, centered_pad_recipe_v1, image_recipe_v1,
    token_budget_recipe_v1, validate_media_recipes_v1,
};
pub use install::{
    BundleSelectionV1, HostLoadErrorV1, InstalledBundleHostV1, ResolvedOperationV1,
    check_planned_components_v1, load_selected_process_bundle_v1, load_selected_topology_bundle_v1,
    requires_weight_plan_v1, resolve_declared_operations_for_architecture_v1,
    resolve_declared_operations_v1, served_sequences_v1, state_contract_v1,
    verified_component_by_role_v1, verify_selected_bundle_v1,
};
pub use kv_cache_dtype::{
    KvCacheChoiceV1, KvCacheDtypeV1, declared_kv_formats_v1, kv_layout_of_v1,
    resolve_kv_cache_dtype_v1,
};
pub use probe::{
    HostDeviceMemoryMeasurementV1, HostDeviceMemoryUsageV1, HostDeviceRankMemoryReportV1,
    HostDeviceRankMemoryUsageV1, HostDeviceRankMemoryV1,
};
pub use reference_forcing::{REFERENCE_TOKENS_ENV_V1, ReferenceTokensV1};
/// The request surface a served deployment admits, re-exported so a program
/// that drives this host names the one it has evidence for without depending
/// on the transport crate itself.
pub use runtime_server::RequestSurfaceV1;
pub use services::{
    BUNDLE_CONFIGURATION_MAGIC_V1, COMPILED_IMAGE_ARENA_VIEW_V1, CONTEXT_PARALLEL_TAG_V1,
    CompiledImageDescriptionV1, CompiledImageViewV1, CompiledRankImageV1, CompiledViewSpanV1,
    DeclaredKvContractV1, DeclaredStateContractV1, FILE_BACKED_COMPONENT_UPLOAD_BUFFER_BYTES_V1,
    HostConfigErrorV1, HostInstanceConfigV1, HostMemoryV1, HostServicesV1Backend, HostTopologyV1,
    KV_APPEND_OPERATION_V1, KV_RANK_POOL_VIEW_V1, ServingKvPolicyV1, kv_append_operation_v1,
    kv_page_view_id_v1,
};

pub mod recurrent;

pub use device::mapped_gather::{MappedGatherPlanV1, MappedGatherPlaneV1};
