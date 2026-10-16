#![deny(unsafe_code)]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::mem::{offset_of, size_of, size_of_val};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;
use std::sync::Mutex;

use runtime_bundle_abi as abi;
use runtime_bundle_abi::{
    BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT, BundleAbiError, BundleApiV1, CollectiveRequestV1,
    CompiledImageSpanV1, CustomAddressSpaceV1, CustomCompletionV1, CustomOperandV1,
    CustomOperationV1, DeviceBufferSpanV1, ExecuteResultSinkV1, HostServiceResultV1,
    HostServicesBackendV1, HostServicesTableV1, HostServicesV1, KernelArgumentV1, LaunchClusterV1,
    LaunchConfigV1, LentLaunchV1, LentRangeV1, LentStreamV1, LogitsRowSinkV1, OwnedLogitsRowV1,
    PhysicalKvLayoutV1, PhysicalKvRequestV1, ResultSinkLimitsV1, StateRequestV1,
    UnsupportedHostServicesBackendV1, catch_panic_callback_status_v1, catch_panic_status_v1,
    checked_u64_to_usize_v1, export_custom_operation_v1, validate_byte_view_v1,
    validate_collective_request_v1, validate_compiled_image_view_v1, validate_custom_operand_v1,
    validate_device_buffer_view_v1, validate_device_pointer_array_v1,
    validate_execute_logits_row_v1, validate_execute_result_sink_v1, validate_host_services_v1,
    validate_kernel_argument_v1, validate_launch_config_v1, validate_mut_byte_view_v1,
    validate_physical_kv_layout_v1, validate_physical_kv_request_v1, validate_state_request_v1,
};

/// The address this fixture host lends for the first operand it is given.
const FIXTURE_CUSTOM_ADDRESS: u64 = 0x0001_0000;

/// What one fixture custom operation observed, and what it answers with.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CustomOperationStateV1 {
    /// 0 answers success, 1 answers its own code, 2 raises a panic.
    mode: u32,
    calls: u32,
    address_space: Option<CustomAddressSpaceV1>,
    stream_is_null: bool,
    ranges: Vec<LentRangeV1>,
}

export_custom_operation_v1!(
    fixture_custom_operation,
    CustomOperationStateV1,
    |state: &mut CustomOperationStateV1, launch: LentLaunchV1<'_>| {
        state.calls += 1;
        state.address_space = Some(launch.address_space);
        state.stream_is_null = launch.stream == LentStreamV1::null();
        state.ranges = launch.ranges.to_vec();
        match state.mode {
            1 => 7,
            2 => panic!("fixture custom operation panic"),
            _ => 0,
        }
    },
);

const HANDLE: abi::BundleHandleV1 = abi::BundleHandleV1::from_raw(0x0001_0000_0000_0001);
const FIXTURE_LOGIT_BITS: [u8; 8] = [0, 0, 128, 63, 0, 0, 0, 64];
const FIXTURE_REQUEST_ID: u64 = 41;
const SECOND_ROW_REQUEST: &[u8] = b"second-row";
const FAILED_CREATE_REQUEST: &[u8] = b"fail";
const FAILED_EXECUTE_REQUEST: &[u8] = b"fail-execute";
const PANIC_REQUEST: &[u8] = b"panic-execute";
const EMPTY_RESULT_REQUEST: &[u8] = b"empty-result";

static CALLS: Mutex<Calls> = Mutex::new(Calls::new());
static TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn c_header_and_rust_layouts_match() {
    let c_layouts = c_layouts();

    assert_layout(
        &c_layouts,
        "byte-view",
        size_of::<abi::BundleByteViewV1>(),
        &[
            offset_of!(abi::BundleByteViewV1, data),
            offset_of!(abi::BundleByteViewV1, length),
        ],
    );
    assert_layout(
        &c_layouts,
        "mut-byte-view",
        size_of::<abi::BundleMutByteViewV1>(),
        &[
            offset_of!(abi::BundleMutByteViewV1, data),
            offset_of!(abi::BundleMutByteViewV1, length),
        ],
    );
    assert_layout(
        &c_layouts,
        "handle",
        size_of::<abi::BundleHandleV1>(),
        &[offset_of!(abi::BundleHandleV1, raw)],
    );
    assert_layout(
        &c_layouts,
        "device-buffer-view",
        size_of::<abi::BundleDeviceBufferViewV1>(),
        &[
            offset_of!(abi::BundleDeviceBufferViewV1, struct_size),
            offset_of!(abi::BundleDeviceBufferViewV1, reserved),
            offset_of!(abi::BundleDeviceBufferViewV1, buffer),
            offset_of!(abi::BundleDeviceBufferViewV1, offset),
            offset_of!(abi::BundleDeviceBufferViewV1, length),
        ],
    );
    assert_layout(
        &c_layouts,
        "compiled-image-view",
        size_of::<abi::BundleRawCompiledImageViewV1>(),
        &[
            offset_of!(abi::BundleRawCompiledImageViewV1, struct_size),
            offset_of!(abi::BundleRawCompiledImageViewV1, rank),
            offset_of!(abi::BundleRawCompiledImageViewV1, buffer),
            offset_of!(abi::BundleRawCompiledImageViewV1, alignment),
        ],
    );
    assert_layout(
        &c_layouts,
        "authenticated-component-metadata",
        size_of::<abi::BundleAuthenticatedComponentMetadataV1>(),
        &[
            offset_of!(abi::BundleAuthenticatedComponentMetadataV1, struct_size),
            offset_of!(abi::BundleAuthenticatedComponentMetadataV1, reserved),
            offset_of!(abi::BundleAuthenticatedComponentMetadataV1, role_length),
            offset_of!(abi::BundleAuthenticatedComponentMetadataV1, byte_length),
            offset_of!(abi::BundleAuthenticatedComponentMetadataV1, sha256),
        ],
    );
    assert_layout(
        &c_layouts,
        "launch-config",
        size_of::<abi::BundleLaunchConfigV1>(),
        &[
            offset_of!(abi::BundleLaunchConfigV1, struct_size),
            offset_of!(abi::BundleLaunchConfigV1, flags),
            offset_of!(abi::BundleLaunchConfigV1, grid_x),
            offset_of!(abi::BundleLaunchConfigV1, grid_y),
            offset_of!(abi::BundleLaunchConfigV1, grid_z),
            offset_of!(abi::BundleLaunchConfigV1, block_x),
            offset_of!(abi::BundleLaunchConfigV1, block_y),
            offset_of!(abi::BundleLaunchConfigV1, block_z),
            offset_of!(abi::BundleLaunchConfigV1, shared_memory_bytes),
            offset_of!(abi::BundleLaunchConfigV1, cluster_x),
            offset_of!(abi::BundleLaunchConfigV1, cluster_y),
            offset_of!(abi::BundleLaunchConfigV1, cluster_z),
        ],
    );
    // The cluster bound is one number in both languages, so a header that
    // widened it without widening this one would fail here.
    assert_values(
        &c_layouts,
        "launch-cluster-bound",
        &[abi::BUNDLE_LAUNCH_V1_MAX_CLUSTER_BLOCKS],
    );
    assert_layout(
        &c_layouts,
        "tensor-map",
        size_of::<abi::BundleTensorMapV1>(),
        &[
            offset_of!(abi::BundleTensorMapV1, struct_size),
            offset_of!(abi::BundleTensorMapV1, rank),
            offset_of!(abi::BundleTensorMapV1, element_type),
            offset_of!(abi::BundleTensorMapV1, swizzle),
            offset_of!(abi::BundleTensorMapV1, interleave),
            offset_of!(abi::BundleTensorMapV1, l2_promotion),
            offset_of!(abi::BundleTensorMapV1, oob_fill),
            offset_of!(abi::BundleTensorMapV1, reserved),
            offset_of!(abi::BundleTensorMapV1, buffer),
            offset_of!(abi::BundleTensorMapV1, global_dims),
            offset_of!(abi::BundleTensorMapV1, global_strides_bytes),
            offset_of!(abi::BundleTensorMapV1, box_dims),
            offset_of!(abi::BundleTensorMapV1, element_strides),
        ],
    );
    assert_layout(
        &c_layouts,
        "kernel-argument",
        size_of::<abi::BundleKernelArgumentV1>(),
        &[
            offset_of!(abi::BundleKernelArgumentV1, struct_size),
            offset_of!(abi::BundleKernelArgumentV1, kind),
            offset_of!(abi::BundleKernelArgumentV1, buffer),
            offset_of!(abi::BundleKernelArgumentV1, scalar_u64),
            offset_of!(abi::BundleKernelArgumentV1, scalar_u32),
            offset_of!(abi::BundleKernelArgumentV1, scalar_i32),
            offset_of!(abi::BundleKernelArgumentV1, opaque),
            offset_of!(abi::BundleKernelArgumentV1, tensor_map),
            offset_of!(abi::BundleKernelArgumentV1, pointer_array),
        ],
    );
    // The size gate is a number the header states and both languages hold the
    // record to, so a record that outgrew it fails in C as well as in Rust.
    assert_values(
        &c_layouts,
        "kernel-argument-size-gate",
        &[abi::BUNDLE_KERNEL_ARGUMENT_V1_MAX_BYTES],
    );
    assert_layout(
        &c_layouts,
        "device-pointer-array",
        size_of::<abi::BundleDevicePointerArrayV1>(),
        &[
            offset_of!(abi::BundleDevicePointerArrayV1, struct_size),
            offset_of!(abi::BundleDevicePointerArrayV1, slot_count),
            offset_of!(abi::BundleDevicePointerArrayV1, slots),
        ],
    );
    // Every argument kind and every closed tensor-map code is one value in both
    // languages, so a header that renumbered one would fail here.
    assert_values(
        &c_layouts,
        "kernel-argument-kinds",
        &[
            abi::BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_BUFFER,
            abi::BUNDLE_KERNEL_ARGUMENT_V1_U64,
            abi::BUNDLE_KERNEL_ARGUMENT_V1_OPAQUE_BYTES,
            abi::BUNDLE_KERNEL_ARGUMENT_V1_U32,
            abi::BUNDLE_KERNEL_ARGUMENT_V1_NULL_DEVICE_POINTER,
            abi::BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP,
            abi::BUNDLE_KERNEL_ARGUMENT_V1_I32,
            abi::BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY,
            abi::BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS,
        ],
    );
    assert_values(
        &c_layouts,
        "tensor-map-codes",
        &[
            abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_U8,
            abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_I32,
            abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_BF16,
            abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_F16,
            abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_F32,
            abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E4M3,
            abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E5M2,
            abi::BUNDLE_TENSOR_MAP_SWIZZLE_V1_NONE,
            abi::BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_128,
        ],
    );
    assert_layout(
        &c_layouts,
        "custom-operand",
        size_of::<abi::BundleCustomOperandV1>(),
        &[
            offset_of!(abi::BundleCustomOperandV1, struct_size),
            offset_of!(abi::BundleCustomOperandV1, flags),
            offset_of!(abi::BundleCustomOperandV1, buffer),
        ],
    );
    assert_layout(
        &c_layouts,
        "custom-launch",
        size_of::<abi::BundleCustomLaunchV1>(),
        &[
            offset_of!(abi::BundleCustomLaunchV1, struct_size),
            offset_of!(abi::BundleCustomLaunchV1, address_space),
            offset_of!(abi::BundleCustomLaunchV1, operand_count),
            offset_of!(abi::BundleCustomLaunchV1, stream),
            offset_of!(abi::BundleCustomLaunchV1, operand_addresses),
            offset_of!(abi::BundleCustomLaunchV1, operand_lengths),
        ],
    );
    // The custom-operation vocabulary is one set of numbers in both languages.
    assert_values(
        &c_layouts,
        "custom-operation-codes",
        &[
            abi::BUNDLE_CUSTOM_OPERAND_FLAG_V1_WRITABLE,
            abi::BUNDLE_CUSTOM_OPERAND_FLAGS_V1_KNOWN,
            abi::BUNDLE_CUSTOM_ADDRESS_SPACE_V1_DEVICE,
            abi::BUNDLE_CUSTOM_ADDRESS_SPACE_V1_HOST,
            abi::BUNDLE_CUSTOM_OPERATION_FLAG_V1_MAY_ENQUEUE,
            abi::BUNDLE_CUSTOM_OPERATION_FLAGS_V1_KNOWN,
        ],
    );
    assert_layout(
        &c_layouts,
        "custom-operation-contained-fault",
        BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT as u32 as usize,
        &[],
    );
    assert_layout(
        &c_layouts,
        "physical-kv-layout",
        size_of::<abi::BundlePhysicalKvLayoutV1>(),
        &[
            offset_of!(abi::BundlePhysicalKvLayoutV1, struct_size),
            offset_of!(abi::BundlePhysicalKvLayoutV1, flags),
            offset_of!(abi::BundlePhysicalKvLayoutV1, rank_count),
            offset_of!(abi::BundlePhysicalKvLayoutV1, page_bytes),
            offset_of!(abi::BundlePhysicalKvLayoutV1, page_alignment),
            offset_of!(abi::BundlePhysicalKvLayoutV1, page_tokens),
            offset_of!(abi::BundlePhysicalKvLayoutV1, state_lifetime),
            offset_of!(abi::BundlePhysicalKvLayoutV1, layout_id),
        ],
    );
    assert_layout(
        &c_layouts,
        "physical-kv-request",
        size_of::<abi::BundlePhysicalKvRequestV1>(),
        &[
            offset_of!(abi::BundlePhysicalKvRequestV1, struct_size),
            offset_of!(abi::BundlePhysicalKvRequestV1, flags),
            offset_of!(abi::BundlePhysicalKvRequestV1, layout),
            offset_of!(abi::BundlePhysicalKvRequestV1, page_capacity),
            offset_of!(abi::BundlePhysicalKvRequestV1, request_capacity),
        ],
    );
    assert_layout(
        &c_layouts,
        "state-request",
        size_of::<abi::BundleStateRequestV1>(),
        &[
            offset_of!(abi::BundleStateRequestV1, struct_size),
            offset_of!(abi::BundleStateRequestV1, flags),
            offset_of!(abi::BundleStateRequestV1, byte_length),
            offset_of!(abi::BundleStateRequestV1, alignment),
            offset_of!(abi::BundleStateRequestV1, state_id),
        ],
    );
    assert_layout(
        &c_layouts,
        "collective-request",
        size_of::<abi::BundleCollectiveRequestV1>(),
        &[
            offset_of!(abi::BundleCollectiveRequestV1, struct_size),
            offset_of!(abi::BundleCollectiveRequestV1, flags),
            offset_of!(abi::BundleCollectiveRequestV1, rank_count),
            offset_of!(abi::BundleCollectiveRequestV1, semantic_contract),
        ],
    );
    assert_layout(
        &c_layouts,
        "create-result",
        size_of::<abi::BundleCreateResultV1>(),
        &[
            offset_of!(abi::BundleCreateResultV1, struct_size),
            offset_of!(abi::BundleCreateResultV1, status),
            offset_of!(abi::BundleCreateResultV1, handle),
        ],
    );
    assert_layout(
        &c_layouts,
        "logits-row",
        size_of::<abi::BundleLogitsRowV1>(),
        &[
            offset_of!(abi::BundleLogitsRowV1, struct_size),
            offset_of!(abi::BundleLogitsRowV1, reserved),
            offset_of!(abi::BundleLogitsRowV1, request_id),
            offset_of!(abi::BundleLogitsRowV1, logit_count),
            offset_of!(abi::BundleLogitsRowV1, logit_bits),
        ],
    );
    assert_layout(
        &c_layouts,
        "logits-view",
        size_of::<abi::BundleLogitsViewV1>(),
        &[
            offset_of!(abi::BundleLogitsViewV1, struct_size),
            offset_of!(abi::BundleLogitsViewV1, reserved),
            offset_of!(abi::BundleLogitsViewV1, context),
            offset_of!(abi::BundleLogitsViewV1, rank),
            offset_of!(abi::BundleLogitsViewV1, element_type),
            offset_of!(abi::BundleLogitsViewV1, buffer),
            offset_of!(abi::BundleLogitsViewV1, row_stride_elements),
            offset_of!(abi::BundleLogitsViewV1, row_count),
            offset_of!(abi::BundleLogitsViewV1, live_width),
            offset_of!(abi::BundleLogitsViewV1, first_item_index),
            offset_of!(abi::BundleLogitsViewV1, reserved_tail),
        ],
    );
    assert_layout(
        &c_layouts,
        "execute-result-sink",
        size_of::<abi::BundleExecuteResultSinkV1>(),
        &[
            offset_of!(abi::BundleExecuteResultSinkV1, struct_size),
            offset_of!(abi::BundleExecuteResultSinkV1, reserved),
            offset_of!(abi::BundleExecuteResultSinkV1, user_data),
            offset_of!(abi::BundleExecuteResultSinkV1, maximum_rows),
            offset_of!(abi::BundleExecuteResultSinkV1, maximum_logits_per_row),
            offset_of!(abi::BundleExecuteResultSinkV1, maximum_total_logits),
            offset_of!(abi::BundleExecuteResultSinkV1, emit_logits_row),
            offset_of!(abi::BundleExecuteResultSinkV1, emit_logits_view),
        ],
    );
    assert_layout(
        &c_layouts,
        "host-services",
        size_of::<abi::BundleHostServicesV1>(),
        &host_service_offsets(),
    );
    assert_layout(
        &c_layouts,
        "api-header",
        size_of::<abi::BundleApiHeaderV1>(),
        &[
            offset_of!(abi::BundleApiHeaderV1, struct_size),
            offset_of!(abi::BundleApiHeaderV1, abi_version),
        ],
    );
    assert_layout(
        &c_layouts,
        "api",
        size_of::<abi::BundleRawApiV1>(),
        &[
            offset_of!(abi::BundleRawApiV1, struct_size),
            offset_of!(abi::BundleRawApiV1, abi_version),
            offset_of!(abi::BundleRawApiV1, create),
            offset_of!(abi::BundleRawApiV1, execute),
            offset_of!(abi::BundleRawApiV1, commit),
            offset_of!(abi::BundleRawApiV1, cancel),
            offset_of!(abi::BundleRawApiV1, destroy),
        ],
    );

    assert_layout(
        &c_layouts,
        "api-retirement",
        size_of::<abi::BundleApiWithRequestRetirementV1>(),
        &[
            offset_of!(abi::BundleApiWithRequestRetirementV1, base),
            offset_of!(abi::BundleApiWithRequestRetirementV1, retire_request_v1),
        ],
    );

    assert_layout(
        &c_layouts,
        "speculation",
        size_of::<abi::BundleRawSpeculationV1>(),
        &[
            offset_of!(abi::BundleRawSpeculationV1, struct_size),
            offset_of!(abi::BundleRawSpeculationV1, maximum_draft_tokens),
            offset_of!(abi::BundleRawSpeculationV1, maximum_verify_tokens),
        ],
    );
    assert_layout(
        &c_layouts,
        "draft-tokens",
        size_of::<abi::BundleDraftTokensV1>(),
        &[
            offset_of!(abi::BundleDraftTokensV1, struct_size),
            offset_of!(abi::BundleDraftTokensV1, reserved),
            offset_of!(abi::BundleDraftTokensV1, request_id),
            offset_of!(abi::BundleDraftTokensV1, token_count),
            offset_of!(abi::BundleDraftTokensV1, token_bytes),
        ],
    );
    assert_layout(
        &c_layouts,
        "kept-rows",
        size_of::<abi::BundleKeptRowsV1>(),
        &[
            offset_of!(abi::BundleKeptRowsV1, struct_size),
            offset_of!(abi::BundleKeptRowsV1, reserved),
            offset_of!(abi::BundleKeptRowsV1, request_id),
            offset_of!(abi::BundleKeptRowsV1, kept_rows),
            offset_of!(abi::BundleKeptRowsV1, reserved_tail),
        ],
    );
    assert_layout(
        &c_layouts,
        "api-speculation",
        size_of::<abi::BundleApiWithSpeculationV1>(),
        &[
            offset_of!(abi::BundleApiWithSpeculationV1, base),
            offset_of!(abi::BundleApiWithSpeculationV1, commit_accepted_v1),
        ],
    );
    assert_layout(
        &c_layouts,
        "prefix-state-info",
        size_of::<abi::BundlePrefixStateInfoV1>(),
        &[
            offset_of!(abi::BundlePrefixStateInfoV1, struct_size),
            offset_of!(abi::BundlePrefixStateInfoV1, status),
            offset_of!(abi::BundlePrefixStateInfoV1, state_pages),
        ],
    );
    assert_layout(
        &c_layouts,
        "prefix-state",
        size_of::<abi::BundlePrefixStateV1>(),
        &[
            offset_of!(abi::BundlePrefixStateV1, struct_size),
            offset_of!(abi::BundlePrefixStateV1, reserved),
            offset_of!(abi::BundlePrefixStateV1, request_id),
            offset_of!(abi::BundlePrefixStateV1, tokens),
            offset_of!(abi::BundlePrefixStateV1, token_ids),
            offset_of!(abi::BundlePrefixStateV1, pages),
        ],
    );
    assert_layout(
        &c_layouts,
        "api-prefix-cache",
        size_of::<abi::BundleApiWithPrefixCacheV1>(),
        &[
            offset_of!(abi::BundleApiWithPrefixCacheV1, base),
            offset_of!(abi::BundleApiWithPrefixCacheV1, describe_prefix_state_v1),
            offset_of!(abi::BundleApiWithPrefixCacheV1, capture_prefix_state_v1),
            offset_of!(abi::BundleApiWithPrefixCacheV1, restore_prefix_state_v1),
        ],
    );
    assert_layout(
        &c_layouts,
        "api-prefix-state-slot",
        size_of::<abi::BundleApiWithPrefixStateSlotV1>(),
        &[
            offset_of!(abi::BundleApiWithPrefixStateSlotV1, base),
            offset_of!(
                abi::BundleApiWithPrefixStateSlotV1,
                restore_prefix_state_in_slot_v1
            ),
        ],
    );
    assert_layout(
        &c_layouts,
        "weight-plan-sink",
        size_of::<abi::weight_plan::BundleWeightPlanSinkV1>(),
        &[
            offset_of!(abi::weight_plan::BundleWeightPlanSinkV1, struct_size),
            offset_of!(abi::weight_plan::BundleWeightPlanSinkV1, layout),
            offset_of!(abi::weight_plan::BundleWeightPlanSinkV1, context),
            offset_of!(abi::weight_plan::BundleWeightPlanSinkV1, emit),
            offset_of!(abi::weight_plan::BundleWeightPlanSinkV1, emit_error),
        ],
    );
    assert_layout(
        &c_layouts,
        "weight-transform-call",
        size_of::<abi::weight_plan::BundleWeightTransformCallV1>(),
        &[
            offset_of!(abi::weight_plan::BundleWeightTransformCallV1, struct_size),
            offset_of!(abi::weight_plan::BundleWeightTransformCallV1, transform),
            offset_of!(abi::weight_plan::BundleWeightTransformCallV1, params),
            offset_of!(abi::weight_plan::BundleWeightTransformCallV1, sources),
            offset_of!(abi::weight_plan::BundleWeightTransformCallV1, source_count),
            offset_of!(abi::weight_plan::BundleWeightTransformCallV1, output_offset),
            offset_of!(abi::weight_plan::BundleWeightTransformCallV1, output),
        ],
    );
    assert_layout(
        &c_layouts,
        "frontend-sink",
        size_of::<abi::frontend::BundleFrontendSinkV1>(),
        &[
            offset_of!(abi::frontend::BundleFrontendSinkV1, struct_size),
            offset_of!(abi::frontend::BundleFrontendSinkV1, layout),
            offset_of!(abi::frontend::BundleFrontendSinkV1, context),
            offset_of!(abi::frontend::BundleFrontendSinkV1, emit),
            offset_of!(abi::frontend::BundleFrontendSinkV1, emit_error),
        ],
    );
    assert_layout(
        &c_layouts,
        "frontend-table",
        size_of::<abi::frontend::BundleFrontendTableV1>(),
        &[
            offset_of!(abi::frontend::BundleFrontendTableV1, struct_size),
            offset_of!(abi::frontend::BundleFrontendTableV1, layout),
            offset_of!(abi::frontend::BundleFrontendTableV1, open),
            offset_of!(abi::frontend::BundleFrontendTableV1, close),
            offset_of!(abi::frontend::BundleFrontendTableV1, render),
            offset_of!(abi::frontend::BundleFrontendTableV1, constraint),
            offset_of!(abi::frontend::BundleFrontendTableV1, parser_open),
            offset_of!(abi::frontend::BundleFrontendTableV1, parser_push),
            offset_of!(abi::frontend::BundleFrontendTableV1, parser_close),
        ],
    );
    assert_layout(
        &c_layouts,
        "frontend-codes",
        abi::frontend::FRONTEND_LAYOUT_V1 as usize,
        &[
            abi::frontend::FRONTEND_PUSH_FLAG_V1_FINAL as usize,
            abi::frontend::FRONTEND_PUSH_FLAGS_V1_KNOWN as usize,
        ],
    );
    assert_layout(
        &c_layouts,
        "frontend-refusal-limit",
        abi::frontend::FRONTEND_REFUSAL_BYTES_MAX_V1,
        &[],
    );
    assert_layout(
        &c_layouts,
        "frontend-opened-limit",
        abi::frontend::FRONTEND_OPENED_BYTES_MAX_V1,
        &[],
    );
    assert_layout(
        &c_layouts,
        "weight-plan-codes",
        abi::weight_plan::WEIGHT_PLAN_LAYOUT_V1 as usize,
        &[],
    );
    assert_layout(
        &c_layouts,
        "sink-draft-tail",
        size_of::<abi::BundleExecuteResultSinkV1>(),
        &[offset_of!(
            abi::BundleExecuteResultSinkV1,
            emit_draft_tokens
        )],
    );
    assert_layout(
        &c_layouts,
        "speculation-service-tail",
        size_of::<abi::BundleHostServicesV1>(),
        &[offset_of!(abi::BundleHostServicesV1, query_speculation_v1)],
    );
    assert_layout(
        &c_layouts,
        "batch-limits",
        size_of::<abi::BundleRawBatchLimitsV1>(),
        &[
            offset_of!(abi::BundleRawBatchLimitsV1, struct_size),
            offset_of!(abi::BundleRawBatchLimitsV1, maximum_scored_rows),
        ],
    );
    assert_layout(
        &c_layouts,
        "batch-limits-service-tail",
        size_of::<abi::BundleHostServicesV1>(),
        &[offset_of!(abi::BundleHostServicesV1, query_batch_limits_v1)],
    );
    assert_layout(
        &c_layouts,
        "cluster-residency",
        size_of::<abi::BundleRawClusterResidencyV1>(),
        &[
            offset_of!(abi::BundleRawClusterResidencyV1, struct_size),
            offset_of!(abi::BundleRawClusterResidencyV1, multiprocessor_count),
            offset_of!(abi::BundleRawClusterResidencyV1, usable_multiprocessors),
            offset_of!(abi::BundleRawClusterResidencyV1, max_active_clusters),
            offset_of!(abi::BundleRawClusterResidencyV1, flags),
        ],
    );
    assert_layout(
        &c_layouts,
        "cluster-residency-service-tail",
        size_of::<abi::BundleHostServicesV1>(),
        &[offset_of!(
            abi::BundleHostServicesV1,
            query_cluster_residency_v1
        )],
    );

    // A launch flag is part of the ABI, not of one side's opinion of it, so the
    // header's value and this crate's constant have to be the same number.
    assert_layout(
        &c_layouts,
        "launch-flag-programmatic-stream-serialization",
        abi::BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION as usize,
        &[],
    );
    assert_layout(
        &c_layouts,
        "launch-flags-known",
        abi::BUNDLE_LAUNCH_FLAGS_V1_KNOWN as usize,
        &[],
    );
    assert_layout(
        &c_layouts,
        "cluster-residency-mps-client",
        abi::BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT as usize,
        &[],
    );
    assert_layout(
        &c_layouts,
        "cluster-residency-clusters-reported",
        abi::BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED as usize,
        &[],
    );
    assert_layout(
        &c_layouts,
        "cluster-residency-cooperative-launch",
        abi::BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH as usize,
        &[],
    );
}

#[test]
fn invalid_pointer_length_pairs_are_rejected() {
    let bytes = [7_u8];
    let mut mutable = [7_u8];

    assert!(
        validate_byte_view_v1(abi::BundleByteViewV1 {
            data: std::ptr::null(),
            length: 1,
        })
        .is_err()
    );
    assert!(
        validate_byte_view_v1(abi::BundleByteViewV1 {
            data: bytes.as_ptr(),
            length: 0,
        })
        .is_err()
    );
    assert!(validate_byte_view_v1(abi::BundleByteViewV1::empty()).is_ok());
    assert!(validate_byte_view_v1(byte_view(&bytes)).is_ok());
    assert!(
        validate_mut_byte_view_v1(abi::BundleMutByteViewV1 {
            data: std::ptr::null_mut(),
            length: 1,
        })
        .is_err()
    );
    assert!(validate_mut_byte_view_v1(abi::BundleMutByteViewV1::from_bytes(&mut mutable)).is_ok());

    let mut invalid_logits_row = fixture_logits_row();
    invalid_logits_row.logit_count = 3;
    assert!(validate_execute_logits_row_v1(invalid_logits_row).is_err());
    assert!(matches!(
        validate_execute_result_sink_v1(abi::BundleExecuteResultSinkV1::empty()),
        Err(BundleAbiError::MissingExecuteResultCallback)
    ));
    assert_eq!(
        checked_u64_to_usize_v1("fixture length", 3).expect("small length converts"),
        3
    );
    assert!(matches!(
        checked_u64_to_usize_v1("fixture length", u64::MAX),
        Err(BundleAbiError::LengthOutOfRange { .. })
    ));
    assert_eq!(
        catch_panic_callback_status_v1(|| -> Result<abi::BundleStatusV1, BundleAbiError> {
            panic!("callback helper panic")
        }),
        abi::BundleStatusV1::Panic
    );
}

#[test]
fn every_opaque_handle_requires_a_slot_and_generation() {
    macro_rules! assert_handle_parts {
        ($handle:ty) => {{
            let valid = <$handle>::from_parts(9, 7);
            assert_eq!(valid.slot(), 7);
            assert_eq!(valid.generation(), 9);
            assert!(valid.has_generation());
            assert!(valid.is_valid());
            assert!(valid.validate().is_ok());
            assert!(!<$handle>::from_parts(0, 7).has_generation());
            assert!(!<$handle>::from_parts(9, 0).has_generation());
            assert!(!<$handle>::from_raw(7).is_valid());
            assert!(matches!(
                <$handle>::from_parts(0, 7).validate(),
                Err(BundleAbiError::InvalidHandle)
            ));
            assert!(matches!(
                <$handle>::from_parts(9, 0).validate(),
                Err(BundleAbiError::InvalidHandle)
            ));
        }};
    }

    assert_handle_parts!(abi::BundleHandleV1);
    assert_handle_parts!(abi::BundleContextHandleV1);
    assert_handle_parts!(abi::BundleMemoryPoolHandleV1);
    assert_handle_parts!(abi::BundleDeviceBufferHandleV1);
    assert_handle_parts!(abi::BundleRankArenaHandleV1);
    assert_handle_parts!(abi::BundleCompiledImageHandleV1);
    assert_handle_parts!(abi::BundleAuthenticatedComponentHandleV1);
    assert_handle_parts!(abi::BundlePeerAccessHandleV1);
    assert_handle_parts!(abi::BundleStreamHandleV1);
    assert_handle_parts!(abi::BundleEventHandleV1);
    assert_handle_parts!(abi::BundleCudaModuleHandleV1);
    assert_handle_parts!(abi::BundleCudaSymbolHandleV1);
    assert_handle_parts!(abi::BundlePhysicalKvHandleV1);
    assert_handle_parts!(abi::BundleKvRequestHandleV1);
    assert_handle_parts!(abi::BundleKvTransactionHandleV1);
    assert_handle_parts!(abi::BundleStateHandleV1);
    assert_handle_parts!(abi::BundleCollectiveHandleV1);
}

#[test]
fn nested_callback_records_validate_before_safe_access() {
    let id = [1_u8];
    let mut device_view = valid_device_buffer_view();
    assert!(validate_device_buffer_view_v1(device_view).is_ok());
    // The length is exact, not a range: this record is embedded by value in
    // five others, whose later fields move with it.
    let mut short_device_view = device_view;
    short_device_view.struct_size = 0;
    assert!(matches!(
        validate_device_buffer_view_v1(short_device_view),
        Err(BundleAbiError::StructSizeExact {
            surface: "device buffer view",
            ..
        })
    ));
    assert!(matches!(
        validate_device_buffer_view_v1(abi::BundleDeviceBufferViewV1::empty()),
        Err(BundleAbiError::InvalidHandle)
    ));
    device_view.reserved = 1;
    assert!(matches!(
        validate_device_buffer_view_v1(device_view),
        Err(BundleAbiError::InvalidReserved { .. })
    ));
    device_view.reserved = 0;
    device_view.offset = u64::MAX;
    assert!(matches!(
        validate_device_buffer_view_v1(device_view),
        Err(BundleAbiError::InvalidRange("device buffer view"))
    ));

    let mut compiled_view = abi::BundleRawCompiledImageViewV1::empty();
    compiled_view.buffer = valid_device_buffer_view();
    compiled_view.alignment = 64;
    assert!(validate_compiled_image_view_v1(compiled_view).is_ok());
    compiled_view.alignment = 3;
    assert!(matches!(
        validate_compiled_image_view_v1(compiled_view),
        Err(BundleAbiError::InvalidAlignment { .. })
    ));

    let mut launch = valid_launch_config();
    assert!(validate_launch_config_v1(launch).is_ok());
    launch.grid_x = 0;
    assert!(matches!(
        validate_launch_config_v1(launch),
        Err(BundleAbiError::InvalidRange("launch config grid_x"))
    ));

    // A flag no version defines cannot be honoured, so it is refused rather
    // than launched without whatever it asked for.
    let mut flagged = valid_launch_config();
    flagged.flags = abi::BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION;
    assert!(validate_launch_config_v1(flagged).is_ok());
    for unknown in [
        abi::BUNDLE_LAUNCH_FLAGS_V1_KNOWN + 1,
        abi::BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION | 0x8000_0000,
        u32::MAX,
    ] {
        let mut undefined = valid_launch_config();
        undefined.flags = unknown;
        assert!(
            matches!(
                validate_launch_config_v1(undefined),
                Err(BundleAbiError::InvalidLaunchFlags(actual)) if actual == unknown
            ),
            "launch flags {unknown:#010x} must be refused"
        );
    }

    let mut opaque_argument = abi::BundleKernelArgumentV1::empty();
    opaque_argument.kind = abi::BUNDLE_KERNEL_ARGUMENT_V1_OPAQUE_BYTES;
    opaque_argument.opaque = byte_view(&id);
    assert!(validate_kernel_argument_v1(opaque_argument).is_ok());
    opaque_argument.kind = u32::MAX;
    assert!(matches!(
        validate_kernel_argument_v1(opaque_argument),
        Err(BundleAbiError::InvalidKernelArgumentKind(u32::MAX))
    ));

    // A scalar argument carries no payload at all, so its kind is the whole of
    // what decides whether it is well formed.
    for kind in [
        abi::BUNDLE_KERNEL_ARGUMENT_V1_U32,
        abi::BUNDLE_KERNEL_ARGUMENT_V1_I32,
        abi::BUNDLE_KERNEL_ARGUMENT_V1_U64,
        abi::BUNDLE_KERNEL_ARGUMENT_V1_NULL_DEVICE_POINTER,
    ] {
        let mut scalar = abi::BundleKernelArgumentV1::empty();
        scalar.kind = kind;
        scalar.scalar_u32 = 7;
        scalar.scalar_i32 = i32::MIN;
        scalar.scalar_u64 = u64::MAX;
        assert!(validate_kernel_argument_v1(scalar).is_ok(), "{kind}");
    }

    let map = valid_tensor_map();
    let mut map_argument = abi::BundleKernelArgumentV1::empty();
    map_argument.kind = abi::BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP;
    map_argument.tensor_map = &raw const map;
    assert!(validate_kernel_argument_v1(map_argument).is_ok());

    // A by-value address array is bounded by its own slot count, and the slots
    // inside that count are the only ones that decide anything.
    let array = valid_device_pointer_array();
    let mut array_argument = abi::BundleKernelArgumentV1::empty();
    array_argument.kind = abi::BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY;
    array_argument.pointer_array = &raw const array;
    assert!(validate_kernel_argument_v1(array_argument).is_ok());
    assert!(validate_device_pointer_array_v1(array).is_ok());

    for slot_count in [
        0,
        abi::BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS + 1,
        u32::MAX,
    ] {
        let mut out_of_range = valid_device_pointer_array();
        out_of_range.slot_count = slot_count;
        assert!(
            matches!(
                validate_device_pointer_array_v1(out_of_range),
                Err(BundleAbiError::InvalidDevicePointerArraySlots(actual))
                    if actual == slot_count
            ),
            "{slot_count} slots must be refused"
        );
    }

    let mut wrong_size = valid_device_pointer_array();
    wrong_size.struct_size = 0;
    assert!(matches!(
        validate_device_pointer_array_v1(wrong_size),
        Err(BundleAbiError::StructSize { .. })
    ));

    // A null slot is the null pointer and carries nothing else. One that also
    // names an offset or a length is a slot filled in wrongly, not an unused
    // one, and it is refused rather than launched as null.
    for poison in 0..2 {
        let mut lying_null = valid_device_pointer_array();
        if poison == 0 {
            lying_null.slots[3].offset = 8;
        } else {
            lying_null.slots[3].length = 8;
        }
        assert!(
            matches!(
                validate_device_pointer_array_v1(lying_null),
                Err(BundleAbiError::InvalidDevicePointerArraySlots(_))
            ),
            "a null slot that carries {poison} must be refused"
        );
    }
    let mut reserved_null = valid_device_pointer_array();
    reserved_null.slots[3].reserved = 1;
    assert!(matches!(
        validate_device_pointer_array_v1(reserved_null),
        Err(BundleAbiError::InvalidReserved { .. })
    ));

    // A named slot inside the count is checked exactly as a device-buffer view
    // is, so an overflowing range there is refused too.
    let mut overflowing_slot = valid_device_pointer_array();
    overflowing_slot.slots[0].offset = u64::MAX;
    overflowing_slot.slots[0].length = 1;
    assert!(matches!(
        validate_device_pointer_array_v1(overflowing_slot),
        Err(BundleAbiError::InvalidRange("device buffer view"))
    ));

    // Slots past the count are never read, so anything at all may sit there.
    let mut noisy_tail = valid_device_pointer_array();
    noisy_tail.slot_count = 2;
    for slot in &mut noisy_tail.slots[2..] {
        slot.struct_size = u32::MAX;
        slot.reserved = u32::MAX;
        slot.offset = u64::MAX;
        slot.length = u64::MAX;
    }
    assert!(validate_device_pointer_array_v1(noisy_tail).is_ok());

    let layout = valid_physical_kv_layout(&id);
    assert!(validate_physical_kv_layout_v1(layout).is_ok());
    let physical_request = valid_physical_kv_request(layout);
    assert!(validate_physical_kv_request_v1(physical_request).is_ok());
    let mut overflowing_physical_request = physical_request;
    overflowing_physical_request.page_capacity = u64::MAX;
    assert!(matches!(
        validate_physical_kv_request_v1(overflowing_physical_request),
        Err(BundleAbiError::InvalidRange("physical KV request"))
    ));

    let state_request = valid_state_request(&id);
    assert!(validate_state_request_v1(state_request).is_ok());
    let mut invalid_state_request = state_request;
    invalid_state_request.alignment = 3;
    assert!(matches!(
        validate_state_request_v1(invalid_state_request),
        Err(BundleAbiError::InvalidAlignment { .. })
    ));

    let collective_request = valid_collective_request(&id);
    assert!(validate_collective_request_v1(collective_request).is_ok());
    let mut invalid_collective_request = collective_request;
    invalid_collective_request.rank_count = 0;
    assert!(matches!(
        validate_collective_request_v1(invalid_collective_request),
        Err(BundleAbiError::InvalidRange("collective request"))
    ));

    let mut row = fixture_logits_row();
    row.reserved = 1;
    assert!(matches!(
        validate_execute_logits_row_v1(row),
        Err(BundleAbiError::InvalidReserved { .. })
    ));
    let mut result_sink = abi::BundleExecuteResultSinkV1::empty();
    result_sink.reserved = 1;
    assert!(matches!(
        validate_execute_result_sink_v1(result_sink),
        Err(BundleAbiError::InvalidReserved { .. })
    ));
}

#[test]
fn callbacks_contain_panics_before_the_c_boundary() {
    let status =
        catch_panic_status_v1(|| -> abi::BundleStatusV1 { panic!("fixture callback panic") });

    assert_eq!(status, abi::BundleStatusV1::Panic);
}

#[test]
fn device_logits_view_checks_live_width_stride_and_element_size() {
    let mut view = abi::BundleLogitsViewV1::empty();
    view.context = abi::BundleContextHandleV1::from_raw(0x1_0000_0001);
    view.buffer.buffer = abi::BundleDeviceBufferHandleV1::from_raw(0x1_0000_0002);
    view.buffer.length = 44;
    view.element_type = abi::BUNDLE_LOGITS_ELEMENT_V1_F32;
    view.row_stride_elements = 8;
    view.row_count = 2;
    view.live_width = 3;
    assert!(abi::validate_execute_logits_view_v1(view).is_ok());

    view.buffer.length = 43;
    assert_eq!(
        abi::validate_execute_logits_view_v1(view),
        Err(BundleAbiError::InvalidLogitsView)
    );
    view.buffer.length = 44;
    view.element_type = abi::BUNDLE_LOGITS_ELEMENT_V1_BF16;
    assert!(abi::validate_execute_logits_view_v1(view).is_ok());
    view.row_stride_elements = 2;
    assert_eq!(
        abi::validate_execute_logits_view_v1(view),
        Err(BundleAbiError::InvalidLogitsView)
    );
    view.row_stride_elements = 8;
    view.element_type = 0;
    assert_eq!(
        abi::validate_execute_logits_view_v1(view),
        Err(BundleAbiError::InvalidLogitsView)
    );
}

#[test]
fn host_backend_receives_only_owned_or_scoped_rust_values() {
    let backend = Rc::new(FixtureHostBackend::default());
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let stream = abi::BundleStreamHandleV1::from_parts(1, 1);
    assert_eq!(table.backend_reference_count(), 2);

    services.log(b"stage-0").expect("log reaches the backend");
    assert_eq!(backend.messages.borrow().as_slice(), [b"stage-0".to_vec()]);

    let context = services
        .select_context(2, 3)
        .expect("context selection returns a checked handle");
    assert_eq!(context, abi::BundleContextHandleV1::from_parts(7, 3));
    assert_eq!(backend.contexts.borrow().as_slice(), [(2, 3)]);

    services
        .copy_host_to_device(stream, fixture_device_span(), &[4_u8, 5, 6])
        .expect("immutable input reaches the backend as a slice");
    assert_eq!(
        backend.host_to_device.borrow().as_slice(),
        [(fixture_device_span(), vec![4_u8, 5, 6])]
    );

    let mut destination = [0_u8; 3];
    services
        .copy_device_to_host(stream, &mut destination, fixture_device_span())
        .expect("mutable output reaches the backend as a slice");
    assert_eq!(destination, [0xAB, 0xAB, 0xAB]);
    assert_eq!(
        backend.device_to_host.borrow().as_slice(),
        [fixture_device_span()]
    );

    let arguments = [
        KernelArgumentV1::OpaqueBytes(vec![9_u8, 8]),
        KernelArgumentV1::DeviceBuffer(fixture_device_span()),
        KernelArgumentV1::ScalarU64(11),
    ];
    services
        .launch_cuda_symbol(
            stream,
            abi::BundleCudaSymbolHandleV1::from_parts(1, 1),
            fixture_launch_config(),
            &arguments,
        )
        .expect("a pointer-array launch reaches the backend as owned arguments");
    let launches = backend.launches.borrow();
    let (config, launched_arguments) = launches.first().expect("one launch is recorded");
    assert_eq!(config.grid_x, 1);
    assert_eq!(launched_arguments.as_slice(), arguments);
    drop(launches);

    services
        .launch_cuda_symbol(
            stream,
            abi::BundleCudaSymbolHandleV1::from_parts(1, 1),
            fixture_launch_config(),
            &[],
        )
        .expect("an empty argument array is a null pointer with a zero count");
    assert_eq!(backend.launches.borrow().len(), 2);

    // A defined flag survives the crossing byte for byte, because a host that
    // received a launch without it would apply a different launch attribute.
    let mut programmatic = fixture_launch_config();
    programmatic.flags = abi::BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION;
    services
        .launch_cuda_symbol(
            stream,
            abi::BundleCudaSymbolHandleV1::from_parts(1, 1),
            programmatic,
            &[],
        )
        .expect("a defined launch flag crosses the ABI");
    assert_eq!(
        backend.launches.borrow()[2].0.flags,
        abi::BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION
    );

    // An undefined one never reaches the backend at all.
    let mut undefined = fixture_launch_config();
    undefined.flags = abi::BUNDLE_LAUNCH_FLAGS_V1_KNOWN + 1;
    assert!(matches!(
        services.launch_cuda_symbol(
            stream,
            abi::BundleCudaSymbolHandleV1::from_parts(1, 1),
            undefined,
            &[],
        ),
        Err(BundleAbiError::InvalidLaunchFlags(_))
    ));
    assert_eq!(backend.launches.borrow().len(), 3);

    let kv = services
        .allocate_physical_kv(&fixture_physical_kv_request())
        .expect("an owned KV request reaches the backend");
    assert_eq!(kv, abi::BundlePhysicalKvHandleV1::from_parts(5, 1));
    assert_eq!(
        backend.physical_kv.borrow()[0].layout.layout_id.as_slice(),
        [1_u8]
    );

    services
        .create_collective(&fixture_collective_request())
        .expect("an owned collective request reaches the backend");
    assert_eq!(
        backend.collectives.borrow()[0].semantic_contract.as_slice(),
        [1_u8]
    );

    let span = services
        .resolve_kv_view(abi::BundleKvRequestHandleV1::from_parts(1, 1), 0, &[1])
        .expect("a returned span is checked before the caller sees it");
    assert_eq!(span, fixture_device_span());
    assert_eq!(backend.kv_views.borrow().as_slice(), [vec![1_u8]]);
}

/// A second collective contract is bytes across V1, not a second V1.
///
/// The request names its contract with a byte view, so a host and a bundle can
/// agree on a new one without either of them changing a record, a function
/// pointer, or a size. This drives both of the contracts a host answers to
/// through the same unchanged create call, including the longer of the two
/// names, and checks that what arrives is byte for byte what was sent. The ABI
/// itself knows neither name: it carries whatever it is handed.
#[test]
fn a_second_collective_contract_crosses_v1_as_bytes_rather_than_as_a_change() {
    const SUM_F32_V1: &[u8] = b"sum-f32-v1";
    const SUM_BF16_V1: &[u8] = b"sum-bf16-v1";
    const RANKS: u64 = 4;

    let backend = Rc::new(FixtureHostBackend::default());
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();

    for (index, contract) in [SUM_F32_V1, SUM_BF16_V1].into_iter().enumerate() {
        let mut request = abi::BundleCollectiveRequestV1::empty();
        request.rank_count = RANKS;
        request.semantic_contract = byte_view(contract);
        assert!(
            validate_collective_request_v1(request).is_ok(),
            "a contract of {} bytes crosses the record V1 froze",
            contract.len()
        );
        assert_eq!(
            usize::try_from(request.struct_size).expect("the struct size fits usize"),
            size_of::<abi::BundleCollectiveRequestV1>(),
            "the record is the size V1 declared, whatever the contract names"
        );

        services
            .create_collective(&CollectiveRequestV1 {
                flags: 0,
                rank_count: RANKS,
                semantic_contract: contract.to_vec(),
            })
            .expect("an owned collective request reaches the backend");
        let collectives = backend.collectives.borrow();
        assert_eq!(collectives[index].semantic_contract.as_slice(), contract);
        assert_eq!(collectives[index].rank_count, RANKS);
    }
    assert_eq!(backend.collectives.borrow().len(), 2);
}

#[test]
fn safe_service_calls_check_arguments_results_and_backend_failures() {
    let backend = Rc::new(FixtureHostBackend::default());
    let table = HostServicesTableV1::new(backend);
    let services = table.caller();
    let stream = abi::BundleStreamHandleV1::from_parts(1, 1);

    assert!(matches!(
        services.release_stream(abi::BundleStreamHandleV1::from_raw(0)),
        Err(BundleAbiError::Status(abi::BundleStatusV1::StaleHandle)),
    ));
    assert!(
        matches!(
            services.synchronize_event(abi::BundleEventHandleV1::from_parts(1, 1)),
            Err(BundleAbiError::Status(abi::BundleStatusV1::Panic)),
        ),
        "a backend panic never crosses the C boundary"
    );

    let mut invalid_launch = fixture_launch_config();
    invalid_launch.grid_x = 0;
    assert!(matches!(
        services.launch_cuda_symbol(
            stream,
            abi::BundleCudaSymbolHandleV1::from_parts(1, 1),
            invalid_launch,
            &[],
        ),
        Err(BundleAbiError::InvalidRange("launch config grid_x"))
    ));

    let mut invalid_request = fixture_physical_kv_request();
    invalid_request.layout.page_alignment = 3;
    assert!(matches!(
        services.allocate_physical_kv(&invalid_request),
        Err(BundleAbiError::InvalidAlignment { .. })
    ));

    assert!(matches!(
        services.copy_host_to_device(
            stream,
            DeviceBufferSpanV1::new(abi::BundleDeviceBufferHandleV1::from_raw(0), 0, 8),
            &[1_u8],
        ),
        Err(BundleAbiError::InvalidHandle)
    ));

    assert!(
        matches!(
            services.select_memory_pool(abi::BundleContextHandleV1::from_parts(1, 1)),
            Err(BundleAbiError::InvalidHandle)
        ),
        "a handle the host returns without a generation is rejected"
    );
    assert!(
        matches!(
            services.resolve_state_view(abi::BundleStateHandleV1::from_parts(1, 1), 0),
            Err(BundleAbiError::InvalidHandle)
        ),
        "a span the host returns without a buffer handle is rejected"
    );
}

#[test]
fn every_service_reaches_its_own_backend_method_through_the_safe_caller() {
    let backend = Rc::new(RecordingHostBackend::default());
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let context = abi::BundleContextHandleV1::from_parts(1, 1);
    let pool = abi::BundleMemoryPoolHandleV1::from_parts(1, 1);
    let stream = abi::BundleStreamHandleV1::from_parts(1, 1);
    let image = abi::BundleCompiledImageHandleV1::from_parts(1, 1);
    let authenticated = abi::BundleAuthenticatedComponentHandleV1::from_parts(1, 1);
    let arena = abi::BundleRankArenaHandleV1::from_parts(1, 1);
    let event = abi::BundleEventHandleV1::from_parts(1, 1);
    let module = abi::BundleCudaModuleHandleV1::from_parts(1, 1);
    let symbol = abi::BundleCudaSymbolHandleV1::from_parts(1, 1);
    let kv = abi::BundlePhysicalKvHandleV1::from_parts(1, 1);
    let kv_request = abi::BundleKvRequestHandleV1::from_parts(1, 1);
    let transaction = abi::BundleKvTransactionHandleV1::from_parts(1, 1);
    let state = abi::BundleStateHandleV1::from_parts(1, 1);
    let collective = abi::BundleCollectiveHandleV1::from_parts(1, 1);
    let span = fixture_device_span();
    let mut destination = [0_u8; 4];
    let mut component_role = [0_u8; 1];
    let mut component_bytes = [0_u8; 3];

    let unsupported: Vec<Result<(), BundleAbiError>> = vec![
        services.log(b"x"),
        services.select_context(0, 0).map(drop),
        services.release_context(context),
        services.select_memory_pool(context).map(drop),
        services.set_memory_pool_peer_access(pool, context, true),
        services.release_memory_pool(pool),
        services
            .allocate_device_buffer(pool, stream, 64, 64)
            .map(drop),
        services.release_device_buffer(span.buffer),
        services.open_compiled_image(b"role").map(drop),
        services
            .upload_rank_arena(context, pool, stream, image, 0)
            .map(drop),
        services.release_rank_arena(arena),
        services
            .resolve_compiled_image_view(image, arena, 0, b"view")
            .map(drop),
        services.release_compiled_image(image),
        services.open_authenticated_component(b"role").map(drop),
        services
            .inspect_authenticated_component(authenticated)
            .map(drop),
        services.read_authenticated_component(
            authenticated,
            &mut component_role,
            &mut component_bytes,
        ),
        services.release_authenticated_component(authenticated),
        services.acquire_peer_access(context, context).map(drop),
        services.release_peer_access(abi::BundlePeerAccessHandleV1::from_parts(1, 1)),
        services.create_stream(context, 0).map(drop),
        services
            .select_stage_stream(context, 0, b"stage", 0)
            .map(drop),
        services.synchronize_stream(stream),
        services.release_stream(stream),
        services.create_event(context).map(drop),
        services.record_event(event, stream),
        services.wait_event(stream, event),
        services.synchronize_event(event),
        services.release_event(event),
        services.copy_host_to_device(stream, span, b"bytes"),
        services.copy_device_to_host(stream, &mut destination, span),
        services.copy_device_to_device(stream, span, span),
        services.copy_peer_to_peer(stream, context, span, context, span),
        services.zero_device_buffer(stream, span),
        services.write_device_pointer_table(stream, span, &[Some(span)]),
        services.load_cuda_module(context, b"module").map(drop),
        services.resolve_cuda_symbol(module, b"symbol").map(drop),
        services.configure_cuda_symbol(symbol, 0),
        services.launch_cuda_symbol(stream, symbol, fixture_launch_config(), &[]),
        services.release_cuda_symbol(symbol),
        services.release_cuda_module(module),
        services
            .allocate_physical_kv(&fixture_physical_kv_request())
            .map(drop),
        services.create_kv_request(kv, 1).map(drop),
        services.prepare_kv(kv_request, b"append").map(drop),
        services.commit_kv(transaction),
        services.cancel_kv(transaction),
        services.resolve_kv_view(kv_request, 0, b"view").map(drop),
        services.release_kv_request(kv_request),
        services.release_physical_kv(kv),
        services.acquire_serving_kv(b"layout").map(drop),
        services.resolve_serving_kv_pool(0).map(drop),
        services.allocate_state(&fixture_state_request()).map(drop),
        services.resolve_state_view(state, 0).map(drop),
        services.release_state(state),
        services
            .create_collective(&fixture_collective_request())
            .map(drop),
        services.execute_collective(collective, stream, b"reduce", &[]),
        services.release_collective(collective),
        services.run_custom_operation(
            context,
            stream,
            CustomCompletionV1::Synchronous,
            b"custom",
            &[CustomOperandV1::read(span)],
            &fixture_custom_operation(&mut CustomOperationStateV1::default()),
        ),
    ];

    for result in &unsupported {
        assert!(matches!(
            result,
            Err(BundleAbiError::Status(abi::BundleStatusV1::Unsupported))
        ));
    }
    assert_eq!(unsupported.len(), APPROVED_SERVICES.len());
    assert_eq!(backend.calls.borrow().as_slice(), APPROVED_SERVICES);
}

#[test]
fn a_custom_operation_is_lent_its_operands_for_exactly_one_call() {
    let backend = Rc::new(FixtureHostBackend::default());
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let context = abi::BundleContextHandleV1::from_parts(1, 1);
    let stream = abi::BundleStreamHandleV1::from_parts(1, 1);
    let read = fixture_device_span();
    let written =
        DeviceBufferSpanV1::new(abi::BundleDeviceBufferHandleV1::from_parts(1, 9), 64, 32);
    let mut state = CustomOperationStateV1::default();

    {
        let operation = fixture_custom_operation(&mut state);
        services
            .run_custom_operation(
                context,
                stream,
                CustomCompletionV1::MayEnqueue,
                b"fixture-operation-v1",
                &[CustomOperandV1::read(read), CustomOperandV1::write(written)],
                &operation,
            )
            .expect("the fixture host runs the operation it was handed");
    }

    // The operation ran once, and what it was lent is what the host resolved:
    // one address and length per operand, in the order they were named.
    assert_eq!(state.calls, 1);
    assert_eq!(state.address_space, Some(CustomAddressSpaceV1::Host));
    assert!(
        state.stream_is_null,
        "a host with no driver lends no stream"
    );
    assert_eq!(
        state.ranges,
        vec![
            LentRangeV1::new(FIXTURE_CUSTOM_ADDRESS + 0x1000, read.length),
            LentRangeV1::new(FIXTURE_CUSTOM_ADDRESS + 0x2000, written.length),
        ]
    );
    let calls = backend.custom_operations.borrow();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].identifier, b"fixture-operation-v1");
    assert_eq!(
        calls[0].completion,
        CustomCompletionV1::MayEnqueue,
        "the completion policy the caller asked for reaches the host"
    );
    assert_eq!(
        calls[0].operands,
        vec![CustomOperandV1::read(read), CustomOperandV1::write(written)],
        "the host is told which operand the operation writes"
    );
}

#[test]
fn a_refused_custom_operation_never_reaches_the_callback() {
    let backend = Rc::new(FixtureHostBackend::default());
    *backend.custom_refusal.borrow_mut() = Some(abi::BundleStatusV1::InvalidArgument);
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let mut state = CustomOperationStateV1::default();

    let refused = {
        let operation = fixture_custom_operation(&mut state);
        services.run_custom_operation(
            abi::BundleContextHandleV1::from_parts(1, 1),
            abi::BundleStreamHandleV1::from_parts(1, 1),
            CustomCompletionV1::MayEnqueue,
            b"fixture-operation-v1",
            &[CustomOperandV1::read(fixture_device_span())],
            &operation,
        )
    };

    assert!(matches!(
        refused,
        Err(BundleAbiError::Status(abi::BundleStatusV1::InvalidArgument))
    ));
    assert_eq!(state.calls, 0, "a refusal invokes nothing");
    assert!(backend.custom_operations.borrow().is_empty());
}

#[test]
fn a_custom_operation_reports_its_own_code_and_contains_its_own_panic() {
    let backend = Rc::new(FixtureHostBackend::default());
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let context = abi::BundleContextHandleV1::from_parts(1, 1);
    let stream = abi::BundleStreamHandleV1::from_parts(1, 1);

    let mut failing = CustomOperationStateV1 {
        mode: 1,
        ..CustomOperationStateV1::default()
    };
    let failed = {
        let operation = fixture_custom_operation(&mut failing);
        services.run_custom_operation(
            context,
            stream,
            CustomCompletionV1::Synchronous,
            b"fixture-operation-v1",
            &[],
            &operation,
        )
    };
    assert_eq!(failed, Err(BundleAbiError::CustomOperation(7)));
    assert_eq!(failing.calls, 1);

    // A panic inside the operation is contained by its own entry and reported
    // as the one code V1 reserves, rather than unwinding into the host.
    let mut panicking = CustomOperationStateV1 {
        mode: 2,
        ..CustomOperationStateV1::default()
    };
    let panicked = {
        let operation = fixture_custom_operation(&mut panicking);
        services.run_custom_operation(
            context,
            stream,
            CustomCompletionV1::Synchronous,
            b"fixture-operation-v1",
            &[],
            &operation,
        )
    };
    assert_eq!(
        panicked,
        Err(BundleAbiError::CustomOperation(
            BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT
        ))
    );
    assert_eq!(panicking.calls, 1);
}

#[test]
fn a_custom_operand_is_checked_before_any_host_resolves_it() {
    let valid = abi::BundleCustomOperandV1 {
        struct_size: u32::try_from(size_of::<abi::BundleCustomOperandV1>())
            .expect("custom operand size fits u32"),
        flags: abi::BUNDLE_CUSTOM_OPERAND_FLAG_V1_WRITABLE,
        buffer: valid_device_buffer_view(),
    };
    assert!(validate_custom_operand_v1(valid).is_ok());

    // The length is exact, not a range: this record is an element of the array
    // run_custom_operation takes, and that callback carries no element stride.
    let mut short = valid;
    short.struct_size = 0;
    assert!(matches!(
        validate_custom_operand_v1(short),
        Err(BundleAbiError::StructSizeExact {
            surface: "custom operand",
            ..
        })
    ));

    // A flag no version defines asks for something no host can honour.
    let mut unknown_flag = valid;
    unknown_flag.flags = abi::BUNDLE_CUSTOM_OPERAND_FLAGS_V1_KNOWN | 0x8000_0000;
    assert!(matches!(
        validate_custom_operand_v1(unknown_flag),
        Err(BundleAbiError::InvalidCustomOperandFlags(_))
    ));

    let mut stale = valid;
    stale.buffer = abi::BundleDeviceBufferViewV1::empty();
    assert!(matches!(
        validate_custom_operand_v1(stale),
        Err(BundleAbiError::InvalidHandle)
    ));

    let mut overflowing = valid;
    overflowing.buffer.offset = u64::MAX;
    assert!(matches!(
        validate_custom_operand_v1(overflowing),
        Err(BundleAbiError::InvalidRange("device buffer view"))
    ));
}

#[test]
fn a_caller_the_bundle_kept_stays_defined_after_destroy() {
    let _guard = TEST_LOCK.lock().expect("ABI test lock is available");
    reset_calls();
    let backend = Rc::new(FixtureHostBackend::default());
    let mut instance = fixture_api()
        .create(fixture_table_for(&backend), b"retain")
        .expect("fixture instance creates");
    instance.destroy().expect("fixture instance destroys");
    drop(instance);

    assert_eq!(
        Rc::strong_count(&backend),
        2,
        "a destroyed instance releases its resources, not the service table the bundle kept"
    );

    call_retained_caller(b"after-destroy")
        .expect("the retained caller still reaches its live backend");

    assert!(
        backend
            .messages
            .borrow()
            .contains(&b"after-destroy".to_vec()),
        "the call reached the retained backend instead of freed storage"
    );
}

#[test]
fn a_bundle_that_kept_its_caller_and_failed_create_still_calls_a_live_backend() {
    let _guard = TEST_LOCK.lock().expect("ABI test lock is available");
    reset_calls();
    let backend = Rc::new(FixtureHostBackend::default());

    assert!(matches!(
        fixture_api().create(fixture_table_for(&backend), FAILED_CREATE_REQUEST),
        Err(BundleAbiError::Status(abi::BundleStatusV1::BundleError))
    ));

    assert_eq!(
        Rc::strong_count(&backend),
        2,
        "the create callback ran, so the table outlives the status the bundle chose to return"
    );

    call_retained_caller(b"after-failed-create")
        .expect("the caller the failed create kept still reaches its live backend");

    assert!(
        backend
            .messages
            .borrow()
            .contains(&b"after-failed-create".to_vec()),
        "the call reached the retained backend instead of freed storage"
    );
}

#[test]
fn a_rejected_create_result_still_retains_the_service_table() {
    let _guard = TEST_LOCK.lock().expect("ABI test lock is available");
    reset_calls();
    let mut short_result = fixture_raw_api();
    short_result.create = Some(create_with_short_result);
    let api = BundleApiV1::new(short_result).expect("fixture API table is valid");
    let backend = Rc::new(FixtureHostBackend::default());

    assert!(matches!(
        api.create(fixture_table_for(&backend), b"short"),
        Err(BundleAbiError::StructSize {
            surface: "create result",
            ..
        })
    ));

    assert_eq!(
        Rc::strong_count(&backend),
        2,
        "a bundle that reported success may hold the caller even when this wrapper rejects it"
    );
}

#[test]
fn a_generated_bundle_calls_host_services_without_unsafe_code() {
    let _guard = TEST_LOCK.lock().expect("ABI test lock is available");
    reset_calls();
    let backend = Rc::new(FixtureHostBackend::default());
    let mut instance = fixture_api()
        .create(fixture_table_for(&backend), b"config")
        .expect("fixture instance creates");

    assert_eq!(
        backend.messages.borrow().as_slice(),
        [b"bundle-create".to_vec()],
        "the bundle logged through the safe caller its create trampoline handed it"
    );
    assert_eq!(backend.contexts.borrow().as_slice(), [(0, 1)]);

    instance.destroy().expect("fixture instance destroys");
}

#[test]
fn v1_tables_and_create_results_reject_version_or_size_drift() {
    let _guard = TEST_LOCK.lock().expect("ABI test lock is available");
    reset_calls();

    let mut wrong_api_version = fixture_raw_api();
    wrong_api_version.abi_version = abi::BUNDLE_ABI_VERSION_V1 + 1;
    assert!(matches!(
        BundleApiV1::new(wrong_api_version),
        Err(BundleAbiError::AbiVersion {
            surface: "bundle API",
            ..
        })
    ));

    let mut short_api = fixture_raw_api();
    short_api.struct_size -= 1;
    assert!(matches!(
        BundleApiV1::new(short_api),
        Err(BundleAbiError::StructSize {
            surface: "bundle API",
            ..
        })
    ));

    let mut wrong_host_version = *fixture_table().services();
    wrong_host_version.abi_version = abi::BUNDLE_ABI_VERSION_V1 + 1;
    assert!(matches!(
        validate_host_services_v1(&wrong_host_version),
        Err(BundleAbiError::AbiVersion {
            surface: "host services",
            ..
        })
    ));

    let mut short_host = *fixture_table().services();
    short_host.struct_size = offset_of!(abi::BundleHostServicesV1, acquire_serving_kv) as u32 - 1;
    assert!(matches!(
        validate_host_services_v1(&short_host),
        Err(BundleAbiError::StructSize {
            surface: "host services",
            ..
        })
    ));

    let mut legacy_prefix = *fixture_table().services();
    legacy_prefix.struct_size = offset_of!(abi::BundleHostServicesV1, acquire_serving_kv) as u32;
    legacy_prefix.acquire_serving_kv = None;
    legacy_prefix.resolve_serving_kv_pool = None;
    validate_host_services_v1(&legacy_prefix)
        .expect("the V1 prefix remains valid without serving services");

    let mut short_result = fixture_raw_api();
    short_result.create = Some(create_with_short_result);
    let api = BundleApiV1::new(short_result).expect("fixture API table is valid");
    assert!(matches!(
        api.create(fixture_table(), &[]),
        Err(BundleAbiError::StructSize {
            surface: "create result",
            ..
        })
    ));

    // A longer table is a later revision of the same prefix, and this version
    // reads it at its own length. Its version is still exact: a table that
    // declares more bytes and a different ABI version is refused for the
    // version, because length never stands in for it.
    let mut longer_api = fixture_raw_api();
    longer_api.struct_size += 8;
    assert!(BundleApiV1::new(longer_api).is_ok());
    let mut longer_api_wrong_version = longer_api;
    longer_api_wrong_version.abi_version = abi::BUNDLE_ABI_VERSION_V1 + 1;
    assert!(matches!(
        BundleApiV1::new(longer_api_wrong_version),
        Err(BundleAbiError::AbiVersion {
            surface: "bundle API",
            ..
        })
    ));

    let mut longer_host = *fixture_table().services();
    longer_host.struct_size += 8;
    assert!(validate_host_services_v1(&longer_host).is_ok());
    let mut longer_host_wrong_version = longer_host;
    longer_host_wrong_version.abi_version = abi::BUNDLE_ABI_VERSION_V1 + 1;
    assert!(matches!(
        validate_host_services_v1(&longer_host_wrong_version),
        Err(BundleAbiError::AbiVersion {
            surface: "host services",
            ..
        })
    ));
}

/// Every V1 record refuses a shorter length and is read at its own.
///
/// The accepted longer length is deliberate forward-tail compatibility: the V1
/// fields are a fixed prefix a later revision may append to, and no reader of
/// this version reads past that prefix, so the tail decides nothing. It is also
/// why `abi_version` stays exact, which the table cases above pin: a longer
/// record is a later revision of the same prefix, and a different version is
/// not. The header states the same rule where a bundle author reads it.
#[test]
fn every_v1_record_refuses_a_short_length_and_is_read_at_its_own() {
    use runtime_bundle_abi::validate_tensor_map_v1;

    macro_rules! assert_forward_tail {
        ($surface:literal, $validate:path, $record:expr) => {{
            let valid = $record;
            let exact = u32::try_from(size_of_val(&valid)).expect("a V1 record fits in u32");
            assert_eq!(
                valid.struct_size,
                exact,
                concat!("a built ", $surface, " declares its own length")
            );
            assert!(
                $validate(valid).is_ok(),
                concat!("this ", $surface, " is valid at its own length")
            );

            let mut short = valid;
            short.struct_size = exact - 1;
            assert!(
                matches!(
                    $validate(short),
                    Err(BundleAbiError::StructSize { surface, expected, actual })
                        if surface == $surface && expected == exact && actual == exact - 1
                ),
                concat!("a ", $surface, " shorter than this layout must be refused")
            );

            let mut longer = valid;
            longer.struct_size = exact + 8;
            assert!(
                $validate(longer).is_ok(),
                concat!("a longer ", $surface, " is read at this version's own length")
            );
        }};
    }

    let id = [1_u8];
    let mut compiled_view = abi::BundleRawCompiledImageViewV1::empty();
    compiled_view.buffer = valid_device_buffer_view();
    compiled_view.alignment = 64;
    let layout = valid_physical_kv_layout(&id);

    // The device buffer view and the physical KV layout are the two by-value
    // nested records, checked below with the two array elements: a tail on
    // either would move its parent's own fields, so neither has one.
    assert_forward_tail!(
        "compiled image view",
        validate_compiled_image_view_v1,
        compiled_view
    );
    assert_forward_tail!(
        "launch config",
        validate_launch_config_v1,
        valid_launch_config()
    );
    // The kernel argument is the exception, and it is checked below rather than
    // here: its length is exact in both directions, so it has no forward tail
    // for this macro to assert.
    assert_forward_tail!(
        "device pointer array",
        validate_device_pointer_array_v1,
        valid_device_pointer_array()
    );
    assert_forward_tail!("tensor map", validate_tensor_map_v1, valid_tensor_map());
    // The custom operand is the other exception, checked below with the kernel
    // argument: both are array elements, so neither has a forward tail.
    assert_forward_tail!(
        "physical KV request",
        validate_physical_kv_request_v1,
        valid_physical_kv_request(layout)
    );
    assert_forward_tail!(
        "state request",
        validate_state_request_v1,
        valid_state_request(&id)
    );
    assert_forward_tail!(
        "collective request",
        validate_collective_request_v1,
        valid_collective_request(&id)
    );
    assert_forward_tail!(
        "logits row",
        validate_execute_logits_row_v1,
        fixture_logits_row()
    );

    // The row callback is the required prefix. The device-view callback is an
    // optional tail whose presence is checked only when a bundle uses it.
    let sink = abi::BundleExecuteResultSinkV1::empty();
    let sink_size = u32::try_from(size_of_val(&sink)).expect("a V1 record fits in u32");
    let prefix_size = offset_of!(abi::BundleExecuteResultSinkV1, emit_logits_view) as u32;
    let mut short_sink = sink;
    short_sink.struct_size = prefix_size - 1;
    assert!(matches!(
        validate_execute_result_sink_v1(short_sink),
        Err(BundleAbiError::StructSize {
            surface: "execute result sink",
            ..
        })
    ));
    let mut longer_sink = sink;
    longer_sink.struct_size = sink_size + 8;
    assert!(matches!(
        validate_execute_result_sink_v1(longer_sink),
        Err(BundleAbiError::MissingExecuteResultCallback)
    ));

    let header = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../include/chop_bundle_v1.h");
    let text = fs::read_to_string(header).expect("the V1 header is readable");
    for stated in [
        "A reader refuses a record shorter than the",
        "accepts one at least that long and reads exactly its own V1 layout",
        "That forward tail is deliberate, and it is deliberately inert.",
        "abi_version is exact.",
    ] {
        assert!(
            text.contains(stated),
            "the V1 header must state the length rule: {stated}"
        );
    }
    // The two rules a caller obeys for the life of one instance. Neither is
    // visible in a signature, so the header is where a caller finds them.
    for stated in [
        "One instance belongs to one thread",
        "At most one transaction is pending per instance",
    ] {
        assert!(
            text.contains(stated),
            "the V1 header must state the lifecycle rule: {stated}"
        );
    }
}

#[test]
fn header_remains_explicitly_draft() {
    let header = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../include/chop_bundle_v1.h");
    let text = fs::read_to_string(header).expect("draft ABI header is readable");
    assert!(text.contains("DRAFT: this V1 header is not stable"));
}

#[test]
fn host_service_table_requires_every_approved_operation() {
    let table = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
    assert!(validate_host_services_v1(table.services()).is_ok());

    macro_rules! assert_required {
        ($field:ident) => {{
            let mut services = *table.services();
            services.$field = None;
            assert!(matches!(
                validate_host_services_v1(&services),
                Err(BundleAbiError::MissingHostService(name)) if name == stringify!($field)
            ));
        }};
    }

    assert_required!(log);
    assert_required!(select_context);
    assert_required!(release_context);
    assert_required!(select_memory_pool);
    assert_required!(set_memory_pool_peer_access);
    assert_required!(release_memory_pool);
    assert_required!(allocate_device_buffer);
    assert_required!(release_device_buffer);
    assert_required!(open_compiled_image);
    assert_required!(upload_rank_arena);
    assert_required!(release_rank_arena);
    assert_required!(resolve_compiled_image_view);
    assert_required!(release_compiled_image);
    assert_required!(open_authenticated_component);
    assert_required!(inspect_authenticated_component);
    assert_required!(read_authenticated_component);
    assert_required!(release_authenticated_component);
    assert_required!(acquire_peer_access);
    assert_required!(release_peer_access);
    assert_required!(create_stream);
    assert_required!(select_stage_stream);
    assert_required!(synchronize_stream);
    assert_required!(release_stream);
    assert_required!(create_event);
    assert_required!(record_event);
    assert_required!(wait_event);
    assert_required!(synchronize_event);
    assert_required!(release_event);
    assert_required!(copy_host_to_device);
    assert_required!(copy_device_to_host);
    assert_required!(copy_device_to_device);
    assert_required!(copy_peer_to_peer);
    assert_required!(zero_device_buffer);
    assert_required!(write_device_pointer_table);
    assert_required!(load_cuda_module);
    assert_required!(resolve_cuda_symbol);
    assert_required!(configure_cuda_symbol);
    assert_required!(launch_cuda_symbol);
    assert_required!(release_cuda_symbol);
    assert_required!(release_cuda_module);
    assert_required!(allocate_physical_kv);
    assert_required!(create_kv_request);
    assert_required!(prepare_kv);
    assert_required!(commit_kv);
    assert_required!(cancel_kv);
    assert_required!(resolve_kv_view);
    assert_required!(release_kv_request);
    assert_required!(release_physical_kv);
    assert_required!(allocate_state);
    assert_required!(resolve_state_view);
    assert_required!(release_state);
    assert_required!(create_collective);
    assert_required!(execute_collective);
    assert_required!(release_collective);
}

#[test]
fn lifecycle_calls_are_exactly_once_and_destroyed_handles_are_stale() {
    let _guard = TEST_LOCK.lock().expect("ABI test lock is available");
    reset_calls();
    let api = fixture_api();
    let bytes = [1_u8, 2, 3];
    let mut sink = fixture_sink();

    let mut committed = api
        .create(fixture_table(), &bytes)
        .expect("fixture instance creates");
    committed
        .execute(&bytes, &mut sink)
        .expect("fixture instance executes");
    committed.commit().expect("fixture transaction commits");
    committed.destroy().expect("fixture instance destroys");
    assert!(
        api.execute_handle(HANDLE, &bytes, &mut sink).is_err(),
        "destroyed generation is not executable"
    );
    drop(committed);

    let mut cancelled = api
        .create(fixture_table(), &bytes)
        .expect("second fixture instance creates");
    cancelled
        .execute(&bytes, &mut sink)
        .expect("second fixture instance executes");
    cancelled.cancel().expect("fixture transaction cancels");
    cancelled
        .destroy()
        .expect("second fixture instance destroys");
    drop(cancelled);

    assert_eq!(
        calls(),
        Calls {
            create: 2,
            execute: 2,
            commit: 1,
            cancel: 1,
            destroy: 2,
            active: false,
            fail_commit: false,
            fail_cancel: false,
            fail_destroy: false,
        }
    );
    assert_eq!(sink.rows.len(), 2);
    assert_eq!(sink.rows[0].request_id(), FIXTURE_REQUEST_ID);
    assert_eq!(sink.rows[0].logit_count(), 2);
    assert_eq!(sink.rows[0].logit_bits(), FIXTURE_LOGIT_BITS);
}

#[test]
fn execute_result_sink_delivers_rows_before_commit_and_enforces_bounds() {
    let _guard = TEST_LOCK.lock().expect("ABI test lock is available");
    reset_calls();
    let api = fixture_api();
    let bytes = [1_u8, 2, 3];
    let mut sink = fixture_sink();
    let mut instance = api
        .create(fixture_table(), &bytes)
        .expect("fixture instance creates");

    instance
        .execute(&bytes, &mut sink)
        .expect("fixture instance executes");
    assert_eq!(
        sink.rows.len(),
        1,
        "result rows arrive before the transaction is finalized"
    );
    instance.commit().expect("fixture transaction commits");

    let mut bounded = fixture_sink();
    assert!(matches!(
        instance.execute(SECOND_ROW_REQUEST, &mut bounded),
        Err(BundleAbiError::Status(
            abi::BundleStatusV1::ResourceExhausted
        ))
    ));
    assert_eq!(
        bounded.rows.len(),
        1,
        "the sink stops the bundle at its declared row limit"
    );

    instance.destroy().expect("fixture instance destroys");
}

#[test]
fn lifecycle_wrapper_rejects_duplicate_or_unclosed_transitions() {
    let _guard = TEST_LOCK.lock().expect("ABI test lock is available");
    reset_calls();
    let api = fixture_api();
    let bytes = [1_u8];
    let mut sink = fixture_sink();
    let mut instance = api
        .create(fixture_table(), &bytes)
        .expect("fixture instance creates");

    assert!(matches!(
        instance.commit(),
        Err(BundleAbiError::Lifecycle("commit"))
    ));
    instance
        .execute(&bytes, &mut sink)
        .expect("execution starts");
    assert!(matches!(
        instance.execute(&bytes, &mut sink),
        Err(BundleAbiError::Lifecycle("execute"))
    ));
    assert!(matches!(
        instance.destroy(),
        Err(BundleAbiError::Lifecycle("destroy"))
    ));
    instance.cancel().expect("execution cancels");
    instance.destroy().expect("instance destroys");
    assert!(matches!(
        instance.destroy(),
        Err(BundleAbiError::Lifecycle("destroy"))
    ));
}

/// What a bundle reports about an execution decides what the caller may do.
///
/// A refusal, including a contained panic, leaves this wrapper ready, so it
/// issues no finalizer at all: a bundle that reported failure while keeping a
/// transaction open would have hidden it from every later call. A success
/// leaves the wrapper pending even when no row arrived, which is what keeps an
/// execution a bundle could not roll back cancellable.
#[test]
fn an_execute_refusal_leaves_the_wrapper_ready_and_an_empty_success_cancellable() {
    let _guard = TEST_LOCK.lock().expect("ABI test lock is available");
    reset_calls();
    let api = fixture_api();
    let mut sink = fixture_sink();
    let mut instance = api
        .create(fixture_table(), &[1_u8])
        .expect("fixture instance creates");

    assert!(matches!(
        instance.execute(FAILED_EXECUTE_REQUEST, &mut sink),
        Err(BundleAbiError::Status(abi::BundleStatusV1::BundleError))
    ));
    assert!(sink.rows.is_empty());
    assert!(matches!(
        instance.commit(),
        Err(BundleAbiError::Lifecycle("commit"))
    ));
    assert!(matches!(
        instance.cancel(),
        Err(BundleAbiError::Lifecycle("cancel"))
    ));

    assert!(matches!(
        instance.execute(PANIC_REQUEST, &mut sink),
        Err(BundleAbiError::Status(abi::BundleStatusV1::Panic))
    ));
    assert!(matches!(
        instance.cancel(),
        Err(BundleAbiError::Lifecycle("cancel"))
    ));

    instance
        .execute(EMPTY_RESULT_REQUEST, &mut sink)
        .expect("an execution with no row is still an execution");
    assert!(
        sink.rows.is_empty(),
        "an empty result gives the caller nothing it could commit"
    );
    assert!(matches!(
        instance.execute(EMPTY_RESULT_REQUEST, &mut sink),
        Err(BundleAbiError::Lifecycle("execute"))
    ));
    instance
        .cancel()
        .expect("the pending execution is reachable through cancel");
    instance.destroy().expect("fixture instance destroys");

    let calls = calls();
    assert_eq!(
        calls.execute, 3,
        "the two refusals and the empty success reached the bundle, and the duplicate did not"
    );
    assert_eq!(
        calls.commit, 0,
        "a wrapper that stayed ready finalizes nothing"
    );
    assert_eq!(calls.cancel, 1);
    assert_eq!(calls.destroy, 1);
}

#[test]
fn terminal_callback_errors_poison_without_retrying_lifecycle_callbacks() {
    let _guard = TEST_LOCK.lock().expect("ABI test lock is available");
    let bytes = [1_u8];
    let mut sink = fixture_sink();

    reset_calls();
    let mut commit_instance = fixture_api()
        .create(fixture_table(), &bytes)
        .expect("commit fixture instance creates");
    commit_instance
        .execute(&bytes, &mut sink)
        .expect("commit fixture execution starts");
    CALLS
        .lock()
        .expect("fixture call state is available")
        .fail_commit = true;
    assert!(matches!(
        commit_instance.commit(),
        Err(BundleAbiError::Status(abi::BundleStatusV1::BundleError))
    ));
    assert!(commit_instance.is_poisoned());
    assert!(matches!(
        commit_instance.commit(),
        Err(BundleAbiError::Poisoned)
    ));
    assert!(matches!(
        commit_instance.cancel(),
        Err(BundleAbiError::Poisoned)
    ));
    assert!(matches!(
        commit_instance.destroy(),
        Err(BundleAbiError::Poisoned)
    ));
    drop(commit_instance);
    let failed_commit_calls = calls();
    assert_eq!(failed_commit_calls.commit, 1);
    assert_eq!(failed_commit_calls.cancel, 0);
    assert_eq!(failed_commit_calls.destroy, 0);

    reset_calls();
    let mut cancel_instance = fixture_api()
        .create(fixture_table(), &bytes)
        .expect("cancel fixture instance creates");
    cancel_instance
        .execute(&bytes, &mut sink)
        .expect("cancel fixture execution starts");
    CALLS
        .lock()
        .expect("fixture call state is available")
        .fail_cancel = true;
    assert!(matches!(
        cancel_instance.cancel(),
        Err(BundleAbiError::Status(abi::BundleStatusV1::BundleError))
    ));
    assert!(cancel_instance.is_poisoned());
    assert!(matches!(
        cancel_instance.cancel(),
        Err(BundleAbiError::Poisoned)
    ));
    assert!(matches!(
        cancel_instance.destroy(),
        Err(BundleAbiError::Poisoned)
    ));
    drop(cancel_instance);
    let failed_cancel_calls = calls();
    assert_eq!(failed_cancel_calls.commit, 0);
    assert_eq!(failed_cancel_calls.cancel, 1);
    assert_eq!(failed_cancel_calls.destroy, 0);

    reset_calls();
    {
        let mut drop_cancel_instance = fixture_api()
            .create(fixture_table(), &bytes)
            .expect("drop cancel fixture instance creates");
        drop_cancel_instance
            .execute(&bytes, &mut sink)
            .expect("drop cancel fixture execution starts");
        CALLS
            .lock()
            .expect("fixture call state is available")
            .fail_cancel = true;
    }
    let drop_failed_cancel_calls = calls();
    assert_eq!(drop_failed_cancel_calls.commit, 0);
    assert_eq!(drop_failed_cancel_calls.cancel, 1);
    assert_eq!(drop_failed_cancel_calls.destroy, 0);

    reset_calls();
    let mut destroy_instance = fixture_api()
        .create(fixture_table(), &bytes)
        .expect("destroy fixture instance creates");
    CALLS
        .lock()
        .expect("fixture call state is available")
        .fail_destroy = true;
    assert!(matches!(
        destroy_instance.destroy(),
        Err(BundleAbiError::Status(abi::BundleStatusV1::BundleError))
    ));
    assert!(destroy_instance.is_poisoned());
    assert!(matches!(
        destroy_instance.destroy(),
        Err(BundleAbiError::Poisoned)
    ));
    drop(destroy_instance);
    let failed_destroy_calls = calls();
    assert_eq!(failed_destroy_calls.commit, 0);
    assert_eq!(failed_destroy_calls.cancel, 0);
    assert_eq!(failed_destroy_calls.destroy, 1);
}

// The fixture bundle below is a downstream implementation: it receives only
// checked Rust values and contains no unsafe code of its own.
abi::export_bundle_api_v1!(
    FIXTURE_API,
    fixture_create,
    fixture_execute,
    fixture_commit,
    fixture_cancel,
    fixture_destroy,
);

fn fixture_create(
    host_services: HostServicesV1,
    configuration: &[u8],
) -> abi::BundleCreateResultV1 {
    assert!(configuration.len() < 8);
    if host_services.log(b"bundle-create").is_ok() {
        let context = host_services
            .select_context(0, 1)
            .expect("the bundle selects a context through the safe caller");
        assert!(context.has_generation());
    }
    // A real bundle keeps its caller for the services it needs during execute.
    RETAINED_CALLER.with(|caller| *caller.borrow_mut() = Some(host_services));
    if configuration == FAILED_CREATE_REQUEST {
        // A bundle may keep the caller it was handed and still report a failure.
        return abi::BundleCreateResultV1::failure(abi::BundleStatusV1::BundleError);
    }
    let mut calls = CALLS.lock().expect("fixture call state is available");
    calls.create += 1;
    calls.active = true;
    abi::BundleCreateResultV1::success(HANDLE)
}

thread_local! {
    /// The caller the fixture bundle kept, exactly as a real bundle would.
    static RETAINED_CALLER: RefCell<Option<HostServicesV1>> = const { RefCell::new(None) };
}

/// Calls one host service through the caller the fixture bundle retained.
fn call_retained_caller(message: &[u8]) -> Result<(), BundleAbiError> {
    RETAINED_CALLER.with(|caller| {
        caller
            .borrow()
            .as_ref()
            .expect("the fixture bundle retained its caller")
            .log(message)
    })
}

fn fixture_execute(
    handle: abi::BundleHandleV1,
    request: &[u8],
    result_sink: &mut ExecuteResultSinkV1,
) -> abi::BundleStatusV1 {
    {
        let mut calls = CALLS.lock().expect("fixture call state is available");
        if !calls.active || handle != HANDLE {
            return abi::BundleStatusV1::StaleHandle;
        }
        calls.execute += 1;
    }
    if request.starts_with(b"refuse-") {
        if request == b"refuse-long" {
            call_retained_caller("check: é\n".repeat(200).as_bytes()).unwrap();
        } else if request != b"refuse-silent" {
            call_retained_caller(b"packing check: item token budget exceeded").unwrap();
        }
        if request == b"refuse-after-service" {
            RETAINED_CALLER.with(|caller| {
                caller
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .copy_host_to_device(
                        abi::BundleStreamHandleV1::from_parts(1, 1),
                        DeviceBufferSpanV1::new(
                            abi::BundleDeviceBufferHandleV1::from_parts(1, 1),
                            0,
                            1,
                        ),
                        &[0],
                    )
                    .unwrap();
            });
        }
        if request == b"refuse-after-row" {
            result_sink
                .emit(FIXTURE_REQUEST_ID, &FIXTURE_LOGIT_BITS)
                .unwrap();
        }
        return abi::BundleStatusV1::InvalidArgument;
    }
    if request == FAILED_EXECUTE_REQUEST {
        // A refusal owes the caller an instance with nothing left open.
        return abi::BundleStatusV1::BundleError;
    }
    if request == PANIC_REQUEST {
        panic!("injected fixture execute panic");
    }
    if request == EMPTY_RESULT_REQUEST {
        // What a bundle reports when it could not close what it opened: the
        // execution is pending and its result is unusable.
        return abi::BundleStatusV1::Success;
    }
    let rows = if request == SECOND_ROW_REQUEST { 2 } else { 1 };
    for _ in 0..rows {
        match result_sink.emit(FIXTURE_REQUEST_ID, &FIXTURE_LOGIT_BITS) {
            Ok(()) => {}
            Err(BundleAbiError::Status(status)) => return status,
            Err(BundleAbiError::ExecuteResultCapacity) => {
                return abi::BundleStatusV1::ResourceExhausted;
            }
            Err(_) => return abi::BundleStatusV1::InvalidArgument,
        }
    }
    abi::BundleStatusV1::Success
}

fn fixture_commit(handle: abi::BundleHandleV1) -> abi::BundleStatusV1 {
    let mut calls = CALLS.lock().expect("fixture call state is available");
    if !calls.active || handle != HANDLE {
        return abi::BundleStatusV1::StaleHandle;
    }
    calls.commit += 1;
    if calls.fail_commit {
        abi::BundleStatusV1::BundleError
    } else {
        abi::BundleStatusV1::Success
    }
}

fn fixture_cancel(handle: abi::BundleHandleV1) -> abi::BundleStatusV1 {
    let mut calls = CALLS.lock().expect("fixture call state is available");
    if !calls.active || handle != HANDLE {
        return abi::BundleStatusV1::StaleHandle;
    }
    calls.cancel += 1;
    if calls.fail_cancel {
        abi::BundleStatusV1::BundleError
    } else {
        abi::BundleStatusV1::Success
    }
}

fn fixture_destroy(handle: abi::BundleHandleV1) -> abi::BundleStatusV1 {
    let mut calls = CALLS.lock().expect("fixture call state is available");
    if !calls.active || handle != HANDLE {
        return abi::BundleStatusV1::StaleHandle;
    }
    calls.destroy += 1;
    if calls.fail_destroy {
        return abi::BundleStatusV1::BundleError;
    }
    calls.active = false;
    abi::BundleStatusV1::Success
}

extern "C" fn create_with_short_result(
    _host_services: *const abi::BundleHostServicesV1,
    _configuration: abi::BundleByteViewV1,
) -> abi::BundleCreateResultV1 {
    abi::BundleCreateResultV1 {
        struct_size: 0,
        status: abi::BundleStatusV1::Success,
        handle: HANDLE,
    }
}

fn fixture_api() -> BundleApiV1 {
    BundleApiV1::new(fixture_raw_api()).expect("fixture API is valid")
}

fn fixture_raw_api() -> abi::BundleRawApiV1 {
    FIXTURE_API
}

fn fixture_table() -> HostServicesTableV1 {
    HostServicesTableV1::new(Rc::new(FixtureHostBackend::default()))
}

fn fixture_table_for(backend: &Rc<FixtureHostBackend>) -> HostServicesTableV1 {
    HostServicesTableV1::new(Rc::clone(backend))
}

/// A host result sink that keeps every delivered row in owned Rust storage.
#[derive(Debug, Default)]
struct FixtureSink {
    rows: Vec<OwnedLogitsRowV1>,
}

impl LogitsRowSinkV1 for FixtureSink {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(1, 2, 2)
    }

    fn emit_row(&mut self, row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        self.rows.push(row);
        Ok(())
    }
}

fn fixture_sink() -> FixtureSink {
    FixtureSink::default()
}

/// One custom operation this fixture host was asked to run.
#[derive(Clone, Debug, Eq, PartialEq)]
struct FixtureCustomOperationV1 {
    identifier: Vec<u8>,
    completion: CustomCompletionV1,
    operands: Vec<CustomOperandV1>,
}

/// A host backend that records the safe values the ABI hands it.
#[derive(Debug, Default)]
struct FixtureHostBackend {
    messages: RefCell<Vec<Vec<u8>>>,
    contexts: RefCell<Vec<(u32, i32)>>,
    host_to_device: RefCell<Vec<(DeviceBufferSpanV1, Vec<u8>)>>,
    device_to_host: RefCell<Vec<DeviceBufferSpanV1>>,
    launches: RefCell<Vec<(LaunchConfigV1, Vec<KernelArgumentV1>)>>,
    physical_kv: RefCell<Vec<PhysicalKvRequestV1>>,
    collectives: RefCell<Vec<CollectiveRequestV1>>,
    kv_views: RefCell<Vec<Vec<u8>>>,
    /// Every custom operation this host invoked, with what it was asked for.
    custom_operations: RefCell<Vec<FixtureCustomOperationV1>>,
    /// The refusal this host answers a custom operation with, if any.
    custom_refusal: RefCell<Option<abi::BundleStatusV1>>,
}

macro_rules! unused_fixture_service {
    ($name:ident($($argument:ident: $type:ty),* $(,)?) -> $result:ty) => {
        fn $name(&self, $($argument: $type),*) -> HostServiceResultV1<$result> {
            Err(abi::BundleStatusV1::Unsupported)
        }
    };
}

impl HostServicesBackendV1 for FixtureHostBackend {
    fn log(&self, message: &[u8]) -> HostServiceResultV1<()> {
        self.messages.borrow_mut().push(message.to_vec());
        Ok(())
    }

    fn select_context(
        &self,
        rank: u32,
        device_ordinal: i32,
    ) -> HostServiceResultV1<abi::BundleContextHandleV1> {
        self.contexts.borrow_mut().push((rank, device_ordinal));
        Ok(abi::BundleContextHandleV1::from_parts(
            7,
            u32::try_from(device_ordinal).unwrap_or(1),
        ))
    }

    fn copy_host_to_device(
        &self,
        _stream: abi::BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        self.host_to_device
            .borrow_mut()
            .push((destination, source.to_vec()));
        Ok(())
    }

    fn copy_device_to_host(
        &self,
        _stream: abi::BundleStreamHandleV1,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        destination.fill(0xAB);
        self.device_to_host.borrow_mut().push(source);
        Ok(())
    }

    fn launch_cuda_symbol(
        &self,
        _stream: abi::BundleStreamHandleV1,
        _symbol: abi::BundleCudaSymbolHandleV1,
        config: LaunchConfigV1,
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        self.launches
            .borrow_mut()
            .push((config, arguments.to_vec()));
        Ok(())
    }

    fn allocate_physical_kv(
        &self,
        request: &PhysicalKvRequestV1,
    ) -> HostServiceResultV1<abi::BundlePhysicalKvHandleV1> {
        self.physical_kv.borrow_mut().push(request.clone());
        Ok(abi::BundlePhysicalKvHandleV1::from_parts(5, 1))
    }

    fn resolve_kv_view(
        &self,
        _request: abi::BundleKvRequestHandleV1,
        _rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        self.kv_views.borrow_mut().push(view_id.to_vec());
        Ok(fixture_device_span())
    }

    fn create_collective(
        &self,
        request: &CollectiveRequestV1,
    ) -> HostServiceResultV1<abi::BundleCollectiveHandleV1> {
        self.collectives.borrow_mut().push(request.clone());
        Ok(abi::BundleCollectiveHandleV1::from_parts(3, 1))
    }

    fn release_stream(&self, stream: abi::BundleStreamHandleV1) -> HostServiceResultV1<()> {
        if stream.has_generation() {
            Ok(())
        } else {
            Err(abi::BundleStatusV1::StaleHandle)
        }
    }

    fn synchronize_event(&self, _event: abi::BundleEventHandleV1) -> HostServiceResultV1<()> {
        panic!("fixture host backend panic")
    }

    unused_fixture_service!(release_context(_context: abi::BundleContextHandleV1) -> ());
    /// Returns a handle with no generation, which the safe caller must reject.
    fn select_memory_pool(
        &self,
        _context: abi::BundleContextHandleV1,
    ) -> HostServiceResultV1<abi::BundleMemoryPoolHandleV1> {
        Ok(abi::BundleMemoryPoolHandleV1::from_raw(0))
    }

    /// Returns a span with no buffer handle, which the safe caller must reject.
    fn resolve_state_view(
        &self,
        _state: abi::BundleStateHandleV1,
        _rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        Ok(DeviceBufferSpanV1::new(
            abi::BundleDeviceBufferHandleV1::from_raw(0),
            0,
            64,
        ))
    }
    unused_fixture_service!(
        set_memory_pool_peer_access(
            _pool: abi::BundleMemoryPoolHandleV1,
            _peer_context: abi::BundleContextHandleV1,
            _enabled: bool,
        ) -> ()
    );
    unused_fixture_service!(
        release_memory_pool(_pool: abi::BundleMemoryPoolHandleV1) -> ()
    );
    unused_fixture_service!(
        allocate_device_buffer(
            _pool: abi::BundleMemoryPoolHandleV1,
            _stream: abi::BundleStreamHandleV1,
            _byte_length: u64,
            _alignment: u64,
        ) -> abi::BundleDeviceBufferHandleV1
    );
    unused_fixture_service!(
        release_device_buffer(_buffer: abi::BundleDeviceBufferHandleV1) -> ()
    );
    unused_fixture_service!(
        open_compiled_image(_component_role: &[u8]) -> abi::BundleCompiledImageHandleV1
    );
    unused_fixture_service!(
        upload_rank_arena(
            _context: abi::BundleContextHandleV1,
            _pool: abi::BundleMemoryPoolHandleV1,
            _stream: abi::BundleStreamHandleV1,
            _image: abi::BundleCompiledImageHandleV1,
            _rank: u32,
        ) -> abi::BundleRankArenaHandleV1
    );
    unused_fixture_service!(release_rank_arena(_arena: abi::BundleRankArenaHandleV1) -> ());
    unused_fixture_service!(
        resolve_compiled_image_view(
            _image: abi::BundleCompiledImageHandleV1,
            _arena: abi::BundleRankArenaHandleV1,
            _rank: u32,
            _view_id: &[u8],
        ) -> CompiledImageSpanV1
    );
    unused_fixture_service!(
        release_compiled_image(_image: abi::BundleCompiledImageHandleV1) -> ()
    );
    unused_fixture_service!(
        acquire_peer_access(
            _owner_context: abi::BundleContextHandleV1,
            _peer_context: abi::BundleContextHandleV1,
        ) -> abi::BundlePeerAccessHandleV1
    );
    unused_fixture_service!(
        release_peer_access(_peer_access: abi::BundlePeerAccessHandleV1) -> ()
    );
    unused_fixture_service!(
        create_stream(
            _context: abi::BundleContextHandleV1,
            _stream_kind: u32,
        ) -> abi::BundleStreamHandleV1
    );
    unused_fixture_service!(
        select_stage_stream(
            _context: abi::BundleContextHandleV1,
            _rank: u32,
            _stage_id: &[u8],
            _stream_kind: u32,
        ) -> abi::BundleStreamHandleV1
    );
    unused_fixture_service!(synchronize_stream(_stream: abi::BundleStreamHandleV1) -> ());
    unused_fixture_service!(
        create_event(_context: abi::BundleContextHandleV1) -> abi::BundleEventHandleV1
    );
    unused_fixture_service!(
        record_event(
            _event: abi::BundleEventHandleV1,
            _stream: abi::BundleStreamHandleV1,
        ) -> ()
    );
    unused_fixture_service!(
        wait_event(
            _stream: abi::BundleStreamHandleV1,
            _event: abi::BundleEventHandleV1,
        ) -> ()
    );
    unused_fixture_service!(release_event(_event: abi::BundleEventHandleV1) -> ());
    unused_fixture_service!(
        copy_device_to_device(
            _stream: abi::BundleStreamHandleV1,
            _destination: DeviceBufferSpanV1,
            _source: DeviceBufferSpanV1,
        ) -> ()
    );
    unused_fixture_service!(
        copy_peer_to_peer(
            _stream: abi::BundleStreamHandleV1,
            _destination_context: abi::BundleContextHandleV1,
            _destination: DeviceBufferSpanV1,
            _source_context: abi::BundleContextHandleV1,
            _source: DeviceBufferSpanV1,
        ) -> ()
    );
    unused_fixture_service!(
        zero_device_buffer(
            _stream: abi::BundleStreamHandleV1,
            _destination: DeviceBufferSpanV1,
        ) -> ()
    );
    unused_fixture_service!(
        write_device_pointer_table(
            _stream: abi::BundleStreamHandleV1,
            _destination: DeviceBufferSpanV1,
            _slots: &[Option<DeviceBufferSpanV1>],
        ) -> ()
    );
    unused_fixture_service!(
        load_cuda_module(
            _context: abi::BundleContextHandleV1,
            _module_bytes: &[u8],
        ) -> abi::BundleCudaModuleHandleV1
    );
    unused_fixture_service!(
        resolve_cuda_symbol(
            _module: abi::BundleCudaModuleHandleV1,
            _symbol_name: &[u8],
        ) -> abi::BundleCudaSymbolHandleV1
    );
    unused_fixture_service!(
        configure_cuda_symbol(
            _symbol: abi::BundleCudaSymbolHandleV1,
            _dynamic_shared_memory_bytes: u32,
        ) -> ()
    );
    unused_fixture_service!(
        release_cuda_symbol(_symbol: abi::BundleCudaSymbolHandleV1) -> ()
    );
    unused_fixture_service!(
        release_cuda_module(_module: abi::BundleCudaModuleHandleV1) -> ()
    );
    unused_fixture_service!(
        create_kv_request(
            _kv: abi::BundlePhysicalKvHandleV1,
            _quota_tokens: u64,
        ) -> abi::BundleKvRequestHandleV1
    );
    unused_fixture_service!(
        prepare_kv(
            _request: abi::BundleKvRequestHandleV1,
            _operation: &[u8],
        ) -> abi::BundleKvTransactionHandleV1
    );
    unused_fixture_service!(commit_kv(_transaction: abi::BundleKvTransactionHandleV1) -> ());
    unused_fixture_service!(cancel_kv(_transaction: abi::BundleKvTransactionHandleV1) -> ());
    unused_fixture_service!(
        release_kv_request(_request: abi::BundleKvRequestHandleV1) -> ()
    );
    unused_fixture_service!(release_physical_kv(_kv: abi::BundlePhysicalKvHandleV1) -> ());
    unused_fixture_service!(
        allocate_state(_request: &StateRequestV1) -> abi::BundleStateHandleV1
    );
    unused_fixture_service!(release_state(_state: abi::BundleStateHandleV1) -> ());
    unused_fixture_service!(
        execute_collective(
            _collective: abi::BundleCollectiveHandleV1,
            _stream: abi::BundleStreamHandleV1,
            _operation: &[u8],
            _arguments: &[KernelArgumentV1],
        ) -> ()
    );
    unused_fixture_service!(
        release_collective(_collective: abi::BundleCollectiveHandleV1) -> ()
    );

    /// Lends made-up addresses to one operation, the way a host lends real ones.
    ///
    /// There is no memory here at all, so what the addresses stand for is the
    /// operand order they were built from. What this fixture does own is the
    /// sequence: a refusal returns before the callback exists, and an accepted
    /// call invokes it exactly once with the operands it was given.
    fn run_custom_operation(
        &self,
        _context: abi::BundleContextHandleV1,
        _stream: abi::BundleStreamHandleV1,
        completion: CustomCompletionV1,
        operation_id: &[u8],
        operands: &[CustomOperandV1],
        operation: &CustomOperationV1,
    ) -> HostServiceResultV1<i32> {
        if let Some(status) = *self.custom_refusal.borrow() {
            return Err(status);
        }
        self.custom_operations
            .borrow_mut()
            .push(FixtureCustomOperationV1 {
                identifier: operation_id.to_vec(),
                completion,
                operands: operands.to_vec(),
            });
        let ranges = operands
            .iter()
            .enumerate()
            .map(|(index, operand)| {
                LentRangeV1::new(
                    FIXTURE_CUSTOM_ADDRESS + (index as u64 + 1) * 0x1000,
                    operand.span.length,
                )
            })
            .collect::<Vec<_>>();
        operation.invoke(LentLaunchV1 {
            address_space: CustomAddressSpaceV1::Host,
            stream: LentStreamV1::null(),
            ranges: &ranges,
        })
    }
}

/// A host backend that records which service the safe caller reached.
#[derive(Debug, Default)]
struct RecordingHostBackend {
    calls: RefCell<Vec<&'static str>>,
}

macro_rules! recording_fixture_service {
    ($name:ident($($argument:ident: $type:ty),* $(,)?) -> $result:ty) => {
        fn $name(&self, $($argument: $type),*) -> HostServiceResultV1<$result> {
            self.calls.borrow_mut().push(stringify!($name));
            Err(abi::BundleStatusV1::Unsupported)
        }
    };
}

impl HostServicesBackendV1 for RecordingHostBackend {
    recording_fixture_service!(log(_message: &[u8]) -> ());
    recording_fixture_service!(select_context(_rank: u32, _device_ordinal: i32) -> abi::BundleContextHandleV1);
    recording_fixture_service!(release_context(_context: abi::BundleContextHandleV1) -> ());
    recording_fixture_service!(select_memory_pool(_context: abi::BundleContextHandleV1) -> abi::BundleMemoryPoolHandleV1);
    recording_fixture_service!(set_memory_pool_peer_access(
            _pool: abi::BundleMemoryPoolHandleV1,
            _peer_context: abi::BundleContextHandleV1,
            _enabled: bool,
        ) -> ());
    recording_fixture_service!(release_memory_pool(_pool: abi::BundleMemoryPoolHandleV1) -> ());
    recording_fixture_service!(allocate_device_buffer(
            _pool: abi::BundleMemoryPoolHandleV1,
            _stream: abi::BundleStreamHandleV1,
            _byte_length: u64,
            _alignment: u64,
        ) -> abi::BundleDeviceBufferHandleV1);
    recording_fixture_service!(release_device_buffer(_buffer: abi::BundleDeviceBufferHandleV1) -> ());
    recording_fixture_service!(open_compiled_image(_component_role: &[u8]) -> abi::BundleCompiledImageHandleV1);
    recording_fixture_service!(upload_rank_arena(
            _context: abi::BundleContextHandleV1,
            _pool: abi::BundleMemoryPoolHandleV1,
            _stream: abi::BundleStreamHandleV1,
            _image: abi::BundleCompiledImageHandleV1,
            _rank: u32,
        ) -> abi::BundleRankArenaHandleV1);
    recording_fixture_service!(release_rank_arena(_arena: abi::BundleRankArenaHandleV1) -> ());
    recording_fixture_service!(resolve_compiled_image_view(
            _image: abi::BundleCompiledImageHandleV1,
            _arena: abi::BundleRankArenaHandleV1,
            _rank: u32,
            _view_id: &[u8],
        ) -> CompiledImageSpanV1);
    recording_fixture_service!(release_compiled_image(_image: abi::BundleCompiledImageHandleV1) -> ());
    recording_fixture_service!(open_authenticated_component(_component_role: &[u8]) -> abi::BundleAuthenticatedComponentHandleV1);
    recording_fixture_service!(inspect_authenticated_component(
            _component: abi::BundleAuthenticatedComponentHandleV1,
        ) -> abi::AuthenticatedComponentMetadataV1);
    recording_fixture_service!(read_authenticated_component(
            _component: abi::BundleAuthenticatedComponentHandleV1,
            _role_destination: &mut [u8],
            _bytes_destination: &mut [u8],
        ) -> ());
    recording_fixture_service!(release_authenticated_component(
            _component: abi::BundleAuthenticatedComponentHandleV1,
        ) -> ());
    recording_fixture_service!(acquire_peer_access(
            _owner_context: abi::BundleContextHandleV1,
            _peer_context: abi::BundleContextHandleV1,
        ) -> abi::BundlePeerAccessHandleV1);
    recording_fixture_service!(release_peer_access(_peer_access: abi::BundlePeerAccessHandleV1) -> ());
    recording_fixture_service!(create_stream(
            _context: abi::BundleContextHandleV1,
            _stream_kind: u32,
        ) -> abi::BundleStreamHandleV1);
    recording_fixture_service!(select_stage_stream(
            _context: abi::BundleContextHandleV1,
            _rank: u32,
            _stage_id: &[u8],
            _stream_kind: u32,
        ) -> abi::BundleStreamHandleV1);
    recording_fixture_service!(synchronize_stream(_stream: abi::BundleStreamHandleV1) -> ());
    recording_fixture_service!(release_stream(_stream: abi::BundleStreamHandleV1) -> ());
    recording_fixture_service!(create_event(_context: abi::BundleContextHandleV1) -> abi::BundleEventHandleV1);
    recording_fixture_service!(record_event(
            _event: abi::BundleEventHandleV1,
            _stream: abi::BundleStreamHandleV1,
        ) -> ());
    recording_fixture_service!(wait_event(
            _stream: abi::BundleStreamHandleV1,
            _event: abi::BundleEventHandleV1,
        ) -> ());
    recording_fixture_service!(synchronize_event(_event: abi::BundleEventHandleV1) -> ());
    recording_fixture_service!(release_event(_event: abi::BundleEventHandleV1) -> ());
    recording_fixture_service!(copy_host_to_device(
            _stream: abi::BundleStreamHandleV1,
            _destination: DeviceBufferSpanV1,
            _source: &[u8],
        ) -> ());
    recording_fixture_service!(copy_device_to_host(
            _stream: abi::BundleStreamHandleV1,
            _destination: &mut [u8],
            _source: DeviceBufferSpanV1,
        ) -> ());
    recording_fixture_service!(copy_device_to_device(
            _stream: abi::BundleStreamHandleV1,
            _destination: DeviceBufferSpanV1,
            _source: DeviceBufferSpanV1,
        ) -> ());
    recording_fixture_service!(copy_peer_to_peer(
            _stream: abi::BundleStreamHandleV1,
            _destination_context: abi::BundleContextHandleV1,
            _destination: DeviceBufferSpanV1,
            _source_context: abi::BundleContextHandleV1,
            _source: DeviceBufferSpanV1,
        ) -> ());
    recording_fixture_service!(zero_device_buffer(
            _stream: abi::BundleStreamHandleV1,
            _destination: DeviceBufferSpanV1,
        ) -> ());
    recording_fixture_service!(write_device_pointer_table(
            _stream: abi::BundleStreamHandleV1,
            _destination: DeviceBufferSpanV1,
            _slots: &[Option<DeviceBufferSpanV1>],
        ) -> ());
    recording_fixture_service!(load_cuda_module(
            _context: abi::BundleContextHandleV1,
            _module_bytes: &[u8],
        ) -> abi::BundleCudaModuleHandleV1);
    recording_fixture_service!(resolve_cuda_symbol(
            _module: abi::BundleCudaModuleHandleV1,
            _symbol_name: &[u8],
        ) -> abi::BundleCudaSymbolHandleV1);
    recording_fixture_service!(configure_cuda_symbol(
            _symbol: abi::BundleCudaSymbolHandleV1,
            _dynamic_shared_memory_bytes: u32,
        ) -> ());
    recording_fixture_service!(launch_cuda_symbol(
            _stream: abi::BundleStreamHandleV1,
            _symbol: abi::BundleCudaSymbolHandleV1,
            _config: LaunchConfigV1,
            _arguments: &[KernelArgumentV1],
        ) -> ());
    recording_fixture_service!(release_cuda_symbol(_symbol: abi::BundleCudaSymbolHandleV1) -> ());
    recording_fixture_service!(release_cuda_module(_module: abi::BundleCudaModuleHandleV1) -> ());
    recording_fixture_service!(allocate_physical_kv(_request: &PhysicalKvRequestV1) -> abi::BundlePhysicalKvHandleV1);
    recording_fixture_service!(create_kv_request(
            _kv: abi::BundlePhysicalKvHandleV1,
            _quota_tokens: u64,
        ) -> abi::BundleKvRequestHandleV1);
    recording_fixture_service!(prepare_kv(
            _request: abi::BundleKvRequestHandleV1,
            _operation: &[u8],
        ) -> abi::BundleKvTransactionHandleV1);
    recording_fixture_service!(commit_kv(_transaction: abi::BundleKvTransactionHandleV1) -> ());
    recording_fixture_service!(cancel_kv(_transaction: abi::BundleKvTransactionHandleV1) -> ());
    recording_fixture_service!(resolve_kv_view(
            _request: abi::BundleKvRequestHandleV1,
            _rank: u32,
            _view_id: &[u8],
        ) -> DeviceBufferSpanV1);
    recording_fixture_service!(release_kv_request(_request: abi::BundleKvRequestHandleV1) -> ());
    recording_fixture_service!(release_physical_kv(_kv: abi::BundlePhysicalKvHandleV1) -> ());
    recording_fixture_service!(acquire_serving_kv(_layout_id: &[u8]) -> u64);
    recording_fixture_service!(resolve_serving_kv_pool(_rank: u32) -> DeviceBufferSpanV1);
    recording_fixture_service!(allocate_state(_request: &StateRequestV1) -> abi::BundleStateHandleV1);
    recording_fixture_service!(resolve_state_view(
            _state: abi::BundleStateHandleV1,
            _rank: u32,
        ) -> DeviceBufferSpanV1);
    recording_fixture_service!(release_state(_state: abi::BundleStateHandleV1) -> ());
    recording_fixture_service!(create_collective(_request: &CollectiveRequestV1) -> abi::BundleCollectiveHandleV1);
    recording_fixture_service!(execute_collective(
            _collective: abi::BundleCollectiveHandleV1,
            _stream: abi::BundleStreamHandleV1,
            _operation: &[u8],
            _arguments: &[KernelArgumentV1],
        ) -> ());
    recording_fixture_service!(release_collective(_collective: abi::BundleCollectiveHandleV1) -> ());
    recording_fixture_service!(run_custom_operation(
            _context: abi::BundleContextHandleV1,
            _stream: abi::BundleStreamHandleV1,
            _completion: CustomCompletionV1,
            _operation_id: &[u8],
            _operands: &[CustomOperandV1],
            _operation: &CustomOperationV1,
        ) -> i32);
}

/// Every approved V1 service, in the order the caller test exercises them.
const APPROVED_SERVICES: &[&str] = &[
    "log",
    "select_context",
    "release_context",
    "select_memory_pool",
    "set_memory_pool_peer_access",
    "release_memory_pool",
    "allocate_device_buffer",
    "release_device_buffer",
    "open_compiled_image",
    "upload_rank_arena",
    "release_rank_arena",
    "resolve_compiled_image_view",
    "release_compiled_image",
    "open_authenticated_component",
    "inspect_authenticated_component",
    "read_authenticated_component",
    "release_authenticated_component",
    "acquire_peer_access",
    "release_peer_access",
    "create_stream",
    "select_stage_stream",
    "synchronize_stream",
    "release_stream",
    "create_event",
    "record_event",
    "wait_event",
    "synchronize_event",
    "release_event",
    "copy_host_to_device",
    "copy_device_to_host",
    "copy_device_to_device",
    "copy_peer_to_peer",
    "zero_device_buffer",
    "write_device_pointer_table",
    "load_cuda_module",
    "resolve_cuda_symbol",
    "configure_cuda_symbol",
    "launch_cuda_symbol",
    "release_cuda_symbol",
    "release_cuda_module",
    "allocate_physical_kv",
    "create_kv_request",
    "prepare_kv",
    "commit_kv",
    "cancel_kv",
    "resolve_kv_view",
    "release_kv_request",
    "release_physical_kv",
    "acquire_serving_kv",
    "resolve_serving_kv_pool",
    "allocate_state",
    "resolve_state_view",
    "release_state",
    "create_collective",
    "execute_collective",
    "release_collective",
    "run_custom_operation",
];

fn byte_view(bytes: &[u8]) -> abi::BundleByteViewV1 {
    abi::BundleByteViewV1::from_bytes(bytes)
}

fn fixture_logits_row() -> abi::BundleLogitsRowV1 {
    abi::BundleLogitsRowV1 {
        struct_size: u32::try_from(size_of::<abi::BundleLogitsRowV1>())
            .expect("logits row size fits u32"),
        reserved: 0,
        request_id: FIXTURE_REQUEST_ID,
        logit_count: 2,
        logit_bits: byte_view(&FIXTURE_LOGIT_BITS),
    }
}

fn fixture_device_span() -> DeviceBufferSpanV1 {
    DeviceBufferSpanV1::new(abi::BundleDeviceBufferHandleV1::from_parts(1, 1), 0, 64)
}

fn fixture_launch_config() -> LaunchConfigV1 {
    LaunchConfigV1 {
        flags: 0,
        grid_x: 1,
        grid_y: 1,
        grid_z: 1,
        block_x: 1,
        block_y: 1,
        block_z: 1,
        shared_memory_bytes: 0,
        cluster: LaunchClusterV1::NONE,
    }
}

fn fixture_physical_kv_request() -> PhysicalKvRequestV1 {
    PhysicalKvRequestV1 {
        flags: 0,
        layout: PhysicalKvLayoutV1 {
            flags: 0,
            rank_count: 1,
            page_bytes: 64,
            page_alignment: 64,
            page_tokens: 1,
            state_lifetime: 0,
            layout_id: vec![1_u8],
        },
        page_capacity: 1,
        request_capacity: 1,
    }
}

fn fixture_state_request() -> StateRequestV1 {
    StateRequestV1 {
        flags: 0,
        byte_length: 64,
        alignment: 64,
        state_id: vec![1_u8],
    }
}

fn fixture_collective_request() -> CollectiveRequestV1 {
    CollectiveRequestV1 {
        flags: 0,
        rank_count: 1,
        semantic_contract: vec![1_u8],
    }
}

fn valid_device_buffer_view() -> abi::BundleDeviceBufferViewV1 {
    let mut view = abi::BundleDeviceBufferViewV1::empty();
    view.buffer = abi::BundleDeviceBufferHandleV1::from_parts(1, 1);
    view.length = 64;
    view
}

fn valid_launch_config() -> abi::BundleLaunchConfigV1 {
    let mut config = abi::BundleLaunchConfigV1::empty();
    config.grid_x = 1;
    config.grid_y = 1;
    config.grid_z = 1;
    config.block_x = 1;
    config.block_y = 1;
    config.block_z = 1;
    config.cluster_x = 1;
    config.cluster_y = 1;
    config.cluster_z = 1;
    config
}

fn valid_physical_kv_layout(id: &[u8]) -> abi::BundlePhysicalKvLayoutV1 {
    let mut layout = abi::BundlePhysicalKvLayoutV1::empty();
    layout.rank_count = 1;
    layout.page_bytes = 64;
    layout.page_alignment = 64;
    layout.page_tokens = 1;
    layout.layout_id = byte_view(id);
    layout
}

fn valid_physical_kv_request(
    layout: abi::BundlePhysicalKvLayoutV1,
) -> abi::BundlePhysicalKvRequestV1 {
    let mut request = abi::BundlePhysicalKvRequestV1::empty();
    request.layout = layout;
    request.page_capacity = 1;
    request.request_capacity = 1;
    request
}

fn valid_state_request(id: &[u8]) -> abi::BundleStateRequestV1 {
    let mut request = abi::BundleStateRequestV1::empty();
    request.byte_length = 64;
    request.alignment = 64;
    request.state_id = byte_view(id);
    request
}

fn valid_collective_request(id: &[u8]) -> abi::BundleCollectiveRequestV1 {
    let mut request = abi::BundleCollectiveRequestV1::empty();
    request.rank_count = 1;
    request.semantic_contract = byte_view(id);
    request
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Calls {
    create: usize,
    execute: usize,
    commit: usize,
    cancel: usize,
    destroy: usize,
    active: bool,
    fail_commit: bool,
    fail_cancel: bool,
    fail_destroy: bool,
}

impl Calls {
    const fn new() -> Self {
        Self {
            create: 0,
            execute: 0,
            commit: 0,
            cancel: 0,
            destroy: 0,
            active: false,
            fail_commit: false,
            fail_cancel: false,
            fail_destroy: false,
        }
    }
}

fn reset_calls() {
    *CALLS.lock().expect("fixture call state is available") = Calls::new();
}

fn calls() -> Calls {
    CALLS
        .lock()
        .expect("fixture call state is available")
        .clone()
}

fn host_service_offsets() -> Vec<usize> {
    vec![
        offset_of!(abi::BundleHostServicesV1, struct_size),
        offset_of!(abi::BundleHostServicesV1, abi_version),
        offset_of!(abi::BundleHostServicesV1, user_data),
        offset_of!(abi::BundleHostServicesV1, log),
        offset_of!(abi::BundleHostServicesV1, select_context),
        offset_of!(abi::BundleHostServicesV1, release_context),
        offset_of!(abi::BundleHostServicesV1, select_memory_pool),
        offset_of!(abi::BundleHostServicesV1, set_memory_pool_peer_access),
        offset_of!(abi::BundleHostServicesV1, release_memory_pool),
        offset_of!(abi::BundleHostServicesV1, allocate_device_buffer),
        offset_of!(abi::BundleHostServicesV1, release_device_buffer),
        offset_of!(abi::BundleHostServicesV1, open_compiled_image),
        offset_of!(abi::BundleHostServicesV1, upload_rank_arena),
        offset_of!(abi::BundleHostServicesV1, release_rank_arena),
        offset_of!(abi::BundleHostServicesV1, resolve_compiled_image_view),
        offset_of!(abi::BundleHostServicesV1, release_compiled_image),
        offset_of!(abi::BundleHostServicesV1, open_authenticated_component),
        offset_of!(abi::BundleHostServicesV1, inspect_authenticated_component),
        offset_of!(abi::BundleHostServicesV1, read_authenticated_component),
        offset_of!(abi::BundleHostServicesV1, release_authenticated_component),
        offset_of!(abi::BundleHostServicesV1, acquire_peer_access),
        offset_of!(abi::BundleHostServicesV1, release_peer_access),
        offset_of!(abi::BundleHostServicesV1, create_stream),
        offset_of!(abi::BundleHostServicesV1, select_stage_stream),
        offset_of!(abi::BundleHostServicesV1, synchronize_stream),
        offset_of!(abi::BundleHostServicesV1, release_stream),
        offset_of!(abi::BundleHostServicesV1, create_event),
        offset_of!(abi::BundleHostServicesV1, record_event),
        offset_of!(abi::BundleHostServicesV1, wait_event),
        offset_of!(abi::BundleHostServicesV1, synchronize_event),
        offset_of!(abi::BundleHostServicesV1, release_event),
        offset_of!(abi::BundleHostServicesV1, copy_host_to_device),
        offset_of!(abi::BundleHostServicesV1, copy_device_to_host),
        offset_of!(abi::BundleHostServicesV1, copy_device_to_device),
        offset_of!(abi::BundleHostServicesV1, copy_peer_to_peer),
        offset_of!(abi::BundleHostServicesV1, zero_device_buffer),
        offset_of!(abi::BundleHostServicesV1, write_device_pointer_table),
        offset_of!(abi::BundleHostServicesV1, load_cuda_module),
        offset_of!(abi::BundleHostServicesV1, resolve_cuda_symbol),
        offset_of!(abi::BundleHostServicesV1, configure_cuda_symbol),
        offset_of!(abi::BundleHostServicesV1, launch_cuda_symbol),
        offset_of!(abi::BundleHostServicesV1, release_cuda_symbol),
        offset_of!(abi::BundleHostServicesV1, release_cuda_module),
        offset_of!(abi::BundleHostServicesV1, allocate_physical_kv),
        offset_of!(abi::BundleHostServicesV1, create_kv_request),
        offset_of!(abi::BundleHostServicesV1, prepare_kv),
        offset_of!(abi::BundleHostServicesV1, commit_kv),
        offset_of!(abi::BundleHostServicesV1, cancel_kv),
        offset_of!(abi::BundleHostServicesV1, resolve_kv_view),
        offset_of!(abi::BundleHostServicesV1, release_kv_request),
        offset_of!(abi::BundleHostServicesV1, release_physical_kv),
        offset_of!(abi::BundleHostServicesV1, allocate_state),
        offset_of!(abi::BundleHostServicesV1, resolve_state_view),
        offset_of!(abi::BundleHostServicesV1, release_state),
        offset_of!(abi::BundleHostServicesV1, create_collective),
        offset_of!(abi::BundleHostServicesV1, execute_collective),
        offset_of!(abi::BundleHostServicesV1, release_collective),
        offset_of!(abi::BundleHostServicesV1, run_custom_operation),
        offset_of!(abi::BundleHostServicesV1, upload_rank_arenas),
    ]
}

/// Asserts one probe line whose numbers are values rather than a layout.
fn assert_values(c_layouts: &BTreeMap<String, Layout>, name: &str, rust_values: &[u32]) {
    let c_layout = c_layouts
        .get(name)
        .unwrap_or_else(|| panic!("C value probe omitted {name}"));
    let mut c_values = vec![c_layout.size];
    c_values.extend(c_layout.offsets.iter().copied());
    let rust_values = rust_values
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    assert_eq!(c_values, rust_values, "{name} values differ");
}

fn assert_layout(
    c_layouts: &BTreeMap<String, Layout>,
    name: &str,
    rust_size: usize,
    rust_offsets: &[usize],
) {
    let c_layout = c_layouts
        .get(name)
        .unwrap_or_else(|| panic!("C layout probe omitted {name}"));
    assert_eq!(c_layout.size, rust_size, "{name} size differs");
    assert_eq!(c_layout.offsets, rust_offsets, "{name} offsets differ");
}

fn c_layouts() -> BTreeMap<String, Layout> {
    let temporary = TempDirectory::new("runtime-bundle-abi-layout");
    let source = temporary.path().join("layout.c");
    let executable = temporary.path().join("layout");
    fs::write(&source, c_layout_probe_source()).expect("C layout probe source is writable");

    let header_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../include");
    let compiler = Command::new("cc")
        .arg("-std=c11")
        .arg("-I")
        .arg(header_directory)
        .arg(&source)
        .arg("-o")
        .arg(&executable)
        .output()
        .expect("C layout compiler starts");
    assert!(
        compiler.status.success(),
        "C layout compiler failed: {}",
        String::from_utf8_lossy(&compiler.stderr)
    );

    let output = Command::new(&executable)
        .output()
        .expect("C layout probe starts");
    assert!(
        output.status.success(),
        "C layout probe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    parse_layouts(&String::from_utf8(output.stdout).expect("C layout output is UTF-8"))
}

fn c_layout_probe_source() -> &'static str {
    r#"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "chop_bundle_v1.h"

#define P(type, field) offsetof(type, field)

int main(void) {
  printf("byte-view %zu %zu %zu\n", sizeof(chop_bundle_byte_view_v1), P(chop_bundle_byte_view_v1, data), P(chop_bundle_byte_view_v1, length));
  printf("mut-byte-view %zu %zu %zu\n", sizeof(chop_bundle_mut_byte_view_v1), P(chop_bundle_mut_byte_view_v1, data), P(chop_bundle_mut_byte_view_v1, length));
  printf("handle %zu %zu\n", sizeof(chop_bundle_handle_v1), P(chop_bundle_handle_v1, raw));
  printf("device-buffer-view %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_device_buffer_view_v1), P(chop_bundle_device_buffer_view_v1, struct_size), P(chop_bundle_device_buffer_view_v1, reserved), P(chop_bundle_device_buffer_view_v1, buffer), P(chop_bundle_device_buffer_view_v1, offset), P(chop_bundle_device_buffer_view_v1, length));
  printf("compiled-image-view %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_compiled_image_view_v1), P(chop_bundle_compiled_image_view_v1, struct_size), P(chop_bundle_compiled_image_view_v1, rank), P(chop_bundle_compiled_image_view_v1, buffer), P(chop_bundle_compiled_image_view_v1, alignment));
  printf("authenticated-component-metadata %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_authenticated_component_metadata_v1), P(chop_bundle_authenticated_component_metadata_v1, struct_size), P(chop_bundle_authenticated_component_metadata_v1, reserved), P(chop_bundle_authenticated_component_metadata_v1, role_length), P(chop_bundle_authenticated_component_metadata_v1, byte_length), P(chop_bundle_authenticated_component_metadata_v1, sha256));
  printf("launch-config %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_launch_config_v1), P(chop_bundle_launch_config_v1, struct_size), P(chop_bundle_launch_config_v1, flags), P(chop_bundle_launch_config_v1, grid_x), P(chop_bundle_launch_config_v1, grid_y), P(chop_bundle_launch_config_v1, grid_z), P(chop_bundle_launch_config_v1, block_x), P(chop_bundle_launch_config_v1, block_y), P(chop_bundle_launch_config_v1, block_z), P(chop_bundle_launch_config_v1, shared_memory_bytes), P(chop_bundle_launch_config_v1, cluster_x), P(chop_bundle_launch_config_v1, cluster_y), P(chop_bundle_launch_config_v1, cluster_z));
  printf("launch-cluster-bound %u\n", CHOP_BUNDLE_LAUNCH_V1_MAX_CLUSTER_BLOCKS);
  printf("launch-flag-programmatic-stream-serialization %u\n", CHOP_BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION);
  printf("launch-flags-known %u\n", CHOP_BUNDLE_LAUNCH_FLAGS_V1_KNOWN);
  printf("cluster-residency-mps-client %u\n", CHOP_BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT);
  printf("cluster-residency-clusters-reported %u\n", CHOP_BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED);
  printf("cluster-residency-cooperative-launch %u\n", CHOP_BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH);
  printf("tensor-map %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_tensor_map_v1), P(chop_bundle_tensor_map_v1, struct_size), P(chop_bundle_tensor_map_v1, rank), P(chop_bundle_tensor_map_v1, element_type), P(chop_bundle_tensor_map_v1, swizzle), P(chop_bundle_tensor_map_v1, interleave), P(chop_bundle_tensor_map_v1, l2_promotion), P(chop_bundle_tensor_map_v1, oob_fill), P(chop_bundle_tensor_map_v1, reserved), P(chop_bundle_tensor_map_v1, buffer), P(chop_bundle_tensor_map_v1, global_dims), P(chop_bundle_tensor_map_v1, global_strides_bytes), P(chop_bundle_tensor_map_v1, box_dims), P(chop_bundle_tensor_map_v1, element_strides));
  printf("kernel-argument %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_kernel_argument_v1), P(chop_bundle_kernel_argument_v1, struct_size), P(chop_bundle_kernel_argument_v1, kind), P(chop_bundle_kernel_argument_v1, buffer), P(chop_bundle_kernel_argument_v1, scalar_u64), P(chop_bundle_kernel_argument_v1, scalar_u32), P(chop_bundle_kernel_argument_v1, scalar_i32), P(chop_bundle_kernel_argument_v1, opaque), P(chop_bundle_kernel_argument_v1, tensor_map), P(chop_bundle_kernel_argument_v1, pointer_array));
  printf("kernel-argument-size-gate %u\n", CHOP_BUNDLE_KERNEL_ARGUMENT_V1_MAX_BYTES);
  printf("device-pointer-array %zu %zu %zu %zu\n", sizeof(chop_bundle_device_pointer_array_v1), P(chop_bundle_device_pointer_array_v1, struct_size), P(chop_bundle_device_pointer_array_v1, slot_count), P(chop_bundle_device_pointer_array_v1, slots));
  printf("kernel-argument-kinds %u %u %u %u %u %u %u %u %u\n", CHOP_BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_BUFFER, CHOP_BUNDLE_KERNEL_ARGUMENT_V1_U64, CHOP_BUNDLE_KERNEL_ARGUMENT_V1_OPAQUE_BYTES, CHOP_BUNDLE_KERNEL_ARGUMENT_V1_U32, CHOP_BUNDLE_KERNEL_ARGUMENT_V1_NULL_DEVICE_POINTER, CHOP_BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP, CHOP_BUNDLE_KERNEL_ARGUMENT_V1_I32, CHOP_BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY, CHOP_BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS);
  printf("tensor-map-codes %u %u %u %u %u %u %u %u %u\n", CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_U8, CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_I32, CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_BF16, CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_F16, CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_F32, CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E4M3, CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E5M2, CHOP_BUNDLE_TENSOR_MAP_SWIZZLE_V1_NONE, CHOP_BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_128);
  printf("custom-operand %zu %zu %zu %zu\n", sizeof(chop_bundle_custom_operand_v1), P(chop_bundle_custom_operand_v1, struct_size), P(chop_bundle_custom_operand_v1, flags), P(chop_bundle_custom_operand_v1, buffer));
  printf("custom-launch %zu %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_custom_launch_v1), P(chop_bundle_custom_launch_v1, struct_size), P(chop_bundle_custom_launch_v1, address_space), P(chop_bundle_custom_launch_v1, operand_count), P(chop_bundle_custom_launch_v1, stream), P(chop_bundle_custom_launch_v1, operand_addresses), P(chop_bundle_custom_launch_v1, operand_lengths));
  printf("custom-operation-codes %u %u %u %u %u %u\n", CHOP_BUNDLE_CUSTOM_OPERAND_FLAG_V1_WRITABLE, CHOP_BUNDLE_CUSTOM_OPERAND_FLAGS_V1_KNOWN, CHOP_BUNDLE_CUSTOM_ADDRESS_SPACE_V1_DEVICE, CHOP_BUNDLE_CUSTOM_ADDRESS_SPACE_V1_HOST, CHOP_BUNDLE_CUSTOM_OPERATION_FLAG_V1_MAY_ENQUEUE, CHOP_BUNDLE_CUSTOM_OPERATION_FLAGS_V1_KNOWN);
  printf("custom-operation-contained-fault %u\n", (uint32_t)CHOP_BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT);
  printf("physical-kv-layout %zu %zu %zu %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_physical_kv_layout_v1), P(chop_bundle_physical_kv_layout_v1, struct_size), P(chop_bundle_physical_kv_layout_v1, flags), P(chop_bundle_physical_kv_layout_v1, rank_count), P(chop_bundle_physical_kv_layout_v1, page_bytes), P(chop_bundle_physical_kv_layout_v1, page_alignment), P(chop_bundle_physical_kv_layout_v1, page_tokens), P(chop_bundle_physical_kv_layout_v1, state_lifetime), P(chop_bundle_physical_kv_layout_v1, layout_id));
  printf("physical-kv-request %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_physical_kv_request_v1), P(chop_bundle_physical_kv_request_v1, struct_size), P(chop_bundle_physical_kv_request_v1, flags), P(chop_bundle_physical_kv_request_v1, layout), P(chop_bundle_physical_kv_request_v1, page_capacity), P(chop_bundle_physical_kv_request_v1, request_capacity));
  printf("state-request %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_state_request_v1), P(chop_bundle_state_request_v1, struct_size), P(chop_bundle_state_request_v1, flags), P(chop_bundle_state_request_v1, byte_length), P(chop_bundle_state_request_v1, alignment), P(chop_bundle_state_request_v1, state_id));
  printf("collective-request %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_collective_request_v1), P(chop_bundle_collective_request_v1, struct_size), P(chop_bundle_collective_request_v1, flags), P(chop_bundle_collective_request_v1, rank_count), P(chop_bundle_collective_request_v1, semantic_contract));
  printf("create-result %zu %zu %zu %zu\n", sizeof(chop_bundle_create_result_v1), P(chop_bundle_create_result_v1, struct_size), P(chop_bundle_create_result_v1, status), P(chop_bundle_create_result_v1, handle));
  printf("logits-row %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_logits_row_v1), P(chop_bundle_logits_row_v1, struct_size), P(chop_bundle_logits_row_v1, reserved), P(chop_bundle_logits_row_v1, request_id), P(chop_bundle_logits_row_v1, logit_count), P(chop_bundle_logits_row_v1, logit_bits));
  printf("logits-view %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_logits_view_v1), P(chop_bundle_logits_view_v1, struct_size), P(chop_bundle_logits_view_v1, reserved), P(chop_bundle_logits_view_v1, context), P(chop_bundle_logits_view_v1, rank), P(chop_bundle_logits_view_v1, element_type), P(chop_bundle_logits_view_v1, buffer), P(chop_bundle_logits_view_v1, row_stride_elements), P(chop_bundle_logits_view_v1, row_count), P(chop_bundle_logits_view_v1, live_width), P(chop_bundle_logits_view_v1, first_item_index), P(chop_bundle_logits_view_v1, reserved_tail));
  printf("execute-result-sink %zu %zu %zu %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_execute_result_sink_v1), P(chop_bundle_execute_result_sink_v1, struct_size), P(chop_bundle_execute_result_sink_v1, reserved), P(chop_bundle_execute_result_sink_v1, user_data), P(chop_bundle_execute_result_sink_v1, maximum_rows), P(chop_bundle_execute_result_sink_v1, maximum_logits_per_row), P(chop_bundle_execute_result_sink_v1, maximum_total_logits), P(chop_bundle_execute_result_sink_v1, emit_logits_row), P(chop_bundle_execute_result_sink_v1, emit_logits_view));
  printf("host-services %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_host_services_v1), P(chop_bundle_host_services_v1, struct_size), P(chop_bundle_host_services_v1, abi_version), P(chop_bundle_host_services_v1, user_data), P(chop_bundle_host_services_v1, log), P(chop_bundle_host_services_v1, select_context), P(chop_bundle_host_services_v1, release_context), P(chop_bundle_host_services_v1, select_memory_pool), P(chop_bundle_host_services_v1, set_memory_pool_peer_access), P(chop_bundle_host_services_v1, release_memory_pool), P(chop_bundle_host_services_v1, allocate_device_buffer), P(chop_bundle_host_services_v1, release_device_buffer), P(chop_bundle_host_services_v1, open_compiled_image), P(chop_bundle_host_services_v1, upload_rank_arena), P(chop_bundle_host_services_v1, release_rank_arena), P(chop_bundle_host_services_v1, resolve_compiled_image_view), P(chop_bundle_host_services_v1, release_compiled_image), P(chop_bundle_host_services_v1, open_authenticated_component), P(chop_bundle_host_services_v1, inspect_authenticated_component), P(chop_bundle_host_services_v1, read_authenticated_component), P(chop_bundle_host_services_v1, release_authenticated_component), P(chop_bundle_host_services_v1, acquire_peer_access), P(chop_bundle_host_services_v1, release_peer_access), P(chop_bundle_host_services_v1, create_stream), P(chop_bundle_host_services_v1, select_stage_stream), P(chop_bundle_host_services_v1, synchronize_stream), P(chop_bundle_host_services_v1, release_stream), P(chop_bundle_host_services_v1, create_event), P(chop_bundle_host_services_v1, record_event), P(chop_bundle_host_services_v1, wait_event), P(chop_bundle_host_services_v1, synchronize_event), P(chop_bundle_host_services_v1, release_event), P(chop_bundle_host_services_v1, copy_host_to_device), P(chop_bundle_host_services_v1, copy_device_to_host), P(chop_bundle_host_services_v1, copy_device_to_device), P(chop_bundle_host_services_v1, copy_peer_to_peer), P(chop_bundle_host_services_v1, zero_device_buffer), P(chop_bundle_host_services_v1, write_device_pointer_table), P(chop_bundle_host_services_v1, load_cuda_module), P(chop_bundle_host_services_v1, resolve_cuda_symbol), P(chop_bundle_host_services_v1, configure_cuda_symbol), P(chop_bundle_host_services_v1, launch_cuda_symbol), P(chop_bundle_host_services_v1, release_cuda_symbol), P(chop_bundle_host_services_v1, release_cuda_module), P(chop_bundle_host_services_v1, allocate_physical_kv), P(chop_bundle_host_services_v1, create_kv_request), P(chop_bundle_host_services_v1, prepare_kv), P(chop_bundle_host_services_v1, commit_kv), P(chop_bundle_host_services_v1, cancel_kv), P(chop_bundle_host_services_v1, resolve_kv_view), P(chop_bundle_host_services_v1, release_kv_request), P(chop_bundle_host_services_v1, release_physical_kv), P(chop_bundle_host_services_v1, allocate_state), P(chop_bundle_host_services_v1, resolve_state_view), P(chop_bundle_host_services_v1, release_state), P(chop_bundle_host_services_v1, create_collective), P(chop_bundle_host_services_v1, execute_collective), P(chop_bundle_host_services_v1, release_collective), P(chop_bundle_host_services_v1, run_custom_operation), P(chop_bundle_host_services_v1, upload_rank_arenas));
  printf("api-retirement %zu %zu %zu\n", sizeof(chop_bundle_api_with_request_retirement_v1), P(chop_bundle_api_with_request_retirement_v1, base), P(chop_bundle_api_with_request_retirement_v1, retire_request_v1));
  printf("speculation %zu %zu %zu %zu\n", sizeof(chop_bundle_speculation_v1), P(chop_bundle_speculation_v1, struct_size), P(chop_bundle_speculation_v1, maximum_draft_tokens), P(chop_bundle_speculation_v1, maximum_verify_tokens));
  printf("draft-tokens %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_draft_tokens_v1), P(chop_bundle_draft_tokens_v1, struct_size), P(chop_bundle_draft_tokens_v1, reserved), P(chop_bundle_draft_tokens_v1, request_id), P(chop_bundle_draft_tokens_v1, token_count), P(chop_bundle_draft_tokens_v1, token_bytes));
  printf("kept-rows %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_kept_rows_v1), P(chop_bundle_kept_rows_v1, struct_size), P(chop_bundle_kept_rows_v1, reserved), P(chop_bundle_kept_rows_v1, request_id), P(chop_bundle_kept_rows_v1, kept_rows), P(chop_bundle_kept_rows_v1, reserved_tail));
  printf("api-speculation %zu %zu %zu\n", sizeof(chop_bundle_api_with_speculation_v1), P(chop_bundle_api_with_speculation_v1, base), P(chop_bundle_api_with_speculation_v1, commit_accepted_v1));
  printf("prefix-state-info %zu %zu %zu %zu\n", sizeof(chop_bundle_prefix_state_info_v1), P(chop_bundle_prefix_state_info_v1, struct_size), P(chop_bundle_prefix_state_info_v1, status), P(chop_bundle_prefix_state_info_v1, state_pages));
  printf("prefix-state %zu %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_prefix_state_v1), P(chop_bundle_prefix_state_v1, struct_size), P(chop_bundle_prefix_state_v1, reserved), P(chop_bundle_prefix_state_v1, request_id), P(chop_bundle_prefix_state_v1, tokens), P(chop_bundle_prefix_state_v1, token_ids), P(chop_bundle_prefix_state_v1, pages));
  printf("api-prefix-cache %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_api_with_prefix_cache_v1), P(chop_bundle_api_with_prefix_cache_v1, base), P(chop_bundle_api_with_prefix_cache_v1, describe_prefix_state_v1), P(chop_bundle_api_with_prefix_cache_v1, capture_prefix_state_v1), P(chop_bundle_api_with_prefix_cache_v1, restore_prefix_state_v1));
  printf("api-prefix-state-slot %zu %zu %zu\n", sizeof(chop_bundle_api_with_prefix_state_slot_v1), P(chop_bundle_api_with_prefix_state_slot_v1, base), P(chop_bundle_api_with_prefix_state_slot_v1, restore_prefix_state_in_slot_v1));
  printf("weight-plan-sink %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_weight_plan_sink_v1), P(chop_bundle_weight_plan_sink_v1, struct_size), P(chop_bundle_weight_plan_sink_v1, layout), P(chop_bundle_weight_plan_sink_v1, context), P(chop_bundle_weight_plan_sink_v1, emit), P(chop_bundle_weight_plan_sink_v1, emit_error));
  printf("weight-transform-call %zu %zu %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_weight_transform_call_v1), P(chop_bundle_weight_transform_call_v1, struct_size), P(chop_bundle_weight_transform_call_v1, transform), P(chop_bundle_weight_transform_call_v1, params), P(chop_bundle_weight_transform_call_v1, sources), P(chop_bundle_weight_transform_call_v1, source_count), P(chop_bundle_weight_transform_call_v1, output_offset), P(chop_bundle_weight_transform_call_v1, output));
  printf("frontend-sink %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_frontend_sink_v1), P(chop_bundle_frontend_sink_v1, struct_size), P(chop_bundle_frontend_sink_v1, layout), P(chop_bundle_frontend_sink_v1, context), P(chop_bundle_frontend_sink_v1, emit), P(chop_bundle_frontend_sink_v1, emit_error));
  printf("frontend-table %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_frontend_table_v1), P(chop_bundle_frontend_table_v1, struct_size), P(chop_bundle_frontend_table_v1, layout), P(chop_bundle_frontend_table_v1, open), P(chop_bundle_frontend_table_v1, close), P(chop_bundle_frontend_table_v1, render), P(chop_bundle_frontend_table_v1, constraint), P(chop_bundle_frontend_table_v1, parser_open), P(chop_bundle_frontend_table_v1, parser_push), P(chop_bundle_frontend_table_v1, parser_close));
  printf("frontend-codes %u %u %u\n", CHOP_FRONTEND_LAYOUT_V1, CHOP_FRONTEND_PUSH_FLAG_V1_FINAL, CHOP_FRONTEND_PUSH_FLAGS_V1_KNOWN);
  printf("weight-plan-codes %u\n", CHOP_WEIGHT_PLAN_LAYOUT_V1);
  printf("frontend-refusal-limit %llu\n", (unsigned long long)CHOP_FRONTEND_REFUSAL_BYTES_MAX_V1);
  printf("frontend-opened-limit %llu\n", (unsigned long long)CHOP_FRONTEND_OPENED_BYTES_MAX_V1);
  printf("sink-draft-tail %zu %zu\n", sizeof(chop_bundle_execute_result_sink_v1), P(chop_bundle_execute_result_sink_v1, emit_draft_tokens));
  printf("speculation-service-tail %zu %zu\n", sizeof(chop_bundle_host_services_v1), P(chop_bundle_host_services_v1, query_speculation_v1));
  printf("batch-limits %zu %zu %zu\n", sizeof(chop_bundle_batch_limits_v1), P(chop_bundle_batch_limits_v1, struct_size), P(chop_bundle_batch_limits_v1, maximum_scored_rows));
  printf("batch-limits-service-tail %zu %zu\n", sizeof(chop_bundle_host_services_v1), P(chop_bundle_host_services_v1, query_batch_limits_v1));
  printf("cluster-residency %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_cluster_residency_v1), P(chop_bundle_cluster_residency_v1, struct_size), P(chop_bundle_cluster_residency_v1, multiprocessor_count), P(chop_bundle_cluster_residency_v1, usable_multiprocessors), P(chop_bundle_cluster_residency_v1, max_active_clusters), P(chop_bundle_cluster_residency_v1, flags));
  printf("cluster-residency-service-tail %zu %zu\n", sizeof(chop_bundle_host_services_v1), P(chop_bundle_host_services_v1, query_cluster_residency_v1));
  printf("api-header %zu %zu %zu\n", sizeof(chop_bundle_api_header_v1), P(chop_bundle_api_header_v1, struct_size), P(chop_bundle_api_header_v1, abi_version));
  printf("api %zu %zu %zu %zu %zu %zu %zu %zu\n", sizeof(chop_bundle_api_v1), P(chop_bundle_api_v1, struct_size), P(chop_bundle_api_v1, abi_version), P(chop_bundle_api_v1, create), P(chop_bundle_api_v1, execute), P(chop_bundle_api_v1, commit), P(chop_bundle_api_v1, cancel), P(chop_bundle_api_v1, destroy));
  return 0;
}
"#
}

fn parse_layouts(output: &str) -> BTreeMap<String, Layout> {
    output
        .lines()
        .map(|line| {
            let mut fields = line.split_whitespace();
            let name = fields.next().expect("C layout line has a name").to_owned();
            let numbers = fields
                .map(|field| field.parse::<usize>().expect("C layout value is a number"))
                .collect::<Vec<_>>();
            let (size, offsets) = numbers
                .split_first()
                .expect("C layout line has a size and offsets");
            (
                name,
                Layout {
                    size: *size,
                    offsets: offsets.to_vec(),
                },
            )
        })
        .collect()
}

#[derive(Debug)]
struct Layout {
    size: usize,
    offsets: Vec<usize>,
}

struct TempDirectory {
    path: PathBuf,
}

impl TempDirectory {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!("{label}-{}", std::process::id()));
        fs::create_dir(&path).expect("temporary ABI directory is creatable");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// The encoding the frozen deep-gemm capture built over a 32 KiB span.
///
/// Nothing about it names a model. It is here because it is a real encoding a
/// bundle asks for, so the checks below refuse variations of something that
/// has to keep working rather than of something invented.
/// A four-slot address array: three named spans, then one null.
fn valid_device_pointer_array() -> abi::BundleDevicePointerArrayV1 {
    let mut array = abi::BundleDevicePointerArrayV1::empty();
    array.slot_count = 4;
    for (index, slot) in array.slots[..3].iter_mut().enumerate() {
        slot.buffer = abi::BundleDeviceBufferHandleV1::from_parts(
            1,
            u32::try_from(index).expect("a slot index fits u32") + 1,
        );
        slot.offset = 0;
        slot.length = 16;
    }
    array
}

fn valid_tensor_map() -> abi::BundleTensorMapV1 {
    let mut map = abi::BundleTensorMapV1::empty();
    map.rank = 2;
    map.element_type = abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_U8;
    map.swizzle = abi::BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_128;
    map.box_dims[0] = 128;
    map.box_dims[1] = 64;
    map.buffer = valid_device_buffer_view();
    map.buffer.length = 32_768;
    map.global_dims[0] = 1_024;
    map.global_dims[1] = 32;
    map.global_strides_bytes[0] = 1_024;
    map.element_strides[0] = 1;
    map.element_strides[1] = 1;
    map
}

#[test]
fn the_tensor_map_encoding_is_closed_and_every_extent_is_checked() {
    use runtime_bundle_abi::validate_tensor_map_v1;

    assert!(validate_tensor_map_v1(valid_tensor_map()).is_ok());

    // The element type and the swizzle are closed sets, so one code outside
    // either is refused rather than read as some default.
    for element_type in [0, 12, u32::MAX] {
        let mut map = valid_tensor_map();
        map.element_type = element_type;
        assert!(
            matches!(
                validate_tensor_map_v1(map),
                Err(BundleAbiError::InvalidTensorMapElement(actual))
                    if actual == element_type
            ),
            "element type {element_type} must be refused"
        );
    }
    for swizzle in [4, u32::MAX] {
        let mut map = valid_tensor_map();
        map.swizzle = swizzle;
        assert!(
            matches!(
                validate_tensor_map_v1(map),
                Err(BundleAbiError::InvalidTensorMapSwizzle(actual)) if actual == swizzle
            ),
            "swizzle {swizzle} must be refused"
        );
    }

    // Each geometry rule is checked on its own, so one broken extent is
    // reported as the extent it is rather than as some later consequence.
    /// One geometry rule, and the edit that breaks exactly that rule.
    type GeometryCase = (&'static str, fn(&mut abi::BundleTensorMapV1));

    let cases: &[GeometryCase] = &[
        ("global dimension", |map| map.global_dims[0] = 0),
        ("global dimension", |map| map.global_dims[1] = 0),
        ("box dimension", |map| map.box_dims[0] = 0),
        ("box dimension", |map| map.box_dims[1] = 0),
        ("box dimension", |map| map.box_dims[1] = 257),
        // A 128-byte swizzle tiles 128 bytes, so a wider box is not this map.
        ("box width", |map| map.box_dims[0] = 129),
        ("stride", |map| map.global_strides_bytes[0] = 0),
        ("stride", |map| map.global_strides_bytes[0] = 1_020),
        // One row of elements has to fit inside the stride it is laid out on.
        ("row width", |map| map.global_dims[0] = 1_040),
        ("row width", |map| {
            map.element_type = abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_I32;
            map.box_dims[0] = 32;
        }),
        // A global dimension above CUDA's maximum is refused before arithmetic.
        ("global dimension", |map| {
            map.element_type = abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_I32;
            map.box_dims[0] = 32;
            map.global_dims[0] = u64::MAX / 2;
        }),
        // Nor must the footprint, and the footprint has to fit the span.
        ("footprint", |map| map.global_dims[1] = 64),
        ("stride", |map| {
            map.global_strides_bytes[0] = 1 << 63;
            map.global_dims[1] = 4;
        }),
    ];
    for (surface, mutate) in cases {
        let mut map = valid_tensor_map();
        mutate(&mut map);
        assert!(
            matches!(
                validate_tensor_map_v1(map),
                Err(BundleAbiError::InvalidTensorMapGeometry(actual)) if actual == *surface
            ),
            "a broken {surface} must be refused, got {:?}",
            validate_tensor_map_v1(map)
        );
    }

    // A tiled map starts on a 16-byte boundary, and the offset inside the span
    // is the part of that address the ABI can see.
    let mut misaligned = valid_tensor_map();
    misaligned.buffer.offset = 8;
    misaligned.buffer.length = 32_768;
    assert!(matches!(
        validate_tensor_map_v1(misaligned),
        Err(BundleAbiError::InvalidTensorMapGeometry("base alignment"))
    ));

    // The span it is encoded over is checked as any other span is.
    let mut bad_span = valid_tensor_map();
    bad_span.buffer.reserved = 1;
    assert!(matches!(
        validate_tensor_map_v1(bad_span),
        Err(BundleAbiError::InvalidReserved { .. })
    ));
}

/// Every element code V1 names, and the width it is read at.
///
/// A width is what every geometry rule is computed from, so an element code
/// whose width the ABI did not know would let a map past checks that mean
/// nothing. The set is closed and total: each code has exactly one width here,
/// and a code outside the set has none.
#[test]
fn every_tensor_map_element_code_names_exactly_one_width() {
    use runtime_bundle_abi::tensor_map_element_bytes_v1;

    for (code, bytes) in [
        (abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_U8, 1),
        (abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_I32, 4),
        (abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_BF16, 2),
        (abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_F16, 2),
        (abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_F32, 4),
        (abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E4M3, 1),
        (abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E5M2, 1),
        (abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_U16, 2),
        (abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN8B, 0),
        (abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN16B, 1),
        (abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_16U6_ALIGN16B, 1),
    ] {
        assert_eq!(
            tensor_map_element_bytes_v1(code),
            Some(bytes),
            "element code {code} must be read at {bytes} bytes"
        );
    }
    for outside in [0, 12, u32::MAX] {
        assert_eq!(tensor_map_element_bytes_v1(outside), None);
    }

    // Each of the seven codes is a descriptor the ABI itself accepts. Whether a
    // host can encode one is the host's answer, not this layer's: the ABI's
    // work is to carry the element a bundle meant rather than one it did not.
    for code in [
        abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_F16,
        abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_F32,
        abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E4M3,
        abi::BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E5M2,
    ] {
        let mut map = valid_tensor_map();
        map.element_type = code;
        // A wider element makes the row wider, so the box and the row stay
        // inside the stride the fixture map declares.
        map.box_dims[0] = 16;
        map.global_dims[0] = 128;
        assert!(
            runtime_bundle_abi::validate_tensor_map_v1(map).is_ok(),
            "element code {code} must be a descriptor the ABI carries"
        );
    }
}

/// Every by-value nested record refuses a tail, in each parent that holds one.
///
/// A nested record sits at a fixed offset inside its parent, and every parent
/// field after it is at an offset computed from its length. A caller whose
/// nested record is longer has moved all of them, and the parent's own range
/// rule would accept that, because the parent is longer too. So each nested
/// record is checked at its exact length, and checked before any later parent
/// field is read.
///
/// Each case below breaks a later parent field as well, in a way that has its
/// own distinct error. Getting the nested length error rather than that one is
/// what says the nested check ran first, and that nothing downstream had already
/// used a field the shifted layout would have moved.
#[test]
fn a_nested_record_with_a_tail_is_refused_in_every_parent_before_its_later_fields() {
    use runtime_bundle_abi::validate_tensor_map_v1;

    let view_exact =
        u32::try_from(size_of::<abi::BundleDeviceBufferViewV1>()).expect("a V1 record fits in u32");
    let grown_view = || {
        let mut view = valid_device_buffer_view();
        view.struct_size = view_exact + 8;
        view
    };
    let is_grown_view = |error: Result<(), BundleAbiError>| {
        matches!(
            error,
            Err(BundleAbiError::StructSizeExact { surface, expected, actual })
                if surface == "device buffer view"
                    && expected == view_exact
                    && actual == view_exact + 8
        )
    };

    // compiled_image_view.buffer, with `alignment` after it. A zero alignment
    // would be InvalidAlignment if the nested check had not run first.
    let mut compiled_view = abi::BundleRawCompiledImageViewV1::empty();
    compiled_view.buffer = grown_view();
    compiled_view.alignment = 0;
    assert!(
        is_grown_view(validate_compiled_image_view_v1(compiled_view)),
        "a grown view inside a compiled image view must be refused first"
    );

    // tensor_map_2d.buffer, with every extent after it. A zero global dimension
    // would be InvalidTensorMapGeometry if the nested check had not run first.
    let mut map = valid_tensor_map();
    map.buffer = grown_view();
    map.buffer.length = 32_768;
    map.global_dims[0] = 0;
    assert!(
        is_grown_view(validate_tensor_map_v1(map)),
        "a grown view inside a tensor map must be refused first"
    );

    // kernel_argument.buffer, with the payload discipline after it. An
    // unselected opaque payload would be UnselectedKernelArgumentPayload.
    let mut argument = abi::BundleKernelArgumentV1::empty();
    argument.kind = abi::BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_BUFFER;
    argument.buffer = grown_view();
    argument.opaque = byte_view(&FIXTURE_LOGIT_BITS);
    assert!(
        is_grown_view(validate_kernel_argument_v1(argument)),
        "a grown view inside a kernel argument must be refused first"
    );

    // custom_operand.buffer, the last field of its parent.
    let operand = abi::BundleCustomOperandV1 {
        struct_size: u32::try_from(size_of::<abi::BundleCustomOperandV1>())
            .expect("custom operand size fits u32"),
        flags: abi::BUNDLE_CUSTOM_OPERAND_FLAG_V1_WRITABLE,
        buffer: grown_view(),
    };
    assert!(
        is_grown_view(validate_custom_operand_v1(operand)),
        "a grown view inside a custom operand must be refused"
    );

    // device_pointer_array.slots[], where a grown slot moves every slot after
    // it. Slot 1 is a null handle carrying an offset, which would otherwise be
    // InvalidDevicePointerArraySlots, so reaching it at all would show here.
    let mut array = valid_device_pointer_array();
    array.slots[0] = grown_view();
    array.slots[1] = abi::BundleDeviceBufferViewV1::empty();
    array.slots[1].offset = 8;
    assert!(
        is_grown_view(validate_device_pointer_array_v1(array)),
        "a grown slot must be refused before the slots after it are read"
    );
    // A null slot is held to the same length, which is the other slot shape.
    let mut null_slot = valid_device_pointer_array();
    null_slot.slots[3].struct_size = view_exact + 8;
    assert!(
        is_grown_view(validate_device_pointer_array_v1(null_slot)),
        "a grown null slot must be refused at the same exact length"
    );

    // physical_kv_request.layout, with page_capacity and request_capacity after
    // it. A capacity that overflows the page product would be InvalidRange.
    let layout_exact =
        u32::try_from(size_of::<abi::BundlePhysicalKvLayoutV1>()).expect("a V1 record fits in u32");
    let id = [1_u8];
    let mut layout = valid_physical_kv_layout(&id);
    layout.struct_size = layout_exact + 8;
    let mut request = valid_physical_kv_request(layout);
    request.page_capacity = u64::MAX;
    assert!(
        matches!(
            validate_physical_kv_request_v1(request),
            Err(BundleAbiError::StructSizeExact { surface, expected, actual })
                if surface == "physical KV layout"
                    && expected == layout_exact
                    && actual == layout_exact + 8
        ),
        "a grown layout inside a KV request must be refused before its capacities"
    );

    // Both nested records refuse a shorter length as well, and both parents
    // still accept a tail of their own: they are class 3.
    let mut short_view = valid_device_buffer_view();
    short_view.struct_size = view_exact - 1;
    assert!(matches!(
        validate_device_buffer_view_v1(short_view),
        Err(BundleAbiError::StructSizeExact { .. })
    ));
    let mut short_layout = valid_physical_kv_layout(&id);
    short_layout.struct_size = layout_exact - 1;
    assert!(matches!(
        validate_physical_kv_layout_v1(short_layout),
        Err(BundleAbiError::StructSizeExact { .. })
    ));
    let mut longer_parent = valid_physical_kv_request(valid_physical_kv_layout(&id));
    longer_parent.struct_size += 8;
    assert!(
        validate_physical_kv_request_v1(longer_parent).is_ok(),
        "the parent keeps its own forward tail"
    );
}

/// The V1 records whose declared length is exact rather than a range.
///
/// The two array elements are reached by advancing from the element before
/// them, and the two by-value nested records are reached at a fixed offset
/// inside a parent whose later fields move with them. Neither kind admits a
/// tail, in either direction.
///
/// These four are all of them. Every other pointer this ABI pairs with a count
/// is an array of a fixed-width scalar, and every other by-value member carries
/// no struct_size at all, so neither declares a length to disagree about.
#[test]
fn both_array_element_records_declare_the_exact_v1_length_and_no_other() {
    let kernel_argument_exact =
        u32::try_from(size_of::<abi::BundleKernelArgumentV1>()).expect("a V1 record fits in u32");
    let mut scalar = abi::BundleKernelArgumentV1::empty();
    scalar.kind = abi::BUNDLE_KERNEL_ARGUMENT_V1_U32;
    assert_eq!(scalar.struct_size, kernel_argument_exact);
    assert!(validate_kernel_argument_v1(scalar).is_ok());

    let operand_exact =
        u32::try_from(size_of::<abi::BundleCustomOperandV1>()).expect("a V1 record fits in u32");
    let operand = abi::BundleCustomOperandV1 {
        struct_size: operand_exact,
        flags: abi::BUNDLE_CUSTOM_OPERAND_FLAG_V1_WRITABLE,
        buffer: valid_device_buffer_view(),
    };
    assert!(validate_custom_operand_v1(operand).is_ok());

    for offset in [-1_i64, 1, 8] {
        let kernel_declared = u32::try_from(i64::from(kernel_argument_exact) + offset)
            .expect("the probed length fits u32");
        let mut wrong = scalar;
        wrong.struct_size = kernel_declared;
        assert!(
            matches!(
                validate_kernel_argument_v1(wrong),
                Err(BundleAbiError::StructSizeExact { surface, expected, actual })
                    if surface == "kernel argument"
                        && expected == kernel_argument_exact
                        && actual == kernel_declared
            ),
            "a kernel argument declaring {kernel_declared} must be refused"
        );

        let operand_declared =
            u32::try_from(i64::from(operand_exact) + offset).expect("the probed length fits u32");
        let mut wrong = operand;
        wrong.struct_size = operand_declared;
        assert!(
            matches!(
                validate_custom_operand_v1(wrong),
                Err(BundleAbiError::StructSizeExact { surface, expected, actual })
                    if surface == "custom operand"
                        && expected == operand_exact
                        && actual == operand_declared
            ),
            "a custom operand declaring {operand_declared} must be refused"
        );
    }

    // Each rule refuses on its own too, which is what the array boundary calls
    // before it advances by a single record.
    for declared in [0, u32::MAX] {
        assert!(matches!(
            runtime_bundle_abi::validate_kernel_argument_struct_size_v1(declared),
            Err(BundleAbiError::StructSizeExact { .. })
        ));
        assert!(matches!(
            runtime_bundle_abi::validate_custom_operand_struct_size_v1(declared),
            Err(BundleAbiError::StructSizeExact { .. })
        ));
    }
    assert!(
        runtime_bundle_abi::validate_kernel_argument_struct_size_v1(kernel_argument_exact).is_ok()
    );
    assert!(runtime_bundle_abi::validate_custom_operand_struct_size_v1(operand_exact).is_ok());

    // The size gate is a ceiling on the kernel argument's layout, and it holds.
    assert!(kernel_argument_exact <= abi::BUNDLE_KERNEL_ARGUMENT_V1_MAX_BYTES);
}

/// One kernel argument is a compact record that names its large payload.
///
/// The two large payloads are reached through a pointer the caller keeps valid
/// for the length of the call, so the record stays small enough to pass an
/// argument list by value. That makes the discriminant load bearing: the
/// payload the kind selects has to be there, and the one it does not select has
/// to be absent, or the record says two things at once.
#[test]
fn a_kernel_argument_is_a_compact_record_that_names_its_payload() {
    assert!(
        size_of::<abi::BundleKernelArgumentV1>()
            <= abi::BUNDLE_KERNEL_ARGUMENT_V1_MAX_BYTES as usize,
        "a V1 kernel argument must stay inside its declared size gate"
    );

    let map = valid_tensor_map();
    let array = valid_device_pointer_array();

    // The payload a kind selects must be present.
    let mut missing_map = abi::BundleKernelArgumentV1::empty();
    missing_map.kind = abi::BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP;
    assert!(matches!(
        validate_kernel_argument_v1(missing_map),
        Err(BundleAbiError::MissingKernelArgumentPayload("tensor map"))
    ));
    let mut missing_array = abi::BundleKernelArgumentV1::empty();
    missing_array.kind = abi::BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY;
    assert!(matches!(
        validate_kernel_argument_v1(missing_array),
        Err(BundleAbiError::MissingKernelArgumentPayload(
            "device pointer array"
        ))
    ));

    // A payload the kind does not select must be absent, whichever of the three
    // it is and whichever kind carries it.
    type PayloadCase = (
        &'static str,
        fn(
            &mut abi::BundleKernelArgumentV1,
            &abi::BundleTensorMapV1,
            &abi::BundleDevicePointerArrayV1,
        ),
    );
    let payloads: &[PayloadCase] = &[
        ("tensor map", |argument, map, _| {
            argument.tensor_map = &raw const *map;
        }),
        ("device pointer array", |argument, _, array| {
            argument.pointer_array = &raw const *array;
        }),
        ("opaque bytes", |argument, _, _| {
            argument.opaque = byte_view(&FIXTURE_LOGIT_BITS);
        }),
    ];
    for kind in [
        abi::BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_BUFFER,
        abi::BUNDLE_KERNEL_ARGUMENT_V1_U64,
        abi::BUNDLE_KERNEL_ARGUMENT_V1_U32,
        abi::BUNDLE_KERNEL_ARGUMENT_V1_I32,
        abi::BUNDLE_KERNEL_ARGUMENT_V1_NULL_DEVICE_POINTER,
        abi::BUNDLE_KERNEL_ARGUMENT_V1_OPAQUE_BYTES,
        abi::BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP,
        abi::BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY,
    ] {
        for (surface, attach) in payloads {
            let mut argument = abi::BundleKernelArgumentV1::empty();
            argument.kind = kind;
            argument.buffer = valid_device_buffer_view();
            match kind {
                abi::BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP => {
                    argument.tensor_map = &raw const map;
                }
                abi::BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY => {
                    argument.pointer_array = &raw const array;
                }
                abi::BUNDLE_KERNEL_ARGUMENT_V1_OPAQUE_BYTES => {
                    argument.opaque = byte_view(&FIXTURE_LOGIT_BITS);
                }
                _ => {}
            }
            let selected = validate_kernel_argument_v1(argument).is_ok();
            assert!(selected, "kind {kind} with only its own payload is valid");
            attach(&mut argument, &map, &array);
            let carries_its_own = matches!(
                (kind, *surface),
                (abi::BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP, "tensor map")
                    | (
                        abi::BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY,
                        "device pointer array"
                    )
                    | (abi::BUNDLE_KERNEL_ARGUMENT_V1_OPAQUE_BYTES, "opaque bytes")
            );
            if carries_its_own {
                assert!(validate_kernel_argument_v1(argument).is_ok());
                continue;
            }
            assert!(
                matches!(
                    validate_kernel_argument_v1(argument),
                    Err(BundleAbiError::UnselectedKernelArgumentPayload(actual))
                        if actual == *surface
                ),
                "kind {kind} must refuse a {surface} payload it does not select"
            );
        }
    }
}

/// What a launch may ask for as a thread-block cluster, and what it may not.
///
/// The default is one block, which is the launch every kernel without a cluster
/// takes. Anything else is an exact shape: every extent is nonzero, the blocks
/// stay inside the bound V1 declares, and the grid divides into whole clusters.
/// A shape that fails one of those is refused rather than rounded to one that
/// would have launched.
#[test]
fn a_launch_cluster_is_the_default_or_a_shape_the_grid_admits() {
    let mut config = valid_launch_config();
    config.grid_x = 8;
    config.grid_y = 4;
    config.grid_z = 2;
    assert!(
        validate_launch_config_v1(config).is_ok(),
        "the one-block default is every launch that asks for no cluster"
    );

    let mut clustered = config;
    clustered.cluster_x = 2;
    clustered.cluster_y = 2;
    clustered.cluster_z = 2;
    assert!(validate_launch_config_v1(clustered).is_ok());

    for zero in 0..3 {
        let mut absent = config;
        match zero {
            0 => absent.cluster_x = 0,
            1 => absent.cluster_y = 0,
            _ => absent.cluster_z = 0,
        }
        assert!(
            matches!(
                validate_launch_config_v1(absent),
                Err(BundleAbiError::InvalidLaunchCluster("cluster dimension"))
            ),
            "cluster extent {zero} must be at least one block"
        );
    }

    // The bound is on the blocks one cluster holds, so a shape that is inside
    // every extent but past the product is still refused.
    let mut oversized = config;
    oversized.grid_x = 16;
    oversized.cluster_x = 4;
    oversized.cluster_y = 4;
    assert!(matches!(
        validate_launch_config_v1(oversized),
        Err(BundleAbiError::InvalidLaunchCluster("cluster blocks"))
    ));

    // A grid that does not divide into whole clusters is not the launch it
    // describes, and V1 has no way to flatten it into one that is.
    let mut ragged = config;
    ragged.grid_y = 3;
    ragged.cluster_y = 2;
    assert!(matches!(
        validate_launch_config_v1(ragged),
        Err(BundleAbiError::InvalidLaunchCluster("cluster grid"))
    ));
}

#[test]
fn every_argument_encoding_round_trips_through_the_raw_record() {
    use runtime_bundle_abi::{TensorMapElementV1, TensorMapSwizzleV1, TensorMapV1};

    let backend = Rc::new(FixtureHostBackend::default());
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let stream = abi::BundleStreamHandleV1::from_parts(1, 1);

    let arguments = [
        KernelArgumentV1::DeviceBuffer(fixture_device_span()),
        KernelArgumentV1::NullDevicePointer,
        // A 32-bit scalar keeps its width, and a 64-bit one keeps a value no
        // 32-bit slot could have carried.
        KernelArgumentV1::ScalarU32(u32::MAX),
        KernelArgumentV1::ScalarU64(u64::from(u32::MAX) + 1),
        KernelArgumentV1::OpaqueBytes(vec![3_u8, 1, 4]),
        KernelArgumentV1::TensorMap(TensorMapV1 {
            buffer: DeviceBufferSpanV1::new(
                abi::BundleDeviceBufferHandleV1::from_parts(1, 2),
                0,
                65_536,
            ),
            rank: 2,
            element_type: TensorMapElementV1::I32,
            swizzle: TensorMapSwizzleV1::None,
            interleave: 0,
            l2_promotion: 3,
            oob_fill: 0,
            global_dims: [8_192, 2, 0, 0, 0],
            global_strides_bytes: [32_768, 0, 0, 0],
            box_dims: [64, 1, 0, 0, 0],
            element_strides: [1, 1, 0, 0, 0],
        }),
    ];
    services
        .launch_cuda_symbol(
            stream,
            abi::BundleCudaSymbolHandleV1::from_parts(1, 1),
            fixture_launch_config(),
            &arguments,
        )
        .expect("every declared encoding crosses the boundary");
    let launches = backend.launches.borrow();
    let (_, observed) = launches.first().expect("one launch is recorded");
    assert_eq!(observed.as_slice(), arguments);

    // Element widths are the ABI's, not the caller's opinion of them.
    assert_eq!(TensorMapElementV1::U8.byte_width(), 1);
    assert_eq!(TensorMapElementV1::I32.byte_width(), 4);
    assert_eq!(TensorMapElementV1::Bf16.byte_width(), 2);
}

fn retirement_api(
    callback: Option<abi::BundleRetireRequestFnV1>,
) -> abi::BundleApiWithRequestRetirementV1 {
    let mut base = fixture_raw_api();
    base.struct_size = size_of::<abi::BundleApiWithRequestRetirementV1>() as u32;
    abi::BundleApiWithRequestRetirementV1 {
        base,
        retire_request_v1: callback,
    }
}

static RETIREMENTS: std::sync::Mutex<Vec<(abi::BundleHandleV1, u64)>> =
    std::sync::Mutex::new(Vec::new());
extern "C" fn retire_fixture(handle: abi::BundleHandleV1, request_id: u64) -> abi::BundleStatusV1 {
    RETIREMENTS.lock().unwrap().push((handle, request_id));
    if request_id == u64::MAX {
        abi::BundleStatusV1::BundleError
    } else {
        abi::BundleStatusV1::Success
    }
}

#[test]
fn retirement_constructor_requires_explicit_complete_non_null_extension() {
    let extended = retirement_api(Some(retire_fixture));
    assert!(
        BundleApiV1::new_with_request_retirement(extended)
            .unwrap()
            .supports_request_retirement()
    );
    assert!(
        !BundleApiV1::new(extended.base)
            .unwrap()
            .supports_request_retirement()
    );
    assert!(!fixture_api().supports_request_retirement());
    assert!(
        !BundleApiV1::new_with_request_retirement(retirement_api(None))
            .unwrap()
            .supports_request_retirement()
    );
    let mut incomplete = extended;
    incomplete.base.struct_size -= 1;
    assert!(
        !BundleApiV1::new_with_request_retirement(incomplete)
            .unwrap()
            .supports_request_retirement()
    );
    assert_eq!(offset_of!(abi::BundleApiWithRequestRetirementV1, base), 0);
    assert_eq!(
        offset_of!(abi::BundleApiWithRequestRetirementV1, retire_request_v1),
        size_of::<abi::BundleRawApiV1>()
    );
}

#[test]
fn retirement_is_ready_only_and_failure_is_terminal() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_calls();
    RETIREMENTS.lock().unwrap().clear();
    let api =
        BundleApiV1::new_with_request_retirement(retirement_api(Some(retire_fixture))).unwrap();
    let mut instance = api.create(fixture_table(), &[]).unwrap();
    let mut sink = fixture_sink();
    instance.retire_request(0).unwrap();
    for committed in [true, false] {
        instance.execute(&[1], &mut sink).unwrap();
        assert!(matches!(
            instance.retire_request(123),
            Err(BundleAbiError::Lifecycle("retire_request"))
        ));
        if committed {
            instance.commit().unwrap();
        } else {
            instance.cancel().unwrap();
        }
        instance.retire_request(1_u64 << 40).unwrap();
    }
    assert_eq!(
        *RETIREMENTS.lock().unwrap(),
        vec![(HANDLE, 0), (HANDLE, 1_u64 << 40), (HANDLE, 1_u64 << 40)]
    );
    assert!(matches!(
        instance.retire_request(u64::MAX),
        Err(BundleAbiError::Status(abi::BundleStatusV1::BundleError))
    ));
    assert!(instance.is_poisoned());
    assert!(matches!(
        instance.retire_request(0),
        Err(BundleAbiError::Poisoned)
    ));
    assert!(matches!(
        instance.execute(&[1], &mut sink),
        Err(BundleAbiError::Poisoned)
    ));
    assert!(matches!(instance.commit(), Err(BundleAbiError::Poisoned)));
    assert!(matches!(instance.cancel(), Err(BundleAbiError::Poisoned)));
    assert!(matches!(instance.destroy(), Err(BundleAbiError::Poisoned)));
    drop(instance);
    assert_eq!(RETIREMENTS.lock().unwrap().len(), 4);
    assert_eq!(calls().destroy, 0);
}

#[test]
fn legacy_retirement_is_noop_but_still_requires_ready() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_calls();
    let mut instance = fixture_api().create(fixture_table(), &[]).unwrap();
    assert!(!instance.supports_request_retirement());
    instance.retire_request(u64::MAX).unwrap();
    instance.execute(&[1], &mut fixture_sink()).unwrap();
    assert!(matches!(
        instance.retire_request(0),
        Err(BundleAbiError::Lifecycle("retire_request"))
    ));
    instance.cancel().unwrap();
    instance.retire_request(0).unwrap();
    instance.destroy().unwrap();
    assert!(matches!(
        instance.retire_request(0),
        Err(BundleAbiError::Lifecycle("retire_request"))
    ));
}

#[test]
fn only_a_refusal_before_host_work_or_results_is_recoverable() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_calls();
    let mut instance = fixture_api().create(fixture_table(), &[1]).unwrap();
    for request in [
        b"refuse-clean".as_slice(),
        b"refuse-after-service",
        b"refuse-after-row",
    ] {
        assert_eq!(
            instance.execute(request, &mut fixture_sink()),
            Err(BundleAbiError::Status(abi::BundleStatusV1::InvalidArgument))
        );
        if request == b"refuse-clean" {
            assert_eq!(
                instance.last_execute_refusal(),
                Some("packing check: item token budget exceeded")
            );
        } else {
            assert_eq!(instance.last_execute_refusal(), None);
        }
        assert!(!instance.is_poisoned());
    }
    instance.execute(&[1], &mut fixture_sink()).unwrap();
    assert_eq!(instance.last_execute_refusal(), None);
    instance.cancel().unwrap();
}

#[test]
fn refusal_diagnostics_are_bounded_and_never_reused_from_an_earlier_call() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_calls();
    let mut instance = fixture_api().create(fixture_table(), &[1]).unwrap();
    assert!(
        instance
            .execute(b"refuse-long", &mut fixture_sink())
            .is_err()
    );
    let message = instance.last_execute_refusal().unwrap();
    assert!(message.starts_with("check:"));
    assert!(message.len() <= 512);
    assert!(!message.contains('\n'));
    assert!(
        instance
            .execute(b"refuse-silent", &mut fixture_sink())
            .is_err()
    );
    assert!(
        instance
            .last_execute_refusal()
            .unwrap()
            .contains("no check diagnostic")
    );
    instance.execute(&[1], &mut fixture_sink()).unwrap();
    CALLS.lock().unwrap().fail_cancel = true;
    assert!(instance.cancel().is_err());
    assert_eq!(
        instance.execute(b"refuse-clean", &mut fixture_sink()),
        Err(BundleAbiError::Poisoned)
    );
    assert_eq!(instance.last_execute_refusal(), None);
}
